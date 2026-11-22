# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq

from bench import (
    LAYOUTS,
    WORKERS,
    check_layout,
    fixture_files,
    generate,
    layout,
    validate_sample,
    warm_pages,
)


class FixtureTest(unittest.TestCase):
    def test_layout_and_contiguous_ids(self):
        for case in LAYOUTS:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / case
                groups, group_rows = layout(case, smoke=True)
                generate(path, groups, group_rows)
                files = fixture_files(path)
                check_layout(files, groups, group_rows)
                with self.assertRaisesRegex(ValueError, "expected .* row groups"):
                    check_layout(files, groups, group_rows + 1)
                ids = [
                    row
                    for file in files
                    for row in pq.read_table(file, columns=["row_id"])[0].to_pylist()
                ]
                self.assertEqual(ids, list(range(sum(groups) * group_rows)))

    def test_warm_cache_rechecks_and_rejects_partial_residency(self):
        partial = {"resident": 1, "total": 2}
        full = {"resident": 2, "total": 2}
        with patch("bench.cache_pages", side_effect=[partial, full]):
            self.assertEqual(warm_pages([]), full)
        with patch("bench.cache_pages", return_value=partial), self.assertRaisesRegex(
            ValueError, "cannot verify warm file page cache"
        ):
            warm_pages([])

    def test_invalid_native_claims_are_rejected(self):
        good = {
            "build_revision": "pinned",
            "plan": "MorselExec: workers=8, NativeParquet[fixture] -> ResultSink",
            "output_rows": 10,
            "parquet_morsels_claimed": 4,
            "count_finalizations": 0,
            "worker_claims": [1, 1, 1, 1, 0, 0, 0, 0],
            "worker_completions": [1, 1, 1, 1, 0, 0, 0, 0],
        }
        validate_sample(good, "required", "scan_low", 4, 10, "pinned")
        for bad in (
            dict(good, worker_claims=[4]),
            dict(good, worker_completions=[1, 1, 1, 0, 0, 0, 0, 0]),
            dict(good, plan="DataSourceExec: file_type=parquet"),
            dict(good, build_revision="unrelated"),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_sample(bad, "required", "scan_low", 4, 10, "pinned")


@unittest.skipUnless(
    sys.platform == "linux"
    and os.environ.get("MORSEL_PARQUET_RUNNER")
    and os.environ.get("MORSEL_BENCH_REVISION"),
    "set MORSEL_PARQUET_RUNNER and MORSEL_BENCH_REVISION for the Linux smoke test",
)
class RunnerTest(unittest.TestCase):
    def test_ab_both_layouts(self):
        script = [sys.executable, "-B", str(Path(__file__).with_name("bench.py"))]
        for case in LAYOUTS:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "fixture with spaces"
                subprocess.run(
                    [*script, "generate", case, str(path), "--smoke"],
                    check=True,
                    timeout=15,
                )
                result = subprocess.check_output(
                    [
                        *script,
                        "ab",
                        case,
                        str(path),
                        "--smoke",
                        "--runner",
                        os.environ["MORSEL_PARQUET_RUNNER"],
                        "--revision",
                        os.environ["MORSEL_BENCH_REVISION"],
                        "--timeout",
                        "15",
                        "--warmups",
                        "0",
                        "--repeats",
                        "2",
                    ],
                    timeout=90,
                )
                header, *trials = map(json.loads, result.splitlines())
                groups, group_rows = layout(case, smoke=True)
                self.assertEqual(header["input_rows"], 8192)
                self.assertEqual(header["row_groups_per_file"], groups)
                self.assertEqual(header["rows_per_group"], group_rows)
                self.assertEqual(len(header["evidence"]), 8)
                self.assertEqual(len(trials), 16)
                for first, second in zip(trials[::2], trials[1::2], strict=True):
                    self.assertEqual(first["query"], second["query"])
                    self.assertEqual(first["rep"], second["rep"])
                    stock, native = (
                        (first, second) if first["rep"] == 1 else (second, first)
                    )
                    self.assertEqual(
                        (stock["mode"], native["mode"]), ("off", "required")
                    )
                    self.assertIsNone(stock["worker_claims"])
                    claims = native["worker_claims"]
                    self.assertEqual(len(claims), WORKERS)
                    self.assertEqual(sum(claims), sum(groups))
                    self.assertEqual(claims, native["worker_completions"])
                    if case == "large-row-groups":
                        self.assertGreaterEqual(claims.count(0), 4)
                    for trial in (stock, native):
                        pages = trial["cache_pages_before"]
                        self.assertGreater(pages["resident"], 0)
                        self.assertEqual(pages["resident"], pages["total"])
                        self.assertGreater(trial["seconds"], 0)
                        query, selectivity = trial["query"].split("_")
                        expected = (
                            1
                            if query == "count"
                            else header["verified_matches"][selectivity]
                        )
                        self.assertEqual(trial["output_rows"], expected)
