# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

import os
import tempfile
import unittest
from pathlib import Path

import pyarrow.parquet as pq

from harness import StockCli, fixture_files, generate, verify, workloads


class SmokeTest(unittest.TestCase):
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
