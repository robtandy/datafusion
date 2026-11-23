# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

"""Unit tests for the TPC-H Q6 profiling harness utilities (#180)."""

from decimal import Decimal
import tempfile
from pathlib import Path
import unittest

import pyarrow as pa

import profile_harness as ph
import profile_perf as pp


class ProfileHarnessTest(unittest.TestCase):
    def test_summarize_morsel_runs(self):
        runs = [
            [
                {"name": "morsel_pending_io", "worker": None, "value": 12_500_000},
                {"name": "morsel_ready_wait", "worker": "0", "value": 2_000_000},
                {"name": "morsel_ready_wait", "worker": "1", "value": 3_000_000},
                {"name": "morsel_decode_work", "worker": "0", "value": 40_000_000},
                {"name": "morsel_decode_work", "worker": "1", "value": 45_000_000},
                {"name": "morsel_process_work", "worker": "0", "value": 5_000_000},
                {"name": "morsel_process_work", "worker": "1", "value": 6_000_000},
                {"name": "morsel_output_wait", "worker": "0", "value": 0},
                {"name": "morsels_claimed", "worker": "0", "value": 24},
                {"name": "morsels_claimed", "worker": "1", "value": 24},
                {"name": "morsels_completed", "worker": "0", "value": 24},
                {"name": "morsels_completed", "worker": "1", "value": 24},
                {"name": "morsel_peak_ready_depth", "worker": None, "value": 8},
                {"name": "morsel_peak_output_depth", "worker": None, "value": 4},
            ]
        ]
        summary = ph.summarize_morsel_runs(runs)
        self.assertEqual(summary["pending_io_ms"], 12.5)
        self.assertEqual(summary["ready_wait_ms"], 5.0)
        self.assertEqual(summary["decode_ms"], 85.0)
        self.assertEqual(summary["process_ms"], 11.0)
        self.assertEqual(summary["output_wait_ms"], 0.0)
        self.assertEqual(summary["morsels_claimed"], 48)
        self.assertEqual(summary["per_worker_claimed"], [24, 24])
        self.assertEqual(summary["peak_ready_depth"], 8)
        self.assertEqual(summary["peak_output_depth"], 4)

    def test_compare_tables_unordered_decimal(self):
        oracle = pa.table(
            {
                "l_orderkey": [2, 1],
                "rev": [Decimal("12.3400"), Decimal("56.7800")],
            }
        )
        actual = pa.table(
            {
                "l_orderkey": [1, 2],
                "rev": [Decimal("56.7800"), Decimal("12.3400")],
            }
        )
        ph.compare_tables(
            oracle, actual, ordered=False, case_name="q6_scan", arm_name="test"
        )

    def test_validate_plan_accepts_pushdown_native_parquet(self):
        sample = {
            "plan": (
                "MorselExec: workers=8, NativeParquet[lineitem, predicate=l_quantity@4 < 24.00]"
                " -> Projection[] -> CountSink"
            ),
            "parquet_morsels_claimed": 48,
            "count_finalizations": 1,
        }
        ph.validate_plan(
            sample, ph.ARMS["morsel_pushdown"], ph.WORKLOADS["q6_count"], 48
        )

    def test_parse_perf_stat(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "sample.stat"
            path.write_text(
                "123456,,cycles:u,1000,100.00,,\n"
                "234567,,instructions:u,1000,100.00,1.90,insn per cycle\n"
                "<not counted>,,cpu-migrations,0,100.00,,\n"
            )
            parsed = pp.parse_perf_stat(path)
            self.assertEqual(parsed["cycles:u"], 123456.0)
            self.assertEqual(parsed["instructions:u"], 234567.0)
            self.assertNotIn("cpu-migrations", parsed)


if __name__ == "__main__":
    unittest.main()
