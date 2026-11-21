# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

import unittest

from compare_parquet import REVISION, pair_schedule, validate


def report_for(mode, mixed_first=False):
    phases = ("mixed", "isolated") if mixed_first else ("isolated", "mixed")
    report = {
        "build_revision": REVISION,
        "valid": True,
        "runtime": {"morsel_mode": mode},
        "runs": [{"mode": phase} for phase in phases],
    }
    for table in ("small", "large"):
        plan = (
            "AggregateExec: mode=Final\n"
            f"  DataSourceExec: file_groups={table}, file_type=parquet\n"
            if mode == "off"
            else f"MorselExec: workers=4, NativeParquet[{table}] -> CountSink\n"
        )
        report[table] = {"plan": plan}
    return report


class ComparisonTest(unittest.TestCase):
    def test_valid_reports(self):
        for mode in ("off", "required"):
            for mixed_first in (False, True):
                with self.subTest(mode=mode, mixed_first=mixed_first):
                    validate(report_for(mode, mixed_first), mode, mixed_first, 4)

    def test_rejects_null_or_wrong_build_revision(self):
        for revision in (None, "0" * 40):
            with self.subTest(revision=revision), self.assertRaisesRegex(
                ValueError, "expected a binary built against"
            ):
                report = dict(report_for("off"), build_revision=revision)
                validate(report, "off", False, 4)

    def test_rejects_invalid_report_and_wrong_runtime_mode(self):
        off = report_for("off")
        for report in (
            dict(off, valid=False),
            dict(off, runtime={"morsel_mode": "required"}),
        ):
            with self.subTest(report=report), self.assertRaisesRegex(
                ValueError, "invalid off report"
            ):
                validate(report, "off", False, 4)

    def test_rejects_wrong_phase_order(self):
        for mixed_first in (False, True):
            with self.subTest(mixed_first=mixed_first), self.assertRaisesRegex(
                ValueError, "expected phases"
            ):
                report = report_for("required", not mixed_first)
                validate(report, "required", mixed_first, 4)

    def test_rejects_nonstock_or_nonparquet_plan(self):
        report = report_for("off")
        stock = report["small"]["plan"]
        for plan in (
            f"MorselExec: worker evidence\n{stock}",
            stock.replace("DataSourceExec", "MemoryExec"),
            stock.replace("file_type=parquet", "file_type=csv"),
        ):
            with self.subTest(plan=plan), self.assertRaisesRegex(
                ValueError, "unexpected off/small plan"
            ):
                validate(dict(report, small={"plan": plan}), "off", False, 4)

    def test_rejects_native_fallback_wrong_sink_worker_budget_or_table(self):
        report = report_for("required")
        stock = report_for("off")
        for table in ("small", "large"):
            native = report[table]["plan"]
            for plan in (
                stock[table]["plan"],
                native.replace("CountSink", "ResultSink"),
                native.replace("workers=4", "workers=3"),
                native.replace(f"NativeParquet[{table}]", "NativeParquet[other]"),
            ):
                with self.subTest(table=table, plan=plan), self.assertRaisesRegex(
                    ValueError, f"unexpected required/{table} plan"
                ):
                    validate({**report, table: {"plan": plan}}, "required", False, 4)

    def test_pair_schedule_balances_backend_and_phase_order(self):
        self.assertEqual(
            [pair_schedule(pair) for pair in range(4)],
            [
                (("off", False), ("required", False)),
                (("required", True), ("off", True)),
                (("off", True), ("required", True)),
                (("required", False), ("off", False)),
            ],
        )


if __name__ == "__main__":
    unittest.main()
