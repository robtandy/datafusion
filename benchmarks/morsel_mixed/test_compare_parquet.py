# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

import unittest

from compare_parquet import REVISION, pair_schedule, validate


def report_for(mode, mixed_first=False, revision=REVISION, pushdown_filters=False):
    phases = ("mixed", "isolated") if mixed_first else ("isolated", "mixed")
    report = {
        "build_revision": revision,
        "valid": True,
        "runtime": {"morsel_mode": mode},
        "runs": [{"mode": phase} for phase in phases],
    }
    for table in ("small", "large"):
        if mode == "off":
            plan = (
                "AggregateExec: mode=Final\n"
                f"  DataSourceExec: file_groups={table}, file_type=parquet\n"
            )
        elif pushdown_filters:
            plan = (
                f"MorselExec: workers=4, NativeParquet[{table}, "
                "predicate=gate@0 < 10] -> CountSink\n"
            )
        else:
            plan = f"MorselExec: workers=4, NativeParquet[{table}] -> CountSink\n"
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

    def test_valid_reports_with_custom_revision_and_pushdown_filters(self):
        custom_rev = "9" * 40
        for mode in ("off", "required"):
            for mixed_first in (False, True):
                with self.subTest(mode=mode, mixed_first=mixed_first):
                    validate(
                        report_for(
                            mode,
                            mixed_first,
                            revision=custom_rev,
                            pushdown_filters=True,
                        ),
                        mode,
                        mixed_first,
                        4,
                        revision=custom_rev,
                        pushdown_filters=True,
                    )

    def test_rejects_mismatched_pushdown_plan_shape(self):
        plain = report_for("required", pushdown_filters=False)
        pushed = report_for("required", pushdown_filters=True)
        with self.assertRaisesRegex(ValueError, "unexpected required/small plan"):
            validate(plain, "required", False, 4, pushdown_filters=True)
        with self.assertRaisesRegex(ValueError, "unexpected required/small plan"):
            validate(pushed, "required", False, 4, pushdown_filters=False)

    def test_stock_pushdown_plan_accepts_no_filter_exec_and_rejects_filter_exec(self):
        stock_pushed = report_for("off", pushdown_filters=True)
        validate(stock_pushed, "off", False, 4, pushdown_filters=True)
        stock_with_filter = dict(
            stock_pushed,
            small={
                "plan": (
                    "AggregateExec: mode=Final\n"
                    "  FilterExec: gate@0 < 10\n"
                    "    DataSourceExec: file_groups=small, file_type=parquet\n"
                )
            },
        )
        with self.assertRaisesRegex(ValueError, "unexpected off/small plan"):
            validate(stock_with_filter, "off", False, 4, pushdown_filters=True)

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
