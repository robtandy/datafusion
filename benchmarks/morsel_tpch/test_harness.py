# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

from decimal import Decimal
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from harness import (
    ARMS,
    TPCH_TABLES,
    WORKLOADS,
    compare_tables,
    dataset_summary,
    generate,
    require_warm_cache,
    table_files,
    validate_plan,
)

EXPECTED_SF001_ROWS = {
    "lineitem": 60175,
    "orders": 15000,
    "partsupp": 8000,
    "part": 2000,
    "customer": 1500,
    "supplier": 100,
    "nation": 25,
    "region": 5,
}


class TpchHarnessTest(unittest.TestCase):
    def test_generate_sf001_is_repeatable_and_standard_layout(self):
        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "first sf001"
            second = Path(tmp) / "second sf001"
            generate(first, sf=0.01, row_group_size=131_072)
            generate(second, sf=0.01, row_group_size=131_072)

            summary = dataset_summary(first)
            self.assertEqual(set(summary["tables"].keys()), set(TPCH_TABLES.keys()))
            for table, expected_rows in EXPECTED_SF001_ROWS.items():
                self.assertEqual(summary["tables"][table]["rows"], expected_rows)
                files_a = table_files(first, table)
                files_b = table_files(second, table)
                self.assertEqual([f.name for f in files_a], [f.name for f in files_b])
                for fa, fb in zip(files_a, files_b):
                    self.assertEqual(fa.read_bytes(), fb.read_bytes())
                    pf = pq.ParquetFile(fa)
                    for rg_idx in range(pf.num_row_groups):
                        rg = pf.metadata.row_group(rg_idx)
                        for col_idx in range(rg.num_columns):
                            col = rg.column(col_idx)
                            self.assertEqual(col.compression, "ZSTD")
                            self.assertTrue(col.is_stats_set)

            lineitem_schema = pq.ParquetFile(
                table_files(first, "lineitem")[0]
            ).schema_arrow
            self.assertEqual(lineitem_schema.field("l_orderkey").type, pa.int64())
            self.assertEqual(
                lineitem_schema.field("l_quantity").type, pa.decimal128(15, 2)
            )
            self.assertEqual(lineitem_schema.field("l_shipdate").type, pa.date32())

    def test_validate_plan_enforces_stock_and_morsel_invariants(self):
        stock_ev = {"plan": "DataSourceExec: file_type=parquet\n"}
        morsel_count_ev = {
            "plan": "MorselExec: workers=4, NativeParquet[lineitem] -> Filter[...] -> CountSink; FinalizedCountSource",
            "parquet_morsels_claimed": 16,
            "count_finalizations": 1,
        }
        morsel_scan_ev = {
            "plan": "MorselExec: workers=4, NativeParquet[lineitem] -> Filter[...] -> Projection[...] -> ResultSink[capacity=2]",
            "parquet_morsels_claimed": 16,
            "count_finalizations": 0,
        }
        rgs = {"lineitem": 16}

        validate_plan(stock_ev, ARMS["stock_no_pushdown"], WORKLOADS["q6_count"], rgs)
        validate_plan(
            morsel_count_ev, ARMS["morsel_no_pushdown"], WORKLOADS["q6_count"], rgs
        )
        validate_plan(
            morsel_scan_ev, ARMS["morsel_no_pushdown"], WORKLOADS["q6_scan"], rgs
        )

        # Stock arm must reject MorselExec plan.
        with self.assertRaises(ValueError):
            validate_plan(
                morsel_count_ev, ARMS["stock_no_pushdown"], WORKLOADS["q6_count"], rgs
            )

        # Morsel arm must reject stock fallback, partial RG claims, or mismatched CountSink.
        for bad in (
            stock_ev,
            dict(morsel_count_ev, parquet_morsels_claimed=8),
            dict(morsel_count_ev, count_finalizations=0),
            dict(morsel_count_ev, plan=morsel_scan_ev["plan"]),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                validate_plan(
                    bad, ARMS["morsel_no_pushdown"], WORKLOADS["q6_count"], rgs
                )

        # Scan query must reject unexpected CountSink / count_finalizations.
        with self.assertRaises(ValueError):
            validate_plan(
                morsel_count_ev, ARMS["morsel_no_pushdown"], WORKLOADS["q6_scan"], rgs
            )

    def test_compare_tables_handles_decimal_and_order_semantics(self):
        oracle = pa.table(
            {
                "k": pa.array([2, 1, 2], type=pa.int64()),
                "v": pa.array(
                    [Decimal("100.00"), Decimal("20.00"), Decimal("100.00")],
                    type=pa.decimal128(15, 2),
                ),
            }
        )
        permuted = pa.table(
            {
                "k": pa.array([1, 2, 2], type=pa.int64()),
                "v": pa.array(
                    [Decimal("20.0000"), Decimal("100.0000"), Decimal("100.0000")],
                    type=pa.decimal128(18, 4),
                ),
            }
        )
        compare_tables(
            oracle, permuted, ordered=False, case_name="q6_scan", arm_name="t"
        )
        with self.assertRaises(ValueError):
            compare_tables(
                oracle, permuted, ordered=True, case_name="q6_scan", arm_name="t"
            )

        wrong_val = pa.table(
            {
                "k": pa.array([1, 2, 2], type=pa.int64()),
                "v": pa.array(
                    [Decimal("20.0000"), Decimal("100.0001"), Decimal("100.0000")],
                    type=pa.decimal128(18, 4),
                ),
            }
        )
        with self.assertRaises(ValueError):
            compare_tables(
                oracle, wrong_val, ordered=False, case_name="q6_scan", arm_name="t"
            )

        # DuckDB float64 AVG vs DataFusion Decimal128(19, 6) AVG
        duck_avg = pa.table(
            {"avg_qty": pa.array([25.575154611454693], type=pa.float64())}
        )
        df_avg = pa.table(
            {"avg_qty": pa.array([Decimal("25.575155")], type=pa.decimal128(19, 6))}
        )
        compare_tables(duck_avg, df_avg, ordered=True, case_name="q1", arm_name="t")

        # DuckDB decimal(38,0) vs DataFusion int64
        duck_sum = pa.table(
            {"cnt": pa.array([Decimal("64")], type=pa.decimal128(38, 0))}
        )
        df_sum = pa.table({"cnt": pa.array([64], type=pa.int64())})
        compare_tables(duck_sum, df_sum, ordered=True, case_name="q12", arm_name="t")

    def test_require_warm_cache_rejects_partial_residency(self):
        with patch("harness.cache_pages", return_value={"resident": 1, "total": 2}):
            with self.assertRaisesRegex(ValueError, "cannot verify warm"):
                require_warm_cache([])

    @unittest.skipUnless(
        os.getenv("MORSEL_TPCH_RUNNER") and os.getenv("MORSEL_BENCH_REVISION"),
        "set MORSEL_TPCH_RUNNER and MORSEL_BENCH_REVISION for the end-to-end SF0.01 smoke test",
    )
    def test_sf001_verify_parity_and_morsel_plan_checks(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "tpch sf001"
            generate(data_dir, sf=0.01, row_group_size=131_072)
            out = subprocess.check_output(
                [
                    sys.executable,
                    "-B",
                    str(Path(__file__).with_name("harness.py")),
                    "verify",
                    str(data_dir),
                    "--runner",
                    os.environ["MORSEL_TPCH_RUNNER"],
                    "--threads",
                    "2",
                    "--timeout",
                    "60",
                ],
                text=True,
                timeout=180,
            )
            report = json.loads(out)
            self.assertEqual(
                {k: v["rows"] for k, v in report["verified"].items()},
                {
                    "q6": 1,
                    "q1": 4,
                    "q3": 10,
                    "q12": 2,
                    "q6_count": 1,
                    "q6_scan": 1191,
                },
            )
            self.assertEqual(
                set(report["morsel_rejections"].keys()), {"q6", "q1", "q3", "q12"}
            )
            for qname, err in report["morsel_rejections"].items():
                self.assertIn("morsel_mode=required", err, msg=qname)

            morsel_ev = {
                (e["case"], e["arm"]): e
                for e in report["evidence"]
                if e["mode"] == "required"
            }
            self.assertIn(("q6_count", "morsel_no_pushdown"), morsel_ev)
            self.assertIn(("q6_scan", "morsel_no_pushdown"), morsel_ev)
            self.assertEqual(
                morsel_ev[("q6_count", "morsel_no_pushdown")][
                    "parquet_morsels_claimed"
                ],
                16,
            )
            self.assertEqual(
                morsel_ev[("q6_count", "morsel_no_pushdown")]["count_finalizations"],
                1,
            )
            self.assertEqual(
                morsel_ev[("q6_scan", "morsel_no_pushdown")]["parquet_morsels_claimed"],
                16,
            )
            self.assertEqual(
                morsel_ev[("q6_scan", "morsel_no_pushdown")]["count_finalizations"],
                0,
            )

    @unittest.skipUnless(
        sys.platform == "linux"
        and os.getenv("MORSEL_TPCH_RUNNER")
        and os.getenv("MORSEL_BENCH_REVISION"),
        "set MORSEL_TPCH_RUNNER and MORSEL_BENCH_REVISION for the Linux bench smoke test",
    )
    def test_sf001_bench_smoke_with_warm_cache_and_summarize(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "tpch sf001 bench"
            generate(data_dir, sf=0.01, row_group_size=131_072)
            out = subprocess.check_output(
                [
                    sys.executable,
                    "-B",
                    str(Path(__file__).with_name("harness.py")),
                    "bench",
                    str(data_dir),
                    "--runner",
                    os.environ["MORSEL_TPCH_RUNNER"],
                    "--revision",
                    os.environ["MORSEL_BENCH_REVISION"],
                    "--threads",
                    "2",
                    "--timeout",
                    "60",
                    "--cache",
                    "warm",
                    "--warmups",
                    "0",
                    "--repeats",
                    "2",
                    "--queries",
                    "q6",
                    "q6_count",
                    "q6_scan",
                    "--roadmap-arms",
                    "stock_default",
                    "duckdb_default",
                    "--morsel-arms",
                    "stock_no_pushdown",
                    "morsel_no_pushdown",
                    "duckdb_default",
                ],
                text=True,
                timeout=180,
            )
            jsonl_path = Path(tmp) / "smoke.jsonl"
            jsonl_path.write_text(out)
            records = [json.loads(line) for line in out.splitlines()]
            header, trials = records[0], records[1:]
            self.assertEqual(header["record_type"], "header")
            self.assertEqual(header["cache_regime"], "warm")
            self.assertIn("q6", header["morsel_rejections"])
            # 2 arms * 2 reps for q6 + 3 arms * 2 reps for q6_count + 3 arms * 2 reps for q6_scan = 16
            self.assertEqual(len(trials), 16)
            for trial in trials:
                self.assertGreater(trial["seconds"], 0)
                before = trial["cache_pages_before"]
                after = trial["cache_pages_after"]
                self.assertGreater(before["total"], 0)
                self.assertEqual(before["resident"], before["total"])
                self.assertEqual(after["resident"], after["total"])

            # Verify alternating arm order on q6_count between rep 1 and rep 2.
            rep1_arms = [
                t["arm"] for t in trials if t["case"] == "q6_count" and t["rep"] == 1
            ]
            rep2_arms = [
                t["arm"] for t in trials if t["case"] == "q6_count" and t["rep"] == 2
            ]
            self.assertEqual(
                rep1_arms, ["stock_no_pushdown", "morsel_no_pushdown", "duckdb_default"]
            )
            self.assertEqual(rep2_arms, list(reversed(rep1_arms)))

            summary = subprocess.check_output(
                [
                    sys.executable,
                    "-B",
                    str(Path(__file__).with_name("harness.py")),
                    "summarize",
                    str(jsonl_path),
                ],
                text=True,
                timeout=30,
            )
            for expected_token in (
                "`q6`",
                "`q6_count`",
                "`q6_scan`",
                "`stock_no_pushdown`",
                "`morsel_no_pushdown`",
                "`duckdb_default`",
            ):
                self.assertIn(expected_token, summary)
