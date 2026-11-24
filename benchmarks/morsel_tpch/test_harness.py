# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

import io
from contextlib import redirect_stdout
from decimal import Decimal
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

from harness import (
    ARMS,
    MORSEL_ADMITTED_QUERIES,
    STANDARD_TPCH_QUERIES,
    TPCH_TABLES,
    WORKLOADS,
    compare_tables,
    dataset_summary,
    generate,
    infer_scale_factor,
    render_workload_sql,
    require_warm_cache,
    resolve_arms,
    resolve_queries,
    summarize_jsonl,
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

EXPECTED_SF001_QUERY_ROWS = {
    "q1": 4,
    "q2": 4,
    "q3": 10,
    "q4": 5,
    "q5": 5,
    "q6": 1,
    "q7": 4,
    "q8": 2,
    "q9": 173,
    "q10": 20,
    "q11": 1,
    "q12": 2,
    "q13": 33,
    "q14": 1,
    "q15": 1,
    "q16": 296,
    "q17": 1,
    "q18": 2,
    "q19": 1,
    "q20": 1,
    "q21": 1,
    "q22": 7,
    "q1_unordered": 4,
    "q6_count": 1,
    "q6_scan": 1191,
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

    def test_generate_sf001_small_row_group_exercises_multi_batch_sharding(self):
        with tempfile.TemporaryDirectory() as tmp:
            out_dir = Path(tmp) / "sf001_rg1000"
            generate(out_dir, sf=0.01, row_group_size=1000)
            summary = dataset_summary(out_dir)

            self.assertEqual(summary["total_rows"], sum(EXPECTED_SF001_ROWS.values()))
            for table, expected_rows in EXPECTED_SF001_ROWS.items():
                self.assertEqual(summary["tables"][table]["rows"], expected_rows)

            # 60,175 rows / 16 files = 3,760 or 3,761 rows/file -> 4 RGs/file = 64 RGs.
            self.assertEqual(summary["tables"]["lineitem"]["files"], 16)
            self.assertEqual(summary["tables"]["lineitem"]["row_groups"], 64)

            # Verify strict primary-key order is preserved across streamed shards.
            prev_key = None
            for part in table_files(out_dir, "lineitem"):
                tbl = pq.read_table(part, columns=["l_orderkey", "l_linenumber"])
                keys = list(
                    zip(
                        tbl.column("l_orderkey").to_pylist(),
                        tbl.column("l_linenumber").to_pylist(),
                    )
                )
                self.assertTrue(keys)
                if prev_key is not None:
                    self.assertLess(prev_key, keys[0])
                self.assertEqual(keys, sorted(keys))
                prev_key = keys[-1]

    def test_validate_plan_enforces_stock_and_morsel_invariants(self):
        stock_ev = {"plan": "DataSourceExec: file_type=parquet\n"}
        morsel_count_ev = {
            "plan": "MorselExec: workers=4, NativeParquet[lineitem] -> Filter[...] -> CountSink; FinalizedCountSource",
            "parquet_morsels_claimed": 16,
            "count_finalizations": 1,
            "aggregate_finalizations": 0,
        }
        morsel_scan_ev = {
            "plan": "MorselExec: workers=4, NativeParquet[lineitem] -> Filter[...] -> Projection[...] -> ResultSink[capacity=2]",
            "parquet_morsels_claimed": 16,
            "count_finalizations": 0,
            "aggregate_finalizations": 0,
        }
        morsel_agg_ev = {
            "plan": "MorselExec: workers=4, NativeParquet[lineitem] -> Filter[...] -> AggregateSink[aggr=[sum(...)]]; FinalizedAggregateSource",
            "parquet_morsels_claimed": 16,
            "count_finalizations": 0,
            "aggregate_finalizations": 1,
        }
        rgs = {"lineitem": 16}

        validate_plan(stock_ev, ARMS["stock_no_pushdown"], WORKLOADS["q6_count"], rgs)
        validate_plan(
            morsel_count_ev, ARMS["morsel_no_pushdown"], WORKLOADS["q6_count"], rgs
        )
        validate_plan(
            morsel_scan_ev, ARMS["morsel_no_pushdown"], WORKLOADS["q6_scan"], rgs
        )
        validate_plan(morsel_agg_ev, ARMS["morsel_no_pushdown"], WORKLOADS["q6"], rgs)
        validate_plan(
            morsel_agg_ev, ARMS["morsel_no_pushdown"], WORKLOADS["q1_unordered"], rgs
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

    def test_all_22_tpch_queries_loaded_and_q15_rewritten_as_cte(self):
        self.assertEqual(len(STANDARD_TPCH_QUERIES), 22)
        for qname in STANDARD_TPCH_QUERIES:
            self.assertIn(qname, WORKLOADS)
            workload = WORKLOADS[qname]
            self.assertTrue(workload.sql)
            self.assertFalse(workload.sql.endswith(";"))
            for table in workload.tables:
                self.assertIn(table, TPCH_TABLES)
                self.assertRegex(
                    workload.sql.lower(),
                    rf"\b{re.escape(table)}\b",
                    msg=f"{qname} declares unused table {table}",
                )

        # Q15 must be rewritten to a single-statement CTE with no DDL view state.
        q15_sql = WORKLOADS["q15"].sql
        self.assertNotIn("create view", q15_sql.lower())
        self.assertNotIn("drop view", q15_sql.lower())
        self.assertTrue(q15_sql.startswith("WITH revenue0"))

        # Q11 must substitute TPC-H FRACTION = 0.0001 / SF cleanly.
        q11_sf5 = render_workload_sql(WORKLOADS["q11"], 5.0)
        self.assertIn("(0.0001 / 5)", q11_sf5)
        self.assertNotIn("__TPCH_Q11_FRACTION__", q11_sf5)
        self.assertEqual(
            infer_scale_factor({"tables": {"part": {"rows": 1_000_000}}}), 5.0
        )

    def test_resolve_queries_and_arms_support_all_morsel_and_comma_lists(self):
        self.assertEqual(
            resolve_queries(["all"], ("q6",)),
            list(STANDARD_TPCH_QUERIES),
        )
        self.assertEqual(
            resolve_queries(["morsel"], ("q1",)),
            list(MORSEL_ADMITTED_QUERIES),
        )
        self.assertEqual(
            resolve_queries(["q1,q3,q6,q12", "q6_count,q6_scan", "q1_unordered"], ()),
            ["q1", "q3", "q6", "q12", "q6_count", "q6_scan", "q1_unordered"],
        )
        # Deduplication preserves first-seen order.
        self.assertEqual(
            resolve_queries(["q6,q1", "q6", "q1,q3"], ()),
            ["q6", "q1", "q3"],
        )
        with self.assertRaisesRegex(ValueError, "unknown query"):
            resolve_queries(["q99"], ())

        self.assertEqual(
            resolve_arms(["stock_default,duckdb_default"], ()),
            ("stock_default", "duckdb_default"),
        )
        with self.assertRaisesRegex(ValueError, "unknown arm"):
            resolve_arms(["not_an_arm"], ())

    def test_compare_tables_normalizes_unaliased_q18_aggregate_name(self):
        duck_q18 = pa.table(
            {
                "c_name": pa.array(["Customer#0001"]),
                "sum(l_quantity)": pa.array(
                    [Decimal("312.00")], type=pa.decimal128(38, 2)
                ),
            }
        )
        df_q18 = pa.table(
            {
                "c_name": pa.array(["Customer#0001"]),
                "sum(lineitem.l_quantity)": pa.array(
                    [Decimal("312.00")], type=pa.decimal128(25, 2)
                ),
            }
        )
        compare_tables(duck_q18, df_q18, ordered=True, case_name="q18", arm_name="t")

        wrong_col = pa.table(
            {
                "c_name": pa.array(["Customer#0001"]),
                "sum(other_col)": pa.array(
                    [Decimal("312.00")], type=pa.decimal128(25, 2)
                ),
            }
        )
        with self.assertRaisesRegex(ValueError, "column mismatch"):
            compare_tables(
                duck_q18, wrong_col, ordered=True, case_name="q18", arm_name="t"
            )

    def test_summarize_jsonl_reports_geomean_and_morsel_rejections(self):
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "sample.jsonl"
            records = [
                {
                    "record_type": "header",
                    "host": "test-host",
                    "arch": "aarch64",
                    "kernel": "7.0",
                    "threads": 8,
                    "cache_regime": "warm",
                    "warmups": 1,
                    "repeats": 1,
                    "build_revision": "deadbeef",
                    "duckdb_version": "2.0.1",
                    "pyarrow_version": "26.0.1",
                    "morsel_rejections": {
                        "q1": "morsel_mode=required: unsupported logical operator: Sort"
                    },
                },
                {
                    "record_type": "trial",
                    "case": "q1",
                    "arm": "stock_default",
                    "seconds": 0.2,
                    "exec_seconds": 0.18,
                    "output_rows": 4,
                    "output_batches": 1,
                    "parquet_morsels_claimed": 0,
                },
                {
                    "record_type": "trial",
                    "case": "q1",
                    "arm": "duckdb_default",
                    "seconds": 0.1,
                    "exec_seconds": None,
                    "output_rows": 4,
                    "output_batches": 1,
                    "parquet_morsels_claimed": None,
                },
                {
                    "record_type": "trial",
                    "case": "q6",
                    "arm": "stock_default",
                    "seconds": 0.4,
                    "exec_seconds": 0.38,
                    "output_rows": 1,
                    "output_batches": 1,
                    "parquet_morsels_claimed": 0,
                },
                {
                    "record_type": "trial",
                    "case": "q6",
                    "arm": "duckdb_default",
                    "seconds": 0.05,
                    "exec_seconds": None,
                    "output_rows": 1,
                    "output_batches": 1,
                    "parquet_morsels_claimed": None,
                },
                # Non-standard slice with both arms must be excluded from standard-query
                # geomean and sum so q1/q6 slices are never double-counted.
                {
                    "record_type": "trial",
                    "case": "q6_count",
                    "arm": "stock_default",
                    "seconds": 9.9,
                    "exec_seconds": 9.8,
                    "output_rows": 1,
                    "output_batches": 1,
                    "parquet_morsels_claimed": 0,
                },
                {
                    "record_type": "trial",
                    "case": "q6_count",
                    "arm": "duckdb_default",
                    "seconds": 0.01,
                    "exec_seconds": None,
                    "output_rows": 1,
                    "output_batches": 1,
                    "parquet_morsels_claimed": None,
                },
            ]
            jsonl.write_text("\n".join(json.dumps(r) for r in records) + "\n")
            buf = io.StringIO()
            with redirect_stdout(buf):
                summarize_jsonl(jsonl)
            out = buf.getvalue()
            # Only q1 (2.0x) and q6 (8.0x) count -> 2/22 standard queries, geomean=4.00x,
            # sum=600.0 ms vs 150.0 ms (q6_count's 9900 ms must be excluded).
            self.assertIn("2/22 standard queries", out)
            self.assertIn("geomean ratio=4.00x", out)
            self.assertIn("sum of medians: 600.0 ms vs 150.0 ms", out)
            self.assertIn("unsupported logical operator: Sort", out)

    def test_require_warm_cache_rejects_partial_residency(self):
        with patch("harness.cache_pages", return_value={"resident": 1, "total": 2}):
            with self.assertRaisesRegex(ValueError, "cannot verify warm"):
                require_warm_cache([])

    def test_sf001_duckdb_profile_reports_operator_timings(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "tpch sf001 dprof"
            generate(data_dir, sf=0.01, row_group_size=131_072)
            cache_mode = "warm" if sys.platform == "linux" else "none"
            out = subprocess.check_output(
                [
                    sys.executable,
                    "-B",
                    str(Path(__file__).with_name("harness.py")),
                    "duckdb-profile",
                    str(data_dir),
                    "--threads",
                    "2",
                    "--timeout",
                    "60",
                    "--cache",
                    cache_mode,
                    "--warmups",
                    "0",
                    "--repeats",
                    "2",
                    "--queries",
                    "q6",
                    "q1_unordered",
                ],
                text=True,
                timeout=120,
            )
            report = json.loads(out)
            self.assertEqual(
                report["methodology"]["isolation"], "fresh_process_and_connection"
            )
            self.assertEqual(report["methodology"]["aggregation"], "median")
            self.assertEqual(set(report["queries"].keys()), {"q6", "q1_unordered"})
            self.assertEqual(report["queries"]["q6"]["output_rows"], 1)
            self.assertEqual(report["queries"]["q1_unordered"]["output_rows"], 4)
            self.assertIn(
                "TABLE_SCAN", report["queries"]["q6"]["operator_timing_ms_median"]
            )
            self.assertIn(
                "HASH_GROUP_BY",
                report["queries"]["q1_unordered"]["operator_timing_ms_median"],
            )
            self.assertGreater(report["queries"]["q6"]["total_operator_ms_median"], 0.0)
            self.assertGreater(
                report["queries"]["q1_unordered"]["total_operator_ms_median"], 0.0
            )

            md = subprocess.check_output(
                [
                    sys.executable,
                    "-B",
                    str(Path(__file__).with_name("harness.py")),
                    "duckdb-profile",
                    str(data_dir),
                    "--threads",
                    "2",
                    "--timeout",
                    "60",
                    "--cache",
                    cache_mode,
                    "--warmups",
                    "0",
                    "--repeats",
                    "1",
                    "--queries",
                    "q6",
                    "--format",
                    "markdown",
                ],
                text=True,
                timeout=120,
            )
            self.assertIn("`q6`", md)
            self.assertIn("`TABLE_SCAN`", md)

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
            self.assertEqual(report["scale_factor"], 0.01)
            self.assertEqual(
                {k: v["rows"] for k, v in report["verified"].items()},
                EXPECTED_SF001_QUERY_ROWS,
            )
            expected_rejected = set(STANDARD_TPCH_QUERIES) - set(
                MORSEL_ADMITTED_QUERIES
            )
            self.assertEqual(set(report["morsel_rejections"].keys()), expected_rejected)
            for qname, err in report["morsel_rejections"].items():
                self.assertIn("morsel_mode=required", err, msg=qname)

            morsel_ev = {
                (e["case"], e["arm"]): e
                for e in report["evidence"]
                if e["mode"] == "required"
            }
            for qname in MORSEL_ADMITTED_QUERIES:
                for arm_name in (
                    "morsel_no_pushdown",
                    "morsel_pruning",
                    "morsel_pushdown",
                ):
                    self.assertIn((qname, arm_name), morsel_ev)
            self.assertEqual(
                morsel_ev[("q6", "morsel_no_pushdown")]["aggregate_finalizations"],
                1,
            )
            self.assertEqual(
                morsel_ev[("q1_unordered", "morsel_pushdown")][
                    "aggregate_finalizations"
                ],
                1,
            )
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
                    "q1",
                    "q6",
                    "q1_unordered",
                    "q6_count",
                    "q6_scan",
                    "--roadmap-arms",
                    "stock_default",
                    "duckdb_default",
                    "--morsel-arms",
                    "stock_no_pushdown",
                    "stock_pushdown_no_repartition",
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
            self.assertIn("q1", header["morsel_rejections"])
            # 2 arms * 2 reps for q1 + 4 arms * 2 reps for (q6, q1_unordered, q6_count, q6_scan) = 36
            self.assertEqual(len(trials), 36)
            for trial in trials:
                self.assertGreater(trial["seconds"], 0)
                if sys.platform == "linux" and sys.maxsize > 2**32:
                    self.assertIsNotNone(trial["exec_cpu_seconds"])
                    self.assertGreater(trial["exec_cpu_seconds"], 0)
                if trial["engine"] == "datafusion":
                    self.assertIsInstance(trial["operator_compute_ms"], dict)
                    self.assertGreater(len(trial["operator_compute_ms"]), 0)
                else:
                    self.assertIsNone(trial["operator_compute_ms"])
                before = trial["cache_pages_before"]
                after = trial["cache_pages_after"]
                self.assertGreater(before["total"], 0)
                self.assertEqual(before["resident"], before["total"])
                self.assertEqual(after["resident"], after["total"])
                morsel_metric_fields = (
                    "morsel_decode_work_ms",
                    "morsel_process_work_ms",
                    "morsel_ready_wait_ms",
                    "morsel_output_wait_ms",
                    "morsel_pending_io_ms",
                    "morsel_peak_ready_depth",
                    "morsel_peak_output_depth",
                )
                if trial["mode"] == "required":
                    for field in morsel_metric_fields:
                        self.assertIsNotNone(trial[field], msg=field)
                        self.assertGreaterEqual(trial[field], 0, msg=field)
                    self.assertEqual(len(trial["worker_claims"]), 2)
                    self.assertEqual(
                        sum(trial["worker_claims"]), trial["parquet_morsels_claimed"]
                    )
                    self.assertEqual(
                        sum(trial["worker_completions"]),
                        trial["parquet_morsels_claimed"],
                    )
                else:
                    for field in morsel_metric_fields:
                        self.assertIsNone(trial[field], msg=field)
                    self.assertIsNone(trial["worker_claims"])
                    self.assertIsNone(trial["worker_completions"])

            # Verify alternating arm order on q6_count between rep 1 and rep 2.
            rep1_arms = [
                t["arm"] for t in trials if t["case"] == "q6_count" and t["rep"] == 1
            ]
            rep2_arms = [
                t["arm"] for t in trials if t["case"] == "q6_count" and t["rep"] == 2
            ]
            self.assertEqual(
                rep1_arms,
                [
                    "stock_no_pushdown",
                    "stock_pushdown_no_repartition",
                    "morsel_no_pushdown",
                    "duckdb_default",
                ],
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
                "`q1_unordered`",
                "`q6_count`",
                "`q6_scan`",
                "`stock_no_pushdown`",
                "`stock_pushdown_no_repartition`",
                "`morsel_no_pushdown`",
                "`duckdb_default`",
                "Decode work ms",
                "Process work ms",
            ):
                self.assertIn(expected_token, summary)

    @unittest.skipUnless(
        os.getenv("MORSEL_TPCH_RUNNER") and os.getenv("MORSEL_BENCH_REVISION"),
        "set MORSEL_TPCH_RUNNER and MORSEL_BENCH_REVISION for the MORSEL_METRICS=0 test",
    )
    def test_sf001_morsel_metrics_disabled_emits_null_fields(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "tpch sf001 metrics off"
            generate(data_dir, sf=0.01, row_group_size=131_072)
            env = dict(os.environ, MORSEL_METRICS="0")
            out = subprocess.check_output(
                [
                    os.environ["MORSEL_TPCH_RUNNER"],
                    "measure",
                    "required",
                    "no_pushdown",
                    "2",
                    str(data_dir),
                    "lineitem",
                    WORKLOADS["q6"].sql,
                ],
                text=True,
                env=env,
                timeout=60,
            )
            sample = json.loads(out)
            self.assertEqual(sample["mode"], "required")
            self.assertEqual(sample["parquet_morsels_claimed"], 16)
            self.assertEqual(sample["aggregate_finalizations"], 1)
            self.assertIsInstance(sample["operator_compute_ms"], dict)
            self.assertGreater(len(sample["operator_compute_ms"]), 0)
            if sys.platform == "linux" and sys.maxsize > 2**32:
                self.assertIsNotNone(sample["exec_cpu_seconds"])
                self.assertGreater(sample["exec_cpu_seconds"], 0)
            for field in (
                "worker_claims",
                "worker_completions",
                "morsel_decode_work_ms",
                "morsel_process_work_ms",
                "morsel_ready_wait_ms",
                "morsel_output_wait_ms",
                "morsel_pending_io_ms",
                "morsel_peak_ready_depth",
                "morsel_peak_output_depth",
            ):
                self.assertIsNone(sample[field], msg=field)
