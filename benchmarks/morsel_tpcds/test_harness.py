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

import pyarrow as pa

import harness


def _runner_path() -> Path | None:
    raw = os.environ.get("MORSEL_TPCDS_RUNNER")
    if not raw:
        return None
    p = Path(raw)
    return p if p.is_file() else None


def _runner_revision() -> str:
    return os.environ.get("MORSEL_BENCH_REVISION", "test-rev")


class QueryParsingTests(unittest.TestCase):
    def test_all_99_queries_loaded(self):
        self.assertEqual(len(harness.WORKLOADS), 99)
        self.assertEqual(
            tuple(harness.WORKLOADS.keys()), harness.STANDARD_TPCDS_QUERIES
        )
        for qname, w in harness.WORKLOADS.items():
            self.assertGreaterEqual(w.statement_count, 1, qname)
            self.assertGreaterEqual(len(w.tables), 1, qname)
            for tbl in w.tables:
                self.assertIn(tbl, harness.TPCDS_TABLES)

    def test_multi_statement_queries_documented(self):
        self.assertEqual(harness.MULTI_STATEMENT_QUERIES, ("q14", "q23", "q24", "q39"))
        for qname in harness.MULTI_STATEMENT_QUERIES:
            self.assertEqual(harness.WORKLOADS[qname].statement_count, 2)

    def test_resolve_queries(self):
        self.assertEqual(
            harness.resolve_queries(["q1,q02", "q03-q05"]),
            ["q01", "q02", "q03", "q04", "q05"],
        )
        self.assertEqual(len(harness.resolve_queries(["all"])), 99)
        with self.assertRaises(ValueError):
            harness.resolve_queries(["q100"])
        with self.assertRaises(ValueError):
            harness.resolve_queries(["q10-q05"])

    def test_duckdb_q90_alias_adaptation(self):
        q90_sql = harness.WORKLOADS["q90"].statements[0]
        adapted = harness.adapt_sql_for_duckdb("q90", q90_sql)
        self.assertIn(') "at",', adapted)
        self.assertNotEqual(q90_sql, adapted)
        q03_sql = harness.WORKLOADS["q03"].statements[0]
        self.assertEqual(harness.adapt_sql_for_duckdb("q03", q03_sql), q03_sql)

    def test_top_level_order_by_detection(self):
        self.assertTrue(
            harness.has_top_level_order_by("SELECT a FROM t ORDER BY a LIMIT 10")
        )
        self.assertFalse(
            harness.has_top_level_order_by(
                "SELECT SUM(x) FROM (SELECT x FROM t ORDER BY x) s LIMIT 10"
            )
        )


class RejectionClassificationTests(unittest.TestCase):
    def test_classify_morsel_rejection_categories(self):
        self.assertEqual(
            harness.classify_morsel_rejection("unsupported logical operator: Window"),
            "Window",
        )
        self.assertEqual(
            harness.classify_morsel_rejection(
                "unsupported GROUP BY: GroupingSet / ROLLUP not supported"
            ),
            "ROLLUP / GROUPING SETS / CUBE",
        )
        self.assertEqual(
            harness.classify_morsel_rejection("unsupported logical operator: Union"),
            "UNION / INTERSECT / EXCEPT",
        )
        self.assertEqual(
            harness.classify_morsel_rejection("unsupported logical operator: Limit"),
            "LIMIT without ORDER BY",
        )
        self.assertEqual(
            harness.classify_morsel_rejection(
                "unsupported logical operator: CrossJoin"
            ),
            "CrossJoin / Non-equi Join",
        )
        self.assertEqual(
            harness.classify_morsel_rejection(
                "unsupported join type Full in native morsel hash join"
            ),
            "Unsupported Join Type (Full)",
        )

    def test_detect_plan_blockers_and_unlock_roadmap(self):
        blockers = harness.detect_plan_blockers(
            [
                "Union\n"
                "  WindowAggr: windowExpr=[[rank()]]\n"
                "    Aggregate: groupBy=[[ROLLUP(a, b)]]\n"
                "      LeftMark Join: a = b\n"
                "        Aggregate: aggr=[[stddev_samp(x)]]\n"
            ],
            [],
            "unsupported logical operator: Union",
        )
        self.assertIn("Window", blockers)
        self.assertIn("UNION / INTERSECT / EXCEPT", blockers)
        self.assertIn("ROLLUP / GROUPING SETS / CUBE", blockers)
        self.assertIn("Unsupported Join Type (LeftMark)", blockers)
        self.assertIn("Unsupported Aggregate (stddev/var)", blockers)

        audit = harness.summarize_rejection_audit(
            {
                "q01": {
                    "admitted": True,
                    "primary_rejection_category": None,
                    "all_blockers": [],
                },
                "q02": {
                    "admitted": False,
                    "primary_rejection_category": "Window",
                    "all_blockers": ["Window"],
                },
                "q03": {
                    "admitted": False,
                    "primary_rejection_category": "Window",
                    "all_blockers": ["Window", "UNION / INTERSECT / EXCEPT"],
                },
                "q04": {
                    "admitted": False,
                    "primary_rejection_category": "UNION / INTERSECT / EXCEPT",
                    "all_blockers": ["UNION / INTERSECT / EXCEPT"],
                },
            }
        )
        self.assertEqual(audit["admitted_count"], 1)
        self.assertEqual(audit["rejected_count"], 3)
        self.assertEqual(audit["greedy_unlock_roadmap"][0]["cumulative_admitted"], 2)
        self.assertEqual(audit["greedy_unlock_roadmap"][1]["cumulative_admitted"], 4)


class TableComparisonTests(unittest.TestCase):
    def test_decimal_float_and_unordered_parity(self):
        oracle = pa.table({"k": ["b", "a"], "v": [1.5, None]})
        actual = pa.table({"k": ["a", "b"], "v": [None, Decimal("1.500000")]})
        harness.compare_tables(
            oracle, actual, ordered=False, case_name="t", arm_name="a"
        )

    def test_mismatch_raises(self):
        oracle = pa.table({"k": [1, 2]})
        actual = pa.table({"k": [1, 3]})
        with self.assertRaises(ValueError):
            harness.compare_tables(
                oracle, actual, ordered=True, case_name="t", arm_name="a"
            )

    def test_strict_mode_exact_decimals_names_and_floats(self):
        # Exact Decimals, matching column names, and 1e-10 float tolerance pass.
        oracle = pa.table(
            {"k": ["a", "b"], "d": [Decimal("12.345000"), None], "f": [1.0, 2.0]}
        )
        same = pa.table(
            {
                "k": ["a", "b"],
                "d": [Decimal("12.345000"), None],
                "f": [1.0 + 1e-10, 2.0],
            }
        )
        harness.compare_tables(
            oracle, same, ordered=True, case_name="t", arm_name="morsel", strict=True
        )

        # Tied ORDER BY rows still pass via ordered-to-multiset fallback in strict mode.
        swapped = pa.table(
            {"k": ["b", "a"], "d": [None, Decimal("12.345000")], "f": [2.0, 1.0]}
        )
        harness.compare_tables(
            oracle, swapped, ordered=True, case_name="t", arm_name="morsel", strict=True
        )

        # Strict mode rejects 1e-6 Decimal drift.
        dec_drift = pa.table(
            {"k": ["a", "b"], "d": [Decimal("12.345001"), None], "f": [1.0, 2.0]}
        )
        with self.assertRaises(ValueError):
            harness.compare_tables(
                oracle,
                dec_drift,
                ordered=True,
                case_name="t",
                arm_name="morsel",
                strict=True,
            )

        # Strict mode rejects 1e-6 float drift.
        flt_drift = pa.table(
            {
                "k": ["a", "b"],
                "d": [Decimal("12.345000"), None],
                "f": [1.0 + 1e-6, 2.0],
            }
        )
        with self.assertRaises(ValueError):
            harness.compare_tables(
                oracle,
                flt_drift,
                ordered=True,
                case_name="t",
                arm_name="morsel",
                strict=True,
            )

        # Strict mode rejects mismatched column names.
        bad_names = pa.table(
            {"k": ["a", "b"], "d_other": [Decimal("12.345000"), None], "f": [1.0, 2.0]}
        )
        with self.assertRaises(ValueError):
            harness.compare_tables(
                oracle,
                bad_names,
                ordered=True,
                case_name="t",
                arm_name="morsel",
                strict=True,
            )

    def test_clickhouse_error_classification(self):
        runner = harness.ClickHouseRunner(
            Path(sys.executable), Path(__file__).resolve().parent, threads=1, timeout=5
        )
        for code in (43, 47, 62, 184, 206, 207, 558):
            with self.assertRaises(harness.ClickHouseDialectError):
                runner._raise_error(
                    "query",
                    "q01",
                    code,
                    f"Code: {code}. DB::Exception: dialect error",
                )
        for non_dialect in (
            "Code: 57. DB::Exception: Table default.date_dim already exists. (TABLE_ALREADY_EXISTS)",
            "Code: 107. DB::Exception: File not found. (FILE_DOESNT_EXIST)",
            "Code: 241. DB::Exception: Memory limit exceeded. (MEMORY_LIMIT_EXCEEDED)",
            "unexpected crash without error code",
        ):
            with self.assertRaises(RuntimeError) as ctx:
                runner._raise_error("query", "q03", 1, non_dialect)
            self.assertNotIsInstance(ctx.exception, harness.ClickHouseDialectError)

    def test_degenerate_result_detection_and_subset_geomean(self):
        empty = pa.table({"a": pa.array([], type=pa.int64())})
        all_zero_null = pa.table(
            {"c": [0], "s": [None], "d": [Decimal("0.00")], "f": [0.0]}
        )
        non_zero = pa.table({"c": [1], "s": [None]})
        two_zero_rows = pa.table({"c": [0, 0]})

        self.assertTrue(harness.is_degenerate_result([empty]))
        self.assertTrue(harness.is_degenerate_result([all_zero_null]))
        self.assertTrue(harness.is_degenerate_result([empty, all_zero_null]))
        self.assertFalse(harness.is_degenerate_result([non_zero]))
        self.assertFalse(harness.is_degenerate_result([two_zero_rows]))
        self.assertFalse(harness.is_degenerate_result([empty, non_zero]))

        # Verify summarize_jsonl prints both all-admitted and non-degenerate-admitted geomeans.
        header = {
            "record_type": "header",
            "host": "test-host",
            "arch": "aarch64",
            "kernel": "7.0",
            "threads": 8,
            "cache_regime": "warm",
            "warmups": 1,
            "repeats": 1,
            "build_revision": "rev",
            "duckdb_version": "2.0.1",
            "clickhouse_version": "26.11",
            "pyarrow_version": "26.0.1",
            "verified": {
                "q01": {
                    "rows": 0,
                    "morsel_admitted": True,
                    "degenerate_result": True,
                },
                "q03": {
                    "rows": 38,
                    "morsel_admitted": True,
                    "degenerate_result": False,
                },
            },
            "degenerate_queries": ["q01"],
            "degenerate_admitted_queries": ["q01"],
        }
        trials = [
            # q01 (degenerate): morsel=10ms, stock=20ms, duckdb=40ms, ch=80ms
            {
                "record_type": "trial",
                "case": "q01",
                "arm": "stock_default",
                "seconds": 0.020,
            },
            {
                "record_type": "trial",
                "case": "q01",
                "arm": "stock_pushdown",
                "seconds": 0.020,
            },
            {
                "record_type": "trial",
                "case": "q01",
                "arm": "morsel_pushdown",
                "seconds": 0.010,
            },
            {
                "record_type": "trial",
                "case": "q01",
                "arm": "duckdb_default",
                "seconds": 0.040,
            },
            {
                "record_type": "trial",
                "case": "q01",
                "arm": "clickhouse_default",
                "seconds": 0.080,
            },
            # q03 (non-degenerate): morsel=90ms, stock=100ms, duckdb=60ms, ch=180ms
            {
                "record_type": "trial",
                "case": "q03",
                "arm": "stock_default",
                "seconds": 0.100,
            },
            {
                "record_type": "trial",
                "case": "q03",
                "arm": "stock_pushdown",
                "seconds": 0.095,
            },
            {
                "record_type": "trial",
                "case": "q03",
                "arm": "morsel_pushdown",
                "seconds": 0.090,
            },
            {
                "record_type": "trial",
                "case": "q03",
                "arm": "duckdb_default",
                "seconds": 0.060,
            },
            {
                "record_type": "trial",
                "case": "q03",
                "arm": "clickhouse_default",
                "seconds": 0.180,
            },
        ]
        with tempfile.TemporaryDirectory() as tmp:
            jsonl_path = Path(tmp) / "summary_test.jsonl"
            jsonl_path.write_text(
                "\n".join(json.dumps(r) for r in [header, *trials]) + "\n"
            )
            md = harness.summarize_jsonl(jsonl_path)
        # All-2 admitted geomean: morsel=sqrt(10*90)=30.0ms, duckdb=sqrt(40*60)=49.0ms
        self.assertIn("**GeoMean (2 admitted)**", md)
        self.assertIn("**30.0**", md)
        # Non-degenerate 1-query geomean: morsel=90.0ms, stock=100.0ms, duckdb=60.0ms (1.50x)
        self.assertIn("**GeoMean (1 non-degenerate admitted)**", md)
        self.assertIn("**90.0**", md)
        self.assertIn("**1.50x**", md)
        self.assertIn("Degenerate admitted queries (1/2", md)
        self.assertIn("`q01`", md)


class EndToEndSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.data_dir = Path(cls.tmp.name) / "tpcds_sf001"
        harness.generate(cls.data_dir, sf=0.01, row_group_size=131_072)
        cls.runner = _runner_path()
        cls.ch_bin = harness.find_clickhouse_binary()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_dataset_generation_deterministic(self):
        summary = harness.dataset_summary(self.data_dir)
        self.assertEqual(len(summary["tables"]), 24)
        self.assertEqual(summary["total_files"], 98)
        self.assertGreater(summary["total_rows"], 100_000)

        second_dir = Path(self.tmp.name) / "tpcds_sf001_copy"
        harness.generate(second_dir, sf=0.01, row_group_size=131_072)
        for table in harness.TPCDS_TABLES:
            files_a = harness.table_files(self.data_dir, table)
            files_b = harness.table_files(second_dir, table)
            self.assertEqual([f.name for f in files_a], [f.name for f in files_b])
            for fa, fb in zip(files_a, files_b):
                self.assertEqual(fa.read_bytes(), fb.read_bytes(), f"{table}/{fa.name}")

    def test_duckdb_single_multi_and_q90_smoke(self):
        dk = harness.DuckDbRunner(self.data_dir, threads=2, timeout=30)
        for qname in ("q03", "q39", "q90"):
            w = harness.WORKLOADS[qname]
            tbls = [dk.query(w, stmt_index=i) for i in range(w.statement_count)]
            measured = dk.measure(w)
            self.assertEqual(measured["output_rows"], sum(t.num_rows for t in tbls))
            self.assertGreater(measured["seconds"], 0.0)

    def test_clickhouse_smoke_and_dialect_logging(self):
        if self.ch_bin is None:
            self.skipTest("clickhouse binary not found")
        ch = harness.ClickHouseRunner(self.ch_bin, self.data_dir, threads=2, timeout=30)
        dk = harness.DuckDbRunner(self.data_dir, threads=2, timeout=30)
        # q03 is compatible with ClickHouse and must succeed across repeated isolated
        # invocations (regression guard against ClickHouse 26.12 persistent default DB
        # causing Code 57 TABLE_ALREADY_EXISTS on the second CREATE VIEW).
        w3 = harness.WORKLOADS["q03"]
        oracle_q3 = dk.query(w3)
        harness.compare_tables(
            oracle_q3, ch.query(w3), True, "q03", "clickhouse_default"
        )
        harness.compare_tables(
            oracle_q3, ch.query(w3), True, "q03", "clickhouse_default"
        )
        m3 = ch.measure(w3)
        self.assertGreaterEqual(m3["seconds"], 0.0)
        # q01 uses uppercase SR_FEE which ClickHouse rejects with Code 47 UNKNOWN_IDENTIFIER.
        with self.assertRaises(harness.ClickHouseDialectError) as ctx:
            ch.query(harness.WORKLOADS["q01"])
        self.assertIn("Code: 47", str(ctx.exception))
        self.assertIn("SR_FEE", str(ctx.exception))

    def test_4way_audit_verify_bench_and_summarize(self):
        if self.runner is None:
            self.skipTest("MORSEL_TPCDS_RUNNER not set")

        # q03 (admitted: star join + agg + TopN), q01 (ClickHouse dialect-incompatible),
        # q12 (rejected by Morsel: Window), q05 (rejected by Morsel: ROLLUP + UNION).
        smoke_queries = ["q03", "q01", "q05", "q12"]
        audit = harness.audit_suite(
            self.data_dir,
            self.runner,
            threads=2,
            timeout=60,
            query_names=smoke_queries,
            verify_parity=True,
        )
        self.assertTrue(audit["queries"]["q03"]["admitted"])
        self.assertTrue(audit["queries"]["q03"]["parity_verified"])
        self.assertIsInstance(audit["queries"]["q03"]["degenerate_result"], bool)
        self.assertIn("degenerate_admitted_queries", audit["summary"])
        self.assertFalse(audit["queries"]["q05"]["admitted"])
        self.assertFalse(audit["queries"]["q12"]["admitted"])
        self.assertIn("Window", audit["queries"]["q12"]["all_blockers"])
        self.assertGreaterEqual(audit["summary"]["rejected_count"], 2)

        arms = (
            harness.DEFAULT_4WAY_ARMS
            if self.ch_bin is not None
            else (
                "stock_default",
                "stock_pushdown",
                "morsel_pushdown",
                "duckdb_default",
            )
        )
        ver = harness.verify_suite(
            self.data_dir,
            self.runner,
            threads=2,
            timeout=60,
            query_names=["q03", "q01", "q12"],
            arms=arms,
            clickhouse_bin=self.ch_bin,
        )
        self.assertIn("morsel_pushdown", ver["verified"]["q03"]["arms"])
        self.assertIsInstance(ver["verified"]["q03"]["degenerate_result"], bool)
        self.assertIn("degenerate_admitted_queries", ver)
        self.assertNotIn("morsel_pushdown", ver["verified"]["q12"]["arms"])
        if self.ch_bin is not None:
            self.assertIn("clickhouse_default", ver["verified"]["q03"]["arms"])
            self.assertIn("q01", ver["clickhouse_incompatible"])

        # Run CLI bench + summarize round-trip on q03 and q12.
        jsonl_path = Path(self.tmp.name) / "smoke.jsonl"
        cmd = [
            sys.executable,
            "-B",
            str(Path(harness.__file__).resolve()),
            "bench",
            str(self.data_dir),
            "--runner",
            str(self.runner),
            "--revision",
            _runner_revision(),
            "--threads",
            "2",
            "--timeout",
            "60",
            "--cache",
            "none",
            "--warmups",
            "0",
            "--repeats",
            "1",
            "--queries",
            "q03,q12",
            "--arms",
            ",".join(arms),
        ]
        if self.ch_bin is not None:
            cmd.extend(["--clickhouse-bin", str(self.ch_bin)])
        res = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=120
        )
        self.assertEqual(res.returncode, 0, res.stderr)
        jsonl_path.write_text(res.stdout)
        records = [json.loads(line) for line in res.stdout.splitlines() if line.strip()]
        self.assertEqual(records[0]["record_type"], "header")
        trials = [r for r in records[1:] if r["record_type"] == "trial"]
        self.assertTrue(
            any(t["case"] == "q03" and t["arm"] == "morsel_pushdown" for t in trials)
        )
        self.assertFalse(
            any(t["case"] == "q12" and t["arm"] == "morsel_pushdown" for t in trials)
        )

        summary_md = harness.summarize_jsonl(jsonl_path)
        self.assertIn("Morsel Admission Summary", summary_md)
        self.assertIn("Admitted Subset 4-Way Comparison", summary_md)
        self.assertIn("`q03`", summary_md)
        self.assertIn("`q12`", summary_md)


if __name__ == "__main__":
    unittest.main()
