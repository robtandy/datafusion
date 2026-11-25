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
    evict_files,
    generate,
    infer_scale_factor,
    render_workload_sql,
    require_cold_cache,
    require_warm_cache,
    resolve_arms,
    resolve_queries,
    summarize_jsonl,
    table_files,
    validate_plan,
    warm_files,
)


def _linux_fstype(path: Path) -> str | None:
    mounts = Path("/proc/mounts")
    if not mounts.is_file():
        return None
    resolved = path.resolve()
    best_len = -1
    best_fstype = None
    for line in mounts.read_text().splitlines():
        parts = line.split()
        if len(parts) < 3:
            continue
        mount_point = Path(parts[1].replace("\\040", " "))
        try:
            resolved.relative_to(mount_point)
        except ValueError:
            continue
        mp_len = len(mount_point.as_posix())
        if mp_len > best_len:
            best_len = mp_len
            best_fstype = parts[2]
    return best_fstype


def _find_disk_temp_parent() -> Path | None:
    candidates: list[Path] = []
    env_dir = os.getenv("MORSEL_TPCH_DISK_TMPDIR")
    if env_dir:
        candidates.append(Path(env_dir))
    candidates.extend(
        [
            Path(tempfile.gettempdir()),
            Path.home() / ".cache",
            Path(__file__).resolve().parent,
        ]
    )
    for cand in candidates:
        if not cand.is_dir() or not os.access(cand, os.W_OK | os.X_OK):
            continue
        fstype = _linux_fstype(cand)
        if fstype is None or fstype in ("tmpfs", "ramfs"):
            continue
        return cand
    return None


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

    def test_validate_plan_handles_multi_table_join_plans(self):
        join_ev = {
            "plan": (
                "MorselExec: workers=4, [Build: NativeParquet[part] -> HashJoinBuildSink]; "
                "NativeParquet[lineitem] -> Filter[...] -> HashJoinProbe[...] -> "
                "AggregateSink[aggr=[sum(...)]]; FinalizedAggregateSource"
            ),
            "parquet_morsels_claimed": 18,
            "count_finalizations": 0,
            "aggregate_finalizations": 1,
            "join_build_finalizations": 1,
            "sort_finalizations": 0,
            "topn_finalizations": 0,
        }
        rgs = {"lineitem": 16, "part": 2}

        validate_plan(join_ev, ARMS["morsel_no_pushdown"], WORKLOADS["q14"], rgs)
        validate_plan(join_ev, ARMS["morsel_no_pushdown"], WORKLOADS["q19"], rgs)
        validate_plan(
            dict(join_ev, parquet_morsels_claimed=10),
            ARMS["morsel_pruning"],
            WORKLOADS["q14"],
            rgs,
        )
        validate_plan(
            dict(join_ev, parquet_morsels_claimed=18),
            ARMS["morsel_pushdown"],
            WORKLOADS["q19"],
            rgs,
        )

        # Missing build-side NativeParquet[part] must fail.
        missing_part = dict(
            join_ev,
            plan="MorselExec: workers=4, NativeParquet[lineitem] -> AggregateSink; FinalizedAggregateSource",
        )
        with self.assertRaisesRegex(ValueError, "missing"):
            validate_plan(
                missing_part, ARMS["morsel_no_pushdown"], WORKLOADS["q14"], rgs
            )

        # Mismatched join_build_finalizations must fail.
        with self.assertRaisesRegex(ValueError, "join_build_finalizations"):
            validate_plan(
                dict(join_ev, join_build_finalizations=0),
                ARMS["morsel_no_pushdown"],
                WORKLOADS["q14"],
                rgs,
            )

        # Verify SortSink and TopNSink validation (Checkpoint B q1, q3, q12).
        q1_ev = {
            "plan": (
                "MorselExec: workers=4, NativeParquet[lineitem] -> AggregateSink; "
                "FinalizedAggregateSource -> SortSink; FinalizedSortSource"
            ),
            "parquet_morsels_claimed": 16,
            "count_finalizations": 0,
            "aggregate_finalizations": 1,
            "join_build_finalizations": 0,
            "sort_finalizations": 1,
            "topn_finalizations": 0,
        }
        validate_plan(q1_ev, ARMS["morsel_no_pushdown"], WORKLOADS["q1"], rgs)
        with self.assertRaisesRegex(ValueError, "SortSink"):
            validate_plan(
                dict(q1_ev, sort_finalizations=0),
                ARMS["morsel_no_pushdown"],
                WORKLOADS["q1"],
                rgs,
            )

        q3_rgs = {"customer": 2, "orders": 8, "lineitem": 16}
        q3_ev = {
            "plan": (
                "MorselExec: workers=4, [Build: NativeParquet[customer] -> HashJoinBuildSink]; "
                "[Build: NativeParquet[orders] -> HashJoinProbe -> HashJoinBuildSink]; "
                "NativeParquet[lineitem] -> HashJoinProbe -> AggregateSink; "
                "FinalizedAggregateSource -> TopNSink[limit=10]; FinalizedTopNSource"
            ),
            "parquet_morsels_claimed": 26,
            "count_finalizations": 0,
            "aggregate_finalizations": 1,
            "join_build_finalizations": 2,
            "sort_finalizations": 0,
            "topn_finalizations": 1,
        }
        validate_plan(q3_ev, ARMS["morsel_no_pushdown"], WORKLOADS["q3"], q3_rgs)
        with self.assertRaisesRegex(ValueError, "TopNSink"):
            validate_plan(
                dict(q3_ev, topn_finalizations=0),
                ARMS["morsel_no_pushdown"],
                WORKLOADS["q3"],
                q3_rgs,
            )

        # Under no_pushdown, claims must equal the sum across all scanned tables (16 + 2 = 18).
        for bad_claims in (16, 17, 19):
            with self.subTest(bad_claims=bad_claims), self.assertRaises(ValueError):
                validate_plan(
                    dict(join_ev, parquet_morsels_claimed=bad_claims),
                    ARMS["morsel_no_pushdown"],
                    WORKLOADS["q14"],
                    rgs,
                )

        # Under pruning/pushdown, claims must be in 1..=sum.
        for bad_claims in (0, 19):
            with self.subTest(pruning_bad=bad_claims), self.assertRaises(ValueError):
                validate_plan(
                    dict(join_ev, parquet_morsels_claimed=bad_claims),
                    ARMS["morsel_pruning"],
                    WORKLOADS["q14"],
                    rgs,
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
                {
                    "record_type": "trial",
                    "case": "q3",
                    "arm": "morsel_no_pushdown",
                    "seconds": 0.15,
                    "exec_seconds": 0.13,
                    "output_rows": 10,
                    "output_batches": 1,
                    "parquet_morsels_claimed": 66,
                    "morsel_decode_work_ms": 120.0,
                    "morsel_process_work_ms": 40.0,
                    "morsel_ready_wait_ms": 5.0,
                    "morsel_output_wait_ms": 0.0,
                    "morsel_pending_io_ms": 15.0,
                    "morsel_peak_ready_depth": 8,
                    "morsel_peak_output_depth": 1,
                    "morsel_pipelines": {
                        "build_0": {
                            "worker_claims": [1, 1],
                            "worker_completions": [1, 1],
                            "decode_work_ms": 10.0,
                            "process_work_ms": 5.0,
                            "ready_wait_ms": 1.0,
                            "output_wait_ms": 0.0,
                            "join_combine_work_ms": 0.5,
                            "join_finalize_work_ms": 1.5,
                            "sort_finalize_work_ms": 0.0,
                            "topn_finalize_work_ms": 0.0,
                        },
                        "topn": {
                            "worker_claims": [1, 0],
                            "worker_completions": [1, 0],
                            "decode_work_ms": 0.0,
                            "process_work_ms": 2.0,
                            "ready_wait_ms": 0.0,
                            "output_wait_ms": 0.0,
                            "join_combine_work_ms": 0.0,
                            "join_finalize_work_ms": 0.0,
                            "sort_finalize_work_ms": 0.0,
                            "topn_finalize_work_ms": 0.2,
                        },
                    },
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
            self.assertIn("| `q3` | `morsel_no_pushdown` | `build_0` |", out)
            self.assertIn("0.5/1.5", out)
            self.assertIn("| `q3` | `morsel_no_pushdown` | `topn` |", out)

    def test_require_warm_cache_rejects_partial_residency(self):
        with patch("harness.cache_pages", return_value={"resident": 1, "total": 2}):
            with self.assertRaisesRegex(ValueError, "cannot verify warm"):
                require_warm_cache([])

    def test_require_cold_cache_rejects_resident_pages(self):
        with patch("harness.cache_pages", return_value={"resident": 1, "total": 2}):
            with self.assertRaisesRegex(ValueError, "cannot verify cold"):
                require_cold_cache([])
        with patch("harness.cache_pages", return_value={"resident": 0, "total": 2}):
            self.assertEqual(require_cold_cache([]), {"resident": 0, "total": 2})

    @unittest.skipUnless(
        sys.platform == "linux",
        "cold-cache eviction via posix_fadvise/mincore requires Linux",
    )
    def test_evict_files_and_require_cold_cache_on_disk(self):
        disk_parent = _find_disk_temp_parent()
        if disk_parent is None:
            self.skipTest("no non-tmpfs writable directory available")
        with tempfile.TemporaryDirectory(
            dir=disk_parent, prefix="morsel_tpch_cold_"
        ) as tmp:
            sample = Path(tmp) / "page_cache_probe.bin"
            sample.write_bytes(b"a" * (256 * 1024))
            warm_files([sample])
            warm_pages = require_warm_cache([sample])
            self.assertGreater(warm_pages["total"], 0)
            self.assertEqual(warm_pages["resident"], warm_pages["total"])

            evict_files([sample])
            cold_pages = require_cold_cache([sample])
            self.assertEqual(cold_pages, {"resident": 0, "total": warm_pages["total"]})

    @unittest.skipUnless(
        sys.platform == "linux",
        "cold-cache duckdb-profile verification requires Linux",
    )
    def test_sf001_duckdb_profile_cold_cache(self):
        disk_parent = _find_disk_temp_parent()
        if disk_parent is None:
            self.skipTest("no non-tmpfs writable directory available")
        with tempfile.TemporaryDirectory(
            dir=disk_parent, prefix="morsel_tpch_dprof_cold_"
        ) as tmp:
            data_dir = Path(tmp) / "tpch sf001 cold"
            generate(data_dir, sf=0.01, row_group_size=131_072)
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
                    "cold",
                    "--warmups",
                    "0",
                    "--repeats",
                    "1",
                    "--queries",
                    "q6",
                ],
                text=True,
                timeout=120,
            )
            report = json.loads(out)
            self.assertEqual(report["cache_regime"], "cold")
            self.assertEqual(report["queries"]["q6"]["output_rows"], 1)
            self.assertIn(
                "TABLE_SCAN", report["queries"]["q6"]["operator_timing_ms_median"]
            )

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
            self.assertEqual(len(expected_rejected), 14)
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
                morsel_ev[("q1", "morsel_no_pushdown")]["aggregate_finalizations"], 1
            )
            self.assertEqual(
                morsel_ev[("q1", "morsel_no_pushdown")]["sort_finalizations"], 1
            )
            self.assertEqual(
                morsel_ev[("q3", "morsel_no_pushdown")]["parquet_morsels_claimed"], 26
            )
            self.assertEqual(
                morsel_ev[("q3", "morsel_no_pushdown")]["join_build_finalizations"], 2
            )
            self.assertEqual(
                morsel_ev[("q3", "morsel_no_pushdown")]["topn_finalizations"], 1
            )
            self.assertEqual(
                morsel_ev[("q6", "morsel_no_pushdown")]["aggregate_finalizations"],
                1,
            )
            self.assertEqual(
                morsel_ev[("q12", "morsel_no_pushdown")]["parquet_morsels_claimed"], 24
            )
            self.assertEqual(
                morsel_ev[("q12", "morsel_no_pushdown")]["join_build_finalizations"], 1
            )
            self.assertEqual(
                morsel_ev[("q12", "morsel_no_pushdown")]["sort_finalizations"], 1
            )
            for join_q in ("q14", "q19"):
                self.assertEqual(
                    morsel_ev[(join_q, "morsel_no_pushdown")][
                        "parquet_morsels_claimed"
                    ],
                    18,
                )
                self.assertEqual(
                    morsel_ev[(join_q, "morsel_no_pushdown")][
                        "aggregate_finalizations"
                    ],
                    1,
                )
                self.assertEqual(
                    morsel_ev[(join_q, "morsel_no_pushdown")][
                        "join_build_finalizations"
                    ],
                    1,
                )
                self.assertIn(
                    "HashJoinProbe", morsel_ev[(join_q, "morsel_no_pushdown")]["plan"]
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
                    "q2",
                    "q1",
                    "q3",
                    "q6",
                    "q12",
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
            self.assertIn("q2", header["morsel_rejections"])
            # 2 arms * 2 reps for q2 + 4 arms * 2 reps for (q1, q3, q6, q12, q6_count, q6_scan) = 52
            self.assertEqual(len(trials), 52)
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
                    "morsel_join_combine_work_ms",
                    "morsel_join_finalize_work_ms",
                    "morsel_sort_finalize_work_ms",
                    "morsel_topn_finalize_work_ms",
                    "morsel_peak_ready_depth",
                    "morsel_peak_output_depth",
                )
                if trial["mode"] == "required":
                    for field in morsel_metric_fields:
                        self.assertIsNotNone(trial[field], msg=field)
                        self.assertGreaterEqual(trial[field], 0, msg=field)
                    self.assertEqual(len(trial["worker_claims"]), 2)
                    pipelines = trial["morsel_pipelines"]
                    self.assertIsInstance(pipelines, dict)
                    self.assertGreater(len(pipelines), 0)
                    parquet_pipe_claims = sum(
                        sum(p["worker_claims"])
                        for name, p in pipelines.items()
                        if name not in ("sort", "topn")
                    )
                    all_pipe_claims = sum(
                        sum(p["worker_claims"]) for p in pipelines.values()
                    )
                    self.assertEqual(
                        parquet_pipe_claims, trial["parquet_morsels_claimed"]
                    )
                    self.assertEqual(all_pipe_claims, sum(trial["worker_claims"]))
                    self.assertEqual(
                        sum(trial["worker_completions"]),
                        sum(trial["worker_claims"]),
                    )
                    if trial["case"] == "q1":
                        self.assertEqual(set(pipelines.keys()), {"main", "sort"})
                    elif trial["case"] == "q3":
                        self.assertEqual(
                            set(pipelines.keys()),
                            {"build_build", "build", "probe", "topn"},
                        )
                    elif trial["case"] == "q12":
                        self.assertEqual(
                            set(pipelines.keys()), {"build", "probe", "sort"}
                        )
                else:
                    for field in morsel_metric_fields:
                        self.assertIsNone(trial[field], msg=field)
                    self.assertIsNone(trial["worker_claims"])
                    self.assertIsNone(trial["worker_completions"])
                    self.assertIsNone(trial["morsel_pipelines"])

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
                "`q1`",
                "`q3`",
                "`q6`",
                "`q12`",
                "`q6_count`",
                "`q6_scan`",
                "`stock_no_pushdown`",
                "`stock_pushdown_no_repartition`",
                "`morsel_no_pushdown`",
                "`duckdb_default`",
                "Decode work ms",
                "Process work ms",
                "Pipeline",
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
                "morsel_join_combine_work_ms",
                "morsel_join_finalize_work_ms",
                "morsel_sort_finalize_work_ms",
                "morsel_topn_finalize_work_ms",
                "morsel_peak_ready_depth",
                "morsel_peak_output_depth",
                "morsel_pipelines",
            ):
                self.assertIsNone(sample[field], msg=field)

    @unittest.skipUnless(
        sys.platform == "linux"
        and os.getenv("MORSEL_TPCH_RUNNER")
        and os.getenv("MORSEL_BENCH_REVISION"),
        "set MORSEL_TPCH_RUNNER and MORSEL_BENCH_REVISION for the Linux cold-cache bench smoke test",
    )
    def test_sf001_bench_cold_cache_smoke(self):
        disk_parent = _find_disk_temp_parent()
        if disk_parent is None:
            self.skipTest("no non-tmpfs writable directory available")
        with tempfile.TemporaryDirectory(
            dir=disk_parent, prefix="morsel_tpch_bench_cold_"
        ) as tmp:
            data_dir = Path(tmp) / "tpch sf001 bench cold"
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
                    "cold",
                    "--warmups",
                    "0",
                    "--repeats",
                    "1",
                    "--queries",
                    "q6",
                    "--morsel-arms",
                    "morsel_no_pushdown",
                    "duckdb_default",
                ],
                text=True,
                timeout=120,
            )
            records = [json.loads(line) for line in out.splitlines()]
            header, trials = records[0], records[1:]
            self.assertEqual(header["cache_regime"], "cold")
            self.assertEqual(len(trials), 2)
            for trial in trials:
                before = trial["cache_pages_before"]
                after = trial["cache_pages_after"]
                self.assertGreater(before["total"], 0)
                self.assertEqual(before["resident"], 0)
                self.assertGreater(after["resident"], 0)
