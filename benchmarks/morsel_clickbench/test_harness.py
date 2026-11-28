# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

from contextlib import redirect_stdout
from dataclasses import replace
import datetime
import io
import json
import os
from pathlib import Path
import platform
import subprocess
import tempfile
import unittest

import pyarrow as pa
import pyarrow.ipc as pa_ipc
import pyarrow.parquet as pq

from harness import (
    ALL_CLICKBENCH_QUERIES,
    ARMS,
    CLICKHOUSE_DIALECT_NOTES,
    DEFAULT_BENCH_ARMS,
    DEFAULT_CLICKHOUSE_DEST,
    DUCKDB_DIALECT_NOTES,
    HITS_SCHEMA_FIELDS,
    QUERIES,
    ClickHouseRunner,
    DataFusionRunner,
    DuckDbOracle,
    DuckDbRunner,
    QuerySpec,
    adapt_sql_for_clickhouse,
    adapt_sql_for_duckdb,
    categorize_morsel_rejection,
    compare_tables,
    dataset_files,
    dataset_summary,
    evict_files,
    generate_smoke,
    group_morsel_rejections,
    hits_arrow_schema,
    probe_morsel_matrix,
    require_cold_cache,
    require_warm_cache,
    resolve_arms,
    resolve_queries,
    strip_order_limit,
    summarize_jsonl,
    validate_plan,
    verify_suite,
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
    env_dir = os.getenv("MORSEL_CLICKBENCH_DISK_TMPDIR")
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


class ClickBenchUnitTests(unittest.TestCase):
    def test_load_all_43_queries(self):
        self.assertEqual(len(ALL_CLICKBENCH_QUERIES), 43)
        self.assertEqual(len(QUERIES), 43)
        for idx in range(43):
            qname = f"q{idx}"
            spec = QUERIES[qname]
            self.assertEqual(spec.name, qname)
            self.assertEqual(spec.index, idx)
            self.assertTrue(spec.sql.upper().startswith("SELECT"))
            self.assertFalse(spec.sql.endswith(";"))

    def test_resolve_queries_and_arms(self):
        self.assertEqual(resolve_queries(None), list(ALL_CLICKBENCH_QUERIES))
        self.assertEqual(resolve_queries(["all"]), list(ALL_CLICKBENCH_QUERIES))
        self.assertEqual(resolve_queries(["q0-q3,q7"]), ["q0", "q1", "q2", "q3", "q7"])
        with self.assertRaises(ValueError):
            resolve_queries(["q99"])
        with self.assertRaises(ValueError):
            resolve_queries(["q10-q5"])

        self.assertEqual(
            resolve_arms(None, DEFAULT_BENCH_ARMS), list(DEFAULT_BENCH_ARMS)
        )
        self.assertEqual(
            resolve_arms(["morsel_pushdown,duckdb_default"], DEFAULT_BENCH_ARMS),
            ["morsel_pushdown", "duckdb_default"],
        )
        with self.assertRaises(ValueError):
            resolve_arms(["nonexistent_arm"], DEFAULT_BENCH_ARMS)

    def test_dialect_adaptations_match_documented_queries(self):
        for qname, spec in QUERIES.items():
            dk = adapt_sql_for_duckdb(qname, spec.sql)
            ch = adapt_sql_for_clickhouse(qname, spec.sql)
            if qname in DUCKDB_DIALECT_NOTES:
                self.assertNotEqual(
                    dk, spec.sql, f"expected DuckDB adaptation for {qname}"
                )
            else:
                self.assertEqual(
                    dk, spec.sql, f"unexpected DuckDB adaptation for {qname}"
                )
            if qname in CLICKHOUSE_DIALECT_NOTES:
                self.assertNotEqual(
                    ch, spec.sql, f"expected ClickHouse adaptation for {qname}"
                )
            else:
                self.assertEqual(
                    ch, spec.sql, f"unexpected ClickHouse adaptation for {qname}"
                )
        self.assertIn("(?-s)", adapt_sql_for_clickhouse("q28", QUERIES["q28"].sql))

    def test_generate_smoke_schema_and_determinism(self):
        with tempfile.TemporaryDirectory() as tmp:
            d1 = Path(tmp) / "smoke1"
            d2 = Path(tmp) / "smoke2"
            s1 = generate_smoke(d1, rows=2000, files=4, row_group_size=500, seed=42)
            s2 = generate_smoke(d2, rows=2000, files=4, row_group_size=500, seed=42)
            self.assertEqual(s1["files"], 4)
            self.assertEqual(s1["rows"], 2000)
            self.assertEqual(s1["columns"], 105)
            self.assertEqual(s1["parquet_bytes"], s2["parquet_bytes"])
            self.assertEqual(len(HITS_SCHEMA_FIELDS), 105)

            f1 = dataset_files(d1)
            f2 = dataset_files(d2)
            self.assertEqual([p.name for p in f1], [p.name for p in f2])
            for p1, p2 in zip(f1, f2):
                self.assertEqual(p1.read_bytes(), p2.read_bytes())

            actual_schema = pq.read_schema(f1[0])
            expected_schema = hits_arrow_schema()
            self.assertEqual(actual_schema.names, expected_schema.names)
            self.assertEqual(actual_schema.types, expected_schema.types)
            self.assertEqual(actual_schema.field("EventDate").type, pa.uint16())
            self.assertEqual(actual_schema.field("URL").type, pa.binary())

    def test_compare_tables_semantics(self):
        spec_exact = QuerySpec("q_exact", 0, "SELECT 1", "exact")
        t1 = pa.table(
            {
                "c0": pa.array([1, 2], type=pa.int64()),
                "c1": pa.array([1.0, 2.5], type=pa.float64()),
                "c2": pa.array(
                    [datetime.datetime(2013, 7, 14, 1, 2), None],
                    type=pa.timestamp("s"),
                ),
            }
        )
        t2 = pa.table(
            {
                "count()": pa.array([1, 2], type=pa.uint64()),
                "avg()": pa.array([1.0 + 1e-12, 2.5], type=pa.float64()),
                "ts": pa.array(
                    [
                        datetime.datetime(
                            2013, 7, 14, 1, 2, tzinfo=datetime.timezone.utc
                        ),
                        None,
                    ],
                    type=pa.timestamp("s", tz="UTC"),
                ),
            }
        )
        compare_tables(t1, t2, spec_exact, "clickhouse_default")

        # Empty stream parity (ClickHouse writes 0-column empty table on 0 rows).
        empty_oracle = pa.table({"a": pa.array([], type=pa.int64())})
        empty_ch = pa.table({})
        compare_tables(empty_oracle, empty_ch, spec_exact, "clickhouse_default")

        # TopN tie handling: interior ties must match as multiset; boundary ties
        # require oracle_lookup verification when cut by LIMIT (n == limit).
        spec_topn = QuerySpec(
            "q_topn",
            12,
            "SELECT k, c, s FROM t GROUP BY k ORDER BY c DESC LIMIT 4",
            "topn",
            sort_key_cols=(1,),
            limit=4,
            group_key_cols=(0,),
        )
        oracle_topn = pa.table(
            {"k": ["a", "b", "c", "d"], "c": [10, 5, 5, 2], "s": [100, 50, 51, 20]}
        )
        oracle_full = pa.table(
            {
                "k": ["a", "b", "c", "d", "z"],
                "c": [10, 5, 5, 2, 2],
                "s": [100, 50, 51, 20, 29],
            }
        )
        actual_valid_tie = pa.table(
            {"k": ["a", "c", "b", "z"], "c": [10, 5, 5, 2], "s": [100, 51, 50, 29]}
        )
        compare_tables(
            oracle_topn,
            actual_valid_tie,
            spec_topn,
            "test_arm",
            oracle_lookup=oracle_full,
        )

        # Without oracle_lookup, boundary mismatch must fail (never silently skip).
        with self.assertRaises(ValueError):
            compare_tables(oracle_topn, actual_valid_tie, spec_topn, "test_arm")

        # Made-up group key at boundary with matching sort key must fail.
        made_up_boundary = pa.table(
            {"k": ["a", "b", "c", "FAKE"], "c": [10, 5, 5, 2], "s": [100, 50, 51, 20]}
        )
        with self.assertRaises(ValueError):
            compare_tables(
                oracle_topn,
                made_up_boundary,
                spec_topn,
                "test_arm",
                oracle_lookup=oracle_full,
            )

        # Valid group key at boundary with wrong non-sort aggregate must fail.
        wrong_agg_boundary = pa.table(
            {"k": ["a", "b", "c", "z"], "c": [10, 5, 5, 2], "s": [100, 50, 51, 999]}
        )
        with self.assertRaises(ValueError):
            compare_tables(
                oracle_topn,
                wrong_agg_boundary,
                spec_topn,
                "test_arm",
                oracle_lookup=oracle_full,
            )

        # Duplicate group key at boundary (repeating a valid key) must fail.
        spec_topn_dup = replace(spec_topn, limit=5)
        oracle_dup = pa.table(
            {
                "k": ["a", "b", "c", "d", "z"],
                "c": [10, 5, 5, 2, 2],
                "s": [100, 50, 51, 20, 29],
            }
        )
        actual_dup = pa.table(
            {
                "k": ["a", "b", "c", "d", "d"],
                "c": [10, 5, 5, 2, 2],
                "s": [100, 50, 51, 20, 20],
            }
        )
        with self.assertRaises(ValueError):
            compare_tables(
                oracle_dup,
                actual_dup,
                spec_topn_dup,
                "test_arm",
                oracle_lookup=oracle_full,
            )

        # When result was NOT cut by LIMIT (num_rows < limit), final group is NOT a
        # boundary: even a key present in oracle_full must fail exact multiset check.
        spec_uncut = QuerySpec(
            "q21",
            21,
            "SELECT k, c, s FROM t GROUP BY k ORDER BY c DESC LIMIT 10",
            "topn",
            sort_key_cols=(1,),
            limit=10,
            group_key_cols=(0,),
        )
        oracle_uncut = pa.table({"k": ["a"], "c": [5], "s": [50]})
        actual_uncut_fake = pa.table({"k": ["z"], "c": [5], "s": [50]})
        with self.assertRaises(ValueError):
            compare_tables(
                oracle_uncut,
                actual_uncut_fake,
                spec_uncut,
                "test_arm",
                oracle_lookup=pa.table({"k": ["a", "z"], "c": [5, 5], "s": [50, 50]}),
            )

        # Interior mismatch must fail.
        bad_interior = pa.table(
            {"k": ["a", "x", "b", "d"], "c": [10, 5, 5, 2], "s": [100, 50, 51, 20]}
        )
        with self.assertRaises(ValueError):
            compare_tables(
                oracle_topn,
                bad_interior,
                spec_topn,
                "test_arm",
                oracle_lookup=oracle_full,
            )

        # Sort-key mismatch at boundary must fail.
        bad_key = pa.table(
            {"k": ["a", "b", "c", "d"], "c": [10, 5, 5, 1], "s": [100, 50, 51, 20]}
        )
        with self.assertRaises(ValueError):
            compare_tables(
                oracle_topn,
                bad_key,
                spec_topn,
                "test_arm",
                oracle_lookup=oracle_full,
            )

    def test_compare_tables_rejects_made_up_q17_q23_q24(self):
        # q17 (limit_unordered): must verify existence, aggregate count, and uniqueness.
        spec_q17 = QuerySpec(
            "q17",
            17,
            'SELECT "UserID", "SearchPhrase", COUNT(*) FROM hits GROUP BY "UserID", "SearchPhrase" LIMIT 2',
            "limit_unordered",
            limit=2,
            group_key_cols=(0, 1),
        )
        oracle_q17 = pa.table({"u": [1, 2], "p": ["a", "b"], "c": [3, 4]})
        full_q17 = pa.table({"u": [1, 2, 3], "p": ["a", "b", "c"], "c": [3, 4, 5]})
        compare_tables(
            oracle_q17,
            pa.table({"u": [3, 1], "p": ["c", "a"], "c": [5, 3]}),
            spec_q17,
            "test_arm",
            oracle_lookup=full_q17,
        )
        # Made-up group with positive count must fail.
        with self.assertRaises(ValueError):
            compare_tables(
                oracle_q17,
                pa.table({"u": [999, 1], "p": ["fake", "a"], "c": [1, 3]}),
                spec_q17,
                "test_arm",
                oracle_lookup=full_q17,
            )
        # Wrong count on valid group must fail.
        with self.assertRaises(ValueError):
            compare_tables(
                oracle_q17,
                pa.table({"u": [3, 1], "p": ["c", "a"], "c": [99, 3]}),
                spec_q17,
                "test_arm",
                oracle_lookup=full_q17,
            )
        # Duplicate valid group must fail.
        with self.assertRaises(ValueError):
            compare_tables(
                oracle_q17,
                pa.table({"u": [1, 1], "p": ["a", "a"], "c": [3, 3]}),
                spec_q17,
                "test_arm",
                oracle_lookup=full_q17,
            )

        # q23 (non-GROUP-BY topn boundary tie): multiset subset check rejects made-up/over-duplicated rows.
        spec_q23 = QuerySpec(
            "q23",
            23,
            "SELECT id, ts FROM hits ORDER BY ts LIMIT 2",
            "topn",
            sort_key_cols=(1,),
            limit=2,
            group_key_cols=(),
        )
        oracle_q23 = pa.table({"id": [1, 2], "ts": [10, 20]})
        cand_q23 = pa.table({"id": [2, 3], "ts": [20, 20]})
        compare_tables(
            oracle_q23,
            pa.table({"id": [1, 3], "ts": [10, 20]}),
            spec_q23,
            "test_arm",
            oracle_lookup=cand_q23,
        )
        with self.assertRaises(ValueError):
            compare_tables(
                oracle_q23,
                pa.table({"id": [1, 999], "ts": [10, 20]}),
                spec_q23,
                "test_arm",
                oracle_lookup=cand_q23,
            )

        # q24 (topn_hidden_key): strict EventTime prefix + cutoff pool check.
        spec_q24 = QuerySpec(
            "q24",
            24,
            'SELECT "SearchPhrase" FROM hits WHERE "SearchPhrase" <> \'\' ORDER BY "EventTime" LIMIT 4',
            "topn_hidden_key",
            limit=4,
        )
        oracle_q24_res = pa.table({"SearchPhrase": ["p1", "p2", "p3", "p4"]})
        q24_pool = pa.table(
            {
                "EventTime": [100, 100, 200, 300, 300],
                "SearchPhrase": ["p1", "p2", "p3", "p4", "p5"],
            }
        )
        # Reordered within EventTime=100 tie + alternative EventTime=300 cutoff row passes.
        compare_tables(
            oracle_q24_res,
            pa.table({"SearchPhrase": ["p2", "p1", "p3", "p5"]}),
            spec_q24,
            "test_arm",
            oracle_q24=q24_pool,
        )
        # Made-up non-empty string at cutoff must fail.
        with self.assertRaises(ValueError):
            compare_tables(
                oracle_q24_res,
                pa.table({"SearchPhrase": ["p1", "p2", "p3", "MADE_UP"]}),
                spec_q24,
                "test_arm",
                oracle_q24=q24_pool,
            )
        # Missing strict-prefix row (EventTime=200 'p3' omitted) must fail.
        with self.assertRaises(ValueError):
            compare_tables(
                oracle_q24_res,
                pa.table({"SearchPhrase": ["p1", "p2", "p4", "p5"]}),
                spec_q24,
                "test_arm",
                oracle_q24=q24_pool,
            )
        # Out-of-order strict prefix (swapping EventTime=200 before EventTime=100) must fail.
        with self.assertRaises(ValueError):
            compare_tables(
                oracle_q24_res,
                pa.table({"SearchPhrase": ["p3", "p1", "p2", "p4"]}),
                spec_q24,
                "test_arm",
                oracle_q24=q24_pool,
            )
        # Over-duplicating a boundary phrase beyond pool multiplicity must fail.
        spec_q24_5 = replace(spec_q24, limit=5)
        q24_pool_extra = pa.table(
            {
                "EventTime": [100, 100, 200, 300, 300, 300],
                "SearchPhrase": ["p1", "p2", "p3", "p4", "p5", "p6"],
            }
        )
        with self.assertRaises(ValueError):
            compare_tables(
                pa.table({"SearchPhrase": ["p1", "p2", "p3", "p4", "p5"]}),
                pa.table({"SearchPhrase": ["p1", "p2", "p3", "p4", "p4"]}),
                spec_q24_5,
                "test_arm",
                oracle_q24=q24_pool_extra,
            )

        # Verify strip_order_limit cleanly removes ORDER BY / LIMIT / OFFSET.
        self.assertEqual(
            strip_order_limit(QUERIES["q17"].sql),
            'SELECT "UserID", "SearchPhrase", COUNT(*) FROM hits GROUP BY "UserID", "SearchPhrase"',
        )
        self.assertNotIn("LIMIT", strip_order_limit(QUERIES["q38"].sql))
        self.assertNotIn("OFFSET", strip_order_limit(QUERIES["q38"].sql))

    def test_rejection_grouping(self):
        rejections = {
            "q4": "morsel_mode=required: unsupported DISTINCT aggregate",
            "q5": "morsel_mode=required: unsupported DISTINCT aggregate",
            "q38": "morsel_mode=required: GlobalLimitExec with skip/offset not supported",
        }
        groups = group_morsel_rejections(rejections)
        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["count"], 2)
        self.assertEqual(groups[0]["queries"], ["q4", "q5"])
        self.assertEqual(groups[1]["count"], 1)
        self.assertEqual(groups[1]["queries"], ["q38"])
        self.assertIn("DISTINCT", categorize_morsel_rejection(rejections["q4"]))

    @unittest.skipUnless(platform.system() == "Linux", "requires Linux mincore")
    def test_warm_and_cold_cache_mincore(self):
        parent = _find_disk_temp_parent()
        if parent is None:
            self.skipTest("no disk-backed temp directory available for cold eviction")
        with tempfile.TemporaryDirectory(dir=parent) as tmp:
            data_dir = Path(tmp) / "smoke"
            generate_smoke(data_dir, rows=4000, files=2, row_group_size=2000, seed=7)
            files = dataset_files(data_dir)
            warm_files(files)
            warm = require_warm_cache(files)
            self.assertGreater(warm["total"], 0)
            self.assertEqual(warm["resident"], warm["total"])

            evict_files(files)
            cold = require_cold_cache(files)
            self.assertEqual(cold["resident"], 0)

    def test_summarize_jsonl_renders_4way_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            jsonl = Path(tmp) / "run.jsonl"
            header = {
                "record_type": "header",
                "host": "test-host",
                "arch": "aarch64",
                "kernel": "7.0.0",
                "threads": 8,
                "cache_regime": "warm",
                "initial_cache_pages": {"resident": 100, "total": 100},
                "warmups": 1,
                "repeats": 3,
                "build_revision": "deadbeef",
                "duckdb_version": "2.0.1",
                "pyarrow_version": "26.0.1",
                "clickhouse_version": "26.11.5.8",
                "dataset": {
                    "files": 4,
                    "rows": 50000,
                    "row_groups": 4,
                    "parquet_bytes": 123456,
                },
                "arms": list(DEFAULT_BENCH_ARMS),
                "queries": ["q0", "q4"],
                "morsel_admitted": ["q0"],
                "morsel_rejections": {
                    "q4": "morsel_mode=required: unsupported DISTINCT aggregate"
                },
                "grouped_rejections": group_morsel_rejections(
                    {"q4": "morsel_mode=required: unsupported DISTINCT aggregate"}
                ),
            }
            trials = []
            for q, arm, secs in [
                ("q0", "morsel_pushdown", 0.010),
                ("q0", "stock_default", 0.020),
                ("q0", "stock_pushdown", 0.018),
                ("q0", "duckdb_default", 0.012),
                ("q0", "clickhouse_default", 0.015),
                ("q4", "stock_default", 0.050),
                ("q4", "stock_pushdown", 0.045),
                ("q4", "duckdb_default", 0.030),
                ("q4", "clickhouse_default", 0.025),
            ]:
                trials.append(
                    {
                        "record_type": "trial",
                        "case": q,
                        "arm": arm,
                        "rep": 1,
                        "seconds": secs,
                        "exec_seconds": secs * 0.8,
                        "output_rows": 1,
                    }
                )
            jsonl.write_text("\n".join(json.dumps(r) for r in [header, *trials]) + "\n")
            buf = io.StringIO()
            with redirect_stdout(buf):
                summarize_jsonl(jsonl)
            out = buf.getvalue()
            self.assertIn("GeoMean (1/2 Morsel-admitted)", out)
            self.assertIn("GeoMean (2/2 all queries)", out)
            self.assertIn(
                "Grouped Morsel (`morsel_mode=required`) Rejection Summary", out
            )
            self.assertIn("DISTINCT / COUNT(DISTINCT) aggregation", out)


class ClickBenchEndToEndSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        runner_env = os.getenv("MORSEL_CLICKBENCH_RUNNER")
        ch_env = os.getenv("MORSEL_CLICKBENCH_CLICKHOUSE")
        cls.runner = Path(runner_env) if runner_env else None
        cls.clickhouse = (
            Path(ch_env)
            if ch_env
            else (
                DEFAULT_CLICKHOUSE_DEST if DEFAULT_CLICKHOUSE_DEST.is_file() else None
            )
        )
        cls._tmp = tempfile.TemporaryDirectory(prefix="morsel-cb-smoke-test-")
        cls.smoke_dir = Path(cls._tmp.name) / "hits_smoke"
        cls.smoke_summary = generate_smoke(
            cls.smoke_dir, rows=50_000, files=4, row_group_size=12_500
        )

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_duckdb_vs_clickhouse_all_43_queries(self):
        if self.clickhouse is None or not self.clickhouse.is_file():
            self.skipTest(
                "ClickHouse binary not available; set MORSEL_CLICKBENCH_CLICKHOUSE"
            )
        dk = DuckDbRunner(self.smoke_dir, threads=4, timeout=60)
        ch = ClickHouseRunner(self.clickhouse, self.smoke_dir, threads=4, timeout=60)
        with DuckDbOracle(self.smoke_dir, threads=4) as oracle_db:
            for qname in ALL_CLICKBENCH_QUERIES:
                spec = QUERIES[qname]
                oracle = dk.query(spec)
                actual = ch.query(spec)
                compare_tables(
                    oracle,
                    actual,
                    spec,
                    "clickhouse_default",
                    oracle_lookup=oracle_db.lookup_candidates,
                    oracle_q24=oracle_db.q24_oracle,
                )

    def test_scaled_threshold_queries_q27_q28_q41_and_q28_newline_sensitivity(self):
        if self.runner is None or not self.runner.is_file():
            self.skipTest("DataFusion runner not built; set MORSEL_CLICKBENCH_RUNNER")
        if self.clickhouse is None or not self.clickhouse.is_file():
            self.skipTest(
                "ClickHouse binary not available; set MORSEL_CLICKBENCH_CLICKHOUSE"
            )

        # Official q27/q28 (HAVING COUNT(*) > 100000) and q41 (OFFSET 10000) return
        # 0 rows on the 50K smoke fixture. Exercise fixture-sized thresholds through
        # all 5 arms to actively test DuckDB strlen, ClickHouse (?-s), and q41 OFFSET.
        scaled_specs = [
            replace(
                QUERIES["q27"],
                sql=QUERIES["q27"].sql.replace(
                    "HAVING COUNT(*) > 100000", "HAVING COUNT(*) > 100"
                ),
            ),
            replace(
                QUERIES["q28"],
                sql=QUERIES["q28"].sql.replace(
                    "HAVING COUNT(*) > 100000", "HAVING COUNT(*) > 100"
                ),
            ),
            replace(
                QUERIES["q41"],
                sql=QUERIES["q41"].sql.replace("OFFSET 10000", "OFFSET 100"),
            ),
        ]

        dk = DuckDbRunner(self.smoke_dir, threads=4, timeout=60)
        ch = ClickHouseRunner(self.clickhouse, self.smoke_dir, threads=4, timeout=60)
        runners = {
            "stock_default": DataFusionRunner(
                self.runner, self.smoke_dir, 4, 60, "off", "default"
            ),
            "stock_pushdown": DataFusionRunner(
                self.runner, self.smoke_dir, 4, 60, "off", "pushdown"
            ),
            "morsel_pushdown": DataFusionRunner(
                self.runner, self.smoke_dir, 4, 60, "required", "pushdown"
            ),
            "clickhouse_default": ch,
        }

        q28_oracle = None
        with DuckDbOracle(self.smoke_dir, threads=4) as oracle_db:
            for spec in scaled_specs:
                oracle = dk.query(spec)
                self.assertGreater(
                    oracle.num_rows,
                    0,
                    f"expected non-empty smoke rows for scaled {spec.name}",
                )
                if spec.name == "q28":
                    q28_oracle = oracle
                    # Verify newline-Referer groups appear with distinct AVG lengths (no tie).
                    ks = oracle.column(0).to_pylist()
                    ls = oracle.column(1).to_pylist()
                    nl_ls = [l for k, l in zip(ks, ls) if "nlhost" in str(k)]
                    self.assertGreaterEqual(len(nl_ls), 5)
                    self.assertEqual(len(set(round(v, 6) for v in nl_ls)), len(nl_ls))
                for arm_name, runner in runners.items():
                    actual = runner.query(spec)
                    compare_tables(
                        oracle,
                        actual,
                        spec,
                        arm_name,
                        oracle_lookup=oracle_db.lookup_candidates,
                    )

            # Negative check: running ClickHouse on scaled q28 WITHOUT the (?-s)
            # rewrite must produce different results and be rejected by compare_tables.
            q28_spec = scaled_specs[1]
            self.assertIsNotNone(q28_oracle)
            raw_sql = f"{ch._view_ddl()};\n{q28_spec.sql}"
            res = subprocess.run(
                [
                    str(self.clickhouse),
                    "local",
                    "--tmp",
                    "--max_threads=4",
                    "--format=ArrowStream",
                    "-q",
                    raw_sql,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=60,
                check=True,
            )
            ch_unadapted = pa_ipc.open_stream(res.stdout).read_all()
            with self.assertRaises(ValueError):
                compare_tables(
                    q28_oracle,
                    ch_unadapted,
                    q28_spec,
                    "clickhouse_unadapted_q28",
                    oracle_lookup=oracle_db.lookup_candidates,
                )

    def test_full_4way_parity_and_morsel_admission_all_43_queries(self):
        if self.runner is None or not self.runner.is_file():
            self.skipTest("DataFusion runner not built; set MORSEL_CLICKBENCH_RUNNER")
        if self.clickhouse is None or not self.clickhouse.is_file():
            self.skipTest(
                "ClickHouse binary not available; set MORSEL_CLICKBENCH_CLICKHOUSE"
            )

        report = verify_suite(
            self.smoke_dir,
            self.runner,
            self.clickhouse,
            threads=4,
            timeout=60,
            query_names=list(ALL_CLICKBENCH_QUERIES),
            arm_names=list(DEFAULT_BENCH_ARMS),
        )
        self.assertEqual(len(report["queries"]), 43)
        self.assertGreater(report["morsel_admitted_count"], 0)
        self.assertEqual(
            report["morsel_admitted_count"] + report["morsel_rejected_count"], 43
        )

        # Verify plan validation and measure JSON on an admitted query and stock arm.
        matrix = probe_morsel_matrix(
            self.smoke_dir,
            self.runner,
            threads=4,
            timeout=60,
            query_names=list(ALL_CLICKBENCH_QUERIES),
        )
        first_admitted = matrix["admitted"][0]
        spec = QUERIES[first_admitted]
        morsel = DataFusionRunner(
            self.runner, self.smoke_dir, 4, 60, mode="required", profile="pushdown"
        )
        m_res = morsel.measure(spec)
        validate_plan(m_res, ARMS["morsel_pushdown"], spec)
        self.assertGreater(m_res["parquet_morsels_claimed"], 0)

        stock = DataFusionRunner(
            self.runner, self.smoke_dir, 4, 60, mode="off", profile="default"
        )
        s_res = stock.measure(spec)
        validate_plan(s_res, ARMS["stock_default"], spec)
        self.assertEqual(s_res["parquet_morsels_claimed"], 0)

        # Confirm ClickHouse measure returns valid statement timings and matching row count.
        ch = ClickHouseRunner(self.clickhouse, self.smoke_dir, threads=4, timeout=60)
        ch_res = ch.measure(spec)
        self.assertEqual(ch_res["output_rows"], m_res["output_rows"])
        self.assertGreaterEqual(ch_res["seconds"], 0.0)

        # Confirm morsel_mode=required explicitly rejects an unsupported window query.
        window_spec = QuerySpec(
            "q_window_neg",
            999,
            'SELECT ROW_NUMBER() OVER (ORDER BY "EventTime") AS rn FROM hits LIMIT 5',
            "exact",
        )
        rej = morsel.plan(window_spec)
        self.assertFalse(rej["admitted"])
        self.assertTrue(rej["error"])
        self.assertEqual(categorize_morsel_rejection(rej["error"]), "Window operator")


if __name__ == "__main__":
    unittest.main()
