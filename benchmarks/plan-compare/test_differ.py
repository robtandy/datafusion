#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

"""Tests of `differ.py`: the expressions, the normalization of both plans, the
comparison and its classification on small plans, and the comparison of the
captured plans of the 47 queries with `docs/planner-comparison.md`.

    python3 -m unittest discover -s benchmarks/plan-compare -p 'test_*.py'
"""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import differ  # noqa: E402


def duck(name: str, *children: dict, **info) -> dict:
    """A node of DuckDB's JSON plan; `info` with `_` for the spaces of the
    names of the details (`Join_Type` is `Join Type`)."""
    return {
        "name": name,
        "children": list(children),
        "extra_info": {key.replace("_", " "): value for key, value in info.items()},
    }


def write_duckdb(directory: pathlib.Path, name: str, root: dict) -> None:
    (directory / f"{name}.json").write_text(json.dumps([root]), encoding="utf-8")


class Dump:
    """A plan dump of the pipeline engine, in the schema of the profile."""

    def __init__(self) -> None:
        self.operators: list[dict] = []
        self.pipelines: list[dict] = []

    def op(self, kind: str, name: str, *inputs: int, rows: int = 0) -> int:
        op_id = len(self.operators)
        self.operators.append(
            {
                "id": op_id,
                "kind": kind,
                "name": name,
                "inputs": list(inputs),
                "pipelines": [],
                "metrics": {
                    "rows_in": {"value": 0, "unit": "rows"},
                    "rows_out": {"value": rows, "unit": "rows"},
                },
            }
        )
        return op_id

    def pipeline(self, source: int, operators: list[int], sink: int) -> None:
        self.pipelines.append(
            {
                "id": len(self.pipelines),
                "meta_pipeline": 0,
                "source": source,
                "operators": operators,
                "sink": sink,
                "dependencies": [],
                "meta_dependencies": [],
            }
        )

    def write(self, directory: pathlib.Path, name: str) -> None:
        dump = {
            "schema": "datafusion-physical-pipeline/profile",
            "version": 1,
            "root": 0,
            "operators": self.operators,
            "pipelines": self.pipelines,
            "meta_pipelines": [],
        }
        (directory / f"{name}.json").write_text(json.dumps(dump), encoding="utf-8")


def scan(table: str, projection: str, extra: str = "") -> str:
    """The description of one of our Parquet scans of `table`."""
    return (
        f"ParquetScanSource: file_groups={{1 group: [[{table}/part-0.parquet]]}}, "
        f"projection=[{projection}], file_type=parquet{extra}"
    )


class ExpressionTest(unittest.TestCase):
    def test_conjuncts_of_both_dialects(self):
        self.assertEqual(
            differ.conjuncts("((a >= 1) AND (b <= 2)) AND (c = 'x AND y')"),
            ["a >= 1", "b <= 2", "c = 'x AND y'"],
        )
        self.assertEqual(
            differ.conjuncts(
                "l_shipdate@10 > 1995-03-15 AND DynamicFilter [ x@0 >= 1 AND x@0 <= 2 ]"
            ),
            ["l_shipdate@10 > 1995-03-15", "DynamicFilter [ x@0 >= 1 AND x@0 <= 2 ]"],
        )
        # A disjunction is one conjunct (OR binds less tightly than AND).
        self.assertEqual(len(differ.conjuncts("a = 1 AND b = 2 OR c = 3")), 1)

    def test_columns(self):
        self.assertEqual(
            differ.columns_of("substr(c_phone@4, 1, 2) IN (SET)", differ.PIPELINE),
            {"c_phone"},
        )
        # DuckDB: identifiers, quoted or not, but not in strings or functions.
        self.assertEqual(
            differ.columns_of(
                "(\"substring\"(c_phone, 1, 2) = 'c_name') AND (\"URL\" != '')",
                differ.DUCKDB,
            ),
            {"c_phone", "URL"},
        )

    def test_constant_columns(self):
        engine = differ.PIPELINE
        self.assertEqual(differ.constant_columns("a_x@0 > 1", engine), frozenset())
        self.assertEqual(
            differ.constant_columns("l_shipdate@1 < l_commitdate@2", engine),
            frozenset(),
        )
        self.assertEqual(
            differ.constant_columns("l_shipdate@1 < 1995-01-01", engine), {"l_shipdate"}
        )
        # Every disjunct bounds p_size and p_brand, one of them p_container.
        self.assertEqual(
            differ.constant_columns(
                "p_brand@1 = A AND p_size@2 <= 5 OR p_brand@1 = B AND p_size@2 <= 10 "
                "AND p_container@3 IN (SET)",
                engine,
            ),
            {"p_brand", "p_size"},
        )

    def test_atoms(self):
        optional = differ.make_atom("optional: (n_name IN ('A', 'B'))", differ.DUCKDB)
        self.assertEqual((optional.kind, optional.pruning_only), ("filter", True))
        self.assertEqual(optional.constant, {"n_name"})
        topn = differ.make_atom("optional: Dynamic Filter (EventTime)", differ.DUCKDB)
        self.assertEqual((topn.kind, topn.columns), ("topn_dynamic", {"EventTime"}))
        not_null = differ.make_atom("(l_orderkey IS NOT NULL)", differ.DUCKDB)
        self.assertEqual(not_null.kind, "is_not_null")
        for text, kind in [
            ("DynamicFilter [ empty ]", "topn_dynamic"),
            (
                "DynamicFilter [ EventTime@5 < 13 OR EventTime@5 = 13 AND WatchID@0 < 2 ]",
                "topn_dynamic",
            ),
            (
                "DynamicFilter [ o_custkey@1 >= 1 AND o_custkey@1 <= 15000 ]",
                "join_dynamic",
            ),
            (
                "DynamicFilter [ k@0 >= 1 AND k@0 <= 9 AND hash_lookup(k@0) ]",
                "join_dynamic",
            ),
            (
                "DynamicFilter [ k@0 >= 1 AND k@0 <= 9 AND k@0 IN (SET) ([1, 9]) ]",
                "join_dynamic",
            ),
        ]:
            self.assertEqual(differ.make_atom(text, differ.PIPELINE).kind, kind, text)


class NormalizationTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_duckdb_plan(self):
        """Projections are removed, the filters above a scan are folded into
        it, the table of a scan is that of its columns, a MARK join of an IN
        list is the IN filter, and the build side of a join is its right
        child."""
        root = duck(
            "HASH_GROUP_BY",
            duck(
                "PROJECTION",
                duck(
                    "HASH_JOIN",
                    duck(
                        "FILTER",
                        duck(
                            "READ_PARQUET",
                            Function="READ_PARQUET",
                            Projections=["l_orderkey", "l_commitdate"],
                            Filters="(l_shipdate < '1995-01-01'::DATE)",
                            Estimated_Cardinality="100",
                        ),
                        Expression="(l_commitdate < l_receiptdate)",
                    ),
                    duck(
                        "FILTER",
                        duck(
                            "HASH_JOIN",
                            duck(
                                "READ_PARQUET",
                                Function="READ_PARQUET",
                                Projections="o_orderkey",
                                Estimated_Cardinality="50",
                            ),
                            duck("COLUMN_DATA_SCAN", Estimated_Cardinality="3"),
                            Join_Type="MARK",
                            Conditions="o_orderpriority = #0",
                        ),
                        Expression="IN (...)",
                    ),
                    Join_Type="INNER",
                    Conditions="l_orderkey = o_orderkey",
                ),
            ),
            Groups="#0",
            Aggregates="count_star()",
        )
        write_duckdb(self.path, "q", root)
        plan = differ.load_duckdb(self.path / "q.json")
        fingerprint = differ.fingerprint(plan)
        self.assertEqual([s.table for s in fingerprint.scans], ["lineitem", "orders"])
        lineitem, orders = fingerprint.scans
        self.assertEqual(
            lineitem.read, {"l_orderkey", "l_commitdate", "l_receiptdate", "l_shipdate"}
        )
        self.assertEqual(lineitem.constant, {"l_shipdate"})
        self.assertEqual(orders.filtered, {"o_orderpriority"})
        (join,) = fingerprint.joins
        self.assertEqual((join.label, join.build), ("(lineitem ⋈ orders)", "orders"))
        self.assertEqual(fingerprint.breaks, {"aggregation": 1, "join build": 1})

    def test_duckdb_scalar_subquery(self):
        """A join with an ungrouped aggregation on SUBQUERY is a subquery and a
        filter of the other input."""
        root = duck(
            "NESTED_LOOP_JOIN",
            duck("READ_PARQUET", Projections="c_acctbal"),
            duck(
                "UNGROUPED_AGGREGATE",
                duck("READ_PARQUET", Projections="c_acctbal"),
                Aggregates="avg(#0)",
            ),
            Join_Type="INNER",
            Conditions="CAST(c_acctbal AS DOUBLE) > SUBQUERY",
        )
        write_duckdb(self.path, "q", root)
        plan = differ.load_duckdb(self.path / "q.json")
        self.assertEqual(plan.root.kind, "scan")
        self.assertEqual(plan.root.attrs["table"], "customer")
        self.assertEqual([s.kind for s in plan.subqueries], ["aggregate"])
        fingerprint = differ.fingerprint(plan)
        self.assertEqual(fingerprint.joins, [])
        self.assertEqual(fingerprint.scans[0].constant, {"c_acctbal"})
        self.assertEqual(fingerprint.breaks, {"aggregation": 1, "scalar subquery": 1})

    def test_our_plan(self):
        """The build side of a join is the input whose pipeline it sinks
        into; the pipelines that finalize its hash table are no pipeline
        breaks; a limit with an offset above a TopK is a top-N with an
        offset; the projections and the caching operators are removed and a
        filter above a scan is folded into it."""
        dump = Dump()
        sink = dump.op("CollectorSink", "CollectorSink", 1)
        limit = dump.op("PhysicalLimit", "PhysicalLimit: skip=5, fetch=10", 2, rows=10)
        topk = dump.op(
            "PhysicalTopK",
            "PhysicalTopK: fetch=15, expr=[o_orderkey@0 ASC]",
            3,
            rows=15,
        )
        join = dump.op(
            "PhysicalHashJoin",
            "PhysicalHashJoin: join_type=Inner, on=[(o_orderkey@0, l_orderkey@0)]",
            4,
            6,
            rows=20,
        )
        caching = dump.op("CachingOperator", "CachingOperator: threshold=256", 5)
        orders = dump.op(
            "ParquetScanSource",
            scan(
                "orders",
                "o_orderkey, o_orderdate",
                ", filter=o_orderdate@1 < 1995-01-01",
            ),
            rows=30,
        )
        filtered = dump.op("PhysicalFilter", "PhysicalFilter: l_tax@1 > 0.01", 7)
        lineitem = dump.op(
            "ParquetScanSource",
            scan(
                "lineitem",
                "l_orderkey, l_tax",
                ", predicate=DynamicFilter [ l_orderkey@0 >= 1 AND l_orderkey@0 <= 9 ]",
            ),
            rows=40,
        )
        build = dump.op("JoinHashTableBuild", "JoinHashTableBuild")
        dump.pipeline(topk, [limit], sink)
        dump.pipeline(lineitem, [filtered, join], topk)
        dump.pipeline(orders, [caching], join)
        dump.pipeline(build, [], join)
        dump.write(self.path, "q")
        plan = differ.load_pipeline(self.path / "q.json")
        fingerprint = differ.fingerprint(plan)
        (order,) = fingerprint.orders
        self.assertEqual((order.kind, order.k, order.offset), ("topn", 10, 5))
        (join_fp,) = fingerprint.joins
        self.assertEqual(join_fp.build, "orders")
        self.assertEqual((join_fp.build_rows, join_fp.probe_rows), (30, 40))
        scans = {s.table: s for s in fingerprint.scans}
        self.assertEqual(scans["orders"].constant, {"o_orderdate"})
        self.assertEqual(scans["lineitem"].constant, {"l_tax"})
        self.assertEqual(scans["lineitem"].filtered, {"l_tax"})
        self.assertEqual(fingerprint.breaks, {"top-N": 1, "join build": 1})


class ComparisonTest(unittest.TestCase):
    """Small plans of both engines, and the classification of their
    divergences."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def duckdb_join(self, probe: str, build: str) -> dict:
        tables = {
            "lineitem": "l_orderkey",
            "orders": "o_orderkey",
            "customer": "c_custkey",
        }
        node = {
            name: duck("READ_PARQUET", Projections=column, Estimated_Cardinality="10")
            for name, column in tables.items()
        }
        return duck(
            "HASH_JOIN",
            node[probe],
            node[build],
            Join_Type="INNER",
            Conditions=f"{tables[probe]} = {tables[build]}",
            Estimated_Cardinality="10",
        )

    def our_join(self, build: str, probe: str) -> Dump:
        columns = {"lineitem": "l_orderkey", "orders": "o_orderkey"}
        dump = Dump()
        sink = dump.op("CollectorSink", "CollectorSink", 1)
        join = dump.op(
            "PhysicalHashJoin",
            f"PhysicalHashJoin: join_type=Inner, on=[({columns[build]}@0, "
            f"{columns[probe]}@0)]",
            2,
            3,
        )
        build_scan = dump.op("ParquetScanSource", scan(build, columns[build]), rows=5)
        probe_scan = dump.op("ParquetScanSource", scan(probe, columns[probe]), rows=7)
        dump.pipeline(probe_scan, [join], sink)
        dump.pipeline(build_scan, [], join)
        return dump

    def compare(self, duckdb_root: dict, ours: Dump) -> differ.Comparison:
        write_duckdb(self.path, "duckdb", duckdb_root)
        ours.write(self.path, "ours")
        return differ.compare(
            "q",
            differ.load_duckdb(self.path / "duckdb.json"),
            differ.load_pipeline(self.path / "ours.json"),
        )

    def test_the_same_shape_matches(self):
        comparison = self.compare(
            self.duckdb_join("lineitem", "orders"), self.our_join("orders", "lineitem")
        )
        self.assertEqual(comparison.verdict, "matches")

    def test_another_build_side_is_cost_driven(self):
        comparison = self.compare(
            self.duckdb_join("lineitem", "orders"), self.our_join("lineitem", "orders")
        )
        self.assertEqual([d.cause.id for d in comparison.divergences], ["build-side"])
        self.assertEqual(comparison.verdict, differ.COST)

    def test_another_join_order_is_a_rule_gap(self):
        duckdb_root = duck(
            "HASH_JOIN",
            duck("READ_PARQUET", Projections="l_orderkey"),
            duck(
                "HASH_JOIN",
                duck("READ_PARQUET", Projections=["o_orderkey", "o_custkey"]),
                duck("READ_PARQUET", Projections="c_custkey"),
                Join_Type="INNER",
                Conditions="o_custkey = c_custkey",
            ),
            Join_Type="INNER",
            Conditions="l_orderkey = o_orderkey",
        )
        dump = Dump()
        sink = dump.op("CollectorSink", "CollectorSink", 1)
        top = dump.op(
            "PhysicalHashJoin",
            "PhysicalHashJoin: join_type=Inner, on=[(c_custkey@0, o_custkey@1)]",
            2,
            3,
        )
        customer = dump.op("ParquetScanSource", scan("customer", "c_custkey"))
        bottom = dump.op(
            "PhysicalHashJoin",
            "PhysicalHashJoin: join_type=Inner, on=[(o_orderkey@0, l_orderkey@0)]",
            4,
            5,
        )
        orders = dump.op("ParquetScanSource", scan("orders", "o_orderkey, o_custkey"))
        lineitem = dump.op("ParquetScanSource", scan("lineitem", "l_orderkey"))
        dump.pipeline(lineitem, [bottom, top], sink)
        dump.pipeline(customer, [], top)
        dump.pipeline(orders, [], bottom)
        comparison = self.compare(duckdb_root, dump)
        self.assertEqual([d.cause.id for d in comparison.divergences], ["join-order"])
        self.assertEqual(comparison.verdict, differ.RULE_GAP)

    def test_a_derived_null_filter_is_a_rule_gap(self):
        duckdb_root = duck(
            "READ_PARQUET",
            Projections="l_orderkey",
            Filters="(l_orderkey IS NOT NULL)",
        )
        dump = Dump()
        dump.op("CollectorSink", "CollectorSink", 1)
        lineitem = dump.op("ParquetScanSource", scan("lineitem", "l_orderkey"))
        dump.pipeline(lineitem, [], 0)
        comparison = self.compare(duckdb_root, dump)
        self.assertEqual(
            [d.cause.id for d in comparison.divergences], ["null-key-filters"]
        )


class PredicateTest(unittest.TestCase):
    """The filters of the scans compared by their meaning."""

    def terms(self, text: str, engine: str = differ.PIPELINE) -> frozenset:
        del engine  # Both dialects have the same parser.
        return frozenset(differ.filter_terms(text))

    def test_both_dialects_normalize_alike(self):
        pairs = [
            ("l_discount@6 <= 0.07", "(l_discount <= 0.07)"),
            ("l_shipdate@10 > 1995-03-15", "(l_shipdate > '1995-03-15'::DATE)"),
            ("l_quantity@4 < 24.00", "(l_quantity < 24)"),
            ("SearchPhrase@9 != ", "(SearchPhrase != '')"),
            ("URL@10 LIKE %google%", "contains(URL, 'google')"),
            ("p_type@4 LIKE %BRASS", "suffix(p_type, 'BRASS')"),
            (
                "p_type@4 NOT LIKE MEDIUM POLISHED%",
                "(NOT prefix(p_type, 'MEDIUM POLISHED'))",
            ),
            (
                "o_comment@8 NOT LIKE %special%requests%",
                "(o_comment !~~ '%special%requests%')",
            ),
            ("p_name@1 LIKE forest%", "(p_name >= 'forest') AND (p_name < 'foresu')"),
            (
                "n_name@1 = GERMANY OR n_name@1 = FRANCE",
                "n_name IN ('FRANCE', 'GERMANY')",
            ),
            (
                "l_shipinstruct@13 = DELIVER IN PERSON",
                "(l_shipinstruct = 'DELIVER IN PERSON')",
            ),
            ("l_receiptdate@12 > l_commitdate@11", "(l_commitdate < l_receiptdate)"),
            (
                "CAST(c_acctbal@5 AS Decimal128(19, 6)) > scalar_subquery(5016.966223)",
                "CAST(c_acctbal AS DOUBLE) > SUBQUERY",
            ),
            ("p_size@5 IN (SET) ([49, 14, 23])", "(p_size IN (14, 23, 49))"),
            # Factored out of a disjunction, or not.
            (
                "p_size@5 >= 1 AND (p_brand@3 = A AND p_size@5 <= 5 OR p_brand@3 = B)",
                "((p_brand = 'A') AND (p_size >= 1) AND (p_size <= 5)) OR "
                "((p_brand = 'B') AND (p_size >= 1))",
            ),
        ]
        for ours, duckdb in pairs:
            self.assertEqual(self.terms(ours), self.terms(duckdb), (ours, duckdb))

    def test_lists_whose_constants_duckdb_does_not_show(self):
        # DuckDB's `IN` lists of `MARK` joins: as many constants.
        unknown = self.terms('"substring"(c_phone, 1, 2) IN (#3)')
        known = self.terms("substr(c_phone@4, 1, 2) IN (SET) ([13, 31, 23])")
        self.assertTrue(all(differ.implies(known, term) for term in unknown))
        self.assertTrue(all(differ.implies(unknown, term) for term in known))
        four = self.terms("substr(c_phone@4, 1, 2) IN (SET) ([13, 31, 23, 29])")
        self.assertFalse(all(differ.implies(four, term) for term in unknown))

    def test_implication_by_bounds(self):
        implies = differ.implies
        terms = self.terms("l_discount@6 >= 0.05 AND l_discount@6 <= 0.07")
        for text, expected in [
            ("l_discount@6 <= 0.07", True),
            ("l_discount@6 <= 0.08", True),
            ("l_discount@6 < 0.07", False),
            ("l_discount@6 <= 0.06", False),
            ("l_discount@6 > 0.04", True),
            ("l_discount@6 != 0.10", True),
            ("l_discount@6 != 0.06", False),
            ("l_discount@6 IS NOT NULL", True),
            ("l_tax@7 IS NOT NULL", False),
        ]:
            (term,) = differ.filter_terms(text)
            self.assertEqual(implies(terms, term), expected, text)
        values = self.terms("p_size@5 IN (SET) ([3, 9])")
        for text, expected in [
            ("p_size@5 IN (SET) ([3, 9, 14])", True),
            ("p_size@5 IN (SET) ([3])", False),
            ("p_size@5 >= 3", True),
            ("p_size@5 < 9", False),
            ("p_size@5 != 4", True),
        ]:
            (term,) = differ.filter_terms(text)
            self.assertEqual(implies(values, term), expected, text)

    def test_implication_by_cases_and_unions(self):
        # q19: the bounds of DuckDB's scan, derived from the disjunction of the
        # join condition, and the disjunctions that our scan evaluates.
        ours = self.terms(
            "p_size@5 >= 1 AND (p_brand@3 = Brand#12 AND p_size@5 <= 5 OR "
            "p_brand@3 = Brand#23 AND p_size@5 <= 10)"
        )
        (brands,) = differ.filter_terms(
            "(p_brand = 'Brand#12') OR (p_brand = 'Brand#23')"
        )
        self.assertTrue(differ.implies(ours, brands))
        (sizes,) = differ.filter_terms("(p_size >= 1) OR (p_size <= 5)")
        self.assertTrue(differ.implies(ours, sizes))
        hull = self.terms("(l_quantity >= 1.00) AND (l_quantity <= 30.00)")
        (ranges,) = differ.filter_terms(
            "l_quantity@4 >= 1.00 AND l_quantity@4 <= 11.00 OR l_quantity@4 >= 10.00 "
            "AND l_quantity@4 <= 20.00 OR l_quantity@4 >= 20.00 AND l_quantity@4 <= 30.00"
        )
        self.assertTrue(differ.implies(hull, ranges))
        self.assertTrue(all(differ.implies(frozenset([ranges]), t) for t in hull))
        (gap,) = differ.filter_terms(
            "l_quantity@4 >= 1.00 AND l_quantity@4 <= 9.00 OR l_quantity@4 >= 10.00 "
            "AND l_quantity@4 <= 30.00"
        )
        self.assertFalse(differ.implies(hull, gap))

    def test_transitive_bounds(self):
        terms = self.terms(
            "l_shipdate@10 < l_commitdate@11 AND l_receiptdate@12 > l_commitdate@11 "
            "AND l_receiptdate@12 < 1995-01-01"
        )
        (bound,) = differ.filter_terms("(l_shipdate < '1995-01-01'::DATE)")
        self.assertFalse(differ.implies(terms, bound))
        self.assertTrue(differ.implies(differ.transitive(terms), bound))

    def compare_q6(self, edit) -> differ.Comparison:
        """q6, with the descriptions of our scan edited by `edit`."""
        dump = json.loads((HERE / "pipeline" / "q6.json").read_text(encoding="utf-8"))
        for op in dump["operators"]:
            if op["kind"] == "ParquetScanSource":
                op["name"] = edit(op["name"])
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "q6.json"
            path.write_text(json.dumps(dump), encoding="utf-8")
            return differ.compare(
                "q6",
                differ.load_duckdb(HERE / "duckdb" / "q6.json"),
                differ.load_pipeline(path),
            )

    def test_a_missing_pushed_predicate_is_a_divergence(self):
        """Without both copies of `l_discount <= 0.07` (of the predicate and
        of the folded filter), our scan still filters `l_discount` (by `>=
        0.05`), but not by that bound."""
        self.assertEqual(self.compare_q6(lambda name: name).verdict, "matches")

        def remove(name: str) -> str:
            for column in ("l_discount@6", "l_discount@2"):
                name = name.replace(f" AND {column} <= 0.07", "")
            # (The pruning predicate, which the differ does not read, still
            # has `l_discount_min@6 <= 0.07`.)
            self.assertNotIn("l_discount@6 <= 0.07", name)
            self.assertNotIn("l_discount@2 <= 0.07", name)
            return name

        comparison = self.compare_q6(remove)
        self.assertEqual(comparison.verdict, differ.RULE_GAP)
        ((cause, text),) = [
            (d.cause.id, f.text) for d in comparison.divergences for f in d.findings
        ]
        self.assertEqual(cause, "scan-filters")
        self.assertIn("`l_discount <= 0.07`", text)

    def test_a_changed_pushed_predicate_is_a_divergence(self):
        comparison = self.compare_q6(lambda name: name.replace("<= 0.07", "<= 0.08"))
        self.assertNotEqual(comparison.verdict, "matches")
        comparison = self.compare_q6(lambda name: name.replace("<= 0.07", "<= 0.06"))
        self.assertNotEqual(comparison.verdict, "matches")


class PlacementTest(unittest.TestCase):
    """The aggregations, top-N, sorts, limits and pipeline breaks compared by
    their locations."""

    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def test_a_top_n_below_its_aggregation_is_a_divergence(self):
        """hits15 with our TopK moved below the aggregation (the same
        operators and pipeline breaks, at other places)."""
        dump = json.loads(
            (HERE / "pipeline" / "hits15.json").read_text(encoding="utf-8")
        )
        operators = {op["kind"]: op for op in dump["operators"]}
        sink, topk = operators["CollectorSink"], operators["PhysicalTopK"]
        aggregate = operators["PhysicalHashAggregate"]
        # sink <- topk <- (projection) <- aggregate <- input: sink <- aggregate
        # <- topk <- input.
        above_aggregate = next(
            op for op in dump["operators"] if aggregate["id"] in op["inputs"]
        )
        below = aggregate["inputs"]
        sink["inputs"] = [aggregate["id"]]
        aggregate["inputs"] = [topk["id"]]
        topk["inputs"] = below
        if above_aggregate is not topk:
            above_aggregate["inputs"] = []
        for pipeline in dump["pipelines"]:
            ends = {pipeline["source"], pipeline["sink"]}
            if ends == {topk["id"], sink["id"]}:
                pipeline["source"] = aggregate["id"]
                pipeline["operators"] = []
            elif ends == {aggregate["id"], topk["id"]}:
                pipeline["source"], pipeline["sink"] = topk["id"], aggregate["id"]
                pipeline["operators"] = []
            elif pipeline["sink"] == aggregate["id"]:
                pipeline["sink"] = topk["id"]
        path = self.path / "hits15.json"
        path.write_text(json.dumps(dump), encoding="utf-8")
        original = differ.compare(
            "hits15",
            differ.load_duckdb(HERE / "duckdb" / "hits15.json"),
            differ.load_pipeline(HERE / "pipeline" / "hits15.json"),
        )
        self.assertEqual(original.verdict, "matches")
        moved = differ.compare(
            "hits15",
            differ.load_duckdb(HERE / "duckdb" / "hits15.json"),
            differ.load_pipeline(path),
        )
        self.assertEqual(moved.verdict, differ.RULE_GAP)
        findings = {f.component: f for d in moved.divergences for f in d.findings}
        self.assertEqual(
            set(findings), {"aggregations", "top-N and limits", "pipeline breaks"}
        )
        # As many pipeline breaks of every kind, at other places.
        self.assertEqual(moved.duckdb.breaks, moved.ours.breaks)
        self.assertIn(
            "`top10` over `hits` below `agg1`", findings["pipeline breaks"].text
        )

    def test_an_aggregation_below_a_join_is_a_divergence(self):
        def aggregate(child):
            return duck("HASH_GROUP_BY", child, Groups="#0", Aggregates="count_star()")

        def join(probe, build):
            return duck(
                "HASH_JOIN",
                probe,
                build,
                Join_Type="INNER",
                Conditions="l_orderkey = o_orderkey",
            )

        def lineitem():
            return duck("READ_PARQUET", Projections="l_orderkey")

        def orders():
            return duck("READ_PARQUET", Projections="o_orderkey")

        above = aggregate(join(lineitem(), orders()))
        below = join(aggregate(lineitem()), orders())
        write_duckdb(self.path, "above", above)
        write_duckdb(self.path, "below", below)
        # Two DuckDB plans compared as if one were ours: the locations only.
        duckdb = differ.fingerprint(differ.load_duckdb(self.path / "above.json"))
        other = differ.fingerprint(differ.load_duckdb(self.path / "below.json"))
        (finding,) = differ.compare_aggregates(duckdb, other)
        self.assertIn("`agg1` over `join[lineitem, orders]` at the top", finding.text)
        self.assertIn("`agg1` over `lineitem` below `join`", finding.text)
        (breaks,) = differ.compare_breaks(duckdb, other)
        self.assertEqual(duckdb.breaks, other.breaks)
        self.assertIn("pipeline breaks differ", breaks.text)

    def test_a_limit_above_or_below_a_join_is_a_divergence(self):
        def scans():
            return (
                duck("READ_PARQUET", Projections="l_orderkey"),
                duck("READ_PARQUET", Projections="o_orderkey"),
            )

        def join(probe, build):
            return duck(
                "HASH_JOIN",
                probe,
                build,
                Join_Type="INNER",
                Conditions="l_orderkey = o_orderkey",
            )

        lineitem, orders = scans()
        above = duck("STREAMING_LIMIT", join(lineitem, orders), Limit="10")
        lineitem, orders = scans()
        below = join(duck("STREAMING_LIMIT", lineitem, Limit="10"), orders)
        write_duckdb(self.path, "above", above)
        write_duckdb(self.path, "below", below)
        duckdb = differ.fingerprint(differ.load_duckdb(self.path / "above.json"))
        other = differ.fingerprint(differ.load_duckdb(self.path / "below.json"))
        (finding,) = differ.compare_orders(duckdb, other)
        self.assertIn(
            "`limit10` over `join[lineitem, orders]` at the top", finding.text
        )
        self.assertIn("`limit10` over `lineitem` below `join`", finding.text)

    def test_the_join_order_does_not_move_the_operators_above_it(self):
        """q5 diverges by its join order only: its aggregation, sort and
        pipeline breaks are at the same places."""
        comparison = differ.compare(
            "q5",
            differ.load_duckdb(HERE / "duckdb" / "q5.json"),
            differ.load_pipeline(HERE / "pipeline" / "q5.json"),
        )
        self.assertEqual([d.cause.id for d in comparison.divergences], ["join-order"])
        self.assertEqual(
            differ.compare_aggregates(comparison.duckdb, comparison.ours), []
        )
        self.assertEqual(differ.compare_orders(comparison.duckdb, comparison.ours), [])
        self.assertEqual(differ.compare_breaks(comparison.duckdb, comparison.ours), [])


class FixturesTest(unittest.TestCase):
    """The comparison of the captured plans of the 47 queries."""

    @classmethod
    def setUpClass(cls):
        queries = differ.parse_queries(
            (HERE / "queries.sql").read_text(encoding="utf-8")
        )
        cls.queries = queries
        cls.comparisons = [
            differ.load(name, sql, HERE / "duckdb", HERE / "pipeline")
            for name, sql in queries
        ]

    def test_every_query_has_a_verdict(self):
        self.assertEqual(len(self.comparisons), 47)
        for comparison in self.comparisons:
            self.assertIn(
                comparison.verdict,
                ("matches", differ.RULE_GAP, differ.COST),
                comparison.query,
            )

    def test_every_divergence_has_a_cause_and_a_note(self):
        self.assertEqual(differ.check_notes(self.comparisons), [])

    def test_every_scan_has_its_table_and_known_columns(self):
        for comparison in self.comparisons:
            for fingerprint in (comparison.duckdb, comparison.ours):
                for scan_ in fingerprint.scans:
                    self.assertIn(scan_.table, differ.SCHEMA, comparison.query)
                    self.assertTrue(
                        scan_.read <= set(differ.SCHEMA[scan_.table]), comparison.query
                    )

    def test_both_plans_have_the_same_tables(self):
        """The plans read the same tables, but where our plan answers from
        statistics."""
        for comparison in self.comparisons:
            duckdb = {s.table for s in comparison.duckdb.scans}
            ours = {s.table for s in comparison.ours.scans}
            if ours:
                self.assertEqual(duckdb, ours, comparison.query)

    def test_the_report_is_up_to_date(self):
        report = (differ.REPORT).read_text(encoding="utf-8")
        self.assertEqual(report, differ.report(self.comparisons))

    def test_the_differ_prints_the_summary_of_the_report(self):
        output = io.StringIO()
        argv = sys.argv
        sys.argv = ["differ.py", "--check"]
        try:
            with contextlib.redirect_stdout(output):
                differ.main()
        finally:
            sys.argv = argv
        summary = differ.summary(self.comparisons)
        self.assertIn(summary, output.getvalue())
        report = differ.REPORT.read_text(encoding="utf-8")
        self.assertIn("## Summary\n\n" + summary + "\n", report)
        for comparison in self.comparisons:
            self.assertIn(
                f"{comparison.query:<7} {comparison.verdict}", output.getvalue()
            )

    def test_the_report_names_the_duckdb_build(self):
        report = differ.REPORT.read_text(encoding="utf-8")
        self.assertIn(differ.DUCKDB_COMMIT, report)
        self.assertIn(differ.DUCKDB_VERSION, report)


if __name__ == "__main__":
    unittest.main()
