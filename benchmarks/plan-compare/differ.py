#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
# SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

"""Compare the pipeline engine's plans of the 47 benchmark queries with
DuckDB's.

Reads, for every query of `queries.sql`, DuckDB's plan (`duckdb/<query>.json`,
the `EXPLAIN (FORMAT JSON)` of `capture.py`) and ours (`pipeline/<query>.json`,
the plan-shape dump of the profile of the execution of the query, written by
`dump_plans`), normalizes both into one fingerprint (the filters and the
projections pushed into every scan; the join order, the join operators and
their build sides; the placement of the aggregations; top-N and limits; and the
pipeline breaks), compares the fingerprints, and classifies every divergence as
a rule gap (our optimizer lacks a rule DuckDB has) or as cost- or
statistics-driven (the rules agree, the choice differs). It prints a verdict for
every query and the summary of `docs/planner-comparison.md`, which `--report`
writes (and `--check` checks). See README.md and docs/planner-comparison.md.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import itertools
import json
import pathlib
import re
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from capture import DUCKDB_COMMIT, parse_queries  # noqa: E402

ROOT = HERE.parent.parent
REPORT = ROOT / "docs" / "planner-comparison.md"

DUCKDB = "duckdb"
PIPELINE = "pipeline"
ENGINES = {DUCKDB: "DuckDB", PIPELINE: "ours"}

#: The tables of the benchmark and their columns: the TPC-H tables of
#: `datafusion/physical-pipeline/tests/tpch_parity.rs` and the synthetic
#: `hits` table of `benches/pipeline_vs_default.rs`. DuckDB's plans do not
#: name the tables that they scan (they are views of `read_parquet`): a scan is
#: of the table of its columns.
SCHEMA = {
    "region": ("r_regionkey", "r_name", "r_comment"),
    "nation": ("n_nationkey", "n_name", "n_regionkey", "n_comment"),
    "supplier": (
        "s_suppkey",
        "s_name",
        "s_address",
        "s_nationkey",
        "s_phone",
        "s_acctbal",
        "s_comment",
    ),
    "part": (
        "p_partkey",
        "p_name",
        "p_mfgr",
        "p_brand",
        "p_type",
        "p_size",
        "p_container",
        "p_retailprice",
        "p_comment",
    ),
    "partsupp": (
        "ps_partkey",
        "ps_suppkey",
        "ps_availqty",
        "ps_supplycost",
        "ps_comment",
    ),
    "customer": (
        "c_custkey",
        "c_name",
        "c_address",
        "c_nationkey",
        "c_phone",
        "c_acctbal",
        "c_mktsegment",
        "c_comment",
    ),
    "orders": (
        "o_orderkey",
        "o_custkey",
        "o_orderstatus",
        "o_totalprice",
        "o_orderdate",
        "o_orderpriority",
        "o_clerk",
        "o_shippriority",
        "o_comment",
    ),
    "lineitem": (
        "l_orderkey",
        "l_partkey",
        "l_suppkey",
        "l_linenumber",
        "l_quantity",
        "l_extendedprice",
        "l_discount",
        "l_tax",
        "l_returnflag",
        "l_linestatus",
        "l_shipdate",
        "l_commitdate",
        "l_receiptdate",
        "l_shipinstruct",
        "l_shipmode",
        "l_comment",
    ),
    "hits": (
        "WatchID",
        "UserID",
        "CounterID",
        "RegionID",
        "ClientIP",
        "EventTime",
        "EventDate",
        "AdvEngineID",
        "SearchEngineID",
        "SearchPhrase",
        "URL",
        "Title",
        "IsRefresh",
        "DontCountHits",
        "ResolutionWidth",
        "MobilePhone",
        "MobilePhoneModel",
    ),
}

COLUMN_TABLE = {
    column: table for table, columns in SCHEMA.items() for column in columns
}

#: The virtual columns of DuckDB's scans (the row ids of late
#: materialization).
ROW_ID_COLUMNS = ("file_index", "file_row_number", "rowid")


# --------------------------------------------------------------------------
# Expressions
#
# Both plans render expressions as text: DuckDB's SQL (`(l_shipdate <=
# '1998-09-02'::DATE)`, columns by name, strings quoted) and DataFusion's
# physical expressions (`l_shipdate@10 <= 1998-09-02`, columns as
# `name@index`, strings unquoted). The differ only splits them into their
# conjuncts and disjuncts (at `AND` and `OR` outside of brackets and strings)
# and finds the columns that they reference: never compares their text.
# --------------------------------------------------------------------------

_OPEN = "([{"
_CLOSE = ")]}"
_DATAFUSION_COLUMN = re.compile(r"\b([A-Za-z_][A-Za-z0-9_]*)@\d+")
_SQL_STRING = re.compile(r"'(?:[^']|'')*'")
_IDENTIFIER = re.compile(r'"([^"]+)"|\b([A-Za-z_][A-Za-z0-9_]*)\b')
_STRICT_COMPARISON = re.compile(r"(?<![<>!=])[<>](?![=>])")


def split_top(text: str, separator: str) -> list[str]:
    """`text` split at `separator` outside of brackets and of single-quoted
    strings (DuckDB's string literals)."""
    parts = []
    depth = 0
    quoted = False
    start = 0
    i = 0
    while i < len(text):
        char = text[i]
        if quoted:
            if char == "'":
                if text.startswith("''", i):
                    i += 2
                    continue
                quoted = False
        elif char == "'":
            quoted = True
        elif char in _OPEN:
            depth += 1
        elif char in _CLOSE:
            depth -= 1
        elif depth == 0 and text.startswith(separator, i):
            parts.append(text[start:i])
            i += len(separator)
            start = i
            continue
        i += 1
    parts.append(text[start:])
    return [part.strip() for part in parts]


def _closing(text: str, start: int) -> int:
    """The index of the bracket that closes the one at `start`."""
    depth = 0
    quoted = False
    for i in range(start, len(text)):
        char = text[i]
        if quoted:
            quoted = char != "'"
        elif char == "'":
            quoted = True
        elif char in _OPEN:
            depth += 1
        elif char in _CLOSE:
            depth -= 1
            if depth == 0:
                return i
    return -1


def strip_parens(text: str) -> str:
    """`text` without the parentheses around all of it."""
    text = text.strip()
    while text.startswith("(") and _closing(text, 0) == len(text) - 1:
        text = text[1:-1].strip()
    return text


def disjuncts(text: str) -> list[str]:
    """The disjuncts of `text` (`OR` binds less tightly than `AND`)."""
    return split_top(strip_parens(text), " OR ")


def conjuncts(text: str) -> list[str]:
    """The conjuncts of `text`, flattened: a disjunction is one conjunct."""
    text = strip_parens(text)
    if len(split_top(text, " OR ")) > 1:
        return [text]
    parts = split_top(text, " AND ")
    if len(parts) == 1:
        return [text]
    return [conjunct for part in parts for conjunct in conjuncts(part)]


def columns_of(text: str, engine: str) -> frozenset[str]:
    """The columns of the benchmark's tables that `text` references."""
    if engine == PIPELINE:
        names = _DATAFUSION_COLUMN.findall(text)
    else:
        text = _SQL_STRING.sub("''", text)
        names = [quoted or bare for quoted, bare in _IDENTIFIER.findall(text)]
    return frozenset(name for name in names if name in COLUMN_TABLE)


def constant_columns(text: str, engine: str) -> frozenset[str]:
    """The columns that `text` (a filter) compares with constants: those of its
    conjuncts of one column (a comparison with a constant, an `IN` list, a
    `LIKE`, `IS NOT NULL`, ... or with the value of a scalar subquery), and the
    columns that every disjunct of a disjunction compares with constants (the
    single-column filters that it implies). These are the filters that a scan
    can evaluate on its own and prune row groups with."""
    branches = disjuncts(text)
    if len(branches) > 1:
        implied = [constant_columns(branch, engine) for branch in branches]
        return frozenset.intersection(*implied)
    parts = conjuncts(text)
    if len(parts) > 1:
        return frozenset().union(*(constant_columns(part, engine) for part in parts))
    columns = columns_of(text, engine)
    return columns if len(columns) == 1 else frozenset()


@dataclasses.dataclass(frozen=True)
class Atom:
    """A conjunct of the filters of a scan."""

    text: str
    columns: frozenset[str]
    #: The columns that it compares with constants, see `constant_columns`.
    constant: frozenset[str]
    #: `filter`, `is_not_null`, `topn_dynamic` (the dynamic filter of a top-N)
    #: or `join_dynamic` (the dynamic filter of a hash join, which only our
    #: plans show).
    kind: str
    #: Whether the scan only prunes with it (DuckDB's `optional:` filters,
    #: which a filter above the scan evaluates, and dynamic filters).
    pruning_only: bool = False


def is_topn_filter(text: str) -> bool:
    """Whether the expression of a `DynamicFilter [...]` of a DataFusion scan
    is that of a TopK: not set yet (`empty`), or a strict comparison with the
    current threshold (`a < x OR a = x AND b < y`). The dynamic filters of hash
    joins are bounds (`>=`, `<=`), membership tests (`hash_lookup`, `IN
    (SET)`) or `false`."""
    text = re.sub(r"IN \(SET\) \(\[.*?\]\)", "IN (SET)", text.strip())
    if text == "empty":
        return True
    if "hash_lookup" in text or "IN (SET)" in text:
        return False
    return bool(_STRICT_COMPARISON.search(text))


def make_atom(text: str, engine: str, pruning_only: bool = False) -> Atom:
    """The atom of the conjunct `text` of a filter of `engine`."""
    text = strip_parens(text)
    if engine == DUCKDB:
        if text.startswith("optional: "):
            return make_atom(text[len("optional: ") :], engine, True)
        dynamic = re.fullmatch(r"Dynamic Filter(?: \((.*)\))?", text)
        if dynamic:
            columns = frozenset(
                name for name in [dynamic.group(1) or ""] if name in COLUMN_TABLE
            )
            return Atom(text, columns, frozenset(), "topn_dynamic", True)
    elif text.startswith("DynamicFilter [") and text.endswith("]"):
        inner = text[len("DynamicFilter [") : -1]
        kind = "topn_dynamic" if is_topn_filter(inner) else "join_dynamic"
        return Atom(text, columns_of(inner, engine), frozenset(), kind, True)
    columns = columns_of(text, engine)
    kind = "filter"
    if len(columns) == 1 and re.fullmatch(r".* IS NOT NULL", text):
        kind = "is_not_null"
    return Atom(text, columns, constant_columns(text, engine), kind, pruning_only)


def atoms_of(text: str, engine: str) -> list[Atom]:
    """The atoms of the conjuncts of the filter `text`."""
    return [make_atom(conjunct, engine) for conjunct in conjuncts(text) if conjunct]


# --------------------------------------------------------------------------
# Plans
# --------------------------------------------------------------------------


@dataclasses.dataclass(eq=False)
class Op:
    """An operator of a normalized plan of either engine.

    `kind` is one of `scan`, `filter`, `join`, `aggregate`, `topn`, `sort`,
    `limit`, `cte` (DuckDB's shared subplan: its definition, then the plan that
    reads it), `cte_scan`, `values`, `empty` (a source of one row or none) and
    `other`. `line` is the operator as its plan shows it (for the excerpts),
    `rows` its rows: DuckDB's estimate, or the rows that it produced in our
    execution.
    """

    kind: str
    children: list[Op]
    line: str
    rows: int | None
    attrs: dict = dataclasses.field(default_factory=dict)

    def walk(self):
        yield self
        for child in self.children:
            yield from child.walk()


@dataclasses.dataclass(eq=False)
class Plan:
    """The normalized plan of a query on one engine."""

    engine: str
    root: Op
    #: The plans of the uncorrelated scalar subqueries, evaluated once.
    subqueries: list[Op]
    #: The pipeline breaks: the operators that consume their input (or, for
    #: a join, its build side) before they produce rows, by kind.
    breaks: list[tuple[str, Op]]
    #: DuckDB's shared subplans, by index: `(name, definition)`.
    ctes: dict[str, tuple[str, Op]] = dataclasses.field(default_factory=dict)

    def roots(self) -> list[Op]:
        return [self.root, *self.subqueries]

    def ops(self):
        for root in self.roots():
            yield from root.walk()


def number(value) -> int | None:
    try:
        return int(str(value).replace(",", ""))
    except (TypeError, ValueError):
        return None


def as_list(value) -> list[str]:
    """A detail of DuckDB's JSON plan: a string, or a list of the lines of a
    detail of several lines."""
    if value is None:
        return []
    if isinstance(value, list):
        return [item for item in value if item != ""]
    return [value] if value != "" else []


# DuckDB ---------------------------------------------------------------------


def duckdb_line(node: dict) -> str:
    """A DuckDB operator as one line, as its text plan shows it."""
    details = []
    for key, value in node["extra_info"].items():
        if key == "Estimated Cardinality" or (
            key == "Function" and value == node["name"]
        ):
            continue
        details.append(f"{key}: {', '.join(as_list(value))}")
    rows = node["extra_info"].get("Estimated Cardinality")
    line = node["name"]
    if details:
        line += " " + "; ".join(details)
    if rows is not None:
        line += f" ~{number(rows):,} {'row' if number(rows) == 1 else 'rows'}"
    return line


def duckdb_raw(node: dict) -> Op:
    """The operator tree of DuckDB's JSON plan, one `Op` per operator."""
    children = [duckdb_raw(child) for child in node["children"]]
    info = node["extra_info"]
    name = node["name"]
    rows = number(info.get("Estimated Cardinality"))
    line = duckdb_line(node)
    if name in ("READ_PARQUET", "TABLE_SCAN", "SEQ_SCAN"):
        projections = as_list(info.get("Projections"))
        atoms = [make_atom(item, DUCKDB) for item in as_list(info.get("Filters"))]
        flat = []
        for atom in atoms:
            # A filter of several lines is one conjunct per line, but a line
            # may itself be a conjunction.
            if atom.kind == "filter" and len(conjuncts(atom.text)) > 1:
                flat += [
                    make_atom(part, DUCKDB, atom.pruning_only)
                    for part in conjuncts(atom.text)
                ]
            else:
                flat.append(atom)
        output = frozenset().union(*(columns_of(item, DUCKDB) for item in projections))
        return Op(
            "scan",
            [],
            line,
            rows,
            {"output": output, "atoms": flat, "limit": None},
        )
    if name == "PROJECTION":
        return Op("projection", children, line, rows)
    if name == "FILTER":
        expression = " AND ".join(
            f"({item})" for item in as_list(info.get("Expression"))
        )
        return Op("filter", children, line, rows, {"expression": expression})
    if name in (
        "HASH_JOIN",
        "NESTED_LOOP_JOIN",
        "PIECEWISE_MERGE_JOIN",
        "IE_JOIN",
        "CROSS_PRODUCT",
        "BLOCKWISE_NL_JOIN",
        "ASOF_JOIN",
        "POSITIONAL_JOIN",
        "DELIM_JOIN",
        "LEFT_DELIM_JOIN",
        "RIGHT_DELIM_JOIN",
    ):
        return Op(
            "join",
            children,
            line,
            rows,
            {
                "operator": name,
                "type": info.get("Join Type", "INNER"),
                "conditions": as_list(info.get("Conditions")),
                # The right child is the build side of DuckDB's joins.
                "build": 1,
            },
        )
    if name in ("HASH_GROUP_BY", "PERFECT_HASH_GROUP_BY", "UNGROUPED_AGGREGATE"):
        return Op(
            "aggregate",
            children,
            line,
            rows,
            {
                "operator": name,
                "groups": len(as_list(info.get("Groups"))),
                "aggregates": as_list(info.get("Aggregates")),
            },
        )
    if name == "TOP_N":
        return Op(
            "topn",
            children,
            line,
            rows,
            {
                "k": number(info.get("Top")),
                "offset": number(info.get("Offset")) or 0,
                "keys": as_list(info.get("Order By")),
            },
        )
    if name == "ORDER_BY":
        return Op("sort", children, line, rows, {"keys": as_list(info.get("Order By"))})
    if name in ("LIMIT", "STREAMING_LIMIT", "LIMIT_PERCENT"):
        return Op(
            "limit",
            children,
            line,
            rows,
            {
                "k": number(info.get("Limit")),
                "offset": number(info.get("Offset")) or 0,
                "blocking": name != "STREAMING_LIMIT",
            },
        )
    if name == "CTE":
        return Op(
            "cte",
            children,
            line,
            rows,
            {"name": info.get("CTE Name"), "index": info.get("Table Index")},
        )
    if name == "CTE_SCAN":
        return Op("cte_scan", children, line, rows, {"index": info.get("CTE Index")})
    if name in ("COLUMN_DATA_SCAN", "DUMMY_SCAN", "EMPTY_RESULT"):
        kind = "values" if name == "COLUMN_DATA_SCAN" else "empty"
        return Op(kind, children, line, rows)
    return Op("other", children, line, rows, {"operator": name})


def single_row(op: Op) -> bool:
    """Whether `op` (DuckDB's) produces exactly one row: an ungrouped
    aggregation."""
    while op.kind in ("projection", "filter") and op.children:
        op = op.children[0]
    return op.kind == "aggregate" and op.attrs["groups"] == 0


def duckdb_normalize(op: Op, subqueries: list[Op]) -> Op:
    """Normalize DuckDB's operator tree `op` into the model of the
    comparison (bottom-up):

    * projections are removed;
    * a join of uncorrelated scalar subquery (a join, on a condition with
      `SUBQUERY`, with an ungrouped aggregation, which DuckDB evaluates once)
      becomes a filter of its other input, and the aggregation a subquery, as
      DataFusion's `ScalarSubqueryExec` does;
    * a `FILTER` of the marker of a `MARK` join of an `IN` list of more
      constants than DuckDB's `IN` clause rewriter allows (a
      `COLUMN_DATA_SCAN` of the constants) becomes the `IN` filter that it
      evaluates, and one of the marker of a `MARK` join of an `IN` or `NOT
      IN` subquery (`SUBQUERY`, `NOT SUBQUERY`) a semi or an anti join;
    * the filters directly above a scan become filters of the scan, as our
      lowering folds the filters of the scans into them.
    """
    op.children = [duckdb_normalize(child, subqueries) for child in op.children]
    if op.kind == "projection":
        return op.children[0]
    if op.kind == "join" and len(op.children) == 2:
        conditions = op.attrs["conditions"]
        if any("SUBQUERY" in condition for condition in conditions):
            for main, subquery in ((0, 1), (1, 0)):
                if single_row(op.children[subquery]) and not single_row(
                    op.children[main]
                ):
                    subqueries.append(op.children[subquery])
                    expression = " AND ".join(f"({c})" for c in conditions)
                    filtered = Op(
                        "filter",
                        [op.children[main]],
                        op.line,
                        op.rows,
                        {"expression": expression, "scalar_subquery": True},
                    )
                    return fold_filter(filtered)
    if op.kind == "filter" and op.children and op.children[0].kind == "join":
        join = op.children[0]
        expression = strip_parens(op.attrs["expression"])
        if join.attrs.get("type") == "MARK":
            build = join.children[1]
            if build.kind == "values" and re.fullmatch(
                r"(NOT )?IN \(\.\.\.\)", expression
            ):
                # An IN list: the filter of the probe side by the constants.
                key = join.attrs["conditions"][0].split(" = ")[0]
                negated = expression.startswith("NOT ")
                text = f"{'NOT ' if negated else ''}{key} IN (...)"
                filtered = Op(
                    "filter",
                    [join.children[0]],
                    f"{op.line} <- {join.line} <- {build.line}",
                    op.rows,
                    {"expression": text},
                )
                return fold_filter(filtered)
            if expression in ("SUBQUERY", "NOT SUBQUERY"):
                join.attrs["type"] = "SEMI" if expression == "SUBQUERY" else "ANTI"
                join.attrs["mark"] = expression
                join.line = f"{op.line} <- {join.line}"
                join.rows = op.rows
                return join
    if op.kind == "filter":
        return fold_filter(op)
    return op


def fold_filter(op: Op) -> Op:
    """Fold the filter `op` into the scan below it, if it is directly above
    one (the filters of a scan of DuckDB, which are of one column each, and the
    filters of several columns directly above it, which DuckDB evaluates in
    the pipeline of the scan, are what our lowering folds into a scan)."""
    child = op.children[0] if op.children else None
    if child is None or child.kind != "scan":
        return op
    engine = child.attrs.get("engine", DUCKDB)
    child.attrs["atoms"] = child.attrs["atoms"] + atoms_of(
        op.attrs["expression"], engine
    )
    child.attrs.setdefault("above", []).append(op.line)
    return child


def duckdb_table(op: Op) -> str:
    """The table of DuckDB's scan `op`: that of its columns."""
    columns = set(op.attrs["output"])
    for atom in op.attrs["atoms"]:
        columns |= atom.columns
    tables = {COLUMN_TABLE[column] for column in columns}
    if len(tables) == 1:
        return tables.pop()
    return "?"


def duckdb_breaks(op: Op, breaks: list) -> None:
    """The pipeline breaks of DuckDB's normalized plan `op`: DuckDB's sinks,
    which its EXPLAIN does not show, but which its operators determine: the
    build side of a join, aggregations, sorts and top-N, `LIMIT` (not
    `STREAMING_LIMIT`), and shared subplans (`CTE`, which DuckDB materializes
    or streams to its readers, which it decides when it builds the
    pipelines)."""
    kind = {
        "join": "join build",
        "aggregate": "aggregation",
        "topn": "top-N",
        "sort": "sort",
        "cte": "shared subplan",
    }.get(op.kind)
    if op.kind == "limit" and op.attrs["blocking"]:
        kind = "limit"
    if kind:
        breaks.append((kind, op))
    for child in op.children:
        duckdb_breaks(child, breaks)


def load_duckdb(path: pathlib.Path) -> Plan:
    """DuckDB's normalized plan of `path`."""
    (root_json,) = json.loads(path.read_text(encoding="utf-8"))
    raw = duckdb_raw(root_json)
    subqueries: list[Op] = []
    root = duckdb_normalize(raw, subqueries)
    ctes = {}
    for op in itertools.chain(root.walk(), *(s.walk() for s in subqueries)):
        if op.kind == "scan":
            op.attrs["table"] = duckdb_table(op)
            op.attrs["engine"] = DUCKDB
        if op.kind == "cte":
            ctes[op.attrs["index"]] = (op.attrs["name"], op.children[0])
    breaks: list = []
    for subquery_root in [root, *subqueries]:
        duckdb_breaks(subquery_root, breaks)
    for subquery in subqueries:
        breaks.append(("scalar subquery", subquery))
    return Plan(DUCKDB, root, subqueries, breaks, ctes)


# Ours -----------------------------------------------------------------------


def split_fields(description: str) -> tuple[str, list[tuple[str, str]]]:
    """The name and the `key=value` fields of the description of one of our
    operators (`Name: key=value, key=value`), in order."""
    name, _, rest = description.partition(": ")
    fields = []
    for part in split_top(rest, ", "):
        key, separator, value = part.partition("=")
        if separator:
            fields.append((key.strip(), value.strip()))
        elif fields:
            # A value with `, ` outside of brackets: part of the last field.
            fields[-1] = (fields[-1][0], f"{fields[-1][1]}, {part}")
    return name, fields


def bracketed_list(value: str) -> list[str]:
    """The items of `[a, b, c]`."""
    value = value.strip()
    if value.startswith("[") and value.endswith("]"):
        value = value[1:-1]
    return [item for item in split_top(value, ", ") if item]


def pipeline_line(op: dict) -> str:
    """One of our operators as one line: its description without the file
    groups, the pruning predicate and the guarantees of scans, the values of
    `IN` lists and the projections of joins, with its rows."""
    line = op["name"]
    start = line.find("file_groups={")
    if start >= 0:
        end = _closing(line, start + len("file_groups="))
        line = line[:start] + line[end + 1 :].removeprefix(", ")
    line = re.sub(r", pruning_predicate=.*?, required_guarantees=\[[^\]]*\]", "", line)
    line = re.sub(r", file_type=parquet", "", line)
    line = re.sub(r", dynamic_rg_pruning=\w+", "", line)
    line = re.sub(r"IN \(SET\) \(\[[^\]]*\]\)", "IN (SET)", line)
    if op["kind"] == "PhysicalHashJoin":
        line = re.sub(r", projection=\[[^\]]*\]$", "", line)
    rows = op["metrics"].get("rows_out", {}).get("value")
    if rows is not None and op["kind"] != "CollectorSink":
        line += f" ({rows:,} {'row' if rows == 1 else 'rows'})"
    return line


def pipeline_table(op: dict) -> str:
    """The table of our scan `op`: the directory of its files."""
    match = re.search(r"\[\[([A-Za-z_]+)/", op["name"])
    return match.group(1) if match else "?"


class PipelineDump:
    """Our plan dump of one query: the operator tree and its pipelines."""

    def __init__(self, dump: dict) -> None:
        if dump.get("schema") != "datafusion-physical-pipeline/profile":
            raise ValueError("not a dump of the pipeline engine's profile")
        if dump.get("version") != 1:
            raise ValueError(f"unknown version {dump.get('version')} of the dump")
        self.operators = {op["id"]: op for op in dump["operators"]}
        self.pipelines = dump["pipelines"]
        self.root = dump["root"]

    def subtree(self, op_id: int) -> set[int]:
        ids = {op_id}
        for child in self.operators[op_id]["inputs"]:
            ids |= self.subtree(child)
        return ids

    def build_input(self, op_id: int) -> int | None:
        """The index of the input of the join `op_id` that it builds on: the
        input whose pipeline it is the sink of (the input that it consumes
        before it produces rows), not counting the pipelines that finalize
        its hash table (whose source is a `JoinHashTableBuild`)."""
        inputs = self.operators[op_id]["inputs"]
        for pipeline in self.pipelines:
            if pipeline["sink"] != op_id:
                continue
            if self.operators[pipeline["source"]]["kind"] == "JoinHashTableBuild":
                continue
            members = {pipeline["source"], *pipeline["operators"]}
            for index, child in enumerate(inputs):
                if members & self.subtree(child):
                    return index
        return None

    def breaks(self) -> list[tuple[str, int]]:
        """The pipeline breaks: the sinks of the pipelines, but for the sink
        of the execution, each once (a join is the sink of the pipeline of
        its build side and of the pipelines that finalize its hash table)."""
        kinds = {
            "PhysicalHashJoin": "join build",
            "PhysicalNestedLoopJoin": "join build",
            "PhysicalCrossJoin": "join build",
            "PhysicalMergeJoin": "join build",
            "PhysicalRangeJoin": "join build",
            "PhysicalAsOfJoin": "join build",
            "PhysicalSymmetricHashJoin": "join build",
            "PhysicalHashAggregate": "aggregation",
            "PhysicalTopK": "top-N",
            "PhysicalPartitionedTopK": "top-N",
            "PhysicalSort": "sort",
            "PhysicalWindow": "window",
            "ScalarSubqueryResult": "scalar subquery",
            "PhysicalMaterializedCte": "shared subplan",
            "PhysicalUnion": "union",
        }
        breaks = []
        seen = set()
        for pipeline in self.pipelines:
            sink = pipeline["sink"]
            kind = self.operators[sink]["kind"]
            if kind == "CollectorSink" or sink in seen:
                continue
            seen.add(sink)
            breaks.append((kinds.get(kind, kind), sink))
        return breaks


JOIN_TYPES = {
    # DataFusion's join type: (the kind, the side it emits or preserves:
    # `left`, `right` or None).
    "Inner": ("inner", None),
    "Left": ("outer", "left"),
    "Right": ("outer", "right"),
    "Full": ("full", None),
    "LeftSemi": ("semi", "left"),
    "RightSemi": ("semi", "right"),
    "LeftAnti": ("anti", "left"),
    "RightAnti": ("anti", "right"),
    "LeftMark": ("mark", "left"),
    "RightMark": ("mark", "right"),
}


def pipeline_op(
    dump: PipelineDump, op_id: int, subqueries: list[Op], by_id: dict[int, Op]
) -> Op:
    """Our operator `op_id` and its inputs, normalized (bottom-up): the
    projections, the `CachingOperator`s that coalesce chunks and the sink of
    the execution are removed; the subqueries of a `PhysicalScalarSubquery`
    become subqueries; a filter directly above a scan becomes a filter of the
    scan; and a `PhysicalLimit` with an offset directly above a `PhysicalTopK`
    of its offset and fetch becomes a top-N with an offset (as DuckDB's
    `TOP_N`). `by_id` maps the ids of the operators to the normalized
    operators that they became."""
    normalized = _pipeline_op(dump, op_id, subqueries, by_id)
    by_id[op_id] = normalized
    return normalized


def _pipeline_op(
    dump: PipelineDump, op_id: int, subqueries: list[Op], by_id: dict[int, Op]
) -> Op:
    op = dump.operators[op_id]
    kind = op["kind"]
    inputs = op["inputs"]
    line = pipeline_line(op)
    rows = op["metrics"].get("rows_out", {}).get("value")
    _, fields = split_fields(op["name"])
    values = collections.defaultdict(list)
    for key, value in fields:
        values[key].append(value)
    if kind == "PhysicalScalarSubquery":
        for subquery in inputs[1:]:
            subqueries.append(pipeline_op(dump, subquery, subqueries, by_id))
        return pipeline_op(dump, inputs[0], subqueries, by_id)
    children = [pipeline_op(dump, child, subqueries, by_id) for child in inputs]
    if kind in ("PhysicalProjection", "CachingOperator", "CollectorSink"):
        return children[0]
    if kind == "ScalarSubqueryResult":
        return children[0]
    if kind == "ParquetScanSource" or kind.endswith("ScanSource"):
        projection = (
            bracketed_list(values["projection"][0]) if values["projection"] else []
        )
        read = frozenset(
            column
            for item in projection
            for column in (
                {item} if item in COLUMN_TABLE else columns_of(item, PIPELINE)
            )
        )
        atoms = []
        for key in ("predicate", "filter"):
            for value in values.get(key, []):
                atoms += atoms_of(value, PIPELINE)
        limit = number(values["limit"][0]) if values.get("limit") else None
        return Op(
            "scan",
            [],
            line,
            rows,
            {
                "table": pipeline_table(op),
                "engine": PIPELINE,
                "read": read,
                "atoms": atoms,
                "limit": limit,
            },
        )
    if kind == "PhysicalFilter":
        expression = (
            values["filter"][0]
            if values.get("filter")
            else op["name"].partition(": ")[2]
        )
        expression = re.sub(r", projection=\[[^\]]*\]$", "", expression)
        filtered = Op("filter", children, line, rows, {"expression": expression})
        return fold_filter(filtered)
    if kind in (
        "PhysicalHashJoin",
        "PhysicalNestedLoopJoin",
        "PhysicalCrossJoin",
        "PhysicalMergeJoin",
        "PhysicalRangeJoin",
        "PhysicalAsOfJoin",
        "PhysicalSymmetricHashJoin",
    ):
        join_type = (values.get("join_type") or ["Inner"])[0]
        conditions = bracketed_list(values["on"][0]) if values.get("on") else []
        return Op(
            "join",
            children,
            line,
            rows,
            {
                "operator": kind,
                "type": join_type,
                "conditions": conditions,
                "filter": (values.get("filter") or [None])[0],
                "build": dump.build_input(op_id),
            },
        )
    if kind == "PhysicalHashAggregate":
        groups = bracketed_list(values["gby"][0]) if values.get("gby") else []
        aggregates = bracketed_list(values["aggr"][0]) if values.get("aggr") else []
        return Op(
            "aggregate",
            children,
            line,
            rows,
            {"operator": kind, "groups": len(groups), "aggregates": aggregates},
        )
    if kind in ("PhysicalTopK", "PhysicalPartitionedTopK"):
        return Op(
            "topn",
            children,
            line,
            rows,
            {
                "k": number(values["fetch"][0]) if values.get("fetch") else None,
                "offset": 0,
                "keys": bracketed_list(values["expr"][0]) if values.get("expr") else [],
            },
        )
    if kind == "PhysicalSort":
        return Op(
            "sort",
            children,
            line,
            rows,
            {"keys": bracketed_list(values["expr"][0]) if values.get("expr") else []},
        )
    if kind == "PhysicalLimit":
        skip = number(values["skip"][0]) if values.get("skip") else 0
        fetch = number(values["fetch"][0]) if values.get("fetch") else None
        child = children[0]
        if (
            child.kind == "topn"
            and child.attrs["offset"] == 0
            and fetch is not None
            and child.attrs["k"] == skip + fetch
        ):
            child.attrs["k"] = fetch
            child.attrs["offset"] = skip
            child.line = f"{line} <- {child.line}"
            child.rows = rows
            return child
        return Op(
            "limit",
            children,
            line,
            rows,
            {"k": fetch, "offset": skip, "blocking": False},
        )
    if kind in ("PhysicalPlaceholderRow", "EmptyScanSource"):
        return Op("empty", children, line, rows)
    return Op("other", children, line, rows, {"operator": kind})


def load_pipeline(path: pathlib.Path) -> Plan:
    """Our normalized plan of `path`."""
    dump = PipelineDump(json.loads(path.read_text(encoding="utf-8")))
    subqueries: list[Op] = []
    by_id: dict[int, Op] = {}
    root = pipeline_op(dump, dump.root, subqueries, by_id)
    breaks = [(kind, by_id[op_id]) for kind, op_id in dump.breaks()]
    return Plan(PIPELINE, root, subqueries, breaks)


# --------------------------------------------------------------------------
# Fingerprints
# --------------------------------------------------------------------------


def resolve(plan: Plan, op: Op) -> Op:
    """`op`, or the definition of the shared subplan that it reads."""
    if op.kind == "cte_scan" and op.attrs["index"] in plan.ctes:
        return plan.ctes[op.attrs["index"]][1]
    if op.kind == "cte":
        return op.children[1]
    return op


def base_tables(plan: Plan, op: Op) -> list[str]:
    """The tables of the scans below `op` (through shared subplans), sorted."""
    op = resolve(plan, op)
    if op.kind == "scan":
        return [op.attrs["table"]]
    if op.kind == "values":
        return ["values"]
    return sorted(table for child in op.children for table in base_tables(plan, child))


def leaf_label(plan: Plan, op: Op) -> str:
    """The label of an input of a join: a table, or an aggregation of tables
    (by its tables, whatever the order of its joins), through filters, sorts
    and limits."""
    op = resolve(plan, op)
    if op.kind == "join":
        return join_label(plan, op)
    if op.kind == "scan":
        return op.attrs["table"]
    if op.kind == "aggregate":
        return f"agg({', '.join(base_tables(plan, op))})"
    if op.kind in ("values", "empty"):
        return op.kind
    if len(op.children) == 1:
        return leaf_label(plan, op.children[0])
    return f"{op.kind}({', '.join(base_tables(plan, op))})"


def join_label(plan: Plan, op: Op) -> str:
    """The label of a join: the labels of its inputs, in canonical order (the
    join order, regardless of the build sides)."""
    op = resolve(plan, op)
    if op.kind != "join":
        return leaf_label(plan, op)
    return (
        "(" + " ⋈ ".join(sorted(join_label(plan, child) for child in op.children)) + ")"
    )


JOIN_KINDS = {
    # DuckDB's join type: (the kind, the side it emits or preserves: `left`
    # (the probe side) or `right` (the build side)).
    "INNER": ("inner", None),
    "LEFT": ("outer", "left"),
    "RIGHT": ("outer", "right"),
    "OUTER": ("full", None),
    "FULL": ("full", None),
    "SEMI": ("semi", "left"),
    "RIGHT_SEMI": ("semi", "right"),
    "ANTI": ("anti", "left"),
    "RIGHT_ANTI": ("anti", "right"),
    "MARK": ("mark", "left"),
    "SINGLE": ("single", "left"),
}


def join_kind(plan: Plan, op: Op) -> str:
    """The kind of the join `op`: `inner` (inner and semi joins, which keep
    the rows that match), `anti`, `outer` (with the side whose rows it keeps:
    `build` or `probe`), `full`, `mark` or `single`."""
    if plan.engine == DUCKDB:
        kind, side = JOIN_KINDS.get(op.attrs["type"], (op.attrs["type"].lower(), None))
        side_index = {"left": 0, "right": 1}.get(side)
    else:
        kind, side = JOIN_TYPES.get(op.attrs["type"], (op.attrs["type"].lower(), None))
        side_index = {"left": 0, "right": 1}.get(side)
    if kind == "semi":
        kind = "inner"
    if kind == "outer" and side_index is not None:
        kind = "outer keeping " + (
            "build" if side_index == op.attrs["build"] else "probe"
        )
    return kind


JOIN_OPERATORS = {
    "HASH_JOIN": "hash",
    "PhysicalHashJoin": "hash",
    "PhysicalSymmetricHashJoin": "hash",
    "NESTED_LOOP_JOIN": "nested loop",
    "BLOCKWISE_NL_JOIN": "nested loop",
    "PhysicalNestedLoopJoin": "nested loop",
    "CROSS_PRODUCT": "cross product",
    "PhysicalCrossJoin": "cross product",
    "PIECEWISE_MERGE_JOIN": "range",
    "IE_JOIN": "range",
    "PhysicalRangeJoin": "range",
    "PhysicalMergeJoin": "merge",
    "ASOF_JOIN": "as-of",
    "PhysicalAsOfJoin": "as-of",
}


@dataclasses.dataclass(eq=False)
class Scan:
    op: Op
    table: str
    #: The columns that it reads: its output and the columns of its filters.
    read: frozenset[str]
    #: The columns of its filters.
    filtered: frozenset[str]
    #: The columns that its filters compare with constants.
    constant: frozenset[str]
    #: Whether a top-N pushes its dynamic filter into it, and the columns of
    #: that filter (none for our TopK filters whose threshold was not set
    #: before the scan ended).
    topn: bool
    topn_columns: frozenset[str]
    atoms: list[Atom]
    limit: int | None


@dataclasses.dataclass(eq=False)
class Join:
    op: Op
    label: str
    operator: str
    kind: str
    build: str
    probe: str
    build_rows: int | None
    probe_rows: int | None


@dataclasses.dataclass(eq=False)
class Aggregate:
    op: Op
    #: The tables of its input.
    input: str
    groups: int
    #: Whether a join consumes its output (an aggregation below a join).
    below_join: bool


@dataclasses.dataclass(eq=False)
class Order:
    op: Op
    kind: str
    k: int | None
    offset: int
    keys: int
    input: str


@dataclasses.dataclass(eq=False)
class Fingerprint:
    """The normalized shape of a plan, see the module documentation."""

    plan: Plan
    scans: list[Scan]
    joins: list[Join]
    aggregates: list[Aggregate]
    orders: list[Order]
    breaks: collections.Counter
    #: The shared subplans of DuckDB: (name, the label of the definition, the
    #: number of readers).
    ctes: list[tuple[str, str, int]]


def fingerprint(plan: Plan) -> Fingerprint:
    scans, joins, aggregates, orders = [], [], [], []
    parents = {}
    for op in plan.ops():
        for child in op.children:
            parents[id(child)] = op
    for op in plan.ops():
        if op.kind == "scan":
            atoms = op.attrs["atoms"]
            static = [atom for atom in atoms if atom.kind in ("filter", "is_not_null")]
            filtered = frozenset().union(*(atom.columns for atom in static))
            read = op.attrs.get("read")
            if read is None:
                read = op.attrs["output"] | filtered
            scans.append(
                Scan(
                    op,
                    op.attrs["table"],
                    frozenset(read),
                    filtered,
                    frozenset().union(*(atom.constant for atom in static)),
                    any(atom.kind == "topn_dynamic" for atom in atoms),
                    frozenset().union(
                        *(atom.columns for atom in atoms if atom.kind == "topn_dynamic")
                    ),
                    atoms,
                    op.attrs.get("limit"),
                )
            )
        elif op.kind == "join":
            op.attrs["label"] = join_label(plan, op)
            build = op.attrs["build"]
            build_child = op.children[build] if build is not None else None
            probe_child = op.children[1 - build] if build is not None else None
            joins.append(
                Join(
                    op,
                    join_label(plan, op),
                    JOIN_OPERATORS.get(op.attrs["operator"], op.attrs["operator"]),
                    join_kind(plan, op),
                    join_label(plan, build_child) if build_child else "?",
                    join_label(plan, probe_child) if probe_child else "?",
                    build_child.rows if build_child else None,
                    probe_child.rows if probe_child else None,
                )
            )
        elif op.kind == "aggregate":
            parent = parents.get(id(op))
            while parent is not None and parent.kind in (
                "filter",
                "sort",
                "topn",
                "limit",
            ):
                parent = parents.get(id(parent))
            aggregates.append(
                Aggregate(
                    op,
                    ", ".join(base_tables(plan, op)),
                    op.attrs["groups"],
                    parent is not None and parent.kind == "join",
                )
            )
        elif op.kind in ("topn", "sort", "limit"):
            orders.append(
                Order(
                    op,
                    op.kind,
                    op.attrs.get("k"),
                    op.attrs.get("offset", 0),
                    len(op.attrs.get("keys", [])),
                    ", ".join(sorted(set(base_tables(plan, op)))),
                )
            )
    ctes = []
    for index, (name, definition) in plan.ctes.items():
        readers = sum(
            1
            for op in plan.ops()
            if op.kind == "cte_scan" and op.attrs["index"] == index
        )
        ctes.append((name, leaf_label(plan, definition), readers))
    breaks = collections.Counter(kind for kind, _ in plan.breaks)
    return Fingerprint(plan, scans, joins, aggregates, orders, breaks, ctes)


# --------------------------------------------------------------------------
# Comparison
# --------------------------------------------------------------------------


@dataclasses.dataclass(frozen=True)
class Cause:
    """Why the plans diverge, and what kind of divergence it is."""

    id: str
    title: str
    #: `rule gap` or `cost/statistics-driven`.
    classification: str
    #: What DuckDB does, and with which rule.
    duckdb: str
    #: What our planner does.
    ours: str
    #: The files that a fix would change (rule gaps).
    change: str = ""
    #: The follow-up issue (rule gaps).
    issue: int | None = None


RULE_GAP = "rule gap"
COST = "cost/statistics-driven"

CAUSES = {
    cause.id: cause
    for cause in (
        Cause(
            "join-order",
            "Join order",
            RULE_GAP,
            "Its join order optimizer (`src/optimizer/join_order/`) enumerates "
            "the join orders of every block of inner, semi and anti joins (DPhyp) "
            "and picks the cheapest by its cardinality estimates, bushy trees "
            "included.",
            "DataFusion has no join reordering: the joins of a query are planned in "
            "the order of its `FROM` clause and of its subqueries (`EliminateCrossJoin` "
            "and `PushDownFilter` turn the cross joins of the `FROM` clause into "
            "inner joins in that order, the decorrelated subqueries are joined last), "
            "and `JoinSelection` only chooses the build side of every join.",
            "`datafusion/physical-optimizer/src/` (a new join reordering rule next to "
            "`join_selection.rs`, whose statistics it would use, registered in "
            "`datafusion/physical-optimizer/src/optimizer.rs`)",
        ),
        Cause(
            "shared-cte",
            "Common table expressions computed once per reference",
            RULE_GAP,
            "Its `CTEInlining` (`src/optimizer/cte_inlining.cpp`) keeps a common "
            "table expression that the query reads more than once and that ends in an "
            "aggregation as one shared subplan (`CTE`, `CTE_SCAN`), which it computes "
            "once and materializes or streams to its readers.",
            "DataFusion inlines every common table expression that is not "
            "`MATERIALIZED`, so the plan computes it once per reference.",
            "`datafusion/sql/src/cte.rs` (plan a common table expression that the "
            "query reads more than once as a `MaterializedCte`, which the pipeline "
            "engine lowers natively)",
        ),
        Cause(
            "delim-join",
            "Correlated subqueries decorrelated by their outer keys",
            RULE_GAP,
            "It decorrelates a correlated subquery with a duplicate-eliminated "
            "join (`src/planner/subquery/flatten_dependent_join.cpp`, then "
            "`Deliminator`): it computes the outer rows once (a shared subplan "
            "`__duckdb_delim_<n>`), evaluates the subquery only for their distinct "
            "correlation keys, and joins its result back.",
            "DataFusion decorrelates the subquery into a join with an aggregation of "
            "the subquery over all of its rows (`ScalarSubqueryToJoin`, "
            "`DecorrelatePredicateSubquery`); the dynamic filters of our hash joins "
            "then restrict the scans of the subquery to the outer keys at run time.",
            "`datafusion/optimizer/src/scalar_subquery_to_join.rs`, "
            "`datafusion/optimizer/src/decorrelate_predicate_subquery.rs` and "
            "`datafusion/optimizer/src/decorrelate.rs`",
        ),
        Cause(
            "late-materialization",
            "Late materialization of top-N",
            RULE_GAP,
            "Its `LateMaterialization` (`src/optimizer/late_materialization.cpp`) "
            "plans a top-N over a scan of many columns as a top-N over the sort keys "
            "and the row ids (`file_index`, `file_row_number`), and a second scan "
            "that reads the other columns of the top rows only (a semi join on the "
            "row ids).",
            "Our TopK reads every column of every row that passes the filters of its "
            "scan (its dynamic filter skips rows, but only once it has its first "
            "rows).",
            "`datafusion/physical-pipeline/src/planner.rs` (the lowering of a TopK "
            "over a Parquet scan) and `datafusion/physical-pipeline/src/sources/parquet.rs` "
            "(row ids, and a scan of given rows)",
        ),
        Cause(
            "null-key-filters",
            "`IS NOT NULL` filters of join keys",
            RULE_GAP,
            "Its filter pushdown (`src/optimizer/pushdown/`) adds an `IS NOT NULL` "
            "filter of every nullable key of an inner or a semi join, and of the "
            "correlation keys of a decorrelated subquery, and pushes it into the "
            "scans.",
            "DataFusion's `FilterNullJoinKeys` "
            "(`datafusion/optimizer/src/filter_null_join_keys.rs`) does so only when "
            "`datafusion.optimizer.filter_null_join_keys` is on, which it is not by "
            "default; the hash joins skip the rows of null keys themselves.",
            "`datafusion/optimizer/src/filter_null_join_keys.rs` (or the defaults of "
            "the sessions of the pipeline engine)",
        ),
        Cause(
            "transitive-filters",
            "Constant bounds derived through comparisons of columns",
            RULE_GAP,
            "Its `FilterCombiner` (`src/optimizer/filter_combiner.cpp`) derives "
            "constant bounds through chains of comparisons of columns (`a < b AND "
            "b < c AND c < k` implies `a < k` and `b < k`), which its scans prune "
            "with.",
            "DataFusion pushes the comparisons as they are: the scan prunes with the "
            "bounds of the query only.",
            "`datafusion/optimizer/src/push_down_filter.rs` (or a rule of "
            "`datafusion/optimizer/src/simplify_expressions/`)",
        ),
        Cause(
            "build-side",
            "Build side of a join",
            COST,
            "Its `BuildProbeSideOptimizer` builds on the input with the smaller "
            "estimated cardinality.",
            "DataFusion's `JoinSelection` builds on the input with the smaller "
            "estimated size (statistics of rows and bytes; with no estimate it keeps "
            "the order of the query).",
        ),
        Cause(
            "exact-statistics",
            "Aggregation answered from exact statistics",
            COST,
            "It scans the table (its estimate of the rows of the scan is exact, but "
            "it plans the scan).",
            "DataFusion's `AggregateStatistics` answers `COUNT(*)`, `MIN` and `MAX` "
            "from the exact statistics of the Parquet metadata (the row counts and "
            "the minima and maxima of the columns): the plan reads no file. Our "
            "plan is ahead, through statistics: not a planner fix.",
        ),
    )
}

COMPONENTS = ("scans", "joins", "aggregations", "top-N and limits", "pipeline breaks")


@dataclasses.dataclass(eq=False)
class Finding:
    """A difference of one component of the fingerprints."""

    component: str
    text: str
    duckdb: list[Op]
    ours: list[Op]
    cause: str | None = None


@dataclasses.dataclass(eq=False)
class Divergence:
    """The findings of one cause."""

    cause: Cause
    findings: list[Finding]

    def ops(self, engine: str) -> list[Op]:
        return [
            op
            for finding in self.findings
            for op in (finding.duckdb if engine == DUCKDB else finding.ours)
        ]


@dataclasses.dataclass(eq=False)
class Comparison:
    query: str
    duckdb: Fingerprint
    ours: Fingerprint
    divergences: list[Divergence]

    @property
    def verdict(self) -> str:
        if not self.divergences:
            return "matches"
        if any(d.cause.classification == RULE_GAP for d in self.divergences):
            return RULE_GAP
        return COST


def plural(count: int, word: str) -> str:
    """`1 scan`, `2 scans`."""
    return f"{count} {word}" if count == 1 else f"{count} {word}s"


def times(count: int) -> str:
    return {0: "never", 1: "once", 2: "twice"}.get(count, f"{count} times")


def columns_text(columns) -> str:
    """Columns, as code."""
    return f"`{', '.join(sorted(columns))}`" if columns else "none"


def code(text: str) -> str:
    return f"`{text}`"


def tables_text(tables: list[str]) -> str:
    """A multiset of tables, as code: `nation` (2), `part`."""
    counts = collections.Counter(tables)
    return (
        ", ".join(
            f"`{table}` ({count})" if count > 1 else f"`{table}`"
            for table, count in sorted(counts.items())
        )
        or "no table"
    )


def pair_scans(duckdb: list[Scan], ours: list[Scan]):
    """Pair the scans of every table of both plans, by the most similar
    columns and filters; the scans of a table that the other plan scans fewer
    times stay unpaired."""
    pairs, left, right = [], [], []
    tables = sorted({scan.table for scan in duckdb + ours})
    for table in tables:
        mine = [scan for scan in duckdb if scan.table == table]
        theirs = [scan for scan in ours if scan.table == table]

        def similarity(a: Scan, b: Scan) -> float:
            def jaccard(x, y):
                return len(x & y) / len(x | y) if x | y else 1.0

            return (
                jaccard(a.read, b.read)
                + jaccard(a.constant, b.constant)
                + jaccard(a.filtered, b.filtered)
            )

        best, best_score = [], -1.0
        if len(mine) <= len(theirs):
            for permutation in itertools.permutations(theirs, len(mine)):
                score = sum(similarity(a, b) for a, b in zip(mine, permutation))
                if score > best_score:
                    best, best_score = list(zip(mine, permutation)), score
        else:
            for permutation in itertools.permutations(mine, len(theirs)):
                score = sum(similarity(a, b) for a, b in zip(permutation, theirs))
                if score > best_score:
                    best, best_score = list(zip(permutation, theirs)), score
        pairs += best
        left += [scan for scan in mine if scan not in {a for a, _ in best}]
        right += [scan for scan in theirs if scan not in {b for _, b in best}]
    return pairs, left, right


def compare_scans(duckdb: Fingerprint, ours: Fingerprint) -> list[Finding]:
    findings = []
    pairs, left, right = pair_scans(duckdb.scans, ours.scans)
    for a, b in pairs:
        differences = []
        cause = None
        if a.read != b.read:
            differences.append(
                f"reads {columns_text(a.read)} in DuckDB, {columns_text(b.read)} in ours"
            )
        only_duckdb = a.constant - b.constant
        only_ours = b.constant - a.constant
        if only_duckdb:
            not_null = {
                column
                for atom in a.atoms
                if atom.kind == "is_not_null"
                for column in atom.columns
            }
            compared = {
                column
                for atom in b.atoms
                if atom.kind == "filter" and len(atom.columns) > 1
                for column in atom.columns
            }
            if only_duckdb <= not_null:
                cause = "null-key-filters"
                differences.append(
                    f"DuckDB filters {columns_text(only_duckdb)} with `IS NOT NULL`"
                )
            elif only_duckdb <= compared:
                cause = "transitive-filters"
                differences.append(
                    f"DuckDB bounds {columns_text(only_duckdb)} by constants, which "
                    "our scan compares with other columns only"
                )
            else:
                differences.append(
                    f"DuckDB filters {columns_text(only_duckdb)} by constants, ours "
                    "does not"
                )
        if only_ours:
            differences.append(
                f"ours filters {columns_text(only_ours)} by constants, DuckDB does not"
            )
        if a.topn != b.topn:
            plans = "DuckDB's" if a.topn else "our"
            columns = a.topn_columns if a.topn else b.topn_columns
            differences.append(
                f"only {plans} top-N pushes its dynamic filter into it (on "
                f"{columns_text(columns)})"
            )
        if a.limit != b.limit:
            differences.append(
                f"the scan limit is {a.limit} in DuckDB, {b.limit} in ours"
            )
        if differences:
            findings.append(
                Finding(
                    "scans",
                    f"the scan of `{a.table}` ({columns_text(a.read)}): "
                    + "; ".join(differences),
                    [a.op],
                    [b.op],
                    cause,
                )
            )
    counts_duckdb = collections.Counter(scan.table for scan in duckdb.scans)
    counts_ours = collections.Counter(scan.table for scan in ours.scans)
    for table in sorted(set(counts_duckdb) | set(counts_ours)):
        if counts_duckdb[table] != counts_ours[table]:
            findings.append(
                Finding(
                    "scans",
                    f"`{table}` is scanned {times(counts_duckdb[table])} in DuckDB, "
                    f"{times(counts_ours[table])} in ours",
                    [scan.op for scan in left if scan.table == table],
                    [scan.op for scan in right if scan.table == table],
                )
            )
    return findings


def compare_joins(duckdb: Fingerprint, ours: Fingerprint) -> list[Finding]:
    findings = []
    remaining = list(ours.joins)
    unmatched = []
    for join in duckdb.joins:
        match = next((other for other in remaining if other.label == join.label), None)
        if match is None:
            unmatched.append(join)
            continue
        remaining.remove(match)
        if join.build != match.build:
            findings.append(
                Finding(
                    "joins",
                    f"the join {code(join.label)} builds on {code(join.build)} in DuckDB "
                    f"(~{join.build_rows:,} rows estimated, probed by "
                    f"~{join.probe_rows:,}), on {code(match.build)} in ours "
                    f"({match.build_rows:,} rows, probed by {match.probe_rows:,})",
                    [join.op],
                    [match.op],
                    "build-side",
                )
            )
        if join.operator != match.operator or join.kind != match.kind:
            findings.append(
                Finding(
                    "joins",
                    f"the join {code(join.label)} is a {join.operator} join ({join.kind}) "
                    f"in DuckDB, a {match.operator} join ({match.kind}) in ours",
                    [join.op],
                    [match.op],
                )
            )
    if unmatched or remaining:
        findings.append(
            Finding(
                "joins",
                "the join order differs: DuckDB joins "
                + (", ".join(code(join.label) for join in unmatched) or "none")
                + "; ours joins "
                + (", ".join(code(join.label) for join in remaining) or "none"),
                [join.op for join in unmatched],
                [join.op for join in remaining],
                "join-order",
            )
        )
    return findings


def compare_aggregates(duckdb: Fingerprint, ours: Fingerprint) -> list[Finding]:
    remaining = list(ours.aggregates)
    unmatched = []
    for aggregate in duckdb.aggregates:
        match = next(
            (
                other
                for other in remaining
                if (other.input, other.groups) == (aggregate.input, aggregate.groups)
            ),
            None,
        )
        if match is None:
            unmatched.append(aggregate)
        else:
            remaining.remove(match)
    if not unmatched and not remaining:
        return []

    def describe(aggregates):
        return (
            ", ".join(
                (
                    "an ungrouped aggregation"
                    if a.groups == 0
                    else f"an aggregation by {a.groups} key{'s' if a.groups > 1 else ''}"
                )
                + f" of {tables_text(a.input.split(', ') if a.input else [])}"
                for a in aggregates
            )
            or "none"
        )

    return [
        Finding(
            "aggregations",
            f"the aggregations differ: DuckDB has {describe(unmatched)} that ours has "
            f"not, ours {describe(remaining)} that DuckDB has not",
            [a.op for a in unmatched],
            [a.op for a in remaining],
        )
    ]


def compare_orders(duckdb: Fingerprint, ours: Fingerprint) -> list[Finding]:
    remaining = list(ours.orders)
    unmatched = []
    for order in duckdb.orders:
        key = (order.kind, order.k, order.offset, order.keys, order.input)
        match = next(
            (o for o in remaining if (o.kind, o.k, o.offset, o.keys, o.input) == key),
            None,
        )
        if match is None:
            unmatched.append(order)
        else:
            remaining.remove(match)
    if not unmatched and not remaining:
        return []

    def describe(orders):
        return (
            ", ".join(
                f"a {({'topn': 'top-N', 'sort': 'sort', 'limit': 'limit'})[o.kind]}"
                + (f" of {o.k}" if o.k is not None else "")
                + (f" after {o.offset}" if o.offset else "")
                + f" by {o.keys} key{'s' if o.keys != 1 else ''} of "
                + tables_text(o.input.split(", "))
                for o in orders
            )
            or "none"
        )

    return [
        Finding(
            "top-N and limits",
            f"DuckDB has {describe(unmatched)} that ours has not, ours "
            f"{describe(remaining)} that DuckDB has not",
            [o.op for o in unmatched],
            [o.op for o in remaining],
        )
    ]


def compare_breaks(duckdb: Fingerprint, ours: Fingerprint) -> list[Finding]:
    if duckdb.breaks == ours.breaks:
        return []
    kinds = sorted(set(duckdb.breaks) | set(ours.breaks))
    differences = [
        f"{kind} {duckdb.breaks[kind]} / {ours.breaks[kind]}"
        for kind in kinds
        if duckdb.breaks[kind] != ours.breaks[kind]
    ]
    changed = {kind for kind in kinds if duckdb.breaks[kind] != ours.breaks[kind]}
    return [
        Finding(
            "pipeline breaks",
            "the pipeline breaks differ (DuckDB / ours): " + ", ".join(differences),
            # The joins are in the excerpts of the findings of the joins.
            [op for kind, op in duckdb.plan.breaks if kind in changed - {"join build"}],
            [op for kind, op in ours.plan.breaks if kind in changed - {"join build"}],
        )
    ]


def attribute(duckdb: Fingerprint, ours: Fingerprint, findings: list[Finding]) -> None:
    """Attribute every finding to its cause.

    First to the rules of DuckDB that change the shape of a whole part of the
    plan, whose findings follow from them: late materialization (a join on
    the row ids of a scan), shared common table expressions (a `CTE`),
    decorrelation by the outer keys (a `CTE` `__duckdb_delim_<n>`); and to the
    exact statistics that let our planner answer the query without a scan.
    Then the findings of scans, joins and their build sides have their own
    causes (see `compare_scans` and `compare_joins`). A finding without a
    cause is `unexplained`: the differ fails on it (a new kind of divergence,
    to classify)."""
    plan = duckdb.plan
    late = any(
        column in " ".join(join.op.attrs["conditions"])
        for join in duckdb.joins
        for column in ROW_ID_COLUMNS
    )
    delim = {
        index
        for index, (name, _) in plan.ctes.items()
        if name.startswith("__duckdb_delim")
    }
    shared_tables = set()
    for index, (name, definition) in plan.ctes.items():
        if index not in delim:
            shared_tables |= set(base_tables(plan, definition))
    # DuckDB's operators of the decorrelation: the shared outer rows, the
    # operators that read them, and the operators above them.
    delim_ops: set[int] = set()
    for op in plan.ops():
        if op.kind == "cte" and op.attrs["index"] in delim:
            for reader in op.children[1].walk():
                if any(
                    o.kind == "cte_scan" and o.attrs["index"] in delim
                    for o in reader.walk()
                ):
                    delim_ops.add(id(reader))
        if any(o.kind == "cte" and o.attrs["index"] in delim for o in op.walk()):
            delim_ops.add(id(op))
    statistics = (
        bool(duckdb.scans)
        and not ours.scans
        and any(op.kind == "empty" for op in ours.plan.ops())
    )

    def tables_of(finding: Finding) -> set[str]:
        tables = set()
        for op in finding.duckdb:
            tables |= set(base_tables(duckdb.plan, op))
        for op in finding.ours:
            tables |= set(base_tables(ours.plan, op))
        return tables

    for finding in list(findings):
        if late:
            finding.cause = "late-materialization"
            if finding.component == "joins":
                finding.text = (
                    "DuckDB joins the top rows back to a second scan by their row ids "
                    "(`file_index`, `file_row_number`) to read their other columns"
                )
        elif statistics:
            finding.cause = "exact-statistics"
        elif shared_tables and (
            (
                finding.component in ("scans", "aggregations")
                and finding.cause is None
                and tables_of(finding) <= shared_tables
            )
            or (
                finding.component == "pipeline breaks"
                and "shared subplan" in finding.text
                and not delim
            )
        ):
            finding.cause = "shared-cte"
        elif delim and finding.cause is None:
            finding.cause = "delim-join"
        elif delim and finding.cause == "join-order":
            split_delim_joins(finding, delim_ops, findings)
    for finding in findings:
        if finding.cause is None:
            finding.cause = "unexplained"


def join_columns(op: Op, engine: str) -> frozenset[str]:
    """The columns of the conditions (and of the filter) of the join `op`."""
    text = " AND ".join(op.attrs["conditions"] + [op.attrs.get("filter") or ""])
    return columns_of(text, engine)


def split_delim_joins(
    finding: Finding, delim_ops: set[int], findings: list[Finding]
) -> None:
    """Split the joins of a join-order finding of a query that DuckDB
    decorrelates by its outer keys into those that follow from the
    decorrelation and the others, which follow from the join order: DuckDB's
    joins of the decorrelation (that read the shared outer rows, or are above
    them), and our joins of the decorrelated subqueries (semi and anti joins,
    and joins with an aggregation) on the columns of DuckDB's."""
    delim_duckdb = [op for op in finding.duckdb if id(op) in delim_ops]
    other_duckdb = [op for op in finding.duckdb if id(op) not in delim_ops]
    columns = frozenset().union(*(join_columns(op, DUCKDB) for op in delim_duckdb))
    delim_ours = [
        op
        for op in finding.ours
        if (
            JOIN_TYPES.get(op.attrs["type"], ("", None))[0] in ("semi", "anti")
            or any(child.kind == "aggregate" for child in op.children)
        )
        and join_columns(op, PIPELINE) & columns
    ]
    other_ours = [op for op in finding.ours if op not in delim_ours]
    finding.cause = "delim-join"
    finding.duckdb, finding.ours = delim_duckdb, delim_ours
    finding.text = (
        "DuckDB joins the shared outer rows with the subquery evaluated for their "
        "keys; ours joins the outer rows with the subquery over all of its rows"
    )
    if other_duckdb or other_ours:
        findings.append(
            Finding(
                "joins",
                "the join order of the rest of the query differs: DuckDB joins "
                + (", ".join(code(op.attrs["label"]) for op in other_duckdb) or "none")
                + "; ours joins "
                + (", ".join(code(op.attrs["label"]) for op in other_ours) or "none"),
                other_duckdb,
                other_ours,
                "join-order",
            )
        )


def compare_ctes(duckdb: Fingerprint, ours: Fingerprint) -> list[Finding]:
    """DuckDB's shared subplans (but those of decorrelations) that our plan
    computes once per reader: the subtrees of our plan with the label of
    their definition."""
    findings = []
    plan = duckdb.plan
    for index, (name, definition) in plan.ctes.items():
        if name.startswith("__duckdb_delim"):
            continue
        label = leaf_label(plan, definition)
        readers = [
            op
            for op in plan.ops()
            if op.kind == "cte_scan" and op.attrs["index"] == index
        ]
        copies = [
            op
            for op in ours.plan.ops()
            if op.kind == definition.kind
            and op.attrs.get("groups") == definition.attrs.get("groups")
            and leaf_label(ours.plan, op) == label
        ]
        if len(readers) > 1 and len(copies) > 1:
            cte = next(
                op
                for op in plan.ops()
                if op.kind == "cte" and op.attrs["index"] == index
            )
            findings.append(
                Finding(
                    "pipeline breaks",
                    f"DuckDB computes `{name}` ({code(label)}) once, in a shared subplan that "
                    f"it reads {times(len(readers))}; ours computes it {times(len(copies))}",
                    [cte, *definition.walk(), *readers],
                    [o for copy in copies for o in copy.walk()],
                    "shared-cte",
                )
            )
    return findings


def compare(query: str, duckdb_plan: Plan, our_plan: Plan) -> Comparison:
    duckdb = fingerprint(duckdb_plan)
    ours = fingerprint(our_plan)
    findings = (
        compare_ctes(duckdb, ours)
        + compare_scans(duckdb, ours)
        + compare_joins(duckdb, ours)
        + compare_aggregates(duckdb, ours)
        + compare_orders(duckdb, ours)
        + compare_breaks(duckdb, ours)
    )
    attribute(duckdb, ours, findings)
    by_cause: dict[str, list[Finding]] = {}
    for finding in findings:
        by_cause.setdefault(finding.cause, []).append(finding)
    divergences = [
        Divergence(CAUSES.get(cause) or Cause(cause, cause, RULE_GAP, "?", "?"), items)
        for cause, items in by_cause.items()
    ]
    return Comparison(query, duckdb, ours, divergences)


def query_tables(sql: str) -> list[str]:
    return [table for table in SCHEMA if re.search(rf"\b{table}\b", sql, re.IGNORECASE)]


def load(
    name: str, sql: str, duckdb_dir: pathlib.Path, pipeline_dir: pathlib.Path
) -> Comparison:
    duckdb_plan = load_duckdb(duckdb_dir / f"{name}.json")
    our_plan = load_pipeline(pipeline_dir / f"{name}.json")
    tables = query_tables(sql)
    for op in duckdb_plan.ops():
        if op.kind == "scan" and op.attrs["table"] == "?" and len(tables) == 1:
            op.attrs["table"] = tables[0]
    return compare(name, duckdb_plan, our_plan)


# --------------------------------------------------------------------------
# Excerpts
# --------------------------------------------------------------------------


def excerpt(plan: Plan, marked: list[Op]) -> str:
    """The part of `plan` that `marked` are in: the marked operators (`*`),
    the inputs of the marked joins, and their ancestors; `...` stands for the
    inputs left out."""
    marked_ids = {id(op) for op in marked}
    included = set(marked_ids)
    for op in plan.ops():
        if id(op) in marked_ids and op.kind == "join":
            included |= {id(child) for child in op.children}
    parents = {}
    for op in plan.ops():
        for child in op.children:
            parents[id(child)] = op
    for op_id in list(included):
        parent = parents.get(op_id)
        while parent is not None:
            included.add(id(parent))
            parent = parents.get(id(parent))
    lines: list[str] = []

    def render(op: Op, depth: int) -> None:
        mark = "*" if id(op) in marked_ids else " "
        parts = op.line.split(" <- ")
        if op.kind == "scan":
            parts = op.attrs.get("above", []) + parts
        for i, part in enumerate(parts):
            lines.append(f"{mark} {'  ' * (depth + i)}{part}")
        depth += len(parts) - 1
        elided = False
        for child in op.children:
            if id(child) in included:
                render(child, depth + 1)
                elided = False
            elif not elided:
                lines.append(f"  {'  ' * (depth + 1)}...")
                elided = True

    for index, root in enumerate(plan.roots()):
        if id(root) not in included:
            continue
        if index > 0:
            lines.append("  scalar subquery:")
        render(root, 1 if index > 0 else 0)
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

#: The likely impact of a divergence on its query.
IMPACTS = ("high", "medium", "low", "none")
IMPACT_SCORES = {"high": 3, "medium": 2, "low": 1, "none": 0}

#: What every divergence likely costs on its query: `(query, cause)` ->
#: `(impact, delta, note)`. The impact is measured on our engine where SQL can
#: give the query the shape of DuckDB's plan (see `measure.py` and
#: `experiments.sql`): `high` if that shape is at least 25% faster, `medium`
#: 10-25%, `low` 3-10%, `none` less than 3% (the noise of the measurements),
#: slower, or if our plan is ahead; `delta` is the time of the query with that
#: shape minus its time with ours, in ms (negative if DuckDB's shape is faster;
#: `None` if not measured). Every divergence has a note, and every note a
#: divergence (`check_notes`).
NOTES: dict[tuple[str, str], tuple[str, float | None, str]] = {
    ("q2", "delim-join"): (
        "none",
        None,
        "Our dynamic join filters already restrict the subquery to the part keys "
        "of the outer rows: its `partsupp` scan returns 66 of 80,000 rows. Both "
        "plans scan the four tables of the subquery once more: nothing to gain.",
    ),
    ("q2", "join-order"): (
        "none",
        -0.02,
        "Both blocks join in the order of the query, `region` last; the dynamic "
        "filters reduce every probe side to at most 207 rows. With DuckDB's join "
        "order written out, our engine runs q2 in 13.80 ms instead of 13.82 ms.",
    ),
    ("q4", "null-key-filters"): (
        "none",
        None,
        "`l_orderkey` has no nulls: the filter removes no row (and the hash join "
        "skips null keys anyway).",
    ),
    ("q5", "join-order"): (
        "none",
        1.29,
        "Our order builds on the 3,031 customers and the 4,081 orders that the "
        "dynamic filters let through and probes 16,281 `lineitem` rows; DuckDB's "
        "probes all of `lineitem` against the joins of the five other tables. "
        "With DuckDB's join order written out, our engine is slower: 14.14 ms "
        "instead of 12.85 ms.",
    ),
    ("q7", "join-order"): (
        "none",
        0.09,
        "The dynamic filters cut every probe side (11,043 `orders` rows, 1,216 "
        "customers): 13.64 ms with DuckDB's join order instead of 13.55 ms.",
    ),
    ("q8", "join-order"): (
        "none",
        -0.28,
        "15.97 ms with DuckDB's join order instead of 16.25 ms (2%, within the "
        "noise): the dynamic filters cut the `lineitem` scan to the 130 parts of "
        "the type.",
    ),
    ("q9", "join-order"): (
        "none",
        1.12,
        "Ours builds on all 80,000 `partsupp` and 150,000 `orders` rows (DuckDB "
        "builds on `orders` too) and the dynamic filters cut `lineitem` to 36,764 "
        "rows; DuckDB's order, `part` with `partsupp` first, is slower on our "
        "engine: 21.10 ms instead of 19.98 ms.",
    ),
    ("q10", "join-order"): (
        "low",
        -0.77,
        "DuckDB joins `customer` with `nation` and `lineitem` with `orders`, then "
        "the two (a bushy tree); ours joins in the order of the query. With "
        "DuckDB's join order written out, q10 takes 13.77 ms instead of 14.54 ms "
        "(5%).",
    ),
    ("q11", "join-order"): (
        "none",
        -0.23,
        "The dynamic filter of `supplier` cuts both `partsupp` scans to 3,760 "
        "rows: 9.44 ms with DuckDB's join order instead of 9.67 ms (2%, within "
        "the noise).",
    ),
    ("q12", "transitive-filters"): (
        "none",
        None,
        "The derived bounds follow from the filters that both plans have "
        "(`l_receiptdate < 1995-01-01` and `l_shipdate < l_commitdate < "
        "l_receiptdate`): they remove no row, and they prune no row group of this "
        "data, whose dates are spread over all row groups.",
    ),
    ("q14", "build-side"): (
        "none",
        None,
        "Ours builds on the 7,594 `lineitem` rows of the month, whose dynamic "
        "filter cuts the `part` scan to 5,538 rows; DuckDB builds on all 20,000 "
        "parts (it estimates 120,252 `lineitem` rows). Our build side is the "
        "smaller one.",
    ),
    ("q15", "shared-cte"): (
        "high",
        -4.39,
        "Ours scans the 22,996 `lineitem` rows of the quarter and aggregates them "
        "twice, for the join and for the `max` of the scalar subquery. With `WITH "
        "revenue0 AS MATERIALIZED (...)`, which the pipeline engine executes "
        "natively, q15 takes 5.67 ms instead of 10.06 ms (44%).",
    ),
    ("q17", "null-key-filters"): (
        "none",
        None,
        "`l_partkey` has no nulls: the filter removes no row.",
    ),
    ("q18", "join-order"): (
        "none",
        -0.21,
        "DuckDB applies the semi join with the large orders to `orders` before "
        "the other joins; our dynamic filters already cut `orders` to 381 rows "
        "and `lineitem` to 2,667 before our joins. 13.14 ms with DuckDB's join "
        "order instead of 13.35 ms (2%, within the noise).",
    ),
    ("q20", "delim-join"): (
        "none",
        None,
        "The dynamic filters restrict our aggregation of `lineitem` to the keys of "
        "the outer rows: its `lineitem` scan returns 27 rows. Both plans scan "
        "`lineitem` once for the subquery: nothing to gain.",
    ),
    ("q21", "join-order"): (
        "medium",
        -2.82,
        "Our anti join (`NOT EXISTS` of `l3`) builds its hash table on the "
        "380,417 `l3` rows with `l_receiptdate > l_commitdate` and probes it "
        "with 7,471 rows; DuckDB applies the anti join first and builds on the "
        "~4,509 (estimated) rows of the inner joins. With DuckDB's join order "
        "written out, q21 takes 18.58 ms instead of 21.40 ms (13%).",
    ),
    ("q21", "null-key-filters"): (
        "none",
        None,
        "`l_orderkey` and `l_suppkey` have no nulls: the filters remove no row.",
    ),
    ("q22", "delim-join"): (
        "none",
        None,
        "Our anti join builds on the 1,880 customers and probes `orders`, which "
        "its dynamic filter cuts to 18,077 rows; DuckDB semi-joins `orders` with "
        "the distinct keys of the shared customers. The same scans: nothing to "
        "gain.",
    ),
    ("hits0", "exact-statistics"): (
        "none",
        None,
        "Our plan reads no file (0.06 ms); DuckDB scans the 1,000,000 rows. Our "
        "plan is ahead.",
    ),
    ("hits6", "exact-statistics"): (
        "none",
        None,
        "Our plan answers `min` and `max` from the statistics of the Parquet "
        "files and reads no file; DuckDB scans `EventDate`. Our plan is ahead.",
    ),
    ("hits23", "late-materialization"): (
        "high",
        -15.20,
        "Our scan reads the 17 columns of the 33,790 rows with `URL LIKE "
        "'%google%'` (the dynamic filter of the TopK skips rows once it has 10); "
        "DuckDB reads `WatchID`, `EventTime` and `URL` for the top-N, then all "
        "columns of 10 rows. Reading the sort keys only, what DuckDB's first scan "
        "reads, takes 8.52 ms instead of 23.72 ms (64%): most of hits23 is "
        "the late materialization it lacks.",
    ),
}


def check_notes(comparisons: list[Comparison]) -> list[str]:
    """The divergences without a note and the notes without a divergence."""
    problems = []
    found = {(c.query, d.cause.id) for c in comparisons for d in c.divergences}
    for key in sorted(found - set(NOTES)):
        problems.append(f"no note for the divergence {key[1]} of {key[0]}")
    for key in sorted(set(NOTES) - found):
        problems.append(f"the note of {key[1]} of {key[0]} has no divergence")
    for key, (impact, _, _) in NOTES.items():
        if impact not in IMPACTS:
            problems.append(f"the note of {key[1]} of {key[0]} has impact {impact!r}")
    for comparison in comparisons:
        for divergence in comparison.divergences:
            if divergence.cause.id not in CAUSES:
                problems.append(
                    f"{comparison.query} has an unexplained divergence: "
                    + "; ".join(f.text for f in divergence.findings)
                )
    return problems


def cause_rows(comparisons: list[Comparison]) -> list[tuple[Cause, list[str], int]]:
    """Every cause with its queries and its impact score, the most common
    first (then by impact, then by title)."""
    queries: dict[str, list[str]] = collections.defaultdict(list)
    scores: dict[str, int] = collections.defaultdict(int)
    for comparison in comparisons:
        for divergence in comparison.divergences:
            queries[divergence.cause.id].append(comparison.query)
            impact = NOTES.get(
                (comparison.query, divergence.cause.id), ("none", None, "")
            )[0]
            scores[divergence.cause.id] += IMPACT_SCORES[impact]
    rows = [(CAUSES[cause], names, scores[cause]) for cause, names in queries.items()]
    rows.sort(key=lambda row: (-len(row[1]), -row[2], row[0].title))
    return rows


def follow_ups(comparisons: list[Comparison]) -> list[tuple[Cause, list[str]]]:
    """The rule gaps with their queries, by their likely impact: the highest
    impact on one of their queries first, then the most time that DuckDB's
    shape saves on our engine, then the most queries."""
    rows = []
    for cause, names, _ in cause_rows(comparisons):
        if cause.classification != RULE_GAP:
            continue
        notes = [NOTES.get((name, cause.id), ("none", None, "")) for name in names]
        highest = max(IMPACT_SCORES[impact] for impact, _, _ in notes)
        saved = -sum(min(delta or 0.0, 0.0) for _, delta, _ in notes)
        rows.append(((-highest, -saved, -len(names), cause.title), cause, names))
    rows.sort(key=lambda row: row[0])
    return [(cause, names) for _, cause, names in rows]


def impact_text(cause: Cause, comparisons: list[Comparison]) -> str:
    counts = collections.Counter(
        NOTES.get((c.query, cause.id), ("none", None, ""))[0]
        for c in comparisons
        for d in c.divergences
        if d.cause.id == cause.id
    )
    return ", ".join(f"{counts[i]} {i}" for i in IMPACTS if counts[i])


def markdown_table(
    header: list[str], rows: list[list[str]], right: set[int] = frozenset()
) -> list[str]:
    """A Markdown table, formatted as prettier formats it."""
    widths = [len(cell) for cell in header]
    for row in rows:
        widths = [max(width, len(cell)) for width, cell in zip(widths, row)]
    widths = [max(width, 3) for width in widths]

    def line(cells):
        padded = [
            cell.rjust(width) if i in right else cell.ljust(width)
            for i, (cell, width) in enumerate(zip(cells, widths))
        ]
        return "| " + " | ".join(padded) + " |"

    separator = (
        "| "
        + " | ".join(
            ("-" * (width - 1) + ":") if i in right else "-" * width
            for i, width in enumerate(widths)
        )
        + " |"
    )
    return [line(header), separator, *(line(row) for row in rows)]


def issue_link(cause: Cause) -> str:
    return f"#{cause.issue}" if cause.issue else "-"


def summary(comparisons: list[Comparison]) -> str:
    """The summary that the differ prints and that the report starts with."""
    verdicts = collections.Counter(c.verdict for c in comparisons)
    lines = [
        f"Of the {len(comparisons)} queries, {verdicts['matches']} match DuckDB's plan "
        f"shape, {verdicts[RULE_GAP]} diverge through a rule gap and "
        f"{verdicts[COST]} only through cost or statistics.",
        "",
        "The divergences, the most common first (a query may have several):",
        "",
    ]
    rows = []
    for cause, names, _ in cause_rows(comparisons):
        rows.append(
            [
                cause.title,
                cause.classification,
                str(len(names)),
                ", ".join(names),
                impact_text(cause, comparisons),
                issue_link(cause),
            ]
        )
    lines += markdown_table(
        [
            "Divergence",
            "Classification",
            "Queries",
            "Which",
            "Likely impact",
            "Follow-up",
        ],
        rows,
        right={2},
    )
    return "\n".join(lines)


#: DuckDB's `PRAGMA version` of the build of `build_duckdb.sh` (see README.md).
DUCKDB_VERSION = "v2.1.0-dev93719"

#: The commit of `datafusion/new_ex` whose plans `pipeline/` holds.
PIPELINE_COMMIT = "29c9d0687"

LICENSE = """<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->"""


def side_by_side(duckdb: str, ours: str) -> list[str]:
    """Two excerpts side by side: an HTML table of two code blocks."""
    return [
        "<table>",
        "<tr><th>DuckDB</th><th>ours</th></tr>",
        "<tr>",
        "<td>",
        "",
        "```text",
        duckdb or "(none)",
        "```",
        "",
        "</td>",
        "<td>",
        "",
        "```text",
        ours or "(none)",
        "```",
        "",
        "</td>",
        "</tr>",
        "</table>",
    ]


def shape(fp: Fingerprint) -> str:
    """A one-line summary of a fingerprint."""
    joins = collections.Counter(f"{j.operator} join" for j in fp.joins)
    parts = [plural(len(fp.scans), "scan")]
    parts += [plural(n, kind) for kind, n in sorted(joins.items())]
    if fp.aggregates:
        parts.append(plural(len(fp.aggregates), "aggregation"))
    orders = collections.Counter(
        {"topn": "top-N", "sort": "sort", "limit": "limit"}[o.kind] for o in fp.orders
    )
    parts += [plural(n, kind) for kind, n in sorted(orders.items())]
    breaks = ", ".join(plural(n, kind) for kind, n in sorted(fp.breaks.items()))
    return "; ".join(parts) + f"; pipeline breaks: {breaks or 'none'}"


def query_section(comparison: Comparison) -> list[str]:
    lines = [f"### {comparison.query}: {comparison.verdict}", ""]
    lines += [
        f"- DuckDB: {shape(comparison.duckdb)}.",
        f"- Ours: {shape(comparison.ours)}.",
        "",
    ]
    for number_, divergence in enumerate(comparison.divergences, 1):
        cause = divergence.cause
        impact, _, note = NOTES.get(
            (comparison.query, cause.id), ("none", None, "(no note)")
        )
        follow_up = f", follow-up {issue_link(cause)}" if cause.issue else ""
        lines += [
            f"**{number_}. {cause.title}** ({cause.classification}{follow_up}; likely "
            f"impact: {impact})",
            "",
        ]
        for finding in divergence.findings:
            lines.append(f"- {finding.component}: {finding.text}")
        lines.append("")
        lines += side_by_side(
            excerpt(comparison.duckdb.plan, divergence.ops(DUCKDB)),
            excerpt(comparison.ours.plan, divergence.ops(PIPELINE)),
        )
        lines += ["", f"Likely cost: {note}", ""]
    return lines


def report(comparisons: list[Comparison]) -> str:
    """docs/planner-comparison.md."""
    lines = [LICENSE, "", *REPORT_INTRO.strip("\n").split("\n"), ""]
    lines += ["## Summary", "", summary(comparisons), ""]
    lines += [*REPORT_FOLLOW_UPS.strip("\n").split("\n"), ""]
    for index, (cause, names) in enumerate(follow_ups(comparisons), 1):
        costly = [
            name
            for name in names
            if NOTES.get((name, cause.id), ("none", None, ""))[0] != "none"
        ]
        issue = f"{issue_link(cause)}: " if cause.issue else ""
        lines.append(
            f"{index}. {issue}**{cause.title}**: costs "
            f"{', '.join(costly) or 'none of the queries'} (impact on "
            f"{', '.join(names)}: {impact_text(cause, comparisons)}); would change "
            f"{cause.change}."
        )
    lines.append("")
    lines += [*REPORT_METHOD.strip("\n").split("\n"), ""]
    lines += ["### The causes", ""]
    for cause in sorted(
        CAUSES.values(), key=lambda c: (c.classification != RULE_GAP, c.title)
    ):
        lines += [
            f"- **{cause.title}** ({cause.classification}"
            + (f", {issue_link(cause)}" if cause.issue else "")
            + ")",
            f"  - DuckDB: {cause.duckdb}",
            f"  - Ours: {cause.ours}",
        ]
        if cause.change:
            lines.append(f"  - A fix would change {cause.change}.")
    lines.append("")
    lines += [*REPORT_INHERENT.strip("\n").split("\n"), ""]
    lines += ["## The queries", ""]
    rows = [
        [
            c.query,
            c.verdict,
            ", ".join(d.cause.title for d in c.divergences) or "-",
        ]
        for c in comparisons
    ]
    lines += markdown_table(["Query", "Verdict", "Divergences"], rows)
    lines.append("")
    for comparison in comparisons:
        if comparison.divergences:
            lines += query_section(comparison)
    while lines and lines[-1] == "":
        lines.pop()
    return "\n".join(lines) + "\n"


REPORT_INTRO = f"""
# Our plans compared with DuckDB 2.0's

The pipeline engine (`datafusion/physical-pipeline`) executes plans the way
DuckDB does: morsel-driven pipelines, with implicit parallelism. So DuckDB's
plans of the same queries over the same data show where our planner (DataFusion's
optimizer, then the lowering of `PipelineQueryPlanner`) leaves performance on
the table. This report compares the plans of the 47 queries of the engine's
benchmark (`datafusion/physical-pipeline/benches/pipeline_vs_default.rs`: the
22 TPC-H queries and 25 ClickBench queries), plan shapes only: no timings of
DuckDB, no `EXPLAIN ANALYZE`.

- **DuckDB**: the head of `main` of `github.com/duckdb/duckdb`, commit
  `{DUCKDB_COMMIT}` (`PRAGMA version`: `{DUCKDB_VERSION}`, DuckDB 2), built
  and captured as `benchmarks/plan-compare/README.md` describes. Its plans are
  `benchmarks/plan-compare/duckdb/<query>.json` (`EXPLAIN (FORMAT JSON)`).
- **Ours**: the plans of `datafusion/new_ex` at `{PIPELINE_COMMIT}`, as the
  benchmark's sessions plan them (16 target partitions, trusting DataFusion's
  built-in functions), from the dump of the profile of their execution
  (`docs/profiling.md` of the crate): `benchmarks/plan-compare/pipeline/<query>.json`.
- **Data**: the Parquet files of the benchmark (TPC-H SF 0.1 and 1,000,000
  `hits` rows, 16 files per table), which both engines read, on 16 threads.

`benchmarks/plan-compare/differ.py` normalizes both plans of every query into
one fingerprint, compares them, classifies every divergence and writes this
report (`differ.py --report`; `differ.py` prints the same summary, and
`differ.py --check` fails if this report is not the one that the plans give).
The notes on what every divergence likely costs are the `NOTES` of the differ.
"""

REPORT_FOLLOW_UPS = """
## Follow-up issues

The rule gaps, ordered by their likely impact on these queries (the highest
impact on one of their queries first, then the time that DuckDB's shape saves on
our engine), each filed as an issue that names the queries it costs and the
files it would change. None of them is implemented here.
"""

REPORT_METHOD = """
## How the plans are compared

Both plans are read from their machine-readable form: DuckDB's JSON plan (the
`name`, `children` and `extra_info` of every operator) and our dump (the
`kind`, `inputs` and pipelines of every operator, and its description). The
differ only takes apart the details that both render as text (the lists of
projections, the conjuncts of filters, the keys of aggregations, joins and
sorts) and finds the columns that they reference; it never compares their
formatting. It normalizes both plans into one model, in which

- projections are left out (both engines place them differently and they do not
  change the shape), and so are our `CachingOperator`s (which coalesce small
  chunks) and the pipelines that finalize the hash table of a join;
- the filters directly above a scan are filters of the scan: our lowering folds
  them into the Parquet scan, DuckDB evaluates them in the pipeline of the scan
  (its scans take filters of one column only);
- an uncorrelated scalar subquery is a subquery that is evaluated once, and the
  comparison with its value a filter: DuckDB joins the single row of the
  subquery (a `NESTED_LOOP_JOIN` or `HASH_JOIN` on `SUBQUERY`), DataFusion
  evaluates it first (`PhysicalScalarSubquery`);
- a `MARK` join of an `IN` list (DuckDB's `IN` clause rewriter turns long lists
  into a join with a `COLUMN_DATA_SCAN` of the constants) is the `IN` filter,
  and a `MARK` join of an `IN` or `NOT IN` subquery with the `FILTER` of its
  marker is a semi or an anti join;
- a `PhysicalLimit` with an offset above a TopK of offset plus fetch rows is a
  top-N with an offset, as DuckDB's `TOP_N`.

The fingerprint of a plan has, for every scan, its table, the columns it reads,
the columns its filters compare with constants (which it can prune row groups
with) and whether a top-N pushes its dynamic filter into it; every join, by its
inputs (the join order, whatever the build side), with its operator (hash,
nested loop, ...), its kind and its build side; every aggregation, by the tables
of its input and its number of keys; every top-N, sort and limit; and the
pipeline breaks: the sinks of our pipelines, and the sinks that DuckDB's
operators make (the build side of a join, aggregations, sorts, top-N and shared
subplans; its `EXPLAIN` does not show its pipelines). Both engines aggregate in
one operator, partially in every thread and then finally: the placement of an
aggregation is the tables below it.

Every difference of the fingerprints is a finding, and the findings of a query
are grouped by their cause into its divergences: the rules of DuckDB that change
the shape of a part of the plan (late materialization, shared common table
expressions, decorrelation by the outer keys) explain all the differences of
that part, and so do the exact statistics that let our planner answer a query
without scanning; then a join that both plans have but build on different
inputs is a cost/statistics-driven divergence, and different joins are a
different join order, a rule gap. Every cause is a rule gap (our optimizer
lacks a rule that DuckDB has: a planner fix) or cost/statistics-driven (the
rules agree but the choice differs, through estimates or statistics; also when
our plan is ahead through statistics that DuckDB does not use). A query
matches if its fingerprints do, is a rule gap if one of its divergences is, and
is cost/statistics-driven otherwise.
"""

REPORT_INHERENT = """
## What is inherent and not worth chasing

- **Exchanges.** DuckDB's plans have no repartition or exchange operators: its
  parallelism is implicit, every pipeline runs on all threads and its sinks
  (hash tables, aggregations, sorts) are shared by them. DataFusion's plans
  have `RepartitionExec`, `CoalescePartitionsExec` and `CoalesceBatchesExec`
  nodes, and two-stage aggregations, but our lowering removes them (see
  `datafusion/physical-pipeline/src/planner.rs`): our plans have no exchanges
  either, and the comparison has nothing to normalize there.
- **Operator names.** The engines name their operators differently; the
  comparison compares what they do, not their names: `READ_PARQUET` is our
  `ParquetScanSource`, `HASH_JOIN` our `PhysicalHashJoin` (whose build side is
  the input its pipeline sinks into; DuckDB's is its right child),
  `HASH_GROUP_BY` and `UNGROUPED_AGGREGATE` our `PhysicalHashAggregate`,
  `TOP_N` our `PhysicalTopK`, `ORDER_BY` our `PhysicalSort`, `FILTER` our
  `PhysicalFilter` or the filter folded into a scan. Join types are named from
  the probe side in DuckDB (`RIGHT_SEMI` emits the rows of the build side) and
  from the build side in DataFusion (`LeftSemi`), and the comparison compares
  what they keep.
- **Projections**, which both engines place differently and evaluate in the
  pipeline of their input, and our `CachingOperator`s; the rewrites of
  aggregate functions (DuckDB's `avg` as a `sum` and a `count`, or a `sum` of
  `x + k` as a `sum` and a `count`); and the representations of the same work:
  scalar subqueries as joins of one row in DuckDB, long `IN` lists as `MARK`
  joins, `IN` and `NOT IN` subqueries as `MARK` joins.
- **DuckDB's `optional:` filters** (`IN` lists and the dynamic filters of top-N),
  which its scans only prune with, and which a filter above the scan evaluates:
  our scans prune with the same filters and evaluate them.

## What the comparison cannot see

- **Join filters.** DuckDB pushes the bounds of the build side of its hash joins
  into the scans of the probe side at run time (`JoinFilterPushdown`), but its
  `EXPLAIN` does not show them; our plans show the dynamic filters of our hash
  joins (`DynamicFilter [...]`, bounds and membership), which our lowering
  routes to the scans below their probe side. The comparison leaves both out.
- **Cardinalities.** DuckDB's rows are its estimates (`~N rows`), ours the rows
  that our operators produced in the execution of the dump (`(N rows)`), after
  the dynamic filters: they vary a little from run to run below dynamic filters.
- **DuckDB's run-time choices**, such as whether a shared subplan is
  materialized or streamed to its readers (`Execution Mode: PIPELINE_DEPENDENT`),
  which DuckDB decides when it builds its pipelines.
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--duckdb",
        type=pathlib.Path,
        default=HERE / "duckdb",
        help="DuckDB's plans (default: %(default)s)",
    )
    parser.add_argument(
        "--pipeline",
        type=pathlib.Path,
        default=HERE / "pipeline",
        help="our plans (default: %(default)s)",
    )
    parser.add_argument(
        "--queries",
        type=pathlib.Path,
        default=HERE / "queries.sql",
        help="the queries (default: %(default)s)",
    )
    parser.add_argument(
        "--report",
        nargs="?",
        type=pathlib.Path,
        const=REPORT,
        help="write the report (default: %(const)s)",
    )
    parser.add_argument(
        "--check",
        nargs="?",
        type=pathlib.Path,
        const=REPORT,
        help="fail if the report is not up to date (default: %(const)s)",
    )
    parser.add_argument(
        "--verbose", action="store_true", help="print the findings of every query"
    )
    args = parser.parse_args()
    queries = parse_queries(args.queries.read_text(encoding="utf-8"))
    comparisons = [load(name, sql, args.duckdb, args.pipeline) for name, sql in queries]
    problems = check_notes(comparisons)
    for comparison in comparisons:
        causes = ", ".join(d.cause.id for d in comparison.divergences)
        print(
            f"{comparison.query:<7} {comparison.verdict}"
            + (f": {causes}" if causes else "")
        )
        if args.verbose:
            for divergence in comparison.divergences:
                for finding in divergence.findings:
                    print(
                        f"        [{divergence.cause.id}] {finding.component}: "
                        f"{finding.text}"
                    )
    print()
    print(summary(comparisons))
    if args.report:
        args.report.write_text(report(comparisons), encoding="utf-8")
        print(f"\nWrote {args.report}")
    if args.check:
        if not args.check.exists() or args.check.read_text(encoding="utf-8") != report(
            comparisons
        ):
            problems.append(f"{args.check} is not up to date: run differ.py --report")
    if problems:
        sys.exit("error:\n" + "\n".join(f"  {problem}" for problem in problems))


if __name__ == "__main__":
    main()
