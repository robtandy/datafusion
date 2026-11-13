<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

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
  `5d254ad870763a9724af68b937ebf4e1440ad040` (`PRAGMA version`: `v2.1.0-dev93719`, DuckDB 2), built
  and captured as `benchmarks/plan-compare/README.md` describes. Its plans are
  `benchmarks/plan-compare/duckdb/<query>.json` (`EXPLAIN (FORMAT JSON)`).
- **Ours**: the plans of `datafusion/new_ex` at `c1d4a6f6e`, as the
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

## Summary

Of the 47 queries, 28 match DuckDB's plan shape, 16 diverge through a rule gap and 3 only through cost or statistics.

Only 4 divergences likely cost something on these queries, measured on our engine: q15 (common table expressions computed once per reference: high), q18 (join order: low), q21 (join order: medium), hits23 (late materialization of top-N: high). The other divergences cost nothing measurable here, or our plan is ahead.

The divergences, the most common first (a query may have several):

| Divergence                                             | Classification         | Queries | Which                                  | Likely impact           | Follow-up |
| ------------------------------------------------------ | ---------------------- | ------: | -------------------------------------- | ----------------------- | --------- |
| Join order                                             | rule gap               |       9 | q2, q5, q7, q8, q9, q10, q11, q18, q21 | 1 medium, 1 low, 7 none | #115      |
| Correlated subqueries decorrelated by their outer keys | rule gap               |       3 | q2, q20, q22                           | 3 none                  | #116      |
| `IS NOT NULL` filters of join keys                     | rule gap               |       3 | q4, q17, q21                           | 3 none                  | #117      |
| Aggregation answered from exact statistics             | cost/statistics-driven |       2 | hits0, hits6                           | 2 none                  | -         |
| Common table expressions computed once per reference   | rule gap               |       1 | q15                                    | 1 high                  | #114      |
| Late materialization of top-N                          | rule gap               |       1 | hits23                                 | 1 high                  | #113      |
| Build side of a join                                   | cost/statistics-driven |       1 | q14                                    | 1 none                  | -         |
| Constant bounds derived through comparisons of columns | rule gap               |       1 | q12                                    | 1 none                  | #118      |

## Follow-up issues

The rule gaps, ordered by their likely impact on these queries (the highest
impact on one of their queries first, then the time that DuckDB's shape saves on
our engine), each filed as an issue that names the queries it costs and the
files it would change. None of them is implemented here.

1. #113: **Late materialization of top-N**: costs hits23 (impact on hits23: 1 high); would change `datafusion/physical-pipeline/src/planner.rs` (the lowering of a TopK over a Parquet scan) and `datafusion/physical-pipeline/src/sources/parquet.rs` (row ids, and a scan of given rows).
2. #114: **Common table expressions computed once per reference**: costs q15 (impact on q15: 1 high); would change `datafusion/sql/src/cte.rs` (plan a common table expression that the query reads more than once as a `MaterializedCte`, which the pipeline engine lowers natively).
3. #115: **Join order**: costs q18, q21 (impact on q2, q5, q7, q8, q9, q10, q11, q18, q21: 1 medium, 1 low, 7 none); would change `datafusion/physical-optimizer/src/` (a new join reordering rule next to `join_selection.rs`, whose statistics it would use, registered in `datafusion/physical-optimizer/src/optimizer.rs`).
4. #116: **Correlated subqueries decorrelated by their outer keys**: costs none of the queries (impact on q2, q20, q22: 3 none); would change `datafusion/optimizer/src/scalar_subquery_to_join.rs`, `datafusion/optimizer/src/decorrelate_predicate_subquery.rs` and `datafusion/optimizer/src/decorrelate.rs`.
5. #117: **`IS NOT NULL` filters of join keys**: costs none of the queries (impact on q4, q17, q21: 3 none); would change `datafusion/optimizer/src/filter_null_join_keys.rs` (or the defaults of the sessions of the pipeline engine).
6. #118: **Constant bounds derived through comparisons of columns**: costs none of the queries (impact on q12: 1 none); would change `datafusion/optimizer/src/push_down_filter.rs` (or a rule of `datafusion/optimizer/src/simplify_expressions/`).

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

### Likely costs

What a divergence likely costs is measured on our engine, where SQL can give
the query the shape of DuckDB's plan: `benchmarks/plan-compare/experiments.sql`
has DuckDB's join orders written out with explicit joins, the common table
expression of q15 `MATERIALIZED` (which DataFusion plans as a
`MaterializedCteExec`, which our engine executes natively), and the top-N of
hits23 over its sort keys only (what DuckDB's first scan reads, an upper bound
of what late materialization saves). `benchmarks/plan-compare/measure.py` runs
them with the benchmark's queries on our engine (an optimized build of
`dump_plans`, 16 threads, profiling on) and the notes quote the medians of 36
executions of every query. The impact of a divergence is high if DuckDB's
shape is at least 25% faster, medium at 10-25%, low at 3-10%, and none below 3%
(the noise of these measurements), if it is slower, if the divergence removes
no row on this data, or if our plan is ahead. These are timings of our engine
only: no timing of DuckDB.

### The causes

- **Common table expressions computed once per reference** (rule gap, #114)
  - DuckDB: Its `CTEInlining` (`src/optimizer/cte_inlining.cpp`) keeps a common table expression that the query reads more than once and that ends in an aggregation as one shared subplan (`CTE`, `CTE_SCAN`), which it computes once and materializes or streams to its readers.
  - Ours: DataFusion inlines every common table expression that is not `MATERIALIZED`, so the plan computes it once per reference.
  - A fix would change `datafusion/sql/src/cte.rs` (plan a common table expression that the query reads more than once as a `MaterializedCte`, which the pipeline engine lowers natively).
- **Constant bounds derived through comparisons of columns** (rule gap, #118)
  - DuckDB: Its `FilterCombiner` (`src/optimizer/filter_combiner.cpp`) derives constant bounds through chains of comparisons of columns (`a < b AND b < c AND c < k` implies `a < k` and `b < k`), which its scans prune with.
  - Ours: DataFusion pushes the comparisons as they are: the scan prunes with the bounds of the query only.
  - A fix would change `datafusion/optimizer/src/push_down_filter.rs` (or a rule of `datafusion/optimizer/src/simplify_expressions/`).
- **Correlated subqueries decorrelated by their outer keys** (rule gap, #116)
  - DuckDB: It decorrelates a correlated subquery with a duplicate-eliminated join (`src/planner/subquery/flatten_dependent_join.cpp`, then `Deliminator`): it computes the outer rows once (a shared subplan `__duckdb_delim_<n>`), evaluates the subquery only for their distinct correlation keys, and joins its result back.
  - Ours: DataFusion decorrelates the subquery into a join with an aggregation of the subquery over all of its rows (`ScalarSubqueryToJoin`, `DecorrelatePredicateSubquery`); the dynamic filters of our hash joins then restrict the scans of the subquery to the outer keys at run time.
  - A fix would change `datafusion/optimizer/src/scalar_subquery_to_join.rs`, `datafusion/optimizer/src/decorrelate_predicate_subquery.rs` and `datafusion/optimizer/src/decorrelate.rs`.
- **Join order** (rule gap, #115)
  - DuckDB: Its join order optimizer (`src/optimizer/join_order/`) enumerates the join orders of every block of inner, semi and anti joins (DPhyp) and picks the cheapest by its cardinality estimates, bushy trees included.
  - Ours: DataFusion has no join reordering: the joins of a query are planned in the order of its `FROM` clause and of its subqueries (`EliminateCrossJoin` and `PushDownFilter` turn the cross joins of the `FROM` clause into inner joins in that order, the decorrelated subqueries are joined last), and `JoinSelection` only chooses the build side of every join.
  - A fix would change `datafusion/physical-optimizer/src/` (a new join reordering rule next to `join_selection.rs`, whose statistics it would use, registered in `datafusion/physical-optimizer/src/optimizer.rs`).
- **Late materialization of top-N** (rule gap, #113)
  - DuckDB: Its `LateMaterialization` (`src/optimizer/late_materialization.cpp`) plans a top-N over a scan of many columns as a top-N over the sort keys and the row ids (`file_index`, `file_row_number`), and a second scan that reads the other columns of the top rows only (a semi join on the row ids).
  - Ours: Our TopK reads every column of every row that passes the filters of its scan (its dynamic filter skips rows, but only once it has its first rows).
  - A fix would change `datafusion/physical-pipeline/src/planner.rs` (the lowering of a TopK over a Parquet scan) and `datafusion/physical-pipeline/src/sources/parquet.rs` (row ids, and a scan of given rows).
- **`IS NOT NULL` filters of join keys** (rule gap, #117)
  - DuckDB: Its filter pushdown (`src/optimizer/pushdown/`) adds an `IS NOT NULL` filter of every nullable key of an inner or a semi join, and of the correlation keys of a decorrelated subquery, and pushes it into the scans.
  - Ours: DataFusion's `FilterNullJoinKeys` (`datafusion/optimizer/src/filter_null_join_keys.rs`) does so only when `datafusion.optimizer.filter_null_join_keys` is on, which it is not by default; the hash joins skip the rows of null keys themselves.
  - A fix would change `datafusion/optimizer/src/filter_null_join_keys.rs` (or the defaults of the sessions of the pipeline engine).
- **Aggregation answered from exact statistics** (cost/statistics-driven)
  - DuckDB: It scans the table (its estimate of the rows of the scan is exact, but it plans the scan).
  - Ours: DataFusion's `AggregateStatistics` answers `COUNT(*)`, `MIN` and `MAX` from the exact statistics of the Parquet metadata (the row counts and the minima and maxima of the columns): the plan reads no file. Our plan is ahead, through statistics: not a planner fix.
- **Build side of a join** (cost/statistics-driven)
  - DuckDB: Its `BuildProbeSideOptimizer` builds on the input with the smaller estimated cardinality.
  - Ours: DataFusion's `JoinSelection` builds on the input with the smaller estimated size (statistics of rows and bytes; with no estimate it keeps the order of the query).

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

## The queries

| Query  | Verdict                | Divergences                                                        |
| ------ | ---------------------- | ------------------------------------------------------------------ |
| q1     | matches                | -                                                                  |
| q2     | rule gap               | Correlated subqueries decorrelated by their outer keys, Join order |
| q3     | matches                | -                                                                  |
| q4     | rule gap               | `IS NOT NULL` filters of join keys                                 |
| q5     | rule gap               | Join order                                                         |
| q6     | matches                | -                                                                  |
| q7     | rule gap               | Join order                                                         |
| q8     | rule gap               | Join order                                                         |
| q9     | rule gap               | Join order                                                         |
| q10    | rule gap               | Join order                                                         |
| q11    | rule gap               | Join order                                                         |
| q12    | rule gap               | Constant bounds derived through comparisons of columns             |
| q13    | matches                | -                                                                  |
| q14    | cost/statistics-driven | Build side of a join                                               |
| q15    | rule gap               | Common table expressions computed once per reference               |
| q16    | matches                | -                                                                  |
| q17    | rule gap               | `IS NOT NULL` filters of join keys                                 |
| q18    | rule gap               | Join order                                                         |
| q19    | matches                | -                                                                  |
| q20    | rule gap               | Correlated subqueries decorrelated by their outer keys             |
| q21    | rule gap               | `IS NOT NULL` filters of join keys, Join order                     |
| q22    | rule gap               | Correlated subqueries decorrelated by their outer keys             |
| hits0  | cost/statistics-driven | Aggregation answered from exact statistics                         |
| hits1  | matches                | -                                                                  |
| hits2  | matches                | -                                                                  |
| hits4  | matches                | -                                                                  |
| hits5  | matches                | -                                                                  |
| hits6  | cost/statistics-driven | Aggregation answered from exact statistics                         |
| hits7  | matches                | -                                                                  |
| hits8  | matches                | -                                                                  |
| hits10 | matches                | -                                                                  |
| hits12 | matches                | -                                                                  |
| hits14 | matches                | -                                                                  |
| hits15 | matches                | -                                                                  |
| hits16 | matches                | -                                                                  |
| hits20 | matches                | -                                                                  |
| hits21 | matches                | -                                                                  |
| hits23 | rule gap               | Late materialization of top-N                                      |
| hits25 | matches                | -                                                                  |
| hits27 | matches                | -                                                                  |
| hits29 | matches                | -                                                                  |
| hits30 | matches                | -                                                                  |
| hits32 | matches                | -                                                                  |
| hits33 | matches                | -                                                                  |
| hits35 | matches                | -                                                                  |
| hits36 | matches                | -                                                                  |
| hits42 | matches                | -                                                                  |

### q2: rule gap

- DuckDB: 9 scans; 9 hash joins; 2 aggregations; 1 top-N; pipeline breaks: 2 aggregations, 9 join builds, 1 shared subplan, 1 top-N.
- Ours: 9 scans; 8 hash joins; 1 aggregation; 1 top-N; pipeline breaks: 1 aggregation, 8 join builds, 1 top-N.

**1. Correlated subqueries decorrelated by their outer keys** (rule gap, follow-up #116; likely impact: none)

- joins: DuckDB joins the shared outer rows with the subquery evaluated for their keys; ours joins the outer rows with the subquery over all of its rows
- aggregations: the aggregations differ: DuckDB has an aggregation by 1 key of `nation` (2), `part`, `partsupp` (2), `region` (2), `supplier` (2), an aggregation by 1 key of `nation`, `part`, `partsupp`, `region`, `supplier` that ours has not, ours an aggregation by 1 key of `nation`, `partsupp`, `region`, `supplier` that DuckDB has not
- pipeline breaks: the pipeline breaks differ (DuckDB / ours): aggregation 2 / 1, join build 9 / 8, shared subplan 1 / 0

<table>
<tr><th>DuckDB</th><th>ours</th></tr>
<tr>
<td>

```text
  TOP_N Top: 100; Order By: s_acctbal DESC, n_name ASC, s_name ASC, p_partkey ASC ~0 rows
*   CTE CTE Name: __duckdb_delim_79; Table Index: 79; Execution Mode: PIPELINE_DEPENDENT ~0 rows
      ...
*     HASH_JOIN Join Type: INNER; Conditions: p_partkey IS NOT DISTINCT FROM p_partkey, ps_supplycost = SUBQUERY ~0 rows
        CTE_SCAN CTE Index: 79 ~0 rows
*       HASH_GROUP_BY Groups: #0; Aggregates: min(#1) ~0 rows
*         HASH_JOIN Join Type: INNER; Conditions: ps_suppkey = s_suppkey ~0 rows
*           HASH_JOIN Join Type: SEMI; Conditions: ps_partkey = p_partkey ~0 rows
              READ_PARQUET Projections: ps_partkey, ps_suppkey, ps_supplycost ~80,000 rows
*             HASH_GROUP_BY Groups: #0 ~0 rows
                ...
            HASH_JOIN Join Type: INNER; Conditions: s_nationkey = n_nationkey ~0 rows
              ...
```

</td>
<td>

```text
  PhysicalTopK: fetch=100, expr=[s_acctbal@0 DESC, n_name@2 ASC NULLS LAST, s_name@1 ASC NULLS LAST, p_partkey@3 ASC NULLS LAST], dynamic_filter=true (48 rows)
*   PhysicalHashJoin: join_type=LeftSemi, on=[(p_partkey@0, ps_partkey@1), (ps_supplycost@7, min(partsupp.ps_supplycost)@0)] (48 rows)
      PhysicalHashJoin: join_type=Inner, on=[(r_regionkey@0, n_regionkey@9)] (66 rows)
        ...
*     PhysicalHashAggregate: gby=[ps_partkey@0 as ps_partkey], aggr=[min(partsupp.ps_supplycost)] (48 rows)
        ...
```

</td>
</tr>
</table>

Likely cost: Our dynamic join filters already restrict the subquery to the part keys of the outer rows: its `partsupp` scan returns 66 of 80,000 rows. Both plans scan the four tables of the subquery once more: nothing to gain.

**2. Join order** (rule gap, follow-up #115; likely impact: none)

- joins: the join order of the rest of the query differs: DuckDB joins `((((nation ⋈ region) ⋈ supplier) ⋈ partsupp) ⋈ part)`, `(((nation ⋈ region) ⋈ supplier) ⋈ partsupp)`, `((nation ⋈ region) ⋈ supplier)`, `(nation ⋈ region)`, `((nation ⋈ region) ⋈ supplier)`, `(nation ⋈ region)`; ours joins `((((part ⋈ partsupp) ⋈ supplier) ⋈ nation) ⋈ region)`, `(((part ⋈ partsupp) ⋈ supplier) ⋈ nation)`, `((part ⋈ partsupp) ⋈ supplier)`, `(part ⋈ partsupp)`, `(((partsupp ⋈ supplier) ⋈ nation) ⋈ region)`, `((partsupp ⋈ supplier) ⋈ nation)`, `(partsupp ⋈ supplier)`

<table>
<tr><th>DuckDB</th><th>ours</th></tr>
<tr>
<td>

```text
  TOP_N Top: 100; Order By: s_acctbal DESC, n_name ASC, s_name ASC, p_partkey ASC ~0 rows
    CTE CTE Name: __duckdb_delim_79; Table Index: 79; Execution Mode: PIPELINE_DEPENDENT ~0 rows
*     HASH_JOIN Join Type: INNER; Conditions: p_partkey = ps_partkey ~0 rows
        READ_PARQUET Projections: p_partkey, p_mfgr; Filters: suffix(p_type, 'BRASS'), (p_size = 15) ~4,000 rows
*       HASH_JOIN Join Type: INNER; Conditions: ps_suppkey = s_suppkey ~0 rows
          READ_PARQUET Projections: ps_partkey, ps_suppkey, ps_supplycost ~80,000 rows
*         HASH_JOIN Join Type: INNER; Conditions: s_nationkey = n_nationkey ~0 rows
            READ_PARQUET Projections: s_suppkey, s_name, s_address, s_nationkey, s_phone, s_acctbal, s_comment ~992 rows
*           HASH_JOIN Join Type: INNER; Conditions: n_regionkey = r_regionkey ~0 rows
              READ_PARQUET Projections: n_nationkey, n_name, n_regionkey ~16 rows
              READ_PARQUET Projections: r_regionkey; Filters: (r_name = 'EUROPE') ~0 rows
      HASH_JOIN Join Type: INNER; Conditions: p_partkey IS NOT DISTINCT FROM p_partkey, ps_supplycost = SUBQUERY ~0 rows
        ...
        HASH_GROUP_BY Groups: #0; Aggregates: min(#1) ~0 rows
          HASH_JOIN Join Type: INNER; Conditions: ps_suppkey = s_suppkey ~0 rows
            ...
*           HASH_JOIN Join Type: INNER; Conditions: s_nationkey = n_nationkey ~0 rows
              READ_PARQUET Projections: s_suppkey, s_nationkey ~992 rows
*             HASH_JOIN Join Type: INNER; Conditions: n_regionkey = r_regionkey ~0 rows
                READ_PARQUET Projections: n_nationkey, n_regionkey ~16 rows
                READ_PARQUET Projections: r_regionkey; Filters: (r_name = 'EUROPE') ~0 rows
```

</td>
<td>

```text
  PhysicalTopK: fetch=100, expr=[s_acctbal@0 DESC, n_name@2 ASC NULLS LAST, s_name@1 ASC NULLS LAST, p_partkey@3 ASC NULLS LAST], dynamic_filter=true (48 rows)
    PhysicalHashJoin: join_type=LeftSemi, on=[(p_partkey@0, ps_partkey@1), (ps_supplycost@7, min(partsupp.ps_supplycost)@0)] (48 rows)
*     PhysicalHashJoin: join_type=Inner, on=[(r_regionkey@0, n_regionkey@9)] (66 rows)
        ParquetScanSource: projection=[r_regionkey, r_name], predicate=r_name@1 = EUROPE, filter=r_name@1 = EUROPE, projection=[r_regionkey@0] (1 row)
*       PhysicalHashJoin: join_type=Inner, on=[(n_nationkey@0, s_nationkey@4)] (66 rows)
          ParquetScanSource: projection=[n_nationkey, n_name, n_regionkey], predicate=DynamicFilter [ n_regionkey@2 >= 3 AND n_regionkey@2 <= 3 ] (5 rows)
*         PhysicalHashJoin: join_type=Inner, on=[(s_suppkey@0, ps_suppkey@2)] (66 rows)
            ParquetScanSource: projection=[s_suppkey, s_name, s_address, s_nationkey, s_phone, s_acctbal, s_comment], predicate=DynamicFilter [ s_nationkey@3 >= 6 AND s_nationkey@3 <= 23 AND s_nationkey@3 IN (SET) ] (207 rows)
*           PhysicalHashJoin: join_type=Inner, on=[(p_partkey@0, ps_partkey@0)] (66 rows)
              ParquetScanSource: projection=[p_partkey, p_mfgr, p_type, p_size], predicate=p_size@5 = 15 AND p_type@4 LIKE %BRASS, filter=p_size@3 = 15 AND p_type@2 LIKE %BRASS, projection=[p_partkey@0, p_mfgr@1] (78 rows)
              ParquetScanSource: projection=[ps_partkey, ps_suppkey, ps_supplycost], predicate=DynamicFilter [ ps_partkey@0 >= 331 AND ps_partkey@0 <= 19800 AND ps_partkey@0 IN (SET) ] AND DynamicFilter [ ps_suppkey@1 >= 7 AND ps_suppkey@1 <= 1000 AND hash_lookup(ps_suppkey@1) ] (66 rows)
      PhysicalHashAggregate: gby=[ps_partkey@0 as ps_partkey], aggr=[min(partsupp.ps_supplycost)] (48 rows)
*       PhysicalHashJoin: join_type=RightSemi, on=[(r_regionkey@0, n_regionkey@2)] (66 rows)
          ParquetScanSource: projection=[r_regionkey, r_name], predicate=r_name@1 = EUROPE, filter=r_name@1 = EUROPE, projection=[r_regionkey@0] (1 row)
*         PhysicalHashJoin: join_type=Inner, on=[(n_nationkey@0, s_nationkey@2)] (66 rows)
            ParquetScanSource: projection=[n_nationkey, n_regionkey], predicate=DynamicFilter [ n_regionkey@2 >= 3 AND n_regionkey@2 <= 3 ] (5 rows)
*           PhysicalHashJoin: join_type=Inner, on=[(s_suppkey@0, ps_suppkey@1)] (66 rows)
              ParquetScanSource: projection=[s_suppkey, s_nationkey], predicate=DynamicFilter [ s_nationkey@3 >= 6 AND s_nationkey@3 <= 23 AND s_nationkey@3 IN (SET) ] (207 rows)
              ParquetScanSource: projection=[ps_partkey, ps_suppkey, ps_supplycost], predicate=DynamicFilter [ ps_suppkey@1 >= 7 AND ps_suppkey@1 <= 1000 AND hash_lookup(ps_suppkey@1) ] AND DynamicFilter [ ps_partkey@0 >= 331 AND ps_partkey@0 <= 19800 AND ps_partkey@0 IN (SET) ] (66 rows)
```

</td>
</tr>
</table>

Likely cost: Both blocks join in the order of the query, `region` last; the dynamic filters reduce every probe side to at most 207 rows. With DuckDB's join order written out, our engine runs q2 in 13.65 ms instead of 13.91 ms (2%, within the noise).

### q4: rule gap

- DuckDB: 2 scans; 1 hash join; 1 aggregation; 1 sort; pipeline breaks: 1 aggregation, 1 join build, 1 sort.
- Ours: 2 scans; 1 hash join; 1 aggregation; 1 sort; pipeline breaks: 1 aggregation, 1 join build, 1 sort.

**1. `IS NOT NULL` filters of join keys** (rule gap, follow-up #117; likely impact: none)

- scans: the scan of `lineitem` (`l_commitdate, l_orderkey, l_receiptdate`): DuckDB filters `l_orderkey` with `IS NOT NULL`

<table>
<tr><th>DuckDB</th><th>ours</th></tr>
<tr>
<td>

```text
  ORDER_BY Order By: o_orderpriority ASC ~6,000 rows
    HASH_GROUP_BY Groups: #0; Aggregates: count_star() ~6,000 rows
      HASH_JOIN Join Type: RIGHT_SEMI; Conditions: o_orderkey = o_orderkey ~6,000 rows
*       FILTER Expression: (l_commitdate < l_receiptdate) ~96,202 rows
*         READ_PARQUET Projections: l_orderkey, l_commitdate, l_receiptdate; Filters: (l_orderkey IS NOT NULL) ~481,011 rows
        ...
```

</td>
<td>

```text
  PhysicalSort: expr=[o_orderpriority@0 ASC NULLS LAST] (5 rows)
    PhysicalHashAggregate: gby=[o_orderpriority@0 as o_orderpriority], aggr=[count(Int64(1))] (5 rows)
      PhysicalHashJoin: join_type=LeftSemi, on=[(o_orderkey@0, l_orderkey@0)] (5,249 rows)
        ...
*       ParquetScanSource: projection=[l_orderkey, l_commitdate, l_receiptdate], predicate=l_receiptdate@12 > l_commitdate@11 AND DynamicFilter [ l_orderkey@0 >= 136 AND l_orderkey@0 <= 599874 AND hash_lookup(l_orderkey@0) ], filter=l_receiptdate@2 > l_commitdate@1, projection=[l_orderkey@0] (14,506 rows)
```

</td>
</tr>
</table>

Likely cost: `l_orderkey` has no nulls: the filter removes no row (and the hash join skips null keys anyway).

### q5: rule gap

- DuckDB: 6 scans; 5 hash joins; 1 aggregation; 1 sort; pipeline breaks: 1 aggregation, 5 join builds, 1 sort.
- Ours: 6 scans; 5 hash joins; 1 aggregation; 1 sort; pipeline breaks: 1 aggregation, 5 join builds, 1 sort.

**1. Join order** (rule gap, follow-up #115; likely impact: none)

- joins: the join order differs: DuckDB joins `(((((nation ⋈ region) ⋈ customer) ⋈ supplier) ⋈ orders) ⋈ lineitem)`, `((((nation ⋈ region) ⋈ customer) ⋈ supplier) ⋈ orders)`, `(((nation ⋈ region) ⋈ customer) ⋈ supplier)`, `((nation ⋈ region) ⋈ customer)`, `(nation ⋈ region)`; ours joins `(((((customer ⋈ orders) ⋈ lineitem) ⋈ supplier) ⋈ nation) ⋈ region)`, `((((customer ⋈ orders) ⋈ lineitem) ⋈ supplier) ⋈ nation)`, `(((customer ⋈ orders) ⋈ lineitem) ⋈ supplier)`, `((customer ⋈ orders) ⋈ lineitem)`, `(customer ⋈ orders)`

<table>
<tr><th>DuckDB</th><th>ours</th></tr>
<tr>
<td>

```text
  ORDER_BY Order By: revenue DESC ~0 rows
    HASH_GROUP_BY Groups: #0; Aggregates: sum(#1) ~0 rows
*     HASH_JOIN Join Type: INNER; Conditions: l_orderkey = o_orderkey, l_suppkey = s_suppkey ~0 rows
        READ_PARQUET Projections: l_orderkey, l_suppkey, l_extendedprice, l_discount ~601,264 rows
*       HASH_JOIN Join Type: INNER; Conditions: o_custkey = c_custkey ~0 rows
          READ_PARQUET Projections: o_orderkey, o_custkey; Filters: ((o_orderdate >= '1994-01-01'::DATE) AND (o_orderdate < '1995-01-01'::DATE)) ~30,000 rows
*         HASH_JOIN Join Type: INNER; Conditions: s_nationkey = c_nationkey ~0 rows
            READ_PARQUET Projections: s_suppkey, s_nationkey ~992 rows
*           HASH_JOIN Join Type: INNER; Conditions: c_nationkey = n_nationkey ~0 rows
              READ_PARQUET Projections: c_custkey, c_nationkey ~14,992 rows
*             HASH_JOIN Join Type: INNER; Conditions: n_regionkey = r_regionkey ~0 rows
                READ_PARQUET Projections: n_nationkey, n_name, n_regionkey ~16 rows
                READ_PARQUET Projections: r_regionkey; Filters: (r_name = 'ASIA') ~0 rows
```

</td>
<td>

```text
  PhysicalSort: expr=[revenue@1 DESC] (5 rows)
    PhysicalHashAggregate: gby=[n_name@2 as n_name], aggr=[sum(lineitem.l_extendedprice * Int64(1) - lineitem.l_discount)] (5 rows)
*     PhysicalHashJoin: join_type=Inner, on=[(r_regionkey@0, n_regionkey@3)] (656 rows)
        ParquetScanSource: projection=[r_regionkey, r_name], predicate=r_name@1 = ASIA, filter=r_name@1 = ASIA, projection=[r_regionkey@0] (1 row)
*       PhysicalHashJoin: join_type=Inner, on=[(n_nationkey@0, s_nationkey@2)] (656 rows)
          ParquetScanSource: projection=[n_nationkey, n_name, n_regionkey], predicate=DynamicFilter [ n_regionkey@2 >= 2 AND n_regionkey@2 <= 2 ] (5 rows)
*         PhysicalHashJoin: join_type=Inner, on=[(s_suppkey@0, l_suppkey@1), (s_nationkey@1, c_nationkey@0)] (656 rows)
            ParquetScanSource: projection=[s_suppkey, s_nationkey], predicate=DynamicFilter [ s_nationkey@3 >= 8 AND s_nationkey@3 <= 21 AND s_nationkey@3 IN (SET) ] (202 rows)
*           PhysicalHashJoin: join_type=Inner, on=[(o_orderkey@1, l_orderkey@0)] (16,347 rows)
*             PhysicalHashJoin: join_type=Inner, on=[(c_custkey@0, o_custkey@1)] (4,081 rows)
                ParquetScanSource: projection=[c_custkey, c_nationkey], predicate=DynamicFilter [ c_nationkey@3 >= 8 AND c_nationkey@3 <= 21 AND c_nationkey@3 IN (SET) ] AND DynamicFilter [ c_nationkey@3 >= 8 AND c_nationkey@3 <= 21 ] (3,031 rows)
                ParquetScanSource: projection=[o_orderkey, o_custkey, o_orderdate], predicate=o_orderdate@4 >= 1994-01-01 AND o_orderdate@4 < 1995-01-01 AND DynamicFilter [ o_custkey@1 >= 16 AND o_custkey@1 <= 15000 AND hash_lookup(o_custkey@1) ], filter=o_orderdate@2 >= 1994-01-01 AND o_orderdate@2 < 1995-01-01, projection=[o_orderkey@0, o_custkey@1] (4,081 rows)
              ParquetScanSource: projection=[l_orderkey, l_suppkey, l_extendedprice, l_discount], predicate=DynamicFilter [ l_orderkey@0 >= 66 AND l_orderkey@0 <= 599976 AND hash_lookup(l_orderkey@0) ] AND DynamicFilter [ l_suppkey@2 >= 9 AND l_suppkey@2 <= 995 ] (16,347 rows)
```

</td>
</tr>
</table>

Likely cost: Our order builds on the 3,031 customers and the 4,081 orders that the dynamic filters let through and probes about 16,300 `lineitem` rows; DuckDB's probes all of `lineitem` against the joins of the five other tables. With DuckDB's join order written out, our engine is slower: 13.56 ms instead of 12.85 ms.

### q7: rule gap

- DuckDB: 6 scans; 5 hash joins; 1 aggregation; 1 sort; pipeline breaks: 1 aggregation, 5 join builds, 1 sort.
- Ours: 6 scans; 5 hash joins; 1 aggregation; 1 sort; pipeline breaks: 1 aggregation, 5 join builds, 1 sort.

**1. Join order** (rule gap, follow-up #115; likely impact: none)

- joins: the join order differs: DuckDB joins `((((nation ⋈ supplier) ⋈ lineitem) ⋈ orders) ⋈ (customer ⋈ nation))`, `(((nation ⋈ supplier) ⋈ lineitem) ⋈ orders)`, `((nation ⋈ supplier) ⋈ lineitem)`, `(nation ⋈ supplier)`, `(customer ⋈ nation)`; ours joins `(((((lineitem ⋈ supplier) ⋈ orders) ⋈ customer) ⋈ nation) ⋈ nation)`, `((((lineitem ⋈ supplier) ⋈ orders) ⋈ customer) ⋈ nation)`, `(((lineitem ⋈ supplier) ⋈ orders) ⋈ customer)`, `((lineitem ⋈ supplier) ⋈ orders)`, `(lineitem ⋈ supplier)`

<table>
<tr><th>DuckDB</th><th>ours</th></tr>
<tr>
<td>

```text
  ORDER_BY Order By: supp_nation ASC, cust_nation ASC, l_year ASC ~738 rows
    HASH_GROUP_BY Groups: #0, #1, #2; Aggregates: sum(#3) ~738 rows
*     HASH_JOIN Join Type: INNER; Conditions: o_custkey = c_custkey, (((n_name = 'FRANCE') AND (n_name = 'GERMANY')) OR ((n_name = 'GERMANY') AND (n_name = 'FRANCE'))) ~845 rows
*       HASH_JOIN Join Type: INNER; Conditions: o_orderkey = l_orderkey ~22,547 rows
          READ_PARQUET Projections: o_orderkey, o_custkey ~150,000 rows
*         HASH_JOIN Join Type: INNER; Conditions: l_suppkey = s_suppkey ~22,547 rows
            READ_PARQUET Projections: l_orderkey, l_suppkey, l_extendedprice, l_discount, l_shipdate; Filters: ((l_shipdate >= '1995-01-01'::DATE) AND (l_shipdate <= '1996-12-31'::DATE)) ~120,252 rows
*           HASH_JOIN Join Type: INNER; Conditions: s_nationkey = n_nationkey ~186 rows
              READ_PARQUET Projections: s_suppkey, s_nationkey ~992 rows
              FILTER Expression: ((n_name = 'FRANCE') OR (n_name = 'GERMANY')) ~3 rows
                READ_PARQUET Projections: n_nationkey, n_name; Filters: optional: (n_name IN ('FRANCE', 'GERMANY')) ~16 rows
*       HASH_JOIN Join Type: INNER; Conditions: c_nationkey = n_nationkey ~2,811 rows
          READ_PARQUET Projections: c_custkey, c_nationkey ~14,992 rows
          FILTER Expression: ((n_name = 'GERMANY') OR (n_name = 'FRANCE')) ~3 rows
            READ_PARQUET Projections: n_nationkey, n_name; Filters: optional: (n_name IN ('GERMANY', 'FRANCE')) ~16 rows
```

</td>
<td>

```text
  PhysicalSort: expr=[supp_nation@0 ASC NULLS LAST, cust_nation@1 ASC NULLS LAST, l_year@2 ASC NULLS LAST] (4 rows)
    PhysicalHashAggregate: gby=[supp_nation@0 as supp_nation, cust_nation@1 as cust_nation, l_year@2 as l_year], aggr=[sum(shipping.volume)] (4 rows)
*     PhysicalHashJoin: join_type=Inner, on=[(n_nationkey@0, c_nationkey@3)], filter=n_name@0 = FRANCE AND n_name@1 = GERMANY OR n_name@0 = GERMANY AND n_name@1 = FRANCE (518 rows)
        ParquetScanSource: projection=[n_nationkey, n_name], predicate=n_name@1 = GERMANY OR n_name@1 = FRANCE, filter=n_name@1 = GERMANY OR n_name@1 = FRANCE (2 rows)
*       PhysicalHashJoin: join_type=Inner, on=[(n_nationkey@0, s_nationkey@0)] (1,060 rows)
          ParquetScanSource: projection=[n_nationkey, n_name], predicate=n_name@1 = FRANCE OR n_name@1 = GERMANY, filter=n_name@1 = FRANCE OR n_name@1 = GERMANY (2 rows)
*         PhysicalHashJoin: join_type=Inner, on=[(c_custkey@0, o_custkey@4)] (1,060 rows)
            ParquetScanSource: projection=[c_custkey, c_nationkey], predicate=DynamicFilter [ c_nationkey@3 >= 6 AND c_nationkey@3 <= 7 ] (1,216 rows)
*           PhysicalHashJoin: join_type=Inner, on=[(o_orderkey@0, l_orderkey@1)] (1,060 rows)
              ParquetScanSource: projection=[o_orderkey, o_custkey], predicate=DynamicFilter [ o_custkey@1 >= 7 AND o_custkey@1 <= 14987 AND hash_lookup(o_custkey@1) ] (11,043 rows)
*             PhysicalHashJoin: join_type=Inner, on=[(s_suppkey@0, l_suppkey@1)] (1,060 rows)
                ParquetScanSource: projection=[s_suppkey, s_nationkey], predicate=DynamicFilter [ s_nationkey@3 >= 6 AND s_nationkey@3 <= 7 ] (84 rows)
                ParquetScanSource: projection=[l_orderkey, l_suppkey, l_extendedprice, l_discount, l_shipdate], predicate=l_shipdate@10 >= 1995-01-01 AND l_shipdate@10 <= 1996-12-31 AND DynamicFilter [ l_suppkey@2 >= 7 AND l_suppkey@2 <= 990 AND l_suppkey@2 IN (SET) ] AND DynamicFilter [ l_orderkey@0 >= 36 AND l_orderkey@0 <= 599974 AND hash_lookup(l_orderkey@0) ], filter=l_shipdate@4 >= 1995-01-01 AND l_shipdate@4 <= 1996-12-31 (1,060 rows)
```

</td>
</tr>
</table>

Likely cost: The dynamic filters cut every probe side (11,043 `orders` rows, 1,216 customers): 13.03 ms with DuckDB's join order instead of 13.22 ms (1%, within the noise).

### q8: rule gap

- DuckDB: 8 scans; 7 hash joins; 1 aggregation; 1 sort; pipeline breaks: 1 aggregation, 7 join builds, 1 sort.
- Ours: 8 scans; 7 hash joins; 1 aggregation; 1 sort; pipeline breaks: 1 aggregation, 7 join builds, 1 sort.

**1. Join order** (rule gap, follow-up #115; likely impact: none)

- joins: the join order differs: DuckDB joins `(((((((nation ⋈ region) ⋈ customer) ⋈ orders) ⋈ lineitem) ⋈ part) ⋈ supplier) ⋈ nation)`, `((((((nation ⋈ region) ⋈ customer) ⋈ orders) ⋈ lineitem) ⋈ part) ⋈ supplier)`, `(((((nation ⋈ region) ⋈ customer) ⋈ orders) ⋈ lineitem) ⋈ part)`, `((((nation ⋈ region) ⋈ customer) ⋈ orders) ⋈ lineitem)`, `(((nation ⋈ region) ⋈ customer) ⋈ orders)`, `((nation ⋈ region) ⋈ customer)`, `(nation ⋈ region)`; ours joins `(((((((lineitem ⋈ part) ⋈ supplier) ⋈ orders) ⋈ customer) ⋈ nation) ⋈ nation) ⋈ region)`, `((((((lineitem ⋈ part) ⋈ supplier) ⋈ orders) ⋈ customer) ⋈ nation) ⋈ nation)`, `(((((lineitem ⋈ part) ⋈ supplier) ⋈ orders) ⋈ customer) ⋈ nation)`, `((((lineitem ⋈ part) ⋈ supplier) ⋈ orders) ⋈ customer)`, `(((lineitem ⋈ part) ⋈ supplier) ⋈ orders)`, `((lineitem ⋈ part) ⋈ supplier)`, `(lineitem ⋈ part)`

<table>
<tr><th>DuckDB</th><th>ours</th></tr>
<tr>
<td>

```text
  ORDER_BY Order By: o_year ASC ~0 rows
    HASH_GROUP_BY Groups: #0; Aggregates: sum(#1), sum(#2) ~0 rows
*     HASH_JOIN Join Type: INNER; Conditions: n_nationkey = s_nationkey ~0 rows
        READ_PARQUET Projections: n_nationkey, n_name ~16 rows
*       HASH_JOIN Join Type: INNER; Conditions: s_suppkey = l_suppkey ~0 rows
          READ_PARQUET Projections: s_suppkey, s_nationkey ~992 rows
*         HASH_JOIN Join Type: INNER; Conditions: p_partkey = l_partkey ~0 rows
            READ_PARQUET Projections: p_partkey; Filters: (p_type = 'ECONOMY ANODIZED STEEL') ~4,000 rows
*           HASH_JOIN Join Type: INNER; Conditions: l_orderkey = o_orderkey ~0 rows
              READ_PARQUET Projections: l_orderkey, l_partkey, l_suppkey, l_extendedprice, l_discount ~601,264 rows
*             HASH_JOIN Join Type: INNER; Conditions: o_custkey = c_custkey ~0 rows
                READ_PARQUET Projections: o_orderkey, o_custkey, o_orderdate; Filters: ((o_orderdate >= '1995-01-01'::DATE) AND (o_orderdate <= '1996-12-31'::DATE)) ~30,000 rows
*               HASH_JOIN Join Type: INNER; Conditions: c_nationkey = n_nationkey ~0 rows
                  READ_PARQUET Projections: c_custkey, c_nationkey ~14,992 rows
*                 HASH_JOIN Join Type: INNER; Conditions: n_regionkey = r_regionkey ~0 rows
                    READ_PARQUET Projections: n_nationkey, n_regionkey ~16 rows
                    READ_PARQUET Projections: r_regionkey; Filters: (r_name = 'AMERICA') ~0 rows
```

</td>
<td>

```text
  PhysicalSort: expr=[o_year@0 ASC NULLS LAST] (2 rows)
    PhysicalHashAggregate: gby=[o_year@0 as o_year], aggr=[sum(CASE WHEN all_nations.nation = Utf8("BRAZIL") THEN all_nations.volume ELSE Int64(0) END), sum(all_nations.volume)] (2 rows)
*     PhysicalHashJoin: join_type=Inner, on=[(r_regionkey@0, n_regionkey@3)] (207 rows)
        ParquetScanSource: projection=[r_regionkey, r_name], predicate=r_name@1 = AMERICA, filter=r_name@1 = AMERICA, projection=[r_regionkey@0] (1 row)
*       PhysicalHashJoin: join_type=Inner, on=[(n_nationkey@0, s_nationkey@2)] (207 rows)
          ParquetScanSource: projection=[n_nationkey, n_name] (25 rows)
*         PhysicalHashJoin: join_type=Inner, on=[(n_nationkey@0, c_nationkey@4)] (207 rows)
            ParquetScanSource: projection=[n_nationkey, n_regionkey], predicate=DynamicFilter [ n_regionkey@2 >= 1 AND n_regionkey@2 <= 1 ] (5 rows)
*           PhysicalHashJoin: join_type=Inner, on=[(c_custkey@0, o_custkey@3)] (207 rows)
              ParquetScanSource: projection=[c_custkey, c_nationkey], predicate=DynamicFilter [ c_nationkey@3 >= 1 AND c_nationkey@3 <= 24 AND c_nationkey@3 IN (SET) ] (2,972 rows)
*             PhysicalHashJoin: join_type=Inner, on=[(o_orderkey@0, l_orderkey@0)] (207 rows)
                ParquetScanSource: projection=[o_orderkey, o_custkey, o_orderdate], predicate=o_orderdate@4 >= 1995-01-01 AND o_orderdate@4 <= 1996-12-31 AND DynamicFilter [ o_custkey@1 >= 1 AND o_custkey@1 <= 14998 AND hash_lookup(o_custkey@1) ], filter=o_orderdate@2 >= 1995-01-01 AND o_orderdate@2 <= 1996-12-31 (9,399 rows)
*               PhysicalHashJoin: join_type=Inner, on=[(s_suppkey@0, l_suppkey@1)] (207 rows)
                  ParquetScanSource: projection=[s_suppkey, s_nationkey], predicate=DynamicFilter [ s_nationkey@3 >= 0 AND s_nationkey@3 <= 24 ] (1,000 rows)
*                 PhysicalHashJoin: join_type=Inner, on=[(p_partkey@0, l_partkey@1)] (207 rows)
                    ParquetScanSource: projection=[p_partkey, p_type], predicate=p_type@4 = ECONOMY ANODIZED STEEL, filter=p_type@1 = ECONOMY ANODIZED STEEL, projection=[p_partkey@0] (130 rows)
                    ParquetScanSource: projection=[l_orderkey, l_partkey, l_suppkey, l_extendedprice, l_discount], predicate=DynamicFilter [ l_partkey@1 >= 120 AND l_partkey@1 <= 19741 AND l_partkey@1 IN (SET) ] AND DynamicFilter [ l_suppkey@2 >= 1 AND l_suppkey@2 <= 1000 ] AND DynamicFilter [ l_orderkey@0 >= 163 AND l_orderkey@0 <= 599878 AND hash_lookup(l_orderkey@0) ] (207 rows)
```

</td>
</tr>
</table>

Likely cost: 15.68 ms with DuckDB's join order instead of 15.95 ms (2%, within the noise): the dynamic filters cut the `lineitem` scan to the rows of the 130 parts of the type.

### q9: rule gap

- DuckDB: 6 scans; 5 hash joins; 1 aggregation; 1 sort; pipeline breaks: 1 aggregation, 5 join builds, 1 sort.
- Ours: 6 scans; 5 hash joins; 1 aggregation; 1 sort; pipeline breaks: 1 aggregation, 5 join builds, 1 sort.

**1. Join order** (rule gap, follow-up #115; likely impact: none)

- joins: the join order differs: DuckDB joins `((((nation ⋈ supplier) ⋈ (part ⋈ partsupp)) ⋈ lineitem) ⋈ orders)`, `(((nation ⋈ supplier) ⋈ (part ⋈ partsupp)) ⋈ lineitem)`, `((nation ⋈ supplier) ⋈ (part ⋈ partsupp))`, `(part ⋈ partsupp)`, `(nation ⋈ supplier)`; ours joins `(((((lineitem ⋈ part) ⋈ supplier) ⋈ partsupp) ⋈ orders) ⋈ nation)`, `((((lineitem ⋈ part) ⋈ supplier) ⋈ partsupp) ⋈ orders)`, `(((lineitem ⋈ part) ⋈ supplier) ⋈ partsupp)`, `((lineitem ⋈ part) ⋈ supplier)`, `(lineitem ⋈ part)`

<table>
<tr><th>DuckDB</th><th>ours</th></tr>
<tr>
<td>

```text
  ORDER_BY Order By: nation ASC, o_year DESC ~116,381 rows
    HASH_GROUP_BY Groups: #0, #1; Aggregates: sum(#2) ~116,381 rows
*     HASH_JOIN Join Type: INNER; Conditions: l_orderkey = o_orderkey ~120,252 rows
*       HASH_JOIN Join Type: INNER; Conditions: l_partkey = p_partkey, l_suppkey = s_suppkey ~120,252 rows
          READ_PARQUET Projections: l_orderkey, l_partkey, l_suppkey, l_quantity, l_extendedprice, l_discount ~601,264 rows
*         HASH_JOIN Join Type: INNER; Conditions: ps_suppkey = s_suppkey ~16,000 rows
*           HASH_JOIN Join Type: INNER; Conditions: ps_partkey = p_partkey ~16,000 rows
              READ_PARQUET Projections: ps_partkey, ps_suppkey, ps_supplycost ~80,000 rows
              READ_PARQUET Projections: p_partkey; Filters: contains(p_name, 'green') ~4,000 rows
*           HASH_JOIN Join Type: INNER; Conditions: s_nationkey = n_nationkey ~992 rows
              READ_PARQUET Projections: s_suppkey, s_nationkey ~992 rows
              READ_PARQUET Projections: n_nationkey, n_name ~16 rows
        READ_PARQUET Projections: o_orderkey, o_orderdate ~150,000 rows
```

</td>
<td>

```text
  PhysicalSort: expr=[nation@0 ASC NULLS LAST, o_year@1 DESC] (175 rows)
    PhysicalHashAggregate: gby=[nation@0 as nation, o_year@1 as o_year], aggr=[sum(profit.amount)] (175 rows)
*     PhysicalHashJoin: join_type=Inner, on=[(n_nationkey@0, s_nationkey@3)] (36,764 rows)
        ParquetScanSource: projection=[n_nationkey, n_name] (25 rows)
*       PhysicalHashJoin: join_type=Inner, on=[(o_orderkey@0, l_orderkey@0)] (36,764 rows)
          ParquetScanSource: projection=[o_orderkey, o_orderdate] (150,000 rows)
*         PhysicalHashJoin: join_type=Inner, on=[(ps_suppkey@1, l_suppkey@2), (ps_partkey@0, l_partkey@1)] (36,764 rows)
            ParquetScanSource: projection=[ps_partkey, ps_suppkey, ps_supplycost] (80,000 rows)
*           PhysicalHashJoin: join_type=Inner, on=[(s_suppkey@0, l_suppkey@2)] (36,764 rows)
              ParquetScanSource: projection=[s_suppkey, s_nationkey], predicate=DynamicFilter [ s_nationkey@3 >= 0 AND s_nationkey@3 <= 24 ] AND DynamicFilter [ s_suppkey@0 >= 1 AND s_suppkey@0 <= 1000 ] (1,000 rows)
*             PhysicalHashJoin: join_type=Inner, on=[(p_partkey@0, l_partkey@1)] (36,764 rows)
                ParquetScanSource: projection=[p_partkey, p_name], predicate=p_name@1 LIKE %green% AND DynamicFilter [ p_partkey@0 >= 1 AND p_partkey@0 <= 20000 ], filter=p_name@1 LIKE %green%, projection=[p_partkey@0] (1,084 rows)
                ParquetScanSource: projection=[l_orderkey, l_partkey, l_suppkey, l_quantity, l_extendedprice, l_discount], predicate=DynamicFilter [ l_partkey@1 >= 9 AND l_partkey@1 <= 19989 AND hash_lookup(l_partkey@1) ] AND DynamicFilter [ l_suppkey@2 >= 1 AND l_suppkey@2 <= 1000 ] AND DynamicFilter [ l_suppkey@2 >= 1 AND l_suppkey@2 <= 1000 AND l_partkey@1 >= 1 AND l_partkey@1 <= 20000 AND hash_lookup(l_suppkey@2, l_partkey@1) ] AND DynamicFilter [ l_orderkey@0 >= 1 AND l_orderkey@0 <= 599976 AND hash_lookup(l_orderkey@0) ] (36,764 rows)
```

</td>
</tr>
</table>

Likely cost: Ours builds on all 80,000 `partsupp` and 150,000 `orders` rows (DuckDB builds on `orders` too) and the dynamic filters cut `lineitem` to 36,764 rows; DuckDB's order, `part` with `partsupp` first, is slower on our engine: 20.33 ms instead of 19.67 ms.

### q10: rule gap

- DuckDB: 4 scans; 3 hash joins; 1 aggregation; 1 top-N; pipeline breaks: 1 aggregation, 3 join builds, 1 top-N.
- Ours: 4 scans; 3 hash joins; 1 aggregation; 1 top-N; pipeline breaks: 1 aggregation, 3 join builds, 1 top-N.

**1. Join order** (rule gap, follow-up #115; likely impact: none)

- joins: the join order differs: DuckDB joins `((customer ⋈ nation) ⋈ (lineitem ⋈ orders))`, `(customer ⋈ nation)`, `(lineitem ⋈ orders)`; ours joins `(((customer ⋈ orders) ⋈ lineitem) ⋈ nation)`, `((customer ⋈ orders) ⋈ lineitem)`, `(customer ⋈ orders)`

<table>
<tr><th>DuckDB</th><th>ours</th></tr>
<tr>
<td>

```text
  TOP_N Top: 20; Order By: revenue DESC ~20 rows
    HASH_GROUP_BY Groups: #0, #1, #2, #3, #4, #5, #6; Aggregates: sum(#7) ~24,050 rows
*     HASH_JOIN Join Type: INNER; Conditions: c_custkey = o_custkey ~24,050 rows
*       HASH_JOIN Join Type: INNER; Conditions: c_nationkey = n_nationkey ~14,992 rows
          READ_PARQUET Projections: c_custkey, c_name, c_address, c_nationkey, c_phone, c_acctbal, c_comment ~14,992 rows
          READ_PARQUET Projections: n_nationkey, n_name ~16 rows
*       HASH_JOIN Join Type: INNER; Conditions: l_orderkey = o_orderkey ~24,050 rows
          READ_PARQUET Projections: l_orderkey, l_extendedprice, l_discount; Filters: (l_returnflag = 'R') ~120,252 rows
          READ_PARQUET Projections: o_orderkey, o_custkey; Filters: ((o_orderdate >= '1993-10-01'::DATE) AND (o_orderdate < '1994-01-01'::DATE)) ~30,000 rows
```

</td>
<td>

```text
  PhysicalTopK: fetch=20, expr=[revenue@2 DESC], dynamic_filter=true (20 rows)
    PhysicalHashAggregate: gby=[c_custkey@0 as c_custkey, c_name@1 as c_name, c_acctbal@4 as c_acctbal, c_phone@3 as c_phone, n_name@8 as n_name, c_address@2 as c_address, c_comment@5 as c_comment], aggr=[sum(lineitem.l_extendedprice * Int64(1) - lineitem.l_discount)] (3,581 rows)
*     PhysicalHashJoin: join_type=Inner, on=[(n_nationkey@0, c_nationkey@3)] (11,516 rows)
        ParquetScanSource: projection=[n_nationkey, n_name] (25 rows)
*       PhysicalHashJoin: join_type=Inner, on=[(o_orderkey@7, l_orderkey@0)] (11,516 rows)
*         PhysicalHashJoin: join_type=Inner, on=[(o_custkey@1, c_custkey@0)] (5,673 rows)
            ParquetScanSource: projection=[o_orderkey, o_custkey, o_orderdate], predicate=o_orderdate@4 >= 1993-10-01 AND o_orderdate@4 < 1994-01-01, filter=o_orderdate@2 >= 1993-10-01 AND o_orderdate@2 < 1994-01-01, projection=[o_orderkey@0, o_custkey@1] (5,703 rows)
            ParquetScanSource: projection=[c_custkey, c_name, c_address, c_nationkey, c_phone, c_acctbal, c_comment], predicate=DynamicFilter [ c_custkey@0 >= 1 AND c_custkey@0 <= 14999 AND hash_lookup(c_custkey@0) ] AND DynamicFilter [ c_nationkey@3 >= 0 AND c_nationkey@3 <= 24 ] (4,014 rows)
          ParquetScanSource: projection=[l_orderkey, l_extendedprice, l_discount, l_returnflag], predicate=l_returnflag@8 = R AND DynamicFilter [ l_orderkey@0 >= 2 AND l_orderkey@0 <= 599912 AND hash_lookup(l_orderkey@0) ], filter=l_returnflag@3 = R, projection=[l_orderkey@0, l_extendedprice@1, l_discount@2] (11,516 rows)
```

</td>
</tr>
</table>

Likely cost: DuckDB joins `customer` with `nation` and `lineitem` with `orders`, then the two (a bushy tree); ours joins in the order of the query. With DuckDB's join order written out, q10 takes 13.88 ms either way.

### q11: rule gap

- DuckDB: 6 scans; 4 hash joins; 2 aggregations; 1 sort; pipeline breaks: 2 aggregations, 4 join builds, 1 scalar subquery, 1 sort.
- Ours: 6 scans; 4 hash joins; 2 aggregations; 1 sort; pipeline breaks: 2 aggregations, 4 join builds, 1 scalar subquery, 1 sort.

**1. Join order** (rule gap, follow-up #115; likely impact: none)

- joins: the join order differs: DuckDB joins `((nation ⋈ supplier) ⋈ partsupp)`, `(nation ⋈ supplier)`, `((nation ⋈ supplier) ⋈ partsupp)`, `(nation ⋈ supplier)`; ours joins `((partsupp ⋈ supplier) ⋈ nation)`, `(partsupp ⋈ supplier)`, `((partsupp ⋈ supplier) ⋈ nation)`, `(partsupp ⋈ supplier)`

<table>
<tr><th>DuckDB</th><th>ours</th></tr>
<tr>
<td>

```text
  ORDER_BY Order By: value DESC ~7,500 rows
    NESTED_LOOP_JOIN Join Type: INNER; Conditions: CAST(sum((ps_supplycost * CAST(ps_availqty AS DECIMAL(18,0)))) AS DECIMAL(38,6)) > SUBQUERY ~7,500 rows
      HASH_GROUP_BY Groups: #0; Aggregates: sum(#1) ~15,000 rows
*       HASH_JOIN Join Type: INNER; Conditions: ps_suppkey = s_suppkey ~15,000 rows
          READ_PARQUET Projections: ps_partkey, ps_suppkey, ps_availqty, ps_supplycost ~80,000 rows
*         HASH_JOIN Join Type: INNER; Conditions: s_nationkey = n_nationkey ~186 rows
            READ_PARQUET Projections: s_suppkey, s_nationkey ~992 rows
            READ_PARQUET Projections: n_nationkey; Filters: (n_name = 'GERMANY') ~3 rows
  scalar subquery:
    UNGROUPED_AGGREGATE Aggregates: sum(#0) ~1 row
*     HASH_JOIN Join Type: INNER; Conditions: ps_suppkey = s_suppkey ~15,000 rows
        READ_PARQUET Projections: ps_suppkey, ps_availqty, ps_supplycost ~80,000 rows
*       HASH_JOIN Join Type: INNER; Conditions: s_nationkey = n_nationkey ~186 rows
          READ_PARQUET Projections: s_suppkey, s_nationkey ~992 rows
          READ_PARQUET Projections: n_nationkey; Filters: (n_name = 'GERMANY') ~3 rows
```

</td>
<td>

```text
  PhysicalSort: expr=[value@1 DESC] (2,443 rows)
    PhysicalFilter: CAST(sum(partsupp.ps_supplycost * partsupp.ps_availqty)@1 AS Decimal128(38, 15)) > scalar_subquery(981240.039329000194048) (2,443 rows)
      PhysicalHashAggregate: gby=[ps_partkey@0 as ps_partkey], aggr=[sum(partsupp.ps_supplycost * partsupp.ps_availqty)] (3,513 rows)
*       PhysicalHashJoin: join_type=Inner, on=[(n_nationkey@0, s_nationkey@3)] (3,760 rows)
          ParquetScanSource: projection=[n_nationkey, n_name], predicate=n_name@1 = GERMANY, filter=n_name@1 = GERMANY, projection=[n_nationkey@0] (1 row)
*         PhysicalHashJoin: join_type=Inner, on=[(s_suppkey@0, ps_suppkey@1)] (3,760 rows)
            ParquetScanSource: projection=[s_suppkey, s_nationkey], predicate=DynamicFilter [ s_nationkey@3 >= 7 AND s_nationkey@3 <= 7 ] (47 rows)
            ParquetScanSource: projection=[ps_partkey, ps_suppkey, ps_availqty, ps_supplycost], predicate=DynamicFilter [ ps_suppkey@1 >= 8 AND ps_suppkey@1 <= 990 AND ps_suppkey@1 IN (SET) ] (3,760 rows)
  scalar subquery:
    PhysicalHashAggregate: gby=[], aggr=[sum(partsupp.ps_supplycost * partsupp.ps_availqty)] (1 row)
*     PhysicalHashJoin: join_type=Inner, on=[(n_nationkey@0, s_nationkey@2)] (3,760 rows)
        ParquetScanSource: projection=[n_nationkey, n_name], predicate=n_name@1 = GERMANY, filter=n_name@1 = GERMANY, projection=[n_nationkey@0] (1 row)
*       PhysicalHashJoin: join_type=Inner, on=[(s_suppkey@0, ps_suppkey@0)] (3,760 rows)
          ParquetScanSource: projection=[s_suppkey, s_nationkey], predicate=DynamicFilter [ s_nationkey@3 >= 7 AND s_nationkey@3 <= 7 ] (47 rows)
          ParquetScanSource: projection=[ps_suppkey, ps_availqty, ps_supplycost], predicate=DynamicFilter [ ps_suppkey@1 >= 8 AND ps_suppkey@1 <= 990 AND ps_suppkey@1 IN (SET) ] (3,760 rows)
```

</td>
</tr>
</table>

Likely cost: The dynamic filter of `supplier` cuts both `partsupp` scans to 3,760 rows: 9.53 ms with DuckDB's join order instead of 9.61 ms (1%, within the noise).

### q12: rule gap

- DuckDB: 2 scans; 1 hash join; 1 aggregation; 1 sort; pipeline breaks: 1 aggregation, 1 join build, 1 sort.
- Ours: 2 scans; 1 hash join; 1 aggregation; 1 sort; pipeline breaks: 1 aggregation, 1 join build, 1 sort.

**1. Constant bounds derived through comparisons of columns** (rule gap, follow-up #118; likely impact: none)

- scans: the scan of `lineitem` (`l_commitdate, l_orderkey, l_receiptdate, l_shipdate, l_shipmode`): DuckDB bounds `l_commitdate, l_shipdate` by constants, which our scan compares with other columns only

<table>
<tr><th>DuckDB</th><th>ours</th></tr>
<tr>
<td>

```text
  ORDER_BY Order By: l_shipmode ASC ~2 rows
    HASH_GROUP_BY Groups: #0; Aggregates: sum(#1), sum(#2) ~2 rows
      HASH_JOIN Join Type: INNER; Conditions: o_orderkey = l_orderkey ~24,050 rows
        ...
*       FILTER Expression: ((l_commitdate < l_receiptdate) AND (l_shipdate < l_commitdate) AND ((l_shipmode = 'MAIL') OR (l_shipmode = 'SHIP'))) ~24,050 rows
*         READ_PARQUET Projections: l_orderkey, l_shipdate, l_commitdate, l_receiptdate, l_shipmode; Filters: (l_shipdate < '1995-01-01'::DATE), (l_commitdate < '1995-01-01'::DATE), ((l_receiptdate >= '1994-01-01'::DATE) AND (l_receiptdate < '1995-01-01'::DATE)), optional: (l_shipmode IN ('MAIL', 'SHIP')) ~120,252 rows
```

</td>
<td>

```text
  PhysicalSort: expr=[l_shipmode@0 ASC NULLS LAST] (2 rows)
    PhysicalHashAggregate: gby=[l_shipmode@0 as l_shipmode], aggr=[sum(CASE WHEN orders.o_orderpriority = Utf8("1-URGENT") OR orders.o_orderpriority = Utf8("2-HIGH") THEN Int64(1) ELSE Int64(0) END), sum(CASE WHEN orders.o_orderpriority != Utf8("1-URGENT") AND orders.o_orderpriority != Utf8("2-HIGH") THEN Int64(1) ELSE Int64(0) END)] (2 rows)
      PhysicalHashJoin: join_type=Inner, on=[(l_orderkey@0, o_orderkey@0)] (3,123 rows)
*       ParquetScanSource: projection=[l_orderkey, l_shipdate, l_commitdate, l_receiptdate, l_shipmode], predicate=(l_shipmode@14 = MAIL OR l_shipmode@14 = SHIP) AND l_receiptdate@12 > l_commitdate@11 AND l_shipdate@10 < l_commitdate@11 AND l_receiptdate@12 >= 1994-01-01 AND l_receiptdate@12 < 1995-01-01, filter=(l_shipmode@4 = MAIL OR l_shipmode@4 = SHIP) AND l_receiptdate@3 > l_commitdate@2 AND l_shipdate@1 < l_commitdate@2 AND l_receiptdate@3 >= 1994-01-01 AND l_receiptdate@3 < 1995-01-01, projection=[l_orderkey@0, l_shipmode@4] (3,123 rows)
        ...
```

</td>
</tr>
</table>

Likely cost: The derived bounds follow from the filters that both plans have (`l_receiptdate < 1995-01-01` and `l_shipdate < l_commitdate < l_receiptdate`): they remove no row, and they prune no row group of this data, whose dates are spread over all row groups.

### q14: cost/statistics-driven

- DuckDB: 2 scans; 1 hash join; 1 aggregation; pipeline breaks: 1 aggregation, 1 join build.
- Ours: 2 scans; 1 hash join; 1 aggregation; pipeline breaks: 1 aggregation, 1 join build.

**1. Build side of a join** (cost/statistics-driven; likely impact: none)

- joins: the join `(lineitem ⋈ part)` builds on `part` in DuckDB (~20,000 rows estimated, probed by ~120,252), on `lineitem` in ours (7,594 rows, probed by 5,538)

<table>
<tr><th>DuckDB</th><th>ours</th></tr>
<tr>
<td>

```text
  UNGROUPED_AGGREGATE Aggregates: sum(#0), sum(#1) ~1 row
*   HASH_JOIN Join Type: INNER; Conditions: l_partkey = p_partkey ~120,252 rows
      READ_PARQUET Projections: l_partkey, l_extendedprice, l_discount; Filters: ((l_shipdate >= '1995-09-01'::DATE) AND (l_shipdate < '1995-10-01'::DATE)) ~120,252 rows
      READ_PARQUET Projections: p_partkey, p_type ~20,000 rows
```

</td>
<td>

```text
  PhysicalHashAggregate: gby=[], aggr=[sum(CASE WHEN part.p_type LIKE Utf8("PROMO%") THEN lineitem.l_extendedprice * Int64(1) - lineitem.l_discount ELSE Int64(0) END), sum(lineitem.l_extendedprice * Int64(1) - lineitem.l_discount)] (1 row)
*   PhysicalHashJoin: join_type=Inner, on=[(l_partkey@0, p_partkey@0)] (7,594 rows)
      ParquetScanSource: projection=[l_partkey, l_extendedprice, l_discount, l_shipdate], predicate=l_shipdate@10 >= 1995-09-01 AND l_shipdate@10 < 1995-10-01, filter=l_shipdate@3 >= 1995-09-01 AND l_shipdate@3 < 1995-10-01, projection=[l_partkey@0, l_extendedprice@1, l_discount@2] (7,594 rows)
      ParquetScanSource: projection=[p_partkey, p_type], predicate=DynamicFilter [ p_partkey@0 >= 1 AND p_partkey@0 <= 19998 AND hash_lookup(p_partkey@0) ] (5,538 rows)
```

</td>
</tr>
</table>

Likely cost: Ours builds on the 7,594 `lineitem` rows of the month, whose dynamic filter cuts the `part` scan to 5,538 rows; DuckDB builds on all 20,000 parts (it estimates 120,252 `lineitem` rows). Our build side is the smaller one.

### q15: rule gap

- DuckDB: 2 scans; 1 hash join; 2 aggregations; 1 sort; pipeline breaks: 2 aggregations, 1 join build, 1 scalar subquery, 1 shared subplan, 1 sort.
- Ours: 3 scans; 1 hash join; 3 aggregations; 1 sort; pipeline breaks: 3 aggregations, 1 join build, 1 scalar subquery, 1 sort.

**1. Common table expressions computed once per reference** (rule gap, follow-up #114; likely impact: high)

- pipeline breaks: DuckDB computes `revenue0` (`agg(lineitem)`) once, in a shared subplan that it reads twice; ours computes it twice
- scans: `lineitem` is scanned once in DuckDB, twice in ours
- aggregations: the aggregations differ: DuckDB has none that ours has not, ours an aggregation by 1 key of `lineitem` that DuckDB has not
- pipeline breaks: the pipeline breaks differ (DuckDB / ours): aggregation 2 / 3, shared subplan 1 / 0

<table>
<tr><th>DuckDB</th><th>ours</th></tr>
<tr>
<td>

```text
* CTE CTE Name: revenue0; Table Index: 0; Execution Mode: PIPELINE_DEPENDENT ~120,252 rows
*   HASH_GROUP_BY Groups: #0; Aggregates: sum(#1) ~120,252 rows
*     READ_PARQUET Projections: l_suppkey, l_extendedprice, l_discount; Filters: ((l_shipdate >= '1996-01-01'::DATE) AND (l_shipdate < '1996-04-01'::DATE)) ~120,252 rows
    ORDER_BY Order By: s_suppkey ASC ~120,252 rows
      HASH_JOIN Join Type: INNER; Conditions: supplier_no = s_suppkey ~120,252 rows
        HASH_JOIN Join Type: INNER; Conditions: total_revenue = SUBQUERY ~120,252 rows
*         CTE_SCAN CTE Index: 0 ~120,252 rows
        ...
  scalar subquery:
*   UNGROUPED_AGGREGATE Aggregates: max(#0) ~1 row
*     CTE_SCAN CTE Index: 0 ~120,252 rows
```

</td>
<td>

```text
  PhysicalSort: expr=[s_suppkey@0 ASC NULLS LAST] (1 row)
    PhysicalHashJoin: join_type=Inner, on=[(s_suppkey@0, supplier_no@0)] (1 row)
      ...
      PhysicalFilter: sum(lineitem.l_extendedprice * Int64(1) - lineitem.l_discount)@1 = scalar_subquery(3178569.4928) (1 row)
*       PhysicalHashAggregate: gby=[l_suppkey@0 as l_suppkey], aggr=[sum(lineitem.l_extendedprice * Int64(1) - lineitem.l_discount)] (1,000 rows)
*         ParquetScanSource: projection=[l_suppkey, l_extendedprice, l_discount, l_shipdate], predicate=l_shipdate@10 >= 1996-01-01 AND l_shipdate@10 < 1996-04-01 AND DynamicFilter [ l_suppkey@2 >= 1 AND l_suppkey@2 <= 1000 ], filter=l_shipdate@3 >= 1996-01-01 AND l_shipdate@3 < 1996-04-01, projection=[l_suppkey@0, l_extendedprice@1, l_discount@2] (22,996 rows)
  scalar subquery:
*   PhysicalHashAggregate: gby=[], aggr=[max(revenue0.total_revenue)] (1 row)
*     PhysicalHashAggregate: gby=[l_suppkey@0 as l_suppkey], aggr=[sum(lineitem.l_extendedprice * Int64(1) - lineitem.l_discount)] (1,000 rows)
*       ParquetScanSource: projection=[l_suppkey, l_extendedprice, l_discount, l_shipdate], predicate=l_shipdate@10 >= 1996-01-01 AND l_shipdate@10 < 1996-04-01, filter=l_shipdate@3 >= 1996-01-01 AND l_shipdate@3 < 1996-04-01, projection=[l_suppkey@0, l_extendedprice@1, l_discount@2] (22,996 rows)
```

</td>
</tr>
</table>

Likely cost: Ours scans the 22,996 `lineitem` rows of the quarter and aggregates them twice, for the join and for the `max` of the scalar subquery. With `WITH revenue0 AS MATERIALIZED (...)`, which the pipeline engine executes natively, q15 takes 5.66 ms instead of 9.30 ms (39%).

### q17: rule gap

- DuckDB: 3 scans; 2 hash joins; 2 aggregations; pipeline breaks: 2 aggregations, 2 join builds.
- Ours: 3 scans; 2 hash joins; 2 aggregations; pipeline breaks: 2 aggregations, 2 join builds.

**1. `IS NOT NULL` filters of join keys** (rule gap, follow-up #117; likely impact: none)

- scans: the scan of `lineitem` (`l_partkey, l_quantity`): DuckDB filters `l_partkey` with `IS NOT NULL`

<table>
<tr><th>DuckDB</th><th>ours</th></tr>
<tr>
<td>

```text
  UNGROUPED_AGGREGATE Aggregates: sum(#0) ~1 row
    HASH_JOIN Join Type: INNER; Conditions: p_partkey IS NOT DISTINCT FROM p_partkey, SUBQUERY > CAST(l_quantity AS DOUBLE) ~481,011 rows
      HASH_GROUP_BY Groups: #0; Aggregates: avg(#1) ~481,011 rows
*       FILTER Expression: (l_partkey IS NOT NULL) ~481,011 rows
*         READ_PARQUET Projections: l_partkey, l_quantity ~601,264 rows
      ...
```

</td>
<td>

```text
  PhysicalHashAggregate: gby=[], aggr=[sum(lineitem.l_extendedprice)] (1 row)
    PhysicalHashJoin: join_type=LeftSemi, on=[(p_partkey@2, l_partkey@1)], filter=CAST(l_quantity@0 AS Decimal128(30, 15)) < Float64(0.2) * avg(lineitem.l_quantity)@1 (954 rows)
      ...
      PhysicalHashAggregate: gby=[l_partkey@0 as l_partkey], aggr=[avg(lineitem.l_quantity)] (23 rows)
*       ParquetScanSource: projection=[l_partkey, l_quantity], predicate=DynamicFilter [ l_partkey@1 >= 2 AND l_partkey@1 <= 19543 AND l_partkey@1 IN (SET) ] (9,607 rows)
```

</td>
</tr>
</table>

Likely cost: `l_partkey` has no nulls: the filter removes no row.

### q18: rule gap

- DuckDB: 4 scans; 3 hash joins; 2 aggregations; 1 top-N; pipeline breaks: 2 aggregations, 3 join builds, 1 top-N.
- Ours: 4 scans; 3 hash joins; 2 aggregations; 1 top-N; pipeline breaks: 2 aggregations, 3 join builds, 1 top-N.

**1. Join order** (rule gap, follow-up #115; likely impact: low)

- joins: the join order differs: DuckDB joins `(((agg(lineitem) ⋈ orders) ⋈ customer) ⋈ lineitem)`, `((agg(lineitem) ⋈ orders) ⋈ customer)`, `(agg(lineitem) ⋈ orders)`; ours joins `(((customer ⋈ orders) ⋈ lineitem) ⋈ agg(lineitem))`, `((customer ⋈ orders) ⋈ lineitem)`, `(customer ⋈ orders)`

<table>
<tr><th>DuckDB</th><th>ours</th></tr>
<tr>
<td>

```text
  TOP_N Top: 100; Order By: o_totalprice DESC, o_orderdate ASC ~100 rows
    HASH_GROUP_BY Groups: #0, #1, #2, #3, #4; Aggregates: sum(#5) ~120,252 rows
*     HASH_JOIN Join Type: INNER; Conditions: l_orderkey = o_orderkey ~120,252 rows
        READ_PARQUET Projections: l_orderkey, l_quantity ~601,264 rows
*       HASH_JOIN Join Type: INNER; Conditions: o_custkey = c_custkey ~30,000 rows
*         HASH_JOIN Join Type: SEMI; Conditions: o_orderkey = #0 ~120,252 rows
            READ_PARQUET Projections: o_orderkey, o_custkey, o_totalprice, o_orderdate; Filters: optional: Dynamic Filter (o_totalprice) ~150,000 rows
            FILTER Expression: (sum(l_quantity) > 300.00) ~120,252 rows
              ...
          READ_PARQUET Projections: c_custkey, c_name ~14,992 rows
```

</td>
<td>

```text
  PhysicalTopK: fetch=100, expr=[o_totalprice@4 DESC, o_orderdate@3 ASC NULLS LAST], dynamic_filter=true (100 rows)
    PhysicalHashAggregate: gby=[c_name@1 as c_name, c_custkey@0 as c_custkey, o_orderkey@2 as o_orderkey, o_orderdate@4 as o_orderdate, o_totalprice@3 as o_totalprice], aggr=[sum(lineitem.l_quantity)] (381 rows)
*     PhysicalHashJoin: join_type=RightSemi, on=[(l_orderkey@0, o_orderkey@2)] (2,667 rows)
        PhysicalFilter: sum(lineitem.l_quantity)@1 > 300.00, projection=[l_orderkey@0] (384 rows)
          ...
*       PhysicalHashJoin: join_type=Inner, on=[(o_orderkey@2, l_orderkey@0)] (2,667 rows)
*         PhysicalHashJoin: join_type=Inner, on=[(c_custkey@0, o_custkey@1)] (381 rows)
            ParquetScanSource: projection=[c_custkey, c_name] (15,000 rows)
            ParquetScanSource: projection=[o_orderkey, o_custkey, o_totalprice, o_orderdate], predicate=DynamicFilter [ o_custkey@1 >= 1 AND o_custkey@1 <= 15000 ] AND DynamicFilter [ o_orderkey@0 >= 1573 AND o_orderkey@0 <= 598661 AND hash_lookup(o_orderkey@0) ] AND DynamicFilter [ empty ] (381 rows)
          ParquetScanSource: projection=[l_orderkey, l_quantity], predicate=DynamicFilter [ l_orderkey@0 >= 1573 AND l_orderkey@0 <= 598661 AND hash_lookup(l_orderkey@0) ] AND DynamicFilter [ l_orderkey@0 >= 1573 AND l_orderkey@0 <= 598661 AND hash_lookup(l_orderkey@0) ] (2,667 rows)
```

</td>
</tr>
</table>

Likely cost: DuckDB semi-joins `orders` with the `IN` subquery (the orders of more than 300 units) before the other joins; our dynamic filters already cut `orders` to about 380 rows and `lineitem` to 2,667 before our joins. With DuckDB's join order written out, q18 takes 13.15 ms instead of 13.72 ms (4%).

### q20: rule gap

- DuckDB: 5 scans; 5 hash joins; 2 aggregations; 1 sort; pipeline breaks: 2 aggregations, 5 join builds, 1 shared subplan, 1 sort.
- Ours: 5 scans; 4 hash joins; 1 aggregation; 1 sort; pipeline breaks: 1 aggregation, 4 join builds, 1 sort.

**1. Correlated subqueries decorrelated by their outer keys** (rule gap, follow-up #116; likely impact: none)

- joins: DuckDB joins the shared outer rows with the subquery evaluated for their keys; ours joins the outer rows with the subquery over all of its rows
- aggregations: the aggregations differ: DuckDB has an aggregation by 2 keys of `lineitem`, `part`, `partsupp`, an aggregation by 2 keys of `part`, `partsupp` that ours has not, ours an aggregation by 2 keys of `lineitem` that DuckDB has not
- pipeline breaks: the pipeline breaks differ (DuckDB / ours): aggregation 2 / 1, join build 5 / 4, shared subplan 1 / 0

<table>
<tr><th>DuckDB</th><th>ours</th></tr>
<tr>
<td>

```text
  ORDER_BY Order By: s_name ASC ~37 rows
*   HASH_JOIN Join Type: RIGHT_SEMI; Conditions: #0 = s_suppkey ~37 rows
*     CTE CTE Name: __duckdb_delim_65; Table Index: 65; Execution Mode: PIPELINE_DEPENDENT ~24,049 rows
        ...
*       HASH_JOIN Join Type: INNER; Conditions: ps_suppkey IS NOT DISTINCT FROM ps_suppkey, ps_partkey IS NOT DISTINCT FROM ps_partkey, SUBQUERY < CAST(ps_availqty AS DECIMAL(38,3)) ~24,049 rows
*         HASH_GROUP_BY Groups: #0, #1; Aggregates: sum(#2) ~24,049 rows
*           HASH_JOIN Join Type: SEMI; Conditions: l_partkey = ps_partkey, l_suppkey = ps_suppkey ~24,050 rows
              READ_PARQUET Projections: l_partkey, l_suppkey, l_quantity; Filters: ((l_shipdate >= '1994-01-01'::DATE) AND (l_shipdate < '1995-01-01'::DATE)) ~120,252 rows
*             HASH_GROUP_BY Groups: #0, #1 ~15,999 rows
                ...
          CTE_SCAN CTE Index: 65 ~16,000 rows
      HASH_JOIN Join Type: INNER; Conditions: s_nationkey = n_nationkey ~186 rows
        ...
```

</td>
<td>

```text
  PhysicalSort: expr=[s_name@0 ASC NULLS LAST] (17 rows)
*   PhysicalHashJoin: join_type=LeftSemi, on=[(s_suppkey@0, ps_suppkey@0)] (17 rows)
      PhysicalHashJoin: join_type=Inner, on=[(n_nationkey@0, s_nationkey@3)] (44 rows)
        ...
*     PhysicalHashJoin: join_type=LeftSemi, on=[(ps_partkey@0, l_partkey@1), (ps_suppkey@1, l_suppkey@2)], filter=CAST(ps_availqty@0 AS Float64) > Float64(0.5) * sum(lineitem.l_quantity)@1 (18 rows)
        PhysicalHashJoin: join_type=RightSemi, on=[(p_partkey@0, ps_partkey@0)] (38 rows)
          ...
*       PhysicalHashAggregate: gby=[l_partkey@0 as l_partkey, l_suppkey@1 as l_suppkey], aggr=[sum(lineitem.l_quantity)] (18 rows)
          ...
```

</td>
</tr>
</table>

Likely cost: The dynamic filters restrict our aggregation of `lineitem` to the keys of the outer rows: its `lineitem` scan returns 27 rows. Both plans scan `lineitem` once for the subquery: nothing to gain.

### q21: rule gap

- DuckDB: 6 scans; 5 hash joins; 1 aggregation; 1 top-N; pipeline breaks: 1 aggregation, 5 join builds, 1 top-N.
- Ours: 6 scans; 5 hash joins; 1 aggregation; 1 top-N; pipeline breaks: 1 aggregation, 5 join builds, 1 top-N.

**1. `IS NOT NULL` filters of join keys** (rule gap, follow-up #117; likely impact: none)

- scans: the scan of `lineitem` (`l_orderkey, l_suppkey`): DuckDB filters `l_orderkey, l_suppkey` with `IS NOT NULL`
- scans: the scan of `lineitem` (`l_commitdate, l_orderkey, l_receiptdate, l_suppkey`): DuckDB filters `l_orderkey, l_suppkey` with `IS NOT NULL`

<table>
<tr><th>DuckDB</th><th>ours</th></tr>
<tr>
<td>

```text
  TOP_N Top: 100; Order By: numwait DESC, s_name ASC ~100 rows
    HASH_GROUP_BY Groups: #0; Aggregates: count_star() ~123 rows
      HASH_JOIN Join Type: RIGHT_SEMI; Conditions: l_orderkey = l_orderkey, l_suppkey IS DISTINCT FROM l_suppkey ~180 rows
*       READ_PARQUET Projections: l_orderkey, l_suppkey; Filters: (l_orderkey IS NOT NULL), (l_suppkey IS NOT NULL) ~481,011 rows
        HASH_JOIN Join Type: RIGHT_ANTI; Conditions: l_orderkey IS NOT DISTINCT FROM l_orderkey, l_suppkey IS DISTINCT FROM l_suppkey ~901 rows
*         FILTER Expression: (l_receiptdate > l_commitdate) ~96,202 rows
*           READ_PARQUET Projections: l_orderkey, l_suppkey, l_commitdate, l_receiptdate; Filters: (l_orderkey IS NOT NULL), (l_suppkey IS NOT NULL) ~481,011 rows
          ...
```

</td>
<td>

```text
  PhysicalTopK: fetch=100, expr=[numwait@1 DESC, s_name@0 ASC NULLS LAST], dynamic_filter=true (45 rows)
    PhysicalHashAggregate: gby=[s_name@0 as s_name], aggr=[count(Int64(1))] (45 rows)
      PhysicalHashJoin: join_type=RightAnti, on=[(l_orderkey@0, l_orderkey@1)], filter=l_suppkey@1 != l_suppkey@0 (426 rows)
*       ParquetScanSource: projection=[l_orderkey, l_suppkey, l_commitdate, l_receiptdate], predicate=l_receiptdate@12 > l_commitdate@11, filter=l_receiptdate@3 > l_commitdate@2, projection=[l_orderkey@0, l_suppkey@1] (380,417 rows)
        PhysicalHashJoin: join_type=LeftSemi, on=[(l_orderkey@1, l_orderkey@0)], filter=l_suppkey@1 != l_suppkey@0 (7,471 rows)
          ...
*         ParquetScanSource: projection=[l_orderkey, l_suppkey], predicate=DynamicFilter [ l_orderkey@0 >= 264 AND l_orderkey@0 <= 599976 AND hash_lookup(l_orderkey@0) ] (36,484 rows)
```

</td>
</tr>
</table>

Likely cost: `l_orderkey` and `l_suppkey` have no nulls: the filters remove no row.

**2. Join order** (rule gap, follow-up #115; likely impact: medium)

- joins: the join order differs: DuckDB joins `(((((nation ⋈ supplier) ⋈ lineitem) ⋈ orders) ⋈ lineitem) ⋈ lineitem)`, `((((nation ⋈ supplier) ⋈ lineitem) ⋈ orders) ⋈ lineitem)`, `(((nation ⋈ supplier) ⋈ lineitem) ⋈ orders)`, `((nation ⋈ supplier) ⋈ lineitem)`, `(nation ⋈ supplier)`; ours joins `(((((lineitem ⋈ supplier) ⋈ orders) ⋈ nation) ⋈ lineitem) ⋈ lineitem)`, `((((lineitem ⋈ supplier) ⋈ orders) ⋈ nation) ⋈ lineitem)`, `(((lineitem ⋈ supplier) ⋈ orders) ⋈ nation)`, `((lineitem ⋈ supplier) ⋈ orders)`, `(lineitem ⋈ supplier)`

<table>
<tr><th>DuckDB</th><th>ours</th></tr>
<tr>
<td>

```text
  TOP_N Top: 100; Order By: numwait DESC, s_name ASC ~100 rows
    HASH_GROUP_BY Groups: #0; Aggregates: count_star() ~123 rows
*     HASH_JOIN Join Type: RIGHT_SEMI; Conditions: l_orderkey = l_orderkey, l_suppkey IS DISTINCT FROM l_suppkey ~180 rows
        READ_PARQUET Projections: l_orderkey, l_suppkey; Filters: (l_orderkey IS NOT NULL), (l_suppkey IS NOT NULL) ~481,011 rows
*       HASH_JOIN Join Type: RIGHT_ANTI; Conditions: l_orderkey IS NOT DISTINCT FROM l_orderkey, l_suppkey IS DISTINCT FROM l_suppkey ~901 rows
          FILTER Expression: (l_receiptdate > l_commitdate) ~96,202 rows
            READ_PARQUET Projections: l_orderkey, l_suppkey, l_commitdate, l_receiptdate; Filters: (l_orderkey IS NOT NULL), (l_suppkey IS NOT NULL) ~481,011 rows
*         HASH_JOIN Join Type: INNER; Conditions: l_orderkey = o_orderkey ~4,509 rows
*           HASH_JOIN Join Type: INNER; Conditions: l_suppkey = s_suppkey ~22,547 rows
              FILTER Expression: (l_receiptdate > l_commitdate) ~120,252 rows
                READ_PARQUET Projections: l_orderkey, l_suppkey, l_commitdate, l_receiptdate ~601,264 rows
*             HASH_JOIN Join Type: INNER; Conditions: s_nationkey = n_nationkey ~186 rows
                READ_PARQUET Projections: s_suppkey, s_name, s_nationkey ~992 rows
                READ_PARQUET Projections: n_nationkey; Filters: (n_name = 'SAUDI ARABIA') ~3 rows
            READ_PARQUET Projections: o_orderkey; Filters: (o_orderstatus = 'F') ~30,000 rows
```

</td>
<td>

```text
  PhysicalTopK: fetch=100, expr=[numwait@1 DESC, s_name@0 ASC NULLS LAST], dynamic_filter=true (45 rows)
    PhysicalHashAggregate: gby=[s_name@0 as s_name], aggr=[count(Int64(1))] (45 rows)
*     PhysicalHashJoin: join_type=RightAnti, on=[(l_orderkey@0, l_orderkey@1)], filter=l_suppkey@1 != l_suppkey@0 (426 rows)
        ParquetScanSource: projection=[l_orderkey, l_suppkey, l_commitdate, l_receiptdate], predicate=l_receiptdate@12 > l_commitdate@11, filter=l_receiptdate@3 > l_commitdate@2, projection=[l_orderkey@0, l_suppkey@1] (380,417 rows)
*       PhysicalHashJoin: join_type=LeftSemi, on=[(l_orderkey@1, l_orderkey@0)], filter=l_suppkey@1 != l_suppkey@0 (7,471 rows)
*         PhysicalHashJoin: join_type=Inner, on=[(n_nationkey@0, s_nationkey@1)] (7,797 rows)
            ParquetScanSource: projection=[n_nationkey, n_name], predicate=n_name@1 = SAUDI ARABIA, filter=n_name@1 = SAUDI ARABIA, projection=[n_nationkey@0] (1 row)
*           PhysicalHashJoin: join_type=Inner, on=[(o_orderkey@0, l_orderkey@2)] (7,797 rows)
              ParquetScanSource: projection=[o_orderkey, o_orderstatus], predicate=o_orderstatus@2 = F, filter=o_orderstatus@1 = F, projection=[o_orderkey@0] (72,889 rows)
*             PhysicalHashJoin: join_type=Inner, on=[(s_suppkey@0, l_suppkey@1)] (7,797 rows)
                ParquetScanSource: projection=[s_suppkey, s_name, s_nationkey], predicate=DynamicFilter [ s_nationkey@3 >= 20 AND s_nationkey@3 <= 20 ] (45 rows)
                ParquetScanSource: projection=[l_orderkey, l_suppkey, l_commitdate, l_receiptdate], predicate=l_receiptdate@12 > l_commitdate@11 AND DynamicFilter [ l_suppkey@2 >= 21 AND l_suppkey@2 <= 968 AND l_suppkey@2 IN (SET) ] AND DynamicFilter [ l_orderkey@0 >= 2 AND l_orderkey@0 <= 599976 AND hash_lookup(l_orderkey@0) ], filter=l_receiptdate@3 > l_commitdate@2, projection=[l_orderkey@0, l_suppkey@1] (7,797 rows)
          ParquetScanSource: projection=[l_orderkey, l_suppkey], predicate=DynamicFilter [ l_orderkey@0 >= 264 AND l_orderkey@0 <= 599976 AND hash_lookup(l_orderkey@0) ] (36,484 rows)
```

</td>
</tr>
</table>

Likely cost: Our anti join (`NOT EXISTS` of `l3`) builds its hash table on the 380,417 `l3` rows with `l_receiptdate > l_commitdate` and probes it with 7,471 rows; DuckDB applies the anti join first and builds on the ~4,509 (estimated) rows of the inner joins. With DuckDB's join order written out, q21 takes 19.10 ms instead of 21.31 ms (10%).

### q22: rule gap

- DuckDB: 3 scans; 2 hash joins; 3 aggregations; 1 sort; pipeline breaks: 3 aggregations, 2 join builds, 1 scalar subquery, 1 shared subplan, 1 sort.
- Ours: 3 scans; 1 hash join; 2 aggregations; 1 sort; pipeline breaks: 2 aggregations, 1 join build, 1 scalar subquery, 1 sort.

**1. Correlated subqueries decorrelated by their outer keys** (rule gap, follow-up #116; likely impact: none)

- joins: DuckDB joins the shared outer rows with the subquery evaluated for their keys; ours joins the outer rows with the subquery over all of its rows
- aggregations: the aggregations differ: DuckDB has an aggregation by 1 key of `customer` (2), `orders`, an aggregation by 1 key of `customer` that ours has not, ours an aggregation by 1 key of `customer`, `orders` that DuckDB has not
- pipeline breaks: the pipeline breaks differ (DuckDB / ours): aggregation 3 / 2, join build 2 / 1, shared subplan 1 / 0

<table>
<tr><th>DuckDB</th><th>ours</th></tr>
<tr>
<td>

```text
  ORDER_BY Order By: cntrycode ASC ~299 rows
*   HASH_GROUP_BY Groups: #0; Aggregates: count_star(), sum(#1) ~299 rows
*     CTE CTE Name: __duckdb_delim_49; Table Index: 49; Execution Mode: PIPELINE_DEPENDENT ~299 rows
        ...
*       FILTER Expression: (NOT SUBQUERY) ~299 rows
*         HASH_JOIN Join Type: MARK; Conditions: c_custkey IS NOT DISTINCT FROM c_custkey ~1,499 rows
            CTE_SCAN CTE Index: 49 ~1,499 rows
*           HASH_JOIN Join Type: SEMI; Conditions: o_custkey = c_custkey ~14,998 rows
              READ_PARQUET Projections: o_custkey ~150,000 rows
*             HASH_GROUP_BY Groups: #0 ~1,499 rows
                ...
  scalar subquery:
*   UNGROUPED_AGGREGATE Aggregates: sum(#0), count(#0) ~1 row
      ...
```

</td>
<td>

```text
  PhysicalSort: expr=[cntrycode@0 ASC NULLS LAST] (7 rows)
*   PhysicalHashAggregate: gby=[cntrycode@0 as cntrycode], aggr=[count(Int64(1)), sum(custsale.c_acctbal)] (7 rows)
*     PhysicalHashJoin: join_type=LeftAnti, on=[(c_custkey@0, o_custkey@0)] (609 rows)
        ParquetScanSource: projection=[c_custkey, c_phone, c_acctbal], predicate=substr(c_phone@4, 1, 2) IN (SET) AND CAST(c_acctbal@5 AS Decimal128(19, 6)) > scalar_subquery(5016.966223), filter=substr(c_phone@1, 1, 2) IN (SET) AND CAST(c_acctbal@2 AS Decimal128(19, 6)) > scalar_subquery(5016.966223) (1,880 rows)
        ParquetScanSource: projection=[o_custkey], predicate=DynamicFilter [ o_custkey@1 >= 13 AND o_custkey@1 <= 14995 AND hash_lookup(o_custkey@1) ] (18,077 rows)
  scalar subquery:
*   PhysicalHashAggregate: gby=[], aggr=[avg(customer.c_acctbal)] (1 row)
      ...
```

</td>
</tr>
</table>

Likely cost: Our anti join builds on the 1,880 customers and probes `orders`, which its dynamic filter cuts to 18,077 rows; DuckDB semi-joins `orders` with the distinct keys of the shared customers. The same scans: nothing to gain.

### hits0: cost/statistics-driven

- DuckDB: 1 scan; 1 aggregation; pipeline breaks: 1 aggregation.
- Ours: 0 scans; pipeline breaks: none.

**1. Aggregation answered from exact statistics** (cost/statistics-driven; likely impact: none)

- scans: `hits` is scanned once in DuckDB, never in ours
- aggregations: the aggregations differ: DuckDB has an ungrouped aggregation of `hits` that ours has not, ours none that DuckDB has not
- pipeline breaks: the pipeline breaks differ (DuckDB / ours): aggregation 1 / 0

<table>
<tr><th>DuckDB</th><th>ours</th></tr>
<tr>
<td>

```text
* UNGROUPED_AGGREGATE Aggregates: count_star() ~1 row
*   READ_PARQUET ~1,000,000 rows
```

</td>
<td>

```text
* PhysicalProjection: expr=[1000000 as count(*)] (1 row)
*   PhysicalPlaceholderRow (1 row)
```

</td>
</tr>
</table>

Likely cost: Our plan reads no file (0.06 ms); DuckDB scans the 1,000,000 rows. Our plan is ahead.

### hits6: cost/statistics-driven

- DuckDB: 1 scan; 1 aggregation; pipeline breaks: 1 aggregation.
- Ours: 0 scans; pipeline breaks: none.

**1. Aggregation answered from exact statistics** (cost/statistics-driven; likely impact: none)

- scans: `hits` is scanned once in DuckDB, never in ours
- aggregations: the aggregations differ: DuckDB has an ungrouped aggregation of `hits` that ours has not, ours none that DuckDB has not
- pipeline breaks: the pipeline breaks differ (DuckDB / ours): aggregation 1 / 0

<table>
<tr><th>DuckDB</th><th>ours</th></tr>
<tr>
<td>

```text
* UNGROUPED_AGGREGATE Aggregates: min(#0), max(#0) ~1 row
*   READ_PARQUET Projections: EventDate ~1,000,000 rows
```

</td>
<td>

```text
* PhysicalProjection: expr=[2013-07-01 as min(hits.EventDate), 2013-07-31 as max(hits.EventDate)] (1 row)
*   PhysicalPlaceholderRow (1 row)
```

</td>
</tr>
</table>

Likely cost: Our plan answers `min` and `max` from the statistics of the Parquet files and reads no file; DuckDB scans `EventDate`. Our plan is ahead.

### hits23: rule gap

- DuckDB: 2 scans; 1 hash join; 1 sort; 1 top-N; pipeline breaks: 1 join build, 1 sort, 1 top-N.
- Ours: 1 scan; 1 top-N; pipeline breaks: 1 top-N.

**1. Late materialization of top-N** (rule gap, follow-up #113; likely impact: high)

- scans: the scan of `hits` (`AdvEngineID, ClientIP, CounterID, DontCountHits, EventDate, EventTime, IsRefresh, MobilePhone, MobilePhoneModel, RegionID, ResolutionWidth, SearchEngineID, SearchPhrase, Title, URL, UserID, WatchID`): only our top-N pushes its dynamic filter into it (on `EventTime, WatchID`)
- scans: `hits` is scanned twice in DuckDB, once in ours
- joins: DuckDB joins the top rows back to a second scan by their row ids (`file_index`, `file_row_number`) to read their other columns
- top-N and limits: DuckDB has a sort by 2 keys of `hits` that ours has not, ours none that DuckDB has not
- pipeline breaks: the pipeline breaks differ (DuckDB / ours): join build 1 / 0, sort 1 / 0

<table>
<tr><th>DuckDB</th><th>ours</th></tr>
<tr>
<td>

```text
* ORDER_BY Order By: EventTime ASC, WatchID ASC ~1,000,000 rows
*   HASH_JOIN Join Type: SEMI; Conditions: file_index = file_index, file_row_number = file_row_number ~1,000,000 rows
*     READ_PARQUET Projections: WatchID, UserID, CounterID, RegionID, ClientIP, EventTime, EventDate, AdvEngineID, SearchEngineID, SearchPhrase, URL, Title, IsRefresh, DontCountHits, ResolutionWidth, MobilePhone, MobilePhoneModel; Filters: optional: contains(URL, 'google') ~1,000,000 rows
      TOP_N Top: 10; Order By: EventTime ASC, WatchID ASC ~10 rows
*       READ_PARQUET Projections: WatchID, EventTime; Filters: optional: Dynamic Filter (EventTime), contains(URL, 'google') ~200,000 rows
```

</td>
<td>

```text
  PhysicalTopK: fetch=10, expr=[EventTime@5 ASC NULLS LAST, WatchID@0 ASC NULLS LAST], dynamic_filter=true (10 rows)
*   ParquetScanSource: projection=[WatchID, UserID, CounterID, RegionID, ClientIP, EventTime, EventDate, AdvEngineID, SearchEngineID, SearchPhrase, URL, Title, IsRefresh, DontCountHits, ResolutionWidth, MobilePhone, MobilePhoneModel], predicate=URL@10 LIKE %google% AND DynamicFilter [ EventTime@5 < 1372638146 OR EventTime@5 = 1372638146 AND WatchID@0 < 2072021522208071178 ], filter=URL@10 LIKE %google% (33,790 rows)
```

</td>
</tr>
</table>

Likely cost: Our scan reads the 17 columns of the 33,790 rows with `URL LIKE '%google%'` (the dynamic filter of the TopK skips rows once it has 10); DuckDB reads `WatchID`, `EventTime` and `URL` for the top-N, then all columns of 10 rows. Reading the sort keys only, what DuckDB's first scan reads, takes 8.35 ms instead of 21.63 ms (61%): most of hits23 is the late materialization it lacks.
