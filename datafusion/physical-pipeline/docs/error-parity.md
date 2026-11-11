<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Error parity: why parts of a plan stay in DataFusion

_A living document. The latest decision is from 2026-11-11 (see the [decision log](#decision-log))._

## Summary

- **The goal.** The pipeline engine must be a drop-in replacement for DataFusion. It must return the same results and raise the same errors.
- **Results are easy, errors are subtle.** Matching results is straightforward. Matching errors is hard, because the two engines don't touch exactly the same rows, or touch them in the same order.
- **The planner asks one question.** Before it runs a piece of a plan natively, it asks: **can this piece fail on any row?**
  - If it can't, it doesn't matter which rows each engine touches, so the piece runs natively.
  - If it might, the piece stays in DataFusion, in a `BridgeOperator`. That reproduces DataFusion's behaviour exactly, error included.
- **What "trusted" means.** In the code and in reviews, "trusted" means only this: we can prove the piece never fails on any row, or that both engines would report its failure identically.
- **Decision (2026-11-11).** Errors raised while _reading and decoding Parquet files_ are out of scope for parity when the engines' dynamic filters skip different parts of a file.
  - Results stay byte-identical.
  - Errors from the query's own expressions still match DataFusion exactly.

## 1. What parity means

|             | Requirement                                                                                                                                                                                         |
| :---------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Results** | Exactly the same schema and bytes as DataFusion. Row order only matters under a top-level `ORDER BY`. Always required.                                                                              |
| **Errors**  | If DataFusion fails, we fail with the same kind of error. If DataFusion succeeds, we succeed. Required, with the exception in [§6](#6-decision-read-errors-under-dynamic-filters-are-out-of-scope). |

## 2. Why errors are the hard part

Both engines evaluate the same expressions over the same data, but not in the same way:

- **Order and parallelism.** DataFusion processes each partition on one thread, one batch after another. The pipeline engine runs a pipeline on many threads, each of which grabs the next chunk of input (2,048 rows) as it goes.
- **Batch boundaries.** DataFusion evaluates expressions on its batches of 8,192 rows; the pipeline engine evaluates them on its chunks. Some errors depend on which batch they happen in.
- **Early stops.** Under a `LIMIT`, either engine may stop before it has read all of its input, after a different number of rows.
- **Dynamic filters.** A TopK (`ORDER BY … LIMIT`) or a hash join publishes a filter while the query runs, for example "only rows with `k ≤ 42` can still matter". The scans below it use that filter to skip files, row groups and rows they haven't read yet. Each engine tightens its filter at different moments, so each one skips different parts of the input.

If no row can cause an error, none of this matters: both engines return the same rows. If one row can, the outcome depends on whether an engine _touches_ that row.

## 3. A concrete example

`SELECT * FROM t ORDER BY k LIMIT 1` over a Parquet file with two row groups:

- Row group 1 has `k` = 0…999.
- Row group 2 has `k` = 1000…1999, plus one value that fails to decode. For example, a legacy INT96 timestamp after the year 2262, which overflows a nanosecond timestamp.

| Engine          | What happens                                                                                                                                                                                              | Outcome     |
| :-------------- | :-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :---------- |
| DataFusion      | Reads row group 1. Its TopK learns that the best `k` is 0 and tells the scan "only `k ≤ 0` matters". Row group 2's statistics say it starts at 1000, so it is skipped and the bad value is never decoded. | Returns `0` |
| Pipeline engine | Thread A reads row group 1 while thread B reads row group 2 at the same moment, before any filter exists. Thread B decodes the bad value.                                                                 | **Error**   |

The same query on the same data gives different outcomes. Had both engines succeeded, the answer would have been the same; the only difference is whether an engine touched the bad row.

## 4. How the planner decides: "can this piece fail on any row?"

`PipelineQueryPlanner` first plans the query with DataFusion. It then replaces DataFusion nodes with native operators wherever it can. For every candidate, it judges what could fail:

| Example                                       | Can it fail?                            | What the planner does                                                                                                                                |
| :-------------------------------------------- | :-------------------------------------- | :--------------------------------------------------------------------------------------------------------------------------------------------------- |
| `a + b` on integers (wraps on overflow)       | No                                      | Runs it natively                                                                                                                                     |
| `CAST(s AS INT)` on strings                   | Yes, on some rows                       | Leaves it in DataFusion when it runs on several threads: the error names the value of the row that fails first                                       |
| Decimal `l_extendedprice * (1 - l_discount)`  | Only by overflowing                     | Runs it natively: whichever row overflows first, both engines raise `ArithmeticOverflow`                                                             |
| Decimal division `x / y`                      | By dividing by zero _or_ by overflowing | Leaves it in DataFusion when it runs on several threads: which of the two kinds of error comes first depends on which rows each engine reaches first |
| A user-defined function `my_udf(x)`           | Unknown                                 | Leaves it in DataFusion: we can't see inside it                                                                                                      |
| Built-in functions (`date_part`, `substr`, …) | Known for each function                 | Runs them natively when the function's failures are known (see `TrustedFunctions`)                                                                   |

The rules live in two files:

- `src/operators/adaptive_filter.rs`: `failures`, `Failures`, `depends_on_batches` and `TrustedFunctions`.
- `src/planner.rs`: `Judgement`, `subtree_failures`, `source_may_fail` and `file_scan_failures`.

## 5. Dynamic filters and file scans

A TopK or hash join whose dynamic filter prunes a scan is only safe to run natively if nothing _below_ it can fail on the rows that the two engines skip differently. That includes the scan itself:

- **The scan's own expressions**, i.e. its pushed-down predicate and projection, are judged like any other expressions.
- **Reading and decoding the file** is the hard part:
  - DataFusion's own Parquet reader does not fail on a well-formed file.
  - A **custom reader** is arbitrary code that could fail on any row. This is a `ParquetFileReaderFactory` that a user plugs in. Our reviewer built one that injects faults on purpose.
  - Some **conversions while decoding** can fail, for example INT96 timestamps that overflow, or dictionary indexes that overflow their type.
- **The planner cannot reliably tell DataFusion's reader from a custom one.**
  - DataFusion attaches a reader factory to every Parquet table scan it plans, including its own default one.
  - It offers no API for identifying which factory is attached: a `ParquetFileReaderFactory` cannot be downcast.
  - Checking the reader's printed (`Debug`) name is unsound, because a custom reader can print the same name.

## 6. Decision: read errors under dynamic filters are out of scope

The project owner decided this on 2026-11-11, to unblock the project's actual goal: unlocking performance and evaluating the morsel-driven, push-based execution approach.

It applies when a native TopK's or hash join's dynamic filter prunes a **Parquet** scan differently from DataFusion's.

**Still guaranteed:**

- Results are byte-identical to DataFusion's whenever both engines succeed.
- Errors from the query's own expressions match DataFusion exactly, as described in §4. That covers filters, projections, join keys, aggregates, and the scan's own predicate and projection.

**May differ (accepted):** whether an error is raised while _reading or decoding_ rows that one engine's dynamic filter skipped. In particular:

- errors from custom `ParquetFileReaderFactory` implementations;
- errors decoding a file's values, or converting them to the table's types. Examples are corrupt files, overflowing INT96 timestamps and dictionary index overflow.

In practice this only matters for malformed data or custom readers. Nothing changes for data that reads cleanly with DataFusion's reader.

**Not affected by this decision:**

- **CSV scans.** CSV values are parsed from text, where bad values are common, so CSV scans are still treated as able to fail, as before.
- **Expression errors on rows that dynamic filters prune differently.** An example is the decimal arithmetic on the probe side of a join. These stay in scope. TPC-H Q15 and Q17 over Parquet are being addressed in #53, by proving that the arithmetic cannot overflow.

### Rules for agents and reviewers

- When deciding whether a TopK or hash join with a dynamic filter can run natively, treat reading and decoding a Parquet scan as unable to fail. Judge only the scan's own expressions.
- Never identify components (reader factories, schema providers, functions) by their `Debug` text.
- Don't request changes for differences that only arise from reading or decoding rows that one engine's dynamic filter skipped. Tests for such cases should document the accepted behaviour, not assert parity.

## 7. Options that were considered

| Option                                                                                                                                                   | Error parity                                                   | Cost                                                    | Parquet TPC-H queries fully native                                                   |
| :------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------- | :------------------------------------------------------ | :----------------------------------------------------------------------------------- |
| **Relax: read errors are out of scope (chosen)**                                                                                                         | Everything except read and decode errors under dynamic filters | None                                                    | 19/22 (Q8, Q15 and Q17 are tracked in #53)                                           |
| Strict, with two small APIs in `datafusion-datasource-parquet`: `ParquetFileReaderFactory::is_builtin()` and a `ParquetSource::schema_provider()` getter | Full                                                           | An API change in the fork; it would need an upstream PR | 19/22                                                                                |
| Strict, without new APIs                                                                                                                                 | Full                                                           | None                                                    | 0/22: every TopK or join with a dynamic filter over Parquet falls back to DataFusion |

If this work is ever proposed upstream, the strict option with the two small APIs is the natural way back to full parity.

## Decision log

| Date       | Decision                                                                                                                                     | Issue / PR   |
| :--------- | :------------------------------------------------------------------------------------------------------------------------------------------- | :----------- |
| 2026-11-11 | Read and decode errors on Parquet rows that dynamic filters prune differently are out of scope. The scan's own expressions are still judged. | #44 / PR #48 |
