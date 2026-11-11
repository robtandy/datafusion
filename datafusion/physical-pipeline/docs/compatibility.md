<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Compatibility with DataFusion

_The project owner decided this on 2026-11-11. It replaces the earlier requirement of byte-for-byte parity with DataFusion (and [Error parity](error-parity.md)). See the [history](#history-why-the-engine-used-to-fall-back) for why that requirement existed and what it cost._

## Policy

The pipeline engine is an execution engine in its own right, plugged into DataFusion after logical planning. Its results are its own, just as DataFusion's results are DataFusion's.

**Required:**

- **Correct results.** Every query returns a correct answer to its SQL.
- **The same output schema** as the physical plan being lowered: field names, data types, nullability and metadata. This is the contract with DataFusion that lets the engine plug in after logical planning.
- **No fallbacks.** Every DataFusion node runs natively, and a node is never left to DataFusion (in a `BridgeOperator`) to reproduce DataFusion's behaviour. The only nodes that may stay in DataFusion are user-defined `ExecutionPlan`s, which are the user's own code.
- **Memory limits are respected.** Native operators reserve their memory in the session's `MemoryPool`, and fail with `ResourcesExhausted` when it runs out. Spilling to disk comes later.

**Ours to decide** (these may differ from DataFusion):

- floating-point rounding that comes from the order of evaluation, for example in a `SUM` of floats;
- row order wherever SQL doesn't define it: without an `ORDER BY`, and among ties under `ORDER BY … LIMIT`;
- behaviour that depends on batch sizes, partitioning or parallelism, for example `MIN`/`MAX` over structs;
- which error is raised when several are possible;
- whether an error is raised for rows that the engine never evaluates, for example rows skipped by dynamic filters or beyond a `LIMIT`;
- capacity limits, for example succeeding where DataFusion overflows a 32-bit offset;
- error messages.

## Testing

- Compare results with DataFusion's **by meaning**, not byte for byte:
  - the schemas are equal;
  - the rows are the same, compared as a multiset, or in order only where SQL defines the order;
  - floating-point values are equal within a relative tolerance;
  - under `ORDER BY … LIMIT`, rows that tie with the last row may differ.
- When one engine fails and the other doesn't, the test must show that this is one of the differences allowed above, not a wrong answer.
- `test_utils::assert_same_results_as_datafusion` implements these comparisons (it is being added; until then, tests compare canonicalized results, e.g. with `assert_byte_eq_with_datafusion` for queries whose results are unambiguous).

## History: why the engine used to fall back

Until 2026-11-11 the engine had to return byte-identical results _and_ raise the same errors as DataFusion. Matching errors was the hard part, because the two engines don't touch exactly the same rows, or touch them in the same order:

- **Order and parallelism.** DataFusion processes each partition on one thread, one batch after another. The pipeline engine runs a pipeline on many threads, each of which grabs the next chunk of 2,048 rows.
- **Batch boundaries.** Some errors, and even some results, depend on the batch they happen in.
- **Early stops.** Under a `LIMIT`, either engine may stop after a different number of rows.
- **Dynamic filters.** A TopK or a hash join publishes a filter while the query runs, and the scans below it skip whatever the filter rules out. Each engine tightens its filter at different moments, so each one skips different parts of the input.

An example: `SELECT * FROM t ORDER BY k LIMIT 1` over a Parquet file whose second row group holds a value that fails to decode.

- DataFusion skips that row group, thanks to its TopK's filter, and returns a row.
- The pipeline engine reads both row groups at once on two threads, decodes the bad value and fails.

To guarantee identical errors, the planner asked one question before running any piece of a plan natively: "can this piece fail on any row?" If it might, the planner left the piece to DataFusion.

That rule drove most of the fallbacks, including:

- casts and divisions on several threads;
- user-defined functions;
- residual join filters;
- TopKs and joins whose dynamic filters prune Parquet scans;
- aggregates whose results depend on batch boundaries.

Proving that pieces couldn't fail took a lot of review effort, and it kept parts of TPC-H in DataFusion. Relaxing the rule is what makes "no fallbacks" possible.

## Decision log

| Date       | Decision                                                                                                                                                                                                                                                                                         |
| :--------- | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 2026-11-11 | Read and decode errors on Parquet rows that dynamic filters prune differently are out of scope.                                                                                                                                                                                                  |
| 2026-11-11 | That decision is superseded: byte and error parity with DataFusion are no longer required. Results must be correct, with the same schema. Behaviour that SQL leaves implementation-defined is the engine's own. There are no fallbacks. Memory limits are enforced by failing with `ResourcesExhausted`. |
