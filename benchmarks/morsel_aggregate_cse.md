<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Aggregate Accumulator CSE, `AVG`-from-`SUM`/`COUNT` Sharing & Direct `Decimal128` Aggregation (#208)

Before: `736c185ced6aff5fdc41dbd3e9b5df2c466280cd` (`morsel-execution` after #207).
After: `agent/morsel-issue-208` (aggregate accumulator CSE, `AVG`-from-`SUM`/`COUNT`
sharing with lazy null-count tracking, direct fixed-width `Decimal128`/`Int64`/`Float64`
accumulators, fused `Decimal128` arithmetic fast paths, and inline 2-column `Utf8View`
group interning).

## Executive Summary

- **TPC-H Q1 (SF=5, 8 threads, 5-round paired medians):**
  - **`morsel_process_work`:** `2,025.8 → 997.3 ms` (**`2.03×` faster**,
    `−1,028.5 ms` aggregate CPU, `−50.8%`).
  - **Exec wall time:** `726.1 → 593.6 ms` (**`1.22×` faster**, `−132.5 ms` vs.
    pre-#208 `morsel_pushdown`; **`1.28×` faster** than `stock_pushdown` at
    `760.6 ms`).
  - **Why:** Q1's 8 aggregates (`4× SUM`, `3× AVG`, `COUNT(*)`) share
    `SUM(l_quantity)` / `SUM(l_extendedprice)` + `COUNT(*)` (with zero-work
    lazy `CountCol` on non-null Parquet batches), run direct fixed-width
    `Decimal128` accumulators without `NullState` bitmap writes, evaluate
    `l_extendedprice * (1 - l_discount)` and `__common_expr_1 * (1 + l_tax)` in
    fused loops avoiding intermediate `Decimal128Array` allocations and
    compiler-rt `__muloti4` calls, and intern `(l_returnflag, l_linestatus)` via
    a 64-slot direct-mapped cache over inlined `Utf8View` keys.
- **TPC-H Q15 (SF=5):**
  - **`morsel_process_work`:** `590.5 → 530.4 ms` (**`1.11×` faster**, `−60.1 ms`
    CPU).
  - **Exec wall time:** `731.6 → 707.3 ms` (**`1.03×` faster**, `−24.3 ms`;
    **`1.32×` faster** than `stock_pushdown` at `936.1 ms`) via `DirectGroupedState`
    support for `Decimal128` fused `a * (1 - b)` aggregation over `Int64`
    `l_suppkey`.
- **TPC-H Q6 (SF=5):**
  - **`morsel_process_work`:** `190.5 → 173.3 ms` (**`1.10×` faster**, `−17.2 ms`
    CPU) via selection-aware fused ungrouped `Decimal128` multiplication
    (`l_extendedprice * l_discount`). Wall time stays decode-bound at
    `334.5 → 335.4 ms` (`1.51×` faster than `stock_pushdown` at `505.1 ms`).
- **TPC-H Q9 (SF=5):**
  - **`morsel_process_work`:** `2,415.8 → 2,379.2 ms` (`1.02×`, `−36.6 ms` CPU),
    wall time `1,460.9 → 1,455.5 ms` (`1.10×` faster than `stock_pushdown` at
    `1,603.1 ms`); remaining CPU is dominated by Parquet decode (`5,312 ms`) and
    the 5-way hash join probe pipeline.

## Method & Environment

- **Host:** `rob-tandy-dduck2`, Ubuntu 24.04.5 LTS, Linux `7.0.0-1012-aws`,
  16 ARM Neoverse-N1 cores (8 pinned via `taskset -c 0-7`), 61 GiB RAM.
- **Idle verification:** `vmstat -y 1 3` immediately before showed `97–99%` CPU
  idle; immediately after showed `95–100%` CPU idle.
- **Dataset & cache residency:** SF=5 TPC-H Parquet (`35` files, `100%` resident
  via `mincore`).
- **Build:** `cargo build --locked --profile release-nonlto -p datafusion --no-default-features --features parquet,sql,datetime_expressions,unicode_expressions --example morsel_tpch_runner`,
  `--threads 8`, `MORSEL_METRICS=1`, batch size `8,192`.
- **Verification:** All measured queries passed exact `pyarrow.Table.equals`
  lossless Arrow IPC parity against `stock_pushdown` before timing.
- **Sampling:** 1 warmup round + 5 measured interleaved rounds (alternating arm
  order each round). Table reports median exec wall time (`min–max` for
  `after_pd`) and median `morsel_process_work_ms` / `morsel_decode_work_ms`.

## Paired SF=5 Results (8 Threads, Warm Page Cache)

| Query | `before_pd` wall (ms) | `after_pd` wall ms (min–max) | `after_nopd` wall (ms) | `stock_pd` wall (ms) | Wall speedup | `before_pd` proc (ms) | `after_pd` proc (ms) | Proc speedup | `before_pd` dec (ms) | `after_pd` dec (ms) |
| ----- | --------------------: | ---------------------------: | ---------------------: | -------------------: | -----------: | --------------------: | -------------------: | -----------: | -------------------: | ------------------: |
| `q1`  |                 726.1 |      **593.6** (588.2–599.1) |                  596.6 |                760.6 |    **1.22×** |               2,025.8 |            **997.3** |    **2.03×** |              3,499.8 |             3,456.5 |
| `q6`  |                 334.5 |          335.4 (330.9–346.2) |                  338.7 |                505.1 |        1.00× |                 190.5 |            **173.3** |    **1.10×** |              2,309.5 |             2,355.2 |
| `q9`  |               1,460.9 |    1,455.5 (1,433.0–1,512.6) |                1,433.7 |              1,603.1 |        1.00× |               2,415.8 |          **2,379.2** |        1.02× |              5,436.1 |             5,312.1 |
| `q15` |                 731.6 |      **707.3** (694.7–724.5) |                  679.6 |                936.1 |    **1.03×** |                 590.5 |            **530.4** |    **1.11×** |              4,549.8 |             4,524.0 |
| `q13` |                 614.8 |          629.2 (560.2–640.5) |                  590.9 |                595.1 |        0.98× |               1,827.2 |              1,850.2 |        0.99× |              1,597.4 |             1,661.4 |
| `q21` |                 927.5 |          919.2 (893.9–931.9) |                1,837.8 |              1,572.3 |        1.01× |               1,238.6 |              1,244.8 |        1.00× |              3,588.0 |             3,618.9 |

## Notes

- **Precision, scale, and overflow safety:**
  - `Decimal128(p, s)` `AVG`-from-`SUM`/`COUNT` sharing is restricted to
    `p <= 25` (where DataFusion's `Avg` keeps its intermediate sum in
    `Decimal128(38, s)`); precisions `p >= 26` remain on `Decimal256`-backed
    accumulators so overflow semantics stay identical to stock DataFusion.
  - `Int32`/`Int64` `AVG` expressions accumulate their sum in `Float64` (via
    `CAST(x AS Float64)`) rather than sharing integer `SUM` accumulators,
    preserving bit-for-bit parity with stock DataFusion above `2^53` and on
    `i64` overflow while still sharing `COUNT(x)` / `COUNT(*)` and deduplicating
    identical `AVG` expressions.
  - Ungrouped `Float64` sums (and decomposed `AVG(Float64)`) always delegate to
    `arrow::compute::sum` so floating-point summation order matches stock
    `SumAccumulator`/`AvgAccumulator`.
  - `CountCol` linked to `CountStar` tracks `COUNT(col) = COUNT(*) - NULL_COUNT(col)`
    with lazy allocation of `null_counts`, doing zero per-row work on non-null
    Parquet batches (where Arrow schemas still declare `nullable=true`) while
    preserving exact all-NULL and mixed-NULL group semantics.
  - `DirectGroupedState` defers adaptive `direct_scatter` until after a full
    pre-aggregation flush (`PREAGG_FLUSH_GROUPS`) and only scatters batches with
    `>= 2,048` selected rows, avoiding tiny-slice fragmentation on selective
    medium-cardinality scans like Q15.
