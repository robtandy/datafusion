<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Morsel Multi-Aggregate Benchmark Note: TPC-H SF1 Q6 & Q1 (Unordered) (#168)

This note records a paired **Stock DataFusion (`morsel_mode=off`) vs Native Morsel
DataFusion (`morsel_mode=required`) vs DuckDB** warm-cache benchmark on real
TPC-H SF1 Parquet files (`lineitem`: 6,001,215 rows, 16 Zstd files, 48 row groups
of 131,072 rows, 177,150,918 bytes on disk) for:

- **`q6`**: Standard TPC-H Q6 — filtered ungrouped
  `SUM(l_extendedprice * l_discount)` over `DECIMAL(15,2)` and `DATE32`.
- **`q1_unordered`**: TPC-H Q1 without `ORDER BY` (until `#159` lands) — filtered
  grouped multi-aggregate (`4× SUM`, `3× AVG`, `COUNT(*)`) over multi-column
  string keys `(l_returnflag, l_linestatus)` and `DECIMAL(15,2)` / `DATE32`.

## Environment and Controls

- **Host:** `rob-tandy-dduck1`, `aarch64`, Linux `7.0.0-1014-aws`, 16 ARM vCPUs,
  61 GiB RAM.
- **CPU pinning and threads:** `taskset -c 0-7` (8 dedicated logical CPUs),
  `--threads 8`, batch size `8,192`.
- **Host idle check:** `vmstat -y 1 3` immediately before the run showed
  92–99% CPU idle; immediately after showed 95–99% CPU idle.
- **Build & versions:**
  - DataFusion revision: `67da9ea9cf859356cd3a2796fffc38e766321bdc`
    (`release-nonlto`, `--locked --no-default-features --features parquet,sql`)
  - Rust `1.99.1` (`58cf8f9c3`), Python `3.12.10`, DuckDB `2.0.1`,
    PyArrow `26.0.1`, NumPy `2.5.5`.
- **Warm page cache verification:** All 98 measured trials started and ended with
  **100% of 4 KiB pages resident** (`43,256 / 43,256` pages for `lineitem`) via
  Linux `mincore`.
- **Parity & plan verification:** Before timing, `harness.py` verified exact
  result parity against DuckDB over lossless Arrow IPC streams (`q6`: 1 row;
  `q1_unordered`: 4 rows). Every `morsel_mode=required` trial executed natively
  under `MorselExec: workers=8, NativeParquet[lineitem...]` with
  `AggregateSink` (`aggregate_finalizations=1`, `parquet_morsels_claimed=48`)
  and scanned the exact same `bytes_scanned` as Stock DataFusion
  (`47,682,742` bytes for `q6`, `51,899,102` bytes for `q1_unordered`).
- **Protocol:** 1 unmeasured warmup round and 7 measured rounds per query/arm
  with alternating arm order. Values below are medians in milliseconds with
  min–max ranges in parentheses.

## Results (SF1 Warm Cache, 8 Threads)

### TPC-H Q6 (Ungrouped `SUM`, 1.90% Selectivity)

| Arm (`mode` / profile)                                                         | Median total ms (min–max) | Median exec ms (min–max) | Reg + plan ms | Morsels claimed |
| ------------------------------------------------------------------------------ | ------------------------: | -----------------------: | ------------: | --------------: |
| `stock_no_pushdown` (`off`, `pushdown_filters=false`, `pruning=false`)         |          79.8 (78.7–89.1) |         71.6 (70.5–80.7) |     4.5 + 3.7 |               0 |
| `morsel_no_pushdown` (`required`, `pushdown_filters=false`, `pruning=false`)   |          82.6 (80.9–84.7) |         75.8 (73.3–77.4) |     4.3 + 2.9 |              48 |
| `morsel_pruning` (`required`, `pushdown_filters=false`, `pruning=true`)        |          85.1 (83.6–88.3) |         76.9 (75.4–79.7) |     4.4 + 4.0 |              48 |
| `stock_default` (`off`, default: `pushdown_filters=false`, `pruning=true`)     |         97.7 (96.0–100.4) |         88.4 (86.8–91.3) |     4.3 + 4.8 |               0 |
| `stock_pushdown` (`off`, `pushdown_filters=true`, `reorder_filters=true`)      |       132.0 (130.2–133.7) |      122.8 (121.4–124.4) |     4.5 + 4.8 |               0 |
| `morsel_pushdown` (`required`, `pushdown_filters=true`, `reorder_filters=true`)|       114.0 (111.7–118.3) |      105.7 (103.2–109.9) |     4.4 + 4.0 |              48 |
| `duckdb_default` (DuckDB defaults)                                             |          56.0 (52.7–66.4) |                        — |             — |               — |

### TPC-H Q1 Unordered (Grouped 8-Aggregate on `(l_returnflag, l_linestatus)`, 98.6% Selectivity)

| Arm (`mode` / profile)                                                         | Median total ms (min–max) | Median exec ms (min–max) | Reg + plan ms | Morsels claimed |
| ------------------------------------------------------------------------------ | ------------------------: | -----------------------: | ------------: | --------------: |
| `stock_no_pushdown` (`off`, `pushdown_filters=false`, `pruning=false`)         |       169.2 (164.6–177.0) |      159.6 (155.0–167.4) |     4.4 + 5.2 |               0 |
| `morsel_no_pushdown` (`required`, `pushdown_filters=false`, `pruning=false`)   |       167.5 (164.0–172.1) |      159.5 (155.2–163.7) |     4.4 + 3.9 |              48 |
| `morsel_pruning` (`required`, `pushdown_filters=false`, `pruning=true`)        |       169.8 (164.5–173.5) |      161.1 (155.2–164.1) |     4.3 + 4.7 |              48 |
| `stock_default` (`off`, default: `pushdown_filters=false`, `pruning=true`)     |       207.1 (199.9–211.4) |      197.0 (189.6–201.1) |     4.3 + 6.1 |               0 |
| `stock_pushdown` (`off`, `pushdown_filters=true`, `reorder_filters=true`)      |       193.8 (193.1–196.2) |      183.8 (182.8–186.0) |     4.3 + 5.9 |               0 |
| `morsel_pushdown` (`required`, `pushdown_filters=true`, `reorder_filters=true`)|       159.2 (156.3–165.0) |      150.6 (147.2–156.5) |     4.2 + 4.7 |              48 |
| `duckdb_default` (DuckDB defaults)                                             |       108.8 (101.4–112.5) |                        — |             — |               — |

## Key Observations

- **TPC-H Q1 (unordered):** Native morsel execution beats Stock DataFusion
  across every scan configuration. `morsel_pushdown` (159.2 ms total / 150.6 ms
  exec) is **1.30× faster than `stock_default`** (207.1 ms) and **1.22× faster
  than `stock_pushdown`** (193.8 ms), closing the gap against DuckDB (108.8 ms)
  from Stock's 1.90× to 1.46×. Under `pushdown_filters=false`,
  `morsel_no_pushdown` (167.5 ms) edges out `stock_no_pushdown` (169.2 ms) and
  is 1.24× faster than `stock_default`. Fused worker-local group accumulation,
  filter-input column pruning (`l_shipdate` dropped before filtering), and
  zero-copy bypass on all-true filter masks avoid intermediate repartitioning
  and batch copying.
- **TPC-H Q6:** `morsel_no_pushdown` (82.6 ms) and `morsel_pruning` (85.1 ms)
  are **1.15–1.18× faster than `stock_default`** (97.7 ms), and
  `morsel_pushdown` (114.0 ms) is **1.16× faster than `stock_pushdown`**
  (132.0 ms). Under matched `no_pushdown` settings, `morsel_no_pushdown` (82.6 ms)
  is within 3.5% of `stock_no_pushdown` (79.8 ms) and 1.48× of DuckDB (56.0 ms).

## Reproduction Commands

Using the harness in this directory (see `benchmarks/morsel_tpch/README.md` for
building `morsel_tpch_runner`, creating the Python venv, and generating
`$data_dir`):

```bash
# Unit + SF0.01 end-to-end parity & plan smoke tests
MORSEL_TPCH_RUNNER="$runner" MORSEL_BENCH_REVISION="$revision" \
  timeout 180s "$py" -B -m unittest discover -s benchmarks/morsel_tpch -v

# Paired SF1 warm-cache benchmark for q6 and q1_unordered
timeout --kill-after=10s 300s taskset -c 0-7 "$py" -B benchmarks/morsel_tpch/harness.py bench \
  "$data_dir" --runner "$runner" --revision "$revision" \
  --threads 8 --timeout 120 --cache warm --warmups 1 --repeats 7 \
  --queries q6 q1_unordered \
  --morsel-arms stock_no_pushdown morsel_no_pushdown morsel_pruning \
    stock_default stock_pushdown morsel_pushdown duckdb_default \
  > "$work/sf1_agg_warm.jsonl"
"$py" -B benchmarks/morsel_tpch/harness.py summarize "$work/sf1_agg_warm.jsonl"
```
