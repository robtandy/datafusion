<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Morsel Join-Type Benchmark Note: Semi, Anti (Null-Aware), Outer, and CrossJoin (#190)

This note records a paired **Stock DataFusion (`morsel_mode=off`) vs Native
Morsel DataFusion (`morsel_mode=required`)** release benchmark across SF=1 and
SF=5 Parquet workloads covering the join shapes added in #190 (`LEFT SEMI`,
`LEFT ANTI`, null-aware `NOT IN`, `LEFT/RIGHT/FULL OUTER` with residual
`Join::filter` and build-side child-pipeline emission, `CROSS JOIN`, and an
`INNER JOIN` baseline).

## Datasets and Workloads

Generated deterministically by `benchmarks/morsel_join/join_types_runner.rs`
as Zstd-compressed Parquet tables matching TPC-H table ratios:

- **SF=1:** `customer` (`150,000` rows, 4 files), `orders` (`1,500,000` rows,
  16 files, `125,000` rows/RG), `supplier` / `supplier_null` (`10,000` rows, 2
  files; `supplier_null` has 1 `NULL` key matching `s_flag = 1`), `threshold`
  (`1` row).
- **SF=5:** `customer` (`750,000` rows, 4 files), `orders` (`7,500,000` rows,
  16 files, `125,000` rows/RG), `supplier` / `supplier_null` (`50,000` rows, 2
  files), `threshold` (`1` row).
- ~20% of customers have no orders (exercising unmatched-build emission on
  `LEFT OUTER`, `RIGHT OUTER`, `FULL OUTER`, and `LEFT ANTI`), ~0.5% of orders
  have unmatched customer keys (exercising unmatched-probe NULL padding on
  `FULL OUTER`), and ~2% of orders carry `special ... requests` comments
  (exercising Q13's `NOT LIKE` residual `Join::filter`).

Workloads (all verified for exact row-count and column-checksum parity and
verified `MorselExec` under `morsel_mode=required` vs stock execution under
`morsel_mode=off`):

1. `inner_baseline`: `orders JOIN customer` + `GROUP BY c_mktsegment`
2. `left_semi_exists`: `orders WHERE EXISTS (SELECT 1 FROM customer WHERE ... AND c_mktsegment = 0)`
3. `left_anti_not_exists`: `customer WHERE NOT EXISTS (SELECT 1 FROM orders WHERE o_custkey = c_custkey)` + `GROUP BY` (smaller `customer` on build -> `BuildAnti` + child emission)
4. `not_in_null_aware`: `orders WHERE o_suppkey NOT IN (SELECT s_suppkey FROM supplier WHERE s_flag = 1)`
5. `not_in_null_short_circuit`: `orders WHERE o_suppkey NOT IN (SELECT s_suppkey FROM supplier_null WHERE s_flag = 1)` (build has a `NULL` key -> short-circuits before probe)
6. `left_outer_q13_filter`: `customer LEFT OUTER JOIN orders ON c_custkey = o_custkey AND o_comment NOT LIKE '%special%requests%'` + `GROUP BY` (`BuildOuter` with residual filter + child emission)
7. `right_outer_child_emit`: `orders RIGHT OUTER JOIN customer` + `COUNT(*), COUNT(o_orderkey)` (`BuildOuter` + atomic matched bitset + child emission)
8. `full_outer_child_emit`: `orders FULL OUTER JOIN customer` + `COUNT(*), COUNT(o_orderkey), COUNT(c_custkey)` (`FullOuter` + unmatched probe NULL padding + child emission)
9. `cross_join_scalar`: `orders CROSS JOIN threshold WHERE o_totalprice > min_price`

## Environment and Controls

- **Host:** `rob-tandy-dduck1`, `aarch64`, Linux `7.0.0-1014-aws`, 16 ARM vCPUs.
- **CPU pinning and threads:** `taskset -c 0-7` (8 dedicated logical CPUs),
  8 Tokio workers (`target_partitions=8`), `batch_size=8,192`.
- **Host idle check:** `vmstat 1 2` showed 95–97% CPU idle before the runs and
  95–98% idle after.
- **Build:** revision `e8caa9e8ff6f879502deac9049fdcb3d34becc2c`,
  `--locked --profile release-nonlto --no-default-features --features parquet,sql`,
  `rustc 1.99.1 (58cf8f9c3 2026-10-13)`.
- **Parquet settings:** matched across both arms (`pushdown_filters=false`,
  `pruning=false`, warm page cache after dataset creation + warmup round).
- **Protocol:** 1 unmeasured warmup round (with parity and plan assertions) and
  5 measured paired rounds with alternating `off`/`required` execution order.

## Results (8 Threads, `batch_size=8,192`)

### SF=1 (`customer=150K`, `orders=1.5M`, `supplier=10K`)

| Workload                    | Stock median ms (min–max) | Morsel median ms (min–max) | Speedup | Output rows | Pipelines (B / P / E) |
| --------------------------- | ------------------------: | -------------------------: | ------: | ----------: | --------------------- |
| `inner_baseline`            |          20.9 (19.1–23.3) |           32.0 (30.9–32.8) |   0.66× |           5 | 4 / 16 / 0            |
| `left_semi_exists`          |          15.7 (14.5–16.5) |           20.2 (18.6–24.2) |   0.78× |           1 | 4 / 16 / 0            |
| `left_anti_not_exists`      |          12.6 (12.3–13.2) |           23.9 (21.3–25.8) |   0.53× |           5 | 4 / 16 / 19           |
| `not_in_null_aware`         |           11.1 (8.6–11.6) |           12.3 (11.3–12.4) |   0.90× |           1 | 2 / 16 / 0            |
| `not_in_null_short_circuit` |             3.7 (3.6–3.8) |              0.7 (0.7–0.8) |   5.14× |           1 | 2 / 0 / 0             |
| `left_outer_q13_filter`     |          32.4 (29.8–41.8) |           41.8 (40.0–50.9) |   0.77× |           5 | 4 / 16 / 19           |
| `right_outer_child_emit`    |          15.7 (14.4–17.4) |           25.8 (24.9–28.1) |   0.61× |           1 | 4 / 16 / 19           |
| `full_outer_child_emit`     |          18.2 (15.6–20.7) |           36.6 (34.4–44.6) |   0.50× |           1 | 4 / 16 / 19           |
| `cross_join_scalar`         |             6.6 (5.0–6.7) |             9.4 (8.4–11.3) |   0.71× |           1 | 1 / 16 / 0            |

### SF=5 (`customer=750K`, `orders=7.5M`, `supplier=50K`)

| Workload                    | Stock median ms (min–max) | Morsel median ms (min–max) | Speedup | Output rows | Pipelines (B / P / E) |
| --------------------------- | ------------------------: | -------------------------: | ------: | ----------: | --------------------- |
| `inner_baseline`            |       164.0 (159.2–171.2) |        192.9 (186.1–198.1) |   0.85× |           5 | 16 / 64 / 0           |
| `left_semi_exists`          |         77.7 (73.8–78.4)  |        109.7 (106.7–114.4) |   0.71× |           1 | 16 / 64 / 0           |
| `left_anti_not_exists`      |       113.6 (112.1–116.1) |        157.5 (154.6–164.6) |   0.72× |           5 | 16 / 64 / 92          |
| `not_in_null_aware`         |         54.4 (54.1–56.8)  |          55.2 (54.1–57.8)  |   0.99× |           1 | 6 / 64 / 0            |
| `not_in_null_short_circuit` |         19.5 (18.8–20.1)  |            1.1 (1.0–1.2)   |  17.59× |           1 | 6 / 0 / 0             |
| `left_outer_q13_filter`     |       242.5 (228.9–252.7) |        257.2 (250.7–262.6) |   0.94× |           5 | 16 / 64 / 92          |
| `right_outer_child_emit`    |       142.3 (138.9–146.5) |        178.8 (171.6–182.9) |   0.80× |           1 | 16 / 64 / 92          |
| `full_outer_child_emit`     |       150.5 (141.7–151.4) |        261.3 (256.0–265.1) |   0.58× |           1 | 16 / 64 / 92          |
| `cross_join_scalar`         |         27.0 (25.0–27.5)  |          37.6 (37.0–39.1)  |   0.72× |           1 | 1 / 64 / 0            |

*(Pipelines column reports `morsels_claimed` across `build` / `probe` / `build_emit` pipelines.)*

## Observations

- **Null-aware `NOT IN` short-circuit (`5.14×` at SF=1, `17.59×` at SF=5):**
  Detecting a `NULL` key on the build side lets morsel execution skip the
  entire probe pipeline (`P = 0`) cleanly, completing SF=5 in `1.1 ms` vs
  `19.5 ms` for stock DataFusion.
- **SF=5 scaling:** Relative efficiency improves from SF=1 to SF=5 across
  nearly all join types as per-pipeline coordination amortizes over 7.5M probe
  rows: `not_in_null_aware` reaches `0.99×` (`55.2 ms` vs `54.4 ms`),
  `left_outer_q13_filter` reaches `0.94×` (`257.2 ms` vs `242.5 ms`),
  `inner_baseline` reaches `0.85×`, and `right_outer_child_emit` reaches
  `0.80×`.
- **Remaining gap vs stock on balanced Parquet scans:** Stock `HashJoinExec`
  uses a dense direct-indexed `ArrayMap` for `0..N` integer keys and can attach
  dynamic build-side min/max filters to Parquet probe scans. Additionally,
  `full_outer_child_emit` (`0.58×` at SF=5) currently probes via the generic
  per-row candidate stepper rather than a specialized flat-int loop, which is a
  straightforward follow-up optimization.

## Reproduction Commands

```bash
work=$(mktemp -d /tmp/morsel-190-bench.XXXXXX)
git worktree add --detach "$work/src" HEAD
mkdir -p "$work/src/datafusion/core/examples"
cp benchmarks/morsel_join/join_types_runner.rs \
   "$work/src/datafusion/core/examples/morsel_join_types_bench.rs"
(
  cd "$work/src"
  CARGO_TARGET_DIR="$work/target" timeout 900s cargo build \
    --locked --profile release-nonlto -j 12 \
    -p datafusion --no-default-features --features parquet,sql \
    --example morsel_join_types_bench
)
timeout --kill-after=10s 180s taskset -c 0-7 \
  "$work/target/release-nonlto/examples/morsel_join_types_bench" 1 8 1 5
timeout --kill-after=10s 300s taskset -c 0-7 \
  "$work/target/release-nonlto/examples/morsel_join_types_bench" 5 8 1 5
git worktree remove --force "$work/src" && rm -rf "$work"
```
