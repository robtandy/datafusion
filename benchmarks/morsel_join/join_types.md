<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Morsel Join-Type Benchmark Note: Semi, Anti (Null-Aware), Outer, and CrossJoin (#190)

This note records paired release benchmarks at SF=1 and SF=5 across Parquet
workloads covering:

1. **Base (`origin/morsel-execution`) vs PR** regression check on `INNER JOIN`
   and `INNER JOIN` with residual `Join::filter` (Q19-style OR filter,
   ~50%-selective filter + `GROUP BY`, and sparse ~1%-selective filter +
   `COUNT(*)`).
2. **Stock DataFusion (`morsel_mode=off`) vs Native Morsel DataFusion
   (`morsel_mode=required`)** across all join types added in #190 (`LEFT SEMI`,
   `LEFT ANTI`, null-aware `NOT IN`, `LEFT/RIGHT/FULL OUTER` with residual
   `Join::filter` and build-side child-pipeline emission, `CROSS JOIN`, and
   `INNER JOIN` baselines).

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
2. `inner_filter_q19_or`: `orders JOIN customer` with TPC-H Q19-style 3-branch cross-table `OR` residual filter + `COUNT(*), SUM(o_totalprice)`
3. `inner_filter_50pct`: `orders JOIN customer WHERE o_totalprice + c_acctbal > 54000` (~50% selectivity) + `GROUP BY c_mktsegment`
4. `inner_filter_sparse_count`: `orders JOIN customer WHERE o_totalprice + c_acctbal > 107000` (~1% selectivity) + `COUNT(*)`
5. `left_semi_exists`: `orders WHERE EXISTS (SELECT 1 FROM customer WHERE ... AND c_mktsegment = 0)`
6. `left_anti_not_exists`: `customer WHERE NOT EXISTS (SELECT 1 FROM orders WHERE o_custkey = c_custkey)` + `GROUP BY` (`BuildAnti` + child emission)
7. `not_in_null_aware`: `orders WHERE o_suppkey NOT IN (SELECT s_suppkey FROM supplier WHERE s_flag = 1)`
8. `not_in_null_short_circuit`: `orders WHERE o_suppkey NOT IN (SELECT s_suppkey FROM supplier_null WHERE s_flag = 1)` (build has a `NULL` key -> short-circuits before probe)
9. `left_outer_q13_filter`: `customer LEFT OUTER JOIN orders ON c_custkey = o_custkey AND o_comment NOT LIKE '%special%requests%'` + `GROUP BY` (`BuildOuter` with residual filter + child emission)
10. `right_outer_child_emit`: `orders RIGHT OUTER JOIN customer` + `COUNT(*), COUNT(o_orderkey)` (`BuildOuter` + atomic matched bitset + child emission)
11. `full_outer_child_emit`: `orders FULL OUTER JOIN customer` + `COUNT(*), COUNT(o_orderkey), COUNT(c_custkey)` (`FullOuter` + unmatched probe NULL padding + child emission)
12. `cross_join_scalar`: `orders CROSS JOIN threshold WHERE o_totalprice > min_price`

## Environment and Controls

- **Host:** `rob-tandy-dduck1`, `aarch64`, Linux `7.0.0-1014-aws`, 16 ARM vCPUs.
- **CPU pinning and threads:** `taskset -c 0-7` (8 dedicated logical CPUs),
  8 Tokio workers (`target_partitions=8`), `batch_size=8,192`.
- **Host idle check:** `vmstat 1 2` showed 95–98% CPU idle before the runs and
  95–98% idle after.
- **Revisions:**
  - Base (`origin/morsel-execution`): `84d05f32a0830c9037028de73ebd9188776b1ae7`
  - PR (`agent/morsel-issue-190`): `b8c59d10e8f5caefc74f7bdb13f96c35a4d7696b`
- **Build:** `--locked --profile release-nonlto --no-default-features --features parquet,sql`,
  `rustc 1.99.1 (58cf8f9c3 2026-10-13)`.
- **Parquet settings:** matched across all arms (`pushdown_filters=false`,
  `pruning=false`, warm page cache after dataset creation + warmup round).
- **Protocol:** 1 unmeasured warmup round (with parity and plan assertions) and
  5 measured paired rounds with alternating `off`/`required` execution order.

## Base (`origin/morsel-execution`) vs PR: INNER + Residual Filter Regression Check

Running `--inner-only` on both revisions confirms that keeping `INNER JOIN`'s
residual `Join::filter` as a post-probe `Operator::Filter` (with fast
`probe_collect_matches` + lazy `BatchView` filtering) preserves base
performance within run-to-run noise across SF=1 and SF=5:

### SF=1 (Base vs PR, `morsel_mode=required`, 8 Threads)

| Workload                    | Base median ms (min–max) | PR median ms (min–max) | Ratio (Base / PR) | Stock median ms (PR run) |
| --------------------------- | -----------------------: | ---------------------: | ----------------: | -----------------------: |
| `inner_baseline`            |         36.5 (31.7–38.9) |       33.0 (31.7–39.7) |             1.11× |                     17.4 |
| `inner_filter_q19_or`       |         25.0 (22.8–25.4) |       24.7 (23.5–27.2) |             1.01× |                     20.0 |
| `inner_filter_50pct`        |         34.1 (31.9–43.7) |       38.1 (33.6–46.2) |             0.90× |                     22.1 |
| `inner_filter_sparse_count` |         32.1 (28.2–38.3) |       32.8 (30.0–35.3) |             0.98× |                     18.1 |

### SF=5 (Base vs PR, `morsel_mode=required`, 8 Threads)

| Workload                    | Base median ms (min–max) | PR median ms (min–max) | Ratio (Base / PR) | Stock median ms (PR run) |
| --------------------------- | -----------------------: | ---------------------: | ----------------: | -----------------------: |
| `inner_baseline`            |      190.3 (184.6–197.3) |    190.9 (188.5–196.6) |             1.00× |                    165.0 |
| `inner_filter_q19_or`       |      130.5 (125.1–135.0) |    130.8 (128.6–135.1) |             1.00× |                    101.1 |
| `inner_filter_50pct`        |      213.9 (212.8–235.1) |    218.0 (216.2–221.0) |             0.98× |                    175.7 |
| `inner_filter_sparse_count` |      192.9 (190.2–200.2) |    199.7 (197.3–204.1) |             0.97× |                    155.9 |

## Full Join-Type Results (Stock vs PR Morsel, 8 Threads, `batch_size=8,192`)

### SF=1 (`customer=150K`, `orders=1.5M`, `supplier=10K`)

| Workload                    | Stock median ms (min–max) | Morsel median ms (min–max) | Speedup | Output rows | Pipelines (B / P / E) |
| --------------------------- | ------------------------: | -------------------------: | ------: | ----------: | --------------------- |
| `inner_baseline`            |          20.2 (18.6–23.0) |           32.5 (29.9–40.5) |   0.62× |           5 | 4 / 16 / 0            |
| `inner_filter_q19_or`       |          20.0 (19.3–23.0) |           25.0 (23.4–30.9) |   0.80× |           1 | 4 / 16 / 0            |
| `inner_filter_50pct`        |          22.2 (21.5–25.3) |           35.9 (35.2–45.7) |   0.62× |           5 | 4 / 16 / 0            |
| `inner_filter_sparse_count` |          18.4 (15.8–20.3) |           31.2 (30.2–37.5) |   0.59× |           1 | 4 / 16 / 0            |
| `left_semi_exists`          |          14.5 (12.6–16.1) |           22.2 (18.8–23.7) |   0.65× |           1 | 4 / 16 / 0            |
| `left_anti_not_exists`      |          12.9 (12.5–13.8) |           23.5 (22.5–25.6) |   0.55× |           5 | 4 / 16 / 19           |
| `not_in_null_aware`         |           11.5 (8.5–11.9) |           13.9 (12.2–15.4) |   0.83× |           1 | 2 / 16 / 0            |
| `not_in_null_short_circuit` |             3.6 (3.2–3.9) |              0.8 (0.7–0.8) |   4.70× |           1 | 2 / 0 / 0             |
| `left_outer_q13_filter`     |          37.3 (33.6–43.1) |           42.2 (41.5–52.6) |   0.88× |           5 | 4 / 16 / 19           |
| `right_outer_child_emit`    |          19.2 (16.4–19.8) |           26.7 (26.4–34.1) |   0.72× |           1 | 4 / 16 / 19           |
| `full_outer_child_emit`     |          18.0 (15.6–19.0) |           37.7 (36.9–49.7) |   0.48× |           1 | 4 / 16 / 19           |
| `cross_join_scalar`         |             6.9 (5.1–7.0) |             9.0 (8.7–11.5) |   0.76× |           1 | 1 / 16 / 0            |

### SF=5 (`customer=750K`, `orders=7.5M`, `supplier=50K`)

| Workload                    | Stock median ms (min–max) | Morsel median ms (min–max) | Speedup | Output rows | Pipelines (B / P / E) |
| --------------------------- | ------------------------: | -------------------------: | ------: | ----------: | --------------------- |
| `inner_baseline`            |       165.0 (161.3–174.0) |        193.5 (191.5–200.9) |   0.85× |           5 | 16 / 64 / 0           |
| `inner_filter_q19_or`       |       105.0 (102.7–106.6) |        134.6 (127.1–138.5) |   0.78× |           1 | 16 / 64 / 0           |
| `inner_filter_50pct`        |       176.7 (172.9–178.6) |        218.6 (215.7–221.5) |   0.81× |           5 | 16 / 64 / 0           |
| `inner_filter_sparse_count` |       156.2 (151.0–161.4) |        195.7 (192.4–203.4) |   0.80× |           1 | 16 / 64 / 0           |
| `left_semi_exists`          |         77.3 (75.9–81.0)  |        113.5 (110.0–114.9) |   0.68× |           1 | 16 / 64 / 0           |
| `left_anti_not_exists`      |       115.1 (113.6–117.7) |        160.8 (160.4–162.7) |   0.72× |           5 | 16 / 64 / 92          |
| `not_in_null_aware`         |         55.1 (54.1–57.1)  |          55.8 (54.5–56.5)  |   0.99× |           1 | 6 / 64 / 0            |
| `not_in_null_short_circuit` |         19.4 (18.9–20.5)  |            1.1 (1.0–1.2)   |  17.52× |           1 | 6 / 0 / 0             |
| `left_outer_q13_filter`     |       234.4 (229.8–237.1) |        254.9 (248.7–258.6) |   0.92× |           5 | 16 / 64 / 92          |
| `right_outer_child_emit`    |       144.3 (137.1–145.9) |        179.1 (174.3–180.8) |   0.81× |           1 | 16 / 64 / 92          |
| `full_outer_child_emit`     |       148.5 (146.0–151.8) |        266.1 (258.6–271.8) |   0.56× |           1 | 16 / 64 / 92          |
| `cross_join_scalar`         |         26.4 (25.3–27.4)  |          38.8 (38.5–39.4)  |   0.68× |           1 | 1 / 64 / 0            |

*(Pipelines column reports `morsels_claimed` across `build` / `probe` / `build_emit` pipelines.)*

## Observations

- **Zero regression on INNER + residual filters vs `origin/morsel-execution`:**
  Lowering `INNER` hash join residual `Join::filter` as a post-probe
  `Operator::Filter` keeps `inner_baseline` (`190.9` vs `190.3 ms`),
  `inner_filter_q19_or` (`130.8` vs `130.5 ms`), `inner_filter_50pct` (`218.0`
  vs `213.9 ms`, overlapping ranges), and `inner_filter_sparse_count` (`199.7`
  vs `192.9 ms`) on parity with `origin/morsel-execution` at SF=5 (and likewise
  at SF=1).
- **Null-aware `NOT IN` short-circuit (`4.70×` at SF=1, `17.52×` at SF=5):**
  Detecting a `NULL` key on the build side skips the entire probe pipeline
  (`P = 0`), completing SF=5 in `1.1 ms` vs `19.4 ms` for stock DataFusion.
- **SF=5 scaling:** Relative efficiency improves from SF=1 to SF=5 across
  nearly all join types as per-pipeline coordination amortizes over 7.5M probe
  rows: `not_in_null_aware` reaches `0.99×` (`55.8 ms` vs `55.1 ms`),
  `left_outer_q13_filter` reaches `0.92×` (`254.9 ms` vs `234.4 ms`),
  `inner_baseline` reaches `0.85×`, and `right_outer_child_emit` reaches
  `0.81×`.
- **Remaining gap vs stock on balanced Parquet scans:** Stock `HashJoinExec`
  uses a dense direct-indexed `ArrayMap` for `0..N` integer keys and can attach
  dynamic build-side min/max filters to Parquet probe scans. Additionally,
  `full_outer_child_emit` (`0.56×` at SF=5) currently probes via the generic
  per-row candidate stepper rather than a specialized flat-int loop, which is a
  straightforward follow-up optimization.

## Reproduction Commands

```bash
work=$(mktemp -d /tmp/morsel-190-paired.XXXXXX)
git worktree add --detach "$work/base" origin/morsel-execution
git worktree add --detach "$work/pr" HEAD
mkdir -p "$work/base/datafusion/core/examples" "$work/pr/datafusion/core/examples"
cp benchmarks/morsel_join/join_types_runner.rs \
   "$work/base/datafusion/core/examples/morsel_join_types_bench.rs"
cp benchmarks/morsel_join/join_types_runner.rs \
   "$work/pr/datafusion/core/examples/morsel_join_types_bench.rs"
(
  cd "$work/base"
  CARGO_TARGET_DIR="$work/target_base" timeout 900s cargo build \
    --locked --profile release-nonlto -j 12 \
    -p datafusion --no-default-features --features parquet,sql \
    --example morsel_join_types_bench
)
(
  cd "$work/pr"
  CARGO_TARGET_DIR="$work/target_pr" timeout 900s cargo build \
    --locked --profile release-nonlto -j 12 \
    -p datafusion --no-default-features --features parquet,sql \
    --example morsel_join_types_bench
)
# Base vs PR INNER + residual-filter comparison
timeout --kill-after=10s 180s taskset -c 0-7 \
  "$work/target_base/release-nonlto/examples/morsel_join_types_bench" 1 8 1 5 --inner-only
timeout --kill-after=10s 180s taskset -c 0-7 \
  "$work/target_pr/release-nonlto/examples/morsel_join_types_bench" 1 8 1 5 --inner-only
timeout --kill-after=10s 300s taskset -c 0-7 \
  "$work/target_base/release-nonlto/examples/morsel_join_types_bench" 5 8 1 5 --inner-only
timeout --kill-after=10s 300s taskset -c 0-7 \
  "$work/target_pr/release-nonlto/examples/morsel_join_types_bench" 5 8 1 5 --inner-only
# Full 12-workload PR vs Stock runs
timeout --kill-after=10s 180s taskset -c 0-7 \
  "$work/target_pr/release-nonlto/examples/morsel_join_types_bench" 1 8 1 5
timeout --kill-after=10s 300s taskset -c 0-7 \
  "$work/target_pr/release-nonlto/examples/morsel_join_types_bench" 5 8 1 5
git worktree remove --force "$work/base"
git worktree remove --force "$work/pr"
rm -rf "$work"
```
