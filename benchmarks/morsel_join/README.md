<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Morsel Two-Pipeline HashJoin Benchmark Note (#158)

This note records a paired **Stock DataFusion (`morsel_mode=off`) vs Native
Morsel DataFusion (`morsel_mode=required`)** release benchmark for the
two-pipeline native `INNER JOIN` (`HashJoinBuildSink` + lock-free
`HashJoinProbe`) over both `MemTable` and Parquet inputs, covering balanced and
skewed probe distributions.

## Datasets and Workloads

Generated deterministically by `benchmarks/morsel_join/join_runner.rs`:

- **Build table (`dim`):** `100,000` rows (`k: Int64` unique `0..100,000`,
  `grp: Int32` in `0..64`, `weight: Int64`, `tag: Utf8`), stored across 4
  Zstd Parquet files (`25,000` rows/RG) and identically as a 4-partition
  `MemTable`.
- **Balanced probe (`fact_balanced`):** `4,000,000` rows (`id: Int64`,
  `k: Int64`, `val: Int64`) across 16 equal files/partitions (`250,000` rows
  each, 2 Parquet row groups of `125,000` rows per file), uniform join keys
  with ~5% unmatched keys and ~1% `NULL` keys.
- **Skewed probe (`fact_skewed`):** `4,000,000` rows across 16 files/partitions
  combining **partition/file skew** (1 hot file/partition with `1,750,000` rows
  across 14 row groups + 15 small files/partitions of `150,000` rows each) and
  **key skew** (50% of probe rows hit 10 hot build keys, 50% uniform across
  `0..100,000`, plus ~5% unmatched and ~1% `NULL` keys).

Workloads (all verified for exact row-count and column-checksum parity and
verified `MorselExec` with `HashJoinBuildSink` + `HashJoinProbe` under
`morsel_mode=required`, and stock `HashJoinExec` under `morsel_mode=off`):

1. `mem_balanced_proj`: `SELECT f.id, d.weight + f.val AS total FROM fact_bal_mem f JOIN dim_mem d ON f.k = d.k`
2. `mem_skewed_proj`: `SELECT f.id, d.weight + f.val AS total FROM fact_skew_mem f JOIN dim_mem d ON f.k = d.k`
3. `mem_skewed_count`: `SELECT COUNT(*) AS n FROM fact_skew_mem f JOIN dim_mem d ON f.k = d.k`
4. `parquet_balanced_agg`: `SELECT d.grp, COUNT(*) AS n, SUM(f.val + d.weight) AS total FROM fact_bal_pq f JOIN dim_pq d ON f.k = d.k GROUP BY d.grp`
5. `parquet_skewed_agg`: `SELECT d.grp, COUNT(*) AS n, SUM(f.val + d.weight) AS total FROM fact_skew_pq f JOIN dim_pq d ON f.k = d.k GROUP BY d.grp`
6. `parquet_balanced_count`: `SELECT COUNT(*) AS n FROM fact_bal_pq f JOIN dim_pq d ON f.k = d.k WHERE f.val > 0`
7. `parquet_skewed_proj`: `SELECT f.id, d.weight + f.val AS total FROM fact_skew_pq f JOIN dim_pq d ON f.k = d.k WHERE f.val % 10 = 0`

## Environment and Controls

- **Host:** `rob-tandy-dduck1`, `aarch64`, Linux `7.0.0-1014-aws`, 16 ARM vCPUs,
  61 GiB RAM.
- **CPU pinning and threads:** `taskset -c 0-7` (8 dedicated logical CPUs),
  8 Tokio workers (`target_partitions=8`), `batch_size=8,192`.
- **Host idle check:** `vmstat 1 2` showed 95–96% CPU idle before the run and
  94–95% idle after.
- **Build:** revision `fb8f6681431d0944d8d256fdc198ca0ba1acdb1d`,
  `--locked --profile release-nonlto --no-default-features --features parquet,sql`,
  `rustc 1.99.1 (58cf8f9c3 2026-10-13)`.
- **Parquet settings:** matched across both arms (`pushdown_filters=false`,
  `pruning=false`, warm page cache after dataset creation + warmup round).
- **Protocol:** 1 unmeasured warmup round (with parity and plan assertions) and
  7 measured paired rounds with alternating `off`/`required` execution order.

## Results (8 Threads, `batch_size=8,192`)

| Workload                 | Stock median ms (min–max) | Morsel median ms (min–max) | Speedup | Output rows | Build / Probe morsels |
| ------------------------ | ------------------------: | -------------------------: | ------: | ----------: | --------------------- |
| `mem_balanced_proj`      |          13.6 (12.6–17.9) |           31.7 (30.4–33.0) |   0.43× |   3,760,256 | 4 / 124               |
| `mem_skewed_proj`        |          49.2 (44.9–51.4) |           24.6 (24.1–28.2) |   2.00× |   3,760,256 | 4 / 125               |
| `mem_skewed_count`       |          15.9 (14.9–16.6) |           11.6 (11.3–14.3) |   1.37× |           1 | 4 / 125               |
| `parquet_balanced_agg`   |          38.0 (36.0–42.9) |           52.5 (48.1–61.7) |   0.72× |          64 | 5 / 32                |
| `parquet_skewed_agg`     |          38.3 (35.4–43.7) |           49.7 (45.3–55.9) |   0.77× |          37 | 5 / 44                |
| `parquet_balanced_count` |          25.8 (23.9–27.6) |           35.2 (30.7–35.8) |   0.73× |           1 | 5 / 32                |
| `parquet_skewed_proj`    |          34.9 (33.5–36.0) |           37.1 (35.1–39.1) |   0.94× |     375,694 | 5 / 44                |

## Observations

- **Skewed in-memory probe workloads:** Native morsel execution is **2.00×
  faster** on `mem_skewed_proj` (`24.6 ms` vs `49.2 ms`) and **1.37× faster**
  on `mem_skewed_count` (`11.6 ms` vs `15.9 ms`). Dynamic morsel claiming
  distributes the 1.75M-row hot partition evenly across all 8 workers (125
  probe morsels), whereas stock DataFusion's static per-partition streams stall
  on the straggler partition.
- **Balanced / Parquet workloads:** On `parquet_skewed_proj`, native morsel
  (`37.1 ms`) tracks stock DataFusion (`34.9 ms`) within 6%. On balanced
  workloads (`mem_balanced_proj`, `parquet_balanced_agg`,
  `parquet_balanced_count`), stock `HashJoinExec` currently leads (`1.3×–2.3×`)
  due to three well-understood gaps that are natural follow-ups:
  1. Dense integer keys (`0..99,999`) trigger stock `HashJoinExec`'s O(1)
     direct-indexed `ArrayMap` (zero hashing), whereas morsel uses a general
     flat open-addressed hash table.
  2. `MorselExec` funnels all 3.76M result rows through a single output stream
     (`Partitioning::UnknownPartitioning(1)`), adding single-channel contention
     on wide unaggregated outputs (`mem_balanced_proj`).
  3. On Parquet probe scans, stock `HashJoinExec` can attach dynamic
     min-max/bloom pushdown filters from the build side.

## Reproduction Commands

The runner compiles as an isolated example inside a detached temporary worktree
without modifying any repository manifests:

```bash
work=$(mktemp -d /tmp/morsel-158-bench.XXXXXX)
git worktree add --detach "$work/src" HEAD
mkdir -p "$work/src/datafusion/core/examples"
cp benchmarks/morsel_join/join_runner.rs \
   "$work/src/datafusion/core/examples/morsel_join_bench.rs"
(
  cd "$work/src"
  CARGO_TARGET_DIR="$work/target" timeout 900s cargo build \
    --locked --profile release-nonlto -j 12 \
    -p datafusion --no-default-features --features parquet,sql \
    --example morsel_join_bench
)
timeout --kill-after=10s 180s taskset -c 0-7 \
  "$work/target/release-nonlto/examples/morsel_join_bench" 8 1 7
git worktree remove --force "$work/src" && rm -rf "$work"
```
