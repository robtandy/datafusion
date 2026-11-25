<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Morsel Native Parallel TopN and Sort Benchmark Note (#159)

This note records paired **Stock DataFusion (`morsel_mode=off`) vs Native
Morsel DataFusion (`morsel_mode=required`)** release benchmarks for the native
pipeline-breaking `TopNSink` (`ORDER BY ... LIMIT K [OFFSET O]`) and parallel
`SortSink` (`ORDER BY ...`) across uniform and skewed Parquet scans (both
integer and out-of-line `Utf8View` sort keys) and 400K×4M Parquet `INNER JOIN`
inputs at `batch_size = 8,192`, `1,024`, and `256`.

## Datasets and Workloads

Generated deterministically by `benchmarks/morsel_sort/sort_runner.rs`:

- **Direct scan schema:** `id: Int64` (unique `0..4,000,000`), `score: Int64`
  (nullable, ~1% `NULL`), `metric: Float64` (nullable, ~1% `NULL`), `grp: Int32`
  (`0..64`), `name: Utf8` (read as `Utf8View` by DataFusion's default Parquet
  reader; >12-byte strings ~28 bytes each so data buffers are out-of-line).
- **Uniform Parquet (`pq_uniform`):** `4,000,000` rows across 16 equal Zstd
  Parquet files (`250,000` rows each, 2 row groups of `125,000` rows per file).
- **Skewed Parquet (`pq_skewed`):** `4,000,000` rows across 16 Zstd Parquet
  files combining **file/partition skew** (1 hot file of `1,750,000` rows
  across 14 row groups + 15 small files of `150,000` rows each) and **key
  skew** (50% of rows collide on 10 hot `score` values).
- **Join Parquet (`dim_pq` × `fact_pq`):**
  - `dim_pq`: `400,000` rows (`k: Int64`, `name: Utf8` read as `Utf8View` with
    ~36-byte out-of-line strings, `weight: Int64`) across 4 Zstd Parquet files.
  - `fact_pq`: `4,000,000` rows (`id: Int64`, `k: Int64`) across 16 Zstd
    Parquet files probing `dim_pq`.

Workloads (all verified for exact ordered row-count and position-sensitive
ordered key-digest parity on warmup, row-count parity and `pool.reserved() == 0`
on every measured round, `MorselExec` with `TopNSink` / `SortSink` under
`morsel_mode=required`, and stock `SortExec` under `morsel_mode=off`):

1. `topn_uniform_limit100`: `SELECT id, score, name FROM pq_uniform ORDER BY score DESC NULLS LAST, id ASC LIMIT 100`
2. `topn_skewed_limit100`: `SELECT id, score, name FROM pq_skewed ORDER BY score DESC NULLS LAST, id ASC LIMIT 100`
3. `topn_utf8view_limit100`: `SELECT id, name, score FROM pq_uniform ORDER BY name ASC NULLS LAST, id ASC LIMIT 100`
4. `topn_join_utf8view_100`: `SELECT d.name, f.id, d.weight FROM fact_pq f JOIN dim_pq d ON f.k = d.k ORDER BY d.name, f.id LIMIT 100`
5. `sort_uniform_single_key`: `SELECT id, score FROM pq_uniform ORDER BY score ASC NULLS LAST`
6. `sort_skewed_single_key`: `SELECT id, score FROM pq_skewed ORDER BY score ASC NULLS LAST`
7. `sort_utf8view_key`: `SELECT id, name FROM pq_uniform ORDER BY name ASC NULLS LAST, id ASC`
8. `sort_join_id`: `SELECT d.name, f.id FROM fact_pq f JOIN dim_pq d ON f.k = d.k ORDER BY f.id`
9. `sort_join_utf8view`: `SELECT d.name, f.id FROM fact_pq f JOIN dim_pq d ON f.k = d.k ORDER BY d.name, f.id`

## Environment and Controls

- **Host:** `rob-tandy-dduck1`, `aarch64`, Linux `7.0.0-1014-aws`, 16 ARM vCPUs,
  61 GiB RAM.
- **CPU pinning and threads:** `taskset -c 0-7` (8 dedicated logical CPUs),
  8 Tokio workers (`target_partitions=8`).
- **Host idle check:** `vmstat 1 2` showed 95–97% CPU idle before and 95–100%
  idle after the `batch_size=8,192` run.
- **Build:** revision `f2924f58d` (`fix(morsel): deduplicate shared buffers and
  compact view arrays in TopN and Sort (#159)`),
  `--locked --profile release-nonlto --no-default-features --features parquet,sql`,
  `rustc 1.99.1 (58cf8f9c3 2026-10-13)`.
- **Parquet settings:** matched across both arms (`pushdown_filters=false`,
  `pruning=false`, default `schema_force_view_types=true`, warm page cache).
- **Protocol:** 1 unmeasured warmup round (with ordered parity and plan
  assertions) and paired alternating `off`/`required` rounds (5 rounds at
  `batch_size=8,192`; 3 rounds at `batch_size=1,024` and `256`), creating a
  fresh physical plan per round. Paired timing at `batch_size=8,192` used a
  2 GiB `GreedyMemoryPool`; `batch_size=1,024` and `256` used
  `UnboundedMemoryPool` for paired timing (stock `ExternalSorterMerge`
  exhausted a 4 GiB `GreedyMemoryPool` on `sort_utf8view_key` at `1,024`),
  with a separate 1 GiB `GreedyMemoryPool` verification confirming all three
  Parquet join queries (`topn_join_utf8view_100`, `sort_join_id`,
  `sort_join_utf8view`) succeed cleanly under `morsel_mode=required` at both
  `batch_size=8,192` and `batch_size=1,024` with `pool.reserved() == 0`.

## Results (8 Threads, `batch_size=8,192`, 2 GiB Pool)

| Workload                  | Stock median ms (min–max) |  Morsel median ms (min–max) | Speedup | Output rows | Morsels |
| ------------------------- | ------------------------: | --------------------------: | ------: | ----------: | ------: |
| `topn_uniform_limit100`   |          76.0 (73.8–76.8) |            62.9 (59.8–65.8) |   1.21× |         100 |      32 |
| `topn_skewed_limit100`    |          70.1 (69.3–77.3) |            60.0 (57.6–64.3) |   1.17× |         100 |      44 |
| `topn_utf8view_limit100`  |          87.6 (86.6–89.0) |            69.5 (63.4–72.2) |   1.26× |         100 |      32 |
| `topn_join_utf8view_100`  |       132.0 (131.1–136.6) |         138.0 (132.2–154.0) |   0.96× |         100 |      36 |
| `sort_uniform_single_key` |       301.2 (297.4–307.3) |         395.4 (386.1–400.1) |   0.76× |   4,000,000 |      32 |
| `sort_skewed_single_key`  |       270.7 (268.9–272.2) |         325.6 (302.3–337.3) |   0.83× |   4,000,000 |      44 |
| `sort_utf8view_key`       |       930.0 (919.5–933.8) |     1,606.2 (1471.7–1751.9) |   0.58× |   4,000,000 |      32 |
| `sort_join_id`            |       379.7 (375.5–393.1) |         557.3 (474.8–593.6) |   0.68× |   4,000,000 |      36 |
| `sort_join_utf8view`      |       879.6 (822.5–915.0) |     1,338.4 (1258.3–1493.7) |   0.66× |   4,000,000 |      36 |

## Small-Batch Scaling (`batch_size=1,024` and `batch_size=256`, 8 Threads)

| Workload                 | `bs=1,024` Stock ms | `bs=1,024` Morsel ms | Speedup | `bs=256` Stock ms | `bs=256` Morsel ms | Speedup |
| ------------------------ | ------------------: | -------------------: | ------: | ----------------: | -----------------: | ------: |
| `topn_uniform_limit100`  |                87.3 |                 69.3 |   1.26× |             122.7 |               72.3 |   1.70× |
| `topn_skewed_limit100`   |                84.0 |                 64.6 |   1.30× |             117.3 |               74.1 |   1.58× |
| `topn_utf8view_limit100` |               105.1 |                 68.8 |   1.53× |             139.8 |               76.7 |   1.82× |
| `topn_join_utf8view_100` |               164.5 |                154.9 |   1.06× |             286.1 |              200.7 |   1.43× |
| `sort_utf8view_key`      |             1,177.9 |              1,667.3 |   0.71× |           1,878.1 |            2,083.1 |   0.90× |
| `sort_join_id`           |               690.0 |                733.5 |   0.94× |           1,608.4 |            1,914.8 |   0.84× |
| `sort_join_utf8view`     |               847.4 |              1,477.3 |   0.57× |           2,416.1 |            2,409.1 |   1.00× |

## Observations

- **`TopNSink` over integer, `Utf8View`, and Parquet HashJoin inputs:**
  - Direct scans are **1.17×–1.26× faster** than stock at `batch_size=8,192`,
    **1.26×–1.53× faster** at `batch_size=1,024`, and **1.58×–1.82× faster** at
    `batch_size=256`, driven by dynamic morsel claiming and the first-column
    primitive + `Utf8`/`Utf8View` threshold fast-reject.
  - On the 400K×4M Parquet join (`topn_join_utf8view_100`), compacting winner
    `Utf8View` columns via `gc()` and deduplicating with
    `RecordBatchMemoryTracker` keeps `MorselTopN` bounded to `O(K+O)` bytes
    inside the 1 GiB pool, matching stock at `batch_size=8,192` (`138.0 ms` vs
    `132.0 ms`) and beating stock at `1,024` (`1.06×`) and `256` (`1.43×`).
- **Full 4M-row `SortSink` over `Utf8View` and Parquet HashJoin inputs:**
  - Tracking shared buffers via `RecordBatchMemoryCounter` /
    `RecordBatchMemoryTracker` and single-pass deduplicating `Utf8View`
    concatenation keeps each worker to ~1 sorted run (8 runs total across 8
    workers) even at `batch_size=256`, instead of exploding into hundreds of
    runs and millions of buffer references.
  - All three Parquet join queries (`topn_join_utf8view_100`, `sort_join_id`,
    `sort_join_utf8view`) succeed cleanly within a 1 GiB pool at
    `batch_size=8,192` and `1,024`, and `sort_join_utf8view` scales cleanly down
    to `batch_size=256` (`2,409.1 ms` vs stock `2,416.1 ms`, `1.00×`).
  - Remaining full-sort gap at `batch_size=8,192` (`0.58×–0.83×`) comes from
    `MorselExec` funneling all 4M merged rows through a single bounded
    `mpsc::channel(2)` and compacting chunked `Utf8View` slices on merge yield.

## Reproduction Commands

The runner compiles as an isolated example inside a detached temporary worktree
without modifying any repository manifests:

```bash
work=$(mktemp -d /tmp/morsel-159-bench.XXXXXX)
git worktree add --detach "$work/src" HEAD
mkdir -p "$work/src/datafusion/core/examples"
cp benchmarks/morsel_sort/sort_runner.rs \
   "$work/src/datafusion/core/examples/morsel_sort_bench.rs"
(
  cd "$work/src"
  CARGO_TARGET_DIR="$work/target" timeout 900s cargo build \
    --locked --profile release-nonlto -j 12 \
    -p datafusion --no-default-features --features parquet,sql \
    --example morsel_sort_bench
)
timeout --kill-after=10s 300s taskset -c 0-7 \
  "$work/target/release-nonlto/examples/morsel_sort_bench" 8 1 5 8192 2048
timeout --kill-after=10s 300s taskset -c 0-7 \
  "$work/target/release-nonlto/examples/morsel_sort_bench" 8 1 3 1024 0
timeout --kill-after=10s 300s taskset -c 0-7 \
  "$work/target/release-nonlto/examples/morsel_sort_bench" 8 1 3 256 0
git worktree remove --force "$work/src" && rm -rf "$work"
```
