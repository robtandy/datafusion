<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Morsel Native Parallel TopN and Sort Benchmark Note (#159)

This note records paired **Stock DataFusion (`morsel_mode=off`) vs Native
Morsel DataFusion (`morsel_mode=required`)** release benchmarks for the native
pipeline-breaking `TopNSink` (`ORDER BY ... LIMIT K [OFFSET O]`) and parallel
`SortSink` (`ORDER BY ...`) across uniform and skewed Parquet scans (integer
and out-of-line `Utf8View` sort keys), large-`K` TopN (`100K`, `1M`, and
`> row count`), and 400K×4M Parquet `INNER JOIN` inputs at
`batch_size = 8,192`, `1,024`, and `256`.

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
5. `topn_limit_100k`: `SELECT id, score FROM pq_uniform ORDER BY score ASC NULLS LAST, id ASC LIMIT 100000`
6. `topn_limit_1m`: `SELECT id, score FROM pq_uniform ORDER BY score ASC NULLS LAST, id ASC LIMIT 1000000`
7. `topn_limit_gt_rows`: `SELECT id, score FROM pq_uniform ORDER BY score ASC NULLS LAST, id ASC LIMIT 10000000`
8. `sort_uniform_single_key`: `SELECT id, score FROM pq_uniform ORDER BY score ASC NULLS LAST`
9. `sort_skewed_single_key`: `SELECT id, score FROM pq_skewed ORDER BY score ASC NULLS LAST`
10. `sort_utf8view_key`: `SELECT id, name FROM pq_uniform ORDER BY name ASC NULLS LAST, id ASC`
11. `sort_join_id`: `SELECT d.name, f.id FROM fact_pq f JOIN dim_pq d ON f.k = d.k ORDER BY f.id`
12. `sort_join_utf8view`: `SELECT d.name, f.id FROM fact_pq f JOIN dim_pq d ON f.k = d.k ORDER BY d.name, f.id`

## Environment and Controls

- **Host:** `rob-tandy-dduck1`, `aarch64`, Linux `7.0.0-1014-aws`, 16 ARM vCPUs,
  61 GiB RAM.
- **CPU pinning and threads:** `taskset -c 0-7` (8 dedicated logical CPUs),
  8 Tokio workers (`target_partitions=8`).
- **Host idle check:** `vmstat 1 2` showed 95–97% CPU idle before and 95–98%
  idle after the `batch_size=8,192` run.
- **Build:** revision `cf8d11cfe` (`perf(morsel): avoid per-batch full-heap
  rebuilds in TopNState (#159)`),
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
| `topn_uniform_limit100`   |          72.9 (72.4–74.0) |            59.1 (57.3–65.4) |   1.23× |         100 |      32 |
| `topn_skewed_limit100`    |          70.2 (69.4–71.4) |            59.3 (58.5–61.9) |   1.18× |         100 |      44 |
| `topn_utf8view_limit100`  |          88.1 (87.3–89.5) |            64.7 (64.1–69.1) |   1.36× |         100 |      32 |
| `topn_join_utf8view_100`  |       135.1 (130.4–137.1) |         137.6 (135.7–139.8) |   0.98× |         100 |      36 |
| `topn_limit_100k`         |       542.5 (538.0–555.4) |         273.4 (272.1–277.4) |   1.98× |     100,000 |      32 |
| `topn_limit_1m`           |   1,170.0 (1167.1–1200.9) |         961.2 (945.6–973.9) |   1.22× |   1,000,000 |      32 |
| `topn_limit_gt_rows`      |   1,447.3 (1419.6–1450.1) |     1,204.9 (1187.8–1399.6) |   1.20× |   4,000,000 |      32 |
| `sort_uniform_single_key` |       293.7 (291.1–296.7) |         362.9 (356.8–369.8) |   0.81× |   4,000,000 |      32 |
| `sort_skewed_single_key`  |       266.4 (264.7–271.4) |         317.1 (309.1–322.7) |   0.84× |   4,000,000 |      44 |
| `sort_utf8view_key`       |       941.3 (919.1–945.4) |     1,534.8 (1513.3–1662.2) |   0.61× |   4,000,000 |      32 |
| `sort_join_id`            |       378.3 (369.3–416.2) |         603.8 (530.7–626.5) |   0.63× |   4,000,000 |      36 |
| `sort_join_utf8view`      |       874.2 (815.1–921.9) |     1,405.0 (1335.8–1601.3) |   0.62× |   4,000,000 |      36 |

## Small-Batch Scaling (`batch_size=1,024` and `batch_size=256`, 8 Threads)

| Workload                  | `bs=1,024` Stock ms | `bs=1,024` Morsel ms | Speedup | `bs=256` Stock ms | `bs=256` Morsel ms | Speedup |
| ------------------------- | ------------------: | -------------------: | ------: | ----------------: | -----------------: | ------: |
| `topn_uniform_limit100`   |                87.7 |                 65.9 |   1.33× |             123.8 |               72.9 |   1.70× |
| `topn_skewed_limit100`    |                85.1 |                 62.8 |   1.35× |             114.9 |               72.3 |   1.59× |
| `topn_utf8view_limit100`  |               103.6 |                 66.4 |   1.56× |             143.4 |               83.8 |   1.71× |
| `topn_join_utf8view_100`  |               178.2 |                155.7 |   1.14× |             285.2 |              194.5 |   1.47× |
| `topn_limit_100k`         |               587.3 |                286.9 |   2.05× |             703.3 |              315.0 |   2.23× |
| `topn_limit_1m`           |             1,192.3 |                992.9 |   1.20× |           1,252.2 |            1,057.1 |   1.18× |
| `topn_limit_gt_rows`      |             1,514.7 |              1,257.5 |   1.20× |           1,671.9 |            1,466.9 |   1.14× |
| `sort_uniform_single_key` |               345.9 |                436.4 |   0.79× |             499.2 |              617.5 |   0.81× |
| `sort_skewed_single_key`  |               341.4 |                372.6 |   0.92× |             609.2 |              557.3 |   1.09× |
| `sort_utf8view_key`       |             1,211.7 |              1,852.8 |   0.65× |           1,899.6 |            1,871.6 |   1.01× |
| `sort_join_id`            |               675.8 |                672.7 |   1.00× |           1,598.4 |            1,821.6 |   0.88× |
| `sort_join_utf8view`      |               846.1 |              1,494.2 |   0.57× |           2,296.3 |            2,427.9 |   0.95× |

## Observations

- **`TopNSink` across small-`K`, large-`K` (`100K`, `1M`, `> rows`), `Utf8View`, and Parquet HashJoin:**
  - Retaining original per-batch row indices in heap entries with on-demand
    sorted `row_map` lookup eliminates per-batch full-heap rebuilds, keeping
    per-batch work strictly `O(winners · log(K+O))` and avoiding `take` copies
    while the heap is filling.
  - Across all three batch sizes (`8,192`, `1,024`, `256`), large-`K` `TopN`
    beats stock DataFusion by **1.98×–2.23×** on `LIMIT 100,000`,
    **1.18×–1.22×** on `LIMIT 1,000,000`, and **1.14×–1.20×** on
    `LIMIT 10,000,000` (`> row count`).
  - Small-`K` direct scans are **1.18×–1.36× faster** at `batch_size=8,192`,
    **1.33×–1.56× faster** at `1,024`, and **1.59×–1.71× faster** at `256`;
    `topn_join_utf8view_100` matches stock at `8,192` (`0.98×`) and beats stock
    at `1,024` (`1.14×`) and `256` (`1.47×`) inside a 1 GiB pool.
- **Full 4M-row `SortSink` over `Utf8View` and Parquet HashJoin inputs:**
  - Shared-buffer deduplication (`SharedSortMemory` +
    `RecordBatchMemoryCounter`) and single-pass deduplicating `Utf8View`
    concatenation keep each worker to ~1 sorted run even at `batch_size=256`.
  - All three Parquet join queries (`topn_join_utf8view_100`, `sort_join_id`,
    `sort_join_utf8view`) succeed cleanly within a 1 GiB pool at
    `batch_size=8,192` and `1,024`, and small-batch scaling at `batch_size=256`
    tracks or beats stock (`0.81×–1.09×`, with `sort_join_utf8view` at `0.95×`
    and `sort_utf8view_key` at `1.01×`).

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
