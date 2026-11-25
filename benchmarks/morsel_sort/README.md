<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Morsel Native Parallel TopN and Sort Benchmark Note (#159)

This note records a paired **Stock DataFusion (`morsel_mode=off`) vs Native
Morsel DataFusion (`morsel_mode=required`)** release benchmark for the native
pipeline-breaking `TopNSink` (`ORDER BY ... LIMIT K [OFFSET O]`) and parallel
`SortSink` (`ORDER BY ...`) over uniform and skewed Parquet datasets.

## Datasets and Workloads

Generated deterministically by `benchmarks/morsel_sort/sort_runner.rs`:

- **Schema:** `id: Int64` (unique `0..4,000,000`), `score: Int64` (nullable,
  ~1% `NULL`), `metric: Float64` (nullable, ~1% `NULL`), `grp: Int32` (`0..64`),
  `tag: Utf8` (6 short string labels).
- **Uniform Parquet (`pq_uniform`):** `4,000,000` rows across 16 equal Zstd
  Parquet files (`250,000` rows each, 2 row groups of `125,000` rows per file),
  uniform random `score` values in `0..10,000,000`.
- **Skewed Parquet (`pq_skewed`):** `4,000,000` rows across 16 Zstd Parquet
  files combining **file/partition skew** (1 hot file of `1,750,000` rows
  across 14 row groups + 15 small files of `150,000` rows each) and **key
  skew** (50% of rows collide on 10 hot `score` values, 50% uniform across
  `0..10,000,000`, plus ~1% `NULL`).

Workloads (all verified for exact ordered row-count and position-sensitive
ordered key-digest parity on warmup, row-count parity on every measured round,
`MorselExec` with `TopNSink` / `SortSink` under `morsel_mode=required`, and
stock `SortExec` under `morsel_mode=off`):

1. `topn_uniform_limit100`: `SELECT id, score, tag FROM pq_uniform ORDER BY score DESC NULLS LAST, id ASC LIMIT 100`
2. `topn_skewed_limit100`: `SELECT id, score, tag FROM pq_skewed ORDER BY score DESC NULLS LAST, id ASC LIMIT 100`
3. `topn_uniform_offset`: `SELECT id, score, metric FROM pq_uniform ORDER BY score ASC NULLS FIRST, id ASC LIMIT 100 OFFSET 50`
4. `topn_skewed_filtered`: `SELECT id, score, tag FROM pq_skewed WHERE grp < 32 ORDER BY score DESC NULLS LAST, id ASC LIMIT 100`
5. `sort_uniform_single_key`: `SELECT id, score FROM pq_uniform ORDER BY score ASC NULLS LAST`
6. `sort_skewed_single_key`: `SELECT id, score FROM pq_skewed ORDER BY score ASC NULLS LAST`
7. `sort_uniform_multi_key`: `SELECT id, grp, score FROM pq_uniform ORDER BY grp ASC, score DESC NULLS LAST`
8. `sort_skewed_multi_key`: `SELECT id, grp, score FROM pq_skewed ORDER BY grp ASC, score DESC NULLS LAST`

## Environment and Controls

- **Host:** `rob-tandy-dduck1`, `aarch64`, Linux `7.0.0-1014-aws`, 16 ARM vCPUs,
  61 GiB RAM.
- **CPU pinning and threads:** `taskset -c 0-7` (8 dedicated logical CPUs),
  8 Tokio workers (`target_partitions=8`), `batch_size=8,192`.
- **Host idle check:** `vmstat 1 2` showed 95–97% CPU idle before the run and
  95–99% idle after.
- **Build:** revision `8c6e8381285902e455172ee2c31faf4ab7cdcb72`,
  `--locked --profile release-nonlto --no-default-features --features parquet,sql`,
  `rustc 1.99.1 (58cf8f9c3 2026-10-13)`.
- **Parquet settings:** matched across both arms (`pushdown_filters=false`,
  `pruning=false`, warm page cache after dataset creation + warmup round).
- **Protocol:** 1 unmeasured warmup round (with ordered parity and plan
  assertions) and 7 measured paired rounds with alternating `off`/`required`
  execution order, creating a fresh physical plan on each round so dynamic
  state in stock `SortExec` / `TopK` is never carried across rounds.

## Results (8 Threads, `batch_size=8,192`)

| Workload                  | Stock median ms (min–max) | Morsel median ms (min–max) | Speedup | Output rows | Morsels |
| ------------------------- | ------------------------: | -------------------------: | ------: | ----------: | ------: |
| `topn_uniform_limit100`   |          38.4 (38.1–41.3) |           33.4 (32.3–37.3) |   1.15× |         100 |      32 |
| `topn_skewed_limit100`    |          36.4 (35.1–37.2) |           30.1 (28.2–32.1) |   1.21× |         100 |      44 |
| `topn_uniform_offset`     |          17.9 (17.6–18.2) |           46.1 (41.4–48.4) |   0.39× |         100 |      32 |
| `topn_skewed_filtered`    |          38.2 (37.8–39.5) |           32.3 (30.1–35.1) |   1.18× |         100 |      44 |
| `sort_uniform_single_key` |       296.9 (296.2–302.8) |        402.8 (396.9–407.7) |   0.74× |   4,000,000 |      32 |
| `sort_skewed_single_key`  |       270.2 (268.2–271.8) |        339.1 (331.7–354.3) |   0.80× |   4,000,000 |      44 |
| `sort_uniform_multi_key`  |       534.7 (527.9–542.7) |        747.8 (706.6–832.8) |   0.72× |   4,000,000 |      32 |
| `sort_skewed_multi_key`   |       515.3 (505.0–521.9) |        705.4 (685.7–733.6) |   0.73× |   4,000,000 |      44 |

## Observations

- **Parquet `TopN` (`ORDER BY ... DESC NULLS LAST LIMIT 100`):** Native morsel
  `TopNSink` is **1.15×–1.21× faster** than stock DataFusion on uniform
  (`33.4 ms` vs `38.4 ms`), skewed (`30.1 ms` vs `36.4 ms`), and filtered
  skewed (`32.3 ms` vs `38.2 ms`) workloads. Dynamic morsel claiming balances
  the 14-row-group hot Parquet file across all 8 workers (44 morsels), and the
  first-column primitive threshold pre-filter skips `RowConverter` encoding and
  payload materialization for almost every rejected row.
- **`topn_uniform_offset` (`ORDER BY score ASC NULLS FIRST, id ASC LIMIT 100 OFFSET 50`):**
  Stock DataFusion (`17.9 ms` vs `46.1 ms`) benefits from `TopKDynamicFilters`
  pushing a dynamic `score IS NULL` filter down into the Parquet reader once
  the top 150 `NULL` rows fill the heap, skipping non-null `id`/`metric` page
  decoding for ~99% of rows. Adding shared TopN dynamic filter pushdown into
  `ParquetScan` is a natural follow-up.
- **Full 4M-row `Sort` (`ORDER BY ...` without limit):** Native morsel
  `SortSink` tracks stock DataFusion (`0.72×–0.80×`) across uniform and skewed
  single-key and multi-key workloads. Two straightforward follow-up
  opportunities account for the remaining gap:
  1. `SortState` concatenates each worker's buffered batches (`~500K` rows)
     into a single contiguous `RecordBatch` before `lexsort_to_indices`,
     incurring a full copy and larger working set than sorting smaller
     L2-resident runs.
  2. `MorselExec` funnels all 4M merged rows (489 output batches) through a
     bounded `mpsc::channel(2)` from the supervisor task to the consumer
     stream, whereas stock `SortPreservingMergeStream` yields batches directly
     on consumer poll.

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
timeout --kill-after=10s 180s taskset -c 0-7 \
  "$work/target/release-nonlto/examples/morsel_sort_bench" 8 1 7
git worktree remove --force "$work/src" && rm -rf "$work"
```
