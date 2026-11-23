<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Intra-row-group Parquet morsel splitting (#167)

Comparison of stock (`morsel_mode = off`), legacy native (`morsel_mode = required`,
default opt-out `ParquetSplitConfig::default()` with `enabled: false`), and split
native (`morsel_mode = required`, opted in via `ParquetSplitConfig { enabled: true, ..Default::default() }`:
`min_row_group_rows = 1 Mi`, `target_slice_rows = 128 Ki`, page-aligned boundary
selection, `256 MiB` query-wide compressed split-buffer cap, `16 MiB` per-slice
compressed cap, slice-sized decoded workspace estimate capped at `128 MiB`, and
atomic Phase-2 budget reservation for row-filter pushdown) on the same files and
binary built from `benchmarks/morsel_parquet_skew/stream_runner.rs` on
`agent/morsel-issue-167`. This note does not replace the pinned
[#156 baseline](README.md#dduck2-warm-cache-baseline) or the pinned
[#128 baseline](../morsel_parquet/README.md).

## Method

`rob-tandy-dduck1`, November 23, 2026 UTC: Ubuntu 24.04.5, Linux
`7.0.0-1014-aws`, AArch64 Neoverse-N1, Btrfs on local EC2 NVMe instance storage,
Rust `1.99.1 (58cf8f9c3)`, Python `3.12.10` (`numpy 2.2.6`, `pyarrow 20.0.0`).
All runs were pinned to CPUs 0–7 (`taskset -c 0-7`) with 8 workers,
`batch_size = 8192`, `repartition_file_scans = false`,
`collect_statistics = false`, `pruning = false`, `bloom_filter_on_read = false`,
1 warmup round, and 5 measured paired rounds rotating arm order each round.
Full Linux page-cache residency was verified via `mincore` before every timed
sample. Every arm passed #128's exact NumPy oracle before timing
(`378,916` / `4,941,572` rows on the 8,388,608-row `#156` layouts;
`759,841` / `9,883,940` rows on the 16,777,216-row `#128` uniform layout).

Fixtures (Zstandard level 3, PLAIN encoding, `use_dictionary = False`, statistics
enabled, PyArrow default 1 MiB data pages):

| Fixture                    | Files | Row groups per file        | Rows / RG | Page/Offset Index | Total bytes |
| -------------------------- | ----: | -------------------------- | --------: | ----------------- | ----------: |
| `large-row-groups-noindex` |     2 | `2, 2` (4 RGs)             | 2,097,152 | No                | 155,608,030 |
| `large-row-groups-indexed` |     2 | `2, 2` (4 RGs)             | 2,097,152 | Yes               | 155,607,514 |
| `file-skew-indexed`        |    16 | `24, 1, 24, 1, 2, 2, 1×10` |   131,072 | Yes               | 155,793,381 |
| `uniform-128rg` (#128)     |    16 | `8 × 16` (128 RGs)         |   131,072 | No                | 311,529,117 |

## 1. `#156` `large-row-groups` no-index control (`pushdown_filters = false`)

With `enable_page_index = true` and splitting enabled, files without an Offset
Index cleanly fall back to unsplit row groups (`parquet_split_row_groups = 0`,
4 morsels across 4 of 8 workers), matching legacy native with zero overhead.

| Query        | Stock ms, median (range) | Native (no-index) ms, median (range) | Active workers | Claims (round 1)           | `bytes_scanned` |
| ------------ | -----------------------: | -----------------------------------: | -------------: | -------------------------- | --------------: |
| `scan_low`   |      298.3 (297.1–300.8) |                  186.3 (184.6–188.8) |            4/8 | `[1, 0, 0, 1, 1, 0, 0, 1]` |      98,222,226 |
| `count_low`  |      145.5 (143.5–146.7) |                    93.0 (92.7–93.6)  |            4/8 | `[0, 0, 0, 1, 1, 1, 0, 1]` |      37,497,725 |
| `scan_high`  |      297.3 (294.2–302.9) |                  190.2 (189.8–193.5) |            4/8 | `[1, 0, 0, 1, 1, 0, 0, 1]` |      98,222,226 |
| `count_high` |      146.6 (144.5–147.6) |                    93.8 (93.0–94.4)  |            4/8 | `[1, 0, 0, 1, 0, 1, 0, 1]` |      37,497,725 |

## 2. `#156` `large-row-groups` indexed (`pushdown_filters = false`)

All 4 large row groups split cleanly into page-aligned slices (`60` slices for
`scan_*`, `32` slices for `count_*`). **All 8 workers claim and complete slices
in every trial**, `bytes_scanned` is **byte-for-byte identical (1.000×)** to
unsplit storage reads, and peak unique compressed buffer residency stays at
21–33 MiB (well below the 256 MiB cap).

| Query        | Stock ms, median (range) | Legacy native ms (4/8 workers) | Split native ms (8/8 workers) | Split vs legacy | Split vs stock | Split claims (round 1)     | `bytes_scanned` (all arms) | Reused bytes | Peak unique bytes |
| ------------ | -----------------------: | -----------------------------: | ----------------------------: | --------------: | -------------: | -------------------------- | -------------------------: | -----------: | ----------------: |
| `scan_low`   |      300.0 (298.8–301.9) |            186.0 (185.5–186.1) |           158.4 (155.7–161.0) |          -14.8% |         -47.2% | `[7, 6, 8, 9, 7, 7, 8, 8]` |                 98,214,742 |   74,547,897 |        34,154,405 |
| `count_low`  |      145.2 (143.7–146.0) |              94.1 (93.2–94.2)  |             68.2 (65.3–71.7)  |          -27.5% |         -53.0% | `[3, 3, 4, 4, 5, 4, 4, 5]` |                 37,495,813 |   16,371,853 |        22,446,309 |
| `scan_high`  |      301.4 (300.3–302.6) |            192.5 (189.6–193.9) |           164.9 (162.4–173.6) |          -14.3% |         -45.3% | `[7, 7, 7, 9, 8, 7, 7, 8]` |                 98,214,742 |   74,547,897 |        33,888,195 |
| `count_high` |      144.6 (143.3–147.1) |              93.7 (93.4–93.9)  |             70.8 (66.8–73.1)  |          -24.4% |         -51.0% | `[4, 4, 5, 5, 4, 3, 4, 3]` |                 37,495,813 |   16,371,853 |        22,718,476 |

Worker ready-wait drops sharply when all 8 workers have claimable slices:
`scan_low` aggregate `morsel_ready_wait` drops from `243.0 ms` → `44.1 ms`, and
`count_low` drops from `98.4 ms` → `34.4 ms`. On `count_*`, snapping slice
boundaries to `delta`'s ~273 Ki-row pages yields `32` slices with only `1.44×`
decode bytes (`16.4 MiB` reused from the broker, zero extra storage reads),
cutting wall time by **24.4–27.5%** over legacy native and **51.0–53.0%** over
stock. On `scan_*`, snapping to `payload`'s ~140 Ki-row pages keeps `payload`
at `1.00×` while `score`, `delta`, and `row_id` straddle boundaries (`1.76×`
total decode bytes, `74.5 MiB` served from the broker with zero duplicate I/O),
yielding **14.3–14.8%** wall-time wins over legacy native and **45.3–47.2%**
over stock.

## 3. `#156` `large-row-groups` indexed (`pushdown_filters = true`)

Splitting also applies to `FilteredBufferedReaders` (with `AsyncRangeBroker`
deduplicating shared boundary pages across Phase-1 and concurrent Phase-2 lanes,
`RowGroupCache` batch-boundary expansion scoped strictly to cached leaves, and
atomic Phase-2 budget pre-reservation at slice admission). Measured with 1
warmup and 10 paired repeats:

| Query        | Stock ms, median (range) | Legacy native ms (4/8 workers) | Split native ms (8/8 workers) | Split vs legacy | Split vs stock | Split claims (round 1)      | `bytes_scanned` (all arms) | Peak unique bytes |
| ------------ | -----------------------: | -----------------------------: | ----------------------------: | --------------: | -------------: | --------------------------- | -------------------------: | ----------------: |
| `scan_low`   |      335.6 (330.5–338.3) |            180.4 (178.7–183.3) |           175.9 (171.9–189.8) |           -2.5% |         -47.6% | `[7, 9, 9, 6, 7, 10, 6, 6]` |                 98,214,742 |        29,575,609 |
| `count_low`  |      162.8 (161.2–163.6) |              91.1 (90.0–92.4)  |             71.3 (65.3–75.0)  |          -21.7% |         -56.2% | `[4, 4, 3, 4, 5, 4, 5, 3]`  |                 37,495,813 |        14,576,713 |
| `scan_high`  |      351.0 (347.4–356.7) |            186.0 (184.8–188.0) |           185.8 (179.4–188.2) |           -0.1% |         -47.1% | `[7, 10, 7, 8, 9, 9, 6, 4]` |                 98,214,742 |        28,437,567 |
| `count_high` |      162.6 (159.8–165.2) |              92.1 (90.2–92.9)  |             87.9 (80.3–101.9) |           -4.6% |         -45.9% | `[4, 4, 4, 4, 4, 4, 4, 4]`  |                 37,495,813 |        14,357,437 |

All 8 workers claim work across all 4 queries with zero duplicate storage I/O
(`bytes_scanned` identical across all arms), staying **45.9–56.2% faster than
stock**. Compared to legacy native, `count_low` shows a clear **21.7%** speedup
(`91.1 → 71.3 ms`) and `count_high` is modestly faster on median (`92.1 →
87.9 ms`, with overlapping ranges), whereas `scan_low` and `scan_high` are
roughly a tie to slightly slower depending on run-to-run scheduling (with
per-slice `RowFilter` + `RowGroupCache` boundary-page decompression overhead
largely offsetting the 4→8 worker concurrency gain on the 4-column scan).

## 4. `#156` `file-skew` indexed control (`pushdown_filters = false`)

Row groups are `131,072` rows (`< 1 Mi` threshold), so `parquet_split_row_groups = 0`:
all 64 row groups stay unsplit and take the fast single-decoder path, spreading
across all 8 workers with zero regression (**2.2–2.4× faster than stock**).

| Query        | Stock ms, median (range) | Legacy native ms (8/8 workers) | Split native ms (8/8 workers) | Split RGs | Claims (round 1)           |
| ------------ | -----------------------: | -----------------------------: | ----------------------------: | --------: | -------------------------- |
| `scan_low`   |      232.5 (222.2–235.0) |              99.2 (94.8–106.4) |              99.0 (96.0–101.7)|         0 | `[8, 8, 8, 8, 8, 8, 8, 8]` |
| `count_low`  |      116.9 (115.5–119.2) |              51.0 (50.3–52.1)  |              52.5 (50.1–57.2) |         0 | `[9, 8, 7, 8, 8, 9, 7, 8]` |
| `scan_high`  |      244.3 (237.9–246.4) |            110.3 (107.4–115.5) |            112.6 (106.7–115.4)|         0 | `[8, 8, 7, 8, 8, 8, 9, 8]` |
| `count_high` |      123.1 (117.7–124.5) |              51.6 (49.8–54.5)  |              51.7 (51.0–53.4) |         0 | `[8, 8, 9, 9, 7, 7, 8, 8]` |

## 5. `#128` uniform 128-RG non-regression check (`pushdown_filters = false`)

Row groups are `131,072` rows (`< 1 Mi` threshold), so `parquet_split_row_groups = 0`
and all 128 row groups stay on the unsplit fast path with identical `bytes_scanned`
and zero regression (all four queries beat paired stock).

| Query        | Stock ms, median (range) | Legacy native (`MORSEL_SPLIT_ENABLED=0`) | Split native (`MORSEL_SPLIT_ENABLED=1`) | Split RGs |
| ------------ | -----------------------: | ---------------------------------------: | --------------------------------------: | --------: |
| `scan_low`   |      174.9 (168.7–179.1) |                      176.2 (172.5–177.8) |                     174.4 (171.5–176.0) |         0 |
| `count_low`  |        93.3 (90.9–95.6)  |                        93.6 (91.8–106.1) |                       91.4 (89.2–93.1)  |         0 |
| `scan_high`  |      197.7 (195.2–199.3) |                      196.4 (194.0–198.1) |                     195.0 (194.2–202.5) |         0 |
| `count_high` |        93.2 (90.5–93.9)  |                        92.5 (91.5–94.5)  |                       90.1 (88.9–93.7)  |         0 |

## 6. Bounded-pool trade-off

Because only split slices reserve compressed and decoded workspace memory in
the query `MemoryPool` (legacy unsplit Parquet readers do not reserve against
the pool), a very tight bounded pool intentionally paces split concurrency to
stay within budget rather than failing. On `large-row-groups-indexed` with
`MORSEL_MEMORY_POOL_BYTES=16777216` (`GreedyMemoryPool(16 MiB)`, 8 workers, 1
warmup + 5 measured runs):

| `pushdown_filters` | Query       | Legacy native (`split=0`) ms, median (range) | Split native (`split=1`) ms, median (range) | Split vs legacy | Split peak unique bytes |
| ------------------ | ----------- | -------------------------------------------: | ------------------------------------------: | --------------: | ----------------------: |
| `false`            | `scan_low`  |                          185.7 (184.0–186.7) |                         516.2 (515.4–523.3) |           2.78× |               7,528,132 |
| `false`            | `count_low` |                            93.6 (93.0–94.9)  |                         115.7 (113.1–116.0) |           1.24× |               6,957,114 |
| `true`             | `scan_low`  |                          183.6 (182.2–184.3) |                         600.2 (582.7–608.2) |           3.27× |               7,395,622 |
| `true`             | `count_low` |                            92.2 (92.1–93.0)  |                         117.3 (116.8–119.5) |           1.27× |               6,782,324 |

On wider projections (e.g. 6-column `SELECT *`), per-slice reservations further
reduce concurrent in-flight slices under a 16 MiB pool (making split native
~3–7× slower than unsplit legacy native), whereas pools too small to fit even
one slice fall back cleanly to the unsplit reader before publishing slice 0.

## Reproduction

Run from the repository root using the committed `benchmarks/morsel_parquet_skew/stream_runner.rs`,
`benchmarks/morsel_parquet_skew/bench.py`, and `benchmarks/morsel_parquet/harness.py`:

```bash
set -e
b=benchmarks/morsel_parquet_skew
work=$(mktemp -d "$HOME/.cache/issue167-repro.XXXXXX")
rev=$(git rev-parse HEAD)

git worktree add --detach "$work/checkout" "$rev"
mkdir -p "$work/checkout/datafusion/core/examples"
cp "$b/stream_runner.rs" "$work/checkout/datafusion/core/examples/morsel_parquet_skew.rs"
(
  cd "$work/checkout"
  MORSEL_BENCH_REVISION="$rev" CARGO_TARGET_DIR="$work/target" \
    cargo build --locked --profile release-nonlto -j 10 \
      -p datafusion --no-default-features --features parquet,sql --example morsel_parquet_skew
)
runner="$work/target/release-nonlto/examples/morsel_parquet_skew"
uv venv --python 3.12.10 "$work/venv"
uv pip install --python "$work/venv/bin/python" -r benchmarks/morsel_parquet/requirements.txt
py="$work/venv/bin/python"

# Verify harness unit + smoke tests
MORSEL_PARQUET_RUNNER="$runner" MORSEL_BENCH_REVISION="$rev" \
  taskset -c 0-7 "$py" -B -m unittest discover -s "$b" -v

# Generate fixtures
"$py" -B "$b/bench.py" generate large-row-groups "$work/large-row-groups-noindex"
"$py" -B "$b/bench.py" generate large-row-groups "$work/large-row-groups-indexed" --page-index
"$py" -B "$b/bench.py" generate file-skew "$work/file-skew-indexed" --page-index
"$py" -B benchmarks/morsel_parquet/harness.py generate "$work/uniform-128rg" \
  --files 16 --groups 8 --group-rows 131072

# 1. No-index control (splitting opted in, falls back cleanly when Offset Index is absent)
taskset -c 0-7 "$py" -B "$b/bench.py" ab large-row-groups "$work/large-row-groups-noindex" \
  --runner "$runner" --revision "$rev" --enable-page-index --enable-split --warmups 1 --repeats 5 > "$work/lrg-noindex.jsonl"

# 2. Indexed large-row-groups (pushdown_filters = false): stock vs legacy native vs split native
taskset -c 0-7 "$py" -B "$b/bench.py" ab large-row-groups "$work/large-row-groups-indexed" \
  --runner "$runner" --revision "$rev" --enable-page-index --compare-split --warmups 1 --repeats 5 > "$work/lrg-pd0.jsonl"

# 3. Indexed large-row-groups (pushdown_filters = true): stock vs legacy native vs split native
taskset -c 0-7 "$py" -B "$b/bench.py" ab large-row-groups "$work/large-row-groups-indexed" \
  --runner "$runner" --revision "$rev" --enable-page-index --pushdown-filters --compare-split \
  --warmups 1 --repeats 10 > "$work/lrg-pd1.jsonl"

# 4. Indexed file-skew control
taskset -c 0-7 "$py" -B "$b/bench.py" ab file-skew "$work/file-skew-indexed" \
  --runner "$runner" --revision "$rev" --enable-page-index --compare-split --warmups 1 --repeats 5 > "$work/file-skew.jsonl"

# 5. #128 uniform 128-RG non-regression check
MORSEL_ENABLE_PAGE_INDEX=1 MORSEL_SPLIT_ENABLED=0 MORSEL_PUSHDOWN_FILTERS=0 \
  taskset -c 0-7 "$py" -B benchmarks/morsel_parquet/harness.py ab "$work/uniform-128rg" \
  --runner "$runner" --threads 8 --cache warm --revision "$rev" --warmups 1 --repeats 5 > "$work/uniform-legacy.jsonl"
MORSEL_ENABLE_PAGE_INDEX=1 MORSEL_SPLIT_ENABLED=1 MORSEL_PUSHDOWN_FILTERS=0 \
  taskset -c 0-7 "$py" -B benchmarks/morsel_parquet/harness.py ab "$work/uniform-128rg" \
  --runner "$runner" --threads 8 --cache warm --revision "$rev" --warmups 1 --repeats 5 > "$work/uniform-split.jsonl"
```
