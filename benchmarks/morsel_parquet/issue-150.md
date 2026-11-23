<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Parquet row-level filter pushdown and late materialization (#150)

Before: `fc952a25ca54ba855986ff0f107a987103b5bfd5` (`morsel-execution` with #148
and #149). After: `a172ce2306750100db27c29bfcc3071de8594d88`. This comparison
does not replace the pinned [#128 baseline](README.md#native-parquet-ab-128).

## Method

`rob-tandy-dduck1`, November 23, 2026 UTC: Ubuntu 24.04.5, Linux
`7.0.0-1014-aws`, 16 Neoverse-N1 vCPUs, 61 GiB RAM, Rust 1.99.1 (`58cf8f9c3`),
Clang/lld 18.1.3, and Python 3.12.10. The private #128 16-file × 8-row-group ×
131,072-row fixture (311,529,117 bytes, Zstandard level 3, PLAIN encoding, no
dictionaries) was generated under `$HOME/.cache` on Btrfs on local EC2 NVMe
instance storage (`compress=zstd:1`). No builds ran during measurement; three
one-second `vmstat -y 1 3` samples before the run showed 97–100% host CPU idle.
Both arms used CPUs 0–7, eight workers, one warmup, and eight timed samples per
arm and query in each regime.

The unmodified #128 harness verified all four queries in every arm against the
NumPy oracle before timing: 759,841 low-selectivity (4.5%) and 9,883,940
high-selectivity (59%) rows and counts. Stock used `morsel_mode=off`; native
used `morsel_mode=required`, claimed all 128 row groups, and finalized one
count per count query. Every warm timed sample started with 76,063/76,063 file
pages resident; every page-cache-cold sample started with 0/76,063 resident
pages.

## Paired `pushdown_filters = true` timings and `bytes_scanned`

Total elapsed ms, from registration through stream exhaustion. Median (min–max).
Row-group stats pruning, page index, and Bloom filters remained disabled (as in
#128) while `datafusion.execution.parquet.pushdown_filters = true`.

Before #150, native forced `pushdown_filters = false` (`pushdown_rows_pruned = 0`)
while stock evaluated `RowFilter`. After #150, both stock and native evaluate
Parquet `RowFilter` with identical `pushdown_rows_pruned` (16,017,375 for
`*_low`; 6,893,276 for `*_high`), `pushdown_rows_matched` (759,841 for `*_low`;
9,883,940 for `*_high`), and `bytes_scanned` (196,691,567 bytes for `scan_*`;
75,280,626 bytes for `count_*`).

| Cache           | Case         |        Stock before |         Stock after |       Native before |        Native after | `bytes_scanned` |
| --------------- | ------------ | ------------------: | ------------------: | ------------------: | ------------------: | --------------: |
| Warm            | `scan_low`   | 215.8 (209.8–224.4) | 212.5 (207.4–221.8) | 173.5 (172.1–178.6) | 209.0 (206.9–215.4) |     196,691,567 |
| Warm            | `scan_high`  | 208.3 (198.5–212.5) | 202.0 (196.5–208.7) | 194.7 (192.3–202.9) | 204.5 (199.2–231.2) |     196,691,567 |
| Warm            | `count_low`  |   94.7 (88.7–100.3) |    92.0 (88.1–96.0) |    91.3 (90.1–94.0) |    87.1 (86.1–90.0) |      75,280,626 |
| Warm            | `count_high` |    91.6 (88.3–93.7) |    91.7 (86.9–92.7) |    94.7 (93.6–97.3) |    86.8 (85.4–89.4) |      75,280,626 |
| Page-cache-cold | `scan_low`   | 236.6 (234.0–237.5) | 235.8 (233.6–242.7) | 186.1 (184.5–189.1) | 230.3 (224.2–235.2) |     196,691,567 |
| Page-cache-cold | `scan_high`  | 222.2 (217.4–225.0) | 222.0 (218.1–229.2) | 213.1 (208.3–216.8) | 222.4 (219.3–228.8) |     196,691,567 |
| Page-cache-cold | `count_low`  |   99.6 (97.8–102.3) |   98.9 (98.0–103.9) |    95.8 (94.2–97.5) |    91.3 (90.3–92.3) |      75,280,626 |
| Page-cache-cold | `count_high` |  101.5 (99.5–105.7) |   99.1 (97.1–102.3) |   98.9 (98.0–106.5) |    92.8 (89.4–94.6) |      75,280,626 |

Key observations under `pushdown_filters = true`:

- **Native matches or beats stock on every workload.** Native is faster than
  stock on `scan_low` (209.0 vs 212.5 ms warm; 230.3 vs 235.8 ms cold),
  `count_low` (87.1 vs 92.0 ms warm; 91.3 vs 98.9 ms cold; 5.3–7.7% faster),
  and `count_high` (86.8 vs 91.7 ms warm; 92.8 vs 99.1 ms cold; 5.3–6.4%
  faster), and within 1.2% on `scan_high`.
- **Filtered `COUNT(*)` improves over before** because projection narrowing
  reduces `NativeParquet`'s output projection to `[]` (zero columns), avoiding
  all output array materialization: `count_low` drops from 91.3 → 87.1 ms warm
  (95.8 → 91.3 ms cold) and `count_high` drops from 94.7 → 86.8 ms warm
  (98.9 → 92.8 ms cold).
- **Time to first 8,192 rows on `scan_low`** drops from 32.7 → 22.4 ms warm
  (stock 21.6 ms) and from 37.1 → 26.0 ms cold (stock 24.3 ms).
- **Why `scan_low` with `pushdown_filters = true` is slower than
  `pushdown_filters = false` in both stock and native on #128:** the #128
  fixture uses uniform-random 4.5% selectivity across PLAIN-encoded fixed-width
  `Int64` columns with page indexes disabled, so every row group and data page
  of `row_id` and `payload` still contains matching rows and must be fetched and
  Zstd-decompressed (`bytes_scanned` stays 196,691,567 bytes in both stock and
  native), while `RowSelection` skipping + predicate-cache overhead for `delta`
  exceeds bulk fixed-width decode + `filter_record_batch`. On workloads with
  unmatched row groups or multi-page non-predicate columns with page indexes
  enabled (covered by `morsel_parquet` integration tests), `RowFilter` skips
  both `bytes_scanned` and decoding of unmatched pages/columns identically to
  stock DataFusion.

## `pushdown_filters = false` regression check (warm)

| Case         |        Stock before |         Stock after |       Native before |        Native after | `bytes_scanned` |
| ------------ | ------------------: | ------------------: | ------------------: | ------------------: | --------------: |
| `scan_low`   | 169.1 (166.0–171.6) | 173.3 (166.2–210.2) | 172.6 (170.6–179.1) | 170.7 (168.7–174.4) |     196,691,567 |
| `scan_high`  | 196.1 (190.4–210.7) | 190.8 (188.3–198.4) | 197.7 (193.4–205.7) | 194.1 (190.6–198.7) |     196,691,567 |
| `count_low`  |    89.6 (87.5–91.7) |    88.3 (86.4–98.0) |    90.3 (89.3–95.1) |    90.8 (89.4–94.4) |      75,280,626 |
| `count_high` |    91.9 (87.6–95.9) |    91.0 (86.8–95.4) |    94.4 (93.2–98.2) |    93.5 (91.5–94.6) |      75,280,626 |

Disabled behavior (`pushdown_filters = false`) is unchanged within measurement
noise.

## #138 wait and buffering metrics (`pushdown_filters = true`)

Medians of per-run totals in ms, before → after (overlapping sums across lanes
or workers).

| Cache           | Case         |   Pending I/O |  Ready wait |       Decode work | Process work | Output wait | Peak ready bytes |
| --------------- | ------------ | ------------: | ----------: | ----------------: | -----------: | ----------: | ---------------: |
| Warm            | `scan_low`   | 247.9 → 150.7 |  19.0 → 9.2 | 1,021.6 → 1,446.4 |  175.4 → 1.6 |       0 → 0 |   19.5 → 4.7 MiB |
| Warm            | `scan_high`  | 224.7 → 131.8 |  18.8 → 9.7 | 1,012.5 → 1,308.1 | 227.6 → 13.1 | 42.7 → 19.7 |   19.5 → 4.7 MiB |
| Warm            | `count_low`  | 133.6 → 185.8 |  11.1 → 9.3 |     458.2 → 592.1 |  132.6 → 0.1 |       0 → 0 |   19.5 → 4.7 MiB |
| Warm            | `count_high` | 134.4 → 176.2 |  12.9 → 8.8 |     456.8 → 581.8 |  163.6 → 0.3 |       0 → 0 |   19.5 → 4.7 MiB |
| Page-cache-cold | `scan_low`   | 478.6 → 224.4 | 57.8 → 22.0 | 1,055.9 → 1,409.9 |  180.2 → 1.6 |       0 → 0 |   19.5 → 4.7 MiB |
| Page-cache-cold | `scan_high`  | 421.4 → 216.6 | 56.8 → 21.4 | 1,031.5 → 1,275.9 | 237.5 → 12.9 | 50.0 → 13.8 |   19.5 → 4.7 MiB |
| Page-cache-cold | `count_low`  | 256.8 → 325.2 | 20.4 → 22.0 |     466.2 → 594.9 |  142.6 → 0.1 |       0 → 0 |   19.5 → 4.7 MiB |
| Page-cache-cold | `count_high` | 250.3 → 339.7 | 22.8 → 23.3 |     463.4 → 595.4 |  170.8 → 0.2 |       0 → 0 |   19.5 → 4.7 MiB |

`morsel_process_work` collapses from 132–237 ms to 0.1–13 ms as predicate
evaluation moves into Parquet `RowFilter` inside `morsel_decode_work` on the
pipeline workers. Producer lanes only prefetch predicate column byte ranges
(phase 1), shrinking ready-queue buffered bytes by 4.1× (19.5 → 4.7 MiB) and
reducing `morsel_ready_wait` on scans from 19.0 → 9.2 ms warm and 57.8 → 22.0 ms
cold. Workers fetch surviving output column ranges (phase 2) lazily via lock-free
object-store range reads only when `RowFilter` leaves matching rows and
non-predicate output columns remain.

## Reproduction

Generate the #128 16 × 8 × 131,072 fixture using the pinned Python environment
in [README.md](README.md#native-parquet-ab-128). Copy `stream_runner.rs` to a
temporary example `datafusion/core/examples/morsel_pushdown_bench.rs` in
detached worktrees for the before and after revisions, replacing
`execution.parquet.pushdown_filters = false;` with
`std::env::var("MORSEL_PUSHDOWN_FILTERS").is_ok_and(|v| v == "1")` (and
attaching `MorselMetrics` when `MORSEL_METRICS=1`). Build each revision with
`--profile release-nonlto -p datafusion --no-default-features --features parquet,sql --example morsel_pushdown_bench`,
then run:

```bash
for pushdown in 1 0; do
  for cache in warm page-cache-cold; do
    MORSEL_PUSHDOWN_FILTERS="$pushdown" MORSEL_METRICS=1 \
      timeout --kill-after=10s 240s taskset -c 0-7 "$py" -B \
      benchmarks/morsel_parquet/harness.py ab "$fixture" \
      --runner "$runner" --revision "$revision" --threads 8 --timeout 120 \
      --cache "$cache" --warmups 1 --repeats 8 > "$revision-pushdown$pushdown-$cache.jsonl"
  done
done
```
