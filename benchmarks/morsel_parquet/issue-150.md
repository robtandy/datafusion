<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Parquet row-level filter pushdown and late materialization (#150)

Before: `fc952a25ca54ba855986ff0f107a987103b5bfd5` (`morsel-execution` with #148
and #149). After: `7f3d917822e4e4b52d85a9333cf47b451f582902`. This comparison
does not replace the pinned [#128 baseline](README.md#native-parquet-ab-128).

## Method

`rob-tandy-dduck1`, November 23, 2026 UTC: Ubuntu 24.04.5, Linux
`7.0.0-1014-aws`, 16 Neoverse-N1 vCPUs, 61 GiB RAM, Rust 1.99.1 (`58cf8f9c3`),
Clang/lld 18.1.3, and Python 3.12.10. The private #128 16-file × 8-row-group ×
131,072-row fixture (311,529,117 bytes, Zstandard level 3, PLAIN encoding, no
dictionaries) was generated under `$HOME/.cache` on Btrfs on local EC2 NVMe
instance storage (`compress=zstd:1`). No builds ran during measurement; both
arms used CPUs 0–7, eight workers, one warmup, and eight timed samples per arm
and query in each regime.

The #128 harness verified all four queries in every arm against the NumPy oracle
before timing: 759,841 low-selectivity (4.5%) and 9,883,940 high-selectivity
(59%) rows and counts. Stock used `morsel_mode=off`; native used
`morsel_mode=required`, claimed all 128 row groups, showed
`NativeParquet[fixture, predicate=...]` under `pushdown_filters = true` (and
`NativeParquet[fixture]` under `pushdown_filters = false`), and finalized one
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
| Warm            | `scan_low`   | 217.7 (213.5–221.9) | 212.2 (203.9–233.8) | 174.2 (170.5–180.3) | 214.3 (210.5–222.9) |     196,691,567 |
| Warm            | `scan_high`  | 204.8 (200.2–208.9) | 203.3 (198.8–209.6) | 197.6 (192.5–206.4) | 201.2 (197.1–205.1) |     196,691,567 |
| Warm            | `count_low`  |   94.5 (87.2–106.1) |   90.8 (87.6–105.2) |    91.8 (90.3–93.7) |    90.0 (87.2–94.0) |      75,280,626 |
| Warm            | `count_high` |   91.9 (89.9–102.2) |    91.7 (87.6–93.5) |   97.2 (92.8–100.9) |    88.3 (85.6–91.7) |      75,280,626 |
| Page-cache-cold | `scan_low`   | 247.5 (231.1–260.0) | 239.9 (239.4–256.5) | 188.4 (182.6–201.5) | 232.1 (228.3–253.0) |     196,691,567 |
| Page-cache-cold | `scan_high`  | 231.6 (220.8–237.8) | 232.7 (228.0–238.4) | 210.6 (208.7–214.0) | 224.1 (220.1–234.1) |     196,691,567 |
| Page-cache-cold | `count_low`  |  102.4 (97.9–105.5) | 106.2 (100.9–120.0) |   95.9 (95.1–110.7) |    93.3 (91.5–99.4) |      75,280,626 |
| Page-cache-cold | `count_high` | 104.2 (101.2–107.8) | 103.5 (101.5–118.8) |  101.8 (97.6–108.0) |   94.4 (92.7–100.3) |      75,280,626 |

Key observations under `pushdown_filters = true`:

- **Native matches or beats stock on every workload.** In warm cache, native
  beats stock on `scan_high` (201.2 vs 203.3 ms), `count_low` (90.0 vs
  90.8 ms), and `count_high` (88.3 vs 91.7 ms), and is within 1% on `scan_low`
  (214.3 vs 212.2 ms). In page-cache-cold, native beats stock across all four
  workloads: `scan_low` (232.1 vs 239.9 ms), `scan_high` (224.1 vs 232.7 ms),
  `count_low` (93.3 vs 106.2 ms; 12.1% faster), and `count_high` (94.4 vs
  103.5 ms; 8.8% faster).
- **Filtered `COUNT(*)` improves over before** because projection narrowing
  reduces `NativeParquet`'s output projection to `[]` (zero columns), avoiding
  all output array materialization: `count_low` drops from 91.8 → 90.0 ms warm
  (95.9 → 93.3 ms cold) and `count_high` drops from 97.2 → 88.3 ms warm
  (101.8 → 94.4 ms cold).
- **Time to first 8,192 rows on `scan_low`** drops from 33.2 → 22.2 ms warm
  (stock 21.4 ms) and from 39.2 → 25.2 ms cold (stock 25.5 ms).
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
| `scan_low`   | 168.7 (167.1–172.3) | 170.0 (163.9–183.7) | 171.4 (168.1–174.7) | 170.8 (169.8–176.4) |     196,691,567 |
| `scan_high`  | 194.7 (188.3–196.3) | 194.3 (186.5–200.3) | 193.6 (191.0–198.7) | 197.0 (192.7–205.7) |     196,691,567 |
| `count_low`  |    91.7 (87.3–94.4) |   90.3 (87.5–104.4) |    90.1 (87.7–91.6) |    90.0 (88.6–92.5) |      75,280,626 |
| `count_high` |    89.8 (87.8–96.3) |    87.9 (87.1–90.8) |    95.4 (92.6–96.1) |   94.7 (92.4–105.1) |      75,280,626 |

Disabled behavior (`pushdown_filters = false`) is unchanged within measurement
noise.

## #138 wait and buffering metrics (`pushdown_filters = true`)

Medians of per-run totals in ms, before → after (overlapping sums across I/O
lanes or workers). Both Phase-1 (predicate columns) and Phase-2 (surviving
output columns) range reads run on I/O lanes and are timed in `morsel_pending_io`;
pipeline workers run `RowFilter` and output decoding purely synchronously inside
`morsel_decode_work` and never await storage I/O.

| Cache           | Case         |   Pending I/O |   Ready wait |       Decode work | Process work | Output wait | Peak ready depth / bytes   |
| --------------- | ------------ | ------------: | -----------: | ----------------: | -----------: | ----------: | -------------------------- |
| Warm            | `scan_low`   | 237.7 → 245.2 |  18.7 → 55.3 | 1,023.7 → 1,312.0 |  176.5 → 1.8 |       0 → 0 | 8 (19.5 MiB) → 6 (4.9 MiB) |
| Warm            | `scan_high`  | 219.7 → 211.2 |  20.7 → 42.0 | 1,019.4 → 1,233.0 | 227.6 → 23.1 | 47.1 → 16.5 | 8 (19.5 MiB) → 7 (5.8 MiB) |
| Warm            | `count_low`  | 132.1 → 184.3 |  10.8 → 28.7 |     459.7 → 587.9 |  137.1 → 0.1 |       0 → 0 | 8 (19.5 MiB) → 4 (2.6 MiB) |
| Warm            | `count_high` | 126.2 → 172.9 |  12.3 → 26.2 |     460.0 → 587.5 |  167.1 → 0.2 |       0 → 0 | 8 (19.5 MiB) → 4 (2.4 MiB) |
| Page-cache-cold | `scan_low`   | 505.9 → 556.5 | 73.7 → 198.9 | 1,043.9 → 1,288.8 |  185.9 → 1.8 |       0 → 0 | 8 (19.5 MiB) → 4 (4.2 MiB) |
| Page-cache-cold | `scan_high`  | 453.9 → 562.3 | 67.1 → 110.5 | 1,026.0 → 1,237.1 | 233.2 → 23.5 | 49.3 → 18.1 | 8 (19.5 MiB) → 6 (5.3 MiB) |
| Page-cache-cold | `count_low`  | 275.7 → 355.9 |  24.0 → 45.5 |     467.5 → 597.9 |  140.3 → 0.1 |       0 → 0 | 8 (19.5 MiB) → 5 (2.9 MiB) |
| Page-cache-cold | `count_high` | 296.5 → 357.2 |  23.9 → 44.9 |     465.2 → 592.5 |  172.1 → 0.2 |       0 → 0 | 8 (19.5 MiB) → 4 (2.4 MiB) |

Accounting notes:

- `morsel_pending_io` measures **all** Parquet I/O (metadata + Phase-1 predicate
  reads on file lanes + Phase-2 output reads on Phase-2 I/O lanes). Because
  scans split each row group into two range-read phases, summed cold-cache
  `morsel_pending_io` increases slightly (505.9 → 556.5 ms on `scan_low`), while
  overlapping I/O lanes keep end-to-end wall time faster than stock.
- `morsel_process_work` drops from 137–233 ms to 0.1–23.5 ms as predicate
  evaluation moves into Parquet `RowFilter` inside `morsel_decode_work` on the
  pipeline workers.
- Before #150 (`PlainBufferedReaders`), `buffered_bytes_estimate` reported the
  full row-group compressed size (`entry.bytes` ≈ 2.43 MiB, including unprojected
  columns such as `unused`). After #150 (`FilteredBufferedReaders`), queued items
  report the actual fetched compressed bytes held by that item (Phase-1 predicate
  bytes for a `FilterStep`, and Phase-1 + Phase-2 fetched bytes for a decode-only
  `BufferedReader`). Because each row group holds its ready-queue slot permit
  across Phase-1, worker `RowFilter`, and Phase-2 I/O until claimed for decode,
  peak ready-queue depth is 4–7 (vs 8 before) while in-flight Phase-2 reads hold
  the remaining permits.

## Reproduction

Generate the #128 16 × 8 × 131,072 fixture using the pinned Python environment
in [README.md](README.md#native-parquet-ab-128). Copy `stream_runner.rs` to a
temporary example `datafusion/core/examples/morsel_pushdown_bench.rs` in
detached worktrees for the before and after revisions, replacing
`execution.parquet.pushdown_filters = false;` with
`std::env::var("MORSEL_PUSHDOWN_FILTERS").is_ok_and(|v| v == "1")` and attaching
`MorselMetrics` when `MORSEL_METRICS=1`. Build each revision with
`--profile release-nonlto -p datafusion --no-default-features --features parquet,sql --example morsel_pushdown_bench`.

For `MORSEL_PUSHDOWN_FILTERS=0`, run `benchmarks/morsel_parquet/harness.py ab`
directly. For `MORSEL_PUSHDOWN_FILTERS=1`, wrap `harness.validate_path` to
accept `NativeParquet[fixture, predicate=...]` in addition to
`NativeParquet[fixture]` (since #128's pinned `harness.py` checks the literal
string `NativeParquet[fixture]`), then run:

```bash
for cache in warm page-cache-cold; do
  MORSEL_PUSHDOWN_FILTERS=1 MORSEL_METRICS=1 \
    timeout --kill-after=10s 240s taskset -c 0-7 "$py" -B "$work/run_ab.py" ab "$fixture" \
    --runner "$runner" --revision "$revision" --threads 8 --timeout 120 \
    --cache "$cache" --warmups 1 --repeats 8 > "$revision-pushdown1-$cache.jsonl"
done
```
