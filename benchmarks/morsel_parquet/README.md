<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Morsel Parquet benchmark

This fork-only harness compares stock and native morsel execution on the same
Parquet fixture and SQL (#128). The original stock CLI baseline (#130) remains
available below; its timing boundary is different.

## Fixture and SQL

The generator writes numbered Parquet files with contiguous `row_id` ranges and
exactly `--groups` row groups of `--group-rows` rows in each file. The smoke
shape is **2 files × 3 groups × 1,000 rows = 6,000 rows** (120,358 bytes on
`dduck2`). The larger baseline shape is **16 × 8 × 131,072 = 16,777,216 rows**
(311,529,117 bytes). Adjust all three dimensions with `generate`; the output
directory must be new. Keep both generated data and run output out of Git.

Values are a pure function of `row_id` and seed `0x5EED`, using a SplitMix64
mixer with unsigned 64-bit wraparound. Layout changes do not change any value
for a given `row_id`. The writer uses Parquet 2.6, data pages v1, Zstandard
level 3, no dictionary encoding, and column statistics. The pinned Python
packages and same platform are used for byte-for-byte fixture comparisons.

| Column    | Parquet logical value | Distribution                                          |
| --------- | --------------------- | ----------------------------------------------------- |
| `row_id`  | INT64, required       | Sequential, globally unique, starts at zero           |
| `score`   | INT32, nullable       | 15 mixed bits; null when `row_id % 19 = 0`            |
| `delta`   | INT32, nullable       | Another 15 mixed bits; null when `row_id % 23 = 0`    |
| `payload` | INT64, nullable       | 48 mixed bits with a separate salt; null every 11 IDs |
| `unused`  | INT64, required       | 48 mixed bits with a separate salt; never selected    |

`workloads()` returns the same four SQL queries for any backend. Both
`SELECT row_id, payload + delta AS projected` (no `ORDER BY`) and
`SELECT COUNT(*) AS n` use:

```sql
FROM fixture WHERE (score + 3 * delta) % 1000 < threshold
```

Thresholds are 50 (low) and 650 (high), yielding about 4.5% and 59% matches
once nulls are excluded. The mixed, two-column modulo expression needs row
values; row-group min/max or footer row counts cannot answer it. The payload
is used only for the projection and can be skipped by the filtered count;
`unused` can be skipped in all four cases.

## Native Parquet A/B (#128)

Build one release-mode executable for both arms. Pin the first comparison to
`d649cacf00fee7125f45ea1427a023b2eab732b1` (the merge of PR #143 for #126),
before the optimizations in #145 or metrics changes in #138. Keep that checkout
and its binary separate from any later-revision comparison. The recipe copies a
benchmark-only Cargo example into the detached checkout so it can reuse the
pinned workspace's `Cargo.lock`, features, profiles and toolchain without
changing any manifests.

From the harness checkout on a Linux remote workspace with `uv` and `taskset`:

```bash
set -e
b=benchmarks/morsel_parquet
work=$(mktemp -d "$HOME/.cache/issue128.XXXXXX")
revision=d649cacf00fee7125f45ea1427a023b2eab732b1

git worktree add --detach "$work/pinned" "$revision"
mkdir -p "$work/pinned/datafusion/core/examples"
cp "$b/stream_runner.rs" "$work/pinned/datafusion/core/examples/morsel_parquet_stream.rs"
(
  cd "$work/pinned"
  MORSEL_BENCH_REVISION="$(git rev-parse HEAD)" CARGO_TARGET_DIR="$work/target" \
    timeout --kill-after=20s 35m cargo build --locked --profile release-nonlto -j 10 \
      -p datafusion --no-default-features --features parquet,sql --example morsel_parquet_stream
)
runner="$work/target/release-nonlto/examples/morsel_parquet_stream"
timeout 120s uv venv --python 3.12.10 "$work/venv"
timeout 180s uv pip install --python "$work/venv/bin/python" -r "$b/requirements.txt"
py="$work/venv/bin/python"

MORSEL_PARQUET_RUNNER="$runner" MORSEL_BENCH_REVISION="$revision" \
  timeout 120s "$py" -B -m unittest discover -s "$b" -v

# Choose a private fixture directory on the device under test, independent of
# the build location. On dduck1 this root is on Btrfs on local instance-store.
fixture_root=$(mktemp -d "${MORSEL_PARQUET_FIXTURE_ROOT:-$HOME/.cache}/morsel-parquet-fixture.XXXXXX")
fixture="$fixture_root/large"
findmnt -T "$fixture_root" -o TARGET,SOURCE,FSTYPE,OPTIONS
lsblk -o NAME,TYPE,MODEL,PKNAME
timeout 60s "$py" -B "$b/harness.py" generate "$fixture" \
  --files 16 --groups 8 --group-rows 131072
# Adapt the CPU list to your host. Run on an otherwise idle workspace.
for cache in warm page-cache-cold; do
  timeout --kill-after=10s 240s taskset -c 0-7 "$py" -B "$b/harness.py" ab "$fixture" \
    --runner "$runner" --revision "$revision" --threads 8 --timeout 120 \
    --cache "$cache" --warmups 1 --repeats 8 > "$work/$cache.jsonl"
done
```

### A/B controls and timing

- `ab` uses `morsel_mode=off` and `morsel_mode=required` explicitly. Both arms
  must match the same exact NumPy oracle (including nullable projected values,
  duplicate/missing ID checks, and both count scalars) before timing. The first
  NDJSON record contains untimed plan evidence for all eight arm/query pairs.
  Native runs must show `NativeParquet[fixture]` under `MorselExec`, claim all
  fixture row groups, and show `CountSink` plus one count finalization for count
  queries only. The stock runs must show a stock Parquet `DataSourceExec`.
  Each trial rechecks the engine revision, path and drained row count.
- Every invocation starts a fresh process, runtime and session. Both arms use
  the same registration SQL and 8,192-row batch size. `--threads` sets target
  partitions, Tokio workers and the Tokio blocking-thread limit (all eight in
  this run); all threads inherit the same eight-CPU affinity. File splitting,
  listing-statistics collection, row-group statistics pruning, row filter
  pushdown, page-index and Bloom pruning are disabled in both arms. The scan
  needs `row_id`, `score`, `delta`, `payload`; count needs only `score`, `delta`.
  Other session options use defaults, not environment overrides.
- The clock starts just before `CREATE EXTERNAL TABLE`, including file listing,
  schema inference and SQL/physical planning (`plan_seconds`). It ends at
  stream exhaustion. `first_output_seconds` is the time to the first nonempty
  result batch; `first_8192_seconds` is the time to at least 8,192 cumulative
  result rows (`null` if the result is smaller). `output_batches` counts
  nonempty batches. Both arms use `execute_stream` and drain all batches,
  with no CSV or collection during timing. Process/runtime/session
  startup, validation, plan/metrics formatting and teardown are not timed.
  Throughput is **logical input rows / total seconds**, not storage bandwidth.
  Each workload pairs the two arms and alternates which arm goes first by round.
- The same configured batch size does not imply the same result batch size.
  Stock `FilterExec` uses a `LimitedBatchCoalescer`: it combines small filtered
  batches toward 8,192 rows, and batches above 4,096 rows can bypass coalescing.
  Native `MorselExec` emits nonempty filtered/projected batches without
  coalescing. Compare the time to 8,192 cumulative rows for equivalent output
  volume rather than treating first-batch latency alone as a speedup.
- `warm` reads the entire private fixture into the Linux file page cache and
  requires 100% residency via `mincore` before each timed arm. `page-cache-cold`
  is for **exclusively used, locally generated fixture files only**. The driver
  syncs those files and requests `POSIX_FADV_DONTNEED` on each before each timed
  arm. It requires zero resident file pages before launching the runner. Cache
  reset and sampling are outside the clock; registration's own footer reads
  occur inside it. Both regimes log residency again after execution. If a
  residency check fails, the driver stops; do not treat partial/unverified
  output as a completed regime. Neither regime resets global/shared caches or
  controls filesystem metadata, hypervisor or storage-device caches. Cold here
  means **Linux file pages only**, not cold physical storage.

### Pinned A/B results: `rob-tandy-dduck1`, 2026-11-19 UTC

- Ubuntu 24.04.5 LTS, Linux `7.0.0-1014-aws`, 16 ARM Neoverse-N1 vCPUs
  (one thread per core), 61 GiB RAM. The run used CPUs 0–7 as above and no
  builds were active. The workspace was not reserved; one-second
  `vmstat -y 1 3` samples just before launching each regime showed 99–100%
  host CPU idle.
- Both modes used the one locked `release-nonlto` build from the pinned
  `d649cacf00fee7125f45ea1427a023b2eab732b1` checkout. Rust 1.99.1 (`58cf8f9c3`,
  `aarch64-unknown-linux-gnu`), Cargo 1.99.1, Clang/lld 18.1.3, Tokio 1.54.1,
  Python 3.12.10, and the pinned NumPy and PyArrow versions. The 16-file,
  128-row-group fixture was 311,529,117 bytes (Parquet Zstandard level 3).
  It was generated under `$HOME/.cache` on the local EC2 NVMe instance-store
  LVM volume `/dev/mapper/vg_ws_ephemeral-volume1p1` (Btrfs, mounted with
  `compress=zstd:1`), not on the EBS-backed overlay root used by `/var/tmp`.
- All 64 warm samples started and ended with 76,063/76,063 file pages
  resident (4 KiB). All 64 page-cache-cold samples started with 0/76,063
  resident; 20,556–54,559 were resident afterward. Device cache state was
  not measured or controlled.
- Both arms matched all 759,841 low-selectivity and 9,883,940 high-selectivity
  projected rows and count scalars. The stock source had eight groups of two
  whole files, with a residual filter and partial/final aggregates for count.
  In the native arm `MorselExec: workers=8, NativeParquet[fixture]` claimed all
  128 row groups in every case, with `CountSink; FinalizedCountSource` and
  `count_finalizations=1` for count (zero for scans). Both arms reported
  `bytes_scanned=196691567` for scans and `75280626` for counts at both
  selectivities. These are Parquet source metrics, not device I/O counters.
- In the pinned implementation, stock scans stream file groups per partition
  and merge outputs, while the morsel source publishes buffered row-group
  readers using up to four file lanes, two pending I/O slots and a ready queue
  of two readers. The native result channel has capacity two; the native count
  sink combines worker-local counts before emitting its single row. The same
  thread budgets and scan options do not imply the same scheduling or I/O
  overlap.

One warmup and eight timed samples per case and mode in each regime. Batch
counts were constant within each case and mode. Timings and throughput are
medians, with observed minimum and maximum times in parentheses. `—` means
fewer than 8,192 rows were returned. Registration/planning is included and
took 5.6–11.5 ms across all samples.

| Cache           | Case         | Mode       | Batches | Input Mrows/s | First batch ms (min–max) | First 8,192 rows ms (min–max) |  Total ms (min–max) |
| --------------- | ------------ | ---------- | ------: | ------------: | -----------------------: | ----------------------------: | ------------------: |
| Warm            | `scan_low`   | `off`      |      96 |          99.1 |         30.8 (30.1–32.4) |              30.8 (30.1–32.4) | 169.3 (167.1–201.3) |
| Warm            | `scan_low`   | `required` |   2,048 |          27.2 |         14.8 (14.4–17.6) |              29.7 (29.3–36.0) | 617.6 (614.4–664.3) |
| Warm            | `count_low`  | `off`      |       1 |         182.1 |        92.1 (89.7–110.7) |                             — |   92.1 (89.7–110.7) |
| Warm            | `count_low`  | `required` |       1 |          67.9 |      246.9 (242.4–256.6) |                             — | 246.9 (242.4–256.6) |
| Warm            | `scan_high`  | `off`      |   2,048 |          86.3 |         16.9 (15.8–19.6) |              16.9 (15.8–19.6) | 194.3 (191.4–237.0) |
| Warm            | `scan_high`  | `required` |   2,048 |          27.0 |         15.0 (14.4–16.3) |              21.7 (20.9–23.1) | 621.9 (616.9–665.5) |
| Warm            | `count_high` | `off`      |       1 |         181.5 |        92.4 (88.1–112.7) |                             — |   92.4 (88.1–112.7) |
| Warm            | `count_high` | `required` |       1 |          68.7 |      244.3 (241.2–263.9) |                             — | 244.3 (241.3–264.0) |
| Page-cache-cold | `scan_low`   | `off`      |      96 |          82.2 |         38.9 (33.3–40.7) |              38.9 (33.3–40.7) | 204.1 (196.0–210.3) |
| Page-cache-cold | `scan_low`   | `required` |   2,048 |          26.9 |         20.0 (18.0–21.4) |              35.0 (33.0–36.9) | 624.2 (619.6–638.0) |
| Page-cache-cold | `count_low`  | `off`      |       1 |         158.8 |      105.7 (102.6–114.7) |                             — | 105.7 (102.6–114.7) |
| Page-cache-cold | `count_low`  | `required` |       1 |          66.7 |      251.5 (245.1–255.8) |                             — | 251.5 (245.1–255.8) |
| Page-cache-cold | `scan_high`  | `off`      |   2,048 |          75.1 |         23.9 (19.7–26.7) |              24.0 (20.0–26.7) | 223.5 (218.9–235.5) |
| Page-cache-cold | `scan_high`  | `required` |   2,048 |          26.7 |         19.2 (18.6–21.7) |              25.9 (25.2–28.5) | 628.3 (624.3–640.3) |
| Page-cache-cold | `count_high` | `off`      |       1 |         152.2 |      110.2 (102.1–114.7) |                             — | 110.2 (102.1–114.8) |
| Page-cache-cold | `count_high` | `required` |       1 |          67.3 |      249.2 (245.0–256.5) |                             — | 249.2 (245.1–256.6) |

On instance-store, the low-selectivity scan's earlier native first batch
mostly reflects batching, not faster delivery of the same number of rows.
Stock emitted 96 batches averaging about 7,915 rows; native emitted 2,048
averaging 371 rows.
Warm, the first-batch medians were 30.8 ms stock and 14.8 ms native, but both
reached 8,192 rows at about 30 ms. Page-cache-cold, the median difference
shrank from 18.9 ms at the first batch to 3.9 ms at 8,192 rows, with overlapping
ranges. For the high-selectivity scan, the native first batch was a few
milliseconds earlier, but native reached 8,192 rows later in both regimes.

Native scans took 3.20–3.65× as long to drain warm and 2.81–3.06× as long
page-cache-cold. Both counts regressed in first output and total time, with
native total medians of 2.64–2.68× stock warm and 2.26–2.38× page-cache-cold.
The page-cache-cold ratios (2.36–3.12× in the original run, 2.26–3.06× here)
apply only to this local instance-store configuration. This comparison
measures the whole path; it does not isolate the cost of buffering, decoding
or the scheduler, and is not a prediction for later revisions.

A separate storage check used the same pinned binary and recipe with
`MORSEL_PARQUET_FIXTURE_ROOT=/var/tmp`, only `page-cache-cold`, and `--repeats 3`.
The fixtures were on the EBS-backed overlay root, and all 24 timed samples
started with zero resident file pages. Median total times were 1.87–1.89 s
for scans and 0.90–0.92 s for counts; native total medians were 0.99–1.03×
stock. The native scans reached 8,192 rows earlier in this check (35–36 ms
versus 85–203 ms), but had similar total runtimes. Neither check establishes
cold physical-device caches.

## Stock-only run (#130)

From the repository root on a Linux host with `uv`, `taskset` and the pinned Rust
toolchain (use CPUs available to your process):

```bash
work=$(mktemp -d /var/tmp/morsel-parquet-130.XXXXXX)
revision=$(git rev-parse HEAD)
b=benchmarks/morsel_parquet
timeout 120s uv venv "$work/venv"
timeout 180s uv pip install --python "$work/venv/bin/python" -r "$b/requirements.txt"
py="$work/venv/bin/python"

CARGO_TARGET_DIR="$work/target" RUSTC_WRAPPER=sccache \
  timeout 35m cargo build --locked --profile release-nonlto -p datafusion-cli
cli="$work/target/release-nonlto/datafusion-cli"
# Omit RUSTC_WRAPPER if sccache is not installed.

MORSEL_STOCK_CLI="$cli" timeout 120s "$py" -B -m unittest discover -s "$b" -v

timeout 240s "$py" -B "$b/harness.py" generate "$work/large" \
  --files 16 --groups 8 --group-rows 131072
timeout 240s taskset -c 0-7 "$py" -B "$b/harness.py" bench "$work/large" \
  --cli "$cli" --revision "$revision" --threads 8 --warmups 1 --repeats 5 \
  > "$work/stock.jsonl"
timeout 240s taskset -c 0-7 "$py" -B "$b/harness.py" verify "$work/large" \
  --cli "$cli" --threads 8
```

The unittest smoke suite generates the default shape twice and compares the
bytes and layout. With `MORSEL_STOCK_CLI` it also checks counts (279 low, 3,506
high), exact projected values including nulls, and the analyzed plans. It also
runs a timed round with cache sampling on a 24-file fixture whose verbose
plan contains long file paths. `verify` compares the count scalar and
projected rows against a NumPy reference without relying on output order. It
checks each row by ID, rejecting missing IDs, duplicates, extra rows and wrong
values. CLI CSV serialization and temporary
storage for these checks are not timed. Verification can use more memory and
scratch space than timing, particularly for a large high-selectivity result.

### Stock CLI timing

Every warmup or timed query gets a fresh CLI process/session with the same
fixture registered as `fixture`. The harness clears inherited `DATAFUSION_*`
settings and uses stock defaults except for target partitions, batch size
(8,192), and EXPLAIN format; `--threads` sets Tokio workers and target
partitions. Each round visits scan-low, count-low, scan-high, count-high in
that order. `StockCli.measure()` uses
`EXPLAIN ANALYZE VERBOSE` around the actual SQL and reads its newline-delimited
JSON output to handle long plan fields. DataFusion's `AnalyzeExec` drains
every input batch, counts the output rows and discards the batches. This
keeps the projection in the executed plan instead of rewriting the scan as a
count, and avoids serializing or collecting its rows at the CLI boundary.

`seconds` is the reported `AnalyzeExec` duration (partition execution/drain;
the clock starts just after dispatching the input tasks). It excludes CLI
startup, table registration, query planning and plan rendering. The CLI does
not expose the first data batch for this bounded query, so
`first_output_seconds` is null, not the time to print the plan. Throughput is
logical input rows divided by that duration, not physical bytes read or
result rows per second. Each timed query samples fixture file page residency
before and after using Linux `mincore`; this does not prove what storage I/O
occurred. No cache is flushed or evicted. `--warmups 0` skips warmup but does
not imply any pages are cold (generation and metadata reads can warm them).

## Stock evidence: `dduck2`, 2026-11-18 UTC

- `datafusion-cli 57.0.0-dev`, optimized Cargo `release-nonlto` profile, built
  from `4cab1b2224c171b7bc6bd72325e9ddc17fab923c` with Rust 1.99.1
  (`58cf8f9c3`); Python 3.12.10.
- Linux `7.0.0-1012-aws`, aarch64 Neoverse-N1, 16 logical CPUs at one thread
  per core, 61 GiB RAM. CLI and driver bound to CPUs 0–7, eight Tokio workers
  and eight target partitions. No local builds were running during the
  reported run; one-second `vmstat` samples before and after showed 90% and
  96% host CPU idle, respectively. The host was not reserved for the benchmark.
- Larger fixture and commands above: one warmup and five measured runs per
  case. All 76,063 file pages (4 KiB) were resident before and after every
  timed query. This is a warm-cache smoke run, not a cold-cache measurement.

| Case         | Drained rows | Median seconds | Min–max seconds | Median million input rows/s |
| ------------ | -----------: | -------------: | --------------: | --------------------------: |
| `scan_low`   |      759,841 |          0.154 |     0.152–0.158 |                       109.0 |
| `count_low`  |            1 |          0.085 |     0.084–0.088 |                       196.9 |
| `scan_high`  |    9,883,940 |          0.171 |     0.170–0.177 |                        98.1 |
| `count_high` |            1 |          0.085 |     0.085–0.087 |                       196.3 |

Untimed verification matched all projected rows and both `COUNT(*)` values:
759,841 (low, 4.53%) and 9,883,940 (high, 58.91%). Do not use these timings
as the stock half of the A/B comparison, which uses a different timing boundary
and build.
