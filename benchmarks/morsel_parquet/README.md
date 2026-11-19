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

timeout 60s "$py" -B "$b/harness.py" generate "$work/large" \
  --files 16 --groups 8 --group-rows 131072
# Adapt the CPU list to your host. Run on an otherwise idle workspace.
for cache in warm page-cache-cold; do
  timeout --kill-after=10s 240s taskset -c 0-7 "$py" -B "$b/harness.py" ab "$work/large" \
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
  stream exhaustion. First output is the first **nonempty** result batch from
  the same clock. Both arms use `execute_stream` and count/drain the result
  batches, with no CSV or collection during timing. Process/runtime/session
  startup, validation, plan/metrics formatting and teardown are not timed.
  Throughput is **logical input rows / total seconds**, not storage bandwidth.
  Each workload pairs the two arms and alternates which arm goes first by round.
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
  (one thread per core), 61 GiB RAM. The run used CPUs 0–7 as above. The
  workspace was not reserved, but no builds were active during the run; the
  local Clippy job was suspended. One-second `vmstat -y 1 3` samples showed
  78–90% host CPU idle just before warm verification, 93–100% between the two
  regimes, and 95–99% afterward.
- Both modes used the one locked `release-nonlto` build from the pinned
  `d649cacf00fee7125f45ea1427a023b2eab732b1` checkout. Rust 1.99.1 (`58cf8f9c3`,
  `aarch64-unknown-linux-gnu`), Cargo 1.99.1, Clang/lld 18.1.3, Tokio 1.54.1,
  Python 3.12.10, and the pinned NumPy and PyArrow versions. The 16-file,
  128-row-group fixture was 311,529,117 bytes on the local Btrfs workspace
  volume (Zstandard level 3).
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

One warmup and eight timed samples per case and mode in each regime. The
entries are medians; parentheses give the observed minimum and maximum for
first output and total runtime. Registration/planning is included and took
5.7–8.9 ms across all samples.

| Cache           | Case         | Mode       | Input Mrows/s |  First ms (min–max) |  Total ms (min–max) |
| --------------- | ------------ | ---------- | ------------: | ------------------: | ------------------: |
| Warm            | `scan_low`   | `off`      |          99.5 |    31.3 (30.7–32.8) | 168.6 (163.6–176.8) |
| Warm            | `scan_low`   | `required` |          27.3 |    15.0 (14.4–15.8) | 615.5 (607.3–627.9) |
| Warm            | `count_low`  | `off`      |         185.1 |    90.7 (87.8–99.9) |    90.7 (87.8–99.9) |
| Warm            | `count_low`  | `required` |          68.4 | 245.3 (239.7–251.8) | 245.4 (239.8–251.8) |
| Warm            | `scan_high`  | `off`      |          86.4 |    16.9 (15.9–17.7) | 194.3 (188.1–211.9) |
| Warm            | `scan_high`  | `required` |          27.1 |    14.6 (14.3–15.5) | 618.0 (611.2–634.6) |
| Warm            | `count_high` | `off`      |         184.1 |    91.1 (87.4–97.0) |    91.1 (87.4–97.0) |
| Warm            | `count_high` | `required` |          68.8 | 244.0 (238.7–252.7) | 244.0 (238.8–252.7) |
| Page-cache-cold | `scan_low`   | `off`      |          84.5 |    36.1 (31.3–38.5) | 198.5 (195.2–200.2) |
| Page-cache-cold | `scan_low`   | `required` |          27.1 |    19.1 (18.3–19.6) | 619.8 (613.8–624.0) |
| Page-cache-cold | `count_low`  | `off`      |         159.7 |  105.1 (99.6–113.6) |  105.1 (99.7–113.6) |
| Page-cache-cold | `count_low`  | `required` |          67.7 | 247.6 (243.2–249.0) | 247.7 (243.2–249.1) |
| Page-cache-cold | `scan_high`  | `off`      |          76.8 |    21.8 (19.2–23.8) | 218.6 (211.4–229.0) |
| Page-cache-cold | `scan_high`  | `required` |          27.0 |    19.1 (18.3–19.5) | 621.4 (615.8–624.8) |
| Page-cache-cold | `count_high` | `off`      |         160.1 | 104.8 (103.8–108.8) | 104.8 (103.8–108.8) |
| Page-cache-cold | `count_high` | `required` |          67.9 | 247.0 (244.6–248.7) | 247.1 (244.6–248.8) |

Native scans reached their first result earlier in both regimes, but took
3.18–3.65× as long to drain warm and 2.84–3.12× as long page-cache-cold.
Both counts regressed in first output and total time; native total medians
were 2.68–2.71× stock warm and 2.36× stock page-cache-cold. The comparison
measures the whole path; it does not isolate the cost of buffering, decoding
or the scheduler, and is not a prediction for later revisions.

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
