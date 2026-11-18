<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Morsel Parquet workload: stock baseline

This fork-only harness runs stock DataFusion without the morsel planner or a
native morsel Parquet adapter. It is not a comparison between backends.

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

## Run

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
high), exact projected values including nulls, and the analyzed plans. `verify`
compares output with a NumPy reference independently of result order: it checks
the count scalar and checks every projected row by ID, rejecting missing IDs,
duplicates, extra rows and wrong values. CLI CSV serialization and temporary
storage for these checks are not timed. Verification can use more memory and
scratch space than timing, particularly for a large high-selectivity result.

### Timing and adding another backend (#128)

Every warmup or timed query gets a fresh CLI process/session with the same
fixture registered as `fixture`. The harness clears inherited `DATAFUSION_*`
settings and uses stock defaults except for target partitions, batch size
(8,192), and EXPLAIN format; `--threads` sets Tokio workers and target
partitions. Each round visits scan-low, count-low, scan-high, count-high in
that order. `StockCli.measure()` uses
`EXPLAIN ANALYZE VERBOSE` around the actual SQL. DataFusion's `AnalyzeExec`
drains every input batch, counts the output rows and discards the batches. This
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

For #128, import `workloads()`, `verify()` and `bench()` from `harness.py`.
Supply a backend with `query(sql, output=...)` for verification (return CSV
bytes or write them to the supplied binary file; include a header, use empty
fields for nulls, and emit no bytes for an empty result). Implement
`measure(sql)` to return a `Measurement`. Use the same fixture and drain all
result batches with equivalent work; keep CSV and parity validation separate
from timing. If the public API in #128 permits first-batch measurement, use it
on **both** backends and define the start and end boundaries identically.

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
759,841 (low, 4.53%) and 9,883,940 (high, 58.91%). Compare both backends on
one quiet host in #128 rather than using these timings as the stock half of
that comparison.
