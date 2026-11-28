<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Morsel ClickBench 4-Way Parquet Benchmark Suite

This fork-only benchmark suite compares **Morsel DataFusion**
(`morsel_mode=required`), **Stock DataFusion** (`stock_default` and
`stock_pushdown`), **DuckDB 2.0.1**, and **ClickHouse (`clickhouse local`)**
across all **43 official ClickBench queries**
(`benchmarks/queries/clickbench/queries/q0.sql`..`q42.sql`) over identical
Parquet files (`#230`). It lives entirely under `benchmarks/morsel_clickbench/`
and reuses the detached-example build pattern from `benchmarks/morsel_tpch/`
without modifying any workspace `Cargo.toml` or production crate.

## Engine Setup and Dataset Modes

Keep all downloaded Parquet datasets, ClickHouse binaries, Python virtual
environments, and `.jsonl` run outputs outside Git (e.g., under `$HOME/.cache/`
or `/tmp/`).

### ClickHouse (`clickhouse local`) Setup

`harness.py install-clickhouse` downloads and installs the pinned official
ClickHouse static release (`v26.11.5.8-stable`, Linux `arm64`/`aarch64` or
`amd64`/`x86_64`) from GitHub Releases, verifies its official `.sha512`
checksum, extracts `usr/bin/clickhouse` to
`$HOME/.cache/morsel-clickbench/bin/clickhouse` (or `--dest`), and checks
`clickhouse local --version`:

```bash
python3 -B benchmarks/morsel_clickbench/harness.py install-clickhouse \
  --dest "$HOME/.cache/morsel-clickbench/bin/clickhouse"
```

Every `clickhouse local` invocation passes `--tmp` (a fresh ephemeral storage
directory so no `default.hits` catalog state leaks across trials or into
`$HOME`), `--max_threads=<N>`, and reads the Parquet files natively via
`file('<glob_or_file>', Parquet)` through the `hits` view.

### Datasets (`download` and `generate-smoke`)

- **Official ClickBench partitioned Parquet (`download`):** Reuses the official
  URLs from `benchmarks/bench.sh`
  (`https://datasets.clickhouse.com/hits_compatible/athena_partitioned/hits_{0..99}.parquet`,
  or `--single` for `hits.parquet`). Use `--count N` (default `100`) to download
  the full 100-file dataset (~99.997M rows, 14.74 GB) or a deterministic prefix
  subset (`hits_0.parquet`..`hits_{N-1}.parquet`), verifying every Parquet
  footer before moving files into place.
- **Deterministic smoke fixture (`generate-smoke`):** Generates a fast,
  byte-for-byte deterministic Parquet dataset (default `50,000` rows across 4
  files `hits_0.parquet`..`hits_3.parquet`, `12,500`-row row groups, Zstd level 3) matching the **exact 105-column ClickBench `hits_partitioned` schema**
  (including `binary` string columns and `uint16` `EventDate`). The 40 queries
  whose thresholds fit 50K rows return non-empty results directly, while `q27`,
  `q28` (`HAVING COUNT(*) > 100000`), and `q41` (`OFFSET 10000`) return 0 rows
  at the official 100M-row thresholds; `test_harness.py` additionally runs
  fixture-sized variants (`HAVING COUNT(*) > 100` for `q27`/`q28` and `OFFSET
100` for `q41`) through all 5 arms (plus a negative ClickHouse `q28` test
  without `(?-s)`) so every dialect adaptation is exercised on non-empty rows.

## Controlled Arms, Timing Boundaries, and Parity

Each timed arm runs in a **fresh process** per query repetition, inheriting the
same CPU affinity (`taskset -c 0-7`) and 8-thread budget (`target_partitions=8`,
Tokio `worker_threads=8` and `max_blocking_threads=8` for DataFusion;
`SET threads = 8` for DuckDB; `--max_threads=8` for ClickHouse). Arm execution
order alternates between odd and even rounds.

### Arms

| Arm                  | Engine     | `morsel_mode` | Key settings                                                                                                                                                          |
| -------------------- | ---------- | ------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `morsel_pushdown`    | DataFusion | `required`    | `binary_as_string=true`, `pushdown_filters=true`, `reorder_filters=true`, `repartition_file_scans=false`, `pruning=true`, `enable_page_index=true`, `batch_size=8192` |
| `stock_default`      | DataFusion | `off`         | DataFusion defaults + `binary_as_string=true`, `batch_size=8192` (`pushdown_filters=false`, `repartition_file_scans=true`)                                            |
| `stock_pushdown`     | DataFusion | `off`         | DataFusion defaults + `binary_as_string=true`, `pushdown_filters=true`, `reorder_filters=true`, `batch_size=8192`                                                     |
| `duckdb_default`     | DuckDB     | —             | `SET threads = 8`, `SET TimeZone = 'UTC'`, `read_parquet(..., binary_as_string=true)`                                                                                 |
| `clickhouse_default` | ClickHouse | —             | `clickhouse local --tmp --max_threads=8 --format=ArrowStream --time`, `file(..., Parquet)`                                                                            |

### `hits` View Registration and Exact Timing Boundary

ClickBench stores `EventDate` as `UInt16` days since `1970-01-01` and string
columns as unannotated Parquet `BYTE_ARRAY` (`binary`). Every engine registers
the same `hits` view transformation (`EventDate` projected at the end via
`EXCEPT`/`EXCLUDE` so `q23`'s `SELECT *` has identical 105-column ordering
across all 4 engines):

- **DataFusion (`clickbench_runner.rs`)**: Excludes OS process startup and Tokio
  runtime creation. Starts the clock immediately before `register_parquet` +
  `CREATE VIEW hits` (`register_seconds`), runs SQL/physical planning
  (`plan_seconds`), and drains `execute_stream` at batch size `8,192` without
  output serialization (`exec_seconds`). Total reported `seconds` =
  `register_seconds + plan_seconds + exec_seconds`.
- **DuckDB (`harness.py _duckdb-exec`)**: Excludes Python startup, `LOAD
parquet`, and `SELECT 1` Arrow C-stream warmup. Starts the clock immediately
  before `CREATE VIEW hits` (`register_seconds`), then drains
  `con.sql(sql).to_arrow_reader(batch_size=8192)` (`exec_seconds`). Total
  reported `seconds` = `register_seconds + exec_seconds`.
- **ClickHouse (`ClickHouseRunner`)**: Excludes `clickhouse local` ELF binary
  startup (~110 ms) by using ClickHouse's built-in `--time` per-statement timer
  while draining results via `--format=ArrowStream` (so every timed trial
  verifies `output_rows` losslessly): the first `--time` value is `CREATE VIEW
hits` (`register_seconds`, Parquet footer/schema binding) and the second is
  the `SELECT` statement (`exec_seconds`). Total reported `seconds` =
  `register_seconds + exec_seconds`. Because each trial runs in a fresh process,
  ClickHouse `--time` includes ~15 ms of first-statement warm-up (repeating
  statements in a single `clickhouse local` process on the 20-file dataset drops
  `CREATE VIEW` from ~15 ms to ~6 ms and `q0` `SELECT COUNT(*)` from ~27 ms to
  ~15 ms). Process wall time is also recorded in the JSONL as
  `process_wall_seconds`.

### Per-Query SQL Dialect Adaptations

DataFusion executes all 43 official SQL files from
`benchmarks/queries/clickbench/queries/q0.sql`..`q42.sql` verbatim. Only the
following explicitly documented dialect adaptations are applied for DuckDB and
ClickHouse so all 4 engines compute identical semantics:

| Query | Engine     | Adaptation & Reason                                                                                                                                                                                                       |
| ----- | ---------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `q3`  | ClickHouse | `AVG("UserID")` -> `AVG(toFloat64("UserID"))`: ClickHouse `avg(Int64)` accumulates in 64-bit integer and silently overflows on 64-bit `UserID`s; `Float64` matches DataFusion/DuckDB.                                     |
| `q18` | DuckDB     | `to_timestamp_seconds("EventTime")` -> `to_timestamp("EventTime")` with `SET TimeZone = 'UTC'`.                                                                                                                           |
| `q18` | ClickHouse | `to_timestamp_seconds("EventTime")` -> `toDateTime("EventTime", 'UTC')`.                                                                                                                                                  |
| `q27` | DuckDB     | `octet_length("URL")` -> `strlen("URL")`: DuckDB `octet_length` requires `BLOB`; `strlen(VARCHAR)` returns UTF-8 byte length matching DataFusion/ClickHouse.                                                              |
| `q28` | DuckDB     | `octet_length("Referer")` -> `strlen("Referer")` (same UTF-8 byte-length semantics as `q27`).                                                                                                                             |
| `q28` | ClickHouse | `REGEXP_REPLACE` pattern prefixed with `(?-s)`: ClickHouse RE2 enables `dot_nl=true` by default (`.` matches `\n`), whereas DataFusion Rust `regex` and DuckDB RE2 default to `dot_nl=false`.                             |
| `q42` | DuckDB     | `to_timestamp_seconds("EventTime")` -> `CAST(to_timestamp("EventTime") AS TIMESTAMP)` with `SET TimeZone = 'UTC'` (timezone-naive UTC matching DataFusion).                                                               |
| `q42` | ClickHouse | `to_timestamp_seconds("EventTime")` -> `toDateTime64("EventTime", 0, 'UTC')`: ClickHouse `ArrowStream` encodes `DateTime` as `uint32` epoch seconds, whereas `DateTime64(0, 'UTC')` exports Arrow `timestamp[s, tz=UTC]`. |

### Lossless Arrow Parity and Warm Page Cache (`mincore`)

- Before any timing begins, `bench` (and `verify`) transfers full results via
  lossless Arrow IPC streams (`StreamWriter` in Rust, `to_arrow_table()` in
  DuckDB, `--format=ArrowStream` in ClickHouse) and verifies parity against
  DuckDB across all 43 queries:
  - Exact ordered equality for single-row aggregates and deterministic sorts
    (`q0`–`q6`, `q20`, `q25`, `q26`, `q29`, `q42`).
  - Exact multiset equality for unordered queries (`q19`) and equal-key tie
    groups without `LIMIT` (`q7`).
  - For `ORDER BY ... LIMIT K [OFFSET O]` queries (`q8`–`q16`, `q18`,
    `q21`–`q23`, `q27`–`q28`, `q30`–`q41`): exact sort-key sequence matching at
    every position, unique group keys for `GROUP BY` queries, and exact multiset
    equality for all interior tie groups as well as any final group when
    `num_rows < LIMIT`. When a tie straddles an actual `LIMIT` (`num_rows ==
LIMIT`) or `OFFSET` boundary and differs from the oracle's arbitrary tie
    choice, every boundary row is verified against DuckDB's full un-limited
    relation (exact group existence + all aggregate columns for `GROUP BY`
    queries, or exact full-row multiset containment for `q23`).
  - For `q17` (`GROUP BY ... LIMIT 10`, no `ORDER BY`): unique `(UserID,
SearchPhrase)` keys, with every returned group verified for existence and
    exact `COUNT(*)` against DuckDB's full `GROUP BY`.
  - For `q24` (`SELECT "SearchPhrase" ... ORDER BY "EventTime" LIMIT 10`):
    verified against a DuckDB oracle query that also projects `"EventTime"`,
    requiring every row with `"EventTime"` strictly below the 10th value to
    appear in ascending `EventTime` group order and every remaining row to come
    from the exact `EventTime == 10th` multiset.
- Under `--cache warm`, `warm_files` reads every Parquet file twice (promoting
  pages to Linux's `active_file` LRU list), and `require_warm_cache` checks via
  Linux `mincore` that **100% of 4 KiB pages** are resident **before and after**
  every single measured trial. Any residency drop aborts the run.

## Reproduction Commands

From the repository root on a Linux host with `uv`, `taskset`, and the pinned
Rust toolchain:

```bash
set -euo pipefail
b=benchmarks/morsel_clickbench
work=$(mktemp -d "/tmp/morsel-clickbench.XXXXXX")
cache_root="${MORSEL_CLICKBENCH_CACHE:-$HOME/.cache/morsel-clickbench}"
revision=$(git rev-parse HEAD)

# 1. Build the standalone runner as a detached Cargo example (no workspace edits).
git worktree add --detach "$work/pinned" "$revision"
mkdir -p "$work/pinned/datafusion/core/examples"
cp "$b/clickbench_runner.rs" "$work/pinned/datafusion/core/examples/morsel_clickbench_runner.rs"
(
  cd "$work/pinned"
  MORSEL_BENCH_REVISION="$revision" CARGO_TARGET_DIR="$work/target" \
    RUSTC_WRAPPER=sccache \
    timeout --kill-after=20s 25m cargo build --locked --profile release-nonlto -j 10 \
      -p datafusion --no-default-features \
      --features parquet,sql,datetime_expressions,string_expressions,unicode_expressions,regex_expressions \
      --example morsel_clickbench_runner
)
runner="$work/target/release-nonlto/examples/morsel_clickbench_runner"

# 2. Create the pinned Python environment and install pinned ClickHouse static binary.
timeout 120s uv venv --python 3.12.10 "$work/venv"
timeout 180s uv pip install --python "$work/venv/bin/python" -r "$b/requirements.txt"
py="$work/venv/bin/python"
ch="$cache_root/bin/clickhouse"
timeout 300s "$py" -B "$b/harness.py" install-clickhouse --dest "$ch"

# 3. Run deterministic unit + 50K-row 43-query 4-way smoke parity & rejection tests.
MORSEL_CLICKBENCH_RUNNER="$runner" MORSEL_CLICKBENCH_CLICKHOUSE="$ch" \
  timeout 300s "$py" -B -m unittest discover -s "$b" -v

# 4. Download the official ClickBench partitioned Parquet subset (20 files = 20M rows)
#    or omit --count 20 for all 100 files (~100M rows, 14.74 GB).
data_dir="$cache_root/hits_20"
timeout 600s "$py" -B "$b/harness.py" download "$data_dir" --count 20 --jobs 8

# 5. Probe Morsel admission / grouped rejection matrix across all 43 queries.
"$py" -B "$b/harness.py" plan-matrix "$data_dir" --runner "$runner" --threads 8

# 6. Run the controlled warm-cache 4-way 43-query benchmark on 8 pinned cores.
vmstat -y 1 3
timeout --kill-after=15s 2400s taskset -c 0-7 "$py" -B "$b/harness.py" bench \
  "$data_dir" --runner "$runner" --clickhouse "$ch" --revision "$revision" \
  --threads 8 --timeout 180 --cache warm --warmups 1 --repeats 5 \
  > "$work/clickbench_20_warm.jsonl"
vmstat -y 1 3

# 7. Print 4-way summary tables and Morsel admission report.
"$py" -B "$b/harness.py" summarize "$work/clickbench_20_warm.jsonl"

# 8. Clean up the detached worktree when finished.
git worktree remove --force "$work/pinned"
```

## 4-Way ClickBench Parquet Baseline Report (`rob-tandy-dduck2`, 2026-11-28 UTC)

### Environment, Dataset Layout, and Warm-Cache Evidence

- **Host:** `rob-tandy-dduck2`, Ubuntu 24.04.5 LTS, Linux `7.0.0-1012-aws`, 16
  ARM Neoverse-N1 vCPUs (1 thread per core), 61 GiB RAM (`cgroup memory.max =
63.78 GiB`).
- **CPU pinning and threads:** `taskset -c 0-7` (8 dedicated logical CPUs),
  `--threads 8`, batch size `8,192`.
- **Host idle check (`vmstat -y 1 3`):** 73–81% idle immediately before the run;
  82–93% idle immediately after (no concurrent Rust builds).
- **Engine versions & build:**
  - DataFusion revision: `1663c620eede06cfde33ebb0fa1beeef0868b714`
    (`release-nonlto`, `--locked --no-default-features --features
parquet,sql,datetime_expressions,string_expressions,unicode_expressions,regex_expressions`)
  - Rust `1.99.1` (`58cf8f9c3`, `aarch64-unknown-linux-gnu`), Cargo `1.99.1`
  - DuckDB `2.0.1`, PyArrow `26.0.1`, NumPy `2.5.5`, Python `3.12.10`
  - ClickHouse `26.11.5.8 (official build)` (`v26.11.5.8-stable`, Linux `arm64`,
    SHA-512
    `482c93a4c938af30e0d6a37788ae988e30c8951e48cc31ecc350c8434a89ddd711b55a54accdc0de3f1a3ab7bdbfe1ca19cac96824bd50717f3b621376c4dada`)
- **Dataset layout (documented 20-file official ClickBench partitioned subset):**
  - **Files:** 20 official partitioned Parquet files
    (`hits_0.parquet`..`hits_19.parquet` from
    `https://datasets.clickhouse.com/hits_compatible/athena_partitioned/`)
  - **Rows / Row groups / Columns / Bytes:** `20,000,000` rows, `61` row groups,
    `105` columns, `2,711,471,866` bytes (`2.53 GiB`)
  - **Storage:** Local EC2 NVMe Btrfs volume
    (`/dev/mapper/vg_ws_ephemeral-volume1p1`, `compress=zstd:1`)
  - **Why 20 files instead of all 100 files:** All 100 official files
    (`99,997,497` rows, `325` row groups, `14,737,666,736` bytes / `13.73 GiB`)
    downloaded cleanly, but on `rob-tandy-dduck2`'s shared-host cgroup the
    sustainable active page-cache ceiling plateaued at ~11.0 GiB (`80.6%`
    residency across 100 files), preventing strict 100% `mincore` warm-cache
    verification (and `q32`'s high-cardinality hash table would trigger further
    eviction). The 20-file (`20,000,000`-row, `2.53 GiB`) multi-file subset
    maintains **100.000% `mincore` warm page-cache residency (`661,990 / 661,990`
    4 KiB pages) before and after all 1,075 measured trials** (`0` residency
    drops, including across all `q32` trials).
- **Protocol:** 1 unmeasured warmup round + 5 measured rounds per query/arm with
  alternating arm order (1,075 measured trials total), following a complete
  43-query 4-way Arrow parity pass across all 5 arms (`0` mismatches).

### 43-Query 4-Way Warm-Cache Comparison (Medians in ms, Total In-Process Query Time)

| Query                               | Rows | `morsel_pushdown` ms | `stock_default` ms | `stock_pushdown` ms | `duckdb_default` ms | `clickhouse_default` ms | Morsel / Stock-PD | Morsel / DuckDB | Morsel / ClickHouse |
| ----------------------------------- | ---: | -------------------: | -----------------: | ------------------: | ------------------: | ----------------------: | ----------------: | --------------: | ------------------: |
| `q0`                                |    1 |                 18.9 |               16.6 |                16.9 |                22.6 |                    44.0 |             1.12x |           0.84x |               0.43x |
| `q1`                                |    1 |                 26.6 |               28.1 |                28.3 |                29.0 |                    44.0 |             0.94x |           0.92x |               0.60x |
| `q2`                                |    1 |                 43.3 |               55.4 |                46.7 |                49.0 |                    53.0 |             0.93x |           0.88x |               0.82x |
| `q3`                                |    1 |                 53.5 |               45.7 |                47.5 |                54.8 |                    63.0 |             1.13x |           0.98x |               0.85x |
| `q4`                                |    1 |                186.3 |              203.1 |               201.0 |               157.4 |                    75.0 |             0.93x |           1.18x |               2.48x |
| `q5`                                |    1 |                251.7 |              307.4 |               326.1 |               285.4 |                   133.0 |             0.77x |           0.88x |               1.89x |
| `q6`                                |    1 |                 24.4 |               17.2 |                23.0 |                43.6 |                    48.0 |             1.06x |           0.56x |               0.51x |
| `q7`                                |   12 |                 27.3 |               29.1 |                35.2 |                33.5 |                    54.0 |             0.77x |           0.81x |               0.50x |
| `q8`                                |   10 |                243.7 |              276.6 |               277.0 |               208.6 |                   215.0 |             0.88x |           1.17x |               1.13x |
| `q9`                                |   10 |                314.2 |              323.4 |               325.7 |               277.9 |                   299.0 |             0.96x |           1.13x |               1.05x |
| `q10`                               |   10 |                 79.3 |               79.5 |               128.6 |                82.3 |                   100.0 |             0.62x |           0.96x |               0.79x |
| `q11`                               |   10 |                 83.7 |               89.5 |               119.3 |                91.3 |                   104.0 |             0.70x |           0.92x |               0.80x |
| `q12`                               |   10 |                249.8 |              280.3 |               287.3 |               252.8 |                   210.0 |             0.87x |           0.99x |               1.19x |
| `q13`                               |   10 |                436.7 |              416.1 |               415.5 |               393.5 |                   268.0 |             1.05x |           1.11x |               1.63x |
| `q14`                               |   10 |                237.5 |              288.8 |               296.0 |               294.4 |                   217.0 |             0.80x |           0.81x |               1.09x |
| `q15`                               |   10 |                173.7 |              241.6 |               231.2 |               177.0 |                   121.0 |             0.75x |           0.98x |               1.44x |
| `q16`                               |   10 |                558.0 |              956.9 |               641.8 |               465.3 |                   431.0 |             0.87x |           1.20x |               1.29x |
| `q17`                               |   10 |                575.8 |              628.0 |               535.1 |               386.2 |                   160.0 |             1.08x |           1.49x |               3.60x |
| `q18`                               |   10 |                886.3 |             1113.0 |              1092.8 |              1068.4 |                   693.0 |             0.81x |           0.83x |               1.28x |
| `q19`                               |    0 |                 46.3 |               42.4 |                49.2 |                35.9 |                    46.0 |             0.94x |           1.29x |               1.01x |
| `q20`                               |    1 |                335.2 |              354.5 |               360.0 |               427.8 |                   366.0 |             0.93x |           0.78x |               0.92x |
| `q21`                               |   10 |                371.6 |              341.5 |               434.8 |               383.8 |                   367.0 |             0.85x |           0.97x |               1.01x |
| `q22`                               |   10 |                749.0 |              785.1 |               916.3 |               873.8 |                   610.0 |             0.82x |           0.86x |               1.23x |
| `q23`                               |   10 |               1344.2 |             2239.4 |               310.1 |               395.0 |                   428.0 |             4.33x |           3.40x |               3.14x |
| `q24`                               |   10 |                137.3 |               98.4 |                89.5 |               121.0 |                    97.0 |             1.53x |           1.13x |               1.42x |
| `q25`                               |   10 |                 95.1 |              126.6 |               130.4 |               153.9 |                   200.0 |             0.73x |           0.62x |               0.48x |
| `q26`                               |   10 |                158.6 |               89.7 |               105.3 |               111.1 |                    95.0 |             1.51x |           1.43x |               1.67x |
| `q27`                               |   25 |                269.7 |              303.1 |               328.7 |               300.8 |                   231.0 |             0.82x |           0.90x |               1.17x |
| `q28`                               |   25 |               1663.9 |             1852.5 |              1832.6 |              4733.7 |                  5313.0 |             0.91x |           0.35x |               0.31x |
| `q29`                               |    1 |                 48.0 |               58.8 |                58.1 |                68.8 |                    73.0 |             0.83x |           0.70x |               0.66x |
| `q30`                               |   10 |                176.8 |              228.8 |               219.3 |               253.1 |                   209.0 |             0.81x |           0.70x |               0.85x |
| `q31`                               |   10 |                223.0 |              233.8 |               227.8 |               291.1 |                   248.0 |             0.98x |           0.77x |               0.90x |
| `q32`                               |   10 |                871.3 |              833.7 |               848.4 |               687.1 |                   841.0 |             1.03x |           1.27x |               1.04x |
| `q33`                               |   10 |                800.4 |              863.8 |               894.2 |               717.4 |                   536.0 |             0.90x |           1.12x |               1.49x |
| `q34`                               |   10 |                826.2 |              874.9 |               898.6 |               758.2 |                   539.0 |             0.92x |           1.09x |               1.53x |
| `q35`                               |   10 |                124.9 |              196.3 |               196.5 |               151.6 |                    98.0 |             0.64x |           0.82x |               1.27x |
| `q36`                               |   10 |                159.1 |              172.2 |               165.4 |               139.4 |                   131.0 |             0.96x |           1.14x |               1.21x |
| `q37`                               |   10 |                 96.8 |               86.9 |                99.5 |               119.7 |                    89.0 |             0.97x |           0.81x |               1.09x |
| `q38`                               |   10 |                 89.6 |               94.4 |                84.6 |                68.9 |                    78.0 |             1.06x |           1.30x |               1.15x |
| `q39`                               |   10 |                288.1 |              331.8 |               314.0 |               268.7 |                   219.0 |             0.92x |           1.07x |               1.32x |
| `q40`                               |   10 |                 52.1 |               44.1 |                51.4 |                40.1 |                    63.0 |             1.01x |           1.30x |               0.83x |
| `q41`                               |    0 |                 51.0 |               41.9 |                47.5 |                41.4 |                    56.0 |             1.07x |           1.23x |               0.91x |
| `q42`                               |   10 |                 41.1 |               40.8 |                42.3 |                98.0 |                    55.0 |             0.97x |           0.42x |               0.75x |
| **GeoMean (43/43 Morsel-admitted)** |    — |            **169.7** |          **180.2** |           **178.6** |           **176.7** |               **161.9** |         **0.95x** |       **0.96x** |           **1.05x** |
| **Sum (43/43 Morsel-admitted)**     |    — |          **13489.9** |        **15731.8** |         **13779.4** |         **15613.1** |             **14320.0** |         **0.98x** |       **0.86x** |           **0.94x** |

#### Execution-Only Geometric Mean Breakdown

Median per-query view/footer registration (`register_seconds`) was `12.6 ms` for
DataFusion (`+4.9–6.2 ms` planning), `3.2 ms` for DuckDB, and `15.0 ms` for
ClickHouse. Excluding registration and planning:

- **All 43 queries (`exec_seconds` GeoMean):** `stock_default` = `110.6 ms`,
  `stock_pushdown` = `111.3 ms`, `morsel_pushdown` = `128.2 ms`,
  `clickhouse_default` = `138.8 ms`, `duckdb_default` = `170.7 ms`. (Stock's
  all-43 `exec_seconds` GeoMean is pulled down by `q0` and `q6`, which Stock
  folds at plan time into `PlaceholderRowExec` with `0.05–0.07 ms` exec time.)
- **41 non-metadata-folded queries (`q0`, `q6` excluded; `exec_seconds`
  GeoMean):** `clickhouse_default` = `149.5 ms`, `morsel_pushdown` = `151.3 ms`
  (**1.01x vs ClickHouse**, **0.94x vs `stock_default` (`160.4 ms`)**, **0.94x
  vs `stock_pushdown` (`161.1 ms`)**, **0.81x vs DuckDB (`186.4 ms`)**).

### Morsel (`morsel_mode=required`) Admission & Grouped Rejection Matrix

| Metric / Rejection Category           |  Count | Queries                                         | Notes                                                                                                                                            |
| ------------------------------------- | -----: | ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Admitted (`MorselExec:` root)**     | **43** | `q0`–`q42` (all 43 official ClickBench queries) | Verified on both the 20-file (20M-row) and 100-file (100M-row) datasets; 100% 4-way Arrow parity across all 43 queries.                          |
| **Rejected (`morsel_mode=required`)** |  **0** | _(none)_                                        | `harness.py plan-matrix` and `summarize` group any future/custom-query rejections by operator category; verified via negative window-query test. |

### Key Observations and ClickBench Optimization Roadmap for Morsel

- **Overall competitiveness:** `morsel_pushdown` wins on **30/43** queries vs
  `stock_default`, **31/43** vs `stock_pushdown`, **25/43** vs DuckDB `2.0.1`,
  and **17/43** vs ClickHouse `26.11.5.8`, achieving the **lowest total sum of
  medians (`13,489.9 ms`)** of all 4 engines and beating Stock and DuckDB on
  43-query geometric mean (`169.7 ms` vs `178.6 ms` / `176.7 ms`).
- **Top optimization opportunity — `q23` (`SELECT * WHERE "URL" LIKE '%google%'
ORDER BY "EventTime" LIMIT 10`, `1344.2 ms` vs `stock_pushdown` `310.1 ms`,
  `4.33x`):** `morsel_pushdown` scanned `2,673,595,281` bytes across all 105
  columns (`adaptive_row_filter_rgs=48`, `adaptive_bulk_decode_rgs=13`,
  `morsel_decode_work_ms=7,787 ms`), whereas `stock_pushdown` scanned only
  `479,069,642` bytes (`5.6x` less) by lazily fetching non-predicate columns
  only after the `URL` `RowFilter` succeeds. Avoiding eager wide-projection
  column fetch on selective `RowFilter` scans would cut ~1 second off `q23`
  alone.
- **Dynamic TopK filter pushdown — `q24` (`137.3` vs `89.5 ms`, `1.53x`) and
  `q26` (`158.6` vs `105.3 ms`, `1.51x`):** Stock DataFusion pushes dynamic
  TopK `EventTime` bounds back into the Parquet scan; wiring dynamic TopN
  threshold pushdown into `NativeParquet` + `TopNSink` would close this gap.
- **Early limit termination — `q17` (`GROUP BY ... LIMIT 10`, `575.8 ms` vs
  ClickHouse `160.0 ms`, `3.60x`):** `MorselExec` materializes all
  `(UserID, SearchPhrase)` groups in `stage_0` (`AggregateSink`) before
  `LimitSink[10]`.
- **Metadata-only statistics folding — `q0` (`18.9` vs `16.6 ms`) and `q6`
  (`24.4` vs `17.2 ms`):** Stock folds `COUNT(*)` and `MIN`/`MAX` via
  `AggregateStatistics` (`PlaceholderRowExec`), whereas `morsel_mode=required`
  executes `NativeParquet -> CountSink`/`AggregateSink`.
- **High-cardinality `COUNT(DISTINCT)` and `GROUP BY` queries:**
  `morsel_pushdown` outperforms `stock_pushdown` on `q4` (`0.93x`), `q5`
  (`0.77x`), `q10` (`0.62x`), `q11` (`0.70x`), `q15` (`0.75x`), `q16` (`0.87x`),
  `q33` (`0.90x`), `q34` (`0.92x`), and `q35` (`0.64x`), while trailing slightly
  on `q13` (`436.7` vs `415.5 ms`, `1.05x`) and `q32` (`871.3` vs `848.4 ms`,
  `1.03x`). ClickHouse's specialized `uniq` and string hash tables still lead on
  `q4` (`75 ms` vs `186 ms`), `q5` (`133 ms` vs `252 ms`), `q13` (`268 ms` vs
  `437 ms`), and `q33`/`q34` (`536–539 ms` vs `800–826 ms`).
- **Regex throughput — `q28` (`REGEXP_REPLACE`):** DataFusion (`morsel_pushdown`
  `1,663.9 ms`, `stock_pushdown` `1,832.6 ms`) is `2.8–3.2x` faster than DuckDB
  (`4,733.7 ms`) and ClickHouse (`5,313.0 ms`) thanks to Rust `regex` SIMD vs
  RE2.
