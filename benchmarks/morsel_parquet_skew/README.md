<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Native Parquet file skew and row-group coarseness (#156)

The [#128 Parquet benchmark](../morsel_parquet/README.md) uses uniform files
and row groups. This experiment uses the same seeded data model, projected
scans and filtered counts at both selectivities, but varies two aspects of the
file layout. Both fixtures have 8,388,608 rows:

| Fixture            | Files | Row groups per file (in path order) | Rows per group |
| ------------------ | ----: | ----------------------------------- | -------------: |
| `file-skew`        |    16 | 24, 1, 24, 1, 2, 2, then ten × 1    |        131,072 |
| `large-row-groups` |     2 | 2, 2                                |      2,097,152 |

In `file-skew`, the two largest files are in different stock file groups and
contain exactly 75% of the rows. `large-row-groups` has only four row groups
for eight workers. Both fixtures use Zstandard level 3, statistics, no
Parquet Page Index and no dictionary encoding. `--smoke` keeps the geometry
but divides the rows per group by 1,024.

## Reproduce

Run from the repository root on a Linux workspace with `uv`, `taskset` and
Rustup. The comparison uses one binary from the #138 metrics merge, pinned
independently of the branch being reviewed. Like #128, the build copies a Cargo
example into a disposable detached checkout; neither workspace manifests nor
engine sources need editing. Use a new fixture directory on the filesystem
under test and choose eight available CPUs on an otherwise idle host.

```bash
set -e
b=benchmarks/morsel_parquet_skew
work=$(mktemp -d "$HOME/.cache/issue156.XXXXXX")
revision=c2809d3b3ee73936b182429e1646af290d797c6e

git worktree add --detach "$work/pinned" "$revision"
mkdir -p "$work/pinned/datafusion/core/examples"
cp "$b/stream_runner.rs" "$work/pinned/datafusion/core/examples/morsel_parquet_skew.rs"
(
  cd "$work/pinned"
  MORSEL_BENCH_REVISION="$(git rev-parse HEAD)" CARGO_TARGET_DIR="$work/target" \
    timeout --kill-after=20s 35m cargo build --locked --profile release-nonlto -j 10 \
      -p datafusion --no-default-features --features parquet,sql --example morsel_parquet_skew
)
runner="$work/target/release-nonlto/examples/morsel_parquet_skew"
timeout 120s uv venv --python 3.12.10 "$work/venv"
timeout 180s uv pip install --python "$work/venv/bin/python" \
  -r benchmarks/morsel_parquet/requirements.txt
py="$work/venv/bin/python"

MORSEL_PARQUET_RUNNER="$runner" MORSEL_BENCH_REVISION="$revision" \
  timeout 120s taskset -c 0-7 "$py" -B -m unittest discover -s "$b" -v

fixture_root=$(mktemp -d "${MORSEL_PARQUET_FIXTURE_ROOT:-$HOME/.cache}/morsel-parquet-skew.XXXXXX")
findmnt -T "$fixture_root" -o TARGET,SOURCE,FSTYPE
for case in file-skew large-row-groups; do
  timeout --kill-after=10s 90s "$py" -B "$b/bench.py" generate "$case" "$fixture_root/$case"
  (uptime; vmstat 1 3) > "$work/$case.host-before.log"
  timeout --kill-after=10s 240s taskset -c 0-7 "$py" -B "$b/bench.py" ab \
    "$case" "$fixture_root/$case" --runner "$runner" --revision "$revision" \
    --timeout 120 --warmups 1 --repeats 5 > "$work/$case.jsonl"
done
```

The harness verifies the footer geometry and checks unordered scan rows,
projected values (including nulls), and both count scalars in each mode against
#128's exact NumPy oracle. Before reporting timings, it verifies the stock
`DataSourceExec` and the `MorselExec`/`NativeParquet[fixture]` paths and checks
that native count queries use `CountSink` and finalize once. It rechecks plans,
build revision, output size, and row-group claims and completions for each
sample. The #138 vectors are in worker-ID order and include workers with zero
claims. Stock has no native worker report and emits `null`, not zeros.

Each sample uses a fresh process, runtime and session, with 8,192-row batches
and eight target partitions and Tokio workers. To isolate whole-file versus
row-group scheduling, stock file-range repartitioning is **disabled** as in
#128. Listing statistics, row-group pruning, decoder filter pushdown, and page
index and Bloom-filter reads are also disabled in both arms. These results are
not a claim about stock with optimizer defaults. Both paths use the same SQL
and drain the same logical results with no CSV writing or collection in the
timed path. The timer includes registration, SQL and physical planning, and
stream exhaustion. Native runs include detailed metrics collection overhead;
formatting the report is outside the timer. The harness verifies full Linux
page-cache residency with `mincore` before every warmup and measured trial,
re-reading the fixture outside the timer if pages have been reclaimed. It
fails if it cannot verify full residency. Each query is paired by mode, with
the leading mode alternating by round.

## dduck2 warm-cache baseline

Recorded on `rob-tandy-dduck2` (AArch64 Neoverse-N1, Linux `7.0.0-1012-aws`,
CPUs 0–7, Btrfs on instance storage) on 2026-11-22 with Rust 1.99.1 and the
pinned Python packages above. No builds or lint jobs were running. Before
`file-skew`, `uptime` reported load averages of 1.80, 1.46, 5.08; before
`large-row-groups`, it reported 1.77, 1.48, 4.97. The two one-second
`vmstat 1 3` samples before each case reported 99% and 98% CPU idle,
excluding the since-boot average. Each query ran for one warmup pair and five
measured pairs.
Both arms of both fixtures matched exactly 378,916 rows at low selectivity
and 4,941,572 at high selectivity, including the scan values and count scalars.
`file-skew` was 155,785,925 bytes (38,046 resident pages before each sample);
`large-row-groups` was 155,608,030 bytes (37,991 resident pages).

| Fixture            | Query        | Stock ms, median (range) | Native ms, median (range) | Native claims by worker ID, round 1 |
| ------------------ | ------------ | -----------------------: | ------------------------: | ----------------------------------- |
| `file-skew`        | `scan_low`   |      226.5 (224.1–235.3) |       321.2 (316.1–322.3) | `[7, 8, 9, 8, 8, 8, 8, 8]`          |
| `file-skew`        | `count_low`  |      118.3 (116.4–141.7) |       129.9 (128.8–131.1) | `[7, 8, 8, 9, 8, 8, 8, 8]`          |
| `file-skew`        | `scan_high`  |      242.3 (240.7–244.3) |       322.7 (321.1–324.1) | `[8, 8, 8, 8, 8, 8, 8, 8]`          |
| `file-skew`        | `count_high` |      117.4 (116.9–118.5) |       131.5 (129.3–133.0) | `[7, 8, 9, 8, 8, 8, 8, 8]`          |
| `large-row-groups` | `scan_low`   |      289.1 (286.4–290.7) |       207.7 (206.9–211.2) | `[1, 1, 1, 0, 0, 0, 0, 1]`          |
| `large-row-groups` | `count_low`  |      142.2 (141.2–150.8) |       108.8 (108.1–109.0) | `[1, 1, 0, 1, 0, 0, 0, 1]`          |
| `large-row-groups` | `scan_high`  |      290.9 (290.3–294.0) |       215.1 (213.8–215.8) | `[0, 0, 0, 1, 1, 1, 0, 1]`          |
| `large-row-groups` | `count_high` |      143.1 (140.6–143.9) |       112.6 (112.1–113.2) | `[0, 0, 0, 1, 1, 1, 0, 1]`          |

Stock `file-skew` plans used eight file partitions, with 25, 25, 4, 2, 2,
2, 2 and 2 row groups assigned statically. Native distributed the 64 readers
across all eight workers in every trial (7–9 claims per worker), but stock
had lower median elapsed times for all four queries on this layout.
Stock `large-row-groups` used two file partitions; repartitioning batches
downstream does not split the scan itself. Native used four of the eight
workers in every trial, each claiming and completing one reader, with the
worker IDs varying by trial. Native had lower median elapsed times for all four
queries on this layout despite leaving four workers without a reader. Claim
counts describe work assignment, not CPU utilization; elapsed times also
include filtering, output handling or count reduction.

See the [intra-row-group splitting design](../../docs/source/contributor-guide/morsel_parquet_splitting.md)
for proposed changes to the native source. This benchmark does not implement
splitting.
