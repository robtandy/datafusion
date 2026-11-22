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
CPUs 0–7, Btrfs on instance storage) with Rust 1.99.1 and the pinned Python
packages above. Each query ran for one warmup pair and five measured pairs.
Both arms of both fixtures matched exactly 378,916 rows at low selectivity
and 4,941,572 at high selectivity, including the scan values and count scalars.
`file-skew` was 155,785,925 bytes (38,046 resident pages before each sample);
`large-row-groups` was 155,608,030 bytes (37,991 resident pages).

| Fixture            | Query        | Stock ms, median (range) | Native ms, median (range) | Native claims by worker ID, round 1 |
| ------------------ | ------------ | -----------------------: | ------------------------: | ----------------------------------- |
| `file-skew`        | `scan_low`   |      389.4 (274.3–533.3) |       564.3 (360.3–671.5) | `[7, 7, 8, 9, 8, 8, 8, 9]`          |
| `file-skew`        | `count_low`  |      173.2 (128.1–214.3) |       225.5 (203.1–319.2) | `[8, 9, 8, 9, 7, 7, 8, 8]`          |
| `file-skew`        | `scan_high`  |      401.2 (357.6–403.4) |       663.8 (540.1–682.9) | `[8, 8, 8, 8, 8, 8, 8, 8]`          |
| `file-skew`        | `count_high` |      224.3 (192.2–296.8) |       282.3 (246.4–295.1) | `[8, 7, 10, 8, 8, 8, 7, 8]`         |
| `large-row-groups` | `scan_low`   |      422.5 (365.5–445.9) |       560.1 (264.2–644.9) | `[1, 1, 1, 0, 0, 0, 0, 1]`          |
| `large-row-groups` | `count_low`  |      197.4 (153.6–331.4) |       169.0 (120.0–198.0) | `[1, 1, 1, 0, 0, 0, 0, 1]`          |
| `large-row-groups` | `scan_high`  |      382.0 (310.5–459.3) |       547.9 (336.7–634.4) | `[1, 1, 1, 0, 0, 0, 0, 1]`          |
| `large-row-groups` | `count_high` |      238.9 (163.2–369.2) |       181.9 (132.5–212.9) | `[1, 1, 1, 0, 0, 0, 0, 1]`          |

Stock `file-skew` plans used eight file partitions, with 25, 25, 4, 2, 2,
2, 2 and 2 row groups assigned statically. Native distributed the 64 readers
across all eight workers in every trial (6–11 claims per worker across all
trials), but that did not improve median elapsed time. Stock `large-row-groups`
used two file partitions; repartitioning batches downstream does not split the
scan itself. Native used four of the eight workers in every trial, each
claiming and completing one reader, with the worker IDs varying by trial.
Native counts had lower medians on this layout, but native projected scans did
not. Claim counts measure scheduling, not CPU utilization; elapsed times also
include path-specific filtering, output handling or count reduction. Other
jobs ran on the host, so the timing ranges are descriptive, not a speedup claim.

See the [intra-row-group splitting design](../../docs/source/contributor-guide/morsel_parquet_splitting.md)
for proposed changes to the native source. This benchmark does not implement
splitting.
