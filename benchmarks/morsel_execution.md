<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# In-memory morsel A/B benchmark (fork issue #127)

This benchmark compares stock execution (`morsel_mode=off`) and native execution
(`morsel_mode=required`) in the same fork binary. The Apache main baseline is
separate: it uses the stock planner and executor at the fork point,
`7e3d65161046baf484e9552b65f71256fc8837bb`, with only the harness and its Cargo
target copied into a detached worktree. `--apache-main` selects only stock and
asserts that the fork-only session setting is absent.

## Workloads and timing

Both queries use one read-only `MemTable`, without I/O or an `ORDER BY`:

```sql
SELECT id + 1 AS result FROM items WHERE score < 64;
SELECT COUNT(*) AS result FROM items WHERE score < 64;
```

The fixture has 8,388,608 rows (128 MiB of input column buffers), 2,048 input
batches of 4,096 rows, and two non-null `Int64` columns. IDs are consecutive
from zero; scores are bytes from `SmallRng` seeded with `127` (locked rand
0.9.4, aarch64). Only the partition boundaries change across configurations, so
the input rows, their order within the flattened input, and the ~25% predicate
selectivity are identical. Native morsels in this revision are four input
batches each (16,384 rows, 512 morsels per query).

| Layout  | Logical partitions | Batches per partition   | Target partitions / native worker cap | Tokio worker threads |
| ------- | -----------------: | ----------------------- | ------------------------------------: | -------------------: |
| Uniform |                  1 | 2,048                   |                                     1 |                    1 |
| Uniform |                  4 | 512 / 512 / 512 / 512   |                                     4 |                    4 |
| Skewed  |                  4 | 1,664 / 128 / 128 / 128 |                                     4 |                    4 |

Skew puts 81.25% of the input rows in the first logical partition without
changing predicate selectivity. There is no separate single-partition skew
case because it would be identical to the single-worker uniform case. Both
modes use the same execution batch size (4,096) and the same Tokio
multi-thread runtime configuration. The drain runs on that worker pool, so
it does not add a query execution thread.

For every query and configuration, the harness inspects the physical plans
before measuring. It checks that stock has no `MorselExec`, native has
`MorselExec` with the requested worker count and a filter, and only native
count uses `CountSink`. It also verifies matching fork schemas, checks each
projected value against the fixture without imposing output order or batching,
and checks the single count value against the fixture cardinality.

- **First result:** wall time from calling `execute_stream` to receiving the
  first non-empty batch.
- **Total:** wall time from the same start through draining the single output
  stream to EOF and dropping all batches. The consumer counts output rows but
  neither materializes nor formats the results.
- **Throughput:** input rows divided by median total time, not output rows
  divided by time. In particular, a count emits just one row.

Plans are made once per query and mode. Fixture generation, session and task
context creation, SQL planning/optimization, plan reset, and oracle checks
are outside the timer. There are five discarded warmups and 31 timed runs per
row; the fork alternates which mode runs first on each warmup and repetition.
Durations are reported as median [p10, p90] of the observed runs, not confidence
intervals.

## Reproduce

In the fork worktree with the benchmark harness, run the following (adapt
CPU affinity and target directories to your workspace). Use separate target
directories: the two checkouts contain crates with the same name and version,
and sharing build artifacts can make the fork run the wrong planner.

```sh
export CARGO_TARGET_DIR="$HOME/.cache/issue127-fork-target"
timeout 1200 taskset -c 12-15 cargo bench --locked --profile release-nonlto -j 10 \
  -p datafusion --no-default-features --features sql --bench morsel_execution
```

For the baseline, from the same fork worktree, use the same harness, toolchain,
Cargo profile and flags in a detached checkout. This is a local benchmark
setup only; no patch is submitted to Apache DataFusion.

```sh
git fetch https://github.com/apache/datafusion.git 7e3d65161046baf484e9552b65f71256fc8837bb
git worktree add --detach /tmp/issue127-apache-main 7e3d65161046baf484e9552b65f71256fc8837bb
cp datafusion/core/benches/morsel_execution.rs /tmp/issue127-apache-main/datafusion/core/benches/
printf '\n[[bench]]\nharness = false\nname = "morsel_execution"\nrequired-features = ["sql"]\n' \
  >> /tmp/issue127-apache-main/datafusion/core/Cargo.toml
(
  cd /tmp/issue127-apache-main
  export CARGO_TARGET_DIR="$HOME/.cache/issue127-apache-target"
  timeout 1200 taskset -c 12-15 cargo bench --locked --profile release-nonlto -j 10 \
    -p datafusion --no-default-features --features sql --bench morsel_execution -- --apache-main
)
```

For a short plan-selection and result-parity test instead of timing, pass
`-- --verify` to the fork benchmark, or run the test command below. Both paths
use a 16-batch fixture with the same layouts.

```sh
timeout 900 cargo test -p datafusion --bench morsel_execution
```

## Remote workspace results

Measured 2026-11-18 UTC on `rob-tandy-dduck1` (Ubuntu 24.04.5 LTS, Linux
`7.0.0-1014-aws`, 16 ARM Neoverse-N1 vCPUs, 61 GiB RAM). Both processes were
restricted to CPUs 12–15; individual runtime threads were not pinned. The
fork revision was `d9a0de0564ad3902393362e3b880045032ccc6c7` (integration
base `3dad9ce14977da4bb36a8c0d4c22c92c87a8ca25`). The baseline revision was
`7e3d65161046baf484e9552b65f71256fc8837bb` plus the two benchmark files described
above. Both used rustc and Cargo 1.99.1 (`aarch64-unknown-linux-gnu`), the
workspace's Clang 18.1.3/lld linker, Tokio 1.54.1, and `release-nonlto`
(optimized, LTO off, 16 codegen units) with only the `sql` package feature enabled.
No Tokio I/O/time drivers or custom blocking pool settings were used.

All modes selected 2,094,791 input rows in every layout. Stock and Apache
main used the same plan shapes, including per-partition partial aggregation
for the four-partition count. The native plans showed `MorselExec` with
`ResultSink[capacity=2]` and `CountSink` for the count query only.

| Layout  | Workers | Query      | Mode        | Input Mrows/s |                First ms |                Total ms |
| ------- | ------: | ---------- | ----------- | ------------: | ----------------------: | ----------------------: |
| uniform |       1 | projection | off         |         424.1 |    0.684 [0.325, 0.704] | 19.782 [19.455, 19.994] |
| uniform |       1 | projection | required    |         314.7 |    0.042 [0.030, 0.052] | 26.657 [26.478, 26.831] |
| uniform |       1 | projection | apache-main |         417.1 |    0.328 [0.311, 0.360] | 20.110 [19.774, 20.940] |
| uniform |       1 | count      | off         |         802.6 | 10.447 [10.100, 10.635] | 10.452 [10.104, 10.640] |
| uniform |       1 | count      | required    |         579.8 | 14.460 [14.381, 14.569] | 14.469 [14.383, 14.571] |
| uniform |       1 | count      | apache-main |         848.2 |   9.886 [9.807, 10.066] |   9.890 [9.811, 10.070] |
| uniform |       4 | projection | off         |         610.7 |    0.344 [0.275, 0.850] |  13.737 [6.809, 18.912] |
| uniform |       4 | projection | required    |         558.3 |    0.044 [0.036, 0.072] |  15.025 [7.805, 16.491] |
| uniform |       4 | projection | apache-main |         898.3 |    0.298 [0.262, 0.407] |   9.338 [6.939, 14.966] |
| uniform |       4 | count      | off         |        2854.8 |    2.936 [2.765, 6.276] |    2.938 [2.767, 6.278] |
| uniform |       4 | count      | required    |        1964.5 |    4.268 [4.046, 7.465] |    4.270 [4.048, 7.468] |
| uniform |       4 | count      | apache-main |        2166.7 |    3.869 [2.740, 5.183] |    3.872 [2.743, 5.186] |
| skewed  |       4 | projection | off         |         435.7 |    0.338 [0.215, 0.836] | 19.253 [18.762, 20.795] |
| skewed  |       4 | projection | required    |         754.6 |    0.051 [0.034, 0.072] |  11.116 [9.966, 18.186] |
| skewed  |       4 | projection | apache-main |         349.4 |    0.429 [0.224, 0.909] | 24.007 [18.966, 27.377] |
| skewed  |       4 | count      | off         |         925.3 |    9.063 [8.725, 9.569] |    9.066 [8.728, 9.572] |
| skewed  |       4 | count      | required    |        1558.4 |    5.380 [5.292, 6.603] |    5.383 [5.295, 6.606] |
| skewed  |       4 | count      | apache-main |         806.7 |  10.396 [8.979, 11.259] |  10.398 [8.981, 11.261] |

Against fork `off`, native delivered the first projected batch sooner in
all three cases, but took 35% longer to drain the single-worker projection
and 38% longer for the single-worker count. In the balanced four-worker case,
native medians were slower for both queries (9% for projection and 45% for
count), with overlapping p10–p90 ranges. Under four-worker partition skew,
native cut median total time by 42% for projection and 41% for count. Count
emits only after finalization, so its first-result and total times are close.

**Limitations.** One invocation per binary on a shared, non-isolated workspace
with other builds and tests running. The Apache comparison is not interleaved
with the fork runs; even the two stock builds showed substantially different
medians on some parallel cases. These results do not isolate a scheduler cost:
operators can emit different batch sizes and the stock parallel projection
stream also merges partitions. Skew here is a partition-size imbalance, not
unequal cost per row or per morsel. The results cover only these warm,
in-memory integer workloads, not cold runs, I/O, joins, grouped aggregation,
or spill.
