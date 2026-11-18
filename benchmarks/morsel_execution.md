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
are outside the timer. After five untimed warmups per mode, each of the 31
timed runs is immediately preceded by an untimed drain of the same query in
the same mode. The fork alternates which mode goes first per repetition, but
both modes (and the Apache baseline) start each sample after their own primer
rather than after the other executor. Durations are reported as median [p10,
p90] of the observed runs, not confidence intervals.

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
use a 16-batch fixture with the same layouts. Positional filters and Criterion
options such as `--save-baseline` forwarded by Cargo are ignored by this
harness.

```sh
timeout 900 cargo test -p datafusion --bench morsel_execution
```

## Remote workspace results

Measured 2026-11-18 UTC on `rob-tandy-dduck1` (Ubuntu 24.04.5 LTS, Linux
`7.0.0-1014-aws`, 16 ARM Neoverse-N1 vCPUs, 61 GiB RAM). Both processes were
restricted to CPUs 12–15; individual runtime threads were not pinned. The
fork revision was `366d34a61bbef2791e947b357ade51ab3079b497` (integration
base `3dad9ce14977da4bb36a8c0d4c22c92c87a8ca25`). The baseline revision was
`7e3d65161046baf484e9552b65f71256fc8837bb` plus the two benchmark files described
above. Both used rustc and Cargo 1.99.1 (`aarch64-unknown-linux-gnu`), the
workspace's Clang 18.1.3/lld linker, Tokio 1.54.1, and `release-nonlto`
(optimized, LTO off, 16 codegen units) with only the `sql` package feature enabled.
No Tokio I/O/time drivers or custom blocking pool settings were used.

The Apache run started at 23:27:37 UTC with a host load average (1/5/15 min)
of `2.43 / 3.86 / 9.98`; the fork run started at 23:27:59 UTC with
`1.94 / 3.66 / 9.79`. In two one-second `vmstat` samples just before each
launch, CPU idle time was 91–98% for Apache and 93–99% for the fork.

All modes selected 2,094,791 input rows in every layout. Stock and Apache
main used the same plan shapes, including per-partition partial aggregation
for the four-partition count. The native plans showed `MorselExec` with
`ResultSink[capacity=2]` and `CountSink` for the count query only.

| Layout  | Workers | Query      | Mode        | Input Mrows/s |                First ms |                Total ms |
| ------- | ------: | ---------- | ----------- | ------------: | ----------------------: | ----------------------: |
| uniform |       1 | projection | off         |         454.3 |    0.305 [0.292, 0.323] | 18.467 [18.328, 18.764] |
| uniform |       1 | projection | required    |         325.5 |    0.029 [0.028, 0.032] | 25.770 [25.540, 26.185] |
| uniform |       1 | projection | apache-main |         443.5 |    0.279 [0.259, 0.295] | 18.913 [18.828, 18.999] |
| uniform |       1 | count      | off         |         946.7 |    8.855 [8.805, 9.093] |    8.861 [8.811, 9.099] |
| uniform |       1 | count      | required    |         611.6 | 13.714 [13.596, 13.921] | 13.716 [13.598, 13.923] |
| uniform |       1 | count      | apache-main |         955.7 |    8.773 [8.746, 8.815] |    8.777 [8.750, 8.820] |
| uniform |       4 | projection | off         |        1312.7 |    0.264 [0.234, 0.280] |    6.391 [6.227, 6.656] |
| uniform |       4 | projection | required    |        1138.4 |    0.040 [0.039, 0.046] |    7.369 [7.307, 7.434] |
| uniform |       4 | projection | apache-main |        1311.7 |    0.256 [0.222, 0.277] |    6.395 [6.152, 6.615] |
| uniform |       4 | count      | off         |        3264.3 |    2.566 [2.505, 2.612] |    2.570 [2.509, 2.615] |
| uniform |       4 | count      | required    |        2183.5 |    3.839 [3.798, 3.915] |    3.842 [3.801, 3.917] |
| uniform |       4 | count      | apache-main |        3324.4 |    2.520 [2.462, 2.562] |    2.523 [2.465, 2.565] |
| skewed  |       4 | projection | off         |         475.1 |    0.209 [0.183, 0.230] | 17.657 [17.471, 17.941] |
| skewed  |       4 | projection | required    |        1112.8 |    0.040 [0.039, 0.054] |    7.538 [7.423, 7.601] |
| skewed  |       4 | projection | apache-main |         483.7 |    0.197 [0.166, 0.217] | 17.341 [17.152, 17.472] |
| skewed  |       4 | count      | off         |        1050.8 |    7.979 [7.894, 8.084] |    7.983 [7.897, 8.087] |
| skewed  |       4 | count      | required    |        2162.4 |    3.876 [3.849, 3.919] |    3.879 [3.851, 3.922] |
| skewed  |       4 | count      | apache-main |        1114.9 |    7.521 [7.481, 7.566] |    7.524 [7.484, 7.570] |

Against fork `off`, native delivered the first projected batch sooner in
all three cases, but took 40% longer to drain the single-worker projection
and 55% longer for the single-worker count. With four balanced partitions,
native median total times were 15% longer for projection and 49% longer for
count, with disjoint p10–p90 ranges. Under four-worker partition skew,
native cut median total times by 57% and 51%, respectively. Native median
total times differed by less than 3% between the two four-worker layouts.
Count emits only after finalization, so its first-result and total times are close.
Apache main median total times were within 6% of fork `off` across all six cases.

**Limitations.** The table reports one invocation per binary on a shared
workspace. The Apache comparison ran in a separate process, not interleaved
with the fork modes. These results do not isolate scheduler cost: operators
can emit different batch sizes, and stock parallel projection also merges
partitions. Skew here is a partition-size imbalance, not unequal cost per
row or per morsel. The results cover only these warm, in-memory integer
workloads, not cold runs, I/O, joins, grouped aggregation, or spill.
