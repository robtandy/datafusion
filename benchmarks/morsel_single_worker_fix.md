<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Single-worker in-memory morsel fix (#145)

The [#144 diagnosis](morsel_single_worker_analysis.md) found that materializing the unused predicate column explained most of the projection gap, and counting true bits rather than materializing filtered batches closed the count gap. Moving the explicit yield from every batch to every morsel on the **parallel** streaming path more than doubled drain time. This patch keeps that path and its yield cadence instead of moving the yield globally.

- A one-worker in-memory stream runs on demand in the consumer, without the worker task, supervisor or result channel. It yields every four **input** batches, including when a filter rejects all rows. For a filter followed by a projection, the filter evaluates on the full input, then applies the mask only to columns read by the projection. No output is coalesced or buffered ahead of demand.
- For in-memory `COUNT(*)`, only `Filter` and `Filter → Projection[]` feed the validity-aware true count of the mask directly into the worker's local count. More complex inputs still run the full operator chain. The per-batch yield, receiver closure check, worker claims, supervisor, and finalized-only result are unchanged.
- The original native Parquet baseline in [#128](https://github.com/robtandy/datafusion/pull/147) stays pinned to the merged #126 revision `d649cacf00fee7125f45ea1427a023b2eab732b1`. No Parquet harness, stock operator, or Parquet async handoff changed in this patch.

## Reproduce

Use the unmodified [#127 in-memory harness](morsel_execution.md). The pre-fix source is `4c43c428f0b2a6c4c1013875be01093707e063ec`; the post-fix source is `bb3be60a3748f48c46607b6ec0ee7de48b22f280`. Stock is `morsel_mode=off` in each executable, not a separate Apache checkout. In both executables the native comparison uses `morsel_mode=required`; the harness checks the plans, matching schemas, exact projected IDs and the finalized `COUNT(*)` value, then checks output row counts in all timed drains. On this fixture all layouts select 2,094,791 of 8,388,608 rows (2,048 input batches of 4,096 rows). The four-worker balanced and skewed layouts are `[512, 512, 512, 512]` and `[1664, 128, 128, 128]` batches per partition.

From the post-fix checkout, build both variants before timing. The commands below reuse one Cargo target in the **same source checkout** and copy each executable before rebuilding. Do not build or test concurrently with timing.

```bash
base=4c43c428f0b2a6c4c1013875be01093707e063ec
post=bb3be60a3748f48c46607b6ec0ee7de48b22f280
scratch=$(mktemp -d "$HOME/.morsel-145.XXXXXX")
git diff "$base" "$post" -- datafusion/core/src/morsel.rs > "$scratch/runtime.patch"
git worktree add --detach "$scratch/src" "$base"
cd "$scratch/src"
export CARGO_TARGET_DIR="$scratch/target"
# Optional: export RUSTC_WRAPPER="$HOME/.local/bin/sccache"
build() {
    timeout "$1" taskset -c 0-11 cargo bench --locked --profile release-nonlto -j 10 \
        -p datafusion --no-default-features --features sql --bench morsel_execution --no-run
}
build 1500
# Expect exactly one matching executable in the fresh target directory.
bench=$(find "$CARGO_TARGET_DIR/release-nonlto/deps" -maxdepth 1 \
    -type f -executable -name 'morsel_execution-*')
cp "$bench" "$scratch/before"
git apply "$scratch/runtime.patch"
build 900
cp "$bench" "$scratch/after"
sha256sum "$scratch/before" "$scratch/after"
# Wait until CPUs 12–15 are idle before timing. Sample /proc/stat and /proc/loadavg during the runs.
for stage in before after after before; do
    timeout 120 taskset -c 12-15 "$scratch/$stage" --verify
    timeout 120 taskset -c 12-15 "$scratch/$stage" --bench
done
```

The drain runs on the same Tokio pool as the query (one or four threads), with the entire process pinned to CPUs 12–15 for both engines and binaries. The harness uses five warmups, 31 timed samples per mode, alternating `off` and `required` within each case, and primes each sample with an untimed drain of the same plan and mode. Planning and verification are outside the timing. First result means first non-empty batch; for `COUNT(*)` this is the finalized row. Total runs to stream EOF.

## Results

Measured 2026-11-21 UTC on `rob-tandy-dduck1` (Linux `7.0.0-1014-aws`, 16 aarch64 Neoverse-N1 vCPUs, 61 GiB RAM) with the pinned Rust/Cargo 1.99.1, Clang/lld 18.1.3, locked Arrow 60.1.0 and Tokio 1.54.1. Neither the build flags nor the harness source changed between executables. No builds or tests ran during the timed sweep. In the ten seconds before the sweep, CPUs 12–15 averaged 3–5% busy individually, with zero recorded steal on those CPUs throughout. The one-minute load average was 7.56 at the first launch and 5.23 at the end. This is a quiet interval on a shared VM, not an isolated host.

`--verify` and the full-fixture correctness checks passed in every invocation. The patched one-worker projection plan included `MorselExec: workers=1, MemTable[items, projection=Some([0, 1])] -> Filter[score@1 < 64, projection=[0]] -> Projection[id@0 + 1 AS result] -> ResultSink[inline]`; four-worker streaming kept the original `ResultSink[capacity=2]`. Count still displayed `CountSink; FinalizedCountSource` in all native cases.

Medians in ms, **round 1 / round 2**. Round 1 ran before then after; round 2 reversed that order. Stock is shown from **both** binaries to make host/build drift visible.

| Layout     | Query      | Metric    | Stock (before binary) | Stock (after binary) |   Native before |    Native after |
| ---------- | ---------- | --------- | --------------------: | -------------------: | --------------: | --------------: |
| 1 worker   | Projection | First     |         0.273 / 0.302 |        0.298 / 0.274 |   0.028 / 0.030 |   0.021 / 0.019 |
| 1 worker   | Projection | Total     |       18.947 / 19.328 |      19.113 / 18.486 | 25.856 / 26.204 | 18.046 / 17.719 |
| 1 worker   | Count      | Finalized |         8.823 / 9.321 |        9.201 / 8.792 | 13.570 / 13.723 |   8.253 / 8.220 |
| 1 worker   | Count      | Total     |         8.828 / 9.326 |        9.206 / 8.797 | 13.572 / 13.726 |   8.256 / 8.223 |
| 4 balanced | Projection | First     |         0.267 / 0.265 |        0.279 / 0.261 |   0.040 / 0.042 |   0.043 / 0.041 |
| 4 balanced | Projection | Total     |         6.575 / 6.655 |        6.701 / 6.546 |   7.493 / 7.454 |   7.530 / 7.445 |
| 4 balanced | Count      | Finalized |         2.583 / 2.668 |        2.660 / 2.544 |   3.900 / 3.875 |   2.402 / 2.367 |
| 4 balanced | Count      | Total     |         2.586 / 2.672 |        2.663 / 2.547 |   3.903 / 3.878 |   2.404 / 2.370 |
| 4 skewed   | Projection | First     |         0.199 / 0.207 |        0.208 / 0.200 |   0.039 / 0.041 |   0.042 / 0.041 |
| 4 skewed   | Projection | Total     |       17.495 / 17.999 |      18.000 / 17.408 |   7.400 / 7.491 |   7.531 / 7.473 |
| 4 skewed   | Count      | Finalized |         7.788 / 8.137 |        8.067 / 7.725 |   3.835 / 3.844 |   2.391 / 2.368 |
| 4 skewed   | Count      | Total     |         7.791 / 8.140 |        8.070 / 7.727 |   3.838 / 3.847 |   2.393 / 2.371 |

- A separate pre-fix run also reproduced the regression: projection took 25.984 ms native vs. 19.015 ms stock; count took 13.547 vs. 8.890 ms.
- The patch cut one-worker native projection total by **30–32%** and count total by **39–40%**. Both were faster than the stock mode in the patched binary on this fixture (by 4–6% and 7–10%, respectively). For scale, round-1 one-worker p10–p90 totals for projection were 25.726–26.029 ms before and 17.848–18.143 ms after; for count they were 13.507–13.679 ms and 8.228–8.328 ms. The percentiles describe these samples, not confidence intervals.
- Native first projection got faster at one worker. At four workers the first projection medians moved by −1 to +3 µs and the total changed by at most 1.8% vs. native before; the timing does not resolve a small effect from scheduling or binary layout. Native skewed projection and count still took substantially less time than stock in both rounds. Balanced projection still trails stock by 12–14%; this patch does not optimize the parallel streaming path.

## Progress, cancellation, and limits

The regression test first failed because the first poll waited for a spawned worker instead of returning the ready in-memory batch. After the fix, a paused consumer triggers no additional scans, and dropping the stream releases the pipeline immediately. A second runnable query makes progress after at most four input batches when the first drain repeatedly receives batches **and** when all its batches are rejected. Another test observes one execution error followed by EOF with no more input reads. The existing shared worker, full-sink cancellation, Parquet I/O, and count finalization/drop/error tests still pass. SQL parity tests cover nullable predicates, empty input, zero qualifying rows, projected column remapping, and a constant projection with no kept columns.

The yield bound is in input batches, not wall time; a synchronous batch or expensive expression cannot be preempted in the middle. Streaming first-batch timings compare **first non-empty** batches, not equivalent row counts: stock can coalesce rows across inputs, unlike this native path. These timings cover only the warm, in-memory fixture and one host; they do not isolate projection pruning from inline execution or predict I/O, other selectivities, or other CPUs.

### Mixed-query check

To complement the deterministic tests, run the [#137 mixed-query harness](morsel_mixed/README.md) in the detached scratch checkout from the reproduction steps. In `benchmarks/morsel_mixed/src/workload.rs`, switch `MorselMode::Off` to `MorselMode::Required` and add `ensure!(plan.name() == "MorselExec", "expected native execution");` immediately after `df.create_physical_plan().await?`. In `src/main.rs` in the same harness, change the metadata `backend` to `"datafusion_morsel"` and `morsel_mode` to `"required"`. Apply those same scratch changes to both binaries; no harness source is changed in the patch.

```bash
cd "$scratch/src"
git restore --source="$base" -- datafusion/core/src/morsel.rs
export CARGO_TARGET_DIR="$scratch/mixed-target"
export CARGO_PROFILE_RELEASE_CODEGEN_UNITS=16 CARGO_PROFILE_RELEASE_LTO=false
build_mixed() {
    timeout 900 taskset -c 0-11 cargo build --release --locked -j 6 \
        --manifest-path benchmarks/morsel_mixed/Cargo.toml
}
build_mixed
cp "$CARGO_TARGET_DIR/release/morsel-mixed" "$scratch/mixed-before"
git apply "$scratch/runtime.patch"
build_mixed
cp "$CARGO_TARGET_DIR/release/morsel-mixed" "$scratch/mixed-after"
for stage in before after after before; do
    timeout 60 taskset -c 12-15 "$scratch/mixed-$stage" \
        --workers 1 --blocking-threads 2 --partitions 1 --batch-size 4096 \
        --seed 42 --small-rows 4096 --large-rows 8388608 --background-lanes 1 \
        --warmup 2 --repetitions 3 --requests 300 --interval-ms 5 \
        --queue-capacity 16 --overdue-ms 20 --timeout-ms 10000 \
        --large-sql 'SELECT v + 1 FROM large WHERE v >= 0' --large-expected-rows 8388608
done
```

One foreground filtered count ran on the same Tokio worker as one lane of repeated background projections. Each phase completed all 300 offers with the correct result (236), no drops, errors, overdue requests, or offers a full interval late. Between 290 and 296 requests per mixed phase started with a background query in flight. On the same host the one-minute load average fell from 2.18 to 1.64 during the sweep.

Ranges across the six repetitions per engine and condition (ms):

| Condition | Native before p95 | Native after p95 | Native before p99 | Native after p99 |
| --------- | ----------------: | ---------------: | ----------------: | ---------------: |
| Isolated  |       0.560–0.805 |      0.541–0.642 |       0.576–1.061 |      0.550–0.703 |
| Mixed     |       0.627–0.798 |      0.768–1.177 |       1.212–1.344 |      1.252–1.377 |

Mixed p95 rose as the lane increased from 50.7–55.3 to 64.0–67.3 completed background scans/s; p99 ranges overlapped. The patched stream yields at four-batch boundaries and the background lane starts another scan immediately, so this holds the offered foreground rate and lane count fixed, not background throughput. It does not separate those effects. Latency includes the paced offer, queue, planning and execution, and is a progress check at one offered rate, not an isolated scheduler measurement.
