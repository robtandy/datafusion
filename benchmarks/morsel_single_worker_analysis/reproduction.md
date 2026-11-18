<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Reproducing the #144 experiment

Run on the fork workspace. No Apache main comparison is needed: each `off` / `required` pair runs in the **same executable**, with identical session and Tokio settings apart from the mode. The baseline is [PR #141](https://github.com/robtandy/datafusion/pull/141) at `f05fbe99e5d4bdd0e452eef7a44663c0c16873e5` (runtime identical to `morsel-execution` at `3dad9ce14977da4bb36a8c0d4c22c92c87a8ca25`).

## Build and run the unmodified harness

```bash
pin=f05fbe99e5d4bdd0e452eef7a44663c0c16873e5
git fetch origin refs/pull/141/head
if ! git cat-file -e "$pin^{commit}"; then git fetch origin "$pin" || exit 1; fi
scratch=$(mktemp -d "$HOME/.morsel-144.XXXXXX")
git worktree add --detach "$scratch/harness" "$pin"
git worktree add --detach "$scratch/experiment" "$pin"

build() (
    cd "$scratch/$1" || exit 1
    export CARGO_TARGET_DIR="$scratch/target-$2"
    export RUSTC_WRAPPER="$HOME/.local/bin/sccache" # optional cache
    timeout 1500 cargo bench --locked --profile release-nonlto -j 12 \
        -p datafusion --no-default-features --features sql \
        --bench morsel_execution --no-run
)
bench_binary() {
    find "$scratch/target-$1/release-nonlto/deps" -maxdepth 1 \
        -type f -executable -name 'morsel_execution-*'
}

build harness baseline
base=$(bench_binary baseline) # expect exactly one executable in the fresh target
sha256sum "$base"
timeout 120 taskset -c 12-15 "$base" --verify
for round in 1 2; do
    timeout 120 taskset -c 12-15 "$base" --bench >"$scratch/pr141-$round.out" 2>"$scratch/pr141-$round.err"
done
```

The fetched PR head matched `pin` when the experiments started. PR #141 later added same-mode priming before each timed sample in `366d34a61`; that revision was not used here. Do not time `cargo bench` while it is compiling; the commands above finish the build first. Start the build under a background wrapper with a log when running unattended. Run `uptime` and `ps -eo pid,psr,pcpu,comm --sort=-pcpu` before and after each sweep; sample `/proc/stat` and `/proc/loadavg` once per second from outside the benchmark's CPU set. Use differences in `/proc/stat` jiffies to check per-CPU busy and steal time, including a short idle interval before starting the sweep. No system tuning is needed.

## Scratch ablation recipe

The extended run was made in the detached `experiment` checkout, not in the submission worktree. The exact scratch diff and run logs were retained locally on `dduck2` at `$HOME/.morsel-144.KjTdNs/` (`experiment.patch`, `run.py`, `*.out`, `*.err`, and `*-load.log`). On that workspace, apply the diff with `git -C "$scratch/experiment" apply "$HOME/.morsel-144.KjTdNs/experiment.patch"`; its SHA-256 is `5ea54ccd15c65c0d8fe9463ee75b89bb3151e5e97ddb4c6f37ce1fa151abf5c8`. To reconstruct the experiment without those files:

- Keep the #141 fixture, plans, alternating order of `off` and `required`, stream drain, five warmups and 31 repetitions. Override both input batch rows and `SessionConfig::with_batch_size` from `MORSEL_BATCH_ROWS` (default `4096`); read the number of input batches from `MORSEL_BATCHES` (default `2048`, or `16` for verification). Select all layouts or only layouts of size `MORSEL_CASE` (either `1` or `4`). Add the unfiltered projection `SELECT id + 1 AS result FROM items` as the third query: its oracle expects all IDs exactly once and no filter in the native plan. For each planned mode, run the oracle once untimed and record the number of non-empty batches, min/median/max rows per batch and number exactly equal to the configured size. Keep the original checks for native worker count, filter presence on filtered queries, and `CountSink` only for count. Check the total received row count on every timed execution.
- Select the following native-only variants by `MORSEL_EXPERIMENT`, read once. Dispatch to separate `run_worker` monomorphizations (const booleans for yield cadence and pruning) **before** looping. `baseline` is the original behavior, `budget` moves the existing yield outside the inner batch loop to the end of each claimed morsel (including a partial final morsel); neither changes cancellation checks, channel capacity, or batch shape.
- `inline` is a one-worker diagnostic compared with `budget`. In `MorselExec::execute`, return a `RecordBatchStreamAdapter` wrapping `stream::try_unfold((Arc<Pipeline>, 0), ...)` instead of calling `Pipeline::start`. For a result sink, walk inputs by index, read and apply the same operators, and return the next non-empty batch on demand; yield before every next group of four and once at EOF for a non-empty input. For count, consume `source.batches.chunks(BATCHES_PER_MORSEL)` in the first poll/future, using the same read, operators, and `CountState::add_rows`, yielding after each chunk. Finalize and apply the same result operators and then return EOF on the next call. This removes the supervisor, worker, joins, result channel, atomic claims, and producer cancellation checks together; it is not a proposed topology change.
- `prune` isolates early projection for the filtered projection fixture. Evaluate the predicate on the original two-column batch, `input.project(&[0])`, `filter_record_batch` on that one-column batch using the original Boolean mask, then apply the trailing projector. The expression still reads `id@0`. Keep per-input yields and sends; don't apply this to count or simple projection. This fixture-specific column index is not a general optimization.
- `material` keeps the original yield cadence and topology. In the projection worker, create one Arrow `BatchCoalescer` per worker with the **two-column** source schema and session batch size. Evaluate the same predicate on each input; call `push_batch_with_filter(input, mask)`, apply the original trailing expression to completed batches and send them. At EOF call `finish_buffered_batch()` and drain the tail. This changes both materialization and the frequency of output projection/channel sends; unlike stock it still copies the predicate column. In the count worker, the plan is a single `Filter` optionally followed by empty `Projection[]`. Evaluate that predicate and accumulate `as_boolean_array(mask.as_ref())?.true_count()` directly instead of materializing the selected rows. Keep the existing `CountSink` combine/finalize and single-row result handoff. Assert the expected operator shape; don't apply this shortcut through arbitrary operators. `both` runs `material` with the same once-per-morsel yield as `budget`. With no filter, `material` and `both` just use the corresponding ordinary worker.
- For `MORSEL_COUNTERS=1` on the one-worker cases, enable `nix`'s `resource` dev feature and call `getrusage(RUSAGE_THREAD)` from the Tokio worker immediately before the start timestamp and after the total timestamp. Subtract user/system CPU microseconds and voluntary/involuntary OS switches, and take the median of each delta independently. The only Tokio query thread in this layout runs the caller, worker(s), and supervisor. Do not interpret per-thread counters this way for four workers; those runs did not collect them.

Build after applying the scratch changes, using a different target directory; then run the following without a concurrent build. `run.py` wraps each call as below and names the logs `r<round>-<workers>-<batch_rows>-<variant>` (or `startup-<batches>-<variant>`). It pins the monitor to CPU 0.

```bash
build experiment experiment
exp=$(bench_binary experiment)
sha256sum "$exp"
variants=(baseline budget inline material both prune)
reverse=(prune both material inline budget baseline)

for variant in "${variants[@]}"; do
    MORSEL_EXPERIMENT=$variant timeout 120 taskset -c 12-15 "$exp" --verify
done

for round in 1 2; do
    order=("${variants[@]}")
    if [ "$round" = 2 ]; then order=("${reverse[@]}"); fi
    for rows in 4096 16384; do
        for variant in "${order[@]}"; do
            MORSEL_CASE=1 MORSEL_BATCH_ROWS=$rows MORSEL_BATCHES=$((8388608 / rows)) \
                MORSEL_EXPERIMENT=$variant MORSEL_COUNTERS=1 \
                timeout 120 taskset -c 12 "$exp" --bench
        done
    done
    for variant in "${order[@]}"; do
        case $variant in inline|prune) continue ;; esac
        MORSEL_CASE=4 MORSEL_EXPERIMENT=$variant \
            timeout 120 taskset -c 12-15 "$exp" --bench
    done
done

for n in 1 16; do
    for variant in baseline budget inline; do
        MORSEL_CASE=1 MORSEL_BATCHES=$n MORSEL_EXPERIMENT=$variant \
            timeout 120 taskset -c 12 "$exp" --bench
    done
done
```

The settings above assume no inherited experiment variables or DataFusion overrides. Avoid comparing wall times across machines or timing across separate `off` / `required` builds. Stock is rerun in every invocation as a drift check.

## Minimal count-only confirmation diff

This is an ablation against the pinned #141 checkout, **not a runtime change in this PR**. Unlike the multi-variant experiment it leaves the harness intact. It measures the mask shortcut with the original per-batch yield even in four-worker execution. The fallback keeps the existing operator chain whenever it cannot be replaced by a cardinality-only calculation.

```diff
diff --git a/datafusion/core/src/morsel.rs b/datafusion/core/src/morsel.rs
index b0eedf8bf..fb07310e1 100644
--- a/datafusion/core/src/morsel.rs
+++ b/datafusion/core/src/morsel.rs
@@ -12,6 +12,7 @@ use std::sync::atomic::{AtomicUsize, Ordering};
 use arrow::array::RecordBatch;
 use arrow::datatypes::{DataType, Schema, SchemaRef};
 use async_trait::async_trait;
+use datafusion_common::cast::as_boolean_array;
 use datafusion_common::config::MorselMode;
 use datafusion_common::nested_struct::adapt_batch_to_schema;
 use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion};
@@ -438,6 +439,19 @@ impl Pipeline {
         tx: mpsc::Sender<RecordBatch>,
     ) -> Result<Option<CountState>> {
         let mut count = matches!(self.sink, Sink::Count { .. }).then(CountState::default);
+        let count_filter = if count.is_some() {
+            match self.operators.as_slice() {
+                [Operator::Filter(predicate)] => Some(predicate),
+                [Operator::Filter(predicate), Operator::Projection(projector)]
+                    if projector.projection().as_ref().is_empty() =>
+                {
+                    Some(predicate)
+                }
+                _ => None,
+            }
+        } else {
+            None
+        };
         while !tx.is_closed() {
             let start = next.fetch_add(BATCHES_PER_MORSEL, Ordering::Relaxed);
             if start >= self.source.batches.len() {
@@ -448,11 +462,21 @@ impl Pipeline {
                 if tx.is_closed() {
                     return Ok(None);
                 }
-                let batch = apply_operators(&self.operators, self.source.read(input)?)?;
+                let input = self.source.read(input)?;
                 if let Some(local) = &mut count {
-                    local.add_rows(batch.num_rows())?;
-                } else if batch.num_rows() > 0 && tx.send(batch).await.is_err() {
-                    return Ok(None);
+                    let rows = if let Some(predicate) = count_filter {
+                        let mask =
+                            predicate.evaluate(&input)?.into_array(input.num_rows())?;
+                        as_boolean_array(mask.as_ref())?.true_count()
+                    } else {
+                        apply_operators(&self.operators, input)?.num_rows()
+                    };
+                    local.add_rows(rows)?;
+                } else {
+                    let batch = apply_operators(&self.operators, input)?;
+                    if batch.num_rows() > 0 && tx.send(batch).await.is_err() {
+                        return Ok(None);
+                    }
                 }
                 tokio::task::yield_now().await;
             }
```

From the report worktree, extract the diff above and use a third target directory:

````bash
awk '/^```diff$/{keep=1; next} /^```$/{if(keep)exit} keep' \
    benchmarks/morsel_single_worker_analysis/reproduction.md >"$scratch/count-only.patch"
git worktree add --detach "$scratch/count-only" "$pin"
git -C "$scratch/count-only" apply "$scratch/count-only.patch"
build count-only count-only
count_only=$(bench_binary count-only)
sha256sum "$count_only"
timeout 120 taskset -c 12-15 "$count_only" --verify
# As above, finish all builds and let the host settle before timing.
timeout 120 taskset -c 12-15 "$base" --bench
timeout 120 taskset -c 12-15 "$count_only" --bench
timeout 120 taskset -c 12-15 "$count_only" --bench
timeout 120 taskset -c 12-15 "$base" --bench
````

The focused scratch test commands (run outside timing) were:

```bash
(
    cd "$scratch/count-only" || exit 1
    export CARGO_TARGET_DIR="$scratch/target-count-only" RUST_BACKTRACE=1
    export RUSTC_WRAPPER="$HOME/.local/bin/sccache"
    timeout 480 cargo test --locked --profile release-nonlto -j 10 \
        -p datafusion --no-default-features --features sql --test morsel_execution
    timeout 480 cargo test --locked --profile release-nonlto -j 10 \
        -p datafusion --no-default-features --features sql --lib morsel::
    timeout 540 cargo test --locked --profile release-nonlto -j 10 \
        -p datafusion --no-default-features --features sql,parquet,compression --lib morsel::
)
```

The `sql`-only integration target passed all nine tests, including filtered count with NULL predicates, empty input and no passing rows. `sql`-only `--lib` did not compile: unrelated core datasource tests refer to Parquet and compression types behind feature gates. Enabling `parquet,compression` passed all eight `morsel::` unit tests, including worker errors, backpressure, and dropping a count before finalization. Those extra features were only used for the unit-test build; benchmark builds are `sql`-only.
