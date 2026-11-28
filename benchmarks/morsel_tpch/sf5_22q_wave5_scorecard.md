<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# TPC-H SF5 Wave-5 22/22 3-Way Scorecard: Post Stock-Gap Fixes (`#217`, `#218`, `#219`) (#220)

This scorecard reports the **Wave-5 complete 22/22 standard TPC-H (`q1`–`q22`)
3-way comparison at Scale Factor 5 (`--sf 5`)** between **Morsel DataFusion
(`morsel_mode=required`)**, **Stock DataFusion (`morsel_mode=off`, both
`stock_default` and `stock_pushdown`)**, and **DuckDB 2.0.1** on quiet
`rob-tandy-dduck2` (8 pinned ARM Neoverse-N1 cores, verified 100% warm Linux
page cache via `mincore`) after all three Wave-5 Stock-gap fix issues merged
onto `morsel-execution`:

1. **`q9`, `q10`, and `q13` Stock-gap fixes (`#217`):** selectivity/saturation
   Bloom gating, lock-free Level-0 radix-partitioned `SplitBlockBloomFilter`,
   zero-slice `RadixChunk` + parallel table drop, two-sided range & projected
   width build-side estimation, sibling forward dynamic filter propagation
   (`part -> partsupp`), 2-key `FlatIntTable<(i64, i64)>`, build-key probe
   substitution + zero-copy contiguous output, and dense integer `DenseIntTable`
   (`ArrayMap`),
2. **`q14`, `q17`, and `q20` Stock-pushdown-gap fixes (`#218`):** infallible
   `JoinBloomFilterExpr` + redundant dynamic range skip in buffered `RowFilter`,
   `2×` two-phase `RowFilter` ready-queue capacity, Parquet-statistics interval
   selectivity in build-side selection + scoped wave-1 `RowFilter` policy,
   transitive equijoin key substitution across `Inner`/`BuildSemi`, and higher
   L1-resident small-build Bloom density + `skip_bulk_probe` fast path, and
3. **`q4`, `q11`, `q16`, and `q22` Stock-gap fixes (`#219`):** widened
   `ensure_stage` channel capacity (`4 → 16`), batch-inlined flat-int
   `BuildSemi`/`BuildAnti` probe, L2-resident `RegisterBloom64` prefilter,
   cache-line-aligned NEON `SplitBlockBloomFilter`, concurrent small-build waves
   and a 1-worker fast path for 1-RG builds, compact-span `DenseIntTable` down
   to 1% density, and scoped `BulkDecode` policies for small-build,
   all-predicate radix, and narrow-payload scans.

It re-runs the exact `#209` 6-arm warm-cache protocol on the identical SF5
Parquet dataset and host, comparing Wave-5 directly against Wave-4
([sf5_22q_wave4_scorecard.md](sf5_22q_wave4_scorecard.md), `#209`, without
overwriting it) as well as the earlier `#195`
([sf5_22q_final_scorecard.md](sf5_22q_final_scorecard.md)) and `#192`
([sf5_22q_scorecard.md](sf5_22q_scorecard.md)) milestones.

## Executive Summary

- **Single-config Morsel (`morsel_pushdown`, `morsel_mode=required`) achieves a
  `22 / 22` clean sweep over BOTH `stock_default` AND `stock_pushdown` — and
  closes within `5.4%` of DuckDB 2.0.1 across the full 22/22 TPC-H SF5 suite:**
  - **22/22 Geometric Mean:** drops **`391.2 → 318.0 ms`** (**`1.23×` faster**
    vs. Wave-4 `#209`, `−73.2 ms` / `−18.7%`; **`2.25×` faster** vs. `#192`'s
    `716.3 ms`), beating **`stock_default` (`458.3 ms`) at `0.69×` (`1.44×`
    faster)**, beating **`stock_pushdown` (`421.0 ms`) at `0.76×` (`1.32×`
    faster)**, and slashing the gap to **DuckDB 2.0.1 (`301.8 ms`)** from
    **`1.26× → 1.05×`** (`+16.2 ms`, down from `2.36×` in `#192` and `1.48×` in
    `#195`).
  - **22/22 Sum of Medians:** drops **`10,319.9 → 8,232.2 ms`** (**`1.25×`
    faster**, **`−2,087.7 ms` / `−2.09 s`** vs. Wave-4 `#209`; `−15.01 s` /
    `2.82×` vs. `#192`'s `23,240.9 ms`), beating **`stock_default`
    (`12,885.6 ms`) at `0.64×` (`1.57×` faster, `−4.65 s`)**, beating
    **`stock_pushdown` (`11,580.1 ms`) at `0.71×` (`1.41×` faster, `−3.35 s`)**,
    and narrowing the sum gap to **DuckDB 2.0.1 (`7,651.6 ms`)** from
    **`1.32× → 1.08×`** (`+580.6 ms` total across all 22 queries — of which
    `q15` accounts for `+414.5 ms` / `71.4%`, mostly from its un-CSE'd double
    `lineitem` scan; across the other **21 / 22 queries**, `morsel_pushdown` is
    at **`1.01×` geomean (`306.7` vs. `303.6 ms`)** and **`1.02×` sum
    (`7,552.4` vs. `7,386.3 ms`)** of DuckDB 2.0.1).
  - **Even non-pushdown Morsel beats `stock_default`:** Both
    `morsel_no_pushdown` (`432.0 ms`, `1.26×` faster vs. Wave-4) and
    `morsel_pruning` (`431.6 ms`, `1.27×` faster vs. Wave-4) now beat
    `stock_default` (`458.3 ms`) at **`0.94×`**.
  - **Oracle gap shrinks further to `2.0 ms` (`0.64%`):** Single-config
    `morsel_pushdown` (`318.0 ms` geomean / `8,232.2 ms` sum) trails the
    per-query `Best Morsel` oracle (`315.9 ms` / `8,182.6 ms`) by just `2.0 ms`
    (`0.64%`, down from `3.0 ms` in Wave-4 and `17.8 ms` in `#195`).
    `morsel_pushdown` is the fastest Morsel arm on **`16 / 22`** queries and
    within `0.6–4.6%` on the remaining 6.
- **Head-to-head query wins for single-config `morsel_pushdown`:**
  - Beats **`stock_default` on `22 / 22` queries** (up from `16 / 22` in Wave-4
    `#209`): every query in `q1`–`q22`, ranging from `0.33×` (`q17`) to `0.97×`
    (`q22`).
  - Beats **`stock_pushdown` on `22 / 22` queries** (up from `15 / 22` in Wave-4
    `#209`): every query in `q1`–`q22`, ranging from `0.50×` (`q21`) to `0.94×`
    (`q20`).
  - Beats **DuckDB 2.0.1 on `11 / 22` queries** (up from `2 / 22` in Wave-4
    `#209` — half the TPC-H suite): **`q13` (`0.67×`)**, **`q14` (`0.81×`)**,
    **`q22` (`0.82×`)**, **`q4` (`0.88×`)**, **`q12` (`0.89×`)**, **`q16`
    (`0.90×`)**, **`q7` (`0.91×`)**, **`q3` (`0.94×`)**, **`q17` (`0.96×`)**,
    **`q20` (`0.96×`)**, and **`q8` (`0.99×`)** — and lands within **2–9% of
    DuckDB 2.0.1** on 5 more (`q21` `1.02×`, `q5` `1.08×`, `q9` `1.08×`, `q10`
    `1.08×`, `q19` `1.09×`).
  - **Zero regressions vs. Wave-4 (19 / 22 genuinely faster beyond host
    drift):** All 10 targeted Stock-gap queries (`q4`, `q9`, `q10`, `q11`,
    `q13`, `q14`, `q16`, `q17`, `q20`, `q22`) dropped **`4,434.3 → 3,090.5 ms`**
    combined (**`1.43×` faster**, `−1,343.8 ms`; **`1.35×`** on 10Q geomean),
    flipping to beat BOTH Stock arms on all `10 / 10` queries and beating DuckDB
    2.0.1 as a 10-query subset (**`0.92×` geomean / `0.93×` sum**). Among the 12
    non-targeted queries (`5,885.6 → 5,141.7 ms`, `−743.9 ms` raw), 9 join
    queries achieved genuine Wave-5 speedups beyond host drift (`q21`
    `−208.0 ms`, `q8` `−121.4 ms`, `q18` `−99.6 ms`, `q3` `−69.0 ms`, `q7`
    `−68.8 ms`, `q5` `−40.2 ms`, `q19` `−39.5 ms`, `q2` `−25.5 ms`, `q12`
    `−13.1 ms`), while `q1` (`−3.3%`), `q6` (`−4.0%`), and `q15` (`−3.5%`) moved
    within the `2.5–3.8%` session-to-session host drift observed on the
    unchanged reference arms.
- **100% native admission, 100% DuckDB parity, 100% warm cache, zero output
  backpressure:** All `22 / 22` standard TPC-H queries passed lossless Arrow IPC
  parity against **DuckDB 2.0.1** at both `--sf 0.01` (`21 / 21` `morsel_tpch`
  unittests passing) and `--sf 5` across all 6 arms, ran strictly under
  `MorselExec: workers=8` with `morsel_mode=required` (`0` fallbacks, `0`
  rejections), verified **100% `mincore` page residency on all `660 / 660`
  measured trials**, and recorded **`0.0 ms` `morsel_output_wait`** across all
  330 measured Morsel trials.

| Arm                                                | `#192` GeoMean | `#195` GeoMean | Wave-4 GeoMean | Wave-5 GeoMean | `W4 → W5` Gain | Wave-5 Sum (22/22) | vs. `stock_default` (GM / Sum) | vs. `stock_pushdown` (GM / Sum) | vs. `duckdb_default` (GM / Sum) |
| -------------------------------------------------- | -------------: | -------------: | -------------: | -------------: | -------------: | -----------------: | -----------------------------: | ------------------------------: | ------------------------------: |
| Stock DataFusion (`stock_default`, `off`)          |       462.6 ms |       474.2 ms |       471.3 ms |       458.3 ms |          1.03× |        12,885.6 ms |                  1.00× / 1.00× |                   1.09× / 1.11× |                   1.52× / 1.68× |
| Stock DataFusion (`stock_pushdown`, `off`)         |              — |       434.2 ms |       435.7 ms |       421.0 ms |          1.03× |        11,580.1 ms |                  0.92× / 0.90× |                   1.00× / 1.00× |                   1.40× / 1.51× |
| Morsel DataFusion (`morsel_no_pushdown`, `req`)    |       690.7 ms |       563.9 ms |       546.0 ms |       432.0 ms |      **1.26×** |        12,303.9 ms |              **0.94× / 0.95×** |                   1.03× / 1.06× |                   1.43× / 1.61× |
| Morsel DataFusion (`morsel_pruning`, `req`)        |              — |       568.4 ms |       550.1 ms |       431.6 ms |      **1.27×** |        12,311.1 ms |              **0.94× / 0.96×** |                   1.03× / 1.06× |                   1.43× / 1.61× |
| **Morsel DataFusion (`morsel_pushdown`, `req`)**   |       716.3 ms |       461.0 ms |       391.2 ms |   **318.0 ms** |      **1.23×** |     **8,232.2 ms** |              **0.69× / 0.64×** |               **0.76× / 0.71×** |               **1.05× / 1.08×** |
| **Best Morsel per query (`morsel_mode=required`)** |       679.7 ms |       443.2 ms |       388.1 ms |   **315.9 ms** |      **1.23×** |     **8,182.6 ms** |              **0.69× / 0.64×** |               **0.75× / 0.71×** |               **1.05× / 1.07×** |
| **DuckDB 2.0.1 (`duckdb_default`)**                |       304.0 ms |       311.6 ms |       309.5 ms |   **301.8 ms** |          1.03× |     **7,651.6 ms** |              **0.66× / 0.59×** |               **0.72× / 0.66×** |               **1.00× / 1.00×** |

_(Baseline stability check: all three unchanged reference arms ran `2.5–3.4%`
faster in Wave-5 than in Wave-4 — `stock_default` `471.3 → 458.3 ms` (`−2.8%`),
`stock_pushdown` `435.7 → 421.0 ms` (`−3.4%`), and `duckdb_default`
`309.5 → 301.8 ms` (`−2.5%`). Pre-run `vmstat` idle was `92–93%` in both waves,
while post-run idle settled at `99–100%` in Wave-5 vs. `93%` in Wave-4. Because
all 6 arms ran counterbalanced in the same session, the head-to-head ratios are
immune to that `~2.5–3.4%` session-to-session host drift and confirm a genuine
`~1.19–1.20×` normalized geomean gain (`morsel_pushdown / stock_default`:
`0.83× → 0.69×`; `morsel_pushdown / stock_pushdown`: `0.90× → 0.76×`;
`morsel_pushdown / duckdb_default`: `1.26× → 1.05×`).)_

- **Wave-5 pillar-by-pillar headline gains (single-config `morsel_pushdown`,
  Wave-4 `#209` → Wave-5 `#220`):**
  - **`#217` (`q9`, `q10`, `q13` — `2,604.2 → 1,654.7 ms`, `1.57×`,
    `−949.5 ms`):**
    - **`q9`:** `1,397.5 → 830.8 ms` (**`1.68×`**, `−566.6 ms`), `join_finalize`
      `276.5 → 44.9 ms` (**`6.16×`**), `process_work` `2,268.5 → 1,129.7 ms`
      (**`2.01×`**), exec CPU `−33.8%`. Flips from `1.37×` behind
      `stock_default` to **`0.82×` (`1.22×` faster)** and **`0.56×` vs.
      `stock_pushdown`**, landing within `1.08×` of DuckDB 2.0.1 (`771.5 ms`).
    - **`q13`:** `590.7 → 344.6 ms` (**`1.71×`**, `−246.1 ms`), `process_work`
      `1,732.3 → 479.4 ms` (**`3.61×`**), exec CPU `−34.7%`. Flips from
      `1.23×` behind `stock_default` and `1.02×` behind `stock_pushdown` to
      **`0.73×` vs. `stock_default`**, **`0.65×` vs. `stock_pushdown`**, and
      **`0.67×` vs. DuckDB 2.0.1 (`1.50×` faster than DuckDB!)**.
    - **`q10`:** `616.0 → 479.3 ms` (**`1.29×`**, `−136.7 ms`), `join_finalize`
      `100.0 → 20.1 ms` (**`4.98×`**), exec CPU `−18.2%`. Flips from `1.17×`
      behind `stock_pushdown` to **`0.92×` vs. `stock_pushdown`** and **`0.75×`
      vs. `stock_default`**, within `1.08×` of DuckDB (`444.4 ms`).
  - **`#218` (`q14`, `q17`, `q20` — `1,170.2 → 930.9 ms`, `1.26×`,
    `−239.3 ms`):**
    - **`q17`:** `499.8 → 382.9 ms` (**`1.31×`**, `−116.9 ms`), `decode_work`
      `3,153.7 → 2,486.1 ms` (`−667.6 ms`), `ready_wait` `136.5 → 57.1 ms`
      (`2.39×`), adapt `480/8 → 488/0`. Flips from `1.07×` behind
      `stock_pushdown` to **`0.84×` vs. `stock_pushdown`**, **`0.33×` vs.
      `stock_default`**, and **`0.96×` vs. DuckDB 2.0.1 (`400.5 ms`)**.
    - **`q14`:** `341.1 → 264.1 ms` (**`1.29×`**, `−77.0 ms`), `join_finalize`
      `35.7 → 2.9 ms` (**`12.3×`**), `ready_wait` `114.8 → 36.1 ms` (`3.18×`).
      Flips from `1.13×` behind `stock_pushdown` to **`0.90×` vs.
      `stock_pushdown`**, **`0.66×` vs. `stock_default`**, and **`0.81×` vs.
      DuckDB 2.0.1 (`326.8 ms`)**.
    - **`q20`:** `329.3 → 283.9 ms` (**`1.16×`**, `−45.4 ms`), `dyn_filters`
      `12 → 15`, `process_work` `18.3 → 4.2 ms` (`4.36×`), `ready_wait`
      `174.9 → 65.1 ms` (`2.69×`), adapt `276/6 → 282/0`. Flips from `1.02×`
      behind `stock_pushdown` to **`0.94×` vs. `stock_pushdown`**, **`0.61×`
      vs. `stock_default`**, and **`0.96×` vs. DuckDB 2.0.1 (`297.1 ms`)**.
  - **`#219` (`q4`, `q11`, `q16`, `q22` — `660.0 → 504.9 ms`, `1.31×`,
    `−155.1 ms`):**
    - **`q4`:** `226.5 → 163.7 ms` (**`1.38×`**, `−62.8 ms`), `decode_work`
      `1,125.2 → 639.0 ms` (`−486.2 ms`), exec CPU `−25.5%`, adapt
      `247/57 → 2/302`. Flips from `1.13×` behind `stock_default` to **`0.81×`
      vs. `stock_default`**, **`0.57×` vs. `stock_pushdown`**, and **`0.88×`
      vs. DuckDB 2.0.1 (`185.5 ms`)**.
    - **`q16`:** `172.0 → 123.4 ms` (**`1.39×`**, `−48.6 ms`), `ready_wait`
      `119.5 → 31.6 ms` (`3.78×`; `stage_0_build_1` `91.6 → 1.2 ms`), adapt
      `39/2 → 29/12`. Flips from `1.30×` / `1.20×` behind Stock to **`0.94×`
      vs. `stock_default`**, **`0.87×` vs. `stock_pushdown`**, and **`0.90×`
      vs. DuckDB 2.0.1 (`137.5 ms`)**.
    - **`q11`:** `135.2 → 103.2 ms` (**`1.31×`**, `−32.1 ms`), `decode_work`
      `620.1 → 479.1 ms`, `process_work` `114.5 → 68.1 ms`, adapt
      `38/30 → 4/64`. Flips from `1.20×` / `1.06×` behind Stock to **`0.91×`
      vs. `stock_default`** and **`0.85×` vs. `stock_pushdown`**.
    - **`q22`:** `126.3 → 114.7 ms` (**`1.10×`**, `−11.6 ms`), `decode_work`
      `528.0 → 403.1 ms`, exec CPU `−15.6%`, adapt `74/2 → 12/64`. Flips from
      `1.02×` behind `stock_default` to **`0.97×` vs. `stock_default`**,
      **`0.83×` vs. `stock_pushdown`**, and **`0.82×` vs. DuckDB 2.0.1
      (`140.7 ms`)**.

## Environment, Controls, and Reproduction Commands

### Host and Build Metadata

- **Host:** `rob-tandy-dduck2`, Ubuntu 24.04.5 LTS, Linux `7.0.0-1012-aws`,
  16 ARM Neoverse-N1 cores (1 thread/core, 64 KiB L1d / 1 MiB L2 per core,
  32 MiB shared L3), 61 GiB RAM.
- **CPU pinning and threads:** `taskset -c 0-7` (8 dedicated CPUs),
  `--threads 8` (Tokio `worker_threads=8`, `max_blocking_threads=8`,
  `target_partitions=8`; DuckDB `SET threads = 8`), batch size `8,192`.
- **Host idle verification (`vmstat -y 1 3`):**
  - Immediately before 22/22 scorecard: **92–93% idle** (`us 7%, sy 0–1%`).
  - Immediately after 22/22 scorecard: **99–100% idle** (`us 0–1%, sy 0%`).
- **Build & versions:**
  - DataFusion revision: `00b594f0ee7932d13a153f90c0d486de4e58dece`
    (`morsel-execution` HEAD after `#217` / PR `#224`, `#218` / PR `#225`, and
    `#219` / PR `#226`), built with `--locked --profile release-nonlto -p
datafusion --no-default-features --features
parquet,sql,datetime_expressions,unicode_expressions --example
morsel_tpch_runner`.
  - Rust `1.99.1` (`58cf8f9c3 2026-10-13`, `aarch64-unknown-linux-gnu`), Cargo
    `1.99.1`, Python `3.12.10`, DuckDB `2.0.1`, PyArrow `26.0.1`, NumPy `2.5.5`.
- **Dataset & page-cache verification:** Identical deterministic SF5 TPC-H
  Parquet dataset (`43,299,825` rows, `35` Zstd level-3 files, `353` row groups
  at `row_group_size=131,072`, `1,400,570,073` bytes / `1,335.7 MiB` on local
  NVMe Btrfs storage under `$HOME/.cache/issue189/tpch_sf5`). All
  **`660 / 660` measured trials** started and ended with **100% of 4 KiB Parquet
  file pages resident** via Linux `mincore`.
- **Protocol:** Exact `#209` protocol — fresh process per trial, `1` unmeasured
  warmup round + `5` measured counterbalanced rounds (alternating arm order
  between odd and even rounds), `MORSEL_METRICS=1` enabled on every Morsel
  trial, lossless Arrow IPC parity verified against DuckDB 2.0.1 across all 6
  arms before timing. All timings below are medians in milliseconds with
  observed `min–max` in parentheses.

### Exact Reproduction Commands

```bash
set -euo pipefail
b=benchmarks/morsel_tpch
work="$HOME/.cache/issue220"
mkdir -p "$work"
revision=$(git rev-parse HEAD)

# 1. Build detached tpch_runner with full TPC-H expression features.
git worktree add --detach "$work/pinned" "$revision"
mkdir -p "$work/pinned/datafusion/core/examples"
cp "$b/tpch_runner.rs" "$work/pinned/datafusion/core/examples/morsel_tpch_runner.rs"
(
  cd "$work/pinned"
  MORSEL_BENCH_REVISION="$revision" CARGO_TARGET_DIR="$work/target" \
    RUSTC_WRAPPER=sccache \
    timeout --kill-after=20s 35m cargo build --locked --profile release-nonlto -j 10 \
      -p datafusion --no-default-features \
      --features parquet,sql,datetime_expressions,unicode_expressions \
      --example morsel_tpch_runner
)
runner="$work/target/release-nonlto/examples/morsel_tpch_runner"

# 2. Create pinned Python venv and run all 21/21 SF=0.01 parity/plan/spill/adaptive-metric unittests.
timeout 120s uv venv --python 3.12.10 "$work/venv"
timeout 180s uv pip install --python "$work/venv/bin/python" -r "$b/requirements.txt"
py="$work/venv/bin/python"
MORSEL_TPCH_RUNNER="$runner" MORSEL_BENCH_REVISION="$revision" \
  timeout 240s "$py" -B -m unittest discover -s "$b" -v

# 3. Reuse deterministic SF5 TPC-H Parquet tables outside Git (or generate fresh).
data_dir="$HOME/.cache/issue189/tpch_sf5"
# timeout 600s "$py" -B "$b/harness.py" generate "$data_dir" --sf 5 --row-group-size 131072

# 4. Run quiet-host warm-cache SF5 22/22 6-arm scorecard (660 timed trials).
vmstat -y 1 3
timeout --kill-after=10s 2400s taskset -c 0-7 "$py" -B "$b/harness.py" bench \
  "$data_dir" --runner "$runner" --revision "$revision" \
  --threads 8 --timeout 180 --cache warm --warmups 1 --repeats 5 \
  --queries all \
  --morsel-arms stock_default,stock_pushdown,morsel_no_pushdown,morsel_pruning,morsel_pushdown,duckdb_default \
  > "$work/sf5_22q_wave5_scorecard.jsonl"
vmstat -y 1 3

# 5. Print summary and 3-way geometric mean table.
"$py" -B "$b/harness.py" summarize "$work/sf5_22q_wave5_scorecard.jsonl"

# 6. Clean up the detached worktree when finished.
git worktree remove --force "$work/pinned"
```

## 1. Full 22/22 TPC-H SF5 3-Way Warm-Cache Scorecard (Wave-5)

Every query verified exact Arrow IPC parity against DuckDB 2.0.1 across all 6
arms before timing and ran with `morsel_mode=required` (`0` fallback, `100%`
`mincore` page residency on all `660 / 660` trials). In the Morsel columns,
**bold** marks the fastest Morsel arm per query (`morsel_pushdown` wins
`16 / 22`). The ratio columns compare **single-config `morsel_pushdown`**
directly against `stock_default`, `stock_pushdown`, and `duckdb_default`
(`< 1.00×` means single-config Morsel is faster; bold highlights wins — **22/22
vs. `stock_default`, 22/22 vs. `stock_pushdown`, and 11/22 vs. DuckDB 2.0.1**).

| Query               |   Rows | `stock_default` ms (min–max) | `stock_pushdown` ms (min–max) | `morsel_no_pushdown` ms (min–max) | `morsel_pruning` ms (min–max) | `morsel_pushdown` ms (min–max) | `duckdb_default` ms (min–max) | `morsel_pd` / Stock Def | `morsel_pd` / Stock PD | `morsel_pd` / DuckDB |
| ------------------- | -----: | ---------------------------: | ----------------------------: | --------------------------------: | ----------------------------: | -----------------------------: | ----------------------------: | ----------------------: | ---------------------: | -------------------: |
| `q1`                |      4 |          768.7 (762.3–776.6) |           743.7 (731.1–752.9) |           **585.5** (571.9–595.1) |           586.8 (572.3–591.8) |            591.2 (587.2–595.4) |           462.4 (459.3–465.7) |               **0.77×** |              **0.79×** |                1.28× |
| `q2`                |    100 |          182.7 (174.8–186.7) |           169.7 (166.2–178.7) |               179.4 (177.8–181.8) |           180.9 (179.6–185.0) |        **144.2** (144.0–146.7) |           116.6 (115.0–125.1) |               **0.79×** |              **0.85×** |                1.24× |
| `q3`                |     10 |          514.1 (502.0–527.2) |           380.8 (372.2–391.3) |               512.9 (509.0–516.6) |           515.7 (511.8–520.2) |        **325.3** (317.0–327.4) |           346.0 (338.5–369.2) |               **0.63×** |              **0.85×** |            **0.94×** |
| `q4`                |      5 |          202.3 (195.8–205.8) |           285.5 (279.9–302.9) |               167.5 (166.6–168.9) |           166.1 (165.0–168.0) |        **163.7** (162.9–167.9) |           185.5 (184.7–192.8) |               **0.81×** |              **0.57×** |            **0.88×** |
| `q5`                |      5 |          703.9 (679.4–732.3) |           471.6 (465.9–483.4) |               698.6 (689.4–708.4) |           701.2 (690.3–712.0) |        **404.7** (399.4–411.8) |           374.7 (370.7–376.7) |               **0.57×** |              **0.86×** |                1.08× |
| `q6`                |      1 |          356.4 (348.3–357.7) |           501.4 (492.3–513.2) |               334.7 (329.1–339.7) |           337.9 (325.9–341.9) |        **331.1** (327.4–339.9) |           202.2 (199.1–203.4) |               **0.93×** |              **0.66×** |                1.64× |
| `q7`                |      4 |          856.5 (850.1–883.5) |           491.1 (483.2–502.1) |               722.7 (720.1–733.8) |           729.3 (717.6–740.0) |        **366.9** (360.5–370.4) |           403.5 (400.2–417.8) |               **0.43×** |              **0.75×** |            **0.91×** |
| `q8`                |      2 |          652.9 (637.0–660.1) |           644.5 (625.9–645.7) |               675.2 (660.6–678.5) |           678.9 (674.7–684.8) |        **447.5** (438.1–454.0) |           449.7 (446.8–456.0) |               **0.69×** |              **0.69×** |            **0.99×** |
| `q9`                |    175 |    1,009.6 (1,004.7–1,030.8) |     1,487.2 (1,482.6–1,549.4) |               944.6 (931.3–956.6) |           934.7 (926.0–942.5) |        **830.8** (827.1–845.9) |           771.5 (757.8–778.5) |               **0.82×** |              **0.56×** |                1.08× |
| `q10`               |     20 |          634.9 (608.6–656.7) |           519.0 (515.5–548.3) |               609.4 (604.5–614.1) |           617.7 (606.2–620.7) |        **479.3** (475.3–484.5) |           444.4 (443.3–447.3) |               **0.75×** |              **0.92×** |                1.08× |
| `q11`               |  4,606 |          113.8 (105.0–116.5) |           121.0 (118.6–123.7) |               111.2 (109.3–115.9) |           112.9 (111.6–115.7) |         **103.2** (99.8–108.9) |              90.6 (89.1–97.6) |               **0.91×** |              **0.85×** |                1.14× |
| `q12`               |      2 |          216.3 (210.7–222.3) |           299.0 (297.8–300.3) |           **188.1** (186.0–224.6) |           194.6 (192.7–199.2) |            195.0 (192.4–197.0) |           218.1 (214.2–224.8) |               **0.90×** |              **0.65×** |            **0.89×** |
| `q13`               |     42 |          473.1 (466.8–487.5) |           532.5 (518.5–550.6) |           **342.7** (341.0–346.7) |           360.0 (347.5–361.5) |            344.6 (340.7–348.0) |           517.7 (512.5–548.0) |               **0.73×** |              **0.65×** |            **0.67×** |
| `q14`               |      1 |          397.8 (393.0–431.1) |           291.9 (286.1–296.6) |               430.8 (428.1–436.8) |           387.3 (381.7–388.6) |        **264.1** (261.7–316.7) |           326.8 (325.0–333.1) |               **0.66×** |              **0.90×** |            **0.81×** |
| `q15`               |      1 |          709.3 (698.9–714.3) |           917.3 (859.0–922.4) |               659.7 (645.6–665.6) |       **649.7** (648.0–673.8) |            679.8 (669.9–715.6) |           265.3 (258.9–273.1) |               **0.96×** |              **0.74×** |                2.56× |
| `q16`               | 27,724 |          131.3 (123.5–138.1) |           141.1 (137.5–153.5) |               135.6 (129.9–161.6) |       **122.2** (120.3–125.0) |            123.4 (120.5–125.6) |           137.5 (131.9–142.1) |               **0.94×** |              **0.87×** |            **0.90×** |
| `q17`               |      1 |    1,157.7 (1,130.0–1,171.4) |           455.4 (445.7–488.8) |         1,032.1 (1,016.7–1,040.1) |     1,017.3 (1,004.3–1,038.0) |        **382.9** (376.5–386.5) |           400.5 (396.5–409.1) |               **0.33×** |              **0.84×** |            **0.96×** |
| `q18`               |    100 |    1,361.7 (1,350.5–1,384.9) |           817.2 (793.6–819.8) |         1,544.3 (1,508.3–1,582.2) |     1,553.0 (1,534.6–1,576.5) |        **571.3** (557.8–588.0) |           457.7 (451.8–467.2) |               **0.42×** |              **0.70×** |                1.25× |
| `q19`               |      1 |          542.5 (536.7–548.3) |           413.6 (403.0–421.7) |               477.3 (468.7–485.8) |           493.2 (480.4–521.8) |        **355.8** (350.9–361.9) |           327.1 (319.3–336.6) |               **0.66×** |              **0.86×** |                1.09× |
| `q20`               |    901 |          462.6 (450.8–508.3) |           303.6 (299.0–314.2) |               525.3 (516.0–532.3) |           532.6 (528.9–539.2) |        **283.9** (277.1–289.7) |           297.1 (292.8–303.0) |               **0.61×** |              **0.94×** |            **0.96×** |
| `q21`               |    100 |    1,318.6 (1,302.7–1,340.4) |     1,454.8 (1,447.7–1,458.6) |         1,315.3 (1,300.7–1,356.0) |     1,325.1 (1,313.2–1,353.4) |        **729.0** (717.0–730.3) |           716.0 (711.9–743.8) |               **0.55×** |              **0.50×** |                1.02× |
| `q22`               |      7 |          118.8 (117.5–122.3) |           138.4 (133.8–144.9) |           **110.8** (108.1–116.0) |           113.9 (109.7–131.0) |            114.7 (112.9–116.2) |           140.7 (136.5–150.5) |               **0.97×** |              **0.83×** |            **0.82×** |
| **GeoMean (22/22)** |      — |                 **458.3 ms** |                  **421.0 ms** |                      **432.0 ms** |                  **431.6 ms** |                   **318.0 ms** |                  **301.8 ms** |               **0.69×** |              **0.76×** |            **1.05×** |
| **Sum (22/22)**     |      — |              **12,885.6 ms** |               **11,580.1 ms** |                   **12,303.9 ms** |               **12,311.1 ms** |                 **8,232.2 ms** |                **7,651.6 ms** |               **0.64×** |              **0.71×** |            **1.08×** |

_(Per-query oracle `Best Morsel` achieves **`315.9 ms`** geomean /
**`8,182.6 ms`** sum — `0.69×` vs. `stock_default`, `0.75×` vs.
`stock_pushdown`, `1.05×` vs. `duckdb_default` — just `2.0 ms` / `0.64%` ahead
of single-config `morsel_pushdown`.)_

## 2. Direct Before vs. After Comparison (Wave-4 `#209` vs. Wave-5 `#220`)

### 2.1 All 22 Queries (`morsel_pushdown` & `Best Morsel`)

Comparing Wave-4 (`78e46e910`, post-`#206`/`#207`/`#208`) directly against
Wave-5 (`00b594f0e`, post-`#217`/`#218`/`#219`) on identical SF5 Parquet tables
and `rob-tandy-dduck2` (`*` marks the 10 Stock-gap target queries from
`#217`/`#218`/`#219`):

| Query               | W4 `morsel_pushdown` | W5 `morsel_pushdown` | `pushdown` Speedup | `pushdown` Delta ms |  W4 Best Morsel | W5 Best Morsel | Best Speedup | W5 `morsel_pd` / Stock Def | W5 `morsel_pd` / Stock PD | W5 `morsel_pd` / DuckDB |
| ------------------- | -------------------: | -------------------: | -----------------: | ------------------: | --------------: | -------------: | -----------: | -------------------------: | ------------------------: | ----------------------: |
| `q1`                |                611.2 |                591.2 |              1.03× |               −20.1 |           596.1 |          585.5 |        1.02× |                  **0.77×** |                 **0.79×** |                   1.28× |
| `q2`                |                169.8 |                144.2 |          **1.18×** |               −25.5 |           169.8 |          144.2 |    **1.18×** |                  **0.79×** |                 **0.85×** |                   1.24× |
| `q3`                |                394.3 |                325.3 |          **1.21×** |               −69.0 |           394.3 |          325.3 |    **1.21×** |                  **0.63×** |                 **0.85×** |               **0.94×** |
| _*`q4`* (`#219`)_*  |                226.5 |            **163.7** |          **1.38×** |               −62.8 |           226.5 |          163.7 |    **1.38×** |                  **0.81×** |                 **0.57×** |               **0.88×** |
| `q5`                |                444.9 |                404.7 |          **1.10×** |               −40.2 |           444.9 |          404.7 |    **1.10×** |                  **0.57×** |                 **0.86×** |                   1.08× |
| `q6`                |                345.0 |                331.1 |              1.04× |               −13.9 |           340.0 |          331.1 |        1.03× |                  **0.93×** |                 **0.66×** |                   1.64× |
| `q7`                |                435.7 |                366.9 |          **1.19×** |               −68.8 |           435.7 |          366.9 |    **1.19×** |                  **0.43×** |                 **0.75×** |               **0.91×** |
| `q8`                |                568.9 |                447.5 |          **1.27×** |              −121.4 |           568.9 |          447.5 |    **1.27×** |                  **0.69×** |                 **0.69×** |               **0.99×** |
| _*`q9`* (`#217`)_*  |              1,397.5 |            **830.8** |          **1.68×** |              −566.6 |         1,336.8 |          830.8 |    **1.61×** |                  **0.82×** |                 **0.56×** |                   1.08× |
| _*`q10`* (`#217`)_* |                616.0 |            **479.3** |          **1.29×** |              −136.7 |           616.0 |          479.3 |    **1.29×** |                  **0.75×** |                 **0.92×** |                   1.08× |
| _*`q11`* (`#219`)_* |                135.2 |            **103.2** |          **1.31×** |               −32.1 |           135.2 |          103.2 |    **1.31×** |                  **0.91×** |                 **0.85×** |                   1.14× |
| `q12`               |                208.1 |                195.0 |          **1.07×** |               −13.1 |           208.1 |          188.1 |    **1.11×** |                  **0.90×** |                 **0.65×** |               **0.89×** |
| _*`q13`* (`#217`)_* |                590.7 |            **344.6** |          **1.71×** |              −246.1 |           562.5 |          342.7 |    **1.64×** |                  **0.73×** |                 **0.65×** |               **0.67×** |
| _*`q14`* (`#218`)_* |                341.1 |            **264.1** |          **1.29×** |               −77.0 |           341.1 |          264.1 |    **1.29×** |                  **0.66×** |                 **0.90×** |               **0.81×** |
| `q15`               |                704.4 |                679.8 |              1.04× |               −24.6 |           688.0 |          649.7 |    **1.06×** |                  **0.96×** |                 **0.74×** |                   2.56× |
| _*`q16`* (`#219`)_* |                172.0 |            **123.4** |          **1.39×** |               −48.6 |           169.5 |          122.2 |    **1.39×** |                  **0.94×** |                 **0.87×** |               **0.90×** |
| _*`q17`* (`#218`)_* |                499.8 |            **382.9** |          **1.31×** |              −116.9 |           499.8 |          382.9 |    **1.31×** |                  **0.33×** |                 **0.84×** |               **0.96×** |
| `q18`               |                670.9 |                571.3 |          **1.17×** |               −99.6 |           670.9 |          571.3 |    **1.17×** |                  **0.42×** |                 **0.70×** |                   1.25× |
| `q19`               |                395.3 |                355.8 |          **1.11×** |               −39.5 |           395.3 |          355.8 |    **1.11×** |                  **0.66×** |                 **0.86×** |                   1.09× |
| _*`q20`* (`#218`)_* |                329.3 |            **283.9** |          **1.16×** |               −45.4 |           329.3 |          283.9 |    **1.16×** |                  **0.61×** |                 **0.94×** |               **0.96×** |
| `q21`               |                937.0 |                729.0 |          **1.29×** |              −208.0 |           937.0 |          729.0 |    **1.29×** |                  **0.55×** |                 **0.50×** |                   1.02× |
| _*`q22`* (`#219`)_* |                126.3 |            **114.7** |          **1.10×** |               −11.6 |           126.3 |          110.8 |    **1.14×** |                  **0.97×** |                 **0.83×** |               **0.82×** |
| **GeoMean (22/22)** |         **391.2 ms** |         **318.0 ms** |          **1.23×** |        **−73.2 ms** |    **388.1 ms** |   **315.9 ms** |    **1.23×** |                  **0.69×** |                 **0.76×** |               **1.05×** |
| **Sum (22/22)**     |      **10,319.9 ms** |       **8,232.2 ms** |          **1.25×** |     **−2,087.7 ms** | **10,192.0 ms** | **8,182.6 ms** |    **1.25×** |                  **0.64×** |                 **0.71×** |               **1.08×** |

### 2.2 Targeted 10-Query Stock-Gap Closure Summary (`q4`, `q9`, `q10`, `q11`, `q13`, `q14`, `q16`, `q17`, `q20`, `q22`)

In Wave-4 (`#209`), these 10 queries were the **only** queries in the entire
22-query TPC-H suite where single-config `morsel_pushdown` trailed either
`stock_default` (`q4`, `q9`, `q11`, `q13`, `q16`, `q22`) or `stock_pushdown`
(`q10`, `q11`, `q13`, `q14`, `q16`, `q17`, `q20`). Every single one now beats
**both** Stock arms — and 7 of the 10 beat DuckDB 2.0.1:

| Query       | Issue  | W4 `morsel_pd` / Stock Def | **W5 `morsel_pd` / Stock Def** | W4 `morsel_pd` / Stock PD | **W5 `morsel_pd` / Stock PD** | W4 → W5 `morsel_pd` ms | W4 → W5 Gain | **W5 vs. DuckDB** |
| ----------- | ------ | -------------------------: | -----------------------------: | ------------------------: | ----------------------------: | ---------------------: | -----------: | ----------------: |
| **`q4`**    | `#219` |                      1.13× |                      **0.81×** |                 **0.78×** |                     **0.57×** |        `226.5 → 163.7` |    **1.38×** |         **0.88×** |
| **`q9`**    | `#217` |                      1.37× |                      **0.82×** |                 **0.92×** |                     **0.56×** |      `1,397.5 → 830.8` |    **1.68×** |             1.08× |
| **`q10`**   | `#217` |                  **0.94×** |                      **0.75×** |                     1.17× |                     **0.92×** |        `616.0 → 479.3` |    **1.29×** |             1.08× |
| **`q11`**   | `#219` |                      1.20× |                      **0.91×** |                     1.06× |                     **0.85×** |        `135.2 → 103.2` |    **1.31×** |             1.14× |
| **`q13`**   | `#217` |                      1.23× |                      **0.73×** |                     1.02× |                     **0.65×** |        `590.7 → 344.6` |    **1.71×** |         **0.67×** |
| **`q14`**   | `#218` |                  **0.85×** |                      **0.66×** |                     1.13× |                     **0.90×** |        `341.1 → 264.1` |    **1.29×** |         **0.81×** |
| **`q16`**   | `#219` |                      1.30× |                      **0.94×** |                     1.20× |                     **0.87×** |        `172.0 → 123.4` |    **1.39×** |         **0.90×** |
| **`q17`**   | `#218` |                  **0.41×** |                      **0.33×** |                     1.07× |                     **0.84×** |        `499.8 → 382.9` |    **1.31×** |         **0.96×** |
| **`q20`**   | `#218` |                  **0.70×** |                      **0.61×** |                     1.02× |                     **0.94×** |        `329.3 → 283.9` |    **1.16×** |         **0.96×** |
| **`q22`**   | `#219` |                      1.02× |                      **0.97×** |                 **0.89×** |                     **0.83×** |        `126.3 → 114.7` |    **1.10×** |         **0.82×** |
| **10Q GM**  | —      |                      0.96× |                      **0.73×** |                     1.02× |                     **0.78×** |        `337.2 → 249.7` |    **1.35×** |         **0.92×** |
| **10Q Sum** | —      |                      0.92× |                      **0.66×** |                     1.00× |                     **0.72×** |    `4,434.3 → 3,090.5` |    **1.43×** |         **0.93×** |

## 3. Deep-Dive Attribution Across the Three Wave-5 Stock-Gap Pillars

### 3.1 `q9`, `q10`, and `q13` Stock-Gap Fixes (`#217` — `−949.5 ms`, `1.57×`)

`#217` delivered nearly half (`45.5%`) of Wave-5's total `−2,087.7 ms` suite
reduction by eliminating Bloom contention/over-filtering, radix slice storms,
generic row-converter hashing on 2-key integer joins, and suboptimal build-side
choices:

| Query     | W4 → W5 wall ms (`morsel_pd`) |   Speedup | Exec CPU W4 → W5 | `decode_work` W4 → W5 | `process_work` W4 → W5 | `join_finalize` W4 → W5 | `ready_wait` W4 → W5 | Primary Mechanism                                                                                                                                                  |
| --------- | ----------------------------: | --------: | ---------------: | --------------------: | ---------------------: | ----------------------: | -------------------: | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **`q9`**  |          `1,397.5 → 830.8 ms` | **1.68×** |  `9,118 → 6,040` |       `5,195 → 4,358` |    **`2,269 → 1,130`** |      **`276.5 → 44.9`** |   **`228.3 → 68.6`** | Sibling `part -> partsupp` pushdown (`build_2` `jf` `153.1 → 1.0 ms`), 2-key `FlatIntTable`, `RadixChunk` (`build_3` `jf` `120.6 → 43.0 ms`), Bloom gating         |
| **`q10`** |            `616.0 → 479.3 ms` | **1.29×** |  `3,893 → 3,183` |       `2,107 → 2,120` |            `792 → 638` |      **`100.0 → 20.1`** |       `176.5 → 93.5` | Two-sided range + projected width flips `orders` onto build (`build_0_build` + `build_0` `jf` `98.7 → 19.9 ms`, `proc` `303 → 95 ms`), `RadixChunk`                |
| **`q13`** |            `590.7 → 344.6 ms` | **1.71×** |  `3,746 → 2,448` |       `1,548 → 1,748` |      **`1,732 → 479`** |           `26.8 → 13.1` |        `57.1 → 33.3` | `RadixChunk` zero-slice `GroupCountRadixMerger` (`stage_0_probe` `proc` `1,697 → 455 ms`), `DenseIntTable` (`customer`), build-key substitution + zero-copy output |

### 3.2 `q14`, `q17`, and `q20` Stock-Pushdown-Gap Fixes (`#218` — `−239.3 ms`, `1.26×`)

`#218` closed all remaining pushdown-specific gaps against `stock_pushdown`
while turning `q14`, `q17`, and `q20` into **DuckDB 2.0.1 wins (`0.81×`,
`0.96×`, `0.96×`)**:

| Query     | W4 → W5 wall ms (`morsel_pd`) |   Speedup | Exec CPU W4 → W5 | `decode_work` W4 → W5 | `process_work` W4 → W5 | `ready_wait` W4 → W5 | `dyn_filters` | Adapt `RF/Bulk` W4 → W5 | Primary Mechanism                                                                                                                                              |
| --------- | ----------------------------: | --------: | ---------------: | --------------------: | ---------------------: | -------------------: | ------------: | ----------------------: | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`q14`** |            `341.1 → 264.1 ms` | **1.29×** |  `2,215 → 1,817` |       `1,556 → 1,547` |         `179.0 → 93.6` |   **`114.8 → 36.1`** |       `3 → 3` |         `237/3 → 244/4` | Interval selectivity (`~1.19%` `l_shipdate`) flips `lineitem` onto build (`jf` `35.7 → 2.9 ms`) + wave-1 RF + Bloom onto `part` + `2×` ready queue             |
| **`q17`** |            `499.8 → 382.9 ms` | **1.31×** |  `3,440 → 2,715` |   **`3,154 → 2,486`** |          `25.7 → 10.5` |   **`136.5 → 57.1`** |       `6 → 6` |     **`480/8 → 488/0`** | Infallible Bloom + redundant range skip + 32b/key L1 Bloom + `skip_bulk_probe` (`0` wasted Bulk RGs) + `2×` ready queue (`build` + `stage_0` decode `−668 ms`) |
| **`q20`** |            `329.3 → 283.9 ms` | **1.16×** |  `2,036 → 1,816` |       `1,761 → 1,631` |           `18.3 → 4.2` |   **`174.9 → 65.1`** | **`12 → 15`** |     **`276/6 → 282/0`** | Transitive `s_suppkey` equijoin pushdown into `stage_0` (`lineitem`), 32b/key L1 Bloom, `skip_bulk_probe`, `2×` ready queue, 1-worker 1-RG builds              |

### 3.3 `q4`, `q11`, `q16`, and `q22` Stock-Gap Fixes (`#219` — `−155.1 ms`, `1.31×`)

`#219` resolved the final four Stock gaps (semi/anti-join probe cost, small-build
coordination overhead, stage channel backpressure, and unnecessary `RowFilter`
overhead on fixed-width/non-equality scans), flipping `q4` (`0.88×`), `q16`
(`0.90×`), and `q22` (`0.82×`) ahead of DuckDB 2.0.1 as well:

| Query     | W4 → W5 wall ms (`morsel_pd`) |   Speedup | Exec CPU W4 → W5 | `decode_work` W4 → W5 | `process_work` W4 → W5 | `ready_wait` W4 → W5 | Adapt `RF/Bulk` W4 → W5 | Primary Mechanism                                                                                                                                                                |
| --------- | ----------------------------: | --------: | ---------------: | --------------------: | ---------------------: | -------------------: | ----------------------: | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **`q4`**  |            `226.5 → 163.7 ms` | **1.38×** |  `1,417 → 1,056` |     **`1,125 → 639`** |        `114.2 → 316.0` |    **`60.4 → 24.2`** |    **`247/57 → 2/302`** | Scoped `BulkDecode` on all-predicate radix scan + fused flat-int `BuildSemi` with L2 `RegisterBloom64` (`morsel_no_pd` also drops `311.3 → 167.5 ms`, `1.86×`!)                  |
| **`q11`** |            `135.2 → 103.2 ms` | **1.31×** |      `788 → 611` |           `620 → 479` |     **`114.5 → 68.1`** |        `56.6 → 34.4` |      **`38/30 → 4/64`** | Compact-span `DenseIntTable` (`supplier`, `4.0%` density) + scoped `BulkDecode` on `partsupp` + 1-worker 1-RG builds (4-build `rw` `27.0 → 2.4 ms`)                              |
| **`q16`** |            `172.0 → 123.4 ms` | **1.39×** |      `737 → 646` |           `254 → 209` |        `266.2 → 269.9` |   **`119.5 → 31.6`** |      **`39/2 → 29/12`** | Widened `ensure_stage` channel (`4 → 16`, `other` `86.4 → 47.8 ms`) + concurrent small-build wave & 1-worker 1-RG (`stage_0_build_1` `rw` `91.6 → 1.2 ms`) + `part` `BulkDecode` |
| **`q22`** |            `126.3 → 114.7 ms` | **1.10×** |      `722 → 609` |       **`528 → 403`** |        `129.3 → 150.9` |      `140.0 → 126.5` |      **`74/2 → 12/64`** | Scoped `BulkDecode` on all-predicate `orders` + fused `RegisterBloom64` `BuildAnti` probe (`morsel_no_pd` drops `169.0 → 110.8 ms`, `1.53×`)                                     |

### 3.4 Bonus Gains Across the 12 Non-Targeted Queries (`−743.9 ms` Raw Combined)

Because `#217`, `#218`, and `#219` fixed core architectural bottlenecks
(`RadixChunk`, lock-free radix Bloom + NEON SIMD, `DenseIntTable`, infallible
Bloom, `2×` ready queue, `skip_bulk_probe`, and concurrent 1-RG builds), **9 of
the 12 non-targeted queries delivered genuine speedups well beyond the
`2.5–3.4%` session-to-session host drift** (`0` regressions), while `q1`, `q6`,
and `q15` moved in line with the unchanged reference arms:

- **`q21` (`937.0 → 729.0 ms`, `1.29×`, `−208.0 ms`):** `join_finalize` drops
  **`3.92×`** (`150.0 → 38.3 ms`), `ready_wait` drops **`2.85×`**
  (`174.5 → 61.3 ms`), and exec CPU falls `−19.7%` (`6,143 → 4,930 ms`),
  bringing `q21` to **`1.02×`** of DuckDB 2.0.1 (`716.0 ms`) and **`0.50×`** of
  `stock_pushdown` (`2.00×` faster).
- **`q8` (`568.9 → 447.5 ms`, `1.27×`, `−121.4 ms`):** `ready_wait` drops
  **`4.41×`** (`216.3 → 49.0 ms`) and `decode_work` falls `−427.9 ms`, flipping
  `q8` ahead of DuckDB 2.0.1 (**`0.99×`**, `447.5` vs. `449.7 ms`).
- **`q18` (`670.9 → 571.3 ms`, `1.17×`, `−99.6 ms`):** `join_finalize` drops
  **`3.53×`** (`32.8 → 9.3 ms`), `ready_wait` drops `2.39×` (`112.6 → 47.1 ms`),
  and `dyn_filters` increases `9 → 11` (`adapt` `297/4 → 301/0`, `−6.6 MB`
  scanned).
- **`q3` (`394.3 → 325.3 ms`, `1.21×`, `−69.0 ms`) & `q7` (`435.7 → 366.9 ms`,
  `1.19×`, `−68.8 ms`):** `join_finalize` drops **`4.95×`** on `q3`
  (`29.2 → 5.9 ms`) and **`5.11×`** on `q7` (`22.5 → 4.4 ms`), while
  `ready_wait` drops **`4.1×`** on both — flipping **both `q3` (`0.94×`) and
  `q7` (`0.91×`) ahead of DuckDB 2.0.1**!
- **`q5` (`−40.2 ms`, `1.10×`, `dyn_filters 15 → 18`), `q19` (`−39.5 ms`,
  `1.11×`, `adapt 245/3 → 248/0`), `q2` (`−25.5 ms`, `1.18×`, `ready_wait`
  `111.1 → 41.2 ms`), and `q12` (`−13.1 ms`, `1.07×`, `morsel_pd / stock_def`
  `0.95× → 0.90×`)** also achieved genuine Wave-5 speedups beyond host drift.
- **Within host drift (`q1`, `q6`, `q15`):** `q1` (`611.2 → 591.2 ms`, `−3.3%`,
  vs. `stock_default` `−3.2%` and DuckDB `−3.6%`), `q6` (`345.0 → 331.1 ms`,
  `−4.0%`, vs. `stock_default` `−4.0%`), and `q15` (`704.4 → 679.8 ms`, `−3.5%`,
  vs. `stock_default` `−3.8%` and DuckDB `−2.6%`) are effectively flat once
  adjusted for session-to-session host drift (`morsel_pd / stock_def` stays
  `0.77×`, `0.93×`, and `0.96×`).

## 4. Native `MorselExec` Execution & `#138` Wait/Work Summary (Single-Config `morsel_pushdown`)

Every row below reports **single-config `morsel_pushdown`** (`MORSEL_METRICS=1`,
`morsel_mode=required`, 8 workers). `Decode`, `Process`, and `Ready wait` are
summed across 8 workers; `Output wait` was **`0.0 ms`** on all 330 measured
Morsel trials. Compared to Wave-4 (`#209`)'s `morsel_pushdown` totals across
22/22:

- Total `Exec` wall time fell **`1.26×`** (`9,997.8 → 7,918.2 ms`,
  `−2,079.6 ms`).
- Total `Exec CPU` fell **`−16.3%`** (`67,822.5 → 56,755.0 ms`, `−11.07 s`).
- Total `Decode` work fell **`−7.9%`** (`46,943.7 → 43,249.0 ms`, `−3.69 s`).
- Total `Process` work fell **`−28.1%`** (`10,755.0 → 7,731.7 ms`, `−3.02 s`).
- Total `Ready wait` fell **`−60.0%`** (`2,651.6 → 1,061.6 ms`, `−1.59 s` —
  `2.50×` lower across 8 workers!).
- Total `Join fin` wall time fell **`4.31×`** (`742.1 → 172.1 ms` — down
  **`42.2×`** from `#192`'s `7,258.4 ms`!).

| Query             | `mincore` pages (100%) | RGs claimed | Dyn filt | Adapt `RF/Bulk` | Stages / `jb` / `agg` / `gc` / `s` |     Exec ms |  Exec CPU ms |    Decode ms |  Process ms | Ready wait ms | Join fin ms | Sort/TopN fin ms | Agg fin / other ms |
| ----------------- | ---------------------: | ----------: | -------: | --------------: | ---------------------------------- | ----------: | -----------: | -----------: | ----------: | ------------: | ----------: | ---------------: | -----------------: |
| `q1`              |                229,147 |         240 |        0 |        `209/31` | `0 / 0 / 1 / 0 / Sort`             |       578.0 |      4,431.6 |      3,317.3 |       981.7 |          21.4 |         0.0 |              0.1 |               40.6 |
| `q2`              |                 44,595 |          78 |       19 |         `68/10` | `1 / 8 / 1 / 0 / TopN`             |       132.5 |        712.2 |        522.5 |       107.4 |          41.2 |         1.6 |              0.1 |               52.2 |
| `q3`              |                297,360 |         310 |        6 |        `276/34` | `0 / 2 / 1 / 0 / TopN`             |       309.9 |      2,201.6 |      1,782.3 |       204.1 |          29.8 |         5.9 |              0.2 |               55.5 |
| `q4`              |                286,442 |         304 |        3 |         `2/302` | `0 / 1 / 1 / 0 / Sort`             |       149.8 |      1,056.1 |        639.0 |       316.0 |          24.2 |         2.3 |              0.0 |               28.1 |
| `q5`              |                298,107 |         313 |       18 |        `254/59` | `0 / 5 / 1 / 0 / Sort`             |       387.5 |      2,717.8 |      2,283.0 |       205.4 |          58.0 |        11.0 |              0.0 |               65.4 |
| `q6`              |                229,147 |         240 |        0 |         `4/236` | `0 / 0 / 1 / 0 / —`                |       318.8 |      2,452.1 |      2,217.9 |       169.1 |          18.1 |         0.0 |              0.0 |               20.4 |
| `q7`              |                298,106 |         313 |       15 |        `296/17` | `0 / 5 / 1 / 0 / Sort`             |       348.4 |      2,533.9 |      2,177.0 |       107.8 |          45.7 |         4.4 |              0.0 |               58.3 |
| `q8`              |                303,951 |         322 |       18 |        `281/40` | `0 / 7 / 1 / 0 / Sort`             |       427.7 |      3,082.2 |      2,695.4 |       109.9 |          49.0 |         9.2 |              0.0 |               67.8 |
| `q9`              |                331,036 |         346 |       17 |         `274/7` | `0 / 5 / 1 / 0 / Sort`             |       812.6 |      6,040.0 |      4,358.0 |     1,129.7 |          68.6 |        44.9 |              0.0 |               81.7 |
| `q10`             |                297,361 |         311 |        9 |        `283/27` | `0 / 3 / 1 / 0 / TopN`             |       463.2 |      3,183.4 |      2,119.8 |       638.0 |          93.5 |        20.1 |              0.2 |               98.1 |
| `q11`             |                 38,750 |          68 |        8 |          `4/64` | `0 / 4 / 2 / 0 / Sort`             |        94.6 |        610.7 |        479.1 |        68.1 |          34.4 |         0.2 |              0.5 |               25.6 |
| `q12`             |                286,442 |         304 |        3 |        `67/237` | `0 / 1 / 1 / 0 / Sort`             |       179.7 |      1,297.4 |        937.4 |       221.9 |          34.6 |         1.6 |              0.0 |               33.2 |
| `q13`             |                 68,213 |          70 |        2 |          `58/6` | `1 / 1 / 1 / 1 / Sort`             |       337.1 |      2,448.1 |      1,747.9 |       479.4 |          33.3 |        13.1 |              0.0 |               45.6 |
| `q14`             |                234,991 |         248 |        3 |         `244/4` | `0 / 1 / 1 / 0 / —`                |       249.7 |      1,817.1 |      1,547.2 |        93.6 |          36.1 |         2.9 |              0.0 |               41.7 |
| `q15`             |                229,892 |         481 |        2 |        `30/450` | `2 / 1 / 3 / 0 / Sort`             |       664.7 |      4,976.6 |      4,307.9 |       419.8 |          40.3 |         2.0 |              0.0 |               71.7 |
| `q16`             |                 44,593 |          41 |        3 |         `29/12` | `1 / 2 / 2 / 0 / Sort`             |       114.5 |        646.1 |        209.3 |       269.9 |          31.6 |         1.8 |              5.0 |               47.8 |
| `q17`             |                234,991 |         488 |        6 |         `488/0` | `1 / 2 / 2 / 0 / —`                |       368.1 |      2,714.9 |      2,486.1 |        10.5 |          57.1 |         0.4 |              0.0 |               55.5 |
| `q18`             |                297,360 |         547 |       11 |         `301/0` | `1 / 3 / 2 / 0 / TopN`             |       556.5 |      4,000.8 |      1,854.2 |       947.9 |          47.1 |         9.3 |              0.0 |              196.9 |
| `q19`             |                234,991 |         248 |        3 |         `248/0` | `0 / 1 / 1 / 0 / —`                |       340.3 |      2,476.7 |      2,246.2 |        14.7 |          45.0 |         1.3 |              0.0 |               56.4 |
| `q20`             |                273,741 |         282 |       15 |         `282/0` | `1 / 4 / 1 / 0 / Sort`             |       267.5 |      1,816.1 |      1,631.4 |         4.2 |          65.1 |         0.7 |              0.0 |               62.4 |
| `q21`             |                287,188 |         786 |       13 |       `542/244` | `0 / 5 / 1 / 0 / TopN`             |       711.7 |      4,930.0 |      3,287.2 |     1,081.5 |          61.3 |        38.3 |              0.0 |              127.3 |
| `q22`             |                 68,213 |          76 |        3 |         `12/64` | `0 / 1 / 2 / 0 / Sort`             |       105.4 |        609.4 |        403.1 |       150.9 |         126.5 |         1.2 |              0.0 |               34.9 |
| **Total (22/22)** |                      — |           — |        — |               — | —                                  | **7,918.2** | **56,755.0** | **43,249.0** | **7,731.7** |   **1,061.6** |   **172.1** |          **6.4** |        **1,367.1** |

### Remaining Headroom vs. DuckDB 2.0.1 (`1.05×` GeoMean / `1.08×` Sum)

With single-config `morsel_pushdown` now beating Stock DataFusion `22 / 22`
(both `stock_default` and `stock_pushdown`) and beating DuckDB 2.0.1 on
`11 / 22` queries, the remaining **`+580.6 ms` net sum gap** to DuckDB 2.0.1
(`8,232.2` vs. `7,651.6 ms`) consists of `+785.7 ms` across four top lagging
queries (`q15`, `q6`, `q1`, `q18`) plus `+206.1 ms` across seven smaller
lagging queries, offset by `−411.1 ms` of net wins across the 11 queries where
Morsel is faster than DuckDB:

1. **Stage CSE on `q15` (`679.8` vs. `265.3 ms`, `+414.5 ms` — `71.4%` of the
   entire 22-query net sum gap):** Materializing `revenue0` once instead of
   scanning `lineitem` twice (`stage_0` + `stage_2`: `605.1 MB` scanned,
   `4,307.9 ms` decode) would save `~320–340 ms`, bringing 22/22
   `morsel_pushdown` to **`308.1–308.9 ms` (`~1.02×` geomean)** and
   **`7,892–7,912 ms` (`~1.03×` sum)** of DuckDB 2.0.1.
2. **Parquet `Decimal128` decode throughput on `q6` (`331.1` vs. `202.2 ms`,
   `+128.9 ms`) and `q1` (`591.2` vs. `462.4 ms`, `+128.7 ms`):** `q6` (`90.4%`
   of exec CPU in Parquet decode) and `q1` (`74.9%` in Parquet decode) are now
   overwhelmingly bound by Arrow-rs Parquet Zstd/`FixedLenByteArray`
   `Decimal128` decoding rather than morsel execution overhead.
3. **High-cardinality `l_orderkey` aggregation on `q18` (`571.3` vs. `457.7 ms`,
   `+113.6 ms`):** Splits across Parquet decode (`1,854.2 ms`), 7.5M-group
   `stage_0` aggregation/join processing (`947.9 ms`), and parallel radix
   aggregate finalization (`196.9 ms` wall).
4. **Seven smaller residual gaps (`+206.1 ms` combined):** Five join queries
   sit within single-digit percentages of DuckDB 2.0.1 (`q21` `+13.0 ms` /
   `1.02×`, `q5` `+30.0 ms` / `1.08×`, `q9` `+59.3 ms` / `1.08×`, `q10`
   `+34.9 ms` / `1.08×`, `q19` `+28.7 ms` / `1.09×`), while small-table `q11`
   (`+12.6 ms` / `1.14×`) and 8-build `q2` (`+27.6 ms` / `1.24×`) account for
   the remaining `40.2 ms`.

## Summary

Wave-5 (`#217` `q9`/`q10`/`q13` fixes, `#218` `q14`/`q17`/`q20` pushdown fixes,
and `#219` `q4`/`q11`/`q16`/`q22` fixes) delivers a **clean `22 / 22` sweep
over Stock DataFusion** and brings single-config Morsel within **`5.4%` of
DuckDB 2.0.1**: **single-config `morsel_pushdown` (`morsel_mode=required`)
drops from `391.2 → 318.0 ms` geomean (`1.23×` faster vs. Wave-4; sum
`10,319.9 → 8,232.2 ms`, `1.25×` / `−2.09 s`), beats `stock_default` on
`22 / 22` queries (`0.69×` GM / `0.64×` sum), beats `stock_pushdown` on
`22 / 22` queries (`0.76×` GM / `0.71×` sum), beats DuckDB 2.0.1 on `11 / 22`
queries (`1.05×` GM / `1.08×` sum overall; `1.01×` GM excluding `q15`), shrinks
the oracle gap to `2.0 ms` (`0.64%`), and preserves 100% DuckDB parity, 100%
native admission, zero output backpressure, and 100% warm-cache `mincore`
verification**.
