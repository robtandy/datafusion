<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# TPC-H SF5 Wave-4 22/22 3-Way Scorecard: Post Cross-Aggregate Pushdown, Adaptive `RowFilter` & Aggregate CSE (#209)

This scorecard reports the **Wave-4 complete 22/22 standard TPC-H (`q1`–`q22`)
3-way comparison at Scale Factor 5 (`--sf 5`)** between **Morsel DataFusion
(`morsel_mode=required`)**, **Stock DataFusion (`morsel_mode=off`, both
`stock_default` and `stock_pushdown`)**, and **DuckDB 2.0.1** on quiet
`rob-tandy-dduck2` (8 pinned ARM Neoverse-N1 cores, verified 100% warm Linux
page cache via `mincore`) after all three Wave-4 optimizations merged onto
`morsel-execution`:

1. **Cross-`Aggregate` / cross-stage dynamic join filter pushdown & selective
   build-side placement (`#206`)**,
2. **Runtime-adaptive per-RG `RowFilter` vs. `BulkDecode` switching &
   DuckDB-style `AdaptiveFilter` conjunct reordering (`#207`)**, and
3. **Aggregate accumulator CSE, `AVG`-from-`SUM`/`COUNT` sharing & direct
   `Decimal128` aggregation (`#208`)**.

It re-runs the exact `#195` 6-arm warm-cache protocol on the identical SF5
Parquet dataset and host, comparing Wave-4 directly against
[sf5_22q_final_scorecard.md](sf5_22q_final_scorecard.md) (`#195`, without
overwriting it) and the earlier `#192` baseline
([sf5_22q_scorecard.md](sf5_22q_scorecard.md)).

## Executive Summary

- **Single-config Morsel (`morsel_pushdown`, `morsel_mode=required`) now
  decisively beats BOTH `stock_default` AND `stock_pushdown` across the entire
  22/22 TPC-H SF5 suite:**
  - **22/22 Geometric Mean:** drops **`461.0 → 391.2 ms`** (**`1.18×` faster**
    vs. `#195`, `−69.8 ms`; **`1.83×` faster** vs. `#192`'s `716.3 ms`), beating
    **`stock_default` (`471.3 ms`) at `0.83×` (`1.20×` faster)**, beating
    **`stock_pushdown` (`435.7 ms`) at `0.90×` (`1.11×` faster)**, and slashing
    the gap to **DuckDB 2.0.1 (`309.5 ms`)** from **`1.48× → 1.26×`** (down from
    `2.36×` in `#192`).
  - **22/22 Sum of Medians:** drops **`12,757.6 → 10,319.9 ms`** (**`1.24×`
    faster**, **`−2,437.7 ms` / `−2.44 s`** vs. `#195`; `−12.92 s` / `2.25×`
    vs. `#192`), beating **`stock_default` (`13,384.5 ms`) at `0.77×` (`1.30×`
    faster, `−3.06 s`)**, beating **`stock_pushdown` (`11,969.1 ms`) at `0.86×`
    (`1.16×` faster, `−1.65 s`)**, and narrowing the sum gap to **DuckDB 2.0.1
    (`7,837.7 ms`)** from **`1.62× → 1.32×`**.
  - **Oracle gap virtually eliminated:** The geometric-mean gap between
    single-config `morsel_pushdown` (`391.2 ms`) and per-query oracle
    `Best Morsel` (`388.1 ms`) shrinks **`5.9×`** from **`17.8 ms`** in `#195`
    to **`3.0 ms` (`< 0.8%`)**. Single-config `morsel_pushdown` is now the
    fastest Morsel arm on **`16 / 22`** queries and within `1.4–4.5%` on the
    remaining 6.
- **Head-to-head query wins for single-config `morsel_pushdown`:**
  - Beats **`stock_default` on `16 / 22` queries** (up from `11 / 22` Best
    Morsel in `#195`): `q1`, `q2`, `q3`, `q5`, `q6`, `q7`, `q8`, `q10`, `q12`,
    `q14`, `q15`, `q17`, `q18`, `q19`, `q20`, and `q21`.
  - Beats **`stock_pushdown` on `15 / 22` queries** (up from `11 / 22` in
    `#195`): `q1`, `q2`, `q3`, `q4`, `q5`, `q6`, `q7`, `q8`, `q9`, `q12`, `q15`,
    `q18`, `q19`, `q21`, and `q22`.
  - Beats **DuckDB 2.0.1 on `2 / 22` queries** (`q12`: `208.1` vs. `219.5 ms`,
    **`0.95×`**; `q22`: `126.3` vs. `144.7 ms`, **`0.87×`**) and lands within
    **1–7% of DuckDB 2.0.1** on **`q14`** (`1.01×`), **`q7`** (`1.06×`), and
    **`q20`** (`1.07×`).
- **100% native admission, 100% DuckDB parity, 100% warm cache, zero output
  backpressure:** All `22 / 22` standard TPC-H queries passed lossless Arrow IPC
  parity against **DuckDB 2.0.1** at both `--sf 0.01` (`21 / 21` `morsel_tpch`
  unittests passing) and `--sf 5` across all 6 arms, ran strictly under
  `MorselExec: workers=8` with `morsel_mode=required` (`0` fallbacks, `0`
  rejections), verified **100% `mincore` page residency on all `660 / 660`
  measured trials**, and recorded **`0.0 ms` `morsel_output_wait`** across all
  330 measured Morsel trials.

| Arm                                                | `#192` GeoMean | `#195` GeoMean | Wave-4 GeoMean | `#195 → W4` Gain | Wave-4 Sum (22/22) | vs. `stock_default` (GM / Sum) | vs. `stock_pushdown` (GM / Sum) | vs. `duckdb_default` (GM / Sum) |
| -------------------------------------------------- | -------------: | -------------: | -------------: | ---------------: | -----------------: | -----------------------------: | ------------------------------: | ------------------------------: |
| Stock DataFusion (`stock_default`, `off`)          |       462.6 ms |       474.2 ms |       471.3 ms |            1.01× |        13,384.5 ms |                  1.00× / 1.00× |                   1.08× / 1.12× |                   1.52× / 1.71× |
| Stock DataFusion (`stock_pushdown`, `off`)         |              — |       434.2 ms |       435.7 ms |            1.00× |        11,969.1 ms |                  0.92× / 0.89× |                   1.00× / 1.00× |                   1.41× / 1.53× |
| Morsel DataFusion (`morsel_no_pushdown`, `req`)    |       690.7 ms |       563.9 ms |       546.0 ms |            1.03× |        15,371.8 ms |                  1.16× / 1.15× |                   1.25× / 1.28× |                   1.76× / 1.96× |
| Morsel DataFusion (`morsel_pruning`, `req`)        |              — |       568.4 ms |       550.1 ms |            1.03× |        15,399.3 ms |                  1.17× / 1.15× |                   1.26× / 1.29× |                   1.78× / 1.96× |
| **Morsel DataFusion (`morsel_pushdown`, `req`)**   |       716.3 ms |       461.0 ms |   **391.2 ms** |        **1.18×** |    **10,319.9 ms** |              **0.83× / 0.77×** |               **0.90× / 0.86×** |               **1.26× / 1.32×** |
| **Best Morsel per query (`morsel_mode=required`)** |       679.7 ms |       443.2 ms |   **388.1 ms** |        **1.14×** |    **10,192.0 ms** |              **0.82× / 0.76×** |               **0.89× / 0.85×** |               **1.25× / 1.30×** |
| **DuckDB 2.0.1 (`duckdb_default`)**                |       304.0 ms |       311.6 ms |   **309.5 ms** |            1.01× |     **7,837.7 ms** |              **0.66× / 0.59×** |               **0.71× / 0.65×** |               **1.00× / 1.00×** |

_(Baseline stability check: all three unchanged reference arms matched `#195`
within `< 0.7%` — `stock_default` `474.2 → 471.3 ms` (`−0.6%`), `stock_pushdown`
`434.2 → 435.7 ms` (`+0.3%`), and `duckdb_default` `311.6 → 309.5 ms` (`−0.7%`),
confirming a clean apples-to-apples comparison on `rob-tandy-dduck2`.)_

- **Wave-4 pillar-by-pillar headline gains (single-config `morsel_pushdown`,
  `#195 → Wave-4`):**
  - **Cross-`Aggregate` dynamic pushdown & build-side selection (`#206`):**
    - **`q17`:** `1,238.2 → 499.8 ms` (**`2.48×`**, `−738.4 ms`),
      `morsel_process_work` `2,156.9 → 25.7 ms` (**`83.9×`**), exec CPU
      `−51.4%`. Flips from `1.03×` behind `stock_default` to **`0.41×`
      (`2.45×` faster)**.
    - **`q20`:** `774.1 → 329.3 ms` (**`2.35×`**, `−444.8 ms`),
      `morsel_process_work` `786.0 → 18.3 ms` (**`43.0×`**), exec CPU
      `−46.2%`, dynamic filters `4 → 12`. Flips from `1.62×` behind
      `stock_default` to **`0.70×` (`1.44×` faster)** and within `1.07×` of
      DuckDB 2.0.1 (`307.0 ms`).
    - **`q21`:** `1,544.1 → 937.0 ms` (**`1.65×`**, `−607.1 ms`),
      `join_finalize` `515.7 → 150.0 ms` (**`3.44×`**). Flips from `1.14×`
      behind `stock_default` to **`0.67×` (`1.49×` faster)** and **`0.61×` vs.
      `stock_pushdown` (`1.64×` faster)**.
    - **`q2`:** `194.2 → 169.8 ms` (**`1.14×`**, `−24.4 ms`), `join_finalize`
      `22.2 → 2.1 ms` (**`10.6×`**), now beating both `stock_default` (`0.94×`)
      and `stock_pushdown` (`0.97×`).
  - **Adaptive per-RG `RowFilter` / `BulkDecode` switching (`#207`):**
    - **`q6`:** `480.2 → 345.0 ms` (**`1.39×`**, `−135.1 ms`, `4` RF / `236`
      Bulk RGs), fully recovering the `#195` pushdown penalty (`345.0` vs.
      `morsel_no_pushdown` `340.0 ms`), beating `stock_default` (`0.93×`) and
      `stock_pushdown` (`0.67×`).
    - **`q15`:** `955.3 → 704.4 ms` (**`1.36×`**, `−250.9 ms`, `30` RF / `450`
      Bulk RGs), recovering within `2.4%` of `morsel_no_pushdown` (`688.0 ms`)
      and beating `stock_default` (`0.96×`) and `stock_pushdown` (`0.77×`).
    - **`q12`:** `267.7 → 208.1 ms` (**`1.29×`**, `−59.6 ms`, `65` RF / `239`
      Bulk RGs), beating `morsel_no_pushdown` (`224.5 ms`), `stock_default`
      (`219.3 ms`, `0.95×`), `stock_pushdown` (`314.6 ms`, `0.66×`), **and
      DuckDB 2.0.1 (`219.5 ms`, `0.95×`)**.
    - **`q9` (partial delta):** `1,418.0 → 1,397.5 ms` (`1.01×`, `−20.5 ms`,
      `245` RF / `4` Bulk RGs); `lineitem` stays in `RowFilter`, leaving a
      `60.7 ms` (`4.5%`) gap vs. `morsel_no_pushdown` (`1,336.8 ms`) while still
      beating `stock_pushdown` (`1,527.2 ms`, `0.92×`).
  - **Aggregate accumulator CSE, `AVG` sharing & direct `Decimal128` (`#208`):**
    - **`q1`:** `721.7 → 611.2 ms` (**`1.18×`**, `−110.5 ms` in
      `morsel_pushdown`; `720.3 → 596.1 ms`, **`1.21×`** in
      `morsel_no_pushdown`), cutting **`morsel_process_work` `1.90×`**
      (`1,930.1 → 1,014.5 ms`, `−915.6 ms` / `−47.4%` aggregate CPU) and
      beating `stock_default` (`794.1 ms`) by **`1.30×` (`0.77×`)** and
      `stock_pushdown` (`766.5 ms`) by **`1.25×` (`0.80×`)**.
    - **`q6` & `q15`:** Fused `Decimal128` arithmetic also cuts
      `morsel_no_pushdown` `process_work` by `1.27×` on `q6` (`222.9 → 174.9 ms`)
      and `1.23×` on `q15` (`538.3 → 435.9 ms`).

## Environment, Controls, and Reproduction Commands

### Host and Build Metadata

- **Host:** `rob-tandy-dduck2`, Ubuntu 24.04.5 LTS, Linux `7.0.0-1012-aws`,
  16 ARM Neoverse-N1 cores (1 thread/core, 64 KiB L1d / 1 MiB L2 per core,
  32 MiB shared L3), 61 GiB RAM.
- **CPU pinning and threads:** `taskset -c 0-7` (8 dedicated CPUs),
  `--threads 8` (Tokio `worker_threads=8`, `max_blocking_threads=8`,
  `target_partitions=8`; DuckDB `SET threads = 8`), batch size `8,192`.
- **Host idle verification (`vmstat -y 1 3`):**
  - Immediately before 22/22 scorecard: **92–93% idle** (`us 6–7%, sy 0–1%`).
  - Immediately after 22/22 scorecard: **75% → 86% → 93% idle** as the final
    benchmark subprocess exited (`us 6%, sy 1%, id 93%` once settled).
- **Build & versions:**
  - DataFusion revision: `78e46e910b29427b831ec2c8a7f8da1f6ca42b10`
    (`agent/morsel-issue-209`, reporting `adaptive_*` RG metrics in
    `tpch_runner.rs` on top of `morsel-execution` HEAD `129c1b227` after `#206`,
    `#207`, and `#208`), built with `--locked --profile release-nonlto -p
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
- **Protocol:** Exact `#195` protocol — fresh process per trial, `1` unmeasured
  warmup round + `5` measured counterbalanced rounds (alternating arm order
  between odd and even rounds), `MORSEL_METRICS=1` enabled on every Morsel
  trial, lossless Arrow IPC parity verified against DuckDB 2.0.1 across all 6
  arms before timing. All timings below are medians in milliseconds with
  observed `min–max` in parentheses.

### Exact Reproduction Commands

```bash
set -euo pipefail
b=benchmarks/morsel_tpch
work="$HOME/.cache/issue209"
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

# 2. Run all 21/21 SF=0.01 parity/plan/spill/adaptive-metric unittests.
py="$HOME/.cache/issue195/venv/bin/python"
MORSEL_TPCH_RUNNER="$runner" MORSEL_BENCH_REVISION="$revision" \
  timeout 240s "$py" -B -m unittest discover -s "$b" -v

# 3. Run quiet-host warm-cache SF5 22/22 6-arm scorecard (660 timed trials).
data_dir="$HOME/.cache/issue189/tpch_sf5"
vmstat -y 1 3
timeout --kill-after=10s 2400s taskset -c 0-7 "$py" -B "$b/harness.py" bench \
  "$data_dir" --runner "$runner" --revision "$revision" \
  --threads 8 --timeout 180 --cache warm --warmups 1 --repeats 5 \
  --queries all \
  --morsel-arms stock_default,stock_pushdown,morsel_no_pushdown,morsel_pruning,morsel_pushdown,duckdb_default \
  > "$work/sf5_22q_wave4_scorecard.jsonl"
vmstat -y 1 3

# 4. Print summary and 3-way geometric mean table.
"$py" -B "$b/harness.py" summarize "$work/sf5_22q_wave4_scorecard.jsonl"

# 5. Clean up the detached worktree when finished.
git worktree remove --force "$work/pinned"
```

## 1. Full 22/22 TPC-H SF5 3-Way Warm-Cache Scorecard (Wave-4)

Every query verified exact Arrow IPC parity against DuckDB 2.0.1 across all 6
arms before timing and ran with `morsel_mode=required` (`0` fallback, `100%`
`mincore` page residency on all `660 / 660` trials). In the Morsel columns,
**bold** marks the fastest Morsel arm per query (`morsel_pushdown` wins
`16 / 22`). The ratio columns compare **single-config `morsel_pushdown`**
directly against `stock_default`, `stock_pushdown`, and `duckdb_default`
(`< 1.00×` means single-config Morsel is faster; bold highlights wins).

| Query               |   Rows | `stock_default` ms (min–max) | `stock_pushdown` ms (min–max) | `morsel_no_pushdown` ms (min–max) | `morsel_pruning` ms (min–max) | `morsel_pushdown` ms (min–max) | `duckdb_default` ms (min–max) | `morsel_pd` / Stock Def | `morsel_pd` / Stock PD | `morsel_pd` / DuckDB |
| ------------------- | -----: | ---------------------------: | ----------------------------: | --------------------------------: | ----------------------------: | -----------------------------: | ----------------------------: | ----------------------: | ---------------------: | -------------------: |
| `q1`                |      4 |          794.1 (762.8–803.8) |           766.5 (748.1–779.2) |           **596.1** (590.1–607.6) |           606.1 (600.4–678.2) |            611.2 (603.7–697.2) |           479.5 (463.6–536.2) |               **0.77×** |              **0.80×** |                1.27× |
| `q2`                |    100 |          181.3 (176.5–192.3) |           175.5 (170.1–181.7) |               240.7 (239.4–244.6) |           252.6 (233.3–259.4) |        **169.8** (167.8–173.0) |           121.9 (113.2–123.7) |               **0.94×** |              **0.97×** |                1.39× |
| `q3`                |     10 |          537.1 (521.1–610.0) |           401.1 (398.6–407.7) |               557.6 (550.0–564.1) |           552.3 (549.6–561.5) |        **394.3** (391.0–401.0) |           347.9 (341.6–352.8) |               **0.73×** |              **0.98×** |                1.13× |
| `q4`                |      5 |          201.1 (200.4–204.0) |           291.4 (284.4–299.5) |               311.3 (306.6–321.3) |           316.4 (313.9–317.8) |        **226.5** (214.9–246.6) |           191.7 (188.4–194.7) |                   1.13× |              **0.78×** |                1.18× |
| `q5`                |      5 |          719.6 (711.9–749.9) |           487.0 (485.3–495.9) |               762.3 (758.7–764.3) |           768.7 (758.2–798.8) |        **444.9** (430.5–464.5) |           379.5 (377.0–389.5) |               **0.62×** |              **0.91×** |                1.17× |
| `q6`                |      1 |          371.3 (363.9–380.2) |           514.8 (504.0–524.3) |           **340.0** (338.4–347.7) |           342.7 (340.7–348.0) |            345.0 (337.5–352.4) |           203.3 (201.3–204.5) |               **0.93×** |              **0.67×** |                1.70× |
| `q7`                |      4 |          891.4 (867.9–896.1) |           510.7 (494.4–535.3) |         1,017.8 (1,002.9–1,048.0) |     1,031.0 (1,005.6–1,038.5) |        **435.7** (434.2–438.8) |           412.6 (410.1–414.5) |               **0.49×** |              **0.85×** |                1.06× |
| `q8`                |      2 |          664.2 (653.4–677.1) |           652.6 (645.4–664.9) |               727.6 (712.8–731.8) |           731.1 (717.6–743.4) |        **568.9** (561.1–597.0) |           457.1 (451.7–461.7) |               **0.86×** |              **0.87×** |                1.24× |
| `q9`                |    175 |    1,023.6 (1,011.4–1,056.2) |     1,527.2 (1,520.8–1,562.6) |     **1,336.8** (1,309.8–1,366.0) |     1,363.4 (1,305.3–1,376.0) |      1,397.5 (1,384.9–1,444.8) |           788.4 (778.8–799.4) |                   1.37× |              **0.92×** |                1.77× |
| `q10`               |     20 |          654.8 (639.5–678.0) |           527.3 (511.1–538.9) |               767.4 (755.5–775.4) |           770.5 (760.0–783.4) |        **616.0** (604.8–616.7) |           457.0 (445.5–467.8) |               **0.94×** |                  1.17× |                1.35× |
| `q11`               |  4,606 |          112.9 (111.1–113.8) |           127.9 (124.1–133.2) |               190.3 (185.3–200.1) |           193.2 (188.8–207.3) |        **135.2** (130.4–137.5) |              95.7 (93.5–99.5) |                   1.20× |                  1.06× |                1.41× |
| `q12`               |      2 |          219.3 (212.9–219.9) |           314.6 (298.2–319.8) |               224.5 (218.0–228.5) |           224.5 (220.2–231.1) |        **208.1** (203.6–232.8) |           219.5 (217.6–243.6) |               **0.95×** |              **0.66×** |            **0.95×** |
| `q13`               |     42 |          479.1 (468.9–534.9) |           579.0 (562.5–597.9) |               570.1 (549.3–594.4) |       **562.5** (554.7–576.9) |            590.7 (586.1–611.9) |           519.2 (516.7–542.2) |                   1.23× |                  1.02× |                1.14× |
| `q14`               |      1 |          402.5 (397.3–414.5) |           300.6 (296.2–312.1) |               426.7 (419.2–436.0) |           430.7 (423.2–434.0) |        **341.1** (340.1–346.2) |           336.5 (331.1–339.2) |               **0.85×** |                  1.13× |                1.01× |
| `q15`               |      1 |          737.4 (724.3–757.2) |           909.2 (904.0–920.5) |           **688.0** (686.8–699.8) |           694.4 (678.7–700.6) |            704.4 (690.6–714.8) |           272.3 (265.0–277.0) |               **0.96×** |              **0.77×** |                2.59× |
| `q16`               | 27,724 |          132.6 (128.8–135.4) |           143.4 (136.8–151.9) |           **169.5** (169.2–171.0) |           173.3 (164.2–175.2) |            172.0 (169.1–175.1) |           141.0 (136.7–147.5) |                   1.30× |                  1.20× |                1.22× |
| `q17`               |      1 |    1,222.8 (1,217.8–1,274.8) |           468.0 (465.8–472.3) |         1,341.7 (1,297.9–1,403.7) |     1,341.0 (1,309.4–1,383.3) |        **499.8** (494.4–515.4) |           405.9 (405.0–410.8) |               **0.41×** |                  1.07× |                1.23× |
| `q18`               |    100 |    1,475.9 (1,429.3–1,492.1) |           844.2 (820.5–862.0) |         1,920.2 (1,862.8–1,937.9) |     1,875.8 (1,860.0–1,958.5) |        **670.9** (656.9–704.3) |           477.7 (472.8–481.8) |               **0.45×** |              **0.79×** |                1.40× |
| `q19`               |      1 |          569.5 (563.3–578.9) |           429.4 (422.7–438.5) |               497.6 (493.6–503.4) |           505.3 (497.8–509.9) |        **395.3** (394.6–399.5) |           337.5 (336.6–355.0) |               **0.69×** |              **0.92×** |                1.17× |
| `q20`               |    901 |          473.0 (455.4–483.8) |           322.6 (315.4–332.6) |               710.1 (691.5–722.5) |           712.2 (697.2–721.6) |        **329.3** (325.0–332.0) |           307.0 (295.7–313.9) |               **0.70×** |                  1.02× |                1.07× |
| `q21`               |    100 |    1,397.0 (1,365.5–1,432.5) |     1,533.8 (1,494.8–1,554.8) |         1,806.6 (1,785.1–1,818.4) |     1,780.1 (1,773.5–1,811.2) |        **937.0** (917.0–966.7) |           741.7 (731.4–760.5) |               **0.67×** |              **0.61×** |                1.26× |
| `q22`               |      7 |          124.0 (122.7–127.5) |           142.3 (135.6–154.4) |               169.0 (166.5–174.1) |           171.6 (164.6–178.8) |        **126.3** (125.8–127.2) |           144.7 (142.6–151.1) |                   1.02× |              **0.89×** |            **0.87×** |
| **GeoMean (22/22)** |      — |                 **471.3 ms** |                  **435.7 ms** |                      **546.0 ms** |                  **550.1 ms** |                   **391.2 ms** |                  **309.5 ms** |               **0.83×** |              **0.90×** |            **1.26×** |
| **Sum (22/22)**     |      — |              **13,384.5 ms** |               **11,969.1 ms** |                   **15,371.8 ms** |               **15,399.3 ms** |                **10,319.9 ms** |                **7,837.7 ms** |               **0.77×** |              **0.86×** |            **1.32×** |

_(Per-query oracle `Best Morsel` achieves **`388.1 ms`** geomean /
**`10,192.0 ms`** sum — `0.82×` vs. `stock_default`, `0.89×` vs.
`stock_pushdown`, `1.25×` vs. `duckdb_default` — just `3.0 ms` / `0.78%` ahead
of single-config `morsel_pushdown`.)_

## 2. Direct Before vs. After Comparison (`#195` vs. Wave-4)

Comparing `#195` (`d78ecdab8`, post-`#193`/`#194`) directly against Wave-4
(`78e46e910`, post-`#206`/`#207`/`#208`) on identical SF5 Parquet tables and
`rob-tandy-dduck2`:

| Query               | `#195` `morsel_pushdown` | Wave-4 `morsel_pushdown` | `pushdown` Speedup | `pushdown` Delta ms | `#195` Best Morsel | Wave-4 Best Morsel | Best Speedup | W4 `morsel_pd` / Stock Def | W4 `morsel_pd` / Stock PD | W4 `morsel_pd` / DuckDB |
| ------------------- | -----------------------: | -----------------------: | -----------------: | ------------------: | -----------------: | -----------------: | -----------: | -------------------------: | ------------------------: | ----------------------: |
| `q1`                |                    721.7 |                    611.2 |          **1.18×** |              −110.5 |              720.3 |              596.1 |    **1.21×** |                  **0.77×** |                 **0.80×** |                   1.27× |
| `q2`                |                    194.2 |                    169.8 |          **1.14×** |               −24.4 |              194.2 |              169.8 |    **1.14×** |                  **0.94×** |                 **0.97×** |                   1.39× |
| `q3`                |                    401.3 |                    394.3 |              1.02× |                −7.0 |              401.3 |              394.3 |        1.02× |                  **0.73×** |                 **0.98×** |                   1.13× |
| `q4`                |                    226.2 |                    226.5 |              1.00× |                +0.3 |              226.2 |              226.5 |        1.00× |                      1.13× |                 **0.78×** |                   1.18× |
| `q5`                |                    441.3 |                    444.9 |              0.99× |                +3.6 |              441.3 |              444.9 |        0.99× |                  **0.62×** |                 **0.91×** |                   1.17× |
| `q6`                |                    480.2 |                    345.0 |          **1.39×** |              −135.1 |              353.3 |              340.0 |        1.04× |                  **0.93×** |                 **0.67×** |                   1.70× |
| `q7`                |                    444.4 |                    435.7 |              1.02× |                −8.7 |              444.4 |              435.7 |        1.02× |                  **0.49×** |                 **0.85×** |                   1.06× |
| `q8`                |                    554.4 |                    568.9 |              0.97× |               +14.6 |              554.4 |              568.9 |        0.97× |                  **0.86×** |                 **0.87×** |                   1.24× |
| `q9`                |                  1,418.0 |                  1,397.5 |              1.01× |               −20.5 |            1,378.7 |            1,336.8 |        1.03× |                      1.37× |                 **0.92×** |                   1.77× |
| `q10`               |                    609.0 |                    616.0 |              0.99× |                +7.0 |              609.0 |              616.0 |        0.99× |                  **0.94×** |                     1.17× |                   1.35× |
| `q11`               |                    144.4 |                    135.2 |          **1.07×** |                −9.1 |              144.4 |              135.2 |    **1.07×** |                      1.20× |                     1.06× |                   1.41× |
| `q12`               |                    267.7 |                    208.1 |          **1.29×** |               −59.6 |              231.6 |              208.1 |    **1.11×** |                  **0.95×** |                 **0.66×** |               **0.95×** |
| `q13`               |                    596.8 |                    590.7 |              1.01× |                −6.1 |              577.0 |              562.5 |        1.03× |                      1.23× |                     1.02× |                   1.14× |
| `q14`               |                    339.2 |                    341.1 |              0.99× |                +1.9 |              339.2 |              341.1 |        0.99× |                  **0.85×** |                     1.13× |                   1.01× |
| `q15`               |                    955.3 |                    704.4 |          **1.36×** |              −250.9 |              710.3 |              688.0 |        1.03× |                  **0.96×** |                 **0.77×** |                   2.59× |
| `q16`               |                    171.9 |                    172.0 |              1.00× |                +0.1 |              171.9 |              169.5 |        1.01× |                      1.30× |                     1.20× |                   1.22× |
| `q17`               |                  1,238.2 |                    499.8 |          **2.48×** |              −738.4 |            1,238.2 |              499.8 |    **2.48×** |                  **0.41×** |                     1.07× |                   1.23× |
| `q18`               |                    693.3 |                    670.9 |              1.03× |               −22.3 |              693.3 |              670.9 |        1.03× |                  **0.45×** |                 **0.79×** |                   1.40× |
| `q19`               |                    414.5 |                    395.3 |          **1.05×** |               −19.2 |              414.5 |              395.3 |    **1.05×** |                  **0.69×** |                 **0.92×** |                   1.17× |
| `q20`               |                    774.1 |                    329.3 |          **2.35×** |              −444.8 |              732.1 |              329.3 |    **2.22×** |                  **0.70×** |                     1.02× |                   1.07× |
| `q21`               |                  1,544.1 |                    937.0 |          **1.65×** |              −607.1 |            1,544.1 |              937.0 |    **1.65×** |                  **0.67×** |                 **0.61×** |                   1.26× |
| `q22`               |                    127.7 |                    126.3 |              1.01× |                −1.5 |              127.7 |              126.3 |        1.01× |                      1.02× |                 **0.89×** |               **0.87×** |
| **GeoMean (22/22)** |             **461.0 ms** |             **391.2 ms** |          **1.18×** |        **−69.8 ms** |       **443.2 ms** |       **388.1 ms** |    **1.14×** |                  **0.83×** |                 **0.90×** |               **1.26×** |
| **Sum (22/22)**     |          **12,757.6 ms** |          **10,319.9 ms** |          **1.24×** |     **−2,437.7 ms** |    **12,247.1 ms** |    **10,192.0 ms** |    **1.20×** |                  **0.77×** |                 **0.86×** |               **1.32×** |

## 3. Deep-Dive Attribution Across the Three Wave-4 Optimizations

### 3.1 Cross-`Aggregate` Dynamic Join Pushdown & Build-Side Selection (`#206`)

Whole-DAG cross-`Aggregate` filter propagation (`stage_0`'s `lineitem` scan
pruned by downstream finalized `part`/`partsupp`/`supplier` builds) plus
selective join build-side estimation delivered the largest single-query wall-time
drops of Wave-4 (`−1,814.7 ms` combined across `q17`, `q20`, `q21`, and `q2` —
nearly 75% of the total `−2,437.7 ms` suite reduction):

| Query     | `#195` → W4 wall ms (`morsel_pd`) |   Speedup | Exec CPU `#195 → W4` | `process_work` `#195 → W4` | `join_finalize` `#195 → W4` | `dyn_filters` | Mechanism                                                                                      |
| --------- | --------------------------------: | --------: | -------------------: | -------------------------: | --------------------------: | ------------: | ---------------------------------------------------------------------------------------------- |
| **`q17`** |              `1,238.2 → 499.8 ms` | **2.48×** |  `7,077.1 → 3,440.4` |       **`2,156.9 → 25.7`** |                `27.2 → 0.8` |       `6 → 6` | `part(0.1%)` build placed on build + pushed across `stage_0`; `stage_0` proc `2,078 → 10.4 ms` |
| **`q20`** |                `774.1 → 329.3 ms` | **2.35×** |  `3,783.9 → 2,036.1` |         **`786.0 → 18.3`** |                 `1.4 → 0.8` |      `4 → 12` | `supplier ⋈ nation` cascades into `partsupp ⋉ part` and across `stage_0` (`lineitem`)          |
| **`q21`** |              `1,544.1 → 937.0 ms` | **1.65×** |  `7,916.0 → 6,143.0` |          `941.2 → 1,245.9` |         **`515.7 → 150.0`** |     `10 → 13` | Selective build-side estimation eliminates 19M-row `l3` hash build (`join_fin` `3.44×` lower)  |
| **`q2`**  |                `194.2 → 169.8 ms` | **1.14×** |      `941.3 → 782.8` |            `127.3 → 135.3` |            **`22.2 → 2.1`** |     `21 → 19` | Filtered `partsupp ⋈ part ⋈ ...` placed on build (`join_finalize` `10.6×` lower)               |

Cross-stage propagation also activates on `q15` (`join_dynamic_filters_pushed`:
`0 → 3`, pushing `supplier`'s `s_suppkey` filter into `stage_2`), bringing
dynamic join filter pushdown coverage to **`20 / 22` TPC-H queries** (every
join query in the suite; only single-table `q1` and `q6` have no joins).

### 3.2 Adaptive Per-RG `RowFilter` / `BulkDecode` Recovery (`#207`)

In `#195`, single-config `morsel_pushdown` lagged `morsel_no_pushdown` by
`17.8 ms` on geomean because dense static-filter scans (`q6`, `q12`, `q15`,
`q9`) paid Parquet `RowFilter` overhead without benefiting from column skipping.
Wave-4's per-RG `RowFilter` vs. `BulkDecode + BatchView` bandit recovers `q6`,
`q12`, and `q15` cleanly while preserving `RowFilter` on selective dynamic-join
probes (`q3`, `q5`, `q7`, `q8`, `q10`, `q14`, `q17`, `q18`, `q19`, `q20`):

| Query     | `#195` `morsel_pd` | W4 `morsel_pd` | `#195 → W4` | W4 `no_pd` | W4 `stock_def` | W4 `stock_pd` | W4 DuckDB |  Decode `#195 → W4` | Adaptive `RF / Bulk` RGs (`switches`) | Outcome                                                                        |
| --------- | -----------------: | -------------: | ----------: | ---------: | -------------: | ------------: | --------: | ------------------: | ------------------------------------: | ------------------------------------------------------------------------------ |
| **`q6`**  |              480.2 |      **345.0** |   **1.39×** |      340.0 |          371.3 |         514.8 |     203.3 | `3,109.8 → 2,296.1` |                       `4 / 236` (`1`) | Fully recovered (within `1.5%` of `no_pd`; beats both Stock arms)              |
| **`q12`** |              267.7 |      **208.1** |   **1.29×** |      224.5 |          219.3 |         314.6 |     219.5 |   `1,414.5 → 939.9` |                      `65 / 239` (`2`) | Beats `no_pd`, both Stock arms, **and DuckDB** (`lineitem` Bulk + `orders` RF) |
| **`q15`** |              955.3 |      **704.4** |   **1.36×** |      688.0 |          737.4 |         909.2 |     272.3 | `5,825.2 → 4,407.3` |                      `30 / 450` (`2`) | Recovered within `2.4%` of `no_pd`; beats both Stock arms                      |
| **`q9`**  |            1,418.0 |        1,397.5 |       1.01× |    1,336.8 |        1,023.6 |       1,527.2 |     788.4 | `5,100.7 → 5,194.6` |                       `245 / 4` (`1`) | Partial (`−20.5 ms`); `lineitem` stays in `RowFilter`, still beats `stock_pd`  |

- **Side effects & honest accounting:**
  - `q11` (`144.4 → 135.2 ms`, `1.07×`, `38/30` RF/Bulk), `q18`
    (`693.3 → 670.9 ms`, `1.03×`, `297/4`), and `q19` (`414.5 → 395.3 ms`,
    `1.05×`, `245/3`) also gain `3–7%` from `#207`'s adaptive filtering and
    conjunct ordering.
  - Small regressions (`≤ 2.6%`) appear on `q8` (`554.4 → 568.9 ms`, `+14.6 ms`)
    and `q10` (`609.0 → 616.0 ms`, `+7.0 ms`), where the 2-RG initial
    `BulkDecode` probe adds minor per-scan overhead; both remain well ahead of
    `stock_default` (`0.86×` and `0.94×`).
  - On `q9`, `lineitem` commits to `RowFilter` (`245` RF / `4` Bulk RGs) because
    per-RG filter cost differences are small relative to the 5-way hash join and
    aggregate CPU, leaving `60.7 ms` (`4.5%`) between `morsel_pushdown` and
    `morsel_no_pushdown` (and `q9` remains `129.7 ms` faster than
    `stock_pushdown`).

### 3.3 Aggregate Accumulator CSE, `AVG` Sharing & Direct `Decimal128` (`#208`)

`#208` eliminates redundant `SUM`/`COUNT` accumulators for `AVG`, removes
intermediate `Decimal128Array` expression allocations via fused arithmetic loops,
and interns `(l_returnflag, l_linestatus)` directly via a 64-slot cache over
inlined `Utf8View` keys:

| Query & Arm                  | `#195` wall ms | W4 wall ms | Wall Speedup | `#195` exec ms | W4 exec ms | `#195` `process_work` | W4 `process_work` | `process_work` Speedup |
| ---------------------------- | -------------: | ---------: | -----------: | -------------: | ---------: | --------------------: | ----------------: | ---------------------: |
| **`q1` (`morsel_pushdown`)** |          721.7 |  **611.2** |    **1.18×** |          708.6 |      598.4 |               1,930.1 |       **1,014.5** |              **1.90×** |
| **`q1` (`morsel_no_pd`)**    |          720.3 |  **596.1** |    **1.21×** |          708.9 |      584.0 |               2,274.5 |       **1,279.8** |              **1.78×** |
| **`q6` (`morsel_no_pd`)**    |          357.7 |  **340.0** |    **1.05×** |          346.9 |      329.4 |                 222.9 |         **174.9** |              **1.27×** |
| **`q15` (`morsel_no_pd`)**   |          710.3 |  **688.0** |        1.03× |          697.3 |      674.4 |                 538.3 |         **435.9** |              **1.23×** |

- On **`q1`**, `morsel_process_work` drops by **`−915.6 ms` (`−47.4%`)** in
  `morsel_pushdown` (`−994.7 ms` / `−43.7%` in `morsel_no_pushdown`), bringing
  `q1` from `0.93×` to **`0.77×` of `stock_default` (`1.30×` faster)** and
  **`0.80×` of `stock_pushdown` (`1.25×` faster)**, and cutting the gap to
  DuckDB 2.0.1 (`479.5 ms`) from `1.56× → 1.27×` (`1.24×` on `morsel_no_pd`).
  The remaining `15.1 ms` difference between `morsel_pushdown` (`611.2 ms`,
  `209` RF / `31` Bulk RGs) and `morsel_no_pushdown` (`596.1 ms`) comes from
  the ~98.5%-pass `l_shipdate <= '1998-09-02'` predicate committing `RowFilter`
  (`decode_work` `3,429.0` vs. `3,174.6 ms`).

## 4. Native `MorselExec` Execution & `#138` Wait/Work Summary (Single-Config `morsel_pushdown`)

Every row below reports **single-config `morsel_pushdown`** (`MORSEL_METRICS=1`,
`morsel_mode=required`, 8 workers). `Decode`, `Process`, and `Ready wait` are
summed across 8 workers; `Output wait` was **`0.0 ms`** on all 330 measured
Morsel trials. Compared to `#195`'s `morsel_pushdown` totals across 22/22:

- Total `Exec` wall time fell **`1.24×`** (`12,432.5 → 9,997.8 ms`, `−2,434.7 ms`).
- Total `Exec CPU` fell **`−13.5%`** (`78,425.4 → 67,822.5 ms`, `−10.60 s`).
- Total `Decode` work fell **`−8.5%`** (`51,298.9 → 46,943.7 ms`).
- Total `Process` work fell **`−15.8%`** (`12,766.4 → 10,755.0 ms`).
- Total `Join fin` wall time fell **`1.57×`** (`1,163.4 → 742.1 ms` — down
  **`9.78×`** from `#192`'s `7,258.4 ms`).

| Query             | `mincore` pages (100%) | RGs claimed | Dyn filt | Adapt `RF/Bulk` | Stages / `jb` / `agg` / `gc` / `s` |     Exec ms |  Exec CPU ms |    Decode ms |   Process ms | Ready wait ms | Join fin ms | Sort/TopN fin ms | Agg fin / other ms |
| ----------------- | ---------------------: | ----------: | -------: | --------------: | ---------------------------------- | ----------: | -----------: | -----------: | -----------: | ------------: | ----------: | ---------------: | -----------------: |
| `q1`              |                229,147 |         240 |        0 |        `209/31` | `0 / 0 / 1 / 0 / Sort`             |       598.4 |      4,544.3 |      3,429.0 |      1,014.5 |          30.3 |         0.0 |              0.1 |               42.8 |
| `q2`              |                 44,595 |          78 |       19 |          `74/4` | `1 / 8 / 1 / 0 / TopN`             |       157.6 |        782.8 |        554.1 |        135.3 |         111.1 |         2.1 |              0.1 |               69.2 |
| `q3`              |                297,360 |         310 |        6 |         `306/4` | `0 / 2 / 1 / 0 / TopN`             |       379.3 |      2,563.8 |      1,933.5 |        174.9 |         122.7 |        29.2 |              0.2 |               86.3 |
| `q4`              |                286,442 |         304 |        3 |        `247/57` | `0 / 1 / 1 / 0 / Sort`             |       212.3 |      1,417.1 |      1,125.2 |        114.2 |          60.4 |        10.3 |              0.0 |               47.1 |
| `q5`              |                298,107 |         313 |       15 |         `308/5` | `0 / 5 / 1 / 0 / Sort`             |       427.7 |      2,915.8 |      2,360.6 |        184.0 |         161.9 |        12.8 |              0.0 |               96.8 |
| `q6`              |                229,147 |         240 |        0 |         `4/236` | `0 / 0 / 1 / 0 / —`                |       332.1 |      2,546.0 |      2,296.1 |        175.4 |          17.9 |         0.0 |              0.0 |               23.2 |
| `q7`              |                298,106 |         313 |       15 |        `263/50` | `0 / 5 / 1 / 0 / Sort`             |       417.9 |      2,824.0 |      2,222.0 |        165.0 |         186.7 |        22.5 |              0.0 |               97.0 |
| `q8`              |                303,951 |         322 |       19 |        `278/43` | `0 / 7 / 1 / 0 / Sort`             |       548.7 |      3,759.6 |      3,123.3 |        156.7 |         216.3 |        21.4 |              0.0 |              117.3 |
| `q9`              |                331,036 |         346 |       16 |         `245/4` | `0 / 5 / 1 / 0 / Sort`             |     1,378.5 |      9,118.2 |      5,194.6 |      2,268.5 |         228.3 |       276.5 |              0.0 |              169.1 |
| `q10`             |                297,361 |         311 |        9 |         `305/5` | `0 / 3 / 1 / 0 / TopN`             |       599.1 |      3,893.1 |      2,106.6 |        792.0 |         176.5 |       100.0 |              0.2 |              136.5 |
| `q11`             |                 38,750 |          68 |        8 |         `38/30` | `0 / 4 / 2 / 0 / Sort`             |       126.1 |        788.2 |        620.1 |        114.5 |          56.6 |         0.1 |              0.5 |               33.6 |
| `q12`             |                286,442 |         304 |        3 |        `65/239` | `0 / 1 / 1 / 0 / Sort`             |       192.0 |      1,346.6 |        939.9 |        245.7 |          49.9 |         5.7 |              0.0 |               38.1 |
| `q13`             |                 68,213 |          70 |        3 |         `26/38` | `1 / 1 / 1 / 1 / Sort`             |       583.2 |      3,746.2 |      1,548.3 |      1,732.3 |          57.1 |        26.8 |              0.0 |              146.3 |
| `q14`             |                234,991 |         248 |        3 |         `237/3` | `0 / 1 / 1 / 0 / —`                |       326.7 |      2,215.3 |      1,556.0 |        179.0 |         114.8 |        35.7 |              0.0 |               74.1 |
| `q15`             |                229,892 |         481 |        3 |        `30/450` | `2 / 1 / 3 / 0 / Sort`             |       689.0 |      5,166.8 |      4,407.3 |        500.6 |          42.2 |         3.3 |              0.0 |               72.1 |
| `q16`             |                 44,593 |          41 |        3 |          `39/2` | `1 / 2 / 2 / 0 / Sort`             |       163.1 |        736.7 |        253.9 |        266.2 |         119.5 |         6.3 |              5.4 |               86.4 |
| `q17`             |                234,991 |         488 |        6 |         `480/8` | `1 / 2 / 2 / 0 / —`                |       485.0 |      3,440.4 |      3,153.7 |         25.7 |         136.5 |         0.8 |              0.0 |               86.7 |
| `q18`             |                297,360 |         547 |        9 |         `297/4` | `1 / 3 / 2 / 0 / TopN`             |       655.1 |      4,508.0 |      1,965.7 |      1,092.3 |         112.6 |        32.8 |              0.0 |              240.0 |
| `q19`             |                234,991 |         248 |        3 |         `245/3` | `0 / 1 / 1 / 0 / —`                |       378.7 |      2,608.3 |      2,313.3 |         24.6 |         160.8 |         1.4 |              0.0 |               85.1 |
| `q20`             |                273,741 |         282 |       12 |         `276/6` | `1 / 4 / 1 / 0 / Sort`             |       312.3 |      2,036.1 |      1,761.1 |         18.3 |         174.9 |         0.8 |              0.0 |               89.1 |
| `q21`             |                287,188 |         786 |       13 |       `528/258` | `0 / 5 / 1 / 0 / TopN`             |       918.5 |      6,143.0 |      3,551.4 |      1,245.9 |         174.5 |       150.0 |              0.0 |              168.9 |
| `q22`             |                 68,213 |          76 |        3 |          `74/2` | `0 / 1 / 2 / 0 / Sort`             |       116.4 |        722.3 |        528.0 |        129.3 |         140.0 |         3.7 |              0.0 |               30.6 |
| **Total (22/22)** |                      — |           — |        — |               — | —                                  | **9,997.8** | **67,822.5** | **46,943.7** | **10,755.0** |   **2,651.6** |   **742.1** |          **6.9** |        **2,036.5** |

### Remaining Headroom vs. DuckDB 2.0.1 (`1.26×` GeoMean / `1.32×` Sum)

With single-config `morsel_pushdown` now ahead of both `stock_default` (`0.83×`)
and `stock_pushdown` (`0.90×`), the remaining `2,482.2 ms` sum gap to DuckDB
2.0.1 (`10,319.9` vs. `7,837.7 ms`) is highly concentrated in four areas:

1. **`q9` (`1,397.5` vs. `788.4 ms`, `+609.1 ms`) & `q21` (`937.0` vs.
   `741.7 ms`, `+195.3 ms`):** Large multi-way hash-join build/probe CPU and
   Parquet decode on `lineitem` (`q9` decode `5,194.6 ms` + process `2,268.5 ms`
   - `join_finalize` `276.5 ms`).
2. **Stage CSE on `q15` (`704.4` vs. `272.3 ms`, `+432.1 ms`):** Materializing
   `revenue0` once instead of scanning `lineitem` twice (`stage_0` + `stage_2`:
   `605.1 MB` scanned, `4,407.3 ms` decode) would cut `q15` roughly in half.
3. **Parquet `Decimal128` decode throughput on `q1`, `q6`, and `q18`
   (`+466.8 ms` combined):** `q1` (`611.2` vs. `479.5 ms`), `q6` (`345.0` vs.
   `203.3 ms`), and `q18` (`670.9` vs. `477.7 ms`) are now overwhelmingly
   dominated by Parquet Zstd/FixedLenByteArray `Decimal128` decoding (`75–90%`
   of total CPU on `q1` and `q6`).
4. **High-selectivity scan `RowFilter` avoidance (`q1`, `q9`):** Avoiding
   `RowFilter` when static predicates pass nearly all rows (`q1` `l_shipdate`)
   or when downstream join CPU dominates (`q9`) would close the final `3.0 ms`
   oracle gap.

## Summary

Wave-4 (`#206` cross-`Aggregate` dynamic pushdown, `#207` adaptive per-RG
`RowFilter`/`BulkDecode` switching, and `#208` aggregate CSE + direct
`Decimal128` aggregation) achieves the project milestone: **single-config
Morsel (`morsel_pushdown`, `morsel_mode=required`) now beats BOTH
`stock_default` (`391.2` vs. `471.3 ms`, `0.83×` / `1.20×` faster; sum
`10,319.9` vs. `13,384.5 ms`, `0.77×`) AND `stock_pushdown` (`391.2` vs.
`435.7 ms`, `0.90×` / `1.11×` faster; sum `10,319.9` vs. `11,969.1 ms`,
`0.86×`) across the entire 22/22 TPC-H SF5 suite**, cuts the oracle gap to
`3.0 ms` (`< 0.8%`), wins `16 / 22` queries vs. `stock_default` and `15 / 22`
vs. `stock_pushdown`, beats DuckDB 2.0.1 on `q12` and `q22`, and closes the
overall 22/22 geomean gap to DuckDB 2.0.1 to **`1.26×`** (`391.2` vs.
`309.5 ms`) with **100% DuckDB parity**, **100% native admission**, and
**100% warm-cache `mincore` verification**.
