<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# TPC-H SF5 Paired Benchmark: Materialized CTE & Common Subplan (Stage) CSE (#228)

This note records the paired quiet-host Scale Factor 5 (`--sf 5`) before/after
benchmark and full 22/22 parity verification for **Materialized CTE & Common
Subplan (Stage) CSE** (`#228`) in the opt-in experimental morsel planner and
DAG runtime (`datafusion/core/src/morsel/`).

## Summary of Results

- **`q15` (`647.2 → 338.0 ms`, `1.91×` faster, `−309.2 ms`):** Inlined
  `revenue0` (`GROUP BY l_suppkey`) is automatically deduplicated into a single
  shared `Stage cse_0 (shared, consumers=2)` feeding both `scalar_sq_0`
  (`MAX(total_revenue)`) and the root `supplier ⋈ revenue0` join. Parquet row
  groups claimed drop **`481 → 241`** (`2.00×`), bytes scanned drop
  **`605.1 → 303.5 MB`** (`1.99×`), and `aggregate_finalizations` drops
  **`3 → 2`**.
- **`q11` (`102.1 → 60.4 ms`, `1.69×` faster, `−41.7 ms`):** Projection-subsumed
  Join CSE shares the filtered `partsupp ⋈ supplier ⋈ nation[GERMANY]` subplan
  as `Stage cse_0 (shared, consumers=2)` (preserving all internal dynamic join
  filter pushdown from `nation → supplier → partsupp`), feeding both
  `scalar_sq_0` (via zero-copy column projection) and the main `GROUP BY
  ps_partkey` aggregate. Row groups claimed drop **`68 → 34`** (`2.00×`), bytes
  scanned drop **`70.2 → 38.4 MB`** (`1.83×`), and `join_build_finalizations`
  drops **`4 → 2`**, flipping `q11` to beat DuckDB 2.0.1 (`92.4 ms`) at
  **`0.65×` (`1.53×` faster)**.
- **Zero regression on `q2`, `q17`, `q20`:** Single-consumer cross-`Aggregate`
  dynamic join filter pushdown (`#206`) is completely untouched — identical row
  groups claimed, identical bytes scanned, identical finalization counts, and
  medians within `±1%`.
- **Full 22/22 SF=5 parity & non-regression:** All **`22 / 22`** standard TPC-H
  queries pass lossless Arrow IPC parity against **DuckDB 2.0.1** under
  `morsel_mode=required` (`0` fallbacks, `0` rejections, `100%` warm-cache
  `mincore` page residency). Single-config `morsel_pushdown` achieves a
  **`295.2 ms` 22-query geometric mean vs. DuckDB 2.0.1's `296.1 ms`
  (`0.997×`)**, winning **`13 / 22`** queries (`q3`, `q4`, `q7`, `q8`, `q11`,
  `q12`, `q13`, `q14`, `q16`, `q17`, `q20`, `q21`, `q22`) and closing the
  22-query sum to **`7,697.1 ms` vs. `7,498.4 ms` (`1.026×`)**.

## Environment & Protocol

- **Host:** `rob-tandy-dduck1` (16 ARM Neoverse-N1 cores; benchmark pinned to 8
  cores via `taskset -c 0-7`, `--threads 8`; quiet host verified via
  `vmstat -y 1 3`: `95–99%` idle before and after).
- **Dataset:** Deterministic TPC-H SF=5 Parquet (`$HOME/.cache/issue193/tpch_sf5`,
  131,072-row row groups, 100% warm Linux page cache verified via `mincore`).
- **Engines & Revisions:**
  - **Before (base):** `morsel-execution` at `1663c620e` (`morsel_pushdown`,
    `morsel_mode=required`).
  - **After (PR `#228`):** `d974cba62` (`morsel_pushdown`,
    `morsel_mode=required`).
  - **Reference:** DuckDB `2.0.1` (`duckdb_default`), run paired in both
    sessions to confirm host stability.
- **Timing Protocol:** Fresh process per trial, `1` unmeasured warmup + `5`
  measured repeats for the paired 5-query focus set (`q15`, `q11`, `q2`, `q17`,
  `q20`), and `1` unmeasured warmup + `3` measured repeats for the full 22/22
  parity & non-regression sweep.

## 1. Paired Before/After SF=5 Focus Benchmark (`q15`, `q11`, `q2`, `q17`, `q20`)

Medians in milliseconds with `min–max` across 5 measured warm-cache repeats:

| Query | Before `morsel_pd` ms (min–max) | After `morsel_pd` ms (min–max) | Speedup | DuckDB-B ms | DuckDB-A ms | After / DuckDB | RGs Claimed (B→A) | Scanned MB (B→A) | Agg Fins (B→A) | Join Builds (B→A) |
| ----- | ------------------------------: | -----------------------------: | ------: | ----------: | ----------: | -------------: | ----------------: | ---------------: | -------------: | ----------------: |
| `q15` |             647.2 (645.3–665.0) |        **338.0** (337.8–349.2) | **1.91×** |       256.8 |       257.4 |          1.31× |     **481 → 241** | **605.1 → 303.5** |      **3 → 2** |             1 → 1 |
| `q11` |             102.1 (101.0–103.9) |         **60.4** (59.0–61.6)   | **1.69×** |        89.9 |        92.4 |      **0.65×** |       **68 → 34** |   **70.2 → 38.4** |          2 → 2 |         **4 → 2** |
| `q2`  |             143.2 (140.8–149.6) |            142.0 (140.6–143.1) |   1.01× |       118.4 |       116.5 |          1.22× |           78 → 78 |       69.1 → 69.1 |          1 → 1 |             8 → 8 |
| `q17` |             376.8 (376.6–379.8) |            375.8 (372.5–378.2) |   1.00× |       395.0 |       391.4 |      **0.96×** |         488 → 488 |     513.3 → 513.3 |          2 → 2 |             2 → 2 |
| `q20` |             273.7 (269.8–291.7) |            271.5 (270.2–281.7) |   1.01× |       290.1 |       287.1 |      **0.95×** |         282 → 282 |     347.0 → 347.0 |          1 → 1 |             4 → 4 |

## 2. Full 22/22 SF=5 Parity & Non-Regression Check (`after` vs. DuckDB 2.0.1)

Every query verified lossless Arrow IPC parity against DuckDB 2.0.1 prior to
timing (3 measured warm-cache repeats, 8 pinned cores):

| Query   |   Rows | `morsel_pushdown` ms (min–max) | `duckdb_default` ms (min–max) | Morsel / DuckDB | RGs Claimed |
| ------- | -----: | -----------------------------: | ----------------------------: | --------------: | ----------: |
| `q1`    |      4 |            581.5 (572.1–583.5) |           449.6 (449.5–451.6) |           1.29× |         240 |
| `q2`    |    100 |            141.5 (140.7–142.1) |           117.5 (113.7–118.8) |           1.20× |          78 |
| `q3`    |     10 |            313.8 (313.4–319.1) |           336.1 (333.2–338.3) |       **0.93×** |         310 |
| `q4`    |      5 |            159.7 (159.1–162.9) |           186.7 (184.6–189.4) |       **0.86×** |         304 |
| `q5`    |      5 |            401.6 (392.6–416.3) |           366.6 (364.5–371.8) |           1.10× |         313 |
| `q6`    |      1 |            332.5 (331.7–336.3) |           196.9 (196.4–197.8) |           1.69× |         240 |
| `q7`    |      4 |            357.0 (354.7–359.1) |           395.4 (393.2–400.0) |       **0.90×** |         313 |
| `q8`    |      2 |            431.9 (426.9–436.8) |           441.6 (435.7–443.4) |       **0.98×** |         322 |
| `q9`    |    175 |            813.3 (813.0–823.0) |           748.6 (742.1–757.2) |           1.09× |         346 |
| `q10`   |     20 |            468.5 (467.3–476.3) |           436.5 (432.0–436.8) |           1.07× |         311 |
| `q11`   |  4,606 |             61.9 (61.0–62.7)   |            90.0 (87.8–90.7)   |       **0.69×** |          34 |
| `q12`   |      2 |            195.7 (191.0–210.4) |           213.6 (211.2–214.5) |       **0.92×** |         304 |
| `q13`   |     42 |            345.5 (337.5–348.6) |           505.8 (499.1–526.8) |       **0.68×** |          70 |
| `q14`   |      1 |            259.2 (256.7–261.6) |           320.7 (318.6–325.8) |       **0.81×** |         248 |
| `q15`   |      1 |            343.3 (342.1–344.3) |           259.2 (257.9–269.0) |           1.32× |         241 |
| `q16`   | 27,724 |            117.0 (115.3–121.4) |           132.3 (131.0–133.3) |       **0.88×** |          41 |
| `q17`   |      1 |            378.7 (373.4–381.6) |           390.7 (388.7–393.8) |       **0.97×** |         488 |
| `q18`   |    100 |            560.0 (547.5–562.2) |           447.4 (443.2–464.5) |           1.25× |         547 |
| `q19`   |      1 |            346.6 (340.5–370.7) |           330.6 (319.6–332.0) |           1.05× |         248 |
| `q20`   |    901 |            273.9 (269.4–277.0) |           290.1 (287.0–296.4) |       **0.94×** |         282 |
| `q21`   |    100 |            704.0 (701.6–704.2) |           706.1 (705.0–713.1) |       **1.00×** |         786 |
| `q22`   |      7 |            110.0 (109.6–112.1) |           136.4 (135.7–137.8) |       **0.81×** |          76 |
| **GM**  | 22/22  |                  **295.2 ms**  |                    296.1 ms   |      **0.997×** |           — |
| **Sum** | 22/22  |                   7,697.1 ms   |                  7,498.4 ms   |         1.026×  |           — |

## Reproduction Commands

```bash
work="$HOME/.cache/issue228"
py="$HOME/.cache/issue193/venv/bin/python"
data_dir="$HOME/.cache/issue193/tpch_sf5"

# 1. Run 21/21 SF=0.01 parity/plan/spill harness unittests
MORSEL_TPCH_RUNNER="$work/morsel_tpch_runner_after_final" \
  MORSEL_BENCH_REVISION="$(git rev-parse HEAD)" \
  "$py" -B -m unittest discover -s benchmarks/morsel_tpch -v

# 2. Paired before/after SF=5 focus benchmark (q15, q11, q2, q17, q20)
taskset -c 0-7 "$py" -B benchmarks/morsel_tpch/harness.py bench \
  "$data_dir" --runner "$work/morsel_tpch_runner_base" \
  --revision 1663c620eede06cfde33ebb0fa1beeef0868b714 \
  --threads 8 --timeout 180 --cache warm --warmups 1 --repeats 5 \
  --queries q15,q11,q2,q17,q20 --morsel-arms morsel_pushdown,duckdb_default \
  > "$work/final_paired_before_5q.jsonl"

taskset -c 0-7 "$py" -B benchmarks/morsel_tpch/harness.py bench \
  "$data_dir" --runner "$work/morsel_tpch_runner_after_final" \
  --revision "$(git rev-parse HEAD)" \
  --threads 8 --timeout 180 --cache warm --warmups 1 --repeats 5 \
  --queries q15,q11,q2,q17,q20 --morsel-arms morsel_pushdown,duckdb_default \
  > "$work/final_paired_after_5q.jsonl"

# 3. Full 22/22 SF=5 parity & non-regression check
taskset -c 0-7 "$py" -B benchmarks/morsel_tpch/harness.py bench \
  "$data_dir" --runner "$work/morsel_tpch_runner_after_final" \
  --revision "$(git rev-parse HEAD)" \
  --threads 8 --timeout 180 --cache warm --warmups 1 --repeats 3 \
  --queries all --morsel-arms morsel_pushdown,duckdb_default \
  > "$work/final_22q_after.jsonl"
```
