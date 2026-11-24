<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# TPC-H SF5 All-22-Query Baseline: Stock DataFusion vs. DuckDB 2.0.1 + Morsel Admission Matrix (#189)

This baseline expands `benchmarks/morsel_tpch/` to **all 22 standard TPC-H
queries (`q1`–`q22`) at Scale Factor 5 (`--sf 5`)** on idle `rob-tandy-dduck2`.
It records:

1. A quiet-host **Stock DataFusion (`stock_default`) vs. DuckDB 2.0.1
   (`duckdb_default`)** warm-cache baseline across all 22 standard TPC-H
   queries (plus geometric mean and sum-of-medians totals).
2. A live **Morsel (`morsel_mode=required`)** 5-arm comparison at SF5 on all
   currently-admitted workloads (`q6`, `q1_unordered`, `q6_count`, `q6_scan`).
3. A **live 22-query Morsel admission & rejection matrix** showing both the
   verbatim planning rejection emitted by `morsel_mode=required` at HEAD and
   the underlying physical operator requirements per query (`Sort`/`TopN`,
   `Inner`/`Semi`/`Anti`/`Left` joins, `ScalarSubquery`, multi-stage
   aggregates, `COUNT(DISTINCT)`).

## Executive Summary

All 25 workloads (all 22 standard TPC-H queries `q1`–`q22` plus `q1_unordered`,
`q6_count`, `q6_scan`) passed exact lossless Arrow IPC result parity against
**DuckDB 2.0.1** at both `--sf 0.01` and `--sf 5`.

- **22-query standard TPC-H (`q1`–`q22`) geometric mean at SF5 (8 pinned cores,
  warm page cache):**
  - Stock DataFusion (`stock_default`): **459.2 ms**
  - DuckDB 2.0.1 (`duckdb_default`): **303.6 ms**
  - **Geometric mean ratio (`stock_default / duckdb_default`): `1.51×`**
    (sum of medians: `12,944.6 ms` vs. `7,676.9 ms`, **`1.69×`**).
- **Where Stock DataFusion already leads or matches DuckDB 2.0.1 at SF5:**
  `q22` (**0.86×**, `120.2` vs. `140.6 ms`), `q13` (**0.88×**, `465.7` vs.
  `527.2 ms`), `q16` (**0.97×**, `131.8` vs. `135.6 ms`), and `q12` (**1.00×**,
  `216.2` vs. `217.1 ms`).
- **Largest Stock vs. DuckDB gaps (`> 2.0×`):** `q18` (**3.00×**, `1,379.0` vs.
  `460.2 ms`), `q17` (**2.95×**, `1,178.6` vs. `400.0 ms`), `q15` (**2.68×**,
  `715.0` vs. `267.1 ms` — Stock scans `lineitem` twice for `revenue0`,
  `605.1 MB` `bytes_scanned`, while DuckDB materializes the CTE once), and `q7`
  (**2.14×**, `871.2` vs. `406.2 ms`).
- **Currently-admitted Morsel workloads at SF5 (`q6`, `q1_unordered`,
  `q6_count`, `q6_scan`):** `morsel_no_pushdown` outperforms unmodified
  `stock_default` by **`1.07–1.09×`** (`−16.6` to `−50.2 ms`) across all four
  queries and matches or edges out tuned `stock_no_pushdown` (`1.00–1.03×`),
  claiming all `240 / 240` `lineitem` row groups cleanly across 8 workers
  (`28–33` row groups per worker) with `0.0 ms` `morsel_output_wait`.
- **Live Morsel admission status across `q1`–`q22`:** **`1 / 22` standard
  queries admitted** (`q6`), plus 3 single-table Morsel slices (`q1_unordered`,
  `q6_count`, `q6_scan`). The remaining **`21 / 22`** standard queries are
  explicitly rejected under `morsel_mode=required`: 18 at top-level `Sort` /
  `TopN`, 2 at `Inner Join` (`q14`, `q19`), and 1 at `LeftSemi Join` (`q17`).

## Environment, Controls, and Reproduction Commands

### Host and Build Metadata

- **Host:** `rob-tandy-dduck2`, Ubuntu 24.04.5 LTS, Linux `7.0.0-1012-aws`,
  16 ARM Neoverse-N1 cores (1 thread/core, 64 KiB L1d / 1 MiB L2 per core,
  32 MiB shared L3), 61 GiB RAM.
- **CPU pinning and threads:** `taskset -c 0-7` (8 pinned CPUs), `--threads 8`
  (Tokio `worker_threads=8`, `max_blocking_threads=8`, `target_partitions=8`;
  DuckDB `SET threads = 8`), batch size `8,192`.
- **Host idle verification:** `vmstat -y 1 3` immediately before the 310-trial
  benchmark reported **87–93% host CPU idle** (`us 6–12%, sy 0–2%`);
  immediately after reported **99% host CPU idle** (`us 0–1%, sy 0%`).
- **Build & versions:**
  - DataFusion revision: `9d512c59c38ea5b9f21c083f0da8033dc7114c38`
    (`agent/morsel-issue-189`, on top of `morsel-execution` HEAD `c21d06f93`),
    built with `--locked --profile release-nonlto -p datafusion
--no-default-features --features
parquet,sql,datetime_expressions,unicode_expressions --example
morsel_tpch_runner` (`datetime_expressions` is required for `EXTRACT(YEAR ...)`
    in `q7`/`q8`/`q9`; `unicode_expressions` is required for `SUBSTRING` in
    `q22`).
  - Rust `1.99.1` (`58cf8f9c3`, `aarch64-unknown-linux-gnu`), Cargo `1.99.1`,
    Python `3.12.10`, DuckDB `2.0.1`, PyArrow `26.0.1`, NumPy `2.5.5`.
- **Protocol:** Fresh process per trial, 1 unmeasured warmup round + 5 measured
  counterbalanced rounds (alternating arm order between odd and even rounds).
  All `310 / 310` measured trials started and ended with **100% of 4 KiB
  Parquet file pages resident** via Linux `mincore`. Timings below are medians
  in milliseconds with observed `min–max` in parentheses.

### SF5 Parquet Dataset Layout (`--sf 5 --row-group-size 131072`)

Generated on local EC2 NVMe Btrfs storage
(`/dev/mapper/vg_ws_ephemeral-volume1p1` under `$HOME/.cache/issue189/tpch_sf5`)
with DuckDB `SET threads = 1; CALL dbgen(sf=5)`, primary-key ordering, Zstd
level 3, Parquet `2.6` / data page `1.0`, dictionary encoding, and column
min/max statistics:

| Table      | Ordering key               |  Files |           Rows | Row groups |     Parquet bytes |  Size (MiB) |
| ---------- | -------------------------- | -----: | -------------: | ---------: | ----------------: | ----------: |
| `lineitem` | `l_orderkey, l_linenumber` |     16 |     29,999,795 |        240 |       938,556,022 |       895.1 |
| `orders`   | `o_orderkey`               |      8 |      7,500,000 |         64 |       234,665,462 |       223.8 |
| `partsupp` | `ps_partkey, ps_suppkey`   |      4 |      4,000,000 |         32 |       155,651,113 |       148.4 |
| `customer` | `c_custkey`                |      2 |        750,000 |          6 |        44,713,200 |        42.6 |
| `part`     | `p_partkey`                |      2 |      1,000,000 |          8 |        23,931,359 |        22.8 |
| `supplier` | `s_suppkey`                |      1 |         50,000 |          1 |         3,048,711 |         2.9 |
| `nation`   | `n_nationkey`              |      1 |             25 |          1 |             2,663 |        0.00 |
| `region`   | `r_regionkey`              |      1 |              5 |          1 |             1,543 |        0.00 |
| **Total**  |                            | **35** | **43,299,825** |    **353** | **1,400,570,073** | **1,335.7** |

For `lineitem`, 16 files × 15 row groups (`14 × 131,072` + 1 tail group of
`~39,979` rows per file) produce `240` row groups (`30` per worker on 8
workers), `orders` produces `64` row groups (`8` per worker), and `partsupp`
produces `32` row groups (`4` per worker).

### Exact Reproduction Commands

```bash
set -euo pipefail
b=benchmarks/morsel_tpch
work="$HOME/.cache/issue189"
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

# 2. Create pinned Python venv and run SF=0.01 22-query parity + plan test suite.
timeout 120s uv venv --python 3.12.10 "$work/venv"
timeout 180s uv pip install --python "$work/venv/bin/python" -r "$b/requirements.txt"
py="$work/venv/bin/python"
MORSEL_TPCH_RUNNER="$runner" MORSEL_BENCH_REVISION="$revision" \
  timeout 180s "$py" -B -m unittest discover -s "$b" -v

# 3. Generate deterministic SF5 TPC-H Parquet tables outside Git (~1m45s).
timeout 600s "$py" -B "$b/harness.py" generate "$work/tpch_sf5" \
  --sf 5 --row-group-size 131072

# 4. Run quiet-host warm-cache SF5 22-query + admitted Morsel benchmark.
vmstat -y 1 3
timeout --kill-after=10s 1800s taskset -c 0-7 "$py" -B "$b/harness.py" bench \
  "$work/tpch_sf5" --runner "$runner" --revision "$revision" \
  --threads 8 --timeout 180 --cache warm --warmups 1 --repeats 5 \
  --queries all,q1_unordered,q6_count,q6_scan \
  --roadmap-arms stock_default,duckdb_default \
  --morsel-arms stock_default,stock_no_pushdown,morsel_no_pushdown,morsel_pushdown,duckdb_default \
  > "$work/sf5_baseline_22q.jsonl"
vmstat -y 1 3

# 5. Summarize results.
"$py" -B "$b/harness.py" summarize "$work/sf5_baseline_22q.jsonl"

# 6. Remove detached worktree when done.
git worktree remove --force "$work/pinned"
```

## 1. All 22 Standard TPC-H Queries at SF5 (Stock DataFusion vs. DuckDB 2.0.1)

Every query ran in fresh processes with `mincore` confirming `100%` 4 KiB page
residency before and after every trial. `Stock / DuckDB` is the ratio of median
total wall times (`< 1.00×` means Stock DataFusion is faster).

| Query             |   Rows | `mincore` pages (100%) | `bytes_scanned` | Stock `stock_default` total ms (min–max) | Stock exec ms | DuckDB `duckdb_default` total ms (min–max) | Stock / DuckDB | Morsel (`required`)   |
| ----------------- | -----: | ---------------------: | --------------: | ---------------------------------------: | ------------: | -----------------------------------------: | -------------: | --------------------- |
| `q1`              |      4 |                229,147 |     259,744,879 |                      767.2 (757.0–776.9) |         752.3 |                        463.8 (459.4–472.6) |          1.65× | Rejected (`Sort`)     |
| `q2`              |    100 |                 44,595 |      69,066,048 |                      178.1 (174.0–193.6) |         161.3 |                        119.9 (119.2–130.8) |          1.49× | Rejected (`Sort`)     |
| `q3`              |     10 |                297,360 |     334,406,980 |                      527.7 (510.6–547.4) |         510.2 |                        344.7 (340.2–371.0) |          1.53× | Rejected (`Sort`)     |
| `q4`              |      5 |                286,442 |     175,082,186 |                      201.3 (196.0–202.5) |         185.6 |                        187.6 (187.2–188.4) |          1.07× | Rejected (`Sort`)     |
| `q5`              |      5 |                298,107 |     374,043,457 |                      691.9 (678.3–704.0) |         672.9 |                        373.3 (370.4–383.4) |          1.85× | Rejected (`Sort`)     |
| `q6`              |      1 |                229,147 |     238,671,437 |                      363.0 (355.6–367.2) |         350.0 |                        201.1 (198.0–206.8) |          1.80× | **Admitted**          |
| `q7`              |      4 |                298,106 |     408,905,403 |                      871.2 (845.3–902.5) |         850.9 |                        406.2 (400.2–411.8) |          2.14× | Rejected (`Sort`)     |
| `q8`              |      2 |                303,951 |     531,613,209 |                      650.3 (650.3–663.0) |         627.1 |                        451.4 (449.0–453.6) |          1.44× | Rejected (`Sort`)     |
| `q9`              |    175 |                331,036 |     554,729,920 |                1,012.6 (1,001.2–1,059.2) |         992.4 |                        768.5 (764.4–783.3) |          1.32× | Rejected (`Sort`)     |
| `q10`             |     20 |                297,361 |     335,186,811 |                      637.3 (621.6–649.1) |         617.7 |                        443.8 (440.5–450.6) |          1.44× | Rejected (`Sort`)     |
| `q11`             |  4,606 |                 38,750 |      70,213,866 |                      108.9 (107.9–111.0) |          96.6 |                          98.0 (90.1–100.3) |          1.11× | Rejected (`Sort`)     |
| `q12`             |      2 |                286,442 |     221,292,133 |                      216.2 (212.8–220.4) |         199.5 |                        217.1 (215.9–222.6) |      **1.00×** | Rejected (`Sort`)     |
| `q13`             |     42 |                 68,213 |     163,187,253 |                      465.7 (462.4–508.7) |         456.8 |                        527.2 (516.7–535.3) |      **0.88×** | Rejected (`Sort`)     |
| `q14`             |      1 |                234,991 |     373,492,768 |                      404.5 (399.8–406.3) |         388.6 |                        325.9 (321.9–328.7) |          1.24× | Rejected (`Inner`)    |
| `q15`             |      1 |                229,892 |     605,140,002 |                      715.0 (708.6–722.3) |         697.0 |                        267.1 (259.1–297.4) |          2.68× | Rejected (`Sort`)     |
| `q16`             | 27,724 |                 44,593 |      22,553,011 |                      131.8 (131.1–138.5) |         120.8 |                        135.6 (134.4–137.3) |      **0.97×** | Rejected (`Sort`)     |
| `q17`             |      1 |                234,991 |     513,275,514 |                1,178.6 (1,161.8–1,184.5) |       1,162.1 |                        400.0 (398.4–404.5) |          2.95× | Rejected (`LeftSemi`) |
| `q18`             |    100 |                297,360 |     237,511,219 |                1,379.0 (1,351.0–1,385.3) |       1,362.1 |                        460.2 (450.6–465.7) |          3.00× | Rejected (`Sort`)     |
| `q19`             |      1 |                234,991 |     369,999,158 |                      553.0 (541.8–557.0) |         536.4 |                        330.2 (326.3–372.7) |          1.67× | Rejected (`Inner`)    |
| `q20`             |    901 |                273,741 |     347,065,289 |                      466.8 (450.7–494.4) |         448.4 |                        296.6 (295.8–300.6) |          1.57× | Rejected (`Sort`)     |
| `q21`             |    100 |                287,188 |     606,479,991 |                1,304.2 (1,297.4–1,315.5) |       1,283.6 |                        718.2 (713.3–721.5) |          1.82× | Rejected (`Sort`)     |
| `q22`             |      7 |                 68,213 |      59,097,054 |                      120.2 (115.4–121.2) |         109.4 |                        140.6 (137.5–142.5) |      **0.86×** | Rejected (`Sort`)     |
| **GeoMean (22q)** |      — |                      — |               — |                             **459.2 ms** |             — |                               **303.6 ms** |      **1.51×** | `1 / 22` admitted     |
| **Sum (22q)**     |      — |                      — |               — |                          **12,944.6 ms** |             — |                             **7,676.9 ms** |      **1.69×** | `1 / 22` admitted     |

## 2. Currently-Admitted Morsel Queries at SF5 (`q6`, `q1_unordered`, `q6_count`, `q6_scan`)

All four admitted queries run over SF5 `lineitem` (`29,999,795` rows, 16 files,
`240` row groups, `229,147 / 229,147` 4 KiB pages resident via `mincore`). Both
Morsel arms executed natively under `MorselExec: workers=8,
NativeParquet[lineitem...]`, claimed all `240 / 240` row groups, and scanned
the exact same `bytes_scanned` as Stock DataFusion (`238,671,437` on `q6`,
`259,744,879` on `q1_unordered`, `82,528,403` on `q6_count`, `285,708,571` on
`q6_scan`).

### 5-Arm Wall-Clock Comparison

| Query          |    Rows | Arm                  | Batches | Morsels claimed | Median total ms (min–max) | Median exec ms (min–max) | vs. `stock_default` | vs. `duckdb_default` |
| -------------- | ------: | -------------------- | ------: | --------------: | ------------------------: | -----------------------: | ------------------: | -------------------: |
| `q6`           |       1 | `stock_default`      |       1 |               0 |       363.0 (355.6–367.2) |      350.0 (343.2–354.6) |               1.00× |                1.80× |
| `q6`           |       1 | `stock_no_pushdown`  |       1 |               0 |       340.5 (337.2–349.4) |      329.2 (326.3–338.5) |               1.07× |                1.69× |
| `q6`           |       1 | `morsel_no_pushdown` |       1 |             240 |   **338.4** (335.2–347.3) |      328.1 (325.0–337.0) |           **1.07×** |                1.68× |
| `q6`           |       1 | `morsel_pushdown`    |       1 |             240 |       491.5 (487.4–522.6) |      478.8 (474.9–510.4) |               0.74× |                2.44× |
| `q6`           |       1 | `duckdb_default`     |       1 |               — |       201.1 (198.0–206.8) |                        — |               1.80× |                1.00× |
| `q1_unordered` |       4 | `stock_default`      |       3 |               0 |       761.9 (755.0–784.9) |      748.0 (740.9–770.7) |               1.00× |                1.63× |
| `q1_unordered` |       4 | `stock_no_pushdown`  |       3 |               0 |       733.1 (726.6–751.9) |      719.7 (714.1–739.3) |               1.04× |                1.57× |
| `q1_unordered` |       4 | `morsel_no_pushdown` |       1 |             240 |   **711.7** (706.0–715.6) |      699.8 (694.4–704.3) |           **1.07×** |                1.53× |
| `q1_unordered` |       4 | `morsel_pushdown`    |       1 |             240 |       719.6 (706.2–723.2) |      706.0 (693.7–710.2) |               1.06× |                1.54× |
| `q1_unordered` |       4 | `duckdb_default`     |       4 |               — |       466.3 (461.3–498.6) |                        — |               1.63× |                1.00× |
| `q6_count`     |       1 | `stock_default`      |       1 |               0 |       203.7 (198.9–205.4) |      191.5 (186.2–192.7) |               1.00× |                2.54× |
| `q6_count`     |       1 | `stock_no_pushdown`  |       1 |               0 |       188.2 (181.6–194.0) |      177.3 (171.5–183.1) |               1.08× |                2.34× |
| `q6_count`     |       1 | `morsel_no_pushdown` |       1 |             240 |   **187.1** (185.0–189.2) |      176.8 (174.4–178.9) |           **1.09×** |                2.33× |
| `q6_count`     |       1 | `morsel_pushdown`    |       1 |             240 |       250.8 (247.1–257.2) |      238.9 (235.4–245.8) |               0.81× |                3.12× |
| `q6_count`     |       1 | `duckdb_default`     |       1 |               — |          80.3 (79.0–89.7) |                        — |               2.54× |                1.00× |
| `q6_scan`      | 569,999 | `stock_default`      |      74 |               0 |       402.0 (397.4–424.4) |      389.5 (385.4–412.2) |               1.00× |                1.65× |
| `q6_scan`      | 569,999 | `stock_no_pushdown`  |      72 |               0 |       375.4 (368.5–378.5) |      364.6 (358.4–368.3) |               1.07× |                1.54× |
| `q6_scan`      | 569,999 | `morsel_no_pushdown` |   72–73 |             240 |   **376.6** (375.9–382.1) |      366.9 (366.2–372.3) |           **1.07×** |                1.55× |
| `q6_scan`      | 569,999 | `morsel_pushdown`    |   73–74 |             240 |       549.4 (535.0–554.2) |      538.0 (524.2–542.7) |               0.73× |                2.26× |
| `q6_scan`      | 569,999 | `duckdb_default`     |     240 |               — |       243.0 (242.4–245.0) |                        — |               1.65× |                1.00× |

### Morsel Wait/Work Breakdown at SF5 (`MORSEL_METRICS=1`, Summed Across 8 Workers)

| Query          | Arm                  | Decode work ms | Process work ms | Ready wait ms | Output wait ms | Pending IO ms | Peak ready/out | Example worker claims (`240` total) |
| -------------- | -------------------- | -------------: | --------------: | ------------: | -------------: | ------------: | -------------: | ----------------------------------- |
| `q6`           | `morsel_no_pushdown` |        2,260.9 |           211.0 |          14.8 |            0.0 |         240.9 |          `8/1` | `[29, 30, 31, 31, 31, 29, 29, 30]`  |
| `q6`           | `morsel_pushdown`    |        3,208.8 |             4.4 |         120.4 |            0.0 |         347.9 |          `5/1` | `[32, 32, 36, 27, 26, 25, 33, 29]`  |
| `q1_unordered` | `morsel_no_pushdown` |        3,175.4 |         2,236.0 |          16.6 |            0.0 |         266.4 |          `8/1` | `[30, 30, 30, 30, 31, 30, 30, 29]`  |
| `q1_unordered` | `morsel_pushdown`    |        3,353.8 |         1,915.3 |          39.2 |            0.0 |         302.6 |          `8/1` | `[32, 15, 47, 34, 25, 18, 32, 37]`  |
| `q6_count`     | `morsel_no_pushdown` |        1,145.2 |           168.6 |          10.8 |            0.0 |         126.4 |          `8/1` | `[29, 30, 30, 30, 30, 29, 33, 29]`  |
| `q6_count`     | `morsel_pushdown`    |        1,559.7 |             0.1 |          80.5 |            0.0 |         213.0 |          `6/1` | `[26, 34, 25, 29, 28, 31, 32, 35]`  |
| `q6_scan`      | `morsel_no_pushdown` |        2,517.2 |           229.9 |          16.8 |            0.0 |         290.8 |          `8/6` | `[29, 33, 29, 29, 31, 31, 28, 30]`  |
| `q6_scan`      | `morsel_pushdown`    |        3,566.6 |             7.4 |         134.0 |            0.0 |         366.3 |          `5/4` | `[33, 30, 29, 30, 21, 40, 30, 27]`  |

Notice how cleanly SF5 scales compared to SF1 (`checkpoint_a_results.md`):
under `morsel_no_pushdown`, aggregate `morsel_ready_wait` across all 8 workers
stays at `10.8–16.8 ms` (`1.4–2.1 ms` per worker) over 240 row groups, and
`morsel_output_wait` remains `0.0 ms`. On `q6`/`q6_count`/`q6_scan`, Parquet
`RowFilter` pushdown (`morsel_pushdown`) still adds `~40%` extra decode work
for the low-selectivity `Decimal128` predicates (as noted in `#173`), while on
`q1_unordered` (98.6% selectivity) `morsel_no_pushdown` (`711.7 ms`) edges out
`morsel_pushdown` (`719.6 ms`).

## 3. Live 22-Query Morsel Admission & Operator-Capability Checklist

Every standard TPC-H query (`q1`–`q22`) was probed with
`tpch_runner plan required no_pushdown` at revision `9d512c59c`. Because
`MorselQueryPlanner` walks logical plans top-down and stops at the first
unsupported operator, the table below lists **both**:

1. The **verbatim planning rejection** emitted under `morsel_mode=required`, and
2. The **full set of physical plan operators** observed in Stock DataFusion's
   SF5 plan (`Sort`/`TopN`, `HashJoin` join types, `ScalarSubquery`,
   multi-stage `AggregateExec`, `COUNT(DISTINCT)`), providing a live checklist
   for the upcoming `Sort`/`TopN`, `Inner Join`, `Semi`/`Anti`/`Outer Join`,
   and Multi-Stage Composition issues.

| Query | Status at HEAD | Underlying Stock Physical Plan Capabilities Needed                           | Verbatim `morsel_mode=required` Planning Rejection                                                                                                                                                                                                   |
| ----- | -------------- | ---------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `q1`  | Rejected       | `Sort`, `Agg` (1 stage)                                                      | `Error during planning: morsel_mode=required: unsupported logical operator: Sort: lineitem.l_returnflag ASC NULLS LAST, lineitem.l_linestatus ASC NULLS LAST`                                                                                        |
| `q2`  | Rejected       | `TopN(100)`, `HashJoin(Inner×6, LeftSemi, RightSemi)`, `Agg`                 | `Error during planning: morsel_mode=required: unsupported logical operator: Sort: supplier.s_acctbal DESC NULLS FIRST, nation.n_name ASC NULLS LAST, supplier.s_name ASC NULLS LAST, part.p_partkey ASC NULLS LAST, fetch=100`                       |
| `q3`  | Rejected       | `TopN(10)`, `HashJoin(Inner×2)`, `Agg`                                       | `Error during planning: morsel_mode=required: unsupported logical operator: Sort: revenue DESC NULLS FIRST, orders.o_orderdate ASC NULLS LAST, fetch=10`                                                                                             |
| `q4`  | Rejected       | `Sort`, `HashJoin(LeftSemi)`, `Agg`                                          | `Error during planning: morsel_mode=required: unsupported logical operator: Sort: orders.o_orderpriority ASC NULLS LAST`                                                                                                                             |
| `q5`  | Rejected       | `Sort`, `HashJoin(Inner×5)`, `Agg`                                           | `Error during planning: morsel_mode=required: unsupported logical operator: Sort: revenue DESC NULLS FIRST`                                                                                                                                          |
| `q6`  | **Admitted**   | `Agg` (ungrouped `SUM`) — **natively admitted**                              | _(admitted: `MorselExec: workers=8, NativeParquet[lineitem...] -> Filter -> AggregateSink`)_                                                                                                                                                         |
| `q7`  | Rejected       | `Sort`, `HashJoin(Inner×5)`, `Agg`                                           | `Error during planning: morsel_mode=required: unsupported logical operator: Sort: shipping.supp_nation ASC NULLS LAST, shipping.cust_nation ASC NULLS LAST, shipping.l_year ASC NULLS LAST`                                                          |
| `q8`  | Rejected       | `Sort`, `HashJoin(Inner×7)`, `Agg`                                           | `Error during planning: morsel_mode=required: unsupported logical operator: Sort: all_nations.o_year ASC NULLS LAST`                                                                                                                                 |
| `q9`  | Rejected       | `Sort`, `HashJoin(Inner×5)`, `Agg`                                           | `Error during planning: morsel_mode=required: unsupported logical operator: Sort: profit.nation ASC NULLS LAST, profit.o_year DESC NULLS FIRST`                                                                                                      |
| `q10` | Rejected       | `TopN(20)`, `HashJoin(Inner×3)`, `Agg`                                       | `Error during planning: morsel_mode=required: unsupported logical operator: Sort: revenue DESC NULLS FIRST, fetch=20`                                                                                                                                |
| `q11` | Rejected       | `Sort`, `HashJoin(Inner×4)`, `MultiStageAgg(×2)`, `ScalarSubquery`           | `Error during planning: morsel_mode=required: unsupported logical operator: Sort: value DESC NULLS FIRST`                                                                                                                                            |
| `q12` | Rejected       | `Sort`, `HashJoin(Inner)`, `Agg`                                             | `Error during planning: morsel_mode=required: unsupported logical operator: Sort: lineitem.l_shipmode ASC NULLS LAST`                                                                                                                                |
| `q13` | Rejected       | `Sort`, `HashJoin(Left)`, `MultiStageAgg(×2)`                                | `Error during planning: morsel_mode=required: unsupported logical operator: Sort: custdist DESC NULLS FIRST, c_orders.c_count DESC NULLS FIRST`                                                                                                      |
| `q14` | Rejected       | `HashJoin(Inner)`, `Agg`                                                     | `Error during planning: morsel_mode=required: unsupported logical operator: Inner Join: lineitem.l_partkey = part.p_partkey`                                                                                                                         |
| `q15` | Rejected       | `Sort`, `HashJoin(Inner)`, `MultiStageAgg(×3)`, `ScalarSubquery`             | `Error during planning: morsel_mode=required: unsupported logical operator: Sort: supplier.s_suppkey ASC NULLS LAST`                                                                                                                                 |
| `q16` | Rejected       | `Sort`, `HashJoin(Inner, RightAnti)`, `MultiStageAgg(×2)`, `COUNT(DISTINCT)` | `Error during planning: morsel_mode=required: unsupported logical operator: Sort: supplier_cnt DESC NULLS FIRST, part.p_brand ASC NULLS LAST, part.p_type ASC NULLS LAST, part.p_size ASC NULLS LAST`                                                |
| `q17` | Rejected       | `HashJoin(Inner, LeftSemi + filter)`, `MultiStageAgg(×2)`                    | `Error during planning: morsel_mode=required: unsupported logical operator: LeftSemi Join: part.p_partkey = __scalar_sq_1.l_partkey Filter: CAST(lineitem.l_quantity AS Decimal128(30, 15)) < __scalar_sq_1.Float64(0.2) * avg(lineitem.l_quantity)` |
| `q18` | Rejected       | `TopN(100)`, `HashJoin(Inner×2, RightSemi)`, `MultiStageAgg(×2)`             | `Error during planning: morsel_mode=required: unsupported logical operator: Sort: orders.o_totalprice DESC NULLS FIRST, orders.o_orderdate ASC NULLS LAST, fetch=100`                                                                                |
| `q19` | Rejected       | `HashJoin(Inner + disjunctive filter)`, `Agg`                                | `Error during planning: morsel_mode=required: unsupported logical operator: Inner Join: lineitem.l_partkey = part.p_partkey Filter: ...` _(3-branch disjunctive `OR` filter on `part`/`lineitem`)_                                                   |
| `q20` | Rejected       | `Sort`, `HashJoin(Inner, LeftSemi×2, RightSemi)`, `Agg`                      | `Error during planning: morsel_mode=required: unsupported logical operator: Sort: supplier.s_name ASC NULLS LAST`                                                                                                                                    |
| `q21` | Rejected       | `TopN(100)`, `HashJoin(Inner×3, LeftSemi, RightAnti)`, `Agg`                 | `Error during planning: morsel_mode=required: unsupported logical operator: Sort: numwait DESC NULLS FIRST, supplier.s_name ASC NULLS LAST, fetch=100`                                                                                               |
| `q22` | Rejected       | `Sort`, `HashJoin(LeftAnti)`, `MultiStageAgg(×2)`, `ScalarSubquery`          | `Error during planning: morsel_mode=required: unsupported logical operator: Sort: custsale.cntrycode ASC NULLS LAST`                                                                                                                                 |

### Capability Unlock Roadmap Summary

- **Already admitted (`1 / 22`):** `q6` (plus slices `q1_unordered`,
  `q6_count`, `q6_scan`).
- **`Sort` alone unlocks (`+1` → `2 / 22`):** `q1`.
- **`Inner Join` + `Sort`/`TopN` + multi-join pipelines unlock (`+9` →
  `11 / 22`):** `q14`, `q19` (with join filter), `q12`, `q3`, `q10`, `q5`,
  `q7`, `q8`, `q9` (note `q14`/`q19` need no `Sort` at all — `Inner Join` alone
  unlocks `q14`, and `Inner Join` with residual filter unlocks `q19`).
- **Semi / Anti / Left Outer Joins (+ multi-stage agg / `COUNT(DISTINCT)`)
  unlock (`+8` → `19 / 22`):** `q4` (`LeftSemi`), `q13` (`Left` + 2-stage agg),
  `q16` (`RightAnti` + `COUNT(DISTINCT)`), `q17` (`LeftSemi` with filter +
  2-stage agg), `q18` (`RightSemi` + 2-stage agg + `TopN`), `q20`
  (`LeftSemi`/`RightSemi`), `q21` (`LeftSemi`/`RightAnti` + `TopN`), `q2`
  (`LeftSemi`/`RightSemi` + `TopN`).
- **`ScalarSubquery` + multi-stage composition unlock (`+3` → `22 / 22`):**
  `q11`, `q15`, `q22` (also needs `LeftAnti`).

## 4. Cross-Engine Parity and Query Handling Notes

All 22 queries are loaded directly from `benchmarks/queries/q1.sql`..`q22.sql`
and verified via lossless Arrow IPC streams against DuckDB 2.0.1:

- **Q15 (`CREATE VIEW revenue0` / `DROP VIEW revenue0`):** Rewritten at load
  time into a single-statement `WITH revenue0 (supplier_no, total_revenue) AS
(...) SELECT ...` CTE executed identically in fresh sessions on both
  DataFusion and DuckDB, eliminating any cross-trial view state leakage and
  allowing `morsel_mode=required` planning to inspect the `SELECT` plan
  directly rather than failing on DDL.
- **Q11 (`__TPCH_Q11_FRACTION__`):** Substitutes `(0.0001 / <sf>)` using the
  scale factor inferred from `part` cardinality (`rows / 200,000`), matching
  TPC-H clause 2.4.11.3 and `benchmarks/src/tpch/mod.rs` (`4,606` rows at SF5;
  `1` row at SF0.01).
- **Documented cross-engine type/name normalizations in `compare_tables`:**
  - `q1` / `q1_unordered` `AVG(DECIMAL(15,2))` and `q8` `mkt_share`: DuckDB
    returns `float64` while DataFusion returns `DECIMAL(19,6)` /
    `DECIMAL(38,8)`; checked within the Decimal's scale (`abs_tol=10^-scale`,
    `rel_tol=1e-9`).
  - `q14` / `q17` floating-point ratios: `math.isclose(..., rel_tol=1e-9,
abs_tol=1e-9)`.
  - `q12` conditional `SUM`: DuckDB returns `DECIMAL(38,0)` while DataFusion
    returns `INT64`; checked by exact integer/Decimal equality.
  - `q7` / `q8` / `q9` `EXTRACT(YEAR FROM ...)`: DuckDB returns `INT64` while
    DataFusion returns `INT32`; checked by exact integer equality.
  - `q18` unaliased `sum(l_quantity)`: DuckDB names the output column
    `sum(l_quantity)` while DataFusion qualifies it as
    `sum(lineitem.l_quantity)`; normalized by stripping the table qualifier
    inside unaliased aggregate column names only.
