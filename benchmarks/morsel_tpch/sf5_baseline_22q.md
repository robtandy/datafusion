<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# TPC-H SF5 All-22-Query Baseline: Stock DataFusion vs. DuckDB 2.0.1 + Morsel Admission Matrix (#189)

This baseline expands `benchmarks/morsel_tpch/` to **all 22 standard TPC-H
queries (`q1`–`q22`) at Scale Factor 5 (`--sf 5`)** on idle `rob-tandy-dduck2`
after `#188` (native two-pipeline Inner `HashJoin`). It records:

1. A quiet-host **Stock DataFusion (`stock_default`) vs. DuckDB 2.0.1
   (`duckdb_default`)** warm-cache baseline across all 22 standard TPC-H
   queries (plus geometric mean and sum-of-medians totals).
2. A live **Morsel (`morsel_mode=required`)** 5-arm comparison at SF5 on all
   six currently-admitted workloads (`q6`, `q14`, `q19`, `q1_unordered`,
   `q6_count`, `q6_scan`).
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
  - Stock DataFusion (`stock_default`): **458.9 ms**
  - DuckDB 2.0.1 (`duckdb_default`): **302.4 ms**
  - **Geometric mean ratio (`stock_default / duckdb_default`): `1.52×`**
    (sum of medians: `12,921.6 ms` vs. `7,654.2 ms`, **`1.69×`**).
- **Where Stock DataFusion already leads or matches DuckDB 2.0.1 at SF5:**
  `q22` (**0.86×**, `121.7` vs. `142.2 ms`), `q13` (**0.91×**, `467.9` vs.
  `513.2 ms`), `q16` (**0.95×**, `129.2` vs. `135.8 ms`), and `q12` (**1.00×**,
  `214.6` vs. `215.3 ms`).
- **Largest Stock vs. DuckDB gaps (`> 2.0×`):** `q18` (**3.00×**, `1,365.5` vs.
  `455.6 ms`), `q17` (**2.94×**, `1,168.6` vs. `398.0 ms`), `q15` (**2.67×**,
  `707.9` vs. `265.4 ms` — Stock scans `lineitem` twice for `revenue0`,
  `605.1 MB` `bytes_scanned`, while DuckDB materializes the CTE once), and `q7`
  (**2.10×**, `847.8` vs. `403.1 ms`).
- **Currently-admitted Morsel workloads at SF5 (`q6`, `q14`, `q19`,
  `q1_unordered`, `q6_count`, `q6_scan`):** The best Morsel arm outperforms
  unmodified `stock_default` by **`1.05–1.13×`** (`−13.0` to `−59.9 ms`) across
  all six workloads:
  - Newly admitted native `HashJoin` query **`q14`** (`morsel_pushdown`,
    **`354.7 ms`**) beats `stock_default` (`402.3 ms`) by **`1.13×`
    (`−47.6 ms`)** and `stock_no_pushdown` (`393.4 ms`) by **`1.11×`**, coming
    within **`1.09×`** of DuckDB 2.0.1 (`326.7 ms`) over the 30M × 1M
    `lineitem ⋈ part` join.
  - Newly admitted native `HashJoin` query **`q19`** (`morsel_no_pushdown`,
    **`521.8 ms`**) beats `stock_default` (`554.3 ms`) by **`1.06×`
    (`−32.5 ms`)** and `stock_no_pushdown` (`545.9 ms`) by **`1.05×`**.
  - Single-table `q6` (`339.3` vs. `364.2 ms`), `q1_unordered` (`720.1` vs.
    `780.0 ms`), `q6_count` (`187.1` vs. `200.1 ms`), and `q6_scan` (`381.2`
    vs. `400.7 ms`) beat `stock_default` by **`1.05–1.08×`**, claiming all
    `240 / 240` `lineitem` row groups (`248 / 248` with `part` on `q14`/`q19`)
    with `0.0 ms` `morsel_output_wait`.
- **Live Morsel admission status across `q1`–`q22`:** **`3 / 22` standard
  queries admitted** (`q6`, `q14`, `q19`), plus 3 single-table Morsel slices
  (`q1_unordered`, `q6_count`, `q6_scan`). The remaining **`19 / 22`** standard
  queries are explicitly rejected under `morsel_mode=required`: 18 at top-level
  `Sort` / `TopN`, and 1 (`q17`) at `unsupported join type LeftSemi in native
morsel hash join (only INNER JOIN is supported)`.

## Environment, Controls, and Reproduction Commands

### Host and Build Metadata

- **Host:** `rob-tandy-dduck2`, Ubuntu 24.04.5 LTS, Linux `7.0.0-1012-aws`,
  16 ARM Neoverse-N1 cores (1 thread/core, 64 KiB L1d / 1 MiB L2 per core,
  32 MiB shared L3), 61 GiB RAM.
- **CPU pinning and threads:** `taskset -c 0-7` (8 pinned CPUs), `--threads 8`
  (Tokio `worker_threads=8`, `max_blocking_threads=8`, `target_partitions=8`;
  DuckDB `SET threads = 8`), batch size `8,192`.
- **Host idle verification:** `vmstat -y 1 3` immediately after the 340-trial
  benchmark reported **96–99% host CPU idle** (`us 0–3%, sy 0–1%`; the
  pre-command `vmstat` caught the tail of the prior `cargo clippy` settling
  `13 → 39 → 79%` idle before the ~1m50s unmeasured verification phase began).
- **Build & versions:**
  - DataFusion revision: `0ff9edecf4c604e463bfd99a7f5f065afad4ab9c`
    (`agent/morsel-issue-189`, merged with `morsel-execution` HEAD `f044eda78`
    after `#188`), built with `--locked --profile release-nonlto -p datafusion
--no-default-features --features
parquet,sql,datetime_expressions,unicode_expressions --example
morsel_tpch_runner` (`datetime_expressions` is required for `EXTRACT(YEAR ...)`
    in `q7`/`q8`/`q9`; `unicode_expressions` is required for `SUBSTRING` in
    `q22`).
  - Rust `1.99.1` (`58cf8f9c3`, `aarch64-unknown-linux-gnu`), Cargo `1.99.1`,
    Python `3.12.10`, DuckDB `2.0.1`, PyArrow `26.0.1`, NumPy `2.5.5`.
- **Protocol:** Fresh process per trial, 1 unmeasured warmup round + 5 measured
  counterbalanced rounds (alternating arm order between odd and even rounds).
  All `340 / 340` measured trials started and ended with **100% of 4 KiB
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
workers), `orders` produces `64` row groups (`8` per worker), `partsupp`
produces `32` row groups (`4` per worker), and `part` produces `8` row groups
(`1` per worker).

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
| `q1`              |      4 |                229,147 |     259,744,879 |                      767.9 (763.6–787.8) |         753.3 |                        464.5 (459.6–538.6) |          1.65× | Rejected (`Sort`)     |
| `q2`              |    100 |                 44,595 |      69,066,048 |                      181.7 (179.1–184.6) |         164.6 |                        123.0 (120.4–125.9) |          1.48× | Rejected (`Sort`)     |
| `q3`              |     10 |                297,360 |     334,406,980 |                      513.8 (499.0–521.1) |         496.5 |                        344.6 (340.7–348.6) |          1.49× | Rejected (`Sort`)     |
| `q4`              |      5 |                286,442 |     175,082,186 |                      203.2 (193.8–205.4) |         187.6 |                        185.3 (183.7–186.4) |          1.10× | Rejected (`Sort`)     |
| `q5`              |      5 |                298,107 |     374,043,457 |                      695.4 (688.9–770.8) |         676.2 |                        374.8 (370.4–379.4) |          1.86× | Rejected (`Sort`)     |
| `q6`              |      1 |                229,147 |     238,671,437 |                      364.2 (358.2–373.5) |         350.8 |                        203.0 (201.9–208.5) |          1.79× | **Admitted**          |
| `q7`              |      4 |                298,106 |     408,905,403 |                      847.8 (845.6–853.4) |         826.9 |                        403.1 (401.7–414.1) |          2.10× | Rejected (`Sort`)     |
| `q8`              |      2 |                303,951 |     531,613,209 |                      654.0 (647.4–666.8) |         630.7 |                        449.3 (445.7–455.2) |          1.46× | Rejected (`Sort`)     |
| `q9`              |    175 |                331,036 |     554,729,920 |                  1,005.0 (988.2–1,048.3) |         983.8 |                        775.6 (770.2–787.9) |          1.30× | Rejected (`Sort`)     |
| `q10`             |     20 |                297,361 |     335,186,811 |                      647.9 (647.0–675.0) |         628.6 |                        447.9 (444.7–448.4) |          1.45× | Rejected (`Sort`)     |
| `q11`             |  4,606 |                 38,750 |      70,213,866 |                      110.0 (106.4–113.5) |          97.6 |                           92.4 (88.3–96.0) |          1.19× | Rejected (`Sort`)     |
| `q12`             |      2 |                286,442 |     221,292,133 |                      214.6 (207.1–218.8) |         198.6 |                        215.3 (213.9–225.1) |      **1.00×** | Rejected (`Sort`)     |
| `q13`             |     42 |                 68,213 |     163,187,253 |                      467.9 (461.4–471.2) |         458.9 |                        513.2 (507.1–515.0) |      **0.91×** | Rejected (`Sort`)     |
| `q14`             |      1 |                234,991 |     373,492,768 |                      402.3 (399.2–405.6) |         386.4 |                        326.7 (320.5–330.1) |          1.23× | **Admitted** (`Join`) |
| `q15`             |      1 |                229,892 |     605,140,002 |                      707.9 (697.1–749.0) |         689.6 |                        265.4 (259.9–277.7) |          2.67× | Rejected (`Sort`)     |
| `q16`             | 27,724 |                 44,593 |      22,553,011 |                      129.2 (127.8–161.7) |         118.5 |                        135.8 (135.1–136.8) |      **0.95×** | Rejected (`Sort`)     |
| `q17`             |      1 |                234,991 |     513,275,514 |                1,168.6 (1,156.8–1,178.4) |       1,152.4 |                        398.0 (394.5–403.2) |          2.94× | Rejected (`LeftSemi`) |
| `q18`             |    100 |                297,360 |     237,511,219 |                1,365.5 (1,350.8–1,375.9) |       1,348.1 |                        455.6 (450.3–469.1) |          3.00× | Rejected (`Sort`)     |
| `q19`             |      1 |                234,991 |     369,999,158 |                      554.3 (545.1–587.1) |         536.6 |                        327.4 (325.6–332.9) |          1.69× | **Admitted** (`Join`) |
| `q20`             |    901 |                273,741 |     347,065,289 |                      466.1 (450.9–496.6) |         447.0 |                        294.1 (290.5–298.4) |          1.59× | Rejected (`Sort`)     |
| `q21`             |    100 |                287,188 |     606,479,991 |                1,332.3 (1,313.5–1,356.2) |       1,311.8 |                        717.2 (713.1–724.0) |          1.86× | Rejected (`Sort`)     |
| `q22`             |      7 |                 68,213 |      59,097,054 |                      121.7 (119.3–123.9) |         110.4 |                        142.2 (137.7–159.1) |      **0.86×** | Rejected (`Sort`)     |
| **GeoMean (22q)** |      — |                      — |               — |                             **458.9 ms** |             — |                               **302.4 ms** |      **1.52×** | `3 / 22` admitted     |
| **Sum (22q)**     |      — |                      — |               — |                          **12,921.6 ms** |             — |                             **7,654.2 ms** |      **1.69×** | `3 / 22` admitted     |

## 2. Currently-Admitted Morsel Queries at SF5 (`q6`, `q14`, `q19`, `q1_unordered`, `q6_count`, `q6_scan`)

All six admitted workloads executed natively under `MorselExec: workers=8`,
claimed 100% of scanned row groups (`240 / 240` for `lineitem`-only queries;
`248 / 248` = `240 lineitem + 8 part` for two-pipeline `HashJoin` queries `q14`
and `q19`), passed exact `mincore` warm-cache verification (`229,147 / 229,147`
pages on `lineitem`-only; `234,991 / 234,991` pages on `lineitem + part`), and
scanned the exact same `bytes_scanned` as Stock DataFusion (`238,671,437` on
`q6`, `373,492,768` on `q14`, `369,999,158` on `q19`, `259,744,879` on
`q1_unordered`, `82,528,403` on `q6_count`, `285,708,571` on `q6_scan`).

### 5-Arm Wall-Clock Comparison

| Query          |    Rows | Arm                  | Batches | Morsels claimed | Median total ms (min–max) | Median exec ms (min–max) | vs. `stock_default` | vs. `duckdb_default` |
| -------------- | ------: | -------------------- | ------: | --------------: | ------------------------: | -----------------------: | ------------------: | -------------------: |
| `q6`           |       1 | `stock_default`      |       1 |               0 |       364.2 (358.2–373.5) |      350.8 (345.3–360.9) |               1.00× |                1.79× |
| `q6`           |       1 | `stock_no_pushdown`  |       1 |               0 |       339.5 (336.8–352.7) |      328.2 (326.4–341.2) |               1.07× |                1.67× |
| `q6`           |       1 | `morsel_no_pushdown` |       1 |             240 |   **339.3** (337.0–343.5) |      328.8 (326.7–333.3) |           **1.07×** |                1.67× |
| `q6`           |       1 | `morsel_pushdown`    |       1 |             240 |       495.6 (483.7–519.9) |      483.6 (472.1–507.8) |               0.73× |                2.44× |
| `q6`           |       1 | `duckdb_default`     |       1 |               — |       203.0 (201.9–208.5) |                        — |               1.79× |                1.00× |
| `q14`          |       1 | `stock_default`      |       1 |               0 |       402.3 (399.2–405.6) |      386.4 (383.5–390.4) |               1.00× |                1.23× |
| `q14`          |       1 | `stock_no_pushdown`  |       1 |               0 |       393.4 (387.1–401.1) |      379.2 (372.9–387.4) |               1.02× |                1.20× |
| `q14`          |       1 | `morsel_no_pushdown` |       1 |             248 |       455.7 (453.1–457.9) |      443.5 (441.2–445.0) |               0.88× |                1.39× |
| `q14`          |       1 | `morsel_pushdown`    |       1 |             248 |   **354.7** (348.7–356.6) |      340.8 (334.6–342.8) |           **1.13×** |            **1.09×** |
| `q14`          |       1 | `duckdb_default`     |       1 |               — |       326.7 (320.5–330.1) |                        — |               1.23× |                1.00× |
| `q19`          |       1 | `stock_default`      |       1 |               0 |       554.3 (545.1–587.1) |      536.6 (528.3–570.1) |               1.00× |                1.69× |
| `q19`          |       1 | `stock_no_pushdown`  |       1 |               0 |       545.9 (533.3–595.6) |      531.4 (519.2–580.5) |               1.02× |                1.67× |
| `q19`          |       1 | `morsel_no_pushdown` |       1 |             248 |   **521.8** (515.7–564.5) |      508.6 (502.2–551.0) |           **1.06×** |                1.59× |
| `q19`          |       1 | `morsel_pushdown`    |       1 |             248 |       686.3 (668.0–689.7) |      670.6 (653.0–674.4) |               0.81× |                2.10× |
| `q19`          |       1 | `duckdb_default`     |       1 |               — |       327.4 (325.6–332.9) |                        — |               1.69× |                1.00× |
| `q1_unordered` |       4 | `stock_default`      |       3 |               0 |       780.0 (775.4–801.6) |      765.0 (761.3–787.5) |               1.00× |                1.66× |
| `q1_unordered` |       4 | `stock_no_pushdown`  |       3 |               0 |       751.0 (738.4–764.9) |      738.1 (725.5–751.7) |               1.04× |                1.60× |
| `q1_unordered` |       4 | `morsel_no_pushdown` |       1 |             240 |       720.7 (706.8–731.7) |      709.1 (695.1–719.9) |               1.08× |                1.54× |
| `q1_unordered` |       4 | `morsel_pushdown`    |       1 |             240 |   **720.1** (713.8–749.8) |      707.2 (701.0–736.5) |           **1.08×** |                1.54× |
| `q1_unordered` |       4 | `duckdb_default`     |       4 |               — |       468.7 (463.9–471.3) |                        — |               1.66× |                1.00× |
| `q6_count`     |       1 | `stock_default`      |       1 |               0 |       200.1 (195.4–215.0) |      188.0 (182.9–202.4) |               1.00× |                2.48× |
| `q6_count`     |       1 | `stock_no_pushdown`  |       1 |               0 |       187.8 (186.1–196.0) |      177.1 (175.6–184.6) |               1.07× |                2.33× |
| `q6_count`     |       1 | `morsel_no_pushdown` |       1 |             240 |   **187.1** (183.8–189.0) |      176.8 (173.7–179.4) |           **1.07×** |                2.32× |
| `q6_count`     |       1 | `morsel_pushdown`    |       1 |             240 |       247.8 (245.3–274.3) |      236.3 (233.7–262.5) |               0.81× |                3.07× |
| `q6_count`     |       1 | `duckdb_default`     |       1 |               — |          80.7 (78.3–84.9) |                        — |               2.48× |                1.00× |
| `q6_scan`      | 569,999 | `stock_default`      |   73–74 |               0 |       400.7 (394.7–405.1) |      388.7 (382.6–392.6) |               1.00× |                1.62× |
| `q6_scan`      | 569,999 | `stock_no_pushdown`  |      72 |               0 |       377.2 (371.5–392.3) |      367.0 (360.7–382.0) |               1.06× |                1.53× |
| `q6_scan`      | 569,999 | `morsel_no_pushdown` |   72–74 |             240 |   **381.2** (375.3–383.0) |      371.7 (365.7–373.1) |           **1.05×** |                1.54× |
| `q6_scan`      | 569,999 | `morsel_pushdown`    |   72–75 |             240 |       546.4 (543.0–565.5) |      534.6 (531.6–553.4) |               0.73× |                2.21× |
| `q6_scan`      | 569,999 | `duckdb_default`     |     240 |               — |       247.2 (241.9–265.6) |                        — |               1.62× |                1.00× |

### Morsel Wait/Work Breakdown at SF5 (`MORSEL_METRICS=1`, Summed Across 8 Workers & Pipelines)

| Query          | Arm                  | Decode work ms | Process work ms | Ready wait ms | Output wait ms | Pending IO ms | Peak ready/out | Example worker claims (`240` or `248` total) |
| -------------- | -------------------- | -------------: | --------------: | ------------: | -------------: | ------------: | -------------: | -------------------------------------------- |
| `q6`           | `morsel_no_pushdown` |        2,246.9 |           211.1 |          14.0 |            0.0 |         245.5 |          `8/1` | `[28, 32, 31, 30, 30, 30, 29, 30]`           |
| `q6`           | `morsel_pushdown`    |        3,240.6 |             4.7 |          99.2 |            0.0 |         315.9 |          `5/1` | `[31, 32, 41, 26, 30, 32, 23, 25]`           |
| `q14`          | `morsel_no_pushdown` |        2,460.9 |           253.3 |          32.0 |            0.0 |         283.7 |          `8/1` | `[31, 31, 32, 30, 31, 31, 31, 31]`           |
| `q14`          | `morsel_pushdown`    |        1,492.5 |            78.3 |         102.6 |            0.0 |         316.2 |          `6/1` | `[32, 31, 35, 36, 30, 29, 28, 27]`           |
| `q19`          | `morsel_no_pushdown` |        3,074.2 |           772.1 |          38.3 |            0.0 |         279.7 |          `8/1` | `[30, 30, 30, 30, 32, 31, 31, 34]`           |
| `q19`          | `morsel_pushdown`    |        4,409.2 |            38.7 |         153.4 |            0.0 |         360.8 |          `5/1` | `[35, 28, 30, 29, 25, 32, 36, 33]`           |
| `q1_unordered` | `morsel_no_pushdown` |        3,207.7 |         2,246.4 |          16.2 |            0.0 |         273.6 |          `8/1` | `[29, 31, 31, 31, 30, 29, 28, 31]`           |
| `q1_unordered` | `morsel_pushdown`    |        3,374.0 |         1,922.2 |          38.9 |            0.0 |         333.5 |          `8/1` | `[25, 28, 30, 34, 31, 44, 26, 22]`           |
| `q6_count`     | `morsel_no_pushdown` |        1,146.7 |           168.5 |           8.4 |            0.0 |         124.1 |          `8/1` | `[30, 29, 30, 29, 30, 30, 32, 30]`           |
| `q6_count`     | `morsel_pushdown`    |        1,557.5 |             0.1 |          74.8 |            0.0 |         214.6 |          `6/1` | `[29, 30, 32, 33, 32, 24, 27, 33]`           |
| `q6_scan`      | `morsel_no_pushdown` |        2,557.4 |           226.8 |          17.3 |            0.0 |         260.3 |          `8/4` | `[32, 30, 28, 28, 30, 31, 31, 30]`           |
| `q6_scan`      | `morsel_pushdown`    |        3,618.6 |             8.1 |         116.3 |            0.0 |         361.5 |          `5/3` | `[24, 31, 29, 30, 35, 27, 36, 28]`           |

Key SF5 Morsel observations:

- **`q14` shows why `RowFilter` pushdown shines on selective `Date32`
  predicates:** `l_shipdate` selects 1 month (~1.2% of `lineitem`), so
  `morsel_pushdown` skips decoding `l_partkey`/`l_extendedprice`/`l_discount`
  for ~98.8% of rows, slashing `decode_work` from `2,460.9` → `1,492.5 ms`
  (`−39%`) and `process_work` from `253.3` → `78.3 ms`, turning a `455.7 ms`
  `morsel_no_pushdown` run into a `354.7 ms` win over both `stock_default`
  (`402.3 ms`, `1.13×`) and `stock_no_pushdown` (`393.4 ms`, `1.11×`) and
  landing within `1.09×` of DuckDB 2.0.1 (`326.7 ms`).
- **`q19` shows the opposite pushdown trade-off:** its probe-side `lineitem`
  filter mixes `Utf8View` (`l_shipmode`, `l_shipinstruct`) and `Decimal128`
  (`l_quantity`), where Parquet `RowFilter` adds `+1,335 ms` of decode work
  (`3,074.2` → `4,409.2 ms`); without pushdown, `morsel_no_pushdown`
  (`521.8 ms`) beats both `stock_default` (`554.3 ms`, `1.06×`) and
  `stock_no_pushdown` (`545.9 ms`, `1.05×`).
- **Low scheduling overhead at SF5:** under `morsel_no_pushdown`, aggregate
  `morsel_ready_wait` across all 8 workers is `8.4–17.3 ms` on single-table
  queries (`32.0–38.3 ms` across two pipelines on `q14`/`q19`), and
  `morsel_output_wait` is `0.0 ms` on every query and arm.

## 3. Live 22-Query Morsel Admission & Operator-Capability Checklist

Every standard TPC-H query (`q1`–`q22`) was probed with
`tpch_runner plan required no_pushdown` at revision `0ff9edecf` (after `#188`
native Inner `HashJoin`). Because `MorselQueryPlanner` walks logical plans
top-down and stops at the first unsupported operator, the table below lists
**both**:

1. The **verbatim planning rejection** emitted under `morsel_mode=required`, and
2. The **full set of physical plan operators** observed in Stock DataFusion's
   SF5 plan (`Sort`/`TopN`, `HashJoin` join types, `ScalarSubquery`,
   multi-stage `AggregateExec`, `COUNT(DISTINCT)`), providing a live checklist
   for the upcoming `Sort`/`TopN`, multi-way `Inner Join`,
   `Semi`/`Anti`/`Outer Join`, and Multi-Stage Composition issues.

| Query | Status at HEAD | Underlying Stock Physical Plan Capabilities Needed                             | Verbatim `morsel_mode=required` Planning Rejection                                                                                                                                                                             |
| ----- | -------------- | ------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `q1`  | Rejected       | `Sort`, `Agg` (1 stage)                                                        | `Error during planning: morsel_mode=required: unsupported logical operator: Sort: lineitem.l_returnflag ASC NULLS LAST, lineitem.l_linestatus ASC NULLS LAST`                                                                  |
| `q2`  | Rejected       | `TopN(100)`, `HashJoin(Inner×6, LeftSemi, RightSemi)`, `Agg`                   | `Error during planning: morsel_mode=required: unsupported logical operator: Sort: supplier.s_acctbal DESC NULLS FIRST, nation.n_name ASC NULLS LAST, supplier.s_name ASC NULLS LAST, part.p_partkey ASC NULLS LAST, fetch=100` |
| `q3`  | Rejected       | `TopN(10)`, `HashJoin(Inner×2)`, `Agg`                                         | `Error during planning: morsel_mode=required: unsupported logical operator: Sort: revenue DESC NULLS FIRST, orders.o_orderdate ASC NULLS LAST, fetch=10`                                                                       |
| `q4`  | Rejected       | `Sort`, `HashJoin(LeftSemi)`, `Agg`                                            | `Error during planning: morsel_mode=required: unsupported logical operator: Sort: orders.o_orderpriority ASC NULLS LAST`                                                                                                       |
| `q5`  | Rejected       | `Sort`, `HashJoin(Inner×5)`, `Agg`                                             | `Error during planning: morsel_mode=required: unsupported logical operator: Sort: revenue DESC NULLS FIRST`                                                                                                                    |
| `q6`  | **Admitted**   | `Agg` (ungrouped `SUM`) — **natively admitted**                                | _(admitted: `MorselExec: workers=8, NativeParquet[lineitem] -> Filter -> AggregateSink`)_                                                                                                                                      |
| `q7`  | Rejected       | `Sort`, `HashJoin(Inner×5)`, `Agg`                                             | `Error during planning: morsel_mode=required: unsupported logical operator: Sort: shipping.supp_nation ASC NULLS LAST, shipping.cust_nation ASC NULLS LAST, shipping.l_year ASC NULLS LAST`                                    |
| `q8`  | Rejected       | `Sort`, `HashJoin(Inner×7)`, `Agg`                                             | `Error during planning: morsel_mode=required: unsupported logical operator: Sort: all_nations.o_year ASC NULLS LAST`                                                                                                           |
| `q9`  | Rejected       | `Sort`, `HashJoin(Inner×5)`, `Agg`                                             | `Error during planning: morsel_mode=required: unsupported logical operator: Sort: profit.nation ASC NULLS LAST, profit.o_year DESC NULLS FIRST`                                                                                |
| `q10` | Rejected       | `TopN(20)`, `HashJoin(Inner×3)`, `Agg`                                         | `Error during planning: morsel_mode=required: unsupported logical operator: Sort: revenue DESC NULLS FIRST, fetch=20`                                                                                                          |
| `q11` | Rejected       | `Sort`, `HashJoin(Inner×4)`, `MultiStageAgg(×2)`, `ScalarSubquery`             | `Error during planning: morsel_mode=required: unsupported logical operator: Sort: value DESC NULLS FIRST`                                                                                                                      |
| `q12` | Rejected       | `Sort`, `HashJoin(Inner)`, `Agg`                                               | `Error during planning: morsel_mode=required: unsupported logical operator: Sort: lineitem.l_shipmode ASC NULLS LAST`                                                                                                          |
| `q13` | Rejected       | `Sort`, `HashJoin(Left)`, `MultiStageAgg(×2)`                                  | `Error during planning: morsel_mode=required: unsupported logical operator: Sort: custdist DESC NULLS FIRST, c_orders.c_count DESC NULLS FIRST`                                                                                |
| `q14` | **Admitted**   | `HashJoin(Inner)`, `Agg` — **natively admitted (`#188`)**                      | _(admitted: `MorselExec: workers=8, [Build: NativeParquet[part] -> HashJoinBuildSink]; NativeParquet[lineitem] -> Filter -> HashJoinProbe -> AggregateSink`)_                                                                  |
| `q15` | Rejected       | `Sort`, `HashJoin(Inner)`, `MultiStageAgg(×3)`, `ScalarSubquery`               | `Error during planning: morsel_mode=required: unsupported logical operator: Sort: supplier.s_suppkey ASC NULLS LAST`                                                                                                           |
| `q16` | Rejected       | `Sort`, `HashJoin(Inner, RightAnti)`, `MultiStageAgg(×2)`, `COUNT(DISTINCT)`   | `Error during planning: morsel_mode=required: unsupported logical operator: Sort: supplier_cnt DESC NULLS FIRST, part.p_brand ASC NULLS LAST, part.p_type ASC NULLS LAST, part.p_size ASC NULLS LAST`                          |
| `q17` | Rejected       | `HashJoin(Inner, LeftSemi + filter)`, `MultiStageAgg(×2)`                      | `Error during planning: morsel_mode=required: unsupported join type LeftSemi in native morsel hash join (only INNER JOIN is supported)`                                                                                        |
| `q18` | Rejected       | `TopN(100)`, `HashJoin(Inner×2, RightSemi)`, `MultiStageAgg(×2)`               | `Error during planning: morsel_mode=required: unsupported logical operator: Sort: orders.o_totalprice DESC NULLS FIRST, orders.o_orderdate ASC NULLS LAST, fetch=100`                                                          |
| `q19` | **Admitted**   | `HashJoin(Inner + disjunctive filter)`, `Agg` — **natively admitted (`#188`)** | _(admitted: `MorselExec: workers=8, [Build: NativeParquet[part] -> Filter -> HashJoinBuildSink]; NativeParquet[lineitem] -> Filter -> HashJoinProbe[filter=...] -> AggregateSink`)_                                            |
| `q20` | Rejected       | `Sort`, `HashJoin(Inner, LeftSemi×2, RightSemi)`, `Agg`                        | `Error during planning: morsel_mode=required: unsupported logical operator: Sort: supplier.s_name ASC NULLS LAST`                                                                                                              |
| `q21` | Rejected       | `TopN(100)`, `HashJoin(Inner×3, LeftSemi, RightAnti)`, `Agg`                   | `Error during planning: morsel_mode=required: unsupported logical operator: Sort: numwait DESC NULLS FIRST, supplier.s_name ASC NULLS LAST, fetch=100`                                                                         |
| `q22` | Rejected       | `Sort`, `HashJoin(LeftAnti)`, `MultiStageAgg(×2)`, `ScalarSubquery`            | `Error during planning: morsel_mode=required: unsupported logical operator: Sort: custsale.cntrycode ASC NULLS LAST`                                                                                                           |

### Capability Unlock Roadmap Summary

- **Already admitted (`3 / 22`):** `q6`, `q14`, `q19` (plus slices
  `q1_unordered`, `q6_count`, `q6_scan`).
- **`Sort` alone unlocks (`+1` → `4 / 22`):** `q1` (and with `#188`'s single
  `Inner Join`, **`q12`** as well once `Sort` lands, bringing single-join +
  `Sort` to `5 / 22`).
- **`Sort` / `TopN` + multi-way `Inner Join` pipelines unlock (`+7` →
  `11 / 22`):** `q12` (1 inner join + `Sort`), `q3` (2 inner joins + `TopN`),
  `q10` (3 inner joins + `TopN`), `q5`, `q7`, `q8`, `q9` (5–7 inner joins +
  `Sort`).
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
