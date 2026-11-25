<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# TPC-H SF5 Full 22/22 3-Way Scorecard: Morsel DataFusion vs. Stock DataFusion vs. DuckDB 2.0.1 (#192)

This scorecard reports the first **complete 22/22 standard TPC-H (`q1`–`q22`)
3-way comparison at Scale Factor 5 (`--sf 5`)** between **Morsel DataFusion
(`morsel_mode=required`)**, **Stock DataFusion (`morsel_mode=off`, default
settings)**, and **DuckDB 2.0.1** on quiet `rob-tandy-dduck2` (8 pinned ARM
Neoverse-N1 cores, verified 100% warm Linux page cache via `mincore`) after
`#191` landed full 22/22 TPC-H multi-stage DAG coverage on `morsel-execution`.

It builds directly on the pre-`#191` SF5 Stock-vs-DuckDB baseline in
[sf5_baseline_22q.md](sf5_baseline_22q.md) (`#189`) and the SF1 Checkpoint B
deep-dive in [checkpoint_b_results.md](checkpoint_b_results.md) (`#174`).

## Executive Summary

- **100% native admission and DuckDB parity across all 22 standard TPC-H
  queries:** Every single query (`q1`–`q22`) passed lossless Arrow IPC result
  parity against **DuckDB 2.0.1** at both `--sf 0.01` (20/20 `morsel_tpch`
  unittests passing across all 22 standard queries + 3 slices) and `--sf 5`,
  and executed strictly under `MorselExec: workers=8` with
  `morsel_mode=required` (**`22 / 22` admitted, `0` fallbacks, `0`
  rejections**).
- **22/22 SF5 warm-cache geometric means (8 pinned cores, fresh process,
  1 warmup + 5 counterbalanced repeats, `440 / 440` trials 100% `mincore`
  resident):**

| Arm                                                | GeoMean (22/22) | Sum of Medians (22/22) | GeoMean Ratio vs. `stock_default` | GeoMean Ratio vs. `duckdb_default` |
| -------------------------------------------------- | --------------: | ---------------------: | --------------------------------: | ---------------------------------: |
| Stock DataFusion (`stock_default`, `off`)          |        462.6 ms |            12,968.3 ms |                             1.00× |                              1.52× |
| Morsel DataFusion (`morsel_no_pushdown`, `req`)    |        690.7 ms |            22,660.6 ms |                             1.49× |                              2.27× |
| Morsel DataFusion (`morsel_pushdown`, `req`)       |        716.3 ms |            23,240.9 ms |                             1.55× |                              2.36× |
| **Best Morsel per query (`morsel_mode=required`)** |    **679.7 ms** |        **22,490.5 ms** |                         **1.47×** |                          **2.24×** |
| **DuckDB 2.0.1 (`duckdb_default`)**                |    **304.0 ms** |         **7,680.5 ms** |                         **0.66×** |                          **1.00×** |

- **Where Morsel already beats Stock DataFusion (`stock_default`) or DuckDB
  2.0.1 at SF5:**
  - Morsel beats `stock_default` on **5 / 22 queries**: **`q14`** (`358.8` vs.
    `403.6 ms`, **`0.89×`** / `1.12×` faster, within `1.09×` of DuckDB),
    **`q1`** (`713.2` vs. `768.9 ms`, **`0.93×`** / `1.08×` faster), **`q19`**
    (`519.7` vs. `551.3 ms`, **`0.94×`** / `1.06×` faster), **`q6`** (`347.7`
    vs. `360.7 ms`, **`0.96×`** / `1.04×` faster), and **`q15`** (`702.4` vs.
    `715.8 ms`, **`0.98×`** / `1.02×` faster).
  - Morsel **beats DuckDB 2.0.1 on `q22`** (`135.0` vs. `139.4 ms`,
    **`0.97×`** / `1.03×` faster than DuckDB).
  - Morsel is within `4–11%` of `stock_default` on `q3` (`1.04×`), `q5`
    (`1.09×`), and `q22` (`1.11×`), with **`0.0 ms` `morsel_output_wait`**
    across all 220 measured Morsel trials and `< 0.5 ms` `SortSink`/`TopNSink`
    finalization overhead on `21 / 22` queries (`3.9 ms` on `q16`'s
    `27,724`-row sort).
- **Why Morsel's 22-query sum is `22,490.5 ms` vs. Stock's `12,968.3 ms`
  (`+9,522.2 ms`) — `#138` metrics pinpoint two massive serial bottlenecks plus
  dynamic join filter pushdown (`#170`):**
  1. **Single-threaded coordinator `HashJoinBuildSink::finalize` consumes
     `7,258.4 ms` (`32.3%` of total 22-query wall time!)** — dominated by
     `q21` (`3,393.2 ms` / `76.8%` of wall!), `q9` (`947.6 ms` / `49.9%`),
     `q18` (`784.2 ms` / `27.9%`), `q7` (`644.7 ms` / `47.4%`), `q12`
     (`627.5 ms` / `73.2%`, also hit by build-side inversion on 7.5M `orders`
     rows), `q8` (`243.3 ms`), `q5` (`139.9 ms`), and `q10` (`116.8 ms`).
  2. **Single-threaded coordinator `AggregateSink::finalize` on high-cardinality
     SF5 `GROUP BY` stages consumes `~3,850 ms` (`~17%` of 22-query wall
     time!)** — concentrated in four multi-stage queries where 8 workers build
     independent `750k–7.5M`-group hash tables and 1 coordinator thread merges
     them serially: `q17` (`~1,249 ms` on 1.0M `l_partkey` groups), `q18`
     (`~1,116 ms` on 7.5M `l_orderkey` groups), `q20` (`~672 ms` on ~2.7M
     `(l_partkey, l_suppkey)` groups), and `q13` (`~485 ms` on 750k `c_custkey`
     groups).
  3. **Lack of dynamic join filter pushdown (`#170`)** leaves Morsel decoding
     and probing millions of non-matching rows on selective star/semi joins
     (`q17` 0.1% `part`, `q18` 0.004% `orders`, `q8` 0.7% `part`, `q20` 1.1%
     `part`, `q9` 5.5% `part`, `q21` 4% `nation`, `q3`/`q5`/`q7`/`q10`), where
     Stock DataFusion already pushes `DynamicFilter` (`20 / 22` queries,
     `16 / 22` with `KEY_RANGE_BITMAP` row-group pruning — saving `19.3 MB` of
     Parquet I/O on `q18` alone).

Together, **parallel radix-partitioned hash tables** (fixing both `HashJoinBuildSink`
and `AggregateSink` coordinator merges) + **selectivity-aware build-side
selection** + **dynamic join filter pushdown (`#170`)** address **>90% of the
Morsel-vs-Stock gap** and the largest Morsel-vs-DuckDB gaps at SF5.

## Environment, Controls, and Reproduction Commands

### Host and Build Metadata

- **Host:** `rob-tandy-dduck2`, Ubuntu 24.04.5 LTS, Linux `7.0.0-1012-aws`,
  16 ARM Neoverse-N1 cores (1 thread/core, 64 KiB L1d / 1 MiB L2 per core,
  32 MiB shared L3), 61 GiB RAM.
- **CPU pinning and threads:** `taskset -c 0-7` (8 dedicated CPUs),
  `--threads 8` (Tokio `worker_threads=8`, `max_blocking_threads=8`,
  `target_partitions=8`; DuckDB `SET threads = 8`), batch size `8,192`.
- **Host idle verification (`vmstat -y 1 3`):**
  - Immediately before benchmark: **89–93% idle** (`us 6–11%, sy 0–1%`).
  - Immediately after benchmark: **95–97% idle** (`us 2–3%, sy 1%`).
- **Build & versions:**
  - DataFusion revision: `3de519ad25ab1ac69378c5f3bbb761842ca0addc`
    (`agent/morsel-issue-192`, harness updates only on top of `morsel-execution`
    HEAD `3ba7893d9` after `#191`), built with `--locked --profile
release-nonlto -p datafusion --no-default-features --features
parquet,sql,datetime_expressions,unicode_expressions --example
morsel_tpch_runner`.
  - Rust `1.99.1` (`58cf8f9c3`, `aarch64-unknown-linux-gnu`), Cargo `1.99.1`,
    Python `3.12.10`, DuckDB `2.0.1`, PyArrow `26.0.1`, NumPy `2.5.5`.
- **Dataset & page-cache verification:** Deterministic `#165`/`#189` SF5 TPC-H
  Parquet dataset (`43,299,825` rows, `35` Zstd level-3 files, `353` row groups
  at `row_group_size=131,072`, `1,400,570,073` bytes / `1,335.7 MiB` on local
  EC2 NVMe Btrfs storage under `$HOME/.cache/issue189/tpch_sf5`; see
  [sf5_baseline_22q.md](sf5_baseline_22q.md#sf5-parquet-dataset-layout---sf-5---row-group-size-131072)
  for the per-table breakdown). All **`440 / 440` measured trials** started and
  ended with **100% of 4 KiB Parquet file pages resident** via Linux `mincore`.
- **Protocol:** Fresh process per trial, `1` unmeasured warmup round + `5`
  measured counterbalanced rounds (alternating arm order between odd and even
  rounds) with `MORSEL_METRICS=1` enabled on every Morsel trial. All timings
  below are medians in milliseconds with observed `min–max` in parentheses.

### Exact Reproduction Commands

```bash
set -euo pipefail
b=benchmarks/morsel_tpch
work="$HOME/.cache/issue192"
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

# 2. Create pinned Python venv and run SF=0.01 22/22 parity + plan unittest suite.
timeout 120s uv venv --python 3.12.10 "$work/venv"
timeout 180s uv pip install --python "$work/venv/bin/python" -r "$b/requirements.txt"
py="$work/venv/bin/python"
MORSEL_TPCH_RUNNER="$runner" MORSEL_BENCH_REVISION="$revision" \
  timeout 240s "$py" -B -m unittest discover -s "$b" -v

# 3. Generate deterministic SF5 TPC-H Parquet tables outside Git (or reuse existing SF5 dir).
data_dir="$work/tpch_sf5"
timeout 600s "$py" -B "$b/harness.py" generate "$data_dir" \
  --sf 5 --row-group-size 131072

# 4. Run quiet-host warm-cache SF5 22/22 3-way scorecard (440 timed trials).
vmstat -y 1 3
timeout --kill-after=10s 2400s taskset -c 0-7 "$py" -B "$b/harness.py" bench \
  "$data_dir" --runner "$runner" --revision "$revision" \
  --threads 8 --timeout 180 --cache warm --warmups 1 --repeats 5 \
  --queries all \
  --morsel-arms stock_default,morsel_no_pushdown,morsel_pushdown,duckdb_default \
  > "$work/sf5_22q_scorecard.jsonl"
vmstat -y 1 3

# 5. Print full summary, per-pipeline metrics, and 3-way geometric mean table.
"$py" -B "$b/harness.py" summarize "$work/sf5_22q_scorecard.jsonl"

# 6. Clean up the detached worktree when finished.
git worktree remove --force "$work/pinned"
```

## 1. Full 22/22 TPC-H SF5 3-Way Warm-Cache Scorecard

Every query verified exact Arrow IPC parity against DuckDB 2.0.1 before timing
and ran with `morsel_mode=required` (`0` fallback, `100%` `mincore` page
residency on all `440 / 440` trials). Ratios compare the faster of the two
Morsel arms (`Best Morsel`) against `stock_default` and `duckdb_default`
(`< 1.00×` means Morsel is faster; bold highlights wins over `stock_default` or
`duckdb_default`).

| Query               |   Rows | `stock_default` ms (min–max) | `morsel_no_pushdown` ms (min–max) | `morsel_pushdown` ms (min–max) | `duckdb_default` ms (min–max) | Best Morsel / Stock | Best Morsel / DuckDB |
| ------------------- | -----: | ---------------------------: | --------------------------------: | -----------------------------: | ----------------------------: | ------------------: | -------------------: |
| `q1`                |      4 |          768.9 (762.0–781.5) |               717.3 (709.3–723.7) |        **713.2** (699.2–756.1) |           465.8 (463.0–485.6) |           **0.93×** |                1.53× |
| `q2`                |    100 |          183.6 (177.7–192.4) |               282.3 (275.3–285.9) |            287.9 (275.8–306.5) |           124.0 (117.8–125.4) |               1.54× |                2.28× |
| `q3`                |     10 |          515.2 (502.5–532.1) |               537.1 (530.0–541.3) |            562.0 (552.7–571.2) |           346.0 (340.8–353.4) |               1.04× |                1.55× |
| `q4`                |      5 |          198.9 (194.1–232.5) |               271.7 (266.6–275.5) |            261.1 (259.2–264.9) |           186.3 (185.8–188.5) |               1.31× |                1.40× |
| `q5`                |      5 |          694.6 (677.5–743.0) |               756.7 (747.9–764.0) |            760.4 (753.0–771.7) |           371.7 (370.6–379.6) |               1.09× |                2.04× |
| `q6`                |      1 |          360.7 (356.8–364.8) |           **347.7** (332.5–368.7) |            485.0 (482.7–492.4) |           201.0 (199.8–204.7) |           **0.96×** |                1.73× |
| `q7`                |      4 |          859.7 (848.8–901.5) |         1,361.1 (1,340.1–1,395.1) |      1,411.6 (1,393.3–1,441.2) |           406.6 (401.1–416.5) |               1.58× |                3.35× |
| `q8`                |      2 |          653.8 (645.9–664.2) |               876.0 (872.9–894.9) |            900.0 (894.0–904.2) |           450.6 (448.8–456.6) |               1.34× |                1.94× |
| `q9`                |    175 |      1,017.8 (993.0–1,069.8) |         1,899.5 (1,897.5–1,912.1) |      1,903.7 (1,882.9–1,941.4) |           773.0 (767.1–777.4) |               1.87× |                2.46× |
| `q10`               |     20 |          653.3 (610.9–654.2) |               806.8 (803.6–821.5) |            780.9 (780.7–799.2) |           445.9 (440.6–451.6) |               1.20× |                1.75× |
| `q11`               |  4,606 |          115.4 (108.5–117.8) |               156.7 (155.2–159.3) |            160.7 (156.8–168.9) |             99.2 (92.9–107.2) |               1.36× |                1.58× |
| `q12`               |      2 |          217.6 (210.8–227.7) |               856.8 (852.7–864.7) |            912.8 (906.4–922.1) |           216.1 (213.8–222.1) |               3.94× |                3.97× |
| `q13`               |     42 |          471.4 (466.8–475.7) |             954.0 (914.7–1,001.4) |            930.9 (906.2–997.3) |           514.7 (511.2–535.6) |               1.98× |                1.81× |
| `q14`               |      1 |          403.6 (396.6–426.7) |               460.3 (452.4–461.7) |        **358.8** (348.6–367.5) |           330.0 (319.2–338.3) |           **0.89×** |                1.09× |
| `q15`               |      1 |          715.8 (703.7–717.6) |           **702.4** (695.1–705.6) |            902.3 (887.0–934.3) |           265.3 (262.0–271.4) |           **0.98×** |                2.65× |
| `q16`               | 27,724 |          139.0 (129.1–145.3) |               212.1 (210.3–214.0) |            220.3 (218.2–225.7) |           136.5 (133.6–137.4) |               1.53× |                1.55× |
| `q17`               |      1 |    1,172.6 (1,168.4–1,179.0) |         2,511.5 (2,496.6–2,538.7) |      2,506.7 (2,497.4–2,546.8) |           401.8 (398.1–403.4) |               2.14× |                6.24× |
| `q18`               |    100 |    1,372.4 (1,358.5–1,391.6) |         2,812.5 (2,796.6–2,880.9) |      2,818.6 (2,805.3–2,833.4) |           450.8 (448.3–465.8) |               2.05× |                6.24× |
| `q19`               |      1 |          551.3 (542.9–566.5) |           **519.7** (518.3–529.7) |            680.8 (669.1–683.9) |           327.2 (324.6–327.5) |           **0.94×** |                1.59× |
| `q20`               |    901 |          473.8 (466.1–514.9) |         1,065.2 (1,058.6–1,075.1) |      1,097.3 (1,074.5–1,113.0) |           301.1 (290.8–304.6) |               2.25× |                3.54× |
| `q21`               |    100 |    1,307.1 (1,304.4–1,330.8) |         4,418.0 (4,382.9–4,486.5) |      4,450.8 (4,440.0–4,524.5) |           727.7 (711.8–739.4) |               3.38× |                6.07× |
| `q22`               |      7 |          121.8 (116.9–123.7) |               135.0 (132.8–138.7) |            135.2 (132.6–141.5) |           139.4 (138.8–143.0) |               1.11× |            **0.97×** |
| **GeoMean (22/22)** |      — |                 **462.6 ms** |                      **690.7 ms** |                   **716.3 ms** |                  **304.0 ms** |           **1.47×** |            **2.24×** |
| **Sum (22/22)**     |      — |              **12,968.3 ms** |                   **22,660.6 ms** |                **23,240.9 ms** |                **7,680.5 ms** |           **1.73×** |            **2.93×** |

Notes on `morsel_no_pushdown` vs. `morsel_pushdown`:

- `morsel_no_pushdown` wins on **16 / 22 queries** (`690.7 ms` geomean vs.
  `716.3 ms` for `morsel_pushdown`), because DataFusion's multi-conjunct
  Parquet `RowFilter` adds decode overhead on dense or multi-column filters
  (`q6`: `347.7` vs. `485.0 ms`; `q15`: `702.4` vs. `902.3 ms`; `q19`: `519.7`
  vs. `680.8 ms`; `q12`: `856.8` vs. `912.8 ms`).
- `morsel_pushdown` wins on **6 / 22 queries** where a selective single-column
  predicate skips decoding wide payload columns: **`q14`** (`358.8` vs.
  `460.3 ms`, `−101.5 ms` / `−22.1%`), **`q10`** (`780.9` vs. `806.8 ms`),
  **`q13`** (`930.9` vs. `954.0 ms`), **`q4`** (`261.1` vs. `271.7 ms`),
  **`q17`** (`2,506.7` vs. `2,511.5 ms`), and **`q1`** (`713.2` vs.
  `717.3 ms`).
- `bytes_scanned` matched identically between `stock_default` and Morsel on
  `20 / 22` queries. On `q3`, `morsel_pushdown` matched `stock_default`
  (`334,406,980` bytes vs. `334,437,108` under `morsel_no_pushdown`). On `q18`,
  `stock_default` scanned `237,511,219` bytes vs. Morsel's `256,827,407` bytes
  (`−19.3 MB`) because Stock pushed a `KEY_RANGE_BITMAP` dynamic join filter
  from the `301`-row `RightSemi` build into both `orders` and `lineitem`.

## 2. 22/22 Native `MorselExec` Execution & Wait/Work Summary

Every row below reports the faster Morsel arm (`Best Morsel`) from the SF5 run
(`MORSEL_METRICS=1`). `Decode`, `Process`, and `Ready wait` are summed across 8
workers (`Output wait` was **`0.0 ms`** on all 220 measured Morsel trials).
`Join fin` and `Sort/TopN fin` are single-threaded coordinator wall times.
`Est. serial agg/other` (`max(0, exec - (decode + process)/8 - join_fin -
sort_topn_fin)`) isolates the remaining single-threaded coordinator wall time
— predominantly **`AggregateSink::finalize`** merging 8 worker hash tables on
high-cardinality `GROUP BY` stages.

| Query | Best Morsel arm      | `mincore` pages (100%) | RGs claimed | Stages / `jb` / `agg` / `gc` / `s` | Exec ms | Exec CPU ms | Decode ms | Process ms | Ready wait ms | Join fin ms | Sort/TopN fin ms | Est. serial agg/other ms |
| ----- | -------------------- | ---------------------: | ----------: | ---------------------------------- | ------: | ----------: | --------: | ---------: | ------------: | ----------: | ---------------: | -----------------------: |
| `q1`  | `morsel_pushdown`    |                229,147 |         240 | `0 / 0 / 1 / 0 / Sort`             |   700.2 |     5,404.8 |   3,322.4 |    1,916.5 |          37.8 |         0.0 |              0.1 |                     45.3 |
| `q2`  | `morsel_no_pushdown` |                 44,595 |          78 | `1 / 8 / 1 / 0 / TopN`             |   272.1 |     1,152.7 |     514.9 |      491.3 |          70.5 |        50.3 |              0.4 |                     95.7 |
| `q3`  | `morsel_no_pushdown` |                297,360 |         310 | `0 / 2 / 1 / 0 / TopN`             |   523.7 |     3,538.7 |   2,381.2 |      967.6 |          43.7 |        62.5 |              0.2 |                     42.5 |
| `q4`  | `morsel_pushdown`    |                286,442 |         304 | `0 / 1 / 1 / 0 / Sort`             |   247.2 |     1,613.6 |     759.1 |      725.4 |          36.4 |        23.3 |              0.0 |                     38.3 |
| `q5`  | `morsel_no_pushdown` |                298,107 |         313 | `0 / 5 / 1 / 0 / Sort`             |   741.9 |     4,655.7 |   2,723.5 |    1,658.8 |          50.3 |       139.9 |              0.0 |                     54.2 |
| `q6`  | `morsel_no_pushdown` |                229,147 |         240 | `0 / 0 / 1 / 0 / —`                |   337.5 |     2,550.1 |   2,300.9 |      218.3 |          13.1 |         0.0 |              0.0 |                     22.6 |
| `q7`  | `morsel_no_pushdown` |                298,106 |         313 | `0 / 5 / 1 / 0 / Sort`             | 1,346.2 |     5,760.0 |   2,844.3 |    2,102.3 |          52.5 |   **644.7** |              0.0 |                     83.1 |
| `q8`  | `morsel_no_pushdown` |                303,951 |         322 | `0 / 7 / 1 / 0 / Sort`             |   858.8 |     4,824.8 |   3,377.3 |    1,018.4 |          66.8 |   **243.3** |              0.0 |                     66.1 |
| `q9`  | `morsel_no_pushdown` |                331,036 |         346 | `0 / 5 / 1 / 0 / Sort`             | 1,883.7 |     7,834.0 |   4,055.9 |    2,662.2 |          91.7 |   **947.6** |              0.0 |                     96.3 |
| `q10` | `morsel_pushdown`    |                297,361 |         311 | `0 / 3 / 1 / 0 / TopN`             |   765.4 |     3,989.2 |   2,875.9 |      758.7 |         154.2 |       116.8 |              0.2 |                **194.1** |
| `q11` | `morsel_no_pushdown` |                 38,750 |          68 | `1 / 4 / 2 / 0 / Sort`             |   148.7 |       965.5 |     474.5 |      427.6 |          47.1 |         4.2 |              0.4 |                     31.3 |
| `q12` | `morsel_no_pushdown` |                286,442 |         304 | `0 / 1 / 1 / 0 / Sort`             |   843.5 |     1,966.8 |     893.9 |      323.5 |          18.2 |   **627.5** |              0.0 |                     63.8 |
| `q13` | `morsel_pushdown`    |                 68,213 |          70 | `1 / 1 / 1 / 1 / Sort`             |   923.5 |     3,685.5 |   1,756.2 |    1,334.7 |          35.7 |        52.3 |              0.0 |                **484.8** |
| `q14` | `morsel_pushdown`    |                234,991 |         248 | `0 / 1 / 1 / 0 / —`                |   344.9 |     1,853.8 |   1,499.4 |       82.2 |          95.1 |        75.0 |              0.0 |                     72.2 |
| `q15` | `morsel_no_pushdown` |                229,892 |         481 | `3 / 1 / 3 / 0 / Sort`             |   689.7 |     5,054.7 |   4,209.5 |      607.4 |          44.8 |         3.1 |              0.0 |                     84.4 |
| `q16` | `morsel_no_pushdown` |                 44,593 |          41 | `1 / 2 / 2 / 0 / Sort`             |   204.4 |       608.1 |     137.6 |      337.3 |          29.6 |        11.6 |              3.9 |                **129.4** |
| `q17` | `morsel_pushdown`    |                234,991 |         488 | `1 / 2 / 2 / 0 / —`                | 2,492.7 |    10,758.1 |   3,485.3 |    5,908.5 |          40.6 |        69.4 |              0.0 |              **1,249.0** |
| `q18` | `morsel_no_pushdown` |                297,360 |         550 | `1 / 3 / 2 / 0 / TopN`             | 2,799.6 |     9,104.0 |   2,368.0 |    4,828.2 |          44.9 |   **784.2** |              0.0 |              **1,115.9** |
| `q19` | `morsel_no_pushdown` |                234,991 |         248 | `0 / 1 / 1 / 0 / —`                |   506.7 |     3,933.8 |   3,069.2 |      746.4 |          36.1 |         0.4 |              0.0 |                     29.3 |
| `q20` | `morsel_no_pushdown` |                273,741 |         282 | `1 / 4 / 1 / 0 / Sort`             | 1,050.8 |     3,717.2 |   1,841.1 |    1,166.2 |          53.9 |         2.9 |              0.1 |                **672.0** |
| `q21` | `morsel_no_pushdown` |                287,188 |         786 | `0 / 5 / 1 / 0 / TopN`             | 4,403.1 |     9,609.6 |   2,720.2 |    2,986.7 |          53.0 | **3,393.2** |              0.0 |                    296.6 |
| `q22` | `morsel_no_pushdown` |                 68,213 |          76 | `1 / 1 / 2 / 0 / Sort`             |   126.7 |       754.7 |     323.7 |      383.8 |          60.9 |         6.3 |              0.0 |                     32.0 |

_(Column `Stages / jb / agg / gc / s` lists intermediate DAG stages,
`join_build_finalizations`, `aggregate_finalizations`,
`group_count_finalizations`, and final `Sort`/`TopN` sink. Summed across all 22
queries, single-threaded `join_finalize` totals **`7,258.4 ms`** (`32.3%` of
`22,490.5 ms`), while `sort_finalize` totals only **`4.9 ms`** and
`topn_finalize` totals **`0.8 ms`**.)_

## 3. `#138` Per-Pipeline Wait/Work Breakdowns for the 10 Slowest SF5 Queries

The 10 slowest Morsel queries at SF5 (`q21`, `q18`, `q17`, `q9`, `q7`, `q20`,
`q13`, `q8`, `q12`, `q10`) account for **`17,427.6 ms` (`77.5%`)** of the
22-query sum (`22,490.5 ms`) and **`+9,259.6 ms` (`97.2%`)** of the total
Morsel-vs-Stock wall-time delta (`+9,522.2 ms`). Every table below shows median
per-pipeline `#138` metrics for the faster Morsel arm (8 workers; coordinator
`Join comb/fin` and `Sort/TopN fin` are single-threaded wall ms; `Output wait`
was `0.0 ms` on every pipeline).

### `q21` — `4,418.0 ms` (`morsel_no_pushdown`) vs. Stock `1,307.1 ms` (`3.38×`) vs. DuckDB `727.7 ms` (`6.07×`)

| Pipeline  | Role in `MorselExec` plan                                   | Decode ms | Process ms | Ready wait ms | Join comb/fin ms | Claims (min..max, sum) |
| --------- | ----------------------------------------------------------- | --------: | ---------: | ------------: | ---------------: | ---------------------: |
| `build_0` | `supplier` (50k rows)                                       |       4.8 |        0.3 |           4.0 |          0.0/2.6 |               0..1 (1) |
| `build_1` | `orders` (`o_orderstatus = 'F'`, ~3.6M rows)                |     245.8 |       28.6 |           9.0 |    0.0/**262.6** |             7..10 (64) |
| `build_2` | `nation` (`SAUDI ARABIA`, 1 row)                            |       0.0 |        0.0 |           3.1 |          0.0/0.0 |               0..1 (1) |
| `build_3` | `lineitem` (unfiltered `30.0M` rows for `EXISTS` LeftSemi)  |     902.3 |      201.1 |          10.3 |  0.2/**1,725.6** |           28..32 (240) |
| `build_4` | `lineitem` (`receipt > commit`, `~18.9M` rows for LeftAnti) |     763.3 |      176.1 |          12.4 |  0.1/**1,413.3** |           28..31 (240) |
| `probe`   | `lineitem` (`receipt > commit`) → 5× probe → `Agg`          |     787.6 |    2,560.7 |          16.0 |          0.0/0.0 |           29..32 (240) |
| `topn`    | `TopNSink(100)`                                             |       0.0 |        0.3 |           0.0 |          0.0/0.0 |               0..1 (1) |

- **Diagnosis:** **`3,393.2 ms` (`76.8%` of total wall time!)** is spent
  building three giant hash tables serially on 1 coordinator thread (`build_3`
  `1,725.6 ms` + `build_4` `1,413.3 ms` + `build_1` `262.6 ms`). Stock
  DataFusion partitions its hash builds across 8 threads (`HashJoinExec`
  `5,035.7 ms` CPU / `~630 ms` parallel wall, `DataSourceExec` `2,501.6 ms`,
  `AggregateExec` `583.1 ms`). Morsel's parallel worker work (`dec + proc =
5,706.9 ms` = `713.4 ms` wall across 8 cores) is already competitive with
  Stock — **radix-partitioned parallel HT build + pushing `SAUDI ARABIA`
  supplier/order dynamic filters into `build_3`/`build_4`** eliminates
  `>3,000 ms` of wall time on `q21` alone.

### `q18` — `2,812.5 ms` (`morsel_no_pushdown`) vs. Stock `1,372.4 ms` (`2.05×`) vs. DuckDB `450.8 ms` (`6.24×`)

| Pipeline        | Role in `MorselExec` plan                                   | Decode ms | Process ms | Ready wait ms | Join comb/fin ms | Claims (min..max, sum) |
| --------------- | ----------------------------------------------------------- | --------: | ---------: | ------------: | ---------------: | ---------------------: |
| `stage_0`       | `lineitem` → `GROUP BY l_orderkey HAVING SUM > 300` (7.5M!) |     799.0 |    1,970.1 |           7.4 |          0.0/0.0 |           29..31 (240) |
| `build_0_build` | `customer` (750k rows)                                      |      46.1 |        3.9 |           7.0 |     0.0/**58.1** |               0..1 (6) |
| `build_0`       | `orders ⋈ customer` (7.5M unfiltered rows!)                 |     727.7 |      934.3 |          22.4 |    0.0/**726.9** |              7..9 (64) |
| `build_1`       | `StageSource[stage_0]` (`301` qualifying `l_orderkey`s!)    |       0.0 |        0.4 |           0.0 |          0.0/0.1 |              6..9 (65) |
| `probe`         | `lineitem` (30M rows) → Inner(`build_0`) → LeftSemi(`301`)  |     785.1 |    1,916.9 |           6.1 |          0.0/0.0 |           28..33 (240) |
| `topn`          | `TopNSink(100)`                                             |       0.0 |        0.2 |           0.0 |          0.0/0.0 |               0..1 (1) |

- **Diagnosis:** A **triple whammy** of all three top bottlenecks on one query:
  1. **`~1,116 ms` serial `AggregateSink::finalize`** merging 8 worker hash
     tables over **7.5M `l_orderkey` groups** in `stage_0` (`exec_cpu`
     unattributed coordinator CPU = `1,123.6 ms`),
  2. **`784.2 ms` serial `HashJoinBuildSink::finalize`** (`build_0` `726.9 ms`
     - `build_0_build` `58.1 ms`) building a full `7.5M`-row `orders ⋈
customer` HT even though `stage_0` produced only **`301` qualifying
       `l_orderkey`s (`0.004%`)**, and
  3. **No dynamic join filter pushdown (`#170`) or semi-join reordering:**
     Stock pushes a `301`-bucket `KEY_RANGE_BITMAP` dynamic filter into `orders`
     and `lineitem` (pruning `19.3 MB` of Parquet I/O), while DuckDB (`450.8 ms`)
     combines partitioned radix aggregation with semi-join pushdown to skip
     nearly all `orders` and probe `lineitem` work.

### `q17` — `2,506.7 ms` (`morsel_pushdown`) vs. Stock `1,172.6 ms` (`2.14×`) vs. DuckDB `401.8 ms` (`6.24×`)

| Pipeline  | Role in `MorselExec` plan                                          | Decode ms | Process ms | Ready wait ms | Join comb/fin ms | Claims (min..max, sum) |
| --------- | ------------------------------------------------------------------ | --------: | ---------: | ------------: | ---------------: | ---------------------: |
| `stage_0` | `lineitem` (30M rows) → `GROUP BY l_partkey` (1.0M groups × 8!)    |   1,217.2 |    5,250.8 |          11.7 |          0.0/0.0 |           29..31 (240) |
| `build_0` | `part` (`Brand#23`, `MED BOX` → **~1,000 rows, 0.1%!**)            |      28.9 |        0.0 |          10.7 |          0.0/0.1 |               1..1 (8) |
| `build_1` | `StageSource[stage_0]` (1.0M `l_partkey` rows → HT)                |       0.0 |        2.9 |           0.0 |     0.0/**69.3** |              3..4 (31) |
| `probe`   | `lineitem` (30M rows) → Inner(`part`) → LeftSemi(`stage_0` + filt) |   2,238.5 |      662.9 |          15.9 |          0.0/0.0 |           28..32 (240) |

- **Diagnosis:** **`~1,249 ms` (`49.8%` of wall time!)** is spent in
  single-threaded `AggregateSink::finalize` merging 8 × 1,000,000-group hash
  tables in `stage_0` (`unattributed coordinator CPU = 1,294.9 ms`), plus
  `5,250.8 ms` of worker CPU populating 8 redundant 1M-group HTs (Stock's
  hash-partitioned `AggregateExec` takes `4,178.8 ms` CPU total with zero
  serial merge). Furthermore, `part` selects only **`~1,000` `p_partkey`s
  (`0.1%`)** — pushing that `1,000`-key dynamic filter into BOTH `stage_0` and
  `probe` `lineitem` scans (as DuckDB does, `401.8 ms`) collapses `stage_0`
  from 1M groups to 1k groups and skips `99.9%` of `probe` column decodes!

### `q9` — `1,899.5 ms` (`morsel_no_pushdown`) vs. Stock `1,017.8 ms` (`1.87×`) vs. DuckDB `773.0 ms` (`2.46×`)

| Pipeline  | Role in `MorselExec` plan                             | Decode ms | Process ms | Ready wait ms | Join comb/fin ms | Claims (min..max, sum) |
| --------- | ----------------------------------------------------- | --------: | ---------: | ------------: | ---------------: | ---------------------: |
| `build_0` | `part` (`p_name LIKE '%green%'`, ~5.5% / ~55k rows)   |     107.6 |       25.3 |          30.4 |          0.0/2.2 |               1..1 (8) |
| `build_1` | `supplier` (50k rows)                                 |       1.2 |        0.2 |           3.2 |          0.0/2.0 |               0..1 (1) |
| `build_2` | `partsupp` (unfiltered `4.0M` rows)                   |     290.5 |       39.6 |          11.4 |    0.0/**372.8** |              4..4 (32) |
| `build_3` | `orders` (unfiltered `7.5M` rows)                     |     195.5 |       23.4 |          14.1 |    0.0/**565.7** |             4..10 (64) |
| `build_4` | `nation` (25 rows)                                    |       0.1 |        0.0 |           2.7 |          0.0/0.0 |               0..1 (1) |
| `probe`   | `lineitem` (30M rows) → 5× Inner probe → `Agg` (175g) |   3,466.6 |    2,568.6 |          23.8 |          0.0/0.0 |           28..33 (240) |
| `sort`    | `SortSink` (175 rows)                                 |       0.0 |        0.1 |           0.0 |          0.0/0.0 |               0..1 (1) |

- **Diagnosis:** **`947.6 ms` (`49.9%` of wall time!)** is single-threaded
  `join_finalize` on `orders` (`565.7 ms`) and `partsupp` (`372.8 ms`).
  Removing just that serial finalize already brings `q9` to `~1,050 ms` (at
  parity with Stock `1,017.8 ms`). Pushing `build_0`'s 5.5% `p_partkey` filter
  into `lineitem` (`3,466.6 ms` decode!) and `partsupp` (`4.0M → 220k` build
  rows) targets the remaining gap to DuckDB (`773.0 ms`).

### `q7` — `1,361.1 ms` (`morsel_no_pushdown`) vs. Stock `859.7 ms` (`1.58×`) vs. DuckDB `406.6 ms` (`3.35×`)

| Pipeline  | Role in `MorselExec` plan                         | Decode ms | Process ms | Ready wait ms | Join comb/fin ms | Claims (min..max, sum) |
| --------- | ------------------------------------------------- | --------: | ---------: | ------------: | ---------------: | ---------------------: |
| `build_0` | `supplier` (50k rows)                             |       1.8 |        0.3 |           3.8 |          0.0/2.2 |               0..1 (1) |
| `build_1` | `orders` (unfiltered `7.5M` rows!)                |     431.3 |       51.8 |          17.5 |    0.0/**591.8** |             6..10 (64) |
| `build_2` | `customer` (unfiltered `750k` rows)               |      16.1 |        2.2 |           6.7 |     0.0/**50.4** |               0..1 (6) |
| `build_3` | `nation` (`FRANCE` / `GERMANY`, 2 rows)           |       0.1 |        0.1 |           3.6 |          0.0/0.0 |               0..1 (1) |
| `build_4` | `nation` (`GERMANY` / `FRANCE`, 2 rows)           |       0.0 |        0.1 |           2.2 |          0.0/0.0 |               0..1 (1) |
| `probe`   | `lineitem` (`1995–1996`) → 5× Inner probe → `Agg` |   2,392.0 |    2,052.8 |          19.3 |          0.0/0.0 |           29..32 (240) |
| `sort`    | `SortSink` (4 rows)                               |       0.0 |        0.1 |           0.0 |          0.0/0.0 |               0..1 (1) |

- **Diagnosis:** **`644.7 ms` (`47.4%` of wall time!)** is single-threaded
  `join_finalize` on unfiltered `orders` (`591.8 ms`) + `customer` (`50.4 ms`).
  Notice Morsel's total CPU (`5,760.0 ms`) is actually **lower** than Stock's
  (`6,076.5 ms`) — parallelizing `join_finalize` alone drops `q7` to
  **`~780–800 ms`**, beating Stock (`859.7 ms`), while pushing the `2/25`
  `FRANCE`/`GERMANY` `nation` filter down into `supplier`/`customer`/`orders`/
  `lineitem` closes the gap to DuckDB (`406.6 ms`).

### `q20` — `1,065.2 ms` (`morsel_no_pushdown`) vs. Stock `473.8 ms` (`2.25×`) vs. DuckDB `301.1 ms` (`3.54×`)

| Pipeline        | Role in `MorselExec` plan                                | Decode ms | Process ms | Ready wait ms | Join comb/fin ms | Claims (min..max, sum) |
| --------------- | -------------------------------------------------------- | --------: | ---------: | ------------: | ---------------: | ---------------------: |
| `stage_0`       | `lineitem` (`1994`, ~4.5M rows) → `GROUP BY (part,supp)` |   1,658.1 |      922.1 |          22.0 |          0.0/0.0 |           28..32 (240) |
| `build_0_build` | `part` (`p_name LIKE 'forest%'`, ~1.1% / ~11k rows)      |      87.5 |        6.5 |          16.7 |          0.0/0.5 |               1..1 (8) |
| `build_0`       | `partsupp` LeftSemi `build_0_build` (~44k rows)          |      90.7 |       78.6 |           7.7 |          0.0/2.3 |              2..5 (32) |
| `build_1_build` | `nation` (`CANADA`, 1 row)                               |       0.0 |        0.0 |           2.7 |          0.0/0.0 |               0..1 (1) |
| `build_1`       | `supplier` ⋈ `nation` (~2k rows)                         |       6.4 |        0.5 |           4.8 |          0.0/0.1 |               0..1 (1) |
| `probe`         | `StageSource[stage_0]` → 2× LeftSemi → `SortSink`        |       0.0 |      156.9 |           0.0 |          0.0/0.0 |            10..11 (84) |
| `build_emit_*`  | Semi-join build emissions (`0` + `1`)                    |       0.0 |        1.1 |           0.0 |          0.0/0.0 |               0..1 (7) |

- **Diagnosis:** `join_finalize` is tiny (`2.9 ms`), yet `exec` is `1,050.8 ms`
  while `(dec + proc)/8` is only `375.9 ms`! **`~672 ms` (`63.1%` of wall
  time!)** is single-threaded `AggregateSink::finalize` merging 8 worker hash
  tables over `~2.7M` `(l_partkey, l_suppkey)` pairs in `stage_0`
  (`unattributed CPU = 707.0 ms`). Radix-partitioned parallel aggregate merge
  brings `q20` straight to parity with Stock (`473.8 ms`), and pushing the
  `1.1%` `forest%` `p_partkey` dynamic filter into `stage_0` slashes `stage_0`
  by `~90×` toward DuckDB (`301.1 ms`).

### `q13` — `930.9 ms` (`morsel_pushdown`) vs. Stock `471.4 ms` (`1.98×`) vs. DuckDB `514.7 ms` (`1.81×`)

| Pipeline        | Role in `MorselExec` plan                                   | Decode ms | Process ms | Ready wait ms | Join comb/fin ms | Claims (min..max, sum) |
| --------------- | ----------------------------------------------------------- | --------: | ---------: | ------------: | ---------------: | ---------------------: |
| `stage_0_build` | `customer` (750k rows)                                      |      31.7 |        6.6 |           6.5 |     0.1/**52.3** |               0..1 (6) |
| `stage_0_probe` | `orders` (`NOT LIKE %special%requests%`) ⟕ `customer` → Agg |   1,723.8 |    1,310.4 |          29.3 |          0.0/0.0 |             2..11 (64) |
| `stage_0_emit`  | Unmatched `customer` build-side emission                    |       0.0 |       13.3 |           0.0 |          0.0/0.0 |            11..13 (92) |
| `main`          | `StageSource[stage_0]` → `GroupCountSink` (42 groups)       |       0.0 |        4.0 |           0.0 |          0.0/0.0 |              2..4 (23) |
| `sort`          | `SortSink` (42 rows)                                        |       0.0 |        0.1 |           0.0 |          0.0/0.0 |               0..1 (1) |

- **Diagnosis:** Morsel's total CPU (`3,685.5 ms`) is ALREADY **lower than
  DuckDB 2.0.1 (`3,746.7 ms`)** and close to Stock (`3,426.7 ms`)! The entire
  `2×` wall-time gap is **`~485 ms` (`52.1%` of wall!)** in single-threaded
  `AggregateSink::finalize` merging 8 worker HTs over `750k` `c_custkey` groups
  plus `52.3 ms` serial `stage_0_build` finalize. Parallelizing both immediately
  projects `q13` to **`~440–460 ms` — beating BOTH Stock (`471.4 ms`) and
  DuckDB (`514.7 ms`)**!

### `q8`, `q12`, and `q10` — Remaining Top-10 Slowest Queries

| Query | Pipeline        | Role in `MorselExec` plan                             | Decode ms | Process ms | Ready wait ms | Join comb/fin ms | Claims (min..max, sum) |
| ----- | --------------- | ----------------------------------------------------- | --------: | ---------: | ------------: | ---------------: | ---------------------: |
| `q8`  | `build_0`       | `part` (`ECONOMY ANODIZED STEEL`, ~0.7% / ~7k rows)   |      43.7 |        5.3 |           9.2 |          0.0/0.3 |               1..1 (8) |
| `q8`  | `build_2`       | `orders` (`1995–1996`, ~2.3M rows)                    |     374.5 |       49.6 |          21.1 |    0.1/**188.7** |             6..12 (64) |
| `q8`  | `build_3`       | `customer` (750k rows)                                |      16.5 |        2.2 |           6.4 |     0.0/**51.0** |               0..1 (6) |
| `q8`  | `probe`         | `lineitem` (30M rows, unfiltered!) → 7× Inner → `Agg` |   2,927.7 |      961.0 |          20.5 |          0.0/0.0 |           29..31 (240) |
| `q12` | `build`         | `orders` (**7.5M unfiltered rows — inverted build!**) |     320.4 |       47.7 |           8.2 |    0.0/**627.5** |             6..11 (64) |
| `q12` | `probe`         | `lineitem` (filtered to ~155k rows — `48×` smaller!)  |     572.2 |      279.6 |          10.9 |          0.0/0.0 |           27..33 (240) |
| `q10` | `build_0_build` | `customer` (750k rows)                                |     360.6 |        6.2 |          94.7 |     0.0/**83.8** |               0..1 (6) |
| `q10` | `build_0`       | `orders` (`1993-Q4`, ~285k rows) ⋈ `customer`         |     360.5 |       51.6 |          33.5 |     0.0/**32.8** |             4..13 (64) |
| `q10` | `probe`         | `lineitem` (`l_returnflag = 'R'`) → 2× Inner → `Agg`  |   2,139.1 |      691.2 |          23.7 |          0.0/0.0 |           23..38 (240) |

- **`q8` (`876.0 ms` vs. Stock `653.8 ms`, DuckDB `450.6 ms`):** `243.3 ms`
  serial `join_finalize` (`build_2` + `build_3`) explains 100% of the gap vs.
  Stock (`+222.2 ms`), while pushing `build_0`'s `0.7%` `p_partkey` dynamic
  filter into `probe` skips `~99%` of `2,927.7 ms` `lineitem` decode toward
  DuckDB (`450.6 ms`).
- **`q12` (`856.8 ms` vs. Stock `217.6 ms`, DuckDB `216.1 ms`):** Exact SF5
  amplification of the `#174` Checkpoint B finding: **`627.5 ms` (`73.2%` of
  wall!)** is single-threaded `join_finalize` on `7.5M` unfiltered `orders`
  rows because `estimate_side_size()` ignores `lineitem`'s `0.5%` filter
  (`~155k` rows). Fixing build-side selection (and parallelizing build
  finalize) projects `q12` from `856.8 ms → ~180–200 ms`, beating both Stock
  and DuckDB!
- **`q10` (`780.9 ms` vs. Stock `653.3 ms`, DuckDB `445.9 ms`):** `116.8 ms`
  serial `join_finalize` + `~194 ms` serial `AggregateSink::finalize` (~190k
  customer groups) + `94.7 ms` `ready_wait` on `customer` (only 6 RGs across 8
  workers) explain the entire `+127.6 ms` delta vs. Stock, while pushing
  `1993-Q4` `o_orderkey` (`3.8%` selectivity) into `lineitem` skips `~96%` of
  `2,139.1 ms` `probe` decode!

## 4. Prioritized SF5 Optimization Roadmap (Where Radix HTs & Dynamic Filter Pushdown Win Biggest)

| Priority | Optimization                                                             | Primary SF5 Targets                                                                                                                                                           | Quantified SF5 Wall-Time Headroom                                                                                                                                                                                                            |
| -------- | ------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **P1**   | **Parallel radix-partitioned `HashJoinBuildSink::finalize`**             | `q21` (`3,393 ms`), `q9` (`948 ms`), `q18` (`784 ms`), `q7` (`645 ms`), `q12` (`628 ms`), `q8` (`243 ms`), `q5` (`140 ms`), `q10` (`117 ms`), `q14` (`75 ms`), `q3` (`63 ms`) | Removes **`~6,300 ms`** (`~87%` of `7,258.4 ms` serial `join_finalize` across 8 cores); flips `q3`, `q5`, `q7`, `q8`, `q9` ahead of or at parity with Stock, and projects `q14` (`~295 ms`) **ahead of DuckDB** (`330.0 ms`).                |
| **P2**   | **Parallel radix-partitioned `AggregateSink::finalize` (merge)**         | `q17` (`~1,249 ms`), `q18` (`~1,116 ms`), `q20` (`~672 ms`), `q13` (`~485 ms`), `q10` (`~194 ms`), `q16` (`~129 ms`)                                                          | Removes **`~3,300+ ms`** of serial high-cardinality `GROUP BY` coordinator merge; flips `q13` (`~450 ms`) **ahead of both Stock (`471 ms`) and DuckDB (`515 ms`)**, and halves `q17`/`q18`/`q20`.                                            |
| **P3**   | **Dynamic join filter pushdown (`#170`) & selective semi-join ordering** | `q17` (`0.1%` part), `q18` (`0.004%` orders), `q8` (`0.7%` part), `q20` (`1.1%` part), `q21` (`4%` nation), `q9` (`5.5%` part), `q10` (`3.8%` orders), `q3`, `q5`, `q7`       | Unlocks **DuckDB-class scan/probe pruning** (Stock already attaches `DynamicFilter` on `20/22` queries, `16/22` with `KEY_RANGE_BITMAP`); biggest single lever to cut Morsel-vs-DuckDB from `2.24×` toward `~1.1–1.3×`.                      |
| **P4**   | **Selectivity-aware join build-side selection**                          | `q12` (`7.5M` `orders` vs. `155k` filtered `lineitem`), `q21` (`build_3`/`build_4`), `q18` (`build_0` vs. `build_1`)                                                          | Eliminates `48×` build-side inversion on `q12` (`856.8 ms → ~180–200 ms`, beating Stock `217.6 ms` and DuckDB `216.1 ms`) and avoids building multi-million-row HTs before applying highly selective semi-joins.                             |
| **P5**   | **CTE / Stage CSE (`q15`) & Parquet `Decimal128` / `RowFilter` polish**  | `q15` (duplicate `lineitem` scan = `605 MB`), `q1`, `q6`, `q19`                                                                                                               | Reusing `stage_0` across `q15`'s scalar `MAX` + supplier join (Morsel DAG already refcounts `remaining_consumers`) cuts `q15` in half (`702 → ~350 ms`); `Decimal128` dict pre-decode & fused `RowFilter` narrow `q1`/`q6`/`q19` vs. DuckDB. |

**Bottom line:** Reaching **22/22 native TPC-H coverage at SF5** with **100%
verified DuckDB parity**, **zero fallback**, **`0.0 ms` output backpressure**,
and **5 outright wins over `stock_default` (`q1`, `q6`, `q14`, `q15`, `q19`) +
1 win over DuckDB 2.0.1 (`q22`)** proves the multi-stage morsel DAG
architecture works end-to-end across every query shape in TPC-H. The remaining
`1.47×` geomean gap vs. `stock_default` (`679.7` vs. `462.6 ms`) is **not** a
parallelism or scheduling overhead — it is overwhelmingly concentrated in two
single-threaded coordinator finalizers (`7,258 ms` `join_finalize` +
`~3,850 ms` high-cardinality `aggregate_finalize` = **`~11.1 s` out of
`22.5 s`**, nearly **50% of total 22-query wall time!**). Parallelizing those
two finalizers via **radix-partitioned hash tables** plus adding **dynamic join
filter pushdown (`#170`)** and **selectivity-aware build-side selection**
projects Morsel ahead of unmodified `stock_default` across the full 22-query
SF5 suite and dramatically narrows the gap to DuckDB 2.0.1.
