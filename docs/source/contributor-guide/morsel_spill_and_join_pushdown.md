---
orphan: true
---

<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Morsel execution: dynamic join filter pushdown, radix-partitioned hash tables, and spilling

Design note for the opt-in `robtandy/datafusion:morsel-execution` experiment.
Nothing here modifies Rust code, stock DataFusion planners, or Apache DataFusion
defaults. See [morsel_pipeline_architecture.md] for the `#157` multi-pipeline
DAG and trait contracts, [morsel_parquet_adapter.md] for the buffered Parquet
morsel handoff, [morsel_parquet_splitting.md] for `#156`/`#167` intra-row-group
splitting, [morsel_metrics.md] for per-run instrumentation, and upstream
context ([apache/datafusion#21719][upstream-scheduling],
[apache/datafusion#2226][upstream-parallelism]).

## Starting point and reference

At fork revision `2aee01e23408a688828675d4c0cd489448759b33`:

- [`ParquetScan::lower`][morsel-parquet-lower] currently forces
  `with_pushdown_filters(false)` (line 148), attaches only static pruning hints
  via `FileScanConfig::try_pushdown_filters` (line 173), and starts buffered
  row-group producers via
  [`ParquetSource::create_buffered_morselizer`][parquet-create-morselizer] and
  [`BufferedReaders`][parquet-buffered].
- [morsel_pipeline_architecture.md] defines the `#157` `Pipeline` /
  `MetaPipeline` DAG, `Source` / `Operator` / `Sink` worker contracts, typed
  `ArtifactKey<T>` handoffs, `Finalized(MetaPipelineId)` and
  `Drained(PipelineId)` dependency barriers, and run-scoped `MemoryPool`
  accounting.
- The in-memory operator milestones — `#150` (Parquet `RowFilter`), `#167`
  (intra-row-group splitting), `#168` (multi-aggregate `GROUP BY`), `#158`
  (two-pipeline `HashJoin`), and `#159` (`TopN` / `Sort`) — establish the
  baseline operators that this document extends with two DuckDB-class scaling
  levers:
  1. **Dynamic join filter pushdown** (min/max bounds, small `InList` /
     `KeyRangeBitmap`, and compact split-block bloom filters) published from
     build finalization into native Parquet probe scans before probe I/O starts.
  2. **Radix-partitioned hash tables and out-of-core spilling** for `GROUP BY`,
     `HashJoin`, and `Sort`, integrating DataFusion's `MemoryPool` and
     `SpillManager` directly with the `#157` multi-pipeline DAG.

The local DuckDB reference checkout is `~/github/duckdb-morsel-reference/`
(`REFERENCE_COMMIT`: `9c1e4c5233af5953e6b35e88fbef5089eaace6e9`):

- `src/execution/operator/join/physical_join.cpp` (`PhysicalJoin::BuildJoinPipelines`,
  line 31; `EmptyResultIfRHSIsEmpty`, line 14; `IsSource()` ->
  `meta_pipeline.CreateChildPipeline`, lines 74–78) shows how a hash join builds
  its RHS in a child `MetaPipeline`, probes on the LHS pipeline, and attaches a
  child pipeline in the same `MetaPipeline` when the join must also act as a
  source (for `RIGHT`/`FULL` unmatched scans or out-of-core spilled partitions).
- `src/include/duckdb/parallel/meta_pipeline.hpp` (lines 26–42, 85–102),
  `src/execution/physical_operator.cpp` (`BuildPipelines`, line 245; `Combine` /
  `PrepareFinalize` / `Finalize`, lines 183–192), and
  `src/parallel/pipeline_executor.cpp` (`FlushAndFinalize` / `PushFinalize`,
  lines 649–716) define the pipeline barrier model reused here.
- DuckDB's `RadixPartitionedHashTable`, `JoinHashTable` eviction, and
  `ProbeSpill` storage internals are outside the trimmed workspace snapshot; the
  design below adapts those algorithms directly to Arrow, `RowConverter`,
  `MemoryPool`, and `SpillManager` rather than mirroring C++ classes.

## 1. Dynamic join filter pushdown into native Parquet probe scans

### 1.1 Build-side summary collection and DAG-ordered publication

In stock [`HashJoinExec`][stock-join-allow], the build and probe streams can be
polled concurrently, so pushdown requires a mutable
[`DynamicFilterPhysicalExpr`][dyn-filter] coordinated by
[`SharedBoundsAccumulator`][shared-bounds], `wait_complete()`, and a per-row-group
[`RowGroupPruner`][rg-pruner] watching `DynamicFilterTracking::Watching`
transitions.

In the `#157` morsel DAG, `P_probe` already declares `requires: [Finalized(G_build)]`.
The scheduler does not call `Source::open` on `P_probe`'s `ParquetScan` until
every build worker has drained, combined, and finalized `G_build`. Consequently:

1. The join filter summary is **100% immutable and complete before any probe
   Parquet footer, page index, bloom filter, or data byte range is fetched**.
2. `ParquetScan::open` on `P_probe` binds the finalized filter once from an
   `ArtifactKey<JoinBuildArtifact>` and clones a per-run `ParquetSource` with
   the pushed predicate attached. No mutable filter state lives on `MorselExec`,
   so repeated or concurrent runs of the same prepared plan stay completely
   isolated.
3. Because the filter is already complete at file-open time, DataFusion's
   existing file-open pruning stack runs at full fidelity without per-RG watch
   overhead (`RowGroupPruner` is unnecessary for join filters, though wrapping
   an already-complete `DynamicFilterPhysicalExpr` via `update()` +
   `mark_complete()` remains compatible with `RowGroupPruner` if a future
   streaming TopK threshold shares the same scan).

#### Three-tier filter summary collected at negligible build overhead

During build `sink` / `combine` / `finalize`, the join build already evaluates
every build key column and hashes each key row with
[`create_hashes`][create-hashes] (`foldhash::fast::FixedState`). Collecting the
pushed filter piggybacks on that work:

```rust
struct LocalJoinFilterSummary {
    /// Per-key min/max over non-null build values (same semantics as `ColumnBounds`).
    bounds: Vec<Option<ColumnBounds>>,
    /// Per-key flag: true if any build row had NULL in key column `i`.
    key_has_null: Vec<bool>,
    /// True if any build row was observed (even if all keys were NULL).
    has_rows: bool,
    /// True if at least one build row had matchable (non-dropped) keys.
    has_matchable_keys: bool,
}

enum JoinMembershipFilter {
    /// No matchable build keys (empty build or all-NULL keys under NullEqualsNothing).
    Empty,
    /// Small build: exact key array (single key or StructArray) within
    /// `hash_join_inlist_pushdown_max_size` and `hash_join_inlist_pushdown_max_distinct_values`.
    InList(ArrayRef),
    /// Medium/large build: cache-resident split-block Bloom filter built over
    /// the join's own 64-bit `foldhash` values, plus optional `KeyRangeBitmap`
    /// for single-column integer/date container pruning.
    Bloom {
        filter: Arc<SplitBlockBloomFilter>,
        seed: SeededRandomState,
        pruning_bitmap: Option<Arc<KeyRangeBitmap>>,
    },
    /// Filter disabled (memory budget tight, unsupported key type, or oversized NDV).
    None,
}

struct FinalizedJoinFilter {
    bounds: Option<PartitionBounds>,
    membership: JoinMembershipFilter,
    key_has_null: Vec<bool>,
    has_rows: bool,
}
```

How each tier is constructed:

1. **Per-key min/max bounds (`sink` + `combine`):** For each key column with a
   total order (integers, dates, timestamps, decimals, finite floats, strings,
   binary), each worker updates `min` and `max` over non-null values in a single
   vectorized pass (`arrow::compute::min` / `max` per batch) and records
   `key_has_null[i] |= array.null_count() > 0`. `combine` merges worker bounds
   using [`PartitionBounds::union`][partition-bounds] semantics (a `NULL` bound
   means that worker saw no non-null keys and never widens another worker's
   range). Build overhead is a few nanoseconds per row, and bounds work for
   **every** build size — including spilled builds (Section 3.3), because bounds
   are updated _before_ any partition evicts to disk.
2. **Small-build `InList` + optional `KeyRangeBitmap` (`finalize`):** Reuse
   DataFusion's existing optimizer thresholds:
   `optimizer.hash_join_inlist_pushdown_max_size` (default `128 KiB`) and
   `optimizer.hash_join_inlist_pushdown_max_distinct_values` (default `150`,
   [`config.rs`][config-inlist]). When the distinct build keys fit both limits,
   emit an `InListExpr::try_new_from_array` (single column, or `StructArray`
   over `struct(...)` for multi-column keys, matching
   [`shared_bounds.rs`][shared-bounds-membership]). Small `InList` predicates
   have a major advantage over bloom filters at container level: DataFusion's
   `PruningPredicate` evaluates `IN (...)` directly against Parquet row-group
   min/max stats, page indexes, **and** Parquet file-stored SBBF bloom filters
   (`prune_by_bloom_filters`). For larger single-column integer/date builds
   with sparse key ranges, optionally build DataFusion's existing
   [`KeyRangeBitmap`][key-range-bitmap] (capped at `128 KiB`,
   `MAX_BUCKETS = 1 << 20`, with container pruning already implemented by
   [`KeyRangeBitmapPruningExpr`][key-range-pruning]).
3. **Cache-resident split-block Bloom filter (`finalize`):** When the build
   exceeds the `InList` threshold (for example, 5K–5M distinct keys), build a
   single-cache-line split-block bloom filter (`256-bit` or `512-bit` blocks,
   8–12 bits per distinct key, targeting ~1%–2% FPP, capped at a configurable
   ceiling such as `4 MiB`–`8 MiB` and reserved in `MemoryPool`):
   - Size the bit array in `finalize` once the exact build row/distinct count
     is known (avoiding pre-sizing guess errors during `sink`).
   - Populate it directly from the 64-bit `foldhash` values already retained
     per radix partition (Section 2.2), using parallel bitwise-OR or
     partition-local shard inserts if the filter uses power-of-two blocks.
   - Wrap it in a private `JoinBloomFilterExpr: PhysicalExpr` (analogous to
     [`HashTableLookupExpr`][hash-table-lookup] in `partitioned_hash_eval.rs`,
     but probing a compact L2/L3-resident bitset instead of the full build hash
     table). In `evaluate()`, compute `create_hashes` over the probe key
     column(s) with the exact same `SeededRandomState`, check 4–8 bits inside
     one cache line per row (`block = (h >> 32) & mask`), and return a
     `BooleanArray`.

#### Publication into `ParquetScan::open`

When `ParquetScan` is lowered as the source of a probe pipeline (or an upstream
scan feeding a `Filter -> Projection -> HashProbe` pipeline with no blocking
operator in between), the planner records a `Vec<JoinFilterBinding>` on the
`ParquetScan` spec mapping each upstream `ArtifactKey<JoinBuildArtifact>` to the
probe key column expression(s) in the scan's projected coordinates (remapped
through any intervening projections/aliases).

At `Source::open`, before calling `create_buffered_morselizer`:

1. Resolve each bound `JoinBuildArtifact` (guaranteed committed by
   `Finalized(G_build)`).
2. If any binding has `FinalizedJoinFilter::short_circuit_probe() == true`
   (Section 1.3), open an empty `ReadyReaders` immediately — zero Parquet
   footer reads, zero data range fetches, zero worker claims.
3. Otherwise, build the combined dynamic predicate `P_dyn = bounds AND membership`
   (with NULL escape if required), combine it with any static scan predicates on
   a per-run clone of `FileScanConfig` / `ParquetSource`, and pass it through
   `FileScanConfig::try_pushdown_filters` (or `ParquetSource::with_predicate`
   after table-schema column remapping, keeping `ParquetScan`'s immutable spec
   untouched).
4. Construct the `Morselizer` from the per-run `ParquetSource` and spawn the
   producer lanes.

### 1.2 Interaction with `#150`, `#167`, and star-schema joins (TPC-H Q3 & Q12)

#### Interaction with `#150` (`execution.parquet.pushdown_filters = true`)

When `#150` enables Parquet `RowFilter` on the decode workers:

- [`prebuild_row_filter_candidates`][row-filter-prebuild] splits the combined
  predicate into conjuncts (`static_predicates`, per-key `col >= min AND col <= max`,
  and `InListExpr` / `JoinBloomFilterExpr`) and
  [`row_filter_from_prebuilt`][row-filter-from-prebuilt] sorts them by
  `required_bytes`.
- Because a join key column (typically `Int32`/`Int64`, 4–8 bytes) is narrow,
  the decoder evaluates `bounds` and `JoinBloomFilterExpr` on **only the join
  key column(s)** on the pipeline workers, skipping decoding of all wide
  non-key payload columns for rejected rows (late materialization).
- When `pushdown_filters = false`, `ParquetSource` still uses `bounds`,
  `InListExpr`, and `KeyRangeBitmap` during file open for:
  - row-group min/max statistics pruning
    ([`prune_by_statistics_with_metadata`][opener-rg-stats]),
  - Parquet file bloom-filter pruning
    ([`prune_by_bloom_filters`][opener-rg-bloom]), and
  - page-index pruning
    ([`prune_plan_with_page_index_and_metrics`][opener-page-prune]).

  `JoinBloomFilterExpr` is omitted from `RowFilter` when `pushdown_filters = false`
  (preserving the user's explicit setting); if desired, it can still run as a
  lightweight selection-producing operator right before `HashProbe` to skip
  random hash-table misses when `HashProbe` uses a large table.

- Because `JoinBloomFilterExpr` is advisory (has false positives, zero false
  negatives), **never drop the downstream `HashProbe` key equality check** and
  never mark a pushed bloom filter as `PushedDown::Yes` for removing any static
  residual `Filter`.

#### Interaction with `#167` (intra-row-group morsel splitting)

In [`ParquetOpener`][opener-page-prune] and [morsel_parquet_splitting.md],
file-open pruning runs before sub-row-group slicing:

```text
metadata -> prune_by_statistics_with_metadata -> prune_by_bloom_filters
         -> prune_plan_with_page_index_and_metrics -> ParquetAccessPlan
         -> (#167) intersect [lo, hi) slices with RowSelection & omit empty slices
         -> per-slice ParquetPushDecoder (range broker) + (#150) RowFilter on workers
```

Because `FinalizedJoinFilter` is installed **before** `ParquetAccessPlan` is
constructed:

- Entire row groups outside `[min, max]` (or excluded by `KeyRangeBitmap` /
  file bloom filters) are removed before `#167` evaluates split eligibility or
  initializes a row-group-local range broker.
- Surviving row groups have non-matching data pages pruned by the Page Index
  into a row-group-local `RowSelection` _before_ `#167` intersects each
  128 Ki-row `[lo, hi)` slice with that selection and drops empty
  intersections, so workers never claim empty sub-ranges or fetch pruned page
  byte ranges through the range broker.
- Each per-slice `ParquetPushDecoder` (`with_row_group_selections`) inherits
  the same immutable `RowFilterContext` (`PrebuiltRowFilterCandidate`s with the
  bound `JoinBloomFilterExpr`), so all workers evaluate late-materialized join
  filters in parallel with zero synchronization.

#### Impact on star-schema joins (TPC-H Q3 and Q12)

- **TPC-H Q3 (`customer ⋈ orders ⋈ lineitem`, `GROUP BY`, `ORDER BY ... LIMIT 10`):**

  ```text
  P0: ParquetScan(customer, c_mktsegment='BUILDING')
        -> HashBuildSink(J1: c_custkey)                           [G0]
  P1: ParquetScan(orders, o_orderdate < '1995-03-15'
                          AND J1_filter(o_custkey))
        -> HashProbe(J1) -> HashBuildSink(J2: o_orderkey)         [G1]  req Finalized(G0)
  P2: ParquetScan(lineitem, l_shipdate > '1995-03-15'
                            AND J2_filter(l_orderkey))
        -> HashProbe(J2) -> RadixGroupBy                          [G2]  req Finalized(G1)
  P3: FinalizedGroups -> TopNSink(LIMIT 10)                       [G3]  req Finalized(G2)
  P4: FinalizedTopN -> Result                                     [G4]  req Finalized(G3)
  ```

  1. `P0` filters `customer` to ~20% of rows (~30K keys at SF1) and publishes a
     ~45 KiB `J1` bloom filter + `[min, max]` on `o_custkey`.
  2. `P1` scans `orders` with `o_orderdate < '1995-03-15'` AND `J1_filter(o_custkey)`.
     With `#150`, ~80% of date-passing `orders` rows are rejected by the
     L2-resident `J1` bloom filter before decoding `o_shippriority` or probing
     `J1`, leaving ~147K surviving `(o_orderkey, o_orderdate, o_shippriority)`
     build rows for `J2` (~9.8% of `orders`).
  3. `P2` receives `J2`'s `l_orderkey` min/max bounds (pruning `lineitem` row
     groups/pages whose `l_orderkey` range falls outside the pre-1995-03-15
     orders, since `orders` and `lineitem` are naturally correlated by
     `orderkey` in TPC-H!) **plus** a ~220 KiB L2-resident `J2` bloom filter.
     On surviving `lineitem` row groups, `RowFilter` rejects ~90%+ of rows
     using only `l_shipdate` and `l_orderkey` without ever decoding
     `l_extendedprice` or `l_discount`.

- **TPC-H Q12 (`orders ⋈ lineitem`, `GROUP BY l_shipmode`, `ORDER BY l_shipmode`):**
  - `lineitem` has selective predicates (`l_shipmode IN ('MAIL', 'SHIP')`,
    `l_commitdate < l_receiptdate`, `l_shipdate < l_commitdate`,
    `l_receiptdate ∈ [1994-01-01, 1995-01-01)`), reducing 6M SF1 rows to ~31K
    rows (~0.5%), whereas `orders` has 1.5M unfiltered rows.
  - Putting the filtered `lineitem` on the build side (`P0`) produces a tiny
    ~48 KiB `o_orderkey` bloom filter + tight `[min, max]` bounds pushed into
    `ParquetScan(orders)` (`P1`). Even when `o_orderkey` spans most of `orders`,
    the L2-resident bloom filter rejects ~98% of `orders` rows inside `RowFilter`
    before decoding `o_orderpriority` (a string column!) and before probing `J1`.

### 1.3 Exactness, NULL semantics, and safety matrix

Dynamic join filters are **advisory semi-join reductions**: false positives are
harmless (caught by `HashProbe`), whereas a single false negative is a silent
wrong-result bug. Follow DataFusion's exact semantics in
[`JoinType::on_lr_is_preserved`][join-type-preserved],
[`empty_build_side_produces_empty_result`][join-type-empty-build],
[`empty_map_produces_empty_result`][join-type-empty-map],
[`HashJoinExec::allow_join_dynamic_filter_pushdown`][stock-join-allow], and
[`SharedBoundsAccumulator::preserve_probe_nulls`][preserve-probe-nulls].

Convention below: **build = left**, **probe = right** (matching DataFusion's
`HashJoinExec`; swap labels if the planner chooses right-as-build).

| Join type (build=L, probe=R)                 | Push filter into probe (`R`)?                                                | Empty build (`!has_rows`) short-circuits probe?  | Non-empty all-NULL build (`has_rows && !has_matchable_keys`, `NullEqualsNothing`) |
| -------------------------------------------- | ---------------------------------------------------------------------------- | ------------------------------------------------ | --------------------------------------------------------------------------------- |
| `Inner`                                      | **Yes**                                                                      | **Yes** (`lit(false)`)                           | Short-circuit probe (`lit(false)`); 0 output rows                                 |
| `Left` (build preserved)                     | **Yes**                                                                      | **Yes** (`lit(false)`)                           | Short-circuit probe (`lit(false)`); emit unmatched build rows via child pipeline  |
| `LeftSemi`, `RightSemi`                      | **Yes**                                                                      | **Yes** (`lit(false)`)                           | Short-circuit probe (`lit(false)`); 0 output rows                                 |
| `LeftAnti`, `LeftMark` (regular)             | **Yes**                                                                      | **Yes** (`lit(false)`)                           | Short-circuit probe (`lit(false)`); unmatched build rows still emitted            |
| `LeftAnti`, `LeftMark` (null-aware `NOT IN`) | **Only if build keys are NOT NULL** (see [`exec.rs:1084`][stock-join-allow]) | **No** (probe emptiness affects NULL build rows) | **Disable pushdown**; must scan probe                                             |
| `Right`, `Full` (probe preserved)            | **No** (unmatched probe rows must survive)                                   | **No**                                           | **No**                                                                            |
| `RightAnti`, `RightMark`                     | **No** (unmatched probe rows are the output)                                 | **No**                                           | **No**                                                                            |

Additional soundness invariants:

- **`NullEquality::NullEqualsNothing` (SQL `=`):** Build rows with `NULL` in
  _any_ join key column cannot match any probe row; exclude them from min/max
  bounds, `InList`, and the bloom filter. Probe rows with `NULL` keys evaluate
  to `NULL`/false under `col >= min AND col <= max` and `JoinBloomFilterExpr`
  and are safely dropped.
- **`NullEquality::NullEqualsNull` (`IS NOT DISTINCT FROM`):**
  - **Single-key join:** If `key_has_null[0]` is true and the probe key is
    nullable, wrap the pushed predicate as `(bounds AND membership) OR probe_key IS NULL`
    (matching [`preserve_probe_nulls`][preserve-probe-nulls]). If the build had
    no `NULL`s, no `IS NULL` escape is needed.
  - **Multi-key join `(k0, k1, ...)`:** A build tuple `(NULL, 5)` can match a
    probe tuple `(NULL, 5)`. Per-key bounds on `k_i` must use
    `(k_i >= min_i AND k_i <= max_i) OR k_i IS NULL` whenever `key_has_null[i]`
    is true (or omit the range bound on `k_i` if all build values of `k_i` are
    `NULL`). The composite bloom filter hashes `(k0, k1, ...)` with
    `create_hashes` (which hashes `NULL`s deterministically!) so `(NULL, 5)` is
    inserted into and probed from the bloom filter **without** needing a wide
    `OR k0 IS NULL` escape that would let `(NULL, 999)` bypass the bloom filter.
- **Multi-key joins under `NullEqualsNothing`:** Per-key min/max bounds are
  conjunctive (`(k0 BETWEEN min0 AND max0) AND (k1 BETWEEN min1 AND max1)`),
  which is sound because matching requires every key to match simultaneously.
  The bloom filter hashes the full composite tuple `(k0, k1, ...)` with
  `create_hashes`, preserving full composite selectivity.
- **Floating-point `NaN` and `-0.0`:** Normalize `-0.0` to `+0.0` in both hash
  computation (matching `float_semantics` in `create_hashes`) and min/max
  bounds. If any build key is `NaN`: under `NullEqualsNothing`, disable min/max
  bounds on that key column (or exclude `NaN` since `NaN != NaN` in SQL
  equality, while under `NullEqualsNull` or Arrow total-order semantics disable
  bounds on that column to stay conservatively sound).
- **Expression keys & schema coercion:** Only push filters whose probe-side
  expressions pass
  [`can_expr_be_pushed_down_with_schemas`][can-push-down] against the scan's
  non-virtual schema (no volatile functions, no missing columns). If a probe
  key involves an implicit cast (e.g., `Int32 -> Int64`), push the cast
  expression in the `RowFilter` / `PruningPredicate` using DataFusion's normal
  `PhysicalExprAdapter` path, or skip pushdown for that key if unsupported.
- **Parquet truncated string/binary statistics:** Handled automatically by
  `RowGroupAccessPlanFilter::prune_by_statistics_with_metadata` when evaluating
  `col >= min AND col <= max`.
- **Configuration & strict mode:** Gate pushdown on
  `config.optimizer.enable_join_dynamic_filter_pushdown` (default `true`,
  controlled also by `enable_dynamic_filter_pushdown`). Skipping dynamic
  pushdown for an ineligible join type or unsupported key type is an
  optimization opt-out, never a `morsel_mode = required` planning failure.

## 2. Radix-partitioned hash tables for L2 cache locality & parallel finalize

### 2.1 Why monolithic hash tables stall beyond L2/L3 cache

In-memory v1 `GROUP BY` (`#168`) and `HashJoin` (`#158`) build worker-local
maps and merge them in `combine` / `finalize`. Once distinct group cardinality
or build-side size grows beyond per-core L2 (~256 KiB–1 MiB) and shared L3
(~16–64 MiB):

1. **Random DRAM latency on insert/probe:** A 256 MiB hash table misses L3 and
   dTLB on almost every lookup (~60–100 ns/probe), turning vectorized 8-core
   execution memory-latency bound instead of compute bound.
2. **Serial Amdahl bottleneck in `combine`/`finalize`:** Merging `W` large
   worker-local tables into one global table on the coordinator thread
   serializes `O(W * N)` hashing/equality work and creates a `2×` peak memory
   spike while all worker locals and the global table coexist.
3. **No natural unit of eviction:** A monolithic hash table cannot evict half
   its rows cleanly when `MemoryReservation::try_grow` fails; it must either
   drain-and-sort everything (stock `AggregateSpill`) or OOM (stock
   `HashJoinExec`).

### 2.2 Radix partitioning in the `#157` trait lifecycle

Partitioning keys by `R` hash bits into `P = 2^R` disjoint partitions solves
all three problems at once:

- **L2-resident build/finalize:** With `R = 6` (`64` partitions) or `R = 8`
  (`256` partitions), a 64 MiB build/group table splits into `256 KiB`–`1 MiB`
  partitions that fit entirely inside each core's private L2 cache.
- **O(`2^R`) lock-free `combine`:** During `sink`, each worker scatters tuples
  into its own `2^R` partition chunk lists. `combine` merely appends chunk
  `Vec`s (and transfers `MemoryReservation`s) per partition ID — zero hashing,
  zero key comparisons, zero shared locks.
- **Parallel lock-free `finalize`:** As sketched in
  [morsel_pipeline_architecture.md], a single logical `Sink::finalize` spawns
  bounded tracked tasks on a `JoinSet`, assigning disjoint radix partition IDs
  `p ∈ 0..2^R` to workers. Each worker builds/merges partition `p` completely
  inside L2 cache and joins back before the supervisor commits one immutable
  `Arc<RadixArtifact>`.

#### Critical Rust / `hashbrown` hash-bit layout invariant

[`create_hashes`][create-hashes] computes 64-bit hashes using
`foldhash::fast::FixedState` ([`hash_utils.rs:30`][hash-random-state]).
`hashbrown::HashTable` (SwissTable) extracts its 7-bit SIMD control-byte tag
(`h2`) from the **top 7 bits** (`h >> 57`) on 64-bit targets.

**Hazard:** If radix partitioning also selects the top `R` bits
(`p = (h >> (64 - R)) as usize`) and passes the unmodified `h` into a
per-partition `hashbrown::HashTable`, every key inside partition `p` has the
same top `R` bits of `h2` — collapsing SwissTable's 128-tag entropy to
`2^(7 - R)` tags (just **1 tag** when `R >= 7`!), causing severe false-positive
tag matches inside every 16-slot SIMD group.

**Required fix:** Never share the same bits between radix partition selection
and `hashbrown`'s `h2` tag:

```rust
/// Extract partition index from top `radix_bits` of `h` (0..=12), and return a
/// rotated hash whose top 7 bits come from unused mid-bits for `hashbrown` `h2`.
#[inline]
fn split_radix_hash(h: u64, radix_bits: u32) -> (usize, u64) {
    if radix_bits == 0 {
        return (0, h);
    }
    let part = (h >> (64 - radix_bits)) as usize;
    let local_hash = h.rotate_left(radix_bits);
    (part, local_hash)
}
```

Rotating left by `radix_bits` moves the next 7 independent bits into `[57..64]`
for `hashbrown`'s `h2` while preserving full 64-bit entropy for low-bit bucket
indexing (`h1`) and recursive re-partitioning (`radix_bits_level_2` simply
rotates again!). For custom join tables using a 16-bit hash salt array, store
`(local_hash >> 48) as u16` for the same reason.

#### Operator-specific adaptive partitioning

1. **`GROUP BY` (`RadixGroupedAggregateSink`):**
   - **Small/low-cardinality fast path (e.g., TPC-H Q1 with 4 groups):** Every
     worker starts with a small L2-sized local pre-aggregation table (capacity
     e.g. 2,048–4,096 groups / ~256 KiB). If input finishes without exceeding
     that threshold, `combine` merges the small worker tables directly with
     `R = 0` — zero partitioning overhead and zero regression on low-cardinality
     queries.
   - **High-cardinality / radix mode:** Once a worker's local pre-aggregation
     table fills and exhibits low reduction (e.g., `groups / input_rows > 0.25`
     or capacity crosses `L2_PREAGG_LIMIT`), flush the partial group states
     (keys + intermediate accumulator states) into `2^R` (`R = 6` or `8`)
     worker-local radix partition buffers and keep using the small L2 table as
     a heavy-hitter combining cache (flushing on fill), **or** scatter directly
     when reduction stays near `1.0`. During parallel `finalize`, each task
     merges all workers' chunks for partition `p` in an L2-sized table and
     emits finalized per-partition `RecordBatch`es (or holds immutable
     per-partition state for `FinalizedGroups` workers to claim disjointly).

2. **`HashJoin` (`RadixHashBuildSink`):**
   - **Tiny build fast path (`R = 0`):** Workers buffer incoming build batches
     locally up to a tiny threshold (e.g., 4,096 rows); if total build rows stay
     small, finalize into a single unpartitioned table (zero radix dispatch
     overhead on small dimension joins).
   - **Radix build (`R = 6..8`):** Workers compute `create_hashes` on build
     keys once, update `LocalJoinFilterSummary`, and scatter rows + `u64` hashes
     into `2^R` per-worker partition buffers. `combine` concatenates partition
     chunk lists in O(`W * 2^R`). Parallel `finalize` assigns disjoint
     partitions `p ∈ 0..2^R` to bounded tracked tasks; each task builds an
     immutable `PartitionJoinTable` (and sets bits in the build bloom filter).
     The committed `Arc<RadixJoinTable>` exposes lock-free probe lookup:
     `let (p, lh) = split_radix_hash(h, table.radix_bits); table.partitions[p].probe(lh, ...)`.

### 2.3 Concrete Rust data layouts over Arrow / `RowConverter` + `MemoryPool`

Two candidate physical layouts inside each radix partition:

| Dimension                 | Layout A: Hybrid specialized + `RowConverter` columnar chunks (**Recommended v1**)                                                                                                                                                                       | Layout B: Packed row-store blocks (DuckDB `TupleDataCollection` analogy)                                                             |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **Group keys**            | Single fixed-width key (`Int32`/`Int64`/`UInt64`/`Date32`): dense `Vec<T>` + `hashbrown::HashTable<(u64, u32)>` (as in `GroupValuesPrimitive`). Multi-column / string / decimal: [`arrow::row::RowConverter`][group-values-rows] (`Rows` per partition). | Custom variable-width row arena (`Vec<AlignedBlock>`) storing key bytes + inline fixed accumulator state.                            |
| **Aggregate states**      | Columnar typed vectors per partition (`Vec<i64>`, `Vec<i128>`, `Vec<f64>`, `NullBufferBuilder`) indexed by dense `group_id: u32`.                                                                                                                        | Inline fixed-size structs per row; variable states (`String` min/max) stored in side heap.                                           |
| **Join build storage**    | Per-partition coalesced `RecordBatch` (or `Vec<RecordBatch>`) + per-partition [`JoinHashMapU32`][join-hash-map] (`head` table + `next: Vec<u32>`, since each of `2^R` partitions fits in `u32`) + optional `Vec<u16>` hash salts.                        | Packed build rows `(next_ptr, hash_salt, key_bytes, payload_bytes)` in 256 KiB blocks.                                               |
| **Arrow / Spill interop** | Zero-copy or single-copy conversion to `RecordBatch` for `MorselExec` output and `SpillManager` IPC serialization; reuses Arrow kernels and `gc_view_arrays`.                                                                                            | Requires custom per-`DataType` serializer/deserializer to and from Arrow `RecordBatch` and `SpillManager`.                           |
| **Cache locality**        | High: every partition's hash table, `Rows` buffer, and columnar state vectors fit together inside L2 during `finalize` and `probe`.                                                                                                                      | Highest for wide multi-column probes/aggregates (single cache miss per row instead of 1 per column), at much higher code complexity. |

**Recommendation:** Adopt **Layout A** for the first radix + spill milestones.
It keeps Arrow `RecordBatch` and `RowConverter` as the universal data model
across `GROUP BY`, `HashJoin`, and `SpillManager`, preserves `#140`/`#168`'s
fast specialized primitive key path, and unlocks 90% of the L2 + parallel
finalize win without maintaining a bespoke multi-type row serialization engine.
Revisit Layout B only if post-radix profiles show columnar gather latency
dominating wide-payload probes.

#### `MemoryPool` accounting discipline

- Register one run-local [`MemoryConsumer`][mem-consumer] per blocking sink
  pipeline with `.with_can_spill(true)` (once spill is enabled for that
  operator; `.with_can_spill(false)` before spill lands so `FairSpillPool`
  classifies it accurately).
- Every worker registers its own `MemoryReservation` (`consumer.register(&pool)`)
  for `LocalSinkState` and calls [`try_grow`][mem-try-grow] **before** growing
  partition chunk buffers, `Rows` arenas, `HashTable` allocations, or
  accumulator vectors.
- On `combine`, transfer reservations using [`MemoryReservation::take`][mem-take]
  / `split` into per-partition reservations inside `GlobalSinkState` — zero
  unreserved window and zero double-counting.
- During parallel `finalize`, each partition task receives ownership of
  `partition_reservation`. When compacting partition chunks into a single
  partition `RecordBatch` + `JoinHashMapU32`, reserve the transient peak while
  inputs and compacted output coexist, then `shrink` immediately after dropping
  the uncompacted input chunks.
- Never charge cloned `Arc` references to already-accounted Arrow buffers; keep
  the reservation attached to the `Arc<RadixArtifact>` (or backing-batch lease)
  so dropping the last consumer releases the reservation automatically.

## 3. Out-of-core spilling for `GROUP BY`, `HashJoin`, and `Sort`

### 3.1 Integrating `SpillManager` and `MemoryPool` with the `#157` DAG

DataFusion already provides a battle-tested spill substrate under
`datafusion/physical-plan/src/spill/` and `datafusion/execution/`:

- [`SpillManager`][spill-manager]: creates temporary spill files via
  `RuntimeEnv::disk_manager`, writes Arrow IPC streams with optional
  `execution.spill_compression` (`Uncompressed` / `Lz4Frame` / `Zstd`), updates
  `SpillMetrics` (`spill_file_count`, `spilled_bytes`, `spilled_rows`), and
  reads files back via
  [`read_spill_as_stream_unbuffered`][spill-read-unbuffered] (using
  [`SpillReaderStream`][spill-reader-stream] with exact-size `MessageFramer`
  buffering over `tokio::fs::File` so concurrent reads do not pin blocking OS
  threads).
- [`InProgressSpillFile`][in-progress-spill]: supports incremental
  `append_batch` / [`append_batch_async`][in-progress-async], automatic
  [`gc_view_arrays`][gc-view-arrays] before serialization (preventing sliced
  `StringView`/`BinaryView` buffers from amplifying spill size), `finish` /
  `finish_async`, and [`abort_async`][in-progress-abort].
- [`SpillFile`][spill-file] and [`RefCountedTempFile`][refcounted-tmp]:
  local temp files are unlinked and subtracted from
  `DiskManager::used_disk_space` / `active_files_count` automatically when the
  last `Arc<dyn SpillFile>` drops ([`disk_manager.rs:437`][refcounted-drop]).
- **Crate visibility note:** `datafusion_physical_plan::spill::SpillManager` is
  currently re-exported with `#[doc(hidden)]`, while `in_progress_spill_file`
  is `pub(crate)` inside `datafusion-physical-plan`. When morsel spill code in
  `datafusion/core/src/morsel/` uses incremental spill files, expose a minimal
  `#[doc(hidden)]` re-export of `InProgressSpillFile` (following DataFusion's
  [internal public APIs](api-health.md#internal-public-apis) policy) rather than
  duplicating IPC framing or `gc_view_arrays`.

#### Shared runtime invariants across all spilling operators

1. **Spill-buffer headroom:** Before entering the hot loop, reserve a small
   dedicated headroom (analogous to `execution.sort_spill_reservation_bytes`,
   default `10 MiB` in `ExternalSorter`, clamped proportionally when the pool
   budget is tiny). When a `try_grow` fails during `sink` or `probe`, the
   operator already holds the scratch budget needed for `gc_view_arrays` and
   IPC framing so spilling itself does not immediately fail with OOM.
2. **Async spill I/O without holding locks:** Call `append_batch_async` /
   `finish_async` on worker tasks without holding any mutex or supervisor
   state gate across `.await`. Keep `InProgressSpillFile`s strictly worker-local
   during `sink` and `probe`; only immutable `Arc<dyn SpillFile>` handles cross
   into `GlobalSinkState` or run artifacts.
3. **Bounded open file descriptors:** With `W = 8` workers and `P = 64` radix
   partitions, keeping `W * P = 512` simultaneously open `InProgressSpillFile`s
   is wasteful and can hit OS `RLIMIT_NOFILE`. Instead, buffer small batches
   per partition in memory up to a flush threshold (e.g., `128 KiB`–`512 KiB`)
   and either:
   - flush the **largest** buffered partition(s) to a finished `SpillFile` via
     `spill_record_batch_and_finish` on `try_grow` failure (0 persistently open
     write FDs per worker!), or
   - group `P` logical partitions into `G <= 16` physical spill stripes when
     streaming probe spills and rotate/close idle writers promptly.
4. **Spill metrics in `#138`:** Attach `SpillMetrics` with a `pipeline` label
   to `RunMetrics` when `MorselMetrics` is enabled, keeping spill counts/bytes
   and spill I/O wait time visible alongside decode, process, and combine/finalize
   time.

### 3.2 Out-of-core `GROUP BY`: radix partition spill

Stock DataFusion's [`AggregateSpill`][stock-agg-spill] handles OOM by draining
the hash table, sorting every buffered group by the full group key, spilling
sorted IPC runs, and replaying through `SortPreservingMergeStream` +
`ClusteredFinalAggregateStream`.

With radix partitioning (Section 2.2), the morsel engine gets a faster,
sort-free common path modeled on DuckDB's `RadixPartitionedHashTable`:

```text
P0: Scan -> [Filter/Proj] -> RadixGroupedAggregateSink (spill on try_grow err) [G0]
P1: FinalizedOrSpilledGroupsSource -> [Proj/TopN/Result]                       [G1]  req Finalized(G0)
```

1. **Triggering spill in `P0::sink`:** When `local_reservation.try_grow` fails:
   - Drain the worker's in-memory pre-aggregation table / in-memory radix
     partition buffers into intermediate-state `RecordBatch`es (canonical
     `spill_schema`: group key columns + intermediate accumulator state columns,
     matching `AggregateMode::Partial` state types).
   - Scatter rows by radix partition `p ∈ 0..2^R` (if not already partitioned)
     and write each sufficiently full partition's batches via `SpillManager`
     into `Vec<Arc<dyn SpillFile>>` keyed by partition `p`.
   - `shrink` the freed memory reservation and continue consuming input morsels.
2. **Combining in `G0::combine`:** Merge each worker's per-partition
   `(in_memory_chunks, Vec<Arc<dyn SpillFile>>)` into `GlobalSinkState::partitions[p]`.
   No disk I/O occurs during `combine`.
3. **Finalizing and streaming in `G0::finalize` -> `P1`:**
   - **Zero-spill path:** `finalize` merges all `2^R` partitions in parallel
     into memory and commits an immutable `InMemoryGroups` artifact.
   - **Spilled path:** `finalize` does **not** load all `2^R` partitions into
     memory simultaneously (which would OOM again). Instead, it flushes any
     remaining large in-memory partition chunks if needed, commits a
     `SpilledGroupsArtifact { partitions: Arc<[PartitionSpillInput]>, spill_manager, next_partition: AtomicUsize }`,
     and lets `P1` workers claim **one radix partition `p` at a time** (with
     concurrency bounded by available `MemoryPool` budget):
     1. Read all in-memory chunks + spill files for partition `p` via
        `SpillManager::read_spill_as_stream_unbuffered`.
     2. Merge partial states into a partition-local hash table (`try_grow`
        checked).
     3. Emit finalized group batches downstream through `P1`'s operators/sink.
     4. Immediately drop partition `p`'s hash table **and** its
        `Vec<Arc<dyn SpillFile>>` handles so memory and disk space are freed
        before claiming partition `p + 1`.
4. **Single-partition overflow and skew fallback:** If reloading a single
   partition `p` still fails `try_grow`:
   - **Recursive radix split (up to `MAX_RADIX_DEPTH = 2`):** Re-partition `p`'s
     chunks + spill files using the next `R'` unused hash bits
     (`h.rotate_left(R)`), spill the sub-partitions, and process each
     sub-partition sequentially.
   - **Heavy-hitter / collision fallback:** If recursive splitting fails to
     reduce max sub-partition size (for example, hash collisions or huge
     per-group state like `ARRAY_AGG`), fall back to DataFusion's sorted-run
     spill-and-merge pattern ([`AggregateSpill`][stock-agg-spill]) for that
     partition, or return `DataFusionError::ResourcesExhausted` if even one
     single group's state exceeds the entire `MemoryPool`.

### 3.3 Out-of-core `HashJoin`: partition eviction, `ProbeSpill`, and child pipeline

Mirroring DuckDB's `PhysicalJoin::BuildJoinPipelines`
(`~/github/duckdb-morsel-reference/src/execution/operator/join/physical_join.cpp:74-78`)
and `JoinHashTable` / `ProbeSpill`, out-of-core `HashJoin` maps onto `#157`'s
existing `Pipeline` / `MetaPipeline` DAG with **zero new DAG primitives**:

```text
P0: Scan(build) -> RadixHashBuildSink                                     [G0]
P1: Scan(probe) -> HashProbe(resident M, ProbeSpill S) -> [suffix] -> S1  [G1]  req Finalized(G0)
P2: SpilledJoinSource(S) -> [suffix] -> S1                                [G1]  req Drained(P1)
```

`P1` and `P2` belong to the **same** `MetaPipeline` `G1` (feeding the same
downstream sink `S1`, whether `Result`, `GroupBy`, or a parent join's build
sink), and `P2` declares `requires: [Drained(P1)]`. This is the exact same
`Drained(P1)` barrier [morsel_pipeline_architecture.md] already uses for
emitting unmatched build rows in `LEFT`/`FULL` outer joins! When `S` is empty
and no unmatched build scan is required, `P2` opens as a 0-work empty source
and drains immediately.

#### Step-by-step execution flow

1. **Build pipeline (`P0` -> `G0`):**
   - Workers scatter build rows + `u64` hashes into `2^R` partitions and update
     `LocalJoinFilterSummary` over **every** build row (so dynamic Parquet
     pruning on `P1` stays active even when `P0` spills!).
   - Whenever `local_reservation.try_grow` fails during `P0::sink`, evict the
     worker's largest in-memory radix partition(s) to disk via `SpillManager`
     (writing canonical build payload batches + optional `u64` hash column) and
     `shrink` the reservation.
   - In `G0::finalize`, compute the total memory required for each partition
     `p ∈ 0..2^R` (payload batches + `JoinHashMapU32` directory). Greedily pack
     the largest subset of partitions `M ⊆ 0..2^R` that fits within
     `available_budget - probe_spill_headroom` (keeping already-in-memory
     partitions first to avoid unnecessary I/O). Spill any remaining unspilled
     partitions into `S = (0..2^R) \ M`.
   - Build immutable `PartitionJoinTable`s in parallel for `p ∈ M` only.
   - Commit `Arc<RadixJoinArtifact>`:
     - `resident: HashMap<usize, PartitionJoinTable>` (`p ∈ M`),
     - `spilled_build: HashMap<usize, Vec<Arc<dyn SpillFile>>>` (`p ∈ S`),
     - `dynamic_filter: FinalizedJoinFilter` (built over all `M ∪ S`),
     - `probe_spill_manifest: Mutex<SpilledProbeManifest>` (populated on `P1`
       worker exits).

2. **Primary probe pipeline (`P1`, `requires: [Finalized(G0)]`):**
   - `ParquetScan::open` binds `dynamic_filter` and prunes probe files/row
     groups/pages/rows before decode.
   - **In-memory fast path (`S.is_empty()`):** Workers allocate no
     `LocalProbeSpill` state at all and probe `resident[p]` lock-free.
   - **Hybrid spill path (`!S.is_empty()`):** Each probe worker owns a
     `LocalProbeSpill` with memory-reserved append buffers for partitions in `S`.
     For each probe batch:
     - Compute `create_hashes` on probe keys once.
     - Rows with `p ∈ M` probe `resident[p]` immediately and emit matches
       downstream (using `#157`'s `HaveMoreOutput` continuation to bound fanout).
     - Rows with `p ∈ S` are appended to `LocalProbeSpill::buffers[p]`; when a
       buffer reaches its flush threshold (or `try_grow` triggers), flush it
       via `SpillManager` (`append_batch_async` / `spill_record_batch_and_finish`)
       into a probe spill file for partition `p`.
     - For `Right` / `Full` / `RightAnti` / `RightMark` joins, unmatched probe
       rows with `NULL` keys (under `NullEqualsNothing`) never match any build
       partition and can be emitted directly during `P1` without spilling!
   - On `P1` worker normal EOF (`final_execute` + worker exit handoff), flush
     and finish all `LocalProbeSpill` buffers (`finish_async`) and hand their
     `Vec<Arc<dyn SpillFile>>` per partition `p ∈ S` to `probe_spill_manifest`.

3. **Follow-up spilled-join pipeline (`P2`, `requires: [Drained(P1)]`):**
   - `P2::open` first emits any unmatched build rows for resident partitions `M`
     (for `Left` / `Full` / `LeftAnti` / `LeftMark` joins using `#157`'s
     merged probe match bitmaps), then drops `M`'s hash tables and releases `M`'s
     memory reservations so the full `MemoryPool` budget is available for `S`.
   - Partition `S` into memory-fitting rounds `B_1, B_2, ..., B_k` (discarding
     any partition `p ∈ S` whose probe spill list is empty if unmatched build
     rows are not needed, e.g. `Inner` / `LeftSemi` / `RightSemi`!).
   - For each round `B_i`:
     1. Load build spill files for all `p ∈ B_i` in parallel and build
        immutable `PartitionJoinTable`s for `B_i`.
     2. Workers claim probe spill files for `p ∈ B_i` dynamically via
        `SpillManager::read_spill_as_stream_unbuffered`, probe `B_i`'s hash
        tables lock-free, and drive matched batches through `P2`'s downstream
        suffix into `G1`'s sink `S1`.
     3. If the join emits unmatched build rows, merge `B_i`'s match bitmaps
        after `B_i`'s probe files finish and emit unmatched build rows of `B_i`.
     4. Drop `B_i`'s hash tables **and** immediately drop all build and probe
        `Arc<dyn SpillFile>` handles for `B_i` before advancing to `B_{i+1}`,
        bounding peak disk space as well as peak RAM.
   - **Single-partition build overflow (extreme key skew):** If one build
     partition `p ∈ S` alone exceeds the memory budget:
     1. If recursion depth `< MAX_RADIX_DEPTH` and distinct key count permits
        splitting, recursively re-partition both `spilled_build[p]` and
        `spilled_probe[p]` on the next `R'` rotated hash bits.
     2. Otherwise (e.g., 100% duplicate-key skew that no hash function can
        split), execute a **block-nested-loop hash sub-pass** over `p`: load a
        memory-bounded slice of `spilled_build[p]` into an in-memory hash table,
        stream all of `spilled_probe[p]` against that slice (carrying a compact
        probe-matched bitmap per probe spill file only if the join preserves
        unmatched probe rows), repeat for the next build slice, emit unmatched
        probe rows after the final slice if required, and drop `p`'s files.

### 3.4 Out-of-core `Sort` (and `TopN`)

```text
P0: Scan -> [Filter/Proj] -> ExternalSortSink (spill sorted runs on try_grow err) [G0]
P1: FinalizedSortMergeSource -> Result                                            [G1]  req Finalized(G0)
```

- **`P0::sink`:** Each worker buffers input batches under a
  `MemoryConsumer::with_can_spill(true)` reservation plus reserved
  `sort_spill_reservation_bytes` headroom (matching [`ExternalSorter`][stock-sorter]).
  On `try_grow` failure, the worker sorts its buffered batches locally
  (preserving exact `ASC`/`DESC` and `NULLS FIRST`/`NULLS LAST`), writes the
  sorted run through `SpillManager` (which compacts view arrays via
  `gc_view_arrays`) into a [`SortedSpillFile`][sorted-spill-file] recording
  `max_record_batch_memory`, and `shrink`s the freed reservation.
- **`G0::combine` & `G0::finalize`:** Combine collects all workers' remaining
  in-memory sorted runs and `Vec<SortedSpillFile>`. If no worker spilled,
  `finalize` commits the in-memory runs for a single-level k-way merge in `P1`.
  If any worker spilled, sort and spill remaining in-memory runs too (matching
  [`ExternalSorter::sort`][stock-sorter-spill-inmem], so in-memory runs cannot
  push the merge over budget) and commit `SpilledSortRuns`.
- **`P1` (`FinalizedSortMergeSource`):** Single ordered reader driving
  DataFusion's [`StreamingMergeBuilder`][sorted-spill-file] /
  [`MultiLevelMergeBuilder`][multi-level-merge] (`FanInHeadroom`), automatically
  respecting `DiskManager::max_spill_merge_fan_in` and `MemoryPool` limits via
  multi-level spill merges when the run count exceeds what fits in memory.
- **`TopN` (`ORDER BY ... LIMIT K OFFSET O`):** Worker-local heaps stay bounded
  to `O(K + O)`. If an unusually large `K + O` causes `try_grow` to fail during
  `TopN`, degrade cleanly to the same sorted-run spill + bounded merge +
  `OFFSET`/`LIMIT` slice rather than failing unnecessarily.

### 3.5 Spill cleanup on normal exit, error, cancellation, and stream drop

Strict resource cleanup extends `#157`'s teardown invariants to disk state:

| Failure / exit path                                                                    | Cleanup guarantee                                                                                                                                                                                                                                                                                                                                                          |
| -------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Normal completion**                                                                  | Partition spill files are dropped eagerly as soon as their partition/round finishes (not held until query EOF). At stream EOF, `DiskManager::active_files_count() == 0` and all `MemoryReservation`s are `0`.                                                                                                                                                              |
| **Worker error or disk-quota exhaustion (`DiskManager::max_temp_directory_size`)**     | The worker calls best-effort `InProgressSpillFile::abort_async().await` on any open async writer before returning the error (matching [`spill_manager.rs:155-162`][spill-manager-abort]), then signals the supervisor's terminal failure transition. The supervisor never falls back to stock DataFusion after native execution has started.                               |
| **Peer error / supervisor cancellation**                                               | The supervisor closes the result receiver, cancels the run token, aborts and joins all worker/producer/finalize tasks (`JoinSet`), and drops `GlobalSinkState`, `SpilledProbeManifest`, and all run artifacts. Dropping every `Arc<dyn SpillFile>` invokes [`RefCountedTempFile::drop`][refcounted-drop] to unlink local files and reclaim `DiskManager::used_disk_space`. |
| **Early consumer stream drop (e.g., `LIMIT` above `MorselExec` or client disconnect)** | `MorselExec`'s stream drop triggers the abort-on-drop `JoinSet` owner and drops run-owned artifact slots. Workers unwind, run best-effort cleanup of in-progress writers on cancellation checks, and drop all `Arc<dyn SpillFile>` and `MemoryReservation` handles; no `SpillFile` or reservation is ever stored on `MorselExec` or `PipelineGraph`.                       |
| **Async / custom `TempFileFactory` backends**                                          | Because `Drop` cannot `.await`, workers check the cancellation token between batches and before/after each `append_batch_async` and call `abort_async().await` explicitly on cancellation before exiting. As documented on [`AsyncSpillWriter`][spill-file], backends also rely on lifecycle rules only as a crash backstop.                                               |

## 4. Test plan, module layout, and phased follow-up issues

### 4.1 Private module additions under `datafusion/core/src/morsel/`

Extending the private module layout in [morsel_pipeline_architecture.md]:

| Path                                        | Responsibility                                                                                                                                                                                                         |
| ------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `datafusion/core/src/morsel/join_filter.rs` | `LocalJoinFilterSummary`, `FinalizedJoinFilter`, `SplitBlockBloomFilter`, `JoinBloomFilterExpr`, join-type/NULL safety gating, and `ParquetScan::open` predicate binding.                                              |
| `datafusion/core/src/morsel/radix.rs`       | `split_radix_hash`, adaptive radix bit selection, per-partition scatter buffers, parallel partition finalize driver over tracked `JoinSet` tasks, and `MemoryReservation` handoffs.                                    |
| `datafusion/core/src/morsel/spill.rs`       | Thin wrapper over `SpillManager` / `InProgressSpillFile`, spill-headroom reservation management, partition-scoped `Arc<dyn SpillFile>` manifests, `abort_async` cancellation helper, and `#138` `SpillMetrics` wiring. |
| `datafusion/physical-plan/src/spill/mod.rs` | Minimal `#[doc(hidden)]` internal re-export of `InProgressSpillFile` (only when Phase 4 lands) if `datafusion/core` invokes incremental async spilling directly.                                                       |

### 4.2 Concrete test plan

1. **Dynamic join filter pushdown (`datafusion/core/tests/morsel_join_pushdown.rs`):**
   - **Pruning proofs on Parquet:** Multi-row-group Parquet fixtures where build
     keys match only 1 of 8 row groups (verifying `row_groups_pruned_statistics > 0`
     and reduced `bytes_scanned`), plus dense-range sparse-key fixtures
     verifying `JoinBloomFilterExpr` prunes >95% of rows inside `RowFilter`
     when `pushdown_filters = true` without decoding wide payload columns.
   - **Empty-build & all-NULL-build short-circuit:** Prove 0 Parquet data range
     reads (gated `AsyncFileReader` that panics on data fetch!) for `INNER` /
     `LEFT SEMI` with empty build or all-NULL build keys under
     `NullEqualsNothing`.
   - **Exactness & NULL parity against stock (`morsel_mode = off`):** Cover
     `NullEqualsNothing`, `NullEqualsNull` (`IS NOT DISTINCT FROM`), single-key
     and multi-key `(NULL, 5)` combinations, `NaN` / `-0.0` float keys,
     truncated string stats, `Inner` / `Left` / `Right` / `Full` / `LeftSemi` /
     `LeftAnti` / `RightAnti` / null-aware `NOT IN`, and multi-join chains
     (TPC-H Q3 and Q12 shapes) with both `pushdown_filters = true` and `false`,
     and with `#167` sub-RG splitting enabled.
   - **Config & isolation:** Verify `enable_join_dynamic_filter_pushdown = false`
     disables pushdown cleanly, and repeated/concurrent executions of one
     prepared `MorselExec` never leak dynamic filters across runs.
2. **Radix-partitioned hash tables (`datafusion/core/src/morsel/tests.rs` + integration tests):**
   - **Bit-layout unit test:** Verify `split_radix_hash` yields uniform top-7
     `h2` tag distribution inside every radix partition for `R ∈ 0..=12`.
   - **Parity & skew:** High-cardinality (1M+ groups / build rows), low-cardinality
     (fast path `R = 0`), heavy-hitter Zipf skew, empty input, and multi-column
     / string / decimal keys matching stock `off`.
   - **Parallel finalize & memory accounting:** Assert parallel partition
     finalize runs on multiple workers, `MorselMetrics` reports combine/finalize
     time separately, and `MemoryPool::reserved() == 0` after completion,
     error, or early drop.
3. **Out-of-core spilling (`datafusion/core/tests/morsel_spill.rs`):**
   - **Forced-OOM spill parity:** Run `GROUP BY`, `HashJoin` (inner + outer),
     and `Sort` / `TopN` under tight [`GreedyMemoryPool`][memory-pools] and
     [`FairSpillPool`][memory-pools] budgets (e.g., 1–8 MiB) on multi-partition
     inputs, asserting exact parity with stock `off` AND `spilled_bytes > 0` /
     `spill_file_count > 0`.
   - **Skew & recursive spill:** Force a single radix partition to exceed the
     pool (both distinct-key skew triggering recursive radix split and 100%
     identical-key skew triggering `HashJoin` block-nested-loop sub-pass /
     `GROUP BY` sorted fallback).
   - **Disk quota & cleanup invariants:** Test `DiskManager`
     `max_temp_directory_size` exhaustion (verifying clean `ResourcesExhausted`
     without stock fallback), mid-spill I/O error injection, and early stream
     drop with full output channel — asserting `active_files_count == 0`,
     `used_disk_space == 0`, `pool.reserved() == 0`, and `abort_count` tracking
     using a mock `TempFileFactory` (matching `AbortTrackingTempFileFactory` in
     [`spill_manager.rs:248`][spill-manager-abort-test]).
4. **Benchmarks (after implementation phases):**
   - Paired Stock DataFusion vs DuckDB vs Morsel (`morsel_mode = required`)
     runs on the `#165` TPC-H SF1 Parquet suite (specifically **Q3**, **Q12**,
     **Q1** high-cardinality variant, and a constrained-memory SF1/SF10 spill
     benchmark) under matched CPU/thread budgets (`taskset`, warm page cache),
     reporting wall time, `bytes_scanned`, `row_groups_pruned_statistics`,
     `pushdown_rows_pruned`, `spilled_bytes`, and peak memory.

### 4.3 Phased follow-up issue breakdown

Each phase is scoped to land as a self-contained PR with clear dependencies and
minimal file overlap:

| Phase       | Proposed follow-up issue                                                        | Depends on             | Scope & acceptance highlight                                                                                                                                                                                                                       |
| ----------- | ------------------------------------------------------------------------------- | ---------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Phase 1** | Dynamic join min/max + small `InList` pushdown into native Parquet probe scans  | `#150`, `#158`, `#167` | `LocalJoinFilterSummary`, `FinalizedJoinFilter` (bounds + `InList` + optional `KeyRangeBitmap`), empty/all-NULL build short-circuit, full NULL/join-type safety matrix, Parquet pruning proofs.                                                    |
| **Phase 2** | Cache-resident split-block Bloom filter `RowFilter` pushdown                    | Phase 1                | `SplitBlockBloomFilter` + `JoinBloomFilterExpr` evaluated on decode workers under `pushdown_filters = true`; paired TPC-H SF1 Q3 & Q12 benchmark showing reduced `bytes_scanned` / decode time.                                                    |
| **Phase 3** | Radix-partitioned hash tables & parallel finalize for `GROUP BY` and `HashJoin` | `#158`, `#168`         | `radix.rs` with `split_radix_hash`, adaptive `R = 0 -> 6..8`, Layout A over Arrow/`RowConverter`, parallel `JoinSet` partition finalize, `MemoryPool` transfer accounting, high-cardinality microbenchmark.                                        |
| **Phase 4** | Out-of-core spilling for `GROUP BY` and `Sort` via `SpillManager`               | Phase 3, `#159`        | `spill.rs` + `#[doc(hidden)]` `InProgressSpillFile` seam; radix partition spill + recursive/sorted skew fallback for `GROUP BY`; sorted-run spill + `MultiLevelMergeBuilder` for `Sort`; tight-pool parity & zero-leak cleanup tests.              |
| **Phase 5** | Out-of-core `HashJoin` (`ProbeSpill` + `Drained(P1)` child pipeline)            | Phase 3, Phase 4       | Build partition eviction (`M` / `S`) preserving full dynamic filter, `LocalProbeSpill` + `SpilledProbeManifest`, `P2` child pipeline in `G1`, single-partition skew block sub-pass, outer-join unmatched emission, constrained-RAM join benchmark. |

[morsel_pipeline_architecture.md]: morsel_pipeline_architecture.md
[morsel_parquet_adapter.md]: morsel_parquet_adapter.md
[morsel_parquet_splitting.md]: morsel_parquet_splitting.md
[morsel_metrics.md]: morsel_metrics.md
[upstream-scheduling]: https://github.com/apache/datafusion/issues/21719
[upstream-parallelism]: https://github.com/apache/datafusion/pull/2226
[morsel-parquet-lower]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/core/src/morsel/parquet.rs#L55-L201
[parquet-create-morselizer]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/datasource-parquet/src/source.rs#L588-L611
[parquet-buffered]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/datasource-parquet/src/opener/buffered.rs#L22-L116
[dyn-filter]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/physical-expr/src/expressions/dynamic_filters/mod.rs#L57-L398
[shared-bounds]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/physical-plan/src/joins/hash_join/shared_bounds.rs#L50-L120
[partition-bounds]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/physical-plan/src/joins/hash_join/shared_bounds.rs#L66-L115
[shared-bounds-membership]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/physical-plan/src/joins/hash_join/shared_bounds.rs#L123-L176
[preserve-probe-nulls]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/physical-plan/src/joins/hash_join/shared_bounds.rs#L917-L968
[stock-join-allow]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/physical-plan/src/joins/hash_join/exec.rs#L1078-L1126
[hash-table-lookup]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/physical-plan/src/joins/hash_join/partitioned_hash_eval.rs#L266-L421
[key-range-bitmap]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/physical-plan/src/joins/key_range_bitmap.rs#L57-L115
[key-range-pruning]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/pruning/src/key_range_bitmap_expr.rs#L20-L56
[rg-pruner]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/datasource-parquet/src/push_decoder.rs#L146-L250
[row-filter-prebuild]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/datasource-parquet/src/row_filter.rs#L475-L514
[row-filter-from-prebuilt]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/datasource-parquet/src/row_filter.rs#L520-L560
[can-push-down]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/datasource-parquet/src/row_filter.rs#L364-L405
[opener-rg-stats]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/datasource-parquet/src/opener/mod.rs#L1358-L1364
[opener-rg-bloom]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/datasource-parquet/src/opener/mod.rs#L1652-L1657
[opener-page-prune]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/datasource-parquet/src/opener/mod.rs#L1707-L1725
[config-inlist]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/common/src/config.rs#L1939-L1960
[join-type-preserved]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/common/src/join_type.rs#L103-L116
[join-type-empty-build]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/common/src/join_type.rs#L136-L147
[join-type-empty-map]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/common/src/join_type.rs#L159-L164
[hash-random-state]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/common/src/hash_utils.rs#L30-L31
[create-hashes]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/common/src/hash_utils.rs#L1362-L1390
[group-values-rows]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/physical-plan/src/aggregates/group_values/row.rs#L32-L108
[join-hash-map]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/physical-plan/src/joins/join_hash_map.rs#L40-L110
[mem-consumer]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/execution/src/memory_pool/mod.rs#L319-L322
[mem-try-grow]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/execution/src/memory_pool/mod.rs#L467-L476
[mem-take]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/execution/src/memory_pool/mod.rs#L507-L512
[memory-pools]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/execution/src/memory_pool/pool.rs#L64-L170
[spill-manager]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/physical-plan/src/spill/spill_manager.rs#L27-L100
[spill-manager-abort]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/physical-plan/src/spill/spill_manager.rs#L155-L162
[spill-read-unbuffered]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/physical-plan/src/spill/spill_manager.rs#L199-L209
[spill-manager-abort-test]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/physical-plan/src/spill/spill_manager.rs#L248-L320
[in-progress-spill]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/physical-plan/src/spill/in_progress_spill_file.rs#L27-L100
[in-progress-async]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/physical-plan/src/spill/in_progress_spill_file.rs#L103-L152
[in-progress-abort]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/physical-plan/src/spill/in_progress_spill_file.rs#L219-L232
[spill-reader-stream]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/physical-plan/src/spill/mod.rs#L56-L86
[gc-view-arrays]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/physical-plan/src/spill/mod.rs#L852-L865
[spill-file]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/execution/src/spill_file.rs#L17-L95
[refcounted-tmp]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/execution/src/disk_manager.rs#L396-L435
[refcounted-drop]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/execution/src/disk_manager.rs#L437-L453
[stock-agg-spill]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/physical-plan/src/aggregates/spill.rs#L49-L120
[stock-sorter]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/physical-plan/src/sorts/sort.rs#L250-L315
[stock-sorter-spill-inmem]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/physical-plan/src/sorts/sort.rs#L445-L465
[sorted-spill-file]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/physical-plan/src/sorts/streaming_merge.rs#L50-L65
[multi-level-merge]: https://github.com/robtandy/datafusion/blob/2aee01e23408a688828675d4c0cd489448759b33/datafusion/physical-plan/src/sorts/multi_level_merge.rs#L34-L55
