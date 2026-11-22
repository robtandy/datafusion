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

At fork revision `18425fd6c2da7c4813f7a27ca0f68553bb36cac9`:

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
  `meta_pipeline.CreateChildPipeline`, lines 79–82) shows how a hash join builds
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
[`SharedBuildAccumulator`][shared-bounds], `wait_complete()`, and a per-row-group
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
   [`shared_bounds.rs`][shared-bounds-membership]). Keep the container-pruning
   boundary explicit:
   - In [`PruningPredicate`][pruning-inlist], `IN (...)` is rewritten for
     row-group min/max and page-index pruning **only** on a bare `Column` up to
     `execution.parquet.max_in_list_size` (default `20`,
     [`config.rs:1495`][config-max-inlist]). Single-column lists of 21–150
     values and multi-key `struct(...)` lists get **no** row-group min/max or
     page-index rewrite under the default cap.
   - Consequently, for `<= 20` single-column keys, `InListExpr` provides both
     container pruning (min/max, page index, and Parquet file-stored SBBF bloom
     filters via `prune_by_bloom_filters`) and exact `RowFilter` evaluation.
     For 21–150 single-column keys or multi-key `struct(...)` lists,
     `InListExpr` provides exact zero-false-positive `RowFilter` evaluation
     (plus Parquet file bloom-filter pruning on single-column keys via
     `LiteralGuarantee`), while container-level min/max and page pruning come
     from the companion per-key `bounds` predicate. For larger sparse
     single-column integer/date builds, DataFusion's [`KeyRangeBitmap`][key-range-bitmap]
     (capped at `128 KiB`, `MAX_BUCKETS = 1 << 20`) can also prune containers,
     but [`KeyRangeBitmapPruningExpr`][key-range-pruning] is `pub(crate)` in
     `datafusion-pruning` and [`build_hash_lookup_pruning_expr`][hash-lookup-pruning]
     (`pruning_predicate.rs:1537-1567`) currently builds it **only** by
     downcasting [`HashTableLookupExpr`][hash-table-lookup] and reading
     `pruning_bitmap()`. A private `JoinBloomFilterExpr` or `InListExpr` in
     `datafusion/core` therefore gets **no** `KeyRangeBitmap` container pruning
     without a narrow workspace-internal seam (for example, a `#[doc(hidden)]`
     `KeyRangeBitmap` provider/pruning hook in `datafusion-pruning`, tracked in
     Section 4.1 and Phase 2 below).
3. **Cache-resident split-block Bloom filter (`finalize`):** When the build
   exceeds the `InList` threshold (for example, 5K–5M distinct keys) and stays
   in memory (or spills with hash-column replay; see Section 3.3), build a
   single-cache-line split-block bloom filter (`256-bit` or `512-bit` blocks,
   8–12 bits per distinct key, targeting ~1%–2% FPP, capped at a configurable
   ceiling such as `4 MiB`–`8 MiB` and reserved in `MemoryPool`):
   - Size the bit array in `finalize` once the exact build row/distinct count
     is known (avoiding pre-sizing guess errors during `sink`).
   - Populate it directly from the 64-bit `foldhash` values already retained
     per radix partition (Section 2.2), using parallel bitwise-OR or
     partition-local shard inserts if the filter uses power-of-two blocks.
     **Spilled-build soundness invariant:** Never populate a bloom filter from
     only resident partitions `M` while omitting evicted partitions `S` (that
     would cause false negatives). On spilled builds, either fall back cleanly
     to **bounds-only** dynamic pushdown (`JoinMembershipFilter::None`, the
     recommended v1 default since builds large enough to spill typically exceed
     the bloom FPP budget anyway), or—when total build cardinality still fits
     the bloom FPP budget—stream the mandatory spilled `u64` hash column of
     every partition in `S` during `finalize` to insert all `M ∪ S` hashes
     before publishing `FinalizedJoinFilter`.
   - Wrap it in a private `JoinBloomFilterExpr: PhysicalExpr` (analogous to
     [`HashTableLookupExpr`][hash-table-lookup] in `partitioned_hash_eval.rs`,
     but probing a compact L2/L3-resident bitset instead of the full build hash
     table). In `evaluate()`, compute `create_hashes` over the probe key
     column(s) with the exact same `SeededRandomState`, select the cache-line
     block from the low bits (`block = (h as usize) & block_mask`) and derive
     intra-block bit positions from the upper 32 bits (`(h >> 32) as u32` via
     standard split-block multiplicative salts) so block selection is orthogonal
     to the radix partition window `[33..57]` (Section 2.2), and return a
     `BooleanArray`.

#### Publication into `ParquetScan::open`

When `ParquetScan` is lowered as the source of a probe pipeline (feeding
`Filter -> Projection -> HashProbe` and possibly further downstream
`HashProbe`s on the same pipeline with no blocking operator in between), the
planner records a `Vec<JoinFilterBinding>` on the `ParquetScan` spec mapping
each eligible downstream `ArtifactKey<JoinBuildArtifact>` to the probe key
column expression(s) in the scan's projected coordinates (remapped by position
through intervening projections/aliases and intervening joins).

**Intervening `HashProbe` push-through rule (`lr_is_preserved`):** A downstream
join `J_down`'s dynamic filter may cross an intervening `HashProbe(J_mid)` onto
`J_mid`'s probe-side scan **only** when `J_mid` preserves its probe (right)
side under DataFusion's [`lr_is_preserved`][lr-is-preserved] and
[`HashJoinExec::gather_filters_for_pushdown`][gather-filters-pushdown] rules
(`exec.rs:1961-2058`, `2810-2827`):

- **Allowed intervening `J_mid` types** (`right_preserved == true`): `Inner`,
  `Right`, `RightSemi`, `RightAnti`, and `RightMark`.
- **Forbidden intervening `J_mid` types** (`right_preserved == false`): `Left`,
  `Full`, `LeftAnti`, and `LeftMark` — stop walking upstream and do **not**
  bind `J_down`'s filter onto `J_mid`'s probe scan.
- For an allowed `J_mid`, map every column reference in `J_down`'s probe key
  expression **by position** through `J_mid`'s output `column_indices` to
  `J_mid`'s probe-side input (rejecting expressions that reference build-side
  or mark columns, unless transferred across equi-join keys where
  `supports_key_transfer` holds and `right_preserved == true`).

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
  `InListExpr` (container min/max and page pruning up to
  `max_in_list_size = 20` on bare `Column`s; Parquet file bloom pruning via
  `LiteralGuarantee` for single-column lists), and—once the Phase 2
  `#[doc(hidden)]` `datafusion-pruning` hook lands—`KeyRangeBitmap` during file
  open for:
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

- Entire row groups outside `[min, max]` (or excluded by file bloom filters,
  or by `KeyRangeBitmap` once the Phase 2 `#[doc(hidden)]` pruning hook lands)
  are removed before `#167` evaluates split eligibility or initializes a
  row-group-local range broker.
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

In standard TPC-H `dbgen` data (including the `#165` Parquet fixture in
`benchmarks/morsel_tpch/`, where `orders` is ordered by `o_orderkey` and
`lineitem` by `l_orderkey, l_linenumber`), `o_orderdate` and `l_receiptdate`
are drawn **independently** of `o_orderkey` / `l_orderkey`. Consequently,
filtering by date still produces build keys spanning nearly the entire
`orderkey` domain (`[1, 6_000_000]` at SF1), so min/max bounds prune **no**
row groups on `orders` or `lineitem` in Q3 or Q12. The large wins in Q3 and
Q12 therefore come from the **L2-resident `JoinBloomFilterExpr` evaluated inside
Parquet `RowFilter`** (`#150`), whereas min/max and `KeyRangeBitmap` pruning
shine on **key-correlated or time-clustered** star-schema tables (e.g., fact
tables clustered by event date or surrogate key ranges):

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
     ~45 KiB `J1` bloom filter on `o_custkey`.
  2. `P1` scans `orders` with `o_orderdate < '1995-03-15'` AND `J1_filter(o_custkey)`.
     With `#150`, ~80% of date-passing `orders` rows are rejected by the
     L2-resident `J1` bloom filter inside `RowFilter` before decoding
     `o_shippriority` or probing `J1`, leaving ~147K surviving
     `(o_orderkey, o_orderdate, o_shippriority)` build rows for `J2` (~9.8% of
     `orders`).
  3. `P2` receives `J2`'s ~220 KiB L2-resident bloom filter on `l_orderkey`.
     Although `J2`'s `l_orderkey` min/max spans the full table range and prunes
     no `lineitem` row groups, `RowFilter` evaluates `l_shipdate` and the `J2`
     bloom filter on `l_orderkey` alone, rejecting ~90%+ of date-passing
     `lineitem` rows before ever decoding `l_extendedprice` or `l_discount` and
     avoiding ~90% of `J2` hash-table probes.

- **TPC-H Q12 (`orders ⋈ lineitem`, `GROUP BY l_shipmode`, `ORDER BY l_shipmode`):**
  - `lineitem` has selective predicates (`l_shipmode IN ('MAIL', 'SHIP')`,
    `l_commitdate < l_receiptdate`, `l_shipdate < l_commitdate`,
    `l_receiptdate ∈ [1994-01-01, 1995-01-01)`), reducing 6M SF1 rows to ~31K
    rows (~0.5%), whereas `orders` has 1.5M unfiltered rows.
  - Putting the filtered `lineitem` on the build side (`P0`) produces a ~48 KiB
    `o_orderkey` bloom filter pushed into `ParquetScan(orders)` (`P1`). Again,
    `o_orderkey` min/max spans almost the entire `orders` range and does not
    prune `orders` row groups; the win comes from the L2-resident bloom filter
    rejecting ~98% of `orders` rows inside `RowFilter` before decoding the
    string column `o_orderpriority` and before probing `J1`.

### 1.3 Exactness, NULL semantics, and safety matrix

Dynamic join filters are **advisory semi-join reductions**: false positives are
harmless (caught by `HashProbe`), whereas a single false negative is a silent
wrong-result bug. Follow DataFusion's exact semantics in
[`JoinType::on_lr_is_preserved`][join-type-preserved],
[`empty_build_side_produces_empty_result`][join-type-empty-build],
[`empty_map_produces_empty_result`][join-type-empty-map],
[`HashJoinExec::allow_join_dynamic_filter_pushdown`][stock-join-allow],
[`state_after_build_ready`][stream-after-build], and
[`SharedBuildAccumulator::preserve_probe_nulls`][preserve-probe-nulls].

Convention below: **build = left**, **probe = right** (matching DataFusion's
`HashJoinExec`; swap labels if the planner chooses right-as-build).

| Join type (build=L, probe=R)                 | Push filter into probe (`R`)?                                                                                                                                             | Empty build (`!has_rows`) short-circuits probe?                                 | Non-empty all-NULL build (`has_rows && !has_matchable_keys`, `NullEqualsNothing`) |
| -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- | --------------------------------------------------------------------------------- |
| `Inner`                                      | **Yes**                                                                                                                                                                   | **Yes** (`lit(false)`)                                                          | Short-circuit probe (`lit(false)`); 0 output rows                                 |
| `Left` (build preserved)                     | **Yes**                                                                                                                                                                   | **Yes** (`lit(false)`)                                                          | Short-circuit probe (`lit(false)`); emit unmatched build rows via child pipeline  |
| `LeftSemi`, `RightSemi`                      | **Yes**                                                                                                                                                                   | **Yes** (`lit(false)`)                                                          | Short-circuit probe (`lit(false)`); 0 output rows                                 |
| `LeftAnti`, `LeftMark` (regular)             | **Yes**                                                                                                                                                                   | **Yes** (`lit(false)`)                                                          | Short-circuit probe (`lit(false)`); unmatched build rows still emitted            |
| `LeftAnti`, `LeftMark` (null-aware `NOT IN`) | **Only if build keys are NOT NULL** ([`exec.rs:1084`][stock-join-allow]); **must** add `OR probe_value_key IS NULL` for nullable `on_right[..V]` (`preserve_probe_nulls`) | **Yes** (`lit(false)`; [`stream.rs:618-636`][stream-after-build], 0 build rows) | **Disable pushdown**; must scan probe (pruning probe rows hides probe emptiness)  |
| `Right`, `Full` (probe preserved)            | **No** (unmatched probe rows must survive)                                                                                                                                | **No**                                                                          | **No**                                                                            |
| `RightAnti`, `RightMark`                     | **No** (unmatched probe rows are the output)                                                                                                                              | **No**                                                                          | **No**                                                                            |

Additional soundness invariants:

- **`NullEquality::NullEqualsNothing` (SQL `=`):** Build rows with `NULL` in
  _any_ join key column cannot match any probe row; exclude them from min/max
  bounds, `InList`, and the bloom filter.
  - For **regular** (non-null-aware) joins, probe rows with `NULL` keys evaluate
    to `NULL`/false under `col >= min AND col <= max` and `JoinBloomFilterExpr`
    and are safely dropped.
  - For **null-aware `NOT IN`** joins (`LeftAnti` / `LeftMark` with
    `null_aware = true`), probe rows with a `NULL` in any `NOT IN` value key
    (`on_right[..V]`, where `on_right[V..]` are correlation scope keys) **must
    never be dropped**, because one probe `NULL` makes `NOT IN` evaluate to
    `UNKNOWN` against every build row in that scope. Always wrap the pushed
    filter with `OR k_i IS NULL` for every nullable key in `on_right[..V]`
    (matching [`SharedBuildAccumulator::preserve_probe_nulls`][preserve-probe-nulls]),
    regardless of whether the build side contained any `NULL`s.
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
  which is sound because matching requires every key to match simultaneously
  (wrapped with the `on_right[..V]` `IS NULL` escape above when null-aware).
  The bloom filter hashes the full composite tuple `(k0, k1, ...)` with
  `create_hashes`, preserving full composite selectivity.
- **Floating-point `NaN` and `-0.0`:** DataFusion's hash-join equality uses
  Arrow's IEEE 754 `totalOrder` `make_comparator` and normalizes only `-0.0` to
  `+0.0` via [`float_semantics::normalize`][join-float-norm]
  (`utils.rs:2830-2860`), so `NaN` build keys **do match** `NaN` probe keys
  (and `create_hashes` hashes canonicalized `NaN` consistently). Always retain
  `NaN` keys in `InList` and the bloom filter. Because standard SQL range
  comparisons (`col >= min AND col <= max`) and Parquet statistics do not
  guarantee total-order inclusion of `NaN`, if any `NaN` build key is observed
  on column `i`, **drop the min/max range bound for column `i`** (or preserve
  `NaN` explicitly via total-order bounds; dropping column `i`'s range bound
  while keeping `InList`/bloom active is simple and conservatively exact).
- **Expression keys & schema coercion:** Only push filters whose probe-side
  expressions pass
  [`can_expr_be_pushed_down_with_schemas`][can-push-down] against the scan's
  non-virtual schema (no volatile functions, no missing columns). If a probe
  key involves an implicit cast (e.g., `Int32 -> Int64`), push the cast
  expression in the `RowFilter` / `PruningPredicate` using DataFusion's normal
  `PhysicalExprAdapter` path, or skip pushdown for that key if unsupported.
- **Intervening joins (`lr_is_preserved`):** Never push a downstream join's
  dynamic filter past an intervening `Left`, `Full`, `LeftAnti`, or `LeftMark`
  `HashProbe` onto its probe-side scan (matching
  [`lr_is_preserved`][lr-is-preserved] in `exec.rs:2810-2827`). Pruning probe
  rows below a `Left` or `Full` join leaves previously-matched build rows
  unmatched, causing the outer join to emit synthetic `NULL`-filled probe
  columns that were not present in the unfiltered result — which then falsely
  match a downstream `IS NOT DISTINCT FROM` join or turn a downstream
  null-aware `NOT IN` into `UNKNOWN` (dropping valid rows). Only cross
  intervening joins where `right_preserved == true` (`Inner`, `Right`,
  `RightSemi`, `RightAnti`, `RightMark`) and remap key columns by position
  through the intervening join's output projection.
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
`hashbrown::HashTable` (SwissTable, 0.17) extracts **two** bit ranges from `h`:

- **`h2` (7-bit SIMD control tag):** top 7 bits (`h >> 57`) on 64-bit targets.
- **`h1` (initial probe bucket index):** low bits (`(h as usize) & bucket_mask`).

**Two hazards to avoid:**

1. Selecting radix partition bits from `[64 - R .. 64]` and passing `h`
   unchanged collapses SwissTable's `h2` tag entropy inside each partition to
   `2^(7 - R)` tags (1 tag when `R >= 7`).
2. Rotating `h` left (`h.rotate_left(R)`) moves the constant partition bits
   straight into the low `R` bits `[0..R]` used by `h1`, collapsing initial
   probe positions to `1 / 2^R` of the bucket array (e.g., for `R = 8` on a
   16,384-slot table, every key in a partition probes from only 64 start slots!).

**Required fix:** Carve out disjoint middle bit windows clear of both `h2`
(`[57..64]`) and `h1` (`[0..33]`), and pass `h` **unchanged** into
`hashbrown::HashTable`:

```text
63      57 56     49 48     41 40     33 32                             0
+---------+---------+---------+---------+-------------------------------+
| h2 (7b) | L0 (8b) | L1 (8b) | L2 (8b) | h1 bucket / bloom block (33b) |
+---------+---------+---------+---------+-------------------------------+
```

```rust
/// Extract radix partition index from a middle bit window disjoint from
/// `hashbrown`'s `h2` ([57..64]) and low `h1` bucket bits ([0..33]).
/// Pass `h` unchanged into `hashbrown::HashTable`.
#[inline]
fn radix_partition(h: u64, shift: u32, radix_bits: u32) -> usize {
    debug_assert!(shift >= 33 && shift + radix_bits <= 57);
    if radix_bits == 0 {
        return 0;
    }
    ((h >> shift) & ((1u64 << radix_bits) - 1)) as usize
}
```

- **Strict bit budget (`R0 + R1 + R2 <= 24`):**
  - **Level 0** (`R0 <= 8`, default `R0 = 6..8`): `shift = 57 - R0` (bits
    `[49..57)` for `R0 = 8`).
  - **Recursive spill levels 1 and 2** (`R1 <= 8`, `R2 <= 8`, with
    `R0 + R1 + R2 <= 24` so `shift >= 33` always holds): step down to the next
    disjoint window `shift = 57 - R0 - R1` (`[41..49)`) and
    `shift = 57 - R0 - R1 - R2` (`[33..41)`).
- **`h1` and `h2` stay 100% intact** across every level (full 7-bit `h2` tag
  entropy in `[57..64)` and 33 independent low bits in `[0..33)` for `h1`,
  covering tables up to `2^33` buckets).
- **Bloom filter orthogonality:** As noted in Section 1.1, `SplitBlockBloomFilter`
  indexes its cache-line block from low bits `(h as usize) & block_mask`
  (`[0..33)`) and mixes the upper 32 bits `(h >> 32) as u32` with multiplicative
  salts for intra-block bit positions, keeping bloom block addressing disjoint
  from the radix partition window `[33..57)`.
- For custom join tables storing a 16-bit hash salt array, extract from bits
  `[17..33)` (`(h >> 17) as u16`) — completely disjoint from the entire radix
  window `[33..57)` and `h2` `[57..64)`, and above `h1` bucket bits for
  L2-sized partitions up to `2^17 = 131,072` buckets (or mix `(h >> 32) ^ h`
  purely for the salt word if a partition exceeds `2^17` buckets) while keeping
  `h` unchanged for `h1`/`h2`.

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
     `let p = radix_partition(h, table.radix_shift, table.radix_bits); table.partitions[p].probe(h, ...)`.

### 2.3 Concrete Rust data layouts over Arrow / `RowConverter` + `MemoryPool`

Two candidate physical layouts inside each radix partition:

| Dimension                 | Layout A: Hybrid specialized + `RowConverter` columnar chunks (**Recommended v1**)                                                                                                                                                                                                                                                                       | Layout B: Packed row-store blocks (DuckDB `TupleDataCollection` analogy)                                                             |
| ------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| **Group keys**            | Single fixed-width key (`Int32`/`Int64`/`UInt64`/`Date32`): dense `Vec<T>` + `hashbrown::HashTable<(u32, u64)>` (per-partition `u32` group IDs, modeled on stock [`GroupValuesPrimitive`][group-values-primitive]'s `HashTable<(usize, u64)>`). Multi-column / string / decimal: [`arrow::row::RowConverter`][group-values-rows] (`Rows` per partition). | Custom variable-width row arena (`Vec<AlignedBlock>`) storing key bytes + inline fixed accumulator state.                            |
| **Aggregate states**      | Columnar typed vectors per partition (`Vec<i64>`, `Vec<i128>`, `Vec<f64>`, `NullBufferBuilder`) indexed by dense `group_id: u32`.                                                                                                                                                                                                                        | Inline fixed-size structs per row; variable states (`String` min/max) stored in side heap.                                           |
| **Join build storage**    | Per-partition coalesced `RecordBatch` (or `Vec<RecordBatch>`) + per-partition [`JoinHashMapU32`][join-hash-map] (`head` table + `next: Vec<u32>`, since each of `2^R` partitions fits in `u32`) + optional `Vec<u16>` hash salts.                                                                                                                        | Packed build rows `(next_ptr, hash_salt, key_bytes, payload_bytes)` in 256 KiB blocks.                                               |
| **Arrow / Spill interop** | Zero-copy or single-copy conversion to `RecordBatch` for `MorselExec` output and `SpillManager` IPC serialization; reuses Arrow kernels and `gc_view_arrays`.                                                                                                                                                                                            | Requires custom per-`DataType` serializer/deserializer to and from Arrow `RecordBatch` and `SpillManager`.                           |
| **Cache locality**        | High: every partition's hash table, `Rows` buffer, and columnar state vectors fit together inside L2 during `finalize` and `probe`.                                                                                                                                                                                                                      | Highest for wide multi-column probes/aggregates (single cache miss per row instead of 1 per column), at much higher code complexity. |

**Recommendation:** Adopt **Layout A** for the first radix + spill milestones.
It keeps Arrow `RecordBatch` and `RowConverter` as the universal data model
across `GROUP BY`, `HashJoin`, and `SpillManager`, preserves `#140`/`#168`'s
fast specialized primitive key path, and unlocks 90% of the L2 + parallel
finalize win without maintaining a bespoke multi-type row serialization engine.
Revisit Layout B only if post-radix profiles show columnar gather latency
dominating wide-payload probes.

#### `MemoryPool` accounting discipline

Respect two exact invariants of DataFusion's `MemoryPool` API
([`memory_pool/mod.rs:304-345`][mem-consumer],
[`pool.rs:223-244`][fair-spill-try-grow]):

1. `MemoryConsumer::register(self, pool)` consumes the consumer by value, and
   `MemoryConsumer` is not `Clone`; creating distinct consumers requires
   [`MemoryConsumer::clone_with_new_id`][mem-clone-id].
2. [`FairSpillPool::try_grow`][fair-spill-try-grow] checks
   `reservation.size() + additional > available` on the **individual
   `MemoryReservation`**, not the sum across multiple `split()` reservations
   sharing one consumer ID. Calling `try_grow` independently on `2^R` split
   reservations would bypass the fair-share cap by up to `2^R×`.

Apply this discipline across radix stages:

- Create a template run-local [`MemoryConsumer`][mem-consumer] per blocking
  sink pipeline with `.with_can_spill(true)` (once spill is enabled for that
  operator; `.with_can_spill(false)` before spill lands so `FairSpillPool`
  classifies it accurately).
- **One growing reservation per registered consumer:**
  - Either register **one shared `Arc<MemoryReservation>`** for the stage (note
    [`MemoryReservation::try_grow(&self, ...)`][mem-try-grow] and `shrink(&self, ...)`
    use `AtomicUsize`, so all workers / finalize tasks share one aggregate byte
    count checked accurately against `FairSpillPool`), **or**
  - Register a distinct consumer per active worker via
    `template.clone_with_new_id().register(&pool)` (bounded by worker count
    `W`, **never** `2^R`!), calling [`try_grow`][mem-try-grow] before growing
    partition chunk buffers, `Rows` arenas, `HashTable` allocations, or
    accumulator vectors.
- **Use `split()` / `take()` only for static/shrinking RAII leases:** When
  moving already-reserved partition chunks into `GlobalSinkState` or attaching
  a finalized partition lease to an artifact, `split()` / [`take()`][mem-take]
  transfers ownership with zero unreserved gap and zero double-counting — but
  **never call `try_grow` on a `split()` child reservation**. Any fresh growth
  during parallel `finalize` (e.g., transient peak while compacting chunks and
  building `JoinHashMapU32`) must grow either on the stage's single shared
  `Arc<MemoryReservation>` or on a per-task `clone_with_new_id()` reservation,
  then `shrink` / drop the transient growth immediately after dropping the
  uncompacted input chunks.
- Never charge cloned `Arc` references to already-accounted Arrow buffers; keep
  each static lease attached to its take-once partition slot (Section 3.2/3.3)
  so dropping a completed partition or round releases its reservation
  immediately.

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
  `DiskManager::spilling_progress().current_bytes` /
  [`DiskManager::spilling_progress().active_files_count`][disk-spilling-progress]
  automatically when the last `Arc<dyn SpillFile>` drops
  ([`disk_manager.rs:437`][refcounted-drop]).
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
2. **Async vs synchronous spill I/O across `#157` traits:**
   - **`Sink::sink` (`P0` build / `GROUP BY` / `Sort`)** is `async` in
     [morsel_pipeline_architecture.md], so sink workers call
     `InProgressSpillFile::append_batch_async` / `finish_async` /
     `abort_async().await` directly without holding any mutex or supervisor
     state gate across `.await`.
   - **`Operator::execute` and `Operator::final_execute` (`HashProbe` in `P1`)**
     are **synchronous** in `#157`, so `LocalProbeSpill` cannot `.await`
     `append_batch_async`, `finish_async`, or `abort_async().await` inside
     `execute` / `final_execute`. Phase 5 v1 therefore uses `SpillManager`'s
     **synchronous** APIs (`spill_record_batch_and_finish` when flushing a
     coalesced partition buffer, or `InProgressSpillFile::append_batch` +
     `finish`) inside `LocalProbeSpill` — noting explicitly that local OS
     temp-file writes perform synchronous buffered file I/O on the probe worker
     and that async-only custom `TempFileFactory` backends are not supported on
     the sync probe-spill path. If non-blocking or async-only backend probe
     spills are required later, add an optional async spill hook to `Operator`
     (e.g., `OperatorStep::NeedAsyncFlush` + `async fn flush_spill` /
     `async fn abort_spill` driven by `driver.rs` on flush, EOF, and teardown)
     without changing the synchronous hot path for non-spilling operators.
   - Keep all `InProgressSpillFile` writers strictly worker-local; only
     finished `Arc<dyn SpillFile>` handles cross into `GlobalSinkState` or run
     artifacts.
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
     `SpilledGroupsArtifact { partitions: Arc<[Mutex<Option<PartitionSpillInput>>]>, spill_manager, next_partition: AtomicUsize }`
     with **take-once per-partition slots**, and lets `P1` workers claim **one
     radix partition `p` at a time** (with concurrency bounded by available
     `MemoryPool` budget):
     1. Claim `p = next_partition.fetch_add(1, Ordering::Relaxed)` and move
        exclusive ownership out of the shared artifact via
        `partitions[p].lock().take()` (taking partition `p`'s in-memory chunks,
        `Vec<Arc<dyn SpillFile>>`, and static `MemoryReservation` lease).
     2. Read all in-memory chunks + spill files for partition `p` via
        `SpillManager::read_spill_as_stream_unbuffered`.
     3. Merge partial states into a partition-local hash table (`try_grow`
        checked).
     4. Emit finalized group batches downstream through `P1`'s operators/sink.
     5. Drop the worker-local hash table **and** the taken
        `PartitionSpillInput` immediately — because the take-once slot no
        longer holds any `Arc<dyn SpillFile>` or reservation for `p`, this
        drops the **last** reference, invoking `RefCountedTempFile::drop` to
        unlink `p`'s temp files and freeing `p`'s memory before claiming
        partition `p + 1`.
4. **Single-partition overflow and skew fallback:** If reloading a single
   partition `p` still fails `try_grow`:
   - **Recursive radix split (up to `MAX_RADIX_DEPTH = 2`):** Re-partition `p`'s
     chunks + spill files using the next lower disjoint hash window
     (`radix_partition(h, 57 - R0 - R1, R1)`), spill the sub-partitions, and
     process each sub-partition sequentially.
   - **Heavy-hitter / collision fallback:** If recursive splitting fails to
     reduce max sub-partition size (for example, hash collisions or huge
     per-group state like `ARRAY_AGG`), fall back to DataFusion's sorted-run
     spill-and-merge pattern ([`AggregateSpill`][stock-agg-spill]) for that
     partition, or return `DataFusionError::ResourcesExhausted` if even one
     single group's state exceeds the entire `MemoryPool`.

### 3.3 Out-of-core `HashJoin`: partition eviction, `ProbeSpill`, and child pipeline

Mirroring DuckDB's `PhysicalJoin::BuildJoinPipelines`
(`~/github/duckdb-morsel-reference/src/execution/operator/join/physical_join.cpp:79-82`)
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
     `LocalJoinFilterSummary` (min/max bounds and null flags) over **every**
     build row before any eviction occurs.
   - Whenever `local_reservation.try_grow` fails during `P0::sink`, evict the
     worker's largest in-memory radix partition(s) to disk via `SpillManager`,
     **always writing the precomputed `u64` hash column alongside the build
     payload batches** (avoiding re-hashing on reload in `P2` and enabling
     hash-only bloom replay if needed), and `shrink` the reservation.
   - In `G0::finalize`, compute the total memory required for each partition
     `p ∈ 0..2^R` (payload batches + `JoinHashMapU32` directory). Greedily pack
     the largest subset of partitions `M ⊆ 0..2^R` that fits within
     `available_budget - probe_spill_headroom` (keeping already-in-memory
     partitions first to avoid unnecessary I/O). Spill any remaining unspilled
     partitions into `S = (0..2^R) \ M` (again including their `u64` hashes).
   - Build immutable `PartitionJoinTable`s in parallel for `p ∈ M` only.
   - Finalize `dynamic_filter` soundly over `M ∪ S`:
     - Per-key min/max bounds and null flags already cover **all** `M ∪ S` rows.
     - If `S` is empty, size and populate `SplitBlockBloomFilter` (or `InList`)
       in parallel from the in-memory hashes of `M`.
     - If `S` is non-empty, **never** build a bloom filter from `M` alone.
       Default in v1 to **bounds-only** (`JoinMembershipFilter::None`), since
       spilled builds usually exceed the memory-capped bloom FPP budget anyway;
       optionally, when total build cardinality across `M ∪ S` still fits the
       bloom FPP budget (e.g., tight-RAM pool with moderate NDV), size the
       filter from total `M ∪ S` count, insert `M`'s in-memory hashes, and
       stream the spilled `u64` hash column from `S` before publishing
       `FinalizedJoinFilter`.
   - Commit `Arc<RadixJoinArtifact>` using **take-once ownership slots** so
     `P2` can release `M` and each completed round `B_i` eagerly rather than
     pinning them behind an immutable shared `Arc`:
     - `resident: RwLock<Option<Arc<ResidentPartitions>>>` (`p ∈ M`, plus `M`'s
       static `MemoryReservation` lease; `P1` workers clone the inner
       `Arc<ResidentPartitions>` at worker start and drop their clones before
       `Drained(P1)`),
     - `spilled_build: Arc<[Mutex<Option<Vec<Arc<dyn SpillFile>>>>]>`
       (take-once slot per partition `p ∈ S`),
     - `dynamic_filter: FinalizedJoinFilter` (guaranteed sound over `M ∪ S`),
     - `probe_spill_manifest: Mutex<SpilledProbeManifest>` (populated on `P1`
       worker exits; per-partition entries are `take()`n by `P2` at round
       start).

2. **Primary probe pipeline (`P1`, `requires: [Finalized(G0)]`):**
   - `ParquetScan::open` binds `dynamic_filter` and prunes probe files/row
     groups/pages/rows before decode.
   - **In-memory fast path (`S.is_empty()`):** Workers allocate no
     `LocalProbeSpill` state at all and probe `resident[p]` lock-free.
   - **Hybrid spill path (`!S.is_empty()`):** Each probe worker clones the
     inner `Arc<ResidentPartitions>` once at `Operator::local`, and owns a
     `LocalProbeSpill` with memory-reserved append buffers for partitions in `S`.
     For each probe batch:
     - Compute `create_hashes` on probe keys once.
     - Rows with `p ∈ M` probe `resident[p]` immediately:
       - For `Inner`, `Left`, `Right`, `Full`, `RightSemi`, `RightAnti`, and
         `RightMark`: emit matched (and, where applicable, probe-side
         unmatched/marked) batches downstream (using `#157`'s `HaveMoreOutput`
         continuation to bound fanout), and record build match bits in
         worker-local bitmaps for `Left` / `Full`.
       - For **`LeftSemi`, `LeftAnti`, and `LeftMark`**: **only set build-side
         match bits** in worker-local match bitmaps — do **not** emit build rows
         during `P1`, so every build row is emitted at most once after all probe
         workers drain and bitmaps merge in `P2`.
     - Rows with `p ∈ S` are appended to `LocalProbeSpill::buffers[p]`; when a
       buffer reaches its flush threshold (or `try_grow` triggers), flush it
       synchronously via `SpillManager::spill_record_batch_and_finish` (or
       sync `InProgressSpillFile::append_batch` + `finish`; see Section 3.1 for
       the optional async `Operator` flush hook) into a probe spill file for
       partition `p`.
     - **Non-null-aware `Right` / `Full` / `RightAnti` / `RightMark` shortcut:**
       Under `NullEqualsNothing`, for **non-null-aware** `Right`, `Full`,
       `RightAnti`, and `RightMark` joins only, probe rows with a `NULL` key
       never match any build partition and can be emitted directly during `P1`
       without spilling. **Never** apply this shortcut to **null-aware
       `RightAnti`** (`probe.key NOT IN (build.key)`, stock `NotIn::ProbeRows`,
       [`not_in.rs:64-85`][not-in-scope]): with a non-empty build side,
       `NULL`-key probe rows evaluate to `UNKNOWN` and must be **dropped**, and
       for uncorrelated `NotIn::ProbeRows` if any build key is `NULL`
       (`key_has_null`), **zero** rows may be emitted.
   - On `P1` worker normal EOF (`final_execute` + worker exit handoff), flush
     and finish all `LocalProbeSpill` buffers synchronously (`finish` /
     `spill_record_batch_and_finish`), drop the worker's
     `Arc<ResidentPartitions>` clone, and hand its `Vec<Arc<dyn SpillFile>>`
     per partition `p ∈ S` (plus worker-local build match bitmaps for `M`) to
     `probe_spill_manifest`.

3. **Follow-up spilled-join pipeline (`P2`, `requires: [Drained(P1)]`):**
   - Because `Drained(P1)` guarantees every `P1` worker has already exited and
     dropped its `Arc<ResidentPartitions>` clone (and the supervisor drops its
     artifact-slot reference once `P2` binds it), `P2::open` takes exclusive
     ownership of `M` via `artifact.resident.write().take()`. It merges
     worker-local build match bitmaps for `M` and emits build-driven results
     for `M` once:
     - **`LeftSemi`**: emit **matched** build rows (`bit == 1`) once.
     - **`Left`, `Full`, `LeftAnti`**: emit **unmatched** build rows
       (`bit == 0`) once.
     - **`LeftMark`**: emit every build row with its final Boolean mark.
       Then `P2` drops the taken `Arc<ResidentPartitions>` — dropping the
       **last** strong reference to `M`'s hash tables and releasing `M`'s
       `MemoryReservation` lease immediately so the full `MemoryPool` budget is
       available for `S`.
   - Partition `S` into memory-fitting rounds `B_1, B_2, ..., B_k`. Any
     partition `p ∈ S` with an empty probe spill list can be `take()`n and
     dropped immediately (unlinking its build spill files) for joins that emit
     nothing for unmatched build rows (`Inner`, `LeftSemi`, `RightSemi`,
     `Right`, `RightAnti`, `RightMark`); for `Left`, `Full`, `LeftAnti`, and
     `LeftMark`, `take()` and stream `p`'s unmatched build rows directly
     without building a hash table, then drop `p`'s files.
   - For each round `B_i`:
     1. Move exclusive ownership of every `p ∈ B_i`'s build and probe
        `Vec<Arc<dyn SpillFile>>` out of `artifact.spilled_build[p]` and
        `artifact.probe_spill_manifest` via `.take()`, then load build spill
        files for `B_i` in parallel and build immutable `PartitionJoinTable`s
        for `B_i`.
     2. Workers claim probe spill files for `p ∈ B_i` dynamically via
        `SpillManager::read_spill_as_stream_unbuffered` and probe `B_i`'s hash
        tables lock-free: drive matched/probe-driven batches through `P2`'s
        downstream suffix into `G1`'s sink `S1` (for `Inner`, `Left`, `Right`,
        `Full`, `RightSemi`, `RightAnti`, `RightMark`), or **only set
        build-side match bits** in worker-local bitmaps for `B_i` (for
        `LeftSemi`, `LeftAnti`, `LeftMark`).
     3. Merge worker-local build match bitmaps for `B_i` after `B_i`'s probe
        files finish, and emit `B_i`'s build-driven rows once (`LeftSemi`
        matched rows; `Left` / `Full` / `LeftAnti` unmatched rows; `LeftMark`
        all build rows with their marks).
     4. Drop `B_i`'s round-local hash tables **and** the taken build/probe
        `Arc<dyn SpillFile>` vectors before advancing to `B_{i+1}` — because no
        shared artifact slot still retains references for `B_i`,
        `RefCountedTempFile::drop` unlinks `B_i`'s temp files immediately,
        bounding peak disk space as well as peak RAM.
   - **Null-aware `NOT IN` (`LeftAnti` / `LeftMark` / `RightAnti`) and radix spill:**
     Build-driven null-aware joins (`NotIn::BuildRows`) require global state
     across all probe rows and all build partitions `M ∪ S`: an uncorrelated
     `(probe_is_non_empty, probe_saw_null)` summary, plus—for correlated
     forms—a per-build-row `UNKNOWN` bitmap populated by probing `NULL`-valued
     probe rows against a correlation-scope map covering all of `M ∪ S`. Stock
     DataFusion rejects `PartitionMode::Partitioned` for null-aware joins for
     this exact reason ([`not_in.rs:119-126`][not-in-partitioned]). **In v1
     (Phase 5), exclude `null_aware = true` joins from radix-partitioned spill**:
     keep them on the unpartitioned in-memory `R = 0` path (returning
     `ResourcesExhausted` if a null-aware build exceeds memory, or using
     `prefer` fallback / `required` rejection if null-aware spill is requested).
     To support out-of-core null-aware joins in a subsequent slice:
     - _Uncorrelated `LeftAnti` / `LeftMark`_: reduce every `P1` worker's
       `(probe_is_non_empty, probe_saw_null)` summary at `Drained(P1)` **before**
       `P2` emits any build row from `M` or `B_i` (so `probe_saw_null == true`
       suppresses unmatched `LeftAnti` rows and marks unmatched `LeftMark` rows
       `NULL` across all of `M ∪ S`).
     - _Correlated `LeftAnti` / `LeftMark`_: radix-partition by the
       **correlation scope keys** `on[value_keys..]` (not all keys `on[..]`) so
       every `NULL`-valued probe row routes to the same radix partition as its
       build scope peers and sets per-build-row `UNKNOWN` bits cleanly within
       `M` or round `B_i`.
   - **Single-partition build overflow (extreme key skew):** If one build
     partition `p ∈ S` alone exceeds the memory budget:
     1. If recursion depth `< MAX_RADIX_DEPTH` and distinct key count permits
        splitting, recursively re-partition both `spilled_build[p]` and
        `spilled_probe[p]` on the next lower disjoint hash window
        (`radix_partition(h, 57 - R0 - R1, R1)`).
     2. Otherwise (e.g., 100% duplicate-key skew that no hash function can
        split), execute a **block-nested-loop hash sub-pass** over `p`: load a
        memory-bounded slice of `spilled_build[p]` into an in-memory hash table,
        stream all of `spilled_probe[p]` against that slice, repeat for the next
        build slice, and drop `p`'s files. Maintain a compact probe-matched
        bitmap per probe spill file for **every join type whose probe emission
        depends on cross-slice match status** — `Right`, `Full`, `RightSemi`,
        `RightAnti`, and `RightMark`:
        - For `RightSemi`, emit a matching probe row only on its first `0 -> 1`
          bit transition so a probe row matching multiple build slices is never
          emitted more than once.
        - For `Right`, `Full`, `RightAnti`, and `RightMark`, record matches
          across all build slices and emit unmatched probe rows (or final mark
          booleans) once after the final build slice finishes.

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
  DataFusion's [`StreamingMergeBuilder`][streaming-merge-builder] /
  [`MultiLevelMergeBuilder`][multi-level-merge] (with
  [`FanInHeadroom`][fan-in-headroom]), automatically respecting
  `DiskManager::max_spill_merge_fan_in` and `MemoryPool` limits via multi-level
  spill merges when the run count exceeds what fits in memory.
- **`TopN` (`ORDER BY ... LIMIT K OFFSET O`):** Worker-local heaps stay bounded
  to `O(K + O)`. If an unusually large `K + O` causes `try_grow` to fail during
  `TopN`, degrade cleanly to the same sorted-run spill + bounded merge +
  `OFFSET`/`LIMIT` slice rather than failing unnecessarily.

### 3.5 Spill cleanup on normal exit, error, cancellation, and stream drop

Strict resource cleanup extends `#157`'s teardown invariants to disk state:

| Failure / exit path                                                                    | Cleanup guarantee                                                                                                                                                                                                                                                                                                                                                                                                                                                                             |
| -------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Normal completion**                                                                  | Take-once artifact slots transfer exclusive ownership of each partition/round to its active consumer, so spill files are unlinked eagerly as soon as their partition/round finishes (not held until query EOF). At stream EOF, `DiskManager::spilling_progress().active_files_count == 0` and all `MemoryReservation`s are `0`.                                                                                                                                                               |
| **Worker error or disk-quota exhaustion (`DiskManager::max_temp_directory_size`)**     | Async sink workers (`P0` build/`GROUP BY`/`Sort`) call best-effort `InProgressSpillFile::abort_async().await` on any open async writer before returning the error (matching [`spill_manager.rs:155-162`][spill-manager-abort]); synchronous `LocalProbeSpill` writers drop their `InProgressSpillFile` directly (`RefCountedTempFile::drop` unlinks local files). The supervisor transitions to terminal failure and never falls back to stock DataFusion after native execution has started. |
| **Peer error / supervisor cancellation**                                               | The supervisor closes the result receiver, cancels the run token, aborts and joins all worker/producer/finalize tasks (`JoinSet`), and drops `GlobalSinkState`, `SpilledProbeManifest`, and all run artifacts. Dropping every `Arc<dyn SpillFile>` invokes [`RefCountedTempFile::drop`][refcounted-drop] to unlink local files and reclaim `DiskManager::used_disk_space`.                                                                                                                    |
| **Early consumer stream drop (e.g., `LIMIT` above `MorselExec` or client disconnect)** | `MorselExec`'s stream drop triggers the abort-on-drop `JoinSet` owner and drops run-owned artifact slots. Workers unwind, run best-effort cleanup of open writers on cancellation checks, and drop all `Arc<dyn SpillFile>` and `MemoryReservation` handles; no `SpillFile` or reservation is ever stored on `MorselExec` or `PipelineGraph`.                                                                                                                                                 |
| **Async / custom `TempFileFactory` backends**                                          | Because `Drop` cannot `.await`, async sink workers (`P0`) check the cancellation token between batches and before/after each `append_batch_async` and call `abort_async().await` explicitly on cancellation before exiting (sync `LocalProbeSpill` in v1 requires local/sync `SpillWriter` unless the optional async `Operator` hook from Section 3.1 is enabled). As documented on [`AsyncSpillWriter`][spill-file], backends also rely on lifecycle rules only as a crash backstop.         |

## 4. Test plan, module layout, and phased follow-up issues

### 4.1 Private module additions under `datafusion/core/src/morsel/`

Extending the private module layout in [morsel_pipeline_architecture.md]:

| Path                                          | Responsibility                                                                                                                                                                                                         |
| --------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `datafusion/core/src/morsel/join_filter.rs`   | `LocalJoinFilterSummary`, `FinalizedJoinFilter`, `SplitBlockBloomFilter`, `JoinBloomFilterExpr`, join-type/NULL safety gating, and `ParquetScan::open` predicate binding.                                              |
| `datafusion/core/src/morsel/radix.rs`         | `radix_partition`, adaptive radix bit selection, per-partition scatter buffers, parallel partition finalize driver over tracked `JoinSet` tasks, and `MemoryReservation` handoffs.                                     |
| `datafusion/core/src/morsel/spill.rs`         | Thin wrapper over `SpillManager` / `InProgressSpillFile`, spill-headroom reservation management, partition-scoped `Arc<dyn SpillFile>` manifests, `abort_async` cancellation helper, and `#138` `SpillMetrics` wiring. |
| `datafusion/physical-plan/src/spill/mod.rs`   | Minimal `#[doc(hidden)]` internal re-export of `InProgressSpillFile` (only when Phase 4 lands) if `datafusion/core` invokes incremental async spilling directly.                                                       |
| `datafusion/pruning/src/pruning_predicate.rs` | Optional `#[doc(hidden)]` internal `KeyRangeBitmap` pruning hook (only when Phase 2 lands) if `JoinBloomFilterExpr` exposes a `KeyRangeBitmap` for container pruning alongside `HashTableLookupExpr`.                  |

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
     `LeftAnti` / `RightAnti` / null-aware `NOT IN`, intervening `LEFT` /
     `FULL` joins beneath a downstream `IS NOT DISTINCT FROM` join and beneath
     a downstream null-aware `NOT IN` join (verifying the downstream join
     filter does **not** push past the `LEFT`/`FULL` join onto its probe scan),
     and multi-join chains (TPC-H Q3 and Q12 shapes) with both
     `pushdown_filters = true` and `false`, and with `#167` sub-RG splitting
     enabled.
   - **Config, spill soundness & isolation:** Verify
     `enable_join_dynamic_filter_pushdown = false` disables pushdown cleanly,
     verify spilled builds with dynamic pushdown enabled (both bounds-only
     fallback and optional hash-column bloom replay) drop zero valid matches
     across resident `M` and spilled `S` partitions, and verify
     repeated/concurrent executions of one prepared `MorselExec` never leak
     dynamic filters across runs.
2. **Radix-partitioned hash tables (`datafusion/core/src/morsel/tests.rs` + integration tests):**
   - **Bit-layout unit test:** Verify `radix_partition` yields uniform **both**
     low-bit `h1` bucket-start distribution (`h & ((1 << k) - 1)` for `k <= 16`)
     **and** top-7-bit `h2` tag distribution (`h >> 57`) inside every radix
     partition across level 0 and recursive spill levels (`R0, R1 ∈ 0..=8`),
     plus uniform bloom block spread.
   - **Parity & skew:** High-cardinality (1M+ groups / build rows), low-cardinality
     (fast path `R = 0`), heavy-hitter Zipf skew, empty input, and multi-column
     / string / decimal keys matching stock `off`.
   - **Parallel finalize & memory accounting:** Assert parallel partition
     finalize runs on multiple workers, `MorselMetrics` reports combine/finalize
     time separately, and `MemoryPool::reserved() == 0` after completion,
     error, or early drop.
3. **Out-of-core spilling (`datafusion/core/tests/morsel_spill.rs`):**
   - **Forced-OOM spill parity:** Run `GROUP BY`, `HashJoin` (inner, outer
     `Left`/`Right`/`Full`, semi `LeftSemi`/`RightSemi`, anti
     `LeftAnti`/`RightAnti`, and mark `LeftMark`/`RightMark`, verifying
     exactly-once `LeftSemi`/`LeftAnti`/`LeftMark` emission from merged `P2`
     bitmaps and no premature `P1` build emission), and `Sort` / `TopN` under
     tight [`GreedyMemoryPool`][memory-pools] and
     [`FairSpillPool`][memory-pools] budgets (e.g., 1–8 MiB) on multi-partition
     inputs, asserting exact parity with stock `off` AND `spilled_bytes > 0` /
     `spill_file_count > 0`.
   - **Null-aware `NOT IN` spill/admission parity:** Cover uncorrelated and
     correlated null-aware `LeftAnti`, `LeftMark`, and `RightAnti` with probe
     `NULL`s, build `NULL`s, and empty builds (asserting v1 excludes them from
     radix spill cleanly without wrong results — and, once lifted, verifying
     exact spilled parity against stock `off`).
   - **Skew & recursive spill:** Force a single radix partition to exceed the
     pool (both distinct-key skew triggering recursive radix split and 100%
     identical-key skew triggering `HashJoin` block-nested-loop sub-pass —
     including `RightSemi` deduplication and `Right`/`Full`/`RightAnti`/`RightMark`
     probe-matched bitmap parity — and `GROUP BY` sorted fallback).
   - **Disk quota & cleanup invariants:** Test `DiskManager`
     `max_temp_directory_size` exhaustion (verifying clean `ResourcesExhausted`
     without stock fallback), mid-spill I/O error injection, and early stream
     drop with full output channel — asserting
     `disk_manager.spilling_progress().active_files_count == 0`,
     `disk_manager.spilling_progress().current_bytes == 0`,
     `pool.reserved() == 0`, and `abort_count` tracking using a mock
     `TempFileFactory` (matching `AbortTrackingTempFileFactory` in
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

| Phase       | Proposed follow-up issue                                                        | Depends on             | Scope & acceptance highlight                                                                                                                                                                                                                                                          |
| ----------- | ------------------------------------------------------------------------------- | ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Phase 1** | Dynamic join min/max + small `InList` pushdown into native Parquet probe scans  | `#150`, `#158`, `#167` | `LocalJoinFilterSummary`, `FinalizedJoinFilter` (bounds + `InList`), empty/all-NULL build short-circuit, full NULL/join-type safety matrix, Parquet pruning proofs (zero cross-crate pruning changes).                                                                                |
| **Phase 2** | Cache-resident split-block Bloom filter `RowFilter` pushdown                    | Phase 1                | `SplitBlockBloomFilter` + `JoinBloomFilterExpr` evaluated on decode workers under `pushdown_filters = true`, plus optional `#[doc(hidden)]` `KeyRangeBitmap` pruning hook in `datafusion-pruning`; paired TPC-H SF1 Q3 & Q12 benchmark showing reduced `bytes_scanned` / decode time. |
| **Phase 3** | Radix-partitioned hash tables & parallel finalize for `GROUP BY` and `HashJoin` | `#158`, `#168`         | `radix.rs` with `radix_partition`, adaptive `R = 0 -> 6..8`, Layout A over Arrow/`RowConverter`, parallel `JoinSet` partition finalize, `MemoryPool` transfer accounting, high-cardinality microbenchmark.                                                                            |
| **Phase 4** | Out-of-core spilling for `GROUP BY` and `Sort` via `SpillManager`               | Phase 3, `#159`        | `spill.rs` + `#[doc(hidden)]` `InProgressSpillFile` seam; radix partition spill + recursive/sorted skew fallback for `GROUP BY`; sorted-run spill + `MultiLevelMergeBuilder` for `Sort`; tight-pool parity & zero-leak cleanup tests.                                                 |
| **Phase 5** | Out-of-core `HashJoin` (`ProbeSpill` + `Drained(P1)` child pipeline)            | Phase 3, Phase 4       | Build partition eviction (`M` / `S`) preserving full dynamic filter, `LocalProbeSpill` + `SpilledProbeManifest`, `P2` child pipeline in `G1`, single-partition skew block sub-pass, outer-join unmatched emission, constrained-RAM join benchmark.                                    |

[morsel_pipeline_architecture.md]: morsel_pipeline_architecture.md
[morsel_parquet_adapter.md]: morsel_parquet_adapter.md
[morsel_parquet_splitting.md]: morsel_parquet_splitting.md
[morsel_metrics.md]: morsel_metrics.md
[upstream-scheduling]: https://github.com/apache/datafusion/issues/21719
[upstream-parallelism]: https://github.com/apache/datafusion/pull/2226
[morsel-parquet-lower]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/core/src/morsel/parquet.rs#L55-L201
[parquet-create-morselizer]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/datasource-parquet/src/source.rs#L588-L611
[parquet-buffered]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/datasource-parquet/src/opener/buffered.rs#L22-L116
[dyn-filter]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/physical-expr/src/expressions/dynamic_filters/mod.rs#L57-L398
[shared-bounds]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/physical-plan/src/joins/hash_join/shared_bounds.rs#L274-L320
[partition-bounds]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/physical-plan/src/joins/hash_join/shared_bounds.rs#L50-L115
[shared-bounds-membership]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/physical-plan/src/joins/hash_join/shared_bounds.rs#L123-L176
[preserve-probe-nulls]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/physical-plan/src/joins/hash_join/shared_bounds.rs#L917-L968
[stock-join-allow]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/physical-plan/src/joins/hash_join/exec.rs#L1078-L1126
[gather-filters-pushdown]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/physical-plan/src/joins/hash_join/exec.rs#L1961-L2058
[lr-is-preserved]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/physical-plan/src/joins/hash_join/exec.rs#L2810-L2827
[stream-after-build]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/physical-plan/src/joins/hash_join/stream.rs#L618-L636
[join-float-norm]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/physical-plan/src/joins/utils.rs#L2830-L2860
[not-in-scope]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/physical-plan/src/joins/hash_join/not_in.rs#L64-L85
[not-in-partitioned]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/physical-plan/src/joins/hash_join/not_in.rs#L119-L126
[hash-table-lookup]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/physical-plan/src/joins/hash_join/partitioned_hash_eval.rs#L266-L421
[key-range-bitmap]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/physical-plan/src/joins/key_range_bitmap.rs#L57-L115
[key-range-pruning]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/pruning/src/key_range_bitmap_expr.rs#L20-L56
[hash-lookup-pruning]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/pruning/src/pruning_predicate.rs#L1537-L1567
[pruning-inlist]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/pruning/src/pruning_predicate.rs#L1586-L1845
[rg-pruner]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/datasource-parquet/src/push_decoder.rs#L146-L250
[row-filter-prebuild]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/datasource-parquet/src/row_filter.rs#L475-L514
[row-filter-from-prebuilt]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/datasource-parquet/src/row_filter.rs#L520-L560
[can-push-down]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/datasource-parquet/src/row_filter.rs#L364-L405
[opener-rg-stats]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/datasource-parquet/src/opener/mod.rs#L1358-L1364
[opener-rg-bloom]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/datasource-parquet/src/opener/mod.rs#L1652-L1657
[opener-page-prune]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/datasource-parquet/src/opener/mod.rs#L1707-L1725
[config-max-inlist]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/common/src/config.rs#L1495-L1500
[config-inlist]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/common/src/config.rs#L1939-L1960
[join-type-preserved]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/common/src/join_type.rs#L103-L116
[join-type-empty-build]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/common/src/join_type.rs#L136-L147
[join-type-empty-map]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/common/src/join_type.rs#L159-L164
[hash-random-state]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/common/src/hash_utils.rs#L30-L31
[create-hashes]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/common/src/hash_utils.rs#L1362-L1390
[group-values-primitive]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/physical-plan/src/aggregates/group_values/single_group_by/primitive.rs#L87-L105
[group-values-rows]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/physical-plan/src/aggregates/group_values/row.rs#L32-L108
[join-hash-map]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/physical-plan/src/joins/join_hash_map.rs#L40-L110
[mem-clone-id]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/execution/src/memory_pool/mod.rs#L304-L312
[mem-consumer]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/execution/src/memory_pool/mod.rs#L304-L345
[mem-try-grow]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/execution/src/memory_pool/mod.rs#L467-L476
[mem-take]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/execution/src/memory_pool/mod.rs#L507-L512
[memory-pools]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/execution/src/memory_pool/pool.rs#L64-L170
[fair-spill-try-grow]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/execution/src/memory_pool/pool.rs#L223-L258
[spill-manager]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/physical-plan/src/spill/spill_manager.rs#L27-L100
[spill-manager-abort]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/physical-plan/src/spill/spill_manager.rs#L155-L162
[spill-read-unbuffered]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/physical-plan/src/spill/spill_manager.rs#L199-L209
[spill-manager-abort-test]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/physical-plan/src/spill/spill_manager.rs#L248-L320
[in-progress-spill]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/physical-plan/src/spill/in_progress_spill_file.rs#L27-L100
[in-progress-async]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/physical-plan/src/spill/in_progress_spill_file.rs#L103-L152
[in-progress-abort]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/physical-plan/src/spill/in_progress_spill_file.rs#L219-L232
[spill-reader-stream]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/physical-plan/src/spill/mod.rs#L56-L86
[gc-view-arrays]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/physical-plan/src/spill/mod.rs#L852-L865
[spill-file]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/execution/src/spill_file.rs#L17-L95
[disk-spilling-progress]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/execution/src/disk_manager.rs#L303-L309
[refcounted-tmp]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/execution/src/disk_manager.rs#L396-L435
[refcounted-drop]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/execution/src/disk_manager.rs#L437-L453
[stock-agg-spill]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/physical-plan/src/aggregates/spill.rs#L49-L120
[stock-sorter]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/physical-plan/src/sorts/sort.rs#L250-L315
[stock-sorter-spill-inmem]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/physical-plan/src/sorts/sort.rs#L445-L465
[sorted-spill-file]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/physical-plan/src/sorts/streaming_merge.rs#L50-L65
[streaming-merge-builder]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/physical-plan/src/sorts/streaming_merge.rs#L76-L115
[fan-in-headroom]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/physical-plan/src/sorts/multi_level_merge.rs#L34-L55
[multi-level-merge]: https://github.com/robtandy/datafusion/blob/18425fd6c2da7c4813f7a27ca0f68553bb36cac9/datafusion/physical-plan/src/sorts/multi_level_merge.rs#L149-L175
