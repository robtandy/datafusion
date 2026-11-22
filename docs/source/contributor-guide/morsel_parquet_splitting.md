---
orphan: true
---

<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Design: intra-row-group Parquet morsels

This is a follow-up design for the fork's experimental native Parquet source,
not an implementation or a change to the stock reader. The [skew benchmark]
separates two problems: dynamic assignment can spread a large file's many row
groups across workers, but four row groups still yield at most four independent
readers. Give workers independent **row ranges**
inside large groups without making each worker read a copy of the group.

[DuckDB's Parquet source] assigns a row-group index from shared scan state and
then initializes and reads it with local state. The [DuckDB performance guide]
also identifies row groups as the unit of Parquet parallelism. Reuse the
claim-next/local-reader pattern, not its interfaces; subdivisions inside a
Parquet row group are the proposed extension, not a claim about what DuckDB
does. In this fork, [the native scheduler] receives buffered readers from file
lanes. [The buffered planner] calls `try_next_reader`, drives `NeedsData` by
fetching byte ranges, and publishes one reader for each surviving row group.
This is the handoff to subdivide. Stock file-range repartitioning does not
subdivide the row groups themselves.

## Row ranges and decoder integration

1. Add an opt-in strategy to the native buffered Parquet path. Keep existing
   defaults and admission rules. After footer/schema adaptation and row-group
   pruning, obtain the actual file leaf projection and lazily enumerate
   `[lo, hi)` in **original row-group row coordinates**, including the final
   partial range. Start with 128 Ki rows per range, only split groups of at
   least 1 Mi rows, and keep the number of active file lanes bounded. No public
   change to `ParquetAccessPlan` is needed. Start with flat fixed-width columns
   and a usable Offset Index for every leaf the decoder will read. Skip splitting
   metadata-only scans and groups with excessive decode duplication. Start by
   rejecting groups when the sum of page and dictionary bytes requested across
   slices exceeds twice the unique requested bytes (for example, one giant
   page per column). This is a heuristic, not an I/O limit.
2. Preserve [`ParquetAccessPlan`] as the logical per-group read/prune plan.
   For each surviving entry, create a `RowSelection` covering the proposed
   range, including skips before and after it so the selection spans the
   original group's row count. Intersect it with the entry's existing
   row-group-local selection (if any) and omit empty intersections. Do not
   slice the already-filtered row stream or repartition based on the number of
   matches. The resulting selections are disjoint and their union is the
   selection the unsplit scan would have used. Pruning remains inexact; retain
   the exact SQL filter in the native pipeline. Leave decoder row-filter
   pushdown off in the first version.
3. Build one [`ParquetPushDecoder`] per published slice, sharing
   `ArrowReaderMetadata`, the adapted projection, and batch-size settings. Use
   `with_row_group_selections(vec![RowGroupSelection::new(rg, Some(selection))])`,
   not the whole-file `with_row_selection` API. Pass exactly the requested
   ranges and their `Bytes` to `push_ranges` when `try_next_reader` returns
   `NeedsData`. A decoder is private mutable state; workers never share one,
   and each drains only its own reader. The opener's [decoder configuration]
   is the place to reuse builder options, and the file continuation advances
   the slice cursor before it advances to the next row group. The current
   `into_builder` mechanism works at **row-group boundaries**; it does not
   turn an in-progress reader into independent intra-group readers.

Keep ordered, limited, encrypted, or otherwise unsupported scans on their
existing path. Missing optional indexes are a normal reason to use the
unsplit reader for that group, not an error or an engine fallback. Malformed
metadata or I/O failures follow the existing reader error path; do not retry an
unsplit group after any of its slices have been published.

## Page boundaries and byte ownership

The [Parquet Page Index spec] separates the `ColumnIndex` (statistics for
predicate pruning) from the `OffsetIndex` (data page byte locations and
first-row positions). Only the latter is required to split. Extend the opener's
[index loading decision] to consider large candidate groups independently of
whether a predicate requires page statistics; load just the needed Offset
Indexes when possible and reuse cached metadata. Respect disabled page-index
reads by using the existing path. Validate usable page locations against the
row-group row count and column-chunk extent before deciding to split.

- **Fixed row boundaries with indexes (first implementation).** Pages are
  column-local: even if the split aligns with one column's page, it may cut
  through another's. [`RowSelection::scan_ranges`] maps the same logical
  selection to physical ranges for each indexed leaf. The [parquet-rs fetch
  logic] requests the relevant full pages plus the dictionary prefix, if
  present. A boundary page can be needed by multiple slices, but only the
  selection's rows are emitted. Workers may decompress such a page or
  dictionary independently; measure that CPU cost separately from I/O.
- **Page-aligned boundaries with indexes (optional refinement).** Prefer a
  common boundary across projected leaves near the target if one exists, or
  choose a nearby boundary that minimizes the cost of duplicated pages in the
  widest leaves. Keep a minimum/maximum morsel size so wildly differing page
  layouts cannot collapse back to a single giant morsel. Do not independently
  align the logical split for each column: the columns would represent
  different rows. Repeated/nested leaves need explicit tests for rows crossing
  page boundaries and are not part of the first implementation.
- **Absent or partial Offset Index (unsplit in v1).** A row selection alone
  still makes parquet-rs request a whole chunk for each unindexed projected
  leaf. Sharing full compressed chunks avoids repeated storage reads, but
  pins the entire projected group and can repeatedly parse, skip or decompress
  its prefix. It is not an acceptable default. A later phase could build a
  temporary offset map by walking page headers once, with bounded I/O and an
  explicit compatibility gate (a V1 page's value count is not a row count for
  repeated columns). Alternatively, decode once sequentially and distribute
  decoded batches to downstream work; that does not parallelize decoding, but
  avoids read amplification and repeated skipping without a synthesized index.

Use a **row-group-local range broker** in the file planner, not a process-wide
cache. Key backing buffers by immutable file identity and byte interval; a
broker belongs to one file snapshot and one group. Work through slices in row
order per group and calculate the last consumer from each page's row span
and the last nonempty slice intersecting it (the last selected slice for a
needed dictionary). Keep only the fetched intervals still needed by later
slices in the broker. Serve `NeedsData` cache hits from `Bytes`
views; only read missing intervals, deduplicating in-flight requests as well as
completed ones. Merge adjacent misses into bounded requests, then give the
decoder views with exactly the offsets and lengths it requested. Do not fetch
unused gaps or issue one full projected-group request per slice. When the last
consumer has received a view, drop the broker reference; live reader views keep
the allocation alive until their last owner drops. Keep dictionary bytes shared
through the last slice that needs them. The planner can remain the sole owner
of the broker and the `AsyncFileReader` in v1; no shared lock is held across
storage I/O or worker polling. Restrict any one lane to one broker frontier at
a time; readers from a prior group can finish using their owned views.

## Bounds, cancellation and metrics

- Retain the scheduler's current four file lanes, two pending I/O phases, two
  ready-reader slots, and one claimed reader per worker. Keep the two-slot
  output queue and batch-level yielding. Create at most one next-slice
  descriptor per lane; no task, reader or future per slice up front.
- Start experiments with a 256 MiB query-wide **compressed split-buffer** cap
  and a 16 MiB per-slice compressed estimate (including dictionary and shared
  pages). Charge object-store requests before issuing them, and retain the
  reservation with the backing allocation through I/O, the broker, ready
  readers, and worker decode. Do not count each `Bytes` view as a new
  allocation, or release the reservation when only the broker drops its view.
  Use an execution memory-pool reservation as well as a split-specific cap.
  Bound coalescing (start at 1 MiB, except for an individual larger page within
  the per-slice cap); account for the full returned buffer, not just its views.
- Check feasibility before publishing a group: bound the current slice plus the
  still-needed boundary/dictionary cache, not the sum of all slices' data.
  If an individual group cannot fit the split policy, use the existing reader
  before publishing any of its slices. Under transient pressure, let workers
  drain published work. Reserve the full incremental footprint before starting
  the read, without holding a ready slot or I/O permit while waiting for memory.
  Reserve frontier headroom for each admitted lane up front (or reduce the
  number of lanes); otherwise all lanes could pin shared pages and wait for
  more memory that none can release. A memory-pool denial that cannot make
  progress must return an error, not wait on its own reservation.
- Compressed bounds are not decoded-memory bounds. Limit concurrent decoders
  and emitted Arrow batches as above, and estimate their workspace separately
  from the split cap. In v1 limit the supported projection to flat fixed-width
  data. Start with a 128 MiB eligibility ceiling per split reader, estimated
  conservatively from fixed-width batches, level buffers and the footer's total
  uncompressed sizes for projected chunks. Reserve the estimate in the query
  memory pool while the reader is live. This can overestimate scratch space;
  a tighter estimate would need page-header information, not just the Offset
  Index. Use the legacy reader for unsupported or oversized groups. Arrow's
  internal allocations, object-store transient copies and buffers held by the
  consumer are not hard-capped by these reservations. A hard cap for decoder
  internals needs allocator hooks or an equivalent parquet-rs contract; do not
  advertise this cap for the legacy reader either.
- Tie file continuations, broker requests, reservations and worker tasks to
  the existing stream. Dropping the stream or receiving a fatal error stops
  issuing slices, aborts owned pending work, drops queued readers and releases
  reservations, including any wait for memory, data or a channel permit. A
  synchronous Arrow poll may finish before the drop is observed; check
  cancellation between batches and before issuing further reads. Report an I/O
  or decode error with file, row-group and row-range context; never replay a
  partially emitted group.
- Keep [per-execution morsel metrics] for claims/completions by worker and
  I/O wait, decode work and queue depth. Add rows selected per slice,
  split/unsplit groups and reasons, merged storage requests and actual data
  bytes returned (separate footer/index bytes), page/dictionary reuse bytes,
  reservation wait and peak **unique** compressed backing bytes. Keep logical
  row-group progress credit once per group, not once per slice. Record both
  per-worker slices and rows: slice counts alone do not show balanced decode
  time. Account for live buffers outside the ready queue in the new byte peak.

## Follow-up validation

1. Unit-test that every original selected row belongs to exactly one slice,
   including fully pruned groups, sparse selections, misaligned page boundaries,
   zero-row input and a short final slice. Cover dictionaries, nulls, mismatched
   page sizes by column, and missing/partial indexes. Confirm nested and other
   unsupported layouts produce the legacy fallback, not silently wrong rows.
2. Use a recording `AsyncFileReader` or object store with indexed fixtures to
   assert that a shared boundary page/dictionary is read once, unrelated groups
   and unprojected leaves are not read, and storage reads are the union of
   needed intervals. Test overlapping reads, a page over the cap, and tiny
   budgets with several lanes; assert progress or a resource error rather than
   deadlock. Check the cap covers ready and claimed readers, not just the cache.
3. Compare exact unordered projected rows (including nulls and IDs) and filtered
   counts with stock execution. Exercise one/two large groups with eight workers,
   mixed split and unsplit groups, and a page-pruned group. Gate the test's worker
   claim order with a barrier if asserting parallel claims; do not assert an
   exact scheduler distribution. Verify fallback paths and default settings.
4. Drop/fail the stream while a lane is fetching, waiting on memory or waiting
   to publish, and while a worker is blocked on output. Assert that the query
   terminates, pending I/O is dropped, reservations return to zero and metrics
   reflect partial runs. Inject a storage error after a successful slice to
   catch accidental replay.
5. Re-run the [skew benchmark] on eight workers, first unchanged as a no-index
   control, then with indexed versions of both layouts. Compare stock, legacy
   native and split native on the same files. Record paired warm-cache elapsed
   times, claim/row distributions, data and metadata I/O, peak reservations,
   and decode work; add a request-latency test before choosing coalescing defaults
   for remote storage. The large-row-group case should offer more than four
   claimable readers without multiplying data bytes by the number of slices;
   the file-skew control should not need splitting to spread its row groups.

[skew benchmark]: https://github.com/robtandy/datafusion/tree/morsel-execution/benchmarks/morsel_parquet_skew
[DuckDB's Parquet source]: https://github.com/duckdb/duckdb/blob/e55f794cae98443cfaf302f41ce8a83a95a7872a/extension/parquet/parquet_multi_file_info.cpp#L592-L612
[DuckDB performance guide]: https://duckdb.org/docs/current/guides/performance/file_formats.html#parquet
[the native scheduler]: https://github.com/robtandy/datafusion/blob/c2809d3b3ee73936b182429e1646af290d797c6e/datafusion/core/src/morsel/parquet.rs
[The buffered planner]: https://github.com/robtandy/datafusion/blob/c2809d3b3ee73936b182429e1646af290d797c6e/datafusion/datasource-parquet/src/opener/buffered.rs
[`ParquetAccessPlan`]: https://github.com/robtandy/datafusion/blob/c2809d3b3ee73936b182429e1646af290d797c6e/datafusion/datasource-parquet/src/access_plan.rs
[`ParquetPushDecoder`]: https://docs.rs/parquet/60.1.0/parquet/arrow/push_decoder/struct.ParquetPushDecoder.html
[decoder configuration]: https://github.com/robtandy/datafusion/blob/c2809d3b3ee73936b182429e1646af290d797c6e/datafusion/datasource-parquet/src/push_decoder.rs#L61-L100
[Parquet Page Index spec]: https://github.com/apache/parquet-format/blob/master/PageIndex.md
[index loading decision]: https://github.com/robtandy/datafusion/blob/c2809d3b3ee73936b182429e1646af290d797c6e/datafusion/datasource-parquet/src/opener/mod.rs#L1391-L1529
[`RowSelection::scan_ranges`]: https://docs.rs/parquet/60.1.0/parquet/arrow/arrow_reader/struct.RowSelection.html#method.scan_ranges
[parquet-rs fetch logic]: https://docs.rs/crate/parquet/60.1.0/source/src/arrow/in_memory_row_group.rs
[per-execution morsel metrics]: morsel_metrics.md
