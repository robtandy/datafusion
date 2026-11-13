<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Operator performance

`benches/operator_micro.rs` measures the important operators of the pipeline
engine in isolation, on mocked inputs: synthetic `ArrayRef`s and `DataChunk`s,
without SQL and without a planner. This document records, for every
configuration of its parameter sweeps, what an operator costs per row, how
close it comes to the memory-bandwidth floor of the bytes it must touch, where
its time goes (`perf`), and the optimizations that this implies. The
[query profiles](profiling.md) tell in which operators a query spends its
time; these numbers tell what makes an operator slow, and what it would take
to make it fast.

In short, on one thread of a 16-core Neoverse-N1 machine:

- The streaming operators are within 2x to 10x of the floor: a projection
  that computes `a + b` runs at 54% of it, a filter at 10% to 32%, half of
  whose time builds and checks its selection vectors. A projection over
  chunks with selection vectors gathers all of its columns, also those it
  passes through.
- The pipeline breakers are 10x to 1000x off the floor (TopKs, which keep few
  rows, 3x to 40x). The probe of a hash join against 1Mi build rows takes
  185 ns per row (0.6% of the floor): every match costs three dependent cache
  misses, one of them in a chain array apart from the rows. The build takes 57 ns per row, 65% of it inserting the rows
  into pointer tables larger than the L2 cache. An aggregation of 64Ki groups
  takes 58 ns per row in a hash table that no longer fits in the L2 cache.
- Sorts are fast on one integer key (120 ns per row for 2Mi random rows) but
  6.5x slower on two keys (781 ns per row) and 4.3x slower on strings (511 ns
  per row), which they compare with `memcmp` calls on row-format keys. A sort
  merges its runs on one thread: it is only 1.7x faster on 16 threads.
- Windows sort their partitions the same way (746 ns per row for a
  `ROW_NUMBER()`), and evaluate running aggregates row by row with 4
  allocations per row (1140 ns per row for a running `SUM`).

The [optimizations](#optimizations-they-imply) at the end are filed as
follow-up issues; none is implemented here.

## Running the benchmark

```text
cargo test --profile release-nonlto -p datafusion-physical-pipeline --bench operator_micro -- --nocapture
```

runs every configuration of every operator (113 of them, in about 4 minutes
after the build), prints a line per configuration as it goes, then the
measured bandwidths and the results, as a text table and as the Markdown table
of [Results](#results). Debug builds run at 1% of the sizes, once.
Environment variables select and scale the configurations (see the module
documentation of the benchmark):

| Variable                           | Meaning                                                                                                | Default (optimized / debug)  |
| ---------------------------------- | ------------------------------------------------------------------------------------------------------ | ---------------------------- |
| `OPERATOR_MICRO_SCALE`             | factor of the input sizes                                                                              | `1` / `0.01`                 |
| `OPERATOR_MICRO_ITERATIONS`        | timed executions of every configuration                                                                | `5` / `1`                    |
| `OPERATOR_MICRO_THREADS`           | thread counts of the partition sweeps, comma-separated                                                 | `1,4,16` (at most the cores) |
| `OPERATOR_MICRO_FILTER`            | substrings separated by `\|`: only the configurations whose `operator configuration ` contains one run | all                          |
| `OPERATOR_MICRO_SAVE`              | a file to write the results to (tab-separated)                                                         | none                         |
| `OPERATOR_MICRO_COMPARE`           | a file of earlier results to compare `ns/row` with                                                     | none                         |
| `OPERATOR_MICRO_BANDWIDTH_MB`      | the MiB of the buffers of the bandwidth measurement                                                    | `512` / `16`                 |
| `OPERATOR_MICRO_COUNT_ALLOCATIONS` | whether to count the allocations                                                                       | `true`                       |

To measure an optimization, save the results of the base and compare the
change with them:

```text
OPERATOR_MICRO_SAVE=/tmp/before.tsv cargo test --profile release-nonlto -p datafusion-physical-pipeline --bench operator_micro -- --nocapture
OPERATOR_MICRO_COMPARE=/tmp/before.tsv cargo test --profile release-nonlto -p datafusion-physical-pipeline --bench operator_micro -- --nocapture
```

The comparison lists the `ns/row` of every configuration before and after, and
the median and largest changes.

## Method

Every configuration builds an operator tree
`DiscardSink(operator(ChunkSource...))` and runs it with the `Executor` on 1,
4 or 16 threads, with profiling on (`Executor::with_profiling`). A
`ChunkSource` hands out pre-built chunks to the threads of its pipeline
through a shared cursor; every chunk has buffers of its own, as the batches of
a scan have. The `DiscardSink` drops the chunks of the operator. The numbers
are those of the `QueryProfile` of the execution, with the metric names of the
[query profiles](profiling.md), so that they line up with them:

- **What counts.** For a projection, a filter and the probe of a hash join:
  the stage of the operator in its pipeline. For the build of a hash join: the
  pipelines whose sink the join is (its `sink` calls, and the
  `JoinHashTableBuild` pipeline that builds its pointer tables) and its
  `finalize_time`. For the other pipeline breakers (aggregation, sort, TopK,
  window): all of their stages and their finalization. For the Parquet decode:
  the `ParquetScanSource`. The harness operators never count.
- **`rows`**: the `rows_in` of the operator, its logical rows (the selected
  rows of chunks with selection vectors), or the `rows_out` of a source.
- **`ns/row`**: `self_time / rows`, the time of the operator per row, summed
  over its threads: its CPU time.
- **`Mrows/s`**: `rows` per second of the wall time of the operator: the
  `elapsed_time` of every pipeline it is measured in, times the share of the
  `active_time` of the pipeline that its stages account for, plus its
  `finalize_time`.
- **`allocs/chunk`** and **`copied B/row`**: the allocations of the operator
  per chunk of its input (`morsels_in`), and the bytes it allocated per row.
  The global allocator of the benchmark counts every `alloc`, `alloc_zeroed`
  and `realloc` (with its new size) of all threads, in per-thread shards,
  before it calls the system allocator. Executions of the same plan without
  the measured part (the sources into a `DiscardSink`, and for the probe the
  join with an empty probe side) count the allocations of the harness, which
  are subtracted. Every value that an operator copies (gathers, computes,
  scatters into rows, encodes into keys, inserts into tables) lands in memory
  that it allocated, so these are the bytes it copies, and an upper bound of
  them: capacity that it does not fill (e.g. of hash tables) counts too.
- **`in B/row`** and **`out B/row`**: `bytes_in / rows` and
  `bytes_out / rows` (see `profiling::chunk_bytes`). For tuple-data ingestion,
  `out` is the size of the collections (`memory_size`).
- **`memory_peak`**: the high-water mark of the memory the operator reserved.
- **`floor B/row`** and **`% floor`**: see [The floor](#the-floor).
- **`cv %`**: the coefficient of variation of `ns/row` over the timed
  executions.
- **`sink/fin/src %`**: the shares of the time of a pipeline breaker spent as
  a sink, finalizing and as a source (for the build of a hash join, the source
  is the `JoinHashTableBuild` that builds the pointer tables).

The first execution of every configuration is a warm-up that checks the
output of the operator: its rows, the order of the keys of sorts and TopKs and
the first `k` keys of TopKs, and the counts of the groups of aggregations. The
numbers are the medians of the 5 timed executions that follow.

Tuple-data ingestion is a data structure rather than an operator: its
benchmark appends the chunks to `PartitionedTupleData`s (with a hash slot, as
the build of a hash join does) directly, one per thread, timing every thread.

Counting the allocations costs 1% to 4% of `ns/row` in most configurations,
and 13% in the one that allocates most (the running `SUM` of a window on
sorted input, 4 allocations per row), measured with
`OPERATOR_MICRO_COUNT_ALLOCATIONS=false`.

### Sweeps

Every benchmark has a default configuration, of 1 thread and chunks of 2048
rows (`DEFAULT_VECTOR_SIZE`), and sweeps one parameter at a time:

| Benchmark         | Default configuration                                                                                                                                                                 | Sweeps                                                                                                                                         |
| ----------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| `projection`      | `SELECT a + b, c` over 4Mi rows of 3 `Int64` columns                                                                                                                                  | expressions (`c, a`; `a + b, a - b, a * b`), selectivity of selection vectors on the input (0.5, 0.1), batch size (256, 1024), threads         |
| `filter`          | `WHERE a < x` that selects half of 4Mi rows of `a`, `b`, `c` (`Int64`) and `s` (`Utf8View`)                                                                                           | selectivity (0.001 to 1), predicate (`a < x AND b < y`; `s < x`), batch size, threads                                                          |
| `hash_join_build` | inner join of 1Mi build rows of distinct `i64` keys and an `Int64` payload, probed by one chunk                                                                                       | build rows (64Ki, 4Mi), rows per key (4, 64), key type, threads                                                                                |
| `hash_join_probe` | 4Mi probe rows against that build side, of which half find a match                                                                                                                    | hit ratio (0 to 1), build rows (2Ki, 64Ki, 4Mi), 4 matches per hit, key type, batch size, threads                                              |
| `aggregate`       | `SELECT k, SUM(v), COUNT(*) GROUP BY k` (in two stages, as DataFusion plans it on several partitions) over 4Mi rows of 1Ki groups                                                     | groups (16, 64Ki, 1Mi, 4Mi), key type, batch size, threads                                                                                     |
| `sort`            | `ORDER BY k` of 2Mi rows of distinct `i64` keys (and a payload), in a random order                                                                                                    | order (sorted, reverse, nearly sorted: 1% of the rows swapped), key type, distinct keys (16, 64Ki), batch size 256, threads                    |
| `top_k`           | `ORDER BY k LIMIT 100` of 4Mi rows of distinct keys, in a random order                                                                                                                | `k` (10, 10Ki, 100Ki), order (sorted, reverse), `utf8v16` keys, threads                                                                        |
| `window`          | `SUM(v) OVER (PARTITION BY p ORDER BY o)` (a running sum, `RANGE` frame) of 2Mi rows of 1Ki window partitions, in a random order                                                      | function (`ROW_NUMBER()`; `SUM(v) OVER (PARTITION BY p)`), window partitions (1: no `PARTITION BY`, 16, 64Ki), input sorted by `p, o`, threads |
| `tuple_data`      | `PartitionedTupleData::append_chunk` of 4Mi rows of 4 `Int64` columns, one partition                                                                                                  | columns (`i64`; `i32, i64, f64`; `i64, utf8v16`), selectivity of the input (0.5), radix partitions (16, 256), batch size, threads              |
| `parquet_decode`  | `ParquetScanSource` over 4Mi rows of one `Int64` column of distinct values, in 16 uncompressed files of row groups of 64Ki rows in an in-memory object store, decoder batches of 8192 | column type (`i32`, `utf8v16`), distinct values (16, 64Ki), codec (Snappy, Zstandard level 1), batch size (2048, 32768), threads               |

The key types are made of ids (distinct ids have distinct keys, in the order
of the ids): `i32` is the id, `i64` the id times `0x9e3779b9` (spread over the
range of `Int64`, so that a hash join hashes it), `i64dense` the id (a hash
join indexes it directly, see `KeyRange`), `2xi64` the two `Int64` columns
`id >> 8` and `id & 0xff`, and `utf8v16` the id as 16 decimal digits,
zero-padded (longer than the 12 bytes that a view inlines, with a common
prefix). The keys of the probe rows that do not find a match lie in the range
of the build keys. The number of partitions is swept as the number of threads
(the target partitions) of the executions, and as the radix partitions of
tuple-data ingestion. The batch size is the number of rows of the chunks (at
most 2048), and for the Parquet decode the batch size of the decoder
(`datafusion.execution.batch_size`), whose batches the scan slices into chunks
of 2048 rows.

## The floor

The floor of a configuration is the time it takes at least to move the bytes
that the operator must read and write, whatever its implementation, at the
memory bandwidth of the machine on as many threads: `% floor` is that time
over the wall time of the operator. The benchmark measures the bandwidth
first, with streaming kernels over buffers of 512 MiB (16 times the last-level
cache), every thread its own buffers, the best of 5 passes (GB/s, in the
second run of [Results](#results)):

| Threads |  Read | Write | Copy (read + write) |
| ------: | ----: | ----: | ------------------: |
|       1 |  19.9 |  37.8 |                31.5 |
|       4 |  72.3 | 114.6 |               107.5 |
|      16 | 130.1 | 148.7 |               151.0 |

The floor time is the largest of three lower bounds: reading the bytes it must
read at the read bandwidth, writing the bytes it must write at the write
bandwidth, and moving both at the bandwidth of the copy (which reads and
writes). The bytes that an operator must read and write are:

| Benchmark         | Must read                                                                                                             | Must write                                                                   |
| ----------------- | --------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------- |
| `projection`      | the cache lines of the values of the columns of its computed expressions (`a` and `b`)                                | its computed columns (8 bytes per row each); passed-through columns: nothing |
| `filter`          | the column of its first conjunct; the cache lines of the column of the second one of the rows that pass the first one | its selection vectors (4 bytes per selected row)                             |
| `hash_join_build` | the build rows: their keys and payload                                                                                | the build rows, once                                                         |
| `hash_join_probe` | the probe keys; for every output row, the other columns of its probe row and of its build row                         | the output rows                                                              |
| `aggregate`       | the keys and the argument of `SUM`                                                                                    | every group (keys, sum and count), once                                      |
| `sort`            | all columns                                                                                                           | all columns, sorted, once                                                    |
| `top_k`           | the key column                                                                                                        | the `k` output rows                                                          |
| `window`          | all columns                                                                                                           | all columns and the value of the window function (8 bytes per row), once     |
| `tuple_data`      | the cache lines of the (selected) values                                                                              | the values, once                                                             |
| `parquet_decode`  | the files                                                                                                             | the decoded column                                                           |

The assumptions of the floor:

- **Every byte crosses the memory bus once.** The inputs of the streaming
  operators, probes, aggregations, sorts, TopKs, windows and tuple-data
  ingestion are 32 MiB to 224 MiB, larger than the 32 MiB last-level cache, so
  they are read from memory (the build sides of 64Ki and 1Mi rows, of 1 MiB
  and 16 MiB, and some Parquet files may stay in it between executions). The
  outputs are counted at the bandwidth of memory too, although the
  `DiscardSink` drops them chunk by chunk (in a cache) and a pipeline would
  hand them to the next operator.
- **A value costs its cache line.** An operator that reads some rows of a
  column reads the 64-byte lines that hold them: at a selectivity of 10%, a
  projection reads 57% of the lines of an `Int64` column.
- **No random access is counted.** Hash tables, the rows of a build side and
  sort runs are touched at random positions, each access costs a cache line
  and, for structures larger than the caches, a memory latency. The floor
  counts none of these, so it is out of reach for the hash joins and
  aggregations whose tables exceed the caches: the [profiles](#profiles)
  count their misses instead.
- **The bandwidth is that of this process on a shared virtual machine.** The
  write kernel on 16 threads measured 149 GB/s in one run and 257 GB/s in the
  other: it ends before its last writes leave the caches, so that its bound is
  loose, and the floors on 16 threads are set by the read or copy bounds.

## Results

The second of two consecutive runs on a 16-core Neoverse-N1 virtual machine
(`release-nonlto`, 5 timed executions per configuration, other processes
running; see [Variance](#variance)):

| operator        | configuration                                                 |   rows | ns/row | Mrows/s | allocs/chunk | copied B/row | in B/row | out B/row | memory_peak | floor B/row | % floor | cv % | sink/fin/src % |
| --------------- | ------------------------------------------------------------- | -----: | -----: | ------: | -----------: | -----------: | -------: | --------: | ----------: | ----------: | ------: | ---: | -------------: |
| projection      | `exprs=a+b,c sel=1.00 batch=2048 threads=1`                   |  4.19M |   1.47 |     677 |         5.00 |         8.14 |     24.0 |      16.0 |           0 |        24.0 |    54.4 |  0.8 |              - |
| projection      | `exprs=c,a sel=1.00 batch=2048 threads=1`                     |  4.19M |  0.070 |   14234 |         1.00 |        0.016 |     24.0 |      16.0 |           0 |           0 |       - |  2.3 |              - |
| projection      | `exprs=a+b,a-b,a*b sel=1.00 batch=2048 threads=1`             |  4.19M |   3.38 |     295 |         11.0 |         24.3 |     24.0 |      24.0 |           0 |        40.0 |    37.5 |  0.6 |              - |
| projection      | `exprs=a+b,c sel=0.50 batch=2048 threads=1`                   |  2.10M |   5.87 |     170 |         14.0 |         32.8 |     24.0 |      16.0 |           0 |        39.9 |    27.2 |  0.6 |              - |
| projection      | `exprs=a+b,c sel=0.10 batch=2048 threads=1`                   | 419.2K |   27.2 |    36.6 |         14.0 |         35.9 |     24.0 |      16.0 |           0 |        99.1 |    16.7 |  0.6 |              - |
| projection      | `exprs=a+b,c sel=1.00 batch=256 threads=1`                    |  4.19M |   4.14 |     240 |         5.00 |         9.16 |     24.0 |      16.0 |           0 |        24.0 |    19.3 |  0.9 |              - |
| projection      | `exprs=a+b,c sel=1.00 batch=1024 threads=1`                   |  4.19M |   1.99 |     499 |         5.00 |         8.29 |     24.0 |      16.0 |           0 |        24.0 |    40.1 |  1.0 |              - |
| projection      | `exprs=a+b,c sel=1.00 batch=2048 threads=4`                   |  4.19M |   1.67 |    2240 |         5.01 |         8.14 |     24.0 |      16.0 |           0 |        24.0 |    50.0 |  2.6 |              - |
| projection      | `exprs=a+b,c sel=1.00 batch=2048 threads=16`                  |  4.19M |   2.38 |    5242 |         5.02 |         8.14 |     24.0 |      16.0 |           0 |        24.0 |    83.3 | 11.6 |              - |
| filter          | `pred=a<x sel=0.5 batch=2048 threads=1`                       |  4.19M |   2.92 |     341 |         10.0 |         2.38 |     56.0 |      28.0 |           0 |       10.00 |    13.7 |  0.5 |              - |
| filter          | `pred=a<x sel=0.001 batch=2048 threads=1`                     |  4.19M |   1.40 |     714 |         9.89 |        0.394 |     56.0 |     0.058 |           0 |        8.00 |    28.7 |  0.8 |              - |
| filter          | `pred=a<x sel=0.01 batch=2048 threads=1`                      |  4.19M |   1.48 |     676 |         10.0 |        0.419 |     56.0 |     0.562 |           0 |        8.04 |    27.2 |  0.8 |              - |
| filter          | `pred=a<x sel=0.1 batch=2048 threads=1`                       |  4.19M |   1.82 |     547 |         10.0 |        0.779 |     56.0 |      5.60 |           0 |        8.40 |    22.0 |  0.6 |              - |
| filter          | `pred=a<x sel=0.9 batch=2048 threads=1`                       |  4.19M |   4.14 |     241 |         10.0 |         3.98 |     56.0 |      50.4 |           0 |        11.6 |     9.7 |  0.3 |              - |
| filter          | `pred=a<x sel=1 batch=2048 threads=1`                         |  4.19M |   1.25 |     798 |         8.01 |        0.352 |     56.0 |      56.0 |           0 |        12.0 |    32.1 |  0.8 |              - |
| filter          | `pred=a<x&b<y sel=0.5 batch=2048 threads=1`                   |  4.19M |   4.28 |     233 |         19.1 |         2.85 |     56.0 |      28.0 |           0 |        18.0 |    18.7 |  0.6 |              - |
| filter          | `pred=utf8<x sel=0.5 batch=2048 threads=1`                    |  4.19M |   11.1 |    90.2 |         16.0 |         10.5 |     56.0 |      28.0 |           0 |        34.0 |    14.5 |  0.2 |              - |
| filter          | `pred=a<x sel=0.5 batch=256 threads=1`                        |  4.19M |   8.10 |     123 |         10.0 |         4.16 |     56.0 |      28.0 |           0 |       10.00 |     4.9 |  4.6 |              - |
| filter          | `pred=a<x sel=0.5 batch=1024 threads=1`                       |  4.19M |   3.80 |     262 |         10.0 |         2.63 |     56.0 |      28.0 |           0 |       10.00 |    10.5 |  0.5 |              - |
| filter          | `pred=a<x sel=0.5 batch=2048 threads=4`                       |  4.19M |   3.89 |     998 |         10.0 |         2.38 |     56.0 |      28.0 |           0 |       10.00 |    11.0 |  0.3 |              - |
| filter          | `pred=a<x sel=0.5 batch=2048 threads=16`                      |  4.19M |   3.87 |    3471 |         10.0 |         2.38 |     56.0 |      28.0 |           0 |       10.00 |    23.0 |  1.8 |              - |
| hash_join_build | `build=1M dups=1 key=i64 threads=1`                           |  1.05M |   56.6 |    17.7 |         8.18 |         80.1 |     16.0 |         0 |      60.34M |        32.0 |     1.8 |  1.0 |        35/0/65 |
| hash_join_build | `build=64K dups=1 key=i64 threads=1`                          |  65.5K |   59.0 |    16.9 |         2.41 |          122 |     16.0 |         0 |       4.56M |        32.0 |     1.7 |  0.7 |        35/2/63 |
| hash_join_build | `build=4M dups=1 key=i64 threads=1`                           |  4.19M |   62.9 |    15.9 |         8.46 |         74.2 |     16.0 |         0 |     222.89M |        32.0 |     1.6 |  1.2 |        32/0/68 |
| hash_join_build | `build=1M dups=4 key=i64 threads=1`                           |  1.05M |   81.5 |    12.3 |         22.4 |          104 |     16.0 |         0 |      55.88M |        32.0 |     1.2 |  1.2 |        24/0/76 |
| hash_join_build | `build=1M dups=64 key=i64 threads=1`                          |  1.05M |   66.8 |    15.0 |         24.9 |          126 |     16.0 |         0 |      64.80M |        32.0 |     1.5 |  0.2 |        30/0/70 |
| hash_join_build | `build=1M dups=1 key=i32 threads=1`                           |  1.05M |   56.9 |    17.6 |         8.07 |         75.8 |     12.0 |         0 |      56.15M |        24.0 |     1.3 |  2.4 |        33/0/67 |
| hash_join_build | `build=1M dups=1 key=i64dense threads=1`                      |  1.05M |   56.4 |    17.7 |         8.13 |         80.1 |     16.0 |         0 |      60.34M |        32.0 |     1.8 |  1.5 |        35/0/65 |
| hash_join_build | `build=1M dups=1 key=2xi64 threads=1`                         |  1.05M |   61.7 |    16.2 |         9.29 |         83.8 |     24.0 |         0 |      64.54M |        48.0 |     2.5 |  0.8 |        39/0/61 |
| hash_join_build | `build=1M dups=1 key=utf8v16 threads=1`                       |  1.05M |   73.1 |    13.7 |         8.65 |          129 |     40.0 |         0 |      88.78M |        80.0 |     3.5 |  0.5 |        51/0/49 |
| hash_join_build | `build=1M dups=1 key=i64 threads=4`                           |  1.05M |   64.9 |    58.2 |         9.72 |          110 |     16.0 |         0 |      66.24M |        32.0 |     1.7 |  2.3 |        42/0/57 |
| hash_join_build | `build=1M dups=1 key=i64 threads=16`                          |  1.05M |   97.9 |     136 |         27.7 |          131 |     16.0 |         0 |      73.41M |        32.0 |     2.9 |  0.4 |        67/0/33 |
| hash_join_probe | `hit=0.50 build=1M dups=1 key=i64 batch=2048 threads=1`       |  4.19M |    185 |    5.40 |         26.8 |         24.9 |     16.0 |      16.0 |      60.34M |        36.0 |     0.6 |  1.1 |              - |
| hash_join_probe | `hit=0.00 build=1M dups=1 key=i64 batch=2048 threads=1`       |  4.19M |   82.7 |    12.1 |         3.91 |        0.129 |     16.0 |         0 |      60.34M |        8.00 |     0.5 |  5.6 |              - |
| hash_join_probe | `hit=0.10 build=1M dups=1 key=i64 batch=2048 threads=1`       |  4.19M |    110 |    9.09 |         26.8 |         5.74 |     16.0 |      3.20 |      60.34M |        13.6 |     0.5 |  3.4 |              - |
| hash_join_probe | `hit=0.90 build=1M dups=1 key=i64 batch=2048 threads=1`       |  4.19M |    234 |    4.28 |         26.8 |         44.1 |     16.0 |      28.8 |      60.34M |        58.4 |     0.8 |  3.3 |              - |
| hash_join_probe | `hit=1.00 build=1M dups=1 key=i64 batch=2048 threads=1`       |  4.19M |    248 |    4.03 |         20.8 |         32.8 |     16.0 |      32.0 |      60.34M |        64.0 |     0.8 |  2.5 |              - |
| hash_join_probe | `hit=0.50 build=2K dups=1 key=i64 batch=2048 threads=1`       |  4.19M |   46.5 |    21.5 |         26.8 |         24.9 |     16.0 |      16.0 |      216.6K |        36.0 |     2.5 |  0.1 |              - |
| hash_join_probe | `hit=0.50 build=64K dups=1 key=i64 batch=2048 threads=1`      |  4.19M |   78.4 |    12.8 |         26.8 |         24.9 |     16.0 |      16.0 |       4.56M |        36.0 |     1.5 |  0.8 |              - |
| hash_join_probe | `hit=0.50 build=4M dups=1 key=i64 batch=2048 threads=1`       |  4.19M |    220 |    4.55 |         26.8 |         24.9 |     16.0 |      16.0 |     222.89M |        36.0 |     0.5 |  0.3 |              - |
| hash_join_probe | `hit=0.50 build=1M dups=4 key=i64 batch=2048 threads=1`       |  4.19M |    393 |    2.54 |         58.1 |         85.8 |     16.0 |      64.0 |      55.88M |         120 |     1.0 |  1.5 |              - |
| hash_join_probe | `hit=0.50 build=1M dups=1 key=i32 batch=2048 threads=1`       |  4.19M |    199 |    5.04 |         26.8 |         20.9 |     12.0 |      12.0 |      56.15M |        26.0 |     0.4 |  2.1 |              - |
| hash_join_probe | `hit=0.50 build=1M dups=1 key=i64dense batch=2048 threads=1`  |  4.19M |    195 |    5.12 |         26.8 |         24.9 |     16.0 |      16.0 |      60.34M |        36.0 |     0.6 |  4.3 |              - |
| hash_join_probe | `hit=0.50 build=1M dups=1 key=2xi64 batch=2048 threads=1`     |  4.19M |    201 |    4.98 |         36.8 |         33.3 |     24.0 |      24.0 |      64.54M |        56.0 |     0.9 |  0.9 |              - |
| hash_join_probe | `hit=0.50 build=1M dups=1 key=utf8v16 batch=2048 threads=1`   |  4.19M |    266 |    3.76 |         38.3 |         65.3 |     40.0 |      40.0 |      88.78M |        96.0 |     1.1 |  2.1 |              - |
| hash_join_probe | `hit=0.50 build=1M dups=1 key=i64 batch=256 threads=1`        |  4.19M |    220 |    4.53 |         27.0 |         31.3 |     16.0 |      16.0 |      60.34M |        36.0 |     0.5 |  1.3 |              - |
| hash_join_probe | `hit=0.50 build=1M dups=1 key=i64 batch=1024 threads=1`       |  4.19M |    202 |    4.95 |         26.9 |         25.8 |     16.0 |      16.0 |      60.34M |        36.0 |     0.6 |  2.3 |              - |
| hash_join_probe | `hit=0.50 build=1M dups=1 key=i64 batch=2048 threads=4`       |  4.19M |    189 |    21.2 |         26.9 |         25.0 |     16.0 |      16.0 |      66.24M |        36.0 |     0.7 |  2.2 |              - |
| hash_join_probe | `hit=0.50 build=1M dups=1 key=i64 batch=2048 threads=16`      |  4.19M |    207 |    75.2 |         26.8 |         25.3 |     16.0 |      16.0 |      74.44M |        36.0 |     1.8 |  2.2 |              - |
| aggregate       | `groups=1K key=i64 batch=2048 threads=1`                      |  4.19M |   12.7 |    78.9 |         13.6 |         8.75 |     16.0 |     0.006 |      141.7K |        16.0 |     6.3 |  0.2 |         99/0/1 |
| aggregate       | `groups=16 key=i64 batch=2048 threads=1`                      |  4.19M |   11.2 |    89.5 |         12.4 |         8.61 |     16.0 |     0.000 |       29.5K |        16.0 |     7.2 |  2.5 |        100/0/0 |
| aggregate       | `groups=64K key=i64 batch=2048 threads=1`                     |  4.19M |   57.5 |    17.4 |         16.0 |         17.7 |     16.0 |     0.375 |       9.79M |        16.4 |     1.4 |  1.2 |         97/0/3 |
| aggregate       | `groups=1M key=i64 batch=2048 threads=1`                      |  4.19M |   46.7 |    21.4 |         75.2 |         64.6 |     16.0 |      6.00 |      72.94M |        22.0 |     1.7 |  0.6 |        35/0/65 |
| aggregate       | `groups=4M key=i64 batch=2048 threads=1`                      |  4.19M |   62.1 |    16.1 |         79.2 |          164 |     16.0 |      24.0 |      72.97M |        40.0 |     2.0 |  0.3 |        26/0/74 |
| aggregate       | `groups=1K key=i32 batch=2048 threads=1`                      |  4.19M |   12.2 |    81.9 |         13.6 |         8.74 |     12.0 |     0.005 |      133.6K |        12.0 |     4.9 |  2.9 |         99/0/1 |
| aggregate       | `groups=1K key=2xi64 batch=2048 threads=1`                    |  4.19M |   25.1 |    39.8 |         16.3 |         9.01 |     24.0 |     0.008 |      218.1K |        24.0 |     4.8 |  0.3 |        100/0/0 |
| aggregate       | `groups=1K key=utf8v16 batch=2048 threads=1`                  |  4.19M |   29.0 |    34.5 |         14.0 |         9.77 |     40.0 |     0.012 |       1.27M |        40.0 |     6.9 |  0.7 |        100/0/0 |
| aggregate       | `groups=1K key=i64 batch=256 threads=1`                       |  4.19M |   20.2 |    49.4 |         12.2 |         12.7 |     16.0 |     0.006 |      128.2K |        16.0 |     4.0 |  1.7 |        100/0/0 |
| aggregate       | `groups=1K key=i64 batch=1024 threads=1`                      |  4.19M |   14.0 |    71.4 |         12.8 |         9.32 |     16.0 |     0.006 |      126.6K |        16.0 |     5.7 |  2.5 |         99/0/1 |
| aggregate       | `groups=1K key=i64 batch=2048 threads=4`                      |  4.19M |   13.7 |     288 |         15.0 |         8.93 |     16.0 |     0.006 |      479.6K |        16.0 |     6.4 |  0.8 |         97/0/2 |
| aggregate       | `groups=1K key=i64 batch=2048 threads=16`                     |  4.19M |   15.4 |     906 |         20.8 |         9.66 |     16.0 |     0.006 |       2.16M |        16.0 |    11.1 |  3.4 |         95/0/5 |
| sort            | `order=random key=i64 distinct=2M batch=2048 threads=1`       |  2.10M |    120 |    8.34 |         37.5 |          101 |     16.0 |      16.0 |      54.57M |        32.0 |     0.8 |  0.4 |        60/0/40 |
| sort            | `order=sorted key=i64 distinct=2M batch=2048 threads=1`       |  2.10M |   44.3 |    22.5 |         23.5 |         68.7 |     16.0 |      16.0 |      54.57M |        32.0 |     2.3 |  0.3 |        51/0/49 |
| sort            | `order=reverse key=i64 distinct=2M batch=2048 threads=1`      |  2.10M |   45.2 |    22.1 |         23.5 |         68.7 |     16.0 |      16.0 |      54.57M |        32.0 |     2.2 |  0.3 |        52/0/48 |
| sort            | `order=nearly key=i64 distinct=2M batch=2048 threads=1`       |  2.10M |   98.2 |    10.2 |         37.5 |          101 |     16.0 |      16.0 |      54.57M |        32.0 |     1.0 |  0.1 |        66/0/34 |
| sort            | `order=random key=i32 distinct=2M batch=2048 threads=1`       |  2.10M |    107 |    9.38 |         37.5 |         81.1 |     12.0 |      12.0 |      44.08M |        24.0 |     0.7 |  0.7 |        54/0/46 |
| sort            | `order=random key=2xi64 distinct=2M batch=2048 threads=1`     |  2.10M |    781 |    1.28 |         44.6 |          155 |     24.0 |      24.0 |     123.27M |        48.0 |     0.2 |  2.6 |        87/0/13 |
| sort            | `order=random key=utf8v16 distinct=2M batch=2048 threads=1`   |  2.10M |    511 |    1.96 |         46.8 |          174 |     40.0 |      40.0 |     117.48M |        80.0 |     0.5 |  3.8 |        83/0/17 |
| sort            | `order=random key=i64 distinct=16 batch=2048 threads=1`       |  2.10M |   72.1 |    13.9 |         24.8 |         72.2 |     16.0 |      16.0 |      54.57M |        32.0 |     1.4 |  0.8 |        61/0/39 |
| sort            | `order=random key=i64 distinct=64K batch=2048 threads=1`      |  2.10M |    109 |    9.16 |         37.5 |          101 |     16.0 |      16.0 |      54.57M |        32.0 |     0.9 |  0.6 |        67/0/33 |
| sort            | `order=random key=i64 distinct=2M batch=256 threads=1`        |  2.10M |    136 |    7.34 |         18.7 |          103 |     16.0 |      16.0 |      54.83M |        32.0 |     0.7 |  0.3 |        65/0/35 |
| sort            | `order=random key=i64 distinct=2M batch=2048 threads=4`       |  2.10M |    128 |    14.2 |         37.5 |          101 |     16.0 |      16.0 |     109.20M |        32.0 |     0.4 |  0.3 |        61/0/39 |
| sort            | `order=random key=i64 distinct=2M batch=2048 threads=16`      |  2.10M |    136 |    14.2 |         39.9 |          102 |     16.0 |      16.0 |     137.92M |        32.0 |     0.3 |  2.5 |        54/0/46 |
| top_k           | `k=100 order=random key=i64 threads=1`                        |  4.19M |   2.22 |     449 |         21.3 |        0.817 |     16.0 |     0.000 |      268.4K |        8.00 |    18.0 |  0.6 |        100/0/0 |
| top_k           | `k=10 order=random key=i64 threads=1`                         |  4.19M |   1.75 |     570 |         14.2 |        0.624 |     16.0 |     0.000 |      115.3K |        8.00 |    22.9 |  0.7 |        100/0/0 |
| top_k           | `k=10000 order=random key=i64 threads=1`                      |  4.19M |   5.08 |     196 |         37.4 |         3.85 |     16.0 |     0.038 |       1.83M |        8.04 |     7.9 |  4.4 |        100/0/0 |
| top_k           | `k=100000 order=random key=i64 threads=1`                     |  4.19M |   17.5 |    57.3 |         37.1 |         19.3 |     16.0 |     0.381 |      16.24M |        8.38 |     2.3 |  0.9 |         99/0/1 |
| top_k           | `k=100 order=sorted key=i64 threads=1`                        |  4.19M |   1.40 |     711 |         10.1 |        0.523 |     16.0 |     0.000 |      118.2K |        8.00 |    28.6 |  1.6 |        100/0/0 |
| top_k           | `k=100 order=reverse key=i64 threads=1`                       |  4.19M |   15.9 |    62.8 |         98.0 |         37.9 |     16.0 |     0.000 |      141.3K |        8.00 |     2.5 |  0.1 |        100/0/0 |
| top_k           | `k=100 order=random key=utf8v16 threads=1`                    |  4.19M |   10.6 |    94.0 |         29.5 |         9.10 |     40.0 |     0.001 |      44.93M |        32.0 |    15.1 |  0.4 |        100/0/0 |
| top_k           | `k=100 order=random key=i64 threads=4`                        |  4.19M |   3.11 |    1232 |         26.7 |         1.06 |     16.0 |     0.000 |      466.3K |        8.00 |    13.6 |  0.4 |        100/0/0 |
| top_k           | `k=100 order=random key=i64 threads=16`                       |  4.19M |   5.11 |    2464 |         35.4 |         1.39 |     16.0 |     0.000 |      913.0K |        8.00 |    15.1 |  2.5 |        100/0/0 |
| window          | `fn=running_sum partitions=1K order=random threads=1`         |  2.10M |   1140 |   0.877 |         8261 |          508 |     24.0 |      32.0 |     222.32M |        56.0 |     0.2 |  1.6 |        55/0/44 |
| window          | `fn=row_number partitions=1K order=random threads=1`          |  2.10M |    746 |    1.34 |         54.8 |          195 |     24.0 |      32.0 |     222.32M |        56.0 |     0.2 |  1.3 |        84/0/16 |
| window          | `fn=partition_sum partitions=1K order=random threads=1`       |  2.10M |    125 |    8.02 |         59.7 |          157 |     24.0 |      32.0 |     167.80M |        56.0 |     1.4 |  0.4 |        53/0/47 |
| window          | `fn=running_sum partitions=1 order=random threads=1`          |  2.10M |    608 |    1.64 |         8227 |          469 |     24.0 |      32.0 |     167.78M |        56.0 |     0.3 |  0.2 |        13/0/87 |
| window          | `fn=running_sum partitions=16 order=random threads=1`         |  2.10M |   1180 |   0.847 |         8226 |          507 |     24.0 |      32.0 |     222.31M |        56.0 |     0.2 |  1.4 |        53/0/47 |
| window          | `fn=running_sum partitions=64K order=random threads=1`        |  2.10M |   1268 |   0.789 |        10737 |          596 |     24.0 |      32.0 |     223.35M |        56.0 |     0.1 |  0.6 |        48/0/52 |
| window          | `fn=running_sum partitions=1K order=sorted threads=1`         |  2.10M |    522 |    1.91 |         8261 |          508 |     24.0 |      32.0 |     222.32M |        56.0 |     0.3 |  1.9 |        12/0/88 |
| window          | `fn=running_sum partitions=1K order=random threads=4`         |  2.10M |    913 |    4.17 |         8566 |          574 |     24.0 |      32.0 |     161.96M |        56.0 |     0.2 |  0.2 |        48/0/52 |
| window          | `fn=running_sum partitions=1K order=random threads=16`        |  2.10M |   1032 |    12.8 |         9515 |          618 |     24.0 |      32.0 |     161.64M |        56.0 |     0.5 |  1.0 |        42/0/58 |
| tuple_data      | `cols=4xi64 sel=1.00 partitions=1 batch=2048 threads=1`       |  4.19M |   24.3 |    37.1 |         6.33 |         45.6 |     32.0 |      41.1 |     172.27M |        64.0 |     7.5 |  0.9 |              - |
| tuple_data      | `cols=i64 sel=1.00 partitions=1 batch=2048 threads=1`         |  4.19M |   8.42 |     107 |         3.15 |         21.6 |     8.00 |      17.1 |      71.60M |        16.0 |     5.5 |  1.2 |              - |
| tuple_data      | `cols=i32,i64,f64 sel=1.00 partitions=1 batch=2048 threads=1` |  4.19M |   16.3 |    55.2 |         5.24 |         33.6 |     20.0 |      29.1 |     121.92M |        40.0 |     7.0 |  0.8 |              - |
| tuple_data      | `cols=i64,utf8v16 sel=1.00 partitions=1 batch=2048 threads=1` |  4.19M |   36.3 |    25.3 |         4.29 |         80.8 |     40.0 |      56.8 |     238.09M |        80.0 |     6.4 |  0.5 |              - |
| tuple_data      | `cols=4xi64 sel=0.50 partitions=1 batch=2048 threads=1`       |  2.10M |   23.1 |    39.2 |         5.17 |         42.2 |     32.0 |      41.1 |      86.27M |        95.8 |    12.6 |  0.9 |              - |
| tuple_data      | `cols=4xi64 sel=1.00 partitions=16 batch=2048 threads=1`      |  4.19M |   35.5 |    25.5 |         6.46 |         47.3 |     32.0 |      41.5 |     174.15M |        64.0 |     5.2 |  0.9 |              - |
| tuple_data      | `cols=4xi64 sel=1.00 partitions=256 batch=2048 threads=1`     |  4.19M |   66.1 |    14.4 |         7.89 |         75.4 |     32.0 |      48.0 |     201.43M |        64.0 |     2.9 |  1.3 |              - |
| tuple_data      | `cols=4xi64 sel=1.00 partitions=1 batch=256 threads=1`        |  4.19M |   32.0 |    28.2 |         6.04 |         48.9 |     32.0 |      41.1 |     172.27M |        64.0 |     5.7 |  0.6 |              - |
| tuple_data      | `cols=4xi64 sel=1.00 partitions=1 batch=1024 threads=1`       |  4.19M |   24.7 |    36.5 |         6.17 |         46.1 |     32.0 |      41.1 |     172.27M |        64.0 |     7.4 |  2.2 |              - |
| tuple_data      | `cols=4xi64 sel=1.00 partitions=1 batch=2048 threads=4`       |  4.19M |   22.2 |     150 |         6.36 |         46.0 |     32.0 |      41.3 |     173.06M |        64.0 |     8.9 |  1.0 |              - |
| tuple_data      | `cols=4xi64 sel=1.00 partitions=1 batch=2048 threads=16`      |  4.19M |   13.6 |     691 |         6.46 |         47.5 |     32.0 |      42.0 |     176.21M |        64.0 |    29.3 |  3.6 |              - |
| parquet_decode  | `col=i64 distinct=4M codec=none batch=8192 threads=1`         |  4.19M |   2.91 |     342 |         7.99 |         17.6 |        0 |      8.00 |           0 |        18.0 |    19.5 |  0.8 |        0/0/100 |
| parquet_decode  | `col=i32 distinct=4M codec=none batch=8192 threads=1`         |  4.19M |   2.40 |     414 |         7.99 |         9.09 |        0 |      4.00 |           0 |        9.97 |    13.1 |  0.4 |        0/0/100 |
| parquet_decode  | `col=utf8v16 distinct=4M codec=none batch=8192 threads=1`     |  4.19M |   8.59 |     116 |         8.37 |         31.3 |        0 |      32.0 |           0 |        53.6 |    19.8 |  0.4 |        0/0/100 |
| parquet_decode  | `col=i64 distinct=16 codec=none batch=8192 threads=1`         |  4.19M |   1.84 |     541 |         7.99 |         9.59 |        0 |      8.00 |           0 |        8.51 |    14.6 |  2.0 |        0/0/100 |
| parquet_decode  | `col=i64 distinct=64K codec=none batch=8192 threads=1`        |  4.19M |   3.13 |     318 |         7.99 |         14.7 |        0 |      8.00 |           0 |        15.0 |    15.2 |  1.4 |        0/0/100 |
| parquet_decode  | `col=i64 distinct=4M codec=snappy batch=8192 threads=1`       |  4.19M |   3.81 |     261 |         8.33 |         27.6 |        0 |      8.00 |           0 |        18.0 |    14.9 |  0.4 |        0/0/100 |
| parquet_decode  | `col=i64 distinct=4M codec=zstd1 batch=8192 threads=1`        |  4.19M |   17.6 |    56.9 |         8.33 |         27.6 |        0 |      8.00 |           0 |        17.4 |     3.1 |  0.2 |        0/0/100 |
| parquet_decode  | `col=i64 distinct=4M codec=none batch=2048 threads=1`         |  4.19M |   3.54 |     282 |         13.7 |         17.0 |        0 |      8.00 |           0 |        18.0 |    16.1 |  0.6 |        0/0/100 |
| parquet_decode  | `col=i64 distinct=4M codec=none batch=32768 threads=1`        |  4.19M |   2.95 |     337 |         6.30 |         20.5 |        0 |      8.00 |           0 |        18.0 |    19.2 |  0.5 |        0/0/100 |
| parquet_decode  | `col=i64 distinct=4M codec=none batch=8192 threads=4`         |  4.19M |   3.10 |    1216 |         8.02 |         17.6 |        0 |      8.00 |           0 |        18.0 |    20.3 |  8.5 |        0/0/100 |
| parquet_decode  | `col=i64 distinct=4M codec=none batch=8192 threads=16`        |  4.19M |   3.45 |    2721 |         8.14 |         17.6 |        0 |      8.00 |           0 |        18.0 |    32.4 |  5.8 |        0/0/100 |

Per operator:

- **Projection.** Evaluating `a + b` costs 1.47 ns per row (54% of the floor)
  and passing columns through 0.07 ns per row (one allocation per chunk). On
  chunks with selection vectors, it gathers the columns of its expressions,
  `c` included, before evaluating: 5.9 ns per selected row at a selectivity of
  0.5, 27 ns at 0.1. Chunks of 256 rows cost 2.8x more per row than chunks
  of 2048. On 16 threads, it is at 83% of the floor.
- **Filter.** 1.25 to 1.48 ns per row if it selects nothing or everything,
  2.9 ns at a selectivity of 0.5 and 4.1 ns at 0.9: building the selection
  vectors costs more than the comparison. A string comparison (`s < x`) costs
  11 ns per row.
- **Hash join build.** 56 to 63 ns per row, whatever the size of the build
  side, the type of its keys (`utf8v16`: 73) or their range, 65% of it in the
  `JoinHashTableBuild` pipeline. Keys of 4 rows each make it 81 ns per row.
  On 16 threads, the time per row grows to 98 ns, and the build runs 7.7x
  faster than on one.
- **Hash join probe.** 185 ns per row against 1Mi build rows at a hit ratio of
  0.5, and 248 ns at 1.0 (83 ns at 0): 0.6% of the floor. Against 2Ki build
  rows (in the L1 and L2 caches) it is 46 ns per row, against 64Ki 78 ns. The
  dense index (`i32`, `i64dense`) is not faster than the hashed table: the time
  goes to the matches. 27 allocations per chunk. On 16 threads, 14x the
  throughput of one.
- **Aggregation.** 11 to 13 ns per row for 16 and 1Ki groups, 58 ns for 64Ki,
  47 ns for 1Mi (whose rows bypass the thread-local tables) and 62 ns for 4Mi
  groups. Two `Int64` keys or strings double the time of 1Ki groups.
- **Sort.** 120 ns per row for 2Mi random `i64` keys, 44 ns if they are
  sorted or reverse sorted, 781 ns for `2xi64` keys and 511 ns for `utf8v16`.
  16 threads sort 1.7x faster than one.
- **TopK.** 2.2 ns per row for `k = 100` on a random order (18% of the
  floor), 1.4 ns on sorted input, 15.9 ns on reverse-sorted input (98
  allocations per chunk), and 17.5 ns for `k = 100Ki`.
- **Window.** A running `SUM` costs 1140 ns per row (8261 allocations per
  chunk of 2048 rows: 4 per row), 522 ns on sorted input; a `ROW_NUMBER()`
  746 ns, the sum of a partition 125 ns. 16 threads are 15x faster than one.
- **Tuple-data ingestion.** 24 ns per row (6 ns per value) for 4 `Int64`
  columns (7.5% of the floor), 8.4 ns for one, 36 ns with a string; 35 ns into
  16 radix partitions and 66 ns into 256.
- **Parquet decode.** 2.9 ns per row for plain `Int64` values (19.5% of the
  floor), 1.8 ns for 16 distinct values (dictionary-encoded), 8.6 ns for
  strings; Snappy adds 0.9 ns per row and Zstandard 15 ns. 16 threads decode
  8x faster than one.

## Variance

The benchmark ran twice in a row on the same machine, the second time with
`OPERATOR_MICRO_COMPARE` set to the results of the first. Between the two
runs, the median change of `ns/row` over the 113 configurations is 1.2%; 83%
of them changed by at most 5% and 93% by at most 10%. The largest changes were
in configurations that allocate and touch much fresh memory: tuple-data
ingestion on 16 threads (26.2 ns per row in the first run, 13.6 in the second;
earlier runs measured either), the probe of keys of 4 rows each (+13%), the
builds of 4Mi rows (-11%) and of keys of 4 rows each (-14%); and a filter on 4
threads (+11%). Within a run, the coefficient of variation of
`ns/row` over the 5 timed executions is 0.8% in the median configuration, at
most 2.9% in 90% of them, and at most 11.6% (a projection on 16 threads). The
machine was shared: other processes ran during the measurements (a load of
1.5 to 3 of its 16 cores), which affects the configurations on 16 threads
most.

## Profiles

The worst configurations of every operator (and their defaults), profiled with
`perf record` and `perf report`:

```text
OPERATOR_MICRO_FILTER="hash_join_probe hit=0.50 build=1M dups=1 key=i64 batch=2048 threads=1 " \
    perf record -F 1999 -g -o probe.data -- target/release-nonlto/deps/operator_micro-<hash> operator_micro --nocapture
perf report -i probe.data --no-children --sort symbol --no-inline
```

The tables below list the top self-time frames of the samples whose call chain
passes through the operator (its `PhysicalOperator` methods, and for the
pipeline breakers their modules), as shares of these samples, which leaves
out the generation of the data and the harness (`perf script`, with frame
pointers; inlined functions count towards their callers). Crate paths are left
out of the frames.

### Projection

`exprs=a+b,c sel=1.00` (1.53 ns per row) and `exprs=a+b,c sel=0.10` (30 ns per
selected row):

| `sel=1.00` | Frame                                                     | `sel=0.10` | Frame                                    |
| ---------: | --------------------------------------------------------- | ---------: | ---------------------------------------- |
|      71.5% | `arity::binary::<Int64Type, Int64Type, integer_op>` (`+`) |      65.2% | `take::take_native::<i64>` (the gathers) |
|       6.8% | `__aarch64_ldadd8_rel{,_relax}` (`Arc` counts)            |       6.3% | `__aarch64_ldadd8_rel{,_relax}`          |
|       2.7% | `numeric::arithmetic_op`                                  |       6.3% | `malloc`, `_int_malloc`                  |
|       1.8% | `PhysicalProjection::execute`                             |       2.6% | `arity::binary` (`+`)                    |
|       1.4% | `RecordBatch::try_new_impl`                               |       1.2% | `RecordBatch::try_new_impl`              |

Without a selection, the projection is the kernel: 72% of its time adds. With
a selection, it gathers `a`, `b` and the passed-through `c` (`copied B/row`:
36 bytes per selected row), and the gathers cost 2/3 of its time.

### Filter

`pred=a<x sel=0.5` (3.22 ns per row) and `pred=utf8<x sel=0.5` (12.0 ns per
row):

| `a<x` | Frame                                                                  | `utf8<x` | Frame                                                    |
| ----: | ---------------------------------------------------------------------- | -------: | -------------------------------------------------------- |
| 37.1% | `Vec<u32>::extend_desugared::<BitIndexU32Iterator>` (selection vector) |    41.4% | `memcmp`, `memcmp@plt`                                   |
| 19.6% | `cmp::apply_op::<&[i64], is_lt>` (the comparison)                      |    34.2% | `cmp::collect_bool::<apply_op<GenericByteViewArray, …>>` |
| 15.2% | `chunk::check_selection_bounds`                                        |    10.0% | `Vec<u32>::extend_desugared::<BitIndexU32Iterator>`      |
|  5.0% | `__aarch64_ldadd8_rel{,_relax}`                                        |     3.9% | `chunk::check_selection_bounds`                          |
|  1.9% | `PhysicalFilter::execute`                                              |     1.6% | `__aarch64_ldadd8_relax`                                 |

The comparison is a fifth of the time of the integer filter; turning its mask
into a selection vector (pushed index by index through the generic
`Vec::extend`) and checking every index of the new selection against the
length of the chunk (`DataChunk::with_selection`) are half of it. The strings
of the string filter share their first 4 bytes, the prefix of their views, so
that every comparison calls `memcmp` for the next 12.

### Hash join build

`build=1M dups=1 key=i64` on 1 thread (57 to 72 ns per row) and on 16 threads
(94 ns per row):

| 1 thread | Frame                                           | 16 threads | Frame                                   |
| -------: | ----------------------------------------------- | ---------: | --------------------------------------- |
|    59.2% | `JoinHashTableBuilder::build_partition`         |      32.2% | `JoinHashTableBuilder::build_partition` |
|    10.5% | `PartitionedTupleData::append`                  |      20.1% | `tuple_data::collection::scatter_rows`  |
|     8.3% | `tuple_data::collection::scatter_rows`          |       8.1% | `PartitionedTupleData::append`          |
|     5.1% | `TupleDataCollection::row_refs`                 |      10.3% | `__aarch64_ldadd8_rel{,_relax}`         |
|     7.0% | `hash_single_array`, `hash_join_keys` (hashing) |       7.2% | `hash_single_array`, `hash_join_keys`   |
|     3.6% | `__memset_zva64`                                |       3.7% | `TupleDataCollection::row_refs`         |

On one thread, the rows are radix-partitioned into 4 partitions
(`radix_bits_for_threads(1)`), whose pointer tables have 512Ki entries each
(4 MiB, four times the 1 MiB L2 cache): inserting a row costs about 35 ns, a
miss of the L2 cache per row. On 16 threads, the atomic reference counts and
counters (10%) come up.

### Hash join probe

`hit=0.50 build=1M dups=1 key=i64` (223 ns per row in the profiled run) and the
same against 2Ki build rows (51 ns per row):

| 1Mi rows | Frame                                     | 2Ki rows | Frame                                           |
| -------: | ----------------------------------------- | -------: | ----------------------------------------------- |
|    43.5% | `JoinHashTable::find_chains::{closure#3}` |    26.4% | `ChainScan::next_batch`                         |
|    34.0% | `ChainScan::next_batch`                   |    26.4% | `ChainScan::reset` (with `find_chains` inlined) |
|     7.4% | `row_matcher::match_column`               |    13.9% | `JoinHashTable::find_chains::{closure#3}`       |
|     4.0% | `ChainScan::reset`                        |    10.7% | `hash_single_array`, `hash_join_keys`           |
|     3.0% | `tuple_data::collection::gather_values`   |     5.3% | `Vec<RowPtr>::from_iter` (twice)                |
|     2.9% | `hash_single_array`, `hash_join_keys`     |     3.8% | `row_matcher::match_column`                     |

`find_chains::{closure#3}` probes the open-addressed table of a probe row,
from the slot of its hash to the first entry with its salt: one random access
into the 16 MiB of entries of the tables per probe row. `ChainScan::next_batch` walks the chain of
every match through the `next` array, apart from the rows (another random
access per match, also for keys that have a single row), and
`match_column` compares the keys of the probe rows with those of the first
rows of their chains (a third). The loop of every probe row branches on the
entry it reads, so the out-of-order core overlaps few of these misses. In the
caches (2Ki rows), the probe still costs 46 ns per row: the per-row work of
`next_batch` and of the lookup loop, and hashing.

### Aggregation

`groups=1K` (14 ns per row), `groups=64K` (63 ns per row) and `groups=4M`
(70 ns per row):

|   1Ki |  64Ki |   4Mi | Frame                                                                     |
| ----: | ----: | ----: | ------------------------------------------------------------------------- |
| 73.3% | 79.8% | 31.5% | `GroupValuesPrimitive::<Int64Type>::intern` (DataFusion's group keys)     |
| 17.2% | 12.9% |  5.7% | `NullState::accumulate` (`SUM`), `accumulate_indices` (`COUNT`)           |
|     - |  0.6% | 26.7% | `RawTable<(usize, u64)>::reserve_rehash` (growing the tables)             |
|     - |  0.2% | 12.3% | `radix_ht::scatter_primitive`, `scatter`, `aggregate_bypassed_or_limited` |
|     - |  0.3% |  5.3% | `hash_single_array`                                                       |

At 64Ki groups, the thread-local table of the groups no longer fits in the L2
cache, and `intern` takes 4.5x longer per row than at 1Ki groups. At 4Mi
groups, the rows bypass the thread-local tables (radix-partitioned once), but
the final aggregation of every partition grows its table from empty, rehashing
it again and again: a quarter of its time.

### Sort

`order=random key=i64` (133 ns per row), `key=2xi64` (922 ns per row) and
`key=utf8v16` (582 ns per row):

| `i64` | Frame                                                                  | `2xi64` | Frame                                                        |
| ----: | ---------------------------------------------------------------------- | ------: | ------------------------------------------------------------ |
| 33.4% | `merge::play_primitive::<Int64Type, Choose>` (merging the runs)        |   45.0% | `memcmp`, `memcmp@plt`                                       |
| 28.1% | `quicksort`, `small_sort_general` of `(u32, i64)` (sorting the runs)   |   21.4% | `Rows::row`                                                  |
| 17.9% | `interleave::interleave_primitive::<Int64Type>` (gathering the output) |    9.7% | `quicksort`, `small_sort_network` of `u32` (`order_by_rows`) |
|  9.5% | `Vec<(usize, usize)>::from_iter` (the indices of `interleave`)         |    7.3% | `__memcpy_generic`                                           |
|  2.2% | `RunMerger::gather`                                                    |    5.7% | `RunMerger::next_batch`                                      |

Single `Int64` keys are sorted as `(index, value)` pairs, and the runs merged
by a tournament over their values (`play_primitive`) into `interleave`
indices. Several keys and strings are encoded into Arrow rows and compared
with `memcmp` calls on every comparison (`utf8v16`: `memcmp` 45%, `quicksort`
of `(u32, u128)` views 24%). On 16 threads, `play_primitive` is 39% of the
time: the merge runs on one thread, so that 16 threads sort only 1.7x faster
than one.

### TopK

`k=100 order=reverse` (17.6 ns per row) and `k=100000 order=random` (19.4 ns
per row):

| Reverse | Frame                                                                | `k = 100Ki` | Frame                                             |
| ------: | -------------------------------------------------------------------- | ----------: | ------------------------------------------------- |
|   27.1% | `select::partition_at_index` of `(u32, i64)` (`select_nth_unstable`) |       30.9% | `quicksort`, `small_sort_general` of `(u32, i64)` |
|   19.1% | `IntoIter<u32>::fold` (`sort_primitive`: building the pairs)         |       12.7% | `interleave::interleave_primitive`                |
|    8.0% | `__aarch64_ldadd8_rel{,_relax}`                                      |       12.3% | `Vec<(usize, usize)>::from_iter`                  |
|    5.0% | `malloc`, `_int_free`                                                |        8.6% | `select::partition_at_index`                      |
|    4.7% | `cmp::apply_op::<&[i64], is_lt>` (the threshold)                     |        4.3% | `cmp::apply_op::<&[i64], is_lt>`                  |

On reverse-sorted input, every row of every chunk passes the threshold, and
the TopK selects the best `k` of every chunk with a `select_nth_unstable` of
the chunk, allocating 98 times per chunk. With a large `k`, it sorts and
gathers (`interleave`) its kept rows again and again.

### Window

`fn=running_sum partitions=1K` on random (1248 ns per row) and sorted input
(575 ns per row), and `fn=row_number` (867 ns per row):

| Random | Sorted | `row_number` | Frame                                                                  |
| -----: | -----: | -----------: | ---------------------------------------------------------------------- |
|  28.7% |   3.2% |        44.5% | `memcmp`, `memcmp@plt` (comparing the sort keys)                       |
|  14.6% |   2.7% |        21.0% | `Rows::row`                                                            |
|   6.7% |      - |         9.6% | `quicksort`, `small_sort_network` of `u32` (`SortKeys::order_by_rows`) |
|   9.6% |  20.4% |            - | `__aarch64_ldadd8_rel{,_relax}` (`Arc` counts of the sliced arrays)    |
|   4.5% |  14.1% |            - | `malloc`, `_int_free`, `cfree`                                         |
|      - |   3.7% |            - | `PlainAggregateWindowExpr::get_aggregate_result_inside_range`          |
|   4.7% |   5.9% |         6.8% | `RunMerger::next_batch`                                                |

The window sorts its rows by `p, o` as Arrow rows compared with `memcmp`,
like a sort on two keys: that is most of the time of `ROW_NUMBER()` and of a
running `SUM` on random input. A running `SUM` (a `RANGE` frame) is evaluated
by DataFusion's `WindowExpr::evaluate` row by row: every row slices the
arguments of the rows of its frame (reference counts, allocations) and
updates an accumulator, 4 allocations per row.

### Tuple-data ingestion

`cols=4xi64 partitions=1` (28 ns per row) and `partitions=256` (74 ns per
row):

| 1 partition | 256 partitions | Frame                                  |
| ----------: | -------------: | -------------------------------------- |
|       71.9% |          61.8% | `tuple_data::collection::scatter_rows` |
|        0.4% |          11.9% | `PartitionedTupleData::append`         |
|        0.3% |           7.0% | `TupleDataCollection::append_rows`     |
|        7.4% |           4.1% | `__memset_zva64` (zeroing new blocks)  |
|        5.8% |           7.0% | `__aarch64_ldadd8_rel{,_relax}`        |

`scatter_rows` scatters every column into the rows of a block (of 41 bytes:
the 8 bytes of the hash, 32 of values and a validity byte) one column after
the other, for all rows of the chunk: 6 passes over 84 KiB of rows per chunk,
more than the 64 KiB L1 cache, at about 6 ns per value. New blocks are zeroed
before they are written.

### Parquet decode

`col=i64 codec=none` (3.3 ns per row), `codec=zstd1` (19.6 ns per row) and
`col=utf8v16` (7.8 ns per row):

| `i64` | Frame                                    | `zstd1` | Frame                                    | `utf8v16` | Frame                                   |
| ----: | ---------------------------------------- | ------: | ---------------------------------------- | --------: | --------------------------------------- |
| 25.6% | `RleDecoder::get_batch_with_dict::<i64>` |   73.2% | `HUF_decompress4X1_usingDTable_internal` |     30.5% | `ByteViewArrayDecoderPlain::read`       |
| 17.1% | `__memcpy_generic`                       |    6.4% | `__memcpy_generic`                       |     17.7% | `ByteViewArrayDecoderDictionary::read`  |
|  9.6% | `__memset_zva64`                         |    4.7% | `RleDecoder::get_batch_with_dict::<i64>` |     15.0% | `simdutf8::validate_utf8_basic_neon`    |
|  7.6% | `BitReader::get_batch::<i32>`            |    1.7% | `__memset_zva64`                         |     12.3% | `generic_bytes_view_builder::make_view` |
|  9.4% | `bit_pack::unpack32`                     |    1.6% | `BitReader::get_batch::<i32>`            |      5.8% | `bit_pack::unpack32`                    |

The decoding is the `parquet` crate's. The writer dictionary-encodes the
first values of every column chunk, until its dictionary is full, and writes
the others plain: the decoder looks the first ones up
(`get_batch_with_dict`) and copies the others. Zstandard spends three
quarters of the time in its Huffman decoder, and strings a sixth in the
validation of their UTF-8.

## Optimizations they imply

Each of these is a follow-up issue; the expected gains follow from the
measurements above.

| Issue                                                     | Operator             | Optimization                                                                               | Measurement                                                                                                              | Expected gain                                                                       |
| --------------------------------------------------------- | -------------------- | ------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------- |
| [#99](https://github.com/robtandy/datafusion/issues/99)   | hash join probe      | probe a chunk in steps whose cache misses overlap (slots, entries, salts, prefetched rows) | 185 ns/row against 1Mi build rows, 46 ns/row in the caches; `find_chains` 44%, `next_batch` 34%                          | 2.5x to 3x on build sides larger than the caches (185 to 60-80 ns/row)              |
| [#100](https://github.com/robtandy/datafusion/issues/100) | hash join probe      | walk single-row chains without reading the separate `next` array                           | `next_batch` 34% to 41% of the probe: a random access per match                                                          | about 30% on unique build keys (185 to ~130 ns/row)                                 |
| [#101](https://github.com/robtandy/datafusion/issues/101) | hash join build      | radix partitions whose pointer tables fit in the L2 cache, prefetched inserts              | 57 ns/row, 65% in the `JoinHashTableBuild`, `build_partition` 59%: 4 partitions of 4 MiB on one thread                   | about 2x (57 to ~30 ns/row)                                                         |
| [#102](https://github.com/robtandy/datafusion/issues/102) | filter               | selection vectors written 64 bits at a time, without re-checking their bounds              | 2.9 ns/row at selectivity 0.5 against 1.25 at 1.0; selection 37% and its check 15%                                       | about 2x between selectivities 0.1 and 0.9                                          |
| [#103](https://github.com/robtandy/datafusion/issues/103) | projection           | evaluate infallible expressions on all physical rows and keep the selection                | 5.9 and 27 ns per selected row at 0.5 and 0.1; `take` 65%, 36 bytes copied per row                                       | about 2x on chunks with selections, no copies of passed-through columns             |
| [#104](https://github.com/robtandy/datafusion/issues/104) | sort, window         | sort normalized fixed-width keys with inline comparisons                                   | 781 ns/row on two `Int64` keys against 120 on one, 511 on strings, 746 for `ROW_NUMBER()`; `memcmp` 45%, `Rows::row` 21% | 4x to 5x on two integer keys, 2x to 3x on strings, the same on the sorts of windows |
| [#105](https://github.com/robtandy/datafusion/issues/105) | sort                 | merge the sorted runs on all threads                                                       | 1.7x from 1 to 16 threads; `play_primitive` 39% on one thread                                                            | 4x to 6x the throughput on 16 threads                                               |
| [#106](https://github.com/robtandy/datafusion/issues/106) | window               | evaluate running aggregates once per peer group, in one pass                               | 522 ns/row on sorted input, 4 allocations per row                                                                        | 7x on sorted input (522 to ~70 ns/row)                                              |
| [#107](https://github.com/robtandy/datafusion/issues/107) | aggregation          | thread-local tables within the L2 cache, final tables sized from their rows                | 58 ns/row at 64Ki groups (`intern` 80%), 62 at 4Mi (`reserve_rehash` 27%)                                                | about 2x at 64Ki groups, 25% at 4Mi                                                 |
| [#108](https://github.com/robtandy/datafusion/issues/108) | tuple-data ingestion | scatter in batches of rows that fit in the L1 cache, no zeroing of new blocks              | 24 ns/row for 4 `Int64` columns (6 ns per value); `scatter_rows` 72%, `memset` 7%                                        | about 2x (24 to ~12 ns/row), shared by joins and aggregations                       |
| [#109](https://github.com/robtandy/datafusion/issues/109) | TopK                 | keep the best rows in a heap without per-chunk selections and arrays                       | 15.9 ns/row on reverse-sorted input (98 allocations per chunk), 17.5 for `k = 100Ki`                                     | about 2x on these inputs                                                            |

The Parquet decode is the `parquet` crate's (dictionary and plain decoding,
decompression, UTF-8 validation) and the string comparisons of filters are
Arrow's kernels: they are left to those projects, and have no issue here.
