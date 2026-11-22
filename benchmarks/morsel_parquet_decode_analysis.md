<!--
SPDX-License-Identifier: Apache-2.0
SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements
-->

# Parquet decode and wait breakdown (#155)

This is a warm-page-cache investigation of the fork's experimental morsel executor, not an upstream DataFusion or DuckDB comparison. The measured base was `c2809d3b3ee73936b182429e1646af290d797c6e` (the `morsel-execution` merge of #138, after #145 and before #148–#150).

**Bottom line:** The eight-worker scan gap is primarily a scaling problem, not the cost of boxing the Parquet stream. Native spends roughly 2.3 aggregate worker-seconds waiting for ready readers per scan; simply enlarging queues did not fix the median, and removing per-batch yields made starvation worse. Output backpressure is small at the tested base but was visible in the fastest wide-producer scans. The clearest independent CPU saving is direct mask counting: #145 bypasses Parquet today, and porting just its count fast path saved 3.6–8.2% in the one-worker tests.

## Measurement conditions

- `rob-tandy-dduck2`, 2026-11-22 UTC: Ubuntu 24.04.5 LTS, Linux `7.0.0-1012-aws`, 16 Arm Neoverse-N1 vCPUs, one thread per core. Runs used CPUs 0–7 except the separately labeled CPU-affinity control. The fixture was generated on the local instance-store Btrfs volume (`compress=zstd:1`); it was 311,529,117 bytes, with 16 files, 128 row groups and 16,777,216 rows.
- Rust 1.99.1 (`58cf8f9c3`), Clang/lld 18.1.3, Tokio 1.54.1, Arrow/Parquet 60.1.0, Python 3.12.10, NumPy 2.2.6 and PyArrow 20.0.0. Both builds used the pinned Cargo lockfile and `release-nonlto`. No CPU pinning was changed between modes within an experiment.
- Kept builds out of measured intervals. Waited for three consecutive one-second `vmstat -y` samples at ≥95% idle before launching each driver (which runs the oracle and warmup before timing). Separate launch snapshots showed 94–100% idle for the baseline and 85–100% for the scratch/profile passes; this was not an exclusive host reservation. Only Linux page-cache residency was controlled, not CPU caches, filesystem metadata caches or device caches.

## What stock and native actually decode

Both paths call `ParquetPushDecoder::try_next_reader`, then drain the same Arrow `ParquetRecordBatchReader` and call `DecoderProjection::map`. Stock does this inside each partition's `FileStream`: it awaits ranges, makes an active reader, and drains that reader before advancing the file. Native's producer drives `try_next_reader` and publishes one buffered reader per row group; a worker drains it as a boxed synchronous iterator-backed stream. Native explicitly yields after every batch and again at row-group completion; stock's forward decoder does not yield explicitly between batches. The explicit native yields are **after** `consume` and outside the `decode` timer.

With row-level pushdown off, both arms read only `row_id`, `score`, `delta`, `payload` for scans and only `score`, `delta` for counts. That is scan projection, not #145's post-predicate projection. #145's single-worker scan optimization checks `Source::Memory` at lowering. Its direct `mask.true_count()` is in `run_memory_worker` (not restricted to one worker), not `run_parquet_worker`. Native Parquet counts apply the residual filter and an empty projection in `consume` before updating the count. The scratch scan arm runs the existing filter-input pruning on Parquet, avoiding materializing `score` after evaluating the full-width predicate. The scratch count arm reuses the existing filter-shape guard and validated direct mask count; it does not change the source projection, read less Parquet, or use metadata to count.

## What the timers mean

- `decode` is the sum of elapsed time inside native reader setup (`into_stream`) and synchronous batch polls, not time awaiting data or time suspended after a `yield_now`. It includes `ParquetRecordBatchReader::next`, `DecoderProjection::map` and the wrappers. The producer's `try_next_reader` is **not** in this bucket. Even synchronous elapsed time can include OS descheduling.
- `process` is filter/projection/count work after decode. `ready` and `output` sum worker suspensions while claiming readers and reserving result slots, respectively, including the delay before the runtime polls the task again. `ready` includes the consumer mutex and EOF. `pending I/O` sums producer suspensions while polling pending planner phases (including metadata), excluding waits for the I/O semaphore or ready-channel permits. It is not disk latency and it can overlap worker waits.
- None of these columns partitions query wall time. Do not compare a sum across workers directly with stock wall time or add producer pending I/O to worker idle time. Queue peaks are published entries only; ready bytes are footer estimates for **all** columns, not live allocated memory, and output bytes are reachable Arrow backing buffers, not total process memory.

## Baseline: stock vs. native at 1, 4 and 8 workers

Eight timed samples per mode/query/width after one warmup. All 192 timed samples began and ended at **76,063/76,063** resident 4 KiB pages. Both engines matched the oracle: 759,841 low and 9,883,940 high qualifying rows, including nullable projected values. Both reported 196,691,567 scan bytes or 75,280,626 count bytes (`bytes_scanned`, not physical I/O). Native claimed and completed all 128 row groups in every sample, and finalized each count once. Stock produced no #138 reports. Native `scan_low` emitted 2,048 batches versus 93 stock at one worker and 96 stock at four/eight; `scan_high` emitted 2,048 in both modes.

Wall times are medians (observed min–max); CPU is median process CPU across all query and blocking threads. Both clocks include registration and planning through stream exhaustion.

| Workers | Query        |          Stock wall ms |         Native wall ms | Native / stock | Stock CPU ms | Native CPU ms |
| ------: | ------------ | ---------------------: | ---------------------: | -------------: | -----------: | ------------: |
|       1 | `scan_low`   | 1110.7 (1104.2–1114.4) | 1096.4 (1092.3–1103.9) |          0.99× |       1111.6 |        1186.7 |
|       1 | `scan_high`  | 1152.4 (1145.1–1161.3) | 1150.6 (1146.0–1162.2) |          1.00× |       1154.2 |        1238.2 |
|       1 | `count_low`  |    571.8 (568.6–574.4) |    563.1 (558.2–567.4) |          0.98× |        573.5 |         597.9 |
|       1 | `count_high` |    578.8 (573.1–614.7) |    593.3 (589.2–604.3) |          1.03× |        578.8 |         627.4 |
|       4 | `scan_low`   |    291.8 (290.8–293.4) |    572.1 (550.9–597.6) |          1.96× |       1146.8 |        1337.4 |
|       4 | `scan_high`  |    306.6 (304.3–308.4) |    584.5 (552.0–592.7) |          1.91× |       1223.5 |        1392.0 |
|       4 | `count_low`  |    149.1 (148.1–150.6) |    213.0 (188.2–235.5) |          1.43× |        582.5 |         656.4 |
|       4 | `count_high` |    150.7 (149.4–151.3) |    210.0 (187.7–224.7) |          1.39× |        588.6 |         674.3 |
|       8 | `scan_low`   |    174.0 (167.5–182.5) |    615.9 (612.0–619.0) |          3.54× |       1193.4 |        1435.2 |
|       8 | `scan_high`  |    200.5 (193.1–235.1) |    618.7 (614.0–623.4) |          3.09× |       1309.1 |        1487.0 |
|       8 | `count_low`  |      93.6 (88.7–107.4) |    245.5 (241.9–245.9) |          2.62× |        602.1 |         718.7 |
|       8 | `count_high` |      93.2 (89.3–107.4) |    246.3 (242.7–282.6) |          2.64× |        608.3 |         744.2 |

Native #138 durations in ms: sum within each run, then take the median of those sums. The claims/completions vector is from the run nearest the native wall median, ordered by worker ID. Each entry is **both** the claim and completion count for that worker; they matched per worker in every sample.

| Workers | Query        | Pending I/O | Ready wait | Decode | Process | Output wait | Per-worker claims = completions |
| ------: | ------------ | ----------: | ---------: | -----: | ------: | ----------: | ------------------------------- |
|       1 | `scan_low`   |       136.6 |        1.8 |  898.5 |   147.5 |         0.0 | 128                             |
|       1 | `scan_high`  |       124.9 |        1.8 |  901.4 |   197.6 |         0.0 | 128                             |
|       1 | `count_low`  |        64.3 |        0.8 |  420.0 |   120.8 |         0.0 | 128                             |
|       1 | `count_high` |        63.9 |        0.8 |  422.7 |   148.4 |         0.0 | 128                             |
|       4 | `scan_low`   |        89.9 |      597.2 |  992.3 |   153.5 |         0.4 | 37/33/30/28                     |
|       4 | `scan_high`  |        88.6 |      592.2 |  994.6 |   202.9 |         0.5 | 35/27/29/37                     |
|       4 | `count_low`  |        39.8 |      115.5 |  444.5 |   128.5 |         0.0 | 29/35/32/32                     |
|       4 | `count_high` |        39.0 |      101.1 |  439.6 |   155.1 |         0.0 | 32/31/31/34                     |
|       8 | `scan_low`   |        90.0 |     2339.4 | 1060.1 |   156.4 |         4.5 | 16/16/16/16/16/16/16/16         |
|       8 | `scan_high`  |        88.1 |     2298.6 | 1063.5 |   206.5 |         2.8 | 17/16/16/16/15/16/16/16         |
|       8 | `count_low`  |        42.1 |      772.3 |  476.7 |   131.7 |         0.0 | 16/16/16/16/15/16/16/17         |
|       8 | `count_high` |        42.0 |      754.8 |  475.5 |   158.2 |         0.0 | 16/16/16/16/16/16/16/16         |

Queue peaks are the maximum of the eight run peaks. Depth and byte peaks need not coincide.

| Workers | Query        | Ready depth | Ready compressed bytes (estimate) | Output depth | Output Arrow bytes |
| ------: | ------------ | ----------: | --------------------------------: | -----------: | -----------------: |
|       1 | `scan_low`   |           2 |                         4,868,745 |            2 |             12,488 |
|       1 | `scan_high`  |           2 |                         4,869,153 |            2 |            157,224 |
|       1 | `count_low`  |           2 |                         4,869,153 |            1 |                  8 |
|       1 | `count_high` |           2 |                         4,869,153 |            1 |                  8 |
|       4 | `scan_low`   |           2 |                         4,903,161 |            2 |             13,408 |
|       4 | `scan_high`  |           2 |                         4,903,161 |            2 |            158,680 |
|       4 | `count_low`  |           2 |                         4,903,161 |            1 |                  8 |
|       4 | `count_high` |           2 |                         4,903,161 |            1 |                  8 |
|       8 | `scan_low`   |           2 |                         4,903,161 |            2 |             13,504 |
|       8 | `scan_high`  |           2 |                         4,903,161 |            2 |            158,712 |
|       8 | `count_low`  |           2 |                         4,903,191 |            1 |                  8 |
|       8 | `count_high` |           2 |                         4,903,161 |            1 |                  8 |

One-worker wall times were within 3% of stock; native failed to keep scaling from four to eight. At eight, scans spent about 47% of available worker wall time waiting for a ready reader (`ready_wait / (workers × wall)`, using medians); counts spent about 39%. Claims were balanced by the end, which does not imply workers were continuously fed. The large 4-worker output wait in the #138 smoke run did **not** reproduce here: 0.5 ms median for `scan_high` at four, 2.8 ms at eight, even though the queue hit capacity. The 8-worker `scan_high` decode number (1,064 aggregate ms) is not comparable to the stock drain (201 wall ms). Stock used 1,309 ms of process CPU, and native used 1,487 ms, a much smaller gap than the wall ratio.

## Scratch controls

All variants kept the same SQL, file/row-group layout, scan options and Parquet column mask. Use the `native` arm from the **same scratch binary** as the denominator, not the separately built baseline.

| Arm                                     | Change relative to scratch native                                                                                                                                                    |
| --------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `direct`                                | Poll the same `BufferedReader` synchronously on the existing morsel trait object, bypassing the second allocation/dispatch for a boxed `Stream`; keep the schema mapping and yields. |
| `row_group_yield`                       | Drop the explicit per-batch yield only; retain per-row-group yields, result send awaits and cancellation checks.                                                                     |
| `direct_row_group_yield`                | Apply both changes.                                                                                                                                                                  |
| `producer8`                             | Increase file lanes, I/O slots and ready slots from 4/2/2 to 8/8/8. Keep the reserve-before-`plan` design, the I/O task spawn and the mutex; this is not a prototype of all of #148. |
| `output32`                              | Increase only the output capacity from 2 to 32; do not coalesce batches. This is not #149.                                                                                           |
| `both_queues`                           | Apply `producer8` and `output32`.                                                                                                                                                    |
| `unmeasured`, `output32_unmeasured`     | Disable detailed metrics to check the observer effect at the two output capacities, especially the buffer-union accounting under the queue mutex.                                    |
| `prune`, `mask`                         | Apply one of #145's optimizations to Parquet at one worker (scan and count, respectively).                                                                                           |
| `stock_blocking16`, `native_blocking16` | Increase only the Tokio blocking cap from 8 to 16 while keeping an eight-CPU affinity. The separate 16-CPU affinity control leaves the two thread caps at 8.                         |

The `--profile` run separately brackets synchronous `reader.next()`, `try_next_reader()` and `DecoderProjection::map` calls with both wall and thread CPU clocks. It brackets `get_byte_ranges` with wall time (including read dispatch and runtime repoll; excludes semaphore acquisition). These probes are opt-in. Thread clock calls and aggregate atomics perturb the profile, especially for `map`, so use it to compare the same operation under the same probes, not to sum a query timeline. `LocalFileSystem::get_ranges` runs on Tokio's blocking pool; Parquet CPU work is on async workers (or the driver for work polled there), not on that blocking pool. Both use CPUs from the same affinity mask.

### Eight workers: stream dispatch, yield placement and the two queues

Wall time (ms), median (min–max), six timed samples per cell after one warmup. All arms passed the same parity/path/cache checks as the baseline.

| Arm                      |          `scan_low` |         `scan_high` |         `count_low` |        `count_high` |
| ------------------------ | ------------------: | ------------------: | ------------------: | ------------------: |
| `native`                 | 619.5 (616.3–622.5) | 624.2 (619.0–625.6) | 245.8 (243.8–248.6) | 246.4 (244.8–249.4) |
| `unmeasured`             | 618.9 (615.9–621.8) | 621.0 (615.8–642.7) | 246.1 (242.4–248.3) | 246.7 (244.6–247.8) |
| `direct`                 | 619.4 (617.0–624.0) | 621.8 (619.4–626.4) | 245.6 (243.7–248.2) | 247.4 (246.5–249.2) |
| `row_group_yield`        | 928.3 (920.1–933.7) | 969.2 (959.4–978.4) | 454.8 (453.7–457.7) | 477.0 (471.5–480.7) |
| `direct_row_group_yield` | 923.4 (913.3–939.3) | 973.5 (966.8–977.0) | 453.8 (450.7–460.3) | 476.1 (472.1–482.4) |
| `producer8`              | 622.5 (614.3–632.0) | 625.4 (306.6–629.0) | 249.0 (244.7–252.0) | 248.7 (246.8–251.2) |
| `output32`               | 619.1 (615.0–650.7) | 623.0 (619.7–627.3) | 246.7 (244.8–248.4) | 247.3 (243.6–249.4) |
| `both_queues`            | 626.0 (621.5–629.0) | 627.1 (606.6–627.6) | 249.0 (247.8–252.2) | 251.5 (214.7–254.1) |
| `output32_unmeasured`    | 618.2 (615.5–622.1) | 621.6 (619.4–626.6) | 245.7 (243.1–251.0) | 248.0 (246.1–253.1) |

For the two high-selectivity cases, the following are median process CPU and median per-run summed stages (ms); depth is the maximum of run peaks. The driver also emits per-worker and peak-byte metrics for every measured arm and case.

| Query        | Arm               |    CPU | Pending I/O | Ready wait | Decode | Process | Output wait | Peak ready / output depth |
| ------------ | ----------------- | -----: | ----------: | ---------: | -----: | ------: | ----------: | ------------------------: |
| `scan_high`  | `native`          | 1493.4 |        86.7 |     2324.8 | 1068.5 |   208.9 |         2.3 |                     2 / 2 |
| `scan_high`  | `direct`          | 1490.2 |        86.7 |     2318.8 | 1067.0 |   205.7 |         3.0 |                     2 / 2 |
| `scan_high`  | `row_group_yield` | 1456.8 |        88.8 |     4445.5 | 1060.3 |   200.5 |         0.7 |                     2 / 2 |
| `scan_high`  | `producer8`       | 1511.3 |       126.1 |     2320.6 | 1066.8 |   206.9 |         9.3 |                     8 / 2 |
| `scan_high`  | `output32`        | 1490.6 |        89.2 |     2319.1 | 1067.2 |   206.5 |         0.0 |                    2 / 32 |
| `scan_high`  | `both_queues`     | 1515.4 |       130.7 |     2320.5 | 1070.1 |   209.0 |         0.3 |                    8 / 32 |
| `count_high` | `native`          |  745.4 |        42.3 |      754.6 |  476.5 |   159.1 |         0.0 |                     2 / 1 |
| `count_high` | `direct`          |  746.5 |        41.5 |      761.9 |  477.6 |   159.0 |         0.0 |                     2 / 1 |
| `count_high` | `row_group_yield` |  709.4 |        38.1 |     2178.3 |  472.9 |   150.4 |         0.0 |                     2 / 1 |
| `count_high` | `producer8`       |  752.5 |        58.7 |      765.3 |  475.8 |   160.1 |         0.0 |                     8 / 1 |
| `count_high` | `output32`        |  746.4 |        43.8 |      755.5 |  476.9 |   158.5 |         0.0 |                     2 / 1 |
| `count_high` | `both_queues`     |  757.0 |        61.6 |      781.2 |  481.9 |   160.0 |         0.0 |                     8 / 1 |

- **Not stream boxing.** Direct polling changed every wall median by less than 0.5%, with overlapping ranges, and barely moved `decode`. Disabling #138 likewise gave no consistent wall improvement, including at output capacity 32.
- **Yields help scheduling on this path.** Removing only the batch yield made the scans 1.50–1.55× slower and counts 1.85–1.94× slower despite _reducing_ process CPU. Ready wait nearly doubled for `scan_high` and nearly tripled for `count_high` with little change to decode or pending I/O. Removing both the stream and the batch yield did not recover the loss. This does not isolate which Tokio task or wakeup is delayed, but rules out simply charging the yielded interval to the synchronous decode timer or treating every yield as removable overhead.
- **More slots alone did not fix the median.** Eight file lanes/I/O/ready slots actually filled an eight-reader queue, but left aggregate ready wait near baseline at the median. `producer8` did have two much faster `scan_high` drains (414.7 and 306.6 ms, versus 625–629 ms in the other four). Ready wait fell to 1,020 and 303 ms, while output wait rose to 269 and 375 ms respectively; decode stayed at 1,039 and 1,030 ms. Those runs show output backpressure can emerge when workers are fed. They are not a reliable capacity-only speedup and do not identify the cause of the scheduling variability.
- **Output capacity alone was not the base bottleneck.** At 32 slots the median output wait was zero, but the original queue already cost little in these runs. Widening both queues did not improve the medians. This only tests the channel capacity, not #149's pre-projection coalescing; the 2,048 vs. 96 `scan_low` batches remain a separate optimization target.

### Eight workers: decoder phase CPU vs. elapsed time

Separate instrumented run, six timed samples per cell. Both engines made 272 `try_next_reader` calls (including requests for data and file EOF), 128 `get_byte_ranges` calls, 2,176 calls to `reader.next` (including row-group EOF) and 2,048 projection mappings in every sample. Values are medians of sums in ms, except query wall median (min–max). The range clock covers only row-group data reads, whereas #138's pending I/O covers other planner I/O as well.

| Query        | Arm                 |          Query wall | Batch next CPU | Batch next elapsed | Reader setup CPU | Projection map CPU | Range elapsed |
| ------------ | ------------------- | ------------------: | -------------: | -----------------: | ---------------: | -----------------: | ------------: |
| `scan_high`  | `stock`             | 194.5 (193.1–216.2) |          950.1 |              961.9 |              5.0 |                3.4 |         146.3 |
| `scan_high`  | `native`            | 622.7 (621.7–626.8) |         1064.1 |             1065.3 |              4.8 |                3.7 |          90.3 |
| `scan_high`  | `stock_blocking16`  | 196.4 (189.3–200.5) |          948.7 |              962.9 |              4.8 |                3.5 |         144.0 |
| `scan_high`  | `native_blocking16` | 624.0 (610.2–631.1) |         1062.9 |             1063.4 |              5.0 |                3.6 |          90.7 |
| `scan_high`  | `row_group_yield`   | 972.6 (971.0–986.0) |         1051.2 |             1050.4 |              5.0 |                3.5 |          92.0 |
| `scan_high`  | `both_queues`       | 625.5 (617.8–631.0) |         1061.8 |             1063.6 |              5.6 |                3.6 |         128.5 |
| `count_high` | `stock`             |    90.9 (88.5–95.8) |          427.1 |              426.4 |              3.0 |                2.6 |          66.8 |
| `count_high` | `native`            | 247.8 (245.6–253.0) |          475.5 |              473.6 |              3.4 |                2.7 |          47.2 |
| `count_high` | `stock_blocking16`  |   96.8 (94.3–108.7) |          427.3 |              427.7 |              3.0 |                2.6 |          87.9 |
| `count_high` | `native_blocking16` | 246.3 (244.1–250.3) |          474.0 |              471.8 |              3.4 |                2.7 |          43.2 |
| `count_high` | `row_group_yield`   | 487.1 (476.9–489.3) |          475.2 |              473.2 |              3.3 |                2.6 |          41.8 |
| `count_high` | `both_queues`       | 249.3 (247.7–255.3) |          475.7 |              473.9 |              3.8 |                2.8 |          64.5 |

There is a real **11–12% increase in `reader.next()` CPU** on the native eight-worker reference (114 ms scan / 48 ms count), not a 3× decoder. The CPU and synchronous elapsed sums are close (the CPU clock brackets a slightly wider interval, so can read higher). Most of the native decode timer is real batch decoding; neither reader creation nor `DecoderProjection::map` accounts for the residual CPU gap. Changing queue or yield settings did little to this CPU difference in these paired tests, even when it changed query wall substantially. The stock range sum is larger despite shorter query wall: concurrent I/O sums measure neither starvation nor disk latency. A 16-thread blocking cap did not recover wall time or batch CPU in native. This cap test is not a change to the native I/O semaphore.

Adding CPU headroom (16 allowed CPUs but still 8 async workers and an 8-thread blocking cap) changed instrumented stock wall from 194.5 to 164.8 ms on `scan_high` and 90.9 to 82.7 ms on `count_high`. Native changed from 622.7 to 619.7 and 247.8 to 244.7 ms respectively (ranges overlapped), with 1,057.7 and 471.8 ms of batch CPU. Allowed CPU headroom helped stock, but neither it nor a higher blocking cap explained native's long ready waits. These controls do not measure time enqueued inside `spawn_blocking` separately from other scheduling delays.

### Hardware counters and Zstd/allocator samples

Five matched `perf stat` runs per row on CPUs 0–7, from the scratch binary with phase probes off. These are whole-process user-space event medians, including query setup/teardown; they are not per-decoder counters. `r17` is Arm PMUv3 `l2d_cache_refill` (`event=0x0017` on this host). The three listed events reported 100% running in each trial.

| Query        | Arm               | Cycles (billions) | Retired instructions (billions) | L2 data refills (millions) |
| ------------ | ----------------- | ----------------: | ------------------------------: | -------------------------: |
| `scan_high`  | `stock`           |             2.790 |                           7.558 |                      9.866 |
| `scan_high`  | `native`          |             2.863 |                           7.660 |                      9.804 |
| `scan_high`  | `row_group_yield` |             2.829 |                           7.655 |                      8.528 |
| `scan_high`  | `producer8`       |             2.857 |                           7.660 |                      9.780 |
| `scan_high`  | `both_queues`     |             2.857 |                           7.659 |                      9.532 |
| `count_high` | `stock`           |             1.349 |                           3.570 |                      3.640 |
| `count_high` | `native`          |             1.496 |                           3.889 |                      4.485 |
| `count_high` | `row_group_yield` |             1.452 |                           3.885 |                      3.250 |
| `count_high` | `producer8`       |             1.491 |                           3.889 |                      4.443 |
| `count_high` | `both_queues`     |             1.501 |                           3.889 |                      4.464 |

Wider queues did not raise aggregate L2 refill counts over native. Native count had 23% more refills than stock, but the no-batch-yield arm lowered refills and had nearly the same batch CPU while taking almost twice as long. This does not support L2 capacity misses from queued row groups as the dominant throughput cause. The counters cannot attribute the count difference to particular arrays, cache residency or CPU migration; an L3 refill event was not exposed on this host.

One `perf record` per stock/native high-selectivity case at 3,999 Hz (`cycles:u`, leaf symbols, roughly 4–8k samples each, no lost samples) showed the same major Zstd sequence/Huffman symbols and Parquet record-reader buffer code in both modes. Visible `ZSTD_`/`HUF_`/`FSE_` symbols were 54% of sampled cycles stock vs. 52% native for the scan, and 42% vs. 39% for the count. Recognizable allocator symbols (`malloc`/`calloc`/`realloc`/`free`/`allocate_aligned`) were about 1.3–1.5% across the four recordings. This is not an allocation trace: it does not measure allocation counts, cross-thread buffer lifetimes, or time in inlined allocation/zero-fill. It shows no new dominant decompressor or allocator hotspot in native and does not identify the cause of the remaining 11–12% batch CPU gap.

### One worker: #145 fast paths transplanted to Parquet

Six timed samples per arm after one warmup. `Process` is the same #138 bucket in both arms; timings are medians in ms (wall ranges in parentheses).

| Query        | Ablation |            Native wall |           Ablated wall | Native → ablated process | Native → ablated decode |
| ------------ | -------- | ---------------------: | ---------------------: | -----------------------: | ----------------------: |
| `scan_low`   | `prune`  | 1101.9 (1092.2–1111.9) | 1094.3 (1088.8–1112.5) |            147.2 → 142.7 |           901.3 → 899.8 |
| `scan_high`  | `prune`  | 1152.0 (1141.2–1160.6) | 1134.4 (1129.5–1139.1) |            196.8 → 183.6 |           901.7 → 900.5 |
| `count_low`  | `mask`   |    565.0 (558.3–570.2) |    544.5 (541.8–548.3) |            121.2 → 102.0 |           420.3 → 420.1 |
| `count_high` | `mask`   |    593.9 (587.7–597.6) |    545.4 (539.7–549.2) |            148.6 → 102.0 |           422.8 → 420.6 |

The direct mask count saved 19–47 ms of processing, or **3.6–8.2% wall time**, without changing decoder work. The high/low count difference nearly disappeared with no filtered arrays to materialize. Scan pruning saved 5–13 ms of processing and 0.7–1.5% wall by the medians, with overlapping `scan_low` wall ranges. Neither optimization is a Parquet decompression improvement.

## Priorities after #148–#150

1. **Re-measure the producer fix and coalescing before tuning decode.** Treat the wide-producer run as a capacity control, not an estimate of #148. If ready wait remains high, separately trace time waiting for a publication permit, an I/O slot, the receiver mutex, and the wake-to-poll delay. Keep per-batch yields in the reference; any new cooperative budget needs its own throughput and first-8,192-row latency test. Run #149's capacity and coalescing arms separately: a fast producer can expose output wait, and coalescing changes projection batch size even when output wait is zero.
2. **Port the direct masked count for eligible Parquet count shapes.** Avoid filtered array materialization when only the row count is used. The 19–47 ms processing saving at one worker is independent of buffer capacity and is not a substitute for #150. Preserve NULL/mask-length behavior and use the general path for other operator shapes. Recheck the guard with pushed versus residual predicates when #150 changes the path.
3. **Consider narrowing scan filter output only if still useful after coalescing/late materialization.** The high-selectivity gain was 18 ms wall at one worker; the low-selectivity ranges overlapped. Do not conflate this with reading fewer predicate/payload columns.
4. **Investigate the residual `reader.next()` CPU only if it persists against the new baseline.** Compare producer-to-worker handoff with same-thread reader creation/consumption at matched input bytes, reader depth and active parallelism; use per-thread cache counters and attributed allocation samples. Neither the stream wrapper nor a new Zstd or allocator hotspot is a supported first target.

## Reproduce

From the fork worktree. The scripts and patches are [supporting material](morsel_parquet_decode_analysis/) for this investigation; apply patches **only** in a disposable checkout. The runner is copied unchanged from [#128](morsel_parquet/README.md) before the metrics-only patch. Pre-read and validate Linux file page residency before and after every timed sample via `mincore`. Use a quiet local host, do not build while sampling, and pin the same eight cores for the three worker counts (adapt CPU IDs to the host). CPU affinity is not reservation.

```bash
set -eu
repo=$PWD
analysis="$repo/benchmarks/morsel_parquet_decode_analysis"
harness="$repo/benchmarks/morsel_parquet"
rev=c2809d3b3ee73936b182429e1646af290d797c6e
work=$(mktemp -d "$HOME/.cache/issue155.XXXXXX")
scratch="$work/scratch"
git worktree add --detach "$scratch" "$rev"
mkdir -p "$scratch/datafusion/core/examples"
cp "$harness/stream_runner.rs" "$scratch/datafusion/core/examples/morsel_parquet_stream.rs"
git -C "$scratch" apply --unidiff-zero "$analysis/runner.patch"

build() (
  cd "$scratch"
  MORSEL_BENCH_REVISION="$rev" CARGO_TARGET_DIR="$work/target" RUSTC_WRAPPER=sccache \
    timeout --kill-after=15s 1500s cargo build --locked --offline \
      --profile release-nonlto -j 12 -p datafusion \
      --no-default-features --features parquet,sql --example morsel_parquet_stream
)
build
cp "$work/target/release-nonlto/examples/morsel_parquet_stream" "$work/baseline"
# Omit --offline if Cargo's dependency cache hasn't been populated; omit
# RUSTC_WRAPPER=sccache if sccache isn't installed.
timeout 120s uv venv --python 3.12.10 "$work/venv"
py="$work/venv/bin/python"
timeout 180s uv pip install --python "$py" -r "$harness/requirements.txt"
timeout 60s "$py" -B "$harness/harness.py" generate "$work/large" \
  --files 16 --groups 8 --group-rows 131072

sample() {
  timeout --kill-after=15s 1200s taskset -c 0-7 "$py" -B \
    "$analysis/measure.py" "$1" "$work/large" "$rev" "${@:2}"
}
for workers in 1 4 8; do
  vmstat -y 1 3
  sample "$work/baseline" "$workers" >"$work/base-$workers.jsonl"
done

git -C "$scratch" apply --unidiff-zero "$analysis/ablations.patch"
build
cp "$work/target/release-nonlto/examples/morsel_parquet_stream" "$work/ablations"
vmstat -y 1 3
sample "$work/ablations" 1 --variants native prune mask --repeats 6 >"$work/single.jsonl"
sample "$work/ablations" 8 --variants native unmeasured direct row_group_yield \
  direct_row_group_yield producer8 output32 both_queues output32_unmeasured \
  --repeats 6 >"$work/controls.jsonl"
sample "$work/ablations" 8 --profile \
  --variants stock native stock_blocking16 native_blocking16 row_group_yield both_queues \
  --cases scan_high count_high --repeats 6 >"$work/profile.jsonl"
timeout --kill-after=15s 1200s taskset -c 0-15 "$py" -B "$analysis/measure.py" \
  "$work/ablations" "$work/large" "$rev" 8 --profile --variants stock native \
  --cases scan_high count_high --repeats 6 >"$work/affinity16.jsonl"
# Optional Arm PMUv3 pass (requires perf and user-space event access).
# Set PERF_BIN to use an extracted perf binary instead of the one on PATH.
vmstat -y 1 3
timeout --kill-after=15s 600s taskset -c 0-7 "$py" -B \
  "$analysis/profile_perf.py" "$work" "${PERF_BIN:-$(command -v perf)}" >"$work/perf.jsonl"

# After the investigation, remove the disposable checkout (never modify tracked engine files here).
git worktree remove --force "$scratch"
```

The hardware pass used `perf` 6.8.12 from Ubuntu's `linux-tools-6.8.0-149` package. The host did not have `perf` on `PATH`; the package and `libslang2`, `libnuma1` and `libtraceevent1` were fetched with `apt download` and extracted locally with `dpkg-deb -x`. `PERF_BIN` can point to that binary with the extracted library folders on `LD_LIBRARY_PATH`; no system installation or privileges are required where user events are enabled.

The measurement boundary matches #128: external-table registration through stream exhaustion, no CSV collection or metric formatting during timing. The driver performs the #128 full NumPy oracle before timing each engine/variant/worker configuration. It verifies the expected physical path, source bytes, output row count and complete claim accounting on every sample. Each case discards one warmup, then rotates variant order by repeat. Batch size is 8,192; target partitions, Tokio async workers and the blocking-thread cap all equal the configured workers unless explicitly varied. As in #128, row-level pushdown, statistics/page-index/Bloom pruning, file splitting and listing statistics are disabled. The timing window includes planning; the reported phase clocks do not include it.

## Verification scope

- The two locked release builds in the recipe succeeded. The baseline and scratch drivers logged 552 measured query runs with verified warm residency; the separate hardware pass logged 50 counter runs and four recordings. The experiment driver and existing stream-runner tests serve as the reproduction and correctness check; no CI wall-time threshold or engine behavior was added.
- `MORSEL_PARQUET_RUNNER="$work/baseline" MORSEL_BENCH_REVISION="$rev" timeout 120s "$py" -B -m unittest discover -s "$harness" -v`: four passed, two legacy CLI-only tests skipped (`MORSEL_STOCK_CLI` was not set; this compares the stream runner).
- `timeout 50s cargo fmt --all -- --check` and `CARGO_TARGET_DIR="$work/lint" RUSTC_WRAPPER=sccache timeout --kill-after=15s 600s taskset -c 0-7 cargo clippy --locked --offline -j 8 --all-targets --all-features -- -D warnings`: passed.
- `timeout 120s ./ci/scripts/doc_prettier_check.sh`, `npx --offline prettier@3.10.1 --check benchmarks/morsel_parquet_decode_analysis.md` (not covered by the repository script), and `git diff --check`: passed.
- No extended Rust test suite, cold-cache run or post-#148–#150 implementation was tested: this is a warm-cache report with scratch controls, not an engine patch.
