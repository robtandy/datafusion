// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Microbenchmarks of the important operators of the pipeline engine on
//! mocked inputs: every operator runs in isolation on synthetic `ArrayRef`s
//! and `DataChunk`s, without SQL and without a planner, over parameter
//! sweeps, and the benchmark prints, for every configuration, the time per
//! row, the throughput, the allocations, the bytes copied and the share of a
//! memory-bandwidth floor that the operator achieves. The results and their
//! analysis are in `docs/operator-performance.md`.
//!
//! As the crate has no `harness = false` benchmark targets, the benchmark is
//! the libtest test `operator_micro`:
//!
//! ```text
//! cargo test --profile release-nonlto -p datafusion-physical-pipeline --bench operator_micro -- --nocapture
//! ```
//!
//! # How an operator is measured
//!
//! Every configuration builds an operator tree `DiscardSink(operator(ChunkSource...))`:
//! `ChunkSource`s hand out pre-built chunks (zero-copy slices of synthetic
//! arrays) to the threads of their pipelines, and the `DiscardSink` drops the
//! chunks of the operator. The `Executor` runs the tree with profiling on, so
//! that the numbers are those of the query profiles of `docs/profiling.md`
//! and line up with them: the time of the operator is the `self_time` of its
//! stages (and the `finalize_time` of its meta pipeline), its rows are its
//! `rows_in`, and `bytes_in`, `bytes_out` and `memory_peak` are its metrics
//! of the same names. The hash join is measured twice: its build (the
//! pipelines whose sink it is, with the `JoinHashTableBuild` pipeline, and
//! its finalization) and its probe (its stage in the probe pipeline).
//! Tuple-data ingestion is a data structure rather than an operator: its
//! benchmark appends the chunks to `PartitionedTupleData`s directly, on as
//! many threads as configured.
//!
//! The first execution of every configuration is a warm-up that checks the
//! output of the operator (its rows, and e.g. their order for a sort); the
//! numbers are the medians of the timed executions that follow.
//!
//! # Columns of the report
//!
//! | Column           | Meaning                                                                                                                                         |
//! | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
//! | `rows`           | the rows of the operator: its `rows_in` (the probe rows of a probe, the build rows of a build), its `rows_out` for a source (Parquet decode)       |
//! | `ns/row`         | `self_time / rows`: the time of the operator per row, summed over its threads (CPU time)                                                        |
//! | `Mrows/s`        | `rows` per second of the wall time of the operator (the elapsed time of its pipelines, attributed to the operator by its share of their active time) |
//! | `allocs/chunk`   | the heap allocations of the operator per chunk of its input (`morsels_in`), net of those of the harness                                           |
//! | `copied B/row`   | the bytes that the operator allocated per row, net of the harness: every byte it copies (gathered, computed or stored in rows, keys and tables) lands in memory it allocated |
//! | `in B/row`       | `bytes_in / rows`, see `profiling::chunk_bytes`                                                                                                 |
//! | `out B/row`      | `bytes_out / rows` (the bytes of the rows a tuple-data collection holds, for tuple-data ingestion)                                              |
//! | `memory_peak`    | the high-water mark of the memory that the operator reserved                                                                                    |
//! | `floor B/row`    | the bytes per row that the operator must read and write, whatever its implementation (see `docs/operator-performance.md`)                         |
//! | `% floor`        | the time it takes to move the floor bytes at the bandwidth of the machine (on as many threads), over the wall time of the operator              |
//! | `cv %`           | the coefficient of variation of `ns/row` over the timed executions                                                                              |
//! | `sink/fin/src %` | the shares of the time of a pipeline breaker spent as a sink, finalizing, and as a source (the `JoinHashTableBuild` for a hash join build)       |
//!
//! The allocations are counted by the global allocator of this benchmark,
//! which counts the allocations of all threads (`alloc`, `alloc_zeroed` and
//! `realloc`, the latter with its new size) in sharded counters before it
//! calls the system allocator. The harness allocations are those of
//! executions of the same plan without the measured part (the sources into a
//! `DiscardSink`, and for a probe the join with an empty probe side), which
//! are subtracted.
//!
//! The bandwidth of the machine is measured first, with streaming reads,
//! writes and copies of buffers far larger than the last-level cache on 1, 4
//! and 16 threads (every thread its own buffers). The time of the floor is
//! the largest of three lower bounds: reading the bytes that the operator
//! must read at the read bandwidth, writing those it must write at the write
//! bandwidth, and moving both at the bandwidth of the copy.
//!
//! # Configuration
//!
//! | Variable                     | Meaning                                                                                             | Default (optimized / debug build) |
//! | ---------------------------- | --------------------------------------------------------------------------------------------------- | --------------------------------- |
//! | `OPERATOR_MICRO_SCALE`       | factor of the input sizes (4Mi rows for most operators)                                             | `1` / `0.01`                      |
//! | `OPERATOR_MICRO_ITERATIONS`  | timed executions of every configuration                                                             | `5` / `1`                         |
//! | `OPERATOR_MICRO_THREADS`     | thread counts of the partition sweeps, comma-separated                                              | `1,4,16` (at most the cores)      |
//! | `OPERATOR_MICRO_FILTER`      | substrings separated by `\|`: only the configurations whose `operator configuration ` contains one run | all                             |
//! | `OPERATOR_MICRO_SAVE`        | a file to write the results to, as tab-separated values                                             | none                              |
//! | `OPERATOR_MICRO_COMPARE`     | a file of earlier results (`OPERATOR_MICRO_SAVE`) to compare `ns/row` with                           | none                              |
//! | `OPERATOR_MICRO_BANDWIDTH_MB`| the MiB of the buffers of the bandwidth measurement                                                 | `512` / `16`                      |
//! | `OPERATOR_MICRO_COUNT_ALLOCATIONS` | whether to count the allocations (`false` measures what counting them costs)                   | `true`                            |
//!
//! To profile a configuration, run the test binary under `perf record` with
//! a filter, e.g.
//!
//! ```text
//! OPERATOR_MICRO_FILTER="hash_join_probe hit=0.50 build=1M dups=1 key=i64 batch=2048 threads=1 " \
//!     perf record -g -- target/release-nonlto/deps/operator_micro-<hash> operator_micro --nocapture
//! ```

use std::collections::HashMap;
use std::fmt::Write as _;
use std::hint::black_box;
use std::io::Write as _;
use std::path::PathBuf;
use std::str::FromStr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Barrier};
use std::time::Duration;

use arrow::array::{
    Array, ArrayRef, AsArray, Float64Array, Int32Array, Int64Array, RecordBatch,
    StringViewArray, StringViewBuilder,
};
use arrow::compute::SortOptions;
use arrow::datatypes::{
    DataType, Field, Float64Type, Int32Type, Int64Type, Schema, SchemaRef,
};
use bytes::Bytes;
use datafusion::functions_aggregate::count::count_udaf;
use datafusion::functions_aggregate::sum::sum_udaf;
use datafusion::functions_window::row_number::row_number_udwf;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_common::instant::Instant;
use datafusion_common::{
    DataFusionError, JoinType, NullEquality, Result, internal_datafusion_err,
    internal_err,
};
use datafusion_datasource::PartitionedFile;
use datafusion_datasource::file_groups::FileGroup;
use datafusion_datasource::file_scan_config::FileScanConfigBuilder;
use datafusion_datasource::memory::MemorySourceConfig;
use datafusion_datasource_parquet::source::ParquetSource;
use datafusion_execution::TaskContext;
use datafusion_execution::object_store::ObjectStoreUrl;
use datafusion_expr::{Operator, WindowFrame, WindowFunctionDefinition};
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::aggregate::AggregateExprBuilder;
use datafusion_physical_expr::expressions::{binary, col, lit};
use datafusion_physical_expr_common::sort_expr::{LexOrdering, PhysicalSortExpr};
use datafusion_physical_pipeline::aggregate::PhysicalHashAggregate;
use datafusion_physical_pipeline::join::PhysicalHashJoin;
use datafusion_physical_pipeline::operator::{downcast_state, downcast_state_mut};
use datafusion_physical_pipeline::operators::filter::PhysicalFilter;
use datafusion_physical_pipeline::operators::projection::PhysicalProjection;
use datafusion_physical_pipeline::profiling::{OperatorRole, QueryProfile, chunk_bytes};
use datafusion_physical_pipeline::sort::{PhysicalSort, PhysicalTopK};
use datafusion_physical_pipeline::sources::ParquetScanSource;
use datafusion_physical_pipeline::tuple_data::{PartitionedTupleData, TupleDataLayout};
use datafusion_physical_pipeline::window::PhysicalWindow;
use datafusion_physical_pipeline::{
    DEFAULT_VECTOR_SIZE, DataChunk, ExecutionContext, Executor, GlobalSourceState,
    LocalSinkState, PhysicalOperator, SelectionVector, SinkResult, SourceResult,
};
use datafusion_physical_plan::aggregates::{
    AggregateExec, AggregateMode, PhysicalGroupBy,
};
use datafusion_physical_plan::windows::create_window_expr;
use object_store::memory::InMemory;
use object_store::path::Path as ObjectPath;
use object_store::{ObjectStore, ObjectStoreExt, PutPayload};
use parking_lot::Mutex;
use parquet::arrow::ArrowWriter;
use parquet::basic::{Compression, ZstdLevel};
use parquet::file::properties::WriterProperties;

// ---------------------------------------------------------------------------
// Allocations
// ---------------------------------------------------------------------------

/// The global allocator of the benchmark: the system allocator, which counts
/// the allocations.
#[global_allocator]
static ALLOCATOR: allocations::Counting = allocations::Counting;

/// A global allocator that counts the allocations of all threads, and the
/// bytes they allocate, in sharded counters (one cache line per shard, a
/// shard per thread), so that the counting neither contends nor slows down
/// the threads of an execution measurably.
mod allocations {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

    const SHARDS: usize = 64;

    /// The counters of the threads of a shard, on a cache line of their own
    /// (Neoverse-N1 prefetches pairs of 64-byte lines).
    #[repr(align(128))]
    struct Shard {
        allocations: AtomicU64,
        bytes: AtomicU64,
    }

    static COUNTERS: [Shard; SHARDS] = [const {
        Shard {
            allocations: AtomicU64::new(0),
            bytes: AtomicU64::new(0),
        }
    }; SHARDS];

    /// The shard of the next thread that allocates.
    static NEXT_SHARD: AtomicUsize = AtomicUsize::new(0);

    /// Whether the allocations are counted.
    static COUNTING: AtomicBool = AtomicBool::new(true);

    /// Count the allocations (or not).
    pub fn set_counting(counting: bool) {
        COUNTING.store(counting, Ordering::Relaxed);
    }

    thread_local! {
        /// The shard of this thread, `usize::MAX` until it first allocates.
        static SHARD: Cell<usize> = const { Cell::new(usize::MAX) };
    }

    /// The allocations and the allocated bytes of all threads.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub struct Counts {
        pub allocations: u64,
        pub bytes: u64,
    }

    impl Counts {
        /// The counts since `earlier`.
        pub fn since(self, earlier: Self) -> Self {
            Self {
                allocations: self.allocations - earlier.allocations,
                bytes: self.bytes - earlier.bytes,
            }
        }

        pub fn add(self, other: Self) -> Self {
            Self {
                allocations: self.allocations + other.allocations,
                bytes: self.bytes + other.bytes,
            }
        }
    }

    /// The counts of all allocations so far.
    pub fn snapshot() -> Counts {
        COUNTERS
            .iter()
            .fold(Counts::default(), |counts, shard| Counts {
                allocations: counts.allocations
                    + shard.allocations.load(Ordering::Relaxed),
                bytes: counts.bytes + shard.bytes.load(Ordering::Relaxed),
            })
    }

    #[inline]
    fn record(bytes: usize) {
        if !COUNTING.load(Ordering::Relaxed) {
            return;
        }
        // A `const` thread local without a destructor: accessing it never
        // allocates.
        let shard = SHARD
            .try_with(|shard| {
                let mut index = shard.get();
                if index == usize::MAX {
                    index = NEXT_SHARD.fetch_add(1, Ordering::Relaxed) % SHARDS;
                    shard.set(index);
                }
                index
            })
            .unwrap_or(0);
        let counters = &COUNTERS[shard];
        counters.allocations.fetch_add(1, Ordering::Relaxed);
        counters.bytes.fetch_add(bytes as u64, Ordering::Relaxed);
    }

    /// The system allocator, counting its allocations.
    pub struct Counting;

    // SAFETY: every method forwards its arguments to the system allocator,
    // which upholds the contract of `GlobalAlloc`.
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            record(layout.size());
            // SAFETY: the caller upholds the contract of `GlobalAlloc::alloc`.
            unsafe { System.alloc(layout) }
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            record(layout.size());
            // SAFETY: the caller upholds the contract of
            // `GlobalAlloc::alloc_zeroed`.
            unsafe { System.alloc_zeroed(layout) }
        }

        unsafe fn realloc(
            &self,
            ptr: *mut u8,
            layout: Layout,
            new_size: usize,
        ) -> *mut u8 {
            record(new_size);
            // SAFETY: the caller upholds the contract of
            // `GlobalAlloc::realloc`.
            unsafe { System.realloc(ptr, layout, new_size) }
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            // SAFETY: the caller upholds the contract of
            // `GlobalAlloc::dealloc`.
            unsafe { System.dealloc(ptr, layout) }
        }
    }
}

use allocations::Counts;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// The value of the environment variable `name`, or `default`.
fn env_or<T: FromStr>(name: &str, default: T) -> T {
    match std::env::var(name) {
        Ok(value) => value
            .trim()
            .parse()
            .unwrap_or_else(|_| panic!("invalid value of {name}: {value}")),
        Err(_) => default,
    }
}

/// The configuration of the benchmark, see the module documentation.
#[derive(Debug, Clone)]
struct BenchConfig {
    scale: f64,
    iterations: usize,
    /// The thread counts of the partition sweeps.
    threads: Vec<usize>,
    filter: Vec<String>,
    save: Option<PathBuf>,
    compare: Option<PathBuf>,
    bandwidth_bytes: usize,
    count_allocations: bool,
}

impl BenchConfig {
    fn from_env() -> Self {
        // Unoptimized builds (e.g. `cargo test --all-targets`) only check
        // that the benchmark runs.
        let optimized = !cfg!(debug_assertions);
        let cores = std::thread::available_parallelism().map_or(4, |n| n.get());
        let threads = std::env::var("OPERATOR_MICRO_THREADS").map_or_else(
            |_| {
                [1, 4, 16]
                    .into_iter()
                    .filter(|&threads| threads <= cores)
                    .collect::<Vec<_>>()
            },
            |value| {
                value
                    .split(',')
                    .map(|threads| {
                        threads.trim().parse().unwrap_or_else(|_| {
                            panic!("invalid value of OPERATOR_MICRO_THREADS: {value}")
                        })
                    })
                    .collect()
            },
        );
        // The configurations have commas (`exprs=a+b,c`): the filters are
        // separated by `|`.
        let filter = std::env::var("OPERATOR_MICRO_FILTER").map_or_else(
            |_| vec![],
            |value| {
                value
                    .split('|')
                    .filter(|part| !part.trim().is_empty())
                    .map(str::to_owned)
                    .collect()
            },
        );
        let path = |name: &str| std::env::var(name).ok().map(PathBuf::from);
        Self {
            scale: env_or("OPERATOR_MICRO_SCALE", if optimized { 1.0 } else { 0.01 }),
            iterations: env_or(
                "OPERATOR_MICRO_ITERATIONS",
                if optimized { 5 } else { 1 },
            )
            .max(1),
            threads,
            filter,
            save: path("OPERATOR_MICRO_SAVE"),
            compare: path("OPERATOR_MICRO_COMPARE"),
            bandwidth_bytes: env_or(
                "OPERATOR_MICRO_BANDWIDTH_MB",
                if optimized { 512 } else { 16 },
            ) << 20,
            count_allocations: env_or("OPERATOR_MICRO_COUNT_ALLOCATIONS", true),
        }
    }
}

// ---------------------------------------------------------------------------
// Synthetic data
// ---------------------------------------------------------------------------

/// SplitMix64: a fast, deterministic generator of the synthetic data.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e37_79b9_7f4a_7c15);
        mix64(self.0)
    }

    /// A uniform value in `0..n`.
    fn below(&mut self, n: u64) -> u64 {
        ((u128::from(self.next_u64()) * u128::from(n)) >> 64) as u64
    }

    /// A uniform value in `[0, 1)`.
    fn unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1_u64 << 53) as f64
    }

    fn shuffle<T>(&mut self, values: &mut [T]) {
        for i in (1..values.len()).rev() {
            let j = self.below(i as u64 + 1) as usize;
            values.swap(i, j);
        }
    }
}

/// The finalizer of SplitMix64, a bijection of `u64`.
fn mix64(value: u64) -> u64 {
    let mut z = value;
    z = (z ^ (z >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
    z ^ (z >> 31)
}

/// The values of `i64` keys are their ids times this factor, so that they
/// spread over the range of `i64`: a hash join hashes them, while it indexes
/// the keys of `i64dense` and `i32` (whose range is narrow) directly.
const KEY_SPREAD: u64 = 0x9e37_79b9;

/// The kinds of the key columns of the sweeps over the key width. The keys
/// are made of ids: distinct ids have distinct keys, in the order of the ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum KeyKind {
    /// `Int32`, the id.
    I32,
    /// `Int64`, the id times [`KEY_SPREAD`].
    I64,
    /// `Int64`, the id.
    I64Dense,
    /// Two `Int64` columns: `id >> 8` and `id & 0xff`.
    I64x2,
    /// `Utf8View`, the id as 16 decimal digits (zero-padded): longer than the
    /// 12 bytes that a view inlines, with a common prefix.
    Utf8,
}

impl KeyKind {
    fn name(self) -> &'static str {
        match self {
            Self::I32 => "i32",
            Self::I64 => "i64",
            Self::I64Dense => "i64dense",
            Self::I64x2 => "2xi64",
            Self::Utf8 => "utf8v16",
        }
    }

    /// The names of the key columns, `{name}` and `{name}2`.
    fn names(self, name: &str) -> Vec<String> {
        match self {
            Self::I64x2 => vec![name.to_owned(), format!("{name}2")],
            _ => vec![name.to_owned()],
        }
    }

    fn fields(self, name: &str) -> Vec<Field> {
        let data_type = match self {
            Self::I32 => DataType::Int32,
            Self::I64 | Self::I64Dense | Self::I64x2 => DataType::Int64,
            Self::Utf8 => DataType::Utf8View,
        };
        self.names(name)
            .into_iter()
            .map(|name| Field::new(name, data_type.clone(), false))
            .collect()
    }

    /// The key columns of the keys with ids `ids`.
    fn columns(self, ids: &[u64]) -> Vec<ArrayRef> {
        let int64 = |f: &dyn Fn(u64) -> i64| -> ArrayRef {
            Arc::new(Int64Array::from_iter_values(ids.iter().map(|&id| f(id))))
        };
        match self {
            Self::I32 => vec![Arc::new(Int32Array::from_iter_values(
                ids.iter().map(|&id| id as i32),
            ))],
            Self::I64 => vec![int64(&|id| (id * KEY_SPREAD) as i64)],
            Self::I64Dense => vec![int64(&|id| id as i64)],
            Self::I64x2 => vec![
                int64(&|id| (id >> 8) as i64),
                int64(&|id| (id & 0xff) as i64),
            ],
            Self::Utf8 => vec![Arc::new(utf8_keys(ids))],
        }
    }

    /// The id of the key of row `row` of `columns` (the inverse of
    /// [`Self::columns`]).
    fn id(self, columns: &[ArrayRef], row: usize) -> u64 {
        match self {
            Self::I32 => columns[0].as_primitive::<Int32Type>().value(row) as u64,
            Self::I64 => {
                columns[0].as_primitive::<Int64Type>().value(row) as u64 / KEY_SPREAD
            }
            Self::I64Dense => columns[0].as_primitive::<Int64Type>().value(row) as u64,
            Self::I64x2 => {
                let high = columns[0].as_primitive::<Int64Type>().value(row) as u64;
                let low = columns[1].as_primitive::<Int64Type>().value(row) as u64;
                (high << 8) | low
            }
            Self::Utf8 => columns[0]
                .as_string_view()
                .value(row)
                .parse()
                .expect("a key of decimal digits"),
        }
    }
}

/// `ids` as strings of 16 decimal digits, zero-padded.
fn utf8_keys(ids: &[u64]) -> StringViewArray {
    let mut builder = StringViewBuilder::with_capacity(ids.len());
    let mut digits = [b'0'; 16];
    for &id in ids {
        let mut value = id;
        for digit in digits.iter_mut().rev() {
            *digit = b'0' + (value % 10) as u8;
            value /= 10;
        }
        builder.append_value(std::str::from_utf8(&digits).expect("ASCII digits"));
    }
    builder.finish()
}

/// `rows` uniform `Int64` values in `0..max`.
fn uniform_i64(rng: &mut Rng, rows: usize, max: u64) -> ArrayRef {
    Arc::new(Int64Array::from_iter_values(
        (0..rows).map(|_| rng.below(max) as i64),
    ))
}

/// The orders of the rows of the sweeps over the sortedness of the input.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Order {
    Random,
    Sorted,
    Reverse,
    /// Sorted, with 1% of the rows swapped with random rows.
    Nearly,
}

impl Order {
    fn name(self) -> &'static str {
        match self {
            Self::Random => "random",
            Self::Sorted => "sorted",
            Self::Reverse => "reverse",
            Self::Nearly => "nearly",
        }
    }

    /// Arrange `values` in this order.
    fn arrange<T: Ord>(self, values: &mut [T], rng: &mut Rng) {
        match self {
            Self::Random => rng.shuffle(values),
            Self::Sorted => values.sort_unstable(),
            Self::Reverse => values.sort_unstable_by(|a, b| b.cmp(a)),
            Self::Nearly => {
                values.sort_unstable();
                let len = values.len() as u64;
                for _ in 0..len / 200 {
                    values.swap(rng.below(len) as usize, rng.below(len) as usize);
                }
            }
        }
    }
}

/// The ids of `rows` keys of `distinct` distinct ids (`i % distinct`), in
/// the order `order`.
fn key_ids(rows: usize, distinct: usize, order: Order, rng: &mut Rng) -> Vec<u64> {
    let distinct = distinct.max(1) as u64;
    let mut ids = (0..rows as u64).map(|i| i % distinct).collect::<Vec<_>>();
    order.arrange(&mut ids, rng);
    ids
}

/// `columns` (of `schema`) as chunks of `batch` rows, every chunk with
/// buffers of its own, as the batches of a scan have (an operator that keeps
/// a slice of a larger array accounts for all of its buffers).
fn chunks_of(
    schema: &SchemaRef,
    columns: &[ArrayRef],
    batch: usize,
) -> Result<Vec<DataChunk>> {
    let rows = columns.first().map_or(0, |column| column.len());
    (0..rows)
        .step_by(batch)
        .map(|offset| {
            let len = batch.min(rows - offset);
            DataChunk::try_new(
                Arc::clone(schema),
                columns
                    .iter()
                    .map(|column| owned(&column.slice(offset, len)))
                    .collect(),
                None,
            )
        })
        .collect()
}

/// A copy of `array` (of a type of the synthetic data) in buffers of its
/// own.
fn owned(array: &ArrayRef) -> ArrayRef {
    match array.data_type() {
        DataType::Int32 => Arc::new(Int32Array::from(
            array.as_primitive::<Int32Type>().values().to_vec(),
        )),
        DataType::Int64 => Arc::new(Int64Array::from(
            array.as_primitive::<Int64Type>().values().to_vec(),
        )),
        DataType::Float64 => Arc::new(Float64Array::from(
            array.as_primitive::<Float64Type>().values().to_vec(),
        )),
        DataType::Utf8View => Arc::new(array.as_string_view().gc()),
        other => unreachable!("no synthetic columns of type {other}"),
    }
}

/// `chunks` with selections of about `selectivity` of their rows, as a
/// filter below the operator would leave them.
fn with_selections(chunks: Vec<DataChunk>, selectivity: f64) -> Result<Vec<DataChunk>> {
    let mut rng = Rng::new(0x5e1e_c7ed);
    chunks
        .into_iter()
        .map(|chunk| {
            let selected = (0..chunk.len() as u32)
                .filter(|_| rng.unit() < selectivity)
                .collect::<Vec<_>>();
            chunk.with_selection(SelectionVector::from(selected))
        })
        .collect()
}

/// The bytes of a cache line: an operator that reads some values of a column
/// reads the lines that hold them.
const CACHE_LINE: usize = 64;

/// The bytes of the cache lines that hold the values of `width` bytes of the
/// rows `rows` (in increasing order) of a column, in a buffer of its own.
fn line_bytes(rows: impl IntoIterator<Item = usize>, width: usize) -> usize {
    let mut lines = 0;
    let mut last = None;
    for row in rows {
        let line = row * width / CACHE_LINE;
        if last != Some(line) {
            lines += 1;
            last = Some(line);
        }
    }
    lines * CACHE_LINE
}

/// The bytes of the cache lines that hold the (selected) values of the
/// fixed-width columns `columns` of `chunks`, of `width` bytes each.
fn selected_line_bytes(chunks: &[DataChunk], columns: usize, width: usize) -> f64 {
    let bytes = chunks
        .iter()
        .map(|chunk| match chunk.selection() {
            Some(selection) => line_bytes(selection.iter(), width),
            None => chunk.len() * width,
        })
        .sum::<usize>();
    (bytes * columns) as f64
}

/// The bytes of the cache lines that hold the (selected) values of `chunk`,
/// of the types of the synthetic data (whose strings are of 16 bytes,
/// stored in the order of their rows).
fn read_line_bytes(chunk: &DataChunk) -> f64 {
    let Some(selection) = chunk.selection() else {
        return chunk_bytes(chunk) as f64;
    };
    let bytes = chunk
        .columns()
        .iter()
        .map(|column| match column.data_type() {
            DataType::Utf8View => 2 * line_bytes(selection.iter(), 16),
            data_type => {
                line_bytes(selection.iter(), data_type.primitive_width().unwrap_or(8))
            }
        })
        .sum::<usize>();
    bytes as f64
}

/// The bytes of `columns` per row (see `profiling::array_bytes`).
fn bytes_per_row(columns: &[ArrayRef]) -> f64 {
    let rows = columns.first().map_or(0, |column| column.len());
    let bytes = columns
        .iter()
        .map(|column| {
            datafusion_physical_pipeline::profiling::array_bytes(column.as_ref())
        })
        .sum::<usize>();
    bytes as f64 / rows.max(1) as f64
}

/// The schema of `fields`.
fn schema_of(fields: Vec<Field>) -> SchemaRef {
    Arc::new(Schema::new(fields))
}

/// `rows` in a short form: `4M`, `64K`, `2048`.
fn rows_label(rows: usize) -> String {
    if rows >= 1 << 20 && rows.is_multiple_of(1 << 20) {
        format!("{}M", rows >> 20)
    } else if rows >= 1 << 10 && rows.is_multiple_of(1 << 10) {
        format!("{}K", rows >> 10)
    } else {
        rows.to_string()
    }
}

// ---------------------------------------------------------------------------
// The harness operators
// ---------------------------------------------------------------------------

/// A parallel source that hands out pre-built chunks, in order, through a
/// cursor shared by its threads.
struct ChunkSource {
    schema: SchemaRef,
    chunks: Vec<DataChunk>,
}

impl std::fmt::Debug for ChunkSource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ChunkSource")
            .field("chunks", &self.chunks.len())
            .finish()
    }
}

impl ChunkSource {
    fn new(schema: SchemaRef, chunks: Vec<DataChunk>) -> Arc<Self> {
        Arc::new(Self { schema, chunks })
    }

    /// The logical rows of the chunks.
    fn rows(&self) -> u64 {
        self.chunks.iter().map(|chunk| chunk.len() as u64).sum()
    }
}

/// The global state of a [`ChunkSource`]: the next chunk to hand out.
struct Cursor {
    next: AtomicUsize,
}

impl GlobalSourceState for Cursor {}

impl PhysicalOperator for ChunkSource {
    fn name(&self) -> &str {
        "ChunkSource"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &[]
    }

    fn is_source(&self) -> bool {
        true
    }

    fn parallel_source(&self) -> bool {
        true
    }

    fn get_global_source_state(
        &self,
        _ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalSourceState>> {
        Ok(Arc::new(Cursor {
            next: AtomicUsize::new(0),
        }))
    }

    fn get_data(
        &self,
        _ctx: &ExecutionContext,
        gstate: &dyn GlobalSourceState,
        _lstate: &mut dyn datafusion_physical_pipeline::LocalSourceState,
    ) -> Result<SourceResult> {
        let cursor = downcast_state::<Cursor>(gstate.as_any(), self.name())?;
        let index = cursor.next.fetch_add(1, Ordering::Relaxed);
        Ok(match self.chunks.get(index) {
            Some(chunk) => SourceResult::Chunk(chunk.clone()),
            None => SourceResult::Finished,
        })
    }
}

/// A parallel sink that drops its chunks, or collects them (compacted) for
/// the checks of the warm-up execution.
struct DiscardSink {
    children: Vec<Arc<dyn PhysicalOperator>>,
    collected: Option<Mutex<Vec<RecordBatch>>>,
}

impl std::fmt::Debug for DiscardSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiscardSink")
            .field("collect", &self.collected.is_some())
            .finish()
    }
}

impl DiscardSink {
    fn new(child: Arc<dyn PhysicalOperator>, collect: bool) -> Arc<Self> {
        Arc::new(Self {
            children: vec![child],
            collected: collect.then(|| Mutex::new(vec![])),
        })
    }

    fn take_collected(&self) -> Vec<RecordBatch> {
        self.collected
            .as_ref()
            .map(|collected| std::mem::take(&mut *collected.lock()))
            .unwrap_or_default()
    }
}

/// The local sink state of a [`DiscardSink`] (none).
struct NoState;

impl LocalSinkState for NoState {}

impl PhysicalOperator for DiscardSink {
    fn name(&self) -> &str {
        "DiscardSink"
    }

    fn schema(&self) -> SchemaRef {
        self.children[0].schema()
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.children
    }

    fn is_sink(&self) -> bool {
        true
    }

    fn parallel_sink(&self) -> bool {
        true
    }

    fn get_local_sink_state(
        &self,
        _ctx: &ExecutionContext,
        _gstate: &dyn datafusion_physical_pipeline::GlobalSinkState,
    ) -> Result<Box<dyn LocalSinkState>> {
        Ok(Box::new(NoState))
    }

    fn sink(
        &self,
        _ctx: &ExecutionContext,
        input: &DataChunk,
        _gstate: &dyn datafusion_physical_pipeline::GlobalSinkState,
        lstate: &mut dyn LocalSinkState,
    ) -> Result<SinkResult> {
        downcast_state_mut::<NoState>(lstate.as_any_mut(), self.name())?;
        if let Some(collected) = &self.collected {
            let batch = input.compact()?;
            collected.lock().push(batch);
        }
        Ok(SinkResult::NeedMoreInput)
    }
}

// ---------------------------------------------------------------------------
// Bandwidth
// ---------------------------------------------------------------------------

/// The memory bandwidth of the machine on `threads` threads, in bytes per
/// nanosecond (GB/s): of streaming reads, writes and copies (counting the
/// bytes read and written) of buffers of [`BenchConfig::bandwidth_bytes`].
#[derive(Debug, Clone, Copy)]
struct Bandwidth {
    threads: usize,
    read: f64,
    write: f64,
    copy: f64,
}

/// Measure the [`Bandwidth`] on `threads` threads, with buffers of
/// `total_bytes` (split over the threads), the best of `repetitions`.
fn measure_bandwidth(
    threads: usize,
    total_bytes: usize,
    repetitions: usize,
) -> Bandwidth {
    let words = (total_bytes / 8 / threads).max(1);
    let barrier = Barrier::new(threads + 1);
    let mut best = [Duration::MAX; 3];
    std::thread::scope(|scope| {
        for thread in 0..threads {
            let barrier = &barrier;
            scope.spawn(move || {
                // Every thread touches its own buffers first.
                let source = vec![thread as u64 + 1; words];
                let mut target = vec![0x5a_u64; words];
                for _ in 0..repetitions {
                    barrier.wait();
                    let sum = source
                        .iter()
                        .fold(0_u64, |sum, &value| sum.wrapping_add(value));
                    black_box(sum);
                    barrier.wait();
                    barrier.wait();
                    target.fill(black_box(0x5a5a_5a5a_5a5a_5a5a));
                    black_box(&target);
                    barrier.wait();
                    barrier.wait();
                    target.copy_from_slice(&source);
                    black_box(&target);
                    barrier.wait();
                }
            });
        }
        for _ in 0..repetitions {
            for time in &mut best {
                barrier.wait();
                let start = Instant::now();
                barrier.wait();
                *time = (*time).min(start.elapsed());
            }
        }
    });
    let bytes = (words * 8 * threads) as f64;
    let rate = |bytes: f64, time: Duration| bytes / time.as_nanos().max(1) as f64;
    Bandwidth {
        threads,
        read: rate(bytes, best[0]),
        write: rate(bytes, best[1]),
        copy: rate(2.0 * bytes, best[2]),
    }
}

// ---------------------------------------------------------------------------
// Measurements
// ---------------------------------------------------------------------------

/// The part of the execution of a plan that the measurement of a
/// configuration covers.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Scope {
    /// All pipelines, and the finalization of all meta pipelines whose sink
    /// is not the `DiscardSink`.
    All,
    /// The pipelines whose sink is the operator, and its finalization (the
    /// build of a hash join, with its `JoinHashTableBuild` pipeline).
    Sink,
    /// The stages of the operator in the pipelines in which it is an
    /// operator (the probe of a hash join, a filter, a projection).
    Operator,
}

/// The plan of one execution of a configuration: the `DiscardSink` above the
/// operator, the operator and the harness sources below it.
struct Plan {
    sink: Arc<DiscardSink>,
    operator: Arc<dyn PhysicalOperator>,
    sources: Vec<Arc<dyn PhysicalOperator>>,
}

impl Plan {
    fn new(
        operator: Arc<dyn PhysicalOperator>,
        sources: Vec<Arc<dyn PhysicalOperator>>,
        collect: bool,
    ) -> Self {
        Self {
            sink: DiscardSink::new(Arc::clone(&operator), collect),
            operator,
            sources,
        }
    }
}

/// The bytes that an operator must read and write in an execution, see
/// `docs/operator-performance.md`.
#[derive(Debug, Clone, Copy, Default)]
struct Floor {
    read: f64,
    write: f64,
}

impl Floor {
    fn bytes(self) -> f64 {
        self.read + self.write
    }

    /// The time it takes at least to move the bytes of the floor at
    /// `bandwidth`, in ns: the largest of the times to read them, to write
    /// them, and to move them all at the bandwidth of a copy (which reads
    /// and writes).
    fn time(self, bandwidth: &Bandwidth) -> f64 {
        (self.read / bandwidth.read)
            .max(self.write / bandwidth.write)
            .max(self.bytes() / bandwidth.copy)
    }
}

/// A configuration of a benchmark.
struct Case {
    operator: &'static str,
    config: String,
    threads: usize,
    scope: Scope,
    /// The sinks of the executions whose allocations are those of the
    /// harness, and not of the measured part of the plan.
    baselines: Vec<Arc<dyn PhysicalOperator>>,
    /// The task context of the executions, if not the default one.
    task_ctx: Option<Arc<TaskContext>>,
}

/// One execution of a plan.
struct Run {
    profile: Arc<QueryProfile>,
    allocations: Counts,
}

/// The measurements of one execution of a configuration.
#[derive(Debug, Clone, Copy, Default)]
struct Sample {
    /// The self time of the measured stages and finalizations, in ns.
    self_time: u64,
    /// The wall time of the measured part, in ns: the elapsed time of every
    /// measured pipeline times the share of its active time that the measured
    /// stages account for, plus the finalizations.
    wall_time: f64,
    rows: u64,
    chunks: u64,
    bytes_in: u64,
    bytes_out: u64,
    memory_peak: u64,
    allocations: Counts,
    /// The self time as a sink, finalizing, as a source and as an operator.
    sink_time: u64,
    finalize_time: u64,
    source_time: u64,
}

impl Sample {
    /// The sample of `run` of `plan`, with the allocations of `baseline`
    /// (the harness) subtracted.
    fn of(run: &Run, plan: &Plan, scope: Scope, baseline: Counts) -> Result<Self> {
        let profile = &run.profile;
        let id = |operator: &dyn PhysicalOperator| {
            profile
                .operator(operator)
                .map(|operator| operator.id)
                .ok_or_else(|| {
                    internal_datafusion_err!(
                        "{} is not part of the profile",
                        operator.name()
                    )
                })
        };
        let target = id(plan.operator.as_ref())?;
        let mut harness = vec![id(plan.sink.as_ref())?];
        for source in &plan.sources {
            harness.push(id(source.as_ref())?);
        }
        let mut sample = Self {
            allocations: Counts {
                allocations: run
                    .allocations
                    .allocations
                    .saturating_sub(baseline.allocations),
                bytes: run.allocations.bytes.saturating_sub(baseline.bytes),
            },
            memory_peak: profile.operators()[target].memory_peak,
            ..Self::default()
        };
        let mut rows_out = 0;
        let mut chunks_out = 0;
        for pipeline in profile.pipelines() {
            let measured = match scope {
                Scope::All => true,
                Scope::Sink => pipeline.sink == target,
                Scope::Operator => pipeline.operators.contains(&target),
            };
            if !measured {
                continue;
            }
            let mut self_time = 0;
            for stage in &pipeline.stages {
                if harness.contains(&stage.operator)
                    || (scope == Scope::Operator && stage.operator != target)
                {
                    continue;
                }
                let counters = &stage.counters;
                self_time += counters.self_time;
                match stage.role {
                    OperatorRole::Sink => sample.sink_time += counters.self_time,
                    OperatorRole::Source => sample.source_time += counters.self_time,
                    OperatorRole::Operator => {}
                }
                if stage.operator == target {
                    sample.rows += counters.rows_in;
                    sample.chunks += counters.morsels_in;
                    sample.bytes_in += counters.bytes_in;
                    sample.bytes_out += counters.bytes_out;
                    rows_out += counters.rows_out;
                    chunks_out += counters.morsels_out;
                }
            }
            sample.self_time += self_time;
            if pipeline.active_time > 0 {
                sample.wall_time += pipeline.elapsed_time as f64 * self_time as f64
                    / pipeline.active_time as f64;
            }
        }
        for meta in profile.meta_pipelines() {
            let measured = match scope {
                Scope::All => !harness.contains(&meta.sink),
                Scope::Sink => meta.sink == target,
                Scope::Operator => false,
            };
            if measured {
                sample.self_time += meta.finalize_time;
                sample.finalize_time += meta.finalize_time;
                sample.wall_time += meta.finalize_time as f64;
            }
        }
        // A source (the Parquet decode) has no input rows.
        if sample.rows == 0 {
            sample.rows = rows_out;
            sample.chunks = chunks_out;
        }
        Ok(sample)
    }
}

/// The results of a configuration: the medians of its samples.
#[derive(Debug, Clone)]
struct Measurement {
    operator: &'static str,
    config: String,
    rows: u64,
    ns_per_row: f64,
    rows_per_second: f64,
    allocations_per_chunk: f64,
    copied_per_row: f64,
    in_per_row: f64,
    out_per_row: f64,
    memory_peak: u64,
    floor_per_row: f64,
    /// The share of the floor, `None` without a floor.
    floor_share: Option<f64>,
    /// The coefficient of variation of `ns_per_row`.
    cv: f64,
    /// The shares of the time as a sink, finalizing and as a source, for
    /// pipeline breakers.
    phases: Option<[f64; 3]>,
}

fn median(values: &mut [f64]) -> f64 {
    values.sort_unstable_by(f64::total_cmp);
    let n = values.len();
    if n == 0 {
        0.0
    } else if n % 2 == 1 {
        values[n / 2]
    } else {
        f64::midpoint(values[n / 2 - 1], values[n / 2])
    }
}

fn coefficient_of_variation(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    if values.len() < 2 || mean == 0.0 {
        return 0.0;
    }
    let variance = values
        .iter()
        .map(|value| (value - mean).powi(2))
        .sum::<f64>()
        / (n - 1.0);
    variance.sqrt() / mean
}

impl Measurement {
    fn new(case: &Case, samples: &[Sample], floor: Floor, bandwidth: &Bandwidth) -> Self {
        let per = |f: &dyn Fn(&Sample) -> f64| {
            median(&mut samples.iter().map(f).collect::<Vec<_>>())
        };
        let rows = samples[0].rows;
        let row_count = rows.max(1) as f64;
        let ns_per_row = samples
            .iter()
            .map(|sample| sample.self_time as f64 / sample.rows.max(1) as f64)
            .collect::<Vec<_>>();
        let wall_time = per(&|sample| sample.wall_time);
        let floor_time = floor.time(bandwidth);
        let phases = (case.scope != Scope::Operator).then(|| {
            let total = per(&|sample| sample.self_time as f64).max(1.0);
            [
                per(&|sample| sample.sink_time as f64) / total,
                per(&|sample| sample.finalize_time as f64) / total,
                per(&|sample| sample.source_time as f64) / total,
            ]
        });
        Self {
            operator: case.operator,
            config: case.config.clone(),
            rows,
            ns_per_row: median(&mut ns_per_row.clone()),
            rows_per_second: row_count / wall_time.max(1.0) * 1e9,
            allocations_per_chunk: per(&|sample| sample.allocations.allocations as f64)
                / samples[0].chunks.max(1) as f64,
            copied_per_row: per(&|sample| sample.allocations.bytes as f64) / row_count,
            in_per_row: per(&|sample| sample.bytes_in as f64) / row_count,
            out_per_row: per(&|sample| sample.bytes_out as f64) / row_count,
            memory_peak: per(&|sample| sample.memory_peak as f64) as u64,
            floor_per_row: floor.bytes() / row_count,
            floor_share: (floor.bytes() > 0.0).then(|| floor_time / wall_time.max(1.0)),
            cv: coefficient_of_variation(&ns_per_row),
            phases,
        }
    }

    fn name(&self) -> String {
        format!("{} {}", self.operator, self.config)
    }
}

/// The benchmark: its configuration, runtime, bandwidths and results.
struct Bench {
    config: BenchConfig,
    runtime: tokio::runtime::Runtime,
    task_ctx: Arc<TaskContext>,
    bandwidths: Vec<Bandwidth>,
    measurements: Vec<Measurement>,
}

impl Bench {
    fn try_new(config: BenchConfig) -> Result<Self> {
        let max_threads = config.threads.iter().copied().max().unwrap_or(1);
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(max_threads)
            .enable_all()
            .build()?;
        let session = SessionConfig::new().with_target_partitions(max_threads);
        let task_ctx = SessionContext::new_with_config(session).task_ctx();
        Ok(Self {
            config,
            runtime,
            task_ctx,
            bandwidths: vec![],
            measurements: vec![],
        })
    }

    fn measure_bandwidths(&mut self) {
        let mut threads = self.config.threads.clone();
        threads.sort_unstable();
        threads.dedup();
        self.bandwidths = threads
            .into_iter()
            .map(|threads| measure_bandwidth(threads, self.config.bandwidth_bytes, 5))
            .collect();
    }

    fn bandwidth(&self, threads: usize) -> Bandwidth {
        self.bandwidths
            .iter()
            .find(|bandwidth| bandwidth.threads == threads)
            .copied()
            .unwrap_or_else(|| measure_bandwidth(threads, self.config.bandwidth_bytes, 3))
    }

    /// `rows` scaled by [`BenchConfig::scale`], a multiple of
    /// [`DEFAULT_VECTOR_SIZE`].
    fn rows(&self, rows: usize) -> usize {
        let scaled = (rows as f64 * self.config.scale) as usize;
        (scaled / DEFAULT_VECTOR_SIZE).max(1) * DEFAULT_VECTOR_SIZE
    }

    /// The thread counts of the partition sweeps other than 1 (the thread
    /// count of all other configurations).
    fn thread_sweep(&self) -> Vec<usize> {
        self.config
            .threads
            .iter()
            .copied()
            .filter(|&threads| threads != 1)
            .collect()
    }

    /// The configurations of `params` that the filter selects, without
    /// duplicates.
    fn select<P: PartialEq>(
        &self,
        operator: &str,
        params: Vec<P>,
        label: impl Fn(&P) -> String,
    ) -> Vec<P> {
        let mut selected: Vec<P> = vec![];
        for param in params {
            let name = format!("{operator} {} ", label(&param));
            let wanted = self.config.filter.is_empty()
                || self
                    .config
                    .filter
                    .iter()
                    .any(|filter| name.contains(filter.as_str()));
            if wanted && !selected.contains(&param) {
                selected.push(param);
            }
        }
        selected
    }

    fn execute(
        &self,
        sink: Arc<dyn PhysicalOperator>,
        threads: usize,
        task_ctx: Option<&Arc<TaskContext>>,
    ) -> Result<Run> {
        let task_ctx = Arc::clone(task_ctx.unwrap_or(&self.task_ctx));
        let executor = Executor::try_new(sink, task_ctx, threads)?.with_profiling(true);
        let before = allocations::snapshot();
        self.runtime.block_on(executor.execute())?;
        let allocations = allocations::snapshot().since(before);
        let profile = executor
            .profile()
            .ok_or_else(|| internal_datafusion_err!("The execution has no profile"))?;
        Ok(Run {
            profile,
            allocations,
        })
    }

    /// Measure `case`: a warm-up execution of `plan` (with `collect`) whose
    /// output `check` checks and whose output rows give the `floor`, the
    /// executions of the baselines, and the timed executions.
    fn measure(
        &mut self,
        case: &Case,
        plan: impl Fn(bool) -> Result<Plan>,
        floor: impl Fn(u64) -> Floor,
        check: impl Fn(&[RecordBatch]) -> Result<()>,
    ) -> Result<()> {
        let context =
            |e: DataFusionError| e.context(format!("{} {}", case.operator, case.config));
        let task_ctx = case.task_ctx.as_ref();
        let warm_up = plan(true)?;
        self.execute(Arc::clone(&warm_up.sink) as _, case.threads, task_ctx)
            .map_err(context)?;
        let output = warm_up.sink.take_collected();
        check(&output).map_err(context)?;
        let floor = floor(output.iter().map(|batch| batch.num_rows() as u64).sum());
        drop(output);
        let mut baseline = Counts::default();
        for sink in &case.baselines {
            self.execute(Arc::clone(sink), case.threads, task_ctx)?;
            let run = self.execute(Arc::clone(sink), case.threads, task_ctx)?;
            baseline = baseline.add(run.allocations);
        }
        let mut samples = vec![];
        for _ in 0..self.config.iterations {
            let plan = plan(false)?;
            let run = self
                .execute(Arc::clone(&plan.sink) as _, case.threads, task_ctx)
                .map_err(context)?;
            samples.push(Sample::of(&run, &plan, case.scope, baseline)?);
        }
        let measurement =
            Measurement::new(case, &samples, floor, &self.bandwidth(case.threads));
        eprintln!(
            "{:<18} {:<50} {:>8.2} ns/row",
            measurement.operator, measurement.config, measurement.ns_per_row
        );
        self.measurements.push(measurement);
        Ok(())
    }

    /// Record the samples of a configuration measured without an executor
    /// (tuple-data ingestion).
    fn record(&mut self, case: &Case, samples: &[Sample], floor: Floor) {
        let measurement =
            Measurement::new(case, samples, floor, &self.bandwidth(case.threads));
        eprintln!(
            "{:<18} {:<50} {:>8.2} ns/row",
            measurement.operator, measurement.config, measurement.ns_per_row
        );
        self.measurements.push(measurement);
    }
}

/// Check that `output` has `expected` rows.
fn expect_rows(output: &[RecordBatch], expected: u64) -> Result<()> {
    let rows = output
        .iter()
        .map(|batch| batch.num_rows() as u64)
        .sum::<u64>();
    if rows != expected {
        return internal_err!("expected {expected} output rows, got {rows}");
    }
    Ok(())
}

/// The columns `names` of `batch`.
fn columns_of(batch: &RecordBatch, names: &[String]) -> Result<Vec<ArrayRef>> {
    names
        .iter()
        .map(|name| {
            batch
                .column_by_name(name)
                .map(Arc::clone)
                .ok_or_else(|| internal_datafusion_err!("no column {name}"))
        })
        .collect()
}

/// The ids of the keys `names` (of kind `key`) of all rows of `output`.
fn output_ids(
    output: &[RecordBatch],
    key: KeyKind,
    names: &[String],
) -> Result<Vec<u64>> {
    let mut ids = vec![];
    for batch in output {
        let columns = columns_of(batch, names)?;
        ids.extend((0..batch.num_rows()).map(|row| key.id(&columns, row)));
    }
    Ok(ids)
}

// ---------------------------------------------------------------------------
// Projection
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Projection {
    /// `c, a`: columns only.
    Columns,
    /// `a + b, c`.
    Sum,
    /// `a + b, a - b, a * b`.
    Arithmetic,
}

impl Projection {
    fn name(self) -> &'static str {
        match self {
            Self::Columns => "c,a",
            Self::Sum => "a+b,c",
            Self::Arithmetic => "a+b,a-b,a*b",
        }
    }

    /// The number of computed columns.
    fn computed(self) -> u64 {
        match self {
            Self::Columns => 0,
            Self::Sum => 1,
            Self::Arithmetic => 3,
        }
    }

    fn exprs(self, schema: &SchemaRef) -> Result<Vec<(Arc<dyn PhysicalExpr>, String)>> {
        let column = |name: &str| col(name, schema);
        let op = |op: Operator| binary(column("a")?, op, column("b")?, schema);
        Ok(match self {
            Self::Columns => vec![(column("c")?, "c".into()), (column("a")?, "a".into())],
            Self::Sum => vec![
                (op(Operator::Plus)?, "s".into()),
                (column("c")?, "c".into()),
            ],
            Self::Arithmetic => vec![
                (op(Operator::Plus)?, "sum".into()),
                (op(Operator::Minus)?, "difference".into()),
                (op(Operator::Multiply)?, "product".into()),
            ],
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ProjectionParams {
    exprs: Projection,
    selectivity: f64,
    batch: usize,
    threads: usize,
}

impl ProjectionParams {
    fn label(&self) -> String {
        format!(
            "exprs={} sel={:.2} batch={} threads={}",
            self.exprs.name(),
            self.selectivity,
            self.batch,
            self.threads
        )
    }
}

/// `SELECT <exprs> FROM t` over 4Mi rows of three `Int64` columns, on chunks
/// with and without selections.
fn projection(bench: &mut Bench) -> Result<()> {
    let default = ProjectionParams {
        exprs: Projection::Sum,
        selectivity: 1.0,
        batch: DEFAULT_VECTOR_SIZE,
        threads: 1,
    };
    let mut params = vec![default];
    params.extend(
        [Projection::Columns, Projection::Arithmetic]
            .map(|exprs| ProjectionParams { exprs, ..default }),
    );
    params.extend([0.5, 0.1].map(|selectivity| ProjectionParams {
        selectivity,
        ..default
    }));
    params.extend([256, 1024].map(|batch| ProjectionParams { batch, ..default }));
    params.extend(
        bench
            .thread_sweep()
            .into_iter()
            .map(|threads| ProjectionParams { threads, ..default }),
    );
    let params = bench.select("projection", params, ProjectionParams::label);
    if params.is_empty() {
        return Ok(());
    }
    let rows = bench.rows(4 << 20);
    let mut rng = Rng::new(1);
    let schema = schema_of(
        ["a", "b", "c"]
            .map(|name| Field::new(name, DataType::Int64, false))
            .to_vec(),
    );
    let columns = (0..3)
        .map(|_| uniform_i64(&mut rng, rows, 1_000_000))
        .collect::<Vec<_>>();
    for params in params {
        let mut chunks = chunks_of(&schema, &columns, params.batch)?;
        if params.selectivity < 1.0 {
            chunks = with_selections(chunks, params.selectivity)?;
        }
        let source = ChunkSource::new(Arc::clone(&schema), chunks);
        let logical_rows = source.rows();
        let exprs = params.exprs.exprs(&schema)?;
        let case = Case {
            operator: "projection",
            config: params.label(),
            threads: params.threads,
            scope: Scope::Operator,
            baselines: vec![DiscardSink::new(Arc::clone(&source) as _, false)],
            task_ctx: None,
        };
        let computed = params.exprs.computed();
        let read_bytes = selected_line_bytes(&source.chunks, 2, 8);
        bench.measure(
            &case,
            |collect| {
                let projection =
                    PhysicalProjection::try_new(Arc::clone(&source) as _, exprs.clone())?;
                Ok(Plan::new(
                    Arc::new(projection),
                    vec![Arc::clone(&source) as _],
                    collect,
                ))
            },
            // The computed columns read (the lines of the selected values
            // of) `a` and `b` and write a column each; the columns are passed
            // through.
            |_| Floor {
                read: if computed > 0 { read_bytes } else { 0.0 },
                write: 8.0 * (computed * logical_rows) as f64,
            },
            |output| expect_rows(output, logical_rows),
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Filter
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Predicate {
    /// `a < x`.
    I64,
    /// `a < x AND b < x`, each of the square root of the selectivity.
    Conjunction,
    /// `s < x`, `s` the 16 decimal digits of `a`.
    Utf8,
}

impl Predicate {
    fn name(self) -> &'static str {
        match self {
            Self::I64 => "a<x",
            Self::Conjunction => "a<x&b<y",
            Self::Utf8 => "utf8<x",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct FilterParams {
    predicate: Predicate,
    selectivity: f64,
    batch: usize,
    threads: usize,
}

impl FilterParams {
    fn label(&self) -> String {
        format!(
            "pred={} sel={} batch={} threads={}",
            self.predicate.name(),
            self.selectivity,
            self.batch,
            self.threads
        )
    }
}

/// The values of the filtered columns are uniform in `0..FILTER_VALUES`.
const FILTER_VALUES: u64 = 1 << 20;

/// `SELECT * FROM t WHERE <predicate>` over 4Mi rows of `a`, `b`, `c`
/// (`Int64`) and `s` (`Utf8View`).
fn filter(bench: &mut Bench) -> Result<()> {
    let default = FilterParams {
        predicate: Predicate::I64,
        selectivity: 0.5,
        batch: DEFAULT_VECTOR_SIZE,
        threads: 1,
    };
    let mut params = vec![default];
    params.extend(
        [0.001, 0.01, 0.1, 0.9, 1.0].map(|selectivity| FilterParams {
            selectivity,
            ..default
        }),
    );
    params.extend([Predicate::Conjunction, Predicate::Utf8].map(|predicate| {
        FilterParams {
            predicate,
            ..default
        }
    }));
    params.extend([256, 1024].map(|batch| FilterParams { batch, ..default }));
    params.extend(
        bench
            .thread_sweep()
            .into_iter()
            .map(|threads| FilterParams { threads, ..default }),
    );
    let params = bench.select("filter", params, FilterParams::label);
    if params.is_empty() {
        return Ok(());
    }
    let rows = bench.rows(4 << 20);
    let mut rng = Rng::new(2);
    let a = (0..rows)
        .map(|_| rng.below(FILTER_VALUES))
        .collect::<Vec<_>>();
    let b = (0..rows)
        .map(|_| rng.below(FILTER_VALUES))
        .collect::<Vec<_>>();
    let schema = schema_of(vec![
        Field::new("a", DataType::Int64, false),
        Field::new("b", DataType::Int64, false),
        Field::new("c", DataType::Int64, false),
        Field::new("s", DataType::Utf8View, false),
    ]);
    let s: ArrayRef = Arc::new(utf8_keys(&a));
    let s_bytes = bytes_per_row(&[Arc::clone(&s)]);
    let columns = vec![
        KeyKind::I64Dense.columns(&a).remove(0),
        KeyKind::I64Dense.columns(&b).remove(0),
        uniform_i64(&mut rng, rows, 1_000_000),
        s,
    ];
    for params in params {
        let source = ChunkSource::new(
            Arc::clone(&schema),
            chunks_of(&schema, &columns, params.batch)?,
        );
        let threshold = |selectivity: f64| (selectivity * FILTER_VALUES as f64) as u64;
        let (predicate, passing_a, selected) = match params.predicate {
            Predicate::I64 | Predicate::Utf8 => {
                let x = threshold(params.selectivity);
                let selected = a.iter().filter(|&&a| a < x).count() as u64;
                let predicate = if params.predicate == Predicate::I64 {
                    binary(col("a", &schema)?, Operator::Lt, lit(x as i64), &schema)?
                } else {
                    let x = utf8_keys(&[x]).value(0).to_owned();
                    binary(
                        col("s", &schema)?,
                        Operator::Lt,
                        lit(datafusion_common::ScalarValue::Utf8View(Some(x))),
                        &schema,
                    )?
                };
                (predicate, 0.0, selected)
            }
            Predicate::Conjunction => {
                let x = threshold(params.selectivity.sqrt());
                // The second conjunct reads the lines of `b` of the rows that
                // pass the first one, in every chunk.
                let passing_a = a
                    .chunks(params.batch)
                    .map(|chunk| {
                        let rows = chunk.iter().enumerate().filter(|&(_, &a)| a < x);
                        line_bytes(rows.map(|(row, _)| row), 8)
                    })
                    .sum::<usize>() as f64;
                let selected =
                    a.iter().zip(&b).filter(|&(&a, &b)| a < x && b < x).count() as u64;
                let conjunct = |name: &str| {
                    binary(col(name, &schema)?, Operator::Lt, lit(x as i64), &schema)
                };
                let predicate =
                    binary(conjunct("a")?, Operator::And, conjunct("b")?, &schema)?;
                (predicate, passing_a, selected)
            }
        };
        let case = Case {
            operator: "filter",
            config: params.label(),
            threads: params.threads,
            scope: Scope::Operator,
            baselines: vec![DiscardSink::new(Arc::clone(&source) as _, false)],
            task_ctx: None,
        };
        let rows = rows as f64;
        bench.measure(
            &case,
            |collect| {
                let filter = PhysicalFilter::try_new(
                    Arc::clone(&source) as _,
                    Arc::clone(&predicate),
                    None,
                )?;
                Ok(Plan::new(
                    Arc::new(filter),
                    vec![Arc::clone(&source) as _],
                    collect,
                ))
            },
            // The filter reads the column of its first conjunct, the column
            // of the second one of the rows that pass the first one, and
            // writes the selection vector (4 bytes per selected row).
            |output_rows| Floor {
                read: match params.predicate {
                    Predicate::I64 => 8.0 * rows,
                    Predicate::Conjunction => 8.0 * rows + passing_a,
                    Predicate::Utf8 => s_bytes * rows,
                },
                write: 4.0 * output_rows as f64,
            },
            |output| expect_rows(output, selected),
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Hash join
// ---------------------------------------------------------------------------

/// The build side of a join: `rows` rows of `rows / duplicates` distinct
/// keys (ids `2 * i`), with an `Int64` payload, in a random order.
struct BuildSide {
    source: Arc<ChunkSource>,
    distinct: u64,
    duplicates: u64,
    /// The bytes per row of the keys and the payload.
    row_bytes: f64,
}

impl BuildSide {
    fn new(rows: usize, duplicates: usize, key: KeyKind, batch: usize) -> Result<Self> {
        let mut rng = Rng::new(3);
        let distinct = (rows / duplicates).max(1);
        let ids = key_ids(rows, distinct, Order::Random, &mut rng)
            .into_iter()
            .map(|id| 2 * id)
            .collect::<Vec<_>>();
        let mut fields = key.fields("bk");
        fields.push(Field::new("bp", DataType::Int64, false));
        let schema = schema_of(fields);
        let mut columns = key.columns(&ids);
        columns.push(uniform_i64(&mut rng, rows, 1_000_000));
        let row_bytes = bytes_per_row(&columns);
        Ok(Self {
            source: ChunkSource::new(
                Arc::clone(&schema),
                chunks_of(&schema, &columns, batch)?,
            ),
            distinct: distinct as u64,
            duplicates: duplicates as u64,
            row_bytes,
        })
    }
}

/// The probe side of a join: `rows` rows whose keys find a match (ids
/// `2 * i` of the build side) with probability `hit`, and miss (ids
/// `2 * i + 1`, in the range of the build keys) otherwise, with an `Int64`
/// payload.
struct ProbeSide {
    source: Arc<ChunkSource>,
    hits: u64,
    key_bytes: f64,
    row_bytes: f64,
}

impl ProbeSide {
    fn new(
        rows: usize,
        hit: f64,
        build: &BuildSide,
        key: KeyKind,
        batch: usize,
    ) -> Result<Self> {
        let mut rng = Rng::new(4);
        let mut hits = 0;
        let ids = (0..rows)
            .map(|_| {
                let id = 2 * rng.below(build.distinct);
                if rng.unit() < hit {
                    hits += 1;
                    id
                } else {
                    id + 1
                }
            })
            .collect::<Vec<_>>();
        let mut fields = key.fields("pk");
        fields.push(Field::new("pq", DataType::Int64, false));
        let schema = schema_of(fields);
        let mut columns = key.columns(&ids);
        let key_bytes = bytes_per_row(&columns);
        columns.push(uniform_i64(&mut rng, rows, 1_000_000));
        let row_bytes = bytes_per_row(&columns);
        Ok(Self {
            source: ChunkSource::new(
                Arc::clone(&schema),
                chunks_of(&schema, &columns, batch)?,
            ),
            hits,
            key_bytes,
            row_bytes,
        })
    }
}

/// The inner join of `build` and `probe` on their keys (of kind `key`).
fn hash_join(
    build: &Arc<ChunkSource>,
    probe: &Arc<ChunkSource>,
    key: KeyKind,
) -> Result<Arc<dyn PhysicalOperator>> {
    let build_schema = build.schema();
    let probe_schema = probe.schema();
    let on = key
        .names("bk")
        .iter()
        .zip(key.names("pk"))
        .map(|(build_key, probe_key)| {
            Ok((
                col(build_key, &build_schema)?,
                col(&probe_key, &probe_schema)?,
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(Arc::new(PhysicalHashJoin::try_new(
        Arc::clone(build) as _,
        Arc::clone(probe) as _,
        on,
        None,
        JoinType::Inner,
        None,
        NullEquality::NullEqualsNothing,
        false,
    )?))
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct BuildParams {
    rows: usize,
    duplicates: usize,
    key: KeyKind,
    threads: usize,
}

impl BuildParams {
    fn label(&self) -> String {
        format!(
            "build={} dups={} key={} threads={}",
            rows_label(self.rows),
            self.duplicates,
            self.key.name(),
            self.threads
        )
    }
}

/// The build of `SELECT * FROM build JOIN probe ON bk = pk`: `rows` build
/// rows (1Mi by default) of `rows / duplicates` distinct keys, probed by one
/// chunk.
fn hash_join_build(bench: &mut Bench) -> Result<()> {
    let default = BuildParams {
        rows: bench.rows(1 << 20),
        duplicates: 1,
        key: KeyKind::I64,
        threads: 1,
    };
    let mut params = vec![default];
    params.extend(
        [bench.rows(64 << 10), bench.rows(4 << 20)]
            .map(|rows| BuildParams { rows, ..default }),
    );
    params.extend([4, 64].map(|duplicates| BuildParams {
        duplicates,
        ..default
    }));
    params.extend(
        [
            KeyKind::I32,
            KeyKind::I64Dense,
            KeyKind::I64x2,
            KeyKind::Utf8,
        ]
        .map(|key| BuildParams { key, ..default }),
    );
    params.extend(
        bench
            .thread_sweep()
            .into_iter()
            .map(|threads| BuildParams { threads, ..default }),
    );
    let params = bench.select("hash_join_build", params, BuildParams::label);
    for params in params {
        let build = BuildSide::new(
            params.rows,
            params.duplicates,
            params.key,
            DEFAULT_VECTOR_SIZE,
        )?;
        let probe = ProbeSide::new(
            DEFAULT_VECTOR_SIZE,
            1.0,
            &build,
            params.key,
            DEFAULT_VECTOR_SIZE,
        )?;
        let case = Case {
            operator: "hash_join_build",
            config: params.label(),
            threads: params.threads,
            scope: Scope::Sink,
            baselines: vec![
                DiscardSink::new(Arc::clone(&build.source) as _, false),
                DiscardSink::new(Arc::clone(&probe.source) as _, false),
            ],
            task_ctx: None,
        };
        let expected = probe.hits * build.duplicates;
        let build_bytes = build.row_bytes * params.rows as f64;
        bench.measure(
            &case,
            |collect| {
                let join = hash_join(&build.source, &probe.source, params.key)?;
                Ok(Plan::new(
                    join,
                    vec![
                        Arc::clone(&build.source) as _,
                        Arc::clone(&probe.source) as _,
                    ],
                    collect,
                ))
            },
            // The build reads its rows and stores them once.
            |_| Floor {
                read: build_bytes,
                write: build_bytes,
            },
            |output| expect_rows(output, expected),
        )?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ProbeParams {
    hit: f64,
    build: usize,
    duplicates: usize,
    key: KeyKind,
    batch: usize,
    threads: usize,
}

impl ProbeParams {
    fn label(&self) -> String {
        format!(
            "hit={:.2} build={} dups={} key={} batch={} threads={}",
            self.hit,
            rows_label(self.build),
            self.duplicates,
            self.key.name(),
            self.batch,
            self.threads
        )
    }
}

/// The probe of `SELECT * FROM build JOIN probe ON bk = pk`: 4Mi probe rows
/// against `build` build rows (1Mi by default), of which a share `hit` find
/// `duplicates` matches each.
fn hash_join_probe(bench: &mut Bench) -> Result<()> {
    let default = ProbeParams {
        hit: 0.5,
        build: bench.rows(1 << 20),
        duplicates: 1,
        key: KeyKind::I64,
        batch: DEFAULT_VECTOR_SIZE,
        threads: 1,
    };
    let mut params = vec![default];
    params.extend([0.0, 0.1, 0.9, 1.0].map(|hit| ProbeParams { hit, ..default }));
    params.extend(
        [
            bench.rows(1 << 10),
            bench.rows(64 << 10),
            bench.rows(4 << 20),
        ]
        .map(|build| ProbeParams { build, ..default }),
    );
    params.push(ProbeParams {
        duplicates: 4,
        ..default
    });
    params.extend(
        [
            KeyKind::I32,
            KeyKind::I64Dense,
            KeyKind::I64x2,
            KeyKind::Utf8,
        ]
        .map(|key| ProbeParams { key, ..default }),
    );
    params.extend([256, 1024].map(|batch| ProbeParams { batch, ..default }));
    params.extend(
        bench
            .thread_sweep()
            .into_iter()
            .map(|threads| ProbeParams { threads, ..default }),
    );
    let params = bench.select("hash_join_probe", params, ProbeParams::label);
    let probe_rows = bench.rows(4 << 20);
    for params in params {
        let build = BuildSide::new(
            params.build,
            params.duplicates,
            params.key,
            DEFAULT_VECTOR_SIZE,
        )?;
        let probe =
            ProbeSide::new(probe_rows, params.hit, &build, params.key, params.batch)?;
        let empty = ChunkSource::new(probe.source.schema(), vec![]);
        let case = Case {
            operator: "hash_join_probe",
            config: params.label(),
            threads: params.threads,
            scope: Scope::Operator,
            // The build, and the probe rows.
            baselines: vec![
                DiscardSink::new(hash_join(&build.source, &empty, params.key)?, false),
                DiscardSink::new(Arc::clone(&probe.source) as _, false),
            ],
            task_ctx: None,
        };
        let expected = probe.hits * build.duplicates;
        bench.measure(
            &case,
            |collect| {
                let join = hash_join(&build.source, &probe.source, params.key)?;
                Ok(Plan::new(
                    join,
                    vec![
                        Arc::clone(&build.source) as _,
                        Arc::clone(&probe.source) as _,
                    ],
                    collect,
                ))
            },
            // The probe reads the keys of all probe rows, and for every
            // output row the build and the probe row, and writes the output
            // row.
            |output_rows| {
                let output_rows = output_rows as f64;
                let row = build.row_bytes + probe.row_bytes;
                Floor {
                    read: probe.key_bytes * probe_rows as f64
                        + output_rows * (row - probe.key_bytes),
                    write: output_rows * row,
                }
            },
            |output| expect_rows(output, expected),
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Aggregation
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq)]
struct AggregateParams {
    groups: usize,
    key: KeyKind,
    batch: usize,
    threads: usize,
}

impl AggregateParams {
    fn label(&self) -> String {
        format!(
            "groups={} key={} batch={} threads={}",
            rows_label(self.groups),
            self.key.name(),
            self.batch,
            self.threads
        )
    }
}

/// `SELECT k, SUM(v), COUNT(*) FROM t GROUP BY k` (in two stages, as
/// DataFusion plans it on several partitions) over the chunks of `source`.
fn aggregation(
    source: &Arc<ChunkSource>,
    key: KeyKind,
) -> Result<Arc<dyn PhysicalOperator>> {
    let schema = source.schema();
    let group_by = PhysicalGroupBy::new_single(
        key.names("k")
            .into_iter()
            .map(|name| Ok((col(&name, &schema)?, name)))
            .collect::<Result<_>>()?,
    );
    let sum = AggregateExprBuilder::new(sum_udaf(), vec![col("v", &schema)?])
        .schema(Arc::clone(&schema))
        .alias("sum(v)")
        .build()?;
    let count = AggregateExprBuilder::new(count_udaf(), vec![lit(1_i64)])
        .schema(Arc::clone(&schema))
        .alias("count(*)")
        .build()?;
    let aggregates = vec![Arc::new(sum), Arc::new(count)];
    // The input of the `AggregateExec`s only carries the schema.
    let input = MemorySourceConfig::try_new_exec(&[vec![]], Arc::clone(&schema), None)?;
    let partial = AggregateExec::try_new(
        AggregateMode::Partial,
        group_by.clone(),
        aggregates.clone(),
        vec![None, None],
        input,
        Arc::clone(&schema),
    )?;
    let partial = Arc::new(partial);
    let last = AggregateExec::try_new(
        AggregateMode::Final,
        group_by.as_final(),
        aggregates,
        vec![None, None],
        Arc::clone(&partial) as _,
        schema,
    )?;
    Ok(Arc::new(PhysicalHashAggregate::try_new_two_stage(
        &partial,
        &last,
        Arc::clone(source) as _,
    )?))
}

/// `SELECT k, SUM(v), COUNT(*) FROM t GROUP BY k` over 4Mi rows of `groups`
/// groups (1Ki by default), every group of the same number of rows, in a
/// random order.
fn aggregate(bench: &mut Bench) -> Result<()> {
    let rows = bench.rows(4 << 20);
    let default = AggregateParams {
        groups: 1 << 10,
        key: KeyKind::I64,
        batch: DEFAULT_VECTOR_SIZE,
        threads: 1,
    };
    let mut params = vec![default];
    params.extend(
        [16, 64 << 10, bench.rows(1 << 20), rows]
            .map(|groups| AggregateParams { groups, ..default }),
    );
    params.extend(
        [KeyKind::I32, KeyKind::I64x2, KeyKind::Utf8]
            .map(|key| AggregateParams { key, ..default }),
    );
    params.extend([256, 1024].map(|batch| AggregateParams { batch, ..default }));
    params.extend(
        bench
            .thread_sweep()
            .into_iter()
            .map(|threads| AggregateParams { threads, ..default }),
    );
    let params = bench.select("aggregate", params, AggregateParams::label);
    for params in params {
        let mut rng = Rng::new(5);
        let ids = key_ids(rows, params.groups, Order::Random, &mut rng);
        let mut fields = params.key.fields("k");
        fields.push(Field::new("v", DataType::Int64, false));
        let schema = schema_of(fields);
        let mut columns = params.key.columns(&ids);
        let key_bytes = bytes_per_row(&columns);
        columns.push(uniform_i64(&mut rng, rows, 1_000_000));
        let input_bytes = bytes_per_row(&columns) * rows as f64;
        let source = ChunkSource::new(
            Arc::clone(&schema),
            chunks_of(&schema, &columns, params.batch)?,
        );
        let case = Case {
            operator: "aggregate",
            config: params.label(),
            threads: params.threads,
            scope: Scope::All,
            baselines: vec![DiscardSink::new(Arc::clone(&source) as _, false)],
            task_ctx: None,
        };
        let groups = params.groups.min(rows) as u64;
        bench.measure(
            &case,
            |collect| {
                Ok(Plan::new(
                    aggregation(&source, params.key)?,
                    vec![Arc::clone(&source) as _],
                    collect,
                ))
            },
            // The aggregation reads its rows, and writes every group (its
            // keys, its sum and its count) once.
            |output_rows| Floor {
                read: input_bytes,
                write: output_rows as f64 * (key_bytes + 16.0),
            },
            |output| {
                expect_rows(output, groups)?;
                let count = output
                    .iter()
                    .map(|batch| {
                        let counts = batch.column(batch.num_columns() - 1);
                        arrow::compute::sum(counts.as_primitive::<Int64Type>())
                            .unwrap_or(0)
                    })
                    .sum::<i64>();
                if count != rows as i64 {
                    return internal_err!(
                        "the groups count {count} rows, expected {rows}"
                    );
                }
                Ok(())
            },
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Sort and top-K
// ---------------------------------------------------------------------------

/// A table of `rows` rows of keys (of kind `key`, `distinct` distinct ones
/// in the order `order`) and an `Int64` payload `p`.
fn sort_input(
    rows: usize,
    distinct: usize,
    order: Order,
    key: KeyKind,
    seed: u64,
) -> (SchemaRef, Vec<ArrayRef>, f64) {
    let mut rng = Rng::new(seed);
    let ids = key_ids(rows, distinct, order, &mut rng);
    let mut fields = key.fields("k");
    fields.push(Field::new("p", DataType::Int64, false));
    let mut columns = key.columns(&ids);
    let key_bytes = bytes_per_row(&columns);
    columns.push(uniform_i64(&mut rng, rows, 1_000_000));
    (schema_of(fields), columns, key_bytes)
}

/// The ascending ordering of the keys of kind `key`.
fn key_ordering(schema: &SchemaRef, key: KeyKind) -> Result<LexOrdering> {
    let exprs = key
        .names("k")
        .iter()
        .map(|name| {
            Ok(PhysicalSortExpr::new(
                col(name, schema)?,
                SortOptions::default(),
            ))
        })
        .collect::<Result<Vec<_>>>()?;
    LexOrdering::new(exprs).ok_or_else(|| internal_datafusion_err!("no sort keys"))
}

/// Check that the ids of the keys of `output` are sorted.
fn expect_sorted(output: &[RecordBatch], key: KeyKind) -> Result<Vec<u64>> {
    let ids = output_ids(output, key, &key.names("k"))?;
    if !ids.is_sorted() {
        return internal_err!("the output is not sorted");
    }
    Ok(ids)
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct SortParams {
    order: Order,
    key: KeyKind,
    /// The distinct keys, 0 for all rows.
    distinct: usize,
    batch: usize,
    threads: usize,
}

impl SortParams {
    fn label(&self, rows: usize) -> String {
        let distinct = if self.distinct == 0 {
            rows
        } else {
            self.distinct
        };
        format!(
            "order={} key={} distinct={} batch={} threads={}",
            self.order.name(),
            self.key.name(),
            rows_label(distinct),
            self.batch,
            self.threads
        )
    }
}

/// `SELECT k, p FROM t ORDER BY k` over 2Mi rows.
fn sort(bench: &mut Bench) -> Result<()> {
    let rows = bench.rows(2 << 20);
    let default = SortParams {
        order: Order::Random,
        key: KeyKind::I64,
        distinct: 0,
        batch: DEFAULT_VECTOR_SIZE,
        threads: 1,
    };
    let mut params = vec![default];
    params.extend(
        [Order::Sorted, Order::Reverse, Order::Nearly]
            .map(|order| SortParams { order, ..default }),
    );
    params.extend(
        [KeyKind::I32, KeyKind::I64x2, KeyKind::Utf8]
            .map(|key| SortParams { key, ..default }),
    );
    params.extend([16, 64 << 10].map(|distinct| SortParams {
        distinct,
        ..default
    }));
    params.push(SortParams {
        batch: 256,
        ..default
    });
    params.extend(
        bench
            .thread_sweep()
            .into_iter()
            .map(|threads| SortParams { threads, ..default }),
    );
    let params = bench.select("sort", params, |params| params.label(rows));
    for params in params {
        let distinct = if params.distinct == 0 {
            rows
        } else {
            params.distinct
        };
        let (schema, columns, _) =
            sort_input(rows, distinct, params.order, params.key, 6);
        let input_bytes = bytes_per_row(&columns) * rows as f64;
        let source = ChunkSource::new(
            Arc::clone(&schema),
            chunks_of(&schema, &columns, params.batch)?,
        );
        let ordering = key_ordering(&schema, params.key)?;
        let case = Case {
            operator: "sort",
            config: params.label(rows),
            threads: params.threads,
            scope: Scope::All,
            baselines: vec![DiscardSink::new(Arc::clone(&source) as _, false)],
            task_ctx: None,
        };
        bench.measure(
            &case,
            |collect| {
                let sort = PhysicalSort::try_new(
                    Arc::clone(&source) as _,
                    ordering.clone(),
                    None,
                )?;
                Ok(Plan::new(
                    Arc::new(sort),
                    vec![Arc::clone(&source) as _],
                    collect,
                ))
            },
            // The sort reads its rows and writes them, sorted, once.
            |_| Floor {
                read: input_bytes,
                write: input_bytes,
            },
            |output| {
                expect_rows(output, rows as u64)?;
                expect_sorted(output, params.key).map(|_| ())
            },
        )?;
    }
    Ok(())
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct TopKParams {
    k: usize,
    order: Order,
    key: KeyKind,
    threads: usize,
}

impl TopKParams {
    fn label(&self) -> String {
        format!(
            "k={} order={} key={} threads={}",
            self.k,
            self.order.name(),
            self.key.name(),
            self.threads
        )
    }
}

/// `SELECT k, p FROM t ORDER BY k LIMIT <k>` over 4Mi rows of distinct keys.
fn top_k(bench: &mut Bench) -> Result<()> {
    let rows = bench.rows(4 << 20);
    let default = TopKParams {
        k: 100,
        order: Order::Random,
        key: KeyKind::I64,
        threads: 1,
    };
    let mut params = vec![default];
    params.extend([10, 10_000, 100_000].map(|k| TopKParams { k, ..default }));
    params.extend(
        [Order::Sorted, Order::Reverse].map(|order| TopKParams { order, ..default }),
    );
    params.push(TopKParams {
        key: KeyKind::Utf8,
        ..default
    });
    params.extend(
        bench
            .thread_sweep()
            .into_iter()
            .map(|threads| TopKParams { threads, ..default }),
    );
    let params = bench.select("top_k", params, TopKParams::label);
    for params in params {
        let k = params.k.min(rows);
        let (schema, columns, key_bytes) =
            sort_input(rows, rows, params.order, params.key, 7);
        let row_bytes = bytes_per_row(&columns);
        let source = ChunkSource::new(
            Arc::clone(&schema),
            chunks_of(&schema, &columns, DEFAULT_VECTOR_SIZE)?,
        );
        let ordering = key_ordering(&schema, params.key)?;
        let case = Case {
            operator: "top_k",
            config: params.label(),
            threads: params.threads,
            scope: Scope::All,
            baselines: vec![DiscardSink::new(Arc::clone(&source) as _, false)],
            task_ctx: None,
        };
        bench.measure(
            &case,
            |collect| {
                let top_k =
                    PhysicalTopK::try_new(Arc::clone(&source) as _, ordering.clone(), k)?;
                Ok(Plan::new(
                    Arc::new(top_k),
                    vec![Arc::clone(&source) as _],
                    collect,
                ))
            },
            // The TopK reads the keys of all rows, and writes its rows.
            |output_rows| Floor {
                read: key_bytes * rows as f64,
                write: row_bytes * output_rows as f64,
            },
            |output| {
                let ids = expect_sorted(output, params.key)?;
                if ids != (0..k as u64).collect::<Vec<_>>() {
                    return internal_err!("the TopK did not return the first {k} keys");
                }
                Ok(())
            },
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Window
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WindowFunction {
    /// `ROW_NUMBER() OVER (PARTITION BY p ORDER BY o)`.
    RowNumber,
    /// `SUM(v) OVER (PARTITION BY p ORDER BY o)`: a running sum.
    RunningSum,
    /// `SUM(v) OVER (PARTITION BY p)`: the sum of the partition.
    PartitionSum,
}

impl WindowFunction {
    fn name(self) -> &'static str {
        match self {
            Self::RowNumber => "row_number",
            Self::RunningSum => "running_sum",
            Self::PartitionSum => "partition_sum",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct WindowParams {
    function: WindowFunction,
    /// The window partitions; 1 for a window without `PARTITION BY`.
    partitions: usize,
    order: Order,
    threads: usize,
}

impl WindowParams {
    fn label(&self) -> String {
        format!(
            "fn={} partitions={} order={} threads={}",
            self.function.name(),
            rows_label(self.partitions),
            self.order.name(),
            self.threads
        )
    }
}

/// The window of `params` over `source` (`p`, `o`, `v`).
fn window(
    source: &Arc<ChunkSource>,
    params: WindowParams,
) -> Result<Arc<dyn PhysicalOperator>> {
    let schema = source.schema();
    let partition_by = if params.partitions > 1 {
        vec![col("p", &schema)?]
    } else {
        vec![]
    };
    let order_by = match params.function {
        WindowFunction::PartitionSum => vec![],
        WindowFunction::RowNumber | WindowFunction::RunningSum => {
            vec![PhysicalSortExpr::new_default(col("o", &schema)?)]
        }
    };
    let (function, args, frame) = match params.function {
        WindowFunction::RowNumber => (
            WindowFunctionDefinition::WindowUDF(row_number_udwf()),
            vec![],
            WindowFrame::new(Some(true)),
        ),
        WindowFunction::RunningSum => (
            WindowFunctionDefinition::AggregateUDF(sum_udaf()),
            vec![col("v", &schema)?],
            WindowFrame::new(Some(true)),
        ),
        WindowFunction::PartitionSum => (
            WindowFunctionDefinition::AggregateUDF(sum_udaf()),
            vec![col("v", &schema)?],
            WindowFrame::new(None),
        ),
    };
    let expr = create_window_expr(
        &function,
        params.function.name().to_owned(),
        &args,
        &partition_by,
        &order_by,
        Arc::new(frame),
        Arc::clone(&schema),
        false,
        false,
        None,
    )?;
    let ordering = partition_by
        .iter()
        .map(|key| PhysicalSortExpr::new_default(Arc::clone(key)))
        .chain(order_by.iter().cloned())
        .collect::<Vec<_>>();
    // A window without partitions is evaluated (in order) on one thread.
    let ordered = partition_by.is_empty() && !order_by.is_empty();
    Ok(Arc::new(PhysicalWindow::try_new(
        Arc::clone(source) as _,
        vec![expr],
        LexOrdering::new(ordering),
        ordered,
    )?))
}

/// `SELECT p, o, v, <fn> OVER (PARTITION BY p ORDER BY o) FROM t` over 2Mi
/// rows.
fn window_bench(bench: &mut Bench) -> Result<()> {
    let rows = bench.rows(2 << 20);
    let default = WindowParams {
        function: WindowFunction::RunningSum,
        partitions: 1 << 10,
        order: Order::Random,
        threads: 1,
    };
    let mut params = vec![default];
    params.extend(
        [WindowFunction::RowNumber, WindowFunction::PartitionSum].map(|function| {
            WindowParams {
                function,
                ..default
            }
        }),
    );
    params.extend([1, 16, 64 << 10].map(|partitions| WindowParams {
        partitions,
        ..default
    }));
    params.push(WindowParams {
        order: Order::Sorted,
        ..default
    });
    params.extend(
        bench
            .thread_sweep()
            .into_iter()
            .map(|threads| WindowParams { threads, ..default }),
    );
    let params = bench.select("window", params, WindowParams::label);
    for params in params {
        let mut rng = Rng::new(8);
        // Sorted: by `p`, then `o`; random: shuffled.
        let mut rows_of = (0..rows as u64)
            .map(|i| (i % params.partitions as u64, rng.below(1_000_000)))
            .collect::<Vec<_>>();
        params.order.arrange(&mut rows_of, &mut rng);
        let schema = schema_of(
            ["p", "o", "v"]
                .map(|name| Field::new(name, DataType::Int64, false))
                .to_vec(),
        );
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from_iter_values(
                rows_of.iter().map(|&(p, _)| p as i64),
            )),
            Arc::new(Int64Array::from_iter_values(
                rows_of.iter().map(|&(_, o)| o as i64),
            )),
            uniform_i64(&mut rng, rows, 1_000_000),
        ];
        let input_bytes = bytes_per_row(&columns) * rows as f64;
        let source = ChunkSource::new(
            Arc::clone(&schema),
            chunks_of(&schema, &columns, DEFAULT_VECTOR_SIZE)?,
        );
        let case = Case {
            operator: "window",
            config: params.label(),
            threads: params.threads,
            scope: Scope::All,
            baselines: vec![DiscardSink::new(Arc::clone(&source) as _, false)],
            task_ctx: None,
        };
        bench.measure(
            &case,
            |collect| {
                Ok(Plan::new(
                    window(&source, params)?,
                    vec![Arc::clone(&source) as _],
                    collect,
                ))
            },
            // The window reads its rows and writes them with the value of the
            // window function (8 bytes) once.
            |output_rows| Floor {
                read: input_bytes,
                write: input_bytes + 8.0 * output_rows as f64,
            },
            |output| expect_rows(output, rows as u64),
        )?;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Tuple-data ingestion
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RowLayout {
    /// One `Int64` column.
    I64,
    /// Four `Int64` columns.
    I64x4,
    /// `Int32`, `Int64` and `Float64`.
    Mixed,
    /// An `Int64` and a `Utf8View` of 16 bytes (stored on the heap of the
    /// collection).
    Utf8,
}

impl RowLayout {
    fn name(self) -> &'static str {
        match self {
            Self::I64 => "i64",
            Self::I64x4 => "4xi64",
            Self::Mixed => "i32,i64,f64",
            Self::Utf8 => "i64,utf8v16",
        }
    }

    fn columns(self, rows: usize, rng: &mut Rng) -> (SchemaRef, Vec<ArrayRef>) {
        let int64 = |name: &str| Field::new(name, DataType::Int64, false);
        let mut values = || uniform_i64(rng, rows, 1 << 40);
        match self {
            Self::I64 => (schema_of(vec![int64("a")]), vec![values()]),
            Self::I64x4 => (
                schema_of(["a", "b", "c", "d"].map(int64).to_vec()),
                (0..4).map(|_| values()).collect(),
            ),
            Self::Mixed => {
                let a = values();
                let b = values();
                let c = values();
                (
                    schema_of(vec![
                        Field::new("a", DataType::Int32, false),
                        int64("b"),
                        Field::new("c", DataType::Float64, false),
                    ]),
                    vec![
                        Arc::new(Int32Array::from_iter_values(
                            a.as_primitive::<Int64Type>()
                                .values()
                                .iter()
                                .map(|&a| a as i32),
                        )) as ArrayRef,
                        b,
                        Arc::new(Float64Array::from_iter_values(
                            c.as_primitive::<Int64Type>()
                                .values()
                                .iter()
                                .map(|&c| c as f64),
                        )),
                    ],
                )
            }
            Self::Utf8 => {
                let a = values();
                let ids = a
                    .as_primitive::<Int64Type>()
                    .values()
                    .iter()
                    .map(|&a| a as u64)
                    .collect::<Vec<_>>();
                (
                    schema_of(vec![
                        int64("a"),
                        Field::new("s", DataType::Utf8View, false),
                    ]),
                    vec![a, Arc::new(utf8_keys(&ids))],
                )
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct TupleParams {
    layout: RowLayout,
    selectivity: f64,
    radix_bits: u32,
    batch: usize,
    threads: usize,
}

impl TupleParams {
    fn label(&self) -> String {
        format!(
            "cols={} sel={:.2} partitions={} batch={} threads={}",
            self.layout.name(),
            self.selectivity,
            1 << self.radix_bits,
            self.batch,
            self.threads
        )
    }
}

/// Append 4Mi rows to `PartitionedTupleData`s (with a hash slot, as the
/// build of a hash join does): one per thread, every thread appending every
/// `threads`-th chunk.
fn tuple_data(bench: &mut Bench) -> Result<()> {
    let default = TupleParams {
        layout: RowLayout::I64x4,
        selectivity: 1.0,
        radix_bits: 0,
        batch: DEFAULT_VECTOR_SIZE,
        threads: 1,
    };
    let mut params = vec![default];
    params.extend(
        [RowLayout::I64, RowLayout::Mixed, RowLayout::Utf8]
            .map(|layout| TupleParams { layout, ..default }),
    );
    params.push(TupleParams {
        selectivity: 0.5,
        ..default
    });
    params.extend([4, 8].map(|radix_bits| TupleParams {
        radix_bits,
        ..default
    }));
    params.extend([256, 1024].map(|batch| TupleParams { batch, ..default }));
    params.extend(
        bench
            .thread_sweep()
            .into_iter()
            .map(|threads| TupleParams { threads, ..default }),
    );
    let params = bench.select("tuple_data", params, TupleParams::label);
    let rows = bench.rows(4 << 20);
    for params in params {
        let mut rng = Rng::new(9);
        let (schema, columns) = params.layout.columns(rows, &mut rng);
        let mut chunks = chunks_of(&schema, &columns, params.batch)?;
        if params.selectivity < 1.0 {
            chunks = with_selections(chunks, params.selectivity)?;
        }
        // Any hash spreads the rows over the partitions.
        let mut next = 0_u64;
        let hashes = chunks
            .iter()
            .map(|chunk| {
                (0..chunk.len())
                    .map(|_| {
                        next += 1;
                        mix64(next)
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        let layout = Arc::new(TupleDataLayout::try_new(Arc::clone(&schema), true)?);
        let logical_rows = chunks.iter().map(|chunk| chunk.len() as u64).sum::<u64>();
        let bytes_in = chunks
            .iter()
            .map(|chunk| chunk_bytes(chunk) as u64)
            .sum::<u64>();
        let case = Case {
            operator: "tuple_data",
            config: params.label(),
            threads: params.threads,
            scope: Scope::Operator,
            baselines: vec![],
            task_ctx: None,
        };
        // The first execution is the warm-up.
        let mut samples = vec![];
        for iteration in 0..=bench.config.iterations {
            let sample = append_tuple_data(&layout, &chunks, &hashes, params)?;
            if sample.rows != logical_rows {
                return internal_err!(
                    "tuple_data {}: stored {} rows, expected {logical_rows}",
                    params.label(),
                    sample.rows
                );
            }
            if iteration > 0 {
                samples.push(Sample {
                    bytes_in,
                    chunks: chunks.len() as u64,
                    ..sample
                });
            }
        }
        // The ingestion reads (the lines of) the values of the rows and
        // writes them once.
        let floor = Floor {
            read: chunks.iter().map(read_line_bytes).sum(),
            write: bytes_in as f64,
        };
        bench.record(&case, &samples, floor);
    }
    Ok(())
}

/// Append `chunks` (with the hashes `hashes`) to `PartitionedTupleData`s of
/// `layout` on `params.threads` threads.
fn append_tuple_data(
    layout: &Arc<TupleDataLayout>,
    chunks: &[DataChunk],
    hashes: &[Vec<u64>],
    params: TupleParams,
) -> Result<Sample> {
    let threads = params.threads;
    let before = allocations::snapshot();
    let start = Instant::now();
    let results = std::thread::scope(|scope| {
        let handles = (0..threads)
            .map(|thread| {
                scope.spawn(move || -> Result<(u64, u64, u64)> {
                    let start = Instant::now();
                    let mut data = PartitionedTupleData::try_new(
                        Arc::clone(layout),
                        params.radix_bits,
                    )?;
                    for (chunk, hashes) in
                        chunks.iter().zip(hashes).skip(thread).step_by(threads)
                    {
                        data.append_chunk(chunk, hashes)?;
                    }
                    let elapsed = start.elapsed().as_nanos() as u64;
                    let rows = data.partitions().iter().map(|p| p.len() as u64).sum();
                    Ok((elapsed, rows, data.memory_size() as u64))
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("the appending thread panicked"))
            .collect::<Result<Vec<_>>>()
    })?;
    let wall_time = start.elapsed().as_nanos() as f64;
    let allocations = allocations::snapshot().since(before);
    let mut sample = Sample {
        wall_time,
        allocations,
        ..Sample::default()
    };
    for (elapsed, rows, memory) in results {
        sample.self_time += elapsed;
        sample.rows += rows;
        sample.memory_peak += memory;
        sample.bytes_out += memory;
    }
    Ok(sample)
}

// ---------------------------------------------------------------------------
// Parquet decode
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Codec {
    None,
    Snappy,
    Zstd,
}

impl Codec {
    fn name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Snappy => "snappy",
            Self::Zstd => "zstd1",
        }
    }

    fn compression(self) -> Result<Compression> {
        Ok(match self {
            Self::None => Compression::UNCOMPRESSED,
            Self::Snappy => Compression::SNAPPY,
            Self::Zstd => Compression::ZSTD(ZstdLevel::try_new(1)?),
        })
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
struct ParquetParams {
    column: KeyKind,
    /// The distinct values, 0 for all rows.
    distinct: usize,
    codec: Codec,
    batch: usize,
    threads: usize,
}

impl ParquetParams {
    fn label(&self, rows: usize) -> String {
        let distinct = if self.distinct == 0 {
            rows
        } else {
            self.distinct
        };
        format!(
            "col={} distinct={} codec={} batch={} threads={}",
            self.column.name(),
            rows_label(distinct),
            self.codec.name(),
            self.batch,
            self.threads
        )
    }
}

/// The files of the Parquet decode benchmark.
const PARQUET_FILES: usize = 16;
/// The rows of their row groups.
const PARQUET_ROW_GROUP_ROWS: usize = 64 << 10;

/// Decode 4Mi rows of a single column from 16 Parquet files of row groups
/// of 64Ki rows, in an in-memory object store.
fn parquet_decode(bench: &mut Bench) -> Result<()> {
    let rows = bench.rows(4 << 20);
    let default = ParquetParams {
        column: KeyKind::I64,
        distinct: 0,
        codec: Codec::None,
        batch: 8192,
        threads: 1,
    };
    let mut params = vec![default];
    params.extend(
        [KeyKind::I32, KeyKind::Utf8].map(|column| ParquetParams { column, ..default }),
    );
    params.extend([16, 64 << 10].map(|distinct| ParquetParams {
        distinct,
        ..default
    }));
    params.extend(
        [Codec::Snappy, Codec::Zstd].map(|codec| ParquetParams { codec, ..default }),
    );
    params.extend([2048, 32768].map(|batch| ParquetParams { batch, ..default }));
    params.extend(
        bench
            .thread_sweep()
            .into_iter()
            .map(|threads| ParquetParams { threads, ..default }),
    );
    let params = bench.select("parquet_decode", params, |params| params.label(rows));
    if params.is_empty() {
        return Ok(());
    }
    let store = Arc::new(InMemory::new());
    let url = ObjectStoreUrl::parse("memory://")?;
    let session = SessionContext::new();
    session
        .register_object_store(url.as_ref(), Arc::clone(&store) as Arc<dyn ObjectStore>);
    let runtime_env = session.runtime_env();
    let mut written = HashMap::new();
    for params in params {
        let distinct = if params.distinct == 0 {
            rows
        } else {
            params.distinct
        };
        let file_key = (params.column.name(), distinct, params.codec.name());
        let next = written.len();
        let (schema, files, file_bytes) = match written.get(&file_key) {
            Some(files) => Clone::clone(files),
            None => {
                let files = write_parquet_files(
                    bench,
                    &store,
                    &format!("{next}"),
                    rows,
                    distinct,
                    params.column,
                    params.codec,
                )?;
                written.insert(file_key, files.clone());
                files
            }
        };
        let session = SessionConfig::new()
            .with_target_partitions(params.threads)
            .with_batch_size(params.batch);
        let task_ctx =
            SessionContext::new_with_config_rt(session, Arc::clone(&runtime_env))
                .task_ctx();
        let config = FileScanConfigBuilder::new(
            url.clone(),
            Arc::new(ParquetSource::new(Arc::clone(&schema))),
        )
        .with_file_groups(vec![FileGroup::new(files)])
        .build();
        let case = Case {
            operator: "parquet_decode",
            config: params.label(rows),
            threads: params.threads,
            scope: Scope::All,
            baselines: vec![],
            task_ctx: Some(task_ctx),
        };
        let key = params.column;
        // The bytes of the decoded column.
        let decoded = bytes_per_row(&key.columns(&[0, 1])) * rows as f64;
        bench.measure(
            &case,
            |collect| {
                let scan: Arc<dyn PhysicalOperator> =
                    Arc::new(ParquetScanSource::try_from_config(config.clone())?);
                Ok(Plan::new(scan, vec![], collect))
            },
            // The decode reads the files and writes the decoded column.
            |_| Floor {
                read: file_bytes as f64,
                write: decoded,
            },
            |output| expect_rows(output, rows as u64),
        )?;
    }
    Ok(())
}

/// Write `rows` values of a column `k` of kind `column` (`distinct`
/// distinct ones, in a random order) to [`PARQUET_FILES`] files in `store`,
/// compressed with `codec`. Returns the schema, the files and their bytes.
fn write_parquet_files(
    bench: &Bench,
    store: &Arc<InMemory>,
    prefix: &str,
    rows: usize,
    distinct: usize,
    column: KeyKind,
    codec: Codec,
) -> Result<(SchemaRef, Vec<PartitionedFile>, u64)> {
    let mut rng = Rng::new(10);
    let ids = key_ids(rows, distinct, Order::Random, &mut rng);
    let schema = schema_of(column.fields("k"));
    let properties = WriterProperties::builder()
        .set_compression(codec.compression()?)
        .set_max_row_group_row_count(Some(PARQUET_ROW_GROUP_ROWS))
        .build();
    let file_rows = rows.div_ceil(PARQUET_FILES);
    let mut files = vec![];
    let mut total = 0;
    for (file, ids) in ids.chunks(file_rows).enumerate() {
        let batch = RecordBatch::try_new(Arc::clone(&schema), column.columns(ids))?;
        let mut buffer = vec![];
        let mut writer = ArrowWriter::try_new(
            &mut buffer,
            Arc::clone(&schema),
            Some(properties.clone()),
        )?;
        writer.write(&batch)?;
        writer.close()?;
        let path = format!("{prefix}/file-{file}.parquet");
        let size = buffer.len() as u64;
        total += size;
        bench.runtime.block_on(store.put(
            &ObjectPath::from(path.as_str()),
            PutPayload::from(Bytes::from(buffer)),
        ))?;
        files.push(PartitionedFile::new(path, size));
    }
    Ok((schema, files, total))
}

// ---------------------------------------------------------------------------
// Report
// ---------------------------------------------------------------------------

/// `value` with an SI suffix: `4.19M`, `65.5K`.
fn si(value: f64) -> String {
    if value >= 1e9 {
        format!("{:.2}G", value / 1e9)
    } else if value >= 1e6 {
        format!("{:.2}M", value / 1e6)
    } else if value >= 1e3 {
        format!("{:.1}K", value / 1e3)
    } else {
        format!("{value:.0}")
    }
}

/// `value` with about three significant digits.
fn short(value: f64) -> String {
    if value == 0.0 {
        "0".to_owned()
    } else if value.abs() >= 100.0 {
        format!("{value:.0}")
    } else if value.abs() >= 10.0 {
        format!("{value:.1}")
    } else if value.abs() >= 1.0 {
        format!("{value:.2}")
    } else {
        format!("{value:.3}")
    }
}

impl Measurement {
    /// The cells of the row of this measurement in the report.
    fn cells(&self) -> Vec<String> {
        vec![
            self.operator.to_owned(),
            self.config.clone(),
            si(self.rows as f64),
            short(self.ns_per_row),
            short(self.rows_per_second / 1e6),
            short(self.allocations_per_chunk),
            short(self.copied_per_row),
            short(self.in_per_row),
            short(self.out_per_row),
            si(self.memory_peak as f64),
            short(self.floor_per_row),
            self.floor_share
                .map_or_else(|| "-".to_owned(), |share| format!("{:.1}", 100.0 * share)),
            format!("{:.1}", 100.0 * self.cv),
            self.phases.map_or_else(
                || "-".to_owned(),
                |[sink, finalize, source]| {
                    format!(
                        "{:.0}/{:.0}/{:.0}",
                        100.0 * sink,
                        100.0 * finalize,
                        100.0 * source
                    )
                },
            ),
        ]
    }
}

const HEADERS: [&str; 14] = [
    "operator",
    "configuration",
    "rows",
    "ns/row",
    "Mrows/s",
    "allocs/chunk",
    "copied B/row",
    "in B/row",
    "out B/row",
    "memory_peak",
    "floor B/row",
    "% floor",
    "cv %",
    "sink/fin/src %",
];

/// The report: the bandwidths, then the measurements as a table and as a
/// Markdown table.
fn report(bench: &Bench) -> String {
    let mut out = String::new();
    let config = &bench.config;
    let _ = writeln!(
        out,
        "\noperator_micro: scale {}, {} timed executions per configuration, {} cores{}",
        config.scale,
        config.iterations,
        std::thread::available_parallelism().map_or(0, |n| n.get()),
        if config.count_allocations {
            ""
        } else {
            ", allocations not counted"
        }
    );
    let _ = writeln!(
        out,
        "\nMemory bandwidth (GB/s, buffers of {} MiB):",
        config.bandwidth_bytes >> 20
    );
    let _ = writeln!(out, "\n| threads | read | write | copy (read + write) |");
    let _ = writeln!(out, "| ------: | ---: | ----: | ------------------: |");
    for bandwidth in &bench.bandwidths {
        let _ = writeln!(
            out,
            "| {} | {:.1} | {:.1} | {:.1} |",
            bandwidth.threads, bandwidth.read, bandwidth.write, bandwidth.copy
        );
    }
    let rows = bench
        .measurements
        .iter()
        .map(Measurement::cells)
        .collect::<Vec<_>>();
    let widths = HEADERS
        .iter()
        .enumerate()
        .map(|(i, header)| {
            rows.iter()
                .map(|row| row[i].len())
                .chain([header.len()])
                .max()
                .unwrap_or(0)
        })
        .collect::<Vec<_>>();
    let line = |cells: &[String]| {
        let mut line = String::new();
        for (i, cell) in cells.iter().enumerate() {
            if i < 2 {
                let _ = write!(line, "{cell:<width$}  ", width = widths[i]);
            } else {
                let _ = write!(line, "{cell:>width$}  ", width = widths[i]);
            }
        }
        line.trim_end().to_owned()
    };
    let _ = writeln!(out);
    let _ = writeln!(out, "{}", line(&HEADERS.map(str::to_owned)));
    for row in &rows {
        let _ = writeln!(out, "{}", line(row));
    }
    let _ = writeln!(out, "\n| {} |", HEADERS.join(" | "));
    let _ = writeln!(
        out,
        "| {} |",
        HEADERS
            .iter()
            .enumerate()
            .map(|(i, _)| if i < 2 { "---" } else { "---:" })
            .collect::<Vec<_>>()
            .join(" | ")
    );
    for row in &rows {
        // The configurations have characters that Markdown interprets
        // (`a*b`, `a<x&b<y`).
        let mut row = row.clone();
        row[1] = format!("`{}`", row[1]);
        let _ = writeln!(out, "| {} |", row.join(" | "));
    }
    out
}

/// The measurements as tab-separated values, for `OPERATOR_MICRO_COMPARE`.
fn tsv(bench: &Bench) -> String {
    let mut out = String::from("operator\tconfiguration\tns_per_row\tmrows_per_s\n");
    for measurement in &bench.measurements {
        let _ = writeln!(
            out,
            "{}\t{}\t{}\t{}",
            measurement.operator,
            measurement.config,
            measurement.ns_per_row,
            measurement.rows_per_second / 1e6
        );
    }
    out
}

/// The comparison of the `ns/row` of the measurements with those of an
/// earlier run (`earlier`, as written by [`tsv`]).
fn comparison(bench: &Bench, earlier: &str) -> String {
    let earlier = earlier
        .lines()
        .skip(1)
        .filter_map(|line| {
            let fields = line.split('\t').collect::<Vec<_>>();
            let ns_per_row = fields.get(2)?.parse::<f64>().ok()?;
            Some((format!("{} {}", fields[0], fields[1]), ns_per_row))
        })
        .collect::<HashMap<_, _>>();
    let mut out = String::from("\nns/row compared with the earlier run:\n\n");
    let mut changes = vec![];
    for measurement in &bench.measurements {
        let name = measurement.name();
        if let Some(&before) = earlier.get(&name) {
            let change = (measurement.ns_per_row - before) / before;
            changes.push((change.abs(), name.clone()));
            let _ = writeln!(
                out,
                "{name:<72} {before:>9.2} {:>9.2} {:>+7.1}%",
                measurement.ns_per_row,
                100.0 * change
            );
        }
    }
    if !changes.is_empty() {
        let mut values = changes
            .iter()
            .map(|(change, _)| *change)
            .collect::<Vec<_>>();
        let median_change = median(&mut values);
        changes.sort_unstable_by(|a, b| b.0.total_cmp(&a.0));
        let within = |limit: f64| {
            changes
                .iter()
                .filter(|(change, _)| *change <= limit)
                .count() as f64
                / changes.len() as f64
        };
        let _ = writeln!(
            out,
            "\n{} configurations: median |change| {:.1}%, {:.0}% within 5%, {:.0}% within 10%, largest {:.1}% ({})",
            changes.len(),
            100.0 * median_change,
            100.0 * within(0.05),
            100.0 * within(0.10),
            100.0 * changes[0].0,
            changes[0].1
        );
    }
    out
}

/// Write `text` to stdout, bypassing the output capture of libtest.
fn print_uncaptured(text: &str) -> Result<()> {
    let mut stdout = std::io::stdout().lock();
    stdout.write_all(text.as_bytes())?;
    stdout.flush()?;
    Ok(())
}

/// The benchmark, see the module documentation.
#[test]
fn operator_micro() -> Result<()> {
    let config = BenchConfig::from_env();
    allocations::set_counting(config.count_allocations);
    let mut bench = Bench::try_new(config)?;
    bench.measure_bandwidths();
    projection(&mut bench)?;
    filter(&mut bench)?;
    hash_join_build(&mut bench)?;
    hash_join_probe(&mut bench)?;
    aggregate(&mut bench)?;
    sort(&mut bench)?;
    top_k(&mut bench)?;
    window_bench(&mut bench)?;
    tuple_data(&mut bench)?;
    parquet_decode(&mut bench)?;
    print_uncaptured(&report(&bench))?;
    if let Some(path) = &bench.config.save {
        std::fs::write(path, tsv(&bench))?;
    }
    if let Some(path) = &bench.config.compare {
        print_uncaptured(&comparison(&bench, &std::fs::read_to_string(path)?))?;
    }
    Ok(())
}
