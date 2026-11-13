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
//! `ChunkSource`s hand out pre-built chunks (every chunk with buffers of its
//! own) to the threads of their pipelines, and the `DiscardSink` drops the
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
//! output of the operator (its rows, and e.g. their order for a sort) and
//! measures the bytes of the output that the operator wrote; an execution
//! without profiling counts its allocations and copies; the other numbers are
//! the medians of the timed executions that follow.
//!
//! # Columns of the report
//!
//! | Column           | Meaning                                                                                                                                         |
//! | ---------------- | ----------------------------------------------------------------------------------------------------------------------------------------------- |
//! | `rows`           | the rows of the operator: its `rows_in` (the probe rows of a probe, the build rows of a build), its `rows_out` for a source (Parquet decode)       |
//! | `ns/row`         | `self_time / rows`: the time of the operator per row, summed over its threads (CPU time)                                                        |
//! | `Mrows/s`        | `rows` per second of the wall time of the operator (the elapsed time of its pipelines, attributed to the operator by its share of their active time) |
//! | `allocs/chunk`   | the heap allocations of the operator per chunk of its input (`morsels_in`)                                                                       |
//! | `alloc B/row`    | the bytes that the operator allocated per row                                                                                                   |
//! | `copied B/row`   | the bytes that the operator copied per row with `memcpy` and `memmove`, into new and into reused buffers, and that its `realloc`s moved          |
//! | `written B/row`  | the bytes per row of the buffers of its output that the operator wrote (gathered, computed or copied), rather than passed through from its input |
//! | `in B/row`       | `bytes_in / rows`, see `profiling::chunk_bytes`                                                                                                 |
//! | `out B/row`      | `bytes_out / rows` (the bytes of the rows a tuple-data collection holds, for tuple-data ingestion)                                              |
//! | `memory_peak`    | the high-water mark of the memory that the operator reserved                                                                                    |
//! | `floor B/row`    | the bytes per row that the operator must read and write, whatever its implementation (see `docs/operator-performance.md`)                         |
//! | `% floor`        | the time it takes to move the floor bytes at the bandwidth of the machine (on as many threads), over the wall time of the operator              |
//! | `cv %`           | the coefficient of variation of `ns/row` over the timed executions                                                                              |
//! | `sink/fin/src %` | the shares of the time of a pipeline breaker spent as a sink, finalizing, and as a source (the `JoinHashTableBuild` for a hash join build)       |
//!
//! The global allocator of this benchmark counts the allocations, and on
//! Linux with glibc the binary interposes `memcpy` and `memmove`, which count
//! the bytes they copy (see [`counters`]); copies that the compiler inlines
//! (of a few bytes) and value-by-value gathers and scatters are not calls of
//! `memcpy`, and show in `written B/row` and `alloc B/row` instead. Only the
//! threads of the benchmark count, in a dedicated execution without
//! profiling (whose own allocations would count too), and only in the window
//! of the execution that the operator is measured in, which the
//! `DiscardSink` marks (see [`Window`]): the whole execution for most
//! operators, the execution until the probe pipeline starts for the build of
//! a hash join, and the probe pipeline for its probe and for the streaming
//! operators. The counts of the harness are those of its sources into a
//! `DiscardSink`, counted in the window that covers the same pipelines of the
//! harness (see [`Baseline`]), and are subtracted. The bytes written are
//! those of the buffers of the output chunks (once each) that do not lie in
//! the buffers of the input chunks.
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
    Array, ArrayData, ArrayRef, AsArray, Float64Array, Int32Array, Int64Array,
    RecordBatch, StringViewArray, StringViewBuilder,
};
use arrow::buffer::Buffer;
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
    DEFAULT_VECTOR_SIZE, DataChunk, EmptyState, ExecutionContext, Executor,
    GlobalSinkState, GlobalSourceState, LocalSinkState, PhysicalOperator,
    SelectionVector, SinkFinalize, SinkResult, SourceResult,
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
// Allocations and copies
// ---------------------------------------------------------------------------

/// The global allocator of the benchmark: the system allocator, which counts
/// the allocations (see [`counters`]).
#[global_allocator]
static ALLOCATOR: counters::Counting = counters::Counting;

/// Counters of the allocations, the allocated bytes and the copied bytes of
/// the threads of a [`counters::Session`] (a [`Bench`]).
///
/// The global allocator counts every `alloc`, `alloc_zeroed` and `realloc`
/// (with its new size), and as moved the bytes that a `realloc` moves to a
/// new block (which depends on the layout of the heap). On
/// Linux with glibc, the benchmark binary interposes the C library's `memcpy`
/// and `memmove` (see [`copies`]), which count the bytes they copy: the bulk
/// copies of all code of the binary, into new and into reused buffers alike.
/// (Copies of a few bytes whose size the compiler knows are inlined, and
/// gathers and scatters copy value by value: these are not calls of
/// `memcpy`.)
///
/// Only the threads that a session tagged count, into the counters of that
/// session and only while the session is enabled, so that the threads of
/// other sessions and of the test harness never disturb a measurement. The
/// counters are sharded (one cache line per shard, a shard per thread), so
/// that counting neither contends nor slows down the threads of an execution
/// measurably.
mod counters {
    use std::alloc::{GlobalAlloc, Layout, System};
    use std::cell::Cell;
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

    /// The most sessions of a process.
    const SESSIONS: usize = 32;
    const SHARDS: usize = 32;

    /// The counters of the threads of a shard, on a cache line of their own
    /// (Neoverse-N1 prefetches pairs of 64-byte lines).
    #[repr(align(128))]
    struct Shard {
        allocations: AtomicU64,
        allocated: AtomicU64,
        copied: AtomicU64,
        moved: AtomicU64,
    }

    static COUNTERS: [[Shard; SHARDS]; SESSIONS] = [const {
        [const {
            Shard {
                allocations: AtomicU64::new(0),
                allocated: AtomicU64::new(0),
                copied: AtomicU64::new(0),
                moved: AtomicU64::new(0),
            }
        }; SHARDS]
    }; SESSIONS];

    /// Whether the threads of a session count.
    static ENABLED: [AtomicBool; SESSIONS] = [const { AtomicBool::new(false) }; SESSIONS];

    /// The next session (0 is no session).
    static NEXT_SESSION: AtomicUsize = AtomicUsize::new(1);

    /// The shard of the next thread that a session tags.
    static NEXT_SHARD: AtomicUsize = AtomicUsize::new(0);

    thread_local! {
        /// The session of this thread (0 if none) and its shard. A `const`
        /// thread local without a destructor: accessing it never allocates
        /// nor copies.
        static THREAD: Cell<(usize, usize)> = const { Cell::new((0, 0)) };
    }

    /// The allocations, the allocated bytes, the bytes copied by `memcpy`
    /// and `memmove` and the bytes moved by `realloc` of a session.
    #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
    pub struct Counts {
        pub allocations: u64,
        pub allocated: u64,
        pub copied: u64,
        pub moved: u64,
    }

    impl Counts {
        /// The counts since `earlier`.
        pub fn since(self, earlier: Self) -> Self {
            Self {
                allocations: self.allocations - earlier.allocations,
                allocated: self.allocated - earlier.allocated,
                copied: self.copied - earlier.copied,
                moved: self.moved - earlier.moved,
            }
        }

        pub fn add(self, other: Self) -> Self {
            Self {
                allocations: self.allocations + other.allocations,
                allocated: self.allocated + other.allocated,
                copied: self.copied + other.copied,
                moved: self.moved + other.moved,
            }
        }

        /// These counts without `other`, at least 0.
        pub fn saturating_sub(self, other: Self) -> Self {
            Self {
                allocations: self.allocations.saturating_sub(other.allocations),
                allocated: self.allocated.saturating_sub(other.allocated),
                copied: self.copied.saturating_sub(other.copied),
                moved: self.moved.saturating_sub(other.moved),
            }
        }
    }

    /// The counters of the threads that it tags.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct Session(usize);

    impl Session {
        pub fn new() -> Self {
            let session = NEXT_SESSION.fetch_add(1, Ordering::Relaxed);
            assert!(session < SESSIONS, "more than {SESSIONS} sessions");
            Self(session)
        }

        /// The session of the current thread, if it has one.
        pub fn current() -> Option<Self> {
            let (session, _) = THREAD.try_with(Cell::get).unwrap_or((0, 0));
            (session != 0).then_some(Self(session))
        }

        /// Count the allocations and copies of the current thread in this
        /// session.
        pub fn tag_current_thread(self) {
            let shard = NEXT_SHARD.fetch_add(1, Ordering::Relaxed) % SHARDS;
            THREAD.with(|thread| thread.set((self.0, shard)));
        }

        /// Count (or not) the allocations and copies of the threads of this
        /// session.
        pub fn set_enabled(self, enabled: bool) {
            ENABLED[self.0].store(enabled, Ordering::SeqCst);
        }

        /// The counts of all threads of this session so far.
        pub fn snapshot(self) -> Counts {
            COUNTERS[self.0]
                .iter()
                .fold(Counts::default(), |counts, shard| Counts {
                    allocations: counts.allocations
                        + shard.allocations.load(Ordering::Relaxed),
                    allocated: counts.allocated + shard.allocated.load(Ordering::Relaxed),
                    copied: counts.copied + shard.copied.load(Ordering::Relaxed),
                    moved: counts.moved + shard.moved.load(Ordering::Relaxed),
                })
        }
    }

    /// Whether the copies of `memcpy` and `memmove` are counted (see
    /// [`copies`](super::copies)).
    pub const COUNTS_COPIES: bool = cfg!(all(target_os = "linux", target_env = "gnu"));

    /// The shard of the current thread, if it counts.
    #[inline]
    fn shard() -> Option<&'static Shard> {
        let (session, shard) = THREAD.try_with(Cell::get).unwrap_or((0, 0));
        (session != 0 && ENABLED[session].load(Ordering::Relaxed))
            .then(|| &COUNTERS[session][shard])
    }

    #[inline]
    fn record_allocation(bytes: usize) {
        if let Some(shard) = shard() {
            shard.allocations.fetch_add(1, Ordering::Relaxed);
            shard.allocated.fetch_add(bytes as u64, Ordering::Relaxed);
        }
    }

    /// Count `bytes` copied by the current thread.
    #[inline]
    pub fn record_copy(bytes: usize) {
        if let Some(shard) = shard() {
            shard.copied.fetch_add(bytes as u64, Ordering::Relaxed);
        }
    }

    /// The system allocator, counting its allocations.
    pub struct Counting;

    // SAFETY: every method forwards its arguments to the system allocator,
    // which upholds the contract of `GlobalAlloc`.
    unsafe impl GlobalAlloc for Counting {
        unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
            record_allocation(layout.size());
            // SAFETY: the caller upholds the contract of `GlobalAlloc::alloc`.
            unsafe { System.alloc(layout) }
        }

        unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
            record_allocation(layout.size());
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
            record_allocation(new_size);
            // SAFETY: the caller upholds the contract of
            // `GlobalAlloc::realloc`.
            let moved = unsafe { System.realloc(ptr, layout, new_size) };
            // A block that moved was copied (by the C library's own
            // `memcpy`, which the benchmark does not interpose).
            if !moved.is_null()
                && moved != ptr
                && let Some(shard) = shard()
            {
                let bytes = layout.size().min(new_size) as u64;
                shard.moved.fetch_add(bytes, Ordering::Relaxed);
            }
            moved
        }

        unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
            // SAFETY: the caller upholds the contract of
            // `GlobalAlloc::dealloc`.
            unsafe { System.dealloc(ptr, layout) }
        }
    }
}

/// The `memcpy` and `memmove` of the benchmark binary, which interpose those
/// of glibc for all code of the binary: they count the bytes they copy (see
/// [`counters`]) and copy them with glibc's `__memcpy_chk` and
/// `__memmove_chk`, which branch to glibc's own implementations (not to
/// these functions) after checking the size of the target, which is unknown
/// here (`usize::MAX`).
#[cfg(all(target_os = "linux", target_env = "gnu"))]
mod copies {
    use std::ffi::c_void;
    use std::hint::black_box;

    unsafe extern "C" {
        fn __memcpy_chk(
            target: *mut c_void,
            source: *const c_void,
            len: usize,
            target_len: usize,
        ) -> *mut c_void;
        fn __memmove_chk(
            target: *mut c_void,
            source: *const c_void,
            len: usize,
            target_len: usize,
        ) -> *mut c_void;
    }

    /// `memcpy`, counting the copied bytes.
    ///
    /// # Safety
    ///
    /// The contract of the C library's `memcpy`.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn memcpy(
        target: *mut c_void,
        source: *const c_void,
        len: usize,
    ) -> *mut c_void {
        super::counters::record_copy(len);
        // The size of the target is opaque to the compiler, so that it does
        // not turn the call back into one of `memcpy` (this function).
        // SAFETY: the caller upholds the contract of `memcpy`, and the target
        // is large enough.
        unsafe { __memcpy_chk(target, source, len, black_box(usize::MAX)) }
    }

    /// `memmove`, counting the moved bytes.
    ///
    /// # Safety
    ///
    /// The contract of the C library's `memmove`.
    #[unsafe(no_mangle)]
    pub unsafe extern "C" fn memmove(
        target: *mut c_void,
        source: *const c_void,
        len: usize,
    ) -> *mut c_void {
        super::counters::record_copy(len);
        // SAFETY: as for `memcpy`.
        unsafe { __memmove_chk(target, source, len, black_box(usize::MAX)) }
    }
}

use counters::{Counts, Session};

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

/// A parallel sink that drops the chunks of the operator, or collects them
/// (compacted) for the checks of the warm-up execution, with the bytes of
/// the output that the operator wrote (see [`DiscardSink::collecting`]).
///
/// It marks the windows of the counts of an execution (see [`Window`]): it
/// takes a snapshot of the counts of the session of the executing thread when
/// the root pipeline starts (the executor creates the global state of the
/// sink of a pipeline before anything else of it) and when the root meta
/// pipeline is finalized (once all pipelines completed).
struct DiscardSink {
    children: Vec<Arc<dyn PhysicalOperator>>,
    collected: Option<Mutex<Collected>>,
    /// The counts when the root pipeline started and when it was finalized,
    /// in the last execution.
    marks: Mutex<[Option<Counts>; 2]>,
}

/// The output that a [`DiscardSink`] collected.
#[derive(Default)]
struct Collected {
    batches: Vec<RecordBatch>,
    /// The memory of the buffers of the input of the operator, sorted and
    /// disjoint: the buffers of the columns that it passes through lie in it.
    input: Vec<(usize, usize)>,
    /// The memory of the buffers of the output columns outside of `input`.
    written: Vec<(usize, usize)>,
}

impl std::fmt::Debug for DiscardSink {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DiscardSink")
            .field("collect", &self.collected.is_some())
            .finish()
    }
}

impl DiscardSink {
    /// A sink that drops the chunks of `child`.
    fn new(child: Arc<dyn PhysicalOperator>) -> Arc<Self> {
        Self::with(child, None)
    }

    /// A sink that collects the chunks of `child`, whose input are the
    /// chunks of the `ChunkSource`s among `sources`, and measures the bytes
    /// of their columns that `child` wrote: the bytes of the buffers that are
    /// not those of its input.
    fn collecting(
        child: Arc<dyn PhysicalOperator>,
        sources: &[Arc<dyn PhysicalOperator>],
    ) -> Arc<Self> {
        let mut input = vec![];
        for source in sources {
            if let Some(source) = source.as_ref().downcast_ref::<ChunkSource>() {
                for chunk in &source.chunks {
                    for column in chunk.columns() {
                        buffer_ranges(&column.to_data(), &mut input);
                    }
                }
            }
        }
        let collected = Collected {
            input: merged(input),
            ..Collected::default()
        };
        Self::with(child, Some(collected))
    }

    fn with(child: Arc<dyn PhysicalOperator>, collected: Option<Collected>) -> Arc<Self> {
        Arc::new(Self {
            children: vec![child],
            collected: collected.map(Mutex::new),
            marks: Mutex::new([None, None]),
        })
    }

    /// The collected batches, and the bytes of the buffers of their columns
    /// that the operator wrote.
    fn take_collected(&self) -> (Vec<RecordBatch>, u64) {
        let Some(collected) = &self.collected else {
            return (vec![], 0);
        };
        let collected = std::mem::take(&mut *collected.lock());
        let written = merged(collected.written)
            .iter()
            .map(|(start, end)| end - start)
            .sum::<usize>();
        (collected.batches, written as u64)
    }

    /// The counts of the last execution, which started at `start`, in its
    /// windows.
    fn windows(&self, start: Counts) -> Result<Windows> {
        let [Some(root_start), Some(root_end)] = *self.marks.lock() else {
            return internal_err!("The execution did not mark the windows of its counts");
        };
        Ok(Windows {
            before_root: root_start.since(start),
            root: root_end.since(root_start),
        })
    }

    /// Mark the window boundary `index` of the current execution.
    fn mark(&self, index: usize) {
        self.marks.lock()[index] = Session::current().map(Session::snapshot);
    }
}

/// Add the memory of the buffers of `data` (and of its children) to
/// `ranges`.
fn buffer_ranges(data: &ArrayData, ranges: &mut Vec<(usize, usize)>) {
    let mut add = |buffer: &Buffer| {
        if !buffer.is_empty() {
            let start = buffer.as_ptr().addr();
            ranges.push((start, start + buffer.len()));
        }
    };
    data.buffers().iter().for_each(&mut add);
    if let Some(nulls) = data.nulls() {
        add(nulls.buffer());
    }
    for child in data.child_data() {
        buffer_ranges(child, ranges);
    }
}

/// `ranges` sorted, with the ranges that overlap or touch merged.
fn merged(mut ranges: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    ranges.sort_unstable();
    let mut merged: Vec<(usize, usize)> = vec![];
    for (start, end) in ranges {
        match merged.last_mut() {
            Some(last) if start <= last.1 => last.1 = last.1.max(end),
            _ => merged.push((start, end)),
        }
    }
    merged
}

/// Whether `range` lies in one of `ranges` (sorted and disjoint).
fn contained(ranges: &[(usize, usize)], (start, end): (usize, usize)) -> bool {
    let index = ranges.partition_point(|&(first, _)| first <= start);
    index > 0 && ranges[index - 1].1 >= end
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

    fn get_global_sink_state(
        &self,
        _ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalSinkState>> {
        self.mark(0);
        Ok(Arc::new(EmptyState))
    }

    fn get_local_sink_state(
        &self,
        _ctx: &ExecutionContext,
        _gstate: &dyn GlobalSinkState,
    ) -> Result<Box<dyn LocalSinkState>> {
        Ok(Box::new(NoState))
    }

    fn sink(
        &self,
        _ctx: &ExecutionContext,
        input: &DataChunk,
        _gstate: &dyn GlobalSinkState,
        lstate: &mut dyn LocalSinkState,
    ) -> Result<SinkResult> {
        downcast_state_mut::<NoState>(lstate.as_any_mut(), self.name())?;
        if let Some(collected) = &self.collected {
            let mut ranges = vec![];
            for column in input.columns() {
                buffer_ranges(&column.to_data(), &mut ranges);
            }
            let batch = input.compact()?;
            let mut collected = collected.lock();
            for range in ranges {
                if !contained(&collected.input, range) {
                    collected.written.push(range);
                }
            }
            collected.batches.push(batch);
        }
        Ok(SinkResult::NeedMoreInput)
    }

    fn finalize(
        &self,
        _ctx: &ExecutionContext,
        _gstate: &dyn GlobalSinkState,
    ) -> Result<SinkFinalize> {
        self.mark(1);
        Ok(SinkFinalize::Ready)
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
///
/// Every thread times its own passes: a pass of all threads takes from the
/// first start to the last end, so that a thread that the system preempts
/// makes it slower, never faster.
fn measure_bandwidth(
    threads: usize,
    total_bytes: usize,
    repetitions: usize,
) -> Bandwidth {
    let words = (total_bytes / 8 / threads).max(1);
    let barrier = Barrier::new(threads);
    // The start and the end of every pass of every kernel, of every thread.
    let passes = std::thread::scope(|scope| {
        let handles = (0..threads)
            .map(|thread| {
                let barrier = &barrier;
                scope.spawn(move || {
                    // Every thread touches its own buffers first.
                    let source = vec![thread as u64 + 1; words];
                    let mut target = vec![0x5a_u64; words];
                    let mut passes = Vec::with_capacity(3 * repetitions);
                    for _ in 0..repetitions {
                        for kernel in 0..3 {
                            barrier.wait();
                            let start = Instant::now();
                            match kernel {
                                0 => {
                                    let sum = source.iter().fold(0_u64, |sum, &value| {
                                        sum.wrapping_add(value)
                                    });
                                    black_box(sum);
                                }
                                1 => {
                                    target.fill(black_box(0x5a5a_5a5a_5a5a_5a5a));
                                    black_box(&target);
                                }
                                _ => {
                                    target.copy_from_slice(&source);
                                    black_box(&target);
                                }
                            }
                            passes.push((start, Instant::now()));
                        }
                    }
                    passes
                })
            })
            .collect::<Vec<_>>();
        handles
            .into_iter()
            .map(|handle| handle.join().expect("the bandwidth thread panicked"))
            .collect::<Vec<_>>()
    });
    let mut best = [Duration::MAX; 3];
    for pass in 0..3 * repetitions {
        let start = passes.iter().map(|passes| passes[pass].0).min();
        let end = passes.iter().map(|passes| passes[pass].1).max();
        if let (Some(start), Some(end)) = (start, end) {
            let kernel = pass % 3;
            best[kernel] = best[kernel].min(end.duration_since(start));
        }
    }
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

impl Scope {
    /// The window of the counts of the operator in this scope.
    fn window(self) -> Window {
        match self {
            Self::All => Window::Execution,
            Self::Sink => Window::BeforeRoot,
            Self::Operator => Window::Root,
        }
    }
}

/// A window of the counts of an execution, as its [`DiscardSink`] marks
/// them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Window {
    /// The execution until its root meta pipeline is finalized.
    Execution,
    /// The execution until the root pipeline starts: the pipelines of the
    /// child meta pipelines (the build side of a hash join) and their
    /// finalizations.
    BeforeRoot,
    /// The root pipeline (whose sink is the `DiscardSink`, e.g. the probe
    /// pipeline of a hash join), until it is finalized.
    Root,
}

/// The counts of an execution in the windows that its [`DiscardSink`]
/// marked.
#[derive(Debug, Clone, Copy, Default)]
struct Windows {
    before_root: Counts,
    root: Counts,
}

impl Windows {
    fn of(self, window: Window) -> Counts {
        match window {
            Window::Execution => self.before_root.add(self.root),
            Window::BeforeRoot => self.before_root,
            Window::Root => self.root,
        }
    }
}

/// A plan whose counts in `window` are those of the harness of a
/// configuration: of its sources into a `DiscardSink`, counted in the window
/// that covers the same pipelines of the harness as the window of the
/// configuration covers (e.g. the whole execution of the build side for the
/// build of a hash join, which is counted until the probe pipeline starts).
struct Baseline {
    sink: Arc<DiscardSink>,
    window: Window,
}

impl Baseline {
    fn of(source: &Arc<ChunkSource>, window: Window) -> Self {
        Self {
            sink: DiscardSink::new(Arc::clone(source) as _),
            window,
        }
    }
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
        let child = Arc::clone(&operator);
        let sink = if collect {
            DiscardSink::collecting(child, &sources)
        } else {
            DiscardSink::new(child)
        };
        Self {
            sink,
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
    /// The plans whose counts are those of the harness, which are subtracted
    /// from the counts of the configuration.
    baselines: Vec<Baseline>,
    /// The task context of the executions, if not the default one.
    task_ctx: Option<Arc<TaskContext>>,
}

/// One execution of a plan.
struct Run {
    /// The profile, if the execution was profiled.
    profile: Option<Arc<QueryProfile>>,
    /// The counts, if the session counted.
    windows: Windows,
}

/// The measurements of one timed execution of a configuration.
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
    /// The self time as a sink, finalizing and as a source.
    sink_time: u64,
    finalize_time: u64,
    source_time: u64,
}

impl Sample {
    /// The sample of the operator of `plan` in `scope` in `profile`, the
    /// profile of an execution of `plan`.
    fn of(profile: &QueryProfile, plan: &Plan, scope: Scope) -> Result<Self> {
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

/// What a configuration allocates, copies and writes: the counts of an
/// execution (without profiling, whose allocations would count too) in the
/// window of its scope, net of the counts of its baselines, and the bytes of
/// its output that the operator wrote, if it produces the output of the
/// execution.
#[derive(Debug, Clone, Copy, Default)]
struct Footprint {
    counts: Counts,
    written: Option<u64>,
}

/// The results of a configuration: the medians of its samples, and its
/// footprint.
#[derive(Debug, Clone)]
struct Measurement {
    operator: &'static str,
    config: String,
    rows: u64,
    ns_per_row: f64,
    rows_per_second: f64,
    allocations_per_chunk: f64,
    allocated_per_row: f64,
    /// The bytes copied by `memcpy` and `memmove` and moved by `realloc`
    /// per row, `None` where the copies are not counted.
    copied_per_row: Option<f64>,
    /// The bytes of the output that the operator wrote per row, `None` if
    /// the operator does not produce the output of the execution.
    written_per_row: Option<f64>,
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
    fn new(
        case: &Case,
        samples: &[Sample],
        footprint: Footprint,
        floor: Floor,
        bandwidth: &Bandwidth,
    ) -> Self {
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
        let counts = footprint.counts;
        Self {
            operator: case.operator,
            config: case.config.clone(),
            rows,
            ns_per_row: median(&mut ns_per_row.clone()),
            rows_per_second: row_count / wall_time.max(1.0) * 1e9,
            allocations_per_chunk: counts.allocations as f64
                / samples[0].chunks.max(1) as f64,
            allocated_per_row: counts.allocated as f64 / row_count,
            copied_per_row: counters::COUNTS_COPIES
                .then(|| (counts.copied + counts.moved) as f64 / row_count),
            written_per_row: footprint.written.map(|written| written as f64 / row_count),
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

/// The benchmark: its configuration, counters, runtime, bandwidths and
/// results.
struct Bench {
    config: BenchConfig,
    /// The counters of the threads of the benchmark: of its runtime and of
    /// the thread that created it, which runs the executions.
    session: Session,
    runtime: tokio::runtime::Runtime,
    task_ctx: Arc<TaskContext>,
    bandwidths: Vec<Bandwidth>,
    measurements: Vec<Measurement>,
}

impl Bench {
    fn try_new(config: BenchConfig) -> Result<Self> {
        let session = Session::new();
        session.tag_current_thread();
        let max_threads = config.threads.iter().copied().max().unwrap_or(1);
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(max_threads)
            .enable_all()
            .on_thread_start(move || session.tag_current_thread())
            .build()?;
        let session_config = SessionConfig::new().with_target_partitions(max_threads);
        let task_ctx = SessionContext::new_with_config(session_config).task_ctx();
        Ok(Self {
            config,
            session,
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

    /// Execute the plan below `sink` on `threads` threads, profiled or not.
    fn execute(
        &self,
        sink: &Arc<DiscardSink>,
        threads: usize,
        task_ctx: Option<&Arc<TaskContext>>,
        profiling: bool,
    ) -> Result<Run> {
        self.session.tag_current_thread();
        let task_ctx = Arc::clone(task_ctx.unwrap_or(&self.task_ctx));
        let executor = Executor::try_new(Arc::clone(sink) as _, task_ctx, threads)?
            .with_profiling(profiling);
        let start = self.session.snapshot();
        self.runtime.block_on(executor.execute())?;
        Ok(Run {
            profile: executor.profile(),
            windows: sink.windows(start)?,
        })
    }

    /// The counts of `plan` in the window of the scope of `case`, net of the
    /// counts of the baselines of `case` in theirs, in executions without
    /// profiling.
    fn count(&self, case: &Case, plan: &Plan) -> Result<Counts> {
        let task_ctx = case.task_ctx.as_ref();
        for baseline in &case.baselines {
            self.execute(&baseline.sink, case.threads, task_ctx, false)?;
        }
        self.session.set_enabled(true);
        let counts = (|| {
            let mut harness = Counts::default();
            for baseline in &case.baselines {
                let run = self.execute(&baseline.sink, case.threads, task_ctx, false)?;
                harness = harness.add(run.windows.of(baseline.window));
            }
            let run = self.execute(&plan.sink, case.threads, task_ctx, false)?;
            Ok(run.windows.of(case.scope.window()).saturating_sub(harness))
        })();
        self.session.set_enabled(false);
        counts
    }

    /// Measure `case`: a warm-up execution of `plan` (with `collect`) whose
    /// output `check` checks and whose output rows give the `floor`, the
    /// counted executions of `plan` and of the baselines, and the timed
    /// executions.
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
        self.execute(&warm_up.sink, case.threads, task_ctx, true)
            .map_err(context)?;
        let (output, written) = warm_up.sink.take_collected();
        check(&output).map_err(context)?;
        let floor = floor(output.iter().map(|batch| batch.num_rows() as u64).sum());
        drop(output);
        let footprint = Footprint {
            counts: self.count(case, &plan(false)?).map_err(context)?,
            // The output of the probe pipeline is not the build's.
            written: (case.scope != Scope::Sink).then_some(written),
        };
        let mut samples = vec![];
        for _ in 0..self.config.iterations {
            let plan = plan(false)?;
            let run = self
                .execute(&plan.sink, case.threads, task_ctx, true)
                .map_err(context)?;
            let profile = run.profile.ok_or_else(|| {
                internal_datafusion_err!("The execution has no profile")
            })?;
            samples.push(Sample::of(&profile, &plan, case.scope)?);
        }
        self.record(case, &samples, footprint, floor);
        Ok(())
    }

    /// Record the results of a configuration.
    fn record(
        &mut self,
        case: &Case,
        samples: &[Sample],
        footprint: Footprint,
        floor: Floor,
    ) {
        let measurement = Measurement::new(
            case,
            samples,
            footprint,
            floor,
            &self.bandwidth(case.threads),
        );
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
            baselines: vec![Baseline::of(&source, Window::Root)],
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
            baselines: vec![Baseline::of(&source, Window::Root)],
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
            // The build is counted until the probe pipeline starts: the
            // harness of the build side is all of its execution into a
            // `DiscardSink`.
            baselines: vec![Baseline::of(&build.source, Window::Execution)],
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
        let case = Case {
            operator: "hash_join_probe",
            config: params.label(),
            threads: params.threads,
            scope: Scope::Operator,
            // The probe is counted in the probe pipeline (the root pipeline),
            // without the build.
            baselines: vec![Baseline::of(&probe.source, Window::Root)],
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
            baselines: vec![Baseline::of(&source, Window::Execution)],
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
            baselines: vec![Baseline::of(&source, Window::Execution)],
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
            baselines: vec![Baseline::of(&source, Window::Execution)],
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
            baselines: vec![Baseline::of(&source, Window::Execution)],
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
            let (sample, _) = append_tuple_data(
                bench.session,
                &layout,
                &chunks,
                &hashes,
                params,
                true,
            )?;
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
        // The counts of the ingestion, net of those of the threads (which
        // the harness spawns and joins).
        bench.session.set_enabled(true);
        let counted =
            append_tuple_data(bench.session, &layout, &chunks, &hashes, params, true);
        let harness =
            append_tuple_data(bench.session, &layout, &chunks, &hashes, params, false);
        bench.session.set_enabled(false);
        let footprint = Footprint {
            counts: counted?.1.saturating_sub(harness?.1),
            written: None,
        };
        // The ingestion reads (the lines of) the values of the rows and
        // writes them once.
        let floor = Floor {
            read: chunks.iter().map(read_line_bytes).sum(),
            write: bytes_in as f64,
        };
        bench.record(&case, &samples, footprint, floor);
    }
    Ok(())
}

/// Append `chunks` (with the hashes `hashes`) to `PartitionedTupleData`s of
/// `layout` on `params.threads` threads (if `append`, otherwise only spawn
/// and join the threads), with the counts of `session` in the meantime.
fn append_tuple_data(
    session: Session,
    layout: &Arc<TupleDataLayout>,
    chunks: &[DataChunk],
    hashes: &[Vec<u64>],
    params: TupleParams,
    append: bool,
) -> Result<(Sample, Counts)> {
    let threads = params.threads;
    session.tag_current_thread();
    let before = session.snapshot();
    let start = Instant::now();
    let results = std::thread::scope(|scope| {
        let handles = (0..threads)
            .map(|thread| {
                scope.spawn(move || -> Result<(u64, u64, u64)> {
                    session.tag_current_thread();
                    if !append {
                        return Ok((0, 0, 0));
                    }
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
    let counts = session.snapshot().since(before);
    let mut sample = Sample {
        wall_time,
        ..Sample::default()
    };
    for (elapsed, rows, memory) in results {
        sample.self_time += elapsed;
        sample.rows += rows;
        sample.memory_peak += memory;
        sample.bytes_out += memory;
    }
    Ok((sample, counts))
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
        // The harness of the scan: an execution of a pipeline of a source
        // without chunks into a `DiscardSink`.
        let idle = ChunkSource::new(Arc::clone(&schema), vec![]);
        let case = Case {
            operator: "parquet_decode",
            config: params.label(rows),
            threads: params.threads,
            scope: Scope::All,
            baselines: vec![Baseline::of(&idle, Window::Execution)],
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
            short(self.allocated_per_row),
            self.copied_per_row.map_or_else(|| "-".to_owned(), short),
            self.written_per_row.map_or_else(|| "-".to_owned(), short),
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

const HEADERS: [&str; 16] = [
    "operator",
    "configuration",
    "rows",
    "ns/row",
    "Mrows/s",
    "allocs/chunk",
    "alloc B/row",
    "copied B/row",
    "written B/row",
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
        if counters::COUNTS_COPIES {
            ""
        } else {
            ", copies not counted"
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

/// The tests of this binary (the benchmark and the tests of its harness)
/// run one after the other, so that they do not disturb each other's
/// timings.
static SERIAL: Mutex<()> = Mutex::new(());

/// The benchmark, see the module documentation.
#[test]
fn operator_micro() -> Result<()> {
    let _serial = SERIAL.lock();
    let config = BenchConfig::from_env();
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

/// Tests of the harness: of its counts of allocations and copies, of the
/// windows and baselines that scope them, and of the bytes written.
mod harness_tests {
    use datafusion_physical_pipeline::{
        GlobalOperatorState, OperatorResult, OperatorState,
    };

    use super::*;

    /// A benchmark of inputs of 1% of the sizes, on 1 and 4 threads, with one
    /// timed execution per configuration.
    fn small_bench() -> Bench {
        Bench::try_new(BenchConfig {
            scale: 0.01,
            iterations: 1,
            threads: vec![1, 4],
            filter: vec![],
            save: None,
            compare: None,
            bandwidth_bytes: 1 << 20,
        })
        .expect("a benchmark")
    }

    /// The measurement of `bench` whose `operator configuration` starts
    /// with `name`.
    fn measurement<'a>(bench: &'a Bench, name: &str) -> &'a Measurement {
        bench
            .measurements
            .iter()
            .find(|measurement| measurement.name().starts_with(name))
            .unwrap_or_else(|| panic!("no measurement of {name}"))
    }

    #[test]
    fn copies_into_reused_buffers_are_counted() {
        let _serial = SERIAL.lock();
        let session = Session::new();
        session.tag_current_thread();
        let source = vec![1_u8; 10_000];
        let mut target = vec![0_u8; 10_000];
        session.set_enabled(true);
        let before = session.snapshot();
        for _ in 0..3 {
            let len = black_box(source.len());
            target[..len].copy_from_slice(&source[..len]);
            black_box(&target);
        }
        let counts = session.snapshot().since(before);
        session.set_enabled(false);
        // The copies into `target` allocate nothing, but they are copies.
        assert_eq!(counts.allocations, 0);
        let expected = if counters::COUNTS_COPIES { 30_000 } else { 0 };
        assert_eq!(counts.copied, expected);
    }

    /// A streaming operator that, for every chunk, allocates `allocations`
    /// buffers of `bytes` bytes, copies the bytes of the source of its thread
    /// into its target (buffers that it reuses), and produces no rows.
    struct Allocating {
        children: Vec<Arc<dyn PhysicalOperator>>,
        allocations: usize,
        bytes: usize,
        /// The source and the target of the copies of every thread.
        buffers: Vec<Mutex<(Vec<u8>, Vec<u8>)>>,
        state: Arc<dyn GlobalOperatorState>,
    }

    impl std::fmt::Debug for Allocating {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Allocating")
                .field("allocations", &self.allocations)
                .field("bytes", &self.bytes)
                .finish()
        }
    }

    impl PhysicalOperator for Allocating {
        fn name(&self) -> &str {
            "Allocating"
        }

        fn schema(&self) -> SchemaRef {
            self.children[0].schema()
        }

        fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
            &self.children
        }

        fn get_global_operator_state(
            &self,
            _ctx: &ExecutionContext,
        ) -> Result<Arc<dyn GlobalOperatorState>> {
            Ok(Arc::clone(&self.state))
        }

        fn execute(
            &self,
            ctx: &ExecutionContext,
            _input: &DataChunk,
            _output: &mut DataChunk,
            _gstate: &dyn GlobalOperatorState,
            _lstate: &mut dyn OperatorState,
        ) -> Result<OperatorResult> {
            for _ in 0..self.allocations {
                black_box(Vec::<u8>::with_capacity(self.bytes));
            }
            let mut buffers = self.buffers[ctx.thread_id].lock();
            let (source, target) = &mut *buffers;
            target.copy_from_slice(black_box(source.as_slice()));
            Ok(OperatorResult::NeedMoreInput)
        }
    }

    #[test]
    fn the_harness_counts_what_an_operator_allocates_and_copies() {
        let _serial = SERIAL.lock();
        let bench = small_bench();
        let schema = schema_of(vec![Field::new("a", DataType::Int64, false)]);
        let column = uniform_i64(&mut Rng::new(11), 64 * DEFAULT_VECTOR_SIZE, 100);
        let chunks = chunks_of(&schema, &[column], DEFAULT_VECTOR_SIZE).expect("chunks");
        let source = ChunkSource::new(schema, chunks);
        let chunks = source.chunks.len() as u64;
        for threads in [1, 4] {
            let count = |allocations: usize, bytes: usize, copied: usize| {
                let operator = Arc::new(Allocating {
                    children: vec![Arc::clone(&source) as _],
                    allocations,
                    bytes,
                    buffers: (0..threads)
                        .map(|_| Mutex::new((vec![1; copied], vec![0; copied])))
                        .collect(),
                    state: Arc::new(EmptyState),
                });
                let case = Case {
                    operator: "allocating",
                    config: String::new(),
                    threads,
                    scope: Scope::Operator,
                    baselines: vec![Baseline::of(&source, Window::Root)],
                    task_ctx: None,
                };
                let plan = Plan::new(operator, vec![Arc::clone(&source) as _], false);
                bench.count(&case, &plan).expect("counts")
            };
            let idle = count(0, 0, 0);
            let busy = count(3, 1000, 4096);
            // What the operator allocates and copies for every chunk, exactly.
            assert_eq!(busy.allocations - idle.allocations, 3 * chunks);
            assert_eq!(busy.allocated - idle.allocated, 3000 * chunks);
            if counters::COUNTS_COPIES {
                assert_eq!(busy.copied - idle.copied, 4096 * chunks);
            }
            // The baseline subtracts the harness (e.g. the copies of the
            // chunks the source hands out, an allocation per chunk), and no
            // more: what is left without an allocation of the operator is
            // what the executor allocates for its stage in the pipeline.
            assert!(
                idle.allocations < chunks / 4,
                "{idle:?} for an operator that allocates nothing, on {threads} threads"
            );
        }
    }

    #[test]
    fn builds_count_their_allocations_without_those_of_the_probe() {
        let _serial = SERIAL.lock();
        let bench = small_bench();
        // 8192 build rows of 128 keys: a probe chunk whose rows all find a
        // match has 131,072 output rows.
        let rows = 4 * DEFAULT_VECTOR_SIZE;
        let build = BuildSide::new(rows, 64, KeyKind::I64, DEFAULT_VECTOR_SIZE)
            .expect("a build side");
        let count = |hit: f64| {
            let probe = ProbeSide::new(
                DEFAULT_VECTOR_SIZE,
                hit,
                &build,
                KeyKind::I64,
                DEFAULT_VECTOR_SIZE,
            )
            .expect("a probe side");
            let case = Case {
                operator: "hash_join_build",
                config: String::new(),
                threads: 1,
                scope: Scope::Sink,
                baselines: vec![Baseline::of(&build.source, Window::Execution)],
                task_ctx: None,
            };
            let join =
                hash_join(&build.source, &probe.source, KeyKind::I64).expect("a join");
            let sources = vec![
                Arc::clone(&build.source) as _,
                Arc::clone(&probe.source) as _,
            ];
            bench
                .count(&case, &Plan::new(join, sources, false))
                .expect("counts")
        };
        let with_matches = count(1.0);
        let without_matches = count(0.0);
        // The output of the probe is not part of the counts of the build.
        // (Whether a `realloc` moves its block depends on the heap.)
        assert_eq!(with_matches.allocations, without_matches.allocations);
        assert_eq!(with_matches.allocated, without_matches.allocated);
        assert_eq!(with_matches.copied, without_matches.copied);
        // The build stores its rows (keys, payload, hashes) in blocks that it
        // allocates: the baseline does not subtract them.
        assert!(
            with_matches.allocated >= rows as u64 * 24,
            "{with_matches:?} for a build of {rows} rows"
        );
    }

    #[test]
    fn the_written_bytes_are_those_of_the_buffers_that_the_operator_wrote() {
        let _serial = SERIAL.lock();
        let mut bench = small_bench();
        let sort_rows = bench.rows(2 << 20);
        let sort_label = SortParams {
            order: Order::Random,
            key: KeyKind::I64,
            distinct: 0,
            batch: DEFAULT_VECTOR_SIZE,
            threads: 1,
        }
        .label(sort_rows);
        bench.config.filter = vec![
            "projection exprs=c,a sel=1.00 batch=2048 threads=1 ".to_owned(),
            "projection exprs=a+b,c sel=1.00 batch=2048 threads=1 ".to_owned(),
            "filter pred=a<x sel=0.5 batch=2048 threads=1 ".to_owned(),
            format!("sort {sort_label} "),
        ];
        projection(&mut bench).expect("projections");
        filter(&mut bench).expect("a filter");
        sort(&mut bench).expect("a sort");
        let written = |name: &str| {
            measurement(&bench, name)
                .written_per_row
                .expect("bytes written")
        };
        // The projection of columns passes them through.
        assert_eq!(written("projection exprs=c,a "), 0.0);
        // `a + b` writes the sum (8 bytes per row) and passes `c` through.
        assert_eq!(written("projection exprs=a+b,c "), 8.0);
        // The filter selects the rows of its input.
        assert_eq!(written("filter pred=a<x "), 0.0);
        // The sort writes both of its columns, sorted.
        let sort = written("sort ");
        assert!(
            (sort - 16.0).abs() < 1e-9,
            "the sort wrote {sort} bytes per row"
        );
    }

    #[test]
    fn probes_count_the_copies_of_the_hashes_of_their_rows() {
        let _serial = SERIAL.lock();
        let mut bench = small_bench();
        bench.config.filter = vec!["hash_join_probe hit=0.00 ".to_owned()];
        hash_join_probe(&mut bench).expect("a probe");
        let probe = measurement(&bench, "hash_join_probe hit=0.00 ");
        // No probe row finds a match: the probe writes no output.
        assert_eq!(probe.written_per_row, Some(0.0));
        if counters::COUNTS_COPIES {
            // But `ChainScan::reset` copies the hash of every probe row (8
            // bytes) into a buffer that it reuses for every chunk.
            let copied = probe.copied_per_row.expect("copies");
            assert!(copied >= 8.0, "the probe copied {copied} bytes per row");
        }
    }
}
