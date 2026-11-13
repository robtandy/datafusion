// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Per-operator profiling of the pipeline engine.
//!
//! In a session whose [`ProfilingOptions`] are enabled
//! (`SET pipeline_profiling.enabled = true`), every execution of the
//! [`Executor`](crate::executor::Executor) records a [`QueryProfile`]:
//!
//! * per operator ([`OperatorProfile`]): its self time, the time the threads
//!   were blocked on it, the morsels, rows and bytes it consumed and produced,
//!   and the high-water mark of the memory it reserved in the memory pool;
//! * per pipeline ([`PipelineProfile`]) and per thread of a pipeline
//!   ([`WorkerProfile`]): the morsels its source distributed, the time its
//!   threads waited in the work queue of the runtime, and their active and
//!   idle time;
//! * per execution ([`ExecutorProfile`]): the same, over all pipelines.
//!
//! `EXPLAIN ANALYZE` shows what fits on its lines (see
//! [`explain`](crate::explain)), and [`QueryProfile::to_json`] dumps all of
//! it: the operator tree, the pipelines it decomposes into and every metric
//! with its unit, in the schema that `docs/profiling.md` documents, which is
//! a contract for the tools that read it.
//!
//! Profiling is off by default. The executor then records only the
//! [`OperatorMetrics`](crate::operator::OperatorMetrics) of every operator,
//! and checks once per call whether it profiles.
//!
//! # Time
//!
//! Every thread of a pipeline (a *worker*) is, at any time, in one of three
//! states, whose times add up to its elapsed time (from its spawn to its
//! completion):
//!
//! * *active*: in a poll of its task, calling the operators of the pipeline
//!   (their *self time*) or running the push loop between the calls (the
//!   *overhead* of the executor);
//! * *blocked*: suspended because a call returned a `Blocked` result, until
//!   its task is polled again (the *blocked time* of the operator);
//! * *queued*: ready to run, but waiting for a thread of the runtime: from
//!   its spawn to its first poll, and after it yielded (the *work-queue wait*).
//!
//! The executor creates the global states of a pipeline before its workers
//! start, and finalizes the sinks of meta pipelines, on the task of the
//! execution: these calls are active time too. So the self times and the
//! blocked times of all operators, the work-queue wait and the overhead add
//! up to the *thread time* of the execution, the wall time of all of its
//! threads ([`ExecutorProfile::thread_time`]); on one thread, with
//! pipelines that run one after the other, to its elapsed time.

use std::cell::RefCell;
use std::collections::HashMap;
use std::fmt::{self, Display};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use arrow::array::{Array, ArrayRef, AsArray, GenericListArray, OffsetSizeTrait};
use arrow::datatypes::DataType;
use datafusion_common::config::ConfigExtension;
use datafusion_common::instant::Instant;
use datafusion_common::{Result, extensions_options};
use datafusion_execution::TaskContext;
use datafusion_execution::memory_pool::{
    MemoryConsumer, MemoryLimit, MemoryPool, MemoryReservation,
};
use datafusion_execution::runtime_env::RuntimeEnv;
use parking_lot::Mutex;
use serde_json::{Map, Value, json};

use crate::chunk::DataChunk;
use crate::operator::{PhysicalOperator, operator_key};
use crate::pipeline::{MetaPipelineId, Pipeline, PipelineGraph, PipelineId};

/// The value of `schema` in the JSON dump of a [`QueryProfile`].
pub const PROFILE_SCHEMA: &str = "datafusion-physical-pipeline/profile";

/// The version of the schema of the JSON dump of a [`QueryProfile`]
/// (`version`), incremented whenever a field or a metric changes its meaning
/// or is removed. Adding fields or metrics does not change it.
pub const PROFILE_VERSION: u64 = 1;

extensions_options! {
    /// The session options of the profiler of the pipeline engine: whether
    /// the executions of the session are profiled (see the
    /// [module documentation](self) and `docs/profiling.md`).
    ///
    /// ```
    /// use datafusion::prelude::{SessionConfig, SessionContext};
    /// use datafusion_physical_pipeline::profiling::ProfilingOptions;
    ///
    /// // Profiles every execution of the session.
    /// let config = SessionConfig::new().with_option_extension(ProfilingOptions::new(true));
    /// let profiled = SessionContext::new_with_config(config);
    ///
    /// // Profiles the executions once `SET pipeline_profiling.enabled = true`
    /// // switched profiling on.
    /// let config = SessionConfig::new().with_option_extension(ProfilingOptions::default());
    /// let session = SessionContext::new_with_config(config);
    /// ```
    pub struct ProfilingOptions {
        /// Profile the executions of the pipeline engine: record the self
        /// time, the blocked time, the morsels, rows and bytes in and out, and
        /// the peak memory of every operator, and the morsels, the work-queue
        /// wait and the active and idle time of every pipeline and of the
        /// execution, which `EXPLAIN ANALYZE` shows and `QueryProfile::to_json`
        /// dumps. Off by default.
        pub enabled: bool, default = false
    }
}

impl ConfigExtension for ProfilingOptions {
    const PREFIX: &'static str = "pipeline_profiling";
}

impl ProfilingOptions {
    /// Options that profile the executions of a session if `enabled`.
    pub fn new(enabled: bool) -> Self {
        Self { enabled }
    }

    /// Whether the executions with the task context `task_ctx` are profiled:
    /// whether its session registers [`ProfilingOptions`] that are enabled.
    pub fn enabled_for(task_ctx: &TaskContext) -> bool {
        task_ctx
            .session_config()
            .options()
            .extensions
            .get::<Self>()
            .is_some_and(|options| options.enabled)
    }
}

// -------------------------------------------------------------------------
// Metrics
// -------------------------------------------------------------------------

/// The unit of a metric of a [`QueryProfile`] (`unit` in the JSON dump).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MetricUnit {
    /// `ns`: durations, and points in time, as the time since the execution
    /// started.
    Nanoseconds,
    /// `rows`.
    Rows,
    /// `morsels`: [`DataChunk`]s, of at most
    /// [`DEFAULT_VECTOR_SIZE`](crate::chunk::DEFAULT_VECTOR_SIZE) rows.
    Morsels,
    /// `bytes`, see [`chunk_bytes`].
    Bytes,
    /// `count`: a number of things, e.g. of threads or of calls.
    Count,
}

impl MetricUnit {
    /// The name of the unit in the JSON dump.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Nanoseconds => "ns",
            Self::Rows => "rows",
            Self::Morsels => "morsels",
            Self::Bytes => "bytes",
            Self::Count => "count",
        }
    }
}

impl Display for MetricUnit {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A metric of a [`QueryProfile`]: its name, its unit and its value.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Metric {
    /// The name of the metric, e.g. `self_time`.
    pub name: &'static str,
    /// The unit of its value.
    pub unit: MetricUnit,
    /// Its value.
    pub value: u64,
}

impl Metric {
    fn new(name: &'static str, unit: MetricUnit, value: u64) -> Self {
        Self { name, unit, value }
    }
}

/// The value of the metric `name` of `metrics`.
fn find(metrics: &[Metric], name: &str) -> Option<u64> {
    metrics
        .iter()
        .find(|metric| metric.name == name)
        .map(|metric| metric.value)
}

/// The metrics of an operator, summed over the threads of a pipeline (a
/// [`StageProfile`]), or over all threads of all pipelines (an
/// [`OperatorProfile`]).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
#[non_exhaustive]
pub struct OperatorCounters {
    /// `self_time` (ns): the time spent in the calls of the operator:
    /// `get_data`, `execute`, `final_execute`, `sink`, `combine`, `finalize`,
    /// the creation of its states and the polls of the futures it waits for.
    pub self_time: u64,
    /// `blocked_time` (ns): the time that the threads were suspended
    /// because a call of the operator returned `Blocked`, until they were
    /// polled again.
    pub blocked_time: u64,
    /// `calls` (count): the calls of `get_data`, `execute`, `final_execute`
    /// and `sink`.
    pub calls: u64,
    /// `rows_in` (rows): the rows of the chunks the operator consumed, as an
    /// operator (`execute`) and as a sink (`sink`).
    pub rows_in: u64,
    /// `morsels_in` (morsels): the chunks the operator consumed.
    pub morsels_in: u64,
    /// `bytes_in` (bytes): the bytes of the chunks the operator consumed
    /// (see [`chunk_bytes`]).
    pub bytes_in: u64,
    /// `rows_out` (rows): the rows of the chunks the operator produced, as a
    /// source (`get_data`) and as an operator (`execute`, `final_execute`).
    pub rows_out: u64,
    /// `morsels_out` (morsels): the non-empty chunks the operator produced.
    pub morsels_out: u64,
    /// `bytes_out` (bytes): the bytes of the chunks the operator produced.
    pub bytes_out: u64,
}

impl OperatorCounters {
    fn add(&mut self, other: &Self) {
        self.self_time += other.self_time;
        self.blocked_time += other.blocked_time;
        self.calls += other.calls;
        self.rows_in += other.rows_in;
        self.morsels_in += other.morsels_in;
        self.bytes_in += other.bytes_in;
        self.rows_out += other.rows_out;
        self.morsels_out += other.morsels_out;
        self.bytes_out += other.bytes_out;
    }

    /// The metrics, in the order of the JSON dump.
    pub fn metrics(&self) -> Vec<Metric> {
        use MetricUnit::*;
        vec![
            Metric::new("self_time", Nanoseconds, self.self_time),
            Metric::new("blocked_time", Nanoseconds, self.blocked_time),
            Metric::new("calls", Count, self.calls),
            Metric::new("rows_in", Rows, self.rows_in),
            Metric::new("morsels_in", Morsels, self.morsels_in),
            Metric::new("bytes_in", Bytes, self.bytes_in),
            Metric::new("rows_out", Rows, self.rows_out),
            Metric::new("morsels_out", Morsels, self.morsels_out),
            Metric::new("bytes_out", Bytes, self.bytes_out),
        ]
    }
}

// -------------------------------------------------------------------------
// Profiles
// -------------------------------------------------------------------------

/// The role of an operator in a pipeline.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperatorRole {
    /// `source`: the operator produces the chunks of the pipeline.
    Source,
    /// `operator`: the operator transforms the chunks of the pipeline.
    Operator,
    /// `sink`: the operator consumes the chunks of the pipeline.
    Sink,
}

impl OperatorRole {
    /// The name of the role in the JSON dump.
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Source => "source",
            Self::Operator => "operator",
            Self::Sink => "sink",
        }
    }
}

/// The profile of one execution of a [`PipelineGraph`] by the
/// [`Executor`](crate::executor::Executor), see the
/// [module documentation](self).
#[derive(Debug, Clone)]
pub struct QueryProfile {
    executor: ExecutorProfile,
    operators: Vec<OperatorProfile>,
    pipelines: Vec<PipelineProfile>,
    meta_pipelines: Vec<MetaPipelineProfile>,
    ids: OperatorIds,
}

/// The ids of the operators of a profile, by [`operator_key`]. Keeps the
/// operators alive, so that their keys identify them.
#[derive(Clone)]
struct OperatorIds {
    ids: HashMap<usize, usize>,
    _operators: Vec<Arc<dyn PhysicalOperator>>,
}

impl fmt::Debug for OperatorIds {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("OperatorIds").finish_non_exhaustive()
    }
}

impl QueryProfile {
    /// The metrics of the execution as a whole.
    pub fn executor(&self) -> &ExecutorProfile {
        &self.executor
    }

    /// All operators, by id: the operator tree below the sink of the root
    /// meta pipeline (e.g. the `CollectorSink` of the execution, the
    /// operator with id `0`), in pre-order, each listed once, then the
    /// operators of the pipelines that are not part of the tree (e.g. the
    /// `JoinHashTableBuild` source of a hash join, which builds its table).
    pub fn operators(&self) -> &[OperatorProfile] {
        &self.operators
    }

    /// The profile of `operator`, if it is part of the profiled execution.
    pub fn operator(&self, operator: &dyn PhysicalOperator) -> Option<&OperatorProfile> {
        self.ids
            .ids
            .get(&operator_key(operator))
            .map(|&id| &self.operators[id])
    }

    /// All pipelines, by [`PipelineId`].
    pub fn pipelines(&self) -> &[PipelineProfile] {
        &self.pipelines
    }

    /// All meta pipelines, by [`MetaPipelineId`].
    pub fn meta_pipelines(&self) -> &[MetaPipelineProfile] {
        &self.meta_pipelines
    }

    /// The profile as JSON, in the schema that `docs/profiling.md`
    /// documents.
    pub fn to_json(&self) -> Value {
        json!({
            "schema": PROFILE_SCHEMA,
            "version": PROFILE_VERSION,
            "root": 0,
            "executor": { "metrics": metrics_json(&self.executor.metrics()) },
            "operators": self.operators.iter().map(OperatorProfile::to_json).collect::<Vec<_>>(),
            "pipelines": self.pipelines.iter().map(PipelineProfile::to_json).collect::<Vec<_>>(),
            "meta_pipelines": self
                .meta_pipelines
                .iter()
                .map(MetaPipelineProfile::to_json)
                .collect::<Vec<_>>(),
        })
    }

    /// [`Self::to_json`], pretty-printed.
    pub fn to_json_string(&self) -> String {
        serde_json::to_string_pretty(&self.to_json()).unwrap_or_default()
    }
}

/// `{"<name>": {"value": <value>, "unit": "<unit>"}, ...}`.
fn metrics_json(metrics: &[Metric]) -> Value {
    let mut object = Map::new();
    for metric in metrics {
        object.insert(
            metric.name.to_string(),
            json!({ "value": metric.value, "unit": metric.unit.as_str() }),
        );
    }
    Value::Object(object)
}

/// The profile of one operator of an execution.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct OperatorProfile {
    /// The id of the operator: its index in [`QueryProfile::operators`].
    pub id: usize,
    /// The kind of the operator, the same for all operators of its type:
    /// [`PhysicalOperator::name`], e.g. `PhysicalHashJoin`.
    pub kind: String,
    /// The operator as `EXPLAIN` describes it, with its parameters:
    /// [`PhysicalOperator::description`].
    pub name: String,
    /// The ids of its inputs ([`PhysicalOperator::children`]), in order.
    pub inputs: Vec<usize>,
    /// The pipelines the operator takes part in, with its role in each.
    pub pipelines: Vec<(PipelineId, OperatorRole)>,
    /// Its metrics, over all threads of all of its pipelines, including the
    /// finalization of its sink.
    pub counters: OperatorCounters,
    /// `memory_peak` (bytes): the high-water mark of the memory reserved in
    /// the memory pool by the consumers that the operator registered (during
    /// its calls), all of its threads together.
    pub memory_peak: u64,
}

impl OperatorProfile {
    /// The metrics of the operator, in the order of the JSON dump.
    pub fn metrics(&self) -> Vec<Metric> {
        let mut metrics = self.counters.metrics();
        metrics.push(Metric::new(
            "memory_peak",
            MetricUnit::Bytes,
            self.memory_peak,
        ));
        metrics
    }

    /// The value of the metric `name`.
    pub fn metric(&self, name: &str) -> Option<u64> {
        find(&self.metrics(), name)
    }

    fn to_json(&self) -> Value {
        json!({
            "id": self.id,
            "kind": self.kind,
            "name": self.name,
            "inputs": self.inputs,
            "pipelines": self
                .pipelines
                .iter()
                .map(|(pipeline, role)| json!({ "pipeline": pipeline, "role": role.as_str() }))
                .collect::<Vec<_>>(),
            "metrics": metrics_json(&self.metrics()),
        })
    }
}

/// The profile of one pipeline of an execution.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct PipelineProfile {
    /// The id of the pipeline.
    pub id: PipelineId,
    /// The id of its meta pipeline.
    pub meta_pipeline: MetaPipelineId,
    /// The id of its source.
    pub source: usize,
    /// The ids of its operators, from the source to the sink.
    pub operators: Vec<usize>,
    /// The id of its sink.
    pub sink: usize,
    /// The pipelines of its meta pipeline that completed before it started.
    pub dependencies: Vec<PipelineId>,
    /// The meta pipelines that were finalized before it started.
    pub meta_dependencies: Vec<MetaPipelineId>,
    /// Whether it skipped its work (its sink or one of its operators had
    /// finished, or cannot produce output).
    pub skipped: bool,
    /// `threads` (count): its threads (workers).
    pub threads: u64,
    /// `morsels` (morsels): the chunks its source distributed to its threads.
    pub morsels: u64,
    /// `rows` (rows): the rows of these chunks.
    pub rows: u64,
    /// `started_at` (ns): when it started, since the execution started.
    pub started_at: u64,
    /// `finished_at` (ns): when its last thread completed.
    pub finished_at: u64,
    /// `elapsed_time` (ns): `finished_at - started_at`.
    pub elapsed_time: u64,
    /// `setup_time` (ns): the time it took to create its global states and
    /// to start its threads, on the task of the execution.
    pub setup_time: u64,
    /// `active_time` (ns): `setup_time` and the active time of its threads.
    pub active_time: u64,
    /// `blocked_time` (ns): the blocked time of its threads.
    pub blocked_time: u64,
    /// `queue_wait_time` (ns): the work-queue wait of its threads.
    pub queue_wait_time: u64,
    /// `idle_time` (ns): `threads * elapsed_time - active_time`, the time
    /// that its threads could have been active but were not: blocked,
    /// queued, not started yet, or done while other threads still ran.
    pub idle_time: u64,
    /// Its stages: the source, the operators and the sink, with the metrics
    /// of their calls in this pipeline.
    pub stages: Vec<StageProfile>,
    /// Its threads, by thread index.
    pub workers: Vec<WorkerProfile>,
}

impl PipelineProfile {
    /// The metrics of the pipeline, in the order of the JSON dump.
    pub fn metrics(&self) -> Vec<Metric> {
        use MetricUnit::*;
        vec![
            Metric::new("threads", Count, self.threads),
            Metric::new("morsels", Morsels, self.morsels),
            Metric::new("rows", Rows, self.rows),
            Metric::new("started_at", Nanoseconds, self.started_at),
            Metric::new("finished_at", Nanoseconds, self.finished_at),
            Metric::new("elapsed_time", Nanoseconds, self.elapsed_time),
            Metric::new("setup_time", Nanoseconds, self.setup_time),
            Metric::new("active_time", Nanoseconds, self.active_time),
            Metric::new("blocked_time", Nanoseconds, self.blocked_time),
            Metric::new("queue_wait_time", Nanoseconds, self.queue_wait_time),
            Metric::new("idle_time", Nanoseconds, self.idle_time),
        ]
    }

    /// The value of the metric `name`.
    pub fn metric(&self, name: &str) -> Option<u64> {
        find(&self.metrics(), name)
    }

    fn to_json(&self) -> Value {
        json!({
            "id": self.id,
            "meta_pipeline": self.meta_pipeline,
            "source": self.source,
            "operators": self.operators,
            "sink": self.sink,
            "dependencies": self.dependencies,
            "meta_dependencies": self.meta_dependencies,
            "skipped": self.skipped,
            "metrics": metrics_json(&self.metrics()),
            "stages": self.stages.iter().map(|stage| json!({
                "operator": stage.operator,
                "role": stage.role.as_str(),
                "metrics": metrics_json(&stage.counters.metrics()),
            })).collect::<Vec<_>>(),
            "workers": self.workers.iter().map(|worker| json!({
                "thread": worker.thread,
                "metrics": metrics_json(&worker.metrics()),
            })).collect::<Vec<_>>(),
        })
    }
}

/// An operator in a pipeline, with the metrics of its calls in the pipeline.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct StageProfile {
    /// The id of the operator.
    pub operator: usize,
    /// Its role in the pipeline.
    pub role: OperatorRole,
    /// Its metrics in the pipeline, over all threads of the pipeline.
    pub counters: OperatorCounters,
}

/// The profile of one thread of a pipeline (a worker, which runs a
/// [`PipelineExecutor`](crate::executor::PipelineExecutor)).
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct WorkerProfile {
    /// The index of the thread in the pipeline, in `0..threads`.
    pub thread: usize,
    /// `started_at` (ns): when the thread was spawned, since the execution
    /// started.
    pub started_at: u64,
    /// `finished_at` (ns): when it completed.
    pub finished_at: u64,
    /// `elapsed_time` (ns): `active_time + blocked_time + queue_wait_time`,
    /// its time from its spawn to its completion.
    pub elapsed_time: u64,
    /// `active_time` (ns): its time in polls of its task.
    pub active_time: u64,
    /// `blocked_time` (ns): its time suspended on a `Blocked` result.
    pub blocked_time: u64,
    /// `queue_wait_time` (ns): its time ready to run, waiting for a thread of
    /// the runtime.
    pub queue_wait_time: u64,
    /// `morsels` (morsels): the chunks it got from the source.
    pub morsels: u64,
    /// `rows` (rows): the rows of these chunks.
    pub rows: u64,
}

impl WorkerProfile {
    /// The metrics of the thread, in the order of the JSON dump.
    pub fn metrics(&self) -> Vec<Metric> {
        use MetricUnit::*;
        vec![
            Metric::new("started_at", Nanoseconds, self.started_at),
            Metric::new("finished_at", Nanoseconds, self.finished_at),
            Metric::new("elapsed_time", Nanoseconds, self.elapsed_time),
            Metric::new("active_time", Nanoseconds, self.active_time),
            Metric::new("blocked_time", Nanoseconds, self.blocked_time),
            Metric::new("queue_wait_time", Nanoseconds, self.queue_wait_time),
            Metric::new("morsels", Morsels, self.morsels),
            Metric::new("rows", Rows, self.rows),
        ]
    }

    /// The value of the metric `name`.
    pub fn metric(&self, name: &str) -> Option<u64> {
        find(&self.metrics(), name)
    }
}

/// The profile of one meta pipeline of an execution.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MetaPipelineProfile {
    /// The id of the meta pipeline.
    pub id: MetaPipelineId,
    /// The id of its sink.
    pub sink: usize,
    /// Its pipelines.
    pub pipelines: Vec<PipelineId>,
    /// Its child meta pipelines.
    pub children: Vec<MetaPipelineId>,
    /// Its parent, `None` for the root.
    pub parent: Option<MetaPipelineId>,
    /// `finalize_time` (ns): the time it took to finalize its sink (part of
    /// the `self_time` of the sink).
    pub finalize_time: u64,
    /// `finalized_at` (ns): when its sink was finalized, since the execution
    /// started.
    pub finalized_at: u64,
}

impl MetaPipelineProfile {
    /// The metrics of the meta pipeline, in the order of the JSON dump.
    pub fn metrics(&self) -> Vec<Metric> {
        vec![
            Metric::new("finalize_time", MetricUnit::Nanoseconds, self.finalize_time),
            Metric::new("finalized_at", MetricUnit::Nanoseconds, self.finalized_at),
        ]
    }

    fn to_json(&self) -> Value {
        json!({
            "id": self.id,
            "sink": self.sink,
            "pipelines": self.pipelines,
            "children": self.children,
            "parent": self.parent,
            "metrics": metrics_json(&self.metrics()),
        })
    }
}

/// The metrics of an execution as a whole.
#[derive(Debug, Clone, Default)]
#[non_exhaustive]
pub struct ExecutorProfile {
    /// `threads` (count): the maximum number of threads of a pipeline (the
    /// `num_threads` of the executor).
    pub threads: u64,
    /// `pipelines` (count): the pipelines of the execution.
    pub pipelines: u64,
    /// `workers` (count): the threads of all pipelines.
    pub workers: u64,
    /// `morsels` (morsels): the chunks that the sources of all pipelines
    /// distributed.
    pub morsels: u64,
    /// `elapsed_time` (ns): the wall time of the execution.
    pub elapsed_time: u64,
    /// `thread_time` (ns): the wall time of all threads of the execution:
    /// `active_time + blocked_time + queue_wait_time`.
    pub thread_time: u64,
    /// `active_time` (ns): the active time of all pipelines, and the time it
    /// took to finalize the meta pipelines.
    pub active_time: u64,
    /// `blocked_time` (ns): the blocked time of all threads.
    pub blocked_time: u64,
    /// `queue_wait_time` (ns): the work-queue wait of all threads.
    pub queue_wait_time: u64,
    /// `idle_time` (ns): `threads * elapsed_time - active_time`, the time
    /// that `threads` threads could have been active, but were not.
    pub idle_time: u64,
    /// `overhead_time` (ns): `active_time` minus the self times of all
    /// operators: the time of the executor itself (the push loop between
    /// the calls of the operators, the scheduling of the pipelines).
    pub overhead_time: u64,
    /// `memory_peak` (bytes): the high-water mark of the memory reserved in
    /// the memory pool during the execution.
    pub memory_peak: u64,
    /// `unattributed_memory_peak` (bytes): the high-water mark of the memory
    /// reserved by consumers that were registered outside of the calls of
    /// the operators of the execution (e.g. by tasks that a bridged plan
    /// spawned).
    pub unattributed_memory_peak: u64,
}

impl ExecutorProfile {
    /// The metrics of the execution, in the order of the JSON dump.
    pub fn metrics(&self) -> Vec<Metric> {
        use MetricUnit::*;
        vec![
            Metric::new("threads", Count, self.threads),
            Metric::new("pipelines", Count, self.pipelines),
            Metric::new("workers", Count, self.workers),
            Metric::new("morsels", Morsels, self.morsels),
            Metric::new("elapsed_time", Nanoseconds, self.elapsed_time),
            Metric::new("thread_time", Nanoseconds, self.thread_time),
            Metric::new("active_time", Nanoseconds, self.active_time),
            Metric::new("blocked_time", Nanoseconds, self.blocked_time),
            Metric::new("queue_wait_time", Nanoseconds, self.queue_wait_time),
            Metric::new("idle_time", Nanoseconds, self.idle_time),
            Metric::new("overhead_time", Nanoseconds, self.overhead_time),
            Metric::new("memory_peak", Bytes, self.memory_peak),
            Metric::new(
                "unattributed_memory_peak",
                Bytes,
                self.unattributed_memory_peak,
            ),
        ]
    }

    /// The value of the metric `name`.
    pub fn metric(&self, name: &str) -> Option<u64> {
        find(&self.metrics(), name)
    }
}

// -------------------------------------------------------------------------
// Recording
// -------------------------------------------------------------------------

/// Distinguishes the profilers of concurrent and nested executions, see
/// [`OperatorScope`].
static NEXT_PROFILER_ID: AtomicU64 = AtomicU64::new(0);

fn nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

/// Records the profile of one execution (one call of
/// [`Executor::execute`](crate::executor::Executor::execute)).
pub(crate) struct ExecutionProfiler {
    id: u64,
    graph: Arc<PipelineGraph>,
    num_threads: usize,
    /// All operators of the execution, by id (see [`QueryProfile::operators`]).
    operators: Vec<Arc<dyn PhysicalOperator>>,
    /// The ids of the inputs of every operator.
    inputs: Vec<Vec<usize>>,
    /// The ids of the operators, by [`operator_key`].
    ids: HashMap<usize, usize>,
    started: Instant,
    pipelines: Vec<Mutex<PipelineRecord>>,
    finalizations: Vec<Mutex<FinalizeRecord>>,
    memory: Arc<MemoryRecorder>,
    finished: Mutex<Option<Instant>>,
    profile: OnceLock<Arc<QueryProfile>>,
}

#[derive(Default)]
struct PipelineRecord {
    started: Option<Instant>,
    finished: Option<Instant>,
    setup: u64,
    threads: usize,
    skipped: bool,
    /// The counters of the stages, summed over the threads.
    stages: Vec<OperatorCounters>,
    workers: Vec<WorkerRecord>,
}

struct WorkerRecord {
    thread: usize,
    spawned: Instant,
    finished: Instant,
    active: u64,
    blocked: u64,
    queue_wait: u64,
    morsels: u64,
    rows: u64,
}

#[derive(Default)]
struct FinalizeRecord {
    finished: Option<Instant>,
    time: u64,
}

impl fmt::Debug for ExecutionProfiler {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExecutionProfiler")
            .field("id", &self.id)
            .field("operators", &self.operators.len())
            .field("pipelines", &self.pipelines.len())
            .finish_non_exhaustive()
    }
}

impl ExecutionProfiler {
    /// Start profiling an execution of `graph` on up to `num_threads`
    /// threads per pipeline.
    pub(crate) fn new(graph: Arc<PipelineGraph>, num_threads: usize) -> Arc<Self> {
        let started = Instant::now();
        let id = NEXT_PROFILER_ID.fetch_add(1, Ordering::Relaxed);
        let mut tree = OperatorTree::default();
        tree.visit(graph.root().sink());
        // Operators of the pipelines that are not part of the tree, if any.
        for operator in graph.operators() {
            tree.visit(&operator);
        }
        let pipelines = graph
            .pipelines()
            .iter()
            .map(|pipeline| {
                Mutex::new(PipelineRecord {
                    stages: vec![
                        OperatorCounters::default();
                        pipeline.operators().len() + 2
                    ],
                    ..Default::default()
                })
            })
            .collect();
        let finalizations = graph
            .meta_pipelines()
            .iter()
            .map(|_| Mutex::default())
            .collect();
        let memory = Arc::new(MemoryRecorder::new(id, tree.operators.len()));
        Arc::new(Self {
            id,
            graph,
            num_threads,
            operators: tree.operators,
            inputs: tree.inputs,
            ids: tree.ids,
            started,
            pipelines,
            finalizations,
            memory,
            finished: Mutex::new(None),
            profile: OnceLock::new(),
        })
    }

    /// A task context like `task_ctx`, whose memory pool records the memory
    /// that the operators reserve.
    pub(crate) fn instrument(&self, task_ctx: &Arc<TaskContext>) -> Arc<TaskContext> {
        let pool: Arc<dyn MemoryPool> = Arc::new(ProfilingMemoryPool {
            inner: Arc::clone(task_ctx.memory_pool()),
            recorder: Arc::clone(&self.memory),
        });
        let mut runtime = RuntimeEnv::clone(&task_ctx.runtime_env());
        runtime.memory_pool = pool;
        Arc::new(TaskContext::new(
            task_ctx.task_id(),
            task_ctx.session_id(),
            task_ctx.session_config().clone(),
            task_ctx.scalar_functions().clone(),
            task_ctx.higher_order_functions().clone(),
            task_ctx.aggregate_functions().clone(),
            task_ctx.window_functions().clone(),
            Arc::new(runtime),
        ))
    }

    /// The profiling of the threads of `pipeline`.
    pub(crate) fn pipeline(
        self: &Arc<Self>,
        pipeline: &Pipeline,
    ) -> Arc<PipelineProfiling> {
        let stage_operators = self
            .stage_operators(pipeline)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        Arc::new(PipelineProfiling {
            profiler: Arc::clone(self),
            pipeline: pipeline.id(),
            stage_operators,
        })
    }

    /// Mark the call of `operator` that runs on this thread, see
    /// [`OperatorScope`].
    pub(crate) fn enter(&self, operator: &dyn PhysicalOperator) -> Option<OperatorScope> {
        self.ids
            .get(&operator_key(operator))
            .map(|&id| OperatorScope::enter(self.id, id))
    }

    /// Record that meta pipeline `meta` was finalized, in `time`.
    pub(crate) fn record_finalize(&self, meta: MetaPipelineId, time: Duration) {
        if let Some(record) = self.finalizations.get(meta) {
            let mut record = record.lock();
            record.finished = Some(Instant::now());
            record.time += nanos(time);
        }
    }

    /// Record that the execution completed.
    pub(crate) fn finish(&self) {
        *self.finished.lock() = Some(Instant::now());
    }

    /// The profile of the execution, which must have completed.
    pub(crate) fn profile(&self) -> Arc<QueryProfile> {
        Arc::clone(self.profile.get_or_init(|| Arc::new(self.build())))
    }

    fn build(&self) -> QueryProfile {
        let finished = self.finished.lock().unwrap_or_else(Instant::now);
        let at = |instant: Instant| nanos(instant.duration_since(self.started));
        let mut counters = vec![OperatorCounters::default(); self.operators.len()];
        let mut roles: Vec<Vec<(PipelineId, OperatorRole)>> =
            vec![vec![]; self.operators.len()];
        let mut executor = ExecutorProfile {
            threads: self.num_threads as u64,
            pipelines: self.pipelines.len() as u64,
            ..Default::default()
        };

        let mut pipelines = vec![];
        for (pipeline, record) in self.graph.pipelines().iter().zip(&self.pipelines) {
            let record = record.lock();
            let mut stages = vec![];
            let stage_operators = self.stage_operators(pipeline);
            for ((operator, role), stage) in
                stage_operators.into_iter().zip(&record.stages)
            {
                counters[operator].add(stage);
                if !roles[operator].contains(&(pipeline.id(), role)) {
                    roles[operator].push((pipeline.id(), role));
                }
                stages.push(StageProfile {
                    operator,
                    role,
                    counters: *stage,
                });
            }
            let mut workers = record
                .workers
                .iter()
                .map(|worker| WorkerProfile {
                    thread: worker.thread,
                    started_at: at(worker.spawned),
                    finished_at: at(worker.finished),
                    elapsed_time: worker.active + worker.blocked + worker.queue_wait,
                    active_time: worker.active,
                    blocked_time: worker.blocked,
                    queue_wait_time: worker.queue_wait,
                    morsels: worker.morsels,
                    rows: worker.rows,
                })
                .collect::<Vec<_>>();
            workers.sort_by_key(|worker| worker.thread);
            let started_at = record.started.map_or(0, at);
            let finished_at = record.finished.map_or(started_at, at).max(started_at);
            let elapsed_time = finished_at - started_at;
            let threads = record.threads as u64;
            let active_time =
                record.setup + workers.iter().map(|w| w.active_time).sum::<u64>();
            let blocked_time = workers.iter().map(|w| w.blocked_time).sum::<u64>();
            let queue_wait_time = workers.iter().map(|w| w.queue_wait_time).sum::<u64>();
            let morsels = workers.iter().map(|w| w.morsels).sum::<u64>();
            executor.workers += workers.len() as u64;
            executor.morsels += morsels;
            executor.active_time += active_time;
            executor.blocked_time += blocked_time;
            executor.queue_wait_time += queue_wait_time;
            pipelines.push(PipelineProfile {
                id: pipeline.id(),
                meta_pipeline: pipeline.meta_pipeline(),
                source: self.id_of(pipeline.source()),
                operators: pipeline
                    .operators()
                    .iter()
                    .map(|op| self.id_of(op))
                    .collect(),
                sink: self.id_of(pipeline.sink()),
                dependencies: pipeline.dependencies().to_vec(),
                meta_dependencies: pipeline.meta_dependencies().to_vec(),
                skipped: record.skipped,
                threads,
                morsels,
                rows: workers.iter().map(|w| w.rows).sum(),
                started_at,
                finished_at,
                elapsed_time,
                setup_time: record.setup,
                active_time,
                blocked_time,
                queue_wait_time,
                idle_time: (threads * elapsed_time).saturating_sub(active_time),
                stages,
                workers,
            });
        }

        let mut meta_pipelines = vec![];
        for (meta, record) in self.graph.meta_pipelines().iter().zip(&self.finalizations)
        {
            let record = record.lock();
            let sink = self.id_of(meta.sink());
            counters[sink].self_time += record.time;
            executor.active_time += record.time;
            meta_pipelines.push(MetaPipelineProfile {
                id: meta.id(),
                sink,
                pipelines: meta.pipelines().to_vec(),
                children: meta.children().to_vec(),
                parent: meta.parent(),
                finalize_time: record.time,
                finalized_at: record.finished.map_or(0, at),
            });
        }

        let operators = self
            .operators
            .iter()
            .enumerate()
            .map(|(id, operator)| OperatorProfile {
                id,
                kind: operator.name().to_string(),
                name: operator.description(),
                inputs: self.inputs[id].clone(),
                pipelines: std::mem::take(&mut roles[id]),
                counters: counters[id],
                memory_peak: self.memory.peak(id),
            })
            .collect::<Vec<_>>();

        executor.elapsed_time = at(finished);
        executor.thread_time =
            executor.active_time + executor.blocked_time + executor.queue_wait_time;
        executor.idle_time = (executor.threads * executor.elapsed_time)
            .saturating_sub(executor.active_time);
        let self_time = operators
            .iter()
            .map(|operator| operator.counters.self_time)
            .sum::<u64>();
        executor.overhead_time = executor.active_time.saturating_sub(self_time);
        executor.memory_peak = self.memory.total.peak();
        executor.unattributed_memory_peak = self.memory.peak(self.operators.len());

        QueryProfile {
            executor,
            operators,
            pipelines,
            meta_pipelines,
            ids: OperatorIds {
                ids: self.ids.clone(),
                _operators: self.operators.clone(),
            },
        }
    }

    fn id_of(&self, operator: &Arc<dyn PhysicalOperator>) -> usize {
        self.ids
            .get(&operator_key(operator.as_ref()))
            .copied()
            .unwrap_or_default()
    }

    /// The ids and roles of the stages of `pipeline`.
    fn stage_operators(&self, pipeline: &Pipeline) -> Vec<(usize, OperatorRole)> {
        std::iter::once((self.id_of(pipeline.source()), OperatorRole::Source))
            .chain(
                pipeline
                    .operators()
                    .iter()
                    .map(|operator| (self.id_of(operator), OperatorRole::Operator)),
            )
            .chain(std::iter::once((
                self.id_of(pipeline.sink()),
                OperatorRole::Sink,
            )))
            .collect()
    }
}

/// The operators of a graph, by id, as [`QueryProfile::operators`] lists
/// them.
#[derive(Default)]
struct OperatorTree {
    operators: Vec<Arc<dyn PhysicalOperator>>,
    inputs: Vec<Vec<usize>>,
    ids: HashMap<usize, usize>,
}

impl OperatorTree {
    /// Add `operator` and its children, unless they were added already, and
    /// return its id.
    fn visit(&mut self, operator: &Arc<dyn PhysicalOperator>) -> usize {
        let key = operator_key(operator.as_ref());
        if let Some(&id) = self.ids.get(&key) {
            return id;
        }
        let id = self.operators.len();
        self.ids.insert(key, id);
        self.operators.push(Arc::clone(operator));
        self.inputs.push(vec![]);
        let inputs = operator
            .children()
            .iter()
            .map(|child| self.visit(child))
            .collect();
        self.inputs[id] = inputs;
        id
    }
}

/// What the threads of a pipeline share for profiling.
pub(crate) struct PipelineProfiling {
    profiler: Arc<ExecutionProfiler>,
    pipeline: PipelineId,
    /// The id of the operator of every stage: the source, the operators and
    /// the sink of the pipeline.
    stage_operators: Vec<usize>,
}

impl PipelineProfiling {
    /// Mark the call of the operator at `stage` that runs on this thread.
    pub(crate) fn enter(&self, stage: usize) -> OperatorScope {
        OperatorScope::enter(self.profiler.id, self.stage_operators[stage])
    }

    /// Record a call of the operator at `stage` that set the pipeline up,
    /// on the task of the execution.
    pub(crate) fn record_setup(&self, stage: usize, time: Duration) {
        let mut record = self.profiler.pipelines[self.pipeline].lock();
        if let Some(counters) = record.stages.get_mut(stage) {
            counters.self_time += nanos(time);
        }
    }

    /// Record that the pipeline, which started at `started`, ran on
    /// `threads` threads (`skipped` its work) and completed, after a setup
    /// of `setup`.
    pub(crate) fn record_run(
        &self,
        started: Instant,
        setup: Duration,
        threads: usize,
        skipped: bool,
    ) {
        let mut record = self.profiler.pipelines[self.pipeline].lock();
        record.started = Some(started);
        record.finished = Some(Instant::now());
        record.setup = nanos(setup);
        record.threads = threads;
        record.skipped = skipped;
    }

    /// Start profiling thread `thread` of the pipeline, which was spawned at
    /// `spawned`.
    pub(crate) fn worker(
        self: &Arc<Self>,
        thread: usize,
        spawned: Instant,
    ) -> WorkerProfiler {
        let stages = self.stage_operators.len();
        let now = Instant::now();
        WorkerProfiler {
            pipeline: Arc::clone(self),
            thread,
            stages: vec![OperatorCounters::default(); stages],
            inputs: vec![None; stages],
            column_bytes: ColumnBytes::default(),
            spawned,
            active_since: Some(now),
            suspended: None,
            blocked_stage: None,
            active: 0,
            blocked: 0,
            queue_wait: nanos(now.duration_since(spawned)),
        }
    }
}

/// What a thread of a pipeline records, merged into the profile of the
/// pipeline when it completes.
///
/// The stages are numbered as in the
/// [`PipelineExecutor`](crate::executor::PipelineExecutor): `0` is the
/// source, `1..=n` are the operators and `n + 1` is the sink.
pub(crate) struct WorkerProfiler {
    pipeline: Arc<PipelineProfiling>,
    thread: usize,
    stages: Vec<OperatorCounters>,
    /// For every stage: the rows and bytes of the chunk that it received as
    /// input and did not consume yet.
    inputs: Vec<Option<(u64, u64)>>,
    column_bytes: ColumnBytes,
    spawned: Instant,
    /// When the current active period of the thread started.
    active_since: Option<Instant>,
    /// When the thread was suspended, and the stage it was blocked on
    /// (`None` if it yielded).
    suspended: Option<(Instant, Option<usize>)>,
    /// The stage whose last call returned `Blocked`, until the thread calls
    /// an operator again.
    blocked_stage: Option<usize>,
    active: u64,
    blocked: u64,
    queue_wait: u64,
}

impl WorkerProfiler {
    /// Mark the call of the operator at `stage` that runs on this thread.
    pub(crate) fn enter(&self, stage: usize) -> OperatorScope {
        self.pipeline.enter(stage)
    }

    /// Record a call of `get_data`, `execute`, `final_execute` or `sink` of
    /// the operator at `stage`, which took `time`.
    #[inline]
    pub(crate) fn record_call(&mut self, stage: usize, time: Duration) {
        let counters = &mut self.stages[stage];
        counters.self_time += nanos(time);
        counters.calls += 1;
        self.blocked_stage = None;
    }

    /// Record time spent in the operator at `stage` outside of its calls of
    /// [`Self::record_call`]: creating its states, combining, polling the
    /// futures it waits for.
    pub(crate) fn record_time(&mut self, stage: usize, time: Duration) {
        self.stages[stage].self_time += nanos(time);
    }

    /// Record that the last call of the operator at `stage` returned
    /// `Blocked`.
    pub(crate) fn record_blocked(&mut self, stage: usize) {
        self.blocked_stage = Some(stage);
    }

    /// Record that the operator at `stage` produced `chunk`, the input of the
    /// next stage.
    #[inline]
    pub(crate) fn record_output(&mut self, stage: usize, chunk: &DataChunk) {
        let rows = chunk.len() as u64;
        let bytes = self.column_bytes.chunk_bytes(chunk) as u64;
        let counters = &mut self.stages[stage];
        counters.rows_out += rows;
        counters.morsels_out += 1;
        counters.bytes_out += bytes;
        if let Some(next) = self.inputs.get_mut(stage + 1) {
            *next = Some((rows, bytes));
        }
    }

    /// Record that the operator at `stage` consumed its input, if it did not
    /// consume it before.
    #[inline]
    pub(crate) fn record_input(&mut self, stage: usize) {
        if let Some((rows, bytes)) = self.inputs[stage].take() {
            let counters = &mut self.stages[stage];
            counters.rows_in += rows;
            counters.morsels_in += 1;
            counters.bytes_in += bytes;
        }
    }

    /// The task of the thread is polled at `now`.
    pub(crate) fn resume(&mut self, now: Instant) {
        if let Some((since, stage)) = self.suspended.take() {
            let time = nanos(now.duration_since(since));
            match stage {
                Some(stage) => {
                    self.blocked += time;
                    self.stages[stage].blocked_time += time;
                }
                None => self.queue_wait += time,
            }
        }
        if self.active_since.is_none() {
            self.active_since = Some(now);
        }
    }

    /// The poll of the task of the thread returned at `now`, and the thread
    /// waits: blocked if the last call returned `Blocked`, else queued.
    pub(crate) fn suspend(&mut self, now: Instant) {
        self.end_active(now);
        self.suspended = Some((now, self.blocked_stage));
    }

    fn end_active(&mut self, now: Instant) {
        if let Some(since) = self.active_since.take() {
            self.active += nanos(now.duration_since(since));
        }
    }

    /// The thread completed at `now`: merge what it recorded into the
    /// profile of its pipeline.
    pub(crate) fn finish(mut self, now: Instant) {
        self.end_active(now);
        let mut record = self.pipeline.profiler.pipelines[self.pipeline.pipeline].lock();
        for (total, stage) in record.stages.iter_mut().zip(&self.stages) {
            total.add(stage);
        }
        record.workers.push(WorkerRecord {
            thread: self.thread,
            spawned: self.spawned,
            finished: now,
            active: self.active,
            blocked: self.blocked,
            queue_wait: self.queue_wait,
            morsels: self.stages[0].morsels_out,
            rows: self.stages[0].rows_out,
        });
    }
}

// -------------------------------------------------------------------------
// Memory
// -------------------------------------------------------------------------

thread_local! {
    /// The operators whose calls run on this thread, the innermost last: the
    /// id of the profiler and the id of the operator. Executions nest (e.g.
    /// the plan of an `EXPLAIN ANALYZE`, or the input of a streaming merge,
    /// which an operator executes during its calls), so that the profiler of
    /// every execution finds its own innermost operator.
    static RUNNING_OPERATORS: RefCell<Vec<(u64, usize)>> = const { RefCell::new(Vec::new()) };
}

/// Marks a call of an operator that runs on this thread, until it is
/// dropped: the memory pool of a profiled execution attributes the
/// consumers that are registered during the call to the operator.
#[must_use]
pub(crate) struct OperatorScope {
    /// The scope must be dropped on the thread that entered it.
    _thread: PhantomData<*const ()>,
}

impl OperatorScope {
    fn enter(profiler: u64, operator: usize) -> Self {
        let _ = RUNNING_OPERATORS.try_with(|running| {
            running.borrow_mut().push((profiler, operator));
        });
        Self {
            _thread: PhantomData,
        }
    }
}

impl Drop for OperatorScope {
    fn drop(&mut self) {
        let _ = RUNNING_OPERATORS.try_with(|running| running.borrow_mut().pop());
    }
}

/// The innermost operator of the profiler `profiler` whose call runs on this
/// thread.
fn running_operator(profiler: u64) -> Option<usize> {
    RUNNING_OPERATORS
        .try_with(|running| {
            running
                .borrow()
                .iter()
                .rev()
                .find(|(id, _)| *id == profiler)
                .map(|&(_, operator)| operator)
        })
        .ok()
        .flatten()
}

/// The number of shards of the map of the owners of the memory consumers, so
/// that threads that grow their reservations rarely wait for each other.
const OWNER_SHARDS: usize = 32;

/// Records the memory that the operators of an execution reserve: every
/// memory consumer is owned by the operator whose call registered it.
struct MemoryRecorder {
    profiler: u64,
    /// The owners of the registered consumers, by consumer id, sharded.
    owners: Vec<Mutex<HashMap<usize, usize>>>,
    /// The memory of every operator, by id, and of the consumers without an
    /// owner (last).
    usage: Vec<MemoryUsage>,
    total: MemoryUsage,
}

#[derive(Debug, Default)]
struct MemoryUsage {
    current: AtomicUsize,
    peak: AtomicUsize,
}

impl MemoryUsage {
    fn grow(&self, bytes: usize) {
        let current = self.current.fetch_add(bytes, Ordering::Relaxed) + bytes;
        self.peak.fetch_max(current, Ordering::Relaxed);
    }

    fn shrink(&self, bytes: usize) {
        let mut current = self.current.load(Ordering::Relaxed);
        while let Err(actual) = self.current.compare_exchange_weak(
            current,
            current.saturating_sub(bytes),
            Ordering::Relaxed,
            Ordering::Relaxed,
        ) {
            current = actual;
        }
    }

    fn peak(&self) -> u64 {
        self.peak.load(Ordering::Relaxed) as u64
    }
}

impl MemoryRecorder {
    fn new(profiler: u64, operators: usize) -> Self {
        Self {
            profiler,
            owners: (0..OWNER_SHARDS).map(|_| Mutex::default()).collect(),
            usage: (0..=operators).map(|_| MemoryUsage::default()).collect(),
            total: MemoryUsage::default(),
        }
    }

    fn unowned(&self) -> usize {
        self.usage.len() - 1
    }

    fn shard(&self, consumer: usize) -> &Mutex<HashMap<usize, usize>> {
        &self.owners[consumer % OWNER_SHARDS]
    }

    fn register(&self, consumer: usize) {
        let owner = running_operator(self.profiler).unwrap_or_else(|| self.unowned());
        self.shard(consumer).lock().insert(consumer, owner);
    }

    fn unregister(&self, consumer: usize) {
        self.shard(consumer).lock().remove(&consumer);
    }

    fn usage(&self, consumer: usize) -> &MemoryUsage {
        let owner = self.shard(consumer).lock().get(&consumer).copied();
        &self.usage[owner.unwrap_or_else(|| self.unowned())]
    }

    fn grow(&self, consumer: usize, bytes: usize) {
        self.usage(consumer).grow(bytes);
        self.total.grow(bytes);
    }

    fn shrink(&self, consumer: usize, bytes: usize) {
        self.usage(consumer).shrink(bytes);
        self.total.shrink(bytes);
    }

    /// The peak memory of the operator with id `operator` (or of the
    /// consumers without an owner).
    fn peak(&self, operator: usize) -> u64 {
        self.usage.get(operator).map_or(0, MemoryUsage::peak)
    }
}

/// The memory pool of a profiled execution: forwards everything to the
/// memory pool of the session, and records the memory of every operator.
#[derive(Debug)]
struct ProfilingMemoryPool {
    inner: Arc<dyn MemoryPool>,
    recorder: Arc<MemoryRecorder>,
}

impl fmt::Debug for MemoryRecorder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MemoryRecorder")
            .field("profiler", &self.profiler)
            .finish_non_exhaustive()
    }
}

impl Display for ProfilingMemoryPool {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        Display::fmt(&self.inner, f)
    }
}

impl MemoryPool for ProfilingMemoryPool {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn register(&self, consumer: &MemoryConsumer) {
        self.recorder.register(consumer.id());
        self.inner.register(consumer);
    }

    fn unregister(&self, consumer: &MemoryConsumer) {
        self.inner.unregister(consumer);
        self.recorder.unregister(consumer.id());
    }

    fn grow(&self, reservation: &MemoryReservation, additional: usize) {
        self.inner.grow(reservation, additional);
        self.recorder.grow(reservation.consumer().id(), additional);
    }

    fn shrink(&self, reservation: &MemoryReservation, shrink: usize) {
        self.inner.shrink(reservation, shrink);
        self.recorder.shrink(reservation.consumer().id(), shrink);
    }

    fn try_grow(&self, reservation: &MemoryReservation, additional: usize) -> Result<()> {
        self.inner.try_grow(reservation, additional)?;
        self.recorder.grow(reservation.consumer().id(), additional);
        Ok(())
    }

    fn reserved(&self) -> usize {
        self.inner.reserved()
    }

    fn memory_limit(&self) -> MemoryLimit {
        self.inner.memory_limit()
    }
}

// -------------------------------------------------------------------------
// Bytes
// -------------------------------------------------------------------------

/// The number of bytes of the rows of `chunk` (`bytes_in` and `bytes_out`):
/// the bytes of the slices of its columns ([`array_bytes`]), scaled to the
/// rows that its selection vector selects, if it has one.
pub fn chunk_bytes(chunk: &DataChunk) -> usize {
    let bytes = chunk
        .columns()
        .iter()
        .map(|column| array_bytes(column.as_ref()))
        .sum::<usize>();
    scale_to_selection(bytes, chunk)
}

/// `bytes`, the bytes of the physical rows of `chunk`, scaled to the rows
/// that its selection vector selects, if it has one.
fn scale_to_selection(bytes: usize, chunk: &DataChunk) -> usize {
    let physical = chunk.physical_len();
    if chunk.selection().is_none() || physical == 0 || chunk.len() == physical {
        return bytes;
    }
    (bytes as u128 * chunk.len() as u128 / physical as u128) as usize
}

/// The maximum number of columns of a chunk that [`ColumnBytes`] keeps.
const CACHED_COLUMNS: usize = 64;

/// [`chunk_bytes`] of the chunks of the stages of a thread, which remembers
/// the bytes of the columns of the last chunk: operators pass many columns on
/// unchanged (e.g. a filter that selects rows, a projection), whose bytes are
/// not computed again (computing them reads all views of a view column).
#[derive(Default)]
struct ColumnBytes {
    /// The columns of the last chunk and their bytes.
    last: Vec<(ArrayRef, usize)>,
    next: Vec<(ArrayRef, usize)>,
}

impl ColumnBytes {
    fn chunk_bytes(&mut self, chunk: &DataChunk) -> usize {
        let columns = chunk.columns();
        if columns.len() > CACHED_COLUMNS {
            self.last.clear();
            return chunk_bytes(chunk);
        }
        let mut bytes = 0;
        for column in columns {
            let column_bytes = self
                .last
                .iter()
                .find(|(array, _)| Arc::ptr_eq(array, column))
                .map_or_else(|| array_bytes(column.as_ref()), |&(_, bytes)| bytes);
            bytes += column_bytes;
            self.next.push((Arc::clone(column), column_bytes));
        }
        std::mem::swap(&mut self.last, &mut self.next);
        self.next.clear();
        scale_to_selection(bytes, chunk)
    }
}

/// The number of bytes of the values of `array`, a slice of a possibly larger
/// array: its null mask (one bit per row) and, by type,
///
/// * fixed-width types (numbers, dates, times, decimals, `FixedSizeBinary`):
///   the values of its rows;
/// * `Boolean`: one bit per row;
/// * strings and binaries with offsets: the offsets and the bytes of the
///   values of its rows;
/// * views (`Utf8View`, `BinaryView`): 16 bytes per row, and the bytes of the
///   values that are not inlined into their views;
/// * lists and maps: the offsets, and the values of their rows;
/// * fixed-size lists and structs: the bytes of their children;
/// * dictionaries: their keys, and all of their values;
/// * other types: what Arrow's `ArrayData::get_slice_memory_size` reports.
pub fn array_bytes(array: &dyn Array) -> usize {
    let len = array.len();
    let nulls = array.nulls().map_or(0, |nulls| nulls.len().div_ceil(8));
    let values = match array.data_type() {
        DataType::Null => 0,
        DataType::Boolean => len.div_ceil(8),
        DataType::FixedSizeBinary(width) => len * (*width).max(0) as usize,
        DataType::Utf8 => offset_bytes(array.as_string::<i32>().value_offsets()),
        DataType::LargeUtf8 => offset_bytes(array.as_string::<i64>().value_offsets()),
        DataType::Binary => offset_bytes(array.as_binary::<i32>().value_offsets()),
        DataType::LargeBinary => offset_bytes(array.as_binary::<i64>().value_offsets()),
        DataType::Utf8View => view_bytes(array.as_string_view().views()),
        DataType::BinaryView => view_bytes(array.as_binary_view().views()),
        DataType::List(_) => list_bytes(array.as_list::<i32>()),
        DataType::LargeList(_) => list_bytes(array.as_list::<i64>()),
        DataType::FixedSizeList(_, _) => {
            array_bytes(array.as_fixed_size_list().values().as_ref())
        }
        DataType::Struct(_) => array
            .as_struct()
            .columns()
            .iter()
            .map(|column| array_bytes(column.as_ref()))
            .sum(),
        DataType::Map(_, _) => {
            let map = array.as_map();
            let offsets = map.value_offsets();
            let (start, end) = offset_range(offsets);
            size_of_val(offsets) + array_bytes(&map.entries().slice(start, end - start))
        }
        DataType::Dictionary(_, _) => {
            // The null mask of the keys is the null mask of the array.
            let dictionary = array.as_any_dictionary();
            return array_bytes(dictionary.keys())
                + array_bytes(dictionary.values().as_ref());
        }
        data_type => match data_type.primitive_width() {
            Some(width) => len * width,
            None => {
                return array
                    .to_data()
                    .get_slice_memory_size()
                    .unwrap_or_else(|_| array.get_array_memory_size());
            }
        },
    };
    values + nulls
}

/// The first and the last of `offsets`.
fn offset_range<O: OffsetSizeTrait>(offsets: &[O]) -> (usize, usize) {
    let start = offsets.first().map_or(0, |offset| offset.as_usize());
    let end = offsets.last().map_or(start, |offset| offset.as_usize());
    (start, end.max(start))
}

/// The bytes of the offsets `offsets` and of the values between them.
fn offset_bytes<O: OffsetSizeTrait>(offsets: &[O]) -> usize {
    let (start, end) = offset_range(offsets);
    size_of_val(offsets) + (end - start)
}

/// The bytes of the views `views` and of the values that are not inlined.
fn view_bytes(views: &[u128]) -> usize {
    let outlined = views
        .iter()
        .map(|&view| {
            let len = view as u32;
            if len > 12 { len as usize } else { 0 }
        })
        .sum::<usize>();
    size_of_val(views) + outlined
}

/// The bytes of the offsets of `list` and of the values of its rows.
fn list_bytes<O: OffsetSizeTrait>(list: &GenericListArray<O>) -> usize {
    let offsets = list.value_offsets();
    let (start, end) = offset_range(offsets);
    size_of_val(offsets) + array_bytes(list.values().slice(start, end - start).as_ref())
}
