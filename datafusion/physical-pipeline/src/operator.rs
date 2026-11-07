// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! The [`PhysicalOperator`] trait, its execution states and results.
//!
//! The design follows DuckDB's `PhysicalOperator`: a single operator type can
//! play up to three roles in the pipelines of a query:
//!
//! * **Source**: produces [`DataChunk`]s at the start of a pipeline
//!   ([`PhysicalOperator::get_data`]).
//! * **Operator**: transforms chunks inside a pipeline in a streaming fashion
//!   ([`PhysicalOperator::execute`]).
//! * **Sink**: consumes all chunks at the end of a pipeline
//!   ([`PhysicalOperator::sink`], [`PhysicalOperator::combine`] and
//!   [`PhysicalOperator::finalize`]).
//!
//! A pipeline breaker, such as a hash aggregate, is the sink of one pipeline
//! and the source of the pipeline that consumes its result. A hash join is the
//! sink of its build-side pipeline and an operator in its probe-side
//! pipeline. How an operator participates in pipelines is decided by
//! [`PhysicalOperator::build_pipelines`].
//!
//! Operators are immutable and can be executed many times. All execution
//! state lives in *state* objects that are created by the operator and owned
//! by the [`Executor`](crate::executor::Executor): one *global* state per
//! execution (shared by all threads, hence `Send + Sync`) and one *local*
//! state per thread (hence only `Send`).

use std::any::Any;
use std::collections::{HashMap, HashSet};
use std::fmt::{self, Debug};
use std::future::Future;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::task::Waker;
use std::time::Duration;

use arrow::datatypes::SchemaRef;
use datafusion_common::{Result, internal_err};
use datafusion_execution::TaskContext;
use futures::future::BoxFuture;
use parking_lot::Mutex;

use crate::chunk::DataChunk;
use crate::pipeline::{MetaPipelineBuilder, PipelineBuilder};

/// Describes why an operator could not make progress and how the pipeline
/// task is resumed (DuckDB's `BLOCKED` result plus its `InterruptState`).
///
/// When an operator returns a `Blocked` result, the
/// [`PipelineExecutor`](crate::executor::PipelineExecutor) suspends the task
/// and calls the operator again *with the same input* once it is resumed.
pub struct TaskInterrupt {
    kind: InterruptKind,
}

pub(crate) enum InterruptKind {
    /// The operator registered [`ExecutionContext::waker`] with the resource
    /// it waits for.
    WakerRegistered,
    /// Re-schedule the task immediately.
    Yield,
    /// Resume once the future completes.
    Future(BoxFuture<'static, ()>),
}

impl TaskInterrupt {
    /// The operator registered [`ExecutionContext::waker`] with whatever it
    /// is waiting for (for example by polling a stream or a channel with a
    /// [`std::task::Context`] built from it). The task is resumed once that
    /// waker is woken.
    ///
    /// Returning this without registering the waker suspends the task
    /// forever.
    pub fn waker_registered() -> Self {
        Self {
            kind: InterruptKind::WakerRegistered,
        }
    }

    /// Yield to the scheduler and resume the task as soon as possible.
    pub fn yield_now() -> Self {
        Self {
            kind: InterruptKind::Yield,
        }
    }

    /// Resume the task once `future` completes.
    ///
    /// The executor drives `future` to completion before it calls the
    /// blocked method again. It only drops `future` early if the blocked call
    /// is not needed any more: when the sink of the pipeline, or an operator
    /// at or after the blocked one, returned `Finished`.
    pub fn wait_for(future: impl Future<Output = ()> + Send + 'static) -> Self {
        Self {
            kind: InterruptKind::Future(Box::pin(future)),
        }
    }

    pub(crate) fn into_kind(self) -> InterruptKind {
        self.kind
    }
}

impl Debug for TaskInterrupt {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let kind = match &self.kind {
            InterruptKind::WakerRegistered => "WakerRegistered",
            InterruptKind::Yield => "Yield",
            InterruptKind::Future(_) => "Future",
        };
        f.debug_tuple("TaskInterrupt").field(&kind).finish()
    }
}

/// Result of [`PhysicalOperator::get_data`].
#[derive(Debug)]
pub enum SourceResult {
    /// The source produced a chunk (of at most
    /// [`DEFAULT_VECTOR_SIZE`](crate::chunk::DEFAULT_VECTOR_SIZE) rows). The
    /// source is called again for more data.
    Chunk(DataChunk),
    /// The source is exhausted for this thread.
    Finished,
    /// The source cannot produce data right now.
    Blocked(TaskInterrupt),
}

/// Result of [`PhysicalOperator::execute`].
#[derive(Debug)]
pub enum OperatorResult {
    /// The operator consumed its input; `output` (possibly empty) is pushed
    /// to the next operator and the operator is called with the next input.
    NeedMoreInput,
    /// The operator has more output for the *same* input: after `output` was
    /// pushed down the pipeline, the operator is called again with the same
    /// input chunk.
    HaveMoreOutput,
    /// The operator will not produce any more output, on any thread. Any
    /// rows in `output` are still pushed down the pipeline, and the operators
    /// after this one are still flushed, but neither this operator nor the
    /// ones before it are called again: all threads of all pipelines that
    /// contain the operator stop pulling from their sources (threads blocked
    /// in their source are woken up, pipelines that did not start yet do not
    /// run their source). A thread that was already calling the operator
    /// concurrently may still call it once more.
    Finished,
    /// The operator cannot make progress; it is called again with the same
    /// input once the task is resumed. `output` is ignored.
    Blocked(TaskInterrupt),
}

/// Result of [`PhysicalOperator::final_execute`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OperatorFinalizeResult {
    /// `output` is pushed down the pipeline and `final_execute` is called
    /// again.
    HaveMoreOutput,
    /// `output` (possibly empty) is the last output of the operator.
    Finished,
}

/// Result of [`PhysicalOperator::sink`].
#[derive(Debug)]
pub enum SinkResult {
    /// The sink consumed the chunk and accepts more input.
    NeedMoreInput,
    /// The sink consumed the chunk and does not need any more input, from any
    /// thread (e.g. a `LIMIT` was reached). All threads of all pipelines that
    /// end in this sink stop: threads blocked in their source are woken up,
    /// pipelines that did not start yet do not run their source, and
    /// operators are not flushed. Every thread still calls
    /// [`PhysicalOperator::combine`], and the sink is finalized once. A
    /// thread that was already pushing a chunk concurrently may still sink
    /// it, so the sink should keep returning `Finished`.
    ///
    /// Pipeline breakers whose output is then no longer needed (all the
    /// pipelines that read from them skip their work) are cancelled the same
    /// way, as if their sink had finished: their threads stop, combine and
    /// they are finalized.
    Finished,
    /// The sink could not consume the chunk; it is called again with the
    /// same chunk once the task is resumed.
    Blocked(TaskInterrupt),
}

/// Result of [`PhysicalOperator::finalize`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SinkFinalize {
    /// The sink is finalized; pipelines that depend on it can run.
    Ready,
    /// The sink is finalized and the operator knows that it cannot produce
    /// any output (e.g. an inner join with an empty build side). Pipelines in
    /// which the operator is the source or an operator are skipped (their
    /// sinks still see a `combine` call, without any input), and the pipeline
    /// breakers that only these pipelines read from (e.g. on the probe side
    /// of the join) are cancelled, see [`SinkResult::Finished`].
    NoOutputPossible,
}

/// Global state of a source, shared by all threads executing a pipeline.
pub trait GlobalSourceState: Any + Send + Sync {
    /// The maximum number of threads that can usefully read from this
    /// source in parallel (DuckDB's `MaxThreads`). The executor never starts
    /// more threads than this for the pipeline.
    fn max_threads(&self) -> usize {
        usize::MAX
    }
}

/// Thread-local state of a source.
pub trait LocalSourceState: Any + Send {}

/// Global state of a streaming operator, shared by all threads.
pub trait GlobalOperatorState: Any + Send + Sync {}

/// Thread-local state of a streaming operator.
pub trait OperatorState: Any + Send {}

/// Global state of a sink, shared by all threads.
///
/// An operator that is a sink and also a source or operator (e.g. a hash
/// join or an aggregate) can access its global sink state from its other
/// roles with [`ExecutionContext::global_sink_state`].
pub trait GlobalSinkState: Any + Send + Sync {}

/// Thread-local state of a sink.
pub trait LocalSinkState: Any + Send {}

macro_rules! impl_state_downcast {
    ($trait:ident) => {
        impl dyn $trait {
            /// Returns `self` as [`Any`], for downcasting to the concrete state.
            pub fn as_any(&self) -> &dyn Any {
                self
            }

            /// Returns `self` as mutable [`Any`], for downcasting to the
            /// concrete state.
            pub fn as_any_mut(&mut self) -> &mut dyn Any {
                self
            }

            /// Downcast to the concrete state type `T`.
            pub fn downcast_ref<T: $trait>(&self) -> Option<&T> {
                self.as_any().downcast_ref::<T>()
            }

            /// Downcast to the concrete state type `T`.
            pub fn downcast_mut<T: $trait>(&mut self) -> Option<&mut T> {
                self.as_any_mut().downcast_mut::<T>()
            }
        }
    };
}

impl_state_downcast!(GlobalSourceState);
impl_state_downcast!(LocalSourceState);
impl_state_downcast!(GlobalOperatorState);
impl_state_downcast!(OperatorState);
impl_state_downcast!(GlobalSinkState);
impl_state_downcast!(LocalSinkState);

impl dyn GlobalSinkState {
    /// Downcast a shared global sink state to the concrete type `T`.
    pub fn downcast_arc<T: GlobalSinkState>(self: Arc<Self>) -> Option<Arc<T>> {
        let any: Arc<dyn Any + Send + Sync> = self;
        any.downcast::<T>().ok()
    }
}

/// A state that holds nothing. It is the default state of every role of a
/// [`PhysicalOperator`].
#[derive(Debug, Default, Clone, Copy)]
pub struct EmptyState;

impl GlobalSourceState for EmptyState {}
impl LocalSourceState for EmptyState {}
impl GlobalOperatorState for EmptyState {}
impl OperatorState for EmptyState {}
impl GlobalSinkState for EmptyState {}
impl LocalSinkState for EmptyState {}

/// Downcast `state` to `T`, returning an internal error naming `operator` on
/// mismatch.
pub fn downcast_state<'a, T: Any>(state: &'a dyn Any, operator: &str) -> Result<&'a T> {
    match state.downcast_ref::<T>() {
        Some(state) => Ok(state),
        None => internal_err!(
            "{operator} received a state of an unexpected type (expected {})",
            std::any::type_name::<T>()
        ),
    }
}

/// Mutable variant of [`downcast_state`].
pub fn downcast_state_mut<'a, T: Any>(
    state: &'a mut dyn Any,
    operator: &str,
) -> Result<&'a mut T> {
    match state.downcast_mut::<T>() {
        Some(state) => Ok(state),
        None => internal_err!(
            "{operator} received a state of an unexpected type (expected {})",
            std::any::type_name::<T>()
        ),
    }
}

/// Runtime metrics of one [`PhysicalOperator`], aggregated over all threads
/// and all the pipelines the operator takes part in.
#[derive(Debug, Default)]
pub struct OperatorMetrics {
    output_rows: AtomicUsize,
    output_chunks: AtomicUsize,
    sink_rows: AtomicUsize,
    sink_chunks: AtomicUsize,
    elapsed_nanos: AtomicU64,
}

impl OperatorMetrics {
    /// Record a chunk of `rows` rows produced by the operator (as a source
    /// or as an operator).
    pub fn record_output(&self, rows: usize) {
        self.output_rows.fetch_add(rows, Ordering::Relaxed);
        self.output_chunks.fetch_add(1, Ordering::Relaxed);
    }

    /// Record a chunk of `rows` rows consumed by the operator as a sink.
    pub fn record_sink(&self, rows: usize) {
        self.sink_rows.fetch_add(rows, Ordering::Relaxed);
        self.sink_chunks.fetch_add(1, Ordering::Relaxed);
    }

    /// Add time spent in the operator.
    pub fn add_elapsed(&self, elapsed: Duration) {
        let nanos = u64::try_from(elapsed.as_nanos()).unwrap_or(u64::MAX);
        self.elapsed_nanos.fetch_add(nanos, Ordering::Relaxed);
    }

    /// Number of rows produced by the operator as a source or an operator.
    pub fn output_rows(&self) -> usize {
        self.output_rows.load(Ordering::Relaxed)
    }

    /// Number of non-empty chunks produced by the operator as a source or an
    /// operator.
    pub fn output_chunks(&self) -> usize {
        self.output_chunks.load(Ordering::Relaxed)
    }

    /// Number of rows consumed by the operator as a sink.
    pub fn sink_rows(&self) -> usize {
        self.sink_rows.load(Ordering::Relaxed)
    }

    /// Number of chunks consumed by the operator as a sink.
    pub fn sink_chunks(&self) -> usize {
        self.sink_chunks.load(Ordering::Relaxed)
    }

    /// Total time spent in the operator, summed over all threads, in
    /// nanoseconds.
    pub fn elapsed_nanos(&self) -> u64 {
        self.elapsed_nanos.load(Ordering::Relaxed)
    }
}

/// Identity of an operator instance, used to key per-execution state.
pub(crate) fn operator_key(operator: &dyn PhysicalOperator) -> usize {
    std::ptr::from_ref(operator).cast::<()>().addr()
}

/// State shared by all tasks of one execution of a pipeline graph.
#[derive(Default)]
pub(crate) struct ExecutionRegistry {
    /// Serializes the creation of global states, so that every state is
    /// created exactly once even if several pipelines start concurrently.
    /// Separate from the maps, which operators may read while a state is
    /// created.
    creation: Mutex<()>,
    sink_states: Mutex<HashMap<usize, Arc<dyn GlobalSinkState>>>,
    operator_states: Mutex<HashMap<usize, Arc<dyn GlobalOperatorState>>>,
    metrics: HashMap<usize, Arc<OperatorMetrics>>,
    /// The structure of the pipelines, to find work that is no longer needed.
    dependencies: PipelineDependencies,
    /// Operators and sinks that finished, and sinks without output.
    finished: Mutex<FinishedOperators>,
    /// Incremented whenever an operator or a sink finishes, so that threads
    /// can cheaply detect that they may have to stop.
    finish_epoch: AtomicUsize,
    /// Wakers of the threads that are blocked, by thread id, woken when an
    /// operator or a sink finishes.
    blocked_threads: Mutex<HashMap<usize, Waker>>,
    next_thread_id: AtomicUsize,
}

/// The pipelines of an execution in terms of operator keys (see
/// `operator_key`), to find the work that is no longer needed once an
/// operator finished.
#[derive(Debug, Default)]
pub(crate) struct PipelineDependencies {
    /// For every pipeline: its source, operators and sink.
    pub(crate) pipelines: Vec<PipelineKeys>,
    /// For every meta pipeline: its sink, and the pipelines that read the
    /// result of its sink (as their source or as an operator).
    pub(crate) meta_pipelines: Vec<MetaPipelineKeys>,
}

#[derive(Debug)]
pub(crate) struct PipelineKeys {
    pub(crate) source: usize,
    pub(crate) operators: Vec<usize>,
    pub(crate) sink: usize,
}

#[derive(Debug)]
pub(crate) struct MetaPipelineKeys {
    pub(crate) sink: usize,
    pub(crate) consumers: Vec<usize>,
}

#[derive(Debug, Default)]
struct FinishedOperators {
    /// Operators that returned `Finished` as a streaming operator.
    operators: HashSet<usize>,
    /// Sinks that returned `Finished`, or whose output is not needed.
    sinks: HashSet<usize>,
    /// Sinks that were finalized with `SinkFinalize::NoOutputPossible`.
    no_output: HashSet<usize>,
}

impl FinishedOperators {
    /// Whether a pipeline that did not start yet will skip its work: its
    /// sink or one of its operators finished, or its source or one of its
    /// operators cannot produce output.
    fn skips(&self, pipeline: &PipelineKeys) -> bool {
        self.sinks.contains(&pipeline.sink)
            || self.no_output.contains(&pipeline.source)
            || pipeline
                .operators
                .iter()
                .any(|op| self.operators.contains(op) || self.no_output.contains(op))
    }

    /// Mark the sinks of the meta pipelines whose output is no longer needed
    /// as finished: those all of whose consumers skip their work, as their
    /// sink or operators finished. Repeats until nothing changes, as this
    /// makes the pipelines of these meta pipelines skip their work too.
    /// Returns `true` if a sink was marked.
    fn cancel_unused(&mut self, dependencies: &PipelineDependencies) -> bool {
        let mut changed = false;
        loop {
            let mut progress = false;
            for meta in &dependencies.meta_pipelines {
                if meta.consumers.is_empty() || self.sinks.contains(&meta.sink) {
                    continue;
                }
                if meta
                    .consumers
                    .iter()
                    .all(|&pipeline| self.skips(&dependencies.pipelines[pipeline]))
                {
                    self.sinks.insert(meta.sink);
                    progress = true;
                }
            }
            if !progress {
                return changed;
            }
            changed = true;
        }
    }
}

impl ExecutionRegistry {
    pub(crate) fn new(
        metrics: HashMap<usize, Arc<OperatorMetrics>>,
        dependencies: PipelineDependencies,
    ) -> Self {
        Self {
            metrics,
            dependencies,
            ..Default::default()
        }
    }

    /// Returns the global sink state of `sink`, creating it on first use.
    pub(crate) fn get_or_create_sink_state(
        &self,
        sink: &dyn PhysicalOperator,
        ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalSinkState>> {
        let key = operator_key(sink);
        let _creation = self.creation.lock();
        if let Some(state) = self.sink_states.lock().get(&key) {
            return Ok(Arc::clone(state));
        }
        let state = sink.get_global_sink_state(ctx)?;
        self.sink_states.lock().insert(key, Arc::clone(&state));
        Ok(state)
    }

    /// Returns the global operator state of `operator`, creating it on first
    /// use. Operators that appear in several pipelines (e.g. above a union)
    /// share one global state.
    pub(crate) fn get_or_create_operator_state(
        &self,
        operator: &dyn PhysicalOperator,
        ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalOperatorState>> {
        let key = operator_key(operator);
        let _creation = self.creation.lock();
        if let Some(state) = self.operator_states.lock().get(&key) {
            return Ok(Arc::clone(state));
        }
        let state = operator.get_global_operator_state(ctx)?;
        self.operator_states.lock().insert(key, Arc::clone(&state));
        Ok(state)
    }

    pub(crate) fn sink_state(
        &self,
        operator: &dyn PhysicalOperator,
    ) -> Option<Arc<dyn GlobalSinkState>> {
        self.sink_states
            .lock()
            .get(&operator_key(operator))
            .map(Arc::clone)
    }

    /// Record the result of finalizing `sink`. A sink that cannot produce
    /// output makes the pipelines that read from it skip their work, which
    /// may make other work unnecessary (see [`Self::set_finished`]).
    pub(crate) fn set_finalize_result(
        &self,
        sink: &dyn PhysicalOperator,
        result: SinkFinalize,
    ) {
        let key = operator_key(sink);
        if result == SinkFinalize::NoOutputPossible {
            let cancelled = {
                let mut finished = self.finished.lock();
                finished.no_output.insert(key);
                finished.cancel_unused(&self.dependencies)
            };
            if cancelled {
                self.notify_finished();
            }
        }
    }

    /// A unique id for a thread (a `PipelineExecutor`) of the execution.
    pub(crate) fn next_thread_id(&self) -> usize {
        self.next_thread_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Changes whenever an operator or a sink finishes.
    pub(crate) fn finish_epoch(&self) -> usize {
        self.finish_epoch.load(Ordering::SeqCst)
    }

    /// Record that `operator` returned `Finished` as a streaming operator
    /// (`as_sink == false`) or as a sink, and wake all blocked threads, so
    /// that the threads of the pipelines that contain it stop.
    ///
    /// The meta pipelines whose output is then no longer needed (all the
    /// pipelines reading from their sink skip their work) are cancelled too:
    /// their sink is marked as finished, so that their pipelines stop.
    pub(crate) fn set_finished(&self, operator: &dyn PhysicalOperator, as_sink: bool) {
        let key = operator_key(operator);
        {
            let mut finished = self.finished.lock();
            let newly_finished = if as_sink {
                finished.sinks.insert(key)
            } else {
                finished.operators.insert(key)
            };
            if !newly_finished {
                return;
            }
            finished.cancel_unused(&self.dependencies);
        }
        self.notify_finished();
    }

    /// Wake all blocked threads after an operator or a sink finished.
    fn notify_finished(&self) {
        self.finish_epoch.fetch_add(1, Ordering::SeqCst);
        let wakers = self
            .blocked_threads
            .lock()
            .drain()
            .map(|(_, waker)| waker)
            .collect::<Vec<_>>();
        for waker in wakers {
            waker.wake();
        }
    }

    /// Whether the pipeline with id `pipeline` skips its work if it starts
    /// now: its sink or one of its operators finished, or its source or one
    /// of its operators cannot produce output.
    pub(crate) fn skips_pipeline(&self, pipeline: usize) -> bool {
        self.dependencies
            .pipelines
            .get(pipeline)
            .is_some_and(|keys| self.finished.lock().skips(keys))
    }

    /// Whether `operator` returned `Finished` as a streaming operator.
    pub(crate) fn is_operator_finished(&self, operator: &dyn PhysicalOperator) -> bool {
        self.finished
            .lock()
            .operators
            .contains(&operator_key(operator))
    }

    /// Whether `operator` returned `Finished` as a sink, or its output is not
    /// needed any more.
    pub(crate) fn is_sink_finished(&self, operator: &dyn PhysicalOperator) -> bool {
        self.finished.lock().sinks.contains(&operator_key(operator))
    }

    /// Register the waker of a blocked thread, to be woken when an operator
    /// or a sink finishes.
    pub(crate) fn register_blocked(&self, thread: usize, waker: &Waker) {
        let mut blocked = self.blocked_threads.lock();
        match blocked.get_mut(&thread) {
            Some(registered) => registered.clone_from(waker),
            None => {
                blocked.insert(thread, waker.clone());
            }
        }
    }

    /// Forget a thread that completed.
    pub(crate) fn unregister(&self, thread: usize) {
        self.blocked_threads.lock().remove(&thread);
    }

    pub(crate) fn metrics(
        &self,
        operator: &dyn PhysicalOperator,
    ) -> Arc<OperatorMetrics> {
        self.metrics
            .get(&operator_key(operator))
            .map(Arc::clone)
            .unwrap_or_default()
    }
}

/// Per-thread context passed to every [`PhysicalOperator`] method.
#[derive(Clone)]
pub struct ExecutionContext {
    /// The DataFusion task context (configuration, memory pool, functions).
    pub task_ctx: Arc<TaskContext>,
    /// Index of the thread executing the pipeline, in `0..num_threads`.
    pub thread_id: usize,
    /// Number of threads executing the pipeline. When a global state is
    /// created or a sink is finalized, the maximum number of threads of any
    /// pipeline of the execution.
    pub num_threads: usize,
    registry: Arc<ExecutionRegistry>,
    waker: Waker,
}

impl ExecutionContext {
    /// Create a context that is not attached to an
    /// [`Executor`](crate::executor::Executor), e.g. to unit test an operator.
    pub fn new(task_ctx: Arc<TaskContext>, thread_id: usize, num_threads: usize) -> Self {
        Self::new_with_registry(
            task_ctx,
            thread_id,
            num_threads,
            Arc::new(ExecutionRegistry::default()),
        )
    }

    pub(crate) fn new_with_registry(
        task_ctx: Arc<TaskContext>,
        thread_id: usize,
        num_threads: usize,
        registry: Arc<ExecutionRegistry>,
    ) -> Self {
        Self {
            task_ctx,
            thread_id,
            num_threads,
            registry,
            waker: Waker::noop().clone(),
        }
    }

    /// The waker of the task currently executing the pipeline.
    ///
    /// Operators that wait for an asynchronous resource (e.g. a
    /// [`Stream`](futures::Stream)) poll it with a
    /// [`std::task::Context`] built from this waker and return
    /// [`TaskInterrupt::waker_registered`] while it is pending.
    pub fn waker(&self) -> &Waker {
        &self.waker
    }

    pub(crate) fn registry(&self) -> &Arc<ExecutionRegistry> {
        &self.registry
    }

    pub(crate) fn set_waker(&mut self, waker: &Waker) {
        self.waker.clone_from(waker);
    }

    /// Returns the global sink state that was created for `operator` in the
    /// current execution, if `operator` is a sink whose state exists.
    ///
    /// This is how an operator that is both a sink and a source / operator
    /// (a hash join, an aggregate, a sort, ...) reaches the data it collected
    /// as a sink. Call it with `self` from within the operator's methods.
    pub fn global_sink_state(
        &self,
        operator: &dyn PhysicalOperator,
    ) -> Option<Arc<dyn GlobalSinkState>> {
        self.registry.sink_state(operator)
    }

    /// The runtime metrics of `operator` in the current execution.
    ///
    /// The executor records output rows, chunks and elapsed time for every
    /// operator call; operators can add their own measurements.
    pub fn metrics(&self, operator: &dyn PhysicalOperator) -> Arc<OperatorMetrics> {
        self.registry.metrics(operator)
    }
}

impl Debug for ExecutionContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ExecutionContext")
            .field("session_id", &self.task_ctx.session_id())
            .field("thread_id", &self.thread_id)
            .field("num_threads", &self.num_threads)
            .finish()
    }
}

/// A physical operator of the pipeline engine.
///
/// See the [module documentation](self) for the roles an operator can play.
/// Every method has a default implementation, so an operator implements
/// only the methods of the roles it plays: a source implements
/// [`Self::is_source`] and [`Self::get_data`], a streaming operator
/// [`Self::execute`], a sink [`Self::is_sink`], [`Self::sink`],
/// [`Self::combine`] and [`Self::finalize`] (and usually also the source
/// role, to emit its result).
pub trait PhysicalOperator: Any + Debug + Send + Sync {
    /// Short name of the operator, e.g. `"BridgeOperator"`.
    fn name(&self) -> &str;

    /// The schema of the chunks produced by this operator.
    fn schema(&self) -> SchemaRef;

    /// The input operators of this operator.
    fn children(&self) -> &[Arc<dyn PhysicalOperator>];

    /// One-line description of the operator for `EXPLAIN` output.
    fn description(&self) -> String {
        self.name().to_owned()
    }

    // ---------------------------------------------------------------------
    // Operator role
    // ---------------------------------------------------------------------

    /// Whether several threads may call [`Self::execute`] concurrently (with
    /// different local states). If `false`, the pipelines that contain this
    /// operator run on a single thread, one pipeline at a time.
    fn parallel_operator(&self) -> bool {
        true
    }

    /// Create the state shared by all threads that execute this operator.
    fn get_global_operator_state(
        &self,
        _ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalOperatorState>> {
        Ok(Arc::new(EmptyState))
    }

    /// Create the state of one thread that executes this operator.
    fn get_operator_state(
        &self,
        _ctx: &ExecutionContext,
        _gstate: &dyn GlobalOperatorState,
    ) -> Result<Box<dyn OperatorState>> {
        Ok(Box::new(EmptyState))
    }

    /// Process the (non-empty) `input` chunk and write the result to
    /// `output`, which is cleared before the call. See [`OperatorResult`]
    /// for how the return value drives the pipeline.
    fn execute(
        &self,
        _ctx: &ExecutionContext,
        _input: &DataChunk,
        _output: &mut DataChunk,
        _gstate: &dyn GlobalOperatorState,
        _lstate: &mut dyn OperatorState,
    ) -> Result<OperatorResult> {
        internal_err!("{} cannot be used as a streaming operator", self.name())
    }

    /// Whether [`Self::final_execute`] must be called once the source of the
    /// pipeline is exhausted (e.g. to flush buffered rows).
    fn requires_final_execute(&self) -> bool {
        false
    }

    /// Flush any output buffered in `lstate` into `output` (cleared before
    /// the call) after the source of the pipeline is exhausted.
    fn final_execute(
        &self,
        _ctx: &ExecutionContext,
        _output: &mut DataChunk,
        _gstate: &dyn GlobalOperatorState,
        _lstate: &mut dyn OperatorState,
    ) -> Result<OperatorFinalizeResult> {
        Ok(OperatorFinalizeResult::Finished)
    }

    // ---------------------------------------------------------------------
    // Source role
    // ---------------------------------------------------------------------

    /// Whether this operator can be the source of a pipeline.
    fn is_source(&self) -> bool {
        false
    }

    /// Whether several threads may call [`Self::get_data`] concurrently. If
    /// `false`, pipelines with this source run on a single thread, one
    /// pipeline at a time.
    fn parallel_source(&self) -> bool {
        false
    }

    /// Create the source state shared by all threads of the pipeline.
    fn get_global_source_state(
        &self,
        _ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalSourceState>> {
        Ok(Arc::new(EmptyState))
    }

    /// Create the source state of one thread of the pipeline.
    fn get_local_source_state(
        &self,
        _ctx: &ExecutionContext,
        _gstate: &dyn GlobalSourceState,
    ) -> Result<Box<dyn LocalSourceState>> {
        Ok(Box::new(EmptyState))
    }

    /// Produce the next chunk of this source for the calling thread.
    fn get_data(
        &self,
        _ctx: &ExecutionContext,
        _gstate: &dyn GlobalSourceState,
        _lstate: &mut dyn LocalSourceState,
    ) -> Result<SourceResult> {
        internal_err!("{} cannot be used as a source", self.name())
    }

    // ---------------------------------------------------------------------
    // Sink role
    // ---------------------------------------------------------------------

    /// Whether this operator is a sink, i.e. a pipeline breaker that
    /// consumes all of its input before producing output.
    fn is_sink(&self) -> bool {
        false
    }

    /// Whether several threads may call [`Self::sink`] concurrently (with
    /// different local states). If `false`, the pipelines that end in this
    /// sink run on a single thread, one pipeline at a time.
    fn parallel_sink(&self) -> bool {
        false
    }

    /// Create the sink state shared by all threads. It is created once per
    /// execution and stays alive until the execution completes, see
    /// [`ExecutionContext::global_sink_state`].
    fn get_global_sink_state(
        &self,
        _ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalSinkState>> {
        Ok(Arc::new(EmptyState))
    }

    /// Create the sink state of one thread.
    fn get_local_sink_state(
        &self,
        _ctx: &ExecutionContext,
        _gstate: &dyn GlobalSinkState,
    ) -> Result<Box<dyn LocalSinkState>> {
        Ok(Box::new(EmptyState))
    }

    /// Consume the (non-empty) `input` chunk.
    fn sink(
        &self,
        _ctx: &ExecutionContext,
        _input: &DataChunk,
        _gstate: &dyn GlobalSinkState,
        _lstate: &mut dyn LocalSinkState,
    ) -> Result<SinkResult> {
        internal_err!("{} cannot be used as a sink", self.name())
    }

    /// Merge the local state of a thread into the global state. Called once
    /// per thread, after the thread finished its part of the pipeline.
    fn combine(
        &self,
        _ctx: &ExecutionContext,
        _gstate: &dyn GlobalSinkState,
        _lstate: &mut dyn LocalSinkState,
    ) -> Result<()> {
        Ok(())
    }

    /// Called once, after all pipelines that end in this sink completed and
    /// before any pipeline that depends on this sink starts.
    fn finalize(
        &self,
        _ctx: &ExecutionContext,
        _gstate: &dyn GlobalSinkState,
    ) -> Result<SinkFinalize> {
        Ok(SinkFinalize::Ready)
    }

    // ---------------------------------------------------------------------
    // Pipeline construction
    // ---------------------------------------------------------------------

    /// Add this operator to the pipeline `current` of the meta pipeline
    /// `meta`, and recurse into its children.
    ///
    /// The default implementation (see [`MetaPipelineBuilder::build_default`])
    /// handles the common cases:
    ///
    /// * a sink becomes the source of `current`, and its single child is
    ///   built into a new child meta pipeline that has this operator as sink;
    /// * a leaf operator becomes the source of `current`;
    /// * an operator with one child is added to `current`, which then
    ///   continues into the child.
    ///
    /// Operators with several children (joins, unions) or that play several
    /// roles in a non-standard way override this method, using the methods
    /// of [`MetaPipelineBuilder`].
    fn build_pipelines(
        &self,
        current: &mut PipelineBuilder,
        meta: &mut MetaPipelineBuilder<'_>,
    ) -> Result<()> {
        meta.build_default(current)
    }
}

impl dyn PhysicalOperator {
    /// Returns `self` as [`Any`], for downcasting to the concrete operator.
    pub fn as_any(&self) -> &dyn Any {
        self
    }

    /// Returns `true` if this operator is of type `T`.
    pub fn is<T: PhysicalOperator>(&self) -> bool {
        self.as_any().is::<T>()
    }

    /// Downcast to the concrete operator type `T`.
    pub fn downcast_ref<T: PhysicalOperator>(&self) -> Option<&T> {
        self.as_any().downcast_ref::<T>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug, Default)]
    struct CountingState {
        count: usize,
    }
    impl LocalSinkState for CountingState {}
    impl GlobalSinkState for CountingState {}

    #[test]
    fn states_downcast() {
        let mut local: Box<dyn LocalSinkState> = Box::new(CountingState::default());
        local.downcast_mut::<CountingState>().unwrap().count += 2;
        assert_eq!(local.downcast_ref::<CountingState>().unwrap().count, 2);
        assert!(local.downcast_ref::<EmptyState>().is_none());
        assert!(local.as_any().is::<CountingState>());

        let global: Arc<dyn GlobalSinkState> = Arc::new(CountingState { count: 3 });
        assert_eq!(global.downcast_ref::<CountingState>().unwrap().count, 3);
        let concrete = Arc::clone(&global).downcast_arc::<CountingState>().unwrap();
        assert_eq!(concrete.count, 3);
        assert!(global.downcast_arc::<EmptyState>().is_none());

        let err = downcast_state::<EmptyState>(local.as_any(), "TestSink").unwrap_err();
        assert!(err.to_string().contains("TestSink"), "{err}");
        assert!(
            downcast_state_mut::<CountingState>(local.as_any_mut(), "TestSink").is_ok()
        );
    }

    #[test]
    fn metrics_accumulate() {
        let metrics = OperatorMetrics::default();
        metrics.record_output(10);
        metrics.record_output(5);
        metrics.record_sink(7);
        metrics.add_elapsed(Duration::from_nanos(42));
        assert_eq!(metrics.output_rows(), 15);
        assert_eq!(metrics.output_chunks(), 2);
        assert_eq!(metrics.sink_rows(), 7);
        assert_eq!(metrics.sink_chunks(), 1);
        assert_eq!(metrics.elapsed_nanos(), 42);
    }

    #[test]
    fn interrupt_debug() {
        assert_eq!(
            format!("{:?}", TaskInterrupt::waker_registered()),
            "TaskInterrupt(\"WakerRegistered\")"
        );
        assert_eq!(
            format!("{:?}", TaskInterrupt::wait_for(async {})),
            "TaskInterrupt(\"Future\")"
        );
        assert_eq!(
            format!("{:?}", TaskInterrupt::yield_now()),
            "TaskInterrupt(\"Yield\")"
        );
    }
}
