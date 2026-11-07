// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Execution of a [`PipelineGraph`].
//!
//! * [`PipelineExecutor`] runs one thread's share of one [`Pipeline`]: a
//!   tight push loop that pulls chunks from the source and pushes them
//!   through the operators into the sink (DuckDB's `PipelineExecutor`).
//! * [`Executor`] schedules the pipelines of a query: every pipeline runs on
//!   up to `num_threads` worker tasks once all of its dependencies are
//!   complete, and the sink of a [`MetaPipeline`] is finalized once all of
//!   its pipelines completed.
//! * [`CollectorSink`] is the sink of the root pipeline: it hands the result
//!   chunks, as [`RecordBatch`]es, to a [`PipelineStream`].

use std::collections::{HashMap, VecDeque};
use std::fmt;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use datafusion_common::instant::Instant;
use datafusion_common::{DataFusionError, Result, internal_err};
use datafusion_common_runtime::{JoinSet, SpawnedTask};
use datafusion_execution::{RecordBatchStream, TaskContext};
use futures::future::BoxFuture;
use futures::stream::FuturesUnordered;
use futures::{Stream, StreamExt, TryStreamExt};
use parking_lot::Mutex;

use crate::chunk::DataChunk;
use crate::operator::{
    EmptyState, ExecutionContext, ExecutionRegistry, GlobalOperatorState,
    GlobalSinkState, GlobalSourceState, InterruptKind, LocalSinkState, LocalSourceState,
    OperatorFinalizeResult, OperatorMetrics, OperatorResult, OperatorState,
    PhysicalOperator, SinkFinalize, SinkResult, SourceResult, TaskInterrupt,
    downcast_state_mut, operator_key,
};
use crate::pipeline::{MetaPipeline, Pipeline, PipelineGraph};

/// Maximum number of steps (roughly: chunks) a [`PipelineExecutor`] processes
/// before it yields back to the async runtime.
const STEPS_PER_POLL: usize = 128;

/// Everything the threads of one pipeline share.
struct PipelineShared {
    source: Arc<dyn PhysicalOperator>,
    operators: Vec<Arc<dyn PhysicalOperator>>,
    sink: Arc<dyn PhysicalOperator>,
    source_state: Arc<dyn GlobalSourceState>,
    operator_states: Vec<Arc<dyn GlobalOperatorState>>,
    sink_state: Arc<dyn GlobalSinkState>,
    source_metrics: Arc<OperatorMetrics>,
    operator_metrics: Vec<Arc<OperatorMetrics>>,
    sink_metrics: Arc<OperatorMetrics>,
    /// An operator of the pipeline reported [`SinkFinalize::NoOutputPossible`]:
    /// skip the push loop entirely.
    skip: bool,
}

/// One step of the push loop.
enum Step {
    /// Progress was made, continue.
    Continue,
    /// An operator is blocked.
    Blocked(TaskInterrupt),
    /// The pipeline is complete for this thread.
    Done,
}

/// Outcome of pushing a chunk through the operators of a pipeline.
enum PushOutcome {
    /// The last chunk (`chunks[n]`) holds output for the sink.
    Output,
    /// No output for the sink; the operators need more input.
    NeedMoreInput,
    /// An operator is blocked.
    Blocked(TaskInterrupt),
}

/// Executes one thread's share of a [`Pipeline`] (DuckDB's
/// `PipelineExecutor`).
///
/// The operators of the pipeline are numbered by *stage*: stage `0` is the
/// source and stage `k` (`1..=n`) is `operators[k - 1]`. `chunks[k]` holds
/// the output of stage `k`, so `chunks[n]` is the chunk pushed into the
/// sink.
///
/// When an operator returns [`OperatorResult::HaveMoreOutput`], its stage is
/// pushed onto the `in_process` stack; before pulling new data from the
/// source, the executor calls the in-process operators again with the same
/// input, from the most downstream one.
pub struct PipelineExecutor {
    shared: Arc<PipelineShared>,
    ctx: ExecutionContext,
    source_state: Box<dyn LocalSourceState>,
    operator_states: Vec<Box<dyn OperatorState>>,
    sink_state: Box<dyn LocalSinkState>,
    chunks: Vec<DataChunk>,
    in_process: Vec<usize>,
    /// Stage whose output is currently pushed through the pipeline (`0` for
    /// source chunks, `k + 1` while flushing `operators[k]`).
    push_stage: usize,
    /// `chunks[n]` still has to be (re-)pushed into the sink.
    pending_sink: bool,
    /// The source is exhausted (or the pipeline finished early).
    source_exhausted: bool,
    /// Index of the next operator to flush with `final_execute`. Operators
    /// before this index are never flushed (they are upstream of an operator
    /// that finished).
    flush_index: usize,
    /// The operator at `flush_index` returned `Finished` from `final_execute`.
    flush_current_finished: bool,
    /// Future the task waits for before continuing.
    wait: Option<BoxFuture<'static, ()>>,
    done: bool,
}

impl PipelineExecutor {
    fn try_new(shared: Arc<PipelineShared>, ctx: ExecutionContext) -> Result<Self> {
        let num_operators = shared.operators.len();
        let sink_state = shared
            .sink
            .get_local_sink_state(&ctx, shared.sink_state.as_ref())?;
        let mut chunks = Vec::with_capacity(num_operators + 1);
        chunks.push(DataChunk::empty(shared.source.schema()));
        chunks.extend(
            shared
                .operators
                .iter()
                .map(|operator| DataChunk::empty(operator.schema())),
        );

        if shared.skip {
            return Ok(Self {
                source_state: Box::new(EmptyState),
                operator_states: vec![],
                sink_state,
                chunks,
                in_process: vec![],
                push_stage: 0,
                pending_sink: false,
                source_exhausted: true,
                flush_index: num_operators,
                flush_current_finished: false,
                wait: None,
                done: false,
                shared,
                ctx,
            });
        }

        let source_state = shared
            .source
            .get_local_source_state(&ctx, shared.source_state.as_ref())?;
        let operator_states = shared
            .operators
            .iter()
            .zip(&shared.operator_states)
            .map(|(operator, gstate)| operator.get_operator_state(&ctx, gstate.as_ref()))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            source_state,
            operator_states,
            sink_state,
            chunks,
            in_process: vec![],
            push_stage: 0,
            pending_sink: false,
            source_exhausted: false,
            flush_index: 0,
            flush_current_finished: false,
            wait: None,
            done: false,
            shared,
            ctx,
        })
    }

    /// Run the pipeline until it is complete or blocked.
    ///
    /// Returns `Poll::Ready` once the source is exhausted, all operators are
    /// flushed and the local sink state was combined into the global one.
    pub fn poll_execute(&mut self, cx: &mut Context<'_>) -> Poll<Result<()>> {
        if self.done {
            return Poll::Ready(Ok(()));
        }
        self.ctx.set_waker(cx.waker());
        if let Some(wait) = self.wait.as_mut() {
            if wait.as_mut().poll(cx).is_pending() {
                return Poll::Pending;
            }
            self.wait = None;
        }
        for _ in 0..STEPS_PER_POLL {
            match self.step() {
                Ok(Step::Continue) => {}
                Ok(Step::Done) => {
                    self.done = true;
                    return Poll::Ready(self.combine());
                }
                Ok(Step::Blocked(interrupt)) => match interrupt.into_kind() {
                    InterruptKind::WakerRegistered => return Poll::Pending,
                    InterruptKind::Yield => {
                        cx.waker().wake_by_ref();
                        return Poll::Pending;
                    }
                    InterruptKind::Future(mut future) => {
                        if future.as_mut().poll(cx).is_pending() {
                            self.wait = Some(future);
                            return Poll::Pending;
                        }
                    }
                },
                Err(e) => {
                    self.done = true;
                    return Poll::Ready(Err(e));
                }
            }
        }
        // Yield to give other tasks a chance to run.
        cx.waker().wake_by_ref();
        Poll::Pending
    }

    /// Run the pipeline to completion.
    pub async fn execute(mut self) -> Result<()> {
        futures::future::poll_fn(|cx| self.poll_execute(cx)).await
    }

    fn step(&mut self) -> Result<Step> {
        // A chunk the sink did not accept yet.
        if self.pending_sink {
            return self.sink_last_chunk();
        }
        // Operators that have more output for their current input.
        if !self.in_process.is_empty() {
            let outcome = self.push(self.push_stage)?;
            return self.handle_push_outcome(outcome);
        }
        if !self.source_exhausted {
            return self.fetch_from_source();
        }
        if let Some(step) = self.flush()? {
            return Ok(step);
        }
        Ok(Step::Done)
    }

    fn fetch_from_source(&mut self) -> Result<Step> {
        let start = Instant::now();
        let result = self.shared.source.get_data(
            &self.ctx,
            self.shared.source_state.as_ref(),
            self.source_state.as_mut(),
        );
        self.shared.source_metrics.add_elapsed(start.elapsed());
        match result? {
            SourceResult::Chunk(chunk) => {
                if chunk.is_empty() {
                    return Ok(Step::Continue);
                }
                self.shared.source_metrics.record_output(chunk.len());
                self.chunks[0] = chunk;
                self.push_stage = 0;
                let outcome = self.push(0)?;
                self.handle_push_outcome(outcome)
            }
            SourceResult::Finished => {
                self.source_exhausted = true;
                Ok(Step::Continue)
            }
            SourceResult::Blocked(interrupt) => Ok(Step::Blocked(interrupt)),
        }
    }

    fn handle_push_outcome(&mut self, outcome: PushOutcome) -> Result<Step> {
        match outcome {
            PushOutcome::Output => {
                self.pending_sink = true;
                self.sink_last_chunk()
            }
            PushOutcome::NeedMoreInput => Ok(Step::Continue),
            PushOutcome::Blocked(interrupt) => Ok(Step::Blocked(interrupt)),
        }
    }

    /// Push the output of stage `initial` through the operators after it, or
    /// resume the most downstream in-process operator.
    fn push(&mut self, initial: usize) -> Result<PushOutcome> {
        let shared = Arc::clone(&self.shared);
        let n = shared.operators.len();
        let mut stage = match self.in_process.pop() {
            Some(stage) => stage,
            None => initial + 1,
        };
        loop {
            if stage > n {
                return Ok(PushOutcome::Output);
            }
            let index = stage - 1;
            let (inputs, outputs) = self.chunks.split_at_mut(stage);
            let input = &inputs[stage - 1];
            let output = &mut outputs[0];
            output.reset();

            let metrics = &shared.operator_metrics[index];
            let start = Instant::now();
            let result = shared.operators[index].execute(
                &self.ctx,
                input,
                output,
                shared.operator_states[index].as_ref(),
                self.operator_states[index].as_mut(),
            );
            metrics.add_elapsed(start.elapsed());
            match result? {
                OperatorResult::NeedMoreInput => {}
                OperatorResult::HaveMoreOutput => self.in_process.push(stage),
                OperatorResult::Finished => self.finish_at(stage),
                OperatorResult::Blocked(interrupt) => {
                    output.reset();
                    self.in_process.push(stage);
                    return Ok(PushOutcome::Blocked(interrupt));
                }
            }

            let rows = self.chunks[stage].len();
            if rows == 0 {
                // No output: resume the most downstream in-process operator,
                // or ask for more input.
                match self.in_process.pop() {
                    Some(in_process) => stage = in_process,
                    None => return Ok(PushOutcome::NeedMoreInput),
                }
            } else {
                metrics.record_output(rows);
                stage += 1;
            }
        }
    }

    fn sink_last_chunk(&mut self) -> Result<Step> {
        let chunk = self.chunks.last().expect("pipeline has a source chunk");
        let rows = chunk.len();
        let start = Instant::now();
        let result = self.shared.sink.sink(
            &self.ctx,
            chunk,
            self.shared.sink_state.as_ref(),
            self.sink_state.as_mut(),
        );
        self.shared.sink_metrics.add_elapsed(start.elapsed());
        match result? {
            SinkResult::NeedMoreInput => {
                self.shared.sink_metrics.record_sink(rows);
                self.pending_sink = false;
                Ok(Step::Continue)
            }
            SinkResult::Finished => {
                self.shared.sink_metrics.record_sink(rows);
                self.pending_sink = false;
                // The sink does not want more data: stop everything,
                // including flushing.
                self.finish_at(self.shared.operators.len());
                Ok(Step::Continue)
            }
            SinkResult::Blocked(interrupt) => Ok(Step::Blocked(interrupt)),
        }
    }

    /// The operator at `stage` (or the sink, for `stage == n`) finished:
    /// stop pulling from the source and never call the operators upstream of
    /// it again.
    fn finish_at(&mut self, stage: usize) {
        self.source_exhausted = true;
        // Stages on the stack are all upstream of `stage`.
        self.in_process.retain(|&s| s > stage);
        if stage > self.flush_index {
            self.flush_index = stage;
            self.flush_current_finished = false;
        }
    }

    /// Flush the operators that require `final_execute`, from the source
    /// side to the sink side. Returns `None` once all operators are flushed.
    fn flush(&mut self) -> Result<Option<Step>> {
        let shared = Arc::clone(&self.shared);
        let n = shared.operators.len();
        while self.flush_index < n {
            let index = self.flush_index;
            let operator = &shared.operators[index];
            if self.flush_current_finished || !operator.requires_final_execute() {
                self.flush_index += 1;
                self.flush_current_finished = false;
                continue;
            }
            let output = &mut self.chunks[index + 1];
            output.reset();
            let metrics = &shared.operator_metrics[index];
            let start = Instant::now();
            let result = operator.final_execute(
                &self.ctx,
                output,
                shared.operator_states[index].as_ref(),
                self.operator_states[index].as_mut(),
            );
            metrics.add_elapsed(start.elapsed());
            if result? == OperatorFinalizeResult::Finished {
                self.flush_current_finished = true;
            }
            let rows = self.chunks[index + 1].len();
            if rows == 0 {
                // Respect the step budget even if an operator keeps
                // returning `HaveMoreOutput` without output.
                return Ok(Some(Step::Continue));
            }
            metrics.record_output(rows);
            self.push_stage = index + 1;
            let outcome = self.push(index + 1)?;
            return self.handle_push_outcome(outcome).map(Some);
        }
        Ok(None)
    }

    fn combine(&mut self) -> Result<()> {
        let start = Instant::now();
        let result = self.shared.sink.combine(
            &self.ctx,
            self.shared.sink_state.as_ref(),
            self.sink_state.as_mut(),
        );
        self.shared.sink_metrics.add_elapsed(start.elapsed());
        result
    }
}

impl fmt::Debug for PipelineExecutor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PipelineExecutor")
            .field("source", &self.shared.source.name())
            .field("operators", &self.shared.operators.len())
            .field("sink", &self.shared.sink.name())
            .field("thread_id", &self.ctx.thread_id)
            .field("in_process", &self.in_process)
            .field("source_exhausted", &self.source_exhausted)
            .finish()
    }
}

/// Runtime metrics of all operators of one execution.
#[derive(Debug)]
pub struct ExecutionMetrics {
    operators: Vec<(Arc<dyn PhysicalOperator>, Arc<OperatorMetrics>)>,
}

impl ExecutionMetrics {
    fn new(graph: &PipelineGraph) -> Self {
        let operators = graph
            .operators()
            .into_iter()
            .map(|operator| (operator, Arc::new(OperatorMetrics::default())))
            .collect();
        Self { operators }
    }

    /// The metrics of every operator, in pipeline order.
    pub fn iter(
        &self,
    ) -> impl Iterator<Item = (&Arc<dyn PhysicalOperator>, &Arc<OperatorMetrics>)> {
        self.operators.iter().map(|(op, metrics)| (op, metrics))
    }

    /// The metrics of `operator`, if it is part of the execution.
    pub fn get(&self, operator: &dyn PhysicalOperator) -> Option<&Arc<OperatorMetrics>> {
        let key = operator_key(operator);
        self.operators
            .iter()
            .find(|(op, _)| operator_key(op.as_ref()) == key)
            .map(|(_, metrics)| metrics)
    }

    fn by_key(&self) -> HashMap<usize, Arc<OperatorMetrics>> {
        self.operators
            .iter()
            .map(|(op, metrics)| (operator_key(op.as_ref()), Arc::clone(metrics)))
            .collect()
    }
}

/// Schedules and runs all pipelines of a [`PipelineGraph`].
///
/// Every pipeline becomes runnable once the pipelines it depends on are
/// complete and the child meta pipelines it depends on are finalized. A
/// runnable pipeline is executed by up to `num_threads` [`PipelineExecutor`]
/// tasks; independent pipelines run concurrently. Once all pipelines of a
/// meta pipeline completed (and every thread combined its local sink state),
/// the sink of the meta pipeline is finalized.
#[derive(Debug)]
pub struct Executor {
    graph: Arc<PipelineGraph>,
    task_ctx: Arc<TaskContext>,
    num_threads: usize,
    metrics: Arc<ExecutionMetrics>,
}

impl Executor {
    /// Build the pipelines of the operator tree below `sink`, which must be
    /// a sink with one child (see [`PipelineGraph::try_new`]).
    pub fn try_new(
        sink: Arc<dyn PhysicalOperator>,
        task_ctx: Arc<TaskContext>,
        num_threads: usize,
    ) -> Result<Self> {
        let graph = Arc::new(PipelineGraph::try_new(sink)?);
        let metrics = Arc::new(ExecutionMetrics::new(&graph));
        Ok(Self {
            graph,
            task_ctx,
            num_threads: num_threads.max(1),
            metrics,
        })
    }

    /// The pipelines executed by this executor.
    pub fn graph(&self) -> &Arc<PipelineGraph> {
        &self.graph
    }

    /// The runtime metrics of the operators, updated during execution.
    pub fn metrics(&self) -> &Arc<ExecutionMetrics> {
        &self.metrics
    }

    /// The maximum number of threads per pipeline.
    pub fn num_threads(&self) -> usize {
        self.num_threads
    }

    /// Execute `root` and return a stream of its output.
    ///
    /// The execution starts when the stream is first polled and is cancelled
    /// when the stream is dropped.
    pub fn execute_stream(
        root: Arc<dyn PhysicalOperator>,
        task_ctx: Arc<TaskContext>,
        num_threads: usize,
    ) -> Result<PipelineStream> {
        let queue = Arc::new(BatchQueue::new(num_threads.max(1) * 2));
        let schema = root.schema();
        let sink = Arc::new(CollectorSink::new(root, Arc::clone(&queue)));
        let executor = Self::try_new(sink, task_ctx, num_threads)?;
        Ok(PipelineStream {
            schema,
            metrics: Arc::clone(executor.metrics()),
            queue,
            state: StreamState::Pending(executor),
        })
    }

    /// Execute `root` and collect its output.
    pub async fn collect(
        root: Arc<dyn PhysicalOperator>,
        task_ctx: Arc<TaskContext>,
        num_threads: usize,
    ) -> Result<Vec<RecordBatch>> {
        Self::execute_stream(root, task_ctx, num_threads)?
            .try_collect()
            .await
    }

    /// Run all pipelines to completion.
    pub async fn execute(&self) -> Result<()> {
        let registry = Arc::new(ExecutionRegistry::new(self.metrics.by_key()));
        let graph = self.graph.as_ref();
        let num_pipelines = graph.pipelines().len();
        let num_events = num_pipelines + graph.meta_pipelines().len();

        // Events `0..num_pipelines` run a pipeline, event
        // `num_pipelines + m` finalizes meta pipeline `m`.
        let mut dependencies = vec![0usize; num_events];
        let mut dependents = vec![vec![]; num_events];
        for pipeline in graph.pipelines() {
            let event = pipeline.id();
            let prerequisites = pipeline.dependencies().iter().copied().chain(
                pipeline
                    .meta_dependencies()
                    .iter()
                    .map(|&meta| num_pipelines + meta),
            );
            for prerequisite in prerequisites {
                dependencies[event] += 1;
                dependents[prerequisite].push(event);
            }
        }
        for meta in graph.meta_pipelines() {
            let event = num_pipelines + meta.id();
            for &pipeline in meta.pipelines() {
                dependencies[event] += 1;
                dependents[pipeline].push(event);
            }
        }

        let mut running = FuturesUnordered::new();
        for (event, &count) in dependencies.iter().enumerate() {
            if count == 0 {
                running.push(self.run_event(event, &registry));
            }
        }
        let mut completed = 0;
        while completed < num_events {
            let Some(result) = running.next().await else {
                return internal_err!(
                    "Pipeline dependency cycle: {completed} of {num_events} events completed"
                );
            };
            let event = result?;
            completed += 1;
            for &dependent in &dependents[event] {
                dependencies[dependent] -= 1;
                if dependencies[dependent] == 0 {
                    running.push(self.run_event(dependent, &registry));
                }
            }
        }
        Ok(())
    }

    /// Run event `event`: pipeline `event` for `event < #pipelines`,
    /// otherwise the finalization of meta pipeline `event - #pipelines`.
    fn run_event<'a>(
        &'a self,
        event: usize,
        registry: &'a Arc<ExecutionRegistry>,
    ) -> BoxFuture<'a, Result<usize>> {
        Box::pin(async move {
            let num_pipelines = self.graph.pipelines().len();
            if event < num_pipelines {
                self.run_pipeline(self.graph.pipeline(event), registry)
                    .await?;
            } else {
                let meta = self.graph.meta_pipeline(event - num_pipelines);
                self.finalize(meta, registry)?;
            }
            Ok(event)
        })
    }

    fn context(
        &self,
        thread_id: usize,
        num_threads: usize,
        registry: &Arc<ExecutionRegistry>,
    ) -> ExecutionContext {
        ExecutionContext::new_with_registry(
            Arc::clone(&self.task_ctx),
            thread_id,
            num_threads,
            Arc::clone(registry),
        )
    }

    async fn run_pipeline(
        &self,
        pipeline: &Pipeline,
        registry: &Arc<ExecutionRegistry>,
    ) -> Result<()> {
        let ctx = self.context(0, self.num_threads, registry);
        // An operator that already knows it cannot produce output makes the
        // whole pipeline a no-op (its sink still combines an empty state).
        let skip = std::iter::once(pipeline.source())
            .chain(pipeline.operators())
            .any(|operator| {
                operator.is_sink()
                    && registry.finalize_result(operator.as_ref())
                        == Some(SinkFinalize::NoOutputPossible)
            });
        let sink_state =
            registry.get_or_create_sink_state(pipeline.sink().as_ref(), &ctx)?;
        let operator_states = pipeline
            .operators()
            .iter()
            .map(|operator| {
                registry.get_or_create_operator_state(operator.as_ref(), &ctx)
            })
            .collect::<Result<Vec<_>>>()?;
        let source_state: Arc<dyn GlobalSourceState> = if skip {
            Arc::new(EmptyState)
        } else {
            pipeline.source().get_global_source_state(&ctx)?
        };
        let num_threads = if skip {
            1
        } else {
            self.pipeline_threads(pipeline, source_state.as_ref())
        };

        let shared = Arc::new(PipelineShared {
            source: Arc::clone(pipeline.source()),
            operators: pipeline.operators().to_vec(),
            sink: Arc::clone(pipeline.sink()),
            source_state,
            operator_states,
            sink_state,
            source_metrics: registry.metrics(pipeline.source().as_ref()),
            operator_metrics: pipeline
                .operators()
                .iter()
                .map(|operator| registry.metrics(operator.as_ref()))
                .collect(),
            sink_metrics: registry.metrics(pipeline.sink().as_ref()),
            skip,
        });

        let mut workers = JoinSet::new();
        for thread_id in 0..num_threads {
            let shared = Arc::clone(&shared);
            let ctx = self.context(thread_id, num_threads, registry);
            workers.spawn(async move {
                PipelineExecutor::try_new(shared, ctx)?.execute().await
            });
        }
        // Dropping `workers` on error aborts the remaining threads.
        while let Some(result) = workers.join_next().await {
            match result {
                Ok(result) => result?,
                Err(e) if e.is_panic() => std::panic::resume_unwind(e.into_panic()),
                Err(e) => return Err(DataFusionError::ExecutionJoin(Box::new(e))),
            }
        }
        Ok(())
    }

    /// The number of threads that execute `pipeline`.
    fn pipeline_threads(
        &self,
        pipeline: &Pipeline,
        source_state: &dyn GlobalSourceState,
    ) -> usize {
        let parallel = pipeline.source().parallel_source()
            && pipeline.sink().parallel_sink()
            && pipeline
                .operators()
                .iter()
                .all(|operator| operator.parallel_operator());
        if !parallel {
            return 1;
        }
        self.num_threads.min(source_state.max_threads()).max(1)
    }

    fn finalize(
        &self,
        meta: &MetaPipeline,
        registry: &Arc<ExecutionRegistry>,
    ) -> Result<()> {
        let ctx = self.context(0, self.num_threads, registry);
        let sink = meta.sink();
        let state = registry.get_or_create_sink_state(sink.as_ref(), &ctx)?;
        let start = Instant::now();
        let result = sink.finalize(&ctx, state.as_ref());
        registry.metrics(sink.as_ref()).add_elapsed(start.elapsed());
        registry.set_finalize_result(sink.as_ref(), result?);
        Ok(())
    }
}

/// Bounded queue of result batches between the [`CollectorSink`] and the
/// [`PipelineStream`].
#[derive(Debug)]
struct BatchQueue {
    capacity: usize,
    inner: Mutex<BatchQueueInner>,
}

#[derive(Debug, Default)]
struct BatchQueueInner {
    batches: VecDeque<RecordBatch>,
    consumer: Option<Waker>,
    producers: Vec<Waker>,
    closed: bool,
}

enum PushResult {
    Pushed,
    Full(RecordBatch),
    Closed,
}

impl BatchQueue {
    fn new(capacity: usize) -> Self {
        Self {
            capacity: capacity.max(1),
            inner: Mutex::new(BatchQueueInner::default()),
        }
    }

    /// Push `batch`; if the queue is full, register `waker` to be woken once
    /// there is room.
    fn push(&self, batch: RecordBatch, waker: &Waker) -> PushResult {
        let mut inner = self.inner.lock();
        if inner.closed {
            return PushResult::Closed;
        }
        if inner.batches.len() >= self.capacity {
            if !inner.producers.iter().any(|w| w.will_wake(waker)) {
                inner.producers.push(waker.clone());
            }
            return PushResult::Full(batch);
        }
        inner.batches.push_back(batch);
        if let Some(consumer) = inner.consumer.take() {
            consumer.wake();
        }
        PushResult::Pushed
    }

    /// Pop a batch; if there is none, register the consumer `waker`.
    fn pop(&self, waker: &Waker) -> Option<RecordBatch> {
        let mut inner = self.inner.lock();
        match inner.batches.pop_front() {
            Some(batch) => {
                for producer in inner.producers.drain(..) {
                    producer.wake();
                }
                Some(batch)
            }
            None => {
                inner.consumer = Some(waker.clone());
                None
            }
        }
    }

    fn close(&self) {
        let mut inner = self.inner.lock();
        inner.closed = true;
        inner.batches.clear();
        for producer in inner.producers.drain(..) {
            producer.wake();
        }
    }
}

/// The sink of the root pipeline: compacts every chunk into a
/// [`RecordBatch`] and hands it to the [`PipelineStream`] of the execution.
#[derive(Debug)]
pub struct CollectorSink {
    children: Vec<Arc<dyn PhysicalOperator>>,
    queue: Arc<BatchQueue>,
}

impl CollectorSink {
    fn new(input: Arc<dyn PhysicalOperator>, queue: Arc<BatchQueue>) -> Self {
        Self {
            children: vec![input],
            queue,
        }
    }
}

#[derive(Default)]
struct CollectorLocalState {
    /// A compacted batch the queue did not accept yet.
    pending: Option<RecordBatch>,
}

impl LocalSinkState for CollectorLocalState {}

impl PhysicalOperator for CollectorSink {
    fn name(&self) -> &str {
        "CollectorSink"
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
        _gstate: &dyn GlobalSinkState,
    ) -> Result<Box<dyn LocalSinkState>> {
        Ok(Box::new(CollectorLocalState::default()))
    }

    fn sink(
        &self,
        ctx: &ExecutionContext,
        input: &DataChunk,
        _gstate: &dyn GlobalSinkState,
        lstate: &mut dyn LocalSinkState,
    ) -> Result<SinkResult> {
        let state = downcast_state_mut::<CollectorLocalState>(
            lstate.as_any_mut(),
            "CollectorSink",
        )?;
        let batch = match state.pending.take() {
            Some(batch) => batch,
            None => input.compact()?,
        };
        match self.queue.push(batch, ctx.waker()) {
            PushResult::Pushed => Ok(SinkResult::NeedMoreInput),
            PushResult::Full(batch) => {
                state.pending = Some(batch);
                Ok(SinkResult::Blocked(TaskInterrupt::waker_registered()))
            }
            // The consumer went away: no more output is needed.
            PushResult::Closed => Ok(SinkResult::Finished),
        }
    }
}

enum StreamState {
    /// The execution has not started yet.
    Pending(Executor),
    Running(SpawnedTask<Result<()>>),
    Done,
}

/// The output of [`Executor::execute_stream`].
pub struct PipelineStream {
    schema: SchemaRef,
    metrics: Arc<ExecutionMetrics>,
    queue: Arc<BatchQueue>,
    state: StreamState,
}

impl PipelineStream {
    /// The runtime metrics of the operators of this execution.
    pub fn metrics(&self) -> &Arc<ExecutionMetrics> {
        &self.metrics
    }
}

impl fmt::Debug for PipelineStream {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PipelineStream")
            .field("schema", &self.schema)
            .finish()
    }
}

impl Stream for PipelineStream {
    type Item = Result<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        loop {
            match std::mem::replace(&mut this.state, StreamState::Done) {
                StreamState::Pending(executor) => {
                    this.state = StreamState::Running(SpawnedTask::spawn(async move {
                        executor.execute().await
                    }));
                }
                StreamState::Running(mut task) => {
                    if let Some(batch) = this.queue.pop(cx.waker()) {
                        this.state = StreamState::Running(task);
                        return Poll::Ready(Some(Ok(batch)));
                    }
                    match Pin::new(&mut task).poll(cx) {
                        Poll::Pending => {
                            this.state = StreamState::Running(task);
                            return Poll::Pending;
                        }
                        // All producers are done: drain the remaining batches.
                        Poll::Ready(Ok(Ok(()))) => {}
                        Poll::Ready(Ok(Err(e))) => {
                            this.queue.close();
                            return Poll::Ready(Some(Err(e)));
                        }
                        Poll::Ready(Err(e)) if e.is_panic() => {
                            std::panic::resume_unwind(e.into_panic())
                        }
                        Poll::Ready(Err(e)) => {
                            return Poll::Ready(Some(Err(
                                DataFusionError::ExecutionJoin(Box::new(e)),
                            )));
                        }
                    }
                }
                StreamState::Done => {
                    return Poll::Ready(this.queue.pop(cx.waker()).map(Ok));
                }
            }
        }
    }
}

impl RecordBatchStream for PipelineStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

impl Drop for PipelineStream {
    fn drop(&mut self) {
        // Producers that are still running (the task is aborted when it is
        // dropped) must not block on a queue nobody reads.
        self.queue.close();
    }
}
