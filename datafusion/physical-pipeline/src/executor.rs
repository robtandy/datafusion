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

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

    use arrow::array::{AsArray, BooleanArray, Int64Array};
    use arrow::datatypes::{DataType, Field, Int64Type, Schema};
    use datafusion_common::exec_err;

    use crate::chunk::SelectionVector;
    use crate::operator::{GlobalSinkState, downcast_state};
    use crate::pipeline::{MetaPipelineBuilder, PipelineBuilder};

    // ---------------------------------------------------------------------
    // Test operators
    // ---------------------------------------------------------------------

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]))
    }

    /// `num_batches` batches of `rows` rows with the values `start..`.
    fn batches(start: i64, num_batches: usize, rows: usize) -> Vec<RecordBatch> {
        (0..num_batches)
            .map(|b| {
                let first = start + (b * rows) as i64;
                RecordBatch::try_new(
                    schema(),
                    vec![Arc::new(Int64Array::from_iter_values(
                        first..first + rows as i64,
                    ))],
                )
                .unwrap()
            })
            .collect()
    }

    /// All values of `batches`, sorted.
    fn sorted_values(batches: &[RecordBatch]) -> Vec<i64> {
        let mut values = values(batches);
        values.sort_unstable();
        values
    }

    fn values(batches: &[RecordBatch]) -> Vec<i64> {
        batches
            .iter()
            .flat_map(|b| b.column(0).as_primitive::<Int64Type>().values().to_vec())
            .collect()
    }

    fn task_ctx() -> Arc<TaskContext> {
        Arc::new(TaskContext::default())
    }

    async fn run(
        root: Arc<dyn PhysicalOperator>,
        threads: usize,
    ) -> Result<Vec<RecordBatch>> {
        Executor::collect(root, task_ctx(), threads).await
    }

    fn graph(root: Arc<dyn PhysicalOperator>) -> Result<PipelineGraph> {
        let queue = Arc::new(BatchQueue::new(1));
        PipelineGraph::try_new(Arc::new(CollectorSink::new(root, queue)))
    }

    /// A parallel source that hands out `batches`, optionally interrupting
    /// itself with all kinds of [`TaskInterrupt`]s.
    #[derive(Debug)]
    struct VecSource {
        batches: Vec<RecordBatch>,
        interrupts: bool,
        max_threads: usize,
        calls: AtomicUsize,
    }

    impl VecSource {
        fn new(batches: Vec<RecordBatch>) -> Self {
            Self {
                batches,
                interrupts: false,
                max_threads: usize::MAX,
                calls: AtomicUsize::new(0),
            }
        }

        fn with_interrupts(mut self) -> Self {
            self.interrupts = true;
            self
        }

        fn with_max_threads(mut self, max_threads: usize) -> Self {
            self.max_threads = max_threads;
            self
        }
    }

    struct VecSourceGlobal {
        next: AtomicUsize,
        max_threads: usize,
    }

    impl GlobalSourceState for VecSourceGlobal {
        fn max_threads(&self) -> usize {
            self.max_threads
        }
    }

    #[derive(Default)]
    struct VecSourceLocal {
        calls: usize,
    }

    impl LocalSourceState for VecSourceLocal {}

    impl PhysicalOperator for VecSource {
        fn name(&self) -> &str {
            "VecSource"
        }

        fn schema(&self) -> SchemaRef {
            schema()
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
            Ok(Arc::new(VecSourceGlobal {
                next: AtomicUsize::new(0),
                max_threads: self.max_threads,
            }))
        }

        fn get_local_source_state(
            &self,
            _ctx: &ExecutionContext,
            _gstate: &dyn GlobalSourceState,
        ) -> Result<Box<dyn LocalSourceState>> {
            Ok(Box::new(VecSourceLocal::default()))
        }

        fn get_data(
            &self,
            ctx: &ExecutionContext,
            gstate: &dyn GlobalSourceState,
            lstate: &mut dyn LocalSourceState,
        ) -> Result<SourceResult> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            let gstate = downcast_state::<VecSourceGlobal>(gstate.as_any(), self.name())?;
            let lstate = lstate.downcast_mut::<VecSourceLocal>().unwrap();
            if self.interrupts {
                lstate.calls += 1;
                match lstate.calls % 4 {
                    1 => return Ok(SourceResult::Blocked(TaskInterrupt::yield_now())),
                    2 => {
                        return Ok(SourceResult::Blocked(TaskInterrupt::wait_for(
                            tokio::task::yield_now(),
                        )));
                    }
                    3 => {
                        // The resource "becomes ready" right away.
                        ctx.waker().wake_by_ref();
                        return Ok(SourceResult::Blocked(
                            TaskInterrupt::waker_registered(),
                        ));
                    }
                    _ => {}
                }
            }
            let index = gstate.next.fetch_add(1, Ordering::Relaxed);
            Ok(match self.batches.get(index) {
                Some(batch) => {
                    SourceResult::Chunk(DataChunk::from_record_batch(batch.clone()))
                }
                None => SourceResult::Finished,
            })
        }
    }

    /// Emits every input chunk `times` times, using `HaveMoreOutput`.
    #[derive(Debug)]
    struct Repeat {
        children: Vec<Arc<dyn PhysicalOperator>>,
        times: usize,
    }

    #[derive(Default)]
    struct RepeatState {
        emitted: usize,
    }

    impl OperatorState for RepeatState {}

    impl PhysicalOperator for Repeat {
        fn name(&self) -> &str {
            "Repeat"
        }

        fn schema(&self) -> SchemaRef {
            self.children[0].schema()
        }

        fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
            &self.children
        }

        fn get_operator_state(
            &self,
            _ctx: &ExecutionContext,
            _gstate: &dyn GlobalOperatorState,
        ) -> Result<Box<dyn OperatorState>> {
            Ok(Box::new(RepeatState::default()))
        }

        fn execute(
            &self,
            _ctx: &ExecutionContext,
            input: &DataChunk,
            output: &mut DataChunk,
            _gstate: &dyn GlobalOperatorState,
            lstate: &mut dyn OperatorState,
        ) -> Result<OperatorResult> {
            let state = lstate.downcast_mut::<RepeatState>().unwrap();
            *output = input.clone();
            state.emitted += 1;
            if state.emitted < self.times {
                Ok(OperatorResult::HaveMoreOutput)
            } else {
                state.emitted = 0;
                Ok(OperatorResult::NeedMoreInput)
            }
        }
    }

    /// Keeps the rows whose value is a multiple of `divisor`, by attaching a
    /// selection vector.
    #[derive(Debug)]
    struct KeepMultiples {
        children: Vec<Arc<dyn PhysicalOperator>>,
        divisor: i64,
    }

    impl PhysicalOperator for KeepMultiples {
        fn name(&self) -> &str {
            "KeepMultiples"
        }

        fn schema(&self) -> SchemaRef {
            self.children[0].schema()
        }

        fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
            &self.children
        }

        fn execute(
            &self,
            _ctx: &ExecutionContext,
            input: &DataChunk,
            output: &mut DataChunk,
            _gstate: &dyn GlobalOperatorState,
            _lstate: &mut dyn OperatorState,
        ) -> Result<OperatorResult> {
            let column = input.compacted_column(0)?;
            let mask = BooleanArray::from_iter(
                column
                    .as_primitive::<Int64Type>()
                    .iter()
                    .map(|v| v.map(|v| v % self.divisor == 0)),
            );
            *output = input
                .clone()
                .with_selection(SelectionVector::from_boolean_mask(&mask))?;
            Ok(OperatorResult::NeedMoreInput)
        }
    }

    /// Buffers all of its input and emits it from `final_execute`, one
    /// buffered chunk per call.
    #[derive(Debug)]
    struct BufferAll {
        children: Vec<Arc<dyn PhysicalOperator>>,
    }

    #[derive(Default)]
    struct BufferState {
        chunks: Vec<DataChunk>,
    }

    impl OperatorState for BufferState {}

    impl PhysicalOperator for BufferAll {
        fn name(&self) -> &str {
            "BufferAll"
        }

        fn schema(&self) -> SchemaRef {
            self.children[0].schema()
        }

        fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
            &self.children
        }

        fn get_operator_state(
            &self,
            _ctx: &ExecutionContext,
            _gstate: &dyn GlobalOperatorState,
        ) -> Result<Box<dyn OperatorState>> {
            Ok(Box::new(BufferState::default()))
        }

        fn execute(
            &self,
            _ctx: &ExecutionContext,
            input: &DataChunk,
            _output: &mut DataChunk,
            _gstate: &dyn GlobalOperatorState,
            lstate: &mut dyn OperatorState,
        ) -> Result<OperatorResult> {
            let state = lstate.downcast_mut::<BufferState>().unwrap();
            state.chunks.push(input.clone());
            Ok(OperatorResult::NeedMoreInput)
        }

        fn requires_final_execute(&self) -> bool {
            true
        }

        fn final_execute(
            &self,
            _ctx: &ExecutionContext,
            output: &mut DataChunk,
            _gstate: &dyn GlobalOperatorState,
            lstate: &mut dyn OperatorState,
        ) -> Result<OperatorFinalizeResult> {
            let state = lstate.downcast_mut::<BufferState>().unwrap();
            if let Some(chunk) = state.chunks.pop() {
                *output = chunk;
            }
            Ok(if state.chunks.is_empty() {
                OperatorFinalizeResult::Finished
            } else {
                OperatorFinalizeResult::HaveMoreOutput
            })
        }
    }

    /// Passes through at most `limit` rows in total (over all threads and
    /// pipelines), then returns `Finished`.
    #[derive(Debug)]
    struct Limit {
        children: Vec<Arc<dyn PhysicalOperator>>,
        limit: usize,
    }

    struct LimitState {
        seen: AtomicUsize,
    }

    impl GlobalOperatorState for LimitState {}

    impl PhysicalOperator for Limit {
        fn name(&self) -> &str {
            "Limit"
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
            Ok(Arc::new(LimitState {
                seen: AtomicUsize::new(0),
            }))
        }

        fn execute(
            &self,
            _ctx: &ExecutionContext,
            input: &DataChunk,
            output: &mut DataChunk,
            gstate: &dyn GlobalOperatorState,
            _lstate: &mut dyn OperatorState,
        ) -> Result<OperatorResult> {
            let state = gstate.downcast_ref::<LimitState>().unwrap();
            let seen = state.seen.fetch_add(input.len(), Ordering::Relaxed);
            if seen >= self.limit {
                return Ok(OperatorResult::Finished);
            }
            let take = input.len().min(self.limit - seen);
            *output = input.slice(0, take);
            Ok(if seen + input.len() >= self.limit {
                OperatorResult::Finished
            } else {
                OperatorResult::NeedMoreInput
            })
        }
    }

    /// Returns `Blocked` before letting every chunk through.
    #[derive(Debug)]
    struct BlockOnce {
        children: Vec<Arc<dyn PhysicalOperator>>,
    }

    #[derive(Default)]
    struct BlockOnceState {
        blocked: bool,
    }

    impl OperatorState for BlockOnceState {}

    impl PhysicalOperator for BlockOnce {
        fn name(&self) -> &str {
            "BlockOnce"
        }

        fn schema(&self) -> SchemaRef {
            self.children[0].schema()
        }

        fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
            &self.children
        }

        fn get_operator_state(
            &self,
            _ctx: &ExecutionContext,
            _gstate: &dyn GlobalOperatorState,
        ) -> Result<Box<dyn OperatorState>> {
            Ok(Box::new(BlockOnceState::default()))
        }

        fn execute(
            &self,
            _ctx: &ExecutionContext,
            input: &DataChunk,
            output: &mut DataChunk,
            _gstate: &dyn GlobalOperatorState,
            lstate: &mut dyn OperatorState,
        ) -> Result<OperatorResult> {
            let state = lstate.downcast_mut::<BlockOnceState>().unwrap();
            state.blocked = !state.blocked;
            if state.blocked {
                // Write garbage that must be ignored.
                *output = input.clone();
                return Ok(OperatorResult::Blocked(TaskInterrupt::wait_for(
                    tokio::task::yield_now(),
                )));
            }
            *output = input.clone();
            Ok(OperatorResult::NeedMoreInput)
        }
    }

    /// Fails on the `fail_at`-th chunk.
    #[derive(Debug)]
    struct FailAt {
        children: Vec<Arc<dyn PhysicalOperator>>,
        fail_at: usize,
        seen: AtomicUsize,
    }

    impl PhysicalOperator for FailAt {
        fn name(&self) -> &str {
            "FailAt"
        }

        fn schema(&self) -> SchemaRef {
            self.children[0].schema()
        }

        fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
            &self.children
        }

        fn execute(
            &self,
            _ctx: &ExecutionContext,
            input: &DataChunk,
            output: &mut DataChunk,
            _gstate: &dyn GlobalOperatorState,
            _lstate: &mut dyn OperatorState,
        ) -> Result<OperatorResult> {
            if self.seen.fetch_add(1, Ordering::Relaxed) + 1 == self.fail_at {
                return exec_err!("FailAt failed as requested");
            }
            *output = input.clone();
            Ok(OperatorResult::NeedMoreInput)
        }
    }

    /// A pipeline breaker: collects its input as a parallel sink and emits it
    /// again as a parallel source.
    #[derive(Debug)]
    struct Materialize {
        children: Vec<Arc<dyn PhysicalOperator>>,
        no_output_if_empty: bool,
        combines: AtomicUsize,
        finalizes: AtomicUsize,
        get_data_calls: AtomicUsize,
    }

    impl Materialize {
        fn new(child: Arc<dyn PhysicalOperator>) -> Self {
            Self {
                children: vec![child],
                no_output_if_empty: false,
                combines: AtomicUsize::new(0),
                finalizes: AtomicUsize::new(0),
                get_data_calls: AtomicUsize::new(0),
            }
        }
    }

    #[derive(Default)]
    struct MaterializeGlobal {
        chunks: Mutex<Vec<DataChunk>>,
        finalized: AtomicBool,
    }

    impl GlobalSinkState for MaterializeGlobal {}

    #[derive(Default)]
    struct MaterializeLocal {
        chunks: Vec<DataChunk>,
    }

    impl LocalSinkState for MaterializeLocal {}

    struct MaterializeSource {
        sink: Arc<MaterializeGlobal>,
        next: AtomicUsize,
    }

    impl GlobalSourceState for MaterializeSource {}

    impl PhysicalOperator for Materialize {
        fn name(&self) -> &str {
            "Materialize"
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
            Ok(Arc::new(MaterializeGlobal::default()))
        }

        fn get_local_sink_state(
            &self,
            _ctx: &ExecutionContext,
            _gstate: &dyn GlobalSinkState,
        ) -> Result<Box<dyn LocalSinkState>> {
            Ok(Box::new(MaterializeLocal::default()))
        }

        fn sink(
            &self,
            _ctx: &ExecutionContext,
            input: &DataChunk,
            _gstate: &dyn GlobalSinkState,
            lstate: &mut dyn LocalSinkState,
        ) -> Result<SinkResult> {
            let state = lstate.downcast_mut::<MaterializeLocal>().unwrap();
            state.chunks.push(input.clone());
            Ok(SinkResult::NeedMoreInput)
        }

        fn combine(
            &self,
            _ctx: &ExecutionContext,
            gstate: &dyn GlobalSinkState,
            lstate: &mut dyn LocalSinkState,
        ) -> Result<()> {
            self.combines.fetch_add(1, Ordering::Relaxed);
            let gstate = gstate.downcast_ref::<MaterializeGlobal>().unwrap();
            let lstate = lstate.downcast_mut::<MaterializeLocal>().unwrap();
            gstate.chunks.lock().append(&mut lstate.chunks);
            Ok(())
        }

        fn finalize(
            &self,
            _ctx: &ExecutionContext,
            gstate: &dyn GlobalSinkState,
        ) -> Result<SinkFinalize> {
            self.finalizes.fetch_add(1, Ordering::Relaxed);
            let gstate = gstate.downcast_ref::<MaterializeGlobal>().unwrap();
            gstate.finalized.store(true, Ordering::Relaxed);
            if self.no_output_if_empty && gstate.chunks.lock().is_empty() {
                Ok(SinkFinalize::NoOutputPossible)
            } else {
                Ok(SinkFinalize::Ready)
            }
        }

        fn is_source(&self) -> bool {
            true
        }

        fn parallel_source(&self) -> bool {
            true
        }

        fn get_global_source_state(
            &self,
            ctx: &ExecutionContext,
        ) -> Result<Arc<dyn GlobalSourceState>> {
            let sink = ctx
                .global_sink_state(self)
                .and_then(|state| state.downcast_arc::<MaterializeGlobal>())
                .expect("the sink state exists");
            assert!(
                sink.finalized.load(Ordering::Relaxed),
                "the source must only run once the sink is finalized"
            );
            Ok(Arc::new(MaterializeSource {
                sink,
                next: AtomicUsize::new(0),
            }))
        }

        fn get_data(
            &self,
            _ctx: &ExecutionContext,
            gstate: &dyn GlobalSourceState,
            _lstate: &mut dyn LocalSourceState,
        ) -> Result<SourceResult> {
            self.get_data_calls.fetch_add(1, Ordering::Relaxed);
            let gstate = gstate.downcast_ref::<MaterializeSource>().unwrap();
            let index = gstate.next.fetch_add(1, Ordering::Relaxed);
            Ok(match gstate.sink.chunks.lock().get(index) {
                Some(chunk) => SourceResult::Chunk(chunk.clone()),
                None => SourceResult::Finished,
            })
        }
    }

    /// A pipeline breaker that counts its input rows and emits the count as
    /// a single row, from a single-threaded source.
    #[derive(Debug)]
    struct Count {
        children: Vec<Arc<dyn PhysicalOperator>>,
    }

    #[derive(Default)]
    struct CountGlobal {
        rows: AtomicUsize,
    }

    impl GlobalSinkState for CountGlobal {}

    #[derive(Default)]
    struct CountSource {
        done: AtomicBool,
    }

    impl GlobalSourceState for CountSource {}

    impl PhysicalOperator for Count {
        fn name(&self) -> &str {
            "Count"
        }

        fn schema(&self) -> SchemaRef {
            schema()
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
            Ok(Arc::new(CountGlobal::default()))
        }

        fn sink(
            &self,
            _ctx: &ExecutionContext,
            input: &DataChunk,
            gstate: &dyn GlobalSinkState,
            _lstate: &mut dyn LocalSinkState,
        ) -> Result<SinkResult> {
            let gstate = gstate.downcast_ref::<CountGlobal>().unwrap();
            gstate.rows.fetch_add(input.len(), Ordering::Relaxed);
            Ok(SinkResult::NeedMoreInput)
        }

        fn is_source(&self) -> bool {
            true
        }

        fn get_global_source_state(
            &self,
            _ctx: &ExecutionContext,
        ) -> Result<Arc<dyn GlobalSourceState>> {
            Ok(Arc::new(CountSource::default()))
        }

        fn get_data(
            &self,
            ctx: &ExecutionContext,
            gstate: &dyn GlobalSourceState,
            _lstate: &mut dyn LocalSourceState,
        ) -> Result<SourceResult> {
            let gstate = gstate.downcast_ref::<CountSource>().unwrap();
            if gstate.done.swap(true, Ordering::Relaxed) {
                return Ok(SourceResult::Finished);
            }
            let sink = ctx.global_sink_state(self).unwrap();
            let rows = sink
                .downcast_ref::<CountGlobal>()
                .unwrap()
                .rows
                .load(Ordering::Relaxed);
            let batch = RecordBatch::try_new(
                schema(),
                vec![Arc::new(Int64Array::from(vec![rows as i64]))],
            )?;
            Ok(SourceResult::Chunk(DataChunk::from_record_batch(batch)))
        }
    }

    /// A sink that accepts `limit` chunks and then returns `Finished`; emits
    /// how many chunks it accepted.
    #[derive(Debug)]
    struct StopAfter {
        children: Vec<Arc<dyn PhysicalOperator>>,
        limit: usize,
    }

    #[derive(Default)]
    struct StopAfterGlobal {
        chunks: AtomicUsize,
    }

    impl GlobalSinkState for StopAfterGlobal {}

    impl PhysicalOperator for StopAfter {
        fn name(&self) -> &str {
            "StopAfter"
        }

        fn schema(&self) -> SchemaRef {
            schema()
        }

        fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
            &self.children
        }

        fn is_sink(&self) -> bool {
            true
        }

        fn get_global_sink_state(
            &self,
            _ctx: &ExecutionContext,
        ) -> Result<Arc<dyn GlobalSinkState>> {
            Ok(Arc::new(StopAfterGlobal::default()))
        }

        fn sink(
            &self,
            _ctx: &ExecutionContext,
            _input: &DataChunk,
            gstate: &dyn GlobalSinkState,
            _lstate: &mut dyn LocalSinkState,
        ) -> Result<SinkResult> {
            let gstate = gstate.downcast_ref::<StopAfterGlobal>().unwrap();
            let chunks = gstate.chunks.fetch_add(1, Ordering::Relaxed) + 1;
            Ok(if chunks >= self.limit {
                SinkResult::Finished
            } else {
                SinkResult::NeedMoreInput
            })
        }

        fn is_source(&self) -> bool {
            true
        }

        fn get_data(
            &self,
            ctx: &ExecutionContext,
            _gstate: &dyn GlobalSourceState,
            _lstate: &mut dyn LocalSourceState,
        ) -> Result<SourceResult> {
            let sink = ctx.global_sink_state(self).unwrap();
            let sink = sink.downcast_ref::<StopAfterGlobal>().unwrap();
            let chunks = sink.chunks.swap(0, Ordering::Relaxed);
            if chunks == 0 {
                return Ok(SourceResult::Finished);
            }
            let batch = RecordBatch::try_new(
                schema(),
                vec![Arc::new(Int64Array::from(vec![chunks as i64]))],
            )?;
            Ok(SourceResult::Chunk(DataChunk::from_record_batch(batch)))
        }
    }

    /// Concatenates the output of its children through a union pipeline.
    #[derive(Debug)]
    struct Union {
        children: Vec<Arc<dyn PhysicalOperator>>,
        order_matters: bool,
    }

    impl PhysicalOperator for Union {
        fn name(&self) -> &str {
            "Union"
        }

        fn schema(&self) -> SchemaRef {
            self.children[0].schema()
        }

        fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
            &self.children
        }

        fn build_pipelines(
            &self,
            current: &mut PipelineBuilder,
            meta: &mut MetaPipelineBuilder<'_>,
        ) -> Result<()> {
            let mut union_pipeline =
                meta.create_union_pipeline(current, self.order_matters)?;
            meta.build(current, &self.children[0])?;
            meta.build(&mut union_pipeline, &self.children[1])
        }
    }

    /// A "right outer join"-like operator: it collects its build side
    /// (`children[1]`) as a sink, passes its probe side (`children[0]`)
    /// through as an operator, and once the probe pipeline completed it emits
    /// the build rows plus one row with the number of probed rows from a
    /// child pipeline.
    #[derive(Debug)]
    struct ProbeThenScan {
        children: Vec<Arc<dyn PhysicalOperator>>,
    }

    #[derive(Default)]
    struct ProbeThenScanGlobal {
        build: Mutex<Vec<DataChunk>>,
        probed: AtomicUsize,
    }

    impl GlobalSinkState for ProbeThenScanGlobal {}

    #[derive(Default)]
    struct ProbeThenScanSource {
        done: AtomicBool,
    }

    impl GlobalSourceState for ProbeThenScanSource {}

    impl ProbeThenScan {
        fn sink_state(&self, ctx: &ExecutionContext) -> Arc<ProbeThenScanGlobal> {
            ctx.global_sink_state(self)
                .and_then(|state| state.downcast_arc::<ProbeThenScanGlobal>())
                .unwrap()
        }
    }

    impl PhysicalOperator for ProbeThenScan {
        fn name(&self) -> &str {
            "ProbeThenScan"
        }

        fn schema(&self) -> SchemaRef {
            schema()
        }

        fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
            &self.children
        }

        fn build_pipelines(
            &self,
            current: &mut PipelineBuilder,
            meta: &mut MetaPipelineBuilder<'_>,
        ) -> Result<()> {
            let this = meta.current_operator()?;
            meta.add_operator(current, Arc::clone(&this))?;
            let last_pipeline = *meta.pipelines().last().unwrap();
            {
                let (mut build_meta, mut build_pipeline) =
                    meta.create_child_meta_pipeline(current, Arc::clone(&this))?;
                build_meta.build(&mut build_pipeline, &self.children[1])?;
            }
            meta.build(current, &self.children[0])?;
            meta.create_child_pipeline(current, this, &last_pipeline)?;
            Ok(())
        }

        fn execute(
            &self,
            ctx: &ExecutionContext,
            input: &DataChunk,
            output: &mut DataChunk,
            _gstate: &dyn GlobalOperatorState,
            _lstate: &mut dyn OperatorState,
        ) -> Result<OperatorResult> {
            self.sink_state(ctx)
                .probed
                .fetch_add(input.len(), Ordering::Relaxed);
            *output = input.clone();
            Ok(OperatorResult::NeedMoreInput)
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
            Ok(Arc::new(ProbeThenScanGlobal::default()))
        }

        fn sink(
            &self,
            _ctx: &ExecutionContext,
            input: &DataChunk,
            gstate: &dyn GlobalSinkState,
            _lstate: &mut dyn LocalSinkState,
        ) -> Result<SinkResult> {
            let gstate = gstate.downcast_ref::<ProbeThenScanGlobal>().unwrap();
            gstate.build.lock().push(input.clone());
            Ok(SinkResult::NeedMoreInput)
        }

        fn is_source(&self) -> bool {
            true
        }

        fn get_global_source_state(
            &self,
            _ctx: &ExecutionContext,
        ) -> Result<Arc<dyn GlobalSourceState>> {
            Ok(Arc::new(ProbeThenScanSource::default()))
        }

        fn get_data(
            &self,
            ctx: &ExecutionContext,
            gstate: &dyn GlobalSourceState,
            _lstate: &mut dyn LocalSourceState,
        ) -> Result<SourceResult> {
            let gstate = gstate.downcast_ref::<ProbeThenScanSource>().unwrap();
            if gstate.done.swap(true, Ordering::Relaxed) {
                return Ok(SourceResult::Finished);
            }
            let sink = self.sink_state(ctx);
            let mut batches = sink
                .build
                .lock()
                .iter()
                .map(DataChunk::compact)
                .collect::<Result<Vec<_>>>()?;
            batches.push(RecordBatch::try_new(
                schema(),
                vec![Arc::new(Int64Array::from(vec![
                    sink.probed.load(Ordering::Relaxed) as i64,
                ]))],
            )?);
            let batch = arrow::compute::concat_batches(&schema(), &batches)?;
            Ok(SourceResult::Chunk(DataChunk::from_record_batch(batch)))
        }
    }

    fn source(batches: Vec<RecordBatch>) -> Arc<dyn PhysicalOperator> {
        Arc::new(VecSource::new(batches))
    }

    // ---------------------------------------------------------------------
    // Pipeline construction
    // ---------------------------------------------------------------------

    #[test]
    fn streaming_operators_form_a_single_pipeline() -> Result<()> {
        let root: Arc<dyn PhysicalOperator> = Arc::new(Repeat {
            children: vec![Arc::new(KeepMultiples {
                children: vec![source(batches(0, 1, 4))],
                divisor: 2,
            })],
            times: 2,
        });
        let graph = graph(root)?;
        assert_eq!(graph.pipelines().len(), 1);
        assert_eq!(graph.meta_pipelines().len(), 1);
        let pipeline = graph.pipeline(0);
        assert_eq!(pipeline.source().name(), "VecSource");
        let names = pipeline
            .operators()
            .iter()
            .map(|op| op.name())
            .collect::<Vec<_>>();
        // Execution order: from the source to the sink.
        assert_eq!(names, vec!["KeepMultiples", "Repeat"]);
        assert_eq!(pipeline.sink().name(), "CollectorSink");
        assert!(pipeline.dependencies().is_empty());
        assert!(pipeline.meta_dependencies().is_empty());
        assert_eq!(
            graph.to_string(),
            "MetaPipeline 0 (sink: CollectorSink)\n  Pipeline 0: VecSource -> KeepMultiples -> Repeat -> CollectorSink\n"
        );
        assert_eq!(graph.operators().len(), 4);
        Ok(())
    }

    #[test]
    fn pipeline_breakers_create_child_meta_pipelines() -> Result<()> {
        let inner: Arc<dyn PhysicalOperator> =
            Arc::new(Materialize::new(source(batches(0, 1, 4))));
        let root: Arc<dyn PhysicalOperator> = Arc::new(Repeat {
            children: vec![Arc::new(Count {
                children: vec![Arc::new(Repeat {
                    children: vec![inner],
                    times: 1,
                })],
            })],
            times: 1,
        });
        let graph = graph(root)?;
        assert_eq!(
            graph.to_string(),
            "MetaPipeline 0 (sink: CollectorSink) children: [1]\n  \
             Pipeline 0: Count -> Repeat -> CollectorSink depends on meta pipelines [1]\n\
             MetaPipeline 1 (sink: Count) children: [2]\n  \
             Pipeline 1: Materialize -> Repeat -> Count depends on meta pipelines [2]\n\
             MetaPipeline 2 (sink: Materialize)\n  \
             Pipeline 2: VecSource -> Materialize\n"
        );
        assert_eq!(graph.root().id(), 0);
        assert_eq!(graph.meta_pipeline(2).parent(), Some(1));
        assert_eq!(graph.meta_pipeline(1).children(), &[2]);
        Ok(())
    }

    #[test]
    fn union_and_child_pipelines() -> Result<()> {
        let union: Arc<dyn PhysicalOperator> = Arc::new(Union {
            children: vec![source(batches(0, 1, 2)), source(batches(10, 1, 2))],
            order_matters: true,
        });
        let root: Arc<dyn PhysicalOperator> = Arc::new(Repeat {
            children: vec![Arc::new(ProbeThenScan {
                children: vec![union, source(batches(100, 1, 2))],
            })],
            times: 1,
        });
        let graph = graph(root)?;
        assert_eq!(
            graph.to_string(),
            "MetaPipeline 0 (sink: CollectorSink) children: [1]\n  \
             Pipeline 0: VecSource -> ProbeThenScan -> Repeat -> CollectorSink depends on meta pipelines [1]\n  \
             Pipeline 2: VecSource -> ProbeThenScan -> Repeat -> CollectorSink depends on pipelines [0] depends on meta pipelines [1]\n  \
             Pipeline 3: ProbeThenScan -> Repeat -> CollectorSink depends on pipelines [0, 2]\n\
             MetaPipeline 1 (sink: ProbeThenScan)\n  \
             Pipeline 1: VecSource -> ProbeThenScan\n"
        );
        Ok(())
    }

    #[test]
    fn invalid_operator_trees_are_rejected() {
        /// A leaf that is not a source.
        #[derive(Debug)]
        struct NotASource;

        impl PhysicalOperator for NotASource {
            fn name(&self) -> &str {
                "NotASource"
            }

            fn schema(&self) -> SchemaRef {
                schema()
            }

            fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
                &[]
            }
        }

        /// An operator with two children that does not build its pipelines.
        #[derive(Debug)]
        struct TwoChildren {
            children: Vec<Arc<dyn PhysicalOperator>>,
        }

        impl PhysicalOperator for TwoChildren {
            fn name(&self) -> &str {
                "TwoChildren"
            }

            fn schema(&self) -> SchemaRef {
                schema()
            }

            fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
                &self.children
            }
        }

        let err = graph(Arc::new(NotASource)).unwrap_err();
        assert!(
            err.to_string().contains("NotASource cannot be the source"),
            "{err}"
        );

        let err = graph(Arc::new(TwoChildren {
            children: vec![source(vec![]), source(vec![])],
        }))
        .unwrap_err();
        assert!(
            err.to_string().contains("must implement build_pipelines"),
            "{err}"
        );

        let err = PipelineGraph::try_new(source(vec![])).unwrap_err();
        assert!(err.to_string().contains("must be a sink"), "{err}");
    }

    // ---------------------------------------------------------------------
    // Execution
    // ---------------------------------------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn executes_parallel_pipeline() -> Result<()> {
        let input = batches(0, 200, 100);
        for threads in [1, 3, 8] {
            let output = run(source(input.clone()), threads).await?;
            assert_eq!(sorted_values(&output), (0..20_000).collect::<Vec<_>>());
            assert!(output.iter().all(|batch| batch.schema() == schema()));
        }
        Ok(())
    }

    #[test]
    fn executes_on_a_current_thread_runtime() -> Result<()> {
        let output = tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(run(source(batches(0, 200, 100)), 4))?;
        assert_eq!(sorted_values(&output), (0..20_000).collect::<Vec<_>>());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn empty_source_produces_no_output() -> Result<()> {
        assert!(run(source(vec![]), 4).await?.is_empty());
        let output = run(
            Arc::new(Count {
                children: vec![source(vec![])],
            }),
            4,
        )
        .await?;
        assert_eq!(values(&output), vec![0]);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn blocked_sources_and_operators_resume() -> Result<()> {
        let root: Arc<dyn PhysicalOperator> = Arc::new(BlockOnce {
            children: vec![Arc::new(
                VecSource::new(batches(0, 50, 10)).with_interrupts(),
            )],
        });
        let output = run(root, 4).await?;
        assert_eq!(sorted_values(&output), (0..500).collect::<Vec<_>>());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn have_more_output_and_selection_vectors() -> Result<()> {
        // Two selective operators compose their selections, and `Repeat`
        // returns `HaveMoreOutput` for every chunk.
        let root: Arc<dyn PhysicalOperator> = Arc::new(Repeat {
            children: vec![Arc::new(KeepMultiples {
                children: vec![Arc::new(KeepMultiples {
                    children: vec![source(batches(0, 10, 100))],
                    divisor: 2,
                })],
                divisor: 3,
            })],
            times: 3,
        });
        let output = run(root, 2).await?;
        let expected = (0..1000)
            .filter(|v| v % 6 == 0)
            .flat_map(|v| [v, v, v])
            .collect::<Vec<_>>();
        assert_eq!(sorted_values(&output), expected);
        // The collector compacts the selection vectors.
        assert!(
            output
                .iter()
                .all(|batch| batch.num_rows() == 17 || batch.num_rows() == 16)
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn final_execute_flushes_buffered_chunks() -> Result<()> {
        // `BufferAll` emits everything from `final_execute`, through
        // `Repeat`, which returns `HaveMoreOutput` while flushing.
        let root: Arc<dyn PhysicalOperator> = Arc::new(Repeat {
            children: vec![Arc::new(BufferAll {
                children: vec![source(batches(0, 30, 7))],
            })],
            times: 2,
        });
        let output = run(root, 3).await?;
        let expected = (0..210).flat_map(|v| [v, v]).collect::<Vec<_>>();
        assert_eq!(sorted_values(&output), expected);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn finished_operator_stops_the_pipeline() -> Result<()> {
        let source = Arc::new(VecSource::new(batches(0, 1000, 10)));
        let root: Arc<dyn PhysicalOperator> = Arc::new(Limit {
            children: vec![Arc::clone(&source) as _],
            limit: 25,
        });
        let output = run(root, 1).await?;
        assert_eq!(values(&output), (0..25).collect::<Vec<_>>());
        // The source was not drained.
        assert!(source.calls.load(Ordering::Relaxed) <= 4);

        // Operators downstream of a finished operator are still flushed,
        // operators upstream of it are not.
        let root: Arc<dyn PhysicalOperator> = Arc::new(BufferAll {
            children: vec![Arc::new(Limit {
                children: vec![Arc::new(BufferAll {
                    children: vec![Arc::new(Limit {
                        children: vec![source_from(batches(0, 10, 10))],
                        limit: 1000,
                    })],
                })],
                limit: 15,
            })],
        });
        let output = run(root, 1).await?;
        assert_eq!(sorted_values(&output).len(), 15);

        // The global operator state is shared by the union pipelines.
        let root: Arc<dyn PhysicalOperator> = Arc::new(Limit {
            children: vec![Arc::new(Union {
                children: vec![
                    source_from(batches(0, 10, 10)),
                    source_from(batches(100, 10, 10)),
                ],
                order_matters: false,
            })],
            limit: 42,
        });
        let output = run(root, 4).await?;
        assert_eq!(values(&output).len(), 42);
        Ok(())
    }

    fn source_from(batches: Vec<RecordBatch>) -> Arc<dyn PhysicalOperator> {
        source(batches)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn finished_sink_stops_the_pipeline() -> Result<()> {
        let source = Arc::new(VecSource::new(batches(0, 1000, 10)));
        let root: Arc<dyn PhysicalOperator> = Arc::new(StopAfter {
            children: vec![Arc::clone(&source) as _],
            limit: 3,
        });
        let output = run(root, 1).await?;
        assert_eq!(values(&output), vec![3]);
        assert!(source.calls.load(Ordering::Relaxed) <= 4);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn pipeline_breakers_finalize_before_their_source_runs() -> Result<()> {
        let materialize = Arc::new(Materialize::new(source(batches(0, 40, 25))));
        let root: Arc<dyn PhysicalOperator> = Arc::new(Count {
            children: vec![Arc::new(Repeat {
                children: vec![Arc::clone(&materialize) as _],
                times: 2,
            })],
        });
        let output = run(root, 4).await?;
        assert_eq!(values(&output), vec![2000]);
        // One combine per thread, one finalize per execution.
        assert_eq!(materialize.combines.load(Ordering::Relaxed), 4);
        assert_eq!(materialize.finalizes.load(Ordering::Relaxed), 1);

        // The number of threads is bounded by `GlobalSourceState::max_threads`.
        let materialize = Arc::new(Materialize::new(Arc::new(
            VecSource::new(batches(0, 40, 25)).with_max_threads(2),
        )));
        let output = run(Arc::clone(&materialize) as _, 8).await?;
        assert_eq!(sorted_values(&output), (0..1000).collect::<Vec<_>>());
        assert_eq!(materialize.combines.load(Ordering::Relaxed), 2);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn no_output_possible_skips_dependent_pipelines() -> Result<()> {
        let materialize = Arc::new(Materialize {
            no_output_if_empty: true,
            ..Materialize::new(source(vec![]))
        });
        let root: Arc<dyn PhysicalOperator> = Arc::new(Count {
            children: vec![Arc::new(Repeat {
                children: vec![Arc::clone(&materialize) as _],
                times: 2,
            })],
        });
        let output = run(root, 4).await?;
        // The pipeline reading from `materialize` is skipped, but `Count`
        // still produces its (empty) result.
        assert_eq!(values(&output), vec![0]);
        assert_eq!(materialize.get_data_calls.load(Ordering::Relaxed), 0);
        assert_eq!(materialize.finalizes.load(Ordering::Relaxed), 1);

        // With input, the pipeline runs.
        let materialize = Arc::new(Materialize {
            no_output_if_empty: true,
            ..Materialize::new(source(batches(0, 2, 3)))
        });
        let output = run(Arc::clone(&materialize) as _, 4).await?;
        assert_eq!(sorted_values(&output), (0..6).collect::<Vec<_>>());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn union_and_child_pipelines_execute_in_dependency_order() -> Result<()> {
        let union: Arc<dyn PhysicalOperator> = Arc::new(Union {
            children: vec![source(batches(0, 5, 10)), source(batches(50, 5, 10))],
            order_matters: false,
        });
        let root: Arc<dyn PhysicalOperator> = Arc::new(ProbeThenScan {
            children: vec![union, source(batches(1000, 2, 3))],
        });
        let output = run(root, 4).await?;
        let mut expected = (0..100).chain(1000..1006).collect::<Vec<_>>();
        // The child pipeline runs after both probe pipelines completed.
        expected.push(100);
        expected.sort_unstable();
        assert_eq!(sorted_values(&output), expected);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn errors_are_propagated() {
        let root: Arc<dyn PhysicalOperator> = Arc::new(FailAt {
            children: vec![source(batches(0, 100, 10))],
            fail_at: 7,
            seen: AtomicUsize::new(0),
        });
        let err = run(root, 4).await.unwrap_err();
        assert!(
            err.to_string().contains("FailAt failed as requested"),
            "{err}"
        );

        // Errors of pipelines below a pipeline breaker are propagated too.
        let root: Arc<dyn PhysicalOperator> =
            Arc::new(Materialize::new(Arc::new(FailAt {
                children: vec![source(batches(0, 100, 10))],
                fail_at: 1,
                seen: AtomicUsize::new(0),
            })));
        let err = run(root, 4).await.unwrap_err();
        assert!(
            err.to_string().contains("FailAt failed as requested"),
            "{err}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn dropping_the_stream_cancels_the_execution() -> Result<()> {
        let source = Arc::new(VecSource::new(batches(0, 100_000, 10)));
        let mut stream =
            Executor::execute_stream(Arc::clone(&source) as _, task_ctx(), 2)?;
        let first = stream.next().await.unwrap()?;
        assert_eq!(first.num_rows(), 10);
        drop(stream);
        let calls = source.calls.load(Ordering::Relaxed);
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        let later = source.calls.load(Ordering::Relaxed);
        assert!(later < 100_000, "the source was drained ({later} calls)");
        // No more work is done once the stream is gone.
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
        assert_eq!(source.calls.load(Ordering::Relaxed), later, "{calls}");
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn metrics_are_recorded() -> Result<()> {
        let source = Arc::new(VecSource::new(batches(0, 10, 10)));
        let filter: Arc<dyn PhysicalOperator> = Arc::new(KeepMultiples {
            children: vec![Arc::clone(&source) as _],
            divisor: 2,
        });
        let stream = Executor::execute_stream(Arc::clone(&filter), task_ctx(), 2)?;
        let metrics = Arc::clone(stream.metrics());
        let output = stream.try_collect::<Vec<_>>().await?;
        assert_eq!(values(&output).len(), 50);

        let source_metrics = metrics.get(source.as_ref()).unwrap();
        assert_eq!(source_metrics.output_rows(), 100);
        assert_eq!(source_metrics.output_chunks(), 10);
        let filter_metrics = metrics.get(filter.as_ref()).unwrap();
        assert_eq!(filter_metrics.output_rows(), 50);
        let (collector, collector_metrics) = metrics.iter().last().unwrap();
        assert_eq!(collector.name(), "CollectorSink");
        assert_eq!(collector_metrics.sink_rows(), 50);
        assert_eq!(collector_metrics.sink_chunks(), 10);
        assert_eq!(metrics.iter().count(), 3);
        Ok(())
    }

    #[test]
    fn batch_queue_applies_backpressure() {
        let queue = BatchQueue::new(1);
        let batch = batches(0, 1, 1).remove(0);
        let waker = Waker::noop();
        assert!(matches!(
            queue.push(batch.clone(), waker),
            PushResult::Pushed
        ));
        assert!(matches!(
            queue.push(batch.clone(), waker),
            PushResult::Full(_)
        ));
        assert_eq!(queue.inner.lock().producers.len(), 1);
        assert!(queue.pop(waker).is_some());
        assert!(queue.inner.lock().producers.is_empty());
        assert!(queue.pop(waker).is_none());
        assert!(queue.inner.lock().consumer.is_some());
        queue.close();
        assert!(matches!(queue.push(batch, waker), PushResult::Closed));
    }
}
