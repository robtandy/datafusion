// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Execution of a [`PipelineGraph`].
//!
//! * [`PipelineExecutor`] runs one thread's share of one [`Pipeline`]: a
//!   tight push loop that pulls chunks from the source and pushes them
//!   through the operators into the sink (DuckDB's `PipelineExecutor`).
//! * [`Executor`] schedules the pipelines of a query: every pipeline runs on
//!   up to `num_threads` threads once all of its dependencies are complete,
//!   and the sink of a [`MetaPipeline`] is finalized once all of its
//!   pipelines completed. The execution runs on the task that polls its
//!   [`PipelineStream`]; a pipeline that runs alone runs its first thread on
//!   that task too, and only its other threads on worker tasks.
//! * [`CollectorSink`] is the sink of the root pipeline: it hands the result
//!   chunks, as [`RecordBatch`]es, to a [`PipelineStream`].

use std::collections::{HashMap, HashSet, VecDeque};
use std::fmt;
use std::ops::Range;
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll, Waker};

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use datafusion_common::instant::Instant;
use datafusion_common::{DataFusionError, Result, internal_err};
use datafusion_common_runtime::JoinSet;
use datafusion_execution::{RecordBatchStream, TaskContext};
use futures::future::BoxFuture;
use futures::stream::FuturesUnordered;
use futures::{Stream, StreamExt, TryStreamExt};
use parking_lot::Mutex;

use crate::chunk::DataChunk;
use crate::operator::{
    EmptyState, ExecutionContext, ExecutionRegistry, GlobalOperatorState,
    GlobalSinkState, GlobalSourceState, InterruptKind, LocalSinkState, LocalSourceState,
    MetaPipelineKeys, OperatorFinalizeResult, OperatorMetrics, OperatorResult,
    OperatorState, PhysicalOperator, PipelineDependencies, PipelineKeys, SinkResult,
    SourceResult, TaskInterrupt, downcast_state_mut, operator_key,
};
use crate::pipeline::{MetaPipeline, Pipeline, PipelineGraph};

/// Maximum number of operator calls (`get_data`, `execute`, `final_execute`
/// and `sink`) a [`PipelineExecutor`] makes per poll before it yields back to
/// the async runtime.
const CALLS_PER_POLL: usize = 128;

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
    /// The pipeline skips its work (see
    /// [`crate::operator::SinkFinalize::NoOutputPossible`] and
    /// [`crate::operator::SinkResult::Finished`]): no push loop at all.
    skip: bool,
}

/// What [`PipelineExecutor::poll_run`] achieved.
enum Progress {
    /// The pipeline is complete for this thread.
    Done,
    /// The task should yield to the scheduler, then continue.
    Yield,
}

/// One step of the push loop.
enum Step {
    /// Progress was made, continue.
    Continue,
    /// The call at a stage is blocked (`0`: the source, `1..=n`: an
    /// operator, `n + 1`: the sink).
    Blocked(usize, TaskInterrupt),
    /// The pipeline is complete for this thread.
    Done,
}

/// A future to wait for before retrying the blocked call at `stage` (see
/// [`Step::Blocked`]).
struct Wait {
    stage: usize,
    future: BoxFuture<'static, ()>,
}

/// Outcome of pushing a chunk through the operators of a pipeline.
enum PushOutcome {
    /// The last chunk (`chunks[n]`) holds output for the sink.
    Output,
    /// No output for the sink; the operators need more input.
    NeedMoreInput,
    /// The operator at a stage is blocked.
    Blocked(usize, TaskInterrupt),
    /// The call budget is exhausted; the push resumes from the
    /// `in_process_operators` stack.
    Yield,
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
/// pushed onto the `in_process_operators` stack; before pulling new data from the
/// source, the executor calls the in-process operators again with the same
/// input, from the most downstream one. The stack also records where to
/// resume when an operator is blocked or when the executor yields because
/// its call budget is exhausted.
///
/// At most `CALLS_PER_POLL` operator calls are made per poll, so a pipeline
/// never monopolizes a thread of the async runtime, even if its operators
/// produce output that is discarded downstream.
pub struct PipelineExecutor {
    shared: Arc<PipelineShared>,
    ctx: ExecutionContext,
    source_state: Box<dyn LocalSourceState>,
    operator_states: Vec<Box<dyn OperatorState>>,
    sink_state: Box<dyn LocalSinkState>,
    chunks: Vec<DataChunk>,
    in_process_operators: Vec<usize>,
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
    /// Future the task waits for before retrying a blocked call.
    wait: Option<Wait>,
    /// Operator calls left in the current poll.
    budget: usize,
    /// Id of this thread in the execution.
    thread: usize,
    /// The `finish_epoch` of the registry when this thread last checked
    /// whether an operator or the sink of the pipeline finished.
    finish_epoch: usize,
    /// Yield as soon as the source produced a second chunk (see
    /// [`Self::yield_on_more_input`]); cleared once it did.
    yield_on_more_input: bool,
    /// The number of non-empty chunks the source produced on this thread.
    source_chunks: usize,
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
                in_process_operators: vec![],
                push_stage: 0,
                pending_sink: false,
                source_exhausted: true,
                flush_index: num_operators,
                flush_current_finished: false,
                wait: None,
                budget: 0,
                thread: ctx.registry().next_thread_id(),
                finish_epoch: 0,
                yield_on_more_input: false,
                source_chunks: 0,
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
            in_process_operators: vec![],
            push_stage: 0,
            pending_sink: false,
            source_exhausted: false,
            flush_index: 0,
            flush_current_finished: false,
            wait: None,
            budget: 0,
            thread: ctx.registry().next_thread_id(),
            finish_epoch: 0,
            yield_on_more_input: false,
            source_chunks: 0,
            done: false,
            shared,
            ctx,
        })
    }

    /// Yield (end the current poll with [`Progress::Yield`]) as soon as the
    /// source produces a second chunk, before pushing it: the source has more
    /// work than this thread did so far, so that the executor can start the
    /// other threads of the pipeline. Once.
    fn yield_on_more_input(&mut self) {
        self.yield_on_more_input = true;
    }

    /// Run the pipeline until it is complete or blocked.
    ///
    /// Returns `Poll::Ready` once the source is exhausted, all operators are
    /// flushed and the local sink state was combined into the global one.
    /// When the call budget of a poll is exhausted, the task is woken and
    /// `Poll::Pending` is returned. [`Self::execute`] yields more politely,
    /// like tokio's cooperative scheduling: prefer it.
    pub fn poll_execute(&mut self, cx: &mut Context<'_>) -> Poll<Result<()>> {
        match self.poll_run(cx) {
            Poll::Ready(Ok(Progress::Done)) => Poll::Ready(Ok(())),
            Poll::Ready(Ok(Progress::Yield)) => {
                cx.waker().wake_by_ref();
                Poll::Pending
            }
            Poll::Ready(Err(e)) => Poll::Ready(Err(e)),
            Poll::Pending => Poll::Pending,
        }
    }

    /// Run the pipeline to completion.
    pub async fn execute(mut self) -> Result<()> {
        loop {
            match futures::future::poll_fn(|cx| self.poll_run(cx)).await? {
                Progress::Done => return Ok(()),
                // Like tokio's cooperative scheduling: run again only after
                // the other ready tasks (and the I/O and timer drivers) had a
                // chance to run. All state needed to resume is kept in `self`.
                Progress::Yield => tokio::task::yield_now().await,
            }
        }
    }

    /// Make progress until the pipeline is complete, blocked, or the call
    /// budget of this poll is exhausted.
    fn poll_run(&mut self, cx: &mut Context<'_>) -> Poll<Result<Progress>> {
        if self.done {
            return Poll::Ready(Ok(Progress::Done));
        }
        self.ctx.set_waker(cx.waker());
        self.apply_finished();
        if let Some(wait) = self.wait.as_mut() {
            if wait.future.as_mut().poll(cx).is_pending() {
                return self.suspend(cx);
            }
            self.wait = None;
        }
        self.budget = CALLS_PER_POLL;
        loop {
            if self.budget == 0 {
                return Poll::Ready(Ok(Progress::Yield));
            }
            match self.step() {
                Ok(Step::Continue) => {}
                Ok(Step::Done) => {
                    self.done = true;
                    self.ctx.registry().unregister(self.thread);
                    return Poll::Ready(self.combine().map(|()| Progress::Done));
                }
                Ok(Step::Blocked(stage, interrupt)) => match interrupt.into_kind() {
                    InterruptKind::WakerRegistered => return self.suspend(cx),
                    InterruptKind::Yield => return Poll::Ready(Ok(Progress::Yield)),
                    InterruptKind::Future(mut future) => {
                        if future.as_mut().poll(cx).is_pending() {
                            self.wait = Some(Wait { stage, future });
                            return self.suspend(cx);
                        }
                    }
                },
                Err(e) => {
                    self.done = true;
                    self.ctx.registry().unregister(self.thread);
                    return Poll::Ready(Err(e));
                }
            }
        }
    }

    /// Suspend the blocked thread. Besides by what it is blocked on, it is
    /// woken when an operator or a sink finishes, which may end its work.
    fn suspend(&mut self, cx: &Context<'_>) -> Poll<Result<Progress>> {
        let registry = self.ctx.registry();
        registry.register_blocked(self.thread, cx.waker());
        if registry.finish_epoch() != self.finish_epoch {
            // An operator or a sink finished in the meantime: check it first.
            return Poll::Ready(Ok(Progress::Yield));
        }
        Poll::Pending
    }

    /// Stop the work that is no longer needed because an operator or the
    /// sink of the pipeline returned `Finished`, on any thread.
    fn apply_finished(&mut self) {
        let epoch = self.ctx.registry().finish_epoch();
        if epoch == self.finish_epoch {
            return;
        }
        self.finish_epoch = epoch;
        let registry = Arc::clone(self.ctx.registry());
        let shared = Arc::clone(&self.shared);
        if registry.is_sink_finished(shared.sink.as_ref()) {
            // The sink does not need any more input, not even the chunk it
            // did not accept yet: all remaining work, including the blocked
            // call, is skipped.
            self.pending_sink = false;
            self.wait = None;
            self.finish_at(shared.operators.len());
            return;
        }
        if let Some(index) = shared
            .operators
            .iter()
            .rposition(|operator| registry.is_operator_finished(operator.as_ref()))
        {
            let stage = index + 1;
            // The source and the operators up to the finished one are not
            // called again: stop waiting for a blocked call among them. A
            // blocked call downstream is still needed: keep waiting for it.
            if self.wait.as_ref().is_some_and(|wait| wait.stage <= stage) {
                self.wait = None;
            }
            self.finish_at(stage);
        }
    }

    /// Consume one call from the budget of the current poll. Returns `false`
    /// if the budget is exhausted.
    fn take_budget(&mut self) -> bool {
        if self.budget == 0 {
            return false;
        }
        self.budget -= 1;
        true
    }

    /// Make progress. Only called with a non-empty budget, so that every
    /// step makes at least one operator call (or completes the pipeline).
    fn step(&mut self) -> Result<Step> {
        self.apply_finished();
        // A chunk the sink did not accept yet.
        if self.pending_sink {
            return self.sink_last_chunk();
        }
        // Operators that have more output for their current input.
        if !self.in_process_operators.is_empty() {
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
        if !self.take_budget() {
            return Ok(Step::Continue);
        }
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
                self.source_chunks += 1;
                if self.yield_on_more_input && self.source_chunks > 1 {
                    // End the poll before the first operator call: the push
                    // resumes from the `in_process_operators` stack (or the
                    // pending sink chunk).
                    self.yield_on_more_input = false;
                    self.budget = 0;
                }
                let outcome = self.push(0)?;
                self.handle_push_outcome(outcome)
            }
            SourceResult::Finished => {
                self.source_exhausted = true;
                Ok(Step::Continue)
            }
            SourceResult::Blocked(interrupt) => Ok(Step::Blocked(0, interrupt)),
        }
    }

    fn handle_push_outcome(&mut self, outcome: PushOutcome) -> Result<Step> {
        match outcome {
            PushOutcome::Output => {
                self.pending_sink = true;
                self.sink_last_chunk()
            }
            PushOutcome::NeedMoreInput | PushOutcome::Yield => Ok(Step::Continue),
            PushOutcome::Blocked(stage, interrupt) => Ok(Step::Blocked(stage, interrupt)),
        }
    }

    /// Push the output of stage `initial` through the operators after it, or
    /// resume the most downstream in-process operator.
    fn push(&mut self, initial: usize) -> Result<PushOutcome> {
        let shared = Arc::clone(&self.shared);
        let n = shared.operators.len();
        let mut stage = match self.in_process_operators.pop() {
            Some(stage) => stage,
            None => initial + 1,
        };
        loop {
            if stage > n {
                return Ok(PushOutcome::Output);
            }
            if !self.take_budget() {
                // Resume with this operator: all stages on the stack are
                // upstream of it, so the stack stays ordered.
                self.in_process_operators.push(stage);
                return Ok(PushOutcome::Yield);
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
                OperatorResult::HaveMoreOutput => self.in_process_operators.push(stage),
                OperatorResult::Finished => {
                    self.finish_at(stage);
                    // Stop the other threads, of all pipelines with this
                    // operator, too.
                    self.ctx
                        .registry()
                        .set_finished(shared.operators[index].as_ref(), false);
                }
                OperatorResult::Blocked(interrupt) => {
                    output.reset();
                    self.in_process_operators.push(stage);
                    return Ok(PushOutcome::Blocked(stage, interrupt));
                }
            }

            let rows = self.chunks[stage].len();
            if rows == 0 {
                // No output: resume the most downstream in-process operator,
                // or ask for more input.
                match self.in_process_operators.pop() {
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
        if !self.take_budget() {
            // `pending_sink` is set: the chunk is sunk in the next poll.
            return Ok(Step::Continue);
        }
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
                // including flushing, on all threads of all pipelines that
                // end in this sink.
                self.finish_at(self.shared.operators.len());
                self.ctx
                    .registry()
                    .set_finished(self.shared.sink.as_ref(), true);
                Ok(Step::Continue)
            }
            SinkResult::Blocked(interrupt) => {
                Ok(Step::Blocked(self.shared.operators.len() + 1, interrupt))
            }
        }
    }

    /// The operator at `stage` (or the sink, for `stage == n`) finished:
    /// stop pulling from the source and never call the operators upstream of
    /// it again.
    fn finish_at(&mut self, stage: usize) {
        self.source_exhausted = true;
        // Stages on the stack are all upstream of `stage`.
        self.in_process_operators.retain(|&s| s > stage);
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
            if !self.take_budget() {
                return Ok(Some(Step::Continue));
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
                continue;
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
            .field("in_process_operators", &self.in_process_operators)
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
/// runnable pipeline is executed by up to `num_threads` [`PipelineExecutor`]s
/// (its threads), each on a worker task; independent pipelines run
/// concurrently, except pipelines that share a sink, operator or source that
/// does not support parallel execution, which run one after the other. Once
/// all pipelines of a meta pipeline completed (and every thread combined its
/// local sink state), the sink of the meta pipeline is finalized.
///
/// The tasks are sized to the work: a pipeline that runs while no other
/// pipeline does runs its first thread on the task of the execution (the task
/// that polls the [`PipelineStream`]), so that a pipeline on a single thread
/// spawns no task, and a source that does not bound its threads (e.g. the
/// result of an aggregation, which may be a single row) only gets more
/// threads once the first one found more than a chunk of work, see
/// [`Executor::execute`].
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
    ///
    /// A pipeline that runs while no other pipeline does runs its first
    /// thread on the task that awaits this future, and its other threads on
    /// worker tasks: right away if its source bounds its threads
    /// ([`GlobalSourceState::max_threads`]), otherwise once the first thread
    /// got a second chunk from the source, used up the call budget of a poll
    /// or is blocked. Other pipelines run all of their threads on worker
    /// tasks, concurrently.
    pub async fn execute(&self) -> Result<()> {
        let registry = Arc::new(ExecutionRegistry::new(
            self.metrics.by_key(),
            pipeline_dependencies(&self.graph),
        ));
        let graph = self.graph.as_ref();
        log::debug!(
            "Executing pipelines with up to {} threads:\n{graph}",
            self.num_threads
        );
        let Events {
            mut dependencies,
            dependents,
            ..
        } = Events::try_new(graph)?;
        let num_events = dependencies.len();
        let num_pipelines = graph.pipelines().len();
        let is_pipeline = |event: usize| event < num_pipelines;

        let mut running = FuturesUnordered::new();
        // The pipelines among the events in `running`.
        let mut running_pipelines = 0;
        let mut ready = (0..num_events)
            .filter(|&event| dependencies[event] == 0)
            .collect::<Vec<_>>();
        let mut completed = 0;
        while completed < num_events {
            // A pipeline that runs while no other pipeline does (the events
            // that become ready later wait for it) runs its first thread on
            // this task, see `run_pipeline`.
            let ready_pipelines = ready.iter().filter(|&&event| is_pipeline(event));
            let alone = running_pipelines == 0 && ready_pipelines.count() == 1;
            for event in ready.drain(..) {
                running_pipelines += usize::from(is_pipeline(event));
                running.push(self.run_event(event, &registry, alone));
            }
            let Some(result) = running.next().await else {
                return internal_err!(
                    "Pipeline dependency cycle: {completed} of {num_events} events completed"
                );
            };
            let event = result?;
            completed += 1;
            running_pipelines -= usize::from(is_pipeline(event));
            for &dependent in &dependents[event] {
                dependencies[dependent] -= 1;
                if dependencies[dependent] == 0 {
                    ready.push(dependent);
                }
            }
        }
        Ok(())
    }

    /// Run event `event`: pipeline `event` for `event < #pipelines`,
    /// otherwise the finalization of meta pipeline `event - #pipelines`.
    /// `alone`: no other pipeline runs while pipeline `event` does.
    fn run_event<'a>(
        &'a self,
        event: usize,
        registry: &'a Arc<ExecutionRegistry>,
        alone: bool,
    ) -> BoxFuture<'a, Result<usize>> {
        Box::pin(async move {
            let num_pipelines = self.graph.pipelines().len();
            if event < num_pipelines {
                self.run_pipeline(self.graph.pipeline(event), registry, alone)
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

    /// Run `pipeline` on its threads (see [`Self::pipeline_threads`]).
    ///
    /// If no other pipeline runs at the same time (`alone`), the first thread
    /// runs on the task of the execution itself, instead of a worker task of
    /// its own, which spares a pipeline on a single thread (e.g. the pipeline
    /// that reads a sort, or a small table) all task overhead. The other
    /// threads run on worker tasks. If the source does not bound its threads
    /// (`GlobalSourceState::max_threads` is `usize::MAX`, e.g. the result of an
    /// aggregation, which may be a single row), they only start once the
    /// first thread found more work than one chunk (or is blocked), so that a
    /// source with little output does not spawn tasks that immediately exit.
    async fn run_pipeline(
        &self,
        pipeline: &Pipeline,
        registry: &Arc<ExecutionRegistry>,
        alone: bool,
    ) -> Result<()> {
        let ctx = self.context(0, self.num_threads, registry);
        // The pipeline is a no-op (its sink still combines an empty state) if
        // an operator already knows it cannot produce output, if its sink
        // does not need more input, or if one of its operators finished.
        let skip = registry.skips_pipeline(pipeline.id());
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
        log::trace!(
            "Starting pipeline {} ({pipeline}) on {num_threads} threads{}",
            pipeline.id(),
            if skip { ", skipped" } else { "" }
        );

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
        let spawn = |workers: &mut JoinSet<Result<()>>, threads: Range<usize>| {
            for thread_id in threads {
                let shared = Arc::clone(&shared);
                let ctx = self.context(thread_id, num_threads, registry);
                workers.spawn(async move {
                    PipelineExecutor::try_new(shared, ctx)?.execute().await
                });
            }
        };
        if !alone {
            spawn(&mut workers, 0..num_threads);
            return join_workers(workers).await;
        }

        let mut first = PipelineExecutor::try_new(
            Arc::clone(&shared),
            self.context(0, num_threads, registry),
        )?;
        if num_threads > 1 && shared.source_state.max_threads() == usize::MAX {
            // Run the first thread until the source produced a second chunk,
            // the thread is blocked, or it used up the budget of a poll.
            first.yield_on_more_input();
            match futures::future::poll_fn(|cx| Poll::Ready(first.poll_run(cx))).await {
                Poll::Ready(Ok(Progress::Done)) => return Ok(()),
                Poll::Ready(Err(e)) => return Err(e),
                Poll::Ready(Ok(Progress::Yield)) | Poll::Pending => {}
            }
        }
        spawn(&mut workers, 1..num_threads);
        // Dropping `workers` (on error, or when the execution is dropped)
        // aborts the other threads.
        futures::future::try_join(first.execute(), join_workers(workers)).await?;
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

/// Wait for the threads `workers` of a pipeline: the first error of any of
/// them, or the panic of any of them, which is resumed. Dropping the future
/// aborts the threads.
async fn join_workers(mut workers: JoinSet<Result<()>>) -> Result<()> {
    while let Some(result) = workers.join_next().await {
        match result {
            Ok(result) => result?,
            Err(e) if e.is_panic() => std::panic::resume_unwind(e.into_panic()),
            Err(e) => return Err(DataFusionError::ExecutionJoin(Box::new(e))),
        }
    }
    Ok(())
}

/// The pipelines of `graph` in terms of operator keys.
fn pipeline_dependencies(graph: &PipelineGraph) -> PipelineDependencies {
    let key = |operator: &Arc<dyn PhysicalOperator>| operator_key(operator.as_ref());
    let pipelines = graph
        .pipelines()
        .iter()
        .map(|pipeline| PipelineKeys {
            source: key(pipeline.source()),
            operators: pipeline.operators().iter().map(key).collect(),
            sink: key(pipeline.sink()),
        })
        .collect();
    // The consumers of a meta pipeline are the pipelines that read the
    // result of its sink: those in which the sink is the source (e.g. an
    // aggregate, or the scan of the unmatched rows of an outer join) or an
    // operator (e.g. the probe of a hash join).
    let meta_pipelines = graph
        .meta_pipelines()
        .iter()
        .map(|meta| {
            let sink = key(meta.sink());
            let consumers = graph
                .pipelines()
                .iter()
                .filter(|pipeline| {
                    key(pipeline.source()) == sink
                        || pipeline.operators().iter().any(|op| key(op) == sink)
                })
                .map(Pipeline::id)
                .collect();
            MetaPipelineKeys { sink, consumers }
        })
        .collect();
    PipelineDependencies {
        pipelines,
        meta_pipelines,
    }
}

/// The events of an execution: event `p` (for `p < #pipelines`) runs pipeline
/// `p`, event `#pipelines + m` finalizes meta pipeline `m`.
struct Events {
    /// For every event, the number of events it waits for.
    dependencies: Vec<usize>,
    /// For every event, the events that wait for it.
    dependents: Vec<Vec<usize>>,
    /// The `(prerequisite, dependent)` pairs, to avoid duplicates.
    edges: HashSet<(usize, usize)>,
}

impl Events {
    fn try_new(graph: &PipelineGraph) -> Result<Self> {
        let num_pipelines = graph.pipelines().len();
        let num_events = num_pipelines + graph.meta_pipelines().len();
        let mut events = Self {
            dependencies: vec![0; num_events],
            dependents: vec![vec![]; num_events],
            edges: HashSet::new(),
        };
        for pipeline in graph.pipelines() {
            for &dependency in pipeline.dependencies() {
                events.add(dependency, pipeline.id());
            }
            for &meta in pipeline.meta_dependencies() {
                events.add(num_pipelines + meta, pipeline.id());
            }
        }
        for meta in graph.meta_pipelines() {
            for &pipeline in meta.pipelines() {
                events.add(pipeline, num_pipelines + meta.id());
            }
        }

        // A sink, operator or source that does not support parallel
        // execution may be part of several pipelines (e.g. the pipelines of
        // the inputs of a union share the sink and the operators above the
        // union). Run these pipelines one after the other, in an order that
        // is compatible with all dependencies, so that such an operator is
        // never called concurrently.
        let order = events.topological_order()?;
        let mut position = vec![0; num_events];
        for (index, &event) in order.iter().enumerate() {
            position[event] = index;
        }
        let mut exclusive: HashMap<usize, Vec<usize>> = HashMap::new();
        for pipeline in graph.pipelines() {
            let non_parallel = std::iter::once(pipeline.source())
                .filter(|source| !source.parallel_source())
                .chain(
                    pipeline
                        .operators()
                        .iter()
                        .filter(|operator| !operator.parallel_operator()),
                )
                .chain(
                    std::iter::once(pipeline.sink()).filter(|sink| !sink.parallel_sink()),
                );
            for operator in non_parallel {
                exclusive
                    .entry(operator_key(operator.as_ref()))
                    .or_default()
                    .push(pipeline.id());
            }
        }
        for mut pipelines in exclusive.into_values() {
            pipelines.sort_by_key(|&pipeline| position[pipeline]);
            pipelines.dedup();
            for pair in pipelines.windows(2) {
                events.add(pair[0], pair[1]);
            }
        }
        Ok(events)
    }

    /// Make `dependent` wait for `prerequisite`.
    fn add(&mut self, prerequisite: usize, dependent: usize) {
        if prerequisite != dependent && self.edges.insert((prerequisite, dependent)) {
            self.dependencies[dependent] += 1;
            self.dependents[prerequisite].push(dependent);
        }
    }

    /// All events, in an order in which every event comes after the events
    /// it waits for.
    fn topological_order(&self) -> Result<Vec<usize>> {
        let mut remaining = self.dependencies.clone();
        let mut ready = (0..remaining.len())
            .filter(|&event| remaining[event] == 0)
            .collect::<VecDeque<_>>();
        let mut order = Vec::with_capacity(remaining.len());
        while let Some(event) = ready.pop_front() {
            order.push(event);
            for &dependent in &self.dependents[event] {
                remaining[dependent] -= 1;
                if remaining[dependent] == 0 {
                    ready.push_back(dependent);
                }
            }
        }
        if order.len() != remaining.len() {
            return internal_err!(
                "Pipeline dependency cycle: only {} of {} events can run",
                order.len(),
                remaining.len()
            );
        }
        Ok(order)
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
    /// The execution, which the stream drives: it runs on the task that polls
    /// the stream, and the threads of the pipelines on worker tasks (see
    /// [`Executor::execute`]).
    Running(BoxFuture<'static, Result<()>>),
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
                    // The execution runs on the task that polls the stream, as
                    // DataFusion's streams do: a query whose pipelines run on a
                    // single thread spawns no task at all.
                    this.state = StreamState::Running(Box::pin(async move {
                        executor.execute().await
                    }));
                }
                StreamState::Running(mut execution) => {
                    if let Some(batch) = this.queue.pop(cx.waker()) {
                        this.state = StreamState::Running(execution);
                        return Poll::Ready(Some(Ok(batch)));
                    }
                    match execution.as_mut().poll(cx) {
                        Poll::Pending => {
                            this.state = StreamState::Running(execution);
                            // The execution may have produced batches before
                            // it had to wait (e.g. for room in the queue).
                            return match this.queue.pop(cx.waker()) {
                                Some(batch) => Poll::Ready(Some(Ok(batch))),
                                None => Poll::Pending,
                            };
                        }
                        // All producers are done: drain the remaining batches.
                        Poll::Ready(Ok(())) => {}
                        Poll::Ready(Err(e)) => {
                            this.queue.close();
                            return Poll::Ready(Some(Err(e)));
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
        // Producers that are still running (dropping the execution aborts
        // its worker tasks) must not block on a queue nobody reads.
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
    use datafusion_common_runtime::SpawnedTask;

    use crate::chunk::SelectionVector;
    use crate::operator::{GlobalSinkState, SinkFinalize, downcast_state};
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
    /// itself with all kinds of [`TaskInterrupt`]s. It records its calls, its
    /// threads (local states) and the tasks that call it.
    #[derive(Debug)]
    struct VecSource {
        batches: Vec<RecordBatch>,
        interrupts: bool,
        max_threads: usize,
        calls: AtomicUsize,
        local_states: AtomicUsize,
        /// The tasks that called `get_data` (`None` outside of a task).
        tasks: Mutex<HashSet<Option<tokio::task::Id>>>,
    }

    impl VecSource {
        fn new(batches: Vec<RecordBatch>) -> Self {
            Self {
                batches,
                interrupts: false,
                max_threads: usize::MAX,
                calls: AtomicUsize::new(0),
                local_states: AtomicUsize::new(0),
                tasks: Mutex::default(),
            }
        }

        /// The number of tasks that called `get_data`, and whether one of
        /// them is the current task.
        fn tasks(&self) -> (usize, bool) {
            let tasks = self.tasks.lock();
            (tasks.len(), tasks.contains(&tokio::task::try_id()))
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
            self.local_states.fetch_add(1, Ordering::Relaxed);
            Ok(Box::new(VecSourceLocal::default()))
        }

        fn get_data(
            &self,
            ctx: &ExecutionContext,
            gstate: &dyn GlobalSourceState,
            lstate: &mut dyn LocalSourceState,
        ) -> Result<SourceResult> {
            self.calls.fetch_add(1, Ordering::Relaxed);
            self.tasks.lock().insert(tokio::task::try_id());
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
        local_states: AtomicUsize,
        combines: AtomicUsize,
        finalizes: AtomicUsize,
        get_data_calls: AtomicUsize,
    }

    impl Materialize {
        fn new(child: Arc<dyn PhysicalOperator>) -> Self {
            Self {
                children: vec![child],
                no_output_if_empty: false,
                local_states: AtomicUsize::new(0),
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
            self.local_states.fetch_add(1, Ordering::Relaxed);
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
        /// Return `SinkFinalize::NoOutputPossible` for an empty build side.
        no_output_if_empty: bool,
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

        fn finalize(
            &self,
            _ctx: &ExecutionContext,
            gstate: &dyn GlobalSinkState,
        ) -> Result<SinkFinalize> {
            let gstate = gstate.downcast_ref::<ProbeThenScanGlobal>().unwrap();
            Ok(
                if self.no_output_if_empty && gstate.build.lock().is_empty() {
                    SinkFinalize::NoOutputPossible
                } else {
                    SinkFinalize::Ready
                },
            )
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
                no_output_if_empty: false,
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
            no_output_if_empty: false,
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
    async fn pipelines_get_the_threads_their_work_needs() -> Result<()> {
        // A source that does not bound its threads, with a single chunk, on 4
        // threads: the first thread runs on the task that polls the stream
        // (here the test, which is not a task) and finds no more work, so no
        // other thread starts, and no task is spawned.
        let source = Arc::new(VecSource::new(batches(0, 1, 10)));
        let output = run(Arc::clone(&source) as _, 4).await?;
        assert_eq!(values(&output), (0..10).collect::<Vec<_>>());
        assert_eq!(source.local_states.load(Ordering::Relaxed), 1);
        // The chunk, then the end of the source.
        assert_eq!(source.calls.load(Ordering::Relaxed), 2);
        assert_eq!(source.tasks(), (1, true));
        // Also when a task polls the stream, and for every pipeline that runs
        // alone: here the pipelines into and out of a pipeline breaker.
        let source = Arc::new(VecSource::new(batches(0, 1, 10)));
        let materialize = Arc::new(Materialize::new(Arc::clone(&source) as _));
        let task = {
            let (source, materialize) = (Arc::clone(&source), Arc::clone(&materialize));
            SpawnedTask::spawn(async move {
                let output = run(materialize, 4).await?;
                Ok::<_, DataFusionError>((output, source.tasks()))
            })
        };
        let (output, tasks) = task.join().await.unwrap()?;
        assert_eq!(values(&output), (0..10).collect::<Vec<_>>());
        assert_eq!(tasks, (1, true));
        assert_eq!(materialize.combines.load(Ordering::Relaxed), 1);
        assert_eq!(materialize.get_data_calls.load(Ordering::Relaxed), 2);

        // With more chunks, the other threads start (on worker tasks) once the
        // first thread got a second chunk.
        let source = Arc::new(VecSource::new(batches(0, 40, 10)));
        let output = run(Arc::clone(&source) as _, 4).await?;
        assert_eq!(sorted_values(&output), (0..400).collect::<Vec<_>>());
        assert_eq!(source.local_states.load(Ordering::Relaxed), 4);
        assert_eq!(source.tasks(), (4, true));
        // A source that bounds its threads gets them right away.
        let source = Arc::new(VecSource::new(batches(0, 1, 10)).with_max_threads(2));
        let output = run(Arc::clone(&source) as _, 4).await?;
        assert_eq!(values(&output), (0..10).collect::<Vec<_>>());
        assert_eq!(source.local_states.load(Ordering::Relaxed), 2);
        assert_eq!(source.tasks(), (2, true));

        // Pipelines that run at the same time (the inputs of a union) run all
        // their threads on worker tasks; one after the other (an ordered
        // union), each of them runs alone.
        for order_matters in [false, true] {
            let inputs =
                [0, 100].map(|start| Arc::new(VecSource::new(batches(start, 1, 10))));
            let root: Arc<dyn PhysicalOperator> = Arc::new(Union {
                children: inputs.iter().map(|input| Arc::clone(input) as _).collect(),
                order_matters,
            });
            let output = run(root, 4).await?;
            assert_eq!(values(&output).len(), 20);
            for input in &inputs {
                let (threads, tasks) = match order_matters {
                    false => (4, (4, false)),
                    true => (1, (1, true)),
                };
                assert_eq!(input.local_states.load(Ordering::Relaxed), threads);
                assert_eq!(input.tasks(), tasks, "order matters: {order_matters}");
            }
        }
        Ok(())
    }

    /// Panics when the threads `threads` create their operator state.
    #[derive(Debug)]
    struct PanicOnThreads {
        children: Vec<Arc<dyn PhysicalOperator>>,
        threads: Range<usize>,
    }

    impl PhysicalOperator for PanicOnThreads {
        fn name(&self) -> &str {
            "PanicOnThreads"
        }

        fn schema(&self) -> SchemaRef {
            self.children[0].schema()
        }

        fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
            &self.children
        }

        fn get_operator_state(
            &self,
            ctx: &ExecutionContext,
            _gstate: &dyn GlobalOperatorState,
        ) -> Result<Box<dyn OperatorState>> {
            assert!(
                !self.threads.contains(&ctx.thread_id),
                "PanicOnThreads panicked as requested"
            );
            Ok(Box::new(EmptyState))
        }

        fn execute(
            &self,
            _ctx: &ExecutionContext,
            input: &DataChunk,
            output: &mut DataChunk,
            _gstate: &dyn GlobalOperatorState,
            _lstate: &mut dyn OperatorState,
        ) -> Result<OperatorResult> {
            *output = input.clone();
            Ok(OperatorResult::NeedMoreInput)
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn panics_are_propagated() {
        // A panic of the first thread (on the task that polls the stream), of
        // the other threads (on worker tasks), and of pipelines that do not
        // run alone.
        for (threads, order_matters) in [(0..1, None), (1..4, None), (0..4, Some(false))]
        {
            let input = || source(batches(0, 40, 10));
            let input = match order_matters {
                Some(order_matters) => Arc::new(Union {
                    children: vec![input(), input()],
                    order_matters,
                }),
                None => input(),
            };
            let root: Arc<dyn PhysicalOperator> = Arc::new(PanicOnThreads {
                children: vec![input],
                threads: threads.clone(),
            });
            let error = SpawnedTask::spawn(run(root, 4)).join().await.unwrap_err();
            assert!(error.is_panic(), "threads {threads:?}: {error}");
            let panic = error.into_panic();
            let message = panic
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| panic.downcast_ref::<String>().map(String::as_str));
            assert_eq!(message, Some("PanicOnThreads panicked as requested"));
        }
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

    /// An operator or sink (depending on `as_sink`) that records how many
    /// threads call it at the same time.
    #[derive(Debug)]
    struct ConcurrencyProbe {
        children: Vec<Arc<dyn PhysicalOperator>>,
        as_sink: bool,
        parallel: bool,
        active: AtomicUsize,
        max_active: AtomicUsize,
        rows: AtomicUsize,
    }

    impl ConcurrencyProbe {
        fn new(child: Arc<dyn PhysicalOperator>, as_sink: bool, parallel: bool) -> Self {
            Self {
                children: vec![child],
                as_sink,
                parallel,
                active: AtomicUsize::new(0),
                max_active: AtomicUsize::new(0),
                rows: AtomicUsize::new(0),
            }
        }

        /// Simulate work while recording concurrent calls.
        fn probe(&self, rows: usize) {
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_active.fetch_max(active, Ordering::SeqCst);
            std::thread::sleep(std::time::Duration::from_micros(200));
            self.rows.fetch_add(rows, Ordering::SeqCst);
            self.active.fetch_sub(1, Ordering::SeqCst);
        }
    }

    #[derive(Default)]
    struct ProbeSource {
        done: AtomicBool,
    }

    impl GlobalSourceState for ProbeSource {}

    impl PhysicalOperator for ConcurrencyProbe {
        fn name(&self) -> &str {
            "ConcurrencyProbe"
        }

        fn schema(&self) -> SchemaRef {
            schema()
        }

        fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
            &self.children
        }

        fn parallel_operator(&self) -> bool {
            self.parallel
        }

        fn execute(
            &self,
            _ctx: &ExecutionContext,
            input: &DataChunk,
            output: &mut DataChunk,
            _gstate: &dyn GlobalOperatorState,
            _lstate: &mut dyn OperatorState,
        ) -> Result<OperatorResult> {
            self.probe(input.len());
            *output = input.clone();
            Ok(OperatorResult::NeedMoreInput)
        }

        fn is_sink(&self) -> bool {
            self.as_sink
        }

        fn parallel_sink(&self) -> bool {
            self.parallel
        }

        fn sink(
            &self,
            _ctx: &ExecutionContext,
            input: &DataChunk,
            _gstate: &dyn GlobalSinkState,
            _lstate: &mut dyn LocalSinkState,
        ) -> Result<SinkResult> {
            self.probe(input.len());
            Ok(SinkResult::NeedMoreInput)
        }

        fn is_source(&self) -> bool {
            self.as_sink
        }

        fn get_global_source_state(
            &self,
            _ctx: &ExecutionContext,
        ) -> Result<Arc<dyn GlobalSourceState>> {
            Ok(Arc::new(ProbeSource::default()))
        }

        fn get_data(
            &self,
            _ctx: &ExecutionContext,
            gstate: &dyn GlobalSourceState,
            _lstate: &mut dyn LocalSourceState,
        ) -> Result<SourceResult> {
            let gstate = gstate.downcast_ref::<ProbeSource>().unwrap();
            if gstate.done.swap(true, Ordering::SeqCst) {
                return Ok(SourceResult::Finished);
            }
            let rows = self.rows.load(Ordering::SeqCst) as i64;
            let batch = RecordBatch::try_new(
                schema(),
                vec![Arc::new(Int64Array::from(vec![rows]))],
            )?;
            Ok(SourceResult::Chunk(DataChunk::from_record_batch(batch)))
        }
    }

    fn four_way_union() -> Arc<dyn PhysicalOperator> {
        let union = |a: Arc<dyn PhysicalOperator>, b: Arc<dyn PhysicalOperator>| {
            Arc::new(Union {
                children: vec![a, b],
                order_matters: false,
            }) as Arc<dyn PhysicalOperator>
        };
        union(
            union(source(batches(0, 20, 10)), source(batches(200, 20, 10))),
            union(source(batches(400, 20, 10)), source(batches(600, 20, 10))),
        )
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn non_parallel_sinks_are_never_called_concurrently() -> Result<()> {
        // Four union pipelines sink into one non-parallel sink.
        let probe = Arc::new(ConcurrencyProbe::new(four_way_union(), true, false));
        let output = run(Arc::clone(&probe) as _, 4).await?;
        assert_eq!(values(&output), vec![800]);
        assert_eq!(probe.max_active.load(Ordering::SeqCst), 1);

        // A parallel sink is called concurrently (control for the probe;
        // retried, as concurrency is not guaranteed on a loaded machine).
        let mut concurrent = false;
        for _ in 0..10 {
            let probe = Arc::new(ConcurrencyProbe::new(four_way_union(), true, true));
            let output = run(Arc::clone(&probe) as _, 4).await?;
            assert_eq!(values(&output), vec![800]);
            if probe.max_active.load(Ordering::SeqCst) > 1 {
                concurrent = true;
                break;
            }
        }
        assert!(concurrent);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn non_parallel_operators_are_never_called_concurrently() -> Result<()> {
        // A non-parallel operator above a union is part of all four union
        // pipelines.
        let probe = Arc::new(ConcurrencyProbe::new(four_way_union(), false, false));
        let output = run(Arc::clone(&probe) as _, 4).await?;
        assert_eq!(sorted_values(&output).len(), 800);
        assert_eq!(probe.max_active.load(Ordering::SeqCst), 1);

        // Control: a parallel operator is called concurrently.
        let mut concurrent = false;
        for _ in 0..10 {
            let probe = Arc::new(ConcurrencyProbe::new(four_way_union(), false, true));
            let output = run(Arc::clone(&probe) as _, 4).await?;
            assert_eq!(sorted_values(&output).len(), 800);
            if probe.max_active.load(Ordering::SeqCst) > 1 {
                concurrent = true;
                break;
            }
        }
        assert!(concurrent);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn serialized_pipelines_keep_their_dependencies() -> Result<()> {
        // The non-parallel operator is part of the probe pipeline, the union
        // pipeline and the child pipeline of `ProbeThenScan`, which already
        // depend on each other (depending on `order_matters`).
        for order_matters in [false, true] {
            let union: Arc<dyn PhysicalOperator> = Arc::new(Union {
                children: vec![source(batches(0, 5, 10)), source(batches(50, 5, 10))],
                order_matters,
            });
            let probe = Arc::new(ConcurrencyProbe::new(
                Arc::new(ProbeThenScan {
                    no_output_if_empty: false,
                    children: vec![union, source(batches(1000, 2, 3))],
                }),
                false,
                false,
            ));
            let output = run(Arc::clone(&probe) as _, 4).await?;
            let mut expected = (0..100).chain(1000..1006).collect::<Vec<_>>();
            expected.push(100);
            expected.sort_unstable();
            assert_eq!(sorted_values(&output), expected);
            assert_eq!(probe.max_active.load(Ordering::SeqCst), 1);
        }
        Ok(())
    }

    /// A source that records how many threads call it at the same time.
    /// Every pipeline it is the source of reads `chunks` chunks.
    #[derive(Debug)]
    struct ProbedSource {
        parallel: bool,
        chunks: usize,
        active: AtomicUsize,
        max_active: AtomicUsize,
    }

    struct ProbedSourceState {
        next: AtomicUsize,
    }

    impl GlobalSourceState for ProbedSourceState {}

    impl PhysicalOperator for ProbedSource {
        fn name(&self) -> &str {
            "ProbedSource"
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
            self.parallel
        }

        fn get_global_source_state(
            &self,
            _ctx: &ExecutionContext,
        ) -> Result<Arc<dyn GlobalSourceState>> {
            Ok(Arc::new(ProbedSourceState {
                next: AtomicUsize::new(0),
            }))
        }

        fn get_data(
            &self,
            _ctx: &ExecutionContext,
            gstate: &dyn GlobalSourceState,
            _lstate: &mut dyn LocalSourceState,
        ) -> Result<SourceResult> {
            let gstate = gstate.downcast_ref::<ProbedSourceState>().unwrap();
            let active = self.active.fetch_add(1, Ordering::SeqCst) + 1;
            self.max_active.fetch_max(active, Ordering::SeqCst);
            std::thread::sleep(std::time::Duration::from_micros(200));
            self.active.fetch_sub(1, Ordering::SeqCst);
            let index = gstate.next.fetch_add(1, Ordering::SeqCst);
            Ok(if index < self.chunks {
                SourceResult::Chunk(DataChunk::from_record_batch(
                    batches(index as i64 * 10, 1, 10).remove(0),
                ))
            } else {
                SourceResult::Finished
            })
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn non_parallel_sources_are_never_called_concurrently() -> Result<()> {
        // The same source operator is the source of both union pipelines.
        let max_active = |parallel: bool| async move {
            let source = Arc::new(ProbedSource {
                parallel,
                chunks: 20,
                active: AtomicUsize::new(0),
                max_active: AtomicUsize::new(0),
            });
            let root: Arc<dyn PhysicalOperator> = Arc::new(Union {
                children: vec![Arc::clone(&source) as _, Arc::clone(&source) as _],
                order_matters: false,
            });
            let output = run(root, 4).await?;
            assert_eq!(values(&output).len(), 400);
            Ok::<_, DataFusionError>(source.max_active.load(Ordering::SeqCst))
        };
        assert_eq!(max_active(false).await?, 1);
        // Control: a parallel source is called concurrently.
        let mut concurrent = false;
        for _ in 0..10 {
            if max_active(true).await? > 1 {
                concurrent = true;
                break;
            }
        }
        assert!(concurrent);
        Ok(())
    }

    /// Emits its input `times` times with `HaveMoreOutput`, counting calls.
    #[derive(Debug)]
    struct Spin {
        children: Vec<Arc<dyn PhysicalOperator>>,
        times: usize,
        calls: AtomicUsize,
    }

    impl PhysicalOperator for Spin {
        fn name(&self) -> &str {
            "Spin"
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
            let calls = self.calls.fetch_add(1, Ordering::SeqCst) + 1;
            *output = input.clone();
            Ok(if calls < self.times {
                OperatorResult::HaveMoreOutput
            } else {
                OperatorResult::NeedMoreInput
            })
        }
    }

    #[test]
    fn push_loop_yields_within_the_cooperative_budget() -> Result<()> {
        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_time()
            .build()
            .unwrap();
        runtime.block_on(async {
            const TIMES: usize = 2_000_000;
            // `Spin` produces output forever (for practical purposes), which
            // the filter downstream discards: no output reaches the sink.
            let spin = Arc::new(Spin {
                children: vec![source(batches(1, 1, 1))],
                times: TIMES,
                calls: AtomicUsize::new(0),
            });
            let root: Arc<dyn PhysicalOperator> = Arc::new(KeepMultiples {
                children: vec![Arc::clone(&spin) as _],
                divisor: 2,
            });

            // A task that competes for the single thread of the runtime.
            let ticks = Arc::new(AtomicUsize::new(0));
            let ticker = {
                let ticks = Arc::clone(&ticks);
                SpawnedTask::spawn(async move {
                    loop {
                        ticks.fetch_add(1, Ordering::SeqCst);
                        tokio::task::yield_now().await;
                    }
                })
            };

            let mut stream = Executor::execute_stream(root, task_ctx(), 1)?;
            let next =
                tokio::time::timeout(std::time::Duration::from_millis(50), stream.next())
                    .await;
            assert!(next.is_err(), "the pipeline should still be running");
            let calls = spin.calls.load(Ordering::SeqCst);
            assert!(calls > 0 && calls < TIMES, "{calls} calls");
            // The ticker ran while the pipeline was busy, regularly: the
            // pipeline yields after every `CALLS_PER_POLL` operator calls
            // (independently of how much CPU time the test gets).
            let ticks = ticks.load(Ordering::SeqCst);
            assert!(
                ticks >= 2 && ticks * CALLS_PER_POLL * 8 >= calls,
                "{ticks} ticks, {calls} calls"
            );

            // Dropping the stream stops the pipeline promptly.
            drop(stream);
            tokio::time::sleep(std::time::Duration::from_millis(10)).await;
            let after_drop = spin.calls.load(Ordering::SeqCst);
            tokio::time::sleep(std::time::Duration::from_millis(20)).await;
            assert_eq!(spin.calls.load(Ordering::SeqCst), after_drop);
            assert!(after_drop < TIMES);
            drop(ticker);
            Ok(())
        })
    }

    /// A parallel source on which threads `0..idle_threads` (all threads if
    /// `None`) block forever, as on an idle streaming source, while the other
    /// threads hand out `batches`.
    #[derive(Debug)]
    struct IdleSource {
        idle_threads: Option<usize>,
        batches: Vec<RecordBatch>,
    }

    impl IdleSource {
        fn new(idle_threads: Option<usize>, batches: Vec<RecordBatch>) -> Self {
            Self {
                idle_threads,
                batches,
            }
        }
    }

    impl PhysicalOperator for IdleSource {
        fn name(&self) -> &str {
            "IdleSource"
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
                max_threads: usize::MAX,
            }))
        }

        fn get_data(
            &self,
            ctx: &ExecutionContext,
            gstate: &dyn GlobalSourceState,
            _lstate: &mut dyn LocalSourceState,
        ) -> Result<SourceResult> {
            if self
                .idle_threads
                .is_none_or(|idle_threads| ctx.thread_id < idle_threads)
            {
                // Nothing will ever wake this thread up.
                return Ok(SourceResult::Blocked(TaskInterrupt::waker_registered()));
            }
            let gstate = gstate.downcast_ref::<VecSourceGlobal>().unwrap();
            let index = gstate.next.fetch_add(1, Ordering::SeqCst);
            Ok(match self.batches.get(index) {
                Some(batch) => {
                    SourceResult::Chunk(DataChunk::from_record_batch(batch.clone()))
                }
                None => SourceResult::Finished,
            })
        }
    }

    /// A parallel sink that returns `Finished` once it accepted `limit`
    /// chunks over all threads, and emits the number of accepted chunks.
    /// Threads that were already pushing a chunk when the sink finished may
    /// still deliver it.
    #[derive(Debug)]
    struct FinishAfter {
        children: Vec<Arc<dyn PhysicalOperator>>,
        limit: usize,
        local_states: AtomicUsize,
        combines: AtomicUsize,
        finalizes: AtomicUsize,
    }

    impl FinishAfter {
        fn new(child: Arc<dyn PhysicalOperator>, limit: usize) -> Self {
            Self {
                children: vec![child],
                limit,
                local_states: AtomicUsize::new(0),
                combines: AtomicUsize::new(0),
                finalizes: AtomicUsize::new(0),
            }
        }
    }

    impl PhysicalOperator for FinishAfter {
        fn name(&self) -> &str {
            "FinishAfter"
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
            Ok(Arc::new(StopAfterGlobal::default()))
        }

        fn get_local_sink_state(
            &self,
            _ctx: &ExecutionContext,
            _gstate: &dyn GlobalSinkState,
        ) -> Result<Box<dyn LocalSinkState>> {
            self.local_states.fetch_add(1, Ordering::SeqCst);
            Ok(Box::new(EmptyState))
        }

        fn sink(
            &self,
            _ctx: &ExecutionContext,
            _input: &DataChunk,
            gstate: &dyn GlobalSinkState,
            _lstate: &mut dyn LocalSinkState,
        ) -> Result<SinkResult> {
            let gstate = gstate.downcast_ref::<StopAfterGlobal>().unwrap();
            let chunks = gstate.chunks.fetch_add(1, Ordering::SeqCst) + 1;
            Ok(if chunks >= self.limit {
                SinkResult::Finished
            } else {
                SinkResult::NeedMoreInput
            })
        }

        fn combine(
            &self,
            _ctx: &ExecutionContext,
            _gstate: &dyn GlobalSinkState,
            _lstate: &mut dyn LocalSinkState,
        ) -> Result<()> {
            self.combines.fetch_add(1, Ordering::SeqCst);
            Ok(())
        }

        fn finalize(
            &self,
            _ctx: &ExecutionContext,
            _gstate: &dyn GlobalSinkState,
        ) -> Result<SinkFinalize> {
            self.finalizes.fetch_add(1, Ordering::SeqCst);
            Ok(SinkFinalize::Ready)
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
            let chunks = sink.chunks.swap(0, Ordering::SeqCst);
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

    async fn run_with_timeout(
        root: Arc<dyn PhysicalOperator>,
        threads: usize,
    ) -> Result<Vec<RecordBatch>> {
        tokio::time::timeout(std::time::Duration::from_secs(10), run(root, threads))
            .await
            .expect("the execution does not complete")
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn finished_sink_stops_threads_blocked_in_their_source() -> Result<()> {
        // Thread 0 blocks forever in its source, thread 1 delivers a chunk
        // and the sink finishes.
        let sink = Arc::new(FinishAfter::new(
            Arc::new(IdleSource::new(Some(1), batches(0, 10, 10))),
            1,
        ));
        let output = run_with_timeout(Arc::clone(&sink) as _, 2).await?;
        assert_eq!(values(&output), vec![1]);
        assert_eq!(sink.local_states.load(Ordering::SeqCst), 2);
        assert_eq!(sink.combines.load(Ordering::SeqCst), 2);
        assert_eq!(sink.finalizes.load(Ordering::SeqCst), 1);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn finished_sink_stops_all_pipelines_into_the_sink() -> Result<()> {
        for (order_matters, threads) in [(true, 1), (false, 1), (false, 4)] {
            // A ready first input finishes the sink, the second input would
            // block forever: ordered, it has not started yet; unordered, it
            // is blocked.
            let union: Arc<dyn PhysicalOperator> = Arc::new(Union {
                children: vec![
                    source(batches(0, 5, 10)),
                    Arc::new(IdleSource::new(None, vec![])),
                ],
                order_matters,
            });
            let sink = Arc::new(FinishAfter::new(union, 1));
            let output = run_with_timeout(Arc::clone(&sink) as _, threads).await?;
            // Threads stop as soon as they see that the sink finished: each
            // of them delivers at most the chunk it was pushing.
            let accepted = values(&output);
            assert_eq!(accepted.len(), 1);
            assert!((1..=threads as i64).contains(&accepted[0]), "{accepted:?}");
            // Every thread of both pipelines combines (a pipeline that starts
            // after the sink finished runs on a single thread), the sink is
            // finalized once.
            let combines = sink.combines.load(Ordering::SeqCst);
            assert_eq!(combines, sink.local_states.load(Ordering::SeqCst));
            if threads == 1 {
                assert_eq!(combines, 2);
            } else {
                assert!(
                    (threads + 1..=2 * threads).contains(&combines),
                    "{combines}"
                );
            }
            assert_eq!(sink.finalizes.load(Ordering::SeqCst), 1);
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn finished_operator_stops_all_threads_and_pipelines() -> Result<()> {
        // Thread 0 blocks forever in its source, thread 1 reaches the limit.
        let root: Arc<dyn PhysicalOperator> = Arc::new(Limit {
            children: vec![Arc::new(IdleSource::new(Some(1), batches(0, 10, 10)))],
            limit: 5,
        });
        let output = run_with_timeout(root, 2).await?;
        assert_eq!(values(&output), (0..5).collect::<Vec<_>>());

        for (order_matters, threads) in [(true, 1), (false, 1), (false, 4)] {
            let union: Arc<dyn PhysicalOperator> = Arc::new(Union {
                children: vec![
                    source(batches(0, 5, 10)),
                    Arc::new(IdleSource::new(None, vec![])),
                ],
                order_matters,
            });
            let root: Arc<dyn PhysicalOperator> = Arc::new(Limit {
                children: vec![union],
                limit: 5,
            });
            let output = run_with_timeout(root, threads).await?;
            assert_eq!(values(&output).len(), 5);
        }
        Ok(())
    }

    /// Blocks on every input chunk with a `TaskInterrupt::wait_for` future
    /// that does work the retry depends on: if the executor called it again
    /// before the future completed, it would wait forever.
    #[derive(Debug)]
    struct AwaitEachChunk {
        children: Vec<Arc<dyn PhysicalOperator>>,
        delay: std::time::Duration,
        as_sink: bool,
        rows: AtomicUsize,
    }

    impl AwaitEachChunk {
        fn new(child: Arc<dyn PhysicalOperator>, as_sink: bool) -> Self {
            Self {
                children: vec![child],
                delay: std::time::Duration::from_millis(50),
                as_sink,
                rows: AtomicUsize::new(0),
            }
        }

        /// `Ok(true)` once the future of the current chunk completed.
        fn ready(&self, state: &mut AwaitState) -> Result<bool, TaskInterrupt> {
            match state.done.take() {
                None => {
                    let done = Arc::new(AtomicBool::new(false));
                    state.done = Some(Arc::clone(&done));
                    let delay = self.delay;
                    Err(TaskInterrupt::wait_for(async move {
                        tokio::time::sleep(delay).await;
                        done.store(true, Ordering::SeqCst);
                    }))
                }
                Some(done) if done.load(Ordering::SeqCst) => Ok(true),
                Some(done) => {
                    // Called again before the future completed: the work it
                    // does is lost, nothing will wake this thread up.
                    state.done = Some(done);
                    Err(TaskInterrupt::waker_registered())
                }
            }
        }
    }

    #[derive(Default)]
    struct AwaitState {
        done: Option<Arc<AtomicBool>>,
    }

    impl OperatorState for AwaitState {}
    impl LocalSinkState for AwaitState {}

    impl PhysicalOperator for AwaitEachChunk {
        fn name(&self) -> &str {
            "AwaitEachChunk"
        }

        fn schema(&self) -> SchemaRef {
            schema()
        }

        fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
            &self.children
        }

        fn get_operator_state(
            &self,
            _ctx: &ExecutionContext,
            _gstate: &dyn GlobalOperatorState,
        ) -> Result<Box<dyn OperatorState>> {
            Ok(Box::new(AwaitState::default()))
        }

        fn execute(
            &self,
            _ctx: &ExecutionContext,
            input: &DataChunk,
            output: &mut DataChunk,
            _gstate: &dyn GlobalOperatorState,
            lstate: &mut dyn OperatorState,
        ) -> Result<OperatorResult> {
            let state = lstate.downcast_mut::<AwaitState>().unwrap();
            match self.ready(state) {
                Ok(_) => {
                    *output = input.clone();
                    Ok(OperatorResult::NeedMoreInput)
                }
                Err(interrupt) => Ok(OperatorResult::Blocked(interrupt)),
            }
        }

        fn is_sink(&self) -> bool {
            self.as_sink
        }

        fn parallel_sink(&self) -> bool {
            true
        }

        fn get_local_sink_state(
            &self,
            _ctx: &ExecutionContext,
            _gstate: &dyn GlobalSinkState,
        ) -> Result<Box<dyn LocalSinkState>> {
            Ok(Box::new(AwaitState::default()))
        }

        fn sink(
            &self,
            _ctx: &ExecutionContext,
            input: &DataChunk,
            _gstate: &dyn GlobalSinkState,
            lstate: &mut dyn LocalSinkState,
        ) -> Result<SinkResult> {
            let state = lstate.downcast_mut::<AwaitState>().unwrap();
            match self.ready(state) {
                Ok(_) => {
                    self.rows.fetch_add(input.len(), Ordering::SeqCst);
                    Ok(SinkResult::NeedMoreInput)
                }
                Err(interrupt) => Ok(SinkResult::Blocked(interrupt)),
            }
        }

        fn is_source(&self) -> bool {
            self.as_sink
        }

        fn get_global_source_state(
            &self,
            _ctx: &ExecutionContext,
        ) -> Result<Arc<dyn GlobalSourceState>> {
            Ok(Arc::new(ProbeSource::default()))
        }

        fn get_data(
            &self,
            _ctx: &ExecutionContext,
            gstate: &dyn GlobalSourceState,
            _lstate: &mut dyn LocalSourceState,
        ) -> Result<SourceResult> {
            let gstate = gstate.downcast_ref::<ProbeSource>().unwrap();
            if gstate.done.swap(true, Ordering::SeqCst) {
                return Ok(SourceResult::Finished);
            }
            let rows = self.rows.load(Ordering::SeqCst) as i64;
            let batch = RecordBatch::try_new(
                schema(),
                vec![Arc::new(Int64Array::from(vec![rows]))],
            )?;
            Ok(SourceResult::Chunk(DataChunk::from_record_batch(batch)))
        }
    }

    /// A parallel source: thread 0 emits `0..10` right away, thread 1 emits
    /// `100..110` after `delay`.
    #[derive(Debug)]
    struct StaggeredSource {
        delay: std::time::Duration,
    }

    #[derive(Default)]
    struct StaggeredState {
        waited: bool,
        emitted: bool,
    }

    impl LocalSourceState for StaggeredState {}

    impl PhysicalOperator for StaggeredSource {
        fn name(&self) -> &str {
            "StaggeredSource"
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

        fn get_local_source_state(
            &self,
            _ctx: &ExecutionContext,
            _gstate: &dyn GlobalSourceState,
        ) -> Result<Box<dyn LocalSourceState>> {
            Ok(Box::new(StaggeredState::default()))
        }

        fn get_data(
            &self,
            ctx: &ExecutionContext,
            _gstate: &dyn GlobalSourceState,
            lstate: &mut dyn LocalSourceState,
        ) -> Result<SourceResult> {
            let state = lstate.downcast_mut::<StaggeredState>().unwrap();
            if state.emitted || ctx.thread_id > 1 {
                return Ok(SourceResult::Finished);
            }
            if ctx.thread_id == 1 && !state.waited {
                state.waited = true;
                return Ok(SourceResult::Blocked(TaskInterrupt::wait_for(
                    tokio::time::sleep(self.delay),
                )));
            }
            state.emitted = true;
            let start = ctx.thread_id as i64 * 100;
            Ok(SourceResult::Chunk(DataChunk::from_record_batch(
                batches(start, 1, 10).remove(0),
            )))
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn finished_operator_keeps_the_waits_of_downstream_work() -> Result<()> {
        // One thread: `Limit` finishes with its last rows, which
        // `AwaitEachChunk` (an operator, or the sink) downstream blocks on.
        let root: Arc<dyn PhysicalOperator> = Arc::new(AwaitEachChunk::new(
            Arc::new(Limit {
                children: vec![source(batches(0, 3, 10))],
                limit: 5,
            }),
            false,
        ));
        let output = run_with_timeout(root, 1).await?;
        assert_eq!(values(&output), (0..5).collect::<Vec<_>>());

        let sink = Arc::new(AwaitEachChunk::new(
            Arc::new(Limit {
                children: vec![source(batches(0, 3, 10))],
                limit: 5,
            }),
            true,
        ));
        let output = run_with_timeout(Arc::clone(&sink) as _, 1).await?;
        assert_eq!(values(&output), vec![5]);

        // Two threads: thread 0 waits in `AwaitEachChunk` (an operator, or
        // the sink) when `Limit` finishes on thread 1.
        for as_sink in [false, true] {
            let delay = std::time::Duration::from_millis(10);
            let root = Arc::new(AwaitEachChunk::new(
                Arc::new(Limit {
                    children: vec![Arc::new(StaggeredSource { delay })],
                    limit: 15,
                }),
                as_sink,
            ));
            let output = run_with_timeout(Arc::clone(&root) as _, 2).await?;
            if as_sink {
                assert_eq!(values(&output), vec![15]);
            } else {
                let expected = (0..10).chain(100..105).collect::<Vec<_>>();
                assert_eq!(sorted_values(&output), expected);
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn finished_sink_cancels_child_meta_pipelines_it_no_longer_needs() -> Result<()>
    {
        // The second input of the union reads from a pipeline breaker whose
        // input never produces anything. Once the sink finished, the breaker
        // is not needed any more.
        for (order_matters, threads, nested) in [
            (true, 1, false),
            (false, 1, false),
            (true, 4, false),
            (false, 4, false),
            (true, 4, true),
            (false, 4, true),
        ] {
            let breaker =
                Arc::new(Materialize::new(Arc::new(IdleSource::new(None, vec![]))));
            // With `nested`, the breaker is read by another breaker, whose
            // output is not needed either.
            let input: Arc<dyn PhysicalOperator> = if nested {
                Arc::new(Materialize::new(Arc::clone(&breaker) as _))
            } else {
                Arc::clone(&breaker) as _
            };
            let union: Arc<dyn PhysicalOperator> = Arc::new(Union {
                children: vec![source(batches(0, 5, 10)), input],
                order_matters,
            });
            let sink = Arc::new(FinishAfter::new(union, 1));
            let output = run_with_timeout(Arc::clone(&sink) as _, threads).await?;
            let accepted = values(&output);
            assert_eq!(accepted.len(), 1);
            assert!((1..=threads as i64).contains(&accepted[0]), "{accepted:?}");
            // Every local sink state is combined, every sink finalized once.
            assert_eq!(
                sink.combines.load(Ordering::SeqCst),
                sink.local_states.load(Ordering::SeqCst)
            );
            assert_eq!(sink.finalizes.load(Ordering::SeqCst), 1);
            assert_eq!(
                breaker.combines.load(Ordering::SeqCst),
                breaker.local_states.load(Ordering::SeqCst)
            );
            assert_eq!(breaker.finalizes.load(Ordering::SeqCst), 1);
            // The breaker produced nothing.
            assert_eq!(breaker.get_data_calls.load(Ordering::SeqCst), 0);
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn no_output_possible_cancels_child_meta_pipelines_it_no_longer_needs()
    -> Result<()> {
        // The build side is empty: the probe pipeline is skipped, so the
        // pipeline breaker on the probe side, whose input never produces
        // anything, is not needed.
        for threads in [1, 4] {
            let breaker =
                Arc::new(Materialize::new(Arc::new(IdleSource::new(None, vec![]))));
            let join: Arc<dyn PhysicalOperator> = Arc::new(ProbeThenScan {
                children: vec![Arc::clone(&breaker) as _, source(vec![])],
                no_output_if_empty: true,
            });
            let output = run_with_timeout(join, threads).await?;
            assert!(output.is_empty());
            assert_eq!(
                breaker.combines.load(Ordering::SeqCst),
                breaker.local_states.load(Ordering::SeqCst)
            );
            assert_eq!(breaker.finalizes.load(Ordering::SeqCst), 1);
        }
        Ok(())
    }

    /// A source that emits `value` once, from a single thread, after `delay`.
    #[derive(Debug)]
    struct DelayedSource {
        value: i64,
        delay: std::time::Duration,
    }

    impl PhysicalOperator for DelayedSource {
        fn name(&self) -> &str {
            "DelayedSource"
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

        fn get_local_source_state(
            &self,
            _ctx: &ExecutionContext,
            _gstate: &dyn GlobalSourceState,
        ) -> Result<Box<dyn LocalSourceState>> {
            Ok(Box::new(StaggeredState::default()))
        }

        fn get_data(
            &self,
            _ctx: &ExecutionContext,
            _gstate: &dyn GlobalSourceState,
            lstate: &mut dyn LocalSourceState,
        ) -> Result<SourceResult> {
            let state = lstate.downcast_mut::<StaggeredState>().unwrap();
            if !state.waited {
                state.waited = true;
                return Ok(SourceResult::Blocked(TaskInterrupt::wait_for(
                    tokio::time::sleep(self.delay),
                )));
            }
            if state.emitted {
                return Ok(SourceResult::Finished);
            }
            state.emitted = true;
            Ok(SourceResult::Chunk(DataChunk::from_record_batch(
                batches(self.value, 1, 1).remove(0),
            )))
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn ordered_unions_keep_the_order_of_nested_inputs() -> Result<()> {
        let a = || source(batches(0, 1, 1));
        let b = || -> Arc<dyn PhysicalOperator> {
            Arc::new(DelayedSource {
                value: 1,
                delay: std::time::Duration::from_millis(30),
            })
        };
        let c = || source(batches(2, 1, 1));
        let union = |left: Arc<dyn PhysicalOperator>,
                     right: Arc<dyn PhysicalOperator>| {
            Arc::new(Union {
                children: vec![left, right],
                order_matters: true,
            }) as Arc<dyn PhysicalOperator>
        };
        for root in [union(union(a(), b()), c()), union(a(), union(b(), c()))] {
            let output = run_with_timeout(root, 4).await?;
            assert_eq!(values(&output), vec![0, 1, 2]);
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn work_still_needed_by_other_pipelines_is_not_cancelled() -> Result<()> {
        // Both probe pipelines of the join (one per input of the union below
        // it) wait for its (slow) build side. The first one is skipped as its
        // source cannot produce output; the second one still needs the build
        // side, which must not be cancelled.
        let empty = Arc::new(Materialize {
            no_output_if_empty: true,
            ..Materialize::new(source(vec![]))
        });
        let union: Arc<dyn PhysicalOperator> = Arc::new(Union {
            children: vec![empty, Arc::new(Materialize::new(source(batches(0, 2, 5))))],
            order_matters: false,
        });
        let join: Arc<dyn PhysicalOperator> = Arc::new(ProbeThenScan {
            children: vec![
                union,
                Arc::new(DelayedSource {
                    value: 100,
                    delay: std::time::Duration::from_millis(30),
                }),
            ],
            no_output_if_empty: false,
        });
        let output = run_with_timeout(join, 4).await?;
        // The probe rows, then the build rows and the number of probed rows.
        let mut expected = (0..10).collect::<Vec<_>>();
        expected.extend([10, 100]);
        assert_eq!(sorted_values(&output), expected);
        Ok(())
    }

    /// A union of any number of inputs, built with
    /// `MetaPipelineBuilder::build_union`.
    #[derive(Debug)]
    struct UnionAll {
        children: Vec<Arc<dyn PhysicalOperator>>,
        order_matters: bool,
    }

    impl PhysicalOperator for UnionAll {
        fn name(&self) -> &str {
            "UnionAll"
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
            meta.build_union(current, &self.children, self.order_matters)
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn ordered_unions_of_many_inputs_keep_their_order() -> Result<()> {
        let a = || source(batches(0, 1, 1));
        let b = || -> Arc<dyn PhysicalOperator> {
            Arc::new(DelayedSource {
                value: 1,
                delay: std::time::Duration::from_millis(30),
            })
        };
        let c = || source(batches(2, 1, 1));
        let d = || -> Arc<dyn PhysicalOperator> {
            Arc::new(DelayedSource {
                value: 3,
                delay: std::time::Duration::from_millis(10),
            })
        };
        let union = |children: Vec<Arc<dyn PhysicalOperator>>, order_matters: bool| {
            Arc::new(UnionAll {
                children,
                order_matters,
            }) as Arc<dyn PhysicalOperator>
        };

        // Every input waits for all pipelines of the inputs before it.
        let pipelines = graph(union(vec![a(), b(), c()], true))?;
        let dependencies = pipelines
            .pipelines()
            .iter()
            .map(|pipeline| pipeline.dependencies().to_vec())
            .collect::<Vec<_>>();
        assert_eq!(dependencies, vec![vec![], vec![0], vec![0, 1]]);

        for root in [
            union(vec![a(), b(), c(), d()], true),
            union(vec![union(vec![a(), b()], true), c(), d()], true),
            union(vec![a(), union(vec![b(), c()], true), d()], true),
            // Nested binary and n-ary unions.
            union(
                vec![
                    Arc::new(Union {
                        children: vec![a(), b()],
                        order_matters: true,
                    }),
                    union(vec![c(), d()], true),
                ],
                true,
            ),
        ] {
            let output = run_with_timeout(root, 4).await?;
            assert_eq!(values(&output), vec![0, 1, 2, 3]);
        }

        // Without order, the inputs run concurrently.
        let pipelines = graph(union(vec![a(), b(), c()], false))?;
        assert!(
            pipelines
                .pipelines()
                .iter()
                .all(|pipeline| pipeline.dependencies().is_empty())
        );
        let output = run_with_timeout(union(vec![a(), b(), c(), d()], false), 4).await?;
        assert_eq!(sorted_values(&output), vec![0, 1, 2, 3]);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn build_side_needed_by_a_child_pipeline_is_not_cancelled() -> Result<()> {
        // A "right outer join" whose probe side cannot produce output: the
        // probe pipelines are skipped, but the child pipeline that scans the
        // (slow) build side for unmatched rows still needs it.
        let empty = || -> Arc<dyn PhysicalOperator> {
            Arc::new(Materialize {
                no_output_if_empty: true,
                ..Materialize::new(source(vec![]))
            })
        };
        let join = |probe: Arc<dyn PhysicalOperator>| -> Arc<dyn PhysicalOperator> {
            Arc::new(ProbeThenScan {
                children: vec![
                    probe,
                    Arc::new(DelayedSource {
                        value: 100,
                        delay: std::time::Duration::from_millis(30),
                    }),
                ],
                no_output_if_empty: false,
            })
        };
        for threads in [1, 4] {
            for root in [
                join(empty()),
                // The child pipeline also has the operators above the join.
                Arc::new(Repeat {
                    children: vec![join(empty())],
                    times: 1,
                }),
                // Two skipped probe pipelines, one per input of the union.
                join(Arc::new(Union {
                    children: vec![empty(), empty()],
                    order_matters: false,
                })),
            ] {
                let output = run_with_timeout(root, threads).await?;
                // The build rows and the number of probed rows.
                assert_eq!(sorted_values(&output), vec![0, 100], "{threads} threads");
            }
        }
        Ok(())
    }
}
