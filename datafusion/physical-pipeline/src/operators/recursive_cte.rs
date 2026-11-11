// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`PhysicalRecursiveCte`] and [`WorkTableScanSource`]: native recursive
//! queries (`WITH RECURSIVE`), which replace DataFusion's `RecursiveQueryExec`
//! and the `WorkTableExec`s of its recursive term.
//!
//! A recursive query has a *static term* and a *recursive term*, which reads
//! the rows that the term before it produced from the *working table*. The
//! query returns the rows of the static term, then the rows of the recursive
//! term evaluated over them (the first iteration), then those of the
//! recursive term evaluated over the rows of the first iteration, and so on,
//! until an iteration produces no new rows. With `UNION` (a distinct
//! recursive query), a row that a term produced before is not new: it is
//! neither returned again nor put in the working table, so the recursion ends
//! once an iteration only produces rows that it returned before. With
//! `UNION ALL`, every row is new.
//!
//! [`PhysicalRecursiveCte`] works like DuckDB's `PhysicalRecursiveCTE`:
//!
//! * it is the **sink** of the pipelines of the static term (its child meta
//!   pipeline, which runs on all threads) and, through a `RecursiveTermSink`,
//!   of the pipelines of every iteration of the recursive term: both collect
//!   the rows of their term into the same `Collector`;
//! * it is the **source** of the pipeline that reads the result of the
//!   query: it emits the rows of the static term, then runs the first
//!   iteration over them as the working table, emits its rows, runs the next
//!   iteration over them, and so on, until an iteration collected no rows.
//!
//! So a `LIMIT` above the query stops the recursion once it has its rows, as
//! in DataFusion, which only runs the next iteration once the rows of the
//! last one were read: queries whose recursion would not end (`SELECT ...
//! FROM t LIMIT 10` over an infinite `UNION ALL`) end. As in DataFusion and
//! DuckDB, the rows of the static term come first, then the rows of every
//! iteration in turn: the source is not parallel, so the rows of an iteration
//! come before those of the next one (in no particular order among
//! themselves).
//!
//! # Iterations
//!
//! DuckDB reschedules the pipelines of the recursive term in its executor
//! for every iteration (`ExecuteRecursivePipelines`). Here, every iteration
//! is an execution of its own: an [`Executor`] runs the pipelines of the
//! recursive term into a `RecursiveTermSink`, which collects the rows of the
//! iteration for the CTE, on all threads of the execution of the query. The
//! source waits for the iteration ([`TaskInterrupt::wait_for`]), and the
//! execution of the iteration is dropped (and its threads aborted) if the
//! rows of the query are not needed any more.
//!
//! Like DataFusion (`reset_plan_states` in `RecursiveQueryStream`), every
//! iteration runs a fresh copy of the recursive term: its DataFusion plan is
//! reset and lowered again, so that the state that its nodes keep between
//! their executions starts afresh in every iteration: the dynamic filters of
//! hash joins, TopKs and aggregations, the results of scalar subqueries, the
//! generators of `generate_series` (which DataFusion executes), the files of
//! the scans. The recursive term that is lowered with the query is a child
//! of the operator only for `EXPLAIN` (and the plans it bridges, see
//! [`crate::bridge::bridged_plans`]): it is never executed.
//!
//! # The working table
//!
//! [`WorkTableScanSource`] replaces a `WorkTableExec` (a scan of the working
//! table) and scans the rows of the last term on all threads, without
//! copying them, with the columns that the `WorkTableExec` projects (by name:
//! the columns of the working table are those of the query). The rows reach
//! it through the task context of the execution of the iteration: an
//! extension of its [`SessionConfig`](datafusion_execution::config::SessionConfig)
//! (`WorkTables`) holds the working table of the CTE, and those of the
//! recursive queries whose recursive terms run the CTE (as their own
//! iterations do), so that the scans of a nested recursive query also read
//! the working tables of the queries around it, as SQL scopes them: a scan
//! reads the table of the innermost recursive query of its name, which is
//! how DataFusion binds them too (`assign_work_table`).
//!
//! A `WorkTableExec` that DataFusion executes reads the working table of
//! DataFusion's `RecursiveQueryExec`, which a [`PhysicalRecursiveCte`] does
//! not fill: [`PhysicalRecursiveCte::try_lower`] leaves a recursive query to
//! DataFusion if a node that reads its working table is executed by
//! DataFusion (bridged, e.g. below a node that no native operator
//! implements), or if the columns of a scan of its working table are not
//! columns of the query.
//!
//! # Memory
//!
//! The CTE keeps the rows of the last term (emitted, then the working table
//! of the next iteration) and those that the current iteration collected, and
//! for `UNION`, a hash table of all rows that it returned (DataFusion's
//! [`GroupValues`], as in DataFusion's `RecursiveQueryExec`, so that rows are
//! distinct exactly when they are in DataFusion: nulls are equal, and so are
//! `NaN`s). Their memory is reserved in the memory pool of the session as the
//! rows are collected, and the query fails with `ResourcesExhausted` once the
//! pool cannot grant it. The rows of a term are released once the iteration
//! that read them completed, so a recursion holds at most the rows of two
//! consecutive terms (and its hash table), however many iterations it runs.

use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow::array::{BooleanArray, RecordBatch, RecordBatchOptions};
use arrow::compute::filter_record_batch;
use arrow::datatypes::{Schema, SchemaRef};
use datafusion_common::instant::Instant;
use datafusion_common::{Result, internal_datafusion_err, internal_err};
use datafusion_execution::TaskContext;
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryPool, MemoryReservation};
use datafusion_physical_plan::aggregates::group_values::{GroupValues, new_group_values};
use datafusion_physical_plan::aggregates::order::GroupClustering;
use datafusion_physical_plan::execution_plan::reset_plan_states;
use datafusion_physical_plan::recursive_query::RecursiveQueryExec;
use datafusion_physical_plan::work_table::WorkTableExec;
use datafusion_physical_plan::{ExecutionPlan, ExecutionPlanProperties};
use futures::future::BoxFuture;
use parking_lot::Mutex;

use crate::bridge::bridged_plans;
use crate::chunk::{DEFAULT_VECTOR_SIZE, DataChunk};
use crate::executor::Executor;
use crate::operator::{
    ExecutionContext, GlobalSinkState, GlobalSourceState, LocalSinkState,
    LocalSourceState, OperatorMetrics, PhysicalOperator, SinkFinalize, SinkResult,
    SourceResult, TaskInterrupt, downcast_state, downcast_state_mut,
};
use crate::pipeline::{MetaPipelineBuilder, PipelineBuilder};
use crate::planner::lower_execution_plan_with_runtime;
use crate::sort::memory_size;

/// A recursive query (`WITH RECURSIVE`), DataFusion's `RecursiveQueryExec`.
///
/// See the [module documentation](self) for how it runs.
#[derive(Debug)]
pub struct PhysicalRecursiveCte {
    /// The name of the CTE, which the scans of its working table refer to.
    name: String,
    schema: SchemaRef,
    /// `UNION` rather than `UNION ALL`.
    is_distinct: bool,
    /// The static term, and the recursive term as it was lowered with the
    /// query (for `EXPLAIN`: every iteration lowers it again).
    children: Vec<Arc<dyn PhysicalOperator>>,
    /// The DataFusion plan of the recursive term, which every iteration
    /// resets and lowers.
    recursive_plan: Arc<dyn ExecutionPlan>,
}

impl PhysicalRecursiveCte {
    /// Lower a DataFusion `RecursiveQueryExec` to a [`PhysicalRecursiveCte`]
    /// with the same schema. `lower_term` lowers its static and its recursive
    /// term (in pipelines that run until their input ends: the CTE reads all
    /// of their rows). Returns `Ok(None)` for any other node.
    ///
    /// DataFusion executes the first partition of every term, which plans
    /// have exactly one of: a recursive query whose terms have other numbers
    /// of partitions, or other data types than the query, is left to
    /// DataFusion. So is a query whose working table the native operator
    /// cannot fill (see the [module documentation](self#the-working-table)):
    /// one with a scan of its working table that DataFusion executes, or
    /// whose columns are not columns of the query.
    pub fn try_lower<F>(
        plan: &Arc<dyn ExecutionPlan>,
        mut lower_term: F,
    ) -> Result<Option<Self>>
    where
        F: FnMut(&Arc<dyn ExecutionPlan>) -> Result<Arc<dyn PhysicalOperator>>,
    {
        let Some(exec) = plan.downcast_ref::<RecursiveQueryExec>() else {
            return Ok(None);
        };
        let name = exec.name();
        let schema = plan.schema();
        let terms = [exec.static_term(), exec.recursive_term()];
        if terms.iter().any(|term| {
            term.output_partitioning().partition_count() != 1
                || !same_data_types(&term.schema(), &schema)
        }) {
            return Ok(None);
        }
        let mut scans = vec![];
        work_table_scans(exec.recursive_term(), name, &mut scans);
        if scans
            .iter()
            .any(|scan| work_table_projection(&scan.schema(), &schema).is_none())
        {
            return Ok(None);
        }
        let static_term = lower_term(exec.static_term())?;
        let recursive_term = lower_term(exec.recursive_term())?;
        if bridges_work_table(&recursive_term, name) {
            return Ok(None);
        }
        Ok(Some(Self {
            name: name.to_owned(),
            schema,
            is_distinct: exec.is_distinct(),
            children: vec![static_term, recursive_term],
            recursive_plan: Arc::clone(exec.recursive_term()),
        }))
    }

    /// The name of the CTE.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Whether the query is a `UNION` (its rows are distinct) rather than a
    /// `UNION ALL`.
    pub fn is_distinct(&self) -> bool {
        self.is_distinct
    }

    /// The static term.
    pub fn static_term(&self) -> &Arc<dyn PhysicalOperator> {
        &self.children[0]
    }

    /// The recursive term, as it was lowered with the query. It is shown by
    /// `EXPLAIN`, but never executed: every iteration lowers the recursive
    /// term again, see the [module documentation](self#iterations).
    pub fn recursive_term(&self) -> &Arc<dyn PhysicalOperator> {
        &self.children[1]
    }

    /// A fresh copy of the recursive term for an iteration: its DataFusion
    /// plan, reset (see `reset_plan_states`) and lowered for the session of
    /// `task_ctx`.
    ///
    /// [`Self::try_lower`] checked that the lowering of the query, for the
    /// session it was planned for (or for any session), lowers the scans of
    /// the working table natively. The session of the execution lowers them
    /// natively too, unless it lowers fewer nodes than the lowering of the
    /// query (e.g. a plan lowered for a session without a memory limit and
    /// executed in one with a limit): the iteration then fails, as DataFusion
    /// would read a working table that the CTE does not fill.
    fn lower_recursive_term(
        &self,
        task_ctx: &TaskContext,
    ) -> Result<Arc<dyn PhysicalOperator>> {
        let plan = reset_plan_states(Arc::clone(&self.recursive_plan))?;
        let options = task_ctx.session_config().options();
        let term = lower_execution_plan_with_runtime(
            &plan,
            options.as_ref(),
            task_ctx.runtime_env().as_ref(),
        )?;
        if bridges_work_table(&term, &self.name) {
            return internal_err!(
                "The recursive term of {} reads its working table in DataFusion",
                self.description()
            );
        }
        Ok(term)
    }

    /// Start the next iteration, over `working_table`: returns the future
    /// that runs it, which the source waits for. Its outcome is set in the
    /// phase of `recursion` (`Phase::Iterating`).
    fn start_iteration(
        &self,
        ctx: &ExecutionContext,
        state: &CteState,
        recursion: &mut Recursion,
        working_table: Arc<WorkingTable>,
    ) -> Result<BoxFuture<'static, ()>> {
        let iterations = recursion
            .iterations
            .get_or_insert_with(|| Iterations::new(&self.name, &ctx.task_ctx));
        // The scans of the working table read the rows of the last term.
        *iterations.work_tables.table.lock() = Some(working_table);
        iterations.count += 1;
        let term = self.lower_recursive_term(&iterations.task_ctx)?;
        let sink: Arc<dyn PhysicalOperator> = Arc::new(RecursiveTermSink::try_new(
            term,
            Arc::clone(&state.collector),
            ctx.metrics(self),
        )?);
        let executor =
            Executor::try_new(sink, Arc::clone(&iterations.task_ctx), state.num_threads)?;
        log::trace!(
            "{}: iteration {} over {} rows",
            self.description(),
            iterations.count,
            recursion.current.as_ref().map_or(0, |table| table.rows)
        );
        let outcome = Arc::new(Mutex::new(None));
        recursion.phase = Phase::Iterating(Arc::clone(&outcome));
        Ok(Box::pin(async move {
            let result = executor.execute().await;
            *outcome.lock() = Some(result);
        }))
    }
}

/// Whether `schema` has the data types of `expected` (and as many fields).
fn same_data_types(schema: &Schema, expected: &Schema) -> bool {
    schema.fields().len() == expected.fields().len()
        && schema
            .fields()
            .iter()
            .zip(expected.fields())
            .all(|(field, expected)| field.data_type() == expected.data_type())
}

/// Collect the scans of the working table of the recursive query `name` in
/// `plan` (a DataFusion plan): the `WorkTableExec`s of that name, but those
/// in the recursive term of a recursive query of the same name in `plan`,
/// which read the working table of that query instead.
fn work_table_scans<'a>(
    plan: &'a Arc<dyn ExecutionPlan>,
    name: &str,
    scans: &mut Vec<&'a WorkTableExec>,
) {
    if let Some(scan) = plan.downcast_ref::<WorkTableExec>() {
        if scan.name() == name {
            scans.push(scan);
        }
        return;
    }
    if let Some(query) = plan.downcast_ref::<RecursiveQueryExec>()
        && query.name() == name
    {
        work_table_scans(query.static_term(), name, scans);
        return;
    }
    for child in plan.children() {
        work_table_scans(child, name, scans);
    }
}

/// Whether DataFusion executes a scan of the working table of the recursive
/// query `name` in the operator tree `operator`: in one of the plans that it
/// bridges (see [`bridged_plans`]).
fn bridges_work_table(operator: &Arc<dyn PhysicalOperator>, name: &str) -> bool {
    bridged_plans(operator).iter().any(|plan| {
        let mut scans = vec![];
        work_table_scans(plan, name, &mut scans);
        !scans.is_empty()
    })
}

/// The columns of the working table (with schema `table`, the schema of the
/// recursive query) that a scan with schema `scan` reads: the column of the
/// same name and data type of every field. `None` if a field has no such
/// column, or several.
fn work_table_projection(scan: &Schema, table: &Schema) -> Option<Vec<usize>> {
    scan.fields()
        .iter()
        .map(|field| {
            let mut columns = table
                .fields()
                .iter()
                .enumerate()
                .filter(|(_, column)| column.name() == field.name());
            match (columns.next(), columns.next()) {
                (Some((index, column)), None)
                    if column.data_type() == field.data_type() =>
                {
                    Some(index)
                }
                _ => None,
            }
        })
        .collect()
}

impl PhysicalOperator for PhysicalRecursiveCte {
    fn name(&self) -> &str {
        "PhysicalRecursiveCte"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.children
    }

    fn description(&self) -> String {
        format!(
            "PhysicalRecursiveCte: name={}, is_distinct={}",
            self.name, self.is_distinct
        )
    }

    // ---------------------------------------------------------------------
    // Sink: the rows of the static term
    // ---------------------------------------------------------------------

    fn is_sink(&self) -> bool {
        true
    }

    fn parallel_sink(&self) -> bool {
        true
    }

    fn get_global_sink_state(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalSinkState>> {
        let collector = Collector::try_new(
            &self.name,
            Arc::clone(&self.schema),
            self.is_distinct,
            ctx.task_ctx.memory_pool(),
        )?;
        Ok(Arc::new(CteState {
            collector: Arc::new(collector),
            num_threads: ctx.num_threads,
            recursion: Mutex::new(Recursion::default()),
        }))
    }

    fn get_local_sink_state(
        &self,
        _ctx: &ExecutionContext,
        _gstate: &dyn GlobalSinkState,
    ) -> Result<Box<dyn LocalSinkState>> {
        Ok(Box::new(CollectedRows::default()))
    }

    fn sink(
        &self,
        _ctx: &ExecutionContext,
        input: &DataChunk,
        gstate: &dyn GlobalSinkState,
        lstate: &mut dyn LocalSinkState,
    ) -> Result<SinkResult> {
        let state = downcast_state::<CteState>(gstate.as_any(), self.name())?;
        let local =
            downcast_state_mut::<CollectedRows>(lstate.as_any_mut(), self.name())?;
        state.collector.collect(input, local)?;
        Ok(SinkResult::NeedMoreInput)
    }

    fn combine(
        &self,
        _ctx: &ExecutionContext,
        gstate: &dyn GlobalSinkState,
        lstate: &mut dyn LocalSinkState,
    ) -> Result<()> {
        let state = downcast_state::<CteState>(gstate.as_any(), self.name())?;
        let local =
            downcast_state_mut::<CollectedRows>(lstate.as_any_mut(), self.name())?;
        state.collector.combine(local);
        Ok(())
    }

    fn finalize(
        &self,
        ctx: &ExecutionContext,
        gstate: &dyn GlobalSinkState,
    ) -> Result<SinkFinalize> {
        let state = downcast_state::<CteState>(gstate.as_any(), self.name())?;
        // Without rows from the static term, the query has none: the
        // recursion does not start. Neither does it if the rows of the query
        // are not needed any more.
        if state.collector.is_empty() || ctx.registry().is_sink_finished(self) {
            return Ok(SinkFinalize::NoOutputPossible);
        }
        Ok(SinkFinalize::Ready)
    }

    // ---------------------------------------------------------------------
    // Source: the rows of the static term, then those of every iteration
    // ---------------------------------------------------------------------

    fn is_source(&self) -> bool {
        true
    }

    fn get_data(
        &self,
        ctx: &ExecutionContext,
        _gstate: &dyn GlobalSourceState,
        _lstate: &mut dyn LocalSourceState,
    ) -> Result<SourceResult> {
        let state = ctx
            .global_sink_state(self)
            .and_then(|state| state.downcast_arc::<CteState>())
            .ok_or_else(|| {
                internal_datafusion_err!(
                    "The static term of {} was not executed",
                    self.description()
                )
            })?;
        let mut recursion = state.recursion.lock();
        loop {
            match std::mem::replace(&mut recursion.phase, Phase::Done) {
                Phase::Start => {
                    let static_rows = state.collector.take();
                    recursion.emit(static_rows);
                }
                Phase::Emitting => {
                    if let Some(chunk) = recursion.next_chunk() {
                        recursion.phase = Phase::Emitting;
                        return Ok(SourceResult::Chunk(chunk));
                    }
                    // All rows of the last term were emitted: they are the
                    // working table of the next iteration, if there are any.
                    let Some(working_table) =
                        recursion.current.clone().filter(|table| table.rows > 0)
                    else {
                        recursion.finish();
                        return Ok(SourceResult::Finished);
                    };
                    let iteration =
                        self.start_iteration(ctx, &state, &mut recursion, working_table)?;
                    return Ok(SourceResult::Blocked(TaskInterrupt::wait_for(iteration)));
                }
                Phase::Iterating(outcome) => {
                    let Some(result) = outcome.lock().take() else {
                        return internal_err!(
                            "An iteration of {} did not complete",
                            self.description()
                        );
                    };
                    if let Err(error) = result {
                        recursion.finish();
                        return Err(error);
                    }
                    // The working table is not read any more: its rows are
                    // released once the rows of the iteration replace them.
                    if let Some(iterations) = &recursion.iterations {
                        iterations.work_tables.table.lock().take();
                    }
                    let new_rows = state.collector.take();
                    recursion.emit(new_rows);
                }
                Phase::Done => return Ok(SourceResult::Finished),
            }
        }
    }

    // ---------------------------------------------------------------------
    // Pipelines
    // ---------------------------------------------------------------------

    fn build_pipelines(
        &self,
        current: &mut PipelineBuilder,
        meta: &mut MetaPipelineBuilder<'_>,
    ) -> Result<()> {
        // The CTE is the source of the pipeline that reads the query, and the
        // sink of the pipelines of the static term. The recursive term takes
        // part in no pipeline of the query: every iteration runs its own.
        let this = meta.current_operator()?;
        meta.set_source(current, Arc::clone(&this))?;
        let (mut static_term, mut pipeline) =
            meta.create_child_meta_pipeline(current, this)?;
        static_term.build(&mut pipeline, self.static_term())
    }
}

/// The global sink state of a [`PhysicalRecursiveCte`], which its source
/// role reads.
#[derive(Debug)]
struct CteState {
    /// Collects the rows of the static term, and those of every iteration.
    collector: Arc<Collector>,
    /// The number of threads of the execution of the query, on which every
    /// iteration runs too.
    num_threads: usize,
    recursion: Mutex<Recursion>,
}

impl GlobalSinkState for CteState {}

/// The progress of the source of a [`PhysicalRecursiveCte`].
#[derive(Debug, Default)]
struct Recursion {
    phase: Phase,
    /// The rows of the last term: emitted, then the working table of the next
    /// iteration.
    current: Option<Arc<WorkingTable>>,
    /// The next block of `current` to emit.
    next_block: usize,
    /// The task context of the iterations, created for the first one.
    iterations: Option<Iterations>,
}

impl Recursion {
    /// Emit `rows`, the rows of the last term.
    fn emit(&mut self, rows: Arc<WorkingTable>) {
        self.current = Some(rows);
        self.next_block = 0;
        self.phase = Phase::Emitting;
    }

    /// The next chunk of the rows of the last term to emit, if any.
    fn next_chunk(&mut self) -> Option<DataChunk> {
        let batch = self.current.as_ref()?.block(self.next_block)?;
        self.next_block += 1;
        Some(DataChunk::from_record_batch(batch))
    }

    /// The recursion ended: release its rows.
    fn finish(&mut self) {
        self.phase = Phase::Done;
        self.current = None;
        self.iterations = None;
    }
}

#[derive(Debug, Default)]
enum Phase {
    /// The rows of the static term are not emitted yet.
    #[default]
    Start,
    /// Emitting the rows of the last term (`Recursion::current`).
    Emitting,
    /// An iteration runs: its outcome, once it completed.
    Iterating(Arc<Mutex<Option<Result<()>>>>),
    /// The recursion ended.
    Done,
}

/// What the iterations of a [`PhysicalRecursiveCte`] run with.
#[derive(Debug)]
struct Iterations {
    /// The task context of the executions of the iterations, with
    /// `work_tables` in its session config.
    task_ctx: Arc<TaskContext>,
    /// The working table of the CTE.
    work_tables: Arc<WorkTables>,
    /// The number of iterations started.
    count: usize,
}

impl Iterations {
    /// The task context for the iterations of the CTE `name` in an execution
    /// with the task context `task_ctx`: `task_ctx`, whose session config
    /// also holds the working table of the CTE, in front of those of the
    /// recursive queries around it (in the session config of `task_ctx`).
    fn new(name: &str, task_ctx: &TaskContext) -> Self {
        let config = task_ctx.session_config();
        let work_tables = Arc::new(WorkTables {
            name: name.to_owned(),
            table: Mutex::new(None),
            outer: config.get_extension::<WorkTables>(),
        });
        let config = config.clone().with_extension(Arc::clone(&work_tables));
        let task_ctx = TaskContext::new(
            task_ctx.task_id(),
            task_ctx.session_id(),
            config,
            task_ctx.scalar_functions().clone(),
            task_ctx.higher_order_functions().clone(),
            task_ctx.aggregate_functions().clone(),
            task_ctx.window_functions().clone(),
            task_ctx.runtime_env(),
        );
        Self {
            task_ctx: Arc::new(task_ctx),
            work_tables,
            count: 0,
        }
    }
}

/// The working tables that the scans of the execution of an iteration read:
/// an extension of the session config of its task context. The working table
/// of a CTE, linked to the working tables of the recursive queries around it.
#[derive(Debug)]
struct WorkTables {
    /// The name of the CTE.
    name: String,
    /// Its working table, while an iteration runs.
    table: Mutex<Option<Arc<WorkingTable>>>,
    /// The working tables of the recursive queries that run the CTE.
    outer: Option<Arc<WorkTables>>,
}

impl WorkTables {
    /// The working table of the innermost recursive query named `name`, if
    /// one of them runs and has a working table.
    fn table(&self, name: &str) -> Option<Arc<WorkingTable>> {
        let mut tables = Some(self);
        while let Some(current) = tables {
            if current.name == name {
                return current.table.lock().clone();
            }
            tables = current.outer.as_deref();
        }
        None
    }
}

/// The rows of a term of a recursion (of the static term or of an
/// iteration), with the schema of the query: emitted by the CTE, and the
/// working table of the next iteration. Its memory is released with it.
#[derive(Debug)]
struct WorkingTable {
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
    /// Blocks of at most [`DEFAULT_VECTOR_SIZE`] rows of `batches`, in order:
    /// `(batch, offset, rows)`.
    blocks: Vec<(usize, usize, usize)>,
    rows: usize,
    /// The memory of the rows.
    _reservation: MemoryReservation,
}

impl WorkingTable {
    fn new(
        schema: SchemaRef,
        batches: Vec<RecordBatch>,
        reservation: MemoryReservation,
    ) -> Self {
        let mut blocks = vec![];
        let mut rows = 0;
        for (index, batch) in batches.iter().enumerate() {
            let num_rows = batch.num_rows();
            rows += num_rows;
            blocks.extend((0..num_rows).step_by(DEFAULT_VECTOR_SIZE).map(|offset| {
                (index, offset, (num_rows - offset).min(DEFAULT_VECTOR_SIZE))
            }));
        }
        Self {
            schema,
            batches,
            blocks,
            rows,
            _reservation: reservation,
        }
    }

    /// The rows of block `index`, if any.
    fn block(&self, index: usize) -> Option<RecordBatch> {
        let &(batch, offset, rows) = self.blocks.get(index)?;
        let batch = &self.batches[batch];
        Some(if rows == batch.num_rows() {
            batch.clone()
        } else {
            batch.slice(offset, rows)
        })
    }
}

/// Collects the rows of the terms of a recursion: the global state of the
/// sinks of their pipelines, shared by all threads (the [`PhysicalRecursiveCte`]
/// for the static term, and the `RecursiveTermSink` of every iteration).
///
/// Every thread collects the rows it sinks (relabeled with the schema of the
/// query), after it removed the rows that the recursion collected before
/// (for `UNION`), and reserves their memory. The rows of the threads that
/// completed are combined, until the CTE takes all of them, once the term is
/// complete.
struct Collector {
    schema: SchemaRef,
    /// The rows of the threads that completed.
    batches: Mutex<Vec<RecordBatch>>,
    /// The memory of the rows collected since they were last taken.
    reservation: MemoryReservation,
    /// For `UNION`: all rows that the recursion collected.
    distinct: Option<Mutex<DistinctRows>>,
}

impl fmt::Debug for Collector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Collector")
            .field("batches", &self.batches.lock().len())
            .field("reserved", &self.reservation.size())
            .field("distinct", &self.distinct.is_some())
            .finish()
    }
}

/// The rows a thread collected.
#[derive(Debug, Default)]
struct CollectedRows {
    batches: Vec<RecordBatch>,
}

impl LocalSinkState for CollectedRows {}

impl Collector {
    /// The collector of the CTE `name` (with schema `schema`), whose memory
    /// is reserved in `pool`: that of the rows by the consumer
    /// `PhysicalRecursiveCte[name]`, that of the hash table of a distinct
    /// recursion by `PhysicalRecursiveCte[name] hash table`.
    fn try_new(
        name: &str,
        schema: SchemaRef,
        is_distinct: bool,
        pool: &Arc<dyn MemoryPool>,
    ) -> Result<Self> {
        let consumer = format!("PhysicalRecursiveCte[{name}]");
        let distinct = if is_distinct {
            Some(Mutex::new(DistinctRows {
                group_values: new_group_values(
                    Arc::clone(&schema),
                    &GroupClustering::None,
                )?,
                groups: vec![],
                reservation: MemoryConsumer::new(format!("{consumer} hash table"))
                    .register(pool),
            }))
        } else {
            None
        };
        Ok(Self {
            schema,
            batches: Mutex::new(vec![]),
            reservation: MemoryConsumer::new(consumer).register(pool),
            distinct,
        })
    }

    /// Collect the rows of `input` into `local`: all of them, or for `UNION`
    /// those that the recursion did not collect before.
    fn collect(&self, input: &DataChunk, local: &mut CollectedRows) -> Result<()> {
        let mut batch = input.compact()?;
        if batch.schema_ref() != &self.schema {
            let options =
                RecordBatchOptions::new().with_row_count(Some(batch.num_rows()));
            batch = RecordBatch::try_new_with_options(
                Arc::clone(&self.schema),
                batch.columns().to_vec(),
                &options,
            )?;
        }
        if let Some(distinct) = &self.distinct {
            batch = distinct.lock().new_rows(batch)?;
        }
        if batch.num_rows() == 0 {
            return Ok(());
        }
        self.reservation.try_grow(memory_size(&batch))?;
        local.batches.push(batch);
        Ok(())
    }

    /// The thread of `local` completed: combine its rows.
    fn combine(&self, local: &mut CollectedRows) {
        self.batches.lock().append(&mut local.batches);
    }

    /// Whether no rows were collected since they were last taken.
    fn is_empty(&self) -> bool {
        self.batches.lock().is_empty()
    }

    /// Take the rows collected since they were last taken, once all threads
    /// combined them, with their memory.
    fn take(&self) -> Arc<WorkingTable> {
        let batches = std::mem::take(&mut *self.batches.lock());
        let reservation = self.reservation.split(self.reservation.size());
        Arc::new(WorkingTable::new(
            Arc::clone(&self.schema),
            batches,
            reservation,
        ))
    }
}

/// The rows that a distinct recursion (`UNION`) collected: a hash table of
/// their values (DataFusion's [`GroupValues`], as in DataFusion's
/// `RecursiveQueryExec`).
struct DistinctRows {
    group_values: Box<dyn GroupValues>,
    /// The group of every row of the batch being deduplicated.
    groups: Vec<usize>,
    /// The memory of the table.
    reservation: MemoryReservation,
}

impl DistinctRows {
    /// The rows of `batch` that are not in the table (once each), which are
    /// added to it.
    fn new_rows(&mut self, batch: RecordBatch) -> Result<RecordBatch> {
        let seen = self.group_values.len();
        self.groups.clear();
        self.group_values
            .intern(batch.columns(), &mut self.groups)?;
        self.reservation.try_resize(
            self.group_values.size() + self.groups.capacity() * size_of::<usize>(),
        )?;
        // The groups of new rows get increasing indices, from `seen` on: a
        // row is new if its group is new and it is the first row of it.
        let mut next = seen;
        let mut new_rows = 0;
        let mask = self
            .groups
            .iter()
            .map(|&group| {
                let new = group >= next;
                if new {
                    next = group + 1;
                    new_rows += 1;
                }
                Some(new)
            })
            .collect::<BooleanArray>();
        if new_rows == batch.num_rows() {
            return Ok(batch);
        }
        Ok(filter_record_batch(&batch, &mask)?)
    }
}

/// The sink of the execution of an iteration of a [`PhysicalRecursiveCte`]:
/// collects the rows of the recursive term (its only child) for the CTE.
#[derive(Debug)]
struct RecursiveTermSink {
    children: Vec<Arc<dyn PhysicalOperator>>,
    collector: Arc<Collector>,
    /// The metrics of the CTE in the execution of the query: the rows of the
    /// iterations are rows that it sinks.
    metrics: Arc<OperatorMetrics>,
}

impl RecursiveTermSink {
    fn try_new(
        term: Arc<dyn PhysicalOperator>,
        collector: Arc<Collector>,
        metrics: Arc<OperatorMetrics>,
    ) -> Result<Self> {
        if !same_data_types(&term.schema(), &collector.schema) {
            return internal_err!(
                "The recursive term of a recursive query has another schema than the query: {:?} vs {:?}",
                term.schema(),
                collector.schema
            );
        }
        Ok(Self {
            children: vec![term],
            collector,
            metrics,
        })
    }
}

impl PhysicalOperator for RecursiveTermSink {
    fn name(&self) -> &str {
        "RecursiveTermSink"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.collector.schema)
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
        Ok(Box::new(CollectedRows::default()))
    }

    fn sink(
        &self,
        _ctx: &ExecutionContext,
        input: &DataChunk,
        _gstate: &dyn GlobalSinkState,
        lstate: &mut dyn LocalSinkState,
    ) -> Result<SinkResult> {
        let local =
            downcast_state_mut::<CollectedRows>(lstate.as_any_mut(), self.name())?;
        let start = Instant::now();
        let result = self.collector.collect(input, local);
        self.metrics.add_elapsed(start.elapsed());
        result?;
        self.metrics.record_sink(input.len());
        Ok(SinkResult::NeedMoreInput)
    }

    fn combine(
        &self,
        _ctx: &ExecutionContext,
        _gstate: &dyn GlobalSinkState,
        lstate: &mut dyn LocalSinkState,
    ) -> Result<()> {
        let local =
            downcast_state_mut::<CollectedRows>(lstate.as_any_mut(), self.name())?;
        self.collector.combine(local);
        Ok(())
    }
}

/// A scan of the working table of a recursive query, DataFusion's
/// `WorkTableExec`: a parallel source of the rows of the last term of the
/// innermost [`PhysicalRecursiveCte`] of its name that runs it, see the
/// [module documentation](self#the-working-table).
///
/// It reads the columns of the working table that have the names of its
/// fields, without copying them, with its schema (whose fields DataFusion
/// declares nullable).
#[derive(Debug)]
pub struct WorkTableScanSource {
    /// The name of the CTE.
    name: String,
    schema: SchemaRef,
}

impl WorkTableScanSource {
    /// A scan of the working table of the recursive query `name`, of the
    /// columns of the fields of `schema`.
    pub fn new(name: impl Into<String>, schema: SchemaRef) -> Self {
        Self {
            name: name.into(),
            schema,
        }
    }

    /// Lower a DataFusion `WorkTableExec` to a [`WorkTableScanSource`] with
    /// the same schema. Returns `Ok(None)` for any other node.
    ///
    /// It scans the working table of the [`PhysicalRecursiveCte`] of its name
    /// that runs it, so that it only runs in the recursive term of one (see
    /// [`PhysicalRecursiveCte::try_lower`]), as a `WorkTableExec` only runs in
    /// the recursive term of a `RecursiveQueryExec`.
    pub fn try_lower(plan: &Arc<dyn ExecutionPlan>) -> Result<Option<Self>> {
        let Some(scan) = plan.downcast_ref::<WorkTableExec>() else {
            return Ok(None);
        };
        Ok(Some(Self::new(scan.name(), plan.schema())))
    }

    /// The name of the recursive query whose working table it scans.
    pub fn name(&self) -> &str {
        &self.name
    }
}

/// The global source state of a [`WorkTableScanSource`]: the working table,
/// and the next block of it to scan.
#[derive(Debug)]
struct WorkTableScanState {
    table: Arc<WorkingTable>,
    /// The columns of the table that the scan reads.
    projection: Vec<usize>,
    next_block: AtomicUsize,
}

impl GlobalSourceState for WorkTableScanState {
    fn max_threads(&self) -> usize {
        self.table.blocks.len().max(1)
    }
}

impl PhysicalOperator for WorkTableScanSource {
    fn name(&self) -> &str {
        "WorkTableScanSource"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &[]
    }

    fn description(&self) -> String {
        format!("WorkTableScanSource: name={}", self.name)
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
        let Some(table) = ctx
            .task_ctx
            .session_config()
            .get_extension::<WorkTables>()
            .and_then(|tables| tables.table(&self.name))
        else {
            return internal_err!(
                "{} runs outside of the recursive term of its recursive query",
                self.description()
            );
        };
        let Some(projection) = work_table_projection(&self.schema, &table.schema) else {
            return internal_err!(
                "The columns of {} are not columns of its recursive query: {:?} vs {:?}",
                self.description(),
                self.schema,
                table.schema
            );
        };
        Ok(Arc::new(WorkTableScanState {
            table,
            projection,
            next_block: AtomicUsize::new(0),
        }))
    }

    fn get_data(
        &self,
        _ctx: &ExecutionContext,
        gstate: &dyn GlobalSourceState,
        _lstate: &mut dyn LocalSourceState,
    ) -> Result<SourceResult> {
        let state = downcast_state::<WorkTableScanState>(gstate.as_any(), self.name())?;
        let block = state.next_block.fetch_add(1, Ordering::Relaxed);
        let Some(batch) = state.table.block(block) else {
            return Ok(SourceResult::Finished);
        };
        let chunk = if self.schema.fields().is_empty() {
            DataChunk::try_new_without_columns(
                Arc::clone(&self.schema),
                batch.num_rows(),
            )?
        } else {
            let columns = state
                .projection
                .iter()
                .map(|&column| Arc::clone(batch.column(column)))
                .collect();
            DataChunk::try_new(Arc::clone(&self.schema), columns, None)?
        };
        Ok(SourceResult::Chunk(chunk))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashMap;
    use std::time::Duration;

    use arrow::array::{AsArray, Int64Array};
    use arrow::datatypes::{DataType, Field, Int64Type};
    use datafusion::datasource::MemTable;
    use datafusion::execution::SessionStateBuilder;
    use datafusion::execution::runtime_env::RuntimeEnvBuilder;
    use datafusion::physical_planner::DefaultPhysicalPlanner;
    use datafusion::prelude::{SessionConfig, SessionContext};
    use datafusion_common::DataFusionError;
    use datafusion_datasource::memory::MemorySourceConfig;
    use datafusion_execution::memory_pool::GreedyMemoryPool;
    use datafusion_physical_plan::collect;
    use datafusion_physical_plan::placeholder_row::PlaceholderRowExec;
    use futures::StreamExt;

    use crate::PipelineQueryPlanner;
    use crate::bridge::{BridgeOperator, PipelineRootExec};
    use crate::explain::{format_operator_tree, format_pipelines};
    use crate::operators::union::test_support::count;
    use crate::planner::lower_execution_plan;
    use crate::test_utils::{
        EngineOutput, compare_outputs, datafusion_context, pipeline_context, run_engines,
        run_sql, test_session_config,
    };

    // ---------------------------------------------------------------------
    // Helpers
    // ---------------------------------------------------------------------

    /// Await `future`, failing the test if it does not complete in time: a
    /// recursion that does not end hangs.
    async fn with_timeout<T>(future: impl Future<Output = T>) -> T {
        tokio::time::timeout(Duration::from_secs(60), future)
            .await
            .expect("the query did not complete")
    }

    /// Whether the DataFusion plan `plan` has a recursive query or a scan of
    /// a working table.
    fn has_recursion(plan: &Arc<dyn ExecutionPlan>) -> bool {
        plan.is::<RecursiveQueryExec>()
            || plan.is::<WorkTableExec>()
            || plan.children().into_iter().any(has_recursion)
    }

    /// The operator tree of the plan of the pipeline engine.
    fn lowered(output: &EngineOutput) -> Arc<dyn PhysicalOperator> {
        let root = output
            .plan
            .downcast_ref::<PipelineRootExec>()
            .expect("a plan of the pipeline engine");
        Arc::clone(root.operator())
    }

    /// Assert that the operator tree `root` runs its recursive queries
    /// natively: they are [`PhysicalRecursiveCte`]s, and DataFusion executes
    /// neither a recursive query nor a scan of a working table.
    fn assert_recursion_is_native(root: &Arc<dyn PhysicalOperator>, sql: &str) {
        assert!(
            count::<PhysicalRecursiveCte>(root) > 0,
            "no PhysicalRecursiveCte\nquery: {sql}\nlowered: {}",
            format_operator_tree(root)
        );
        assert!(
            bridged_plans(root).iter().all(|plan| !has_recursion(plan)),
            "DataFusion executes a recursion\nquery: {sql}\nlowered: {}",
            format_operator_tree(root)
        );
    }

    /// Assert that `sql` returns the same results on DataFusion and on the
    /// pipeline engine, both configured with `config` and with the tables of
    /// `setup`, and that the pipeline engine runs its recursive queries
    /// natively (see [`assert_recursion_is_native`]). Returns the operator
    /// tree of the pipeline engine.
    ///
    /// The results are compared in canonical form (see
    /// [`compare_outputs`]): the results of the queries are unambiguous.
    async fn assert_native_recursion<F>(
        sql: &str,
        config: SessionConfig,
        setup: F,
    ) -> Arc<dyn PhysicalOperator>
    where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        let (expected, actual) = with_timeout(run_engines(sql, config, setup))
            .await
            .unwrap_or_else(|e| {
                panic!("Failed to run query on both engines: {e}\nquery: {sql}")
            });
        let root = lowered(&actual);
        if let Err(message) = compare_outputs(&expected, &actual) {
            panic!(
                "Results differ: {message}\nquery: {sql}\nlowered: {}",
                format_operator_tree(&root)
            );
        }
        assert_recursion_is_native(&root, sql);
        root
    }

    /// The directory of the data files of the recursive queries of
    /// DataFusion's `cte.slt`.
    fn data_dir() -> String {
        format!(
            "{}/../core/tests/data/recursive_cte",
            env!("CARGO_MANIFEST_DIR")
        )
    }

    async fn execute(ctx: &SessionContext, sql: &str) -> Result<()> {
        ctx.sql(sql).await?.collect().await?;
        Ok(())
    }

    /// Write the Parquet files of the recursive queries of `cte.slt` into
    /// `dir`.
    async fn write_parquet_files(dir: &str) -> Result<()> {
        let ctx = datafusion_context(test_session_config());
        for statement in [
            format!(
                "COPY (SELECT i AS k, i AS v1, i AS v2 FROM generate_series(1, 3) t(i)) \
                 TO '{dir}/test.parquet'"
            ),
            format!(
                "COPY (SELECT CAST(value AS DOUBLE) AS r FROM generate_series(0, 128)) \
                 TO '{dir}/dynamic_filter_recursive.parquet' STORED AS PARQUET"
            ),
            format!(
                "COPY (SELECT value AS k FROM generate_series(1, 100)) \
                 TO '{dir}/hash_join_dynamic_filter.parquet' STORED AS PARQUET"
            ),
        ] {
            execute(&ctx, &statement).await?;
        }
        Ok(())
    }

    /// Register the tables of the recursive queries of `cte.slt` in `ctx`,
    /// with the Parquet files of [`write_parquet_files`] in `dir`.
    async fn register_cte_tables(ctx: &SessionContext, dir: &str) -> Result<()> {
        let data = data_dir();
        for table in [
            "balance",
            "growth",
            "prices",
            "sales",
            "salespersons",
            "closure",
        ] {
            execute(
                ctx,
                &format!(
                    "CREATE EXTERNAL TABLE {table} STORED AS CSV \
                     LOCATION '{data}/{table}.csv' OPTIONS ('format.has_header' 'true')"
                ),
            )
            .await?;
        }
        for table in [
            "test",
            "dynamic_filter_recursive",
            "hash_join_dynamic_filter",
        ] {
            execute(
                ctx,
                &format!(
                    "CREATE EXTERNAL TABLE {table} STORED AS PARQUET \
                     LOCATION '{dir}/{table}.parquet'"
                ),
            )
            .await?;
        }
        for statement in [
            "CREATE TABLE t(a BIGINT) AS VALUES (1), (2), (3)",
            "CREATE TABLE cte_schema_records (id VARCHAR NOT NULL, parent_id VARCHAR, \
             ts TIMESTAMP NOT NULL, val VARCHAR)",
            "INSERT INTO cte_schema_records VALUES \
             ('a00', NULL, TIMESTAMP '2025-01-01 00:00:00', 'v_span'), \
             ('a01', 'a00', TIMESTAMP '2025-01-01 00:00:01', 'v_log_1'), \
             ('a02', 'a01', TIMESTAMP '2025-01-01 00:00:02', 'v_log_2'), \
             ('a03', 'a02', TIMESTAMP '2025-01-01 00:00:03', 'v_log_3'), \
             ('a04', 'a03', TIMESTAMP '2025-01-01 00:00:04', 'v_log_4'), \
             ('a05', 'a04', TIMESTAMP '2025-01-01 00:00:05', 'v_log_5')",
            "CREATE TABLE recursive_cte_pk(id INT NOT NULL PRIMARY KEY)",
            "INSERT INTO recursive_cte_pk VALUES (2), (1)",
        ] {
            execute(ctx, statement).await?;
        }
        Ok(())
    }

    /// The recursive query of `cte.slt` with a static term derived from a
    /// table, which it runs with a batch size of 2 too.
    const BALANCES: &str = "WITH RECURSIVE balances AS ( \
            SELECT * from balance \
            UNION ALL \
            SELECT time + 1 as time, name, account_balance + 10 as account_balance \
            FROM balances WHERE time < 10) \
        SELECT * FROM balances ORDER BY time, name, account_balance";

    /// The recursive queries of DataFusion's `cte.slt` (in its order), but
    /// those whose planning fails, over the tables of
    /// [`register_cte_tables`].
    const CTE_SLT_QUERIES: &[&str] = &[
        // trivial recursive CTE works
        "WITH RECURSIVE nodes AS (SELECT 1 as id UNION ALL SELECT id + 1 as id FROM nodes \
         WHERE id < 10) SELECT * FROM nodes",
        // recursive CTE with a column-list alias
        "WITH RECURSIVE t(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM t WHERE n < 10) \
         SELECT n FROM t",
        // recursive CTE with a multi-column column-list alias
        "WITH RECURSIVE t(a, b) AS (SELECT 1, 2 UNION ALL SELECT a + 1, b * 2 FROM t \
         WHERE a < 5) SELECT a, b FROM t",
        // recursive CTE with a column-list alias and UNION (DISTINCT)
        "WITH RECURSIVE t(n) AS (SELECT 1 UNION SELECT n + 1 FROM t WHERE n < 5) \
         SELECT n FROM t",
        // recursive CTE with a quoted, case-sensitive column-list alias
        "WITH RECURSIVE t(\"N\") AS (SELECT 1 UNION ALL SELECT \"N\" + 1 FROM t \
         WHERE \"N\" < 5) SELECT \"N\" FROM t",
        // simple deduplicating recursive CTE works
        "WITH RECURSIVE nodes AS (SELECT id from (VALUES (1), (2)) nodes(id) UNION \
         SELECT id + 1 as id FROM nodes WHERE id < 4) SELECT * FROM nodes",
        // deduplicating recursive CTE with two variables works
        "WITH RECURSIVE ranges AS (SELECT min, max from (VALUES (1, 1), (2, 2)) \
         ranges(min, max) UNION SELECT min, max + 1 as max FROM ranges WHERE max < 4) \
         SELECT * FROM ranges",
        // recursive CTE with static term derived from table works
        BALANCES,
        // recursive CTE with recursive join works
        "WITH RECURSIVE balances AS (SELECT time as time, name as name, \
         account_balance as account_balance FROM balance UNION ALL \
         SELECT time + 1 as time, balances.name, \
         account_balance + growth.account_growth as account_balance \
         FROM balances JOIN growth ON balances.name = growth.name WHERE time < 10) \
         SELECT * FROM balances ORDER BY time, name, account_balance",
        // recursive CTE with aggregations works
        "WITH RECURSIVE nodes AS (SELECT 1 as id UNION ALL SELECT id + 1 as id FROM nodes \
         WHERE id < 10) SELECT sum(id) FROM nodes",
        // referencing CTE multiple times does not error
        "WITH RECURSIVE my_cte AS (SELECT a from t UNION ALL SELECT a+2 as a FROM my_cte \
         WHERE a<5) SELECT * FROM my_cte t1, my_cte",
        // CTE within recursive CTE works
        "WITH RECURSIVE \"recursive_cte\" AS (SELECT 1 as \"val\" UNION ALL ( \
           WITH \"sub_cte\" AS (SELECT time, 1 as \"val\" FROM \
             (SELECT DISTINCT \"time\" FROM \"balance\")) \
           SELECT 2 as \"val\" FROM \"recursive_cte\" FULL JOIN \"sub_cte\" ON 1 = 1 \
           WHERE \"recursive_cte\".\"val\" < 2)) \
         SELECT * FROM \"recursive_cte\"",
        // CTE within window function inside nested CTE works
        "WITH RECURSIVE \"recursive_cte\" AS (( \
           WITH \"min_prices_row_num_cte\" AS (SELECT MIN(\"prices\".\"prices_row_num\") \
             AS \"prices_row_num\" FROM \"prices\"), \
           \"min_prices_row_num_cte_second\" AS (SELECT \
             MIN(\"prices\".\"prices_row_num\") AS \"prices_row_num_advancement\" \
             FROM \"prices\" WHERE \"prices\".\"prices_row_num\" > \
             (SELECT \"prices_row_num\" FROM \"min_prices_row_num_cte\")) \
           SELECT 0.0 AS \"beg\", (0.0 + 50) AS \"end\", \
             (SELECT \"prices_row_num\" FROM \"min_prices_row_num_cte\") \
               AS \"prices_row_num\", \
             (SELECT \"prices_row_num_advancement\" FROM \
               \"min_prices_row_num_cte_second\") AS \"prices_row_num_advancement\" \
           FROM \"prices\" WHERE \"prices\".\"prices_row_num\" = \
             (SELECT DISTINCT \"prices_row_num\" FROM \"min_prices_row_num_cte\") \
         ) UNION ALL ( \
           WITH \"min_prices_row_num_cte\" AS (SELECT \
             \"prices\".\"prices_row_num\" AS \"prices_row_num\", \
             LEAD(\"prices\".\"prices_row_num\", 1) OVER (ORDER BY \"prices_row_num\") \
               AS \"prices_row_num_advancement\" \
             FROM (SELECT DISTINCT \"prices_row_num\" FROM \"prices\") AS \"prices\") \
           SELECT \"recursive_cte\".\"end\" AS \"beg\", \
             (\"recursive_cte\".\"end\" + 50) AS \"end\", \
             \"min_prices_row_num_cte\".\"prices_row_num\" AS \"prices_row_num\", \
             \"min_prices_row_num_cte\".\"prices_row_num_advancement\" \
               AS \"prices_row_num_advancement\" \
           FROM \"recursive_cte\" \
           FULL JOIN \"prices\" ON \"prices\".\"prices_row_num\" = \
             \"recursive_cte\".\"prices_row_num_advancement\" \
           FULL JOIN \"min_prices_row_num_cte\" ON \
             \"min_prices_row_num_cte\".\"prices_row_num\" = COALESCE( \
               \"prices\".\"prices_row_num\", \
               \"recursive_cte\".\"prices_row_num_advancement\") \
           WHERE \"recursive_cte\".\"prices_row_num_advancement\" IS NOT NULL)) \
         SELECT DISTINCT * FROM \"recursive_cte\" ORDER BY \"prices_row_num\" ASC",
        // group by works within recursive cte
        "WITH RECURSIVE region_sales AS ( \
           SELECT s.salesperson_id AS salesperson_id, SUM(s.sale_amount) AS amount, \
             0 as level FROM sales s GROUP BY s.salesperson_id \
           UNION ALL \
           SELECT sp.manager_id AS salesperson_id, SUM(rs.amount) AS amount, \
             MIN(rs.level) + 1 as level FROM region_sales rs \
           INNER JOIN salespersons sp ON rs.salesperson_id = sp.salesperson_id \
           WHERE sp.manager_id IS NOT NULL GROUP BY sp.manager_id) \
         SELECT salesperson_id, MAX(amount) as amount, MAX(level) as hierarchy_level \
         FROM region_sales GROUP BY salesperson_id \
         ORDER BY hierarchy_level ASC, salesperson_id ASC",
        // nested recursive ctes
        "WITH RECURSIVE outer_cte AS (SELECT 1 as a UNION ALL ( \
           WITH RECURSIVE nested_cte AS (SELECT 1 as a UNION ALL \
             SELECT a + 2 as a FROM nested_cte where a < 3) \
           SELECT outer_cte.a + 2 as a FROM outer_cte JOIN nested_cte USING(a) \
           WHERE nested_cte.a < 4)) \
         SELECT a FROM outer_cte",
        // https://github.com/apache/datafusion/issues/9680
        "WITH RECURSIVE recursive_cte AS (SELECT 1 as val UNION ALL ( \
           WITH sub_cte AS (SELECT 2 as val) \
           SELECT 2 as val FROM recursive_cte CROSS JOIN sub_cte \
           WHERE recursive_cte.val < 2)) \
         SELECT * FROM recursive_cte",
        // https://github.com/apache/datafusion/issues/9794: the terms have
        // different types
        "WITH RECURSIVE my_cte AS(SELECT 1::int AS a UNION ALL \
         SELECT a::bigint+2 FROM my_cte WHERE a<3) SELECT *, arrow_typeof(a) FROM my_cte",
        // casting the seed to the intended result type preserves fractional
        // values
        "WITH RECURSIVE my_cte AS (SELECT 1::DOUBLE AS a UNION ALL \
         SELECT a + 0.5::DOUBLE FROM my_cte WHERE a < 2) \
         SELECT a, arrow_typeof(a) FROM my_cte ORDER BY a",
        // safe decimal widening
        "WITH RECURSIVE my_cte AS (SELECT 1::DECIMAL(20, 5) AS a UNION ALL \
         SELECT 2::DECIMAL(10, 2) FROM my_cte WHERE a < 2) SELECT count(*) FROM my_cte",
        // every signed BIGINT value fits in DECIMAL(19, 0)
        "WITH RECURSIVE my_cte AS (SELECT 0::DECIMAL(19, 0) AS lo, \
         0::DECIMAL(19, 0) AS hi UNION ALL \
         SELECT '-9223372036854775808'::BIGINT, '9223372036854775807'::BIGINT \
         FROM my_cte WHERE lo = 0) SELECT lo::VARCHAR, hi::VARCHAR FROM my_cte",
        // unsigned BIGINT requires all 20 integer digits
        "WITH RECURSIVE my_cte AS (SELECT 0::DECIMAL(20, 0) AS a UNION ALL \
         SELECT arrow_cast('18446744073709551615', 'UInt64') FROM my_cte WHERE a = 0) \
         SELECT a::VARCHAR FROM my_cte",
        // safe widening within date and string types
        "WITH RECURSIVE my_cte AS (SELECT arrow_cast(DATE '1970-01-01', 'Date64') AS a \
         UNION ALL SELECT DATE '1970-01-02' FROM my_cte \
         WHERE a = arrow_cast(DATE '1970-01-01', 'Date64')) SELECT count(*) FROM my_cte",
        "WITH RECURSIVE my_cte AS (SELECT arrow_cast('a', 'LargeUtf8') AS a UNION ALL \
         SELECT 'b' FROM my_cte WHERE a = 'a') SELECT count(*) FROM my_cte",
        "WITH RECURSIVE my_cte AS (SELECT arrow_cast('a', 'Utf8View') AS a UNION ALL \
         SELECT arrow_cast('b', 'LargeUtf8') FROM my_cte WHERE a = 'a') \
         SELECT count(*) FROM my_cte",
        "WITH RECURSIVE my_cte AS (SELECT arrow_cast(X'616263', 'Binary') AS a UNION ALL \
         SELECT arrow_cast(X'646566', 'FixedSizeBinary(3)') FROM my_cte \
         WHERE a = X'616263') SELECT count(*) FROM my_cte",
        // a recursive CTE and a non-recursive CTE at the same time
        "WITH RECURSIVE non_recursive_cte AS (SELECT 1), recursive_cte AS ( \
         SELECT 1 AS a UNION ALL SELECT a+2 FROM recursive_cte WHERE a < 3) \
         SELECT * FROM non_recursive_cte, recursive_cte",
        // https://github.com/apache/datafusion/issues/16998: a TopK in the
        // recursive term, and a recursion that only a limit ends
        "with recursive r as (select 0 as k, 0 as v union all \
         (select * from r order by v limit 1)) select * from r limit 5",
        // transitive closure with loop
        "WITH RECURSIVE trans AS (SELECT * FROM closure UNION \
         SELECT l.start, r.end FROM trans as l, closure AS r WHERE l.end = r.start) \
         SELECT * FROM trans ORDER BY start, end",
        // column-name stability under ORDER BY + LIMIT: the schemas of the
        // batches are compared too
        "WITH RECURSIVE descendants AS ( \
           SELECT id, parent_id, ts, upper(val) AS val FROM cte_schema_records \
           WHERE id = 'a00' \
           UNION ALL \
           SELECT r.id, r.parent_id, r.ts, upper(r.val) FROM cte_schema_records r \
           INNER JOIN descendants d ON r.parent_id = d.id) \
         SELECT id, parent_id, ts, val FROM descendants ORDER BY ts ASC LIMIT 10",
        // recursive CTE nullability widening
        "WITH RECURSIVE t(a, b) AS (SELECT 0 AS a, 0 AS b UNION ALL \
         SELECT b AS a, CAST(NULL AS INT) AS b FROM t WHERE a IS NOT NULL) SELECT * FROM t",
        "WITH RECURSIVE t(a, b) AS (SELECT 0 AS a, 0 AS b UNION ALL \
         SELECT b AS a, CAST(NULL AS INT) AS b FROM t WHERE a IS NOT NULL) \
         SELECT * FROM t WHERE a IS NOT NULL",
        "WITH RECURSIVE t(a) AS (SELECT 1 AS a UNION ALL \
         SELECT CAST(NULL AS INT) AS a FROM t WHERE a IS NOT NULL) \
         SELECT * FROM t WHERE a IS NULL",
        "WITH RECURSIVE t(a) AS (SELECT 1 AS a UNION \
         SELECT CAST(NULL AS INT) AS a FROM t WHERE a IS NOT NULL) \
         SELECT * FROM t WHERE a IS NULL",
        "WITH RECURSIVE t(a, b) AS (SELECT 0 AS a, 0 AS b UNION ALL \
         SELECT b AS a, CAST(NULL AS INT) AS b FROM t WHERE a IS NOT NULL) \
         SELECT * FROM t WHERE b IS NULL",
        "WITH RECURSIVE t(a) AS (SELECT 1::INT AS a UNION ALL \
         SELECT CAST(NULL AS BIGINT) AS a FROM t WHERE a IS NOT NULL) \
         SELECT * FROM t WHERE a IS NULL",
        "WITH RECURSIVE t(a) AS (SELECT CAST(NULL AS INT) AS a UNION ALL \
         SELECT 1 AS a FROM t WHERE a IS NULL) SELECT * FROM t",
        "WITH RECURSIVE t(a) AS (SELECT 1 AS a UNION ALL \
         SELECT CAST(NULL AS INT) AS a FROM t WHERE a IS NOT NULL) \
         SELECT COUNT(*), COUNT(a), SUM(a) FROM t",
        "WITH RECURSIVE t(a) AS (SELECT 1 AS a UNION ALL \
         SELECT CAST(NULL AS INT) AS a FROM t WHERE a IS NOT NULL) \
         SELECT x FROM (SELECT a AS x FROM t) WHERE x IS NULL",
        "WITH RECURSIVE t(a, b) AS (SELECT 1::INT AS a, 2::INT AS b UNION ALL \
         SELECT a + 1 AS a, CAST(NULL AS BIGINT) AS b FROM t WHERE a < 2) \
         SELECT * FROM t WHERE b IS NULL",
        "WITH RECURSIVE t(a) AS (SELECT 1 AS a UNION ALL \
         SELECT CAST(NULL AS INT) AS a FROM t WHERE a IS NOT NULL) \
         SELECT t.a, u.b FROM t LEFT JOIN (SELECT 1 AS b) u ON t.a = u.b \
         WHERE u.b IS NULL",
        // a recursive CTE does not inherit the functional dependencies of
        // its static term
        "SELECT DISTINCT id FROM (WITH RECURSIVE t(id) AS ( \
           SELECT id FROM recursive_cte_pk UNION ALL SELECT id - 1 FROM t WHERE id > 1) \
         SELECT id FROM t)",
        // projection optimization in recursive CTEs
        // (https://github.com/apache/datafusion/issues/22249)
        "with recursive t(k, v) as (select 1 k, 10 v union all \
         select 2, 20 from t where k = 1) select v from t order by 1",
        "with recursive t(k, v) as (select 1 k, 10 v union all \
         select 2, 20 from t where v = 10) select v from t order by 1",
        // columns that the outer query does not select still make the rows
        // of a UNION distinct
        "with recursive t(k, v) as (select 1 k, 10 v union \
         select 2, 10 from t where v = 10) select v from t order by 1",
        // both terms are optimized, and the recursive term reads the result
        // of a scalar subquery
        "with recursive r as (select k, v1 from test union all \
         select k * 10, v1 from r where k < (select v2 from test where k = 2)) \
         select * from r order by 1, 2",
        // dynamic filters produced by a recursive term are reset between
        // iterations
        "WITH RECURSIVE w(n, m) AS (SELECT 0, CAST(NULL AS DOUBLE) UNION ALL \
         SELECT w.n + 1, x.m FROM w CROSS JOIN ( \
           SELECT MAX(r) AS m FROM dynamic_filter_recursive WHERE r % 2 = 0) x \
         WHERE w.n < 3) SELECT n, m FROM w",
        // hash join dynamic filters are shared with the probe scan across
        // iterations (#26641)
        "WITH RECURSIVE w(n) AS (SELECT 1 UNION ALL SELECT t.k FROM w \
         JOIN hash_join_dynamic_filter t ON t.k = w.n + 1 WHERE w.n < 5) \
         SELECT * FROM w ORDER BY n",
    ];

    // ---------------------------------------------------------------------
    // The recursive queries of `cte.slt`
    // ---------------------------------------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn recursive_queries_of_cte_slt_are_native() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let dir = dir.path().to_str().unwrap().to_owned();
        write_parquet_files(&dir).await?;
        let setup = async |ctx: &SessionContext| register_cte_tables(ctx, &dir).await;
        for target_partitions in [4, 1, 16] {
            let config = test_session_config().with_target_partitions(target_partitions);
            for sql in CTE_SLT_QUERIES {
                assert_native_recursion(sql, config.clone(), &setup).await;
            }
        }
        // With batches of 2 rows, every term has several chunks.
        let config = test_session_config().with_batch_size(2);
        assert_native_recursion(BALANCES, config, &setup).await;
        // The test of the dynamic filters of hash joins, as `cte.slt` runs it.
        let mut config = test_session_config().with_target_partitions(1);
        config.options_mut().execution.parquet.pushdown_filters = true;
        assert_native_recursion(CTE_SLT_QUERIES.last().unwrap(), config, &setup).await;
        Ok(())
    }

    // ---------------------------------------------------------------------
    // Lowering
    // ---------------------------------------------------------------------

    /// The first recursive query of `plan`, a DataFusion plan.
    fn recursive_query(plan: &Arc<dyn ExecutionPlan>) -> Option<Arc<dyn ExecutionPlan>> {
        if plan.is::<RecursiveQueryExec>() {
            return Some(Arc::clone(plan));
        }
        plan.children().into_iter().find_map(recursive_query)
    }

    /// The values of the first column, an `Int64` column, of `batches`,
    /// sorted.
    fn sorted_values(batches: &[RecordBatch]) -> Vec<i64> {
        let mut values = batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column(0)
                    .as_primitive::<Int64Type>()
                    .iter()
                    .map(|value| value.expect("a value"))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        values.sort_unstable();
        values
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn lowering_of_recursive_queries() -> Result<()> {
        let ctx = datafusion_context(test_session_config());
        let state = ctx.state();
        let lower = |plan: &Arc<dyn ExecutionPlan>| {
            lower_execution_plan_with_runtime(
                plan,
                state.config_options(),
                state.runtime_env(),
            )
        };
        for (union, is_distinct) in [("UNION ALL", false), ("UNION", true)] {
            let sql = format!(
                "WITH RECURSIVE nodes(id) AS (SELECT 1 {union} SELECT id + 1 FROM nodes \
                 WHERE id < 3) SELECT * FROM nodes"
            );
            let plan = ctx.sql(&sql).await?.create_physical_plan().await?;
            let query = recursive_query(&plan).expect("a recursive query");
            let cte = PhysicalRecursiveCte::try_lower(&query, lower)?.expect("native");
            assert_eq!(cte.name(), "nodes");
            assert_eq!(cte.is_distinct(), is_distinct);
            assert_eq!(cte.schema(), query.schema());
            assert_eq!(
                cte.description(),
                format!("PhysicalRecursiveCte: name=nodes, is_distinct={is_distinct}")
            );
            assert!(cte.is_sink() && cte.parallel_sink());
            assert!(cte.is_source() && !cte.parallel_source());
            assert_eq!(cte.children().len(), 2);
            // The recursive term, as lowered with the query, scans the
            // working table.
            let scans = count::<WorkTableScanSource>(cte.recursive_term());
            assert_eq!(scans, 1, "{}", format_operator_tree(cte.recursive_term()));
            assert_eq!(count::<WorkTableScanSource>(cte.static_term()), 0);
            // The static term sinks into the CTE, which is the source of the
            // pipeline of the query; the recursive term takes part in no
            // pipeline.
            let root: Arc<dyn PhysicalOperator> = Arc::new(cte);
            assert_eq!(
                format_pipelines(&root)?,
                "MetaPipeline 0 (sink: ResultSink) children: [1]\n  \
                 Pipeline 0: PhysicalRecursiveCte -> ResultSink depends on meta pipelines [1]\n\
                 MetaPipeline 1 (sink: PhysicalRecursiveCte)\n  \
                 Pipeline 1: PhysicalPlaceholderRow -> PhysicalProjection -> PhysicalRecursiveCte\n"
            );
            // It runs like DataFusion's.
            let output = with_timeout(Executor::collect(root, ctx.task_ctx(), 4)).await?;
            assert_eq!(sorted_values(&output), vec![1, 2, 3]);
        }

        // Any other node.
        let schema = Arc::new(Schema::new(vec![
            Field::new("n", DataType::Int64, true),
            Field::new("s", DataType::Utf8, true),
        ]));
        let placeholder: Arc<dyn ExecutionPlan> =
            Arc::new(PlaceholderRowExec::new(Arc::clone(&schema)));
        assert!(PhysicalRecursiveCte::try_lower(&placeholder, lower)?.is_none());
        assert!(WorkTableScanSource::try_lower(&placeholder)?.is_none());

        // A scan of a working table, of its columns that the `WorkTableExec`
        // projects.
        let scan: Arc<dyn ExecutionPlan> = Arc::new(WorkTableExec::new(
            "t".to_string(),
            Arc::clone(&schema),
            Some(vec![1]),
        )?);
        let source = WorkTableScanSource::try_lower(&scan)?.expect("a scan");
        assert_eq!(source.name(), "t");
        assert_eq!(source.schema(), scan.schema());
        assert_eq!(source.schema().field(0).name(), "s");
        assert_eq!(source.description(), "WorkTableScanSource: name=t");
        assert!(source.is_source() && source.parallel_source());
        assert!(source.children().is_empty());
        // It reads the working table of a recursive query that runs it: here
        // there is none.
        let error = with_timeout(Executor::collect(Arc::new(source), ctx.task_ctx(), 4))
            .await
            .unwrap_err();
        assert!(
            error.to_string().contains("outside of the recursive term"),
            "{error}"
        );
        Ok(())
    }

    /// A recursive query `t` over the rows of `static_term`, whose recursive
    /// term is a scan of its working table with the fields `scan_fields`:
    /// with `UNION`, the recursion ends after its first iteration.
    fn scan_recursion(
        static_term: Arc<dyn ExecutionPlan>,
        scan_fields: Vec<Field>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let schema = Arc::new(Schema::new(vec![Field::new("n", DataType::Int64, true)]));
        let scan = WorkTableExec::new(
            "t".to_string(),
            Arc::new(Schema::new(scan_fields)),
            None,
        )?;
        Ok(Arc::new(RecursiveQueryExec::try_new(
            "t".to_string(),
            schema,
            static_term,
            Arc::new(scan),
            true,
        )?))
    }

    /// The rows 1, 2 and 3 of an `n BIGINT` column, in `partitions`
    /// partitions (the first one has all of them).
    fn numbers_exec(partitions: usize) -> Result<Arc<dyn ExecutionPlan>> {
        let schema = Arc::new(Schema::new(vec![Field::new("n", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from(vec![1, 2, 3]))],
        )?;
        let mut data = vec![vec![batch]];
        data.resize(partitions, vec![]);
        Ok(MemorySourceConfig::try_new_exec(&data, schema, None)?)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn recursive_queries_whose_working_tables_datafusion_reads_are_left_to_it()
    -> Result<()> {
        let ctx = datafusion_context(test_session_config());
        let state = ctx.state();
        let lower = |plan: &Arc<dyn ExecutionPlan>| {
            lower_execution_plan_with_runtime(
                plan,
                state.config_options(),
                state.runtime_env(),
            )
        };
        let n = || Field::new("n", DataType::Int64, true);

        // A plan whose recursive term is a bare scan of its working table is
        // native, and runs like DataFusion's.
        let make = || scan_recursion(numbers_exec(1)?, vec![n()]);
        let root = lower(&make()?)?;
        assert!(
            root.is::<PhysicalRecursiveCte>(),
            "{}",
            format_operator_tree(&root)
        );
        let expected = collect(make()?, ctx.task_ctx()).await?;
        let actual = with_timeout(Executor::collect(root, ctx.task_ctx(), 4)).await?;
        assert_eq!(sorted_values(&expected), vec![1, 2, 3]);
        assert_eq!(sorted_values(&actual), vec![1, 2, 3]);

        // DataFusion executes the first partition of every term only: terms
        // of other numbers of partitions are left to DataFusion.
        let plan = scan_recursion(numbers_exec(2)?, vec![n()])?;
        assert!(PhysicalRecursiveCte::try_lower(&plan, lower)?.is_none());

        // A scan of the working table reads its columns by name: a scan of a
        // field that is not a column of the query is left to DataFusion
        // (which reads it by position).
        let make = || {
            scan_recursion(
                numbers_exec(1)?,
                vec![Field::new("m", DataType::Int64, true)],
            )
        };
        assert!(PhysicalRecursiveCte::try_lower(&make()?, lower)?.is_none());
        let root = lower(&make()?)?;
        assert!(
            root.is::<BridgeOperator>(),
            "{}",
            format_operator_tree(&root)
        );
        let output = with_timeout(Executor::collect(root, ctx.task_ctx(), 4)).await?;
        assert_eq!(sorted_values(&output), vec![1, 2, 3]);

        // Lowered for any session, cross joins are left to DataFusion (they
        // do not reserve their memory): so is a recursive query whose working
        // table is scanned below one, as DataFusion would read its working
        // table, which the native operator does not fill.
        let sql = "WITH RECURSIVE recursive_cte AS (SELECT 1 as val UNION ALL ( \
                     WITH sub_cte AS (SELECT 2 as val) \
                     SELECT 2 as val FROM recursive_cte CROSS JOIN sub_cte \
                     WHERE recursive_cte.val < 2)) \
                   SELECT * FROM recursive_cte";
        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        let root = lower_execution_plan(&plan)?;
        assert_eq!(count::<PhysicalRecursiveCte>(&root), 0);
        assert!(
            bridged_plans(&root)
                .iter()
                .any(|plan| plan.is::<RecursiveQueryExec>()),
            "{}",
            format_operator_tree(&root)
        );
        let output = with_timeout(Executor::collect(root, ctx.task_ctx(), 4)).await?;
        assert_eq!(sorted_values(&output), vec![1, 2]);
        // For a session without a memory limit, it is native.
        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        let root = lower(&plan)?;
        assert_recursion_is_native(&root, sql);
        Ok(())
    }

    // ---------------------------------------------------------------------
    // Execution
    // ---------------------------------------------------------------------

    /// The plan of type `plan_type` of the output of `EXPLAIN` `sql` in `ctx`.
    async fn explained(
        ctx: &SessionContext,
        sql: &str,
        plan_type: &str,
    ) -> Result<String> {
        let batches = ctx.sql(sql).await?.collect().await?;
        for batch in &batches {
            let types = batch.column(0).as_string::<i32>();
            let plans = batch.column(1).as_string::<i32>();
            for (t, plan) in types.iter().zip(plans.iter()) {
                if t == Some(plan_type) {
                    return Ok(plan.unwrap_or_default().to_string());
                }
            }
        }
        internal_err!("No {plan_type} in the output of {sql}")
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn explain_shows_the_recursion() -> Result<()> {
        let ctx = pipeline_context(test_session_config());
        let sql = "WITH RECURSIVE nodes(id) AS (SELECT 1 UNION ALL SELECT id + 1 FROM nodes \
                   WHERE id < 10) SELECT * FROM nodes";
        let plan = explained(&ctx, &format!("EXPLAIN {sql}"), "physical_plan").await?;
        for line in [
            "PipelineRootExec: pipelines=2\n",
            "\n  PhysicalRecursiveCte: name=nodes, is_distinct=false, pipelines=[0, 1]\n",
            "\n          WorkTableScanSource: name=nodes\n",
            "\n  Pipeline 1: PhysicalPlaceholderRow -> PhysicalProjection -> PhysicalRecursiveCte\n",
            "\n  Pipeline 0: PhysicalRecursiveCte -> CollectorSink, after pipelines [1]\n",
        ] {
            assert!(plan.contains(line), "{line}: {plan}");
        }
        assert!(!plan.contains("Bridge"), "{plan}");

        // The CTE emits the 10 rows of the query, and sinks the rows of all of
        // its terms: the row of the static term and those of the 9
        // iterations.
        let plan =
            explained(&ctx, &format!("EXPLAIN ANALYZE {sql}"), "Plan with Metrics")
                .await?;
        let line = plan
            .lines()
            .find(|line| line.trim_start().starts_with("PhysicalRecursiveCte"))
            .expect("the CTE");
        assert!(line.contains("output_rows=10,"), "{plan}");
        assert!(line.contains("sink_rows=10,"), "{plan}");
        Ok(())
    }

    /// Register the table `numbers` of the values `0..rows` (`v BIGINT NOT
    /// NULL`), in 4 partitions.
    fn register_numbers(ctx: &SessionContext, rows: i64) -> Result<()> {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
        let partitions = (0..4)
            .map(|partition| {
                let values = (partition..rows).step_by(4);
                let batch = RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![Arc::new(Int64Array::from_iter_values(values))],
                )?;
                Ok(vec![batch])
            })
            .collect::<Result<Vec<_>>>()?;
        ctx.register_table("numbers", Arc::new(MemTable::try_new(schema, partitions)?))?;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn recursions_of_many_rows_run_on_all_threads() {
        let setup = async |ctx: &SessionContext| register_numbers(ctx, 10_000);
        for sql in [
            // 5 terms of 10,000 rows.
            "WITH RECURSIVE t(n) AS (SELECT v FROM numbers UNION ALL \
             SELECT n + 10000 FROM t WHERE n < 40000) \
             SELECT count(*), sum(n), min(n), max(n) FROM t",
            // With UNION, only the new rows of an iteration are its rows: half
            // of the rows of the first iteration are rows of the static term.
            "WITH RECURSIVE t(n) AS (SELECT v FROM numbers UNION \
             SELECT n + 5000 FROM t WHERE n < 30000) \
             SELECT count(*), sum(n), min(n), max(n) FROM t",
            // With duplicates in the static term, and rows of every iteration
            // that are rows of the static term.
            "WITH RECURSIVE t(n, m) AS (SELECT v % 5000, v % 2 FROM numbers UNION \
             SELECT n + 2500, m FROM t WHERE n < 20000) \
             SELECT count(*), sum(n), sum(m) FROM t",
            // Rows of strings, and a hash join in the recursive term.
            "WITH RECURSIVE t(n, s) AS ( \
               SELECT v AS n, CAST(v AS VARCHAR) AS s FROM numbers WHERE v < 100 \
               UNION ALL \
               SELECT t.n + 100, concat(t.s, '.', numbers.v) FROM t \
               JOIN numbers ON numbers.v = t.n % 10 WHERE t.n < 1000) \
             SELECT * FROM t",
            // Without rows in the static term, the recursion does not start.
            "WITH RECURSIVE t(n) AS (SELECT v FROM numbers WHERE v < 0 UNION ALL \
             SELECT n + 1 FROM t) SELECT * FROM t",
            "WITH RECURSIVE t(n) AS (SELECT v FROM numbers WHERE v < 0 UNION \
             SELECT n + 1 FROM t) SELECT count(*), sum(n) FROM t",
        ] {
            for target_partitions in [1, 4, 16] {
                let config =
                    test_session_config().with_target_partitions(target_partitions);
                assert_native_recursion(sql, config, &setup).await;
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn distinct_recursions_compare_rows_like_datafusion() {
        // A UNION whose recursive term returns its working table ends after
        // its first iteration, if its rows are equal to the rows it returned
        // before (with UNION ALL, it would not end).
        for sql in [
            // Nulls are equal.
            "WITH RECURSIVE t(a) AS (SELECT CAST(NULL AS INT) UNION SELECT a FROM t) \
             SELECT * FROM t",
            "WITH RECURSIVE t(a, b) AS (SELECT 1, CAST(NULL AS VARCHAR) UNION \
             SELECT a, b FROM t) SELECT * FROM t",
            // And so are NaNs.
            "WITH RECURSIVE t(a) AS (SELECT CAST('NaN' AS DOUBLE) UNION \
             SELECT a + 1 FROM t) SELECT * FROM t",
            // Zeros of either sign.
            "WITH RECURSIVE t(a) AS (SELECT CAST(0.0 AS DOUBLE) UNION \
             SELECT -a FROM t) SELECT * FROM t",
            // Nested values.
            "WITH RECURSIVE t(a) AS (SELECT [1, NULL] UNION SELECT a FROM t) \
             SELECT * FROM t",
            "WITH RECURSIVE t(a) AS (SELECT named_struct('x', 1, 'y', \
             CAST(NULL AS VARCHAR)) UNION SELECT a FROM t) SELECT * FROM t",
        ] {
            assert_native_recursion(sql, test_session_config(), async |_| Ok(())).await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_limit_ends_a_recursion_that_does_not_end() -> Result<()> {
        // The rows of every term come before those of the next one, so the
        // first rows of the query are those of its first terms: the rows of
        // these queries are unambiguous.
        for sql in [
            "WITH RECURSIVE t(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM t) \
             SELECT * FROM t LIMIT 10",
            "WITH RECURSIVE t(n) AS (SELECT v FROM (VALUES (1), (2), (3)) AS s(v) \
             UNION ALL SELECT n + 3 FROM t) SELECT * FROM t LIMIT 9",
        ] {
            for target_partitions in [1, 4] {
                let config =
                    test_session_config().with_target_partitions(target_partitions);
                assert_native_recursion(sql, config, async |_| Ok(())).await;
            }
        }
        // Here, which rows DataFusion returns depends on the partitions of
        // its filter, which race (SQL does not define which rows a LIMIT
        // without an ORDER BY returns): the pipeline engine, which filters
        // the rows of the recursion in their order, returns those of the
        // first iterations, while DataFusion returns any rows of the query.
        let sql = "WITH RECURSIVE t(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM t) \
                   SELECT n * 2 AS m FROM t WHERE n % 3 = 0 LIMIT 5";
        for target_partitions in [1, 4] {
            let config = test_session_config().with_target_partitions(target_partitions);
            for ctx in [datafusion_context(config.clone()), pipeline_context(config)] {
                let output = with_timeout(run_sql(&ctx, sql)).await?;
                let rows = sorted_values(&output.batches);
                assert_eq!(rows.len(), 5, "{rows:?}");
                assert!(rows.windows(2).all(|pair| pair[0] < pair[1]), "{rows:?}");
                assert!(rows.iter().all(|m| m % 6 == 0), "{rows:?}");
                if output.plan.is::<PipelineRootExec>() {
                    assert_recursion_is_native(&lowered(&output), sql);
                    assert_eq!(rows, vec![6, 12, 18, 24, 30]);
                }
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn scans_read_the_working_table_of_the_recursive_query_of_their_name() {
        for sql in [
            // The static term of the inner recursive query reads the working
            // table of the outer one, the recursive term its own.
            "WITH RECURSIVE a(x) AS (SELECT 1 UNION ALL ( \
               WITH RECURSIVE b(y) AS (SELECT x FROM a UNION ALL \
                 SELECT y + 1 FROM b WHERE y < 3) \
               SELECT y + 10 FROM b WHERE y < 20)) \
             SELECT * FROM a",
            // Recursive queries of the same name in different subqueries.
            "SELECT * FROM \
               (WITH RECURSIVE t(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM t \
                WHERE n < 3) SELECT n FROM t) l, \
               (WITH RECURSIVE t(n) AS (SELECT 10 UNION ALL SELECT n + 1 FROM t \
                WHERE n < 12) SELECT n FROM t) r",
        ] {
            let root =
                assert_native_recursion(sql, test_session_config(), async |_| Ok(()))
                    .await;
            assert_eq!(count::<PhysicalRecursiveCte>(&root), 2, "{sql}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn every_iteration_runs_a_fresh_copy_of_the_recursive_term() {
        // The generator of a `generate_series` (which DataFusion executes)
        // produces its rows once: the second iteration would join with none
        // of them if it ran the plan of the first one again.
        for sql in [
            "WITH RECURSIVE t(n) AS (SELECT 1 UNION SELECT n + g.value FROM t \
             CROSS JOIN generate_series(1, 2) g WHERE n < 6) SELECT * FROM t",
            "WITH RECURSIVE t(n) AS (SELECT 1 UNION ALL SELECT n + g.value FROM t \
             JOIN generate_series(1, 3) g ON g.value = n % 3 + 1 WHERE n < 20) \
             SELECT * FROM t",
        ] {
            let root =
                assert_native_recursion(sql, test_session_config(), async |_| Ok(()))
                    .await;
            assert!(!bridged_plans(&root).is_empty(), "{sql}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn recursive_queries_anywhere_in_a_query() {
        let setup = async |ctx: &SessionContext| register_numbers(ctx, 1000);
        let numbers = "WITH RECURSIVE t(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM t \
                       WHERE n < 100)";
        for sql in [
            // In a scalar subquery.
            format!(
                "SELECT v, ({numbers} SELECT max(n) FROM t) AS m FROM numbers WHERE v < 3"
            ),
            // Below an aggregation and a sort.
            format!(
                "{numbers} SELECT n % 7 AS k, count(*), sum(n) FROM t GROUP BY n % 7 ORDER BY k"
            ),
            // On both sides of hash joins.
            format!("{numbers} SELECT t.n, v FROM t JOIN numbers ON v = t.n * 3"),
            format!("{numbers} SELECT t.n, v FROM numbers JOIN t ON v = t.n * 3"),
            // On the probe side of a join that cannot produce rows, which does
            // not read it (here, a recursion that does not end).
            "WITH RECURSIVE t(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM t) \
             SELECT * FROM (SELECT v FROM numbers WHERE v < 0) e JOIN t ON e.v = t.n"
                .to_string(),
            // In the inputs of unions.
            format!(
                "{numbers} SELECT n FROM t UNION ALL SELECT v FROM numbers WHERE v < 5"
            ),
            format!("{numbers} SELECT n FROM t UNION SELECT n * 2 FROM t"),
        ] {
            for target_partitions in [1, 4] {
                let config =
                    test_session_config().with_target_partitions(target_partitions);
                assert_native_recursion(&sql, config, &setup).await;
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn the_errors_of_an_iteration_fail_the_query() -> Result<()> {
        // The fourth iteration divides by zero.
        let sql = "WITH RECURSIVE t(n) AS (SELECT 3 UNION ALL \
                   SELECT n - 1 FROM t WHERE 10 / n > 0) SELECT * FROM t";
        let pipeline = pipeline_context(test_session_config());
        let plan = pipeline.sql(sql).await?.create_physical_plan().await?;
        let root = plan
            .downcast_ref::<PipelineRootExec>()
            .expect("a pipeline plan");
        assert_recursion_is_native(root.operator(), sql);
        for ctx in [datafusion_context(test_session_config()), pipeline] {
            let error = with_timeout(run_sql(&ctx, sql)).await.unwrap_err();
            assert!(error.to_string().contains("Divide by zero"), "{error}");
        }
        Ok(())
    }

    // ---------------------------------------------------------------------
    // Memory
    // ---------------------------------------------------------------------

    /// A session of the pipeline engine whose memory pool is `pool`.
    fn pipeline_context_with_pool(pool: Arc<dyn MemoryPool>) -> Result<SessionContext> {
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_pool(pool)
            .build_arc()?;
        let planner =
            PipelineQueryPlanner::new(Arc::new(DefaultPhysicalPlanner::default()));
        let state = SessionStateBuilder::new()
            .with_config(test_session_config())
            .with_runtime_env(runtime)
            .with_default_features()
            .with_query_planner(Arc::new(planner))
            .build();
        Ok(SessionContext::new_with_state(state))
    }

    /// A memory pool without a limit that records the memory that every
    /// consumer reserves, and its peak.
    #[derive(Debug, Default)]
    struct PeakPool {
        /// By the name of the consumer: the memory it reserves, and its peak.
        consumers: Mutex<HashMap<String, (usize, usize)>>,
    }

    impl PeakPool {
        fn record(&self, reservation: &MemoryReservation, grown: usize, shrunk: usize) {
            let mut consumers = self.consumers.lock();
            let (reserved, peak) = consumers
                .entry(reservation.consumer().name().to_owned())
                .or_default();
            *reserved = *reserved + grown - shrunk;
            *peak = (*peak).max(*reserved);
        }

        /// The memory that the consumer `name` reserves, and its peak.
        fn consumer(&self, name: &str) -> (usize, usize) {
            self.consumers.lock().get(name).copied().unwrap_or_default()
        }
    }

    impl fmt::Display for PeakPool {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "PeakPool")
        }
    }

    impl MemoryPool for PeakPool {
        fn name(&self) -> &str {
            "PeakPool"
        }

        fn grow(&self, reservation: &MemoryReservation, additional: usize) {
            self.record(reservation, additional, 0);
        }

        fn shrink(&self, reservation: &MemoryReservation, shrink: usize) {
            self.record(reservation, 0, shrink);
        }

        fn try_grow(
            &self,
            reservation: &MemoryReservation,
            additional: usize,
        ) -> Result<()> {
            self.grow(reservation, additional);
            Ok(())
        }

        fn reserved(&self) -> usize {
            self.consumers
                .lock()
                .values()
                .map(|(reserved, _)| reserved)
                .sum()
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn recursions_fail_once_the_memory_pool_is_exhausted() -> Result<()> {
        for union in ["UNION ALL", "UNION"] {
            // 100,000 rows of 8 bytes, collected from the static term, do not
            // fit in 256 KiB.
            let sql = format!(
                "WITH RECURSIVE t(n) AS (SELECT v FROM numbers {union} \
                 SELECT n + 1 FROM t WHERE n < 0) SELECT count(*) FROM t"
            );
            let ctx =
                pipeline_context_with_pool(Arc::new(GreedyMemoryPool::new(256 << 10)))?;
            register_numbers(&ctx, 100_000)?;
            let plan = ctx.sql(&sql).await?.create_physical_plan().await?;
            let root = plan
                .downcast_ref::<PipelineRootExec>()
                .expect("a pipeline plan");
            assert_recursion_is_native(root.operator(), &sql);
            let error = with_timeout(collect(plan, ctx.task_ctx()))
                .await
                .unwrap_err();
            assert!(
                matches!(error.find_root(), DataFusionError::ResourcesExhausted(_)),
                "{error}"
            );
            assert!(
                error.to_string().contains("PhysicalRecursiveCte[t]"),
                "{error}"
            );
            // DataFusion fails too.
            let ctx = datafusion_context(test_session_config());
            let state = SessionStateBuilder::new_from_existing(ctx.state())
                .with_runtime_env(
                    RuntimeEnvBuilder::new()
                        .with_memory_pool(Arc::new(GreedyMemoryPool::new(256 << 10)))
                        .build_arc()?,
                )
                .build();
            let ctx = SessionContext::new_with_state(state);
            register_numbers(&ctx, 100_000)?;
            let error = with_timeout(run_sql(&ctx, &sql)).await.unwrap_err();
            assert!(
                matches!(error.find_root(), DataFusionError::ResourcesExhausted(_)),
                "{error}"
            );
            // In a pool in which the rows fit, both engines return them.
            let ctx =
                pipeline_context_with_pool(Arc::new(GreedyMemoryPool::new(64 << 20)))?;
            register_numbers(&ctx, 100_000)?;
            let actual = with_timeout(run_sql(&ctx, &sql)).await?;
            let ctx = datafusion_context(test_session_config());
            register_numbers(&ctx, 100_000)?;
            let expected = with_timeout(run_sql(&ctx, &sql)).await?;
            compare_outputs(&expected, &actual).map_err(DataFusionError::Internal)?;
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn recursions_hold_the_rows_of_two_terms_at_most() -> Result<()> {
        // 51 terms of 4,096 rows of 8 bytes (32 KiB): 1.6 MiB in all.
        let term = 4096 * size_of::<i64>();
        for union in ["UNION ALL", "UNION"] {
            let sql = format!(
                "WITH RECURSIVE t(n) AS (SELECT v FROM numbers {union} \
                 SELECT n + 4096 FROM t WHERE n < 4096 * 50) \
                 SELECT count(*), sum(n), max(n) FROM t"
            );
            let pool = Arc::new(PeakPool::default());
            let ctx = pipeline_context_with_pool(Arc::clone(&pool) as _)?;
            register_numbers(&ctx, 4096)?;
            let actual = with_timeout(run_sql(&ctx, &sql)).await?;
            assert_recursion_is_native(&lowered(&actual), &sql);
            let ctx = datafusion_context(test_session_config());
            register_numbers(&ctx, 4096)?;
            let expected = with_timeout(run_sql(&ctx, &sql)).await?;
            compare_outputs(&expected, &actual).map_err(DataFusionError::Internal)?;
            // The rows of the last term and those of the current iteration.
            let (reserved, peak) = pool.consumer("PhysicalRecursiveCte[t]");
            assert!(peak >= term && peak <= 3 * term, "{union}: {peak} bytes");
            assert_eq!(reserved, 0, "{union}");
            // For UNION, the hash table of the rows grows with all of them.
            let (reserved, peak) = pool.consumer("PhysicalRecursiveCte[t] hash table");
            match union {
                "UNION" => assert!(peak > 51 * term, "{peak} bytes"),
                _ => assert_eq!(peak, 0),
            }
            assert_eq!(reserved, 0, "{union}");
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn dropping_the_output_ends_a_recursion() -> Result<()> {
        let pool = Arc::new(PeakPool::default());
        let ctx = pipeline_context_with_pool(Arc::clone(&pool) as _)?;
        let sql = "WITH RECURSIVE t(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM t) \
                   SELECT * FROM t";
        let mut stream = ctx.sql(sql).await?.execute_stream().await?;
        let mut rows = 0;
        while rows < 100 {
            rows += with_timeout(stream.next())
                .await
                .expect("more rows")?
                .num_rows();
        }
        drop(stream);
        // The recursion stops, and releases its memory.
        let rows = || pool.consumer("PhysicalRecursiveCte[t]");
        with_timeout(async {
            while rows().0 > 0 {
                tokio::time::sleep(Duration::from_millis(10)).await;
            }
        })
        .await;
        assert!(rows().1 > 0);
        Ok(())
    }
}
