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
//! buffers of materialized CTEs and the generators of `generate_series`
//! (which DataFusion executes), the files of the scans. The recursive term
//! that is lowered with the query is a child of the operator only for
//! `EXPLAIN` (and the plans it bridges, see
//! [`crate::bridge::bridged_plans`]): it is never executed.
//!
//! Every execution of the plan shares some of that state, which
//! `reset_plan_states` resets in place. So that concurrent executions of a
//! query (of one plan) do not reset and read each other's, every execution
//! of the operator runs a copy of the recursive term with scalar subqueries,
//! materialized CTEs and dynamic filters of its own (see
//! `with_own_scalar_subqueries`, `with_own_materialized_ctes` and
//! `with_own_dynamic_filters`).
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
//! not fill. So [`PhysicalRecursiveCte::try_lower`] replaces the
//! `WorkTableExec`s of the recursive term by [`PipelineWorkTableExec`]s,
//! DataFusion `ExecutionPlan`s that the planner lowers to
//! [`WorkTableScanSource`]s, and which read the working table of the CTE in
//! the same way where DataFusion executes them: in the plans of the
//! `BridgeOperator`s of the recursive term (e.g. a node that no native
//! operator implements, or a cross join that the planner leaves to DataFusion
//! in a session with a memory limit). The recursion itself always runs
//! natively, whatever DataFusion executes in its terms.
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
//!
//! The memory of the rows is the memory that their batches retain
//! (`RetainedMemory`): all of their buffers, also those of the arrays that
//! they are zero-copy slices of (e.g. a row that a limit slices from a batch
//! of a table retains the whole batch), each counted once for all the batches
//! that share it (e.g. the chunks that a scan slices from one batch, or a
//! column that the recursive term passes on to the next term).

use std::collections::HashMap;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow::array::{ArrayRef, BooleanArray, RecordBatch, RecordBatchOptions};
use arrow::compute::filter_record_batch;
use arrow::datatypes::{Schema, SchemaRef};
use datafusion_common::instant::Instant;
use datafusion_common::metadata::FieldMetadata;
use datafusion_common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion_common::utils::memory::RecordBatchMemoryTracker;
use datafusion_common::{DataFusionError, Result, internal_datafusion_err, internal_err};
use datafusion_datasource::file::FileSource;
use datafusion_datasource::file_scan_config::FileScanConfigBuilder;
use datafusion_datasource::source::DataSourceExec;
use datafusion_datasource_parquet::source::ParquetSource;
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryPool, MemoryReservation};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_expr::physical_planning_context::ScalarSubqueryResults;
use datafusion_physical_expr::expressions::{DynamicFilterPhysicalExpr, lit};
use datafusion_physical_expr::projection::ProjectionExpr;
use datafusion_physical_expr::scalar_subquery::ScalarSubqueryExpr;
use datafusion_physical_expr::{EquivalenceProperties, Partitioning, PhysicalExpr};
use datafusion_physical_plan::aggregates::AggregateExec;
use datafusion_physical_plan::aggregates::group_values::{GroupValues, new_group_values};
use datafusion_physical_plan::aggregates::order::GroupClustering;
use datafusion_physical_plan::coop::cooperative;
use datafusion_physical_plan::execution_plan::{
    Boundedness, EmissionType, SchedulingType, reset_plan_states,
};
use datafusion_physical_plan::filter::{FilterExec, FilterExecBuilder};
use datafusion_physical_plan::joins::HashJoinExec;
use datafusion_physical_plan::materialized_cte::{
    MaterializedCteBuffer, MaterializedCteExec, MaterializedCteScanExec,
};
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::recursive_query::RecursiveQueryExec;
use datafusion_physical_plan::scalar_subquery::{ScalarSubqueryExec, ScalarSubqueryLink};
use datafusion_physical_plan::sorts::sort::SortExec;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::work_table::WorkTableExec;
use datafusion_physical_plan::{
    ChildrenPropertiesMode, DisplayAs, DisplayFormatType, ExecutionPlan,
    ExecutionPlanProperties, PlanProperties, ReplaceChildrenOptions,
};
use futures::future::BoxFuture;
use parking_lot::Mutex;

use crate::chunk::{DEFAULT_VECTOR_SIZE, DataChunk};
use crate::executor::Executor;
use crate::operator::{
    ExecutionContext, GlobalSinkState, GlobalSourceState, LocalSinkState,
    LocalSourceState, OperatorMetrics, PhysicalOperator, SinkFinalize, SinkResult,
    SourceResult, TaskInterrupt, downcast_state, downcast_state_mut,
};
use crate::pipeline::{MetaPipelineBuilder, PipelineBuilder};
use crate::planner::lower_execution_plan_with_runtime;

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
    /// The DataFusion plan of the recursive term, with
    /// [`PipelineWorkTableExec`]s, which every iteration resets and lowers.
    recursive_plan: Arc<dyn ExecutionPlan>,
}

impl PhysicalRecursiveCte {
    /// Lower a DataFusion `RecursiveQueryExec` to a [`PhysicalRecursiveCte`]
    /// with the same schema. `lower_term` lowers its static and its recursive
    /// term (in pipelines that run until their input ends: the CTE reads all
    /// of their rows). Returns `Ok(None)` for any other node.
    ///
    /// The `WorkTableExec`s of the recursive term that scan the working table
    /// of the query are replaced by [`PipelineWorkTableExec`]s first (see
    /// `with_pipeline_work_tables`), which `lower_term` lowers to
    /// [`WorkTableScanSource`]s, and which read the working table of the CTE
    /// where DataFusion executes them too.
    ///
    /// DataFusion executes the first partition of every term, which plans
    /// have exactly one of: a recursive query whose terms have other numbers
    /// of partitions, or other data types than the query, is left to
    /// DataFusion. So is a query with a scan of its working table whose
    /// columns are not columns of the query (which only a plan built by hand
    /// can have, see `work_table_projection`).
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
        let recursive_plan = with_pipeline_work_tables(exec.recursive_term(), name)?;
        let static_term = lower_term(exec.static_term())?;
        let recursive_term = lower_term(&recursive_plan)?;
        Ok(Some(Self {
            name: name.to_owned(),
            schema,
            is_distinct: exec.is_distinct(),
            children: vec![static_term, recursive_term],
            recursive_plan,
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

    /// A fresh copy of the recursive term for an iteration: `plan`, the
    /// DataFusion plan of the recursive term of the execution (see
    /// [`Iterations`]), reset (see `reset_plan_states`) and lowered for the
    /// session of `task_ctx`. Its scans of the working table read the working
    /// table of the CTE, wherever they are: natively, or in the plans that
    /// DataFusion executes (see [`PipelineWorkTableExec`]).
    fn lower_recursive_term(
        plan: &Arc<dyn ExecutionPlan>,
        task_ctx: &TaskContext,
    ) -> Result<Arc<dyn PhysicalOperator>> {
        let plan = reset_plan_states(Arc::clone(plan))?;
        let options = task_ctx.session_config().options();
        lower_execution_plan_with_runtime(
            &plan,
            options.as_ref(),
            task_ctx.runtime_env().as_ref(),
        )
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
        let iterations = match recursion.iterations.take() {
            Some(iterations) => iterations,
            None => Iterations::try_new(&self.name, &ctx.task_ctx, &self.recursive_plan)?,
        };
        let iterations = recursion.iterations.insert(iterations);
        // The scans of the working table read the rows of the last term.
        *iterations.work_tables.table.lock() = Some(working_table);
        iterations.count += 1;
        let term =
            Self::lower_recursive_term(&iterations.recursive_plan, &iterations.task_ctx)?;
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

/// `plan` (a DataFusion plan) with [`PipelineWorkTableExec`]s in place of its
/// scans of the working table of the recursive query `name` (see
/// [`work_table_scans`]), which read the working table of a
/// [`PhysicalRecursiveCte`] instead of the one of DataFusion's
/// `RecursiveQueryExec`: natively (lowered to [`WorkTableScanSource`]s), and
/// also where DataFusion executes them, in a plan that a `BridgeOperator`
/// executes. The nodes above them are rebuilt (as DataFusion's
/// `RecursiveQueryExec` rebuilds them when it binds its `WorkTableExec`s).
fn with_pipeline_work_tables(
    plan: &Arc<dyn ExecutionPlan>,
    name: &str,
) -> Result<Arc<dyn ExecutionPlan>> {
    if let Some(scan) = plan.downcast_ref::<WorkTableExec>() {
        return Ok(if scan.name() == name {
            Arc::new(PipelineWorkTableExec::new(name, plan.schema()))
        } else {
            Arc::clone(plan)
        });
    }
    // The recursive term (the second child) of a recursive query of the same
    // name scans the working table of that query.
    let shadows = plan
        .downcast_ref::<RecursiveQueryExec>()
        .is_some_and(|query| query.name() == name);
    let children = plan.children();
    let mut replaced = false;
    let mut new_children = Vec::with_capacity(children.len());
    for (index, child) in children.into_iter().enumerate() {
        let new_child = if shadows && index == 1 {
            Arc::clone(child)
        } else {
            with_pipeline_work_tables(child, name)?
        };
        replaced |= !Arc::ptr_eq(&new_child, child);
        new_children.push(new_child);
    }
    if !replaced {
        return Ok(Arc::clone(plan));
    }
    Arc::clone(plan).replace_children(
        new_children,
        ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
    )
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

/// What the iterations of an execution of a [`PhysicalRecursiveCte`] run
/// with.
#[derive(Debug)]
struct Iterations {
    /// The DataFusion plan of the recursive term, with dynamic filters of
    /// this execution (see [`with_own_dynamic_filters`]).
    recursive_plan: Arc<dyn ExecutionPlan>,
    /// The task context of the executions of the iterations, with
    /// `work_tables` in its session config.
    task_ctx: Arc<TaskContext>,
    /// The working table of the CTE.
    work_tables: Arc<WorkTables>,
    /// The number of iterations started.
    count: usize,
}

impl Iterations {
    /// What the iterations of the CTE `name`, with the recursive term
    /// `recursive_plan`, run with in an execution with the task context
    /// `task_ctx`: a copy of `recursive_plan` with scalar subqueries,
    /// materialized CTEs and dynamic filters of its own, and `task_ctx`, whose
    /// session config also holds the working table of the CTE, in front of
    /// those of the recursive queries around it (in the session config of
    /// `task_ctx`).
    fn try_new(
        name: &str,
        task_ctx: &TaskContext,
        recursive_plan: &Arc<dyn ExecutionPlan>,
    ) -> Result<Self> {
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
        let recursive_plan = with_own_scalar_subqueries(recursive_plan)?;
        // The materialized CTEs before the filters: `MaterializedCteExec::new`
        // (also when the filters are copied) gives the buffer of the CTE its
        // body.
        let recursive_plan = with_own_materialized_ctes(&recursive_plan)?;
        Ok(Self {
            recursive_plan: with_own_dynamic_filters(&recursive_plan)?,
            task_ctx: Arc::new(task_ctx),
            work_tables,
            count: 0,
        })
    }
}

/// The new result containers of the scalar subqueries of a copy of a plan,
/// after the containers of the plan that they replace.
type NewResults = Vec<(ScalarSubqueryResults, ScalarSubqueryResults)>;

/// The new container of `results` in `new_results`, if it has one.
fn new_results_of<'a>(
    new_results: &'a NewResults,
    results: &ScalarSubqueryResults,
) -> Option<&'a ScalarSubqueryResults> {
    new_results
        .iter()
        .find(|(old, _)| ScalarSubqueryResults::ptr_eq(old, results))
        .map(|(_, new)| new)
}

/// A copy of `plan` (a DataFusion plan) with scalar subqueries of its own:
/// every `ScalarSubqueryExec` of `plan` is built again with a new container
/// of the results of its subqueries (`ScalarSubqueryResults`), which the
/// `ScalarSubqueryExpr`s that read them read instead.
///
/// A `ScalarSubqueryExec` shares the container with the expressions that read
/// its results, and with every execution of the plan, and `reset_plan_states`
/// clears it in place: an iteration of an execution of a recursive query (of
/// one plan) would clear the results that the iterations of the others read,
/// which then fail (`ScalarSubqueryExpr evaluated before the subquery was
/// executed`). DataFusion's planner puts the scalar subqueries of a query at
/// its root, around its recursive queries, but a plan built otherwise may
/// have some in a recursive term.
///
/// The expressions are built again in the predicates of `FilterExec`s and of
/// Parquet scans, and in the expressions of `ProjectionExec`s. A container
/// that another node of `plan` reads is kept, with its `ScalarSubqueryExec`
/// (whose executions then share it), and so is `plan` if a node cannot be
/// built again (e.g. a hash join with a prepared build side, with other
/// children).
fn with_own_scalar_subqueries(
    plan: &Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>> {
    // The containers of the scalar subqueries, with their numbers of results:
    // all of their indexes, of their subqueries and of their readers.
    let mut containers: Vec<(ScalarSubqueryResults, usize)> = vec![];
    plan.apply(|node| {
        if let Some(exec) = node.downcast_ref::<ScalarSubqueryExec>() {
            let index = exec.subqueries().iter().map(|link| link.index.as_usize());
            let len = index.max().map_or(0, |index| index + 1);
            match containers.iter_mut().find(|(results, _)| {
                ScalarSubqueryResults::ptr_eq(results, exec.results())
            }) {
                Some((_, count)) => *count = (*count).max(len),
                None => containers.push((exec.results().clone(), len)),
            }
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    if containers.is_empty() {
        return Ok(Arc::clone(plan));
    }
    let mut shared: Vec<ScalarSubqueryResults> = vec![];
    plan.apply(|node| {
        let rebuilt = node.is::<FilterExec>()
            || node.is::<ProjectionExec>()
            || node.downcast_ref::<DataSourceExec>().is_some_and(|scan| {
                scan.downcast_to_file_source::<ParquetSource>().is_some()
            });
        node.apply_expressions(&mut |root| {
            root.apply(|expr| {
                if let Some(read) = expr.downcast_ref::<ScalarSubqueryExpr>() {
                    let container = containers.iter_mut().find(|(results, _)| {
                        ScalarSubqueryResults::ptr_eq(results, read.results())
                    });
                    match container {
                        Some((_, count)) if rebuilt => {
                            *count = (*count).max(read.index().as_usize() + 1);
                        }
                        Some((results, _)) => shared.push(results.clone()),
                        None => {}
                    }
                }
                Ok(TreeNodeRecursion::Continue)
            })
        })?;
        Ok(TreeNodeRecursion::Continue)
    })?;
    let new_results: NewResults = containers
        .into_iter()
        .filter(|(results, _)| {
            !shared
                .iter()
                .any(|shared| ScalarSubqueryResults::ptr_eq(shared, results))
        })
        .map(|(results, count)| (results, ScalarSubqueryResults::new(count)))
        .collect();
    if new_results.is_empty() {
        return Ok(Arc::clone(plan));
    }
    Ok(
        with_new_results(plan, &new_results).unwrap_or_else(|error| {
            log::debug!("The scalar subqueries of a recursive term are shared: {error}");
            Arc::clone(plan)
        }),
    )
}

/// `plan` with the new containers of `new_results` in place of the containers
/// that they replace, in the `ScalarSubqueryExec`s and in the expressions that
/// read their results (see [`with_own_scalar_subqueries`]).
fn with_new_results(
    plan: &Arc<dyn ExecutionPlan>,
    new_results: &NewResults,
) -> Result<Arc<dyn ExecutionPlan>> {
    let children = plan.children();
    let mut replaced = false;
    let mut new_children = Vec::with_capacity(children.len());
    for child in children {
        let new_child = with_new_results(child, new_results)?;
        replaced |= !Arc::ptr_eq(&new_child, child);
        new_children.push(new_child);
    }
    let exec = plan
        .downcast_ref::<ScalarSubqueryExec>()
        .and_then(|exec| Some((exec, new_results_of(new_results, exec.results())?)));
    let node = match exec {
        Some((exec, results)) => {
            // Built with its new container, which `replace_children` keeps.
            let mut children = new_children.into_iter();
            let Some(input) = children.next() else {
                return internal_err!("ScalarSubqueryExec has an input");
            };
            let subqueries = exec
                .subqueries()
                .iter()
                .zip(children)
                .map(|(link, plan)| ScalarSubqueryLink {
                    plan,
                    index: link.index,
                })
                .collect();
            Arc::new(ScalarSubqueryExec::new(input, subqueries, results.clone()))
        }
        None if replaced => Arc::clone(plan).replace_children(
            new_children,
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )?,
        None => Arc::clone(plan),
    };
    with_new_results_read(node, new_results)
}

/// `node` reading the results of the new containers of `new_results` in place
/// of those of the containers that they replace, if it is a `FilterExec`, a
/// `ProjectionExec` or a scan of Parquet files that reads some of them.
fn with_new_results_read(
    node: Arc<dyn ExecutionPlan>,
    new_results: &NewResults,
) -> Result<Arc<dyn ExecutionPlan>> {
    if let Some(filter) = node.downcast_ref::<FilterExec>() {
        return Ok(
            match with_new_results_in(filter.predicate(), new_results)? {
                Some(predicate) => Arc::new(
                    FilterExecBuilder::from(filter)
                        .with_predicate(predicate)
                        .build()?,
                ),
                None => node,
            },
        );
    }
    if let Some(projection) = node.downcast_ref::<ProjectionExec>() {
        let mut rewritten = false;
        let exprs = projection
            .expr()
            .iter()
            .map(|projected| {
                Ok(match with_new_results_in(&projected.expr, new_results)? {
                    Some(expr) => {
                        rewritten = true;
                        ProjectionExpr::new(expr, projected.alias.clone())
                    }
                    None => projected.clone(),
                })
            })
            .collect::<Result<Vec<_>>>()?;
        if !rewritten {
            return Ok(node);
        }
        let schema = projection.schema();
        let rebuilt = ProjectionExec::try_new_with_schema_metadata(
            exprs,
            Arc::clone(projection.input()),
            &schema,
        )?;
        if rebuilt.schema() != schema {
            return internal_err!("The projection does not keep its schema");
        }
        return Ok(Arc::new(rebuilt));
    }
    if let Some(scan) = node.downcast_ref::<DataSourceExec>()
        && let Some((config, source)) = scan.downcast_to_file_source::<ParquetSource>()
        && let Some(predicate) = source.filter()
        && let Some(predicate) = with_new_results_in(&predicate, new_results)?
    {
        let config = FileScanConfigBuilder::from(config.clone())
            .with_source(Arc::new(source.with_predicate(predicate)))
            .build();
        return Ok(Arc::new(scan.clone().with_data_source(Arc::new(config))));
    }
    Ok(node)
}

/// `expr` reading the results of the new containers of `new_results` in place
/// of those of the containers that they replace, `None` if it reads none of
/// them.
fn with_new_results_in(
    expr: &Arc<dyn PhysicalExpr>,
    new_results: &NewResults,
) -> Result<Option<Arc<dyn PhysicalExpr>>> {
    let rewritten = Arc::clone(expr).transform_up(|expr| {
        let Some(read) = expr.downcast_ref::<ScalarSubqueryExpr>() else {
            return Ok(Transformed::no(expr));
        };
        let Some(results) = new_results_of(new_results, read.results()) else {
            return Ok(Transformed::no(expr));
        };
        // Its field: its type, nullability and metadata.
        let field = read.return_field(&Schema::empty())?;
        let metadata = FieldMetadata::new_from_field(&field);
        let new_read = ScalarSubqueryExpr::new_with_metadata(
            field.data_type().clone(),
            field.is_nullable(),
            (!metadata.is_empty()).then_some(metadata),
            read.index(),
            results.clone(),
        );
        Ok(Transformed::yes(Arc::new(new_read) as Arc<dyn PhysicalExpr>))
    })?;
    Ok(rewritten.transformed.then_some(rewritten.data))
}

/// The new buffers of the materialized CTEs of a copy of a plan, by the ids of
/// the buffers of the plan that they replace.
type NewBuffers = HashMap<u64, Arc<MaterializedCteBuffer>>;

/// A copy of `plan` (a DataFusion plan) with materialized CTEs of its own:
/// every `MaterializedCteExec` of `plan` (of `WITH x AS MATERIALIZED (...)`,
/// which DataFusion executes) is built again with a new buffer, with the same
/// id, to which the scans of the CTE (`MaterializedCteScanExec`s) in `plan`
/// are bound.
///
/// A `MaterializedCteExec` and its scans share the buffer into which the
/// first of them to run computes the body of the CTE. Every execution of the
/// plan shares it too, and `reset_plan_states` resets it in place (so that
/// every iteration computes the body again, which may read the working table),
/// and gives it the body of the new plan: the iterations of concurrent
/// executions of a recursive query (of one plan) would reset each other's
/// buffers, read each other's rows, and run each other's plans, which then
/// fail. So every execution of a [`PhysicalRecursiveCte`] runs a copy of its
/// recursive term with materialized CTEs of its own, which its iterations
/// reset.
///
/// A CTE declared in the recursive term is only scanned in it. The scans of a
/// CTE declared around the recursive query keep the buffer of that CTE, which
/// is computed once for the whole query. If a node of `plan` cannot be built
/// again (e.g. a hash join with a prepared build side, with other children),
/// `plan` is kept as it is.
fn with_own_materialized_ctes(
    plan: &Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let mut buffers = NewBuffers::new();
    plan.apply(|node| {
        if let Some(cte) = node.downcast_ref::<MaterializedCteExec>() {
            let id = cte.buffer().id();
            buffers.entry(id).or_insert_with(|| {
                Arc::new(MaterializedCteBuffer::new(id, materialized_cte_name(cte)))
            });
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    if buffers.is_empty() {
        return Ok(Arc::clone(plan));
    }
    Ok(with_new_buffers(plan, &buffers).unwrap_or_else(|error| {
        log::debug!("The materialized CTEs of a recursive term are shared: {error}");
        Arc::clone(plan)
    }))
}

/// The name of `cte`, for the memory consumers and the errors of the new
/// buffer of a copy: `MaterializedCteExec` only shows it in its display
/// (`name=<name>` in a tree render), whatever it shows otherwise.
fn materialized_cte_name(cte: &MaterializedCteExec) -> String {
    struct TreeRender<'a>(&'a MaterializedCteExec);
    impl fmt::Display for TreeRender<'_> {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            self.0.fmt_as(DisplayFormatType::TreeRender, f)
        }
    }
    let display = TreeRender(cte).to_string();
    match display.strip_prefix("name=") {
        Some(name) => name.to_owned(),
        None => display,
    }
}

/// `plan` with the new buffers of `buffers` in place of the buffers that they
/// replace, in the `MaterializedCteExec`s and their scans (see
/// [`with_own_materialized_ctes`]).
fn with_new_buffers(
    plan: &Arc<dyn ExecutionPlan>,
    buffers: &NewBuffers,
) -> Result<Arc<dyn ExecutionPlan>> {
    if let Some(scan) = plan.downcast_ref::<MaterializedCteScanExec>()
        && let Some(buffer) = buffers.get(&scan.id())
    {
        return Ok(Arc::new(scan.bind(Arc::clone(buffer))));
    }
    let children = plan.children();
    let mut replaced = false;
    let mut new_children = Vec::with_capacity(children.len());
    for child in children {
        let new_child = with_new_buffers(child, buffers)?;
        replaced |= !Arc::ptr_eq(&new_child, child);
        new_children.push(new_child);
    }
    if let Some(cte) = plan.downcast_ref::<MaterializedCteExec>()
        && let Some(buffer) = buffers.get(&cte.buffer().id())
    {
        // Built with its new buffer, not with `replace_children`, which would
        // give the buffer of `plan` the new body.
        let Ok([body, continuation]) = <[_; 2]>::try_from(new_children) else {
            return internal_err!("MaterializedCteExec has a body and a continuation");
        };
        return Ok(Arc::new(MaterializedCteExec::new(
            body,
            continuation,
            Arc::clone(buffer),
        )));
    }
    if !replaced {
        return Ok(Arc::clone(plan));
    }
    Arc::clone(plan).replace_children(
        new_children,
        ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
    )
}

/// The new dynamic filters of a copy of a plan, by the `expression_id`s of the
/// filters of the plan that they replace.
type NewFilters = HashMap<u64, Arc<DynamicFilterPhysicalExpr>>;

/// A copy of `plan` (a DataFusion plan) with dynamic filters of its own: every
/// `DynamicFilterPhysicalExpr` that a node of `plan` produces (a hash join,
/// an aggregation or a TopK, the nodes that produce them) is replaced by a new
/// filter with the same children, in the node that produces it (an
/// aggregation also with bounds of its own, see `with_own_bounds`) and in the
/// nodes that read it (the predicates of `FilterExec`s and of Parquet scans,
/// into which DataFusion pushes the filters).
///
/// The nodes of a plan share their dynamic filters with every execution of the
/// plan, and `reset_plan_states` resets them in place, for the next execution:
/// the iterations of concurrent executions of a recursive query (of one plan)
/// would set and read each other's filters, e.g. the bounds of the build keys
/// of a hash join in another iteration, with which the scan of its probe side
/// would skip the rows of this one. So every execution of a
/// [`PhysicalRecursiveCte`] runs a copy of its recursive term with filters of
/// its own, which its iterations reset.
///
/// A node that reads a filter but that this cannot rebuild with another one
/// keeps the filter of `plan`, which no execution sets any more (its producer
/// sets a new one): it prunes nothing with it.
fn with_own_dynamic_filters(
    plan: &Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn ExecutionPlan>> {
    let mut filters = NewFilters::new();
    plan.apply(|node| {
        for produced in node.dynamic_expressions_produced() {
            if let Some(filter) = produced.downcast_ref::<DynamicFilterPhysicalExpr>()
                && let Some(id) = filter.expression_id()
            {
                filters.entry(id).or_insert_with(|| {
                    let children = filter.children().into_iter().cloned().collect();
                    Arc::new(DynamicFilterPhysicalExpr::new(children, lit(true)))
                });
            }
        }
        Ok(TreeNodeRecursion::Continue)
    })?;
    if filters.is_empty() {
        return Ok(Arc::clone(plan));
    }
    Ok(with_new_filters(plan, &filters))
}

/// `plan` with the new filters of `filters` in place of the filters that they
/// replace, in the nodes that produce and read them (see
/// [`with_own_dynamic_filters`]).
///
/// A node that cannot be built again (with another filter, e.g. an
/// aggregation that cannot have bounds of its own, or with other children,
/// e.g. a hash join with a prepared build side) is kept as it is, rather than
/// failing the query: the new filters that the readers of its filters read
/// are then set by nothing, so they prune nothing, and a subtree that cannot
/// be built again keeps the filters of `plan`, shared with its other
/// executions.
fn with_new_filters(
    plan: &Arc<dyn ExecutionPlan>,
    filters: &NewFilters,
) -> Arc<dyn ExecutionPlan> {
    let kept = |error: DataFusionError| {
        log::debug!(
            "{} keeps the dynamic filters of its plan: {error}",
            plan.name()
        );
        Arc::clone(plan)
    };
    let children = plan.children();
    let mut replaced = false;
    let mut new_children = Vec::with_capacity(children.len());
    for child in children {
        let new_child = with_new_filters(child, filters);
        replaced |= !Arc::ptr_eq(&new_child, child);
        new_children.push(new_child);
    }
    let node = if replaced {
        match Arc::clone(plan).replace_children(
            new_children,
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        ) {
            Ok(node) => node,
            Err(error) => return kept(error),
        }
    } else {
        Arc::clone(plan)
    };
    with_new_produced_filter(Arc::clone(&node), filters)
        .and_then(|node| with_new_read_filters(node, filters))
        .unwrap_or_else(|error| {
            log::debug!(
                "{} keeps the dynamic filters of its plan: {error}",
                node.name()
            );
            node
        })
}

/// `node` producing the new filter of `filters` in place of the filter that
/// it produces, if it is a hash join, an aggregation or a TopK that produces
/// one.
// `with_dynamic_filter_expr` is deprecated: DataFusion only uses it to restore
// the filters of the plans that it decodes, and this to copy plans.
#[expect(deprecated)]
fn with_new_produced_filter(
    node: Arc<dyn ExecutionPlan>,
    filters: &NewFilters,
) -> Result<Arc<dyn ExecutionPlan>> {
    let Some(filter) = node
        .dynamic_expressions_produced()
        .iter()
        .find_map(|produced| filters.get(&produced.expression_id()?))
        .cloned()
    else {
        return Ok(node);
    };
    if let Some(join) = node.downcast_ref::<HashJoinExec>() {
        return Ok(Arc::new(
            join.builder().build()?.with_dynamic_filter_expr(filter)?,
        ));
    }
    if let Some(aggregate) = node.downcast_ref::<AggregateExec>() {
        return Ok(Arc::new(
            with_own_bounds(aggregate)?.with_dynamic_filter_expr(filter)?,
        ));
    }
    if let Some(sort) = node.downcast_ref::<SortExec>() {
        return Ok(Arc::new(sort.clone().with_dynamic_filter_expr(filter)?));
    }
    Ok(node)
}

/// A copy of `aggregate`, an aggregation that produces a dynamic filter, with
/// bounds of its own: DataFusion's aggregation (when DataFusion executes it)
/// sets the filter from the bounds of its aggregates (e.g. `v > <the largest
/// v so far>` for `MAX(v)`), which its partitions share, and so would the
/// copies of `aggregate` that share them. Only the aggregations without groups
/// and limits of the first stage (of `MIN` and `MAX`) produce dynamic
/// filters, which `AggregateExec::try_new` builds again with new bounds. Any
/// other aggregation (which it would build otherwise) is an error: it is then
/// kept as it is, with the filter of the plan (see [`with_new_filters`]).
fn with_own_bounds(aggregate: &AggregateExec) -> Result<AggregateExec> {
    let rebuilt = AggregateExec::try_new(
        *aggregate.mode(),
        aggregate.group_expr().clone(),
        aggregate.aggr_expr().to_vec(),
        aggregate.filter_expr().to_vec(),
        Arc::clone(aggregate.input()),
        aggregate.input_schema(),
    )?;
    if aggregate.limit_options().is_none()
        && rebuilt.limit_options().is_none()
        && rebuilt.schema() == aggregate.schema()
        && !rebuilt.dynamic_expressions_produced().is_empty()
    {
        Ok(rebuilt)
    } else {
        internal_err!("{} cannot be built again", aggregate.name())
    }
}

/// `node` reading the new filters of `filters` in place of the filters that
/// they replace, if it is a `FilterExec` or a scan of Parquet files that
/// reads some of them.
fn with_new_read_filters(
    node: Arc<dyn ExecutionPlan>,
    filters: &NewFilters,
) -> Result<Arc<dyn ExecutionPlan>> {
    if let Some(filter) = node.downcast_ref::<FilterExec>() {
        return Ok(match with_new_filters_in(filter.predicate(), filters)? {
            Some(predicate) => Arc::new(
                FilterExecBuilder::from(filter)
                    .with_predicate(predicate)
                    .build()?,
            ),
            None => node,
        });
    }
    if let Some(scan) = node.downcast_ref::<DataSourceExec>()
        && let Some((config, source)) = scan.downcast_to_file_source::<ParquetSource>()
        && let Some(predicate) = source.filter()
        && let Some(predicate) = with_new_filters_in(&predicate, filters)?
    {
        let config = FileScanConfigBuilder::from(config.clone())
            .with_source(Arc::new(source.with_predicate(predicate)))
            .build();
        return Ok(Arc::new(scan.clone().with_data_source(Arc::new(config))));
    }
    Ok(node)
}

/// `expr` with the new filters of `filters` in place of the filters that they
/// replace, `None` if it reads none of them.
fn with_new_filters_in(
    expr: &Arc<dyn PhysicalExpr>,
    filters: &NewFilters,
) -> Result<Option<Arc<dyn PhysicalExpr>>> {
    let rewritten = Arc::clone(expr).transform_up(|expr| {
        let new_filter = expr
            .downcast_ref::<DynamicFilterPhysicalExpr>()
            .and_then(|filter| filters.get(&filter.expression_id()?));
        let Some(new_filter) = new_filter else {
            return Ok(Transformed::no(expr));
        };
        // The children of a filter that was pushed through projections are
        // the columns of the node that reads it: so are those of its copy.
        let children = expr.children().into_iter().cloned().collect();
        let new_filter: Arc<dyn PhysicalExpr> = Arc::clone(new_filter) as _;
        Ok(Transformed::yes(new_filter.with_new_children(children)?))
    })?;
    Ok(rewritten.transformed.then_some(rewritten.data))
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
/// working table of the next iteration. The memory that they retain is
/// released with it.
#[derive(Debug)]
struct WorkingTable {
    schema: SchemaRef,
    batches: Vec<RecordBatch>,
    /// Blocks of at most [`DEFAULT_VECTOR_SIZE`] rows of `batches`, in order:
    /// `(batch, offset, rows)`.
    blocks: Vec<(usize, usize, usize)>,
    rows: usize,
    /// The memory that the recursion retains, which `batches` are part of.
    memory: Arc<RetainedMemory>,
}

impl WorkingTable {
    /// The table of `batches`, whose memory `memory` retains.
    fn new(
        schema: SchemaRef,
        batches: Vec<RecordBatch>,
        memory: Arc<RetainedMemory>,
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
            memory,
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

    /// The columns `projection` of the rows of block `index`, and the number
    /// of these rows, if there is such a block.
    fn scan(&self, index: usize, projection: &[usize]) -> Option<(Vec<ArrayRef>, usize)> {
        let batch = self.block(index)?;
        let columns = projection
            .iter()
            .map(|&column| Arc::clone(batch.column(column)))
            .collect();
        Some((columns, batch.num_rows()))
    }
}

impl Drop for WorkingTable {
    fn drop(&mut self) {
        // Before the buffers of the batches may be dropped.
        self.memory.release(&self.batches);
    }
}

/// The memory that the rows of a recursion retain: the rows of its last term
/// and those that the current iteration collected (or more, until the
/// tables of the terms before are dropped).
///
/// A batch retains all of its buffers, also the buffers of the arrays that
/// its columns are slices of (which may be much larger than the rows of the
/// batch, e.g. a row sliced from a batch of a table), and batches may share
/// buffers (e.g. the slices of a batch, or a column that the recursive term
/// passes on to the next term). So the memory of the buffers is reserved in
/// full, and once for all the batches that share them, and so is the memory
/// of the array objects (DataFusion's [`RecordBatchMemoryTracker`]): a
/// buffer is released once no batch of the recursion retains it any more.
/// Besides, every batch retains its own structure (see
/// [`batch_structure_size`]), which arrays without buffers (e.g. of nulls)
/// consist of alone.
#[derive(Debug)]
struct RetainedMemory {
    state: Mutex<RetainedState>,
}

#[derive(Debug)]
struct RetainedState {
    /// The buffers and the array objects of the retained batches.
    arrays: RecordBatchMemoryTracker,
    /// The sizes of the structures of the retained batches.
    structures: usize,
    /// The memory of both.
    reservation: MemoryReservation,
}

impl RetainedMemory {
    fn new(reservation: MemoryReservation) -> Self {
        Self {
            state: Mutex::new(RetainedState {
                arrays: RecordBatchMemoryTracker::new(),
                structures: 0,
                reservation,
            }),
        }
    }

    /// Retain `batch`: reserve the memory of its structure, and of its
    /// buffers and array objects that no retained batch retains yet. Fails
    /// with `ResourcesExhausted` (without retaining it) if the memory pool
    /// cannot grant it.
    fn retain(&self, batch: &RecordBatch) -> Result<()> {
        let state = &mut *self.state.lock();
        let structure = batch_structure_size(batch);
        let added = state.arrays.count_batch_with_array_overhead(batch) + structure;
        if let Err(error) = state.reservation.try_grow(added) {
            state.arrays.uncount_batch_with_array_overhead(batch);
            return Err(error);
        }
        state.structures += structure;
        Ok(())
    }

    /// The recursion does not retain `batches` any more: release the memory
    /// of their structures, and of their buffers and array objects that no
    /// other retained batch retains.
    fn release(&self, batches: &[RecordBatch]) {
        let state = &mut *self.state.lock();
        for batch in batches {
            state.arrays.uncount_batch_with_array_overhead(batch);
            state.structures =
                state.structures.saturating_sub(batch_structure_size(batch));
        }
        let retained = state.arrays.memory_usage() + state.structures;
        state.reservation.resize(retained);
    }

    /// The memory reserved for the retained batches.
    fn reserved(&self) -> usize {
        self.state.lock().reservation.size()
    }
}

/// The memory of the structure of `batch` that the CTE retains with it,
/// besides its arrays: the batch itself (in the vector of the batches of its
/// term), its vector of columns, and the blocks of the working table that
/// slice it.
fn batch_structure_size(batch: &RecordBatch) -> usize {
    let blocks = batch.num_rows().div_ceil(DEFAULT_VECTOR_SIZE).max(1);
    size_of::<RecordBatch>()
        + batch.num_columns() * size_of::<ArrayRef>()
        + blocks * size_of::<(usize, usize, usize)>()
}

/// Collects the rows of the terms of a recursion: the global state of the
/// sinks of their pipelines, shared by all threads (the [`PhysicalRecursiveCte`]
/// for the static term, and the `RecursiveTermSink` of every iteration).
///
/// Every thread collects the rows it sinks (relabeled with the schema of the
/// query), after it removed the rows that the recursion collected before
/// (for `UNION`), and reserves the memory that they retain (see
/// [`RetainedMemory`]). The rows of the threads that completed are combined,
/// until the CTE takes all of them, once the term is complete.
struct Collector {
    schema: SchemaRef,
    /// The rows of the threads that completed.
    batches: Mutex<Vec<RecordBatch>>,
    /// The memory that the rows of the recursion retain.
    memory: Arc<RetainedMemory>,
    /// For `UNION`: all rows that the recursion collected.
    distinct: Option<Mutex<DistinctRows>>,
}

impl fmt::Debug for Collector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Collector")
            .field("batches", &self.batches.lock().len())
            .field("reserved", &self.memory.reserved())
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
        let reservation = MemoryConsumer::new(consumer).register(pool);
        Ok(Self {
            schema,
            batches: Mutex::new(vec![]),
            memory: Arc::new(RetainedMemory::new(reservation)),
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
        self.memory.retain(&batch)?;
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
    /// combined them: their memory is retained until the table is dropped.
    fn take(&self) -> Arc<WorkingTable> {
        let batches = std::mem::take(&mut *self.batches.lock());
        Arc::new(WorkingTable::new(
            Arc::clone(&self.schema),
            batches,
            Arc::clone(&self.memory),
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

    /// Lower a [`PipelineWorkTableExec`] or a DataFusion `WorkTableExec` to a
    /// [`WorkTableScanSource`] with the same schema. Returns `Ok(None)` for
    /// any other node.
    ///
    /// It scans the working table of the [`PhysicalRecursiveCte`] of its name
    /// that runs it, so that it only runs in the recursive term of one (see
    /// [`PhysicalRecursiveCte::try_lower`]), as a `WorkTableExec` only runs in
    /// the recursive term of a `RecursiveQueryExec`.
    pub fn try_lower(plan: &Arc<dyn ExecutionPlan>) -> Result<Option<Self>> {
        let name = if let Some(scan) = plan.downcast_ref::<PipelineWorkTableExec>() {
            scan.cte_name()
        } else if let Some(scan) = plan.downcast_ref::<WorkTableExec>() {
            scan.name()
        } else {
            return Ok(None);
        };
        Ok(Some(Self::new(name, plan.schema())))
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
        let (table, projection) =
            scanned_work_table(&self.name, &self.schema, &ctx.task_ctx)?;
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
        let Some((columns, rows)) = state.table.scan(block, &state.projection) else {
            return Ok(SourceResult::Finished);
        };
        let chunk = if self.schema.fields().is_empty() {
            DataChunk::try_new_without_columns(Arc::clone(&self.schema), rows)?
        } else {
            DataChunk::try_new(Arc::clone(&self.schema), columns, None)?
        };
        Ok(SourceResult::Chunk(chunk))
    }
}

/// The working table that a scan of the working table of the recursive query
/// `name`, with the fields of `schema`, reads in an execution with the task
/// context `task_ctx` (see the [module documentation](self#the-working-table)),
/// and the columns of the table that it reads.
fn scanned_work_table(
    name: &str,
    schema: &Schema,
    task_ctx: &TaskContext,
) -> Result<(Arc<WorkingTable>, Vec<usize>)> {
    let Some(table) = task_ctx
        .session_config()
        .get_extension::<WorkTables>()
        .and_then(|tables| tables.table(name))
    else {
        return internal_err!(
            "A scan of the working table of {name} runs outside of the recursive term of \
             its recursive query"
        );
    };
    let Some(projection) = work_table_projection(schema, &table.schema) else {
        return internal_err!(
            "The columns of a scan of the working table of {name} are not columns of its \
             recursive query: {schema:?} vs {:?}",
            table.schema
        );
    };
    Ok((table, projection))
}

/// A scan of the working table of a [`PhysicalRecursiveCte`] as a DataFusion
/// [`ExecutionPlan`]: it takes the place of the `WorkTableExec`s of the
/// recursive term of the CTE (see [`PhysicalRecursiveCte::try_lower`]), so
/// that the nodes of the recursive term that DataFusion executes (in the plans
/// of `BridgeOperator`s, e.g. a node that no native operator implements, or
/// that the planner leaves to DataFusion in sessions with a memory limit) read
/// the working table of the CTE too, rather than the one of DataFusion's
/// `RecursiveQueryExec`, which the CTE does not fill.
///
/// The planner lowers it to a [`WorkTableScanSource`]. Where DataFusion
/// executes it, it streams the rows of the working table that a
/// [`WorkTableScanSource`] with its schema scans, which it finds in the task
/// context of its execution in the same way (a `BridgeOperator` executes its
/// plan with the task context of its pipeline).
#[derive(Debug)]
pub struct PipelineWorkTableExec {
    /// The name of the CTE.
    name: String,
    properties: Arc<PlanProperties>,
}

impl PipelineWorkTableExec {
    /// A scan of the working table of the recursive query `name`, of the
    /// columns of the fields of `schema`: like a `WorkTableExec` with schema
    /// `schema`.
    pub fn new(name: impl Into<String>, schema: SchemaRef) -> Self {
        let properties = PlanProperties::new(
            EquivalenceProperties::new(schema),
            Partitioning::UnknownPartitioning(1),
            EmissionType::Incremental,
            Boundedness::Bounded,
        )
        .with_scheduling_type(SchedulingType::Cooperative);
        Self {
            name: name.into(),
            properties: Arc::new(properties),
        }
    }

    /// The name of the recursive query whose working table it scans.
    pub fn cte_name(&self) -> &str {
        &self.name
    }
}

impl DisplayAs for PipelineWorkTableExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(f, "PipelineWorkTableExec: name={}", self.name)
            }
            DisplayFormatType::TreeRender => write!(f, "name={}", self.name),
        }
    }
}

impl ExecutionPlan for PipelineWorkTableExec {
    fn name(&self) -> &str {
        "PipelineWorkTableExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }

    fn replace_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
        _options: ReplaceChildrenOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if !children.is_empty() {
            return internal_err!("PipelineWorkTableExec does not have children");
        }
        Ok(self)
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.replace_children(
            children,
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        if partition != 0 {
            return internal_err!(
                "PipelineWorkTableExec has a single partition, got partition {partition}"
            );
        }
        let schema = self.schema();
        let (table, projection) = scanned_work_table(&self.name, &schema, &context)?;
        let batch_schema = Arc::clone(&schema);
        let batches = (0..table.blocks.len()).map(move |block| {
            let (columns, rows) = table
                .scan(block, &projection)
                .ok_or_else(|| internal_datafusion_err!("No block {block}"))?;
            let options = RecordBatchOptions::new().with_row_count(Some(rows));
            Ok(RecordBatch::try_new_with_options(
                Arc::clone(&batch_schema),
                columns,
                &options,
            )?)
        });
        let stream =
            RecordBatchStreamAdapter::new(schema, futures::stream::iter(batches));
        Ok(Box::pin(cooperative(stream)))
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
    use datafusion::functions_aggregate::min_max::max_udaf;
    use datafusion::physical_planner::DefaultPhysicalPlanner;
    use datafusion::prelude::{SessionConfig, SessionContext};
    use datafusion_common::DataFusionError;
    use datafusion_common::config::Dialect;
    use datafusion_common_runtime::SpawnedTask;
    use datafusion_datasource::memory::MemorySourceConfig;
    use datafusion_execution::memory_pool::GreedyMemoryPool;
    use datafusion_physical_expr::PhysicalSortExpr;
    use datafusion_physical_expr::aggregate::AggregateExprBuilder;
    use datafusion_physical_expr::expressions::col;
    use datafusion_physical_plan::aggregates::{AggregateMode, PhysicalGroupBy};
    use datafusion_physical_plan::collect;
    use datafusion_physical_plan::joins::{CrossJoinExec, NestedLoopJoinExec};
    use datafusion_physical_plan::placeholder_row::PlaceholderRowExec;
    use datafusion_physical_plan::union::UnionExec;
    use datafusion_physical_plan::windows::{BoundedWindowAggExec, WindowAggExec};
    use futures::StreamExt;

    use crate::PipelineQueryPlanner;
    use crate::bridge::{BridgeOperator, PipelineRootExec, bridged_plans};
    use crate::explain::{format_operator_tree, format_pipelines};
    use crate::operators::union::test_support::count;
    use crate::planner::lower_execution_plan;
    use crate::test_utils::{
        EngineOutput, compare_outputs, datafusion_context, pipeline_context, run_sql,
        test_session_config,
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
    /// the working table of one (a `WorkTableExec`, unlike the
    /// [`PipelineWorkTableExec`]s, which scan the working tables of native
    /// CTEs).
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
    /// neither a recursive query nor a scan of the working table of one (but
    /// it may execute scans of the working tables of the native CTEs, see
    /// [`PipelineWorkTableExec`]).
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
    /// [`assert_same_results_in`]).
    async fn assert_native_recursion<F>(
        sql: &str,
        config: SessionConfig,
        setup: F,
    ) -> Arc<dyn PhysicalOperator>
    where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        let datafusion = datafusion_context(config.clone());
        assert_native_recursion_in(sql, datafusion, pipeline_context(config), setup).await
    }

    /// [`assert_native_recursion`], with the session `datafusion` of
    /// DataFusion and the session `pipeline` of the pipeline engine.
    async fn assert_native_recursion_in<F>(
        sql: &str,
        datafusion: SessionContext,
        pipeline: SessionContext,
        setup: F,
    ) -> Arc<dyn PhysicalOperator>
    where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        let (_, actual) = assert_same_results_in(sql, datafusion, pipeline, setup).await;
        let root = lowered(&actual);
        assert_recursion_is_native(&root, sql);
        root
    }

    /// Assert that `sql` returns the same results in the session `datafusion`
    /// of DataFusion and in the session `pipeline` of the pipeline engine,
    /// both set up with `setup`, and return their outputs.
    ///
    /// The results are compared in canonical form (see
    /// [`compare_outputs`]): the results of the queries are unambiguous.
    async fn assert_same_results_in<F>(
        sql: &str,
        datafusion: SessionContext,
        pipeline: SessionContext,
        setup: F,
    ) -> (EngineOutput, EngineOutput)
    where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        setup(&datafusion).await.expect("setup");
        setup(&pipeline).await.expect("setup");
        let outputs = with_timeout(async {
            Ok::<_, DataFusionError>((
                run_sql(&datafusion, sql).await?,
                run_sql(&pipeline, sql).await?,
            ))
        })
        .await;
        let (expected, actual) = outputs.unwrap_or_else(|e| {
            panic!("Failed to run query on both engines: {e}\nquery: {sql}")
        });
        if let Err(message) = compare_outputs(&expected, &actual) {
            panic!(
                "Results differ: {message}\nquery: {sql}\nlowered: {}",
                format_operator_tree(&lowered(&actual))
            );
        }
        (expected, actual)
    }

    /// The directory of the data files of the recursive queries of
    /// DataFusion's `cte.slt`.
    fn data_dir() -> String {
        format!(
            "{}/../core/tests/data/recursive_cte",
            env!("CARGO_MANIFEST_DIR")
        )
    }

    /// Execute the statement `sql` in `ctx`.
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

    /// The recursive queries of `cte_materialized.slt` (whose results do not
    /// depend on the values of `random()`), and whether the recursion is
    /// native: DataFusion executes the `MaterializedCteExec`s (which the
    /// planner does not lower yet) with their whole subtrees, so the
    /// recursions in their bodies and continuations are only compared.
    const CTE_MATERIALIZED_SLT_QUERIES: &[(&str, bool)] = &[
        // a recursive CTE that is also materialized
        (
            "WITH RECURSIVE r AS MATERIALIZED ( \
               SELECT 1 AS n UNION ALL SELECT n + 1 FROM r WHERE n < 3) \
             SELECT count(*) FROM r a, r b",
            false,
        ),
        // a recursive CTE that reads a materialized CTE of the same block
        (
            "WITH RECURSIVE m AS MATERIALIZED (SELECT 3 AS lim), \
               r AS (SELECT 1 AS n UNION ALL SELECT n + 1 FROM r, m WHERE n < m.lim) \
             SELECT n FROM r",
            false,
        ),
        // a volatile CTE declared outside the recursive CTE, read by its
        // recursive term: evaluated once if materialized, else in every
        // iteration
        (
            "WITH RECURSIVE m AS MATERIALIZED (SELECT random() AS x), \
               r AS (SELECT 1 AS n, x FROM m \
                 UNION ALL SELECT n + 1, m.x FROM r, m WHERE n < 4) \
             SELECT count(DISTINCT x) FROM r",
            false,
        ),
        (
            "WITH RECURSIVE m AS NOT MATERIALIZED (SELECT random() AS x), \
               r AS (SELECT 1 AS n, x FROM m \
                 UNION ALL SELECT n + 1, m.x FROM r, m WHERE n < 4) \
             SELECT count(DISTINCT x) FROM r",
            true,
        ),
        (
            "WITH RECURSIVE m AS (SELECT random() AS x), \
               r AS (SELECT 1 AS n, x FROM m \
                 UNION ALL SELECT n + 1, m.x FROM r, m WHERE n < 4) \
             SELECT count(DISTINCT x) FROM r",
            true,
        ),
        // a volatile CTE declared inside the recursive term: evaluated again
        // in every iteration
        (
            "WITH RECURSIVE r AS ( \
               SELECT 1 AS n, CAST(0 AS DOUBLE PRECISION) AS x UNION ALL \
               SELECT n + 1, s.x FROM r, \
                 (WITH m AS MATERIALIZED (SELECT random() AS x) SELECT x FROM m) s \
               WHERE n < 4) \
             SELECT count(DISTINCT x) FROM r WHERE n > 1",
            true,
        ),
        (
            "WITH RECURSIVE r AS ( \
               SELECT 1 AS n, CAST(0 AS DOUBLE PRECISION) AS x UNION ALL \
               SELECT n + 1, s.x FROM r, \
                 (WITH m AS NOT MATERIALIZED (SELECT random() AS x) SELECT x FROM m) s \
               WHERE n < 4) \
             SELECT count(DISTINCT x) FROM r WHERE n > 1",
            true,
        ),
        (
            "WITH RECURSIVE r AS ( \
               SELECT 1 AS n, CAST(0 AS DOUBLE PRECISION) AS x UNION ALL \
               SELECT n + 1, s.x FROM r, \
                 (WITH m AS (SELECT random() AS x) SELECT x FROM m) s \
               WHERE n < 4) \
             SELECT count(DISTINCT x) FROM r WHERE n > 1",
            true,
        ),
        // two references, inside one iteration, to a CTE declared in the
        // recursive term
        (
            "WITH RECURSIVE r AS ( \
               SELECT 1 AS n, true AS same UNION ALL \
               SELECT n + 1, s.same FROM r, (WITH m AS MATERIALIZED \
                 (SELECT random() AS x) SELECT a.x = b.x AS same FROM m a, m b) s \
               WHERE n < 4) \
             SELECT n, same FROM r",
            true,
        ),
        (
            "WITH RECURSIVE r AS ( \
               SELECT 1 AS n, true AS same UNION ALL \
               SELECT n + 1, s.same FROM r, (WITH m AS NOT MATERIALIZED \
                 (SELECT random() AS x) SELECT a.x = b.x AS same FROM m a, m b) s \
               WHERE n < 4) \
             SELECT n, same FROM r",
            true,
        ),
        (
            "WITH RECURSIVE r AS ( \
               SELECT 1 AS n, true AS same UNION ALL \
               SELECT n + 1, s.same FROM r, (WITH m AS \
                 (SELECT random() AS x) SELECT a.x = b.x AS same FROM m a, m b) s \
               WHERE n < 4) \
             SELECT n, same FROM r",
            true,
        ),
        // a CTE in the recursive term whose body reads the recursive CTE: it
        // is evaluated again in every iteration, or the query does not end
        (
            "WITH RECURSIVE r AS (SELECT 1 AS n UNION ALL SELECT n + 1 FROM \
               (WITH m AS MATERIALIZED (SELECT n FROM r) SELECT n FROM m) WHERE n < 4) \
             SELECT n FROM r",
            true,
        ),
        (
            "WITH RECURSIVE r AS (SELECT 1 AS n UNION ALL SELECT n + 1 FROM \
               (WITH m AS NOT MATERIALIZED (SELECT n FROM r) SELECT n FROM m) \
               WHERE n < 4) \
             SELECT n FROM r",
            true,
        ),
        (
            "WITH RECURSIVE r AS (SELECT 1 AS n UNION ALL SELECT n + 1 FROM \
               (WITH m AS (SELECT n FROM r) SELECT n FROM m) WHERE n < 4) \
             SELECT n FROM r",
            true,
        ),
        // a recursive CTE referenced two times, with a volatile static term:
        // evaluated once if materialized, else once per reference
        (
            "WITH RECURSIVE r AS MATERIALIZED ( \
               SELECT random() AS x, 1 AS n UNION ALL SELECT x, n + 1 FROM r WHERE n < 3) \
             SELECT count(DISTINCT x) FROM (SELECT x FROM r UNION ALL SELECT x FROM r)",
            false,
        ),
        (
            "WITH RECURSIVE r AS NOT MATERIALIZED ( \
               SELECT random() AS x, 1 AS n UNION ALL SELECT x, n + 1 FROM r WHERE n < 3) \
             SELECT count(DISTINCT x) FROM (SELECT x FROM r UNION ALL SELECT x FROM r)",
            true,
        ),
        (
            "WITH RECURSIVE r AS ( \
               SELECT random() AS x, 1 AS n UNION ALL SELECT x, n + 1 FROM r WHERE n < 3) \
             SELECT count(DISTINCT x) FROM (SELECT x FROM r UNION ALL SELECT x FROM r)",
            true,
        ),
    ];

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn recursive_queries_of_cte_materialized_slt() {
        // sqlparser only parses `[NOT] MATERIALIZED` for the PostgreSQL
        // dialect, as `cte_materialized.slt` sets it.
        let mut config = test_session_config();
        config.options_mut().sql_parser.dialect = Dialect::PostgreSQL;
        let setup = async |_: &SessionContext| Ok(());
        for (sql, native) in CTE_MATERIALIZED_SLT_QUERIES {
            let datafusion = datafusion_context(config.clone());
            let pipeline = pipeline_context(config.clone());
            if *native {
                assert_native_recursion_in(sql, datafusion, pipeline, setup).await;
            } else {
                assert_same_results_in(sql, datafusion, pipeline, setup).await;
            }
        }
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
    async fn lowering_of_recursive_queries_built_by_hand() -> Result<()> {
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
        Ok(())
    }

    /// The numbers of [`PipelineWorkTableExec`]s and of `WorkTableExec`s in the
    /// DataFusion plan `plan`.
    fn work_table_scans_of(plan: &Arc<dyn ExecutionPlan>) -> (usize, usize) {
        let mut counts = (
            usize::from(plan.is::<PipelineWorkTableExec>()),
            usize::from(plan.is::<WorkTableExec>()),
        );
        for child in plan.children() {
            let (native, datafusion) = work_table_scans_of(child);
            counts.0 += native;
            counts.1 += datafusion;
        }
        counts
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn pipeline_work_tables_take_the_place_of_the_scans_of_the_working_table()
    -> Result<()> {
        let n = || Field::new("n", DataType::Int64, true);
        let scan = |name: &str| -> Result<Arc<dyn ExecutionPlan>> {
            Ok(Arc::new(WorkTableExec::new(
                name.to_string(),
                Arc::new(Schema::new(vec![n()])),
                None,
            )?))
        };
        // The scans of `t` in a union with a scan of `u`, and in a recursive
        // query `t`, whose recursive term reads its own working table (as SQL
        // scopes them, and as DataFusion binds them).
        let inner: Arc<dyn ExecutionPlan> = Arc::new(RecursiveQueryExec::try_new(
            "t".to_string(),
            Arc::new(Schema::new(vec![n()])),
            scan("t")?,
            scan("t")?,
            true,
        )?);
        let plan = UnionExec::try_new(vec![scan("t")?, scan("u")?, inner])?;
        assert_eq!(work_table_scans_of(&plan), (0, 4));
        let bound = with_pipeline_work_tables(&plan, "t")?;
        assert_eq!(work_table_scans_of(&bound), (2, 2));
        assert_eq!(bound.schema(), plan.schema());
        let inner = recursive_query(&bound).expect("the inner recursive query");
        let inner = inner.downcast_ref::<RecursiveQueryExec>().unwrap();
        assert!(inner.static_term().is::<PipelineWorkTableExec>());
        assert!(inner.recursive_term().is::<WorkTableExec>());
        // Plans without scans of the working table are not rebuilt.
        let unbound = with_pipeline_work_tables(&plan, "v")?;
        assert!(Arc::ptr_eq(&unbound, &plan));

        // A `PipelineWorkTableExec` is lowered to a scan of the working table.
        let schema = Arc::new(Schema::new(vec![n()]));
        let exec: Arc<dyn ExecutionPlan> =
            Arc::new(PipelineWorkTableExec::new("t", Arc::clone(&schema)));
        let source = WorkTableScanSource::try_lower(&exec)?.expect("a scan");
        assert_eq!(source.name(), "t");
        assert_eq!(source.schema(), schema);
        // It has the properties of a `WorkTableExec`.
        let datafusion = scan("t")?;
        assert_eq!(exec.schema(), datafusion.schema());
        assert_eq!(exec.output_partitioning().partition_count(), 1);
        assert_eq!(exec.boundedness(), datafusion.boundedness());
        assert_eq!(exec.pipeline_behavior(), datafusion.pipeline_behavior());
        assert!(exec.children().is_empty());
        assert_eq!(
            datafusion_physical_plan::displayable(exec.as_ref())
                .one_line()
                .to_string()
                .trim_end(),
            "PipelineWorkTableExec: name=t"
        );
        let with_children = Arc::clone(&exec).replace_children(
            vec![Arc::clone(&datafusion)],
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        );
        assert!(with_children.is_err());
        // It reads the working table of a recursive query that runs it: here
        // there is none.
        let ctx = datafusion_context(test_session_config());
        let error = exec.execute(1, ctx.task_ctx()).err().expect("an error");
        assert!(error.to_string().contains("single partition"), "{error}");
        let error = exec.execute(0, ctx.task_ctx()).err().expect("an error");
        assert!(
            error.to_string().contains("outside of the recursive term"),
            "{error}"
        );
        Ok(())
    }

    /// The recursive query of `cte.slt` (for issue 9680) whose recursive term
    /// scans its working table below a cross join.
    const CROSS_JOIN: &str = "WITH RECURSIVE recursive_cte AS (SELECT 1 as val UNION ALL ( \
           WITH sub_cte AS (SELECT 2 as val) \
           SELECT 2 as val FROM recursive_cte CROSS JOIN sub_cte \
           WHERE recursive_cte.val < 2)) \
         SELECT * FROM recursive_cte";

    /// Whether the DataFusion plan `plan` has a [`PipelineWorkTableExec`].
    fn scans_native_work_table(plan: &Arc<dyn ExecutionPlan>) -> bool {
        plan.is::<PipelineWorkTableExec>()
            || plan.children().into_iter().any(scans_native_work_table)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn the_nodes_that_datafusion_executes_read_the_working_table_of_the_cte()
    -> Result<()> {
        let ctx = datafusion_context(test_session_config());
        // Lowered for any session, cross joins are left to DataFusion (they do
        // not reserve their memory yet): DataFusion executes the cross join of
        // the recursive term, whose scan of the working table reads the
        // working table of the native CTE.
        let plan = ctx.sql(CROSS_JOIN).await?.create_physical_plan().await?;
        let root = lower_execution_plan(&plan)?;
        assert_recursion_is_native(&root, CROSS_JOIN);
        let cte = root
            .downcast_ref::<PhysicalRecursiveCte>()
            .expect("a native recursive query");
        let bridged = bridged_plans(cte.recursive_term());
        assert!(
            bridged.iter().any(scans_native_work_table),
            "{}",
            format_operator_tree(&root)
        );
        let output = with_timeout(Executor::collect(root, ctx.task_ctx(), 4)).await?;
        assert_eq!(sorted_values(&output), vec![1, 2]);
        Ok(())
    }

    /// A session of DataFusion and one of the pipeline engine with `config`;
    /// the memory pool of the latter has a limit of 1 GiB.
    fn sessions_with_a_memory_limit(
        config: SessionConfig,
    ) -> Result<(SessionContext, SessionContext)> {
        let pool = Arc::new(GreedyMemoryPool::new(1 << 30));
        Ok((
            datafusion_context(config.clone()),
            pipeline_context_with_pool(config, pool)?,
        ))
    }

    /// Whether a recursive query of the DataFusion plan `plan` is below a node
    /// that the planner leaves to DataFusion, with its whole subtree, in a
    /// session whose memory pool has a limit: a nested loop join, a cross join
    /// or a window (see `lower_execution_plan_with_runtime`).
    fn is_below_node_of_unlimited_memory(plan: &Arc<dyn ExecutionPlan>) -> bool {
        fn has_recursive_query(plan: &Arc<dyn ExecutionPlan>) -> bool {
            plan.is::<RecursiveQueryExec>()
                || plan.children().into_iter().any(has_recursive_query)
        }
        let unlimited = plan.is::<CrossJoinExec>()
            || plan.is::<NestedLoopJoinExec>()
            || plan.is::<WindowAggExec>()
            || plan.is::<BoundedWindowAggExec>();
        (unlimited && has_recursive_query(plan))
            || plan
                .children()
                .into_iter()
                .any(is_below_node_of_unlimited_memory)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn recursions_are_native_in_sessions_with_a_memory_limit() -> Result<()> {
        // In a session whose memory pool has a limit, the planner leaves the
        // windows, nested loop joins and cross joins to DataFusion (they do
        // not reserve their memory yet), with their whole subtrees: also
        // those of recursive terms that scan the working table, which read the
        // working table of the native CTE. So the recursive queries of
        // `cte.slt` are native but those below such nodes: the two inputs of
        // the cross join of `SELECT * FROM my_cte t1, my_cte`.
        let dir = tempfile::tempdir()?;
        let dir = dir.path().to_str().unwrap().to_owned();
        write_parquet_files(&dir).await?;
        let setup = async |ctx: &SessionContext| register_cte_tables(ctx, &dir).await;
        let mut native = 0;
        for sql in CTE_SLT_QUERIES {
            let (datafusion, pipeline) =
                sessions_with_a_memory_limit(test_session_config())?;
            let (expected, actual) =
                assert_same_results_in(sql, datafusion, pipeline, &setup).await;
            if !is_below_node_of_unlimited_memory(&expected.plan) {
                assert_recursion_is_native(&lowered(&actual), sql);
                native += 1;
            }
        }
        assert_eq!(native, CTE_SLT_QUERIES.len() - 1);
        // The cross join of the recursive term is left to DataFusion, and
        // reads the working table of the native CTE.
        let (datafusion, pipeline) = sessions_with_a_memory_limit(test_session_config())?;
        let root =
            assert_native_recursion_in(CROSS_JOIN, datafusion, pipeline, &setup).await;
        assert!(
            bridged_plans(&root).iter().any(scans_native_work_table),
            "{}",
            format_operator_tree(&root)
        );
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

    /// The `expression_id`s of the dynamic filters that the nodes of `plan`
    /// produce, and of those that its scans read.
    fn dynamic_filters_of(plan: &Arc<dyn ExecutionPlan>) -> Result<(Vec<u64>, Vec<u64>)> {
        let (mut produced, mut read) = (vec![], vec![]);
        plan.apply(|node| {
            produced.extend(
                node.dynamic_expressions_produced()
                    .iter()
                    .filter_map(|filter| filter.expression_id()),
            );
            if node.is::<DataSourceExec>() {
                node.apply_expressions(&mut |root| {
                    root.apply(|expr| {
                        read.extend(expr.expression_id());
                        Ok(TreeNodeRecursion::Continue)
                    })
                })?;
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        Ok((produced, read))
    }

    /// Write the Parquet file `nums.parquet` of the values 1 to 10,000 (`k`)
    /// into `dir`, and return a session of the pipeline engine with a table
    /// `nums` of it, on 1 thread, which pushes filters into Parquet scans.
    async fn nums_session(dir: &str) -> Result<SessionContext> {
        execute(
            &datafusion_context(test_session_config()),
            &format!(
                "COPY (SELECT value AS k FROM generate_series(1, 10000)) \
                 TO '{dir}/nums.parquet' STORED AS PARQUET"
            ),
        )
        .await?;
        let mut config = test_session_config().with_target_partitions(1);
        config.options_mut().execution.parquet.pushdown_filters = true;
        let ctx = pipeline_context(config);
        execute(
            &ctx,
            &format!(
                "CREATE EXTERNAL TABLE nums STORED AS PARQUET LOCATION '{dir}/nums.parquet'"
            ),
        )
        .await?;
        Ok(ctx)
    }

    /// A recursive query whose hash join sets its dynamic filter to the key of
    /// the iteration, with which the scan of its probe side (of `nums`)
    /// skips the other rows.
    const JOIN_WITH_DYNAMIC_FILTER: &str = "WITH RECURSIVE w(n) AS (SELECT 1 \
        UNION ALL SELECT t.k FROM w JOIN nums t ON t.k = w.n + 1 WHERE w.n < 50) \
        SELECT * FROM w";

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn copies_of_recursive_terms_have_dynamic_filters_of_their_own() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let ctx = nums_session(dir.path().to_str().unwrap()).await?;
        // The plan of DataFusion that the pipeline engine lowers.
        let plan = ctx
            .sql(JOIN_WITH_DYNAMIC_FILTER)
            .await?
            .create_physical_plan()
            .await?;
        let plan = Arc::clone(
            plan.downcast_ref::<PipelineRootExec>()
                .expect("a pipeline plan")
                .input(),
        );
        let query = recursive_query(&plan).expect("a recursive query");
        let term = query
            .downcast_ref::<RecursiveQueryExec>()
            .unwrap()
            .recursive_term();
        // The hash join produces a filter that the scan of its probe side
        // reads.
        let (produced, read) = dynamic_filters_of(term)?;
        assert_eq!(produced.len(), 1, "{produced:?}");
        assert_eq!(read, produced);
        // A copy produces and reads a new filter, and the plan is unchanged.
        let copy = with_own_dynamic_filters(term)?;
        let (copy_produced, copy_read) = dynamic_filters_of(&copy)?;
        assert_eq!(copy_produced.len(), 1, "{copy_produced:?}");
        assert_eq!(copy_read, copy_produced);
        assert_ne!(copy_produced, produced);
        assert_eq!(dynamic_filters_of(term)?, (produced.clone(), read));
        assert_eq!(copy.schema(), term.schema());
        // So does every copy.
        let (other_produced, _) = dynamic_filters_of(&with_own_dynamic_filters(term)?)?;
        assert!(other_produced != produced && other_produced != copy_produced);
        // A plan without dynamic filters is its own copy.
        let unfiltered = numbers_exec(1)?;
        assert!(Arc::ptr_eq(
            &with_own_dynamic_filters(&unfiltered)?,
            &unfiltered
        ));
        Ok(())
    }

    /// A node that passes the rows of its input on, but cannot be built again
    /// with another input (as a hash join with a prepared build side).
    #[derive(Debug)]
    struct Unrebuildable {
        input: Arc<dyn ExecutionPlan>,
    }

    impl DisplayAs for Unrebuildable {
        fn fmt_as(
            &self,
            _t: DisplayFormatType,
            f: &mut fmt::Formatter<'_>,
        ) -> fmt::Result {
            write!(f, "Unrebuildable")
        }
    }

    impl ExecutionPlan for Unrebuildable {
        fn name(&self) -> &str {
            "Unrebuildable"
        }

        fn properties(&self) -> &Arc<PlanProperties> {
            self.input.properties()
        }

        fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
            vec![&self.input]
        }

        fn apply_expressions(
            &self,
            _f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
        ) -> Result<TreeNodeRecursion> {
            Ok(TreeNodeRecursion::Continue)
        }

        fn replace_children(
            self: Arc<Self>,
            _children: Vec<Arc<dyn ExecutionPlan>>,
            _options: ReplaceChildrenOptions,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            internal_err!("Unrebuildable cannot be built again")
        }

        fn with_new_children(
            self: Arc<Self>,
            children: Vec<Arc<dyn ExecutionPlan>>,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            self.replace_children(
                children,
                ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
            )
        }

        fn execute(
            &self,
            partition: usize,
            context: Arc<TaskContext>,
        ) -> Result<SendableRecordBatchStream> {
            self.input.execute(partition, context)
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    #[expect(deprecated)] // `with_dynamic_filter_expr`, to set up the filter
    async fn nodes_that_cannot_be_rebuilt_keep_the_dynamic_filters_of_the_plan()
    -> Result<()> {
        // A hash join whose dynamic filter a `FilterExec` of its probe side
        // reads, below a node that cannot be built again.
        let build = numbers_exec(1)?;
        let probe = numbers_exec(1)?;
        let key = col("n", &probe.schema())?;
        let filter = Arc::new(DynamicFilterPhysicalExpr::new(
            vec![Arc::clone(&key)],
            lit(true),
        ));
        let read: Arc<dyn ExecutionPlan> =
            Arc::new(FilterExec::try_new(Arc::clone(&filter) as _, probe)?);
        let probe: Arc<dyn ExecutionPlan> = Arc::new(Unrebuildable { input: read });
        let join = HashJoinExec::try_new(
            Arc::clone(&build),
            probe,
            vec![(col("n", &build.schema())?, key)],
            None,
            &datafusion_common::JoinType::Inner,
            None,
            datafusion_physical_plan::joins::PartitionMode::CollectLeft,
            datafusion_common::NullEquality::NullEqualsNothing,
            false,
        )?
        .with_dynamic_filter_expr(Arc::clone(&filter))?;
        let join: Arc<dyn ExecutionPlan> = Arc::new(join);
        let filters_of = |plan: &Arc<dyn ExecutionPlan>| {
            let mut ids = (vec![], vec![]);
            let _ = plan.apply(|node| {
                ids.0.extend(
                    node.dynamic_expressions_produced()
                        .iter()
                        .filter_map(|filter| filter.expression_id()),
                );
                if node.is::<FilterExec>() {
                    let _ = node.apply_expressions(&mut |root| {
                        root.apply(|expr| {
                            ids.1.extend(expr.expression_id());
                            Ok(TreeNodeRecursion::Continue)
                        })
                    });
                }
                Ok(TreeNodeRecursion::Continue)
            });
            ids
        };
        let (produced, read) = filters_of(&join);
        assert_eq!(read, produced);
        // The copy of the join produces a new filter, but the `FilterExec` is
        // kept with the filter of the plan, which nothing sets any more: it
        // filters nothing out.
        let copy = with_own_dynamic_filters(&join)?;
        let (copy_produced, copy_read) = filters_of(&copy);
        assert_eq!(copy_produced.len(), 1);
        assert_ne!(copy_produced, produced);
        assert_eq!(copy_read, read);
        let ctx = datafusion_context(test_session_config());
        let output = collect(copy, ctx.task_ctx()).await?;
        assert_eq!(output.iter().map(RecordBatch::num_rows).sum::<usize>(), 3);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn copies_of_aggregations_have_bounds_of_their_own() -> Result<()> {
        // A partial `MAX(v)` over the values 0 to 99, whose dynamic filter
        // DataFusion's aggregation sets to `v > <the largest value so far>`:
        // the largest value is kept in bounds shared by its partitions.
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
        let max = AggregateExprBuilder::new(max_udaf(), vec![col("v", &schema)?])
            .schema(Arc::clone(&schema))
            .alias("max(v)")
            .build()?;
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from_iter_values(0..100))],
        )?;
        let input =
            MemorySourceConfig::try_new_exec(&[vec![batch]], Arc::clone(&schema), None)?;
        let aggregate: Arc<dyn ExecutionPlan> = Arc::new(AggregateExec::try_new(
            AggregateMode::Partial,
            PhysicalGroupBy::default(),
            vec![Arc::new(max)],
            vec![None],
            input,
            schema,
        )?);
        let filter_of = |plan: &Arc<dyn ExecutionPlan>| -> Result<String> {
            let [filter] = plan.dynamic_expressions_produced().try_into().unwrap();
            let filter = filter.downcast_ref::<DynamicFilterPhysicalExpr>().unwrap();
            Ok(filter.current()?.to_string())
        };
        let copy = with_own_dynamic_filters(&aggregate)?;
        assert_eq!(copy.schema(), aggregate.schema());
        // Every execution of the copy sets its own filter, from its own
        // bounds...
        let ctx = datafusion_context(test_session_config());
        collect(Arc::clone(&copy), ctx.task_ctx()).await?;
        assert_eq!(filter_of(&copy)?, "v@0 > 99");
        assert_eq!(filter_of(&aggregate)?, "true");
        // ...which an execution of the plan does not find set: it sets its
        // filter too.
        collect(Arc::clone(&aggregate), ctx.task_ctx()).await?;
        assert_eq!(filter_of(&aggregate)?, "v@0 > 99");
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_executions_of_a_plan_have_their_own_dynamic_filters() -> Result<()>
    {
        // The executions of one plan must not skip their rows with the keys of
        // each other's iterations.
        let dir = tempfile::tempdir()?;
        let ctx = nums_session(dir.path().to_str().unwrap()).await?;
        let sql = JOIN_WITH_DYNAMIC_FILTER;
        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        let root = plan
            .downcast_ref::<PipelineRootExec>()
            .expect("a pipeline plan");
        assert_recursion_is_native(root.operator(), sql);
        for _ in 0..4 {
            let executions = (0..8)
                .map(|_| {
                    let (plan, task_ctx) = (Arc::clone(&plan), ctx.task_ctx());
                    SpawnedTask::spawn(async move { collect(plan, task_ctx).await })
                })
                .collect::<Vec<_>>();
            for execution in executions {
                let batches = with_timeout(execution.join())
                    .await
                    .expect("the execution panicked")?;
                assert_eq!(sorted_values(&batches), (1..=50).collect::<Vec<_>>());
            }
        }
        Ok(())
    }

    /// `WITH RECURSIVE t(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM t WHERE
    /// n < (SELECT sum(k) FROM nums)) SELECT * FROM t`, with `nums.k` 1 to 10
    /// (so 55 rows), built by hand to run its scalar subquery in the
    /// recursive term: DataFusion's planner puts the `ScalarSubqueryExec`
    /// around the recursive query, and this plan around its recursive term.
    async fn scalar_subquery_in_the_recursive_term() -> Result<Arc<dyn ExecutionPlan>> {
        let ctx = datafusion_context(test_session_config());
        execute(
            &ctx,
            "CREATE TABLE nums AS SELECT value AS k FROM generate_series(1, 10)",
        )
        .await?;
        let sql = "WITH RECURSIVE t(n) AS (SELECT 1 UNION ALL SELECT n + 1 FROM t \
                   WHERE n < (SELECT sum(k) FROM nums)) SELECT * FROM t";
        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        let root = plan
            .downcast_ref::<ScalarSubqueryExec>()
            .expect("a scalar subquery around the recursive query");
        let recursion = root
            .input()
            .downcast_ref::<RecursiveQueryExec>()
            .expect("a recursive query");
        let recursive_term = Arc::new(ScalarSubqueryExec::new(
            Arc::clone(recursion.recursive_term()),
            root.subqueries().to_vec(),
            root.results().clone(),
        ));
        Ok(Arc::new(RecursiveQueryExec::try_new(
            recursion.name().to_owned(),
            recursion.schema(),
            Arc::clone(recursion.static_term()),
            recursive_term,
            recursion.is_distinct(),
        )?))
    }

    /// The containers of the results of the `ScalarSubqueryExec`s of `plan`,
    /// and of the `ScalarSubqueryExpr`s that its `FilterExec`s read.
    fn scalar_subquery_results_of(
        plan: &Arc<dyn ExecutionPlan>,
    ) -> Result<(Vec<ScalarSubqueryResults>, Vec<ScalarSubqueryResults>)> {
        let (mut execs, mut reads) = (vec![], vec![]);
        plan.apply(|node| {
            if let Some(exec) = node.downcast_ref::<ScalarSubqueryExec>() {
                execs.push(exec.results().clone());
            }
            if let Some(filter) = node.downcast_ref::<FilterExec>() {
                filter.predicate().apply(|expr| {
                    if let Some(read) = expr.downcast_ref::<ScalarSubqueryExpr>() {
                        reads.push(read.results().clone());
                    }
                    Ok(TreeNodeRecursion::Continue)
                })?;
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        Ok((execs, reads))
    }

    #[tokio::test]
    async fn copies_of_recursive_terms_have_scalar_subqueries_of_their_own() -> Result<()>
    {
        let plan = scalar_subquery_in_the_recursive_term().await?;
        let term = plan.children()[1];
        let (execs, reads) = scalar_subquery_results_of(term)?;
        let ([results], [read]) = (&execs[..], &reads[..]) else {
            panic!("a scalar subquery read by a filter");
        };
        assert!(ScalarSubqueryResults::ptr_eq(results, read));
        // Every copy has a container of its own, which its filter reads, and
        // the plan keeps its own.
        let copies = [
            with_own_scalar_subqueries(term)?,
            with_own_scalar_subqueries(term)?,
        ];
        let copies = copies.each_ref().map(|copy| {
            let (execs, reads) = scalar_subquery_results_of(copy).unwrap();
            let ([copy_results], [copy_read]) = (&execs[..], &reads[..]) else {
                panic!("a scalar subquery read by a filter");
            };
            assert!(ScalarSubqueryResults::ptr_eq(copy_results, copy_read));
            assert!(!ScalarSubqueryResults::ptr_eq(copy_results, results));
            copy_results.clone()
        });
        assert!(!ScalarSubqueryResults::ptr_eq(&copies[0], &copies[1]));
        let (execs, _) = scalar_subquery_results_of(term)?;
        assert!(ScalarSubqueryResults::ptr_eq(&execs[0], results));
        // A container that a node reads that this does not build again (here
        // the expression of a sort) is shared.
        let exec = term.downcast_ref::<ScalarSubqueryExec>().unwrap();
        let read: Arc<dyn PhysicalExpr> = Arc::new(ScalarSubqueryExpr::new(
            DataType::Int64,
            true,
            exec.subqueries()[0].index,
            results.clone(),
        ));
        let sort = Arc::new(SortExec::new(
            [PhysicalSortExpr::new_default(read)].into(),
            Arc::clone(exec.input()),
        ));
        let sorted: Arc<dyn ExecutionPlan> = Arc::new(ScalarSubqueryExec::new(
            sort,
            exec.subqueries().to_vec(),
            results.clone(),
        ));
        let (execs, _) =
            scalar_subquery_results_of(&with_own_scalar_subqueries(&sorted)?)?;
        assert!(ScalarSubqueryResults::ptr_eq(&execs[0], results));
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_executions_of_a_plan_have_scalar_subqueries_of_their_own()
    -> Result<()> {
        // DataFusion runs the hand-built plan, one execution at a time.
        let plan = scalar_subquery_in_the_recursive_term().await?;
        let ctx = datafusion_context(test_session_config());
        let expected = (1..=55).collect::<Vec<_>>();
        let batches = collect(Arc::clone(&plan), ctx.task_ctx()).await?;
        assert_eq!(sorted_values(&batches), expected);
        // The executions of one plan must not clear the results that each
        // other's iterations read.
        let state = ctx.state();
        let root = lower_execution_plan_with_runtime(
            &plan,
            state.config_options(),
            state.runtime_env(),
        )?;
        assert_recursion_is_native(&root, "a scalar subquery in the recursive term");
        let plan: Arc<dyn ExecutionPlan> =
            Arc::new(PipelineRootExec::try_new(root, plan)?);
        for _ in 0..2 {
            let executions = (0..64)
                .map(|_| {
                    let (plan, task_ctx) = (Arc::clone(&plan), ctx.task_ctx());
                    SpawnedTask::spawn(async move { collect(plan, task_ctx).await })
                })
                .collect::<Vec<_>>();
            for execution in executions {
                let batches = with_timeout(execution.join())
                    .await
                    .expect("the execution panicked")?;
                assert_eq!(sorted_values(&batches), expected);
            }
        }
        Ok(())
    }

    /// A recursive query whose recursive term computes a materialized CTE
    /// from the working table, in every iteration.
    const MATERIALIZED_CTE_IN_RECURSIVE_TERM: &str = "WITH RECURSIVE r AS ( \
           SELECT 1 AS n UNION ALL SELECT n + 1 FROM \
           (WITH m AS MATERIALIZED (SELECT n FROM r) SELECT n FROM m) WHERE n < 50) \
         SELECT n FROM r";

    #[tokio::test]
    async fn copies_of_recursive_terms_have_materialized_ctes_of_their_own() -> Result<()>
    {
        let mut config = test_session_config();
        config.options_mut().sql_parser.dialect = Dialect::PostgreSQL;
        let ctx = datafusion_context(config);
        let sql = MATERIALIZED_CTE_IN_RECURSIVE_TERM;
        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        let recursion = recursive_query(&plan).expect("a recursive query");
        let term = recursion.children()[1];
        let buffers = |plan: &Arc<dyn ExecutionPlan>| {
            let mut buffers = vec![];
            let _ = plan.apply(|node| {
                if let Some(cte) = node.downcast_ref::<MaterializedCteExec>() {
                    buffers.push((Arc::clone(cte.buffer()), materialized_cte_name(cte)));
                }
                Ok(TreeNodeRecursion::Continue)
            });
            buffers
        };
        let [(buffer, name)] = buffers(term).try_into().unwrap();
        assert_eq!(name, "m");
        // Every copy has a buffer of its own, with the id and the name of the
        // buffer of the plan, which keeps its own.
        let copies = [
            with_own_materialized_ctes(term)?,
            with_own_materialized_ctes(term)?,
        ];
        let copies = copies.each_ref().map(|copy| {
            let [(copy_buffer, copy_name)] = buffers(copy).try_into().unwrap();
            assert!(!Arc::ptr_eq(&copy_buffer, &buffer));
            assert_eq!((copy_buffer.id(), copy_name.as_str()), (buffer.id(), "m"));
            copy_buffer
        });
        assert!(!Arc::ptr_eq(&copies[0], &copies[1]));
        let [(unchanged, _)] = buffers(term).try_into().unwrap();
        assert!(Arc::ptr_eq(&unchanged, &buffer));
        // A plan without materialized CTEs is its own copy.
        let unmaterialized = recursion.children()[0];
        assert!(Arc::ptr_eq(
            &with_own_materialized_ctes(unmaterialized)?,
            unmaterialized
        ));
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_executions_of_a_plan_have_materialized_ctes_of_their_own()
    -> Result<()> {
        // The executions of one plan must not read the rows that a
        // materialized CTE of its recursive term (which DataFusion executes)
        // buffers for the iterations of the others.
        let mut config = test_session_config();
        config.options_mut().sql_parser.dialect = Dialect::PostgreSQL;
        let ctx = pipeline_context(config);
        let sql = MATERIALIZED_CTE_IN_RECURSIVE_TERM;
        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        let root = plan
            .downcast_ref::<PipelineRootExec>()
            .expect("a pipeline plan");
        assert_recursion_is_native(root.operator(), sql);
        for _ in 0..4 {
            let executions = (0..8)
                .map(|_| {
                    let (plan, task_ctx) = (Arc::clone(&plan), ctx.task_ctx());
                    SpawnedTask::spawn(async move { collect(plan, task_ctx).await })
                })
                .collect::<Vec<_>>();
            for execution in executions {
                let batches = with_timeout(execution.join())
                    .await
                    .expect("the execution panicked")?;
                assert_eq!(sorted_values(&batches), (1..=50).collect::<Vec<_>>());
            }
        }
        Ok(())
    }

    /// A recursive query whose recursive term has a partial `MIN`/`MAX`
    /// aggregation below a window. In a session with a memory limit, the
    /// planner leaves the window to DataFusion, with its subtree, so
    /// DataFusion's aggregation sets its dynamic filter (`k < min OR k >
    /// max`), which the scan of `nums` reads, from the bounds of its
    /// aggregates. Every iteration aggregates the rows `n + 250` and `n + 750`
    /// of its `n`, so the bounds of the iterations differ.
    const AGGREGATION_IN_A_WINDOW: &str = "WITH RECURSIVE w(n) AS (SELECT 1 UNION ALL \
        SELECT lo - 249 FROM (SELECT lo, hi, row_number() OVER () AS rn FROM \
          (SELECT min(t.k) AS lo, max(t.k) AS hi FROM w \
           JOIN nums t ON t.k = w.n + 250 OR t.k = w.n + 750) m) \
        WHERE hi = lo + 500 AND lo < 280 AND rn = 1) \
        SELECT * FROM w";

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concurrent_executions_of_a_plan_have_aggregation_bounds_of_their_own()
    -> Result<()> {
        // The values 1 to 1,000 in row groups of 10 rows, which the dynamic
        // filters of the aggregation prune.
        let dir = tempfile::tempdir()?;
        let dir = dir.path().to_str().unwrap();
        execute(
            &datafusion_context(test_session_config()),
            &format!(
                "COPY (SELECT value AS k FROM generate_series(1, 1000)) \
                 TO '{dir}/nums.parquet' STORED AS PARQUET \
                 OPTIONS (MAX_ROW_GROUP_SIZE 10)"
            ),
        )
        .await?;
        let mut config = test_session_config()
            .with_target_partitions(4)
            .with_batch_size(16);
        config.options_mut().execution.parquet.pushdown_filters = true;
        let pool = Arc::new(GreedyMemoryPool::new(1 << 30));
        let ctx = pipeline_context_with_pool(config, pool)?;
        execute(
            &ctx,
            &format!(
                "CREATE EXTERNAL TABLE nums STORED AS PARQUET LOCATION '{dir}/nums.parquet'"
            ),
        )
        .await?;
        let sql = AGGREGATION_IN_A_WINDOW;
        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        let root = plan
            .downcast_ref::<PipelineRootExec>()
            .expect("a pipeline plan");
        assert_recursion_is_native(root.operator(), sql);
        // DataFusion executes the window, with the partial aggregation, whose
        // filter the scan reads.
        let [window] = bridged_plans(root.operator()).try_into().unwrap();
        let mut partial_aggregations = vec![];
        window.apply(|node| {
            if let Some(aggregate) = node.downcast_ref::<AggregateExec>()
                && *aggregate.mode() == AggregateMode::Partial
                && !aggregate.dynamic_expressions_produced().is_empty()
            {
                partial_aggregations.push(Arc::clone(node));
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        let window_plan = datafusion_physical_plan::displayable(window.as_ref())
            .indent(true)
            .to_string();
        let [aggregate] = partial_aggregations.try_into().expect(&window_plan);
        let (produced, read) = dynamic_filters_of(&window)?;
        assert!(
            produced.iter().all(|id| read.contains(id)),
            "{produced:?} {read:?}"
        );
        // The copy of that aggregation that an execution runs has bounds of
        // its own: with the bounds of the plan, after DataFusion ran the copy
        // (over the values 1 to 100), the aggregation itself would find its
        // bounds set already, and leave its filter `true`.
        let input_schema = aggregate.children()[0].schema();
        let columns = input_schema
            .fields()
            .iter()
            .map(|_| Arc::new(Int64Array::from_iter_values(1..=100)) as ArrayRef)
            .collect();
        let batch = RecordBatch::try_new(Arc::clone(&input_schema), columns)?;
        let input = MemorySourceConfig::try_new_exec(&[vec![batch]], input_schema, None)?;
        let aggregate = aggregate.replace_children(
            vec![input],
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )?;
        let filter_of = |plan: &Arc<dyn ExecutionPlan>| -> Result<String> {
            let [filter] = plan.dynamic_expressions_produced().try_into().unwrap();
            let filter = filter.downcast_ref::<DynamicFilterPhysicalExpr>().unwrap();
            Ok(filter.current()?.to_string())
        };
        let copy = with_own_dynamic_filters(&aggregate)?;
        collect(Arc::clone(&copy), ctx.task_ctx()).await?;
        assert_ne!(filter_of(&copy)?, "true");
        assert_eq!(filter_of(&aggregate)?, "true");
        collect(Arc::clone(&aggregate), ctx.task_ctx()).await?;
        assert_eq!(filter_of(&aggregate)?, filter_of(&copy)?);
        // The executions of one plan must not prune their rows with the
        // bounds of each other's iterations.
        for _ in 0..4 {
            let executions = (0..8)
                .map(|_| {
                    let (plan, task_ctx) = (Arc::clone(&plan), ctx.task_ctx());
                    SpawnedTask::spawn(async move { collect(plan, task_ctx).await })
                })
                .collect::<Vec<_>>();
            for execution in executions {
                let batches = with_timeout(execution.join())
                    .await
                    .expect("the execution panicked")?;
                assert_eq!(sorted_values(&batches), (1..=30).collect::<Vec<_>>());
            }
        }
        Ok(())
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

    /// A session of the pipeline engine with `config`, whose memory pool is
    /// `pool`.
    fn pipeline_context_with_pool(
        config: SessionConfig,
        pool: Arc<dyn MemoryPool>,
    ) -> Result<SessionContext> {
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_pool(pool)
            .build_arc()?;
        let planner =
            PipelineQueryPlanner::new(Arc::new(DefaultPhysicalPlanner::default()));
        let state = SessionStateBuilder::new()
            .with_config(config)
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
            let ctx = pipeline_context_with_pool(
                test_session_config(),
                Arc::new(GreedyMemoryPool::new(256 << 10)),
            )?;
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
            let ctx = pipeline_context_with_pool(
                test_session_config(),
                Arc::new(GreedyMemoryPool::new(64 << 20)),
            )?;
            register_numbers(&ctx, 100_000)?;
            let actual = with_timeout(run_sql(&ctx, &sql)).await?;
            let ctx = datafusion_context(test_session_config());
            register_numbers(&ctx, 100_000)?;
            let expected = with_timeout(run_sql(&ctx, &sql)).await?;
            compare_outputs(&expected, &actual).map_err(DataFusionError::Internal)?;
        }
        Ok(())
    }

    /// Register the table `big` of a single batch of the values `0..rows`
    /// (`v BIGINT NOT NULL`), of which the scans slice their chunks.
    fn register_big(ctx: &SessionContext, rows: i64) -> Result<()> {
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int64Array::from_iter_values(0..rows))],
        )?;
        ctx.register_table(
            "big",
            Arc::new(MemTable::try_new(schema, vec![vec![batch]])?),
        )?;
        Ok(())
    }

    /// Run `sql` over the table `big` of `rows` rows (see [`register_big`]) on
    /// the pipeline engine, with a [`PeakPool`], and assert that it returns
    /// the rows that DataFusion returns. Returns the memory that the rows of
    /// the recursive query `t` reserved: now (after the query), and its peak.
    async fn reserved_for_big_rows(sql: &str, rows: i64) -> Result<(usize, usize)> {
        let pool = Arc::new(PeakPool::default());
        let ctx =
            pipeline_context_with_pool(test_session_config(), Arc::clone(&pool) as _)?;
        register_big(&ctx, rows)?;
        let actual = with_timeout(run_sql(&ctx, sql)).await?;
        assert_recursion_is_native(&lowered(&actual), sql);
        let ctx = datafusion_context(test_session_config());
        register_big(&ctx, rows)?;
        let expected = with_timeout(run_sql(&ctx, sql)).await?;
        compare_outputs(&expected, &actual).map_err(DataFusionError::Internal)?;
        Ok(pool.consumer("PhysicalRecursiveCte[t]"))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn recursions_reserve_the_buffers_that_their_rows_retain() -> Result<()> {
        // A batch of 1,000,000 rows of 8 bytes.
        let rows = 1_000_000;
        let batch = 8 * rows as usize;

        // A row that the limit slices from the batch retains all of it: the
        // CTE reserves the whole batch, which does not fit in 64 KiB.
        let one_row = "WITH RECURSIVE t(n) AS ((SELECT v FROM big LIMIT 1) UNION ALL \
                       SELECT n + 1 FROM t WHERE n < 3) SELECT * FROM t";
        let ctx = pipeline_context_with_pool(
            test_session_config(),
            Arc::new(GreedyMemoryPool::new(64 << 10)),
        )?;
        register_big(&ctx, rows)?;
        let plan = ctx.sql(one_row).await?.create_physical_plan().await?;
        let root = plan
            .downcast_ref::<PipelineRootExec>()
            .expect("a pipeline plan");
        assert_recursion_is_native(root.operator(), one_row);
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
        let (reserved, peak) = reserved_for_big_rows(one_row, rows).await?;
        assert!(peak >= batch && peak < batch + (64 << 10), "{peak} bytes");
        assert_eq!(reserved, 0);

        // The scan slices the chunks of the term from the same batch: they
        // share its buffer, which the CTE reserves once (besides the ~489
        // chunks themselves, see `batch_structure_size`), not 489 times.
        let all_rows = "WITH RECURSIVE t(n) AS (SELECT v FROM big UNION ALL \
                        SELECT n + 1 FROM t WHERE n < 0) SELECT count(*), sum(n) FROM t";
        let (reserved, peak) = reserved_for_big_rows(all_rows, rows).await?;
        assert!(peak >= batch && peak < batch + (1 << 20), "{peak} bytes");
        assert_eq!(reserved, 0);

        // The recursive term passes the strings of the static term (1,000 of
        // 1,000 bytes) on: the terms share their buffers, which the CTE
        // reserves once.
        let strings = "WITH RECURSIVE t(n, s) AS (SELECT v, repeat('x', 1000) FROM big \
                       UNION ALL SELECT n + 1000, s FROM t WHERE n < 5000) \
                       SELECT count(*), sum(n), sum(length(s)) FROM t";
        let (reserved, peak) = reserved_for_big_rows(strings, 1000).await?;
        let strings = 1000 * 1000;
        assert!(peak >= strings && peak < strings * 3 / 2, "{peak} bytes");
        assert_eq!(reserved, 0);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn recursions_reserve_the_batches_that_they_retain() -> Result<()> {
        // 100,000 batches of a single null, which have no buffers: the CTE
        // retains their array objects and the batches themselves (at least
        // the 40 bytes of a `RecordBatch` each, 4 MB), which do not fit in
        // 64 KiB.
        let sql = "WITH RECURSIVE t(n) AS (SELECT NULL FROM generate_series(1, 100000) \
                   UNION ALL SELECT n FROM t WHERE false) SELECT count(*) FROM t";
        let config = test_session_config().with_batch_size(1);
        let ctx = pipeline_context_with_pool(
            config.clone(),
            Arc::new(GreedyMemoryPool::new(64 << 10)),
        )?;
        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        let root = plan
            .downcast_ref::<PipelineRootExec>()
            .expect("a pipeline plan");
        assert_recursion_is_native(root.operator(), sql);
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
        // Without a limit, the CTE reserves them while it retains them.
        let pool = Arc::new(PeakPool::default());
        let ctx = pipeline_context_with_pool(config.clone(), Arc::clone(&pool) as _)?;
        let actual = with_timeout(run_sql(&ctx, sql)).await?;
        let expected = with_timeout(run_sql(&datafusion_context(config), sql)).await?;
        compare_outputs(&expected, &actual).map_err(DataFusionError::Internal)?;
        let (reserved, peak) = pool.consumer("PhysicalRecursiveCte[t]");
        assert!(peak >= 100_000 * size_of::<RecordBatch>(), "{peak} bytes");
        assert_eq!(reserved, 0);
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
            let ctx = pipeline_context_with_pool(
                test_session_config(),
                Arc::clone(&pool) as _,
            )?;
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
        let ctx =
            pipeline_context_with_pool(test_session_config(), Arc::clone(&pool) as _)?;
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
