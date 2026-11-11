// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`PhysicalHashAggregate`]: thread-local, radix-partitioned hash
//! aggregation.
//!
//! DataFusion aggregates in two stages over its partitions: an
//! `AggregateExec(Partial)` aggregates every input partition into partial
//! states, a `RepartitionExec(Hash)` (or a `CoalescePartitionsExec` without
//! `GROUP BY`) sends the partial states of every group to one partition
//! through channels, re-hashing their keys, and an
//! `AggregateExec(FinalPartitioned)` (`Final`) merges them, hashing the keys
//! once more. [`PhysicalHashAggregate::try_lower`] replaces the whole chain
//! (or a single `AggregateExec` in `Single` / `SinglePartitioned` mode) by a
//! single pipeline breaker, modelled after DuckDB's `PhysicalHashAggregate`:
//!
//! * as a **sink**, every thread aggregates the rows of its chunks into its
//!   own thread-local state, in parallel and without synchronization;
//! * `combine` hands the thread-local state to the global state, partitioned
//!   for the final aggregation;
//! * as a **source**, the threads of the pipeline that reads the aggregation
//!   finalize it in parallel and emit its result.
//!
//! There are two implementations:
//!
//! * without `GROUP BY` (`ungrouped`): every thread updates its own
//!   accumulators; `combine` merges them into the global accumulators, which
//!   `finalize` evaluates into the single output row, also for an empty
//!   input;
//! * with `GROUP BY` (`radix_ht`): every thread aggregates into its own
//!   hash table (`grouped_ht`) of bounded size; whenever the table is full,
//!   and on `combine`, its groups are scattered to 64 radix partitions by the
//!   upper bits of the hash of their keys; the source threads claim the
//!   partitions and finalize each of them on its own, merging the partial
//!   states of its groups into a new hash table.
//!
//! # Correctness
//!
//! The operator computes what DataFusion computes: it uses the expressions,
//! the accumulators ([`Accumulator`] and [`GroupsAccumulator`], created by the
//! `AggregateFunctionExpr`s of the plan) and the output schema of the
//! `AggregateExec`s it replaces, and DataFusion's own group key storage
//! (`GroupValues`), so the groups (`NULL`s, `-0.0`, `NaN`s, ...) and the
//! aggregate values are DataFusion's, including:
//!
//! * `FILTER (WHERE ...)`: the arguments of an aggregate are evaluated for
//!   the rows that pass its filter only;
//! * `DISTINCT` aggregates, by their accumulators (DataFusion plans a single
//!   `COUNT(DISTINCT x)` as two aggregations, both lowered);
//! * `GROUPING SETS`, `ROLLUP` and `CUBE`: the keys of every grouping set
//!   are aggregated, with the `__grouping_id` column, and the grouping sets
//!   without group columns (e.g. the grand total of a `ROLLUP`) produce a
//!   row for an empty input too;
//! * ordered aggregates (`ARRAY_AGG(x ORDER BY y)`, `FIRST_VALUE`,
//!   `LAST_VALUE`, ...): the threads see the rows in no particular order, so
//!   the accumulators that DataFusion allows to rely on the order of their
//!   input (with a sort below the aggregation) are replaced by accumulators
//!   that order the values themselves. Aggregates with a hard ordering
//!   requirement (e.g. `NTH_VALUE(x, n ORDER BY y)`, `STRING_AGG(x, ','
//!   ORDER BY y)`) get their rows in that order: every thread sorts its rows
//!   before it aggregates them, as DataFusion sorts every input partition;
//! * the limits of DataFusion's aggregations of `SELECT DISTINCT ... LIMIT n`
//!   and of `ORDER BY ... LIMIT n` by the group key or by a `MIN` / `MAX`
//!   aggregate: the operator produces groups among which the limit or the
//!   sort above it finds its rows.
//!
//! What depends on how the rows are split into batches, partial states,
//! partitions and threads is the engine's own, as is which error a failing
//! query raises: the order of the output rows (DataFusion's order of groups
//! is not defined either), the rounding of floating point sums, the order of
//! the values of an `ARRAY_AGG` without `ORDER BY`, which of the groups that
//! tie a top-k aggregation keeps, and which groups a soft limit keeps.
//!
//! # Dynamic filters
//!
//! While it aggregates its input, the operator tightens the dynamic filters
//! that the scans below it read, so that they skip the files, row groups and
//! rows that cannot change its result (see `dynamic_filter`): the filter of
//! the bounds of the `MIN` and `MAX` aggregates of an aggregation without
//! `GROUP BY` that DataFusion's partial `AggregateExec` produces, which the
//! threads set to their bounds as DataFusion's partitions do. Which rows the
//! scans skip depends on the threads too, and is the engine's own.
//!
//! [`Accumulator`]: datafusion_expr::Accumulator
//! [`GroupsAccumulator`]: datafusion_expr::GroupsAccumulator

mod dynamic_filter;
mod grouped_ht;
mod hash_aggregate;
mod radix_ht;
mod ungrouped;

pub use hash_aggregate::PhysicalHashAggregate;

/// Test support: a differential check of DataFusion plans whose
/// aggregations are lowered to [`PhysicalHashAggregate`]s.
///
/// Until the planner lowers plans to native operators, [`lower`] does, so
/// that the output of the native aggregation is compared with DataFusion's
/// on real query plans.
///
/// [`lower`]: test_support::lower
#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use std::collections::HashSet;

    use arrow::array::{Array, AsArray, RecordBatch, RecordBatchOptions};
    use arrow::compute::{
        SortColumn, SortOptions, concat, concat_batches, filter_record_batch,
        lexsort_to_indices, sort_to_indices, take, take_record_batch,
    };
    use arrow::datatypes::SchemaRef;
    use datafusion::prelude::{SessionConfig, SessionContext};
    use datafusion_common::{DataFusionError, Result, ScalarValue, internal_err};
    use datafusion_expr::AggregateUDF;
    use datafusion_physical_expr::{LexOrdering, PhysicalExpr};
    use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
    use datafusion_physical_plan::filter::FilterExec;
    use datafusion_physical_plan::projection::ProjectionExec;
    use datafusion_physical_plan::repartition::RepartitionExec;
    use datafusion_physical_plan::sorts::sort::SortExec;
    use datafusion_physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
    use datafusion_physical_plan::{ExecutionPlan, ExecutionPlanProperties, collect};
    use parking_lot::Mutex;

    use super::PhysicalHashAggregate;
    use super::radix_ht::TableCapacity;
    use crate::bridge::{BridgeOperator, PipelineRootExec};
    use crate::chunk::{DEFAULT_VECTOR_SIZE, DataChunk};
    use crate::explain::format_operator_tree;
    use crate::operator::{
        ExecutionContext, GlobalOperatorState, GlobalSinkState, GlobalSourceState,
        LocalSinkState, LocalSourceState, OperatorResult, OperatorState,
        PhysicalOperator, SinkResult, SourceResult, downcast_state,
    };
    use crate::operators::limit::PhysicalLimit;
    use crate::operators::placeholder_row::PhysicalPlaceholderRow;
    use crate::operators::union::PhysicalUnion;
    use crate::operators::union::test_support::count;
    use crate::test_utils::{
        EngineOutput, compare_outputs, datafusion_context, has_top_level_order_by,
        pipeline_context, register_test_tables, run_sql, test_session_config,
    };

    /// Lower `plan` with the native operators, standing in for the planner:
    ///
    /// * aggregations with [`PhysicalHashAggregate::try_lower`];
    /// * limits, unions and placeholder rows with their `try_lower`;
    /// * `CoalescePartitionsExec`s without a fetch, and `RepartitionExec`s
    ///   that do not preserve the order, are removed (the pipeline engine
    ///   merges the partitions of their input anyway);
    /// * `ProjectionExec`s, `FilterExec`s (without a fetch), `SortExec`s and
    ///   `SortPreservingMergeExec`s with [`TestProjection`]s, [`TestFilter`]s
    ///   and [`TestSort`]s, stand-ins for the native projection, filter and
    ///   sort, so that the aggregations below them are lowered too;
    /// * scans with the native sources of
    ///   [`try_create_scan_source`](crate::sources::try_create_scan_source);
    /// * every other node is bridged, with its subtree.
    pub(crate) fn lower(
        plan: &Arc<dyn ExecutionPlan>,
    ) -> Result<Arc<dyn PhysicalOperator>> {
        lower_with(plan, None)
    }

    /// Thread-local tables of at most 64 groups, which are flushed (and
    /// grown) all the time.
    pub(crate) const TINY_TABLES: TableCapacity = TableCapacity {
        initial: 8,
        maximum: 64,
    };

    /// Lower `plan` like [`lower`], with the thread-local tables of the
    /// grouped aggregations of [`TINY_TABLES`].
    pub(crate) fn lower_with_tiny_tables(
        plan: &Arc<dyn ExecutionPlan>,
    ) -> Result<Arc<dyn PhysicalOperator>> {
        lower_with(plan, Some(TINY_TABLES))
    }

    /// Lower `plan` like [`lower`], with thread-local tables of capacity
    /// `capacity` (the default if `None`).
    fn lower_with(
        plan: &Arc<dyn ExecutionPlan>,
        capacity: Option<TableCapacity>,
    ) -> Result<Arc<dyn PhysicalOperator>> {
        let lower = |plan: &Arc<dyn ExecutionPlan>| lower_with(plan, capacity);
        if let Some(aggregate) = PhysicalHashAggregate::try_lower(plan, lower)? {
            return Ok(Arc::new(match capacity {
                Some(capacity) => aggregate.with_table_capacity(capacity),
                None => aggregate,
            }));
        }
        if let Some(limit) = PhysicalLimit::try_lower(plan, lower)? {
            return Ok(Arc::new(limit));
        }
        if let Some(union) = PhysicalUnion::try_lower(plan, lower)? {
            return Ok(Arc::new(union));
        }
        if let Some(placeholder) = PhysicalPlaceholderRow::try_lower(plan)? {
            return Ok(Arc::new(placeholder));
        }
        if let Some(coalesce) = plan.downcast_ref::<CoalescePartitionsExec>()
            && coalesce.fetch().is_none()
            && coalesce.input().output_partitioning().partition_count() > 0
        {
            return lower(coalesce.input());
        }
        if let Some(repartition) = plan.downcast_ref::<RepartitionExec>()
            && !repartition.preserve_order()
            && repartition.input().output_partitioning().partition_count() > 0
        {
            return lower(repartition.input());
        }
        if let Some(projection) = plan.downcast_ref::<ProjectionExec>() {
            return Ok(Arc::new(TestProjection {
                children: vec![lower(projection.input())?],
                exprs: projection
                    .expr()
                    .iter()
                    .map(|expr| Arc::clone(&expr.expr))
                    .collect(),
                schema: projection.schema(),
            }));
        }
        if let Some(filter) = plan.downcast_ref::<FilterExec>()
            && filter.fetch().is_none()
        {
            return Ok(Arc::new(TestFilter {
                children: vec![lower(filter.input())?],
                predicate: Arc::clone(filter.predicate()),
                projection: filter.projection().as_ref().map(|p| p.to_vec()),
                schema: filter.schema(),
            }));
        }
        if let Some(sort) = plan.downcast_ref::<SortExec>() {
            return Ok(Arc::new(TestSort {
                children: vec![lower(sort.input())?],
                expr: sort.expr().clone(),
                fetch: sort.fetch(),
            }));
        }
        if let Some(merge) = plan.downcast_ref::<SortPreservingMergeExec>() {
            return Ok(Arc::new(TestSort {
                children: vec![lower(merge.input())?],
                expr: merge.expr().clone(),
                fetch: merge.fetch(),
            }));
        }
        if let Some(source) = crate::sources::try_create_scan_source(plan)? {
            return Ok(source);
        }
        Ok(Arc::new(BridgeOperator::new(Arc::clone(plan))))
    }

    /// Evaluates the expressions of a `ProjectionExec` on the compacted input
    /// chunks: a minimal stand-in for the native projection.
    #[derive(Debug)]
    pub(crate) struct TestProjection {
        children: Vec<Arc<dyn PhysicalOperator>>,
        exprs: Vec<Arc<dyn PhysicalExpr>>,
        schema: SchemaRef,
    }

    impl PhysicalOperator for TestProjection {
        fn name(&self) -> &str {
            "TestProjection"
        }

        fn schema(&self) -> SchemaRef {
            Arc::clone(&self.schema)
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
            let batch = input.compact()?;
            let columns = self
                .exprs
                .iter()
                .map(|expr| expr.evaluate(&batch)?.into_array(batch.num_rows()))
                .collect::<Result<Vec<_>>>()?;
            let options =
                RecordBatchOptions::new().with_row_count(Some(batch.num_rows()));
            *output = DataChunk::from_record_batch(RecordBatch::try_new_with_options(
                Arc::clone(&self.schema),
                columns,
                &options,
            )?);
            Ok(OperatorResult::NeedMoreInput)
        }
    }

    /// Filters the compacted input chunks with the predicate of a
    /// `FilterExec`, and applies its projection: a minimal stand-in for the
    /// native filter.
    #[derive(Debug)]
    pub(crate) struct TestFilter {
        children: Vec<Arc<dyn PhysicalOperator>>,
        predicate: Arc<dyn PhysicalExpr>,
        projection: Option<Vec<usize>>,
        schema: SchemaRef,
    }

    impl PhysicalOperator for TestFilter {
        fn name(&self) -> &str {
            "TestFilter"
        }

        fn schema(&self) -> SchemaRef {
            Arc::clone(&self.schema)
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
            let batch = input.compact()?;
            let mask = self
                .predicate
                .evaluate(&batch)?
                .into_array(batch.num_rows())?;
            let filtered = filter_record_batch(&batch, mask.as_boolean())?;
            let filtered = match &self.projection {
                Some(projection) => filtered.project(projection)?,
                None => filtered,
            };
            let options =
                RecordBatchOptions::new().with_row_count(Some(filtered.num_rows()));
            *output = DataChunk::from_record_batch(RecordBatch::try_new_with_options(
                Arc::clone(&self.schema),
                filtered.columns().to_vec(),
                &options,
            )?);
            Ok(OperatorResult::NeedMoreInput)
        }
    }

    /// Sorts its whole input by the expressions of a `SortExec` or a
    /// `SortPreservingMergeExec`, keeping the first `fetch` rows: a minimal
    /// stand-in for the native sort. (A `SortExec` that sorts every partition
    /// sorts all rows instead, which is all its consumers rely on: a merge of
    /// the sorted partitions, or an operator that requires the order.)
    #[derive(Debug)]
    pub(crate) struct TestSort {
        children: Vec<Arc<dyn PhysicalOperator>>,
        expr: LexOrdering,
        fetch: Option<usize>,
    }

    #[derive(Debug, Default)]
    struct TestSortState {
        batches: Mutex<Vec<RecordBatch>>,
        sorted: Mutex<Option<RecordBatch>>,
    }

    impl GlobalSinkState for TestSortState {}

    #[derive(Debug)]
    struct TestSortSource {
        sorted: RecordBatch,
        offset: AtomicUsize,
    }

    impl GlobalSourceState for TestSortSource {}

    impl PhysicalOperator for TestSort {
        fn name(&self) -> &str {
            "TestSort"
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
            Ok(Arc::new(TestSortState::default()))
        }

        fn sink(
            &self,
            _ctx: &ExecutionContext,
            input: &DataChunk,
            gstate: &dyn GlobalSinkState,
            _lstate: &mut dyn LocalSinkState,
        ) -> Result<SinkResult> {
            let state = downcast_state::<TestSortState>(gstate.as_any(), self.name())?;
            state.batches.lock().push(input.compact()?);
            Ok(SinkResult::NeedMoreInput)
        }

        fn finalize(
            &self,
            _ctx: &ExecutionContext,
            gstate: &dyn GlobalSinkState,
        ) -> Result<crate::operator::SinkFinalize> {
            let state = downcast_state::<TestSortState>(gstate.as_any(), self.name())?;
            let batch = concat_batches(&self.schema(), &*state.batches.lock())?;
            let columns = self
                .expr
                .iter()
                .map(|sort| {
                    Ok(SortColumn {
                        values: sort
                            .expr
                            .evaluate(&batch)?
                            .into_array(batch.num_rows())?,
                        options: Some(sort.options),
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let indices = lexsort_to_indices(&columns, self.fetch)?;
            *state.sorted.lock() = Some(take_record_batch(&batch, &indices)?);
            Ok(crate::operator::SinkFinalize::Ready)
        }

        fn is_source(&self) -> bool {
            true
        }

        fn get_global_source_state(
            &self,
            ctx: &ExecutionContext,
        ) -> Result<Arc<dyn GlobalSourceState>> {
            let Some(state) = ctx.global_sink_state(self) else {
                return internal_err!("TestSort is a source before it was a sink");
            };
            let state = downcast_state::<TestSortState>(state.as_any(), self.name())?;
            let Some(sorted) = state.sorted.lock().clone() else {
                return internal_err!("TestSort is a source before it was finalized");
            };
            Ok(Arc::new(TestSortSource {
                sorted,
                offset: AtomicUsize::new(0),
            }))
        }

        fn get_data(
            &self,
            _ctx: &ExecutionContext,
            gstate: &dyn GlobalSourceState,
            _lstate: &mut dyn LocalSourceState,
        ) -> Result<SourceResult> {
            let state = downcast_state::<TestSortSource>(gstate.as_any(), self.name())?;
            let offset = state
                .offset
                .fetch_add(DEFAULT_VECTOR_SIZE, Ordering::Relaxed);
            let rows = state.sorted.num_rows();
            if offset >= rows {
                return Ok(SourceResult::Finished);
            }
            let len = DEFAULT_VECTOR_SIZE.min(rows - offset);
            Ok(SourceResult::Chunk(DataChunk::from_record_batch(
                state.sorted.slice(offset, len),
            )))
        }
    }

    /// Assert that `sql` fails on DataFusion and on the pipeline engine, with
    /// errors that contain one of `messages` (which of several errors a query
    /// raises is each engine's own): through the `PipelineQueryPlanner` and
    /// with the plan lowered by [`lower`], in which at least one aggregation is
    /// a [`PhysicalHashAggregate`], with the default and with tiny
    /// thread-local tables. Both sessions have the configuration `config`, and
    /// `setup` registers the tables of the query.
    pub(crate) async fn assert_aggregate_fails_with_setup<F>(
        sql: &str,
        config: SessionConfig,
        setup: F,
        messages: &[&str],
    ) where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        let check = |engine: &str, error: &DataFusionError| {
            let error = error.to_string();
            assert!(
                messages.iter().any(|message| error.contains(message)),
                "{engine} fails with another error than {messages:?}: {error}\nquery: {sql}"
            );
        };
        let ctx = datafusion_context(config.clone());
        setup(&ctx).await.expect("setup");
        let error = run_sql(&ctx, sql).await.expect_err("DataFusion fails");
        check("DataFusion", &error);
        let pipeline = pipeline_context(config);
        setup(&pipeline).await.expect("setup");
        match run_sql(&pipeline, sql).await {
            Ok(_) => panic!(
                "The pipeline engine succeeds, DataFusion fails: {error}\nquery: {sql}"
            ),
            Err(e) => check("The planner", &e),
        }
        for (lower, tables) in [
            (lower as fn(&_) -> _, "default"),
            (lower_with_tiny_tables, "tiny"),
        ] {
            let plan = ctx
                .sql(sql)
                .await
                .expect("planning")
                .create_physical_plan()
                .await
                .expect("physical planning");
            let root = lower(&plan).expect("lowering");
            assert!(count::<PhysicalHashAggregate>(&root) > 0, "{sql}");
            let exec: Arc<dyn ExecutionPlan> =
                Arc::new(PipelineRootExec::try_new(root, plan).expect("root"));
            match collect(exec, ctx.task_ctx()).await {
                Ok(_) => panic!(
                    "The pipeline engine succeeds, DataFusion fails: {error}\nquery: {sql}"
                ),
                Err(e) => check(&format!("The lowering ({tables} tables)"), &e),
            }
        }
    }

    /// A function `count` that looks like DataFusion's built-in `count`: of
    /// the same name, implemented by a type of the same name (`Count`) with
    /// DataFusion's signature, so that its `Debug` output is DataFusion's, and
    /// with DataFusion's accumulators. Only the type that implements it is
    /// not DataFusion's (see `TrustedAggregateFunctions`).
    pub(crate) fn lookalike_count() -> AggregateUDF {
        AggregateUDF::from(lookalikes::Count::new())
    }

    /// A function `sum` that looks like DataFusion's built-in `sum`, like
    /// [`lookalike_count`].
    pub(crate) fn lookalike_sum() -> AggregateUDF {
        AggregateUDF::from(lookalikes::Sum::new())
    }

    /// A function `max` that looks like DataFusion's built-in `max`, like
    /// [`lookalike_count`], of the same order (`is_descending`), whose
    /// aggregates DataFusion's top-k aggregations may order their groups by.
    pub(crate) fn lookalike_max() -> AggregateUDF {
        AggregateUDF::from(lookalikes::Max::new())
    }

    /// The types of [`lookalike_count`], [`lookalike_sum`] and
    /// [`lookalike_max`].
    mod lookalikes {
        use std::sync::Arc;

        use arrow::datatypes::{DataType, FieldRef};
        use datafusion::functions_aggregate::count::count_udaf;
        use datafusion::functions_aggregate::min_max::max_udaf;
        use datafusion::functions_aggregate::sum::sum_udaf;
        use datafusion_common::{Result, ScalarValue};
        use datafusion_expr::function::{AccumulatorArgs, StateFieldsArgs};
        use datafusion_expr::utils::AggregateOrderSensitivity;
        use datafusion_expr::{
            Accumulator, AggregateUDF, AggregateUDFImpl, GroupsAccumulator, ReversedUDAF,
            SetMonotonicity, Signature,
        };

        /// A type named like the type of the built-in function `$builtin`
        /// (`$name`), with its fields (its signature), which implements the
        /// function `$function` with the accumulators of the built-in.
        macro_rules! lookalike {
            ($name:ident, $function:literal, $builtin:ident) => {
                #[derive(Debug, PartialEq, Eq, Hash)]
                pub(crate) struct $name {
                    signature: Signature,
                }

                impl $name {
                    pub(crate) fn new() -> Self {
                        Self {
                            signature: $builtin().signature().clone(),
                        }
                    }

                    fn builtin() -> Arc<AggregateUDF> {
                        $builtin()
                    }
                }

                impl AggregateUDFImpl for $name {
                    fn name(&self) -> &str {
                        $function
                    }

                    fn signature(&self) -> &Signature {
                        &self.signature
                    }

                    fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
                        Self::builtin().inner().return_type(arg_types)
                    }

                    fn return_field(&self, arg_fields: &[FieldRef]) -> Result<FieldRef> {
                        Self::builtin().inner().return_field(arg_fields)
                    }

                    fn accumulator(
                        &self,
                        args: AccumulatorArgs,
                    ) -> Result<Box<dyn Accumulator>> {
                        Self::builtin().inner().accumulator(args)
                    }

                    fn state_fields(
                        &self,
                        args: StateFieldsArgs,
                    ) -> Result<Vec<FieldRef>> {
                        Self::builtin().inner().state_fields(args)
                    }

                    fn groups_accumulator_supported(
                        &self,
                        args: AccumulatorArgs,
                    ) -> bool {
                        Self::builtin().inner().groups_accumulator_supported(args)
                    }

                    fn create_groups_accumulator(
                        &self,
                        args: AccumulatorArgs,
                    ) -> Result<Box<dyn GroupsAccumulator>> {
                        Self::builtin().inner().create_groups_accumulator(args)
                    }

                    fn reverse_expr(&self) -> ReversedUDAF {
                        ReversedUDAF::Identical
                    }

                    fn order_sensitivity(&self) -> AggregateOrderSensitivity {
                        Self::builtin().inner().order_sensitivity()
                    }

                    fn default_value(&self, data_type: &DataType) -> Result<ScalarValue> {
                        Self::builtin().inner().default_value(data_type)
                    }

                    fn set_monotonicity(&self, data_type: &DataType) -> SetMonotonicity {
                        Self::builtin().inner().set_monotonicity(data_type)
                    }

                    fn coerce_types(
                        &self,
                        arg_types: &[DataType],
                    ) -> Result<Vec<DataType>> {
                        Self::builtin().inner().coerce_types(arg_types)
                    }

                    fn is_descending(&self) -> Option<bool> {
                        Self::builtin().inner().is_descending()
                    }
                }
            };
        }

        lookalike!(Count, "count", count_udaf);
        lookalike!(Sum, "sum", sum_udaf);
        lookalike!(Max, "max", max_udaf);
    }

    /// Assert that `sql` produces byte-identical results on DataFusion and on
    /// the pipeline engine, both through the `PipelineQueryPlanner` and with
    /// the plan lowered by [`lower`], in which at least one aggregation is a
    /// [`PhysicalHashAggregate`]. The query can use the tables of
    /// [`register_test_tables`]. Returns the lowered operator tree.
    pub(crate) async fn assert_aggregate_parity(sql: &str) -> Arc<dyn PhysicalOperator> {
        assert_aggregate_parity_with_config(sql, test_session_config()).await
    }

    /// [`assert_aggregate_parity`] with a custom session configuration.
    pub(crate) async fn assert_aggregate_parity_with_config(
        sql: &str,
        config: SessionConfig,
    ) -> Arc<dyn PhysicalOperator> {
        assert_aggregate_parity_with_setup(sql, config, async |ctx: &SessionContext| {
            register_test_tables(ctx)
        })
        .await
    }

    /// [`assert_aggregate_parity`] with a custom session configuration, and
    /// `setup` registering the tables of the query.
    pub(crate) async fn assert_aggregate_parity_with_setup<F>(
        sql: &str,
        config: SessionConfig,
        setup: F,
    ) -> Arc<dyn PhysicalOperator>
    where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        let root = assert_lowered_parity_with_setup(sql, config, setup).await;
        assert!(
            count::<PhysicalHashAggregate>(&root) > 0,
            "No aggregation was lowered\nquery: {sql}\nlowered: {}",
            format_operator_tree(&root)
        );
        root
    }

    /// Like [`assert_aggregate_parity_with_setup`], but without requiring
    /// that an aggregation is lowered, e.g. for the aggregations that
    /// [`PhysicalHashAggregate::try_lower`] leaves to DataFusion. Returns the
    /// lowered operator tree.
    pub(crate) async fn assert_lowered_parity_with_setup<F>(
        sql: &str,
        config: SessionConfig,
        setup: F,
    ) -> Arc<dyn PhysicalOperator>
    where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        assert_lowered_results_with_setup(sql, config.clone(), config, setup).await
    }

    /// `config` without DataFusion's limited aggregations (`lim=[n]`, see
    /// [`PhysicalHashAggregate::try_lower`]): DataFusion then produces all
    /// groups of its aggregations, and its sorts and limits above them select
    /// their rows, the results that the native limited aggregations are
    /// checked against.
    pub(crate) fn without_limited_aggregations(config: SessionConfig) -> SessionConfig {
        config
            .set_bool("datafusion.optimizer.enable_topk_aggregation", false)
            .set_bool(
                "datafusion.optimizer.enable_distinct_aggregation_soft_limit",
                false,
            )
    }

    /// Assert that `sql` produces the results of DataFusion in a session of
    /// configuration `reference` (e.g. `config` or
    /// [`without_limited_aggregations`]) on the pipeline engine in a session
    /// of configuration `config`: through the `PipelineQueryPlanner` and with
    /// the plan lowered by [`lower`], with the default and with tiny
    /// thread-local tables. `setup` registers the tables of the query. Returns
    /// the lowered operator tree.
    pub(crate) async fn assert_lowered_results_with_setup<F>(
        sql: &str,
        config: SessionConfig,
        reference: SessionConfig,
        setup: F,
    ) -> Arc<dyn PhysicalOperator>
    where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        let ctx = datafusion_context(reference);
        setup(&ctx).await.expect("setup");
        let expected = run_sql(&ctx, sql).await.expect("DataFusion");
        let mut lowered = None;
        for (engine, actual, root) in lowered_outputs(sql, config, &setup).await {
            if let Err(message) = compare_outputs(&expected, &actual) {
                panic!(
                    "Results differ ({engine}): {message}\nquery: {sql}\nlowered: {}",
                    format_operator_tree(&root)
                );
            }
            if engine != "planner" {
                lowered.get_or_insert(root);
            }
        }
        lowered.expect("a lowered plan")
    }

    /// The outputs of `sql` on the pipeline engine in sessions of
    /// configuration `config` set up by `setup`: through the
    /// `PipelineQueryPlanner` (`planner`), and with the plan lowered by
    /// [`lower`], with the default and with tiny thread-local tables
    /// (`default` and `tiny`), with their lowered operator trees.
    pub(crate) async fn lowered_outputs<F>(
        sql: &str,
        config: SessionConfig,
        setup: F,
    ) -> Vec<(&'static str, EngineOutput, Arc<dyn PhysicalOperator>)>
    where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        let mut outputs = vec![];
        let pipeline = pipeline_context(config.clone());
        setup(&pipeline).await.expect("setup");
        let planned = run_sql(&pipeline, sql)
            .await
            .unwrap_or_else(|e| panic!("The planner fails: {e}\nquery: {sql}"));
        let Some(root) = planned.plan.downcast_ref::<PipelineRootExec>() else {
            panic!("The pipeline engine did not plan the query\nquery: {sql}");
        };
        let root = Arc::clone(root.operator());
        outputs.push(("planner", planned, root));
        outputs.extend(test_lowering_outputs(sql, config, setup).await);
        outputs
    }

    /// The outputs of `sql` with the plan lowered by [`lower`], with the
    /// default and with tiny thread-local tables, in sessions of
    /// configuration `config` set up by `setup`, see [`lowered_outputs`].
    pub(crate) async fn test_lowering_outputs<F>(
        sql: &str,
        config: SessionConfig,
        setup: F,
    ) -> Vec<(&'static str, EngineOutput, Arc<dyn PhysicalOperator>)>
    where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        let mut outputs = vec![];
        let ctx = datafusion_context(config);
        setup(&ctx).await.expect("setup");
        for (lower, tables) in [
            (lower as fn(&_) -> _, "default"),
            (lower_with_tiny_tables, "tiny"),
        ] {
            // A new plan: executing a plan changes the state of some nodes.
            let df = ctx.sql(sql).await.expect("planning");
            let ordered = has_top_level_order_by(df.logical_plan());
            let plan = df.create_physical_plan().await.expect("physical planning");
            let root = lower(&plan).expect("lowering");
            let exec: Arc<dyn ExecutionPlan> = Arc::new(
                PipelineRootExec::try_new(Arc::clone(&root), plan).expect("root"),
            );
            let batches = collect(Arc::clone(&exec), ctx.task_ctx())
                .await
                .unwrap_or_else(|e| {
                    panic!(
                        "Failed to run the lowered plan ({tables} tables): {e}\n\
                         query: {sql}\nlowered: {}",
                        format_operator_tree(&root)
                    )
                });
            let output = EngineOutput {
                schema: exec.schema(),
                plan: exec,
                batches,
                ordered,
            };
            outputs.push((tables, output, root));
        }
        outputs
    }

    /// The rows of `batches`.
    pub(crate) fn rows_of(batches: &[RecordBatch]) -> Vec<Vec<ScalarValue>> {
        batches
            .iter()
            .flat_map(|batch| {
                (0..batch.num_rows()).map(move |row| {
                    batch
                        .columns()
                        .iter()
                        .map(|column| ScalarValue::try_from_array(column, row))
                        .collect::<Result<Vec<_>>>()
                        .expect("a row of scalars")
                })
            })
            .collect()
    }

    /// Assert that `actual`, the rows of a query `... ORDER BY v [DESC] LIMIT
    /// limit` whose rows may tie in `v` (their column `value`, `NULL`s last),
    /// are rows of `all`, the rows of the query without its `ORDER BY` and
    /// `LIMIT`, whose values of `v` are the `limit` best ones of `all`, in
    /// their order: which of the rows that tie with the last one the query
    /// returns is the engine's own.
    pub(crate) fn assert_top_rows(
        actual: &[RecordBatch],
        all: &[RecordBatch],
        value: usize,
        descending: bool,
        limit: usize,
        context: &str,
    ) {
        let values = |batches: &[RecordBatch]| {
            let arrays = batches
                .iter()
                .map(|batch| batch.column(value).as_ref())
                .collect::<Vec<_>>();
            concat(&arrays).expect("the values of the rows")
        };
        let all_values = values(all);
        let options = SortOptions::new(descending, false);
        let best = sort_to_indices(&all_values, Some(options), Some(limit))
            .expect("the best values");
        let best = take(&all_values, &best, None).expect("the best values");
        let actual_values = values(actual);
        assert_eq!(
            actual_values.to_data(),
            best.to_data(),
            "Not the best values (in their order): {context}"
        );
        let all = rows_of(all).into_iter().collect::<HashSet<_>>();
        for row in rows_of(actual) {
            assert!(
                all.contains(&row),
                "Not a row of the query: {row:?}, {context}"
            );
        }
    }

    /// Assert that `actual`, the rows of a query `... LIMIT limit` without an
    /// `ORDER BY`, of distinct rows (e.g. `SELECT DISTINCT ...`), are `limit`
    /// distinct rows (or all of them) of `all`, the rows of the query without
    /// its `LIMIT`: which rows the query returns is the engine's own.
    pub(crate) fn assert_any_rows(
        actual: &[RecordBatch],
        all: &[RecordBatch],
        limit: usize,
        context: &str,
    ) {
        let all = rows_of(all).into_iter().collect::<HashSet<_>>();
        let actual = rows_of(actual);
        assert_eq!(actual.len(), limit.min(all.len()), "{context}");
        let distinct = actual.iter().collect::<HashSet<_>>();
        assert_eq!(distinct.len(), actual.len(), "Rows twice: {context}");
        for row in &actual {
            assert!(
                all.contains(row),
                "Not a row of the query: {row:?}, {context}"
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{
        ArrayRef, DictionaryArray, Float64Array, Int32Array, Int64Array, ListArray,
        RecordBatch, StringArray,
    };
    use arrow::datatypes::{DataType, Field, Int32Type, Schema};
    use datafusion::datasource::MemTable;
    use datafusion::prelude::SessionContext;
    use datafusion_common::Result;

    use datafusion::prelude::ParquetReadOptions;
    use datafusion_physical_plan::ExecutionPlan;
    use datafusion_physical_plan::aggregates::{AggregateExec, AggregateMode};

    use super::PhysicalHashAggregate;
    use super::test_support::*;
    use crate::operators::union::PhysicalUnion;
    use crate::operators::union::test_support::count;
    use crate::sources::test_harness::write_parquet;
    use crate::test_utils::{
        datafusion_context, register_test_tables, test_session_config, test_table_t1,
    };

    async fn assert_all_parity(queries: &[&str]) {
        for sql in queries {
            assert_aggregate_parity(sql).await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_ungrouped_aggregates() {
        assert_all_parity(&[
            "SELECT COUNT(*), COUNT(a), SUM(b), MIN(c), MAX(c), MIN(d), MAX(d), AVG(g) FROM t1",
            "SELECT SUM(a), MIN(a), MAX(b), AVG(a), COUNT(DISTINCT e) FROM t1",
            "SELECT MIN(e), MAX(e), MIN(f), MAX(f), MIN(ts), MAX(ts), MIN(h), MAX(h) FROM t1",
            "SELECT SUM(g), MIN(g), MAX(g), AVG(g), COUNT(g) FROM t1 WHERE a > 0",
            "SELECT MEDIAN(a), MEDIAN(g), APPROX_DISTINCT(b), APPROX_DISTINCT(d) FROM t1",
            "SELECT BOOL_AND(e), BOOL_OR(e), BIT_AND(a), BIT_OR(b), BIT_XOR(a) FROM t1",
            "SELECT COUNT(*) + 1 AS n, MAX(a) - MIN(a) AS spread FROM t1",
            "SELECT COUNT(*), SUM(w), MIN(name) FROM t2",
        ])
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_ungrouped_aggregates_of_empty_input() {
        assert_all_parity(&[
            "SELECT COUNT(*), COUNT(a), SUM(a), MIN(d), MAX(c), AVG(g) FROM t_empty",
            "SELECT MEDIAN(a), APPROX_DISTINCT(b), BOOL_AND(e) FROM t_empty",
            "SELECT ARRAY_AGG(id ORDER BY id), FIRST_VALUE(a ORDER BY id) FROM t_empty",
            // Inputs that are empty after a filter.
            "SELECT COUNT(*), SUM(a), MIN(h), AVG(g) FROM t1 WHERE id < 0",
            "SELECT COUNT(DISTINCT a), SUM(DISTINCT b) FROM t1 WHERE id > 10000",
            "SELECT COUNT(a) FILTER (WHERE a > 1000), SUM(b) FILTER (WHERE false) FROM t1",
        ])
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_grouped_by_one_column() {
        assert_all_parity(&[
            // Primitives, with nulls.
            "SELECT a, COUNT(*), SUM(b), MIN(c), MAX(c), AVG(g) FROM t1 GROUP BY a",
            "SELECT b, COUNT(*), MIN(id) FROM t1 GROUP BY b",
            "SELECT e, COUNT(*), SUM(a), MAX(d) FROM t1 GROUP BY e",
            "SELECT f, COUNT(*), MAX(ts) FROM t1 GROUP BY f",
            "SELECT ts, COUNT(*) FROM t1 GROUP BY ts",
            // Strings and string views.
            "SELECT d, COUNT(*) AS n, SUM(a) AS sa, MIN(h) AS mh, MAX(f) AS mf FROM t1 GROUP BY d",
            "SELECT h, COUNT(*), MAX(id), MIN(d) FROM t1 GROUP BY h",
            // Decimals.
            "SELECT g, COUNT(*), MIN(a) FROM t1 GROUP BY g",
            "SELECT d, SUM(g), AVG(g), MIN(g), MAX(g) FROM t1 GROUP BY d",
            // Floats: `-0.0` and `0.0` are one group, NaNs another.
            "SELECT c, COUNT(*), MIN(id), MAX(id) FROM t1 GROUP BY c",
            "SELECT k, COUNT(*), SUM(w), MIN(name) FROM t2 GROUP BY k",
            // Group expressions.
            "SELECT a % 7 AS m, COUNT(*), SUM(b) FROM t1 GROUP BY a % 7",
            "SELECT upper(d) AS u, COUNT(*) FROM t1 GROUP BY upper(d)",
            "SELECT floor(c) AS fc, COUNT(*) FROM t1 GROUP BY floor(c)",
            // No groups.
            "SELECT d, COUNT(*), SUM(a) FROM t_empty GROUP BY d",
            "SELECT a, COUNT(*) FROM t1 WHERE id < 0 GROUP BY a",
        ])
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_grouped_by_several_columns() {
        assert_all_parity(&[
            "SELECT d, e, COUNT(*), SUM(a) FROM t1 GROUP BY d, e",
            "SELECT d, e, f, COUNT(*), MIN(c), MAX(h) FROM t1 GROUP BY d, e, f",
            "SELECT a, c, COUNT(*) FROM t1 GROUP BY a, c",
            "SELECT g, h, e, COUNT(*), MAX(b) FROM t1 GROUP BY g, h, e",
            "SELECT DISTINCT d, e FROM t1",
            "SELECT DISTINCT a, d, e FROM t1",
            "SELECT e, f, COUNT(*) FROM t1 GROUP BY e, f HAVING COUNT(*) > 3",
        ])
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_high_cardinality_groups() {
        assert_all_parity(&[
            "SELECT id, COUNT(*), SUM(a), MIN(d) FROM t1 GROUP BY id",
            "SELECT id, a, COUNT(*) FROM t1 GROUP BY id, a",
            "SELECT id % 1000 AS k, COUNT(*), MAX(h) FROM t1 GROUP BY id % 1000",
            "SELECT DISTINCT id, h FROM t1",
        ])
        .await;
        let setup = async |ctx: &SessionContext| register_big_table(ctx);
        for (sql, thread_counts) in [
            (
                "SELECT k, COUNT(*), SUM(v), MIN(s), MAX(s) FROM big GROUP BY k",
                &[1, 4, 16][..],
            ),
            ("SELECT s, COUNT(*), SUM(v) FROM big GROUP BY s", &[4]),
            ("SELECT k, s, COUNT(*) FROM big GROUP BY k, s", &[4]),
            (
                "SELECT COUNT(*), COUNT(DISTINCT k), SUM(v), APPROX_DISTINCT(s) FROM big",
                &[1, 4, 16],
            ),
            ("SELECT f, COUNT(*) FROM big GROUP BY f", &[4]),
            ("SELECT dict, COUNT(*), MIN(k) FROM big GROUP BY dict", &[4]),
            ("SELECT dict, s, MAX(v) FROM big GROUP BY dict, s", &[4]),
            (
                "SELECT k % 7 AS m, COUNT(DISTINCT s), MEDIAN(v) FROM big GROUP BY k % 7",
                &[4],
            ),
        ] {
            for &threads in thread_counts {
                let config = test_session_config().with_target_partitions(threads);
                assert_aggregate_parity_with_setup(sql, config, &setup).await;
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_filtered_aggregates() {
        assert_all_parity(&[
            "SELECT d, COUNT(a) FILTER (WHERE b > 0) FROM t1 GROUP BY d",
            "SELECT e, SUM(b) FILTER (WHERE a < 0), COUNT(*) FILTER (WHERE d = 'beta'), \
             MIN(h) FILTER (WHERE c > 0), COUNT(*) FROM t1 GROUP BY e",
            "SELECT COUNT(*) FILTER (WHERE e), SUM(a) FILTER (WHERE a IS NULL) FROM t1",
            // A filter that is NULL for some rows.
            "SELECT d, COUNT(*) FILTER (WHERE a > 0), SUM(g) FILTER (WHERE e) FROM t1 GROUP BY d",
            // The arguments are only evaluated for the rows that pass the
            // filter: a division by zero of the other rows is no error.
            "SELECT d, SUM(1000 / a) FILTER (WHERE a <> 0) FROM t1 GROUP BY d",
            "SELECT SUM(1000 / a) FILTER (WHERE a <> 0), COUNT(*) FROM t1",
            "SELECT d, COUNT(DISTINCT a) FILTER (WHERE b > 100) FROM t1 GROUP BY d",
            "SELECT d, ARRAY_AGG(id ORDER BY id) FILTER (WHERE a > 45) FROM t1 GROUP BY d",
        ])
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_distinct_aggregates() {
        assert_all_parity(&[
            "SELECT COUNT(DISTINCT a) FROM t1",
            "SELECT COUNT(DISTINCT a), SUM(DISTINCT b), COUNT(DISTINCT d) FROM t1",
            "SELECT e, f, COUNT(DISTINCT a) AS da FROM t1 GROUP BY e, f",
            "SELECT d, COUNT(DISTINCT a), SUM(DISTINCT b), COUNT(*) FROM t1 GROUP BY d",
            "SELECT d, COUNT(DISTINCT h), AVG(DISTINCT a) FROM t1 GROUP BY d",
            "SELECT e, SUM(DISTINCT g), COUNT(DISTINCT c) FROM t1 GROUP BY e",
            "SELECT ARRAY_AGG(DISTINCT d ORDER BY d) FROM t1",
        ])
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_grouping_sets() {
        assert_all_parity(&[
            "SELECT d, SUM(g) FROM t1 GROUP BY ROLLUP (d)",
            "SELECT d, e, SUM(g), COUNT(*) FROM t1 GROUP BY ROLLUP (d, e)",
            "SELECT d, e, SUM(g), COUNT(*), MIN(h) FROM t1 GROUP BY CUBE (d, e)",
            "SELECT d, e, f, COUNT(*) FROM t1 GROUP BY CUBE (d, e, f)",
            "SELECT d, e, COUNT(*) FROM t1 GROUP BY GROUPING SETS ((d), (e), ())",
            // A repeated grouping set: the grouping ids differ by their ordinal.
            "SELECT d, e, COUNT(*) FROM t1 GROUP BY GROUPING SETS ((d), (e), (d), ())",
            "SELECT COUNT(*) FROM t1 GROUP BY GROUPING SETS ((), ())",
            "SELECT COUNT(*), SUM(a) FROM t1 GROUP BY GROUPING SETS (())",
            "SELECT GROUPING(d) AS gd, GROUPING(e) AS ge, d, e, COUNT(*) FROM t1 GROUP BY CUBE (d, e)",
            "SELECT d, e, COUNT(DISTINCT a), SUM(b) FILTER (WHERE a > 0) FROM t1 GROUP BY ROLLUP (d, e)",
            // Without input rows, the grouping sets without group columns
            // still produce a row.
            "SELECT d, COUNT(*), SUM(a) FROM t_empty GROUP BY ROLLUP (d)",
            "SELECT d, e, COUNT(*) FROM t_empty GROUP BY CUBE (d, e)",
            "SELECT d, COUNT(*) FROM t_empty GROUP BY GROUPING SETS ((d), (), ())",
            "SELECT d, COUNT(*) FROM t1 WHERE id < 0 GROUP BY ROLLUP (d)",
            "SELECT COUNT(*), MEDIAN(a) FROM t_empty GROUP BY GROUPING SETS (())",
        ])
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_ordered_aggregates() {
        assert_all_parity(&[
            "SELECT ARRAY_AGG(id ORDER BY id) FROM t1 WHERE a > 40",
            "SELECT ARRAY_AGG(a ORDER BY id DESC) FROM t1",
            "SELECT d, ARRAY_AGG(id ORDER BY id) FROM t1 GROUP BY d",
            "SELECT e, ARRAY_AGG(h ORDER BY id DESC) FROM t1 GROUP BY e",
            "SELECT d, FIRST_VALUE(id ORDER BY id), LAST_VALUE(a ORDER BY id) FROM t1 GROUP BY d",
            "SELECT FIRST_VALUE(a ORDER BY id DESC), LAST_VALUE(h ORDER BY id DESC) FROM t1",
            "SELECT e, FIRST_VALUE(d ORDER BY b DESC NULLS LAST, id) FROM t1 GROUP BY e",
            "SELECT d, STRING_AGG(h, ',' ORDER BY id) FROM t1 GROUP BY d",
            "SELECT STRING_AGG(d, '|' ORDER BY id DESC) FROM t1 WHERE id < 300",
            "SELECT e, NTH_VALUE(a, 3 ORDER BY id) FROM t1 GROUP BY e",
            "SELECT NTH_VALUE(h, 2 ORDER BY id DESC) FROM t1",
            // Several aggregates with different orders.
            "SELECT d, ARRAY_AGG(id ORDER BY id), FIRST_VALUE(id ORDER BY id DESC), \
             STRING_AGG(h, ';' ORDER BY id DESC) FROM t1 GROUP BY d",
            "SELECT d, e, ARRAY_AGG(a ORDER BY id) FROM t1 GROUP BY ROLLUP (d, e)",
        ])
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_aggregate_functions() {
        assert_all_parity(&[
            "SELECT d, SUM(a), AVG(a), MIN(a), MAX(a), COUNT(a), MEDIAN(a), \
             APPROX_DISTINCT(a) FROM t1 GROUP BY d",
            "SELECT e, SUM(b), AVG(b), MIN(b), MAX(b), COUNT(b), MEDIAN(b), \
             APPROX_DISTINCT(h) FROM t1 GROUP BY e",
            "SELECT d, MEDIAN(c), MIN(c), MAX(c), APPROX_DISTINCT(f) FROM t1 GROUP BY d",
            "SELECT d, AVG(g), MEDIAN(g), SUM(g) FROM t1 GROUP BY d",
            "SELECT e, BOOL_AND(a > 0), BOOL_OR(b > 5000), BIT_AND(a), BIT_OR(b), \
             BIT_XOR(a) FROM t1 GROUP BY e",
            "SELECT d, MIN(f), MAX(ts), MIN(e), MAX(h), COUNT(DISTINCT f) FROM t1 GROUP BY d",
        ])
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_aggregations_in_plans() {
        let root = assert_aggregate_parity(
            "SELECT k, COUNT(*), SUM(x) FROM \
             (SELECT a AS k, b AS x FROM t1 UNION ALL SELECT k, w FROM t2) GROUP BY k",
        )
        .await;
        // Several pipelines sink into the aggregation.
        assert_eq!(count::<PhysicalUnion>(&root), 1);
        assert_all_parity(&[
            // An aggregation of an aggregation.
            "SELECT n, COUNT(*) FROM (SELECT d, COUNT(*) AS n FROM t1 GROUP BY d) GROUP BY n",
            "SELECT MAX(n), SUM(n) FROM (SELECT a, COUNT(*) AS n FROM t1 GROUP BY a)",
            // HAVING.
            "SELECT a, COUNT(*) FROM t1 GROUP BY a HAVING COUNT(*) > 50",
            "SELECT d, SUM(a) FROM t1 GROUP BY d HAVING MIN(id) < 10",
            // Joins below and above.
            "SELECT t2.k, COUNT(*), SUM(t1.b) FROM t1 JOIN t2 ON t1.a = t2.k GROUP BY t2.k",
            // Sorted and limited results.
            "SELECT d, COUNT(*) AS n FROM t1 GROUP BY d ORDER BY d",
            "SELECT a, COUNT(*) AS n FROM t1 GROUP BY a ORDER BY n DESC, a LIMIT 5",
            "SELECT d, MAX(a) FROM t1 GROUP BY d ORDER BY MAX(a) DESC, d LIMIT 3",
            "SELECT COUNT(*) FROM (SELECT DISTINCT d FROM t1 LIMIT 3)",
        ])
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_aggregates_with_other_partitionings() {
        for target_partitions in [1, 3, 16] {
            let config = test_session_config().with_target_partitions(target_partitions);
            for sql in [
                "SELECT COUNT(*), SUM(a), MIN(d), AVG(g) FROM t1",
                "SELECT COUNT(*), SUM(a) FROM t_empty",
                "SELECT d, COUNT(*), SUM(a), MEDIAN(b) FROM t1 GROUP BY d",
                "SELECT e, f, COUNT(DISTINCT a) FROM t1 GROUP BY e, f",
                "SELECT d, e, COUNT(*) FROM t1 GROUP BY CUBE (d, e)",
                "SELECT d, ARRAY_AGG(id ORDER BY id) FROM t1 GROUP BY d",
                "SELECT e, NTH_VALUE(a, 2 ORDER BY id) FROM t1 GROUP BY e",
                "SELECT d, COUNT(*) FROM t_empty GROUP BY ROLLUP (d)",
            ] {
                assert_aggregate_parity_with_config(sql, config.clone()).await;
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_aggregates_with_other_plans_and_settings() {
        let settings: [&[(&str, &str)]; 5] = [
            // Partitioned joins: `SinglePartitioned` aggregations of their
            // output, which is partitioned by the group keys.
            &[
                (
                    "datafusion.optimizer.hash_join_single_partition_threshold",
                    "0",
                ),
                (
                    "datafusion.optimizer.hash_join_single_partition_threshold_rows",
                    "0",
                ),
            ],
            // `Final` aggregations of coalesced partial states.
            &[("datafusion.optimizer.repartition_aggregations", "false")],
            // DataFusion's legacy aggregation stream.
            &[("datafusion.execution.enable_migration_aggregate", "false")],
            // DataFusion skips the partial aggregation.
            &[
                (
                    "datafusion.execution.skip_partial_aggregation_probe_rows_threshold",
                    "1",
                ),
                (
                    "datafusion.execution.skip_partial_aggregation_probe_ratio_threshold",
                    "0.0",
                ),
            ],
            &[("datafusion.execution.batch_size", "100")],
        ];
        for settings in settings {
            let mut config = test_session_config();
            for (key, value) in settings {
                config = config.set_str(key, value);
            }
            for sql in [
                "SELECT t1.a, COUNT(*), SUM(t2.w), MIN(t1.d) FROM t1 JOIN t2 ON t1.a = t2.k \
                 GROUP BY t1.a",
                "SELECT d, e, COUNT(*), SUM(a), AVG(g) FROM t1 GROUP BY d, e",
                "SELECT id % 100 AS k, COUNT(*), MEDIAN(a) FROM t1 GROUP BY id % 100",
                "SELECT d, COUNT(DISTINCT a), ARRAY_AGG(id ORDER BY id) FROM t1 GROUP BY d",
                "SELECT d, e, COUNT(*) FROM t1 GROUP BY ROLLUP (d, e)",
                "SELECT COUNT(*), SUM(b) FILTER (WHERE a > 0) FROM t1",
            ] {
                assert_aggregate_parity_with_config(sql, config.clone()).await;
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_nested_and_dictionary_keys() {
        let setup = async |ctx: &SessionContext| {
            register_test_tables(ctx)?;
            register_big_table(ctx)
        };
        for sql in [
            "SELECT l, COUNT(*), SUM(k) FROM big GROUP BY l",
            "SELECT l, dict, COUNT(*) FROM big GROUP BY l, dict",
            "SELECT [a, b] AS ab, COUNT(*) FROM t1 GROUP BY [a, b]",
            "SELECT named_struct('d', d, 'e', e) AS de, COUNT(*) FROM t1 \
             GROUP BY named_struct('d', d, 'e', e)",
            "SELECT dict, ARRAY_AGG(k ORDER BY k) FROM big WHERE k < 300 GROUP BY dict",
            "SELECT dict, l, COUNT(*) FROM big GROUP BY ROLLUP (dict, l)",
        ] {
            assert_aggregate_parity_with_setup(sql, test_session_config(), &setup).await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_aggregates_of_parquet_files() -> Result<()> {
        // Three files of small row groups, scanned by the native
        // `ParquetScanSource`. DataFusion pushes a dynamic filter of the
        // ungrouped `MIN` and `MAX` aggregates into the scan, which it
        // tightens as it aggregates, and so does the native aggregation (see
        // `dynamic_filter`, whose tests show how the scans prune with it).
        let dir = tempfile::tempdir()?;
        let batches = test_table_t1()?.into_iter().flatten().collect::<Vec<_>>();
        let properties = ::parquet::file::properties::WriterProperties::builder()
            .set_max_row_group_row_count(Some(500))
            .build();
        for (i, batches) in batches.chunks(2).enumerate() {
            let path = dir.path().join(format!("part-{i}.parquet"));
            write_parquet(
                &path,
                batches[0].schema(),
                batches,
                Some(properties.clone()),
            )?;
        }
        let path = dir.path().to_str().expect("a UTF-8 path").to_string();
        let setup = async |ctx: &SessionContext| {
            ctx.register_parquet("p", &path, ParquetReadOptions::default())
                .await
        };
        for sql in [
            // (Without a filter, DataFusion answers from the statistics.)
            "SELECT MIN(a), MAX(a), MIN(b), MAX(d) FROM p WHERE e",
            "SELECT MIN(id), MAX(id), COUNT(*) FROM p WHERE a > 0",
            "SELECT d, COUNT(*), SUM(a), MIN(g), MAX(h) FROM p GROUP BY d",
            "SELECT e, f, COUNT(DISTINCT a), MEDIAN(b) FROM p GROUP BY e, f",
            "SELECT d, e, COUNT(*) FROM p GROUP BY ROLLUP (d, e)",
            "SELECT id % 100 AS k, ARRAY_AGG(id ORDER BY id) FROM p GROUP BY id % 100",
        ] {
            for target_partitions in [1, 4] {
                let config =
                    test_session_config().with_target_partitions(target_partitions);
                assert_aggregate_parity_with_setup(sql, config, &setup).await;
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn soft_limited_distinct_aggregations_are_native() {
        // `SELECT DISTINCT ... LIMIT n`: DataFusion's aggregations (with a soft
        // limit, `lim=[n]`) stop reading their input once they have `n` groups,
        // and so do the native ones, whose threads stop once one of them has
        // `n` groups. The limit above returns any `n` of them.
        let setup = async |ctx: &SessionContext| {
            register_test_tables(ctx)?;
            let schema =
                Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]));
            let batch = |values: Vec<i32>| {
                RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![Arc::new(Int32Array::from(values)) as ArrayRef],
                )
            };
            let s = vec![vec![
                batch(vec![5, 3])?,
                batch(vec![4, 4, 1])?,
                batch(vec![2])?,
            ]];
            ctx.register_table("s", Arc::new(MemTable::try_new(schema, s)?))?;
            Ok(())
        };
        for threads in [1, 4, 16] {
            let config = test_session_config().with_target_partitions(threads);
            let ctx = datafusion_context(config.clone());
            setup(&ctx).await.expect("setup");
            for (sql, all, limit) in [
                (
                    "SELECT DISTINCT a FROM s LIMIT 2",
                    "SELECT DISTINCT a FROM s",
                    2,
                ),
                (
                    "SELECT DISTINCT d, e FROM t1 LIMIT 3",
                    "SELECT DISTINCT d, e FROM t1",
                    3,
                ),
                (
                    "SELECT DISTINCT id % 1000 AS m, h FROM t1 LIMIT 10",
                    "SELECT DISTINCT id % 1000 AS m, h FROM t1",
                    10,
                ),
                // More groups than there are.
                (
                    "SELECT DISTINCT a FROM t1 LIMIT 1000",
                    "SELECT DISTINCT a FROM t1",
                    1000,
                ),
                (
                    "SELECT DISTINCT d FROM t1 WHERE id < 0 LIMIT 3",
                    "SELECT DISTINCT d FROM t1 WHERE id < 0",
                    3,
                ),
            ] {
                let plan = ctx.sql(sql).await.unwrap().create_physical_plan().await;
                let plan = plan.expect("physical planning");
                assert!(!soft_limits(&plan).is_empty(), "{sql}");
                let all = ctx.sql(all).await.unwrap().collect().await.unwrap();
                // (The planner leaves the plans below limits to DataFusion.)
                for (engine, output, root) in
                    lowered_outputs(sql, config.clone(), &setup).await
                {
                    let context = format!("{engine} on {threads} threads, query: {sql}");
                    if engine != "planner" {
                        assert_eq!(native_soft_limits(&root), [limit], "{context}");
                    }
                    assert_any_rows(&output.batches, &all, limit, &context);
                }
            }
        }
        // Aggregated again, which only counts the groups.
        assert_aggregate_parity(
            "SELECT COUNT(*) FROM (SELECT DISTINCT d FROM t1 LIMIT 3)",
        )
        .await;
        // A thread stops consuming its input once it has the groups of the
        // limit: on a single thread, the aggregation never sees (nor divides
        // by) the zero of the second batch.
        let setup = async |ctx: &SessionContext| {
            let schema =
                Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]));
            let batch = |values: Vec<i32>| {
                RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![Arc::new(Int32Array::from(values)) as ArrayRef],
                )
            };
            let t = vec![vec![batch(vec![1])?, batch(vec![0])?]];
            ctx.register_table("t", Arc::new(MemTable::try_new(schema, t)?))?;
            Ok(())
        };
        let sql = "SELECT COUNT(*) FROM (SELECT DISTINCT 1000 / a FROM t LIMIT 1)";
        let config = test_session_config().with_target_partitions(1);
        let root = assert_aggregate_parity_with_setup(sql, config, setup).await;
        assert_eq!(native_soft_limits(&root), [1]);
    }

    /// The limits of the `AggregateExec`s with a soft limit in `plan`.
    fn soft_limits(plan: &Arc<dyn ExecutionPlan>) -> Vec<usize> {
        let mut limits = vec![];
        if let Some(aggregate) = plan.downcast_ref::<AggregateExec>()
            && let Some(limit) = aggregate.limit_options()
            && limit.descending().is_none()
        {
            limits.push(limit.limit());
        }
        for child in plan.children() {
            limits.extend(soft_limits(child));
        }
        limits
    }

    /// The soft limits of the native aggregations of `root`.
    fn native_soft_limits(root: &Arc<dyn crate::PhysicalOperator>) -> Vec<usize> {
        let mut limits = vec![];
        if let Some(aggregate) = root.downcast_ref::<PhysicalHashAggregate>() {
            limits.extend(aggregate.spec().soft_limit);
        }
        for child in root.children() {
            limits.extend(native_soft_limits(child));
        }
        limits
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn top_k_aggregations_keep_only_their_top_groups() {
        // DataFusion's top-k aggregation of an `ORDER BY ... LIMIT n` by the
        // group key (`a`) or a `MIN` / `MAX` aggregate keeps only the top
        // `n` groups in every stage, so the projection between it and the
        // sort only evaluates `1000 / ...` for those, never for `0`. So does
        // the native top-k aggregation, which keeps the top groups of all of
        // its groups, on any number of threads, of keys and values of any
        // type.
        let setup = async |ctx: &SessionContext| register_test_tables(ctx);
        let distinct = "SELECT 1000 / a AS v, a FROM (VALUES (0), (1), (2), (3)) AS t(a) \
             GROUP BY a ORDER BY a DESC LIMIT 1";
        let max = "SELECT 1000 / MAX(b) AS v, a \
             FROM (VALUES (0, 0), (1, 5), (2, 7), (3, 1)) AS t(a, b) \
             GROUP BY a ORDER BY MAX(b) DESC LIMIT 1";
        let single = &[AggregateMode::Single][..];
        let coalesced = &[AggregateMode::Final, AggregateMode::Partial][..];
        let partitioned = &[AggregateMode::FinalPartitioned, AggregateMode::Partial][..];
        for (sql, stages, native_aggregations) in [
            // A single stage on 1 thread, and two stages that are fused on 4.
            (distinct, [single, partitioned], 1),
            (max, [single, partitioned], 1),
            // `t1` has 4 partitions. Its `d` is a `Utf8` string, its `h` a
            // `Utf8View`.
            (
                "SELECT DISTINCT d FROM t1 ORDER BY d LIMIT 3",
                [coalesced, partitioned],
                1,
            ),
            (
                "SELECT DISTINCT h FROM t1 ORDER BY h LIMIT 3",
                [coalesced, partitioned],
                1,
            ),
            // An aggregation of the result of a top-k aggregation.
            (
                "SELECT COUNT(*), SUM(v) FROM (SELECT 1000 / a AS v, a \
                 FROM (VALUES (0), (1), (2), (3)) AS t(a) GROUP BY a ORDER BY a DESC LIMIT 2)",
                [single, partitioned],
                2,
            ),
        ] {
            for (threads, stages) in [1, 4].into_iter().zip(stages) {
                let config = test_session_config().with_target_partitions(threads);
                let ctx = datafusion_context(config.clone());
                setup(&ctx).await.expect("setup");
                let plan = ctx
                    .sql(sql)
                    .await
                    .expect("planning")
                    .create_physical_plan()
                    .await
                    .expect("physical planning");
                assert_eq!(top_k_aggregations(&plan), stages, "{sql}");
                let root = assert_lowered_parity_with_setup(sql, config, &setup).await;
                assert_eq!(
                    count::<PhysicalHashAggregate>(&root),
                    native_aggregations,
                    "{sql}"
                );
            }
        }
    }

    /// The modes of the `AggregateExec`s with a top-k limit in `plan`, from
    /// the top.
    fn top_k_aggregations(plan: &Arc<dyn ExecutionPlan>) -> Vec<AggregateMode> {
        let mut modes = vec![];
        if let Some(aggregate) = plan.downcast_ref::<AggregateExec>()
            && aggregate
                .limit_options()
                .is_some_and(|limit| limit.descending().is_some())
        {
            modes.push(*aggregate.mode());
        }
        for child in plan.children() {
            modes.extend(top_k_aggregations(child));
        }
        modes
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn grouping_ids_of_many_columns() {
        // `__grouping_id` holds a bit per group column and the ordinal of a
        // repeated grouping set, in at most 64 bits. Without input rows, the
        // native aggregation computes it for the grouping sets without
        // columns itself.
        let keys = |n: usize| {
            (0..n)
                .map(|i| format!("k{i}"))
                .collect::<Vec<_>>()
                .join(", ")
        };
        for rows in [0, 3] {
            let setup = async |ctx: &SessionContext| register_wide_table(ctx, rows);
            let config = test_session_config();
            for (columns, sets, error) in [
                // 64 columns and the ordinal 1 need 65 bits.
                (64, "(), ()", Some("require 65 bits, which exceeds 64")),
                (65, "()", Some("Grouping sets with more than 64 columns")),
                // At the boundary.
                (63, "(), ()", None),
                (64, "()", None),
            ] {
                let sql = format!(
                    "SELECT COUNT(*), SUM(k0) FROM wide GROUP BY GROUPING SETS (({}), {sets})",
                    keys(columns)
                );
                match error {
                    Some(error) => {
                        assert_aggregate_fails_with_setup(
                            &sql,
                            config.clone(),
                            &setup,
                            &[error],
                        )
                        .await
                    }
                    None => {
                        assert_aggregate_parity_with_setup(&sql, config.clone(), &setup)
                            .await;
                    }
                }
            }
        }
    }

    /// Register table `wide`: `rows` rows of 65 integer columns `k0` to
    /// `k64`.
    fn register_wide_table(ctx: &SessionContext, rows: i32) -> Result<()> {
        let schema = Arc::new(Schema::new(
            (0..65)
                .map(|i| Field::new(format!("k{i}"), DataType::Int32, true))
                .collect::<Vec<_>>(),
        ));
        let columns = (0..65)
            .map(|i| {
                Arc::new(Int32Array::from_iter_values((0..rows).map(|r| r * i)))
                    as ArrayRef
            })
            .collect();
        let batch = RecordBatch::try_new(Arc::clone(&schema), columns)?;
        ctx.register_table(
            "wide",
            Arc::new(MemTable::try_new(schema, vec![vec![batch]])?),
        )?;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn aggregations_fail_on_the_errors_of_their_rows() {
        for (sql, message) in [
            ("SELECT SUM(1000 / a) FROM t1", "Divide by zero"),
            (
                "SELECT d, SUM(1000 / a) FROM t1 GROUP BY d",
                "Divide by zero",
            ),
            (
                "SELECT 1000 / a AS q, COUNT(*) FROM t1 GROUP BY 1000 / a",
                "Divide by zero",
            ),
        ] {
            // On 1, 4 and 16 threads (on several, the planner leaves the
            // aggregations of divisions to DataFusion, the lowering of the tests
            // does not).
            for target_partitions in [1, 4, 16] {
                assert_aggregate_fails_with_setup(
                    sql,
                    test_session_config().with_target_partitions(target_partitions),
                    async |ctx: &SessionContext| register_test_tables(ctx),
                    &[message],
                )
                .await;
            }
        }
    }

    /// Register table `big`: 60,000 rows in 4 skewed partitions, with a
    /// high-cardinality key `k` (30,011 values), strings `s` (4,999 values),
    /// values `v`, floats `f` (with `-0.0`, `0.0` and `NaN`s), a dictionary
    /// column `dict` and lists `l`, all with nulls.
    fn register_big_table(ctx: &SessionContext) -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, true),
            Field::new("s", DataType::Utf8, true),
            Field::new("v", DataType::Int64, true),
            Field::new("f", DataType::Float64, true),
            Field::new_dictionary("dict", DataType::Int32, DataType::Utf8, true),
            Field::new_list("l", Field::new_list_field(DataType::Int32, true), true),
        ]));
        let batch = |rows: std::ops::Range<i64>| -> Result<RecordBatch> {
            let k = Int64Array::from_iter(
                rows.clone()
                    .map(|i| (i % 101 != 7).then_some((i * 7_919) % 30_011)),
            );
            let s = StringArray::from_iter(
                rows.clone()
                    .map(|i| (i % 89 != 3).then(|| format!("s{}", (i * 31) % 4_999))),
            );
            let v = Int64Array::from_iter(
                rows.clone().map(|i| (i % 13 != 5).then_some(i * 3 - 1_000)),
            );
            let f = Float64Array::from_iter(rows.clone().map(|i| match i % 11 {
                0 => None,
                1 => Some(-0.0),
                2 => Some(0.0),
                3 => Some(f64::NAN),
                4 => Some(-f64::NAN),
                j => Some((i % 37) as f64 / j as f64),
            }));
            let words = rows
                .clone()
                .map(|i| (i % 17 != 2).then(|| format!("d{}", i % 50)))
                .collect::<Vec<_>>();
            let dict = words
                .iter()
                .map(Option::as_deref)
                .collect::<DictionaryArray<Int32Type>>();
            let l = ListArray::from_iter_primitive::<Int32Type, _, _>(rows.map(|i| {
                (i % 19 != 4).then(|| {
                    (0..i % 3)
                        .map(|j| (j != 1 || i % 2 == 0).then_some((i % 5 + j) as i32))
                        .collect::<Vec<_>>()
                })
            }));
            let columns: Vec<ArrayRef> = vec![
                Arc::new(k),
                Arc::new(s),
                Arc::new(v),
                Arc::new(f),
                Arc::new(dict),
                Arc::new(l),
            ];
            Ok(RecordBatch::try_new(Arc::clone(&schema), columns)?)
        };
        let partitions = vec![
            vec![batch(0..30_000)?, batch(30_000..35_000)?],
            vec![],
            vec![batch(35_000..59_000)?],
            vec![batch(59_000..60_000)?],
        ];
        ctx.register_table("big", Arc::new(MemTable::try_new(schema, partitions)?))?;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn ungrouped_and_grouped_aggregations_are_lowered() {
        let root = assert_aggregate_parity("SELECT COUNT(*), SUM(a) FROM t1").await;
        let is_ungrouped = |root: &Arc<dyn crate::PhysicalOperator>| {
            let mut operator = Arc::clone(root);
            loop {
                if let Some(aggregate) = operator.downcast_ref::<PhysicalHashAggregate>()
                {
                    return aggregate.is_ungrouped();
                }
                operator = Arc::clone(&operator.children()[0]);
            }
        };
        assert!(is_ungrouped(&root));
        let root = assert_aggregate_parity("SELECT d, SUM(a) FROM t1 GROUP BY d").await;
        assert!(!is_ungrouped(&root));
        let root = assert_aggregate_parity(
            "SELECT COUNT(*), SUM(a) FROM t1 GROUP BY GROUPING SETS (())",
        )
        .await;
        assert!(!is_ungrouped(&root));
    }
}
