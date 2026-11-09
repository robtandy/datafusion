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
//! # Byte-for-byte compatibility
//!
//! The operator computes exactly what DataFusion computes: it uses the
//! expressions, the accumulators ([`Accumulator`] and [`GroupsAccumulator`],
//! created by the `AggregateFunctionExpr`s of the plan) and the output schema
//! of the `AggregateExec`s it replaces, and DataFusion's own group key
//! storage (`GroupValues`), so the groups (`NULL`s, `-0.0`, `NaN`s, ...)
//! and the aggregate values are DataFusion's, including:
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
//!   before it aggregates them, as DataFusion sorts every input partition.
//!
//! Only the order of the output rows differs: DataFusion's order of groups
//! is not defined either.
//!
//! [`Accumulator`]: datafusion_expr::Accumulator
//! [`GroupsAccumulator`]: datafusion_expr::GroupsAccumulator

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

    use arrow::array::{AsArray, RecordBatch, RecordBatchOptions};
    use arrow::compute::{
        SortColumn, concat_batches, filter_record_batch, lexsort_to_indices,
        take_record_batch,
    };
    use arrow::datatypes::SchemaRef;
    use datafusion::prelude::{SessionConfig, SessionContext};
    use datafusion_common::{Result, internal_err};
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
        EngineOutput, assert_byte_eq_with_datafusion_with_config, compare_outputs,
        datafusion_context, register_test_tables, run_sql, test_session_config,
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

    /// Assert that `sql` fails with an error that contains `message`, both on
    /// DataFusion and with the plan lowered by [`lower`], in which at least
    /// one aggregation is a [`PhysicalHashAggregate`].
    pub(crate) async fn assert_aggregate_fails_like_datafusion(sql: &str, message: &str) {
        let ctx = datafusion_context(test_session_config());
        register_test_tables(&ctx).expect("setup");
        let error = run_sql(&ctx, sql).await.expect_err("DataFusion fails");
        assert!(error.to_string().contains(message), "DataFusion: {error}");
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
            Ok(_) => panic!("The pipeline engine succeeds, DataFusion fails: {error}"),
            Err(e) => assert!(e.to_string().contains(message), "pipeline engine: {e}"),
        }
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
        assert_byte_eq_with_datafusion_with_config(sql, config.clone(), &setup).await;

        let ctx = datafusion_context(config);
        setup(&ctx).await.expect("setup");
        let expected = run_sql(&ctx, sql).await.expect("DataFusion");
        let mut lowered = None;
        for (lower, tables) in [
            (lower as fn(&_) -> _, "default"),
            (lower_with_tiny_tables, "tiny"),
        ] {
            // A new plan: executing a plan changes the state of some nodes.
            let plan = ctx
                .sql(sql)
                .await
                .expect("planning")
                .create_physical_plan()
                .await
                .expect("physical planning");
            let root = lower(&plan).expect("lowering");
            assert!(
                count::<PhysicalHashAggregate>(&root) > 0,
                "No aggregation was lowered\nquery: {sql}\nlowered: {}",
                format_operator_tree(&root)
            );
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
            let actual = EngineOutput {
                schema: exec.schema(),
                plan: exec,
                batches,
                ordered: expected.ordered,
            };
            if let Err(message) = compare_outputs(&expected, &actual) {
                panic!(
                    "Results differ ({tables} tables): {message}\nquery: {sql}\n\
                     lowered: {}",
                    format_operator_tree(&root)
                );
            }
            lowered.get_or_insert(root);
        }
        lowered.expect("a lowered plan")
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

    use super::PhysicalHashAggregate;
    use super::test_support::*;
    use crate::operators::union::PhysicalUnion;
    use crate::operators::union::test_support::count;
    use crate::sources::test_harness::write_parquet;
    use crate::test_utils::{register_test_tables, test_session_config, test_table_t1};

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
            "SELECT DISTINCT d FROM t1 ORDER BY d LIMIT 3",
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
        // tightens as it aggregates; the native aggregation leaves it as it
        // is, which filters nothing.
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
    async fn soft_limited_distinct_aggregations_are_left_to_datafusion() {
        // Tables of a single partition. DataFusion's `DISTINCT ... LIMIT n`
        // stops reading as soon as it has `n` groups: after the first batch
        // here, so it never divides by zero (`t`), and returns the groups of
        // the rows it read first (`s`). A native aggregation would read all
        // batches.
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
            let t = vec![vec![batch(vec![1])?, batch(vec![0])?]];
            ctx.register_table(
                "t",
                Arc::new(MemTable::try_new(Arc::clone(&schema), t)?),
            )?;
            let s = vec![vec![
                batch(vec![5, 3])?,
                batch(vec![4, 4, 1])?,
                batch(vec![2])?,
            ]];
            ctx.register_table("s", Arc::new(MemTable::try_new(schema, s)?))?;
            Ok(())
        };
        for (sql, thread_counts) in [
            (
                "SELECT COUNT(*) FROM (SELECT DISTINCT 1000 / a FROM t LIMIT 1)",
                &[1, 4][..],
            ),
            (
                "SELECT COUNT(*), MIN(x), MAX(x) FROM (SELECT DISTINCT a AS x FROM s LIMIT 2)",
                &[1],
            ),
        ] {
            for &threads in thread_counts {
                let config = test_session_config().with_target_partitions(threads);
                let root = assert_aggregate_parity_with_setup(sql, config, &setup).await;
                // Only the outer aggregation is native.
                assert_eq!(count::<PhysicalHashAggregate>(&root), 1, "{sql}");
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn aggregation_errors_like_datafusion() {
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
            assert_aggregate_fails_like_datafusion(sql, message).await;
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
