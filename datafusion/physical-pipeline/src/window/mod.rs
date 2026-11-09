// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! `PhysicalWindow`: window function evaluation.
//!
//! [`PhysicalWindow`] replaces DataFusion's `WindowAggExec` and
//! `BoundedWindowAggExec` (window functions such as `ROW_NUMBER()`,
//! `RANK()`, `LAG`/`LEAD`, `NTILE`, `FIRST_VALUE`, and aggregates over
//! `ROWS`, `RANGE` and `GROUPS` frames), together with the `SortExec`,
//! `SortPreservingMergeExec`, `CoalescePartitionsExec` and
//! `RepartitionExec`s that DataFusion plans below them to partition and sort
//! their input. Like DuckDB's `PhysicalWindow`, it is a pipeline breaker that
//! partitions and sorts its input itself, in parallel:
//!
//! * **Sink**: every thread distributes the rows of the chunks it pushes over
//!   radix *buckets*, by the hash of their `PARTITION BY` keys (all rows of a
//!   window partition are in the same bucket), and sorts the rows of each
//!   bucket into sorted runs (see [`SortKeys`](crate::sort::SortKeys)).
//! * **Finalize**: the threads of a `WindowPartitionSort` pipeline, which
//!   runs once all input pipelines completed, merge the runs of the buckets
//!   in parallel, one bucket at a time, and find the window partitions of
//!   each bucket (its rows with equal `PARTITION BY` keys).
//! * **Source**: the threads claim the sorted buckets, evaluate the window
//!   functions on every window partition, with DataFusion's
//!   [`WindowExpr::evaluate`](datafusion_physical_expr::window::WindowExpr::evaluate)
//!   on the rows of the partition in their order, as `WindowAggExec` does,
//!   and emit the rows with their window values in chunks of at most
//!   [`DEFAULT_VECTOR_SIZE`](crate::chunk::DEFAULT_VECTOR_SIZE) rows.
//!
//! See [`window_op`] for the details and [`PhysicalWindow::try_lower`] for
//! the plans it lowers.

pub mod window_op;

pub use window_op::PhysicalWindow;

/// Test support of the tests of [`PhysicalWindow`]: a lowering with the
/// native operators, standing in for the planner, and differential checks of
/// SQL queries lowered with it.
#[cfg(test)]
pub(crate) mod test_support {
    use std::fmt;
    use std::sync::Arc;
    use std::time::Duration;

    use arrow::array::{RecordBatch, RecordBatchOptions};
    use arrow::compute::filter_record_batch;
    use arrow::datatypes::SchemaRef;
    use datafusion::prelude::{SessionConfig, SessionContext};
    use datafusion_common::Result;
    use datafusion_common::cast::as_boolean_array;
    use datafusion_common::config::ConfigOptions;
    use datafusion_physical_expr::LexOrdering;
    use datafusion_physical_expr::expressions::Column;
    use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
    use datafusion_physical_plan::filter::FilterExec;
    use datafusion_physical_plan::projection::ProjectionExec;
    use datafusion_physical_plan::repartition::RepartitionExec;
    use datafusion_physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
    use datafusion_physical_plan::{
        ExecutionPlan, ExecutionPlanProperties, Partitioning, collect,
    };

    use super::PhysicalWindow;
    use crate::aggregate::PhysicalHashAggregate;
    use crate::bridge::{BridgeOperator, PipelineRootExec};
    use crate::chunk::DataChunk;
    use crate::explain::format_operator_tree;
    use crate::join::PhysicalHashJoin;
    use crate::operator::{
        ExecutionContext, GlobalOperatorState, OperatorResult, OperatorState,
        PhysicalOperator,
    };
    use crate::operators::limit::PhysicalLimit;
    use crate::operators::placeholder_row::PhysicalPlaceholderRow;
    use crate::operators::union::PhysicalUnion;
    use crate::sort::PhysicalSort;
    use crate::test_utils::{
        EngineOutput, assert_byte_eq_with_datafusion_with_config, compare_outputs,
        datafusion_context, register_test_tables, run_sql,
    };

    /// Lower `plan` with the native operators, standing in for the planner
    /// until it lowers windows (#10):
    ///
    /// * windows with [`PhysicalWindow::try_lower`];
    /// * sorts (with `crate::sort::try_lower`), hash joins, aggregations,
    ///   limits, unions and placeholder rows with their `try_lower`;
    /// * a `SortPreservingMergeExec` without fetch that is not the merge of a
    ///   sort (e.g. above a window of several partitions) with a
    ///   [`PhysicalSort`] by its keys: merging sorted partitions is sorting
    ///   all rows;
    /// * `CoalescePartitionsExec`s without a fetch and `RepartitionExec`s
    ///   that do not preserve the order and do not evaluate expressions (round
    ///   robin, or hash partitioning by columns) are removed: the pipeline
    ///   engine merges the partitions of their input anyway;
    /// * `ProjectionExec`s and `FilterExec`s (without a fetch) with
    ///   [`TestMap`]s, minimal stand-ins for the native operators, so that the
    ///   operators below them are lowered too;
    /// * scans with the native sources;
    /// * every other node is bridged, with its subtree.
    pub(crate) fn lower(
        plan: &Arc<dyn ExecutionPlan>,
    ) -> Result<Arc<dyn PhysicalOperator>> {
        if let Some(window) = PhysicalWindow::try_lower(plan, lower)? {
            return Ok(Arc::new(window));
        }
        if let Some(sort) =
            crate::sort::try_lower(plan, &ConfigOptions::default(), lower)?
        {
            return Ok(sort);
        }
        if let Some(join) = PhysicalHashJoin::try_lower(plan, lower)? {
            return Ok(Arc::new(join));
        }
        if let Some(aggregate) = PhysicalHashAggregate::try_lower(plan, lower)? {
            return Ok(Arc::new(aggregate));
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
        if let Some(merge) = plan.downcast_ref::<SortPreservingMergeExec>()
            && merge.fetch().is_none()
            && merge.input().output_partitioning().partition_count() > 0
        {
            let ordering = LexOrdering::clone(merge.expr());
            return Ok(Arc::new(PhysicalSort::try_new(
                lower(merge.input())?,
                ordering,
                None,
            )?));
        }
        if let Some(coalesce) = plan.downcast_ref::<CoalescePartitionsExec>()
            && coalesce.fetch().is_none()
            && coalesce.input().output_partitioning().partition_count() > 0
        {
            return lower(coalesce.input());
        }
        if let Some(repartition) = plan.downcast_ref::<RepartitionExec>()
            && !repartition.preserve_order()
            && match repartition.partitioning() {
                Partitioning::RoundRobinBatch(_) => true,
                Partitioning::Hash(exprs, _) => {
                    exprs.iter().all(|expr| expr.is::<Column>())
                }
                _ => false,
            }
        {
            return lower(repartition.input());
        }
        if let Some(projection) = plan.downcast_ref::<ProjectionExec>() {
            let exprs = projection
                .expr()
                .iter()
                .map(|expr| Arc::clone(&expr.expr))
                .collect::<Vec<_>>();
            let schema = projection.schema();
            return Ok(Arc::new(TestMap {
                name: "TestProjection",
                children: vec![lower(projection.input())?],
                schema: Arc::clone(&schema),
                map: Box::new(move |batch| {
                    let columns = exprs
                        .iter()
                        .map(|expr| expr.evaluate(batch)?.into_array(batch.num_rows()))
                        .collect::<Result<Vec<_>>>()?;
                    let options =
                        RecordBatchOptions::new().with_row_count(Some(batch.num_rows()));
                    Ok(RecordBatch::try_new_with_options(
                        Arc::clone(&schema),
                        columns,
                        &options,
                    )?)
                }),
            }));
        }
        if let Some(filter) = plan.downcast_ref::<FilterExec>()
            && filter.fetch().is_none()
        {
            let predicate = Arc::clone(filter.predicate());
            let projection = filter.projection().as_deref().map(<[usize]>::to_vec);
            let schema = filter.schema();
            return Ok(Arc::new(TestMap {
                name: "TestFilter",
                children: vec![lower(filter.input())?],
                schema: Arc::clone(&schema),
                map: Box::new(move |batch| {
                    let mask = predicate.evaluate(batch)?.into_array(batch.num_rows())?;
                    let filtered = filter_record_batch(batch, as_boolean_array(&mask)?)?;
                    let columns = match &projection {
                        Some(projection) => projection
                            .iter()
                            .map(|&i| Arc::clone(filtered.column(i)))
                            .collect(),
                        None => filtered.columns().to_vec(),
                    };
                    let options = RecordBatchOptions::new()
                        .with_row_count(Some(filtered.num_rows()));
                    Ok(RecordBatch::try_new_with_options(
                        Arc::clone(&schema),
                        columns,
                        &options,
                    )?)
                }),
            }));
        }
        if let Some(source) = crate::sources::try_create_scan_source(plan)? {
            return Ok(source);
        }
        Ok(Arc::new(BridgeOperator::new(Arc::clone(plan))))
    }

    type MapFn = Box<dyn Fn(&RecordBatch) -> Result<RecordBatch> + Send + Sync>;

    /// A streaming operator that maps the compacted input chunks with `map`.
    pub(crate) struct TestMap {
        name: &'static str,
        children: Vec<Arc<dyn PhysicalOperator>>,
        schema: SchemaRef,
        map: MapFn,
    }

    impl fmt::Debug for TestMap {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct(self.name).finish_non_exhaustive()
        }
    }

    impl PhysicalOperator for TestMap {
        fn name(&self) -> &str {
            self.name
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
            *output = DataChunk::from_record_batch((self.map)(&input.compact()?)?);
            Ok(OperatorResult::NeedMoreInput)
        }
    }

    /// Await `future`, failing the test if it does not complete in time.
    pub(crate) async fn with_timeout<T>(future: impl Future<Output = T>) -> T {
        tokio::time::timeout(Duration::from_secs(120), future)
            .await
            .expect("the execution did not complete")
    }

    /// Assert that `sql` produces byte-identical results on DataFusion and on
    /// the pipeline engine, both through the `PipelineQueryPlanner` and with
    /// the plan lowered by [`lower`], on sessions configured with `config`
    /// and set up with `setup`. Returns the lowered operator tree.
    pub(crate) async fn assert_native_parity_with_setup<F>(
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
        let expected = with_timeout(run_sql(&ctx, sql)).await.expect("DataFusion");
        // A new plan: executing a plan changes the state of some nodes.
        let plan = ctx
            .sql(sql)
            .await
            .expect("planning")
            .create_physical_plan()
            .await
            .expect("physical planning");
        let root = lower(&plan).expect("lowering");
        let tree = format_operator_tree(&root);
        let exec: Arc<dyn ExecutionPlan> = Arc::new(
            PipelineRootExec::try_new(Arc::clone(&root), plan).expect("a pipeline root"),
        );
        let batches = with_timeout(collect(Arc::clone(&exec), ctx.task_ctx()))
            .await
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to run the lowered plan: {e}\nquery: {sql}\nlowered: {tree}"
                )
            });
        let actual = EngineOutput {
            schema: exec.schema(),
            plan: exec,
            batches,
            ordered: expected.ordered,
        };
        if let Err(message) = compare_outputs(&expected, &actual) {
            panic!("Results differ: {message}\nquery: {sql}\nlowered: {tree}");
        }
        root
    }

    /// [`assert_native_parity_with_setup`] with `config` and the default test
    /// tables.
    pub(crate) async fn assert_native_parity_with_config(
        sql: &str,
        config: SessionConfig,
    ) -> Arc<dyn PhysicalOperator> {
        let setup = async |ctx: &SessionContext| register_test_tables(ctx);
        assert_native_parity_with_setup(sql, config, setup).await
    }
}
