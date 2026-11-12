// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Helpers of the tests of the native lowering of the nodes that used to be
//! left to DataFusion (see `docs/fallbacks.md`).

use std::sync::Arc;

use datafusion::execution::SessionStateBuilder;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::physical_planner::DefaultPhysicalPlanner;
use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_common::{DataFusionError, Result};
use datafusion_execution::memory_pool::GreedyMemoryPool;

use crate::PipelineQueryPlanner;
use crate::bridge::{PipelineRootExec, bridged_plans};
use crate::explain::format_operator_tree;
use crate::operator::PhysicalOperator;
use crate::test_utils::{compare_outputs, run_engines, test_session_config};

/// The number of operators of type `T` in the operator tree `operator`.
pub(crate) fn count<T: PhysicalOperator>(operator: &Arc<dyn PhysicalOperator>) -> usize {
    usize::from(operator.is::<T>())
        + operator.children().iter().map(count::<T>).sum::<usize>()
}

/// Assert that `sql` returns the same results through the
/// [`PipelineQueryPlanner`](crate::PipelineQueryPlanner) as on default
/// DataFusion, in sessions configured with `config` and set up with `setup`
/// (compared by meaning, in their canonical forms: the queries of these tests
/// have unambiguous results), and that the pipeline engine runs all of it
/// natively: no part of its plan is left to DataFusion. Returns the operator
/// tree of the pipeline engine.
///
/// The results are compared with `compare_outputs` until
/// `test_utils::assert_same_results_as_datafusion` exists (#55), which
/// compares them by meaning also where they are ambiguous.
pub(crate) async fn assert_lowered<F>(
    sql: &str,
    config: SessionConfig,
    setup: F,
) -> Arc<dyn PhysicalOperator>
where
    F: AsyncFn(&SessionContext) -> Result<()>,
{
    let (expected, actual) = run_engines(sql, config, setup)
        .await
        .unwrap_or_else(|e| panic!("Failed to run query on both engines: {e}\n{sql}"));
    let root = Arc::clone(
        actual
            .plan
            .downcast_ref::<PipelineRootExec>()
            .expect("a pipeline plan")
            .operator(),
    );
    if let Err(message) = compare_outputs(&expected, &actual) {
        panic!(
            "Results differ: {message}\nquery: {sql}\nlowered: {}",
            format_operator_tree(&root)
        );
    }
    let bridged = bridged_plans(&root);
    assert!(
        bridged.is_empty(),
        "Parts of the plan are left to DataFusion: {:?}\nquery: {sql}\nlowered: {}",
        bridged.iter().map(|plan| plan.name()).collect::<Vec<_>>(),
        format_operator_tree(&root)
    );
    root
}

/// A `SessionContext` that runs queries with the pipeline engine, with the
/// configuration of the tests and a memory pool of `limit` bytes.
pub(crate) fn pipeline_context_with_memory_limit(limit: usize) -> SessionContext {
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::new(GreedyMemoryPool::new(limit)))
        .build_arc()
        .unwrap();
    let planner = PipelineQueryPlanner::new(Arc::new(DefaultPhysicalPlanner::default()));
    let state = SessionStateBuilder::new()
        .with_config(test_session_config())
        .with_runtime_env(runtime)
        .with_default_features()
        .with_query_planner(Arc::new(planner))
        .build();
    SessionContext::new_with_state(state)
}

/// A `TaskContext` with a memory pool of `limit` bytes.
pub(crate) fn task_context_with_memory_limit(
    limit: usize,
) -> Arc<datafusion_execution::TaskContext> {
    let runtime = RuntimeEnvBuilder::new()
        .with_memory_pool(Arc::new(GreedyMemoryPool::new(limit)))
        .build_arc()
        .unwrap();
    Arc::new(datafusion_execution::TaskContext::default().with_runtime(runtime))
}

/// Assert that `result` failed with `ResourcesExhausted`.
pub(crate) fn assert_exhausted<T>(result: Result<T>) {
    match result {
        Ok(_) => panic!("succeeds, expected ResourcesExhausted"),
        Err(error) => assert!(
            matches!(error.find_root(), DataFusionError::ResourcesExhausted(_)),
            "{error}"
        ),
    }
}

/// The operator tree that the pipeline engine of `ctx` lowers `sql` to.
pub(crate) async fn lowered(
    ctx: &SessionContext,
    sql: &str,
) -> Result<Arc<dyn PhysicalOperator>> {
    let plan = ctx.sql(sql).await?.create_physical_plan().await?;
    let root = plan
        .downcast_ref::<PipelineRootExec>()
        .expect("a pipeline plan");
    Ok(Arc::clone(root.operator()))
}

/// Assert that `sql` fails with `ResourcesExhausted` in `ctx`, whose memory
/// pool cannot hold what its native operators reserve.
pub(crate) async fn assert_resources_exhausted(ctx: &SessionContext, sql: &str) {
    let result = match ctx.sql(sql).await {
        Ok(df) => df.collect().await,
        Err(e) => Err(e),
    };
    match result {
        Ok(_) => panic!("{sql} succeeds, expected ResourcesExhausted"),
        Err(error) => assert!(
            matches!(error.find_root(), DataFusionError::ResourcesExhausted(_)),
            "{sql}: {error}"
        ),
    }
}

/// Assert that the plan that `make_plan` makes returns the same rows through
/// the pipeline engine (lowered with the options and the runtime of a
/// session configured with `config`) as on DataFusion, compared in their
/// canonical forms (in their order if `ordered`), and that the pipeline
/// engine runs all of it natively. Returns the operator tree of the pipeline
/// engine.
pub(crate) async fn assert_plan_lowered<F>(
    make_plan: F,
    config: SessionConfig,
    ordered: bool,
) -> Arc<dyn PhysicalOperator>
where
    F: Fn() -> Arc<dyn datafusion_physical_plan::ExecutionPlan>,
{
    use crate::test_utils::{EngineOutput, datafusion_context};
    use datafusion_physical_plan::collect;

    let ctx = datafusion_context(config);
    let output =
        |plan: Arc<dyn datafusion_physical_plan::ExecutionPlan>, batches| EngineOutput {
            schema: plan.schema(),
            plan,
            batches,
            ordered,
        };
    let plan = make_plan();
    let batches = collect(Arc::clone(&plan), ctx.task_ctx()).await.unwrap();
    let expected = output(plan, batches);
    let state = ctx.state();
    let root: Arc<dyn datafusion_physical_plan::ExecutionPlan> =
        crate::planner::create_pipeline_plan_with_runtime(
            make_plan(),
            state.config_options(),
            state.runtime_env(),
        )
        .unwrap();
    let batches = collect(Arc::clone(&root), ctx.task_ctx()).await.unwrap();
    let operator = Arc::clone(
        root.downcast_ref::<PipelineRootExec>()
            .expect("a pipeline plan")
            .operator(),
    );
    let actual = output(root, batches);
    if let Err(message) = compare_outputs(&expected, &actual) {
        panic!(
            "Results differ: {message}\nlowered: {}",
            format_operator_tree(&operator)
        );
    }
    let bridged = bridged_plans(&operator);
    assert!(
        bridged.is_empty(),
        "Parts of the plan are left to DataFusion: {:?}\nlowered: {}",
        bridged.iter().map(|plan| plan.name()).collect::<Vec<_>>(),
        format_operator_tree(&operator)
    );
    operator
}

/// The native lowering of the nodes that only distribute the rows of their
/// inputs, and of the merges of sorted partitions, see `planner.rs`.
mod merges_and_redistributions {
    use super::*;

    use arrow::array::{Int64Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
    use datafusion::datasource::MemTable;
    use datafusion_common::{ScalarValue, SplitPoint};
    use datafusion_datasource::memory::MemorySourceConfig;
    use datafusion_expr::{SortExpr, col as logical_col};
    use datafusion_physical_expr::expressions::col;
    use datafusion_physical_expr::{LexOrdering, PhysicalSortExpr, RangePartitioning};
    use datafusion_physical_plan::repartition::RepartitionExec;
    use datafusion_physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
    use datafusion_physical_plan::{ExecutionPlan, Partitioning};

    use crate::sort::{PhysicalSort, PhysicalTopK};

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("v", DataType::Int64, false),
            Field::new("w", DataType::Int64, true),
        ]))
    }

    /// The rows `values`, sorted by `v`.
    fn batch(values: std::ops::Range<i64>) -> RecordBatch {
        RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int64Array::from_iter_values(values.clone())),
                Arc::new(Int64Array::from_iter(
                    values.map(|v| (v % 7 != 3).then_some(v % 11)),
                )),
            ],
        )
        .unwrap()
    }

    /// Three partitions, every one sorted by `v`.
    fn sorted_partitions() -> Vec<Vec<RecordBatch>> {
        vec![
            vec![batch(0..3000), batch(3000..3100)],
            vec![],
            vec![batch(1000..1500), batch(9000..9500)],
        ]
    }

    fn ordering() -> LexOrdering {
        LexOrdering::new(vec![PhysicalSortExpr::new_default(
            col("v", &schema()).unwrap(),
        )])
        .unwrap()
    }

    /// The sorted partitions, scanned with their ordering.
    fn sorted_scan() -> Arc<dyn ExecutionPlan> {
        let source = MemorySourceConfig::try_new(&sorted_partitions(), schema(), None)
            .unwrap()
            .try_with_sort_information(vec![ordering()])
            .unwrap();
        datafusion_datasource::source::DataSourceExec::from_data_source(source)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn merges_of_sorted_partitions_are_sorts() {
        // A table of sorted partitions: DataFusion merges them.
        let setup = async |ctx: &SessionContext| {
            let table =
                MemTable::try_new(schema(), sorted_partitions())?.with_sort_order(vec![
                    vec![SortExpr::new(logical_col("v"), true, false)],
                ]);
            ctx.register_table("sorted", Arc::new(table))?;
            Ok(())
        };
        for (sql, sorts, topks) in [
            ("SELECT * FROM sorted ORDER BY v", 1, 0),
            ("SELECT v, w FROM sorted WHERE w > 3 ORDER BY v", 1, 0),
            ("SELECT * FROM sorted ORDER BY v LIMIT 10", 0, 1),
            ("SELECT * FROM sorted ORDER BY v LIMIT 10 OFFSET 3000", 0, 1),
        ] {
            let root = assert_lowered(sql, test_session_config(), &setup).await;
            assert_eq!(count::<PhysicalSort>(&root), sorts, "{sql}");
            assert_eq!(count::<PhysicalTopK>(&root), topks, "{sql}");
        }
        // A merge of the sorted partitions of a scan, with a fetch.
        for fetch in [None, Some(1), Some(4000)] {
            let make_plan = || -> Arc<dyn ExecutionPlan> {
                Arc::new(
                    SortPreservingMergeExec::new(ordering(), sorted_scan())
                        .with_fetch(fetch),
                )
            };
            let root = assert_plan_lowered(make_plan, test_session_config(), true).await;
            assert_eq!(
                count::<PhysicalSort>(&root) + count::<PhysicalTopK>(&root),
                1,
                "{fetch:?}"
            );
        }
    }

    /// A `CoalesceBatchesExec` (which DataFusion does not plan any more) of
    /// `input`, with a limit of `fetch` rows.
    #[expect(deprecated)]
    fn coalesce_batches(
        input: Arc<dyn ExecutionPlan>,
        fetch: Option<usize>,
    ) -> Arc<dyn ExecutionPlan> {
        Arc::new(
            datafusion_physical_plan::coalesce_batches::CoalesceBatchesExec::new(
                input, 8192,
            )
            .with_fetch(fetch),
        )
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn coalescings_with_a_fetch_of_a_partition_are_limits() {
        let single = || -> Arc<dyn ExecutionPlan> {
            let source = MemorySourceConfig::try_new(
                &[vec![batch(0..3000), batch(3000..5000)]],
                schema(),
                None,
            )
            .unwrap();
            datafusion_datasource::source::DataSourceExec::from_data_source(source)
        };
        for fetch in [1, 2047, 2049, 4000, 6000] {
            let make_plan = || coalesce_batches(single(), Some(fetch));
            let root = assert_plan_lowered(make_plan, test_session_config(), true).await;
            assert!(
                root.is::<crate::operators::limit::PhysicalLimit>(),
                "{fetch}"
            );
        }
        // Of several partitions, it limits every partition: DataFusion runs it.
        let root = crate::planner::lower_execution_plan(&coalesce_batches(
            sorted_scan(),
            Some(10),
        ))
        .unwrap();
        assert!(root.is::<crate::bridge::BridgeOperator>());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn repartitionings_that_preserve_the_order_or_by_range_are_removed() {
        let v = col("v", &schema()).unwrap();
        let range = RangePartitioning::try_new(
            ordering(),
            vec![
                SplitPoint::new(vec![ScalarValue::Int64(Some(1200))]),
                SplitPoint::new(vec![ScalarValue::Int64(Some(5000))]),
            ],
        )
        .unwrap();
        for (partitioning, preserve_order) in [
            (Partitioning::RoundRobinBatch(4), true),
            (Partitioning::Hash(vec![Arc::clone(&v)], 5), true),
            (Partitioning::Range(range.clone()), false),
            (Partitioning::Range(range), true),
        ] {
            let repartition = || -> Arc<dyn ExecutionPlan> {
                let repartition =
                    RepartitionExec::try_new(sorted_scan(), partitioning.clone())
                        .unwrap();
                Arc::new(if preserve_order {
                    repartition.with_preserve_order()
                } else {
                    repartition
                })
            };
            let root =
                assert_plan_lowered(repartition, test_session_config(), false).await;
            assert!(
                root.is::<crate::sources::MemoryScanSource>(),
                "{partitioning}"
            );
            // And merged.
            if preserve_order {
                let merged = || -> Arc<dyn ExecutionPlan> {
                    Arc::new(SortPreservingMergeExec::new(ordering(), repartition()))
                };
                let root = assert_plan_lowered(merged, test_session_config(), true).await;
                assert_eq!(count::<PhysicalSort>(&root), 1, "{partitioning}");
            }
        }
    }
}
