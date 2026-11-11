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
