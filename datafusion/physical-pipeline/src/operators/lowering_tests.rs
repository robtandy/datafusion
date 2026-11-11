// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Helpers of the tests of the native lowering of the nodes that used to be
//! left to DataFusion (see `docs/fallbacks.md`).

use std::sync::Arc;

use datafusion::prelude::{SessionConfig, SessionContext};
use datafusion_common::Result;

use crate::bridge::{PipelineRootExec, bridged_plans};
use crate::explain::format_operator_tree;
use crate::operator::PhysicalOperator;
use crate::test_utils::{compare_outputs, run_engines};

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
