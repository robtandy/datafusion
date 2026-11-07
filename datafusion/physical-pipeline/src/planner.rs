// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`PipelineQueryPlanner`]: plugs the pipeline engine into a DataFusion
//! `SessionContext`.
//!
//! The planner first runs a regular DataFusion [`PhysicalPlanner`]
//! (normally `DefaultPhysicalPlanner`) so that the plan is bound, coerced and
//! optimized exactly as in default DataFusion: the output schema (including
//! field metadata), the physical expressions and all pushdowns into scans are
//! those of DataFusion. The resulting [`ExecutionPlan`] is then *lowered* into
//! a [`PhysicalOperator`] tree by [`lower_execution_plan`], and wrapped in a
//! [`PipelineRootExec`] that runs it with the pipeline engine.
//!
//! Nodes that have no native pipeline operator yet are wrapped in a
//! [`BridgeOperator`], which executes the whole subtree with DataFusion's own
//! implementation, so every plan can be executed.
//!
//! # Example
//!
//! ```ignore
//! use datafusion::execution::SessionStateBuilder;
//! use datafusion::physical_planner::DefaultPhysicalPlanner;
//! use datafusion::prelude::*;
//! use datafusion_physical_pipeline::planner::PipelineQueryPlanner;
//!
//! let planner = PipelineQueryPlanner::new(Arc::new(DefaultPhysicalPlanner::default()));
//! let state = SessionStateBuilder::new()
//!     .with_default_features()
//!     .with_query_planner(Arc::new(planner))
//!     .build();
//! let ctx = SessionContext::new_with_state(state);
//! ```

use std::fmt;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion_common::Result;
use datafusion_expr::LogicalPlan;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_session::{PhysicalPlanner, QueryPlanner, Session};

use crate::bridge::{BridgeOperator, PipelineRootExec};
use crate::operator::PhysicalOperator;

/// A [`QueryPlanner`] that executes queries with the pipeline engine.
///
/// See the [module documentation](self) for details.
pub struct PipelineQueryPlanner {
    physical_planner: Arc<dyn PhysicalPlanner>,
}

impl PipelineQueryPlanner {
    /// Create a planner that lowers the plans created by `physical_planner`
    /// (usually `datafusion::physical_planner::DefaultPhysicalPlanner`).
    pub fn new(physical_planner: Arc<dyn PhysicalPlanner>) -> Self {
        Self { physical_planner }
    }

    /// The DataFusion physical planner whose plans are lowered.
    pub fn physical_planner(&self) -> &Arc<dyn PhysicalPlanner> {
        &self.physical_planner
    }
}

impl fmt::Debug for PipelineQueryPlanner {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PipelineQueryPlanner")
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl QueryPlanner for PipelineQueryPlanner {
    async fn create_physical_plan(
        &self,
        logical_plan: &LogicalPlan,
        session: &dyn Session,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let plan = self
            .physical_planner
            .create_physical_plan(logical_plan, session)
            .await?;
        Ok(create_pipeline_plan(plan)?)
    }
}

/// Lower `plan` and wrap it into a [`PipelineRootExec`].
pub fn create_pipeline_plan(
    plan: Arc<dyn ExecutionPlan>,
) -> Result<Arc<PipelineRootExec>> {
    let root = lower_execution_plan(&plan)?;
    Ok(Arc::new(PipelineRootExec::try_new(root, plan)?))
}

/// Lower a DataFusion [`ExecutionPlan`] tree into a [`PhysicalOperator`] tree.
///
/// Every node without a native pipeline operator is executed by a
/// [`BridgeOperator`], together with its whole subtree. No node is lowered
/// natively yet: the native operators (scans, filters, projections, joins,
/// aggregates, sorts, ...) are wired in here as they are implemented.
pub fn lower_execution_plan(
    plan: &Arc<dyn ExecutionPlan>,
) -> Result<Arc<dyn PhysicalOperator>> {
    Ok(Arc::new(BridgeOperator::new(Arc::clone(plan))))
}
