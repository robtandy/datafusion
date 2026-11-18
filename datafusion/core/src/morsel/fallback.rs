// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::any::Any;
use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion_common::config::{ConfigOptions, TableOptions};
use datafusion_common::{DFSchema, Result};
use datafusion_execution::TaskContext;
use datafusion_execution::config::SessionConfig;
use datafusion_execution::runtime_env::RuntimeEnv;
use datafusion_expr::execution_props::ExecutionProps;
use datafusion_expr::registry::ExtensionTypeRegistryRef;
use datafusion_expr::{
    AggregateUDF, Expr, HigherOrderUDF, LogicalPlan, ScalarUDF, WindowUDF,
};
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::operator_statistics::StatisticsRegistry;
use datafusion_session::{
    CatalogProviderList, PhysicalOptimizerRule, QueryPlanner, Session,
};

use crate::execution::session_state::{DefaultQueryPlanner, SessionState};
use crate::physical_planner::{DefaultPhysicalPlanner, PhysicalPlanner};

pub(super) async fn create_physical_plan(
    plan: &LogicalPlan,
    session: &dyn Session,
) -> Result<Arc<dyn ExecutionPlan>> {
    // Providers can invoke the session's planner from scan. Give them a default
    // planner as well, without changing the session used for other queries.
    let planner = DefaultPhysicalPlanner::default();
    if let Some(state) = session.as_any().downcast_ref::<SessionState>() {
        // Some providers downcast sessions to SessionState; keep those working.
        let fallback = state.clone_with_default_query_planner();
        planner.create_physical_plan(plan, &fallback).await
    } else {
        let fallback = FallbackSession {
            session,
            table_options: session.table_options().clone(),
        };
        planner.create_physical_plan(plan, &fallback).await
    }
}

struct FallbackSession<'a> {
    session: &'a dyn Session,
    table_options: TableOptions,
}

#[async_trait]
impl Session for FallbackSession<'_> {
    fn session_id(&self) -> &str {
        self.session.session_id()
    }

    fn config(&self) -> &SessionConfig {
        self.session.config()
    }

    fn config_options(&self) -> &ConfigOptions {
        self.session.config_options()
    }

    fn catalog_list(&self) -> Arc<dyn CatalogProviderList> {
        self.session.catalog_list()
    }

    fn query_planner(&self) -> Arc<dyn QueryPlanner + Send + Sync> {
        Arc::new(DefaultQueryPlanner {})
    }

    fn optimize(&self, plan: &LogicalPlan) -> Result<LogicalPlan> {
        self.session.optimize(plan)
    }

    fn physical_optimizers(&self) -> &[Arc<dyn PhysicalOptimizerRule + Send + Sync>] {
        self.session.physical_optimizers()
    }

    fn statistics_registry(&self) -> Option<&StatisticsRegistry> {
        self.session.statistics_registry()
    }

    async fn create_physical_plan(
        &self,
        plan: &LogicalPlan,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        DefaultPhysicalPlanner::default()
            .create_physical_plan(&self.optimize(plan)?, self)
            .await
    }

    fn create_physical_expr(
        &self,
        expr: Expr,
        df_schema: &DFSchema,
    ) -> Result<Arc<dyn PhysicalExpr>> {
        self.session.create_physical_expr(expr, df_schema)
    }

    fn scalar_functions(&self) -> &HashMap<String, Arc<ScalarUDF>> {
        self.session.scalar_functions()
    }

    fn higher_order_functions(&self) -> &HashMap<String, Arc<HigherOrderUDF>> {
        self.session.higher_order_functions()
    }

    fn aggregate_functions(&self) -> &HashMap<String, Arc<AggregateUDF>> {
        self.session.aggregate_functions()
    }

    fn window_functions(&self) -> &HashMap<String, Arc<WindowUDF>> {
        self.session.window_functions()
    }

    fn extension_type_registry(&self) -> &ExtensionTypeRegistryRef {
        self.session.extension_type_registry()
    }

    fn runtime_env(&self) -> &Arc<RuntimeEnv> {
        self.session.runtime_env()
    }

    fn execution_props(&self) -> &ExecutionProps {
        self.session.execution_props()
    }

    fn as_any(&self) -> &dyn Any {
        self.session.as_any()
    }

    fn table_options(&self) -> &TableOptions {
        &self.table_options
    }

    fn default_table_options(&self) -> TableOptions {
        self.session.default_table_options()
    }

    fn table_options_mut(&mut self) -> &mut TableOptions {
        &mut self.table_options
    }

    fn task_ctx(&self) -> Arc<TaskContext> {
        self.session.task_ctx()
    }
}
