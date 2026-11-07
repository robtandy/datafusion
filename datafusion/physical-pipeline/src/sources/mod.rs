// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Native scan sources that hand out chunks of at most
//! [`DEFAULT_VECTOR_SIZE`](crate::chunk::DEFAULT_VECTOR_SIZE) rows to all the
//! threads of a pipeline.
//!
//! * [`MemoryScanSource`] scans in-memory batches (`MemTable`, `VALUES`):
//!   all batches of all partitions are cut into slices of at most
//!   `DEFAULT_VECTOR_SIZE` rows, which the threads claim through a shared
//!   atomic cursor. One huge partition is therefore scanned by all threads.
//! * [`EmptyScanSource`] replaces `EmptyExec`: it produces no rows.
//!
//! [`try_create_scan_source`] converts the scans of a DataFusion plan into
//! these sources.
//!
//! # Output
//!
//! A source produces exactly the rows that the union of all partitions of
//! the scan it replaces produces, and exactly DataFusion's batch schemas. A
//! memory scan applies a `LIMIT` (`fetch`) pushed into the scan to every
//! partition, as DataFusion does, so it produces the same rows as DataFusion.
//!
//! The order of the rows is preserved only where DataFusion's plan promises
//! it to the plan above it, i.e. for scans with a single partition and an
//! output ordering (then a single thread reads the scan in order, like a
//! [`BridgeOperator`](crate::bridge::BridgeOperator) over a single
//! partition does).

use std::sync::Arc;

use datafusion_common::Result;
use datafusion_datasource::memory::MemorySourceConfig;
use datafusion_datasource::source::DataSourceExec;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::empty::EmptyExec;

use crate::operator::PhysicalOperator;

pub mod empty;
pub mod memory;

pub use empty::EmptyScanSource;
pub use memory::MemoryScanSource;

/// Convert a DataFusion scan into a native pipeline source.
///
/// Converts
///
/// * a `DataSourceExec` that scans a [`MemorySourceConfig`] into a
///   [`MemoryScanSource`],
/// * an [`EmptyExec`] into an [`EmptyScanSource`].
///
/// The source has exactly the schema of `plan`. Returns `Ok(None)` for every
/// other plan, which the caller then runs with a
/// [`BridgeOperator`](crate::bridge::BridgeOperator). See the
/// [module documentation](self) for the rows the sources produce and in which
/// order.
pub fn try_create_scan_source(
    plan: &Arc<dyn ExecutionPlan>,
) -> Result<Option<Arc<dyn PhysicalOperator>>> {
    if let Some(empty) = plan.downcast_ref::<EmptyExec>() {
        return Ok(Some(Arc::new(EmptyScanSource::from_exec(empty))));
    }
    let Some(exec) = plan.downcast_ref::<DataSourceExec>() else {
        return Ok(None);
    };
    let source: Arc<dyn PhysicalOperator> =
        if exec.data_source().is::<MemorySourceConfig>() {
            Arc::new(MemoryScanSource::try_from_exec(exec)?)
        } else {
            return Ok(None);
        };
    Ok(Some(source))
}

/// Alias of [`try_create_scan_source`].
pub use self::try_create_scan_source as try_from_execution_plan;

/// Differential tests of the native scan sources: like
/// [`crate::test_utils::assert_byte_eq_with_datafusion`], but the scans of the
/// plan are lowered to native sources.
///
/// [`PipelineQueryPlanner`](crate::planner::PipelineQueryPlanner) does not
/// lower scans natively yet (it bridges every plan), so the harness uses its
/// own query planner: it plans the query with DataFusion's
/// `DefaultPhysicalPlanner` and replaces every scan that
/// [`try_create_scan_source`] converts by a [`PipelineRootExec`] that runs
/// the native source with the pipeline executor (on `target_partitions`
/// threads). A query that is only a scan is thus executed exactly as the
/// pipeline engine executes it; DataFusion executes the operators above the
/// scans otherwise. The results are compared with
/// [`compare_outputs`](crate::test_utils::compare_outputs): exactly the same
/// schemas and Arrow data (modulo the row order, unless the query has an
/// `ORDER BY`).
///
/// [`PipelineRootExec`]: crate::bridge::PipelineRootExec
#[cfg(test)]
pub(crate) mod test_harness {
    use std::fmt;
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use async_trait::async_trait;
    use datafusion::execution::SessionStateBuilder;
    use datafusion::execution::runtime_env::RuntimeEnv;
    use datafusion::physical_planner::{DefaultPhysicalPlanner, PhysicalPlanner};
    use datafusion::prelude::{SessionConfig, SessionContext};
    use datafusion_common::tree_node::{Transformed, TransformedResult, TreeNode};
    use datafusion_common::{DataFusionError, Result};
    use datafusion_expr::LogicalPlan;
    use datafusion_physical_plan::{ExecutionPlan, displayable};
    use datafusion_session::{QueryPlanner, Session};

    use super::try_create_scan_source;
    use crate::bridge::PipelineRootExec;
    use crate::operator::PhysicalOperator;
    use crate::test_utils::{
        compare_outputs, datafusion_context, register_test_tables, run_sql,
        test_session_config,
    };

    /// Converts a scan into a native source, like [`try_create_scan_source`].
    pub(crate) type LowerScan =
        fn(&Arc<dyn ExecutionPlan>) -> Result<Option<Arc<dyn PhysicalOperator>>>;

    /// Plans queries with DataFusion and runs the scans that `lower`
    /// converts with the pipeline engine.
    struct NativeScanPlanner {
        lower: LowerScan,
        /// The number of scans lowered by the last planned query.
        lowered: Arc<AtomicUsize>,
    }

    impl fmt::Debug for NativeScanPlanner {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct("NativeScanPlanner").finish_non_exhaustive()
        }
    }

    #[async_trait]
    impl QueryPlanner for NativeScanPlanner {
        async fn create_physical_plan(
            &self,
            logical_plan: &LogicalPlan,
            session: &dyn Session,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            let plan = DefaultPhysicalPlanner::default()
                .create_physical_plan(logical_plan, session)
                .await?;
            lower_scans(plan, self.lower, &self.lowered)
        }
    }

    /// Replace every scan of `plan` that `lower` converts by a
    /// [`PipelineRootExec`] that runs the native source.
    pub(crate) fn lower_scans(
        plan: Arc<dyn ExecutionPlan>,
        lower: LowerScan,
        lowered: &AtomicUsize,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        plan.transform_up(|node| {
            let Some(source) = lower(&node)? else {
                return Ok(Transformed::no(node));
            };
            lowered.fetch_add(1, Ordering::Relaxed);
            let root: Arc<dyn ExecutionPlan> =
                Arc::new(PipelineRootExec::try_new(source, node)?);
            Ok(Transformed::yes(root))
        })
        .data()
    }

    /// A `SessionContext` whose scans run with native sources (converted by
    /// `lower`).
    fn native_scan_context(
        config: SessionConfig,
        lower: LowerScan,
        lowered: Arc<AtomicUsize>,
    ) -> SessionContext {
        let state = SessionStateBuilder::new()
            .with_config(config)
            .with_runtime_env(Arc::new(RuntimeEnv::default()))
            .with_default_features()
            .with_query_planner(Arc::new(NativeScanPlanner { lower, lowered }))
            .build();
        SessionContext::new_with_state(state)
    }

    /// Assert that `sql` produces byte-identical results on default
    /// DataFusion and with the scans lowered to native sources by `lower`,
    /// on sessions configured with `config` and set up with `setup`.
    ///
    /// Returns the number of lowered scans, which must not be zero.
    pub(crate) async fn assert_native_scans_byte_eq_with<F>(
        sql: &str,
        config: SessionConfig,
        setup: F,
        lower: LowerScan,
    ) -> usize
    where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        let lowered = Arc::new(AtomicUsize::new(0));
        let outputs = async {
            let datafusion = datafusion_context(config.clone());
            setup(&datafusion).await?;
            let native = native_scan_context(config, lower, Arc::clone(&lowered));
            setup(&native).await?;
            let expected = run_sql(&datafusion, sql).await?;
            lowered.store(0, Ordering::Relaxed);
            let actual = run_sql(&native, sql).await?;
            Ok::<_, DataFusionError>((expected, actual))
        }
        .await;
        let (expected, actual) = match outputs {
            Ok(outputs) => outputs,
            Err(e) => panic!("Failed to run query on both engines: {e}\nquery: {sql}"),
        };
        let plan = displayable(actual.plan.as_ref()).indent(true).to_string();
        if let Err(message) = compare_outputs(&expected, &actual) {
            panic!("Results differ: {message}\nquery: {sql}\nplan:\n{plan}");
        }
        let lowered = lowered.load(Ordering::Relaxed);
        assert!(
            lowered > 0,
            "No scan was lowered\nquery: {sql}\nplan:\n{plan}"
        );
        lowered
    }

    /// [`assert_native_scans_byte_eq_with`] with [`try_create_scan_source`].
    pub(crate) async fn assert_native_scans_byte_eq_with_config<F>(
        sql: &str,
        config: SessionConfig,
        setup: F,
    ) -> usize
    where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        assert_native_scans_byte_eq_with(sql, config, setup, try_create_scan_source).await
    }

    /// [`assert_native_scans_byte_eq_with_config`] with the harness's
    /// default configuration.
    pub(crate) async fn assert_native_scans_byte_eq_with_setup<F>(
        sql: &str,
        setup: F,
    ) -> usize
    where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        assert_native_scans_byte_eq_with_config(sql, test_session_config(), setup).await
    }

    /// [`assert_native_scans_byte_eq_with_setup`] with the tables of
    /// [`register_test_tables`].
    pub(crate) async fn assert_native_scans_byte_eq(sql: &str) -> usize {
        assert_native_scans_byte_eq_with_setup(sql, async |ctx: &SessionContext| {
            register_test_tables(ctx)
        })
        .await
    }
}
