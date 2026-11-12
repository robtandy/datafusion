// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`ExplainSource`]: the plans that an `EXPLAIN` shows (an `ExplainExec`).

use std::sync::Arc;

use arrow::array::{RecordBatch, StringBuilder};
use arrow::datatypes::SchemaRef;
use datafusion_common::Result;
use datafusion_common::display::StringifiedPlan;
use datafusion_execution::memory_pool::MemoryConsumer;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::explain::ExplainExec;

use super::batches::BatchChunks;
use crate::operator::{
    ExecutionContext, GlobalSourceState, LocalSourceState, PhysicalOperator,
    SourceResult, downcast_state_mut,
};

/// A source that produces the rows of an `ExplainExec`: one row for every
/// plan that the `EXPLAIN` shows (its type and its text), exactly as
/// `ExplainExec` produces them.
///
/// The plans shown are those that the statement displays (all of them in
/// verbose mode, otherwise the final logical and physical plans), and a plan
/// whose text is the text of the plan shown before it, and that a
/// non-verbose `EXPLAIN` would not show, is shown as `SAME TEXT AS ABOVE`.
#[derive(Debug)]
pub struct ExplainSource {
    /// The rows of the `EXPLAIN`.
    batch: RecordBatch,
}

impl ExplainSource {
    /// The source of the rows of `exec`.
    pub fn try_new(exec: &ExplainExec) -> Result<Self> {
        let schema = exec.schema();
        let plans = exec
            .stringified_plans()
            .iter()
            .filter(|plan| plan.should_display(exec.verbose()))
            .collect::<Vec<_>>();
        let mut types = StringBuilder::with_capacity(plans.len(), 1024);
        let mut texts = StringBuilder::with_capacity(plans.len(), 1024);
        let mut previous: Option<&StringifiedPlan> = None;
        for plan in plans {
            types.append_value(plan.plan_type.to_string());
            match previous {
                // Like `ExplainExec`, which shows a plan whose text is the
                // text above it only if a non-verbose `EXPLAIN` shows it.
                Some(previous)
                    if previous.plan == plan.plan && !plan.should_display(false) =>
                {
                    texts.append_value("SAME TEXT AS ABOVE");
                }
                _ => texts.append_value(&*plan.plan),
            }
            previous = Some(plan);
        }
        let batch = RecordBatch::try_new(
            schema,
            vec![Arc::new(types.finish()), Arc::new(texts.finish())],
        )?;
        Ok(Self { batch })
    }

    /// Lower `plan` to an [`ExplainSource`] if it is an `ExplainExec`.
    pub fn try_lower(plan: &Arc<dyn ExecutionPlan>) -> Result<Option<Self>> {
        plan.downcast_ref::<ExplainExec>()
            .map(Self::try_new)
            .transpose()
    }
}

/// Local source state of an [`ExplainSource`]: the rows it emits.
struct ExplainLocalState {
    chunks: Option<BatchChunks>,
}

impl LocalSourceState for ExplainLocalState {}

impl PhysicalOperator for ExplainSource {
    fn name(&self) -> &str {
        "ExplainSource"
    }

    fn schema(&self) -> SchemaRef {
        self.batch.schema()
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &[]
    }

    fn description(&self) -> String {
        format!("ExplainSource: plans={}", self.batch.num_rows())
    }

    fn is_source(&self) -> bool {
        true
    }

    fn get_local_source_state(
        &self,
        _ctx: &ExecutionContext,
        _gstate: &dyn GlobalSourceState,
    ) -> Result<Box<dyn LocalSourceState>> {
        Ok(Box::new(ExplainLocalState { chunks: None }))
    }

    fn get_data(
        &self,
        ctx: &ExecutionContext,
        _gstate: &dyn GlobalSourceState,
        lstate: &mut dyn LocalSourceState,
    ) -> Result<SourceResult> {
        let lstate =
            downcast_state_mut::<ExplainLocalState>(lstate.as_any_mut(), self.name())?;
        let chunks = match &mut lstate.chunks {
            Some(chunks) => chunks,
            None => {
                // The source is not parallel: a single thread emits the rows.
                let reservation = MemoryConsumer::new("ExplainSource")
                    .register(ctx.task_ctx.memory_pool());
                let mut chunks = BatchChunks::new(reservation);
                chunks.set(self.batch.clone())?;
                lstate.chunks.insert(chunks)
            }
        };
        Ok(match chunks.next_chunk() {
            Some(chunk) => SourceResult::Chunk(chunk),
            None => SourceResult::Finished,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use datafusion_physical_plan::collect;

    use crate::bridge::{PipelineRootExec, bridged_plans};
    use crate::operators::lowering_tests::count;
    use crate::test_utils::{
        pipeline_context, register_test_tables, test_session_config,
    };

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn explains_are_native() -> Result<()> {
        for sql in [
            "EXPLAIN SELECT 1",
            "EXPLAIN SELECT a, COUNT(*) FROM t1 GROUP BY a",
            "EXPLAIN VERBOSE SELECT a FROM t1 WHERE a > 3 ORDER BY a",
            "EXPLAIN FORMAT TREE SELECT * FROM t1 JOIN t2 ON t1.a = t2.k",
            "EXPLAIN FORMAT INDENT SELECT * FROM generate_series(1, 3)",
        ] {
            for config in [
                test_session_config(),
                test_session_config().with_target_partitions(1),
            ] {
                let ctx = pipeline_context(config);
                register_test_tables(&ctx)?;
                let plan = ctx.sql(sql).await?.create_physical_plan().await?;
                let root = plan.downcast_ref::<PipelineRootExec>().unwrap();
                assert!(bridged_plans(root.operator()).is_empty(), "{sql}");
                assert_eq!(count::<ExplainSource>(root.operator()), 1, "{sql}");
                // The rows of DataFusion's `ExplainExec` of the statement.
                assert!(root.input().is::<ExplainExec>());
                let expected = collect(Arc::clone(root.input()), ctx.task_ctx()).await?;
                let actual = collect(Arc::clone(&plan), ctx.task_ctx()).await?;
                assert_eq!(actual, expected, "{sql}");
                assert!(actual[0].num_rows() > 0, "{sql}");
            }
        }
        Ok(())
    }

    #[test]
    fn plans_that_repeat_the_plan_above_are_not_shown_again() -> Result<()> {
        use arrow::array::AsArray;
        use datafusion_common::display::PlanType;
        use datafusion_expr::LogicalPlan;

        let schema = LogicalPlan::explain_schema();
        let plans = vec![
            StringifiedPlan::new(PlanType::InitialLogicalPlan, "plan a"),
            StringifiedPlan::new(PlanType::FinalLogicalPlan, "plan a"),
            StringifiedPlan::new(PlanType::InitialPhysicalPlan, "plan b"),
            StringifiedPlan::new(
                PlanType::OptimizedPhysicalPlan {
                    optimizer_name: "rule".to_string(),
                },
                "plan b",
            ),
            StringifiedPlan::new(PlanType::FinalPhysicalPlan, "plan b"),
        ];
        let texts = |verbose| -> Result<Vec<String>> {
            let exec = ExplainExec::new(Arc::clone(&schema), plans.clone(), verbose);
            let source = ExplainSource::try_new(&exec)?;
            Ok(source
                .batch
                .column(1)
                .as_string::<i32>()
                .iter()
                .map(|text| text.unwrap().to_string())
                .collect())
        };
        assert_eq!(texts(false)?, vec!["plan a", "plan b"]);
        assert_eq!(
            texts(true)?,
            vec!["plan a", "plan a", "plan b", "SAME TEXT AS ABOVE", "plan b"]
        );
        Ok(())
    }
}
