// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`EmptyScanSource`]: the source of `EmptyExec`.

use std::sync::Arc;

use arrow::datatypes::SchemaRef;
use datafusion_common::Result;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::empty::EmptyExec;

use crate::operator::{
    ExecutionContext, GlobalSourceState, LocalSourceState, PhysicalOperator, SourceResult,
};

/// A source that produces no rows, with a given schema.
///
/// It replaces DataFusion's [`EmptyExec`] (an empty relation, i.e. the old
/// `EmptyExec` with `produce_one_row = false`), which the optimizer plans
/// e.g. for `WHERE false` or `LIMIT 0`. The one-row relation
/// (`PlaceholderRowExec`) is a different operator.
#[derive(Debug)]
pub struct EmptyScanSource {
    schema: SchemaRef,
}

impl EmptyScanSource {
    /// Create a source without rows whose schema is `schema`.
    pub fn new(schema: SchemaRef) -> Self {
        Self { schema }
    }

    /// Create the source of `plan`, with exactly its schema.
    pub fn from_exec(plan: &EmptyExec) -> Self {
        Self::new(plan.schema())
    }
}

/// Global source state of an [`EmptyScanSource`]: there is nothing to share,
/// a single thread is enough.
struct EmptyScanGlobalState;

impl GlobalSourceState for EmptyScanGlobalState {
    fn max_threads(&self) -> usize {
        1
    }
}

impl PhysicalOperator for EmptyScanSource {
    fn name(&self) -> &str {
        "EmptyScanSource"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &[]
    }

    fn is_source(&self) -> bool {
        true
    }

    fn parallel_source(&self) -> bool {
        true
    }

    fn get_global_source_state(
        &self,
        _ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalSourceState>> {
        Ok(Arc::new(EmptyScanGlobalState))
    }

    fn get_data(
        &self,
        _ctx: &ExecutionContext,
        _gstate: &dyn GlobalSourceState,
        _lstate: &mut dyn LocalSourceState,
    ) -> Result<SourceResult> {
        Ok(SourceResult::Finished)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashMap;

    use arrow::array::{Int32Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::datasource::MemTable;
    use datafusion::prelude::SessionContext;
    use datafusion_execution::TaskContext;

    use crate::executor::Executor;
    use crate::operator::EmptyState;
    use crate::sources::test_harness::{
        assert_native_scans_byte_eq, assert_native_scans_byte_eq_with_setup,
    };

    fn schema() -> SchemaRef {
        let metadata = HashMap::from([("key".to_string(), "value".to_string())]);
        Arc::new(
            Schema::new(vec![
                Field::new("a", DataType::Int32, false).with_metadata(metadata.clone()),
                Field::new("b", DataType::Utf8View, true),
            ])
            .with_metadata(metadata),
        )
    }

    #[tokio::test]
    async fn produces_no_rows() -> Result<()> {
        let exec = EmptyExec::new(schema()).with_partitions(3);
        let source = EmptyScanSource::from_exec(&exec);
        assert_eq!(source.schema(), schema());
        assert_eq!(source.name(), "EmptyScanSource");
        assert_eq!(source.description(), "EmptyScanSource");
        assert!(source.is_source() && source.children().is_empty());
        let ctx = ExecutionContext::new(Arc::new(TaskContext::default()), 0, 4);
        let gstate = source.get_global_source_state(&ctx)?;
        assert_eq!(gstate.max_threads(), 1);
        assert!(matches!(
            source.get_data(&ctx, gstate.as_ref(), &mut EmptyState)?,
            SourceResult::Finished
        ));
        let output =
            Executor::collect(Arc::new(source), Arc::new(TaskContext::default()), 4)
                .await?;
        assert!(output.is_empty());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_empty_relations() {
        for sql in [
            "SELECT * FROM t1 WHERE false",
            "SELECT a, h FROM t1 LIMIT 0",
            "SELECT * FROM t1 WHERE 1 = 2 ORDER BY id",
            "SELECT COUNT(*), SUM(a) FROM t1 WHERE false",
            "SELECT t2.k, t1.id FROM t2 LEFT JOIN (SELECT * FROM t1 WHERE false) t1 ON t1.a = t2.k",
        ] {
            assert_native_scans_byte_eq(sql).await;
        }
        // The schema of the empty relation, with its metadata, is the
        // schema of the query.
        let batch = RecordBatch::try_new(
            schema(),
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(arrow::array::StringViewArray::from(vec![Some("x"), None])),
            ],
        )
        .unwrap();
        let setup = async |ctx: &SessionContext| {
            ctx.register_table(
                "meta",
                Arc::new(MemTable::try_new(schema(), vec![vec![batch.clone()]])?),
            )?;
            Ok(())
        };
        for sql in [
            "SELECT * FROM meta WHERE false",
            "SELECT b FROM meta LIMIT 0",
        ] {
            assert_native_scans_byte_eq_with_setup(sql, &setup).await;
        }
    }
}
