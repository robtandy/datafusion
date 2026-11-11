// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`PhysicalAsyncFunc`]: the evaluation of asynchronous scalar functions
//! (`AsyncFuncExec`, below the projections, filters and aggregations with
//! async UDFs).

use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::{Result, internal_err};
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_physical_expr::async_scalar_function::AsyncFuncExpr;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::async_func::AsyncFuncExec;
use parking_lot::Mutex;

use crate::chunk::DataChunk;
use crate::operator::{
    ExecutionContext, GlobalOperatorState, OperatorResult, OperatorState,
    PhysicalOperator, TaskInterrupt, downcast_state_mut,
};

/// Evaluates asynchronous scalar functions (async UDFs) on the rows of its
/// input and appends their results to them, as `AsyncFuncExec` does: its
/// output has the columns of its input, followed by one column for every
/// function (`__async_fn_0`, ...), which the projection, filter or
/// aggregation above it reads.
///
/// For every chunk of its input, a thread evaluates all functions, one after
/// the other, with `AsyncFuncExpr::invoke_with_args` (which calls the async
/// UDF on batches of its ideal batch size), in a future that the task of the
/// thread waits for ([`TaskInterrupt::wait_for`]), without blocking its
/// thread: the threads of the pipeline evaluate the functions of their
/// chunks concurrently. The memory of the rows that a thread evaluates, and
/// of their results, is reserved in the memory pool of the session.
#[derive(Debug)]
pub struct PhysicalAsyncFunc {
    /// The input, the only child.
    children: Vec<Arc<dyn PhysicalOperator>>,
    /// The functions, in the order of their output columns.
    functions: Vec<Arc<AsyncFuncExpr>>,
    schema: SchemaRef,
}

impl PhysicalAsyncFunc {
    /// Evaluate `functions` on the rows of `input`, appending their results
    /// to them: the output has the schema `schema`.
    pub fn try_new(
        input: Arc<dyn PhysicalOperator>,
        functions: Vec<Arc<AsyncFuncExpr>>,
        schema: SchemaRef,
    ) -> Result<Self> {
        let input_fields = input.schema().fields().len();
        if schema.fields().len() != input_fields + functions.len()
            || schema.fields()[..input_fields] != input.schema().fields()[..]
        {
            return internal_err!(
                "PhysicalAsyncFunc appends the results of {} functions to the columns of its input, not {schema}",
                functions.len()
            );
        }
        Ok(Self {
            children: vec![input],
            functions,
            schema,
        })
    }

    /// Lower an `AsyncFuncExec` to a [`PhysicalAsyncFunc`], whose input is
    /// `lower_input` of the input of the node. Returns `Ok(None)` for any
    /// other node, without calling `lower_input`.
    pub fn try_lower<F>(
        plan: &Arc<dyn ExecutionPlan>,
        lower_input: F,
    ) -> Result<Option<Self>>
    where
        F: FnOnce(&Arc<dyn ExecutionPlan>) -> Result<Arc<dyn PhysicalOperator>>,
    {
        let Some(exec) = plan.downcast_ref::<AsyncFuncExec>() else {
            return Ok(None);
        };
        // The roots of the expressions of the node are its functions.
        let mut functions = vec![];
        let mut unexpected = None;
        plan.apply_expressions(&mut |expr| {
            match expr.downcast_ref::<AsyncFuncExpr>() {
                Some(function) => functions.push(Arc::new(function.clone())),
                None => unexpected = Some(expr.to_string()),
            }
            Ok(TreeNodeRecursion::Continue)
        })?;
        if let Some(expr) = unexpected {
            return internal_err!(
                "AsyncFuncExec evaluates an expression that is not an async function: {expr}"
            );
        }
        let input = lower_input(exec.input())?;
        Self::try_new(input, functions, plan.schema()).map(Some)
    }

    /// The functions that the operator evaluates.
    pub fn functions(&self) -> &[Arc<AsyncFuncExpr>] {
        &self.functions
    }
}

/// The rows of the chunk that a thread evaluates, once the future of their
/// evaluation completed.
type Evaluated = Arc<Mutex<Option<Result<RecordBatch>>>>;

/// The state of a thread of a [`PhysicalAsyncFunc`]: the evaluation of the
/// chunk it waits for, and the memory of its rows.
struct AsyncFuncState {
    evaluated: Option<Evaluated>,
    reservation: MemoryReservation,
}

impl OperatorState for AsyncFuncState {}

/// The rows of `batch`, followed by the results of `functions` on them, with
/// the schema `schema`.
async fn evaluate(
    batch: RecordBatch,
    functions: Vec<Arc<AsyncFuncExpr>>,
    schema: SchemaRef,
    options: Arc<ConfigOptions>,
) -> Result<RecordBatch> {
    let mut columns = batch.columns().to_vec();
    for function in &functions {
        let result = function
            .invoke_with_args(&batch, Arc::clone(&options))
            .await?;
        columns.push(result.to_array(batch.num_rows())?);
    }
    Ok(RecordBatch::try_new(schema, columns)?)
}

impl PhysicalOperator for PhysicalAsyncFunc {
    fn name(&self) -> &str {
        "PhysicalAsyncFunc"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.children
    }

    fn description(&self) -> String {
        let functions = self
            .functions
            .iter()
            .map(|function| function.to_string())
            .collect::<Vec<_>>()
            .join(", ");
        format!("PhysicalAsyncFunc: async_expr=[{functions}]")
    }

    fn get_operator_state(
        &self,
        ctx: &ExecutionContext,
        _gstate: &dyn GlobalOperatorState,
    ) -> Result<Box<dyn OperatorState>> {
        let reservation =
            MemoryConsumer::new("PhysicalAsyncFunc").register(ctx.task_ctx.memory_pool());
        Ok(Box::new(AsyncFuncState {
            evaluated: None,
            reservation,
        }))
    }

    fn execute(
        &self,
        ctx: &ExecutionContext,
        input: &DataChunk,
        output: &mut DataChunk,
        _gstate: &dyn GlobalOperatorState,
        lstate: &mut dyn OperatorState,
    ) -> Result<OperatorResult> {
        let state =
            downcast_state_mut::<AsyncFuncState>(lstate.as_any_mut(), self.name())?;
        // The rows that the thread emitted before were handed on.
        if let Some(evaluated) = state.evaluated.take() {
            // The executor calls the operator again with the same input once
            // the evaluation completed.
            let Some(result) = evaluated.lock().take() else {
                return internal_err!(
                    "PhysicalAsyncFunc was called before the evaluation of its chunk completed"
                );
            };
            let batch = result?;
            state
                .reservation
                .try_resize(batch.get_array_memory_size())?;
            *output = DataChunk::from_record_batch(batch);
            return Ok(OperatorResult::NeedMoreInput);
        }
        let batch = input.compact()?;
        state
            .reservation
            .try_resize(batch.get_array_memory_size())?;
        let evaluated: Evaluated = Arc::default();
        let slot = Arc::clone(&evaluated);
        let functions = self.functions.clone();
        let schema = Arc::clone(&self.schema);
        let options = Arc::clone(ctx.task_ctx.session_config().options());
        state.evaluated = Some(evaluated);
        Ok(OperatorResult::Blocked(TaskInterrupt::wait_for(
            async move {
                let result = evaluate(batch, functions, schema, options).await;
                *slot.lock() = Some(result);
            },
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::{AtomicUsize, Ordering};

    use arrow::array::{Array, ArrayRef};
    use arrow::compute::kernels::numeric::add;
    use arrow::datatypes::DataType;
    use async_trait::async_trait;
    use datafusion::prelude::SessionContext;
    use datafusion_expr::async_udf::{AsyncScalarUDF, AsyncScalarUDFImpl};
    use datafusion_expr::{
        ColumnarValue, ScalarFunctionArgs, ScalarUDFImpl, Signature, Volatility,
    };

    use crate::operators::lowering_tests::{
        assert_lowered, assert_resources_exhausted, count, lowered,
        pipeline_context_with_memory_limit,
    };
    use crate::test_utils::{register_test_tables, test_session_config};

    /// `async_add(x, y)`: `x + y` of two `BIGINT`s, computed after yielding
    /// to the runtime, in batches of `batch_size` rows (if any).
    #[derive(Debug, PartialEq, Eq, Hash)]
    struct AsyncAdd {
        signature: Signature,
        batch_size: Option<usize>,
    }

    /// How many times `async_add` was called.
    static CALLS: AtomicUsize = AtomicUsize::new(0);

    impl AsyncAdd {
        fn udf(batch_size: Option<usize>) -> datafusion_expr::ScalarUDF {
            AsyncScalarUDF::new(Arc::new(Self {
                signature: Signature::exact(
                    vec![DataType::Int64, DataType::Int64],
                    Volatility::Volatile,
                ),
                batch_size,
            }))
            .into_scalar_udf()
        }
    }

    impl ScalarUDFImpl for AsyncAdd {
        fn name(&self) -> &str {
            "async_add"
        }

        fn signature(&self) -> &Signature {
            &self.signature
        }

        fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
            Ok(DataType::Int64)
        }

        fn invoke_with_args(&self, _args: ScalarFunctionArgs) -> Result<ColumnarValue> {
            internal_err!("async_add is an async function")
        }
    }

    #[async_trait]
    impl AsyncScalarUDFImpl for AsyncAdd {
        fn ideal_batch_size(&self) -> Option<usize> {
            self.batch_size
        }

        async fn invoke_async_with_args(
            &self,
            args: ScalarFunctionArgs,
        ) -> Result<ColumnarValue> {
            CALLS.fetch_add(1, Ordering::Relaxed);
            tokio::task::yield_now().await;
            let rows = args.number_rows;
            let arrays = args
                .args
                .iter()
                .map(|arg| arg.to_array(rows))
                .collect::<Result<Vec<ArrayRef>>>()?;
            if let Some(batch_size) = self.batch_size {
                assert!(rows <= batch_size, "{rows} rows in batches of {batch_size}");
            }
            let sum = add(&arrays[0], &arrays[1])?;
            assert_eq!(sum.len(), rows);
            Ok(ColumnarValue::Array(sum))
        }
    }

    fn setup(ctx: &SessionContext, batch_size: Option<usize>) -> Result<()> {
        register_test_tables(ctx)?;
        ctx.register_udf(AsyncAdd::udf(batch_size));
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn async_functions_are_native() {
        for sql in [
            // In a projection.
            "SELECT id, async_add(id, b) AS s FROM t1",
            "SELECT id, async_add(a, 1) AS x, async_add(b, a) AS y FROM t1 WHERE e",
            "SELECT async_add(1, 2) AS three FROM t1",
            // In a filter.
            "SELECT id, b FROM t1 WHERE async_add(id, b) > 1000",
            // In an aggregation.
            "SELECT d, SUM(async_add(id, b)) AS s, COUNT(*) AS n FROM t1 GROUP BY d",
            // Over no rows.
            "SELECT async_add(id, b) FROM t_empty",
            // Below a sort and a join.
            "SELECT t2.name, async_add(t1.id, t2.w) AS v FROM t1 JOIN t2 ON t1.a = t2.k \
             ORDER BY v, t2.name",
        ] {
            for batch_size in [None, Some(7)] {
                for target_partitions in [1, 4] {
                    let config =
                        test_session_config().with_target_partitions(target_partitions);
                    let calls = CALLS.load(Ordering::Relaxed);
                    let root =
                        assert_lowered(sql, config, async |ctx: &SessionContext| {
                            setup(ctx, batch_size)
                        })
                        .await;
                    assert_eq!(count::<PhysicalAsyncFunc>(&root), 1, "{sql}");
                    if !sql.contains("t_empty") {
                        assert!(CALLS.load(Ordering::Relaxed) > calls, "{sql}");
                    }
                }
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn the_rows_that_a_thread_evaluates_reserve_memory() -> Result<()> {
        let sql = "SELECT async_add(id, b) FROM t1";
        let ctx = pipeline_context_with_memory_limit(1000);
        setup(&ctx, None)?;
        let root = lowered(&ctx, sql).await?;
        assert_eq!(count::<PhysicalAsyncFunc>(&root), 1);
        assert_resources_exhausted(&ctx, sql).await;
        Ok(())
    }
}
