// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`PhysicalPartitionedTopK`]: the top rows of every partition of a ranking
//! window function (`PartitionedTopKExec`).

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, RecordBatch, UInt32Array, UInt64Array};
use arrow::compute::{cast, take_record_batch};
use arrow::datatypes::{FieldRef, SchemaRef};
use arrow::row::{OwnedRow, RowConverter, SortField};
use datafusion_common::{Result, internal_err};
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_physical_expr::{LexOrdering, PhysicalSortExpr};
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::sorts::partitioned_topk::{
    PartitionedTopKExec, WindowFnKind,
};
use datafusion_physical_plan::sorts::sort::SortExec;
use datafusion_physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;

use crate::chunk::DataChunk;
use crate::operator::{
    ExecutionContext, GlobalOperatorState, OperatorResult, OperatorState,
    PhysicalOperator, downcast_state_mut,
};

/// Keeps the top rows of every partition of a ranking window function, as
/// `PartitionedTopKExec` does (for `ROW_NUMBER() OVER (PARTITION BY p ORDER
/// BY o) <= k`, `RANK` or `DENSE_RANK`): the rows whose `ROW_NUMBER`, `RANK`
/// or `DENSE_RANK` in their partition is at most `fetch`, in the order of
/// their partition keys and then of their order keys, followed by their
/// ranks if the node appends them (when it replaces the window).
///
/// Its input is the native sort of the rows by their partition keys and then
/// their order keys (see [`Self::try_lower`]), which a single thread emits in
/// order: the operator is not parallel, and ranks the rows as they pass,
/// like a window over sorted rows, keeping those of a rank of at most
/// `fetch`. It holds no rows itself (the native sort holds them, see
/// [`crate::sort`]), but the keys of the last row it ranked, and, while it
/// ranks the rows of a chunk, their keys in the row format, the indices and
/// the ranks of the rows that it keeps, and these rows: it reserves them in
/// the memory pool of the session before it allocates them (an estimate of
/// the keys, which it then corrects).
#[derive(Debug)]
pub struct PhysicalPartitionedTopK {
    /// The sorted input, the only child.
    children: Vec<Arc<dyn PhysicalOperator>>,
    schema: SchemaRef,
    expr: LexOrdering,
    partition_prefix_len: usize,
    fetch: usize,
    fn_kind: WindowFnKind,
    /// The column of the ranks, if they are appended.
    ranking_field: Option<FieldRef>,
}

impl PhysicalPartitionedTopK {
    /// Lower a `PartitionedTopKExec` to a [`PhysicalPartitionedTopK`] over
    /// the native sort of the rows of its input by its keys: `lower_sort`
    /// lowers the merge of the sorts of every partition of its input, which
    /// it may refuse (`Ok(None)`, e.g. if the native sort does not order the
    /// keys as DataFusion does), and then so does this. Returns `Ok(None)`
    /// for any other node, without calling `lower_sort`.
    pub fn try_lower<F>(
        plan: &Arc<dyn ExecutionPlan>,
        lower_sort: F,
    ) -> Result<Option<Self>>
    where
        F: FnOnce(&Arc<dyn ExecutionPlan>) -> Result<Option<Arc<dyn PhysicalOperator>>>,
    {
        let Some(topk) = plan.downcast_ref::<PartitionedTopKExec>() else {
            return Ok(None);
        };
        let expr = topk.expr().clone();
        let partition_prefix_len = topk.partition_prefix_len();
        if partition_prefix_len == 0 || partition_prefix_len > expr.len() {
            return internal_err!(
                "PartitionedTopKExec partitions by {partition_prefix_len} of {} keys",
                expr.len()
            );
        }
        let sort = SortExec::new(expr.clone(), Arc::clone(topk.input()))
            .with_preserve_partitioning(true);
        let sorted: Arc<dyn ExecutionPlan> =
            Arc::new(SortPreservingMergeExec::new(expr.clone(), Arc::new(sort)));
        let Some(input) = lower_sort(&sorted)? else {
            return Ok(None);
        };
        Ok(Some(Self {
            children: vec![input],
            schema: plan.schema(),
            expr,
            partition_prefix_len,
            fetch: topk.fetch(),
            fn_kind: topk.fn_kind(),
            ranking_field: topk.ranking_field().cloned(),
        }))
    }

    /// The row converter of the keys `keys` of the rows of the input.
    fn converter(&self, keys: &[PhysicalSortExpr]) -> Result<RowConverter> {
        let schema = self.children[0].schema();
        let fields = keys
            .iter()
            .map(|key| {
                Ok(SortField::new_with_options(
                    key.expr.data_type(&schema)?,
                    key.options,
                ))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(RowConverter::new(fields)?)
    }
}

/// The state of the thread of a [`PhysicalPartitionedTopK`]: the keys of the
/// last row it ranked, and its ranks.
struct RankingState {
    partition_keys: RowConverter,
    order_keys: RowConverter,
    /// The partition keys and the order keys of the last row.
    last: Option<(OwnedRow, OwnedRow)>,
    row_number: u64,
    rank: u64,
    dense_rank: u64,
    reservation: MemoryReservation,
}

impl OperatorState for RankingState {}

impl RankingState {
    /// The memory of the keys of the last row.
    fn last_memory(&self) -> usize {
        self.last.as_ref().map_or(0, |(partition, order)| {
            partition.row().as_ref().len() + order.row().as_ref().len()
        })
    }
}

/// The memory that the values of `arrays` occupy (of their slices only).
fn arrays_memory(arrays: &[ArrayRef]) -> usize {
    arrays
        .iter()
        .map(|array| {
            array
                .to_data()
                .get_slice_memory_size()
                .unwrap_or_else(|_| array.get_array_memory_size())
        })
        .sum()
}

/// An upper bound of the memory of the row format of `arrays` (of `rows`
/// rows): a variable-length value takes at most twice its bytes, plus a few
/// bytes, and every row an offset.
fn row_format_estimate(arrays: &[ArrayRef], rows: usize) -> usize {
    2 * arrays_memory(arrays) + rows * (arrays.len() * 40 + size_of::<usize>())
}

impl PhysicalOperator for PhysicalPartitionedTopK {
    fn name(&self) -> &str {
        "PhysicalPartitionedTopK"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.children
    }

    fn description(&self) -> String {
        let ranking = if self.ranking_field.is_some() {
            ", ranked=true"
        } else {
            ""
        };
        format!(
            "PhysicalPartitionedTopK: fn={:?}, fetch={}, partition=[{}], order=[{}]{ranking}",
            self.fn_kind,
            self.fetch,
            LexOrdering::new(self.expr[..self.partition_prefix_len].iter().cloned())
                .map(|ordering| ordering.to_string())
                .unwrap_or_default(),
            LexOrdering::new(self.expr[self.partition_prefix_len..].iter().cloned())
                .map(|ordering| ordering.to_string())
                .unwrap_or_default(),
        )
    }

    fn parallel_operator(&self) -> bool {
        false
    }

    fn get_operator_state(
        &self,
        ctx: &ExecutionContext,
        _gstate: &dyn GlobalOperatorState,
    ) -> Result<Box<dyn OperatorState>> {
        let reservation = MemoryConsumer::new("PhysicalPartitionedTopK")
            .register(ctx.task_ctx.memory_pool());
        Ok(Box::new(RankingState {
            partition_keys: self.converter(&self.expr[..self.partition_prefix_len])?,
            order_keys: self.converter(&self.expr[self.partition_prefix_len..])?,
            last: None,
            row_number: 0,
            rank: 0,
            dense_rank: 0,
            reservation,
        }))
    }

    fn execute(
        &self,
        _ctx: &ExecutionContext,
        input: &DataChunk,
        output: &mut DataChunk,
        _gstate: &dyn GlobalOperatorState,
        lstate: &mut dyn OperatorState,
    ) -> Result<OperatorResult> {
        let state = downcast_state_mut::<RankingState>(lstate.as_any_mut(), self.name())?;
        let batch = input.compact()?;
        let evaluate = |keys: &[PhysicalSortExpr]| -> Result<Vec<ArrayRef>> {
            keys.iter()
                .map(|key| key.expr.evaluate(&batch)?.into_array(batch.num_rows()))
                .collect()
        };
        let partition_keys = evaluate(&self.expr[..self.partition_prefix_len])?;
        let order_keys = evaluate(&self.expr[self.partition_prefix_len..])?;
        let num_rows = batch.num_rows();
        // The keys in the row format (an estimate), the indices and the
        // ranks of the rows that are kept, and these rows (at most all rows
        // of the chunk, and their ranks), before they are allocated.
        let keys = row_format_estimate(&partition_keys, num_rows)
            + row_format_estimate(&order_keys, num_rows);
        let kept_rows = num_rows * (size_of::<u32>() + 2 * size_of::<u64>())
            + crate::sort::memory_size(&batch);
        let last = state.last_memory();
        state.reservation.try_resize(last + keys + kept_rows)?;
        let partitions = state.partition_keys.convert_columns(&partition_keys)?;
        let orders = state.order_keys.convert_columns(&order_keys)?;
        drop((partition_keys, order_keys));
        state
            .reservation
            .try_resize(last + partitions.size() + orders.size() + kept_rows)?;
        let mut kept = Vec::with_capacity(num_rows);
        let mut ranks = Vec::with_capacity(num_rows);
        for row in 0..num_rows {
            let partition = partitions.row(row);
            let order = orders.row(row);
            // Compared with the previous row: of the chunk, or the last row
            // of the previous chunk.
            let previous = match row {
                0 => state
                    .last
                    .as_ref()
                    .map(|(partition, order)| (partition.row(), order.row())),
                _ => Some((partitions.row(row - 1), orders.row(row - 1))),
            };
            let (same_partition, same_order) = match previous {
                Some((last_partition, last_order)) => {
                    let same_partition = last_partition == partition;
                    (same_partition, same_partition && last_order == order)
                }
                None => (false, false),
            };
            if !same_partition {
                state.row_number = 0;
                state.rank = 0;
                state.dense_rank = 0;
            }
            state.row_number += 1;
            if !same_order {
                state.rank = state.row_number;
                state.dense_rank += 1;
            }
            let rank = match self.fn_kind {
                WindowFnKind::RowNumber => state.row_number,
                WindowFnKind::Rank => state.rank,
                WindowFnKind::DenseRank => state.dense_rank,
            };
            if rank <= self.fetch as u64 {
                kept.push(row as u32);
                ranks.push(rank);
            }
        }
        // The keys of the last row, which the next chunk compares with, and
        // the indices, ranks and rows of this chunk; the keys of this chunk
        // in the row format are released.
        if num_rows > 0 {
            state.last = Some((
                partitions.row(num_rows - 1).owned(),
                orders.row(num_rows - 1).owned(),
            ));
        }
        let last = state.last_memory();
        drop((partitions, orders));
        state.reservation.try_resize(last + kept_rows)?;
        let rows = take_record_batch(&batch, &UInt32Array::from(kept))?;
        let mut columns = rows.columns().to_vec();
        if let Some(field) = &self.ranking_field {
            let ranks: ArrayRef = Arc::new(UInt64Array::from(ranks));
            columns.push(cast(&ranks, field.data_type())?);
        }
        let options =
            arrow::array::RecordBatchOptions::new().with_row_count(Some(rows.num_rows()));
        let rows = RecordBatch::try_new_with_options(
            Arc::clone(&self.schema),
            columns,
            &options,
        )?;
        // The kept rows, until the next chunk (the indices and ranks are
        // released).
        state
            .reservation
            .try_resize(last + crate::sort::memory_size(&rows))?;
        *output = DataChunk::from_record_batch(rows);
        Ok(OperatorResult::NeedMoreInput)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use datafusion::prelude::SessionContext;

    use arrow::array::{AsArray, Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Int32Type, Schema};
    use datafusion_common::DataFusionError;
    use datafusion_execution::TaskContext;
    use datafusion_execution::memory_pool::GreedyMemoryPool;
    use datafusion_execution::runtime_env::RuntimeEnvBuilder;
    use datafusion_physical_expr::expressions::col;

    use crate::executor::Executor;
    use crate::operators::lowering_tests::{assert_lowered, count};
    use crate::sources::MemoryScanSource;
    use crate::test_utils::{register_test_tables, test_session_config};

    /// The `ROW_NUMBER() <= 1` of the rows of partitions `p` (of keys of
    /// 100 KB) ordered by `o`, over rows already sorted by `p` and `o`.
    fn topk_of_large_keys() -> Result<Arc<dyn PhysicalOperator>> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("p", DataType::Utf8, false),
            Field::new("o", DataType::Int32, false),
        ]));
        let keys = (0..20)
            .flat_map(|key| {
                std::iter::repeat_n(format!("{key:02}{}", "x".repeat(100_000)), 3)
            })
            .collect::<Vec<_>>();
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringArray::from(keys)),
                Arc::new(Int32Array::from_iter_values((0..60).map(|i| i % 3))),
            ],
        )?;
        let source =
            MemoryScanSource::try_new(&[vec![batch]], Arc::clone(&schema), None, None)?
                .with_preserve_order(true);
        let expr = LexOrdering::new(vec![
            PhysicalSortExpr::new_default(col("p", &schema)?),
            PhysicalSortExpr::new_default(col("o", &schema)?),
        ])
        .unwrap();
        Ok(Arc::new(PhysicalPartitionedTopK {
            children: vec![Arc::new(source)],
            schema,
            expr,
            partition_prefix_len: 1,
            fetch: 1,
            fn_kind: WindowFnKind::RowNumber,
            ranking_field: None,
        }))
    }

    fn task_context(memory: usize) -> Arc<TaskContext> {
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_pool(Arc::new(GreedyMemoryPool::new(memory)))
            .build_arc()
            .unwrap();
        Arc::new(TaskContext::default().with_runtime(runtime))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn the_keys_and_the_kept_rows_reserve_memory() -> Result<()> {
        let error = Executor::collect(topk_of_large_keys()?, task_context(1), 4)
            .await
            .unwrap_err();
        assert!(
            matches!(error.find_root(), DataFusionError::ResourcesExhausted(_)),
            "{error}"
        );
        // In a pool that holds them, the first row of every partition, and
        // nothing stays reserved.
        let ctx = task_context(100 << 20);
        let output =
            Executor::collect(topk_of_large_keys()?, Arc::clone(&ctx), 4).await?;
        let orders = output
            .iter()
            .flat_map(|batch| {
                batch
                    .column(1)
                    .as_primitive::<Int32Type>()
                    .values()
                    .to_vec()
            })
            .collect::<Vec<_>>();
        assert_eq!(orders, vec![0; 20]);
        assert_eq!(ctx.memory_pool().reserved(), 0);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn partitioned_topks_are_native() {
        let mut config = test_session_config();
        config.options_mut().optimizer.enable_window_topn = true;
        for sql in [
            // The ranking column is kept, or replaced by the operator.
            "SELECT * FROM (SELECT id, d, a, ROW_NUMBER() OVER (PARTITION BY d ORDER BY id) AS rn \
             FROM t1) WHERE rn <= 3",
            "SELECT id, d FROM (SELECT id, d, ROW_NUMBER() OVER (PARTITION BY d ORDER BY id DESC) AS rn \
             FROM t1) WHERE rn < 4",
            "SELECT * FROM (SELECT id, e, a, RANK() OVER (PARTITION BY e ORDER BY a NULLS FIRST) AS rk \
             FROM t1) WHERE rk <= 5",
            "SELECT * FROM (SELECT id, e, a, DENSE_RANK() OVER (PARTITION BY e ORDER BY a) AS dr \
             FROM t1) WHERE dr <= 2",
            "SELECT * FROM (SELECT k, w, name, RANK() OVER (PARTITION BY k, name ORDER BY w DESC) AS rk \
             FROM t2) WHERE rk <= 1",
            "SELECT * FROM (SELECT id, d, ROW_NUMBER() OVER (PARTITION BY d ORDER BY id) AS rn \
             FROM t_empty) WHERE rn <= 3",
        ] {
            for target_partitions in [1, 4] {
                let config = config.clone().with_target_partitions(target_partitions);
                let root = assert_lowered(sql, config, async |ctx: &SessionContext| {
                    register_test_tables(ctx)
                })
                .await;
                assert_eq!(count::<PhysicalPartitionedTopK>(&root), 1, "{sql}");
            }
        }
    }
}
