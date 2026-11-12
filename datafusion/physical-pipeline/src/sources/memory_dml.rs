// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`MemoryDmlSource`]: the `DELETE` and the `UPDATE` of a `MemTable`
//! (`MemDeleteExec` and `MemUpdateExec`).

use std::sync::Arc;

use arrow::array::{Array, BooleanArray, RecordBatch, UInt64Array};
use arrow::compute::kernels::zip::zip;
use arrow::compute::{and, filter_record_batch};
use arrow::datatypes::SchemaRef;
use datafusion_common::{Result, internal_datafusion_err};
use datafusion_datasource::memory::{MemDeleteExec, MemUpdateExec, PartitionData};
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_expr::SortExpr;
use datafusion_physical_expr_common::physical_expr::PhysicalExpr;
use datafusion_physical_plan::ExecutionPlan;
use parking_lot::Mutex;
use tokio::sync::OwnedRwLockWriteGuard;

use crate::chunk::DataChunk;
use crate::operator::{
    ExecutionContext, GlobalSourceState, LocalSourceState, PhysicalOperator,
    SourceResult, TaskInterrupt, downcast_state_mut,
};
use crate::sort::memory_size;
use crate::sources::batches::batch_overhead;

/// What a [`MemoryDmlSource`] does to the rows that its predicates match.
#[derive(Debug)]
enum Statement {
    /// Delete them (`DELETE`).
    Delete,
    /// Update them (`UPDATE`): every field of the table (with the schema
    /// `table_schema`) with a `Some` entry of `set_exprs` (one entry for
    /// every field, in field order) gets the value of its expression.
    Update {
        table_schema: SchemaRef,
        set_exprs: Vec<Option<Arc<dyn PhysicalExpr>>>,
    },
}

/// Deletes (or updates) the rows of the partitions of a `MemTable` that match
/// all predicates of a `DELETE` (or an `UPDATE`), all rows without any, and
/// then emits the number of rows that it deleted (or updated), as
/// `MemDeleteExec` (or `MemUpdateExec`) does.
///
/// It clears the declared sort order of the table first, and then changes
/// the partitions one after the other, each under its write lock (it waits
/// for the lock if a scan of the table holds it), as DataFusion does: a row
/// is deleted (or updated) if all predicates are true for it (not false or
/// null), and the `SET` expressions of an `UPDATE` are only evaluated for the
/// rows that it updates. The rows that it builds for a partition are reserved
/// in the memory pool of the session until they replace the rows of the
/// partition (the table then holds them).
#[derive(Debug)]
pub struct MemoryDmlSource {
    /// The partitions of the table, shared with the `MemTable`.
    partitions: Vec<PartitionData>,
    /// The declared sort order of the table, which the statement clears.
    sort_order: Arc<Mutex<Vec<Vec<SortExpr>>>>,
    /// The predicates of the `WHERE` clause; none matches every row.
    predicates: Vec<Arc<dyn PhysicalExpr>>,
    statement: Statement,
    /// The schema of the count that the source emits.
    schema: SchemaRef,
}

impl MemoryDmlSource {
    /// Lower a `MemDeleteExec` or a `MemUpdateExec` to a
    /// [`MemoryDmlSource`]. Returns `None` for any other node.
    pub fn try_lower(plan: &Arc<dyn ExecutionPlan>) -> Option<Self> {
        let schema = plan.schema();
        if let Some(delete) = plan.downcast_ref::<MemDeleteExec>() {
            return Some(Self {
                partitions: delete.partitions().to_vec(),
                sort_order: Arc::clone(delete.sort_order()),
                predicates: delete.predicates().to_vec(),
                statement: Statement::Delete,
                schema,
            });
        }
        let update = plan.downcast_ref::<MemUpdateExec>()?;
        Some(Self {
            partitions: update.partitions().to_vec(),
            sort_order: Arc::clone(update.sort_order()),
            predicates: update.predicates().to_vec(),
            statement: Statement::Update {
                table_schema: Arc::clone(update.table_schema()),
                set_exprs: update.set_exprs().to_vec(),
            },
            schema,
        })
    }

    /// The rows of `batch` that all predicates match (`true`, not false or
    /// null), and their number.
    fn matches(&self, batch: &RecordBatch) -> Result<(BooleanArray, usize)> {
        let mut mask: Option<BooleanArray> = None;
        for predicate in &self.predicates {
            let array = predicate.evaluate(batch)?.into_array(batch.num_rows())?;
            let array =
                array
                    .as_any()
                    .downcast_ref::<BooleanArray>()
                    .ok_or_else(|| {
                        internal_datafusion_err!("Filter did not evaluate to boolean")
                    })?;
            mask = Some(match mask {
                Some(mask) => and(&mask, array)?,
                None => array.clone(),
            });
        }
        let mask = match mask {
            // Only `true` matches: nulls become `false`.
            Some(mask) => mask.iter().map(|value| Some(value == Some(true))).collect(),
            None => BooleanArray::from(vec![true; batch.num_rows()]),
        };
        let count = mask.true_count();
        Ok((mask, count))
    }

    /// `batch` without the rows of `mask` (which are `count` rows), or with
    /// them updated.
    fn change(
        &self,
        batch: &RecordBatch,
        mask: &BooleanArray,
        count: usize,
    ) -> Result<Option<RecordBatch>> {
        if count == 0 {
            return Ok(Some(batch.clone()));
        }
        match &self.statement {
            Statement::Delete => {
                if count == batch.num_rows() {
                    return Ok(None);
                }
                let keep = arrow::compute::not(mask)?;
                Ok(Some(filter_record_batch(batch, &keep)?))
            }
            Statement::Update {
                table_schema,
                set_exprs,
            } => {
                let columns = table_schema
                    .fields()
                    .iter()
                    .enumerate()
                    .map(|(index, field)| {
                        let original =
                            batch.column_by_name(field.name()).ok_or_else(|| {
                                internal_datafusion_err!(
                                    "Column '{}' not found in batch",
                                    field.name()
                                )
                            })?;
                        let Some(set_expr) =
                            set_exprs.get(index).and_then(Option::as_ref)
                        else {
                            return Ok(Arc::clone(original));
                        };
                        // The expression is evaluated for the updated rows
                        // only (it is null for the others), so that, e.g., a
                        // division by zero of a row that is not updated does
                        // not fail.
                        let values = set_expr
                            .evaluate_selection(batch, mask)?
                            .into_array(batch.num_rows())?;
                        Ok(zip(mask, &values.as_ref(), &original.as_ref())?)
                    })
                    .collect::<Result<Vec<_>>>()?;
                Ok(Some(RecordBatch::try_new(
                    Arc::clone(table_schema),
                    columns,
                )?))
            }
        }
    }

    /// Delete (or update) the matching rows of `partition`, reserving the
    /// rows that replace them in `reservation`. Returns the number of rows
    /// that it deleted (or updated).
    fn change_partition(
        &self,
        partition: &mut Vec<RecordBatch>,
        reservation: &MemoryReservation,
    ) -> Result<u64> {
        let mut changed = 0;
        // The list of the batches that replace those of the partition (also
        // of those it keeps), before it is allocated.
        reservation.try_grow(partition.iter().map(batch_overhead).sum())?;
        let mut batches = Vec::with_capacity(partition.len());
        for batch in partition.iter().filter(|batch| batch.num_rows() > 0) {
            let (mask, count) = self.matches(batch)?;
            changed += count as u64;
            if let Some(batch) = self.change(batch, &mask, count)? {
                if count > 0 {
                    reservation.try_grow(memory_size(&batch))?;
                }
                batches.push(batch);
            }
        }
        *partition = batches;
        Ok(changed)
    }
}

/// The state of the thread of a [`MemoryDmlSource`].
struct DmlState {
    /// Whether the declared sort order of the table was cleared.
    started: bool,
    /// The next partition to change.
    next: usize,
    /// The write lock of the next partition, once the thread waited for it.
    lock: Arc<Mutex<Option<OwnedRwLockWriteGuard<Vec<RecordBatch>>>>>,
    /// The number of rows changed so far.
    count: u64,
    emitted: bool,
    reservation: MemoryReservation,
}

impl LocalSourceState for DmlState {}

impl PhysicalOperator for MemoryDmlSource {
    fn name(&self) -> &str {
        "MemoryDmlSource"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &[]
    }

    fn description(&self) -> String {
        let statement = match self.statement {
            Statement::Delete => "DELETE",
            Statement::Update { .. } => "UPDATE",
        };
        let predicates = self
            .predicates
            .iter()
            .map(ToString::to_string)
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "MemoryDmlSource: {statement}, partitions={}, predicates=[{predicates}]",
            self.partitions.len()
        )
    }

    fn is_source(&self) -> bool {
        true
    }

    fn get_local_source_state(
        &self,
        ctx: &ExecutionContext,
        _gstate: &dyn GlobalSourceState,
    ) -> Result<Box<dyn LocalSourceState>> {
        // The source is not parallel: a single thread changes the table.
        let reservation =
            MemoryConsumer::new("MemoryDmlSource").register(ctx.task_ctx.memory_pool());
        Ok(Box::new(DmlState {
            started: false,
            next: 0,
            lock: Arc::new(Mutex::new(None)),
            count: 0,
            emitted: false,
            reservation,
        }))
    }

    fn get_data(
        &self,
        _ctx: &ExecutionContext,
        _gstate: &dyn GlobalSourceState,
        lstate: &mut dyn LocalSourceState,
    ) -> Result<SourceResult> {
        let state = downcast_state_mut::<DmlState>(lstate.as_any_mut(), self.name())?;
        if state.emitted {
            return Ok(SourceResult::Finished);
        }
        if !state.started {
            // The rows that the statement leaves hold no known order.
            *self.sort_order.lock() = vec![];
            state.started = true;
        }
        while let Some(partition) = self.partitions.get(state.next) {
            let waited = state.lock.lock().take();
            let mut guard = match waited {
                Some(guard) => guard,
                None => match Arc::clone(partition).try_write_owned() {
                    Ok(guard) => guard,
                    Err(_) => {
                        // A scan of the table holds the lock: wait for it.
                        let lock = Arc::clone(&state.lock);
                        let partition = Arc::clone(partition);
                        return Ok(SourceResult::Blocked(TaskInterrupt::wait_for(
                            async move {
                                let guard = partition.write_owned().await;
                                *lock.lock() = Some(guard);
                            },
                        )));
                    }
                },
            };
            let changed = self.change_partition(&mut guard, &state.reservation);
            drop(guard);
            // The table holds the rows that replaced those of the partition.
            state.reservation.free();
            state.count += changed?;
            state.next += 1;
        }
        state.emitted = true;
        let count = RecordBatch::try_new(
            Arc::clone(&self.schema),
            vec![Arc::new(UInt64Array::from(vec![state.count]))],
        )?;
        Ok(SourceResult::Chunk(DataChunk::from_record_batch(count)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::atomic::{AtomicBool, Ordering as AtomicOrdering};
    use std::time::Duration;

    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field, Schema};
    use datafusion::datasource::MemTable;
    use datafusion::prelude::{SessionContext, col};

    use crate::operators::lowering_tests::{
        assert_lowered, assert_resources_exhausted, count, lowered,
        pipeline_context_with_memory_limit,
    };
    use crate::test_utils::{pipeline_context, test_session_config};

    /// `t`: ids, values (with nulls and zeros) and names (with nulls), in
    /// three partitions (one of them empty, one of two batches), sorted by
    /// `id`.
    fn table() -> Result<MemTable> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("v", DataType::Int64, true),
            Field::new("name", DataType::Utf8, true),
        ]));
        let batch = |ids: std::ops::Range<i64>| -> Result<RecordBatch> {
            Ok(RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from_iter_values(ids.clone())),
                    Arc::new(Int64Array::from_iter(
                        ids.clone().map(|i| (i % 7 != 0).then_some((i * 37) % 11)),
                    )),
                    Arc::new(StringArray::from_iter(
                        ids.map(|i| (i % 5 != 0).then(|| format!("n{}", i % 3))),
                    )),
                ],
            )?)
        };
        let table = MemTable::try_new(
            Arc::clone(&schema),
            vec![
                vec![batch(0..300)?, batch(300..310)?],
                vec![],
                vec![batch(310..1000)?],
            ],
        )?;
        Ok(table.with_sort_order(vec![vec![col("id").sort(true, false)]]))
    }

    async fn register_table(ctx: &SessionContext) -> Result<()> {
        ctx.register_table("t", Arc::new(table()?))?;
        Ok(())
    }

    const STATEMENTS: [&str; 9] = [
        "DELETE FROM t WHERE v > 5",
        "DELETE FROM t WHERE v IS NULL OR id % 3 = 0",
        // A null predicate (of a null name) deletes nothing.
        "DELETE FROM t WHERE v > 2 AND name <> 'n1'",
        "DELETE FROM t WHERE id > 10000",
        "DELETE FROM t",
        "UPDATE t SET v = v * 2, name = 'changed' WHERE id % 5 = 1",
        // The division is evaluated only for the rows that are updated.
        "UPDATE t SET v = 100 / v WHERE v <> 0",
        "UPDATE t SET v = NULL WHERE name IS NULL",
        "UPDATE t SET name = concat(name, '!')",
    ];

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn deletes_and_updates_of_memory_tables_are_native() {
        for statement in STATEMENTS {
            for target_partitions in [1, 4] {
                let config =
                    test_session_config().with_target_partitions(target_partitions);
                // The same counts.
                let root =
                    assert_lowered(statement, config.clone(), register_table).await;
                assert_eq!(count::<MemoryDmlSource>(&root), 1, "{statement}");
                // The same rows afterwards.
                let setup = async |ctx: &SessionContext| {
                    register_table(ctx).await?;
                    ctx.sql(statement).await?.collect().await?;
                    Ok(())
                };
                assert_lowered("SELECT * FROM t ORDER BY id", config, setup).await;
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_statement_waits_for_the_scans_of_the_table() -> Result<()> {
        let ctx = pipeline_context(test_session_config());
        let table = Arc::new(table()?);
        ctx.register_table("t", Arc::clone(&table) as _)?;
        let root = lowered(&ctx, "DELETE FROM t WHERE v > 5").await?;
        assert_eq!(count::<MemoryDmlSource>(&root), 1);
        // A scan holds the lock of the last partition.
        let scan = table.batches[2].read().await;
        let finished = AtomicBool::new(false);
        let delete = async {
            let count = ctx.sql("DELETE FROM t WHERE v > 5").await?.collect().await;
            finished.store(true, AtomicOrdering::SeqCst);
            count
        };
        let release = async {
            tokio::time::sleep(Duration::from_millis(100)).await;
            assert!(!finished.load(AtomicOrdering::SeqCst));
            drop(scan);
        };
        let (count, ()) = futures::join!(delete, release);
        let count = count?;
        let count = count[0]
            .column(0)
            .as_any()
            .downcast_ref::<UInt64Array>()
            .unwrap()
            .value(0);
        let left = ctx
            .sql("SELECT count(*) FROM t WHERE v > 5")
            .await?
            .collect()
            .await?;
        let left = left[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .value(0);
        assert!(count > 0);
        assert_eq!(left, 0);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn the_rows_that_replace_those_of_a_partition_reserve_memory() -> Result<()> {
        let statement = "UPDATE t SET name = 'changed' WHERE id % 2 = 0";
        let ctx = pipeline_context_with_memory_limit(1);
        register_table(&ctx).await?;
        assert_eq!(
            count::<MemoryDmlSource>(&lowered(&ctx, statement).await?),
            1
        );
        assert_resources_exhausted(&ctx, statement).await;
        // Also when no row changes: the batches of a partition are listed
        // again.
        assert_resources_exhausted(&ctx, "DELETE FROM t WHERE id > 100000").await;
        // The first partition, whose rows did not fit, is unchanged.
        let changed = ctx
            .sql("SELECT * FROM t WHERE name = 'changed'")
            .await?
            .collect()
            .await?;
        assert_eq!(changed.iter().map(RecordBatch::num_rows).sum::<usize>(), 0);
        // In a pool that holds them, nothing stays reserved.
        let ctx = pipeline_context_with_memory_limit(100 << 20);
        register_table(&ctx).await?;
        ctx.sql(statement).await?.collect().await?;
        assert_eq!(ctx.runtime_env().memory_pool.reserved(), 0);
        Ok(())
    }
}
