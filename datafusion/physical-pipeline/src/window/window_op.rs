// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`PhysicalWindow`]: a parallel, partitioning and sorting window operator.
//!
//! A window function is evaluated on the rows of each *window partition*
//! (the rows with equal `PARTITION BY` keys), in the order of its `ORDER BY`
//! keys. DataFusion's `WindowAggExec` and `BoundedWindowAggExec` expect their
//! input partitioned (by a hash `RepartitionExec`) and sorted (by a
//! `SortExec`) accordingly; the window operator replaces them together with
//! these nodes, and partitions and sorts its input itself:
//!
//! ```text
//! MetaPipeline 0 (sink: ...) children: [1]
//!   Pipeline 0: PhysicalWindow -> ... depends on meta pipelines [1]
//! MetaPipeline 1 (sink: PhysicalWindow)
//!   Pipeline 1: input -> PhysicalWindow
//!   Pipeline 2: WindowPartitionSort -> PhysicalWindow depends on pipelines [1]
//! ```
//!
//! * **Sink** (pipeline 1): every thread compacts the chunks it pushes and
//!   distributes their rows over `2^k` radix *buckets* by the hash of their
//!   partition keys, so that all rows of a window partition are in one
//!   bucket. It sorts the rows of each bucket (by the partition keys and the
//!   order of the rows within the partitions, see below) into sorted runs,
//!   once it buffered many of them and when it completes (`combine`), which
//!   hands the runs to the global sink state.
//! * **Parallel finalize** (pipeline 2): the executor finalizes a sink in a
//!   single call, so the window's sink meta pipeline has a second pipeline,
//!   whose `WindowPartitionSort` source runs on all threads once all input
//!   pipelines completed: every `get_data` claims a bucket, merges its runs
//!   (see [`RunMerger`]) and finds its window partitions (its runs of rows
//!   with equal partition keys, see `evaluate_partition_ranges`). `finalize`
//!   sorts the buckets that were not sorted (if any), and returns
//!   [`SinkFinalize::NoOutputPossible`] without rows.
//! * **Source** (pipeline 0): the threads claim the sorted buckets; for every
//!   window partition, they evaluate the window expressions with
//!   [`WindowExpr::evaluate`] on a batch of the rows of the partition in
//!   their order, exactly as `WindowAggExec` does (and as
//!   `BoundedWindowAggExec` does incrementally, with the same results), and
//!   emit the rows with their window values in chunks of at most
//!   [`DEFAULT_VECTOR_SIZE`] rows.
//!
//! # Stages
//!
//! DataFusion evaluates window functions with different `PARTITION BY` or
//! `ORDER BY` clauses in separate window nodes, and stacks them without a
//! sort in between where the order of the rows that one node emits (the
//! order of its input) satisfies the next one: the upper node then reads the
//! rows in the order in which the lower one evaluated them. A window
//! operator evaluates such a stack in *stages*, on the same sorted rows: the
//! window functions of the first stage on its window partitions, then those
//! of the next stage on its partitions (of the rows with the columns of the
//! stage before), and so on. The partitions of every stage are runs of rows
//! of the sort, so the buckets are partitioned by the partition keys of the
//! stage with the fewest of them.
//!
//! Where DataFusion sorts the rows between two window nodes of several
//! partitions, for an upper node of another order whose partition keys
//! include the group keys of the window below, the upper node is a
//! *re-sorting* stage: every one of its window partitions is within a group,
//! so it sorts the rows of every group again (by its partition keys and the
//! ordering of its input) before it evaluates them, rather than partitioning
//! and sorting all rows again. Otherwise, where DataFusion sorts or merges the
//! rows between two window nodes (for windows with other partitions or
//! orders), or the partitions of the upper node are not runs of rows of the
//! sort of the lower one, the upper node is a window operator of its own,
//! which partitions and sorts the output of the window operator of the lower
//! one again.
//!
//! # Order
//!
//! The rows of a window partition are evaluated in the order in which
//! DataFusion's window node reads them: the ordering of its input (whatever
//! established it: a sort, a merge of sorted partitions, a window below or a
//! scan of sorted files), with the partition keys first if the ordering does
//! not start with them. Rows that this ordering does not order (peers) come
//! in any order, which SQL does not define.
//!
//! A window whose DataFusion node has a single output partition, i.e. whose
//! consumers may rely on the order of its output, uses a single bucket and
//! emits its rows in the order of its input, on a single thread. Its input is
//! still sorted by all threads into runs, which a single thread merges. If the
//! partition keys do not lead the ordering of its input (a
//! `BoundedWindowAggExec` in `Linear` or `PartiallySorted` mode), the window
//! sorts the rows it evaluated again, by that ordering, before it emits them.
//! A window of several output partitions emits the buckets in any order, in
//! parallel, and the rows of each window partition in their order.
//!
//! # Memory
//!
//! The window operator keeps all rows in memory (it does not spill), and
//! reserves it in the memory pool of the session: the chunks that the threads
//! buffer as they arrive (copies of their rows if they are slices of larger
//! arrays, see [`compact_to_keep`]; a chunk whose rows a thread distributes
//! over several buckets, which keep slices of it, once, until all of them
//! are sorted), what sorting them into runs and merging the runs of a bucket
//! allocates before it allocates it, the sorted buckets until their rows were
//! emitted, and the evaluated rows of a bucket until they are emitted. It
//! fails with `ResourcesExhausted` when the pool cannot hold them.
//!
//! It reads all rows of its input before it emits anything, so windows over
//! inputs that never end are left to DataFusion, whose windows stream (see
//! [`PhysicalWindow::try_lower`]).

use std::collections::VecDeque;
use std::fmt::Write;
use std::ops::Range;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow::array::{
    Array, ArrayRef, RecordBatch, RecordBatchOptions, UInt32Array, new_null_array,
};
use arrow::compute::{
    SortColumn, SortOptions, concat, concat_batches, take_record_batch,
};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow_ord::ord::make_comparator;
use datafusion_common::utils::evaluate_partition_ranges;
use datafusion_common::{
    DataFusionError, Result, internal_datafusion_err, internal_err, not_impl_err,
};
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::window::WindowExpr;
use datafusion_physical_expr::{LexOrdering, PhysicalExpr, PhysicalSortExpr};
use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::repartition::RepartitionExec;
use datafusion_physical_plan::sorts::sort::SortExec;
use datafusion_physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion_physical_plan::windows::{BoundedWindowAggExec, WindowAggExec};
use datafusion_physical_plan::{
    ChildrenPropertiesMode, ExecutionPlan, ExecutionPlanProperties, Partitioning,
    ReplaceChildrenOptions,
};
use parking_lot::Mutex;

use crate::chunk::{DEFAULT_VECTOR_SIZE, DataChunk};
use crate::join::hash_table::{hash_join_keys, radix_bits_for_threads};
use crate::operator::{
    ExecutionContext, GlobalSinkState, GlobalSourceState, LocalSinkState,
    LocalSourceState, PhysicalOperator, SinkFinalize, SinkResult, SourceResult,
    TaskInterrupt, downcast_state, downcast_state_mut,
};
use crate::pipeline::{MetaPipelineBuilder, PipelineBuilder};
use crate::sort::merge::{RunMerger, SortedRun, is_capacity_error};
use crate::sort::sort_sink::{SORT_RUN_BYTES, SORT_RUN_ROWS};
use crate::sort::{
    ReservedMemory, SortKeys, array_memory, buffered_memory, compact_to_keep,
    memory_size, metadata_memory,
};

/// Evaluates window functions, with the semantics of DataFusion's
/// `WindowAggExec` and `BoundedWindowAggExec`.
///
/// See the [module documentation](self) for its pipelines, its stages and
/// the order of its rows.
#[derive(Debug)]
pub struct PhysicalWindow {
    /// The input, the only child.
    children: Vec<Arc<dyn PhysicalOperator>>,
    spec: WindowSpec,
}

/// Everything about a window operator that does not depend on its input,
/// only on its schema.
#[derive(Debug)]
struct WindowSpec {
    /// The stages, evaluated one after the other on every group of rows.
    stages: Vec<WindowStage>,
    /// The order of the rows of a bucket: the partition keys of the first
    /// stage (unless they lead `ordering`), then `ordering`. `None` if the
    /// rows have no order (no partition keys and no ordering).
    keys: Option<SortKeys>,
    /// The leading keys of `keys` that the stages are evaluated on groups of
    /// rows of: as many as the partition keys of the stage with the fewest of
    /// them, all of whose partitions are runs of rows with equal group keys.
    group_keys: Vec<Arc<dyn PhysicalExpr>>,
    /// Whether the partition keys of the first stage lead `ordering`, so that
    /// the rows are evaluated (and emitted) in the order of `ordering`.
    partition_keys_lead: bool,
    /// Whether all rows are evaluated in one bucket and emitted in the order
    /// of the input of the window: of `keys`, or of `emit_order`.
    ordered: bool,
    /// The order in which an ordered window emits its rows if it is not the
    /// order of `keys`: the ordering of its input, which its partition keys do
    /// not lead (see the [module documentation](self)). The window sorts the
    /// rows of its single bucket by `keys` to evaluate them, and the evaluated
    /// rows by this order.
    emit_order: Option<SortKeys>,
    /// Whether the group keys can be hashed, to distribute the rows over
    /// buckets.
    hashable: bool,
    /// The output schema: the schema of the last stage.
    schema: SchemaRef,
}

/// The window functions of one DataFusion window node.
#[derive(Debug)]
struct WindowStage {
    /// The columns of the rows of the stage before (or of the input of the
    /// window, for the first stage) that the stage reads, in order (`None`
    /// for all of them): the projections between two stacked window nodes.
    projection: Option<Vec<usize>>,
    /// The schema of the rows the window functions are evaluated on.
    input_schema: SchemaRef,
    window_exprs: Vec<Arc<dyn WindowExpr>>,
    /// The partition keys (of the first window expression, like DataFusion).
    partition_by: Vec<Arc<dyn PhysicalExpr>>,
    /// Whether the partitions of the stage are the groups of rows with equal
    /// group keys.
    whole_groups: bool,
    /// For a *re-sorting* stage, the order in which it evaluates the rows of a
    /// group: its partition keys (unless they lead its ordering) and the
    /// ordering of the input of its window node. The stage sorts the rows of
    /// every group by these keys first, as DataFusion sorts the rows between
    /// the two window nodes. `None` if the stage evaluates the rows of a group
    /// in the order of the stage before.
    sort: Option<SortKeys>,
    /// The schema of the rows with the window values: the input columns,
    /// then one column per window expression.
    schema: SchemaRef,
}

impl WindowStage {
    fn try_new(
        projection: Option<Vec<usize>>,
        input_schema: &SchemaRef,
        window_exprs: Vec<Arc<dyn WindowExpr>>,
    ) -> Result<Self> {
        let Some(first) = window_exprs.first() else {
            return internal_err!(
                "PhysicalWindow requires at least one window expression"
            );
        };
        let partition_by = first.partition_by().to_vec();
        // The schema of `WindowAggExec` and `BoundedWindowAggExec`: the input
        // columns, then the window columns, with the input's metadata.
        let fields = input_schema
            .fields()
            .iter()
            .cloned()
            .chain(
                window_exprs
                    .iter()
                    .map(|expr| expr.field())
                    .collect::<Result<Vec<_>>>()?,
            )
            .collect::<Vec<_>>();
        let schema = Arc::new(Schema::new_with_metadata(
            fields,
            input_schema.metadata().clone(),
        ));
        Ok(Self {
            projection,
            input_schema: Arc::clone(input_schema),
            window_exprs,
            partition_by,
            whole_groups: false,
            sort: None,
            schema,
        })
    }

    /// Evaluate the window functions of the stage on `batch`, rows of the
    /// stage before (or of the input) that have the same group keys, in their
    /// order: the rows the stage reads, with their window values. A
    /// re-sorting stage reserves what sorting the rows allocates in
    /// `reservation` while it sorts them (see [`sorted_rows`]).
    fn evaluate(
        &self,
        batch: RecordBatch,
        reservation: &MemoryReservation,
    ) -> Result<RecordBatch> {
        let num_rows = batch.num_rows();
        let options = RecordBatchOptions::new().with_row_count(Some(num_rows));
        let batch = match &self.projection {
            Some(projection) => RecordBatch::try_new_with_options(
                Arc::clone(&self.input_schema),
                projection
                    .iter()
                    .map(|&column| Arc::clone(batch.column(column)))
                    .collect(),
                &options,
            )?,
            None => batch,
        };
        let batch = match &self.sort {
            Some(keys) => sorted_rows(keys, &self.input_schema, batch, reservation)?,
            None => batch,
        };
        let partitions: Vec<Range<usize>> = if self.whole_groups {
            std::iter::once(0..num_rows).collect()
        } else {
            let columns = partition_keys(&self.partition_by, &batch)?
                .into_iter()
                .map(|values| SortColumn {
                    values,
                    options: None,
                })
                .collect::<Vec<_>>();
            evaluate_partition_ranges(num_rows, &columns)?
        };
        let mut values =
            vec![Vec::with_capacity(partitions.len()); self.window_exprs.len()];
        for range in partitions {
            let len = range.len();
            let partition = if len == num_rows {
                batch.clone()
            } else {
                batch.slice(range.start, len)
            };
            for (expr, values) in self.window_exprs.iter().zip(&mut values) {
                let result = expr.evaluate(&partition)?;
                if result.len() != len {
                    return internal_err!(
                        "The window expression {} returned {} values for a partition of {len} rows",
                        expr.name(),
                        result.len()
                    );
                }
                values.push(result);
            }
        }
        let mut columns = batch.columns().to_vec();
        for values in values {
            columns.push(match values.as_slice() {
                [values] => Arc::clone(values),
                _ => concat(&values.iter().map(|v| v.as_ref()).collect::<Vec<_>>())?,
            });
        }
        Ok(RecordBatch::try_new_with_options(
            Arc::clone(&self.schema),
            columns,
            &options,
        )?)
    }
}

/// The rows of `batch` (with schema `schema`) sorted by `keys`, in one batch.
///
/// What sorting them allocates is reserved in `reservation` while they are
/// sorted (see [`SortKeys::sort`]), and freed once they are: the memory of the
/// sorted rows is the caller's to reserve, with the rows that it evaluates.
fn sorted_rows(
    keys: &SortKeys,
    schema: &SchemaRef,
    batch: RecordBatch,
    reservation: &MemoryReservation,
) -> Result<RecordBatch> {
    let runs = keys.sort(schema, std::slice::from_ref(&batch), None, reservation)?;
    reservation.shrink(runs.iter().map(SortedRun::memory_size).sum());
    match runs.len() {
        0 => Ok(batch),
        1 => Ok(runs
            .into_iter()
            .map(SortedRun::into_batch)
            .next()
            .unwrap_or(batch)),
        // The runs hold consecutive rows of the sort.
        _ => {
            let batches = runs
                .into_iter()
                .map(SortedRun::into_batch)
                .collect::<Vec<_>>();
            Ok(concat_batches(schema, &batches)?)
        }
    }
}

/// The values of the keys `keys` of the rows of `batch`.
fn partition_keys(
    keys: &[Arc<dyn PhysicalExpr>],
    batch: &RecordBatch,
) -> Result<Vec<ArrayRef>> {
    keys.iter()
        .map(|key| key.evaluate(batch)?.into_array(batch.num_rows()))
        .collect()
}

impl WindowSpec {
    /// A window of a single stage, see [`PhysicalWindow::try_new`].
    fn try_new(
        input_schema: &SchemaRef,
        window_exprs: Vec<Arc<dyn WindowExpr>>,
        ordering: Option<LexOrdering>,
        ordered: bool,
    ) -> Result<Self> {
        let mut stage = WindowStage::try_new(None, input_schema, window_exprs)?;
        let partition_by = distinct(&stage.partition_by);
        // The partition keys lead the order of the rows unless they are the
        // first keys of `ordering` (in any order and direction).
        let partition_keys_lead = partition_by.iter().all(|key| {
            ordering.as_ref().is_some_and(|ordering| {
                ordering
                    .iter()
                    .take(partition_by.len())
                    .any(|sort_expr| sort_expr.expr.as_ref() == key.as_ref())
            })
        });
        // An ordered window emits its rows in the order of its input: if the
        // partition keys do not lead it, the evaluated rows are sorted by it
        // again (its expressions are bound to the input of the window, whose
        // columns lead the rows with the window values).
        let emit_order = match &ordering {
            Some(ordering) if ordered && !partition_keys_lead => {
                if !SortKeys::supports(ordering, &stage.schema)? {
                    return not_impl_err!(
                        "PhysicalWindow cannot sort rows by {ordering}: the row format does not support its types"
                    );
                }
                Some(SortKeys::try_new(ordering.clone(), &stage.schema)?)
            }
            _ => None,
        };
        let sort_exprs = match partition_keys_lead {
            true => vec![],
            false => partition_by
                .iter()
                .map(|key| PhysicalSortExpr::new_default(Arc::clone(key)))
                .collect(),
        }
        .into_iter()
        .chain(ordering.into_iter().flatten());
        let keys = match LexOrdering::new(sort_exprs) {
            Some(ordering) => {
                if !SortKeys::supports(&ordering, input_schema)? {
                    return not_impl_err!(
                        "PhysicalWindow cannot sort rows by {ordering}: the row format does not support its types"
                    );
                }
                Some(SortKeys::try_new(ordering, input_schema)?)
            }
            None => None,
        };
        // The partition keys are the first keys of the sort (or there are
        // none).
        let group_keys = sort_prefix(keys.as_ref(), partition_by.len());
        if group_keys.len() != partition_by.len() {
            return internal_err!(
                "The partition keys of a PhysicalWindow do not lead the order of its rows"
            );
        }
        stage.whole_groups = true;
        let schema = Arc::clone(&stage.schema);
        Ok(Self {
            hashable: hashable(&group_keys, input_schema)?,
            stages: vec![stage],
            keys,
            group_keys,
            partition_keys_lead,
            ordered,
            emit_order,
            schema,
        })
    }

    /// This window with one more stage: the window functions
    /// `window_exprs`, on the output of the last stage, projected to the
    /// columns `projection` (all of them if `None`), with schema
    /// `input_schema`. `None` if the rows of the window partitions of the
    /// stage are not runs of rows of the sort of this window (or are not in
    /// the order DataFusion reads them).
    fn with_stage(
        mut self,
        projection: Option<Vec<usize>>,
        input_schema: &SchemaRef,
        window_exprs: Vec<Arc<dyn WindowExpr>>,
    ) -> Result<Option<Self>> {
        // The rows that DataFusion's window node reads are in the order in
        // which the first stage read them (`ordering`); the native window
        // evaluates them in the order of its sort, which is that order only
        // if the partition keys of the first stage lead it, and no stage
        // sorted the rows of the groups again.
        if !self.partition_keys_lead
            || self.stages.iter().any(|stage| stage.sort.is_some())
        {
            return Ok(None);
        }
        let mut stage = WindowStage::try_new(projection, input_schema, window_exprs)?;
        // The partition keys of the stage, as columns of the input of the
        // window: they must be the first keys of the sort, so that the
        // partitions of the stage are runs of rows of the sort.
        let Some(columns) = self.partition_columns(&stage) else {
            return Ok(None);
        };
        let prefix = sort_prefix(self.keys.as_ref(), columns.len());
        let leads = prefix.len() == columns.len()
            && prefix.iter().all(|key| {
                key.downcast_ref::<Column>()
                    .is_some_and(|key| columns.contains(&key.index()))
            });
        if !leads {
            return Ok(None);
        }
        if columns.len() < self.group_keys.len() {
            // The stage has fewer partition keys: the groups are coarser.
            self.group_keys = prefix;
            for stage in &mut self.stages {
                stage.whole_groups = distinct(&stage.partition_by).len() == columns.len();
            }
        }
        stage.whole_groups = columns.len() == self.group_keys.len();
        self.hashable = hashable(&self.group_keys, &self.stages[0].input_schema)?;
        self.schema = Arc::clone(&stage.schema);
        self.stages.push(stage);
        Ok(Some(self))
    }

    /// This window with one more, *re-sorting*, stage: the window functions
    /// `window_exprs`, on the output of the last stage projected to the
    /// columns `projection` (all of them if `None`), with schema
    /// `input_schema`, in the order `ordering` within its window partitions:
    /// the stage sorts the rows of every group by its partition keys and
    /// `ordering` before it evaluates them, as DataFusion sorts the rows
    /// between two window nodes of other orders. `None` if the window is
    /// ordered (its rows are emitted in the order of the input of its last
    /// window node), or if a window partition of the stage may span several
    /// groups: its partition keys must include the group keys.
    fn with_resorting_stage(
        mut self,
        projection: Option<Vec<usize>>,
        input_schema: &SchemaRef,
        window_exprs: Vec<Arc<dyn WindowExpr>>,
        ordering: Option<LexOrdering>,
    ) -> Result<Option<Self>> {
        if self.ordered {
            return Ok(None);
        }
        let mut stage = WindowStage::try_new(projection, input_schema, window_exprs)?;
        let Some(columns) = self.partition_columns(&stage) else {
            return Ok(None);
        };
        let covers_groups = !self.group_keys.is_empty()
            && self.group_keys.iter().all(|key| {
                key.downcast_ref::<Column>()
                    .is_some_and(|key| columns.contains(&key.index()))
            });
        if !covers_groups {
            return Ok(None);
        }
        // The order of the rows of a group: the partition keys of the stage,
        // unless they lead `ordering`, then `ordering`.
        let partition_by = distinct(&stage.partition_by);
        let partition_keys_lead = partition_by.iter().all(|key| {
            ordering.as_ref().is_some_and(|ordering| {
                ordering
                    .iter()
                    .take(partition_by.len())
                    .any(|sort_expr| sort_expr.expr.as_ref() == key.as_ref())
            })
        });
        let sort_exprs = match partition_keys_lead {
            true => vec![],
            false => partition_by
                .iter()
                .map(|key| PhysicalSortExpr::new_default(Arc::clone(key)))
                .collect(),
        }
        .into_iter()
        .chain(ordering.into_iter().flatten());
        let Some(sort) = LexOrdering::new(sort_exprs) else {
            return Ok(None);
        };
        if !SortKeys::supports(&sort, input_schema)?
            || sort.iter().any(|sort_expr| {
                sort_expr
                    .expr
                    .data_type(input_schema)
                    .map_or(true, |data_type| has_union(&data_type))
            })
        {
            return Ok(None);
        }
        stage.sort = Some(SortKeys::try_new(sort, input_schema)?);
        stage.whole_groups = columns.len() == self.group_keys.len();
        self.schema = Arc::clone(&stage.schema);
        self.stages.push(stage);
        Ok(Some(self))
    }

    /// The partition keys of `stage`, a stage on the output of the last stage,
    /// as distinct columns of the input of the window, `None` if one of them
    /// is not a column of the input of the window (e.g. a window value or a
    /// computed key).
    fn partition_columns(&self, stage: &WindowStage) -> Option<Vec<usize>> {
        let mut columns = vec![];
        for key in &stage.partition_by {
            let column = key.downcast_ref::<Column>()?;
            let column = match &stage.projection {
                Some(projection) => *projection.get(column.index())?,
                None => column.index(),
            };
            let column = self.input_column(column)?;
            if !columns.contains(&column) {
                columns.push(column);
            }
        }
        Some(columns)
    }

    /// The column of the input of the window that the column `column` of the
    /// output of the last stage passes through, `None` if it is a window
    /// value.
    fn input_column(&self, mut column: usize) -> Option<usize> {
        for (index, stage) in self.stages.iter().enumerate().rev() {
            // A window value of the stage.
            if column >= stage.input_schema.fields().len() {
                return None;
            }
            if let Some(projection) = &stage.projection {
                column = *projection.get(column)?;
            } else if index == 0 {
                break;
            }
        }
        Some(column)
    }

    /// The number of buckets of an execution on `threads` threads.
    fn num_buckets(&self, threads: usize) -> usize {
        if self.ordered || self.group_keys.is_empty() || !self.hashable || threads <= 1 {
            1
        } else {
            1 << radix_bits_for_threads(threads)
        }
    }
}

/// The distinct expressions of `exprs`, in order.
fn distinct(exprs: &[Arc<dyn PhysicalExpr>]) -> Vec<Arc<dyn PhysicalExpr>> {
    let mut distinct: Vec<Arc<dyn PhysicalExpr>> = Vec::with_capacity(exprs.len());
    for expr in exprs {
        if !distinct.iter().any(|other| other.as_ref() == expr.as_ref()) {
            distinct.push(Arc::clone(expr));
        }
    }
    distinct
}

/// The first `len` keys of the sort `keys` (fewer if there are not as many).
fn sort_prefix(keys: Option<&SortKeys>, len: usize) -> Vec<Arc<dyn PhysicalExpr>> {
    keys.map(SortKeys::ordering)
        .into_iter()
        .flat_map(|ordering| ordering.iter())
        .take(len)
        .map(|sort_expr| Arc::clone(&sort_expr.expr))
        .collect()
}

/// Whether rows can be distributed by the hash of the values of `keys`,
/// evaluated on rows with schema `schema`: there are keys, and the hash
/// function supports their types.
fn hashable(keys: &[Arc<dyn PhysicalExpr>], schema: &Schema) -> Result<bool> {
    if keys.is_empty() {
        return Ok(false);
    }
    for key in keys {
        let nulls = new_null_array(&key.data_type(schema)?, 1);
        if hash_join_keys(&[nulls], 1, &mut vec![]).is_err() {
            return Ok(false);
        }
    }
    Ok(true)
}

impl PhysicalWindow {
    /// Evaluate the window expressions `window_exprs`, which must all have the
    /// same partition keys (only the keys of the first one are used), on the
    /// rows of `input`, with the semantics of a `WindowAggExec` with these
    /// expressions. The rows of a window partition are evaluated in the order
    /// `ordering` (in any order without ordering). The output has the
    /// columns of the input, then one column per window expression.
    ///
    /// If `ordered`, all rows are evaluated on one thread and emitted in the
    /// order `ordering` (by the partition keys without ordering): the window
    /// sorts them by the partition keys and `ordering` to evaluate them, and,
    /// if the partition keys do not lead `ordering`, the evaluated rows by
    /// `ordering` again. Otherwise, the window partitions are evaluated in
    /// parallel and emitted in any order.
    pub fn try_new(
        input: Arc<dyn PhysicalOperator>,
        window_exprs: Vec<Arc<dyn WindowExpr>>,
        ordering: Option<LexOrdering>,
        ordered: bool,
    ) -> Result<Self> {
        let spec = WindowSpec::try_new(&input.schema(), window_exprs, ordering, ordered)?;
        Ok(Self {
            children: vec![input],
            spec,
        })
    }

    /// Lower a DataFusion `WindowAggExec` or `BoundedWindowAggExec`, with the
    /// nodes below it that only distribute and sort its input (and the window
    /// nodes it is stacked on), to a [`PhysicalWindow`] with the same
    /// semantics and schema. `lower_input` lowers the input of these nodes.
    ///
    /// The native window evaluates every window expression (of any window
    /// function or aggregate, built-in or user-defined, with any frame,
    /// `FILTER` and arguments) with [`WindowExpr::evaluate`] on all rows of
    /// each window partition, in the order of the input of the window node
    /// (see the [module documentation](self)), as `WindowAggExec` does:
    /// `BoundedWindowAggExec` computes the same values incrementally. It
    /// reads its input whatever established the order of the input (sorts,
    /// merges, windows below, sorted scans), and it is stacked on the native
    /// window of a window below, as a stage, as a re-sorting stage, or as a
    /// window of its own (where DataFusion sorts or merges in between). The
    /// nodes below the window that only distribute and sort its input are
    /// replaced too (the native window partitions and sorts its input itself):
    /// `SortExec`s, `SortPreservingMergeExec`s, `CoalescePartitionsExec`s and
    /// `RepartitionExec`s (round robin or hash) without a fetch that read all
    /// partitions of their input, also below the projections (of columns or
    /// of computed values) between them and the window, which are lowered
    /// with the input, below these nodes (see `rows_below`).
    ///
    /// Returns `Ok(None)` for any other node, without calling `lower_input`,
    /// and for the windows that are left to DataFusion:
    ///
    /// * windows with a `WindowStateObserver`, a callback that observes the
    ///   accumulators of DataFusion's window as its partitions close, which
    ///   the native window does not have;
    /// * windows that are not one window over all rows of their input: of no
    ///   partitions (DataFusion executes nothing), of several partitions
    ///   without partition keys or whose input is not partitioned by its
    ///   partition keys (DataFusion evaluates every partition separately), and
    ///   whose partitions are not those of their input (stale properties);
    /// * windows over inputs that never end (unbounded inputs): the native
    ///   window reads all of its input before it emits a row, while
    ///   DataFusion's window streams;
    /// * windows whose rows are ordered by keys that the Arrow row format does
    ///   not support, or by unions: the row format orders the values of a
    ///   union by their type first, also for descending keys, and its nulls
    ///   among the values of their type, so that such keys would not be in the
    ///   order of the window.
    pub fn try_lower<F>(
        plan: &Arc<dyn ExecutionPlan>,
        mut lower_input: F,
    ) -> Result<Option<Self>>
    where
        F: FnMut(&Arc<dyn ExecutionPlan>) -> Result<Arc<dyn PhysicalOperator>>,
    {
        let Some((spec, rows)) = analyze(plan)? else {
            return Ok(None);
        };
        if spec.schema != plan.schema() {
            return internal_err!(
                "PhysicalWindow would have schema {:?}, but the {} has schema {:?}",
                spec.schema,
                plan.name(),
                plan.schema()
            );
        }
        let input_schema = Arc::clone(&spec.stages[0].input_schema);
        let lowered = lower_input(&rows)?;
        if lowered.schema() != input_schema {
            return internal_err!(
                "The lowered input {} of a {} has schema {:?}, expected {:?}",
                lowered.name(),
                plan.name(),
                lowered.schema(),
                input_schema
            );
        }
        Ok(Some(Self {
            children: vec![lowered],
            spec,
        }))
    }

    /// The input of the window.
    pub fn input(&self) -> &Arc<dyn PhysicalOperator> {
        &self.children[0]
    }

    /// The window expressions of every stage (of every DataFusion window
    /// node, from the lowest one).
    pub fn window_exprs(&self) -> Vec<&[Arc<dyn WindowExpr>]> {
        self.spec
            .stages
            .iter()
            .map(|stage| stage.window_exprs.as_slice())
            .collect()
    }

    /// The order of the rows of a bucket (the partition keys, unless they
    /// lead the ordering of the window, then that ordering), `None` without
    /// order.
    pub fn sort_order(&self) -> Option<&LexOrdering> {
        self.spec.keys.as_ref().map(SortKeys::ordering)
    }

    /// Whether the window emits all rows on a single thread, in the order of
    /// [`Self::sort_order`].
    pub fn is_ordered(&self) -> bool {
        self.spec.ordered
    }

    /// Sort the buffered rows of `bucket` into runs. The memory of the runs is
    /// reserved in `reservation` instead of the memory of the rows.
    fn sort_buffered_rows(
        &self,
        bucket: &mut LocalBucket,
        reservation: &MemoryReservation,
    ) -> Result<()> {
        let Some(keys) = &self.spec.keys else {
            return Ok(());
        };
        let batches = std::mem::take(&mut bucket.batches);
        let chunks = std::mem::take(&mut bucket.chunks);
        bucket.rows = 0;
        bucket.bytes = 0;
        let reserved = std::mem::take(&mut bucket.reserved);
        let runs = keys.sort(&self.input().schema(), &batches, None, reservation)?;
        drop(batches);
        // The chunks that the rows are slices of are freed once the slices of
        // all buckets are sorted.
        drop(chunks);
        reservation.shrink(reserved);
        bucket.runs.extend(runs);
        Ok(())
    }

    /// Add `batch` to the rows of `bucket`, sorting them into a run once
    /// there are many. Its memory is reserved in `reservation`, unless it is a
    /// slice of a chunk whose memory `chunk` reserves (for its slices in all
    /// buckets of the thread).
    fn buffer(
        &self,
        bucket: &mut LocalBucket,
        batch: RecordBatch,
        chunk: Option<&Arc<MemoryReservation>>,
        reservation: &MemoryReservation,
    ) -> Result<()> {
        let bytes = buffered_memory(&batch);
        match chunk {
            Some(chunk) => bucket.chunks.push(Arc::clone(chunk)),
            None => {
                reservation.try_grow(bytes)?;
                bucket.reserved += bytes;
            }
        }
        bucket.rows += batch.num_rows();
        bucket.bytes += bytes;
        bucket.batches.push(batch);
        if bucket.rows >= SORT_RUN_ROWS || bucket.bytes >= SORT_RUN_BYTES {
            self.sort_buffered_rows(bucket, reservation)?;
        }
        Ok(())
    }

    /// Sort the rows of `input`, the rows of a bucket, and find its groups.
    ///
    /// The memory of `input` is reserved in `reservation`, and so is what
    /// merging its runs allocates, before it is allocated: the sorted bucket
    /// takes the memory of its rows and groups from `reservation` into a
    /// reservation of its own, and the rest is freed.
    fn sort_bucket(
        &self,
        input: BucketInput,
        reservation: &MemoryReservation,
    ) -> Result<SortedBucket> {
        let BucketInput {
            mut runs,
            batches,
            memory,
        } = input;
        let mut working = ReservedMemory::with_reserved(reservation, memory);
        runs.retain(|run| run.num_rows() > 0);
        let batches = match &self.spec.keys {
            // The rows of a single run are sorted already.
            Some(_) if runs.len() == 1 => {
                runs.into_iter().map(SortedRun::into_batch).collect()
            }
            Some(_) => {
                // The merge gathers all rows into new batches, and the
                // positions of the rows in their runs.
                let rows = runs.iter().map(SortedRun::num_rows).sum::<usize>();
                let gathered = runs
                    .iter()
                    .map(|run| buffered_memory(run.batch()))
                    .sum::<usize>();
                working.try_grow(gathered + rows * 2 * size_of::<(usize, usize)>())?;
                let mut merger = RunMerger::new(self.input().schema(), runs, None);
                let mut sorted = vec![];
                while let Some(batch) = merger.next_batch(usize::MAX)? {
                    sorted.push(batch);
                }
                sorted
            }
            None => batches
                .into_iter()
                .filter(|batch| batch.num_rows() > 0)
                .collect(),
        };
        let groups = self.groups(&batches, &mut working)?;
        let memory = batches.iter().map(buffered_memory).sum::<usize>()
            + groups.capacity() * size_of::<Range<usize>>()
            + (batches.len() + 1) * size_of::<usize>();
        working.keep(memory)?;
        Ok(SortedBucket::new(
            batches,
            groups,
            reservation.split(memory),
        ))
    }

    /// The groups of the sorted rows `batches`: the runs of rows with equal
    /// group keys, as ranges of rows of all batches. The group keys that
    /// evaluating them allocates (of a batch at a time) and the groups are
    /// reserved in `working` as they are allocated.
    fn groups(
        &self,
        batches: &[RecordBatch],
        working: &mut ReservedMemory<'_>,
    ) -> Result<Vec<Range<usize>>> {
        let num_rows = batches.iter().map(RecordBatch::num_rows).sum::<usize>();
        if self.spec.group_keys.is_empty() {
            return Ok((num_rows > 0).then_some(0..num_rows).into_iter().collect());
        }
        // The groups of each batch, merged with the last one of the batch
        // before if their keys are equal.
        let mut groups: Vec<Range<usize>> = vec![];
        let mut previous: Option<Vec<ArrayRef>> = None;
        let mut offset = 0;
        for batch in batches {
            let keys = partition_keys(&self.spec.group_keys, batch)?;
            working.try_grow(
                keys.iter()
                    .filter(|key| {
                        !batch
                            .columns()
                            .iter()
                            .any(|column| Arc::ptr_eq(column, key))
                    })
                    .map(|key| key.get_array_memory_size())
                    .sum(),
            )?;
            let columns = keys
                .iter()
                .map(|values| SortColumn {
                    values: Arc::clone(values),
                    options: None,
                })
                .collect::<Vec<_>>();
            let ranges = evaluate_partition_ranges(batch.num_rows(), &columns)?;
            working.try_grow(ranges.len() * size_of::<Range<usize>>())?;
            for (index, range) in ranges.into_iter().enumerate() {
                let continues = index == 0
                    && match &previous {
                        Some(previous) => same_keys(previous, &keys)?,
                        None => false,
                    };
                match groups.last_mut() {
                    Some(last) if continues => last.end = offset + range.end,
                    _ => groups.push(offset + range.start..offset + range.end),
                }
            }
            offset += batch.num_rows();
            previous = Some(keys);
        }
        Ok(groups)
    }

    /// Evaluate the stages on the group `range` of rows of `bucket`: its rows
    /// with their window values, and the memory that they hold besides the
    /// rows of the bucket (their window values, and their rows if they are
    /// copied), which is reserved in the reservation of the bucket. What
    /// re-sorting stages allocate while they sort is reserved in it while they
    /// sort.
    fn evaluate_group(
        &self,
        bucket: &SortedBucket,
        range: Range<usize>,
    ) -> Result<(RecordBatch, usize)> {
        let (rows, copied) = bucket.rows(&self.input().schema(), range)?;
        let mut batch = rows.clone();
        for stage in &self.spec.stages {
            batch = stage.evaluate(batch, &bucket.reservation)?;
        }
        // The columns of the rows of the bucket that the stages pass on are
        // slices of its rows, unless the rows are copied.
        let shared = match copied {
            true => 0,
            false => batch
                .columns()
                .iter()
                .filter(|column| {
                    rows.columns().iter().any(|row| Arc::ptr_eq(row, column))
                })
                .map(|column| array_memory(column.as_ref()))
                .sum(),
        };
        let memory = buffered_memory(&batch).saturating_sub(shared);
        bucket.reservation.try_grow(memory)?;
        Ok((batch, memory))
    }
}

/// Analyze the window node `plan` (see [`PhysicalWindow::try_lower`]): the
/// specification of the native window, and the DataFusion plan of the rows
/// that it reads, or `None` if it is left to DataFusion.
fn analyze(
    plan: &Arc<dyn ExecutionPlan>,
) -> Result<Option<(WindowSpec, Arc<dyn ExecutionPlan>)>> {
    let Some(WindowNode {
        window_exprs,
        input,
    }) = window_node(plan)
    else {
        return Ok(None);
    };
    let Some(first) = window_exprs.first() else {
        return Ok(None);
    };
    let partitions = plan.output_partitioning().partition_count();
    if partitions == 0 || partitions != input.output_partitioning().partition_count() {
        return Ok(None);
    }
    // The native window reads all rows of its input before it emits any: an
    // input that never ends is left to DataFusion, whose window streams.
    if input.boundedness().is_unbounded() {
        return Ok(None);
    }
    let partition_by = first.partition_by();
    let ordered = partitions == 1;
    // DataFusion evaluates every partition of the window separately: its
    // window partitions are those of the whole input only if the input keeps
    // rows with equal partition keys in one partition.
    if !ordered
        && (partition_by.is_empty()
            || !input.output_partitioning().keeps_keys_local(
                partition_by,
                input.equivalence_properties(),
                true,
            ))
    {
        return Ok(None);
    }
    let input_schema = input.schema();
    // A window stacked on a window below it, without a node in between that
    // sorts or merges the rows, reads them in the order in which the window
    // below evaluated them: it is evaluated as a stage of the native window of
    // the window below, on the same sorted rows, if its partitions are runs of
    // these rows (see `WindowSpec::with_stage`).
    if let Some(WindowBelow {
        projection,
        window: below,
    }) = window_below(input)
        && let Some((below_spec, rows)) = analyze(below)?
        && below_spec.ordered == ordered
        && let Some(spec) =
            below_spec.with_stage(projection, &input_schema, window_exprs.to_vec())?
    {
        return Ok(Some((spec, rows)));
    }
    // A window stacked on a window below it with other orders, where
    // DataFusion sorts the rows (and maybe repartitions them) in between, is a
    // re-sorting stage of the native window of the window below if its window
    // partitions are within the groups of that window: it sorts the rows of
    // every group again (see `WindowSpec::with_resorting_stage`).
    if !ordered
        && let Some(WindowBelow {
            projection,
            window: below,
        }) = window_below_any_sort(input)
        && let Some((below_spec, rows)) = analyze(below)?
        && let Some(spec) = below_spec.with_resorting_stage(
            projection,
            &input_schema,
            window_exprs.to_vec(),
            input.output_ordering().cloned(),
        )?
    {
        return Ok(Some((spec, rows)));
    }
    // Otherwise the native window partitions and sorts the rows of its input
    // itself, by the ordering of its input: whatever established it (a sort, a
    // merge, a window below or a scan of sorted files), the window partitions
    // are evaluated in that order (with the partition keys first), and rows
    // that it does not order are peers, whose order is the engine's own. A
    // window above another one with other partitions or orders, which
    // DataFusion sorts or merges in between, is a native window over the
    // native window of the window below.
    let rows = rows_below(input)?;
    let ordering = input.output_ordering().cloned();
    if ordering.as_ref().is_some_and(|ordering| {
        ordering.iter().any(|sort_expr| {
            sort_expr
                .expr
                .data_type(&input_schema)
                .map_or(true, |data_type| has_union(&data_type))
        })
    }) {
        return Ok(None);
    }
    let spec = match WindowSpec::try_new(
        &input_schema,
        window_exprs.to_vec(),
        ordering,
        ordered,
    ) {
        Ok(spec) => spec,
        Err(DataFusionError::NotImplemented(_)) => return Ok(None),
        Err(e) => return Err(e),
    };
    Ok(Some((spec, rows)))
}

fn partitions(plan: &Arc<dyn ExecutionPlan>) -> usize {
    plan.output_partitioning().partition_count()
}

/// A DataFusion window node: its window expressions and its input.
struct WindowNode<'a> {
    window_exprs: &'a [Arc<dyn WindowExpr>],
    input: &'a Arc<dyn ExecutionPlan>,
}

/// The window expressions and the input of `plan`, if it is a window node
/// (without a `WindowStateObserver`).
fn window_node(plan: &Arc<dyn ExecutionPlan>) -> Option<WindowNode<'_>> {
    if let Some(window) = plan.downcast_ref::<WindowAggExec>() {
        Some(WindowNode {
            window_exprs: window.window_expr(),
            input: window.input(),
        })
    } else if let Some(window) = plan.downcast_ref::<BoundedWindowAggExec>()
        && window.state_observer().is_none()
    {
        Some(WindowNode {
            window_exprs: window.window_expr(),
            input: window.input(),
        })
    } else {
        None
    }
}

/// A window node below the input of another one, see [`window_below`].
struct WindowBelow<'a> {
    /// The columns of the window node that the columns of the input are,
    /// `None` for all of them, in order.
    projection: Option<Vec<usize>>,
    window: &'a Arc<dyn ExecutionPlan>,
}

/// The window node below `input`, the input of a window node, through
/// projections of columns and redistributions that keep the order of the
/// rows (a `RepartitionExec` that preserves the order or reads a single
/// partition, and a `CoalescePartitionsExec` of a single partition).
fn window_below(input: &Arc<dyn ExecutionPlan>) -> Option<WindowBelow<'_>> {
    let mut projection: Option<Vec<usize>> = None;
    let mut node = input;
    loop {
        if window_node(node).is_some() {
            return Some(WindowBelow {
                projection,
                window: node,
            });
        }
        node = if let Some(exec) = node.downcast_ref::<ProjectionExec>() {
            let columns = exec
                .expr()
                .iter()
                .map(|expr| expr.expr.downcast_ref::<Column>().map(Column::index))
                .collect::<Option<Vec<_>>>()?;
            projection = Some(match projection {
                Some(projection) => projection
                    .iter()
                    .map(|&column| columns.get(column).copied())
                    .collect::<Option<Vec<_>>>()?,
                None => columns,
            });
            exec.input()
        } else if let Some(repartition) = node.downcast_ref::<RepartitionExec>()
            && (repartition.preserve_order() || partitions(repartition.input()) == 1)
            && distributes_rows(repartition)
        {
            repartition.input()
        } else if let Some(coalesce) = node.downcast_ref::<CoalescePartitionsExec>()
            && coalesce.fetch().is_none()
            && partitions(coalesce.input()) == 1
        {
            coalesce.input()
        } else {
            return None;
        };
    }
}

/// The window node below `input`, the input of a window node, through
/// projections of columns and the nodes that only distribute and sort rows
/// (see [`strip_distribution`]), which a re-sorting stage replaces.
fn window_below_any_sort(input: &Arc<dyn ExecutionPlan>) -> Option<WindowBelow<'_>> {
    let mut projection: Option<Vec<usize>> = None;
    let mut node = input;
    loop {
        let stripped = strip_distribution(node);
        if window_node(stripped).is_some() {
            return Some(WindowBelow {
                projection,
                window: stripped,
            });
        }
        let exec = stripped.downcast_ref::<ProjectionExec>()?;
        let columns = exec
            .expr()
            .iter()
            .map(|expr| expr.expr.downcast_ref::<Column>().map(Column::index))
            .collect::<Option<Vec<_>>>()?;
        projection = Some(match projection {
            Some(projection) => projection
                .iter()
                .map(|&column| columns.get(column).copied())
                .collect::<Option<Vec<_>>>()?,
            None => columns,
        });
        node = exec.input();
    }
}

/// Whether the values `previous` of the group keys of the last row of a
/// batch are equal to the values `keys` of the first row of the next batch,
/// as `evaluate_partition_ranges` compares them (floats by their bits, NULLs
/// equal).
fn same_keys(previous: &[ArrayRef], keys: &[ArrayRef]) -> Result<bool> {
    for (previous, keys) in previous.iter().zip(keys) {
        let last = previous.len() - 1;
        let compare =
            make_comparator(previous.as_ref(), keys.as_ref(), SortOptions::default())?;
        if compare(last, 0).is_ne() {
            return Ok(false);
        }
    }
    Ok(true)
}

/// The rows that a window reads, of `input`, the input of the window node:
/// `input` without the nodes that only distribute and sort its rows (see
/// [`strip_distribution`]), which the native window replaces, also below the
/// projections above them, which are moved below these nodes. A projection
/// computes every row on its own (columns, or any expression), so it computes
/// the same rows below the sorts, merges and repartitionings as above them,
/// and the native window partitions and sorts the projected rows itself.
fn rows_below(input: &Arc<dyn ExecutionPlan>) -> Result<Arc<dyn ExecutionPlan>> {
    let stripped = strip_distribution(input);
    let Some(projection) = stripped.downcast_ref::<ProjectionExec>() else {
        return Ok(Arc::clone(stripped));
    };
    let rows = rows_below(projection.input())?;
    if Arc::ptr_eq(&rows, projection.input()) {
        return Ok(Arc::clone(stripped));
    }
    Arc::clone(stripped).replace_children(
        vec![rows],
        ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
    )
}

/// `plan` without the nodes that only distribute and sort its rows (see
/// [`PhysicalWindow::try_lower`]): sorts, merges of sorted partitions, merges
/// of partitions and repartitionings, without a fetch, that read all
/// partitions of their input.
fn strip_distribution(plan: &Arc<dyn ExecutionPlan>) -> &Arc<dyn ExecutionPlan> {
    let mut plan = plan;
    loop {
        let next = if let Some(sort) = plan.downcast_ref::<SortExec>() {
            // A sort of all its input reads only the first partition of it.
            let reads_all = if sort.preserve_partitioning() {
                partitions(plan) == partitions(sort.input())
            } else {
                partitions(sort.input()) == 1
            };
            (sort.fetch().is_none() && reads_all).then(|| sort.input())
        } else if let Some(merge) = plan.downcast_ref::<SortPreservingMergeExec>() {
            (merge.fetch().is_none() && partitions(merge.input()) > 0)
                .then(|| merge.input())
        } else if let Some(coalesce) = plan.downcast_ref::<CoalescePartitionsExec>() {
            (coalesce.fetch().is_none() && partitions(coalesce.input()) > 0)
                .then(|| coalesce.input())
        } else if let Some(repartition) = plan.downcast_ref::<RepartitionExec>() {
            (distributes_rows(repartition) && partitions(repartition.input()) > 0)
                .then(|| repartition.input())
        } else {
            None
        };
        match next {
            Some(next) => plan = next,
            None => return plan,
        }
    }
}

/// Whether `repartition` distributes the rows of its input over its
/// partitions (round robin, or by the hash of expressions), which the native
/// window does itself, by its partition keys.
fn distributes_rows(repartition: &RepartitionExec) -> bool {
    matches!(
        repartition.partitioning(),
        Partitioning::RoundRobinBatch(_) | Partitioning::Hash(_, _)
    )
}

/// Whether values of `data_type` are unions, or have unions inside them.
fn has_union(data_type: &DataType) -> bool {
    match data_type {
        DataType::Union(_, _) => true,
        DataType::Dictionary(_, values) => has_union(values),
        DataType::RunEndEncoded(_, values) => has_union(values.data_type()),
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::ListView(field)
        | DataType::LargeListView(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => has_union(field.data_type()),
        DataType::Struct(fields) => {
            fields.iter().any(|field| has_union(field.data_type()))
        }
        _ => false,
    }
}

/// The rows of a bucket collected by the threads that completed: sorted runs,
/// or the batches as they came if the rows have no order.
#[derive(Debug, Default)]
struct BucketInput {
    runs: Vec<SortedRun>,
    batches: Vec<RecordBatch>,
    /// The memory of the runs and the batches, which the reservation of the
    /// window holds.
    memory: usize,
}

impl BucketInput {
    fn is_empty(&self) -> bool {
        self.runs.iter().all(|run| run.num_rows() == 0)
            && self.batches.iter().all(|batch| batch.num_rows() == 0)
    }
}

/// The rows of a bucket in their order, and its groups.
#[derive(Debug)]
struct SortedBucket {
    /// The sorted rows: all of them in one batch, unless they do not fit.
    batches: Vec<RecordBatch>,
    /// `offsets[i]` is the index of the first row of `batches[i]`.
    offsets: Vec<usize>,
    /// The groups of rows with equal group keys, in order: ranges of rows.
    groups: Vec<Range<usize>>,
    /// The memory of the rows and the groups, and of the rows of the bucket
    /// that were evaluated and not emitted yet (see [`BucketEmitter`]),
    /// freed with the bucket.
    reservation: MemoryReservation,
}

impl SortedBucket {
    fn new(
        batches: Vec<RecordBatch>,
        groups: Vec<Range<usize>>,
        reservation: MemoryReservation,
    ) -> Self {
        let mut offsets = Vec::with_capacity(batches.len() + 1);
        let mut num_rows = 0;
        offsets.push(0);
        for batch in &batches {
            num_rows += batch.num_rows();
            offsets.push(num_rows);
        }
        Self {
            batches,
            offsets,
            groups,
            reservation,
        }
    }

    /// The rows `range` as one batch with schema `schema`: a slice of one
    /// batch, or the concatenation of slices of several, and whether they are
    /// a concatenation (a copy of the rows).
    fn rows(
        &self,
        schema: &SchemaRef,
        range: Range<usize>,
    ) -> Result<(RecordBatch, bool)> {
        let first = self
            .offsets
            .partition_point(|&offset| offset <= range.start)
            - 1;
        let mut slices = vec![];
        let mut batch = first;
        while batch < self.batches.len() && self.offsets[batch] < range.end {
            let start = range.start.max(self.offsets[batch]) - self.offsets[batch];
            let end = range.end.min(self.offsets[batch + 1]) - self.offsets[batch];
            slices.push(self.batches[batch].slice(start, end - start));
            batch += 1;
        }
        match slices.as_slice() {
            [slice] => Ok((slice.clone(), false)),
            _ => Ok((concat_batches(schema, &slices)?, true)),
        }
    }
}

/// The global sink state of a [`PhysicalWindow`], also read by its source.
#[derive(Debug)]
struct WindowState {
    /// The number of radix bits of the buckets (`2^bits` buckets).
    bits: u32,
    /// The rows of every bucket.
    buckets: Vec<Mutex<BucketInput>>,
    /// The next bucket to sort.
    next_sort: AtomicUsize,
    /// The sorted buckets, until the source takes them.
    sorted: Vec<Mutex<Option<SortedBucket>>>,
    /// The memory of the rows that all threads buffered and of their runs,
    /// until the buckets are sorted (which take the memory of their rows).
    reservation: MemoryReservation,
}

impl GlobalSinkState for WindowState {}

impl WindowState {
    /// Sort the next bucket that is not sorted yet. Returns `false` once all
    /// buckets were claimed.
    fn sort_next_bucket(&self, window: &PhysicalWindow) -> Result<bool> {
        loop {
            let bucket = self.next_sort.fetch_add(1, Ordering::Relaxed);
            let Some(input) = self.buckets.get(bucket) else {
                return Ok(false);
            };
            let input = std::mem::take(&mut *input.lock());
            if input.is_empty() {
                self.reservation.shrink(input.memory);
                continue;
            }
            let sorted = window.sort_bucket(input, &self.reservation)?;
            *self.sorted[bucket].lock() = Some(sorted);
            return Ok(true);
        }
    }
}

/// The local sink state of a [`PhysicalWindow`]: the rows of the thread, by
/// bucket.
#[derive(Debug, Default)]
struct WindowLocalState {
    buckets: Vec<LocalBucket>,
    hashes: Vec<u64>,
}

impl LocalSinkState for WindowLocalState {}

/// The rows of a bucket collected by a thread.
#[derive(Debug, Default)]
struct LocalBucket {
    /// Compacted chunks that are not sorted yet (or the rows without order).
    batches: Vec<RecordBatch>,
    /// The number of rows of `batches`.
    rows: usize,
    /// The memory of `batches` (see [`buffered_memory`]).
    bytes: usize,
    /// The memory of the batches of `batches` that the bucket holds alone
    /// (the chunks of a window of a single bucket, or of a single bucket of
    /// a chunk), which the reservation of the window holds.
    reserved: usize,
    /// The reservations of the chunks that the other batches of `batches` are
    /// slices of (the slices of the rows of a chunk of every bucket): every
    /// chunk is reserved once, for its slices in all buckets of the thread,
    /// and freed once all of them are sorted.
    chunks: Vec<Arc<MemoryReservation>>,
    /// The sorted runs of the thread, whose memory the reservation of the
    /// window holds.
    runs: Vec<SortedRun>,
}

/// The global source state of a [`PhysicalWindow`]: the sorted buckets,
/// which the threads claim.
#[derive(Debug)]
struct WindowSourceState {
    buckets: Vec<Mutex<Option<SortedBucket>>>,
    next: AtomicUsize,
    non_empty: usize,
}

impl GlobalSourceState for WindowSourceState {
    fn max_threads(&self) -> usize {
        self.non_empty.max(1)
    }
}

/// The local source state of a [`PhysicalWindow`]: the bucket the thread
/// emits.
#[derive(Debug, Default)]
struct WindowLocalSourceState {
    current: Option<BucketEmitter>,
}

impl LocalSourceState for WindowLocalSourceState {}

/// Evaluates the groups of a bucket, in order, and emits their rows with
/// their window values in chunks.
///
/// The memory that the evaluated rows hold besides the rows of the bucket is
/// reserved in the reservation of the bucket until they are emitted.
#[derive(Debug)]
struct BucketEmitter {
    bucket: SortedBucket,
    /// The next group to evaluate.
    next_group: usize,
    /// Evaluated rows that were not emitted yet, in order, with the memory
    /// that they hold (see [`PhysicalWindow::evaluate_group`]), freed once
    /// all of their rows were emitted.
    pending: VecDeque<(RecordBatch, usize)>,
    pending_rows: usize,
    /// For a window with an emit order (see [`WindowSpec::emit_order`]): the
    /// evaluated rows of all groups, sorted by that order, once all groups
    /// were evaluated.
    reordered: Option<RunMerger>,
}

impl BucketEmitter {
    fn new(bucket: SortedBucket) -> Self {
        Self {
            bucket,
            next_group: 0,
            pending: VecDeque::new(),
            pending_rows: 0,
            reordered: None,
        }
    }

    /// The next chunk of at most [`DEFAULT_VECTOR_SIZE`] rows, `None` once all
    /// rows of the bucket were emitted.
    fn next_chunk(&mut self, window: &PhysicalWindow) -> Result<Option<DataChunk>> {
        if let Some(emit_order) = &window.spec.emit_order {
            return self.next_reordered_chunk(window, emit_order);
        }
        while self.pending_rows < DEFAULT_VECTOR_SIZE
            && self.next_group < self.bucket.groups.len()
        {
            let range = self.bucket.groups[self.next_group].clone();
            self.next_group += 1;
            let (rows, memory) = window.evaluate_group(&self.bucket, range)?;
            self.pending_rows += rows.num_rows();
            self.pending.push_back((rows, memory));
        }
        if self.pending_rows == 0 {
            return Ok(None);
        }
        // Large groups are emitted as they are (in slices of at most
        // `DEFAULT_VECTOR_SIZE` rows), small ones are concatenated. Every part
        // comes with the memory that is freed once it is emitted: of the
        // evaluated rows whose last rows it holds.
        let mut parts = vec![];
        let mut len = 0;
        while let Some((front, memory)) = self.pending.pop_front() {
            let rows = front.num_rows();
            let large = rows >= DEFAULT_VECTOR_SIZE / 2;
            if large && !parts.is_empty() {
                self.pending.push_front((front, memory));
                break;
            }
            let take = rows.min(DEFAULT_VECTOR_SIZE - len);
            if take < rows {
                self.pending
                    .push_front((front.slice(take, rows - take), memory));
                parts.push((front.slice(0, take), 0));
            } else {
                parts.push((front, memory));
            }
            len += take;
            if large || len == DEFAULT_VECTOR_SIZE {
                break;
            }
        }
        let (batch, emitted) = match parts.as_slice() {
            [(part, memory)] => (part.clone(), *memory),
            _ => {
                let batches = parts.iter().map(|(part, _)| part).collect::<Vec<_>>();
                match concat_batches(&window.spec.schema, batches) {
                    Ok(batch) => (batch, parts.iter().map(|(_, memory)| memory).sum()),
                    // The parts do not fit into one batch (e.g. their
                    // dictionaries need more values than their keys address):
                    // emit them one by one.
                    Err(error) if is_capacity_error(&error) => {
                        let mut parts = parts.into_iter();
                        let first = parts.next().ok_or_else(|| {
                            internal_datafusion_err!("A chunk of no parts")
                        })?;
                        for part in parts.rev() {
                            self.pending.push_front(part);
                        }
                        first
                    }
                    Err(error) => return Err(error.into()),
                }
            }
        };
        self.pending_rows -= batch.num_rows();
        self.bucket.reservation.shrink(emitted);
        Ok(Some(DataChunk::from_record_batch(batch)))
    }

    /// The next chunk of the rows of a window with the emit order
    /// `emit_order`: the rows of all groups, once evaluated, sorted by it.
    fn next_reordered_chunk(
        &mut self,
        window: &PhysicalWindow,
        emit_order: &SortKeys,
    ) -> Result<Option<DataChunk>> {
        if self.reordered.is_none() {
            let mut evaluated = Vec::with_capacity(self.bucket.groups.len());
            while self.next_group < self.bucket.groups.len() {
                let range = self.bucket.groups[self.next_group].clone();
                self.next_group += 1;
                evaluated.push(window.evaluate_group(&self.bucket, range)?.0);
            }
            // The memory of the rows of the bucket and of the evaluated rows,
            // which the runs of the sorted evaluated rows replace.
            let reservation = &self.bucket.reservation;
            let held = reservation.size();
            let schema = &window.spec.schema;
            let runs = emit_order.sort(schema, &evaluated, None, reservation)?;
            // The rows of the bucket are evaluated and sorted: release them.
            drop(evaluated);
            self.bucket.batches = vec![];
            self.bucket.offsets = vec![0];
            self.bucket.groups = vec![];
            reservation.shrink(held);
            self.reordered = Some(RunMerger::new(Arc::clone(schema), runs, None));
        }
        let Some(merger) = &mut self.reordered else {
            return internal_err!("The rows of the window were not sorted");
        };
        let batch = merger.next_batch(DEFAULT_VECTOR_SIZE)?;
        // Free the memory of the runs that the merger dropped.
        let released = self
            .bucket
            .reservation
            .size()
            .saturating_sub(merger.memory_size());
        self.bucket.reservation.shrink(released);
        Ok(batch.map(DataChunk::from_record_batch))
    }
}

impl PhysicalOperator for PhysicalWindow {
    fn name(&self) -> &str {
        "PhysicalWindow"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.spec.schema)
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.children
    }

    fn description(&self) -> String {
        let spec = &self.spec;
        let stages = spec
            .stages
            .iter()
            .map(|stage| {
                let exprs = stage
                    .window_exprs
                    .iter()
                    .map(|expr| expr.name().to_string())
                    .collect::<Vec<_>>()
                    .join(", ");
                match &stage.sort {
                    Some(keys) => format!("[{exprs}] (sort=[{}])", keys.ordering()),
                    None => format!("[{exprs}]"),
                }
            })
            .collect::<Vec<_>>()
            .join(" -> ");
        let mut description = format!("PhysicalWindow: wdw={stages}");
        if let Some(keys) = &spec.keys {
            let _ = write!(description, ", sort=[{}]", keys.ordering());
        }
        if spec.ordered {
            description.push_str(", ordered");
        }
        description
    }

    // ---------------------------------------------------------------------
    // Sink
    // ---------------------------------------------------------------------

    fn is_sink(&self) -> bool {
        true
    }

    fn parallel_sink(&self) -> bool {
        true
    }

    fn get_global_sink_state(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalSinkState>> {
        let num_buckets = self.spec.num_buckets(ctx.num_threads);
        let reservation =
            MemoryConsumer::new(self.name()).register(ctx.task_ctx.memory_pool());
        Ok(Arc::new(WindowState {
            bits: num_buckets.trailing_zeros(),
            buckets: (0..num_buckets).map(|_| Mutex::default()).collect(),
            next_sort: AtomicUsize::new(0),
            sorted: (0..num_buckets).map(|_| Mutex::default()).collect(),
            reservation,
        }))
    }

    fn get_local_sink_state(
        &self,
        _ctx: &ExecutionContext,
        gstate: &dyn GlobalSinkState,
    ) -> Result<Box<dyn LocalSinkState>> {
        let state = downcast_state::<WindowState>(gstate.as_any(), self.name())?;
        Ok(Box::new(WindowLocalState {
            buckets: (0..state.buckets.len())
                .map(|_| LocalBucket::default())
                .collect(),
            hashes: vec![],
        }))
    }

    fn sink(
        &self,
        _ctx: &ExecutionContext,
        input: &DataChunk,
        gstate: &dyn GlobalSinkState,
        lstate: &mut dyn LocalSinkState,
    ) -> Result<SinkResult> {
        let state = downcast_state::<WindowState>(gstate.as_any(), self.name())?;
        let local =
            downcast_state_mut::<WindowLocalState>(lstate.as_any_mut(), self.name())?;
        let batch = input.compact()?;
        let reservation = &state.reservation;
        if local.buckets.len() == 1 {
            // The rows are kept until they are sorted (or evaluated): not the
            // arrays that they may be slices of.
            let batch = compact_to_keep(batch)?;
            self.buffer(&mut local.buckets[0], batch, None, reservation)?;
            return Ok(SinkResult::NeedMoreInput);
        }
        if self.spec.keys.is_none() {
            return internal_err!(
                "{} distributes rows without order over several buckets",
                self.name()
            );
        }
        // Distribute the rows over the buckets by the hash of their keys (the
        // upper bits of the hash pick the bucket): gather them in the order of
        // their buckets, and hand every bucket its slice of the rows.
        let num_rows = batch.num_rows();
        let keys = partition_keys(&self.spec.group_keys, &batch)?;
        hash_join_keys(&keys, num_rows, &mut local.hashes)?;
        let shift = u64::BITS - state.bits;
        let num_buckets = local.buckets.len();
        let mut starts = vec![0; num_buckets + 1];
        for &hash in &local.hashes {
            starts[(hash >> shift) as usize + 1] += 1;
        }
        for bucket in 0..num_buckets {
            starts[bucket + 1] += starts[bucket];
        }
        let mut next = starts.clone();
        let mut order = vec![0; num_rows];
        for (row, &hash) in local.hashes.iter().enumerate() {
            let bucket = (hash >> shift) as usize;
            order[next[bucket]] = row as u32;
            next[bucket] += 1;
        }
        if let Some(bucket) =
            (0..num_buckets).find(|&b| starts[b + 1] - starts[b] == num_rows)
        {
            // All rows are in one bucket, which keeps them until it sorts
            // them: not the arrays that they may be slices of.
            let batch = compact_to_keep(batch)?;
            self.buffer(&mut local.buckets[bucket], batch, None, reservation)?;
            return Ok(SinkResult::NeedMoreInput);
        }
        // (`take` copies the rows of the chunk but the values of dictionaries
        // and list views, which the reordered chunk then keeps.)
        let sorted =
            compact_to_keep(take_record_batch(&batch, &UInt32Array::from(order))?)?;
        drop(batch);
        // Every bucket keeps a slice of the rows of the chunk, which keeps all
        // of the chunk: it is reserved once (its rows, and the metadata of all
        // slices), until the slices of all buckets are sorted.
        let slices = (0..num_buckets)
            .filter(|&bucket| starts[bucket + 1] > starts[bucket])
            .map(|bucket| {
                let (start, end) = (starts[bucket], starts[bucket + 1]);
                (bucket, sorted.slice(start, end - start))
            })
            .collect::<Vec<_>>();
        let chunk = Arc::new(reservation.new_empty());
        chunk.try_grow(
            memory_size(&sorted)
                + slices
                    .iter()
                    .map(|(_, slice)| metadata_memory(slice))
                    .sum::<usize>(),
        )?;
        drop(sorted);
        for (bucket, slice) in slices {
            self.buffer(&mut local.buckets[bucket], slice, Some(&chunk), reservation)?;
        }
        Ok(SinkResult::NeedMoreInput)
    }

    fn combine(
        &self,
        _ctx: &ExecutionContext,
        gstate: &dyn GlobalSinkState,
        lstate: &mut dyn LocalSinkState,
    ) -> Result<()> {
        let state = downcast_state::<WindowState>(gstate.as_any(), self.name())?;
        let local =
            downcast_state_mut::<WindowLocalState>(lstate.as_any_mut(), self.name())?;
        for (bucket, global) in local.buckets.iter_mut().zip(&state.buckets) {
            self.sort_buffered_rows(bucket, &state.reservation)?;
            // (The rows without order are those of a window of a single
            // bucket: no slices of chunks of several buckets.)
            if !bucket.chunks.is_empty() {
                return internal_err!(
                    "{} keeps the slices of chunks of several buckets",
                    self.name()
                );
            }
            let mut global = global.lock();
            global.memory += bucket
                .runs
                .iter()
                .map(SortedRun::memory_size)
                .sum::<usize>()
                + std::mem::take(&mut bucket.reserved);
            global.runs.append(&mut bucket.runs);
            global.batches.append(&mut bucket.batches);
        }
        Ok(())
    }

    fn finalize(
        &self,
        ctx: &ExecutionContext,
        gstate: &dyn GlobalSinkState,
    ) -> Result<SinkFinalize> {
        let state = downcast_state::<WindowState>(gstate.as_any(), self.name())?;
        // The output of a canceled window is not needed any more.
        if ctx.registry().is_sink_finished(self) {
            return Ok(SinkFinalize::NoOutputPossible);
        }
        // Sort the buckets that the sort pipeline did not.
        while state.sort_next_bucket(self)? {}
        let empty = state.sorted.iter().all(|bucket| bucket.lock().is_none());
        Ok(if empty {
            SinkFinalize::NoOutputPossible
        } else {
            SinkFinalize::Ready
        })
    }

    // ---------------------------------------------------------------------
    // Source
    // ---------------------------------------------------------------------

    fn is_source(&self) -> bool {
        true
    }

    fn parallel_source(&self) -> bool {
        true
    }

    fn get_global_source_state(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalSourceState>> {
        let state = window_state(ctx, self)?;
        let buckets = state
            .sorted
            .iter()
            .map(|bucket| Mutex::new(bucket.lock().take()))
            .collect::<Vec<_>>();
        let non_empty = buckets
            .iter()
            .filter(|bucket| bucket.lock().is_some())
            .count();
        Ok(Arc::new(WindowSourceState {
            buckets,
            next: AtomicUsize::new(0),
            non_empty,
        }))
    }

    fn get_local_source_state(
        &self,
        _ctx: &ExecutionContext,
        _gstate: &dyn GlobalSourceState,
    ) -> Result<Box<dyn LocalSourceState>> {
        Ok(Box::new(WindowLocalSourceState::default()))
    }

    fn get_data(
        &self,
        _ctx: &ExecutionContext,
        gstate: &dyn GlobalSourceState,
        lstate: &mut dyn LocalSourceState,
    ) -> Result<SourceResult> {
        let global = downcast_state::<WindowSourceState>(gstate.as_any(), self.name())?;
        let local = downcast_state_mut::<WindowLocalSourceState>(
            lstate.as_any_mut(),
            self.name(),
        )?;
        loop {
            if let Some(emitter) = &mut local.current {
                if let Some(chunk) = emitter.next_chunk(self)? {
                    return Ok(SourceResult::Chunk(chunk));
                }
                local.current = None;
            }
            // Claim the next bucket.
            let next = global.next.fetch_add(1, Ordering::Relaxed);
            let Some(bucket) = global.buckets.get(next) else {
                return Ok(SourceResult::Finished);
            };
            if let Some(bucket) = bucket.lock().take() {
                local.current = Some(BucketEmitter::new(bucket));
            }
        }
    }

    // ---------------------------------------------------------------------
    // Pipelines
    // ---------------------------------------------------------------------

    fn build_pipelines(
        &self,
        current: &mut PipelineBuilder,
        meta: &mut MetaPipelineBuilder<'_>,
    ) -> Result<()> {
        let this = meta.current_operator()?;
        // The window is the source of the pipeline that reads its output.
        meta.set_source(current, Arc::clone(&this))?;
        // The input sinks into the window, in a child meta pipeline, whose
        // last pipeline sorts the buckets in parallel.
        let (mut child, mut input_pipeline) =
            meta.create_child_meta_pipeline(current, Arc::clone(&this))?;
        let sort_pipeline = child.create_union_pipeline(&input_pipeline, false)?;
        child.set_source(
            &sort_pipeline,
            Arc::new(WindowPartitionSort::new(Arc::clone(&this))),
        )?;
        child.build(&mut input_pipeline, self.input())?;
        for pipeline in child.pipelines() {
            if pipeline != sort_pipeline {
                child.add_dependency(&sort_pipeline, &pipeline)?;
            }
        }
        Ok(())
    }
}

/// The global state of the window in `ctx`'s execution: the global state of
/// `window` as a sink.
fn window_state(
    ctx: &ExecutionContext,
    window: &dyn PhysicalOperator,
) -> Result<Arc<WindowState>> {
    ctx.global_sink_state(window)
        .and_then(|state| state.downcast_arc::<WindowState>())
        .ok_or_else(|| {
            internal_datafusion_err!("The input of {} was not executed", window.name())
        })
}

/// The source of the pipeline that sorts the buckets of a [`PhysicalWindow`]
/// in parallel, once all its rows are combined: every call of `get_data`
/// sorts one bucket (and yields, as sorting a large bucket takes a while),
/// until all buckets are claimed. It produces no rows.
#[derive(Debug)]
struct WindowPartitionSort {
    /// The window (a [`PhysicalWindow`]).
    window: Arc<dyn PhysicalOperator>,
    schema: SchemaRef,
}

impl WindowPartitionSort {
    fn new(window: Arc<dyn PhysicalOperator>) -> Self {
        Self {
            window,
            schema: Arc::new(Schema::new(Vec::<Field>::new())),
        }
    }
}

/// The global state of a [`WindowPartitionSort`].
#[derive(Debug)]
struct WindowPartitionSortState {
    window: Arc<WindowState>,
    non_empty: usize,
}

impl GlobalSourceState for WindowPartitionSortState {
    fn max_threads(&self) -> usize {
        self.non_empty.max(1)
    }
}

impl PhysicalOperator for WindowPartitionSort {
    fn name(&self) -> &str {
        "WindowPartitionSort"
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
        ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalSourceState>> {
        let window = window_state(ctx, self.window.as_ref())?;
        let non_empty = window
            .buckets
            .iter()
            .filter(|bucket| !bucket.lock().is_empty())
            .count();
        Ok(Arc::new(WindowPartitionSortState { window, non_empty }))
    }

    fn get_data(
        &self,
        _ctx: &ExecutionContext,
        gstate: &dyn GlobalSourceState,
        _lstate: &mut dyn LocalSourceState,
    ) -> Result<SourceResult> {
        let state =
            downcast_state::<WindowPartitionSortState>(gstate.as_any(), self.name())?;
        let Some(window) = self.window.downcast_ref::<PhysicalWindow>() else {
            return internal_err!(
                "WindowPartitionSort sorts the rows of a PhysicalWindow"
            );
        };
        Ok(if state.window.sort_next_bucket(window)? {
            // Let other tasks run between the buckets.
            SourceResult::Blocked(TaskInterrupt::yield_now())
        } else {
            SourceResult::Finished
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::array::{AsArray, Int32Array, Int64Array, StringArray};
    use arrow::datatypes::{Int64Type, UInt64Type};
    use datafusion::datasource::MemTable;
    use datafusion::prelude::{SessionConfig, SessionContext};
    use datafusion_execution::TaskContext;
    use datafusion_physical_expr::expressions::col;
    use datafusion_physical_plan::displayable;

    use crate::executor::Executor;
    use crate::explain::{format_operator_tree, format_pipelines};
    use crate::operators::union::test_support::{
        TestSource, assert_lowered_plan_byte_eq, count, run, values_chunk, values_schema,
    };
    use crate::test_utils::{
        datafusion_context, register_test_tables, test_session_config, test_table_t1,
    };
    use crate::window::test_support::{
        assert_native_parity_with_config, assert_native_parity_with_setup, lower,
    };
    use datafusion_datasource::memory::MemorySourceConfig;
    use datafusion_physical_plan::filter::FilterExec;
    use datafusion_physical_plan::union::UnionExec;

    /// The number of stages of the native windows of the tree below (and
    /// including) `operator`.
    fn stages(operator: &Arc<dyn PhysicalOperator>) -> usize {
        operator
            .downcast_ref::<PhysicalWindow>()
            .map_or(0, |window| window.window_exprs().len())
            + operator.children().iter().map(stages).sum::<usize>()
    }

    /// The number of window nodes of `plan`.
    fn window_nodes(plan: &Arc<dyn ExecutionPlan>) -> usize {
        usize::from(plan.is::<WindowAggExec>() || plan.is::<BoundedWindowAggExec>())
            + plan.children().into_iter().map(window_nodes).sum::<usize>()
    }

    /// The physical plan of `sql` on the test tables, with `config`.
    async fn physical_plan(sql: &str, config: SessionConfig) -> Arc<dyn ExecutionPlan> {
        let ctx = datafusion_context(config);
        register_test_tables(&ctx).unwrap();
        ctx.sql(sql)
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap()
    }

    /// Assert that `sql` matches DataFusion with `config`, with every window
    /// node lowered to a native window if `native`, with none otherwise.
    async fn assert_windows_with_config(sql: &str, config: SessionConfig, native: bool) {
        let plan = physical_plan(sql, config.clone()).await;
        let windows = window_nodes(&plan);
        assert!(
            windows > 0,
            "query: {sql}\n{}",
            displayable(plan.as_ref()).indent(true)
        );
        let root = assert_native_parity_with_config(sql, config).await;
        assert_eq!(
            stages(&root),
            if native { windows } else { 0 },
            "query: {sql}\nplan:\n{}\nlowered: {}",
            displayable(plan.as_ref()).indent(true),
            format_operator_tree(&root)
        );
    }

    async fn assert_windows(sql: &str) {
        assert_windows_with_config(sql, test_session_config(), true).await;
    }

    // ---------------------------------------------------------------------
    // Window functions
    // ---------------------------------------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn ranking_functions() {
        for sql in [
            "SELECT id, ROW_NUMBER() OVER (PARTITION BY d ORDER BY id) AS rn FROM t1",
            "SELECT id, ROW_NUMBER() OVER (ORDER BY id DESC) AS rn FROM t1",
            "SELECT COUNT(*) OVER () AS n, ROW_NUMBER() OVER (ORDER BY id) AS rn FROM t1",
            "SELECT id, RANK() OVER (PARTITION BY e ORDER BY a) AS r, DENSE_RANK() OVER (PARTITION BY e ORDER BY a) AS dr FROM t1",
            "SELECT id, RANK() OVER (ORDER BY a DESC NULLS FIRST) AS r FROM t1",
            "SELECT id, PERCENT_RANK() OVER (PARTITION BY d ORDER BY b) AS pr, CUME_DIST() OVER (PARTITION BY d ORDER BY b) AS cd FROM t1",
            "SELECT id, NTILE(7) OVER (PARTITION BY d ORDER BY id) AS n FROM t1",
            "SELECT id, NTILE(3) OVER (ORDER BY id) AS n FROM t1",
        ] {
            assert_windows(sql).await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn value_functions() {
        for sql in [
            "SELECT id, LAG(a) OVER (ORDER BY id) AS prev, LEAD(a, 2, -1) OVER (ORDER BY id) AS next FROM t1",
            "SELECT id, LAG(b, 3) OVER (PARTITION BY d ORDER BY id) AS l, LEAD(c, 1, 0.5) OVER (PARTITION BY d ORDER BY id) AS n FROM t1",
            "SELECT id, LAG(h, 1, 'none') OVER (PARTITION BY e ORDER BY id DESC) AS l FROM t1",
            "SELECT id, FIRST_VALUE(h) OVER (PARTITION BY d ORDER BY id) AS fv, LAST_VALUE(g) OVER (PARTITION BY d ORDER BY id) AS lv FROM t1",
            "SELECT id, LAST_VALUE(ts) OVER (PARTITION BY e ORDER BY id ROWS BETWEEN UNBOUNDED PRECEDING AND UNBOUNDED FOLLOWING) AS lv FROM t1",
            "SELECT id, NTH_VALUE(f, 3) OVER (PARTITION BY d ORDER BY id) AS nv FROM t1",
            "SELECT id, FIRST_VALUE(c) OVER (PARTITION BY d ORDER BY id ROWS BETWEEN 2 FOLLOWING AND 5 FOLLOWING) AS fv FROM t1",
        ] {
            assert_windows(sql).await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn aggregate_functions_over_frames() {
        for sql in [
            // Sliding and cumulative `ROWS` frames.
            "SELECT id, SUM(b) OVER (PARTITION BY a ORDER BY id ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS s FROM t1",
            // (`b` has ties within the partitions: `id` orders them.)
            "SELECT id, SUM(c) OVER (PARTITION BY a ORDER BY b, id ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS s FROM t1",
            "SELECT a, b, SUM(b) OVER (PARTITION BY a ORDER BY b ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS s FROM t1",
            "SELECT id, SUM(b) OVER (ORDER BY id ROWS BETWEEN 2 PRECEDING AND 1 FOLLOWING) AS s FROM t1",
            "SELECT id, SUM(c) OVER (PARTITION BY d ORDER BY id ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS s FROM t1",
            // (Averages of floating-point values; integers are cast to
            // `Float64`.)
            "SELECT id, AVG(c) OVER (PARTITION BY e ORDER BY id ROWS UNBOUNDED PRECEDING) AS a FROM t1",
            "SELECT id, AVG(a) OVER (PARTITION BY d ORDER BY id ROWS BETWEEN 2 PRECEDING AND 2 FOLLOWING) AS a, AVG(DISTINCT a) OVER (PARTITION BY e) AS d FROM t1",
            "SELECT id, MIN(h) OVER (PARTITION BY d ORDER BY id ROWS BETWEEN 5 PRECEDING AND 5 FOLLOWING) AS mn, MAX(ts) OVER (PARTITION BY d ORDER BY id ROWS BETWEEN 5 PRECEDING AND 5 FOLLOWING) AS mx FROM t1",
            "SELECT id, COUNT(c) OVER (ORDER BY id ROWS BETWEEN CURRENT ROW AND 10 FOLLOWING) AS n FROM t1",
            // `RANGE` frames, with peers and with offsets.
            "SELECT id, SUM(b) OVER (PARTITION BY d ORDER BY a) AS s FROM t1",
            "SELECT id, SUM(a) OVER (ORDER BY a RANGE BETWEEN 5 PRECEDING AND 5 FOLLOWING) AS s FROM t1",
            "SELECT id, COUNT(*) OVER (PARTITION BY e ORDER BY f RANGE BETWEEN INTERVAL '3' DAY PRECEDING AND CURRENT ROW) AS n FROM t1",
            "SELECT id, MAX(b) OVER (ORDER BY g DESC RANGE BETWEEN 100 PRECEDING AND 100 FOLLOWING) AS m FROM t1",
            // `GROUPS` frames.
            "SELECT id, SUM(b) OVER (PARTITION BY d ORDER BY a GROUPS BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS s FROM t1",
            "SELECT id, COUNT(*) OVER (ORDER BY f GROUPS BETWEEN 2 PRECEDING AND CURRENT ROW) AS n FROM t1",
            // Whole partitions.
            "SELECT id, COUNT(*) OVER (PARTITION BY d) AS n, SUM(a) OVER (PARTITION BY d) AS s FROM t1",
            "SELECT id, MIN(c) OVER (PARTITION BY e) AS mn, MAX(h) OVER () AS mx FROM t1",
            "SELECT id, COUNT(DISTINCT a) OVER (PARTITION BY d) AS n FROM t1",
        ] {
            assert_windows(sql).await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn partition_keys_of_many_types() {
        for keys in [
            "d", "h", "e", "f", "g", "c", "ts", "a, e", "d, e, f", "a % 3", "id % 5",
            "d, d", "e, d, e",
        ] {
            let sql = format!(
                "SELECT id, ROW_NUMBER() OVER (PARTITION BY {keys} ORDER BY id) AS rn, \
                 SUM(b) OVER (PARTITION BY {keys} ORDER BY id ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS s \
                 FROM t1"
            );
            assert_windows(&sql).await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sort_orders() {
        for order in [
            "id",
            "id DESC",
            "c NULLS FIRST, id",
            "c DESC NULLS LAST, id",
            "d DESC, h, id",
            "ts DESC NULLS FIRST, id DESC",
            "e, g DESC, id",
        ] {
            let sql = format!(
                "SELECT id, ROW_NUMBER() OVER (PARTITION BY a ORDER BY {order}) AS rn, \
                 LAG(id) OVER (ORDER BY {order}) AS prev FROM t1"
            );
            assert_windows(&sql).await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn several_windows() {
        for sql in [
            "SELECT id, LAG(a) OVER (ORDER BY id) AS prev, RANK() OVER (PARTITION BY e ORDER BY a, id) AS r FROM t1",
            "SELECT id, ROW_NUMBER() OVER (PARTITION BY d ORDER BY id) AS rn, SUM(b) OVER (PARTITION BY d) AS s FROM t1",
            "SELECT id, PERCENT_RANK() OVER (PARTITION BY d ORDER BY id) AS pr, NTILE(3) OVER (ORDER BY id) AS n FROM t1",
            "SELECT id, ROW_NUMBER() OVER (PARTITION BY d ORDER BY id) AS r1, ROW_NUMBER() OVER (PARTITION BY e ORDER BY id) AS r2, ROW_NUMBER() OVER (ORDER BY b, id) AS r3 FROM t1",
            "SELECT id, rn, SUM(rn) OVER (PARTITION BY d ORDER BY id) AS s FROM (SELECT id, d, ROW_NUMBER() OVER (PARTITION BY e ORDER BY id) AS rn FROM t1)",
        ] {
            assert_windows(sql).await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn ordered_results() {
        for sql in [
            // The order of the window satisfies the `ORDER BY`: DataFusion
            // relies on the order of the output of the window.
            "SELECT id, ROW_NUMBER() OVER (ORDER BY id) AS rn FROM t1 ORDER BY id",
            "SELECT id, SUM(b) OVER (ORDER BY id ROWS BETWEEN 2 PRECEDING AND 1 FOLLOWING) AS s FROM t1 ORDER BY id LIMIT 100",
            "SELECT id, ROW_NUMBER() OVER (ORDER BY id DESC) AS rn FROM t1 ORDER BY rn",
            // A merge of the sorted partitions of the window, or a sort.
            "SELECT id, ROW_NUMBER() OVER (PARTITION BY d ORDER BY id) AS rn FROM t1 ORDER BY d, id",
            "SELECT id, RANK() OVER (PARTITION BY e ORDER BY a) AS r FROM t1 ORDER BY r, id",
            "SELECT id, rn FROM (SELECT id, ROW_NUMBER() OVER (PARTITION BY d ORDER BY b DESC, id) AS rn FROM t1) WHERE rn <= 3 ORDER BY id",
        ] {
            assert_windows(sql).await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn empty_and_small_inputs() {
        for sql in [
            "SELECT id, ROW_NUMBER() OVER (PARTITION BY d ORDER BY id) AS rn FROM t_empty",
            "SELECT id, SUM(b) OVER (ORDER BY id ROWS 1 PRECEDING) AS s FROM t_empty",
            "SELECT id, ROW_NUMBER() OVER (PARTITION BY d ORDER BY id) AS rn FROM t1 WHERE id = 7",
            "SELECT id, LEAD(id) OVER (ORDER BY id) AS n FROM t1 WHERE id < 2",
            "SELECT id, RANK() OVER (PARTITION BY d ORDER BY a) AS r FROM t1 WHERE a IS NULL",
        ] {
            assert_windows(sql).await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn windows_over_joins_and_aggregates() {
        for sql in [
            "SELECT d, SUM(a) AS s, RANK() OVER (ORDER BY SUM(a)) AS r FROM t1 GROUP BY d",
            "SELECT e, f, COUNT(*) AS n, ROW_NUMBER() OVER (PARTITION BY e ORDER BY f) AS rn FROM t1 GROUP BY e, f",
            "SELECT t1.id, t2.name, ROW_NUMBER() OVER (PARTITION BY t2.name ORDER BY t1.id, t2.w) AS rn FROM t1 JOIN t2 ON t1.a = t2.k",
            "SELECT k, w, SUM(w) OVER (PARTITION BY k ORDER BY w, name) AS s FROM t2 UNION ALL SELECT a, b, b + 1 FROM t1 WHERE id < 10",
            // A window over a union, and computed sort keys.
            "SELECT x, RANK() OVER (ORDER BY x) AS r FROM (SELECT id AS x FROM t1 UNION ALL SELECT CAST(k AS BIGINT) FROM t2)",
            "SELECT id, ROW_NUMBER() OVER (PARTITION BY e ORDER BY a + b, id) AS rn FROM t1",
            "SELECT id, LAG(id) OVER (ORDER BY b * 2 DESC, id) AS p FROM t1",
        ] {
            assert_windows(sql).await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn partitionings() {
        let unrepartitioned = {
            let mut config = test_session_config();
            config.options_mut().optimizer.repartition_windows = false;
            config
        };
        let configs = [1, 3, 16]
            .into_iter()
            .map(|partitions| test_session_config().with_target_partitions(partitions))
            .chain([unrepartitioned]);
        for config in configs {
            for sql in [
                "SELECT id, ROW_NUMBER() OVER (PARTITION BY d ORDER BY id) AS rn FROM t1",
                "SELECT id, ROW_NUMBER() OVER (PARTITION BY d ORDER BY id) AS rn FROM t1 ORDER BY d, id",
                "SELECT id, SUM(b) OVER (PARTITION BY e ORDER BY a GROUPS BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS s FROM t1",
                "SELECT id, LAG(a) OVER (ORDER BY id) AS prev, RANK() OVER (PARTITION BY e ORDER BY a, id) AS r FROM t1",
            ] {
                assert_windows_with_config(sql, config.clone(), true).await;
            }
        }
    }

    // ---------------------------------------------------------------------
    // Stages
    // ---------------------------------------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn stacked_windows_are_stages_of_one_window() {
        // The queries, with the number of native windows with 1 and with 4
        // target partitions.
        for (sql, windows) in [
            // The second window reads the rows in the order in which the first
            // one evaluated them: its row numbers follow the first one's.
            (
                "SELECT COUNT(*) FROM (SELECT ROW_NUMBER() OVER (ORDER BY a, b) AS r1, ROW_NUMBER() OVER (ORDER BY a) AS r2 FROM t1) WHERE r1 <> r2",
                [1, 1],
            ),
            (
                "SELECT COUNT(*) FROM (SELECT id, r, LAG(r) OVER (PARTITION BY d ORDER BY a) AS p FROM (SELECT id, d, a, RANK() OVER (PARTITION BY d ORDER BY a) AS r FROM t1)) WHERE p > r",
                [1, 1],
            ),
            (
                "SELECT id, ROW_NUMBER() OVER (PARTITION BY d ORDER BY a, b, id) AS r1, SUM(b) OVER (PARTITION BY d ORDER BY a) AS s, COUNT(*) OVER (PARTITION BY d) AS n FROM t1",
                [1, 1],
            ),
            // Coarser partitions in a later stage: groups of the first key. With
            // several partitions, DataFusion repartitions the rows (and sorts
            // them again) between the windows.
            (
                "SELECT id, ROW_NUMBER() OVER (PARTITION BY d, e ORDER BY id) AS r1, SUM(a) OVER (PARTITION BY d ORDER BY e, id) AS s FROM t1",
                [1, 2],
            ),
            // A projection between the windows that drops and reorders columns.
            (
                "SELECT id, rn, SUM(rn) OVER (PARTITION BY d ORDER BY id) AS s FROM (SELECT id, d, ROW_NUMBER() OVER (PARTITION BY d ORDER BY id) AS rn FROM t1)",
                [1, 1],
            ),
            // A sort between the windows: two windows.
            (
                "SELECT id, LAG(a) OVER (ORDER BY id) AS prev, RANK() OVER (PARTITION BY e ORDER BY a, id) AS r FROM t1",
                [2, 2],
            ),
        ] {
            for (partitions, windows) in [1, 4].into_iter().zip(windows) {
                let config = test_session_config().with_target_partitions(partitions);
                let plan = physical_plan(sql, config.clone()).await;
                let root = assert_native_parity_with_config(sql, config).await;
                let tree = format_operator_tree(&root);
                let context = format!(
                    "query: {sql}\n{}\n{tree}",
                    displayable(plan.as_ref()).indent(true)
                );
                assert_eq!(stages(&root), window_nodes(&plan), "{context}");
                assert_eq!(count::<PhysicalWindow>(&root), windows, "{context}");
            }
        }
    }

    /// Assert that `sql` matches DataFusion with 1, 4 and 16 target
    /// partitions, with all of its window nodes lowered to native windows: as
    /// many as `windows` says (with 1 target partition, with more).
    async fn assert_stacked_windows(sql: &str, windows: [usize; 2]) {
        for (partitions, windows) in [(1, windows[0]), (4, windows[1]), (16, windows[1])]
        {
            let config = test_session_config().with_target_partitions(partitions);
            let plan = physical_plan(sql, config.clone()).await;
            let root = assert_native_parity_with_config(sql, config).await;
            let tree = format_operator_tree(&root);
            let context = format!(
                "query: {sql}\n{} partitions\n{}\n{tree}",
                partitions,
                displayable(plan.as_ref()).indent(true)
            );
            assert_eq!(stages(&root), window_nodes(&plan), "{context}");
            assert_eq!(count::<PhysicalWindow>(&root), windows, "{context}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn windows_stacked_on_windows_of_other_partitions_or_orders_are_native() {
        // DataFusion sorts or merges the rows between the windows (as for the
        // window queries of the stress corpus that used to be left to it). The
        // upper window is a native window over the native window of the lower
        // one, which partitions and sorts its rows again, or a re-sorting stage
        // of the lower one if its window partitions are within the groups of
        // the lower one (see
        // `windows_of_the_same_partitions_in_other_orders_are_re_sorting_stages`).
        for (sql, windows) in [
            // A merge of the sorted partitions of the lower window, for an
            // ordered window by its partition keys (a stage of the lower window
            // on one partition, where DataFusion does not sort in between).
            (
                "SELECT id, a, c, RANK() OVER (PARTITION BY a ORDER BY c DESC, id) AS r, DENSE_RANK() OVER (ORDER BY a) AS dr FROM t1",
                [1, 2],
            ),
            // Three windows of different partitions.
            (
                "SELECT id, ROW_NUMBER() OVER (PARTITION BY e ORDER BY id) AS r1, SUM(b) OVER (PARTITION BY d ORDER BY id) AS s, COUNT(*) OVER (ORDER BY id DESC) AS n FROM t1",
                [3, 3],
            ),
            // A window over a window, through a projection that computes a
            // value.
            (
                "SELECT id, rn, SUM(x) OVER (PARTITION BY e ORDER BY id) AS s FROM (SELECT id, e, rn, rn * 2 AS x FROM (SELECT id, e, ROW_NUMBER() OVER (PARTITION BY d ORDER BY id) AS rn FROM t1))",
                [2, 2],
            ),
        ] {
            assert_stacked_windows(sql, windows).await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn windows_of_the_same_partitions_in_other_orders_are_re_sorting_stages() {
        // A window of several partitions whose partition keys include those of
        // the window below it, in another order (DataFusion sorts the rows of
        // every partition again between them): a stage of the native window of
        // the window below that sorts the rows of every group again. Windows of
        // a single partition, whose rows are emitted in order, are windows of
        // their own.
        for (sql, windows) in [
            (
                "SELECT d, id, b, NTILE(4) OVER (PARTITION BY d ORDER BY b, id) AS q, FIRST_VALUE(id) OVER (PARTITION BY d ORDER BY b DESC, id) AS f FROM t1",
                [2, 1],
            ),
            (
                "SELECT h, id, b, NTILE(4) OVER (PARTITION BY h ORDER BY b, id) AS q, LAST_VALUE(id) OVER (PARTITION BY h ORDER BY b DESC, id ROWS BETWEEN 1 PRECEDING AND 1 FOLLOWING) AS f FROM t1",
                [2, 1],
            ),
            // Three stages, two of which sort again.
            (
                "SELECT id, SUM(b) OVER (PARTITION BY d ORDER BY id) AS s, LAG(id) OVER (PARTITION BY d ORDER BY b DESC, id) AS l, RANK() OVER (PARTITION BY d ORDER BY a NULLS FIRST, id) AS r FROM t1",
                [3, 1],
            ),
            // Finer partitions in the re-sorting stage (several of its window
            // partitions in a group).
            (
                "SELECT id, ROW_NUMBER() OVER (PARTITION BY d ORDER BY id) AS r1, ROW_NUMBER() OVER (PARTITION BY d, e ORDER BY b, id) AS r2 FROM t1",
                [2, 1],
            ),
            // A computed partition key (projected below the windows).
            (
                "SELECT id, COUNT(*) OVER (PARTITION BY a % 3 ORDER BY id) AS n1, SUM(b) OVER (PARTITION BY a % 3 ORDER BY b, id) AS n2 FROM t1",
                [2, 1],
            ),
            // Through a projection of columns, over a stage.
            (
                "SELECT id, ROW_NUMBER() OVER (PARTITION BY d ORDER BY id) AS r1, SUM(rn) OVER (PARTITION BY d ORDER BY b, id) AS s FROM (SELECT id, d, b, ROW_NUMBER() OVER (PARTITION BY d ORDER BY a, id) AS rn FROM t1)",
                [3, 1],
            ),
            // Partitions of the upper window that span several groups of the
            // lower one: a window of its own.
            (
                "SELECT id, ROW_NUMBER() OVER (PARTITION BY d, e ORDER BY id) AS r1, SUM(a) OVER (PARTITION BY e ORDER BY b, id) AS s FROM t1",
                [2, 2],
            ),
        ] {
            assert_stacked_windows(sql, windows).await;
        }
        // An empty input (of a single partition, which DataFusion does not
        // repartition: ordered windows).
        assert_stacked_windows(
            "SELECT id, NTILE(4) OVER (PARTITION BY d ORDER BY b, id) AS q, FIRST_VALUE(id) OVER (PARTITION BY d ORDER BY b DESC, id) AS f FROM t_empty",
            [2, 2],
        )
        .await;
    }

    // ---------------------------------------------------------------------
    // Lowering
    // ---------------------------------------------------------------------

    /// The test tables, and `sorted`: the rows of `t1` in a single partition,
    /// declared sorted by `a` (which has ties), in an order of `a` in which
    /// the ties are not in the order of `id`.
    async fn setup_sorted(ctx: &SessionContext) -> Result<()> {
        register_test_tables(ctx)?;
        let batches = test_table_t1()?.into_iter().flatten().collect::<Vec<_>>();
        let schema = batches[0].schema();
        let all = concat_batches(&schema, &batches)?;
        let a = all.column(1).as_primitive::<arrow::datatypes::Int32Type>();
        let mut order = (0..all.num_rows()).collect::<Vec<_>>();
        // Sorted by `a` (NULLs last), ties in descending order of `id`.
        order.sort_by_key(|&row| (a.is_null(row), a.value(row), std::cmp::Reverse(row)));
        let indices =
            UInt32Array::from_iter_values(order.into_iter().map(|row| row as u32));
        let sorted = take_record_batch(&all, &indices)?;
        let table = MemTable::try_new(schema, vec![vec![sorted]])?
            .with_sort_order(vec![vec![datafusion::prelude::col("a").sort(true, false)]]);
        ctx.register_table("sorted", Arc::new(table))?;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn windows_over_inputs_in_any_order_are_native() {
        // DataFusion reads the rows of a sorted input in its order, ties
        // included; the native window sorts them by that order, and its peers
        // come in any order. The functions whose values do not depend on the
        // order of peers return DataFusion's values.
        for sql in [
            "SELECT id, RANK() OVER (ORDER BY a) AS r FROM sorted",
            "SELECT id, COUNT(*) OVER (ORDER BY a) AS n FROM sorted",
            "SELECT id, SUM(b) OVER (ORDER BY a) AS s FROM sorted",
            "SELECT id, ROW_NUMBER() OVER (PARTITION BY d ORDER BY id) AS rn FROM sorted",
            // Without order, any row number.
            "SELECT ROW_NUMBER() OVER () AS rn FROM sorted",
        ] {
            let root =
                assert_native_parity_with_setup(sql, test_session_config(), setup_sorted)
                    .await;
            assert_eq!(count::<PhysicalWindow>(&root), 1, "query: {sql}");
        }
        // The row numbers of the peers of `a`, in any order: every row once,
        // with its rank among the rows of other values of `a`.
        let ctx = datafusion_context(test_session_config());
        setup_sorted(&ctx).await.unwrap();
        let sql = "SELECT id, a, ROW_NUMBER() OVER (ORDER BY a) AS rn, \
                   RANK() OVER (ORDER BY a) AS r, COUNT(*) OVER (ORDER BY a) AS n FROM sorted";
        let plan = ctx
            .sql(sql)
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        let root = lower(&plan).unwrap();
        assert_eq!(count::<PhysicalWindow>(&root), 1);
        let exec =
            Arc::new(crate::bridge::PipelineRootExec::try_new(root, plan).unwrap());
        let output = datafusion_physical_plan::collect(exec, ctx.task_ctx())
            .await
            .unwrap();
        let mut ids = vec![];
        let mut numbers = vec![];
        for batch in &output {
            let id = batch.column(0).as_primitive::<Int64Type>();
            let rn = batch.column(2).as_primitive::<UInt64Type>();
            let r = batch.column(3).as_primitive::<UInt64Type>();
            let n = batch.column(4).as_primitive::<Int64Type>();
            for row in 0..batch.num_rows() {
                // The row number of a row is among those of its peers: from
                // its rank to the number of rows up to its last peer.
                let number = rn.value(row);
                assert!(r.value(row) <= number && number <= n.value(row) as u64);
                ids.push(id.value(row));
                numbers.push(number);
            }
        }
        ids.sort_unstable();
        numbers.sort_unstable();
        assert_eq!(ids, (0..5000).collect::<Vec<_>>());
        assert_eq!(numbers, (1..=5000).collect::<Vec<_>>());
        // A `BoundedWindowAggExec` over a merge of partitions, which emits as
        // it reads its input in DataFusion.
        assert_windows("SELECT ROW_NUMBER() OVER () AS rn FROM t1").await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn windows_of_expressions_that_can_fail_are_native() {
        // Arguments that can fail, a `FILTER`, and an average of decimals,
        // which fails where it does not fit its type: the native window
        // evaluates them on whole window partitions.
        for sql in [
            "SELECT id, SUM(10 / (a + 51)) OVER (PARTITION BY d ORDER BY id) AS s FROM t1",
            "SELECT id, SUM(a) FILTER (WHERE a > 0) OVER (PARTITION BY d ORDER BY id) AS s FROM t1",
            "SELECT id, COUNT(b) FILTER (WHERE b % 3 = 0) OVER (ORDER BY id ROWS BETWEEN 5 PRECEDING AND 5 FOLLOWING) AS n FROM t1",
            "SELECT id, AVG(g) OVER (PARTITION BY e ORDER BY id ROWS UNBOUNDED PRECEDING) AS a FROM t1",
            "SELECT id, SUM(a * 2 + 1) OVER (PARTITION BY d ORDER BY id) AS s FROM t1",
            "SELECT id, ROW_NUMBER() OVER (PARTITION BY 100 / (a + 51) ORDER BY 10 / (b + 1), id) AS rn FROM t1",
        ] {
            assert_windows(sql).await;
        }
    }

    /// A user-defined window function that returns its argument, and fails on
    /// values from 10 on: DataFusion's `BoundedWindowAggExec` evaluates it
    /// row by row, as it reads its input.
    #[derive(Debug, PartialEq, Eq, Hash)]
    struct Checked {
        name: &'static str,
        signature: datafusion_expr::Signature,
    }

    /// The error of [`Checked`] and [`CheckedSum`].
    const CHECKED_FAILURE: &str = "is not checked";

    impl Checked {
        fn udwf(name: &'static str) -> datafusion_expr::WindowUDF {
            datafusion_expr::WindowUDF::from(Self {
                name,
                signature: datafusion_expr::Signature::exact(
                    vec![DataType::Int64],
                    datafusion_expr::Volatility::Immutable,
                ),
            })
        }
    }

    impl datafusion_expr::WindowUDFImpl for Checked {
        fn name(&self) -> &str {
            self.name
        }

        fn signature(&self) -> &datafusion_expr::Signature {
            &self.signature
        }

        fn partition_evaluator(
            &self,
            _args: datafusion_expr::function::PartitionEvaluatorArgs,
        ) -> Result<Box<dyn datafusion_expr::PartitionEvaluator>> {
            Ok(Box::new(CheckedEvaluator))
        }

        fn field(
            &self,
            field_args: datafusion_expr::function::WindowUDFFieldArgs,
        ) -> Result<arrow::datatypes::FieldRef> {
            Ok(Arc::new(Field::new(
                field_args.name(),
                DataType::Int64,
                true,
            )))
        }
    }

    #[derive(Debug)]
    struct CheckedEvaluator;

    /// `value`, or an error if it is 10 or more.
    fn checked(value: i64) -> Result<i64> {
        if value >= 10 {
            return datafusion_common::exec_err!("{value} {CHECKED_FAILURE}");
        }
        Ok(value)
    }

    impl datafusion_expr::PartitionEvaluator for CheckedEvaluator {
        fn supports_bounded_execution(&self) -> bool {
            true
        }

        fn evaluate(
            &mut self,
            values: &[ArrayRef],
            range: &Range<usize>,
        ) -> Result<datafusion_common::ScalarValue> {
            let value = values[0].as_primitive::<Int64Type>().value(range.start);
            Ok(datafusion_common::ScalarValue::Int64(Some(checked(value)?)))
        }
    }

    /// A user-defined aggregate: the sum of values below 10, which fails on
    /// values from 10 on.
    #[derive(Debug, Default)]
    struct CheckedSum(i64);

    impl datafusion_expr::Accumulator for CheckedSum {
        fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
            for value in values[0].as_primitive::<Int64Type>().iter().flatten() {
                self.0 += checked(value)?;
            }
            Ok(())
        }

        fn evaluate(&mut self) -> Result<datafusion_common::ScalarValue> {
            Ok(datafusion_common::ScalarValue::Int64(Some(self.0)))
        }

        fn size(&self) -> usize {
            size_of_val(self)
        }

        fn state(&mut self) -> Result<Vec<datafusion_common::ScalarValue>> {
            Ok(vec![self.evaluate()?])
        }

        fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
            self.update_batch(states)
        }
    }

    /// A session with the table `t` of the values `0..15` of `x` (in another
    /// order, in batches of 2 rows), the window function [`Checked`] (as
    /// `checked`, and as `lag`, which it replaces), and the aggregate
    /// [`CheckedSum`] (as `checked_sum`).
    async fn checked_setup(ctx: &SessionContext) -> Result<()> {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int64, false)]));
        let values = [7, 3, 14, 0, 9, 12, 1, 5, 11, 2, 8, 13, 4, 10, 6];
        let batches = values
            .chunks(2)
            .map(|values| {
                let values = Int64Array::from_iter_values(values.iter().copied());
                RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(values)])
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        ctx.register_table("t", Arc::new(MemTable::try_new(schema, vec![batches])?))?;
        ctx.register_udwf(Checked::udwf("checked"));
        ctx.register_udwf(Checked::udwf("lag"));
        ctx.register_udaf(datafusion_expr::create_udaf(
            "checked_sum",
            vec![DataType::Int64],
            Arc::new(DataType::Int64),
            datafusion_expr::Volatility::Immutable,
            Arc::new(|_| Ok(Box::new(CheckedSum::default()))),
            Arc::new(vec![DataType::Int64]),
        ));
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn user_defined_window_functions_are_native() -> Result<()> {
        use crate::bridge::PipelineRootExec;
        use crate::test_utils::run_sql;
        let config = || test_session_config().with_batch_size(2);
        // User-defined window functions and aggregates (also under the name of
        // one of DataFusion's) are evaluated like DataFusion's.
        for sql in [
            "SELECT x, checked(x) OVER (ORDER BY x) AS c FROM t WHERE x < 10",
            "SELECT x, lag(x) OVER (ORDER BY x DESC) AS c FROM t WHERE x < 10",
            "SELECT x, checked_sum(x) OVER (PARTITION BY x % 2 ORDER BY x) AS s FROM t WHERE x < 10",
            "SELECT x, checked_sum(x) OVER (ORDER BY x ROWS BETWEEN UNBOUNDED PRECEDING AND 1 FOLLOWING) AS s FROM t WHERE x < 8",
        ] {
            let root =
                assert_native_parity_with_setup(sql, config(), checked_setup).await;
            assert_eq!(count::<PhysicalWindow>(&root), 1, "query: {sql}");
        }
        // DataFusion sorts the rows, and then evaluates the functions on the
        // sorted batches as it reads them: the limit stops it before the first
        // value that fails. The native window evaluates all rows of the window
        // partition it reads, and may fail: whether an error is raised for rows
        // beyond a limit is the engine's own. Without a limit, both fail.
        for (sql, datafusion_fails) in [
            (
                "SELECT x, checked(x) OVER (ORDER BY x) AS c FROM t ORDER BY x LIMIT 1",
                false,
            ),
            (
                "SELECT x, lag(x) OVER (ORDER BY x) AS c FROM t ORDER BY x LIMIT 1",
                false,
            ),
            (
                "SELECT x, checked_sum(x) OVER (ORDER BY x) AS s FROM t ORDER BY x LIMIT 1",
                false,
            ),
            ("SELECT x, checked(x) OVER (ORDER BY x) AS c FROM t", true),
            (
                "SELECT x, checked_sum(x) OVER (PARTITION BY x % 2 ORDER BY x) AS s FROM t",
                true,
            ),
        ] {
            let ctx = datafusion_context(config());
            checked_setup(&ctx).await?;
            let expected = run_sql(&ctx, sql).await;
            assert_eq!(expected.is_err(), datafusion_fails, "query: {sql}");
            let plan = ctx.sql(sql).await?.create_physical_plan().await?;
            let root = lower(&plan)?;
            assert_eq!(count::<PhysicalWindow>(&root), 1, "query: {sql}");
            let exec: Arc<dyn ExecutionPlan> =
                Arc::new(PipelineRootExec::try_new(root, plan)?);
            let actual =
                datafusion_physical_plan::collect(Arc::clone(&exec), ctx.task_ctx())
                    .await;
            match (expected, actual) {
                (Ok(expected), Ok(batches)) => {
                    let actual = crate::test_utils::EngineOutput {
                        schema: exec.schema(),
                        plan: exec,
                        batches,
                        ordered: expected.ordered,
                    };
                    let compared = crate::test_utils::compare_outputs(&expected, &actual);
                    assert!(compared.is_ok(), "query: {sql}: {compared:?}");
                }
                (_, Err(error)) => {
                    assert!(error.to_string().contains(CHECKED_FAILURE), "{error}");
                }
                (Err(error), Ok(_)) => {
                    panic!("query: {sql}: only DataFusion fails: {error}")
                }
            }
        }
        // DataFusion's functions, also with these batches.
        let root = assert_native_parity_with_setup(
            "SELECT x, row_number() OVER (ORDER BY x) AS r, sum(x) OVER (ORDER BY x) AS s FROM t ORDER BY x LIMIT 3",
            config(),
            checked_setup,
        )
        .await;
        assert_eq!(count::<PhysicalWindow>(&root), 1);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn narrow_dictionaries_are_native() {
        let setup = async |ctx: &SessionContext| {
            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int64, false),
                Field::new(
                    "s",
                    DataType::Dictionary(
                        Box::new(DataType::Int8),
                        Box::new(DataType::Utf8),
                    ),
                    true,
                ),
            ]));
            // 120 distinct values in all, which fit into the keys.
            let partitions = (0..3)
                .map(|p| {
                    let ids = Int64Array::from_iter_values((0..40).map(|i| p * 40 + i));
                    let values = (0..40)
                        .map(|i| format!("value {}", p * 40 + i))
                        .collect::<Vec<_>>();
                    let strings = values
                        .iter()
                        .map(String::as_str)
                        .collect::<arrow::array::DictionaryArray<
                        arrow::datatypes::Int8Type,
                    >>();
                    let batch = RecordBatch::try_new(
                        Arc::clone(&schema),
                        vec![Arc::new(ids), Arc::new(strings)],
                    )?;
                    Ok(vec![batch])
                })
                .collect::<Result<Vec<_>>>()?;
            ctx.register_table("dict", Arc::new(MemTable::try_new(schema, partitions)?))?;
            Ok(())
        };
        for sql in [
            "SELECT id, s, ROW_NUMBER() OVER (ORDER BY id) AS rn FROM dict",
            "SELECT id, s, FIRST_VALUE(s) OVER (PARTITION BY id % 3 ORDER BY id) AS f FROM dict",
        ] {
            let root =
                assert_native_parity_with_setup(sql, test_session_config(), setup).await;
            assert_eq!(count::<PhysicalWindow>(&root), 1, "query: {sql}");
        }
    }

    // ---------------------------------------------------------------------
    // Execution
    // ---------------------------------------------------------------------

    /// A window over the values `0..n` of [`values_schema`] in chunks of
    /// `size` rows (in a scrambled order), with `ROW_NUMBER()` by `v % 7`
    /// ordered by `v` if `partitioned`, by `v` otherwise.
    fn row_numbers(n: i64, size: usize, partitioned: bool) -> Result<PhysicalWindow> {
        let scrambled = (0..n).map(|i| (i * 7919) % n).collect::<Vec<_>>();
        let chunks = scrambled
            .chunks(size)
            .map(|values| values_chunk(values.iter().copied()))
            .collect();
        let input = Arc::new(TestSource::with_schema(values_schema(), chunks));
        let schema = values_schema();
        let v = col("v", &schema)?;
        let partition_by = if partitioned {
            vec![
                Arc::new(datafusion_physical_expr::expressions::BinaryExpr::new(
                    Arc::clone(&v),
                    datafusion_expr::Operator::Modulo,
                    datafusion_physical_expr::expressions::lit(7_i64),
                )) as Arc<dyn PhysicalExpr>,
            ]
        } else {
            vec![]
        };
        let order_by = LexOrdering::new([PhysicalSortExpr::new_default(Arc::clone(&v))])
            .expect("an ordering");
        let window_expr = datafusion_physical_plan::windows::create_window_expr(
            &datafusion_expr::WindowFunctionDefinition::WindowUDF(
                datafusion::functions_window::row_number::row_number_udwf(),
            ),
            "rn".to_string(),
            &[],
            &partition_by,
            order_by.as_ref(),
            Arc::new(datafusion_expr::WindowFrame::new(Some(true))),
            Arc::clone(&schema),
            false,
            false,
            None,
        )?;
        let mut ordering = partition_by
            .iter()
            .map(|key| PhysicalSortExpr::new_default(Arc::clone(key)))
            .collect::<Vec<_>>();
        ordering.extend(order_by.iter().cloned());
        PhysicalWindow::try_new(
            input,
            vec![window_expr],
            LexOrdering::new(ordering),
            !partitioned,
        )
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn ordered_windows_emit_their_rows_in_order() -> Result<()> {
        let n = 10_007;
        for threads in [1, 4] {
            let window = row_numbers(n, 1000, false)?;
            assert!(window.is_ordered());
            let output = run(Arc::new(window), threads).await?;
            assert!(
                output
                    .iter()
                    .all(|batch| batch.num_rows() <= DEFAULT_VECTOR_SIZE)
            );
            let values = output
                .iter()
                .flat_map(|batch| {
                    batch
                        .column(0)
                        .as_primitive::<Int64Type>()
                        .values()
                        .to_vec()
                })
                .collect::<Vec<_>>();
            let numbers = output
                .iter()
                .flat_map(|batch| {
                    batch
                        .column(1)
                        .as_primitive::<UInt64Type>()
                        .values()
                        .to_vec()
                })
                .collect::<Vec<_>>();
            assert_eq!(values, (0..n).collect::<Vec<_>>());
            assert_eq!(numbers, (1..=n as u64).collect::<Vec<_>>());
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn partitioned_windows_evaluate_every_partition() -> Result<()> {
        let n = 10_007;
        for threads in [1, 4, 7] {
            let window = row_numbers(n, 300, true)?;
            assert!(!window.is_ordered());
            let output = run(Arc::new(window), threads).await?;
            assert!(
                output
                    .iter()
                    .all(|batch| batch.num_rows() <= DEFAULT_VECTOR_SIZE)
            );
            let mut rows = output
                .iter()
                .flat_map(|batch| {
                    let values = batch.column(0).as_primitive::<Int64Type>();
                    let numbers = batch.column(1).as_primitive::<UInt64Type>();
                    values
                        .values()
                        .iter()
                        .copied()
                        .zip(numbers.values().iter().copied())
                })
                .collect::<Vec<_>>();
            rows.sort_unstable();
            // The row number of `v` in its partition `v % 7` is `v / 7 + 1`.
            let expected = (0..n).map(|v| (v, (v / 7 + 1) as u64)).collect::<Vec<_>>();
            assert_eq!(rows, expected, "{threads} threads");
        }
        Ok(())
    }

    #[test]
    fn pipelines_of_a_window() -> Result<()> {
        let window: Arc<dyn PhysicalOperator> = Arc::new(row_numbers(10, 3, true)?);
        let pipelines = format_pipelines(&window)?;
        assert_eq!(
            pipelines,
            "MetaPipeline 0 (sink: ResultSink) children: [1]\n\
             \x20 Pipeline 0: PhysicalWindow -> ResultSink depends on meta pipelines [1]\n\
             MetaPipeline 1 (sink: PhysicalWindow)\n\
             \x20 Pipeline 1: TestSource -> PhysicalWindow\n\
             \x20 Pipeline 2: WindowPartitionSort -> PhysicalWindow depends on pipelines [1]\n"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn windows_without_rows_emit_nothing() -> Result<()> {
        let schema = values_schema();
        let input = Arc::new(TestSource::with_schema(Arc::clone(&schema), vec![]));
        let window = row_numbers(0, 1, true)?;
        let window = PhysicalWindow::try_new(
            input,
            window.window_exprs()[0].to_vec(),
            window.sort_order().cloned(),
            false,
        )?;
        assert!(run(Arc::new(window), 4).await?.is_empty());
        Ok(())
    }

    #[test]
    fn groups_continue_across_sorted_batches() -> Result<()> {
        // `d` is the partition key: the rows of a group can be split over
        // several sorted batches (e.g. if they do not fit into one).
        let schema = Arc::new(Schema::new(vec![
            Field::new("d", DataType::Utf8, true),
            Field::new("v", DataType::Int32, false),
        ]));
        let batch = |d: Vec<Option<&str>>, v: Vec<i32>| {
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(StringArray::from(d)) as ArrayRef,
                    Arc::new(Int32Array::from(v)),
                ],
            )
            .unwrap()
        };
        let batches = vec![
            batch(vec![Some("a"), Some("a"), Some("b")], vec![1, 2, 3]),
            batch(vec![Some("b"), Some("c")], vec![4, 5]),
            batch(vec![Some("c")], vec![6]),
            batch(vec![None, None], vec![7, 8]),
        ];
        let d = col("d", &schema)?;
        let window_expr = datafusion_physical_plan::windows::create_window_expr(
            &datafusion_expr::WindowFunctionDefinition::WindowUDF(
                datafusion::functions_window::row_number::row_number_udwf(),
            ),
            "rn".to_string(),
            &[],
            std::slice::from_ref(&d),
            &[],
            Arc::new(datafusion_expr::WindowFrame::new(None)),
            Arc::clone(&schema),
            false,
            false,
            None,
        )?;
        let input = Arc::new(TestSource::with_schema(Arc::clone(&schema), vec![]));
        let ordering = LexOrdering::new([PhysicalSortExpr::new_default(d)]);
        let window = PhysicalWindow::try_new(input, vec![window_expr], ordering, false)?;
        let pool: Arc<dyn datafusion_execution::memory_pool::MemoryPool> =
            Arc::new(datafusion_execution::memory_pool::UnboundedMemoryPool::default());
        let reservation = MemoryConsumer::new("test").register(&pool);
        let groups = window.groups(&batches, &mut ReservedMemory::new(&reservation))?;
        assert_eq!(groups, vec![0..2, 2..4, 4..6, 6..8]);
        assert_eq!(reservation.size(), 0);
        let bucket = SortedBucket::new(batches, groups, reservation);
        let (rows, memory) = window.evaluate_group(&bucket, 2..4)?;
        // The rows of the group, with their window values, which are reserved.
        assert_eq!(bucket.reservation.size(), memory);
        assert!(memory >= 2 * size_of::<u64>(), "{memory}");
        assert_eq!(rows.num_rows(), 2);
        assert_eq!(
            rows.column(1)
                .as_primitive::<arrow::datatypes::Int32Type>()
                .values(),
            &[3, 4]
        );
        assert_eq!(
            rows.column(2).as_primitive::<UInt64Type>().values(),
            &[1, 2]
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn windows_can_be_executed_again() -> Result<()> {
        let ctx = datafusion_context(test_session_config());
        register_test_tables(&ctx)?;
        let plan = ctx
            .sql(
                "SELECT id, ROW_NUMBER() OVER (PARTITION BY d ORDER BY id) AS rn FROM t1",
            )
            .await?
            .create_physical_plan()
            .await?;
        let root = lower(&plan)?;
        let task_ctx = Arc::new(TaskContext::default());
        let first =
            Executor::collect(Arc::clone(&root), Arc::clone(&task_ctx), 4).await?;
        let second = Executor::collect(root, task_ctx, 4).await?;
        let rows = |batches: &[RecordBatch]| {
            batches.iter().map(RecordBatch::num_rows).sum::<usize>()
        };
        assert_eq!(rows(&first), 5000);
        assert_eq!(rows(&second), 5000);
        Ok(())
    }

    // ---------------------------------------------------------------------
    // Hand-built plans
    // ---------------------------------------------------------------------

    /// A scan of the rows of `t1` (in its 4 partitions).
    fn t1_scan() -> Result<Arc<dyn ExecutionPlan>> {
        let partitions = test_table_t1()?;
        let schema = partitions[0][0].schema();
        Ok(MemorySourceConfig::try_new_exec(&partitions, schema, None)?)
    }

    /// `ROW_NUMBER() OVER (PARTITION BY d ORDER BY id)` over the rows of
    /// `input` (of the schema of `t1`), in the window node of `mode`.
    fn row_number_plan(
        input: Arc<dyn ExecutionPlan>,
        mode: Option<datafusion_physical_plan::InputOrderMode>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let schema = input.schema();
        let id = PhysicalSortExpr::new_default(col("id", &schema)?);
        let window_expr = datafusion_physical_plan::windows::create_window_expr(
            &datafusion_expr::WindowFunctionDefinition::WindowUDF(
                datafusion::functions_window::row_number::row_number_udwf(),
            ),
            "rn".to_string(),
            &[],
            &[col("d", &schema)?],
            std::slice::from_ref(&id),
            Arc::new(datafusion_expr::WindowFrame::new(Some(true))),
            Arc::clone(&schema),
            false,
            false,
            None,
        )?;
        Ok(match mode {
            Some(mode) => Arc::new(BoundedWindowAggExec::try_new(
                vec![window_expr],
                input,
                mode,
                true,
            )?),
            None => Arc::new(WindowAggExec::try_new(vec![window_expr], input, true)?),
        })
    }

    fn sort_exec(
        input: Arc<dyn ExecutionPlan>,
        keys: &[&str],
        preserve_partitioning: bool,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let schema = input.schema();
        let ordering = LexOrdering::new(
            keys.iter()
                .map(|key| Ok(PhysicalSortExpr::new_default(col(key, &schema)?)))
                .collect::<Result<Vec<_>>>()?,
        )
        .expect("sort keys");
        Ok(Arc::new(
            SortExec::new(ordering, input)
                .with_preserve_partitioning(preserve_partitioning),
        ))
    }

    fn hash_repartition(input: Arc<dyn ExecutionPlan>) -> Result<Arc<dyn ExecutionPlan>> {
        let d = col("d", &input.schema())?;
        Ok(Arc::new(RepartitionExec::try_new(
            input,
            Partitioning::Hash(vec![d], 4),
        )?))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn windows_of_all_input_order_modes() -> Result<()> {
        use crate::operators::union::test_support::assert_lowered_plan_byte_eq;
        use datafusion_physical_plan::InputOrderMode;
        let task_ctx = || {
            Arc::new(TaskContext::default().with_session_config(test_session_config()))
        };
        // Partitioned by `d`, sorted by `d` and `id` (`Sorted`) or by `id` only
        // (`Linear`, `PartiallySorted` is the same here): the native window
        // sorts the rows of each partition by `d` and `id`.
        for (mode, keys) in [
            (Some(InputOrderMode::Sorted), vec!["d", "id"]),
            (None, vec!["d", "id"]),
            (Some(InputOrderMode::Linear), vec!["id"]),
        ] {
            let make = || {
                let sorted = sort_exec(hash_repartition(t1_scan()?)?, &keys, true)?;
                row_number_plan(sorted, mode.clone())
            };
            let root = assert_lowered_plan_byte_eq(make, lower, task_ctx(), false).await;
            assert_eq!(count::<PhysicalWindow>(&root), 1, "{mode:?}");
        }
        // In a single partition, the rows are emitted in the order of the
        // input, also if it does not start with the partition keys: the native
        // window sorts the rows it evaluated again, by the order of its input.
        for (mode, keys) in [
            (Some(InputOrderMode::Sorted), vec!["d", "id"]),
            (Some(InputOrderMode::Linear), vec!["id"]),
            (None, vec!["d", "id"]),
        ] {
            let make = || {
                let coalesced = Arc::new(CoalescePartitionsExec::new(t1_scan()?));
                row_number_plan(sort_exec(coalesced, &keys, false)?, mode.clone())
            };
            let root = assert_lowered_plan_byte_eq(make, lower, task_ctx(), true).await;
            assert_eq!(count::<PhysicalWindow>(&root), 1, "{mode:?}");
        }
        // Also in an order of other keys, under a limit, which takes the rows
        // of that order.
        let make = || {
            let coalesced = Arc::new(CoalescePartitionsExec::new(t1_scan()?));
            let sorted = sort_exec(coalesced, &["e", "id"], false)?;
            let window = row_number_plan(sorted, Some(InputOrderMode::Linear))?;
            let limit = datafusion_physical_plan::limit::GlobalLimitExec::new(
                window,
                10,
                Some(25),
            );
            Ok::<_, DataFusionError>(Arc::new(limit) as Arc<dyn ExecutionPlan>)
        };
        let root = assert_lowered_plan_byte_eq(make, lower, task_ctx(), true).await;
        assert_eq!(count::<PhysicalWindow>(&root), 1);
        Ok(())
    }

    /// A projection of `id`, `d` and the computed `b + 1 AS x` of the rows of
    /// `input` (of the schema of `t1`).
    fn project_x(input: Arc<dyn ExecutionPlan>) -> Result<Arc<dyn ExecutionPlan>> {
        let schema = input.schema();
        let x = datafusion_physical_expr::expressions::BinaryExpr::new(
            col("b", &schema)?,
            datafusion_expr::Operator::Plus,
            datafusion_physical_expr::expressions::lit(1_i64),
        );
        let exprs: Vec<(Arc<dyn PhysicalExpr>, String)> = vec![
            (col("id", &schema)?, "id".to_string()),
            (col("d", &schema)?, "d".to_string()),
            (Arc::new(x), "x".to_string()),
        ];
        Ok(Arc::new(ProjectionExec::try_new(exprs, input)?))
    }

    /// `SUM(x) OVER (PARTITION BY d ORDER BY id)` (if `partitioned`, `OVER
    /// (ORDER BY id)` otherwise) of the rows of `input`, which have the columns
    /// `id`, `d` and `x` (see [`project_x`]), in the window node of `mode` (a
    /// `WindowAggExec` if `None`).
    fn sum_of_x_plan(
        input: Arc<dyn ExecutionPlan>,
        partitioned: bool,
        mode: Option<datafusion_physical_plan::InputOrderMode>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let schema = input.schema();
        let partition_by = match partitioned {
            true => vec![col("d", &schema)?],
            false => vec![],
        };
        let id = PhysicalSortExpr::new_default(col("id", &schema)?);
        let window_expr = datafusion_physical_plan::windows::create_window_expr(
            &datafusion_expr::WindowFunctionDefinition::AggregateUDF(
                datafusion::functions_aggregate::sum::sum_udaf(),
            ),
            "s".to_string(),
            &[col("x", &schema)?],
            &partition_by,
            std::slice::from_ref(&id),
            Arc::new(datafusion_expr::WindowFrame::new(Some(true))),
            Arc::clone(&schema),
            false,
            false,
            None,
        )?;
        Ok(match mode {
            Some(mode) => Arc::new(BoundedWindowAggExec::try_new(
                vec![window_expr],
                input,
                mode,
                partitioned,
            )?),
            None => Arc::new(WindowAggExec::try_new(
                vec![window_expr],
                input,
                partitioned,
            )?),
        })
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn windows_over_projections_of_sorts_are_native() -> Result<()> {
        use crate::bridge::bridged_plans;
        use datafusion_physical_plan::InputOrderMode;
        // `Window -> ProjectionExec[id, d, b + 1 AS x] -> SortExec`, of a single
        // partition or of the hash partitions of `d`: the projection, which
        // computes a value, is lowered with the input of the native window,
        // below the sort that the native window replaces (a projection
        // computes every row on its own).
        let task_ctx = || {
            Arc::new(TaskContext::default().with_session_config(test_session_config()))
        };
        for partitioned in [false, true] {
            for mode in [None, Some(InputOrderMode::Sorted)] {
                let make = || {
                    let sorted = match partitioned {
                        true => {
                            sort_exec(hash_repartition(t1_scan()?)?, &["d", "id"], true)?
                        }
                        false => {
                            let coalesced =
                                Arc::new(CoalescePartitionsExec::new(t1_scan()?));
                            sort_exec(coalesced, &["id"], false)?
                        }
                    };
                    sum_of_x_plan(project_x(sorted)?, partitioned, mode.clone())
                };
                let case = format!("partitioned: {partitioned}, {mode:?}");
                let plan = make()?;
                let window = PhysicalWindow::try_lower(&plan, lower)?;
                let window = window.unwrap_or_else(|| panic!("{case}: not lowered"));
                assert_eq!(window.input().name(), "TestProjection", "{case}");
                let root =
                    assert_lowered_plan_byte_eq(make, lower, task_ctx(), !partitioned)
                        .await;
                let tree = format_operator_tree(&root);
                assert_eq!(count::<PhysicalWindow>(&root), 1, "{case}\n{tree}");
                assert!(bridged_plans(&root).is_empty(), "{case}\n{tree}");
            }
        }
        // Also below a merge of the sorted partitions, and through two
        // projections (the second one of columns).
        let make = || {
            let sorted = sort_exec(t1_scan()?, &["id"], true)?;
            let projected = project_x(sorted)?;
            let schema = projected.schema();
            let columns: Vec<(Arc<dyn PhysicalExpr>, String)> = vec![
                (col("x", &schema)?, "x".to_string()),
                (col("d", &schema)?, "d".to_string()),
                (col("id", &schema)?, "id".to_string()),
            ];
            let reordered: Arc<dyn ExecutionPlan> =
                Arc::new(ProjectionExec::try_new(columns, projected)?);
            let ordering = LexOrdering::new([PhysicalSortExpr::new_default(col(
                "id",
                &reordered.schema(),
            )?)])
            .expect("an ordering");
            let merged = Arc::new(SortPreservingMergeExec::new(ordering, reordered));
            sum_of_x_plan(merged, false, Some(InputOrderMode::Sorted))
        };
        let root = assert_lowered_plan_byte_eq(make, lower, task_ctx(), true).await;
        let tree = format_operator_tree(&root);
        assert_eq!(count::<PhysicalWindow>(&root), 1, "{tree}");
        assert!(bridged_plans(&root).is_empty(), "{tree}");
        // And in SQL, over an aggregation whose projection computes a value.
        assert_windows(
            "SELECT d, m, RANK() OVER (ORDER BY m DESC, d) AS r FROM (SELECT d, COUNT(*) + 1 AS m FROM t1 GROUP BY d)",
        )
        .await;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn errors_of_the_input_are_raised() -> Result<()> {
        use crate::operators::union::test_support::{
            AfterBatch, INJECTED_FAILURE, assert_lowered_plan_fails_like_datafusion,
            batch_plan,
        };
        let make = || {
            let batch = test_table_t1()?[0][0].clone();
            let input = batch_plan(batch, AfterBatch::Fail)?;
            row_number_plan(
                sort_exec(input, &["d", "id"], false)?,
                Some(datafusion_physical_plan::InputOrderMode::Sorted),
            )
        };
        let root = assert_lowered_plan_fails_like_datafusion(
            make,
            lower,
            test_session_config(),
            INJECTED_FAILURE,
        )
        .await;
        assert_eq!(count::<PhysicalWindow>(&root), 1);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn dictionary_partition_keys() {
        // Keys of 32 bits (narrower ones are left to DataFusion), with
        // different dictionaries in the partitions of the table.
        let setup = async |ctx: &SessionContext| {
            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int64, false),
                Field::new(
                    "s",
                    DataType::Dictionary(
                        Box::new(DataType::Int32),
                        Box::new(DataType::Utf8),
                    ),
                    true,
                ),
            ]));
            let partitions = (0..3)
                .map(|p| {
                    let ids = Int64Array::from_iter_values((0..500).map(|i| p * 500 + i));
                    let values = (0..500)
                        .map(|i| {
                            (i % 11 != 3).then(|| format!("value {}", (p * 7 + i) % 40))
                        })
                        .collect::<Vec<_>>();
                    let strings = values
                        .iter()
                        .map(Option::as_deref)
                        .collect::<arrow::array::DictionaryArray<
                        arrow::datatypes::Int32Type,
                    >>();
                    let batch = RecordBatch::try_new(
                        Arc::clone(&schema),
                        vec![Arc::new(ids), Arc::new(strings)],
                    )?;
                    Ok(vec![batch])
                })
                .collect::<Result<Vec<_>>>()?;
            ctx.register_table("dict", Arc::new(MemTable::try_new(schema, partitions)?))?;
            Ok(())
        };
        for sql in [
            "SELECT id, s, ROW_NUMBER() OVER (PARTITION BY s ORDER BY id) AS rn FROM dict",
            "SELECT id, s, FIRST_VALUE(s) OVER (ORDER BY id ROWS BETWEEN 3 PRECEDING AND CURRENT ROW) AS f FROM dict",
        ] {
            let root =
                assert_native_parity_with_setup(sql, test_session_config(), setup).await;
            assert_eq!(count::<PhysicalWindow>(&root), 1, "query: {sql}");
        }
    }

    // ---------------------------------------------------------------------
    // Streaming windows, memory limits and independent partitions
    // ---------------------------------------------------------------------

    /// A partition of a `StreamingTableExec` that produces a batch, and then
    /// what `tail` says.
    #[derive(Debug)]
    struct OneBatch {
        batch: RecordBatch,
        tail: Tail,
    }

    /// What a [`OneBatch`] does after its batch.
    #[derive(Debug, Clone, Copy)]
    enum Tail {
        /// It never produces anything again.
        Pending,
        /// It fails at once.
        Fail,
        /// It fails after a while (an unbounded stream that breaks).
        FailLater,
    }

    /// The error of a [`OneBatch`] that fails.
    const STREAM_FAILURE: &str = "the stream fails after its batch";

    impl datafusion_physical_plan::streaming::PartitionStream for OneBatch {
        fn schema(&self) -> &SchemaRef {
            self.batch.schema_ref()
        }

        fn execute(
            &self,
            _ctx: Arc<TaskContext>,
        ) -> datafusion_execution::SendableRecordBatchStream {
            use futures::StreamExt;
            let batch = futures::stream::iter([Ok(self.batch.clone())]);
            let tail = match self.tail {
                Tail::Pending => futures::stream::pending().boxed(),
                Tail::Fail => futures::stream::iter([datafusion_common::exec_err!(
                    "{STREAM_FAILURE}"
                )])
                .boxed(),
                Tail::FailLater => futures::stream::once(async {
                    tokio::time::sleep(std::time::Duration::from_secs(2)).await;
                    datafusion_common::exec_err!("{STREAM_FAILURE}")
                })
                .boxed(),
            };
            Box::pin(
                datafusion_physical_plan::stream::RecordBatchStreamAdapter::new(
                    self.batch.schema(),
                    batch.chain(tail),
                ),
            )
        }
    }

    /// A stream of `partitions` partitions of the values `0..100` of
    /// [`values_schema`] (declared sorted by `v` if `sorted`), each of which
    /// then fails (if `fail`) or never ends.
    fn one_batch_stream(
        partitions: usize,
        fail: bool,
        sorted: bool,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let tail = if fail { Tail::Fail } else { Tail::Pending };
        stream_of(0..100, partitions, tail, sorted)
    }

    /// A stream of `partitions` partitions of the `values` of
    /// [`values_schema`] (declared sorted by `v` if `sorted`), each of which
    /// then does what `tail` says.
    fn stream_of(
        values: Range<i64>,
        partitions: usize,
        tail: Tail,
        sorted: bool,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let schema = values_schema();
        let batch = values_chunk(values).into_record_batch()?;
        let partition: Arc<dyn datafusion_physical_plan::streaming::PartitionStream> =
            Arc::new(OneBatch { batch, tail });
        let ordering = if sorted {
            LexOrdering::new([PhysicalSortExpr::new_default(col("v", &schema)?)])
        } else {
            None
        };
        Ok(Arc::new(
            datafusion_physical_plan::streaming::StreamingTableExec::try_new(
                schema,
                vec![partition; partitions],
                None,
                ordering,
                // (A stream that fails at once is bounded; `RepartitionExec`
                // emits the rows of unbounded inputs without waiting for a
                // full batch.)
                !matches!(tail, Tail::Fail),
                None,
            )?,
        ))
    }

    /// `ROW_NUMBER()` over the rows of `input` (of [`values_schema`]), by
    /// the window frame of `ORDER BY v` if `order_by`, partitioned by `v % 2`
    /// if `partitioned`, in a `BoundedWindowAggExec` of `mode`.
    fn bounded_row_numbers(
        input: Arc<dyn ExecutionPlan>,
        partitioned: bool,
        order_by: bool,
        mode: datafusion_physical_plan::InputOrderMode,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let schema = input.schema();
        let v = col("v", &schema)?;
        let partition_by = if partitioned {
            vec![
                Arc::new(datafusion_physical_expr::expressions::BinaryExpr::new(
                    Arc::clone(&v),
                    datafusion_expr::Operator::Modulo,
                    datafusion_physical_expr::expressions::lit(2_i64),
                )) as Arc<dyn PhysicalExpr>,
            ]
        } else {
            vec![]
        };
        let order_by = if order_by {
            vec![PhysicalSortExpr::new_default(Arc::clone(&v))]
        } else {
            vec![]
        };
        let window_expr = datafusion_physical_plan::windows::create_window_expr(
            &datafusion_expr::WindowFunctionDefinition::WindowUDF(
                datafusion::functions_window::row_number::row_number_udwf(),
            ),
            "rn".to_string(),
            &[],
            &partition_by,
            &order_by,
            Arc::new(datafusion_expr::WindowFrame::new(
                (!order_by.is_empty()).then_some(true),
            )),
            Arc::clone(&schema),
            false,
            false,
            None,
        )?;
        Ok(Arc::new(BoundedWindowAggExec::try_new(
            vec![window_expr],
            input,
            mode,
            partitioned,
        )?))
    }

    /// The first batch of `plan` (executed with `task_ctx`), or `None` if it
    /// does not come within `seconds`.
    async fn first_batch_within(
        plan: Arc<dyn ExecutionPlan>,
        task_ctx: Arc<TaskContext>,
        seconds: f64,
    ) -> Option<RecordBatch> {
        use futures::StreamExt;
        let mut stream = datafusion_physical_plan::execute_stream(plan, task_ctx).ok()?;
        let timeout = std::time::Duration::from_secs_f64(seconds);
        tokio::time::timeout(timeout, stream.next())
            .await
            .ok()
            .flatten()
            .and_then(|batch| batch.ok())
    }

    /// The first batch of `plan`, which should come at once.
    async fn first_batch(
        plan: Arc<dyn ExecutionPlan>,
        task_ctx: Arc<TaskContext>,
    ) -> Option<RecordBatch> {
        first_batch_within(plan, task_ctx, 5.0).await
    }

    /// Whether `plan` emits nothing for a while.
    async fn emits_nothing(
        plan: Arc<dyn ExecutionPlan>,
        task_ctx: Arc<TaskContext>,
    ) -> bool {
        first_batch_within(plan, task_ctx, 0.5).await.is_none()
    }

    /// A `UnionExec` of the sorted rows `0..100` of [`values_schema`] and the
    /// sorted rows of `input`.
    fn sorted_union(input: Arc<dyn ExecutionPlan>) -> Result<Arc<dyn ExecutionPlan>> {
        let batch = values_chunk(0..100).into_record_batch()?;
        let scan =
            MemorySourceConfig::try_new_exec(&[vec![batch]], values_schema(), None)?;
        UnionExec::try_new(vec![
            sort_exec(scan, &["v"], false)?,
            sort_exec(input, &["v"], false)?,
        ])
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn windows_over_streams_that_never_end_are_left_to_datafusion() -> Result<()> {
        use crate::bridge::PipelineRootExec;
        use datafusion_physical_plan::InputOrderMode;
        use datafusion_physical_plan::limit::GlobalLimitExec;
        let task_ctx = || {
            Arc::new(TaskContext::default().with_session_config(test_session_config()))
        };
        // Without a sort that reads all of its input below it, a
        // `BoundedWindowAggExec` emits rows as it reads its input: over a
        // repartitioning or a merge of partitions in no particular order, and
        // over a sort of an input that is sorted already, which DataFusion
        // passes through. Over inputs that never end, the native window, which
        // reads all of its input first, would emit nothing: such windows are
        // left to DataFusion. Over finite inputs, they are native.
        let plans = |fail: bool| -> Result<Vec<Arc<dyn ExecutionPlan>>> {
            let repartitioned = Arc::new(RepartitionExec::try_new(
                one_batch_stream(2, fail, false)?,
                Partitioning::Hash(
                    vec![Arc::new(
                        datafusion_physical_expr::expressions::BinaryExpr::new(
                            col("v", &values_schema())?,
                            datafusion_expr::Operator::Modulo,
                            datafusion_physical_expr::expressions::lit(2_i64),
                        ),
                    )],
                    2,
                ),
            )?);
            let merged = Arc::new(CoalescePartitionsExec::new(one_batch_stream(
                2, fail, false,
            )?));
            let sorted = sort_exec(one_batch_stream(1, fail, true)?, &["v"], false)?;
            let mut plans = vec![
                bounded_row_numbers(repartitioned, true, false, InputOrderMode::Linear)?,
                bounded_row_numbers(merged, false, false, InputOrderMode::Sorted)?,
                bounded_row_numbers(sorted, false, true, InputOrderMode::Sorted)?,
            ];
            if !fail {
                // Sorts read all rows of their input before they emit
                // anything, but a merge of their partitions in no particular
                // order emits the rows of every sort as they come.
                plans.push(bounded_row_numbers(
                    Arc::new(CoalescePartitionsExec::new(sorted_union(
                        one_batch_stream(1, false, false)?,
                    )?)),
                    false,
                    false,
                    InputOrderMode::Sorted,
                )?);
            }
            Ok(plans)
        };
        // (Every engine executes plans of its own: a repartitioning can be
        // executed once.)
        let pairs =
            |fail: bool| -> Result<_> { Ok(plans(fail)?.into_iter().zip(plans(fail)?)) };
        for (index, (expected_plan, plan)) in pairs(false)?.enumerate() {
            // An input that never ends: DataFusion emits the rows of the first
            // batch.
            let expected = first_batch(expected_plan, task_ctx()).await;
            assert!(
                expected.is_some_and(|batch| batch.num_rows() > 0),
                "plan {index}"
            );
            assert!(PhysicalWindow::try_lower(&plan, lower)?.is_none());
            let root = lower(&plan)?;
            assert_eq!(count::<PhysicalWindow>(&root), 0, "plan {index}");
            let exec = Arc::new(PipelineRootExec::try_new(root, plan)?);
            let actual = first_batch(exec, task_ctx()).await;
            assert!(
                actual.is_some_and(|batch| batch.num_rows() > 0),
                "plan {index}"
            );
        }
        let limited = |plan: Arc<dyn ExecutionPlan>| -> Arc<dyn ExecutionPlan> {
            let merged = Arc::new(CoalescePartitionsExec::new(plan));
            Arc::new(GlobalLimitExec::new(merged, 0, Some(1)))
        };
        for (index, (expected_plan, plan)) in pairs(true)?.enumerate() {
            // A finite input that fails after its first batch: under a limit,
            // DataFusion returns a row of the first batch (or fails, over a
            // repartitioning, as `InputOrderMode::Linear` waits for more rows).
            // The native window reads all of its input, and fails: whether an
            // error is raised for rows beyond a limit is the engine's own.
            let rows = |batches: Vec<RecordBatch>| {
                batches.iter().map(RecordBatch::num_rows).sum::<usize>()
            };
            let expected =
                datafusion_physical_plan::collect(limited(expected_plan), task_ctx())
                    .await
                    .map(rows);
            match &expected {
                Ok(rows) => assert_eq!(*rows, 1, "plan {index}"),
                Err(error) => {
                    assert_eq!(index, 0);
                    assert!(error.to_string().contains(STREAM_FAILURE));
                }
            }
            let plan = limited(plan);
            let root = lower(&plan)?;
            assert_eq!(count::<PhysicalWindow>(&root), 1, "plan {index}");
            let exec = Arc::new(PipelineRootExec::try_new(root, plan)?);
            let error = datafusion_physical_plan::collect(exec, task_ctx())
                .await
                .expect_err("the native window reads all of its input");
            assert!(error.to_string().contains(STREAM_FAILURE), "plan {index}");
        }
        // A merge of sorted partitions that preserves their order waits for
        // the first rows of all of them: DataFusion's window emits nothing
        // while an input of the sorts does not end (an input that never ends
        // is left to DataFusion).
        let merged = |input| -> Result<Arc<dyn ExecutionPlan>> {
            let union = sorted_union(input)?;
            let ordering = union.output_ordering().cloned().unwrap();
            bounded_row_numbers(
                Arc::new(SortPreservingMergeExec::new(ordering, union)),
                false,
                true,
                InputOrderMode::Sorted,
            )
        };
        let expected_plan = merged(one_batch_stream(1, false, false)?)?;
        assert!(emits_nothing(expected_plan, task_ctx()).await);
        let plan = merged(one_batch_stream(1, false, false)?)?;
        let root = lower(&plan)?;
        assert_eq!(count::<PhysicalWindow>(&root), 0);
        let exec = Arc::new(PipelineRootExec::try_new(root, plan)?);
        assert!(emits_nothing(exec, task_ctx()).await);
        // ... and over finite inputs, it returns the rows of DataFusion's.
        let finite = || {
            let batch = values_chunk(100..150).into_record_batch()?;
            merged(MemorySourceConfig::try_new_exec(
                &[vec![batch]],
                values_schema(),
                None,
            )?)
        };
        let root = assert_lowered_plan_byte_eq(finite, lower, task_ctx(), false).await;
        assert_eq!(count::<PhysicalWindow>(&root), 1);
        // A `WindowAggExec` reads all of its input before it emits anything:
        // over a merge of partitions, it is native.
        assert_windows("SELECT id, COUNT(*) OVER () AS n, SUM(a) OVER () AS s FROM t1")
            .await;
        Ok(())
    }

    /// `ROW_NUMBER() OVER (PARTITION BY v ORDER BY v)` over the rows of
    /// `input` (of [`values_schema`], partitioned by `v`), in a
    /// `BoundedWindowAggExec` (if `bounded`) or a `WindowAggExec`.
    fn row_numbers_by_v(
        input: Arc<dyn ExecutionPlan>,
        bounded: bool,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let schema = input.schema();
        let v = col("v", &schema)?;
        let window_expr = datafusion_physical_plan::windows::create_window_expr(
            &datafusion_expr::WindowFunctionDefinition::WindowUDF(
                datafusion::functions_window::row_number::row_number_udwf(),
            ),
            "rn".to_string(),
            &[],
            std::slice::from_ref(&v),
            &[PhysicalSortExpr::new_default(Arc::clone(&v))],
            Arc::new(datafusion_expr::WindowFrame::new(Some(true))),
            Arc::clone(&schema),
            false,
            false,
            None,
        )?;
        Ok(if bounded {
            Arc::new(BoundedWindowAggExec::try_new(
                vec![window_expr],
                input,
                datafusion_physical_plan::InputOrderMode::Sorted,
                true,
            )?)
        } else {
            Arc::new(WindowAggExec::try_new(vec![window_expr], input, true)?)
        })
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn windows_over_partitions_that_end_early_are_native_over_finite_inputs()
    -> Result<()> {
        use crate::bridge::PipelineRootExec;
        use datafusion_physical_plan::limit::GlobalLimitExec;
        let task_ctx = || {
            Arc::new(TaskContext::default().with_session_config(test_session_config()))
        };
        // A window over the sorted partitions of a filter with a fetch, of a
        // repartitioning by `v`: the filter ends a partition once it emitted
        // its row, whether the other partition ended or not. The only batch
        // of the input has rows with `v = 1` only, in one partition, so that
        // DataFusion's window emits that row while the other partition waits
        // for the input, which never ends or fails later: such inputs are left
        // to DataFusion. Over a finite input, the window is native, with or
        // without the fetch.
        let window = |input: Arc<dyn ExecutionPlan>, fetch, bounded| {
            let schema = input.schema();
            let v = col("v", &schema)?;
            let repartitioned = Arc::new(RepartitionExec::try_new(
                input,
                Partitioning::Hash(vec![v], 2),
            )?);
            let filtered = FilterExec::try_new(
                datafusion_physical_expr::expressions::lit(true),
                repartitioned,
            )?
            .with_fetch(fetch)
            .expect("a filter with a fetch");
            row_numbers_by_v(sort_exec(filtered, &["v"], true)?, bounded)
        };
        let limited = |plan: Arc<dyn ExecutionPlan>| -> Arc<dyn ExecutionPlan> {
            let merged = Arc::new(CoalescePartitionsExec::new(plan));
            Arc::new(GlobalLimitExec::new(merged, 0, Some(1)))
        };
        // The rows of `plan`, within a few seconds.
        let rows = async |plan: Arc<dyn ExecutionPlan>| {
            let collect = datafusion_physical_plan::collect(plan, task_ctx());
            let timeout = std::time::Duration::from_secs(10);
            let batches = tokio::time::timeout(timeout, collect).await;
            let batches = batches.expect("no hang").expect("no error");
            concat_batches(&batches[0].schema(), &batches).unwrap()
        };
        for bounded in [true, false] {
            for tail in [Tail::Pending, Tail::FailLater] {
                let make = || {
                    let input = stream_of(1..2, 1, tail, false)?;
                    Ok::<_, DataFusionError>(limited(window(input, Some(1), bounded)?))
                };
                // DataFusion returns the row of `v = 1`.
                let expected = rows(make()?).await;
                assert_eq!(expected.num_rows(), 1, "{tail:?}");
                let plan = make()?;
                let root = lower(&plan)?;
                assert_eq!(count::<PhysicalWindow>(&root), 0, "{tail:?}");
                let exec = Arc::new(PipelineRootExec::try_new(root, plan)?);
                assert_eq!(rows(exec).await, expected, "{tail:?}");
            }
            // Over a finite input, the window is native, and returns
            // DataFusion's rows.
            for fetch in [None, Some(1000)] {
                let finite = || {
                    let batch =
                        values_chunk((0..300).map(|v| v % 50)).into_record_batch()?;
                    let scan = MemorySourceConfig::try_new_exec(
                        &[vec![batch]],
                        values_schema(),
                        None,
                    )?;
                    window(scan, fetch, bounded)
                };
                let root =
                    assert_lowered_plan_byte_eq(finite, lower, task_ctx(), false).await;
                assert_eq!(count::<PhysicalWindow>(&root), 1, "fetch {fetch:?}");
            }
        }
        Ok(())
    }

    #[test]
    fn the_buckets_keep_copies_of_the_values_of_dictionaries_that_are_slices()
    -> Result<()> {
        use arrow::array::DictionaryArray;
        // `ROW_NUMBER() OVER (PARTITION BY d ORDER BY d)` on 4 threads, of a
        // chunk of 64 rows of a dictionary whose values are a slice of an array
        // of 1 MiB: the rows of the chunk are distributed over several buckets,
        // which keep slices of the chunk reordered by bucket, and `take` shares
        // the values of a dictionary with the reordered chunk. The buckets keep
        // copies of the values only, not the array that they are a slice of.
        let data_type =
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Int64));
        let schema = Arc::new(Schema::new(vec![Field::new("d", data_type, false)]));
        let d = col("d", &schema)?;
        let order_by = LexOrdering::new([PhysicalSortExpr::new_default(Arc::clone(&d))])
            .expect("an ordering");
        let window_expr = datafusion_physical_plan::windows::create_window_expr(
            &datafusion_expr::WindowFunctionDefinition::WindowUDF(
                datafusion::functions_window::row_number::row_number_udwf(),
            ),
            "rn".to_string(),
            &[],
            std::slice::from_ref(&d),
            order_by.as_ref(),
            Arc::new(datafusion_expr::WindowFrame::new(Some(true))),
            Arc::clone(&schema),
            false,
            false,
            None,
        )?;
        let input = Arc::new(TestSource::with_schema(Arc::clone(&schema), vec![]));
        let ordering = LexOrdering::new([
            PhysicalSortExpr::new_default(Arc::clone(&d)),
            PhysicalSortExpr::new_default(d),
        ]);
        let window = PhysicalWindow::try_new(input, vec![window_expr], ordering, false)?;
        let ctx = ExecutionContext::new(Arc::new(TaskContext::default()), 0, 4);
        let global = window.get_global_sink_state(&ctx)?;
        let mut local = window.get_local_sink_state(&ctx, global.as_ref())?;
        let values = Int64Array::from_iter_values(0..1 << 17);
        let keys = Int32Array::from_iter_values(0..64);
        let column = DictionaryArray::try_new(keys, Arc::new(values.slice(0, 64)))?;
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(column)])?;
        window.sink(
            &ctx,
            &DataChunk::from_record_batch(batch),
            global.as_ref(),
            local.as_mut(),
        )?;
        let buckets = &local
            .downcast_ref::<WindowLocalState>()
            .expect("the local state of a window")
            .buckets;
        let slices = buckets
            .iter()
            .flat_map(|bucket| &bucket.batches)
            .collect::<Vec<_>>();
        assert!(slices.len() > 1, "the rows of a single bucket");
        for slice in slices {
            let values = slice
                .column(0)
                .as_dictionary::<arrow::datatypes::Int32Type>()
                .values();
            assert!(
                values.get_array_memory_size() < 4096,
                "{}",
                values.get_array_memory_size()
            );
        }
        Ok(())
    }

    #[test]
    fn the_chunks_of_several_buckets_are_reserved_until_their_slices_are_sorted()
    -> Result<()> {
        use datafusion_execution::memory_pool::{GreedyMemoryPool, MemoryPool};
        use datafusion_execution::runtime_env::RuntimeEnvBuilder;
        // A window partitioned by `v % 7`, on 4 threads, distributes the rows
        // of every chunk over several buckets: every bucket keeps a slice of
        // the chunk reordered by bucket, which keeps all of the chunk. The
        // reordered chunk is reserved once, until the slices of all buckets
        // are sorted (and their runs are reserved instead).
        let window = row_numbers(2048, 2048, true)?;
        let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(64 << 20));
        let runtime = RuntimeEnvBuilder::new()
            .with_memory_pool(Arc::clone(&pool))
            .build_arc()?;
        let task_ctx = Arc::new(TaskContext::default().with_runtime(runtime));
        let ctx = ExecutionContext::new(task_ctx, 0, 4);
        let global = window.get_global_sink_state(&ctx)?;
        let mut local = window.get_local_sink_state(&ctx, global.as_ref())?;
        let chunk = values_chunk(0..2048);
        window.sink(&ctx, &chunk, global.as_ref(), local.as_mut())?;
        // The rows of the chunk, and the metadata of every slice.
        let slices = local
            .downcast_ref::<WindowLocalState>()
            .expect("the local state of a window")
            .buckets
            .iter()
            .flat_map(|bucket| &bucket.batches)
            .map(metadata_memory)
            .sum::<usize>();
        let reordered = memory_size(&chunk.compact()?) + slices;
        assert_eq!(pool.reserved(), reordered);
        let state = global
            .downcast_ref::<WindowState>()
            .expect("the state of a window");
        let buckets = &mut local
            .downcast_mut::<WindowLocalState>()
            .expect("the local state of a window")
            .buckets;
        let mut slices = buckets
            .iter_mut()
            .filter(|bucket| !bucket.batches.is_empty());
        let first = slices.next().expect("a bucket with rows");
        assert!(slices.count() > 0, "the rows of a single bucket");
        // A bucket that sorts its rows (e.g. once it holds many of them)
        // releases its slice, which the slices of the other buckets keep: the
        // chunk stays reserved, with the run of the bucket.
        window.sort_buffered_rows(first, &state.reservation)?;
        let run = first.runs.iter().map(SortedRun::memory_size).sum::<usize>();
        assert_eq!(pool.reserved(), reordered + run);
        window.combine(&ctx, global.as_ref(), local.as_mut())?;
        let runs = state
            .buckets
            .iter()
            .map(|bucket| bucket.lock().memory)
            .sum::<usize>();
        assert!(runs > 0);
        assert_eq!(pool.reserved(), runs);
        drop(local);
        drop(global);
        assert_eq!(pool.reserved(), 0);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn windows_reserve_their_rows_in_the_memory_pool() -> Result<()> {
        use crate::bridge::PipelineRootExec;
        use crate::operators::lowering_tests::assert_exhausted;
        use crate::sort::test_support::assert_memory_freed;
        use datafusion_execution::memory_pool::{GreedyMemoryPool, MemoryPool};
        use datafusion_execution::runtime_env::RuntimeEnvBuilder;
        // The window is native whatever the memory pool of the session: it
        // reserves the rows it buffers, sorts and evaluates, and fails with
        // `ResourcesExhausted` once the pool cannot hold them, ...
        let task_ctx = |pool: &Arc<dyn MemoryPool>| -> Result<Arc<TaskContext>> {
            let runtime = RuntimeEnvBuilder::new()
                .with_memory_pool(Arc::clone(pool))
                .build_arc()?;
            Ok(Arc::new(
                TaskContext::default()
                    .with_session_config(test_session_config())
                    .with_runtime(runtime),
            ))
        };
        // A window of several partitions (in several buckets), and one of a
        // single partition whose rows are sorted again by the order of its
        // input once they are evaluated.
        for single in [false, true] {
            let make = || {
                use datafusion_physical_plan::InputOrderMode;
                if single {
                    let coalesced = Arc::new(CoalescePartitionsExec::new(t1_scan()?));
                    let sorted = sort_exec(coalesced, &["id"], false)?;
                    row_number_plan(sorted, Some(InputOrderMode::Linear))
                } else {
                    let sorted =
                        sort_exec(hash_repartition(t1_scan()?)?, &["d", "id"], true)?;
                    row_number_plan(sorted, Some(InputOrderMode::Sorted))
                }
            };
            let plan = make()?;
            let root = lower(&plan)?;
            assert_eq!(count::<PhysicalWindow>(&root), 1, "single: {single}");
            // 64 KiB do not hold the 5000 rows of `t1`.
            let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(64 * 1024));
            let exec = Arc::new(PipelineRootExec::try_new(root, plan)?);
            let result = datafusion_physical_plan::collect(exec, task_ctx(&pool)?).await;
            let error = result.as_ref().map(|_| ()).unwrap_err().to_string();
            assert!(
                error.contains("PhysicalWindow"),
                "single: {single}: {error}"
            );
            assert_exhausted(result);
            assert_memory_freed(pool.as_ref()).await;
            // ... and returns its rows when it can hold them, and frees their
            // memory once they are emitted. (DataFusion's sorts reserve 10 MB
            // to merge every partition.)
            let pool: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(256 << 20));
            let root =
                assert_lowered_plan_byte_eq(make, lower, task_ctx(&pool)?, single).await;
            assert_eq!(count::<PhysicalWindow>(&root), 1, "single: {single}");
            assert_eq!(pool.reserved(), 0, "single: {single}");
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn windows_of_independent_partitions_are_native() {
        // A `Partitioned` hash join emits its rows partitioned by its keys, so
        // the window by these keys sorts every partition of the join without
        // repartitioning it: in DataFusion, every window partition reads the
        // rows of its partition of the join only, and may emit them while
        // another one waits for its input. The native window reads all of its
        // (finite) input first, and returns the same rows.
        let mut config = test_session_config();
        let optimizer = &mut config.options_mut().optimizer;
        optimizer.hash_join_single_partition_threshold = 0;
        optimizer.hash_join_single_partition_threshold_rows = 0;
        assert_windows_with_config(
            "SELECT t1.id, ROW_NUMBER() OVER (PARTITION BY t1.a ORDER BY t1.id, t2.name, t2.w) AS rn FROM t1 JOIN t2 ON t1.a = t2.k",
            config,
            true,
        )
        .await;
        // Over a repartitioning (through a projection), the window is native.
        assert_windows(
            "SELECT id, ROW_NUMBER() OVER (PARTITION BY x ORDER BY id) AS rn FROM (SELECT id, a % 7 AS x FROM t1)",
        )
        .await;
    }
}
