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
//! # Order
//!
//! The rows of a window partition are evaluated in the order in which
//! DataFusion's window node reads them: the ordering of its input, which a
//! `SortExec` below it established, with the partition keys first if the
//! ordering does not start with them. Rows that this ordering does not order
//! (ties) come in any order, as from DataFusion's sort.
//!
//! A window whose DataFusion node has a single output partition, i.e. whose
//! consumers may rely on the order of its output, uses a single bucket and
//! emits its rows in their sort order, on a single thread. Its input is still
//! sorted by all threads into runs, which a single thread merges. A window
//! of several output partitions emits the buckets in any order, in
//! parallel, and the rows of each window partition in their order.
//!
//! The window operator keeps all rows in memory: it does not spill, and
//! does not reserve memory in the memory pool.

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
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::window::{
    PlainAggregateWindowExpr, SlidingAggregateWindowExpr, StandardWindowExpr, WindowExpr,
};
use datafusion_physical_expr::{LexOrdering, PhysicalExpr, PhysicalSortExpr};
use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::repartition::RepartitionExec;
use datafusion_physical_plan::sorts::sort::SortExec;
use datafusion_physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion_physical_plan::windows::{BoundedWindowAggExec, WindowAggExec};
use datafusion_physical_plan::{ExecutionPlan, ExecutionPlanProperties, Partitioning};
use parking_lot::Mutex;

use crate::chunk::{DEFAULT_VECTOR_SIZE, DataChunk};
use crate::join::hash_table::{hash_join_keys, radix_bits_for_threads};
use crate::operator::{
    ExecutionContext, GlobalSinkState, GlobalSourceState, LocalSinkState,
    LocalSourceState, PhysicalOperator, SinkFinalize, SinkResult, SourceResult,
    TaskInterrupt, downcast_state, downcast_state_mut,
};
use crate::operators::adaptive_filter::is_infallible;
use crate::pipeline::{MetaPipelineBuilder, PipelineBuilder};
use crate::sort::merge::{RunMerger, SortedRun};
use crate::sort::sort_sink::{SORT_RUN_BYTES, SORT_RUN_ROWS};
use crate::sort::{SortKeys, memory_size};

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
    /// of `keys`.
    ordered: bool,
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
            schema,
        })
    }

    /// Evaluate the window functions of the stage on `batch`, rows of the
    /// stage before (or of the input) that have the same group keys, in their
    /// order: the rows the stage reads, with their window values.
    fn evaluate(&self, batch: RecordBatch) -> Result<RecordBatch> {
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
        // if the partition keys of the first stage lead it.
        if !self.partition_keys_lead {
            return Ok(None);
        }
        let mut stage = WindowStage::try_new(projection, input_schema, window_exprs)?;
        // The partition keys of the stage, as columns of the input of the
        // window: they must be the first keys of the sort, so that the
        // partitions of the stage are runs of rows of the sort.
        let mut columns = vec![];
        for key in &stage.partition_by {
            let Some(column) = key.downcast_ref::<Column>() else {
                return Ok(None);
            };
            let column = match &stage.projection {
                Some(projection) => projection.get(column.index()).copied(),
                None => Some(column.index()),
            };
            let Some(column) = column.and_then(|column| self.input_column(column)) else {
                return Ok(None);
            };
            if !columns.contains(&column) {
                columns.push(column);
            }
        }
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
    /// order of their sort: by `ordering` if the partition keys lead it,
    /// otherwise by the partition keys and then by `ordering`. Otherwise, the
    /// window partitions are evaluated in parallel and emitted in any order.
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
    /// Returns `Ok(None)` for any other node, without calling `lower_input`,
    /// and for the windows that are left to DataFusion:
    ///
    /// * windows DataFusion does not evaluate on all rows of each window
    ///   partition: with a `WindowStateObserver`; of no partitions (DataFusion
    ///   executes nothing); of several partitions without partition keys, or
    ///   whose input is not partitioned by its partition keys (DataFusion
    ///   evaluates every partition separately); whose partitions are not
    ///   those of their input (stale properties);
    /// * windows whose input is in an order that DataFusion did not establish
    ///   with a sort (or a merge of several partitions in no particular
    ///   order) below the window: e.g. a scan of a sorted file, which
    ///   DataFusion reads in its order, also where its ordering has ties. The
    ///   native window sorts its rows by that ordering, so it could evaluate
    ///   rows with equal keys in another order. A window stacked on another
    ///   one (through projections of columns and redistributions that keep
    ///   the order) reads the rows in the order in which the window below
    ///   evaluated them: it is lowered as a stage of the window below (see
    ///   the [module documentation](self)), if its partitions are runs of
    ///   rows of the sort of that window, and left to DataFusion otherwise;
    /// * windows with an expression that can fail on some rows
    ///   (conservatively, see [`is_infallible`]): the arguments, partition
    ///   keys and order keys of the window functions, a `FILTER` of an
    ///   aggregate (always, as it cannot be inspected), and the keys of the
    ///   ordering of the rows. DataFusion evaluates them on other batches,
    ///   and `BoundedWindowAggExec` emits its rows as it goes, so that, e.g.
    ///   under a limit, one of them could fail where the other does not.
    ///   Window expressions of other types than DataFusion's (standard,
    ///   aggregate and sliding aggregate ones) are left to DataFusion too;
    /// * windows whose rows of a single partition must be in an order that
    ///   does not start with the partition keys (a `BoundedWindowAggExec` in
    ///   `Linear` or `PartiallySorted` mode with a single partition): the
    ///   native window emits its rows in the order of its sort, which starts
    ///   with the partition keys;
    /// * windows whose input or output has dictionaries with keys of 8 or 16
    ///   bits, or run-end encodings with 16-bit run ends: the native window
    ///   gathers other rows together than DataFusion (whole partitions,
    ///   buckets, sorted runs and chunks), which can overflow such keys or
    ///   run ends where DataFusion's batches do not, or the other way round;
    /// * windows whose rows are ordered by keys that the Arrow row format does
    ///   not support, or by unions (the native window sorts by the row
    ///   format, which orders unions differently from DataFusion's sort).
    ///
    /// The nodes below the window that only distribute and sort its input
    /// are replaced too (the native window partitions and sorts its input
    /// itself): `SortExec`s, `SortPreservingMergeExec`s,
    /// `CoalescePartitionsExec`s and `RepartitionExec`s (round robin or hash)
    /// without limit, whose expressions cannot fail, and that read all
    /// partitions of their input.
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
        let lowered = lower_input(rows)?;
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

    /// Sort the buffered rows of `bucket` into runs.
    fn sort_buffered_rows(&self, bucket: &mut LocalBucket) -> Result<()> {
        let Some(keys) = &self.spec.keys else {
            return Ok(());
        };
        let batches = std::mem::take(&mut bucket.batches);
        bucket.rows = 0;
        bucket.bytes = 0;
        bucket
            .runs
            .extend(keys.sort(&self.input().schema(), &batches, None)?);
        Ok(())
    }

    /// Add `batch` to the rows of `bucket`, sorting them into a run once
    /// there are many.
    fn buffer(&self, bucket: &mut LocalBucket, batch: RecordBatch) -> Result<()> {
        bucket.rows += batch.num_rows();
        bucket.bytes += memory_size(&batch);
        bucket.batches.push(batch);
        if bucket.rows >= SORT_RUN_ROWS || bucket.bytes >= SORT_RUN_BYTES {
            self.sort_buffered_rows(bucket)?;
        }
        Ok(())
    }

    /// Sort the rows of `input`, the rows of a bucket, and find its groups.
    fn sort_bucket(&self, input: BucketInput) -> Result<SortedBucket> {
        let BucketInput { mut runs, batches } = input;
        runs.retain(|run| run.num_rows() > 0);
        let batches = match &self.spec.keys {
            // The rows of a single run are sorted already.
            Some(_) if runs.len() == 1 => {
                runs.into_iter().map(SortedRun::into_batch).collect()
            }
            Some(_) => {
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
        let groups = self.groups(&batches)?;
        Ok(SortedBucket::new(batches, groups))
    }

    /// The groups of the sorted rows `batches`: the runs of rows with equal
    /// group keys, as ranges of rows of all batches.
    fn groups(&self, batches: &[RecordBatch]) -> Result<Vec<Range<usize>>> {
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
            let columns = keys
                .iter()
                .map(|values| SortColumn {
                    values: Arc::clone(values),
                    options: None,
                })
                .collect::<Vec<_>>();
            let ranges = evaluate_partition_ranges(batch.num_rows(), &columns)?;
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
    /// with their window values.
    fn evaluate_group(
        &self,
        bucket: &SortedBucket,
        range: Range<usize>,
    ) -> Result<RecordBatch> {
        let mut batch = bucket.rows(&self.input().schema(), range)?;
        for stage in &self.spec.stages {
            batch = stage.evaluate(batch)?;
        }
        Ok(batch)
    }
}

/// Analyze the window node `plan` (see [`PhysicalWindow::try_lower`]): the
/// specification of the native window, and the DataFusion plan of the rows
/// that it reads, or `None` if it is left to DataFusion.
fn analyze(
    plan: &Arc<dyn ExecutionPlan>,
) -> Result<Option<(WindowSpec, &Arc<dyn ExecutionPlan>)>> {
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
    if !window_exprs
        .iter()
        .all(|expr| window_expr_cannot_fail(expr, &input_schema))
        || plan
            .schema()
            .fields()
            .iter()
            .any(|field| has_narrow_encoding(field.data_type()))
    {
        return Ok(None);
    }
    let (rows, order) = without_distribution(input);
    let spec = if order.sorted || order.interleaved || sorted_by_a_sort(rows) {
        // The rows are sorted (or merged in no particular order) below the
        // window: the native window sorts them by the ordering of its input.
        let ordering = input.output_ordering().cloned();
        if ordering.as_ref().is_some_and(|ordering| {
            ordering.iter().any(|sort_expr| {
                !is_infallible(&sort_expr.expr, &input_schema)
                    || sort_expr
                        .expr
                        .data_type(&input_schema)
                        .map_or(true, |data_type| has_union(&data_type))
            })
        }) {
            return Ok(None);
        }
        match WindowSpec::try_new(&input_schema, window_exprs.to_vec(), ordering, ordered)
        {
            Ok(spec) => spec,
            Err(DataFusionError::NotImplemented(_)) => return Ok(None),
            Err(e) => return Err(e),
        }
    } else {
        // The rows are in the order of a window below (or of another node,
        // left to DataFusion).
        let Some(WindowBelow {
            projection,
            window: below,
        }) = window_below(input)
        else {
            return Ok(None);
        };
        let Some((below_spec, rows)) = analyze(below)? else {
            return Ok(None);
        };
        if below_spec.ordered != ordered {
            return Ok(None);
        }
        match below_spec.with_stage(projection, &input_schema, window_exprs.to_vec())? {
            Some(spec) => return Ok(Some((spec, rows))),
            None => return Ok(None),
        }
    };
    if ordered && !spec.partition_keys_lead && !spec.group_keys.is_empty() {
        return Ok(None);
    }
    Ok(Some((spec, rows)))
}

/// Whether the rows of `plan` are in an order that a sort established: it is
/// a `SortExec` (with a fetch, which the window does not replace), or a
/// `SortPreservingMergeExec` of such rows.
fn sorted_by_a_sort(plan: &Arc<dyn ExecutionPlan>) -> bool {
    match plan.downcast_ref::<SortPreservingMergeExec>() {
        Some(merge) => sorted_by_a_sort(merge.input()),
        None => plan.is::<SortExec>(),
    }
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
    let partitions =
        |plan: &Arc<dyn ExecutionPlan>| plan.output_partitioning().partition_count();
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
            && partitioning_cannot_fail(repartition)
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

/// Whether evaluating `expr` on rows with schema `schema` cannot fail: it is
/// a standard, aggregate or sliding aggregate window expression (of
/// DataFusion) without `FILTER`, whose arguments, partition keys and order
/// keys cannot fail (see [`is_infallible`]).
fn window_expr_cannot_fail(expr: &Arc<dyn WindowExpr>, schema: &Schema) -> bool {
    let any = expr.as_any();
    let known = any.is::<StandardWindowExpr>()
        || ((any.is::<PlainAggregateWindowExpr>()
            || any.is::<SlidingAggregateWindowExpr>())
            && !has_filter(expr));
    let expressions = expr.all_expressions();
    known
        && expressions
            .args
            .iter()
            .chain(&expressions.partition_by_exprs)
            .chain(&expressions.order_by_exprs)
            .all(|expr| is_infallible(expr, schema))
}

/// Whether an aggregate window expression has a `FILTER`. Its filter is not
/// public, but its `Debug` representation ends with it: an expression that
/// does not end with `filter: None` is considered to have one.
fn has_filter(expr: &Arc<dyn WindowExpr>) -> bool {
    !format!("{expr:?}").ends_with("filter: None }")
}

/// What the nodes removed from the input of a window did with the order of
/// its rows.
#[derive(Debug, Default, Clone, Copy)]
struct Reordering {
    /// A `SortExec` sorted them.
    sorted: bool,
    /// A `CoalescePartitionsExec` or `RepartitionExec` merged the rows of
    /// several partitions in no particular order.
    interleaved: bool,
}

/// `plan`, the input of a window, without the nodes that only distribute and
/// sort its rows (see [`PhysicalWindow::try_lower`]), and what these nodes
/// did with the order of the rows.
fn without_distribution(
    plan: &Arc<dyn ExecutionPlan>,
) -> (&Arc<dyn ExecutionPlan>, Reordering) {
    let partitions =
        |plan: &Arc<dyn ExecutionPlan>| plan.output_partitioning().partition_count();
    let mut plan = plan;
    let mut order = Reordering::default();
    loop {
        let next = if let Some(sort) = plan.downcast_ref::<SortExec>() {
            // A sort of all its input reads only the first partition of it.
            let reads_all = if sort.preserve_partitioning() {
                partitions(plan) == partitions(sort.input())
            } else {
                partitions(sort.input()) == 1
            };
            let schema = sort.input().schema();
            let stripped = sort.fetch().is_none()
                && reads_all
                && sort
                    .expr()
                    .iter()
                    .all(|sort_expr| is_infallible(&sort_expr.expr, &schema));
            order.sorted |= stripped;
            stripped.then(|| sort.input())
        } else if let Some(merge) = plan.downcast_ref::<SortPreservingMergeExec>() {
            let schema = merge.input().schema();
            (merge.fetch().is_none()
                && partitions(merge.input()) > 0
                && merge
                    .expr()
                    .iter()
                    .all(|sort_expr| is_infallible(&sort_expr.expr, &schema)))
            .then(|| merge.input())
        } else if let Some(coalesce) = plan.downcast_ref::<CoalescePartitionsExec>() {
            let stripped = coalesce.fetch().is_none() && partitions(coalesce.input()) > 0;
            order.interleaved |= stripped && partitions(coalesce.input()) > 1;
            stripped.then(|| coalesce.input())
        } else if let Some(repartition) = plan.downcast_ref::<RepartitionExec>() {
            let stripped = partitioning_cannot_fail(repartition)
                && partitions(repartition.input()) > 0;
            order.interleaved |= stripped
                && !repartition.preserve_order()
                && partitions(repartition.input()) > 1;
            stripped.then(|| repartition.input())
        } else {
            None
        };
        match next {
            Some(next) => plan = next,
            None => return (plan, order),
        }
    }
}

/// Whether the partitioning of `repartition` (round robin, or by the hash of
/// expressions) cannot fail.
fn partitioning_cannot_fail(repartition: &RepartitionExec) -> bool {
    let schema = repartition.input().schema();
    match repartition.partitioning() {
        Partitioning::RoundRobinBatch(_) => true,
        Partitioning::Hash(exprs, _) => {
            exprs.iter().all(|expr| is_infallible(expr, &schema))
        }
        _ => false,
    }
}

/// Whether values of `data_type` (or of its children) are dictionary encoded
/// with keys of 8 or 16 bits, or run-end encoded with run ends of 16 bits:
/// arrays of those hold only so many distinct values or rows, so that
/// whether gathering rows into one array overflows them depends on which
/// rows are gathered together.
fn has_narrow_encoding(data_type: &DataType) -> bool {
    match data_type {
        DataType::Dictionary(keys, values) => {
            matches!(
                keys.as_ref(),
                DataType::Int8 | DataType::UInt8 | DataType::Int16 | DataType::UInt16
            ) || has_narrow_encoding(values)
        }
        DataType::RunEndEncoded(run_ends, values) => {
            run_ends.data_type() == &DataType::Int16
                || has_narrow_encoding(values.data_type())
        }
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::ListView(field)
        | DataType::LargeListView(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => has_narrow_encoding(field.data_type()),
        DataType::Struct(fields) => fields
            .iter()
            .any(|field| has_narrow_encoding(field.data_type())),
        DataType::Union(fields, _) => fields
            .iter()
            .any(|(_, field)| has_narrow_encoding(field.data_type())),
        _ => false,
    }
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
}

impl SortedBucket {
    fn new(batches: Vec<RecordBatch>, groups: Vec<Range<usize>>) -> Self {
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
        }
    }

    /// The rows `range` as one batch with schema `schema`: a slice of one
    /// batch, or the concatenation of slices of several.
    fn rows(&self, schema: &SchemaRef, range: Range<usize>) -> Result<RecordBatch> {
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
            [slice] => Ok(slice.clone()),
            _ => Ok(concat_batches(schema, &slices)?),
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
                continue;
            }
            let sorted = window.sort_bucket(input)?;
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
    /// The estimated memory size of `batches`.
    bytes: usize,
    /// The sorted runs of the thread.
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
#[derive(Debug)]
struct BucketEmitter {
    bucket: SortedBucket,
    /// The next group to evaluate.
    next_group: usize,
    /// Evaluated rows that were not emitted yet, in order.
    pending: VecDeque<RecordBatch>,
    pending_rows: usize,
}

impl BucketEmitter {
    fn new(bucket: SortedBucket) -> Self {
        Self {
            bucket,
            next_group: 0,
            pending: VecDeque::new(),
            pending_rows: 0,
        }
    }

    /// The next chunk of at most [`DEFAULT_VECTOR_SIZE`] rows, `None` once all
    /// rows of the bucket were emitted.
    fn next_chunk(&mut self, window: &PhysicalWindow) -> Result<Option<DataChunk>> {
        while self.pending_rows < DEFAULT_VECTOR_SIZE
            && self.next_group < self.bucket.groups.len()
        {
            let range = self.bucket.groups[self.next_group].clone();
            self.next_group += 1;
            let rows = window.evaluate_group(&self.bucket, range)?;
            self.pending_rows += rows.num_rows();
            self.pending.push_back(rows);
        }
        if self.pending_rows == 0 {
            return Ok(None);
        }
        // Large groups are emitted as they are (in slices of at most
        // `DEFAULT_VECTOR_SIZE` rows), small ones are concatenated.
        let mut parts = vec![];
        let mut len = 0;
        while let Some(front) = self.pending.pop_front() {
            let rows = front.num_rows();
            let large = rows >= DEFAULT_VECTOR_SIZE / 2;
            if large && !parts.is_empty() {
                self.pending.push_front(front);
                break;
            }
            let take = rows.min(DEFAULT_VECTOR_SIZE - len);
            if take < rows {
                self.pending.push_front(front.slice(take, rows - take));
            }
            parts.push(if take < rows {
                front.slice(0, take)
            } else {
                front
            });
            len += take;
            if large || len == DEFAULT_VECTOR_SIZE {
                break;
            }
        }
        self.pending_rows -= len;
        let batch = match parts.as_slice() {
            [part] => part.clone(),
            _ => concat_batches(&window.spec.schema, &parts)?,
        };
        Ok(Some(DataChunk::from_record_batch(batch)))
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
                format!("[{exprs}]")
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
        Ok(Arc::new(WindowState {
            bits: num_buckets.trailing_zeros(),
            buckets: (0..num_buckets).map(|_| Mutex::default()).collect(),
            next_sort: AtomicUsize::new(0),
            sorted: (0..num_buckets).map(|_| Mutex::default()).collect(),
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
        if local.buckets.len() == 1 {
            self.buffer(&mut local.buckets[0], batch)?;
            return Ok(SinkResult::NeedMoreInput);
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
        let sorted =
            match (0..num_buckets).find(|&b| starts[b + 1] - starts[b] == num_rows) {
                // All rows are in one bucket.
                Some(_) => batch,
                None => take_record_batch(&batch, &UInt32Array::from(order))?,
            };
        for bucket in 0..num_buckets {
            let (start, end) = (starts[bucket], starts[bucket + 1]);
            if end > start {
                self.buffer(
                    &mut local.buckets[bucket],
                    sorted.slice(start, end - start),
                )?;
            }
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
            self.sort_buffered_rows(bucket)?;
            let mut global = global.lock();
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
        TestSource, count, run, values_chunk, values_schema,
    };
    use crate::test_utils::{
        datafusion_context, register_test_tables, test_session_config, test_table_t1,
    };
    use crate::window::test_support::{
        assert_native_parity_with_config, assert_native_parity_with_setup, lower,
    };

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
            "SELECT ROW_NUMBER() OVER () AS rn FROM t1",
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
            "SELECT id, AVG(g) OVER (PARTITION BY e ORDER BY id ROWS UNBOUNDED PRECEDING) AS a FROM t1",
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
    async fn windows_that_differ_from_datafusion_are_left_to_it() {
        // DataFusion reads the rows of a sorted input in its order, ties
        // included; the native window would sort them again.
        for sql in [
            "SELECT id, ROW_NUMBER() OVER (ORDER BY a) AS rn FROM sorted",
            "SELECT id, LAG(id) OVER (ORDER BY a) AS prev FROM sorted",
            "SELECT id, ROW_NUMBER() OVER () AS rn FROM sorted",
        ] {
            let root =
                assert_native_parity_with_setup(sql, test_session_config(), setup_sorted)
                    .await;
            assert_eq!(count::<PhysicalWindow>(&root), 0, "query: {sql}");
        }
        // With a sort, the window is native.
        let root = assert_native_parity_with_setup(
            "SELECT id, ROW_NUMBER() OVER (PARTITION BY d ORDER BY id) AS rn FROM sorted",
            test_session_config(),
            setup_sorted,
        )
        .await;
        assert_eq!(count::<PhysicalWindow>(&root), 1);
        // Arguments that can fail, and a `FILTER`.
        for sql in [
            "SELECT id, SUM(10 / (a + 51)) OVER (PARTITION BY d ORDER BY id) AS s FROM t1",
            "SELECT id, SUM(a) FILTER (WHERE a > 0) OVER (PARTITION BY d ORDER BY id) AS s FROM t1",
        ] {
            assert_windows_with_config(sql, test_session_config(), false).await;
        }
        // Infallible arithmetic stays native.
        assert_windows(
            "SELECT id, SUM(a * 2 + 1) OVER (PARTITION BY d ORDER BY id) AS s FROM t1",
        )
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn narrow_dictionaries_are_left_to_datafusion() {
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
        let root = assert_native_parity_with_setup(
            "SELECT id, s, ROW_NUMBER() OVER (ORDER BY id) AS rn FROM dict",
            test_session_config(),
            setup,
        )
        .await;
        assert_eq!(count::<PhysicalWindow>(&root), 0);
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
        let groups = window.groups(&batches)?;
        assert_eq!(groups, vec![0..2, 2..4, 4..6, 6..8]);
        let bucket = SortedBucket::new(batches, groups);
        let rows = window.evaluate_group(&bucket, 2..4)?;
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
        Ok(
            datafusion_datasource::memory::MemorySourceConfig::try_new_exec(
                &partitions,
                schema,
                None,
            )?,
        )
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
        // input: the native window does not reproduce it if it does not start
        // with the partition keys.
        for (mode, keys, native) in [
            (Some(InputOrderMode::Sorted), vec!["d", "id"], 1),
            (Some(InputOrderMode::Linear), vec!["id"], 0),
        ] {
            let make = || {
                let coalesced = Arc::new(CoalescePartitionsExec::new(t1_scan()?));
                row_number_plan(sort_exec(coalesced, &keys, false)?, mode.clone())
            };
            let root = assert_lowered_plan_byte_eq(make, lower, task_ctx(), true).await;
            assert_eq!(count::<PhysicalWindow>(&root), native, "{mode:?}");
        }
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
}
