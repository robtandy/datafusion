// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`PhysicalPartialSort`]: the sort of an input that is sorted by a prefix
//! of the sort keys (`PartialSortExec`).

use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::compute::{concat_batches, lexsort_to_indices, take_record_batch};
use arrow::datatypes::SchemaRef;
use arrow_ord::ord::make_comparator;
use datafusion_common::utils::evaluate_partition_ranges;
use datafusion_common::{Result, internal_err};
use datafusion_datasource::file_scan_config::{FileScanConfig, FileScanConfigBuilder};
use datafusion_datasource::memory::MemorySourceConfig;
use datafusion_datasource::source::{DataSource, DataSourceExec};
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_physical_expr::LexOrdering;
use datafusion_physical_plan::aggregates::AggregateExec;
use datafusion_physical_plan::async_func::AsyncFuncExec;
use datafusion_physical_plan::buffer::BufferExec;
use datafusion_physical_plan::coop::CooperativeExec;
use datafusion_physical_plan::filter::FilterExec;
use datafusion_physical_plan::limit::LocalLimitExec;
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::repartition::RepartitionExec;
use datafusion_physical_plan::sorts::partial_sort::PartialSortExec;
use datafusion_physical_plan::sorts::sort::SortExec;
use datafusion_physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion_physical_plan::streaming::StreamingTableExec;
use datafusion_physical_plan::union::UnionExec;
use datafusion_physical_plan::unnest::UnnestExec;
use datafusion_physical_plan::windows::{BoundedWindowAggExec, WindowAggExec};
use datafusion_physical_plan::{
    ChildrenPropertiesMode, ExecutionPlan, ExecutionPlanProperties, Partitioning,
    ReplaceChildrenOptions,
};

use crate::chunk::{DEFAULT_VECTOR_SIZE, DataChunk};
use crate::operator::{
    ExecutionContext, GlobalOperatorState, OperatorFinalizeResult, OperatorResult,
    OperatorState, PhysicalOperator, downcast_state_mut, operator_key,
};
use crate::operators::topk_of_partitions::{
    PhysicalPartialSortOfPartitions, PhysicalTopKOfPartitions,
};
use crate::operators::union::PhysicalUnion;
use crate::pipeline::PipelineGraph;
use crate::sort::{PhysicalSort, PhysicalTopK};
use crate::sources::batches::{batch_overhead, chunk_memory};

/// Sorts an input whose rows are sorted by a prefix of its sort keys (the
/// first `common_prefix_length` keys), as `PartialSortExec` does: it buffers
/// the rows of a group of rows with the same prefix until a row of another
/// prefix arrives, which completes the group (and the groups before it), and
/// then emits the rows of the completed groups, sorted by all keys. So it
/// streams, also over inputs that never end, which DataFusion plans it for.
/// With a fetch, it emits the first `fetch` rows, and then finishes.
///
/// It is not a parallel operator, so that the pipeline that contains it runs
/// on a single thread, which sees the chunks of its input in their order
/// (the order of its single partition). It is only lowered in a single
/// pipeline: an input whose rows come from several pipelines (e.g. a union)
/// is sorted by a native sort instead (see [`PhysicalPartialSort::try_lower`]).
/// Over several partitions (a `PartialSortExec` that preserves its
/// partitioning), whose rows the pipeline engine does not keep apart, it
/// sorts the runs of rows with the same prefix as they arrive: no native
/// operator depends on the order of the rows of several partitions, which
/// they partition, merge or sort themselves. A fetch of several partitions
/// limits every partition: such a partial sort is lowered as the union of a
/// partial sort of every partition.
///
/// The memory of the rows that it buffers, and of the sorted rows that it
/// did not emit yet, is reserved in the memory pool of the session, and so is
/// the memory that sorting a group allocates, before it is allocated.
#[derive(Debug)]
pub struct PhysicalPartialSort {
    /// The input, the only child.
    children: Vec<Arc<dyn PhysicalOperator>>,
    expr: LexOrdering,
    /// The number of keys by which the input is sorted.
    common_prefix_length: usize,
    fetch: Option<usize>,
}

impl PhysicalPartialSort {
    /// Sort `input`, sorted by the first `common_prefix_length` keys of
    /// `expr`, by `expr`, keeping the first `fetch` rows (if any).
    pub fn try_new(
        input: Arc<dyn PhysicalOperator>,
        expr: LexOrdering,
        common_prefix_length: usize,
        fetch: Option<usize>,
    ) -> Result<Self> {
        if common_prefix_length == 0 || common_prefix_length > expr.len() {
            return internal_err!(
                "PhysicalPartialSort sorts by {} keys, its input by {common_prefix_length}",
                expr.len()
            );
        }
        Ok(Self {
            children: vec![input],
            expr,
            common_prefix_length,
            fetch,
        })
    }

    /// Lower a `PartialSortExec` to a [`PhysicalPartialSort`], whose input
    /// is `lower_input` of the input of the node. Returns `Ok(None)` for any
    /// other node, without calling `lower_input`.
    ///
    /// A `PartialSortExec` with a fetch of several partitions (that preserves
    /// its partitioning) keeps the first `fetch` rows of every partition: it
    /// is lowered as the [`PhysicalUnion`] of the partial sorts of the plans
    /// of its partitions (see [`partitions_of`]), every one with the fetch.
    /// The partitions of a hash or range `RepartitionExec`, which only an
    /// execution of its input distributes, are kept by a
    /// [`PhysicalTopKOfPartitions`] of its input instead. The partitions of
    /// other inputs that the planner cannot split (e.g. of a round robin
    /// repartitioning, which depend on the batches in which DataFusion reads
    /// its rows) are those of the pipeline engine: a single one, which keeps
    /// the first `fetch` rows of all of them.
    ///
    /// The rows of an input that the pipeline engine runs in several
    /// pipelines (e.g. of a union below a `CoalescePartitionsExec`) would
    /// reach the partial sort in several states, which would sort and limit
    /// them apart. Their partial sort is lowered as the native sort of all of
    /// them by its keys, with its fetch (a [`PhysicalTopK`] or a
    /// [`PhysicalSort`]): DataFusion sorts the rows of such inputs, whose
    /// partitions interleave, by their prefix only if they all have the same
    /// prefix, so their partial sort is their sort.
    pub fn try_lower<F>(
        plan: &Arc<dyn ExecutionPlan>,
        mut lower_input: F,
    ) -> Result<Option<Arc<dyn PhysicalOperator>>>
    where
        F: FnMut(&Arc<dyn ExecutionPlan>) -> Result<Arc<dyn PhysicalOperator>>,
    {
        let Some(sort) = plan.downcast_ref::<PartialSortExec>() else {
            return Ok(None);
        };
        // The partitions of a hash or range repartitioning are the rows that
        // an execution of its input routes to them: every one keeps its
        // first rows, by a TopK of its own, of rows routed once.
        if limits_every_partition(sort)
            && let Some(repartition) = sort.input().downcast_ref::<RepartitionExec>()
            && matches!(
                repartition.partitioning(),
                Partitioning::Hash(..) | Partitioning::Range(_)
            )
            && let Some(fetch) = sort.fetch().filter(|&fetch| fetch > 0)
        {
            let input = lower_input(repartition.input())?;
            let partitioning = repartition.partitioning().clone();
            // The rows that the repartitioning of a single partition routes
            // to a partition keep their order, sorted by the prefix: every
            // partition is sorted as they arrive, in a single pipeline.
            if repartition.input().output_partitioning().partition_count() <= 1 {
                let sort: Arc<dyn PhysicalOperator> =
                    Arc::new(PhysicalPartialSortOfPartitions::try_new(
                        Arc::clone(&input),
                        partitioning.clone(),
                        sort.expr().clone(),
                        sort.common_prefix_length(),
                        fetch,
                    )?);
                if pipelines_of(&sort).is_ok_and(|pipelines| pipelines == 1) {
                    return Ok(Some(sort));
                }
            }
            // The rows of a repartitioning of several partitions arrive in
            // no order: every partition keeps its first rows by a TopK,
            // once the input ended. An input that never ends is sorted as
            // a single partition (the partitions of the pipeline engine).
            if repartition.input().boundedness().is_unbounded() {
                return Self::sort_of(
                    input,
                    sort.expr().clone(),
                    sort.common_prefix_length(),
                    sort.fetch(),
                )
                .map(Some);
            }
            let topk = PhysicalTopKOfPartitions::try_new(
                input,
                partitioning,
                sort.expr().clone(),
                fetch,
            )?;
            return Ok(Some(Arc::new(topk)));
        }
        let inputs = match limits_every_partition(sort) {
            true => partitions_of(sort.input())?,
            false => None,
        }
        .unwrap_or_else(|| vec![Arc::clone(sort.input())]);
        let mut sorts = inputs
            .iter()
            .map(|input| {
                Self::sort_of(
                    lower_input(input)?,
                    sort.expr().clone(),
                    sort.common_prefix_length(),
                    sort.fetch(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        if sorts.len() == 1 {
            return Ok(sorts.pop());
        }
        let union = PhysicalUnion::try_new_with_schema(sorts, plan.schema())?;
        Ok(Some(Arc::new(union)))
    }

    /// The partial sort of `input` (see [`Self::try_new`]) if it runs in a
    /// single pipeline, the sort of `input` by `expr` otherwise (see
    /// [`Self::try_lower`]).
    fn sort_of(
        input: Arc<dyn PhysicalOperator>,
        expr: LexOrdering,
        common_prefix_length: usize,
        fetch: Option<usize>,
    ) -> Result<Arc<dyn PhysicalOperator>> {
        let partial: Arc<dyn PhysicalOperator> = Arc::new(Self::try_new(
            Arc::clone(&input),
            expr.clone(),
            common_prefix_length,
            fetch,
        )?);
        if pipelines_of(&partial).is_ok_and(|pipelines| pipelines == 1) {
            return Ok(partial);
        }
        Ok(match fetch {
            Some(fetch) if fetch > 0 => {
                Arc::new(PhysicalTopK::try_new(input, expr, fetch)?)
            }
            _ => Arc::new(PhysicalSort::try_new(input, expr, fetch)?),
        })
    }

    /// The ranges of the rows of `batch` with the same prefix, in order.
    pub(crate) fn prefix_ranges(
        &self,
        batch: &RecordBatch,
    ) -> Result<Vec<std::ops::Range<usize>>> {
        let keys = self.expr[..self.common_prefix_length]
            .iter()
            .map(|expr| expr.evaluate_to_sort_column(batch))
            .collect::<Result<Vec<_>>>()?;
        evaluate_partition_ranges(batch.num_rows(), &keys)
    }

    /// Whether the first row of `next` has the prefix of the last row of
    /// `previous`.
    pub(crate) fn same_prefix(
        &self,
        previous: &RecordBatch,
        next: &RecordBatch,
    ) -> Result<bool> {
        let last = previous.slice(previous.num_rows() - 1, 1);
        let first = next.slice(0, 1);
        for expr in &self.expr[..self.common_prefix_length] {
            let last = expr.evaluate_to_sort_column(&last)?;
            let first = expr.evaluate_to_sort_column(&first)?;
            let comparator = make_comparator(
                last.values.as_ref(),
                first.values.as_ref(),
                expr.options,
            )?;
            if !comparator(0, 0).is_eq() {
                return Ok(false);
            }
        }
        Ok(true)
    }

    /// The memory that sorting `rows` (see [`Self::sort`]) holds and
    /// allocates at most: the rows, their concatenation, their sort keys and
    /// sort indices, and the sorted rows.
    pub(crate) fn sort_memory(&self, rows: &[RecordBatch]) -> usize {
        let memory = rows
            .iter()
            .map(RecordBatch::get_array_memory_size)
            .sum::<usize>();
        let count = rows.iter().map(RecordBatch::num_rows).sum::<usize>();
        3 * memory + count * (size_of::<u32>() + size_of::<u64>() * self.expr.len())
    }

    /// `rows` (of complete groups), sorted by all keys, at most `limit` of
    /// them.
    pub(crate) fn sort(
        &self,
        rows: &[RecordBatch],
        limit: Option<usize>,
    ) -> Result<RecordBatch> {
        let schema = self.children[0].schema();
        let batch = concat_batches(&schema, rows)?;
        let keys = self
            .expr
            .iter()
            .map(|expr| expr.evaluate_to_sort_column(&batch))
            .collect::<Result<Vec<_>>>()?;
        let indices = lexsort_to_indices(&keys, limit)?;
        Ok(take_record_batch(&batch, &indices)?)
    }
}

/// Whether `sort` keeps the first rows of every partition of several
/// partitions: it has a fetch, and its input several partitions.
fn limits_every_partition(sort: &PartialSortExec) -> bool {
    sort.fetch().is_some() && sort.input().output_partitioning().partition_count() > 1
}

/// The input and the keys of the `PartialSortExec` that `merge` merges, if
/// it keeps the first `fetch` rows of every partition of several partitions
/// and the merge keeps at most `fetch` rows: these are the first rows of all
/// rows of the input by the keys of the partial sort (of which the keys of
/// the merge are a prefix), as every partition keeps its first `fetch` rows,
/// among which are its rows among the first rows of all partitions. A native
/// TopK of the input keeps them. `None` if `merge` merges anything else, or
/// keeps more rows (the partitions then decide which: see
/// [`PhysicalPartialSort::try_lower`]).
pub(crate) fn first_rows_of_merged_partitions(
    merge: &SortPreservingMergeExec,
) -> Option<(&Arc<dyn ExecutionPlan>, &LexOrdering)> {
    let sort = merge.input().downcast_ref::<PartialSortExec>()?;
    if !limits_every_partition(sort) {
        return None;
    }
    match (merge.fetch(), sort.fetch()) {
        (Some(merged), Some(sorted)) if merged <= sorted => {
            Some((sort.input(), sort.expr()))
        }
        _ => None,
    }
}

/// The plans of the partitions of `plan`, every one of a single partition
/// with the rows of its partition, in their order: of a scan of in-memory
/// batches (every partition with its batches), of a scan of files (every
/// partition with its file group), of a streaming table (every partition
/// with its stream), and of the unions of them and of the nodes above them
/// that compute every partition of the same partition of their input
/// (projections, filters, limits, sorts, aggregations and windows of every
/// partition, and the nodes that only change the size of batches). `None`
/// for any other plan of several partitions, e.g. of a repartitioning, whose
/// rows only an execution of its input distributes over its partitions.
pub(crate) fn partitions_of(
    plan: &Arc<dyn ExecutionPlan>,
) -> Result<Option<Vec<Arc<dyn ExecutionPlan>>>> {
    let count = plan.output_partitioning().partition_count();
    if count <= 1 {
        return Ok(Some(vec![Arc::clone(plan)]));
    }
    if let Some(scan) = plan.downcast_ref::<DataSourceExec>() {
        let source = scan.data_source();
        if let Some(memory) = source.downcast_ref::<MemorySourceConfig>() {
            return memory_partitions(memory, &plan.schema()).map(Some);
        }
        if let Some(files) = source.downcast_ref::<FileScanConfig>() {
            let partitions = files
                .file_groups
                .iter()
                .map(|group| {
                    let config = FileScanConfigBuilder::from(files.clone())
                        .with_file_groups(vec![group.clone()])
                        .build();
                    DataSourceExec::from_data_source(config) as Arc<dyn ExecutionPlan>
                })
                .collect();
            return Ok(Some(partitions));
        }
        return Ok(None);
    }
    if let Some(streaming) = plan.downcast_ref::<StreamingTableExec>() {
        let partitions = streaming
            .partitions()
            .iter()
            .map(|partition| {
                let exec = StreamingTableExec::try_new(
                    Arc::clone(streaming.partition_schema()),
                    vec![Arc::clone(partition)],
                    streaming.projection().as_deref(),
                    streaming.projected_output_ordering(),
                    streaming.is_infinite(),
                    streaming.limit(),
                )?;
                Ok(Arc::new(exec) as Arc<dyn ExecutionPlan>)
            })
            .collect::<Result<Vec<_>>>()?;
        return Ok(Some(partitions));
    }
    if plan.is::<UnionExec>() {
        let mut partitions = Vec::with_capacity(count);
        for input in plan.children() {
            let Some(of_input) = partitions_of(input)? else {
                return Ok(None);
            };
            partitions.extend(of_input);
        }
        return Ok(Some(partitions));
    }
    let of_every_partition = plan.is::<ProjectionExec>()
        || plan.is::<FilterExec>()
        || plan.is::<LocalLimitExec>()
        || is_coalesce_batches(plan)
        || plan.is::<CooperativeExec>()
        || plan.is::<BufferExec>()
        || plan.is::<SortExec>()
        || plan.is::<PartialSortExec>()
        || plan.is::<AggregateExec>()
        || plan.is::<BoundedWindowAggExec>()
        || plan.is::<WindowAggExec>()
        || plan.is::<UnnestExec>()
        || plan.is::<AsyncFuncExec>();
    if of_every_partition
        && let [input] = plan.children()[..]
        && input.output_partitioning().partition_count() == count
        && let Some(partitions) = partitions_of(input)?
    {
        return partitions
            .into_iter()
            .map(|partition| {
                Arc::clone(plan).replace_children(
                    vec![partition],
                    ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
                )
            })
            .collect::<Result<Vec<_>>>()
            .map(Some);
    }
    Ok(None)
}

/// Whether `plan` is a `CoalesceBatchesExec` (which DataFusion does not plan
/// any more, but which other planners may).
#[expect(deprecated)]
fn is_coalesce_batches(plan: &Arc<dyn ExecutionPlan>) -> bool {
    plan.is::<datafusion_physical_plan::coalesce_batches::CoalesceBatchesExec>()
}

/// The scans of the partitions of `config`, whose output schema is
/// `schema`: every one scans the (projected) batches of its partition.
fn memory_partitions(
    config: &MemorySourceConfig,
    schema: &SchemaRef,
) -> Result<Vec<Arc<dyn ExecutionPlan>>> {
    config
        .partitions()
        .iter()
        .map(|batches| {
            let batches = match config.projection() {
                Some(projection) => batches
                    .iter()
                    .map(|batch| batch.project(projection))
                    .collect::<Result<Vec<_>, _>>()?,
                None => batches.clone(),
            };
            let partition =
                MemorySourceConfig::try_new(&[batches], Arc::clone(schema), None)?
                    .with_limit(config.fetch())
                    .with_show_sizes(config.show_sizes())
                    .try_with_sort_information(config.sort_information().to_vec())?;
            Ok(DataSourceExec::from_data_source(partition) as Arc<dyn ExecutionPlan>)
        })
        .collect()
}

/// The number of pipelines that contain `operator` in the pipelines of the
/// operator tree of which it is the root.
pub(crate) fn pipelines_of(operator: &Arc<dyn PhysicalOperator>) -> Result<usize> {
    let sink = Arc::new(PlaceholderSink {
        children: vec![Arc::clone(operator)],
    });
    let graph = PipelineGraph::try_new(sink)?;
    let key = operator_key(operator.as_ref());
    Ok(graph
        .pipelines()
        .iter()
        .filter(|pipeline| {
            pipeline
                .operators()
                .iter()
                .any(|other| operator_key(other.as_ref()) == key)
        })
        .count())
}

/// The sink of the pipelines that [`pipelines_of`] counts.
#[derive(Debug)]
struct PlaceholderSink {
    children: Vec<Arc<dyn PhysicalOperator>>,
}

impl PhysicalOperator for PlaceholderSink {
    fn name(&self) -> &str {
        "PlaceholderSink"
    }

    fn schema(&self) -> SchemaRef {
        self.children[0].schema()
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.children
    }

    fn is_sink(&self) -> bool {
        true
    }
}

/// The memory that buffering `batch` holds: its arrays, and the batch in the
/// list of batches.
fn batch_memory(batch: &RecordBatch) -> usize {
    batch.get_array_memory_size() + batch_overhead(batch)
}

/// The state of the thread of a [`PhysicalPartialSort`].
struct PartialSortState {
    /// The rows of the last group of rows with the same prefix, which may
    /// continue in the next chunk.
    group: Vec<RecordBatch>,
    /// The memory of `group` (see [`batch_memory`]).
    group_memory: usize,
    /// The sorted rows that are not emitted yet, and the first of them.
    sorted: Option<RecordBatch>,
    /// The memory of `sorted`.
    sorted_memory: usize,
    offset: usize,
    /// Whether the chunk that the operator is called with was consumed.
    consumed: bool,
    /// The rows that may still be emitted.
    remaining: Option<usize>,
    reservation: MemoryReservation,
}

impl OperatorState for PartialSortState {}

impl PartialSortState {
    /// Reserve the memory of the buffered and the sorted rows.
    fn reserve(&self) -> Result<()> {
        self.reserve_with(0)
    }

    /// Reserve the memory of the buffered and the sorted rows (their arrays,
    /// and the batches in the list of batches), and `extra` bytes, before the
    /// operator allocates them.
    fn reserve_with(&self, extra: usize) -> Result<()> {
        self.reservation
            .try_resize(self.group_memory + self.sorted_memory + extra)
    }

    /// Buffer `batch`, which continues the last group.
    fn push_group(&mut self, batch: RecordBatch) {
        self.group_memory += batch_memory(&batch);
        self.group.push(batch);
    }

    /// Take the rows of the last group.
    fn take_group(&mut self) -> Vec<RecordBatch> {
        self.group_memory = 0;
        std::mem::take(&mut self.group)
    }

    /// Emit the next chunk of the sorted rows into `output`. Returns whether
    /// sorted rows are left.
    fn emit(&mut self, output: &mut DataChunk) -> bool {
        let Some(sorted) = &self.sorted else {
            return false;
        };
        let len = (sorted.num_rows() - self.offset).min(DEFAULT_VECTOR_SIZE);
        *output = DataChunk::from_record_batch(sorted.slice(self.offset, len));
        self.offset += len;
        if self.offset < sorted.num_rows() {
            return true;
        }
        self.sorted = None;
        self.sorted_memory = 0;
        false
    }

    /// Make `sorted` the sorted rows to emit, keeping at most the rows that
    /// may still be emitted.
    fn set_sorted(&mut self, sorted: RecordBatch) {
        let sorted = match &mut self.remaining {
            Some(remaining) => {
                let rows = sorted.num_rows().min(*remaining);
                *remaining -= rows;
                sorted.slice(0, rows)
            }
            None => sorted,
        };
        self.offset = 0;
        self.sorted = (sorted.num_rows() > 0).then_some(sorted);
        self.sorted_memory = self.sorted.as_ref().map_or(0, batch_memory);
    }

    fn done(&self) -> bool {
        self.remaining == Some(0) && self.sorted.is_none()
    }
}

impl PhysicalOperator for PhysicalPartialSort {
    fn name(&self) -> &str {
        "PhysicalPartialSort"
    }

    fn schema(&self) -> SchemaRef {
        self.children[0].schema()
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.children
    }

    fn description(&self) -> String {
        let fetch = self
            .fetch
            .map(|fetch| format!(", fetch={fetch}"))
            .unwrap_or_default();
        format!(
            "PhysicalPartialSort: expr=[{}], common_prefix_length={}{fetch}",
            self.expr, self.common_prefix_length
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
        let reservation = MemoryConsumer::new("PhysicalPartialSort")
            .register(ctx.task_ctx.memory_pool());
        Ok(Box::new(PartialSortState {
            group: vec![],
            group_memory: 0,
            sorted: None,
            sorted_memory: 0,
            offset: 0,
            consumed: false,
            remaining: self.fetch,
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
        let state =
            downcast_state_mut::<PartialSortState>(lstate.as_any_mut(), self.name())?;
        if !state.consumed {
            state.consumed = true;
            // The rows of the chunk, which the operator holds.
            state.reserve_with(chunk_memory(input))?;
            let batch = input.compact()?;
            let ranges = self.prefix_ranges(&batch)?;
            let Some(last) = ranges.last() else {
                return internal_err!(
                    "PhysicalPartialSort received a chunk without rows"
                );
            };
            let continues = match state.group.last() {
                Some(previous) => self.same_prefix(previous, &batch)?,
                None => true,
            };
            if ranges.len() == 1 && continues {
                // The chunk continues the last group.
                state.push_group(batch);
            } else {
                // The groups before the last group of the chunk are complete.
                let mut complete = state.take_group();
                if last.start > 0 {
                    complete.push(batch.slice(0, last.start));
                }
                state.push_group(batch.slice(last.start, last.len()));
                if !complete.is_empty() {
                    state.reserve_with(self.sort_memory(&complete))?;
                    let sorted = self.sort(&complete, state.remaining)?;
                    state.set_sorted(sorted);
                }
            }
            state.reserve()?;
        }
        if state.emit(output) {
            return Ok(OperatorResult::HaveMoreOutput);
        }
        state.consumed = false;
        state.reserve()?;
        if state.done() {
            return Ok(OperatorResult::Finished);
        }
        Ok(OperatorResult::NeedMoreInput)
    }

    fn requires_final_execute(&self) -> bool {
        true
    }

    fn final_execute(
        &self,
        _ctx: &ExecutionContext,
        output: &mut DataChunk,
        _gstate: &dyn GlobalOperatorState,
        lstate: &mut dyn OperatorState,
    ) -> Result<OperatorFinalizeResult> {
        let state =
            downcast_state_mut::<PartialSortState>(lstate.as_any_mut(), self.name())?;
        // The last group is complete once the input ends.
        if !state.group.is_empty() && state.remaining != Some(0) {
            let group = state.take_group();
            state.reserve_with(self.sort_memory(&group))?;
            let sorted = self.sort(&group, state.remaining)?;
            state.set_sorted(sorted);
            state.reserve()?;
        }
        state.take_group();
        if state.emit(output) {
            return Ok(OperatorFinalizeResult::HaveMoreOutput);
        }
        state.reservation.free();
        Ok(OperatorFinalizeResult::Finished)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::fmt::Write;
    use std::path::Path;

    use arrow::array::{AsArray, Int32Array};
    use arrow::datatypes::{DataType, Field, Int32Type, Schema};
    use datafusion::prelude::SessionContext;
    use datafusion_execution::TaskContext;
    use datafusion_physical_expr::PhysicalSortExpr;
    use datafusion_physical_expr::expressions::{col, lit};

    use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
    use datafusion_physical_plan::union::UnionExec;

    use crate::executor::Executor;
    use crate::operators::lowering_tests::{
        assert_exhausted, assert_lowered, assert_plan_lowered,
        assert_resources_exhausted, count, lowered, pipeline_context_with_memory_limit,
        task_context_with_memory_limit,
    };
    use crate::sources::MemoryScanSource;
    use crate::test_utils::{register_test_tables, test_session_config};
    use datafusion_physical_plan::collect;

    /// Write `rows` rows sorted by `a` and `b` (in groups of about 37 rows
    /// with the same `a` and `b`), whose `c` and `d` are not sorted, to the
    /// CSV file `path`.
    fn write_sorted_csv(path: &Path, rows: i32) -> Result<()> {
        write_sorted_csv_rows(path, 0..rows)
    }

    /// Register `s`, an unbounded table of a sorted CSV file (a stream that
    /// ends), ordered by `a` and `b`.
    async fn register_sorted_stream(ctx: &SessionContext, path: &Path) -> Result<()> {
        register_test_tables(ctx)?;
        let sql = format!(
            "CREATE UNBOUNDED EXTERNAL TABLE s (a INT, b INT, c INT, d INT) \
             STORED AS CSV WITH ORDER (a ASC, b ASC) LOCATION '{}' \
             OPTIONS ('format.has_header' 'true')",
            path.display()
        );
        ctx.sql(&sql).await?.collect().await?;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn partial_sorts_are_native() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("sorted.csv");
        write_sorted_csv(&path, 10_000)?;
        let setup = async |ctx: &SessionContext| register_sorted_stream(ctx, &path).await;
        for sql in [
            "SELECT * FROM s ORDER BY a, b, c, d",
            "SELECT a, b, c FROM s ORDER BY a, b, c DESC",
            "SELECT * FROM s ORDER BY a, b, d NULLS FIRST, c",
            "SELECT * FROM s ORDER BY a, b, c, d LIMIT 50",
            "SELECT * FROM s ORDER BY a, b, c, d LIMIT 5000",
        ] {
            for target_partitions in [1, 4] {
                let config =
                    test_session_config().with_target_partitions(target_partitions);
                let root = assert_lowered(sql, config, &setup).await;
                assert_eq!(count::<PhysicalPartialSort>(&root), 1, "{sql}");
            }
        }
        Ok(())
    }

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]))
    }

    /// A partial sort by `a` (the prefix) and `b` of `batches` (sorted by
    /// `a`), keeping `fetch` rows.
    fn partial_sort(
        batches: Vec<RecordBatch>,
        fetch: Option<usize>,
    ) -> Result<Arc<dyn PhysicalOperator>> {
        let source = MemoryScanSource::try_new(&[batches], schema(), None, None)?
            .with_preserve_order(true);
        let schema = schema();
        let expr = LexOrdering::new(vec![
            PhysicalSortExpr::new_default(col("a", &schema)?),
            PhysicalSortExpr::new_default(col("b", &schema)?),
        ])
        .unwrap();
        Ok(Arc::new(PhysicalPartialSort::try_new(
            Arc::new(source),
            expr,
            1,
            fetch,
        )?))
    }

    fn batch(a: Vec<i32>, b: Vec<i32>) -> RecordBatch {
        RecordBatch::try_new(
            schema(),
            vec![Arc::new(Int32Array::from(a)), Arc::new(Int32Array::from(b))],
        )
        .unwrap()
    }

    fn rows(batches: &[RecordBatch]) -> Vec<(i32, i32)> {
        batches
            .iter()
            .flat_map(|batch| {
                let a = batch.column(0).as_primitive::<Int32Type>();
                let b = batch.column(1).as_primitive::<Int32Type>();
                (0..batch.num_rows())
                    .map(|i| (a.value(i), b.value(i)))
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn groups_that_span_chunks_are_sorted_once_complete() -> Result<()> {
        // A group of `a = 1` over three chunks, one of `a = 2` of a single
        // row, and a group of 5000 rows that the operator emits in chunks.
        let batches = vec![
            batch(vec![0, 0, 1, 1], vec![9, 3, 7, 5]),
            batch(vec![1], vec![1]),
            batch(vec![1, 1, 2, 3], vec![8, 2, 4, 6]),
            batch(vec![3; 5000], (0..5000).rev().collect()),
        ];
        // The rows, sorted by `a` and `b`.
        let mut input = rows(&batches);
        input.sort_unstable();
        let output = Executor::collect(
            partial_sort(batches.clone(), None)?,
            Arc::new(TaskContext::default()),
            4,
        )
        .await?;
        assert!(
            output
                .iter()
                .all(|batch| batch.num_rows() <= DEFAULT_VECTOR_SIZE)
        );
        assert_eq!(rows(&output), input);
        // With a fetch, the first rows.
        for fetch in [0, 1, 3, 7, 9, 4000] {
            let output = Executor::collect(
                partial_sort(batches.clone(), Some(fetch))?,
                Arc::new(TaskContext::default()),
                4,
            )
            .await?;
            assert_eq!(rows(&output), input[..fetch.min(input.len())], "{fetch}");
        }
        Ok(())
    }

    /// The plan of a partial sort by `a` (the prefix) and `b`, keeping `fetch`
    /// rows of every one of two partitions (each sorted by `a`), merged with
    /// a fetch of `merge_fetch` if `merged`.
    fn partitioned_partial_sort(
        fetch: Option<usize>,
        merged: bool,
        merge_fetch: Option<usize>,
    ) -> Arc<dyn ExecutionPlan> {
        let schema = schema();
        let a = PhysicalSortExpr::new_default(col("a", &schema).unwrap());
        let b = PhysicalSortExpr::new_default(col("b", &schema).unwrap());
        let partitions = vec![
            vec![batch(vec![100, 100, 100, 101, 102], vec![3, 1, 2, 0, 7])],
            vec![
                batch(vec![0, 0, 0], vec![5, 1, 4]),
                batch(vec![0, 1, 1], vec![2, 9, 8]),
            ],
        ];
        let source = MemorySourceConfig::try_new(&partitions, Arc::clone(&schema), None)
            .unwrap()
            .try_with_sort_information(vec![LexOrdering::new(vec![a.clone()]).unwrap()])
            .unwrap();
        let scan = DataSourceExec::from_data_source(source);
        let ordering = LexOrdering::new(vec![a, b]).unwrap();
        let sort: Arc<dyn ExecutionPlan> = Arc::new(
            PartialSortExec::new(ordering.clone(), scan, 1)
                .with_preserve_partitioning(true)
                .with_fetch(fetch),
        );
        if !merged {
            return sort;
        }
        Arc::new(SortPreservingMergeExec::new(ordering, sort).with_fetch(merge_fetch))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn fetches_of_several_partitions_limit_every_partition() -> Result<()> {
        // Merged with a fetch of at most the fetch of the partial sorts: the
        // first rows of the merge are the first rows of all rows, which a
        // native TopK of the input of the partial sorts keeps.
        for (fetch, merge_fetch) in [(1, 1), (2, 1), (2, 2), (5, 3), (20, 20)] {
            let make_plan =
                || partitioned_partial_sort(Some(fetch), true, Some(merge_fetch));
            let root = assert_plan_lowered(make_plan, test_session_config(), true).await;
            assert_eq!(count::<PhysicalTopK>(&root), 1, "{fetch}");
            assert_eq!(count::<PhysicalPartialSort>(&root), 0, "{fetch}");
        }
        let ctx = crate::test_utils::datafusion_context(test_session_config());
        let first = collect(
            partitioned_partial_sort(Some(2), true, Some(1)),
            ctx.task_ctx(),
        )
        .await?;
        assert_eq!(rows(&first), vec![(0, 1)]);
        // Without a fetch, the partial sort of several partitions is native,
        // also below a merge.
        for merged in [false, true] {
            let make_plan = || partitioned_partial_sort(None, merged, None);
            let root =
                assert_plan_lowered(make_plan, test_session_config(), merged).await;
            assert_eq!(count::<PhysicalPartialSort>(&root), 1, "{merged}");
        }
        // With a fetch, every partition keeps its first rows: the partial
        // sort is the union of the partial sorts of every partition, alone or
        // below a merge that keeps more rows than every partition.
        for (fetch, merged, merge_fetch) in [
            (2, false, None),
            (1, false, None),
            (20, false, None),
            (2, true, None),
            (2, true, Some(3)),
            (1, true, Some(20)),
        ] {
            let make_plan = || partitioned_partial_sort(Some(fetch), merged, merge_fetch);
            let root =
                assert_plan_lowered(make_plan, test_session_config(), merged).await;
            assert_eq!(count::<PhysicalPartialSort>(&root), 2, "{fetch}");
            assert_eq!(count::<PhysicalUnion>(&root), 1, "{fetch}");
        }
        let both = collect(
            partitioned_partial_sort(Some(2), true, None),
            ctx.task_ctx(),
        )
        .await?;
        assert_eq!(rows(&both), vec![(0, 1), (0, 2), (100, 1), (100, 2)]);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn fetches_of_hash_repartitionings_limit_every_partition() -> Result<()> {
        // The two partitions of `partitioned_partial_sort`, repartitioned by
        // a hash of `b` into three partitions (merged in the order of `a`):
        // every one of them keeps its first row, which DataFusion's
        // partitioner decides.
        let make_plan = |merge_fetch: Option<Option<usize>>| -> Arc<dyn ExecutionPlan> {
            let sort = partitioned_partial_sort(Some(1), false, None);
            let scan = Arc::clone(sort.children()[0]);
            let schema = schema();
            let hash = Partitioning::Hash(vec![col("b", &schema).unwrap()], 3);
            let repartition = RepartitionExec::try_new(scan, hash)
                .unwrap()
                .with_preserve_order();
            let ordering = LexOrdering::new(vec![
                PhysicalSortExpr::new_default(col("a", &schema).unwrap()),
                PhysicalSortExpr::new_default(col("b", &schema).unwrap()),
            ])
            .unwrap();
            let sort: Arc<dyn ExecutionPlan> = Arc::new(
                PartialSortExec::new(ordering.clone(), Arc::new(repartition), 1)
                    .with_preserve_partitioning(true)
                    .with_fetch(Some(1)),
            );
            match merge_fetch {
                None => sort,
                Some(fetch) => Arc::new(
                    SortPreservingMergeExec::new(ordering, sort).with_fetch(fetch),
                ),
            }
        };
        for merge_fetch in [None, Some(None), Some(Some(2))] {
            let root = assert_plan_lowered(
                || make_plan(merge_fetch),
                test_session_config(),
                merge_fetch.is_some(),
            )
            .await;
            assert_eq!(
                count::<PhysicalTopKOfPartitions>(&root),
                1,
                "{merge_fetch:?}"
            );
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn the_rows_of_repartitionings_by_volatile_keys_are_routed_once() -> Result<()>
    {
        // 10,000 ids, repartitioned by a hash of `random()` into four
        // partitions, every one of which keeps up to 10,000 rows: every id
        // once.
        let config = test_session_config();
        let ctx = crate::test_utils::datafusion_context(config.clone());
        let input = ctx
            .sql("SELECT 0 AS k, value AS id, random() AS r FROM generate_series(1, 10000)")
            .await?
            .create_physical_plan()
            .await?;
        let make_plan = || -> Arc<dyn ExecutionPlan> {
            let schema = input.schema();
            let hash = Partitioning::Hash(vec![col("r", &schema).unwrap()], 4);
            let repartition = RepartitionExec::try_new(Arc::clone(&input), hash).unwrap();
            let ordering = LexOrdering::new(vec![
                PhysicalSortExpr::new_default(col("k", &schema).unwrap()),
                PhysicalSortExpr::new_default(col("id", &schema).unwrap()),
            ])
            .unwrap();
            let sort = PartialSortExec::new(ordering, Arc::new(repartition), 1)
                .with_preserve_partitioning(true)
                .with_fetch(Some(10_000));
            let id = vec![(col("id", &schema).unwrap(), "id".to_owned())];
            Arc::new(ProjectionExec::try_new(id, Arc::new(sort)).unwrap())
        };
        let root = assert_plan_lowered(make_plan, config, false).await;
        assert_eq!(count::<PhysicalTopKOfPartitions>(&root), 1);
        let output = Executor::collect(root, ctx.task_ctx(), 4).await?;
        let mut ids = output
            .iter()
            .flat_map(|batch| {
                batch
                    .column(0)
                    .as_primitive::<arrow::datatypes::Int64Type>()
                    .values()
                    .to_vec()
            })
            .collect::<Vec<_>>();
        ids.sort_unstable();
        assert_eq!(ids, (1..=10_000).collect::<Vec<_>>());
        Ok(())
    }

    /// Write the rows `rows` of [`write_sorted_csv`] to the CSV file `path`.
    fn write_sorted_csv_rows(path: &Path, rows: std::ops::Range<i32>) -> Result<()> {
        let mut csv = String::from("a,b,c,d\n");
        for i in rows {
            let (a, b, c) = (i / 1000, (i % 1000) / 37, (i * 7919) % 101);
            let d = if i % 13 == 0 {
                String::new()
            } else {
                ((i * 31) % 17).to_string()
            };
            writeln!(csv, "{a},{b},{c},{d}").unwrap();
        }
        std::fs::write(path, csv)?;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn fetches_of_several_file_groups_limit_every_group() -> Result<()> {
        let dir = tempfile::tempdir()?;
        write_sorted_csv_rows(&dir.path().join("1.csv"), 0..3000)?;
        write_sorted_csv_rows(&dir.path().join("2.csv"), 3000..6000)?;
        let config = test_session_config().with_target_partitions(2);
        let ctx = SessionContext::new_with_config(config.clone());
        let sql = format!(
            "CREATE EXTERNAL TABLE f (a INT, b INT, c INT, d INT) STORED AS CSV \
             WITH ORDER (a ASC, b ASC) LOCATION '{}' OPTIONS ('format.has_header' 'true')",
            dir.path().display()
        );
        ctx.sql(&sql).await?.collect().await?;
        let mut scan = ctx
            .sql("SELECT * FROM f")
            .await?
            .create_physical_plan()
            .await?;
        while !scan.is::<DataSourceExec>() {
            scan = Arc::clone(scan.children()[0]);
        }
        assert_eq!(scan.output_partitioning().partition_count(), 2);
        let schema = scan.schema();
        let ordering = LexOrdering::new(
            ["a", "b", "c", "d"]
                .map(|name| PhysicalSortExpr::new_default(col(name, &schema).unwrap())),
        )
        .unwrap();
        for (fetch, merge_fetch) in [(10, None), (10, Some(None)), (10, Some(Some(15)))] {
            let make_plan = || -> Arc<dyn ExecutionPlan> {
                let sort = Arc::new(
                    PartialSortExec::new(ordering.clone(), Arc::clone(&scan), 2)
                        .with_preserve_partitioning(true)
                        .with_fetch(Some(fetch)),
                );
                match merge_fetch {
                    None => sort,
                    Some(merge_fetch) => Arc::new(
                        SortPreservingMergeExec::new(ordering.clone(), sort)
                            .with_fetch(merge_fetch),
                    ),
                }
            };
            let root =
                assert_plan_lowered(make_plan, config.clone(), merge_fetch.is_some())
                    .await;
            assert_eq!(count::<PhysicalPartialSort>(&root), 2, "{merge_fetch:?}");
        }
        Ok(())
    }

    /// A partial sort by `a` (the prefix) and `b`, keeping `fetch` rows, of
    /// the coalesced union of two scans whose rows all have the same `a`.
    fn partial_sort_of_union(fetch: Option<usize>) -> Arc<dyn ExecutionPlan> {
        let scan = |b: Vec<i32>| -> Arc<dyn ExecutionPlan> {
            let batch = batch(vec![7; b.len()], b);
            MemorySourceConfig::try_new_exec(&[vec![batch]], schema(), None).unwrap()
        };
        let schema = schema();
        let union = UnionExec::try_new(vec![scan(vec![5, 9]), scan(vec![1, 3])]).unwrap();
        let coalesce = Arc::new(CoalescePartitionsExec::new(union));
        let ordering = LexOrdering::new(vec![
            PhysicalSortExpr::new_default(col("a", &schema).unwrap()),
            PhysicalSortExpr::new_default(col("b", &schema).unwrap()),
        ])
        .unwrap();
        Arc::new(PartialSortExec::new(ordering, coalesce, 1).with_fetch(fetch))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn inputs_of_several_pipelines_are_sorted_together() -> Result<()> {
        // The rows of both inputs of the union reach the sort in pipelines
        // of their own: it sorts them together, and keeps the first rows of
        // all of them.
        let ctx = crate::test_utils::datafusion_context(test_session_config());
        for (fetch, expected) in [
            (None, vec![1, 3, 5, 9]),
            (Some(1), vec![1]),
            (Some(3), vec![1, 3, 5]),
        ] {
            let make_plan = || partial_sort_of_union(fetch);
            let root = assert_plan_lowered(make_plan, test_session_config(), true).await;
            assert_eq!(count::<PhysicalPartialSort>(&root), 0, "{fetch:?}");
            assert_eq!(
                count::<PhysicalSort>(&root) + count::<PhysicalTopK>(&root),
                1,
                "{fetch:?}"
            );
            let output = collect(make_plan(), ctx.task_ctx()).await?;
            let b = rows(&output)
                .into_iter()
                .map(|(_, b)| b)
                .collect::<Vec<_>>();
            assert_eq!(b, expected, "{fetch:?}");
        }
        // The partial sort of a single pipeline stays a partial sort.
        let single = partial_sort(vec![batch(vec![1, 1], vec![2, 1])], Some(1))?;
        assert!(pipelines_of(&single).is_ok_and(|pipelines| pipelines == 1));
        Ok(())
    }

    /// A partition of a streaming table, of `batches`.
    #[derive(Debug)]
    struct StreamPartition(Vec<RecordBatch>);

    impl datafusion_physical_plan::streaming::PartitionStream for StreamPartition {
        fn schema(&self) -> &SchemaRef {
            static SCHEMA: std::sync::LazyLock<SchemaRef> =
                std::sync::LazyLock::new(schema);
            &SCHEMA
        }

        fn execute(
            &self,
            _ctx: Arc<TaskContext>,
        ) -> datafusion_execution::SendableRecordBatchStream {
            let batches = futures::stream::iter(self.0.clone().into_iter().map(Ok));
            Box::pin(
                datafusion_physical_plan::stream::RecordBatchStreamAdapter::new(
                    schema(),
                    batches,
                ),
            )
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn fetches_of_several_streams_limit_every_stream() -> Result<()> {
        // Two partitions of a streaming table, with the same prefix: every
        // partition keeps its first row, alone and below a merge that keeps
        // more rows.
        let make_plan = |merge_fetch: Option<Option<usize>>| -> Arc<dyn ExecutionPlan> {
            let schema = schema();
            let partitions = [vec![5, 9], vec![1, 3]]
                .into_iter()
                .map(|b| {
                    let partition = StreamPartition(vec![batch(vec![0; b.len()], b)]);
                    Arc::new(partition)
                        as Arc<dyn datafusion_physical_plan::streaming::PartitionStream>
                })
                .collect();
            let a = PhysicalSortExpr::new_default(col("a", &schema).unwrap());
            let b = PhysicalSortExpr::new_default(col("b", &schema).unwrap());
            let table = StreamingTableExec::try_new(
                Arc::clone(&schema),
                partitions,
                None,
                [LexOrdering::new(vec![a.clone()]).unwrap()],
                false,
                None,
            )
            .unwrap();
            let ordering = LexOrdering::new(vec![a, b]).unwrap();
            let sort: Arc<dyn ExecutionPlan> = Arc::new(
                PartialSortExec::new(ordering.clone(), Arc::new(table), 1)
                    .with_preserve_partitioning(true)
                    .with_fetch(Some(1)),
            );
            match merge_fetch {
                None => sort,
                Some(fetch) => Arc::new(
                    SortPreservingMergeExec::new(ordering, sort).with_fetch(fetch),
                ),
            }
        };
        for merge_fetch in [None, Some(None), Some(Some(2))] {
            let root = assert_plan_lowered(
                || make_plan(merge_fetch),
                test_session_config(),
                merge_fetch.is_some(),
            )
            .await;
            assert_eq!(count::<PhysicalPartialSort>(&root), 2, "{merge_fetch:?}");
        }
        let ctx = crate::test_utils::datafusion_context(test_session_config());
        let merged = collect(make_plan(Some(Some(2))), ctx.task_ctx()).await?;
        assert_eq!(rows(&merged), vec![(0, 1), (0, 5)]);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn the_memory_of_a_group_of_many_batches_is_counted_as_they_arrive() {
        // A group of 20,000 chunks of a row: the memory of the group is kept
        // up to date as they arrive, not counted again for every chunk
        // (which takes quadratic time).
        let batches = (0..20_000)
            .map(|i| batch(vec![1], vec![(i * 7919) % 20_000]))
            .collect::<Vec<_>>();
        let sort = partial_sort(batches, None).unwrap();
        let run = Executor::collect(sort, Arc::new(TaskContext::default()), 4);
        let output = tokio::time::timeout(std::time::Duration::from_secs(60), run)
            .await
            .expect("a group of many chunks is sorted in linear time")
            .unwrap();
        let b = rows(&output)
            .into_iter()
            .map(|(_, b)| b)
            .collect::<Vec<_>>();
        assert_eq!(b, (0..20_000).collect::<Vec<_>>());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn buffered_batches_without_columns_reserve_memory() -> Result<()> {
        // A partial sort by literals of rows without columns, whose buffered
        // batches take memory of their own.
        let schema = Arc::new(Schema::empty());
        let options = arrow::array::RecordBatchOptions::new().with_row_count(Some(10));
        let batches = (0..1000)
            .map(|_| {
                RecordBatch::try_new_with_options(Arc::clone(&schema), vec![], &options)
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        let source =
            MemoryScanSource::try_new(&[batches], Arc::clone(&schema), None, None)?
                .with_preserve_order(true);
        let expr = LexOrdering::new(vec![
            PhysicalSortExpr::new_default(lit(1)),
            PhysicalSortExpr::new_default(lit(2)),
        ])
        .unwrap();
        let sort = PhysicalPartialSort::try_new(Arc::new(source), expr, 1, None)?;
        let output =
            Executor::collect(Arc::new(sort), task_context_with_memory_limit(1), 4).await;
        assert_exhausted(output);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn buffered_rows_reserve_memory() -> Result<()> {
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("sorted.csv");
        write_sorted_csv(&path, 10_000)?;
        let ctx = pipeline_context_with_memory_limit(2000);
        register_sorted_stream(&ctx, &path).await?;
        let sql = "SELECT * FROM s ORDER BY a, b, c, d";
        let root = lowered(&ctx, sql).await?;
        assert_eq!(count::<PhysicalPartialSort>(&root), 1);
        assert_resources_exhausted(&ctx, sql).await;
        Ok(())
    }
}
