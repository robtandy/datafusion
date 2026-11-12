// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`PhysicalTopKOfPartitions`]: the first rows of every partition of a hash
//! or range repartitioning (a `PartialSortExec` with a fetch of the
//! partitions of a `RepartitionExec`).

use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow::array::RecordBatch;
use arrow::compute::{concat_batches, lexsort_to_indices, take_record_batch};
use arrow::datatypes::SchemaRef;
use datafusion_common::{Result, internal_err};
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_physical_expr::{LexOrdering, Partitioning};
use datafusion_physical_plan::metrics::Time;
use datafusion_physical_plan::repartition::BatchPartitioner;
use parking_lot::Mutex;

use crate::chunk::{DEFAULT_VECTOR_SIZE, DataChunk};
use crate::operator::{
    ExecutionContext, GlobalOperatorState, GlobalSinkState, GlobalSourceState,
    LocalSinkState, LocalSourceState, OperatorFinalizeResult, OperatorResult,
    OperatorState, PhysicalOperator, SinkFinalize, SinkResult, SourceResult,
    downcast_state, downcast_state_mut,
};
use crate::operators::partial_sort::PhysicalPartialSort;
use crate::sources::batches::{batch_overhead, chunk_memory};

/// Keeps the first `fetch` rows by `expr` of every partition of the hash or
/// range repartitioning `partitioning` of its input, as a `PartialSortExec`
/// with a fetch keeps them of every partition of a `RepartitionExec`: the
/// rows of a partition sorted by a prefix of `expr` are sorted by `expr`, and
/// their first `fetch` rows are its first rows by `expr`.
///
/// Its input sinks into it on all threads: every thread routes the rows of
/// its chunks to their partitions with DataFusion's `BatchPartitioner`, as
/// the `RepartitionExec` does, once for every row (also by keys that are
/// volatile, e.g. computed by `random()`), and every partition keeps its rows,
/// sorted and cut to the first `fetch` rows whenever it holds more than twice
/// as many (and at least [`DEFAULT_VECTOR_SIZE`] more). Once the input ends,
/// the source emits the first rows of every partition, sorted, on all
/// threads. The rows that the partitions hold, and what sorting them
/// allocates, are reserved in the memory pool of the session.
#[derive(Debug)]
pub struct PhysicalTopKOfPartitions {
    children: Vec<Arc<dyn PhysicalOperator>>,
    partitioning: Partitioning,
    expr: LexOrdering,
    fetch: usize,
}

impl PhysicalTopKOfPartitions {
    /// The first `fetch` rows by `expr` of every partition of the hash or
    /// range repartitioning `partitioning` of `input`.
    pub fn try_new(
        input: Arc<dyn PhysicalOperator>,
        partitioning: Partitioning,
        expr: LexOrdering,
        fetch: usize,
    ) -> Result<Self> {
        if !matches!(
            partitioning,
            Partitioning::Hash(..) | Partitioning::Range(_)
        ) {
            return internal_err!(
                "PhysicalTopKOfPartitions keeps the rows of a hash or range repartitioning, got {partitioning}"
            );
        }
        Ok(Self {
            children: vec![input],
            partitioning,
            expr,
            fetch,
        })
    }

    /// The first `fetch` rows of `rows`, sorted.
    fn first_rows(&self, rows: &[RecordBatch]) -> Result<RecordBatch> {
        let batch = concat_batches(&self.schema(), rows)?;
        let keys = self
            .expr
            .iter()
            .map(|expr| expr.evaluate_to_sort_column(&batch))
            .collect::<Result<Vec<_>>>()?;
        let indices = lexsort_to_indices(&keys, Some(self.fetch))?;
        Ok(take_record_batch(&batch, &indices)?)
    }

    /// The memory that sorting `rows` holds and allocates at most: the rows,
    /// their concatenation, their sort keys and sort indices, and the sorted
    /// rows.
    fn sort_memory(&self, rows: &[RecordBatch]) -> usize {
        let memory = rows.iter().map(batch_memory).sum::<usize>();
        let count = rows.iter().map(RecordBatch::num_rows).sum::<usize>();
        3 * memory + count * (size_of::<u32>() + size_of::<u64>() * self.expr.len())
    }
}

/// The memory that holding `batch` takes.
fn batch_memory(batch: &RecordBatch) -> usize {
    batch.get_array_memory_size() + batch_overhead(batch)
}

/// The rows of a partition.
#[derive(Default)]
struct PartitionRows {
    batches: Vec<RecordBatch>,
    rows: usize,
    memory: usize,
}

/// The global sink state of a [`PhysicalTopKOfPartitions`]: the rows of
/// every partition, and their memory.
struct Partitions {
    partitions: Vec<Mutex<PartitionRows>>,
    reservation: MemoryReservation,
}

impl GlobalSinkState for Partitions {}

/// The local sink state of a [`PhysicalTopKOfPartitions`]: the partitioner
/// of the thread.
struct Router(BatchPartitioner);

impl LocalSinkState for Router {}

/// The global source state of a [`PhysicalTopKOfPartitions`]: the first rows
/// of every partition, sorted, and the next chunk of them to claim.
struct FirstRows {
    batches: Vec<RecordBatch>,
    /// For every chunk: its batch and the offset of its first row.
    chunks: Vec<(usize, usize)>,
    next: AtomicUsize,
}

impl GlobalSourceState for FirstRows {
    fn max_threads(&self) -> usize {
        self.chunks.len().max(1)
    }
}

impl PhysicalOperator for PhysicalTopKOfPartitions {
    fn name(&self) -> &str {
        "PhysicalTopKOfPartitions"
    }

    fn schema(&self) -> SchemaRef {
        self.children[0].schema()
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.children
    }

    fn description(&self) -> String {
        format!(
            "PhysicalTopKOfPartitions: partitioning={}, expr=[{}], fetch={}",
            self.partitioning, self.expr, self.fetch
        )
    }

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
        let reservation = MemoryConsumer::new("PhysicalTopKOfPartitions")
            .register(ctx.task_ctx.memory_pool());
        let partitions = (0..self.partitioning.partition_count())
            .map(|_| Mutex::default())
            .collect();
        Ok(Arc::new(Partitions {
            partitions,
            reservation,
        }))
    }

    fn get_local_sink_state(
        &self,
        _ctx: &ExecutionContext,
        _gstate: &dyn GlobalSinkState,
    ) -> Result<Box<dyn LocalSinkState>> {
        let partitioner =
            BatchPartitioner::try_new(self.partitioning.clone(), Time::new(), 0, 1)?;
        Ok(Box::new(Router(partitioner)))
    }

    fn sink(
        &self,
        _ctx: &ExecutionContext,
        input: &DataChunk,
        gstate: &dyn GlobalSinkState,
        lstate: &mut dyn LocalSinkState,
    ) -> Result<SinkResult> {
        let state = downcast_state::<Partitions>(gstate.as_any(), self.name())?;
        let router = downcast_state_mut::<Router>(lstate.as_any_mut(), self.name())?;
        // The rows of the chunk, and their partitions, as large as the chunk
        // at most.
        let estimate = 2 * chunk_memory(input);
        state.reservation.try_grow(estimate)?;
        let batch = input.compact()?;
        let mut routed = vec![];
        for part in router.0.partition_iter(batch)? {
            let (partition, rows) = part?;
            if rows.num_rows() > 0 {
                routed.push((partition, rows));
            }
        }
        let held = routed
            .iter()
            .map(|(_, rows)| batch_memory(rows))
            .sum::<usize>();
        state.reservation.try_grow(held)?;
        state.reservation.shrink(estimate);
        for (partition, rows) in routed {
            let Some(partition) = state.partitions.get(partition) else {
                return internal_err!(
                    "{} routed rows to partition {partition}",
                    self.name()
                );
            };
            let mut partition = partition.lock();
            partition.rows += rows.num_rows();
            partition.memory += batch_memory(&rows);
            partition.batches.push(rows);
            // Cut to the first rows once the partition holds more than twice
            // as many.
            if partition.rows > (2 * self.fetch).max(self.fetch + DEFAULT_VECTOR_SIZE) {
                state
                    .reservation
                    .try_grow(self.sort_memory(&partition.batches))?;
                let first = self.first_rows(&partition.batches)?;
                state
                    .reservation
                    .shrink(self.sort_memory(&partition.batches) + partition.memory);
                partition.rows = first.num_rows();
                partition.memory = batch_memory(&first);
                state.reservation.try_grow(partition.memory)?;
                partition.batches = vec![first];
            }
        }
        Ok(SinkResult::NeedMoreInput)
    }

    fn finalize(
        &self,
        _ctx: &ExecutionContext,
        gstate: &dyn GlobalSinkState,
    ) -> Result<SinkFinalize> {
        let state = downcast_state::<Partitions>(gstate.as_any(), self.name())?;
        for partition in &state.partitions {
            let mut partition = partition.lock();
            if partition.batches.is_empty() {
                continue;
            }
            state
                .reservation
                .try_grow(self.sort_memory(&partition.batches))?;
            let first = self.first_rows(&partition.batches)?;
            state
                .reservation
                .shrink(self.sort_memory(&partition.batches) + partition.memory);
            partition.rows = first.num_rows();
            partition.memory = batch_memory(&first);
            state.reservation.try_grow(partition.memory)?;
            partition.batches = vec![first];
        }
        Ok(SinkFinalize::Ready)
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
        let Some(state) = ctx.global_sink_state(self) else {
            return internal_err!("{} emits its rows before its input ran", self.name());
        };
        let state = downcast_state::<Partitions>(state.as_any(), self.name())?;
        let batches = state
            .partitions
            .iter()
            .flat_map(|partition| partition.lock().batches.clone())
            .collect::<Vec<_>>();
        let chunks = batches
            .iter()
            .enumerate()
            .flat_map(|(index, batch)| {
                (0..batch.num_rows())
                    .step_by(DEFAULT_VECTOR_SIZE)
                    .map(move |offset| (index, offset))
            })
            .collect();
        Ok(Arc::new(FirstRows {
            batches,
            chunks,
            next: AtomicUsize::new(0),
        }))
    }

    fn get_data(
        &self,
        _ctx: &ExecutionContext,
        gstate: &dyn GlobalSourceState,
        _lstate: &mut dyn LocalSourceState,
    ) -> Result<SourceResult> {
        let state = downcast_state::<FirstRows>(gstate.as_any(), self.name())?;
        let next = state.next.fetch_add(1, Ordering::Relaxed);
        let Some(&(index, offset)) = state.chunks.get(next) else {
            return Ok(SourceResult::Finished);
        };
        let batch = &state.batches[index];
        let len = (batch.num_rows() - offset).min(DEFAULT_VECTOR_SIZE);
        Ok(SourceResult::Chunk(DataChunk::from_record_batch(
            batch.slice(offset, len),
        )))
    }
}

/// Sorts the rows of every partition of the hash or range repartitioning
/// `partitioning` of its input, a single partition sorted by the first
/// `common_prefix_length` keys of `expr`, keeping the first `fetch` rows of
/// every partition, as a `PartialSortExec` with a fetch does of every
/// partition of a `RepartitionExec` of a single partition: the rows that the
/// repartitioning routes to a partition keep their order, sorted by the
/// prefix.
///
/// It streams, also over inputs that never end: it routes the rows of every
/// chunk once (with DataFusion's `BatchPartitioner`), buffers the rows of
/// the last group of every partition (with the same prefix) until a row of
/// another prefix arrives, and emits the rows of the groups that are
/// complete, sorted (see [`PhysicalPartialSort`]), until every partition
/// emitted its first `fetch` rows. It is not parallel: a single thread sees
/// the chunks of its input in their order. The rows that it buffers and has
/// not emitted yet, and what sorting a group allocates, are reserved in the
/// memory pool of the session.
#[derive(Debug)]
pub struct PhysicalPartialSortOfPartitions {
    /// The sort of the rows of a partition, over the input.
    sort: PhysicalPartialSort,
    partitioning: Partitioning,
    fetch: usize,
}

impl PhysicalPartialSortOfPartitions {
    /// Sort the rows of every partition of `partitioning` of `input`, sorted
    /// by the first `common_prefix_length` keys of `expr`, by `expr`,
    /// keeping their first `fetch` rows.
    pub fn try_new(
        input: Arc<dyn PhysicalOperator>,
        partitioning: Partitioning,
        expr: LexOrdering,
        common_prefix_length: usize,
        fetch: usize,
    ) -> Result<Self> {
        if !matches!(
            partitioning,
            Partitioning::Hash(..) | Partitioning::Range(_)
        ) {
            return internal_err!(
                "PhysicalPartialSortOfPartitions sorts the rows of a hash or range repartitioning, got {partitioning}"
            );
        }
        Ok(Self {
            sort: PhysicalPartialSort::try_new(input, expr, common_prefix_length, None)?,
            partitioning,
            fetch,
        })
    }

    /// Add `rows`, routed to `partition`, to its last group: the groups that
    /// they complete are sorted, cut to the rows that the partition may
    /// still emit, and queued in `state`.
    fn add(
        &self,
        state: &mut SortState,
        partition: usize,
        rows: RecordBatch,
    ) -> Result<()> {
        let fetch = self.fetch;
        let Some(group) = state.partitions.get_mut(partition) else {
            return internal_err!("{} routed rows to partition {partition}", self.name());
        };
        if group.emitted == fetch {
            return Ok(());
        }
        let ranges = self.sort.prefix_ranges(&rows)?;
        let Some(last) = ranges.last().cloned() else {
            return Ok(());
        };
        let continues = match group.rows.last() {
            Some(previous) => self.sort.same_prefix(previous, &rows)?,
            None => true,
        };
        if ranges.len() == 1 && continues {
            group.memory += batch_memory(&rows);
            group.rows.push(rows);
            return Ok(());
        }
        // The groups before the last group of the chunk are complete.
        let mut complete = std::mem::take(&mut group.rows);
        if last.start > 0 {
            complete.push(rows.slice(0, last.start));
        }
        let rest = rows.slice(last.start, last.len());
        let held = std::mem::replace(&mut group.memory, batch_memory(&rest));
        group.rows.push(rest);
        let limit = fetch - group.emitted;
        state
            .reservation
            .try_grow(self.sort.sort_memory(&complete))?;
        let sorted = self.sort.sort(&complete, Some(limit))?;
        let sorted_memory = self.sort.sort_memory(&complete) + held;
        state
            .reservation
            .shrink(sorted_memory.min(state.reservation.size()));
        let group = &mut state.partitions[partition];
        group.emitted += sorted.num_rows();
        if group.emitted == fetch {
            group.memory = 0;
            group.rows.clear();
        }
        state.queue(sorted)
    }

    /// Emit the next chunk of the queued rows into `output`. Returns whether
    /// queued rows are left.
    fn emit(state: &mut SortState, output: &mut DataChunk) -> bool {
        let Some(front) = state.output.front_mut() else {
            return false;
        };
        let len = front.num_rows().min(DEFAULT_VECTOR_SIZE);
        *output = DataChunk::from_record_batch(front.slice(0, len));
        if len == front.num_rows() {
            let emitted = state
                .output
                .pop_front()
                .map_or(0, |batch| batch_memory(&batch));
            state
                .reservation
                .shrink(emitted.min(state.reservation.size()));
        } else {
            *front = front.slice(len, front.num_rows() - len);
        }
        !state.output.is_empty()
    }
}

/// The last group of a partition of a [`PhysicalPartialSortOfPartitions`],
/// and the number of rows that the partition emitted.
#[derive(Default)]
struct PartitionGroup {
    rows: Vec<RecordBatch>,
    memory: usize,
    emitted: usize,
}

/// The state of the thread of a [`PhysicalPartialSortOfPartitions`].
struct SortState {
    partitioner: BatchPartitioner,
    partitions: Vec<PartitionGroup>,
    /// The sorted rows of complete groups that are not emitted yet.
    output: VecDeque<RecordBatch>,
    /// Whether the chunk that the operator is called with was consumed.
    consumed: bool,
    reservation: MemoryReservation,
}

impl OperatorState for SortState {}

impl SortState {
    /// Queue `sorted`, reserving its memory.
    fn queue(&mut self, sorted: RecordBatch) -> Result<()> {
        if sorted.num_rows() > 0 {
            self.reservation.try_grow(batch_memory(&sorted))?;
            self.output.push_back(sorted);
        }
        Ok(())
    }

    /// Whether every partition emitted its rows.
    fn done(&self, fetch: usize) -> bool {
        self.output.is_empty()
            && self
                .partitions
                .iter()
                .all(|partition| partition.emitted == fetch)
    }
}

impl PhysicalOperator for PhysicalPartialSortOfPartitions {
    fn name(&self) -> &str {
        "PhysicalPartialSortOfPartitions"
    }

    fn schema(&self) -> SchemaRef {
        self.sort.schema()
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        self.sort.children()
    }

    fn description(&self) -> String {
        format!(
            "PhysicalPartialSortOfPartitions: partitioning={}, {}, fetch={}",
            self.partitioning,
            self.sort.description(),
            self.fetch
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
        let reservation = MemoryConsumer::new("PhysicalPartialSortOfPartitions")
            .register(ctx.task_ctx.memory_pool());
        Ok(Box::new(SortState {
            partitioner: BatchPartitioner::try_new(
                self.partitioning.clone(),
                Time::new(),
                0,
                1,
            )?,
            partitions: (0..self.partitioning.partition_count())
                .map(|_| PartitionGroup::default())
                .collect(),
            output: VecDeque::new(),
            consumed: false,
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
        let state = downcast_state_mut::<SortState>(lstate.as_any_mut(), self.name())?;
        if !state.consumed {
            state.consumed = true;
            // The rows of the chunk, and their partitions, as large as the
            // chunk at most, which the partitions hold.
            state.reservation.try_grow(2 * chunk_memory(input))?;
            let batch = input.compact()?;
            let mut routed = vec![];
            for part in state.partitioner.partition_iter(batch)? {
                let (partition, rows) = part?;
                if rows.num_rows() > 0 {
                    routed.push((partition, rows));
                }
            }
            state.reservation.shrink(2 * chunk_memory(input));
            let held = routed
                .iter()
                .map(|(_, rows)| batch_memory(rows))
                .sum::<usize>();
            state.reservation.try_grow(held)?;
            for (partition, rows) in routed {
                self.add(state, partition, rows)?;
            }
            // The memory of the groups, and of the queued rows.
            let memory = state
                .partitions
                .iter()
                .map(|group| group.memory)
                .sum::<usize>()
                + state.output.iter().map(batch_memory).sum::<usize>();
            state.reservation.try_resize(memory)?;
        }
        if Self::emit(state, output) {
            return Ok(OperatorResult::HaveMoreOutput);
        }
        state.consumed = false;
        if state.done(self.fetch) {
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
        let state = downcast_state_mut::<SortState>(lstate.as_any_mut(), self.name())?;
        // The last groups are complete once the input ends.
        for partition in 0..state.partitions.len() {
            let group = &mut state.partitions[partition];
            if group.rows.is_empty() || group.emitted == self.fetch {
                continue;
            }
            let rows = std::mem::take(&mut group.rows);
            let held = std::mem::take(&mut group.memory);
            let limit = self.fetch - group.emitted;
            state.reservation.try_grow(self.sort.sort_memory(&rows))?;
            let sorted = self.sort.sort(&rows, Some(limit))?;
            state.reservation.shrink(
                (self.sort.sort_memory(&rows) + held).min(state.reservation.size()),
            );
            state.partitions[partition].emitted += sorted.num_rows();
            state.queue(sorted)?;
        }
        if Self::emit(state, output) {
            return Ok(OperatorFinalizeResult::HaveMoreOutput);
        }
        state.reservation.free();
        Ok(OperatorFinalizeResult::Finished)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use datafusion_physical_expr::PhysicalSortExpr;
    use datafusion_physical_expr::expressions::col;
    use datafusion_physical_plan::ExecutionPlan;
    use datafusion_physical_plan::repartition::RepartitionExec;
    use datafusion_physical_plan::sorts::partial_sort::PartialSortExec;
    use datafusion_physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;

    use crate::executor::Executor;
    use crate::operators::lowering_tests::{
        assert_exhausted, assert_plan_lowered, count, task_context_with_memory_limit,
    };
    use crate::test_utils::{datafusion_context, test_session_config};

    /// The partial sort by `k` (the prefix, constant) and `id`, with a fetch
    /// of `fetch`, of the four partitions of a hash repartitioning by `id` of
    /// 10,000 rows (cut to their first rows many times), merged (with a
    /// fetch of `merge_fetch`) if `merged`.
    async fn partial_sort_of_partitions(
        fetch: usize,
        merged: bool,
        merge_fetch: Option<usize>,
    ) -> Result<impl Fn() -> Arc<dyn ExecutionPlan>> {
        let ctx = datafusion_context(test_session_config());
        let input = ctx
            .sql("SELECT 0 AS k, value * 7919 % 10007 AS id FROM generate_series(1, 10000)")
            .await?
            .create_physical_plan()
            .await?;
        Ok(move || -> Arc<dyn ExecutionPlan> {
            let schema = input.schema();
            let hash = Partitioning::Hash(vec![col("id", &schema).unwrap()], 4);
            let repartition = RepartitionExec::try_new(Arc::clone(&input), hash).unwrap();
            let ordering = LexOrdering::new(vec![
                PhysicalSortExpr::new_default(col("k", &schema).unwrap()),
                PhysicalSortExpr::new_default(col("id", &schema).unwrap()),
            ])
            .unwrap();
            let sort: Arc<dyn ExecutionPlan> = Arc::new(
                PartialSortExec::new(ordering.clone(), Arc::new(repartition), 1)
                    .with_preserve_partitioning(true)
                    .with_fetch(Some(fetch)),
            );
            match merged {
                false => sort,
                true => Arc::new(
                    SortPreservingMergeExec::new(ordering, sort).with_fetch(merge_fetch),
                ),
            }
        })
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn the_first_rows_of_every_partition_are_native() -> Result<()> {
        for (fetch, merged, merge_fetch) in [
            (5, false, None),
            (1, true, None),
            (5, true, Some(12)),
            (3000, false, None),
        ] {
            let make_plan =
                partial_sort_of_partitions(fetch, merged, merge_fetch).await?;
            let root =
                assert_plan_lowered(make_plan, test_session_config(), merged).await;
            assert_eq!(count::<PhysicalTopKOfPartitions>(&root), 1, "{fetch}");
        }
        Ok(())
    }

    /// A partition of batches of `(a, b)` rows, that never ends if
    /// `endless`.
    #[derive(Debug)]
    struct Pairs(Vec<RecordBatch>, bool);

    fn pairs_schema() -> SchemaRef {
        Arc::new(arrow::datatypes::Schema::new(vec![
            arrow::datatypes::Field::new("a", arrow::datatypes::DataType::Int64, false),
            arrow::datatypes::Field::new("b", arrow::datatypes::DataType::Int64, false),
        ]))
    }

    impl datafusion_physical_plan::streaming::PartitionStream for Pairs {
        fn schema(&self) -> &SchemaRef {
            static SCHEMA: std::sync::LazyLock<SchemaRef> =
                std::sync::LazyLock::new(pairs_schema);
            &SCHEMA
        }

        fn execute(
            &self,
            _ctx: Arc<datafusion_execution::TaskContext>,
        ) -> datafusion_execution::SendableRecordBatchStream {
            use futures::StreamExt;
            let batches = futures::stream::iter(self.0.clone().into_iter().map(Ok));
            let batches = match self.1 {
                true => batches.chain(futures::stream::pending()).boxed(),
                false => batches.boxed(),
            };
            Box::pin(
                datafusion_physical_plan::stream::RecordBatchStreamAdapter::new(
                    pairs_schema(),
                    batches,
                ),
            )
        }
    }

    /// The partial sort by `a` (the prefix) and `b`, with a fetch of
    /// `fetch`, of the three partitions of a hash repartitioning by `b` of a
    /// stream (that never ends if `endless`) of the rows `a = v / 10` (sorted)
    /// and `b = v * 7 % 10` of the values `v` below 1000, merged (with a
    /// fetch of `merge_fetch`) if `merged`.
    fn partial_sort_of_a_stream(
        fetch: usize,
        endless: bool,
        merged: bool,
        merge_fetch: Option<usize>,
    ) -> Arc<dyn ExecutionPlan> {
        let schema = pairs_schema();
        let batches = (0..1000)
            .collect::<Vec<i64>>()
            .chunks(7)
            .map(|values| {
                let a = values.iter().map(|v| v / 10).collect::<Vec<_>>();
                let b = values.iter().map(|v| v * 7 % 10).collect::<Vec<_>>();
                RecordBatch::try_new(
                    pairs_schema(),
                    vec![
                        Arc::new(arrow::array::Int64Array::from(a)),
                        Arc::new(arrow::array::Int64Array::from(b)),
                    ],
                )
                .unwrap()
            })
            .collect();
        let a = PhysicalSortExpr::new_default(col("a", &schema).unwrap());
        let b = PhysicalSortExpr::new_default(col("b", &schema).unwrap());
        let stream = datafusion_physical_plan::streaming::StreamingTableExec::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Pairs(batches, endless))],
            None,
            [LexOrdering::new(vec![a.clone()]).unwrap()],
            endless,
            None,
        )
        .unwrap();
        let hash = Partitioning::Hash(vec![col("b", &schema).unwrap()], 3);
        let repartition = RepartitionExec::try_new(Arc::new(stream), hash).unwrap();
        let ordering = LexOrdering::new(vec![a, b]).unwrap();
        let sort: Arc<dyn ExecutionPlan> = Arc::new(
            PartialSortExec::new(ordering.clone(), Arc::new(repartition), 1)
                .with_preserve_partitioning(true)
                .with_fetch(Some(fetch)),
        );
        match merged {
            false => sort,
            true => Arc::new(
                SortPreservingMergeExec::new(ordering, sort).with_fetch(merge_fetch),
            ),
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn the_partitions_of_a_stream_are_sorted_as_they_arrive() {
        // Of a stream that ends.
        for (fetch, merged, merge_fetch) in [
            (2, false, None),
            (25, false, None),
            (2, true, None),
            (5, true, Some(12)),
            (5000, false, None),
        ] {
            let make_plan =
                || partial_sort_of_a_stream(fetch, false, merged, merge_fetch);
            let root =
                assert_plan_lowered(make_plan, test_session_config(), merged).await;
            assert_eq!(
                count::<PhysicalPartialSortOfPartitions>(&root),
                1,
                "{fetch}"
            );
        }
        // Of a stream that never ends: every partition ends once it emitted
        // its rows (DataFusion's partial sorts end too).
        for (fetch, merged, merge_fetch) in [(2, true, Some(6)), (3, true, None)] {
            let make_plan = || partial_sort_of_a_stream(fetch, true, merged, merge_fetch);
            let root =
                assert_plan_lowered(make_plan, test_session_config(), merged).await;
            assert_eq!(
                count::<PhysicalPartialSortOfPartitions>(&root),
                1,
                "{fetch}"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn the_rows_of_the_partitions_reserve_memory() -> Result<()> {
        let make_plan = partial_sort_of_partitions(5, false, None).await?;
        let ctx = datafusion_context(test_session_config());
        let state = ctx.state();
        let root = crate::planner::create_pipeline_plan_with_runtime(
            make_plan(),
            state.config_options(),
            state.runtime_env(),
        )?;
        let operator = Arc::clone(root.operator());
        assert_eq!(count::<PhysicalTopKOfPartitions>(&operator), 1);
        let output =
            Executor::collect(operator, task_context_with_memory_limit(1), 4).await;
        assert_exhausted(output);
        // In a pool that holds them, nothing stays reserved.
        let operator = Arc::clone(root.operator());
        let task_ctx = task_context_with_memory_limit(64 << 20);
        let output = Executor::collect(operator, Arc::clone(&task_ctx), 4).await?;
        assert_eq!(output.iter().map(RecordBatch::num_rows).sum::<usize>(), 20);
        assert_eq!(task_ctx.memory_pool().reserved(), 0);
        Ok(())
    }
}
