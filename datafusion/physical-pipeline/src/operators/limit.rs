// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! `PhysicalLimit`: `OFFSET` / `LIMIT` with early pipeline termination.
//!
//! [`PhysicalLimit`] skips the first `skip` rows of its input and emits at
//! most `fetch` of the rows after them, like DataFusion's `GlobalLimitExec`
//! (a `LocalLimitExec` over a single partition is a limit with `skip == 0`).
//!
//! It is a streaming operator, modelled after DuckDB's
//! `PhysicalStreamingLimit`: no rows are buffered. Every input chunk claims
//! the positions of its rows from a counter that is shared by all threads and
//! all pipelines that contain the limit (the pipelines of the inputs of a
//! `UNION ALL` below it). The operator emits the rows whose position is in
//! `skip..skip + fetch`, slicing the chunks that straddle a boundary (with
//! [`DataChunk::slice`], which does not copy), and returns
//! [`OperatorResult::Finished`] as soon as all positions up to `skip + fetch`
//! are claimed. That stops every thread of every pipeline that contains the
//! limit, so the scans and joins below it stop early, and cancels the
//! pipeline breakers whose output is then no longer needed.
//!
//! Which rows a limit returns depends on the order of its input:
//!
//! * If the order is arbitrary (e.g. DataFusion merges several partitions in
//!   the order in which their batches arrive), the limit is a parallel
//!   operator. The threads claim positions in the order in which they reach
//!   the limit, so any `skip` rows are skipped and any `fetch` rows are
//!   returned, exactly the freedom DataFusion's limit has.
//! * If the order matters (e.g. the input is sorted), the limit is created
//!   with [`PhysicalLimit::with_preserve_order`]. It is then not a parallel
//!   operator, so the pipelines that contain it run on a single thread and,
//!   for the inputs of an ordered union, one after the other: the positions
//!   are the positions in the input order, and the limit returns exactly the
//!   rows `skip..skip + fetch` of its input, in order. (DuckDB plans the same
//!   non-parallel streaming limit when the source cannot restore the order
//!   of parallel threads with batch indexes, which this engine does not
//!   have.)

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow::datatypes::SchemaRef;
use datafusion_common::{Result, internal_err};
use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion_physical_plan::limit::{GlobalLimitExec, LocalLimitExec};
use datafusion_physical_plan::{ExecutionPlan, ExecutionPlanProperties};

use crate::chunk::DataChunk;
use crate::operator::{
    ExecutionContext, GlobalOperatorState, OperatorResult, OperatorState,
    PhysicalOperator, downcast_state,
};

/// Skips the first `skip` rows of its input and emits at most `fetch` of the
/// rows after them (all of them if `fetch` is `None`).
///
/// See the [module documentation](self) for how the rows are counted over
/// all threads, and when the order of the input is preserved.
#[derive(Debug)]
pub struct PhysicalLimit {
    /// The input, the only child.
    children: Vec<Arc<dyn PhysicalOperator>>,
    skip: usize,
    fetch: Option<usize>,
    preserve_order: bool,
}

impl PhysicalLimit {
    /// Skip `skip` rows of `input`, then emit at most `fetch` rows.
    ///
    /// The limit is a parallel operator for an input whose order does not
    /// matter, see [`Self::with_preserve_order`].
    pub fn new(
        input: Arc<dyn PhysicalOperator>,
        skip: usize,
        fetch: Option<usize>,
    ) -> Self {
        Self {
            children: vec![input],
            skip,
            fetch,
            preserve_order: false,
        }
    }

    /// If `preserve_order` is `true`, the limit selects the rows by their
    /// position in the order of its input: it is not a parallel operator, so
    /// that the pipelines that contain it run on a single thread, which sees
    /// the chunks of the input in order.
    pub fn with_preserve_order(mut self, preserve_order: bool) -> Self {
        self.preserve_order = preserve_order;
        self
    }

    /// Lower a DataFusion limit node to a [`PhysicalLimit`]:
    ///
    /// * `GlobalLimitExec` (its `skip` and `fetch`);
    /// * `LocalLimitExec` over a single partition (a limit of that
    ///   partition). Over several partitions it limits every partition to
    ///   `fetch` rows, which a limit of the whole input cannot do: it is not
    ///   lowered (`Ok(None)`), unless the merge above it makes it redundant
    ///   (see below);
    /// * `CoalescePartitionsExec` with a `fetch`: the engine merges the
    ///   partitions of its input anyway, what remains is the limit. If its
    ///   input is a `LocalLimitExec` that keeps at least `fetch` rows of every
    ///   partition, that limit does not restrict which rows the merge can
    ///   return (at most `fetch` rows, which may all come from one
    ///   partition), and the native limit replaces both.
    ///
    /// The order of the input is preserved (see [`Self::with_preserve_order`])
    /// unless it is arbitrary in DataFusion too: the input merges several
    /// partitions in the order in which their batches arrive.
    ///
    /// `lower_input` lowers the input of the node. Returns `Ok(None)` for any
    /// other node.
    pub fn try_lower<F>(
        plan: &Arc<dyn ExecutionPlan>,
        mut lower_input: F,
    ) -> Result<Option<Self>>
    where
        F: FnMut(&Arc<dyn ExecutionPlan>) -> Result<Arc<dyn PhysicalOperator>>,
    {
        let (input, skip, fetch, preserve_order) =
            if let Some(limit) = plan.downcast_ref::<GlobalLimitExec>() {
                let input = limit.input();
                (input, limit.skip(), limit.fetch(), order_matters(input))
            } else if let Some(limit) = plan.downcast_ref::<LocalLimitExec>() {
                let input = limit.input();
                if input.output_partitioning().partition_count() > 1 {
                    return Ok(None);
                }
                (input, 0, Some(limit.fetch()), order_matters(input))
            } else if let Some(coalesce) = plan.downcast_ref::<CoalescePartitionsExec>()
                && let Some(fetch) = coalesce.fetch()
            {
                let input = match coalesce.input().downcast_ref::<LocalLimitExec>() {
                    Some(local) if local.fetch() >= fetch => local.input(),
                    _ => coalesce.input(),
                };
                let preserve_order = input.output_partitioning().partition_count() <= 1
                    && order_matters(input);
                (input, 0, Some(fetch), preserve_order)
            } else {
                return Ok(None);
            };
        let input = lower_input(input)?;
        if input.schema() != plan.schema() {
            return internal_err!(
                "The lowered input {} of {} has schema {:?}, expected {:?}",
                input.name(),
                plan.name(),
                input.schema(),
                plan.schema()
            );
        }
        Ok(Some(
            Self::new(input, skip, fetch).with_preserve_order(preserve_order),
        ))
    }

    /// The input of the limit.
    pub fn input(&self) -> &Arc<dyn PhysicalOperator> {
        &self.children[0]
    }

    /// The number of rows to skip.
    pub fn skip(&self) -> usize {
        self.skip
    }

    /// The maximum number of rows to emit after the skipped ones, `None` for
    /// all of them.
    pub fn fetch(&self) -> Option<usize> {
        self.fetch
    }

    /// Whether the rows are selected by their position in the input order,
    /// see [`Self::with_preserve_order`].
    pub fn preserve_order(&self) -> bool {
        self.preserve_order
    }

    /// The position after the last row to emit, `None` if all rows after the
    /// skipped ones are emitted.
    fn end(&self) -> Option<usize> {
        self.fetch.map(|fetch| self.skip.saturating_add(fetch))
    }
}

/// Whether the rows that a limit selects from `input`, a plan with a single
/// output partition, depend on its order. They do not if `input` merges
/// several partitions in the order in which their batches arrive (a
/// `CoalescePartitionsExec`, possibly below operators that keep the order of
/// their input): DataFusion's result is arbitrary then.
///
/// In all other cases, the order is deterministic (e.g. sorted, or the order
/// of a single partition), and the limit must see its input in that order.
/// That costs no parallelism compared to DataFusion, which executes the
/// single partition on one thread too.
fn order_matters(input: &Arc<dyn ExecutionPlan>) -> bool {
    let mut plan = input;
    loop {
        if let Some(coalesce) = plan.downcast_ref::<CoalescePartitionsExec>()
            && coalesce.input().output_partitioning().partition_count() > 1
        {
            return false;
        }
        match plan.children().as_slice() {
            [child] if plan.maintains_input_order().first() == Some(&true) => {
                plan = child;
            }
            _ => return true,
        }
    }
}

/// The global operator state of a [`PhysicalLimit`], shared by all threads
/// and pipelines.
#[derive(Debug, Default)]
struct LimitState {
    /// The number of row positions claimed by the input chunks so far.
    claimed: AtomicUsize,
}

impl GlobalOperatorState for LimitState {}

impl PhysicalOperator for PhysicalLimit {
    fn name(&self) -> &str {
        "PhysicalLimit"
    }

    fn schema(&self) -> SchemaRef {
        self.input().schema()
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.children
    }

    fn description(&self) -> String {
        let fetch = self
            .fetch
            .map_or_else(|| "None".to_string(), |fetch| fetch.to_string());
        let mut description = format!("PhysicalLimit: skip={}, fetch={fetch}", self.skip);
        if self.preserve_order {
            description.push_str(", preserve_order=true");
        }
        description
    }

    fn parallel_operator(&self) -> bool {
        !self.preserve_order
    }

    fn get_global_operator_state(
        &self,
        _ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalOperatorState>> {
        Ok(Arc::new(LimitState::default()))
    }

    fn execute(
        &self,
        _ctx: &ExecutionContext,
        input: &DataChunk,
        output: &mut DataChunk,
        gstate: &dyn GlobalOperatorState,
        _lstate: &mut dyn OperatorState,
    ) -> Result<OperatorResult> {
        let state = downcast_state::<LimitState>(gstate.as_any(), self.name())?;
        let end = self.end();
        // Nothing is emitted after the skipped rows: the input is not needed.
        if end == Some(self.skip) {
            return Ok(OperatorResult::Finished);
        }
        // Avoid claiming positions that cannot matter any more.
        let claimed = state.claimed.load(Ordering::Relaxed);
        match end {
            // All positions up to the end were claimed (by another thread).
            Some(end) if claimed >= end => return Ok(OperatorResult::Finished),
            // All rows are skipped already and there is no fetch: the
            // remaining rows are all emitted.
            None if claimed >= self.skip => {
                *output = input.clone();
                return Ok(OperatorResult::NeedMoreInput);
            }
            _ => {}
        }

        let len = input.len();
        let start = state.claimed.fetch_add(len, Ordering::Relaxed);
        let stop = start.saturating_add(len);
        // The positions to emit: `skip..end`, of `start..stop`.
        let first = start.max(self.skip);
        let last = end.map_or(stop, |end| end.min(stop));
        if first < last {
            *output = if first == start && last == stop {
                input.clone()
            } else {
                input.slice(first - start, last - first)
            };
        }
        Ok(match end {
            Some(end) if stop >= end => OperatorResult::Finished,
            _ => OperatorResult::NeedMoreInput,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::compute::concat_batches;
    use datafusion_datasource::memory::MemorySourceConfig;
    use datafusion_physical_expr::expressions::col;
    use datafusion_physical_expr::{LexOrdering, PhysicalSortExpr};
    use datafusion_physical_plan::sorts::sort::SortExec;
    use datafusion_physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;

    use crate::bridge::{BridgeOperator, bridged_plans};
    use crate::chunk::SelectionVector;
    use crate::explain::format_pipelines;
    use crate::operators::union::PhysicalUnion;
    use crate::operators::union::test_support::{
        TestSource, assert_native_plan_byte_eq, assert_parity, assert_parity_with_config,
        count, run, sorted_values, values, values_chunk, values_chunks, values_schema,
    };
    use crate::test_utils::{test_session_config, test_table_t1};

    /// Chunks of all sizes, so that `skip` and `skip + fetch` fall on chunk
    /// boundaries and inside chunks.
    const SIZES: [usize; 7] = [7, 2048, 1, 100, 2048, 30, 1000];

    fn limit(
        input: Arc<dyn PhysicalOperator>,
        skip: usize,
        fetch: Option<usize>,
        preserve_order: bool,
    ) -> Arc<dyn PhysicalOperator> {
        Arc::new(
            PhysicalLimit::new(input, skip, fetch).with_preserve_order(preserve_order),
        )
    }

    fn source(chunks: Vec<DataChunk>) -> Arc<dyn PhysicalOperator> {
        Arc::new(TestSource::new(chunks))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn limit_emits_the_rows_from_skip_to_skip_plus_fetch() -> Result<()> {
        let total = SIZES.iter().sum::<usize>();
        let cases = [
            (0, None),
            (0, Some(0)),
            (0, Some(1)),
            (0, Some(7)),
            (0, Some(8)),
            (0, Some(2055)),
            (0, Some(total)),
            (0, Some(total + 1)),
            (0, Some(usize::MAX)),
            (1, Some(5)),
            (6, Some(2)),
            (7, Some(2048)),
            (7, None),
            (2054, Some(3)),
            (2055, Some(1)),
            (100, Some(4000)),
            (5, Some(0)),
            (total - 1, Some(5)),
            (total - 1, None),
            (total, Some(5)),
            (total, None),
            (total + 10, None),
            (total + 10, Some(1)),
            (usize::MAX, Some(3)),
            (usize::MAX, None),
        ];
        for (skip, fetch) in cases {
            let first = skip.min(total);
            let last = fetch
                .map_or(total, |fetch| skip.saturating_add(fetch))
                .clamp(first, total);
            let expected = (first as i64..last as i64).collect::<Vec<_>>();
            let case = format!("skip {skip}, fetch {fetch:?}");
            // In input order, on any number of threads.
            for threads in [1, 4] {
                let input = source(values_chunks(0, &SIZES));
                let output = run(limit(input, skip, fetch, true), threads).await?;
                assert_eq!(values(&output), expected, "{case}, {threads} threads");
            }
            // In parallel, the threads claim the rows in the order in which
            // they reach the limit: any rows, as many as in order.
            let output = run(
                limit(source(values_chunks(0, &SIZES)), skip, fetch, false),
                1,
            )
            .await?;
            assert_eq!(values(&output), expected, "{case}, 1 thread");
            let output = run(
                limit(source(values_chunks(0, &SIZES)), skip, fetch, false),
                4,
            )
            .await?;
            let mut actual = sorted_values(&output);
            assert_eq!(actual.len(), expected.len(), "{case}, 4 threads");
            actual.dedup();
            assert_eq!(actual.len(), expected.len(), "{case}: duplicate rows");
            assert!(
                actual.iter().all(|v| (0..total as i64).contains(v)),
                "{case}"
            );
            assert!(output.iter().all(|batch| batch.num_rows() > 0));
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn limit_slices_the_logical_rows_of_selected_chunks() -> Result<()> {
        // 50 even values, then 100 values in reverse order.
        let evens = values_chunk(0..100).with_selection(SelectionVector::from(
            (0..100).step_by(2).collect::<Vec<u32>>(),
        ))?;
        let reversed = values_chunk(100..200).with_selection(SelectionVector::from(
            (0..100).rev().collect::<Vec<u32>>(),
        ))?;
        let input = source(vec![evens, reversed]);
        let output = run(limit(input, 30, Some(40), true), 4).await?;
        let expected = (60..100)
            .step_by(2)
            .chain((180..200).rev())
            .collect::<Vec<i64>>();
        assert_eq!(values(&output), expected);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn limit_stops_its_input_early() -> Result<()> {
        // 1000 chunks of 10 rows.
        let chunks = || values_chunks(0, &[10; 1000]);
        for (skip, fetch, chunks_needed) in [(0, 25, 3), (95, 10, 11), (0, 0, 0)] {
            // In order, a single thread reads exactly the chunks it needs.
            let input = Arc::new(TestSource::new(chunks()));
            let output =
                run(limit(Arc::clone(&input) as _, skip, Some(fetch), true), 4).await?;
            let expected = (skip as i64..(skip + fetch) as i64).collect::<Vec<_>>();
            assert_eq!(values(&output), expected);
            // With `fetch == 0`, the first chunk is not needed, but it is
            // read before the limit sees it.
            assert_eq!(input.produced(), chunks_needed.max(1), "skip {skip}");

            // In parallel, threads that were already reading a chunk when the
            // limit finished may still deliver it.
            let input = Arc::new(TestSource::new(chunks()));
            let output =
                run(limit(Arc::clone(&input) as _, skip, Some(fetch), false), 4).await?;
            assert_eq!(values(&output).len(), fetch);
            assert!(
                input.produced() <= chunks_needed + 4,
                "{}",
                input.produced()
            );
        }

        // Without a fetch, all rows after the skipped ones are needed.
        for preserve_order in [true, false] {
            let input = Arc::new(TestSource::new(chunks()));
            let root = limit(Arc::clone(&input) as _, 9995, None, preserve_order);
            let output = run(root, 4).await?;
            if preserve_order {
                assert_eq!(values(&output), (9995..10_000).collect::<Vec<_>>());
            } else {
                // The threads skip any 9995 rows.
                let mut actual = sorted_values(&output);
                actual.dedup();
                assert_eq!(actual.len(), 5);
            }
            assert_eq!(input.produced(), 1000);
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn finished_limit_stops_idle_threads_and_union_inputs() -> Result<()> {
        // Thread 0 never gets any data, thread 1 reaches the limit.
        let input = TestSource::new(values_chunks(0, &[10; 10])).with_idle_threads(1);
        let output = run(limit(Arc::new(input), 0, Some(5), false), 2).await?;
        assert_eq!(sorted_values(&output).len(), 5);

        // The second input of the union never produces anything. In order, it
        // does not even start.
        for (preserve_order, threads) in [(false, 1), (false, 4), (true, 4)] {
            let idle = TestSource::new(vec![]).with_idle_threads(usize::MAX);
            let union = PhysicalUnion::try_new(vec![
                source(values_chunks(0, &[10; 10])),
                Arc::new(idle),
            ])?
            .with_preserve_order(preserve_order);
            let root = limit(Arc::new(union), 3, Some(5), preserve_order);
            let output = run(root, threads).await?;
            if preserve_order {
                assert_eq!(values(&output), (3..8).collect::<Vec<_>>());
            } else {
                assert_eq!(values(&output).len(), 5, "{threads} threads");
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn limit_counts_the_rows_of_all_union_inputs() -> Result<()> {
        let inputs = || {
            vec![
                source(values_chunks(0, &[10; 10])),
                source(values_chunks(100, &[50, 100])),
                source(values_chunks(250, &[25, 25])),
            ]
        };
        // The pipelines of all inputs share the limit.
        let union = Arc::new(PhysicalUnion::try_new(inputs())?);
        let root = limit(union, 0, Some(42), false);
        assert_eq!(
            format_pipelines(&root)?,
            "MetaPipeline 0 (sink: ResultSink)\n  \
             Pipeline 0: TestSource -> PhysicalLimit -> ResultSink\n  \
             Pipeline 1: TestSource -> PhysicalLimit -> ResultSink\n  \
             Pipeline 2: TestSource -> PhysicalLimit -> ResultSink\n"
        );
        let output = run(root, 4).await?;
        let mut actual = sorted_values(&output);
        actual.dedup();
        assert_eq!(actual.len(), 42);

        // Over an ordered union, the inputs come one after the other.
        let union = Arc::new(PhysicalUnion::try_new(inputs())?.with_preserve_order(true));
        let output = run(limit(union, 90, Some(170), true), 4).await?;
        assert_eq!(values(&output), (90..260).collect::<Vec<_>>());
        Ok(())
    }

    #[test]
    fn limit_properties() {
        let input = source(vec![values_chunk(0..3)]);
        let parallel = PhysicalLimit::new(Arc::clone(&input), 5, Some(10));
        assert!(parallel.parallel_operator());
        assert!(!parallel.preserve_order());
        assert_eq!((parallel.skip(), parallel.fetch()), (5, Some(10)));
        assert!(Arc::ptr_eq(parallel.input(), &input));
        assert_eq!(parallel.schema(), input.schema());
        assert_eq!(parallel.description(), "PhysicalLimit: skip=5, fetch=10");

        let ordered = PhysicalLimit::new(input, 0, None).with_preserve_order(true);
        assert!(!ordered.parallel_operator());
        assert!(!ordered.is_sink() && !ordered.is_source());
        assert_eq!(
            ordered.description(),
            "PhysicalLimit: skip=0, fetch=None, preserve_order=true"
        );
    }

    // ---------------------------------------------------------------------
    // Lowering
    // ---------------------------------------------------------------------

    /// `t1` with its 4 skewed partitions, or in a single partition.
    fn t1_scan(single_partition: bool) -> Result<Arc<dyn ExecutionPlan>> {
        let partitions = test_table_t1()?;
        let schema = partitions[0][0].schema();
        let partitions = if single_partition {
            vec![partitions.into_iter().flatten().collect()]
        } else {
            partitions
        };
        Ok(MemorySourceConfig::try_new_exec(&partitions, schema, None)?)
    }

    /// `t1` with all rows in a single batch per partition, sorted by `id`
    /// within each partition.
    fn sorted_t1_scan() -> Result<Arc<dyn ExecutionPlan>> {
        let partitions = test_table_t1()?;
        let schema = partitions[0][0].schema();
        let partitions = partitions
            .iter()
            .map(|batches| Ok(vec![concat_batches(&schema, batches)?]))
            .collect::<Result<Vec<_>>>()?;
        let ordering = id_ordering(&schema)?;
        let source = MemorySourceConfig::try_new(&partitions, schema, None)?
            .try_with_sort_information(vec![ordering])?;
        Ok(datafusion_datasource::source::DataSourceExec::from_data_source(source))
    }

    fn id_ordering(schema: &SchemaRef) -> Result<LexOrdering> {
        Ok(
            LexOrdering::new(vec![PhysicalSortExpr::new_default(col("id", schema)?)])
                .unwrap(),
        )
    }

    /// Lower `plan` with [`PhysicalLimit::try_lower`], bridging its input.
    fn try_lower(plan: Arc<dyn ExecutionPlan>) -> Result<Option<PhysicalLimit>> {
        PhysicalLimit::try_lower(&plan, |input| {
            Ok(Arc::new(BridgeOperator::new(Arc::clone(input))))
        })
    }

    #[test]
    fn lowering_of_limit_nodes() -> Result<()> {
        let lowered = |plan| -> Result<(usize, Option<usize>, bool)> {
            let limit = try_lower(plan)?.expect("a limit");
            Ok((limit.skip(), limit.fetch(), limit.preserve_order()))
        };
        let coalesced = || -> Result<Arc<dyn ExecutionPlan>> {
            Ok(Arc::new(CoalescePartitionsExec::new(t1_scan(false)?)))
        };
        let merged = || -> Result<Arc<dyn ExecutionPlan>> {
            let scan = t1_scan(false)?;
            let ordering = id_ordering(&scan.schema())?;
            let sort =
                SortExec::new(ordering.clone(), scan).with_preserve_partitioning(true);
            Ok(Arc::new(SortPreservingMergeExec::new(
                ordering,
                Arc::new(sort),
            )))
        };

        // The rows a limit over merged partitions returns are arbitrary.
        assert_eq!(
            lowered(Arc::new(GlobalLimitExec::new(coalesced()?, 5, Some(10))))?,
            (5, Some(10), false)
        );
        assert_eq!(
            lowered(Arc::new(
                CoalescePartitionsExec::new(t1_scan(false)?).with_fetch(Some(7))
            ))?,
            (0, Some(7), false)
        );
        // A sorted input, or a single partition, has a deterministic order.
        assert_eq!(
            lowered(Arc::new(GlobalLimitExec::new(merged()?, 3, None)))?,
            (3, None, true)
        );
        assert_eq!(
            lowered(Arc::new(GlobalLimitExec::new(t1_scan(true)?, 0, Some(2))))?,
            (0, Some(2), true)
        );
        assert_eq!(
            lowered(Arc::new(LocalLimitExec::new(t1_scan(true)?, 4)))?,
            (0, Some(4), true)
        );
        assert_eq!(
            lowered(Arc::new(
                CoalescePartitionsExec::new(t1_scan(true)?).with_fetch(Some(9))
            ))?,
            (0, Some(9), true)
        );

        // A limit per partition is not a limit of all partitions...
        for scan in [t1_scan(false)?, sorted_t1_scan()?] {
            assert!(try_lower(Arc::new(LocalLimitExec::new(scan, 4)))?.is_none());
        }
        // ... unless a merge above it keeps at most as many rows: then the
        // native limit replaces both.
        for (local_fetch, absorbed) in [(7, true), (15, true), (6, false)] {
            let local = Arc::new(LocalLimitExec::new(t1_scan(false)?, local_fetch));
            let merged = Arc::new(CoalescePartitionsExec::new(local).with_fetch(Some(7)));
            let limit = try_lower(merged)?.expect("a limit");
            assert_eq!((limit.skip(), limit.fetch()), (0, Some(7)));
            let bridged = limit.input().downcast_ref::<BridgeOperator>().unwrap();
            assert_eq!(
                bridged.plan().is::<LocalLimitExec>(),
                !absorbed,
                "local fetch {local_fetch}"
            );
        }
        // Other nodes, including a coalesce without fetch.
        assert!(try_lower(coalesced()?)?.is_none());
        let mut called = false;
        let scan = t1_scan(false)?;
        assert!(
            PhysicalLimit::try_lower(&scan, |_| {
                called = true;
                internal_err!("not called")
            })?
            .is_none()
        );
        assert!(!called);

        // The lowered input must have the schema of the limit.
        let err = PhysicalLimit::try_lower(
            &(Arc::new(GlobalLimitExec::new(coalesced()?, 0, Some(1))) as _),
            |_| Ok(source(vec![values_chunk(0..1)])),
        )
        .unwrap_err();
        assert!(err.to_string().contains("has schema"), "{err}");
        Ok(())
    }

    // ---------------------------------------------------------------------
    // Differential tests
    // ---------------------------------------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_limit_and_offset() {
        for sql in [
            // A parallel limit (over merged partitions), with constant rows so
            // that the result does not depend on the rows selected.
            "SELECT 1 AS one, 'x' AS x FROM t1 LIMIT 10 OFFSET 5",
            "SELECT 'x' AS x FROM t1 OFFSET 4990",
            "SELECT 'x' AS x FROM t1 WHERE a > 0 LIMIT 3000 OFFSET 7",
            // Limits that keep all rows.
            "SELECT * FROM t1 LIMIT 6000",
            "SELECT * FROM t1 LIMIT 6000 OFFSET 0",
            "SELECT * FROM t2 LIMIT 150",
            // Limits in input order (a sorted input).
            "SELECT id, d FROM t1 ORDER BY id LIMIT 10 OFFSET 5",
            "SELECT * FROM t1 ORDER BY id LIMIT 3000 OFFSET 1000",
            "SELECT id, h FROM t1 ORDER BY id DESC OFFSET 4990",
            "SELECT id, c FROM t1 ORDER BY c, id LIMIT 2049 OFFSET 2047",
            "SELECT * FROM t1 ORDER BY id LIMIT 10 OFFSET 4995",
            "SELECT a FROM (SELECT a, id FROM t1 ORDER BY id LIMIT 20) LIMIT 5 OFFSET 2",
            // A `LocalLimitExec` per partition of an aggregate (that has fewer
            // groups than the limit) below a merge with the same fetch: the
            // native limit replaces both.
            "SELECT DISTINCT d FROM t1 LIMIT 100",
            "SELECT d, COUNT(*) AS n FROM t1 GROUP BY d LIMIT 50",
        ] {
            let root = assert_parity(sql).await;
            assert!(count::<PhysicalLimit>(&root) > 0, "{sql}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_empty_limits() {
        for sql in [
            // DataFusion plans an empty relation for these.
            "SELECT * FROM t1 LIMIT 0",
            "SELECT * FROM t1 ORDER BY id LIMIT 0",
            "SELECT * FROM t1 LIMIT 0 OFFSET 10",
            // OFFSET reaching or exceeding the number of rows.
            "SELECT * FROM t1 LIMIT 10 OFFSET 5000",
            "SELECT * FROM t1 OFFSET 6000",
            "SELECT * FROM t1 ORDER BY id LIMIT 5 OFFSET 6000",
            "SELECT id FROM t1 ORDER BY id OFFSET 5000",
            "SELECT * FROM t_empty LIMIT 5 OFFSET 2",
            "SELECT * FROM t_empty ORDER BY id OFFSET 1",
        ] {
            assert_parity(sql).await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_limits_over_unions() {
        for sql in [
            // The fetch of the limit is pushed into the inputs of the union.
            "SELECT 'same' AS s FROM t2 UNION ALL SELECT 'same' FROM t1 LIMIT 10 OFFSET 3",
            "SELECT 'same' AS s FROM t2 UNION ALL SELECT 'same' FROM t1 LIMIT 7",
            "SELECT k FROM t2 UNION ALL SELECT a FROM t1 LIMIT 10000",
            "SELECT k FROM t2 UNION ALL SELECT a FROM t1 LIMIT 10 OFFSET 5150",
        ] {
            let root = assert_parity(sql).await;
            assert!(count::<PhysicalLimit>(&root) > 0, "{sql}");
            assert_eq!(count::<PhysicalUnion>(&root), 1, "{sql}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_limits_with_other_partitionings() {
        for target_partitions in [1, 3, 16] {
            let config = test_session_config().with_target_partitions(target_partitions);
            for sql in [
                "SELECT 1 AS one FROM t1 LIMIT 10 OFFSET 5",
                "SELECT id, d FROM t1 ORDER BY id LIMIT 100 OFFSET 2040",
                "SELECT * FROM t1 OFFSET 6000",
            ] {
                assert_parity_with_config(sql, config.clone()).await;
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_limit_plans() -> Result<()> {
        // Limits that SQL does not produce: DataFusion plans an empty relation
        // for a `LIMIT 0`.
        let make = |skip, fetch| {
            move || -> Result<Arc<dyn ExecutionPlan>> {
                Ok(Arc::new(GlobalLimitExec::new(
                    Arc::new(CoalescePartitionsExec::new(t1_scan(false)?)),
                    skip,
                    fetch,
                )))
            }
        };
        for (skip, fetch) in [(0, Some(0)), (7, Some(0)), (0, Some(6000)), (0, None)] {
            let root = assert_native_plan_byte_eq(make(skip, fetch), false).await;
            assert_eq!(count::<PhysicalLimit>(&root), 1);
        }
        let local = || -> Result<Arc<dyn ExecutionPlan>> {
            Ok(Arc::new(LocalLimitExec::new(t1_scan(false)?, 0)))
        };
        assert_native_plan_byte_eq(local, false).await;

        // The order of a single partition is deterministic: the limits return
        // the same rows as DataFusion, in the same order.
        for (skip, fetch) in [
            (2500, Some(100)),
            (4000, None),
            (0, Some(2049)),
            (4999, Some(7)),
        ] {
            let make = move || -> Result<Arc<dyn ExecutionPlan>> {
                Ok(Arc::new(GlobalLimitExec::new(t1_scan(true)?, skip, fetch)))
            };
            assert_native_plan_byte_eq(make, true).await;
        }
        let local = || -> Result<Arc<dyn ExecutionPlan>> {
            Ok(Arc::new(LocalLimitExec::new(t1_scan(true)?, 3000)))
        };
        assert_native_plan_byte_eq(local, true).await;
        Ok(())
    }

    /// Two partitions of 10 rows: the values `0..10` and `100..110`, or all
    /// `7` if `constant`.
    fn two_partitions(constant: bool) -> Result<Arc<dyn ExecutionPlan>> {
        let partition = |start: i64| {
            let values = if constant {
                values_chunk([7; 10])
            } else {
                values_chunk(start..start + 10)
            };
            vec![values.into_record_batch().unwrap()]
        };
        Ok(MemorySourceConfig::try_new_exec(
            &[partition(0), partition(100)],
            values_schema(),
            None,
        )?)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_local_limits_keep_their_rows_per_partition() -> Result<()> {
        // A limit per partition: 3 rows of each partition.
        let local = || -> Result<Arc<dyn ExecutionPlan>> {
            Ok(Arc::new(LocalLimitExec::new(two_partitions(false)?, 3)))
        };
        let root = assert_native_plan_byte_eq(local, false).await;
        assert_eq!(count::<PhysicalLimit>(&root), 0);

        // Below a merge that keeps more rows than the limit of a partition, the
        // limit per partition decides which rows are merged.
        let merged = || -> Result<Arc<dyn ExecutionPlan>> {
            let local = Arc::new(LocalLimitExec::new(two_partitions(false)?, 3));
            Ok(Arc::new(
                CoalescePartitionsExec::new(local).with_fetch(Some(6)),
            ))
        };
        let root = assert_native_plan_byte_eq(merged, false).await;
        assert_eq!(count::<PhysicalLimit>(&root), 1);
        assert!(bridged_plans(&root)[0].is::<LocalLimitExec>());
        let offset = || -> Result<Arc<dyn ExecutionPlan>> {
            let local = Arc::new(LocalLimitExec::new(two_partitions(true)?, 3));
            let merged = CoalescePartitionsExec::new(local).with_fetch(Some(5));
            Ok(Arc::new(GlobalLimitExec::new(Arc::new(merged), 1, Some(4))))
        };
        let root = assert_native_plan_byte_eq(offset, false).await;
        assert_eq!(count::<PhysicalLimit>(&root), 2);
        assert!(bridged_plans(&root)[0].is::<LocalLimitExec>());

        // A merge that keeps at most as many rows as the limit of a partition
        // makes that limit redundant: the native limit replaces both.
        let absorbed = || -> Result<Arc<dyn ExecutionPlan>> {
            let local = Arc::new(LocalLimitExec::new(two_partitions(true)?, 3));
            Ok(Arc::new(
                CoalescePartitionsExec::new(local).with_fetch(Some(2)),
            ))
        };
        let root = assert_native_plan_byte_eq(absorbed, false).await;
        assert_eq!(count::<PhysicalLimit>(&root), 1);
        assert!(
            bridged_plans(&root)[0].is::<datafusion_datasource::source::DataSourceExec>()
        );
        Ok(())
    }
}
