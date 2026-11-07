// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! `PhysicalLimit`: `OFFSET` / `LIMIT` with early pipeline termination.
//!
//! [`PhysicalLimit`] skips the first `skip` rows of its input and emits at
//! most `fetch` of the rows after them, like DataFusion's `GlobalLimitExec`
//! (`LocalLimitExec` is a limit with `skip == 0`).
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
    /// * `LocalLimitExec` (a `fetch` per input partition). The pipeline
    ///   engine has no partitions, so it limits its whole input to `fetch`
    ///   rows. That is equivalent, because DataFusion only plans a
    ///   `LocalLimitExec` below a node that merges its partitions and keeps
    ///   at most `fetch` rows of them. It is not lowered (`Ok(None)`) if its
    ///   input has several sorted partitions, whose rows such a merge may
    ///   choose by their order within each partition;
    /// * `CoalescePartitionsExec` with a `fetch`: the engine merges the
    ///   partitions of its input anyway, what remains is the limit.
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
        let (input, skip, fetch, preserve_order) = if let Some(limit) =
            plan.downcast_ref::<GlobalLimitExec>()
        {
            let input = limit.input();
            (input, limit.skip(), limit.fetch(), order_matters(input))
        } else if let Some(limit) = plan.downcast_ref::<LocalLimitExec>() {
            let input = limit.input();
            let preserve_order = if input.output_partitioning().partition_count() > 1 {
                if input.output_ordering().is_some() {
                    return Ok(None);
                }
                false
            } else {
                order_matters(input)
            };
            (input, 0, Some(limit.fetch()), preserve_order)
        } else if let Some(coalesce) = plan.downcast_ref::<CoalescePartitionsExec>()
            && let Some(fetch) = coalesce.fetch()
        {
            let input = coalesce.input();
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
