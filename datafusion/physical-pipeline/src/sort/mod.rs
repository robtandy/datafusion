// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! `PhysicalSort` and `PhysicalTopK`: parallel sort and top-k.
//!
//! [`PhysicalSort`] (`ORDER BY`) and [`PhysicalTopK`] (`ORDER BY ... LIMIT
//! k`) replace DataFusion's `SortExec`, together with the
//! `SortPreservingMergeExec` or `CoalescePartitionsExec` that merges its
//! partitions. Like DuckDB's `PhysicalOrder` and `PhysicalTopN`, both are
//! pipeline breakers that sort in parallel and merge once:
//!
//! * **Sink**: every thread of the input pipeline collects the chunks it
//!   pushes in its local sink state (compacting their selection vectors).
//!   [`PhysicalSort`] sorts them into *runs* of bounded size
//!   (`lexsort_to_indices` and `take` for a single sort key of a primitive,
//!   boolean, string or binary type, the Arrow row format otherwise).
//!   [`PhysicalTopK`] keeps the best `k` rows of the thread instead and
//!   discards the rows that cannot make it into the result as they arrive,
//!   see [`topk_sink`].
//! * **Combine**: every thread hands its sorted runs to the global sink state.
//! * **Finalize**: a [`RunMerger`] is set up over the runs of all threads.
//! * **Source**: the sort is the source of the pipeline that consumes its
//!   output. The merger merges the runs with a tree of losers (see [`merge`])
//!   and emits the rows in order, in chunks of at most
//!   [`DEFAULT_VECTOR_SIZE`] rows, stopping after `fetch` rows. The source is
//!   not parallel: the pipeline that reads the sorted output runs on a single
//!   thread, which keeps the order, as DataFusion reads the single partition
//!   of the merge.
//!
//! The sort keys of the runs are encoded in the Arrow row format, in which
//! comparing two keys is a `memcmp`, so the merge works for every sort key
//! the row format supports (sorts of other keys are left to DataFusion).
//!
//! # Lowering
//!
//! [`try_lower`] (and [`PhysicalSort::try_lower`] /
//! [`PhysicalTopK::try_lower`]) lowers
//!
//! * a `SortExec` with a single output partition (over an input with a
//!   single partition, e.g. a `CoalescePartitionsExec`, which the native sort
//!   replaces: it sorts all partitions of its input);
//! * a `SortPreservingMergeExec` over a `SortExec` that sorts every
//!   partition, possibly through `ProjectionExec`s (DataFusion plans the
//!   projection of `GROUP BY ... ORDER BY` between them) and merges of the
//!   single partition below them (as for `SELECT b FROM t ORDER BY a LIMIT
//!   k`). The native sort sorts all rows of the input of the `SortExec` by
//!   the order of the top merge, above the projections, which are lowered
//!   with the input.
//!
//! The result keeps `fetch` rows if the `SortExec` or the merge has a fetch:
//! that is a [`PhysicalTopK`], and the [`DynamicFilterPhysicalExpr`] of the
//! `SortExec` (which DataFusion pushes into the scans below it) is updated by
//! the native operator. Nodes whose rows the native sort cannot reproduce
//! exactly are not lowered, so that DataFusion executes them, see
//! [`try_lower`].
//!
//! The native operators keep all rows in memory: they do not spill and do
//! not reserve memory in the memory pool. As in DataFusion, rows with equal
//! sort keys may come in any order.
//!
//! [`DynamicFilterPhysicalExpr`]: datafusion_physical_expr::expressions::DynamicFilterPhysicalExpr

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, RecordBatch, RecordBatchOptions, UInt32Array};
use arrow::compute::{SortColumn, concat, lexsort_to_indices, take, take_arrays};
use arrow::datatypes::{DataType, Schema, SchemaRef};
use arrow_row::{RowConverter, SortField};
use arrow_select::interleave::interleave;
use datafusion_common::{Result, exec_datafusion_err, internal_err};
use datafusion_physical_expr::LexOrdering;
use datafusion_physical_expr::expressions::{Column, DynamicFilterPhysicalExpr, Literal};
use datafusion_physical_expr::projection::update_expr;
use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::sorts::sort::SortExec;
use datafusion_physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion_physical_plan::{
    ChildrenPropertiesMode, ExecutionPlan, ExecutionPlanProperties,
    ReplaceChildrenOptions,
};
use parking_lot::Mutex;

use crate::chunk::{DEFAULT_VECTOR_SIZE, DataChunk};
use crate::operator::{
    GlobalSourceState, PhysicalOperator, SourceResult, downcast_state,
};

pub mod merge;
pub mod sort_sink;
pub mod topk_sink;

pub use merge::{RunMerger, SortedRun};
pub use sort_sink::PhysicalSort;
pub use topk_sink::{PhysicalTopK, TopKDynamicFilter};

/// Lower a sort of a DataFusion plan to a [`PhysicalSort`], or to a
/// [`PhysicalTopK`] if only the first rows of the sorted output are kept.
///
/// The lowered nodes are
///
/// * a `SortExec` with a single output partition: either it sorts every
///   partition of an input with a single partition, or it sorts all of its
///   input (`preserve_partitioning = false`), which then has a single
///   partition, as DataFusion executes only the first partition of its
///   input. If that input is a `CoalescePartitionsExec` without a fetch, the
///   native sort replaces it too and sorts all partitions of its input;
/// * a `SortPreservingMergeExec` over a `SortExec` with the same number of
///   partitions, possibly through `ProjectionExec`s and further
///   `SortPreservingMergeExec`s, all of which (but the lowest merge, which
///   merges the partitions of the sort) have a single partition. The input of
///   every merge must be ordered by the order of the merge, so that the top
///   merge produces all rows of the input of the `SortExec` in its order, as
///   the native sort does (above the projections, which are lowered with the
///   input).
///
/// A fetch of the `SortExec` (a TopK of every partition) and the fetches of
/// the merges make a [`PhysicalTopK`] of the smallest one, if that is a top-k
/// of the whole input: the merges must not keep more rows than the fetch of a
/// sort of several partitions. The [`DynamicFilterPhysicalExpr`] of the
/// `SortExec` is updated by the native operator, for its leading sort keys
/// that are (through the projections) the leading sort keys of the
/// `SortExec`.
///
/// Sorts whose rows the native operators cannot reproduce exactly are not
/// lowered (`Ok(None)`), so that DataFusion executes them:
///
/// * sorts of no rows (`fetch == Some(0)`), which DataFusion ends without
///   reading their input;
/// * sorts whose input is already sorted (DataFusion passes the input
///   through, with a limit, so it may not read all of it), and TopKs whose
///   input is sorted by a prefix of the sort keys, which DataFusion ends as
///   soon as no later row can make it into the result;
/// * projections that compute a value above a sort or a merge with a fetch:
///   DataFusion evaluates them on the rows that are kept only, the native
///   operators on all rows (and a computation can fail);
/// * plans that DataFusion does not execute as a sort of all rows of the
///   input: nodes that advertise a different number of partitions than their
///   input has, merges of no partitions, and sorts of several partitions that
///   are not merged;
/// * sort keys that the Arrow row format does not support.
///
/// `lower_input` lowers the input of the sort (with the projections above
/// it). Returns `Ok(None)` for any other node, without calling
/// `lower_input`.
///
/// [`DynamicFilterPhysicalExpr`]: datafusion_physical_expr::expressions::DynamicFilterPhysicalExpr
pub fn try_lower<F>(
    plan: &Arc<dyn ExecutionPlan>,
    lower_input: F,
) -> Result<Option<Arc<dyn PhysicalOperator>>>
where
    F: FnMut(&Arc<dyn ExecutionPlan>) -> Result<Arc<dyn PhysicalOperator>>,
{
    let Some(sort) = SortPlan::try_new(plan)? else {
        return Ok(None);
    };
    let operator: Arc<dyn PhysicalOperator> = match sort.fetch {
        None => Arc::new(sort.into_sort(plan, lower_input)?),
        Some(_) => Arc::new(sort.into_topk(plan, lower_input)?),
    };
    Ok(Some(operator))
}

/// A sort of a DataFusion plan that the native operators reproduce, see
/// [`try_lower`].
#[derive(Debug)]
pub(crate) struct SortPlan {
    /// The sort keys of the native sort, over `input`.
    ordering: LexOrdering,
    /// The number of rows to keep.
    fetch: Option<usize>,
    /// The rows to sort: the input of the `SortExec` (or the input of the
    /// `CoalescePartitionsExec` it sorts), with the projections between the
    /// top merge and the `SortExec` above it.
    input: Arc<dyn ExecutionPlan>,
    /// The dynamic filter of the `SortExec`, if the native operator can
    /// update it: with the expressions of the `SortExec`'s input that its
    /// leading sort keys correspond to.
    dynamic_filter: Option<TopKDynamicFilter>,
}

impl SortPlan {
    /// Analyze `plan`, see [`try_lower`].
    pub(crate) fn try_new(plan: &Arc<dyn ExecutionPlan>) -> Result<Option<Self>> {
        let partitions =
            |plan: &Arc<dyn ExecutionPlan>| plan.output_partitioning().partition_count();
        // The merges and the projections between them, from the top, down to
        // the sort.
        let mut chain = vec![];
        let mut node = plan;
        while node.is::<SortPreservingMergeExec>()
            || (!chain.is_empty() && node.is::<ProjectionExec>())
        {
            chain.push(node);
            node = node.children()[0];
        }
        let Some(sort) = node.downcast_ref::<SortExec>() else {
            return Ok(None);
        };
        let sort_input = sort.input();
        let merges = chain
            .iter()
            .filter_map(|node| node.downcast_ref::<SortPreservingMergeExec>())
            .collect::<Vec<_>>();

        // DataFusion executes the partitions that a node advertises: they must
        // be the partitions of its input, all of which the native sort reads.
        // A sort of all its input executes the first partition of its input,
        // and a merge of no partitions fails.
        let sorted_partitions = partitions(node);
        if sort.preserve_partitioning() {
            if sorted_partitions != partitions(sort_input) {
                return Ok(None);
            }
        } else if sorted_partitions != 1 || partitions(sort_input) != 1 {
            return Ok(None);
        }
        let consistent = chain.iter().all(|node| {
            let input_partitions = partitions(node.children()[0]);
            if node.is::<SortPreservingMergeExec>() {
                input_partitions > 0
            } else {
                partitions(node) == input_partitions
            }
        });
        // Several sorted partitions are only one sorted output when merged.
        if !consistent || (merges.is_empty() && sorted_partitions != 1) {
            return Ok(None);
        }

        // Only the lowest merge merges several partitions (those of the sort);
        // the merges above it limit a single sorted partition.
        let merge_fetch = merges.iter().filter_map(|merge| merge.fetch()).min();
        let fetch = match (sort.fetch(), merge_fetch) {
            (None, merge_fetch) => merge_fetch,
            // The top rows of a single partition are the top rows.
            (Some(sort_fetch), merge_fetch) if sorted_partitions == 1 => {
                Some(merge_fetch.map_or(sort_fetch, |fetch| fetch.min(sort_fetch)))
            }
            // The first `merge_fetch` rows of the top `sort_fetch` rows of every
            // partition are the top `merge_fetch` rows, if `merge_fetch <=
            // sort_fetch`.
            (Some(sort_fetch), Some(merge_fetch)) if merge_fetch <= sort_fetch => {
                Some(merge_fetch)
            }
            _ => return Ok(None),
        };
        if fetch == Some(0) {
            return Ok(None);
        }

        // DataFusion passes a sorted input through (with a limit), and ends a
        // TopK whose input is sorted by a prefix of the keys early: they may
        // not read all of their input, the native operators do.
        let input_properties = sort_input.equivalence_properties();
        if input_properties.ordering_satisfy(sort.expr().clone())? {
            return Ok(None);
        }
        if sort.fetch().is_some()
            && !input_properties
                .extract_common_sort_prefix(sort.expr().clone())?
                .0
                .is_empty()
        {
            return Ok(None);
        }

        // A merge produces all rows in its order only if every partition of
        // its input is ordered by it. The native sort sorts by the order of the
        // top merge.
        for merge in &merges {
            if !merge
                .input()
                .equivalence_properties()
                .ordering_satisfy(merge.expr().clone())?
            {
                return Ok(None);
            }
        }
        let ordering = match merges.first() {
            Some(merge) => merge.expr().clone(),
            None => sort.expr().clone(),
        };

        // DataFusion evaluates a projection above a TopK or a merge with a
        // fetch on the rows they keep only; the native operator evaluates the
        // projections on all rows, which makes no difference for projections
        // that cannot fail.
        let mut limited = sort.fetch().is_some();
        for node in chain.iter().rev() {
            if let Some(merge) = node.downcast_ref::<SortPreservingMergeExec>() {
                limited |= merge.fetch().is_some();
            } else if let Some(projection) = node.downcast_ref::<ProjectionExec>()
                && limited
                && projection
                    .expr()
                    .iter()
                    .any(|expr| !expr.expr.is::<Column>() && !expr.expr.is::<Literal>())
            {
                return Ok(None);
            }
        }

        // The rows to sort: the input of the sort, without a merge of its
        // partitions, with the projections above it.
        let mut input = Arc::clone(sort_input);
        if let Some(coalesce) = sort_input.downcast_ref::<CoalescePartitionsExec>()
            && coalesce.fetch().is_none()
            && partitions(coalesce.input()) > 0
        {
            input = Arc::clone(coalesce.input());
        }
        let projections = chain
            .iter()
            .filter(|node| node.is::<ProjectionExec>())
            .collect::<Vec<_>>();
        for projection in projections.iter().rev() {
            input = Arc::clone(projection).replace_children(
                vec![input],
                ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
            )?;
        }
        if !SortKeys::supports(&ordering, &input.schema())? {
            return Ok(None);
        }

        let dynamic_filter = match fetch {
            Some(_) => dynamic_filter(sort, &ordering, &projections)?,
            None => None,
        };
        Ok(Some(Self {
            ordering,
            fetch,
            input,
            dynamic_filter,
        }))
    }

    /// The number of rows the sort keeps, `None` for all of them.
    pub(crate) fn fetch(&self) -> Option<usize> {
        self.fetch
    }

    /// Lower the sort of `plan` to a [`PhysicalSort`].
    pub(crate) fn into_sort<F>(
        self,
        plan: &Arc<dyn ExecutionPlan>,
        lower_input: F,
    ) -> Result<PhysicalSort>
    where
        F: FnMut(&Arc<dyn ExecutionPlan>) -> Result<Arc<dyn PhysicalOperator>>,
    {
        let input = self.lower_input(plan, lower_input)?;
        PhysicalSort::try_new(input, self.ordering, self.fetch)
    }

    /// Lower the sort of `plan`, which has a fetch, to a [`PhysicalTopK`].
    pub(crate) fn into_topk<F>(
        self,
        plan: &Arc<dyn ExecutionPlan>,
        lower_input: F,
    ) -> Result<PhysicalTopK>
    where
        F: FnMut(&Arc<dyn ExecutionPlan>) -> Result<Arc<dyn PhysicalOperator>>,
    {
        let Some(fetch) = self.fetch else {
            return internal_err!("A sort without fetch is not a TopK");
        };
        let input = self.lower_input(plan, lower_input)?;
        let topk = PhysicalTopK::try_new(input, self.ordering, fetch)?;
        Ok(match self.dynamic_filter {
            Some(filter) => topk.with_dynamic_filter(filter)?,
            None => topk,
        })
    }

    fn lower_input<F>(
        &self,
        plan: &Arc<dyn ExecutionPlan>,
        mut lower_input: F,
    ) -> Result<Arc<dyn PhysicalOperator>>
    where
        F: FnMut(&Arc<dyn ExecutionPlan>) -> Result<Arc<dyn PhysicalOperator>>,
    {
        let input = lower_input(&self.input)?;
        if input.schema() != plan.schema() {
            return internal_err!(
                "The lowered input {} of {} has schema {:?}, expected {:?}",
                input.name(),
                plan.name(),
                input.schema(),
                plan.schema()
            );
        }
        Ok(input)
    }
}

/// The dynamic filter of `sort`, a TopK, for a native TopK with sort keys
/// `ordering` over the output of `projections` (from the top) above `sort`:
/// the keys of `ordering`, mapped through the projections, that are the
/// leading keys of `sort`, from the first one.
fn dynamic_filter(
    sort: &SortExec,
    ordering: &LexOrdering,
    projections: &[&&Arc<dyn ExecutionPlan>],
) -> Result<Option<TopKDynamicFilter>> {
    let Some(filter) = sort
        .dynamic_expressions_produced()
        .into_iter()
        .find(|expr| expr.is::<DynamicFilterPhysicalExpr>())
    else {
        return Ok(None);
    };
    let Ok(filter) = (filter as Arc<dyn std::any::Any + Send + Sync>)
        .downcast::<DynamicFilterPhysicalExpr>()
    else {
        return Ok(None);
    };
    let mut keys = vec![];
    for (sort_expr, sort_key) in ordering.iter().zip(sort.expr().iter()) {
        let mut expr = Some(Arc::clone(&sort_expr.expr));
        for projection in projections {
            let Some(projection) = projection.downcast_ref::<ProjectionExec>() else {
                return Ok(None);
            };
            expr = match expr {
                Some(expr) => update_expr(&expr, projection.expr(), true)?,
                None => None,
            };
        }
        match expr {
            Some(expr)
                if expr.as_ref() == sort_key.expr.as_ref()
                    && sort_expr.options == sort_key.options =>
            {
                keys.push(Arc::clone(&sort_key.expr));
            }
            _ => break,
        }
    }
    if keys.is_empty() {
        return Ok(None);
    }
    Ok(Some(TopKDynamicFilter::new(filter, keys)))
}

/// The sort keys of a native sort: evaluates the sort expressions and sorts
/// rows into [`SortedRun`]s, whose keys it encodes in the Arrow row format
/// (with the same [`RowConverter`] for all runs, so that they can be
/// merged).
#[derive(Debug)]
pub(crate) struct SortKeys {
    ordering: LexOrdering,
    converter: RowConverter,
    /// Sort with `lexsort_to_indices` rather than by the encoded keys: a
    /// single sort key of a type that `sort_to_indices` orders exactly like
    /// the row format (primitive, boolean, string and binary types).
    sort_with_kernel: bool,
}

impl SortKeys {
    /// The sort keys `ordering` of rows with schema `schema`.
    pub(crate) fn try_new(ordering: LexOrdering, schema: &Schema) -> Result<Self> {
        let types = data_types(&ordering, schema)?;
        let fields = ordering
            .iter()
            .zip(&types)
            .map(|(sort_expr, data_type)| {
                SortField::new_with_options(data_type.clone(), sort_expr.options)
            })
            .collect::<Vec<_>>();
        let converter = RowConverter::new(fields)?;
        let sort_with_kernel =
            matches!(types.as_slice(), [data_type] if sorts_like_rows(data_type));
        Ok(Self {
            ordering,
            converter,
            sort_with_kernel,
        })
    }

    /// Whether the row format supports the sort keys `ordering` of rows with
    /// schema `schema`.
    pub(crate) fn supports(ordering: &LexOrdering, schema: &Schema) -> Result<bool> {
        let fields = data_types(ordering, schema)?
            .into_iter()
            .map(SortField::new)
            .collect::<Vec<_>>();
        Ok(RowConverter::supports_fields(&fields))
    }

    /// The sort keys.
    pub(crate) fn ordering(&self) -> &LexOrdering {
        &self.ordering
    }

    /// The values of the sort keys of the rows of `batch`.
    pub(crate) fn evaluate(&self, batch: &RecordBatch) -> Result<Vec<ArrayRef>> {
        self.ordering
            .iter()
            .map(|sort_expr| sort_expr.expr.evaluate(batch)?.into_array(batch.num_rows()))
            .collect()
    }

    /// Sort the rows of `batches` into a run with schema `schema`, keeping
    /// only the first `limit` rows (all rows if `None`). Returns `None` if
    /// there are no rows to keep.
    pub(crate) fn sort(
        &self,
        schema: &SchemaRef,
        batches: &[RecordBatch],
        limit: Option<usize>,
    ) -> Result<Option<SortedRun>> {
        let batches = batches
            .iter()
            .filter(|batch| batch.num_rows() > 0)
            .collect::<Vec<_>>();
        let num_rows = batches.iter().map(|batch| batch.num_rows()).sum::<usize>();
        let len = limit.map_or(num_rows, |limit| limit.min(num_rows));
        if len == 0 {
            return Ok(None);
        }
        let num_rows_u32 = u32::try_from(num_rows)
            .map_err(|_| exec_datafusion_err!("Cannot sort {num_rows} rows at once"))?;
        let limit = (len < num_rows).then_some(len);
        let keys = batches
            .iter()
            .map(|batch| self.evaluate(batch))
            .collect::<Result<Vec<_>>>()?;

        // The positions of the sorted rows in the concatenation of `batches`,
        // and the encoded keys of the sorted rows.
        let (indices, sorted_keys) = if self.sort_with_kernel {
            let key = match keys.as_slice() {
                [keys] => Arc::clone(&keys[0]),
                _ => {
                    concat(&keys.iter().map(|keys| keys[0].as_ref()).collect::<Vec<_>>())?
                }
            };
            let column = SortColumn {
                values: Arc::clone(&key),
                options: Some(self.ordering[0].options),
            };
            let indices = lexsort_to_indices(&[column], limit)?;
            let sorted = take(key.as_ref(), &indices, None)?;
            let sorted_keys = self.converter.convert_columns(&[sorted])?;
            (indices, sorted_keys)
        } else {
            let mut rows = self.converter.empty_rows(num_rows, 0);
            for keys in &keys {
                self.converter.append(&mut rows, keys)?;
            }
            let mut order = (0..num_rows_u32).collect::<Vec<_>>();
            let compare =
                |a: &u32, b: &u32| rows.row(*a as usize).cmp(&rows.row(*b as usize));
            if len < num_rows {
                order.select_nth_unstable_by(len - 1, compare);
                order.truncate(len);
            }
            order.sort_unstable_by(compare);
            let data_len = order.iter().map(|&i| rows.row_len(i as usize)).sum();
            let mut sorted_keys = self.converter.empty_rows(len, data_len);
            for &i in &order {
                sorted_keys.push(rows.row(i as usize));
            }
            (UInt32Array::from(order), sorted_keys)
        };

        let columns = match batches.as_slice() {
            [batch] => take_arrays(batch.columns(), &indices, None)?,
            _ => {
                // The batch and row of every sorted row.
                let mut starts = Vec::with_capacity(batches.len());
                let mut start = 0;
                for batch in &batches {
                    starts.push(start);
                    start += batch.num_rows();
                }
                let positions = indices
                    .values()
                    .iter()
                    .map(|&index| {
                        let index = index as usize;
                        let batch = starts.partition_point(|&start| start <= index) - 1;
                        (batch, index - starts[batch])
                    })
                    .collect::<Vec<_>>();
                (0..schema.fields().len())
                    .map(|column| {
                        let values = batches
                            .iter()
                            .map(|batch| batch.column(column).as_ref())
                            .collect::<Vec<&dyn Array>>();
                        Ok(interleave(&values, &positions)?)
                    })
                    .collect::<Result<Vec<_>>>()?
            }
        };
        let options = RecordBatchOptions::new().with_row_count(Some(len));
        let batch =
            RecordBatch::try_new_with_options(Arc::clone(schema), columns, &options)?;
        Ok(Some(SortedRun::try_new(batch, sorted_keys)?))
    }
}

/// The data types of the sort keys `ordering` of rows with schema `schema`.
fn data_types(ordering: &LexOrdering, schema: &Schema) -> Result<Vec<DataType>> {
    ordering
        .iter()
        .map(|sort_expr| sort_expr.expr.data_type(schema))
        .collect()
}

/// Whether `sort_to_indices` sorts values of `data_type` exactly like the row
/// format orders them (types without children and dictionaries).
fn sorts_like_rows(data_type: &DataType) -> bool {
    data_type.is_primitive()
        || matches!(
            data_type,
            DataType::Boolean
                | DataType::Utf8
                | DataType::LargeUtf8
                | DataType::Utf8View
                | DataType::Binary
                | DataType::LargeBinary
                | DataType::BinaryView
                | DataType::FixedSizeBinary(_)
        )
}

/// The sorted output of a [`PhysicalSort`] or [`PhysicalTopK`], set up when
/// the sink is finalized and handed to the source.
#[derive(Debug, Default)]
pub(crate) struct SortedOutput {
    merger: Mutex<Option<RunMerger>>,
}

impl SortedOutput {
    /// Set the merger of the runs collected by the sink.
    pub(crate) fn set(&self, merger: RunMerger) {
        *self.merger.lock() = Some(merger);
    }

    /// The global source state of the operator `name`: the merger, which
    /// the source takes over.
    pub(crate) fn take_source(&self, name: &str) -> Result<Arc<dyn GlobalSourceState>> {
        match self.merger.lock().take() {
            Some(merger) => Ok(Arc::new(MergeSource {
                merger: Mutex::new(merger),
            })),
            None => internal_err!("{name} is read before its sink is finalized"),
        }
    }
}

/// The global source state of a [`PhysicalSort`] or [`PhysicalTopK`]. The
/// source is not parallel, so a single thread reads the merger.
#[derive(Debug)]
struct MergeSource {
    merger: Mutex<RunMerger>,
}

impl GlobalSourceState for MergeSource {}

/// The next chunk of the sorted output of the source state `gstate` of the
/// operator `name`.
pub(crate) fn next_sorted_chunk(
    gstate: &dyn GlobalSourceState,
    name: &str,
) -> Result<SourceResult> {
    let source = downcast_state::<MergeSource>(gstate.as_any(), name)?;
    let batch = source.merger.lock().next_batch(DEFAULT_VECTOR_SIZE)?;
    Ok(match batch {
        Some(batch) => SourceResult::Chunk(DataChunk::from_record_batch(batch)),
        None => SourceResult::Finished,
    })
}

/// Test support: a lowering with the native sorts, standing in for the
/// planner, and differential checks of SQL queries lowered with it.
#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::Arc;
    use std::time::Duration;

    use arrow::array::{RecordBatch, RecordBatchOptions};
    use arrow::datatypes::SchemaRef;
    use datafusion::prelude::{SessionConfig, SessionContext};
    use datafusion_common::Result;
    use datafusion_physical_expr::PhysicalExpr;
    use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
    use datafusion_physical_plan::projection::ProjectionExec;
    use datafusion_physical_plan::repartition::RepartitionExec;
    use datafusion_physical_plan::{ExecutionPlan, ExecutionPlanProperties, collect};

    use crate::bridge::{BridgeOperator, PipelineRootExec};
    use crate::chunk::DataChunk;
    use crate::explain::format_operator_tree;
    use crate::operator::{
        ExecutionContext, GlobalOperatorState, OperatorResult, OperatorState,
        PhysicalOperator,
    };
    use crate::operators::limit::PhysicalLimit;
    use crate::operators::placeholder_row::PhysicalPlaceholderRow;
    use crate::operators::union::PhysicalUnion;
    use crate::sources::try_create_scan_source;
    use crate::test_utils::{
        EngineOutput, assert_byte_eq_with_datafusion_with_config, compare_outputs,
        datafusion_context, register_test_tables, run_sql, test_session_config,
    };

    /// Lower `plan` with the native operators, standing in for the planner
    /// until it lowers sorts (#10):
    ///
    /// * sorts with [`try_lower`](super::try_lower);
    /// * limits, unions and placeholder rows with their `try_lower`;
    /// * `CoalescePartitionsExec`s without a fetch (over at least one
    ///   partition) and `RepartitionExec`s that do not preserve the order are
    ///   removed: the pipeline engine merges the partitions of their input
    ///   anyway;
    /// * `ProjectionExec`s with a [`TestProjection`], so that the operators
    ///   below them are lowered too;
    /// * scans with the native sources;
    /// * every other node is bridged, with its subtree.
    pub(crate) fn lower(
        plan: &Arc<dyn ExecutionPlan>,
    ) -> Result<Arc<dyn PhysicalOperator>> {
        if let Some(sort) = super::try_lower(plan, lower)? {
            return Ok(sort);
        }
        if let Some(limit) = PhysicalLimit::try_lower(plan, lower)? {
            return Ok(Arc::new(limit));
        }
        if let Some(union) = PhysicalUnion::try_lower(plan, lower)? {
            return Ok(Arc::new(union));
        }
        if let Some(placeholder) = PhysicalPlaceholderRow::try_lower(plan)? {
            return Ok(Arc::new(placeholder));
        }
        if let Some(coalesce) = plan.downcast_ref::<CoalescePartitionsExec>()
            && coalesce.fetch().is_none()
            && coalesce.input().output_partitioning().partition_count() > 0
        {
            return lower(coalesce.input());
        }
        if let Some(repartition) = plan.downcast_ref::<RepartitionExec>()
            && !repartition.preserve_order()
        {
            return lower(repartition.input());
        }
        if let Some(projection) = plan.downcast_ref::<ProjectionExec>() {
            return Ok(Arc::new(TestProjection {
                children: vec![lower(projection.input())?],
                exprs: projection
                    .expr()
                    .iter()
                    .map(|expr| Arc::clone(&expr.expr))
                    .collect(),
                schema: projection.schema(),
            }));
        }
        if let Some(source) = try_create_scan_source(plan)? {
            return Ok(source);
        }
        Ok(Arc::new(BridgeOperator::new(Arc::clone(plan))))
    }

    /// Evaluates the expressions of a `ProjectionExec` on the compacted input
    /// chunks: a minimal stand-in for the native projection.
    #[derive(Debug)]
    pub(crate) struct TestProjection {
        children: Vec<Arc<dyn PhysicalOperator>>,
        exprs: Vec<Arc<dyn PhysicalExpr>>,
        schema: SchemaRef,
    }

    impl PhysicalOperator for TestProjection {
        fn name(&self) -> &str {
            "TestProjection"
        }

        fn schema(&self) -> SchemaRef {
            Arc::clone(&self.schema)
        }

        fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
            &self.children
        }

        fn execute(
            &self,
            _ctx: &ExecutionContext,
            input: &DataChunk,
            output: &mut DataChunk,
            _gstate: &dyn GlobalOperatorState,
            _lstate: &mut dyn OperatorState,
        ) -> Result<OperatorResult> {
            let batch = input.compact()?;
            let columns = self
                .exprs
                .iter()
                .map(|expr| expr.evaluate(&batch)?.into_array(batch.num_rows()))
                .collect::<Result<Vec<_>>>()?;
            let options =
                RecordBatchOptions::new().with_row_count(Some(batch.num_rows()));
            *output = DataChunk::from_record_batch(RecordBatch::try_new_with_options(
                Arc::clone(&self.schema),
                columns,
                &options,
            )?);
            Ok(OperatorResult::NeedMoreInput)
        }
    }

    /// Assert that `sql` produces byte-identical results on DataFusion and on
    /// the pipeline engine, both through the `PipelineQueryPlanner` and with
    /// the plan lowered by [`lower`]. Returns the lowered operator tree.
    pub(crate) async fn assert_parity(sql: &str) -> Arc<dyn PhysicalOperator> {
        assert_parity_with_config(sql, test_session_config()).await
    }

    /// [`assert_parity`] with the session configuration `config`.
    pub(crate) async fn assert_parity_with_config(
        sql: &str,
        config: SessionConfig,
    ) -> Arc<dyn PhysicalOperator> {
        let setup = async |ctx: &SessionContext| register_test_tables(ctx);
        assert_byte_eq_with_datafusion_with_config(sql, config.clone(), setup).await;
        assert_native_byte_eq(sql, config, setup).await
    }

    /// Assert that `sql` produces byte-identical results on DataFusion and on
    /// the pipeline engine with the plan lowered by [`lower`], on sessions
    /// configured with `config` and set up with `setup`. Returns the lowered
    /// operator tree.
    pub(crate) async fn assert_native_byte_eq<F>(
        sql: &str,
        config: SessionConfig,
        setup: F,
    ) -> Arc<dyn PhysicalOperator>
    where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        let ctx = datafusion_context(config);
        setup(&ctx).await.expect("setup");
        let expected = run_sql(&ctx, sql).await.expect("DataFusion");
        // A new plan: executing a plan changes the state of some nodes (e.g.
        // the dynamic filters of TopKs).
        let plan = ctx
            .sql(sql)
            .await
            .expect("planning")
            .create_physical_plan()
            .await
            .expect("physical planning");
        let root = lower(&plan).expect("lowering");
        let exec: Arc<dyn ExecutionPlan> = Arc::new(
            PipelineRootExec::try_new(Arc::clone(&root), plan).expect("a pipeline root"),
        );
        let execution = collect(Arc::clone(&exec), ctx.task_ctx());
        let batches = tokio::time::timeout(Duration::from_secs(60), execution)
            .await
            .unwrap_or_else(|_| panic!("The execution did not complete\nquery: {sql}"))
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to run the lowered plan: {e}\nquery: {sql}\nlowered: {}",
                    format_operator_tree(&root)
                )
            });
        let actual = EngineOutput {
            schema: exec.schema(),
            plan: exec,
            batches,
            ordered: expected.ordered,
        };
        if let Err(message) = compare_outputs(&expected, &actual) {
            panic!(
                "Results differ: {message}\nquery: {sql}\nlowered: {}",
                format_operator_tree(&root)
            );
        }
        root
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::array::{
        AsArray, BooleanArray, Date32Array, Decimal128Array, DictionaryArray,
        Float64Array, Int32Array, StringArray, StringViewArray, TimestampNanosecondArray,
    };
    use arrow::compute::{SortOptions, concat_batches};
    use arrow::datatypes::{Field, Int32Type, Int64Type};
    use datafusion::prelude::{ParquetReadOptions, SessionContext};
    use datafusion_datasource::memory::MemorySourceConfig;
    use datafusion_datasource::source::DataSourceExec;
    use datafusion_expr::Operator;
    use datafusion_physical_expr::PhysicalSortExpr;
    use datafusion_physical_expr::expressions::{BinaryExpr, col, lit};
    use datafusion_physical_plan::Partitioning;
    use datafusion_physical_plan::repartition::RepartitionExec;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use super::test_support::{
        assert_native_byte_eq, assert_parity, assert_parity_with_config, lower,
    };
    use crate::bridge::{BridgeOperator, bridged_plans};
    use crate::explain::format_operator_tree;
    use crate::operators::limit::PhysicalLimit;
    use crate::operators::union::test_support::{
        AfterBatch, INJECTED_FAILURE, assert_lowered_plan_byte_eq,
        assert_lowered_plan_fails_like_datafusion, batch_plan, count, values_chunk,
        values_schema,
    };
    use crate::sources::ParquetScanSource;
    use crate::sources::test_harness::{data_batch, data_schema, write_parquet};
    use crate::test_utils::{test_session_config, test_table_t1};

    // ---------------------------------------------------------------------
    // Sort keys
    // ---------------------------------------------------------------------

    /// The sort keys `columns` (with `options`) of a schema with these
    /// columns.
    fn sort_keys(schema: &SchemaRef, keys: &[(&str, SortOptions)]) -> SortKeys {
        let ordering = LexOrdering::new(keys.iter().map(|(name, options)| {
            PhysicalSortExpr::new(col(name, schema).unwrap(), *options)
        }))
        .unwrap();
        SortKeys::try_new(ordering, schema).unwrap()
    }

    /// Random columns of every type the sorts handle specially, with nulls.
    fn random_columns(rng: &mut StdRng, n: usize) -> Vec<ArrayRef> {
        let null = |rng: &mut StdRng| rng.random_range(0..8) == 0;
        let words = [
            "",
            "a",
            "ab",
            "b",
            "a fairly long string that is not inlined",
        ];
        let int = Int32Array::from_iter(
            (0..n).map(|_| (!null(rng)).then(|| rng.random_range(-5..5))),
        );
        let float = Float64Array::from_iter((0..n).map(|_| {
            (!null(rng)).then(|| match rng.random_range(0..6) {
                0 => f64::NAN,
                1 => -0.0,
                2 => 0.0,
                3 => f64::NEG_INFINITY,
                _ => f64::from(rng.random_range(-3..3)) / 2.0,
            })
        }));
        let string = StringArray::from_iter(
            (0..n).map(|_| (!null(rng)).then(|| words[rng.random_range(0..words.len())])),
        );
        let view = StringViewArray::from_iter(
            (0..n).map(|_| (!null(rng)).then(|| words[rng.random_range(0..words.len())])),
        );
        let boolean = BooleanArray::from_iter(
            (0..n).map(|_| (!null(rng)).then(|| rng.random_bool(0.5))),
        );
        let decimal = Decimal128Array::from_iter(
            (0..n).map(|_| (!null(rng)).then(|| i128::from(rng.random_range(-300..300)))),
        )
        .with_precision_and_scale(12, 2)
        .unwrap();
        let date = Date32Array::from_iter(
            (0..n).map(|_| (!null(rng)).then(|| rng.random_range(0..9))),
        );
        let ts = TimestampNanosecondArray::from_iter(
            (0..n).map(|_| (!null(rng)).then(|| rng.random_range(-4..4))),
        );
        let dictionary = DictionaryArray::<Int32Type>::from_iter(
            (0..n).map(|_| (!null(rng)).then(|| words[rng.random_range(0..words.len())])),
        );
        vec![
            Arc::new(int),
            Arc::new(float),
            Arc::new(string),
            Arc::new(view),
            Arc::new(boolean),
            Arc::new(decimal),
            Arc::new(date),
            Arc::new(ts),
            Arc::new(dictionary),
        ]
    }

    #[test]
    fn sorted_runs_are_sorted_by_their_keys() -> Result<()> {
        let mut rng = StdRng::seed_from_u64(1);
        let all_options = [
            SortOptions::new(false, false),
            SortOptions::new(false, true),
            SortOptions::new(true, false),
            SortOptions::new(true, true),
        ];
        for case in 0..40 {
            let columns = random_columns(&mut rng, 500);
            let fields = columns
                .iter()
                .enumerate()
                .map(|(i, column)| {
                    Field::new(format!("c{i}"), column.data_type().clone(), true)
                })
                .collect::<Vec<_>>();
            let schema = Arc::new(Schema::new(fields));
            let batch = RecordBatch::try_new(Arc::clone(&schema), columns)?;
            // Several batches of different sizes, one of them empty.
            let batches = [
                batch.slice(0, 120),
                batch.slice(120, 0),
                batch.slice(120, 380),
            ];
            let num_keys = rng.random_range(1..4);
            let names = (0..num_keys)
                .map(|_| format!("c{}", rng.random_range(0..schema.fields().len())))
                .collect::<Vec<_>>();
            let mut keys = vec![];
            for name in &names {
                if !keys.iter().any(|(key, _)| key == name) {
                    keys.push((name.as_str(), all_options[rng.random_range(0..4)]));
                }
            }
            let sort_keys = sort_keys(&schema, &keys);
            for limit in [None, Some(1), Some(77), Some(499), Some(500), Some(800)] {
                let run = sort_keys.sort(&schema, &batches, limit)?.expect("rows");
                assert_eq!(run.num_rows(), limit.unwrap_or(500).min(500));
                // The keys are the encoded keys of the rows of the run, in
                // order.
                let encoded = sort_keys
                    .converter
                    .convert_columns(&sort_keys.evaluate(run.batch())?)?;
                for row in 0..run.num_rows() {
                    assert_eq!(run.key(row), encoded.row(row), "case {case}");
                    if row > 0 {
                        assert!(
                            run.key(row - 1) <= run.key(row),
                            "case {case}: {keys:?}"
                        );
                    }
                }
                // The first rows: no row left out comes before the last row.
                let all = sort_keys
                    .converter
                    .convert_columns(&sort_keys.evaluate(&batch)?)?;
                if let Some(last) = run.num_rows().checked_sub(1) {
                    let before = (0..500).filter(|&i| all.row(i) < run.key(last)).count();
                    assert!(before <= last, "case {case}: {keys:?}, limit {limit:?}");
                }
                // A single key of a simple type is sorted with the sort
                // kernel, in the same order as by the row format.
                if sort_keys.sort_with_kernel {
                    let mut by_rows =
                        SortKeys::try_new(sort_keys.ordering.clone(), &schema)?;
                    by_rows.sort_with_kernel = false;
                    let other = by_rows.sort(&schema, &batches, limit)?.expect("rows");
                    for row in 0..run.num_rows() {
                        assert_eq!(run.key(row), other.key(row), "case {case}: {keys:?}");
                    }
                }
            }
            assert!(sort_keys.sort(&schema, &batches, Some(0))?.is_none());
            assert!(sort_keys.sort(&schema, &[], None)?.is_none());
        }
        Ok(())
    }

    #[test]
    fn simple_single_keys_are_sorted_with_the_kernel() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("i", DataType::Int64, true),
            Field::new("s", DataType::Utf8View, true),
            Field::new(
                "d",
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                true,
            ),
        ]));
        let options = SortOptions::default();
        assert!(sort_keys(&schema, &[("i", options)]).sort_with_kernel);
        assert!(sort_keys(&schema, &[("s", options)]).sort_with_kernel);
        assert!(!sort_keys(&schema, &[("d", options)]).sort_with_kernel);
        assert!(!sort_keys(&schema, &[("i", options), ("s", options)]).sort_with_kernel);
    }

    // ---------------------------------------------------------------------
    // Lowering
    // ---------------------------------------------------------------------

    /// `t1` with its 4 skewed partitions (`partitions == 4`), in a single
    /// partition (`1`) or without partitions (`0`).
    fn t1_scan(partitions: usize) -> Arc<dyn ExecutionPlan> {
        let t1 = test_table_t1().unwrap();
        let schema = t1[0][0].schema();
        let t1 = match partitions {
            0 => vec![],
            1 => vec![t1.into_iter().flatten().collect()],
            4 => t1,
            _ => unreachable!(),
        };
        MemorySourceConfig::try_new_exec(&t1, schema, None).unwrap()
    }

    /// `t1` in a single partition sorted by `id`, which the scan declares.
    fn sorted_t1_scan() -> Arc<dyn ExecutionPlan> {
        let t1 = test_table_t1().unwrap();
        let schema = t1[0][0].schema();
        let batch = concat_batches(&schema, t1.iter().flatten()).unwrap();
        let ordering = ordering(&schema, &[("id", false)]);
        let source = MemorySourceConfig::try_new(&[vec![batch]], schema, None)
            .unwrap()
            .try_with_sort_information(vec![ordering])
            .unwrap();
        DataSourceExec::from_data_source(source)
    }

    /// The SQL default orderings of the columns `keys` of `schema`: `ASC
    /// NULLS LAST`, or `DESC NULLS FIRST` if descending.
    fn ordering(schema: &SchemaRef, keys: &[(&str, bool)]) -> LexOrdering {
        LexOrdering::new(keys.iter().map(|&(name, descending)| {
            PhysicalSortExpr::new(
                col(name, schema).unwrap(),
                SortOptions::new(descending, descending),
            )
        }))
        .unwrap()
    }

    fn sort_exec(
        input: Arc<dyn ExecutionPlan>,
        keys: &[(&str, bool)],
        preserve_partitioning: bool,
        fetch: Option<usize>,
    ) -> Arc<dyn ExecutionPlan> {
        let ordering = ordering(&input.schema(), keys);
        Arc::new(
            SortExec::new(ordering, input)
                .with_fetch(fetch)
                .with_preserve_partitioning(preserve_partitioning),
        )
    }

    fn merge(
        input: Arc<dyn ExecutionPlan>,
        keys: &[(&str, bool)],
        fetch: Option<usize>,
    ) -> Arc<dyn ExecutionPlan> {
        let ordering = ordering(&input.schema(), keys);
        Arc::new(SortPreservingMergeExec::new(ordering, input).with_fetch(fetch))
    }

    /// Lower `plan` with [`try_lower`], bridging its input. Returns the
    /// operator and the input plan it was lowered with.
    fn try_lower_bridged(
        plan: &Arc<dyn ExecutionPlan>,
    ) -> Option<(Arc<dyn PhysicalOperator>, Arc<dyn ExecutionPlan>)> {
        let mut input = None;
        let lowered = try_lower(plan, |plan| {
            assert!(input.is_none(), "the input is lowered once");
            input = Some(Arc::clone(plan));
            Ok(Arc::new(BridgeOperator::new(Arc::clone(plan))))
        })
        .unwrap();
        match lowered {
            Some(operator) => Some((operator, input.expect("the lowered input"))),
            None => {
                assert!(input.is_none(), "the input is only lowered with the sort");
                None
            }
        }
    }

    /// The fetch of a lowered sort: `None` for a [`PhysicalSort`] without
    /// fetch, `Some(k)` for a [`PhysicalTopK`].
    fn lowered_fetch(operator: &Arc<dyn PhysicalOperator>) -> Option<usize> {
        if let Some(sort) = operator.downcast_ref::<PhysicalSort>() {
            assert_eq!(sort.fetch(), None);
            None
        } else {
            Some(operator.downcast_ref::<PhysicalTopK>().expect("a sort").k())
        }
    }

    fn lowered_ordering(operator: &Arc<dyn PhysicalOperator>) -> String {
        match operator.downcast_ref::<PhysicalSort>() {
            Some(sort) => sort.ordering().to_string(),
            None => operator
                .downcast_ref::<PhysicalTopK>()
                .unwrap()
                .ordering()
                .to_string(),
        }
    }

    #[test]
    fn lowering_of_merged_sorts() {
        let id = [("id", false)];
        // A merge of the sorted partitions: a sort of all rows.
        let scan = t1_scan(4);
        let plan = merge(sort_exec(Arc::clone(&scan), &id, true, None), &id, None);
        let (sort, input) = try_lower_bridged(&plan).expect("a sort");
        assert_eq!(lowered_fetch(&sort), None);
        assert_eq!(lowered_ordering(&sort), "id@0 ASC NULLS LAST");
        assert!(
            Arc::ptr_eq(&input, &scan),
            "the input of the SortExec is sorted"
        );
        assert_eq!(sort.schema(), plan.schema());
        assert_eq!(
            sort.description(),
            "PhysicalSort: expr=[id@0 ASC NULLS LAST]"
        );

        // The fetches of the merge and of the sort of every partition.
        for (partitions, sort_fetch, merge_fetch, expected) in [
            (4, Some(10), Some(10), Some(Some(10))),
            (4, Some(10), Some(3), Some(Some(3))),
            (4, None, Some(3), Some(Some(3))),
            // The top 10 rows of every partition are not the top 20 or all
            // rows of all partitions...
            (4, Some(10), Some(20), None),
            (4, Some(10), None, None),
            // ... unless there is a single partition.
            (1, Some(10), Some(20), Some(Some(10))),
            (1, Some(10), None, Some(Some(10))),
            (1, Some(10), Some(4), Some(Some(4))),
            (1, None, None, Some(None)),
            // Sorts of no rows are left to DataFusion.
            (4, Some(0), Some(0), None),
            (4, None, Some(0), None),
            (1, Some(0), None, None),
        ] {
            let plan = merge(
                sort_exec(
                    t1_scan(partitions),
                    &[("d", true), ("id", false)],
                    true,
                    sort_fetch,
                ),
                &[("d", true), ("id", false)],
                merge_fetch,
            );
            let lowered = try_lower_bridged(&plan);
            let case = format!(
                "{partitions} partitions, fetch {sort_fetch:?} / {merge_fetch:?}"
            );
            assert_eq!(
                lowered.as_ref().map(|(sort, _)| lowered_fetch(sort)),
                expected,
                "{case}"
            );
            if let Some((sort, _)) = lowered {
                assert_eq!(
                    lowered_ordering(&sort),
                    "d@4 DESC, id@0 ASC NULLS LAST",
                    "{case}"
                );
                // The TopK updates the dynamic filter of a SortExec with a
                // fetch.
                if let Some(topk) = sort.downcast_ref::<PhysicalTopK>() {
                    assert_eq!(
                        topk.dynamic_filter().is_some(),
                        sort_fetch.is_some(),
                        "{case}"
                    );
                }
            }
        }
        // The partitions must be sorted by the order of the merge.
        let plan = merge(
            sort_exec(t1_scan(4), &id, true, None),
            &[("a", false)],
            None,
        );
        assert!(try_lower_bridged(&plan).is_none());
        // A merge of no partitions fails in DataFusion.
        let plan = merge(sort_exec(t1_scan(0), &id, true, None), &id, None);
        assert!(try_lower_bridged(&plan).is_none());
        // A merge of something else than a sort is left to DataFusion.
        assert!(try_lower_bridged(&merge(sorted_t1_scan(), &id, None)).is_none());
    }

    #[test]
    fn lowering_of_sorts_of_a_single_partition() {
        let keys = [("c", false), ("id", true)];
        // A sort of all rows of a single partition, or of the partitions of a
        // `CoalescePartitionsExec`, which the native sort replaces.
        for (plan, input) in [
            (sort_exec(t1_scan(1), &keys, false, None), None),
            (sort_exec(t1_scan(1), &keys, true, Some(5)), None),
            (
                sort_exec(
                    Arc::new(CoalescePartitionsExec::new(t1_scan(4))),
                    &keys,
                    false,
                    Some(7),
                ),
                Some("DataSourceExec"),
            ),
        ] {
            let (sort, lowered_input) = try_lower_bridged(&plan).expect("a sort");
            let sort_input = Arc::clone(plan.children()[0]);
            match input {
                None => assert!(Arc::ptr_eq(&lowered_input, &sort_input)),
                Some(name) => assert_eq!(lowered_input.name(), name),
            }
            assert_eq!(lowered_fetch(&sort), plan.fetch());
            assert_eq!(lowered_ordering(&sort), "c@3 ASC NULLS LAST, id@0 DESC");
        }
        // A `CoalescePartitionsExec` with a fetch is kept, as is one of no
        // partitions (DataFusion fails to execute it).
        for coalesce in [
            CoalescePartitionsExec::new(t1_scan(4)).with_fetch(Some(9)),
            CoalescePartitionsExec::new(t1_scan(0)),
        ] {
            let plan = sort_exec(Arc::new(coalesce), &keys, false, None);
            let (_, lowered_input) = try_lower_bridged(&plan).expect("a sort");
            assert!(lowered_input.is::<CoalescePartitionsExec>());
        }
        // DataFusion sorts only the first of several partitions of the input of
        // a sort of all rows, and a sort of several partitions sorts every
        // partition: neither is a sort of all rows unless merged.
        assert!(try_lower_bridged(&sort_exec(t1_scan(4), &keys, false, None)).is_none());
        assert!(try_lower_bridged(&sort_exec(t1_scan(4), &keys, true, None)).is_none());
        assert!(try_lower_bridged(&sort_exec(t1_scan(0), &keys, true, None)).is_none());
        assert!(
            try_lower_bridged(&sort_exec(t1_scan(1), &keys, false, Some(0))).is_none()
        );
    }

    #[test]
    fn sorts_of_sorted_inputs_are_left_to_datafusion() {
        // DataFusion passes an input that is sorted already through.
        let scan = sorted_t1_scan();
        for fetch in [None, Some(3)] {
            let plan = sort_exec(Arc::clone(&scan), &[("id", false)], false, fetch);
            assert!(try_lower_bridged(&plan).is_none(), "fetch {fetch:?}");
        }
        // A TopK of an input sorted by a prefix of its keys ends early.
        let plan = sort_exec(
            Arc::clone(&scan),
            &[("id", false), ("a", false)],
            false,
            Some(3),
        );
        assert!(try_lower_bridged(&plan).is_none());
        // Without a fetch the sort reads its whole input, and so does a TopK
        // whose keys do not start with the order of the input.
        let plan = sort_exec(
            Arc::clone(&scan),
            &[("id", false), ("a", false)],
            false,
            None,
        );
        assert!(try_lower_bridged(&plan).is_some());
        let plan = sort_exec(
            Arc::clone(&scan),
            &[("a", false), ("id", false)],
            false,
            Some(3),
        );
        assert!(try_lower_bridged(&plan).is_some());
    }

    #[test]
    fn sorts_whose_partitioning_is_stale_are_left_to_datafusion() {
        let id = [("id", false)];
        let keep = ReplaceChildrenOptions::new(ChildrenPropertiesMode::Keep);
        // A sort of every partition whose input was replaced without
        // recomputing its properties: it advertises 4 partitions of an input
        // with a single one.
        let stale = sort_exec(t1_scan(4), &id, true, None)
            .replace_children(vec![t1_scan(1)], keep)
            .unwrap();
        assert_eq!(stale.output_partitioning().partition_count(), 4);
        assert!(try_lower_bridged(&merge(stale, &id, None)).is_none());
        // The same for a projection between the merge and the sort.
        let projection = |input| -> Arc<dyn ExecutionPlan> {
            let exprs =
                vec![(col("id", &t1_scan(1).schema()).unwrap(), "id".to_string())];
            Arc::new(ProjectionExec::try_new(exprs, input).unwrap())
        };
        let stale = projection(sort_exec(t1_scan(4), &id, true, None))
            .replace_children(vec![sort_exec(t1_scan(1), &id, true, None)], keep)
            .unwrap();
        assert_eq!(stale.output_partitioning().partition_count(), 4);
        assert!(try_lower_bridged(&merge(stale, &id, None)).is_none());
        let fresh = projection(sort_exec(t1_scan(1), &id, true, None));
        assert!(try_lower_bridged(&merge(fresh, &id, None)).is_some());
    }

    /// A projection of `id AS x` and `a`, or `a * 2 AS a` if `computed`, over
    /// `input`.
    fn project(input: Arc<dyn ExecutionPlan>, computed: bool) -> Arc<dyn ExecutionPlan> {
        let schema = input.schema();
        let a = col("a", &schema).unwrap();
        let a = if computed {
            Arc::new(BinaryExpr::new(a, Operator::Multiply, lit(2_i32)))
        } else {
            a
        };
        let exprs = vec![
            (col("id", &schema).unwrap(), "x".to_string()),
            (a, "a".to_string()),
        ];
        Arc::new(ProjectionExec::try_new(exprs, input).unwrap())
    }

    /// A projection of the columns of `input`, with column `from` renamed to
    /// `to`.
    fn rename(
        input: Arc<dyn ExecutionPlan>,
        from: &str,
        to: &str,
    ) -> Arc<dyn ExecutionPlan> {
        let schema = input.schema();
        let exprs = schema
            .fields()
            .iter()
            .map(|field| {
                let name = if field.name() == from {
                    to
                } else {
                    field.name()
                };
                (col(field.name(), &schema).unwrap(), name.to_string())
            })
            .collect::<Vec<_>>();
        Arc::new(ProjectionExec::try_new(exprs, input).unwrap())
    }

    #[test]
    fn lowering_of_merged_sorts_below_projections() {
        let id = [("id", true)];
        // A merge of projections of sorted partitions: the native sort sorts
        // the projected rows, by the order of the merge.
        for fetch in [None, Some(10)] {
            let plan = merge(
                project(sort_exec(t1_scan(4), &id, true, fetch), false),
                &[("x", true)],
                fetch,
            );
            let (sort, input) = try_lower_bridged(&plan).expect("a sort");
            assert_eq!(lowered_fetch(&sort), fetch);
            assert_eq!(lowered_ordering(&sort), "x@0 DESC");
            let projection = input
                .downcast_ref::<ProjectionExec>()
                .expect("the projection");
            assert!(
                projection.input().is::<DataSourceExec>(),
                "the sort is removed"
            );
            assert_eq!(input.schema(), plan.schema());
            if let Some(topk) = sort.downcast_ref::<PhysicalTopK>() {
                // The filter of the SortExec on `id`, the key `x` is mapped to.
                let filter = topk.dynamic_filter().expect("a dynamic filter");
                assert_eq!(filter.keys().len(), 1);
                assert_eq!(filter.keys()[0].to_string(), "id@0");
                assert!(topk.description().ends_with("dynamic_filter=true"));
            }
            // Two projections.
            let plan = merge(
                rename(
                    project(sort_exec(t1_scan(4), &id, true, fetch), false),
                    "x",
                    "z",
                ),
                &[("z", true)],
                fetch,
            );
            let (sort, input) = try_lower_bridged(&plan).expect("a sort");
            assert_eq!(lowered_fetch(&sort), fetch);
            let projection = input.downcast_ref::<ProjectionExec>().unwrap();
            assert!(projection.input().is::<ProjectionExec>());
            if let Some(topk) = sort.downcast_ref::<PhysicalTopK>() {
                assert_eq!(topk.dynamic_filter().unwrap().keys()[0].to_string(), "id@0");
            }
        }
        // A computed projection is evaluated on all rows, which is what
        // DataFusion does without a fetch, but not with one.
        let plan = merge(
            project(sort_exec(t1_scan(4), &id, true, None), true),
            &[("x", true)],
            None,
        );
        let (sort, _) = try_lower_bridged(&plan).expect("a sort");
        assert_eq!(lowered_ordering(&sort), "x@0 DESC");
        let plan = merge(
            project(sort_exec(t1_scan(4), &id, true, None), true),
            &[("x", true)],
            Some(5),
        );
        assert_eq!(lowered_fetch(&try_lower_bridged(&plan).unwrap().0), Some(5));
        let plan = merge(
            project(sort_exec(t1_scan(4), &id, true, Some(5)), true),
            &[("x", true)],
            Some(5),
        );
        assert!(try_lower_bridged(&plan).is_none());
    }

    #[test]
    fn lowering_of_merges_of_merged_sorts() {
        // DataFusion plans a merge of a projection of a merge of the sorted
        // partitions, for `SELECT b, a FROM t ORDER BY a LIMIT k`: the top
        // merge limits the single partition of the merge below it.
        let id = [("id", true)];
        for (top_fetch, fetch, expected) in [
            (Some(5), Some(10), Some(Some(5))),
            (Some(20), Some(10), Some(Some(10))),
            (None, Some(10), Some(Some(10))),
            (Some(7), None, Some(Some(7))),
            (None, None, Some(None)),
            (Some(0), Some(10), None),
        ] {
            let plan = merge(
                rename(
                    merge(sort_exec(t1_scan(4), &id, true, fetch), &id, fetch),
                    "id",
                    "x",
                ),
                &[("x", true)],
                top_fetch,
            );
            let lowered = try_lower_bridged(&plan);
            let case = format!("fetch {top_fetch:?} / {fetch:?}");
            assert_eq!(
                lowered.as_ref().map(|(sort, _)| lowered_fetch(sort)),
                expected,
                "{case}"
            );
            if let Some((sort, input)) = lowered {
                assert_eq!(lowered_ordering(&sort), "x@0 DESC", "{case}");
                // The projection of the input of the sort, without the merges.
                let projection = input
                    .downcast_ref::<ProjectionExec>()
                    .expect("a projection");
                assert!(projection.input().is::<DataSourceExec>(), "{case}");
                if let Some(topk) = sort.downcast_ref::<PhysicalTopK>() {
                    assert_eq!(
                        topk.dynamic_filter().is_some(),
                        fetch.is_some(),
                        "{case}"
                    );
                }
            }
        }
        // A computed projection above a merge with a fetch is evaluated on the
        // rows the merge keeps only.
        let plan = merge(
            project(
                merge(sort_exec(t1_scan(4), &id, true, None), &id, Some(3)),
                true,
            ),
            &[("x", true)],
            None,
        );
        assert!(try_lower_bridged(&plan).is_none());
        let plan = merge(
            project(
                merge(sort_exec(t1_scan(4), &id, true, None), &id, None),
                true,
            ),
            &[("x", true)],
            Some(3),
        );
        assert_eq!(lowered_fetch(&try_lower_bridged(&plan).unwrap().0), Some(3));
    }

    #[test]
    fn lowering_checks_the_node_and_the_lowered_input() -> Result<()> {
        // Other nodes are not lowered, without lowering their input.
        let not_called =
            |_: &Arc<dyn ExecutionPlan>| -> Result<Arc<dyn PhysicalOperator>> {
                panic!("not called")
            };
        for plan in [
            t1_scan(4),
            Arc::new(CoalescePartitionsExec::new(t1_scan(4))) as _,
            Arc::new(RepartitionExec::try_new(
                t1_scan(1),
                Partitioning::RoundRobinBatch(4),
            )?) as _,
        ] {
            assert!(try_lower(&plan, not_called)?.is_none());
        }
        // `PhysicalSort::try_lower` lowers the sorts without fetch,
        // `PhysicalTopK::try_lower` the others.
        let id = [("id", false)];
        let sort = merge(sort_exec(t1_scan(4), &id, true, None), &id, None);
        let topk = merge(sort_exec(t1_scan(4), &id, true, Some(3)), &id, Some(3));
        assert!(PhysicalTopK::try_lower(&sort, not_called)?.is_none());
        assert!(PhysicalSort::try_lower(&topk, not_called)?.is_none());
        let bridge =
            |plan: &Arc<dyn ExecutionPlan>| -> Result<Arc<dyn PhysicalOperator>> {
                Ok(Arc::new(BridgeOperator::new(Arc::clone(plan))))
            };
        assert!(PhysicalSort::try_lower(&sort, bridge)?.is_some());
        assert_eq!(PhysicalTopK::try_lower(&topk, bridge)?.unwrap().k(), 3);
        // The lowered input must have the schema of the sort.
        let err = try_lower(&sort, |_| {
            Ok(Arc::new(BridgeOperator::new(batch_plan(
                values_chunk(0..1).into_record_batch()?,
                AfterBatch::End,
            )?)))
        })
        .unwrap_err();
        assert!(err.to_string().contains("has schema"), "{err}");
        Ok(())
    }

    // ---------------------------------------------------------------------
    // Differential tests
    // ---------------------------------------------------------------------

    /// The number of native sorts in the tree below `root`.
    fn sorts(root: &Arc<dyn PhysicalOperator>) -> usize {
        count::<PhysicalSort>(root) + count::<PhysicalTopK>(root)
    }

    async fn assert_sorted_natively(queries: &[&str]) {
        for sql in queries {
            let root = assert_parity(sql).await;
            assert!(sorts(&root) > 0, "{sql}\n{}", format_operator_tree(&root));
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_order_by_a_column_of_every_type() {
        assert_sorted_natively(&[
            "SELECT * FROM t1 ORDER BY id",
            "SELECT * FROM t1 ORDER BY id DESC",
            "SELECT id, a FROM t1 ORDER BY a, id",
            "SELECT id, a FROM t1 ORDER BY a DESC NULLS LAST, id",
            "SELECT id, a FROM t1 ORDER BY a NULLS FIRST, id DESC",
            "SELECT id, b FROM t1 ORDER BY b DESC, id",
            // Floats with NaN, -0.0 and 0.0.
            "SELECT id, c FROM t1 ORDER BY c, id",
            "SELECT id, c FROM t1 ORDER BY c DESC NULLS LAST, id",
            "SELECT id, c FROM t1 ORDER BY c NULLS FIRST, id DESC",
            "SELECT id, d FROM t1 ORDER BY d, id",
            "SELECT id, d FROM t1 ORDER BY d DESC NULLS FIRST, id DESC",
            "SELECT id, e FROM t1 ORDER BY e, id",
            "SELECT id, f FROM t1 ORDER BY f DESC, id",
            "SELECT id, g FROM t1 ORDER BY g, id",
            "SELECT id, g FROM t1 ORDER BY g DESC NULLS LAST, id DESC",
            "SELECT id, h FROM t1 ORDER BY h DESC NULLS LAST, id",
            "SELECT id, ts FROM t1 ORDER BY ts NULLS FIRST, id",
            // Ties of rows that are identical in the output.
            "SELECT a FROM t1 ORDER BY a",
            "SELECT c FROM t1 ORDER BY c DESC",
            "SELECT d, e FROM t1 ORDER BY d, e NULLS FIRST",
            "SELECT h FROM t1 ORDER BY h",
            "SELECT k, name FROM t2 ORDER BY k DESC, name",
        ])
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_order_by_several_columns_and_expressions() {
        assert_sorted_natively(&[
            "SELECT * FROM t1 ORDER BY e, d DESC, a NULLS FIRST, id",
            "SELECT * FROM t1 ORDER BY f, g DESC NULLS LAST, c, id DESC",
            "SELECT id, a + b AS s FROM t1 ORDER BY s, id",
            "SELECT id, upper(d) AS u FROM t1 ORDER BY u DESC, id",
            "SELECT id, b FROM t1 ORDER BY a * 2 + 1 DESC, id",
            "SELECT id, c FROM t1 ORDER BY abs(c) NULLS FIRST, id",
            "SELECT id, h, d FROM t1 ORDER BY h || d, id",
            "SELECT id FROM t1 WHERE a > 0 ORDER BY b, id",
        ])
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_order_by_limit() {
        for sql in [
            "SELECT * FROM t1 ORDER BY id LIMIT 10",
            "SELECT * FROM t1 ORDER BY id DESC LIMIT 1",
            "SELECT id, a FROM t1 ORDER BY a NULLS FIRST, id LIMIT 7",
            "SELECT id, a FROM t1 ORDER BY a DESC NULLS LAST, id LIMIT 25",
            "SELECT id, c FROM t1 ORDER BY c, id LIMIT 100",
            "SELECT id, c FROM t1 ORDER BY c DESC NULLS LAST, id LIMIT 2500",
            "SELECT id, d FROM t1 ORDER BY d DESC, id LIMIT 777",
            "SELECT id, g FROM t1 ORDER BY g, id LIMIT 50",
            "SELECT id, h FROM t1 ORDER BY h, id DESC LIMIT 1024",
            "SELECT id, ts FROM t1 ORDER BY ts DESC, id LIMIT 2049",
            "SELECT * FROM t1 ORDER BY e NULLS FIRST, f, id LIMIT 4999",
            "SELECT * FROM t1 ORDER BY b, id LIMIT 5000",
            "SELECT * FROM t1 ORDER BY b, id LIMIT 6000",
            "SELECT id, b FROM t1 ORDER BY a * 2 + 1 DESC, id LIMIT 33",
            // Ties of rows that are identical in the output, around the
            // zeros of `c`.
            "SELECT c FROM t1 ORDER BY c LIMIT 2300",
            "SELECT c FROM t1 ORDER BY c DESC LIMIT 2550",
            "SELECT a, e FROM t1 ORDER BY a, e LIMIT 1000",
            // With an offset.
            "SELECT id, d FROM t1 ORDER BY id LIMIT 10 OFFSET 5",
            "SELECT id, c FROM t1 ORDER BY c DESC, id LIMIT 3000 OFFSET 2047",
        ] {
            let root = assert_parity(sql).await;
            assert_eq!(
                count::<PhysicalTopK>(&root),
                1,
                "{sql}\n{}",
                format_operator_tree(&root)
            );
        }
        // A limit of the sorted rows.
        let root = assert_parity("SELECT id FROM t1 ORDER BY id LIMIT 5 OFFSET 3").await;
        assert_eq!(count::<PhysicalLimit>(&root), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_order_by_of_aggregates_joins_and_subqueries() {
        assert_sorted_natively(&[
            // Projections between the merge and the sort.
            "SELECT d, COUNT(*) AS n FROM t1 GROUP BY d ORDER BY d",
            "SELECT d, COUNT(*) AS n FROM t1 GROUP BY d ORDER BY d DESC LIMIT 3",
            "SELECT d, COUNT(*) AS n FROM t1 GROUP BY d ORDER BY n DESC, d LIMIT 4",
            "SELECT d AS x, SUM(b) AS s FROM t1 GROUP BY d ORDER BY x DESC",
            "SELECT e, f, COUNT(DISTINCT a) AS da FROM t1 GROUP BY e, f ORDER BY f, e",
            "SELECT t1.id, t2.name FROM t1 JOIN t2 ON t1.a = t2.k ORDER BY t1.id, t2.name, t2.w LIMIT 50",
            "SELECT t1.id, t2.w FROM t1 JOIN t2 ON t1.a = t2.k ORDER BY t2.w DESC, t1.id",
            "SELECT * FROM (SELECT id, a FROM t1 ORDER BY a NULLS FIRST, id LIMIT 10) ORDER BY id",
        ])
        .await;
        // Sorts that DataFusion pushes below a union or a join, and sorts that
        // it removes, are left to DataFusion.
        for sql in [
            "SELECT id FROM t1 UNION ALL SELECT k FROM t2 ORDER BY 1",
            "SELECT id, d FROM t1 WHERE id IN (SELECT w FROM t2) ORDER BY d, id",
            "SELECT * FROM t1 WHERE id = 4999 ORDER BY id DESC LIMIT 3",
        ] {
            let root = assert_parity(sql).await;
            assert_eq!(sorts(&root), 0, "{sql}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_sorts_of_empty_single_row_and_skewed_inputs() {
        assert_sorted_natively(&[
            "SELECT * FROM t_empty ORDER BY id",
            "SELECT * FROM t_empty ORDER BY c DESC, id LIMIT 5",
            "SELECT * FROM t1 WHERE id < 0 ORDER BY a",
            "SELECT * FROM t1 WHERE id < 0 ORDER BY a LIMIT 3",
            "SELECT * FROM t1 WHERE id = 42 ORDER BY a",
            "SELECT * FROM t1 WHERE id = 4999 ORDER BY a DESC LIMIT 3",
            // `t1` has skewed partitions: most rows in one, none in another.
            "SELECT * FROM t1 WHERE id >= 4990 OR id < 5 ORDER BY id DESC",
            "SELECT * FROM t2 ORDER BY w, k, name",
        ])
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_sorts_with_other_configurations() {
        for target_partitions in [1, 3, 16] {
            let config = test_session_config().with_target_partitions(target_partitions);
            for sql in [
                "SELECT * FROM t1 ORDER BY c, id",
                "SELECT id, d FROM t1 ORDER BY d DESC, id LIMIT 100",
                "SELECT d, COUNT(*) AS n FROM t1 GROUP BY d ORDER BY n, d LIMIT 3",
            ] {
                let root = assert_parity_with_config(sql, config.clone()).await;
                assert_eq!(sorts(&root), 1, "{sql}");
            }
        }
        // A `SortExec` of a `CoalescePartitionsExec` rather than a merge of
        // sorted partitions.
        let mut config = test_session_config();
        config.options_mut().optimizer.repartition_sorts = false;
        for sql in [
            "SELECT * FROM t1 ORDER BY h DESC, id",
            "SELECT id, b FROM t1 ORDER BY b NULLS FIRST, id LIMIT 30",
        ] {
            let root = assert_parity_with_config(sql, config.clone()).await;
            assert_eq!(sorts(&root), 1, "{sql}");
            assert!(bridged_plans(&root).is_empty(), "{sql}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_sort_plans() {
        // Plans that SQL does not produce here.
        type Plan = fn() -> Arc<dyn ExecutionPlan>;
        let plans: [Plan; 4] = [
            // A sort of all partitions of a `CoalescePartitionsExec`.
            || {
                let coalesce = Arc::new(CoalescePartitionsExec::new(t1_scan(4)));
                sort_exec(coalesce, &[("c", true), ("id", false)], false, None)
            },
            // The top 9 rows of every partition, merged to the top 7.
            || {
                let keys = [("d", true), ("id", false)];
                merge(sort_exec(t1_scan(4), &keys, true, Some(9)), &keys, Some(7))
            },
            // A TopK of a single partition.
            || sort_exec(t1_scan(1), &[("g", false), ("id", false)], true, Some(3000)),
            // A merge of fewer rows than the TopK of the only partition.
            || {
                let keys = [("ts", true), ("id", true)];
                merge(sort_exec(t1_scan(1), &keys, true, Some(9)), &keys, Some(4))
            },
        ];
        let task_ctx = Arc::new(
            datafusion_execution::TaskContext::default()
                .with_session_config(test_session_config()),
        );
        for make in plans {
            let root = assert_lowered_plan_byte_eq(
                || Ok(make()),
                lower,
                Arc::clone(&task_ctx),
                true,
            )
            .await;
            assert_eq!(sorts(&root), 1, "{}", format_operator_tree(&root));
            assert!(bridged_plans(&root).is_empty());
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sorts_fail_like_datafusion() {
        // An input that fails after its first batch fails the sort and the
        // TopK, which read all of their input, as in DataFusion.
        for fetch in [None, Some(1)] {
            let make = move || -> Result<Arc<dyn ExecutionPlan>> {
                let batch = values_chunk((0..100).rev()).into_record_batch()?;
                let input = batch_plan(batch, AfterBatch::Fail)?;
                Ok(sort_exec(input, &[("v", false)], false, fetch))
            };
            let root = assert_lowered_plan_fails_like_datafusion(
                make,
                lower,
                test_session_config(),
                INJECTED_FAILURE,
            )
            .await;
            assert_eq!(sorts(&root), 1, "fetch {fetch:?}");
        }
        // A sort of a `CoalescePartitionsExec` of no partitions fails.
        let make = || -> Result<Arc<dyn ExecutionPlan>> {
            let coalesce = Arc::new(CoalescePartitionsExec::new(t1_scan(0)));
            Ok(sort_exec(coalesce, &[("id", false)], false, None))
        };
        let root = assert_lowered_plan_fails_like_datafusion(
            make,
            lower,
            test_session_config(),
            "requires at least one input partition",
        )
        .await;
        assert_eq!(sorts(&root), 1);
        // A sort key that fails to evaluate.
        let root = assert_lowered_plan_fails_like_datafusion(
            || {
                let scan = t1_scan(4);
                let schema = scan.schema();
                let divided =
                    BinaryExpr::new(lit(10_i64), Operator::Divide, col("id", &schema)?);
                let ordering =
                    LexOrdering::new([PhysicalSortExpr::new_default(Arc::new(divided))])
                        .unwrap();
                let sort = SortExec::new(ordering.clone(), scan)
                    .with_preserve_partitioning(true);
                Ok(Arc::new(SortPreservingMergeExec::new(ordering, Arc::new(sort))) as _)
            },
            lower,
            test_session_config(),
            "Divide by zero",
        )
        .await;
        assert_eq!(sorts(&root), 1);
        let _ = values_schema();
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn topk_prunes_parquet_files_with_the_dynamic_filter() -> Result<()> {
        // 20 files of 500 rows each, in order of `id`.
        let dir = tempfile::tempdir()?;
        for i in 0..20 {
            let path = dir.path().join(format!("p/{i:02}.parquet"));
            write_parquet(
                &path,
                data_schema(),
                &[data_batch(i * 500..(i + 1) * 500)],
                None,
            )?;
        }
        let path = dir
            .path()
            .join("p")
            .to_str()
            .expect("a UTF-8 path")
            .to_string();
        let setup = async |ctx: &SessionContext| {
            ctx.register_parquet("p", &path, ParquetReadOptions::default())
                .await
        };
        for target_partitions in [1, 4] {
            let config = test_session_config().with_target_partitions(target_partitions);
            for sql in [
                "SELECT * FROM p ORDER BY id LIMIT 10",
                "SELECT id, s FROM p ORDER BY a DESC NULLS LAST, id LIMIT 10",
                "SELECT * FROM p ORDER BY c NULLS FIRST, id DESC LIMIT 1700",
            ] {
                let root = assert_native_byte_eq(sql, config.clone(), &setup).await;
                let operators = std::iter::successors(Some(Arc::clone(&root)), |op| {
                    op.children().first().map(Arc::clone)
                })
                .collect::<Vec<_>>();
                let tree = format_operator_tree(&root);
                let topk = operators
                    .iter()
                    .position(|op| op.is::<PhysicalTopK>())
                    .unwrap_or_else(|| panic!("{sql}\n{tree}"));
                // The TopK reads the native scan (through projections).
                assert!(
                    operators[topk..]
                        .iter()
                        .any(|op| op.is::<ParquetScanSource>()),
                    "{sql}\n{tree}"
                );
                assert!(bridged_plans(&root).is_empty(), "{sql}\n{tree}");
                let topk = operators[topk].downcast_ref::<PhysicalTopK>().unwrap();
                let filter = topk.dynamic_filter().expect("the dynamic filter");
                assert!(filter.filter().is_complete());
                assert_ne!(filter.filter().current()?.to_string(), "true");
            }
        }
        // On a single thread, the files after the first ones are pruned once
        // the threshold is known.
        let ctx = crate::test_utils::datafusion_context(
            test_session_config().with_target_partitions(1),
        );
        setup(&ctx).await?;
        let plan = ctx
            .sql("SELECT * FROM p ORDER BY id LIMIT 10")
            .await?
            .create_physical_plan()
            .await?;
        let root = lower(&plan)?;
        let exec = Arc::new(crate::bridge::PipelineRootExec::try_new(
            Arc::clone(&root),
            plan,
        )?);
        let output = datafusion_physical_plan::collect(
            Arc::clone(&exec) as Arc<dyn ExecutionPlan>,
            ctx.task_ctx(),
        )
        .await?;
        let ids = output
            .iter()
            .flat_map(|batch| {
                batch
                    .column(0)
                    .as_primitive::<Int64Type>()
                    .values()
                    .to_vec()
            })
            .collect::<Vec<_>>();
        assert_eq!(ids, (0..10).collect::<Vec<_>>());
        let metrics = exec.execution_metrics().expect("metrics");
        let (_, scan) = metrics
            .iter()
            .find(|(operator, _)| operator.is::<ParquetScanSource>())
            .expect("the scan");
        assert!(
            scan.output_rows() < 5000,
            "the scan read {} rows",
            scan.output_rows()
        );
        Ok(())
    }
}
