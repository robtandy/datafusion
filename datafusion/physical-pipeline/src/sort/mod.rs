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
//! A single sort key of a type that [`compares_directly`] (integers, floats,
//! decimals, temporal values, booleans, strings and binary values) is not
//! encoded: the runs hold its sorted values, which the merge compares
//! directly in the typed arrays (in the order of the row format, floats in
//! their total order). The keys of all other sorts are encoded in the Arrow
//! row format, in which comparing two keys is a `memcmp`, so the merge works
//! for every sort key the row format supports (sorts of other keys are left
//! to DataFusion).
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
//!   partition, possibly through `ProjectionExec`s of columns and literals
//!   (DataFusion plans the projection of `GROUP BY ... ORDER BY` between them)
//!   and merges of the single partition below them (as for `SELECT b FROM t
//!   ORDER BY a LIMIT k`). The native sort sorts all rows of the input of the
//!   `SortExec` by the order of the top merge, above the projections, which
//!   are lowered with the input.
//!
//! The result keeps `fetch` rows if the `SortExec` or a merge has a fetch:
//! that is a [`PhysicalTopK`], whose threshold updates the
//! [`DynamicFilterPhysicalExpr`] of the `SortExec` (which DataFusion pushes
//! into the scans below it), or, for a merge with a fetch of a sort of keys
//! that can fail to evaluate, a [`PhysicalSort`] with a fetch, which evaluates
//! the keys of all rows like DataFusion's sort. A TopK by float keys checks
//! the last row of its result, and executes DataFusion's TopK instead if
//! DataFusion's result may differ (see [`FloatKeyFallback`]). Sorts whose rows
//! (or errors) the native operators cannot reproduce exactly are not lowered,
//! so that DataFusion executes them, see [`try_lower`].
//!
//! The native operators keep all rows in memory: they do not spill and do
//! not reserve memory in the memory pool. As in DataFusion, rows with equal
//! sort keys may come in any order.
//!
//! [`DynamicFilterPhysicalExpr`]: datafusion_physical_expr::expressions::DynamicFilterPhysicalExpr

use std::collections::VecDeque;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, RecordBatch, RecordBatchOptions, UInt32Array,
};
use arrow::compute::{SortColumn, SortOptions, concat, lexsort_to_indices, take};
use arrow::datatypes::{DataType, Schema, SchemaRef};
use arrow_row::{OwnedRow, RowConverter, SortField};
use datafusion_common::config::ConfigOptions;
use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion_common::{Result, exec_datafusion_err, exec_err, internal_err};
use datafusion_datasource::memory::MemorySourceConfig;
use datafusion_datasource::source::DataSourceExec;
use datafusion_expr::Operator;
use datafusion_physical_expr::LexOrdering;
use datafusion_physical_expr::expressions::{
    BinaryExpr, Column, DynamicFilterPhysicalExpr, IsNotNullExpr, IsNullExpr, Literal,
    NotExpr,
};
use datafusion_physical_expr::projection::update_expr;
use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion_physical_plan::execution_plan::reset_plan_states;
use datafusion_physical_plan::filter::FilterExec;
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::repartition::RepartitionExec;
use datafusion_physical_plan::sorts::sort::SortExec;
use datafusion_physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion_physical_plan::union::UnionExec;
use datafusion_physical_plan::{
    ChildrenPropertiesMode, ExecutionPlan, ExecutionPlanProperties, PhysicalExpr,
    ReplaceChildrenOptions,
};
use parking_lot::Mutex;

use crate::bridge::BridgeOperator;
use crate::chunk::{DEFAULT_VECTOR_SIZE, DataChunk};
use crate::operator::{
    EmptyState, ExecutionContext, GlobalSourceState, LocalSourceState, PhysicalOperator,
    SourceResult, downcast_state,
};

pub mod merge;
pub mod sort_sink;
pub mod topk_sink;

pub use merge::{KeyColumn, RunKeys, RunMerger, SortedRun, compares_directly};
use merge::{gather_rows_in_batches, gather_values, is_capacity_error};
pub use sort_sink::PhysicalSort;
pub use topk_sink::{FloatKeyFallback, PhysicalTopK, TopKDynamicFilter};

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
///   the native sort does. Without projections, the native sort sorts by the
///   keys of the `SortExec`, and evaluates them on all rows as DataFusion
///   does. With projections, which must project columns and literals only, it
///   sorts the projected rows (the projections are lowered with the input) by
///   the keys of the top merge.
///
/// A fetch of the `SortExec` (a TopK of every partition) and the fetches of
/// the merges keep the smallest number of rows, if that is a top-k of the
/// whole input: the merges must not keep more rows than the fetch of a sort of
/// several partitions. That is a [`PhysicalTopK`], which updates the
/// [`DynamicFilterPhysicalExpr`] of the `SortExec` for its leading sort keys
/// that are (through the projections) the leading sort keys of the
/// `SortExec`. Over a `SortExec` without fetch, whose keys DataFusion
/// evaluates on all rows, a merge with a fetch is a [`PhysicalSort`] with a
/// fetch instead if a key can fail to evaluate: a TopK discards rows before
/// it evaluates their keys.
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
/// * TopKs (sorts with a fetch) with a key that can fail to evaluate (other
///   than a column or a literal): DataFusion's TopK evaluates its keys only on
///   the batches that its threshold does not discard entirely, so which keys
///   it evaluates depends on the order of its batches;
/// * TopKs with a float key that their threshold compares, unless the native
///   TopK keeps the rows of DataFusion's TopK: as many rows (the fetch of the
///   `SortExec`), by its keys (through the projections above it). Its
///   threshold compares `-0.0 = 0.0` while it sorts `-0.0` first, so which
///   rows it returns can depend on the order of its batches, but only if the
///   last row of the result has a zero in such a key: the native TopK checks
///   that row, and executes DataFusion's TopK instead if it does (see
///   [`FloatKeyFallback`]);
/// * TopKs whose dynamic filter DataFusion pushed into a node below them (e.g.
///   a Parquet scan), if that node or a node between it and the TopK
///   evaluates an expression that can fail, is not a scan, projection,
///   filter, union, merge or repartition of partitions, or has a fetch (it
///   stops after a number of rows that pass the filter): the filter prunes
///   the rows of that node as the threshold tightens, which happens at other
///   times in DataFusion than natively, so these nodes process other rows;
/// * merges over projections whose keys, or the keys of whose sort, can fail
///   to evaluate: the native sort evaluates only the keys of the top merge;
/// * merges without projections with a key that can fail to evaluate and is
///   not the key of the sort at its position;
/// * projections between the merges and the sort that compute a value (other
///   than a column or a literal): DataFusion evaluates them lazily, on the
///   sorted rows that the merges read, which read the first batches of every
///   partition and stop with their fetch, or when a node above them stops
///   reading (e.g. a limit). The native sort projects all rows before it
///   sorts them, in chunks of its input, and a computation can fail. So can
///   a literal with 32-bit offsets (e.g. a string, repeated for every row of
///   a chunk), so such projections are left to DataFusion too;
/// * plans that DataFusion does not execute as a sort of all rows of the
///   input: nodes that advertise a different number of partitions than their
///   input has, merges of no partitions, and sorts of several partitions that
///   are not merged;
/// * sort keys that the Arrow row format does not support, and keys of unions
///   (also inside other values): the native sort orders keys by the row
///   format (unless the sort kernel sorts them), which orders unions unlike
///   DataFusion's sort, e.g. the values of each type in ascending order for a
///   descending key;
/// * rows with dictionaries with keys of fewer than 32 bits, or run-end
///   encodings with run ends of fewer than 32 bits (also in nested types): the
///   native sort materializes runs (of up to [`SORT_RUN_ROWS`] rows) and
///   chunks of more rows than DataFusion's batches, whose dictionaries can
///   need more values, and whose runs more rows, than such keys and run ends
///   address, where DataFusion's batches fit. This includes columns of the
///   `SortExec` that projections above it remove: DataFusion sorts them,
///   and whether it fails depends on its batches;
/// * sort keys (of the sort or of a merge) that evaluate to such encodings,
///   or that compute values with 32-bit offsets, also inside the key (e.g.
///   `CAST(CAST(s AS Dictionary(Int8, Utf8)) AS Utf8)`, `upper(d)` or
///   `CAST(repeat(s, 12582912) AS Utf8)`, but not a column of such values):
///   the native sort evaluates its keys on each chunk, DataFusion on its
///   batches, or on all rows of a small partition at once, so one of them
///   can fail where the other does not;
/// * TopKs of more than one row whose rows have values with 32-bit offsets
///   (`Utf8`, `Binary`, lists, maps, also inside dictionaries, run-end
///   encodings and structs), and sorts of run-end encoded values with 32-bit
///   offsets. DataFusion's TopK gathers all its rows into one batch (when it
///   emits and when it compacts them), and its sort takes chunks of
///   `batch_size` sorted rows, in which run-end encoded values can need a
///   value per row. Neither splits rows that need more than the 2 GiB that
///   32-bit offsets address, so whether DataFusion fails depends on its
///   batches; the native operators split such rows into smaller runs and
///   chunks, as DataFusion's merges do. TopKs of at most
///   [`MAX_TOPK_ROWS_WITH_OFFSETS`] rows whose values with 32-bit offsets are
///   `Utf8` or `Binary` columns of in-memory batches (`MemorySourceConfig`s)
///   that occupy less than [`MAX_TOPK_INPUT_OFFSET_BYTES`] are lowered: none
///   of their rows can need more than an array addresses;
/// * sorts without a fetch of values with 32-bit offsets, in sessions whose
///   `sort_in_place_threshold_bytes` exceeds 2 GiB: DataFusion's sort
///   concatenates the batches of a partition while they occupy fewer bytes,
///   and such a concatenation can need more than 32-bit offsets address. Up
///   to 2 GiB (1 MiB by default), its concatenations fit.
///
/// [`SORT_RUN_ROWS`]: sort_sink::SORT_RUN_ROWS
///
/// `options` are the options of the session that executes `plan`: e.g.
/// whether DataFusion's sort concatenates large batches depends on them.
/// `lower_input` lowers the input of the sort (with the projections above
/// it). Returns `Ok(None)` for any other node, without calling
/// `lower_input`.
///
/// [`DynamicFilterPhysicalExpr`]: datafusion_physical_expr::expressions::DynamicFilterPhysicalExpr
pub fn try_lower<F>(
    plan: &Arc<dyn ExecutionPlan>,
    options: &ConfigOptions,
    lower_input: F,
) -> Result<Option<Arc<dyn PhysicalOperator>>>
where
    F: FnMut(&Arc<dyn ExecutionPlan>) -> Result<Arc<dyn PhysicalOperator>>,
{
    let Some(sort) = SortPlan::try_new(plan, options)? else {
        return Ok(None);
    };
    let operator: Arc<dyn PhysicalOperator> = match sort.topk {
        false => Arc::new(sort.into_sort(plan, lower_input)?),
        true => Arc::new(sort.into_topk(plan, lower_input)?),
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
    /// Whether the native operator is a [`PhysicalTopK`] (with `fetch`), or a
    /// [`PhysicalSort`] (with `fetch`, if any).
    topk: bool,
    /// The rows to sort: the input of the `SortExec` (or the input of the
    /// `CoalescePartitionsExec` it sorts), with the projections between the
    /// top merge and the `SortExec` above it.
    input: Arc<dyn ExecutionPlan>,
    /// The dynamic filter of the `SortExec`, if the native operator can
    /// update it: with the expressions of the `SortExec`'s input that its
    /// leading sort keys correspond to.
    dynamic_filter: Option<TopKDynamicFilter>,
    /// For a TopK whose threshold compares floats, DataFusion's TopK (the
    /// plan), which the native TopK runs if its result may differ.
    float_key_fallback: Option<FloatKeyFallback>,
}

impl SortPlan {
    /// Analyze `plan`, executed with `options`, see [`try_lower`].
    pub(crate) fn try_new(
        plan: &Arc<dyn ExecutionPlan>,
        options: &ConfigOptions,
    ) -> Result<Option<Self>> {
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

        // DataFusion materializes some rows of the sort in single arrays,
        // without splitting rows that do not fit (32-bit offsets address 2 GiB
        // of values): its TopK gathers all its rows into one batch, also while
        // it compacts them, and its sort takes chunks of `batch_size` sorted
        // rows of each run, in which run-end encoded values can need a value
        // per row. Whether it fails depends on its batches; the native
        // operators split such rows, so these sorts are left to DataFusion,
        // unless the rows of the TopK cannot need that many bytes (see
        // `topk_rows_fit_into_arrays`).
        let rows = sort.schema();
        let types = || rows.fields().iter().map(|field| field.data_type());
        let unsplit = match sort.fetch() {
            Some(fetch) => {
                fetch > 1
                    && types().any(has_32_bit_offsets)
                    && !topk_rows_fit_into_arrays(sort, fetch)
            }
            None => types().any(has_run_end_encoded_offsets),
        };
        if unsplit {
            return Ok(None);
        }
        // DataFusion's sort also concatenates the batches of a partition (or
        // groups of them) while they occupy fewer than
        // `sort_in_place_threshold_bytes` (their memory and the bytes of their
        // rows). Above 2 GiB, a concatenation of values with 32-bit offsets
        // can need more than those address, and fails.
        let threshold = options.execution.sort_in_place_threshold_bytes;
        if sort.fetch().is_none()
            && threshold > i32::MAX as usize
            && types().any(has_32_bit_offsets)
        {
            return Ok(None);
        }

        // A merge produces all rows in its order only if every partition of
        // its input is ordered by it.
        for merge in &merges {
            if !merge
                .input()
                .equivalence_properties()
                .ordering_satisfy(merge.expr().clone())?
            {
                return Ok(None);
            }
        }
        // DataFusion evaluates the keys of the sort on all rows, and the keys of
        // the merges on the rows they read. Without projections, the native
        // sort sorts by the keys of the sort (whose order satisfies the order of
        // the merges): it evaluates them on all rows too, and a key of a merge
        // must be the key of the sort at its position, or not fail. With
        // projections, it sorts the projected rows by the keys of the top merge,
        // and evaluates neither the keys of the sort nor those of the other
        // merges: none of them may fail.
        let projected = chain.iter().any(|node| node.is::<ProjectionExec>());
        let ordering = if let Some(top) = merges.first().filter(|_| projected) {
            if !cannot_fail(sort.expr())
                || merges.iter().any(|merge| !cannot_fail(merge.expr()))
            {
                return Ok(None);
            }
            top.expr().clone()
        } else {
            let keys_of_the_sort = merges.iter().all(|merge| {
                merge.expr().iter().enumerate().all(|(index, key)| {
                    cannot_fail_to_evaluate(&key.expr)
                        || sort.expr().get(index).is_some_and(|sort_key| {
                            sort_key.expr.as_ref() == key.expr.as_ref()
                        })
                })
            });
            if !keys_of_the_sort {
                return Ok(None);
            }
            sort.expr().clone()
        };

        // DataFusion's TopK (a sort with a fetch) evaluates its keys only on
        // the batches that its threshold does not discard entirely: which keys
        // it evaluates depends on the order of its batches, which the native
        // operators do not reproduce. Such TopKs are left to DataFusion if a
        // key can fail.
        if sort.fetch().is_some() && !cannot_fail(sort.expr()) {
            return Ok(None);
        }
        // Its threshold also compares floats with `-0.0 = 0.0`, while it sorts
        // `-0.0` first: once its threshold is a zero, it discards the zeros of
        // the other sign that come later, also those that it sorts before the
        // threshold. That changes its result only if the last row of the
        // result has a zero in a float key that the threshold compares (see
        // `FloatKeyFallback`), which the native TopK checks: then it runs
        // DataFusion's TopK instead. So its last row must be the last row of
        // DataFusion's TopK: it keeps as many rows, by the keys of the sort
        // (through the projections, see below).
        let float_keys = match sort.fetch() {
            Some(_) => compared_float_keys(sort.expr(), &sort_input.schema())?,
            None => vec![],
        };
        if !float_keys.is_empty() && fetch != sort.fetch() {
            return Ok(None);
        }
        // The dynamic filter of DataFusion's TopK prunes the rows of the nodes
        // that consume it as the threshold tightens, at other times than the
        // native TopK does: such TopKs are left to DataFusion if the work on
        // the pruned rows can fail.
        if sort.fetch().is_some()
            && let Some(filter_id) = sort
                .dynamic_expressions_produced()
                .iter()
                .find_map(|expr| expr.expression_id())
            && pruned_work_cannot_fail(sort_input, filter_id)? == Some(false)
        {
            return Ok(None);
        }
        // Over a sort without fetch, a merge with a fetch reads the sorted rows
        // of all of the input, whose keys are all evaluated: the native operator
        // is a TopK, which discards rows before it evaluates their keys, only if
        // its keys cannot fail. Otherwise it is a sort with a fetch.
        let topk = fetch.is_some() && cannot_fail(&ordering);

        // DataFusion evaluates the projections between the merges and the sort
        // lazily, on the sorted rows that are read: the merges read the first
        // batches of every partition, and stop with their fetch or when a node
        // above them (e.g. a limit) stops reading. The native sort projects all
        // rows (in chunks of its input) before it sorts them, which only makes
        // no difference for projections that cannot fail (of columns and
        // literals), and that compute no values of which an array holds only
        // so many (e.g. a string literal, repeated for every row).
        let projections = chain
            .iter()
            .filter_map(|node| node.downcast_ref::<ProjectionExec>())
            .collect::<Vec<_>>();
        if !float_keys.is_empty() && !orders_like_the_sort(sort, &ordering, &projections)?
        {
            return Ok(None);
        }
        for projection in &projections {
            let schema = projection.input().schema();
            for expr in projection.expr() {
                if !cannot_fail_to_evaluate(&expr.expr)
                    || computes_limited_values(&expr.expr, &schema)?
                {
                    return Ok(None);
                }
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
        for projection in chain
            .iter()
            .rev()
            .filter(|node| node.is::<ProjectionExec>())
        {
            input = Arc::clone(projection).replace_children(
                vec![input],
                ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
            )?;
        }
        if !SortKeys::supports(&ordering, &input.schema())? {
            return Ok(None);
        }
        // The native sort materializes runs (and chunks) of more rows than the
        // batches of DataFusion's sort, so their dictionaries can need more
        // values than narrow keys address, and their run-end encodings longer
        // runs than narrow run ends address, where DataFusion's batches fit.
        // DataFusion's sort materializes the rows of the `SortExec`, also the
        // columns that projections above it remove (which the native sort
        // projects before it sorts), and whether it fails depends on its
        // batches too.
        let narrow = |schema: SchemaRef| {
            schema
                .fields()
                .iter()
                .any(|field| has_narrow_encoding(field.data_type()))
        };
        if narrow(sort.schema()) || narrow(input.schema()) {
            return Ok(None);
        }
        // For the same reason, keys that compute values of which an array holds
        // only so many: the native sort evaluates its keys on each chunk, while
        // DataFusion evaluates the keys of its sort on its batches, or on all
        // rows of a small partition at once (which it sorts together), and the
        // keys of its merges on their batches. And keys of unions (also inside
        // other values): the native sort orders them by the row format, which
        // orders unions unlike DataFusion's sort (`make_comparator`), e.g. the
        // values of a type in ascending order for a descending key.
        let keys = [
            (&ordering, input.schema()),
            (sort.expr(), sort_input.schema()),
        ]
        .into_iter()
        .chain(
            merges
                .iter()
                .map(|merge| (merge.expr(), merge.input().schema())),
        );
        for (ordering, schema) in keys {
            for sort_expr in ordering.iter() {
                if computes_limited_values(&sort_expr.expr, &schema)?
                    || has_union(&sort_expr.expr.data_type(&schema)?)
                {
                    return Ok(None);
                }
            }
        }

        let dynamic_filter = match topk {
            true => dynamic_filter(sort, &ordering, &projections)?,
            false => None,
        };
        let float_key_fallback = (topk && !float_keys.is_empty())
            .then(|| FloatKeyFallback::new(Arc::clone(plan), float_keys));
        Ok(Some(Self {
            ordering,
            fetch,
            topk,
            input,
            dynamic_filter,
            float_key_fallback,
        }))
    }

    /// Whether the native operator is a [`PhysicalTopK`] rather than a
    /// [`PhysicalSort`].
    pub(crate) fn is_topk(&self) -> bool {
        self.topk
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

    /// Lower the sort of `plan`, a TopK, to a [`PhysicalTopK`].
    pub(crate) fn into_topk<F>(
        self,
        plan: &Arc<dyn ExecutionPlan>,
        lower_input: F,
    ) -> Result<PhysicalTopK>
    where
        F: FnMut(&Arc<dyn ExecutionPlan>) -> Result<Arc<dyn PhysicalOperator>>,
    {
        let (true, Some(fetch)) = (self.topk, self.fetch) else {
            return internal_err!("The sort is not a TopK");
        };
        let input = self.lower_input(plan, lower_input)?;
        let mut topk = PhysicalTopK::try_new(input, self.ordering, fetch)?;
        if let Some(filter) = self.dynamic_filter {
            topk = topk.with_dynamic_filter(filter)?;
        }
        if let Some(fallback) = self.float_key_fallback {
            topk = topk.with_float_key_fallback(fallback)?;
        }
        Ok(topk)
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
    projections: &[&ProjectionExec],
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

/// Whether the sort keys `ordering` of the rows that `projections` (from the
/// top) project from the rows of `sort` are the sort keys of `sort` (through
/// the projections), all of them, with the same options: whether they order
/// the rows exactly like `sort`.
fn orders_like_the_sort(
    sort: &SortExec,
    ordering: &LexOrdering,
    projections: &[&ProjectionExec],
) -> Result<bool> {
    if ordering.len() != sort.expr().len() {
        return Ok(false);
    }
    for (sort_expr, sort_key) in ordering.iter().zip(sort.expr().iter()) {
        let mut expr = Some(Arc::clone(&sort_expr.expr));
        for projection in projections {
            expr = match expr {
                Some(expr) => update_expr(&expr, projection.expr(), true)?,
                None => None,
            };
        }
        let same = expr.is_some_and(|expr| expr.as_ref() == sort_key.expr.as_ref());
        if !same || sort_expr.options != sort_key.options {
            return Ok(false);
        }
    }
    Ok(true)
}

/// Whether `expr` cannot fail to evaluate: a column, a literal, a dynamic
/// filter (which compares columns with literals), or a comparison, boolean
/// combination, negation or null check of such expressions.
fn cannot_fail_to_evaluate(expr: &Arc<dyn PhysicalExpr>) -> bool {
    if expr.is::<Column>()
        || expr.is::<Literal>()
        || expr.is::<DynamicFilterPhysicalExpr>()
    {
        return true;
    }
    let infallible = match expr.downcast_ref::<BinaryExpr>() {
        Some(binary) => matches!(
            binary.op(),
            Operator::Eq
                | Operator::NotEq
                | Operator::Lt
                | Operator::LtEq
                | Operator::Gt
                | Operator::GtEq
                | Operator::IsDistinctFrom
                | Operator::IsNotDistinctFrom
                | Operator::And
                | Operator::Or
        ),
        None => {
            expr.is::<NotExpr>() || expr.is::<IsNullExpr>() || expr.is::<IsNotNullExpr>()
        }
    };
    infallible && expr.children().into_iter().all(cannot_fail_to_evaluate)
}

/// How the dynamic filter with id `filter_id` of a TopK whose input is `plan`
/// affects `plan`: the nodes that consume the filter (e.g. a Parquet scan)
/// skip the rows that it rejects, so which rows they, and the nodes between
/// them and the TopK, process depends on when the threshold of the TopK
/// tightens, which differs between DataFusion and the native TopK.
///
/// Returns `None` if no node of `plan` consumes the filter, `Some(true)` if
/// the nodes that consume it and the nodes above them only pass rows through
/// (scans, projections, filters, unions, merges and repartitions of
/// partitions, without a fetch: a node with a fetch stops after a number of
/// rows that pass the filter) and only evaluate expressions that cannot
/// fail, so that skipping rows makes no difference but in speed, and
/// `Some(false)` otherwise.
fn pruned_work_cannot_fail(
    plan: &Arc<dyn ExecutionPlan>,
    filter_id: u64,
) -> Result<Option<bool>> {
    let mut consumes = false;
    let mut cannot_fail = true;
    plan.apply_expressions(&mut |expr| {
        expr.apply(|expr| {
            consumes |= expr.expression_id() == Some(filter_id);
            Ok(TreeNodeRecursion::Continue)
        })?;
        cannot_fail &= cannot_fail_to_evaluate(expr);
        Ok(TreeNodeRecursion::Continue)
    })?;
    let mut below = None;
    for child in plan.children() {
        if let Some(safe) = pruned_work_cannot_fail(child, filter_id)? {
            below = Some(below.unwrap_or(true) && safe);
        }
    }
    if !consumes && below.is_none() {
        return Ok(None);
    }
    let passes_rows = (plan.is::<DataSourceExec>()
        || plan.is::<ProjectionExec>()
        || plan.is::<FilterExec>()
        || plan.is::<UnionExec>()
        || plan.is::<CoalescePartitionsExec>()
        || plan.is::<RepartitionExec>())
        && plan.fetch().is_none();
    Ok(Some(below.unwrap_or(true) && passes_rows && cannot_fail))
}

/// Whether no key of `ordering` can fail to evaluate.
fn cannot_fail(ordering: &LexOrdering) -> bool {
    ordering
        .iter()
        .all(|sort_expr| cannot_fail_to_evaluate(&sort_expr.expr))
}

/// The positions of the sort keys `ordering` (of rows with schema `schema`)
/// that the threshold of DataFusion's TopK compares and that are floats: the
/// keys with floats before the first nested key (at which it stops
/// comparing).
fn compared_float_keys(ordering: &LexOrdering, schema: &Schema) -> Result<Vec<usize>> {
    let mut keys = vec![];
    for (index, data_type) in data_types(ordering, schema)?.iter().enumerate() {
        if data_type.is_nested() {
            break;
        }
        if has_floats(data_type) {
            keys.push(index);
        }
    }
    Ok(keys)
}

/// The most rows of a TopK whose rows have `Utf8` or `Binary` values that the
/// native operators lower (see [`try_lower`]).
pub const MAX_TOPK_ROWS_WITH_OFFSETS: usize = 65_536;

/// The most bytes of the `Utf8` and `Binary` values of the in-memory input of
/// a TopK whose rows have such values that the native operators lower (see
/// [`try_lower`]): half of what their 32-bit offsets address.
pub const MAX_TOPK_INPUT_OFFSET_BYTES: usize = 1 << 30;

/// Whether the rows of `sort`, DataFusion's TopK of `fetch` rows, cannot
/// need more bytes than the 32-bit offsets of a single array address, when
/// DataFusion gathers them into one batch: its rows have values with 32-bit
/// offsets only in `Utf8` and `Binary` columns (not inside other values,
/// which can repeat child values, nor run-end encoded or in dictionaries,
/// whose values a row can repeat), it keeps at most
/// [`MAX_TOPK_ROWS_WITH_OFFSETS`] rows, and its rows are rows of in-memory
/// batches whose `Utf8` and `Binary` values occupy less than
/// [`MAX_TOPK_INPUT_OFFSET_BYTES`] (see [`in_memory_offset_bytes`]): the
/// values of any of its rows fit into an array.
fn topk_rows_fit_into_arrays(sort: &SortExec, fetch: usize) -> bool {
    let offsets_in_columns_only = sort.schema().fields().iter().all(|field| {
        let data_type = field.data_type();
        !has_32_bit_offsets(data_type)
            || matches!(data_type, DataType::Utf8 | DataType::Binary)
    });
    fetch <= MAX_TOPK_ROWS_WITH_OFFSETS
        && offsets_in_columns_only
        && in_memory_offset_bytes(sort.input())
            .is_some_and(|bytes| bytes < MAX_TOPK_INPUT_OFFSET_BYTES)
}

/// The bytes of the `Utf8` and `Binary` values of the rows of `plan`, if
/// these are values of in-memory batches: `plan` scans
/// `MemorySourceConfig`s, the bytes of the `Utf8` and `Binary` values of
/// whose batches it returns (as a concatenation of each column of a batch
/// would need them, also if batches share their buffers), and filters,
/// repartitions, merges or unions their rows, or projects their columns
/// (without computing values with 32-bit offsets). Every array of
/// such values of the rows of `plan` has at most that many bytes. `None`
/// for any other plan.
fn in_memory_offset_bytes(plan: &Arc<dyn ExecutionPlan>) -> Option<usize> {
    if let Some(scan) = plan.downcast_ref::<DataSourceExec>() {
        let config = scan.data_source().downcast_ref::<MemorySourceConfig>()?;
        let mut bytes = 0_usize;
        for batch in config.partitions().iter().flatten() {
            let columns = match config.projection() {
                Some(projection) => projection
                    .iter()
                    .map(|&index| batch.columns().get(index))
                    .collect::<Option<Vec<_>>>()?,
                None => batch.columns().iter().collect(),
            };
            for column in columns {
                bytes = bytes.checked_add(offset_value_bytes(column.as_ref()))?;
            }
        }
        return Some(bytes);
    }
    if let Some(projection) = plan.downcast_ref::<ProjectionExec>() {
        // A projection must not compute values with 32-bit offsets, which may
        // need more bytes than its input.
        let schema = projection.input().schema();
        for expr in projection.expr() {
            if !expr.expr.is::<Column>()
                && has_32_bit_offsets(&expr.expr.data_type(&schema).ok()?)
            {
                return None;
            }
        }
    } else if !(plan.is::<FilterExec>()
        || plan.is::<RepartitionExec>()
        || plan.is::<CoalescePartitionsExec>()
        || plan.is::<UnionExec>())
    {
        return None;
    }
    let mut bytes = 0_usize;
    for child in plan.children() {
        bytes = bytes.checked_add(in_memory_offset_bytes(child)?)?;
    }
    Some(bytes)
}

/// Whether values of `data_type` (or of its children) are dictionary encoded
/// with keys of fewer than 32 bits, or run-end encoded with run ends of fewer
/// than 32 bits: an array of those addresses at most 65536 dictionary values
/// or 32767 rows.
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

/// Whether values of `data_type` (or of its children, dictionary values or
/// run-end encoded values) have 32-bit offsets, which address at most 2 GiB
/// of values (or 2^31 child values) in one array.
fn has_32_bit_offsets(data_type: &DataType) -> bool {
    match data_type {
        DataType::Utf8
        | DataType::Binary
        | DataType::List(_)
        | DataType::ListView(_)
        | DataType::Map(_, _) => true,
        DataType::Dictionary(_, values) => has_32_bit_offsets(values),
        DataType::RunEndEncoded(_, values) => has_32_bit_offsets(values.data_type()),
        DataType::LargeList(field)
        | DataType::LargeListView(field)
        | DataType::FixedSizeList(field, _) => has_32_bit_offsets(field.data_type()),
        DataType::Struct(fields) => fields
            .iter()
            .any(|field| has_32_bit_offsets(field.data_type())),
        DataType::Union(fields, _) => fields
            .iter()
            .any(|(_, field)| has_32_bit_offsets(field.data_type())),
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

/// Whether values of `data_type` (or of its children) are run-end encoded
/// with values that have 32-bit offsets (see [`has_32_bit_offsets`]): rows of
/// such an array taken in another order can need a value per row.
fn has_run_end_encoded_offsets(data_type: &DataType) -> bool {
    match data_type {
        DataType::RunEndEncoded(_, values) => has_32_bit_offsets(values.data_type()),
        DataType::Dictionary(_, values) => has_run_end_encoded_offsets(values),
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::ListView(field)
        | DataType::LargeListView(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => has_run_end_encoded_offsets(field.data_type()),
        DataType::Struct(fields) => fields
            .iter()
            .any(|field| has_run_end_encoded_offsets(field.data_type())),
        DataType::Union(fields, _) => fields
            .iter()
            .any(|(_, field)| has_run_end_encoded_offsets(field.data_type())),
        _ => false,
    }
}

/// Whether evaluating `expr` on rows with schema `schema` computes values of
/// which an array holds only so many: values with a narrow encoding (see
/// [`has_narrow_encoding`]), or values with 32-bit offsets (see
/// [`has_32_bit_offsets`]) that `expr` or an expression in it computes, such
/// as `CAST(repeat(s, 12582912) AS Utf8)` (2 GiB in 171 rows) or a literal
/// string (repeated for every row). A column computes nothing: its values
/// are those of its input. Whether evaluating such an expression fails
/// depends on how many rows it is evaluated on at once.
fn computes_limited_values(
    expr: &Arc<dyn PhysicalExpr>,
    schema: &Schema,
) -> Result<bool> {
    let data_type = expr.data_type(schema)?;
    if has_narrow_encoding(&data_type)
        || (!expr.is::<Column>() && has_32_bit_offsets(&data_type))
    {
        return Ok(true);
    }
    for child in expr.children() {
        if computes_limited_values(child, schema)? {
            return Ok(true);
        }
    }
    Ok(false)
}

/// Whether the values of `data_type` are floats (possibly dictionary or
/// run-end encoded).
fn has_floats(data_type: &DataType) -> bool {
    match data_type {
        DataType::Float16 | DataType::Float32 | DataType::Float64 => true,
        DataType::Dictionary(_, values) => has_floats(values),
        DataType::RunEndEncoded(_, values) => has_floats(values.data_type()),
        _ => false,
    }
}

/// The sort keys of a native sort: evaluates the sort expressions and sorts
/// rows into [`SortedRun`]s.
///
/// A single sort key of a type that [`compares_directly`] (e.g. an integer,
/// float, decimal, date, timestamp or string) is sorted with the sort kernel
/// (`lexsort_to_indices`), and its runs hold the sorted key values
/// ([`RunKeys::Column`]), which the merge compares directly: the keys are
/// never encoded. Other sort keys are encoded in the Arrow row format, with
/// the same [`RowConverter`] for all runs, so that they can be merged.
#[derive(Debug)]
pub(crate) struct SortKeys {
    ordering: LexOrdering,
    converter: RowConverter,
    /// A single sort key that [`compares_directly`]: sorted with
    /// `lexsort_to_indices`, which orders it exactly like the row format, and
    /// compared directly in the runs.
    direct: bool,
    /// The column of the rows that the single direct sort key is, if it is
    /// one: the runs take their keys from their rows.
    key_column: Option<usize>,
}

/// The sorted keys of the rows of [`SortKeys::sort`].
enum SortedKeys {
    /// Encoded rows.
    Rows(arrow_row::Rows),
    /// The values of the single sort key.
    Values(ArrayRef),
    /// The single sort key is this column of the rows.
    Column(usize),
    /// The values of the single sort key, which do not fit into one array:
    /// gathered with the rows of every run.
    Gather,
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
        let direct =
            matches!(types.as_slice(), [data_type] if compares_directly(data_type));
        let key_column = ordering
            .first()
            .expr
            .downcast_ref::<Column>()
            .map(Column::index)
            .filter(|_| direct);
        Ok(Self {
            ordering,
            converter,
            direct,
            key_column,
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

    /// The sort key of row `row` of `run` (a run of these sort keys), encoded
    /// in the Arrow row format.
    pub(crate) fn encoded_key(&self, run: &SortedRun, row: usize) -> Result<OwnedRow> {
        match run.keys() {
            RunKeys::Rows(rows) => Ok(rows.row(row).owned()),
            RunKeys::Column(column) => {
                let values = column.values().slice(row, 1);
                Ok(self.converter.convert_columns(&[values])?.row(0).owned())
            }
        }
    }

    /// Sort the rows of `batches` into runs with schema `schema`, keeping
    /// only the first `limit` rows (all rows if `None`). The runs hold
    /// consecutive rows of the sort, in order: a single run, unless the rows
    /// do not fit into one batch (see [`gather_rows_in_batches`]). Returns no
    /// run if there are no rows to keep.
    pub(crate) fn sort(
        &self,
        schema: &SchemaRef,
        batches: &[RecordBatch],
        limit: Option<usize>,
    ) -> Result<Vec<SortedRun>> {
        let batches = batches
            .iter()
            .filter(|batch| batch.num_rows() > 0)
            .collect::<Vec<_>>();
        let num_rows = batches.iter().map(|batch| batch.num_rows()).sum::<usize>();
        let len = limit.map_or(num_rows, |limit| limit.min(num_rows));
        if len == 0 {
            return Ok(vec![]);
        }
        if u32::try_from(num_rows).is_err() {
            return exec_err!("Cannot sort {num_rows} rows at once");
        }
        let limit = (len < num_rows).then_some(len);
        let key_values = batches
            .iter()
            .map(|batch| self.evaluate(batch))
            .collect::<Result<Vec<_>>>()?;

        // The positions of the sorted rows in the concatenation of `batches`,
        // and the sorted keys.
        let (order, sorted_keys) = match self.direct {
            true => match self.sort_with_kernel(&key_values, limit)? {
                Some((indices, key)) => {
                    let sorted_keys = match self.key_column {
                        Some(column) => SortedKeys::Column(column),
                        None => SortedKeys::Values(take(key.as_ref(), &indices, None)?),
                    };
                    (indices.values().to_vec(), sorted_keys)
                }
                None => {
                    let (order, _) = self.order_by_rows(&key_values, num_rows, len)?;
                    let sorted_keys = match self.key_column {
                        Some(column) => SortedKeys::Column(column),
                        None => SortedKeys::Gather,
                    };
                    (order, sorted_keys)
                }
            },
            false => {
                let (order, rows) = self.order_by_rows(&key_values, num_rows, len)?;
                let data_len = order.iter().map(|&i| rows.row_len(i as usize)).sum();
                let mut sorted_rows = self.converter.empty_rows(len, data_len);
                for &i in &order {
                    sorted_rows.push(rows.row(i as usize));
                }
                (order, SortedKeys::Rows(sorted_rows))
            }
        };

        // The batch and row of every sorted row.
        let mut starts = Vec::with_capacity(batches.len());
        let mut start = 0;
        for batch in &batches {
            starts.push(start);
            start += batch.num_rows();
        }
        let positions = order
            .iter()
            .map(|&index| {
                let index = index as usize;
                let batch = starts.partition_point(|&start| start <= index) - 1;
                (batch, index - starts[batch])
            })
            .collect::<Vec<_>>();
        let gathered = gather_rows_in_batches(schema, &batches, &positions)?;
        let sorted_keys = match sorted_keys {
            SortedKeys::Rows(rows) if gathered.len() == 1 => {
                return gathered
                    .into_iter()
                    .zip([rows])
                    .map(|(batch, rows)| SortedRun::try_new(batch, rows))
                    .collect();
            }
            sorted_keys => sorted_keys,
        };
        let options = self.ordering[0].options;
        let mut runs = Vec::with_capacity(gathered.len());
        let mut start = 0;
        for mut batch in gathered {
            let rows = start..start + batch.num_rows();
            start = rows.end;
            let keys = match &sorted_keys {
                // Split the keys like the rows.
                SortedKeys::Rows(sorted_rows) => {
                    let data_len = rows.clone().map(|row| sorted_rows.row_len(row)).sum();
                    let mut keys = self.converter.empty_rows(rows.len(), data_len);
                    for row in rows {
                        keys.push(sorted_rows.row(row));
                    }
                    RunKeys::Rows(keys)
                }
                SortedKeys::Values(values) => {
                    let values = compact_views(&values.slice(rows.start, rows.len()));
                    RunKeys::Column(KeyColumn::try_new(values, options)?)
                }
                SortedKeys::Column(column) => {
                    // The rows keep the compacted views too, which release the
                    // buffers of the unsorted rows.
                    let values = compact_views(batch.column(*column));
                    if !Arc::ptr_eq(&values, batch.column(*column)) {
                        let mut columns = batch.columns().to_vec();
                        columns[*column] = Arc::clone(&values);
                        let batch_options = RecordBatchOptions::new()
                            .with_row_count(Some(batch.num_rows()));
                        batch = RecordBatch::try_new_with_options(
                            batch.schema(),
                            columns,
                            &batch_options,
                        )?;
                    }
                    RunKeys::Column(KeyColumn::try_new(values, options)?)
                }
                SortedKeys::Gather => {
                    let values = key_values
                        .iter()
                        .map(|keys| keys[0].as_ref())
                        .collect::<Vec<_>>();
                    let positions = &positions[rows];
                    runs_with_gathered_keys(
                        &mut runs, batch, positions, &values, options,
                    )?;
                    continue;
                }
            };
            runs.push(SortedRun::try_new_with_keys(batch, keys)?);
        }
        Ok(runs)
    }

    /// The positions of the first `len` rows (of `num_rows`) in the
    /// concatenation of the batches whose sort keys are `keys`, sorted by
    /// their keys encoded in the row format, and the encoded keys of all
    /// rows.
    fn order_by_rows(
        &self,
        keys: &[Vec<ArrayRef>],
        num_rows: usize,
        len: usize,
    ) -> Result<(Vec<u32>, arrow_row::Rows)> {
        let mut rows = self.converter.empty_rows(num_rows, 0);
        for keys in keys {
            self.converter.append(&mut rows, keys)?;
        }
        let num_rows_u32 = u32::try_from(num_rows)
            .map_err(|_| exec_datafusion_err!("Cannot sort {num_rows} rows at once"))?;
        let mut order = (0..num_rows_u32).collect::<Vec<_>>();
        let compare =
            |a: &u32, b: &u32| rows.row(*a as usize).cmp(&rows.row(*b as usize));
        if len < num_rows {
            order.select_nth_unstable_by(len - 1, compare);
            order.truncate(len);
        }
        order.sort_unstable_by(compare);
        Ok((order, rows))
    }

    /// Sort the single sort key `keys` (of every batch) with
    /// `lexsort_to_indices`: the positions of the first `limit` rows in the
    /// concatenation of the batches, and the concatenated keys. `None` if
    /// the keys of all batches do not fit into one array (then they are sorted
    /// by their encoded rows, which have no such limit).
    fn sort_with_kernel(
        &self,
        keys: &[Vec<ArrayRef>],
        limit: Option<usize>,
    ) -> Result<Option<(UInt32Array, ArrayRef)>> {
        let key = match keys {
            [keys] => Arc::clone(&keys[0]),
            _ => {
                let keys = keys.iter().map(|keys| keys[0].as_ref()).collect::<Vec<_>>();
                // `concat` would only fail after copying the values.
                let bytes = keys
                    .iter()
                    .map(|key| offset_value_bytes(*key))
                    .sum::<usize>();
                if bytes > i32::MAX as usize {
                    return Ok(None);
                }
                match concat(&keys) {
                    Ok(key) => key,
                    Err(error) if is_capacity_error(&error) => return Ok(None),
                    Err(error) => return Err(error.into()),
                }
            }
        };
        let column = SortColumn {
            values: Arc::clone(&key),
            options: Some(self.ordering[0].options),
        };
        let indices = lexsort_to_indices(&[column], limit)?;
        Ok(Some((indices, key)))
    }
}

/// The sorted key values `values` of a run, with the data of their views
/// (of `Utf8View` and `BinaryView` values) copied into compact buffers, in
/// order: the merge compares the keys of a run in order, so it reads them
/// from consecutive memory, rather than from the buffers of the unsorted
/// rows, in random order. Other arrays are returned as they are.
fn compact_views(values: &ArrayRef) -> ArrayRef {
    match values.data_type() {
        DataType::Utf8View => Arc::new(values.as_string_view().gc()),
        DataType::BinaryView => Arc::new(values.as_binary_view().gc()),
        _ => Arc::clone(values),
    }
}

/// Add the runs of the sorted rows `batch`, the rows `positions` (`(batch,
/// row)`) of batches whose values of a single sort key (with `options`) are
/// `keys`, to `runs`: the rows with their key values, gathered at the same
/// positions, in a single run, or in several runs if the key values do not
/// fit into one array.
fn runs_with_gathered_keys(
    runs: &mut Vec<SortedRun>,
    batch: RecordBatch,
    positions: &[(usize, usize)],
    keys: &[&dyn Array],
    options: SortOptions,
) -> Result<()> {
    match gather_values(keys, positions) {
        Ok(values) => {
            let keys =
                RunKeys::Column(KeyColumn::try_new(compact_views(&values), options)?);
            runs.push(SortedRun::try_new_with_keys(batch, keys)?);
        }
        Err(error) if positions.len() > 1 && is_capacity_error(&error) => {
            let half = positions.len().div_ceil(2);
            let (first, second) = positions.split_at(half);
            runs_with_gathered_keys(runs, batch.slice(0, half), first, keys, options)?;
            let rest = batch.slice(half, second.len());
            runs_with_gathered_keys(runs, rest, second, keys, options)?;
        }
        Err(error) => return Err(error.into()),
    }
    Ok(())
}

/// The bytes of the values of `array` if it is a `Utf8` or `Binary` array,
/// whose 32-bit offsets address at most `i32::MAX` bytes of values (0 for
/// other arrays).
fn offset_value_bytes(array: &dyn Array) -> usize {
    let offsets = match array.data_type() {
        DataType::Utf8 => array.as_string::<i32>().offsets(),
        DataType::Binary => array.as_binary::<i32>().offsets(),
        _ => return 0,
    };
    (offsets.last() - offsets.first()) as usize
}

/// The size of the memory that the rows of `batch` occupy: the bytes they
/// would take in new arrays (more for zero-copy slices of view arrays).
pub(crate) fn memory_size(batch: &RecordBatch) -> usize {
    batch
        .columns()
        .iter()
        .map(|column| {
            column
                .to_data()
                .get_slice_memory_size()
                .unwrap_or_else(|_| column.get_array_memory_size())
        })
        .sum()
}

/// The data types of the sort keys `ordering` of rows with schema `schema`.
fn data_types(ordering: &LexOrdering, schema: &Schema) -> Result<Vec<DataType>> {
    ordering
        .iter()
        .map(|sort_expr| sort_expr.expr.data_type(schema))
        .collect()
}

/// The sorted output of a [`PhysicalSort`] or [`PhysicalTopK`], set up when
/// the sink is finalized and handed to the source.
#[derive(Debug, Default)]
pub(crate) struct SortedOutput {
    output: Mutex<Option<Output>>,
}

/// What the source of a [`PhysicalSort`] or [`PhysicalTopK`] emits.
#[derive(Debug)]
enum Output {
    /// The merge of the runs that the sink collected.
    Merger(RunMerger),
    /// The merged rows, in chunks of at most [`DEFAULT_VECTOR_SIZE`] rows.
    Batches(VecDeque<RecordBatch>),
    /// The output of a DataFusion plan, see [`FloatKeyFallback`].
    Plan(Arc<dyn ExecutionPlan>),
}

impl SortedOutput {
    /// Set the merger of the runs collected by the sink.
    pub(crate) fn set(&self, merger: RunMerger) {
        *self.output.lock() = Some(Output::Merger(merger));
    }

    /// Set the merged rows, in chunks of at most [`DEFAULT_VECTOR_SIZE`] rows.
    pub(crate) fn set_batches(&self, batches: VecDeque<RecordBatch>) {
        *self.output.lock() = Some(Output::Batches(batches));
    }

    /// Emit the output of the DataFusion plan `plan` instead.
    pub(crate) fn set_plan(&self, plan: Arc<dyn ExecutionPlan>) {
        *self.output.lock() = Some(Output::Plan(plan));
    }

    /// The global source state of the operator `name`, which takes over the
    /// output: the merger, the merged rows, or the execution of a plan (with
    /// a fresh state, see `reset_plan_states`).
    pub(crate) fn take_source(
        &self,
        ctx: &ExecutionContext,
        name: &str,
    ) -> Result<Arc<dyn GlobalSourceState>> {
        let source = match self.output.lock().take() {
            Some(Output::Merger(merger)) => SortedSource::Merger(Mutex::new(merger)),
            Some(Output::Batches(batches)) => SortedSource::Batches(Mutex::new(batches)),
            Some(Output::Plan(plan)) => {
                let bridge = BridgeOperator::new(reset_plan_states(plan)?);
                let state = bridge.get_global_source_state(ctx)?;
                SortedSource::Plan { bridge, state }
            }
            None => return internal_err!("{name} is read before its sink is finalized"),
        };
        Ok(Arc::new(source))
    }
}

/// The global source state of a [`PhysicalSort`] or [`PhysicalTopK`]. The
/// source is not parallel, so a single thread reads it.
enum SortedSource {
    Merger(Mutex<RunMerger>),
    Batches(Mutex<VecDeque<RecordBatch>>),
    /// A DataFusion plan, executed as by a [`BridgeOperator`], with its
    /// global source state.
    Plan {
        bridge: BridgeOperator,
        state: Arc<dyn GlobalSourceState>,
    },
}

impl GlobalSourceState for SortedSource {}

/// The local source state of the source state `gstate` of the operator
/// `name`.
pub(crate) fn sorted_local_source_state(
    ctx: &ExecutionContext,
    gstate: &dyn GlobalSourceState,
    name: &str,
) -> Result<Box<dyn LocalSourceState>> {
    match downcast_state::<SortedSource>(gstate.as_any(), name)? {
        SortedSource::Plan { bridge, state } => {
            bridge.get_local_source_state(ctx, state.as_ref())
        }
        _ => Ok(Box::new(EmptyState)),
    }
}

/// The next chunk of the sorted output of the source state `gstate` (with
/// the local state `lstate`) of the operator `name`.
pub(crate) fn next_sorted_chunk(
    ctx: &ExecutionContext,
    gstate: &dyn GlobalSourceState,
    lstate: &mut dyn LocalSourceState,
    name: &str,
) -> Result<SourceResult> {
    let batch = match downcast_state::<SortedSource>(gstate.as_any(), name)? {
        SortedSource::Merger(merger) => merger.lock().next_batch(DEFAULT_VECTOR_SIZE)?,
        SortedSource::Batches(batches) => batches.lock().pop_front(),
        SortedSource::Plan { bridge, state } => {
            return bridge.get_data(ctx, state.as_ref(), lstate);
        }
    };
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
    use datafusion_common::config::ConfigOptions;
    use datafusion_physical_expr::PhysicalExpr;
    use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
    use datafusion_physical_plan::projection::ProjectionExec;
    use datafusion_physical_plan::repartition::RepartitionExec;
    use datafusion_physical_plan::{ExecutionPlan, ExecutionPlanProperties, collect};

    use crate::aggregate::PhysicalHashAggregate;
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
    /// * aggregations, limits, unions and placeholder rows with their
    ///   `try_lower`;
    /// * `CoalescePartitionsExec`s without a fetch and `RepartitionExec`s that
    ///   do not preserve the order (over at least one partition) are removed:
    ///   the pipeline engine merges the partitions of their input anyway;
    /// * `ProjectionExec`s with a [`TestProjection`], so that the operators
    ///   below them are lowered too;
    /// * scans with the native sources;
    /// * every other node is bridged, with its subtree.
    ///
    /// For sessions with the default options, see [`lower_with`].
    pub(crate) fn lower(
        plan: &Arc<dyn ExecutionPlan>,
    ) -> Result<Arc<dyn PhysicalOperator>> {
        lower_with(plan, &ConfigOptions::default())
    }

    /// Lower `plan` like [`lower`], for sessions with `options`.
    pub(crate) fn lower_with(
        plan: &Arc<dyn ExecutionPlan>,
        options: &ConfigOptions,
    ) -> Result<Arc<dyn PhysicalOperator>> {
        let lower = |plan: &Arc<dyn ExecutionPlan>| lower_with(plan, options);
        if let Some(sort) = super::try_lower(plan, options, lower)? {
            return Ok(sort);
        }
        if let Some(aggregate) = PhysicalHashAggregate::try_lower(plan, lower)? {
            return Ok(Arc::new(aggregate));
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
            && repartition.input().output_partitioning().partition_count() > 0
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
        FixedSizeListArray, Float32Array, Float64Array, Int32Array, Int64Array,
        LargeListArray, LargeStringArray, ListArray, ListViewArray, MapArray, RunArray,
        StringArray, StringViewArray, StructArray, TimestampNanosecondArray, UnionArray,
    };
    use arrow::buffer::{NullBuffer, OffsetBuffer, ScalarBuffer};
    use arrow::compute::{SortOptions, concat_batches};
    use arrow::datatypes::{Field, Int32Type, Int64Type, UnionFields};
    use arrow_ord::ord::make_comparator;
    use arrow_row::{RowConverter, SortField};
    use datafusion::prelude::{ParquetReadOptions, SessionContext};
    use datafusion_datasource::memory::MemorySourceConfig;
    use datafusion_datasource::source::DataSourceExec;
    use datafusion_expr::Operator;
    use datafusion_physical_expr::PhysicalSortExpr;
    use datafusion_physical_expr::expressions::{BinaryExpr, CastExpr, col, lit};
    use datafusion_physical_plan::Partitioning;
    use datafusion_physical_plan::repartition::RepartitionExec;
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    use super::test_support::{
        assert_native_byte_eq, assert_parity, assert_parity_with_config, lower,
        lower_with,
    };
    use datafusion_common::ScalarValue;
    use datafusion_physical_plan::filter::FilterExecBuilder;
    use datafusion_physical_plan::limit::GlobalLimitExec;

    use crate::aggregate::PhysicalHashAggregate;
    use crate::bridge::{BridgeOperator, PipelineRootExec, bridged_plans};
    use crate::explain::format_operator_tree;
    use crate::operators::limit::PhysicalLimit;
    use crate::operators::union::test_support::{
        AfterBatch, INJECTED_FAILURE, assert_lowered_plan_byte_eq,
        assert_lowered_plan_fails_like_datafusion, batch_plan, count, values_chunk,
        values_schema,
    };
    use crate::sources::ParquetScanSource;
    use crate::sources::test_harness::{data_batch, data_schema, write_parquet};
    use crate::test_utils::{
        datafusion_context, run_sql, test_session_config, test_table_t1,
    };
    use datafusion::datasource::MemTable;
    use datafusion::prelude::SessionConfig;
    use datafusion_physical_plan::collect;

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
            let mut sort_keys = sort_keys(&schema, &keys);
            if case % 2 == 1 {
                // A computed key (a cast to the same type): sorted directly
                // if there is a single key, by its values rather than by a
                // column of the rows.
                let ordering = LexOrdering::new(sort_keys.ordering.iter().map(|key| {
                    let data_type = key.expr.data_type(&schema).unwrap();
                    let cast =
                        Arc::new(CastExpr::new(Arc::clone(&key.expr), data_type, None));
                    PhysicalSortExpr::new(cast, key.options)
                }))
                .unwrap();
                sort_keys = SortKeys::try_new(ordering, &schema)?;
                assert!(sort_keys.key_column.is_none());
            }
            for limit in [None, Some(1), Some(77), Some(499), Some(500), Some(800)] {
                let mut runs = sort_keys.sort(&schema, &batches, limit)?;
                assert_eq!(runs.len(), 1, "the rows fit into one run");
                let run = runs.pop().expect("a run");
                assert_eq!(run.num_rows(), limit.unwrap_or(500).min(500));
                // A single key of a simple type is compared directly, other
                // keys are encoded.
                assert_eq!(
                    matches!(run.keys(), RunKeys::Column(_)),
                    sort_keys.direct,
                    "case {case}: {keys:?}"
                );
                // The keys are the keys of the rows of the run, in order.
                let encoded = sort_keys
                    .converter
                    .convert_columns(&sort_keys.evaluate(run.batch())?)?;
                let key = |row| sort_keys.encoded_key(&run, row).unwrap();
                for row in 0..run.num_rows() {
                    assert_eq!(key(row).row(), encoded.row(row), "case {case}");
                    if row > 0 {
                        assert!(key(row - 1) <= key(row), "case {case}: {keys:?}");
                        assert!(run.compare_rows(row - 1, &run, row)?.is_le());
                    }
                }
                // The first rows: no row left out comes before the last row.
                let all = sort_keys
                    .converter
                    .convert_columns(&sort_keys.evaluate(&batch)?)?;
                if let Some(last) = run.num_rows().checked_sub(1) {
                    let before =
                        (0..500).filter(|&i| all.row(i) < key(last).row()).count();
                    assert!(before <= last, "case {case}: {keys:?}, limit {limit:?}");
                }
                // A single key of a simple type is sorted with the sort
                // kernel, in the same order as by the row format.
                if sort_keys.direct {
                    let mut by_rows =
                        SortKeys::try_new(sort_keys.ordering.clone(), &schema)?;
                    by_rows.direct = false;
                    by_rows.key_column = None;
                    let other = by_rows
                        .sort(&schema, &batches, limit)?
                        .pop()
                        .expect("a run");
                    assert!(matches!(other.keys(), RunKeys::Rows(_)));
                    for row in 0..run.num_rows() {
                        assert_eq!(
                            Some(key(row).row()),
                            other.encoded_key(row),
                            "case {case}: {keys:?}"
                        );
                    }
                }
            }
            assert!(sort_keys.sort(&schema, &batches, Some(0))?.is_empty());
            assert!(sort_keys.sort(&schema, &[], None)?.is_empty());
        }
        Ok(())
    }

    #[test]
    fn the_row_format_orders_nested_keys_like_datafusion() {
        // The native sort orders keys that the sort kernel does not sort by the
        // row format; DataFusion's sort orders them with `make_comparator`.
        // Both orders agree for the nested and encoded keys that the native
        // sort accepts, for every sort option (keys of unions, which they order
        // differently, are left to DataFusion).
        let mut rng = StdRng::seed_from_u64(42);
        let item = |data_type: DataType| Arc::new(Field::new("item", data_type, true));
        let int = |rng: &mut StdRng, n: usize| -> ArrayRef {
            Arc::new(Int32Array::from_iter(
                (0..n).map(|_| rng.random_bool(0.8).then(|| rng.random_range(0..4))),
            ))
        };
        let string = |rng: &mut StdRng, n: usize| -> ArrayRef {
            Arc::new(StringArray::from_iter((0..n).map(|_| {
                rng.random_bool(0.8)
                    .then(|| ["", "a", "ab", "b"][rng.random_range(0..4)])
            })))
        };
        let nulls = |rng: &mut StdRng, n: usize| {
            Some(NullBuffer::from_iter((0..n).map(|_| rng.random_bool(0.85))))
        };
        let lengths = |rng: &mut StdRng, n: usize| {
            (0..n)
                .map(|_| rng.random_range(0..4))
                .collect::<Vec<usize>>()
        };
        let structs = |rng: &mut StdRng, n: usize| -> ArrayRef {
            let fields = vec![
                Field::new("a", DataType::Int32, true),
                Field::new("b", DataType::Utf8, true),
            ];
            let children = vec![int(rng, n), string(rng, n)];
            Arc::new(StructArray::new(fields.into(), children, nulls(rng, n)))
        };
        let list = |rng: &mut StdRng,
                    n: usize,
                    values: &dyn Fn(&mut StdRng, usize) -> ArrayRef| {
            let lengths = lengths(rng, n);
            let values = values(rng, lengths.iter().sum());
            let item = item(values.data_type().clone());
            let offsets = OffsetBuffer::from_lengths(lengths);
            Arc::new(ListArray::new(item, offsets, values, nulls(rng, n))) as ArrayRef
        };
        let dictionary = |rng: &mut StdRng, n: usize| {
            let keys =
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
            arrow::compute::cast(&string(rng, n), &keys).unwrap()
        };
        type Generate<'a> = Box<dyn Fn(&mut StdRng, usize) -> ArrayRef + 'a>;
        let generators: Vec<(&str, Generate)> = vec![
            ("struct", Box::new(structs)),
            ("list", Box::new(|rng, n| list(rng, n, &int))),
            ("list of structs", Box::new(|rng, n| list(rng, n, &structs))),
            (
                "struct of a list",
                Box::new(|rng, n| {
                    let list = list(rng, n, &int);
                    let field = Field::new("l", list.data_type().clone(), true);
                    Arc::new(StructArray::new(
                        vec![field].into(),
                        vec![list],
                        nulls(rng, n),
                    ))
                }),
            ),
            (
                "large list",
                Box::new(|rng, n| {
                    let lengths = lengths(rng, n);
                    let values = int(rng, lengths.iter().sum());
                    let offsets = OffsetBuffer::from_lengths(lengths);
                    let item = item(DataType::Int32);
                    Arc::new(LargeListArray::new(item, offsets, values, nulls(rng, n)))
                }),
            ),
            (
                "fixed-size list",
                Box::new(|rng, n| {
                    let values = int(rng, 2 * n);
                    let item = item(DataType::Int32);
                    Arc::new(FixedSizeListArray::new(item, 2, values, nulls(rng, n)))
                }),
            ),
            (
                "list view",
                Box::new(|rng, n| {
                    let lengths = lengths(rng, n);
                    let values = int(rng, lengths.iter().sum());
                    let mut offset = 0;
                    let offsets = lengths.iter().map(|&length| {
                        offset += length as i32;
                        offset - length as i32
                    });
                    let offsets = ScalarBuffer::from_iter(offsets);
                    let sizes =
                        ScalarBuffer::from_iter(lengths.iter().map(|&l| l as i32));
                    let item = item(DataType::Int32);
                    Arc::new(ListViewArray::new(
                        item,
                        offsets,
                        sizes,
                        values,
                        nulls(rng, n),
                    ))
                }),
            ),
            ("dictionary", Box::new(dictionary)),
            (
                "list of dictionaries",
                Box::new(|rng, n| list(rng, n, &dictionary)),
            ),
            (
                "run-end encoded",
                Box::new(|rng, n| {
                    let mut run_ends = vec![];
                    let mut end = 0;
                    while end < n {
                        end = (end + rng.random_range(1..4)).min(n);
                        run_ends.push(end as i32);
                    }
                    let values = int(rng, run_ends.len());
                    let run_ends = Int32Array::from(run_ends);
                    Arc::new(RunArray::<Int32Type>::try_new(&run_ends, &values).unwrap())
                }),
            ),
            (
                "map",
                Box::new(|rng, n| {
                    let lengths = lengths(rng, n);
                    let entries = lengths.iter().sum();
                    let keys = StringArray::from_iter_values(
                        (0..entries).map(|_| ["a", "b", "c"][rng.random_range(0..3)]),
                    );
                    let fields = vec![
                        Field::new("keys", DataType::Utf8, false),
                        Field::new("values", DataType::Int32, true),
                    ];
                    let children = vec![Arc::new(keys) as ArrayRef, int(rng, entries)];
                    let entries = StructArray::new(fields.into(), children, None);
                    let field = Arc::new(Field::new(
                        "entries",
                        entries.data_type().clone(),
                        false,
                    ));
                    let offsets = OffsetBuffer::from_lengths(lengths);
                    Arc::new(MapArray::new(field, offsets, entries, nulls(rng, n), false))
                }),
            ),
        ];
        for (name, generate) in &generators {
            for _ in 0..4 {
                let array = generate(&mut rng, 200);
                for (descending, nulls_first) in
                    [(false, false), (false, true), (true, false), (true, true)]
                {
                    let options = SortOptions::new(descending, nulls_first);
                    let field =
                        SortField::new_with_options(array.data_type().clone(), options);
                    assert!(RowConverter::supports_fields(std::slice::from_ref(&field)));
                    let rows = RowConverter::new(vec![field])
                        .unwrap()
                        .convert_columns(&[Arc::clone(&array)])
                        .unwrap();
                    let compare = make_comparator(&array, &array, options).unwrap();
                    let mut by_rows = (0..array.len()).collect::<Vec<_>>();
                    by_rows.sort_by(|&a, &b| rows.row(a).cmp(&rows.row(b)));
                    let mut by_comparator = (0..array.len()).collect::<Vec<_>>();
                    by_comparator.sort_by(|&a, &b| compare(a, b));
                    let case = format!("{name}, {options:?}");
                    assert!(
                        by_rows.windows(2).all(|w| compare(w[0], w[1]).is_le()),
                        "{case}"
                    );
                    assert!(
                        by_comparator
                            .windows(2)
                            .all(|w| rows.row(w[0]) <= rows.row(w[1])),
                        "{case}"
                    );
                }
            }
        }
    }

    #[test]
    fn simple_single_keys_are_sorted_with_the_kernel_and_compared_directly() {
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
        let direct = |keys: &[(&str, SortOptions)]| {
            let keys = sort_keys(&schema, keys);
            (keys.direct, keys.key_column)
        };
        assert_eq!(direct(&[("i", options)]), (true, Some(0)));
        assert_eq!(direct(&[("s", options)]), (true, Some(1)));
        assert_eq!(direct(&[("d", options)]), (false, None));
        assert_eq!(direct(&[("i", options), ("s", options)]), (false, None));
        // A computed key is compared directly, by its values.
        let ordering = LexOrdering::new([PhysicalSortExpr::new_default(Arc::new(
            BinaryExpr::new(col("i", &schema).unwrap(), Operator::Plus, lit(1_i64)),
        ))])
        .unwrap();
        let keys = SortKeys::try_new(ordering, &schema).unwrap();
        assert_eq!((keys.direct, keys.key_column), (true, None));
    }

    #[test]
    fn direct_keys_that_do_not_fit_into_one_array_are_gathered_with_their_rows() {
        // 1025 one-row batches whose strings share a 2 MiB buffer, sorted by a
        // computed string key (a cast to the same type, which keeps the
        // buffer): the keys do not fit into one array, so they are sorted by
        // their encoded rows, and the key values of every run are gathered
        // with its rows, in runs whose keys fit.
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, false)]));
        let s: ArrayRef = Arc::new(StringArray::from(vec!["x".repeat(2 << 20)]));
        let batches = (0..1025)
            .map(|_| RecordBatch::try_new(Arc::clone(&schema), vec![Arc::clone(&s)]))
            .collect::<std::result::Result<Vec<_>, _>>()
            .unwrap();
        let cast = CastExpr::new(col("s", &schema).unwrap(), DataType::Utf8, None);
        let ordering =
            LexOrdering::new([PhysicalSortExpr::new_default(Arc::new(cast))]).unwrap();
        let sort_keys = SortKeys::try_new(ordering, &schema).unwrap();
        assert_eq!((sort_keys.direct, sort_keys.key_column), (true, None));
        let runs = sort_keys.sort(&schema, &batches, None).unwrap();
        assert_eq!(
            runs.iter().map(SortedRun::num_rows).collect::<Vec<_>>(),
            [513, 512]
        );
        for run in &runs {
            let RunKeys::Column(keys) = run.keys() else {
                panic!("the keys of the run are not compared directly");
            };
            assert_eq!(keys.values().as_ref(), run.batch().column(0).as_ref());
        }
        let mut merger = RunMerger::new(Arc::clone(&schema), runs, None);
        let mut rows = 0;
        while let Some(batch) = merger.next_batch(2048).unwrap() {
            rows += batch.num_rows();
        }
        assert_eq!(rows, 1025);
    }

    // ---------------------------------------------------------------------
    // Lowering
    // ---------------------------------------------------------------------

    /// `t1` with its 4 skewed partitions (`partitions == 4`), in a single
    /// partition (`1`) or without partitions (`0`).
    /// The partitions of `t1`, with `d` as `Utf8View`: TopKs of rows with
    /// 32-bit offsets are left to DataFusion (see
    /// `topks_of_rows_with_32_bit_offsets_are_left_to_datafusion`), and the
    /// tests of the lowering would not lower them.
    fn t1_partitions() -> (SchemaRef, Vec<Vec<RecordBatch>>) {
        let t1 = test_table_t1().unwrap();
        let schema = t1[0][0].schema();
        let view = |data_type: &DataType| match data_type {
            DataType::Utf8 => DataType::Utf8View,
            data_type => data_type.clone(),
        };
        let fields = schema
            .fields()
            .iter()
            .map(|field| {
                field
                    .as_ref()
                    .clone()
                    .with_data_type(view(field.data_type()))
            })
            .collect::<Vec<_>>();
        let schema =
            Arc::new(Schema::new_with_metadata(fields, schema.metadata().clone()));
        let t1 = t1
            .into_iter()
            .map(|partition| {
                partition
                    .into_iter()
                    .map(|batch| {
                        let columns = batch
                            .columns()
                            .iter()
                            .map(|column| {
                                arrow::compute::cast(column, &view(column.data_type()))
                            })
                            .collect::<std::result::Result<Vec<_>, _>>()
                            .unwrap();
                        RecordBatch::try_new(Arc::clone(&schema), columns).unwrap()
                    })
                    .collect()
            })
            .collect();
        (schema, t1)
    }

    fn t1_scan(partitions: usize) -> Arc<dyn ExecutionPlan> {
        let (schema, t1) = t1_partitions();
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
        let (schema, t1) = t1_partitions();
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
        let lowered = try_lower(plan, &ConfigOptions::default(), |plan| {
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
        let keys = [("g", false), ("id", true)];
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
            assert_eq!(lowered_ordering(&sort), "g@7 ASC NULLS LAST, id@0 DESC");
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
        // DataFusion evaluates a computed projection lazily, on the sorted
        // rows that the merge (and the nodes above it) reads, with or without
        // fetches: it is left to DataFusion.
        for (sort_fetch, merge_fetch) in
            [(None, None), (None, Some(5)), (Some(5), Some(5))]
        {
            let plan = merge(
                project(sort_exec(t1_scan(4), &id, true, sort_fetch), true),
                &[("x", true)],
                merge_fetch,
            );
            assert!(
                try_lower_bridged(&plan).is_none(),
                "{sort_fetch:?} / {merge_fetch:?}"
            );
        }
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
        // A computed projection between the merges is evaluated lazily too.
        for (inner_fetch, top_fetch) in [(Some(3), None), (None, Some(3)), (None, None)] {
            let plan = merge(
                project(
                    merge(sort_exec(t1_scan(4), &id, true, None), &id, inner_fetch),
                    true,
                ),
                &[("x", true)],
                top_fetch,
            );
            assert!(
                try_lower_bridged(&plan).is_none(),
                "{inner_fetch:?} / {top_fetch:?}"
            );
        }
    }

    #[test]
    fn lowering_checks_the_node_and_the_lowered_input() -> Result<()> {
        let options = ConfigOptions::default();
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
            assert!(try_lower(&plan, &options, not_called)?.is_none());
        }
        // `PhysicalSort::try_lower` lowers the sorts without fetch,
        // `PhysicalTopK::try_lower` the others.
        let id = [("id", false)];
        let sort = merge(sort_exec(t1_scan(4), &id, true, None), &id, None);
        let topk = merge(sort_exec(t1_scan(4), &id, true, Some(3)), &id, Some(3));
        assert!(PhysicalTopK::try_lower(&sort, &options, not_called)?.is_none());
        assert!(PhysicalSort::try_lower(&topk, &options, not_called)?.is_none());
        let bridge =
            |plan: &Arc<dyn ExecutionPlan>| -> Result<Arc<dyn PhysicalOperator>> {
                Ok(Arc::new(BridgeOperator::new(Arc::clone(plan))))
            };
        assert!(PhysicalSort::try_lower(&sort, &options, bridge)?.is_some());
        assert_eq!(
            PhysicalTopK::try_lower(&topk, &options, bridge)?
                .unwrap()
                .k(),
            3
        );
        // The lowered input must have the schema of the sort.
        let err = try_lower(&sort, &options, |_| {
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
    async fn differential_order_by_dictionary_and_nested_keys() {
        assert_sorted_natively(&[
            "SELECT id, arrow_cast(d, 'Dictionary(Int32, Utf8)') AS dd FROM t1 ORDER BY dd, id",
            "SELECT id, arrow_cast(d, 'Dictionary(Int64, LargeUtf8)') AS dd FROM t1 ORDER BY dd DESC NULLS LAST, id LIMIT 30",
            "SELECT id, make_array(a, b) AS l FROM t1 ORDER BY l, id",
            "SELECT id, arrow_cast(make_array(a, b), 'LargeList(Int64)') AS l FROM t1 ORDER BY l DESC, id LIMIT 40",
            "SELECT id, named_struct('x', a, 'y', d) AS s FROM t1 ORDER BY s NULLS FIRST, id",
            "SELECT id, named_struct('x', a, 'y', h) AS s FROM t1 ORDER BY s DESC, id LIMIT 17",
            "SELECT id, arrow_cast(make_array(c), 'LargeList(Float64)') AS l FROM t1 ORDER BY l, id LIMIT 2400",
        ])
        .await;
        // TopKs of rows with values that have 32-bit offsets, also in
        // dictionaries, lists and structs, are left to DataFusion, see
        // `topks_of_rows_with_32_bit_offsets_are_left_to_datafusion`.
        for sql in [
            "SELECT id, arrow_cast(d, 'Dictionary(Int64, Utf8)') AS dd FROM t1 ORDER BY dd DESC NULLS LAST, id LIMIT 30",
            "SELECT id, make_array(a, b) AS l FROM t1 ORDER BY l DESC, id LIMIT 40",
            "SELECT id, named_struct('x', a, 'y', d) AS s FROM t1 ORDER BY s DESC, id LIMIT 17",
        ] {
            assert_eq!(sorts(&assert_parity(sql).await), 0, "{sql}");
        }
        // Dictionaries with narrow keys are left to DataFusion, see
        // `sorts_of_narrow_dictionaries_are_left_to_datafusion`.
        let sql = "SELECT id, arrow_cast(d, 'Dictionary(Int8, Utf8)') AS dd FROM t1 \
                   ORDER BY dd DESC NULLS LAST, id LIMIT 30";
        assert_eq!(sorts(&assert_parity(sql).await), 0);
    }

    /// A table `z` with a float column `v` and a unique `id`, in a single
    /// partition: a batch of 3000 `first` values, then a batch of 5 `then`
    /// values.
    fn register_zeros(ctx: &SessionContext, first: f64, then: f64) -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Float64, true),
            Field::new("id", DataType::Int64, false),
        ]));
        let batch = |value: f64, ids: std::ops::Range<i64>| {
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Float64Array::from(vec![value; ids.clone().count()])),
                    Arc::new(Int64Array::from_iter_values(ids)),
                ],
            )
        };
        let partition = vec![batch(first, 0..3000)?, batch(then, 3000..3005)?];
        let table = MemTable::try_new(Arc::clone(&schema), vec![partition])?;
        ctx.register_table("z", Arc::new(table))?;
        Ok(())
    }

    /// The native TopKs below `root`, and how many executions of each ran
    /// DataFusion's TopK instead (`None` if it does not check its floats).
    fn topk_fallbacks(root: &Arc<dyn PhysicalOperator>) -> Vec<Option<usize>> {
        let mut fallbacks = vec![];
        if let Some(topk) = root.downcast_ref::<PhysicalTopK>() {
            fallbacks.push(topk.float_key_fallback().map(FloatKeyFallback::executions));
        }
        for child in root.children() {
            fallbacks.extend(topk_fallbacks(child));
        }
        fallbacks
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn topks_of_float_keys_whose_last_row_is_a_zero_run_datafusion_s_topk()
    -> Result<()> {
        // DataFusion's TopK compares `-0.0 = 0.0` but sorts `-0.0` first: once
        // its heap is full of `0.0`s, its threshold `v < 0.0` discards the
        // `-0.0`s that come later, which its sort returns first. Which rows it
        // returns depends on the order of its batches. The last row of the
        // result of these TopKs is a zero, so the native TopK runs DataFusion's
        // TopK instead of returning its own result (its own threshold keeps
        // both zeros, it returned the `-0.0`s first).
        for (first, then, descending) in [(0.0, -0.0, ""), (-0.0, 0.0, " DESC")] {
            let setup = async |ctx: &SessionContext| register_zeros(ctx, first, then);
            for target_partitions in [1, 4] {
                let config =
                    test_session_config().with_target_partitions(target_partitions);
                for sql in [
                    format!("SELECT v FROM z ORDER BY v{descending} LIMIT 10"),
                    format!("SELECT v, id FROM z ORDER BY v{descending}, id LIMIT 3002"),
                ] {
                    let root = assert_native_byte_eq(&sql, config.clone(), &setup).await;
                    let tree = format_operator_tree(&root);
                    assert_eq!(topk_fallbacks(&root), [Some(1)], "{sql}\n{tree}");
                }
                // A key that can fail is left to DataFusion.
                let sql =
                    format!("SELECT v FROM z ORDER BY id % 2, v{descending} LIMIT 10");
                let root = assert_native_byte_eq(&sql, config.clone(), &setup).await;
                assert_eq!(sorts(&root), 0, "{sql}\n{}", format_operator_tree(&root));
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn topks_of_float_keys_are_native() -> Result<()> {
        // Floats with zeros of both signs, NaNs and infinities, in skewed
        // partitions of several batches. The last row of the result of these
        // TopKs has no zero in a float key, so DataFusion's threshold discarded
        // no row of it, and the native TopK returns its own result.
        let schema = Arc::new(Schema::new(vec![
            Field::new("v", DataType::Float64, true),
            Field::new("w", DataType::Float32, true),
            Field::new("id", DataType::Int64, false),
        ]));
        let value = |i: i64| match i % 11 {
            0 => Some(-0.0),
            1 => Some(0.0),
            2 => Some(f64::NAN),
            3 => None,
            4 => Some(f64::NEG_INFINITY),
            5 => Some(f64::INFINITY),
            _ => Some((i % 7 - 3) as f64 / 2.0),
        };
        let batch = |ids: std::ops::Range<i64>| {
            let v = Float64Array::from_iter(ids.clone().map(value));
            let w = Float32Array::from_iter(
                ids.clone().map(|i| value(i / 3).map(|v| v as f32)),
            );
            let columns: Vec<ArrayRef> = vec![
                Arc::new(v),
                Arc::new(w),
                Arc::new(Int64Array::from_iter_values(ids)),
            ];
            RecordBatch::try_new(Arc::clone(&schema), columns)
        };
        let partitions = vec![
            vec![batch(0..1500)?, batch(1500..1501)?, batch(1501..4000)?],
            vec![],
            vec![batch(4000..4300)?],
        ];
        let setup = async |ctx: &SessionContext| {
            let table = MemTable::try_new(Arc::clone(&schema), partitions.clone())?;
            ctx.register_table("f", Arc::new(table))?;
            Ok(())
        };
        for target_partitions in [1, 3, 4] {
            let config = test_session_config().with_target_partitions(target_partitions);
            for sql in [
                // The last rows: -1.0, inf, -0.5, NaN, 1.5, 0.5, -0.5.
                "SELECT v, id FROM f ORDER BY v, id LIMIT 900",
                "SELECT v, id FROM f ORDER BY v DESC, id LIMIT 1000",
                "SELECT v, id FROM f ORDER BY v NULLS FIRST, id LIMIT 1600",
                "SELECT id FROM f ORDER BY v DESC NULLS LAST, id LIMIT 300",
                "SELECT v, w, id FROM f ORDER BY w DESC, v, id LIMIT 777",
                "SELECT v, id FROM f ORDER BY v, id LIMIT 30 OFFSET 2500",
                // Float keys after a key that decides.
                "SELECT v, id FROM f ORDER BY id, v LIMIT 10",
                // Fewer rows than the fetch: DataFusion's TopK had no threshold.
                "SELECT v, id FROM f ORDER BY v, id LIMIT 5000",
            ] {
                let root = assert_native_byte_eq(sql, config.clone(), &setup).await;
                let tree = format_operator_tree(&root);
                assert_eq!(topk_fallbacks(&root), [Some(0)], "{sql}\n{tree}");
            }
            // The last row has a zero (`-0.0`), also after a key that decides:
            // DataFusion's TopK runs.
            for sql in [
                "SELECT v, id FROM f ORDER BY v, id LIMIT 1301",
                "SELECT v, id FROM f ORDER BY id, v LIMIT 1101",
            ] {
                let root = assert_native_byte_eq(sql, config.clone(), &setup).await;
                let tree = format_operator_tree(&root);
                assert_eq!(topk_fallbacks(&root), [Some(1)], "{sql}\n{tree}");
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn topks_of_aggregated_floats_are_native() {
        // As ClickBench's `hits27`: a TopK by an average, over a native
        // aggregation, through the planner and with the lowering of the tests.
        for sql in [
            "SELECT a, AVG(b) AS l, COUNT(*) AS n FROM t1 GROUP BY a \
             ORDER BY l DESC, a LIMIT 10",
            "SELECT e, AVG(a) AS l FROM t1 WHERE a > 0 GROUP BY e HAVING COUNT(*) > 10 \
             ORDER BY l, e LIMIT 2",
            "SELECT h, MIN(c) AS m FROM t1 GROUP BY h ORDER BY m NULLS FIRST, h LIMIT 3",
            "SELECT h, MAX(c) AS m, COUNT(*) AS n FROM t1 GROUP BY h ORDER BY m DESC, h LIMIT 2",
        ] {
            let root = assert_parity(sql).await;
            let tree = format_operator_tree(&root);
            assert_eq!(count::<PhysicalTopK>(&root), 1, "{sql}\n{tree}");
            assert!(topk_fallbacks(&root)[0].is_some(), "{sql}\n{tree}");
        }
    }

    #[test]
    fn float_keys_that_datafusion_s_threshold_compares() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("i", DataType::Int64, true),
            Field::new("f", DataType::Float64, true),
            Field::new(
                "d",
                DataType::Dictionary(
                    Box::new(DataType::Int32),
                    Box::new(DataType::Float32),
                ),
                true,
            ),
            Field::new(
                "l",
                DataType::List(Arc::new(Field::new("item", DataType::Float64, true))),
                true,
            ),
        ]));
        let keys = |names: &[&str]| {
            let ordering =
                LexOrdering::new(names.iter().map(|name| {
                    PhysicalSortExpr::new_default(col(name, &schema).unwrap())
                }))
                .unwrap();
            compared_float_keys(&ordering, &schema).unwrap()
        };
        assert_eq!(keys(&["i"]), Vec::<usize>::new());
        assert_eq!(keys(&["f", "i", "d"]), [0, 2]);
        // The threshold stops comparing at the first nested key.
        assert_eq!(keys(&["i", "l", "f"]), Vec::<usize>::new());
        assert_eq!(keys(&["f", "l", "d"]), [0]);

        // The check of the last row.
        let fallback = FloatKeyFallback::new(t1_scan(1), vec![1]);
        let ordering = LexOrdering::new([
            PhysicalSortExpr::new_default(col("i", &schema)?),
            PhysicalSortExpr::new_default(col("f", &schema)?),
        ])
        .unwrap();
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int64Array::from(vec![Some(0), None, Some(2), Some(3)])),
                Arc::new(Float64Array::from(vec![
                    Some(0.0),
                    Some(-0.0),
                    None,
                    Some(f64::NAN),
                ])),
                arrow::compute::cast(
                    &Float32Array::from(vec![0.0_f32; 4]),
                    schema.field(2).data_type(),
                )?,
                Arc::new(ListArray::new_null(
                    Arc::new(Field::new("item", DataType::Float64, true)),
                    4,
                )),
            ],
        )?;
        let differs = (0..4)
            .map(|row| fallback.may_differ(&ordering, &batch, row))
            .collect::<Result<Vec<_>>>()?;
        assert_eq!(differs, [true, true, false, false]);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn merges_with_a_fetch_of_sorts_of_floats_are_topks() -> Result<()> {
        // A merge with a fetch of sorted partitions keeps the first rows of
        // DataFusion's sort, in which `-0.0` comes before `0.0`: the native
        // TopK keeps those too.
        let schema =
            Arc::new(Schema::new(vec![Field::new("v", DataType::Float64, true)]));
        let batch = |values: Vec<f64>| {
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(Float64Array::from(values))],
            )
            .unwrap()
        };
        let task_ctx = Arc::new(
            datafusion_execution::TaskContext::default()
                .with_session_config(test_session_config().with_target_partitions(1)),
        );
        for (first, then, descending) in [(0.0, -0.0, false), (-0.0, 0.0, true)] {
            let make = || -> Result<Arc<dyn ExecutionPlan>> {
                let partitions = [
                    vec![batch(vec![first; 3000]), batch(vec![then; 5])],
                    vec![batch(vec![then; 2]), batch(vec![first; 700])],
                ];
                let scan = MemorySourceConfig::try_new_exec(
                    &partitions,
                    Arc::clone(&schema),
                    None,
                )?;
                Ok(merge(
                    sort_exec(scan, &[("v", descending)], true, None),
                    &[("v", descending)],
                    Some(10),
                ))
            };
            let root =
                assert_lowered_plan_byte_eq(make, lower, Arc::clone(&task_ctx), true)
                    .await;
            assert!(root.is::<PhysicalTopK>(), "{}", format_operator_tree(&root));
        }
        Ok(())
    }

    /// A scan of three batches of 64 rows `(id, s)`, in three partitions or
    /// in one: `id` is unique, and the 192 values of `s` are distinct, in a
    /// dictionary with keys of type `key_type`.
    fn dictionary_scan(key_type: DataType, partitions: usize) -> Arc<dyn ExecutionPlan> {
        let dictionary =
            DataType::Dictionary(Box::new(key_type), Box::new(DataType::Utf8));
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("s", dictionary.clone(), true),
        ]));
        let batches = (0..3_i64)
            .map(|partition| {
                // The ids of the batches interleave, so that the batches of the
                // sorted output mix the values of all batches.
                let id = (0..64).map(|i| i * 3 + partition).collect::<Vec<_>>();
                let s = id
                    .iter()
                    .map(|id| format!("value {id}"))
                    .collect::<Vec<_>>();
                let s = arrow::compute::cast(&StringArray::from(s), &dictionary).unwrap();
                let batch = RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![Arc::new(Int64Array::from(id)), s],
                )
                .unwrap();
                vec![batch]
            })
            .collect::<Vec<_>>();
        let partitions = match partitions {
            1 => vec![batches.into_iter().flatten().collect()],
            3 => batches,
            _ => unreachable!(),
        };
        MemorySourceConfig::try_new_exec(&partitions, schema, None).unwrap()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sorts_of_narrow_dictionaries_that_projections_remove_are_left_to_datafusion()
    -> Result<()> {
        // `SPM[id] -> Projection[id] -> SortExec[id]` of a partition of three
        // 64-row batches with 192 distinct `Int8`-keyed dictionary values:
        // DataFusion's sort sorts the rows of the `SortExec`, and concatenates
        // the batches of the small partition, whose dictionaries need more
        // values than the keys address. The native sort projects the rows
        // before it sorts them, without the dictionaries, and returned the
        // rows. Such sorts are left to DataFusion; `Int32` keys stay native.
        let config = test_session_config().with_target_partitions(1);
        for (key_type, fails) in [(DataType::Int8, true), (DataType::Int32, false)] {
            let make = || -> Result<Arc<dyn ExecutionPlan>> {
                let scan = dictionary_scan(key_type.clone(), 1);
                let sort = sort_exec(scan, &[("id", false)], true, None);
                let id = col("id", &sort.schema())?;
                let projection =
                    ProjectionExec::try_new(vec![(id, "id".to_string())], sort)?;
                Ok(merge(Arc::new(projection), &[("id", false)], None))
            };
            let root = match fails {
                true => {
                    let message = "Dictionary key bigger than the key type";
                    assert_lowered_plan_fails_like_datafusion(
                        make,
                        lower,
                        config.clone(),
                        message,
                    )
                    .await
                }
                false => {
                    let task_ctx = Arc::new(
                        datafusion_execution::TaskContext::default()
                            .with_session_config(config.clone()),
                    );
                    assert_lowered_plan_byte_eq(make, lower, task_ctx, true).await
                }
            };
            assert_eq!(sorts(&root) == 1, !fails, "{key_type}");
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sorts_of_narrow_dictionaries_are_left_to_datafusion() {
        // The native sort materializes runs and chunks of more rows than
        // DataFusion's batches, whose dictionaries can then need more values
        // than narrow keys address: DataFusion merges the sorted partitions
        // into batches of 64 rows, each of which uses at most 64 of the 192
        // values, but 192 values do not fit in `Int8` keys. Such sorts are
        // left to DataFusion.
        let config = test_session_config()
            .with_target_partitions(1)
            .with_batch_size(64);
        let task_ctx = Arc::new(
            datafusion_execution::TaskContext::default().with_session_config(config),
        );
        for (key_type, native) in [
            (DataType::Int8, false),
            (DataType::UInt8, false),
            (DataType::Int16, false),
            (DataType::Int32, true),
        ] {
            for fetch in [None, Some(150)] {
                // A TopK of rows with `Utf8` values is left to DataFusion too.
                let native = native && fetch.is_none();
                let key = key_type.clone();
                let make = move || -> Result<Arc<dyn ExecutionPlan>> {
                    let sort = sort_exec(
                        dictionary_scan(key.clone(), 3),
                        &[("id", false)],
                        true,
                        fetch,
                    );
                    Ok(merge(sort, &[("id", false)], fetch))
                };
                let root =
                    assert_lowered_plan_byte_eq(make, lower, Arc::clone(&task_ctx), true)
                        .await;
                assert_eq!(sorts(&root) == 1, native, "{key_type}, fetch {fetch:?}");
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sorts_by_computed_narrow_dictionaries_are_left_to_datafusion() -> Result<()>
    {
        // DataFusion sorts the three 64-row batches of a small partition at
        // once, so it evaluates the key `CAST(s AS Dictionary(Int8, Utf8))` on
        // all 192 rows, whose 192 distinct values do not fit in `Int8` keys.
        // The native sort evaluates its keys on each chunk, of 64 rows here,
        // and returned the rows. Sorts by keys that evaluate to narrow
        // encodings (also inside the key) are left to DataFusion; `Int32` keys
        // of `LargeUtf8` values stay native.
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("s", DataType::Utf8, false),
        ]));
        let batches = (0..3_i64)
            .map(|batch| {
                let id = (0..64).map(|i| batch * 64 + i).collect::<Vec<_>>();
                let s = id
                    .iter()
                    .map(|id| format!("value {id}"))
                    .collect::<Vec<_>>();
                let columns: Vec<ArrayRef> = vec![
                    Arc::new(Int64Array::from(id)),
                    Arc::new(StringArray::from(s)),
                ];
                RecordBatch::try_new(Arc::clone(&schema), columns).unwrap()
            })
            .collect::<Vec<_>>();
        let dictionary = |keys: DataType| {
            DataType::Dictionary(Box::new(keys), Box::new(DataType::Utf8))
        };
        let cast =
            |expr: Arc<dyn PhysicalExpr>, data_type: DataType| -> Arc<dyn PhysicalExpr> {
                Arc::new(CastExpr::new(expr, data_type, None))
            };
        let s = col("s", &schema)?;
        let keys = [
            // (key, DataFusion fails, native)
            (
                cast(Arc::clone(&s), dictionary(DataType::Int8)),
                true,
                false,
            ),
            (
                cast(
                    cast(Arc::clone(&s), dictionary(DataType::Int8)),
                    DataType::Utf8,
                ),
                true,
                false,
            ),
            (
                cast(Arc::clone(&s), dictionary(DataType::Int16)),
                false,
                false,
            ),
            // `Utf8` values have 32-bit offsets too, see
            // `sorts_by_computed_keys_with_32_bit_offsets_are_left_to_datafusion`.
            (
                cast(Arc::clone(&s), dictionary(DataType::Int32)),
                false,
                false,
            ),
            (
                cast(
                    Arc::clone(&s),
                    DataType::Dictionary(
                        Box::new(DataType::Int32),
                        Box::new(DataType::LargeUtf8),
                    ),
                ),
                false,
                true,
            ),
        ];
        let config = test_session_config()
            .with_target_partitions(1)
            .with_batch_size(64);
        for (key, fails, native) in keys {
            let make = || -> Result<Arc<dyn ExecutionPlan>> {
                let scan = MemorySourceConfig::try_new_exec(
                    std::slice::from_ref(&batches),
                    Arc::clone(&schema),
                    None,
                )?;
                let ordering = LexOrdering::new([
                    PhysicalSortExpr::new_default(Arc::clone(&key)),
                    PhysicalSortExpr::new_default(col("id", &schema)?),
                ])
                .unwrap();
                Ok(Arc::new(SortExec::new(ordering, scan)))
            };
            let root = match fails {
                true => {
                    assert_lowered_plan_fails_like_datafusion(
                        make,
                        lower,
                        config.clone(),
                        "Dictionary key bigger than the key type",
                    )
                    .await
                }
                false => {
                    let task_ctx = Arc::new(
                        datafusion_execution::TaskContext::default()
                            .with_session_config(config.clone()),
                    );
                    assert_lowered_plan_byte_eq(make, lower, task_ctx, true).await
                }
            };
            assert_eq!(sorts(&root) == 1, native, "{key}");
        }
        // The same sorts in SQL, where DataFusion simplifies `arrow_cast` into
        // a cast.
        let setup = async |ctx: &SessionContext| -> Result<()> {
            let table = MemTable::try_new(Arc::clone(&schema), vec![batches.clone()])?;
            ctx.register_table("t", Arc::new(table))?;
            Ok(())
        };
        for (keys, fails, native) in [
            (
                "Int8, Utf8",
                Some("Dictionary key bigger than the key type"),
                false,
            ),
            ("Int32, Utf8", None, false),
            ("Int32, LargeUtf8", None, true),
        ] {
            let sql = format!(
                "SELECT id, s FROM t ORDER BY arrow_cast(s, 'Dictionary({keys})'), id"
            );
            let root =
                assert_outcome_like_datafusion(&sql, config.clone(), &setup, fails).await;
            assert_eq!(sorts(&root) == 1, native, "{sql}");
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sorts_by_computed_keys_with_32_bit_offsets_are_left_to_datafusion()
    -> Result<()> {
        // DataFusion sorts the three 64-row batches of a small partition at
        // once, so it evaluates the key `CAST(repeat(s, 12582912) AS Utf8)`
        // on all 192 rows: 2.25 GiB of `LargeUtf8` values, more than `Utf8`
        // addresses. The native sort evaluates its keys on each chunk, of 64
        // rows here, and returned the rows. Sorts by keys that compute values
        // with 32-bit offsets (also inside the key) are left to DataFusion.
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("s", DataType::LargeUtf8, false),
        ]));
        let batches = (0..3_i64)
            .map(|batch| {
                let columns: Vec<ArrayRef> = vec![
                    Arc::new(Int64Array::from_iter_values(
                        (0..64).map(|i| batch * 64 + i),
                    )),
                    Arc::new(LargeStringArray::from(vec!["x"; 64])),
                ];
                RecordBatch::try_new(Arc::clone(&schema), columns).unwrap()
            })
            .collect::<Vec<_>>();
        let setup = async |ctx: &SessionContext| -> Result<()> {
            let table = MemTable::try_new(Arc::clone(&schema), vec![batches.clone()])?;
            ctx.register_table("t", Arc::new(table))?;
            Ok(())
        };
        let config = test_session_config().with_target_partitions(1);
        let sql = "SELECT id FROM t ORDER BY id, arrow_cast(repeat(s, 12582912), 'Utf8')";
        let message = "LargeString array too large to cast to String array";
        let root =
            assert_outcome_like_datafusion(sql, config, &setup, Some(message)).await;
        assert_eq!(sorts(&root), 0);

        // Columns compute nothing, and other values stay native.
        for (sql, native) in [
            ("SELECT id, d FROM t1 ORDER BY d, id", true),
            ("SELECT id, d FROM t1 ORDER BY upper(d), id", false),
            ("SELECT id, d FROM t1 ORDER BY length(upper(d)), id", false),
            (
                "SELECT id, d FROM t1 ORDER BY arrow_cast(d, 'LargeUtf8'), id",
                true,
            ),
            ("SELECT id, d FROM t1 ORDER BY length(d), id", true),
            ("SELECT id, a FROM t1 ORDER BY a + b, id", true),
        ] {
            let root = assert_parity(sql).await;
            let tree = format_operator_tree(&root);
            assert_eq!(sorts(&root) == 1, native, "{sql}\n{tree}");
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sorts_below_projections_of_large_literals_are_left_to_datafusion()
    -> Result<()> {
        // `SPM(fetch=1) -> Projection[id, 2 MiB string] -> SortExec[id]` of one
        // batch of 1024 rows, with batches of 64 rows: DataFusion evaluates the
        // literal on the sorted batches that it reads, of 64 rows, while the
        // native sort projects the chunks of its input, here all 1024 rows,
        // which need 2 GiB of strings. Projections of literals with 32-bit
        // offsets are left to DataFusion; other literals stay native.
        let schema =
            Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false)]));
        let ids = Int64Array::from_iter_values((0..1024).rev());
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(ids)])?;
        let config = test_session_config().with_batch_size(64);
        let task_ctx = Arc::new(
            datafusion_execution::TaskContext::default().with_session_config(config),
        );
        for (literal, native) in [
            (ScalarValue::Utf8(Some("x".repeat(2 << 20))), false),
            (ScalarValue::Int64(Some(7)), true),
        ] {
            let make = || -> Result<Arc<dyn ExecutionPlan>> {
                let scan = MemorySourceConfig::try_new_exec(
                    &[vec![batch.clone()]],
                    Arc::clone(&schema),
                    None,
                )?;
                let sort = sort_exec(scan, &[("id", false)], true, None);
                let exprs = vec![
                    (col("id", &sort.schema())?, "id".to_string()),
                    (lit(literal.clone()), "l".to_string()),
                ];
                let projection = Arc::new(ProjectionExec::try_new(exprs, sort)?);
                Ok(merge(projection, &[("id", false)], Some(1)))
            };
            let root =
                assert_lowered_plan_byte_eq(make, lower, Arc::clone(&task_ctx), true)
                    .await;
            assert_eq!(sorts(&root) == 1, native, "{literal:?}");
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sorts_that_datafusion_concatenates_beyond_offsets_are_left_to_it()
    -> Result<()> {
        // DataFusion's sort concatenates the batches of a partition while they
        // occupy fewer than `sort_in_place_threshold_bytes`. With a threshold
        // of `usize::MAX`, it concatenates 1025 one-row batches whose strings
        // share a 2 MiB buffer, which need 2 GiB of strings in one array, and
        // fails; the native sort returned the first row. Up to 2 GiB, which
        // its reservations count for every batch, its concatenations fit.
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("s", DataType::Utf8, false),
        ]));
        let s: ArrayRef = Arc::new(StringArray::from(vec!["x".repeat(2 << 20)]));
        let batches = (0..1025)
            .map(|id| {
                let columns: Vec<ArrayRef> =
                    vec![Arc::new(Int64Array::from(vec![id])), Arc::clone(&s)];
                RecordBatch::try_new(Arc::clone(&schema), columns).unwrap()
            })
            .collect::<Vec<_>>();
        let make = || -> Result<Arc<dyn ExecutionPlan>> {
            let scan = MemorySourceConfig::try_new_exec(
                std::slice::from_ref(&batches),
                Arc::clone(&schema),
                None,
            )?;
            let sort = sort_exec(scan, &[("id", false)], true, None);
            Ok(merge(sort, &[("id", false)], Some(1)))
        };
        let mut config = test_session_config();
        config.options_mut().execution.sort_in_place_threshold_bytes = usize::MAX;
        let options = Arc::clone(config.options());
        let lower = |plan: &Arc<dyn ExecutionPlan>| lower_with(plan, &options);
        let message = "Offset overflow error";
        let root =
            assert_lowered_plan_fails_like_datafusion(make, lower, config, message).await;
        assert_eq!(sorts(&root), 0);
        // With thresholds up to 2 GiB, and for sorts of values without 32-bit
        // offsets, the sort stays native.
        let bridge =
            |plan: &Arc<dyn ExecutionPlan>| -> Result<Arc<dyn PhysicalOperator>> {
                Ok(Arc::new(BridgeOperator::new(Arc::clone(plan))))
            };
        let mut options = ConfigOptions::default();
        assert!(try_lower(&make()?, &options, bridge)?.is_some());
        options.execution.sort_in_place_threshold_bytes = i32::MAX as usize;
        assert!(try_lower(&make()?, &options, bridge)?.is_some());
        options.execution.sort_in_place_threshold_bytes = usize::MAX;
        assert!(try_lower(&make()?, &options, bridge)?.is_none());
        let view = sort_exec(t1_scan(4), &[("id", false)], true, None);
        assert!(
            try_lower(&merge(view, &[("id", false)], None), &options, bridge)?.is_some()
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn topks_whose_dynamic_filter_reaches_a_fetch_are_left_to_datafusion()
    -> Result<()> {
        // `SortExec(fetch=1) -> FilterExec(the sort's dynamic filter, fetch=10)`
        // over the one-row batches `[10, 20, 21, ..., 28, 0]`, with batches of
        // one row: DataFusion's TopK publishes the threshold 10 after the first
        // row, so its filter drops 20 to 28 and reads 0 within its fetch of 10
        // rows. The native TopK publishes its threshold later, and the filter
        // stopped after 10 to 28, before 0. Which rows a node with a fetch
        // between the TopK and the nodes that consume its dynamic filter passes
        // depends on when the threshold tightens: such TopKs are left to
        // DataFusion. Without the fetch, the TopK stays native.
        let schema = Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]));
        let batches = [10, 20, 21, 22, 23, 24, 25, 26, 27, 28, 0]
            .map(|v| {
                let columns: Vec<ArrayRef> = vec![Arc::new(Int64Array::from(vec![v]))];
                RecordBatch::try_new(Arc::clone(&schema), columns).unwrap()
            })
            .to_vec();
        let config = test_session_config()
            .with_target_partitions(1)
            .with_batch_size(1);
        let task_ctx = Arc::new(
            datafusion_execution::TaskContext::default().with_session_config(config),
        );
        /// A TopK of `v` over the plan that `filtered` makes of its dynamic
        /// filter and `scan`.
        fn topk_of(
            scan: Arc<dyn ExecutionPlan>,
            filtered: impl Fn(
                Arc<dyn PhysicalExpr>,
                Arc<dyn ExecutionPlan>,
            ) -> Result<Arc<dyn ExecutionPlan>>,
        ) -> Result<Arc<dyn ExecutionPlan>> {
            let placeholder =
                FilterExecBuilder::new(lit(true), Arc::clone(&scan)).build()?;
            let sort = sort_exec(Arc::new(placeholder), &[("v", false)], false, Some(1));
            let filter = sort
                .dynamic_expressions_produced()
                .into_iter()
                .find(|expr| expr.is::<DynamicFilterPhysicalExpr>())
                .expect("the dynamic filter of the TopK");
            sort.replace_children(
                vec![filtered(filter, scan)?],
                ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
            )
        }
        for (fetch, native) in [(Some(10), false), (None, true)] {
            let make = || -> Result<Arc<dyn ExecutionPlan>> {
                let scan = MemorySourceConfig::try_new_exec(
                    std::slice::from_ref(&batches),
                    Arc::clone(&schema),
                    None,
                )?;
                topk_of(scan, |filter, scan| {
                    let filter = FilterExecBuilder::new(filter, scan)
                        .with_fetch(fetch)
                        .with_batch_size(1)
                        .build()?;
                    Ok(Arc::new(filter))
                })
            };
            let root =
                assert_lowered_plan_byte_eq(make, lower, Arc::clone(&task_ctx), true)
                    .await;
            assert_eq!(sorts(&root) == 1, native, "fetch {fetch:?}");
        }
        // Also a coalesce with a fetch above the filter.
        let scan = MemorySourceConfig::try_new_exec(
            &[batches[..5].to_vec(), batches[5..].to_vec()],
            Arc::clone(&schema),
            None,
        )?;
        let plan = topk_of(scan, |filter, scan| {
            let filter = FilterExecBuilder::new(filter, scan).build()?;
            let coalesce =
                CoalescePartitionsExec::new(Arc::new(filter)).with_fetch(Some(10));
            Ok(Arc::new(coalesce))
        })?;
        assert!(try_lower_bridged(&plan).is_none());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sorts_by_union_keys_are_left_to_datafusion() -> Result<()> {
        // The row format orders unions unlike DataFusion's sort: for the dense
        // union rows `(type id, value)` `[(0, 1), (0, 2), (1, 3), (1, 4)]` with
        // the ids `0..4`, `ORDER BY u DESC, id` returned the ids `[2, 3, 0, 1]`
        // natively and `[3, 2, 1, 0]` in DataFusion, and nulls came elsewhere.
        // Sorts by union keys, also inside structs and lists, are left to
        // DataFusion, for every sort option; other nested keys stay native.
        let fields = UnionFields::try_new(
            vec![0, 1],
            vec![
                Field::new("a", DataType::Int32, true),
                Field::new("b", DataType::Int32, true),
            ],
        )?;
        let rows = [
            (0, Some(1)),
            (0, Some(2)),
            (1, Some(3)),
            (1, Some(4)),
            (0, None),
            (1, None),
        ];
        let type_ids =
            ScalarBuffer::from(rows.map(|(type_id, _)| type_id as i8).to_vec());
        let values = |type_id| {
            Int32Array::from_iter(rows.map(|(t, v)| v.filter(|_| t == type_id)))
        };
        let dense = || -> Result<ArrayRef> {
            let mut counts = [0; 2];
            let offsets = rows.map(|(type_id, _)| {
                counts[type_id] += 1;
                counts[type_id] - 1
            });
            let children = [0, 1].map(|type_id| {
                let values = rows.iter().filter(|(t, _)| *t == type_id).map(|(_, v)| *v);
                Arc::new(Int32Array::from_iter(values)) as ArrayRef
            });
            Ok(Arc::new(UnionArray::try_new(
                fields.clone(),
                type_ids.clone(),
                Some(ScalarBuffer::from(offsets.to_vec())),
                children.to_vec(),
            )?))
        };
        let sparse = || -> Result<ArrayRef> {
            let children = vec![Arc::new(values(0)) as ArrayRef, Arc::new(values(1))];
            Ok(Arc::new(UnionArray::try_new(
                fields.clone(),
                type_ids.clone(),
                None,
                children,
            )?))
        };
        let in_struct = |union: ArrayRef| -> ArrayRef {
            let field = Field::new("u", union.data_type().clone(), true);
            Arc::new(StructArray::new(vec![field].into(), vec![union], None))
        };
        let in_list = |union: ArrayRef| -> ArrayRef {
            let item = Arc::new(Field::new("item", union.data_type().clone(), true));
            let offsets = OffsetBuffer::from_lengths(std::iter::repeat_n(1, union.len()));
            Arc::new(ListArray::new(item, offsets, union, None))
        };
        let config = test_session_config().with_target_partitions(1);
        let task_ctx = Arc::new(
            datafusion_execution::TaskContext::default().with_session_config(config),
        );
        for (name, keys) in [
            ("dense", dense()?),
            ("sparse", sparse()?),
            ("dense in a struct", in_struct(dense()?)),
            ("sparse in a list", in_list(sparse()?)),
        ] {
            let schema = Arc::new(Schema::new(vec![
                Field::new("id", DataType::Int64, false),
                Field::new("u", keys.data_type().clone(), true),
            ]));
            let ids = Int64Array::from_iter_values(0..rows.len() as i64);
            let batch =
                RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(ids), keys])?;
            for (descending, nulls_first, fetch) in [
                (false, false, None),
                (false, true, None),
                (true, false, None),
                (true, true, None),
                (true, false, Some(3)),
            ] {
                let make = || -> Result<Arc<dyn ExecutionPlan>> {
                    let scan = MemorySourceConfig::try_new_exec(
                        &[vec![batch.clone()]],
                        Arc::clone(&schema),
                        None,
                    )?;
                    let options = SortOptions::new(descending, nulls_first);
                    let ordering = LexOrdering::new([
                        PhysicalSortExpr::new(col("u", &schema)?, options),
                        PhysicalSortExpr::new_default(col("id", &schema)?),
                    ])
                    .unwrap();
                    Ok(Arc::new(SortExec::new(ordering, scan).with_fetch(fetch)))
                };
                let root =
                    assert_lowered_plan_byte_eq(make, lower, Arc::clone(&task_ctx), true)
                        .await;
                let case =
                    format!("{name}, descending {descending}, nulls first {nulls_first}");
                assert_eq!(sorts(&root), 0, "{case}, fetch {fetch:?}");
            }
            // Also merges of sorted partitions.
            let scan = MemorySourceConfig::try_new_exec(
                &[vec![batch.clone()], vec![batch]],
                Arc::clone(&schema),
                None,
            )?;
            let sort = sort_exec(scan, &[("u", true), ("id", false)], true, None);
            let merged = merge(sort, &[("u", true), ("id", false)], None);
            assert!(try_lower_bridged(&merged).is_none(), "{name}");
        }
        // A struct key without unions stays native.
        let sort = sort_exec(t1_scan(1), &[("id", false)], false, None);
        assert!(try_lower_bridged(&sort).is_some());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn merges_with_a_fetch_of_sorts_of_run_end_encoded_floats_keep_negative_zeros()
    {
        // A merge with a fetch of a sort of run-end encoded floats keeps the
        // first rows of DataFusion's sort, in which the `-0.0`s come first. The
        // threshold of the native TopK must select both zeros, also for run-end
        // encoded values.
        let values = Arc::new(Field::new("values", DataType::Float64, true));
        let run_ends = Arc::new(Field::new("run_ends", DataType::Int32, false));
        let ree = DataType::RunEndEncoded(Arc::clone(&run_ends), Arc::clone(&values));
        let schema = Arc::new(Schema::new(vec![Field::new("v", ree, true)]));
        let batch = |value: f64, len: i32| {
            let array = RunArray::<Int32Type>::try_new(
                &Int32Array::from(vec![len]),
                &Float64Array::from(vec![value]),
            )
            .unwrap();
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(array)]).unwrap()
        };
        let config = test_session_config().with_target_partitions(1);
        let task_ctx = Arc::new(
            datafusion_execution::TaskContext::default().with_session_config(config),
        );
        for (first, then, descending) in [(0.0, -0.0, false), (-0.0, 0.0, true)] {
            let make = || -> Result<Arc<dyn ExecutionPlan>> {
                let partitions = [vec![batch(first, 2048), batch(then, 5)]];
                let scan = MemorySourceConfig::try_new_exec(
                    &partitions,
                    Arc::clone(&schema),
                    None,
                )?;
                let sort = sort_exec(scan, &[("v", descending)], true, None);
                Ok(merge(sort, &[("v", descending)], Some(10)))
            };
            let root =
                assert_lowered_plan_byte_eq(make, lower, Arc::clone(&task_ctx), true)
                    .await;
            assert!(root.is::<PhysicalTopK>(), "{}", format_operator_tree(&root));
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn topks_of_large_strings_sort_rows_that_fit_into_arrays() {
        // 1024 one-row batches of a 2 MiB `Utf8` value (all sharing one
        // buffer), then `"a"`: DataFusion's TopK of 1 row returns `"a"`. The
        // native TopK buffered the 1024 rows (2 GiB) and concatenated their
        // keys, whose offsets overflowed: it now also reduces its buffer once
        // it holds about `TOPK_BUFFER_BYTES` bytes, and sorts keys that do not
        // fit into one array by their encoded rows.
        let schema = Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, false)]));
        let large = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(StringArray::from(vec!["b".repeat(2 << 20)]))],
        )
        .unwrap();
        let small = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(StringArray::from(vec!["a"]))],
        )
        .unwrap();
        let mut partition = vec![large; 1024];
        partition.push(small);
        let config = test_session_config().with_target_partitions(1);
        let task_ctx = Arc::new(
            datafusion_execution::TaskContext::default().with_session_config(config),
        );
        let make = || -> Result<Arc<dyn ExecutionPlan>> {
            let scan = MemorySourceConfig::try_new_exec(
                std::slice::from_ref(&partition),
                Arc::clone(&schema),
                None,
            )?;
            Ok(sort_exec(scan, &[("s", false)], false, Some(1)))
        };
        let root = assert_lowered_plan_byte_eq(make, lower, task_ctx, true).await;
        assert!(root.is::<PhysicalTopK>(), "{}", format_operator_tree(&root));

        // Keys that do not fit into one array are not sorted by the kernel,
        // but by their encoded rows.
        let sort_keys = sort_keys(&schema, &[("s", SortOptions::default())]);
        assert!(sort_keys.direct);
        let keys = partition
            .iter()
            .map(|batch| sort_keys.evaluate(batch))
            .collect::<Result<Vec<_>>>()
            .unwrap();
        assert!(
            sort_keys
                .sort_with_kernel(&keys, Some(2))
                .unwrap()
                .is_none()
        );
        let some = &keys[1000..];
        assert!(sort_keys.sort_with_kernel(some, Some(2)).unwrap().is_some());
    }

    #[test]
    fn sorted_rows_that_do_not_fit_into_one_batch_are_split_into_runs() {
        // 1025 one-row batches in descending order of `k`, whose strings share
        // a 2 MiB buffer: sorted, they need 2 GiB of string values in one
        // array, more than its 32-bit offsets address. They become
        // consecutive runs, which merge into the sorted rows.
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("s", DataType::Utf8, false),
        ]));
        let s: ArrayRef = Arc::new(StringArray::from(vec!["x".repeat(2 << 20)]));
        let batches = (0..1025)
            .rev()
            .map(|k| {
                let columns: Vec<ArrayRef> =
                    vec![Arc::new(Int64Array::from(vec![k])), Arc::clone(&s)];
                RecordBatch::try_new(Arc::clone(&schema), columns).unwrap()
            })
            .collect::<Vec<_>>();
        let sort_keys = sort_keys(&schema, &[("k", SortOptions::default())]);
        let runs = sort_keys.sort(&schema, &batches, None).unwrap();
        assert_eq!(
            runs.iter().map(SortedRun::num_rows).collect::<Vec<_>>(),
            [513, 512]
        );
        let mut merger = RunMerger::new(Arc::clone(&schema), runs, None);
        let mut keys = vec![];
        while let Some(batch) = merger.next_batch(2048).unwrap() {
            keys.extend_from_slice(batch.column(0).as_primitive::<Int64Type>().values());
        }
        assert_eq!(keys, (0..1025).collect::<Vec<_>>());
    }

    /// One partition of two batches of 512 rows `(k, p)`, with interleaving
    /// keys `k` and the run-end encoded payloads `p` of `payload`.
    fn interleaving_payloads_scan(
        payload: fn(&str) -> ArrayRef,
    ) -> Arc<dyn ExecutionPlan> {
        let batches = [(0, "x"), (1, "y")].map(|(first, letter)| {
            let keys = Int64Array::from_iter_values((0..512).map(|i| i * 2 + first));
            (Arc::new(keys) as ArrayRef, payload(letter))
        });
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("p", batches[0].1.data_type().clone(), true),
        ]));
        let batches = batches
            .into_iter()
            .map(|(keys, payload)| {
                RecordBatch::try_new(Arc::clone(&schema), vec![keys, payload]).unwrap()
            })
            .collect::<Vec<_>>();
        MemorySourceConfig::try_new_exec(&[batches], schema, None).unwrap()
    }

    /// 512 rows of a run-end encoded 2 MiB string of `letter`.
    fn run_end_encoded_strings(letter: &str) -> ArrayRef {
        Arc::new(
            RunArray::<Int32Type>::try_new(
                &Int32Array::from(vec![512]),
                &StringArray::from(vec![letter.repeat(2 << 20)]),
            )
            .unwrap(),
        )
    }

    /// Whether a sort of rows `(k, p)` with `p` of `data_type`, by `k` and
    /// with `fetch`, is lowered.
    fn lowers_sort_of_payload(data_type: &DataType, fetch: Option<usize>) -> bool {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("p", data_type.clone(), true),
        ]));
        let scan = MemorySourceConfig::try_new_exec(&[vec![]], schema, None).unwrap();
        try_lower_bridged(&sort_exec(scan, &[("k", false)], false, fetch)).is_some()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn topks_of_rows_with_32_bit_offsets_are_left_to_datafusion() {
        // DataFusion's TopK gathers its rows into one batch (when it emits and
        // when it compacts them), without splitting rows that do not fit, so
        // it fails where the native TopK, which splits them, returns rows.
        let config = test_session_config();
        // Two batches with interleaving keys and a constant, distinct, run-end
        // encoded 2 MiB string each: the top 1024 rows alternate between the
        // batches and need 2 GiB of values.
        let make = || -> Result<Arc<dyn ExecutionPlan>> {
            let scan = interleaving_payloads_scan(run_end_encoded_strings);
            Ok(sort_exec(scan, &[("k", false)], false, Some(1024)))
        };
        let message = "Offset overflow error: 2147483648";
        let root = assert_lowered_plan_fails_like_datafusion(
            make,
            lower,
            config.clone(),
            message,
        )
        .await;
        assert_eq!(sorts(&root), 0);
        // 1100 one-row batches whose strings share a 2 MiB buffer.
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("s", DataType::Utf8, false),
        ]));
        let s: ArrayRef = Arc::new(StringArray::from(vec!["x".repeat(2 << 20)]));
        let batches = (0..1100)
            .map(|k| {
                let columns: Vec<ArrayRef> =
                    vec![Arc::new(Int64Array::from(vec![k])), Arc::clone(&s)];
                RecordBatch::try_new(Arc::clone(&schema), columns).unwrap()
            })
            .collect::<Vec<_>>();
        let make = || -> Result<Arc<dyn ExecutionPlan>> {
            let scan = MemorySourceConfig::try_new_exec(
                std::slice::from_ref(&batches),
                Arc::clone(&schema),
                None,
            )?;
            Ok(sort_exec(scan, &[("k", false)], false, Some(2000)))
        };
        let root = assert_lowered_plan_fails_like_datafusion(
            make,
            lower,
            config,
            "Offset overflow error",
        )
        .await;
        assert_eq!(sorts(&root), 0);

        // Rows whose values have 32-bit offsets, also inside them, are left to
        // DataFusion; a TopK of one row cannot need more than an array holds,
        // nor a TopK of `Utf8` or `Binary` columns of small in-memory batches
        // (here none, see `topks_of_bounded_strings_are_native`).
        let field = |data_type: DataType| Arc::new(Field::new("x", data_type, true));
        let dictionary = |values: DataType| {
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(values))
        };
        let run_end_encoded = |values: DataType| {
            let run_ends = Arc::new(Field::new("run_ends", DataType::Int32, false));
            DataType::RunEndEncoded(run_ends, field(values))
        };
        // Their full sorts stay native, except for run-end encoded strings,
        // see `sorts_of_run_end_encoded_strings_are_left_to_datafusion`.
        for (data_type, topk_natively, sorted_natively) in [
            (DataType::Utf8, true, true),
            (DataType::Binary, true, true),
            (DataType::List(field(DataType::Int32)), false, true),
            (DataType::LargeList(field(DataType::Utf8)), false, true),
            (dictionary(DataType::Utf8), false, true),
            (
                DataType::Struct(vec![field(DataType::Utf8)].into()),
                false,
                true,
            ),
            (run_end_encoded(DataType::Utf8), false, false),
        ] {
            let topk = lowers_sort_of_payload(&data_type, Some(2));
            assert_eq!(topk, topk_natively, "{data_type}");
            assert!(lowers_sort_of_payload(&data_type, Some(1)), "{data_type}");
            let lowered = lowers_sort_of_payload(&data_type, None);
            assert_eq!(lowered, sorted_natively, "{data_type}");
        }
        for data_type in [
            DataType::Int64,
            DataType::Utf8View,
            DataType::LargeUtf8,
            DataType::LargeBinary,
            DataType::FixedSizeBinary(4),
            DataType::LargeList(field(DataType::Int64)),
            dictionary(DataType::LargeUtf8),
            DataType::Struct(vec![field(DataType::Utf8View)].into()),
            run_end_encoded(DataType::Int64),
        ] {
            assert!(lowers_sort_of_payload(&data_type, Some(2)), "{data_type}");
        }
    }

    /// A scan of `batches` one-row batches `(k, s)` in one partition, whose `k`
    /// counts down and whose `Utf8` values `s` all share one buffer of
    /// `bytes` bytes.
    fn shared_strings_scan(batches: i64, bytes: usize) -> Arc<dyn ExecutionPlan> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("s", DataType::Utf8, false),
        ]));
        let s: ArrayRef = Arc::new(StringArray::from(vec!["x".repeat(bytes)]));
        let batches = (0..batches)
            .rev()
            .map(|k| {
                let columns: Vec<ArrayRef> =
                    vec![Arc::new(Int64Array::from(vec![k])), Arc::clone(&s)];
                RecordBatch::try_new(Arc::clone(&schema), columns).unwrap()
            })
            .collect::<Vec<_>>();
        MemorySourceConfig::try_new_exec(&[batches], schema, None).unwrap()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn topks_of_bounded_strings_are_native() -> Result<()> {
        // DataFusion's TopK gathers its rows into one batch. If its rows are
        // rows of in-memory batches whose `Utf8` and `Binary` values occupy
        // less than 1 GiB, no rows of them can need more than the 2 GiB that
        // 32-bit offsets address, and it never fails: such TopKs (of at most
        // 65536 rows) are native.
        for sql in [
            "SELECT id, d FROM t1 ORDER BY id DESC LIMIT 65536",
            "SELECT d, id FROM t1 WHERE a > 0 ORDER BY d NULLS FIRST, id LIMIT 40",
            "SELECT id, d, d AS e FROM t1 ORDER BY a DESC, id LIMIT 2000",
            "SELECT * FROM (SELECT id, d FROM t1 UNION ALL SELECT k, name FROM t2) \
             ORDER BY id, d LIMIT 5100",
        ] {
            let root = assert_parity(sql).await;
            let tree = format_operator_tree(&root);
            assert_eq!(count::<PhysicalTopK>(&root), 1, "{sql}\n{tree}");
        }
        // Rows of `Utf8` values that a projection computes (which can need more
        // bytes than its input), or of nested values, are left to DataFusion.
        for sql in [
            "SELECT id, upper(d) AS u FROM t1 ORDER BY u, id LIMIT 30",
            "SELECT id, arrow_cast(d, 'Binary') AS b FROM t1 ORDER BY b, id LIMIT 30",
            "SELECT id, named_struct('d', d) AS s FROM t1 ORDER BY s, id LIMIT 30",
        ] {
            let root = assert_parity(sql).await;
            assert_eq!(sorts(&root), 0, "{sql}\n{}", format_operator_tree(&root));
        }

        // The fetch, the bytes of the values, and the nodes between the TopK
        // and the in-memory batches.
        let topk = |input: Arc<dyn ExecutionPlan>, fetch: usize| {
            let plan = match input.output_partitioning().partition_count() {
                1 => sort_exec(input, &[("k", false)], false, Some(fetch)),
                _ => {
                    let sort = sort_exec(input, &[("k", false)], true, Some(fetch));
                    merge(sort, &[("k", false)], Some(fetch))
                }
            };
            try_lower_bridged(&plan).is_some()
        };
        let scan = || shared_strings_scan(100, 1 << 20);
        assert!(topk(scan(), MAX_TOPK_ROWS_WITH_OFFSETS));
        assert!(!topk(scan(), MAX_TOPK_ROWS_WITH_OFFSETS + 1));
        // 1023 and 1024 rows of 1 MiB (in a single buffer of 1 MiB).
        assert!(topk(shared_strings_scan(1023, 1 << 20), 2));
        assert!(!topk(shared_strings_scan(1024, 1 << 20), 2));
        let union = |inputs: Vec<Arc<dyn ExecutionPlan>>| -> Arc<dyn ExecutionPlan> {
            UnionExec::try_new(inputs).unwrap()
        };
        assert!(topk(union(vec![scan(), scan()]), 10));
        let half = || shared_strings_scan(512, 1 << 20);
        assert!(!topk(union(vec![half(), half()]), 10));
        let filter = FilterExecBuilder::new(
            Arc::new(BinaryExpr::new(
                col("k", &scan().schema())?,
                Operator::Gt,
                lit(3_i64),
            )),
            scan(),
        )
        .build()?;
        assert!(topk(Arc::new(filter), 10));
        let repartition =
            RepartitionExec::try_new(scan(), Partitioning::RoundRobinBatch(4))?;
        assert!(topk(Arc::new(repartition), 10));
        let projected = |s: Arc<dyn PhysicalExpr>| -> Result<Arc<dyn ExecutionPlan>> {
            let k = col("k", &scan().schema())?;
            let exprs = vec![(k, "k".to_string()), (s, "s".to_string())];
            Ok(Arc::new(ProjectionExec::try_new(exprs, scan())?))
        };
        assert!(topk(projected(col("s", &scan().schema())?)?, 10));
        let computed = Arc::new(CastExpr::new(
            col("k", &scan().schema())?,
            DataType::Utf8,
            None,
        ));
        assert!(!topk(projected(computed)?, 10));
        // Not in memory.
        let limit = Arc::new(GlobalLimitExec::new(scan(), 0, Some(50)));
        assert!(!topk(limit, 10));
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sorts_of_run_end_encoded_strings_are_left_to_datafusion() {
        // DataFusion's sort takes chunks of `batch_size` sorted rows of a run
        // without splitting rows that do not fit: one batch of two runs of a
        // 2 MiB string, by keys that interleave the runs, needs a string per
        // sorted row, 2 GiB in a chunk of 1024 rows, and DataFusion fails with
        // the default `batch_size`. With 64 rows per chunk it would succeed.
        // Sorts of run-end encoded values with 32-bit offsets are left to it.
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, false),
            Field::new("p", run_end_encoded_strings("x").data_type().clone(), true),
        ]));
        let keys = (0..512).map(|i| i * 2).chain((0..512).map(|i| i * 2 + 1));
        let payload = RunArray::<Int32Type>::try_new(
            &Int32Array::from(vec![512, 1024]),
            &StringArray::from(vec!["x".repeat(2 << 20), "y".repeat(2 << 20)]),
        )
        .unwrap();
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int64Array::from_iter_values(keys)),
            Arc::new(payload),
        ];
        let batch = RecordBatch::try_new(Arc::clone(&schema), columns).unwrap();
        let make = || -> Result<Arc<dyn ExecutionPlan>> {
            let scan = MemorySourceConfig::try_new_exec(
                &[vec![batch.clone()]],
                Arc::clone(&schema),
                None,
            )?;
            Ok(sort_exec(scan, &[("k", false)], false, None))
        };
        let message = "Offset overflow error: 2147483648";
        let root = assert_lowered_plan_fails_like_datafusion(
            make,
            lower,
            test_session_config(),
            message,
        )
        .await;
        assert_eq!(sorts(&root), 0);
        // Also merged sorts of partitions, merges with a fetch of a sort, and
        // nested run-end encoded strings. Run-end encoded values without
        // 32-bit offsets stay native.
        let merged = |payload: fn(&str) -> ArrayRef, fetch: Option<usize>| {
            let sort = sort_exec(
                interleaving_payloads_scan(payload),
                &[("k", false)],
                true,
                None,
            );
            try_lower_bridged(&merge(sort, &[("k", false)], fetch)).is_some()
        };
        assert!(!merged(run_end_encoded_strings, None));
        assert!(!merged(run_end_encoded_strings, Some(10)));
        let field = |data_type: DataType| Arc::new(Field::new("x", data_type, true));
        let run_end_encoded = |values: DataType| {
            let run_ends = Arc::new(Field::new("run_ends", DataType::Int32, false));
            DataType::RunEndEncoded(run_ends, field(values))
        };
        let nested = DataType::Struct(
            vec![field(DataType::List(field(run_end_encoded(
                DataType::Utf8,
            ))))]
            .into(),
        );
        assert!(!lowers_sort_of_payload(&nested, None));
        assert!(!lowers_sort_of_payload(
            &run_end_encoded(DataType::Binary),
            None
        ));
        assert!(lowers_sort_of_payload(
            &run_end_encoded(DataType::Float64),
            None
        ));
        assert!(lowers_sort_of_payload(
            &run_end_encoded(DataType::LargeUtf8),
            None
        ));
    }

    /// A scan of the rows `(a, b)` of `partitions`, a list of batches per
    /// partition.
    fn ab_scan(partitions: &[Vec<(Vec<i64>, Vec<i64>)>]) -> Arc<dyn ExecutionPlan> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int64, false),
        ]));
        let partitions = partitions
            .iter()
            .map(|batches| {
                batches
                    .iter()
                    .map(|(a, b)| {
                        RecordBatch::try_new(
                            Arc::clone(&schema),
                            vec![
                                Arc::new(Int64Array::from(a.clone())),
                                Arc::new(Int64Array::from(b.clone())),
                            ],
                        )
                        .unwrap()
                    })
                    .collect()
            })
            .collect::<Vec<_>>();
        MemorySourceConfig::try_new_exec(&partitions, schema, None).unwrap()
    }

    /// The sort keys `a, 10 / b` (`10 / b` fails if `b` is 0).
    fn a_and_ten_over_b(schema: &SchemaRef) -> LexOrdering {
        let ten_over_b =
            BinaryExpr::new(lit(10_i64), Operator::Divide, col("b", schema).unwrap());
        LexOrdering::new([
            PhysicalSortExpr::new(
                col("a", schema).unwrap(),
                SortOptions::new(false, false),
            ),
            PhysicalSortExpr::new(Arc::new(ten_over_b), SortOptions::new(false, false)),
        ])
        .unwrap()
    }

    /// Run `make` with DataFusion and lowered with [`lower`] on `threads`
    /// threads; both must fail with a division by zero if `fails`, and
    /// produce the same rows (in order) otherwise. Returns the lowered tree.
    async fn assert_like_datafusion(
        make: impl Fn() -> Result<Arc<dyn ExecutionPlan>>,
        threads: usize,
        fails: bool,
    ) -> Arc<dyn PhysicalOperator> {
        let config = test_session_config().with_target_partitions(threads);
        if fails {
            assert_lowered_plan_fails_like_datafusion(
                make,
                lower,
                config,
                "Divide by zero",
            )
            .await
        } else {
            let task_ctx = Arc::new(
                datafusion_execution::TaskContext::default().with_session_config(config),
            );
            assert_lowered_plan_byte_eq(make, lower, task_ctx, true).await
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn merges_of_sorts_evaluate_all_keys_of_the_sort() {
        // A merge by `a` of the partitions of a sort by `a, 10 / b`: DataFusion
        // evaluates `10 / b` on all rows, although `a` decides the order.
        for (fails, threads) in [(true, 1), (true, 4), (false, 4)] {
            let make = move || -> Result<Arc<dyn ExecutionPlan>> {
                let b = if fails { 0 } else { 3 };
                let scan = ab_scan(&[
                    vec![((0..100).collect(), vec![1; 100])],
                    vec![(vec![500, 700], vec![b, 2])],
                ]);
                let keys = a_and_ten_over_b(&scan.schema());
                let sort =
                    SortExec::new(keys.clone(), scan).with_preserve_partitioning(true);
                let merge_keys = LexOrdering::new([keys[0].clone()]).unwrap();
                Ok(Arc::new(SortPreservingMergeExec::new(
                    merge_keys,
                    Arc::new(sort),
                )))
            };
            let root = assert_like_datafusion(make, threads, fails).await;
            let sort = root.downcast_ref::<PhysicalSort>().expect("a sort");
            assert_eq!(sort.ordering().len(), 2, "the keys of the SortExec");
        }
        // Over a projection, the native sort evaluates only the keys of the
        // top merge: the keys of the sort must not fail.
        let scan = ab_scan(&[vec![(vec![1], vec![1])]]);
        let keys = a_and_ten_over_b(&scan.schema());
        let sort = Arc::new(SortExec::new(keys, scan).with_preserve_partitioning(true));
        let plan = merge(rename(sort, "b", "c"), &[("a", false)], None);
        assert!(try_lower_bridged(&plan).is_none());
    }

    /// The rows `a = 0..10_000`, in `partitions` partitions of consecutive
    /// values, each in reverse order, with `b = 1`, but `b = 0` for the last
    /// row of the sort, `a = 9999`.
    fn reversed_ab_scan(partitions: usize) -> Arc<dyn ExecutionPlan> {
        let size = 10_000 / partitions as i64;
        let partitions = (0..partitions as i64)
            .map(|partition| {
                let a = (partition * size..(partition + 1) * size)
                    .rev()
                    .collect::<Vec<_>>();
                let b = a.iter().map(|&a| i64::from(a != 9999)).collect();
                vec![(a, b)]
            })
            .collect::<Vec<_>>();
        ab_scan(&partitions)
    }

    /// A projection of `a` and `10 / b AS q` over `input`.
    fn project_ten_over_b(input: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
        let schema = input.schema();
        let ten_over_b =
            BinaryExpr::new(lit(10_i64), Operator::Divide, col("b", &schema).unwrap());
        let exprs: Vec<(Arc<dyn PhysicalExpr>, String)> = vec![
            (col("a", &schema).unwrap(), "a".to_string()),
            (Arc::new(ten_over_b), "q".to_string()),
        ];
        Arc::new(ProjectionExec::try_new(exprs, input).unwrap())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn computed_projections_above_sorts_are_left_to_datafusion() -> Result<()> {
        // DataFusion evaluates a projection between a merge and a sort lazily,
        // on the sorted rows that are read: with a fetch of 1 (of the merge, or
        // of a limit above it), on the first batches of every partition only.
        // `10 / b` fails only for the last row of the sort, `a = 9999`, which
        // DataFusion never projects, so it returns `(0, 10)`. The native sort
        // would project all rows before sorting them.
        let config = test_session_config().with_batch_size(1024);
        let task_ctx = Arc::new(
            datafusion_execution::TaskContext::default().with_session_config(config),
        );
        type Make = Box<dyn Fn() -> Result<Arc<dyn ExecutionPlan>>>;
        for partitions in [1, 4] {
            let sorted = move || {
                let scan = reversed_ab_scan(partitions);
                project_ten_over_b(sort_exec(scan, &[("a", false)], true, None))
            };
            let plans: [Make; 2] = [
                // `SPM(fetch=1) -> Projection[a, 10 / b] -> SortExec[a]`.
                Box::new(move || Ok(merge(sorted(), &[("a", false)], Some(1)))),
                // `GlobalLimitExec(1) -> SPM -> Projection[a, 10 / b] ->
                // SortExec[a]`.
                Box::new(move || {
                    let merged = merge(sorted(), &[("a", false)], None);
                    Ok(Arc::new(GlobalLimitExec::new(merged, 0, Some(1))))
                }),
            ];
            for make in plans {
                let output = collect(make()?, Arc::clone(&task_ctx)).await?;
                let row = (
                    output[0].column(0).as_primitive::<Int64Type>().value(0),
                    output[0].column(1).as_primitive::<Int64Type>().value(0),
                );
                assert_eq!(row, (0, 10), "DataFusion's result");
                let root =
                    assert_lowered_plan_byte_eq(make, lower, Arc::clone(&task_ctx), true)
                        .await;
                assert_eq!(sorts(&root), 0, "{}", format_operator_tree(&root));
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn topks_of_keys_that_can_fail_are_left_to_datafusion() {
        // After 2048 rows with `b = 1`, DataFusion's TopK of `a, 10 / b`
        // evaluates the keys of the whole batch `[(-1, 1), (3000, 0)]`, whose
        // first row comes before its threshold: `10 / 0` fails, although the
        // TopK discards the row `(3000, 0)`.
        for threads in [1, 4] {
            let make = || -> Result<Arc<dyn ExecutionPlan>> {
                let scan = ab_scan(&[vec![
                    ((0..2048).collect(), vec![1; 2048]),
                    (vec![-1, 3000], vec![1, 0]),
                ]]);
                let keys = a_and_ten_over_b(&scan.schema());
                Ok(Arc::new(SortExec::new(keys, scan).with_fetch(Some(10))))
            };
            let root = assert_like_datafusion(make, threads, true).await;
            assert_eq!(sorts(&root), 0, "{}", format_operator_tree(&root));
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn merges_with_a_fetch_of_sorts_of_keys_that_can_fail_sort_all_rows() {
        // A merge with a fetch of a sort without fetch: DataFusion's sort
        // evaluates `10 / b` on all rows, also on the batch whose rows all come
        // after the first 10 rows, which a TopK discards before evaluating its
        // keys. The native operator is a sort with a fetch.
        for (fails, threads) in [(true, 1), (true, 4), (false, 1), (false, 4)] {
            let make = move || -> Result<Arc<dyn ExecutionPlan>> {
                let b = if fails { 0 } else { 5 };
                let scan = ab_scan(&[vec![
                    ((0..2048).rev().collect(), vec![1; 2048]),
                    ((5000..5100).collect(), vec![b; 100]),
                ]]);
                let keys = a_and_ten_over_b(&scan.schema());
                let sort =
                    SortExec::new(keys.clone(), scan).with_preserve_partitioning(true);
                Ok(Arc::new(
                    SortPreservingMergeExec::new(keys, Arc::new(sort))
                        .with_fetch(Some(10)),
                ))
            };
            let root = assert_like_datafusion(make, threads, fails).await;
            let sort = root
                .downcast_ref::<PhysicalSort>()
                .expect("a sort with a fetch");
            assert_eq!(sort.fetch(), Some(10));
        }
        // With keys that cannot fail, it is a TopK.
        let make = || -> Result<Arc<dyn ExecutionPlan>> {
            let scan = ab_scan(&[vec![((0..2048).rev().collect(), vec![1; 2048])]]);
            Ok(merge(
                sort_exec(scan, &[("a", false)], true, None),
                &[("a", false)],
                Some(10),
            ))
        };
        let root = assert_like_datafusion(make, 4, false).await;
        assert!(root.is::<PhysicalTopK>());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_order_by_limit() {
        // (`d` is a `Utf8` column of a `MemTable`, whose TopKs are native, see
        // `topks_of_bounded_strings_are_native`; `h` is a `Utf8View` column.)
        for sql in [
            "SELECT * EXCLUDE (d) FROM t1 ORDER BY id LIMIT 10",
            "SELECT * FROM t1 ORDER BY id DESC LIMIT 1",
            "SELECT id, a FROM t1 ORDER BY a NULLS FIRST, id LIMIT 7",
            "SELECT id, a FROM t1 ORDER BY a DESC NULLS LAST, id LIMIT 25",
            "SELECT id, h FROM t1 ORDER BY h DESC, id LIMIT 777",
            "SELECT id, g FROM t1 ORDER BY g, id LIMIT 50",
            "SELECT id, g FROM t1 ORDER BY g DESC NULLS LAST, id LIMIT 2500",
            "SELECT id, h FROM t1 ORDER BY h, id DESC LIMIT 1024",
            "SELECT id, ts FROM t1 ORDER BY ts DESC, id LIMIT 2049",
            "SELECT * EXCLUDE (d) FROM t1 ORDER BY e NULLS FIRST, f, id LIMIT 4999",
            "SELECT * EXCLUDE (d) FROM t1 ORDER BY b, id LIMIT 5000",
            "SELECT * EXCLUDE (d) FROM t1 ORDER BY b, id LIMIT 6000",
            "SELECT id, a + b AS s FROM t1 ORDER BY s DESC, id LIMIT 33",
            // Keys that cannot fail although they are not columns.
            "SELECT id, a FROM t1 ORDER BY a > 0 DESC, a IS NULL, id LIMIT 30",
            // Ties of rows that are identical in the output.
            "SELECT a, e FROM t1 ORDER BY a, e LIMIT 1000",
            "SELECT h FROM t1 ORDER BY h DESC LIMIT 2600",
            // With an offset.
            "SELECT id, h FROM t1 ORDER BY id LIMIT 10 OFFSET 5",
            "SELECT id, f FROM t1 ORDER BY f DESC, id LIMIT 3000 OFFSET 2047",
            // Rows with `Utf8` values of a `MemTable`.
            "SELECT * FROM t1 ORDER BY id LIMIT 10",
            "SELECT id, d FROM t1 ORDER BY d DESC, id LIMIT 777",
            "SELECT d FROM t1 ORDER BY d DESC LIMIT 2600",
            "SELECT id, d FROM t1 ORDER BY d DESC, id LIMIT 1",
            // Float keys, with `NaN`, `-0.0` and `0.0` (see
            // `topks_of_float_keys_are_native`).
            "SELECT id, c FROM t1 ORDER BY c, id LIMIT 100",
            "SELECT id, c FROM t1 ORDER BY c DESC NULLS LAST, id LIMIT 2500",
            "SELECT c FROM t1 ORDER BY c LIMIT 2300",
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
        // TopKs of keys that can fail to evaluate are left to DataFusion.
        let sql = "SELECT id, b FROM t1 ORDER BY a * 2 + 1 DESC, id LIMIT 33";
        let root = assert_parity(sql).await;
        assert_eq!(sorts(&root), 0, "{sql}\n{}", format_operator_tree(&root));
        // A limit of the sorted rows.
        let root = assert_parity("SELECT id FROM t1 ORDER BY id LIMIT 5 OFFSET 3").await;
        assert_eq!(count::<PhysicalLimit>(&root), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_random_order_by() {
        // Random sort keys (columns of every type and expressions), directions
        // and null orders, filters, limits and offsets, on 1 to 16 target
        // partitions. `id` breaks the ties.
        let mut rng = StdRng::seed_from_u64(99);
        let keys = [
            "id", "a", "b", "c", "d", "e", "f", "g", "h", "ts", "a + b", "upper(d)",
            "a % 7", "c * 2",
        ];
        let filters = ["", " WHERE a > 0", " WHERE id % 3 = 1", " WHERE b < 5000"];
        for _ in 0..64 {
            let ordering = (0..rng.random_range(1..4))
                .map(|_| {
                    let key = keys[rng.random_range(0..keys.len())];
                    let direction = ["", " ASC", " DESC"][rng.random_range(0..3)];
                    let nulls =
                        ["", " NULLS FIRST", " NULLS LAST"][rng.random_range(0..3)];
                    format!("{key}{direction}{nulls}")
                })
                .collect::<Vec<_>>()
                .join(", ");
            let filter = filters[rng.random_range(0..filters.len())];
            let limit = match rng.random_range(0..4) {
                0 => String::new(),
                1 => format!(" LIMIT {}", rng.random_range(1..20)),
                2 => format!(" LIMIT {}", rng.random_range(1..6000)),
                _ => format!(
                    " LIMIT {} OFFSET {}",
                    rng.random_range(1..3000),
                    rng.random_range(0..3000)
                ),
            };
            let sql = format!(
                "SELECT id, a, c, d FROM t1{filter} ORDER BY {ordering}, id{limit}"
            );
            let target_partitions = [1, 3, 4, 16][rng.random_range(0..4)];
            let config = test_session_config().with_target_partitions(target_partitions);
            assert_parity_with_config(&sql, config).await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_order_by_of_aggregates_joins_and_subqueries() {
        // Projections between the merge and the sort, over native
        // aggregations.
        for sql in [
            "SELECT d, COUNT(*) AS n FROM t1 GROUP BY d ORDER BY d",
            "SELECT h, COUNT(*) AS n FROM t1 GROUP BY h ORDER BY n DESC, h LIMIT 4",
            "SELECT d AS x, SUM(b) AS s FROM t1 GROUP BY d ORDER BY x DESC",
            "SELECT e, f, COUNT(DISTINCT a) AS da FROM t1 GROUP BY e, f ORDER BY f, e",
            "SELECT a, SUM(b) AS s FROM t1 GROUP BY a ORDER BY s DESC, a LIMIT 20",
            // A projection that cannot fail although it computes a value.
            "SELECT d, COUNT(*) > 800 AS big, COUNT(*) AS n FROM t1 GROUP BY d ORDER BY d",
        ] {
            let root = assert_parity(sql).await;
            let tree = format_operator_tree(&root);
            assert!(sorts(&root) > 0, "{sql}\n{tree}");
            assert!(count::<PhysicalHashAggregate>(&root) > 0, "{sql}\n{tree}");
        }
        assert_sorted_natively(&[
            // DataFusion keeps the top groups only in the aggregation of a
            // top-k aggregation, which the native aggregation leaves to it.
            "SELECT h, COUNT(*) AS n FROM t1 GROUP BY h ORDER BY h DESC LIMIT 3",
            "SELECT a, MAX(id) AS m FROM t1 GROUP BY a ORDER BY m DESC LIMIT 20",
            "SELECT t1.id, t2.w FROM t1 JOIN t2 ON t1.a = t2.k ORDER BY t1.id, t2.w LIMIT 50",
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
            "SELECT * EXCLUDE (d) FROM t_empty ORDER BY g DESC, id LIMIT 5",
            "SELECT * FROM t1 WHERE id < 0 ORDER BY a",
            "SELECT * EXCLUDE (d) FROM t1 WHERE id < 0 ORDER BY a LIMIT 3",
            "SELECT * FROM t1 WHERE id = 42 ORDER BY a",
            "SELECT * EXCLUDE (d) FROM t1 WHERE id = 4999 ORDER BY a DESC LIMIT 3",
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
                "SELECT id, h FROM t1 ORDER BY h DESC, id LIMIT 100",
                "SELECT h, COUNT(*) AS n FROM t1 GROUP BY h ORDER BY n, h LIMIT 3",
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

    /// Assert that `sql` (on sessions configured with `config` and set up
    /// with `setup`) fails with an error containing `fails` in DataFusion and
    /// lowered with [`lower`], or produces the same bytes if `fails` is
    /// `None`. Returns the lowered operator tree.
    async fn assert_outcome_like_datafusion<F>(
        sql: &str,
        config: SessionConfig,
        setup: F,
        fails: Option<&str>,
    ) -> Arc<dyn PhysicalOperator>
    where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        let Some(message) = fails else {
            return assert_native_byte_eq(sql, config, setup).await;
        };
        let ctx = datafusion_context(config);
        setup(&ctx).await.expect("setup");
        let error = run_sql(&ctx, sql).await.expect_err("DataFusion fails");
        assert!(error.to_string().contains(message), "DataFusion: {error}");
        let plan = ctx
            .sql(sql)
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        let root = lower(&plan).unwrap();
        let exec = Arc::new(PipelineRootExec::try_new(Arc::clone(&root), plan).unwrap());
        let error = collect(exec, ctx.task_ctx())
            .await
            .expect_err("the lowered plan fails");
        assert!(error.to_string().contains(message), "native: {error}");
        root
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn topks_whose_dynamic_filter_prunes_work_that_can_fail_are_left_to_datafusion()
    -> Result<()> {
        // 20 Parquet files of 500 rows `(a, b)`, in order of `a`, with `b = 1`
        // but `b = 0` for one row of the second file. The dynamic filter of the
        // TopK prunes the files that its threshold rules out: DataFusion's
        // TopK, on a single partition, tightens it after the first file, so
        // its scan never evaluates `10 / b` on the second one. The native TopK
        // tightens it later (and its scan reads files on several threads), so
        // which rows the scan evaluates `10 / b` on differs: such TopKs are
        // left to DataFusion.
        let dir = tempfile::tempdir()?;
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Int64, false),
        ]));
        for file in 0..20_i64 {
            let a = (file * 500..(file + 1) * 500).collect::<Vec<_>>();
            let b = a.iter().map(|&a| i64::from(a != 750)).collect::<Vec<_>>();
            let batch = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(Int64Array::from(a)), Arc::new(Int64Array::from(b))],
            )?;
            let path = dir.path().join(format!("p/{file:02}.parquet"));
            write_parquet(&path, Arc::clone(&schema), &[batch], None)?;
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
        let config = test_session_config().with_target_partitions(1);
        for (sql, fails) in [
            // `10 / b` is pushed into the scan: DataFusion never evaluates it on
            // the second file.
            ("SELECT a, 10 / b AS q FROM p ORDER BY a LIMIT 1", false),
            // DataFusion's filter evaluates it on the second file (read before
            // the threshold is known).
            ("SELECT a FROM p WHERE 10 / b > 1 ORDER BY a LIMIT 1", true),
        ] {
            let fails = fails.then_some("Divide by zero");
            let root =
                assert_outcome_like_datafusion(sql, config.clone(), &setup, fails).await;
            assert_eq!(sorts(&root), 0, "{sql}\n{}", format_operator_tree(&root));
        }
        // Work that cannot fail: the native TopK updates the dynamic filter.
        for sql in [
            "SELECT a, b FROM p ORDER BY a LIMIT 1",
            "SELECT a FROM p WHERE b > 0 AND a IS NOT NULL ORDER BY a DESC LIMIT 3",
        ] {
            let root = assert_native_byte_eq(sql, config.clone(), &setup).await;
            assert_eq!(
                count::<PhysicalTopK>(&root),
                1,
                "{sql}\n{}",
                format_operator_tree(&root)
            );
        }
        Ok(())
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
                "SELECT * FROM p ORDER BY d NULLS FIRST, id DESC LIMIT 1700",
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
        let ctx = datafusion_context(test_session_config().with_target_partitions(1));
        setup(&ctx).await?;
        let plan = ctx
            .sql("SELECT * FROM p ORDER BY id LIMIT 10")
            .await?
            .create_physical_plan()
            .await?;
        let root = lower(&plan)?;
        let exec = Arc::new(PipelineRootExec::try_new(Arc::clone(&root), plan)?);
        let output =
            collect(Arc::clone(&exec) as Arc<dyn ExecutionPlan>, ctx.task_ctx()).await?;
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
