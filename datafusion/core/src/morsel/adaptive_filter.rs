// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! DuckDB-inspired adaptive conjunct reordering for multi-conjunct `AND`
//! filters in the native morsel execution path.

use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::{DataType, Schema};
use datafusion_common::cast::as_boolean_array;
use datafusion_common::instant::Instant;
use datafusion_common::tree_node::TreeNode;
use datafusion_common::{Result, ScalarValue, internal_err};
use datafusion_expr::{ColumnarValue, Operator as ExprOperator};
use datafusion_physical_expr::expressions::{
    BinaryExpr, Column, InListExpr, IsNotNullExpr, IsNullExpr, LikeExpr, Literal, NotExpr,
};
use datafusion_physical_expr::scalar_subquery::ScalarSubqueryExpr;
use datafusion_physical_expr::utils::split_conjunction;
use datafusion_physical_expr::{PhysicalExpr, conjunction};
use datafusion_physical_expr_common::physical_expr::is_volatile;

use super::batch::{
    BatchView, Selection, collect_expr_column_indices, effective_boolean_buffer,
    remap_expr_columns,
};
#[cfg(any(feature = "parquet", test))]
use super::join_filter::JoinBloomFilterExpr;

const DEFAULT_WARMUP_ITERATIONS: usize = 5;
const DEFAULT_EXECUTE_INTERVAL: usize = 20;
const DEFAULT_OBSERVE_INTERVAL: usize = 10;
const INITIAL_SWAP_LIKELINESS: u32 = 100;
const DEFAULT_PRNG_SEED: u64 = 0x9e37_79b9_7f4a_7c15;

/// DuckDB-style observe/execute adjacent-swap bandit for reordering independent
/// `AND` filter conjuncts at runtime.
///
/// Follows the state machine of DuckDB's `AdaptiveFilter` (`adaptive_filter.cpp`):
/// after a brief warmup, it runs for `execute_interval` iterations to establish
/// `prev_mean`, randomly chooses an adjacent pair `(swap_idx, swap_idx + 1)`
/// gated by `swap_likeliness[swap_idx]`, and evaluates the swapped order for
/// `observe_interval` iterations. Improvements keep the swap and reset
/// likeliness to `100`; regressions revert the swap and halve likeliness
/// (floored at `1`).
#[derive(Debug, Clone)]
pub(super) struct AdaptiveFilter {
    permutation: Vec<usize>,
    swap_likeliness: Vec<u32>,
    disable_permutations: bool,
    iteration_count: usize,
    swap_idx: usize,
    right_random_border: u32,
    warmup_iterations: usize,
    observe_interval: usize,
    execute_interval: usize,
    runtime_sum: f64,
    prev_mean: f64,
    observe: bool,
    warmup: bool,
    rng_state: u64,
}

impl AdaptiveFilter {
    pub(super) fn new(
        initial_permutation: Vec<usize>,
        disable_permutations: bool,
    ) -> Self {
        Self::with_intervals(
            initial_permutation,
            disable_permutations,
            DEFAULT_WARMUP_ITERATIONS,
            DEFAULT_EXECUTE_INTERVAL,
            DEFAULT_OBSERVE_INTERVAL,
        )
    }

    pub(super) fn with_intervals(
        initial_permutation: Vec<usize>,
        disable_permutations: bool,
        warmup_iterations: usize,
        execute_interval: usize,
        observe_interval: usize,
    ) -> Self {
        let len = initial_permutation.len();
        let disable = disable_permutations || len <= 1;
        let pair_count = len.saturating_sub(1);
        Self {
            permutation: initial_permutation,
            swap_likeliness: vec![INITIAL_SWAP_LIKELINESS; pair_count],
            disable_permutations: disable,
            iteration_count: 0,
            swap_idx: 0,
            right_random_border: (INITIAL_SWAP_LIKELINESS as usize * pair_count) as u32,
            warmup_iterations,
            observe_interval: observe_interval.max(1),
            execute_interval: execute_interval.max(1),
            runtime_sum: 0.0,
            prev_mean: 0.0,
            observe: false,
            warmup: warmup_iterations > 0,
            rng_state: DEFAULT_PRNG_SEED,
        }
    }

    #[cfg(test)]
    pub(super) fn with_seed(mut self, seed: u64) -> Self {
        self.rng_state = if seed == 0 { DEFAULT_PRNG_SEED } else { seed };
        self
    }

    #[inline]
    pub(super) fn permutation(&self) -> &[usize] {
        &self.permutation
    }

    #[inline]
    #[cfg(test)]
    pub(super) fn disable_permutations(&self) -> bool {
        self.disable_permutations
    }

    #[inline]
    #[cfg(test)]
    pub(super) fn swap_likeliness(&self) -> &[u32] {
        &self.swap_likeliness
    }

    #[inline]
    pub(super) fn begin_filter(&self) -> Option<Instant> {
        if self.disable_permutations {
            return None;
        }
        Some(Instant::now())
    }

    #[inline]
    pub(super) fn end_filter(&mut self, start: Option<Instant>, num_rows: usize) {
        let Some(start) = start else {
            return;
        };
        if num_rows == 0 {
            return;
        }
        let cost_per_row = start.elapsed().as_nanos() as f64 / num_rows as f64;
        self.adapt_runtime_statistics(cost_per_row);
    }

    pub(super) fn adapt_runtime_statistics(&mut self, cost: f64) {
        if self.disable_permutations {
            return;
        }
        self.iteration_count += 1;
        self.runtime_sum += cost;

        if self.warmup {
            if self.iteration_count == self.warmup_iterations {
                self.iteration_count = 0;
                self.runtime_sum = 0.0;
                self.observe = false;
                self.warmup = false;
            }
            return;
        }

        if self.observe && self.iteration_count == self.observe_interval {
            let trial_mean = self.runtime_sum / self.iteration_count as f64;
            if self.prev_mean - trial_mean <= 0.0 {
                self.permutation.swap(self.swap_idx, self.swap_idx + 1);
                if self.swap_likeliness[self.swap_idx] > 1 {
                    self.swap_likeliness[self.swap_idx] /= 2;
                }
            } else {
                self.swap_likeliness[self.swap_idx] = INITIAL_SWAP_LIKELINESS;
            }
            self.observe = false;
            self.iteration_count = 0;
            self.runtime_sum = 0.0;
        } else if !self.observe && self.iteration_count == self.execute_interval {
            self.prev_mean = self.runtime_sum / self.iteration_count as f64;
            let random_number = self.next_random_u32(self.right_random_border);
            self.swap_idx = (random_number / INITIAL_SWAP_LIKELINESS) as usize;
            let likeliness =
                random_number - INITIAL_SWAP_LIKELINESS * self.swap_idx as u32;
            if self.swap_likeliness[self.swap_idx] > likeliness {
                self.permutation.swap(self.swap_idx, self.swap_idx + 1);
                self.observe = true;
            }
            self.iteration_count = 0;
            self.runtime_sum = 0.0;
        }
    }

    fn next_random_u32(&mut self, upper_exclusive: u32) -> u32 {
        debug_assert!(upper_exclusive > 0);
        // xorshift64* for deterministic, dependency-free uniform sampling.
        let mut x = self.rng_state;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.rng_state = x;
        let mixed = x.wrapping_mul(0x2545_f491_4f6c_dd1d);
        ((mixed >> 32) as u32) % upper_exclusive
    }
}

/// Returns `true` iff `expr` is non-volatile, contains no scalar subqueries,
/// and cannot throw an error on any valid Arrow input values.
///
/// Strictly conservative: arithmetic (`+`, `-`, `*`, `/`, `%`), `CAST`,
/// `TryCast`, regex matching, `CASE`, scalar UDFs, and unknown `PhysicalExpr`
/// implementations return `false` so left-to-right short-circuiting is never
/// violated when any conjunct could fail.
pub(super) fn is_infallible_conjunct(expr: &Arc<dyn PhysicalExpr>) -> bool {
    if is_volatile(expr) {
        return false;
    }
    if expr
        .exists(|e| Ok(e.is::<ScalarSubqueryExpr>()))
        .unwrap_or(true)
    {
        return false;
    }
    is_infallible_node(expr)
}

fn is_infallible_node(expr: &Arc<dyn PhysicalExpr>) -> bool {
    if expr.is::<Column>() || expr.is::<Literal>() {
        return true;
    }
    #[cfg(any(feature = "parquet", test))]
    if let Some(bloom) = expr.downcast_ref::<JoinBloomFilterExpr>() {
        return bloom.children().into_iter().all(is_infallible_node);
    }
    if let Some(e) = expr.downcast_ref::<IsNullExpr>() {
        return is_infallible_node(e.arg());
    }
    if let Some(e) = expr.downcast_ref::<IsNotNullExpr>() {
        return is_infallible_node(e.arg());
    }
    if let Some(e) = expr.downcast_ref::<NotExpr>() {
        return is_infallible_node(e.arg());
    }
    if let Some(e) = expr.downcast_ref::<InListExpr>() {
        return is_infallible_node(e.expr()) && e.list().iter().all(is_infallible_node);
    }
    if let Some(e) = expr.downcast_ref::<LikeExpr>() {
        return is_infallible_node(e.expr()) && is_infallible_node(e.pattern());
    }
    if let Some(e) = expr.downcast_ref::<BinaryExpr>() {
        return matches!(
            e.op(),
            ExprOperator::Eq
                | ExprOperator::NotEq
                | ExprOperator::Lt
                | ExprOperator::LtEq
                | ExprOperator::Gt
                | ExprOperator::GtEq
                | ExprOperator::IsDistinctFrom
                | ExprOperator::IsNotDistinctFrom
                | ExprOperator::And
                | ExprOperator::Or
        ) && is_infallible_node(e.left())
            && is_infallible_node(e.right());
    }
    false
}

fn data_type_comparison_weight(dt: &DataType) -> u64 {
    match dt {
        DataType::Boolean
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Date32
        | DataType::Date64
        | DataType::Float16
        | DataType::Float32
        | DataType::Float64
        | DataType::Time32(_)
        | DataType::Time64(_)
        | DataType::Timestamp(_, _)
        | DataType::Duration(_) => 1,
        DataType::Decimal32(_, _) | DataType::Decimal64(_, _) => 1,
        DataType::Decimal128(_, _) | DataType::Decimal256(_, _) => 2,
        DataType::Utf8
        | DataType::LargeUtf8
        | DataType::Utf8View
        | DataType::Binary
        | DataType::LargeBinary
        | DataType::BinaryView
        | DataType::FixedSizeBinary(_) => 5,
        DataType::Dictionary(_, val) => data_type_comparison_weight(val),
        _ => 4,
    }
}

/// Cheap-first static cost heuristic inspired by DuckDB's `ExpressionHeuristics`.
/// Lower cost sorts earlier in the initial permutation.
pub(super) fn estimate_conjunct_cost(
    expr: &Arc<dyn PhysicalExpr>,
    schema: &Schema,
) -> u64 {
    if expr.is::<IsNullExpr>() || expr.is::<IsNotNullExpr>() {
        return 1;
    }
    #[cfg(any(feature = "parquet", test))]
    if let Some(bloom) = expr.downcast_ref::<JoinBloomFilterExpr>() {
        // L1/L2-resident Bloom filters come from small/selective build sides;
        // multi-MB Bloom filters spill to L3/DRAM and are typically much less
        // selective (e.g. unfiltered dimension tables).
        let bytes = bloom.filter_memory_bytes();
        return if bytes <= 64 * 1024 {
            3
        } else if bytes <= 512 * 1024 {
            6
        } else {
            20
        };
    }
    if let Some(not) = expr.downcast_ref::<NotExpr>() {
        return estimate_conjunct_cost(not.arg(), schema).saturating_add(1);
    }
    if expr.is::<LikeExpr>() {
        return 200;
    }
    if let Some(in_list) = expr.downcast_ref::<InListExpr>() {
        let weight = in_list
            .expr()
            .data_type(schema)
            .map(|dt| data_type_comparison_weight(&dt))
            .unwrap_or(2);
        let list_penalty = if in_list.list().len() <= 8 { 12 } else { 20 };
        return list_penalty * weight;
    }
    if let Some(bin) = expr.downcast_ref::<BinaryExpr>() {
        match bin.op() {
            ExprOperator::And => {
                let l = estimate_conjunct_cost(bin.left(), schema);
                let r = estimate_conjunct_cost(bin.right(), schema);
                return l.min(r).saturating_add(2);
            }
            ExprOperator::Or => {
                let l = estimate_conjunct_cost(bin.left(), schema);
                let r = estimate_conjunct_cost(bin.right(), schema);
                return l.saturating_add(r).saturating_add(20);
            }
            ExprOperator::RegexMatch
            | ExprOperator::RegexIMatch
            | ExprOperator::RegexNotMatch
            | ExprOperator::RegexNotIMatch => return 300,
            op => {
                let op_base = match op {
                    ExprOperator::Eq | ExprOperator::IsNotDistinctFrom => 5,
                    ExprOperator::Lt
                    | ExprOperator::LtEq
                    | ExprOperator::Gt
                    | ExprOperator::GtEq => 8,
                    ExprOperator::NotEq | ExprOperator::IsDistinctFrom => 12,
                    _ => 25,
                };
                let weight = bin
                    .left()
                    .data_type(schema)
                    .or_else(|_| bin.right().data_type(schema))
                    .map(|dt| data_type_comparison_weight(&dt))
                    .unwrap_or(2);
                return op_base * weight;
            }
        }
    }
    100
}

/// Shared analyzer passed into `datasource-parquet` so `RowFilter` candidate
/// ordering uses the exact same infallibility and cost heuristic (including
/// native recognition of `JoinBloomFilterExpr`).
#[cfg(feature = "parquet")]
pub(super) fn analyze_conjunct(
    expr: &Arc<dyn PhysicalExpr>,
    schema: &Schema,
) -> (bool, u64) {
    (
        is_infallible_conjunct(expr),
        estimate_conjunct_cost(expr, schema),
    )
}

fn expr_is_expensive(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> bool {
    estimate_conjunct_cost(expr, schema) >= 25
}

#[derive(Debug, Clone)]
struct PreparedConjunct {
    expr: Arc<dyn PhysicalExpr>,
    needed_cols: Vec<usize>,
    rebound_expr: Arc<dyn PhysicalExpr>,
    is_expensive: bool,
}

impl PreparedConjunct {
    fn try_new(expr: Arc<dyn PhysicalExpr>, schema: &Schema) -> Result<Self> {
        let needed_cols = collect_expr_column_indices(std::slice::from_ref(&expr));
        let rebound_expr = remap_expr_columns(Arc::clone(&expr), &needed_cols)?;
        let is_expensive = expr_is_expensive(&expr, schema);
        Ok(Self {
            expr,
            needed_cols,
            rebound_expr,
            is_expensive,
        })
    }

    fn eval_on_all(&self, batch: RecordBatch) -> Result<BatchView> {
        match self.expr.evaluate(&batch)? {
            ColumnarValue::Scalar(ScalarValue::Boolean(Some(true))) => {
                Ok(BatchView::all(batch))
            }
            ColumnarValue::Scalar(ScalarValue::Boolean(Some(false) | None)) => {
                Ok(BatchView::empty(batch.schema()))
            }
            ColumnarValue::Scalar(_) => {
                internal_err!("Cannot create filter_array from non-boolean predicates")
            }
            ColumnarValue::Array(array) => {
                let Ok(mask) = as_boolean_array(&array) else {
                    return internal_err!(
                        "Cannot create filter_array from non-boolean predicates"
                    );
                };
                BatchView::from_boolean_array(batch, mask)
            }
        }
    }

    fn eval_compact(&self, view: BatchView) -> Result<BatchView> {
        let compact = view.materialize_columns(&self.needed_cols)?;
        match self.rebound_expr.evaluate(&compact)? {
            ColumnarValue::Scalar(ScalarValue::Boolean(Some(true))) => Ok(view),
            ColumnarValue::Scalar(ScalarValue::Boolean(Some(false) | None)) => {
                Ok(BatchView::empty(view.schema()))
            }
            ColumnarValue::Scalar(_) => {
                internal_err!("Cannot create filter_array from non-boolean predicates")
            }
            ColumnarValue::Array(array) => {
                let Ok(mask) = as_boolean_array(&array) else {
                    return internal_err!(
                        "Cannot create filter_array from non-boolean predicates"
                    );
                };
                let physical_rows = view.batch().num_rows();
                let new_selection =
                    view.selection().compose_compact_mask(physical_rows, mask)?;
                if new_selection.is_empty(physical_rows) {
                    Ok(BatchView::empty(view.schema()))
                } else {
                    let (batch, _) = view.into_parts();
                    BatchView::try_new(batch, new_selection)
                }
            }
        }
    }

    fn eval_mask_and(&self, view: BatchView) -> Result<BatchView> {
        let (batch, selection) = view.into_parts();
        let Selection::Mask { bits, .. } = selection else {
            return self.eval_compact(BatchView::try_new(batch, selection)?);
        };
        match self.expr.evaluate(&batch)? {
            ColumnarValue::Scalar(ScalarValue::Boolean(Some(true))) => {
                let selected = bits.count_set_bits();
                BatchView::try_new(batch, Selection::Mask { bits, selected })
            }
            ColumnarValue::Scalar(ScalarValue::Boolean(Some(false) | None)) => {
                Ok(BatchView::empty(batch.schema()))
            }
            ColumnarValue::Scalar(_) => {
                internal_err!("Cannot create filter_array from non-boolean predicates")
            }
            ColumnarValue::Array(array) => {
                let Ok(mask) = as_boolean_array(&array) else {
                    return internal_err!(
                        "Cannot create filter_array from non-boolean predicates"
                    );
                };
                let physical_rows = batch.num_rows();
                if mask.len() != physical_rows {
                    // Fall back to standard validation/slicing on mismatched masks.
                    let selected = bits.count_set_bits();
                    return self.eval_compact(BatchView::try_new(
                        batch,
                        Selection::Mask { bits, selected },
                    )?);
                }
                let combined = &bits & &effective_boolean_buffer(mask);
                let selected = combined.count_set_bits();
                if selected == 0 {
                    Ok(BatchView::empty(batch.schema()))
                } else if selected == physical_rows {
                    Ok(BatchView::all(batch))
                } else {
                    BatchView::try_new(
                        batch,
                        Selection::Mask {
                            bits: combined,
                            selected,
                        },
                    )
                }
            }
        }
    }

    fn evaluate(
        &self,
        view: BatchView,
        allow_full_eval_mask_and: bool,
    ) -> Result<BatchView> {
        if view.is_empty() {
            return Ok(BatchView::empty(view.schema()));
        }
        if view.is_all() {
            let (batch, _) = view.into_parts();
            return self.eval_on_all(batch);
        }
        if allow_full_eval_mask_and
            && let Selection::Mask { selected, .. } = view.selection()
        {
            let total = view.batch().num_rows();
            let prefer_full_mask_and = if self.is_expensive {
                // Expensive predicates (LIKE, string InList, Bloom) benefit
                // from compaction as soon as < 50% of rows survive.
                *selected * 2 >= total
            } else {
                // Cheap fixed-width comparisons run at SIMD speed; only pay
                // compaction + scatter overhead when < 12.5% of rows survive.
                *selected * 8 >= total
            };
            if prefer_full_mask_and {
                return self.eval_mask_and(view);
            }
        }
        self.eval_compact(view)
    }
}

/// Immutable compiled filter operator supporting DuckDB-style per-worker
/// adaptive conjunct reordering on infallible `AND` predicates while strictly
/// preserving left-to-right short-circuit order whenever any conjunct can throw.
#[derive(Debug, Clone)]
pub(super) struct CompiledFilter {
    conjuncts: Vec<PreparedConjunct>,
    initial_permutation: Vec<usize>,
    all_infallible: bool,
}

impl CompiledFilter {
    pub(super) fn try_new(
        predicate: Arc<dyn PhysicalExpr>,
        schema: &Schema,
    ) -> Result<Self> {
        let raw_conjuncts: Vec<Arc<dyn PhysicalExpr>> =
            split_conjunction(&predicate).into_iter().cloned().collect();
        if raw_conjuncts.len() <= 1 {
            let single = PreparedConjunct::try_new(predicate, schema)?;
            return Ok(Self {
                conjuncts: vec![single],
                initial_permutation: vec![0],
                all_infallible: false,
            });
        }

        let all_infallible = raw_conjuncts.iter().all(is_infallible_conjunct);
        if !all_infallible {
            // Exact DuckDB safety rule: if ANY conjunct can throw or is
            // volatile, disable permutation and preserve exact left-to-right
            // short-circuit order.
            let conjuncts = raw_conjuncts
                .into_iter()
                .map(|expr| PreparedConjunct::try_new(expr, schema))
                .collect::<Result<Vec<_>>>()?;
            let initial_permutation = (0..conjuncts.len()).collect();
            return Ok(Self {
                conjuncts,
                initial_permutation,
                all_infallible: false,
            });
        }

        // Fuse infallible conjuncts that reference the exact same non-empty
        // column set (e.g. `l_shipdate >= ... AND l_shipdate < ...`) so range
        // bounds on one column stay cache-local in a single evaluation pass.
        type ConjunctGroup = (Vec<usize>, Vec<Arc<dyn PhysicalExpr>>);
        let mut grouped: Vec<ConjunctGroup> = Vec::with_capacity(raw_conjuncts.len());
        for expr in raw_conjuncts {
            let cols = collect_expr_column_indices(std::slice::from_ref(&expr));
            if !cols.is_empty()
                && let Some((_, group_exprs)) =
                    grouped.iter_mut().find(|(g_cols, _)| g_cols == &cols)
            {
                group_exprs.push(expr);
            } else {
                grouped.push((cols, vec![expr]));
            }
        }

        let fused_exprs: Vec<Arc<dyn PhysicalExpr>> = grouped
            .into_iter()
            .map(|(_, exprs)| conjunction(exprs))
            .collect();

        let conjuncts = fused_exprs
            .into_iter()
            .map(|expr| PreparedConjunct::try_new(expr, schema))
            .collect::<Result<Vec<_>>>()?;

        let mut initial_permutation: Vec<usize> = (0..conjuncts.len()).collect();
        if conjuncts.len() > 1 {
            let costs: Vec<u64> = conjuncts
                .iter()
                .map(|c| estimate_conjunct_cost(&c.expr, schema))
                .collect();
            initial_permutation.sort_by_key(|&idx| (costs[idx], idx));
        }

        Ok(Self {
            conjuncts,
            initial_permutation,
            all_infallible: true,
        })
    }

    pub(super) fn new_worker_state(&self) -> AdaptiveFilter {
        let disable = !self.all_infallible || self.conjuncts.len() <= 1;
        AdaptiveFilter::new(self.initial_permutation.clone(), disable)
    }

    #[cfg(test)]
    pub(super) fn new_worker_state_with_intervals(
        &self,
        warmup: usize,
        execute_interval: usize,
        observe_interval: usize,
    ) -> AdaptiveFilter {
        let disable = !self.all_infallible || self.conjuncts.len() <= 1;
        AdaptiveFilter::with_intervals(
            self.initial_permutation.clone(),
            disable,
            warmup,
            execute_interval,
            observe_interval,
        )
    }

    #[cfg(test)]
    pub(super) fn num_conjuncts(&self) -> usize {
        self.conjuncts.len()
    }

    pub(super) fn evaluate_view(
        &self,
        mut view: BatchView,
        adaptive: &mut AdaptiveFilter,
    ) -> Result<BatchView> {
        if view.is_empty() {
            return Ok(BatchView::empty(view.schema()));
        }
        if self.conjuncts.len() == 1 {
            return view.filter(&self.conjuncts[0].expr);
        }

        let input_rows = view.num_rows();
        let start = adaptive.begin_filter();
        let perm_len = adaptive.permutation().len();
        debug_assert_eq!(perm_len, self.conjuncts.len());
        for pos in 0..perm_len {
            let idx = adaptive.permutation()[pos];
            view = self.conjuncts[idx].evaluate(view, self.all_infallible)?;
            if view.is_empty() {
                break;
            }
        }
        adaptive.end_filter(start, input_rows);
        Ok(view)
    }

    pub(super) fn count_filtered_rows(
        &self,
        batch: &RecordBatch,
        adaptive: &mut AdaptiveFilter,
    ) -> Result<usize> {
        if self.conjuncts.len() == 1 {
            return super::count::count_filtered_rows(batch, &self.conjuncts[0].expr);
        }
        let view = self.evaluate_view(BatchView::all(batch.clone()), adaptive)?;
        Ok(view.num_rows())
    }
}

/// Per-worker cache of compiled adaptive filters indexed by pipeline operator
/// position. Built lazily on the first batch seen by each filter operator so
/// schemas after intervening `Projection` or `HashProbe` operators are always
/// exact, with zero cross-thread lock contention.
#[derive(Debug, Default)]
pub(super) struct WorkerFilterState {
    slots: Vec<Option<(CompiledFilter, AdaptiveFilter)>>,
}

impl WorkerFilterState {
    fn get_or_init(
        &mut self,
        op_idx: usize,
        predicate: &Arc<dyn PhysicalExpr>,
        schema: &Schema,
    ) -> Result<&mut (CompiledFilter, AdaptiveFilter)> {
        if self.slots.len() <= op_idx {
            self.slots.resize_with(op_idx + 1, || None);
        }
        if self.slots[op_idx].is_none() {
            let compiled = CompiledFilter::try_new(Arc::clone(predicate), schema)?;
            let adaptive = compiled.new_worker_state();
            self.slots[op_idx] = Some((compiled, adaptive));
        }
        Ok(self.slots[op_idx].as_mut().expect("initialized above"))
    }

    pub(super) fn is_multi_conjunct(
        &mut self,
        op_idx: usize,
        predicate: &Arc<dyn PhysicalExpr>,
        schema: &Schema,
    ) -> Result<bool> {
        let (compiled, _) = self.get_or_init(op_idx, predicate, schema)?;
        Ok(compiled.conjuncts.len() > 1)
    }

    pub(super) fn evaluate_view(
        &mut self,
        op_idx: usize,
        predicate: &Arc<dyn PhysicalExpr>,
        view: BatchView,
    ) -> Result<BatchView> {
        if view.is_empty() {
            return Ok(BatchView::empty(view.schema()));
        }
        let schema = view.batch().schema();
        let (compiled, adaptive) =
            self.get_or_init(op_idx, predicate, schema.as_ref())?;
        compiled.evaluate_view(view, adaptive)
    }

    pub(super) fn count_filtered_rows(
        &mut self,
        op_idx: usize,
        predicate: &Arc<dyn PhysicalExpr>,
        batch: &RecordBatch,
    ) -> Result<usize> {
        if batch.num_rows() == 0 {
            return Ok(0);
        }
        let schema = batch.schema();
        let (compiled, adaptive) =
            self.get_or_init(op_idx, predicate, schema.as_ref())?;
        compiled.count_filtered_rows(batch, adaptive)
    }
}

#[cfg(test)]
mod tests {
    use std::fmt::{self, Display, Formatter};
    use std::hash::Hash;

    use arrow::array::{BooleanArray, Int32Array, StringArray};
    use arrow::datatypes::Field;
    use datafusion_expr::{col, lit};
    use datafusion_physical_expr::expressions::CastExpr;
    use datafusion_physical_expr::planner::logical2physical;

    use super::*;

    #[test]
    fn adaptive_bandit_reorders_and_converges_deterministically() {
        // Conjunct 0: cost 100.0, selectivity 0.90 (expensive + unselective)
        // Conjunct 1: cost 1.0, selectivity 0.05 (cheap + selective)
        // Order [0, 1] expected cost = 100.0 + 0.90 * 1.0 = 100.9
        // Order [1, 0] expected cost = 1.0 + 0.05 * 100.0 = 6.0
        let expected_cost = |perm: &[usize]| -> f64 {
            let costs = [100.0, 1.0];
            let sels = [0.90, 0.05];
            let mut total = 0.0;
            let mut survival = 1.0;
            for &idx in perm {
                total += survival * costs[idx];
                survival *= sels[idx];
            }
            total
        };

        let mut af =
            AdaptiveFilter::with_intervals(vec![0, 1], false, 2, 4, 2).with_seed(42);
        assert_eq!(af.permutation(), &[0, 1]);

        for _ in 0..200 {
            let cost = expected_cost(af.permutation());
            af.adapt_runtime_statistics(cost);
        }

        assert_eq!(af.permutation(), &[1, 0]);
        // Subsequent bad swaps get reverted and decay swap_likeliness toward 1.
        assert!(af.swap_likeliness()[0] < INITIAL_SWAP_LIKELINESS);
    }

    #[test]
    fn adaptive_bandit_reorders_three_conjuncts_to_optimal_order() {
        // Optimal order is [2, 1, 0]:
        // 0: cost 50, sel 0.9
        // 1: cost 10, sel 0.2
        // 2: cost 2,  sel 0.05
        let expected_cost = |perm: &[usize]| -> f64 {
            let costs = [50.0, 10.0, 2.0];
            let sels = [0.9, 0.2, 0.05];
            let mut total = 0.0;
            let mut survival = 1.0;
            for &idx in perm {
                total += survival * costs[idx];
                survival *= sels[idx];
            }
            total
        };

        let mut af =
            AdaptiveFilter::with_intervals(vec![0, 1, 2], false, 2, 4, 2).with_seed(7);
        for _ in 0..800 {
            let cost = expected_cost(af.permutation());
            af.adapt_runtime_statistics(cost);
        }
        assert_eq!(af.permutation(), &[2, 1, 0]);
    }

    #[test]
    fn compiled_filter_disables_permutation_on_division_and_preserves_short_circuit() {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, false)]));
        // `x <> 0 AND 100 / x > 3`
        let expr = logical2physical(
            &(col("x")
                .not_eq(lit(0i32))
                .and((lit(100i32) / col("x")).gt(lit(3i32)))),
            &schema,
        );
        let compiled = CompiledFilter::try_new(expr, &schema).unwrap();
        let mut state = compiled.new_worker_state_with_intervals(1, 2, 1);

        assert!(state.disable_permutations());
        assert_eq!(state.permutation(), &[0, 1]);

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(vec![0, 10, 50, 0, 25, 100]))],
        )
        .unwrap();
        for _ in 0..30 {
            let view = compiled
                .evaluate_view(BatchView::all(batch.clone()), &mut state)
                .expect("guard x <> 0 must prevent division by zero");
            let out = view.materialize().unwrap();
            let xs = out.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
            assert_eq!(xs.values(), &[10, 25]);
            assert_eq!(state.permutation(), &[0, 1]);
        }
    }

    #[test]
    fn compiled_filter_disables_permutation_on_cast() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("flag", DataType::Int32, false),
            Field::new("s", DataType::Utf8, false),
        ]));
        let guard: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("flag", 0)),
            ExprOperator::Eq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(1)))),
        ));
        let cast_pred: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(CastExpr::new(
                Arc::new(Column::new("s", 1)),
                DataType::Int32,
                None,
            )),
            ExprOperator::Gt,
            Arc::new(Literal::new(ScalarValue::Int32(Some(10)))),
        ));
        let pred = conjunction([guard, cast_pred]);
        let compiled = CompiledFilter::try_new(pred, &schema).unwrap();
        let mut state = compiled.new_worker_state_with_intervals(1, 2, 1);

        assert!(state.disable_permutations());
        assert_eq!(state.permutation(), &[0, 1]);

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![0, 1, 0, 1])),
                Arc::new(StringArray::from(vec!["not_a_number", "42", "bad", "5"])),
            ],
        )
        .unwrap();
        let out = compiled
            .evaluate_view(BatchView::all(batch), &mut state)
            .expect("guard flag = 1 must prevent CAST error on non-numeric rows")
            .materialize()
            .unwrap();
        assert_eq!(out.num_rows(), 1);
    }

    #[derive(Debug, PartialEq, Eq, Hash)]
    struct FallibleCustomExpr;

    impl Display for FallibleCustomExpr {
        fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
            write!(f, "fallible_custom")
        }
    }

    impl PhysicalExpr for FallibleCustomExpr {
        fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
            Ok(DataType::Boolean)
        }
        fn nullable(&self, _input_schema: &Schema) -> Result<bool> {
            Ok(false)
        }
        fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
            if batch.num_rows() > 0 {
                return internal_err!("should not be evaluated on rejected rows");
            }
            Ok(ColumnarValue::Array(Arc::new(BooleanArray::from(Vec::<
                bool,
            >::new(
            )))))
        }
        fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
            vec![]
        }
        fn with_new_children(
            self: Arc<Self>,
            _children: Vec<Arc<dyn PhysicalExpr>>,
        ) -> Result<Arc<dyn PhysicalExpr>> {
            Ok(self)
        }
        fn fmt_sql(&self, f: &mut Formatter<'_>) -> fmt::Result {
            write!(f, "fallible_custom")
        }
    }

    #[test]
    fn compiled_filter_disables_permutation_on_unknown_fallible_expr() {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, false)]));
        let guard: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("x", 0)),
            ExprOperator::Eq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(999)))),
        ));
        let pred =
            conjunction([guard, Arc::new(FallibleCustomExpr) as Arc<dyn PhysicalExpr>]);
        let compiled = CompiledFilter::try_new(pred, &schema).unwrap();
        let mut state = compiled.new_worker_state();

        assert!(state.disable_permutations());
        assert_eq!(state.permutation(), &[0, 1]);

        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(vec![1, 2, 3]))],
        )
        .unwrap();
        let view = compiled
            .evaluate_view(BatchView::all(batch), &mut state)
            .expect("guard x = 999 short-circuits before FallibleCustomExpr");
        assert!(view.is_empty());
    }

    #[test]
    fn compiled_filter_starts_cheap_first_and_reorders_real_like_predicates() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("s", DataType::Utf8, false),
            Field::new("k", DataType::Int32, false),
        ]));
        // Written expensive-first: `s LIKE '%abc%' AND k = 7`
        let like_expr: Arc<dyn PhysicalExpr> = Arc::new(LikeExpr::new(
            false,
            false,
            Arc::new(Column::new("s", 0)),
            Arc::new(Literal::new(ScalarValue::Utf8(Some("%abc%".to_string())))),
        ));
        let eq_expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("k", 1)),
            ExprOperator::Eq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(7)))),
        ));
        let pred = conjunction([like_expr, eq_expr]);
        let compiled = CompiledFilter::try_new(pred, &schema).unwrap();
        let state = compiled.new_worker_state();

        assert!(!state.disable_permutations());
        assert_eq!(compiled.num_conjuncts(), 2);
        // Heuristic places cheap `k = 7` (index 1) ahead of expensive `LIKE` (index 0).
        assert_eq!(state.permutation(), &[1, 0]);
    }

    #[test]
    fn compiled_filter_disables_permutation_on_volatile_udf() {
        use datafusion_expr::{ScalarUDF, ScalarUDFImpl, Signature, Volatility};
        use datafusion_physical_expr::ScalarFunctionExpr;

        #[derive(Debug, PartialEq, Eq, Hash)]
        struct VolatilePred {
            signature: Signature,
        }

        impl ScalarUDFImpl for VolatilePred {
            fn name(&self) -> &str {
                "volatile_pred"
            }
            fn signature(&self) -> &Signature {
                &self.signature
            }
            fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
                Ok(DataType::Boolean)
            }
            fn invoke_with_args(
                &self,
                args: datafusion_expr::ScalarFunctionArgs,
            ) -> Result<ColumnarValue> {
                Ok(ColumnarValue::Array(Arc::new(BooleanArray::from(vec![
                    true;
                    args.number_rows
                ]))))
            }
        }

        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, false)]));
        let udf = Arc::new(ScalarUDF::new_from_impl(VolatilePred {
            signature: Signature::exact(vec![DataType::Int32], Volatility::Volatile),
        }));
        let vol_expr: Arc<dyn PhysicalExpr> = Arc::new(ScalarFunctionExpr::new(
            "volatile_pred",
            udf,
            vec![Arc::new(Column::new("x", 0))],
            Arc::new(Field::new("r", DataType::Boolean, false)),
            Arc::new(datafusion_common::config::ConfigOptions::default()),
        ));
        let cheap_expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            Arc::new(Column::new("x", 0)),
            ExprOperator::Eq,
            Arc::new(Literal::new(ScalarValue::Int32(Some(1)))),
        ));
        let pred = conjunction([vol_expr, cheap_expr]);
        let compiled = CompiledFilter::try_new(pred, &schema).unwrap();
        let state = compiled.new_worker_state();

        assert!(state.disable_permutations());
        assert_eq!(state.permutation(), &[0, 1]);
    }

    #[test]
    fn compiled_filter_runtime_reorders_equal_heuristic_like_predicates() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Utf8, false),
            Field::new("b", DataType::Utf8, false),
        ]));
        // Both conjuncts have identical static heuristic cost (LikeExpr = 200),
        // so initial order is [0, 1]. Conjunct 0 (`a LIKE '%common%'`) passes
        // 100% of rows and scans long strings, while conjunct 1 (`b LIKE '%needle%'`)
        // rejects 99.9% of rows on short strings -> runtime bandit must swap to [1, 0].
        let like_a: Arc<dyn PhysicalExpr> = Arc::new(LikeExpr::new(
            false,
            false,
            Arc::new(Column::new("a", 0)),
            Arc::new(Literal::new(ScalarValue::Utf8(Some(
                "%zzz_rare_suffix_target%".to_string(),
            )))),
        ));
        let like_b: Arc<dyn PhysicalExpr> = Arc::new(LikeExpr::new(
            false,
            false,
            Arc::new(Column::new("b", 1)),
            Arc::new(Literal::new(ScalarValue::Utf8(Some(
                "%needle%".to_string(),
            )))),
        ));
        let pred = conjunction([like_a, like_b]);
        let compiled = CompiledFilter::try_new(pred, &schema).unwrap();
        let mut state = compiled
            .new_worker_state_with_intervals(2, 4, 3)
            .with_seed(11);
        assert!(!state.disable_permutations());
        assert_eq!(state.permutation(), &[0, 1]);

        let n = 4096usize;
        let long_pad = "x".repeat(256);
        let a_vals: Vec<String> = (0..n)
            .map(|_| format!("{long_pad}zzz_rare_suffix_target"))
            .collect();
        let b_vals: Vec<&str> = (0..n)
            .map(|i| if i == 0 { "needle" } else { "no" })
            .collect();
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(StringArray::from(a_vals)),
                Arc::new(StringArray::from(b_vals)),
            ],
        )
        .unwrap();

        for _ in 0..80 {
            let view = compiled
                .evaluate_view(BatchView::all(batch.clone()), &mut state)
                .unwrap();
            assert_eq!(view.num_rows(), 1);
        }
        assert_eq!(
            state.permutation(),
            &[1, 0],
            "expected runtime bandit to move selective LIKE ahead of unselective long-string LIKE"
        );
    }
}
