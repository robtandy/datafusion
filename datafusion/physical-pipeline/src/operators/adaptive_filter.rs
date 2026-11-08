// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`AdaptiveFilter`]: runtime reordering of the conjuncts of a filter
//! predicate, inspired by DuckDB's `AdaptiveFilter`
//! (`src/execution/adaptive_filter.cpp`).
//!
//! A [`PhysicalFilter`](super::filter::PhysicalFilter) splits its predicate
//! `a AND b AND c` into its conjuncts and evaluates them one after the other,
//! each one only on the rows that the conjuncts before it did not reject,
//! until no row is left. The order matters: a cheap conjunct that removes
//! most rows should come first. The best order is not known when the query is
//! planned, and can change while it runs, so every thread measures the cost
//! (elapsed nanoseconds per input row) and the selectivity (`selected_rows /
//! input_rows`) of every conjunct in its [`AdaptiveFilterState`], and
//! periodically orders the conjuncts by decreasing rank
//! `(1.0 - selectivity) / cost`, the order that minimizes the expected cost
//! of evaluating independent conjuncts.
//!
//! # Fallible conjuncts
//!
//! A conjunct that can fail, e.g. `10 / x > 1` (division by zero), must fail
//! exactly when it fails in DataFusion. DataFusion evaluates the conjuncts in
//! the order of the predicate, and the rows a fallible conjunct is evaluated
//! on depend on the conjuncts before it (see
//! [`PhysicalFilter`](super::filter::PhysicalFilter)). So a fallible conjunct
//! never moves, and no conjunct moves across it: only runs of consecutive
//! infallible conjuncts between fallible ones (the *segments* of the filter)
//! are reordered. Within a segment the order has no observable effect: the
//! conjuncts cannot fail, and the rows that the next fallible conjunct sees
//! are those that no conjunct of the segment rejects, whatever their order.
//! DuckDB is stricter and does not reorder at all if a conjunct can fail. A
//! conjunct is considered infallible only if it is built from expressions
//! that are known to never fail, see [`is_infallible`].
//!
//! Like DataFusion's optimizer, the conjuncts are not reordered at all if one
//! of them is volatile (e.g. `random() < 0.5`): reordering changes how often
//! it is evaluated.

use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;

use arrow::datatypes::Schema;
use datafusion_common::Result;
use datafusion_expr::Operator;
use datafusion_physical_expr::expressions::{
    BinaryExpr, CastExpr, Column, InListExpr, IsNotNullExpr, IsNullExpr, Literal, NotExpr,
};
use datafusion_physical_expr::{PhysicalExpr, split_conjunction};
use datafusion_physical_expr_common::physical_expr::is_volatile;

/// Number of chunks a thread evaluates in the order of the predicate, to
/// collect statistics, before it reorders the conjuncts for the first time.
pub const WARMUP_CHUNKS: usize = 4;

/// Number of chunks between two reorderings of the conjuncts, after the
/// warm-up.
pub const ADAPT_INTERVAL: usize = 16;

/// The weight of the statistics of the past intervals, relative to those of
/// the last one, when the conjuncts are reordered: the statistics decay
/// exponentially so that the order follows the data.
const DECAY: f64 = 0.5;

/// Lower bound of the cost of a conjunct, in nanoseconds per row, so that
/// ranks stay finite.
const MIN_COST_NANOS: f64 = 1e-3;

/// One conjunct of a filter predicate.
#[derive(Debug, Clone)]
pub struct FilterConjunct {
    expr: Arc<dyn PhysicalExpr>,
    fallible: bool,
}

impl FilterConjunct {
    /// The conjunct.
    pub fn expr(&self) -> &Arc<dyn PhysicalExpr> {
        &self.expr
    }

    /// Whether evaluating the conjunct can fail (see [`is_infallible`]).
    pub fn is_fallible(&self) -> bool {
        self.fallible
    }
}

/// The conjuncts of a filter predicate and the orders in which they may be
/// evaluated.
///
/// See the [module documentation](self).
#[derive(Debug, Clone)]
pub struct AdaptiveFilter {
    conjuncts: Vec<FilterConjunct>,
    /// The segments of conjuncts that are reordered.
    segments: Vec<Range<usize>>,
}

impl AdaptiveFilter {
    /// Split `predicate`, which refers to `input_schema`, into its top-level
    /// conjuncts.
    ///
    /// Nested `AND`s are flattened, in order: `(a AND b) AND c` has the
    /// conjuncts `a`, `b` and `c`, like in DataFusion's evaluation of `AND`.
    /// A predicate that is not an `AND` (e.g. `a OR b`) has a single
    /// conjunct.
    pub fn try_new(
        predicate: &Arc<dyn PhysicalExpr>,
        input_schema: &Schema,
    ) -> Result<Self> {
        let conjuncts = split_conjunction(predicate)
            .into_iter()
            .map(|expr| {
                Ok(FilterConjunct {
                    expr: Arc::clone(expr),
                    fallible: !is_infallible(expr, input_schema)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        let volatile = conjuncts.iter().any(|conjunct| is_volatile(&conjunct.expr));
        let segments = if volatile {
            vec![]
        } else {
            infallible_runs(&conjuncts)
        };
        Ok(Self {
            conjuncts,
            segments,
        })
    }

    /// The conjuncts, in the order of the predicate.
    pub fn conjuncts(&self) -> &[FilterConjunct] {
        &self.conjuncts
    }

    /// The segments of the filter: the ranges of conjuncts that are
    /// reordered, the maximal runs of at least two consecutive infallible
    /// conjuncts (none if a conjunct is volatile). Every other conjunct is
    /// always evaluated at its position in the predicate.
    pub fn segments(&self) -> &[Range<usize>] {
        &self.segments
    }

    /// Whether the conjuncts can be evaluated in more than one order, i.e.
    /// whether the filter has a segment. Statistics are only collected for
    /// adaptive filters.
    pub fn is_adaptive(&self) -> bool {
        !self.segments.is_empty()
    }

    /// Create the state of one thread: the conjuncts are evaluated in the
    /// order of the predicate until enough statistics are collected.
    pub fn create_state(&self) -> AdaptiveFilterState {
        AdaptiveFilterState {
            permutation: (0..self.conjuncts.len()).collect(),
            segments: self.segments.clone(),
            stats: vec![ConjunctStats::default(); self.conjuncts.len()],
            chunks: 0,
            next_adaptation: WARMUP_CHUNKS,
        }
    }
}

/// The maximal runs of at least two consecutive infallible conjuncts.
fn infallible_runs(conjuncts: &[FilterConjunct]) -> Vec<Range<usize>> {
    let mut runs = vec![];
    let mut start = 0;
    for end in 0..=conjuncts.len() {
        if end == conjuncts.len() || conjuncts[end].fallible {
            if end - start >= 2 {
                runs.push(start..end);
            }
            start = end + 1;
        }
    }
    runs
}

/// Statistics of one conjunct, summed over the chunks of a thread. They
/// decay exponentially: every reordering halves them, so that recent chunks
/// weigh more.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct ConjunctStats {
    /// The rows the conjunct was evaluated on.
    pub input_rows: f64,
    /// The rows the conjunct selected.
    pub selected_rows: f64,
    /// The time spent evaluating the conjunct, in nanoseconds.
    pub elapsed_nanos: f64,
}

impl ConjunctStats {
    /// The fraction of its input rows the conjunct selects, if it was
    /// evaluated at all.
    pub fn selectivity(&self) -> Option<f64> {
        (self.input_rows > 0.0)
            .then(|| (self.selected_rows / self.input_rows).clamp(0.0, 1.0))
    }

    /// The cost of the conjunct in nanoseconds per input row, if it was
    /// evaluated at all.
    pub fn cost(&self) -> Option<f64> {
        (self.input_rows > 0.0)
            .then(|| (self.elapsed_nanos / self.input_rows).max(MIN_COST_NANOS))
    }

    /// The rank of the conjunct, `(1.0 - selectivity) / cost`: the fraction
    /// of its input rows the conjunct removes, per nanosecond it spends on a
    /// row. Evaluating independent conjuncts by decreasing rank minimizes the
    /// expected cost: `a` before `b` costs `cost(a) + selectivity(a) *
    /// cost(b)` per row, which is less than the opposite order exactly if
    /// `rank(a) > rank(b)`.
    pub fn rank(&self) -> Option<f64> {
        Some((1.0 - self.selectivity()?) / self.cost()?)
    }

    fn decay(&mut self) {
        self.input_rows *= DECAY;
        self.selected_rows *= DECAY;
        self.elapsed_nanos *= DECAY;
    }
}

/// The evaluation order and the runtime statistics of the conjuncts of an
/// [`AdaptiveFilter`] in one thread.
#[derive(Debug, Clone)]
pub struct AdaptiveFilterState {
    /// The order in which the conjuncts are evaluated (indices into
    /// [`AdaptiveFilter::conjuncts`]). Only the conjuncts of a segment move,
    /// within their segment.
    permutation: Vec<usize>,
    /// See [`AdaptiveFilter::segments`].
    segments: Vec<Range<usize>>,
    stats: Vec<ConjunctStats>,
    /// Chunks evaluated by this thread.
    chunks: usize,
    /// The number of chunks after which the conjuncts are reordered next.
    next_adaptation: usize,
}

impl AdaptiveFilterState {
    /// The order in which the conjuncts are evaluated: indices into
    /// [`AdaptiveFilter::conjuncts`].
    pub fn permutation(&self) -> &[usize] {
        &self.permutation
    }

    /// Whether statistics should be collected (see
    /// [`AdaptiveFilter::is_adaptive`]): if not, [`Self::record`] and
    /// [`Self::end_chunk`] do not have to be called.
    pub fn is_adaptive(&self) -> bool {
        !self.segments.is_empty()
    }

    /// The statistics of every conjunct (in the order of the predicate).
    pub fn stats(&self) -> &[ConjunctStats] {
        &self.stats
    }

    /// Record that conjunct `conjunct` selected `selected_rows` of
    /// `input_rows` rows in `elapsed`.
    pub fn record(
        &mut self,
        conjunct: usize,
        input_rows: usize,
        selected_rows: usize,
        elapsed: Duration,
    ) {
        let stats = &mut self.stats[conjunct];
        stats.input_rows += input_rows as f64;
        stats.selected_rows += selected_rows as f64;
        stats.elapsed_nanos += elapsed.as_nanos() as f64;
    }

    /// Called once a chunk was filtered: after the warm-up, and then every
    /// [`ADAPT_INTERVAL`] chunks, the conjuncts are reordered by decreasing
    /// rank (see [`Self::adapt`]).
    pub fn end_chunk(&mut self) {
        if !self.is_adaptive() {
            return;
        }
        self.chunks += 1;
        if self.chunks >= self.next_adaptation {
            self.adapt();
            self.next_adaptation = self.chunks + ADAPT_INTERVAL;
        }
    }

    /// Order the conjuncts of every segment by decreasing rank, then let the
    /// statistics decay. Conjuncts that are not part of a segment keep their
    /// position.
    ///
    /// Conjuncts without statistics (that were never evaluated, because the
    /// conjuncts before them rejected all rows) come last in their segment,
    /// and ties keep the current order, so that the order does not
    /// oscillate.
    pub fn adapt(&mut self) {
        let ranks = self
            .stats
            .iter()
            .map(|stats| stats.rank().unwrap_or(f64::NEG_INFINITY))
            .collect::<Vec<_>>();
        for segment in &self.segments {
            // A stable sort: ties keep the current order.
            self.permutation[segment.clone()]
                .sort_by(|&a, &b| ranks[b].total_cmp(&ranks[a]));
        }
        self.stats.iter_mut().for_each(ConjunctStats::decay);
    }
}

/// Whether evaluating `expr` (on `schema`) can never fail, whatever the
/// values of its input rows.
///
/// This is conservative: only columns, literals, comparisons, `AND`, `OR`,
/// `NOT`, `IS [NOT] NULL`, `IN` lists and widening casts are infallible
/// (with infallible inputs). DataFusion considers all of them cheap and
/// infallible too (`is_cheap_and_infallible` in its `BinaryExpr`), so it
/// never removes rejected rows before evaluating them. Arithmetic may fail
/// on overflow (whether a `BinaryExpr` checks for overflow is not visible),
/// division on zero, and functions or `CASE` are not inspected, so they are
/// all considered fallible.
pub fn is_infallible(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> Result<bool> {
    let node_is_infallible = if let Some(binary) = expr.downcast_ref::<BinaryExpr>() {
        use Operator::*;
        matches!(
            binary.op(),
            Eq | NotEq
                | Lt
                | LtEq
                | Gt
                | GtEq
                | IsDistinctFrom
                | IsNotDistinctFrom
                | And
                | Or
        )
    } else if let Some(cast) = expr.downcast_ref::<CastExpr>() {
        cast.is_bigger_cast(&cast.expr().data_type(schema)?)
    } else {
        expr.is::<Column>()
            || expr.is::<Literal>()
            || expr.is::<NotExpr>()
            || expr.is::<IsNullExpr>()
            || expr.is::<IsNotNullExpr>()
            || expr.is::<InListExpr>()
    };
    if !node_is_infallible {
        return Ok(false);
    }
    for child in expr.children() {
        if !is_infallible(child, schema)? {
            return Ok(false);
        }
    }
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::datatypes::{DataType, Field, SchemaRef};

    use crate::operators::filter::test_support::physical_expr;

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Int64, true),
            Field::new("s", DataType::Utf8, true),
            Field::new("e", DataType::Boolean, true),
        ]))
    }

    fn adaptive_filter(predicate: &str) -> AdaptiveFilter {
        let schema = schema();
        AdaptiveFilter::try_new(&physical_expr(predicate, &schema), &schema).unwrap()
    }

    fn fallibility(predicate: &str) -> Vec<bool> {
        adaptive_filter(predicate)
            .conjuncts()
            .iter()
            .map(FilterConjunct::is_fallible)
            .collect()
    }

    /// A state for conjuncts with the given selectivities and costs
    /// (nanoseconds per row); the conjuncts listed in `fallible` are
    /// fallible.
    fn state_of(conjuncts: &[(f64, f64)], fallible: &[usize]) -> AdaptiveFilterState {
        let conjuncts_of_filter = (0..conjuncts.len())
            .map(|c| FilterConjunct {
                expr: physical_expr("true", &schema()),
                fallible: fallible.contains(&c),
            })
            .collect::<Vec<_>>();
        let filter = AdaptiveFilter {
            segments: infallible_runs(&conjuncts_of_filter),
            conjuncts: conjuncts_of_filter,
        };
        let mut state = filter.create_state();
        for (conjunct, &(selectivity, cost)) in conjuncts.iter().enumerate() {
            record(&mut state, conjunct, selectivity, cost);
        }
        state
    }

    fn record(
        state: &mut AdaptiveFilterState,
        conjunct: usize,
        selectivity: f64,
        cost: f64,
    ) {
        state.record(
            conjunct,
            1000,
            (selectivity * 1000.0) as usize,
            Duration::from_nanos((cost * 1000.0) as u64),
        );
    }

    #[test]
    fn splits_nested_conjunctions_in_order() {
        let filter =
            adaptive_filter("(b > 1 AND b < 2) AND (s = 'x' AND (e OR a IS NULL))");
        let conjuncts = filter
            .conjuncts()
            .iter()
            .map(|c| c.expr().to_string())
            .collect::<Vec<_>>();
        assert_eq!(
            conjuncts,
            vec!["b@1 > 1", "b@1 < 2", "s@2 = x", "e@3 OR a@0 IS NULL"]
        );
        // `OR` is a single conjunct.
        assert_eq!(adaptive_filter("a > 1 OR b < 2").conjuncts().len(), 1);
    }

    #[test]
    fn classifies_fallible_conjuncts() {
        for (predicate, fallible) in [
            ("a > 1", false),
            ("a IS NULL", false),
            ("a IS NOT NULL", false),
            ("NOT (a = 1)", false),
            ("a IN (1, 2, 3)", false),
            ("a > 1 OR b < 2 OR s IS NULL", false),
            ("e", false),
            ("true", false),
            ("a <> b", false),
            // A widening cast cannot fail.
            ("CAST(a AS BIGINT) = b", false),
            ("a + 1 > 2", true),
            ("10 / a > 1", true),
            ("b % 3 = 0", true),
            ("CAST(s AS INT) > 1", true),
            ("CAST(b AS INT) > 1", true),
            ("upper(s) = 'X'", true),
            ("s LIKE 'x%'", true),
            ("CASE WHEN a > 0 THEN b > 1 ELSE false END", true),
            ("a > 1 OR 10 / a > 1", true),
        ] {
            assert_eq!(fallibility(predicate), vec![fallible], "{predicate}");
        }
        assert_eq!(
            fallibility("a <> 0 AND 10 / a > 1 AND b < 2"),
            vec![false, true, false]
        );
    }

    #[test]
    fn only_runs_of_infallible_conjuncts_are_reordered() {
        // The segments, as `(start, end)`.
        for (predicate, segments) in [
            ("a > 1", vec![]),
            ("a > 1 AND b < 2", vec![(0, 2)]),
            ("a > 1 AND b < 2 AND s = 'x'", vec![(0, 3)]),
            // Nothing moves across a fallible conjunct, in either direction.
            ("10 / a > 1 AND b < 2", vec![]),
            ("a <> 0 AND 10 / a > 1", vec![]),
            ("a <> 0 AND 10 / a > 1 AND b < 2", vec![]),
            (
                "a > 1 AND b < 2 AND 10 / a > 1 AND e AND s = 'x'",
                vec![(0, 2), (3, 5)],
            ),
            ("a + 1 > 2 AND 10 / a > 1 AND b % 2 = 0", vec![]),
            // Volatile conjuncts keep the order.
            ("random() < 0.5 AND b < 2 AND a > 1", vec![]),
            ("b < 2 AND a > 1 AND random() < 0.5", vec![]),
        ] {
            let filter = adaptive_filter(predicate);
            let actual = filter
                .segments()
                .iter()
                .map(|segment| (segment.start, segment.end))
                .collect::<Vec<_>>();
            assert_eq!(actual, segments, "{predicate}");
            assert_eq!(filter.is_adaptive(), !segments.is_empty(), "{predicate}");
            assert_eq!(
                filter.create_state().is_adaptive(),
                !segments.is_empty(),
                "{predicate}"
            );
        }
    }

    #[test]
    fn orders_conjuncts_by_rank_after_the_warmup() {
        // Ranks: (1 - 0.9) / 1 = 0.1, (1 - 0.5) / 1 = 0.5, (1 - 0.1) / 3 = 0.3.
        let mut state = state_of(&[(0.9, 1.0), (0.5, 1.0), (0.1, 3.0)], &[]);
        let ranks = state
            .stats()
            .iter()
            .map(|stats| stats.rank().unwrap())
            .collect::<Vec<_>>();
        for (rank, expected) in ranks.iter().zip([0.1, 0.5, 0.3]) {
            assert!((rank - expected).abs() < 1e-9, "{ranks:?}");
        }
        for _ in 1..WARMUP_CHUNKS {
            state.end_chunk();
            assert_eq!(state.permutation(), &[0, 1, 2]);
        }
        state.end_chunk();
        assert_eq!(state.permutation(), &[1, 2, 0]);
        // The statistics decay.
        assert_eq!(state.stats()[0].input_rows, 500.0);
        assert!((state.stats()[0].rank().unwrap() - 0.1).abs() < 1e-9);

        // The order follows the data: `0` becomes the most selective.
        for _ in 0..3 {
            record(&mut state, 0, 0.0, 1.0);
        }
        for _ in 1..ADAPT_INTERVAL {
            state.end_chunk();
            assert_eq!(state.permutation(), &[1, 2, 0]);
        }
        state.end_chunk();
        assert_eq!(state.permutation(), &[0, 1, 2]);
    }

    #[test]
    fn fallible_conjuncts_never_move_and_nothing_moves_across_them() {
        // Conjunct `2` is fallible. The ranks increase with the index: without
        // the fallible conjunct, the order would be reversed.
        let conjuncts = [(0.9, 1.0), (0.8, 1.0), (0.7, 1.0), (0.6, 1.0), (0.5, 1.0)];
        let mut state = state_of(&conjuncts, &[2]);
        state.adapt();
        assert_eq!(state.permutation(), &[1, 0, 2, 4, 3]);

        // A fallible first conjunct stays first, even if the conjuncts after
        // it are much more selective, and so does a fallible last conjunct.
        let mut state = state_of(&[(0.9, 1.0), (0.1, 1.0), (0.0, 1.0)], &[0]);
        state.adapt();
        assert_eq!(state.permutation(), &[0, 2, 1]);
        let mut state = state_of(&[(0.9, 1.0), (0.1, 1.0), (0.0, 1.0)], &[2]);
        state.adapt();
        assert_eq!(state.permutation(), &[1, 0, 2]);
        // Single infallible conjuncts between fallible ones never move.
        let mut state = state_of(&[(0.9, 1.0), (0.1, 1.0), (0.0, 1.0)], &[1]);
        assert!(!state.is_adaptive());
        state.adapt();
        assert_eq!(state.permutation(), &[0, 1, 2]);
    }

    #[test]
    fn unobserved_conjuncts_come_last_and_ties_keep_the_order() {
        // `0` removes every row, so the others were never evaluated.
        let mut state = state_of(&[(0.0, 1.0), (0.5, 1.0), (0.5, 1.0)], &[]);
        state.stats[1] = ConjunctStats::default();
        state.stats[2] = ConjunctStats::default();
        state.permutation = vec![2, 1, 0];
        state.adapt();
        assert_eq!(state.permutation(), &[0, 2, 1]);
        assert_eq!(state.stats()[1].selectivity(), None);
        assert_eq!(state.stats()[1].cost(), None);

        // Equal ranks keep the current order.
        let mut state = state_of(&[(0.5, 1.0), (0.5, 1.0), (1.0, 1.0)], &[]);
        state.permutation = vec![1, 0, 2];
        state.adapt();
        assert_eq!(state.permutation(), &[1, 0, 2]);
    }

    #[test]
    fn non_adaptive_states_never_reorder() {
        let mut state = adaptive_filter("a <> 0 AND 10 / a > 1 AND b < 2").create_state();
        record(&mut state, 2, 0.0, 1.0);
        for _ in 0..100 {
            state.end_chunk();
        }
        assert_eq!(state.permutation(), &[0, 1, 2]);
    }

    #[test]
    fn costs_are_bounded_below() {
        let mut stats = ConjunctStats::default();
        assert_eq!(stats.rank(), None);
        stats.input_rows = 10.0;
        stats.selected_rows = 5.0;
        // No measurable time.
        assert_eq!(stats.cost(), Some(MIN_COST_NANOS));
        assert!(stats.rank().unwrap().is_finite());
    }
}
