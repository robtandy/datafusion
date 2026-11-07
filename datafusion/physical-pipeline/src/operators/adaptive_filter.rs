// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`AdaptiveFilter`]: runtime reordering of the conjuncts of a filter
//! predicate, inspired by DuckDB's `AdaptiveFilter`
//! (`src/execution/adaptive_filter.cpp`).
//!
//! A [`PhysicalFilter`](super::filter::PhysicalFilter) splits its predicate
//! `a AND b AND c` into its conjuncts and evaluates them one after the other,
//! each one only on the rows selected by the conjuncts before it, until no
//! row is left. The order matters: a cheap conjunct that removes most rows
//! should come first. The best order is not known when the query is planned,
//! and can change while it runs, so every thread measures the cost (elapsed
//! nanoseconds per input row) and the selectivity (`selected_rows /
//! input_rows`) of every conjunct in its [`AdaptiveFilterState`], and
//! periodically orders the conjuncts by decreasing rank
//! `(1.0 - selectivity) / cost`, the order that minimizes the expected cost
//! of evaluating independent conjuncts.
//!
//! # Fallible conjuncts
//!
//! A conjunct that can fail, e.g. `10 / x > 1` (division by zero), must not
//! be evaluated on more rows than in the order of the predicate: for
//! `x <> 0 AND 10 / x > 1`, the division must only see the rows for which
//! `x <> 0`. DuckDB disables reordering altogether if a conjunct can fail;
//! here every fallible conjunct is instead kept after all the conjuncts that
//! precede it in the predicate, so that it is only evaluated on rows that
//! all of them selected, while the conjuncts that cannot fail are reordered
//! freely (evaluating them on more or fewer rows has no observable effect).
//! A conjunct is considered infallible only if it is built from expressions
//! that are known to never fail, see [`is_infallible`].
//!
//! Like DataFusion's optimizer, the conjuncts are not reordered at all if one
//! of them is volatile (e.g. `random() < 0.5`): reordering changes how often
//! it is evaluated.

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
    /// Whether a conjunct is volatile.
    volatile: bool,
}

impl AdaptiveFilter {
    /// Split `predicate`, which refers to `input_schema`, into its top-level
    /// conjuncts.
    ///
    /// Nested `AND`s are flattened, in order: `(a AND b) AND c` has the
    /// conjuncts `a`, `b` and `c`. A predicate that is not an `AND` (e.g.
    /// `a OR b`) has a single conjunct.
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
        Ok(Self {
            conjuncts,
            volatile,
        })
    }

    /// The conjuncts, in the order of the predicate.
    pub fn conjuncts(&self) -> &[FilterConjunct] {
        &self.conjuncts
    }

    /// Whether the conjuncts can be evaluated in more than one order: no
    /// conjunct is volatile, there are at least two conjuncts and one of the
    /// conjuncts after the first is infallible (it may then move before the
    /// first one).
    ///
    /// Statistics are only collected for adaptive filters.
    pub fn is_adaptive(&self) -> bool {
        !self.volatile
            && self
                .conjuncts
                .iter()
                .skip(1)
                .any(|conjunct| !conjunct.fallible)
    }

    /// Create the state of one thread: the conjuncts are evaluated in the
    /// order of the predicate until enough statistics are collected.
    pub fn create_state(&self) -> AdaptiveFilterState {
        AdaptiveFilterState {
            permutation: (0..self.conjuncts.len()).collect(),
            fallible: self.conjuncts.iter().map(|c| c.fallible).collect(),
            adaptive: self.is_adaptive(),
            stats: vec![ConjunctStats::default(); self.conjuncts.len()],
            chunks: 0,
            next_adaptation: WARMUP_CHUNKS,
        }
    }
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
    /// [`AdaptiveFilter::conjuncts`]).
    permutation: Vec<usize>,
    /// Whether each conjunct is fallible.
    fallible: Vec<bool>,
    /// See [`AdaptiveFilter::is_adaptive`].
    adaptive: bool,
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
        self.adaptive
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
        if !self.adaptive {
            return;
        }
        self.chunks += 1;
        if self.chunks >= self.next_adaptation {
            self.adapt();
            self.next_adaptation = self.chunks + ADAPT_INTERVAL;
        }
    }

    /// Order the conjuncts by decreasing rank, keeping every fallible
    /// conjunct after all the conjuncts that precede it in the predicate,
    /// then let the statistics decay.
    ///
    /// The order is built greedily: at every position, the conjunct with the
    /// highest rank among those that may come next is chosen. Conjuncts
    /// without statistics (that were never evaluated, because the conjuncts
    /// before them removed all rows) come last, and ties keep the current
    /// order, so that the order does not oscillate.
    pub fn adapt(&mut self) {
        let n = self.permutation.len();
        let ranks = self
            .stats
            .iter()
            .map(|stats| stats.rank().unwrap_or(f64::NEG_INFINITY))
            .collect::<Vec<_>>();
        let mut placed = vec![false; n];
        // All conjuncts before this one (in the predicate) are placed.
        let mut first_unplaced = 0;
        let mut permutation = Vec::with_capacity(n);
        while permutation.len() < n {
            let mut best: Option<usize> = None;
            for &candidate in &self.permutation {
                let available = !placed[candidate]
                    && (!self.fallible[candidate] || candidate == first_unplaced);
                if available && best.is_none_or(|best| ranks[candidate] > ranks[best]) {
                    best = Some(candidate);
                }
            }
            // The first unplaced conjunct is always available.
            let best = best.unwrap_or(first_unplaced);
            placed[best] = true;
            permutation.push(best);
            while first_unplaced < n && placed[first_unplaced] {
                first_unplaced += 1;
            }
        }
        self.permutation = permutation;
        self.stats.iter_mut().for_each(ConjunctStats::decay);
    }
}

/// Whether evaluating `expr` (on `schema`) can never fail, whatever the
/// values of its input rows.
///
/// This is conservative: only columns, literals, comparisons, `AND`, `OR`,
/// `NOT`, `IS [NOT] NULL`, `IN` lists and widening casts are infallible
/// (with infallible inputs). DataFusion itself evaluates these on rows that
/// a previous conjunct already removed (see `BinaryExpr::evaluate_conjunction`).
/// Arithmetic may fail on overflow (whether a `BinaryExpr` checks for
/// overflow is not visible), division on zero, and functions or `CASE` are
/// not inspected, so they are all considered fallible.
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
