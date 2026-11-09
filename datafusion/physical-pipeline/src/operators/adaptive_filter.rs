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
//! that are known to never fail, see [`fallibility`].
//!
//! # Errors that depend on DataFusion's batches
//!
//! DataFusion decides per batch, from the share of rows that are left, on
//! which rows it evaluates the conjuncts after the first one of an `AND`, and
//! the right side of an `OR` (see `BinaryExpr::evaluate_conjunction` and
//! `check_short_circuit`), if on any: on an empty batch, it evaluates the
//! right side of an `OR`. Likewise, it evaluates a value after the first one
//! of an `IN` list on all rows of a batch unless the values before it match
//! all of them. If such an operand can fail on some rows and not on others,
//! or fails whenever it is evaluated (e.g. a pattern that does not compile),
//! whether a query fails depends on how the rows are split into batches,
//! which differs between DataFusion and a pipeline, and so does an
//! expression that fails on DataFusion's empty batches, which pipelines do
//! not keep: see [`depends_on_batches`].
//!
//! Like DataFusion's optimizer, the conjuncts are not reordered at all if one
//! of them is volatile (e.g. `random() < 0.5`): reordering changes how often
//! it is evaluated.

use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{RecordBatch, RecordBatchOptions, new_null_array};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit, UnionMode};
use datafusion_common::tree_node::TreeNode;
use datafusion_common::{Result, ScalarValue};
use datafusion_expr::Operator;
use datafusion_physical_expr::expressions::{
    BinaryExpr, CaseExpr, CastExpr, Column, InListExpr, IsNotNullExpr, IsNullExpr,
    LikeExpr, Literal, NotExpr,
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
    fallibility: Fallibility,
}

impl FilterConjunct {
    /// The conjunct.
    pub fn expr(&self) -> &Arc<dyn PhysicalExpr> {
        &self.expr
    }

    /// How evaluating the conjunct can fail (see [`fallibility`]).
    pub fn fallibility(&self) -> Fallibility {
        self.fallibility
    }

    /// Whether evaluating the conjunct can fail (see [`fallibility`]).
    pub fn is_fallible(&self) -> bool {
        self.fallibility != Fallibility::Infallible
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
            .map(|expr| FilterConjunct {
                expr: Arc::clone(expr),
                fallibility: fallibility(expr, input_schema),
            })
            .collect::<Vec<_>>();
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
        if end == conjuncts.len() || conjuncts[end].is_fallible() {
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

/// How evaluating an expression can fail.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Fallibility {
    /// It never fails.
    Infallible,
    /// It fails whenever it is evaluated, even without rows, e.g.
    /// `s ~ 'a{1000000}'`, whose constant pattern is too big to compile, or an
    /// operation on types that Arrow does not support (e.g. a comparison of
    /// dictionaries of dictionaries, or a cast of booleans into a dictionary).
    Uniform,
    /// It may fail on some rows and not on others, e.g. `10 / x`.
    PerRow,
}

/// How evaluating `expr` (on `schema`) can fail: the most fallible of its
/// nodes.
///
/// This is conservative. Only columns, literals, `AND`, `OR`, `NOT`,
/// `IS [NOT] NULL`, comparisons, `IN` lists, widening casts (also into and
/// out of dictionaries, see `cast_fallibility`), arithmetic on numbers that
/// wraps on overflow (DataFusion's default) and divisions of numbers by a
/// constant that is neither `0` nor `-1` are infallible, if Arrow supports the
/// types of their operands: they fail uniformly otherwise (e.g. `NOT` of an
/// integer, or a comparison of dictionaries of dictionaries, see
/// `fails_on_operand_types`). `LIKE` and regular expressions with a constant
/// pattern only fail if the pattern does not compile, which does not depend on
/// the rows (the pattern is compiled before any row is matched): they are
/// infallible if evaluating them on an empty batch succeeds, and fail
/// uniformly otherwise (see `constant_pattern_fallibility` for
/// case-insensitive `LIKE`s). Everything else, functions and `CASE` included,
/// may fail on some rows.
pub fn fallibility(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> Fallibility {
    expr.children()
        .into_iter()
        .map(|child| fallibility(child, schema))
        .fold(node_fallibility(expr, schema), Ord::max)
}

/// Whether evaluating `expr` (on `schema`) can never fail, whatever the
/// values of its input rows (see [`fallibility`]).
pub fn is_infallible(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> bool {
    fallibility(expr, schema) == Fallibility::Infallible
}

/// How `expr` itself (not its children) can fail.
fn node_fallibility(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> Fallibility {
    match value_fallibility(expr, schema) {
        // An expression that cannot fail on the values of its operands may
        // fail on their types, whatever the values (e.g. `NOT` of an integer,
        // a comparison of dictionaries of dictionaries, or a cast of booleans
        // into a dictionary), and then whenever it is evaluated.
        Fallibility::Infallible if fails_on_operand_types(expr, schema) => {
            Fallibility::Uniform
        }
        // Or on the number of rows it is evaluated on together.
        Fallibility::Infallible if expands_values(expr, schema) => Fallibility::PerRow,
        fallibility => fallibility,
    }
}

/// How `expr` itself can fail on the values of its operands, if Arrow
/// supports their types.
fn value_fallibility(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> Fallibility {
    use Fallibility::*;
    if let Some(binary) = expr.downcast_ref::<BinaryExpr>() {
        use Operator::*;
        return match binary.op() {
            Eq | NotEq | Lt | LtEq | Gt | GtEq | IsDistinctFrom | IsNotDistinctFrom
            | And | Or => Infallible,
            Plus | Minus | Multiply
                if wraps_on_overflow(binary) && is_number(binary, schema) =>
            {
                Infallible
            }
            Divide | Modulo if is_safe_division(binary, schema) => Infallible,
            LikeMatch | ILikeMatch | NotLikeMatch | NotILikeMatch | RegexMatch
            | RegexIMatch | RegexNotMatch | RegexNotIMatch
                if is_constant(binary.right()) =>
            {
                constant_pattern_fallibility(expr, schema)
            }
            _ => PerRow,
        };
    }
    if let Some(like) = expr.downcast_ref::<LikeExpr>() {
        return if is_constant(like.pattern()) {
            constant_pattern_fallibility(expr, schema)
        } else {
            PerRow
        };
    }
    if let Some(cast) = expr.downcast_ref::<CastExpr>() {
        return cast_fallibility(cast, schema);
    }
    if expr.is::<Column>()
        || expr.is::<Literal>()
        || expr.is::<NotExpr>()
        || expr.is::<IsNullExpr>()
        || expr.is::<IsNotNullExpr>()
        || expr.is::<InListExpr>()
    {
        Infallible
    } else {
        PerRow
    }
}

/// Whether `expr`, one of the expressions of [`value_fallibility`], fails on
/// the types of its operands: whether Arrow supports them does not depend on
/// the values, so evaluating `expr` on null values of these types tells. The
/// operands are not evaluated: they are replaced by columns of nulls.
fn fails_on_operand_types(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> bool {
    let operands = expr.children();
    if operands.is_empty() {
        return false;
    }
    let Ok(data_types) = operands
        .iter()
        .map(|operand| operand.data_type(schema))
        .collect::<Result<Vec<_>>>()
    else {
        return true;
    };
    // Without a null to probe with (e.g. an empty union has no value, so no
    // null), conservatively, fail.
    if !data_types
        .iter()
        .all(|data_type| can_build_nulls(data_type, 1))
    {
        return true;
    }
    let fields = data_types
        .iter()
        .enumerate()
        .map(|(index, data_type)| {
            Field::new(format!("operand_{index}"), data_type.clone(), true)
        })
        .collect::<Vec<_>>();
    let columns = fields
        .iter()
        .enumerate()
        .map(|(index, field)| Arc::new(Column::new(field.name(), index)) as _)
        .collect();
    let Ok(probe) = Arc::clone(expr).with_new_children(columns) else {
        return true;
    };
    let nulls = data_types
        .iter()
        .map(|data_type| new_null_array(data_type, 1))
        .collect();
    RecordBatch::try_new(Arc::new(Schema::new(fields)), nulls)
        .map_err(Into::into)
        .and_then(|batch| probe.evaluate(&batch)?.into_array(1))
        .is_err()
}

/// How `cast` can fail on the values it casts. It cannot if it widens values
/// (see `CastExpr::is_bigger_cast`), including the values of a dictionary
/// that it unpacks, and into the values of a dictionary that it packs a
/// constant into, or values that the keys of the dictionary can always index
/// (keys of 32 bits or more: a batch has fewer rows than that), or into the
/// values of a dictionary that it casts a dictionary (but not a dictionary of
/// dictionaries) into, whose keys hold all keys of the dictionary it casts
/// (Arrow casts the keys, which index all values of the dictionary, however
/// many: e.g. nulls take no memory), if Arrow supports it (see
/// [`fails_on_operand_types`]: e.g. Arrow cannot pack booleans into a
/// dictionary).
fn cast_fallibility(cast: &CastExpr, schema: &Schema) -> Fallibility {
    let Ok(from) = cast.expr().data_type(schema) else {
        return Fallibility::PerRow;
    };
    let from_values = match &from {
        DataType::Dictionary(_, values) => values.as_ref(),
        from => from,
    };
    let (to_values, keys_fit) = match cast.cast_type() {
        DataType::Dictionary(keys, values) => (
            values.as_ref(),
            match &from {
                DataType::Dictionary(from_keys, from_values) => {
                    !matches!(from_values.as_ref(), DataType::Dictionary(..))
                        && largest_key(from_keys)
                            .zip(largest_key(keys))
                            .is_some_and(|(from, to)| from <= to)
                }
                _ => {
                    cast.expr().is::<Literal>()
                        || matches!(
                            keys.as_ref(),
                            DataType::Int32
                                | DataType::Int64
                                | DataType::UInt32
                                | DataType::UInt64
                        )
                }
            },
        ),
        to => (to, true),
    };
    if keys_fit && CastExpr::check_bigger_cast(to_values, from_values) {
        Fallibility::Infallible
    } else {
        Fallibility::PerRow
    }
}

/// The largest key of dictionaries with keys of `keys`.
fn largest_key(keys: &DataType) -> Option<u64> {
    let largest = match keys {
        DataType::Int8 => i8::MAX as u64,
        DataType::Int16 => i16::MAX as u64,
        DataType::Int32 => i32::MAX as u64,
        DataType::Int64 => i64::MAX as u64,
        DataType::UInt8 => u8::MAX.into(),
        DataType::UInt16 => u16::MAX.into(),
        DataType::UInt32 => u32::MAX.into(),
        DataType::UInt64 => u64::MAX,
        _ => return None,
    };
    Some(largest)
}

/// How a `LIKE` or a regular expression match with a constant pattern can
/// fail: only if Arrow cannot compile its pattern, which it compiles before
/// it matches any row, so also on an empty batch. Except for a
/// case-insensitive `LIKE`: Arrow matches strings that are all ASCII (as on
/// an empty batch) with an ASCII pattern of a simple form (e.g. without
/// wildcards) without compiling the pattern, so it is also evaluated on a
/// constant string that is not ASCII. If only this fails, it fails on some
/// rows (strings that are not all ASCII) and not on others.
fn constant_pattern_fallibility(
    expr: &Arc<dyn PhysicalExpr>,
    schema: &Schema,
) -> Fallibility {
    if is_volatile(expr) {
        // Never evaluated before the execution: it may count its calls.
        return Fallibility::PerRow;
    }
    if fails_without_rows(expr, schema) {
        return Fallibility::Uniform;
    }
    match on_non_ascii_strings(expr, schema) {
        None => Fallibility::Infallible,
        Some(Ok(probe)) if !fails_on_one_row(&probe) => Fallibility::Infallible,
        Some(_) => Fallibility::PerRow,
    }
}

/// Whether `expr` is a case-insensitive `LIKE` with a constant pattern.
fn is_case_insensitive_constant_like(expr: &Arc<dyn PhysicalExpr>) -> bool {
    match expr.downcast_ref::<BinaryExpr>() {
        Some(binary) => {
            matches!(binary.op(), Operator::ILikeMatch | Operator::NotILikeMatch)
                && is_constant(binary.right())
        }
        None => expr
            .downcast_ref::<LikeExpr>()
            .is_some_and(|like| like.case_insensitive() && is_constant(like.pattern())),
    }
}

/// For a case-insensitive `LIKE`, the same `LIKE` of a constant string that
/// is not ASCII, of the type of the strings it matches.
fn on_non_ascii_strings(
    expr: &Arc<dyn PhysicalExpr>,
    schema: &Schema,
) -> Option<Result<Arc<dyn PhysicalExpr>>> {
    let non_ascii = |strings: &Arc<dyn PhysicalExpr>| -> Result<Arc<dyn PhysicalExpr>> {
        let value = ScalarValue::from("é").cast_to(&strings.data_type(schema)?)?;
        Ok(Arc::new(Literal::new(value)))
    };
    if let Some(binary) = expr.downcast_ref::<BinaryExpr>()
        && matches!(binary.op(), Operator::ILikeMatch | Operator::NotILikeMatch)
    {
        return Some(
            non_ascii(binary.left()).map(|left| -> Arc<dyn PhysicalExpr> {
                Arc::new(BinaryExpr::new(
                    left,
                    *binary.op(),
                    Arc::clone(binary.right()),
                ))
            }),
        );
    }
    let like = expr.downcast_ref::<LikeExpr>()?;
    like.case_insensitive().then(|| {
        non_ascii(like.expr()).map(|strings| -> Arc<dyn PhysicalExpr> {
            Arc::new(LikeExpr::new(
                like.negated(),
                true,
                strings,
                Arc::clone(like.pattern()),
            ))
        })
    })
}

/// Whether evaluating `expr`, which references no column, on one row fails.
fn fails_on_one_row(expr: &Arc<dyn PhysicalExpr>) -> bool {
    let options = RecordBatchOptions::new().with_row_count(Some(1));
    RecordBatch::try_new_with_options(Arc::new(Schema::empty()), vec![], &options)
        .map_err(Into::into)
        .and_then(|batch| expr.evaluate(&batch)?.into_array(1))
        .is_err()
}

/// Whether `expr` is a constant: a literal, possibly cast (DataFusion casts
/// a pattern into the type of the strings it matches, e.g. `Utf8View`).
fn is_constant(expr: &Arc<dyn PhysicalExpr>) -> bool {
    expr.is::<Literal>()
        || expr
            .downcast_ref::<CastExpr>()
            .is_some_and(|cast| is_constant(cast.expr()))
}

/// Whether evaluating `expr` on an empty batch of `schema` fails (or,
/// conservatively, whether there is no array of a type of `schema`).
fn fails_without_rows(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> bool {
    if !schema
        .fields()
        .iter()
        .all(|field| can_build_nulls(field.data_type(), 0))
    {
        return true;
    }
    let batch = RecordBatch::new_empty(Arc::new(schema.clone()));
    expr.evaluate(&batch)
        .and_then(|value| value.into_array(0))
        .is_err()
}

/// Whether `binary` wraps on overflow instead of failing (DataFusion's
/// default). The flag is not public, but takes part in the equality of
/// `BinaryExpr`s.
fn wraps_on_overflow(binary: &BinaryExpr) -> bool {
    let wrapping = BinaryExpr::new(
        Arc::clone(binary.left()),
        *binary.op(),
        Arc::clone(binary.right()),
    );
    *binary == wrapping
}

/// Whether both operands of `binary` are integers or floating point numbers.
fn is_number(binary: &BinaryExpr, schema: &Schema) -> bool {
    [binary.left(), binary.right()].into_iter().all(|operand| {
        operand
            .data_type(schema)
            .is_ok_and(|data_type| data_type.is_integer() || data_type.is_floating())
    })
}

/// Whether the division (or remainder) `binary` cannot fail: of floating
/// point numbers, which never fails, or of integers by a constant that is
/// neither `0` (division by zero) nor `-1` (overflow of the minimum value).
fn is_safe_division(binary: &BinaryExpr, schema: &Schema) -> bool {
    let (Ok(left), Ok(right)) = (
        binary.left().data_type(schema),
        binary.right().data_type(schema),
    ) else {
        return false;
    };
    if left.is_floating() && right.is_floating() {
        return true;
    }
    let Some(divisor) = binary.right().downcast_ref::<Literal>() else {
        return false;
    };
    let divisor = divisor.value();
    left.is_integer()
        && right.is_integer()
        && !ScalarValue::new_zero(&right).is_ok_and(|zero| *divisor == zero)
        && !ScalarValue::new_negative_one(&right)
            .is_ok_and(|minus_one| *divisor == minus_one)
}

/// Whether the errors of `expr` (on `schema`) may depend on how its input
/// rows are split into batches, empty batches included.
///
/// DataFusion's `FilterExec` and `ProjectionExec` evaluate their expressions
/// on every batch of their input, empty ones included. Native operators
/// evaluate them on chunks, which are not DataFusion's batches, and never on
/// empty chunks, so they only lower expressions for which this is `false`.
/// It is `true` if:
///
/// * `expr` is volatile: it may observe how often it is evaluated, e.g. a
///   function that counts its calls. It is never evaluated here either;
/// * a part of `expr` is dynamic (see `PhysicalExpr::snapshot`), e.g. a
///   `DynamicFilterPhysicalExpr`, which a `TopK` or a join updates during the
///   execution: its children are the columns it reads, not the expression it
///   evaluates, which is not known before the execution;
/// * several parts of `expr` may fail (see [`failing_parts`]): which error
///   DataFusion raises first depends on its batches, e.g. on a batch of
///   `("1", 0)` then `("invalid", 1)` rows, `CAST(s AS INT) + 10 / x` fails
///   to cast in DataFusion, but divides by zero on the first chunk of a
///   pipeline;
/// * the right side of an `AND` or an `OR`, or a value after the first one
///   of an `IN` list, may fail on some rows and not on others
///   ([`Fallibility::PerRow`]). DataFusion evaluates the right side of an
///   `AND` or an `OR` on all rows of a batch, on the rows that are left if at
///   most 20% of them are, or on no row at all if none is left, and a value of
///   an `IN` list on all rows of a batch unless the values before it match
///   all of them, so whether it fails depends on the batches: e.g.
///   `x <> 0 AND 10 / x > 1` fails on a batch with 25% of zeros, but not on a
///   batch with 85% of them;
/// * a part of `expr` fails whenever it is evaluated
///   ([`Fallibility::Uniform`]), which may depend on the batches: e.g.
///   DataFusion evaluates the right side of
///   `x = 0 OR (y = 0 OR s ~ 'a{1000000}')` unless `x = 0` on all rows of the
///   batch, and then `s ~ 'a{1000000}'` unless `y = 0` on all of them, so it
///   fails on rows with `x = 0` then rows with `y = 0` in one batch, but not
///   in two; and it evaluates the right side of an `OR` on empty batches;
/// * a case-insensitive `LIKE` fails only on strings that are not all ASCII
///   (see `constant_pattern_fallibility`): what Arrow considers ASCII depends
///   on how the strings are represented, e.g. on the bytes beneath nulls,
///   which DataFusion's `filter` keeps and the `take` of a native operator
///   drops;
/// * `expr` may fail and fails on an empty batch, e.g.
///   `regexp_like(s, 'a{1000000}')` (unlike `regexp_count`, which only
///   compiles its pattern if there are rows);
/// * a part of `expr` other than a column produces lists with 32-bit offsets
///   or run-end encoded arrays with run ends of less than 64 bits (at any
///   depth): building them overflows their offsets or run ends beyond as many
///   values in a batch as they can count, which nulls and runs reach without
///   memory, e.g. when `CASE WHEN x THEN a ELSE b END` merges lists of `a` and
///   of `b`, or when a constant list is broadcast to the rows of a batch, so
///   whether it fails depends on the number of rows of the batches;
/// * a part of `expr` other than a column copies values that its operands
///   share between rows (the values of dictionaries and of runs, and what
///   views point to), or a constant (that is not null, nor an empty string or
///   binary), into strings or binaries with 32-bit offsets (at any depth):
///   e.g. `CAST(d AS Utf8)` unpacks a dictionary of a single string of 2 MiB,
///   which 1024 rows, but not 1023, overflow, without taking more memory than
///   one row, and `CASE WHEN x THEN '…' END` copies a constant for every row
///   with `x`;
/// * `expr` is such a constant (e.g. a string that a projection outputs),
///   which whoever evaluates it copies for every row of the batch;
/// * a part of `expr` that may fail is evaluated on, or produces,
///   dictionaries, e.g. `CAST(d AS INT)`, which casts all values of the
///   dictionary of `d`, even those that no row references (when DataFusion
///   coalesces batches, e.g. the output of a `FilterExec`, it merges their
///   dictionaries without these values, while chunks keep the dictionaries
///   of their batches), or `CAST(s AS Dictionary(Int8, Utf8))`, which fails
///   on batches with more than 128 distinct values.
pub fn depends_on_batches(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> bool {
    if is_volatile(expr) || is_dynamic(expr) || failing_parts(expr, schema) > 1 {
        return true;
    }
    let (fallibility, depends_on_batches) = analyze(expr, schema);
    depends_on_batches
        || (fallibility != Fallibility::Infallible && fails_without_rows(expr, schema))
        // Whoever evaluates it copies it for every row of the batch.
        || is_broadcast_constant(expr, schema)
}

/// Whether a part of `expr` is dynamic: it has a snapshot (see
/// `PhysicalExpr::snapshot`), the expression it currently evaluates, which
/// may change during the execution, and which its children do not show (or
/// it fails to take one).
fn is_dynamic(expr: &Arc<dyn PhysicalExpr>) -> bool {
    expr.exists(|expr| Ok(!matches!(expr.snapshot(), Ok(None))))
        .unwrap_or(true)
}

/// The number of the parts of `expr` that may fail (on some rows or on all of
/// them): if there are several, which error DataFusion raises first depends on
/// its batches, as it evaluates each part on all rows of a batch before the
/// next one, while a pipeline evaluates each part on a chunk. A `CASE` does not
/// fail itself, but its branches may.
pub fn failing_parts(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> usize {
    let itself = usize::from(
        !expr.is::<CaseExpr>()
            && node_fallibility(expr, schema) != Fallibility::Infallible,
    );
    expr.children()
        .into_iter()
        .map(|child| failing_parts(child, schema))
        .fold(itself, usize::saturating_add)
}

/// The fallibility of `expr` (see [`fallibility`]), and whether its errors
/// depend on the batches it is evaluated on, but for failures on empty
/// batches (see [`depends_on_batches`]). Each node is classified once.
fn analyze(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> (Fallibility, bool) {
    let node = node_fallibility(expr, schema);
    let children = expr.children();
    let analyzed = children
        .iter()
        .map(|child| analyze(child, schema))
        .collect::<Vec<_>>();
    let fallibility = analyzed
        .iter()
        .map(|&(fallibility, _)| fallibility)
        .fold(node, Ord::max);
    let conditional = conditional_operands(expr);
    let depends_on_batches = analyzed.iter().any(|&(_, depends)| depends)
        // It fails whenever DataFusion evaluates it.
        || node == Fallibility::Uniform
        // A case-insensitive `LIKE` that only fails on strings that are not
        // all ASCII: what Arrow considers ASCII depends on the representation
        // of the strings (e.g. the bytes beneath nulls, which `filter` keeps
        // and `take` drops).
        || (node == Fallibility::PerRow && is_case_insensitive_constant_like(expr))
        // DataFusion evaluates an operand that may fail on some rows on rows
        // that depend on the batch.
        || children.iter().zip(&analyzed).any(|(child, &(fallibility, _))| {
            fallibility == Fallibility::PerRow
                && conditional.iter().any(|operand| Arc::ptr_eq(operand, child))
        })
        // It builds lists with 32-bit offsets or narrow run ends (a column
        // only passes its own on), which overflow beyond as many values in a
        // batch as they can count, so whether it fails depends on the number
        // of rows of the batches.
        || (!expr.is::<Column>()
            && expr
                .data_type(schema)
                .is_ok_and(|data_type| contains_narrow_lengths(&data_type)))
        // It copies values that its operands share between rows, or a
        // constant, into arrays whose 32-bit offsets overflow beyond 2 GiB of
        // bytes.
        || expands_values(expr, schema)
        // It may fail and evaluates or produces dictionaries.
        || (node != Fallibility::Infallible
            && children.iter().copied().chain([expr]).any(|expr| {
                expr.data_type(schema)
                    .is_ok_and(|data_type| contains_dictionaries(&data_type))
            }));
    (fallibility, depends_on_batches)
}

/// The operands of `expr` that DataFusion evaluates on rows that depend on
/// the batch: the right side of an `AND` or an `OR`, and the values after
/// the first one of an `IN` list.
fn conditional_operands(expr: &Arc<dyn PhysicalExpr>) -> &[Arc<dyn PhysicalExpr>] {
    if let Some(binary) = expr.downcast_ref::<BinaryExpr>() {
        match binary.op() {
            Operator::And | Operator::Or => std::slice::from_ref(binary.right()),
            _ => &[],
        }
    } else if let Some(in_list) = expr.downcast_ref::<InListExpr>() {
        in_list.list().get(1..).unwrap_or_default()
    } else {
        &[]
    }
}

/// Whether arrays of `data_type` contain, at any depth, lists with 32-bit
/// offsets (`List`, `ListView` or `Map`), or run-end encoded arrays with run
/// ends of less than 64 bits: building them (e.g. merging the branches of a
/// `CASE`, or broadcasting a constant to the rows of a batch) overflows the
/// offsets or the run ends beyond as many values as they can count, which
/// nulls and runs reach without memory.
fn contains_narrow_lengths(data_type: &DataType) -> bool {
    match data_type {
        DataType::List(_) | DataType::ListView(_) | DataType::Map(..) => true,
        DataType::RunEndEncoded(run_ends, values) => {
            run_ends.data_type() != &DataType::Int64
                || contains_narrow_lengths(values.data_type())
        }
        DataType::LargeList(field)
        | DataType::LargeListView(field)
        | DataType::FixedSizeList(field, _) => contains_narrow_lengths(field.data_type()),
        DataType::Struct(fields) => fields
            .iter()
            .any(|field| contains_narrow_lengths(field.data_type())),
        DataType::Union(fields, _) => fields
            .iter()
            .any(|(_, field)| contains_narrow_lengths(field.data_type())),
        DataType::Dictionary(_, values) => contains_narrow_lengths(values),
        _ => false,
    }
}

/// The most values, at any depth, of the columns of nulls that the probes of
/// the classification evaluate expressions on: a value of a fixed-size list
/// stands for its size of values.
const MAX_PROBE_VALUES: usize = 1 << 16;

/// Whether Arrow builds an array of `len` nulls of `data_type` (with
/// `new_null_array`, and empty ones with `new_empty_array`), with at most
/// [`MAX_PROBE_VALUES`] values at any depth. It cannot (it panics) for unions
/// without fields, which have no value, so no null, run ends that cannot
/// count the nulls, as one run, nor types of which there is no array
/// (fixed-size binaries or lists of negative sizes, dictionaries with keys
/// that are not integers, run ends that are not `Int16`, `Int32` or `Int64`,
/// times of microseconds or nanoseconds in 32 bits, or of seconds or
/// milliseconds in 64 bits, maps whose entries are not structs of a key and a
/// value).
fn can_build_nulls(data_type: &DataType, len: usize) -> bool {
    // The values of the fixed-size lists of `len` values of `size`.
    let values = |size: i32| {
        usize::try_from(size)
            .ok()
            .and_then(|size| size.checked_mul(len))
    };
    if len > MAX_PROBE_VALUES {
        return false;
    }
    match data_type {
        DataType::FixedSizeBinary(size) => {
            values(*size).is_some_and(|bytes| bytes <= MAX_PROBE_VALUES)
        }
        DataType::FixedSizeList(field, size) => {
            values(*size).is_some_and(|values| can_build_nulls(field.data_type(), values))
        }
        DataType::Struct(fields) => fields
            .iter()
            .all(|field| can_build_nulls(field.data_type(), len)),
        DataType::Time32(unit) => {
            matches!(unit, TimeUnit::Second | TimeUnit::Millisecond)
        }
        DataType::Time64(unit) => {
            matches!(unit, TimeUnit::Microsecond | TimeUnit::Nanosecond)
        }
        // Their values are empty.
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::ListView(field)
        | DataType::LargeListView(field) => can_build_nulls(field.data_type(), 0),
        DataType::Map(field, _) => {
            matches!(field.data_type(), DataType::Struct(fields) if fields.len() == 2)
                && can_build_nulls(field.data_type(), 0)
        }
        DataType::Dictionary(keys, values) => {
            keys.is_dictionary_key_type() && can_build_nulls(values, 0)
        }
        // Every child of a sparse union has the nulls, the first one of a
        // dense union.
        DataType::Union(fields, mode) => {
            (len == 0 || !fields.is_empty())
                && fields.iter().enumerate().all(|(index, (_, field))| {
                    let dense = *mode == UnionMode::Dense && index > 0;
                    can_build_nulls(field.data_type(), if dense { 0 } else { len })
                })
        }
        DataType::RunEndEncoded(run_ends, values) => {
            let counts = match run_ends.data_type() {
                DataType::Int16 => i16::try_from(len).is_ok(),
                DataType::Int32 => i32::try_from(len).is_ok(),
                DataType::Int64 => i64::try_from(len).is_ok(),
                _ => false,
            };
            counts && can_build_nulls(values.data_type(), len.min(1))
        }
        _ => true,
    }
}

/// Whether `expr` may copy values into arrays with 32-bit byte offsets
/// (`Utf8`, `Binary`) more times than its input has them: values that its
/// operands share between rows (the values of dictionaries and of runs, and
/// the bytes or values that views point to, any number of times), e.g. a cast
/// that unpacks a dictionary, or a constant operand, which it may copy for
/// every row (see [`is_broadcast_constant`]), e.g. `CASE WHEN x THEN '…' END`.
/// Their offsets overflow beyond 2 GiB of bytes in an array (e.g. a string of
/// 2 MiB, copied for 1024 rows), which the input reaches with more rows, but
/// not more memory: whether it fails depends on how many rows it is evaluated
/// on together.
fn expands_values(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> bool {
    if expr.is::<Column>() {
        return false;
    }
    let into_byte_offsets = expr
        .data_type(schema)
        .is_ok_and(|data_type| contains_byte_offsets(&data_type));
    // `IS [NOT] NULL` reads the validity of its operand, not its values.
    let reads_values = !expr.is::<IsNullExpr>() && !expr.is::<IsNotNullExpr>();
    expr.children().iter().any(|child| {
        if is_constant_tree(child) {
            into_byte_offsets && is_broadcast_constant(child, schema)
        } else {
            child.data_type(schema).is_ok_and(|data_type| {
                (into_byte_offsets && shares_values(&data_type))
                    || (reads_values && copies_shared_payloads(&data_type))
            })
        }
    })
}

/// Whether arrays of `data_type` contain, at any depth, values that rows
/// share, which Arrow copies for every row that has them in their own type
/// when it extracts or expands them (e.g. to cast them into any type), and
/// whose 32-bit offsets (see [`has_32_bit_offsets`]) may then overflow: the
/// children of dense unions, the values of runs, and those of list views.
fn copies_shared_payloads(data_type: &DataType) -> bool {
    let payload = |data_type: &DataType| {
        has_32_bit_offsets(data_type) || copies_shared_payloads(data_type)
    };
    match data_type {
        DataType::Union(fields, UnionMode::Dense) => {
            fields.iter().any(|(_, field)| payload(field.data_type()))
        }
        DataType::RunEndEncoded(_, field)
        | DataType::ListView(field)
        | DataType::LargeListView(field) => payload(field.data_type()),
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => copies_shared_payloads(field.data_type()),
        DataType::Struct(fields) => fields
            .iter()
            .any(|field| copies_shared_payloads(field.data_type())),
        DataType::Union(fields, UnionMode::Sparse) => fields
            .iter()
            .any(|(_, field)| copies_shared_payloads(field.data_type())),
        DataType::Dictionary(_, values) => copies_shared_payloads(values),
        _ => false,
    }
}

/// Whether arrays of `data_type` contain, at any depth, 32-bit offsets or run
/// ends that add up the lengths of their values, which overflow beyond as
/// many bytes or values as they count: of strings and binaries (see
/// [`contains_byte_offsets`]), of lists and maps, and of runs (see
/// [`contains_narrow_lengths`]).
pub(crate) fn has_32_bit_offsets(data_type: &DataType) -> bool {
    contains_byte_offsets(data_type) || contains_narrow_lengths(data_type)
}

/// Whether `expr` is a constant (see `is_constant_tree`) whose value may have
/// bytes with 32-bit offsets (see [`contains_byte_offsets`]): not a literal
/// null, nor an empty string or binary. Whoever evaluates it on a batch may
/// copy it for every row of the batch: e.g. `ProjectionExec` outputs it as a
/// column, and `CASE WHEN x THEN '…' END` copies it for every row with `x`.
fn is_broadcast_constant(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> bool {
    if let Some(literal) = expr.downcast_ref::<Literal>() {
        let value = literal.value();
        let no_bytes = match value {
            ScalarValue::Utf8(value) => value.as_deref().is_none_or(str::is_empty),
            ScalarValue::Binary(value) => value.as_deref().is_none_or(<[u8]>::is_empty),
            value => value.is_null(),
        };
        return !no_bytes && contains_byte_offsets(&value.data_type());
    }
    is_constant_tree(expr)
        && expr
            .data_type(schema)
            .is_ok_and(|data_type| contains_byte_offsets(&data_type))
}

/// Whether all the leaves of `expr` are literals: it reads nothing of the
/// batch it is evaluated on.
fn is_constant_tree(expr: &Arc<dyn PhysicalExpr>) -> bool {
    let children = expr.children();
    if children.is_empty() {
        expr.is::<Literal>()
    } else {
        children.into_iter().all(is_constant_tree)
    }
}

/// Whether arrays of `data_type` contain, at any depth, values that several
/// rows may share: dictionaries, run-end encoded arrays, views, and dense
/// unions (whose rows may point to the same value of a child).
fn shares_values(data_type: &DataType) -> bool {
    match data_type {
        DataType::Dictionary(..)
        | DataType::RunEndEncoded(..)
        | DataType::Utf8View
        | DataType::BinaryView
        | DataType::ListView(_)
        | DataType::LargeListView(_)
        | DataType::Union(_, UnionMode::Dense) => true,
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => shares_values(field.data_type()),
        DataType::Struct(fields) => {
            fields.iter().any(|field| shares_values(field.data_type()))
        }
        DataType::Union(fields, _) => fields
            .iter()
            .any(|(_, field)| shares_values(field.data_type())),
        _ => false,
    }
}

/// Whether arrays of `data_type` contain, at any depth, bytes with 32-bit
/// offsets (`Utf8`, `Binary`), but for the values of dictionaries, which
/// Arrow builds once for all rows (packing distinct values, or casting the
/// values of a dictionary).
fn contains_byte_offsets(data_type: &DataType) -> bool {
    match data_type {
        DataType::Utf8 | DataType::Binary => true,
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::ListView(field)
        | DataType::LargeListView(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => contains_byte_offsets(field.data_type()),
        DataType::Struct(fields) => fields
            .iter()
            .any(|field| contains_byte_offsets(field.data_type())),
        DataType::Union(fields, _) => fields
            .iter()
            .any(|(_, field)| contains_byte_offsets(field.data_type())),
        DataType::RunEndEncoded(_, values) => contains_byte_offsets(values.data_type()),
        _ => false,
    }
}

/// Whether arrays of `data_type` contain dictionaries, at any depth.
fn contains_dictionaries(data_type: &DataType) -> bool {
    match data_type {
        DataType::Dictionary(..) => true,
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::ListView(field)
        | DataType::LargeListView(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => contains_dictionaries(field.data_type()),
        DataType::Struct(fields) => fields
            .iter()
            .any(|field| contains_dictionaries(field.data_type())),
        DataType::Union(fields, _) => fields
            .iter()
            .any(|(_, field)| contains_dictionaries(field.data_type())),
        DataType::RunEndEncoded(_, values) => contains_dictionaries(values.data_type()),
        _ => false,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::datatypes::{DataType, Field, SchemaRef};

    use crate::operators::filter::test_support::{dynamic_filter, physical_expr};

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Int64, true),
            Field::new("s", DataType::Utf8, true),
            Field::new("e", DataType::Boolean, true),
            Field::new("c", DataType::Float64, true),
            Field::new(
                "d",
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                true,
            ),
            Field::new("u", DataType::Utf8View, true),
        ]))
    }

    fn adaptive_filter(predicate: &str) -> AdaptiveFilter {
        let schema = schema();
        AdaptiveFilter::try_new(&physical_expr(predicate, &schema), &schema).unwrap()
    }

    fn fallible_conjuncts(predicate: &str) -> Vec<bool> {
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
                fallibility: if fallible.contains(&c) {
                    Fallibility::PerRow
                } else {
                    Fallibility::Infallible
                },
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
        use Fallibility::*;
        let schema = schema();
        for (predicate, expected) in [
            ("a > 1", Infallible),
            ("a IS NULL", Infallible),
            ("a IS NOT NULL", Infallible),
            ("NOT (a = 1)", Infallible),
            ("a IN (1, 2, 3)", Infallible),
            ("a > 1 OR b < 2 OR s IS NULL", Infallible),
            ("e", Infallible),
            ("true", Infallible),
            ("a <> b", Infallible),
            // A widening cast cannot fail.
            ("CAST(a AS BIGINT) = b", Infallible),
            // Arithmetic wraps on overflow.
            ("a + 1 > 2", Infallible),
            ("b * b - a > 2", Infallible),
            ("c * 2.0 > 1.0", Infallible),
            // Divisions by constants other than 0 and -1, and of floats.
            ("b % 3 = 0", Infallible),
            ("b / 7 > 1", Infallible),
            ("c / c > 1.0", Infallible),
            ("10 / a > 1", PerRow),
            ("b / 0 > 1", PerRow),
            ("b % -1 = 0", PerRow),
            ("b / a > 1", PerRow),
            // A constant pattern only fails if it does not compile, and then
            // whenever it is evaluated, even without rows.
            ("s LIKE 'x%'", Infallible),
            ("s NOT ILIKE '%x_y%'", Infallible),
            ("s ~ '^a'", Infallible),
            ("s !~* 'a+b'", Infallible),
            ("s ~ 'a{1000000}'", Uniform),
            ("s ~ 'a+' AND s LIKE 'a%' OR s ~ 'a{1000000}'", Uniform),
            // The pattern is cast into the type of the strings.
            ("u LIKE '%x%'", Infallible),
            ("u ILIKE '%x_'", Infallible),
            ("u ~ 'a+'", Infallible),
            ("u ~ 'a{1000000}'", Uniform),
            ("u LIKE u", PerRow),
            ("s LIKE s", PerRow),
            ("CAST(s AS INT) > 1", PerRow),
            // Unpacking a dictionary cannot fail, nor packing a constant.
            ("CAST(d AS VARCHAR) = 'x'", Infallible),
            ("CAST(d AS VARCHAR(10)) = 'x'", Infallible),
            ("d = 'x'", Infallible),
            ("d IN ('x', 'y')", Infallible),
            ("CAST(d AS INT) > 1", PerRow),
            ("CAST(b AS INT) > 1", PerRow),
            ("upper(s) = 'X'", PerRow),
            ("CASE WHEN a > 0 THEN b > 1 ELSE false END", PerRow),
            ("a > 1 OR 10 / a > 1", PerRow),
        ] {
            let expr = physical_expr(predicate, &schema);
            assert_eq!(fallibility(&expr, &schema), expected, "{predicate}");
            assert_eq!(
                is_infallible(&expr, &schema),
                expected == Infallible,
                "{predicate}"
            );
        }
        assert_eq!(
            fallible_conjuncts(
                "a <> 0 AND 10 / a > 1 AND b < 2 AND s LIKE 'x' AND s ~ 'a{1000000}'"
            ),
            vec![false, true, false, false, true]
        );

        // Arithmetic that fails on overflow can fail.
        let checked: Arc<dyn PhysicalExpr> = Arc::new(
            BinaryExpr::new(
                physical_expr("b", &schema),
                Operator::Plus,
                physical_expr("b", &schema),
            )
            .with_fail_on_overflow(true),
        );
        assert_eq!(fallibility(&checked, &schema), PerRow);
        let wrapping: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            physical_expr("b", &schema),
            Operator::Plus,
            physical_expr("b", &schema),
        ));
        assert_eq!(fallibility(&wrapping, &schema), Infallible);

        // On strings that are all ASCII, as on an empty batch, Arrow matches
        // an ASCII `ILIKE` pattern of a simple form without compiling it: a
        // pattern of 100000 `s` (each `[sSſ]`) only fails on other strings.
        let unicode = "s".repeat(100_000);
        for (predicate, expected) in [
            (format!("s ILIKE '{unicode}'"), PerRow),
            (format!("u ILIKE '%{unicode}'"), PerRow),
            (format!("s ILIKE '%{unicode}%'"), Uniform),
            (format!("s LIKE '{unicode}'"), Infallible),
        ] {
            let expr = physical_expr(&predicate, &schema);
            assert_eq!(fallibility(&expr, &schema), expected, "{expected:?}");
        }
        assert!(depends_on_batches(
            &physical_expr(&format!("a > 0 AND s ILIKE '{unicode}'"), &schema),
            &schema
        ));
        // Whether Arrow compiles it depends on the representation of the
        // strings, e.g. on bytes beneath nulls, which `filter` keeps and
        // `take` drops: it depends on the batches anywhere.
        assert!(depends_on_batches(
            &physical_expr(&format!("s ILIKE '{unicode}' AND a > 0"), &schema),
            &schema
        ));

        // Arrow cannot compare some types, whatever the values.
        let nested = Arc::new(Schema::new(vec![
            Field::new(
                "n",
                DataType::Dictionary(
                    Box::new(DataType::Int32),
                    Box::new(DataType::Dictionary(
                        Box::new(DataType::Int32),
                        Box::new(DataType::Utf8),
                    )),
                ),
                true,
            ),
            Field::new(
                "l",
                DataType::List(Arc::new(Field::new_list_field(DataType::Int32, true))),
                true,
            ),
        ]));
        let compare = |name: &str, op: Operator| -> Arc<dyn PhysicalExpr> {
            Arc::new(BinaryExpr::new(
                physical_expr(name, &nested),
                op,
                physical_expr(name, &nested),
            ))
        };
        for (expr, expected) in [
            (compare("n", Operator::Eq), Uniform),
            (compare("n", Operator::Lt), Uniform),
            (compare("n", Operator::IsDistinctFrom), Uniform),
            (compare("l", Operator::Eq), Infallible),
            (compare("l", Operator::Lt), Infallible),
            (
                datafusion_physical_expr::expressions::in_list(
                    physical_expr("n", &nested),
                    vec![physical_expr("n", &nested)],
                    &false,
                    &nested,
                )
                .unwrap(),
                Uniform,
            ),
        ] {
            assert_eq!(fallibility(&expr, &nested), expected, "{expr}");
        }

        // Packing values into a dictionary fails on batches with more distinct
        // values than its keys can index.
        let pack = |keys: DataType| -> Arc<dyn PhysicalExpr> {
            let dictionary =
                DataType::Dictionary(Box::new(keys), Box::new(DataType::Utf8));
            Arc::new(CastExpr::new(physical_expr("s", &schema), dictionary, None))
        };
        assert_eq!(fallibility(&pack(DataType::Int8), &schema), PerRow);
        assert!(depends_on_batches(&pack(DataType::Int8), &schema));
        assert_eq!(fallibility(&pack(DataType::Int32), &schema), Infallible);
        assert!(!depends_on_batches(&pack(DataType::Int32), &schema));
        // Casting a dictionary into another casts its keys, which may index
        // more values than the batch has rows (e.g. nulls, which take no
        // memory).
        let dictionary = |keys: DataType, values: DataType| {
            DataType::Dictionary(Box::new(keys), Box::new(values))
        };
        for (from_keys, to_keys, expected) in [
            (DataType::UInt64, DataType::Int32, PerRow),
            (DataType::Int64, DataType::Int32, PerRow),
            (DataType::UInt32, DataType::Int32, PerRow),
            (DataType::Int64, DataType::UInt32, PerRow),
            (DataType::UInt8, DataType::Int8, PerRow),
            (DataType::Int32, DataType::Int32, Infallible),
            (DataType::Int32, DataType::UInt32, Infallible),
            (DataType::Int8, DataType::UInt8, Infallible),
            (DataType::UInt32, DataType::Int64, Infallible),
            (DataType::Int64, DataType::UInt64, Infallible),
        ] {
            let schema = Schema::new(vec![Field::new(
                "d",
                dictionary(from_keys.clone(), DataType::Null),
                true,
            )]);
            let cast: Arc<dyn PhysicalExpr> = Arc::new(CastExpr::new(
                physical_expr("d", &Arc::new(schema.clone())),
                dictionary(to_keys.clone(), DataType::Null),
                None,
            ));
            assert_eq!(
                fallibility(&cast, &schema),
                expected,
                "{from_keys} into {to_keys}"
            );
        }
        // Nor does it know the keys of a dictionary of dictionaries.
        let nested = Schema::new(vec![Field::new(
            "d",
            dictionary(DataType::Int8, dictionary(DataType::UInt64, DataType::Utf8)),
            true,
        )]);
        let cast: Arc<dyn PhysicalExpr> = Arc::new(CastExpr::new(
            physical_expr("d", &Arc::new(nested.clone())),
            dictionary(
                DataType::Int64,
                dictionary(DataType::UInt64, DataType::Utf8),
            ),
            None,
        ));
        assert_eq!(fallibility(&cast, &nested), PerRow);
        // Arrow cannot pack booleans into a dictionary, whatever the values.
        let booleans: Arc<dyn PhysicalExpr> = Arc::new(CastExpr::new(
            physical_expr("e", &schema),
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Boolean)),
            None,
        ));
        assert_eq!(fallibility(&booleans, &schema), Uniform);
        assert!(depends_on_batches(&booleans, &schema));
        // Nor apply `NOT`, `AND` or `OR` to integers.
        for expr in [
            Arc::new(NotExpr::new(physical_expr("a", &schema))) as Arc<dyn PhysicalExpr>,
            Arc::new(BinaryExpr::new(
                physical_expr("a", &schema),
                Operator::And,
                physical_expr("a", &schema),
            )),
            Arc::new(BinaryExpr::new(
                physical_expr("e", &schema),
                Operator::Or,
                physical_expr("a", &schema),
            )),
        ] {
            assert_eq!(fallibility(&expr, &schema), Uniform, "{expr}");
            assert!(depends_on_batches(&expr, &schema), "{expr}");
        }
        let not: Arc<dyn PhysicalExpr> =
            Arc::new(NotExpr::new(physical_expr("e", &schema)));
        assert_eq!(fallibility(&not, &schema), Infallible);
        // Nor add integers of different types.
        let mixed: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            physical_expr("a", &schema),
            Operator::Plus,
            physical_expr("b", &schema),
        ));
        assert_eq!(fallibility(&mixed, &schema), Uniform);
    }

    #[test]
    fn finds_errors_that_depend_on_batches() {
        let schema = schema();
        for (predicate, expected) in [
            // DataFusion decides per batch on which rows it evaluates the
            // conjuncts after the first one, and the right side of `OR`.
            ("a <> 0 AND 10 / a > 1", true),
            ("a = 0 OR 10 / a > 1", true),
            ("a > 0 AND upper(s) = 'X'", true),
            ("(a > 0 AND 10 / a > 1) OR e", true),
            ("e OR (a > 0 AND 10 / a > 1)", true),
            ("CASE WHEN a > 0 AND 10 / a > 1 THEN 1 ELSE 0 END = 1", true),
            ("a > 0 AND s LIKE s", true),
            // DataFusion evaluates a value after the first one of an `IN`
            // list on all rows of a batch, unless the values before it match
            // all of them.
            ("b IN (1, 10 / a)", true),
            ("b NOT IN (1, 2, b / a)", true),
            ("b IN (10 / a, 1)", false),
            ("b IN (1, 2, a, b + 1)", false),
            // A pattern that does not compile fails whenever DataFusion
            // evaluates it, which depends on the batches: on the batches on
            // which the left sides are not all true, and on empty batches.
            ("s ~ 'a{1000000}'", true),
            ("a = 0 OR s ~ 'a{1000000}'", true),
            ("a = 0 OR (b = 0 OR s ~ 'a{1000000}')", true),
            ("a <> 0 AND (b <> 0 OR s ~ 'a{1000000}')", true),
            ("CASE WHEN a > 0 THEN s ~ 'a{1000000}' ELSE false END", true),
            // DataFusion merges the dictionaries of the batches it coalesces,
            // without the values that no row references, and a cast of a
            // dictionary casts all of its values.
            ("CAST(d AS INT) > 1", true),
            ("upper(d) = 'X'", true),
            ("CAST(d AS VARCHAR) = 'x' AND d <> 'y'", false),
            // Functions may fail on empty batches.
            ("regexp_like(s, 'a{1000000}')", true),
            ("a = 0 OR regexp_count(s, 'a{1000000}') > 0", true),
            // The first conjunct and the left side of `OR` are evaluated on all
            // rows.
            ("10 / a > 1 AND b = 1", false),
            ("10 / a > 1 OR b = 1", false),
            ("upper(s) = 'X' AND a > 0", false),
            ("regexp_count(s, 'a+') > 0 AND a > 0", false),
            // `regexp_count` only compiles its pattern if there are rows: it
            // fails on the first row, whatever the batches.
            ("regexp_count(s, 'a{1000000}') > 0 AND a > 0", false),
            // Patterns that compile cannot fail.
            ("a > 0 AND s LIKE 'x%'", false),
            ("a = 1 OR s LIKE 'x%'", false),
            ("a = 0 OR (b = 0 OR s ~ 'a+')", false),
            ("a > 0 AND u LIKE '%x%'", false),
            ("a > 0 AND u ~ 'a{1000000}'", true),
            ("a > 0 AND b % 3 = 0 AND b + 1 > 0", false),
            // `CASE` evaluates its branches on the rows they are for.
            ("CASE WHEN a <> 0 THEN 10 / a > 1 ELSE false END", false),
            ("a > 0 AND b > 0", false),
        ] {
            let expr = physical_expr(predicate, &schema);
            assert_eq!(depends_on_batches(&expr, &schema), expected, "{predicate}");
        }
    }

    #[test]
    fn finds_several_failures() {
        // Which of several parts that may fail fails first depends on the
        // batches: DataFusion evaluates each part on all rows of a batch.
        let schema = schema();
        for (predicate, expected) in [
            ("CAST(s AS INT) + 10 / a > 0", true),
            ("CAST(10 / a AS SMALLINT) > 0", true),
            ("upper(s) = lower(s)", true),
            ("10 / a > 1", false),
            ("upper(s) = 'X' AND a > 0", false),
            // `CASE` itself does not fail.
            ("CASE WHEN a <> 0 THEN 10 / a > 1 ELSE false END", false),
        ] {
            let expr = physical_expr(predicate, &schema);
            assert_eq!(depends_on_batches(&expr, &schema), expected, "{predicate}");
        }
    }

    #[test]
    fn finds_lists_built_per_batch() {
        // Lists with 32-bit offsets that an expression builds overflow beyond
        // `i32::MAX` values in a batch: a column only passes its lists on.
        let list = |data_type: DataType| {
            DataType::List(Arc::new(Field::new_list_field(data_type, true)))
        };
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Boolean, true),
            Field::new("a", list(DataType::Null), true),
            Field::new("b", list(DataType::Null), true),
            Field::new(
                "l",
                DataType::LargeList(Arc::new(Field::new_list_field(
                    DataType::Null,
                    true,
                ))),
                true,
            ),
        ]));
        for (expr, expected) in [
            ("a", false),
            ("a IS NOT NULL", false),
            ("CASE WHEN x THEN a ELSE b END IS NOT NULL", true),
            ("CASE WHEN x THEN l ELSE l END IS NOT NULL", false),
        ] {
            let expr = physical_expr(expr, &schema);
            assert_eq!(depends_on_batches(&expr, &schema), expected, "{expr}");
        }
        let constant = crate::operators::filter::test_support::constant_list_of_nulls();
        assert!(depends_on_batches(&constant, &schema));
        // So do run-end encoded arrays with run ends of less than 64 bits.
        let runs = crate::operators::filter::test_support::constant_list_of_short_runs();
        assert!(depends_on_batches(&runs, &schema));
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
    fn dynamic_expressions_depend_on_batches() {
        // Whatever they evaluate, also in other expressions.
        let schema = schema();
        let dynamic = dynamic_filter("a > 0", &schema);
        assert!(!depends_on_batches(
            &physical_expr("a > 0", &schema),
            &schema
        ));
        assert!(depends_on_batches(&dynamic, &schema));
        let nested: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            physical_expr("b > 0", &schema),
            Operator::Or,
            dynamic,
        ));
        assert!(depends_on_batches(&nested, &schema));
    }

    #[test]
    fn finds_expansions_of_values() {
        use Fallibility::*;
        // A union of `Int32` and of `data_type`.
        let union = |data_type: DataType, mode: UnionMode| {
            DataType::Union(
                arrow::datatypes::UnionFields::try_new(
                    [0, 1],
                    [
                        Field::new("i", DataType::Int32, true),
                        Field::new("v", data_type, true),
                    ],
                )
                .unwrap(),
                mode,
            )
        };
        let strings = Arc::new(Field::new("values", DataType::Utf8, true));
        let dictionary =
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
        let schema = Arc::new(Schema::new(vec![
            Field::new("d", dictionary.clone(), true),
            Field::new("v", DataType::Utf8View, true),
            Field::new(
                "r",
                DataType::RunEndEncoded(
                    Arc::new(Field::new("run_ends", DataType::Int32, false)),
                    Arc::clone(&strings),
                ),
                true,
            ),
            Field::new(
                "t",
                DataType::Struct(vec![Field::new("d", dictionary, true)].into()),
                true,
            ),
            Field::new("s", DataType::Utf8, true),
            Field::new("e", DataType::Boolean, true),
            Field::new("du", union(DataType::Utf8, UnionMode::Dense), false),
            Field::new("su", union(DataType::Utf8, UnionMode::Sparse), false),
            Field::new("di", union(DataType::Float64, UnionMode::Dense), false),
        ]));
        let cast = |name: &str, to: DataType| -> Arc<dyn PhysicalExpr> {
            Arc::new(CastExpr::new(physical_expr(name, &schema), to, None))
        };
        for (expr, expected) in [
            // Unpacking dictionaries, runs or views into 32-bit offsets, also
            // nested ones.
            (cast("d", DataType::Utf8), (PerRow, true)),
            (cast("d", DataType::Binary), (PerRow, true)),
            (cast("r", DataType::Utf8), (PerRow, true)),
            (cast("v", DataType::Utf8), (PerRow, true)),
            (
                cast(
                    "t",
                    DataType::Struct(vec![Field::new("d", DataType::Utf8, true)].into()),
                ),
                (PerRow, true),
            ),
            // Extracting the strings of a dense union, or expanding runs,
            // copies them in their own type first, whatever the type it casts
            // them into.
            (cast("du", DataType::Utf8), (PerRow, true)),
            (cast("du", DataType::LargeUtf8), (PerRow, true)),
            (cast("du", DataType::Utf8View), (PerRow, true)),
            (cast("r", DataType::LargeUtf8), (PerRow, true)),
            // But not for `IS NULL`, nor for sparse unions, whose rows do not
            // share values, nor for unions of values without offsets.
            (
                Arc::new(IsNullExpr::new(physical_expr("du", &schema))),
                (Infallible, false),
            ),
            (cast("su", DataType::Utf8), (PerRow, false)),
            (cast("di", DataType::Float64), (PerRow, false)),
            // Nor into 64-bit offsets or views, nor into the values of a
            // dictionary, which Arrow casts once.
            (cast("d", DataType::LargeUtf8), (Infallible, false)),
            (cast("d", DataType::Utf8View), (Infallible, false)),
            (
                cast(
                    "d",
                    DataType::Dictionary(
                        Box::new(DataType::Int64),
                        Box::new(DataType::Utf8),
                    ),
                ),
                (Infallible, false),
            ),
            // Nor what does not share values.
            (cast("s", DataType::Utf8View), (Infallible, false)),
            (physical_expr("d", &schema), (Infallible, false)),
            (physical_expr("d IS NULL", &schema), (Infallible, false)),
        ] {
            assert_eq!(
                (
                    fallibility(&expr, &schema),
                    depends_on_batches(&expr, &schema)
                ),
                expected,
                "{expr}"
            );
        }

        // Constants, which are copied for every row.
        let constant = |value: ScalarValue| -> Arc<dyn PhysicalExpr> {
            Arc::new(Literal::new(value))
        };
        let case = |value: ScalarValue| -> Arc<dyn PhysicalExpr> {
            Arc::new(
                CaseExpr::try_new(
                    None,
                    vec![(physical_expr("e", &schema), constant(value))],
                    None,
                )
                .unwrap(),
            )
        };
        let string = |value: &str| ScalarValue::Utf8(Some(value.to_string()));
        for (expr, depends) in [
            (case(string("abc")), true),
            (case(ScalarValue::Binary(Some(vec![1]))), true),
            (constant(string("abc")), true),
            (
                Arc::new(IsNullExpr::new(case(string("abc")))) as Arc<dyn PhysicalExpr>,
                true,
            ),
            // Unless they are null or empty, with 64-bit offsets or views, or
            // compared.
            (case(string("")), false),
            (case(ScalarValue::Utf8(None)), false),
            (constant(string("")), false),
            (case(ScalarValue::LargeUtf8(Some("abc".to_string()))), false),
            (case(ScalarValue::Utf8View(Some("abc".to_string()))), false),
            (case(ScalarValue::Int64(Some(1))), false),
            (physical_expr("s = 'abc'", &schema), false),
        ] {
            assert_eq!(depends_on_batches(&expr, &schema), depends, "{expr}");
        }
    }

    #[test]
    fn finds_types_without_nulls() {
        let field = |data_type: DataType| Arc::new(Field::new("f", data_type, true));
        let list = |data_type: DataType, size: i32| {
            DataType::FixedSizeList(field(data_type), size)
        };
        let runs = |run_ends: DataType| {
            DataType::RunEndEncoded(
                Arc::new(Field::new("run_ends", run_ends, false)),
                field(DataType::Int32),
            )
        };
        let union = |types: Vec<DataType>, mode: UnionMode| {
            DataType::Union(
                arrow::datatypes::UnionFields::try_new(
                    0..types.len() as i8,
                    types
                        .into_iter()
                        .map(|data_type| Field::new("f", data_type, true)),
                )
                .unwrap(),
                mode,
            )
        };
        let long_runs = list(runs(DataType::Int16), 32768);
        for (data_type, len, built) in [
            (DataType::Int32, 1, true),
            (DataType::Int32, MAX_PROBE_VALUES + 1, false),
            // An empty union has no null, but empty arrays.
            (union(vec![], UnionMode::Sparse), 1, false),
            (union(vec![], UnionMode::Sparse), 0, true),
            // One run of the nulls.
            (runs(DataType::Int16), 32767, true),
            (long_runs.clone(), 1, false),
            (long_runs.clone(), 0, true),
            (list(runs(DataType::Int32), 32768), 1, true),
            (runs(DataType::Utf8), 0, false),
            (
                DataType::Struct(vec![Field::new("l", long_runs.clone(), true)].into()),
                1,
                false,
            ),
            // The values of lists and dictionaries are empty, and so are the
            // other children of a dense union than its first one.
            (DataType::new_list(long_runs.clone(), true), 1, true),
            (union(vec![long_runs.clone()], UnionMode::Dense), 1, false),
            (
                union(vec![DataType::Int32, long_runs.clone()], UnionMode::Dense),
                1,
                true,
            ),
            (
                union(vec![DataType::Int32, long_runs.clone()], UnionMode::Sparse),
                1,
                false,
            ),
            // Values of fixed-size lists stand for their size of values.
            (list(DataType::Int64, 1 << 16), 1, true),
            (list(DataType::Int64, 1 << 16), 2, false),
            (list(list(DataType::Int64, 1 << 10), 1 << 10), 1, false),
            (DataType::FixedSizeBinary(16), 1, true),
            (DataType::FixedSizeBinary((1 << 16) + 1), 1, false),
            // There is no array of these types.
            (list(DataType::Int64, -1), 1, false),
            (DataType::FixedSizeBinary(-1), 0, false),
            (
                DataType::Dictionary(Box::new(DataType::Utf8), Box::new(DataType::Utf8)),
                0,
                false,
            ),
            (
                DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
                1,
                true,
            ),
            // Nor of times of microseconds or nanoseconds in 32 bits, or of
            // seconds or milliseconds in 64 bits, at any depth.
            (DataType::Time32(TimeUnit::Second), 1, true),
            (DataType::Time32(TimeUnit::Millisecond), 1, true),
            (DataType::Time32(TimeUnit::Microsecond), 0, false),
            (DataType::Time32(TimeUnit::Nanosecond), 0, false),
            (DataType::Time64(TimeUnit::Second), 0, false),
            (DataType::Time64(TimeUnit::Millisecond), 0, false),
            (DataType::Time64(TimeUnit::Microsecond), 1, true),
            (DataType::Time64(TimeUnit::Nanosecond), 1, true),
            (
                DataType::new_list(DataType::Time64(TimeUnit::Second), true),
                1,
                false,
            ),
            (
                DataType::Dictionary(
                    Box::new(DataType::Int8),
                    Box::new(DataType::Time32(TimeUnit::Nanosecond)),
                ),
                1,
                false,
            ),
            // Nor of maps whose entries are not structs of a key and a value.
            (
                DataType::Map(
                    Arc::new(Field::new(
                        "entries",
                        DataType::Struct(
                            vec![
                                Field::new("key", DataType::Int32, false),
                                Field::new("value", DataType::Utf8, true),
                            ]
                            .into(),
                        ),
                        false,
                    )),
                    false,
                ),
                1,
                true,
            ),
            (DataType::Map(field(DataType::Int32), false), 0, false),
            (
                DataType::Map(
                    field(DataType::Struct(
                        vec![Field::new("key", DataType::Int32, false)].into(),
                    )),
                    false,
                ),
                0,
                false,
            ),
        ] {
            assert_eq!(
                can_build_nulls(&data_type, len),
                built,
                "{data_type} x {len}"
            );
            if built {
                assert_eq!(new_null_array(&data_type, len).len(), len);
            }
        }
    }

    #[test]
    fn builds_the_nulls_of_the_types_it_claims_to() {
        // `can_build_nulls` must only claim what `new_null_array` builds, which
        // panics otherwise: check it on many types, also nested.
        use arrow::datatypes::{IntervalUnit, UnionFields};
        let units = [
            TimeUnit::Second,
            TimeUnit::Millisecond,
            TimeUnit::Microsecond,
            TimeUnit::Nanosecond,
        ];
        let mut flat = vec![
            DataType::Null,
            DataType::Boolean,
            DataType::Int8,
            DataType::UInt64,
            DataType::Float16,
            DataType::Date32,
            DataType::Date64,
            DataType::Utf8,
            DataType::LargeUtf8,
            DataType::Utf8View,
            DataType::Binary,
            DataType::BinaryView,
            DataType::FixedSizeBinary(0),
            DataType::FixedSizeBinary(-1),
            DataType::Interval(IntervalUnit::MonthDayNano),
            DataType::Decimal32(9, 2),
            DataType::Decimal128(0, 0),
            DataType::Decimal256(100, 0),
            DataType::Union(
                UnionFields::try_new(Vec::<i8>::new(), Vec::<Field>::new()).unwrap(),
                UnionMode::Sparse,
            ),
        ];
        for unit in units {
            flat.extend([
                DataType::Time32(unit),
                DataType::Time64(unit),
                DataType::Duration(unit),
                DataType::Timestamp(unit, Some("+01:00".into())),
            ]);
        }
        let mut types = flat.clone();
        for inner in &flat {
            for nullable in [true, false] {
                let field = Arc::new(Field::new("f", inner.clone(), nullable));
                let entries = DataType::Struct(
                    vec![
                        Field::new("key", DataType::Int32, false),
                        Field::new("value", inner.clone(), nullable),
                    ]
                    .into(),
                );
                types.extend([
                    DataType::List(Arc::clone(&field)),
                    DataType::LargeListView(Arc::clone(&field)),
                    DataType::FixedSizeList(Arc::clone(&field), 2),
                    DataType::Struct(vec![Arc::clone(&field)].into()),
                    DataType::Map(Arc::new(Field::new("entries", entries, false)), false),
                    DataType::Map(Arc::clone(&field), false),
                    DataType::RunEndEncoded(
                        Arc::new(Field::new("run_ends", DataType::Int16, false)),
                        Arc::clone(&field),
                    ),
                    DataType::Dictionary(
                        Box::new(DataType::UInt16),
                        Box::new(inner.clone()),
                    ),
                ]);
                for mode in [UnionMode::Sparse, UnionMode::Dense] {
                    types.push(DataType::Union(
                        UnionFields::try_new(
                            vec![0, 5],
                            vec![
                                Field::new("a", inner.clone(), nullable),
                                Field::new("b", DataType::Int32, true),
                            ],
                        )
                        .unwrap(),
                        mode,
                    ));
                }
            }
        }
        let wrong = types
            .iter()
            .flat_map(|data_type| [(data_type, 0), (data_type, 1)])
            .filter(|&(data_type, len)| {
                can_build_nulls(data_type, len)
                    && std::panic::catch_unwind(|| new_null_array(data_type, len))
                        .is_err()
            })
            .map(|(data_type, len)| format!("{data_type} x {len}"))
            .collect::<Vec<_>>();
        assert!(wrong.is_empty(), "{wrong:#?}");
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
