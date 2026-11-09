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
//! DataFusion also evaluates each part of an expression on all rows of a
//! batch before the next one: if several parts may fail, which error it
//! raises first depends on its batches, unless all of them raise the same
//! error. Arithmetic that fails on overflows does: `+`, `-` and `*` of
//! decimals (which Arrow computes with overflow checks) only fail with an
//! arithmetic overflow, so chains of them, e.g.
//! `l_extendedprice * (1 - l_discount) * (1 + l_tax)`, do not depend on the
//! batches (see [`Failures`]).
//!
//! Functions are opaque, but for the built-in scalar functions of DataFusion
//! whose implementations are known (e.g. `date_part`, `substr` or `abs`, see
//! `BuiltinFunction`) in the sessions that trust them (see
//! [`TrustedFunctions`]): they fail on the values of a row independently of
//! the other rows, if at all, in a single step over the rows of a batch, and
//! only build strings from the bytes of their arguments (see
//! `function_fallibility`). They are identified by the types that implement
//! them, which the sessions register, so that a user-defined function that
//! shadows one (e.g. Spark's `abs`) stays opaque.
//!
//! String constants are copied for every row of a batch by whoever evaluates
//! them, so they overflow the 32-bit offsets of strings on batches of enough
//! rows: constants of a few bytes are only evaluated natively if the plan
//! bounds the rows of the batches of DataFusion (see
//! [`depends_on_batches_with_rows`]).
//!
//! Like DataFusion's optimizer, the conjuncts are not reordered at all if one
//! of them is volatile (e.g. `random() < 0.5`): reordering changes how often
//! it is evaluated.

use std::any::{Any, TypeId};
use std::fmt::{self, Write as _};
use std::ops::Range;
use std::sync::Arc;
use std::time::Duration;

use arrow::array::{RecordBatch, RecordBatchOptions, new_null_array};
use arrow::datatypes::{DataType, Field, Schema, TimeUnit, UnionMode};
use datafusion_common::config::{ConfigEntry, ConfigExtension, ExtensionOptions};
use datafusion_common::tree_node::TreeNode;
use datafusion_common::{Result, ScalarValue, config_err};
use datafusion_expr::{ColumnarValue, Operator, ScalarUDF};
use datafusion_physical_expr::expressions::{
    BinaryExpr, CaseExpr, CastExpr, Column, InListExpr, IsNotNullExpr, IsNullExpr,
    LikeExpr, Literal, NegativeExpr, NotExpr, TryCastExpr,
};
use datafusion_physical_expr::{PhysicalExpr, ScalarFunctionExpr, split_conjunction};
use datafusion_physical_expr_common::physical_expr::is_volatile;

use crate::chunk::DEFAULT_VECTOR_SIZE;

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
        let volatile = conjuncts
            .iter()
            .any(|conjunct| must_not_be_evaluated(&conjunct.expr));
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
    /// conjuncts (none if a conjunct is volatile, or dynamic, which may hide
    /// a volatile expression). Every other conjunct is always evaluated at its
    /// position in the predicate.
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
/// out of dictionaries, see `cast_fallibility`), arithmetic on integers and
/// floating point numbers that wraps on overflow (DataFusion's default),
/// divisions of numbers by a constant that is neither `0` nor `-1`, and the
/// built-in functions that cannot fail on values, in the sessions that trust
/// them (e.g. `date_part` or `strpos`, see `function_fallibility` and
/// [`TrustedFunctions`]) are infallible, if Arrow supports
/// the types of their operands: they fail uniformly otherwise (e.g. `NOT` of
/// an integer, or a comparison of dictionaries of dictionaries, see
/// `fails_on_operand_types`). `LIKE` and regular expressions with a constant
/// pattern only fail if the pattern does not compile, which does not depend on
/// the rows (the pattern is compiled before any row is matched): they are
/// infallible if evaluating them on an empty batch succeeds, and fail
/// uniformly otherwise (see `constant_pattern_fallibility` for
/// case-insensitive `LIKE`s). Everything else, other functions, decimal
/// arithmetic (which fails on overflows) and `CASE` included, may fail on
/// some rows, as may the built-in functions that copy string or binary
/// constants for every row of the batch (e.g. `strpos(s, 'x')`, see
/// `function_fallibility`), which overflow on batches of enough rows.
pub fn fallibility(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> Fallibility {
    fallibility_with(expr, schema, 0)
}

/// How evaluating `expr` can fail (see [`fallibility`]), if it may copy
/// string and binary constants of at most `bytes` bytes for every row of the
/// batches it is evaluated on (see `constant_bytes`).
fn fallibility_with(
    expr: &Arc<dyn PhysicalExpr>,
    schema: &Schema,
    bytes: usize,
) -> Fallibility {
    expr.children()
        .into_iter()
        .map(|child| fallibility_with(child, schema, bytes))
        .fold(node_fallibility(expr, schema, bytes), Ord::max)
}

/// Whether evaluating `expr` (on `schema`) can never fail, whatever the
/// values of its input rows (see [`fallibility`]).
pub fn is_infallible(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> bool {
    fallibility(expr, schema) == Fallibility::Infallible
}

/// How `expr` itself (not its children) can fail, if it may copy string and
/// binary constants of at most `bytes` bytes for every row of the batches it is
/// evaluated on (see [`depends_on_batches_with_rows`]).
fn node_fallibility(
    expr: &Arc<dyn PhysicalExpr>,
    schema: &Schema,
    bytes: usize,
) -> Fallibility {
    match value_fallibility(expr, schema, bytes) {
        // An expression that cannot fail on the values of its operands may
        // fail on their types, whatever the values (e.g. `NOT` of an integer,
        // a comparison of dictionaries of dictionaries, or a cast of booleans
        // into a dictionary), and then whenever it is evaluated.
        Fallibility::Infallible if fails_on_operand_types(expr, schema) => {
            Fallibility::Uniform
        }
        // Or on the number of rows it is evaluated on together.
        Fallibility::Infallible
            if expands_values(expr, schema, bytes)
                || builds_byte_offsets(expr, schema, bytes) =>
        {
            Fallibility::PerRow
        }
        fallibility => fallibility,
    }
}

/// How `expr` itself can fail on the values of its operands, if Arrow
/// supports their types (and copies constants of at most `bytes` bytes).
fn value_fallibility(
    expr: &Arc<dyn PhysicalExpr>,
    schema: &Schema,
    bytes: usize,
) -> Fallibility {
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
    if let Some(function) = expr.downcast_ref::<ScalarFunctionExpr>() {
        return function_fallibility(function, schema, bytes).unwrap_or(PerRow);
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
/// operands are not evaluated: they are replaced by columns of nulls. But for
/// the constant arguments of a function (see `is_constant`), which it may
/// require to be constants (e.g. the part of `date_part`) and whose values it
/// checks before it reads any row (e.g. the granularity of `date_trunc`, or
/// the pattern of `regexp_like`): they are kept.
fn fails_on_operand_types(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> bool {
    let operands = expr.children();
    if operands.is_empty() {
        return false;
    }
    let keeps_constants = expr.is::<ScalarFunctionExpr>();
    let mut fields = vec![];
    let mut nulls = vec![];
    let mut children = Vec::with_capacity(operands.len());
    for operand in operands {
        if keeps_constants && is_constant(operand) {
            children.push(Arc::clone(operand));
            continue;
        }
        let Ok(data_type) = operand.data_type(schema) else {
            return true;
        };
        // Without a null to probe with (e.g. an empty union has no value, so
        // no null), conservatively, fail.
        if !can_build_nulls(&data_type, 1) {
            return true;
        }
        let index = fields.len();
        let field = Field::new(format!("operand_{index}"), data_type, true);
        children.push(Arc::new(Column::new(field.name(), index)) as _);
        nulls.push(new_null_array(field.data_type(), 1));
        fields.push(field);
    }
    let Ok(probe) = Arc::clone(expr).with_new_children(children) else {
        return true;
    };
    if must_not_be_evaluated(&probe) {
        return true;
    }
    let options = RecordBatchOptions::new().with_row_count(Some(1));
    RecordBatch::try_new_with_options(Arc::new(Schema::new(fields)), nulls, &options)
        .map_err(Into::into)
        .and_then(|batch| probe.evaluate(&batch)?.into_array(1))
        .is_err()
}

/// How `cast` can fail on the values it casts. It cannot if it widens values
/// (see `CastExpr::is_bigger_cast` and [`is_infallible_numeric_cast`]),
/// including the values of a dictionary that it unpacks, and into the values
/// of a dictionary that it packs a constant into, or values that the keys of
/// the dictionary can always index (keys of 32 bits or more: a batch has fewer
/// rows than that), or into the values of a dictionary that it casts a
/// dictionary (but not a dictionary of dictionaries) into, whose keys hold all
/// keys of the dictionary it casts (Arrow casts the keys, which index all
/// values of the dictionary, however many: e.g. nulls take no memory), if
/// Arrow supports it (see [`fails_on_operand_types`]: e.g. Arrow cannot pack
/// booleans into a dictionary).
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
    if keys_fit
        && (CastExpr::check_bigger_cast(to_values, from_values)
            || is_infallible_numeric_cast(from_values, to_values))
    {
        Fallibility::Infallible
    } else {
        Fallibility::PerRow
    }
}

/// Whether Arrow casts every value of `from` into `to` without failing,
/// whatever its value (also beyond the precision of a decimal), although the
/// cast may lose precision (which `CastExpr::is_bigger_cast` does not allow):
///
/// * integers and decimals into `Float32` or `Float64`, which Arrow converts
///   like `as` does, rounding, and saturating to infinities;
/// * integers into decimals with enough digits for all of them
///   (`p - s >= digits`, see `integer_digits`), which Arrow then converts
///   with `unary` (see `cast_integer_to_decimal`), e.g. the
///   `CAST(1 AS DECIMAL(20, 0))` of `1 - l_discount`;
/// * decimals into decimals of at least as many bytes that keep their digits
///   (`p1 + (s2 - s1) <= p2`, with `s2 >= s1`): Arrow multiplies them by
///   `10^(s2 - s1)` without checks (see `make_upscaler`), which cannot
///   overflow the values of the precision of `from`, e.g.
///   `CAST(c_acctbal AS DECIMAL(19, 6))` of a `DECIMAL(15, 2)`.
fn is_infallible_numeric_cast(from: &DataType, to: &DataType) -> bool {
    if (from.is_integer() || from.is_decimal())
        && matches!(to, DataType::Float32 | DataType::Float64)
    {
        return true;
    }
    let Some((to_bytes, p2, s2)) = decimal_parts(to) else {
        return false;
    };
    if let Some(digits) = integer_digits(from) {
        return i16::from(p2) - i16::from(s2) >= digits;
    }
    let Some((from_bytes, p1, s1)) = decimal_parts(from) else {
        return false;
    };
    to_bytes >= from_bytes
        && s2 >= s1
        && i16::from(p1) + i16::from(s2) - i16::from(s1) <= i16::from(p2)
}

/// The number of decimal digits that no integer of `data_type` has more of,
/// the smallest number of digits whose power of ten overflows the type: e.g.
/// 10 for `Int32`, whose largest value is `2147483647`.
fn integer_digits(data_type: &DataType) -> Option<i16> {
    Some(match data_type {
        DataType::Int8 | DataType::UInt8 => 3,
        DataType::Int16 | DataType::UInt16 => 5,
        DataType::Int32 | DataType::UInt32 => 10,
        DataType::Int64 => 19,
        DataType::UInt64 => 20,
        _ => return None,
    })
}

/// The bytes of a value, the precision and the scale of the decimal type
/// `data_type`.
fn decimal_parts(data_type: &DataType) -> Option<(u8, u8, i8)> {
    match *data_type {
        DataType::Decimal32(precision, scale) => Some((4, precision, scale)),
        DataType::Decimal64(precision, scale) => Some((8, precision, scale)),
        DataType::Decimal128(precision, scale) => Some((16, precision, scale)),
        DataType::Decimal256(precision, scale) => Some((32, precision, scale)),
        _ => None,
    }
}

/// The built-in scalar functions of DataFusion (of `datafusion-functions`)
/// whose implementations the classification knows (see
/// [`function_fallibility`]), in the sessions that trust them (see
/// [`TrustedFunctions`]).
///
/// A function is one of them only if it is implemented by the very type of a
/// built-in function that the session that planned it trusts: a user-defined
/// function may shadow a built-in one under its name, e.g. the `abs` of
/// `datafusion-spark` (`SparkAbs`), which fails on overflows in ANSI mode, also
/// with a type of the same name (but of another crate), and stays opaque.
/// (This crate does not depend on `datafusion-functions`, so it cannot name
/// their types: the sessions register them.)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BuiltinFunction {
    Abs,
    BTrim,
    Ceil,
    CharacterLength,
    DatePart,
    DateTrunc,
    EndsWith,
    Exp,
    Floor,
    Ln,
    Log10,
    Lower,
    LTrim,
    NullIf,
    OctetLength,
    Power,
    RegexpLike,
    Round,
    RTrim,
    Sqrt,
    StartsWith,
    Strpos,
    Substr,
    ToTimestamp,
    Trunc,
    Upper,
}

impl BuiltinFunction {
    /// The functions, with their names (their aliases, e.g. `substring` or
    /// `trim`, call them by these names) and the names of the types of
    /// `datafusion-functions` that implement them.
    const ALL: [(Self, &'static str, &'static str); 26] = [
        (Self::Abs, "abs", "AbsFunc"),
        (Self::BTrim, "btrim", "BTrimFunc"),
        (Self::Ceil, "ceil", "CeilFunc"),
        (
            Self::CharacterLength,
            "character_length",
            "CharacterLengthFunc",
        ),
        (Self::DatePart, "date_part", "DatePartFunc"),
        (Self::DateTrunc, "date_trunc", "DateTruncFunc"),
        (Self::EndsWith, "ends_with", "EndsWithFunc"),
        (Self::Exp, "exp", "ExpFunc"),
        (Self::Floor, "floor", "FloorFunc"),
        (Self::Ln, "ln", "LnFunc"),
        (Self::Log10, "log10", "Log10Func"),
        (Self::Lower, "lower", "LowerFunc"),
        (Self::LTrim, "ltrim", "LtrimFunc"),
        (Self::NullIf, "nullif", "NullIfFunc"),
        (Self::OctetLength, "octet_length", "OctetLengthFunc"),
        (Self::Power, "power", "PowerFunc"),
        (Self::RegexpLike, "regexp_like", "RegexpLikeFunc"),
        (Self::Round, "round", "RoundFunc"),
        (Self::RTrim, "rtrim", "RtrimFunc"),
        (Self::Sqrt, "sqrt", "SqrtFunc"),
        (Self::StartsWith, "starts_with", "StartsWithFunc"),
        (Self::Strpos, "strpos", "StrposFunc"),
        (Self::Substr, "substr", "SubstrFunc"),
        (Self::ToTimestamp, "to_timestamp", "ToTimestampFunc"),
        (Self::Trunc, "trunc", "TruncFunc"),
        (Self::Upper, "upper", "UpperFunc"),
    ];

    /// The built-in function that `function` calls, if the session that
    /// planned it (whose configuration it holds) trusts the type that
    /// implements it (see [`TrustedFunctions`]).
    fn of(function: &ScalarFunctionExpr) -> Option<Self> {
        let udf = function.fun();
        let &(builtin, ..) = Self::ALL.iter().find(|(_, name, _)| *name == udf.name())?;
        let trusted = function
            .config_options()
            .extensions
            .get::<TrustedFunctions>()?;
        trusted
            .trusts(builtin, implementation_type(udf))
            .then_some(builtin)
    }

    /// The name of the function.
    fn name(self) -> &'static str {
        Self::ALL
            .iter()
            .find(|(builtin, ..)| *builtin == self)
            .map_or("", |(_, name, _)| name)
    }

    /// Whether every value of the function is (a part of) the value of its
    /// first argument on the same row, or null: it builds strings or binaries
    /// of at most as many bytes as its first argument has (see
    /// [`builds_byte_offsets`]).
    fn selects_bytes(self) -> bool {
        matches!(
            self,
            Self::Substr | Self::BTrim | Self::LTrim | Self::RTrim | Self::NullIf
        )
    }

    /// Whether argument `index` of a call with `args` is a parameter that the
    /// function reads as a scalar if it is a constant, and neither copies for
    /// every row of the batch nor returns, so that it may have any number of
    /// bytes (see `function_fallibility`): the part of `date_part` and the
    /// granularity of `date_trunc` (which must be scalars), the formats of
    /// `to_timestamp`, the characters to trim (see `Hint::AcceptsSingular`),
    /// the second argument of `nullif`, the arguments of `starts_with` and
    /// `ends_with` (which make arrays of a single row of scalars), and the
    /// pattern and the flags of `regexp_like` if both are constants. Not, e.g.,
    /// the substring that `strpos` searches, which it copies for every row of
    /// the batch (see `make_scalar_function`).
    fn is_parameter(self, index: usize, args: &[Arc<dyn PhysicalExpr>]) -> bool {
        match self {
            Self::DatePart | Self::DateTrunc => index == 0,
            Self::ToTimestamp
            | Self::BTrim
            | Self::LTrim
            | Self::RTrim
            | Self::NullIf => index > 0,
            Self::StartsWith | Self::EndsWith => true,
            Self::RegexpLike => index > 0 && args[1..].iter().all(is_constant),
            _ => false,
        }
    }

    /// Whether the function fails, if at all, with the same kind of error on
    /// all rows (see `fails_with_one_kind`): all of them but `to_timestamp`
    /// without formats, which fails to parse a string with a parse error of
    /// Arrow (see `string_to_datetime`), but on a date beyond the range of
    /// nanoseconds with an execution error. (With formats, it fails with
    /// execution errors only, see `string_to_datetime_formatted`.)
    fn fails_with_one_kind(self, args: &[Arc<dyn PhysicalExpr>]) -> bool {
        self != Self::ToTimestamp || args.len() > 1
    }
}

/// The type that implements `udf`.
fn implementation_type(udf: &ScalarUDF) -> TypeId {
    let implementation: &dyn Any = udf.inner().as_ref();
    implementation.type_id()
}

/// The name of the type that implements `udf`, which its `Debug` output
/// starts with (the built-in functions of DataFusion derive `Debug`), e.g.
/// `UpperFunc`: [`TrustedFunctions::new`] only trusts functions implemented
/// by types of the names of the built-in ones (it does not identify them, see
/// [`BuiltinFunction::of`]).
fn implementation_name(udf: &ScalarUDF) -> String {
    /// Collects the identifier that the output written to it starts with, and
    /// stops the formatting at the first other character.
    struct Identifier(String);

    impl fmt::Write for Identifier {
        fn write_str(&mut self, output: &str) -> fmt::Result {
            let end = output
                .find(|c: char| !(c.is_alphanumeric() || c == '_'))
                .unwrap_or(output.len());
            self.0.push_str(&output[..end]);
            if end < output.len() {
                Err(fmt::Error)
            } else {
                Ok(())
            }
        }
    }

    let mut name = Identifier(String::new());
    // Fails once the name is written, which stops the formatting.
    let _ = write!(name, "{:?}", udf.inner());
    name.0
}

/// The built-in scalar functions of DataFusion whose implementations a
/// session trusts the native operators to know: a configuration extension of
/// the session (see `BuiltinFunction` for the functions that they know).
///
/// The classification knows the built-in functions by the types that
/// implement them, which this crate cannot name (it does not depend on
/// `datafusion-functions`), so a session registers the built-in functions
/// that it plans with: they are known in the plans of the session, and in no
/// other (where every function is opaque, see [`depends_on_batches`]). Their
/// calls hold the configuration of the session that planned them.
///
/// ```
/// use datafusion::prelude::{SessionConfig, SessionContext};
/// use datafusion_physical_pipeline::operators::adaptive_filter::TrustedFunctions;
///
/// // The built-in functions of DataFusion, which a session registers by default.
/// let functions = datafusion::functions::all_default_functions();
/// let trusted = TrustedFunctions::new(&functions);
/// assert!(trusted.names().contains(&"date_part"));
/// let config = SessionConfig::new().with_option_extension(trusted);
/// let ctx = SessionContext::new_with_config(config);
/// ```
///
/// A call is known only if its function is implemented by the type of a
/// trusted function of its name: a user-defined function that shadows a
/// built-in one under its name stays opaque, also if its type has the name of
/// the built-in one. [`Self::new`] only trusts functions of the names, and
/// implemented by types of the names, of the built-in ones (so that a
/// session whose registry has, e.g., Spark's `abs`, of `SparkAbs`, does not
/// trust it): the caller vouches that they are DataFusion's.
#[derive(Debug, Clone, Default)]
pub struct TrustedFunctions {
    /// The trusted functions, and the types that implement them.
    implementations: Vec<(BuiltinFunction, TypeId)>,
}

impl TrustedFunctions {
    /// Trust `functions`, the built-in scalar functions of DataFusion (e.g.
    /// `datafusion::functions::all_default_functions()`): those that the
    /// classification knows, and whose types have the names of the types of
    /// DataFusion that implement them.
    pub fn new(functions: &[Arc<ScalarUDF>]) -> Self {
        let mut implementations: Vec<(BuiltinFunction, TypeId)> = vec![];
        for function in functions {
            let Some(&(builtin, _, type_name)) = BuiltinFunction::ALL
                .iter()
                .find(|(_, name, _)| *name == function.name())
            else {
                continue;
            };
            let implementation = (builtin, implementation_type(function));
            if implementation_name(function) == type_name
                && !implementations.contains(&implementation)
            {
                implementations.push(implementation);
            }
        }
        Self { implementations }
    }

    /// The names of the trusted functions.
    pub fn names(&self) -> Vec<&'static str> {
        self.implementations
            .iter()
            .map(|(builtin, _)| builtin.name())
            .collect()
    }

    /// Whether `function`, implemented by `implementation`, is trusted.
    fn trusts(&self, function: BuiltinFunction, implementation: TypeId) -> bool {
        self.implementations.contains(&(function, implementation))
    }
}

impl ExtensionOptions for TrustedFunctions {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }

    fn cloned(&self) -> Box<dyn ExtensionOptions> {
        Box::new(self.clone())
    }

    fn set(&mut self, key: &str, _value: &str) -> Result<()> {
        config_err!(
            "{key} cannot be set: functions are trusted with `TrustedFunctions::new`"
        )
    }

    fn entries(&self) -> Vec<ConfigEntry> {
        vec![ConfigEntry {
            key: format!("{}.functions", Self::PREFIX),
            value: Some(self.names().join(",")),
            description: "The built-in scalar functions whose implementations the native operators of the pipeline engine know",
        }]
    }
}

impl ConfigExtension for TrustedFunctions {
    const PREFIX: &'static str = "pipeline_trusted_functions";
}

/// How a call of a built-in function that the classification knows (see
/// [`BuiltinFunction`]) can fail on the values of its arguments, if
/// DataFusion supports their types (see [`fails_on_operand_types`]), or
/// `None` if the function is opaque (see `is_opaque`): if it is not one of
/// them, or if its arguments
///
/// * are not all flat (lists, structs, unions, maps, dictionaries or run-end
///   encoded arrays): functions map all values of the dictionaries and runs
///   they are called on, also those that no row references, or decode them
///   (e.g. `substr` decides per batch whether it decodes a dictionary), so
///   their errors and the sizes of what they build depend on the batches;
/// * or may evaluate to scalars on some batches and to arrays on others (see
///   [`may_change_shape`], e.g. a `CASE` returns the value of a branch as is
///   if all rows take it): functions evaluate scalars and arrays differently,
///   e.g. `regexp_like` compiles a scalar pattern even if all strings are
///   null, but the patterns of an array only for strings that are not null,
///   and `to_timestamp` of a scalar string fails with an array of formats;
/// * or are constants that they may copy for every row of the batch (e.g.
///   `make_scalar_function` pads scalar arguments to the rows of the batch),
///   see [`is_broadcast_constant`], but for parameters that they read as
///   scalars (e.g. the part of `date_part`, see
///   `BuiltinFunction::is_parameter`);
/// * or, for some of them, are not constants, see below.
///
/// The known functions compute the value of a row from the values of its
/// arguments on the row, and fail, if at all, on these values only: their
/// errors do not depend on the batches (unless several parts of an
/// expression may fail with different errors, see [`Failures`]). These cannot
/// fail on values, only on types or constant arguments that they do not
/// support (e.g. `date_part('epoch', t)` of times, which
/// `fails_on_operand_types` finds), whatever the rows:
///
/// * `date_part` (and `extract`), `strpos`, `character_length` (and
///   `length`), `octet_length`, `nullif`, `ln`, `log10` and `exp`, and
///   `starts_with` and `ends_with` of strings (they cast binaries into
///   strings);
/// * `substr` (and `substring`) with constant positions, of which the start
///   is not `i64::MIN` and the length is not negative (it fails on the rows
///   that are not null otherwise);
/// * `btrim` (and `trim`), `ltrim` and `rtrim` without characters to trim,
///   or with constant ones (which they cast into the type of the strings: they
///   would cast a column of them on whole batches);
/// * `abs` but of signed integers (it fails on the smallest one), and
///   `floor`, `ceil` and `trunc` of floating point numbers and `round` of them
///   to a constant number of decimal places (of decimals, they fail on values
///   beyond the precision of their results);
/// * `regexp_like` with a constant pattern (and flags), which it compiles
///   before it matches any value.
///
/// The known functions must also evaluate the rows of a batch in a single
/// step, which fails, if at all, on the first row that fails: a function that
/// processes a whole batch in steps that fail with different errors (e.g. that
/// casts all of its values before it converts any) fails with the error of
/// the first step on a batch, but with the error of the second on a chunk on
/// which the first step does not fail. So these are opaque:
///
/// * `floor`, `ceil`, `trunc` and `round` of decimals and integers, and
///   `round` to numbers of decimal places of each row (e.g.
///   `calculate_binary_math` casts the decimal places of the whole batch
///   before it rounds any value);
/// * `regexp_like` with flags, but for a constant pattern with constant
///   flags: it checks the flags of the whole batch for the global option `g`
///   (a planning error) before it compiles any pattern (an execution error),
///   and it copies flags that are views or have 64-bit offsets into 32-bit
///   offsets, which overflow on batches of more than 2 GiB of flags;
/// * `to_timestamp` but of strings: e.g. it casts decimals into
///   `Decimal128(38, 9)` (a cast error on the whole batch) before it converts
///   them into timestamps (an execution error), and unsigned integers into
///   timestamps of seconds before nanoseconds (conservatively, of all other
///   types).
///
/// The others may fail on some rows, in a single step: `substr` and `abs` but
/// for the cases above, `regexp_like` with patterns of each row and without
/// flags (it compiles the pattern of a row when it matches its value),
/// `upper` and `lower` (on a value whose conversion is too large for a
/// string), `sqrt` (of negative numbers), `power` (of zero to a negative
/// power), `date_trunc` (of timestamps beyond the range of nanoseconds) and
/// `to_timestamp` of strings (that are not timestamps).
fn function_fallibility(
    function: &ScalarFunctionExpr,
    schema: &Schema,
    bytes: usize,
) -> Option<Fallibility> {
    use BuiltinFunction::*;
    use Fallibility::*;
    let builtin = BuiltinFunction::of(function)?;
    let args = function.args();
    let types = args
        .iter()
        .map(|arg| arg.data_type(schema))
        .collect::<Result<Vec<_>>>()
        .ok()?;
    if !types.iter().all(is_flat)
        || args.iter().any(may_change_shape)
        || args.iter().enumerate().any(|(index, arg)| {
            !builtin.is_parameter(index, args)
                && is_broadcast_constant(arg, schema, bytes)
        })
    {
        return None;
    }
    let first = types.first()?;
    // Whether argument `index` is a constant, or absent.
    let constant = |index: usize| args.get(index).is_none_or(is_constant);
    Some(match builtin {
        DatePart | Strpos | CharacterLength | OctetLength | NullIf | Ln | Log10 | Exp => {
            Infallible
        }
        StartsWith | EndsWith if types.iter().all(is_string) => Infallible,
        StartsWith | EndsWith => return None,
        Substr if has_safe_positions(&args[1..]) => Infallible,
        BTrim | LTrim | RTrim if constant(1) => Infallible,
        BTrim | LTrim | RTrim => return None,
        Abs if !first.is_signed_integer() => Infallible,
        Floor | Ceil | Trunc if first.is_floating() => Infallible,
        Round if first.is_floating() && constant(1) => Infallible,
        Floor | Ceil | Trunc | Round => return None,
        RegexpLike if constant(1) && constant(2) => Infallible,
        RegexpLike if args.len() == 2 => PerRow,
        RegexpLike => return None,
        ToTimestamp if is_string(first) => PerRow,
        ToTimestamp => return None,
        Substr | Abs | Upper | Lower | Sqrt | Power | DateTrunc => PerRow,
    })
}

/// Whether arrays of `data_type` are flat: neither nested (lists, structs,
/// unions or maps), nor dictionaries, nor run-end encoded.
fn is_flat(data_type: &DataType) -> bool {
    !data_type.is_nested()
        && !matches!(
            data_type,
            DataType::Dictionary(..) | DataType::RunEndEncoded(..)
        )
}

/// Whether `data_type` is a type of intervals or of durations.
fn is_interval_or_duration(data_type: &DataType) -> bool {
    matches!(data_type, DataType::Interval(_) | DataType::Duration(_))
}

/// Whether `data_type` is a type of strings.
fn is_string(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View
    )
}

/// Whether the positions of `substr`, its start and its length (if any), are
/// literals on which it cannot fail (see `get_true_start_end`): nulls, a
/// start that is not `i64::MIN` (whose predecessor overflows) and a length
/// that is not negative.
fn has_safe_positions(positions: &[Arc<dyn PhysicalExpr>]) -> bool {
    positions.iter().enumerate().all(|(index, position)| {
        position
            .downcast_ref::<Literal>()
            .is_some_and(|literal| match *literal.value() {
                ScalarValue::Int64(None) => true,
                ScalarValue::Int64(Some(start)) if index == 0 => start != i64::MIN,
                ScalarValue::Int64(Some(length)) => length >= 0,
                _ => false,
            })
    })
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
    if must_not_be_evaluated(expr) {
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
    if must_not_be_evaluated(expr)
        || !schema
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
///   evaluates, which is not known before the execution (nor evaluated here:
///   it may be volatile);
/// * several parts of `expr` may fail, with different errors (see
///   [`Failures`]): which error DataFusion raises first depends on its
///   batches, e.g. on a batch of `("1", 0)` then `("invalid", 1)` rows,
///   `CAST(s AS INT) + 10 / x` fails to cast in DataFusion, but divides by
///   zero on the first chunk of a pipeline (but parts that can only fail with
///   arithmetic overflows, e.g. `a * (1 - b) * (1 + c)` of decimals, raise the
///   same error whatever the batches);
/// * a part of `expr` may fail with different kinds of errors on different
///   rows (see `fails_with_one_kind`): a pipeline evaluates the chunks of a
///   partition on several threads, and raises the error of the chunk that
///   fails first, which may not be the first chunk that fails, e.g. on rows
///   of `'3000-01-01'` then `'invalid'`, `to_timestamp(s)` fails in DataFusion
///   on the date beyond the range of nanoseconds (an execution error), but
///   may fail to parse `'invalid'` first in a pipeline (a parse error);
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
///   binary, nor, if the batches are known to have few enough rows, a string
///   or binary of at most [`MAX_CONSTANT_BYTES`] bytes, see
///   [`depends_on_batches_with_rows`]), into strings or binaries with 32-bit
///   offsets (at any depth): e.g. `CAST(d AS Utf8)` unpacks a dictionary of a
///   single string of 2 MiB, which 1024 rows, but not 1023, overflow, without
///   taking more memory than one row, `CASE WHEN x THEN '…' END` copies a
///   constant of 2 MiB for every row with `x`, and a constant of 1 KiB
///   overflows on a batch of 2^21 rows;
/// * `expr` is such a constant (e.g. a string that a projection outputs),
///   which whoever evaluates it copies for every row of the batch;
/// * a part of `expr` other than a column builds strings or binaries with
///   32-bit offsets from more bytes than one of its operands has, e.g.
///   `s || s`, which overflows on a batch of more than 1 GiB of strings, but
///   not on a chunk of a quarter of its rows (but for casts into the type of
///   their operand, functions that return parts of the values of their first
///   argument, e.g. `substr`, and `CASE` expressions whose values come from a
///   single column of their type, or are all constants that the batches can
///   hold for each of their rows, see [`depends_on_batches_with_rows`], e.g.
///   `CASE WHEN x THEN '1-URGENT' ELSE '2-HIGH' END`);
/// * a part of `expr` negates an operand that may evaluate to a scalar on some
///   batches and to an array on others (e.g. a `CASE`, which returns the
///   value of a branch as is if all rows take it): `-` negates a scalar with
///   overflow checks, but an array wrapping, so
///   `-(CASE WHEN x THEN y ELSE -9223372036854775808 END)` fails on a chunk
///   whose rows all take the `ELSE` branch, not on a batch with others;
/// * a part of `expr` matches strings with patterns (`LIKE` or a regular
///   expression) that may evaluate to a scalar on some batches and to an
///   array on others (or such strings): Arrow compiles the case-insensitive
///   patterns of an array, but may match a scalar one with ASCII strings
///   without compiling it, so `s ILIKE CASE WHEN x THEN 'sss…' ELSE p END`
///   fails on a batch with rows of both branches (the pattern is too big to
///   compile), not on a chunk whose rows all take the first one;
/// * a `CASE` of `expr` returns unions (at any depth): it merges the values of
///   its branches with nulls for the rows that none takes, which Arrow cannot
///   do with unions without nullable fields, so
///   `CASE b WHEN true THEN u END` fails on a batch with rows of both kinds,
///   not on chunks whose rows all take the branch, or none;
/// * a part of `expr` is opaque to the classification (a function, but for
///   the built-in functions it knows, see `function_fallibility`): whatever
///   it returns, its implementation may fail depending on the rows it is
///   evaluated on together, e.g. `repeat` fails on more than 4 GiB of strings
///   in a batch (8192 rows of `repeat(s, 1 << 19)`), which a chunk of 2048
///   rows does not reach, and a function that returns the lengths of `s || s`
///   overflows the offsets of the strings it builds;
/// * a part of `expr` that may fail is evaluated on, or produces,
///   dictionaries, e.g. `CAST(d AS INT)`, which casts all values of the
///   dictionary of `d`, even those that no row references (when DataFusion
///   coalesces batches, e.g. the output of a `FilterExec`, it merges their
///   dictionaries without these values, while chunks keep the dictionaries
///   of their batches), or `CAST(s AS Dictionary(Int8, Utf8))`, which fails
///   on batches with more than 128 distinct values.
pub fn depends_on_batches(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> bool {
    depends_on_batches_with_rows(expr, schema, None)
}

/// Whether the errors of `expr` (on `schema`) may depend on how its input
/// rows are split into batches, empty batches included (see
/// [`depends_on_batches`]), knowing that DataFusion evaluates `expr` on
/// batches of at most `max_rows` rows (if known): then string and binary
/// constants of at most [`MAX_CONSTANT_BYTES`] bytes, of which these batches
/// (and the chunks of the native operators) can hold a copy for each of their
/// rows, do not depend on the batches. Otherwise, only null and empty ones do
/// not, but for the parameters of functions that they do not copy (e.g. the
/// part of `date_part`, see `function_fallibility`): e.g. a constant of 1 KiB
/// overflows the 32-bit offsets of strings on a batch of 2^21 rows, which a
/// batch without columns has without memory.
pub fn depends_on_batches_with_rows(
    expr: &Arc<dyn PhysicalExpr>,
    schema: &Schema,
    max_rows: Option<usize>,
) -> bool {
    let bytes = constant_bytes(max_rows);
    if must_not_be_evaluated(expr) || failures_with(expr, schema, bytes).order_matters() {
        return true;
    }
    let (fallibility, depends_on_batches) = analyze(expr, schema, bytes);
    depends_on_batches
        || (fallibility != Fallibility::Infallible && fails_without_rows(expr, schema))
        // Whoever evaluates it copies it for every row of the batch.
        || is_broadcast_constant(expr, schema, bytes)
}

/// Whether `expr` must not be evaluated before the execution (to classify
/// it): a volatile part may observe how often it is evaluated (e.g. count
/// its calls), and a dynamic part (see [`is_dynamic`]) evaluates an
/// expression that its children do not show (e.g. a volatile one).
fn must_not_be_evaluated(expr: &Arc<dyn PhysicalExpr>) -> bool {
    is_volatile(expr) || is_dynamic(expr)
}

/// Whether a part of `expr` is dynamic: it has a snapshot (see
/// `PhysicalExpr::snapshot`), the expression it currently evaluates, which
/// may change during the execution, and which its children do not show (or
/// it fails to take one).
fn is_dynamic(expr: &Arc<dyn PhysicalExpr>) -> bool {
    expr.exists(|expr| Ok(!matches!(expr.snapshot(), Ok(None))))
        .unwrap_or(true)
}

/// The parts of one or more expressions that may fail, on some rows or on all
/// of them (see [`failures`]), and whether they may fail with different
/// errors.
///
/// DataFusion evaluates each part of an expression on all rows of a batch
/// before the next one, and each operator on a whole batch before the next
/// one, while a pipeline evaluates them on chunks: if several parts may fail
/// with different errors, which error comes first depends on the batches
/// (see [`Self::order_matters`]).
///
/// But parts that can only fail with arithmetic overflows (see
/// `fails_only_on_overflows`) raise the same error, whatever the batches
/// and the order in which they are evaluated: `+`, `-` and `*` of decimals,
/// which Arrow computes with overflow checks (also when DataFusion asks it to
/// wrap on overflow, its default), and of integers that DataFusion asks to
/// fail on overflow, fail with `ArrowError::ArithmeticOverflow` on the rows
/// whose values overflow, and in no other way if Arrow supports the types of
/// their operands (otherwise they fail whenever they are evaluated, see
/// [`Fallibility::Uniform`]). So DataFusion fails on a batch with such a row
/// exactly when a pipeline fails on the chunk with it, with an arithmetic
/// overflow, which does not depend on the order in which the parts or the
/// rows are evaluated: e.g. in the chains of TPC-H,
/// `l_extendedprice * (1 - l_discount) * (1 + l_tax)`.
///
/// And a pipeline evaluates the chunks of a partition of DataFusion on
/// several threads, so it raises the error of the chunk that fails first,
/// which may be any chunk that fails, while DataFusion raises the error of the
/// first row of the partition that fails: even a single part that may fail
/// with different kinds of errors on different rows (see
/// `fails_with_one_kind`), e.g. `to_timestamp(s)`, which fails to parse a
/// string with a parse error, but on a date beyond the range of nanoseconds
/// with an execution error, raises an error whose kind depends on the
/// threads.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Failures {
    /// The number of parts that may fail.
    parts: usize,
    /// The number of the parts that may fail with other errors than arithmetic
    /// overflows.
    others: usize,
    /// The number of the parts that may fail with different kinds of errors
    /// on different rows.
    mixed: usize,
}

impl Failures {
    /// No part may fail.
    pub const NONE: Self = Self {
        parts: 0,
        others: 0,
        mixed: 0,
    };

    /// A part that can only fail with arithmetic overflows.
    const OVERFLOW: Self = Self {
        parts: 1,
        others: 0,
        mixed: 0,
    };

    /// A part that may fail with another error, the same on all rows.
    const ANY: Self = Self {
        parts: 1,
        others: 1,
        mixed: 0,
    };

    /// A part that may fail with different kinds of errors on different rows.
    const MIXED: Self = Self {
        parts: 1,
        others: 1,
        mixed: 1,
    };

    /// The number of parts that may fail.
    pub fn parts(&self) -> usize {
        self.parts
    }

    /// Whether no part may fail.
    pub fn is_none(&self) -> bool {
        self.parts == 0
    }

    /// Whether all parts that may fail can only fail with arithmetic
    /// overflows (also if no part may fail).
    pub fn only_overflows(&self) -> bool {
        self.others == 0
    }

    /// Whether which error DataFusion raises first may depend on the order in
    /// which it evaluates the parts, and so on its batches: several parts may
    /// fail, not all of them only with arithmetic overflows. Or which kind of
    /// error a pipeline raises first may depend on the order in which its
    /// threads evaluate the chunks of a partition: a part may fail with
    /// different kinds of errors on different rows.
    pub fn order_matters(&self) -> bool {
        (self.parts > 1 && self.others > 0) || self.mixed > 0
    }

    /// The parts of `self` and of `other`.
    #[must_use]
    pub fn and(self, other: Self) -> Self {
        Self {
            parts: self.parts.saturating_add(other.parts),
            others: self.others.saturating_add(other.others),
            mixed: self.mixed.saturating_add(other.mixed),
        }
    }
}

/// The parts of `expr` that may fail (on some rows or on all of them), see
/// [`Failures`]: if several may fail with different errors, which error
/// DataFusion raises first depends on its batches, as it evaluates each part
/// on all rows of a batch before the next one, while a pipeline evaluates each
/// part on a chunk. A `CASE` does not fail itself, but its branches may.
pub fn failures(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> Failures {
    failures_with(expr, schema, 0)
}

/// The parts of `expr` that may fail (see [`failures`]), knowing that
/// DataFusion evaluates `expr` on batches of at most `max_rows` rows (if
/// known, see [`depends_on_batches_with_rows`]).
pub fn failures_with_rows(
    expr: &Arc<dyn PhysicalExpr>,
    schema: &Schema,
    max_rows: Option<usize>,
) -> Failures {
    failures_with(expr, schema, constant_bytes(max_rows))
}

/// The parts of `expr` that may fail, if it may copy constants of at most
/// `bytes` bytes for every row.
fn failures_with(
    expr: &Arc<dyn PhysicalExpr>,
    schema: &Schema,
    bytes: usize,
) -> Failures {
    let itself = if expr.is::<CaseExpr>() {
        Failures::NONE
    } else {
        match node_fallibility(expr, schema, bytes) {
            Fallibility::Infallible => Failures::NONE,
            Fallibility::PerRow if fails_only_on_overflows(expr, schema) => {
                Failures::OVERFLOW
            }
            Fallibility::PerRow if !fails_with_one_kind(expr, schema) => Failures::MIXED,
            _ => Failures::ANY,
        }
    };
    expr.children()
        .into_iter()
        .map(|child| failures_with(child, schema, bytes))
        .fold(itself, Failures::and)
}

/// The number of the parts of `expr` that may fail (on some rows or on all of
/// them), see [`failures`], whatever their errors: more than one may fail
/// with different errors, which DataFusion raises in an order that depends
/// on its batches, unless they can all only fail with arithmetic overflows
/// (see [`Failures::order_matters`]), which this does not tell.
pub fn failing_parts(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> usize {
    failures(expr, schema).parts()
}

/// Whether `expr`, which may fail on some rows (and not whenever it is
/// evaluated, see [`Fallibility::Uniform`]), can only fail with arithmetic
/// overflows (`ArrowError::ArithmeticOverflow`): `+`, `-` and `*` of
/// decimals of the same width, which Arrow computes with overflow checks
/// whether DataFusion asks it to wrap on overflow or not (see `decimal_op`),
/// and of integers of the same type that DataFusion asks to fail on overflow
/// (with `add_checked`, `sub_checked` and `mul_checked`; it wraps by default,
/// which is infallible), if Arrow supports the types of their operands (see
/// [`fails_on_operand_types`]: e.g. Arrow fails to multiply decimals whose
/// scales add up to more than the largest scale, whenever it multiplies
/// them). Arrow only applies them to the rows that are not null.
fn fails_only_on_overflows(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> bool {
    let Some(binary) = expr.downcast_ref::<BinaryExpr>() else {
        return false;
    };
    if !matches!(
        binary.op(),
        Operator::Plus | Operator::Minus | Operator::Multiply
    ) {
        return false;
    }
    let (Ok(left), Ok(right)) = (
        binary.left().data_type(schema),
        binary.right().data_type(schema),
    ) else {
        return false;
    };
    let same_numbers = match (decimal_parts(&left), decimal_parts(&right)) {
        (Some((left_bytes, ..)), Some((right_bytes, ..))) => left_bytes == right_bytes,
        (None, None) => left.is_integer() && left == right,
        _ => false,
    };
    same_numbers && !fails_on_operand_types(expr, schema)
}

/// Whether `expr` itself, which may fail on some rows (see
/// [`Fallibility::PerRow`]), fails with the same kind of error on all the
/// rows that it fails on (see [`Failures`]): a pipeline evaluates the chunks
/// of a partition on several threads, and raises the error of the chunk that
/// fails first, whose kind must not depend on which chunk that is. So do:
///
/// * the parts that can only fail with arithmetic overflows (see
///   `fails_only_on_overflows`), negations (which Arrow computes wrapping, but
///   for scalars, with overflow checks), `TRY_CAST`s (on types that Arrow does
///   not cast), matches of patterns of each row (that do not compile),
///   concatenations of strings (whose offsets overflow), and additions and
///   subtractions of dates, times, timestamps, intervals and durations (each
///   of which Arrow computes with one kind of check, e.g. `Date arithmetic
///   overflow` compute errors for a date and an interval, see `date_op`);
/// * divisions and remainders, see `divides_with_one_kind`;
/// * the casts that only fail with cast errors, see `casts_with_one_kind`;
/// * the built-in functions that the classification knows, but for
///   `to_timestamp` without formats (see `BuiltinFunction::fails_with_one_kind`).
///
/// Not the others (e.g. arithmetic on dates or intervals): conservatively,
/// they may fail with different kinds of errors.
fn fails_with_one_kind(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> bool {
    use Operator::*;
    if fails_only_on_overflows(expr, schema)
        || expr.is::<NegativeExpr>()
        || expr.is::<TryCastExpr>()
        || expr.is::<LikeExpr>()
    {
        return true;
    }
    if let Some(binary) = expr.downcast_ref::<BinaryExpr>() {
        let is_temporal = |operand: &Arc<dyn PhysicalExpr>| {
            operand.data_type(schema).is_ok_and(|data_type| {
                data_type.is_temporal() || is_interval_or_duration(&data_type)
            })
        };
        return match binary.op() {
            LikeMatch | ILikeMatch | NotLikeMatch | NotILikeMatch | RegexMatch
            | RegexIMatch | RegexNotMatch | RegexNotIMatch | StringConcat => true,
            Plus | Minus => is_temporal(binary.left()) || is_temporal(binary.right()),
            Divide | Modulo => divides_with_one_kind(binary, schema),
            _ => false,
        };
    }
    if let Some(cast) = expr.downcast_ref::<CastExpr>() {
        return cast
            .expr()
            .data_type(schema)
            .is_ok_and(|from| casts_with_one_kind(&from, cast.cast_type()));
    }
    if let Some(function) = expr.downcast_ref::<ScalarFunctionExpr>() {
        return BuiltinFunction::of(function)
            .is_some_and(|builtin| builtin.fails_with_one_kind(function.args()));
    }
    false
}

/// Whether the division or the remainder `binary` fails, if at all, with the
/// same kind of error on all rows (see `fails_with_one_kind`). Arrow divides
/// integers with checks, which fail with `ArrowError::DivideByZero` on a
/// divisor of `0`, but with `ArrowError::ArithmeticOverflow` on the smallest
/// integer divided by `-1`: only on one of them if the divisor is a
/// constant, or the dividend a constant that is not negative, or unsigned. It
/// computes
/// the remainders of integers wrapping, but for divisors of `0`. And it
/// rescales decimals before it divides them, which overflows, so only a
/// constant divisor that is not `0` divides them with one kind of error.
fn divides_with_one_kind(binary: &BinaryExpr, schema: &Schema) -> bool {
    let (Ok(left), Ok(right)) = (
        binary.left().data_type(schema),
        binary.right().data_type(schema),
    ) else {
        return false;
    };
    // The value of a constant operand (e.g. a literal that DataFusion casts
    // into the type of the other operand).
    let constant = |expr: &Arc<dyn PhysicalExpr>| {
        if !is_constant(expr) {
            return None;
        }
        let batch = RecordBatch::new_empty(Arc::new(Schema::empty()));
        match expr.evaluate(&batch) {
            Ok(ColumnarValue::Scalar(value)) => Some(value),
            _ => None,
        }
    };
    if left.is_integer() && right.is_integer() {
        return binary.op() == &Operator::Modulo
            || left.is_unsigned_integer()
            || constant(binary.right()).is_some()
            || constant(binary.left()).is_some_and(|dividend| {
                ScalarValue::new_zero(&left).is_ok_and(|zero| {
                    dividend
                        .partial_cmp(&zero)
                        .is_some_and(|order| order.is_ge())
                })
            });
    }
    left.is_decimal()
        && right.is_decimal()
        && constant(binary.right()).is_some_and(|divisor| {
            !divisor.is_null()
                && ScalarValue::new_zero(&right).is_ok_and(|zero| divisor != zero)
        })
}

/// Whether Arrow fails to cast values of `from` into `to`, if at all, with
/// cast errors only (`ArrowError::CastError`): numbers into numbers (see
/// `cast_numeric_arrays`), strings into numbers, booleans, dates and times
/// (see `parse_string`), and unions whose children it so casts (it casts the
/// child of the type of `to`, or of another, see `union_extract_by_type`).
/// And numbers into strings, which can only fail if their offsets overflow.
/// Conservatively, not the other casts, e.g. of strings into decimals or
/// timestamps, nor of decimals.
fn casts_with_one_kind(from: &DataType, to: &DataType) -> bool {
    let is_number =
        |data_type: &DataType| data_type.is_integer() || data_type.is_floating();
    if let DataType::Union(fields, _) = from {
        return fields.iter().all(|(_, field)| {
            field.data_type() == to || casts_with_one_kind(field.data_type(), to)
        });
    }
    (is_number(from) && (is_number(to) || is_string(to)))
        || (is_string(from)
            && (is_number(to)
                || matches!(
                    to,
                    DataType::Boolean
                        | DataType::Date32
                        | DataType::Date64
                        | DataType::Time32(_)
                        | DataType::Time64(_)
                )))
}

/// The fallibility of `expr` (see [`fallibility`]), and whether its errors
/// depend on the batches it is evaluated on, but for failures on empty
/// batches (see [`depends_on_batches`]). Each node is classified once.
fn analyze(
    expr: &Arc<dyn PhysicalExpr>,
    schema: &Schema,
    bytes: usize,
) -> (Fallibility, bool) {
    let node = node_fallibility(expr, schema, bytes);
    let children = expr.children();
    let analyzed = children
        .iter()
        .map(|child| analyze(child, schema, bytes))
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
        // bytes, or builds such arrays from more bytes than an operand has.
        || expands_values(expr, schema, bytes)
        || builds_byte_offsets(expr, schema, bytes)
        // It is opaque (e.g. a function): whatever it returns, its
        // implementation may fail depending on the rows it is evaluated on
        // together (e.g. on the size of the values it builds, which
        // DataFusion may limit per batch, or which overflow their offsets).
        || is_opaque(expr, schema, bytes)
        // `-` negates a scalar with overflow checks, but an array wrapping
        // (see `NegativeExpr`): of an operand that may be a scalar on some
        // batches and an array on others, whether it fails depends on them.
        || expr
            .downcast_ref::<NegativeExpr>()
            .is_some_and(|negative| may_change_shape(negative.arg()))
        // Arrow matches a scalar pattern and an array of patterns differently
        // (it compiles case-insensitive patterns of an array, but may match a
        // scalar one with ASCII strings without compiling it, and compiles a
        // scalar pattern even without rows): of operands that may be a scalar
        // on some batches and an array on others, whether it fails depends on
        // them.
        || (matches_patterns(expr) && expr.children().into_iter().any(may_change_shape))
        // `CASE` merges the values of its branches with nulls for the rows
        // that none takes, which Arrow cannot do with unions without nullable
        // fields (a union has no nulls of its own): whether it merges any
        // depends on the batch (all rows of a chunk may take a branch, or
        // none).
        || (expr.is::<CaseExpr>()
            && expr
                .data_type(schema)
                .is_ok_and(|data_type| contains_unions(&data_type)))
        // It may fail and evaluates or produces dictionaries.
        || (node != Fallibility::Infallible
            && children.iter().copied().chain([expr]).any(|expr| {
                expr.data_type(schema)
                    .is_ok_and(|data_type| contains_dictionaries(&data_type))
            }));
    (fallibility, depends_on_batches)
}

/// Whether `expr` is opaque to the classification: not one of the
/// expressions of DataFusion whose evaluation it knows (columns, literals,
/// `BinaryExpr`, `LIKE`, casts, `NOT`, negations, `IS [NOT] NULL`, `IN` lists,
/// `CASE`, and the calls of built-in functions that it knows, see
/// [`function_fallibility`]), which apply Arrow's kernels to their operands,
/// e.g. another function, whose implementation it cannot know (e.g. a
/// function that returns integers may build strings to compute them).
fn is_opaque(expr: &Arc<dyn PhysicalExpr>, schema: &Schema, bytes: usize) -> bool {
    if let Some(function) = expr.downcast_ref::<ScalarFunctionExpr>() {
        return function_fallibility(function, schema, bytes).is_none();
    }
    !(expr.is::<Column>()
        || expr.is::<Literal>()
        || expr.is::<BinaryExpr>()
        || expr.is::<LikeExpr>()
        || expr.is::<CastExpr>()
        || expr.is::<TryCastExpr>()
        || expr.is::<NotExpr>()
        || expr.is::<NegativeExpr>()
        || expr.is::<IsNullExpr>()
        || expr.is::<IsNotNullExpr>()
        || expr.is::<InListExpr>()
        || expr.is::<CaseExpr>())
}

/// Whether `expr` may evaluate to a scalar on some batches and to an array
/// on others: `CASE` returns the value of a branch as is if all rows take it,
/// `AND` and `OR` a constant when they short-circuit, and a function whose
/// arguments are all constants an array on a batch of a single row (see
/// `ScalarFunctionExpr::evaluate`) and a scalar on others, as may any
/// expression that has one of them as a part.
fn may_change_shape(expr: &Arc<dyn PhysicalExpr>) -> bool {
    expr.exists(|expr| {
        Ok(expr.is::<CaseExpr>()
            || expr.downcast_ref::<BinaryExpr>().is_some_and(|binary| {
                matches!(binary.op(), Operator::And | Operator::Or)
            })
            || (expr.is::<ScalarFunctionExpr>()
                && expr.children().into_iter().all(is_constant_tree)))
    })
    .unwrap_or(true)
}

/// Whether arrays of `data_type` contain unions, at any depth.
fn contains_unions(data_type: &DataType) -> bool {
    match data_type {
        DataType::Union(..) => true,
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::ListView(field)
        | DataType::LargeListView(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => contains_unions(field.data_type()),
        DataType::Struct(fields) => fields
            .iter()
            .any(|field| contains_unions(field.data_type())),
        DataType::Dictionary(_, values) => contains_unions(values),
        DataType::RunEndEncoded(_, values) => contains_unions(values.data_type()),
        _ => false,
    }
}

/// Whether `expr` matches strings with patterns: `LIKE` and the regular
/// expression operators.
fn matches_patterns(expr: &Arc<dyn PhysicalExpr>) -> bool {
    use Operator::*;
    expr.is::<LikeExpr>()
        || expr.downcast_ref::<BinaryExpr>().is_some_and(|binary| {
            matches!(
                binary.op(),
                LikeMatch
                    | ILikeMatch
                    | NotLikeMatch
                    | NotILikeMatch
                    | RegexMatch
                    | RegexIMatch
                    | RegexNotMatch
                    | RegexNotIMatch
            )
        })
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
fn expands_values(expr: &Arc<dyn PhysicalExpr>, schema: &Schema, bytes: usize) -> bool {
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
            into_byte_offsets && is_broadcast_constant(child, schema, bytes)
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
/// more bytes with 32-bit offsets (see [`contains_byte_offsets`]) than
/// whoever evaluates it may copy for every row of a batch: not a literal
/// null, nor a string or binary of at most `bytes` bytes (see
/// `has_bounded_bytes`). E.g. `ProjectionExec` outputs it as a column, and
/// `CASE WHEN x THEN '…' END` copies it for every row with `x`.
fn is_broadcast_constant(
    expr: &Arc<dyn PhysicalExpr>,
    schema: &Schema,
    bytes: usize,
) -> bool {
    if let Some(literal) = expr.downcast_ref::<Literal>() {
        return !has_bounded_bytes(expr, bytes)
            && contains_byte_offsets(&literal.value().data_type());
    }
    is_constant_tree(expr)
        && expr
            .data_type(schema)
            .is_ok_and(|data_type| contains_byte_offsets(&data_type))
}

/// The most bytes of a string or binary constant with 32-bit offsets that
/// the native operators evaluate although it may be copied for every row of a
/// batch: 1 KiB, far more than the constants of analytical queries (e.g.
/// `'1-URGENT'`), of which a chunk of 2048 rows holds at most 2 MiB.
///
/// Copied for every row of a batch, a constant overflows the 32-bit offsets of
/// strings (2 GiB) on batches of enough rows, e.g. a constant of 1 KiB on a
/// batch of 2^21 rows (which a batch without columns has without memory), but
/// not on the chunks of a pipeline. So the native operators only evaluate
/// such constants (but null or empty ones) if the plan bounds the rows of the
/// batches of DataFusion, and only as many bytes as these batches hold for
/// each of their rows (see `constant_bytes` and
/// [`depends_on_batches_with_rows`]).
pub const MAX_CONSTANT_BYTES: usize = 1024;

/// The most bytes of a string or binary constant that an expression may copy
/// for every row of batches of at most `max_rows` rows without depending on
/// them (see [`depends_on_batches_with_rows`]): none if `max_rows` is not
/// known, otherwise at most [`MAX_CONSTANT_BYTES`], and few enough for the
/// 32-bit offsets of strings (`i32::MAX` bytes) of `max_rows` copies, and of
/// the copies for the rows of a chunk (at most [`DEFAULT_VECTOR_SIZE`] rows).
fn constant_bytes(max_rows: Option<usize>) -> usize {
    max_rows.map_or(0, |rows| {
        let rows = rows.max(DEFAULT_VECTOR_SIZE);
        (i32::MAX as usize / rows).min(MAX_CONSTANT_BYTES)
    })
}

/// Whether `expr` is a literal of bounded size: a null, or a string or binary
/// of at most `bytes` bytes (see `constant_bytes`), which the batches it is
/// evaluated on can hold for each of their rows, so that copying it for every
/// row does not depend on the batches.
fn has_bounded_bytes(expr: &Arc<dyn PhysicalExpr>, bytes: usize) -> bool {
    expr.downcast_ref::<Literal>()
        .is_some_and(|literal| match literal.value() {
            ScalarValue::Utf8(value) => {
                value.as_deref().is_none_or(|value| value.len() <= bytes)
            }
            ScalarValue::Binary(value) => {
                value.as_deref().is_none_or(|value| value.len() <= bytes)
            }
            value => value.is_null(),
        })
}

/// Whether `expr` is a literal without bytes: a null, or an empty string or
/// binary.
fn has_no_bytes(expr: &Arc<dyn PhysicalExpr>) -> bool {
    expr.downcast_ref::<Literal>()
        .is_some_and(|literal| match literal.value() {
            ScalarValue::Utf8(value) => value.as_deref().is_none_or(str::is_empty),
            ScalarValue::Binary(value) => value.as_deref().is_none_or(<[u8]>::is_empty),
            value => value.is_null(),
        })
}

/// Whether `expr`, but a column or a literal (see [`is_broadcast_constant`]),
/// builds strings or binaries with 32-bit offsets (see
/// [`contains_byte_offsets`]), which overflow beyond 2 GiB of bytes in an
/// array, from more bytes than one of its operands has (whose arrays fit in
/// their offsets): e.g. `s || s` on a batch of more than 1 GiB of strings,
/// which chunks of fewer rows do not reach, or `CAST(x AS Utf8)` of enough
/// integers. Not a cast into the type of its operand, which passes it on, nor
/// a built-in function that returns parts of the values of its first argument
/// (of its type, e.g. `substr`, see `BuiltinFunction::selects_bytes`), nor a
/// `CASE` whose values are, but for literals without bytes, those of a single
/// column of its type, which it selects for each row, or literals of at most
/// `bytes` bytes (see `has_bounded_bytes`), of which it builds at most `bytes`
/// per row, e.g. `CASE WHEN x THEN '1-URGENT' ELSE '2-HIGH' END`.
fn builds_byte_offsets(
    expr: &Arc<dyn PhysicalExpr>,
    schema: &Schema,
    bytes: usize,
) -> bool {
    if expr.is::<Column>() || expr.is::<Literal>() {
        return false;
    }
    let Ok(data_type) = expr.data_type(schema) else {
        return true;
    };
    if !contains_byte_offsets(&data_type) {
        return false;
    }
    if let Some(cast) = expr.downcast_ref::<CastExpr>() {
        return cast
            .expr()
            .data_type(schema)
            .map_or(true, |from| from != data_type);
    }
    if let Some(function) = expr.downcast_ref::<ScalarFunctionExpr>()
        && BuiltinFunction::of(function).is_some_and(BuiltinFunction::selects_bytes)
    {
        return function.args().first().is_none_or(|first| {
            first
                .data_type(schema)
                .map_or(true, |first| first != data_type)
        });
    }
    if let Some(case) = expr.downcast_ref::<CaseExpr>() {
        let values = case
            .when_then_expr()
            .iter()
            .map(|(_, then)| then)
            .chain(case.else_expr())
            .collect::<Vec<_>>();
        if values.iter().all(|value| has_bounded_bytes(value, bytes)) {
            return false;
        }
        let mut values = values.into_iter().filter(|value| !has_no_bytes(value));
        let selected = values.next();
        return values.next().is_some()
            || selected.is_some_and(|value| {
                !value.is::<Column>()
                    || value
                        .data_type(schema)
                        .map_or(true, |selected| selected != data_type)
            });
    }
    true
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

    use crate::operators::filter::test_support::{
        dynamic_filter, physical_expr, physical_expr_in, trusting,
    };

    /// The most rows of the batches of DataFusion in the tests that know them
    /// (DataFusion's default batch size), which hold a constant of
    /// `MAX_CONSTANT_BYTES` bytes for each of their rows.
    const ROWS: Option<usize> = Some(8192);

    /// Whether the errors of `expr` depend on batches of at most `ROWS` rows.
    fn depends(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> bool {
        depends_on_batches_with_rows(expr, schema, ROWS)
    }

    /// How `expr` can fail on batches of at most `ROWS` rows.
    fn fallibility_on_rows(expr: &Arc<dyn PhysicalExpr>, schema: &Schema) -> Fallibility {
        fallibility_with(expr, schema, constant_bytes(ROWS))
    }

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
            ("CAST(s AS INT) = 1 AND a > 0", false),
            // `-` negates a scalar with overflow checks, but an array
            // wrapping: an operand that may be either depends on the batches.
            ("-a > 0", false),
            ("-(a + 1) > 0", false),
            ("-(CASE WHEN a > 0 THEN b ELSE 0 END) > 0", true),
            ("-(CASE WHEN a > 0 THEN b END) IS NULL", true),
            // So does Arrow match scalar patterns and arrays of patterns
            // differently (also of strings).
            ("s ILIKE s", false),
            ("s LIKE CASE WHEN a > 0 THEN 'x%' ELSE s END", true),
            ("s ILIKE CASE WHEN a > 0 THEN 'x%' ELSE s END", true),
            ("CASE WHEN a > 0 THEN s ELSE 'x' END LIKE 'y%'", true),
            ("s ~ CASE WHEN a > 0 THEN 'x' ELSE s END", true),
            // But functions are opaque: whatever they return, they may fail
            // depending on the rows they are evaluated on together (e.g.
            // `repeat` limits the size of its strings).
            ("regexp_count(s, 'a+') > 0 AND a > 0", true),
            ("regexp_count(s, 'a{1000000}') > 0 AND a > 0", true),
            ("repeat(s, b) IS NULL", true),
            // Unless they are built-in functions whose implementations are
            // known, which cannot fail on values, also after other conjuncts
            // or in `OR`.
            ("length(s) = 1 AND a > 0", false),
            ("a > 0 AND length(s) = 1", false),
            ("a > 0 AND strpos(u, 'x') > 1", false),
            ("a > 0 OR character_length(u) = 2", false),
            ("a > 0 AND substr(s, 1, 2) = 'ab'", false),
            ("a > 0 AND regexp_like(s, 'a+')", false),
            ("a > 0 AND abs(c) > 1.0", false),
            // But for those that build strings with 32-bit offsets from more
            // bytes than their arguments have (`upper` of a 'ΐ' is 3 times
            // longer), and those that may fail on some rows after other
            // conjuncts (`abs` of the smallest integer, a negative length).
            ("upper(s) = 'X' AND a > 0", true),
            ("upper(u) = 'X'", false),
            ("a > 0 AND upper(u) = 'X'", true),
            ("a > 0 AND abs(b) > 1", true),
            ("abs(b) > 1 AND a > 0", false),
            ("a > 0 AND substr(s, 1, -1) = 'x'", true),
            ("a > 0 AND substr(s, 1, b) = 'x'", true),
            ("substr(s, 1, b) = 'x' AND a > 0", false),
            // A pattern that does not compile fails whenever it is evaluated.
            ("a > 0 AND regexp_like(s, 'a{1000000}')", true),
            ("regexp_like(s, 'a{1000000}')", true),
            ("a > 0 AND regexp_like(s, s)", true),
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
            assert_eq!(depends(&expr, &schema), expected, "{predicate}");
        }
        // On batches of unknown sizes, `strpos` copies the substring it
        // searches for every row of a batch, which may overflow (see
        // `constants_need_batches_of_known_sizes`).
        let expr = physical_expr("a > 0 AND strpos(u, 'x') > 1", &schema);
        assert!(depends_on_batches(&expr, &schema));
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
            ("CAST(s AS INT) = 1 AND a > 0", false),
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
    fn cases_of_unions_depend_on_batches() {
        let union = DataType::Union(
            arrow::datatypes::UnionFields::try_new(
                [0],
                [Field::new("i", DataType::Int32, false)],
            )
            .unwrap(),
            UnionMode::Dense,
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("b", DataType::Boolean, true),
            Field::new("u", union.clone(), false),
            Field::new(
                "t",
                DataType::Struct(vec![Field::new("u", union, false)].into()),
                true,
            ),
        ]));
        let case = |then: &str| -> Arc<dyn PhysicalExpr> {
            Arc::new(
                CaseExpr::try_new(
                    None,
                    vec![(physical_expr("b", &schema), physical_expr(then, &schema))],
                    None,
                )
                .unwrap(),
            )
        };
        assert!(depends_on_batches(&case("u"), &schema));
        assert!(depends_on_batches(&case("t"), &schema));
        assert!(!depends_on_batches(&case("1"), &schema));
        assert!(!depends_on_batches(
            &physical_expr("u IS NULL", &schema),
            &schema
        ));
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
    fn finds_strings_built_from_more_bytes() {
        // Strings with 32-bit offsets built from more bytes than an operand
        // has overflow beyond 2 GiB of bytes in a batch, which a batch of more
        // rows reaches without its columns overflowing.
        let schema = Arc::new(Schema::new(vec![
            Field::new("s", DataType::Utf8, true),
            Field::new("t", DataType::Utf8, true),
            Field::new("l", DataType::LargeUtf8, true),
            Field::new("a", DataType::Int32, true),
            Field::new("e", DataType::Boolean, true),
        ]));
        let cast = |name: &str, to: DataType| -> Arc<dyn PhysicalExpr> {
            Arc::new(CastExpr::new(physical_expr(name, &schema), to, None))
        };
        let case = |then: &str, otherwise: Option<&str>| -> Arc<dyn PhysicalExpr> {
            Arc::new(
                CaseExpr::try_new(
                    None,
                    vec![(physical_expr("e", &schema), physical_expr(then, &schema))],
                    otherwise.map(|otherwise| physical_expr(otherwise, &schema)),
                )
                .unwrap(),
            )
        };
        for (expr, depends) in [
            (physical_expr("s || s", &schema), true),
            (physical_expr("s || t", &schema), true),
            (cast("a", DataType::Utf8), true),
            (cast("l", DataType::Utf8), true),
            (case("s", Some("t")), true),
            (case("s", Some("'x'")), true),
            // A cast into the type of its operand passes it on, and a `CASE`
            // selects, for each row, a value of its only column with bytes.
            (cast("s", DataType::Utf8), false),
            (cast("s", DataType::LargeUtf8), false),
            (case("s", None), false),
            (case("s", Some("''")), false),
            (case("NULL", Some("s")), false),
            (physical_expr("s = t", &schema), false),
        ] {
            assert_eq!(depends_on_batches(&expr, &schema), depends, "{expr}");
        }
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
            // But not for `IS NULL`, nor for unions of values without offsets.
            // (A sparse union does not share values, but casting it into
            // strings builds them, see `finds_strings_built_from_more_bytes`.)
            (
                Arc::new(IsNullExpr::new(physical_expr("du", &schema))),
                (Infallible, false),
            ),
            (cast("su", DataType::Utf8), (PerRow, true)),
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

        // Constants of more than `MAX_CONSTANT_BYTES` bytes, which are copied
        // for every row, on batches of at most `ROWS` rows, and constants of
        // any bytes on batches of unknown sizes (see
        // `constants_need_batches_of_known_sizes`).
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
        let large = "x".repeat(MAX_CONSTANT_BYTES + 1);
        let bounded = "x".repeat(MAX_CONSTANT_BYTES);
        for (expr, expected) in [
            (case(string(&large)), (true, true)),
            (
                case(ScalarValue::Binary(Some(vec![1; MAX_CONSTANT_BYTES + 1]))),
                (true, true),
            ),
            (constant(string(&large)), (true, true)),
            (
                Arc::new(IsNullExpr::new(case(string(&large)))) as Arc<dyn PhysicalExpr>,
                (true, true),
            ),
            // Unless they are of bounded sizes (which a batch of at most `ROWS`
            // rows holds for each of its rows).
            (case(string("abc")), (false, true)),
            (case(string(&bounded)), (false, true)),
            (case(ScalarValue::Binary(Some(vec![1]))), (false, true)),
            (constant(string("abc")), (false, true)),
            (constant(string(&bounded)), (false, true)),
            (
                Arc::new(IsNullExpr::new(case(string("abc")))) as Arc<dyn PhysicalExpr>,
                (false, true),
            ),
            // Or null or empty, with 64-bit offsets or views, or compared.
            (case(string("")), (false, false)),
            (case(ScalarValue::Utf8(None)), (false, false)),
            (constant(string("")), (false, false)),
            (
                case(ScalarValue::LargeUtf8(Some(large.clone()))),
                (false, false),
            ),
            (
                case(ScalarValue::Utf8View(Some(large.clone()))),
                (false, false),
            ),
            (case(ScalarValue::Int64(Some(1))), (false, false)),
            (physical_expr("s = 'abc'", &schema), (false, false)),
            (
                physical_expr(&format!("s = '{large}'"), &schema),
                (false, false),
            ),
        ] {
            assert_eq!(
                (depends(&expr, &schema), depends_on_batches(&expr, &schema)),
                expected,
                "{expr}"
            );
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

    /// Columns of TPC-H's tables (`DECIMAL(15, 2)` prices, `DATE` dates,
    /// strings and views), and of other types.
    fn tpch_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("l_quantity", DataType::Decimal128(15, 2), true),
            Field::new("l_extendedprice", DataType::Decimal128(15, 2), true),
            Field::new("l_discount", DataType::Decimal128(15, 2), true),
            Field::new("l_tax", DataType::Decimal128(15, 2), true),
            Field::new("o_orderdate", DataType::Date32, true),
            Field::new("o_orderpriority", DataType::Utf8, true),
            Field::new("c_phone", DataType::Utf8View, true),
            Field::new("ps_availqty", DataType::Int32, true),
            Field::new("big", DataType::Decimal256(50, 2), true),
            Field::new("s", DataType::Utf8, true),
            Field::new("x", DataType::Float64, true),
            Field::new("n", DataType::Int64, true),
            Field::new("ts", DataType::Timestamp(TimeUnit::Nanosecond, None), true),
            Field::new(
                "d",
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                true,
            ),
        ]))
    }

    #[test]
    fn decimal_arithmetic_fails_only_on_overflows() {
        use Fallibility::*;
        let schema = tpch_schema();
        for (expr, only_overflows) in [
            // The chains of TPC-H (Q1, Q7 to Q10, Q14): the integers are cast
            // into decimals with enough digits for them, which cannot fail.
            ("l_extendedprice * (1 - l_discount) * (1 + l_tax)", true),
            (
                "l_extendedprice * (1 - l_discount) - l_tax * l_quantity",
                true,
            ),
            ("l_extendedprice * (1 - l_discount) + big", true),
            ("ps_availqty * l_extendedprice + l_quantity", true),
            // A cast of strings, or a division, may fail otherwise.
            (
                "l_extendedprice * (1 - l_discount) + CAST(s AS DECIMAL(15, 2))",
                false,
            ),
            (
                "l_extendedprice / l_discount + l_extendedprice * l_tax",
                false,
            ),
            ("l_extendedprice * l_tax + 10 / n", false),
        ] {
            let expr = physical_expr(expr, &schema);
            let failures = failures(&expr, &schema);
            assert!(failures.parts() > 1, "{expr}: {failures:?}");
            assert_eq!(failing_parts(&expr, &schema), failures.parts(), "{expr}");
            assert_eq!(failures.only_overflows(), only_overflows, "{expr}");
            assert_eq!(failures.order_matters(), !only_overflows, "{expr}");
            assert_eq!(
                depends_on_batches(&expr, &schema),
                !only_overflows,
                "{expr}"
            );
            assert_eq!(fallibility(&expr, &schema), PerRow, "{expr}");
        }
        // Decimals that Arrow does not multiply (their scales add up to more
        // than 38) fail whenever they are multiplied, not with overflows.
        let scales = Arc::new(Schema::new(vec![
            Field::new("p", DataType::Decimal128(38, 30), true),
            Field::new("q", DataType::Decimal128(38, 20), true),
        ]));
        let product: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            physical_expr("p", &scales),
            Operator::Multiply,
            physical_expr("q", &scales),
        ));
        assert!(!failures(&product, &scales).only_overflows());
        assert!(depends_on_batches(&product, &scales));
        // Arithmetic on integers that fails on overflow only fails with
        // overflows too, with decimal arithmetic or not.
        let checked = |left: Arc<dyn PhysicalExpr>, op, right| -> Arc<dyn PhysicalExpr> {
            Arc::new(BinaryExpr::new(left, op, right).with_fail_on_overflow(true))
        };
        let n = || physical_expr("n", &schema);
        let integers =
            checked(checked(n(), Operator::Multiply, n()), Operator::Minus, n());
        assert_eq!(failures(&integers, &schema).parts(), 2);
        assert!(failures(&integers, &schema).only_overflows());
        assert!(!depends_on_batches(&integers, &schema));
        let with_decimals: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            physical_expr("CAST(l_tax * l_quantity AS DOUBLE)", &schema),
            Operator::Gt,
            Arc::new(CastExpr::new(integers, DataType::Float64, None)),
        ));
        assert!(failures(&with_decimals, &schema).only_overflows());
        assert!(!depends_on_batches(&with_decimals, &schema));
        // Several parts that only fail on overflows, behind a guard, still
        // depend on the batches: DataFusion decides per batch on which rows
        // it evaluates them.
        let guarded = physical_expr(
            "l_tax > 0 AND l_extendedprice * (1 - l_discount) > 1",
            &schema,
        );
        assert!(depends_on_batches(&guarded, &schema));
        // Failures add up: arithmetic overflows and other errors are told
        // apart.
        let overflow = failures(&physical_expr("l_tax * l_quantity", &schema), &schema);
        let other = failures(&physical_expr("10 / n", &schema), &schema);
        assert!(overflow.only_overflows() && !overflow.order_matters());
        assert!(!other.only_overflows() && !other.order_matters());
        assert!(!overflow.and(overflow).order_matters());
        assert!(overflow.and(other).order_matters());
        assert!(other.and(other).order_matters());
        assert!(!other.and(Failures::NONE).order_matters());
        assert!(Failures::NONE.is_none() && Failures::NONE.only_overflows());
    }

    #[test]
    fn widening_numeric_casts_cannot_fail() {
        use Fallibility::*;
        let decimal = |precision, scale| DataType::Decimal128(precision, scale);
        for (from, to, expected) in [
            // Decimals into decimals that keep their digits.
            (decimal(15, 2), decimal(19, 6), Infallible),
            (decimal(15, 2), decimal(15, 2), Infallible),
            (decimal(15, 2), decimal(38, 2), Infallible),
            (decimal(15, 2), DataType::Decimal256(19, 6), Infallible),
            (DataType::Decimal32(9, 2), decimal(9, 2), Infallible),
            (decimal(15, 2), decimal(18, 6), PerRow),
            (decimal(15, 2), decimal(15, 1), PerRow),
            (decimal(15, 2), decimal(14, 2), PerRow),
            (DataType::Decimal256(15, 2), decimal(20, 2), PerRow),
            (decimal(15, 2), DataType::Decimal64(18, 2), PerRow),
            // Decimals and integers into floats.
            (decimal(15, 2), DataType::Float64, Infallible),
            (DataType::Decimal256(76, 0), DataType::Float32, Infallible),
            (DataType::Int64, DataType::Float32, Infallible),
            (DataType::UInt64, DataType::Float64, Infallible),
            (DataType::Float64, decimal(38, 10), PerRow),
            (decimal(15, 2), DataType::Int64, PerRow),
            // Integers into decimals with enough digits for all of them.
            (DataType::Int64, decimal(20, 0), Infallible),
            (DataType::Int64, decimal(19, 0), Infallible),
            (DataType::Int64, decimal(21, 2), Infallible),
            (DataType::Int64, decimal(18, 0), PerRow),
            (DataType::Int32, decimal(10, 0), Infallible),
            (DataType::Int32, decimal(12, 3), PerRow),
            (DataType::UInt64, decimal(20, 0), Infallible),
            (DataType::UInt64, decimal(19, 0), PerRow),
            (DataType::Int8, DataType::Decimal32(3, 0), Infallible),
            (DataType::UInt8, DataType::Decimal32(2, 0), PerRow),
        ] {
            let schema = Schema::new(vec![Field::new("v", from.clone(), true)]);
            let cast: Arc<dyn PhysicalExpr> = Arc::new(CastExpr::new(
                Arc::new(Column::new("v", 0)),
                to.clone(),
                None,
            ));
            assert_eq!(fallibility(&cast, &schema), expected, "{from} into {to}");
            if expected == Infallible {
                // Arrow casts any value of the type, beyond its precision
                // too, the largest and the smallest one included.
                let values = match &from {
                    DataType::Decimal128(..) => Arc::new(
                        arrow::array::Decimal128Array::from(vec![
                            i128::MAX,
                            i128::MIN,
                            0,
                        ])
                        .with_data_type(from.clone()),
                    )
                        as Arc<dyn arrow::array::Array>,
                    DataType::Decimal256(..) => Arc::new(
                        arrow::array::Decimal256Array::from(vec![
                            arrow_buffer::i256::MAX,
                            arrow_buffer::i256::MIN,
                        ])
                        .with_data_type(from.clone()),
                    ),
                    DataType::Decimal32(..) => Arc::new(
                        arrow::array::Decimal32Array::from(vec![i32::MAX, i32::MIN])
                            .with_data_type(from.clone()),
                    ),
                    DataType::Int64 => {
                        Arc::new(arrow::array::Int64Array::from(vec![i64::MAX, i64::MIN]))
                    }
                    DataType::Int32 => {
                        Arc::new(arrow::array::Int32Array::from(vec![i32::MAX, i32::MIN]))
                    }
                    DataType::Int8 => {
                        Arc::new(arrow::array::Int8Array::from(vec![i8::MAX, i8::MIN]))
                    }
                    DataType::UInt64 => {
                        Arc::new(arrow::array::UInt64Array::from(vec![u64::MAX, 0]))
                    }
                    other => panic!("no values of {other}"),
                };
                let batch =
                    RecordBatch::try_new(Arc::new(schema.clone()), vec![values]).unwrap();
                let cast = cast.evaluate(&batch).and_then(|value| value.into_array(2));
                assert!(cast.is_ok(), "{from} into {to}: {cast:?}");
            }
        }
    }

    #[test]
    fn builtin_functions_are_known() {
        use Fallibility::*;
        let schema = tpch_schema();
        let large = "x".repeat(MAX_CONSTANT_BYTES + 1);
        for (expr, expected, depends_on) in [
            // Infallible built-in functions (but for the types and the
            // constant arguments they do not support).
            (
                "extract(year from o_orderdate)".to_owned(),
                Infallible,
                false,
            ),
            ("date_part('month', ts)".to_owned(), Infallible, false),
            ("date_part('invalid', ts)".to_owned(), Uniform, true),
            (
                "substring(c_phone from 1 for 2)".to_owned(),
                Infallible,
                false,
            ),
            ("substr(s, 3)".to_owned(), Infallible, false),
            (
                "substr(o_orderpriority, 1, 1) = '1'".to_owned(),
                Infallible,
                false,
            ),
            ("trim(s)".to_owned(), Infallible, false),
            ("ltrim(c_phone, '1')".to_owned(), Infallible, false),
            ("rtrim(s, 'x')".to_owned(), Infallible, false),
            ("strpos(c_phone, '1')".to_owned(), Infallible, false),
            ("strpos(s, s)".to_owned(), Infallible, false),
            ("character_length(s)".to_owned(), Infallible, false),
            ("length(c_phone)".to_owned(), Infallible, false),
            ("octet_length(s)".to_owned(), Infallible, false),
            ("starts_with(s, c_phone)".to_owned(), Infallible, false),
            ("ends_with(c_phone, '9')".to_owned(), Infallible, false),
            ("nullif(c_phone, '10')".to_owned(), Infallible, false),
            ("nullif(n, 0)".to_owned(), Infallible, false),
            ("abs(x)".to_owned(), Infallible, false),
            ("abs(l_discount)".to_owned(), Infallible, false),
            ("floor(x) + ceil(x)".to_owned(), Infallible, false),
            (
                "round(x, 2) + round(x) + trunc(x)".to_owned(),
                Infallible,
                false,
            ),
            ("ln(x) + log10(x) + exp(x)".to_owned(), Infallible, false),
            ("regexp_like(s, '^[0-9]+$')".to_owned(), Infallible, false),
            (
                "regexp_like(c_phone, 'a{1000000}')".to_owned(),
                Uniform,
                true,
            ),
            // Built-in functions that may fail on some rows.
            ("abs(n)".to_owned(), PerRow, false),
            ("sqrt(x)".to_owned(), PerRow, false),
            ("power(x, 2)".to_owned(), PerRow, false),
            ("substr(s, 1, -1)".to_owned(), PerRow, false),
            ("substr(c_phone, n, 2)".to_owned(), PerRow, false),
            ("regexp_like(s, c_phone)".to_owned(), PerRow, false),
            ("upper(c_phone)".to_owned(), PerRow, false),
            ("date_trunc('month', ts)".to_owned(), PerRow, false),
            ("to_timestamp(s, '%Y-%m-%d')".to_owned(), PerRow, false),
            // Without formats, `to_timestamp` fails with parse errors and with
            // execution errors, of which a pipeline may raise either first.
            ("to_timestamp(s)".to_owned(), PerRow, true),
            // `upper` and `lower` build strings with 32-bit offsets of more bytes
            // than their arguments have, which overflow on a batch.
            ("lower(s)".to_owned(), PerRow, true),
            // A failure on empty batches, which DataFusion evaluates.
            ("date_trunc('invalid', ts)".to_owned(), PerRow, true),
            // Opaque: built-in functions that process a whole batch in steps
            // that fail with different errors (e.g. cast decimal places of all
            // rows before rounding any value, check flags of all rows for `g`
            // before compiling any pattern, or cast decimals before converting
            // them into timestamps), which fail first on a batch and on a
            // chunk differently, and, conservatively, `to_timestamp` of
            // anything but strings.
            ("ceil(l_discount)".to_owned(), PerRow, true),
            (
                "floor(l_discount) + trunc(l_discount)".to_owned(),
                PerRow,
                true,
            ),
            ("round(l_discount, 1)".to_owned(), PerRow, true),
            ("round(x, ps_availqty)".to_owned(), PerRow, true),
            ("regexp_like(s, c_phone, 'i')".to_owned(), PerRow, true),
            ("regexp_like(s, 'a', s)".to_owned(), PerRow, true),
            ("to_timestamp(l_quantity)".to_owned(), PerRow, true),
            ("to_timestamp(n)".to_owned(), PerRow, true),
            ("to_timestamp(x)".to_owned(), PerRow, true),
            // Opaque: not a built-in function that the classification knows.
            ("repeat(s, 2)".to_owned(), PerRow, true),
            ("concat(s, c_phone)".to_owned(), PerRow, true),
            ("md5(s)".to_owned(), PerRow, true),
            // Nor of dictionaries, nor of characters to trim that are not
            // constants, nor of constants of more than `MAX_CONSTANT_BYTES`.
            ("substr(d, 1, 2)".to_owned(), PerRow, true),
            ("character_length(d)".to_owned(), PerRow, true),
            ("btrim(s, s)".to_owned(), PerRow, true),
            (format!("strpos(s, '{large}')"), PerRow, true),
            (format!("strpos(s, '{}')", &large[1..]), Infallible, false),
            // Nor of arguments that may be scalars on some batches and arrays
            // on others, which they evaluate differently (`to_timestamp` of a
            // scalar fails with an array of formats).
            (
                "to_timestamp(CASE WHEN n > 0 THEN '2020-01-01' END, s)".to_owned(),
                PerRow,
                true,
            ),
            (
                "regexp_like(s, CASE WHEN n > 0 THEN '[' ELSE 'a' END)".to_owned(),
                PerRow,
                true,
            ),
            (
                "regexp_like(s, 'a', CASE WHEN n > 0 THEN 'i' END)".to_owned(),
                PerRow,
                true,
            ),
            ("abs(CASE WHEN n > 0 THEN x END)".to_owned(), PerRow, true),
            // But calls of functions of columns are arrays.
            ("abs(abs(x))".to_owned(), Infallible, false),
        ] {
            let expr = physical_expr(&expr, &schema);
            assert_eq!(fallibility_on_rows(&expr, &schema), expected, "{expr}");
            assert_eq!(depends(&expr, &schema), depends_on, "{expr}");
        }
        // After other conjuncts, those that may fail on some rows depend on
        // the batches, the others do not.
        for (expr, depends) in [
            ("n > 0 AND extract(year from o_orderdate) = 1995", false),
            (
                "n > 0 AND substring(c_phone from 1 for 2) IN ('13', '31')",
                false,
            ),
            ("n > 0 OR regexp_like(s, '^a')", false),
            ("n > 0 AND sqrt(x) > 1", true),
            ("n > 0 AND date_trunc('month', ts) > ts", true),
        ] {
            let expr = physical_expr(expr, &schema);
            assert_eq!(depends_on_batches(&expr, &schema), depends, "{expr}");
        }
        // A function whose arguments are all constants returns an array on a
        // batch of one row, and a scalar on others: `-` negates them
        // differently.
        let negated = Arc::new(NegativeExpr::new(physical_expr("abs(-5)", &schema)))
            as Arc<dyn PhysicalExpr>;
        assert!(depends_on_batches(&negated, &schema));
        let negated = Arc::new(NegativeExpr::new(physical_expr("abs(x)", &schema)))
            as Arc<dyn PhysicalExpr>;
        assert!(!depends_on_batches(&negated, &schema));
    }

    /// A user-defined function named `abs`, which shadows the built-in one.
    #[derive(Debug, PartialEq, Eq, Hash)]
    struct ShadowingAbs {
        signature: datafusion_expr::Signature,
    }

    impl datafusion_expr::ScalarUDFImpl for ShadowingAbs {
        fn name(&self) -> &str {
            "abs"
        }

        fn signature(&self) -> &datafusion_expr::Signature {
            &self.signature
        }

        fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
            Ok(arg_types[0].clone())
        }

        fn invoke_with_args(
            &self,
            args: datafusion_expr::ScalarFunctionArgs,
        ) -> Result<ColumnarValue> {
            Ok(args.args[0].clone())
        }
    }

    /// A user-defined function named `abs`, which shadows the built-in one,
    /// and whose type has the name of the type of the built-in one.
    #[derive(Debug, PartialEq, Eq, Hash)]
    struct AbsFunc {
        signature: datafusion_expr::Signature,
    }

    impl datafusion_expr::ScalarUDFImpl for AbsFunc {
        fn name(&self) -> &str {
            "abs"
        }

        fn signature(&self) -> &datafusion_expr::Signature {
            &self.signature
        }

        fn return_type(&self, arg_types: &[DataType]) -> Result<DataType> {
            Ok(arg_types[0].clone())
        }

        fn invoke_with_args(
            &self,
            args: datafusion_expr::ScalarFunctionArgs,
        ) -> Result<ColumnarValue> {
            Ok(args.args[0].clone())
        }
    }

    #[test]
    fn builtin_functions_are_identified_by_their_implementations() {
        use datafusion_common::config::ConfigOptions;
        // Every known function is the built-in function of its name, which a
        // session registers by default, in the sessions that trust it.
        let ctx = datafusion::prelude::SessionContext::new();
        let trusted =
            TrustedFunctions::new(&datafusion::functions::all_default_functions());
        assert_eq!(trusted.names().len(), BuiltinFunction::ALL.len());
        let mut trusting_options = ConfigOptions::default();
        trusting_options.extensions.insert(trusted);
        let trusting_options = Arc::new(trusting_options);
        for (builtin, name, implementation) in BuiltinFunction::ALL {
            let udf = datafusion::execution::FunctionRegistry::udf(&ctx, name)
                .unwrap_or_else(|_| panic!("no built-in function {name}"));
            assert_eq!(udf.name(), name);
            assert_eq!(builtin.name(), name);
            assert_eq!(implementation_name(&udf), implementation, "{name}");
            let call = |options: &Arc<ConfigOptions>| {
                ScalarFunctionExpr::new(
                    name,
                    Arc::clone(&udf),
                    vec![],
                    Arc::new(Field::new(name, DataType::Int64, true)),
                    Arc::clone(options),
                )
            };
            assert_eq!(
                BuiltinFunction::of(&call(&trusting_options)),
                Some(builtin),
                "{name}"
            );
            // Not in a session that does not trust it.
            let untrusting_options = Arc::new(ConfigOptions::default());
            assert_eq!(BuiltinFunction::of(&call(&untrusting_options)), None);
        }
        // A function that shadows a built-in one under its name is opaque in
        // a session that trusts the built-in one, also if its type has the
        // name of the type of the built-in one.
        let schema = tpch_schema();
        let session = datafusion::prelude::SessionContext::new_with_config(trusting(
            datafusion::prelude::SessionConfig::new(),
        ));
        let builtin = physical_expr_in(&session, "abs(x)", &schema);
        assert!(!depends(&builtin, &schema));
        assert_eq!(fallibility(&builtin, &schema), Fallibility::Infallible);
        let signature =
            datafusion_expr::Signature::any(1, datafusion_expr::Volatility::Immutable);
        let shadowing = ScalarUDF::new_from_impl(ShadowingAbs {
            signature: signature.clone(),
        });
        assert_eq!(implementation_name(&shadowing), "ShadowingAbs");
        let same_name = ScalarUDF::new_from_impl(AbsFunc { signature });
        assert_eq!(implementation_name(&same_name), "AbsFunc");
        for udf in [shadowing.clone(), same_name] {
            session.register_udf(udf);
            let call = physical_expr_in(&session, "abs(x)", &schema);
            assert_eq!(call.to_string(), builtin.to_string());
            let function = call.downcast_ref::<ScalarFunctionExpr>().unwrap();
            assert_eq!(BuiltinFunction::of(function), None);
            assert!(depends(&call, &schema));
            assert_eq!(fallibility(&call, &schema), Fallibility::PerRow);
        }
        // The functions of the names of built-in ones whose types do not have
        // the names of theirs are not trusted.
        assert!(
            TrustedFunctions::new(&[Arc::new(shadowing)])
                .names()
                .is_empty()
        );
        // Nor can a session trust functions by setting the extension (e.g. with
        // `SET`), whose entries only tell the trusted ones.
        let mut options = ConfigOptions::default();
        options.extensions.insert(TrustedFunctions::default());
        assert!(
            options
                .set("pipeline_trusted_functions.functions", "abs")
                .is_err()
        );
        let entries = trusting_options.entries();
        let entry = entries
            .iter()
            .find(|entry| entry.key == "pipeline_trusted_functions.functions")
            .unwrap();
        assert!(entry.value.as_deref().unwrap().contains("date_part"));
        // Nor are any in the sessions that do not trust them.
        let untrusted = physical_expr_in(
            &datafusion::prelude::SessionContext::new(),
            "abs(x)",
            &schema,
        );
        assert!(depends(&untrusted, &schema));
        assert_eq!(fallibility(&untrusted, &schema), Fallibility::PerRow);
    }

    #[test]
    fn parts_that_fail_with_several_kinds_of_errors_depend_on_batches() {
        // A pipeline evaluates the chunks of a partition on several threads,
        // and raises the error of the chunk that fails first: a part that may
        // fail with different kinds of errors on different rows raises an
        // error whose kind depends on the threads.
        let schema = tpch_schema();
        for (expr, expected) in [
            // Parse errors, and execution errors beyond the nanoseconds.
            ("to_timestamp(s)", true),
            ("to_timestamp(s, '%Y-%m-%d')", false),
            // Divisions by zero, and of the smallest integer by -1 (overflows).
            ("n / n", true),
            ("10 / n", false),
            ("-10 / n", true),
            ("n / 0", false),
            ("n / -1", false),
            ("n % n", false),
            // Overflows when decimals are rescaled, and divisions by zero.
            ("l_extendedprice / l_discount", true),
            ("l_extendedprice % l_discount", true),
            ("l_extendedprice / 7", false),
            ("l_extendedprice / 0", true),
            // Casts with cast errors only.
            ("CAST(n AS INT)", false),
            ("CAST(x AS BIGINT)", false),
            ("CAST(s AS INT)", false),
            ("CAST(s AS DATE)", false),
            ("CAST(s AS BOOLEAN)", false),
            ("CAST(n AS VARCHAR) = s", false),
            // Casts of strings into decimals, or of decimals into narrower
            // ones, also fail on their precision.
            ("CAST(s AS DECIMAL(5, 2))", true),
            ("CAST(l_quantity AS DECIMAL(5, 2))", true),
            ("CAST(s AS TIMESTAMP)", true),
            // Arithmetic on dates and intervals.
            ("o_orderdate + INTERVAL '1 day'", false),
            ("ts - INTERVAL '1 month'", false),
            // Decimal arithmetic, which can only overflow, and its parts.
            ("l_extendedprice * (1 - l_discount) * (1 + l_tax)", false),
        ] {
            let expr = physical_expr(expr, &schema);
            assert_eq!(depends(&expr, &schema), expected, "{expr}");
            assert_eq!(failures(&expr, &schema).order_matters(), expected, "{expr}");
        }
    }

    #[test]
    fn bounded_constants_do_not_depend_on_batches() {
        let schema = tpch_schema();
        let large = "x".repeat(MAX_CONSTANT_BYTES + 1);
        // Whether the errors depend on batches of at most `ROWS` rows, and on
        // batches of unknown sizes.
        for (expr, expected) in [
            (
                "CASE WHEN o_orderpriority = '1-URGENT' OR o_orderpriority = '2-HIGH' \
                 THEN '1-URGENT' ELSE '2-HIGH' END"
                    .to_owned(),
                (false, true),
            ),
            (
                "CASE WHEN n > 0 THEN 'HIGH' WHEN n < 0 THEN 'LOW' END".to_owned(),
                (false, true),
            ),
            (
                "CASE WHEN n > 0 THEN 'HIGH' END = 'HIGH'".to_owned(),
                (false, true),
            ),
            ("'1-URGENT'".to_owned(), (false, true)),
            (format!("'{}'", &large[1..]), (false, true)),
            // Constants of more bytes may overflow the 32-bit offsets of a
            // batch of DataFusion, but not of a chunk.
            (
                format!("CASE WHEN n > 0 THEN '{large}' ELSE 'LOW' END"),
                (true, true),
            ),
            (format!("'{large}'"), (true, true)),
            // A column and a constant build strings of more bytes than the
            // column has.
            (
                "CASE WHEN n > 0 THEN s ELSE 'LOW' END".to_owned(),
                (true, true),
            ),
            ("CASE WHEN n > 0 THEN s END".to_owned(), (false, false)),
            // Constants without bytes, and compared ones, are not copied.
            ("CASE WHEN n > 0 THEN '' END".to_owned(), (false, false)),
            ("o_orderpriority = '1-URGENT'".to_owned(), (false, false)),
        ] {
            let expr = physical_expr(&expr, &schema);
            assert_eq!(
                (depends(&expr, &schema), depends_on_batches(&expr, &schema)),
                expected,
                "{expr}"
            );
        }
    }

    #[test]
    fn constants_need_batches_of_known_sizes() {
        // The bytes of the constants that batches of at most so many rows
        // hold for each of their rows, and the chunks of the pipeline.
        for (rows, bytes) in [
            (None, 0),
            (Some(0), MAX_CONSTANT_BYTES),
            (ROWS, MAX_CONSTANT_BYTES),
            (Some((1 << 21) - 1), MAX_CONSTANT_BYTES),
            (Some(1 << 21), MAX_CONSTANT_BYTES - 1),
            (Some(1 << 30), 1),
            (Some(usize::MAX), 0),
        ] {
            assert_eq!(constant_bytes(rows), bytes, "{rows:?}");
        }
        // Built-in functions copy their constant arguments for every row of
        // the batch (e.g. `strpos` pads them, see `make_scalar_function`),
        // but for the parameters that they read as scalars.
        let schema = tpch_schema();
        for (expr, depends_on) in [
            ("extract(year from o_orderdate)", false),
            ("date_trunc('month', ts)", false),
            ("to_timestamp(s, '%Y-%m-%d', '%d/%m/%Y')", false),
            ("ltrim(c_phone, '1')", false),
            ("nullif(c_phone, '10')", false),
            ("starts_with(s, 'abc')", false),
            ("ends_with('abc', s)", false),
            ("regexp_like(s, '^a', 'i')", false),
            ("strpos(c_phone, '1')", true),
            ("nullif('10', s)", true),
            // Of views, which have no offsets to overflow.
            ("nullif('10', c_phone)", false),
            ("substr('abc', n)", true),
            ("regexp_like(s, c_phone, 'i')", true),
            ("strpos(c_phone, '')", false),
        ] {
            let expr = physical_expr(expr, &schema);
            assert_eq!(depends_on_batches(&expr, &schema), depends_on, "{expr}");
        }
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
