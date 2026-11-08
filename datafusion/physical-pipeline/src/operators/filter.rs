// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`PhysicalFilter`]: selection-vector based filtering.
//!
//! The native counterpart of DataFusion's `FilterExec`. Instead of copying
//! the selected rows of every column (`filter_record_batch`), the filter
//! attaches a [`SelectionVector`] to its input chunk:
//!
//! 1. The predicate is split into its conjuncts by an [`AdaptiveFilter`],
//!    which also decides the order in which they are evaluated.
//! 2. The conjuncts are evaluated one after the other, each on the columns
//!    it references. A conjunct rejects the rows for which it is `false`. A
//!    row is selected if every conjunct is `true` for it: a `NULL` row is not
//!    selected, but, as in DataFusion, it is not rejected either, so the
//!    following conjuncts are still evaluated on it.
//! 3. As soon as every row is rejected, the remaining conjuncts are skipped
//!    and no output is produced.
//! 4. The output chunk shares the columns of the input (`Arc` clones, after
//!    applying the projection embedded in the `FilterExec`, if any) and
//!    carries the selected rows as its selection. No column is copied: the
//!    columns that are not referenced by the predicate are never touched.
//!
//! # The rows a conjunct is evaluated on
//!
//! A conjunct that can fail (see
//! [`fallibility`](super::adaptive_filter::fallibility)), e.g.
//! `10 / x > 1`, is evaluated on exactly the rows on which DataFusion
//! evaluates it (`BinaryExpr::evaluate_conjunction`), so that the filter
//! fails exactly when `FilterExec` fails on the same rows:
//!
//! * on the rows that the conjuncts before it did not reject, if they
//!   rejected some rows and at most 20% of the rows of the chunk are left
//!   (DataFusion's `PRE_SELECTION_THRESHOLD`): `x <> 0 AND 10 / x > 1` does
//!   not divide by zero if at least 80% of the rows have `x = 0`;
//! * on all rows of the chunk otherwise, including the rejected ones:
//!   `x <> 0 AND 10 / x > 1` divides by zero if some, but less than 80%, of
//!   the rows have `x = 0`.
//!
//! The [`AdaptiveFilter`] never moves a conjunct across a fallible one, so a
//! fallible conjunct always follows the same conjuncts as in DataFusion.
//!
//! A conjunct that cannot fail is evaluated on the rows that were not
//! rejected, or, if more than a quarter of the physical rows are left, on
//! all physical rows, which avoids copying its columns; its result for the
//! other rows is ignored.
//!
//! Only the columns a conjunct references are compacted (`take`) to the rows
//! it is evaluated on. Since `p AND q` is `true` exactly when both `p` and
//! `q` are `true`, the selected rows are the rows for which the predicate is
//! `true`, as in DataFusion.
//!
//! # Lowering a `FilterExec`
//!
//! DataFusion applies its 20% threshold to the batches it filters, the
//! filter to its chunks, which are not DataFusion's batches (sources split
//! batches into chunks, operators coalesce their output into batches, and
//! pipelines skip empty chunks, while `FilterExec` also evaluates its
//! predicate on empty batches). So [`PhysicalFilter::try_from_filter_exec`]
//! only lowers predicates whose errors do not depend on the batches (see
//! [`depends_on_batches`]): for the others, e.g. `x <> 0 AND 10 / x > 1`,
//! `x = 0 OR 10 / x > 1` or `x = 0 OR s ~ 'a{1000000}'` (whose pattern does
//! not compile), the `FilterExec` must be executed by DataFusion, on the
//! batches of its input plan. For the same reason, a `FilterExec` with a
//! `fetch`, which stops after the batch in which the fetch is reached, is
//! only lowered if its predicate cannot fail at all (and not with a fetch of
//! 0), and a `FilterExec` whose output has dictionaries with narrow keys,
//! which it may fail to merge when it coalesces its output batches, is not
//! lowered. A planner must also keep the plans below a limit that stops the
//! execution early in DataFusion, see
//! [`PhysicalFilter::supports_filter_exec`].

use std::borrow::Cow;
use std::fmt::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow::array::{Array, AsArray, RecordBatch};
use arrow::datatypes::{DataType, Field, SchemaRef};
use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder};
use datafusion_common::instant::Instant;
use datafusion_common::{Result, ScalarValue, internal_err, not_impl_err, plan_err};
use datafusion_expr::ColumnarValue;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_plan::filter::FilterExec;
use datafusion_physical_plan::{ExecutionPlan, ExecutionPlanProperties};
use itertools::Itertools;

use super::adaptive_filter::{
    AdaptiveFilter, AdaptiveFilterState, depends_on_batches, is_infallible,
};
use super::projection::{BoundExprs, can_gather, check_lowered_input, logical_rows};
use crate::chunk::{DataChunk, SelectionVector};
use crate::operator::{
    ExecutionContext, GlobalOperatorState, OperatorResult, OperatorState,
    PhysicalOperator, downcast_state, downcast_state_mut,
};

/// DataFusion's `PRE_SELECTION_THRESHOLD` (in `BinaryExpr`): before a
/// conjunct that may fail is evaluated, the rows that the conjuncts before
/// it rejected are only removed if at most this fraction of the rows is
/// left.
const PRE_SELECTION_THRESHOLD: f32 = 0.2;

/// An infallible conjunct is evaluated on all physical rows of a chunk,
/// instead of only on the rows that are not rejected, if more than one in
/// this many rows are left: gathering the columns of the rows (`take`)
/// costs more per row than evaluating a cheap conjunct.
const DENSE_SELECTION: usize = 4;

/// Filters the [`DataChunk`]s of its input with a predicate by attaching a
/// [`SelectionVector`], like DataFusion's `FilterExec`.
///
/// See the [module documentation](self).
#[derive(Debug)]
pub struct PhysicalFilter {
    children: Vec<Arc<dyn PhysicalOperator>>,
    predicate: Arc<dyn PhysicalExpr>,
    /// The conjuncts of `predicate` and their evaluation orders.
    filter: AdaptiveFilter,
    /// The conjuncts of `filter`, bound to the input columns they reference.
    conjuncts: Vec<BoundExprs>,
    /// The position of the last fallible conjunct (fallible conjuncts never
    /// move).
    last_fallible: Option<usize>,
    /// The input columns of the output, if not all of them.
    projection: Option<Vec<usize>>,
    /// The maximum number of rows to produce.
    fetch: Option<usize>,
    schema: SchemaRef,
}

impl PhysicalFilter {
    /// Create a filter that selects the rows of the chunks of `input` for
    /// which `predicate` is `true`, and outputs the input columns
    /// `projection` (all of them if `None`).
    pub fn try_new(
        input: Arc<dyn PhysicalOperator>,
        predicate: Arc<dyn PhysicalExpr>,
        projection: Option<Vec<usize>>,
    ) -> Result<Self> {
        let input_schema = input.schema();
        let schema = match &projection {
            Some(projection) => Arc::new(input_schema.project(projection)?),
            None => Arc::clone(&input_schema),
        };
        Self::try_new_with_schema(input, predicate, projection, schema)
    }

    /// Whether [`Self::try_from_filter_exec`] can lower `exec` with exactly
    /// its semantics, including its errors. It cannot if
    ///
    /// * the errors of the predicate depend on how DataFusion splits the
    ///   rows into batches, empty ones included (see [`depends_on_batches`]),
    ///   e.g. `x <> 0 AND 10 / x > 1` or `x = 0 OR s ~ 'a{1000000}'`:
    ///   DataFusion must evaluate it on the batches of the input plan of
    ///   `exec`;
    /// * the input of `exec` has lists of values that take no memory (e.g.
    ///   fixed-size lists of `1 << 30` nulls), which the native operators
    ///   cannot gather with `take` (see `can_gather`);
    /// * the output of `exec` has dictionaries with keys of less than 32 bits
    ///   (or run ends of less than 32 bits), or lists with 32-bit offsets, at
    ///   any depth: `FilterExec` coalesces its output batches, merging the
    ///   dictionaries of distinct batches, which fails if they have more values
    ///   than the keys can index (e.g. two batches of 100 distinct values, with
    ///   `Int8` keys), and adding up the offsets of lists, which fails beyond
    ///   `i32::MAX` values (e.g. two lists of `1 << 30` nulls). Whether it
    ///   fails depends on the batches of the input plan, and the native filter
    ///   does not coalesce its output;
    /// * `exec` has a `fetch` and its input has several partitions: the
    ///   fetch limits every partition, but the chunks of a pipeline do not
    ///   belong to partitions;
    /// * `exec` has a `fetch` and its predicate can fail: `FilterExec`
    ///   evaluates the predicate on whole batches until the fetch is reached,
    ///   so it may fail on rows the native filter, which stops earlier, never
    ///   sees;
    /// * `exec` has a `fetch` of 0: `FilterExec` evaluates its predicate on
    ///   the first batch of its input, even an empty one, and stops, but
    ///   pipelines skip empty chunks, so the native filter would pull more of
    ///   its input (whose errors it would raise).
    ///
    /// A planner must also keep the plans below a limit that stops the
    /// execution early (e.g. the input of a `GlobalLimitExec`, of a node with
    /// a fetch, or of an aggregation with a soft limit) in DataFusion, filters
    /// included, unless a sort that reads all of its input is in between: a
    /// sort whose order its input does not satisfy, without a fetch or with
    /// a fetch that shares no prefix with the order of its input (a sort that
    /// its input satisfies streams it, and such a TopK may stop early). DataFusion pulls its input lazily, batch by batch, but for
    /// `FilterExec`, which pulls ahead to coalesce its output, so the rows that
    /// are evaluated before the limit is reached, and so the errors, depend on
    /// these pulls, which a pipeline (on several threads, with a cache of
    /// small chunks) does not reproduce.
    ///
    /// Likewise, a planner must keep a filter or a projection with a part that
    /// may fail (see
    /// [`failing_parts`](super::adaptive_filter::failing_parts)) in DataFusion,
    /// with its input plan, if native operators below it in the same pipeline
    /// have parts that may fail too: DataFusion evaluates each operator on a
    /// whole batch before the next one, a pipeline each chunk with all of its
    /// operators, so which error comes first depends on the batches.
    pub fn supports_filter_exec(exec: &FilterExec) -> bool {
        unsupported_filter_exec(exec).is_none()
    }

    /// Lower a DataFusion [`FilterExec`]: apply its predicate, projection and
    /// fetch to the chunks of `input` (the lowered input of `exec`) and
    /// produce exactly its output schema.
    ///
    /// Fails if `exec` is not supported, see [`Self::supports_filter_exec`].
    pub fn try_from_filter_exec(
        exec: &FilterExec,
        input: Arc<dyn PhysicalOperator>,
    ) -> Result<Self> {
        if let Some(reason) = unsupported_filter_exec(exec) {
            return not_impl_err!(
                "PhysicalFilter cannot lower this FilterExec: {reason}"
            );
        }
        check_lowered_input(exec.input().schema().as_ref(), input.as_ref())?;
        let projection = exec
            .projection()
            .as_ref()
            .map(|projection| projection.to_vec());
        Ok(Self::try_new_with_schema(
            input,
            Arc::clone(exec.predicate()),
            projection,
            exec.schema(),
        )?
        .with_fetch(exec.fetch()))
    }

    fn try_new_with_schema(
        input: Arc<dyn PhysicalOperator>,
        predicate: Arc<dyn PhysicalExpr>,
        projection: Option<Vec<usize>>,
        schema: SchemaRef,
    ) -> Result<Self> {
        let input_schema = input.schema();
        match predicate.data_type(input_schema.as_ref())? {
            DataType::Boolean => {}
            other => {
                return plan_err!(
                    "Filter predicate must return BOOLEAN values, got {other:?}"
                );
            }
        }
        let projected = match &projection {
            Some(projection) => input_schema.project(projection)?,
            None => input_schema.as_ref().clone(),
        };
        let same_types = projected.fields().len() == schema.fields().len()
            && projected
                .fields()
                .iter()
                .zip(schema.fields())
                .all(|(projected, field)| projected.data_type() == field.data_type());
        if !same_types {
            return internal_err!(
                "The schema of PhysicalFilter does not match its projected input: {schema:?} vs {projected:?}"
            );
        }
        let filter = AdaptiveFilter::try_new(&predicate, input_schema.as_ref())?;
        let conjuncts = filter
            .conjuncts()
            .iter()
            .map(|conjunct| {
                BoundExprs::try_new([Arc::clone(conjunct.expr())], input_schema.as_ref())
            })
            .collect::<Result<Vec<_>>>()?;
        let last_fallible = filter
            .conjuncts()
            .iter()
            .rposition(|conjunct| conjunct.is_fallible());
        Ok(Self {
            children: vec![input],
            predicate,
            filter,
            conjuncts,
            last_fallible,
            projection,
            fetch: None,
            schema,
        })
    }

    /// Produce at most `fetch` rows: the first `fetch` rows of the input that
    /// pass the filter, then return [`OperatorResult::Finished`]. Like a
    /// `FilterExec` with a single input partition, a filter with a fetch is
    /// executed by a single thread (see [`PhysicalOperator::parallel_operator`]),
    /// so that it filters its input in order. It stops as soon as the fetch
    /// is reached, so if its predicate can fail, it may not fail where
    /// `FilterExec`, which evaluates whole batches, fails (see
    /// [`Self::supports_filter_exec`]).
    pub fn with_fetch(mut self, fetch: Option<usize>) -> Self {
        self.fetch = fetch;
        self
    }

    /// The filter predicate.
    pub fn predicate(&self) -> &Arc<dyn PhysicalExpr> {
        &self.predicate
    }

    /// The conjuncts of the predicate.
    pub fn adaptive_filter(&self) -> &AdaptiveFilter {
        &self.filter
    }

    /// The input columns of the output, if not all of them.
    pub fn projection(&self) -> Option<&[usize]> {
        self.projection.as_deref()
    }

    /// The maximum number of rows the filter produces.
    pub fn fetch(&self) -> Option<usize> {
        self.fetch
    }

    /// Whether the errors of the predicate depend on how the rows are split
    /// into batches (see [`depends_on_batches`]), in which case the filter
    /// may fail on other inputs than `FilterExec`, which evaluates it on
    /// other batches.
    pub fn errors_depend_on_batches(&self) -> bool {
        depends_on_batches(&self.predicate, self.children[0].schema().as_ref())
    }
}

/// Why `exec` cannot be lowered with exactly its semantics, see
/// [`PhysicalFilter::supports_filter_exec`].
fn unsupported_filter_exec(exec: &FilterExec) -> Option<String> {
    let input_schema = exec.input().schema();
    if depends_on_batches(exec.predicate(), input_schema.as_ref()) {
        return Some(format!(
            "the errors of its predicate {} depend on DataFusion's batches",
            exec.predicate()
        ));
    }
    if let Some(field) = input_schema
        .fields()
        .iter()
        .find(|field| !can_gather(field.data_type()))
    {
        return Some(format!(
            "its input column '{}' has lists of values that take no memory, which the \
             native operators cannot gather",
            field.name()
        ));
    }
    if let Some(field) = exec
        .schema()
        .fields()
        .iter()
        .find(|field| may_fail_to_coalesce(field.data_type()))
    {
        return Some(format!(
            "it may fail to coalesce the dictionaries, run ends or lists of its output column '{}'",
            field.name()
        ));
    }
    if exec.fetch()? == 0 {
        return Some(
            "it has a fetch of 0: `FilterExec` evaluates its predicate on the first \
             batch of its input, even an empty one, and stops, but pipelines skip \
             empty chunks"
                .to_owned(),
        );
    }
    let partitions = exec.input().output_partitioning().partition_count();
    if partitions > 1 {
        return Some(format!(
            "it limits each of its {partitions} input partitions to its fetch"
        ));
    }
    if !is_infallible(exec.predicate(), input_schema.as_ref()) {
        return Some(format!(
            "it has a fetch and its predicate {} can fail",
            exec.predicate()
        ));
    }
    None
}

/// Whether `FilterExec` may fail to coalesce batches of `data_type`, at any
/// depth: it concatenates the arrays of the batches it coalesces, which
/// merges the dictionaries of distinct batches, and fails when their keys
/// cannot index all of their values (keys of less than 32 bits), adds up run
/// ends (run ends of less than 32 bits), and the offsets of lists (32-bit
/// offsets, which lists of nulls overflow without memory). The 32-bit offsets
/// of strings and binaries overflow too, but on more than 2 GiB of data. And
/// Arrow's `filter` rebuilds run-end encoded arrays with its own child
/// fields, so their fields must be Arrow's.
fn may_fail_to_coalesce(data_type: &DataType) -> bool {
    let narrow = |index: &DataType| {
        !matches!(
            index,
            DataType::Int32 | DataType::Int64 | DataType::UInt32 | DataType::UInt64
        )
    };
    match data_type {
        DataType::Dictionary(keys, values) => {
            narrow(keys) || may_fail_to_coalesce(values)
        }
        // Arrow's `filter` rebuilds run-end encoded arrays with its own fields
        // (see `RunArray::try_new`), so the output would not have its type.
        DataType::RunEndEncoded(run_ends, values) => {
            narrow(run_ends.data_type())
                || **run_ends
                    != Field::new("run_ends", run_ends.data_type().clone(), false)
                || **values != Field::new("values", values.data_type().clone(), true)
                || may_fail_to_coalesce(values.data_type())
        }
        // Their 32-bit offsets overflow on more than `i32::MAX` values, which
        // need no memory if they are nulls.
        DataType::List(_) | DataType::ListView(_) | DataType::Map(..) => true,
        DataType::LargeList(field)
        | DataType::LargeListView(field)
        | DataType::FixedSizeList(field, _) => may_fail_to_coalesce(field.data_type()),
        DataType::Struct(fields) => fields
            .iter()
            .any(|field| may_fail_to_coalesce(field.data_type())),
        DataType::Union(fields, _) => fields
            .iter()
            .any(|(_, field)| may_fail_to_coalesce(field.data_type())),
        _ => false,
    }
}

/// The global state of a [`PhysicalFilter`]: the rows it may still produce.
#[derive(Debug)]
struct FilterGlobalState {
    /// `None` without a fetch.
    remaining: Option<AtomicUsize>,
}

impl GlobalOperatorState for FilterGlobalState {}

impl FilterGlobalState {
    /// Whether the filter produced all the rows of its fetch.
    fn is_exhausted(&self) -> bool {
        self.remaining
            .as_ref()
            .is_some_and(|remaining| remaining.load(Ordering::Acquire) == 0)
    }

    /// Claim up to `rows` rows of the fetch. Returns the number of rows that
    /// may be produced and whether the fetch is exhausted.
    fn claim(&self, rows: usize) -> (usize, bool) {
        let Some(remaining) = &self.remaining else {
            return (rows, false);
        };
        let mut available = remaining.load(Ordering::Acquire);
        loop {
            let claimed = rows.min(available);
            match remaining.compare_exchange_weak(
                available,
                available - claimed,
                Ordering::AcqRel,
                Ordering::Acquire,
            ) {
                Ok(_) => return (claimed, claimed == available),
                Err(actual) => available = actual,
            }
        }
    }
}

/// The state of a [`PhysicalFilter`] in one thread.
#[derive(Debug)]
struct FilterState {
    adaptive: AdaptiveFilterState,
}

impl OperatorState for FilterState {}

/// The rows of an input chunk that pass the filter.
enum Selected {
    /// No row passes.
    Nothing,
    /// Every (logical) row passes.
    All,
    /// The physical rows that pass, a subset of the logical rows.
    Rows(SelectionVector),
}

/// The result of a conjunct for the rows it was evaluated on.
enum Mask {
    /// Every row is `true`.
    True,
    /// Every row is `false`.
    False,
    /// Every row is `NULL`.
    Null,
    /// The rows that are not `false` (`true` or `NULL`), and those that are
    /// `NULL` (`None` if none is).
    Mixed {
        not_false: BooleanBuffer,
        null: Option<BooleanBuffer>,
    },
}

/// The rows of a [`Mask`]: how the rows the conjunct was evaluated on relate
/// to the logical rows of the chunk.
enum MaskRows<'a> {
    /// The rows that are not rejected, in order.
    Undecided,
    /// All logical rows.
    Logical,
    /// All physical rows; the logical rows are `incoming` (all physical rows
    /// if `None`).
    Physical(Option<&'a SelectionVector>),
}

/// The logical rows of a chunk that the conjuncts evaluated so far did not
/// reject, as bitmaps over the logical rows.
struct Undecided {
    /// The number of logical rows of the chunk.
    num_rows: usize,
    /// The rows that are not rejected (`None` for all logical rows).
    rows: Option<BooleanBuffer>,
    /// The rows of `rows` for which a conjunct was `NULL` (`None` if none
    /// is). Such rows are not selected, but the next conjuncts are still
    /// evaluated on them.
    nulls: Option<BooleanBuffer>,
    /// The number of rows of `rows`.
    count: usize,
    /// The number of rows of `nulls`.
    null_count: usize,
}

impl Undecided {
    fn all(num_rows: usize) -> Self {
        Self {
            num_rows,
            rows: None,
            nulls: None,
            count: num_rows,
            null_count: 0,
        }
    }

    /// The number of rows that are not rejected.
    fn len(&self) -> usize {
        self.count
    }

    /// The number of rows for which every conjunct so far was `true`.
    fn true_count(&self) -> usize {
        self.count - self.null_count
    }

    /// The physical rows that are not rejected (`None` for all physical
    /// rows), given the physical rows of the logical rows, `incoming`.
    fn physical_rows<'a>(
        &self,
        incoming: Option<&'a SelectionVector>,
    ) -> Option<Cow<'a, SelectionVector>> {
        let Some(rows) = &self.rows else {
            return incoming.map(Cow::Borrowed);
        };
        let mut physical = Vec::with_capacity(self.count);
        match incoming {
            Some(incoming) => {
                let incoming = incoming.indices();
                physical.extend(rows.set_indices().map(|row| incoming[row]));
            }
            None => physical.extend(rows.set_indices_u32()),
        }
        Some(Cow::Owned(SelectionVector::from(physical)))
    }

    /// Reject the rows for which the conjunct with result `mask` is `false`,
    /// and record those for which it is `NULL`.
    fn update(&mut self, mask: Mask, mask_rows: &MaskRows<'_>) {
        let (not_false, null) = match mask {
            Mask::True => return,
            Mask::False => {
                self.rows = Some(BooleanBuffer::new_unset(self.num_rows));
                self.nulls = None;
                self.count = 0;
                self.null_count = 0;
                return;
            }
            Mask::Null => {
                self.nulls = Some(
                    self.rows
                        .clone()
                        .unwrap_or_else(|| BooleanBuffer::new_set(self.num_rows)),
                );
                self.null_count = self.count;
                return;
            }
            Mask::Mixed { not_false, null } => (not_false, null),
        };
        // The mask, over the logical rows.
        let (not_false, null) = match mask_rows {
            MaskRows::Logical | MaskRows::Physical(None) => (not_false, null),
            MaskRows::Undecided => match &self.rows {
                None => (not_false, null),
                // Scatter the results to the rows they were computed for.
                Some(rows) => (
                    scatter(&not_false, rows, self.num_rows),
                    null.map(|null| scatter(&null, rows, self.num_rows)),
                ),
            },
            MaskRows::Physical(Some(incoming)) => (
                gather(&not_false, incoming),
                null.map(|null| gather(&null, incoming)),
            ),
        };
        let rows = match &self.rows {
            Some(rows) => rows & &not_false,
            None => not_false,
        };
        let nulls = match (&self.nulls, null) {
            (None, None) => None,
            (Some(nulls), None) => Some(nulls & &rows),
            (None, Some(null)) => Some(&null & &rows),
            (Some(nulls), Some(null)) => Some(&(nulls | &null) & &rows),
        };
        self.count = rows.count_set_bits();
        self.null_count = nulls.as_ref().map_or(0, BooleanBuffer::count_set_bits);
        self.nulls = nulls.filter(|_| self.null_count > 0);
        self.rows = Some(rows);
    }

    /// The selected rows: those for which every conjunct was `true`.
    fn into_selected(self, incoming: Option<&SelectionVector>) -> Selected {
        let true_count = self.true_count();
        if true_count == 0 {
            return Selected::Nothing;
        }
        let selected = match (self.rows, self.nulls) {
            (None, None) => return Selected::All,
            (Some(rows), None) => rows,
            (rows, Some(nulls)) => {
                let rows = rows.unwrap_or_else(|| BooleanBuffer::new_set(self.num_rows));
                &rows & &!&nulls
            }
        };
        let mut physical = Vec::with_capacity(true_count);
        match incoming {
            Some(incoming) => {
                let incoming = incoming.indices();
                physical.extend(selected.set_indices().map(|row| incoming[row]));
            }
            None => physical.extend(selected.set_indices_u32()),
        }
        Selected::Rows(SelectionVector::from(physical))
    }
}

/// The bitmap over `num_rows` rows whose bit for the `i`-th set bit of
/// `rows` is the `i`-th bit of `values`.
fn scatter(
    values: &BooleanBuffer,
    rows: &BooleanBuffer,
    num_rows: usize,
) -> BooleanBuffer {
    let mut builder = BooleanBufferBuilder::new(num_rows);
    builder.append_n(num_rows, false);
    for (value, row) in values.iter().zip(rows.set_indices()) {
        if value {
            builder.set_bit(row, true);
        }
    }
    builder.finish()
}

/// The bitmap whose `i`-th bit is the bit `rows[i]` of `values`.
fn gather(values: &BooleanBuffer, rows: &SelectionVector) -> BooleanBuffer {
    BooleanBuffer::from_iter(rows.iter().map(|row| values.value(row)))
}

impl PhysicalOperator for PhysicalFilter {
    fn name(&self) -> &str {
        "PhysicalFilter"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.children
    }

    fn description(&self) -> String {
        let mut description = format!("PhysicalFilter: {}", self.predicate);
        if let Some(projection) = &self.projection {
            let input_schema = self.children[0].schema();
            let _ = write!(
                description,
                ", projection=[{}]",
                projection
                    .iter()
                    .map(|&index| format!("{}@{index}", input_schema.field(index).name()))
                    .join(", ")
            );
        }
        if let Some(fetch) = self.fetch {
            let _ = write!(description, ", fetch={fetch}");
        }
        description
    }

    /// A filter with a fetch is executed by a single thread, which filters
    /// its input in order and produces its first `fetch` rows, as
    /// `FilterExec` does for a single partition.
    fn parallel_operator(&self) -> bool {
        self.fetch.is_none()
    }

    fn get_global_operator_state(
        &self,
        _ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalOperatorState>> {
        Ok(Arc::new(FilterGlobalState {
            remaining: self.fetch.map(AtomicUsize::new),
        }))
    }

    fn get_operator_state(
        &self,
        _ctx: &ExecutionContext,
        _gstate: &dyn GlobalOperatorState,
    ) -> Result<Box<dyn OperatorState>> {
        Ok(Box::new(FilterState {
            adaptive: self.filter.create_state(),
        }))
    }

    fn execute(
        &self,
        _ctx: &ExecutionContext,
        input: &DataChunk,
        output: &mut DataChunk,
        gstate: &dyn GlobalOperatorState,
        lstate: &mut dyn OperatorState,
    ) -> Result<OperatorResult> {
        let gstate = downcast_state::<FilterGlobalState>(gstate.as_any(), self.name())?;
        let state = downcast_state_mut::<FilterState>(lstate.as_any_mut(), self.name())?;
        if gstate.is_exhausted() {
            return Ok(OperatorResult::Finished);
        }
        let (selection, num_rows) = match self.select(input, &mut state.adaptive)? {
            // Short circuit: the output stays empty.
            Selected::Nothing => return Ok(OperatorResult::NeedMoreInput),
            Selected::All => (logical_rows(input).cloned(), input.len()),
            Selected::Rows(rows) => {
                let num_rows = rows.len();
                (Some(rows), num_rows)
            }
        };
        let (num_rows, exhausted) = gstate.claim(num_rows);
        if num_rows > 0 {
            *output = self.output_chunk(input, selection, num_rows)?;
        }
        Ok(if exhausted {
            OperatorResult::Finished
        } else {
            OperatorResult::NeedMoreInput
        })
    }
}

impl PhysicalFilter {
    /// Evaluate the conjuncts on the (non-empty) chunk `input`, in the order
    /// of `state`, and record their statistics.
    fn select(
        &self,
        input: &DataChunk,
        state: &mut AdaptiveFilterState,
    ) -> Result<Selected> {
        let incoming = logical_rows(input);
        let mut undecided = Undecided::all(input.len());
        let timed = state.is_adaptive();
        for position in 0..self.conjuncts.len() {
            if undecided.len() == 0 {
                // Every row is rejected: DataFusion stops here too.
                break;
            }
            if undecided.true_count() == 0
                && self.last_fallible.is_none_or(|last| position > last)
            {
                // No row can pass, and the remaining conjuncts cannot fail.
                break;
            }
            let conjunct = state.permutation()[position];
            let input_rows = undecided.len();
            let start = timed.then(Instant::now);
            self.evaluate_conjunct(conjunct, input, incoming, &mut undecided)?;
            if let Some(start) = start {
                state.record(conjunct, input_rows, undecided.len(), start.elapsed());
            }
        }
        state.end_chunk();
        Ok(undecided.into_selected(incoming))
    }

    /// Evaluate conjunct `conjunct` on `input` (whose logical rows are
    /// `incoming`, all physical rows if `None`), and reject the rows of
    /// `undecided` for which it is `false`.
    fn evaluate_conjunct(
        &self,
        conjunct: usize,
        input: &DataChunk,
        incoming: Option<&SelectionVector>,
        undecided: &mut Undecided,
    ) -> Result<()> {
        let bound = &self.conjuncts[conjunct];
        let expr = &bound.exprs()[0];
        let left = undecided.len();
        if self.filter.conjuncts()[conjunct].is_fallible() {
            // Evaluate the conjunct on the rows DataFusion evaluates it on:
            // only on the rows that are left if some rows were rejected and
            // at most 20% of them are left, otherwise on all rows.
            let num_rows = undecided.num_rows;
            let pre_select = left < num_rows
                && left as f32 / num_rows as f32 <= PRE_SELECTION_THRESHOLD;
            if !pre_select {
                let batch = bound.batch(input, incoming)?;
                undecided.update(evaluate_mask(expr, &batch)?, &MaskRows::Logical);
                return Ok(());
            }
        } else if left * DENSE_SELECTION > input.physical_len() {
            // Many rows are left and the conjunct cannot fail on the other
            // ones: evaluating it on all physical rows is cheaper than
            // gathering its columns.
            let batch = bound.batch(input, None)?;
            undecided.update(evaluate_mask(expr, &batch)?, &MaskRows::Physical(incoming));
            return Ok(());
        }
        let rows = undecided.physical_rows(incoming);
        let batch = bound.batch(input, rows.as_deref())?;
        undecided.update(evaluate_mask(expr, &batch)?, &MaskRows::Undecided);
        Ok(())
    }

    /// The output chunk: the (projected) columns of `input` with the
    /// selection `selection`, limited to `num_rows` rows.
    fn output_chunk(
        &self,
        input: &DataChunk,
        selection: Option<SelectionVector>,
        num_rows: usize,
    ) -> Result<DataChunk> {
        if self.schema.fields().is_empty() {
            return DataChunk::try_new_without_columns(
                Arc::clone(&self.schema),
                num_rows,
            );
        }
        let columns = match &self.projection {
            Some(projection) => projection
                .iter()
                .map(|&index| Arc::clone(input.column(index)))
                .collect(),
            None => input.columns().to_vec(),
        };
        let chunk = DataChunk::try_new(Arc::clone(&self.schema), columns, selection)?;
        Ok(if num_rows < chunk.len() {
            chunk.slice(0, num_rows)
        } else {
            chunk
        })
    }
}

/// Evaluate a conjunct on `batch`, with the semantics of `FilterExec`: the
/// result must be a boolean.
fn evaluate_mask(expr: &Arc<dyn PhysicalExpr>, batch: &RecordBatch) -> Result<Mask> {
    let num_rows = batch.num_rows();
    let array = match expr.evaluate(batch)? {
        ColumnarValue::Scalar(ScalarValue::Boolean(value)) => {
            return Ok(match value {
                Some(true) => Mask::True,
                Some(false) => Mask::False,
                None => Mask::Null,
            });
        }
        value => value.into_array(num_rows)?,
    };
    let Some(mask) = array.as_boolean_opt() else {
        return internal_err!("Cannot create filter_array from non-boolean predicates");
    };
    if mask.len() != num_rows {
        return internal_err!(
            "Filter predicate returned {} values for {num_rows} rows",
            mask.len()
        );
    }
    let null = mask
        .nulls()
        .filter(|nulls| nulls.null_count() > 0)
        .map(|nulls| !nulls.inner());
    Ok(match null {
        None => match mask.values().count_set_bits() {
            0 => Mask::False,
            count if count == num_rows => Mask::True,
            _ => Mask::Mixed {
                not_false: mask.values().clone(),
                null: None,
            },
        },
        Some(null) => {
            if mask.null_count() == num_rows {
                return Ok(Mask::Null);
            }
            Mask::Mixed {
                not_false: mask.values() | &null,
                null: Some(null),
            }
        }
    })
}

/// Test support shared by the tests of the native streaming operators.
///
/// The planner does not lower any node natively yet, so the
/// differential tests of these operators lower the plans themselves:
/// [`lower`] replaces every `FilterExec` with a [`PhysicalFilter`] wrapped
/// in a [`CachingOperator`](super::caching::CachingOperator) and every
/// `ProjectionExec` with a [`PhysicalProjection`](super::projection::PhysicalProjection),
/// and executes all other nodes with DataFusion.
#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::Arc;

    use arrow::array::{
        ArrayRef, DictionaryArray, Int32Array, Int64Array, RecordBatch, StringArray,
        StringViewArray,
    };
    use arrow::datatypes::{DataType, Field, Int8Type, Int32Type, Schema, SchemaRef};
    use datafusion::datasource::MemTable;
    use datafusion::prelude::{SessionConfig, SessionContext};
    use datafusion_common::tree_node::TreeNodeRecursion;
    use datafusion_common::{DFSchema, DataFusionError, Result};
    use datafusion_datasource::memory::MemorySourceConfig;
    use datafusion_execution::TaskContext;
    use datafusion_expr::{
        ColumnarValue, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
        Volatility,
    };
    use datafusion_physical_expr::expressions::Column;
    use datafusion_physical_expr::{PhysicalExpr, ScalarFunctionExpr};
    use datafusion_physical_plan::aggregates::{AggregateExec, AggregateInputMode};
    use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
    use datafusion_physical_plan::filter::FilterExec;
    use datafusion_physical_plan::limit::LocalLimitExec;
    use datafusion_physical_plan::projection::ProjectionExec;
    use datafusion_physical_plan::repartition::RepartitionExec;
    use datafusion_physical_plan::sorts::sort::SortExec;
    use datafusion_physical_plan::{
        ChildrenPropertiesMode, Distribution, ExecutionPlan, ExecutionPlanProperties,
        Partitioning, ReplaceChildrenOptions, collect,
    };

    use super::PhysicalFilter;
    use crate::bridge::{BridgeOperator, PipelineRootExec};
    use crate::chunk::DataChunk;
    use crate::operator::{
        ExecutionContext, GlobalOperatorState, OperatorFinalizeResult, OperatorResult,
        OperatorState, PhysicalOperator,
    };
    use crate::operators::adaptive_filter::{depends_on_batches, failing_parts};
    use crate::operators::caching::CachingOperator;
    use crate::operators::projection::PhysicalProjection;
    use crate::test_utils::{
        EngineOutput, compare_outputs, datafusion_context, has_top_level_order_by,
        register_test_tables, run_sql, test_session_config,
    };

    /// Lower `plan` into an operator tree that executes the `FilterExec`s
    /// and `ProjectionExec`s natively where this keeps DataFusion's results
    /// and errors exactly:
    ///
    /// * `FilterExec` becomes `CachingOperator(PhysicalFilter)` and
    ///   `ProjectionExec` becomes `PhysicalProjection`, if supported (see
    ///   [`PhysicalFilter::supports_filter_exec`] and
    ///   [`PhysicalProjection::supports_projection_exec`]).
    /// * Below a limit that stops the execution early (unless a sort that
    ///   reads all of its input, see [`reads_all_input`], is in between), the
    ///   whole plan is executed by DataFusion.
    /// * A `FilterExec` or a `ProjectionExec` that is not lowered, and a node
    ///   whose expressions may fail depending on DataFusion's batches, are
    ///   executed by DataFusion with their whole input plan, so that they see
    ///   DataFusion's batches.
    /// * Scans become the native sources of
    ///   [`try_create_scan_source`](crate::sources::try_create_scan_source).
    /// * `CoalescePartitionsExec` (without fetch) and round robin
    ///   `RepartitionExec`s above native operators are removed: they only
    ///   change the partitioning, which the pipeline engine does not need.
    /// * Every other node is executed by DataFusion (in a `BridgeOperator`),
    ///   see [`bridged_plan`].
    pub(crate) fn lower(
        plan: &Arc<dyn ExecutionPlan>,
    ) -> Result<Arc<dyn PhysicalOperator>> {
        lower_below(plan, false)
    }

    /// [`lower`] `plan`, which is below a limit that stops the execution
    /// early if `limited`.
    fn lower_below(
        plan: &Arc<dyn ExecutionPlan>,
        limited: bool,
    ) -> Result<Arc<dyn PhysicalOperator>> {
        if limited && !reads_all_input(plan) {
            // DataFusion pulls the input of a limit lazily, batch by batch,
            // but for `FilterExec`, which pulls ahead to coalesce its output:
            // the rows evaluated before the limit is reached, and so the
            // errors, depend on these pulls, which a pipeline (on several
            // threads, with a cache of small chunks) does not reproduce.
            return Ok(Arc::new(BridgeOperator::new(Arc::clone(plan))));
        }
        // DataFusion evaluates each operator on a whole batch before the next
        // one, a pipeline each chunk with all of its operators: if native
        // operators of a pipeline have several parts that may fail, which
        // error comes first depends on the batches.
        if let Some(filter) = plan.downcast_ref::<FilterExec>() {
            if PhysicalFilter::supports_filter_exec(filter) {
                let input = lower_below(filter.input(), filter.fetch().is_some())?;
                let own = failing_parts(filter.predicate(), &filter.input().schema());
                if own == 0 || native_failing_parts(&input) == 0 {
                    let filter: Arc<dyn PhysicalOperator> =
                        Arc::new(PhysicalFilter::try_from_filter_exec(filter, input)?);
                    return Ok(Arc::new(CachingOperator::new(filter)));
                }
            }
            return Ok(Arc::new(BridgeOperator::new(Arc::clone(plan))));
        }
        if let Some(projection) = plan.downcast_ref::<ProjectionExec>() {
            if PhysicalProjection::supports_projection_exec(projection) {
                let input = lower_below(projection.input(), false)?;
                let input_schema = projection.input().schema();
                let own = projection
                    .expr()
                    .iter()
                    .map(|expr| failing_parts(&expr.expr, &input_schema))
                    .sum::<usize>();
                if own == 0 || native_failing_parts(&input) == 0 {
                    return Ok(Arc::new(PhysicalProjection::try_from_projection_exec(
                        projection, input,
                    )?));
                }
            }
            return Ok(Arc::new(BridgeOperator::new(Arc::clone(plan))));
        }
        if let Some(scan) = crate::sources::try_create_scan_source(plan)? {
            return Ok(scan);
        }
        if has_batch_dependent_expressions(plan)? {
            return Ok(Arc::new(BridgeOperator::new(Arc::clone(plan))));
        }
        if only_repartitions(plan) {
            let input = lower_below(plan.children()[0], limited)?;
            if !input.is::<BridgeOperator>() {
                return Ok(input);
            }
        }
        Ok(Arc::new(BridgeOperator::new(bridged_plan(plan, limited)?)))
    }

    /// The parts that may fail (see [`failing_parts`]) of the native filters
    /// and projections of `operator`, which evaluate the same chunks one after
    /// the other, down to the operators that DataFusion or a source executes.
    fn native_failing_parts(operator: &Arc<dyn PhysicalOperator>) -> usize {
        let input_schema = || operator.children()[0].schema();
        let own = if let Some(filter) = operator.downcast_ref::<PhysicalFilter>() {
            failing_parts(filter.predicate(), &input_schema())
        } else if let Some(projection) = operator.downcast_ref::<PhysicalProjection>() {
            let input_schema = input_schema();
            projection
                .exprs()
                .iter()
                .map(|(expr, _)| failing_parts(expr, &input_schema))
                .sum()
        } else if operator.is::<CachingOperator>() {
            0
        } else {
            return 0;
        };
        operator
            .children()
            .iter()
            .map(native_failing_parts)
            .fold(own, usize::saturating_add)
    }

    /// Whether `plan` reads all of its input, even if a limit above it stops
    /// early: a sort whose order its input does not satisfy, without fetch,
    /// or with a fetch (a TopK) that shares no prefix with the order of its
    /// input. A sort that its input satisfies streams it, and a TopK can stop
    /// once the next rows of its input sort after its own (see
    /// `SortExec::execute`).
    fn reads_all_input(plan: &Arc<dyn ExecutionPlan>) -> bool {
        let Some(sort) = plan.downcast_ref::<SortExec>() else {
            return false;
        };
        let Ok((prefix, satisfied)) = sort
            .input()
            .equivalence_properties()
            .extract_common_sort_prefix(sort.expr().clone())
        else {
            return false;
        };
        !satisfied
            && match sort.fetch() {
                None => true,
                Some(fetch) => fetch > 0 && prefix.is_empty(),
            }
    }

    /// Whether some expression of `plan` may fail depending on the batches
    /// it is evaluated on, or several parts of its expressions may fail (see
    /// [`failing_parts`](crate::operators::adaptive_filter::failing_parts)):
    /// judged on the schema of the input of `plan` if it has a single child,
    /// and without schema otherwise, which is conservative (expressions that
    /// may fail then fail on empty batches).
    /// The aggregations that merge partial states evaluate no expression on
    /// their input (the arguments of their aggregate functions are those of
    /// the partial aggregations).
    fn has_batch_dependent_expressions(plan: &Arc<dyn ExecutionPlan>) -> Result<bool> {
        if plan
            .downcast_ref::<AggregateExec>()
            .is_some_and(|aggregate| {
                aggregate.mode().input_mode() == AggregateInputMode::Partial
            })
        {
            return Ok(false);
        }
        let schema = match plan.children().as_slice() {
            [child] => child.schema(),
            _ => Arc::new(Schema::empty()),
        };
        let mut batch_dependent = false;
        let mut parts = 0_usize;
        plan.apply_expressions(&mut |expr| {
            batch_dependent |= depends_on_batches(expr, &schema);
            parts = parts.saturating_add(failing_parts(expr, &schema));
            Ok(TreeNodeRecursion::Continue)
        })?;
        Ok(batch_dependent || parts > 1)
    }

    /// `plan`, to be executed by DataFusion, with its children lowered: a
    /// child that contains native operators is executed by a
    /// [`PipelineRootExec`], unless `plan` needs the partitioning or the
    /// order of the child (or limits every partition of the child) and the
    /// child has several partitions, which a `PipelineRootExec` (a single
    /// partition) does not keep. DataFusion then executes the child too, with
    /// its own children lowered.
    ///
    /// The children of a node with a fetch (or of an aggregation with a soft
    /// limit, which stops reading its input once it has enough groups) are
    /// below a limit that stops the execution early, unless the node reads
    /// all of its input (see [`reads_all_input`]).
    fn bridged_plan(
        plan: &Arc<dyn ExecutionPlan>,
        limited: bool,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let soft_limit = plan
            .downcast_ref::<AggregateExec>()
            .is_some_and(|aggregate| aggregate.limit_options().is_some());
        let limited =
            !reads_all_input(plan) && (limited || plan.fetch().is_some() || soft_limit);
        let children = plan.children();
        let new_children = children
            .iter()
            .enumerate()
            .map(|(index, child)| child_plan(plan, index, child, limited))
            .collect::<Result<Vec<_>>>()?;
        let unchanged = children
            .iter()
            .zip(&new_children)
            .all(|(child, new_child)| Arc::ptr_eq(child, new_child));
        if unchanged {
            return Ok(Arc::clone(plan));
        }
        Arc::clone(plan).replace_children(
            new_children,
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
    }

    /// Whether `plan` only changes the partitioning of its input.
    fn only_repartitions(plan: &Arc<dyn ExecutionPlan>) -> bool {
        if let Some(coalesce) = plan.downcast_ref::<CoalescePartitionsExec>() {
            return coalesce.fetch().is_none();
        }
        plan.downcast_ref::<RepartitionExec>()
            .is_some_and(|repartition| {
                matches!(repartition.partitioning(), Partitioning::RoundRobinBatch(_))
                    && !repartition.preserve_order()
            })
    }

    /// The plan that DataFusion executes as child `index` of `parent`.
    fn child_plan(
        parent: &Arc<dyn ExecutionPlan>,
        index: usize,
        child: &Arc<dyn ExecutionPlan>,
        limited: bool,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let lowered = lower_below(child, limited)?;
        if let Some(bridge) = lowered.downcast_ref::<BridgeOperator>() {
            return Ok(Arc::clone(bridge.plan()));
        }
        let single_partition = child.output_partitioning().partition_count() == 1;
        // A `FilterExec` with a fetch and a `LocalLimitExec` limit every
        // partition of their input.
        let limits_partitions = (parent.is::<FilterExec>() && parent.fetch().is_some())
            || parent.is::<LocalLimitExec>();
        let unconstrained = matches!(
            parent
                .input_distribution_requirements()
                .child_distribution(index),
            None | Some(
                Distribution::UnspecifiedDistribution | Distribution::SinglePartition
            )
        ) && child.output_ordering().is_none()
            && !limits_partitions;
        if !single_partition && !unconstrained {
            return bridged_plan(child, limited);
        }
        Ok(Arc::new(PipelineRootExec::try_new(
            lowered,
            Arc::clone(child),
        )?))
    }

    /// The native operators of a lowered plan (see [`lower`]).
    #[derive(Debug, Default, Clone, Copy, PartialEq, Eq)]
    pub(crate) struct NativeOperators {
        pub(crate) filters: usize,
        pub(crate) projections: usize,
        pub(crate) caching: usize,
        /// Native scan sources.
        pub(crate) scans: usize,
    }

    impl NativeOperators {
        /// Count the native operators of an operator tree, including those
        /// of the `PipelineRootExec`s inside bridged plans.
        pub(crate) fn of(operator: &Arc<dyn PhysicalOperator>) -> Self {
            let mut counts = Self::default();
            counts.add_operator(operator);
            counts
        }

        fn add_operator(&mut self, operator: &Arc<dyn PhysicalOperator>) {
            self.filters += usize::from(operator.is::<PhysicalFilter>());
            self.projections += usize::from(operator.is::<PhysicalProjection>());
            self.caching += usize::from(operator.is::<CachingOperator>());
            self.scans += usize::from(
                operator.children().is_empty() && !operator.is::<BridgeOperator>(),
            );
            if let Some(bridge) = operator.downcast_ref::<BridgeOperator>() {
                self.add_plan(bridge.plan());
            }
            for child in operator.children() {
                self.add_operator(child);
            }
        }

        fn add_plan(&mut self, plan: &Arc<dyn ExecutionPlan>) {
            if let Some(root) = plan.downcast_ref::<PipelineRootExec>() {
                self.add_operator(root.operator());
            }
            for child in plan.children() {
                self.add_plan(child);
            }
        }
    }

    /// The result of running a query with DataFusion and with the native
    /// operators.
    pub(crate) struct NativeRun {
        pub(crate) expected: EngineOutput,
        pub(crate) actual: EngineOutput,
        /// The plan executed for `actual`.
        pub(crate) root: Arc<PipelineRootExec>,
        pub(crate) operators: NativeOperators,
    }

    /// Run `sql` with DataFusion and with the plan of DataFusion lowered by
    /// [`lower`], on sessions configured with `config` and set up with
    /// `setup`.
    pub(crate) async fn run_native<F>(
        sql: &str,
        config: SessionConfig,
        setup: F,
    ) -> Result<NativeRun>
    where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        let datafusion = datafusion_context(config.clone());
        setup(&datafusion).await?;
        let native = datafusion_context(config);
        setup(&native).await?;

        let expected = run_sql(&datafusion, sql).await?;
        let df = native.sql(sql).await?;
        let ordered = has_top_level_order_by(df.logical_plan());
        let plan = df.create_physical_plan().await?;
        let operator = lower(&plan)?;
        let operators = NativeOperators::of(&operator);
        let root = Arc::new(PipelineRootExec::try_new(operator, plan)?);
        let batches = collect(
            Arc::clone(&root) as Arc<dyn ExecutionPlan>,
            native.task_ctx(),
        )
        .await?;
        let actual = EngineOutput {
            plan: Arc::clone(&root) as Arc<dyn ExecutionPlan>,
            schema: root.schema(),
            batches,
            ordered,
        };
        Ok(NativeRun {
            expected,
            actual,
            root,
            operators,
        })
    }

    /// Assert that `sql` produces byte-identical results with DataFusion and
    /// with its native operators (see [`lower`]), like
    /// [`assert_byte_eq_with_datafusion_with_setup`](crate::test_utils::assert_byte_eq_with_datafusion_with_setup).
    /// `setup` registers the tables used by the query, e.g. [`all_tables`].
    pub(crate) async fn assert_native_byte_eq_with_setup<F>(
        sql: &str,
        setup: F,
    ) -> NativeRun
    where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        assert_native_byte_eq_with_config(sql, test_session_config(), setup).await
    }

    /// Like [`assert_native_byte_eq_with_setup`], with a custom session
    /// configuration.
    pub(crate) async fn assert_native_byte_eq_with_config<F>(
        sql: &str,
        config: SessionConfig,
        setup: F,
    ) -> NativeRun
    where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        let run = match run_native(sql, config, setup).await {
            Ok(run) => run,
            Err(e) => panic!("Failed to run query on both engines: {e}\nquery: {sql}"),
        };
        if let Err(message) = compare_outputs(&run.expected, &run.actual) {
            panic!("Results differ: {message}\nquery: {sql}");
        }
        run
    }

    /// Execute `plan`, created by `make_plan`, with DataFusion and with its
    /// native operators (see [`lower`]), and assert that the results are
    /// byte-identical (in the same order if `ordered`).
    pub(crate) async fn assert_native_plan_byte_eq(
        make_plan: impl Fn() -> Arc<dyn ExecutionPlan>,
        ordered: bool,
    ) -> NativeRun {
        let task_ctx = Arc::new(
            TaskContext::default()
                .with_session_config(SessionConfig::new().with_target_partitions(4)),
        );
        let plan = make_plan();
        let expected = EngineOutput {
            schema: plan.schema(),
            batches: collect(Arc::clone(&plan), Arc::clone(&task_ctx))
                .await
                .unwrap(),
            plan,
            ordered,
        };
        let plan = make_plan();
        let operator = lower(&plan).unwrap();
        let operators = NativeOperators::of(&operator);
        let root = Arc::new(PipelineRootExec::try_new(operator, plan).unwrap());
        let batches = collect(Arc::clone(&root) as Arc<dyn ExecutionPlan>, task_ctx)
            .await
            .unwrap();
        let actual = EngineOutput {
            plan: Arc::clone(&root) as Arc<dyn ExecutionPlan>,
            schema: root.schema(),
            batches,
            ordered,
        };
        if let Err(message) = compare_outputs(&expected, &actual) {
            panic!("Results differ: {message}");
        }
        NativeRun {
            expected,
            actual,
            root,
            operators,
        }
    }

    /// Execute `plan`, created by `make_plan`, with DataFusion and with its
    /// native operators (see [`lower`]), and assert that either both succeed
    /// with byte-identical results (in the same order if `ordered`), or both
    /// fail with the same error. Returns the error, if any, and the native
    /// operators that were used.
    pub(crate) async fn assert_native_plan_parity(
        make_plan: impl Fn() -> Arc<dyn ExecutionPlan>,
        ordered: bool,
    ) -> (Option<String>, NativeOperators) {
        let config = SessionConfig::new().with_target_partitions(4);
        assert_native_plan_parity_with_config(make_plan, ordered, config).await
    }

    /// [`assert_native_plan_parity`] with the session configuration `config`
    /// (whose target partitions are the threads of the pipelines).
    pub(crate) async fn assert_native_plan_parity_with_config(
        make_plan: impl Fn() -> Arc<dyn ExecutionPlan>,
        ordered: bool,
        config: SessionConfig,
    ) -> (Option<String>, NativeOperators) {
        let task_ctx = Arc::new(TaskContext::default().with_session_config(config));
        let plan = make_plan();
        let expected =
            collect(Arc::clone(&plan), Arc::clone(&task_ctx))
                .await
                .map(|batches| EngineOutput {
                    schema: plan.schema(),
                    batches,
                    plan,
                    ordered,
                });
        let plan = make_plan();
        let operator = lower(&plan).unwrap();
        let operators = NativeOperators::of(&operator);
        let root = executable(operator, plan).unwrap();
        let actual =
            collect(Arc::clone(&root), task_ctx)
                .await
                .map(|batches| EngineOutput {
                    plan: Arc::clone(&root),
                    schema: root.schema(),
                    batches,
                    ordered,
                });
        let error = match (expected, actual) {
            (Ok(expected), Ok(actual)) => {
                if let Err(message) = compare_outputs(&expected, &actual) {
                    panic!("Results differ: {message}");
                }
                None
            }
            (Err(expected), Err(actual)) => {
                assert_eq!(
                    error_kind(&expected),
                    error_kind(&actual),
                    "Errors differ: {expected} vs {actual}"
                );
                Some(expected.find_root().strip_backtrace())
            }
            (Ok(_), Err(actual)) => panic!("Only the native operators failed: {actual}"),
            (Err(expected), Ok(_)) => panic!("Only DataFusion failed: {expected}"),
        };
        (error, operators)
    }

    /// Run `sql` with DataFusion and with its native operators (see
    /// [`lower`]), and assert that either both succeed, with byte-identical
    /// results, or both fail with the same error. Returns the error, if any,
    /// and the native operators that were used.
    pub(crate) async fn assert_native_parity_with_setup<F>(
        sql: &str,
        setup: F,
    ) -> (Option<String>, NativeOperators)
    where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        assert_native_parity_with_config(sql, test_session_config(), setup).await
    }

    /// [`assert_native_parity_with_setup`] with the session configuration
    /// `config`.
    pub(crate) async fn assert_native_parity_with_config<F>(
        sql: &str,
        config: SessionConfig,
        setup: F,
    ) -> (Option<String>, NativeOperators)
    where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        let datafusion = datafusion_context(config.clone());
        setup(&datafusion).await.unwrap();
        let native = datafusion_context(config);
        setup(&native).await.unwrap();

        let expected = run_sql(&datafusion, sql).await;
        let df = native.sql(sql).await.unwrap();
        let ordered = has_top_level_order_by(df.logical_plan());
        let plan = df.create_physical_plan().await.unwrap();
        let operator = lower(&plan).unwrap();
        let operators = NativeOperators::of(&operator);
        let root = executable(operator, plan).unwrap();
        let actual = collect(Arc::clone(&root), native.task_ctx())
            .await
            .map(|batches| EngineOutput {
                plan: Arc::clone(&root) as Arc<dyn ExecutionPlan>,
                schema: root.schema(),
                batches,
                ordered,
            });
        let error = match (expected, actual) {
            (Ok(expected), Ok(actual)) => {
                if let Err(message) = compare_outputs(&expected, &actual) {
                    panic!("Results differ: {message}\nquery: {sql}");
                }
                None
            }
            (Err(expected), Err(actual)) => {
                assert_eq!(
                    error_kind(&expected),
                    error_kind(&actual),
                    "Errors differ: {expected} vs {actual}\nquery: {sql}"
                );
                Some(expected.find_root().strip_backtrace())
            }
            (Ok(_), Err(actual)) => {
                panic!("Only the native operators failed: {actual}\nquery: {sql}")
            }
            (Err(expected), Ok(_)) => {
                panic!("Only DataFusion failed: {expected}\nquery: {sql}")
            }
        };
        (error, operators)
    }

    /// The kind of an error, e.g. `DivideByZero` or `CastError`, without the
    /// data it reports: with several partitions, which row fails first is
    /// not deterministic, neither in DataFusion nor in the pipeline.
    fn error_kind(error: &DataFusionError) -> String {
        let debug = match error.find_root() {
            DataFusionError::ArrowError(error, _) => format!("ArrowError::{error:?}"),
            error => format!("{error:?}"),
        };
        debug.split('(').next().unwrap_or_default().to_owned()
    }

    /// Parse the SQL expression `sql` on `schema` into a physical expression,
    /// with DataFusion's type coercion.
    pub(crate) fn physical_expr(sql: &str, schema: &SchemaRef) -> Arc<dyn PhysicalExpr> {
        let ctx = SessionContext::new();
        let df_schema = DFSchema::try_from(schema.as_ref().clone()).unwrap();
        let expr = ctx.parse_sql_expr(sql, &df_schema).unwrap();
        ctx.create_physical_expr(expr, &df_schema).unwrap()
    }

    /// A batch of the columns `x` and `y` (`Int32`) and `s` (`Utf8`, `"a"` on
    /// all rows), with the runs `((x, y), rows)`.
    pub(crate) fn xys(runs: &[((i32, i32), usize)]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int32, true),
            Field::new("y", DataType::Int32, true),
            Field::new("s", DataType::Utf8, true),
        ]));
        let rows = || {
            runs.iter()
                .flat_map(|&(xy, rows)| std::iter::repeat_n(xy, rows))
        };
        let x = rows().map(|(x, _)| x).collect::<Int32Array>();
        let y = rows().map(|(_, y)| y).collect::<Int32Array>();
        let s = StringArray::from_iter_values(rows().map(|_| "a"));
        RecordBatch::try_new(schema, vec![Arc::new(x), Arc::new(y), Arc::new(s)]).unwrap()
    }

    /// A batch of the columns `x` (`Int32`) and `s` (`Utf8`), with the runs
    /// `((x, s), rows)`.
    pub(crate) fn xss(runs: &[((i32, &str), usize)]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int32, true),
            Field::new("s", DataType::Utf8, true),
        ]));
        let rows = || {
            runs.iter()
                .flat_map(|&(xs, rows)| std::iter::repeat_n(xs, rows))
        };
        let x = rows().map(|(x, _)| x).collect::<Int32Array>();
        let s = StringArray::from_iter_values(rows().map(|(_, s)| s));
        RecordBatch::try_new(schema, vec![Arc::new(x), Arc::new(s)]).unwrap()
    }

    /// A column that is not a child of the expression (like the `CustomColumn`
    /// of the tests of DataFusion's `CaseExpr`): the input columns it reads
    /// are opaque.
    #[derive(Debug, Hash, PartialEq, Eq)]
    pub(crate) struct OpaqueColumn(Column);

    /// The [`OpaqueColumn`] of the column `name` of `schema`.
    pub(crate) fn opaque_column(name: &str, schema: &Schema) -> Arc<dyn PhysicalExpr> {
        Arc::new(OpaqueColumn(Column::new_with_schema(name, schema).unwrap()))
    }

    impl std::fmt::Display for OpaqueColumn {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "opaque({})", self.0)
        }
    }

    impl PhysicalExpr for OpaqueColumn {
        fn data_type(&self, input_schema: &Schema) -> Result<DataType> {
            self.0.data_type(input_schema)
        }

        fn nullable(&self, input_schema: &Schema) -> Result<bool> {
            self.0.nullable(input_schema)
        }

        fn evaluate(&self, batch: &RecordBatch) -> Result<ColumnarValue> {
            self.0.evaluate(batch)
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

        fn fmt_sql(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "opaque({})", self.0.name())
        }
    }

    /// A volatile function that counts its calls: it returns the number of
    /// times it was called, as a string, on all rows.
    #[derive(Debug)]
    pub(crate) struct CountCalls {
        calls: Arc<std::sync::atomic::AtomicUsize>,
        signature: Signature,
    }

    impl PartialEq for CountCalls {
        fn eq(&self, other: &Self) -> bool {
            Arc::ptr_eq(&self.calls, &other.calls)
        }
    }

    impl Eq for CountCalls {}

    impl std::hash::Hash for CountCalls {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            Arc::as_ptr(&self.calls).hash(state);
        }
    }

    impl ScalarUDFImpl for CountCalls {
        fn name(&self) -> &str {
            "count_calls"
        }

        fn signature(&self) -> &Signature {
            &self.signature
        }

        fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
            Ok(DataType::Utf8)
        }

        fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
            let calls = self.calls.fetch_add(1, std::sync::atomic::Ordering::SeqCst) + 1;
            let values = std::iter::repeat_n(calls.to_string(), args.number_rows);
            Ok(ColumnarValue::Array(Arc::new(
                StringArray::from_iter_values(values),
            )))
        }
    }

    /// `count_calls()` (see [`CountCalls`]), counting in `calls`.
    pub(crate) fn count_calls(
        calls: &Arc<std::sync::atomic::AtomicUsize>,
        schema: &Schema,
    ) -> Arc<dyn PhysicalExpr> {
        let udf = ScalarUDF::new_from_impl(CountCalls {
            calls: Arc::clone(calls),
            signature: Signature::nullary(Volatility::Volatile),
        });
        Arc::new(
            ScalarFunctionExpr::try_new(
                Arc::new(udf),
                vec![],
                schema,
                Arc::new(datafusion_common::config::ConfigOptions::default()),
            )
            .unwrap(),
        )
    }

    /// A batch of 2048 rows of `x` (`Int32`), alternately `0` and `1`, and of
    /// `s` (`Utf8`), `"a"` but on the first row, which is null, above the
    /// bytes of `"é"`, which are not ASCII.
    pub(crate) fn non_ascii_beneath_a_null() -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int32, true),
            Field::new("s", DataType::Utf8, true),
        ]));
        let x = Int32Array::from_iter_values((0..2048).map(|row| row % 2));
        let values = std::iter::once("é")
            .chain(std::iter::repeat_n("a", 2047))
            .collect::<String>();
        let offsets = arrow::buffer::OffsetBuffer::from_lengths(
            std::iter::once("é".len()).chain(std::iter::repeat_n(1, 2047)),
        );
        let nulls = arrow::buffer::NullBuffer::from_iter((0..2048).map(|row| row != 0));
        let s = StringArray::try_new(offsets, values.into_bytes().into(), Some(nulls))
            .unwrap();
        RecordBatch::try_new(schema, vec![Arc::new(x), Arc::new(s)]).unwrap()
    }

    /// A batch of two columns `a` and `b` of dictionaries of dictionaries
    /// (`Dictionary(Int32, Dictionary(Int32, Utf8))`), which Arrow cannot
    /// compare.
    pub(crate) fn nested_dictionaries() -> RecordBatch {
        let column = || -> ArrayRef {
            let inner = DictionaryArray::<Int32Type>::try_new(
                Int32Array::from(vec![0, 1]),
                Arc::new(StringArray::from(vec!["p", "q"])),
            )
            .unwrap();
            Arc::new(
                DictionaryArray::<Int32Type>::try_new(
                    Int32Array::from(vec![0, 1, 0]),
                    Arc::new(inner),
                )
                .unwrap(),
            )
        };
        let (a, b) = (column(), column());
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", a.data_type().clone(), true),
            Field::new("b", b.data_type().clone(), true),
        ]));
        RecordBatch::try_new(schema, vec![a, b]).unwrap()
    }

    /// `x = 0` packed into a dictionary of booleans, on `schema`: Arrow cannot
    /// pack booleans into a dictionary, so this fails whenever it is
    /// evaluated.
    pub(crate) fn packed_booleans(schema: &SchemaRef) -> Arc<dyn PhysicalExpr> {
        Arc::new(datafusion_physical_expr::expressions::CastExpr::new(
            physical_expr("x = 0", schema),
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Boolean)),
            None,
        ))
    }

    /// `x = 0 OR (y = 0 OR packed_booleans IS NOT NULL)`, on `schema`.
    pub(crate) fn nested_packed_booleans(schema: &SchemaRef) -> Arc<dyn PhysicalExpr> {
        use datafusion_expr::Operator::Or;
        use datafusion_physical_expr::expressions::{BinaryExpr, IsNotNullExpr};
        let packed = Arc::new(IsNotNullExpr::new(packed_booleans(schema)));
        let inner = Arc::new(BinaryExpr::new(physical_expr("y = 0", schema), Or, packed));
        Arc::new(BinaryExpr::new(physical_expr("x = 0", schema), Or, inner))
    }

    /// Two batches of 512 rows of `x` (`Int32`), `1` on the first 256 rows and
    /// `0` on the others, and of `l`, lists of nulls, empty but on the first
    /// row, a list of `1 << 30` nulls (which take no memory).
    pub(crate) fn long_lists_of_nulls() -> Vec<RecordBatch> {
        let field = Arc::new(Field::new_list_field(DataType::Null, true));
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int32, false),
            Field::new("l", DataType::List(Arc::clone(&field)), true),
        ]));
        (0..2)
            .map(|_| {
                let x = Int32Array::from_iter_values(
                    (0..512).map(|row| i32::from(row < 256)),
                );
                let offsets = arrow::buffer::OffsetBuffer::from_lengths(
                    std::iter::once(1 << 30).chain(std::iter::repeat_n(0, 511)),
                );
                let nulls = Arc::new(arrow::array::NullArray::new(1 << 30));
                let l = arrow::array::ListArray::try_new(
                    Arc::clone(&field),
                    offsets,
                    nulls,
                    None,
                )
                .unwrap();
                RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(x), Arc::new(l)])
                    .unwrap()
            })
            .collect()
    }

    /// The plan that executes `operator`, the lowering of `plan` (see
    /// [`lower`]): a `PipelineRootExec`, or if `operator` executes everything
    /// in DataFusion (a `BridgeOperator`), the plan it executes, as a planner
    /// would not run a pipeline for it (and a `PipelineRootExec` checks that
    /// the batches have its schema exactly, which DataFusion's do not always:
    /// Arrow's `filter` renames the fields of run-end encoded arrays).
    pub(crate) fn executable(
        operator: Arc<dyn PhysicalOperator>,
        plan: Arc<dyn ExecutionPlan>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        if let Some(bridge) = operator.downcast_ref::<BridgeOperator>() {
            return Ok(Arc::clone(bridge.plan()));
        }
        Ok(Arc::new(PipelineRootExec::try_new(operator, plan)?))
    }

    /// The number of rows of `plan`, executed on a task, or its error, or the
    /// message of its panic.
    pub(crate) async fn outcome(
        plan: Arc<dyn ExecutionPlan>,
        task_ctx: Arc<TaskContext>,
    ) -> std::result::Result<usize, String> {
        match datafusion_common_runtime::SpawnedTask::spawn(collect(plan, task_ctx))
            .join()
            .await
        {
            Ok(Ok(batches)) => Ok(batches.iter().map(RecordBatch::num_rows).sum()),
            Ok(Err(error)) => Err(error.strip_backtrace()),
            Err(join) if join.is_panic() => {
                let panic = join.into_panic();
                Err(panic
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                    .unwrap_or_default())
            }
            Err(join) => Err(join.to_string()),
        }
    }

    /// Execute `plan` with DataFusion and lowered (see [`lower`]), each on a
    /// task, and assert that the outcomes are the same, with no native filter
    /// or projection.
    pub(crate) async fn assert_same_outcome_in_datafusion(
        plan: Arc<dyn ExecutionPlan>,
        task_ctx: Arc<TaskContext>,
    ) -> std::result::Result<usize, String> {
        let operator = lower(&plan).unwrap();
        let operators = NativeOperators::of(&operator);
        assert_eq!((operators.filters, operators.projections), (0, 0));
        let lowered = executable(operator, Arc::clone(&plan)).unwrap();
        let expected = outcome(plan, Arc::clone(&task_ctx)).await;
        assert_eq!(outcome(lowered, task_ctx).await, expected);
        expected
    }

    /// A batch of 4096 rows of `x` (`Boolean`), `true` on the first 2048 rows,
    /// and of `a` and `b`, lists of nulls (which take no memory), empty but a
    /// list of `1 << 30` nulls on the first row (`a`) and on the 2049th row
    /// (`b`).
    pub(crate) fn lists_to_merge() -> RecordBatch {
        let field = Arc::new(Field::new_list_field(DataType::Null, true));
        let list = |row: usize| -> ArrayRef {
            let offsets = arrow::buffer::OffsetBuffer::from_lengths(
                (0..4096).map(|other| if other == row { 1 << 30 } else { 0 }),
            );
            let nulls = Arc::new(arrow::array::NullArray::new(1 << 30));
            Arc::new(
                arrow::array::ListArray::try_new(
                    Arc::clone(&field),
                    offsets,
                    nulls,
                    None,
                )
                .unwrap(),
            )
        };
        let x =
            arrow::array::BooleanArray::from_iter((0..4096).map(|row| Some(row < 2048)));
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Boolean, true),
            Field::new("a", DataType::List(Arc::clone(&field)), true),
            Field::new("b", DataType::List(Arc::clone(&field)), true),
        ]));
        RecordBatch::try_new(schema, vec![Arc::new(x), list(0), list(2048)]).unwrap()
    }

    /// A batch of the rows `x` (`Int32`) and of `f`, fixed-size lists of
    /// `1 << 30` values that take no memory: nulls, or if `runs`, one run of
    /// `42` (`RunEndEncoded<Int64, Int32>`).
    pub(crate) fn wide_lists(x: &[i32], runs: bool) -> RecordBatch {
        let values = x.len() << 30;
        let child: ArrayRef = if runs {
            Arc::new(
                arrow::array::RunArray::<arrow::datatypes::Int64Type>::try_new(
                    &Int64Array::from(vec![values as i64]),
                    &Int32Array::from(vec![42]),
                )
                .unwrap(),
            )
        } else {
            Arc::new(arrow::array::NullArray::new(values))
        };
        let field = Arc::new(Field::new_list_field(child.data_type().clone(), true));
        let f = arrow::array::FixedSizeListArray::try_new(
            Arc::clone(&field),
            1 << 30,
            child,
            None,
        )
        .unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int32, false),
            Field::new("f", DataType::FixedSizeList(field, 1 << 30), true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(x.to_vec())), Arc::new(f)],
        )
        .unwrap()
    }

    /// A batch of 8 rows of `x` (`Int32`), alternately `1` and `0`, and of `r`,
    /// run-end encoded values whose child fields have other names than
    /// Arrow's (`ends` and `samples`).
    pub(crate) fn runs_with_named_fields() -> RecordBatch {
        let ends = Int32Array::from(vec![2, 4, 6, 8]);
        let samples = Int32Array::from(vec![1, 2, 3, 4]);
        let data_type = DataType::RunEndEncoded(
            Arc::new(Field::new("ends", DataType::Int32, false)),
            Arc::new(Field::new("samples", DataType::Int32, true)),
        );
        let data = arrow::array::ArrayData::try_new(
            data_type.clone(),
            8,
            None,
            0,
            vec![],
            vec![
                arrow::array::Array::to_data(&ends),
                arrow::array::Array::to_data(&samples),
            ],
        )
        .unwrap();
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int32, false),
            Field::new("r", data_type, true),
        ]));
        let x = Int32Array::from_iter_values((0..8).map(|row| i32::from(row % 2 == 0)));
        RecordBatch::try_new(schema, vec![Arc::new(x), arrow::array::make_array(data)])
            .unwrap()
    }

    /// A constant fixed-size list of 8 values, one run of `42` with 16-bit run
    /// ends (`RunEndEncoded<Int16, Int32>`): broadcast to more than 4095 rows,
    /// its values overflow the run ends.
    pub(crate) fn constant_list_of_short_runs() -> Arc<dyn PhysicalExpr> {
        let runs = arrow::array::RunArray::<arrow::datatypes::Int16Type>::try_new(
            &arrow::array::Int16Array::from(vec![8]),
            &Int32Array::from(vec![42]),
        )
        .unwrap();
        let data_type = arrow::array::Array::data_type(&runs).clone();
        let field = Arc::new(Field::new_list_field(data_type, true));
        let list =
            arrow::array::FixedSizeListArray::try_new(field, 8, Arc::new(runs), None)
                .unwrap();
        Arc::new(datafusion_physical_expr::expressions::Literal::new(
            datafusion_common::ScalarValue::FixedSizeList(Arc::new(list)),
        ))
    }

    /// A constant list of `1 << 19` nulls.
    pub(crate) fn constant_list_of_nulls() -> Arc<dyn PhysicalExpr> {
        let field = Arc::new(Field::new_list_field(DataType::Null, true));
        let list = arrow::array::ListArray::try_new(
            field,
            arrow::buffer::OffsetBuffer::from_lengths([1 << 19]),
            Arc::new(arrow::array::NullArray::new(1 << 19)),
            None,
        )
        .unwrap();
        Arc::new(datafusion_physical_expr::expressions::Literal::new(
            datafusion_common::ScalarValue::List(Arc::new(list)),
        ))
    }

    /// Two batches of a dictionary column `d` (`Int32` keys) with two rows,
    /// `"1"` and `"x"`, then `"2"` and `"y"`, with distinct dictionaries.
    pub(crate) fn dictionary_batches() -> Vec<RecordBatch> {
        let data_type =
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
        let schema = Arc::new(Schema::new(vec![Field::new("d", data_type, true)]));
        [["1", "x"], ["2", "y"]]
            .into_iter()
            .map(|values| {
                let d = DictionaryArray::<Int32Type>::try_new(
                    Int32Array::from(vec![0, 1]),
                    Arc::new(StringArray::from(values.to_vec())),
                )
                .unwrap();
                RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(d)]).unwrap()
            })
            .collect()
    }

    /// Two batches of 100 rows of a dictionary column `d`, with `keys`, and of
    /// the row number `i`: the 100 values of the first dictionary (`a0`, `a1`,
    /// …) are all distinct from those of the second one (`b0`, …).
    pub(crate) fn distinct_dictionary_batches(keys: &DataType) -> Vec<RecordBatch> {
        let data_type =
            DataType::Dictionary(Box::new(keys.clone()), Box::new(DataType::Utf8));
        let schema = Arc::new(Schema::new(vec![
            Field::new("d", data_type.clone(), true),
            Field::new("i", DataType::Int32, false),
        ]));
        ["a", "b"]
            .into_iter()
            .zip([0, 100])
            .map(|(prefix, first_row)| {
                let values = StringArray::from_iter_values(
                    (0..100).map(|v| format!("{prefix}{v}")),
                );
                let d = DictionaryArray::<Int32Type>::try_new(
                    Int32Array::from_iter_values(0..100),
                    Arc::new(values),
                )
                .unwrap();
                let d = arrow::compute::cast(&d, &data_type).unwrap();
                let i = Int32Array::from_iter_values(first_row..first_row + 100);
                RecordBatch::try_new(Arc::clone(&schema), vec![d, Arc::new(i)]).unwrap()
            })
            .collect()
    }

    /// A scan of `batches` (of `schema`, possibly empty or none), in a single
    /// partition.
    pub(crate) fn scan_batches(
        schema: &SchemaRef,
        batches: &[RecordBatch],
    ) -> Arc<dyn ExecutionPlan> {
        MemorySourceConfig::try_new_exec(&[batches.to_vec()], Arc::clone(schema), None)
            .unwrap()
    }

    /// Register `batches` as table `name`, with one partition per entry.
    pub(crate) fn register_partitions(
        ctx: &SessionContext,
        name: &str,
        partitions: Vec<Vec<RecordBatch>>,
    ) -> Result<()> {
        let schema = partitions
            .iter()
            .flatten()
            .next()
            .expect("at least one batch")
            .schema();
        ctx.register_table(name, Arc::new(MemTable::try_new(schema, partitions)?))?;
        Ok(())
    }

    /// Register the default test tables and:
    ///
    /// * `fallible`: 6000 rows in 3 partitions with `id BIGINT NOT NULL`,
    ///   `x INT` (90% of the rows are `0`, 2% are `NULL`), `s VARCHAR` (10%
    ///   of the rows are numbers) and `big BIGINT` (values that overflow an
    ///   `INT`, except on 10% of the rows). Guards like `x <> 0` select few
    ///   enough rows (at most 20%) for DataFusion to evaluate the conjuncts
    ///   after them only on the selected rows, so that DataFusion does not
    ///   fail on such queries either.
    /// * `vd`: 5000 rows in 3 partitions with `i BIGINT NOT NULL`, `sv`
    ///   (`Utf8View`), `d32` (`Dictionary(Int32, Utf8)`) and `d8`
    ///   (`Dictionary(Int8, Utf8)`), with nulls and different dictionaries
    ///   in every batch.
    pub(crate) fn register_all_tables(ctx: &SessionContext) -> Result<()> {
        register_test_tables(ctx)?;
        register_partitions(ctx, "fallible", fallible_table()?)?;
        register_partitions(ctx, "vd", view_dictionary_table()?)
    }

    /// [`register_all_tables`], as the setup of the differential tests.
    pub(crate) async fn all_tables(ctx: &SessionContext) -> Result<()> {
        register_all_tables(ctx)
    }

    /// The partitions of table `fallible` of [`register_all_tables`].
    pub(crate) fn fallible_table() -> Result<Vec<Vec<RecordBatch>>> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("x", DataType::Int32, true),
            Field::new("s", DataType::Utf8, true),
            Field::new("big", DataType::Int64, true),
        ]));
        let batch = |ids: std::ops::Range<i64>| -> Result<RecordBatch> {
            let x = Int32Array::from_iter(ids.clone().map(|i| match i {
                _ if i % 50 == 7 => None,
                _ if i % 10 == 3 => Some(match (i % 23) as i32 - 11 {
                    0 => 7,
                    v => v,
                }),
                _ => Some(0),
            }));
            let s = StringArray::from_iter(ids.clone().map(|i| match i {
                _ if i % 31 == 5 => None,
                _ if i % 10 == 0 => Some(format!("{}", i % 1000)),
                _ => Some(format!("n/a {i}")),
            }));
            let big = Int64Array::from_iter(ids.clone().map(|i| {
                (i % 37 != 2).then_some(if i % 10 == 0 { i } else { i * 10_000_000_000 })
            }));
            Ok(RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from_iter_values(ids)),
                    Arc::new(x),
                    Arc::new(s),
                    Arc::new(big),
                ],
            )?)
        };
        Ok(vec![
            vec![batch(0..2000)?],
            vec![batch(2000..2500)?, batch(2500..5000)?],
            vec![batch(5000..6000)?],
        ])
    }

    /// The partitions of table `vd` of [`register_all_tables`].
    pub(crate) fn view_dictionary_table() -> Result<Vec<Vec<RecordBatch>>> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("i", DataType::Int64, false),
            Field::new("sv", DataType::Utf8View, true),
            Field::new(
                "d32",
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                true,
            ),
            Field::new(
                "d8",
                DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
                true,
            ),
        ]));
        let batch = |ids: std::ops::Range<i64>| -> Result<RecordBatch> {
            let sv = StringViewArray::from_iter(ids.clone().map(|i| {
                (i % 17 != 4).then(|| {
                    if i % 2 == 0 {
                        format!("short{}", i % 9)
                    } else {
                        format!("a string that is longer than twelve bytes {}", i % 300)
                    }
                })
            }));
            // Every batch has its own dictionaries.
            let d32 = ids
                .clone()
                .map(|i| (i % 13 != 6).then(|| format!("k{}", i % 40)))
                .collect::<Vec<_>>();
            let d32 = d32
                .iter()
                .map(Option::as_deref)
                .collect::<DictionaryArray<Int32Type>>();
            let d8 = ids
                .clone()
                .map(|i| (i % 19 != 8).then(|| format!("x{}", (i * 7) % 90)))
                .collect::<Vec<_>>();
            let d8 = d8
                .iter()
                .map(Option::as_deref)
                .collect::<DictionaryArray<Int8Type>>();
            Ok(RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from_iter_values(ids)),
                    Arc::new(sv),
                    Arc::new(d32),
                    Arc::new(d8),
                ],
            )?)
        };
        Ok(vec![
            vec![batch(0..2100)?, batch(2100..2200)?],
            vec![batch(2200..4000)?],
            vec![batch(4000..5000)?],
        ])
    }

    /// A table `name` of 3 partitions with 4 batches of 100 rows each, with
    /// the columns `id` and `x`, `y`, `a`, `b` (`INT`, nullable) given by
    /// `row(id)`. The batches are not split into several chunks, so the
    /// filter decides like DataFusion for the same rows.
    pub(crate) fn register_guard_table(
        ctx: &SessionContext,
        name: &str,
        row: impl Fn(i64) -> [Option<i32>; 4],
    ) -> Result<()> {
        let schema = Arc::new(Schema::new(
            std::iter::once(Field::new("id", DataType::Int64, false))
                .chain(
                    ["x", "y", "a", "b"]
                        .into_iter()
                        .map(|name| Field::new(name, DataType::Int32, true)),
                )
                .collect::<Vec<_>>(),
        ));
        let batch = |batch: i64| {
            let ids = batch * 100..(batch + 1) * 100;
            let rows = ids.clone().map(&row).collect::<Vec<_>>();
            let mut columns: Vec<ArrayRef> =
                vec![Arc::new(Int64Array::from_iter_values(ids))];
            for column in 0..4 {
                columns.push(Arc::new(Int32Array::from_iter(
                    rows.iter().map(|row| row[column]),
                )));
            }
            RecordBatch::try_new(Arc::clone(&schema), columns).unwrap()
        };
        register_partitions(
            ctx,
            name,
            vec![vec![batch(0), batch(1)], vec![batch(2)], vec![batch(3)]],
        )
    }

    /// The tables of the error parity tests: [`register_all_tables`] and the
    /// guard tables `dense`, `sparse`, `nulls` and `nulls_ok`.
    pub(crate) async fn guard_tables(ctx: &SessionContext) -> Result<()> {
        register_all_tables(ctx)?;
        // Half of the rows have `x = 0`; `y = 1` on 2% of the rows, and only
        // where `x <> 0`.
        register_guard_table(ctx, "dense", |i| {
            let x = if i % 2 == 0 { 0 } else { 1 + (i % 7) as i32 };
            [Some(x), Some(i32::from(i % 50 == 1)), None, None]
        })?;
        // 90% of the rows have `x = 0`.
        register_guard_table(ctx, "sparse", |i| {
            let x = if i % 10 == 3 { 1 + (i % 4) as i32 } else { 0 };
            [Some(x), None, None, None]
        })?;
        // `a > 0` is `NULL` on 5% of the rows and `false` on the others; `b
        // = 0` on the `NULL` rows of `nulls`, but not of `nulls_ok`.
        register_guard_table(ctx, "nulls", |i| {
            let null = i % 20 == 7;
            [None, None, (!null).then_some(-1), Some(i32::from(!null))]
        })?;
        register_guard_table(ctx, "nulls_ok", |i| {
            let null = i % 20 == 7;
            [None, None, (!null).then_some(-1), Some(2)]
        })
    }

    /// A source that emits `batches` (in one partition, so in order).
    pub(crate) fn source(
        schema: SchemaRef,
        batches: Vec<RecordBatch>,
    ) -> Arc<dyn PhysicalOperator> {
        let plan = MemorySourceConfig::try_new_exec(&[batches], schema, None)
            .expect("valid memory source");
        Arc::new(BridgeOperator::new(plan))
    }

    /// A context to call operators directly.
    pub(crate) fn context() -> ExecutionContext {
        ExecutionContext::new(Arc::new(TaskContext::default()), 0, 1)
    }

    /// The states of `operator` for one thread.
    pub(crate) fn states(
        operator: &dyn PhysicalOperator,
    ) -> (Arc<dyn GlobalOperatorState>, Box<dyn OperatorState>) {
        let ctx = context();
        let gstate = operator.get_global_operator_state(&ctx).unwrap();
        let lstate = operator.get_operator_state(&ctx, gstate.as_ref()).unwrap();
        (gstate, lstate)
    }

    /// Push `inputs` through `operator` like a pipeline executor with a
    /// single thread does (including `HaveMoreOutput` and the final flush),
    /// and return the non-empty output chunks and whether the operator
    /// finished.
    pub(crate) fn drive(
        operator: &dyn PhysicalOperator,
        inputs: impl IntoIterator<Item = DataChunk>,
    ) -> Result<(Vec<DataChunk>, bool)> {
        let (gstate, mut lstate) = states(operator);
        drive_with_states(operator, inputs, gstate.as_ref(), lstate.as_mut())
    }

    /// Like [`drive`], with given states.
    pub(crate) fn drive_with_states(
        operator: &dyn PhysicalOperator,
        inputs: impl IntoIterator<Item = DataChunk>,
        gstate: &dyn GlobalOperatorState,
        lstate: &mut dyn OperatorState,
    ) -> Result<(Vec<DataChunk>, bool)> {
        let ctx = context();
        let mut outputs = vec![];
        let mut output = DataChunk::empty(operator.schema());
        for input in inputs {
            loop {
                output.reset();
                let result =
                    operator.execute(&ctx, &input, &mut output, gstate, lstate)?;
                if !output.is_empty() {
                    outputs.push(output.clone());
                }
                match result {
                    OperatorResult::NeedMoreInput => break,
                    OperatorResult::HaveMoreOutput => {}
                    OperatorResult::Finished => return Ok((outputs, true)),
                    OperatorResult::Blocked(interrupt) => {
                        panic!("unexpected {interrupt:?}")
                    }
                }
            }
        }
        if operator.requires_final_execute() {
            loop {
                output.reset();
                let result = operator.final_execute(&ctx, &mut output, gstate, lstate)?;
                if !output.is_empty() {
                    outputs.push(output.clone());
                }
                if result == OperatorFinalizeResult::Finished {
                    break;
                }
            }
        }
        Ok((outputs, false))
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;

    use arrow::array::{ArrayRef, Int32Array, Int64Array, StringArray, StringViewArray};
    use arrow::compute::filter_record_batch;
    use arrow::datatypes::{Field, Schema};
    use datafusion_execution::TaskContext;
    use datafusion_physical_expr::{LexOrdering, PhysicalSortExpr};
    use datafusion_physical_plan::filter::batch_filter;
    use datafusion_physical_plan::sorts::sort::SortExec;

    use crate::test_utils::test_session_config;

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Int64, false),
            Field::new("s", DataType::Utf8, true),
            Field::new("v", DataType::Utf8View, true),
        ]))
    }

    /// `rows` rows starting at `start`: `a` (nulls on every 7th row), `b`,
    /// `s` and `v`.
    fn batch(start: i64, rows: usize) -> RecordBatch {
        let ids = start..start + rows as i64;
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int32Array::from_iter(
                ids.clone()
                    .map(|i| (i % 7 != 3).then_some((i % 100) as i32)),
            )),
            Arc::new(Int64Array::from_iter_values(ids.clone())),
            Arc::new(StringArray::from_iter(
                ids.clone()
                    .map(|i| (i % 5 != 1).then(|| format!("s{}", i % 11))),
            )),
            Arc::new(StringViewArray::from_iter(ids.map(|i| {
                (i % 9 != 2).then(|| format!("a string view longer than 12 bytes {i}"))
            }))),
        ];
        RecordBatch::try_new(schema(), columns).unwrap()
    }

    fn new_filter(predicate: &str, projection: Option<Vec<usize>>) -> PhysicalFilter {
        let schema = schema();
        PhysicalFilter::try_new(
            source(Arc::clone(&schema), vec![]),
            physical_expr(predicate, &schema),
            projection,
        )
        .unwrap()
    }

    /// Filter `input` with `filter` in a fresh thread state.
    fn execute(
        filter: &PhysicalFilter,
        input: &DataChunk,
    ) -> (DataChunk, OperatorResult) {
        let (gstate, mut lstate) = states(filter);
        let mut output = DataChunk::empty(filter.schema());
        let result = filter
            .execute(
                &context(),
                input,
                &mut output,
                gstate.as_ref(),
                lstate.as_mut(),
            )
            .unwrap();
        (output, result)
    }

    /// What `FilterExec` produces for `predicate` on the logical rows of
    /// `input`.
    fn expected(predicate: &str, input: &DataChunk) -> RecordBatch {
        batch_filter(
            &input.compact().unwrap(),
            &physical_expr(predicate, &schema()),
        )
        .unwrap()
    }

    fn same_buffers(left: &ArrayRef, right: &ArrayRef) -> bool {
        left.to_data().buffers()[0].as_ptr() == right.to_data().buffers()[0].as_ptr()
    }

    #[test]
    fn selects_rows_with_a_selection_vector_without_copying() {
        let input = DataChunk::from_record_batch(batch(0, 100));
        let filter = new_filter("a > 50", None);
        let (output, result) = execute(&filter, &input);
        assert!(matches!(result, OperatorResult::NeedMoreInput));
        assert_eq!(output.schema(), &schema());
        // The rows with `a > 50`: `NULL` is not selected.
        let indices = (0..100u32)
            .filter(|i| i % 7 != 3 && *i > 50)
            .collect::<Vec<_>>();
        assert_eq!(output.selection().unwrap().indices(), indices.as_slice());
        // The columns are the input columns.
        for (output, input) in output.columns().iter().zip(input.columns()) {
            assert!(Arc::ptr_eq(output, input));
        }
        assert_eq!(output.compact().unwrap(), expected("a > 50", &input));
    }

    #[test]
    fn all_true_keeps_the_input_and_all_false_produces_nothing() {
        let input = DataChunk::from_record_batch(batch(0, 100));
        let (output, result) = execute(&new_filter("b >= 0", None), &input);
        assert!(matches!(result, OperatorResult::NeedMoreInput));
        assert!(output.selection().is_none());
        assert_eq!(output.len(), 100);
        assert!(Arc::ptr_eq(output.column(2), input.column(2)));

        for predicate in [
            "b < 0",
            "a > 50 AND b < 10",
            "false",
            "CAST(NULL AS BOOLEAN)",
        ] {
            let (output, result) = execute(&new_filter(predicate, None), &input);
            assert!(
                matches!(result, OperatorResult::NeedMoreInput),
                "{predicate}"
            );
            assert!(output.is_empty(), "{predicate}");
        }
        let (output, _) = execute(&new_filter("true", None), &input);
        assert_eq!(output.len(), 100);
        assert!(output.selection().is_none());
    }

    #[test]
    fn conjuncts_refine_the_selection_of_the_input() {
        // The input already has a selection, e.g. from a previous filter.
        let input = DataChunk::from_record_batch(batch(0, 2000));
        for (incoming, predicate) in [
            // Dense incoming selections (more than a quarter of the rows):
            // infallible conjuncts are evaluated on all physical rows.
            (
                (0..2000).filter(|i| i % 3 != 0).collect::<Vec<u32>>(),
                "a > 10 AND s <> 's3'",
            ),
            // Sparse: the referenced columns are gathered.
            (
                (0..2000).filter(|i| i % 9 == 0).collect(),
                "a > 10 AND s <> 's3' AND v LIKE '%1%'",
            ),
            // Not increasing, dense and sparse.
            (
                (0..2000).rev().step_by(3).collect(),
                "a % 2 = 0 AND b > 100",
            ),
            (
                (0..2000).rev().step_by(9).collect(),
                "a > 10 AND s <> 's3' AND b > 100",
            ),
            ((500..1500).collect(), "a < 90 OR s IS NULL"),
        ] {
            let input = input
                .clone()
                .with_selection(SelectionVector::from(incoming))
                .unwrap();
            let (output, _) = execute(&new_filter(predicate, None), &input);
            assert_eq!(
                output.compact().unwrap(),
                expected(predicate, &input),
                "{predicate}"
            );
            // Still the physical columns of the input.
            assert!(Arc::ptr_eq(output.column(3), input.column(3)));
        }
    }

    #[test]
    fn short_circuits_once_no_row_is_left() {
        // `10 / z` fails on every row: it is never evaluated, because `a < 0`
        // selects no row.
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("z", DataType::Int32, false),
        ]));
        let input = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from_iter_values(0..100)),
                Arc::new(Int32Array::from_iter_values(std::iter::repeat_n(0, 100))),
            ],
        )
        .unwrap();
        let filter = PhysicalFilter::try_new(
            source(Arc::clone(&schema), vec![]),
            physical_expr("a < 0 AND 10 / z > 1", &schema),
            None,
        )
        .unwrap();
        let (outputs, finished) = drive(
            &filter,
            std::iter::repeat_n(DataChunk::from(input.clone()), 50),
        )
        .unwrap();
        assert!(outputs.is_empty());
        assert!(!finished);
        // Without the guard, the division fails.
        let filter = PhysicalFilter::try_new(
            source(Arc::clone(&schema), vec![]),
            physical_expr("10 / z > 1 AND a < 0", &schema),
            None,
        )
        .unwrap();
        let err = drive(&filter, [DataChunk::from(input)]).unwrap_err();
        assert!(err.to_string().contains("Divide by zero"), "{err}");
    }

    /// A filter for `predicate` on the columns of `batch`.
    fn filter_for(predicate: &str, schema: &SchemaRef) -> PhysicalFilter {
        PhysicalFilter::try_new(
            source(Arc::clone(schema), vec![]),
            physical_expr(predicate, schema),
            None,
        )
        .unwrap()
    }

    /// Filter `input` natively (in the thread state `lstate`) and with
    /// DataFusion's `FilterExec` kernel, which evaluates the predicate on the
    /// logical rows of `input`, and assert that both select the same rows or
    /// both fail with the same error. Returns the error, if any.
    fn assert_filter_parity(
        filter: &PhysicalFilter,
        input: &DataChunk,
        gstate: &dyn GlobalOperatorState,
        lstate: &mut dyn OperatorState,
    ) -> Option<String> {
        let batch = input.compact().unwrap();
        let expected = batch_filter(&batch, filter.predicate());
        let actual = drive_with_states(filter, [input.clone()], gstate, lstate).map(
            |(outputs, _)| {
                let outputs = outputs
                    .iter()
                    .map(|chunk| chunk.compact().unwrap())
                    .collect::<Vec<_>>();
                arrow::compute::concat_batches(&filter.schema(), &outputs).unwrap()
            },
        );
        match (expected, actual) {
            (Ok(expected), Ok(actual)) => {
                assert_eq!(actual, expected, "{}", filter.predicate());
                None
            }
            (Err(expected), Err(actual)) => {
                let expected = expected.find_root().strip_backtrace();
                assert_eq!(
                    actual.find_root().strip_backtrace(),
                    expected,
                    "{}",
                    filter.predicate()
                );
                Some(expected)
            }
            (expected, actual) => panic!(
                "{}: DataFusion returned {expected:?}, the native filter {actual:?}",
                filter.predicate()
            ),
        }
    }

    /// A batch with the columns `x` and `y` (`Int32`, nullable).
    fn xy(x: Vec<Option<i32>>, y: Vec<Option<i32>>) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int32, true),
            Field::new("y", DataType::Int32, true),
        ]));
        RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(x)), Arc::new(Int32Array::from(y))],
        )
        .unwrap()
    }

    /// Whether filtering `batch` with `predicate` fails, natively and in
    /// DataFusion (both must agree).
    fn fails(predicate: &str, batch: &RecordBatch) -> bool {
        let filter = filter_for(predicate, &batch.schema());
        let (gstate, mut lstate) = states(&filter);
        assert_filter_parity(
            &filter,
            &DataChunk::from(batch.clone()),
            gstate.as_ref(),
            lstate.as_mut(),
        )
        .is_some()
    }

    #[test]
    fn dense_guards_fail_like_datafusion() {
        let some = |values: &[i32]| values.iter().map(|&v| Some(v)).collect::<Vec<_>>();
        // `x <> 0` leaves 3 of the 4 rows: DataFusion evaluates `10 / x` on
        // all rows, including `x = 0`, and fails.
        let batch = xy(some(&[0, 1, 1, 1]), some(&[1, 1, 1, 1]));
        assert!(fails("x <> 0 AND 10 / x > 1", &batch));
        // At most 20% of the rows left: only those are divided.
        let batch = xy(some(&[0, 0, 0, 0, 2]), some(&[1; 5]));
        assert!(!fails("x <> 0 AND 10 / x > 1", &batch));
        let batch = xy(some(&[0, 0, 0, 0, 0, 0, 0, 2, 3, 0]), some(&[1; 10]));
        assert!(!fails("x <> 0 AND 10 / x > 1", &batch));
        let batch = xy(some(&[0, 0, 0, 0, 0, 0, 2, 3, 4, 0]), some(&[1; 10]));
        assert!(fails("x <> 0 AND 10 / x > 1", &batch));
        // A guard on another column.
        let batch = xy(some(&[0, 5, 0, 5, 5]), some(&[0, 1, 0, 1, 1]));
        assert!(fails("y = 1 AND 10 / x > 1", &batch));
        let batch = xy(some(&[0, 5, 0, 0, 0, 0]), some(&[0, 1, 0, 0, 0, 0]));
        assert!(!fails("y = 1 AND 10 / x > 1", &batch));
        // Everything rejected: the division is not evaluated.
        let batch = xy(some(&[0, 0, 0]), some(&[1, 1, 1]));
        assert!(!fails("x <> 0 AND 10 / x > 1", &batch));
        // The first conjunct is always evaluated on all rows.
        assert!(fails("10 / x > 1 AND x <> 0", &batch));
    }

    #[test]
    fn nullable_conjuncts_fail_like_datafusion() {
        let mut x = vec![Some(0)];
        x.extend([Some(-1); 9]);
        let mut y = vec![Some(0)];
        y.extend([Some(5); 9]);
        // `x > 0` is `NULL` for the first row, which is not selected but not
        // rejected either: DataFusion evaluates `10 / y` on it and fails.
        let mut nullable_x = x.clone();
        nullable_x[0] = None;
        assert!(fails(
            "x > 0 AND 10 / y > 1",
            &xy(nullable_x.clone(), y.clone())
        ));
        assert!(!fails("x > 0 AND 10 / y > 1", &xy(x, y.clone())));
        // Only `NULL` rows left: they are evaluated too.
        assert!(fails(
            "x > 0 AND 10 / y > 1",
            &xy(vec![None; 10], y.clone())
        ));
        assert!(!fails(
            "x > 0 AND 10 / y > 1",
            &xy(vec![None; 10], vec![Some(5); 10])
        ));
        // A `NULL` and a `false` row: the `NULL` row is evaluated.
        let mut z = vec![None, Some(0)];
        z.extend([Some(-1); 8]);
        assert!(fails("x > 0 AND 10 / y > 1", &xy(z, y)));
    }

    #[test]
    fn chunks_with_a_selection_fail_like_their_rows() {
        // DataFusion evaluates the predicate on the rows of the chunk: those
        // of its selection.
        let mut x = vec![Some(0); 100];
        x[90] = Some(1);
        x[95] = Some(2);
        let batch = xy(x, vec![Some(1); 100]);
        let filter = filter_for("x <> 0 AND 10 / x > 1", &batch.schema());
        let (gstate, mut lstate) = states(&filter);
        for (selection, fail) in [
            // 2 of 100 rows: no failure.
            ((0..100).collect::<Vec<u32>>(), false),
            // 2 of 4 rows: a failure.
            (vec![89, 90, 94, 95], true),
            // 2 of 10 rows: no failure.
            ((86..96).collect(), false),
            // 2 of 3 rows, not increasing.
            (vec![95, 0, 90], true),
        ] {
            let input = DataChunk::from(batch.clone())
                .with_selection(SelectionVector::from(selection))
                .unwrap();
            let error =
                assert_filter_parity(&filter, &input, gstate.as_ref(), lstate.as_mut());
            assert_eq!(error.is_some(), fail, "{input:?}");
        }
    }

    #[test]
    fn adaptation_does_not_change_errors() {
        // The selective `y = 1` must not move before the division: after
        // warm-up chunks without `x = 0`, a chunk with `x = 0` and `y = 0`
        // fails, as in DataFusion.
        let batch = |zeros: bool| {
            xy(
                (0..2048)
                    .map(|i| Some(if zeros && i % 2 == 0 { 0 } else { 1 + i % 5 }))
                    .collect(),
                (0..2048).map(|i| Some(i32::from(i % 50 == 1))).collect(),
            )
        };
        let filter = filter_for("10 / x > 1 AND y = 1 AND x > 0", &batch(false).schema());
        assert_eq!(
            filter
                .adaptive_filter()
                .segments()
                .iter()
                .map(|s| (s.start, s.end))
                .collect::<Vec<_>>(),
            vec![(1, 3)]
        );
        let (gstate, mut lstate) = states(&filter);
        let chunks = super::super::adaptive_filter::WARMUP_CHUNKS
            + super::super::adaptive_filter::ADAPT_INTERVAL;
        for _ in 0..chunks {
            let error = assert_filter_parity(
                &filter,
                &DataChunk::from(batch(false)),
                gstate.as_ref(),
                lstate.as_mut(),
            );
            assert!(error.is_none());
        }
        let permutation = lstate
            .downcast_ref::<FilterState>()
            .unwrap()
            .adaptive
            .permutation()
            .to_vec();
        // The division stays first; `y = 1` moved before `x > 0`.
        assert_eq!(permutation, vec![0, 1, 2]);
        let error = assert_filter_parity(
            &filter,
            &DataChunk::from(batch(true)),
            gstate.as_ref(),
            lstate.as_mut(),
        );
        assert!(error.unwrap().contains("Divide by zero"));
    }

    /// A pseudo-random number generator (a linear congruential generator).
    struct Lcg(u64);

    impl Lcg {
        fn below(&mut self, n: u64) -> u64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            (self.0 >> 33) % n
        }
    }

    #[test]
    fn errors_match_datafusion_on_random_chunks() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int32, true),
            Field::new("y", DataType::Int32, true),
            Field::new("z", DataType::Int32, false),
        ]));
        let mut random = Lcg(42);
        // Chunks of up to 300 rows in which a varying share of the rows has
        // `x = 0`, `y = 0` or `NULL`s: guards leave below and above 20% of the
        // rows.
        let chunks = (0..400)
            .map(|_| {
                let rows = 1 + random.below(300) as usize;
                let zeros = random.below(100);
                let nulls = random.below(30);
                let value = |random: &mut Lcg| match random.below(100) {
                    p if p < nulls => None,
                    p if p < nulls + zeros => Some(0),
                    _ => Some(1 + random.below(6) as i32),
                };
                let x = (0..rows)
                    .map(|_| value(&mut random))
                    .collect::<Int32Array>();
                let y = (0..rows)
                    .map(|_| value(&mut random))
                    .collect::<Int32Array>();
                let z = Int32Array::from_iter_values(
                    (0..rows).map(|_| random.below(10) as i32),
                );
                let batch = RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![Arc::new(x), Arc::new(y), Arc::new(z)],
                )
                .unwrap();
                let chunk = DataChunk::from(batch);
                if random.below(4) == 0 {
                    // A chunk with a selection.
                    let selection = (0..rows as u32)
                        .filter(|_| random.below(3) != 0)
                        .collect::<Vec<_>>();
                    if !selection.is_empty() {
                        return chunk
                            .with_selection(SelectionVector::from(selection))
                            .unwrap();
                    }
                }
                chunk
            })
            .collect::<Vec<_>>();
        for predicate in [
            "x <> 0 AND 10 / x > 1",
            "y > 2 AND 10 / x > 1",
            "10 / x > 1 AND y = 1",
            "y = 1 AND z > 3 AND 10 / x > 1 AND y < 5",
            "x IS NOT NULL AND y > 0 AND 100 % x = 1 AND z <> 4",
            "(x = 0 OR 10 / x > 2) AND y > 1",
            "y <> 0 AND 10 / y > 1 AND x <> 0 AND 10 / x > 1",
            "x > 0 AND z > 2 AND 10 / (x - 1) > 1",
            "CASE WHEN x <> 0 THEN 10 / x > 1 ELSE false END AND y > 0",
            "x > 0 AND y > 0 AND z > 0",
        ] {
            let filter = filter_for(predicate, &schema);
            let (gstate, mut lstate) = states(&filter);
            let mut errors = 0;
            let mut rows = 0;
            for chunk in &chunks {
                match assert_filter_parity(
                    &filter,
                    chunk,
                    gstate.as_ref(),
                    lstate.as_mut(),
                ) {
                    Some(_) => errors += 1,
                    None => rows += 1,
                }
            }
            // Both outcomes are covered (`CASE` guards its division, and the
            // last predicate cannot fail).
            assert!(rows > 0, "{predicate}");
            let can_fail =
                !predicate.starts_with("CASE") && !predicate.ends_with("z > 0");
            assert_eq!(errors > 0, can_fail, "{predicate}");
        }
    }

    #[test]
    fn adapts_the_order_of_the_conjuncts() {
        // `b >= 0` selects everything, `upper(s) = 'S7'` 9% of the rows.
        let filter = new_filter("b >= 0 AND s IS NOT NULL AND upper(s) = 'S7'", None);
        assert!(filter.adaptive_filter().is_adaptive());
        let (gstate, mut lstate) = states(&filter);
        let permutation = |lstate: &dyn OperatorState| {
            lstate
                .downcast_ref::<FilterState>()
                .unwrap()
                .adaptive
                .permutation()
                .to_vec()
        };
        // `upper(s) = 'S7'` (a function) is fallible: it stays after the
        // others.
        for b in 0..20 {
            let input = DataChunk::from_record_batch(batch(b * 2048, 2048));
            let (outputs, _) = drive_with_states(
                &filter,
                [input.clone()],
                gstate.as_ref(),
                lstate.as_mut(),
            )
            .unwrap();
            assert_eq!(outputs.len(), 1);
            assert_eq!(
                outputs[0].compact().unwrap(),
                expected("b >= 0 AND s IS NOT NULL AND upper(s) = 'S7'", &input)
            );
        }
        // `s IS NOT NULL` removes some rows, `b >= 0` none: it moved first.
        assert_eq!(permutation(lstate.as_ref()), vec![1, 0, 2]);

        // Infallible conjuncts are reordered freely. (The conjuncts that
        // select every row have a rank of 0, so this order does not depend
        // on the measured costs.)
        let filter = new_filter("b >= 0 AND b < 1000000000 AND a = 7", None);
        let (gstate, mut lstate) = states(&filter);
        for b in 0..(super::super::adaptive_filter::WARMUP_CHUNKS as i64) {
            let input = DataChunk::from_record_batch(batch(b * 2048, 2048));
            drive_with_states(&filter, [input], gstate.as_ref(), lstate.as_mut())
                .unwrap();
        }
        assert_eq!(permutation(lstate.as_ref()), vec![2, 0, 1]);
    }

    #[test]
    fn embedded_projection_shares_the_input_columns() {
        let input = DataChunk::from_record_batch(batch(0, 100));
        let filter = new_filter("a > 50", Some(vec![3, 1]));
        assert_eq!(
            filter
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().as_str())
                .collect::<Vec<_>>(),
            vec!["v", "b"]
        );
        let (output, _) = execute(&filter, &input);
        assert!(Arc::ptr_eq(output.column(0), input.column(3)));
        assert!(Arc::ptr_eq(output.column(1), input.column(1)));
        assert!(same_buffers(output.column(1), input.column(1)));
        assert_eq!(
            output.compact().unwrap(),
            expected("a > 50", &input).project(&[3, 1]).unwrap()
        );

        // An empty projection only keeps the number of rows.
        let (output, _) = execute(&new_filter("a > 50", Some(vec![])), &input);
        assert_eq!(output.num_columns(), 0);
        assert_eq!(output.len(), expected("a > 50", &input).num_rows());
    }

    #[test]
    fn chunks_without_columns_keep_their_row_count() {
        let schema = Arc::new(Schema::empty());
        let filter = PhysicalFilter::try_new(
            source(Arc::clone(&schema), vec![]),
            physical_expr("true", &schema),
            None,
        )
        .unwrap();
        let input = DataChunk::try_new_without_columns(Arc::clone(&schema), 10).unwrap();
        let (output, _) = execute(&filter, &input);
        assert_eq!(output.len(), 10);
        // A selection of a chunk without columns is only a row count.
        let input = input
            .with_selection(SelectionVector::from(vec![2, 5, 9]))
            .unwrap();
        let (output, _) = execute(&filter, &input);
        assert_eq!(output.len(), 3);
        assert_eq!(output.compact().unwrap().num_rows(), 3);
    }

    #[test]
    fn fetch_limits_the_rows_of_all_states() {
        let filter = new_filter("b % 10 = 1", None).with_fetch(Some(25));
        let ctx = context();
        let gstate = filter.get_global_operator_state(&ctx).unwrap();
        // Two local states sharing the global state, as for the pipelines of
        // the inputs of a union below the filter (which run one after the
        // other, as the filter is not parallel): 10 rows per chunk.
        let mut threads = (0..2)
            .map(|_| filter.get_operator_state(&ctx, gstate.as_ref()).unwrap())
            .collect::<Vec<_>>();
        let mut rows = vec![];
        let mut results = vec![];
        for i in 0..4 {
            let lstate = &mut threads[i % 2];
            let input = DataChunk::from_record_batch(batch(i as i64 * 100, 100));
            let mut output = DataChunk::empty(filter.schema());
            let result = filter
                .execute(&ctx, &input, &mut output, gstate.as_ref(), lstate.as_mut())
                .unwrap();
            rows.push(output.len());
            results.push(matches!(result, OperatorResult::Finished));
        }
        assert_eq!(rows, vec![10, 10, 5, 0]);
        assert_eq!(results, vec![false, false, true, true]);

        // `fetch = 0` produces nothing.
        let filter = new_filter("true", None).with_fetch(Some(0));
        let (outputs, finished) =
            drive(&filter, [DataChunk::from_record_batch(batch(0, 10))]).unwrap();
        assert!(outputs.is_empty());
        assert!(finished);
    }

    #[test]
    fn errors_depend_on_batches_after_the_first_conjunct_and_in_or() {
        // The first conjunct is evaluated on all rows, whatever the batches.
        assert!(!new_filter("a > 0 AND b > 0", None).errors_depend_on_batches());
        assert!(!new_filter("10 / a > 1 AND b > 0", None).errors_depend_on_batches());
        assert!(new_filter("a > 0 AND 10 / a > 1", None).errors_depend_on_batches());
        // The right side of an `OR`, also on its own.
        assert!(new_filter("a = 0 OR 10 / a > 1", None).errors_depend_on_batches());
        assert!(!new_filter("10 / a > 1 OR a = 0", None).errors_depend_on_batches());
    }

    #[test]
    fn rejects_non_boolean_predicates() {
        let schema = schema();
        let err = PhysicalFilter::try_new(
            source(Arc::clone(&schema), vec![]),
            physical_expr("a + 1", &schema),
            None,
        )
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("Filter predicate must return BOOLEAN values"),
            "{err}"
        );
    }

    #[test]
    fn description_shows_predicate_projection_and_fetch() {
        let filter = new_filter("b > 50", Some(vec![3, 1])).with_fetch(Some(5));
        assert_eq!(
            filter.description(),
            "PhysicalFilter: b@1 > 50, projection=[v@3, b@1], fetch=5"
        );
        assert_eq!(filter.fetch(), Some(5));
        assert_eq!(filter.projection(), Some([3, 1].as_slice()));
    }

    #[test]
    fn results_match_filter_record_batch_over_many_chunks() {
        // Whatever the order of the conjuncts, the result is the same.
        let predicates = [
            "a > 20 AND s <> 's4' AND v LIKE '%7%' AND b % 3 = 0",
            "s IS NOT NULL AND a IS NULL",
            "(a > 90 OR s = 's1') AND v IS NOT NULL AND b < 50000",
        ];
        for predicate in predicates {
            let filter = new_filter(predicate, Some(vec![1, 2]));
            let (gstate, mut lstate) = states(&filter);
            for b in 0..40 {
                let batch = batch(b * 1500, 1500);
                let input = DataChunk::from(batch.clone());
                let (outputs, _) =
                    drive_with_states(&filter, [input], gstate.as_ref(), lstate.as_mut())
                        .unwrap();
                let mask = physical_expr(predicate, &schema())
                    .evaluate(&batch)
                    .unwrap()
                    .into_array(batch.num_rows())
                    .unwrap();
                let reference = filter_record_batch(
                    &batch.project(&[1, 2]).unwrap(),
                    mask.as_boolean(),
                )
                .unwrap();
                let actual = outputs
                    .iter()
                    .map(|chunk| chunk.compact().unwrap())
                    .collect::<Vec<_>>();
                assert_eq!(
                    arrow::compute::concat_batches(&filter.schema(), &actual).unwrap(),
                    reference,
                    "{predicate}"
                );
            }
        }
    }

    // ---------------------------------------------------------------------
    // Differential tests
    // ---------------------------------------------------------------------

    async fn assert_all_native_byte_eq(queries: &[&str]) {
        for sql in queries {
            let run = assert_native_byte_eq_with_setup(sql, all_tables).await;
            assert!(run.operators.filters > 0, "no native filter: {sql}");
            assert!(run.operators.scans > 0, "no native scan: {sql}");
        }
    }

    /// Like [`assert_all_native_byte_eq`], for queries whose filters must stay
    /// in DataFusion, because their errors may depend on DataFusion's batches.
    async fn assert_all_byte_eq_in_datafusion(queries: &[&str]) {
        for sql in queries {
            let run = assert_native_byte_eq_with_setup(sql, all_tables).await;
            assert_eq!(run.operators.filters, 0, "native filter: {sql}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_selective_filters() {
        assert_all_native_byte_eq(&[
            "SELECT * FROM t1 WHERE a > 10",
            "SELECT * FROM t1 WHERE id % 1000 = 7",
            "SELECT id, a, h FROM t1 WHERE a > 10 AND d <> 'beta'",
            "SELECT * FROM t1 WHERE id < 0",
            "SELECT * FROM t_empty WHERE a > 0",
            "SELECT * FROM t1 WHERE a IS NULL",
            "SELECT id, c FROM t1 WHERE c > -0.0",
            "SELECT id, f, ts FROM t1 WHERE f > DATE '1970-01-02' AND ts < TIMESTAMP '2024-01-10 00:00:00'",
            "SELECT id, g FROM t1 WHERE g > 0 AND g < 100.5",
            "SELECT id FROM t1 WHERE a IN (1, 2, 3, -5) AND d IS NOT NULL",
            "SELECT id, e FROM t1 WHERE e",
            "SELECT id, e FROM t1 WHERE NOT e",
            "SELECT * FROM t2 WHERE k > 0",
        ])
        .await;
        // `isnan` is a function, which may fail (as far as the filter knows)
        // on the rows that DataFusion decides per batch to evaluate the right
        // side of `OR` on.
        assert_all_byte_eq_in_datafusion(&[
            "SELECT id, c FROM t1 WHERE c IS NULL OR c = 0.0 OR isnan(c)",
        ])
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_multi_conjunct_and_or_filters() {
        assert_all_native_byte_eq(&[
            "SELECT id, a, b FROM t1 WHERE a > 0 AND b < 5000 AND c > 0.0 AND e",
            "SELECT id, d FROM t1 WHERE a > 40 OR d = 'gamma'",
            "SELECT id FROM t1 WHERE (a > 0 OR b IS NULL) AND (d LIKE 'a%' OR h LIKE '%01%') AND NOT e",
            "SELECT id, h FROM t1 WHERE h > 'a fairly long view string 0150' AND id % 2 = 0 AND a < 30",
            "SELECT id FROM t1 WHERE b % 7 = 1 AND a + b > 100 AND c * 2.0 > 1.0",
            "SELECT id FROM t1 WHERE ((a > 0 AND b > 0) OR (a < 0 AND b < 100)) AND c IS NOT NULL",
            "SELECT id FROM t1 WHERE a BETWEEN -10 AND 10 AND d IN ('alpha', 'beta', '') AND f IS NOT NULL",
            "SELECT id, a FROM t1 WHERE CASE WHEN a > 0 THEN b > 100 ELSE c < 0.0 END AND g IS NOT NULL",
        ])
        .await;
        // A function after the first conjunct (DataFusion's optimizer moves
        // function calls after the cheap conjuncts).
        assert_all_byte_eq_in_datafusion(&[
            "SELECT id FROM t1 WHERE b % 7 = 1 AND a + b > 100 AND upper(d) = 'ALPHA'",
            "SELECT id FROM t1 WHERE upper(d) = 'ALPHA' AND b % 7 = 1 AND a + b > 100",
        ])
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_fallible_conjuncts() {
        // A fallible first conjunct is evaluated on all rows, whatever the
        // batches: these filters are native. (DataFusion's optimizer moves
        // function calls and `LIKE` after the cheap conjuncts.)
        assert_all_native_byte_eq(&[
            "SELECT id, s FROM fallible WHERE CAST(id AS INT) % 10 = 0 AND s IS NOT NULL",
            "SELECT id, x FROM fallible WHERE upper(s) <> 'X'",
            "SELECT id, x FROM fallible WHERE upper(s) <> 'X' AND s LIKE '%1%'",
            "SELECT id, x FROM fallible WHERE 10 / (x + 100) < 1 AND x <> 0",
        ])
        .await;
        // DataFusion divides only the rows with `x <> 0` because, in each
        // batch, at most 20% of the rows are left: such guards depend on the
        // batches, so the filters stay in DataFusion.
        assert_all_byte_eq_in_datafusion(&[
            "SELECT id, x FROM fallible WHERE x <> 0 AND 10 / x > 1",
            "SELECT id, x FROM fallible WHERE x <> 0 AND 10 / x > 1 AND id % 2 = 0",
            "SELECT id FROM fallible WHERE x <> 0 AND 100 % x = 0",
            // (DataFusion's optimizer moves cheap conjuncts such as casts
            // before regular expressions, so the guard must be cheap too.)
            "SELECT id, s FROM fallible WHERE id % 10 = 0 AND CAST(s AS INT) > 500",
            "SELECT id, big FROM fallible WHERE big < 100000 AND CAST(big AS INT) % 2 = 0",
        ])
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_errors_match_datafusion() {
        // Whether a native filter is used, and whether the query fails (in
        // both engines).
        for (sql, native, fails) in [
            // Dense guards: DataFusion divides all rows, including `x = 0`.
            // Guards depend on the batches: the filters stay in DataFusion.
            (
                "SELECT * FROM dense WHERE x <> 0 AND 10 / x > 1",
                false,
                true,
            ),
            (
                "SELECT id FROM dense WHERE x > 0 AND x < 100 AND 100 % x = 1",
                false,
                true,
            ),
            // A selective conjunct after the division does not guard it, even
            // once the conjuncts are reordered at runtime.
            (
                "SELECT * FROM dense WHERE 10 / x > 1 AND y = 1 AND id > 0",
                true,
                true,
            ),
            // Sparse guards: only the rows that are left are divided.
            (
                "SELECT * FROM sparse WHERE x <> 0 AND 10 / x > 1",
                false,
                false,
            ),
            (
                "SELECT id, 10 / x AS q FROM sparse WHERE x <> 0",
                true,
                false,
            ),
            (
                "SELECT id, 10 / x AS q FROM dense WHERE x <> 0",
                true,
                false,
            ),
            // `NULL` rows are not rejected: the division is evaluated on them.
            (
                "SELECT * FROM nulls WHERE a > 0 AND 10 / b > 1",
                false,
                true,
            ),
            (
                "SELECT * FROM nulls_ok WHERE a > 0 AND 10 / b > 1",
                false,
                false,
            ),
            (
                "SELECT * FROM nulls WHERE a IS NULL AND 10 / b > 1",
                false,
                true,
            ),
            // No guard.
            ("SELECT id FROM fallible WHERE 10 / x > 1", true, true),
            (
                "SELECT id FROM fallible WHERE CAST(s AS INT) > 5",
                true,
                true,
            ),
            (
                "SELECT id FROM fallible WHERE 10 / x > 1 OR x = 0",
                true,
                true,
            ),
            (
                "SELECT id FROM fallible WHERE x <> 0 AND 10 / x > 1",
                false,
                false,
            ),
        ] {
            let (error, operators) =
                assert_native_parity_with_setup(sql, guard_tables).await;
            assert_eq!(operators.filters > 0, native, "{sql}");
            assert_eq!(error.is_some(), fails, "{sql}: {error:?}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_string_view_and_dictionary_filters() {
        assert_all_native_byte_eq(&[
            "SELECT d32, i FROM vd WHERE d32 = 'k3'",
            "SELECT i, d32, sv FROM vd WHERE sv LIKE '%7%' AND d8 <> 'x2'",
            // Patterns that compile cannot fail, after other conjuncts too.
            "SELECT i, sv FROM vd WHERE i % 3 = 0 AND sv LIKE '%7%'",
            "SELECT i, sv FROM vd WHERE i % 2 = 0 AND (sv ILIKE 'A%' OR sv ~ '7$')",
            "SELECT i, d32 FROM vd WHERE d32 IN ('k1', 'k2', 'k39') OR d8 IS NULL",
            "SELECT i, sv FROM vd WHERE sv > 'short4' AND i % 3 = 0",
            "SELECT i, d32 FROM vd WHERE d8 = 'x5' AND d32 IS NOT NULL",
        ])
        .await;
        // `FilterExec` coalesces its output, merging the dictionaries of `d8`
        // (`Int8` keys), which may fail.
        assert_all_byte_eq_in_datafusion(&[
            "SELECT * FROM vd WHERE d32 = 'k3'",
            "SELECT i, d8, sv FROM vd WHERE sv LIKE '%7%' AND d8 <> 'x2'",
            "SELECT * FROM vd WHERE sv IS NULL",
        ])
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_filters_below_other_operators() {
        assert_all_native_byte_eq(&[
            "SELECT COUNT(*) FROM t1 WHERE a > 0",
            "SELECT d, COUNT(*), SUM(a) FROM t1 WHERE a > 0 AND e GROUP BY d",
            "SELECT t1.id, t2.name FROM t1 JOIN t2 ON t1.a = t2.k WHERE t1.b > 10 AND t2.w < 100",
            "SELECT t1.id, t2.k FROM t1 LEFT JOIN t2 ON t1.a = t2.k WHERE t1.id % 10 = 3",
            "SELECT id, a FROM t1 WHERE a > 0 ORDER BY id",
            "SELECT id, h FROM t1 WHERE h LIKE '%1%' ORDER BY h DESC, id LIMIT 20",
            "SELECT * FROM (SELECT * FROM t1 ORDER BY id LIMIT 100) WHERE a > 0",
            "SELECT id FROM t1 WHERE a > 40 UNION ALL SELECT k FROM t2 WHERE k < 0",
            "SELECT id, ROW_NUMBER() OVER (PARTITION BY d ORDER BY id) AS rn FROM t1 WHERE a > 0",
            "SELECT d, MAX(h) FROM t1 WHERE id % 3 = 0 GROUP BY d HAVING COUNT(*) > 100",
        ])
        .await;
        // The limit is pushed into the filter (`fetch = 7`), which is lowered:
        // with a fetch, it runs on one thread, and pulls its input like
        // `FilterExec`. Below it, the merge of the sorted partitions, which
        // stops early, stays in DataFusion with its input (see
        // `plans_below_limits_stay_in_datafusion`).
        let sql = "SELECT * FROM (SELECT id, a FROM t1 ORDER BY id LIMIT 500) WHERE a > 0 LIMIT 7";
        let run = assert_native_byte_eq_with_setup(sql, all_tables).await;
        assert_eq!(
            (run.operators.filters, run.operators.scans),
            (1, 0),
            "{sql}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_filters_on_any_number_of_threads() {
        for target_partitions in [1, 3, 16] {
            let config = test_session_config().with_target_partitions(target_partitions);
            for sql in [
                "SELECT id, a, h FROM t1 WHERE a > 10 AND d <> 'beta'",
                "SELECT id FROM t1 WHERE id % 1000 = 7",
                "SELECT id, a FROM t1 WHERE a > 0 ORDER BY id",
                "SELECT * FROM (SELECT id, a FROM t1 ORDER BY id LIMIT 500) WHERE a > 0",
                "SELECT id, s FROM fallible WHERE CAST(id AS INT) % 10 = 0 AND s IS NOT NULL",
            ] {
                let run =
                    assert_native_byte_eq_with_config(sql, config.clone(), all_tables)
                        .await;
                assert!(run.operators.filters > 0, "{sql}");
            }
        }
    }

    /// A plan `FilterExec(a % 2 = 0, fetch = 3)` over `partitions`
    /// partitions of 20 rows each.
    fn filter_with_fetch(partitions: usize) -> Arc<dyn ExecutionPlan> {
        let schema = Arc::new(Schema::new(vec![Field::new("a", DataType::Int64, false)]));
        let batch = |start: i64| {
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(Int64Array::from_iter_values(start..start + 20))],
            )
            .unwrap()
        };
        let partitions = (0..partitions as i64)
            .map(|p| vec![batch(p * 100)])
            .collect::<Vec<_>>();
        let input = datafusion_datasource::memory::MemorySourceConfig::try_new_exec(
            &partitions,
            Arc::clone(&schema),
            None,
        )
        .unwrap();
        let predicate = physical_expr("a % 2 = 0", &schema);
        Arc::new(
            datafusion_physical_plan::filter::FilterExecBuilder::new(predicate, input)
                .with_fetch(Some(3))
                .build()
                .unwrap(),
        )
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn per_partition_fetch_is_kept() {
        // DataFusion produces 3 rows per partition: 6 rows for 2 partitions.
        // The chunks of a pipeline do not belong to partitions, so such a
        // `FilterExec` stays in DataFusion.
        let plan = filter_with_fetch(2);
        let exec = plan.downcast_ref::<FilterExec>().unwrap();
        assert!(!PhysicalFilter::supports_filter_exec(exec));
        let err =
            PhysicalFilter::try_from_filter_exec(exec, lower(exec.input()).unwrap())
                .unwrap_err();
        assert!(
            err.to_string().contains("each of its 2 input partitions"),
            "{err}"
        );
        let run = assert_native_plan_byte_eq(|| filter_with_fetch(2), false).await;
        assert_eq!(run.operators.filters, 0);
        let rows = run
            .actual
            .batches
            .iter()
            .map(RecordBatch::num_rows)
            .sum::<usize>();
        assert_eq!(rows, 6);

        // With a single partition, the native filter produces the first 3
        // rows, in order, on a single thread.
        let plan = filter_with_fetch(1);
        let exec = plan.downcast_ref::<FilterExec>().unwrap();
        assert!(PhysicalFilter::supports_filter_exec(exec));
        let filter =
            PhysicalFilter::try_from_filter_exec(exec, lower(exec.input()).unwrap())
                .unwrap();
        assert!(!filter.parallel_operator());
        assert!(new_filter("a > 0", None).parallel_operator());
        let run = assert_native_plan_byte_eq(|| filter_with_fetch(1), true).await;
        assert_eq!(run.operators.filters, 1);
        let rows = run
            .actual
            .batches
            .iter()
            .map(RecordBatch::num_rows)
            .sum::<usize>();
        assert_eq!(rows, 3);

        // In SQL, a limit above the filter enforces the fetch over all
        // partitions: the filter stays in DataFusion, with the same result.
        for sql in [
            "SELECT id FROM t1 WHERE id < 5 LIMIT 100",
            "SELECT id, a FROM t1 WHERE a > 1000 LIMIT 10",
        ] {
            let run = assert_native_byte_eq_with_setup(sql, all_tables).await;
            assert_eq!(run.operators.filters, 0, "{sql}");
        }
    }

    /// A batch of a column `x` with the runs `(value, rows)`.
    fn runs(runs: &[(i32, usize)]) -> RecordBatch {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, true)]));
        let x = runs
            .iter()
            .flat_map(|&(value, rows)| std::iter::repeat_n(value, rows))
            .collect::<Int32Array>();
        RecordBatch::try_new(schema, vec![Arc::new(x)]).unwrap()
    }

    /// A scan of `batch`, in a single partition.
    fn scan(batch: &RecordBatch) -> Arc<dyn ExecutionPlan> {
        datafusion_datasource::memory::MemorySourceConfig::try_new_exec(
            &[vec![batch.clone()]],
            batch.schema(),
            None,
        )
        .unwrap()
    }

    fn filter_exec(
        predicate: &str,
        input: Arc<dyn ExecutionPlan>,
        fetch: Option<usize>,
    ) -> Arc<dyn ExecutionPlan> {
        let predicate = physical_expr(predicate, &input.schema());
        Arc::new(
            datafusion_physical_plan::filter::FilterExecBuilder::new(predicate, input)
                .with_fetch(fetch)
                .build()
                .unwrap(),
        )
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn batch_dependent_predicates_stay_in_datafusion() {
        // One batch of 4096 rows, 1848 zeros then 2248 ones. Split into two
        // chunks, 90% of the rows of the first chunk are zeros, but only 45%
        // of the batch: DataFusion divides all rows by `x` and fails.
        let batch = runs(&[(0, 1848), (1, 2248)]);
        for predicate in ["x <> 0 AND 10 / x > 1", "x = 0 OR 10 / x > 1"] {
            let plan = filter_exec(predicate, scan(&batch), None);
            let exec = plan.downcast_ref::<FilterExec>().unwrap();
            assert!(!PhysicalFilter::supports_filter_exec(exec), "{predicate}");
            let err =
                PhysicalFilter::try_from_filter_exec(exec, lower(exec.input()).unwrap())
                    .unwrap_err();
            assert!(
                err.to_string().contains("depend on DataFusion's batches"),
                "{err}"
            );

            let (error, operators) = assert_native_plan_parity(
                || filter_exec(predicate, scan(&batch), None),
                false,
            )
            .await;
            assert!(error.unwrap().contains("Divide by zero"), "{predicate}");
            assert_eq!(operators.filters, 0, "{predicate}");

            // On the chunks of the pipeline, the predicate does not fail.
            let filter = filter_for(predicate, &batch.schema());
            assert!(filter.errors_depend_on_batches());
            let chunks =
                [batch.slice(0, 2048), batch.slice(2048, 2048)].map(DataChunk::from);
            assert!(drive(&filter, chunks).is_ok(), "{predicate}");
        }
        // Predicates whose errors do not depend on the batches are lowered.
        for (predicate, fails) in [("10 / x > 1 AND x <> 0", true), ("x <> 0", false)] {
            let (error, operators) = assert_native_plan_parity(
                || filter_exec(predicate, scan(&batch), None),
                false,
            )
            .await;
            assert_eq!(error.is_some(), fails, "{predicate}");
            assert_eq!(operators.filters, 1, "{predicate}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn opaque_columns_see_all_input_columns() {
        // `opaque(x)` reads `x` without a `Column` child: bound to the columns
        // that the predicate references (`y`, or none), it would read `y` as
        // `x`, or no column at all.
        let batch = xys(&[((0, 1), 100), ((2, 2), 100), ((3, 0), 100)]);
        let schema = batch.schema();
        let one: Arc<dyn PhysicalExpr> =
            Arc::new(datafusion_physical_expr::expressions::Literal::new(
                ScalarValue::Int32(Some(1)),
            ));
        for (left, op, right) in [
            (
                physical_expr("y", &schema),
                datafusion_expr::Operator::Eq,
                opaque_column("x", &schema),
            ),
            (
                opaque_column("x", &schema),
                datafusion_expr::Operator::Gt,
                one,
            ),
        ] {
            let predicate: Arc<dyn PhysicalExpr> = Arc::new(
                datafusion_physical_expr::expressions::BinaryExpr::new(left, op, right),
            );
            let plan = || -> Arc<dyn ExecutionPlan> {
                Arc::new(
                    FilterExec::try_new(Arc::clone(&predicate), scan(&batch)).unwrap(),
                )
            };
            let (error, operators) = assert_native_plan_parity(plan, false).await;
            assert!(error.is_none(), "{predicate}: {error:?}");
            assert_eq!(operators.filters, 1, "{predicate}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn case_insensitive_patterns_that_fail_on_unicode_stay_in_datafusion() {
        // One batch of 4096 rows, `(x, s) = (1, "a")` then `(0, "é")`. On
        // strings that are all ASCII, Arrow matches an ASCII `ILIKE` pattern
        // without wildcards without a regular expression; on others, it
        // compiles one, here of 100000 `s` (each `[sSſ]`), which is too big.
        // DataFusion evaluates the `ILIKE` on all rows of the batch and fails;
        // the native filter only on the first chunk, which is all ASCII.
        let pattern = "s".repeat(100_000);
        let batch = xss(&[((1, "a"), 2048), ((0, "é"), 2048)]);
        let predicate = format!("x > 0 AND s ILIKE '{pattern}'");
        let plan = || filter_exec(&predicate, scan(&batch), None);
        let filter_plan = plan();
        let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
        assert!(!PhysicalFilter::supports_filter_exec(exec));
        let (error, operators) = assert_native_plan_parity(plan, false).await;
        let error = error.unwrap();
        assert!(error.contains("Unable to build regex"), "{error}");
        assert_eq!(operators.filters, 0);
        // On the chunks of the pipeline, the predicate does not fail.
        let filter = filter_for(&predicate, &batch.schema());
        assert!(filter.errors_depend_on_batches());
        let chunks = [batch.slice(0, 2048), batch.slice(2048, 2048)].map(DataChunk::from);
        assert!(drive(&filter, chunks).is_ok());
        // Patterns that compile are lowered.
        for predicate in ["x > 0 AND s ILIKE 'A%'", "x > 0 AND s NOT ILIKE '%É'"] {
            let (error, operators) = assert_native_plan_parity(
                || filter_exec(predicate, scan(&batch), None),
                false,
            )
            .await;
            assert!(error.is_none(), "{predicate}");
            assert_eq!(operators.filters, 1, "{predicate}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn volatile_predicates_are_not_evaluated_before_execution() {
        use datafusion_physical_expr::expressions::{BinaryExpr, LikeExpr, lit};
        // `count_calls()` returns how often it was called: DataFusion calls it
        // once for the batch of 3 rows, and selects them all. Evaluating it
        // before (on an empty batch, to find whether it fails without rows, or
        // to compile the pattern of a `LIKE`) would change its results.
        let batch = runs(&[(1, 3)]);
        let schema = batch.schema();
        /// A predicate of `count_calls()`.
        type Predicate = fn(Arc<dyn PhysicalExpr>) -> Arc<dyn PhysicalExpr>;
        let predicates: [Predicate; 3] = [
            |calls| {
                Arc::new(BinaryExpr::new(
                    calls,
                    datafusion_expr::Operator::Eq,
                    lit("1"),
                ))
            },
            |calls| Arc::new(LikeExpr::new(false, false, calls, lit("1%"))),
            |calls| Arc::new(LikeExpr::new(false, true, calls, lit("1"))),
        ];
        let filter = |predicate: Arc<dyn PhysicalExpr>| -> Arc<dyn ExecutionPlan> {
            Arc::new(FilterExec::try_new(predicate, scan(&batch)).unwrap())
        };
        for predicate in predicates {
            let calls = Arc::new(AtomicUsize::new(0));
            let expr = predicate(count_calls(&calls, &schema));
            let plan = filter(Arc::clone(&expr));
            let exec = plan.downcast_ref::<FilterExec>().unwrap();
            assert!(!PhysicalFilter::supports_filter_exec(exec), "{expr}");
            assert!(depends_on_batches(&expr, &schema), "{expr}");
            let _ = crate::operators::adaptive_filter::fallibility(&expr, &schema);
            AdaptiveFilter::try_new(&expr, &schema).unwrap();
            lower(&plan).unwrap();
            assert_eq!(calls.load(Ordering::SeqCst), 0, "{expr}");

            let (error, operators) = assert_native_plan_parity(
                || filter(predicate(count_calls(&Arc::default(), &schema))),
                false,
            )
            .await;
            assert!(error.is_none(), "{expr}");
            assert_eq!(operators.filters, 0, "{expr}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn case_insensitive_patterns_on_bytes_beneath_nulls_stay_in_datafusion() {
        // `x = 0` selects every other row, the first one too, whose string is
        // null above bytes that are not ASCII. DataFusion's `filter` keeps
        // these bytes: Arrow finds that the strings are not all ASCII and
        // compiles the pattern of the `ILIKE`, which is too big. The native
        // filter gathers the selected rows with `take`, which drops them: the
        // strings are ASCII, and the pattern is not compiled.
        let batch = non_ascii_beneath_a_null();
        let schema = batch.schema();
        let predicate = format!("s ILIKE '{}'", "s".repeat(100_000));
        let plan =
            || filter_exec(&predicate, filter_exec("x = 0", scan(&batch), None), None);
        let filter_plan = plan();
        let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
        assert!(!PhysicalFilter::supports_filter_exec(exec));
        let (error, operators) = assert_native_plan_parity(plan, false).await;
        let error = error.unwrap();
        assert!(error.contains("Unable to build regex"), "{error}");
        assert_eq!(operators.filters, 0);
        let selected = DataChunk::from(batch.clone())
            .with_selection(SelectionVector::from(
                (0..2048).step_by(2).collect::<Vec<_>>(),
            ))
            .unwrap();
        assert!(drive(&filter_for(&predicate, &schema), [selected]).is_ok());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn fetch_with_comparisons_that_arrow_cannot_evaluate_stays_in_datafusion() {
        // Arrow cannot compare dictionaries of dictionaries. With `fetch = 0`,
        // `FilterExec` evaluates its predicate on the first batch, and fails.
        let batch = nested_dictionaries();
        let schema = batch.schema();
        let plan = || -> Arc<dyn ExecutionPlan> {
            let predicate =
                Arc::new(datafusion_physical_expr::expressions::BinaryExpr::new(
                    physical_expr("a", &schema),
                    datafusion_expr::Operator::Eq,
                    physical_expr("b", &schema),
                ));
            Arc::new(
                datafusion_physical_plan::filter::FilterExecBuilder::new(
                    predicate,
                    scan(&batch),
                )
                .with_fetch(Some(0))
                .build()
                .unwrap(),
            )
        };
        let filter_plan = plan();
        let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
        assert!(!PhysicalFilter::supports_filter_exec(exec));
        let (error, operators) = assert_native_plan_parity(plan, true).await;
        let error = error.unwrap();
        assert!(error.contains("Invalid comparison operation"), "{error}");
        assert_eq!(operators.filters, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn fetch_with_case_insensitive_patterns_stays_in_datafusion() {
        // With `fetch = 0`, `FilterExec` evaluates its predicate on the first
        // batch, whose strings are not all ASCII: Arrow compiles the pattern of
        // `ILIKE`, which is too big, and fails (see
        // `case_insensitive_patterns_that_fail_on_unicode_stay_in_datafusion`).
        let batch = xss(&[((1, "a"), 2048), ((0, "é"), 2048)]);
        let predicate = format!("s ILIKE '{}'", "s".repeat(100_000));
        let plan = || filter_exec(&predicate, scan(&batch), Some(0));
        let filter_plan = plan();
        let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
        assert!(!PhysicalFilter::supports_filter_exec(exec));
        let (error, operators) = assert_native_plan_parity(plan, true).await;
        let error = error.unwrap();
        assert!(error.contains("Unable to build regex"), "{error}");
        assert_eq!(operators.filters, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn filters_whose_output_dictionaries_may_overflow_stay_in_datafusion() {
        // One partition of two batches of 100 rows, with 100 distinct values
        // each, in dictionaries with `Int8` keys. `FilterExec` coalesces them
        // into one batch, and merging their 200 values overflows the keys.
        // The native filter does not coalesce its output: it selected the 200
        // rows.
        let batches = distinct_dictionary_batches(&DataType::Int8);
        let schema = batches[0].schema();
        let predicate = "d <> 'missing'";
        let plan = || filter_exec(predicate, scan_batches(&schema, &batches), None);
        let filter_plan = plan();
        let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
        assert!(!PhysicalFilter::supports_filter_exec(exec));
        let err =
            PhysicalFilter::try_from_filter_exec(exec, lower(exec.input()).unwrap())
                .unwrap_err();
        assert!(err.to_string().contains("dictionaries"), "{err}");
        let overflow = "Dictionary key bigger than the key type";
        let (error, operators) = assert_native_plan_parity(plan, false).await;
        assert!(error.unwrap().contains(overflow));
        assert_eq!(operators.filters, 0);
        let filter = filter_for(predicate, &schema);
        assert!(drive(&filter, batches.iter().cloned().map(DataChunk::from)).is_ok());

        // The same in SQL, on one target partition (with more, DataFusion
        // distributes the batches to several filters).
        let setup = async |ctx: &datafusion::prelude::SessionContext| {
            register_partitions(ctx, "narrow", vec![batches.clone()])
        };
        let (error, operators) = assert_native_parity_with_config(
            "SELECT * FROM narrow WHERE d <> 'missing'",
            test_session_config().with_target_partitions(1),
            setup,
        )
        .await;
        assert!(error.unwrap().contains(overflow));
        assert_eq!(operators.filters, 0);

        // Without the dictionaries in its output, or with keys of 32 bits,
        // which can index the values of any batch, the filter is lowered.
        let projected = || -> Arc<dyn ExecutionPlan> {
            Arc::new(
                datafusion_physical_plan::filter::FilterExecBuilder::new(
                    physical_expr(predicate, &schema),
                    scan_batches(&schema, &batches),
                )
                .apply_projection(Some(vec![1]))
                .unwrap()
                .build()
                .unwrap(),
            )
        };
        let (error, operators) = assert_native_plan_parity(projected, false).await;
        assert!(error.is_none());
        assert_eq!(operators.filters, 1);
        let wide = distinct_dictionary_batches(&DataType::Int32);
        let (error, operators) = assert_native_plan_parity(
            || filter_exec(predicate, scan_batches(&wide[0].schema(), &wide), None),
            false,
        )
        .await;
        assert!(error.is_none());
        assert_eq!(operators.filters, 1);
    }

    #[test]
    fn finds_outputs_that_may_fail_to_coalesce() {
        let utf8 = |keys: DataType| {
            DataType::Dictionary(Box::new(keys), Box::new(DataType::Utf8))
        };
        let field = |data_type: DataType| Arc::new(Field::new("f", data_type, true));
        for (data_type, fails) in [
            (utf8(DataType::Int8), true),
            (utf8(DataType::UInt16), true),
            (utf8(DataType::Int32), false),
            (utf8(DataType::UInt64), false),
            (DataType::List(field(utf8(DataType::Int8))), true),
            // Lists of more than `i32::MAX` values overflow their offsets (lists
            // of nulls take no memory).
            (DataType::List(field(DataType::Null)), true),
            (DataType::List(field(DataType::Int32)), true),
            (DataType::ListView(field(DataType::Int32)), true),
            (DataType::LargeList(field(DataType::Null)), false),
            (DataType::LargeList(field(utf8(DataType::Int32))), false),
            (DataType::LargeList(field(utf8(DataType::Int8))), true),
            (
                DataType::Struct(vec![field(utf8(DataType::UInt8))].into()),
                true,
            ),
            (
                DataType::RunEndEncoded(
                    Arc::new(Field::new("run_ends", DataType::Int16, false)),
                    Arc::new(Field::new("values", DataType::Utf8, true)),
                ),
                true,
            ),
            (
                DataType::RunEndEncoded(
                    Arc::new(Field::new("run_ends", DataType::Int32, false)),
                    Arc::new(Field::new("values", DataType::Utf8, true)),
                ),
                false,
            ),
            (DataType::Utf8, false),
            // Arrow's `filter` renames the fields of run-end encoded arrays.
            (
                DataType::RunEndEncoded(
                    Arc::new(Field::new("ends", DataType::Int32, false)),
                    field(DataType::Utf8),
                ),
                true,
            ),
            (
                DataType::RunEndEncoded(
                    Arc::new(Field::new("run_ends", DataType::Int64, false)),
                    Arc::new(Field::new("values", DataType::Utf8, false)),
                ),
                true,
            ),
            (
                DataType::RunEndEncoded(
                    Arc::new(Field::new("run_ends", DataType::Int64, false)),
                    Arc::new(Field::new("values", DataType::Utf8, true)),
                ),
                false,
            ),
        ] {
            assert_eq!(may_fail_to_coalesce(&data_type), fails, "{data_type}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn in_lists_with_fallible_values_stay_in_datafusion() {
        // One batch of 4096 rows, `(x, y) = (1, 0)` then `(2, 1)`. `x = 1` is
        // not true on all rows of the batch, so DataFusion evaluates `10 / y`
        // on all of them, and fails. But it is true on all rows of the first
        // chunk of the pipeline, and `y = 1` on all rows of the second one.
        let batch = xys(&[((1, 0), 2048), ((2, 1), 2048)]);
        for predicate in ["x IN (1, 10 / y)", "x NOT IN (1, 10 / y)"] {
            let plan = filter_exec(predicate, scan(&batch), None);
            let exec = plan.downcast_ref::<FilterExec>().unwrap();
            assert!(!PhysicalFilter::supports_filter_exec(exec), "{predicate}");
            let (error, operators) = assert_native_plan_parity(
                || filter_exec(predicate, scan(&batch), None),
                false,
            )
            .await;
            assert!(error.unwrap().contains("Divide by zero"), "{predicate}");
            assert_eq!(operators.filters, 0, "{predicate}");

            // On the chunks of the pipeline, the predicate does not fail.
            let filter = filter_for(predicate, &batch.schema());
            assert!(filter.errors_depend_on_batches());
            let chunks =
                [batch.slice(0, 2048), batch.slice(2048, 2048)].map(DataChunk::from);
            assert!(drive(&filter, chunks).is_ok(), "{predicate}");
        }
        // The first value is evaluated on all rows.
        let (error, operators) = assert_native_plan_parity(
            || filter_exec("x IN (10 / y, 1)", scan(&batch), None),
            false,
        )
        .await;
        assert!(error.unwrap().contains("Divide by zero"));
        assert_eq!(operators.filters, 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn patterns_that_fail_behind_short_circuits_stay_in_datafusion() {
        // One batch of 4096 rows, `(x, y) = (0, 1)` then `(1, 0)`. Neither
        // `x = 0` nor `y = 0` is true on all rows of the batch, so DataFusion
        // evaluates `s ~ 'a{1000000}'`, whose pattern is too big to compile.
        // But `x = 0` is true on all rows of the first chunk of the pipeline,
        // and `y = 0` on all rows of the second one.
        let batch = xys(&[((0, 1), 2048), ((1, 0), 2048)]);
        let predicate = "x = 0 OR (y = 0 OR s ~ 'a{1000000}')";
        let plan = filter_exec(predicate, scan(&batch), None);
        let exec = plan.downcast_ref::<FilterExec>().unwrap();
        assert!(!PhysicalFilter::supports_filter_exec(exec));
        let (error, operators) = assert_native_plan_parity(
            || filter_exec(predicate, scan(&batch), None),
            false,
        )
        .await;
        assert!(error.unwrap().contains("did not compile"));
        assert_eq!(operators.filters, 0);

        // On the chunks of the pipeline, the predicate does not fail.
        let filter = filter_for(predicate, &batch.schema());
        assert!(filter.errors_depend_on_batches());
        let chunks = [batch.slice(0, 2048), batch.slice(2048, 2048)].map(DataChunk::from);
        assert!(drive(&filter, chunks).is_ok());

        // A pattern that compiles cannot fail.
        let predicate = "x = 0 OR (y = 0 OR s ~ 'a+')";
        let (error, operators) = assert_native_plan_parity(
            || filter_exec(predicate, scan(&batch), None),
            false,
        )
        .await;
        assert!(error.is_none());
        assert_eq!(operators.filters, 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn predicates_that_fail_on_empty_batches_stay_in_datafusion() {
        // DataFusion evaluates the predicate on every batch, empty ones
        // included, on which `OR` evaluates its right side: after an empty
        // batch, `x = 0 OR s ~ 'a{1000000}'` fails even if `x = 0` on all
        // rows. Without batches, it is not evaluated at all.
        let schema = xys(&[]).schema();
        let empty = RecordBatch::new_empty(Arc::clone(&schema));
        let zeros = xys(&[((0, 0), 2048)]);
        for (predicate, batches, fails) in [
            (
                "x = 0 OR s ~ 'a{1000000}'",
                vec![empty.clone(), zeros.clone()],
                true,
            ),
            ("x = 0 OR s ~ 'a{1000000}'", vec![zeros.clone()], false),
            ("x = 0 OR s ~ 'a{1000000}'", vec![], false),
            // Functions may fail on empty batches too.
            ("regexp_like(s, 'a{1000000}')", vec![empty.clone()], true),
            ("regexp_like(s, 'a{1000000}')", vec![], false),
        ] {
            let plan = || filter_exec(predicate, scan_batches(&schema, &batches), None);
            let filter_plan = plan();
            let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
            assert!(!PhysicalFilter::supports_filter_exec(exec), "{predicate}");
            let (error, operators) = assert_native_plan_parity(plan, false).await;
            let input = format!("{predicate} on {} batches", batches.len());
            assert_eq!(error.is_some(), fails, "{input}: {error:?}");
            assert_eq!(operators.filters, 0, "{input}");
        }
        // A pattern that compiles cannot fail, with or without empty batches,
        // and `regexp_count` only compiles its pattern if there are rows.
        for (predicate, batches, fails) in [
            (
                "x = 0 OR s ~ 'a+'",
                vec![empty.clone(), zeros.clone()],
                false,
            ),
            ("x = 0 OR s ~ 'a+'", vec![zeros.clone()], false),
            ("x = 0 OR s ~ 'a+'", vec![], false),
            (
                "regexp_count(s, 'a{1000000}') > 0",
                vec![empty.clone()],
                false,
            ),
            (
                "regexp_count(s, 'a{1000000}') > 0",
                vec![empty, zeros],
                true,
            ),
        ] {
            let (error, operators) = assert_native_plan_parity(
                || filter_exec(predicate, scan_batches(&schema, &batches), None),
                false,
            )
            .await;
            let input = format!("{predicate} on {} batches", batches.len());
            assert_eq!(error.is_some(), fails, "{input}: {error:?}");
            assert_eq!(operators.filters, 1, "{input}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn predicates_on_dictionary_values_stay_in_datafusion() {
        // The first `FilterExec` coalesces the rows `"1"` and `"2"` into one
        // batch, merging the dictionaries of the two batches without the
        // values that no row references, `"x"` and `"y"`, so DataFusion casts
        // `"1"` and `"2"`. The chunks of a pipeline keep their dictionaries:
        // casting `"x"` fails.
        let batches = dictionary_batches();
        let schema = batches[0].schema();
        let plan = || {
            let input = filter_exec(
                "d <> 'x' AND d <> 'y'",
                scan_batches(&schema, &batches),
                None,
            );
            filter_exec("CAST(d AS INT) > 0", input, None)
        };
        let filter_plan = plan();
        let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
        assert!(!PhysicalFilter::supports_filter_exec(exec));
        let (error, operators) = assert_native_plan_parity(plan, false).await;
        assert!(error.is_none());
        assert_eq!(operators.filters, 0);
        let chunk = DataChunk::from(batches[0].clone())
            .with_selection(SelectionVector::from(vec![0]))
            .unwrap();
        assert!(drive(&filter_for("CAST(d AS INT) > 0", &schema), [chunk]).is_err());
    }

    /// `GlobalLimitExec(1)` above `FilterExec(q >= 0)`, coalescing its output
    /// into batches of `filter_batch_size` rows, above
    /// `ProjectionExec(10 / x AS q)`, over a batch of `rows` ones and a batch
    /// of `rows` zeros.
    fn limited_filter(rows: usize, filter_batch_size: usize) -> Arc<dyn ExecutionPlan> {
        let (ones, zeros) = (runs(&[(1, rows)]), runs(&[(0, rows)]));
        let schema = ones.schema();
        let projection = Arc::new(
            datafusion_physical_plan::projection::ProjectionExec::try_new(
                [datafusion_physical_expr::projection::ProjectionExpr::new(
                    physical_expr("10 / x", &schema),
                    "q",
                )],
                scan_batches(&schema, &[ones, zeros]),
            )
            .unwrap(),
        );
        let predicate = physical_expr("q >= 0", &projection.schema());
        let filter = datafusion_physical_plan::filter::FilterExecBuilder::new(
            predicate, projection,
        )
        .with_batch_size(filter_batch_size)
        .build()
        .unwrap();
        Arc::new(datafusion_physical_plan::limit::GlobalLimitExec::new(
            Arc::new(filter),
            0,
            Some(1),
        ))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn plans_below_limits_stay_in_datafusion() {
        // `FilterExec` pulls its input until it has a batch of its batch
        // size, or its input ends: with batches of 8192 rows, it pulls the
        // zeros, which the projection divides by, before the limit is reached.
        // A native filter, which does not coalesce its output, reaches it with
        // the ones.
        let (error, operators) =
            assert_native_plan_parity(|| limited_filter(2048, 8192), true).await;
        assert!(error.unwrap().contains("Divide by zero"));
        assert_eq!((operators.filters, operators.projections), (0, 0));
        // With batches of 32 rows, `FilterExec` returns the ones, and the limit
        // is reached before it pulls the zeros. The native cache keeps the 32
        // rows of a chunk, and pulls the zeros.
        let (error, operators) =
            assert_native_plan_parity(|| limited_filter(32, 32), true).await;
        assert!(error.is_none());
        assert_eq!((operators.filters, operators.projections), (0, 0));
    }

    /// `FilterExec(q >= 0)`, coalescing its output into batches of 32 rows,
    /// above `ProjectionExec(id, 10 / x AS q)`, over a batch of 32 rows with
    /// `x = 1` then a batch of 32 rows with `x = 0`, ordered by `id` (`0` to
    /// `63`), which the scan declares if `ordered`.
    fn filtered_divisions(ordered: bool) -> Arc<dyn ExecutionPlan> {
        use datafusion_physical_expr::projection::ProjectionExpr;
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("x", DataType::Int32, false),
        ]));
        let batch = |first: i64, x: i32| {
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from_iter_values(first..first + 32)),
                    Arc::new(Int32Array::from_iter_values(std::iter::repeat_n(x, 32))),
                ],
            )
            .unwrap()
        };
        let mut source = datafusion_datasource::memory::MemorySourceConfig::try_new(
            &[vec![batch(0, 1), batch(32, 0)]],
            Arc::clone(&schema),
            None,
        )
        .unwrap();
        if ordered {
            let id = LexOrdering::new([PhysicalSortExpr::new_default(physical_expr(
                "id", &schema,
            ))])
            .unwrap();
            source = source.try_with_sort_information(vec![id]).unwrap();
        }
        let projection = Arc::new(
            datafusion_physical_plan::projection::ProjectionExec::try_new(
                [
                    ProjectionExpr::new(physical_expr("id", &schema), "id"),
                    ProjectionExpr::new(physical_expr("10 / x", &schema), "q"),
                ],
                datafusion_datasource::source::DataSourceExec::from_data_source(source),
            )
            .unwrap(),
        );
        let predicate = physical_expr("q >= 0", &projection.schema());
        Arc::new(
            datafusion_physical_plan::filter::FilterExecBuilder::new(
                predicate, projection,
            )
            .with_batch_size(32)
            .build()
            .unwrap(),
        )
    }

    /// A sort of [`filtered_divisions`] by `keys`, with `fetch`.
    fn sorted_divisions(
        keys: &[&str],
        fetch: Option<usize>,
        ordered: bool,
    ) -> Arc<dyn ExecutionPlan> {
        let input = filtered_divisions(ordered);
        let ordering = LexOrdering::new(keys.iter().map(|key| {
            PhysicalSortExpr::new_default(physical_expr(key, &input.schema()))
        }))
        .unwrap();
        Arc::new(SortExec::new(ordering, input).with_fetch(fetch))
    }

    /// `GlobalLimitExec(1)` above `input`.
    fn limit_one(input: Arc<dyn ExecutionPlan>) -> Arc<dyn ExecutionPlan> {
        Arc::new(datafusion_physical_plan::limit::GlobalLimitExec::new(
            input,
            0,
            Some(1),
        ))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn sorts_that_may_stop_early_keep_their_input_in_datafusion() {
        // The input of the sort is ordered by `id`. A sort by `id` streams it,
        // and a sort by `(id, q)` with a fetch (a TopK) stops after a batch
        // once it has its rows and the next `id`s are greater: both stop
        // before `FilterExec` pulls the zeros, which the projection divides
        // by. A native cache, which keeps the 32 rows of the first chunk,
        // would pull them.
        let stopping: [fn() -> Arc<dyn ExecutionPlan>; 2] = [
            || limit_one(sorted_divisions(&["id"], None, true)),
            || sorted_divisions(&["id", "q"], Some(1), true),
        ];
        for plan in stopping {
            let (error, operators) = assert_native_plan_parity(plan, true).await;
            assert!(error.is_none(), "{error:?}");
            assert_eq!((operators.filters, operators.projections), (0, 0));
        }
        // A sort whose order its input does not satisfy reads all of its
        // input, in DataFusion as natively: what is below it is lowered.
        let reading: [fn() -> Arc<dyn ExecutionPlan>; 2] = [
            || limit_one(sorted_divisions(&["id"], None, false)),
            || sorted_divisions(&["id", "q"], Some(1), false),
        ];
        for plan in reading {
            let (error, operators) = assert_native_plan_parity(plan, true).await;
            assert!(error.unwrap().contains("Divide by zero"));
            assert_eq!(operators.filters, 1);
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn casts_that_arrow_cannot_evaluate_stay_in_datafusion() {
        // On one batch of `(x, y) = (0, 1)` then `(1, 0)` rows, DataFusion
        // evaluates the cast, as neither `x = 0` nor `y = 0` is true on all of
        // its rows, and fails; on the chunks of a pipeline, `x = 0` and then
        // `y = 0` are true on all rows.
        let mixed = xys(&[((0, 1), 2048), ((1, 0), 2048)]);
        let schema = mixed.schema();
        // DataFusion also evaluates it on an empty batch, which a pipeline
        // skips.
        let empty = RecordBatch::new_empty(Arc::clone(&schema));
        let nested = || nested_packed_booleans(&schema);
        let alone = || -> Arc<dyn PhysicalExpr> {
            Arc::new(datafusion_physical_expr::expressions::IsNotNullExpr::new(
                packed_booleans(&schema),
            ))
        };
        for (predicate, batch) in [
            (&nested as &dyn Fn() -> Arc<dyn PhysicalExpr>, &mixed),
            (&alone, &empty),
        ] {
            let plan = || -> Arc<dyn ExecutionPlan> {
                Arc::new(
                    FilterExec::try_new(
                        predicate(),
                        scan_batches(&schema, std::slice::from_ref(batch)),
                    )
                    .unwrap(),
                )
            };
            let filter_plan = plan();
            let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
            assert!(!PhysicalFilter::supports_filter_exec(exec));
            let (error, operators) = assert_native_plan_parity(plan, false).await;
            let error = error.unwrap();
            assert!(error.contains("dictionary packing"), "{error}");
            assert_eq!(operators.filters, 0);
        }
        // In SQL, the `OR`s are reassociated, but the filter and the
        // projection stay in DataFusion.
        let setup = async |ctx: &datafusion::prelude::SessionContext| {
            register_partitions(ctx, "t", vec![vec![mixed.clone()]])
        };
        let predicate = "x = 0 OR (y = 0 OR \
            arrow_cast(x = 0, 'Dictionary(Int32, Boolean)') IS NOT NULL)";
        for sql in [
            format!("SELECT * FROM t WHERE {predicate}"),
            format!("SELECT {predicate} AS m FROM t"),
        ] {
            let (_, operators) = assert_native_parity_with_setup(&sql, setup).await;
            assert_eq!(operators.filters + operators.projections, 0, "{sql}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn filters_whose_output_lists_may_overflow_stay_in_datafusion() {
        // `FilterExec` coalesces the first 256 rows of both batches into one,
        // whose lists have `2 << 30` nulls: more than their 32-bit offsets can
        // count, and Arrow's `concat` panics. The native filter does not
        // coalesce its output: it returned the 512 rows.
        let batches = long_lists_of_nulls();
        let schema = batches[0].schema();
        let plan = filter_exec("x > 0", scan_batches(&schema, &batches), None);
        let exec = plan.downcast_ref::<FilterExec>().unwrap();
        assert!(!PhysicalFilter::supports_filter_exec(exec));
        let task_ctx = Arc::new(TaskContext::default());
        let overflow = assert_same_outcome_in_datafusion(plan, task_ctx).await;
        assert!(overflow.unwrap_err().contains("offset overflow"));

        // The same in SQL, on one target partition.
        let config = test_session_config().with_target_partitions(1);
        let ctx = crate::test_utils::datafusion_context(config);
        register_partitions(&ctx, "lists", vec![batches]).unwrap();
        let df = ctx.sql("SELECT * FROM lists WHERE x > 0").await.unwrap();
        let plan = df.create_physical_plan().await.unwrap();
        let overflow = assert_same_outcome_in_datafusion(plan, ctx.task_ctx()).await;
        assert!(overflow.unwrap_err().contains("offset overflow"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn lists_built_per_batch_stay_in_datafusion() {
        // `CASE WHEN x THEN a ELSE b END` merges the lists of `a` on the first
        // 2048 rows (`1 << 30` nulls) and of `b` on the others (`1 << 30` nulls
        // too): on the batch of 4096 rows, their 32-bit offsets overflow, on
        // each chunk of 2048 rows of a pipeline, they do not. The filter only
        // outputs `x`.
        let batch = lists_to_merge();
        let schema = batch.schema();
        let plan = || -> Arc<dyn ExecutionPlan> {
            let case = physical_expr("CASE WHEN x THEN a ELSE b END", &schema);
            let predicate = Arc::new(
                datafusion_physical_expr::expressions::IsNotNullExpr::new(case),
            );
            Arc::new(
                datafusion_physical_plan::filter::FilterExecBuilder::new(
                    predicate,
                    scan_batches(&schema, std::slice::from_ref(&batch)),
                )
                .apply_projection(Some(vec![0]))
                .unwrap()
                .build()
                .unwrap(),
            )
        };
        let filter_plan = plan();
        let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
        assert!(!PhysicalFilter::supports_filter_exec(exec));
        let outcome =
            assert_same_outcome_in_datafusion(plan(), Arc::new(TaskContext::default()))
                .await;
        assert!(outcome.unwrap_err().contains("overflow"));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn lists_of_values_without_memory_stay_in_datafusion() {
        // Arrow's `take`, with which the native operators gather rows (e.g. the
        // cache, to copy the rows it keeps), computes the indices of the values
        // of fixed-size lists in 32 bits: with lists of `1 << 30` nulls, or of
        // runs, they overflow from the fourth row on. DataFusion's `filter`
        // does not.
        let cases: [(&[i32], bool); 3] = [
            (&[0, 0, 0, 0, 0, 1], false),
            (&[1, 0, 1, 0, 1, 1], false),
            (&[0, 0, 0, 0, 1, 0, 1, 0], true),
        ];
        for (x, runs) in cases {
            let batch = wide_lists(x, runs);
            let schema = batch.schema();
            let plan = || {
                filter_exec(
                    "x > 0",
                    scan_batches(&schema, std::slice::from_ref(&batch)),
                    None,
                )
            };
            let filter_plan = plan();
            let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
            assert!(!PhysicalFilter::supports_filter_exec(exec));
            let outcome = assert_same_outcome_in_datafusion(
                plan(),
                Arc::new(TaskContext::default()),
            )
            .await;
            assert_eq!(outcome, Ok(x.iter().filter(|&&x| x > 0).count()));
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn predicates_with_several_failures_stay_in_datafusion() {
        // `FilterExec` evaluates `CAST(s AS INT)` on all rows of the batch
        // before `10 / x`, and fails to cast `"invalid"`; on the first chunk of
        // a pipeline, `"1"` casts, and `10 / x` divides by zero.
        let batch = xss(&[((0, "1"), 2048), ((1, "invalid"), 2048)]);
        let predicate = "CAST(s AS INT) + 10 / x > 0";
        let plan = || filter_exec(predicate, scan(&batch), None);
        let filter_plan = plan();
        let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
        assert!(!PhysicalFilter::supports_filter_exec(exec));
        // On one thread, a pipeline evaluates the first chunk first.
        let config = datafusion::prelude::SessionConfig::new().with_target_partitions(1);
        let (error, operators) =
            assert_native_plan_parity_with_config(plan, false, config).await;
        assert!(error.unwrap().contains("Cannot cast string 'invalid'"));
        assert_eq!(operators.filters, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn chains_with_several_failures_stay_in_datafusion() {
        // `ProjectionExec(x, CAST(s AS INT))` casts all rows of the batch, and
        // fails on `"invalid"`, before `FilterExec(10 / x > 1)` sees the batch;
        // a pipeline evaluates both operators on its first chunk first, and
        // divides by zero.
        let batch = xss(&[((0, "1"), 2048), ((1, "invalid"), 2048)]);
        let schema = batch.schema();
        let plan = |cast: &str| {
            let projection = Arc::new(
                datafusion_physical_plan::projection::ProjectionExec::try_new(
                    [
                        datafusion_physical_expr::projection::ProjectionExpr::new(
                            physical_expr("x", &schema),
                            "x",
                        ),
                        datafusion_physical_expr::projection::ProjectionExpr::new(
                            physical_expr(cast, &schema),
                            "c",
                        ),
                    ],
                    scan(&batch),
                )
                .unwrap(),
            );
            filter_exec("10 / x > 1", projection, None)
        };
        // Each operator alone could be lowered.
        let filter_plan = plan("CAST(s AS INT)");
        let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
        assert!(PhysicalFilter::supports_filter_exec(exec));
        let projection = exec
            .input()
            .downcast_ref::<datafusion_physical_plan::projection::ProjectionExec>()
            .unwrap();
        assert!(
            crate::operators::projection::PhysicalProjection::supports_projection_exec(
                projection
            )
        );
        // On one thread, a pipeline evaluates the first chunk first.
        let config =
            || datafusion::prelude::SessionConfig::new().with_target_partitions(1);
        let (error, operators) = assert_native_plan_parity_with_config(
            || plan("CAST(s AS INT)"),
            false,
            config(),
        )
        .await;
        assert!(error.unwrap().contains("Cannot cast string 'invalid'"));
        assert_eq!((operators.filters, operators.projections), (0, 0));
        // Without a part that may fail in the projection, both are lowered.
        let (error, operators) =
            assert_native_plan_parity_with_config(|| plan("s"), false, config()).await;
        assert!(error.unwrap().contains("Divide by zero"));
        assert_eq!((operators.filters, operators.projections), (1, 1));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn filters_of_runs_with_named_fields_stay_in_datafusion() {
        // Arrow's `filter` rebuilds run-end encoded arrays with its own names
        // of fields, so the output of `FilterExec` does not have its type any
        // more (and its coalescer fails); a native filter keeps the type.
        let batch = runs_with_named_fields();
        let schema = batch.schema();
        let plan = || {
            filter_exec(
                "x > 0",
                scan_batches(&schema, std::slice::from_ref(&batch)),
                None,
            )
        };
        let filter_plan = plan();
        let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
        assert!(!PhysicalFilter::supports_filter_exec(exec));
        let task_ctx = Arc::new(TaskContext::default());
        let datafusion = outcome(plan(), Arc::clone(&task_ctx)).await;
        assert_eq!(
            assert_same_outcome_in_datafusion(plan(), task_ctx).await,
            datafusion
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn operations_on_empty_unions_stay_in_datafusion() {
        // An empty union has no value, so no row: there is no null of its
        // type to probe `IS NULL` with.
        let empty_union = DataType::Union(
            arrow::datatypes::UnionFields::try_new(Vec::<i8>::new(), Vec::<Field>::new())
                .unwrap(),
            arrow::datatypes::UnionMode::Sparse,
        );
        let schema = Arc::new(Schema::new(vec![Field::new("u", empty_union, true)]));
        let empty = RecordBatch::new_empty(Arc::clone(&schema));
        let plan = || -> Arc<dyn ExecutionPlan> {
            let predicate =
                Arc::new(datafusion_physical_expr::expressions::IsNullExpr::new(
                    physical_expr("u", &schema),
                ));
            Arc::new(
                FilterExec::try_new(
                    predicate,
                    scan_batches(&schema, std::slice::from_ref(&empty)),
                )
                .unwrap(),
            )
        };
        let filter_plan = plan();
        let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
        assert!(!PhysicalFilter::supports_filter_exec(exec));
        let (error, operators) = assert_native_plan_parity(plan, false).await;
        assert!(error.is_none());
        assert_eq!(operators.filters, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn operations_on_unsupported_types_stay_in_datafusion() {
        // `NOT` of an integer fails whenever it is evaluated, on an empty batch
        // too, which a pipeline skips.
        let schema = runs(&[]).schema();
        let empty = RecordBatch::new_empty(Arc::clone(&schema));
        let plan = || -> Arc<dyn ExecutionPlan> {
            let not = Arc::new(datafusion_physical_expr::expressions::NotExpr::new(
                physical_expr("x", &schema),
            ));
            Arc::new(
                FilterExec::try_new(
                    not,
                    scan_batches(&schema, std::slice::from_ref(&empty)),
                )
                .unwrap(),
            )
        };
        let filter_plan = plan();
        let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
        assert!(!PhysicalFilter::supports_filter_exec(exec));
        let (error, operators) = assert_native_plan_parity(plan, false).await;
        assert!(error.is_some());
        assert_eq!(operators.filters, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn fetch_zero_stays_in_datafusion() {
        // With `fetch = 0`, `FilterExec` evaluates its predicate on the first
        // batch of its input, even an empty one, and stops: it never pulls the
        // zeros, which its input divides by. Pipelines skip empty chunks: a
        // native filter would pull them.
        let schema = runs(&[]).schema();
        let plan = || {
            let batches = [
                RecordBatch::new_empty(Arc::clone(&schema)),
                runs(&[(0, 10)]),
            ];
            let projection = Arc::new(
                datafusion_physical_plan::projection::ProjectionExec::try_new(
                    [datafusion_physical_expr::projection::ProjectionExpr::new(
                        physical_expr("10 / x", &schema),
                        "q",
                    )],
                    scan_batches(&schema, &batches),
                )
                .unwrap(),
            );
            filter_exec("q >= 0", projection, Some(0))
        };
        let filter_plan = plan();
        let exec = filter_plan.downcast_ref::<FilterExec>().unwrap();
        assert!(!PhysicalFilter::supports_filter_exec(exec));
        let err =
            PhysicalFilter::try_from_filter_exec(exec, lower(exec.input()).unwrap())
                .unwrap_err();
        assert!(err.to_string().contains("fetch of 0"), "{err}");
        let (error, operators) = assert_native_plan_parity(plan, true).await;
        assert!(error.is_none(), "{error:?}");
        assert_eq!(operators.filters, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn fetch_with_fallible_predicates_stays_in_datafusion() {
        // `FilterExec` evaluates its predicate on whole batches until the fetch
        // is reached, on the first batch even with `fetch = 0`.
        for (batch, predicate, fetch) in [
            (runs(&[(0, 10)]), "10 / x > 1", 0),
            (runs(&[(1, 2048), (0, 2048)]), "10 / x > 1", 1),
            (runs(&[(1, 2048), (0, 2048)]), "x >= 0 AND 100 % x = 1", 1),
        ] {
            let plan = filter_exec(predicate, scan(&batch), Some(fetch));
            let exec = plan.downcast_ref::<FilterExec>().unwrap();
            assert!(!PhysicalFilter::supports_filter_exec(exec), "{predicate}");
            let (error, operators) = assert_native_plan_parity(
                || filter_exec(predicate, scan(&batch), Some(fetch)),
                true,
            )
            .await;
            assert_eq!(operators.filters, 0, "{predicate}");
            if predicate.contains('/') {
                assert!(error.unwrap().contains("Divide by zero"), "{predicate}");
            }
        }
        // A predicate that fails whenever it is evaluated is not lowered
        // either, but a pattern that compiles cannot fail.
        let strings = RecordBatch::try_new(
            Arc::new(Schema::new(vec![Field::new("s", DataType::Utf8, true)])),
            vec![Arc::new(StringArray::from(vec!["a", "b"]))],
        )
        .unwrap();
        for (predicate, lowered) in [("s ~ 'a{1000000}'", false), ("s LIKE 'a%'", true)] {
            let plan = filter_exec(predicate, scan(&strings), Some(1));
            let exec = plan.downcast_ref::<FilterExec>().unwrap();
            assert_eq!(
                PhysicalFilter::supports_filter_exec(exec),
                lowered,
                "{predicate}"
            );
            let (error, operators) = assert_native_plan_parity(
                || filter_exec(predicate, scan(&strings), Some(1)),
                true,
            )
            .await;
            assert_eq!(error.is_some(), !lowered, "{predicate}");
            assert_eq!(operators.filters, usize::from(lowered), "{predicate}");
        }
        // A predicate that cannot fail is lowered with its fetch.
        let batch = runs(&[(1, 2048), (0, 2048)]);
        let (error, operators) = assert_native_plan_parity(
            || filter_exec("x >= 0", scan(&batch), Some(1)),
            true,
        )
        .await;
        assert!(error.is_none());
        assert_eq!(operators.filters, 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn fallible_operators_below_limits_stay_in_datafusion() {
        // A limit stops the execution after the batch in which it is
        // reached: DataFusion evaluates `10 / x` on all 4096 rows and fails,
        // a native projection only on the first chunk of 2048 ones.
        let batch = runs(&[(1, 2048), (0, 2048)]);
        let ones = runs(&[(1, 4096)]);
        let limit = |expr: &str, filter: bool, batch: &RecordBatch| {
            let input = scan(batch);
            let input: Arc<dyn ExecutionPlan> = if filter {
                filter_exec(expr, input, None)
            } else {
                let expr = physical_expr(expr, &input.schema());
                Arc::new(
                    datafusion_physical_plan::projection::ProjectionExec::try_new(
                        [datafusion_physical_expr::projection::ProjectionExpr::new(
                            expr, "q",
                        )],
                        input,
                    )
                    .unwrap(),
                )
            };
            Arc::new(datafusion_physical_plan::limit::GlobalLimitExec::new(
                input,
                0,
                Some(1),
            )) as Arc<dyn ExecutionPlan>
        };
        for (expr, filter) in [("10 / x", false), ("10 / x > 1", true)] {
            let (error, operators) =
                assert_native_plan_parity(|| limit(expr, filter, &batch), true).await;
            assert!(error.unwrap().contains("Divide by zero"), "{expr}");
            assert_eq!(operators.filters + operators.projections, 0, "{expr}");
        }
        // Infallible operators stay in DataFusion below the limit too, with
        // their input (see `plans_below_limits_stay_in_datafusion`).
        for (expr, filter) in [("x + 1", false), ("x >= 0", true)] {
            let (error, operators) =
                assert_native_plan_parity(|| limit(expr, filter, &ones), true).await;
            assert!(error.is_none(), "{expr}");
            assert_eq!(operators.filters + operators.projections, 0, "{expr}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    // `with_limit_options` is the only public way to build a soft limit.
    #[expect(deprecated)]
    async fn fallible_operators_below_soft_limits_stay_in_datafusion() {
        use datafusion_physical_plan::aggregates::{
            AggregateExec, AggregateMode, LimitOptions, PhysicalGroupBy,
        };
        // `SELECT DISTINCT x ... LIMIT 1`: the aggregation stops reading its
        // input after the batch in which it has one group. DataFusion divides
        // all 4096 rows of the batch, a native filter only the first chunk.
        let batch = runs(&[(1, 2048), (0, 2048)]);
        let plan = || {
            let filter = filter_exec("10 / x > 0", scan(&batch), None);
            let schema = filter.schema();
            let group_by = PhysicalGroupBy::new_single(vec![(
                physical_expr("x", &schema),
                "x".to_owned(),
            )]);
            let aggregate = AggregateExec::try_new(
                AggregateMode::Partial,
                group_by,
                vec![],
                vec![],
                filter,
                schema,
            )
            .unwrap()
            .with_limit_options(Some(LimitOptions::new(1)));
            assert!(aggregate.limit_options().is_some());
            Arc::new(aggregate) as Arc<dyn ExecutionPlan>
        };
        let (error, operators) = assert_native_plan_parity(plan, false).await;
        assert!(error.unwrap().contains("Divide by zero"));
        assert_eq!(operators.filters, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn filter_with_fetch_is_lowered_with_its_limit() {
        let sql = "SELECT * FROM (SELECT id, a FROM t1 ORDER BY id LIMIT 500) WHERE a > 0 LIMIT 7";
        let run = assert_native_byte_eq_with_setup(sql, all_tables).await;
        let rows = run
            .actual
            .batches
            .iter()
            .map(RecordBatch::num_rows)
            .sum::<usize>();
        assert_eq!(rows, 7);
        let description = crate::explain::format_operator_tree(run.root.operator());
        assert!(description.contains("fetch=7"), "{description}");
    }
}
