// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`PhysicalProjection`]: expression evaluation over [`DataChunk`]s.
//!
//! The native counterpart of DataFusion's `ProjectionExec`. Every output
//! column is the result of a [`PhysicalExpr`] of the DataFusion plan, so the
//! output is exactly DataFusion's:
//!
//! * If every expression is a column of the input (e.g. `SELECT b, a AS x`),
//!   the output columns are the input [`ArrayRef`]s and the
//!   [`SelectionVector`] of the input is kept: no data is copied, not even for
//!   the output of a filter.
//! * Otherwise the expressions are evaluated on a [`RecordBatch`] of the
//!   input columns they reference. If the input carries a selection, only
//!   these columns are compacted (each once, however many expressions use
//!   it), so the expressions only ever see the selected rows: a row removed
//!   by a filter never causes an error, e.g. `SELECT 10 / x ... WHERE x <> 0`
//!   never divides by zero, exactly as in DataFusion where the filter output
//!   is compacted. Input columns that are not referenced are never touched.
//!
//! So an expression fails exactly when it fails in `ProjectionExec` for the
//! same rows. The projection evaluates its expressions on chunks, not on the
//! batches `ProjectionExec` sees (e.g. the batches a `FilterExec` coalesces
//! its output into, or empty batches, which pipelines skip), so
//! [`PhysicalProjection::try_from_projection_exec`] only lowers expressions
//! whose errors do not depend on the batches (see [`depends_on_batches`]):
//! `x = 0 OR 10 / x > 1`, whose right side DataFusion evaluates on all rows of
//! a batch unless at most 20% of them have `x <> 0`, or
//! `x = 0 OR s ~ 'a{1000000}'`, whose right side does not compile and is
//! evaluated on empty batches, must be evaluated by DataFusion, on the
//! batches of its input plan.
//!
//! The evaluation of expressions on some of the rows of a chunk (`BoundExprs`)
//! is shared with [`PhysicalFilter`](super::filter::PhysicalFilter).

use std::sync::Arc;

use arrow::array::{ArrayRef, RecordBatch, RecordBatchOptions};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef, UnionMode};
use arrow_select::take::take;
use datafusion_common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion_common::{Result, internal_datafusion_err, internal_err, not_impl_err};
use datafusion_physical_expr::expressions::{
    Column, LambdaExpr, LambdaVariable, Literal,
};
use datafusion_physical_expr::projection::{ProjectionExpr, ProjectionExprs};
use datafusion_physical_expr::utils::collect_columns;
use datafusion_physical_expr::{PhysicalExpr, ScalarFunctionExpr};
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::projection::ProjectionExec;
use itertools::Itertools;

use super::adaptive_filter::{depends_on_batches, failing_parts, has_32_bit_offsets};
use crate::chunk::{DataChunk, SelectionVector};
use crate::operator::{
    ExecutionContext, GlobalOperatorState, OperatorResult, OperatorState,
    PhysicalOperator,
};

/// Evaluates a list of expressions on the [`DataChunk`]s of its input, like
/// DataFusion's `ProjectionExec`.
///
/// See the [module documentation](self) for how selections are handled.
#[derive(Debug)]
pub struct PhysicalProjection {
    children: Vec<Arc<dyn PhysicalOperator>>,
    /// The expressions and the names of the output columns.
    exprs: Vec<(Arc<dyn PhysicalExpr>, String)>,
    schema: SchemaRef,
    /// The input column of every output column, if every expression is a
    /// column of the input.
    pass_through: Option<Vec<usize>>,
    /// All expressions, bound to the input columns they reference.
    bound: BoundExprs,
}

impl PhysicalProjection {
    /// Create a projection that evaluates `exprs` on the chunks of `input`,
    /// naming the output column of every expression by its alias.
    ///
    /// The output schema is derived like the one of `ProjectionExec`: the
    /// data type, nullability and field metadata of every expression and the
    /// schema metadata of the input.
    pub fn try_new(
        input: Arc<dyn PhysicalOperator>,
        exprs: Vec<(Arc<dyn PhysicalExpr>, String)>,
    ) -> Result<Self> {
        let projection = exprs
            .iter()
            .map(|(expr, alias)| ProjectionExpr::new(Arc::clone(expr), alias.clone()))
            .collect::<ProjectionExprs>();
        let schema = Arc::new(projection.project_schema(&input.schema())?);
        Self::try_new_with_schema(input, exprs, schema)
    }

    /// Whether [`Self::try_from_projection_exec`] can lower `exec` with
    /// exactly its semantics, including its errors: whether none of its
    /// expressions has errors that depend on how DataFusion splits the rows
    /// into batches, empty ones included (see [`depends_on_batches`]), at most
    /// one part of all of its expressions may fail (which error
    /// `ProjectionExec`, which evaluates each expression on a whole batch
    /// before the next one, raises first depends on its batches, see
    /// [`failing_parts`]), and its input has no lists of values that take
    /// less than a byte each (e.g. fixed-size lists of `1 << 30` nulls,
    /// booleans, or empty lists or binaries), which the native operators cannot
    /// gather with `take`, and no run-end encoded arrays whose child fields are
    /// not Arrow's (see `can_gather`).
    ///
    /// Below a limit that stops the execution early (e.g. a
    /// `GlobalLimitExec`), a planner must keep the whole plan in DataFusion,
    /// and so it must a projection with a part that may fail above native
    /// operators of the same pipeline with parts that may fail (see
    /// [`supports_filter_exec`](super::filter::PhysicalFilter::supports_filter_exec)):
    /// the rows evaluated before the limit is reached, and which error comes
    /// first, depend on DataFusion's batches.
    pub fn supports_projection_exec(exec: &ProjectionExec) -> bool {
        let input_schema = exec.input().schema();
        let failing_parts = exec
            .expr()
            .iter()
            .map(|projection| failing_parts(&projection.expr, input_schema.as_ref()))
            .fold(0, usize::saturating_add);
        failing_parts <= 1
            && input_schema
                .fields()
                .iter()
                .all(|field| can_gather(field.data_type()))
            && !exec.expr().iter().any(|projection| {
                depends_on_batches(&projection.expr, input_schema.as_ref())
            })
    }

    /// Lower a DataFusion [`ProjectionExec`]: evaluate its expressions on the
    /// chunks of `input` (the lowered input of `exec`) and produce exactly
    /// its output schema.
    ///
    /// Fails if `exec` is not supported, see
    /// [`Self::supports_projection_exec`].
    pub fn try_from_projection_exec(
        exec: &ProjectionExec,
        input: Arc<dyn PhysicalOperator>,
    ) -> Result<Self> {
        if !Self::supports_projection_exec(exec) {
            return not_impl_err!(
                "PhysicalProjection cannot lower this ProjectionExec: the errors of one of its expressions depend on DataFusion's batches"
            );
        }
        check_lowered_input(exec.input().schema().as_ref(), input.as_ref())?;
        let exprs = exec
            .expr()
            .iter()
            .map(|projection| (Arc::clone(&projection.expr), projection.alias.clone()))
            .collect();
        Self::try_new_with_schema(input, exprs, exec.schema())
    }

    /// Create a projection that evaluates `exprs` on the chunks of `input`
    /// and produces chunks of `schema`, which must have one field of the
    /// data type of every expression.
    pub fn try_new_with_schema(
        input: Arc<dyn PhysicalOperator>,
        exprs: Vec<(Arc<dyn PhysicalExpr>, String)>,
        schema: SchemaRef,
    ) -> Result<Self> {
        let input_schema = input.schema();
        if schema.fields().len() != exprs.len() {
            return internal_err!(
                "PhysicalProjection has {} expressions but its schema has {} fields",
                exprs.len(),
                schema.fields().len()
            );
        }
        for ((expr, _), field) in exprs.iter().zip(schema.fields()) {
            let data_type = expr.data_type(input_schema.as_ref())?;
            if &data_type != field.data_type() {
                return internal_err!(
                    "PhysicalProjection expression {expr} has type {data_type} but its output field '{}' has type {}",
                    field.name(),
                    field.data_type()
                );
            }
        }
        let pass_through = exprs
            .iter()
            .map(|(expr, _)| expr.downcast_ref::<Column>().map(Column::index))
            .collect::<Option<Vec<_>>>();
        let bound = BoundExprs::try_new(
            exprs.iter().map(|(expr, _)| Arc::clone(expr)),
            input_schema.as_ref(),
        )?;
        Ok(Self {
            children: vec![input],
            exprs,
            schema,
            pass_through,
            bound,
        })
    }

    /// The expressions and the names of the output columns.
    pub fn exprs(&self) -> &[(Arc<dyn PhysicalExpr>, String)] {
        &self.exprs
    }

    /// Whether every expression is a column of the input, so that chunks are
    /// projected without copying any data (and keep their selection).
    pub fn is_pass_through(&self) -> bool {
        self.pass_through.is_some()
    }
}

impl PhysicalOperator for PhysicalProjection {
    fn name(&self) -> &str {
        "PhysicalProjection"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.children
    }

    fn description(&self) -> String {
        format!(
            "PhysicalProjection: expr=[{}]",
            self.exprs
                .iter()
                .map(|(expr, alias)| format!("{expr} as {alias}"))
                .join(", ")
        )
    }

    fn execute(
        &self,
        _ctx: &ExecutionContext,
        input: &DataChunk,
        output: &mut DataChunk,
        _gstate: &dyn GlobalOperatorState,
        _lstate: &mut dyn OperatorState,
    ) -> Result<OperatorResult> {
        *output = self.project(input)?;
        Ok(OperatorResult::NeedMoreInput)
    }
}

impl PhysicalProjection {
    /// Project the (non-empty) chunk `input`.
    fn project(&self, input: &DataChunk) -> Result<DataChunk> {
        if self.schema.fields().is_empty() {
            return DataChunk::try_new_without_columns(
                Arc::clone(&self.schema),
                input.len(),
            );
        }
        if let Some(indices) = &self.pass_through {
            // Only pointers are copied; the selection, if any, is kept.
            let columns = indices
                .iter()
                .map(|&index| Arc::clone(input.column(index)))
                .collect();
            return DataChunk::try_new(
                Arc::clone(&self.schema),
                columns,
                input.selection().cloned(),
            );
        }
        // Compact the referenced columns (if there is a selection) and
        // evaluate the expressions on the selected rows only, exactly like
        // `ProjectionExec` evaluates them on the compacted output of a filter.
        let batch = self.bound.batch(input, logical_rows(input))?;
        let num_rows = batch.num_rows();
        let arrays = self
            .bound
            .exprs()
            .iter()
            .map(|expr| expr.evaluate(&batch)?.into_array_of_size(num_rows))
            .collect::<Result<Vec<_>>>()?;
        // Like `Projector::project_batch`, this validates the arrays against
        // the output schema, e.g. their nullability.
        let batch = RecordBatch::try_new(Arc::clone(&self.schema), arrays)?;
        Ok(DataChunk::from_record_batch(batch))
    }
}

/// Check that the schema of a lowered input operator matches the schema of
/// the DataFusion plan it was lowered from.
pub(crate) fn check_lowered_input(
    expected: &Schema,
    input: &dyn PhysicalOperator,
) -> Result<()> {
    let actual = input.schema();
    let same_types = expected.fields().len() == actual.fields().len()
        && expected
            .fields()
            .iter()
            .zip(actual.fields())
            .all(|(expected, actual)| expected.data_type() == actual.data_type());
    if !same_types {
        return internal_err!(
            "The schema of the lowered input {} does not match the schema of the plan: {:?} vs {:?}",
            input.name(),
            actual,
            expected
        );
    }
    Ok(())
}

/// The selection of the logical rows of `chunk` among its physical rows,
/// `None` if all physical rows are selected.
///
/// A chunk without columns only has a row count, its logical rows are its
/// "physical" rows.
pub(crate) fn logical_rows(chunk: &DataChunk) -> Option<&SelectionVector> {
    if chunk.num_columns() == 0 {
        None
    } else {
        chunk.selection()
    }
}

/// Physical expressions bound to the input columns they reference.
///
/// A [`PhysicalExpr`] is evaluated on a [`RecordBatch`]. To evaluate
/// expressions on some of the rows of a [`DataChunk`], only the columns they
/// reference have to be gathered: the expressions are rewritten to refer to a
/// batch of only these columns (in input order, see [`Self::batch`]), and the
/// other columns are never touched. Expressions that may read columns they do
/// not reference as `Column`s (with lambdas, or opaque leaves, see
/// `has_opaque_leaves`) see all input columns, unchanged.
#[derive(Debug, Clone)]
pub(crate) struct BoundExprs {
    /// The input columns referenced by the expressions, in increasing order.
    columns: Vec<usize>,
    /// The schema of `columns`: the input schema projected to them.
    schema: SchemaRef,
    /// The expressions, rewritten to refer to `schema`.
    exprs: Vec<Arc<dyn PhysicalExpr>>,
}

impl BoundExprs {
    /// Bind `exprs`, which refer to `input_schema`, to the columns they
    /// reference.
    pub(crate) fn try_new(
        exprs: impl IntoIterator<Item = Arc<dyn PhysicalExpr>>,
        input_schema: &Schema,
    ) -> Result<Self> {
        let exprs = exprs.into_iter().collect::<Vec<_>>();
        let num_fields = input_schema.fields().len();
        if exprs
            .iter()
            .any(|expr| has_lambdas(expr) || has_opaque_leaves(expr))
        {
            // The parameters of a lambda are numbered after the columns of
            // the input, and opaque leaves may read any input column: such
            // expressions see all input columns, unchanged.
            return Ok(Self {
                columns: (0..num_fields).collect(),
                schema: Arc::new(input_schema.clone()),
                exprs,
            });
        }
        let mut columns = exprs
            .iter()
            .flat_map(collect_columns)
            .map(|column| column.index())
            .collect::<Vec<_>>();
        columns.sort_unstable();
        columns.dedup();
        if let Some(&index) = columns.last()
            && index >= num_fields
        {
            return internal_err!(
                "Column index {index} out of bounds for an input with {num_fields} columns"
            );
        }
        let schema = Arc::new(input_schema.project(&columns)?);
        let mut positions = vec![None; num_fields];
        for (position, &index) in columns.iter().enumerate() {
            positions[index] = Some(position);
        }
        let exprs = exprs
            .into_iter()
            .map(|expr| remap_columns(expr, &positions))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            columns,
            schema,
            exprs,
        })
    }

    /// The bound expressions, to be evaluated on the batches returned by
    /// [`Self::batch`].
    pub(crate) fn exprs(&self) -> &[Arc<dyn PhysicalExpr>] {
        &self.exprs
    }

    /// The referenced columns of `chunk` at the physical rows `rows` (all
    /// physical rows if `None`), as a batch on which [`Self::exprs`] can be
    /// evaluated.
    ///
    /// Without `rows` no data is copied. Otherwise the columns are gathered
    /// with Arrow's `take` kernel, or sliced if `rows` is a contiguous range.
    pub(crate) fn batch(
        &self,
        chunk: &DataChunk,
        rows: Option<&SelectionVector>,
    ) -> Result<RecordBatch> {
        let num_rows = rows.map_or_else(|| chunk.physical_len(), SelectionVector::len);
        let columns = self
            .columns
            .iter()
            .map(|&index| {
                let column = chunk.columns().get(index).ok_or_else(|| {
                    internal_datafusion_err!(
                        "Column index {index} out of bounds for a chunk with {} columns",
                        chunk.columns().len()
                    )
                })?;
                match rows {
                    Some(rows) => gather(column, rows),
                    None => Ok(Arc::clone(column)),
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let options = RecordBatchOptions::new().with_row_count(Some(num_rows));
        Ok(RecordBatch::try_new_with_options(
            Arc::clone(&self.schema),
            columns,
            &options,
        )?)
    }
}

/// Whether `expr` has a leaf that may read input columns without referencing
/// them as `Column`s (e.g. a custom expression without children): a leaf that
/// is neither a column, a literal nor a function (whose arguments are its
/// children). DataFusion does not project the input of a `CASE` for such
/// expressions either (see `CaseExpr`).
fn has_opaque_leaves(expr: &Arc<dyn PhysicalExpr>) -> bool {
    expr.exists(|expr| {
        Ok(expr.children().is_empty()
            && !expr.is::<Column>()
            && !expr.is::<Literal>()
            && !expr.is::<ScalarFunctionExpr>())
    })
    .unwrap_or(true)
}

/// Whether `expr` contains a lambda function or one of its parameters.
fn has_lambdas(expr: &Arc<dyn PhysicalExpr>) -> bool {
    expr.exists(|expr| Ok(expr.is::<LambdaExpr>() || expr.is::<LambdaVariable>()))
        .unwrap_or(true)
}

/// Rewrite the column references of `expr` from input column `i` to
/// `positions[i]`.
fn remap_columns(
    expr: Arc<dyn PhysicalExpr>,
    positions: &[Option<usize>],
) -> Result<Arc<dyn PhysicalExpr>> {
    expr.transform_down(|expr| {
        let Some(column) = expr.downcast_ref::<Column>() else {
            return Ok(Transformed::no(expr));
        };
        let position = positions
            .get(column.index())
            .copied()
            .flatten()
            .ok_or_else(|| {
                internal_datafusion_err!("Column {column} was not collected")
            })?;
        if position == column.index() {
            return Ok(Transformed::no(expr));
        }
        Ok(Transformed::yes(Arc::new(Column::new(
            column.name(),
            position,
        ))))
    })
    .data()
}

/// Whether values of `data_type` may take less than a byte each: nulls,
/// booleans, run-end encoded values (one run stands for any number of
/// values), empty fixed-size lists of any type or binaries, and structs or
/// fixed-size lists of such values. Arrays of them may have more values than
/// 32-bit offsets, indices or keys count without taking gigabytes (or any
/// memory).
pub(crate) fn takes_little_memory(data_type: &DataType) -> bool {
    match data_type {
        DataType::Null | DataType::Boolean | DataType::RunEndEncoded(..) => true,
        DataType::Struct(fields) => fields
            .iter()
            .all(|field| takes_little_memory(field.data_type())),
        // Empty lists have no value, whatever their type.
        DataType::FixedSizeList(_, 0) | DataType::FixedSizeBinary(0) => true,
        DataType::FixedSizeList(field, _) => takes_little_memory(field.data_type()),
        _ => false,
    }
}

/// Whether the native operators can gather rows of arrays of `data_type`
/// (with [`gather`], or `DataChunk::compact`): not if they contain, at any
/// depth, lists whose values may take less than a byte each (nulls, booleans,
/// run-end encoded values, of which one run stands for any number of values,
/// empty fixed-size lists of any type or binaries, or structs or fixed-size
/// lists of such values), which may have more values than Arrow's `take` can
/// count (it computes the indices of the values of fixed-size lists in 32
/// bits, which count at most 512 MiB of booleans), and whose cost is
/// proportional to the number of values. DataFusion filters such arrays
/// without copying their values. Nor run-end encoded arrays whose child fields
/// are not Arrow's, which Arrow's `filter` renames, nor dense unions with
/// children with 32-bit offsets, whose values rows may share.
pub(crate) fn can_gather(data_type: &DataType) -> bool {
    match data_type {
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::ListView(field)
        | DataType::LargeListView(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => {
            !takes_little_memory(field.data_type()) && can_gather(field.data_type())
        }
        DataType::Struct(fields) => {
            fields.iter().all(|field| can_gather(field.data_type()))
        }
        // Rows of a dense union may point to the same value of a child, which
        // `take` (and DataFusion's `filter`) copy for every row, so the 32-bit
        // offsets of the child may overflow on some batches and not others.
        DataType::Union(fields, UnionMode::Dense)
            if fields
                .iter()
                .any(|(_, field)| has_32_bit_offsets(field.data_type())) =>
        {
            false
        }
        DataType::Union(fields, _) => fields
            .iter()
            .all(|(_, field)| can_gather(field.data_type())),
        DataType::Dictionary(_, values) => can_gather(values),
        // Arrow's `filter` rebuilds run-end encoded arrays with its own child
        // fields: the batches of DataFusion may not have other ones.
        DataType::RunEndEncoded(run_ends, values) => {
            **run_ends == Field::new("run_ends", run_ends.data_type().clone(), false)
                && **values == Field::new("values", values.data_type().clone(), true)
                && can_gather(values.data_type())
        }
        _ => true,
    }
}

/// The rows `rows` of `column`: a slice if `rows` is a contiguous range,
/// otherwise a copy made with Arrow's `take` kernel.
pub(crate) fn gather(column: &ArrayRef, rows: &SelectionVector) -> Result<ArrayRef> {
    if let Some((offset, len)) = rows.as_contiguous_range() {
        if offset == 0 && len == column.len() {
            return Ok(Arc::clone(column));
        }
        return Ok(column.slice(offset, len));
    }
    if rows.is_empty() {
        return Ok(column.slice(0, 0));
    }
    Ok(take(column.as_ref(), &rows.to_uint32_array(), None)?)
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashMap;

    use arrow::array::{AsArray, Int32Array, Int64Array, StringArray, StringViewArray};
    use arrow::datatypes::{DataType, Field, Int32Type, Int64Type};
    use datafusion::prelude::SessionContext;
    use datafusion_common::DataFusionError;
    use datafusion_datasource::memory::MemorySourceConfig;
    use datafusion_expr::Operator;
    use datafusion_physical_expr::expressions::BinaryExpr;
    use datafusion_physical_plan::filter::FilterExec;

    use crate::operators::filter::test_support::*;

    fn schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Int64, false),
            Field::new("s", DataType::Utf8, true),
            Field::new("v", DataType::Utf8View, true),
        ]))
    }

    fn batch(rows: i64) -> RecordBatch {
        let columns: Vec<ArrayRef> = vec![
            Arc::new(Int32Array::from_iter(
                (0..rows).map(|i| (i % 7 != 3).then_some((i % 100) as i32)),
            )),
            Arc::new(Int64Array::from_iter_values(0..rows)),
            Arc::new(StringArray::from_iter(
                (0..rows).map(|i| (i % 5 != 1).then(|| format!("s{}", i % 11))),
            )),
            Arc::new(StringViewArray::from_iter((0..rows).map(|i| {
                Some(format!("a string view longer than 12 bytes {i}"))
            }))),
        ];
        RecordBatch::try_new(schema(), columns).unwrap()
    }

    fn projection(exprs: &[(&str, &str)]) -> PhysicalProjection {
        let schema = schema();
        let exprs = exprs
            .iter()
            .map(|(sql, alias)| (physical_expr(sql, &schema), (*alias).to_owned()))
            .collect();
        PhysicalProjection::try_new(source(schema, vec![]), exprs).unwrap()
    }

    fn project(projection: &PhysicalProjection, input: &DataChunk) -> Result<DataChunk> {
        let (gstate, mut lstate) = states(projection);
        let mut output = DataChunk::empty(projection.schema());
        let result = projection.execute(
            &context(),
            input,
            &mut output,
            gstate.as_ref(),
            lstate.as_mut(),
        )?;
        assert!(matches!(result, OperatorResult::NeedMoreInput));
        Ok(output)
    }

    /// What `ProjectionExec` produces for the logical rows of `input`.
    fn expected(projection: &PhysicalProjection, input: &DataChunk) -> RecordBatch {
        let exec = ProjectionExec::try_new(
            projection.exprs().iter().map(|(expr, alias)| {
                ProjectionExpr::new(Arc::clone(expr), alias.clone())
            }),
            MemorySourceConfig::try_new_exec(&[vec![]], schema(), None).unwrap(),
        )
        .unwrap();
        assert_eq!(exec.schema(), projection.schema());
        let arrays = exec
            .expr()
            .iter()
            .map(|p| {
                p.expr
                    .evaluate(&input.compact().unwrap())
                    .unwrap()
                    .into_array(input.len())
                    .unwrap()
            })
            .collect();
        RecordBatch::try_new(exec.schema(), arrays).unwrap()
    }

    fn selected(rows: i64, selection: Vec<u32>) -> DataChunk {
        DataChunk::from_record_batch(batch(rows))
            .with_selection(SelectionVector::from(selection))
            .unwrap()
    }

    #[test]
    fn columns_are_passed_through_with_their_selection() {
        let projection = projection(&[("v", "x"), ("a", "a"), ("v", "y")]);
        assert!(projection.is_pass_through());
        let input = selected(100, vec![3, 17, 42, 99]);
        let output = project(&projection, &input).unwrap();
        assert_eq!(
            output
                .schema()
                .fields()
                .iter()
                .map(|f| f.name().as_str())
                .collect::<Vec<_>>(),
            vec!["x", "a", "y"]
        );
        // Only pointers are copied, and the selection is kept.
        assert!(Arc::ptr_eq(output.column(0), input.column(3)));
        assert!(Arc::ptr_eq(output.column(1), input.column(0)));
        assert!(Arc::ptr_eq(output.column(2), input.column(3)));
        assert_eq!(output.selection(), input.selection());
        assert_eq!(output.compact().unwrap(), expected(&projection, &input));

        // Without a selection.
        let input = DataChunk::from_record_batch(batch(10));
        let output = project(&projection, &input).unwrap();
        assert!(output.selection().is_none());
        assert_eq!(output.compact().unwrap(), expected(&projection, &input));
    }

    #[test]
    fn expressions_only_compact_the_referenced_columns() {
        let projection = projection(&[
            ("a + 1", "a1"),
            ("upper(s)", "u"),
            ("b", "b"),
            ("concat(s, 'x')", "c"),
        ]);
        assert!(!projection.is_pass_through());
        // `v` is not referenced: it is never compacted.
        assert_eq!(projection.bound.columns, vec![0, 1, 2]);
        for input in [
            selected(2000, (0..2000).filter(|i| i % 3 == 0).collect()),
            selected(2000, (0..2000).rev().step_by(7).collect()),
            // A contiguous selection is sliced.
            selected(2000, (100..300).collect()),
            DataChunk::from_record_batch(batch(50)),
        ] {
            let output = project(&projection, &input).unwrap();
            // The output is compacted.
            assert!(output.selection().is_none());
            assert_eq!(output.len(), input.len());
            assert_eq!(output.compact().unwrap(), expected(&projection, &input));
        }
    }

    #[test]
    fn expressions_are_bound_to_the_columns_they_reference() {
        let schema = schema();
        let bound = BoundExprs::try_new(
            [
                physical_expr("upper(s)", &schema),
                physical_expr("b + 1", &schema),
            ],
            &schema,
        )
        .unwrap();
        assert_eq!(bound.columns, vec![1, 2]);
        assert_eq!(bound.schema.as_ref(), &schema.project(&[1, 2]).unwrap());
        assert_eq!(bound.exprs()[0].to_string(), "upper(s@1)");
        assert_eq!(bound.exprs()[1].to_string(), "b@0 + 1");
        let input = selected(100, vec![7, 3]);
        let batch = bound.batch(&input, input.selection()).unwrap();
        assert_eq!(batch.num_rows(), 2);
        assert_eq!(
            batch.column(0).as_primitive::<Int64Type>(),
            &Int64Array::from(vec![7, 3])
        );
        // Without rows, the columns are not copied.
        let batch = bound.batch(&input, None).unwrap();
        assert!(Arc::ptr_eq(batch.column(1), input.column(2)));

        // Lambda parameters are numbered after the input columns: an
        // expression with lambdas sees all columns, and is not rewritten.
        let variable: Arc<dyn PhysicalExpr> = Arc::new(LambdaVariable::new(
            4,
            Arc::new(Field::new("v", DataType::Int64, true)),
        ));
        let expr: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            variable,
            Operator::Eq,
            physical_expr("b", &schema),
        ));
        let bound = BoundExprs::try_new([Arc::clone(&expr)], &schema).unwrap();
        assert_eq!(bound.columns, vec![0, 1, 2, 3]);
        assert!(Arc::ptr_eq(&bound.exprs()[0], &expr));
    }

    #[test]
    fn fallible_expressions_only_see_the_selected_rows() {
        let schema = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int32, true),
            Field::new("y", DataType::Int32, true),
        ]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![0, 2, 0, 5, 0])),
                Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
            ],
        )
        .unwrap();
        let projection = PhysicalProjection::try_new(
            source(Arc::clone(&schema), vec![]),
            vec![
                (physical_expr("10 / x", &schema), "q".to_owned()),
                (physical_expr("y", &schema), "y".to_owned()),
            ],
        )
        .unwrap();
        // The rows of a filter `x <> 0`.
        let input = DataChunk::from_record_batch(batch.clone())
            .with_selection(SelectionVector::from(vec![1, 3]))
            .unwrap();
        let output = project(&projection, &input).unwrap().compact().unwrap();
        assert_eq!(
            output.column(0).as_primitive::<Int64Type>(),
            &Int64Array::from(vec![5, 2])
        );
        assert_eq!(
            output.column(1).as_primitive::<Int32Type>(),
            &Int32Array::from(vec![2, 4])
        );
        // The division fails on the other rows.
        let err = project(&projection, &DataChunk::from_record_batch(batch)).unwrap_err();
        assert!(err.to_string().contains("Divide by zero"), "{err}");
    }

    #[test]
    fn literals_and_empty_projections_keep_the_row_count() {
        let schema = Arc::new(Schema::empty());
        let input = DataChunk::try_new_without_columns(Arc::clone(&schema), 5).unwrap();
        let projection = PhysicalProjection::try_new(
            source(Arc::clone(&schema), vec![]),
            vec![(physical_expr("1", &schema), "one".to_owned())],
        )
        .unwrap();
        let output = project(&projection, &input).unwrap();
        assert_eq!(output.len(), 5);
        assert_eq!(output.column(0).len(), 5);

        let projection = projection_of(&[]);
        let input = selected(10, vec![1, 2, 3]);
        let output = project(&projection, &input).unwrap();
        assert_eq!(output.num_columns(), 0);
        assert_eq!(output.len(), 3);

        // Literals on a chunk with a selection.
        let projection = projection_of(&[("'x'", "x"), ("a", "a")]);
        let output = project(&projection, &input).unwrap();
        assert_eq!(output.compact().unwrap(), expected(&projection, &input));
    }

    fn projection_of(exprs: &[(&str, &str)]) -> PhysicalProjection {
        projection(exprs)
    }

    #[test]
    fn schema_keeps_metadata_and_nullability() {
        let metadata = HashMap::from([("origin".to_owned(), "test".to_owned())]);
        let schema = Arc::new(
            Schema::new(vec![
                Field::new("a", DataType::Int32, false).with_metadata(metadata.clone()),
                Field::new("b", DataType::Int32, true),
            ])
            .with_metadata(metadata.clone()),
        );
        let exprs = vec![
            (physical_expr("a", &schema), "x".to_owned()),
            (physical_expr("a + b", &schema), "y".to_owned()),
        ];
        let projection = PhysicalProjection::try_new(
            source(Arc::clone(&schema), vec![]),
            exprs.clone(),
        )
        .unwrap();
        let exec = ProjectionExec::try_new(
            exprs
                .into_iter()
                .map(|(expr, alias)| ProjectionExpr::new(expr, alias)),
            MemorySourceConfig::try_new_exec(&[vec![]], Arc::clone(&schema), None)
                .unwrap(),
        )
        .unwrap();
        assert_eq!(projection.schema(), exec.schema());
        assert_eq!(projection.schema().field(0).metadata(), &metadata);
        assert!(!projection.schema().field(0).is_nullable());
        assert_eq!(projection.schema().metadata(), &metadata);

        // Lowering a `ProjectionExec` keeps its schema.
        let lowered = PhysicalProjection::try_from_projection_exec(
            &exec,
            source(Arc::clone(&schema), vec![]),
        )
        .unwrap();
        assert_eq!(lowered.schema(), exec.schema());
        assert_eq!(
            lowered.description(),
            "PhysicalProjection: expr=[a@0 as x, a@0 + b@1 as y]"
        );
    }

    #[test]
    fn invalid_schemas_and_nulls_in_non_nullable_fields_are_errors() {
        let schema = schema();
        let input = source(Arc::clone(&schema), vec![]);
        // A non-nullable output field that gets nulls fails, like in
        // `ProjectionExec`.
        let output_schema =
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, false)]));
        let projection = PhysicalProjection::try_new_with_schema(
            Arc::clone(&input),
            vec![(physical_expr("a + CAST(1 AS INT)", &schema), "a".to_owned())],
            Arc::clone(&output_schema),
        )
        .unwrap();
        let err =
            project(&projection, &DataChunk::from_record_batch(batch(10))).unwrap_err();
        assert!(matches!(err, DataFusionError::ArrowError(_, _)), "{err}");
        assert!(err.to_string().contains("non-nullable"), "{err}");

        let err = PhysicalProjection::try_new_with_schema(
            Arc::clone(&input),
            vec![(physical_expr("s", &schema), "a".to_owned())],
            output_schema,
        )
        .unwrap_err();
        assert!(err.to_string().contains("has type Utf8"), "{err}");

        // The lowered input must have the schema of the plan.
        let exec = ProjectionExec::try_new(
            [ProjectionExpr::new(physical_expr("b", &schema), "b")],
            MemorySourceConfig::try_new_exec(&[vec![]], Arc::clone(&schema), None)
                .unwrap(),
        )
        .unwrap();
        let other = Arc::new(Schema::new(vec![Field::new("z", DataType::Utf8, true)]));
        let err =
            PhysicalProjection::try_from_projection_exec(&exec, source(other, vec![]))
                .unwrap_err();
        assert!(err.to_string().contains("does not match"), "{err}");
    }

    // ---------------------------------------------------------------------
    // Differential tests
    // ---------------------------------------------------------------------

    async fn assert_all_native_byte_eq(queries: &[&str]) {
        for sql in queries {
            let run = assert_native_byte_eq_with_setup(sql, all_tables).await;
            assert!(run.operators.projections > 0, "no native projection: {sql}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_projections_over_filtered_chunks() {
        assert_all_native_byte_eq(&[
            "SELECT id, a + b AS s, c * 2 AS c2, length(d) AS u, NOT e AS ne FROM t1 WHERE a > 10",
            "SELECT a AS x, id, h, a AS y FROM t1 WHERE a > 10 AND d <> 'beta'",
            "SELECT id, CASE WHEN a IS NULL THEN 0 WHEN a < 0 THEN -1 ELSE 1 END AS sign FROM t1 WHERE b > 100",
            "SELECT 1 AS one, '' AS s FROM t1 WHERE a > 0",
            "SELECT id, CAST(g AS DOUBLE) AS gd, ts FROM t1 WHERE f > DATE '1970-01-01'",
            "SELECT id, c, -c AS minus_c FROM t1 WHERE c IS NULL OR c = 0.0 OR isnan(c)",
            "SELECT id * 2 AS id2, b FROM t1 WHERE id % 1000 = 7",
            "SELECT id, a + b AS s FROM t1",
            "SELECT id + 1 AS i FROM t_empty WHERE a > 0",
            "SELECT id, a + 1 AS a1 FROM t1 WHERE a > 0 ORDER BY id",
            "SELECT d, COUNT(*) AS n FROM t1 WHERE a > 0 GROUP BY d",
            "SELECT t1.id, t2.name, t1.a * t2.w AS aw FROM t1 JOIN t2 ON t1.a = t2.k WHERE t1.b > 10",
            "SELECT id FROM t1 WHERE a > 40 UNION ALL SELECT k FROM t2 WHERE k < 0",
        ])
        .await;
        // Several functions (whose failures are not known) or other parts that
        // may fail: which error comes first depends on the batches. Or string
        // constants, copied for every row of a batch, whose 32-bit offsets
        // overflow on batches with enough rows. Or functions that build
        // strings, whose sizes DataFusion may limit per batch.
        assert_all_projections_in_datafusion(&[
            "SELECT id, length(h) AS len, concat(d, h) AS cat FROM t1 WHERE h LIKE '%01%'",
            "SELECT id, a + b AS s, upper(d) AS u FROM t1 WHERE a > 10",
            "SELECT id, CAST(g AS DOUBLE) AS gd, f + INTERVAL '1 day' AS next_day FROM t1",
            "SELECT id, CASE WHEN a IS NULL THEN 'none' WHEN a < 0 THEN 'neg' ELSE 'pos' END AS sign FROM t1 WHERE b > 100",
            "SELECT 1 AS one, 'x' AS s FROM t1 WHERE a > 0",
        ])
        .await;
    }

    /// Assert that `queries` return the same results with the native operators
    /// as with DataFusion, without native projections.
    async fn assert_all_projections_in_datafusion(queries: &[&str]) {
        for sql in queries {
            let run = assert_native_byte_eq_with_setup(sql, all_tables).await;
            assert_eq!(run.operators.projections, 0, "native projection: {sql}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_fallible_projections() {
        // The expressions fail on the rows the filter removes.
        assert_all_native_byte_eq(&[
            "SELECT id, 10 / x AS q FROM fallible WHERE x <> 0",
            "SELECT id, 10 % x AS r, x FROM fallible WHERE x <> 0 AND id % 2 = 1",
            "SELECT id, CAST(s AS INT) AS n FROM fallible WHERE id % 10 = 0",
            "SELECT id, CAST(big AS INT) AS b32 FROM fallible WHERE big < 100000",
        ])
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_projection_errors_match_datafusion() {
        // Whether a native projection is used, and whether the query fails
        // (in both engines).
        for (sql, native, fails) in [
            // No guard: `x = 0` on half of the rows.
            ("SELECT id, 10 / x AS q FROM dense", true, true),
            // Guarded by the filter, or by `CASE`.
            (
                "SELECT id, 10 / x AS q FROM dense WHERE x <> 0",
                true,
                false,
            ),
            (
                "SELECT id, CASE WHEN x <> 0 THEN 10 / x END AS q FROM dense",
                true,
                false,
            ),
            // `OR` divides all rows of a batch unless at most 20% have
            // `x <> 0`: the errors depend on the batches, so these projections
            // stay in DataFusion.
            (
                "SELECT id, x = 0 OR 10 / x > 1 AS b FROM dense",
                false,
                true,
            ),
            (
                "SELECT id, x = 0 OR 10 / x > 1 AS b FROM sparse",
                false,
                false,
            ),
            // The filter keeps the rows with `b = 0`.
            (
                "SELECT id, 10 / b AS q FROM nulls WHERE a IS NULL",
                true,
                true,
            ),
            (
                "SELECT id, 10 / b AS q FROM nulls_ok WHERE a IS NULL",
                true,
                false,
            ),
        ] {
            let (error, operators) =
                assert_native_parity_with_setup(sql, guard_tables).await;
            assert_eq!(operators.projections > 0, native, "{sql}");
            assert_eq!(error.is_some(), fails, "{sql}: {error:?}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn batch_dependent_projections_see_datafusion_batches() {
        // Two batches of 2048 rows: 1848 zeros then 200 ones, and 2048 ones.
        // `FilterExec` coalesces them into one batch, in which 55% of the rows
        // have `x <> 0`: DataFusion divides all of them by `x` and fails. On
        // the chunks of a pipeline, 10% of the rows of the first chunk have
        // `x <> 0`, so only these would be divided.
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, false)]));
        let batch = |zeros: usize| {
            RecordBatch::try_new(
                Arc::clone(&schema),
                vec![Arc::new(Int32Array::from_iter_values(
                    (0..2048).map(|i| i32::from(i >= zeros)),
                ))],
            )
            .unwrap()
        };
        let batches = vec![batch(1848), batch(0)];
        let setup = async |ctx: &SessionContext| {
            register_partitions(ctx, "coalesced", vec![batches.clone()])
        };
        let sql = "SELECT x = 0 OR 10 / x > 1 AS b FROM coalesced WHERE x >= 0";
        let (error, operators) = assert_native_parity_with_setup(sql, setup).await;
        assert!(error.unwrap().contains("Divide by zero"));
        // The projection stays in DataFusion, with the filter below it.
        assert_eq!(operators.projections + operators.filters, 0);

        let exec = ProjectionExec::try_new(
            [ProjectionExpr::new(
                physical_expr("x = 0 OR 10 / x > 1", &schema),
                "b",
            )],
            MemorySourceConfig::try_new_exec(
                std::slice::from_ref(&batches),
                Arc::clone(&schema),
                None,
            )
            .unwrap(),
        )
        .unwrap();
        assert!(!PhysicalProjection::supports_projection_exec(&exec));
        let err = PhysicalProjection::try_from_projection_exec(
            &exec,
            source(Arc::clone(&schema), vec![]),
        )
        .unwrap_err();
        assert!(
            err.to_string().contains("depend on DataFusion's batches"),
            "{err}"
        );
        // On the chunks, the projection does not fail.
        let projection = PhysicalProjection::try_new(
            source(Arc::clone(&schema), vec![]),
            vec![(
                physical_expr("x = 0 OR 10 / x > 1", &schema),
                "b".to_owned(),
            )],
        )
        .unwrap();
        for batch in batches {
            assert!(project(&projection, &DataChunk::from(batch)).is_ok());
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn projections_that_fail_on_some_batches_stay_in_datafusion() {
        let schema = xys(&[]).schema();
        let plan = |expr: &str, batches: &[RecordBatch]| -> Arc<dyn ExecutionPlan> {
            Arc::new(
                ProjectionExec::try_new(
                    [ProjectionExpr::new(physical_expr(expr, &schema), "m")],
                    scan_batches(&schema, batches),
                )
                .unwrap(),
            )
        };
        // One batch of 4096 rows, `(x, y) = (0, 1)` then `(1, 0)`: neither
        // `x = 0` nor `y = 0` is true on all of its rows, but `x = 0` is on all
        // rows of the first chunk, and `y = 0` on all rows of the second one.
        let mixed = xys(&[((0, 1), 2048), ((1, 0), 2048)]);
        let empty = RecordBatch::new_empty(Arc::clone(&schema));
        let zeros = xys(&[((0, 0), 2048)]);
        let nested = "x = 0 OR (y = 0 OR s ~ 'a{1000000}')";
        for (expr, batches, fails) in [
            (nested, vec![mixed.clone()], true),
            // `x IN (0, 10 / y)` evaluates `10 / y` on all rows of the batch,
            // but on none of the first chunk.
            ("x IN (0, 10 / (y - 1))", vec![mixed.clone()], true),
            // On an empty batch, `OR` evaluates its right side.
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
            let exec_plan = plan(expr, &batches);
            let exec = exec_plan.downcast_ref::<ProjectionExec>().unwrap();
            assert!(
                !PhysicalProjection::supports_projection_exec(exec),
                "{expr}"
            );
            let (error, operators) =
                assert_native_plan_parity(|| plan(expr, &batches), false).await;
            let input = format!("{expr} on {} batches", batches.len());
            assert_eq!(error.is_some(), fails, "{input}: {error:?}");
            assert_eq!(operators.projections, 0, "{input}");
        }
        // On the chunks of the pipeline, the projection does not fail.
        let native = PhysicalProjection::try_new(
            source(Arc::clone(&schema), vec![]),
            vec![(physical_expr(nested, &schema), "m".to_owned())],
        )
        .unwrap();
        for chunk in [mixed.slice(0, 2048), mixed.slice(2048, 2048)] {
            assert!(project(&native, &DataChunk::from(chunk.clone())).is_ok());
        }
        let native = PhysicalProjection::try_new(
            source(Arc::clone(&schema), vec![]),
            vec![(
                physical_expr("x IN (0, 10 / (y - 1))", &schema),
                "m".to_owned(),
            )],
        )
        .unwrap();
        assert!(project(&native, &DataChunk::from(mixed.slice(0, 2048))).is_ok());
        // A pattern that compiles cannot fail.
        for batches in [vec![mixed], vec![empty, zeros], vec![]] {
            let (error, operators) = assert_native_plan_parity(
                || plan("x = 0 OR (y = 0 OR s ~ 'a+')", &batches),
                false,
            )
            .await;
            assert!(error.is_none());
            assert_eq!(operators.projections, 1);
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn opaque_columns_see_all_input_columns() {
        // `opaque(x)` reads `x` without a `Column` child: bound to the columns
        // that the expressions reference (`y`, or none), it would read `y` as
        // `x`, or no column at all.
        let batch = xys(&[((0, 1), 100), ((2, 2), 100), ((3, 0), 100)]);
        let schema = batch.schema();
        let sum: Arc<dyn PhysicalExpr> = Arc::new(BinaryExpr::new(
            opaque_column("x", &schema),
            Operator::Plus,
            physical_expr("y", &schema),
        ));
        for exprs in [
            vec![(sum, "z")],
            vec![(opaque_column("x", &schema), "x")],
            vec![
                (physical_expr("s", &schema), "s"),
                (opaque_column("y", &schema), "y"),
            ],
        ] {
            let plan = || -> Arc<dyn ExecutionPlan> {
                Arc::new(
                    ProjectionExec::try_new(
                        exprs.iter().map(|(expr, name)| {
                            ProjectionExpr::new(Arc::clone(expr), *name)
                        }),
                        scan_batches(&schema, std::slice::from_ref(&batch)),
                    )
                    .unwrap(),
                )
            };
            let (error, operators) = assert_native_plan_parity(plan, false).await;
            assert!(error.is_none(), "{error:?}");
            assert_eq!(operators.projections, 1);
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn case_insensitive_patterns_that_fail_on_unicode_stay_in_datafusion() {
        // One batch of 4096 rows, `(x, s) = (1, "a")` then `(0, "é")`:
        // DataFusion evaluates `s ILIKE 'sss…'` on all rows of the batch, and
        // compiles its pattern (too big) because some strings are not ASCII,
        // the native projection only on the first chunk (all ASCII), on which
        // Arrow matches the pattern without compiling it.
        let batch = xss(&[((1, "a"), 2048), ((0, "é"), 2048)]);
        let schema = batch.schema();
        let expr = format!("x > 0 AND s ILIKE '{}'", "s".repeat(100_000));
        let plan = || -> Arc<dyn ExecutionPlan> {
            Arc::new(
                ProjectionExec::try_new(
                    [ProjectionExpr::new(physical_expr(&expr, &schema), "m")],
                    scan_batches(&schema, std::slice::from_ref(&batch)),
                )
                .unwrap(),
            )
        };
        let projection_plan = plan();
        let exec = projection_plan.downcast_ref::<ProjectionExec>().unwrap();
        assert!(!PhysicalProjection::supports_projection_exec(exec));
        let (error, operators) = assert_native_plan_parity(plan, false).await;
        assert!(error.unwrap().contains("Unable to build regex"));
        assert_eq!(operators.projections, 0);
        let native = PhysicalProjection::try_new(
            source(Arc::clone(&schema), vec![]),
            vec![(physical_expr(&expr, &schema), "m".to_owned())],
        )
        .unwrap();
        assert!(project(&native, &DataChunk::from(batch.slice(0, 2048))).is_ok());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn volatile_expressions_are_not_evaluated_before_execution() {
        use std::sync::atomic::{AtomicUsize, Ordering};
        // `count_calls()` returns how often it was called: DataFusion calls it
        // once for the batch of 3 rows. Evaluating it before (e.g. on an empty
        // batch, to find whether it fails without rows) would change its
        // results.
        let batch = xys(&[((1, 1), 3)]);
        let schema = batch.schema();
        let plan = |calls: &Arc<AtomicUsize>| -> Arc<dyn ExecutionPlan> {
            Arc::new(
                ProjectionExec::try_new(
                    [ProjectionExpr::new(count_calls(calls, &schema), "c")],
                    scan_batches(&schema, std::slice::from_ref(&batch)),
                )
                .unwrap(),
            )
        };
        let calls = Arc::new(AtomicUsize::new(0));
        let projection_plan = plan(&calls);
        let exec = projection_plan.downcast_ref::<ProjectionExec>().unwrap();
        assert!(!PhysicalProjection::supports_projection_exec(exec));
        lower(&projection_plan).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        let (error, operators) =
            assert_native_plan_parity(|| plan(&Arc::default()), false).await;
        assert!(error.is_none());
        assert_eq!(operators.projections, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn case_insensitive_patterns_on_bytes_beneath_nulls_stay_in_datafusion() {
        // `x = 0` selects every other row, the first one too, whose string is
        // null above bytes that are not ASCII: DataFusion's `filter` keeps
        // them, so Arrow compiles the pattern of the `ILIKE`, which is too
        // big; the native projection gathers the rows with `take`, which drops
        // them, so Arrow does not compile the pattern.
        let batch = non_ascii_beneath_a_null();
        let schema = batch.schema();
        let expr = format!("s ILIKE '{}'", "s".repeat(100_000));
        let plan = || -> Arc<dyn ExecutionPlan> {
            let filter = FilterExec::try_new(
                physical_expr("x = 0", &schema),
                scan_batches(&schema, std::slice::from_ref(&batch)),
            )
            .unwrap();
            Arc::new(
                ProjectionExec::try_new(
                    [ProjectionExpr::new(physical_expr(&expr, &schema), "m")],
                    Arc::new(filter),
                )
                .unwrap(),
            )
        };
        let projection_plan = plan();
        let exec = projection_plan.downcast_ref::<ProjectionExec>().unwrap();
        assert!(!PhysicalProjection::supports_projection_exec(exec));
        let (error, operators) = assert_native_plan_parity(plan, false).await;
        assert!(error.unwrap().contains("Unable to build regex"));
        assert_eq!(operators.projections, 0);
        let native = PhysicalProjection::try_new(
            source(Arc::clone(&schema), vec![]),
            vec![(physical_expr(&expr, &schema), "m".to_owned())],
        )
        .unwrap();
        let selected = selected_chunk(&batch, (0..2048).step_by(2).collect());
        assert!(project(&native, &selected).is_ok());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn casts_that_arrow_cannot_evaluate_stay_in_datafusion() {
        // Arrow cannot pack booleans into a dictionary: the cast fails whenever
        // it is evaluated, which depends on DataFusion's batches (see the
        // test of the same name for filters), empty ones included.
        let mixed = xys(&[((0, 1), 2048), ((1, 0), 2048)]);
        let schema = mixed.schema();
        let empty = RecordBatch::new_empty(Arc::clone(&schema));
        for (expr, batch) in [
            (nested_packed_booleans(&schema), &mixed),
            (packed_booleans(&schema), &empty),
        ] {
            let plan = || -> Arc<dyn ExecutionPlan> {
                Arc::new(
                    ProjectionExec::try_new(
                        [ProjectionExpr::new(Arc::clone(&expr), "m")],
                        scan_batches(&schema, std::slice::from_ref(batch)),
                    )
                    .unwrap(),
                )
            };
            let projection_plan = plan();
            let exec = projection_plan.downcast_ref::<ProjectionExec>().unwrap();
            assert!(
                !PhysicalProjection::supports_projection_exec(exec),
                "{expr}"
            );
            let (error, operators) = assert_native_plan_parity(plan, false).await;
            assert!(error.unwrap().contains("dictionary packing"), "{expr}");
            assert_eq!(operators.projections, 0, "{expr}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn lists_built_per_batch_stay_in_datafusion() {
        // On the batch of 4096 rows, the 32-bit offsets of the lists overflow,
        // on each chunk of 2048 rows of a pipeline, they do not: when
        // `CASE WHEN x THEN a ELSE b END` merges the lists of `a` on the first
        // 2048 rows (`1 << 30` nulls) and of `b` on the others (`1 << 30` too),
        // and when a constant list of `1 << 19` nulls is broadcast to the
        // rows.
        let batch = lists_to_merge();
        let schema = batch.schema();
        for expr in [
            physical_expr("CASE WHEN x THEN a ELSE b END", &schema),
            constant_list_of_nulls(),
        ] {
            let plan = || -> Arc<dyn ExecutionPlan> {
                Arc::new(
                    ProjectionExec::try_new(
                        [ProjectionExpr::new(Arc::clone(&expr), "m")],
                        scan_batches(&schema, std::slice::from_ref(&batch)),
                    )
                    .unwrap(),
                )
            };
            let projection_plan = plan();
            let exec = projection_plan.downcast_ref::<ProjectionExec>().unwrap();
            assert!(
                !PhysicalProjection::supports_projection_exec(exec),
                "{expr}"
            );
            let task_ctx = Arc::new(datafusion_execution::TaskContext::default());
            let outcome = assert_same_outcome_in_datafusion(plan(), task_ctx).await;
            assert!(outcome.unwrap_err().contains("overflow"), "{expr}");
        }
        // So do 16-bit run ends: a constant fixed-size list of 8 values in one
        // run, broadcast to the rows, has 32768 values on the batch of 4096
        // rows, and 16384 on a chunk of 2048.
        let ints = xys(&[((0, 0), 4096)]);
        let ints_schema = ints.schema();
        let plan = || -> Arc<dyn ExecutionPlan> {
            Arc::new(
                ProjectionExec::try_new(
                    [ProjectionExpr::new(constant_list_of_short_runs(), "m")],
                    scan_batches(&ints_schema, std::slice::from_ref(&ints)),
                )
                .unwrap(),
            )
        };
        let projection_plan = plan();
        let exec = projection_plan.downcast_ref::<ProjectionExec>().unwrap();
        assert!(!PhysicalProjection::supports_projection_exec(exec));
        let task_ctx = Arc::new(datafusion_execution::TaskContext::default());
        assert!(
            assert_same_outcome_in_datafusion(plan(), task_ctx)
                .await
                .is_err()
        );
        // Lists of nulls cannot be gathered at all (see `can_gather`): a
        // projection of the column stays in DataFusion too, and returns the
        // 4096 rows (whose lists are not compared, as they have `1 << 31`
        // values).
        let plan = || -> Arc<dyn ExecutionPlan> {
            Arc::new(
                ProjectionExec::try_new(
                    [ProjectionExpr::new(physical_expr("a", &schema), "a")],
                    scan_batches(&schema, std::slice::from_ref(&batch)),
                )
                .unwrap(),
            )
        };
        let task_ctx = Arc::new(datafusion_execution::TaskContext::default());
        assert_eq!(
            assert_same_outcome_in_datafusion(plan(), task_ctx).await,
            Ok(4096)
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn projections_with_several_failures_stay_in_datafusion() {
        // `ProjectionExec` evaluates `CAST(s AS INT)` on all rows of the batch
        // before `10 / x`, and fails to cast `"invalid"`; on the first chunk of
        // a pipeline, `"1"` casts, and `10 / x` divides by zero.
        let batch = xss(&[((0, "1"), 2048), ((1, "invalid"), 2048)]);
        let schema = batch.schema();
        let plan = || -> Arc<dyn ExecutionPlan> {
            Arc::new(
                ProjectionExec::try_new(
                    [
                        ProjectionExpr::new(
                            physical_expr("CAST(s AS INT)", &schema),
                            "c",
                        ),
                        ProjectionExpr::new(physical_expr("10 / x", &schema), "q"),
                    ],
                    scan_batches(&schema, std::slice::from_ref(&batch)),
                )
                .unwrap(),
            )
        };
        let projection_plan = plan();
        let exec = projection_plan.downcast_ref::<ProjectionExec>().unwrap();
        assert!(!PhysicalProjection::supports_projection_exec(exec));
        // On one thread, a pipeline evaluates the first chunk first.
        let config = datafusion::prelude::SessionConfig::new().with_target_partitions(1);
        let (error, operators) =
            assert_native_plan_parity_with_config(plan, false, config).await;
        assert!(error.unwrap().contains("Cannot cast string 'invalid'"));
        assert_eq!(operators.projections, 0);
        // A single expression that may fail is lowered.
        let (error, operators) = assert_native_plan_parity(
            || -> Arc<dyn ExecutionPlan> {
                Arc::new(
                    ProjectionExec::try_new(
                        [
                            ProjectionExpr::new(physical_expr("10 / x", &schema), "q"),
                            ProjectionExpr::new(physical_expr("x + 1", &schema), "y"),
                        ],
                        scan_batches(&schema, std::slice::from_ref(&batch)),
                    )
                    .unwrap(),
                )
            },
            false,
        )
        .await;
        assert!(error.unwrap().contains("Divide by zero"));
        assert_eq!(operators.projections, 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn lists_of_tiny_values_stay_in_datafusion() {
        // A projection of `f IS NULL`, or of `f` (passed through), above a
        // filter gathers the selected rows of `f`, fixed-size lists of
        // `1 << 30` nulls, runs, empty lists or binaries, or booleans, with
        // Arrow's `take`, whose 32-bit indices overflow (see the test of the
        // same name for filters).
        let cases: [(&[i32], TinyValues, usize); 5] = [
            (&[1, 0, 1, 0, 1, 1], TinyValues::Nulls, 4),
            (&[0, 0, 0, 0, 1, 0, 1, 0], TinyValues::Runs, 2),
            (&[0, 0, 0, 0, 0, 1, 0, 1], TinyValues::EmptyLists, 2),
            (&[0, 0, 0, 0, 0, 1, 0, 1], TinyValues::EmptyBinaries, 2),
            (&[0, 0, 0, 0, 1], TinyValues::Booleans, 1),
        ];
        for (x, values, rows) in cases {
            let batch = wide_lists(x, values);
            let schema = batch.schema();
            for expr in ["f IS NULL", "f"] {
                let plan = || -> Arc<dyn ExecutionPlan> {
                    let filter = FilterExec::try_new(
                        physical_expr("x > 0", &schema),
                        scan_batches(&schema, std::slice::from_ref(&batch)),
                    )
                    .unwrap();
                    Arc::new(
                        ProjectionExec::try_new(
                            [ProjectionExpr::new(physical_expr(expr, &schema), "n")],
                            Arc::new(filter),
                        )
                        .unwrap(),
                    )
                };
                let projection_plan = plan();
                let exec = projection_plan.downcast_ref::<ProjectionExec>().unwrap();
                assert!(
                    !PhysicalProjection::supports_projection_exec(exec),
                    "{expr} of {values:?}"
                );
                let task_ctx = Arc::new(datafusion_execution::TaskContext::default());
                assert_eq!(
                    assert_same_outcome_in_datafusion(plan(), task_ctx).await,
                    Ok(rows),
                    "{expr} of {values:?}"
                );
            }
        }
    }

    #[test]
    fn lists_of_tiny_values_cannot_be_gathered() {
        let list = |values: DataType, size: i32| {
            DataType::FixedSizeList(Arc::new(Field::new_list_field(values, true)), size)
        };
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
        let runs = DataType::RunEndEncoded(
            Arc::new(Field::new("run_ends", DataType::Int64, false)),
            Arc::new(Field::new("values", DataType::Int32, true)),
        );
        for (data_type, gathered) in [
            (list(DataType::Null, 1 << 30), false),
            (list(runs.clone(), 1 << 30), false),
            (DataType::new_list(runs.clone(), true), false),
            (list(list(runs.clone(), 2), 4), false),
            (
                DataType::Struct(
                    vec![Field::new("f", list(runs.clone(), 2), true)].into(),
                ),
                false,
            ),
            // Empty lists take no memory, whatever their type, nor empty
            // binaries, and booleans take a bit.
            (list(list(DataType::Int32, 0), 1 << 30), false),
            (list(list(DataType::Utf8, 0), 2), false),
            (DataType::new_list(list(DataType::Int32, 0), true), false),
            (list(DataType::FixedSizeBinary(0), 1 << 30), false),
            (
                DataType::new_list(DataType::FixedSizeBinary(0), true),
                false,
            ),
            (list(DataType::Boolean, 1 << 30), false),
            (DataType::new_list(DataType::Boolean, true), false),
            (
                list(
                    DataType::Struct(
                        vec![Field::new("b", DataType::Boolean, true)].into(),
                    ),
                    4,
                ),
                false,
            ),
            (list(DataType::Struct(Vec::<Field>::new().into()), 4), false),
            // Rows of dense unions may share the strings of a child, but not
            // those of sparse unions, nor values without offsets.
            (union(DataType::Utf8, UnionMode::Dense), false),
            (union(list(DataType::Int32, 2), UnionMode::Dense), true),
            (
                union(DataType::new_list(DataType::Int8, true), UnionMode::Dense),
                false,
            ),
            (union(DataType::Utf8, UnionMode::Sparse), true),
            (union(DataType::Float64, UnionMode::Dense), true),
            // An empty list or binary, a boolean or a run-end encoded column
            // has one value per row (and binaries of a byte take a byte).
            (list(DataType::Int32, 0), true),
            (DataType::FixedSizeBinary(0), true),
            (list(DataType::FixedSizeBinary(1), 1 << 30), true),
            (DataType::Boolean, true),
            (runs, true),
            (
                DataType::RunEndEncoded(
                    Arc::new(Field::new("ends", DataType::Int64, false)),
                    Arc::new(Field::new("samples", DataType::Int32, true)),
                ),
                false,
            ),
            (list(DataType::Float32, 768), true),
            (DataType::new_list(DataType::Utf8, true), true),
        ] {
            assert_eq!(can_gather(&data_type), gathered, "{data_type}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn operations_on_unsupported_types_stay_in_datafusion() {
        // `NOT` of an integer fails whenever it is evaluated, on an empty batch
        // too, which a pipeline skips.
        let schema = xys(&[]).schema();
        let empty = RecordBatch::new_empty(Arc::clone(&schema));
        let plan = || -> Arc<dyn ExecutionPlan> {
            let not = Arc::new(datafusion_physical_expr::expressions::NotExpr::new(
                physical_expr("x", &schema),
            ));
            Arc::new(
                ProjectionExec::try_new(
                    [ProjectionExpr::new(not, "n")],
                    scan_batches(&schema, std::slice::from_ref(&empty)),
                )
                .unwrap(),
            )
        };
        let projection_plan = plan();
        let exec = projection_plan.downcast_ref::<ProjectionExec>().unwrap();
        assert!(!PhysicalProjection::supports_projection_exec(exec));
        let (error, operators) = assert_native_plan_parity(plan, false).await;
        assert!(error.is_some());
        assert_eq!(operators.projections, 0);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concatenations_of_strings_stay_in_datafusion() {
        let _large = one_large_test_at_a_time().await;
        // `ProjectionExec` evaluates `s || s` on the batch of 8192 rows, 2 GiB
        // of strings, whose 32-bit offsets overflow, but a pipeline on chunks
        // of 2048 rows (see the test of the same name for filters).
        let batch = large_strings();
        let schema = batch.schema();
        let plan = || -> Arc<dyn ExecutionPlan> {
            Arc::new(
                ProjectionExec::try_new(
                    [ProjectionExpr::new(
                        physical_expr("(s || s) IS NOT NULL", &schema),
                        "b",
                    )],
                    scan_batches(&schema, std::slice::from_ref(&batch)),
                )
                .unwrap(),
            )
        };
        let projection_plan = plan();
        let exec = projection_plan.downcast_ref::<ProjectionExec>().unwrap();
        assert!(!PhysicalProjection::supports_projection_exec(exec));
        let task_ctx = Arc::new(datafusion_execution::TaskContext::default());
        let outcome = assert_same_outcome_in_datafusion(plan(), task_ctx).await;
        assert!(
            outcome
                .as_ref()
                .is_err_and(|error| error.contains("Offset overflow error: 2147483648")),
            "{outcome:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn functions_building_values_of_variable_lengths_stay_in_datafusion() {
        // `ProjectionExec` evaluates `repeat(s, n)` on the batch of 8192 rows,
        // 4 GiB of strings, more than DataFusion allows, but a pipeline on
        // chunks of 2048 rows (see the test of the same name for filters).
        let batch = repeated_strings();
        let schema = batch.schema();
        let plan = || -> Arc<dyn ExecutionPlan> {
            Arc::new(
                ProjectionExec::try_new(
                    [ProjectionExpr::new(
                        physical_expr("repeat(s, n) IS NOT NULL", &schema),
                        "b",
                    )],
                    scan_batches(&schema, std::slice::from_ref(&batch)),
                )
                .unwrap(),
            )
        };
        let projection_plan = plan();
        let exec = projection_plan.downcast_ref::<ProjectionExec>().unwrap();
        assert!(!PhysicalProjection::supports_projection_exec(exec));
        let task_ctx = Arc::new(datafusion_execution::TaskContext::default());
        let outcome = assert_same_outcome_in_datafusion(plan(), task_ctx).await;
        assert!(
            outcome
                .as_ref()
                .is_err_and(|error| error.contains("string size overflow on repeat")),
            "{outcome:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn dynamic_expressions_stay_in_datafusion() {
        // `ProjectionExec` evaluates `x <> 0 AND 10 / x > 1` on the batch of
        // 1848 zeros then 2248 ones, more than 20% of which pass `x <> 0`, so
        // it divides all rows, and fails, but a pipeline on chunks of 2048
        // rows, the first of which has fewer than 20% of ones: a dynamic
        // expression, whose children are not the expression it evaluates,
        // stays in DataFusion (see `dynamic_predicates_stay_in_datafusion`).
        let x = Int32Array::from_iter_values(
            std::iter::repeat_n(0, 1848).chain(std::iter::repeat_n(1, 2248)),
        );
        let batch = RecordBatch::try_from_iter([("x", Arc::new(x) as ArrayRef)]).unwrap();
        let schema = batch.schema();
        let plan = || -> Arc<dyn ExecutionPlan> {
            Arc::new(
                ProjectionExec::try_new(
                    [ProjectionExpr::new(
                        dynamic_filter("x <> 0 AND 10 / x > 1", &schema),
                        "b",
                    )],
                    scan_batches(&schema, std::slice::from_ref(&batch)),
                )
                .unwrap(),
            )
        };
        let projection_plan = plan();
        let exec = projection_plan.downcast_ref::<ProjectionExec>().unwrap();
        assert!(!PhysicalProjection::supports_projection_exec(exec));
        let task_ctx = Arc::new(datafusion_execution::TaskContext::default());
        let outcome = assert_same_outcome_in_datafusion(plan(), task_ctx).await;
        assert!(
            outcome
                .as_ref()
                .is_err_and(|error| error.contains("Divide by zero")),
            "{outcome:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn projections_of_types_without_arrays_stay_in_datafusion() {
        // The classification probes `u IS NULL` on a null of the type of `u`,
        // which Arrow cannot build for types it has no arrays of (but a schema
        // can have, e.g. of a table without batches; see
        // `operations_on_types_without_nulls_stay_in_datafusion`).
        for data_type in times_without_arrays() {
            let schema =
                Arc::new(Schema::new(vec![Field::new("u", data_type.clone(), true)]));
            let plan = || -> Arc<dyn ExecutionPlan> {
                let is_null =
                    Arc::new(datafusion_physical_expr::expressions::IsNullExpr::new(
                        physical_expr("u", &schema),
                    ));
                Arc::new(
                    ProjectionExec::try_new(
                        [ProjectionExpr::new(is_null as Arc<dyn PhysicalExpr>, "b")],
                        scan_batches(&schema, &[]),
                    )
                    .unwrap(),
                )
            };
            let projection_plan = plan();
            let exec = projection_plan.downcast_ref::<ProjectionExec>().unwrap();
            assert!(
                !PhysicalProjection::supports_projection_exec(exec),
                "{data_type}"
            );
            let task_ctx = Arc::new(datafusion_execution::TaskContext::default());
            assert_eq!(
                assert_same_outcome_in_datafusion(plan(), task_ctx).await,
                Ok(0),
                "{data_type}"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn casts_of_dense_unions_of_shared_strings_stay_in_datafusion() {
        // Casting a dense union extracts its child of strings first, copying a
        // string of 2 MiB for each of the 1200 rows of the batch of
        // `ProjectionExec` that point to it, whose 32-bit offsets overflow,
        // whatever the type it casts into, but not for the 600 rows of a chunk
        // (see `filters_of_dense_unions_of_shared_strings_stay_in_datafusion`).
        let batch = dense_union_of_a_large_string();
        let schema = batch.schema();
        for to in [DataType::Utf8, DataType::LargeUtf8, DataType::Utf8View] {
            let plan = || -> Arc<dyn ExecutionPlan> {
                let cast =
                    Arc::new(datafusion_physical_expr::expressions::CastExpr::new(
                        physical_expr("u", &schema),
                        to.clone(),
                        None,
                    ));
                Arc::new(
                    ProjectionExec::try_new(
                        [ProjectionExpr::new(cast as Arc<dyn PhysicalExpr>, "c")],
                        scan_batches(&schema, std::slice::from_ref(&batch)),
                    )
                    .unwrap(),
                )
            };
            let projection_plan = plan();
            let exec = projection_plan.downcast_ref::<ProjectionExec>().unwrap();
            assert!(!PhysicalProjection::supports_projection_exec(exec), "{to}");
            let task_ctx = Arc::new(datafusion_execution::TaskContext::default());
            let outcome = assert_same_outcome_in_datafusion(plan(), task_ctx).await;
            assert!(
                outcome
                    .as_ref()
                    .is_err_and(|error| error.to_lowercase().contains("overflow")),
                "{to}: {outcome:?}"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn broadcast_constants_stay_in_datafusion() {
        // `ProjectionExec` evaluates `CASE WHEN x THEN '…' END` on the batch,
        // which copies a string of 2 MiB for each of its 2048 rows with `x`,
        // and panics, but a pipeline on chunks of 2048 rows, on which `CASE`
        // returns the constant itself, or nulls (see the test of the same
        // name for filters). A constant that a projection outputs is copied
        // for every row of the batch too.
        let batch = half_true();
        let schema = batch.schema();
        let plan = || -> Arc<dyn ExecutionPlan> {
            Arc::new(
                ProjectionExec::try_new(
                    [ProjectionExpr::new(case_of_a_large_constant(&schema), "b")],
                    scan_batches(&schema, std::slice::from_ref(&batch)),
                )
                .unwrap(),
            )
        };
        let projection_plan = plan();
        let exec = projection_plan.downcast_ref::<ProjectionExec>().unwrap();
        assert!(!PhysicalProjection::supports_projection_exec(exec));
        let task_ctx = Arc::new(datafusion_execution::TaskContext::default());
        let outcome = assert_same_outcome_in_datafusion(plan(), task_ctx).await;
        assert!(
            outcome
                .as_ref()
                .is_err_and(|error| error.to_lowercase().contains("overflow")),
            "{outcome:?}"
        );
        let constant = |value: &str| {
            let projection = ProjectionExec::try_new(
                [ProjectionExpr::new(
                    Arc::new(Literal::new(datafusion_common::ScalarValue::Utf8(Some(
                        value.to_string(),
                    )))) as Arc<dyn PhysicalExpr>,
                    "c",
                )],
                scan_batches(&schema, std::slice::from_ref(&batch)),
            )
            .unwrap();
            PhysicalProjection::supports_projection_exec(&projection)
        };
        assert!(!constant("x"));
        assert!(constant(""));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn unpacking_repeated_values_stays_in_datafusion() {
        // `ProjectionExec` unpacks the dictionary on all rows of the batch
        // (half of them pass `id < 2048`), and the 32-bit offsets of 2048
        // strings of 2 MiB overflow, but a pipeline evaluates it on chunks of
        // 2048 rows: those of the first one have null keys, and no row of the
        // second one passes `id < 2048`, so `AND` does not unpack them (see
        // the test of the same name for filters).
        let batch = repeated_large_string(4096, 2048);
        let schema = batch.schema();
        let plan = || -> Arc<dyn ExecutionPlan> {
            Arc::new(
                ProjectionExec::try_new(
                    [ProjectionExpr::new(guarded_unpacking(&schema, 2048), "b")],
                    scan_batches(&schema, std::slice::from_ref(&batch)),
                )
                .unwrap(),
            )
        };
        let projection_plan = plan();
        let exec = projection_plan.downcast_ref::<ProjectionExec>().unwrap();
        assert!(!PhysicalProjection::supports_projection_exec(exec));
        let task_ctx = Arc::new(datafusion_execution::TaskContext::default());
        let outcome = assert_same_outcome_in_datafusion(plan(), task_ctx).await;
        assert!(
            outcome
                .as_ref()
                .is_err_and(|error| error.to_lowercase().contains("overflow")),
            "{outcome:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn casts_of_dictionary_keys_stay_in_datafusion() {
        // `ProjectionExec` evaluates the cast of the keys of `d` on all rows of
        // the batch (half of them pass `x = 0`), and fails on `1 << 31`, but a
        // pipeline evaluates it on chunks of 2048 rows, the second of which
        // has a single row that passes `x = 0`, the only one `AND` evaluates
        // the cast on (see the test of the same name for filters).
        let batch = large_dictionary_keys();
        let schema = batch.schema();
        let plan = || -> Arc<dyn ExecutionPlan> {
            Arc::new(
                ProjectionExec::try_new(
                    [ProjectionExpr::new(guarded_key_cast(&schema), "b")],
                    scan_batches(&schema, std::slice::from_ref(&batch)),
                )
                .unwrap(),
            )
        };
        let projection_plan = plan();
        let exec = projection_plan.downcast_ref::<ProjectionExec>().unwrap();
        assert!(!PhysicalProjection::supports_projection_exec(exec));
        let task_ctx = Arc::new(datafusion_execution::TaskContext::default());
        let outcome = assert_same_outcome_in_datafusion(plan(), task_ctx).await;
        assert!(
            outcome
                .as_ref()
                .is_err_and(|error| error.contains("Can't cast value 2147483648")),
            "{outcome:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn projections_of_dictionary_values_stay_in_datafusion() {
        // `FilterExec` coalesces the rows `"1"` and `"2"` into one batch,
        // merging the dictionaries of the two batches without the values that
        // no row references, `"x"` and `"y"`: DataFusion casts `"1"` and `"2"`.
        // The chunks of a pipeline keep their dictionaries: casting `"x"`
        // fails.
        let batches = dictionary_batches();
        let schema = batches[0].schema();
        let plan = || -> Arc<dyn ExecutionPlan> {
            let filter = FilterExec::try_new(
                physical_expr("d <> 'x' AND d <> 'y'", &schema),
                scan_batches(&schema, &batches),
            )
            .unwrap();
            Arc::new(
                ProjectionExec::try_new(
                    [ProjectionExpr::new(
                        physical_expr("CAST(d AS INT)", &schema),
                        "i",
                    )],
                    Arc::new(filter),
                )
                .unwrap(),
            )
        };
        let projection_plan = plan();
        let exec = projection_plan.downcast_ref::<ProjectionExec>().unwrap();
        assert!(!PhysicalProjection::supports_projection_exec(exec));
        let (error, operators) = assert_native_plan_parity(plan, false).await;
        assert!(error.is_none());
        assert_eq!(operators.projections, 0);
        let native = PhysicalProjection::try_new(
            source(Arc::clone(&schema), vec![]),
            vec![(physical_expr("CAST(d AS INT)", &schema), "i".to_owned())],
        )
        .unwrap();
        let chunk = selected_chunk(&batches[0], vec![0]);
        assert!(project(&native, &chunk).is_err());
        // Unpacking the dictionary cannot fail.
        let (error, operators) = assert_native_plan_parity(
            || -> Arc<dyn ExecutionPlan> {
                let filter = FilterExec::try_new(
                    physical_expr("d <> 'x' AND d <> 'y'", &schema),
                    scan_batches(&schema, &batches),
                )
                .unwrap();
                Arc::new(
                    ProjectionExec::try_new(
                        [ProjectionExpr::new(
                            physical_expr("CAST(d AS VARCHAR)", &schema),
                            "v",
                        )],
                        Arc::new(filter),
                    )
                    .unwrap(),
                )
            },
            false,
        )
        .await;
        assert!(error.is_none());
        assert_eq!(operators.projections, 1);
    }

    /// `batch`, with the selection `selection`.
    fn selected_chunk(batch: &RecordBatch, selection: Vec<u32>) -> DataChunk {
        DataChunk::from(batch.clone())
            .with_selection(SelectionVector::from(selection))
            .unwrap()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_projections_of_views_and_dictionaries() {
        assert_all_native_byte_eq(&[
            "SELECT d8, sv, d32 AS d FROM vd WHERE d32 = 'k1'",
            "SELECT i, character_length(sv) AS c FROM vd WHERE i % 5 = 0",
            "SELECT CAST(d32 AS VARCHAR) || '!' AS x FROM vd WHERE sv IS NOT NULL",
            "SELECT i, sv, strpos(sv, 'g') AS g FROM vd WHERE sv LIKE 'a string%' AND i % 7 = 0",
            "SELECT d8 AS d, i + 1 AS j FROM vd WHERE d8 IS NOT NULL",
        ])
        .await;
        // `upper` takes the dictionary itself, and may evaluate all of its
        // values, which depend on DataFusion's batches; `||` and `length` may
        // both fail; `concat` and `substr` build strings, whose sizes
        // DataFusion may limit per batch.
        assert_all_projections_in_datafusion(&[
            "SELECT i, upper(d32) AS u FROM vd WHERE i % 5 = 0",
            "SELECT i, concat(sv, d8) AS c FROM vd WHERE i % 5 = 0",
            "SELECT i, sv, substr(sv, 3) AS tail FROM vd WHERE sv LIKE 'a string%' AND i % 7 = 0",
            "SELECT CAST(d32 AS VARCHAR) || '!' AS x, length(sv) AS l FROM vd WHERE sv IS NOT NULL",
        ])
        .await;
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_projections_with_metadata() {
        let metadata = HashMap::from([("origin".to_owned(), "test".to_owned())]);
        let schema = Arc::new(
            Schema::new(vec![
                Field::new("v", DataType::Int32, true).with_metadata(metadata.clone()),
                Field::new("s", DataType::Utf8View, false),
            ])
            .with_metadata(metadata),
        );
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from_iter(
                    (0..3000).map(|i| (i % 3 != 0).then_some(i)),
                )),
                Arc::new(StringViewArray::from_iter_values(
                    (0..3000).map(|i| format!("value number {i}")),
                )),
            ],
        )
        .unwrap();
        let setup = async |ctx: &SessionContext| {
            register_partitions(
                ctx,
                "meta",
                vec![vec![batch.slice(0, 2500)], vec![batch.slice(2500, 500)]],
            )
        };
        for sql in [
            "SELECT v AS w, s FROM meta WHERE v > 100",
            "SELECT s, v + 1 AS w FROM meta WHERE v % 7 = 0",
            "SELECT s, v FROM meta WHERE v > 100 ORDER BY v",
        ] {
            let run = assert_native_byte_eq_with_setup(sql, &setup).await;
            assert!(
                run.operators.projections + run.operators.filters > 0,
                "{sql}"
            );
        }
    }
}
