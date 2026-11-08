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
use arrow::datatypes::{Schema, SchemaRef};
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

use super::adaptive_filter::depends_on_batches;
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
    /// into batches, empty ones included (see [`depends_on_batches`]).
    ///
    /// Below a limit that stops the execution early (e.g. a
    /// `GlobalLimitExec`), the expressions are evaluated on fewer rows when
    /// the limit is reached than in DataFusion, which evaluates whole
    /// batches: a planner should only lower the projections whose
    /// expressions cannot fail there (see
    /// [`fallibility`](super::adaptive_filter::fallibility)).
    pub fn supports_projection_exec(exec: &ProjectionExec) -> bool {
        let input_schema = exec.input().schema();
        !exec
            .expr()
            .iter()
            .any(|projection| depends_on_batches(&projection.expr, input_schema.as_ref()))
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
            "SELECT id, a + b AS s, c * 2 AS c2, upper(d) AS u, NOT e AS ne FROM t1 WHERE a > 10",
            "SELECT a AS x, id, h, a AS y FROM t1 WHERE a > 10 AND d <> 'beta'",
            "SELECT id, CASE WHEN a IS NULL THEN 'none' WHEN a < 0 THEN 'neg' ELSE 'pos' END AS sign FROM t1 WHERE b > 100",
            "SELECT id, length(h) AS len, concat(d, h) AS cat FROM t1 WHERE h LIKE '%01%'",
            "SELECT 1 AS one, 'x' AS s FROM t1 WHERE a > 0",
            "SELECT id, CAST(g AS DOUBLE) AS gd, f + INTERVAL '1 day' AS next_day, ts FROM t1 WHERE f > DATE '1970-01-01'",
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
            "SELECT i, concat(sv, d8) AS c FROM vd WHERE i % 5 = 0",
            "SELECT CAST(d32 AS VARCHAR) || '!' AS x, length(sv) AS l FROM vd WHERE sv IS NOT NULL",
            "SELECT i, sv, substr(sv, 3) AS tail FROM vd WHERE sv LIKE 'a string%' AND i % 7 = 0",
            "SELECT d8 AS d, i + 1 AS j FROM vd WHERE d8 IS NOT NULL",
        ])
        .await;
        // `upper` takes the dictionary itself, and may evaluate all of its
        // values, which depend on DataFusion's batches.
        for sql in ["SELECT i, upper(d32) AS u FROM vd WHERE i % 5 = 0"] {
            let run = assert_native_byte_eq_with_setup(sql, all_tables).await;
            assert_eq!(run.operators.projections, 0, "native projection: {sql}");
        }
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
