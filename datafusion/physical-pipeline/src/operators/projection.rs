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
//! The evaluation of expressions on some of the rows of a chunk (`BoundExprs`)
//! is shared with [`PhysicalFilter`](super::filter::PhysicalFilter).

use std::sync::Arc;

use arrow::array::{ArrayRef, RecordBatch, RecordBatchOptions};
use arrow::datatypes::{Schema, SchemaRef};
use arrow_select::take::take;
use datafusion_common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion_common::{Result, internal_datafusion_err, internal_err};
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::{Column, LambdaExpr, LambdaVariable};
use datafusion_physical_expr::projection::{ProjectionExpr, ProjectionExprs};
use datafusion_physical_expr::utils::collect_columns;
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::projection::ProjectionExec;
use itertools::Itertools;

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

    /// Lower a DataFusion [`ProjectionExec`]: evaluate its expressions on the
    /// chunks of `input` (the lowered input of `exec`) and produce exactly
    /// its output schema.
    pub fn try_from_projection_exec(
        exec: &ProjectionExec,
        input: Arc<dyn PhysicalOperator>,
    ) -> Result<Self> {
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
/// other columns are never touched.
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
        if exprs.iter().any(has_lambdas) {
            // The parameters of a lambda are numbered after the columns of
            // the input: such expressions see all input columns, unchanged.
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
