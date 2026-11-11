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
//!   the output columns are the input [`ArrayRef`](arrow::array::ArrayRef)s
//!   and the [`SelectionVector`] of the input is kept: no data is copied, not
//!   even for the output of a filter.
//! * Otherwise the expressions are evaluated on a [`RecordBatch`] of the
//!   input columns they reference. If the input carries a selection, only
//!   these columns are compacted (each once, however many expressions use
//!   it), so the expressions only ever see the selected rows: a row removed
//!   by a filter never causes an error, e.g. `SELECT 10 / x ... WHERE x <> 0`
//!   never divides by zero, exactly as in DataFusion where the filter output
//!   is compacted. Input columns that are not referenced are never touched.
//!
//! [`PhysicalProjection::try_from_projection_exec`] lowers every
//! `ProjectionExec`, whatever its expressions may fail with: e.g. the decimal
//! arithmetic of TPC-H (`l_extendedprice * (1 - l_discount) * (1 + l_tax)`),
//! the decimal division of TPC-H Q8 by a sum that may be 0, built-in and
//! user-defined functions, or `x = 0 OR 10 / x > 1`. The projection evaluates
//! them on its chunks rather than on the batches of `ProjectionExec`, so which
//! error it raises when several rows or expressions fail, or whether an
//! expression whose errors in DataFusion depend on its batches fails at all,
//! can differ from DataFusion: that is the engine's own (see
//! `docs/compatibility.md`). Only inputs whose rows the native operators
//! cannot gather are left to DataFusion (see
//! [`PhysicalProjection::supports_projection_exec`]).
//!
//! The evaluation of expressions on some of the rows of a chunk (`BoundExprs`)
//! is shared with [`PhysicalFilter`](super::filter::PhysicalFilter).

use std::sync::Arc;

use arrow::array::{RecordBatch, RecordBatchOptions};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
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

use crate::chunk::{DataChunk, Gather, SelectionVector};
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

    /// Whether [`Self::try_from_projection_exec`] can lower `exec`: whether
    /// the native operators can gather the rows of its input, which has no
    /// lists of values that take less than a byte each (e.g. fixed-size lists
    /// of `1 << 30` nulls, booleans, or empty lists or binaries), which the
    /// native operators cannot gather with `take`, and no run-end encoded
    /// arrays whose child fields are not Arrow's (see `can_gather`).
    ///
    /// Every expression is lowered, whatever it may fail with, see the
    /// [module documentation](self).
    pub fn supports_projection_exec(exec: &ProjectionExec) -> bool {
        exec.input()
            .schema()
            .fields()
            .iter()
            .all(|field| can_gather(field.data_type()))
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
                "PhysicalProjection cannot lower this ProjectionExec: the native operators cannot gather the rows of its input"
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
    /// Without `rows` no data is copied. Otherwise the columns are sliced if
    /// `rows` is a contiguous range, and copied with Arrow's `filter` or
    /// `take` kernels otherwise (see [`Gather`]).
    pub(crate) fn batch(
        &self,
        chunk: &DataChunk,
        rows: Option<&SelectionVector>,
    ) -> Result<RecordBatch> {
        let num_rows = rows.map_or_else(|| chunk.physical_len(), SelectionVector::len);
        let gather = rows.filter(|_| !self.columns.is_empty()).map(|rows| {
            let data_types = self.schema.fields().iter().map(|field| field.data_type());
            Gather::new(rows, chunk.physical_len(), data_types)
        });
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
                match &gather {
                    Some(gather) => gather.column(column),
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
/// (with [`Gather`], e.g. in `DataChunk::compact`): not if they contain, at any
/// depth, lists whose values may take less than a byte each (nulls, booleans,
/// run-end encoded values, of which one run stands for any number of values,
/// empty fixed-size lists of any type or binaries, or structs or fixed-size
/// lists of such values), which may have more values than Arrow's `take` can
/// count (it computes the indices of the values of fixed-size lists in 32
/// bits, which count at most 512 MiB of booleans), and whose cost is
/// proportional to the number of values. DataFusion filters such arrays
/// without copying their values. Nor run-end encoded arrays whose child fields
/// are not Arrow's, which Arrow's `filter` renames.
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

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashMap;

    use arrow::array::{
        ArrayRef, AsArray, Int32Array, Int64Array, StringArray, StringViewArray,
    };
    use arrow::datatypes::{DataType, Field, Int32Type, Int64Type, UnionMode};
    use datafusion::prelude::{SessionConfig, SessionContext};
    use datafusion_common::DataFusionError;
    use datafusion_datasource::memory::MemorySourceConfig;
    use datafusion_execution::TaskContext;
    use datafusion_expr::Operator;
    use datafusion_physical_expr::expressions::BinaryExpr;
    use datafusion_physical_plan::Partitioning;
    use datafusion_physical_plan::filter::FilterExec;
    use datafusion_physical_plan::repartition::RepartitionExec;

    use crate::operators::adaptive_filter::MAX_CONSTANT_BYTES;
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
    fn expressions_with_opaque_leaves_are_bound_to_all_columns() {
        // `opaque(b)` reads `b` without a `Column` child: bound to the columns
        // that the expressions reference, it would read another column.
        let schema = schema();
        let bound = BoundExprs::try_new(
            [opaque_column("b", &schema), physical_expr("a + 1", &schema)],
            &schema,
        )
        .unwrap();
        assert_eq!(
            bound.columns,
            (0..schema.fields().len()).collect::<Vec<_>>()
        );
        assert_eq!(bound.schema.as_ref(), schema.as_ref());
        assert_eq!(bound.exprs()[0].to_string(), "opaque(b@1)");
        assert_eq!(bound.exprs()[1].to_string(), "CAST(a@0 AS Int64) + 1");
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
            "SELECT id, a + b AS s, c * 2 AS c2, d = 'beta' AS u, NOT e AS ne FROM t1 WHERE a > 10",
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
            // Built-in functions whose implementations are known, a widening
            // cast of decimals into floats with another part that may fail,
            // and string constants of bounded sizes, which a batch holds for
            // each of its rows.
            "SELECT id, a + b AS s, length(d) AS u FROM t1 WHERE a > 10",
            "SELECT id, CAST(g AS DOUBLE) AS gd, f + INTERVAL '1 day' AS next_day FROM t1",
            "SELECT id, CASE WHEN a IS NULL THEN 'none' WHEN a < 0 THEN 'neg' ELSE 'pos' END AS sign FROM t1 WHERE b > 100",
            "SELECT 1 AS one, 'x' AS s FROM t1 WHERE a > 0",
        ])
        .await;
        // Also several functions (whose failures are not known) or other parts
        // that may fail, string constants of more than `MAX_CONSTANT_BYTES`,
        // copied for every row of a batch, whose 32-bit offsets overflow on
        // batches with enough rows, and functions that build strings, whose
        // sizes DataFusion may limit per batch.
        let large = format!(
            "SELECT 1 AS one, '{}' AS s FROM t1 WHERE a > 0",
            "x".repeat(MAX_CONSTANT_BYTES + 1)
        );
        assert_all_native_byte_eq(&[
            "SELECT id, length(h) AS len, concat(d, h) AS cat FROM t1 WHERE h LIKE '%01%'",
            "SELECT id, a + b AS s, upper(d) AS u FROM t1 WHERE a > 10",
            "SELECT id, CAST(a AS SMALLINT) AS n, f + INTERVAL '1 day' AS next_day FROM t1",
            "SELECT id, CASE WHEN a < 0 THEN d ELSE 'pos' END AS sign FROM t1 WHERE b > 100",
            &large,
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
    async fn differential_errors_of_native_projections() {
        // Whether DataFusion fails, and whether the native projection fails:
        // all projections are native.
        for (sql, datafusion_fails, native_fails) in [
            // No guard: `x = 0` on half of the rows.
            ("SELECT id, 10 / x AS q FROM dense", true, true),
            // Guarded by the filter, or by `CASE`.
            (
                "SELECT id, 10 / x AS q FROM dense WHERE x <> 0",
                false,
                false,
            ),
            (
                "SELECT id, CASE WHEN x <> 0 THEN 10 / x END AS q FROM dense",
                false,
                false,
            ),
            // `OR` divides all rows of a batch unless at most 20% have
            // `x <> 0`, on the batches of DataFusion and on the chunks of the
            // native projection.
            ("SELECT id, x = 0 OR 10 / x > 1 AS b FROM dense", true, true),
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
                false,
                false,
            ),
        ] {
            let outcome = assert_allowed_native_outcome_with_config(
                sql,
                crate::test_utils::test_session_config(),
                guard_tables,
                &["Divide by zero"],
            )
            .await;
            assert!(outcome.operators.projections > 0, "{sql}");
            assert_eq!(
                outcome.datafusion.is_some(),
                datafusion_fails,
                "{sql}: {outcome:?}"
            );
            assert_eq!(outcome.native.is_some(), native_fails, "{sql}: {outcome:?}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn batch_dependent_projections_are_native() {
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
        let NativeOutcome {
            datafusion: error,
            operators,
            ..
        } = assert_allowed_native_outcome_with_setup(sql, setup, &["Divide by zero"])
            .await;
        assert!(error.unwrap().contains("Divide by zero"));
        // The projection and the filter below it are native: whether the
        // native projection, on the chunks of the filter, divides by zero is
        // the engine's own (behaviour that depends on the batches).
        assert_eq!(operators.projections + operators.filters, 2);

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
        assert!(PhysicalProjection::supports_projection_exec(&exec));
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
    async fn projections_that_fail_on_some_batches_are_native() {
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
            assert!(PhysicalProjection::supports_projection_exec(exec), "{expr}");
            let NativeOutcome {
                datafusion: error,
                operators,
                ..
            } = assert_allowed_native_plan_outcome(
                || plan(expr, &batches),
                false,
                &["did not compile", "Divide by zero"],
            )
            .await;
            let input = format!("{expr} on {} batches", batches.len());
            assert_eq!(error.is_some(), fails, "{input}: {error:?}");
            assert_eq!(operators.projections, 1, "{input}");
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
            let NativeOutcome {
                datafusion: error,
                operators,
                ..
            } = assert_allowed_native_plan_outcome(
                || plan("x = 0 OR (y = 0 OR s ~ 'a+')", &batches),
                false,
                &["did not compile", "Divide by zero"],
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
        // `x`, or no column at all. An opaque expression may also fail
        // depending on the rows it is evaluated on together, so it stays in
        // DataFusion (see `expressions_with_opaque_leaves_are_bound_to_all_columns`
        // for the binding).
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
            let NativeOutcome {
                datafusion: error,
                operators,
                ..
            } = assert_allowed_native_plan_outcome(plan, false, &[]).await;
            assert!(error.is_none(), "{error:?}");
            assert_eq!(operators.projections, 1);
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn case_insensitive_patterns_that_fail_on_unicode_are_native() {
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
        assert!(PhysicalProjection::supports_projection_exec(exec));
        let NativeOutcome {
            datafusion: error,
            operators,
            ..
        } = assert_allowed_native_plan_outcome(plan, false, &["Unable to build regex"])
            .await;
        assert!(error.unwrap().contains("Unable to build regex"));
        assert_eq!(operators.projections, 1);
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
        assert!(PhysicalProjection::supports_projection_exec(exec));
        lower(&projection_plan).unwrap();
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        let NativeOutcome {
            datafusion: error,
            operators,
            ..
        } = assert_allowed_native_plan_outcome(|| plan(&Arc::default()), false, &[])
            .await;
        assert!(error.is_none());
        assert_eq!(operators.projections, 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn case_insensitive_patterns_on_bytes_beneath_nulls_are_native() {
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
        assert!(PhysicalProjection::supports_projection_exec(exec));
        let NativeOutcome {
            datafusion: error,
            operators,
            ..
        } = assert_allowed_native_plan_outcome(plan, false, &["Unable to build regex"])
            .await;
        assert!(error.unwrap().contains("Unable to build regex"));
        assert_eq!(operators.projections, 1);
        let native = PhysicalProjection::try_new(
            source(Arc::clone(&schema), vec![]),
            vec![(physical_expr(&expr, &schema), "m".to_owned())],
        )
        .unwrap();
        let selected = selected_chunk(&batch, (0..2048).step_by(2).collect());
        assert!(project(&native, &selected).is_ok());
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn casts_that_arrow_cannot_evaluate_are_native() {
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
            assert!(PhysicalProjection::supports_projection_exec(exec), "{expr}");
            let NativeOutcome {
                datafusion: error,
                operators,
                ..
            } = assert_allowed_native_plan_outcome(
                plan,
                false,
                &["Unsupported output type for dictionary packing"],
            )
            .await;
            assert!(error.unwrap().contains("dictionary packing"), "{expr}");
            assert_eq!(operators.projections, 1, "{expr}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn lists_built_per_batch_are_native_if_their_rows_can_be_gathered() {
        // On the batch of 4096 rows, the 32-bit offsets of the lists overflow,
        // on each chunk of 2048 rows of a pipeline, they do not: when
        // `CASE WHEN x THEN a ELSE b END` merges the lists of `a` on the first
        // 2048 rows (`1 << 30` nulls) and of `b` on the others (`1 << 30` too),
        // and when a constant list of `1 << 19` nulls is broadcast to the
        // rows. But the native operators cannot gather the rows of the lists
        // of nulls of the input (see `can_gather`): these projections stay in
        // DataFusion.
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
            let task_ctx = Arc::new(TaskContext::default());
            let outcome = assert_same_outcome_in_datafusion(plan(), task_ctx).await;
            assert!(outcome.unwrap_err().contains("overflow"), "{expr}");
        }
        // So do 16-bit run ends: a constant fixed-size list of 8 values in one
        // run, broadcast to the rows, has 32768 values on the batch of 4096
        // rows, and 16384 on a chunk of 2048. The rows of the input can be
        // gathered: the projection is native, and whether it fails depends on
        // the batches (which is the engine's own).
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
        assert!(PhysicalProjection::supports_projection_exec(exec));
        let task_ctx = Arc::new(TaskContext::default());
        assert!(
            assert_allowed_native_outcome(plan(), task_ctx, &["Option::unwrap()"])
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
        let task_ctx = Arc::new(TaskContext::default());
        assert_eq!(
            assert_same_outcome_in_datafusion(plan(), task_ctx).await,
            Ok(4096)
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn projections_with_several_failures_are_native() {
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
        assert!(PhysicalProjection::supports_projection_exec(exec));
        // On one thread, a pipeline evaluates the first chunk first.
        let config = SessionConfig::new().with_target_partitions(1);
        let NativeOutcome {
            datafusion: error,
            operators,
            ..
        } = assert_allowed_native_plan_outcome_with_config(
            plan,
            false,
            config,
            &["Cannot cast string 'invalid'", "Divide by zero"],
        )
        .await;
        assert!(error.unwrap().contains("Cannot cast string 'invalid'"));
        assert_eq!(operators.projections, 1);
        // A single expression that may fail is lowered.
        let NativeOutcome {
            datafusion: error,
            operators,
            ..
        } = assert_allowed_native_plan_outcome(
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
            &["Cannot cast string 'invalid'", "Divide by zero"],
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
                let task_ctx = Arc::new(TaskContext::default());
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
            // Rows of dense unions may share the strings or lists of a child,
            // which gathering them copies for every row, as DataFusion does.
            (union(DataType::Utf8, UnionMode::Dense), true),
            (union(list(DataType::Int32, 2), UnionMode::Dense), true),
            (
                union(DataType::new_list(DataType::Int8, true), UnionMode::Dense),
                true,
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
    async fn operations_on_unsupported_types_are_native() {
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
        assert!(PhysicalProjection::supports_projection_exec(exec));
        let NativeOutcome {
            datafusion: error,
            operators,
            ..
        } = assert_allowed_native_plan_outcome(
            plan,
            false,
            &["could not cast array of type Int32"],
        )
        .await;
        assert!(error.is_some());
        assert_eq!(operators.projections, 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn cases_of_unions_are_native() {
        // `ProjectionExec` evaluates `CASE b WHEN true THEN u END` on the
        // batch, which merges unions without nullable fields with nulls, and
        // fails, but a pipeline on chunks whose rows all take the branch, or
        // none does (see the test of the same name for filters).
        let batch = unions_without_nulls();
        let schema = batch.schema();
        let plan = || -> Arc<dyn ExecutionPlan> {
            Arc::new(
                ProjectionExec::try_new(
                    [ProjectionExpr::new(case_of_unions(&schema), "c")],
                    scan_batches(&schema, std::slice::from_ref(&batch)),
                )
                .unwrap(),
            )
        };
        let projection_plan = plan();
        let exec = projection_plan.downcast_ref::<ProjectionExec>().unwrap();
        assert!(PhysicalProjection::supports_projection_exec(exec));
        let task_ctx = Arc::new(TaskContext::default());
        let outcome = assert_allowed_native_outcome(
            plan(),
            task_ctx,
            &["Cannot extend a union with no nullable fields with nulls"],
        )
        .await;
        assert!(
            outcome.as_ref().is_err_and(|error| error
                .contains("Cannot extend a union with no nullable fields with nulls")),
            "{outcome:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn projections_above_filters_between_failures_are_native() {
        // `FilterExec(x >= 0)` pulls both batches of its input, and the cast of
        // the second one fails, before the projection above it divides by zero
        // the rows of the first one; a native filter would pass them on at
        // once (see `filters_between_failures_stay_in_datafusion`).
        let plan = || -> Arc<dyn ExecutionPlan> {
            let input = filtered_casts();
            let schema = input.schema();
            Arc::new(
                ProjectionExec::try_new(
                    [ProjectionExpr::new(
                        physical_expr("10 / x + c + l", &schema),
                        "v",
                    )],
                    input,
                )
                .unwrap(),
            )
        };
        let config = || SessionConfig::new().with_target_partitions(1);
        let NativeOutcome {
            datafusion: error,
            operators,
            ..
        } = assert_allowed_native_plan_outcome_with_config(
            plan,
            false,
            config(),
            &["Cannot cast string 'invalid'", "Divide by zero"],
        )
        .await;
        let error = error.unwrap();
        assert!(error.contains("Cannot cast string 'invalid'"), "{error}");
        // The filter stays in DataFusion, with its input; the projection
        // above it is lowered.
        assert_eq!((operators.filters, operators.projections), (1, 2));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn matches_of_patterns_that_may_be_scalars_are_native() {
        // `ProjectionExec` matches the strings with the array of patterns that
        // `CASE` returns on its batch, which Arrow compiles, and one is too
        // big, but a pipeline with the scalar that it returns on the first
        // chunk, which Arrow matches with ASCII strings without compiling it
        // (see the test of the same name for filters).
        let batch = guarded_patterns();
        let schema = batch.schema();
        let plan = || -> Arc<dyn ExecutionPlan> {
            Arc::new(
                ProjectionExec::try_new(
                    [ProjectionExpr::new(guarded_pattern_match(&schema), "m")],
                    scan_batches(&schema, std::slice::from_ref(&batch)),
                )
                .unwrap(),
            )
        };
        let projection_plan = plan();
        let exec = projection_plan.downcast_ref::<ProjectionExec>().unwrap();
        assert!(PhysicalProjection::supports_projection_exec(exec));
        let NativeOutcome {
            datafusion: error,
            operators,
            ..
        } = assert_allowed_native_plan_outcome(plan, false, &["Unable to build regex"])
            .await;
        let error = error.unwrap();
        assert!(error.contains("Unable to build regex"), "{error}");
        assert_eq!(operators.projections, 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn negations_of_values_that_may_be_scalars_are_native() {
        // `ProjectionExec` negates the array that `CASE` returns on its batch,
        // wrapping, but a pipeline the scalar that it returns on the first
        // chunk, all of whose rows take the `ELSE` branch, with overflow checks
        // (see the test of the same name for filters).
        let batch = guarded_minimums();
        let schema = batch.schema();
        let plan = || -> Arc<dyn ExecutionPlan> {
            Arc::new(
                ProjectionExec::try_new(
                    [ProjectionExpr::new(negated_case(&schema), "n")],
                    scan_batches(&schema, std::slice::from_ref(&batch)),
                )
                .unwrap(),
            )
        };
        let projection_plan = plan();
        let exec = projection_plan.downcast_ref::<ProjectionExec>().unwrap();
        assert!(PhysicalProjection::supports_projection_exec(exec));
        let NativeOutcome {
            datafusion: error,
            operators,
            ..
        } = assert_allowed_native_plan_outcome(plan, false, &["Arithmetic overflow"])
            .await;
        assert!(error.is_none(), "{error:?}");
        assert_eq!(operators.projections, 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn functions_are_native() {
        let _large = one_large_test_at_a_time().await;
        // `ProjectionExec` evaluates `concat_length(s)`, which builds `s || s`,
        // on the batch of 8192 rows, 2 GiB of strings, whose 32-bit offsets
        // overflow, but a pipeline on chunks of 2048 rows (see the test of the
        // same name for filters).
        let batch = large_strings();
        let schema = batch.schema();
        let plan = || -> Arc<dyn ExecutionPlan> {
            Arc::new(
                ProjectionExec::try_new(
                    [ProjectionExpr::new(concat_length(&schema), "n")],
                    scan_batches(&schema, std::slice::from_ref(&batch)),
                )
                .unwrap(),
            )
        };
        let projection_plan = plan();
        let exec = projection_plan.downcast_ref::<ProjectionExec>().unwrap();
        assert!(PhysicalProjection::supports_projection_exec(exec));
        let task_ctx = Arc::new(TaskContext::default());
        let outcome =
            assert_allowed_native_outcome(plan(), task_ctx, &["Offset overflow error"])
                .await;
        assert!(
            outcome
                .as_ref()
                .is_err_and(|error| error.contains("Offset overflow error: 2147483648")),
            "{outcome:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn concatenations_of_strings_are_native() {
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
        assert!(PhysicalProjection::supports_projection_exec(exec));
        let task_ctx = Arc::new(TaskContext::default());
        let outcome =
            assert_allowed_native_outcome(plan(), task_ctx, &["Offset overflow error"])
                .await;
        assert!(
            outcome
                .as_ref()
                .is_err_and(|error| error.contains("Offset overflow error: 2147483648")),
            "{outcome:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn functions_building_values_of_variable_lengths_are_native() {
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
        assert!(PhysicalProjection::supports_projection_exec(exec));
        let task_ctx = Arc::new(TaskContext::default());
        let outcome = assert_allowed_native_outcome(
            plan(),
            task_ctx,
            &["string size overflow on repeat"],
        )
        .await;
        assert!(
            outcome
                .as_ref()
                .is_err_and(|error| error.contains("string size overflow on repeat")),
            "{outcome:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn dynamic_expressions_are_not_evaluated_before_execution() {
        // `first_call()` is volatile, but hidden in a dynamic expression:
        // evaluating it while checking whether the projection is supported
        // (e.g. to find whether the `LIKE` fails without rows, before finding
        // that it is dynamic) would make it return `false` during the
        // execution (see the test of the same name for filters).
        let batch = RecordBatch::try_from_iter([(
            "x",
            Arc::new(Int32Array::from(vec![1, 2, 3])) as ArrayRef,
        )])
        .unwrap();
        let schema = batch.schema();
        let projection = |expr: Arc<dyn PhysicalExpr>| -> Arc<dyn ExecutionPlan> {
            Arc::new(
                ProjectionExec::try_new(
                    [ProjectionExpr::new(expr, "m")],
                    scan_batches(&schema, std::slice::from_ref(&batch)),
                )
                .unwrap(),
            )
        };
        let calls = Arc::new(std::sync::atomic::AtomicUsize::new(0));
        let plan = projection(dynamic_first_call(&calls, &schema));
        let exec = plan.downcast_ref::<ProjectionExec>().unwrap();
        assert!(PhysicalProjection::supports_projection_exec(exec));
        assert_eq!(calls.load(std::sync::atomic::Ordering::SeqCst), 0);
        // The fallback returns `true` on all rows, as DataFusion does.
        let NativeOutcome {
            datafusion: error,
            operators,
            ..
        } = assert_allowed_native_plan_outcome(
            || projection(dynamic_first_call(&Arc::default(), &schema)),
            false,
            &[],
        )
        .await;
        assert!(error.is_none(), "{error:?}");
        assert_eq!(operators.projections, 1);
        let batches = datafusion_physical_plan::collect(
            projection(dynamic_first_call(&Arc::default(), &schema)),
            Arc::new(TaskContext::default()),
        )
        .await
        .unwrap();
        let values = batches[0].column(0).as_boolean();
        assert_eq!(values.true_count(), 3);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn dynamic_expressions_are_native() {
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
        assert!(PhysicalProjection::supports_projection_exec(exec));
        let task_ctx = Arc::new(TaskContext::default());
        let outcome =
            assert_allowed_native_outcome(plan(), task_ctx, &["Divide by zero"]).await;
        assert!(
            outcome
                .as_ref()
                .is_err_and(|error| error.contains("Divide by zero")),
            "{outcome:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn projections_of_types_without_arrays_are_native() {
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
                PhysicalProjection::supports_projection_exec(exec),
                "{data_type}"
            );
            let task_ctx = Arc::new(TaskContext::default());
            assert_eq!(
                assert_allowed_native_outcome(plan(), task_ctx, &[]).await,
                Ok(0),
                "{data_type}"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn casts_of_dense_unions_of_shared_strings_are_native() {
        let _large = one_large_test_at_a_time().await;
        // Casting a dense union extracts its child of strings first, copying a
        // string of 2 MiB for each of the 1200 rows of the batch of
        // `ProjectionExec` that point to it, whose 32-bit offsets overflow,
        // whatever the type it casts into, but not for the 600 rows of a chunk
        // of the native projection (`docs/compatibility.md` allows the engines
        // to have other capacity limits, and to raise other errors).
        let batch = dense_union_of_a_large_string();
        let schema = batch.schema();
        let halves = [batch.slice(0, 2048), batch.slice(2048, 2048)];
        for to in [DataType::Utf8, DataType::LargeUtf8, DataType::Utf8View] {
            let plan = |batches: &[RecordBatch]| -> Arc<dyn ExecutionPlan> {
                let cast =
                    Arc::new(datafusion_physical_expr::expressions::CastExpr::new(
                        physical_expr("u", &schema),
                        to.clone(),
                        None,
                    ));
                Arc::new(
                    ProjectionExec::try_new(
                        [ProjectionExpr::new(cast as Arc<dyn PhysicalExpr>, "c")],
                        scan_batches(&schema, batches),
                    )
                    .unwrap(),
                )
            };
            let projection_plan = plan(std::slice::from_ref(&batch));
            let exec = projection_plan.downcast_ref::<ProjectionExec>().unwrap();
            assert!(PhysicalProjection::supports_projection_exec(exec), "{to}");
            let task_ctx = Arc::new(TaskContext::default());
            let datafusion =
                outcome(Arc::clone(&projection_plan), Arc::clone(&task_ctx)).await;
            assert!(
                datafusion
                    .as_ref()
                    .is_err_and(|error| error.to_lowercase().contains("overflow")),
                "{to}: {datafusion:?}"
            );
            let operator = lower(&projection_plan).unwrap();
            assert_eq!(NativeOperators::of(&operator).projections, 1, "{to}");
            let native = executable(operator, projection_plan).unwrap();
            let native = outcome(native, Arc::clone(&task_ctx)).await;
            // The cast returns nulls for the integers of the union, in a column
            // that the plan declares non-nullable: on each half of the batch,
            // DataFusion does not overflow, and fails on these nulls, as the
            // native projection does on the whole batch.
            for half in &halves {
                let expected =
                    outcome(plan(std::slice::from_ref(half)), Arc::clone(&task_ctx))
                        .await;
                assert!(
                    expected
                        .as_ref()
                        .is_err_and(|error| error.contains("non-nullable")),
                    "{to}: {expected:?}"
                );
            }
            assert!(
                native
                    .as_ref()
                    .is_err_and(|error| error.contains("non-nullable")),
                "{to}: {native:?}"
            );
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn broadcast_constants_are_native() {
        // `ProjectionExec` evaluates `CASE WHEN x THEN '…' END` on the batch,
        // which copies a string of 2 MiB for each of its 2048 rows with `x`,
        // and panics, but a pipeline on chunks of 2048 rows, on which `CASE`
        // returns the constant itself, or nulls (see the test of the same
        // name for filters). A constant that a projection outputs is copied
        // for every row of the batch too, unless it has at most
        // `MAX_CONSTANT_BYTES` bytes, which batches hold for each of their
        // rows.
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
        assert!(PhysicalProjection::supports_projection_exec(exec));
        let task_ctx = Arc::new(TaskContext::default());
        let outcome =
            assert_allowed_native_outcome(plan(), task_ctx, &["offset overflow"]).await;
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
        let max = MAX_CONSTANT_BYTES;
        assert!(constant(&"x".repeat(max + 1)));
        assert!(constant(&"x".repeat(2 << 20)));
        assert!(constant(&"x".repeat(max)));
        assert!(constant("x"));
        assert!(constant(""));
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn unpacking_repeated_values_is_native() {
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
        assert!(PhysicalProjection::supports_projection_exec(exec));
        let task_ctx = Arc::new(TaskContext::default());
        let outcome =
            assert_allowed_native_outcome(plan(), task_ctx, &["Offset overflow error"])
                .await;
        assert!(
            outcome
                .as_ref()
                .is_err_and(|error| error.to_lowercase().contains("overflow")),
            "{outcome:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn casts_of_dictionary_keys_are_native() {
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
        assert!(PhysicalProjection::supports_projection_exec(exec));
        let task_ctx = Arc::new(TaskContext::default());
        let outcome = assert_allowed_native_outcome(
            plan(),
            task_ctx,
            &["Can't cast value 2147483648 to type Int32"],
        )
        .await;
        assert!(
            outcome
                .as_ref()
                .is_err_and(|error| error.contains("Can't cast value 2147483648")),
            "{outcome:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn projections_of_dictionary_values_are_native() {
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
        assert!(PhysicalProjection::supports_projection_exec(exec));
        let NativeOutcome {
            datafusion: error,
            operators,
            ..
        } = assert_allowed_native_plan_outcome(plan, false, &["Cannot cast string"])
            .await;
        assert!(error.is_none());
        assert_eq!(operators.projections, 1);
        let native = PhysicalProjection::try_new(
            source(Arc::clone(&schema), vec![]),
            vec![(physical_expr("CAST(d AS INT)", &schema), "i".to_owned())],
        )
        .unwrap();
        let chunk = selected_chunk(&batches[0], vec![0]);
        assert!(project(&native, &chunk).is_err());
        // Unpacking the dictionary cannot fail.
        let NativeOutcome {
            datafusion: error,
            operators,
            ..
        } = assert_allowed_native_plan_outcome(
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
            &["Cannot cast string"],
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
            "SELECT CAST(d32 AS VARCHAR) || '!' AS x FROM vd WHERE sv IS NOT NULL",
            "SELECT d8 AS d, i + 1 AS j FROM vd WHERE d8 IS NOT NULL",
        ])
        .await;
        // Built-in functions whose implementations are known, of views.
        assert_all_native_byte_eq(&[
            "SELECT i, character_length(sv) AS c FROM vd WHERE i % 5 = 0",
            "SELECT i, sv, substr(sv, 3) AS tail FROM vd WHERE sv LIKE 'a string%' AND i % 7 = 0",
            "SELECT i, upper(sv) AS u, btrim(sv, 'as') AS t FROM vd WHERE i % 3 = 0",
            "SELECT CAST(d32 AS VARCHAR) || '!' AS x, length(sv) AS l FROM vd WHERE sv IS NOT NULL",
        ])
        .await;
        // `strpos` copies the substring it searches for every row of a batch
        // (see `make_scalar_function`), on batches of any size.
        let sql = "SELECT i, sv, strpos(sv, 'g') AS g FROM vd WHERE sv LIKE 'a string%' AND i % 7 = 0";
        for config in [unrepartitioned(), crate::test_utils::test_session_config()] {
            let run = assert_native_byte_eq_with_config(sql, config, all_tables).await;
            assert!(run.operators.projections > 0, "no native projection: {sql}");
        }
        // Functions of dictionaries (e.g. `upper` takes the dictionary itself,
        // and may evaluate all of its values, which depend on the batches).
        assert_all_native_byte_eq(&[
            "SELECT i, upper(d32) AS u FROM vd WHERE i % 5 = 0",
            "SELECT i, length(d8) AS u FROM vd WHERE i % 5 = 0",
            "SELECT i, concat(sv, d8) AS c FROM vd WHERE i % 5 = 0",
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

    /// Assert that `queries` (on [`tpch_tables`]) return byte-identical
    /// results natively, that every projection and filter of their plans can
    /// be lowered, and that their projections are native (but those that
    /// DataFusion executes because a node above needs their partitions, e.g.
    /// a `SortPreservingMergeExec`).
    async fn assert_all_lowered_natively(queries: &[&str]) {
        for sql in queries {
            let run = assert_native_byte_eq_with_setup(sql, tpch_tables).await;
            let (projections, _) = assert_all_lowerable(&run.expected.plan);
            assert!(projections > 0, "no projection: {sql}");
            assert!(run.operators.projections > 0, "no native projection: {sql}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_tpch_projections_are_native() {
        assert_all_lowered_natively(&[
            // Decimal arithmetic chains (Q1, Q3, Q5, Q7 to Q10, Q14, Q19): their
            // parts may only fail with arithmetic overflows.
            "SELECT l_orderkey, l_extendedprice * (1 - l_discount) * (1 + l_tax) AS charge \
             FROM lineitem WHERE l_shipdate <= DATE '1998-09-02'",
            "SELECT l_returnflag, l_linestatus, sum(l_quantity) AS sum_qty, \
             sum(l_extendedprice) AS sum_base_price, \
             sum(l_extendedprice * (1 - l_discount)) AS sum_disc_price, \
             sum(l_extendedprice * (1 - l_discount) * (1 + l_tax)) AS sum_charge, \
             avg(l_quantity) AS avg_qty, avg(l_discount) AS avg_disc, count(*) AS count_order \
             FROM lineitem WHERE l_shipdate <= DATE '1998-09-02' \
             GROUP BY l_returnflag, l_linestatus ORDER BY l_returnflag, l_linestatus",
            "SELECT l_orderkey, l_extendedprice * (1 - l_discount) - l_tax * l_quantity AS amount \
             FROM lineitem",
            // `extract(year from ...)` (Q7 to Q9), with decimal arithmetic.
            "SELECT extract(year from l_shipdate) AS l_year, \
             l_extendedprice * (1 - l_discount) AS volume FROM lineitem \
             WHERE l_shipdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'",
            "SELECT o_year, sum(volume) AS revenue FROM (\
             SELECT extract(year from o_orderdate) AS o_year, \
             o_totalprice * (1 + o_custkey % 3) AS volume FROM orders) \
             GROUP BY o_year ORDER BY o_year",
            // `substring(c_phone from 1 for 2)` (Q22).
            "SELECT substring(c_phone from 1 for 2) AS cntrycode, c_acctbal FROM customer \
             WHERE substring(c_phone from 1 for 2) IN ('13', '31', '23', '29', '30', '18', '17') \
             AND c_acctbal > 0.00",
            "SELECT cntrycode, count(*) AS numcust, sum(c_acctbal) AS totacctbal FROM (\
             SELECT substring(c_phone from 1 for 2) AS cntrycode, c_acctbal FROM customer \
             WHERE substring(c_phone from 1 for 2) IN ('13', '31', '23', '29', '30', '18', '17') \
             AND c_acctbal > 0.00) GROUP BY cntrycode ORDER BY cntrycode",
            // `coalesce` (a `CASE` of views), widening decimal casts and
            // casts into floats (Q14, Q22), other built-in functions.
            "SELECT c_custkey, coalesce(c_phone, 'unknown') AS phone, \
             CAST(c_acctbal AS DECIMAL(19, 6)) AS balance FROM customer",
            "SELECT 100.00 * CAST(promo AS DOUBLE) / CAST(total AS DOUBLE) AS promo_revenue FROM (\
             SELECT sum(CASE WHEN l_shipmode LIKE 'AIR%' THEN l_extendedprice * (1 - l_discount) \
             ELSE 0 END) AS promo, sum(l_extendedprice * (1 - l_discount)) AS total \
             FROM lineitem WHERE l_shipdate >= DATE '1995-09-01' AND l_shipdate < DATE '1996-09-01')",
            "SELECT o_orderkey, upper(o_orderpriority) AS p, \
             character_length(o_orderpriority) AS n, date_part('month', o_orderdate) AS m, \
             abs(o_totalprice) AS a FROM orders \
             WHERE starts_with(o_orderpriority, '1') OR o_orderkey % 10 = 0",
            // Of strings with 32-bit offsets, the functions that return parts
            // of them.
            "SELECT o_orderkey, character_length(o_comment) AS n, octet_length(o_comment) AS b, \
             substr(o_comment, 9, 3) AS c, trim(o_comment) AS t, \
             nullif(o_comment, 'comment 7 special requests') AS u FROM orders",
        ])
        .await;
        // Constant strings of bounded sizes (Q12-style priorities), which are
        // copied for every row, on batches whose rows the plan bounds: the
        // batches of the scans of the 3 partitions of the tables, which a
        // repartition into 4 partitions (before a filter) does not bound (see
        // `max_batch_rows`).
        for (sql, _repartitioned) in [
            (
                "SELECT o_orderkey, CASE WHEN o_orderpriority = '1-URGENT' \
                 OR o_orderpriority = '2-HIGH' THEN '1-URGENT' ELSE '2-HIGH' END AS priority \
                 FROM orders WHERE o_orderdate >= DATE '1994-01-01'",
                true,
            ),
            (
                "SELECT o_orderkey, CASE WHEN o_totalprice > 250000.00 THEN 'HIGH' \
                 WHEN o_totalprice > 100000.00 THEN 'MEDIUM' ELSE 'LOW' END AS level, \
                 'TPC-H' AS benchmark FROM orders",
                false,
            ),
        ] {
            let run =
                assert_native_byte_eq_with_config(sql, unrepartitioned(), tpch_tables)
                    .await;
            let (projections, _) = assert_all_lowerable(&run.expected.plan);
            assert!(projections > 0, "no projection: {sql}");
            assert!(run.operators.projections > 0, "no native projection: {sql}");
            let run = assert_native_byte_eq_with_setup(sql, tpch_tables).await;
            assert!(run.operators.projections > 0, "no native projection: {sql}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn tpch_projections_are_native() {
        // The sessions of `test_session_config` (4 threads) do not register the
        // functions they trust (see `TrustedFunctions`): by default, they trust
        // the built-in functions of DataFusion, so that the
        // `PipelineQueryPlanner` lowers the projections of TPC-H natively, with
        // every node of their plans (see
        // `builtin_functions_of_tpch_filters_are_native_by_default` for the
        // filters).
        let config = crate::test_utils::test_session_config();
        assert!(
            config
                .options()
                .extensions
                .get::<crate::operators::adaptive_filter::TrustedFunctions>()
                .is_none()
        );
        for sql in [
            // `extract(year from ...)` (Q7 to Q9), with decimal arithmetic.
            "SELECT extract(year from l_shipdate) AS l_year, \
             l_extendedprice * (1 - l_discount) AS volume FROM lineitem \
             WHERE l_shipdate BETWEEN DATE '1995-01-01' AND DATE '1996-12-31'",
            "SELECT o_year, sum(volume) AS revenue FROM (\
             SELECT extract(year from o_orderdate) AS o_year, \
             l_extendedprice * (1 - l_discount) AS volume \
             FROM lineitem, orders WHERE l_orderkey = o_orderkey) \
             GROUP BY o_year ORDER BY o_year",
            // `substring(c_phone from 1 for 2)` (Q22), in a projection and in a
            // filter.
            "SELECT cntrycode, count(*) AS numcust, sum(c_acctbal) AS totacctbal FROM (\
             SELECT substring(c_phone from 1 for 2) AS cntrycode, c_acctbal FROM customer \
             WHERE substring(c_phone from 1 for 2) IN ('13', '31', '23', '29', '30', '18', '17') \
             AND c_acctbal > 0.00) GROUP BY cntrycode ORDER BY cntrycode",
            // Constant strings of bounded sizes, on the batches of a scan of
            // memory (which the plan bounds).
            "SELECT o_orderkey, CASE WHEN o_orderpriority = '1-URGENT' \
             OR o_orderpriority = '2-HIGH' THEN '1-URGENT' ELSE '2-HIGH' END AS priority \
             FROM orders",
            // The decimal division of Q8 by a divisor that is not 0 where it
            // divides (a `NULLIF` of `0`, a built-in function, or a `CASE`),
            // which can only fail with arithmetic overflows: on any number of
            // threads.
            "SELECT o_year, sum(CASE WHEN o_orderpriority = '1-URGENT' THEN volume ELSE 0 END) \
             / NULLIF(sum(volume), 0) AS mkt_share FROM (\
             SELECT extract(year from o_orderdate) AS o_year, \
             l_extendedprice * (1 - l_discount) AS volume, o_orderpriority \
             FROM lineitem, orders WHERE l_orderkey = o_orderkey) \
             GROUP BY o_year ORDER BY o_year",
            "SELECT o_year, CASE WHEN sum(volume) <> 0 THEN \
             sum(CASE WHEN o_orderpriority = '1-URGENT' THEN volume ELSE 0 END) \
             / sum(volume) END AS mkt_share FROM (\
             SELECT extract(year from o_orderdate) AS o_year, \
             l_extendedprice * (1 - l_discount) AS volume, o_orderpriority \
             FROM lineitem, orders WHERE l_orderkey = o_orderkey) \
             GROUP BY o_year ORDER BY o_year",
        ] {
            let (plan, root) =
                assert_planned_byte_eq(sql, config.clone(), tpch_tables).await;
            let (projections, filters) = assert_all_lowerable(&plan);
            assert!(projections > 0, "no projection: {sql}");
            assert!(
                crate::bridge::bridged_plans(&root).is_empty(),
                "{sql}\n{}",
                crate::explain::format_operator_tree(&root)
            );
            let native = NativeOperators::of(&root);
            assert_eq!(
                (native.projections, native.filters),
                (projections, filters),
                "{sql}"
            );
        }
        // Also in a session that trusts no function (whose calls are opaque to
        // the classification of errors).
        let untrusting = config.clone().with_option_extension(
            crate::operators::adaptive_filter::TrustedFunctions::default(),
        );
        let sql = "SELECT substring(c_phone from 1 for 2) AS cntrycode, c_acctbal \
                   FROM customer";
        let (plan, root) = assert_planned_byte_eq(sql, untrusting, tpch_tables).await;
        let projection = plan.downcast_ref::<ProjectionExec>().expect("a projection");
        assert!(PhysicalProjection::supports_projection_exec(projection));
        assert_eq!(NativeOperators::of(&root).projections, 1, "{sql}");
        // And the decimal division of Q8 by a sum that may be 0, which may fail
        // with a division by zero (on a sum of 0) and with an overflow (on a
        // quotient beyond the precision of its type): which of them is raised
        // is the engine's own (see `docs/compatibility.md`).
        let sql = "SELECT o_year, sum(CASE WHEN o_orderpriority = '1-URGENT' THEN volume ELSE 0 END) \
                   / sum(volume) AS mkt_share FROM (\
                   SELECT extract(year from o_orderdate) AS o_year, \
                   l_extendedprice * (1 - l_discount) AS volume, o_orderpriority \
                   FROM lineitem, orders WHERE l_orderkey = o_orderkey) \
                   GROUP BY o_year ORDER BY o_year";
        let (_, root) = assert_planned_byte_eq(sql, config, tpch_tables).await;
        assert!(
            crate::bridge::bridged_plans(&root).is_empty(),
            "{sql}\n{}",
            crate::explain::format_operator_tree(&root)
        );
    }

    /// A session configuration whose plans scan the 3 partitions of the
    /// tables of the tests without repartitioning them, so that their nodes
    /// bound the rows of the batches of DataFusion (see `max_batch_rows`).
    fn unrepartitioned() -> SessionConfig {
        SessionConfig::new().with_target_partitions(3)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_decimal_overflows_of_native_operators() {
        // `a * b * c` overflows `Decimal128` on some rows of every partition:
        // the first product on the rows of `x = 1`, the second one on the rows
        // of `x = 2`. Whatever the batches and the order of the rows, both
        // engines fail with an arithmetic overflow.
        let setup = async |ctx: &SessionContext| {
            let schema = Arc::new(Schema::new(vec![
                Field::new("x", DataType::Int32, false),
                Field::new("a", DataType::Decimal128(38, 0), false),
                Field::new("b", DataType::Decimal128(38, 0), false),
                Field::new("c", DataType::Decimal128(38, 0), false),
                Field::new("s", DataType::Utf8, false),
            ]));
            let batch = |rows: std::ops::Range<i32>| {
                let x = Int32Array::from_iter_values(rows.clone().map(|i| i % 3));
                let decimal = |values: Vec<i128>| -> ArrayRef {
                    Arc::new(
                        arrow::array::Decimal128Array::from(values)
                            .with_precision_and_scale(38, 0)
                            .unwrap(),
                    )
                };
                let a = decimal(
                    rows.clone()
                        .map(|i| if i % 3 == 1 { 10_i128.pow(20) } else { 7 })
                        .collect(),
                );
                let b = decimal(
                    rows.clone()
                        .map(|i| {
                            if i % 3 == 1 {
                                10_i128.pow(19)
                            } else {
                                10_i128.pow(18)
                            }
                        })
                        .collect(),
                );
                let c = decimal(
                    rows.clone()
                        .map(|i| if i % 3 == 2 { 10_i128.pow(20) } else { 3 })
                        .collect(),
                );
                let s = StringArray::from_iter_values(
                    rows.map(|i| if i % 3 == 0 { "1" } else { "x" }),
                );
                RecordBatch::try_new(
                    Arc::clone(&schema),
                    vec![Arc::new(x), a, b, c, Arc::new(s)],
                )
                .unwrap()
            };
            register_partitions(
                ctx,
                "overflows",
                vec![
                    vec![batch(0..3000), batch(3000..3100)],
                    vec![batch(3100..5000)],
                ],
            )
        };
        // Whether the native operators fail (DataFusion fails on the same
        // queries, but for the guarded one).
        for (sql, _, native_fails) in [
            // Several products that may only overflow.
            ("SELECT a * b * c AS v FROM overflows", true, true),
            ("SELECT a * b AS ab, b * c AS bc FROM overflows", true, true),
            (
                "SELECT x, a * b * c AS v FROM overflows WHERE x = 0",
                true,
                false,
            ),
            (
                "SELECT a * b * c AS v FROM overflows WHERE x = 2",
                true,
                true,
            ),
            ("SELECT x FROM overflows WHERE a * b * c > 0", true, true),
            // After a guard: DataFusion decides per batch on which rows it
            // evaluates them (here on all of them: a third of the rows have
            // `x = 0`), the native filter only evaluates them on the rows
            // with `x = 0`, which do not overflow.
            (
                "SELECT x FROM overflows WHERE x = 0 AND a * b > c * a",
                true,
                false,
            ),
            (
                "SELECT a * b AS v FROM overflows WHERE b * c > 0",
                true,
                true,
            ),
            // With a part that fails otherwise (a cast of strings): which error
            // comes first is the engine's own.
            (
                "SELECT a * b + CAST(s AS DECIMAL(38, 0)) AS v FROM overflows",
                true,
                true,
            ),
            (
                "SELECT a * b + CAST(s AS DECIMAL(38, 0)) AS v FROM overflows WHERE x = 0",
                true,
                false,
            ),
        ] {
            let outcome = assert_allowed_native_outcome_with_config(
                sql,
                crate::test_utils::test_session_config(),
                &setup,
                &["Overflow", "Cannot cast"],
            )
            .await;
            assert!(
                outcome.operators.projections + outcome.operators.filters > 0,
                "{sql}"
            );
            assert_eq!(outcome.native.is_some(), native_fails, "{sql}: {outcome:?}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn decimal_overflows_raise_the_same_error_on_chunks() {
        // One batch of 4096 rows: on the first 2048 rows, the second product of
        // `a * b * c` overflows, on the others the first one. `ProjectionExec`
        // computes `a * b` on all rows first, and fails on row 2048; a pipeline
        // computes both products on the first chunk first, and fails on row 0.
        // The rows differ, but both fail with an arithmetic overflow.
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Decimal128(38, 0), false),
            Field::new("b", DataType::Decimal128(38, 0), false),
            Field::new("c", DataType::Decimal128(38, 0), false),
        ]));
        let decimal = |values: Vec<i128>| -> ArrayRef {
            Arc::new(
                arrow::array::Decimal128Array::from(values)
                    .with_precision_and_scale(38, 0)
                    .unwrap(),
            )
        };
        let first = |i: usize| i < 2048;
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                decimal(
                    (0..4096)
                        .map(|i| if first(i) { 10 } else { 10_i128.pow(20) })
                        .collect(),
                ),
                decimal((0..4096).map(|_| 10_i128.pow(19)).collect()),
                decimal(
                    (0..4096)
                        .map(|i| if first(i) { 10_i128.pow(19) } else { 1 })
                        .collect(),
                ),
            ],
        )
        .unwrap();
        let expr = physical_expr("a * b * c", &schema);
        let plan = || -> Arc<dyn ExecutionPlan> {
            Arc::new(
                ProjectionExec::try_new(
                    [ProjectionExpr::new(Arc::clone(&expr), "v")],
                    scan_batches(&schema, std::slice::from_ref(&batch)),
                )
                .unwrap(),
            )
        };
        let projection_plan = plan();
        let exec = projection_plan.downcast_ref::<ProjectionExec>().unwrap();
        assert!(PhysicalProjection::supports_projection_exec(exec));
        let config = SessionConfig::new().with_target_partitions(1);
        let (error, operators) =
            assert_native_plan_parity_with_config(plan, false, config).await;
        let error = error.unwrap();
        assert!(error.contains("Arithmetic overflow"), "{error}");
        assert_eq!(operators.projections, 1);
        // The errors are raised on other rows.
        let datafusion_error =
            datafusion_physical_plan::collect(plan(), Arc::new(TaskContext::default()))
                .await
                .unwrap_err()
                .to_string();
        assert!(
            datafusion_error.contains(&format!(
                "{} * {}",
                10_i128.pow(20),
                10_i128.pow(19)
            )),
            "{datafusion_error}"
        );
        let native = PhysicalProjection::try_new(
            source(Arc::clone(&schema), vec![]),
            vec![(Arc::clone(&expr), "v".to_owned())],
        )
        .unwrap();
        let native_error = project(&native, &DataChunk::from(batch.slice(0, 2048)))
            .unwrap_err()
            .to_string();
        assert!(
            native_error.contains(&format!("{} * {}", 10_i128.pow(20), 10_i128.pow(19))),
            "{native_error}"
        );
        assert!(
            native_error.contains("Arithmetic overflow"),
            "{native_error}"
        );
    }

    /// A batch of 4096 rows of `flag` (`Boolean`), `true` on the first 2048
    /// rows and `false` on the others, of `fmt` (`Utf8`), `%Y-%m-%d`, and of
    /// `s` (`Utf8`), null: on the chunks of 2048 rows of a pipeline, a `CASE`
    /// of `flag` returns the value of a branch as is, a scalar, and an array on
    /// the batch.
    fn flags_and_formats() -> RecordBatch {
        RecordBatch::try_from_iter([
            (
                "flag",
                Arc::new(arrow::array::BooleanArray::from_iter(
                    (0..4096).map(|row| Some(row < 2048)),
                )) as ArrayRef,
            ),
            (
                "fmt",
                Arc::new(StringArray::from(vec!["%Y-%m-%d"; 4096])) as ArrayRef,
            ),
            (
                "s",
                Arc::new(StringArray::from(vec![None::<&str>; 4096])) as ArrayRef,
            ),
        ])
        .unwrap()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn functions_of_values_that_may_be_scalars_are_native() {
        // Functions may evaluate scalars and arrays differently: `to_timestamp`
        // of a string and an array of formats fails on a scalar string, which a
        // `CASE` returns on the first chunk of a pipeline, but not on the array
        // that it returns on the batch; `regexp_like` compiles a scalar pattern
        // (`[`, which does not compile) even if all strings are null, but the
        // patterns of an array only for strings that are not null.
        let batch = flags_and_formats();
        let schema = batch.schema();
        for expr in [
            "to_timestamp(CASE WHEN flag THEN '2020-01-01' WHEN NOT flag THEN '2020-01-02' END, fmt)",
            "regexp_like(s, CASE WHEN flag THEN '[' WHEN NOT flag THEN 'a' END, \
             CASE WHEN flag THEN 'i' WHEN NOT flag THEN 'm' END)",
        ] {
            let expr = physical_expr(expr, &schema);
            let plan = || -> Arc<dyn ExecutionPlan> {
                Arc::new(
                    ProjectionExec::try_new(
                        [ProjectionExpr::new(Arc::clone(&expr), "v")],
                        scan_batches(&schema, std::slice::from_ref(&batch)),
                    )
                    .unwrap(),
                )
            };
            let projection_plan = plan();
            let exec = projection_plan.downcast_ref::<ProjectionExec>().unwrap();
            assert!(PhysicalProjection::supports_projection_exec(exec), "{expr}");
            let NativeOutcome {
                datafusion: error,
                operators,
                ..
            } = assert_allowed_native_plan_outcome(
                plan,
                false,
                &["Unsupported data type", "did not compile"],
            )
            .await;
            assert!(error.is_none(), "{expr}: {error:?}");
            assert_eq!(operators.projections, 1, "{expr}");
            // On the first chunk of a pipeline, it fails.
            let native = PhysicalProjection::try_new(
                source(Arc::clone(&schema), vec![]),
                vec![(Arc::clone(&expr), "v".to_owned())],
            )
            .unwrap();
            assert!(
                project(&native, &DataChunk::from(batch.slice(0, 2048))).is_err(),
                "{expr}"
            );
        }
    }

    /// A projection of `expr` (on the schema of `batch`) over a scan of `batch`,
    /// whether it is supported, and the native projection of `expr`.
    fn project_batch(
        expr: &str,
        batch: &RecordBatch,
    ) -> (Arc<dyn ExecutionPlan>, bool, PhysicalProjection) {
        let schema = batch.schema();
        let expr = physical_expr(expr, &schema);
        let plan: Arc<dyn ExecutionPlan> = Arc::new(
            ProjectionExec::try_new(
                [ProjectionExpr::new(Arc::clone(&expr), "v")],
                scan_batches(&schema, std::slice::from_ref(batch)),
            )
            .unwrap(),
        );
        let supported = PhysicalProjection::supports_projection_exec(
            plan.downcast_ref::<ProjectionExec>().unwrap(),
        );
        let native = PhysicalProjection::try_new(
            source(schema, vec![]),
            vec![(expr, "v".to_owned())],
        )
        .unwrap();
        (plan, supported, native)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn regexp_like_with_flags_of_rows_is_native() {
        // With flags of each row, `regexp_like` checks the flags of the whole
        // batch for `g` before it compiles any pattern: DataFusion fails to
        // plan the global option on the batch of 4096 rows, but a pipeline
        // would fail to compile `[` on its first chunk, whose flags are `i`.
        let rows = 4096;
        let batch = RecordBatch::try_from_iter([
            (
                "s",
                Arc::new(StringArray::from(vec!["a"; rows])) as ArrayRef,
            ),
            (
                "p",
                Arc::new(StringArray::from_iter_values(
                    (0..rows).map(|row| if row < 2048 { "[" } else { "a" }),
                )),
            ),
            (
                "f",
                Arc::new(StringArray::from_iter_values(
                    (0..rows).map(|row| if row < 2048 { "i" } else { "g" }),
                )),
            ),
        ])
        .unwrap();
        let (plan, supported, native) = project_batch("regexp_like(s, p, f)", &batch);
        assert!(supported);
        let task_ctx = Arc::new(TaskContext::default());
        let outcome = assert_allowed_native_outcome(
            plan,
            task_ctx,
            &[
                "Error during planning",
                "did not compile",
                "unclosed character class",
            ],
        )
        .await;
        assert!(
            outcome
                .as_ref()
                .is_err_and(|error| error.contains("Error during planning")
                    && error.contains("global")),
            "{outcome:?}"
        );
        let error = project(&native, &DataChunk::from(batch.slice(0, 2048))).unwrap_err();
        assert!(
            matches!(error.find_root(), DataFusionError::Execution(_)),
            "{error}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn regexp_like_with_large_flags_of_rows_is_native() {
        let _large = one_large_test_at_a_time().await;
        // `regexp_like` copies flags of each row that are views into 32-bit
        // offsets for the whole batch: 4096 flags of 512 KiB overflow them
        // (2 GiB) on the batch of DataFusion, which panics, but not on chunks
        // of 2048 rows (no pattern compiles: all strings are null).
        let rows = 4096;
        let flag = "i".repeat(512 * 1024);
        let mut flags = arrow::array::StringViewBuilder::new();
        let block = flags.append_block(arrow::buffer::Buffer::from(flag.as_bytes()));
        for _ in 0..rows {
            flags.try_append_view(block, 0, flag.len() as u32).unwrap();
        }
        let batch = RecordBatch::try_from_iter([
            (
                "s",
                Arc::new(StringArray::from(vec![None::<&str>; rows])) as ArrayRef,
            ),
            ("f", Arc::new(flags.finish())),
        ])
        .unwrap();
        let (plan, supported, _) = project_batch("regexp_like(s, 'a', f)", &batch);
        assert!(supported);
        let task_ctx = Arc::new(TaskContext::default());
        let outcome =
            assert_allowed_native_outcome(plan, task_ctx, &["offset overflow"]).await;
        assert!(
            outcome
                .as_ref()
                .is_err_and(|error| error.contains("offset overflow")),
            "{outcome:?}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn to_timestamp_of_numbers_is_native() {
        // `to_timestamp` casts decimals into `Decimal128(38, 9)` before it
        // converts them into timestamps, on the whole batch: DataFusion fails
        // to cast `10^60` on the batch of 4096 rows, but a pipeline would fail
        // to convert `10^12` seconds into nanoseconds on its first chunk.
        let rows = 4096;
        use arrow::array::ArrowNativeTypeOp;
        let ten = arrow::datatypes::i256::from_i128(10);
        let values = arrow::array::Decimal256Array::from_iter_values(
            (0..rows).map(|row| ten.pow_wrapping(if row < 2048 { 12 } else { 60 })),
        )
        .with_precision_and_scale(76, 0)
        .unwrap();
        let batch =
            RecordBatch::try_from_iter([("d", Arc::new(values) as ArrayRef)]).unwrap();
        let (plan, supported, native) = project_batch("to_timestamp(d)", &batch);
        assert!(supported);
        let task_ctx = Arc::new(TaskContext::default());
        let outcome = assert_allowed_native_outcome(
            plan,
            task_ctx,
            &["Cast error", "Execution error", "Parser error"],
        )
        .await;
        assert!(
            outcome
                .as_ref()
                .is_err_and(|error| error.contains("Cast error")),
            "{outcome:?}"
        );
        let error = project(&native, &DataChunk::from(batch.slice(0, 2048))).unwrap_err();
        assert!(
            matches!(error.find_root(), DataFusionError::Execution(_)),
            "{error}"
        );
    }

    /// The kind of `error`: the variant of its root, and of the error of Arrow
    /// that it wraps.
    fn error_kind(error: &DataFusionError) -> String {
        let debug = match error.find_root() {
            DataFusionError::ArrowError(error, _) => format!("ArrowError::{error:?}"),
            error => format!("{error:?}"),
        };
        debug.split('(').next().unwrap_or_default().to_owned()
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn parts_that_fail_with_several_kinds_of_errors_are_native() {
        // DataFusion evaluates the rows of a partition in order, and fails on
        // row 2047, but a pipeline evaluates the chunks of the partition on
        // several threads, and raises the error of the chunk that fails first,
        // which may be the second one, with another kind of error:
        // `to_timestamp` without formats fails on a date beyond the range of
        // nanoseconds with an execution error, but to parse a string with a
        // parse error; a division of integers fails on the smallest integer
        // divided by -1 with an overflow, but on a divisor of 0 with a division
        // by zero; and so does a division of decimals by a divisor that may be
        // 0, e.g. the `sum(...) / sum(volume)` of TPC-H Q8 (of decimals of
        // scale 4): on a divisor of 0 with a division by zero, but on a
        // quotient beyond the values of its type with an overflow.
        let rows = 4096;
        let strings = StringArray::from_iter_values((0..rows).map(|row| match row {
            0..2047 => "2020-01-01",
            2047 => "3000-01-01",
            _ => "not timestamp",
        }));
        let dividends = Int32Array::from_iter_values(
            (0..rows).map(|row| if row == 2047 { i32::MIN } else { 1 }),
        );
        let divisors = Int32Array::from_iter_values((0..rows).map(|row| match row {
            0..2047 => 1,
            2047 => -1,
            _ => 0,
        }));
        // The sums of TPC-H Q8, `DECIMAL(38, 4)`.
        let decimals = |values: Vec<i128>| -> ArrayRef {
            Arc::new(
                arrow::array::Decimal128Array::from(values)
                    .with_precision_and_scale(38, 4)
                    .unwrap(),
            )
        };
        for (expr, batch, kinds) in [
            (
                "to_timestamp(s)",
                RecordBatch::try_from_iter([("s", Arc::new(strings) as ArrayRef)])
                    .unwrap(),
                ["Execution", "ArrowError::ParseError"],
            ),
            (
                "x / y",
                RecordBatch::try_from_iter([
                    ("x", Arc::new(dividends) as ArrayRef),
                    ("y", Arc::new(divisors) as ArrayRef),
                ])
                .unwrap(),
                ["ArrowError::ArithmeticOverflow", "ArrowError::DivideByZero"],
            ),
            (
                "x / y",
                RecordBatch::try_from_iter([
                    (
                        "x",
                        decimals(
                            (0..rows)
                                .map(|row| if row < 2048 { 1 } else { 10_i128.pow(37) })
                                .collect(),
                        ),
                    ),
                    (
                        "y",
                        decimals((0..rows).map(|row| i128::from(row != 2047)).collect()),
                    ),
                ])
                .unwrap(),
                ["ArrowError::DivideByZero", "ArrowError::ArithmeticOverflow"],
            ),
        ] {
            let (plan, supported, native) = project_batch(expr, &batch);
            assert!(supported, "{expr}");
            for threads in 2..=4 {
                for _ in 0..8 {
                    let config = SessionConfig::new().with_target_partitions(threads);
                    let NativeOutcome {
                        datafusion: error,
                        operators,
                        ..
                    } = assert_allowed_native_plan_outcome_with_config(
                        || Arc::clone(&plan),
                        false,
                        config,
                        &[
                            "Parser error",
                            "Execution error",
                            "Arithmetic overflow",
                            "Divide by zero",
                            "Cast error",
                            "Invalid argument",
                        ],
                    )
                    .await;
                    assert_eq!(operators.projections, 1, "{expr}");
                    assert!(error.is_some(), "{expr}");
                }
            }
            // The chunks fail with different kinds of errors.
            let outcome = |offset| {
                let chunk = DataChunk::from(batch.slice(offset, 2048));
                error_kind(&project(&native, &chunk).unwrap_err())
            };
            assert_eq!([outcome(0), outcome(2048)], kinds, "{expr}");
        }
        // With formats, `to_timestamp` fails with execution errors only, as
        // the remainders of integers fail with divisions by zero.
        let batch = RecordBatch::try_from_iter([
            (
                "s",
                Arc::new(StringArray::from(vec!["2020-01-01"])) as ArrayRef,
            ),
            ("x", Arc::new(Int32Array::from(vec![1]))),
            ("y", Arc::new(Int32Array::from(vec![1]))),
        ])
        .unwrap();
        for expr in ["to_timestamp(s, '%Y-%m-%d')", "x % y", "10 / y"] {
            let (_, supported, _) = project_batch(expr, &batch);
            assert!(supported, "{expr}");
        }
    }

    /// A user-defined function named `abs`, implemented by a type named like
    /// the built-in one, which returns the number of rows of the batch that it
    /// is evaluated on: its results depend on the batches.
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

        fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
            Ok(DataType::Int64)
        }

        fn invoke_with_args(
            &self,
            args: datafusion_expr::ScalarFunctionArgs,
        ) -> Result<datafusion_expr::ColumnarValue> {
            let rows = args.number_rows;
            Ok(datafusion_expr::ColumnarValue::Array(Arc::new(
                Int64Array::from(vec![rows as i64; rows]),
            )))
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn functions_that_shadow_builtin_ones_are_native() {
        // `abs` is a function of a type named `AbsFunc`, like the built-in one,
        // but another type, which returns the number of rows it is called
        // with: DataFusion evaluates it on a batch of 4096 rows, which it
        // returns, and the native projection on chunks of 2048 rows. Its
        // result depends on the batches, which is the engine's own: the
        // projection is native.
        let setup = async |ctx: &SessionContext| {
            ctx.register_udf(datafusion_expr::ScalarUDF::new_from_impl(AbsFunc {
                signature: datafusion_expr::Signature::any(
                    1,
                    datafusion_expr::Volatility::Immutable,
                ),
            }));
            let x = Int64Array::from_iter_values(0..4096);
            let batch = RecordBatch::try_from_iter([("x", Arc::new(x) as ArrayRef)])?;
            register_partitions(ctx, "shadowed", vec![vec![batch]])
        };
        let sql = "SELECT x, abs(x) AS n FROM shadowed WHERE x >= 0";
        let run = run_native(sql, crate::test_utils::test_session_config(), &setup)
            .await
            .unwrap();
        assert_eq!(run.operators.projections, 1);
        let rows = |batches: &[RecordBatch]| {
            batches
                .iter()
                .flat_map(|batch| {
                    batch
                        .column(1)
                        .as_primitive::<Int64Type>()
                        .values()
                        .to_vec()
                })
                .collect::<Vec<_>>()
        };
        assert_eq!(rows(&run.expected.batches), vec![4096; 4096]);
        let native = rows(&run.actual.batches);
        assert_eq!(native.len(), 4096);
        assert!(native.iter().all(|&rows| rows <= 2048), "{native:?}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn constants_of_batches_of_unknown_sizes_are_native() {
        let _large = one_large_test_at_a_time().await;
        // A constant of 1 KiB, copied for each of the 2^21 rows of a batch
        // without columns (which takes no memory), overflows the 32-bit
        // offsets of strings (2 GiB) in `ProjectionExec`, but not on the
        // chunks of 2048 rows of a pipeline.
        let schema = Arc::new(Schema::empty());
        let batch = |rows: usize| {
            RecordBatch::try_new_with_options(
                Arc::clone(&schema),
                vec![],
                &RecordBatchOptions::new().with_row_count(Some(rows)),
            )
            .unwrap()
        };
        let project =
            |value: &str, input: Arc<dyn ExecutionPlan>| -> Arc<dyn ExecutionPlan> {
                let literal = Arc::new(Literal::new(
                    datafusion_common::ScalarValue::Utf8(Some(value.to_owned())),
                ));
                Arc::new(
                    ProjectionExec::try_new(
                        [ProjectionExpr::new(literal as Arc<dyn PhysicalExpr>, "s")],
                        input,
                    )
                    .unwrap(),
                )
            };
        let supported = |plan: &Arc<dyn ExecutionPlan>| {
            PhysicalProjection::supports_projection_exec(
                plan.downcast_ref::<ProjectionExec>().unwrap(),
            )
        };
        let large = "x".repeat(MAX_CONSTANT_BYTES);
        let plan = project(&large, scan_batches(&schema, &[batch(1 << 21)]));
        assert!(supported(&plan));
        // DataFusion splits the batches of its scans into batches of the
        // batch size of the session, which the plan does not tell.
        let config = SessionConfig::new().with_batch_size(1 << 21);
        let task_ctx = Arc::new(TaskContext::default().with_session_config(config));
        let outcome = assert_allowed_native_outcome(plan, task_ctx, &["overflow"]).await;
        assert!(
            outcome
                .as_ref()
                .is_err_and(|error| error.to_lowercase().contains("overflow")),
            "{outcome:?}"
        );
        // A batch of a row less holds the constant for each of its rows.
        assert!(supported(&project(
            &large,
            scan_batches(&schema, &[batch((1 << 21) - 1)])
        )));
        assert!(supported(&project(
            &format!("{large}x"),
            scan_batches(&schema, &[batch(1)])
        )));
        // Batches of any size, e.g. of a repartition (into batches of the
        // batch size of the session), do not hold a copy of a constant of
        // any size for each of their rows, but of an empty one.
        let repartition = |input| -> Arc<dyn ExecutionPlan> {
            Arc::new(
                RepartitionExec::try_new(input, Partitioning::RoundRobinBatch(2))
                    .unwrap(),
            )
        };
        let scan = scan_batches(&schema, &[batch(4096)]);
        assert!(supported(&project("1-URGENT", Arc::clone(&scan))));
        assert!(supported(&project(
            "1-URGENT",
            repartition(Arc::clone(&scan))
        )));
        assert!(supported(&project("", repartition(scan))));
    }
}
