// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! The joins by a single range predicate (`PiecewiseMergeJoinExec`), which
//! the native nested loop join executes.

use std::collections::BTreeSet;
use std::sync::Arc;

use arrow::datatypes::Schema;
use datafusion_common::tree_node::{Transformed, TreeNode};
use datafusion_common::{JoinSide, Result};
use datafusion_physical_expr::expressions::{BinaryExpr, Column};
use datafusion_physical_expr::utils::collect_columns;
use datafusion_physical_plan::joins::utils::{ColumnIndex, JoinFilter};
use datafusion_physical_plan::joins::{NestedLoopJoinExec, PiecewiseMergeJoinExec};
use datafusion_physical_plan::sorts::sort::SortExec;
use datafusion_physical_plan::{ExecutionPlan, PhysicalExpr};

/// The `NestedLoopJoinExec` of the join of `plan`, if it is a
/// `PiecewiseMergeJoinExec`: the join of its buffered input (on the left)
/// and its streamed input (on the right), of the same type, with the filter
/// `on.0 <operator> on.1` (e.g. `l.a < r.b`). It joins the same rows, with the
/// same output schema, which the native nested loop join produces (see
/// `PhysicalNestedLoopJoin::try_lower`).
///
/// The `SortExec` that DataFusion plans below the buffered input of the
/// merge join, which a nested loop join does not need, is removed (but a
/// sort with a fetch, which keeps some rows only).
pub(crate) fn nested_loop_join_of(
    plan: &Arc<dyn ExecutionPlan>,
) -> Result<Option<Arc<dyn ExecutionPlan>>> {
    let Some(join) = plan.downcast_ref::<PiecewiseMergeJoinExec>() else {
        return Ok(None);
    };
    let left = match join.buffered().downcast_ref::<SortExec>() {
        Some(sort) if sort.fetch().is_none() => Arc::clone(sort.input()),
        _ => Arc::clone(join.buffered()),
    };
    let right = Arc::clone(join.streamed());
    let (left_expr, right_expr) = &join.on;
    let filter = range_filter(
        &left.schema(),
        &right.schema(),
        left_expr,
        join.operator,
        right_expr,
    )?;
    let nested_loop =
        NestedLoopJoinExec::try_new(left, right, Some(filter), &join.join_type(), None)?;
    Ok(Some(Arc::new(nested_loop)))
}

/// The join filter `left_expr <operator> right_expr`, of an expression of the
/// columns of the left input `left` and an expression of the columns of the
/// right input `right`: over the columns that they read, of both inputs.
fn range_filter(
    left: &Schema,
    right: &Schema,
    left_expr: &Arc<dyn PhysicalExpr>,
    operator: datafusion_expr::Operator,
    right_expr: &Arc<dyn PhysicalExpr>,
) -> Result<JoinFilter> {
    let columns = |expr: &Arc<dyn PhysicalExpr>| {
        collect_columns(expr)
            .iter()
            .map(Column::index)
            .collect::<BTreeSet<_>>()
            .into_iter()
            .collect::<Vec<_>>()
    };
    let left_columns = columns(left_expr);
    let right_columns = columns(right_expr);
    let mut fields = vec![];
    let mut column_indices = vec![];
    for (schema, columns, side) in [
        (left, &left_columns, JoinSide::Left),
        (right, &right_columns, JoinSide::Right),
    ] {
        for &index in columns {
            fields.push(schema.field(index).clone());
            column_indices.push(ColumnIndex { index, side });
        }
    }
    // The columns of the expressions, in the intermediate batch of the
    // filter: those of the left input first.
    let rewrite = |expr: &Arc<dyn PhysicalExpr>, columns: &[usize], offset: usize| {
        Arc::clone(expr)
            .transform(|node| match node.downcast_ref::<Column>() {
                Some(column) => {
                    let position = columns
                        .iter()
                        .position(|&index| index == column.index())
                        .unwrap_or_default();
                    let column: Arc<dyn PhysicalExpr> =
                        Arc::new(Column::new(column.name(), offset + position));
                    Ok(Transformed::yes(column))
                }
                None => Ok(Transformed::no(node)),
            })
            .map(|transformed| transformed.data)
    };
    let expression = Arc::new(BinaryExpr::new(
        rewrite(left_expr, &left_columns, 0)?,
        operator,
        rewrite(right_expr, &right_columns, left_columns.len())?,
    ));
    Ok(JoinFilter::new(
        expression,
        column_indices,
        Arc::new(Schema::new(fields)),
    ))
}
