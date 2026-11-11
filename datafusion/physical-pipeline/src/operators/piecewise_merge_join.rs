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

#[cfg(test)]
mod tests {
    use super::*;

    use datafusion::prelude::SessionContext;
    use datafusion_common::tree_node::TreeNodeRecursion;

    use crate::join::nested_loop::PhysicalNestedLoopJoin;
    use crate::operators::lowering_tests::{assert_lowered, count};
    use crate::test_utils::{register_test_tables, test_session_config};

    async fn setup(ctx: &SessionContext) -> Result<()> {
        register_test_tables(ctx)?;
        let rows = ctx.sql("SELECT id, a, b FROM t1 WHERE id < 400").await?;
        ctx.register_table("s", rows.cache().await?.into_view())?;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn piecewise_merge_joins_are_native() {
        let mut config = test_session_config();
        config.options_mut().optimizer.enable_piecewise_merge_join = true;
        // `s`: the first 400 rows of `t1`.
        for sql in [
            "SELECT s.id, t2.k, t2.w FROM s JOIN t2 ON s.a < t2.k",
            "SELECT s.id, t2.k FROM s JOIN t2 ON s.b + 1 >= t2.w * 2",
            "SELECT t2.k, s.id FROM t2 LEFT JOIN s ON t2.w > s.b",
            "SELECT t2.k, s.id FROM t2 RIGHT JOIN s ON t2.w <= s.b",
            "SELECT t2.k, s.id FROM t2 FULL JOIN s ON t2.k > s.a",
            "SELECT k, name FROM t2 WHERE EXISTS (SELECT 1 FROM s WHERE s.a > t2.k * 10)",
            "SELECT k, name FROM t2 WHERE NOT EXISTS (SELECT 1 FROM s WHERE s.a > t2.k * 10)",
            "SELECT id FROM s WHERE EXISTS (SELECT 1 FROM t2 WHERE t2.w < s.b)",
            "SELECT id FROM s WHERE NOT EXISTS (SELECT 1 FROM t2 WHERE t2.w < s.b)",
        ] {
            for target_partitions in [1, 4] {
                let config = config.clone().with_target_partitions(target_partitions);
                let datafusion_plan = {
                    let ctx = crate::test_utils::datafusion_context(config.clone());
                    setup(&ctx).await.unwrap();
                    ctx.sql(sql)
                        .await
                        .unwrap()
                        .create_physical_plan()
                        .await
                        .unwrap()
                };
                let merge_joins = {
                    let mut joins = 0;
                    TreeNode::apply(&datafusion_plan, |node| {
                        joins += usize::from(node.is::<PiecewiseMergeJoinExec>());
                        Ok(TreeNodeRecursion::Continue)
                    })
                    .unwrap();
                    joins
                };
                let root = assert_lowered(sql, config, setup).await;
                assert_eq!(count::<PhysicalNestedLoopJoin>(&root), merge_joins, "{sql}");
                assert!(merge_joins > 0, "{sql}");
            }
        }
    }
}
