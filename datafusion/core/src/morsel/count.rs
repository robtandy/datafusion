// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::sync::Arc;

use arrow::array::{Int64Array, RecordBatch};
use arrow::datatypes::SchemaRef;
use datafusion_common::{Result, exec_datafusion_err};
use datafusion_expr::Expr;
use datafusion_expr::expr::AggregateFunctionParams;
use datafusion_expr::logical_plan::Aggregate;
use datafusion_expr::utils::COUNT_STAR_EXPANSION;
use datafusion_functions_aggregate::count::Count;

pub(super) fn count_star(aggregate: &Aggregate) -> Option<&Expr> {
    let [expr] = aggregate.aggr_expr.as_slice() else {
        return None;
    };
    let mut unaliased = expr;
    while let Expr::Alias(alias) = unaliased {
        unaliased = &alias.expr;
    }
    let Expr::AggregateFunction(function) = unaliased else {
        return None;
    };
    let AggregateFunctionParams {
        args,
        distinct,
        filter,
        order_by,
        null_treatment,
    } = &function.params;
    (aggregate.group_expr.is_empty()
        && function.func.inner().is::<Count>()
        && matches!(args.as_slice(), [Expr::Literal(value, _)] if value == &COUNT_STAR_EXPANSION)
        && !distinct
        && filter.is_none()
        && order_by.is_empty()
        && null_treatment.is_none())
    .then_some(expr)
}

#[derive(Clone, Copy, Debug, Default)]
pub(super) struct CountState(i64);

impl CountState {
    pub(super) fn add_rows(&mut self, rows: usize) -> Result<()> {
        let rows = i64::try_from(rows).map_err(|_| count_overflow())?;
        self.combine(Self(rows))
    }

    pub(super) fn combine(&mut self, local: Self) -> Result<()> {
        self.0 = self.0.checked_add(local.0).ok_or_else(count_overflow)?;
        Ok(())
    }
}

fn count_overflow() -> datafusion_common::DataFusionError {
    exec_datafusion_err!("Morsel COUNT(*) exceeds Int64")
}

#[derive(Debug)]
pub(super) struct CountSink {
    schema: SchemaRef,
    #[cfg(test)]
    pub(super) finalize_gate: Option<super::tests::FinalizeGate>,
}

impl CountSink {
    pub(super) fn new(schema: SchemaRef) -> Self {
        Self {
            schema,
            #[cfg(test)]
            finalize_gate: None,
        }
    }

    pub(super) fn finalize(&self, combined: CountState) -> Result<FinalizedCount> {
        Ok(FinalizedCount(RecordBatch::try_new(
            Arc::clone(&self.schema),
            vec![Arc::new(Int64Array::from(vec![combined.0]))],
        )?))
    }
}

pub(super) struct FinalizedCount(RecordBatch);

impl FinalizedCount {
    pub(super) fn into_batch(self) -> RecordBatch {
        self.0
    }
}
