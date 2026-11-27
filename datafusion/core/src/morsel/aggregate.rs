// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::fmt;
use std::hash::Hash;
use std::sync::Arc;

use arrow::array::types::{
    Decimal128Type, Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type,
    UInt32Type, UInt64Type,
};
use arrow::array::{
    Array, ArrayRef, ArrowPrimitiveType, Decimal128Array, Float64Array, Int32Array,
    Int64Array, NullBufferBuilder, PrimitiveArray, RecordBatch, StringViewArray,
    UInt64Array,
};
use arrow::datatypes::{
    DECIMAL128_MAX_PRECISION, DECIMAL128_MAX_SCALE, DataType, DecimalType, Field, Schema,
    SchemaRef,
};
use arrow::error::ArrowError;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::{
    DataFusionError, HashMap, Result, ScalarValue, exec_err, internal_err,
};
use datafusion_execution::memory_pool::{MemoryPool, MemoryReservation};
use datafusion_expr::expr::{AggregateFunctionParams, physical_name};
use datafusion_expr::logical_plan::Aggregate;
use datafusion_expr::physical_planning_context::PhysicalPlanningContext;
use datafusion_expr::{
    Accumulator, EmitTo, Expr, GroupsAccumulator, Operator as ExprOperator,
};
use datafusion_functions_aggregate::average::Avg;
use datafusion_functions_aggregate::count::Count;
use datafusion_functions_aggregate::min_max::{Max, Min};
use datafusion_functions_aggregate::sum::Sum;
use datafusion_physical_expr::aggregate::{
    AggregateFunctionExpr, LoweredAggregateBuilder,
};
use datafusion_physical_expr::expressions::{BinaryExpr, CastExpr, Column, Literal};
use datafusion_physical_expr::{
    GroupsAccumulatorAdapter, PhysicalExpr, create_physical_expr,
};
use datafusion_physical_plan::aggregates::PhysicalGroupBy;
use datafusion_physical_plan::aggregates::group_values::{GroupValues, new_group_values};
use datafusion_physical_plan::aggregates::order::GroupClustering;
use datafusion_physical_plan::execution_plan::apply_expression_roots;
use datafusion_session::Session;
use tokio::sync::mpsc;

use super::batch::{
    BatchView, Selection, collect_expr_column_indices, remap_expr_columns,
};
use super::count::count_overflow;
use super::group_count::{IntKeyHash, IntKeyType, hash_int_key_array};
use super::metrics::{self, RunMetrics};
use super::radix_agg::{
    NULL_GROUP_HASH, PREAGG_FLUSH_BYTES, PREAGG_FLUSH_GROUPS, RADIX_TRANSITION_GROUPS,
    RadixFinalizeConfig, RadixGroupMerger, RadixPartitionStore, ResultOpsRef,
    emit_finalized_batch, finalize_and_emit_radix, has_float_group_type,
    hash_group_columns,
};
use super::spill::SpillContext;
use super::{Operator, apply_operators, supported_expr};

const MERGE_CHUNK_SIZE: usize = 4096;
const MAX_KEPT_LITERAL_ARRAY_BYTES: usize = 1024 * 1024;

pub(super) fn validate_aggregate(
    aggregate: &Aggregate,
) -> Result<std::result::Result<(), String>> {
    if aggregate.group_expr.is_empty() && aggregate.aggr_expr.is_empty() {
        return Ok(Err(
            "empty aggregate is not supported in native morsel aggregates".into(),
        ));
    }
    for expr in &aggregate.group_expr {
        if matches!(expr, Expr::GroupingSet(_)) {
            return Ok(Err(
                "GROUPING SETS, ROLLUP, and CUBE are not supported in native morsel aggregates"
                    .into(),
            ));
        }
        if !supported_expr(expr)? {
            return Ok(Err(format!("unsupported GROUP BY expression: {expr}")));
        }
    }
    for expr in &aggregate.aggr_expr {
        let mut unaliased = expr;
        while let Expr::Alias(alias) = unaliased {
            unaliased = &alias.expr;
        }
        let Expr::AggregateFunction(function) = unaliased else {
            return Ok(Err(format!("unsupported aggregate expression: {expr}")));
        };
        let AggregateFunctionParams {
            args,
            distinct,
            filter,
            order_by,
            null_treatment,
        } = &function.params;
        let inner = function.func.inner();
        let supported_func = inner.is::<Count>()
            || inner.is::<Sum>()
            || inner.is::<Min>()
            || inner.is::<Max>()
            || inner.is::<Avg>();
        if !supported_func {
            return Ok(Err(format!(
                "unsupported aggregate function '{}' in native morsel aggregates",
                function.func.name()
            )));
        }
        if *distinct && !(inner.is::<Count>() && args.len() == 1) {
            return Ok(Err(format!(
                "DISTINCT aggregates are only supported for single-argument COUNT(DISTINCT expr) in native morsel aggregates: {expr}"
            )));
        }
        if filter.is_some() {
            return Ok(Err(format!(
                "FILTER clauses are not supported in native morsel aggregates: {expr}"
            )));
        }
        if !order_by.is_empty() {
            return Ok(Err(format!(
                "ORDER BY in aggregate functions is not supported in native morsel aggregates: {expr}"
            )));
        }
        if null_treatment.is_some() {
            return Ok(Err(format!(
                "null treatment in aggregate functions is not supported in native morsel aggregates: {expr}"
            )));
        }
        for arg in args {
            if !supported_expr(arg)? {
                return Ok(Err(format!(
                    "unsupported aggregate argument expression: {arg}"
                )));
            }
        }
    }
    Ok(Ok(()))
}

struct LiteralCache {
    value: ScalarValue,
    cached: Option<ArrayRef>,
}

impl LiteralCache {
    fn new(value: ScalarValue) -> Self {
        Self {
            value,
            cached: None,
        }
    }

    fn array_of_size(&mut self, size: usize) -> Result<ArrayRef> {
        if let Some(cached) = &self.cached
            && cached.len() >= size
        {
            return Ok(cached.slice(0, size));
        }
        let array = self.value.to_array_of_size(size)?;
        if array.get_array_memory_size() <= MAX_KEPT_LITERAL_ARRAY_BYTES {
            self.cached = Some(Arc::clone(&array));
        }
        Ok(array)
    }
}

struct ArgEvaluator {
    expr: Arc<dyn PhysicalExpr>,
    literal: Option<LiteralCache>,
}

impl ArgEvaluator {
    fn new(expr: Arc<dyn PhysicalExpr>) -> Self {
        let literal = expr
            .downcast_ref::<Literal>()
            .map(|lit| LiteralCache::new(lit.value().clone()));
        Self { expr, literal }
    }

    fn evaluate(&mut self, batch: &RecordBatch) -> Result<ArrayRef> {
        match &mut self.literal {
            Some(literal) => literal.array_of_size(batch.num_rows()),
            None => self
                .expr
                .evaluate(batch)?
                .into_array_of_size(batch.num_rows()),
        }
    }
}

fn create_groups_accumulator(
    aggr_expr: &Arc<AggregateFunctionExpr>,
    validate_adapter: bool,
) -> Result<Box<dyn GroupsAccumulator>> {
    if aggr_expr.groups_accumulator_supported() {
        aggr_expr.create_groups_accumulator()
    } else {
        if validate_adapter {
            let _ = aggr_expr.create_accumulator()?;
        }
        let expr = Arc::clone(aggr_expr);
        Ok(Box::new(GroupsAccumulatorAdapter::new(move || {
            expr.create_accumulator()
        })))
    }
}

fn arrays_memory_size(arrays: &[ArrayRef]) -> usize {
    arrays
        .iter()
        .map(|array| array.get_array_memory_size())
        .fold(0usize, usize::saturating_add)
}

/// Exact `Decimal128` averager matching `DecimalAverager::<Decimal128Type>` in
/// `datafusion-functions-aggregate-common`.
#[derive(Clone, Copy, Debug)]
struct Decimal128Averager {
    scale_mul: i128,
    target_precision: u8,
    target_scale: i8,
}

impl Decimal128Averager {
    fn try_new(sum_scale: i8, target_precision: u8, target_scale: i8) -> Result<Self> {
        if sum_scale < 0
            || target_scale < sum_scale
            || target_scale > DECIMAL128_MAX_SCALE
            || target_precision > DECIMAL128_MAX_PRECISION
        {
            return exec_err!("Arithmetic Overflow in AvgAccumulator");
        }
        let sum_mul = 10_i128.wrapping_pow(sum_scale as u32);
        let target_mul = 10_i128.wrapping_pow(target_scale as u32);
        if target_mul < sum_mul {
            return exec_err!("Arithmetic Overflow in AvgAccumulator");
        }
        let scale_mul = target_mul.wrapping_div(sum_mul);
        Ok(Self {
            scale_mul,
            target_precision,
            target_scale,
        })
    }

    #[inline(always)]
    fn avg(&self, sum: i128, count: i64) -> Result<i128> {
        if let Some(scaled) = sum.checked_mul(self.scale_mul) {
            let new_value = scaled.wrapping_div(count as i128);
            if Decimal128Type::validate_decimal_precision(
                new_value,
                self.target_precision,
                self.target_scale,
            )
            .is_ok()
            {
                return Ok(new_value);
            }
        }
        exec_err!("Arithmetic Overflow in AvgAccumulator")
    }
}

#[inline(always)]
fn checked_mul_i128_fast(a: i128, b: i128) -> Result<i128> {
    let a64 = a as i64;
    let b64 = b as i64;
    // Two `i64` values have product bounded by `2^126 < i128::MAX`, so the
    // fast path avoids compiler-rt `__muloti4` while keeping exact overflow checks.
    if (a64 as i128 == a) && (b64 as i128 == b) {
        Ok((a64 as i128) * (b64 as i128))
    } else {
        checked_mul_i128_slow(a, b)
    }
}

#[cold]
#[inline(never)]
fn checked_mul_i128_slow(a: i128, b: i128) -> Result<i128> {
    a.checked_mul(b).ok_or_else(|| {
        DataFusionError::ArrowError(
            Box::new(ArrowError::ArithmeticOverflow(format!(
                "Overflow happened on: {a} * {b}"
            ))),
            None,
        )
    })
}

#[inline(always)]
fn checked_add_i128(a: i128, b: i128) -> Result<i128> {
    a.checked_add(b).ok_or_else(|| {
        DataFusionError::ArrowError(
            Box::new(ArrowError::ArithmeticOverflow(format!(
                "Overflow happened on: {a} + {b}"
            ))),
            None,
        )
    })
}

#[inline(always)]
fn checked_sub_i128(a: i128, b: i128) -> Result<i128> {
    a.checked_sub(b).ok_or_else(|| {
        DataFusionError::ArrowError(
            Box::new(ArrowError::ArithmeticOverflow(format!(
                "Overflow happened on: {a} - {b}"
            ))),
            None,
        )
    })
}

/// Fused `Decimal128` input expression evaluated directly without intermediate
/// array allocations while preserving exact `arrow-arith` overflow and NULL semantics.
#[derive(Clone, Copy, Debug)]
enum Decimal128Input {
    Col(usize),
    MulColCol { a: usize, b: usize },
    MulColAddLit { a: usize, b: usize, lit: i128 },
    MulColSubFromLit { a: usize, b: usize, lit: i128 },
    MulColSubLit { a: usize, b: usize, lit: i128 },
}

impl Decimal128Input {
    fn remap(&self, needed_columns: &[usize]) -> Option<Self> {
        let map = |idx: usize| needed_columns.binary_search(&idx).ok();
        match *self {
            Self::Col(c) => Some(Self::Col(map(c)?)),
            Self::MulColCol { a, b } => Some(Self::MulColCol {
                a: map(a)?,
                b: map(b)?,
            }),
            Self::MulColAddLit { a, b, lit } => Some(Self::MulColAddLit {
                a: map(a)?,
                b: map(b)?,
                lit,
            }),
            Self::MulColSubFromLit { a, b, lit } => Some(Self::MulColSubFromLit {
                a: map(a)?,
                b: map(b)?,
                lit,
            }),
            Self::MulColSubLit { a, b, lit } => Some(Self::MulColSubLit {
                a: map(a)?,
                b: map(b)?,
                lit,
            }),
        }
    }

    fn eval_to_array(&self, batch: &RecordBatch, out_dt: &DataType) -> Result<ArrayRef> {
        let (p, s) = match out_dt {
            DataType::Decimal128(p, s) => (*p, *s),
            _ => {
                return internal_err!(
                    "Decimal128Input expected Decimal128 output type, got {out_dt}"
                );
            }
        };
        let n = batch.num_rows();
        match *self {
            Self::Col(c) => {
                let arr = batch
                    .column(c)
                    .as_any()
                    .downcast_ref::<Decimal128Array>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected Decimal128 column at index {c}"
                        )
                    })?;
                Ok(Arc::new(arr.clone().with_precision_and_scale(p, s)?))
            }
            Self::MulColCol { a, b }
            | Self::MulColAddLit { a, b, .. }
            | Self::MulColSubFromLit { a, b, .. }
            | Self::MulColSubLit { a, b, .. } => {
                let arr_a = batch
                    .column(a)
                    .as_any()
                    .downcast_ref::<Decimal128Array>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected Decimal128 column at index {a}"
                        )
                    })?;
                let arr_b = batch
                    .column(b)
                    .as_any()
                    .downcast_ref::<Decimal128Array>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected Decimal128 column at index {b}"
                        )
                    })?;
                let va = arr_a.values();
                let vb = arr_b.values();
                let mut out = Vec::with_capacity(n);
                let nulls = if arr_a.null_count() == 0 && arr_b.null_count() == 0 {
                    match *self {
                        Self::MulColCol { .. } => {
                            for (&a_v, &b_v) in va.iter().zip(vb.iter()) {
                                out.push(checked_mul_i128_fast(a_v, b_v)?);
                            }
                        }
                        Self::MulColAddLit { lit, .. } => {
                            for (&a_v, &b_v) in va.iter().zip(vb.iter()) {
                                let rhs = checked_add_i128(lit, b_v)?;
                                out.push(checked_mul_i128_fast(a_v, rhs)?);
                            }
                        }
                        Self::MulColSubFromLit { lit, .. } => {
                            for (&a_v, &b_v) in va.iter().zip(vb.iter()) {
                                let rhs = checked_sub_i128(lit, b_v)?;
                                out.push(checked_mul_i128_fast(a_v, rhs)?);
                            }
                        }
                        Self::MulColSubLit { lit, .. } => {
                            for (&a_v, &b_v) in va.iter().zip(vb.iter()) {
                                let rhs = checked_sub_i128(b_v, lit)?;
                                out.push(checked_mul_i128_fast(a_v, rhs)?);
                            }
                        }
                        Self::Col(_) => unreachable!(),
                    }
                    None
                } else {
                    let na = arr_a.nulls();
                    let nb = arr_b.nulls();
                    let mut nb_builder = NullBufferBuilder::new(n);
                    for i in 0..n {
                        let valid = na.is_none_or(|m| m.is_valid(i))
                            && nb.is_none_or(|m| m.is_valid(i));
                        if valid {
                            let val = match *self {
                                Self::MulColCol { .. } => {
                                    checked_mul_i128_fast(va[i], vb[i])?
                                }
                                Self::MulColAddLit { lit, .. } => {
                                    let rhs = checked_add_i128(lit, vb[i])?;
                                    checked_mul_i128_fast(va[i], rhs)?
                                }
                                Self::MulColSubFromLit { lit, .. } => {
                                    let rhs = checked_sub_i128(lit, vb[i])?;
                                    checked_mul_i128_fast(va[i], rhs)?
                                }
                                Self::MulColSubLit { lit, .. } => {
                                    let rhs = checked_sub_i128(vb[i], lit)?;
                                    checked_mul_i128_fast(va[i], rhs)?
                                }
                                Self::Col(_) => unreachable!(),
                            };
                            out.push(val);
                            nb_builder.append_non_null();
                        } else {
                            out.push(0);
                            nb_builder.append_null();
                        }
                    }
                    nb_builder.finish()
                };
                Ok(Arc::new(
                    Decimal128Array::new(out.into(), nulls)
                        .with_precision_and_scale(p, s)?,
                ))
            }
        }
    }
}

fn unwrap_decimal128_noop_cast<'a>(
    mut expr: &'a Arc<dyn PhysicalExpr>,
    schema: &SchemaRef,
) -> &'a Arc<dyn PhysicalExpr> {
    while let Some(cast) = expr.downcast_ref::<CastExpr>() {
        let Ok(DataType::Decimal128(p_in, s_in)) = cast.expr().data_type(schema) else {
            break;
        };
        if let DataType::Decimal128(p_out, s_out) = *cast.cast_type()
            && s_in == s_out
            && p_out >= p_in
        {
            expr = cast.expr();
        } else {
            break;
        }
    }
    expr
}

fn extract_scaled_decimal128_lit(
    expr: &Arc<dyn PhysicalExpr>,
    target_scale: i8,
) -> Option<i128> {
    let lit = expr.downcast_ref::<Literal>()?;
    let ScalarValue::Decimal128(Some(val), _, lit_scale) = *lit.value() else {
        return None;
    };
    match lit_scale.cmp(&target_scale) {
        std::cmp::Ordering::Equal => Some(val),
        std::cmp::Ordering::Less => {
            let diff = u32::try_from(target_scale as i16 - lit_scale as i16).ok()?;
            let pow = 10_i128.checked_pow(diff)?;
            val.checked_mul(pow)
        }
        std::cmp::Ordering::Greater => None,
    }
}

fn try_extract_decimal128_input(
    expr: &Arc<dyn PhysicalExpr>,
    schema: &SchemaRef,
) -> Option<Decimal128Input> {
    let expr = unwrap_decimal128_noop_cast(expr, schema);
    let DataType::Decimal128(_, out_s) = expr.data_type(schema).ok()? else {
        return None;
    };
    if let Some(col) = expr.downcast_ref::<Column>() {
        return Some(Decimal128Input::Col(col.index()));
    }
    let bin = expr.downcast_ref::<BinaryExpr>()?;
    if *bin.op() != ExprOperator::Multiply {
        return None;
    }
    let left = unwrap_decimal128_noop_cast(bin.left(), schema);
    let right = unwrap_decimal128_noop_cast(bin.right(), schema);
    let (DataType::Decimal128(_, s1), DataType::Decimal128(_, s2)) =
        (left.data_type(schema).ok()?, right.data_type(schema).ok()?)
    else {
        return None;
    };
    if s1 as i16 + s2 as i16 != out_s as i16 || out_s > DECIMAL128_MAX_SCALE {
        return None;
    }
    if let (Some(ca), Some(cb)) = (
        left.downcast_ref::<Column>(),
        right.downcast_ref::<Column>(),
    ) {
        return Some(Decimal128Input::MulColCol {
            a: ca.index(),
            b: cb.index(),
        });
    }
    let match_col_and_linear =
        |col_expr: &Arc<dyn PhysicalExpr>, lin_expr: &Arc<dyn PhysicalExpr>| {
            let ca = col_expr.downcast_ref::<Column>()?;
            let lin = lin_expr.downcast_ref::<BinaryExpr>()?;
            if !matches!(lin.op(), ExprOperator::Plus | ExprOperator::Minus) {
                return None;
            }
            let DataType::Decimal128(_, lin_s) = lin_expr.data_type(schema).ok()? else {
                return None;
            };
            let ll = unwrap_decimal128_noop_cast(lin.left(), schema);
            let lr = unwrap_decimal128_noop_cast(lin.right(), schema);
            let (DataType::Decimal128(_, sll), DataType::Decimal128(_, slr)) =
                (ll.data_type(schema).ok()?, lr.data_type(schema).ok()?)
            else {
                return None;
            };
            if sll.max(slr) != lin_s {
                return None;
            }
            match *lin.op() {
                ExprOperator::Plus => {
                    if let Some(cb) = lr.downcast_ref::<Column>()
                        && slr == lin_s
                        && let Some(lit) = extract_scaled_decimal128_lit(ll, lin_s)
                    {
                        return Some(Decimal128Input::MulColAddLit {
                            a: ca.index(),
                            b: cb.index(),
                            lit,
                        });
                    }
                    if let Some(cb) = ll.downcast_ref::<Column>()
                        && sll == lin_s
                        && let Some(lit) = extract_scaled_decimal128_lit(lr, lin_s)
                    {
                        return Some(Decimal128Input::MulColAddLit {
                            a: ca.index(),
                            b: cb.index(),
                            lit,
                        });
                    }
                    None
                }
                ExprOperator::Minus => {
                    if let Some(cb) = lr.downcast_ref::<Column>()
                        && slr == lin_s
                        && let Some(lit) = extract_scaled_decimal128_lit(ll, lin_s)
                    {
                        return Some(Decimal128Input::MulColSubFromLit {
                            a: ca.index(),
                            b: cb.index(),
                            lit,
                        });
                    }
                    if let Some(cb) = ll.downcast_ref::<Column>()
                        && sll == lin_s
                        && let Some(lit) = extract_scaled_decimal128_lit(lr, lin_s)
                    {
                        return Some(Decimal128Input::MulColSubLit {
                            a: ca.index(),
                            b: cb.index(),
                            lit,
                        });
                    }
                    None
                }
                _ => None,
            }
        };
    match_col_and_linear(left, right).or_else(|| match_col_and_linear(right, left))
}

/// Fused evaluator for hot `Decimal128` projection expressions (`a * b`,
/// `a * (lit ± b)`, and `e1 - e2` such as TPC-H Q1/Q9) avoiding intermediate
/// array allocations and `__muloti4` calls while preserving exact overflow/NULL parity.
pub(super) fn try_eval_fast_decimal128_expr(
    expr: &Arc<dyn PhysicalExpr>,
    batch: &RecordBatch,
    out_dt: &DataType,
) -> Result<Option<ArrayRef>> {
    let DataType::Decimal128(out_p, out_s) = *out_dt else {
        return Ok(None);
    };
    let schema = batch.schema();
    if let Some(input) = try_extract_decimal128_input(expr, &schema)
        && !matches!(input, Decimal128Input::Col(_))
    {
        return Ok(Some(input.eval_to_array(batch, out_dt)?));
    }
    let unwrapped = unwrap_decimal128_noop_cast(expr, &schema);
    let Some(bin) = unwrapped.downcast_ref::<BinaryExpr>() else {
        return Ok(None);
    };
    if *bin.op() != ExprOperator::Minus {
        return Ok(None);
    }
    let left = unwrap_decimal128_noop_cast(bin.left(), &schema);
    let right = unwrap_decimal128_noop_cast(bin.right(), &schema);
    let (Ok(DataType::Decimal128(_, sl)), Ok(DataType::Decimal128(_, sr))) =
        (left.data_type(&schema), right.data_type(&schema))
    else {
        return Ok(None);
    };
    if sl != sr || sl != out_s {
        return Ok(None);
    }
    let (
        Some(Decimal128Input::MulColSubFromLit {
            a: la,
            b: lb,
            lit: l_lit,
        }),
        Some(Decimal128Input::MulColCol { a: ra, b: rb }),
    ) = (
        try_extract_decimal128_input(left, &schema),
        try_extract_decimal128_input(right, &schema),
    )
    else {
        return Ok(None);
    };
    let n = batch.num_rows();
    let arr_la = batch
        .column(la)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .ok_or_else(|| {
            datafusion_common::internal_datafusion_err!("Expected Decimal128 at {la}")
        })?;
    let arr_lb = batch
        .column(lb)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .ok_or_else(|| {
            datafusion_common::internal_datafusion_err!("Expected Decimal128 at {lb}")
        })?;
    let arr_ra = batch
        .column(ra)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .ok_or_else(|| {
            datafusion_common::internal_datafusion_err!("Expected Decimal128 at {ra}")
        })?;
    let arr_rb = batch
        .column(rb)
        .as_any()
        .downcast_ref::<Decimal128Array>()
        .ok_or_else(|| {
            datafusion_common::internal_datafusion_err!("Expected Decimal128 at {rb}")
        })?;
    let (vla, vlb, vra, vrb) = (
        arr_la.values(),
        arr_lb.values(),
        arr_ra.values(),
        arr_rb.values(),
    );
    let mut out = Vec::with_capacity(n);
    let nulls = if arr_la.null_count() == 0
        && arr_lb.null_count() == 0
        && arr_ra.null_count() == 0
        && arr_rb.null_count() == 0
    {
        for i in 0..n {
            let rhs_l = checked_sub_i128(l_lit, vlb[i])?;
            let prod_l = checked_mul_i128_fast(vla[i], rhs_l)?;
            let prod_r = checked_mul_i128_fast(vra[i], vrb[i])?;
            out.push(checked_sub_i128(prod_l, prod_r)?);
        }
        None
    } else {
        let (nla, nlb, nra, nrb) = (
            arr_la.nulls(),
            arr_lb.nulls(),
            arr_ra.nulls(),
            arr_rb.nulls(),
        );
        let mut nb = NullBufferBuilder::new(n);
        for i in 0..n {
            let valid = nla.is_none_or(|m| m.is_valid(i))
                && nlb.is_none_or(|m| m.is_valid(i))
                && nra.is_none_or(|m| m.is_valid(i))
                && nrb.is_none_or(|m| m.is_valid(i));
            if valid {
                let rhs_l = checked_sub_i128(l_lit, vlb[i])?;
                let prod_l = checked_mul_i128_fast(vla[i], rhs_l)?;
                let prod_r = checked_mul_i128_fast(vra[i], vrb[i])?;
                out.push(checked_sub_i128(prod_l, prod_r)?);
                nb.append_non_null();
            } else {
                out.push(0);
                nb.append_null();
            }
        }
        nb.finish()
    };
    Ok(Some(Arc::new(
        Decimal128Array::new(out.into(), nulls).with_precision_and_scale(out_p, out_s)?,
    )))
}

#[derive(Clone, Debug)]
enum SharedAvgKind {
    Float64,
    Int64ToFloat64,
    Int128ToFloat64,
    Decimal128(Decimal128Averager),
}

#[derive(Clone, Debug)]
enum OutputAggPlan {
    Direct {
        phys_idx: usize,
    },
    SumInt64FromWide {
        phys_idx: usize,
    },
    AvgFromSumCount {
        sum_phys_idx: usize,
        count_phys_idx: usize,
        kind: SharedAvgKind,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum SumIntInputKind {
    Int64,
    Int32,
}

#[derive(Clone, Debug)]
enum PhysAggKind {
    CountStar,
    CountCol {
        direct_col: Option<usize>,
        rebound_direct_col: Option<usize>,
        count_star_idx: Option<usize>,
    },
    SumDecimal128 {
        data_type: DataType,
        fast_input: Option<Decimal128Input>,
        rebound_fast_input: Option<Decimal128Input>,
    },
    SumInt64 {
        direct_col: Option<(usize, SumIntInputKind)>,
        rebound_direct_col: Option<(usize, SumIntInputKind)>,
    },
    SumInt64Wide {
        direct_col: Option<usize>,
        rebound_direct_col: Option<usize>,
    },
    SumFloat64 {
        direct_col: Option<usize>,
        rebound_direct_col: Option<usize>,
    },
    Generic(Arc<AggregateFunctionExpr>),
}

#[derive(Clone, Debug)]
struct PhysAggSpec {
    kind: PhysAggKind,
    args: Vec<Arc<dyn PhysicalExpr>>,
    rebound_args: Vec<Arc<dyn PhysicalExpr>>,
    state_types: Vec<DataType>,
}

#[derive(Clone, Debug)]
struct AggregatePlan {
    phys_aggrs: Vec<PhysAggSpec>,
    output_plan: Vec<OutputAggPlan>,
    count_star_idx: Option<usize>,
}

impl AggregatePlan {
    fn empty() -> Self {
        Self {
            phys_aggrs: Vec::new(),
            output_plan: Vec::new(),
            count_star_idx: None,
        }
    }
}

fn is_count_star_aggr(
    aggr: &AggregateFunctionExpr,
    input_schema: &SchemaRef,
) -> Result<bool> {
    if !aggr.fun().inner().is::<Count>()
        || aggr.is_distinct()
        || aggr.ignore_nulls()
        || !aggr.order_bys().is_empty()
    {
        return Ok(false);
    }
    let exprs = aggr.expressions();
    let [arg] = exprs.as_slice() else {
        return Ok(false);
    };
    if arg
        .downcast_ref::<Literal>()
        .is_some_and(|l| !l.value().is_null())
    {
        return Ok(true);
    }
    Ok(!arg.nullable(input_schema)?)
}

fn same_aggr_expr(a: &AggregateFunctionExpr, b: &AggregateFunctionExpr) -> bool {
    a.fun() == b.fun()
        && a.is_distinct() == b.is_distinct()
        && a.ignore_nulls() == b.ignore_nulls()
        && a.is_reversed() == b.is_reversed()
        && a.order_bys() == b.order_bys()
        && a.field().data_type() == b.field().data_type()
        && a.expressions().len() == b.expressions().len()
        && a.expressions()
            .iter()
            .zip(b.expressions().iter())
            .all(|(x, y)| x.eq(y))
}

fn unwrap_widening_int_cast(
    expr: &Arc<dyn PhysicalExpr>,
    input_schema: &SchemaRef,
) -> Option<(Arc<dyn PhysicalExpr>, DataType)> {
    let cast = expr.downcast_ref::<CastExpr>()?;
    let inner_dt = cast.expr().data_type(input_schema).ok()?;
    let target_dt = cast.cast_type();
    match (&inner_dt, target_dt) {
        (
            DataType::Int8 | DataType::Int16 | DataType::Int32,
            DataType::Int64 | DataType::Float64,
        )
        | (DataType::Int64, DataType::Float64) => {
            Some((Arc::clone(cast.expr()), inner_dt))
        }
        _ => None,
    }
}

fn extract_sum_int_direct_col(
    arg: &Arc<dyn PhysicalExpr>,
    input_schema: &SchemaRef,
) -> Option<(usize, SumIntInputKind)> {
    if let Some(col) = arg.downcast_ref::<Column>()
        && input_schema.field(col.index()).data_type() == &DataType::Int64
    {
        return Some((col.index(), SumIntInputKind::Int64));
    }
    if let Some(cast) = arg.downcast_ref::<CastExpr>()
        && cast.cast_type() == &DataType::Int64
        && let Some(col) = cast.expr().downcast_ref::<Column>()
        && input_schema.field(col.index()).data_type() == &DataType::Int32
    {
        return Some((col.index(), SumIntInputKind::Int32));
    }
    None
}

enum AvgShareCandidate {
    Decimal128 {
        arg: Arc<dyn PhysicalExpr>,
        sum_dt: DataType,
        averager: Decimal128Averager,
    },
    Float64 {
        arg: Arc<dyn PhysicalExpr>,
    },
    IntSmallToFloat64 {
        base_expr: Arc<dyn PhysicalExpr>,
    },
    Int64ToFloat64 {
        base_expr: Arc<dyn PhysicalExpr>,
    },
}

fn classify_shareable_avg(
    aggr: &AggregateFunctionExpr,
    input_schema: &SchemaRef,
) -> Result<Option<AvgShareCandidate>> {
    if !aggr.fun().inner().is::<Avg>()
        || aggr.is_distinct()
        || aggr.ignore_nulls()
        || !aggr.order_bys().is_empty()
    {
        return Ok(None);
    }
    let exprs = aggr.expressions();
    let [arg] = exprs.as_slice() else {
        return Ok(None);
    };
    let arg_dt = arg.data_type(input_schema)?;
    let out_field = aggr.field();
    let out_dt = out_field.data_type();
    match (&arg_dt, out_dt) {
        // `Avg` keeps its intermediate sum in `Decimal128(38, s)` iff `p + 13 <= 38`
        // (i.e. `p <= 25`); higher precisions widen to `Decimal256(76, s)`.
        (DataType::Decimal128(p, s), DataType::Decimal128(out_p, out_s)) if *p <= 25 => {
            if let Ok(averager) = Decimal128Averager::try_new(*s, *out_p, *out_s) {
                return Ok(Some(AvgShareCandidate::Decimal128 {
                    arg: Arc::clone(arg),
                    sum_dt: DataType::Decimal128(DECIMAL128_MAX_PRECISION, *s),
                    averager,
                }));
            }
        }
        (DataType::Float64, DataType::Float64) => {
            if let Some((base_expr, inner_dt)) =
                unwrap_widening_int_cast(arg, input_schema)
            {
                return match inner_dt {
                    DataType::Int8 | DataType::Int16 | DataType::Int32 => {
                        Ok(Some(AvgShareCandidate::IntSmallToFloat64 { base_expr }))
                    }
                    DataType::Int64 => {
                        Ok(Some(AvgShareCandidate::Int64ToFloat64 { base_expr }))
                    }
                    _ => Ok(None),
                };
            }
            return Ok(Some(AvgShareCandidate::Float64 {
                arg: Arc::clone(arg),
            }));
        }
        _ => {}
    }
    Ok(None)
}

fn build_aggregate_plan(
    aggr_exprs: &[Arc<AggregateFunctionExpr>],
    input_schema: &SchemaRef,
) -> Result<AggregatePlan> {
    if aggr_exprs.is_empty() {
        return Ok(AggregatePlan::empty());
    }

    let mut phys_aggrs: Vec<PhysAggSpec> = Vec::with_capacity(aggr_exprs.len());
    let mut output_plan: Vec<Option<OutputAggPlan>> = vec![None; aggr_exprs.len()];

    // Pre-scan for `COUNT(*)` and shareable `AVG` candidates so pass ordering is
    // independent of column order in the SELECT list (`AVG` before or after `SUM`).
    let mut has_explicit_count_star = false;
    let mut avg_candidates: Vec<Option<AvgShareCandidate>> =
        Vec::with_capacity(aggr_exprs.len());
    let mut decomposable_avg_count = 0usize;
    for aggr in aggr_exprs {
        if is_count_star_aggr(aggr, input_schema)? {
            has_explicit_count_star = true;
        }
        let cand = classify_shareable_avg(aggr, input_schema)?;
        if matches!(
            cand,
            Some(
                AvgShareCandidate::Decimal128 { .. } | AvgShareCandidate::Float64 { .. }
            )
        ) {
            decomposable_avg_count += 1;
        }
        avg_candidates.push(cand);
    }

    let mut count_star_idx: Option<usize> = None;
    let ensure_count_star = |phys_aggrs: &mut Vec<PhysAggSpec>,
                             count_star_idx: &mut Option<usize>|
     -> usize {
        if let Some(idx) = *count_star_idx {
            return idx;
        }
        let idx = phys_aggrs.len();
        phys_aggrs.push(PhysAggSpec {
            kind: PhysAggKind::CountStar,
            args: Vec::new(),
            rebound_args: Vec::new(),
            state_types: vec![DataType::Int64],
        });
        *count_star_idx = Some(idx);
        idx
    };

    let ensure_count_for_expr = |expr: &Arc<dyn PhysicalExpr>,
                                 phys_aggrs: &mut Vec<PhysAggSpec>,
                                 count_star_idx: &mut Option<usize>|
     -> Result<usize> {
        if !expr.nullable(input_schema)? {
            return Ok(ensure_count_star(phys_aggrs, count_star_idx));
        }
        if let Some(idx) = phys_aggrs.iter().position(|p| {
            matches!(p.kind, PhysAggKind::CountCol { .. })
                && p.args.len() == 1
                && p.args[0].eq(expr)
        }) {
            return Ok(idx);
        }
        let direct_col = expr.downcast_ref::<Column>().map(|c| c.index());
        let idx = phys_aggrs.len();
        phys_aggrs.push(PhysAggSpec {
            kind: PhysAggKind::CountCol {
                direct_col,
                rebound_direct_col: None,
                count_star_idx: None,
            },
            args: vec![Arc::clone(expr)],
            rebound_args: Vec::new(),
            state_types: vec![DataType::Int64],
        });
        Ok(idx)
    };

    // Pass 1: lower all non-shareable-AVG aggregates with exact CSE and direct fast kinds.
    for (out_idx, aggr) in aggr_exprs.iter().enumerate() {
        if avg_candidates[out_idx].is_some() {
            continue;
        }
        if is_count_star_aggr(aggr, input_schema)? {
            let idx = ensure_count_star(&mut phys_aggrs, &mut count_star_idx);
            output_plan[out_idx] = Some(OutputAggPlan::Direct { phys_idx: idx });
            continue;
        }
        let inner = aggr.fun().inner();
        let simple =
            !aggr.is_distinct() && !aggr.ignore_nulls() && aggr.order_bys().is_empty();
        if simple && inner.is::<Count>() && aggr.expressions().len() == 1 {
            let arg = &aggr.expressions()[0];
            let idx = ensure_count_for_expr(arg, &mut phys_aggrs, &mut count_star_idx)?;
            output_plan[out_idx] = Some(OutputAggPlan::Direct { phys_idx: idx });
            continue;
        }
        if simple && inner.is::<Sum>() && aggr.expressions().len() == 1 {
            let arg = &aggr.expressions()[0];
            let out_field = aggr.field();
            let out_dt = out_field.data_type();
            let existing = phys_aggrs.iter().position(|p| match (&p.kind, out_dt) {
                (PhysAggKind::SumDecimal128 { data_type, .. }, dt)
                    if data_type == dt && p.args.len() == 1 && p.args[0].eq(arg) =>
                {
                    true
                }
                (PhysAggKind::SumInt64 { .. }, DataType::Int64)
                | (PhysAggKind::SumFloat64 { .. }, DataType::Float64)
                    if p.args.len() == 1 && p.args[0].eq(arg) =>
                {
                    true
                }
                _ => false,
            });
            if let Some(idx) = existing {
                output_plan[out_idx] = Some(OutputAggPlan::Direct { phys_idx: idx });
                continue;
            }
            match out_dt {
                DataType::Decimal128(_, _) => {
                    let fast_input = try_extract_decimal128_input(arg, input_schema);
                    let idx = phys_aggrs.len();
                    phys_aggrs.push(PhysAggSpec {
                        kind: PhysAggKind::SumDecimal128 {
                            data_type: out_dt.clone(),
                            fast_input,
                            rebound_fast_input: None,
                        },
                        args: vec![Arc::clone(arg)],
                        rebound_args: Vec::new(),
                        state_types: vec![out_dt.clone()],
                    });
                    output_plan[out_idx] = Some(OutputAggPlan::Direct { phys_idx: idx });
                    continue;
                }
                DataType::Int64 => {
                    let direct_col = extract_sum_int_direct_col(arg, input_schema);
                    let idx = phys_aggrs.len();
                    phys_aggrs.push(PhysAggSpec {
                        kind: PhysAggKind::SumInt64 {
                            direct_col,
                            rebound_direct_col: None,
                        },
                        args: vec![Arc::clone(arg)],
                        rebound_args: Vec::new(),
                        state_types: vec![DataType::Int64],
                    });
                    output_plan[out_idx] = Some(OutputAggPlan::Direct { phys_idx: idx });
                    continue;
                }
                DataType::Float64 => {
                    let direct_col = arg
                        .downcast_ref::<Column>()
                        .filter(|c| {
                            input_schema.field(c.index()).data_type()
                                == &DataType::Float64
                        })
                        .map(|c| c.index());
                    let idx = phys_aggrs.len();
                    phys_aggrs.push(PhysAggSpec {
                        kind: PhysAggKind::SumFloat64 {
                            direct_col,
                            rebound_direct_col: None,
                        },
                        args: vec![Arc::clone(arg)],
                        rebound_args: Vec::new(),
                        state_types: vec![DataType::Float64],
                    });
                    output_plan[out_idx] = Some(OutputAggPlan::Direct { phys_idx: idx });
                    continue;
                }
                _ => {}
            }
        }

        // Generic exact CSE across all other aggregate functions (MIN, MAX,
        // COUNT(DISTINCT), multi-arg COUNT, Decimal256 SUM/AVG, UInt64, etc.).
        if let Some(idx) = phys_aggrs.iter().position(|p| match &p.kind {
            PhysAggKind::Generic(existing) => same_aggr_expr(existing, aggr),
            _ => false,
        }) {
            output_plan[out_idx] = Some(OutputAggPlan::Direct { phys_idx: idx });
        } else {
            let state_types = aggr
                .state_fields()?
                .into_iter()
                .map(|f| f.data_type().clone())
                .collect();
            let idx = phys_aggrs.len();
            phys_aggrs.push(PhysAggSpec {
                kind: PhysAggKind::Generic(Arc::clone(aggr)),
                args: aggr.expressions(),
                rebound_args: Vec::new(),
                state_types,
            });
            output_plan[out_idx] = Some(OutputAggPlan::Direct { phys_idx: idx });
        }
    }

    // Pass 2: lower shareable `AVG` aggregates, reusing existing `SUM`/`COUNT`
    // accumulators or decomposing into shared `SUM` + `COUNT` when beneficial.
    for (out_idx, aggr) in aggr_exprs.iter().enumerate() {
        let Some(cand) = &avg_candidates[out_idx] else {
            continue;
        };
        // Exact duplicate of an already-resolved `AVG`: reuse its OutputAggPlan.
        if let Some(prev_idx) = (0..out_idx)
            .find(|&i| output_plan[i].is_some() && same_aggr_expr(&aggr_exprs[i], aggr))
        {
            output_plan[out_idx] = output_plan[prev_idx].clone();
            continue;
        }

        let resolved: Option<OutputAggPlan> = match cand {
            AvgShareCandidate::Decimal128 {
                arg,
                sum_dt,
                averager,
            } => {
                let existing_sum = phys_aggrs.iter().position(|p| {
                    matches!(p.kind, PhysAggKind::SumDecimal128 { .. })
                        && p.args.len() == 1
                        && p.args[0].eq(arg)
                });
                let sum_phys_idx = if let Some(idx) = existing_sum {
                    Some(idx)
                } else if has_explicit_count_star || decomposable_avg_count >= 2 {
                    let fast_input = try_extract_decimal128_input(arg, input_schema);
                    let idx = phys_aggrs.len();
                    phys_aggrs.push(PhysAggSpec {
                        kind: PhysAggKind::SumDecimal128 {
                            data_type: sum_dt.clone(),
                            fast_input,
                            rebound_fast_input: None,
                        },
                        args: vec![Arc::clone(arg)],
                        rebound_args: Vec::new(),
                        state_types: vec![sum_dt.clone()],
                    });
                    Some(idx)
                } else {
                    None
                };
                if let Some(sum_phys_idx) = sum_phys_idx {
                    let count_phys_idx =
                        ensure_count_for_expr(arg, &mut phys_aggrs, &mut count_star_idx)?;
                    Some(OutputAggPlan::AvgFromSumCount {
                        sum_phys_idx,
                        count_phys_idx,
                        kind: SharedAvgKind::Decimal128(*averager),
                    })
                } else {
                    None
                }
            }
            AvgShareCandidate::Float64 { arg } => {
                let existing_sum = phys_aggrs.iter().position(|p| {
                    matches!(p.kind, PhysAggKind::SumFloat64 { .. })
                        && p.args.len() == 1
                        && p.args[0].eq(arg)
                });
                let sum_phys_idx = if let Some(idx) = existing_sum {
                    Some(idx)
                } else if has_explicit_count_star || decomposable_avg_count >= 2 {
                    let direct_col = arg
                        .downcast_ref::<Column>()
                        .filter(|c| {
                            input_schema.field(c.index()).data_type()
                                == &DataType::Float64
                        })
                        .map(|c| c.index());
                    let idx = phys_aggrs.len();
                    phys_aggrs.push(PhysAggSpec {
                        kind: PhysAggKind::SumFloat64 {
                            direct_col,
                            rebound_direct_col: None,
                        },
                        args: vec![Arc::clone(arg)],
                        rebound_args: Vec::new(),
                        state_types: vec![DataType::Float64],
                    });
                    Some(idx)
                } else {
                    None
                };
                if let Some(sum_phys_idx) = sum_phys_idx {
                    let count_phys_idx =
                        ensure_count_for_expr(arg, &mut phys_aggrs, &mut count_star_idx)?;
                    Some(OutputAggPlan::AvgFromSumCount {
                        sum_phys_idx,
                        count_phys_idx,
                        kind: SharedAvgKind::Float64,
                    })
                } else {
                    None
                }
            }
            AvgShareCandidate::IntSmallToFloat64 { base_expr } => {
                let existing_sum = phys_aggrs.iter().position(|p| {
                    matches!(p.kind, PhysAggKind::SumInt64 { .. })
                        && p.args.len() == 1
                        && unwrap_widening_int_cast(&p.args[0], input_schema)
                            .is_some_and(|(inner, _)| inner.eq(base_expr))
                });
                if let Some(sum_phys_idx) = existing_sum {
                    let count_phys_idx = ensure_count_for_expr(
                        base_expr,
                        &mut phys_aggrs,
                        &mut count_star_idx,
                    )?;
                    Some(OutputAggPlan::AvgFromSumCount {
                        sum_phys_idx,
                        count_phys_idx,
                        kind: SharedAvgKind::Int64ToFloat64,
                    })
                } else {
                    None
                }
            }
            AvgShareCandidate::Int64ToFloat64 { base_expr } => {
                let existing_sum = phys_aggrs.iter().position(|p| {
                    matches!(
                        p.kind,
                        PhysAggKind::SumInt64 { .. } | PhysAggKind::SumInt64Wide { .. }
                    ) && p.args.len() == 1
                        && p.args[0].eq(base_expr)
                });
                if let Some(sum_phys_idx) = existing_sum {
                    // Promote `SumInt64` to `SumInt64Wide` (`i128`) so `SUM` keeps exact
                    // wrapping `i64` semantics while `AVG` avoids `i64` overflow parity bugs.
                    if matches!(
                        phys_aggrs[sum_phys_idx].kind,
                        PhysAggKind::SumInt64 { .. }
                    ) {
                        let direct_col = base_expr
                            .downcast_ref::<Column>()
                            .filter(|c| {
                                input_schema.field(c.index()).data_type()
                                    == &DataType::Int64
                            })
                            .map(|c| c.index());
                        phys_aggrs[sum_phys_idx].kind = PhysAggKind::SumInt64Wide {
                            direct_col,
                            rebound_direct_col: None,
                        };
                        phys_aggrs[sum_phys_idx].state_types =
                            vec![DataType::Decimal128(DECIMAL128_MAX_PRECISION, 0)];
                        for plan in output_plan.iter_mut().flatten() {
                            if matches!(
                                plan,
                                OutputAggPlan::Direct { phys_idx }
                                    if *phys_idx == sum_phys_idx
                            ) {
                                *plan = OutputAggPlan::SumInt64FromWide {
                                    phys_idx: sum_phys_idx,
                                };
                            }
                        }
                    }
                    let count_phys_idx = ensure_count_for_expr(
                        base_expr,
                        &mut phys_aggrs,
                        &mut count_star_idx,
                    )?;
                    Some(OutputAggPlan::AvgFromSumCount {
                        sum_phys_idx,
                        count_phys_idx,
                        kind: SharedAvgKind::Int128ToFloat64,
                    })
                } else {
                    None
                }
            }
        };

        if let Some(plan) = resolved {
            output_plan[out_idx] = Some(plan);
        } else {
            let state_types = aggr
                .state_fields()?
                .into_iter()
                .map(|f| f.data_type().clone())
                .collect();
            let idx = phys_aggrs.len();
            phys_aggrs.push(PhysAggSpec {
                kind: PhysAggKind::Generic(Arc::clone(aggr)),
                args: aggr.expressions(),
                rebound_args: Vec::new(),
                state_types,
            });
            output_plan[out_idx] = Some(OutputAggPlan::Direct { phys_idx: idx });
        }
    }

    // Link every `CountCol` to `CountStar` when present so non-null batches do zero work.
    for phys in &mut phys_aggrs {
        if let PhysAggKind::CountCol {
            count_star_idx: slot,
            ..
        } = &mut phys.kind
        {
            *slot = count_star_idx;
        }
    }

    let output_plan = output_plan
        .into_iter()
        .map(|o| {
            o.ok_or_else(|| {
                datafusion_common::internal_datafusion_err!(
                    "Unresolved output aggregate plan entry"
                )
            })
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(AggregatePlan {
        phys_aggrs,
        output_plan,
        count_star_idx,
    })
}

fn evaluate_output_plan(
    output_plan: &[OutputAggPlan],
    phys_arrays: &[ArrayRef],
) -> Result<Vec<ArrayRef>> {
    let mut out = Vec::with_capacity(output_plan.len());
    for entry in output_plan {
        match entry {
            OutputAggPlan::Direct { phys_idx } => {
                out.push(Arc::clone(&phys_arrays[*phys_idx]));
            }
            OutputAggPlan::SumInt64FromWide { phys_idx } => {
                let arr = phys_arrays[*phys_idx]
                    .as_any()
                    .downcast_ref::<Decimal128Array>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected Decimal128 wide sum array"
                        )
                    })?;
                let vals: Vec<i64> = arr.values().iter().map(|&v| v as i64).collect();
                out.push(Arc::new(Int64Array::new(vals.into(), arr.nulls().cloned())));
            }
            OutputAggPlan::AvgFromSumCount {
                sum_phys_idx,
                count_phys_idx,
                kind,
            } => {
                let counts = phys_arrays[*count_phys_idx]
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected Int64 count array for shared AVG"
                        )
                    })?;
                let len = counts.len();
                let count_vals = counts.values();
                match kind {
                    SharedAvgKind::Float64 => {
                        let sums = phys_arrays[*sum_phys_idx]
                            .as_any()
                            .downcast_ref::<Float64Array>()
                            .ok_or_else(|| {
                                datafusion_common::internal_datafusion_err!(
                                    "Expected Float64 sum array for shared AVG"
                                )
                            })?;
                        let sum_vals = sums.values();
                        let mut vals = Vec::with_capacity(len);
                        let mut nulls = NullBufferBuilder::new(len);
                        for i in 0..len {
                            let c = count_vals[i];
                            if c > 0 && sums.is_valid(i) {
                                vals.push(sum_vals[i] / (c as f64));
                                nulls.append_non_null();
                            } else {
                                vals.push(0.0);
                                nulls.append_null();
                            }
                        }
                        out.push(Arc::new(Float64Array::new(
                            vals.into(),
                            nulls.finish(),
                        )));
                    }
                    SharedAvgKind::Int64ToFloat64 => {
                        let sums = phys_arrays[*sum_phys_idx]
                            .as_any()
                            .downcast_ref::<Int64Array>()
                            .ok_or_else(|| {
                                datafusion_common::internal_datafusion_err!(
                                    "Expected Int64 sum array for shared AVG"
                                )
                            })?;
                        let sum_vals = sums.values();
                        let mut vals = Vec::with_capacity(len);
                        let mut nulls = NullBufferBuilder::new(len);
                        for i in 0..len {
                            let c = count_vals[i];
                            if c > 0 && sums.is_valid(i) {
                                vals.push((sum_vals[i] as f64) / (c as f64));
                                nulls.append_non_null();
                            } else {
                                vals.push(0.0);
                                nulls.append_null();
                            }
                        }
                        out.push(Arc::new(Float64Array::new(
                            vals.into(),
                            nulls.finish(),
                        )));
                    }
                    SharedAvgKind::Int128ToFloat64 => {
                        let sums = phys_arrays[*sum_phys_idx]
                            .as_any()
                            .downcast_ref::<Decimal128Array>()
                            .ok_or_else(|| {
                                datafusion_common::internal_datafusion_err!(
                                    "Expected Decimal128 wide sum array for shared AVG"
                                )
                            })?;
                        let sum_vals = sums.values();
                        let mut vals = Vec::with_capacity(len);
                        let mut nulls = NullBufferBuilder::new(len);
                        for i in 0..len {
                            let c = count_vals[i];
                            if c > 0 && sums.is_valid(i) {
                                vals.push((sum_vals[i] as f64) / (c as f64));
                                nulls.append_non_null();
                            } else {
                                vals.push(0.0);
                                nulls.append_null();
                            }
                        }
                        out.push(Arc::new(Float64Array::new(
                            vals.into(),
                            nulls.finish(),
                        )));
                    }
                    SharedAvgKind::Decimal128(averager) => {
                        let sums = phys_arrays[*sum_phys_idx]
                            .as_any()
                            .downcast_ref::<Decimal128Array>()
                            .ok_or_else(|| {
                                datafusion_common::internal_datafusion_err!(
                                    "Expected Decimal128 sum array for shared AVG"
                                )
                            })?;
                        let sum_vals = sums.values();
                        let mut vals = Vec::with_capacity(len);
                        let mut nulls = NullBufferBuilder::new(len);
                        for i in 0..len {
                            let c = count_vals[i];
                            if c > 0 && sums.is_valid(i) {
                                vals.push(averager.avg(sum_vals[i], c)?);
                                nulls.append_non_null();
                            } else {
                                vals.push(0);
                                nulls.append_null();
                            }
                        }
                        out.push(Arc::new(
                            Decimal128Array::new(vals.into(), nulls.finish())
                                .with_precision_and_scale(
                                    averager.target_precision,
                                    averager.target_scale,
                                )?,
                        ));
                    }
                }
            }
        }
    }
    Ok(out)
}

#[derive(Clone, Debug)]
enum DirectAggSpec {
    CountStar,
    CountCol {
        col: usize,
        count_star_idx: Option<usize>,
    },
    SumInt64(usize),
    SumInt32(usize),
    SumInt64Wide(usize),
    SumFloat64(usize),
    SumDecimal128 {
        data_type: DataType,
        input: Decimal128Input,
    },
    AvgFloat64(usize),
    MinInt64(usize),
    MinInt32(usize),
    MaxInt64(usize),
    MaxInt32(usize),
}

#[derive(Clone, Debug)]
struct DirectGroupedSpec {
    key_index: usize,
    key_type: IntKeyType,
    aggs: Vec<DirectAggSpec>,
    output_plan: Vec<OutputAggPlan>,
    count_star_idx: Option<usize>,
}

fn try_build_direct_grouped_spec(
    group_exprs: &[Arc<dyn PhysicalExpr>],
    plan: &AggregatePlan,
    input_schema: &SchemaRef,
) -> Option<DirectGroupedSpec> {
    let [group_expr] = group_exprs else {
        return None;
    };
    let key_col = group_expr.downcast_ref::<Column>()?;
    let key_index = key_col.index();
    let key_type = IntKeyType::from_data_type(input_schema.field(key_index).data_type())?;

    let mut aggs = Vec::with_capacity(plan.phys_aggrs.len());
    for phys in &plan.phys_aggrs {
        let spec = match &phys.kind {
            PhysAggKind::CountStar => DirectAggSpec::CountStar,
            PhysAggKind::CountCol {
                direct_col: Some(col),
                count_star_idx,
                ..
            } => DirectAggSpec::CountCol {
                col: *col,
                count_star_idx: *count_star_idx,
            },
            PhysAggKind::SumInt64 {
                direct_col: Some((col, SumIntInputKind::Int64)),
                ..
            } => DirectAggSpec::SumInt64(*col),
            PhysAggKind::SumInt64 {
                direct_col: Some((col, SumIntInputKind::Int32)),
                ..
            } => DirectAggSpec::SumInt32(*col),
            PhysAggKind::SumInt64Wide {
                direct_col: Some(col),
                ..
            } => DirectAggSpec::SumInt64Wide(*col),
            PhysAggKind::SumFloat64 {
                direct_col: Some(col),
                ..
            } => DirectAggSpec::SumFloat64(*col),
            PhysAggKind::SumDecimal128 {
                data_type,
                fast_input: Some(input),
                ..
            } => DirectAggSpec::SumDecimal128 {
                data_type: data_type.clone(),
                input: *input,
            },
            PhysAggKind::Generic(aggr) => {
                if aggr.is_distinct() {
                    return None;
                }
                let inner = aggr.fun().inner();
                let exprs = aggr.expressions();
                let [arg] = exprs.as_slice() else {
                    return None;
                };
                let idx = arg.downcast_ref::<Column>()?.index();
                let out_field = aggr.field();
                let out_dt = out_field.data_type();
                if inner.is::<Avg>() {
                    match (input_schema.field(idx).data_type(), out_dt) {
                        (DataType::Float64, DataType::Float64) => {
                            DirectAggSpec::AvgFloat64(idx)
                        }
                        _ => return None,
                    }
                } else if inner.is::<Min>() {
                    match (input_schema.field(idx).data_type(), out_dt) {
                        (DataType::Int64, DataType::Int64) => {
                            DirectAggSpec::MinInt64(idx)
                        }
                        (DataType::Int32, DataType::Int32) => {
                            DirectAggSpec::MinInt32(idx)
                        }
                        _ => return None,
                    }
                } else if inner.is::<Max>() {
                    match (input_schema.field(idx).data_type(), out_dt) {
                        (DataType::Int64, DataType::Int64) => {
                            DirectAggSpec::MaxInt64(idx)
                        }
                        (DataType::Int32, DataType::Int32) => {
                            DirectAggSpec::MaxInt32(idx)
                        }
                        _ => return None,
                    }
                } else {
                    return None;
                }
            }
            _ => return None,
        };
        aggs.push(spec);
    }

    Some(DirectGroupedSpec {
        key_index,
        key_type,
        aggs,
        output_plan: plan.output_plan.clone(),
        count_star_idx: plan.count_star_idx,
    })
}

fn direct_grouped_partial_schema(spec: &DirectGroupedSpec) -> SchemaRef {
    let mut fields = Vec::with_capacity(1 + spec.aggs.len() * 2);
    fields.push(Field::new("k", spec.key_type.data_type(), true));
    for (idx, agg) in spec.aggs.iter().enumerate() {
        match agg {
            DirectAggSpec::CountStar | DirectAggSpec::CountCol { .. } => {
                fields.push(Field::new(format!("c_{idx}"), DataType::Int64, false));
            }
            DirectAggSpec::SumInt64(_) | DirectAggSpec::SumInt32(_) => {
                fields.push(Field::new(format!("s_{idx}"), DataType::Int64, true));
            }
            DirectAggSpec::SumInt64Wide(_) => {
                fields.push(Field::new(
                    format!("s_{idx}"),
                    DataType::Decimal128(DECIMAL128_MAX_PRECISION, 0),
                    true,
                ));
            }
            DirectAggSpec::SumFloat64(_) => {
                fields.push(Field::new(format!("s_{idx}"), DataType::Float64, true));
            }
            DirectAggSpec::SumDecimal128 { data_type, .. } => {
                fields.push(Field::new(format!("s_{idx}"), data_type.clone(), true));
            }
            DirectAggSpec::AvgFloat64(_) => {
                fields.push(Field::new(format!("s_{idx}"), DataType::Float64, false));
                fields.push(Field::new(format!("n_{idx}"), DataType::UInt64, false));
            }
            DirectAggSpec::MinInt64(_) | DirectAggSpec::MaxInt64(_) => {
                fields.push(Field::new(format!("m_{idx}"), DataType::Int64, true));
            }
            DirectAggSpec::MinInt32(_) | DirectAggSpec::MaxInt32(_) => {
                fields.push(Field::new(format!("m_{idx}"), DataType::Int32, true));
            }
        }
    }
    Arc::new(Schema::new(fields))
}

#[derive(Debug)]
pub(super) struct AggregateSink {
    schema: SchemaRef,
    group_schema: Option<SchemaRef>,
    partial_schema: Option<SchemaRef>,
    state_col_counts: Vec<usize>,
    has_float_keys: bool,
    group_exprs: Vec<Arc<dyn PhysicalExpr>>,
    group_names: Vec<String>,
    aggr_exprs: Vec<Arc<AggregateFunctionExpr>>,
    plan: Box<AggregatePlan>,
    needed_columns: Vec<usize>,
    rebound_group_exprs: Vec<Arc<dyn PhysicalExpr>>,
    direct_grouped: Option<Box<DirectGroupedSpec>>,
    #[cfg(test)]
    pub(super) finalize_gate: Option<super::tests::FinalizeGate>,
}

impl AggregateSink {
    #[cfg_attr(not(test), expect(dead_code))]
    pub(super) fn try_new(
        aggregate: &Aggregate,
        input_schema: &SchemaRef,
        session: &dyn Session,
    ) -> Result<std::result::Result<Self, String>> {
        Self::try_new_with_context(
            aggregate,
            input_schema,
            session,
            &PhysicalPlanningContext::default(),
        )
    }

    pub(super) fn try_new_with_context(
        aggregate: &Aggregate,
        input_schema: &SchemaRef,
        session: &dyn Session,
        planning_ctx: &PhysicalPlanningContext,
    ) -> Result<std::result::Result<Self, String>> {
        let logical_input_schema = aggregate.input.schema();
        let props = session.execution_props();

        let (group_schema, group_exprs, group_names, mut output_fields) =
            if aggregate.group_expr.is_empty() {
                (None, Vec::new(), Vec::new(), Vec::new())
            } else {
                let mut pairs = Vec::with_capacity(aggregate.group_expr.len());
                let mut exprs = Vec::with_capacity(aggregate.group_expr.len());
                let mut names = Vec::with_capacity(aggregate.group_expr.len());
                for expr in &aggregate.group_expr {
                    let phys = create_physical_expr(
                        expr,
                        logical_input_schema,
                        props,
                        planning_ctx,
                    )?;
                    let name = physical_name(expr)?;
                    exprs.push(Arc::clone(&phys));
                    names.push(name.clone());
                    pairs.push((phys, name));
                }
                let group_by = PhysicalGroupBy::new_single(pairs);
                let g_schema = group_by.group_schema(input_schema)?;
                if let Err(err) =
                    new_group_values(Arc::clone(&g_schema), &GroupClustering::None)
                {
                    return Ok(Err(format!("unsupported GROUP BY key type: {err}")));
                }
                let fields = g_schema.fields().iter().cloned().collect::<Vec<_>>();
                (Some(g_schema), exprs, names, fields)
            };

        let mut aggr_exprs = Vec::with_capacity(aggregate.aggr_expr.len());
        for expr in &aggregate.aggr_expr {
            let lowered = LoweredAggregateBuilder::new(
                expr,
                logical_input_schema,
                input_schema,
                props,
                planning_ctx,
            )
            .build()?;
            let distinct_ok = lowered.aggregate.is_distinct()
                && lowered.aggregate.fun().inner().is::<Count>()
                && lowered.aggregate.expressions().len() == 1;
            if lowered.filter.is_some()
                || !lowered.order_bys.is_empty()
                || (lowered.aggregate.is_distinct() && !distinct_ok)
                || lowered.aggregate.ignore_nulls()
            {
                return Ok(Err(format!(
                    "unsupported aggregate modifier on {}",
                    lowered.aggregate.name()
                )));
            }
            let aggr = lowered.aggregate;
            // Validate accumulator construction eagerly at planning time.
            let check = if group_schema.is_some() {
                create_groups_accumulator(&aggr, true).map(|_| ())
            } else {
                aggr.create_accumulator().map(|_| ())
            };
            if let Err(err) = check {
                return Ok(Err(format!(
                    "unsupported aggregate '{}': {err}",
                    aggr.name()
                )));
            }
            output_fields.push(aggr.field());
            aggr_exprs.push(aggr);
        }

        let schema = Arc::new(Schema::new_with_metadata(
            output_fields,
            input_schema.metadata().clone(),
        ));

        let mut plan = build_aggregate_plan(&aggr_exprs, input_schema)?;
        let mut all_exprs: Vec<Arc<dyn PhysicalExpr>> = group_exprs.clone();
        for phys in &plan.phys_aggrs {
            all_exprs.extend(phys.args.iter().cloned());
        }
        let needed_columns = collect_expr_column_indices(&all_exprs);
        let rebound_group_exprs = group_exprs
            .iter()
            .map(|e| remap_expr_columns(Arc::clone(e), &needed_columns))
            .collect::<Result<Vec<_>>>()?;
        for phys in &mut plan.phys_aggrs {
            phys.rebound_args = phys
                .args
                .iter()
                .map(|e| remap_expr_columns(Arc::clone(e), &needed_columns))
                .collect::<Result<Vec<_>>>()?;
            match &mut phys.kind {
                PhysAggKind::CountCol {
                    direct_col,
                    rebound_direct_col,
                    ..
                } => {
                    *rebound_direct_col =
                        direct_col.and_then(|c| needed_columns.binary_search(&c).ok());
                }
                PhysAggKind::SumDecimal128 {
                    fast_input,
                    rebound_fast_input,
                    ..
                } => {
                    *rebound_fast_input =
                        fast_input.and_then(|fi| fi.remap(&needed_columns));
                }
                PhysAggKind::SumInt64 {
                    direct_col,
                    rebound_direct_col,
                } => {
                    *rebound_direct_col = direct_col.and_then(|(c, kind)| {
                        needed_columns.binary_search(&c).ok().map(|rc| (rc, kind))
                    });
                }
                PhysAggKind::SumInt64Wide {
                    direct_col,
                    rebound_direct_col,
                }
                | PhysAggKind::SumFloat64 {
                    direct_col,
                    rebound_direct_col,
                } => {
                    *rebound_direct_col =
                        direct_col.and_then(|c| needed_columns.binary_search(&c).ok());
                }
                PhysAggKind::CountStar | PhysAggKind::Generic(_) => {}
            }
        }
        let direct_grouped =
            try_build_direct_grouped_spec(&group_exprs, &plan, input_schema)
                .map(Box::new);
        let has_float_keys = group_schema.as_ref().is_some_and(has_float_group_type);
        let (partial_schema, state_col_counts) = if let Some(spec) = &direct_grouped {
            (Some(direct_grouped_partial_schema(spec)), Vec::new())
        } else if let Some(gs) = &group_schema {
            let mut fields: Vec<Arc<Field>> = gs
                .fields()
                .iter()
                .map(|f| Arc::new((**f).clone().with_nullable(true)))
                .collect();
            let mut counts = Vec::with_capacity(plan.phys_aggrs.len());
            for (i, phys) in plan.phys_aggrs.iter().enumerate() {
                counts.push(phys.state_types.len());
                for (j, dt) in phys.state_types.iter().enumerate() {
                    fields.push(Arc::new(Field::new(
                        format!("s_{i}_{j}"),
                        dt.clone(),
                        true,
                    )));
                }
            }
            (Some(Arc::new(Schema::new(fields))), counts)
        } else {
            (None, Vec::new())
        };

        Ok(Ok(Self {
            schema,
            group_schema,
            partial_schema,
            state_col_counts,
            has_float_keys,
            group_exprs,
            group_names,
            aggr_exprs,
            plan: Box::new(plan),
            needed_columns,
            rebound_group_exprs,
            direct_grouped,
            #[cfg(test)]
            finalize_gate: None,
        }))
    }

    pub(super) fn try_new_distinct(
        input_schema: &SchemaRef,
    ) -> Result<std::result::Result<Self, String>> {
        if input_schema.fields().is_empty() {
            return Ok(Err(
                "DISTINCT on zero-column schema is not supported in native morsels"
                    .into(),
            ));
        }
        let mut group_exprs: Vec<Arc<dyn PhysicalExpr>> =
            Vec::with_capacity(input_schema.fields().len());
        let mut group_names = Vec::with_capacity(input_schema.fields().len());
        for (idx, field) in input_schema.fields().iter().enumerate() {
            group_exprs.push(Arc::new(Column::new(field.name(), idx)));
            group_names.push(field.name().clone());
        }
        if let Err(err) =
            new_group_values(Arc::clone(input_schema), &GroupClustering::None)
        {
            return Ok(Err(format!("unsupported DISTINCT key type: {err}")));
        }
        let needed_columns: Vec<usize> = (0..input_schema.fields().len()).collect();
        let rebound_group_exprs = group_exprs.clone();
        let plan = AggregatePlan::empty();
        let direct_grouped =
            try_build_direct_grouped_spec(&group_exprs, &plan, input_schema)
                .map(Box::new);
        let has_float_keys = has_float_group_type(input_schema);
        let partial_schema = if let Some(spec) = &direct_grouped {
            Some(direct_grouped_partial_schema(spec))
        } else {
            let fields: Vec<Arc<Field>> = input_schema
                .fields()
                .iter()
                .map(|f| Arc::new((**f).clone().with_nullable(true)))
                .collect();
            Some(Arc::new(Schema::new(fields)))
        };
        Ok(Ok(Self {
            schema: Arc::clone(input_schema),
            group_schema: Some(Arc::clone(input_schema)),
            partial_schema,
            state_col_counts: Vec::new(),
            has_float_keys,
            group_exprs,
            group_names,
            aggr_exprs: Vec::new(),
            plan: Box::new(plan),
            needed_columns,
            rebound_group_exprs,
            direct_grouped,
            #[cfg(test)]
            finalize_gate: None,
        }))
    }

    pub(super) fn is_grouped(&self) -> bool {
        self.group_schema.is_some()
    }

    #[cfg(test)]
    pub(super) fn physical_accumulator_count(&self) -> usize {
        self.plan.phys_aggrs.len()
    }

    #[cfg(feature = "parquet")]
    pub(super) fn group_exprs(&self) -> &[Arc<dyn PhysicalExpr>] {
        &self.group_exprs
    }

    pub(super) fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    pub(super) fn partial_schema(&self) -> Option<&SchemaRef> {
        self.partial_schema.as_ref()
    }

    #[cfg_attr(not(test), expect(dead_code))]
    pub(super) fn new_state(
        &self,
        reservation: MemoryReservation,
    ) -> Result<AggregateState> {
        self.new_state_with_spill(reservation, None)
    }

    pub(super) fn new_state_with_spill(
        &self,
        reservation: MemoryReservation,
        spill: Option<SpillContext>,
    ) -> Result<AggregateState> {
        if let Some(spec) = &self.direct_grouped {
            let partial_schema = Arc::clone(
                self.partial_schema
                    .as_ref()
                    .expect("direct grouped partial schema"),
            );
            return Ok(AggregateState::DirectGrouped(DirectGroupedState::new(
                spec,
                partial_schema,
                reservation,
                spill,
            )));
        }

        let aggr_args = self
            .plan
            .phys_aggrs
            .iter()
            .map(|phys| phys.args.iter().cloned().map(ArgEvaluator::new).collect())
            .collect::<Vec<_>>();
        let rebound_aggr_args = self
            .plan
            .phys_aggrs
            .iter()
            .map(|phys| {
                phys.rebound_args
                    .iter()
                    .cloned()
                    .map(ArgEvaluator::new)
                    .collect()
            })
            .collect::<Vec<_>>();

        match &self.group_schema {
            None => {
                let accumulators = self
                    .plan
                    .phys_aggrs
                    .iter()
                    .map(FastUngroupedAccumulator::from_spec)
                    .collect::<Result<Vec<_>>>()?;
                let all_direct_view = accumulators
                    .iter()
                    .all(FastUngroupedAccumulator::supports_direct_view);
                Ok(AggregateState::Ungrouped(UngroupedState {
                    accumulators,
                    aggr_args,
                    needed_columns: self.needed_columns.clone(),
                    rebound_aggr_args,
                    all_direct_view,
                    has_rows: false,
                    reservation,
                }))
            }
            Some(group_schema) => {
                let group_values = FastGroupValues::new(Arc::clone(group_schema))?;
                let group_exprs = self
                    .group_exprs
                    .iter()
                    .cloned()
                    .map(ArgEvaluator::new)
                    .collect();
                let rebound_group_exprs = self
                    .rebound_group_exprs
                    .iter()
                    .cloned()
                    .map(ArgEvaluator::new)
                    .collect();
                let accumulators = self
                    .plan
                    .phys_aggrs
                    .iter()
                    .map(FastGroupsAccumulator::from_spec)
                    .collect::<Result<Vec<_>>>()?;
                let partial_schema = Arc::clone(
                    self.partial_schema
                        .as_ref()
                        .expect("grouped partial schema"),
                );
                Ok(AggregateState::Grouped(GroupedState {
                    group_values,
                    group_schema: Arc::clone(group_schema),
                    group_exprs,
                    rebound_group_exprs,
                    accumulators,
                    count_star_idx: self.plan.count_star_idx,
                    aggr_args,
                    rebound_aggr_args,
                    needed_columns: self.needed_columns.clone(),
                    group_indices: Vec::new(),
                    partial_schema,
                    has_float_keys: self.has_float_keys,
                    table_tracked_bytes: 0,
                    radix: None,
                    spill,
                    hash_buf: Vec::new(),
                    reservation,
                }))
            }
        }
    }

    pub(super) fn finalize(&self, combined: AggregateState) -> FinalizedAggregate {
        FinalizedAggregate {
            schema: Arc::clone(&self.schema),
            group_schema: self.group_schema.clone(),
            plan: (*self.plan).clone(),
            state_col_counts: self.state_col_counts.clone(),
            has_float_keys: self.has_float_keys,
            direct_grouped: self.direct_grouped.as_deref().cloned(),
            state: combined,
        }
    }

    pub(super) fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        let aggr_args: Vec<Arc<dyn PhysicalExpr>> = self
            .aggr_exprs
            .iter()
            .flat_map(|aggr| aggr.expressions())
            .collect();
        apply_expression_roots(self.group_exprs.iter().chain(aggr_args.iter()), f)
    }

    pub(super) fn display_summary(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "AggregateSink[")?;
        if !self.group_names.is_empty() {
            write!(f, "group=[{}]", self.group_names.join(", "))?;
            if !self.aggr_exprs.is_empty() {
                write!(f, ", ")?;
            }
        }
        if !self.aggr_exprs.is_empty() {
            write!(f, "aggr=[")?;
            for (i, aggr) in self.aggr_exprs.iter().enumerate() {
                if i > 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", aggr.name())?;
            }
            write!(f, "]")?;
        }
        write!(f, "]")
    }
}

pub(super) enum AggregateState {
    Ungrouped(UngroupedState),
    Grouped(GroupedState),
    DirectGrouped(DirectGroupedState),
}

impl AggregateState {
    #[cfg(test)]
    pub(super) fn add_view(&mut self, view: &BatchView) -> Result<()> {
        if view.is_empty() {
            return Ok(());
        }
        match self {
            Self::Ungrouped(state) => state.add_view(view),
            Self::Grouped(state) => state.add_view(view),
            Self::DirectGrouped(state) => state.add_view(view),
        }
    }

    pub(super) fn add_view_spillable(
        &mut self,
        view: &BatchView,
        tx: &mpsc::Sender<RecordBatch>,
    ) -> Result<()> {
        if view.is_empty() {
            return Ok(());
        }
        match self {
            Self::Ungrouped(state) => state.add_view(view),
            Self::Grouped(state) => state.add_view_spillable(view, tx),
            Self::DirectGrouped(state) => state.add_view_spillable(view, tx),
        }
    }

    #[cfg(test)]
    pub(super) fn add_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        if batch.num_rows() == 0 {
            return Ok(());
        }
        self.add_view(&BatchView::all(batch.clone()))
    }

    pub(super) async fn merge_from(
        &mut self,
        other: Self,
        tx: &mpsc::Sender<RecordBatch>,
    ) -> Result<bool> {
        match (self, other) {
            (Self::Ungrouped(dst), Self::Ungrouped(src)) => {
                dst.merge_from(src)?;
                Ok(true)
            }
            (Self::Grouped(dst), Self::Grouped(src)) => dst.merge_from(src, tx).await,
            (Self::DirectGrouped(dst), Self::DirectGrouped(src)) => {
                dst.merge_from(src, tx).await
            }
            _ => internal_err!("Morsel aggregate worker state kind mismatch"),
        }
    }
}

#[inline]
fn has_nested_logical_nulls(dt: &DataType) -> bool {
    matches!(
        dt,
        DataType::Null
            | DataType::Dictionary(_, _)
            | DataType::Union(_, _)
            | DataType::RunEndEncoded(_, _)
    )
}

#[inline]
fn array_has_no_nulls_fast(array: &ArrayRef) -> bool {
    !has_nested_logical_nulls(array.data_type()) && array.null_count() == 0
}

#[inline]
fn ensure_has_val(
    has_val: &mut Option<Vec<bool>>,
    prev_num_groups: usize,
    total_num_groups: usize,
) -> &mut Vec<bool> {
    let hv = has_val.get_or_insert_with(|| vec![true; prev_num_groups]);
    if hv.len() < total_num_groups {
        hv.resize(total_num_groups, false);
    }
    hv
}

fn build_null_buffer_from_has_val(
    has_val: Option<Vec<bool>>,
) -> Option<arrow::buffer::NullBuffer> {
    let hv = has_val?;
    let mut builder = NullBufferBuilder::new(hv.len());
    for v in hv {
        builder.append(v);
    }
    builder.finish()
}

macro_rules! for_each_selected_row_simple {
    ($selection:expr, $num_rows:expr, $row:ident => $body:expr) => {
        match $selection {
            Selection::All => {
                for $row in 0..$num_rows {
                    $body
                }
            }
            Selection::Mask { bits, .. } => {
                for $row in bits.set_indices() {
                    $body
                }
            }
            Selection::Indices(indices) => {
                for &idx in indices.values() {
                    let $row = idx as usize;
                    $body
                }
            }
        }
    };
}

enum FastUngroupedAccumulator {
    CountStar {
        count: i64,
    },
    CountCol {
        direct_col: Option<usize>,
        rebound_direct_col: Option<usize>,
        count: i64,
    },
    SumDecimal128 {
        data_type: DataType,
        fast_input: Option<Decimal128Input>,
        rebound_fast_input: Option<Decimal128Input>,
        sum: i128,
        has_val: bool,
    },
    SumInt64 {
        direct_col: Option<(usize, SumIntInputKind)>,
        rebound_direct_col: Option<(usize, SumIntInputKind)>,
        sum: i64,
        has_val: bool,
    },
    SumInt64Wide {
        direct_col: Option<usize>,
        rebound_direct_col: Option<usize>,
        sum: i128,
        has_val: bool,
    },
    SumFloat64 {
        direct_col: Option<usize>,
        rebound_direct_col: Option<usize>,
        sum: f64,
        has_val: bool,
    },
    Generic(Box<dyn Accumulator>),
}

impl FastUngroupedAccumulator {
    fn from_spec(spec: &PhysAggSpec) -> Result<Self> {
        Ok(match &spec.kind {
            PhysAggKind::CountStar => Self::CountStar { count: 0 },
            PhysAggKind::CountCol {
                direct_col,
                rebound_direct_col,
                ..
            } => Self::CountCol {
                direct_col: *direct_col,
                rebound_direct_col: *rebound_direct_col,
                count: 0,
            },
            PhysAggKind::SumDecimal128 {
                data_type,
                fast_input,
                rebound_fast_input,
            } => Self::SumDecimal128 {
                data_type: data_type.clone(),
                fast_input: *fast_input,
                rebound_fast_input: *rebound_fast_input,
                sum: 0,
                has_val: false,
            },
            PhysAggKind::SumInt64 {
                direct_col,
                rebound_direct_col,
            } => Self::SumInt64 {
                direct_col: *direct_col,
                rebound_direct_col: *rebound_direct_col,
                sum: 0,
                has_val: false,
            },
            PhysAggKind::SumInt64Wide {
                direct_col,
                rebound_direct_col,
            } => Self::SumInt64Wide {
                direct_col: *direct_col,
                rebound_direct_col: *rebound_direct_col,
                sum: 0,
                has_val: false,
            },
            PhysAggKind::SumFloat64 {
                direct_col,
                rebound_direct_col,
            } => Self::SumFloat64 {
                direct_col: *direct_col,
                rebound_direct_col: *rebound_direct_col,
                sum: 0.0,
                has_val: false,
            },
            PhysAggKind::Generic(aggr) => Self::Generic(aggr.create_accumulator()?),
        })
    }

    fn supports_direct_view(&self) -> bool {
        match self {
            Self::CountStar { .. } => true,
            Self::CountCol { direct_col, .. } => direct_col.is_some(),
            Self::SumDecimal128 { fast_input, .. } => fast_input.is_some(),
            Self::SumInt64 { direct_col, .. } => direct_col.is_some(),
            Self::SumInt64Wide { direct_col, .. } => direct_col.is_some(),
            Self::SumFloat64 { direct_col, .. } => direct_col.is_some(),
            Self::Generic(_) => false,
        }
    }

    fn size(&self) -> usize {
        match self {
            Self::Generic(acc) => acc.size(),
            _ => size_of::<Self>(),
        }
    }

    fn update_view_direct(&mut self, view: &BatchView) -> Result<()> {
        let batch = view.batch();
        let selection = view.selection();
        let batch_rows = batch.num_rows();
        let selected_rows = view.num_rows();
        match self {
            Self::CountStar { count } => {
                let delta = i64::try_from(selected_rows).map_err(|_| count_overflow())?;
                *count = count.checked_add(delta).ok_or_else(count_overflow)?;
            }
            Self::CountCol {
                direct_col: Some(col),
                count,
                ..
            } => {
                let arr = batch.column(*col);
                if matches!(arr.data_type(), DataType::Null) {
                    return Ok(());
                }
                if array_has_no_nulls_fast(arr) {
                    let delta =
                        i64::try_from(selected_rows).map_err(|_| count_overflow())?;
                    *count = count.checked_add(delta).ok_or_else(count_overflow)?;
                } else if let Some(nulls) = arr.logical_nulls() {
                    if nulls.null_count() == 0 {
                        let delta =
                            i64::try_from(selected_rows).map_err(|_| count_overflow())?;
                        *count = count.checked_add(delta).ok_or_else(count_overflow)?;
                    } else if nulls.null_count() < arr.len() {
                        let mut valid = 0i64;
                        if view.is_all() {
                            valid = (arr.len() - nulls.null_count()) as i64;
                        } else {
                            for_each_selected_row_simple!(selection, batch_rows, row => {
                                if nulls.is_valid(row) {
                                    valid += 1;
                                }
                            });
                        }
                        *count = count.checked_add(valid).ok_or_else(count_overflow)?;
                    }
                } else {
                    let delta =
                        i64::try_from(selected_rows).map_err(|_| count_overflow())?;
                    *count = count.checked_add(delta).ok_or_else(count_overflow)?;
                }
            }
            Self::SumDecimal128 {
                fast_input: Some(input),
                sum,
                has_val,
                ..
            } => {
                let mut acc = *sum;
                let mut saw = *has_val;
                match *input {
                    Decimal128Input::Col(c) => {
                        let arr = batch
                            .column(c)
                            .as_any()
                            .downcast_ref::<Decimal128Array>()
                            .ok_or_else(|| {
                                datafusion_common::internal_datafusion_err!(
                                    "Expected Decimal128 column at {c}"
                                )
                            })?;
                        let vals = arr.values();
                        if arr.null_count() == 0 {
                            for_each_selected_row_simple!(selection, batch_rows, row => {
                                acc = acc.wrapping_add(vals[row]);
                            });
                            saw = true;
                        } else if arr.null_count() < arr.len() {
                            let nulls = arr.nulls().unwrap();
                            for_each_selected_row_simple!(selection, batch_rows, row => {
                                if nulls.is_valid(row) {
                                    acc = acc.wrapping_add(vals[row]);
                                    saw = true;
                                }
                            });
                        }
                    }
                    Decimal128Input::MulColCol { a, b }
                    | Decimal128Input::MulColAddLit { a, b, .. }
                    | Decimal128Input::MulColSubFromLit { a, b, .. }
                    | Decimal128Input::MulColSubLit { a, b, .. } => {
                        let arr_a = batch
                            .column(a)
                            .as_any()
                            .downcast_ref::<Decimal128Array>()
                            .ok_or_else(|| {
                                datafusion_common::internal_datafusion_err!(
                                    "Expected Decimal128 column at {a}"
                                )
                            })?;
                        let arr_b = batch
                            .column(b)
                            .as_any()
                            .downcast_ref::<Decimal128Array>()
                            .ok_or_else(|| {
                                datafusion_common::internal_datafusion_err!(
                                    "Expected Decimal128 column at {b}"
                                )
                            })?;
                        let va = arr_a.values();
                        let vb = arr_b.values();
                        if arr_a.null_count() == 0 && arr_b.null_count() == 0 {
                            match *input {
                                Decimal128Input::MulColCol { .. } => {
                                    for_each_selected_row_simple!(selection, batch_rows, row => {
                                        acc = acc.wrapping_add(checked_mul_i128_fast(va[row], vb[row])?);
                                    });
                                }
                                Decimal128Input::MulColAddLit { lit, .. } => {
                                    for_each_selected_row_simple!(selection, batch_rows, row => {
                                        let rhs = checked_add_i128(lit, vb[row])?;
                                        acc = acc.wrapping_add(checked_mul_i128_fast(va[row], rhs)?);
                                    });
                                }
                                Decimal128Input::MulColSubFromLit { lit, .. } => {
                                    for_each_selected_row_simple!(selection, batch_rows, row => {
                                        let rhs = checked_sub_i128(lit, vb[row])?;
                                        acc = acc.wrapping_add(checked_mul_i128_fast(va[row], rhs)?);
                                    });
                                }
                                Decimal128Input::MulColSubLit { lit, .. } => {
                                    for_each_selected_row_simple!(selection, batch_rows, row => {
                                        let rhs = checked_sub_i128(vb[row], lit)?;
                                        acc = acc.wrapping_add(checked_mul_i128_fast(va[row], rhs)?);
                                    });
                                }
                                Decimal128Input::Col(_) => unreachable!(),
                            }
                            saw = true;
                        } else if arr_a.null_count() < arr_a.len()
                            && arr_b.null_count() < arr_b.len()
                        {
                            let na = arr_a.nulls();
                            let nb = arr_b.nulls();
                            for_each_selected_row_simple!(selection, batch_rows, row => {
                                if na.is_none_or(|m| m.is_valid(row))
                                    && nb.is_none_or(|m| m.is_valid(row))
                                {
                                    let val = match *input {
                                        Decimal128Input::MulColCol { .. } => {
                                            checked_mul_i128_fast(va[row], vb[row])?
                                        }
                                        Decimal128Input::MulColAddLit { lit, .. } => {
                                            let rhs = checked_add_i128(lit, vb[row])?;
                                            checked_mul_i128_fast(va[row], rhs)?
                                        }
                                        Decimal128Input::MulColSubFromLit { lit, .. } => {
                                            let rhs = checked_sub_i128(lit, vb[row])?;
                                            checked_mul_i128_fast(va[row], rhs)?
                                        }
                                        Decimal128Input::MulColSubLit { lit, .. } => {
                                            let rhs = checked_sub_i128(vb[row], lit)?;
                                            checked_mul_i128_fast(va[row], rhs)?
                                        }
                                        Decimal128Input::Col(_) => unreachable!(),
                                    };
                                    acc = acc.wrapping_add(val);
                                    saw = true;
                                }
                            });
                        }
                    }
                }
                *sum = acc;
                *has_val = saw;
            }
            Self::SumInt64 {
                direct_col: Some((col, kind)),
                sum,
                has_val,
                ..
            } => {
                let mut acc = *sum;
                let mut saw = *has_val;
                match kind {
                    SumIntInputKind::Int64 => {
                        let arr = batch
                            .column(*col)
                            .as_any()
                            .downcast_ref::<Int64Array>()
                            .ok_or_else(|| {
                                datafusion_common::internal_datafusion_err!(
                                    "Expected Int64 column at {col}"
                                )
                            })?;
                        let vals = arr.values();
                        if arr.null_count() == 0 {
                            for_each_selected_row_simple!(selection, batch_rows, row => {
                                acc = acc.wrapping_add(vals[row]);
                            });
                            saw = true;
                        } else if arr.null_count() < arr.len() {
                            let nulls = arr.nulls().unwrap();
                            for_each_selected_row_simple!(selection, batch_rows, row => {
                                if nulls.is_valid(row) {
                                    acc = acc.wrapping_add(vals[row]);
                                    saw = true;
                                }
                            });
                        }
                    }
                    SumIntInputKind::Int32 => {
                        let arr = batch
                            .column(*col)
                            .as_any()
                            .downcast_ref::<Int32Array>()
                            .ok_or_else(|| {
                                datafusion_common::internal_datafusion_err!(
                                    "Expected Int32 column at {col}"
                                )
                            })?;
                        let vals = arr.values();
                        if arr.null_count() == 0 {
                            for_each_selected_row_simple!(selection, batch_rows, row => {
                                acc = acc.wrapping_add(vals[row] as i64);
                            });
                            saw = true;
                        } else if arr.null_count() < arr.len() {
                            let nulls = arr.nulls().unwrap();
                            for_each_selected_row_simple!(selection, batch_rows, row => {
                                if nulls.is_valid(row) {
                                    acc = acc.wrapping_add(vals[row] as i64);
                                    saw = true;
                                }
                            });
                        }
                    }
                }
                *sum = acc;
                *has_val = saw;
            }
            Self::SumInt64Wide {
                direct_col: Some(col),
                sum,
                has_val,
                ..
            } => {
                let arr = batch
                    .column(*col)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected Int64 column at {col}"
                        )
                    })?;
                let vals = arr.values();
                let mut acc = *sum;
                let mut saw = *has_val;
                if arr.null_count() == 0 {
                    for_each_selected_row_simple!(selection, batch_rows, row => {
                        acc = acc.wrapping_add(vals[row] as i128);
                    });
                    saw = true;
                } else if arr.null_count() < arr.len() {
                    let nulls = arr.nulls().unwrap();
                    for_each_selected_row_simple!(selection, batch_rows, row => {
                        if nulls.is_valid(row) {
                            acc = acc.wrapping_add(vals[row] as i128);
                            saw = true;
                        }
                    });
                }
                *sum = acc;
                *has_val = saw;
            }
            Self::SumFloat64 {
                direct_col: Some(col),
                sum,
                has_val,
                ..
            } => {
                let arr = batch
                    .column(*col)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected Float64 column at {col}"
                        )
                    })?;
                let vals = arr.values();
                let mut acc = *sum;
                let mut saw = *has_val;
                if arr.null_count() == 0 {
                    for_each_selected_row_simple!(selection, batch_rows, row => {
                        acc += vals[row];
                    });
                    saw = true;
                } else if arr.null_count() < arr.len() {
                    let nulls = arr.nulls().unwrap();
                    for_each_selected_row_simple!(selection, batch_rows, row => {
                        if nulls.is_valid(row) {
                            acc += vals[row];
                            saw = true;
                        }
                    });
                }
                *sum = acc;
                *has_val = saw;
            }
            _ => {
                return internal_err!(
                    "update_view_direct called on non-direct ungrouped accumulator"
                );
            }
        }
        Ok(())
    }

    fn update_dense(
        &mut self,
        batch: &RecordBatch,
        rebound: bool,
        args: &mut [ArgEvaluator],
    ) -> Result<()> {
        match self {
            Self::CountStar { count } => {
                let delta =
                    i64::try_from(batch.num_rows()).map_err(|_| count_overflow())?;
                *count = count.checked_add(delta).ok_or_else(count_overflow)?;
            }
            Self::CountCol {
                direct_col,
                rebound_direct_col,
                count,
            } => {
                let col = if rebound {
                    *rebound_direct_col
                } else {
                    *direct_col
                };
                let evaluated;
                let arr = if let Some(c) = col {
                    batch.column(c)
                } else {
                    evaluated = args[0].evaluate(batch)?;
                    &evaluated
                };
                if matches!(arr.data_type(), DataType::Null) {
                    return Ok(());
                }
                let valid = if array_has_no_nulls_fast(arr) {
                    arr.len()
                } else if let Some(nulls) = arr.logical_nulls() {
                    arr.len() - nulls.null_count()
                } else {
                    arr.len()
                };
                let delta = i64::try_from(valid).map_err(|_| count_overflow())?;
                *count = count.checked_add(delta).ok_or_else(count_overflow)?;
            }
            Self::SumDecimal128 {
                fast_input,
                rebound_fast_input,
                sum,
                has_val,
                ..
            } => {
                let input = if rebound {
                    *rebound_fast_input
                } else {
                    *fast_input
                };
                if let Some(fi) = input {
                    let mut tmp = Self::SumDecimal128 {
                        data_type: DataType::Null,
                        fast_input: Some(fi),
                        rebound_fast_input: None,
                        sum: *sum,
                        has_val: *has_val,
                    };
                    tmp.update_view_direct(&BatchView::all(batch.clone()))?;
                    if let Self::SumDecimal128 {
                        sum: s,
                        has_val: hv,
                        ..
                    } = tmp
                    {
                        *sum = s;
                        *has_val = hv;
                    }
                    return Ok(());
                }
                let evaluated = args[0].evaluate(batch)?;
                let arr = evaluated
                    .as_any()
                    .downcast_ref::<Decimal128Array>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected Decimal128 array for SUM"
                        )
                    })?;
                if let Some(x) = arrow::compute::sum(arr) {
                    *sum = sum.wrapping_add(x);
                    *has_val = true;
                }
            }
            Self::SumInt64 {
                direct_col,
                rebound_direct_col,
                sum,
                has_val,
            } => {
                let col = if rebound {
                    *rebound_direct_col
                } else {
                    *direct_col
                };
                if let Some((c, kind)) = col {
                    let mut tmp = Self::SumInt64 {
                        direct_col: Some((c, kind)),
                        rebound_direct_col: None,
                        sum: *sum,
                        has_val: *has_val,
                    };
                    tmp.update_view_direct(&BatchView::all(batch.clone()))?;
                    if let Self::SumInt64 {
                        sum: s,
                        has_val: hv,
                        ..
                    } = tmp
                    {
                        *sum = s;
                        *has_val = hv;
                    }
                    return Ok(());
                }
                let evaluated = args[0].evaluate(batch)?;
                let arr =
                    evaluated
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .ok_or_else(|| {
                            datafusion_common::internal_datafusion_err!(
                                "Expected Int64 array for SUM"
                            )
                        })?;
                if let Some(x) = arrow::compute::sum(arr) {
                    *sum = sum.wrapping_add(x);
                    *has_val = true;
                }
            }
            Self::SumInt64Wide {
                direct_col,
                rebound_direct_col,
                sum,
                has_val,
            } => {
                let col = if rebound {
                    *rebound_direct_col
                } else {
                    *direct_col
                };
                let evaluated;
                let arr_ref = if let Some(c) = col {
                    batch.column(c)
                } else {
                    evaluated = args[0].evaluate(batch)?;
                    &evaluated
                };
                let arr =
                    arr_ref
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .ok_or_else(|| {
                            datafusion_common::internal_datafusion_err!(
                                "Expected Int64 array for wide SUM"
                            )
                        })?;
                let vals = arr.values();
                if arr.null_count() == 0 {
                    if !vals.is_empty() {
                        let mut acc = *sum;
                        for &v in vals.iter() {
                            acc = acc.wrapping_add(v as i128);
                        }
                        *sum = acc;
                        *has_val = true;
                    }
                } else if arr.null_count() < arr.len() {
                    let nulls = arr.nulls().unwrap();
                    let mut acc = *sum;
                    for (i, &v) in vals.iter().enumerate() {
                        if nulls.is_valid(i) {
                            acc = acc.wrapping_add(v as i128);
                            *has_val = true;
                        }
                    }
                    *sum = acc;
                }
            }
            Self::SumFloat64 {
                direct_col,
                rebound_direct_col,
                sum,
                has_val,
            } => {
                let col = if rebound {
                    *rebound_direct_col
                } else {
                    *direct_col
                };
                let evaluated;
                let arr_ref = if let Some(c) = col {
                    batch.column(c)
                } else {
                    evaluated = args[0].evaluate(batch)?;
                    &evaluated
                };
                let arr =
                    arr_ref
                        .as_any()
                        .downcast_ref::<Float64Array>()
                        .ok_or_else(|| {
                            datafusion_common::internal_datafusion_err!(
                                "Expected Float64 array for SUM"
                            )
                        })?;
                if let Some(x) = arrow::compute::sum(arr) {
                    *sum += x;
                    *has_val = true;
                }
            }
            Self::Generic(acc) => {
                let arrays = args
                    .iter_mut()
                    .map(|arg| arg.evaluate(batch))
                    .collect::<Result<Vec<_>>>()?;
                acc.update_batch(&arrays)?;
            }
        }
        Ok(())
    }

    fn merge_from(&mut self, other: &mut Self) -> Result<()> {
        match (self, other) {
            (Self::CountStar { count: dst }, Self::CountStar { count: src })
            | (Self::CountCol { count: dst, .. }, Self::CountCol { count: src, .. }) => {
                *dst = dst.checked_add(*src).ok_or_else(count_overflow)?;
            }
            (
                Self::SumDecimal128 {
                    sum: ds,
                    has_val: dh,
                    ..
                },
                Self::SumDecimal128 {
                    sum: ss,
                    has_val: sh,
                    ..
                },
            )
            | (
                Self::SumInt64Wide {
                    sum: ds,
                    has_val: dh,
                    ..
                },
                Self::SumInt64Wide {
                    sum: ss,
                    has_val: sh,
                    ..
                },
            ) => {
                if *sh {
                    *ds = ds.wrapping_add(*ss);
                    *dh = true;
                }
            }
            (
                Self::SumInt64 {
                    sum: ds,
                    has_val: dh,
                    ..
                },
                Self::SumInt64 {
                    sum: ss,
                    has_val: sh,
                    ..
                },
            ) => {
                if *sh {
                    *ds = ds.wrapping_add(*ss);
                    *dh = true;
                }
            }
            (
                Self::SumFloat64 {
                    sum: ds,
                    has_val: dh,
                    ..
                },
                Self::SumFloat64 {
                    sum: ss,
                    has_val: sh,
                    ..
                },
            ) => {
                if *sh {
                    *ds += *ss;
                    *dh = true;
                }
            }
            (Self::Generic(dst), Self::Generic(src)) => {
                let states: Vec<ArrayRef> = src
                    .state()?
                    .into_iter()
                    .map(|scalar| scalar.to_array_of_size(1))
                    .collect::<Result<_>>()?;
                dst.merge_batch(&states)?;
            }
            _ => return internal_err!("Morsel ungrouped accumulator kind mismatch"),
        }
        Ok(())
    }

    fn evaluate_to_array(&mut self) -> Result<ArrayRef> {
        match self {
            Self::CountStar { count } | Self::CountCol { count, .. } => {
                Ok(Arc::new(Int64Array::from(vec![*count])))
            }
            Self::SumDecimal128 {
                data_type,
                sum,
                has_val,
                ..
            } => {
                let val = has_val.then_some(*sum);
                let (p, s) = match data_type {
                    DataType::Decimal128(p, s) => (*p, *s),
                    _ => return internal_err!("Expected Decimal128 type"),
                };
                Ok(Arc::new(
                    Decimal128Array::from(vec![val]).with_precision_and_scale(p, s)?,
                ))
            }
            Self::SumInt64 { sum, has_val, .. } => {
                Ok(Arc::new(Int64Array::from(vec![has_val.then_some(*sum)])))
            }
            Self::SumInt64Wide { sum, has_val, .. } => Ok(Arc::new(
                Decimal128Array::from(vec![has_val.then_some(*sum)])
                    .with_precision_and_scale(DECIMAL128_MAX_PRECISION, 0)?,
            )),
            Self::SumFloat64 { sum, has_val, .. } => {
                Ok(Arc::new(Float64Array::from(vec![has_val.then_some(*sum)])))
            }
            Self::Generic(acc) => acc.evaluate()?.to_array_of_size(1),
        }
    }
}

pub(super) struct UngroupedState {
    accumulators: Vec<FastUngroupedAccumulator>,
    aggr_args: Vec<Vec<ArgEvaluator>>,
    needed_columns: Vec<usize>,
    rebound_aggr_args: Vec<Vec<ArgEvaluator>>,
    all_direct_view: bool,
    has_rows: bool,
    reservation: MemoryReservation,
}

impl UngroupedState {
    fn tracked_bytes(&self) -> usize {
        self.accumulators
            .iter()
            .map(FastUngroupedAccumulator::size)
            .fold(0usize, usize::saturating_add)
    }

    fn update_reservation(&mut self) -> Result<()> {
        let bytes = self.tracked_bytes();
        if bytes != self.reservation.size() {
            self.reservation.try_resize(bytes)?;
        }
        Ok(())
    }

    fn add_view(&mut self, view: &BatchView) -> Result<()> {
        if self.all_direct_view {
            for acc in &mut self.accumulators {
                acc.update_view_direct(view)?;
            }
            self.has_rows = true;
            return self.update_reservation();
        }
        if view.is_all() {
            return self.add_batch(view.batch());
        }
        let compact = view.materialize_columns(&self.needed_columns)?;
        for (acc, args) in self
            .accumulators
            .iter_mut()
            .zip(&mut self.rebound_aggr_args)
        {
            acc.update_dense(&compact, true, args)?;
        }
        self.has_rows = true;
        self.update_reservation()
    }

    fn add_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        for (acc, args) in self.accumulators.iter_mut().zip(&mut self.aggr_args) {
            acc.update_dense(batch, false, args)?;
        }
        self.has_rows = true;
        self.update_reservation()
    }

    fn merge_from(&mut self, mut other: Self) -> Result<()> {
        if !other.has_rows {
            return Ok(());
        }
        for (dst, src) in self
            .accumulators
            .iter_mut()
            .zip(other.accumulators.iter_mut())
        {
            dst.merge_from(src)?;
        }
        self.has_rows = true;
        drop(other);
        self.update_reservation()
    }

    async fn emit(
        mut self,
        schema: &SchemaRef,
        output_plan: &[OutputAggPlan],
        result_operators: &[Operator],
        tx: &mpsc::Sender<RecordBatch>,
        measurement: Option<&RunMetrics>,
    ) -> Result<()> {
        if tx.is_closed() {
            return Ok(());
        }
        let phys_arrays: Vec<ArrayRef> = self
            .accumulators
            .iter_mut()
            .map(FastUngroupedAccumulator::evaluate_to_array)
            .collect::<Result<_>>()?;
        let columns = evaluate_output_plan(output_plan, &phys_arrays)?;
        let batch = RecordBatch::try_new(Arc::clone(schema), columns)?;
        drop(self);
        let batch = apply_operators(result_operators, batch)?;
        if batch.num_rows() > 0 && !tx.is_closed() {
            if let Some(run) = measurement {
                metrics::send(run, None, tx, batch).await;
            } else {
                let _ = tx.send(batch).await;
            }
        }
        Ok(())
    }
}

enum FastGroupsAccumulator {
    CountStar {
        counts: Vec<i64>,
    },
    CountCol {
        direct_col: Option<usize>,
        rebound_direct_col: Option<usize>,
        count_star_idx: Option<usize>,
        counts: Vec<i64>,
        null_counts: Option<Vec<i64>>,
    },
    SumDecimal128 {
        data_type: DataType,
        fast_input: Option<Decimal128Input>,
        rebound_fast_input: Option<Decimal128Input>,
        sums: Vec<i128>,
        has_val: Option<Vec<bool>>,
    },
    SumInt64 {
        direct_col: Option<(usize, SumIntInputKind)>,
        rebound_direct_col: Option<(usize, SumIntInputKind)>,
        sums: Vec<i64>,
        has_val: Option<Vec<bool>>,
    },
    SumInt64Wide {
        direct_col: Option<usize>,
        rebound_direct_col: Option<usize>,
        sums: Vec<i128>,
        has_val: Option<Vec<bool>>,
    },
    SumFloat64 {
        direct_col: Option<usize>,
        rebound_direct_col: Option<usize>,
        sums: Vec<f64>,
        has_val: Option<Vec<bool>>,
    },
    Generic(Box<dyn GroupsAccumulator>),
}

impl FastGroupsAccumulator {
    fn from_spec(spec: &PhysAggSpec) -> Result<Self> {
        Ok(match &spec.kind {
            PhysAggKind::CountStar => Self::CountStar { counts: Vec::new() },
            PhysAggKind::CountCol {
                direct_col,
                rebound_direct_col,
                count_star_idx,
            } => Self::CountCol {
                direct_col: *direct_col,
                rebound_direct_col: *rebound_direct_col,
                count_star_idx: *count_star_idx,
                counts: Vec::new(),
                null_counts: None,
            },
            PhysAggKind::SumDecimal128 {
                data_type,
                fast_input,
                rebound_fast_input,
            } => Self::SumDecimal128 {
                data_type: data_type.clone(),
                fast_input: *fast_input,
                rebound_fast_input: *rebound_fast_input,
                sums: Vec::new(),
                has_val: None,
            },
            PhysAggKind::SumInt64 {
                direct_col,
                rebound_direct_col,
            } => Self::SumInt64 {
                direct_col: *direct_col,
                rebound_direct_col: *rebound_direct_col,
                sums: Vec::new(),
                has_val: None,
            },
            PhysAggKind::SumInt64Wide {
                direct_col,
                rebound_direct_col,
            } => Self::SumInt64Wide {
                direct_col: *direct_col,
                rebound_direct_col: *rebound_direct_col,
                sums: Vec::new(),
                has_val: None,
            },
            PhysAggKind::SumFloat64 {
                direct_col,
                rebound_direct_col,
            } => Self::SumFloat64 {
                direct_col: *direct_col,
                rebound_direct_col: *rebound_direct_col,
                sums: Vec::new(),
                has_val: None,
            },
            PhysAggKind::Generic(aggr) => {
                Self::Generic(create_groups_accumulator(aggr, false)?)
            }
        })
    }

    fn size(&self) -> usize {
        match self {
            Self::CountStar { counts } => {
                counts.capacity().saturating_mul(size_of::<i64>())
            }
            Self::CountCol {
                counts,
                null_counts,
                ..
            } => counts
                .capacity()
                .saturating_mul(size_of::<i64>())
                .saturating_add(
                    null_counts
                        .as_ref()
                        .map_or(0, |v| v.capacity().saturating_mul(size_of::<i64>())),
                ),
            Self::SumDecimal128 { sums, has_val, .. }
            | Self::SumInt64Wide { sums, has_val, .. } => sums
                .capacity()
                .saturating_mul(size_of::<i128>())
                .saturating_add(has_val.as_ref().map_or(0, Vec::capacity)),
            Self::SumInt64 { sums, has_val, .. } => sums
                .capacity()
                .saturating_mul(size_of::<i64>())
                .saturating_add(has_val.as_ref().map_or(0, Vec::capacity)),
            Self::SumFloat64 { sums, has_val, .. } => sums
                .capacity()
                .saturating_mul(size_of::<f64>())
                .saturating_add(has_val.as_ref().map_or(0, Vec::capacity)),
            Self::Generic(acc) => acc.size(),
        }
    }

    fn update_dense(
        &mut self,
        batch: &RecordBatch,
        rebound: bool,
        args: &mut [ArgEvaluator],
        group_indices: &[usize],
        prev_num_groups: usize,
        total_num_groups: usize,
    ) -> Result<()> {
        match self {
            Self::CountStar { counts } => {
                if counts.len() < total_num_groups {
                    counts.resize(total_num_groups, 0);
                }
                for &gid in group_indices {
                    // SAFETY: `gid < total_num_groups <= counts.len()`
                    let c = unsafe { counts.get_unchecked_mut(gid) };
                    *c = c.checked_add(1).ok_or_else(count_overflow)?;
                }
            }
            Self::CountCol {
                direct_col,
                rebound_direct_col,
                count_star_idx,
                counts,
                null_counts,
            } => {
                let col = if rebound {
                    *rebound_direct_col
                } else {
                    *direct_col
                };
                let evaluated;
                let arr = if let Some(c) = col {
                    batch.column(c)
                } else {
                    evaluated = args[0].evaluate(batch)?;
                    &evaluated
                };
                if count_star_idx.is_some() {
                    if array_has_no_nulls_fast(arr) {
                        if let Some(nc) = null_counts.as_mut()
                            && nc.len() < total_num_groups
                        {
                            nc.resize(total_num_groups, 0);
                        }
                        return Ok(());
                    }
                    if matches!(arr.data_type(), DataType::Null) {
                        let nc = null_counts.get_or_insert_with(Vec::new);
                        if nc.len() < total_num_groups {
                            nc.resize(total_num_groups, 0);
                        }
                        for &gid in group_indices {
                            let c = unsafe { nc.get_unchecked_mut(gid) };
                            *c = c.checked_add(1).ok_or_else(count_overflow)?;
                        }
                        return Ok(());
                    }
                    if let Some(nulls) = arr.logical_nulls()
                        && nulls.null_count() > 0
                    {
                        let nc = null_counts.get_or_insert_with(Vec::new);
                        if nc.len() < total_num_groups {
                            nc.resize(total_num_groups, 0);
                        }
                        if nulls.null_count() == arr.len() {
                            for &gid in group_indices {
                                let c = unsafe { nc.get_unchecked_mut(gid) };
                                *c = c.checked_add(1).ok_or_else(count_overflow)?;
                            }
                        } else {
                            for (row, &gid) in group_indices.iter().enumerate() {
                                if !nulls.is_valid(row) {
                                    let c = unsafe { nc.get_unchecked_mut(gid) };
                                    *c = c.checked_add(1).ok_or_else(count_overflow)?;
                                }
                            }
                        }
                    } else if let Some(nc) = null_counts.as_mut()
                        && nc.len() < total_num_groups
                    {
                        nc.resize(total_num_groups, 0);
                    }
                } else {
                    if counts.len() < total_num_groups {
                        counts.resize(total_num_groups, 0);
                    }
                    if matches!(arr.data_type(), DataType::Null) {
                        return Ok(());
                    }
                    if array_has_no_nulls_fast(arr) {
                        for &gid in group_indices {
                            let c = unsafe { counts.get_unchecked_mut(gid) };
                            *c = c.checked_add(1).ok_or_else(count_overflow)?;
                        }
                    } else if let Some(nulls) = arr.logical_nulls() {
                        if nulls.null_count() == 0 {
                            for &gid in group_indices {
                                let c = unsafe { counts.get_unchecked_mut(gid) };
                                *c = c.checked_add(1).ok_or_else(count_overflow)?;
                            }
                        } else if nulls.null_count() < arr.len() {
                            for (row, &gid) in group_indices.iter().enumerate() {
                                if nulls.is_valid(row) {
                                    let c = unsafe { counts.get_unchecked_mut(gid) };
                                    *c = c.checked_add(1).ok_or_else(count_overflow)?;
                                }
                            }
                        }
                    } else {
                        for &gid in group_indices {
                            let c = unsafe { counts.get_unchecked_mut(gid) };
                            *c = c.checked_add(1).ok_or_else(count_overflow)?;
                        }
                    }
                }
            }
            Self::SumDecimal128 {
                fast_input,
                rebound_fast_input,
                sums,
                has_val,
                ..
            } => {
                if sums.len() < total_num_groups {
                    sums.resize(total_num_groups, 0);
                }
                let input = if rebound {
                    *rebound_fast_input
                } else {
                    *fast_input
                };
                match input {
                    Some(Decimal128Input::Col(c)) => {
                        let arr = batch
                            .column(c)
                            .as_any()
                            .downcast_ref::<Decimal128Array>()
                            .ok_or_else(|| {
                                datafusion_common::internal_datafusion_err!(
                                    "Expected Decimal128 column at {c}"
                                )
                            })?;
                        update_sum_decimal128_array(
                            arr,
                            group_indices,
                            prev_num_groups,
                            total_num_groups,
                            sums,
                            has_val,
                        );
                    }
                    Some(
                        fi @ (Decimal128Input::MulColCol { a, b }
                        | Decimal128Input::MulColAddLit { a, b, .. }
                        | Decimal128Input::MulColSubFromLit { a, b, .. }
                        | Decimal128Input::MulColSubLit { a, b, .. }),
                    ) => {
                        let arr_a = batch
                            .column(a)
                            .as_any()
                            .downcast_ref::<Decimal128Array>()
                            .ok_or_else(|| {
                                datafusion_common::internal_datafusion_err!(
                                    "Expected Decimal128 column at {a}"
                                )
                            })?;
                        let arr_b = batch
                            .column(b)
                            .as_any()
                            .downcast_ref::<Decimal128Array>()
                            .ok_or_else(|| {
                                datafusion_common::internal_datafusion_err!(
                                    "Expected Decimal128 column at {b}"
                                )
                            })?;
                        update_sum_decimal128_fused(
                            fi,
                            arr_a,
                            arr_b,
                            group_indices,
                            prev_num_groups,
                            total_num_groups,
                            sums,
                            has_val,
                        )?;
                    }
                    None => {
                        let evaluated = args[0].evaluate(batch)?;
                        let arr = evaluated
                            .as_any()
                            .downcast_ref::<Decimal128Array>()
                            .ok_or_else(|| {
                                datafusion_common::internal_datafusion_err!(
                                    "Expected Decimal128 array for SUM"
                                )
                            })?;
                        update_sum_decimal128_array(
                            arr,
                            group_indices,
                            prev_num_groups,
                            total_num_groups,
                            sums,
                            has_val,
                        );
                    }
                }
            }
            Self::SumInt64 {
                direct_col,
                rebound_direct_col,
                sums,
                has_val,
            } => {
                if sums.len() < total_num_groups {
                    sums.resize(total_num_groups, 0);
                }
                let col = if rebound {
                    *rebound_direct_col
                } else {
                    *direct_col
                };
                match col {
                    Some((c, SumIntInputKind::Int32)) => {
                        let arr = batch
                            .column(c)
                            .as_any()
                            .downcast_ref::<Int32Array>()
                            .ok_or_else(|| {
                            datafusion_common::internal_datafusion_err!(
                                "Expected Int32 column at {c}"
                            )
                        })?;
                        let vals = arr.values();
                        if arr.null_count() == 0 {
                            if let Some(hv) = has_val.as_mut() {
                                if hv.len() < total_num_groups {
                                    hv.resize(total_num_groups, false);
                                }
                                for (&v, &gid) in vals.iter().zip(group_indices.iter()) {
                                    unsafe {
                                        let s = sums.get_unchecked_mut(gid);
                                        *s = s.wrapping_add(v as i64);
                                        *hv.get_unchecked_mut(gid) = true;
                                    }
                                }
                            } else {
                                for (&v, &gid) in vals.iter().zip(group_indices.iter()) {
                                    unsafe {
                                        let s = sums.get_unchecked_mut(gid);
                                        *s = s.wrapping_add(v as i64);
                                    }
                                }
                            }
                        } else {
                            let hv = ensure_has_val(
                                has_val,
                                prev_num_groups,
                                total_num_groups,
                            );
                            if arr.null_count() < arr.len() {
                                let nulls = arr.nulls().unwrap();
                                for (row, (&v, &gid)) in
                                    vals.iter().zip(group_indices.iter()).enumerate()
                                {
                                    if nulls.is_valid(row) {
                                        unsafe {
                                            let s = sums.get_unchecked_mut(gid);
                                            *s = s.wrapping_add(v as i64);
                                            *hv.get_unchecked_mut(gid) = true;
                                        }
                                    }
                                }
                            }
                        }
                    }
                    other => {
                        let evaluated;
                        let arr_ref = if let Some((c, SumIntInputKind::Int64)) = other {
                            batch.column(c)
                        } else {
                            evaluated = args[0].evaluate(batch)?;
                            &evaluated
                        };
                        let arr = arr_ref
                            .as_any()
                            .downcast_ref::<Int64Array>()
                            .ok_or_else(|| {
                                datafusion_common::internal_datafusion_err!(
                                    "Expected Int64 array for SUM"
                                )
                            })?;
                        update_sum_int64_array(
                            arr,
                            group_indices,
                            prev_num_groups,
                            total_num_groups,
                            sums,
                            has_val,
                        );
                    }
                }
            }
            Self::SumInt64Wide {
                direct_col,
                rebound_direct_col,
                sums,
                has_val,
            } => {
                if sums.len() < total_num_groups {
                    sums.resize(total_num_groups, 0);
                }
                let col = if rebound {
                    *rebound_direct_col
                } else {
                    *direct_col
                };
                let evaluated;
                let arr_ref = if let Some(c) = col {
                    batch.column(c)
                } else {
                    evaluated = args[0].evaluate(batch)?;
                    &evaluated
                };
                let arr =
                    arr_ref
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .ok_or_else(|| {
                            datafusion_common::internal_datafusion_err!(
                                "Expected Int64 array for wide SUM"
                            )
                        })?;
                let vals = arr.values();
                if arr.null_count() == 0 {
                    if let Some(hv) = has_val.as_mut() {
                        if hv.len() < total_num_groups {
                            hv.resize(total_num_groups, false);
                        }
                        for (&v, &gid) in vals.iter().zip(group_indices.iter()) {
                            unsafe {
                                let s = sums.get_unchecked_mut(gid);
                                *s = s.wrapping_add(v as i128);
                                *hv.get_unchecked_mut(gid) = true;
                            }
                        }
                    } else {
                        for (&v, &gid) in vals.iter().zip(group_indices.iter()) {
                            unsafe {
                                let s = sums.get_unchecked_mut(gid);
                                *s = s.wrapping_add(v as i128);
                            }
                        }
                    }
                } else {
                    let hv = ensure_has_val(has_val, prev_num_groups, total_num_groups);
                    if arr.null_count() < arr.len() {
                        let nulls = arr.nulls().unwrap();
                        for (row, (&v, &gid)) in
                            vals.iter().zip(group_indices.iter()).enumerate()
                        {
                            if nulls.is_valid(row) {
                                unsafe {
                                    let s = sums.get_unchecked_mut(gid);
                                    *s = s.wrapping_add(v as i128);
                                    *hv.get_unchecked_mut(gid) = true;
                                }
                            }
                        }
                    }
                }
            }
            Self::SumFloat64 {
                direct_col,
                rebound_direct_col,
                sums,
                has_val,
            } => {
                if sums.len() < total_num_groups {
                    sums.resize(total_num_groups, 0.0);
                }
                let col = if rebound {
                    *rebound_direct_col
                } else {
                    *direct_col
                };
                let evaluated;
                let arr_ref = if let Some(c) = col {
                    batch.column(c)
                } else {
                    evaluated = args[0].evaluate(batch)?;
                    &evaluated
                };
                let arr =
                    arr_ref
                        .as_any()
                        .downcast_ref::<Float64Array>()
                        .ok_or_else(|| {
                            datafusion_common::internal_datafusion_err!(
                                "Expected Float64 array for SUM"
                            )
                        })?;
                update_sum_float64_array(
                    arr,
                    group_indices,
                    prev_num_groups,
                    total_num_groups,
                    sums,
                    has_val,
                );
            }
            Self::Generic(acc) => {
                let arrays = args
                    .iter_mut()
                    .map(|arg| arg.evaluate(batch))
                    .collect::<Result<Vec<_>>>()?;
                acc.update_batch(&arrays, group_indices, None, total_num_groups)?;
            }
        }
        Ok(())
    }

    fn merge_state(
        &mut self,
        states: &[ArrayRef],
        group_indices: &[usize],
        prev_num_groups: usize,
        total_num_groups: usize,
        cs_vals: Option<&[i64]>,
        same_as_cs: bool,
    ) -> Result<()> {
        match self {
            Self::CountStar { counts } => {
                if counts.len() < total_num_groups {
                    counts.resize(total_num_groups, 0);
                }
                let arr =
                    states[0]
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .ok_or_else(|| {
                            datafusion_common::internal_datafusion_err!(
                                "Expected Int64 partial count star column"
                            )
                        })?;
                for (&c, &gid) in arr.values().iter().zip(group_indices.iter()) {
                    let slot = unsafe { counts.get_unchecked_mut(gid) };
                    *slot = slot.checked_add(c).ok_or_else(count_overflow)?;
                }
            }
            Self::CountCol {
                count_star_idx,
                counts,
                null_counts,
                ..
            } => {
                if count_star_idx.is_some() {
                    if same_as_cs {
                        if let Some(nc) = null_counts.as_mut()
                            && nc.len() < total_num_groups
                        {
                            nc.resize(total_num_groups, 0);
                        }
                        return Ok(());
                    }
                    let cs = cs_vals.ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Missing CountStar partial state while merging linked CountCol"
                        )
                    })?;
                    let cc = states[0]
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .ok_or_else(|| {
                            datafusion_common::internal_datafusion_err!(
                                "Expected Int64 partial count column"
                            )
                        })?
                        .values();
                    if let Some(nc) = null_counts.as_mut()
                        && nc.len() < total_num_groups
                    {
                        nc.resize(total_num_groups, 0);
                    }
                    for ((&cs_v, &cc_v), &gid) in
                        cs.iter().zip(cc.iter()).zip(group_indices.iter())
                    {
                        let diff = cs_v - cc_v;
                        if diff != 0 {
                            let nc = null_counts.get_or_insert_with(Vec::new);
                            if nc.len() < total_num_groups {
                                nc.resize(total_num_groups, 0);
                            }
                            let slot = unsafe { nc.get_unchecked_mut(gid) };
                            *slot = slot.checked_add(diff).ok_or_else(count_overflow)?;
                        }
                    }
                } else {
                    if counts.len() < total_num_groups {
                        counts.resize(total_num_groups, 0);
                    }
                    let arr = states[0]
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .ok_or_else(|| {
                            datafusion_common::internal_datafusion_err!(
                                "Expected Int64 partial count column"
                            )
                        })?;
                    for (&c, &gid) in arr.values().iter().zip(group_indices.iter()) {
                        let slot = unsafe { counts.get_unchecked_mut(gid) };
                        *slot = slot.checked_add(c).ok_or_else(count_overflow)?;
                    }
                }
            }
            Self::SumDecimal128 { sums, has_val, .. }
            | Self::SumInt64Wide { sums, has_val, .. } => {
                if sums.len() < total_num_groups {
                    sums.resize(total_num_groups, 0);
                }
                let arr = states[0]
                    .as_any()
                    .downcast_ref::<Decimal128Array>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected Decimal128 partial sum column"
                        )
                    })?;
                update_sum_decimal128_array(
                    arr,
                    group_indices,
                    prev_num_groups,
                    total_num_groups,
                    sums,
                    has_val,
                );
            }
            Self::SumInt64 { sums, has_val, .. } => {
                if sums.len() < total_num_groups {
                    sums.resize(total_num_groups, 0);
                }
                let arr =
                    states[0]
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .ok_or_else(|| {
                            datafusion_common::internal_datafusion_err!(
                                "Expected Int64 partial sum column"
                            )
                        })?;
                update_sum_int64_array(
                    arr,
                    group_indices,
                    prev_num_groups,
                    total_num_groups,
                    sums,
                    has_val,
                );
            }
            Self::SumFloat64 { sums, has_val, .. } => {
                if sums.len() < total_num_groups {
                    sums.resize(total_num_groups, 0.0);
                }
                let arr = states[0]
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected Float64 partial sum column"
                        )
                    })?;
                update_sum_float64_array(
                    arr,
                    group_indices,
                    prev_num_groups,
                    total_num_groups,
                    sums,
                    has_val,
                );
            }
            Self::Generic(acc) => {
                acc.merge_batch(states, group_indices, total_num_groups)?;
            }
        }
        Ok(())
    }

    fn drain_fast_array(&mut self, cs_arr: Option<&Arc<Int64Array>>) -> Result<ArrayRef> {
        match self {
            Self::CountStar { counts } => {
                let vals = std::mem::take(counts);
                Ok(Arc::new(Int64Array::from(vals)))
            }
            Self::CountCol {
                count_star_idx,
                counts,
                null_counts,
                ..
            } => {
                if count_star_idx.is_some() {
                    let cs = cs_arr.ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Missing CountStar array when draining linked CountCol"
                        )
                    })?;
                    let Some(nc) = null_counts.take() else {
                        return Ok(Arc::clone(cs) as ArrayRef);
                    };
                    let cs_vals = cs.values();
                    let out: Vec<i64> = cs_vals
                        .iter()
                        .enumerate()
                        .map(|(i, &total)| total - nc.get(i).copied().unwrap_or(0))
                        .collect();
                    Ok(Arc::new(Int64Array::from(out)))
                } else {
                    let vals = std::mem::take(counts);
                    Ok(Arc::new(Int64Array::from(vals)))
                }
            }
            Self::SumDecimal128 {
                data_type,
                sums,
                has_val,
                ..
            } => {
                let vals = std::mem::take(sums);
                let nulls = build_null_buffer_from_has_val(has_val.take());
                let (p, s) = match data_type {
                    DataType::Decimal128(p, s) => (*p, *s),
                    _ => return internal_err!("Expected Decimal128 type"),
                };
                Ok(Arc::new(
                    Decimal128Array::new(vals.into(), nulls)
                        .with_precision_and_scale(p, s)?,
                ))
            }
            Self::SumInt64 { sums, has_val, .. } => {
                let vals = std::mem::take(sums);
                let nulls = build_null_buffer_from_has_val(has_val.take());
                Ok(Arc::new(Int64Array::new(vals.into(), nulls)))
            }
            Self::SumInt64Wide { sums, has_val, .. } => {
                let vals = std::mem::take(sums);
                let nulls = build_null_buffer_from_has_val(has_val.take());
                Ok(Arc::new(
                    Decimal128Array::new(vals.into(), nulls)
                        .with_precision_and_scale(DECIMAL128_MAX_PRECISION, 0)?,
                ))
            }
            Self::SumFloat64 { sums, has_val, .. } => {
                let vals = std::mem::take(sums);
                let nulls = build_null_buffer_from_has_val(has_val.take());
                Ok(Arc::new(Float64Array::new(vals.into(), nulls)))
            }
            Self::Generic(_) => unreachable!("handled by caller"),
        }
    }
}

#[inline]
fn update_sum_decimal128_array(
    arr: &Decimal128Array,
    group_indices: &[usize],
    prev_num_groups: usize,
    total_num_groups: usize,
    sums: &mut [i128],
    has_val: &mut Option<Vec<bool>>,
) {
    let vals = arr.values();
    if arr.null_count() == 0 {
        if let Some(hv) = has_val.as_mut() {
            if hv.len() < total_num_groups {
                hv.resize(total_num_groups, false);
            }
            for (&v, &gid) in vals.iter().zip(group_indices.iter()) {
                unsafe {
                    let s = sums.get_unchecked_mut(gid);
                    *s = s.wrapping_add(v);
                    *hv.get_unchecked_mut(gid) = true;
                }
            }
        } else {
            for (&v, &gid) in vals.iter().zip(group_indices.iter()) {
                unsafe {
                    let s = sums.get_unchecked_mut(gid);
                    *s = s.wrapping_add(v);
                }
            }
        }
    } else {
        let hv = ensure_has_val(has_val, prev_num_groups, total_num_groups);
        if arr.null_count() < arr.len() {
            let nulls = arr.nulls().unwrap();
            for (row, (&v, &gid)) in vals.iter().zip(group_indices.iter()).enumerate() {
                if nulls.is_valid(row) {
                    unsafe {
                        let s = sums.get_unchecked_mut(gid);
                        *s = s.wrapping_add(v);
                        *hv.get_unchecked_mut(gid) = true;
                    }
                }
            }
        }
    }
}

#[expect(clippy::too_many_arguments)]
fn update_sum_decimal128_fused(
    input: Decimal128Input,
    arr_a: &Decimal128Array,
    arr_b: &Decimal128Array,
    group_indices: &[usize],
    prev_num_groups: usize,
    total_num_groups: usize,
    sums: &mut [i128],
    has_val: &mut Option<Vec<bool>>,
) -> Result<()> {
    let va = arr_a.values();
    let vb = arr_b.values();
    if arr_a.null_count() == 0 && arr_b.null_count() == 0 {
        if let Some(hv) = has_val.as_mut() {
            if hv.len() < total_num_groups {
                hv.resize(total_num_groups, false);
            }
            match input {
                Decimal128Input::MulColCol { .. } => {
                    for ((&a_v, &b_v), &gid) in
                        va.iter().zip(vb.iter()).zip(group_indices.iter())
                    {
                        let val = checked_mul_i128_fast(a_v, b_v)?;
                        unsafe {
                            let s = sums.get_unchecked_mut(gid);
                            *s = s.wrapping_add(val);
                            *hv.get_unchecked_mut(gid) = true;
                        }
                    }
                }
                Decimal128Input::MulColAddLit { lit, .. } => {
                    for ((&a_v, &b_v), &gid) in
                        va.iter().zip(vb.iter()).zip(group_indices.iter())
                    {
                        let rhs = checked_add_i128(lit, b_v)?;
                        let val = checked_mul_i128_fast(a_v, rhs)?;
                        unsafe {
                            let s = sums.get_unchecked_mut(gid);
                            *s = s.wrapping_add(val);
                            *hv.get_unchecked_mut(gid) = true;
                        }
                    }
                }
                Decimal128Input::MulColSubFromLit { lit, .. } => {
                    for ((&a_v, &b_v), &gid) in
                        va.iter().zip(vb.iter()).zip(group_indices.iter())
                    {
                        let rhs = checked_sub_i128(lit, b_v)?;
                        let val = checked_mul_i128_fast(a_v, rhs)?;
                        unsafe {
                            let s = sums.get_unchecked_mut(gid);
                            *s = s.wrapping_add(val);
                            *hv.get_unchecked_mut(gid) = true;
                        }
                    }
                }
                Decimal128Input::MulColSubLit { lit, .. } => {
                    for ((&a_v, &b_v), &gid) in
                        va.iter().zip(vb.iter()).zip(group_indices.iter())
                    {
                        let rhs = checked_sub_i128(b_v, lit)?;
                        let val = checked_mul_i128_fast(a_v, rhs)?;
                        unsafe {
                            let s = sums.get_unchecked_mut(gid);
                            *s = s.wrapping_add(val);
                            *hv.get_unchecked_mut(gid) = true;
                        }
                    }
                }
                Decimal128Input::Col(_) => unreachable!(),
            }
        } else {
            match input {
                Decimal128Input::MulColCol { .. } => {
                    for ((&a_v, &b_v), &gid) in
                        va.iter().zip(vb.iter()).zip(group_indices.iter())
                    {
                        let val = checked_mul_i128_fast(a_v, b_v)?;
                        unsafe {
                            let s = sums.get_unchecked_mut(gid);
                            *s = s.wrapping_add(val);
                        }
                    }
                }
                Decimal128Input::MulColAddLit { lit, .. } => {
                    for ((&a_v, &b_v), &gid) in
                        va.iter().zip(vb.iter()).zip(group_indices.iter())
                    {
                        let rhs = checked_add_i128(lit, b_v)?;
                        let val = checked_mul_i128_fast(a_v, rhs)?;
                        unsafe {
                            let s = sums.get_unchecked_mut(gid);
                            *s = s.wrapping_add(val);
                        }
                    }
                }
                Decimal128Input::MulColSubFromLit { lit, .. } => {
                    for ((&a_v, &b_v), &gid) in
                        va.iter().zip(vb.iter()).zip(group_indices.iter())
                    {
                        let rhs = checked_sub_i128(lit, b_v)?;
                        let val = checked_mul_i128_fast(a_v, rhs)?;
                        unsafe {
                            let s = sums.get_unchecked_mut(gid);
                            *s = s.wrapping_add(val);
                        }
                    }
                }
                Decimal128Input::MulColSubLit { lit, .. } => {
                    for ((&a_v, &b_v), &gid) in
                        va.iter().zip(vb.iter()).zip(group_indices.iter())
                    {
                        let rhs = checked_sub_i128(b_v, lit)?;
                        let val = checked_mul_i128_fast(a_v, rhs)?;
                        unsafe {
                            let s = sums.get_unchecked_mut(gid);
                            *s = s.wrapping_add(val);
                        }
                    }
                }
                Decimal128Input::Col(_) => unreachable!(),
            }
        }
    } else {
        let hv = ensure_has_val(has_val, prev_num_groups, total_num_groups);
        if arr_a.null_count() < arr_a.len() && arr_b.null_count() < arr_b.len() {
            let na = arr_a.nulls();
            let nb = arr_b.nulls();
            for (row, &gid) in group_indices.iter().enumerate() {
                if na.is_none_or(|m| m.is_valid(row))
                    && nb.is_none_or(|m| m.is_valid(row))
                {
                    let val = match input {
                        Decimal128Input::MulColCol { .. } => {
                            checked_mul_i128_fast(va[row], vb[row])?
                        }
                        Decimal128Input::MulColAddLit { lit, .. } => {
                            let rhs = checked_add_i128(lit, vb[row])?;
                            checked_mul_i128_fast(va[row], rhs)?
                        }
                        Decimal128Input::MulColSubFromLit { lit, .. } => {
                            let rhs = checked_sub_i128(lit, vb[row])?;
                            checked_mul_i128_fast(va[row], rhs)?
                        }
                        Decimal128Input::MulColSubLit { lit, .. } => {
                            let rhs = checked_sub_i128(vb[row], lit)?;
                            checked_mul_i128_fast(va[row], rhs)?
                        }
                        Decimal128Input::Col(_) => unreachable!(),
                    };
                    unsafe {
                        let s = sums.get_unchecked_mut(gid);
                        *s = s.wrapping_add(val);
                        *hv.get_unchecked_mut(gid) = true;
                    }
                }
            }
        }
    }
    Ok(())
}

#[inline]
fn update_sum_int64_array(
    arr: &Int64Array,
    group_indices: &[usize],
    prev_num_groups: usize,
    total_num_groups: usize,
    sums: &mut [i64],
    has_val: &mut Option<Vec<bool>>,
) {
    let vals = arr.values();
    if arr.null_count() == 0 {
        if let Some(hv) = has_val.as_mut() {
            if hv.len() < total_num_groups {
                hv.resize(total_num_groups, false);
            }
            for (&v, &gid) in vals.iter().zip(group_indices.iter()) {
                unsafe {
                    let s = sums.get_unchecked_mut(gid);
                    *s = s.wrapping_add(v);
                    *hv.get_unchecked_mut(gid) = true;
                }
            }
        } else {
            for (&v, &gid) in vals.iter().zip(group_indices.iter()) {
                unsafe {
                    let s = sums.get_unchecked_mut(gid);
                    *s = s.wrapping_add(v);
                }
            }
        }
    } else {
        let hv = ensure_has_val(has_val, prev_num_groups, total_num_groups);
        if arr.null_count() < arr.len() {
            let nulls = arr.nulls().unwrap();
            for (row, (&v, &gid)) in vals.iter().zip(group_indices.iter()).enumerate() {
                if nulls.is_valid(row) {
                    unsafe {
                        let s = sums.get_unchecked_mut(gid);
                        *s = s.wrapping_add(v);
                        *hv.get_unchecked_mut(gid) = true;
                    }
                }
            }
        }
    }
}

#[inline]
fn update_sum_float64_array(
    arr: &Float64Array,
    group_indices: &[usize],
    prev_num_groups: usize,
    total_num_groups: usize,
    sums: &mut [f64],
    has_val: &mut Option<Vec<bool>>,
) {
    let vals = arr.values();
    if arr.null_count() == 0 {
        if let Some(hv) = has_val.as_mut() {
            if hv.len() < total_num_groups {
                hv.resize(total_num_groups, false);
            }
            for (&v, &gid) in vals.iter().zip(group_indices.iter()) {
                unsafe {
                    let s = sums.get_unchecked_mut(gid);
                    *s += v;
                    *hv.get_unchecked_mut(gid) = true;
                }
            }
        } else {
            for (&v, &gid) in vals.iter().zip(group_indices.iter()) {
                unsafe {
                    let s = sums.get_unchecked_mut(gid);
                    *s += v;
                }
            }
        }
    } else {
        let hv = ensure_has_val(has_val, prev_num_groups, total_num_groups);
        if arr.null_count() < arr.len() {
            let nulls = arr.nulls().unwrap();
            for (row, (&v, &gid)) in vals.iter().zip(group_indices.iter()).enumerate() {
                if nulls.is_valid(row) {
                    unsafe {
                        let s = sums.get_unchecked_mut(gid);
                        *s += v;
                        *hv.get_unchecked_mut(gid) = true;
                    }
                }
            }
        }
    }
}

fn drain_fast_groups_state(
    accumulators: &mut [FastGroupsAccumulator],
    count_star_idx: Option<usize>,
) -> Result<Vec<Vec<ArrayRef>>> {
    let cs_arr: Option<Arc<Int64Array>> = if let Some(cs_idx) = count_star_idx {
        let FastGroupsAccumulator::CountStar { counts } = &mut accumulators[cs_idx]
        else {
            return internal_err!("Expected CountStar at count_star_idx");
        };
        Some(Arc::new(Int64Array::from(std::mem::take(counts))))
    } else {
        None
    };

    let mut out = Vec::with_capacity(accumulators.len());
    for (idx, acc) in accumulators.iter_mut().enumerate() {
        if Some(idx) == count_star_idx {
            out.push(vec![Arc::clone(cs_arr.as_ref().unwrap()) as ArrayRef]);
        } else if let FastGroupsAccumulator::Generic(g) = acc {
            out.push(g.state(EmitTo::All)?);
        } else {
            out.push(vec![acc.drain_fast_array(cs_arr.as_ref())?]);
        }
    }
    Ok(out)
}

fn evaluate_fast_groups(
    accumulators: &mut [FastGroupsAccumulator],
    count_star_idx: Option<usize>,
    output_plan: &[OutputAggPlan],
) -> Result<Vec<ArrayRef>> {
    let cs_arr: Option<Arc<Int64Array>> = if let Some(cs_idx) = count_star_idx {
        let FastGroupsAccumulator::CountStar { counts } = &mut accumulators[cs_idx]
        else {
            return internal_err!("Expected CountStar at count_star_idx");
        };
        Some(Arc::new(Int64Array::from(std::mem::take(counts))))
    } else {
        None
    };

    let mut phys_arrays = Vec::with_capacity(accumulators.len());
    for (idx, acc) in accumulators.iter_mut().enumerate() {
        if Some(idx) == count_star_idx {
            phys_arrays.push(Arc::clone(cs_arr.as_ref().unwrap()) as ArrayRef);
        } else if let FastGroupsAccumulator::Generic(g) = acc {
            phys_arrays.push(g.evaluate(EmitTo::All)?);
        } else {
            phys_arrays.push(acc.drain_fast_array(cs_arr.as_ref())?);
        }
    }
    evaluate_output_plan(output_plan, &phys_arrays)
}

/// Sentinel representing `NULL` in an inlined `Utf8View` (`len <= 12`).
/// No valid inlined view can have `(v as u32) == u32::MAX`.
const NULL_INLINE_VIEW: u128 = u128::MAX;
const TWO_UTF8VIEW_CACHE_SIZE: usize = 64;

struct InlineTwoUtf8ViewGroupValues {
    group_schema: SchemaRef,
    map: HashMap<(u128, u128), usize>,
    keys0: Vec<u128>,
    keys1: Vec<u128>,
    cache: [(u128, u128, u32); TWO_UTF8VIEW_CACHE_SIZE],
}

impl InlineTwoUtf8ViewGroupValues {
    fn new(group_schema: SchemaRef) -> Self {
        Self {
            group_schema,
            map: HashMap::new(),
            keys0: Vec::new(),
            keys1: Vec::new(),
            cache: [(0, 0, 0); TWO_UTF8VIEW_CACHE_SIZE],
        }
    }

    #[inline(always)]
    fn cache_slot(k0: u128, k1: u128) -> usize {
        let h0 = ((k0 as u64) ^ ((k0 >> 64) as u64)).wrapping_mul(0x9e37_79b9_7f4a_7c15);
        let h1 = ((k1 as u64) ^ ((k1 >> 64) as u64)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
        ((h0 ^ h1) >> 58) as usize
    }

    #[inline(always)]
    fn intern_pair(&mut self, k0: u128, k1: u128) -> Option<usize> {
        let slot = Self::cache_slot(k0, k1);
        // SAFETY: `slot < 64 == TWO_UTF8VIEW_CACHE_SIZE`.
        // Only inlined views (`len <= 12`) or `NULL_INLINE_VIEW` are ever stored
        // in `cache`, so a cache hit guarantees both inputs are inlined.
        let entry = unsafe { self.cache.get_unchecked(slot) };
        if entry.2 != 0 && entry.0 == k0 && entry.1 == k1 {
            return Some((entry.2 - 1) as usize);
        }
        self.intern_pair_slow(k0, k1, slot)
    }

    #[cold]
    fn intern_pair_slow(&mut self, k0: u128, k1: u128, slot: usize) -> Option<usize> {
        if (k0 != NULL_INLINE_VIEW && (k0 as u32) > 12)
            || (k1 != NULL_INLINE_VIEW && (k1 as u32) > 12)
        {
            return None;
        }
        let gid = if let Some(&existing) = self.map.get(&(k0, k1)) {
            existing
        } else {
            let gid = self.keys0.len();
            self.map.insert((k0, k1), gid);
            self.keys0.push(k0);
            self.keys1.push(k1);
            gid
        };
        if gid < u32::MAX as usize {
            self.cache[slot] = (k0, k1, (gid + 1) as u32);
        }
        Some(gid)
    }

    fn emit_inlined_arrays(&mut self) -> Result<Vec<ArrayRef>> {
        let build_col = |mut views: Vec<u128>| -> Result<ArrayRef> {
            let has_null = views.contains(&NULL_INLINE_VIEW);
            let nulls = if has_null {
                let mut builder = NullBufferBuilder::new(views.len());
                for v in &mut views {
                    if *v == NULL_INLINE_VIEW {
                        *v = 0;
                        builder.append_null();
                    } else {
                        builder.append_non_null();
                    }
                }
                builder.finish()
            } else {
                None
            };
            Ok(Arc::new(StringViewArray::try_new(
                views.into(),
                vec![],
                nulls,
            )?))
        };
        let k0 = std::mem::take(&mut self.keys0);
        let k1 = std::mem::take(&mut self.keys1);
        self.map.clear();
        self.cache = [(0, 0, 0); TWO_UTF8VIEW_CACHE_SIZE];
        Ok(vec![build_col(k0)?, build_col(k1)?])
    }

    fn size(&self) -> usize {
        self.map
            .allocation_size()
            .saturating_add(
                self.keys0
                    .capacity()
                    .saturating_add(self.keys1.capacity())
                    .saturating_mul(size_of::<u128>()),
            )
            .saturating_add(size_of_val(&self.cache))
    }
}

enum FastGroupValues {
    TwoUtf8View(Box<InlineTwoUtf8ViewGroupValues>),
    Generic(Box<dyn GroupValues>),
}

impl FastGroupValues {
    fn new(group_schema: SchemaRef) -> Result<Self> {
        if group_schema.fields().len() == 2
            && group_schema.field(0).data_type() == &DataType::Utf8View
            && group_schema.field(1).data_type() == &DataType::Utf8View
        {
            return Ok(Self::TwoUtf8View(Box::new(
                InlineTwoUtf8ViewGroupValues::new(group_schema),
            )));
        }
        Ok(Self::Generic(new_group_values(
            group_schema,
            &GroupClustering::None,
        )?))
    }

    fn len(&self) -> usize {
        match self {
            Self::TwoUtf8View(gv) => gv.keys0.len(),
            Self::Generic(gv) => gv.len(),
        }
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn size(&self) -> usize {
        match self {
            Self::TwoUtf8View(gv) => gv.size(),
            Self::Generic(gv) => gv.size(),
        }
    }

    fn emit(&mut self, emit_to: EmitTo) -> Result<Vec<ArrayRef>> {
        match self {
            Self::TwoUtf8View(gv) => {
                debug_assert!(matches!(emit_to, EmitTo::All));
                gv.emit_inlined_arrays()
            }
            Self::Generic(gv) => gv.emit(emit_to),
        }
    }

    fn intern(&mut self, cols: &[ArrayRef], groups: &mut Vec<usize>) -> Result<()> {
        match self {
            Self::TwoUtf8View(gv) => {
                let a0 = cols[0]
                    .as_any()
                    .downcast_ref::<StringViewArray>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected StringViewArray for group key 0"
                        )
                    })?;
                let a1 = cols[1]
                    .as_any()
                    .downcast_ref::<StringViewArray>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected StringViewArray for group key 1"
                        )
                    })?;
                let v0 = a0.views();
                let v1 = a1.views();
                let n = v0.len();
                groups.clear();
                groups.try_reserve(n).map_err(|e| {
                    datafusion_common::exec_datafusion_err!(
                        "Morsel TwoUtf8View index buffer allocation failed: {e}"
                    )
                })?;
                let mut saw_non_inlined = false;
                if a0.null_count() == 0 && a1.null_count() == 0 {
                    for i in 0..n {
                        if let Some(gid) = gv.intern_pair(v0[i], v1[i]) {
                            groups.push(gid);
                        } else {
                            saw_non_inlined = true;
                            break;
                        }
                    }
                } else {
                    let n0 = a0.nulls();
                    let n1 = a1.nulls();
                    for i in 0..n {
                        let k0 = if n0.is_none_or(|m| m.is_valid(i)) {
                            v0[i]
                        } else {
                            NULL_INLINE_VIEW
                        };
                        let k1 = if n1.is_none_or(|m| m.is_valid(i)) {
                            v1[i]
                        } else {
                            NULL_INLINE_VIEW
                        };
                        if let Some(gid) = gv.intern_pair(k0, k1) {
                            groups.push(gid);
                        } else {
                            saw_non_inlined = true;
                            break;
                        }
                    }
                }
                if saw_non_inlined {
                    // Non-inlined string (> 12 bytes) encountered: seamlessly
                    // promote to generic `GroupValues` while preserving all
                    // existing group IDs `0..K`, then re-intern this batch.
                    let mut generic = new_group_values(
                        Arc::clone(&gv.group_schema),
                        &GroupClustering::None,
                    )?;
                    if !gv.keys0.is_empty() {
                        let existing_cols = gv.emit_inlined_arrays()?;
                        let mut tmp = Vec::new();
                        generic.intern(&existing_cols, &mut tmp)?;
                        debug_assert!(tmp.iter().enumerate().all(|(i, &g)| i == g));
                    }
                    *self = Self::Generic(generic);
                    return self.intern(cols, groups);
                }
                Ok(())
            }
            Self::Generic(gv) => gv.intern(cols, groups),
        }
    }
}

pub(super) struct GroupedState {
    group_values: FastGroupValues,
    group_schema: SchemaRef,
    group_exprs: Vec<ArgEvaluator>,
    rebound_group_exprs: Vec<ArgEvaluator>,
    accumulators: Vec<FastGroupsAccumulator>,
    count_star_idx: Option<usize>,
    aggr_args: Vec<Vec<ArgEvaluator>>,
    rebound_aggr_args: Vec<Vec<ArgEvaluator>>,
    needed_columns: Vec<usize>,
    group_indices: Vec<usize>,
    partial_schema: SchemaRef,
    has_float_keys: bool,
    table_tracked_bytes: usize,
    radix: Option<RadixPartitionStore>,
    spill: Option<SpillContext>,
    hash_buf: Vec<u64>,
    reservation: MemoryReservation,
}

impl GroupedState {
    fn tracked_bytes(&self) -> usize {
        let gv = self.group_values.size();
        let accs = self
            .accumulators
            .iter()
            .map(FastGroupsAccumulator::size)
            .fold(0usize, usize::saturating_add);
        let indices = self
            .group_indices
            .capacity()
            .saturating_mul(size_of::<usize>());
        gv.saturating_add(accs).saturating_add(indices)
    }

    fn update_reservation(&mut self) -> Result<()> {
        let bytes = self.tracked_bytes();
        if bytes > self.table_tracked_bytes {
            self.reservation
                .try_grow(bytes - self.table_tracked_bytes)?;
        } else if bytes < self.table_tracked_bytes {
            self.reservation.shrink(self.table_tracked_bytes - bytes);
        }
        self.table_tracked_bytes = bytes;
        Ok(())
    }

    fn sync_reservation_unconditional(&mut self) {
        let bytes = self.tracked_bytes();
        if bytes > self.table_tracked_bytes {
            self.reservation.grow(bytes - self.table_tracked_bytes);
        } else if bytes < self.table_tracked_bytes {
            self.reservation.shrink(self.table_tracked_bytes - bytes);
        }
        self.table_tracked_bytes = bytes;
    }

    fn estimate_batch_delta(&self, num_rows: usize, key_arrays: &[ArrayRef]) -> usize {
        let key_bytes = arrays_memory_size(key_arrays);
        let per_row = 24usize.saturating_add(self.accumulators.len().saturating_mul(24));
        key_bytes.saturating_add(num_rows.saturating_mul(per_row))
    }

    fn check_batch_headroom(&self, est_bytes: usize) -> bool {
        if est_bytes == 0 {
            return true;
        }
        let probe = self.reservation.new_empty();
        probe.try_grow(est_bytes).is_ok()
    }

    fn flush_to_radix(&mut self, tx: &mpsc::Sender<RecordBatch>) -> Result<()> {
        if self.group_values.is_empty() {
            return Ok(());
        }
        let group_arrays = self.group_values.emit(EmitTo::All)?;
        // Reset `group_values` cleanly after `EmitTo::All`: `GroupValuesColumn`
        // (multi-column GROUP BY) replaces its value vectors without clearing
        // its internal hash table unless rebuilt/cleared.
        self.group_values = FastGroupValues::new(Arc::clone(&self.group_schema))?;
        hash_group_columns(&group_arrays, self.has_float_keys, &mut self.hash_buf)?;
        let mut cols = group_arrays;
        for states in
            drain_fast_groups_state(&mut self.accumulators, self.count_star_idx)?
        {
            cols.extend(states);
        }
        let batch = RecordBatch::try_new(Arc::clone(&self.partial_schema), cols)?;
        self.sync_reservation_unconditional();

        let radix = self
            .radix
            .get_or_insert_with(RadixPartitionStore::new_default);
        radix.push_partial_batch(
            batch,
            &self.hash_buf,
            self.table_tracked_bytes,
            &self.reservation,
            self.spill.as_ref(),
            tx,
        )
    }

    fn spill_on_oom(&mut self, tx: &mpsc::Sender<RecordBatch>) -> Result<()> {
        let Some(spill_ctx) = self.spill.clone() else {
            return Err(DataFusionError::ResourcesExhausted(
                "GroupedAggregate spilling is disabled".into(),
            ));
        };
        spill_ctx.mark_spilled();
        if !self.group_values.is_empty() {
            self.flush_to_radix(tx)?;
        }
        let radix = self
            .radix
            .get_or_insert_with(RadixPartitionStore::new_default);
        radix.spill_all_mem(
            self.table_tracked_bytes,
            &self.reservation,
            &spill_ctx,
            tx,
        )?;
        Ok(())
    }

    fn maybe_flush_preagg(&mut self, tx: &mpsc::Sender<RecordBatch>) -> Result<()> {
        let len = self.group_values.len();
        let should_flush = if self.radix.is_some() {
            len >= PREAGG_FLUSH_GROUPS || self.table_tracked_bytes >= PREAGG_FLUSH_BYTES
        } else {
            len >= RADIX_TRANSITION_GROUPS
        };
        if should_flush {
            self.flush_to_radix(tx)?;
        }
        Ok(())
    }

    fn eval_group_exprs(
        &mut self,
        batch: &RecordBatch,
        rebound: bool,
    ) -> Result<Vec<ArrayRef>> {
        if rebound {
            self.rebound_group_exprs
                .iter_mut()
                .map(|expr| expr.evaluate(batch))
                .collect()
        } else {
            self.group_exprs
                .iter_mut()
                .map(|expr| expr.evaluate(batch))
                .collect()
        }
    }

    fn ingest_dense_with_group_arrays(
        &mut self,
        batch: &RecordBatch,
        rebound: bool,
        group_arrays: &[ArrayRef],
    ) -> Result<()> {
        let prev_num_groups = self.group_values.len();
        self.group_values
            .intern(group_arrays, &mut self.group_indices)?;
        let total_num_groups = self.group_values.len();

        let args_iter = if rebound {
            &mut self.rebound_aggr_args
        } else {
            &mut self.aggr_args
        };
        for (acc, args) in self.accumulators.iter_mut().zip(args_iter.iter_mut()) {
            acc.update_dense(
                batch,
                rebound,
                args,
                &self.group_indices,
                prev_num_groups,
                total_num_groups,
            )?;
        }
        Ok(())
    }

    #[cfg(test)]
    fn ingest_dense_unchecked(
        &mut self,
        batch: &RecordBatch,
        rebound: bool,
    ) -> Result<()> {
        let group_arrays = self.eval_group_exprs(batch, rebound)?;
        self.ingest_dense_with_group_arrays(batch, rebound, &group_arrays)
    }

    #[cfg(test)]
    fn add_view(&mut self, view: &BatchView) -> Result<()> {
        if view.is_all() {
            return self.add_batch(view.batch());
        }
        let compact = view.materialize_columns(&self.needed_columns)?;
        self.ingest_dense_unchecked(&compact, true)?;
        self.update_reservation()
    }

    #[cfg(test)]
    fn add_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        self.ingest_dense_unchecked(batch, false)?;
        self.update_reservation()
    }

    fn add_view_spillable(
        &mut self,
        view: &BatchView,
        tx: &mpsc::Sender<RecordBatch>,
    ) -> Result<()> {
        let (batch, rebound) = if view.is_all() {
            (view.batch().clone(), false)
        } else {
            (view.materialize_columns(&self.needed_columns)?, true)
        };
        if batch.num_rows() == 0 {
            return Ok(());
        }

        let group_arrays = self.eval_group_exprs(&batch, rebound)?;

        // If spilling is enabled and we already hold groups, pre-check whether
        // this batch might exceed the pool; flush consistent state first so we
        // stay well within budget.
        if self.spill.is_some() && !self.group_values.is_empty() {
            let est = self.estimate_batch_delta(batch.num_rows(), &group_arrays);
            if !self.check_batch_headroom(est) {
                self.spill_on_oom(tx)?;
            }
        }

        self.ingest_dense_with_group_arrays(&batch, rebound, &group_arrays)?;
        if let Err(err) = self.update_reservation() {
            if matches!(err, DataFusionError::ResourcesExhausted(_))
                && self.spill.is_some()
            {
                // Both `group_values` and `accumulators` already include `batch`
                // and are 100% consistent: spill them cleanly now and verify
                // minimum headroom.
                self.spill_on_oom(tx)?;
                return Ok(());
            }
            return Err(err);
        }
        self.maybe_flush_preagg(tx)
    }

    async fn merge_from(
        &mut self,
        mut other: Self,
        tx: &mpsc::Sender<RecordBatch>,
    ) -> Result<bool> {
        if self.radix.is_some()
            || other.radix.is_some()
            || self
                .group_values
                .len()
                .saturating_add(other.group_values.len())
                >= RADIX_TRANSITION_GROUPS
        {
            self.flush_to_radix(tx)?;
            other.flush_to_radix(tx)?;
            if tx.is_closed() {
                return Ok(false);
            }
            if let Some(other_radix) = other.radix.take() {
                let radix = self
                    .radix
                    .get_or_insert_with(RadixPartitionStore::new_default);
                let transferred = other.reservation.free();
                self.reservation.grow(transferred);
                radix.combine_from(other_radix);
            }
            return Ok(true);
        }

        if other.group_values.is_empty() {
            return Ok(true);
        }
        if self.group_values.is_empty() {
            std::mem::swap(self, &mut other);
            return Ok(true);
        }
        if other.group_values.len() > self.group_values.len() {
            std::mem::swap(self, &mut other);
        }

        // Emit all groups from `other` once in O(N) instead of repeated O(N)
        // `EmitTo::First` shifts, then merge in zero-copy slices so we stay
        // cooperative on cancellation and track peak merge memory accurately.
        let total_other = other.group_values.len();
        let group_arrays = other.group_values.emit(EmitTo::All)?;
        let state_arrays =
            drain_fast_groups_state(&mut other.accumulators, other.count_star_idx)?;
        let emitted_bytes = arrays_memory_size(&group_arrays).saturating_add(
            state_arrays
                .iter()
                .map(|states| arrays_memory_size(states))
                .fold(0usize, usize::saturating_add),
        );
        let emitted_reservation = other.reservation.new_empty();
        drop(other);
        emitted_reservation.try_grow(emitted_bytes)?;

        let cs_full_arr = self
            .count_star_idx
            .map(|cs_idx| Arc::clone(&state_arrays[cs_idx][0]));
        let cs_full_vals = cs_full_arr.as_ref().map(|a| {
            a.as_any()
                .downcast_ref::<Int64Array>()
                .expect("CountStar partial array must be Int64")
                .values()
        });

        let mut offset = 0usize;
        while offset < total_other {
            if tx.is_closed() {
                return Ok(false);
            }
            let len = (total_other - offset).min(MERGE_CHUNK_SIZE);
            let chunk_keys: Vec<ArrayRef> = group_arrays
                .iter()
                .map(|array| array.slice(offset, len))
                .collect();
            let prev_num_groups = self.group_values.len();
            self.group_values
                .intern(&chunk_keys, &mut self.group_indices)?;
            let total_num_groups = self.group_values.len();
            let cs_chunk = cs_full_vals.map(|vals| &vals[offset..offset + len]);

            for (acc, states) in self.accumulators.iter_mut().zip(&state_arrays) {
                let same_as_cs = cs_full_arr
                    .as_ref()
                    .is_some_and(|cs| states.len() == 1 && Arc::ptr_eq(&states[0], cs));
                let chunk_states: Vec<ArrayRef> = states
                    .iter()
                    .map(|array| array.slice(offset, len))
                    .collect();
                acc.merge_state(
                    &chunk_states,
                    &self.group_indices,
                    prev_num_groups,
                    total_num_groups,
                    cs_chunk,
                    same_as_cs,
                )?;
            }
            self.update_reservation()?;

            offset += len;
            if offset < total_other {
                tokio::task::yield_now().await;
            }
        }
        drop(group_arrays);
        drop(cs_full_arr);
        drop(state_arrays);
        drop(emitted_reservation);
        Ok(true)
    }

    async fn emit(
        mut self,
        schema: &SchemaRef,
        output_plan: &[OutputAggPlan],
        result_operators: &[Operator],
        batch_size: usize,
        tx: &mpsc::Sender<RecordBatch>,
        measurement: Option<&RunMetrics>,
    ) -> Result<()> {
        if self.group_values.is_empty() || tx.is_closed() {
            return Ok(());
        }
        let mut columns = self.group_values.emit(EmitTo::All)?;
        columns.extend(evaluate_fast_groups(
            &mut self.accumulators,
            self.count_star_idx,
            output_plan,
        )?);
        let batch = RecordBatch::try_new(Arc::clone(schema), columns)?;
        let batch_bytes = batch.get_array_memory_size();
        let batch_reservation = self.reservation.new_empty();
        drop(self);
        batch_reservation.try_grow(batch_bytes)?;
        emit_finalized_batch(batch, result_operators, batch_size, tx, measurement)
            .await?;
        drop(batch_reservation);
        Ok(())
    }
}

enum DirectAggState {
    CountStar(Vec<i64>),
    CountCol {
        col: usize,
        count_star_idx: Option<usize>,
        counts: Vec<i64>,
        null_counts: Option<Vec<i64>>,
    },
    SumInt64 {
        col: usize,
        sums: Vec<i64>,
        has_val: Option<Vec<bool>>,
    },
    SumInt32 {
        col: usize,
        sums: Vec<i64>,
        has_val: Option<Vec<bool>>,
    },
    SumInt64Wide {
        col: usize,
        sums: Vec<i128>,
        has_val: Option<Vec<bool>>,
    },
    SumFloat64 {
        col: usize,
        sums: Vec<f64>,
        has_val: Option<Vec<bool>>,
    },
    SumDecimal128 {
        data_type: DataType,
        input: Decimal128Input,
        sums: Vec<i128>,
        has_val: Option<Vec<bool>>,
    },
    AvgFloat64 {
        col: usize,
        sums: Vec<f64>,
        counts: Vec<u64>,
    },
    MinInt64 {
        col: usize,
        vals: Vec<i64>,
        has_val: Vec<bool>,
    },
    MinInt32 {
        col: usize,
        vals: Vec<i32>,
        has_val: Vec<bool>,
    },
    MaxInt64 {
        col: usize,
        vals: Vec<i64>,
        has_val: Vec<bool>,
    },
    MaxInt32 {
        col: usize,
        vals: Vec<i32>,
        has_val: Vec<bool>,
    },
}

impl DirectAggState {
    fn from_spec(spec: &DirectAggSpec) -> Self {
        match spec {
            DirectAggSpec::CountStar => Self::CountStar(Vec::new()),
            DirectAggSpec::CountCol {
                col,
                count_star_idx,
            } => Self::CountCol {
                col: *col,
                count_star_idx: *count_star_idx,
                counts: Vec::new(),
                null_counts: None,
            },
            DirectAggSpec::SumInt64(col) => Self::SumInt64 {
                col: *col,
                sums: Vec::new(),
                has_val: None,
            },
            DirectAggSpec::SumInt32(col) => Self::SumInt32 {
                col: *col,
                sums: Vec::new(),
                has_val: None,
            },
            DirectAggSpec::SumInt64Wide(col) => Self::SumInt64Wide {
                col: *col,
                sums: Vec::new(),
                has_val: None,
            },
            DirectAggSpec::SumFloat64(col) => Self::SumFloat64 {
                col: *col,
                sums: Vec::new(),
                has_val: None,
            },
            DirectAggSpec::SumDecimal128 { data_type, input } => Self::SumDecimal128 {
                data_type: data_type.clone(),
                input: *input,
                sums: Vec::new(),
                has_val: None,
            },
            DirectAggSpec::AvgFloat64(col) => Self::AvgFloat64 {
                col: *col,
                sums: Vec::new(),
                counts: Vec::new(),
            },
            DirectAggSpec::MinInt64(col) => Self::MinInt64 {
                col: *col,
                vals: Vec::new(),
                has_val: Vec::new(),
            },
            DirectAggSpec::MinInt32(col) => Self::MinInt32 {
                col: *col,
                vals: Vec::new(),
                has_val: Vec::new(),
            },
            DirectAggSpec::MaxInt64(col) => Self::MaxInt64 {
                col: *col,
                vals: Vec::new(),
                has_val: Vec::new(),
            },
            DirectAggSpec::MaxInt32(col) => Self::MaxInt32 {
                col: *col,
                vals: Vec::new(),
                has_val: Vec::new(),
            },
        }
    }

    fn resize(&mut self, num_groups: usize) {
        match self {
            Self::CountStar(c) => c.resize(num_groups, 0),
            Self::CountCol {
                count_star_idx,
                counts,
                null_counts,
                ..
            } => {
                if count_star_idx.is_some() {
                    if let Some(nc) = null_counts.as_mut() {
                        nc.resize(num_groups, 0);
                    }
                } else {
                    counts.resize(num_groups, 0);
                }
            }
            Self::SumInt64 { sums, has_val, .. }
            | Self::SumInt32 { sums, has_val, .. } => {
                sums.resize(num_groups, 0);
                if let Some(hv) = has_val.as_mut() {
                    hv.resize(num_groups, false);
                }
            }
            Self::SumInt64Wide { sums, has_val, .. }
            | Self::SumDecimal128 { sums, has_val, .. } => {
                sums.resize(num_groups, 0);
                if let Some(hv) = has_val.as_mut() {
                    hv.resize(num_groups, false);
                }
            }
            Self::SumFloat64 { sums, has_val, .. } => {
                sums.resize(num_groups, 0.0);
                if let Some(hv) = has_val.as_mut() {
                    hv.resize(num_groups, false);
                }
            }
            Self::AvgFloat64 { sums, counts, .. } => {
                sums.resize(num_groups, 0.0);
                counts.resize(num_groups, 0);
            }
            Self::MinInt64 { vals, has_val, .. }
            | Self::MaxInt64 { vals, has_val, .. } => {
                vals.resize(num_groups, 0);
                has_val.resize(num_groups, false);
            }
            Self::MinInt32 { vals, has_val, .. }
            | Self::MaxInt32 { vals, has_val, .. } => {
                vals.resize(num_groups, 0);
                has_val.resize(num_groups, false);
            }
        }
    }

    fn memory_bytes(&self) -> usize {
        match self {
            Self::CountStar(c) => c.capacity().saturating_mul(size_of::<i64>()),
            Self::CountCol {
                counts,
                null_counts,
                ..
            } => counts
                .capacity()
                .saturating_mul(size_of::<i64>())
                .saturating_add(
                    null_counts
                        .as_ref()
                        .map_or(0, |v| v.capacity().saturating_mul(size_of::<i64>())),
                ),
            Self::SumInt64 { sums, has_val, .. }
            | Self::SumInt32 { sums, has_val, .. } => sums
                .capacity()
                .saturating_mul(size_of::<i64>())
                .saturating_add(has_val.as_ref().map_or(0, Vec::capacity)),
            Self::SumInt64Wide { sums, has_val, .. }
            | Self::SumDecimal128 { sums, has_val, .. } => sums
                .capacity()
                .saturating_mul(size_of::<i128>())
                .saturating_add(has_val.as_ref().map_or(0, Vec::capacity)),
            Self::SumFloat64 { sums, has_val, .. } => sums
                .capacity()
                .saturating_mul(size_of::<f64>())
                .saturating_add(has_val.as_ref().map_or(0, Vec::capacity)),
            Self::AvgFloat64 { sums, counts, .. } => sums
                .capacity()
                .saturating_mul(size_of::<f64>())
                .saturating_add(counts.capacity().saturating_mul(size_of::<u64>())),
            Self::MinInt64 { vals, has_val, .. }
            | Self::MaxInt64 { vals, has_val, .. } => vals
                .capacity()
                .saturating_mul(size_of::<i64>())
                .saturating_add(has_val.capacity()),
            Self::MinInt32 { vals, has_val, .. }
            | Self::MaxInt32 { vals, has_val, .. } => vals
                .capacity()
                .saturating_mul(size_of::<i32>())
                .saturating_add(has_val.capacity()),
        }
    }

    fn update_from_view(
        &mut self,
        view: &BatchView,
        group_indices: &[usize],
        prev_num_groups: usize,
        num_groups: usize,
    ) -> Result<()> {
        let batch = view.batch();
        let selection = view.selection();
        macro_rules! for_each_selected_row {
            ($row:ident, $gid:ident => $body:expr) => {
                match selection {
                    Selection::All => {
                        for ($row, &$gid) in group_indices.iter().enumerate() {
                            $body
                        }
                    }
                    Selection::Mask { bits, .. } => {
                        for ($row, &$gid) in bits.set_indices().zip(group_indices.iter())
                        {
                            $body
                        }
                    }
                    Selection::Indices(indices) => {
                        for (&idx, &$gid) in
                            indices.values().iter().zip(group_indices.iter())
                        {
                            let $row = idx as usize;
                            $body
                        }
                    }
                }
            };
        }

        match self {
            Self::CountStar(counts) => {
                for &gid in group_indices {
                    counts[gid] =
                        counts[gid].checked_add(1).ok_or_else(count_overflow)?;
                }
            }
            Self::CountCol {
                col,
                count_star_idx,
                counts,
                null_counts,
            } => {
                let array = batch.column(*col);
                if count_star_idx.is_some() {
                    if array_has_no_nulls_fast(array) {
                        return Ok(());
                    }
                    if matches!(array.data_type(), DataType::Null) {
                        let nc = null_counts.get_or_insert_with(Vec::new);
                        if nc.len() < num_groups {
                            nc.resize(num_groups, 0);
                        }
                        for &gid in group_indices {
                            nc[gid] =
                                nc[gid].checked_add(1).ok_or_else(count_overflow)?;
                        }
                    } else if let Some(nulls) = array.logical_nulls()
                        && nulls.null_count() > 0
                    {
                        let nc = null_counts.get_or_insert_with(Vec::new);
                        if nc.len() < num_groups {
                            nc.resize(num_groups, 0);
                        }
                        if nulls.null_count() == array.len() {
                            for &gid in group_indices {
                                nc[gid] =
                                    nc[gid].checked_add(1).ok_or_else(count_overflow)?;
                            }
                        } else {
                            for_each_selected_row!(row, gid => {
                                if !nulls.is_valid(row) {
                                    nc[gid] =
                                        nc[gid].checked_add(1).ok_or_else(count_overflow)?;
                                }
                            });
                        }
                    }
                } else if matches!(array.data_type(), DataType::Null) {
                    // Every element of a NullArray is logically NULL.
                } else if array_has_no_nulls_fast(array) {
                    for &gid in group_indices {
                        counts[gid] =
                            counts[gid].checked_add(1).ok_or_else(count_overflow)?;
                    }
                } else if let Some(nulls) = array.logical_nulls() {
                    let null_count = nulls.null_count();
                    if null_count == 0 {
                        for &gid in group_indices {
                            counts[gid] =
                                counts[gid].checked_add(1).ok_or_else(count_overflow)?;
                        }
                    } else if null_count < array.len() {
                        for_each_selected_row!(row, gid => {
                            if nulls.is_valid(row) {
                                counts[gid] =
                                    counts[gid].checked_add(1).ok_or_else(count_overflow)?;
                            }
                        });
                    }
                } else {
                    for &gid in group_indices {
                        counts[gid] =
                            counts[gid].checked_add(1).ok_or_else(count_overflow)?;
                    }
                }
            }
            Self::SumInt64 { col, sums, has_val } => {
                let arr = batch
                    .column(*col)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected Int64 column for SUM"
                        )
                    })?;
                let values = arr.values();
                if arr.null_count() == 0 {
                    if let Some(hv) = has_val.as_mut() {
                        for_each_selected_row!(row, gid => {
                            sums[gid] = sums[gid].wrapping_add(values[row]);
                            hv[gid] = true;
                        });
                    } else {
                        for_each_selected_row!(row, gid => {
                            sums[gid] = sums[gid].wrapping_add(values[row]);
                        });
                    }
                } else {
                    let hv = ensure_has_val(has_val, prev_num_groups, num_groups);
                    if arr.null_count() < arr.len() {
                        let nulls = arr.nulls().unwrap();
                        for_each_selected_row!(row, gid => {
                            if nulls.is_valid(row) {
                                sums[gid] = sums[gid].wrapping_add(values[row]);
                                hv[gid] = true;
                            }
                        });
                    }
                }
            }
            Self::SumInt32 { col, sums, has_val } => {
                let arr = batch
                    .column(*col)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected Int32 column for SUM"
                        )
                    })?;
                let values = arr.values();
                if arr.null_count() == 0 {
                    if let Some(hv) = has_val.as_mut() {
                        for_each_selected_row!(row, gid => {
                            sums[gid] = sums[gid].wrapping_add(values[row] as i64);
                            hv[gid] = true;
                        });
                    } else {
                        for_each_selected_row!(row, gid => {
                            sums[gid] = sums[gid].wrapping_add(values[row] as i64);
                        });
                    }
                } else {
                    let hv = ensure_has_val(has_val, prev_num_groups, num_groups);
                    if arr.null_count() < arr.len() {
                        let nulls = arr.nulls().unwrap();
                        for_each_selected_row!(row, gid => {
                            if nulls.is_valid(row) {
                                sums[gid] = sums[gid].wrapping_add(values[row] as i64);
                                hv[gid] = true;
                            }
                        });
                    }
                }
            }
            Self::SumInt64Wide { col, sums, has_val } => {
                let arr = batch
                    .column(*col)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected Int64 column for wide SUM"
                        )
                    })?;
                let values = arr.values();
                if arr.null_count() == 0 {
                    if let Some(hv) = has_val.as_mut() {
                        for_each_selected_row!(row, gid => {
                            sums[gid] = sums[gid].wrapping_add(values[row] as i128);
                            hv[gid] = true;
                        });
                    } else {
                        for_each_selected_row!(row, gid => {
                            sums[gid] = sums[gid].wrapping_add(values[row] as i128);
                        });
                    }
                } else {
                    let hv = ensure_has_val(has_val, prev_num_groups, num_groups);
                    if arr.null_count() < arr.len() {
                        let nulls = arr.nulls().unwrap();
                        for_each_selected_row!(row, gid => {
                            if nulls.is_valid(row) {
                                sums[gid] = sums[gid].wrapping_add(values[row] as i128);
                                hv[gid] = true;
                            }
                        });
                    }
                }
            }
            Self::SumFloat64 { col, sums, has_val } => {
                let arr = batch
                    .column(*col)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected Float64 column for SUM"
                        )
                    })?;
                let values = arr.values();
                if arr.null_count() == 0 {
                    if let Some(hv) = has_val.as_mut() {
                        for_each_selected_row!(row, gid => {
                            sums[gid] += values[row];
                            hv[gid] = true;
                        });
                    } else {
                        for_each_selected_row!(row, gid => {
                            sums[gid] += values[row];
                        });
                    }
                } else {
                    let hv = ensure_has_val(has_val, prev_num_groups, num_groups);
                    if arr.null_count() < arr.len() {
                        let nulls = arr.nulls().unwrap();
                        for_each_selected_row!(row, gid => {
                            if nulls.is_valid(row) {
                                sums[gid] += values[row];
                                hv[gid] = true;
                            }
                        });
                    }
                }
            }
            Self::SumDecimal128 {
                input,
                sums,
                has_val,
                ..
            } => match *input {
                Decimal128Input::Col(col) => {
                    let arr = batch
                        .column(col)
                        .as_any()
                        .downcast_ref::<Decimal128Array>()
                        .ok_or_else(|| {
                            datafusion_common::internal_datafusion_err!(
                                "Expected Decimal128 column for SUM"
                            )
                        })?;
                    let values = arr.values();
                    if arr.null_count() == 0 {
                        if let Some(hv) = has_val.as_mut() {
                            for_each_selected_row!(row, gid => {
                                sums[gid] = sums[gid].wrapping_add(values[row]);
                                hv[gid] = true;
                            });
                        } else {
                            for_each_selected_row!(row, gid => {
                                sums[gid] = sums[gid].wrapping_add(values[row]);
                            });
                        }
                    } else {
                        let hv = ensure_has_val(has_val, prev_num_groups, num_groups);
                        if arr.null_count() < arr.len() {
                            let nulls = arr.nulls().unwrap();
                            for_each_selected_row!(row, gid => {
                                if nulls.is_valid(row) {
                                    sums[gid] = sums[gid].wrapping_add(values[row]);
                                    hv[gid] = true;
                                }
                            });
                        }
                    }
                }
                Decimal128Input::MulColCol { a, b }
                | Decimal128Input::MulColAddLit { a, b, .. }
                | Decimal128Input::MulColSubFromLit { a, b, .. }
                | Decimal128Input::MulColSubLit { a, b, .. } => {
                    let arr_a = batch
                        .column(a)
                        .as_any()
                        .downcast_ref::<Decimal128Array>()
                        .ok_or_else(|| {
                            datafusion_common::internal_datafusion_err!(
                                "Expected Decimal128 column at {a}"
                            )
                        })?;
                    let arr_b = batch
                        .column(b)
                        .as_any()
                        .downcast_ref::<Decimal128Array>()
                        .ok_or_else(|| {
                            datafusion_common::internal_datafusion_err!(
                                "Expected Decimal128 column at {b}"
                            )
                        })?;
                    let va = arr_a.values();
                    let vb = arr_b.values();
                    if arr_a.null_count() == 0 && arr_b.null_count() == 0 {
                        if let Some(hv) = has_val.as_mut() {
                            for_each_selected_row!(row, gid => {
                                let val = match *input {
                                    Decimal128Input::MulColCol { .. } => {
                                        checked_mul_i128_fast(va[row], vb[row])?
                                    }
                                    Decimal128Input::MulColAddLit { lit, .. } => {
                                        let rhs = checked_add_i128(lit, vb[row])?;
                                        checked_mul_i128_fast(va[row], rhs)?
                                    }
                                    Decimal128Input::MulColSubFromLit { lit, .. } => {
                                        let rhs = checked_sub_i128(lit, vb[row])?;
                                        checked_mul_i128_fast(va[row], rhs)?
                                    }
                                    Decimal128Input::MulColSubLit { lit, .. } => {
                                        let rhs = checked_sub_i128(vb[row], lit)?;
                                        checked_mul_i128_fast(va[row], rhs)?
                                    }
                                    Decimal128Input::Col(_) => unreachable!(),
                                };
                                sums[gid] = sums[gid].wrapping_add(val);
                                hv[gid] = true;
                            });
                        } else {
                            match *input {
                                Decimal128Input::MulColCol { .. } => {
                                    for_each_selected_row!(row, gid => {
                                        let val = checked_mul_i128_fast(va[row], vb[row])?;
                                        sums[gid] = sums[gid].wrapping_add(val);
                                    });
                                }
                                Decimal128Input::MulColAddLit { lit, .. } => {
                                    for_each_selected_row!(row, gid => {
                                        let rhs = checked_add_i128(lit, vb[row])?;
                                        let val = checked_mul_i128_fast(va[row], rhs)?;
                                        sums[gid] = sums[gid].wrapping_add(val);
                                    });
                                }
                                Decimal128Input::MulColSubFromLit { lit, .. } => {
                                    for_each_selected_row!(row, gid => {
                                        let rhs = checked_sub_i128(lit, vb[row])?;
                                        let val = checked_mul_i128_fast(va[row], rhs)?;
                                        sums[gid] = sums[gid].wrapping_add(val);
                                    });
                                }
                                Decimal128Input::MulColSubLit { lit, .. } => {
                                    for_each_selected_row!(row, gid => {
                                        let rhs = checked_sub_i128(vb[row], lit)?;
                                        let val = checked_mul_i128_fast(va[row], rhs)?;
                                        sums[gid] = sums[gid].wrapping_add(val);
                                    });
                                }
                                Decimal128Input::Col(_) => unreachable!(),
                            }
                        }
                    } else {
                        let hv = ensure_has_val(has_val, prev_num_groups, num_groups);
                        if arr_a.null_count() < arr_a.len()
                            && arr_b.null_count() < arr_b.len()
                        {
                            let na = arr_a.nulls();
                            let nb = arr_b.nulls();
                            for_each_selected_row!(row, gid => {
                                if na.is_none_or(|m| m.is_valid(row))
                                    && nb.is_none_or(|m| m.is_valid(row))
                                {
                                    let val = match *input {
                                        Decimal128Input::MulColCol { .. } => {
                                            checked_mul_i128_fast(va[row], vb[row])?
                                        }
                                        Decimal128Input::MulColAddLit { lit, .. } => {
                                            let rhs = checked_add_i128(lit, vb[row])?;
                                            checked_mul_i128_fast(va[row], rhs)?
                                        }
                                        Decimal128Input::MulColSubFromLit { lit, .. } => {
                                            let rhs = checked_sub_i128(lit, vb[row])?;
                                            checked_mul_i128_fast(va[row], rhs)?
                                        }
                                        Decimal128Input::MulColSubLit { lit, .. } => {
                                            let rhs = checked_sub_i128(vb[row], lit)?;
                                            checked_mul_i128_fast(va[row], rhs)?
                                        }
                                        Decimal128Input::Col(_) => unreachable!(),
                                    };
                                    sums[gid] = sums[gid].wrapping_add(val);
                                    hv[gid] = true;
                                }
                            });
                        }
                    }
                }
            },
            Self::AvgFloat64 { col, sums, counts } => {
                let arr = batch
                    .column(*col)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected Float64 column for AVG"
                        )
                    })?;
                let values = arr.values();
                if arr.null_count() == 0 {
                    for_each_selected_row!(row, gid => {
                        sums[gid] += values[row];
                        counts[gid] += 1;
                    });
                } else if arr.null_count() < arr.len() {
                    let nulls = arr.nulls().unwrap();
                    for_each_selected_row!(row, gid => {
                        if nulls.is_valid(row) {
                            sums[gid] += values[row];
                            counts[gid] += 1;
                        }
                    });
                }
            }
            Self::MinInt64 { col, vals, has_val } => {
                let arr = batch
                    .column(*col)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected Int64 column for MIN"
                        )
                    })?;
                let values = arr.values();
                if arr.null_count() == 0 {
                    for_each_selected_row!(row, gid => {
                        let v = values[row];
                        if !has_val[gid] || v < vals[gid] {
                            vals[gid] = v;
                            has_val[gid] = true;
                        }
                    });
                } else if arr.null_count() < arr.len() {
                    let nulls = arr.nulls().unwrap();
                    for_each_selected_row!(row, gid => {
                        if nulls.is_valid(row) {
                            let v = values[row];
                            if !has_val[gid] || v < vals[gid] {
                                vals[gid] = v;
                                has_val[gid] = true;
                            }
                        }
                    });
                }
            }
            Self::MinInt32 { col, vals, has_val } => {
                let arr = batch
                    .column(*col)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected Int32 column for MIN"
                        )
                    })?;
                let values = arr.values();
                if arr.null_count() == 0 {
                    for_each_selected_row!(row, gid => {
                        let v = values[row];
                        if !has_val[gid] || v < vals[gid] {
                            vals[gid] = v;
                            has_val[gid] = true;
                        }
                    });
                } else if arr.null_count() < arr.len() {
                    let nulls = arr.nulls().unwrap();
                    for_each_selected_row!(row, gid => {
                        if nulls.is_valid(row) {
                            let v = values[row];
                            if !has_val[gid] || v < vals[gid] {
                                vals[gid] = v;
                                has_val[gid] = true;
                            }
                        }
                    });
                }
            }
            Self::MaxInt64 { col, vals, has_val } => {
                let arr = batch
                    .column(*col)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected Int64 column for MAX"
                        )
                    })?;
                let values = arr.values();
                if arr.null_count() == 0 {
                    for_each_selected_row!(row, gid => {
                        let v = values[row];
                        if !has_val[gid] || v > vals[gid] {
                            vals[gid] = v;
                            has_val[gid] = true;
                        }
                    });
                } else if arr.null_count() < arr.len() {
                    let nulls = arr.nulls().unwrap();
                    for_each_selected_row!(row, gid => {
                        if nulls.is_valid(row) {
                            let v = values[row];
                            if !has_val[gid] || v > vals[gid] {
                                vals[gid] = v;
                                has_val[gid] = true;
                            }
                        }
                    });
                }
            }
            Self::MaxInt32 { col, vals, has_val } => {
                let arr = batch
                    .column(*col)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected Int32 column for MAX"
                        )
                    })?;
                let values = arr.values();
                if arr.null_count() == 0 {
                    for_each_selected_row!(row, gid => {
                        let v = values[row];
                        if !has_val[gid] || v > vals[gid] {
                            vals[gid] = v;
                            has_val[gid] = true;
                        }
                    });
                } else if arr.null_count() < arr.len() {
                    let nulls = arr.nulls().unwrap();
                    for_each_selected_row!(row, gid => {
                        if nulls.is_valid(row) {
                            let v = values[row];
                            if !has_val[gid] || v > vals[gid] {
                                vals[gid] = v;
                                has_val[gid] = true;
                            }
                        }
                    });
                }
            }
        }
        Ok(())
    }

    fn merge_group(
        &mut self,
        dst_gid: usize,
        src: &Self,
        src_gid: usize,
        prev_num_groups: usize,
        num_groups: usize,
    ) -> Result<()> {
        match (self, src) {
            (Self::CountStar(dst), Self::CountStar(s)) => {
                dst[dst_gid] = dst[dst_gid]
                    .checked_add(s[src_gid])
                    .ok_or_else(count_overflow)?;
            }
            (
                Self::CountCol {
                    count_star_idx,
                    counts: dst,
                    null_counts: dst_nc,
                    ..
                },
                Self::CountCol {
                    counts: s,
                    null_counts: src_nc,
                    ..
                },
            ) => {
                if count_star_idx.is_some() {
                    if let Some(s_nc) = src_nc {
                        let delta = s_nc.get(src_gid).copied().unwrap_or(0);
                        if delta != 0 {
                            let d_nc = dst_nc.get_or_insert_with(Vec::new);
                            if d_nc.len() < num_groups {
                                d_nc.resize(num_groups, 0);
                            }
                            d_nc[dst_gid] = d_nc[dst_gid]
                                .checked_add(delta)
                                .ok_or_else(count_overflow)?;
                        }
                    }
                } else {
                    dst[dst_gid] = dst[dst_gid]
                        .checked_add(s[src_gid])
                        .ok_or_else(count_overflow)?;
                }
            }
            (
                Self::SumInt64 {
                    sums: ds,
                    has_val: dh,
                    ..
                },
                Self::SumInt64 {
                    sums: ss,
                    has_val: sh,
                    ..
                },
            )
            | (
                Self::SumInt32 {
                    sums: ds,
                    has_val: dh,
                    ..
                },
                Self::SumInt32 {
                    sums: ss,
                    has_val: sh,
                    ..
                },
            ) => {
                let src_has = sh.as_ref().is_none_or(|v| v[src_gid]);
                if src_has {
                    ds[dst_gid] = ds[dst_gid].wrapping_add(ss[src_gid]);
                    if let Some(hv) = dh.as_mut() {
                        hv[dst_gid] = true;
                    }
                } else if dh.is_none() && dst_gid >= prev_num_groups {
                    let _ = ensure_has_val(dh, prev_num_groups, num_groups);
                }
            }
            (
                Self::SumInt64Wide {
                    sums: ds,
                    has_val: dh,
                    ..
                },
                Self::SumInt64Wide {
                    sums: ss,
                    has_val: sh,
                    ..
                },
            )
            | (
                Self::SumDecimal128 {
                    sums: ds,
                    has_val: dh,
                    ..
                },
                Self::SumDecimal128 {
                    sums: ss,
                    has_val: sh,
                    ..
                },
            ) => {
                let src_has = sh.as_ref().is_none_or(|v| v[src_gid]);
                if src_has {
                    ds[dst_gid] = ds[dst_gid].wrapping_add(ss[src_gid]);
                    if let Some(hv) = dh.as_mut() {
                        hv[dst_gid] = true;
                    }
                } else if dh.is_none() && dst_gid >= prev_num_groups {
                    let _ = ensure_has_val(dh, prev_num_groups, num_groups);
                }
            }
            (
                Self::SumFloat64 {
                    sums: ds,
                    has_val: dh,
                    ..
                },
                Self::SumFloat64 {
                    sums: ss,
                    has_val: sh,
                    ..
                },
            ) => {
                let src_has = sh.as_ref().is_none_or(|v| v[src_gid]);
                if src_has {
                    ds[dst_gid] += ss[src_gid];
                    if let Some(hv) = dh.as_mut() {
                        hv[dst_gid] = true;
                    }
                } else if dh.is_none() && dst_gid >= prev_num_groups {
                    let _ = ensure_has_val(dh, prev_num_groups, num_groups);
                }
            }
            (
                Self::AvgFloat64 {
                    sums: ds,
                    counts: dc,
                    ..
                },
                Self::AvgFloat64 {
                    sums: ss,
                    counts: sc,
                    ..
                },
            ) => {
                if sc[src_gid] > 0 {
                    ds[dst_gid] += ss[src_gid];
                    dc[dst_gid] += sc[src_gid];
                }
            }
            (
                Self::MinInt64 {
                    vals: dv,
                    has_val: dh,
                    ..
                },
                Self::MinInt64 {
                    vals: sv,
                    has_val: sh,
                    ..
                },
            ) => {
                if sh[src_gid] && (!dh[dst_gid] || sv[src_gid] < dv[dst_gid]) {
                    dv[dst_gid] = sv[src_gid];
                    dh[dst_gid] = true;
                }
            }
            (
                Self::MinInt32 {
                    vals: dv,
                    has_val: dh,
                    ..
                },
                Self::MinInt32 {
                    vals: sv,
                    has_val: sh,
                    ..
                },
            ) => {
                if sh[src_gid] && (!dh[dst_gid] || sv[src_gid] < dv[dst_gid]) {
                    dv[dst_gid] = sv[src_gid];
                    dh[dst_gid] = true;
                }
            }
            (
                Self::MaxInt64 {
                    vals: dv,
                    has_val: dh,
                    ..
                },
                Self::MaxInt64 {
                    vals: sv,
                    has_val: sh,
                    ..
                },
            ) => {
                if sh[src_gid] && (!dh[dst_gid] || sv[src_gid] > dv[dst_gid]) {
                    dv[dst_gid] = sv[src_gid];
                    dh[dst_gid] = true;
                }
            }
            (
                Self::MaxInt32 {
                    vals: dv,
                    has_val: dh,
                    ..
                },
                Self::MaxInt32 {
                    vals: sv,
                    has_val: sh,
                    ..
                },
            ) => {
                if sh[src_gid] && (!dh[dst_gid] || sv[src_gid] > dv[dst_gid]) {
                    dv[dst_gid] = sv[src_gid];
                    dh[dst_gid] = true;
                }
            }
            _ => return internal_err!("DirectGroupedState accumulator mismatch"),
        }
        Ok(())
    }

    fn drain_to_partial_arrays(
        &mut self,
        cs_arr: Option<&Arc<Int64Array>>,
        out: &mut Vec<ArrayRef>,
    ) -> Result<()> {
        match self {
            Self::CountStar(c) => {
                let vals = std::mem::take(c);
                out.push(Arc::new(Int64Array::from(vals)));
            }
            Self::CountCol {
                count_star_idx,
                counts,
                null_counts,
                ..
            } => {
                if count_star_idx.is_some() {
                    let cs = cs_arr.ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Missing CountStar array in DirectAggState drain"
                        )
                    })?;
                    let Some(nc) = null_counts.take() else {
                        out.push(Arc::clone(cs) as ArrayRef);
                        return Ok(());
                    };
                    let vals: Vec<i64> = cs
                        .values()
                        .iter()
                        .enumerate()
                        .map(|(i, &total)| total - nc.get(i).copied().unwrap_or(0))
                        .collect();
                    out.push(Arc::new(Int64Array::from(vals)));
                } else {
                    let vals = std::mem::take(counts);
                    out.push(Arc::new(Int64Array::from(vals)));
                }
            }
            Self::SumInt64 { sums, has_val, .. }
            | Self::SumInt32 { sums, has_val, .. } => {
                let vals = std::mem::take(sums);
                let nulls = build_null_buffer_from_has_val(has_val.take());
                out.push(Arc::new(Int64Array::new(vals.into(), nulls)));
            }
            Self::SumInt64Wide { sums, has_val, .. } => {
                let vals = std::mem::take(sums);
                let nulls = build_null_buffer_from_has_val(has_val.take());
                out.push(Arc::new(
                    Decimal128Array::new(vals.into(), nulls)
                        .with_precision_and_scale(DECIMAL128_MAX_PRECISION, 0)?,
                ));
            }
            Self::SumFloat64 { sums, has_val, .. } => {
                let vals = std::mem::take(sums);
                let nulls = build_null_buffer_from_has_val(has_val.take());
                out.push(Arc::new(Float64Array::new(vals.into(), nulls)));
            }
            Self::SumDecimal128 {
                data_type,
                sums,
                has_val,
                ..
            } => {
                let vals = std::mem::take(sums);
                let nulls = build_null_buffer_from_has_val(has_val.take());
                let (p, s) = match data_type {
                    DataType::Decimal128(p, s) => (*p, *s),
                    _ => return internal_err!("Expected Decimal128 type"),
                };
                out.push(Arc::new(
                    Decimal128Array::new(vals.into(), nulls)
                        .with_precision_and_scale(p, s)?,
                ));
            }
            Self::AvgFloat64 { sums, counts, .. } => {
                let s = std::mem::take(sums);
                let c = std::mem::take(counts);
                out.push(Arc::new(Float64Array::from(s)));
                out.push(Arc::new(UInt64Array::from(c)));
            }
            Self::MinInt64 { vals, has_val, .. }
            | Self::MaxInt64 { vals, has_val, .. } => {
                let v = std::mem::take(vals);
                let hv = std::mem::take(has_val);
                let mut nulls = NullBufferBuilder::new(hv.len());
                for b in hv {
                    nulls.append(b);
                }
                out.push(Arc::new(Int64Array::new(v.into(), nulls.finish())));
            }
            Self::MinInt32 { vals, has_val, .. }
            | Self::MaxInt32 { vals, has_val, .. } => {
                let v = std::mem::take(vals);
                let hv = std::mem::take(has_val);
                let mut nulls = NullBufferBuilder::new(hv.len());
                for b in hv {
                    nulls.append(b);
                }
                out.push(Arc::new(Int32Array::new(v.into(), nulls.finish())));
            }
        }
        Ok(())
    }

    fn merge_partial_columns(
        &mut self,
        batch: &RecordBatch,
        col_offset: &mut usize,
        group_indices: &[usize],
        prev_num_groups: usize,
        num_groups: usize,
        cs_col: Option<&ArrayRef>,
    ) -> Result<()> {
        match self {
            Self::CountStar(counts) => {
                let arr = batch
                    .column(*col_offset)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected Int64 partial count column"
                        )
                    })?;
                *col_offset += 1;
                for (&c, &gid) in arr.values().iter().zip(group_indices.iter()) {
                    counts[gid] =
                        counts[gid].checked_add(c).ok_or_else(count_overflow)?;
                }
            }
            Self::CountCol {
                count_star_idx,
                counts,
                null_counts,
                ..
            } => {
                let col_ref = batch.column(*col_offset);
                *col_offset += 1;
                if count_star_idx.is_some() {
                    if let Some(cs_ref) = cs_col
                        && Arc::ptr_eq(col_ref, cs_ref)
                    {
                        return Ok(());
                    }
                    let cs = cs_col
                        .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
                        .ok_or_else(|| {
                            datafusion_common::internal_datafusion_err!(
                                "Missing CountStar partial column in DirectAggState merge"
                            )
                        })?
                        .values();
                    let cc = col_ref
                        .as_any()
                        .downcast_ref::<Int64Array>()
                        .ok_or_else(|| {
                            datafusion_common::internal_datafusion_err!(
                                "Expected Int64 partial count column"
                            )
                        })?
                        .values();
                    for ((&cs_v, &cc_v), &gid) in
                        cs.iter().zip(cc.iter()).zip(group_indices.iter())
                    {
                        let diff = cs_v - cc_v;
                        if diff != 0 {
                            let nc = null_counts.get_or_insert_with(Vec::new);
                            if nc.len() < num_groups {
                                nc.resize(num_groups, 0);
                            }
                            nc[gid] =
                                nc[gid].checked_add(diff).ok_or_else(count_overflow)?;
                        }
                    }
                } else {
                    let arr = col_ref.as_any().downcast_ref::<Int64Array>().ok_or_else(
                        || {
                            datafusion_common::internal_datafusion_err!(
                                "Expected Int64 partial count column"
                            )
                        },
                    )?;
                    for (&c, &gid) in arr.values().iter().zip(group_indices.iter()) {
                        counts[gid] =
                            counts[gid].checked_add(c).ok_or_else(count_overflow)?;
                    }
                }
            }
            Self::SumInt64 { sums, has_val, .. }
            | Self::SumInt32 { sums, has_val, .. } => {
                let arr = batch
                    .column(*col_offset)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected Int64 partial sum column"
                        )
                    })?;
                *col_offset += 1;
                update_sum_int64_array(
                    arr,
                    group_indices,
                    prev_num_groups,
                    num_groups,
                    sums,
                    has_val,
                );
            }
            Self::SumInt64Wide { sums, has_val, .. }
            | Self::SumDecimal128 { sums, has_val, .. } => {
                let arr = batch
                    .column(*col_offset)
                    .as_any()
                    .downcast_ref::<Decimal128Array>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected Decimal128 partial sum column"
                        )
                    })?;
                *col_offset += 1;
                update_sum_decimal128_array(
                    arr,
                    group_indices,
                    prev_num_groups,
                    num_groups,
                    sums,
                    has_val,
                );
            }
            Self::SumFloat64 { sums, has_val, .. } => {
                let arr = batch
                    .column(*col_offset)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected Float64 partial sum column"
                        )
                    })?;
                *col_offset += 1;
                update_sum_float64_array(
                    arr,
                    group_indices,
                    prev_num_groups,
                    num_groups,
                    sums,
                    has_val,
                );
            }
            Self::AvgFloat64 { sums, counts, .. } => {
                let s_arr = batch
                    .column(*col_offset)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected Float64 partial avg sum column"
                        )
                    })?;
                let c_arr = batch
                    .column(*col_offset + 1)
                    .as_any()
                    .downcast_ref::<UInt64Array>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected UInt64 partial avg count column"
                        )
                    })?;
                *col_offset += 2;
                for ((&s, &c), &gid) in s_arr
                    .values()
                    .iter()
                    .zip(c_arr.values().iter())
                    .zip(group_indices.iter())
                {
                    sums[gid] += s;
                    counts[gid] += c;
                }
            }
            Self::MinInt64 { vals, has_val, .. } => {
                let arr = batch
                    .column(*col_offset)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected Int64 partial min column"
                        )
                    })?;
                *col_offset += 1;
                let src = arr.values();
                if arr.null_count() == 0 {
                    for (&v, &gid) in src.iter().zip(group_indices.iter()) {
                        if !has_val[gid] || v < vals[gid] {
                            vals[gid] = v;
                            has_val[gid] = true;
                        }
                    }
                } else {
                    let nulls = arr.nulls().unwrap();
                    for (i, (&v, &gid)) in
                        src.iter().zip(group_indices.iter()).enumerate()
                    {
                        if nulls.is_valid(i) && (!has_val[gid] || v < vals[gid]) {
                            vals[gid] = v;
                            has_val[gid] = true;
                        }
                    }
                }
            }
            Self::MinInt32 { vals, has_val, .. } => {
                let arr = batch
                    .column(*col_offset)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected Int32 partial min column"
                        )
                    })?;
                *col_offset += 1;
                let src = arr.values();
                if arr.null_count() == 0 {
                    for (&v, &gid) in src.iter().zip(group_indices.iter()) {
                        if !has_val[gid] || v < vals[gid] {
                            vals[gid] = v;
                            has_val[gid] = true;
                        }
                    }
                } else {
                    let nulls = arr.nulls().unwrap();
                    for (i, (&v, &gid)) in
                        src.iter().zip(group_indices.iter()).enumerate()
                    {
                        if nulls.is_valid(i) && (!has_val[gid] || v < vals[gid]) {
                            vals[gid] = v;
                            has_val[gid] = true;
                        }
                    }
                }
            }
            Self::MaxInt64 { vals, has_val, .. } => {
                let arr = batch
                    .column(*col_offset)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected Int64 partial max column"
                        )
                    })?;
                *col_offset += 1;
                let src = arr.values();
                if arr.null_count() == 0 {
                    for (&v, &gid) in src.iter().zip(group_indices.iter()) {
                        if !has_val[gid] || v > vals[gid] {
                            vals[gid] = v;
                            has_val[gid] = true;
                        }
                    }
                } else {
                    let nulls = arr.nulls().unwrap();
                    for (i, (&v, &gid)) in
                        src.iter().zip(group_indices.iter()).enumerate()
                    {
                        if nulls.is_valid(i) && (!has_val[gid] || v > vals[gid]) {
                            vals[gid] = v;
                            has_val[gid] = true;
                        }
                    }
                }
            }
            Self::MaxInt32 { vals, has_val, .. } => {
                let arr = batch
                    .column(*col_offset)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected Int32 partial max column"
                        )
                    })?;
                *col_offset += 1;
                let src = arr.values();
                if arr.null_count() == 0 {
                    for (&v, &gid) in src.iter().zip(group_indices.iter()) {
                        if !has_val[gid] || v > vals[gid] {
                            vals[gid] = v;
                            has_val[gid] = true;
                        }
                    }
                } else {
                    let nulls = arr.nulls().unwrap();
                    for (i, (&v, &gid)) in
                        src.iter().zip(group_indices.iter()).enumerate()
                    {
                        if nulls.is_valid(i) && (!has_val[gid] || v > vals[gid]) {
                            vals[gid] = v;
                            has_val[gid] = true;
                        }
                    }
                }
            }
        }
        Ok(())
    }

    fn evaluate_range(
        &self,
        start: usize,
        len: usize,
        cs_arr: Option<&Arc<Int64Array>>,
    ) -> Result<ArrayRef> {
        let end = start + len;
        Ok(match self {
            Self::CountStar(c) => Arc::new(Int64Array::from(c[start..end].to_vec())),
            Self::CountCol {
                count_star_idx,
                counts,
                null_counts,
                ..
            } => {
                if count_star_idx.is_some() {
                    let cs = cs_arr.ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Missing CountStar array in DirectAggState evaluate_range"
                        )
                    })?;
                    let Some(nc) = null_counts else {
                        return Ok(Arc::clone(cs) as ArrayRef);
                    };
                    let vals: Vec<i64> = cs
                        .values()
                        .iter()
                        .enumerate()
                        .map(|(i, &total)| {
                            total - nc.get(start + i).copied().unwrap_or(0)
                        })
                        .collect();
                    Arc::new(Int64Array::from(vals))
                } else {
                    Arc::new(Int64Array::from(counts[start..end].to_vec()))
                }
            }
            Self::SumInt64 { sums, has_val, .. }
            | Self::SumInt32 { sums, has_val, .. } => {
                let nulls = has_val.as_ref().and_then(|hv| {
                    let mut b = NullBufferBuilder::new(len);
                    for &v in &hv[start..end] {
                        b.append(v);
                    }
                    b.finish()
                });
                Arc::new(Int64Array::new(sums[start..end].to_vec().into(), nulls))
            }
            Self::SumInt64Wide { sums, has_val, .. } => {
                let nulls = has_val.as_ref().and_then(|hv| {
                    let mut b = NullBufferBuilder::new(len);
                    for &v in &hv[start..end] {
                        b.append(v);
                    }
                    b.finish()
                });
                Arc::new(
                    Decimal128Array::new(sums[start..end].to_vec().into(), nulls)
                        .with_precision_and_scale(DECIMAL128_MAX_PRECISION, 0)?,
                )
            }
            Self::SumFloat64 { sums, has_val, .. } => {
                let nulls = has_val.as_ref().and_then(|hv| {
                    let mut b = NullBufferBuilder::new(len);
                    for &v in &hv[start..end] {
                        b.append(v);
                    }
                    b.finish()
                });
                Arc::new(Float64Array::new(sums[start..end].to_vec().into(), nulls))
            }
            Self::SumDecimal128 {
                data_type,
                sums,
                has_val,
                ..
            } => {
                let nulls = has_val.as_ref().and_then(|hv| {
                    let mut b = NullBufferBuilder::new(len);
                    for &v in &hv[start..end] {
                        b.append(v);
                    }
                    b.finish()
                });
                let (p, s) = match data_type {
                    DataType::Decimal128(p, s) => (*p, *s),
                    _ => return internal_err!("Expected Decimal128 type"),
                };
                Arc::new(
                    Decimal128Array::new(sums[start..end].to_vec().into(), nulls)
                        .with_precision_and_scale(p, s)?,
                )
            }
            Self::AvgFloat64 { sums, counts, .. } => {
                let mut vals = Vec::with_capacity(len);
                let mut nulls = NullBufferBuilder::new(len);
                for i in start..end {
                    if counts[i] > 0 {
                        vals.push(sums[i] / counts[i] as f64);
                        nulls.append_non_null();
                    } else {
                        vals.push(0.0);
                        nulls.append_null();
                    }
                }
                Arc::new(Float64Array::new(vals.into(), nulls.finish()))
            }
            Self::MinInt64 { vals, has_val, .. }
            | Self::MaxInt64 { vals, has_val, .. } => {
                let mut nulls = NullBufferBuilder::new(len);
                for &v in &has_val[start..end] {
                    nulls.append(v);
                }
                Arc::new(Int64Array::new(
                    vals[start..end].to_vec().into(),
                    nulls.finish(),
                ))
            }
            Self::MinInt32 { vals, has_val, .. }
            | Self::MaxInt32 { vals, has_val, .. } => {
                let mut nulls = NullBufferBuilder::new(len);
                for &v in &has_val[start..end] {
                    nulls.append(v);
                }
                Arc::new(Int32Array::new(
                    vals[start..end].to_vec().into(),
                    nulls.finish(),
                ))
            }
        })
    }
}

fn build_direct_raw_partial_batch(
    view: &BatchView,
    spec: &DirectGroupedSpec,
    partial_schema: &SchemaRef,
) -> Result<RecordBatch> {
    let batch = if view.is_all() {
        view.batch().clone()
    } else {
        view.clone().materialize()?
    };
    let n = batch.num_rows();
    let mut cols: Vec<ArrayRef> = Vec::with_capacity(partial_schema.fields().len());
    cols.push(Arc::clone(batch.column(spec.key_index)));
    let ones_col: Option<ArrayRef> = spec
        .count_star_idx
        .map(|_| Arc::new(Int64Array::from(vec![1i64; n])) as ArrayRef);
    for agg in &spec.aggs {
        match agg {
            DirectAggSpec::CountStar => {
                cols.push(Arc::clone(ones_col.as_ref().unwrap()));
            }
            DirectAggSpec::CountCol { col, .. } => {
                let arr = batch.column(*col);
                if matches!(arr.data_type(), DataType::Null) {
                    cols.push(Arc::new(Int64Array::from(vec![0i64; n])));
                } else if array_has_no_nulls_fast(arr) {
                    if let Some(ones) = &ones_col {
                        cols.push(Arc::clone(ones));
                    } else {
                        cols.push(Arc::new(Int64Array::from(vec![1i64; n])));
                    }
                } else if let Some(nulls) = arr.logical_nulls() {
                    if nulls.null_count() == 0 {
                        if let Some(ones) = &ones_col {
                            cols.push(Arc::clone(ones));
                        } else {
                            cols.push(Arc::new(Int64Array::from(vec![1i64; n])));
                        }
                    } else {
                        let counts: Vec<i64> =
                            (0..n).map(|i| i64::from(nulls.is_valid(i))).collect();
                        cols.push(Arc::new(Int64Array::from(counts)));
                    }
                } else if let Some(ones) = &ones_col {
                    cols.push(Arc::clone(ones));
                } else {
                    cols.push(Arc::new(Int64Array::from(vec![1i64; n])));
                }
            }
            DirectAggSpec::SumInt64(col)
            | DirectAggSpec::MinInt64(col)
            | DirectAggSpec::MaxInt64(col)
            | DirectAggSpec::SumFloat64(col)
            | DirectAggSpec::MinInt32(col)
            | DirectAggSpec::MaxInt32(col) => {
                cols.push(Arc::clone(batch.column(*col)));
            }
            DirectAggSpec::SumInt32(col) => {
                let arr = batch
                    .column(*col)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected Int32 column for SUM"
                        )
                    })?;
                let vals: Vec<i64> = arr.values().iter().map(|&v| v as i64).collect();
                cols.push(Arc::new(Int64Array::new(vals.into(), arr.nulls().cloned())));
            }
            DirectAggSpec::SumInt64Wide(col) => {
                let arr = batch
                    .column(*col)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected Int64 column for wide SUM"
                        )
                    })?;
                let vals: Vec<i128> = arr.values().iter().map(|&v| v as i128).collect();
                cols.push(Arc::new(
                    Decimal128Array::new(vals.into(), arr.nulls().cloned())
                        .with_precision_and_scale(DECIMAL128_MAX_PRECISION, 0)?,
                ));
            }
            DirectAggSpec::SumDecimal128 { data_type, input } => {
                cols.push(input.eval_to_array(&batch, data_type)?);
            }
            DirectAggSpec::AvgFloat64(col) => {
                let arr = batch
                    .column(*col)
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| {
                        datafusion_common::internal_datafusion_err!(
                            "Expected Float64 column for AVG"
                        )
                    })?;
                if arr.null_count() == 0 {
                    cols.push(Arc::new(Float64Array::from(arr.values().to_vec())));
                    cols.push(Arc::new(UInt64Array::from(vec![1u64; n])));
                } else {
                    let nulls = arr.nulls().unwrap();
                    let mut sums = Vec::with_capacity(n);
                    let mut counts = Vec::with_capacity(n);
                    for (i, &v) in arr.values().iter().enumerate() {
                        if nulls.is_valid(i) {
                            sums.push(v);
                            counts.push(1u64);
                        } else {
                            sums.push(0.0);
                            counts.push(0u64);
                        }
                    }
                    cols.push(Arc::new(Float64Array::from(sums)));
                    cols.push(Arc::new(UInt64Array::from(counts)));
                }
            }
        }
    }
    Ok(RecordBatch::try_new(Arc::clone(partial_schema), cols)?)
}

struct DirectIntGroupMap<T: ArrowPrimitiveType>
where
    T::Native: Eq + Hash + IntKeyHash,
{
    key_index: usize,
    map: HashMap<T::Native, usize>,
    null_gid: Option<usize>,
    keys: Vec<Option<T::Native>>,
    aggs: Vec<DirectAggState>,
    output_plan: Vec<OutputAggPlan>,
    count_star_idx: Option<usize>,
    group_indices: Vec<usize>,
    table_tracked_bytes: usize,
    reservation: MemoryReservation,
}

impl<T: ArrowPrimitiveType> DirectIntGroupMap<T>
where
    T::Native: Eq + Hash + IntKeyHash,
{
    fn new(spec: &DirectGroupedSpec, reservation: MemoryReservation) -> Self {
        Self {
            key_index: spec.key_index,
            map: HashMap::new(),
            null_gid: None,
            keys: Vec::new(),
            aggs: spec.aggs.iter().map(DirectAggState::from_spec).collect(),
            output_plan: spec.output_plan.clone(),
            count_star_idx: spec.count_star_idx,
            group_indices: Vec::new(),
            table_tracked_bytes: 0,
            reservation,
        }
    }

    fn tracked_bytes(&self) -> usize {
        let map_bytes = self.map.allocation_size();
        let keys_bytes = self
            .keys
            .capacity()
            .saturating_mul(size_of::<Option<T::Native>>());
        let aggs_bytes = self
            .aggs
            .iter()
            .map(DirectAggState::memory_bytes)
            .fold(0usize, usize::saturating_add);
        let idx_bytes = self
            .group_indices
            .capacity()
            .saturating_mul(size_of::<usize>());
        map_bytes
            .saturating_add(keys_bytes)
            .saturating_add(aggs_bytes)
            .saturating_add(idx_bytes)
    }

    fn update_reservation(&mut self) -> Result<()> {
        let bytes = self.tracked_bytes();
        if bytes > self.table_tracked_bytes {
            self.reservation
                .try_grow(bytes - self.table_tracked_bytes)?;
        } else if bytes < self.table_tracked_bytes {
            self.reservation.shrink(self.table_tracked_bytes - bytes);
        }
        self.table_tracked_bytes = bytes;
        Ok(())
    }

    fn sync_reservation_unconditional(&mut self) {
        let bytes = self.tracked_bytes();
        if bytes > self.table_tracked_bytes {
            self.reservation.grow(bytes - self.table_tracked_bytes);
        } else if bytes < self.table_tracked_bytes {
            self.reservation.shrink(self.table_tracked_bytes - bytes);
        }
        self.table_tracked_bytes = bytes;
    }

    fn drain_to_partial_batch(
        &mut self,
        partial_schema: &SchemaRef,
    ) -> Result<(RecordBatch, Vec<u64>)> {
        let total = self.keys.len();
        let mut key_vals = Vec::with_capacity(total);
        let mut hashes = Vec::with_capacity(total);
        let mut key_nulls = NullBufferBuilder::new(total);
        for key in self.keys.drain(..) {
            match key {
                Some(k) => {
                    key_vals.push(k);
                    hashes.push(k.hash_group_key());
                    key_nulls.append_non_null();
                }
                None => {
                    key_vals.push(T::Native::default());
                    hashes.push(NULL_GROUP_HASH);
                    key_nulls.append_null();
                }
            }
        }
        self.map = HashMap::new();
        self.null_gid = None;
        self.group_indices.clear();
        let cs_arr: Option<Arc<Int64Array>> = if let Some(cs_idx) = self.count_star_idx {
            let DirectAggState::CountStar(c) = &mut self.aggs[cs_idx] else {
                return internal_err!("Expected CountStar at count_star_idx");
            };
            Some(Arc::new(Int64Array::from(std::mem::take(c))))
        } else {
            None
        };
        let mut cols = Vec::with_capacity(partial_schema.fields().len());
        cols.push(Arc::new(PrimitiveArray::<T>::new(
            key_vals.into(),
            key_nulls.finish(),
        )) as ArrayRef);
        for (idx, agg) in self.aggs.iter_mut().enumerate() {
            if Some(idx) == self.count_star_idx {
                cols.push(Arc::clone(cs_arr.as_ref().unwrap()) as ArrayRef);
            } else {
                agg.drain_to_partial_arrays(cs_arr.as_ref(), &mut cols)?;
            }
        }
        self.sync_reservation_unconditional();
        let batch = RecordBatch::try_new(Arc::clone(partial_schema), cols)?;
        Ok((batch, hashes))
    }

    fn merge_partial_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        if batch.num_rows() == 0 {
            return Ok(());
        }
        let key_arr = batch
            .column(0)
            .as_any()
            .downcast_ref::<PrimitiveArray<T>>()
            .ok_or_else(|| {
                datafusion_common::internal_datafusion_err!(
                    "DirectGrouped partial key array type mismatch"
                )
            })?;
        let prev_num_groups = self.keys.len();
        let values = key_arr.values();
        self.group_indices.clear();
        self.group_indices.try_reserve(values.len()).map_err(|e| {
            datafusion_common::exec_datafusion_err!(
                "Morsel DirectGrouped index buffer allocation failed: {e}"
            )
        })?;
        if key_arr.null_count() == 0 {
            for &k in values.iter() {
                let gid = self.intern_key_unchecked(k);
                self.group_indices.push(gid);
            }
        } else {
            let nulls = key_arr.nulls().unwrap();
            for (i, &k) in values.iter().enumerate() {
                let gid = if nulls.is_valid(i) {
                    self.intern_key_unchecked(k)
                } else {
                    self.intern_null()
                };
                self.group_indices.push(gid);
            }
        }
        let num_groups = self.keys.len();
        for agg in &mut self.aggs {
            agg.resize(num_groups);
        }
        let cs_col_idx = self.count_star_idx.map(|cs_idx| {
            let mut offset = 1usize;
            for agg in &self.aggs[..cs_idx] {
                offset += if matches!(agg, DirectAggState::AvgFloat64 { .. }) {
                    2
                } else {
                    1
                };
            }
            offset
        });
        let cs_col = cs_col_idx.map(|idx| batch.column(idx));
        let mut col_offset = 1usize;
        for agg in &mut self.aggs {
            agg.merge_partial_columns(
                batch,
                &mut col_offset,
                &self.group_indices,
                prev_num_groups,
                num_groups,
                cs_col,
            )?;
        }
        self.update_reservation()
    }

    #[inline]
    fn intern_key_unchecked(&mut self, key: T::Native) -> usize {
        if let Some(&gid) = self.map.get(&key) {
            return gid;
        }
        let gid = self.keys.len();
        self.map.insert(key, gid);
        self.keys.push(Some(key));
        gid
    }

    #[inline]
    fn intern_key(&mut self, key: T::Native) -> Result<usize> {
        if let Some(&gid) = self.map.get(&key) {
            return Ok(gid);
        }
        self.insert_new_key(key)
    }

    #[cold]
    fn insert_new_key(&mut self, key: T::Native) -> Result<usize> {
        if self.map.len() == self.map.capacity() {
            let min_alloc = 8 * (size_of::<(T::Native, usize)>() + 1) + 32;
            let next_bytes = self.map.allocation_size().saturating_mul(2).max(min_alloc);
            self.reservation.try_grow(next_bytes)?;
            let gid = self.keys.len();
            self.map.insert(key, gid);
            self.keys.push(Some(key));
            self.reservation.shrink(next_bytes);
            self.update_reservation()?;
            return Ok(gid);
        }
        let gid = self.keys.len();
        self.map.insert(key, gid);
        self.keys.push(Some(key));
        Ok(gid)
    }

    #[inline]
    fn intern_null(&mut self) -> usize {
        if let Some(gid) = self.null_gid {
            return gid;
        }
        let gid = self.keys.len();
        self.null_gid = Some(gid);
        self.keys.push(None);
        gid
    }

    fn add_view_unchecked(&mut self, view: &BatchView) -> Result<()> {
        let key_col = view.batch().column(self.key_index);
        let key_arr = key_col
            .as_any()
            .downcast_ref::<PrimitiveArray<T>>()
            .ok_or_else(|| {
                datafusion_common::internal_datafusion_err!(
                    "Morsel DirectGrouped key type mismatch: expected {:?}, got {:?}",
                    T::DATA_TYPE,
                    key_col.data_type()
                )
            })?;
        let prev_num_groups = self.keys.len();
        let values = key_arr.values();
        let null_count = key_arr.null_count();
        self.group_indices.clear();
        let selected = view.num_rows();
        self.group_indices.try_reserve(selected).map_err(|e| {
            datafusion_common::exec_datafusion_err!(
                "Morsel DirectGrouped index buffer allocation failed: {e}"
            )
        })?;

        match view.selection() {
            Selection::All => {
                if null_count == 0 {
                    for &k in values.iter() {
                        let gid = self.intern_key_unchecked(k);
                        self.group_indices.push(gid);
                    }
                } else if null_count == key_arr.len() {
                    let gid = self.intern_null();
                    self.group_indices.resize(selected, gid);
                } else {
                    let nulls = key_arr.nulls().unwrap();
                    for (i, &k) in values.iter().enumerate() {
                        let gid = if nulls.is_valid(i) {
                            self.intern_key_unchecked(k)
                        } else {
                            self.intern_null()
                        };
                        self.group_indices.push(gid);
                    }
                }
            }
            Selection::Mask { bits, .. } => {
                if null_count == 0 {
                    for i in bits.set_indices() {
                        let gid = self.intern_key_unchecked(values[i]);
                        self.group_indices.push(gid);
                    }
                } else if null_count == key_arr.len() {
                    let gid = self.intern_null();
                    self.group_indices.resize(selected, gid);
                } else {
                    let nulls = key_arr.nulls().unwrap();
                    for i in bits.set_indices() {
                        let gid = if nulls.is_valid(i) {
                            self.intern_key_unchecked(values[i])
                        } else {
                            self.intern_null()
                        };
                        self.group_indices.push(gid);
                    }
                }
            }
            Selection::Indices(indices) => {
                if null_count == 0 {
                    for &idx in indices.values() {
                        let gid = self.intern_key_unchecked(values[idx as usize]);
                        self.group_indices.push(gid);
                    }
                } else if null_count == key_arr.len() {
                    let gid = self.intern_null();
                    self.group_indices.resize(selected, gid);
                } else {
                    let nulls = key_arr.nulls().unwrap();
                    for &idx in indices.values() {
                        let i = idx as usize;
                        let gid = if nulls.is_valid(i) {
                            self.intern_key_unchecked(values[i])
                        } else {
                            self.intern_null()
                        };
                        self.group_indices.push(gid);
                    }
                }
            }
        }

        let num_groups = self.keys.len();
        for agg in &mut self.aggs {
            agg.resize(num_groups);
        }
        for agg in &mut self.aggs {
            agg.update_from_view(view, &self.group_indices, prev_num_groups, num_groups)?;
        }
        Ok(())
    }

    #[cfg(test)]
    fn add_view(&mut self, view: &BatchView) -> Result<()> {
        self.add_view_unchecked(view)?;
        self.update_reservation()
    }

    async fn merge_from(
        &mut self,
        mut other: Self,
        tx: &mpsc::Sender<RecordBatch>,
    ) -> Result<bool> {
        if other.keys.is_empty() {
            return Ok(true);
        }
        if self.keys.is_empty() {
            std::mem::swap(self, &mut other);
            return Ok(true);
        }
        if other.keys.len() > self.keys.len() {
            std::mem::swap(self, &mut other);
        }

        let total_other = other.keys.len();
        for src_gid in 0..total_other {
            if src_gid > 0 && src_gid.is_multiple_of(MERGE_CHUNK_SIZE) {
                if tx.is_closed() {
                    return Ok(false);
                }
                self.update_reservation()?;
                tokio::task::yield_now().await;
            }
            let prev_num_groups = self.keys.len();
            let dst_gid = match other.keys[src_gid] {
                Some(k) => self.intern_key(k)?,
                None => self.intern_null(),
            };
            let num_groups = self.keys.len();
            for (dst_agg, src_agg) in self.aggs.iter_mut().zip(&other.aggs) {
                dst_agg.resize(num_groups);
                dst_agg.merge_group(
                    dst_gid,
                    src_agg,
                    src_gid,
                    prev_num_groups,
                    num_groups,
                )?;
            }
        }
        drop(other);
        self.update_reservation()?;
        Ok(true)
    }

    async fn emit(
        self,
        schema: &SchemaRef,
        result_operators: &[Operator],
        batch_size: usize,
        tx: &mpsc::Sender<RecordBatch>,
        measurement: Option<&RunMetrics>,
    ) -> Result<()> {
        let total = self.keys.len();
        if total == 0 || tx.is_closed() {
            return Ok(());
        }
        let batch_size = batch_size.max(1);
        let mut offset = 0usize;
        while offset < total && !tx.is_closed() {
            let len = (total - offset).min(batch_size);
            let mut key_vals = Vec::with_capacity(len);
            let mut key_nulls = NullBufferBuilder::new(len);
            for key in &self.keys[offset..offset + len] {
                match key {
                    Some(k) => {
                        key_vals.push(*k);
                        key_nulls.append_non_null();
                    }
                    None => {
                        key_vals.push(T::Native::default());
                        key_nulls.append_null();
                    }
                }
            }
            let cs_arr: Option<Arc<Int64Array>> =
                if let Some(cs_idx) = self.count_star_idx {
                    let DirectAggState::CountStar(c) = &self.aggs[cs_idx] else {
                        return internal_err!("Expected CountStar at count_star_idx");
                    };
                    Some(Arc::new(Int64Array::from(c[offset..offset + len].to_vec())))
                } else {
                    None
                };
            let mut phys_cols = Vec::with_capacity(self.aggs.len());
            for (idx, agg) in self.aggs.iter().enumerate() {
                if Some(idx) == self.count_star_idx {
                    phys_cols.push(Arc::clone(cs_arr.as_ref().unwrap()) as ArrayRef);
                } else {
                    phys_cols.push(agg.evaluate_range(offset, len, cs_arr.as_ref())?);
                }
            }
            let mut columns = Vec::with_capacity(1 + self.output_plan.len());
            columns.push(Arc::new(PrimitiveArray::<T>::new(
                key_vals.into(),
                key_nulls.finish(),
            )) as ArrayRef);
            columns.extend(evaluate_output_plan(&self.output_plan, &phys_cols)?);
            let batch = RecordBatch::try_new(Arc::clone(schema), columns)?;
            let batch_bytes = batch.get_array_memory_size();
            let batch_res = self.reservation.new_empty();
            batch_res.try_grow(batch_bytes)?;
            let out = apply_operators(result_operators, batch)?;
            if out.num_rows() > 0 {
                let sent = if let Some(run) = measurement {
                    metrics::send(run, None, tx, out).await
                } else {
                    tx.send(out).await.is_ok()
                };
                drop(batch_res);
                if !sent {
                    break;
                }
            } else {
                drop(batch_res);
            }
            offset += len;
            if offset < total {
                tokio::task::yield_now().await;
            }
        }
        drop(self);
        Ok(())
    }
}

enum DirectGroupedKind {
    Int8(DirectIntGroupMap<Int8Type>),
    Int16(DirectIntGroupMap<Int16Type>),
    Int32(DirectIntGroupMap<Int32Type>),
    Int64(DirectIntGroupMap<Int64Type>),
    UInt8(DirectIntGroupMap<UInt8Type>),
    UInt16(DirectIntGroupMap<UInt16Type>),
    UInt32(DirectIntGroupMap<UInt32Type>),
    UInt64(DirectIntGroupMap<UInt64Type>),
}

macro_rules! dispatch_direct_kind {
    ($kind:expr, $m:ident => $body:expr) => {
        match $kind {
            DirectGroupedKind::Int8($m) => $body,
            DirectGroupedKind::Int16($m) => $body,
            DirectGroupedKind::Int32($m) => $body,
            DirectGroupedKind::Int64($m) => $body,
            DirectGroupedKind::UInt8($m) => $body,
            DirectGroupedKind::UInt16($m) => $body,
            DirectGroupedKind::UInt32($m) => $body,
            DirectGroupedKind::UInt64($m) => $body,
        }
    };
}

impl DirectGroupedKind {
    fn new(spec: &DirectGroupedSpec, reservation: MemoryReservation) -> Self {
        match spec.key_type {
            IntKeyType::Int8 => Self::Int8(DirectIntGroupMap::new(spec, reservation)),
            IntKeyType::Int16 => Self::Int16(DirectIntGroupMap::new(spec, reservation)),
            IntKeyType::Int32 => Self::Int32(DirectIntGroupMap::new(spec, reservation)),
            IntKeyType::Int64 => Self::Int64(DirectIntGroupMap::new(spec, reservation)),
            IntKeyType::UInt8 => Self::UInt8(DirectIntGroupMap::new(spec, reservation)),
            IntKeyType::UInt16 => Self::UInt16(DirectIntGroupMap::new(spec, reservation)),
            IntKeyType::UInt32 => Self::UInt32(DirectIntGroupMap::new(spec, reservation)),
            IntKeyType::UInt64 => Self::UInt64(DirectIntGroupMap::new(spec, reservation)),
        }
    }

    fn len(&self) -> usize {
        dispatch_direct_kind!(self, m => m.keys.len())
    }

    fn is_empty(&self) -> bool {
        dispatch_direct_kind!(self, m => m.keys.is_empty())
    }

    fn table_tracked_bytes(&self) -> usize {
        dispatch_direct_kind!(self, m => m.table_tracked_bytes)
    }

    fn reservation(&self) -> &MemoryReservation {
        dispatch_direct_kind!(self, m => &m.reservation)
    }

    fn into_reservation(self) -> MemoryReservation {
        dispatch_direct_kind!(self, m => m.reservation)
    }

    fn update_reservation(&mut self) -> Result<()> {
        dispatch_direct_kind!(self, m => m.update_reservation())
    }

    fn drain_to_partial_batch(
        &mut self,
        partial_schema: &SchemaRef,
    ) -> Result<(RecordBatch, Vec<u64>)> {
        dispatch_direct_kind!(self, m => m.drain_to_partial_batch(partial_schema))
    }

    fn merge_partial_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        dispatch_direct_kind!(self, m => m.merge_partial_batch(batch))
    }

    fn add_view_unchecked(&mut self, view: &BatchView) -> Result<()> {
        dispatch_direct_kind!(self, m => m.add_view_unchecked(view))
    }

    async fn emit(
        self,
        schema: &SchemaRef,
        result_operators: &[Operator],
        batch_size: usize,
        tx: &mpsc::Sender<RecordBatch>,
        measurement: Option<&RunMetrics>,
    ) -> Result<()> {
        dispatch_direct_kind!(self, m => {
            m.emit(schema, result_operators, batch_size, tx, measurement).await
        })
    }
}

pub(super) struct DirectGroupedState {
    kind: DirectGroupedKind,
    spec: DirectGroupedSpec,
    partial_schema: SchemaRef,
    radix: Option<RadixPartitionStore>,
    spill: Option<SpillContext>,
    direct_scatter: bool,
    rows_since_flush: usize,
    hash_buf: Vec<u64>,
}

impl DirectGroupedState {
    fn new(
        spec: &DirectGroupedSpec,
        partial_schema: SchemaRef,
        reservation: MemoryReservation,
        spill: Option<SpillContext>,
    ) -> Self {
        Self {
            kind: DirectGroupedKind::new(spec, reservation),
            spec: spec.clone(),
            partial_schema,
            radix: None,
            spill,
            direct_scatter: false,
            rows_since_flush: 0,
            hash_buf: Vec::new(),
        }
    }

    #[cfg(test)]
    fn add_view(&mut self, view: &BatchView) -> Result<()> {
        dispatch_direct_kind!(&mut self.kind, m => m.add_view(view))
    }

    fn flush_to_radix(&mut self, tx: &mpsc::Sender<RecordBatch>) -> Result<()> {
        if self.kind.is_empty() {
            return Ok(());
        }
        let (partial, hashes) = self.kind.drain_to_partial_batch(&self.partial_schema)?;
        let radix = self
            .radix
            .get_or_insert_with(RadixPartitionStore::new_default);
        let extra = self.kind.table_tracked_bytes();
        let res = self.kind.reservation();
        radix.push_partial_batch(partial, &hashes, extra, res, self.spill.as_ref(), tx)
    }

    fn spill_on_oom(&mut self, tx: &mpsc::Sender<RecordBatch>) -> Result<()> {
        let Some(spill_ctx) = self.spill.clone() else {
            return Err(DataFusionError::ResourcesExhausted(
                "DirectGrouped spilling is disabled".into(),
            ));
        };
        spill_ctx.mark_spilled();
        if !self.kind.is_empty() {
            self.flush_to_radix(tx)?;
        }
        let radix = self
            .radix
            .get_or_insert_with(RadixPartitionStore::new_default);
        let extra = self.kind.table_tracked_bytes();
        let res = self.kind.reservation();
        radix.spill_all_mem(extra, res, &spill_ctx, tx)?;
        Ok(())
    }

    fn add_view_spillable(
        &mut self,
        view: &BatchView,
        tx: &mpsc::Sender<RecordBatch>,
    ) -> Result<()> {
        if view.is_empty() {
            return Ok(());
        }
        // Adaptive direct radix scatter for low-reduction high-cardinality
        // integer GROUP BY (e.g. TPC-H Q13: 750K distinct c_custkey over 7.5M rows).
        // Skip scattering tiny filtered views (e.g. Q15's 3.7%-selectivity views)
        // which would otherwise fragment into tens of thousands of tiny 4-row slices.
        if self.direct_scatter && view.num_rows() >= 2048 {
            let partial =
                build_direct_raw_partial_batch(view, &self.spec, &self.partial_schema)?;
            hash_int_key_array(
                partial.column(0),
                self.spec.key_type,
                &mut self.hash_buf,
            )?;
            let radix = self
                .radix
                .get_or_insert_with(RadixPartitionStore::new_default);
            let extra = self.kind.table_tracked_bytes();
            let res = self.kind.reservation();
            return radix.push_partial_batch(
                partial,
                &self.hash_buf,
                extra,
                res,
                self.spill.as_ref(),
                tx,
            );
        }

        // Pre-check memory before mutating `kind` when spilling is enabled and
        // groups are already buffered.
        if self.spill.is_some() && !self.kind.is_empty() {
            let per_row = 32usize.saturating_add(self.spec.aggs.len().saturating_mul(16));
            let est = view.num_rows().saturating_mul(per_row);
            let probe = self.kind.reservation().new_empty();
            if probe.try_grow(est).is_err() {
                drop(probe);
                self.spill_on_oom(tx)?;
            }
        }

        self.kind.add_view_unchecked(view)?;
        self.rows_since_flush = self.rows_since_flush.saturating_add(view.num_rows());
        if let Err(err) = self.kind.update_reservation() {
            if matches!(err, DataFusionError::ResourcesExhausted(_))
                && self.spill.is_some()
            {
                self.spill_on_oom(tx)?;
                self.rows_since_flush = 0;
                return Ok(());
            }
            return Err(err);
        }

        let groups = self.kind.len();
        let had_radix = self.radix.is_some();
        let should_flush = if had_radix {
            groups >= PREAGG_FLUSH_GROUPS
                || self.kind.table_tracked_bytes() >= PREAGG_FLUSH_BYTES
        } else {
            groups >= RADIX_TRANSITION_GROUPS
        };
        if should_flush {
            let rows = self.rows_since_flush.max(1);
            // Switch to direct radix scatter only after a full pre-aggregation
            // flush (`had_radix`) shows <30% reduction; checking on the first
            // 4096-group transition misclassifies medium-cardinality uniform
            // keys (e.g. Q15's 50K suppliers over 1.1M rows).
            if had_radix && groups.saturating_mul(10) > rows.saturating_mul(7) {
                self.direct_scatter = true;
            }
            self.rows_since_flush = 0;
            self.flush_to_radix(tx)?;
        }
        Ok(())
    }

    async fn merge_from(
        &mut self,
        mut other: Self,
        tx: &mpsc::Sender<RecordBatch>,
    ) -> Result<bool> {
        if self.radix.is_some()
            || other.radix.is_some()
            || self.kind.len().saturating_add(other.kind.len()) >= RADIX_TRANSITION_GROUPS
        {
            self.flush_to_radix(tx)?;
            other.flush_to_radix(tx)?;
            if tx.is_closed() {
                return Ok(false);
            }
            if let Some(other_radix) = other.radix.take() {
                let radix = self
                    .radix
                    .get_or_insert_with(RadixPartitionStore::new_default);
                let transferred = other.kind.reservation().free();
                self.kind.reservation().grow(transferred);
                radix.combine_from(other_radix);
            }
            return Ok(true);
        }

        match (&mut self.kind, other.kind) {
            (DirectGroupedKind::Int8(a), DirectGroupedKind::Int8(b)) => {
                a.merge_from(b, tx).await
            }
            (DirectGroupedKind::Int16(a), DirectGroupedKind::Int16(b)) => {
                a.merge_from(b, tx).await
            }
            (DirectGroupedKind::Int32(a), DirectGroupedKind::Int32(b)) => {
                a.merge_from(b, tx).await
            }
            (DirectGroupedKind::Int64(a), DirectGroupedKind::Int64(b)) => {
                a.merge_from(b, tx).await
            }
            (DirectGroupedKind::UInt8(a), DirectGroupedKind::UInt8(b)) => {
                a.merge_from(b, tx).await
            }
            (DirectGroupedKind::UInt16(a), DirectGroupedKind::UInt16(b)) => {
                a.merge_from(b, tx).await
            }
            (DirectGroupedKind::UInt32(a), DirectGroupedKind::UInt32(b)) => {
                a.merge_from(b, tx).await
            }
            (DirectGroupedKind::UInt64(a), DirectGroupedKind::UInt64(b)) => {
                a.merge_from(b, tx).await
            }
            _ => internal_err!("Morsel DirectGrouped worker state key type mismatch"),
        }
    }
}

struct GroupedPartitionTable {
    group_values: Box<dyn GroupValues>,
    accumulators: Vec<FastGroupsAccumulator>,
    group_indices: Vec<usize>,
    reservation: MemoryReservation,
}

impl GroupedPartitionTable {
    fn tracked_bytes(&self) -> usize {
        let gv = self.group_values.size();
        let accs = self
            .accumulators
            .iter()
            .map(FastGroupsAccumulator::size)
            .fold(0usize, usize::saturating_add);
        let idx = self
            .group_indices
            .capacity()
            .saturating_mul(size_of::<usize>());
        gv.saturating_add(accs).saturating_add(idx)
    }
}

struct GroupedRadixMerger {
    schema: SchemaRef,
    group_schema: SchemaRef,
    partial_schema: SchemaRef,
    plan: AggregatePlan,
    state_col_counts: Vec<usize>,
    has_float_keys: bool,
}

impl RadixGroupMerger for GroupedRadixMerger {
    type Table = GroupedPartitionTable;

    fn partial_schema(&self) -> &SchemaRef {
        &self.partial_schema
    }

    fn num_group_cols(&self) -> usize {
        self.group_schema.fields().len()
    }

    fn new_table(&self, reservation: MemoryReservation) -> Result<Self::Table> {
        let group_values =
            new_group_values(Arc::clone(&self.group_schema), &GroupClustering::None)?;
        let accumulators = self
            .plan
            .phys_aggrs
            .iter()
            .map(FastGroupsAccumulator::from_spec)
            .collect::<Result<Vec<_>>>()?;
        Ok(GroupedPartitionTable {
            group_values,
            accumulators,
            group_indices: Vec::new(),
            reservation,
        })
    }

    fn merge_partial_batch(
        &self,
        table: &mut Self::Table,
        batch: &RecordBatch,
    ) -> Result<()> {
        if batch.num_rows() == 0 {
            return Ok(());
        }
        let n_keys = self.num_group_cols();
        let key_cols = &batch.columns()[..n_keys];
        let prev_groups = table.group_values.len();
        table
            .group_values
            .intern(key_cols, &mut table.group_indices)?;
        let total_groups = table.group_values.len();
        let cs_col_idx = self.plan.count_star_idx.map(|cs_idx| {
            n_keys
                + self.state_col_counts[..cs_idx]
                    .iter()
                    .copied()
                    .sum::<usize>()
        });
        let cs_col_ref = cs_col_idx.map(|idx| batch.column(idx));
        let cs_vals = cs_col_ref.map(|col| {
            col.as_any()
                .downcast_ref::<Int64Array>()
                .expect("CountStar partial column must be Int64")
                .values()
                .as_ref()
        });
        let mut offset = n_keys;
        for (acc, &n_state) in table.accumulators.iter_mut().zip(&self.state_col_counts) {
            let state_cols = &batch.columns()[offset..offset + n_state];
            let same_as_cs = cs_col_ref
                .is_some_and(|cs| n_state == 1 && Arc::ptr_eq(&state_cols[0], cs));
            acc.merge_state(
                state_cols,
                &table.group_indices,
                prev_groups,
                total_groups,
                cs_vals,
                same_as_cs,
            )?;
            offset += n_state;
        }
        let bytes = table.tracked_bytes();
        table.reservation.try_resize(bytes)?;
        Ok(())
    }

    fn hash_partial_batch(
        &self,
        batch: &RecordBatch,
        hashes: &mut Vec<u64>,
    ) -> Result<()> {
        let n_keys = self.num_group_cols();
        hash_group_columns(&batch.columns()[..n_keys], self.has_float_keys, hashes)
    }

    fn table_is_empty(&self, table: &Self::Table) -> bool {
        table.group_values.is_empty()
    }

    fn emit_table<'a>(
        &'a self,
        mut table: Self::Table,
        result_operators: &'a [Operator],
        batch_size: usize,
        tx: &'a mpsc::Sender<RecordBatch>,
        measurement: Option<&'a RunMetrics>,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            if table.group_values.is_empty() || tx.is_closed() {
                return Ok(());
            }
            let mut columns = table.group_values.emit(EmitTo::All)?;
            columns.extend(evaluate_fast_groups(
                &mut table.accumulators,
                self.plan.count_star_idx,
                &self.plan.output_plan,
            )?);
            let batch = RecordBatch::try_new(Arc::clone(&self.schema), columns)?;
            drop(table);
            emit_finalized_batch(batch, result_operators, batch_size, tx, measurement)
                .await?;
            Ok(())
        })
    }
}

struct DirectGroupedRadixMerger {
    schema: SchemaRef,
    partial_schema: SchemaRef,
    spec: DirectGroupedSpec,
}

impl RadixGroupMerger for DirectGroupedRadixMerger {
    type Table = DirectGroupedKind;

    fn partial_schema(&self) -> &SchemaRef {
        &self.partial_schema
    }

    fn num_group_cols(&self) -> usize {
        1
    }

    fn new_table(&self, reservation: MemoryReservation) -> Result<Self::Table> {
        Ok(DirectGroupedKind::new(&self.spec, reservation))
    }

    fn merge_partial_batch(
        &self,
        table: &mut Self::Table,
        batch: &RecordBatch,
    ) -> Result<()> {
        table.merge_partial_batch(batch)
    }

    fn hash_partial_batch(
        &self,
        batch: &RecordBatch,
        hashes: &mut Vec<u64>,
    ) -> Result<()> {
        hash_int_key_array(batch.column(0), self.spec.key_type, hashes)
    }

    fn table_is_empty(&self, table: &Self::Table) -> bool {
        table.is_empty()
    }

    fn emit_table<'a>(
        &'a self,
        table: Self::Table,
        result_operators: &'a [Operator],
        batch_size: usize,
        tx: &'a mpsc::Sender<RecordBatch>,
        measurement: Option<&'a RunMetrics>,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<()>> + Send + 'a>> {
        Box::pin(async move {
            table
                .emit(&self.schema, result_operators, batch_size, tx, measurement)
                .await
        })
    }
}

pub(super) struct FinalizedAggregate {
    schema: SchemaRef,
    group_schema: Option<SchemaRef>,
    plan: AggregatePlan,
    state_col_counts: Vec<usize>,
    has_float_keys: bool,
    direct_grouped: Option<DirectGroupedSpec>,
    state: AggregateState,
}

impl FinalizedAggregate {
    pub(super) fn has_spilled(&self) -> bool {
        match &self.state {
            AggregateState::Ungrouped(_) => false,
            AggregateState::Grouped(s) => s.radix.as_ref().is_some_and(|r| r.spilled),
            AggregateState::DirectGrouped(s) => {
                s.radix.as_ref().is_some_and(|r| r.spilled)
            }
        }
    }

    pub(super) async fn emit(
        self,
        result_operators: ResultOpsRef,
        workers: usize,
        batch_size: usize,
        memory_pool: &Arc<dyn MemoryPool>,
        tx: mpsc::Sender<RecordBatch>,
        measurement: Option<Arc<RunMetrics>>,
    ) -> Result<()> {
        match self.state {
            AggregateState::Ungrouped(state) => {
                state
                    .emit(
                        &self.schema,
                        &self.plan.output_plan,
                        result_operators.ops(),
                        &tx,
                        measurement.as_deref(),
                    )
                    .await
            }
            AggregateState::Grouped(mut state) => {
                if state.radix.is_some() {
                    state.flush_to_radix(&tx)?;
                    let store = state.radix.take().expect("radix store");
                    let spill = state.spill.take();
                    let partial_schema = state.partial_schema;
                    let input_res = state.reservation;
                    let merger = Arc::new(GroupedRadixMerger {
                        schema: self.schema,
                        group_schema: self
                            .group_schema
                            .expect("grouped aggregate group_schema"),
                        partial_schema,
                        plan: self.plan,
                        state_col_counts: self.state_col_counts,
                        has_float_keys: self.has_float_keys,
                    });
                    return finalize_and_emit_radix(
                        merger,
                        store,
                        input_res,
                        RadixFinalizeConfig {
                            workers,
                            batch_size,
                            memory_pool,
                            spill,
                            consumer_name: "MorselAggregateFinalize",
                            result_operators,
                            measurement,
                        },
                        tx,
                    )
                    .await;
                }
                state
                    .emit(
                        &self.schema,
                        &self.plan.output_plan,
                        result_operators.ops(),
                        batch_size,
                        &tx,
                        measurement.as_deref(),
                    )
                    .await
            }
            AggregateState::DirectGrouped(mut state) => {
                if state.radix.is_some() {
                    state.flush_to_radix(&tx)?;
                    let store = state.radix.take().expect("radix store");
                    let spill = state.spill.take();
                    let partial_schema = state.partial_schema;
                    let input_res = state.kind.into_reservation();
                    let merger = Arc::new(DirectGroupedRadixMerger {
                        schema: self.schema,
                        partial_schema,
                        spec: self.direct_grouped.expect("direct grouped spec"),
                    });
                    return finalize_and_emit_radix(
                        merger,
                        store,
                        input_res,
                        RadixFinalizeConfig {
                            workers,
                            batch_size,
                            memory_pool,
                            spill,
                            consumer_name: "MorselAggregateFinalize",
                            result_operators,
                            measurement,
                        },
                        tx,
                    )
                    .await;
                }
                state
                    .kind
                    .emit(
                        &self.schema,
                        result_operators.ops(),
                        batch_size,
                        &tx,
                        measurement.as_deref(),
                    )
                    .await
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{Int64Array, StringArray};
    use arrow::datatypes::{DataType, Field};
    use datafusion_common_runtime::SpawnedTask;
    use datafusion_execution::memory_pool::{
        GreedyMemoryPool, MemoryConsumer, MemoryPool, UnboundedMemoryPool,
    };
    use datafusion_expr::{LogicalPlan, col};
    use datafusion_functions_aggregate::expr_fn::{avg, count, max, min, sum};

    fn test_reservation(pool: &Arc<dyn MemoryPool>) -> MemoryReservation {
        MemoryConsumer::new("test_morsel_aggregate").register(pool)
    }

    #[tokio::test]
    async fn grouped_memory_reservation_tracks_growth_merge_peak_and_cleans_up()
    -> Result<()> {
        let input_schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Utf8, true),
            Field::new("v", DataType::Int64, true),
        ]));
        let ctx = crate::prelude::SessionContext::new();
        let mem = Arc::new(crate::datasource::MemTable::try_new(
            Arc::clone(&input_schema),
            vec![vec![]],
        )?);
        ctx.register_table("t", mem)?;
        let logical = ctx
            .table("t")
            .await?
            .aggregate(
                vec![col("k")],
                vec![
                    sum(col("v")),
                    min(col("v")),
                    max(col("v")),
                    avg(datafusion_expr::cast(col("v"), DataType::Float64)),
                    count(col("v")),
                ],
            )?
            .into_unoptimized_plan();
        let LogicalPlan::Aggregate(aggregate) = logical else {
            unreachable!();
        };
        let sink = AggregateSink::try_new(&aggregate, &input_schema, &ctx.state())?
            .expect("valid aggregate sink");

        let make_batch = |start: usize, count: usize| -> Result<RecordBatch> {
            let keys: Vec<String> = (start..start + count)
                .map(|i| format!("key-{i:08}-padding-bytes"))
                .collect();
            let vals: Vec<i64> = (start..start + count).map(|i| i as i64).collect();
            Ok(RecordBatch::try_new(
                Arc::clone(&input_schema),
                vec![
                    Arc::new(StringArray::from(keys)),
                    Arc::new(Int64Array::from(vals)),
                ],
            )?)
        };

        // Measure exact tracked bytes of one 1024-group worker state.
        let unbounded: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
        let mut probe = sink.new_state(test_reservation(&unbounded))?;
        probe.add_batch(&make_batch(0, 1024)?)?;
        let one_worker_bytes = unbounded.reserved();
        assert!(one_worker_bytes > 0);
        drop(probe);
        assert_eq!(unbounded.reserved(), 0);

        // 1. Worker accumulation fails cleanly under a tight memory budget.
        let tight: Arc<dyn MemoryPool> =
            Arc::new(GreedyMemoryPool::new(one_worker_bytes / 2));
        {
            let mut state = sink.new_state(test_reservation(&tight))?;
            let err = state.add_batch(&make_batch(0, 1024)?).unwrap_err();
            assert!(
                matches!(err, DataFusionError::ResourcesExhausted(_)),
                "expected ResourcesExhausted, got: {err}"
            );
        }
        assert_eq!(tight.reserved(), 0);

        // 2. Merge peak fails cleanly when two disjoint worker tables fit individually,
        //    but growing the merged table while holding emitted state exceeds the budget.
        let merge_budget = one_worker_bytes * 2 + one_worker_bytes / 4;
        let merge_pool: Arc<dyn MemoryPool> =
            Arc::new(GreedyMemoryPool::new(merge_budget));
        let (tx, _rx) = mpsc::channel(2);
        {
            let mut a = sink.new_state(test_reservation(&merge_pool))?;
            let mut b = sink.new_state(test_reservation(&merge_pool))?;
            a.add_batch(&make_batch(0, 1024)?)?;
            b.add_batch(&make_batch(1024, 1024)?)?;
            assert_eq!(merge_pool.reserved(), 2 * one_worker_bytes);
            let err = a.merge_from(b, &tx).await.unwrap_err();
            assert!(
                matches!(err, DataFusionError::ResourcesExhausted(_)),
                "expected ResourcesExhausted at merge peak, got: {err}"
            );
        }
        assert_eq!(merge_pool.reserved(), 0);

        // 3. Full accumulate + merge + bounded emit releases all memory.
        let ample: Arc<dyn MemoryPool> =
            Arc::new(GreedyMemoryPool::new(one_worker_bytes * 16));
        let (tx, mut rx) = mpsc::channel(4);
        let mut a = sink.new_state(test_reservation(&ample))?;
        let mut b = sink.new_state(test_reservation(&ample))?;
        a.add_batch(&make_batch(0, 200)?)?;
        b.add_batch(&make_batch(100, 200)?)?;
        assert!(a.merge_from(b, &tx).await?);
        assert!(ample.reserved() > 0);
        let finalized = sink.finalize(a);
        let emit_pool = Arc::clone(&ample);
        let emit = SpawnedTask::spawn(async move {
            finalized
                .emit(ResultOpsRef::Empty, 1, 64, &emit_pool, tx, None)
                .await
        });
        let mut total_rows = 0;
        while let Some(batch) = rx.recv().await {
            assert!(batch.num_rows() <= 64);
            total_rows += batch.num_rows();
        }
        emit.join_unwind().await.unwrap()?;
        assert_eq!(total_rows, 300);
        assert_eq!(ample.reserved(), 0);
        Ok(())
    }

    #[tokio::test]
    async fn direct_grouped_selection_views_and_memory_limits() -> Result<()> {
        let input_schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int32, true),
            Field::new("v", DataType::Int64, true),
            Field::new("f", DataType::Float64, true),
        ]));
        let ctx = crate::prelude::SessionContext::new();
        let mem = Arc::new(crate::datasource::MemTable::try_new(
            Arc::clone(&input_schema),
            vec![vec![]],
        )?);
        ctx.register_table("t", mem)?;
        let logical = ctx
            .table("t")
            .await?
            .aggregate(
                vec![col("k")],
                vec![
                    count(datafusion_expr::lit(1i64)),
                    count(col("v")),
                    sum(col("v")),
                    min(col("v")),
                    max(col("v")),
                    avg(col("f")),
                ],
            )?
            .into_unoptimized_plan();
        let LogicalPlan::Aggregate(aggregate) = logical else {
            unreachable!();
        };
        let sink = AggregateSink::try_new(&aggregate, &input_schema, &ctx.state())?
            .expect("valid aggregate sink");
        assert!(sink.direct_grouped.is_some());

        let base = RecordBatch::try_new(
            Arc::clone(&input_schema),
            vec![
                Arc::new(Int32Array::from(vec![
                    Some(99),
                    Some(1),
                    None,
                    Some(1),
                    Some(2),
                    None,
                    Some(2),
                    Some(88),
                ])),
                Arc::new(Int64Array::from(vec![
                    Some(999),
                    Some(10),
                    None,
                    Some(20),
                    None,
                    Some(5),
                    Some(30),
                    Some(888),
                ])),
                Arc::new(Float64Array::from(vec![
                    Some(99.0),
                    Some(1.0),
                    Some(2.0),
                    Some(3.0),
                    None,
                    Some(4.0),
                    Some(6.0),
                    Some(88.0),
                ])),
            ],
        )?;
        // Slice offset=1, len=6 -> keys: [1, NULL, 1, 2, NULL, 2]
        let sliced = base.slice(1, 6);
        let mask = arrow::array::BooleanArray::from(vec![
            Some(true),  // idx 0: k=1, v=10, f=1.0
            Some(true),  // idx 1: k=NULL, v=NULL, f=2.0
            None,        // idx 2: rejected (NULL predicate)
            Some(true),  // idx 3: k=2, v=NULL, f=NULL
            Some(true),  // idx 4: k=NULL, v=5, f=4.0
            Some(false), // idx 5: rejected
        ]);
        let view_mask = BatchView::from_boolean_array(sliced.clone(), &mask)?;
        let view_idx = view_mask.clone().with_indices_selection();

        let pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
        let mut s_mask = sink.new_state(test_reservation(&pool))?;
        let mut s_idx = sink.new_state(test_reservation(&pool))?;
        s_mask.add_view(&view_mask)?;
        s_idx.add_view(&view_idx)?;

        // Merge both workers -> doubles counts/sums while preserving min/max/avg.
        let (tx, mut rx) = mpsc::channel(4);
        assert!(s_mask.merge_from(s_idx, &tx).await?);
        let finalized = sink.finalize(s_mask);
        let emit_pool = Arc::clone(&pool);
        let emit = SpawnedTask::spawn(async move {
            finalized
                .emit(ResultOpsRef::Empty, 1, 2, &emit_pool, tx, None)
                .await
        });
        let mut results = std::collections::BTreeMap::new();
        while let Some(batch) = rx.recv().await {
            let ks = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap();
            let c_star = batch
                .column(1)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let c_val = batch
                .column(2)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let sums = batch
                .column(3)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let mins = batch
                .column(4)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let maxs = batch
                .column(5)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap();
            let avgs = batch
                .column(6)
                .as_any()
                .downcast_ref::<Float64Array>()
                .unwrap();
            for i in 0..batch.num_rows() {
                let k = ks.is_valid(i).then(|| ks.value(i));
                results.insert(
                    k,
                    (
                        c_star.value(i),
                        c_val.value(i),
                        sums.is_valid(i).then(|| sums.value(i)),
                        mins.is_valid(i).then(|| mins.value(i)),
                        maxs.is_valid(i).then(|| maxs.value(i)),
                        avgs.is_valid(i).then(|| avgs.value(i)),
                    ),
                );
            }
        }
        emit.join_unwind().await.unwrap()?;
        assert_eq!(pool.reserved(), 0);
        assert_eq!(results.len(), 3);
        assert_eq!(
            results[&Some(1)],
            (2, 2, Some(20), Some(10), Some(10), Some(1.0))
        );
        assert_eq!(results[&Some(2)], (2, 0, None, None, None, None));
        assert_eq!(
            results[&None],
            (4, 2, Some(10), Some(5), Some(5), Some(3.0))
        );

        // Tight memory pool fails cleanly and releases all memory on drop.
        let tight: Arc<dyn MemoryPool> = Arc::new(GreedyMemoryPool::new(1024));
        {
            let big = RecordBatch::try_new(
                Arc::clone(&input_schema),
                vec![
                    Arc::new(Int32Array::from_iter_values(0..1024)),
                    Arc::new(Int64Array::from_iter_values(0..1024)),
                    Arc::new(Float64Array::from_iter_values((0..1024).map(|i| i as f64))),
                ],
            )?;
            let mut state = sink.new_state(test_reservation(&tight))?;
            let err = state.add_view(&BatchView::all(big)).unwrap_err();
            assert!(
                matches!(err, DataFusionError::ResourcesExhausted(_)),
                "{err}"
            );
        }
        assert_eq!(tight.reserved(), 0);
        Ok(())
    }

    #[tokio::test]
    async fn direct_grouped_count_col_handles_null_type_and_dict_logical_nulls_across_selections()
    -> Result<()> {
        let dict_type =
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8));
        let input_schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int32, true),
            Field::new("n", DataType::Null, true),
            Field::new("d", dict_type, true),
        ]));
        let ctx = crate::prelude::SessionContext::new();
        let mem = Arc::new(crate::datasource::MemTable::try_new(
            Arc::clone(&input_schema),
            vec![vec![]],
        )?);
        ctx.register_table("t", mem)?;
        let logical = ctx
            .table("t")
            .await?
            .aggregate(vec![col("k")], vec![count(col("n")), count(col("d"))])?
            .into_unoptimized_plan();
        let LogicalPlan::Aggregate(aggregate) = logical else {
            unreachable!();
        };
        let sink = AggregateSink::try_new(&aggregate, &input_schema, &ctx.state())?
            .expect("valid aggregate sink");
        assert!(sink.direct_grouped.is_some());

        let dict_values = Arc::new(StringArray::from(vec![Some("a"), None, Some("b")]));
        let dict_keys = Int32Array::from(vec![
            Some(0), // k=1, "a" -> counted
            Some(1), // k=1, NULL in dict -> NOT counted
            None,    // k=1, NULL key -> NOT counted
            Some(2), // k=2, "b" -> counted
            Some(1), // k=2, NULL in dict -> NOT counted
        ]);
        let dict_arr =
            arrow::array::DictionaryArray::<Int32Type>::try_new(dict_keys, dict_values)?;
        let batch = RecordBatch::try_new(
            Arc::clone(&input_schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 1, 1, 2, 2])),
                Arc::new(arrow::array::NullArray::new(5)),
                Arc::new(dict_arr),
            ],
        )?;

        let mask = arrow::array::BooleanArray::from(vec![true, true, false, true, true]);
        let view_all = BatchView::all(batch.clone());
        let view_mask = BatchView::from_boolean_array(batch, &mask)?;
        let view_idx = view_mask.clone().with_indices_selection();

        let pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
        for (label, view) in [
            ("all", view_all),
            ("mask", view_mask),
            ("indices", view_idx),
        ] {
            let mut state = sink.new_state(test_reservation(&pool))?;
            state.add_view(&view)?;
            let (tx, mut rx) = mpsc::channel(4);
            let finalized = sink.finalize(state);
            let emit_pool = Arc::clone(&pool);
            let emit = SpawnedTask::spawn(async move {
                finalized
                    .emit(ResultOpsRef::Empty, 1, 16, &emit_pool, tx, None)
                    .await
            });
            let mut map = std::collections::BTreeMap::new();
            while let Some(b) = rx.recv().await {
                let ks = b.column(0).as_any().downcast_ref::<Int32Array>().unwrap();
                let cn = b.column(1).as_any().downcast_ref::<Int64Array>().unwrap();
                let cd = b.column(2).as_any().downcast_ref::<Int64Array>().unwrap();
                for i in 0..b.num_rows() {
                    map.insert(ks.value(i), (cn.value(i), cd.value(i)));
                }
            }
            emit.join_unwind().await.unwrap()?;
            assert_eq!(map[&1], (0, 1), "{label}");
            assert_eq!(map[&2], (0, 1), "{label}");
        }
        assert_eq!(pool.reserved(), 0);
        Ok(())
    }

    #[tokio::test]
    async fn cse_and_avg_sharing_reduces_physical_accumulators() -> Result<()> {
        // TPC-H Q1 non-nullable schema: 4 SUM + 3 AVG + COUNT(*) = 8 output aggs,
        // should collapse to 6 physical accumulators (5 SUMs + 1 COUNT(*)).
        let input_schema = Arc::new(Schema::new(vec![
            Field::new("rf", DataType::Utf8View, false),
            Field::new("ls", DataType::Utf8View, false),
            Field::new("qty", DataType::Decimal128(15, 2), false),
            Field::new("price", DataType::Decimal128(15, 2), false),
            Field::new("disc", DataType::Decimal128(15, 2), false),
            Field::new("tax", DataType::Decimal128(15, 2), false),
        ]));
        let ctx = crate::prelude::SessionContext::new();
        let mem = Arc::new(crate::datasource::MemTable::try_new(
            Arc::clone(&input_schema),
            vec![vec![]],
        )?);
        ctx.register_table("t", mem)?;
        let logical = ctx
            .table("t")
            .await?
            .aggregate(
                vec![col("rf"), col("ls")],
                vec![
                    sum(col("qty")).alias("s_qty"),
                    sum(col("qty")).alias("s_qty_dup"),
                    sum(col("price")).alias("s_price"),
                    avg(col("qty")).alias("a_qty"),
                    avg(col("qty")).alias("a_qty_dup"),
                    avg(col("price")).alias("a_price"),
                    avg(col("disc")).alias("a_disc"),
                    count(datafusion_expr::lit(1i64)).alias("c1"),
                    count(datafusion_expr::lit(2i64)).alias("c2"),
                ],
            )?
            .into_unoptimized_plan();
        let LogicalPlan::Aggregate(aggregate) = logical else {
            unreachable!();
        };
        let sink = AggregateSink::try_new(&aggregate, &input_schema, &ctx.state())?
            .expect("valid aggregate sink");
        // 9 outputs -> 4 physical accumulators:
        // sum(qty), sum(price), sum(disc), count(*)!
        assert_eq!(sink.aggr_exprs.len(), 9);
        assert_eq!(sink.physical_accumulator_count(), 4);
        Ok(())
    }

    #[tokio::test]
    async fn tpch_q1_q6_q15_fast_paths_fire() -> Result<()> {
        let input_schema = Arc::new(Schema::new(vec![
            Field::new("l_suppkey", DataType::Int64, true),
            Field::new("l_quantity", DataType::Decimal128(15, 2), true),
            Field::new("l_extendedprice", DataType::Decimal128(15, 2), true),
            Field::new("l_discount", DataType::Decimal128(15, 2), true),
            Field::new("l_tax", DataType::Decimal128(15, 2), true),
            Field::new("l_returnflag", DataType::Utf8View, true),
            Field::new("l_linestatus", DataType::Utf8View, true),
        ]));
        let ctx = crate::prelude::SessionContext::new();
        let mem = Arc::new(crate::datasource::MemTable::try_new(
            Arc::clone(&input_schema),
            vec![vec![]],
        )?);
        ctx.register_table("lineitem", mem)?;

        // Q1: check AggregateSink plan after DataFusion logical optimizer
        let q1_plan = ctx
            .sql(
                "SELECT l_returnflag, l_linestatus,
                    SUM(l_quantity), SUM(l_extendedprice),
                    SUM(l_extendedprice * (1 - l_discount)),
                    SUM(l_extendedprice * (1 - l_discount) * (1 + l_tax)),
                    AVG(l_quantity), AVG(l_extendedprice), AVG(l_discount),
                    COUNT(*)
                 FROM lineitem GROUP BY l_returnflag, l_linestatus",
            )
            .await?
            .into_optimized_plan()?;
        let mut q1_agg = None;
        let _ = datafusion_common::tree_node::TreeNode::apply(&q1_plan, |node| {
            if let LogicalPlan::Aggregate(agg) = node {
                q1_agg = Some(agg.clone());
            }
            Ok(TreeNodeRecursion::Continue)
        });
        let q1_agg = q1_agg.expect("q1 agg");
        let q1_in_schema: SchemaRef = Arc::new(q1_agg.input.schema().as_arrow().clone());
        let q1_sink = AggregateSink::try_new(&q1_agg, &q1_in_schema, &ctx.state())?
            .expect("q1 sink");
        for (i, phys) in q1_sink.plan.phys_aggrs.iter().enumerate() {
            if let PhysAggKind::SumDecimal128 { fast_input, .. } = &phys.kind {
                assert!(
                    fast_input.is_some(),
                    "Q1 SumDecimal128[{i}] did not match fast_input! arg={:?}",
                    phys.args[0]
                );
            }
        }
        // Check Q15 uses DirectGroupedSpec with fused Decimal128Input
        let q15_plan = ctx
            .sql(
                "SELECT l_suppkey, SUM(l_extendedprice * (1 - l_discount))
                 FROM lineitem GROUP BY l_suppkey",
            )
            .await?
            .into_optimized_plan()?;
        let mut q15_agg = None;
        let _ = datafusion_common::tree_node::TreeNode::apply(&q15_plan, |node| {
            if let LogicalPlan::Aggregate(agg) = node {
                q15_agg = Some(agg.clone());
            }
            Ok(TreeNodeRecursion::Continue)
        });
        let q15_agg = q15_agg.expect("q15 agg");
        let q15_in_schema: SchemaRef =
            Arc::new(q15_agg.input.schema().as_arrow().clone());
        let q15_sink = AggregateSink::try_new(&q15_agg, &q15_in_schema, &ctx.state())?
            .expect("q15 sink");
        assert!(
            q15_sink.direct_grouped.is_some(),
            "Q15 should use DirectGroupedSpec; phys={:?}",
            q15_sink.plan.phys_aggrs
        );

        // Decimal128(26, 2) must NOT share AVG's Decimal256 sum with SUM's i128 sum.
        let wide_schema = Arc::new(Schema::new(vec![
            Field::new("d25", DataType::Decimal128(25, 2), false),
            Field::new("d26", DataType::Decimal128(26, 2), false),
        ]));
        let wide_ctx = crate::prelude::SessionContext::new();
        wide_ctx.register_table(
            "w",
            Arc::new(crate::datasource::MemTable::try_new(
                Arc::clone(&wide_schema),
                vec![vec![]],
            )?),
        )?;
        let wide_plan = wide_ctx
            .sql("SELECT SUM(d25), AVG(d25), SUM(d26), AVG(d26), COUNT(*) FROM w")
            .await?
            .into_optimized_plan()?;
        let mut wide_agg = None;
        let _ = datafusion_common::tree_node::TreeNode::apply(&wide_plan, |node| {
            if let LogicalPlan::Aggregate(agg) = node {
                wide_agg = Some(agg.clone());
            }
            Ok(TreeNodeRecursion::Continue)
        });
        let wide_agg = wide_agg.expect("wide agg");
        let wide_in_schema: SchemaRef =
            Arc::new(wide_agg.input.schema().as_arrow().clone());
        let wide_sink =
            AggregateSink::try_new(&wide_agg, &wide_in_schema, &wide_ctx.state())?
                .expect("wide sink");
        // d25 shares SUM + COUNT(*) -> 1 sum + 1 count(*)
        // d26 keeps separate SUM + Generic AVG -> 1 sum + 1 generic avg
        // Total: 4 physical accumulators for 5 outputs!
        assert_eq!(wide_sink.physical_accumulator_count(), 4);
        assert!(matches!(
            wide_sink.plan.output_plan[1],
            OutputAggPlan::AvgFromSumCount { .. }
        ));
        assert!(matches!(
            wide_sink.plan.output_plan[3],
            OutputAggPlan::Direct { .. }
        ));

        // InlineTwoUtf8ViewGroupValues seamlessly promotes on >12-byte strings while
        // preserving group IDs and NULL handling.
        let gv_schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Utf8View, true),
            Field::new("b", DataType::Utf8View, true),
        ]));
        let mut gv = FastGroupValues::new(gv_schema)?;
        assert!(matches!(gv, FastGroupValues::TwoUtf8View(_)));
        let b1: Vec<ArrayRef> = vec![
            Arc::new(StringViewArray::from(vec![
                Some("A"),
                None,
                Some("A"),
                Some("N"),
            ])),
            Arc::new(StringViewArray::from(vec![
                Some("F"),
                Some("O"),
                Some("F"),
                None,
            ])),
        ];
        let mut gids = Vec::new();
        gv.intern(&b1, &mut gids)?;
        assert_eq!(gids, vec![0, 1, 0, 2]);
        assert!(matches!(gv, FastGroupValues::TwoUtf8View(_)));

        let b2: Vec<ArrayRef> = vec![
            Arc::new(StringViewArray::from(vec![
                Some("A"),
                Some("this-is-a-non-inlined-long-string"),
                None,
            ])),
            Arc::new(StringViewArray::from(vec![Some("F"), Some("O"), Some("O")])),
        ];
        gv.intern(&b2, &mut gids)?;
        assert!(matches!(gv, FastGroupValues::Generic(_)));
        assert_eq!(gids, vec![0, 3, 1]);
        assert_eq!(gv.len(), 4);
        Ok(())
    }
}
