// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::fmt;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, RecordBatch};
use arrow::datatypes::{Schema, SchemaRef};
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::{Result, ScalarValue, internal_err};
use datafusion_execution::memory_pool::MemoryReservation;
use datafusion_expr::expr::{AggregateFunctionParams, physical_name};
use datafusion_expr::logical_plan::Aggregate;
use datafusion_expr::physical_planning_context::PhysicalPlanningContext;
use datafusion_expr::{Accumulator, EmitTo, Expr, GroupsAccumulator};
use datafusion_functions_aggregate::average::Avg;
use datafusion_functions_aggregate::count::Count;
use datafusion_functions_aggregate::min_max::{Max, Min};
use datafusion_functions_aggregate::sum::Sum;
use datafusion_physical_expr::aggregate::{
    AggregateFunctionExpr, LoweredAggregateBuilder,
};
use datafusion_physical_expr::expressions::Literal;
use datafusion_physical_expr::{
    GroupsAccumulatorAdapter, PhysicalExpr, create_physical_expr,
};
use datafusion_physical_plan::aggregates::PhysicalGroupBy;
use datafusion_physical_plan::aggregates::group_values::{GroupValues, new_group_values};
use datafusion_physical_plan::aggregates::order::GroupClustering;
use datafusion_physical_plan::execution_plan::apply_expression_roots;
use datafusion_session::Session;
use tokio::sync::mpsc;

use super::metrics::{self, RunMetrics};
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
        if *distinct {
            return Ok(Err(format!(
                "DISTINCT aggregates are not supported in native morsel aggregates: {expr}"
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

#[derive(Debug)]
pub(super) struct AggregateSink {
    schema: SchemaRef,
    group_schema: Option<SchemaRef>,
    group_exprs: Vec<Arc<dyn PhysicalExpr>>,
    group_names: Vec<String>,
    aggr_exprs: Vec<Arc<AggregateFunctionExpr>>,
    #[cfg(test)]
    pub(super) finalize_gate: Option<super::tests::FinalizeGate>,
}

impl AggregateSink {
    pub(super) fn try_new(
        aggregate: &Aggregate,
        input_schema: &SchemaRef,
        session: &dyn Session,
    ) -> Result<std::result::Result<Self, String>> {
        let logical_input_schema = aggregate.input.schema();
        let props = session.execution_props();
        let planning_ctx = PhysicalPlanningContext::default();

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
                        &planning_ctx,
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
                &planning_ctx,
            )
            .build()?;
            if lowered.filter.is_some()
                || !lowered.order_bys.is_empty()
                || lowered.aggregate.is_distinct()
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

        Ok(Ok(Self {
            schema,
            group_schema,
            group_exprs,
            group_names,
            aggr_exprs,
            #[cfg(test)]
            finalize_gate: None,
        }))
    }

    pub(super) fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    pub(super) fn new_state(
        &self,
        reservation: MemoryReservation,
    ) -> Result<AggregateState> {
        let aggr_args = self
            .aggr_exprs
            .iter()
            .map(|aggr| {
                aggr.expressions()
                    .into_iter()
                    .map(ArgEvaluator::new)
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        match &self.group_schema {
            None => {
                let accumulators = self
                    .aggr_exprs
                    .iter()
                    .map(|aggr| aggr.create_accumulator())
                    .collect::<Result<Vec<_>>>()?;
                Ok(AggregateState::Ungrouped(UngroupedState {
                    accumulators,
                    aggr_args,
                    has_rows: false,
                    reservation,
                }))
            }
            Some(group_schema) => {
                let group_values =
                    new_group_values(Arc::clone(group_schema), &GroupClustering::None)?;
                let group_exprs = self
                    .group_exprs
                    .iter()
                    .cloned()
                    .map(ArgEvaluator::new)
                    .collect();
                let accumulators = self
                    .aggr_exprs
                    .iter()
                    .map(|aggr| create_groups_accumulator(aggr, false))
                    .collect::<Result<Vec<_>>>()?;
                Ok(AggregateState::Grouped(GroupedState {
                    group_values,
                    group_exprs,
                    accumulators,
                    aggr_args,
                    group_indices: Vec::new(),
                    reservation,
                }))
            }
        }
    }

    pub(super) fn finalize(&self, combined: AggregateState) -> FinalizedAggregate {
        FinalizedAggregate {
            schema: Arc::clone(&self.schema),
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
}

impl AggregateState {
    pub(super) fn add_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        if batch.num_rows() == 0 {
            return Ok(());
        }
        match self {
            Self::Ungrouped(state) => state.add_batch(batch),
            Self::Grouped(state) => state.add_batch(batch),
        }
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
            _ => internal_err!("Morsel aggregate worker state kind mismatch"),
        }
    }
}

pub(super) struct UngroupedState {
    accumulators: Vec<Box<dyn Accumulator>>,
    aggr_args: Vec<Vec<ArgEvaluator>>,
    has_rows: bool,
    reservation: MemoryReservation,
}

impl UngroupedState {
    fn tracked_bytes(&self) -> usize {
        self.accumulators
            .iter()
            .map(|acc| acc.size())
            .fold(0usize, usize::saturating_add)
    }

    fn update_reservation(&mut self) -> Result<()> {
        let bytes = self.tracked_bytes();
        if bytes != self.reservation.size() {
            self.reservation.try_resize(bytes)?;
        }
        Ok(())
    }

    fn add_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        for (acc, args) in self.accumulators.iter_mut().zip(&mut self.aggr_args) {
            let arrays = args
                .iter_mut()
                .map(|arg| arg.evaluate(batch))
                .collect::<Result<Vec<_>>>()?;
            acc.update_batch(&arrays)?;
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
            let states: Vec<ArrayRef> = src
                .state()?
                .into_iter()
                .map(|scalar| scalar.to_array_of_size(1))
                .collect::<Result<_>>()?;
            dst.merge_batch(&states)?;
        }
        self.has_rows = true;
        drop(other);
        self.update_reservation()
    }

    async fn emit(
        mut self,
        schema: &SchemaRef,
        result_operators: &[Operator],
        tx: &mpsc::Sender<RecordBatch>,
        measurement: Option<&RunMetrics>,
    ) -> Result<()> {
        if tx.is_closed() {
            return Ok(());
        }
        let columns: Vec<ArrayRef> = self
            .accumulators
            .iter_mut()
            .map(|acc| acc.evaluate()?.to_array_of_size(1))
            .collect::<Result<_>>()?;
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

pub(super) struct GroupedState {
    group_values: Box<dyn GroupValues>,
    group_exprs: Vec<ArgEvaluator>,
    accumulators: Vec<Box<dyn GroupsAccumulator>>,
    aggr_args: Vec<Vec<ArgEvaluator>>,
    group_indices: Vec<usize>,
    reservation: MemoryReservation,
}

impl GroupedState {
    fn tracked_bytes(&self) -> usize {
        let gv = self.group_values.size();
        let accs = self
            .accumulators
            .iter()
            .map(|acc| acc.size())
            .fold(0usize, usize::saturating_add);
        let indices = self
            .group_indices
            .capacity()
            .saturating_mul(size_of::<usize>());
        gv.saturating_add(accs).saturating_add(indices)
    }

    fn update_reservation(&mut self) -> Result<()> {
        let bytes = self.tracked_bytes();
        if bytes != self.reservation.size() {
            self.reservation.try_resize(bytes)?;
        }
        Ok(())
    }

    fn add_batch(&mut self, batch: &RecordBatch) -> Result<()> {
        let group_arrays = self
            .group_exprs
            .iter_mut()
            .map(|expr| expr.evaluate(batch))
            .collect::<Result<Vec<_>>>()?;
        self.group_values
            .intern(&group_arrays, &mut self.group_indices)?;
        let total_num_groups = self.group_values.len();
        self.update_reservation()?;

        for (acc, args) in self.accumulators.iter_mut().zip(&mut self.aggr_args) {
            let arrays = args
                .iter_mut()
                .map(|arg| arg.evaluate(batch))
                .collect::<Result<Vec<_>>>()?;
            acc.update_batch(&arrays, &self.group_indices, None, total_num_groups)?;
        }
        self.update_reservation()
    }

    async fn merge_from(
        &mut self,
        mut other: Self,
        tx: &mpsc::Sender<RecordBatch>,
    ) -> Result<bool> {
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
        let state_arrays: Vec<Vec<ArrayRef>> = other
            .accumulators
            .iter_mut()
            .map(|acc| acc.state(EmitTo::All))
            .collect::<Result<_>>()?;
        let emitted_bytes = arrays_memory_size(&group_arrays).saturating_add(
            state_arrays
                .iter()
                .map(|states| arrays_memory_size(states))
                .fold(0usize, usize::saturating_add),
        );
        let emitted_reservation = other.reservation.new_empty();
        drop(other);
        emitted_reservation.try_grow(emitted_bytes)?;

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
            self.group_values
                .intern(&chunk_keys, &mut self.group_indices)?;
            let total_num_groups = self.group_values.len();
            self.update_reservation()?;

            for (acc, states) in self.accumulators.iter_mut().zip(&state_arrays) {
                let chunk_states: Vec<ArrayRef> = states
                    .iter()
                    .map(|array| array.slice(offset, len))
                    .collect();
                acc.merge_batch(&chunk_states, &self.group_indices, total_num_groups)?;
            }
            self.update_reservation()?;

            offset += len;
            if offset < total_other {
                tokio::task::yield_now().await;
            }
        }
        drop(group_arrays);
        drop(state_arrays);
        drop(emitted_reservation);
        Ok(true)
    }

    async fn emit(
        mut self,
        schema: &SchemaRef,
        result_operators: &[Operator],
        batch_size: usize,
        tx: &mpsc::Sender<RecordBatch>,
        measurement: Option<&RunMetrics>,
    ) -> Result<()> {
        if self.group_values.is_empty() || tx.is_closed() {
            return Ok(());
        }
        let batch_size = batch_size.max(1);
        let mut columns = self.group_values.emit(EmitTo::All)?;
        for acc in &mut self.accumulators {
            columns.push(acc.evaluate(EmitTo::All)?);
        }
        let batch = RecordBatch::try_new(Arc::clone(schema), columns)?;
        let batch_bytes = batch.get_array_memory_size();
        let batch_reservation = self.reservation.new_empty();
        drop(self);
        batch_reservation.try_grow(batch_bytes)?;

        let total_rows = batch.num_rows();
        let mut offset = 0usize;
        while offset < total_rows && !tx.is_closed() {
            let len = (total_rows - offset).min(batch_size);
            let slice = batch.slice(offset, len);
            let out = apply_operators(result_operators, slice)?;
            if out.num_rows() > 0 {
                let sent = if let Some(run) = measurement {
                    metrics::send(run, None, tx, out).await
                } else {
                    tx.send(out).await.is_ok()
                };
                if !sent {
                    break;
                }
            }
            offset += len;
            if offset < total_rows {
                tokio::task::yield_now().await;
            }
        }
        drop(batch);
        drop(batch_reservation);
        Ok(())
    }
}

pub(super) struct FinalizedAggregate {
    schema: SchemaRef,
    state: AggregateState,
}

impl FinalizedAggregate {
    pub(super) async fn emit(
        self,
        result_operators: &[Operator],
        batch_size: usize,
        tx: mpsc::Sender<RecordBatch>,
        measurement: Option<&RunMetrics>,
    ) -> Result<()> {
        match self.state {
            AggregateState::Ungrouped(state) => {
                state
                    .emit(&self.schema, result_operators, &tx, measurement)
                    .await
            }
            AggregateState::Grouped(state) => {
                state
                    .emit(&self.schema, result_operators, batch_size, &tx, measurement)
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
                matches!(
                    err,
                    datafusion_common::DataFusionError::ResourcesExhausted(_)
                ),
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
                matches!(
                    err,
                    datafusion_common::DataFusionError::ResourcesExhausted(_)
                ),
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
        let emit =
            SpawnedTask::spawn(async move { finalized.emit(&[], 64, tx, None).await });
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
}
