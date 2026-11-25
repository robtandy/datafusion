// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::fmt;
use std::hash::Hash;
use std::sync::Arc;

use arrow::array::types::{
    Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt32Type,
    UInt64Type,
};
use arrow::array::{
    Array, ArrayRef, ArrowPrimitiveType, Float64Array, Int32Array, Int64Array,
    NullBufferBuilder, PrimitiveArray, RecordBatch,
};
use arrow::datatypes::{DataType, Schema, SchemaRef};
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::{HashMap, Result, ScalarValue, internal_err};
use datafusion_execution::memory_pool::MemoryReservation;
use datafusion_expr::expr::{AggregateFunctionParams, physical_name};
use datafusion_expr::logical_plan::Aggregate;
use datafusion_expr::physical_planning_context::PhysicalPlanningContext;
use datafusion_expr::utils::COUNT_STAR_EXPANSION;
use datafusion_expr::{Accumulator, EmitTo, Expr, GroupsAccumulator};
use datafusion_functions_aggregate::average::Avg;
use datafusion_functions_aggregate::count::Count;
use datafusion_functions_aggregate::min_max::{Max, Min};
use datafusion_functions_aggregate::sum::Sum;
use datafusion_physical_expr::aggregate::{
    AggregateFunctionExpr, LoweredAggregateBuilder,
};
use datafusion_physical_expr::expressions::{Column, Literal};
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
use super::group_count::IntKeyType;
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

#[derive(Clone, Copy, Debug)]
enum DirectAggSpec {
    CountStar,
    CountCol(usize),
    SumInt64(usize),
    SumInt32(usize),
    SumFloat64(usize),
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
}

fn try_build_direct_grouped_spec(
    group_exprs: &[Arc<dyn PhysicalExpr>],
    aggr_exprs: &[Arc<AggregateFunctionExpr>],
    input_schema: &SchemaRef,
) -> Option<DirectGroupedSpec> {
    let [group_expr] = group_exprs else {
        return None;
    };
    let key_col = group_expr.downcast_ref::<Column>()?;
    let key_index = key_col.index();
    let key_type = IntKeyType::from_data_type(input_schema.field(key_index).data_type())?;

    let mut aggs = Vec::with_capacity(aggr_exprs.len());
    for aggr in aggr_exprs {
        if aggr.is_distinct() {
            return None;
        }
        let inner = aggr.fun().inner();
        let args = aggr.expressions();
        let out_dt = aggr.field().data_type().clone();
        let spec = if inner.is::<Count>() {
            match args.as_slice() {
                [arg]
                    if arg
                        .downcast_ref::<Literal>()
                        .is_some_and(|l| l.value() == &COUNT_STAR_EXPANSION) =>
                {
                    DirectAggSpec::CountStar
                }
                [arg] => DirectAggSpec::CountCol(arg.downcast_ref::<Column>()?.index()),
                _ => return None,
            }
        } else if inner.is::<Sum>() {
            let [arg] = args.as_slice() else {
                return None;
            };
            let idx = arg.downcast_ref::<Column>()?.index();
            match (input_schema.field(idx).data_type(), &out_dt) {
                (DataType::Int64, DataType::Int64) => DirectAggSpec::SumInt64(idx),
                (DataType::Int32, DataType::Int64) => DirectAggSpec::SumInt32(idx),
                (DataType::Float64, DataType::Float64) => DirectAggSpec::SumFloat64(idx),
                _ => return None,
            }
        } else if inner.is::<Avg>() {
            let [arg] = args.as_slice() else {
                return None;
            };
            let idx = arg.downcast_ref::<Column>()?.index();
            match (input_schema.field(idx).data_type(), &out_dt) {
                (DataType::Float64, DataType::Float64) => DirectAggSpec::AvgFloat64(idx),
                _ => return None,
            }
        } else if inner.is::<Min>() {
            let [arg] = args.as_slice() else {
                return None;
            };
            let idx = arg.downcast_ref::<Column>()?.index();
            match (input_schema.field(idx).data_type(), &out_dt) {
                (DataType::Int64, DataType::Int64) => DirectAggSpec::MinInt64(idx),
                (DataType::Int32, DataType::Int32) => DirectAggSpec::MinInt32(idx),
                _ => return None,
            }
        } else if inner.is::<Max>() {
            let [arg] = args.as_slice() else {
                return None;
            };
            let idx = arg.downcast_ref::<Column>()?.index();
            match (input_schema.field(idx).data_type(), &out_dt) {
                (DataType::Int64, DataType::Int64) => DirectAggSpec::MaxInt64(idx),
                (DataType::Int32, DataType::Int32) => DirectAggSpec::MaxInt32(idx),
                _ => return None,
            }
        } else {
            return None;
        };
        aggs.push(spec);
    }

    Some(DirectGroupedSpec {
        key_index,
        key_type,
        aggs,
    })
}

#[derive(Debug)]
pub(super) struct AggregateSink {
    schema: SchemaRef,
    group_schema: Option<SchemaRef>,
    group_exprs: Vec<Arc<dyn PhysicalExpr>>,
    group_names: Vec<String>,
    aggr_exprs: Vec<Arc<AggregateFunctionExpr>>,
    needed_columns: Vec<usize>,
    rebound_group_exprs: Vec<Arc<dyn PhysicalExpr>>,
    rebound_aggr_args: Vec<Vec<Arc<dyn PhysicalExpr>>>,
    direct_grouped: Option<DirectGroupedSpec>,
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

        let mut all_exprs: Vec<Arc<dyn PhysicalExpr>> = group_exprs.clone();
        for aggr in &aggr_exprs {
            all_exprs.extend(aggr.expressions());
        }
        let needed_columns = collect_expr_column_indices(&all_exprs);
        let rebound_group_exprs = group_exprs
            .iter()
            .map(|e| remap_expr_columns(Arc::clone(e), &needed_columns))
            .collect::<Result<Vec<_>>>()?;
        let rebound_aggr_args = aggr_exprs
            .iter()
            .map(|aggr| {
                aggr.expressions()
                    .into_iter()
                    .map(|e| remap_expr_columns(e, &needed_columns))
                    .collect::<Result<Vec<_>>>()
            })
            .collect::<Result<Vec<_>>>()?;
        let direct_grouped =
            try_build_direct_grouped_spec(&group_exprs, &aggr_exprs, input_schema);

        Ok(Ok(Self {
            schema,
            group_schema,
            group_exprs,
            group_names,
            aggr_exprs,
            needed_columns,
            rebound_group_exprs,
            rebound_aggr_args,
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
        let direct_grouped =
            try_build_direct_grouped_spec(&group_exprs, &[], input_schema);
        Ok(Ok(Self {
            schema: Arc::clone(input_schema),
            group_schema: Some(Arc::clone(input_schema)),
            group_exprs,
            group_names,
            aggr_exprs: Vec::new(),
            needed_columns,
            rebound_group_exprs,
            rebound_aggr_args: Vec::new(),
            direct_grouped,
            #[cfg(test)]
            finalize_gate: None,
        }))
    }

    pub(super) fn is_grouped(&self) -> bool {
        self.group_schema.is_some()
    }

    pub(super) fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    pub(super) fn new_state(
        &self,
        reservation: MemoryReservation,
    ) -> Result<AggregateState> {
        if let Some(spec) = &self.direct_grouped {
            return Ok(AggregateState::DirectGrouped(DirectGroupedState::new(
                spec,
                reservation,
            )));
        }

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
        let rebound_aggr_args = self
            .rebound_aggr_args
            .iter()
            .map(|args| args.iter().cloned().map(ArgEvaluator::new).collect())
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
                    needed_columns: self.needed_columns.clone(),
                    rebound_aggr_args,
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
                let rebound_group_exprs = self
                    .rebound_group_exprs
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
                    rebound_group_exprs,
                    accumulators,
                    aggr_args,
                    rebound_aggr_args,
                    needed_columns: self.needed_columns.clone(),
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
    DirectGrouped(DirectGroupedState),
}

impl AggregateState {
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

pub(super) struct UngroupedState {
    accumulators: Vec<Box<dyn Accumulator>>,
    aggr_args: Vec<Vec<ArgEvaluator>>,
    needed_columns: Vec<usize>,
    rebound_aggr_args: Vec<Vec<ArgEvaluator>>,
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

    fn add_view(&mut self, view: &BatchView) -> Result<()> {
        if view.is_all() {
            return self.add_batch(view.batch());
        }
        let compact = view.materialize_columns(&self.needed_columns)?;
        for (acc, args) in self
            .accumulators
            .iter_mut()
            .zip(&mut self.rebound_aggr_args)
        {
            let arrays = args
                .iter_mut()
                .map(|arg| arg.evaluate(&compact))
                .collect::<Result<Vec<_>>>()?;
            acc.update_batch(&arrays)?;
        }
        self.has_rows = true;
        self.update_reservation()
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
    rebound_group_exprs: Vec<ArgEvaluator>,
    accumulators: Vec<Box<dyn GroupsAccumulator>>,
    aggr_args: Vec<Vec<ArgEvaluator>>,
    rebound_aggr_args: Vec<Vec<ArgEvaluator>>,
    needed_columns: Vec<usize>,
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

    fn add_view(&mut self, view: &BatchView) -> Result<()> {
        if view.is_all() {
            return self.add_batch(view.batch());
        }
        let compact = view.materialize_columns(&self.needed_columns)?;
        let group_arrays = self
            .rebound_group_exprs
            .iter_mut()
            .map(|expr| expr.evaluate(&compact))
            .collect::<Result<Vec<_>>>()?;
        self.group_values
            .intern(&group_arrays, &mut self.group_indices)?;
        let total_num_groups = self.group_values.len();
        self.update_reservation()?;

        for (acc, args) in self
            .accumulators
            .iter_mut()
            .zip(&mut self.rebound_aggr_args)
        {
            let arrays = args
                .iter_mut()
                .map(|arg| arg.evaluate(&compact))
                .collect::<Result<Vec<_>>>()?;
            acc.update_batch(&arrays, &self.group_indices, None, total_num_groups)?;
        }
        self.update_reservation()
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

enum DirectAggState {
    CountStar(Vec<i64>),
    CountCol {
        col: usize,
        counts: Vec<i64>,
    },
    SumInt64 {
        col: usize,
        sums: Vec<i64>,
        has_val: Vec<bool>,
    },
    SumInt32 {
        col: usize,
        sums: Vec<i64>,
        has_val: Vec<bool>,
    },
    SumFloat64 {
        col: usize,
        sums: Vec<f64>,
        has_val: Vec<bool>,
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
    fn from_spec(spec: DirectAggSpec) -> Self {
        match spec {
            DirectAggSpec::CountStar => Self::CountStar(Vec::new()),
            DirectAggSpec::CountCol(col) => Self::CountCol {
                col,
                counts: Vec::new(),
            },
            DirectAggSpec::SumInt64(col) => Self::SumInt64 {
                col,
                sums: Vec::new(),
                has_val: Vec::new(),
            },
            DirectAggSpec::SumInt32(col) => Self::SumInt32 {
                col,
                sums: Vec::new(),
                has_val: Vec::new(),
            },
            DirectAggSpec::SumFloat64(col) => Self::SumFloat64 {
                col,
                sums: Vec::new(),
                has_val: Vec::new(),
            },
            DirectAggSpec::AvgFloat64(col) => Self::AvgFloat64 {
                col,
                sums: Vec::new(),
                counts: Vec::new(),
            },
            DirectAggSpec::MinInt64(col) => Self::MinInt64 {
                col,
                vals: Vec::new(),
                has_val: Vec::new(),
            },
            DirectAggSpec::MinInt32(col) => Self::MinInt32 {
                col,
                vals: Vec::new(),
                has_val: Vec::new(),
            },
            DirectAggSpec::MaxInt64(col) => Self::MaxInt64 {
                col,
                vals: Vec::new(),
                has_val: Vec::new(),
            },
            DirectAggSpec::MaxInt32(col) => Self::MaxInt32 {
                col,
                vals: Vec::new(),
                has_val: Vec::new(),
            },
        }
    }

    fn resize(&mut self, num_groups: usize) {
        match self {
            Self::CountStar(c) | Self::CountCol { counts: c, .. } => {
                c.resize(num_groups, 0)
            }
            Self::SumInt64 { sums, has_val, .. }
            | Self::SumInt32 { sums, has_val, .. } => {
                sums.resize(num_groups, 0);
                has_val.resize(num_groups, false);
            }
            Self::SumFloat64 { sums, has_val, .. } => {
                sums.resize(num_groups, 0.0);
                has_val.resize(num_groups, false);
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
            Self::CountStar(c) | Self::CountCol { counts: c, .. } => {
                c.capacity().saturating_mul(size_of::<i64>())
            }
            Self::SumInt64 { sums, has_val, .. }
            | Self::SumInt32 { sums, has_val, .. } => sums
                .capacity()
                .saturating_mul(size_of::<i64>())
                .saturating_add(has_val.capacity()),
            Self::SumFloat64 { sums, has_val, .. } => sums
                .capacity()
                .saturating_mul(size_of::<f64>())
                .saturating_add(has_val.capacity()),
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
            Self::CountCol { col, counts } => {
                let array = batch.column(*col);
                if matches!(array.data_type(), DataType::Null) {
                    // Every element of a NullArray is logically NULL.
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
                    for_each_selected_row!(row, gid => {
                        sums[gid] = sums[gid].wrapping_add(values[row]);
                        has_val[gid] = true;
                    });
                } else if arr.null_count() < arr.len() {
                    let nulls = arr.nulls().unwrap();
                    for_each_selected_row!(row, gid => {
                        if nulls.is_valid(row) {
                            sums[gid] = sums[gid].wrapping_add(values[row]);
                            has_val[gid] = true;
                        }
                    });
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
                    for_each_selected_row!(row, gid => {
                        sums[gid] = sums[gid].wrapping_add(values[row] as i64);
                        has_val[gid] = true;
                    });
                } else if arr.null_count() < arr.len() {
                    let nulls = arr.nulls().unwrap();
                    for_each_selected_row!(row, gid => {
                        if nulls.is_valid(row) {
                            sums[gid] = sums[gid].wrapping_add(values[row] as i64);
                            has_val[gid] = true;
                        }
                    });
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
                    for_each_selected_row!(row, gid => {
                        sums[gid] += values[row];
                        has_val[gid] = true;
                    });
                } else if arr.null_count() < arr.len() {
                    let nulls = arr.nulls().unwrap();
                    for_each_selected_row!(row, gid => {
                        if nulls.is_valid(row) {
                            sums[gid] += values[row];
                            has_val[gid] = true;
                        }
                    });
                }
            }
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

    fn merge_group(&mut self, dst_gid: usize, src: &Self, src_gid: usize) -> Result<()> {
        match (self, src) {
            (Self::CountStar(dst), Self::CountStar(s))
            | (Self::CountCol { counts: dst, .. }, Self::CountCol { counts: s, .. }) => {
                dst[dst_gid] = dst[dst_gid]
                    .checked_add(s[src_gid])
                    .ok_or_else(count_overflow)?;
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
                if sh[src_gid] {
                    ds[dst_gid] = ds[dst_gid].wrapping_add(ss[src_gid]);
                    dh[dst_gid] = true;
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
                if sh[src_gid] {
                    ds[dst_gid] += ss[src_gid];
                    dh[dst_gid] = true;
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

    fn evaluate_range(&self, start: usize, len: usize) -> ArrayRef {
        let end = start + len;
        match self {
            Self::CountStar(c) | Self::CountCol { counts: c, .. } => {
                Arc::new(Int64Array::from(c[start..end].to_vec()))
            }
            Self::SumInt64 { sums, has_val, .. }
            | Self::SumInt32 { sums, has_val, .. } => {
                let mut nulls = NullBufferBuilder::new(len);
                for &v in &has_val[start..end] {
                    nulls.append(v);
                }
                Arc::new(Int64Array::new(
                    sums[start..end].to_vec().into(),
                    nulls.finish(),
                ))
            }
            Self::SumFloat64 { sums, has_val, .. } => {
                let mut nulls = NullBufferBuilder::new(len);
                for &v in &has_val[start..end] {
                    nulls.append(v);
                }
                Arc::new(Float64Array::new(
                    sums[start..end].to_vec().into(),
                    nulls.finish(),
                ))
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
        }
    }
}

struct DirectIntGroupMap<T: ArrowPrimitiveType>
where
    T::Native: Eq + Hash,
{
    key_index: usize,
    map: HashMap<T::Native, usize>,
    null_gid: Option<usize>,
    keys: Vec<Option<T::Native>>,
    aggs: Vec<DirectAggState>,
    group_indices: Vec<usize>,
    reservation: MemoryReservation,
}

impl<T: ArrowPrimitiveType> DirectIntGroupMap<T>
where
    T::Native: Eq + Hash,
{
    fn new(
        key_index: usize,
        agg_specs: &[DirectAggSpec],
        reservation: MemoryReservation,
    ) -> Self {
        Self {
            key_index,
            map: HashMap::new(),
            null_gid: None,
            keys: Vec::new(),
            aggs: agg_specs
                .iter()
                .copied()
                .map(DirectAggState::from_spec)
                .collect(),
            group_indices: Vec::new(),
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
        if bytes != self.reservation.size() {
            self.reservation.try_resize(bytes)?;
        }
        Ok(())
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

    fn add_view(&mut self, view: &BatchView) -> Result<()> {
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
                        let gid = self.intern_key(k)?;
                        self.group_indices.push(gid);
                    }
                } else if null_count == key_arr.len() {
                    let gid = self.intern_null();
                    self.group_indices.resize(selected, gid);
                } else {
                    let nulls = key_arr.nulls().unwrap();
                    for (i, &k) in values.iter().enumerate() {
                        let gid = if nulls.is_valid(i) {
                            self.intern_key(k)?
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
                        let gid = self.intern_key(values[i])?;
                        self.group_indices.push(gid);
                    }
                } else if null_count == key_arr.len() {
                    let gid = self.intern_null();
                    self.group_indices.resize(selected, gid);
                } else {
                    let nulls = key_arr.nulls().unwrap();
                    for i in bits.set_indices() {
                        let gid = if nulls.is_valid(i) {
                            self.intern_key(values[i])?
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
                        let gid = self.intern_key(values[idx as usize])?;
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
                            self.intern_key(values[i])?
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
        self.update_reservation()?;

        for agg in &mut self.aggs {
            agg.update_from_view(view, &self.group_indices)?;
        }
        Ok(())
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
            let dst_gid = match other.keys[src_gid] {
                Some(k) => self.intern_key(k)?,
                None => self.intern_null(),
            };
            let num_groups = self.keys.len();
            for (dst_agg, src_agg) in self.aggs.iter_mut().zip(&other.aggs) {
                dst_agg.resize(num_groups);
                dst_agg.merge_group(dst_gid, src_agg, src_gid)?;
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
            let mut columns = Vec::with_capacity(1 + self.aggs.len());
            columns.push(Arc::new(PrimitiveArray::<T>::new(
                key_vals.into(),
                key_nulls.finish(),
            )) as ArrayRef);
            for agg in &self.aggs {
                columns.push(agg.evaluate_range(offset, len));
            }
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

pub(super) struct DirectGroupedState(DirectGroupedKind);

impl DirectGroupedState {
    fn new(spec: &DirectGroupedSpec, reservation: MemoryReservation) -> Self {
        Self(match spec.key_type {
            IntKeyType::Int8 => DirectGroupedKind::Int8(DirectIntGroupMap::new(
                spec.key_index,
                &spec.aggs,
                reservation,
            )),
            IntKeyType::Int16 => DirectGroupedKind::Int16(DirectIntGroupMap::new(
                spec.key_index,
                &spec.aggs,
                reservation,
            )),
            IntKeyType::Int32 => DirectGroupedKind::Int32(DirectIntGroupMap::new(
                spec.key_index,
                &spec.aggs,
                reservation,
            )),
            IntKeyType::Int64 => DirectGroupedKind::Int64(DirectIntGroupMap::new(
                spec.key_index,
                &spec.aggs,
                reservation,
            )),
            IntKeyType::UInt8 => DirectGroupedKind::UInt8(DirectIntGroupMap::new(
                spec.key_index,
                &spec.aggs,
                reservation,
            )),
            IntKeyType::UInt16 => DirectGroupedKind::UInt16(DirectIntGroupMap::new(
                spec.key_index,
                &spec.aggs,
                reservation,
            )),
            IntKeyType::UInt32 => DirectGroupedKind::UInt32(DirectIntGroupMap::new(
                spec.key_index,
                &spec.aggs,
                reservation,
            )),
            IntKeyType::UInt64 => DirectGroupedKind::UInt64(DirectIntGroupMap::new(
                spec.key_index,
                &spec.aggs,
                reservation,
            )),
        })
    }

    fn add_view(&mut self, view: &BatchView) -> Result<()> {
        match &mut self.0 {
            DirectGroupedKind::Int8(m) => m.add_view(view),
            DirectGroupedKind::Int16(m) => m.add_view(view),
            DirectGroupedKind::Int32(m) => m.add_view(view),
            DirectGroupedKind::Int64(m) => m.add_view(view),
            DirectGroupedKind::UInt8(m) => m.add_view(view),
            DirectGroupedKind::UInt16(m) => m.add_view(view),
            DirectGroupedKind::UInt32(m) => m.add_view(view),
            DirectGroupedKind::UInt64(m) => m.add_view(view),
        }
    }

    async fn merge_from(
        &mut self,
        other: Self,
        tx: &mpsc::Sender<RecordBatch>,
    ) -> Result<bool> {
        match (&mut self.0, other.0) {
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

    async fn emit(
        self,
        schema: &SchemaRef,
        result_operators: &[Operator],
        batch_size: usize,
        tx: &mpsc::Sender<RecordBatch>,
        measurement: Option<&RunMetrics>,
    ) -> Result<()> {
        match self.0 {
            DirectGroupedKind::Int8(m) => {
                m.emit(schema, result_operators, batch_size, tx, measurement)
                    .await
            }
            DirectGroupedKind::Int16(m) => {
                m.emit(schema, result_operators, batch_size, tx, measurement)
                    .await
            }
            DirectGroupedKind::Int32(m) => {
                m.emit(schema, result_operators, batch_size, tx, measurement)
                    .await
            }
            DirectGroupedKind::Int64(m) => {
                m.emit(schema, result_operators, batch_size, tx, measurement)
                    .await
            }
            DirectGroupedKind::UInt8(m) => {
                m.emit(schema, result_operators, batch_size, tx, measurement)
                    .await
            }
            DirectGroupedKind::UInt16(m) => {
                m.emit(schema, result_operators, batch_size, tx, measurement)
                    .await
            }
            DirectGroupedKind::UInt32(m) => {
                m.emit(schema, result_operators, batch_size, tx, measurement)
                    .await
            }
            DirectGroupedKind::UInt64(m) => {
                m.emit(schema, result_operators, batch_size, tx, measurement)
                    .await
            }
        }
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
            AggregateState::DirectGrouped(state) => {
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
        let emit =
            SpawnedTask::spawn(async move { finalized.emit(&[], 2, tx, None).await });
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
                matches!(
                    err,
                    datafusion_common::DataFusionError::ResourcesExhausted(_)
                ),
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
            let emit =
                SpawnedTask::spawn(
                    async move { finalized.emit(&[], 16, tx, None).await },
                );
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
}
