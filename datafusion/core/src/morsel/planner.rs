// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Recursive logical-to-morsel-DAG planner (`MorselQueryPlanner`).

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;

use arrow::array::{RecordBatch, RecordBatchOptions, new_null_array};
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
#[cfg(feature = "parquet")]
use datafusion_common::DFSchemaRef;
use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion};
use datafusion_common::utils::combine_limit;
use datafusion_common::{
    DFSchema, JoinType, NullEquality, Result, ScalarValue, internal_err, plan_err,
    project_schema,
};
use datafusion_expr::expr::physical_name;
use datafusion_expr::logical_plan::{
    Aggregate, Distinct, EmptyRelation, Join, Limit, MaterializedCteId,
    MaterializedCtePlan, Sort, TableScan, Values,
};
use datafusion_expr::physical_planning_context::{
    PhysicalPlanningContext, ScalarSubqueryResults, SubqueryIndex,
};
use datafusion_expr::{ColumnarValue, Expr, FetchType, LogicalPlan, SkipType, Subquery};
use datafusion_physical_expr::aggregate::LoweredAggregateBuilder;
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::projection::{ProjectionExpr, ProjectionExprs};
use datafusion_physical_expr::{
    EquivalenceProperties, LexOrdering, create_physical_expr, create_physical_sort_exprs,
};
use datafusion_physical_plan::joins::utils::build_join_schema;
use datafusion_session::Session;

use crate::datasource::{MemTable, source_as_provider};

use super::aggregate::{self, AggregateSink};
use super::count::{self, CountSink};
use super::dag::{
    ScalarSubquerySpec, ScalarSubqueryState, StageId, StageSource, StageSpec,
    collect_pipeline_stage_deps, collect_scalar_subqueries_post_order,
};
use super::group_count::{self, GroupCountSink, IntKeyType};
use super::hash_join::{
    BuildSide, CompiledJoinFilter, HashJoinBuildConfig, HashJoinBuildSink, HashJoinProbe,
    HashJoinProbeConfig, JoinOutputCol, ProbeJoinKind, is_supported_join_key_type,
};
#[cfg(feature = "parquet")]
use super::parquet::{ParquetScan, bind_pipeline_dynamic_join_filters};
use super::sort::{SortSink, TopNSink};
use super::{
    BoxFuture, Lowering, MemorySource, MorselExec, MorselQueryPlanner, Operator,
    Pipeline, PostSort, SideEstimate, Sink, Source, assign_join_build_ids,
    choose_build_side, estimate_join_side_size, estimate_side_size, prune_filter_input,
    result_capacity, sink_result_capacity, supported_expr, unsupported_provider_reason,
};

#[cfg(feature = "parquet")]
struct ParquetScanMeta {
    filters: Vec<Expr>,
    projected_schema: DFSchemaRef,
}

struct OpenPipeline {
    source: Source,
    #[cfg(feature = "parquet")]
    parquet_scan_meta: Option<ParquetScanMeta>,
    operators: Vec<Operator>,
    schema: SchemaRef,
    ordering: Option<LexOrdering>,
}

struct BlockingPipeline {
    source: Source,
    operators: Vec<Operator>,
    sink: Sink,
    schema: SchemaRef,
    ordering: Option<LexOrdering>,
    estimate: SideEstimate,
}

enum LoweredNode {
    Open(Box<OpenPipeline>),
    Blocking(Box<BlockingPipeline>),
}

#[derive(Clone, Copy, Default)]
struct OpenOptimizeHint {
    is_count_star: bool,
    skip_filter_pruning: bool,
    is_general_agg: bool,
    has_sort: bool,
    always_prune_filter: bool,
}

impl LoweredNode {
    fn schema(&self) -> &SchemaRef {
        match self {
            Self::Open(open) => &open.schema,
            Self::Blocking(blocking) => &blocking.schema,
        }
    }
}

#[derive(Clone)]
struct CteBinding {
    stage_id: StageId,
    schema: SchemaRef,
    ordered: bool,
    ordering: Option<LexOrdering>,
}

struct StageBinding {
    stage_id: StageId,
    schema: SchemaRef,
    ordered: bool,
    ordering: Option<LexOrdering>,
}

struct DagBuilder<'a> {
    workers: NonZeroUsize,
    session: &'a dyn Session,
    planning_ctx: PhysicalPlanningContext,
    stages: Vec<StageSpec>,
    cte_stages: HashMap<MaterializedCteId, CteBinding>,
    lowered_subqueries: Vec<bool>,
    scalar_subquery_specs: Vec<ScalarSubquerySpec>,
    next_filter_id: usize,
}

impl MorselQueryPlanner {
    pub(super) async fn try_lower(
        &self,
        plan: &LogicalPlan,
        session: &dyn Session,
    ) -> Result<Lowering> {
        let all_subqueries = collect_scalar_subqueries_post_order(plan);
        let num_subqueries = all_subqueries.len();
        let results = ScalarSubqueryResults::new(num_subqueries);
        #[allow(clippy::allow_attributes, clippy::mutable_key_type)]
        let mut index_map: datafusion_common::HashMap<Subquery, SubqueryIndex> =
            datafusion_common::HashMap::with_capacity(num_subqueries);
        for (i, sq) in all_subqueries.into_iter().enumerate() {
            index_map.insert(sq, SubqueryIndex::new(i));
        }
        let planning_ctx = if index_map.is_empty() {
            PhysicalPlanningContext::default()
        } else {
            PhysicalPlanningContext::new(index_map, results.clone())
        };
        let lowered_subqueries = vec![false; num_subqueries];

        let mut builder = DagBuilder {
            workers: self.workers,
            session,
            planning_ctx,
            stages: Vec::new(),
            cte_stages: HashMap::new(),
            lowered_subqueries,
            scalar_subquery_specs: Vec::new(),
            next_filter_id: 0,
        };

        let root_node = match builder.lower_node(plan, false).await? {
            Ok(node) => node,
            Err(reason) => return Ok(Lowering::Unsupported(reason)),
        };

        let (pipeline, schema, root_ordering) = match root_node {
            LoweredNode::Open(mut open) => {
                builder.optimize_open_pipeline(&mut open, OpenOptimizeHint::default())?;
                builder.finalize_pipeline_joins(&mut open.source, &mut open.operators)?;
                let pipeline = Pipeline {
                    source: open.source,
                    operators: open.operators,
                    workers: self.workers,
                    sink: Sink::Result,
                    sink_capacity: result_capacity(self.workers),
                };
                (pipeline, open.schema, open.ordering)
            }
            LoweredNode::Blocking(mut blocking) => {
                builder.finalize_pipeline_joins(
                    &mut blocking.source,
                    &mut blocking.operators,
                )?;
                let sink_capacity = sink_result_capacity(&blocking.sink, self.workers);
                let pipeline = Pipeline {
                    source: blocking.source,
                    operators: blocking.operators,
                    workers: self.workers,
                    sink: blocking.sink,
                    sink_capacity,
                };
                (pipeline, blocking.schema, blocking.ordering)
            }
        };

        let scalar_subqueries = if builder.scalar_subquery_specs.is_empty() {
            None
        } else {
            Some(Arc::new(ScalarSubqueryState::new(
                builder.scalar_subquery_specs,
                results,
            )))
        };

        Ok(Lowering::Native(MorselExec::try_new_dag(
            builder.stages,
            scalar_subqueries,
            pipeline,
            schema,
            root_ordering,
        )?))
    }
}

fn project_ordering(
    schema: &SchemaRef,
    ordering: Option<LexOrdering>,
    ops: &[Operator],
) -> Result<Option<LexOrdering>> {
    let Some(ordering) = ordering else {
        return Ok(None);
    };
    let mut eq = EquivalenceProperties::new(Arc::clone(schema));
    eq.add_ordering(ordering);
    for op in ops {
        match op {
            Operator::Filter { .. } => {}
            Operator::Projection(projector) => {
                let mapping = projector.projection().projection_mapping(eq.schema())?;
                eq = eq.project(&mapping, Arc::clone(projector.output_schema()));
            }
            Operator::HashProbe(_) => return Ok(None),
        }
    }
    Ok(eq.output_ordering())
}

impl DagBuilder<'_> {
    fn ensure_expr_scalar_subqueries<'b>(
        &'b mut self,
        expr: &'b Expr,
    ) -> BoxFuture<'b, Result<std::result::Result<(), String>>> {
        Box::pin(async move {
            let mut subqueries = Vec::new();
            let mut correlated = false;
            expr.apply(|e| {
                if let Expr::ScalarSubquery(sq) = e {
                    if !sq.outer_ref_columns.is_empty() {
                        correlated = true;
                        return Ok(TreeNodeRecursion::Stop);
                    }
                    subqueries.push(sq.clone());
                }
                Ok(TreeNodeRecursion::Continue)
            })?;
            if correlated {
                return Ok(Err(
                    "correlated scalar subquery is not supported in native morsels"
                        .into(),
                ));
            }
            for sq in subqueries {
                let Some(idx) = self.planning_ctx.index_of(&sq) else {
                    return internal_err!(
                        "Morsel planner missing SubqueryIndex for scalar subquery"
                    );
                };
                if idx.as_usize() >= self.lowered_subqueries.len() {
                    self.lowered_subqueries.resize(idx.as_usize() + 1, false);
                }
                if self.lowered_subqueries[idx.as_usize()] {
                    continue;
                }
                if sq.subquery.schema().fields().len() != 1 {
                    return Ok(Err(format!(
                        "scalar subquery must return 1 column, got {}",
                        sq.subquery.schema().fields().len()
                    )));
                }
                let lowered = match self.lower_node(&sq.subquery, false).await? {
                    Ok(node) => node,
                    Err(reason) => return Ok(Err(reason)),
                };
                let binding = self.seal_into_stage(
                    lowered,
                    Some(format!("scalar_sq_{}", idx.as_usize())),
                )?;
                if binding.schema.fields().len() != 1 {
                    return internal_err!(
                        "Morsel scalar subquery stage produced {} columns",
                        binding.schema.fields().len()
                    );
                }
                let data_type = binding.schema.field(0).data_type().clone();
                self.lowered_subqueries[idx.as_usize()] = true;
                self.scalar_subquery_specs.push(ScalarSubquerySpec {
                    stage_id: binding.stage_id,
                    index: idx,
                    data_type,
                });
            }
            Ok(Ok(()))
        })
    }

    fn lower_node<'b>(
        &'b mut self,
        plan: &'b LogicalPlan,
        count_only: bool,
    ) -> BoxFuture<'b, Result<std::result::Result<LoweredNode, String>>> {
        Box::pin(async move {
            let mut post_nodes: Vec<&'b LogicalPlan> = Vec::new();
            let mut node = plan;
            loop {
                match node {
                    LogicalPlan::SubqueryAlias(alias) => {
                        node = &alias.input;
                    }
                    LogicalPlan::Filter(filter) => {
                        if !supported_expr(&filter.predicate)? {
                            return Ok(Err(format!(
                                "unsupported filter predicate: {}",
                                filter.predicate
                            )));
                        }
                        post_nodes.push(node);
                        node = &filter.input;
                    }
                    LogicalPlan::Projection(projection) => {
                        for expr in &projection.expr {
                            if !supported_expr(expr)? {
                                return Ok(Err(format!(
                                    "unsupported projection expression: {expr}"
                                )));
                            }
                        }
                        post_nodes.push(node);
                        node = &projection.input;
                    }
                    _ => break,
                }
            }

            match node {
                LogicalPlan::TableScan(scan) => {
                    if scan.fetch.is_some() || scan.skip.is_some() {
                        let skip = scan.skip.unwrap_or(0);
                        let fetch = scan.fetch;
                        let mut open =
                            match self.lower_scan_open(scan, Vec::new()).await? {
                                Ok(v) => v,
                                Err(reason) => return Ok(Err(reason)),
                            };
                        self.optimize_open_pipeline(
                            &mut open,
                            OpenOptimizeHint::default(),
                        )?;
                        return self
                            .finish_limit_plan(open, skip, fetch, post_nodes)
                            .await;
                    }
                    let open = match self.lower_scan_open(scan, post_nodes).await? {
                        Ok(v) => v,
                        Err(reason) => return Ok(Err(reason)),
                    };
                    Ok(Ok(LoweredNode::Open(Box::new(open))))
                }
                LogicalPlan::EmptyRelation(empty) => {
                    let source = self.lower_empty_relation(empty)?;
                    let (operators, schema) = match self
                        .lower_operators(post_nodes, Arc::clone(source.schema()))
                        .await?
                    {
                        Ok(v) => v,
                        Err(reason) => return Ok(Err(reason)),
                    };
                    Ok(Ok(LoweredNode::Open(Box::new(OpenPipeline {
                        source,
                        #[cfg(feature = "parquet")]
                        parquet_scan_meta: None,
                        operators,
                        schema,
                        ordering: None,
                    }))))
                }
                LogicalPlan::Values(values) => {
                    let source = match self.lower_values(values)? {
                        Ok(s) => s,
                        Err(reason) => return Ok(Err(reason)),
                    };
                    let (operators, schema) = match self
                        .lower_operators(post_nodes, Arc::clone(source.schema()))
                        .await?
                    {
                        Ok(v) => v,
                        Err(reason) => return Ok(Err(reason)),
                    };
                    Ok(Ok(LoweredNode::Open(Box::new(OpenPipeline {
                        source,
                        #[cfg(feature = "parquet")]
                        parquet_scan_meta: None,
                        operators,
                        schema,
                        ordering: None,
                    }))))
                }
                LogicalPlan::MaterializedCte(MaterializedCtePlan::Cte(cte)) => {
                    let cont_count_only = count_only && post_nodes.is_empty();
                    if cte.scans()?.is_empty() {
                        let inner = match self
                            .lower_node(cte.continuation(), cont_count_only)
                            .await?
                        {
                            Ok(n) => n,
                            Err(reason) => return Ok(Err(reason)),
                        };
                        return self.apply_post_nodes(inner, post_nodes).await;
                    }
                    let body_node = match self.lower_node(cte.body(), false).await? {
                        Ok(n) => n,
                        Err(reason) => return Ok(Err(reason)),
                    };
                    let binding = self
                        .seal_into_stage(body_node, Some(format!("cte_{}", cte.name)))?;
                    let prev = self.cte_stages.insert(
                        cte.id,
                        CteBinding {
                            stage_id: binding.stage_id,
                            schema: binding.schema,
                            ordered: binding.ordered,
                            ordering: binding.ordering,
                        },
                    );
                    let cont_res =
                        self.lower_node(cte.continuation(), cont_count_only).await?;
                    match prev {
                        Some(old) => {
                            self.cte_stages.insert(cte.id, old);
                        }
                        None => {
                            self.cte_stages.remove(&cte.id);
                        }
                    }
                    let cont_node = match cont_res {
                        Ok(n) => n,
                        Err(reason) => return Ok(Err(reason)),
                    };
                    self.apply_post_nodes(cont_node, post_nodes).await
                }
                LogicalPlan::MaterializedCte(MaterializedCtePlan::Scan(scan)) => {
                    let Some(binding) = self.cte_stages.get(&scan.id).cloned() else {
                        return Ok(Err(format!(
                            "unbound MaterializedCteScan '{}' in native morsels",
                            scan.name
                        )));
                    };
                    let source = Source::Stage(StageSource {
                        stage_id: binding.stage_id,
                        schema: Arc::clone(&binding.schema),
                        ordered: binding.ordered,
                    });
                    let (operators, schema) = match self
                        .lower_operators(post_nodes, Arc::clone(&binding.schema))
                        .await?
                    {
                        Ok(v) => v,
                        Err(reason) => return Ok(Err(reason)),
                    };
                    let ordering =
                        project_ordering(&binding.schema, binding.ordering, &operators)?;
                    Ok(Ok(LoweredNode::Open(Box::new(OpenPipeline {
                        source,
                        #[cfg(feature = "parquet")]
                        parquet_scan_meta: None,
                        operators,
                        schema,
                        ordering,
                    }))))
                }
                LogicalPlan::Join(join) => {
                    let join_count_only = count_only && post_nodes.is_empty();
                    self.lower_join_plan(join, post_nodes, join_count_only)
                        .await
                }
                LogicalPlan::Distinct(distinct) => {
                    self.lower_distinct_plan(distinct, post_nodes).await
                }
                LogicalPlan::Aggregate(aggregate) => {
                    self.lower_aggregate_plan(aggregate, post_nodes).await
                }
                LogicalPlan::Sort(sort) => {
                    self.lower_sort_plan(sort, 0, sort.fetch, post_nodes).await
                }
                LogicalPlan::Limit(limit) => {
                    self.lower_limit_plan(limit, post_nodes).await
                }
                _ => Ok(Err(format!(
                    "unsupported logical operator: {}",
                    node.display()
                ))),
            }
        })
    }

    async fn lower_scan_open(
        &mut self,
        scan: &TableScan,
        post_nodes: Vec<&LogicalPlan>,
    ) -> Result<std::result::Result<OpenPipeline, String>> {
        let source = match self.lower_scan_source(scan).await? {
            Ok(s) => s,
            Err(reason) => return Ok(Err(reason)),
        };
        #[cfg(feature = "parquet")]
        let parquet_scan_meta =
            matches!(source, Source::Parquet(_)).then(|| ParquetScanMeta {
                filters: scan.filters.clone(),
                projected_schema: Arc::clone(&scan.projected_schema),
            });
        let (operators, schema) = match self
            .lower_operators(post_nodes, Arc::clone(source.schema()))
            .await?
        {
            Ok(v) => v,
            Err(reason) => return Ok(Err(reason)),
        };
        Ok(Ok(OpenPipeline {
            source,
            #[cfg(feature = "parquet")]
            parquet_scan_meta,
            operators,
            schema,
            ordering: None,
        }))
    }

    async fn lower_as_open(
        &mut self,
        plan: &LogicalPlan,
        count_only: bool,
    ) -> Result<std::result::Result<OpenPipeline, String>> {
        self.lower_as_open_with_hint(plan, count_only, OpenOptimizeHint::default())
            .await
    }

    async fn lower_as_open_with_hint(
        &mut self,
        plan: &LogicalPlan,
        count_only: bool,
        hint: OpenOptimizeHint,
    ) -> Result<std::result::Result<OpenPipeline, String>> {
        let node = match self.lower_node(plan, count_only).await? {
            Ok(n) => n,
            Err(reason) => return Ok(Err(reason)),
        };
        match node {
            LoweredNode::Open(mut open) => {
                self.optimize_open_pipeline(&mut open, hint)?;
                Ok(Ok(*open))
            }
            LoweredNode::Blocking(blocking) => {
                let binding =
                    self.seal_into_stage(LoweredNode::Blocking(blocking), None)?;
                Ok(Ok(OpenPipeline {
                    source: Source::Stage(StageSource {
                        stage_id: binding.stage_id,
                        schema: Arc::clone(&binding.schema),
                        ordered: binding.ordered,
                    }),
                    #[cfg(feature = "parquet")]
                    parquet_scan_meta: None,
                    operators: Vec::new(),
                    schema: binding.schema,
                    ordering: binding.ordering,
                }))
            }
        }
    }

    async fn apply_post_nodes(
        &mut self,
        node: LoweredNode,
        post_nodes: Vec<&LogicalPlan>,
    ) -> Result<std::result::Result<LoweredNode, String>> {
        if post_nodes.is_empty() {
            return Ok(Ok(node));
        }
        let in_schema = Arc::clone(node.schema());
        let (ops, out_schema) = match self.lower_operators(post_nodes, in_schema).await? {
            Ok(v) => v,
            Err(reason) => return Ok(Err(reason)),
        };
        match node {
            LoweredNode::Open(mut open) => {
                let ordering = project_ordering(&open.schema, open.ordering, &ops)?;
                open.operators.extend(ops);
                open.schema = out_schema;
                open.ordering = ordering;
                Ok(Ok(LoweredNode::Open(open)))
            }
            LoweredNode::Blocking(mut blocking) => {
                let ordering =
                    project_ordering(&blocking.schema, blocking.ordering, &ops)?;
                blocking.sink.extend_result_operators(ops)?;
                blocking.schema = out_schema;
                blocking.ordering = ordering;
                Ok(Ok(LoweredNode::Blocking(blocking)))
            }
        }
    }

    fn seal_into_stage(
        &mut self,
        node: LoweredNode,
        label_hint: Option<String>,
    ) -> Result<StageBinding> {
        let (pipeline, output_schema, ordering, estimate) = match node {
            LoweredNode::Open(mut open) => {
                self.optimize_open_pipeline(&mut open, OpenOptimizeHint::default())?;
                self.finalize_pipeline_joins(&mut open.source, &mut open.operators)?;
                let estimate =
                    estimate_side_size(&open.source, &open.operators, &self.stages);
                let pipeline = Arc::new(Pipeline {
                    source: open.source,
                    operators: open.operators,
                    workers: self.workers,
                    sink: Sink::Result,
                    sink_capacity: result_capacity(self.workers),
                });
                (pipeline, open.schema, open.ordering, estimate)
            }
            LoweredNode::Blocking(mut blocking) => {
                self.finalize_pipeline_joins(
                    &mut blocking.source,
                    &mut blocking.operators,
                )?;
                let sink_capacity = sink_result_capacity(&blocking.sink, self.workers);
                let pipeline = Arc::new(Pipeline {
                    source: blocking.source,
                    operators: blocking.operators,
                    workers: self.workers,
                    sink: blocking.sink,
                    sink_capacity,
                });
                (
                    pipeline,
                    blocking.schema,
                    blocking.ordering,
                    blocking.estimate,
                )
            }
        };

        let ordered = ordering.is_some();
        let dependencies = collect_pipeline_stage_deps(&pipeline);
        let id = StageId(self.stages.len());
        let label = label_hint.unwrap_or_else(|| format!("stage_{}", id.0));
        self.stages.push(StageSpec {
            id,
            label,
            pipeline,
            output_schema: Arc::clone(&output_schema),
            estimate,
            dependencies,
        });
        Ok(StageBinding {
            stage_id: id,
            schema: output_schema,
            ordered,
            ordering,
        })
    }

    fn finalize_pipeline_joins(
        &mut self,
        source: &mut Source,
        operators: &mut [Operator],
    ) -> Result<()> {
        assign_join_build_ids(operators, &mut self.next_filter_id)?;
        #[cfg(not(feature = "parquet"))]
        let _ = source;
        #[cfg(feature = "parquet")]
        {
            let opt = &self.session.config_options().optimizer;
            if opt.enable_dynamic_filter_pushdown
                && opt.enable_join_dynamic_filter_pushdown
            {
                bind_pipeline_dynamic_join_filters(
                    source,
                    operators,
                    opt.hash_join_inlist_pushdown_max_distinct_values,
                    opt.hash_join_inlist_pushdown_max_size,
                )?;
            }
        }
        Ok(())
    }

    fn optimize_open_pipeline(
        &self,
        open: &mut OpenPipeline,
        hint: OpenOptimizeHint,
    ) -> Result<()> {
        #[cfg(not(feature = "parquet"))]
        let _ = hint.is_count_star;
        #[cfg(feature = "parquet")]
        if let Source::Parquet(ref mut parquet_scan) = open.source
            && let Some(meta) = open.parquet_scan_meta.take()
        {
            parquet_scan.pushdown_operators(
                &mut open.operators,
                hint.is_count_star,
                &meta.filters,
                &meta.projected_schema,
                self.session,
            )?;
        }
        let has_join = open
            .operators
            .iter()
            .any(|op| matches!(op, Operator::HashProbe(_)));
        if !hint.skip_filter_pruning && !has_join {
            let prune_filter = match open.source {
                Source::Memory(_) | Source::Stage(_) => {
                    self.workers == NonZeroUsize::MIN
                        || hint.is_general_agg
                        || hint.has_sort
                        || hint.always_prune_filter
                }
                #[cfg(feature = "parquet")]
                Source::Parquet(_) => true,
            };
            if prune_filter {
                prune_filter_input(&mut open.operators, open.source.schema());
            }
        }
        Ok(())
    }

    async fn lower_scan_source(
        &self,
        scan: &TableScan,
    ) -> Result<std::result::Result<Source, String>> {
        if !scan.statistics_requests.is_empty() {
            return Ok(Err(format!(
                "native morsel scan of {} does not support statistics requests",
                scan.table_name
            )));
        }
        let Ok(provider) = source_as_provider(&scan.source) else {
            return Ok(Err(format!(
                "unsupported table source for {}",
                scan.table_name
            )));
        };
        if let Some(table) = provider.downcast_ref::<MemTable>() {
            if !table.sort_order.lock().is_empty() || !scan.filters.is_empty() {
                return Ok(Err(format!(
                    "MemTable scan of {} has advertised ordering or scan filters",
                    scan.table_name
                )));
            }
            let mut batches = Vec::new();
            for partition in &table.batches {
                batches.extend(partition.read().await.iter().cloned());
            }
            Ok(Ok(Source::Memory(MemorySource {
                table: scan.table_name.to_string(),
                batches,
                projection: scan.projection.clone(),
                schema: project_schema(&provider.schema(), scan.projection.as_deref())?,
            })))
        } else {
            #[cfg(feature = "parquet")]
            if let Some(table) =
                provider.downcast_ref::<crate::datasource::listing::ListingTable>()
            {
                ParquetScan::lower(table, scan, self.session)
                    .await
                    .map(|res| res.map(Source::Parquet))
            } else {
                Ok(Err(unsupported_provider_reason(&scan.table_name)))
            }
            #[cfg(not(feature = "parquet"))]
            Ok(Err(unsupported_provider_reason(&scan.table_name)))
        }
    }

    fn lower_empty_relation(&self, empty: &EmptyRelation) -> Result<Source> {
        let schema = Arc::clone(empty.schema.inner());
        let batches = if empty.produce_one_row {
            let columns = schema
                .fields()
                .iter()
                .map(|f| new_null_array(f.data_type(), 1))
                .collect();
            let batch = RecordBatch::try_new_with_options(
                Arc::clone(&schema),
                columns,
                &RecordBatchOptions::new().with_row_count(Some(1)),
            )?;
            vec![batch]
        } else {
            Vec::new()
        };
        Ok(Source::Memory(MemorySource {
            table: "EmptyRelation".to_owned(),
            batches,
            projection: None,
            schema,
        }))
    }

    fn lower_values(
        &self,
        values: &Values,
    ) -> Result<std::result::Result<Source, String>> {
        let schema = Arc::clone(values.schema.inner());
        if values.values.is_empty() {
            return Ok(Ok(Source::Memory(MemorySource {
                table: "Values".to_owned(),
                batches: Vec::new(),
                projection: None,
                schema,
            })));
        }
        for row in &values.values {
            for expr in row {
                if !supported_expr(expr)? {
                    return Ok(Err(format!("unsupported VALUES expression: {expr}")));
                }
                if expr.exists(|e| Ok(matches!(e, Expr::ScalarSubquery(_))))? {
                    return Ok(Err(format!(
                        "scalar subquery in VALUES is not supported in native morsels: {expr}"
                    )));
                }
            }
        }
        let props = self.session.execution_props();
        let placeholder_batch = RecordBatch::try_new_with_options(
            Arc::new(Schema::empty()),
            vec![],
            &RecordBatchOptions::new().with_row_count(Some(1)),
        )?;
        let n_rows = values.values.len();
        let n_cols = schema.fields().len();
        let mut arrays = Vec::with_capacity(n_cols);
        for col_idx in 0..n_cols {
            let mut scalars = Vec::with_capacity(n_rows);
            for row in &values.values {
                let phys = create_physical_expr(
                    &row[col_idx],
                    &values.schema,
                    props,
                    &self.planning_ctx,
                )?;
                let scalar = match phys.evaluate(&placeholder_batch)? {
                    ColumnarValue::Scalar(s) => s,
                    ColumnarValue::Array(arr) if arr.len() == 1 => {
                        ScalarValue::try_from_array(&arr, 0)?
                    }
                    ColumnarValue::Array(_) => {
                        return plan_err!("Cannot have array values in a values list");
                    }
                };
                scalars.push(scalar);
            }
            arrays.push(ScalarValue::iter_to_array(scalars)?);
        }
        let batch = RecordBatch::try_new_with_options(
            Arc::clone(&schema),
            arrays,
            &RecordBatchOptions::new().with_row_count(Some(n_rows)),
        )?;
        Ok(Ok(Source::Memory(MemorySource {
            table: "Values".to_owned(),
            batches: vec![batch],
            projection: None,
            schema,
        })))
    }

    async fn lower_distinct_plan(
        &mut self,
        distinct: &Distinct,
        post_nodes: Vec<&LogicalPlan>,
    ) -> Result<std::result::Result<LoweredNode, String>> {
        let Distinct::All(input) = distinct else {
            return Ok(Err(
                "DISTINCT ON is not supported in native morsel execution".into(),
            ));
        };
        let open = match self.lower_as_open(input, false).await? {
            Ok(v) => v,
            Err(reason) => return Ok(Err(reason)),
        };
        let input_est = estimate_side_size(&open.source, &open.operators, &self.stages);
        let sink = match AggregateSink::try_new_distinct(&open.schema)? {
            Ok(s) => s,
            Err(reason) => return Ok(Err(reason)),
        };
        let (result_operators, out_schema) = match self
            .lower_operators(post_nodes, Arc::clone(sink.schema()))
            .await?
        {
            Ok(v) => v,
            Err(reason) => return Ok(Err(reason)),
        };
        let estimate = SideEstimate {
            is_empty: input_est.is_empty,
            mem_rows: input_est.mem_rows.map(|r| (r / 2).max(1)),
            approx_bytes: (input_est.approx_bytes / 2).max(64),
        };
        Ok(Ok(LoweredNode::Blocking(Box::new(BlockingPipeline {
            source: open.source,
            operators: open.operators,
            sink: Sink::Aggregate {
                sink,
                result_operators,
                post_sort: None,
            },
            schema: out_schema,
            ordering: None,
            estimate,
        }))))
    }

    async fn lower_aggregate_plan(
        &mut self,
        aggregate: &Aggregate,
        post_nodes: Vec<&LogicalPlan>,
    ) -> Result<std::result::Result<LoweredNode, String>> {
        enum AggKind<'x> {
            Count {
                count_expr: &'x Expr,
            },
            GroupCount {
                key_expr: &'x Expr,
                key_index: usize,
                key_type: IntKeyType,
                count_expr: &'x Expr,
            },
            General,
        }

        let kind = if let Some(count_expr) = count::count_star_aggr_expr(aggregate) {
            if aggregate.group_expr.is_empty() {
                AggKind::Count { count_expr }
            } else if let Ok(key) = group_count::group_by_integer_column(aggregate)? {
                AggKind::GroupCount {
                    key_expr: key.expr,
                    key_index: key.column_index,
                    key_type: key.key_type,
                    count_expr,
                }
            } else {
                match aggregate::validate_aggregate(aggregate)? {
                    Ok(()) => AggKind::General,
                    Err(reason) => return Ok(Err(reason)),
                }
            }
        } else {
            match aggregate::validate_aggregate(aggregate)? {
                Ok(()) => AggKind::General,
                Err(reason) => return Ok(Err(reason)),
            }
        };

        for expr in aggregate.group_expr.iter().chain(&aggregate.aggr_expr) {
            if let Err(reason) = self.ensure_expr_scalar_subqueries(expr).await? {
                return Ok(Err(reason));
            }
        }

        let is_count_star = matches!(kind, AggKind::Count { .. });
        let skip_filter_pruning =
            matches!(kind, AggKind::Count { .. } | AggKind::GroupCount { .. });
        let is_general_agg = matches!(kind, AggKind::General);

        let child = match self.lower_node(&aggregate.input, is_count_star).await? {
            Ok(n) => n,
            Err(reason) => return Ok(Err(reason)),
        };
        let mut open = match child {
            LoweredNode::Open(open) => *open,
            LoweredNode::Blocking(blocking) => {
                let binding =
                    self.seal_into_stage(LoweredNode::Blocking(blocking), None)?;
                OpenPipeline {
                    source: Source::Stage(StageSource {
                        stage_id: binding.stage_id,
                        schema: Arc::clone(&binding.schema),
                        ordered: binding.ordered,
                    }),
                    #[cfg(feature = "parquet")]
                    parquet_scan_meta: None,
                    operators: Vec::new(),
                    schema: binding.schema,
                    ordering: binding.ordering,
                }
            }
        };
        self.optimize_open_pipeline(
            &mut open,
            OpenOptimizeHint {
                is_count_star,
                skip_filter_pruning,
                is_general_agg,
                ..Default::default()
            },
        )?;
        let input_est = estimate_side_size(&open.source, &open.operators, &self.stages);
        let input_schema = open.schema;

        let (sink, out_schema, estimate) = match kind {
            AggKind::Count { count_expr } => {
                let field = LoweredAggregateBuilder::new(
                    count_expr,
                    aggregate.input.schema(),
                    &input_schema,
                    self.session.execution_props(),
                    &self.planning_ctx,
                )
                .build()?
                .aggregate
                .field();
                let count_schema = Arc::new(Schema::new_with_metadata(
                    vec![field],
                    input_schema.metadata().clone(),
                ));
                let sink = CountSink::new(Arc::clone(&count_schema));
                let (result_operators, result_schema) =
                    match self.lower_operators(post_nodes, count_schema).await? {
                        Ok(v) => v,
                        Err(reason) => return Ok(Err(reason)),
                    };
                let est = SideEstimate {
                    is_empty: false,
                    mem_rows: Some(1),
                    approx_bytes: 64,
                };
                (
                    Sink::Count {
                        sink,
                        result_operators,
                        post_sort: None,
                    },
                    result_schema,
                    est,
                )
            }
            AggKind::GroupCount {
                key_expr,
                key_index,
                key_type,
                count_expr,
            } => {
                let Some(input_field) = input_schema.fields().get(key_index) else {
                    return internal_err!(
                        "Morsel GROUP BY key index {key_index} out of bounds for schema"
                    );
                };
                let Some(physical_key_type) =
                    IntKeyType::from_data_type(input_field.data_type())
                else {
                    return Ok(Err(format!(
                        "unsupported GROUP BY key type {}: only a single integer column is supported",
                        input_field.data_type()
                    )));
                };
                if physical_key_type != key_type {
                    return internal_err!(
                        "Morsel GROUP BY key type mismatch between logical and physical schemas"
                    );
                }
                let key_name = physical_name(key_expr)?;
                let key_field = Arc::new(
                    Field::new(
                        &key_name,
                        input_field.data_type().clone(),
                        input_field.is_nullable(),
                    )
                    .with_metadata(input_field.metadata().clone()),
                );
                let count_field = LoweredAggregateBuilder::new(
                    count_expr,
                    aggregate.input.schema(),
                    &input_schema,
                    self.session.execution_props(),
                    &self.planning_ctx,
                )
                .build()?
                .aggregate
                .field();
                let agg_schema = Arc::new(Schema::new_with_metadata(
                    vec![key_field, count_field],
                    input_schema.metadata().clone(),
                ));
                let sink = GroupCountSink::new(
                    Arc::clone(&agg_schema),
                    key_index,
                    physical_key_type,
                    key_name,
                );
                let (result_operators, result_schema) =
                    match self.lower_operators(post_nodes, agg_schema).await? {
                        Ok(v) => v,
                        Err(reason) => return Ok(Err(reason)),
                    };
                let est = SideEstimate {
                    is_empty: input_est.is_empty,
                    mem_rows: input_est.mem_rows.map(|r| (r / 4).max(1)),
                    approx_bytes: (input_est.approx_bytes / 4).max(64),
                };
                (
                    Sink::GroupCount {
                        sink,
                        result_operators,
                        post_sort: None,
                    },
                    result_schema,
                    est,
                )
            }
            AggKind::General => {
                let sink = match AggregateSink::try_new_with_context(
                    aggregate,
                    &input_schema,
                    self.session,
                    &self.planning_ctx,
                )? {
                    Ok(sink) => sink,
                    Err(reason) => return Ok(Err(reason)),
                };
                let est = if sink.is_grouped() {
                    SideEstimate {
                        is_empty: input_est.is_empty,
                        mem_rows: input_est.mem_rows.map(|r| (r / 4).max(1)),
                        approx_bytes: (input_est.approx_bytes / 4).max(64),
                    }
                } else {
                    SideEstimate {
                        is_empty: false,
                        mem_rows: Some(1),
                        approx_bytes: 64,
                    }
                };
                let (result_operators, result_schema) = match self
                    .lower_operators(post_nodes, Arc::clone(sink.schema()))
                    .await?
                {
                    Ok(v) => v,
                    Err(reason) => return Ok(Err(reason)),
                };
                (
                    Sink::Aggregate {
                        sink,
                        result_operators,
                        post_sort: None,
                    },
                    result_schema,
                    est,
                )
            }
        };

        Ok(Ok(LoweredNode::Blocking(Box::new(BlockingPipeline {
            source: open.source,
            operators: open.operators,
            sink,
            schema: out_schema,
            ordering: None,
            estimate,
        }))))
    }

    async fn lower_sort_plan(
        &mut self,
        sort: &Sort,
        skip: usize,
        fetch: Option<usize>,
        post_nodes: Vec<&LogicalPlan>,
    ) -> Result<std::result::Result<LoweredNode, String>> {
        if sort.expr.is_empty() {
            return Ok(Err("Sort requires at least one sort expression".into()));
        }
        for sort_expr in &sort.expr {
            if !supported_expr(&sort_expr.expr)? {
                return Ok(Err(format!(
                    "unsupported sort expression: {}",
                    sort_expr.expr
                )));
            }
            if let Err(reason) =
                self.ensure_expr_scalar_subqueries(&sort_expr.expr).await?
            {
                return Ok(Err(reason));
            }
        }

        let child = match self.lower_node(&sort.input, false).await? {
            Ok(n) => n,
            Err(reason) => return Ok(Err(reason)),
        };

        let sort_input_schema = Arc::clone(child.schema());
        let sort_exprs = create_physical_sort_exprs(
            &sort.expr,
            sort.input.schema(),
            self.session.execution_props(),
            &self.planning_ctx,
        )?;
        let Some(ordering) = LexOrdering::new(sort_exprs) else {
            return Ok(Err("Sort requires at least one sort expression".into()));
        };
        let (sort_result_operators, out_schema) = match self
            .lower_operators(post_nodes, Arc::clone(&sort_input_schema))
            .await?
        {
            Ok(v) => v,
            Err(reason) => return Ok(Err(reason)),
        };
        let out_ordering = project_ordering(
            &sort_input_schema,
            Some(ordering.clone()),
            &sort_result_operators,
        )?;

        let post_sort = if let Some(k) = fetch
            && (k == 0 || skip.checked_add(k).is_some())
        {
            let topn_sink = match TopNSink::try_new(
                Arc::clone(&sort_input_schema),
                ordering,
                skip,
                k,
            )? {
                Ok(s) => s,
                Err(reason) => return Ok(Err(reason)),
            };
            PostSort::TopN {
                sink: topn_sink,
                result_operators: sort_result_operators,
            }
        } else {
            let sort_sink = match SortSink::try_new(
                Arc::clone(&sort_input_schema),
                ordering,
                skip,
                fetch,
            )? {
                Ok(s) => s,
                Err(reason) => return Ok(Err(reason)),
            };
            PostSort::Sort {
                sink: sort_sink,
                result_operators: sort_result_operators,
            }
        };

        match child {
            LoweredNode::Blocking(mut blocking)
                if blocking.sink.can_attach_post_sort() =>
            {
                blocking.sink.attach_post_sort(post_sort);
                blocking.schema = out_schema;
                blocking.ordering = out_ordering;
                if let Some(k) = fetch {
                    blocking.estimate = SideEstimate {
                        is_empty: k == 0 || blocking.estimate.is_empty,
                        mem_rows: Some(k),
                        approx_bytes: (k as u64).saturating_mul(64),
                    };
                }
                Ok(Ok(LoweredNode::Blocking(blocking)))
            }
            other => {
                let mut open = match other {
                    LoweredNode::Open(mut open) => {
                        self.optimize_open_pipeline(
                            &mut open,
                            OpenOptimizeHint {
                                has_sort: true,
                                ..Default::default()
                            },
                        )?;
                        *open
                    }
                    LoweredNode::Blocking(blocking) => {
                        let binding =
                            self.seal_into_stage(LoweredNode::Blocking(blocking), None)?;
                        OpenPipeline {
                            source: Source::Stage(StageSource {
                                stage_id: binding.stage_id,
                                schema: Arc::clone(&binding.schema),
                                ordered: binding.ordered,
                            }),
                            #[cfg(feature = "parquet")]
                            parquet_scan_meta: None,
                            operators: Vec::new(),
                            schema: binding.schema,
                            ordering: binding.ordering,
                        }
                    }
                };
                let input_est =
                    estimate_side_size(&open.source, &open.operators, &self.stages);
                let estimate = match fetch {
                    Some(0) => SideEstimate {
                        is_empty: true,
                        mem_rows: Some(0),
                        approx_bytes: 0,
                    },
                    Some(k) => SideEstimate {
                        is_empty: input_est.is_empty,
                        mem_rows: Some(k),
                        approx_bytes: (k as u64).saturating_mul(64),
                    },
                    None => input_est,
                };
                let sink = match post_sort {
                    PostSort::TopN {
                        sink,
                        result_operators,
                    } => Sink::TopN {
                        sink,
                        result_operators,
                    },
                    PostSort::Sort {
                        sink,
                        result_operators,
                    } => Sink::Sort {
                        sink,
                        result_operators,
                    },
                };
                open.ordering = out_ordering;
                Ok(Ok(LoweredNode::Blocking(Box::new(BlockingPipeline {
                    source: open.source,
                    operators: open.operators,
                    sink,
                    schema: out_schema,
                    ordering: open.ordering,
                    estimate,
                }))))
            }
        }
    }

    async fn finish_limit_plan(
        &mut self,
        open: OpenPipeline,
        skip: usize,
        fetch: Option<usize>,
        post_nodes: Vec<&LogicalPlan>,
    ) -> Result<std::result::Result<LoweredNode, String>> {
        let input_est = estimate_side_size(&open.source, &open.operators, &self.stages);
        let (result_operators, out_schema) = match self
            .lower_operators(post_nodes, Arc::clone(&open.schema))
            .await?
        {
            Ok(v) => v,
            Err(reason) => return Ok(Err(reason)),
        };
        let out_ordering =
            project_ordering(&open.schema, open.ordering, &result_operators)?;
        let estimate = match fetch {
            Some(0) => SideEstimate {
                is_empty: true,
                mem_rows: Some(0),
                approx_bytes: 0,
            },
            Some(k) => SideEstimate {
                is_empty: input_est.is_empty,
                mem_rows: Some(k),
                approx_bytes: (k as u64).saturating_mul(64),
            },
            None => input_est,
        };
        Ok(Ok(LoweredNode::Blocking(Box::new(BlockingPipeline {
            source: open.source,
            operators: open.operators,
            sink: Sink::Limit {
                skip,
                fetch,
                result_operators,
            },
            schema: out_schema,
            ordering: out_ordering,
            estimate,
        }))))
    }

    async fn lower_limit_plan(
        &mut self,
        first_limit: &Limit,
        post_nodes: Vec<&LogicalPlan>,
    ) -> Result<std::result::Result<LoweredNode, String>> {
        let mut combined_skip = 0usize;
        let mut combined_fetch: Option<usize> = None;
        let mut cur_limit = first_limit;
        let mut between_projections: Vec<&LogicalPlan> = Vec::new();
        let (below_limit, all_post_nodes) = loop {
            let SkipType::Literal(skip) = cur_limit.get_skip_type()? else {
                return Ok(Err(format!(
                    "unsupported non-literal OFFSET expression in native morsels: {}",
                    cur_limit.input.display()
                )));
            };
            let FetchType::Literal(fetch) = cur_limit.get_fetch_type()? else {
                return Ok(Err(format!(
                    "unsupported non-literal LIMIT expression in native morsels: {}",
                    cur_limit.input.display()
                )));
            };
            (combined_skip, combined_fetch) =
                combine_limit(combined_skip, combined_fetch, skip, fetch);

            let mut next = cur_limit.input.as_ref();
            let mut projs = Vec::new();
            loop {
                match next {
                    LogicalPlan::SubqueryAlias(alias) => next = &alias.input,
                    LogicalPlan::Projection(p) => {
                        for expr in &p.expr {
                            if !supported_expr(expr)? {
                                return Ok(Err(format!(
                                    "unsupported projection expression: {expr}"
                                )));
                            }
                        }
                        projs.push(next);
                        next = &p.input;
                    }
                    _ => break,
                }
            }
            match next {
                LogicalPlan::Limit(inner_limit) => {
                    between_projections.extend(projs);
                    cur_limit = inner_limit;
                }
                LogicalPlan::Sort(sort) => {
                    // Fold Limit(+Projection*) directly into Sort -> TopN fast path!
                    let mut all_post = post_nodes;
                    all_post.extend(between_projections);
                    all_post.extend(projs);
                    let (skip, fetch) =
                        combine_limit(combined_skip, combined_fetch, 0, sort.fetch);
                    return self.lower_sort_plan(sort, skip, fetch, all_post).await;
                }
                LogicalPlan::TableScan(scan) => {
                    // Fold scan.skip/scan.fetch into the enclosing Limit so scan+LIMIT
                    // stays a single streaming Sink::Limit over an open scan with no stage.
                    let (skip, fetch) = combine_limit(
                        combined_skip,
                        combined_fetch,
                        scan.skip.unwrap_or(0),
                        scan.fetch,
                    );
                    between_projections.extend(projs);
                    let mut open =
                        match self.lower_scan_open(scan, between_projections).await? {
                            Ok(v) => v,
                            Err(reason) => return Ok(Err(reason)),
                        };
                    self.optimize_open_pipeline(&mut open, OpenOptimizeHint::default())?;
                    return self.finish_limit_plan(open, skip, fetch, post_nodes).await;
                }
                _ => {
                    let mut all_post = post_nodes;
                    all_post.extend(between_projections);
                    break (cur_limit.input.as_ref(), all_post);
                }
            }
        };

        // Standalone Limit (without Sort or direct TableScan underneath).
        let open = match self.lower_as_open(below_limit, false).await? {
            Ok(v) => v,
            Err(reason) => return Ok(Err(reason)),
        };
        self.finish_limit_plan(open, combined_skip, combined_fetch, all_post_nodes)
            .await
    }

    async fn lower_join_plan(
        &mut self,
        join: &Join,
        post_nodes: Vec<&LogicalPlan>,
        count_only: bool,
    ) -> Result<std::result::Result<LoweredNode, String>> {
        if !matches!(
            join.join_type,
            JoinType::Inner
                | JoinType::Left
                | JoinType::Right
                | JoinType::Full
                | JoinType::LeftSemi
                | JoinType::RightSemi
                | JoinType::LeftAnti
                | JoinType::RightAnti
        ) {
            return Ok(Err(format!(
                "unsupported join type {:?} in native morsel hash join",
                join.join_type
            )));
        }

        let is_cross = join.on.is_empty();
        if is_cross {
            if join.join_type != JoinType::Inner {
                return Ok(Err(format!(
                    "non-equi {:?} join without equi-join keys is not supported in native morsel join",
                    join.join_type
                )));
            }
            if join.null_aware {
                return Ok(Err(
                    "null-aware join without equi-join keys is not supported in native morsel join"
                        .into(),
                ));
            }
        }

        if join.null_aware {
            if !matches!(join.join_type, JoinType::LeftAnti | JoinType::RightAnti) {
                return Ok(Err(format!(
                    "null-aware {:?} join is not supported in native morsel hash join",
                    join.join_type
                )));
            }
            if join.null_equality != NullEquality::NullEqualsNothing {
                return Ok(Err(
                    "null-aware join with IS NOT DISTINCT FROM is not supported in native morsel hash join"
                        .into(),
                ));
            }
            if join.on.len() != 1 || join.filter.is_some() {
                return Ok(Err(
                    "correlated, multi-key, or filtered null-aware NOT IN join is not supported in native morsel hash join"
                        .into(),
                ));
            }
        }

        for (l_expr, r_expr) in &join.on {
            if !supported_expr(l_expr)? {
                return Ok(Err(format!(
                    "unsupported left join key expression: {l_expr}"
                )));
            }
            if !supported_expr(r_expr)? {
                return Ok(Err(format!(
                    "unsupported right join key expression: {r_expr}"
                )));
            }
            if let Err(reason) = self.ensure_expr_scalar_subqueries(l_expr).await? {
                return Ok(Err(reason));
            }
            if let Err(reason) = self.ensure_expr_scalar_subqueries(r_expr).await? {
                return Ok(Err(reason));
            }
        }
        if let Some(filter) = &join.filter {
            if !supported_expr(filter)? {
                return Ok(Err(format!("unsupported join filter expression: {filter}")));
            }
            if let Err(reason) = self.ensure_expr_scalar_subqueries(filter).await? {
                return Ok(Err(reason));
            }
        }

        let join_side_hint = OpenOptimizeHint {
            always_prune_filter: true,
            ..Default::default()
        };
        let left_open = match self
            .lower_as_open_with_hint(&join.left, false, join_side_hint)
            .await?
        {
            Ok(v) => v,
            Err(reason) => return Ok(Err(reason)),
        };
        let right_open = match self
            .lower_as_open_with_hint(&join.right, false, join_side_hint)
            .await?
        {
            Ok(v) => v,
            Err(reason) => return Ok(Err(reason)),
        };
        let (left_source, left_ops, left_schema) =
            (left_open.source, left_open.operators, left_open.schema);
        let (right_source, right_ops, right_schema) =
            (right_open.source, right_open.operators, right_open.schema);

        let props = self.session.execution_props();
        let mut left_key_exprs = Vec::with_capacity(join.on.len());
        let mut right_key_exprs = Vec::with_capacity(join.on.len());
        let mut left_key_names = Vec::with_capacity(join.on.len());
        let mut right_key_names = Vec::with_capacity(join.on.len());
        let mut key_types = Vec::with_capacity(join.on.len());

        for (l_expr, r_expr) in &join.on {
            let l_phys = create_physical_expr(
                l_expr,
                join.left.schema(),
                props,
                &self.planning_ctx,
            )?;
            let r_phys = create_physical_expr(
                r_expr,
                join.right.schema(),
                props,
                &self.planning_ctx,
            )?;
            let l_type = l_phys.data_type(&left_schema)?;
            let r_type = r_phys.data_type(&right_schema)?;
            if l_type != r_type {
                return Ok(Err(format!(
                    "mismatched join key types: left={l_type}, right={r_type}"
                )));
            }
            if !is_supported_join_key_type(&l_type) {
                return Ok(Err(format!(
                    "unsupported join key type {l_type} in native morsel hash join"
                )));
            }
            left_key_names.push(physical_name(l_expr)?);
            right_key_names.push(physical_name(r_expr)?);
            left_key_exprs.push(l_phys);
            right_key_exprs.push(r_phys);
            key_types.push(l_type);
        }

        let left_len = left_schema.fields().len();
        let join_output_schema =
            Arc::new(build_join_schema(&left_schema, &right_schema, &join.join_type).0);

        let inner_post_filter = !is_cross && join.join_type == JoinType::Inner;
        let mut post_ops = Vec::new();
        if inner_post_filter && let Some(filter_expr) = &join.filter {
            let predicate = create_physical_expr(
                filter_expr,
                &join.schema,
                props,
                &self.planning_ctx,
            )?;
            if predicate.data_type(&join_output_schema)? != DataType::Boolean {
                return plan_err!(
                    "Morsel join filter predicate must return Boolean: {predicate}"
                );
            }
            post_ops.push(Operator::Filter {
                predicate,
                projection: None,
            });
        }
        let (extra_post_ops, mut final_schema) = match self
            .lower_operators(post_nodes, Arc::clone(&join_output_schema))
            .await?
        {
            Ok(v) => v,
            Err(reason) => return Ok(Err(reason)),
        };
        post_ops.extend(extra_post_ops);

        let build_side = if join.null_aware {
            match join.join_type {
                JoinType::LeftAnti => BuildSide::Right,
                JoinType::RightAnti => BuildSide::Left,
                _ => unreachable!(),
            }
        } else {
            choose_build_side(
                estimate_join_side_size(&left_source, &left_ops, &self.stages),
                estimate_join_side_size(&right_source, &right_ops, &self.stages),
            )
        };
        let kind = ProbeJoinKind::from_logical(
            join.join_type,
            build_side,
            join.null_aware,
            is_cross,
        )?;

        let (
            mut build_source,
            mut build_ops,
            build_schema,
            build_key_exprs,
            build_key_names,
            probe_source,
            mut probe_ops,
            probe_key_exprs,
            probe_key_names,
        ) = match build_side {
            BuildSide::Left => (
                left_source,
                left_ops,
                Arc::clone(&left_schema),
                left_key_exprs,
                left_key_names,
                right_source,
                right_ops,
                right_key_exprs,
                right_key_names,
            ),
            BuildSide::Right => (
                right_source,
                right_ops,
                Arc::clone(&right_schema),
                right_key_exprs,
                right_key_names,
                left_source,
                left_ops,
                left_key_exprs,
                left_key_names,
            ),
        };

        let map_join_col = |join_col: usize,
                            payload_cols: &mut Vec<usize>|
         -> JoinOutputCol {
            let (col_side, side_idx) = match join.join_type {
                JoinType::LeftSemi | JoinType::LeftAnti => (BuildSide::Left, join_col),
                JoinType::RightSemi | JoinType::RightAnti => (BuildSide::Right, join_col),
                JoinType::Inner | JoinType::Left | JoinType::Right | JoinType::Full => {
                    if join_col < left_len {
                        (BuildSide::Left, join_col)
                    } else {
                        (BuildSide::Right, join_col - left_len)
                    }
                }
                JoinType::LeftMark | JoinType::RightMark => unreachable!(),
            };
            if col_side == build_side {
                let pos = payload_cols
                    .iter()
                    .position(|&c| c == side_idx)
                    .unwrap_or_else(|| {
                        payload_cols.push(side_idx);
                        payload_cols.len() - 1
                    });
                JoinOutputCol::Build(pos)
            } else {
                JoinOutputCol::Probe(side_idx)
            }
        };

        let (mut build_payload_columns, output_projection, probe_output_schema) =
            if count_only && post_ops.is_empty() {
                let empty = Arc::new(Schema::new_with_metadata(
                    Vec::<Field>::new(),
                    join_output_schema.metadata().clone(),
                ));
                final_schema = Arc::clone(&empty);
                (Vec::new(), Vec::new(), empty)
            } else if let Some(Operator::Projection(projector)) = post_ops.first()
                && projector
                    .projection()
                    .as_ref()
                    .iter()
                    .all(|p| p.expr.is::<Column>())
            {
                let mut payload_cols: Vec<usize> = Vec::new();
                let mut output_proj =
                    Vec::with_capacity(projector.projection().as_ref().len());
                for p in projector.projection().as_ref() {
                    let join_col = p.expr.downcast_ref::<Column>().unwrap().index();
                    output_proj.push(map_join_col(join_col, &mut payload_cols));
                }
                let out_schema = Arc::clone(projector.output_schema());
                post_ops.remove(0);
                (payload_cols, output_proj, out_schema)
            } else if let Some(Operator::Projection(projector)) = post_ops.first_mut()
                && projector.projection().column_indices().len()
                    < join_output_schema.fields().len()
            {
                let needed = projector.projection().column_indices();
                let narrow = ProjectionExprs::from_indices(&needed, &join_output_schema);
                let narrowed_schema = Arc::new(join_output_schema.project(&needed)?);
                let remapped = projector
                    .projection()
                    .clone()
                    .try_map_exprs(|expr| narrow.project_expr(&expr))?
                    .make_projector_with_schema_metadata(
                        &narrowed_schema,
                        projector.output_schema(),
                    )?;
                let mut payload_cols: Vec<usize> = Vec::new();
                if remapped.output_schema() == projector.output_schema() {
                    let mut output_proj = Vec::with_capacity(needed.len());
                    for &join_col in &needed {
                        output_proj.push(map_join_col(join_col, &mut payload_cols));
                    }
                    *projector = remapped;
                    (payload_cols, output_proj, narrowed_schema)
                } else {
                    let out_len = join_output_schema.fields().len();
                    let mut output_proj = Vec::with_capacity(out_len);
                    for join_col in 0..out_len {
                        output_proj.push(map_join_col(join_col, &mut payload_cols));
                    }
                    (payload_cols, output_proj, Arc::clone(&join_output_schema))
                }
            } else {
                let mut payload_cols: Vec<usize> = Vec::new();
                let out_len = join_output_schema.fields().len();
                let mut output_proj = Vec::with_capacity(out_len);
                for join_col in 0..out_len {
                    output_proj.push(map_join_col(join_col, &mut payload_cols));
                }
                (payload_cols, output_proj, Arc::clone(&join_output_schema))
            };

        let compiled_filter = if !inner_post_filter
            && let Some(filter_expr) = &join.filter
        {
            let left_df_schema = join.left.schema();
            let right_df_schema = join.right.schema();
            let cols = filter_expr.column_refs();

            let mut left_field_indices: Vec<usize> = cols
                .iter()
                .filter_map(|c| left_df_schema.index_of_column(c).ok())
                .collect();
            left_field_indices.sort_unstable();
            left_field_indices.dedup();

            let mut right_field_indices: Vec<usize> = cols
                .iter()
                .filter_map(|c| right_df_schema.index_of_column(c).ok())
                .collect();
            right_field_indices.sort_unstable();
            right_field_indices.dedup();

            let (filter_df_fields, filter_fields): (Vec<_>, Vec<_>) = left_field_indices
                .iter()
                .map(|&i| {
                    let (q, f) = left_df_schema.qualified_field(i);
                    ((q.cloned(), Arc::clone(f)), left_schema.field(i).clone())
                })
                .chain(right_field_indices.iter().map(|&i| {
                    let (q, f) = right_df_schema.qualified_field(i);
                    ((q.cloned(), Arc::clone(f)), right_schema.field(i).clone())
                }))
                .unzip();

            let metadata: HashMap<_, _> = left_df_schema
                .metadata()
                .clone()
                .into_iter()
                .chain(right_df_schema.metadata().clone())
                .collect();
            let filter_df_schema =
                DFSchema::new_with_metadata(filter_df_fields, metadata.clone())?;
            let filter_schema =
                Arc::new(Schema::new_with_metadata(filter_fields, metadata));

            let filter_phys = create_physical_expr(
                filter_expr,
                &filter_df_schema,
                props,
                &self.planning_ctx,
            )?;
            if filter_phys.data_type(&filter_schema)? != DataType::Boolean {
                return plan_err!(
                    "Morsel join filter predicate must return Boolean: {filter_phys}"
                );
            }

            let mut filter_columns =
                Vec::with_capacity(left_field_indices.len() + right_field_indices.len());
            for left_idx in left_field_indices {
                let col = if build_side == BuildSide::Left {
                    let pos = build_payload_columns
                        .iter()
                        .position(|&c| c == left_idx)
                        .unwrap_or_else(|| {
                            build_payload_columns.push(left_idx);
                            build_payload_columns.len() - 1
                        });
                    JoinOutputCol::Build(pos)
                } else {
                    JoinOutputCol::Probe(left_idx)
                };
                filter_columns.push(col);
            }
            for right_idx in right_field_indices {
                let col = if build_side == BuildSide::Right {
                    let pos = build_payload_columns
                        .iter()
                        .position(|&c| c == right_idx)
                        .unwrap_or_else(|| {
                            build_payload_columns.push(right_idx);
                            build_payload_columns.len() - 1
                        });
                    JoinOutputCol::Build(pos)
                } else {
                    JoinOutputCol::Probe(right_idx)
                };
                filter_columns.push(col);
            }

            Some(CompiledJoinFilter {
                expr: filter_phys,
                schema: filter_schema,
                columns: filter_columns,
            })
        } else {
            None
        };

        prune_filter_input(&mut post_ops, &probe_output_schema);

        let payload_schema = Arc::new(build_schema.project(&build_payload_columns)?);
        let build_sink = match HashJoinBuildSink::try_new(HashJoinBuildConfig {
            key_exprs: build_key_exprs,
            key_names: build_key_names,
            key_types: key_types.clone(),
            null_equality: join.null_equality,
            keep_null_key_rows: kind.keep_null_key_build_rows(),
            track_null_keys: kind.is_null_aware_anti(),
            needs_matched_bitset: kind.needs_matched_bitset(),
            payload_columns: build_payload_columns,
            payload_schema,
            dynamic_filter: None,
        })? {
            Ok(sink) => Arc::new(sink),
            Err(reason) => return Ok(Err(reason)),
        };

        self.finalize_pipeline_joins(&mut build_source, &mut build_ops)?;
        let build_pipeline = Arc::new(Pipeline {
            source: build_source,
            operators: build_ops,
            workers: self.workers,
            sink: Sink::HashJoinBuild(build_sink),
            sink_capacity: 1,
        });

        let probe = Arc::new(HashJoinProbe::new(HashJoinProbeConfig {
            build_pipeline,
            build_side,
            kind,
            join_type: join.join_type,
            null_equality: join.null_equality,
            probe_key_exprs,
            probe_key_names,
            key_types,
            filter: compiled_filter,
            output_projection,
            output_schema: probe_output_schema,
            build_id: 0,
        }));
        probe_ops.push(Operator::HashProbe(probe));
        probe_ops.extend(post_ops);
        Ok(Ok(LoweredNode::Open(Box::new(OpenPipeline {
            source: probe_source,
            #[cfg(feature = "parquet")]
            parquet_scan_meta: None,
            operators: probe_ops,
            schema: final_schema,
            ordering: None,
        }))))
    }

    async fn lower_operators(
        &mut self,
        nodes: Vec<&LogicalPlan>,
        mut schema: SchemaRef,
    ) -> Result<std::result::Result<(Vec<Operator>, SchemaRef), String>> {
        let mut operators = Vec::with_capacity(nodes.len());
        let props = self.session.execution_props();
        for node in nodes.into_iter().rev() {
            match node {
                LogicalPlan::Filter(filter) => {
                    if let Err(reason) = self
                        .ensure_expr_scalar_subqueries(&filter.predicate)
                        .await?
                    {
                        return Ok(Err(reason));
                    }
                    let predicate = create_physical_expr(
                        &filter.predicate,
                        filter.input.schema(),
                        props,
                        &self.planning_ctx,
                    )?;
                    if predicate.data_type(&schema)? != DataType::Boolean {
                        return plan_err!(
                            "Morsel filter predicate must return Boolean: {predicate}"
                        );
                    }
                    operators.push(Operator::Filter {
                        predicate,
                        projection: None,
                    });
                }
                LogicalPlan::Projection(projection) => {
                    for expr in &projection.expr {
                        if let Err(reason) =
                            self.ensure_expr_scalar_subqueries(expr).await?
                        {
                            return Ok(Err(reason));
                        }
                    }
                    let input = projection.input.schema();
                    let expressions = projection
                        .expr
                        .iter()
                        .map(|expr| {
                            let alias = match expr {
                                Expr::Column(column) => schema
                                    .field(input.index_of_column(column)?)
                                    .name()
                                    .clone(),
                                _ => physical_name(expr)?,
                            };
                            Ok(ProjectionExpr {
                                expr: create_physical_expr(
                                    expr,
                                    input,
                                    props,
                                    &self.planning_ctx,
                                )?,
                                alias,
                            })
                        })
                        .collect::<Result<ProjectionExprs>>()?;
                    let projector = expressions.make_projector_with_schema_metadata(
                        &schema,
                        projection.schema.as_arrow(),
                    )?;
                    schema = Arc::clone(projector.output_schema());
                    operators.push(Operator::Projection(projector));
                }
                _ => unreachable!(),
            }
        }
        Ok(Ok((operators, schema)))
    }
}
