// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::HashMap;
use std::num::NonZeroUsize;
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow::datatypes::SchemaRef;
use datafusion_catalog::{ScanArgs, TableProvider};
use datafusion_common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion_common::{DFSchema, Result, internal_datafusion_err, internal_err};
use datafusion_common_runtime::JoinSet;
use datafusion_datasource::PartitionedFile;
use datafusion_datasource::file::FileSource;
use datafusion_datasource::file_compression_type::FileCompressionType;
use datafusion_datasource::file_scan_config::FileScanConfig;
use datafusion_datasource::morsel::Morselizer;
use datafusion_datasource::source::{DataSource, DataSourceExec};
use datafusion_datasource_parquet::{file_format::ParquetFormat, source::ParquetSource};
use datafusion_execution::TaskContext;
use datafusion_expr::expr_rewriter::unnormalize_cols;
use datafusion_expr::physical_planning_context::PhysicalPlanningContext;
use datafusion_expr::{Expr, TableScan};
use datafusion_physical_expr::expressions::Column;
use datafusion_physical_expr::projection::ProjectionExprs;
use datafusion_physical_expr::scalar_subquery::ScalarSubqueryExpr;
use datafusion_physical_expr::utils::{collect_columns, split_conjunction};
use datafusion_physical_expr::{PhysicalExpr, conjunction, create_physical_expr};
use datafusion_physical_expr_common::physical_expr::is_volatile;
use datafusion_physical_plan::empty::EmptyExec;
use datafusion_physical_plan::filter_pushdown::PushedDown;
use datafusion_physical_plan::metrics::{ExecutionPlanMetricsSet, MetricBuilder};
use datafusion_session::Session;

use crate::datasource::listing::ListingTable;

use super::hash_join::{JoinOutputCol, ProbeJoinKind};
use super::join_filter::{
    BoundJoinPredicates, DynamicJoinFilterConfig, DynamicJoinFilterData,
    JoinFilterBinding,
};
use super::metrics::RunMetrics;
use super::{Operator, Sink, Source, WorkerState, supported_expr};

mod ready;
#[cfg(test)]
mod tests;

pub(super) use ready::{ClaimedWork, ReadyReaders};
use ready::{Phase2Queue, Phase2Task, Publisher};

const MAX_READ_AHEAD: usize = 16;

#[derive(Debug)]
pub(super) struct ParquetScan {
    pub(super) table: String,
    pub(super) schema: SchemaRef,
    // An admitted listing scan may have no files, without losing its projected schema.
    config: Option<Box<FileScanConfig>>,
    files: Arc<[PartitionedFile]>,
    pushdown_filters: bool,
    dynamic_join_filters: Vec<JoinFilterBinding>,
    pub(super) metrics: ExecutionPlanMetricsSet,
}

impl ParquetScan {
    pub(super) async fn lower(
        table: &ListingTable,
        scan: &TableScan,
        session: &dyn Session,
    ) -> Result<std::result::Result<Self, String>> {
        let unsupported = |feature| {
            Ok(Err(format!(
                "native Parquet morsels do not support {feature} on {}",
                scan.table_name
            )))
        };
        let options = table.options();
        let Some(format) = options.format.downcast_ref::<ParquetFormat>() else {
            return unsupported("non-Parquet listing scans");
        };
        if !options.table_partition_cols.is_empty() {
            return unsupported("partition columns");
        }
        if !options.file_sort_order.is_empty() || options.output_partitioning.is_some() {
            return unsupported("declared file ordering or distribution");
        }
        let crypto = &format.options().crypto;
        if crypto.file_decryption.is_some() || crypto.factory_id.is_some() {
            return unsupported("Parquet encryption");
        }
        let filters: Vec<_> = unnormalize_cols(scan.filters.iter().cloned())
            .into_iter()
            .collect();
        for filter in &filters {
            if !supported_expr(filter)? {
                return unsupported("a scan-context or asynchronous filter");
            }
        }
        let leaf = table
            .scan_with_args(
                session,
                ScanArgs::default()
                    .with_projection(scan.projection.as_deref())
                    .with_filters(Some(&filters)),
            )
            .await?;
        let plan = leaf.plan();
        let mut source = Self {
            table: scan.table_name.to_string(),
            schema: plan.schema(),
            config: None,
            files: Arc::default(),
            pushdown_filters: false,
            dynamic_join_filters: Vec::new(),
            metrics: ExecutionPlanMetricsSet::new(),
        };
        if source.schema.as_ref() != scan.projected_schema.as_arrow() {
            return unsupported("a scan schema different from the logical projection");
        }
        if plan.is::<EmptyExec>() {
            return Ok(Ok(source));
        }
        let Some(config) = plan
            .downcast_ref::<DataSourceExec>()
            .and_then(|exec| exec.data_source().downcast_ref::<FileScanConfig>())
        else {
            return unsupported("a wrapped Parquet scan");
        };
        let Some(parquet) = config.file_source().downcast_ref::<ParquetSource>() else {
            return unsupported("a non-Parquet file source");
        };
        if config.limit.is_some()
            || config.preserves_file_order()
            || config.output_partitioning.is_some()
        {
            return unsupported("scan limits, ordering or distribution");
        }
        if config.file_compression_type != FileCompressionType::UNCOMPRESSED
            || config
                .file_groups
                .iter()
                .flat_map(|group| group.iter())
                .any(|file| {
                    file.range.is_some()
                        || !file.extensions.is_empty()
                        || !file.partition_values.is_empty()
                })
        {
            return unsupported("a split, compressed or extended file layout");
        }
        if !config.table_partition_cols().is_empty()
            || !parquet.table_schema().virtual_columns().is_empty()
        {
            return unsupported("partition or virtual columns");
        }
        if let Some(feature) = parquet.unsupported_buffered_morsel_source() {
            return unsupported(feature);
        }

        let pushdown_filters =
            session.config_options().execution.parquet.pushdown_filters
                || parquet.table_parquet_options().global.pushdown_filters;
        let mut config = config.clone();
        if !pushdown_filters {
            config.file_source = Arc::new(parquet.clone().with_pushdown_filters(false));
            Self::attach_pruning_hints(
                &mut config,
                &scan.filters,
                &scan.projected_schema,
                session,
            )?;
        }
        source.pushdown_filters = pushdown_filters;
        source.metrics = config.file_source().metrics().clone();
        source.files = std::mem::take(&mut config.file_groups)
            .into_iter()
            .flat_map(|group| group.into_inner())
            .collect();
        source.config = Some(Box::new(config));
        Ok(Ok(source))
    }

    fn attach_pruning_hints(
        config: &mut FileScanConfig,
        scan_filters: &[Expr],
        projected_schema: &DFSchema,
        session: &dyn Session,
    ) -> Result<()> {
        // The listing provider's file predicates are inexact. Attach pruning
        // hints in projected coordinates, without removing any logical Filter.
        let physical_hints = scan_filters
            .iter()
            .filter(|expr| {
                !expr.is_volatile()
                    && !expr
                        .exists(|e| Ok(matches!(e, Expr::ScalarSubquery(_))))
                        .unwrap_or(true)
                    && expr
                        .column_refs()
                        .iter()
                        .all(|col| projected_schema.has_column(col))
            })
            .map(|expr| {
                create_physical_expr(
                    expr,
                    projected_schema,
                    session.execution_props(),
                    &PhysicalPlanningContext::default(),
                )
            })
            .collect::<Result<Vec<_>>>()?;
        if !physical_hints.is_empty() {
            let mut prune_only = session.config_options().clone();
            prune_only.execution.parquet.pushdown_filters = false;
            let update = config.try_pushdown_filters(physical_hints, &prune_only)?;
            if update
                .filters
                .iter()
                .any(|pushed| matches!(pushed, PushedDown::Yes))
            {
                return internal_err!(
                    "Native Parquet scan received exact filter pushdown"
                );
            }
            if let Some(updated) = update.updated_node {
                *config = updated
                    .downcast_ref::<FileScanConfig>()
                    .ok_or_else(|| {
                        internal_datafusion_err!(
                            "Expected FileScanConfig from Parquet filter pushdown"
                        )
                    })?
                    .clone();
            }
        }
        Ok(())
    }

    pub(super) fn pushdown_operators(
        &mut self,
        operators: &mut Vec<Operator>,
        is_count_sink: bool,
        scan_filters: &[Expr],
        scan_projected_schema: &DFSchema,
        session: &dyn Session,
    ) -> Result<()> {
        if !self.pushdown_filters {
            return Ok(());
        }
        let Some(config) = self.config.as_mut() else {
            return Ok(());
        };

        let leading_filters = operators
            .iter()
            .take_while(|op| {
                matches!(
                    op,
                    Operator::Filter {
                        projection: None,
                        ..
                    }
                )
            })
            .count();
        if leading_filters == 0 {
            if !scan_filters.is_empty() {
                if let Some(parquet) =
                    config.file_source().downcast_ref::<ParquetSource>()
                {
                    config.file_source =
                        Arc::new(parquet.clone().with_pushdown_filters(false));
                }
                Self::attach_pruning_hints(
                    config,
                    scan_filters,
                    scan_projected_schema,
                    session,
                )?;
                self.metrics = config.file_source().metrics().clone();
            }
            return Ok(());
        }

        let drained: Vec<_> = operators.drain(..leading_filters).collect();
        let mut candidates = Vec::new();
        let mut residual = Vec::new();
        for op in drained {
            let Operator::Filter { predicate, .. } = op else {
                unreachable!();
            };
            for expr in split_conjunction(&predicate) {
                let has_scalar_subquery =
                    expr.exists(|e| Ok(e.is::<ScalarSubqueryExpr>()))?;
                if is_volatile(expr) || has_scalar_subquery {
                    residual.push(Arc::clone(expr));
                } else {
                    candidates.push(Arc::clone(expr));
                }
            }
        }
        if !candidates.is_empty() {
            let mut pushdown_cfg = session.config_options().clone();
            pushdown_cfg.execution.parquet.pushdown_filters = true;
            let update =
                config.try_pushdown_filters(candidates.clone(), &pushdown_cfg)?;
            if update.filters.len() != candidates.len() {
                return internal_err!(
                    "Parquet filter pushdown returned {} results for {} filters",
                    update.filters.len(),
                    candidates.len()
                );
            }
            for (expr, pushed) in candidates.into_iter().zip(update.filters) {
                if matches!(pushed, PushedDown::No) {
                    residual.push(expr);
                }
            }
            if let Some(updated) = update.updated_node {
                **config = updated
                    .downcast_ref::<FileScanConfig>()
                    .ok_or_else(|| {
                        internal_datafusion_err!(
                            "Expected FileScanConfig from Parquet filter pushdown"
                        )
                    })?
                    .clone();
            }
        }
        if !residual.is_empty() {
            operators.insert(
                0,
                Operator::Filter {
                    predicate: conjunction(residual),
                    projection: None,
                },
            );
        }

        // Narrow the Parquet output projection when pushed-down filter columns
        // are no longer referenced by remaining operators or the sink.
        let mut needed_cols = Vec::new();
        let mut first_projection = None;
        for (idx, op) in operators.iter().enumerate() {
            match op {
                Operator::Filter { predicate, .. } => {
                    needed_cols.extend(
                        collect_columns(predicate).into_iter().map(|c| c.index()),
                    );
                }
                Operator::Projection(projector) => {
                    needed_cols.extend(projector.projection().column_indices());
                    first_projection = Some(idx);
                    break;
                }
                Operator::HashProbe(_) => break,
            }
        }
        if first_projection.is_some() || is_count_sink {
            needed_cols.sort_unstable();
            needed_cols.dedup();
            if needed_cols.len() < self.schema.fields().len() {
                let narrow = ProjectionExprs::from_indices(&needed_cols, &self.schema);
                if let Some(updated) = config.try_swapping_with_projection(&narrow)? {
                    let updated = updated
                        .downcast_ref::<FileScanConfig>()
                        .ok_or_else(|| {
                            internal_datafusion_err!(
                                "Expected FileScanConfig from Parquet projection pushdown"
                            )
                        })?
                        .clone();
                    let narrowed_schema = updated.projected_schema()?;
                    if narrowed_schema.as_ref() == &self.schema.project(&needed_cols)? {
                        let end = first_projection.map_or(operators.len(), |i| i + 1);
                        let mut remapped = Vec::with_capacity(end);
                        for op in &operators[..end] {
                            match op {
                                Operator::Filter {
                                    predicate,
                                    projection,
                                } => {
                                    debug_assert!(projection.is_none());
                                    remapped.push(Operator::Filter {
                                        predicate: narrow.project_expr(predicate)?,
                                        projection: None,
                                    });
                                }
                                Operator::Projection(projector) => {
                                    let proj = projector
                                        .projection()
                                        .clone()
                                        .try_map_exprs(|expr| narrow.project_expr(&expr))?
                                        .make_projector_with_schema_metadata(
                                            &narrowed_schema,
                                            projector.output_schema(),
                                        )?;
                                    remapped.push(Operator::Projection(proj));
                                }
                                Operator::HashProbe(_) => unreachable!(),
                            }
                        }
                        for (slot, op) in operators[..end].iter_mut().zip(remapped) {
                            *slot = op;
                        }
                        **config = updated;
                        self.schema = narrowed_schema;
                    }
                }
            }
        }

        self.metrics = config.file_source().metrics().clone();
        Ok(())
    }

    pub(super) fn pushdown_filters(&self) -> bool {
        self.pushdown_filters
    }

    pub(super) fn pushed_predicate(&self) -> Option<Arc<dyn PhysicalExpr>> {
        self.config.as_ref()?.file_source().exact_filter()
    }

    pub(super) fn is_empty(&self) -> bool {
        self.files.is_empty()
    }

    pub(super) fn total_file_bytes(&self) -> u64 {
        self.files
            .iter()
            .map(|f| f.object_meta.size)
            .fold(0u64, u64::saturating_add)
    }

    pub(super) fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        match &self.config {
            Some(config) => config.apply_expressions(f),
            None => Ok(TreeNodeRecursion::Continue),
        }
    }

    pub(super) fn add_dynamic_join_filter(&mut self, binding: JoinFilterBinding) {
        if self.config.is_some() && !self.files.is_empty() {
            self.dynamic_join_filters.push(binding);
        }
    }

    pub(super) fn start(
        &self,
        context: &TaskContext,
        tasks: &mut JoinSet<Result<WorkerState>>,
        workers: NonZeroUsize,
        measurement: Option<&Arc<RunMetrics>>,
        published_filters: &HashMap<usize, Arc<DynamicJoinFilterData>>,
    ) -> Result<ReadyReaders> {
        let read_ahead = workers.get().min(MAX_READ_AHEAD);
        let (publish, phase2, ready) = ready::channel(
            read_ahead,
            measurement.cloned(),
            MetricBuilder::new(&self.metrics).counter("parquet_morsels_claimed", 0),
        );
        if let Some(base_config) = &self.config {
            let mut runtime_config_owned: Option<FileScanConfig> = None;
            if !self.dynamic_join_filters.is_empty() && !published_filters.is_empty() {
                let mut dynamic_preds = Vec::new();
                for binding in &self.dynamic_join_filters {
                    let Some(filter_data) = published_filters.get(&binding.filter_id)
                    else {
                        continue;
                    };
                    match binding.build_predicates(
                        filter_data,
                        self.pushdown_filters,
                        &self.schema,
                    )? {
                        BoundJoinPredicates::ShortCircuitEmpty => {
                            MetricBuilder::new(&self.metrics)
                                .counter("join_dynamic_filter_short_circuits", 0)
                                .add(1);
                            return Ok(ready);
                        }
                        BoundJoinPredicates::Predicates(preds) => {
                            dynamic_preds.extend(preds);
                        }
                    }
                }
                if !dynamic_preds.is_empty() {
                    let mut runtime_opts =
                        context.session_config().options().as_ref().clone();
                    runtime_opts.execution.parquet.pushdown_filters =
                        self.pushdown_filters;
                    let pushed_count = dynamic_preds.len();
                    let update =
                        base_config.try_pushdown_filters(dynamic_preds, &runtime_opts)?;
                    if let Some(updated) = update.updated_node {
                        let updated_cfg = updated
                            .downcast_ref::<FileScanConfig>()
                            .ok_or_else(|| {
                                internal_datafusion_err!(
                                    "Expected FileScanConfig from dynamic join filter pushdown"
                                )
                            })?
                            .clone();
                        MetricBuilder::new(&self.metrics)
                            .counter("join_dynamic_filters_pushed", 0)
                            .add(pushed_count);
                        runtime_config_owned = Some(updated_cfg);
                    }
                }
            }
            let config = runtime_config_owned.as_ref().unwrap_or(base_config);
            let parquet = config
                .file_source()
                .downcast_ref::<ParquetSource>()
                .ok_or_else(|| {
                    internal_datafusion_err!("Expected the admitted ParquetSource")
                })?;
            let has_pushed_filter = parquet.exact_filter().is_some();
            let work = Arc::new(FileWork {
                morselizer: parquet.create_buffered_morselizer(config, context)?,
                files: Arc::clone(&self.files),
                next: AtomicUsize::new(0),
                publish,
                measurement: measurement.cloned(),
            });
            // Each lane owns one planner and can drive at most one pending I/O
            // phase or hold one reader waiting to publish.
            for _ in 0..read_ahead.min(self.files.len()) {
                let work = Arc::clone(&work);
                tasks.spawn(async move {
                    work.produce().await?;
                    Ok(WorkerState::None)
                });
            }
            if has_pushed_filter && !self.files.is_empty() {
                // Phase-2 I/O lanes fetch post-RowFilter output ranges requested
                // by workers so workers never await object-store I/O.
                for _ in 0..read_ahead {
                    let phase2 = phase2.clone();
                    let measurement = measurement.cloned();
                    tasks.spawn(async move {
                        run_phase2_lane(phase2, measurement).await?;
                        Ok(WorkerState::None)
                    });
                }
            }
        }
        Ok(ready)
    }
}

fn source_schema(source: &Source) -> SchemaRef {
    match source {
        Source::Memory(m) => Arc::clone(&m.schema),
        Source::Parquet(p) => Arc::clone(&p.schema),
        Source::Stage(s) => Arc::clone(&s.schema),
    }
}

fn compute_pipeline_schemas(
    source: &Source,
    operators: &[Operator],
) -> Result<Vec<SchemaRef>> {
    let mut schemas = Vec::with_capacity(operators.len() + 1);
    schemas.push(source_schema(source));
    for op in operators {
        let prev = schemas.last().expect("non-empty schemas");
        let next = match op {
            Operator::Filter {
                projection: None, ..
            } => Arc::clone(prev),
            Operator::Filter {
                projection: Some(indices),
                ..
            } => Arc::new(prev.project(indices)?),
            Operator::Projection(projector) => Arc::clone(projector.output_schema()),
            Operator::HashProbe(probe) => Arc::clone(&probe.output_schema),
        };
        schemas.push(next);
    }
    Ok(schemas)
}

/// Bind dynamic join filters from every eligible `HashProbe` in `operators`
/// onto both `source` (if `Source::Parquet`) and any earlier `Inner` join's
/// `build_pipeline` (recursively), enabling cascading pushdown across
/// multi-join star schemas (e.g., TPC-H Q5 `region -> nation -> supplier ->
/// customer -> orders -> lineitem`).
pub(super) fn bind_pipeline_dynamic_join_filters(
    source: &mut Source,
    operators: &mut [Operator],
    max_inlist_values: usize,
    max_inlist_bytes: usize,
) -> Result<()> {
    let schemas = compute_pipeline_schemas(source, operators)?;

    for op_idx in 0..operators.len() {
        let (filter_id, probe_keys) = match &operators[op_idx] {
            Operator::HashProbe(probe) if probe.kind.allows_dynamic_filter_pushdown() => {
                (probe.filter_id, probe.probe_key_exprs().to_vec())
            }
            _ => continue,
        };
        if probe_keys.is_empty() {
            continue;
        }

        let key_opts: Vec<Option<Arc<dyn PhysicalExpr>>> =
            probe_keys.into_iter().map(Some).collect();
        let (bound_any, any_pushdown) = push_filter_into_prefix(
            source,
            &mut operators[..op_idx],
            &schemas[..=op_idx],
            filter_id,
            &key_opts,
        )?;
        if bound_any
            && let Operator::HashProbe(probe_arc) = &mut operators[op_idx]
            && let Some(probe) = Arc::get_mut(probe_arc)
            && let Some(build_pipeline) = Arc::get_mut(&mut probe.build_pipeline)
            && let Sink::HashJoinBuild(sink_arc) = &mut build_pipeline.sink
            && let Some(sink) = Arc::get_mut(sink_arc)
        {
            sink.enable_dynamic_filter(DynamicJoinFilterConfig {
                max_inlist_values,
                max_inlist_bytes,
                build_bloom: any_pushdown,
            });
        }
    }
    Ok(())
}

/// Push `key_exprs` (in `schemas[prefix_ops.len()]` coordinates) into:
/// 1. `source` (if `Source::Parquet`), and
/// 2. any `Inner` `HashProbe` inside `prefix_ops` (recursively into its
///    `build_pipeline`) when no non-`Inner` join lies in between.
///
/// Returns `(bound_any, any_target_has_pushdown_filters)`.
fn push_filter_into_prefix(
    source: &mut Source,
    prefix_ops: &mut [Operator],
    schemas: &[SchemaRef],
    filter_id: usize,
    key_exprs: &[Option<Arc<dyn PhysicalExpr>>],
) -> Result<(bool, bool)> {
    let mut bound_any = false;
    let mut any_pushdown = false;

    // 1. Try pushing directly onto `source` (if Parquet).
    if let Source::Parquet(pq) = source
        && pq.config.is_some()
        && !pq.files.is_empty()
    {
        let mut mapped_keys = Vec::with_capacity(key_exprs.len());
        for expr_opt in key_exprs {
            let mapped = match expr_opt {
                Some(e) => remap_probe_key_to_scan(Arc::clone(e), prefix_ops, schemas)?,
                None => None,
            };
            mapped_keys.push(mapped);
        }
        let all_keys_bound =
            !mapped_keys.is_empty() && mapped_keys.iter().all(Option::is_some);
        let any_key_bound = mapped_keys.iter().any(Option::is_some);
        if any_key_bound {
            pq.add_dynamic_join_filter(JoinFilterBinding {
                filter_id,
                probe_key_exprs: mapped_keys,
                all_keys_bound,
            });
            bound_any = true;
            any_pushdown |= pq.pushdown_filters();
        }
    }

    // 2. Try pushing onto earlier `Inner` joins' build pipelines in `prefix_ops`.
    // Walk backwards from the end of `prefix_ops`; stop as soon as we cross a
    // non-`Inner` join (whose unmatched rows could be altered by build pruning).
    for u in (0..prefix_ops.len()).rev() {
        let is_inner = match &prefix_ops[u] {
            Operator::HashProbe(p) => p.kind == ProbeJoinKind::Inner,
            _ => continue,
        };
        if !is_inner {
            break;
        }
        let mut build_key_opts: Vec<Option<Arc<dyn PhysicalExpr>>> =
            Vec::with_capacity(key_exprs.len());
        let mut any_mapped_to_build = false;
        for expr_opt in key_exprs {
            let mapped_to_u_out = match expr_opt {
                Some(e) => remap_probe_key_to_scan(
                    Arc::clone(e),
                    &prefix_ops[u + 1..],
                    &schemas[u + 1..],
                )?,
                None => None,
            };
            let mapped_to_build = match mapped_to_u_out {
                Some(e) => {
                    let Operator::HashProbe(p) = &prefix_ops[u] else {
                        unreachable!()
                    };
                    let Sink::HashJoinBuild(build_sink) = &p.build_pipeline.sink else {
                        unreachable!()
                    };
                    let build_schemas = compute_pipeline_schemas(
                        &p.build_pipeline.source,
                        &p.build_pipeline.operators,
                    )?;
                    let build_sink_input_schema = build_schemas.last().unwrap();
                    remap_expr_to_build_input(
                        e,
                        &p.output_projection,
                        build_sink.payload_columns(),
                        build_sink_input_schema,
                    )?
                }
                None => None,
            };
            if mapped_to_build.is_some() {
                any_mapped_to_build = true;
            }
            build_key_opts.push(mapped_to_build);
        }
        if any_mapped_to_build
            && let Operator::HashProbe(probe_arc) = &mut prefix_ops[u]
            && let Some(probe) = Arc::get_mut(probe_arc)
            && let Some(build_pipeline) = Arc::get_mut(&mut probe.build_pipeline)
        {
            let build_schemas = compute_pipeline_schemas(
                &build_pipeline.source,
                &build_pipeline.operators,
            )?;
            let (sub_bound, sub_pushdown) = push_filter_into_prefix(
                &mut build_pipeline.source,
                &mut build_pipeline.operators,
                &build_schemas,
                filter_id,
                &build_key_opts,
            )?;
            bound_any |= sub_bound;
            any_pushdown |= sub_pushdown;
        }
    }

    Ok((bound_any, any_pushdown))
}

fn remap_expr_to_build_input(
    expr: Arc<dyn PhysicalExpr>,
    output_projection: &[JoinOutputCol],
    payload_columns: &[usize],
    build_input_schema: &SchemaRef,
) -> Result<Option<Arc<dyn PhysicalExpr>>> {
    let mut saw_col = false;
    let mut unmappable = false;
    let mapped = expr
        .transform_up(|node| {
            if let Some(col) = node.downcast_ref::<Column>() {
                saw_col = true;
                match output_projection[col.index()] {
                    JoinOutputCol::Build(payload_pos) => {
                        let build_col_idx = payload_columns[payload_pos];
                        return Ok(Transformed::yes(Arc::new(Column::new(
                            build_input_schema.field(build_col_idx).name(),
                            build_col_idx,
                        ))));
                    }
                    JoinOutputCol::Probe(_) => {
                        unmappable = true;
                    }
                }
            }
            Ok(Transformed::no(node))
        })?
        .data;
    if !saw_col || unmappable {
        return Ok(None);
    }
    let has_subquery = mapped.exists(|e| Ok(e.is::<ScalarSubqueryExpr>()))?;
    if is_volatile(&mapped) || has_subquery {
        return Ok(None);
    }
    Ok(Some(mapped))
}

fn remap_probe_key_to_scan(
    mut expr: Arc<dyn PhysicalExpr>,
    preceding_ops: &[Operator],
    schemas: &[SchemaRef],
) -> Result<Option<Arc<dyn PhysicalExpr>>> {
    for step in (0..preceding_ops.len()).rev() {
        let prev_schema = &schemas[step];
        match &preceding_ops[step] {
            Operator::Filter {
                projection: None, ..
            } => {}
            Operator::Filter {
                projection: Some(indices),
                ..
            } => {
                expr = expr
                    .transform_up(|node| {
                        if let Some(col) = node.downcast_ref::<Column>() {
                            let src_idx = indices[col.index()];
                            return Ok(Transformed::yes(Arc::new(Column::new(
                                prev_schema.field(src_idx).name(),
                                src_idx,
                            ))));
                        }
                        Ok(Transformed::no(node))
                    })?
                    .data;
            }
            Operator::Projection(projector) => {
                let proj_exprs = projector.projection().as_ref();
                expr = expr
                    .transform_up(|node| {
                        if let Some(col) = node.downcast_ref::<Column>() {
                            return Ok(Transformed::yes(Arc::clone(
                                &proj_exprs[col.index()].expr,
                            )));
                        }
                        Ok(Transformed::no(node))
                    })?
                    .data;
            }
            Operator::HashProbe(mid_probe) => {
                if mid_probe.needs_build_emission() {
                    return Ok(None);
                }
                let mut unmappable = false;
                expr = expr
                    .transform_up(|node| {
                        if let Some(col) = node.downcast_ref::<Column>() {
                            match mid_probe.output_projection[col.index()] {
                                JoinOutputCol::Probe(p_idx) => {
                                    return Ok(Transformed::yes(Arc::new(Column::new(
                                        prev_schema.field(p_idx).name(),
                                        p_idx,
                                    ))));
                                }
                                JoinOutputCol::Build(_) => {
                                    unmappable = true;
                                }
                            }
                        }
                        Ok(Transformed::no(node))
                    })?
                    .data;
                if unmappable {
                    return Ok(None);
                }
            }
        }
    }
    let has_subquery = expr.exists(|e| Ok(e.is::<ScalarSubqueryExpr>()))?;
    if is_volatile(&expr) || has_subquery {
        return Ok(None);
    }
    Ok(Some(expr))
}

async fn run_phase2_lane(
    phase2: Phase2Queue,
    measurement: Option<Arc<RunMetrics>>,
) -> Result<()> {
    while let Some(Phase2Task {
        mut io,
        permit,
        filter_guard,
    }) = phase2.claim().await
    {
        loop {
            let next = match &measurement {
                Some(run) => Arc::clone(run).io(io).await?,
                None => io.await?,
            };
            let Some(mut step) = next.plan()? else {
                break;
            };
            let mut morsels = step.take_morsels();
            let mut ready = step.take_ready_planners();
            let pending = step.take_pending_planner();
            if let Some(morsel) = morsels.pop() {
                if !morsels.is_empty() || !ready.is_empty() || pending.is_some() {
                    return internal_err!(
                        "Expected a single decode-only Parquet morsel from Phase-2 planner"
                    );
                }
                phase2.publish_morsel(morsel, permit, filter_guard);
                break;
            }
            if ready.len() == 2 && pending.is_none() {
                let rg_filter = ready.pop().unwrap();
                phase2.publish_filter(rg_filter, permit, filter_guard);
                break;
            }
            if let Some(more_io) = pending {
                if !ready.is_empty() {
                    return internal_err!(
                        "Unexpected ready planner alongside Phase-2 pending I/O"
                    );
                }
                io = more_io;
                continue;
            }
            return internal_err!(
                "Unexpected MorselPlan from Phase-2 Parquet continuation"
            );
        }
    }
    Ok(())
}

struct FileWork {
    morselizer: Box<dyn Morselizer>,
    files: Arc<[PartitionedFile]>,
    next: AtomicUsize,
    publish: Publisher,
    measurement: Option<Arc<RunMetrics>>,
}

impl FileWork {
    async fn produce(&self) -> Result<()> {
        while let Some(file) = self.files.get(self.next.fetch_add(1, Ordering::Relaxed)) {
            self.produce_file(file).await.map_err(|error| {
                error.context(format!(
                    "Planning Parquet morsels for {}",
                    file.object_meta.location
                ))
            })?;
            // Files pruned without publishing a reader may not suspend in I/O.
            tokio::task::coop::consume_budget().await;
        }
        Ok(())
    }

    async fn produce_file(&self, file: &PartitionedFile) -> Result<()> {
        let mut planner = self.morselizer.plan_file(file.clone())?;
        while let Some(mut step) = planner.plan()? {
            let mut morsels = step.take_morsels();
            let mut ready = step.take_ready_planners();
            let pending = step.take_pending_planner();
            if ready.len() == 2 && morsels.is_empty() && pending.is_none() {
                let rg_filter = ready.pop().unwrap();
                let file_continuation = ready.pop().unwrap();
                self.publish.send_filter(rg_filter).await;
                planner = file_continuation;
                continue;
            }
            if morsels.len() > 1 || ready.len() + usize::from(pending.is_some()) > 1 {
                return internal_err!(
                    "Expected at most one Parquet reader and one file continuation per step"
                );
            }
            if let Some(morsel) = morsels.pop() {
                self.publish.send(morsel).await;
            }
            planner = if let Some(io) = pending {
                match &self.measurement {
                    Some(run) => Arc::clone(run).io(io).await?,
                    None => io.await?,
                }
            } else if let Some(next) = ready.pop() {
                next
            } else {
                break;
            };
        }
        Ok(())
    }
}
