// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::collections::{HashMap, HashSet};
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
use datafusion_datasource_parquet::{
    AdaptiveRowFilterConfig, AdaptiveRowFilterContext, file_format::ParquetFormat,
    source::ParquetSource,
};
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

use super::adaptive_filter::{self, CompiledFilter};
use super::dag::{ScalarSubqueryState, StageSpec, count_dag_stage_refs};
use super::hash_join::{JoinOutputCol, ProbeJoinKind};
use super::join_filter::{
    BoundJoinPredicates, DynamicJoinFilterConfig, DynamicJoinFilterData,
    JoinFilterBinding,
};
use super::metrics::RunMetrics;
use super::{Operator, Pipeline, PostSort, Sink, Source, WorkerState, supported_expr};

mod ready;
#[cfg(test)]
mod tests;

pub(super) use ready::{ClaimedWork, ReadyReaders};
use ready::{Phase2Queue, Phase2Task, Publisher};

const MAX_READ_AHEAD: usize = 16;

#[derive(Debug, Clone)]
struct BulkScanPlan {
    /// Wider file projection decoding both output columns and static predicate
    /// columns when `pushdown_operators` narrowed the scan schema (`None` when
    /// the output schema already contains all predicate columns).
    bulk_projection: Option<ProjectionExprs>,
    bulk_schema: SchemaRef,
    /// Mapping from `bulk_schema` column indices to `ParquetScan.schema`
    /// (`None` when identity).
    bulk_to_output: Option<Vec<usize>>,
    /// Exact static conjuncts pushed down (`PushedDown::Yes`), in
    /// `bulk_schema` coordinates.
    pushed_static_preds: Vec<Arc<dyn PhysicalExpr>>,
}

/// Worker-side execution plan for Mode B (`BulkDecode + BatchView`) row groups.
#[derive(Debug)]
pub(in crate::morsel) struct BulkFilterPlan {
    pub(in crate::morsel) compiled_filter: Option<CompiledFilter>,
    pub(in crate::morsel) bulk_to_output: Option<Vec<usize>>,
    pub(in crate::morsel) output_schema: SchemaRef,
}

#[derive(Debug)]
pub(super) struct ParquetScan {
    pub(super) table: String,
    pub(super) schema: SchemaRef,
    // An admitted listing scan may have no files, without losing its projected schema.
    config: Option<Box<FileScanConfig>>,
    files: Arc<[PartitionedFile]>,
    pushdown_filters: bool,
    bulk_scan_plan: Option<BulkScanPlan>,
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
            bulk_scan_plan: None,
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
        let mut pushed_static_preds = Vec::new();
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
                if matches!(pushed, PushedDown::Yes) {
                    pushed_static_preds.push(expr);
                } else {
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
        let mut narrowed_for_bulk = false;
        if first_projection.is_some() || is_count_sink {
            needed_cols.sort_unstable();
            needed_cols.dedup();
            if needed_cols.len() < self.schema.fields().len() {
                let pre_narrow_schema = Arc::clone(&self.schema);
                let pre_narrow_projection = config
                    .file_source()
                    .downcast_ref::<ParquetSource>()
                    .map(|p| p.projection_exprs().clone());
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
                        if !pushed_static_preds.is_empty()
                            && let Some(pre_proj) = pre_narrow_projection
                        {
                            // Keep only columns needed by either output or
                            // pushed static predicates in `bulk_schema`.
                            let mut bulk_cols = needed_cols.clone();
                            for expr in &pushed_static_preds {
                                bulk_cols.extend(
                                    collect_columns(expr).into_iter().map(|c| c.index()),
                                );
                            }
                            bulk_cols.sort_unstable();
                            bulk_cols.dedup();
                            if bulk_cols.len() == pre_narrow_schema.fields().len() {
                                self.bulk_scan_plan = Some(BulkScanPlan {
                                    bulk_projection: Some(pre_proj),
                                    bulk_schema: pre_narrow_schema,
                                    bulk_to_output: Some(needed_cols.clone()),
                                    pushed_static_preds: pushed_static_preds.clone(),
                                });
                            } else {
                                let bulk_narrow = ProjectionExprs::from_indices(
                                    &bulk_cols,
                                    &pre_narrow_schema,
                                );
                                let remapped_preds = pushed_static_preds
                                    .iter()
                                    .map(|e| bulk_narrow.project_expr(e))
                                    .collect::<Result<Vec<_>>>()?;
                                let bulk_to_output: Vec<usize> = needed_cols
                                    .iter()
                                    .map(|col_idx| {
                                        bulk_cols.binary_search(col_idx).expect(
                                            "needed_cols is a subset of bulk_cols",
                                        )
                                    })
                                    .collect();
                                let selected_proj_exprs: Vec<_> = bulk_cols
                                    .iter()
                                    .map(|&idx| pre_proj.as_ref()[idx].clone())
                                    .collect();
                                let bulk_schema =
                                    Arc::new(pre_narrow_schema.project(&bulk_cols)?);
                                self.bulk_scan_plan = Some(BulkScanPlan {
                                    bulk_projection: Some(ProjectionExprs::new(
                                        selected_proj_exprs,
                                    )),
                                    bulk_schema,
                                    bulk_to_output: Some(bulk_to_output),
                                    pushed_static_preds: remapped_preds,
                                });
                            }
                            narrowed_for_bulk = true;
                        }
                    }
                }
            }
        }
        if !narrowed_for_bulk && !pushed_static_preds.is_empty() {
            self.bulk_scan_plan = Some(BulkScanPlan {
                bulk_projection: None,
                bulk_schema: Arc::clone(&self.schema),
                bulk_to_output: None,
                pushed_static_preds,
            });
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

    pub(super) fn dynamic_join_filters(&self) -> &[JoinFilterBinding] {
        &self.dynamic_join_filters
    }

    pub(super) fn start(
        &self,
        context: &TaskContext,
        tasks: &mut JoinSet<Result<WorkerState>>,
        workers: NonZeroUsize,
        measurement: Option<&Arc<RunMetrics>>,
        published_filters: &HashMap<usize, Arc<DynamicJoinFilterData>>,
        keep_dynamic_preds_in_bulk: bool,
    ) -> Result<ReadyReaders> {
        let read_ahead = workers.get().min(MAX_READ_AHEAD);
        let claimed_counter =
            MetricBuilder::new(&self.metrics).counter("parquet_morsels_claimed", 0);
        let Some(base_config) = &self.config else {
            let (_, _, ready) =
                ready::channel(read_ahead, measurement.cloned(), claimed_counter, None);
            return Ok(ready);
        };

        let mut runtime_config_owned: Option<FileScanConfig> = None;
        let mut pushed_dynamic_preds: Vec<Arc<dyn PhysicalExpr>> = Vec::new();
        if !self.dynamic_join_filters.is_empty() && !published_filters.is_empty() {
            let mut dynamic_preds = Vec::new();
            for binding in &self.dynamic_join_filters {
                let Some(filter_data) = published_filters.get(&binding.filter_id) else {
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
                        let (_, _, ready) = ready::channel(
                            read_ahead,
                            measurement.cloned(),
                            claimed_counter,
                            None,
                        );
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
                runtime_opts.execution.parquet.pushdown_filters = self.pushdown_filters;
                let pushed_count = dynamic_preds.len();
                let update = base_config
                    .try_pushdown_filters(dynamic_preds.clone(), &runtime_opts)?;
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
                    if self.pushdown_filters {
                        for (pred, pushed) in
                            dynamic_preds.into_iter().zip(update.filters)
                        {
                            if matches!(pushed, PushedDown::Yes) {
                                pushed_dynamic_preds.push(pred);
                            }
                        }
                    }
                    runtime_config_owned = Some(updated_cfg);
                }
            }
        }
        let pushed_dynamic_filter = !pushed_dynamic_preds.is_empty();
        let config = runtime_config_owned.as_ref().unwrap_or(base_config);
        let parquet = config
            .file_source()
            .downcast_ref::<ParquetSource>()
            .ok_or_else(|| {
                internal_datafusion_err!("Expected the admitted ParquetSource")
            })?;
        let has_pushed_filter = parquet.exact_filter().is_some();
        let can_adapt = has_pushed_filter
            && self.pushdown_filters
            && (self.bulk_scan_plan.is_some() || pushed_dynamic_filter);

        let bulk_filter_plan = if can_adapt {
            let bulk_schema = self.bulk_scan_plan.as_ref().map_or_else(
                || Arc::clone(&self.schema),
                |plan| Arc::clone(&plan.bulk_schema),
            );
            let bulk_to_output = self
                .bulk_scan_plan
                .as_ref()
                .and_then(|plan| plan.bulk_to_output.clone());
            let mut bulk_preds: Vec<Arc<dyn PhysicalExpr>> = self
                .bulk_scan_plan
                .as_ref()
                .map(|plan| plan.pushed_static_preds.clone())
                .unwrap_or_default();
            if keep_dynamic_preds_in_bulk {
                for dyn_pred in pushed_dynamic_preds {
                    let remapped = if let Some(cols) = &bulk_to_output {
                        dyn_pred
                            .transform_up(|node| {
                                if let Some(col) = node.downcast_ref::<Column>() {
                                    let bulk_idx = cols[col.index()];
                                    return Ok(Transformed::yes(Arc::new(Column::new(
                                        bulk_schema.field(bulk_idx).name(),
                                        bulk_idx,
                                    ))));
                                }
                                Ok(Transformed::no(node))
                            })?
                            .data
                    } else {
                        dyn_pred
                    };
                    bulk_preds.push(remapped);
                }
            }
            let compiled_filter = (!bulk_preds.is_empty())
                .then(|| CompiledFilter::try_new(conjunction(bulk_preds), &bulk_schema))
                .transpose()?;
            Some(Arc::new(BulkFilterPlan {
                compiled_filter,
                bulk_to_output,
                output_schema: Arc::clone(&self.schema),
            }))
        } else {
            None
        };

        let (publish, phase2, ready) = ready::channel(
            read_ahead,
            measurement.cloned(),
            claimed_counter,
            bulk_filter_plan,
        );

        let adaptive_ctx = can_adapt.then(|| {
            let adaptive_cfg = context
                .session_config()
                .get_extension::<AdaptiveRowFilterConfig>()
                .map(|c| (*c).clone())
                .unwrap_or_default();
            let bulk_proj = self.bulk_scan_plan.as_ref().and_then(|plan| {
                plan.bulk_projection
                    .as_ref()
                    .map(|proj| (proj.clone(), Arc::clone(&plan.bulk_schema)))
            });
            Arc::new(AdaptiveRowFilterContext::new(
                adaptive_cfg,
                self.files.len(),
                pushed_dynamic_filter,
                bulk_proj,
                Some(Arc::new(adaptive_filter::analyze_conjunct)),
                &self.metrics,
            ))
        });
        let work = Arc::new(FileWork {
            morselizer: parquet.create_adaptive_buffered_morselizer(
                config,
                context,
                adaptive_ctx,
            )?,
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

fn compute_operator_schemas(
    base_schema: &SchemaRef,
    operators: &[Operator],
) -> Result<Vec<SchemaRef>> {
    let mut schemas = Vec::with_capacity(operators.len() + 1);
    schemas.push(Arc::clone(base_schema));
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

fn compute_pipeline_schemas(
    source: &Source,
    operators: &[Operator],
) -> Result<Vec<SchemaRef>> {
    compute_operator_schemas(&source_schema(source), operators)
}

/// Bind dynamic join filters across the entire DAG (root pipeline, intermediate
/// stages, and nested join build pipelines), pushing filters into both direct
/// probe `ParquetScan`s, earlier `Inner`/`BuildSemi`/`BuildAnti` join build
/// pipelines, and single-consumer upstream grouped-`Aggregate`/`GroupCount`/`Sort`
/// stages.
pub(super) fn bind_dag_dynamic_join_filters(
    stages: &mut [StageSpec],
    scalar_subqueries: Option<&ScalarSubqueryState>,
    root_pipeline: &mut Pipeline,
    max_inlist_values: usize,
    max_inlist_bytes: usize,
) -> Result<()> {
    let stage_ref_counts = count_dag_stage_refs(stages, scalar_subqueries, root_pipeline);

    bind_pipeline_dynamic_join_filters_with_stages(
        &mut root_pipeline.source,
        &mut root_pipeline.operators,
        stages,
        &stage_ref_counts,
        max_inlist_values,
        max_inlist_bytes,
    )?;

    for idx in (0..stages.len()).rev() {
        let (earlier_stages, rest) = stages.split_at_mut(idx);
        if let Some(pipeline) = Arc::get_mut(&mut rest[0].pipeline) {
            bind_pipeline_dynamic_join_filters_with_stages(
                &mut pipeline.source,
                &mut pipeline.operators,
                earlier_stages,
                &stage_ref_counts,
                max_inlist_values,
                max_inlist_bytes,
            )?;
        }
    }
    Ok(())
}

pub(super) fn collect_pipeline_bound_filter_ids(
    pipeline: &Pipeline,
    stages: &[StageSpec],
    out: &mut HashSet<usize>,
) {
    match &pipeline.source {
        Source::Parquet(pq) => {
            for binding in pq.dynamic_join_filters() {
                out.insert(binding.filter_id);
            }
        }
        Source::Stage(stage_src) => {
            if let Some(spec) = stages.get(stage_src.stage_id.0) {
                collect_pipeline_bound_filter_ids(&spec.pipeline, stages, out);
            }
        }
        Source::Memory(_) => {}
    }
    for op in &pipeline.operators {
        if let Operator::HashProbe(probe) = op {
            collect_pipeline_bound_filter_ids(&probe.build_pipeline, stages, out);
        }
    }
}

fn build_pipeline_is_filtered(pipeline: &Pipeline) -> bool {
    match &pipeline.source {
        Source::Stage(_) => return true,
        Source::Parquet(pq) if !pq.dynamic_join_filters().is_empty() => return true,
        _ => {}
    }
    if super::pipeline_filter_shift(&pipeline.source, &pipeline.operators) > 0 {
        return true;
    }
    pipeline.operators.iter().any(|op| {
        matches!(
            op,
            Operator::HashProbe(probe)
                if matches!(
                    probe.kind,
                    ProbeJoinKind::Inner
                        | ProbeJoinKind::ProbeSemi
                        | ProbeJoinKind::BuildSemi
                        | ProbeJoinKind::BuildAnti
                ) && build_pipeline_is_filtered(&probe.build_pipeline)
        )
    })
}

fn can_reach(adj: &[Vec<usize>], start: usize, target: usize) -> bool {
    if start == target {
        return true;
    }
    let mut visited = vec![false; adj.len()];
    let mut stack = vec![start];
    visited[start] = true;
    while let Some(u) = stack.pop() {
        for &v in &adj[u] {
            if v == target {
                return true;
            }
            if !visited[v] {
                visited[v] = true;
                stack.push(v);
            }
        }
    }
    false
}

fn bind_pipeline_dynamic_join_filters_with_stages(
    source: &mut Source,
    operators: &mut [Operator],
    stages_prefix: &mut [StageSpec],
    stage_ref_counts: &[usize],
    max_inlist_values: usize,
    max_inlist_bytes: usize,
) -> Result<()> {
    // Bind joins inside nested build pipelines first.
    for op in operators.iter_mut() {
        if let Operator::HashProbe(probe_arc) = op
            && let Some(probe) = Arc::get_mut(probe_arc)
            && let Some(build_pipeline) = Arc::get_mut(&mut probe.build_pipeline)
        {
            bind_pipeline_dynamic_join_filters_with_stages(
                &mut build_pipeline.source,
                &mut build_pipeline.operators,
                stages_prefix,
                stage_ref_counts,
                max_inlist_values,
                max_inlist_bytes,
            )?;
        }
    }

    let schemas = compute_pipeline_schemas(source, operators)?;

    let mut backward_bound: Vec<(usize, bool)> = Vec::new();
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
            stages_prefix,
            stage_ref_counts,
        )?;
        if bound_any {
            backward_bound.push((op_idx, any_pushdown));
        }
    }

    // Enable dynamic filters AFTER the full backward loop completes so earlier
    // builds that received a backward dynamic filter (e.g. TPC-H Q18 `orders`
    // filtered by `stage_0`) are recognized as filtered before sibling forward
    // propagation runs.
    for (op_idx, any_pushdown) in backward_bound {
        if let Operator::HashProbe(probe_arc) = &mut operators[op_idx]
            && let Some(probe) = Arc::get_mut(probe_arc)
            && let Some(build_pipeline) = Arc::get_mut(&mut probe.build_pipeline)
        {
            let build_is_filtered = build_pipeline_is_filtered(build_pipeline);
            if let Sink::HashJoinBuild(sink_arc) = &mut build_pipeline.sink
                && let Some(sink) = Arc::get_mut(sink_arc)
            {
                sink.enable_dynamic_filter(DynamicJoinFilterConfig {
                    max_inlist_values,
                    max_inlist_bytes,
                    build_bloom: any_pushdown,
                    build_is_filtered,
                });
            }
        }
    }

    // Sibling forward dynamic filter propagation across `Inner` / `ProbeSemi`
    // joins in the same pipeline sharing a probe key column (e.g., TPC-H Q9
    // `part.p_partkey = lineitem.l_partkey = partsupp.ps_partkey`).
    let probe_op_indices: Vec<usize> = operators
        .iter()
        .enumerate()
        .filter_map(|(i, op)| matches!(op, Operator::HashProbe(_)).then_some(i))
        .collect();
    if probe_op_indices.len() >= 2 {
        let mut filter_id_to_pos: HashMap<usize, usize> =
            HashMap::with_capacity(probe_op_indices.len());
        for (pos, &op_idx) in probe_op_indices.iter().enumerate() {
            if let Operator::HashProbe(p) = &operators[op_idx] {
                filter_id_to_pos.insert(p.filter_id, pos);
            }
        }
        let mut deps: Vec<Vec<usize>> = vec![Vec::new(); probe_op_indices.len()];
        let mut bound_ids = HashSet::new();
        for (dst_pos, &op_idx) in probe_op_indices.iter().enumerate() {
            bound_ids.clear();
            if let Operator::HashProbe(p) = &operators[op_idx] {
                collect_pipeline_bound_filter_ids(
                    &p.build_pipeline,
                    stages_prefix,
                    &mut bound_ids,
                );
            }
            for fid in &bound_ids {
                if let Some(&src_pos) = filter_id_to_pos.get(fid)
                    && src_pos != dst_pos
                {
                    deps[src_pos].push(dst_pos);
                }
            }
        }

        for a_pos in 0..probe_op_indices.len() {
            let a = probe_op_indices[a_pos];
            let (a_filter_id, a_null_eq, a_probe_keys, a_key_types, a_shift) =
                match &operators[a] {
                    Operator::HashProbe(p) if p.kind.allows_dynamic_filter_pushdown() => {
                        let Sink::HashJoinBuild(sink) = &p.build_pipeline.sink else {
                            continue;
                        };
                        let shift = super::pipeline_filter_shift(
                            &p.build_pipeline.source,
                            &p.build_pipeline.operators,
                        );
                        (
                            p.filter_id,
                            p.null_equality(),
                            p.probe_key_exprs().to_vec(),
                            sink.key_types().to_vec(),
                            shift,
                        )
                    }
                    _ => continue,
                };
            // Only propagate forward from genuinely filtered non-Stage builds
            // into unfiltered sibling builds so selective dimension tables run
            // before large unfiltered siblings without inverting stage-vs-dim order.
            if a_shift == 0 || a_probe_keys.is_empty() {
                continue;
            }

            let mut a_bound_any = false;
            let mut a_any_pushdown = false;

            for (b_pos, &b) in probe_op_indices.iter().enumerate().skip(a_pos + 1) {
                // Stop if any operator between `a` and `b` can fabricate NULL-padded
                // probe rows or emit unmatched build rows instead of filtering probe rows.
                let safe_span = operators[a + 1..b].iter().all(|op| match op {
                    Operator::Filter { .. } | Operator::Projection(_) => true,
                    Operator::HashProbe(mid) => matches!(
                        mid.kind,
                        ProbeJoinKind::Inner
                            | ProbeJoinKind::ProbeSemi
                            | ProbeJoinKind::ProbeAnti { .. }
                    ),
                });
                if !safe_span {
                    break;
                }
                if can_reach(&deps, b_pos, a_pos) {
                    continue;
                }

                let mapped_for_b_build: Option<Vec<Option<Arc<dyn PhysicalExpr>>>> = {
                    let Operator::HashProbe(b_probe) = &operators[b] else {
                        continue;
                    };
                    if !matches!(
                        b_probe.kind,
                        ProbeJoinKind::Inner | ProbeJoinKind::ProbeSemi
                    ) || b_probe.null_equality() != a_null_eq
                    {
                        continue;
                    }
                    let b_shift = super::pipeline_filter_shift(
                        &b_probe.build_pipeline.source,
                        &b_probe.build_pipeline.operators,
                    );
                    if b_shift > 0 {
                        continue;
                    }
                    let Sink::HashJoinBuild(b_sink) = &b_probe.build_pipeline.sink else {
                        continue;
                    };
                    let b_probe_keys = b_probe.probe_key_exprs();
                    let b_build_keys = b_sink.key_exprs();
                    let b_key_types = b_sink.key_types();
                    if b_probe_keys.len() != b_build_keys.len() {
                        continue;
                    }

                    let mut remapped_b_keys: Vec<Option<Arc<dyn PhysicalExpr>>> =
                        Vec::with_capacity(b_probe_keys.len());
                    for bk in b_probe_keys {
                        remapped_b_keys.push(remap_probe_key_to_scan(
                            Arc::clone(bk),
                            &operators[a..b],
                            &schemas[a..=b],
                        )?);
                    }

                    let mut target_keys: Vec<Option<Arc<dyn PhysicalExpr>>> =
                        vec![None; a_probe_keys.len()];
                    let mut matched_any = false;
                    for (j, (ak, a_dt)) in
                        a_probe_keys.iter().zip(a_key_types.iter()).enumerate()
                    {
                        for (k, rb_opt) in remapped_b_keys.iter().enumerate() {
                            if let Some(rb) = rb_opt
                                && rb.as_ref() == ak.as_ref()
                                && b_key_types.get(k) == Some(a_dt)
                            {
                                target_keys[j] = Some(Arc::clone(&b_build_keys[k]));
                                matched_any = true;
                                break;
                            }
                        }
                    }
                    matched_any.then_some(target_keys)
                };

                if let Some(target_keys) = mapped_for_b_build
                    && let Operator::HashProbe(b_arc) = &mut operators[b]
                    && let Some(b_probe) = Arc::get_mut(b_arc)
                    && let Some(b_build_pipeline) =
                        Arc::get_mut(&mut b_probe.build_pipeline)
                {
                    let b_build_schemas = compute_pipeline_schemas(
                        &b_build_pipeline.source,
                        &b_build_pipeline.operators,
                    )?;
                    let (sub_bound, sub_pushdown) = push_filter_into_prefix(
                        &mut b_build_pipeline.source,
                        &mut b_build_pipeline.operators,
                        &b_build_schemas,
                        a_filter_id,
                        &target_keys,
                        stages_prefix,
                        stage_ref_counts,
                    )?;
                    if sub_bound {
                        deps[a_pos].push(b_pos);
                        a_bound_any = true;
                        a_any_pushdown |= sub_pushdown;
                    }
                }
            }

            if a_bound_any
                && let Operator::HashProbe(a_arc) = &mut operators[a]
                && let Some(a_probe) = Arc::get_mut(a_arc)
                && let Some(a_build_pipeline) = Arc::get_mut(&mut a_probe.build_pipeline)
                && let Sink::HashJoinBuild(a_sink_arc) = &mut a_build_pipeline.sink
                && let Some(a_sink) = Arc::get_mut(a_sink_arc)
            {
                a_sink.enable_dynamic_filter(DynamicJoinFilterConfig {
                    max_inlist_values,
                    max_inlist_bytes,
                    build_bloom: a_any_pushdown,
                    build_is_filtered: true,
                });
            }
        }
    }
    Ok(())
}

/// Push `key_exprs` (in `schemas[prefix_ops.len()]` coordinates) into:
/// 1. `source` (if `Source::Parquet`, or recursively across a single-consumer
///    `Source::Stage` backed by a grouped `Aggregate`, `GroupCount`, `Sort`, or
///    `Result` sink), and
/// 2. any earlier `Inner`/`BuildSemi`/`BuildAnti` `HashProbe` inside `prefix_ops`
///    (recursively into its `build_pipeline`).
///
/// Returns `(bound_any, any_target_has_pushdown_filters)`.
fn push_filter_into_prefix(
    source: &mut Source,
    prefix_ops: &mut [Operator],
    schemas: &[SchemaRef],
    filter_id: usize,
    key_exprs: &[Option<Arc<dyn PhysicalExpr>>],
    stages_prefix: &mut [StageSpec],
    stage_ref_counts: &[usize],
) -> Result<(bool, bool)> {
    let mut bound_any = false;
    let mut any_pushdown = false;

    // 1. Try pushing directly onto `source` (Parquet or single-consumer Stage).
    match source {
        Source::Parquet(pq) if pq.config.is_some() && !pq.files.is_empty() => {
            let mut mapped_keys = Vec::with_capacity(key_exprs.len());
            for expr_opt in key_exprs {
                let mapped = match expr_opt {
                    Some(e) => {
                        remap_probe_key_to_scan(Arc::clone(e), prefix_ops, schemas)?
                    }
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
        Source::Stage(stage_src)
            if stage_src.stage_id.0 < stages_prefix.len()
                && stage_ref_counts[stage_src.stage_id.0] == 1 =>
        {
            let mut stage_out_keys = Vec::with_capacity(key_exprs.len());
            for expr_opt in key_exprs {
                let mapped = match expr_opt {
                    Some(e) => {
                        remap_probe_key_to_scan(Arc::clone(e), prefix_ops, schemas)?
                    }
                    None => None,
                };
                stage_out_keys.push(mapped);
            }
            if stage_out_keys.iter().any(Option::is_some) {
                let (earlier_stages, rest) =
                    stages_prefix.split_at_mut(stage_src.stage_id.0);
                let target_stage = &mut rest[0];
                if let Some(target_pipeline) = Arc::get_mut(&mut target_stage.pipeline) {
                    let target_schemas = compute_pipeline_schemas(
                        &target_pipeline.source,
                        &target_pipeline.operators,
                    )?;
                    let pre_sink_schema =
                        target_schemas.last().expect("non-empty schemas");
                    if let Some(pre_sink_keys) = remap_keys_through_stage_sink(
                        &target_pipeline.sink,
                        pre_sink_schema,
                        &stage_out_keys,
                    )? && pre_sink_keys.iter().any(Option::is_some)
                    {
                        let (sub_bound, sub_pushdown) = push_filter_into_prefix(
                            &mut target_pipeline.source,
                            &mut target_pipeline.operators,
                            &target_schemas,
                            filter_id,
                            &pre_sink_keys,
                            earlier_stages,
                            stage_ref_counts,
                        )?;
                        bound_any |= sub_bound;
                        any_pushdown |= sub_pushdown;
                    }
                }
            }
        }
        _ => {}
    }

    // 2. Try pushing onto earlier `Inner`/`BuildSemi`/`BuildAnti` joins' build
    // pipelines in `prefix_ops`. Stop as soon as we cross an incompatible join.
    for u in (0..prefix_ops.len()).rev() {
        let kind = match &prefix_ops[u] {
            Operator::HashProbe(p) => p.kind,
            _ => continue,
        };
        if !matches!(
            kind,
            ProbeJoinKind::Inner | ProbeJoinKind::BuildSemi | ProbeJoinKind::BuildAnti
        ) {
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
                stages_prefix,
                stage_ref_counts,
            )?;
            bound_any |= sub_bound;
            any_pushdown |= sub_pushdown;
        }
        // BuildSemi/BuildAnti only emit build columns; earlier probe columns do not survive.
        if kind != ProbeJoinKind::Inner {
            break;
        }
    }

    Ok((bound_any, any_pushdown))
}

type MappedKeyExprs = Vec<Option<Arc<dyn PhysicalExpr>>>;

fn remap_keys_through_ops(
    keys: &[Option<Arc<dyn PhysicalExpr>>],
    ops: &[Operator],
    base_schema: &SchemaRef,
) -> Result<MappedKeyExprs> {
    if ops.is_empty() {
        return Ok(keys.to_vec());
    }
    let schemas = compute_operator_schemas(base_schema, ops)?;
    let mut out = Vec::with_capacity(keys.len());
    for key in keys {
        let mapped = match key {
            Some(e) => remap_probe_key_to_scan(Arc::clone(e), ops, &schemas)?,
            None => None,
        };
        out.push(mapped);
    }
    Ok(out)
}

/// Remap join key expressions from a stage's final output schema backward
/// through its `Sink` (and any post-sink `result_operators` / `Sort`) into the
/// stage pipeline's pre-sink schema.
///
/// Only grouped `Aggregate` / `GroupCount` (direct pass-through `GROUP BY`
/// column references, never ungrouped, `TopN`, or `Sort` with `OFFSET`/`LIMIT`),
/// plain `Sort` (without `OFFSET`/`LIMIT`), and `Result` stages allow sound pushdown.
fn remap_keys_through_stage_sink(
    sink: &Sink,
    pre_sink_schema: &SchemaRef,
    stage_out_keys: &[Option<Arc<dyn PhysicalExpr>>],
) -> Result<Option<MappedKeyExprs>> {
    let remap_post_sort = |post_sort: &Option<PostSort>,
                           keys: &[Option<Arc<dyn PhysicalExpr>>]|
     -> Result<Option<MappedKeyExprs>> {
        match post_sort {
            None => Ok(Some(keys.to_vec())),
            Some(PostSort::Sort {
                sink: sort_sink,
                result_operators,
            }) => {
                if sort_sink.has_skip_or_fetch() {
                    return Ok(None);
                }
                Ok(Some(remap_keys_through_ops(
                    keys,
                    result_operators,
                    sort_sink.input_schema(),
                )?))
            }
            Some(PostSort::TopN { .. }) => Ok(None),
        }
    };

    match sink {
        Sink::Aggregate {
            sink: agg_sink,
            result_operators,
            post_sort,
        } => {
            if !agg_sink.is_grouped() {
                return Ok(None);
            }
            let Some(after_agg_keys) = remap_post_sort(post_sort, stage_out_keys)? else {
                return Ok(None);
            };
            let agg_out_keys = remap_keys_through_ops(
                &after_agg_keys,
                result_operators,
                agg_sink.schema(),
            )?;
            let group_exprs = agg_sink.group_exprs();
            let mut pre_sink_keys = Vec::with_capacity(agg_out_keys.len());
            for key_opt in agg_out_keys {
                let mapped = key_opt.and_then(|expr| {
                    let out_col = expr.downcast_ref::<Column>()?;
                    let group_expr = group_exprs.get(out_col.index())?;
                    let in_col = group_expr.downcast_ref::<Column>()?;
                    let field = pre_sink_schema.fields().get(in_col.index())?;
                    Some(Arc::new(Column::new(field.name(), in_col.index()))
                        as Arc<dyn PhysicalExpr>)
                });
                pre_sink_keys.push(mapped);
            }
            Ok(Some(pre_sink_keys))
        }
        Sink::GroupCount {
            sink: gc_sink,
            result_operators,
            post_sort,
        } => {
            let Some(after_gc_keys) = remap_post_sort(post_sort, stage_out_keys)? else {
                return Ok(None);
            };
            let gc_out_keys = remap_keys_through_ops(
                &after_gc_keys,
                result_operators,
                gc_sink.schema(),
            )?;
            let mut pre_sink_keys = Vec::with_capacity(gc_out_keys.len());
            for key_opt in gc_out_keys {
                let mapped = key_opt.and_then(|expr| {
                    let out_col = expr.downcast_ref::<Column>()?;
                    if out_col.index() != 0 {
                        return None;
                    }
                    let field = pre_sink_schema.fields().get(gc_sink.key_index)?;
                    Some(Arc::new(Column::new(field.name(), gc_sink.key_index))
                        as Arc<dyn PhysicalExpr>)
                });
                pre_sink_keys.push(mapped);
            }
            Ok(Some(pre_sink_keys))
        }
        Sink::Sort {
            sink: sort_sink,
            result_operators,
        } => {
            if sort_sink.has_skip_or_fetch() {
                return Ok(None);
            }
            Ok(Some(remap_keys_through_ops(
                stage_out_keys,
                result_operators,
                sort_sink.input_schema(),
            )?))
        }
        Sink::Result => Ok(Some(stage_out_keys.to_vec())),
        Sink::Count { .. }
        | Sink::TopN { .. }
        | Sink::Limit { .. }
        | Sink::HashJoinBuild(_) => Ok(None),
    }
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
    let mut crossed_residual_filter = false;
    for step in (0..preceding_ops.len()).rev() {
        let prev_schema = &schemas[step];
        match &preceding_ops[step] {
            Operator::Filter {
                projection: None, ..
            } => {
                crossed_residual_filter = true;
            }
            Operator::Filter {
                projection: Some(indices),
                ..
            } => {
                crossed_residual_filter = true;
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
    // Never push a potentially-fallible expression (e.g. `CAST(s AS BIGINT)`)
    // below an unpushed residual `Operator::Filter` that may guard it.
    if crossed_residual_filter && !expr.is::<Column>() {
        return Ok(None);
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
