// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! The dynamic filters of a [`PhysicalHashAggregate`](super::PhysicalHashAggregate):
//! predicates of its input rows that it tightens while it aggregates them,
//! with which the nodes below it that read them (e.g. scans of Parquet
//! files) skip the files, row groups and rows that cannot change its result.
//!
//! # DataFusion's filters
//!
//! DataFusion's partial `AggregateExec` of an aggregation without `GROUP BY`
//! whose aggregates are all `MIN` or `MAX` of columns produces a dynamic
//! filter (see `ExecutionPlan::dynamic_expressions_produced`), which its
//! `FilterPushdown` rule pushes into the scans below it. Its partitions
//! share the *bounds* of the aggregates, the least value of every `MIN` and
//! the greatest value of every `MAX` so far, and after every batch they set
//! the filter to the rows that may change them: `a < min(a) OR b > max(b) OR
//! ...` (`true` while an aggregate has no value yet). An aggregation lowered
//! from such an `AggregateExec` updates the same filter the same way, with the
//! bounds of the accumulators of all of its threads after every chunk (see
//! [`AggregateFilter::update_bounds`]), and marks it complete once it
//! aggregated its input.
//!
//! # Which values are bounded
//!
//! A filter compares the values of a column with a value of the aggregation,
//! and the scans prune with the statistics of the column, so the native
//! aggregation only sets the filters of DataFusion's built-in `MIN` and `MAX`
//! (identified by the types that implement them: DataFusion goes by their
//! names, so that a user-defined function named `max` gets a filter too), of
//! columns whose values the comparisons and the statistics order like the
//! aggregates do (see [`has_exact_bounds`]). Not floating point numbers: the
//! statistics of Parquet files leave their `NaN`s out, so that a scan would
//! skip a row group whose `NaN` is the `MAX` of the aggregation (a `NaN` is
//! greater than any other value, and a `-NaN` less: Arrow orders floats
//! totally). The filters of other aggregations stay `true`, which prunes
//! nothing.
//!
//! As DataFusion's, the filters are only updated in sessions that push the
//! dynamic filters of aggregations down
//! (`datafusion.optimizer.enable_aggregate_dynamic_filter_pushdown`).
//!
//! A filter is shared by all executions of the aggregation, like the dynamic
//! filters of DataFusion's plans: an execution after another one starts from
//! the filter that the first one left, so the plan must be reset before it is
//! executed again (see `PipelineRootExec::reset_state`, which lowers the
//! reset plan again).

use std::any::TypeId;
use std::cmp::Ordering;
use std::sync::Arc;

use arrow::datatypes::{DataType, SchemaRef};
use datafusion_common::{Result, ScalarValue, internal_err};
use datafusion_expr::{Accumulator, Operator};
use datafusion_functions_aggregate::min_max;
use datafusion_physical_expr::PhysicalExpr;
use datafusion_physical_expr::expressions::{
    BinaryExpr, Column, DynamicFilterPhysicalExpr, lit,
};
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::aggregates::AggregateExec;
use parking_lot::Mutex;

use super::hash_aggregate::implementation_type;
use crate::operator::ExecutionContext;

/// A dynamic filter that an aggregation updates while it aggregates its
/// input, see the [module documentation](self).
#[derive(Debug)]
pub(crate) struct AggregateFilter {
    /// The filters, which nodes of the input read: those that DataFusion's
    /// `AggregateExec` produces. All of them are set to the same predicate.
    filters: Vec<Arc<DynamicFilterPhysicalExpr>>,
    /// The aggregates of the aggregation, all of them: the filters select the
    /// rows that may change their bounds, `a < min(a) OR b > max(b) OR ...`
    /// (see [`AggregateFilter::update_bounds`]).
    bounded: Vec<BoundedAggregate>,
}

/// A `MIN` or `MAX` aggregate whose bound an [`AggregateFilter`] compares
/// its argument with.
#[derive(Debug)]
pub(crate) struct BoundedAggregate {
    /// The index of the aggregate among the aggregates of the aggregation.
    aggregate: usize,
    /// Whether the aggregate is a `MAX` (otherwise a `MIN`).
    max: bool,
    /// The argument of the aggregate: a column of the input rows.
    argument: Arc<dyn PhysicalExpr>,
}

impl BoundedAggregate {
    /// Whether `value` is a better bound than `bound` (`NULL`: none yet): a
    /// smaller value for a `MIN`, a greater one for a `MAX`.
    fn is_better(&self, value: &ScalarValue, bound: &ScalarValue) -> Result<bool> {
        if value.is_null() {
            return Ok(false);
        }
        if bound.is_null() {
            return Ok(true);
        }
        let expected = match self.max {
            true => Ordering::Greater,
            false => Ordering::Less,
        };
        match value.partial_cmp(bound) {
            Some(ordering) => Ok(ordering == expected),
            None => internal_err!(
                "Cannot compare the bounds {value:?} and {bound:?} of an aggregate"
            ),
        }
    }
}

impl AggregateFilter {
    /// The dynamic filters that `raw`, the `AggregateExec` that consumes the
    /// input rows of an aggregation, produces, if they are DataFusion's
    /// filters of the bounds of its `MIN` and `MAX` aggregates that the
    /// native aggregation sets (see [`bounded_aggregates`]).
    pub(crate) fn of_aggregate_exec(raw: &AggregateExec) -> Option<Self> {
        let filters = dynamic_filters(raw)?;
        if filters.is_empty() {
            return None;
        }
        Some(Self {
            filters,
            bounded: bounded_aggregates(raw)?,
        })
    }

    /// The filters that the aggregation updates.
    #[cfg(test)]
    pub(crate) fn filters(&self) -> &[Arc<DynamicFilterPhysicalExpr>] {
        &self.filters
    }

    /// The bounds of all threads of an execution of the aggregation: `NULL`
    /// (no bound yet).
    pub(crate) fn shared_bounds(&self) -> SharedBounds {
        SharedBounds {
            bounds: Mutex::new(vec![ScalarValue::Null; self.bounded.len()]),
        }
    }

    /// Update `shared`, the bounds of all threads, with the values of
    /// `accumulators`, the accumulators of the aggregates of a thread that
    /// aggregated a chunk, and set the filter to the rows that may change
    /// them if they changed, like the partitions of DataFusion's
    /// `AggregateExec` do after every batch: `a < min(a) OR b > max(b) OR
    /// ...`, or `true` while an aggregate has no bound (a row that does not
    /// change the other bounds may give it its first value).
    ///
    /// `known` are the bounds of all threads that the thread saw last: only
    /// if the value of one of its accumulators is better, the thread locks
    /// the bounds of all threads (and updates `known`), so that the threads
    /// rarely wait for each other once the bounds are tight.
    pub(crate) fn update_bounds(
        &self,
        accumulators: &mut [Box<dyn Accumulator>],
        known: &mut [ScalarValue],
        shared: &SharedBounds,
    ) -> Result<()> {
        let bounded = &self.bounded;
        let mut values = Vec::with_capacity(bounded.len());
        let mut improves = false;
        for (aggregate, known) in bounded.iter().zip(known.iter()) {
            let Some(accumulator) = accumulators.get_mut(aggregate.aggregate) else {
                return internal_err!(
                    "No accumulator for the bounded aggregate {}",
                    aggregate.aggregate
                );
            };
            let value = accumulator.evaluate()?;
            improves |= aggregate.is_better(&value, known)?;
            values.push(value);
        }
        if !improves {
            return Ok(());
        }
        let mut bounds = shared.bounds.lock();
        let mut changed = false;
        for ((aggregate, value), bound) in
            bounded.iter().zip(values).zip(bounds.iter_mut())
        {
            if aggregate.is_better(&value, bound)? {
                *bound = value;
                changed = true;
            }
        }
        known.clone_from_slice(&bounds);
        if changed {
            // Updated under the lock, so that the filter is set to the
            // tightest bounds last.
            let predicate = bounds_predicate(bounded, &bounds);
            for filter in &self.filters {
                filter.update(Arc::clone(&predicate))?;
            }
        }
        Ok(())
    }

    /// Mark the filters complete: the aggregation will not change them any
    /// more.
    pub(crate) fn mark_complete(&self) {
        for filter in &self.filters {
            filter.mark_complete();
        }
    }
}

/// `a < min(a) OR b > max(b) OR ...` for the `bounds` of the aggregates
/// `bounded`, or `true` if an aggregate has no bound yet, like DataFusion's
/// `AggregateExec` builds it.
fn bounds_predicate(
    bounded: &[BoundedAggregate],
    bounds: &[ScalarValue],
) -> Arc<dyn PhysicalExpr> {
    if bounds.iter().any(ScalarValue::is_null) {
        return lit(true);
    }
    bounded
        .iter()
        .zip(bounds)
        .map(|(aggregate, bound)| {
            let op = match aggregate.max {
                true => Operator::Gt,
                false => Operator::Lt,
            };
            Arc::new(BinaryExpr::new(
                Arc::clone(&aggregate.argument),
                op,
                lit(bound.clone()),
            )) as Arc<dyn PhysicalExpr>
        })
        .reduce(|left, right| Arc::new(BinaryExpr::new(left, Operator::Or, right)))
        .unwrap_or_else(|| lit(true))
}

/// The bounds of the `MIN` and `MAX` aggregates of all threads of an
/// execution of an aggregation (see [`AggregateFilter::update_bounds`]):
/// DataFusion's `shared_bound`s of its `AggregateExec`.
#[derive(Debug)]
pub(crate) struct SharedBounds {
    /// The bound of every bounded aggregate, `NULL` if it has none yet.
    bounds: Mutex<Vec<ScalarValue>>,
}

impl SharedBounds {
    /// No bounds, as a thread knows them before it aggregated a row.
    pub(crate) fn unknown(&self) -> Vec<ScalarValue> {
        vec![ScalarValue::Null; self.bounds.lock().len()]
    }
}

/// Whether the session of `ctx` updates the dynamic filters of aggregations,
/// as DataFusion's `AggregateExec` does only if
/// `datafusion.optimizer.enable_aggregate_dynamic_filter_pushdown` is set.
pub(crate) fn filters_enabled(ctx: &ExecutionContext) -> bool {
    ctx.task_ctx
        .session_config()
        .options()
        .optimizer
        .enable_aggregate_dynamic_filter_pushdown
}

/// The dynamic filters that `plan` produces (see
/// `ExecutionPlan::dynamic_expressions_produced`), `None` if one of them is
/// not a [`DynamicFilterPhysicalExpr`].
fn dynamic_filters(
    plan: &dyn ExecutionPlan,
) -> Option<Vec<Arc<DynamicFilterPhysicalExpr>>> {
    plan.dynamic_expressions_produced()
        .into_iter()
        .map(|filter| {
            let filter: Arc<dyn std::any::Any + Send + Sync> = filter;
            filter.downcast::<DynamicFilterPhysicalExpr>().ok()
        })
        .collect()
}

/// The aggregates of `raw`, an `AggregateExec` that consumes input rows, whose
/// bounds a filter compares their arguments with (see
/// [`FilterKind::Bounds`]), if it has no `GROUP BY` and all of its aggregates
/// are DataFusion's built-in `MIN` or `MAX` (identified by the types that
/// implement them) of a column whose values have exact bounds (see
/// [`is_bounded_column`]), as DataFusion's `AggregateExec` produces filters
/// for its aggregates named `min` and `max` of columns.
fn bounded_aggregates(raw: &AggregateExec) -> Option<Vec<BoundedAggregate>> {
    if !raw.group_expr().is_true_no_grouping() || raw.aggr_expr().is_empty() {
        return None;
    }
    let schema = raw.input().schema();
    raw.aggr_expr()
        .iter()
        .enumerate()
        .map(|(index, aggregate)| {
            let implementation = implementation_type(aggregate.fun());
            let max = if implementation == TypeId::of::<min_max::Max>() {
                true
            } else if implementation == TypeId::of::<min_max::Min>() {
                false
            } else {
                return None;
            };
            let arguments = aggregate.expressions();
            let [argument] = arguments.as_slice() else {
                return None;
            };
            is_bounded_column(argument, &schema).then(|| BoundedAggregate {
                aggregate: index,
                max,
                argument: Arc::clone(argument),
            })
        })
        .collect()
}

/// Whether `expr` is a column of `schema` whose values have exact bounds
/// (see [`has_exact_bounds`]): a filter that compares it with a value cannot
/// fail, wherever a scan evaluates it, and prunes like it compares.
fn is_bounded_column(expr: &Arc<dyn PhysicalExpr>, schema: &SchemaRef) -> bool {
    expr.is::<Column>()
        && expr
            .data_type(schema)
            .is_ok_and(|data_type| has_exact_bounds(&data_type))
}

/// Whether the values of `data_type` are ordered by Arrow's comparisons (with
/// which a filter compares them, and the aggregates order them) like the
/// statistics of Parquet files bound them: integers, decimals, dates, times,
/// timestamps, durations, booleans, strings and binaries. Not floating point
/// numbers, whose `NaN`s (greater, or for `-NaN` less, than any other value
/// in Arrow's total order) the statistics leave out, nor dictionaries,
/// intervals and nested values.
pub(crate) fn has_exact_bounds(data_type: &DataType) -> bool {
    data_type.is_integer()
        || (data_type.is_temporal() && !matches!(data_type, DataType::Interval(_)))
        || matches!(
            data_type,
            DataType::Decimal32(_, _)
                | DataType::Decimal64(_, _)
                | DataType::Decimal128(_, _)
                | DataType::Decimal256(_, _)
                | DataType::Boolean
                | DataType::Utf8
                | DataType::LargeUtf8
                | DataType::Utf8View
                | DataType::Binary
                | DataType::LargeBinary
                | DataType::BinaryView
                | DataType::FixedSizeBinary(_)
        )
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use arrow::array::{
        ArrayRef, BooleanArray, Date32Array, Decimal128Array, Float64Array, Int32Array,
        Int64Array, RecordBatch, StringArray,
    };
    use arrow::datatypes::{Field, Schema};
    use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
    use datafusion_common::tree_node::{TreeNode, TreeNodeRecursion};
    use datafusion_datasource::file_scan_config::FileScanConfig;
    use datafusion_datasource::source::DataSourceExec;
    use datafusion_physical_plan::collect;
    use datafusion_physical_plan::metrics::MetricValue;

    use super::super::PhysicalHashAggregate;
    use super::super::test_support::{
        assert_aggregate_parity_with_setup, assert_lowered_results_with_setup,
        lookalike_max,
    };
    use super::*;
    use crate::bridge::{PipelineRootExec, bridged_plans};
    use crate::explain::format_operator_tree;
    use crate::operator::PhysicalOperator;
    use crate::sources::ParquetScanSource;
    use crate::sources::test_harness::write_parquet;
    use crate::test_utils::{pipeline_context, test_session_config};

    /// The number of files of the Parquet table `m`, and of rows of every file.
    const M_FILES: i64 = 16;
    const M_ROWS: i64 = 4096;

    /// Write the Parquet table `m` into `dir`: [`M_FILES`] files of
    /// [`M_ROWS`] rows, in two row groups each (of chunks large enough that
    /// the filters of the queries do not cache them, see `CachingOperator`),
    /// whose values are bounded by the values of every file before them: file
    /// `i` has values `x` from `i` to `1000 - i` (both of them, in its first
    /// two rows, followed by `i + 1` and `999 - i`), and `s`,
    /// `d`, `dec` and `f` of the same order (`f` with a `NaN` in its last
    /// file). Once a thread aggregated the `MIN` or the `MAX` of a file, a
    /// filter of its bounds prunes all files after it. `id` is unique and
    /// ascending, `e` is true for two of three rows, `n` is always `NULL` and
    /// `g` has 7 values.
    fn write_table_m(dir: &Path) -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("x", DataType::Int64, true),
            Field::new("s", DataType::Utf8, true),
            Field::new("d", DataType::Date32, true),
            Field::new("dec", DataType::Decimal128(10, 2), true),
            Field::new("f", DataType::Float64, true),
            Field::new("e", DataType::Boolean, true),
            Field::new("n", DataType::Int64, true),
            Field::new("g", DataType::Int32, true),
        ]));
        let properties = ::parquet::file::properties::WriterProperties::builder()
            .set_max_row_group_row_count(Some(M_ROWS as usize / 2))
            .build();
        for file in 0..M_FILES {
            let x = (0..M_ROWS)
                .map(|row| match row {
                    0 => file,
                    1 => 1000 - file,
                    2 => file + 1,
                    3 => 999 - file,
                    _ => file + 1 + (row * 37) % (999 - 2 * file),
                })
                .collect::<Vec<_>>();
            let columns: Vec<ArrayRef> = vec![
                Arc::new(Int64Array::from_iter_values(
                    (0..M_ROWS).map(|row| file * M_ROWS + row),
                )),
                Arc::new(Int64Array::from(x.clone())),
                Arc::new(StringArray::from_iter_values(
                    x.iter().map(|x| format!("s{x:04}")),
                )),
                Arc::new(Date32Array::from_iter_values(
                    x.iter().map(|&x| 19_000 + x as i32),
                )),
                Arc::new(
                    Decimal128Array::from_iter_values(
                        x.iter().map(|&x| i128::from(x) * 100 + 5),
                    )
                    .with_precision_and_scale(10, 2)?,
                ),
                Arc::new(Float64Array::from_iter_values(x.iter().enumerate().map(
                    |(row, &x)| match (file, row) {
                        (15, 4) => f64::NAN,
                        _ => x as f64,
                    },
                ))),
                Arc::new(BooleanArray::from_iter(
                    (0..M_ROWS).map(|row| Some(row % 3 != 2)),
                )),
                Arc::new(Int64Array::from(vec![None; M_ROWS as usize])),
                Arc::new(Int32Array::from_iter_values(
                    (0..M_ROWS).map(|row| (row % 7) as i32),
                )),
            ];
            let batch = RecordBatch::try_new(Arc::clone(&schema), columns)?;
            write_parquet(
                &dir.join(format!("m/{file:02}.parquet")),
                Arc::clone(&schema),
                &[batch],
                Some(properties.clone()),
            )?;
        }
        Ok(())
    }

    /// Register the Parquet tables of `dir` (see [`write_table_m`]).
    async fn register_tables(ctx: &SessionContext, dir: &Path) -> Result<()> {
        for table in ["m"] {
            let path = dir.join(table);
            let path = path.to_str().expect("a UTF-8 path");
            ctx.register_parquet(table, path, ParquetReadOptions::default())
                .await?;
        }
        Ok(())
    }

    /// A directory with the Parquet tables of the tests.
    fn write_tables() -> Result<tempfile::TempDir> {
        let dir = tempfile::tempdir()?;
        write_table_m(dir.path())?;
        Ok(dir)
    }

    /// The first scan of files of `plan`.
    fn find_file_scan(plan: &Arc<dyn ExecutionPlan>) -> Option<Arc<dyn ExecutionPlan>> {
        if plan
            .downcast_ref::<DataSourceExec>()
            .is_some_and(|scan| scan.data_source().is::<FileScanConfig>())
        {
            return Some(Arc::clone(plan));
        }
        plan.children().into_iter().find_map(find_file_scan)
    }

    /// The number of items pruned by the scan of files `scan` (native or not:
    /// the native scans record the metrics of the scans of DataFusion that
    /// they replace), by its pruning metric `name`.
    fn pruned(scan: &Arc<dyn ExecutionPlan>, name: &str) -> usize {
        match scan.metrics().and_then(|metrics| metrics.sum_by_name(name)) {
            Some(MetricValue::PruningMetrics {
                pruning_metrics, ..
            }) => pruning_metrics.pruned(),
            Some(other) => other.as_usize(),
            None => 0,
        }
    }

    /// The files and row groups that the scan of files `scan` pruned with
    /// statistics.
    fn pruned_by_statistics(scan: &Arc<dyn ExecutionPlan>) -> usize {
        pruned(scan, "files_ranges_pruned_statistics")
            + pruned(scan, "row_groups_pruned_statistics")
    }

    /// The dynamic filters that the `AggregateExec`s of `plan` produce.
    fn aggregate_exec_filters(
        plan: &Arc<dyn ExecutionPlan>,
    ) -> Vec<Arc<DynamicFilterPhysicalExpr>> {
        let mut filters = vec![];
        plan.apply(|node| {
            if node.is::<AggregateExec>() {
                filters.extend(dynamic_filters(node.as_ref()).expect("dynamic filters"));
            }
            Ok(TreeNodeRecursion::Continue)
        })
        .expect("a traversal");
        filters
    }

    /// The dynamic filters that the native aggregations of `root` update.
    fn native_filters(
        root: &Arc<dyn PhysicalOperator>,
    ) -> Vec<Arc<DynamicFilterPhysicalExpr>> {
        let mut filters = vec![];
        if let Some(aggregate) = root.downcast_ref::<PhysicalHashAggregate>()
            && let Some(filter) = &aggregate.spec().dynamic_filter
        {
            filters.extend(filter.filters().iter().cloned());
        }
        for child in root.children() {
            filters.extend(native_filters(child));
        }
        filters
    }

    /// The outcome of an execution of a query by the pipeline engine.
    struct Execution {
        /// The dynamic filters of the `AggregateExec`s of DataFusion's plan.
        datafusion_filters: Vec<Arc<DynamicFilterPhysicalExpr>>,
        /// The dynamic filters that the native aggregations update.
        native_filters: Vec<Arc<DynamicFilterPhysicalExpr>>,
        /// The (first) scan of files of DataFusion's plan.
        scan: Arc<dyn ExecutionPlan>,
        /// The rows that the native scans produced.
        scanned_rows: usize,
        /// The lowered operator tree.
        tree: String,
    }

    /// Execute `root`, lowered from `plan`, in `ctx`.
    async fn execute_lowered(
        ctx: &SessionContext,
        plan: Arc<dyn ExecutionPlan>,
        root: Arc<dyn PhysicalOperator>,
    ) -> Result<Execution> {
        let tree = format_operator_tree(&root);
        assert!(bridged_plans(&root).is_empty(), "{tree}");
        let execution = Execution {
            datafusion_filters: aggregate_exec_filters(&plan),
            native_filters: native_filters(&root),
            scan: find_file_scan(&plan).expect("a scan of files"),
            scanned_rows: 0,
            tree,
        };
        let exec = Arc::new(PipelineRootExec::try_new(root, plan)?);
        collect(Arc::clone(&exec) as _, ctx.task_ctx()).await?;
        let scanned_rows = exec
            .execution_metrics()
            .expect("the metrics of the execution")
            .iter()
            .filter(|(operator, _)| operator.is::<ParquetScanSource>())
            .map(|(_, metrics)| metrics.output_rows())
            .sum();
        Ok(Execution {
            scanned_rows,
            ..execution
        })
    }

    /// Execute `sql` with the `PipelineQueryPlanner`, in a session of
    /// configuration `config` with the tables of `dir`. The plan is fully
    /// native.
    async fn execute(sql: &str, config: SessionConfig, dir: &Path) -> Result<Execution> {
        let ctx = pipeline_context(config);
        register_tables(&ctx, dir).await?;
        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        let Some(root) = plan.downcast_ref::<PipelineRootExec>() else {
            panic!("The pipeline engine did not plan {sql}");
        };
        execute_lowered(&ctx, Arc::clone(root.input()), Arc::clone(root.operator())).await
    }

    /// `config` with `pushdown_filters`.
    fn with_pushdown_filters(
        config: SessionConfig,
        pushdown_filters: bool,
    ) -> SessionConfig {
        let mut config = config;
        config.options_mut().execution.parquet.pushdown_filters = pushdown_filters;
        config
    }

    /// The queries of `MIN` and `MAX` aggregates of `m` for which DataFusion
    /// produces dynamic filters, and the predicates that the filters are set
    /// to.
    const BOUNDED_QUERIES: [(&str, &str); 4] = [
        (
            "SELECT MIN(x), MAX(x) FROM m WHERE e",
            "x@0 < 0 OR x@0 > 1000",
        ),
        ("SELECT MAX(x) FROM m WHERE g > 0", "x@0 > 1000"),
        // (Without the rows of `x` 0 and 1000, the first two of the first
        // file.)
        (
            "SELECT MIN(s), MAX(d), MIN(dec) FROM m WHERE id > 1",
            "s@0 < s0001 OR d@1 > 2024-10-03 OR dec@2 < 1.05",
        ),
        // With a filter of an aggregate: the rows that do not change the
        // bounds are pruned, whether they pass it or not.
        (
            "SELECT MIN(x) FILTER (WHERE g = 3), MAX(x) FROM m WHERE e",
            "x@0 < 1 OR x@0 > 1000",
        ),
    ];

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn aggregations_update_the_dynamic_filters_of_datafusion() -> Result<()> {
        // DataFusion's partial aggregations of `MIN` and `MAX` without `GROUP
        // BY` produce dynamic filters, which the scans of `m` read. The native
        // aggregations set them to the bounds of their aggregates of all
        // threads, as DataFusion does, so the scans prune the files whose
        // values are within the bounds: every file but the first one of every
        // thread (of 4).
        let dir = write_tables()?;
        let setup = async |ctx: &SessionContext| register_tables(ctx, dir.path()).await;
        for pushdown_filters in [false, true] {
            let config = with_pushdown_filters(test_session_config(), pushdown_filters);
            for (sql, predicate) in BOUNDED_QUERIES {
                let context = format!("{sql} (pushdown_filters: {pushdown_filters})");
                let execution = execute(sql, config.clone(), dir.path()).await?;
                let [filter] = execution.datafusion_filters.as_slice() else {
                    panic!("DataFusion produces one filter: {context}");
                };
                // The native aggregation updates the filter of DataFusion.
                let [native] = execution.native_filters.as_slice() else {
                    panic!("No filter is updated: {context}\n{}", execution.tree);
                };
                assert!(Arc::ptr_eq(native, filter), "{context}");
                assert!(
                    execution.tree.contains("dynamic_filter=true"),
                    "{context}\n{}",
                    execution.tree
                );
                assert!(filter.is_complete(), "{context}");
                assert_eq!(filter.current()?.to_string(), predicate, "{context}");
                // At most one file per thread is read.
                let pruned = pruned_by_statistics(&execution.scan);
                assert!(
                    pruned >= (M_FILES - 4) as usize,
                    "{context}: {pruned} pruned"
                );
                assert!(
                    execution.scanned_rows <= 4 * M_ROWS as usize,
                    "{context}: {} rows scanned",
                    execution.scanned_rows
                );
                assert_aggregate_parity_with_setup(sql, config.clone(), &setup).await;
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn filters_of_other_aggregations_stay_true() -> Result<()> {
        // The native aggregations do not set the filters of `MIN` and `MAX`
        // of floating point values, whose `NaN`s the statistics of Parquet
        // files leave out (a `NaN` of the last file of `m` is the `MAX` of
        // `f`, which the bounds of the files before would prune), nor of
        // functions that are only named `min` or `max`: DataFusion's filters
        // stay `true`, and the scans read every row. The results are those of
        // DataFusion without its filters. A filter of an aggregate without
        // values (`n` is always `NULL`) is set to `true`.
        let dir = write_tables()?;
        let setup = async |ctx: &SessionContext| register_tables(ctx, dir.path()).await;
        let config = test_session_config();
        let mut reference = config.clone();
        reference
            .options_mut()
            .optimizer
            .enable_aggregate_dynamic_filter_pushdown = false;
        let all_rows = (M_FILES * M_ROWS) as usize;
        for sql in [
            "SELECT MIN(f), MAX(f) FROM m WHERE e",
            "SELECT MAX(f), MAX(x) FROM m WHERE g > 0",
        ] {
            let execution = execute(sql, config.clone(), dir.path()).await?;
            assert_eq!(execution.datafusion_filters.len(), 1, "{sql}");
            assert!(execution.native_filters.is_empty(), "{sql}");
            assert_eq!(
                execution.datafusion_filters[0].current()?.to_string(),
                "true",
                "{sql}"
            );
            assert_eq!(execution.scanned_rows, all_rows, "{sql}");
            assert_lowered_results_with_setup(
                sql,
                config.clone(),
                reference.clone(),
                &setup,
            )
            .await;
        }
        // A `MAX` that is not DataFusion's.
        let lookalike = async |ctx: &SessionContext| {
            register_tables(ctx, dir.path()).await?;
            ctx.register_udaf(lookalike_max());
            Ok(())
        };
        let sql = "SELECT MAX(x) FROM m WHERE e";
        let ctx = pipeline_context(config.clone());
        lookalike(&ctx).await?;
        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        let root = plan
            .downcast_ref::<PipelineRootExec>()
            .expect("a pipeline plan");
        let execution =
            execute_lowered(&ctx, Arc::clone(root.input()), Arc::clone(root.operator()))
                .await?;
        assert_eq!(execution.datafusion_filters.len(), 1, "{sql}");
        assert!(execution.native_filters.is_empty(), "{sql}");
        assert_eq!(execution.scanned_rows, all_rows, "{sql}");
        assert_lowered_results_with_setup(sql, config.clone(), reference, &lookalike)
            .await;
        // An aggregate without values.
        let sql = "SELECT MIN(x), MAX(n) FROM m WHERE e";
        let execution = execute(sql, config.clone(), dir.path()).await?;
        let [filter] = execution.native_filters.as_slice() else {
            panic!("No filter is updated: {sql}\n{}", execution.tree);
        };
        assert_eq!(filter.current()?.to_string(), "true", "{sql}");
        assert_eq!(execution.scanned_rows, all_rows, "{sql}");
        assert_aggregate_parity_with_setup(sql, config, &setup).await;
        Ok(())
    }
}
