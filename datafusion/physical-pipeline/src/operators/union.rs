// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! `PhysicalUnion`: parallel `UNION ALL` over several child pipelines.
//!
//! [`PhysicalUnion`] replaces DataFusion's `UnionExec` and `InterleaveExec`.
//! Like DuckDB's `PhysicalUnion`, it does not move any data: it only builds
//! pipelines (see [`MetaPipelineBuilder::build_union`]). The first input
//! continues the pipeline of the operators above the union, and every other
//! input gets a union pipeline that shares these operators and the sink.
//! Unless the order of the inputs must be kept
//! ([`PhysicalUnion::with_preserve_order`]), the pipelines of all inputs run
//! concurrently, each on all worker threads, so the chunks of all inputs are
//! pushed through the operators above the union in parallel, without any
//! channel between the inputs and their consumer.
//!
//! The schema of the union is the schema of `UnionExec`: the field names of
//! the first input, a field is nullable if it is nullable in any input, and
//! the field and schema metadata of all inputs are merged. The chunks of an
//! input whose fields differ from the fields of the union (in their names,
//! nullability or metadata: the data types must be equal) are relabeled with
//! the fields of the union by a `UnionCoercion` operator, without copying any
//! data, like the `ProjectionExec` that `UnionExec` puts above such an input.
//! Like that projection, it keeps the schema metadata of the input: the
//! batches of DataFusion's union carry the schema metadata of the input they
//! come from, not the merged schema metadata of the union.

use std::sync::Arc;

use arrow::datatypes::{Schema, SchemaRef};
use datafusion_common::metadata::merge_metadata_for_union;
use datafusion_common::{Result, exec_err, internal_err, plan_err};
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::union::{InterleaveExec, UnionExec, can_interleave};

use crate::chunk::DataChunk;
use crate::operator::{
    ExecutionContext, GlobalOperatorState, OperatorResult, OperatorState,
    PhysicalOperator,
};
use crate::pipeline::{MetaPipelineBuilder, PipelineBuilder};

/// `UNION ALL` of any number of inputs.
///
/// See the [module documentation](self) for its pipelines and schema.
#[derive(Debug)]
pub struct PhysicalUnion {
    /// The inputs. An input whose fields are not the fields of the union is
    /// wrapped in a `UnionCoercion`.
    children: Vec<Arc<dyn PhysicalOperator>>,
    schema: SchemaRef,
    preserve_order: bool,
}

impl PhysicalUnion {
    /// The union of `inputs`, with the schema that DataFusion's `UnionExec`
    /// has for inputs with the schemas of `inputs`.
    pub fn try_new(inputs: Vec<Arc<dyn PhysicalOperator>>) -> Result<Self> {
        let schema = union_schema(&inputs)?;
        Self::try_new_with_schema(inputs, schema)
    }

    /// The union of `inputs`, with schema `schema` (e.g. the schema of the
    /// `UnionExec` being lowered). Every input must have the data types of
    /// `schema`; the field names, nullability and metadata may differ.
    ///
    /// The chunks of an input keep its schema metadata, see the
    /// [module documentation](self).
    pub fn try_new_with_schema(
        inputs: Vec<Arc<dyn PhysicalOperator>>,
        schema: SchemaRef,
    ) -> Result<Self> {
        if inputs.is_empty() {
            return internal_err!("PhysicalUnion requires at least one input");
        }
        let children = inputs
            .into_iter()
            .enumerate()
            .map(|(index, input)| coerce_input(index, input, &schema))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            children,
            schema,
            preserve_order: false,
        })
    }

    /// If `preserve_order` is `true`, the pipelines of every input only start
    /// once all pipelines of the inputs before it completed, so that the
    /// output comes in input order (if the operators above the union keep
    /// it): all rows of the first input, then all rows of the second one,
    /// and so on. Otherwise all inputs are executed concurrently.
    pub fn with_preserve_order(mut self, preserve_order: bool) -> Self {
        self.preserve_order = preserve_order;
        self
    }

    /// Lower a DataFusion `UnionExec` or `InterleaveExec` to a
    /// [`PhysicalUnion`] with the same schema. `lower_input` lowers the
    /// inputs of the node. Returns `Ok(None)` for any other node.
    ///
    /// `InterleaveExec` combines hash partitioned inputs partition by
    /// partition; the pipeline engine has no partitions, so it is a union of
    /// its inputs too. An `InterleaveExec` whose inputs are not partitioned
    /// alike (`can_interleave`) fails in DataFusion and is not lowered, so
    /// that DataFusion executes it.
    pub fn try_lower<F>(
        plan: &Arc<dyn ExecutionPlan>,
        lower_input: F,
    ) -> Result<Option<Self>>
    where
        F: FnMut(&Arc<dyn ExecutionPlan>) -> Result<Arc<dyn PhysicalOperator>>,
    {
        let inputs = if let Some(union) = plan.downcast_ref::<UnionExec>() {
            union.inputs()
        } else if let Some(interleave) = plan.downcast_ref::<InterleaveExec>()
            && can_interleave(interleave.inputs().iter())
        {
            interleave.inputs()
        } else {
            return Ok(None);
        };
        let inputs = inputs.iter().map(lower_input).collect::<Result<Vec<_>>>()?;
        Self::try_new_with_schema(inputs, plan.schema()).map(Some)
    }

    /// Whether the inputs are executed one after the other, in order, see
    /// [`Self::with_preserve_order`].
    pub fn preserve_order(&self) -> bool {
        self.preserve_order
    }
}

impl PhysicalOperator for PhysicalUnion {
    fn name(&self) -> &str {
        "PhysicalUnion"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.children
    }

    fn description(&self) -> String {
        if self.preserve_order {
            "PhysicalUnion: preserve_order=true".to_string()
        } else {
            "PhysicalUnion".to_string()
        }
    }

    fn build_pipelines(
        &self,
        current: &mut PipelineBuilder,
        meta: &mut MetaPipelineBuilder<'_>,
    ) -> Result<()> {
        meta.build_union(current, &self.children, self.preserve_order)
    }
}

/// The schema of DataFusion's `UnionExec` for inputs with the schemas of
/// `inputs` (`union_schema` in `datafusion_physical_plan::union`): the names
/// of the fields of the first input; the first nullable field among the
/// inputs (or the field of the first input if none is nullable), with the
/// merged metadata of the fields of all inputs; the merged metadata of all
/// schemas.
fn union_schema(inputs: &[Arc<dyn PhysicalOperator>]) -> Result<SchemaRef> {
    let Some(first) = inputs.first() else {
        return internal_err!("PhysicalUnion requires at least one input");
    };
    let first_schema = first.schema();
    let schemas = inputs
        .iter()
        .map(|input| input.schema())
        .collect::<Vec<_>>();
    if schemas
        .iter()
        .all(|schema| Arc::ptr_eq(schema, &first_schema) || schema == &first_schema)
    {
        return Ok(first_schema);
    }
    let num_fields = first_schema.fields().len();
    for (index, schema) in schemas.iter().enumerate() {
        check_field_count(index, schema, num_fields)?;
    }
    let fields = (0..num_fields)
        .map(|i| {
            let field = schemas
                .iter()
                .find(|schema| schema.field(i).is_nullable())
                .unwrap_or(&first_schema)
                .field(i);
            let metadata =
                merge_metadata_for_union(schemas.iter().map(|s| s.field(i).metadata()));
            field
                .clone()
                .with_metadata(metadata)
                .with_name(first_schema.field(i).name())
        })
        .collect::<Vec<_>>();
    let metadata = merge_metadata_for_union(schemas.iter().map(|s| s.metadata()));
    Ok(Arc::new(Schema::new_with_metadata(fields, metadata)))
}

fn check_field_count(index: usize, schema: &Schema, expected: usize) -> Result<()> {
    let num_fields = schema.fields().len();
    if num_fields != expected {
        return exec_err!(
            "PhysicalUnion requires all inputs to have the same number of fields as the \
             union ({expected}), but input {index} has {num_fields} fields"
        );
    }
    Ok(())
}

/// `input`, the input at `index` of a union with schema `schema`, wrapped in
/// a [`UnionCoercion`] if its fields are not the fields of `schema`.
///
/// Like `UnionExec`, an input whose schema differs only in its metadata is
/// not changed: DataFusion's union passes the batches of its inputs through,
/// and the projection that coerces the fields of an input (`coerce_schema` in
/// `datafusion_physical_plan::union`) keeps the schema metadata of the input
/// too.
fn coerce_input(
    index: usize,
    input: Arc<dyn PhysicalOperator>,
    schema: &SchemaRef,
) -> Result<Arc<dyn PhysicalOperator>> {
    let input_schema = input.schema();
    if input_schema.fields() == schema.fields() {
        return Ok(input);
    }
    check_field_count(index, &input_schema, schema.fields().len())?;
    for (i, (input_field, field)) in input_schema
        .fields()
        .iter()
        .zip(schema.fields())
        .enumerate()
    {
        if input_field.data_type() != field.data_type() {
            return plan_err!(
                "PhysicalUnion requires all inputs to have the data types of the union; \
                 column {i} of input {index} has type {}, but the union expects {}",
                input_field.data_type(),
                field.data_type()
            );
        }
    }
    Ok(Arc::new(UnionCoercion {
        children: vec![input],
        schema: Arc::new(Schema::new_with_metadata(
            schema.fields().clone(),
            input_schema.metadata().clone(),
        )),
    }))
}

/// Relabels the chunks of a union input with the fields of the union: same
/// columns (and selection), but the field names, nullability and metadata of
/// the union. No data is copied.
#[derive(Debug)]
struct UnionCoercion {
    children: Vec<Arc<dyn PhysicalOperator>>,
    schema: SchemaRef,
}

impl PhysicalOperator for UnionCoercion {
    fn name(&self) -> &str {
        "UnionCoercion"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.children
    }

    fn execute(
        &self,
        _ctx: &ExecutionContext,
        input: &DataChunk,
        output: &mut DataChunk,
        _gstate: &dyn GlobalOperatorState,
        _lstate: &mut dyn OperatorState,
    ) -> Result<OperatorResult> {
        // The fields of a coerced input differ, so it has columns, and
        // `input` is not empty.
        *output = DataChunk::try_new(
            Arc::clone(&self.schema),
            input.columns().to_vec(),
            input.selection().cloned(),
        )?;
        Ok(OperatorResult::NeedMoreInput)
    }
}

/// Test support shared by the tests of [`PhysicalUnion`],
/// [`PhysicalLimit`](super::limit::PhysicalLimit) and
/// [`PhysicalPlaceholderRow`](super::placeholder_row::PhysicalPlaceholderRow):
/// test sources, and a differential check of DataFusion plans lowered with
/// these operators.
///
/// Until the planner lowers plans to these operators, [`lower`] does, so
/// that their output is compared with DataFusion's on real query plans.
#[cfg(test)]
pub(crate) mod test_support {
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use arrow::array::{AsArray, Int64Array, RecordBatch, RecordBatchOptions};
    use arrow::datatypes::{DataType, Field, Int64Type, Schema, SchemaRef};
    use datafusion::prelude::{SessionConfig, SessionContext};
    use datafusion_common::Result;
    use datafusion_execution::{SendableRecordBatchStream, TaskContext};
    use datafusion_physical_expr::PhysicalExpr;
    use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
    use datafusion_physical_plan::projection::ProjectionExec;
    use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
    use datafusion_physical_plan::streaming::{PartitionStream, StreamingTableExec};
    use datafusion_physical_plan::{ExecutionPlan, collect};

    use super::PhysicalUnion;
    use crate::bridge::{BridgeOperator, PipelineRootExec};
    use crate::chunk::DataChunk;
    use crate::executor::Executor;
    use crate::explain::format_operator_tree;
    use crate::operator::{
        ExecutionContext, GlobalOperatorState, GlobalSourceState, LocalSourceState,
        OperatorResult, OperatorState, PhysicalOperator, SourceResult, TaskInterrupt,
        downcast_state, downcast_state_mut,
    };
    use crate::operators::limit::PhysicalLimit;
    use crate::operators::placeholder_row::PhysicalPlaceholderRow;
    use crate::test_utils::{
        EngineOutput, assert_byte_eq_with_datafusion_with_config, compare_outputs,
        datafusion_context, register_test_tables, run_sql, test_session_config,
    };

    // ---------------------------------------------------------------------
    // Lowering
    // ---------------------------------------------------------------------

    /// Lower `plan` with the native operators of this module, standing in
    /// for the planner:
    ///
    /// * limits, unions and placeholder rows with `try_lower`;
    /// * `CoalescePartitionsExec`s without a fetch are removed (the pipeline
    ///   engine merges the partitions of their input anyway);
    /// * `ProjectionExec`s with a [`TestProjection`], a stand-in for the
    ///   native projection, so that the operators below them are lowered
    ///   too;
    /// * scans with the native sources of
    ///   [`try_create_scan_source`](crate::sources::try_create_scan_source);
    /// * every other node is bridged, with its subtree.
    pub(crate) fn lower(
        plan: &Arc<dyn ExecutionPlan>,
    ) -> Result<Arc<dyn PhysicalOperator>> {
        if let Some(limit) = PhysicalLimit::try_lower(plan, lower)? {
            return Ok(Arc::new(limit));
        }
        if let Some(union) = PhysicalUnion::try_lower(plan, lower)? {
            return Ok(Arc::new(union));
        }
        if let Some(placeholder) = PhysicalPlaceholderRow::try_lower(plan)? {
            return Ok(Arc::new(placeholder));
        }
        if let Some(coalesce) = plan.downcast_ref::<CoalescePartitionsExec>() {
            return lower(coalesce.input());
        }
        if let Some(projection) = plan.downcast_ref::<ProjectionExec>() {
            return Ok(Arc::new(TestProjection {
                children: vec![lower(projection.input())?],
                exprs: projection
                    .expr()
                    .iter()
                    .map(|expr| Arc::clone(&expr.expr))
                    .collect(),
                schema: projection.schema(),
            }));
        }
        if let Some(source) = crate::sources::try_create_scan_source(plan)? {
            return Ok(source);
        }
        Ok(Arc::new(BridgeOperator::new(Arc::clone(plan))))
    }

    /// Evaluates the expressions of a `ProjectionExec` on the compacted input
    /// chunks: a minimal stand-in for the native projection.
    #[derive(Debug)]
    pub(crate) struct TestProjection {
        children: Vec<Arc<dyn PhysicalOperator>>,
        exprs: Vec<Arc<dyn PhysicalExpr>>,
        schema: SchemaRef,
    }

    impl PhysicalOperator for TestProjection {
        fn name(&self) -> &str {
            "TestProjection"
        }

        fn schema(&self) -> SchemaRef {
            Arc::clone(&self.schema)
        }

        fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
            &self.children
        }

        fn execute(
            &self,
            _ctx: &ExecutionContext,
            input: &DataChunk,
            output: &mut DataChunk,
            _gstate: &dyn GlobalOperatorState,
            _lstate: &mut dyn OperatorState,
        ) -> Result<OperatorResult> {
            let batch = input.compact()?;
            let columns = self
                .exprs
                .iter()
                .map(|expr| expr.evaluate(&batch)?.into_array(batch.num_rows()))
                .collect::<Result<Vec<_>>>()?;
            let options =
                RecordBatchOptions::new().with_row_count(Some(batch.num_rows()));
            *output = DataChunk::from_record_batch(RecordBatch::try_new_with_options(
                Arc::clone(&self.schema),
                columns,
                &options,
            )?);
            Ok(OperatorResult::NeedMoreInput)
        }
    }

    /// The number of operators of type `T` in the tree below (and including)
    /// `operator`.
    pub(crate) fn count<T: PhysicalOperator>(
        operator: &Arc<dyn PhysicalOperator>,
    ) -> usize {
        usize::from(operator.is::<T>())
            + operator.children().iter().map(count::<T>).sum::<usize>()
    }

    // ---------------------------------------------------------------------
    // Differential checks
    // ---------------------------------------------------------------------

    /// Assert that `sql` produces byte-identical results on DataFusion and on
    /// the pipeline engine, both through the `PipelineQueryPlanner`
    /// ([`assert_byte_eq_with_datafusion_with_config`]) and with the plan
    /// lowered by [`lower`] ([`assert_native_byte_eq`]). Returns the lowered
    /// operator tree.
    pub(crate) async fn assert_parity(sql: &str) -> Arc<dyn PhysicalOperator> {
        assert_parity_with_config(sql, test_session_config()).await
    }

    /// [`assert_parity`] with a custom session configuration.
    pub(crate) async fn assert_parity_with_config(
        sql: &str,
        config: SessionConfig,
    ) -> Arc<dyn PhysicalOperator> {
        let setup = async |ctx: &SessionContext| register_test_tables(ctx);
        assert_byte_eq_with_datafusion_with_config(sql, config.clone(), setup).await;
        assert_native_byte_eq(sql, config, setup).await
    }

    /// Assert that `sql` produces byte-identical results on DataFusion and on
    /// the pipeline engine with the plan lowered by [`lower`]. `setup`
    /// registers the tables of the query. Returns the lowered operator tree.
    pub(crate) async fn assert_native_byte_eq<F>(
        sql: &str,
        config: SessionConfig,
        setup: F,
    ) -> Arc<dyn PhysicalOperator>
    where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        let ctx = datafusion_context(config);
        setup(&ctx).await.expect("setup");
        let expected = run_sql(&ctx, sql).await.expect("DataFusion");
        // A new plan: executing a plan changes the state of some nodes.
        let plan = ctx
            .sql(sql)
            .await
            .expect("planning")
            .create_physical_plan()
            .await
            .expect("physical planning");
        let ordered = expected.ordered;
        let (root, actual) = run_lowered(plan, ctx.task_ctx(), ordered)
            .await
            .unwrap_or_else(|e| {
                panic!("Failed to run the lowered plan: {e}\nquery: {sql}")
            });
        if let Err(message) = compare_outputs(&expected, &actual) {
            panic!(
                "Results differ: {message}\nquery: {sql}\nlowered: {}",
                format_operator_tree(&root)
            );
        }
        root
    }

    /// Assert that the plan created by `make_plan` produces byte-identical
    /// results when executed by DataFusion and when lowered by [`lower`] and
    /// executed by the pipeline engine, in the same order if `ordered`.
    /// `make_plan` is called once per engine. Returns the lowered operator
    /// tree.
    pub(crate) async fn assert_native_plan_byte_eq(
        make_plan: impl Fn() -> Result<Arc<dyn ExecutionPlan>>,
        ordered: bool,
    ) -> Arc<dyn PhysicalOperator> {
        let plan = make_plan().expect("plan");
        let root = lower(&plan).expect("lowering");
        assert_operator_byte_eq(
            make_plan().expect("plan"),
            Arc::clone(&root),
            plan,
            ordered,
        )
        .await;
        root
    }

    /// Assert that executing `root` with the pipeline engine produces
    /// byte-identical results to executing `expected` with DataFusion, in the
    /// same order if `ordered`. `plan` is the plan `root` was lowered from: a
    /// plan equal to `expected`, but not executed yet.
    pub(crate) async fn assert_operator_byte_eq(
        expected: Arc<dyn ExecutionPlan>,
        root: Arc<dyn PhysicalOperator>,
        plan: Arc<dyn ExecutionPlan>,
        ordered: bool,
    ) {
        let task_ctx =
            Arc::new(TaskContext::default().with_session_config(test_session_config()));
        let description = datafusion_physical_plan::displayable(expected.as_ref())
            .indent(true)
            .to_string();
        let batches = with_timeout(collect(Arc::clone(&expected), Arc::clone(&task_ctx)))
            .await
            .expect("DataFusion");
        let expected = EngineOutput {
            schema: expected.schema(),
            plan: expected,
            batches,
            ordered,
        };
        let actual = run_native(Arc::clone(&root), plan, task_ctx, ordered)
            .await
            .unwrap_or_else(|e| {
                panic!("Failed to run the operators: {e}\n{description}")
            });
        if let Err(message) = compare_outputs(&expected, &actual) {
            panic!(
                "Results differ: {message}\nplan:\n{description}\noperators: {}",
                format_operator_tree(&root)
            );
        }
    }

    async fn run_lowered(
        plan: Arc<dyn ExecutionPlan>,
        task_ctx: Arc<TaskContext>,
        ordered: bool,
    ) -> Result<(Arc<dyn PhysicalOperator>, EngineOutput)> {
        let root = lower(&plan)?;
        let output = run_native(Arc::clone(&root), plan, task_ctx, ordered).await?;
        Ok((root, output))
    }

    /// Execute `root`, lowered from `plan`, with the pipeline engine.
    async fn run_native(
        root: Arc<dyn PhysicalOperator>,
        plan: Arc<dyn ExecutionPlan>,
        task_ctx: Arc<TaskContext>,
        ordered: bool,
    ) -> Result<EngineOutput> {
        let exec: Arc<dyn ExecutionPlan> =
            Arc::new(PipelineRootExec::try_new(root, plan)?);
        let batches = with_timeout(collect(Arc::clone(&exec), task_ctx)).await?;
        Ok(EngineOutput {
            schema: exec.schema(),
            plan: exec,
            batches,
            ordered,
        })
    }

    // ---------------------------------------------------------------------
    // Test sources
    // ---------------------------------------------------------------------

    /// The schema of [`values_chunk`]: a single `v BIGINT NOT NULL` column.
    pub(crate) fn values_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("v", DataType::Int64, false)]))
    }

    /// A chunk with the values `values` in column `v`.
    pub(crate) fn values_chunk(values: impl IntoIterator<Item = i64>) -> DataChunk {
        let array = Int64Array::from_iter_values(values);
        DataChunk::from_record_batch(
            RecordBatch::try_new(values_schema(), vec![Arc::new(array)]).unwrap(),
        )
    }

    /// Chunks with the consecutive values `start..`, of `sizes` rows each.
    pub(crate) fn values_chunks(start: i64, sizes: &[usize]) -> Vec<DataChunk> {
        let mut next = start;
        sizes
            .iter()
            .map(|&size| {
                let chunk = values_chunk(next..next + size as i64);
                next += size as i64;
                chunk
            })
            .collect()
    }

    /// The values of column `v` of `batches`, in order.
    pub(crate) fn values(batches: &[RecordBatch]) -> Vec<i64> {
        batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column(0)
                    .as_primitive::<Int64Type>()
                    .values()
                    .to_vec()
            })
            .collect()
    }

    /// The values of column `v` of `batches`, sorted.
    pub(crate) fn sorted_values(batches: &[RecordBatch]) -> Vec<i64> {
        let mut values = values(batches);
        values.sort_unstable();
        values
    }

    /// A parallel source that hands out `chunks` in order, through a counter
    /// shared by its threads. Threads `0..idle_threads` never produce
    /// anything and never complete, like the threads reading an idle
    /// partition of a streaming source.
    #[derive(Debug)]
    pub(crate) struct TestSource {
        schema: SchemaRef,
        chunks: Vec<DataChunk>,
        idle_threads: usize,
        /// Every thread waits this long before it reads its first chunk.
        delay: Option<std::time::Duration>,
        /// The number of chunks handed out.
        produced: AtomicUsize,
    }

    impl TestSource {
        pub(crate) fn new(chunks: Vec<DataChunk>) -> Self {
            let schema = chunks
                .first()
                .map_or_else(values_schema, |chunk| Arc::clone(chunk.schema()));
            Self::with_schema(schema, chunks)
        }

        pub(crate) fn with_schema(schema: SchemaRef, chunks: Vec<DataChunk>) -> Self {
            Self {
                schema,
                chunks,
                idle_threads: 0,
                delay: None,
                produced: AtomicUsize::new(0),
            }
        }

        /// Make every thread wait for `delay` before it reads its first
        /// chunk.
        pub(crate) fn with_delay(mut self, delay: std::time::Duration) -> Self {
            self.delay = Some(delay);
            self
        }

        /// Make threads `0..idle_threads` idle.
        pub(crate) fn with_idle_threads(mut self, idle_threads: usize) -> Self {
            self.idle_threads = idle_threads;
            self
        }

        /// The number of chunks handed out so far.
        pub(crate) fn produced(&self) -> usize {
            self.produced.load(Ordering::SeqCst)
        }
    }

    #[derive(Debug, Default)]
    struct TestSourceState {
        next: AtomicUsize,
    }

    impl GlobalSourceState for TestSourceState {}

    #[derive(Debug, Default)]
    struct TestSourceLocal {
        waited: bool,
    }

    impl LocalSourceState for TestSourceLocal {}

    impl PhysicalOperator for TestSource {
        fn name(&self) -> &str {
            "TestSource"
        }

        fn schema(&self) -> SchemaRef {
            Arc::clone(&self.schema)
        }

        fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
            &[]
        }

        fn is_source(&self) -> bool {
            true
        }

        fn parallel_source(&self) -> bool {
            true
        }

        fn get_global_source_state(
            &self,
            _ctx: &ExecutionContext,
        ) -> Result<Arc<dyn GlobalSourceState>> {
            Ok(Arc::new(TestSourceState::default()))
        }

        fn get_local_source_state(
            &self,
            _ctx: &ExecutionContext,
            _gstate: &dyn GlobalSourceState,
        ) -> Result<Box<dyn LocalSourceState>> {
            Ok(Box::new(TestSourceLocal::default()))
        }

        fn get_data(
            &self,
            ctx: &ExecutionContext,
            gstate: &dyn GlobalSourceState,
            lstate: &mut dyn LocalSourceState,
        ) -> Result<SourceResult> {
            if ctx.thread_id < self.idle_threads {
                // Nothing wakes this thread up, except the executor.
                return Ok(SourceResult::Blocked(TaskInterrupt::waker_registered()));
            }
            let lstate =
                downcast_state_mut::<TestSourceLocal>(lstate.as_any_mut(), "TestSource")?;
            if let Some(delay) = self.delay
                && !std::mem::replace(&mut lstate.waited, true)
            {
                return Ok(SourceResult::Blocked(TaskInterrupt::wait_for(
                    tokio::time::sleep(delay),
                )));
            }
            let state = downcast_state::<TestSourceState>(gstate.as_any(), self.name())?;
            let index = state.next.fetch_add(1, Ordering::SeqCst);
            Ok(match self.chunks.get(index) {
                Some(chunk) => {
                    self.produced.fetch_add(1, Ordering::SeqCst);
                    SourceResult::Chunk(chunk.clone())
                }
                None => SourceResult::Finished,
            })
        }
    }

    /// Execute `root` on `threads` threads, with a timeout: some tests check
    /// that executions complete.
    pub(crate) async fn run(
        root: Arc<dyn PhysicalOperator>,
        threads: usize,
    ) -> Result<Vec<RecordBatch>> {
        with_timeout(Executor::collect(
            root,
            Arc::new(TaskContext::default()),
            threads,
        ))
        .await
    }

    /// Await `execution`, failing the test if it does not complete in time.
    async fn with_timeout<T>(execution: impl Future<Output = T>) -> T {
        tokio::time::timeout(std::time::Duration::from_secs(30), execution)
            .await
            .expect("the execution did not complete")
    }

    /// A `StreamingTableExec` partition whose stream never produces anything,
    /// like an idle partition of a streaming source.
    #[derive(Debug)]
    struct PendingPartition {
        schema: SchemaRef,
    }

    impl PartitionStream for PendingPartition {
        fn schema(&self) -> &SchemaRef {
            &self.schema
        }

        fn execute(&self, _ctx: Arc<TaskContext>) -> SendableRecordBatchStream {
            Box::pin(RecordBatchStreamAdapter::new(
                Arc::clone(&self.schema),
                futures::stream::pending(),
            ))
        }
    }

    /// A `StreamingTableExec` partition that produces a single batch.
    #[derive(Debug)]
    struct BatchPartition {
        batch: RecordBatch,
    }

    impl PartitionStream for BatchPartition {
        fn schema(&self) -> &SchemaRef {
            self.batch.schema_ref()
        }

        fn execute(&self, _ctx: Arc<TaskContext>) -> SendableRecordBatchStream {
            Box::pin(RecordBatchStreamAdapter::new(
                self.batch.schema(),
                futures::stream::iter([Ok(self.batch.clone())]),
            ))
        }
    }

    /// A plan with a single partition with schema `schema` that never
    /// produces anything.
    pub(crate) fn pending_plan(schema: SchemaRef) -> Result<Arc<dyn ExecutionPlan>> {
        let partition = Arc::new(PendingPartition {
            schema: Arc::clone(&schema),
        });
        streaming_plan(schema, partition)
    }

    /// A plan with a single partition that produces `batch`.
    pub(crate) fn batch_plan(batch: RecordBatch) -> Result<Arc<dyn ExecutionPlan>> {
        let schema = batch.schema();
        streaming_plan(schema, Arc::new(BatchPartition { batch }))
    }

    fn streaming_plan(
        schema: SchemaRef,
        partition: Arc<dyn PartitionStream>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(StreamingTableExec::try_new(
            schema,
            vec![partition],
            None,
            vec![],
            false,
            None,
        )?))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::collections::HashMap;
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::time::Duration;

    use arrow::array::{Int32Array, RecordBatch, StringArray};
    use arrow::datatypes::{DataType, Field};
    use datafusion_datasource::memory::MemorySourceConfig;
    use datafusion_physical_expr::expressions::col;
    use datafusion_physical_plan::empty::EmptyExec;
    use datafusion_physical_plan::repartition::RepartitionExec;
    use datafusion_physical_plan::{
        ChildrenPropertiesMode, ExecutionPlanProperties, Partitioning,
        ReplaceChildrenOptions,
    };

    use super::test_support::{
        TestSource, assert_native_plan_byte_eq, assert_operator_byte_eq, assert_parity,
        assert_parity_with_config, count, run, sorted_values, values, values_chunk,
        values_chunks, values_schema,
    };
    use crate::bridge::BridgeOperator;
    use crate::chunk::SelectionVector;
    use crate::explain::format_pipelines;
    use crate::operator::{
        GlobalSourceState, LocalSourceState, SourceResult, TaskInterrupt, downcast_state,
    };
    use crate::operators::placeholder_row::PhysicalPlaceholderRow;
    use crate::test_utils::{test_session_config, test_table_t2};

    fn metadata(pairs: &[(&str, &str)]) -> HashMap<String, String> {
        pairs
            .iter()
            .map(|(key, value)| (key.to_string(), value.to_string()))
            .collect()
    }

    /// Schemas with the same data types (`Int32`, `Utf8`) and different field
    /// names, nullability and field metadata. All but the last one have the
    /// same schema metadata.
    fn schemas() -> Vec<SchemaRef> {
        let schema_metadata = metadata(&[("s", "1")]);
        vec![
            Arc::new(
                Schema::new(vec![
                    Field::new("a", DataType::Int32, false)
                        .with_metadata(metadata(&[("k1", "v1")])),
                    Field::new("b", DataType::Utf8, true),
                ])
                .with_metadata(schema_metadata.clone()),
            ),
            Arc::new(
                Schema::new(vec![
                    Field::new("x", DataType::Int32, true)
                        .with_metadata(metadata(&[("k2", "v2")])),
                    Field::new("y", DataType::Utf8, false)
                        .with_metadata(metadata(&[("k1", "other")])),
                ])
                .with_metadata(schema_metadata.clone()),
            ),
            Arc::new(
                Schema::new(vec![
                    Field::new("a", DataType::Int32, false),
                    Field::new("b", DataType::Utf8, false),
                ])
                .with_metadata(schema_metadata),
            ),
            Arc::new(
                Schema::new(vec![
                    Field::new("a", DataType::Int32, true),
                    Field::new("b", DataType::Utf8, true),
                ])
                .with_metadata(metadata(&[("t", "2"), ("s", "3")])),
            ),
        ]
    }

    /// A batch with schema `schema` (see [`schemas`]) and the values
    /// `values`, as integers and strings.
    fn batch(schema: &SchemaRef, values: std::ops::Range<i32>) -> RecordBatch {
        RecordBatch::try_new(
            Arc::clone(schema),
            vec![
                Arc::new(Int32Array::from_iter_values(values.clone())),
                Arc::new(StringArray::from_iter_values(
                    values.map(|v| format!("value {v}")),
                )),
            ],
        )
        .unwrap()
    }

    fn empty_source(schema: &SchemaRef) -> Arc<dyn PhysicalOperator> {
        Arc::new(TestSource::with_schema(Arc::clone(schema), vec![]))
    }

    #[test]
    fn union_schema_is_the_schema_of_union_exec() -> Result<()> {
        let schemas = schemas();
        for indices in [
            vec![0, 1],
            vec![1, 0],
            vec![0, 2],
            vec![2, 0],
            vec![2, 2],
            vec![0, 1, 2],
            vec![2, 1, 0, 1],
            vec![0, 3],
            vec![3, 0, 1],
            vec![3, 3],
        ] {
            let expected = UnionExec::try_new(
                indices
                    .iter()
                    .map(|&i| {
                        Arc::new(EmptyExec::new(Arc::clone(&schemas[i])))
                            as Arc<dyn ExecutionPlan>
                    })
                    .collect(),
            )?
            .schema();
            let union = PhysicalUnion::try_new(
                indices.iter().map(|&i| empty_source(&schemas[i])).collect(),
            )?;
            assert_eq!(union.schema(), expected, "inputs {indices:?}");
            // The inputs whose fields differ are relabeled; like DataFusion,
            // all inputs keep their schema metadata.
            for (&i, child) in indices.iter().zip(union.children()) {
                assert_eq!(child.schema().fields(), expected.fields());
                assert_eq!(child.schema().metadata(), schemas[i].metadata());
                assert_eq!(
                    child.is::<UnionCoercion>(),
                    schemas[i].fields() != expected.fields()
                );
            }
        }
        Ok(())
    }

    #[test]
    fn union_rejects_inputs_that_do_not_match() {
        let schemas = schemas();
        let one_field =
            Arc::new(Schema::new(vec![Field::new("a", DataType::Int32, true)]));
        let err = PhysicalUnion::try_new(vec![
            empty_source(&schemas[0]),
            empty_source(&one_field),
        ])
        .unwrap_err();
        assert!(err.to_string().contains("same number of fields"), "{err}");
        let err = PhysicalUnion::try_new_with_schema(
            vec![empty_source(&one_field)],
            Arc::clone(&schemas[0]),
        )
        .unwrap_err();
        assert!(err.to_string().contains("same number of fields"), "{err}");

        let other_types = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int64, false),
            Field::new("b", DataType::Utf8, true),
        ]));
        let err = PhysicalUnion::try_new(vec![
            empty_source(&schemas[0]),
            empty_source(&other_types),
        ])
        .unwrap_err();
        assert!(
            err.to_string()
                .contains("column 0 of input 1 has type Int64"),
            "{err}"
        );

        let err = PhysicalUnion::try_new(vec![]).unwrap_err();
        assert!(err.to_string().contains("at least one input"), "{err}");
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn union_relabels_the_chunks_of_coerced_inputs() -> Result<()> {
        // The union schema is the schema of the first input.
        let nullable = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Utf8, true),
        ]));
        let not_null = Arc::new(Schema::new(vec![
            Field::new("x", DataType::Int32, false),
            Field::new("y", DataType::Utf8, false),
        ]));
        let first =
            TestSource::new(vec![DataChunk::from_record_batch(batch(&nullable, 0..10))]);
        let selected = DataChunk::from_record_batch(batch(&not_null, 10..20))
            .with_selection(SelectionVector::from(vec![9, 0, 4]))?;
        let second = TestSource::new(vec![
            selected,
            DataChunk::from_record_batch(batch(&not_null, 20..23)),
        ]);
        let union: Arc<dyn PhysicalOperator> = Arc::new(PhysicalUnion::try_new(vec![
            Arc::new(first),
            Arc::new(second),
        ])?);
        assert_eq!(union.schema(), nullable);
        assert_eq!(
            format_pipelines(&union)?,
            "MetaPipeline 0 (sink: ResultSink)\n  \
             Pipeline 0: TestSource -> ResultSink\n  \
             Pipeline 1: TestSource -> UnionCoercion -> ResultSink\n"
        );

        let output = run(union, 4).await?;
        assert!(output.iter().all(|batch| batch.schema() == nullable));
        let mut values = output
            .iter()
            .flat_map(|batch| {
                let ints = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap();
                let strings = batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap();
                (0..batch.num_rows())
                    .map(|i| (ints.value(i), strings.value(i).to_string()))
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        values.sort_unstable();
        let expected = (0..10)
            .chain([10, 14, 19, 20, 21, 22])
            .map(|v| (v, format!("value {v}")))
            .collect::<Vec<_>>();
        assert_eq!(values, expected);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn union_of_inputs_without_columns() -> Result<()> {
        // Inputs without columns, whose schemas differ only in their metadata.
        let with_metadata =
            Arc::new(Schema::empty().with_metadata(metadata(&[("m", "1")])));
        let empty = Arc::new(Schema::empty());
        let rows = |schema: &SchemaRef, rows| -> Result<Arc<dyn PhysicalOperator>> {
            let chunk = DataChunk::try_new_without_columns(Arc::clone(schema), rows)?;
            Ok(Arc::new(TestSource::with_schema(
                Arc::clone(schema),
                vec![chunk],
            )))
        };
        let union: Arc<dyn PhysicalOperator> = Arc::new(PhysicalUnion::try_new(vec![
            rows(&with_metadata, 1)?,
            rows(&empty, 3)?,
        ])?);
        assert_eq!(union.schema(), with_metadata);
        assert!(union.children()[1].is::<TestSource>());
        let output = run(union, 4).await?;
        assert_eq!(output.iter().map(RecordBatch::num_rows).sum::<usize>(), 4);
        // Every chunk keeps the schema (metadata) of its input.
        for batch in &output {
            let expected = if batch.num_rows() == 1 {
                Arc::clone(&with_metadata)
            } else {
                Arc::new(Schema::empty())
            };
            assert_eq!(batch.schema(), expected);
        }

        // A union of a single input is that input.
        let union: Arc<dyn PhysicalOperator> =
            Arc::new(PhysicalUnion::try_new(vec![Arc::new(
                PhysicalPlaceholderRow::new(empty).with_rows(2),
            )])?);
        assert_eq!(
            format_pipelines(&union)?,
            "MetaPipeline 0 (sink: ResultSink)\n  \
             Pipeline 0: PhysicalPlaceholderRow -> ResultSink\n"
        );
        let output = run(union, 1).await?;
        assert_eq!(output.iter().map(RecordBatch::num_rows).sum::<usize>(), 2);
        Ok(())
    }

    /// A source that only produces its chunk (`value`) once all `inputs`
    /// sources of a test started: the inputs of a union only complete if they
    /// run concurrently.
    #[derive(Debug)]
    struct RendezvousSource {
        started: Arc<AtomicUsize>,
        inputs: usize,
        value: i64,
    }

    #[derive(Default)]
    struct RendezvousState {
        registered: AtomicBool,
        emitted: AtomicBool,
    }

    impl GlobalSourceState for RendezvousState {}

    impl PhysicalOperator for RendezvousSource {
        fn name(&self) -> &str {
            "RendezvousSource"
        }

        fn schema(&self) -> SchemaRef {
            values_schema()
        }

        fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
            &[]
        }

        fn is_source(&self) -> bool {
            true
        }

        fn parallel_source(&self) -> bool {
            true
        }

        fn get_global_source_state(
            &self,
            _ctx: &ExecutionContext,
        ) -> Result<Arc<dyn GlobalSourceState>> {
            Ok(Arc::new(RendezvousState::default()))
        }

        fn get_data(
            &self,
            _ctx: &ExecutionContext,
            gstate: &dyn GlobalSourceState,
            _lstate: &mut dyn LocalSourceState,
        ) -> Result<SourceResult> {
            let state = downcast_state::<RendezvousState>(gstate.as_any(), self.name())?;
            if !state.registered.swap(true, Ordering::SeqCst) {
                self.started.fetch_add(1, Ordering::SeqCst);
            }
            if self.started.load(Ordering::SeqCst) < self.inputs {
                let started = Arc::clone(&self.started);
                let inputs = self.inputs;
                return Ok(SourceResult::Blocked(TaskInterrupt::wait_for(async move {
                    while started.load(Ordering::SeqCst) < inputs {
                        tokio::time::sleep(Duration::from_millis(1)).await;
                    }
                })));
            }
            if state.emitted.swap(true, Ordering::SeqCst) {
                return Ok(SourceResult::Finished);
            }
            Ok(SourceResult::Chunk(values_chunk([self.value])))
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn union_executes_its_inputs_concurrently() -> Result<()> {
        for (inputs, threads) in [(2, 1), (3, 4), (8, 2)] {
            let started = Arc::new(AtomicUsize::new(0));
            let union = PhysicalUnion::try_new(
                (0..inputs)
                    .map(|value| {
                        Arc::new(RendezvousSource {
                            started: Arc::clone(&started),
                            inputs,
                            value: value as i64,
                        }) as Arc<dyn PhysicalOperator>
                    })
                    .collect(),
            )?;
            let output = run(Arc::new(union), threads).await?;
            assert_eq!(
                sorted_values(&output),
                (0..inputs as i64).collect::<Vec<_>>()
            );
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn ordered_union_emits_its_inputs_one_after_the_other() -> Result<()> {
        let input = |start, delay| {
            let source = TestSource::new(values_chunks(start, &[10; 10]));
            Arc::new(source.with_delay(Duration::from_millis(delay)))
                as Arc<dyn PhysicalOperator>
        };
        let union = |inputs| {
            PhysicalUnion::try_new(inputs)
                .unwrap()
                .with_preserve_order(true)
        };
        // A slow first input, and a nested union.
        let nested = union(vec![input(100, 0), input(200, 20)]);
        assert_eq!(nested.description(), "PhysicalUnion: preserve_order=true");
        let root = union(vec![input(0, 30), Arc::new(nested), input(300, 10)]);
        let output = run(Arc::new(root), 4).await?;
        // Within an input, the threads deliver their chunks in any order.
        let inputs = values(&output).iter().map(|v| v / 100).collect::<Vec<_>>();
        assert!(inputs.is_sorted(), "{inputs:?}");
        assert_eq!(sorted_values(&output), (0..400).collect::<Vec<_>>());
        Ok(())
    }

    // ---------------------------------------------------------------------
    // Lowering
    // ---------------------------------------------------------------------

    fn t2_scan() -> Result<Arc<dyn ExecutionPlan>> {
        let partitions = test_table_t2()?;
        let schema = partitions[0][0].schema();
        Ok(MemorySourceConfig::try_new_exec(&partitions, schema, None)?)
    }

    /// `t2`, hash partitioned on `k` into 4 partitions.
    fn t2_by_k() -> Result<Arc<dyn ExecutionPlan>> {
        let scan = t2_scan()?;
        let k = col("k", &scan.schema())?;
        Ok(Arc::new(RepartitionExec::try_new(
            scan,
            Partitioning::Hash(vec![k], 4),
        )?))
    }

    fn bridge(plan: &Arc<dyn ExecutionPlan>) -> Result<Arc<dyn PhysicalOperator>> {
        Ok(Arc::new(BridgeOperator::new(Arc::clone(plan))))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn lowering_of_union_nodes() -> Result<()> {
        let union = UnionExec::try_new(vec![t2_scan()?, t2_scan()?, t2_scan()?])?;
        let lowered = PhysicalUnion::try_lower(&union, bridge)?.expect("a union");
        assert_eq!(lowered.schema(), union.schema());
        assert_eq!(lowered.children().len(), 3);
        assert!(
            lowered
                .children()
                .iter()
                .all(|child| child.is::<BridgeOperator>())
        );
        assert!(!lowered.preserve_order());
        assert_eq!(lowered.description(), "PhysicalUnion");

        let interleave: Arc<dyn ExecutionPlan> =
            Arc::new(InterleaveExec::try_new(vec![t2_by_k()?, t2_by_k()?])?);
        let lowered = PhysicalUnion::try_lower(&interleave, bridge)?.expect("a union");
        assert_eq!(lowered.schema(), interleave.schema());
        assert_eq!(lowered.children().len(), 2);
        // Inputs that are not partitioned alike, which DataFusion fails to
        // interleave: DataFusion executes the node.
        let mismatched = Arc::clone(&interleave).replace_children(
            vec![t2_scan()?, t2_by_k()?],
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )?;
        assert!(mismatched.is::<InterleaveExec>());
        let task_ctx = Arc::new(datafusion_execution::TaskContext::default());
        let failed =
            datafusion_physical_plan::collect(Arc::clone(&mismatched), task_ctx).await;
        assert!(failed.is_err());
        assert!(PhysicalUnion::try_lower(&mismatched, bridge)?.is_none());

        let mut called = false;
        assert!(
            PhysicalUnion::try_lower(&t2_scan()?, |_| {
                called = true;
                internal_err!("not called")
            })?
            .is_none()
        );
        assert!(!called);
        Ok(())
    }

    // ---------------------------------------------------------------------
    // Differential tests
    // ---------------------------------------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_union_all() {
        for sql in [
            "SELECT id FROM t1 WHERE a > 40 UNION ALL SELECT k FROM t2",
            "SELECT k, name FROM t2 UNION ALL SELECT k, name FROM t2",
            "SELECT id, d FROM t1 UNION ALL SELECT k, name FROM t2 UNION ALL SELECT id, h FROM t1",
            "SELECT * FROM t2 UNION ALL SELECT * FROM t2 UNION ALL SELECT * FROM t2 \
             UNION ALL SELECT * FROM t2 UNION ALL SELECT * FROM t2",
            "SELECT * FROM t_empty UNION ALL SELECT * FROM t1 UNION ALL SELECT * FROM t_empty",
            "SELECT a, b, c, d, e, f, g, h, ts FROM t1 \
             UNION ALL SELECT k, w, CAST(w AS DOUBLE), name, k > 0, NULL, NULL, upper(name), NULL \
             FROM t2",
            "SELECT * FROM (SELECT k, w FROM t2 UNION ALL SELECT a, b FROM t1) AS u WHERE k > 0",
            "(SELECT id FROM t1 ORDER BY id LIMIT 3) UNION ALL (SELECT k FROM t2 ORDER BY k DESC LIMIT 3)",
            "SELECT 1 AS x UNION ALL SELECT 2 UNION ALL SELECT 3",
        ] {
            let root = assert_parity(sql).await;
            assert!(count::<PhysicalUnion>(&root) > 0, "{sql}");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_union_all_with_other_partitionings() {
        for target_partitions in [1, 3, 16] {
            let config = test_session_config().with_target_partitions(target_partitions);
            for sql in [
                "SELECT id, d FROM t1 UNION ALL SELECT k, name FROM t2 UNION ALL SELECT id, h FROM t1",
                "SELECT k FROM t2 UNION ALL SELECT k FROM t2 UNION ALL SELECT a FROM t1",
            ] {
                let root = assert_parity_with_config(sql, config.clone()).await;
                assert!(count::<PhysicalUnion>(&root) > 0, "{sql}");
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_union_plans() -> Result<()> {
        // Inputs with different field names, nullability and field metadata.
        // DataFusion's union coerces them with projections, the native union
        // with `UnionCoercion`. (With different schema metadata, the batches
        // of DataFusion's union do not all have the same schema, which the
        // differential harness requires.)
        let schemas = schemas();
        let scan = |i: usize, partitions: usize| -> Result<Arc<dyn ExecutionPlan>> {
            let schema = &schemas[i];
            let partitions = (0..partitions)
                .map(|p| {
                    vec![batch(
                        schema,
                        p as i32 * 1000..p as i32 * 1000 + 300 * (p as i32 + 1),
                    )]
                })
                .collect::<Vec<_>>();
            Ok(MemorySourceConfig::try_new_exec(
                &partitions,
                Arc::clone(schema),
                None,
            )?)
        };
        for indices in [vec![0, 1], vec![1, 2, 0], vec![2, 2]] {
            let inputs = || -> Result<Vec<Arc<dyn ExecutionPlan>>> {
                indices.iter().map(|&i| scan(i, i + 2)).collect()
            };
            let expected = UnionExec::try_new(inputs()?)?;
            let union: Arc<dyn PhysicalOperator> = Arc::new(PhysicalUnion::try_new(
                inputs()?.iter().map(bridge).collect::<Result<_>>()?,
            )?);
            let coerced = union
                .children()
                .iter()
                .filter(|child| child.is::<UnionCoercion>())
                .count();
            assert_eq!(coerced > 0, indices != [2, 2], "{indices:?}");
            let plan = UnionExec::try_new(inputs()?)?;
            assert_operator_byte_eq(expected, union, plan, false).await;
        }

        // A union of hash partitioned inputs.
        let interleave = || -> Result<Arc<dyn ExecutionPlan>> {
            Ok(Arc::new(InterleaveExec::try_new(vec![
                t2_by_k()?,
                t2_by_k()?,
                t2_by_k()?,
            ])?))
        };
        let root = assert_native_plan_byte_eq(interleave, false).await;
        assert_eq!(count::<PhysicalUnion>(&root), 1);
        assert_eq!(interleave()?.output_partitioning().partition_count(), 4);
        Ok(())
    }
}
