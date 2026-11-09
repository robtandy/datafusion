// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`PhysicalNestedLoopJoin`] and [`PhysicalCrossJoin`]: joins that compare
//! every probe row with every build row.
//!
//! [`PhysicalNestedLoopJoin`] replaces DataFusion's `NestedLoopJoinExec` (a
//! join without equi-join keys, whose `ON` clause is a [`JoinFilter`]), with
//! its semantics: all ten join types, its residual filter, its embedded
//! projection and exactly its output schema. [`PhysicalCrossJoin`] replaces
//! `CrossJoinExec`, the Cartesian product of its inputs. Both are the same
//! operator, a [`LoopJoin`], which builds on its first input (the *left*
//! side, DataFusion's build side) and probes with its second one (the
//! *right* side). The join types have DataFusion's names: a `Left` join
//! emits the build rows without a match (DuckDB, which builds on the right
//! side, calls that a `RIGHT` join).
//!
//! # Pipelines
//!
//! Like DuckDB's `PhysicalNestedLoopJoin` and `PhysicalCrossProduct`, a loop
//! join is a sink, an operator and a source:
//!
//! ```text
//! MetaPipeline 0 (sink: ...) children: [1]
//!   Pipeline 0: probe side -> PhysicalNestedLoopJoin -> ... depends on meta pipelines [1]
//!   Pipeline 3: PhysicalNestedLoopJoin -> ... depends on pipelines [0]
//! MetaPipeline 1 (sink: PhysicalNestedLoopJoin)
//!   Pipeline 1: build side -> PhysicalNestedLoopJoin
//!   Pipeline 2: ProbeStreamsInit -> PhysicalNestedLoopJoin
//! ```
//!
//! * **Build** (the join as sink, pipeline 1): every thread compacts the
//!   chunks of the build side it pushes into record batches, which `combine`
//!   hands to the global sink state; `finalize` turns them into the shared,
//!   read-only build side: the batches as they are, addressed by a global row
//!   index (a *segmented batch*, like DataFusion's `LogicalBatch`), so that
//!   nothing is copied or concatenated.
//! * **Probe** (the join as operator, pipeline 0): all threads of the probe
//!   pipeline pull chunks directly from the probe side and join them with the
//!   shared build side, block of build rows after block of build rows. The
//!   join filter is evaluated on the pairs of a block (at most
//!   [`DEFAULT_VECTOR_SIZE`] of them, or a single build row with all rows of
//!   the probe chunk) like DataFusion evaluates it: on an intermediate batch
//!   of the columns it reads, with the filter's schema. The matching pairs
//!   are emitted in chunks of [`DEFAULT_VECTOR_SIZE`] rows; a probe chunk
//!   whose output does not fit into one chunk is continued with
//!   [`OperatorResult::HaveMoreOutput`]. The joins that emit probe rows by
//!   whether they matched (`Right`, `Full`, `RightSemi`, `RightAnti` and
//!   `RightMark`) emit them once the probe chunk was joined with all build
//!   rows.
//! * **Final build rows** (the join as source, pipeline 3): the `Left`,
//!   `Full`, `LeftSemi`, `LeftAnti` and `LeftMark` joins record the build
//!   rows that matched in an atomic bitmap (`Vec<AtomicU64>`, shared by all
//!   threads); once all probe pipelines completed, the threads scan the build
//!   rows in parallel and emit the rows without a match (with a match for
//!   `LeftSemi`, all of them with their mark for `LeftMark`).
//!
//! # Semantics
//!
//! Like DataFusion, a nested loop join reads its whole probe side, also if
//! its build side is empty, and a cross join does not read its probe side if
//! its build side is empty ([`SinkFinalize::NoOutputPossible`]). A join
//! lowered from a DataFusion plan creates the DataFusion streams of the probe
//! input of that plan when the execution starts and drops them without
//! polling them, as `NestedLoopJoinExec::execute` and `CrossJoinExec::execute`
//! create their probe streams before they read their build side, so that the
//! same errors are raised (e.g. of a scan of an object store that is not
//! registered).

use std::fmt::{self, Write};
use std::marker::PhantomData;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use arrow::array::{
    Array, ArrayRef, BooleanArray, RecordBatch, RecordBatchOptions, UInt32Array,
    new_null_array,
};
use arrow::buffer::BooleanBuffer;
use arrow::datatypes::{DataType, Schema, SchemaRef};
use arrow_select::interleave::interleave;
use arrow_select::take::take;
use datafusion_common::cast::as_boolean_array;
use datafusion_common::{
    DataFusionError, JoinSide, JoinType, Result, internal_datafusion_err, internal_err,
    plan_err, resources_datafusion_err,
};
use datafusion_datasource::file_scan_config::FileScanConfig;
use datafusion_datasource::memory::MemorySourceConfig;
use datafusion_datasource::source::DataSourceExec;
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion_physical_plan::empty::EmptyExec;
use datafusion_physical_plan::execution_plan::reset_plan_states;
use datafusion_physical_plan::joins::utils::{
    ColumnIndex, JoinFilter, build_join_schema,
};
use datafusion_physical_plan::joins::{CrossJoinExec, NestedLoopJoinExec};
use datafusion_physical_plan::memory::LazyMemoryExec;
use datafusion_physical_plan::placeholder_row::PlaceholderRowExec;
use datafusion_physical_plan::{ExecutionPlan, ExecutionPlanProperties};
use parking_lot::Mutex;

use crate::bridge::BridgeOperator;
use crate::chunk::{DEFAULT_VECTOR_SIZE, DataChunk};
use crate::operator::{
    ExecutionContext, GlobalOperatorState, GlobalSinkState, GlobalSourceState,
    LocalSinkState, LocalSourceState, OperatorResult, OperatorState, PhysicalOperator,
    SinkFinalize, SinkResult, SourceResult, downcast_state, downcast_state_mut,
};
use crate::operators::adaptive_filter::is_infallible;
use crate::pipeline::{MetaPipelineBuilder, PipelineBuilder};
use crate::sort::memory_size;

/// The join filter is evaluated on blocks of at most this many pairs of rows
/// (or on a single build row and all rows of a probe chunk).
const BLOCK_PAIRS: usize = DEFAULT_VECTOR_SIZE;

/// An `execute` call of the probe joins at most about this many pairs of rows
/// before it returns, so that a probe chunk with few matches over a large
/// build side does not keep its thread busy for long. The call then returns
/// [`OperatorResult::HaveMoreOutput`] (possibly without output) and continues
/// where it stopped.
const PAIRS_PER_CALL: usize = 32 * DEFAULT_VECTOR_SIZE;

/// The kind of a [`LoopJoin`]: a nested loop join ([`NestedLoop`]) or a cross
/// join ([`Cross`]).
pub trait LoopJoinKind: fmt::Debug + Send + Sync + 'static {
    /// The name of the operator.
    const NAME: &'static str;
    /// Whether the join is a cross join: an inner join of all pairs, which
    /// does not read its probe side if its build side is empty.
    const CROSS: bool;
}

/// The kind of [`PhysicalNestedLoopJoin`].
#[derive(Debug)]
pub struct NestedLoop;

impl LoopJoinKind for NestedLoop {
    const NAME: &'static str = "PhysicalNestedLoopJoin";
    const CROSS: bool = false;
}

/// The kind of [`PhysicalCrossJoin`].
#[derive(Debug)]
pub struct Cross;

impl LoopJoinKind for Cross {
    const NAME: &'static str = "PhysicalCrossJoin";
    const CROSS: bool = true;
}

/// A join without equi-join keys, with the semantics of DataFusion's
/// `NestedLoopJoinExec`: see the [module documentation](self).
pub type PhysicalNestedLoopJoin = LoopJoin<NestedLoop>;

/// The Cartesian product of two inputs, with the semantics of DataFusion's
/// `CrossJoinExec`: see the [module documentation](self).
pub type PhysicalCrossJoin = LoopJoin<Cross>;

/// A join that compares every row of its probe side (its right input) with
/// every row of its build side (its left input): a
/// [`PhysicalNestedLoopJoin`] or a [`PhysicalCrossJoin`].
///
/// See the [module documentation](self) for its pipelines and semantics.
pub struct LoopJoin<K: LoopJoinKind> {
    /// The build side (the left input) and the probe side (the right input).
    children: Vec<Arc<dyn PhysicalOperator>>,
    spec: JoinSpec,
    /// The probe input of the DataFusion join the join was lowered from,
    /// whose streams DataFusion creates when it executes the join, before it
    /// reads its build side: the join creates them too (see
    /// [`LoopJoin::create_probe_streams`]), to raise the same errors.
    probe_plan: Option<Arc<dyn ExecutionPlan>>,
    /// The number of partitions of the probe input of the DataFusion join,
    /// which decides whether DataFusion can spill its build side.
    probe_partitions: usize,
    kind: PhantomData<K>,
}

impl<K: LoopJoinKind> fmt::Debug for LoopJoin<K> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct(K::NAME)
            .field("join_type", &self.spec.join_type)
            .field("filter", &self.spec.filter)
            .field("projection", &self.spec.projection)
            .field("schema", &self.spec.schema)
            .field("children", &self.children)
            .finish_non_exhaustive()
    }
}

/// Everything about a join that does not depend on its inputs, only on
/// their schemas.
#[derive(Debug)]
struct JoinSpec {
    join_type: JoinType,
    filter: Option<JoinFilter>,
    projection: Option<Vec<usize>>,
    /// The schema of the join without its projection.
    join_schema: SchemaRef,
    /// The output schema.
    schema: SchemaRef,
    /// The input column of every output column (or the mark column).
    column_indices: Vec<ColumnIndex>,
}

impl JoinSpec {
    /// The join of a `NestedLoopJoinExec` with these arguments.
    fn try_new(
        build_schema: &SchemaRef,
        probe_schema: &SchemaRef,
        filter: Option<JoinFilter>,
        join_type: JoinType,
        projection: Option<Vec<usize>>,
    ) -> Result<Self> {
        let (join_schema, all_columns) =
            build_join_schema(build_schema, probe_schema, &join_type);
        let join_schema = Arc::new(join_schema);
        let (schema, column_indices) = match &projection {
            Some(projection) => {
                let columns = projection
                    .iter()
                    .map(|&i| {
                        all_columns.get(i).cloned().ok_or_else(|| {
                            internal_datafusion_err!(
                                "Projection index {i} out of bounds for a join with {} columns",
                                all_columns.len()
                            )
                        })
                    })
                    .collect::<Result<Vec<_>>>()?;
                (Arc::new(join_schema.project(projection)?), columns)
            }
            None => (Arc::clone(&join_schema), all_columns),
        };
        if let Some(filter) = &filter {
            check_filter(filter, build_schema, probe_schema)?;
        }
        Ok(Self {
            join_type,
            filter,
            projection,
            join_schema,
            schema,
            column_indices,
        })
    }

    /// The join of a `CrossJoinExec` of inputs with these schemas: all
    /// columns of the build side, then all columns of the probe side, with
    /// the metadata of both schemas (the probe side's wins), like
    /// `CrossJoinExec::new`.
    fn cross(build_schema: &SchemaRef, probe_schema: &SchemaRef) -> Self {
        let mut metadata = build_schema.metadata().clone();
        metadata.extend(probe_schema.metadata().clone());
        let fields = build_schema
            .fields()
            .iter()
            .chain(probe_schema.fields())
            .cloned()
            .collect::<Vec<_>>();
        let schema = Arc::new(Schema::new_with_metadata(fields, metadata));
        let column_indices = (0..build_schema.fields().len())
            .map(|index| ColumnIndex {
                index,
                side: JoinSide::Left,
            })
            .chain((0..probe_schema.fields().len()).map(|index| ColumnIndex {
                index,
                side: JoinSide::Right,
            }))
            .collect();
        Self {
            join_type: JoinType::Inner,
            filter: None,
            projection: None,
            join_schema: Arc::clone(&schema),
            schema,
            column_indices,
        }
    }

    /// Whether the join emits pairs of a build row and a probe row (the
    /// other joins emit rows of one side only).
    fn emits_pairs(&self) -> bool {
        matches!(
            self.join_type,
            JoinType::Inner | JoinType::Left | JoinType::Right | JoinType::Full
        )
    }

    /// Whether the join emits build rows once the probe side is exhausted
    /// (DataFusion's `need_produce_result_in_final`), and therefore records
    /// the build rows that matched.
    fn emits_final_build_rows(&self) -> bool {
        matches!(
            self.join_type,
            JoinType::Left
                | JoinType::Full
                | JoinType::LeftSemi
                | JoinType::LeftAnti
                | JoinType::LeftMark
        )
    }

    /// Whether the join emits probe rows by whether they matched, once they
    /// were joined with all build rows (DataFusion's
    /// `need_produce_right_in_final`).
    fn emits_probe_rows(&self) -> bool {
        matches!(
            self.join_type,
            JoinType::Right
                | JoinType::Full
                | JoinType::RightSemi
                | JoinType::RightAnti
                | JoinType::RightMark
        )
    }

    /// Whether the join emits nothing but probe rows by whether they matched:
    /// once all rows of a probe chunk matched, the other build rows do not
    /// matter for it.
    fn only_needs_probe_matches(&self) -> bool {
        matches!(
            self.join_type,
            JoinType::RightSemi | JoinType::RightAnti | JoinType::RightMark
        )
    }

    /// Whether the join emits nothing but build rows by whether they matched:
    /// build rows that matched already do not matter any more.
    fn only_needs_build_matches(&self) -> bool {
        matches!(
            self.join_type,
            JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark
        )
    }
}

/// Check that the columns that `filter` reads exist in the inputs.
fn check_filter(
    filter: &JoinFilter,
    build_schema: &Schema,
    probe_schema: &Schema,
) -> Result<()> {
    let columns = filter.column_indices();
    if columns.len() != filter.schema().fields().len() {
        return plan_err!(
            "The filter of a nested loop join reads {} columns, but its schema has {} fields",
            columns.len(),
            filter.schema().fields().len()
        );
    }
    for column in columns {
        let exists = match column.side {
            JoinSide::Left => column.index < build_schema.fields().len(),
            JoinSide::Right => column.index < probe_schema.fields().len(),
            JoinSide::None => false,
        };
        if !exists {
            return plan_err!(
                "The filter of a nested loop join reads column {} of side {:?}, which its input does not have",
                column.index,
                column.side
            );
        }
    }
    Ok(())
}

impl PhysicalNestedLoopJoin {
    /// A nested loop join of `build` (DataFusion's left input) and `probe`
    /// (its right input), with the semantics of a `NestedLoopJoinExec` with
    /// the same arguments: the pairs of rows for which `filter` is true (all
    /// pairs without filter), with the output schema of that
    /// `NestedLoopJoinExec`.
    pub fn try_new(
        build: Arc<dyn PhysicalOperator>,
        probe: Arc<dyn PhysicalOperator>,
        filter: Option<JoinFilter>,
        join_type: JoinType,
        projection: Option<Vec<usize>>,
    ) -> Result<Self> {
        let spec = JoinSpec::try_new(
            &build.schema(),
            &probe.schema(),
            filter,
            join_type,
            projection,
        )?;
        Ok(Self::with_spec(build, probe, spec))
    }

    /// Lower a DataFusion `NestedLoopJoinExec` to a [`PhysicalNestedLoopJoin`]
    /// with the same semantics and schema. `lower_input` lowers the inputs of
    /// the join. Returns `Ok(None)` for any other node, and for the joins that
    /// are left to DataFusion:
    ///
    /// * the joins DataFusion does not execute as a join of all rows of both
    ///   inputs: with a left input of another number of partitions than one
    ///   (DataFusion fails to execute it), and without probe partitions (the
    ///   join then has no output partitions, so that DataFusion executes
    ///   neither input);
    /// * a join whose filter can fail on some rows (conservatively, see
    ///   [`is_infallible`]): DataFusion evaluates it on other blocks of pairs
    ///   than the native join, and skips the pairs of probe rows that matched
    ///   already for `RightSemi` and `RightAnti` joins, so that, e.g. under a
    ///   limit, one of them could fail where the other does not;
    /// * a join with dictionaries with keys of 8 or 16 bits, or run-end
    ///   encodings with 16-bit run ends, in either input: DataFusion
    ///   concatenates its output into batches of `batch_size` rows, the
    ///   native join gathers chunks of other rows, and either can overflow
    ///   such keys or run ends where the other does not;
    /// * a join that produces dynamic filters (none does today).
    ///
    /// The build side is lowered without the `CoalescePartitionsExec` (without
    /// limit) that DataFusion puts on it: the build side of the native join
    /// collects the rows of all its threads. The probe side of a join is left
    /// to DataFusion (bridged) if its lowered form has pipeline breakers, e.g.
    /// a join or an aggregation: their pipelines run independently of the
    /// build side of this join, while DataFusion reads the probe side only
    /// once the build side is collected, so that they would compute (and
    /// raise errors) where DataFusion does not.
    pub fn try_lower<F>(
        plan: &Arc<dyn ExecutionPlan>,
        mut lower_input: F,
    ) -> Result<Option<Self>>
    where
        F: FnMut(&Arc<dyn ExecutionPlan>) -> Result<Arc<dyn PhysicalOperator>>,
    {
        let Some(join) = plan.downcast_ref::<NestedLoopJoinExec>() else {
            return Ok(None);
        };
        if !executes_all_rows(join.left(), join.right())
            || !plan.dynamic_expressions_produced().is_empty()
            || join.filter().is_some_and(|filter| {
                !is_infallible(filter.expression(), filter.schema())
            })
            || has_narrow_encodings(join.left(), join.right())
        {
            return Ok(None);
        }
        let spec = match JoinSpec::try_new(
            &join.left().schema(),
            &join.right().schema(),
            join.filter().cloned(),
            *join.join_type(),
            join.projection().as_deref().map(<[usize]>::to_vec),
        ) {
            Ok(spec) => spec,
            Err(DataFusionError::Plan(_)) => return Ok(None),
            Err(e) => return Err(e),
        };
        if spec.schema != plan.schema() {
            return internal_err!(
                "PhysicalNestedLoopJoin would have schema {:?}, but the NestedLoopJoinExec has schema {:?}",
                spec.schema,
                plan.schema()
            );
        }
        Self::lower_inputs(spec, join.left(), join.right(), &mut lower_input)
    }
}

impl PhysicalCrossJoin {
    /// The Cartesian product of `build` (DataFusion's left input) and
    /// `probe` (its right input), with the semantics and the output schema
    /// of a `CrossJoinExec` of inputs with these schemas.
    pub fn try_new(
        build: Arc<dyn PhysicalOperator>,
        probe: Arc<dyn PhysicalOperator>,
    ) -> Result<Self> {
        let spec = JoinSpec::cross(&build.schema(), &probe.schema());
        Ok(Self::with_spec(build, probe, spec))
    }

    /// Lower a DataFusion `CrossJoinExec` to a [`PhysicalCrossJoin`] with the
    /// same semantics and schema. `lower_input` lowers the inputs of the join.
    /// Returns `Ok(None)` for any other node, and for the joins that are left
    /// to DataFusion: those it does not execute as a join of all rows of both
    /// inputs, and those with narrow dictionaries or run-end encodings (see
    /// [`PhysicalNestedLoopJoin::try_lower`]), and those whose probe input
    /// cannot be copied with its own state (see `reset_plan_states`): with a
    /// leaf other than a scan of files or of memory, or of no rows (e.g. a
    /// `StreamingTableExec`, whose partitions may open a file or take a
    /// channel when they are executed), or with a node that is not copied.
    /// `CrossJoinExec::execute` creates the streams of its probe input right
    /// away, also when it does not read them because its build side is empty;
    /// the native join creates them too, on such a copy, to raise the same
    /// errors.
    ///
    /// The inputs are lowered like those of a nested loop join, see
    /// [`PhysicalNestedLoopJoin::try_lower`].
    pub fn try_lower<F>(
        plan: &Arc<dyn ExecutionPlan>,
        mut lower_input: F,
    ) -> Result<Option<Self>>
    where
        F: FnMut(&Arc<dyn ExecutionPlan>) -> Result<Arc<dyn PhysicalOperator>>,
    {
        let Some(join) = plan.downcast_ref::<CrossJoinExec>() else {
            return Ok(None);
        };
        if !executes_all_rows(&join.left, &join.right)
            || has_narrow_encodings(&join.left, &join.right)
            || !probe_streams_can_be_replayed(&join.right)
        {
            return Ok(None);
        }
        let spec = JoinSpec::cross(&join.left.schema(), &join.right.schema());
        if spec.schema != plan.schema() {
            return internal_err!(
                "PhysicalCrossJoin would have schema {:?}, but the CrossJoinExec has schema {:?}",
                spec.schema,
                plan.schema()
            );
        }
        Self::lower_inputs(spec, &join.left, &join.right, &mut lower_input)
    }
}

/// Whether DataFusion executes a join of `left` and `right` (a
/// `NestedLoopJoinExec` or a `CrossJoinExec`) as a join of all rows of both
/// inputs: its left input has a single partition (otherwise executing it
/// fails), and its right input has partitions (otherwise the join has no
/// output partitions and DataFusion executes neither input).
fn executes_all_rows(
    left: &Arc<dyn ExecutionPlan>,
    right: &Arc<dyn ExecutionPlan>,
) -> bool {
    left.output_partitioning().partition_count() == 1
        && right.output_partitioning().partition_count() > 0
}

/// Whether an input of a join of `left` and `right` has dictionaries with
/// keys of 8 or 16 bits, or run-end encodings with run ends of 16 bits.
fn has_narrow_encodings(
    left: &Arc<dyn ExecutionPlan>,
    right: &Arc<dyn ExecutionPlan>,
) -> bool {
    [left, right].iter().any(|input| {
        input
            .schema()
            .fields()
            .iter()
            .any(|field| has_narrow_encoding(field.data_type()))
    })
}

impl<K: LoopJoinKind> LoopJoin<K> {
    fn with_spec(
        build: Arc<dyn PhysicalOperator>,
        probe: Arc<dyn PhysicalOperator>,
        spec: JoinSpec,
    ) -> Self {
        Self {
            children: vec![build, probe],
            spec,
            probe_plan: None,
            probe_partitions: 1,
            kind: PhantomData,
        }
    }

    /// The join with `spec` of the lowered DataFusion inputs `left` and
    /// `right`, see [`PhysicalNestedLoopJoin::try_lower`].
    fn lower_inputs<F>(
        spec: JoinSpec,
        left: &Arc<dyn ExecutionPlan>,
        right: &Arc<dyn ExecutionPlan>,
        lower_input: &mut F,
    ) -> Result<Option<Self>>
    where
        F: FnMut(&Arc<dyn ExecutionPlan>) -> Result<Arc<dyn PhysicalOperator>>,
    {
        let build_input = without_coalesce(left);
        let build = lower_input(build_input)?;
        check_lowered_input(build_input, &build)?;
        let mut probe = lower_input(right)?;
        check_lowered_input(right, &probe)?;
        // DataFusion reads the probe side only once the build side is
        // collected. The pipelines of the pipeline breakers of a native probe
        // side do not wait for the build side, so they would compute (and
        // raise errors) where DataFusion does not: such a probe side is left
        // to DataFusion, which executes it only once the join reads it.
        if has_pipeline_breakers(&probe) {
            probe = Arc::new(BridgeOperator::new(Arc::clone(right)));
        }
        let mut join = Self::with_spec(build, probe, spec);
        join.probe_plan = probe_streams_can_be_replayed(right).then(|| Arc::clone(right));
        join.probe_partitions = right.output_partitioning().partition_count();
        Ok(Some(join))
    }

    /// The build side (DataFusion's left input).
    pub fn build_side(&self) -> &Arc<dyn PhysicalOperator> {
        &self.children[0]
    }

    /// The probe side (DataFusion's right input).
    pub fn probe_side(&self) -> &Arc<dyn PhysicalOperator> {
        &self.children[1]
    }

    /// The join type (`Inner` for a cross join).
    pub fn join_type(&self) -> JoinType {
        self.spec.join_type
    }

    /// The join filter (`None` for a cross join).
    pub fn filter(&self) -> Option<&JoinFilter> {
        self.spec.filter.as_ref()
    }

    /// The output columns, as indices of the columns of the join without
    /// projection.
    pub fn projection(&self) -> Option<&[usize]> {
        self.spec.projection.as_deref()
    }

    /// Create the streams of all partitions of the probe input of the
    /// DataFusion join the join was lowered from and drop them without
    /// polling them, as DataFusion creates the probe stream of a partition
    /// before it reads its build side (and also if it then does not read its
    /// probe side): the same errors (e.g. of a scan of an object store that
    /// is not registered), nothing read.
    ///
    /// The streams are those of a copy of the probe input with its own state
    /// (see `reset_plan_states`, and [`probe_streams_can_be_replayed`]):
    /// creating them can start tasks that already read (e.g. those of a
    /// `CoalescePartitionsExec`), which must not take what the probe side of
    /// the join still reads (e.g. the channels of a `RepartitionExec`).
    fn create_probe_streams(&self, ctx: &ExecutionContext) -> Result<()> {
        let Some(probe) = &self.probe_plan else {
            return Ok(());
        };
        let probe = reset_plan_states(Arc::clone(probe))?;
        for partition in 0..probe.output_partitioning().partition_count() {
            drop(probe.execute(partition, Arc::clone(&ctx.task_ctx))?);
        }
        Ok(())
    }

    /// Whether the memory of the build side may exceed the memory pool, as
    /// DataFusion then spills it. A `CrossJoinExec` never spills (it fails
    /// with `ResourcesExhausted` like the native join), a
    /// `NestedLoopJoinExec` spills its build side when the disk manager can
    /// create temporary files, unless the coordinated fallback is disabled
    /// for a join that emits build rows at the end over several probe
    /// partitions. The native join does not spill: where DataFusion would, it
    /// keeps the build side in memory beyond the limit of the pool rather than
    /// failing.
    fn build_may_exceed_memory_pool(&self, ctx: &ExecutionContext) -> bool {
        if K::CROSS {
            return false;
        }
        let task_ctx = &ctx.task_ctx;
        let fallback_disabled = !task_ctx
            .session_config()
            .options()
            .execution
            .enable_nlj_coordinated_fallback
            && self.spec.emits_final_build_rows()
            && self.probe_partitions > 1;
        task_ctx.runtime_env().disk_manager.tmp_files_enabled() && !fallback_disabled
    }

    /// Join the rows of the current probe chunk of `state` with the next
    /// build rows, writing the output to `output`. Returns `true` once the
    /// probe chunk is complete.
    fn probe_next(&self, state: &mut ProbeState, output: &mut DataChunk) -> Result<bool> {
        let spec = &self.spec;
        let ProbeState { build, current } = state;
        let build: &BuildSide = build;
        let Some(chunk) = current.as_mut() else {
            return internal_err!("{} has no probe chunk", K::NAME);
        };
        let mut joined = 0;
        loop {
            if chunk.pending_probe.len() >= DEFAULT_VECTOR_SIZE {
                *output = self.pairs_chunk(build, chunk, DEFAULT_VECTOR_SIZE)?;
                return Ok(false);
            }
            let probe_decided = spec.only_needs_probe_matches() && chunk.unmatched == 0;
            if chunk.next_build < build.rows.num_rows() && !probe_decided {
                if joined >= PAIRS_PER_CALL {
                    // Let the pipeline run: continued in the next call.
                    return Ok(false);
                }
                joined += self.join_block(build, chunk)?;
                continue;
            }
            if !chunk.pending_probe.is_empty() {
                let num_pairs = chunk.pending_probe.len();
                *output = self.pairs_chunk(build, chunk, num_pairs)?;
                return Ok(false);
            }
            if let Some(rows) = self.probe_rows_chunk(chunk)? {
                *output = rows;
            }
            return Ok(true);
        }
    }

    /// Join the rows of `chunk` with its next block of build rows: at most
    /// [`BLOCK_PAIRS`] pairs (or one build row), all of them rows of one
    /// segment of the build side. Records the matches and the pairs to emit.
    /// Returns the number of pairs joined.
    fn join_block(&self, build: &BuildSide, chunk: &mut ProbeChunk) -> Result<usize> {
        let spec = &self.spec;
        let rows = &build.rows;
        let segment_index = rows.segment_of(chunk.next_build);
        let segment = &rows.segments[segment_index];
        let segment_end = rows.offsets[segment_index + 1];
        // The build rows that matched already need no more matches: skip them.
        if spec.only_needs_build_matches()
            && let Some(matched) = &build.matched
        {
            match matched.next_unset(chunk.next_build..segment_end) {
                Some(row) => chunk.next_build = row,
                None => {
                    chunk.next_build = segment_end;
                    return Ok(0);
                }
            }
        }
        let first = chunk.next_build;
        let start = first - rows.offsets[segment_index];
        let probe_rows = chunk.batch.num_rows();
        let len = (BLOCK_PAIRS / probe_rows)
            .max(1)
            .min(segment.num_rows() - start);
        chunk.next_build += len;
        let num_pairs = len * probe_rows;
        let mask = match &spec.filter {
            Some(filter) => {
                Some(evaluate_filter(filter, segment, start, len, &chunk.batch)?)
            }
            None => None,
        };
        let Some(mask) = mask else {
            // Without filter, all pairs match.
            if spec.emits_pairs() {
                chunk.pending_build.try_reserve(num_pairs).map_err(|e| {
                    resources_datafusion_err!(
                        "Failed to allocate the pairs of a join: {e}"
                    )
                })?;
                chunk.pending_probe.try_reserve(num_pairs).map_err(|e| {
                    resources_datafusion_err!(
                        "Failed to allocate the pairs of a join: {e}"
                    )
                })?;
                for row in first..first + len {
                    chunk
                        .pending_build
                        .extend(std::iter::repeat_n(row, probe_rows));
                    chunk.pending_probe.extend(0..probe_rows as u32);
                }
            }
            if spec.emits_probe_rows() {
                chunk.matched.fill(true);
                chunk.unmatched = 0;
            }
            if let Some(matched) = &build.matched {
                (first..first + len).for_each(|row| matched.set(row));
            }
            return Ok(num_pairs);
        };
        let mut block_matched = vec![false; len];
        for pair in mask.set_indices() {
            let (build_row, probe_row) = (pair / probe_rows, pair % probe_rows);
            block_matched[build_row] = true;
            if spec.emits_probe_rows() && !chunk.matched[probe_row] {
                chunk.matched[probe_row] = true;
                chunk.unmatched -= 1;
            }
            if spec.emits_pairs() {
                chunk.pending_build.push(first + build_row);
                chunk.pending_probe.push(probe_row as u32);
            }
        }
        if let Some(matched) = &build.matched {
            for (row, _) in block_matched.iter().enumerate().filter(|(_, m)| **m) {
                matched.set(first + row);
            }
        }
        Ok(num_pairs)
    }

    /// The first `num_pairs` pending pairs of `chunk` as an output chunk.
    fn pairs_chunk(
        &self,
        build: &BuildSide,
        chunk: &mut ProbeChunk,
        num_pairs: usize,
    ) -> Result<DataChunk> {
        let probe_rows = UInt32Array::from(chunk.pending_probe[..num_pairs].to_vec());
        let output = self.output_chunk(
            num_pairs,
            Some(BuildRowsRef::Global(
                &build.rows,
                &chunk.pending_build[..num_pairs],
            )),
            Some((&chunk.batch, &probe_rows)),
            None,
        )?;
        chunk.pending_build.drain(..num_pairs);
        chunk.pending_probe.drain(..num_pairs);
        Ok(output)
    }

    /// The probe rows of `chunk` that a `Right`, `Full`, `RightSemi`,
    /// `RightAnti` or `RightMark` join emits once they were joined with all
    /// build rows, `None` if there is none.
    fn probe_rows_chunk(&self, chunk: &ProbeChunk) -> Result<Option<DataChunk>> {
        let rows = |matched: bool| {
            chunk
                .matched
                .iter()
                .enumerate()
                .filter(|&(_, &m)| m == matched)
                .map(|(row, _)| row as u32)
                .collect::<UInt32Array>()
        };
        let (rows, mark) = match self.spec.join_type {
            JoinType::Right | JoinType::Full | JoinType::RightAnti => (rows(false), None),
            JoinType::RightSemi => (rows(true), None),
            JoinType::RightMark => {
                let mark: ArrayRef = Arc::new(BooleanArray::from(chunk.matched.clone()));
                let all = (0..chunk.batch.num_rows() as u32).collect::<UInt32Array>();
                (all, Some(mark))
            }
            _ => return Ok(None),
        };
        if rows.is_empty() {
            return Ok(None);
        }
        let output = self.output_chunk(
            rows.len(),
            None,
            Some((&chunk.batch, &rows)),
            mark.as_ref(),
        )?;
        Ok(Some(output))
    }

    /// The final build rows of the next block of build rows claimed by this
    /// thread, `None` once all blocks are claimed.
    fn next_final_rows(&self, global: &FinalScanState) -> Result<Option<DataChunk>> {
        let build = global.build.as_ref();
        let Some(matched) = &build.matched else {
            return internal_err!("{} does not record the matched build rows", K::NAME);
        };
        loop {
            let block = global.next_block.fetch_add(1, Ordering::Relaxed);
            let Some(&(segment, start, len)) = global.blocks.get(block) else {
                return Ok(None);
            };
            let first = build.rows.offsets[segment] + start;
            let flags = (first..first + len)
                .map(|row| matched.get(row))
                .collect::<Vec<_>>();
            let select = |keep: bool| {
                flags
                    .iter()
                    .enumerate()
                    .filter(|&(_, &m)| m == keep)
                    .map(|(row, _)| (start + row) as u32)
                    .collect::<UInt32Array>()
            };
            let (rows, mark): (UInt32Array, Option<ArrayRef>) = match self.spec.join_type
            {
                JoinType::Left | JoinType::Full | JoinType::LeftAnti => {
                    (select(false), None)
                }
                JoinType::LeftSemi => (select(true), None),
                JoinType::LeftMark => (
                    (start as u32..(start + len) as u32).collect(),
                    Some(Arc::new(BooleanArray::from(flags))),
                ),
                join_type => {
                    return internal_err!(
                        "A {join_type} join does not emit build rows at the end"
                    );
                }
            };
            if rows.is_empty() {
                continue;
            }
            let segment = &build.rows.segments[segment];
            let chunk = self.output_chunk(
                rows.len(),
                Some(BuildRowsRef::Segment(segment, &rows)),
                None,
                mark.as_ref(),
            )?;
            return Ok(Some(chunk));
        }
    }

    /// An output chunk of `num_rows` rows:
    ///
    /// * the build columns are taken from the rows `build` (NULL if `None`);
    /// * the probe columns are taken from the rows `probe.1` of `probe.0`
    ///   (NULL if `None`);
    /// * the mark column is `mark`.
    fn output_chunk(
        &self,
        num_rows: usize,
        build: Option<BuildRowsRef<'_>>,
        probe: Option<(&RecordBatch, &UInt32Array)>,
        mark: Option<&ArrayRef>,
    ) -> Result<DataChunk> {
        let spec = &self.spec;
        if num_rows == 0 {
            return Ok(DataChunk::empty(Arc::clone(&spec.schema)));
        }
        if spec.schema.fields().is_empty() {
            return DataChunk::try_new_without_columns(
                Arc::clone(&spec.schema),
                num_rows,
            );
        }
        let gather = build.map(BuildGather::new).transpose()?;
        let columns = spec
            .column_indices
            .iter()
            .zip(spec.schema.fields())
            .map(|(column, field)| match column.side {
                JoinSide::Left => match &gather {
                    Some(gather) => gather.column(column.index),
                    None => Ok(new_null_array(field.data_type(), num_rows)),
                },
                JoinSide::Right => match probe {
                    Some((batch, rows)) => take_rows(batch.column(column.index), rows),
                    None => Ok(new_null_array(field.data_type(), num_rows)),
                },
                JoinSide::None => mark.map(Arc::clone).ok_or_else(|| {
                    internal_datafusion_err!("Missing mark column of the join")
                }),
            })
            .collect::<Result<Vec<_>>>()?;
        DataChunk::try_new(Arc::clone(&spec.schema), columns, None)
    }
}

/// Evaluate `filter` on the pairs of the rows `start..start + len` of the
/// build segment `segment` (each with all rows of `probe`, build row after
/// build row): whether each pair matches (`NULL` does not).
fn evaluate_filter(
    filter: &JoinFilter,
    segment: &RecordBatch,
    start: usize,
    len: usize,
    probe: &RecordBatch,
) -> Result<BooleanBuffer> {
    let probe_rows = probe.num_rows();
    let num_pairs = len * probe_rows;
    // With a single build row, the probe columns are used as they are.
    let build_indices = UInt32Array::from_iter_values(
        (start..start + len).flat_map(|row| std::iter::repeat_n(row as u32, probe_rows)),
    );
    let probe_indices = (len > 1).then(|| {
        UInt32Array::from_iter_values((0..len).flat_map(|_| 0..probe_rows as u32))
    });
    let columns = filter
        .column_indices()
        .iter()
        .map(|column| match column.side {
            JoinSide::Left => Ok(take(
                segment.column(column.index).as_ref(),
                &build_indices,
                None,
            )?),
            JoinSide::Right => match &probe_indices {
                Some(indices) => {
                    Ok(take(probe.column(column.index).as_ref(), indices, None)?)
                }
                None => Ok(Arc::clone(probe.column(column.index))),
            },
            JoinSide::None => internal_err!("A join filter cannot read the mark column"),
        })
        .collect::<Result<Vec<_>>>()?;
    let options = RecordBatchOptions::new().with_row_count(Some(num_pairs));
    let intermediate = RecordBatch::try_new_with_options(
        Arc::clone(filter.schema()),
        columns,
        &options,
    )?;
    let result = filter
        .expression()
        .evaluate(&intermediate)?
        .into_array(num_pairs)?;
    let mask = as_boolean_array(&result)?;
    if mask.len() != num_pairs {
        return internal_err!(
            "The join filter returned {} values for {num_pairs} pairs",
            mask.len()
        );
    }
    Ok(match mask.nulls() {
        Some(nulls) => mask.values() & nulls.inner(),
        None => mask.values().clone(),
    })
}

/// The rows `indices` of `array`: zero copy if they are a contiguous range.
fn take_rows(array: &ArrayRef, indices: &UInt32Array) -> Result<ArrayRef> {
    let values = indices.values();
    if let (Some(&first), Some(&last)) = (values.first(), values.last())
        && indices.null_count() == 0
        && last.checked_sub(first).map(|span| span as usize + 1) == Some(values.len())
        && values.windows(2).all(|w| w[1] == w[0] + 1)
    {
        let (first, len) = (first as usize, values.len());
        if first == 0 && len == array.len() {
            return Ok(Arc::clone(array));
        }
        return Ok(array.slice(first, len));
    }
    Ok(take(array.as_ref(), indices, None)?)
}

/// The build rows of an output chunk.
#[derive(Clone, Copy)]
enum BuildRowsRef<'a> {
    /// Rows of the build side, by their global index.
    Global(&'a SegmentedBatch, &'a [usize]),
    /// Rows of one segment of the build side.
    Segment(&'a RecordBatch, &'a UInt32Array),
}

/// How the columns of build rows are gathered: taken from a single segment,
/// or interleaved from several.
enum BuildGather<'a> {
    Take(&'a RecordBatch, UInt32Array),
    TakeRef(&'a RecordBatch, &'a UInt32Array),
    Interleave(Vec<&'a RecordBatch>, Vec<(usize, usize)>),
}

impl<'a> BuildGather<'a> {
    fn new(rows: BuildRowsRef<'a>) -> Result<Self> {
        match rows {
            BuildRowsRef::Segment(segment, rows) => Ok(Self::TakeRef(segment, rows)),
            BuildRowsRef::Global(build, rows) => {
                let (Some(&first), Some(&last)) = (rows.first(), rows.last()) else {
                    return internal_err!("No build rows to gather");
                };
                let first_segment = build.segment_of(first);
                let segment_rows =
                    build.offsets[first_segment]..build.offsets[first_segment + 1];
                // The pairs of a probe chunk come in the order of their build
                // rows: all rows are in the segment of the first one if the
                // last one is.
                debug_assert!(rows.is_sorted());
                if segment_rows.contains(&last) {
                    let offset = segment_rows.start;
                    let indices = rows
                        .iter()
                        .map(|&row| (row - offset) as u32)
                        .collect::<UInt32Array>();
                    return Ok(Self::Take(&build.segments[first_segment], indices));
                }
                // Rows of several segments: interleave the segments they come
                // from.
                let mut slots = vec![usize::MAX; build.segments.len()];
                let mut sources = vec![];
                let mut positions = Vec::with_capacity(rows.len());
                let mut segment = first_segment;
                for &row in rows {
                    if !(build.offsets[segment]..build.offsets[segment + 1])
                        .contains(&row)
                    {
                        segment = build.segment_of(row);
                    }
                    if slots[segment] == usize::MAX {
                        slots[segment] = sources.len();
                        sources.push(&build.segments[segment]);
                    }
                    positions.push((slots[segment], row - build.offsets[segment]));
                }
                Ok(Self::Interleave(sources, positions))
            }
        }
    }

    /// The build column `column` of the gathered rows.
    fn column(&self, column: usize) -> Result<ArrayRef> {
        match self {
            Self::Take(segment, rows) => take_rows(segment.column(column), rows),
            Self::TakeRef(segment, rows) => take_rows(segment.column(column), rows),
            Self::Interleave(sources, positions) => {
                let values = sources
                    .iter()
                    .map(|segment| segment.column(column).as_ref())
                    .collect::<Vec<&dyn Array>>();
                Ok(interleave(&values, positions)?)
            }
        }
    }
}

/// The input `plan` of the build side of a join without the
/// `CoalescePartitionsExec`s (without limit) that only merge its partitions:
/// the build side of a native join collects the rows of all its threads.
fn without_coalesce(plan: &Arc<dyn ExecutionPlan>) -> &Arc<dyn ExecutionPlan> {
    let mut plan = plan;
    while let Some(coalesce) = plan.downcast_ref::<CoalescePartitionsExec>()
        && coalesce.fetch().is_none()
        // Without input partitions, DataFusion fails to execute it.
        && coalesce.input().output_partitioning().partition_count() > 0
    {
        plan = coalesce.input();
    }
    plan
}

/// Check that the lowered form `lowered` of the input `input` of a join has
/// the schema of `input`.
fn check_lowered_input(
    input: &Arc<dyn ExecutionPlan>,
    lowered: &Arc<dyn PhysicalOperator>,
) -> Result<()> {
    if lowered.schema() != input.schema() {
        return internal_err!(
            "The lowered input {} of a join has schema {:?}, expected {:?}",
            lowered.name(),
            lowered.schema(),
            input.schema()
        );
    }
    Ok(())
}

/// Whether `operator`, or an operator below it, is a pipeline breaker (a
/// sink, e.g. a join or an aggregation), whose input runs in pipelines of its
/// own. A bridged plan is a source: DataFusion executes it when it is read.
fn has_pipeline_breakers(operator: &Arc<dyn PhysicalOperator>) -> bool {
    operator.is_sink() || operator.children().iter().any(has_pipeline_breakers)
}

/// Whether the streams of a copy of `plan` with its own state (see
/// `reset_plan_states`) can be created (and dropped) without changing what
/// `plan` itself reads: all its leaves are scans that read only when polled
/// (of files or memory, or of no rows), and all its other nodes are copied.
/// Other leaves, e.g. a `StreamingTableExec`, whose partitions may open a
/// file or take a channel when they are executed, are not replayed.
fn probe_streams_can_be_replayed(plan: &Arc<dyn ExecutionPlan>) -> bool {
    fn replayed(plan: &Arc<dyn ExecutionPlan>, copy: &Arc<dyn ExecutionPlan>) -> bool {
        let (children, copies) = (plan.children(), copy.children());
        if children.is_empty() {
            return plan.is::<EmptyExec>()
                || plan.is::<PlaceholderRowExec>()
                || (plan.is::<LazyMemoryExec>() && !Arc::ptr_eq(plan, copy))
                || plan.downcast_ref::<DataSourceExec>().is_some_and(|exec| {
                    exec.data_source().is::<MemorySourceConfig>()
                        || exec.data_source().is::<FileScanConfig>()
                });
        }
        !Arc::ptr_eq(plan, copy)
            && children.len() == copies.len()
            && children
                .iter()
                .zip(copies)
                .all(|(child, copy)| replayed(child, copy))
    }
    reset_plan_states(Arc::clone(plan)).is_ok_and(|copy| replayed(plan, &copy))
}

/// Whether values of `data_type` (or of its children) are dictionary encoded
/// with keys of 8 or 16 bits, or run-end encoded with run ends of 16 bits:
/// arrays of those hold only so many distinct values or rows, so that
/// whether gathering rows into one array overflows them depends on which
/// rows are gathered together.
fn has_narrow_encoding(data_type: &DataType) -> bool {
    match data_type {
        DataType::Dictionary(keys, values) => {
            matches!(
                keys.as_ref(),
                DataType::Int8 | DataType::UInt8 | DataType::Int16 | DataType::UInt16
            ) || has_narrow_encoding(values)
        }
        DataType::RunEndEncoded(run_ends, values) => {
            run_ends.data_type() == &DataType::Int16
                || has_narrow_encoding(values.data_type())
        }
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::ListView(field)
        | DataType::LargeListView(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => has_narrow_encoding(field.data_type()),
        DataType::Struct(fields) => fields
            .iter()
            .any(|field| has_narrow_encoding(field.data_type())),
        DataType::Union(fields, _) => fields
            .iter()
            .any(|(_, field)| has_narrow_encoding(field.data_type())),
        _ => false,
    }
}

/// The rows of the build side, in the batches (*segments*) in which the
/// threads of the build pipelines collected them, addressed by a global row
/// index (like DataFusion's `LogicalBatch`): segment `i` holds the rows
/// `offsets[i]..offsets[i + 1]`. Nothing is copied or concatenated.
#[derive(Debug)]
struct SegmentedBatch {
    segments: Vec<RecordBatch>,
    offsets: Vec<usize>,
}

impl SegmentedBatch {
    fn new(batches: Vec<RecordBatch>) -> Self {
        let segments = batches
            .into_iter()
            .filter(|batch| batch.num_rows() > 0)
            .collect::<Vec<_>>();
        let mut offsets = Vec::with_capacity(segments.len() + 1);
        let mut rows = 0;
        offsets.push(rows);
        for segment in &segments {
            rows += segment.num_rows();
            offsets.push(rows);
        }
        Self { segments, offsets }
    }

    fn num_rows(&self) -> usize {
        self.offsets[self.offsets.len() - 1]
    }

    /// The segment of the (existing) row `row`.
    fn segment_of(&self, row: usize) -> usize {
        self.offsets.partition_point(|&offset| offset <= row) - 1
    }
}

/// An atomic bitmap of the build rows that matched, set by all threads of the
/// probe pipelines.
#[derive(Debug)]
struct MatchedRows {
    words: Vec<AtomicU64>,
}

impl MatchedRows {
    fn try_new(num_rows: usize) -> Result<Self> {
        let num_words = num_rows.div_ceil(64);
        let mut words = Vec::new();
        words.try_reserve_exact(num_words).map_err(|e| {
            resources_datafusion_err!("Failed to allocate the matched build rows: {e}")
        })?;
        words.extend((0..num_words).map(|_| AtomicU64::new(0)));
        Ok(Self { words })
    }

    /// The memory of a bitmap of `num_rows` rows.
    fn memory_size(num_rows: usize) -> usize {
        num_rows.div_ceil(64) * size_of::<u64>()
    }

    fn set(&self, row: usize) {
        let bit = 1 << (row % 64);
        let word = &self.words[row / 64];
        if word.load(Ordering::Relaxed) & bit == 0 {
            word.fetch_or(bit, Ordering::Relaxed);
        }
    }

    fn get(&self, row: usize) -> bool {
        self.words[row / 64].load(Ordering::Relaxed) & (1 << (row % 64)) != 0
    }

    /// The first row of `rows` that did not match, if any.
    fn next_unset(&self, rows: std::ops::Range<usize>) -> Option<usize> {
        let mut row = rows.start;
        while row < rows.end {
            let word = self.words[row / 64].load(Ordering::Relaxed);
            // The rows of the word from `row` on that did not match.
            let unset = !word & (u64::MAX << (row % 64));
            if unset != 0 {
                let found = row - row % 64 + unset.trailing_zeros() as usize;
                return (found < rows.end).then_some(found);
            }
            row += 64 - row % 64;
        }
        None
    }
}

/// The build side of a join once it is finalized, shared by all threads.
#[derive(Debug)]
struct BuildSide {
    rows: SegmentedBatch,
    /// The build rows that matched, for the joins that emit build rows at the
    /// end.
    matched: Option<MatchedRows>,
}

/// The global state of a join, shared by all its roles: its global sink
/// state.
struct JoinState {
    /// The build batches of the threads that completed.
    batches: Mutex<Vec<RecordBatch>>,
    /// The build side, once the join is finalized.
    build: OnceLock<Arc<BuildSide>>,
    /// Whether the memory of the build side may exceed the memory pool (see
    /// [`LoopJoin::build_may_exceed_memory_pool`]).
    may_exceed_memory_pool: bool,
    /// The memory of the build side.
    reservation: MemoryReservation,
    /// Whether the streams of the DataFusion probe input were created (see
    /// [`LoopJoin::create_probe_streams`]), or the error of creating them.
    probe_streams: OnceLock<std::result::Result<(), Arc<DataFusionError>>>,
}

impl GlobalSinkState for JoinState {}

impl JoinState {
    /// Account for `bytes` more bytes of the build side.
    fn reserve(&self, bytes: usize) -> Result<()> {
        if self.may_exceed_memory_pool {
            self.reservation.grow(bytes);
            Ok(())
        } else {
            self.reservation.try_grow(bytes)
        }
    }

    fn build(&self, operator: &str) -> Result<Arc<BuildSide>> {
        self.build.get().map(Arc::clone).ok_or_else(|| {
            internal_datafusion_err!("The build side of {operator} is not finalized")
        })
    }
}

/// The state of a thread of a build pipeline: the rows it collected.
#[derive(Debug, Default)]
struct BuildLocalState {
    batches: Vec<RecordBatch>,
}

impl LocalSinkState for BuildLocalState {}

/// The state of a thread of the probe pipeline.
struct ProbeState {
    build: Arc<BuildSide>,
    /// The probe chunk being joined, until all its output was emitted.
    current: Option<ProbeChunk>,
}

impl OperatorState for ProbeState {}

/// A probe chunk being joined.
struct ProbeChunk {
    /// The (logical) rows of the chunk.
    batch: RecordBatch,
    /// The next build row to join the rows with.
    next_build: usize,
    /// For the joins that emit probe rows by whether they matched: whether
    /// each row matched, and how many did not.
    matched: Vec<bool>,
    unmatched: usize,
    /// The pairs that matched, but were not emitted yet: the global index of
    /// the build row and the probe row.
    pending_build: Vec<usize>,
    pending_probe: Vec<u32>,
}

/// The global state of the scan of the final build rows.
struct FinalScanState {
    build: Arc<BuildSide>,
    /// Blocks of at most [`DEFAULT_VECTOR_SIZE`] build rows: `(segment,
    /// first row, rows)`.
    blocks: Vec<(usize, usize, usize)>,
    /// The next block to scan.
    next_block: AtomicUsize,
}

impl GlobalSourceState for FinalScanState {
    fn max_threads(&self) -> usize {
        self.blocks.len().max(1)
    }
}

/// The global state of the join in `ctx`'s execution: the global state of
/// `join` as a sink.
fn join_state(
    ctx: &ExecutionContext,
    join: &dyn PhysicalOperator,
) -> Result<Arc<JoinState>> {
    ctx.global_sink_state(join)
        .and_then(|state| state.downcast_arc::<JoinState>())
        .ok_or_else(|| {
            internal_datafusion_err!("The build side of {} was not executed", join.name())
        })
}

impl<K: LoopJoinKind> PhysicalOperator for LoopJoin<K> {
    fn name(&self) -> &str {
        K::NAME
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.spec.schema)
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.children
    }

    fn description(&self) -> String {
        let spec = &self.spec;
        if K::CROSS {
            return K::NAME.to_string();
        }
        let mut description = format!("{}: join_type={}", K::NAME, spec.join_type);
        if let Some(filter) = &spec.filter {
            let _ = write!(description, ", filter={}", filter.expression());
        }
        if let Some(projection) = &spec.projection {
            let columns = projection
                .iter()
                .map(|&i| format!("{}@{i}", spec.join_schema.field(i).name()))
                .collect::<Vec<_>>()
                .join(", ");
            let _ = write!(description, ", projection=[{columns}]");
        }
        description
    }

    // ---------------------------------------------------------------------
    // Sink: the build side
    // ---------------------------------------------------------------------

    fn is_sink(&self) -> bool {
        true
    }

    fn parallel_sink(&self) -> bool {
        true
    }

    fn get_global_sink_state(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalSinkState>> {
        let reservation =
            MemoryConsumer::new(K::NAME).register(ctx.task_ctx.memory_pool());
        Ok(Arc::new(JoinState {
            batches: Mutex::new(vec![]),
            build: OnceLock::new(),
            may_exceed_memory_pool: self.build_may_exceed_memory_pool(ctx),
            reservation,
            probe_streams: OnceLock::new(),
        }))
    }

    fn get_local_sink_state(
        &self,
        ctx: &ExecutionContext,
        gstate: &dyn GlobalSinkState,
    ) -> Result<Box<dyn LocalSinkState>> {
        let state = downcast_state::<JoinState>(gstate.as_any(), self.name())?;
        // DataFusion creates the streams of its probe side when it executes
        // the join, before it reads anything: the first thread of a build
        // pipeline creates them once its source is created, while the other
        // threads wait, before any of them reads anything, so that the join
        // raises the same errors, also if its build side never completes.
        state
            .probe_streams
            .get_or_init(|| self.create_probe_streams(ctx).map_err(Arc::new))
            .as_ref()
            .map_err(|e| DataFusionError::Shared(Arc::clone(e)))?;
        Ok(Box::new(BuildLocalState::default()))
    }

    fn sink(
        &self,
        _ctx: &ExecutionContext,
        input: &DataChunk,
        gstate: &dyn GlobalSinkState,
        lstate: &mut dyn LocalSinkState,
    ) -> Result<SinkResult> {
        let state = downcast_state::<JoinState>(gstate.as_any(), self.name())?;
        let local =
            downcast_state_mut::<BuildLocalState>(lstate.as_any_mut(), self.name())?;
        let batch = input.compact()?;
        state.reserve(memory_size(&batch))?;
        local.batches.push(batch);
        Ok(SinkResult::NeedMoreInput)
    }

    fn combine(
        &self,
        _ctx: &ExecutionContext,
        gstate: &dyn GlobalSinkState,
        lstate: &mut dyn LocalSinkState,
    ) -> Result<()> {
        let state = downcast_state::<JoinState>(gstate.as_any(), self.name())?;
        let local =
            downcast_state_mut::<BuildLocalState>(lstate.as_any_mut(), self.name())?;
        state.batches.lock().append(&mut local.batches);
        Ok(())
    }

    fn finalize(
        &self,
        ctx: &ExecutionContext,
        gstate: &dyn GlobalSinkState,
    ) -> Result<SinkFinalize> {
        let state = downcast_state::<JoinState>(gstate.as_any(), self.name())?;
        // The output of a canceled join is not needed any more: its build
        // pipelines stopped or did not run, and the pipelines that read its
        // output are skipped.
        if ctx.registry().is_sink_finished(self) {
            return Ok(SinkFinalize::NoOutputPossible);
        }
        let rows = SegmentedBatch::new(std::mem::take(&mut *state.batches.lock()));
        let num_rows = rows.num_rows();
        let matched = if self.spec.emits_final_build_rows() {
            state.reserve(MatchedRows::memory_size(num_rows))?;
            Some(MatchedRows::try_new(num_rows)?)
        } else {
            None
        };
        state
            .build
            .set(Arc::new(BuildSide { rows, matched }))
            .map_err(|_| internal_datafusion_err!("{} was finalized twice", K::NAME))?;
        // Like `CrossJoinExec`, which does not read its probe side if its
        // build side is empty. `NestedLoopJoinExec` reads its whole probe side
        // in any case.
        if K::CROSS && num_rows == 0 {
            return Ok(SinkFinalize::NoOutputPossible);
        }
        Ok(SinkFinalize::Ready)
    }

    // ---------------------------------------------------------------------
    // Operator: the probe side
    // ---------------------------------------------------------------------

    fn parallel_operator(&self) -> bool {
        true
    }

    fn get_operator_state(
        &self,
        ctx: &ExecutionContext,
        _gstate: &dyn GlobalOperatorState,
    ) -> Result<Box<dyn OperatorState>> {
        let build = join_state(ctx, self)?.build(self.name())?;
        Ok(Box::new(ProbeState {
            build,
            current: None,
        }))
    }

    fn execute(
        &self,
        _ctx: &ExecutionContext,
        input: &DataChunk,
        output: &mut DataChunk,
        _gstate: &dyn GlobalOperatorState,
        lstate: &mut dyn OperatorState,
    ) -> Result<OperatorResult> {
        let state = downcast_state_mut::<ProbeState>(lstate.as_any_mut(), self.name())?;
        if state.current.is_none() {
            // A new probe chunk: join its logical rows.
            let batch = input.compact()?;
            let num_rows = batch.num_rows();
            let tracks_probe = self.spec.emits_probe_rows();
            state.current = Some(ProbeChunk {
                batch,
                next_build: 0,
                matched: if tracks_probe {
                    vec![false; num_rows]
                } else {
                    vec![]
                },
                unmatched: num_rows,
                pending_build: vec![],
                pending_probe: vec![],
            });
        }
        if self.probe_next(state, output)? {
            state.current = None;
            Ok(OperatorResult::NeedMoreInput)
        } else {
            Ok(OperatorResult::HaveMoreOutput)
        }
    }

    // ---------------------------------------------------------------------
    // Source: the build rows emitted after probing
    // ---------------------------------------------------------------------

    fn is_source(&self) -> bool {
        self.spec.emits_final_build_rows()
    }

    fn parallel_source(&self) -> bool {
        true
    }

    fn get_global_source_state(
        &self,
        ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalSourceState>> {
        let build = join_state(ctx, self)?.build(self.name())?;
        let blocks = build
            .rows
            .segments
            .iter()
            .enumerate()
            .flat_map(|(segment, batch)| {
                (0..batch.num_rows())
                    .step_by(DEFAULT_VECTOR_SIZE)
                    .map(move |start| {
                        let len = DEFAULT_VECTOR_SIZE.min(batch.num_rows() - start);
                        (segment, start, len)
                    })
            })
            .collect();
        Ok(Arc::new(FinalScanState {
            build,
            blocks,
            next_block: AtomicUsize::new(0),
        }))
    }

    fn get_data(
        &self,
        _ctx: &ExecutionContext,
        gstate: &dyn GlobalSourceState,
        _lstate: &mut dyn LocalSourceState,
    ) -> Result<SourceResult> {
        let global = downcast_state::<FinalScanState>(gstate.as_any(), self.name())?;
        Ok(match self.next_final_rows(global)? {
            Some(chunk) => SourceResult::Chunk(chunk),
            None => SourceResult::Finished,
        })
    }

    // ---------------------------------------------------------------------
    // Pipelines
    // ---------------------------------------------------------------------

    fn build_pipelines(
        &self,
        current: &mut PipelineBuilder,
        meta: &mut MetaPipelineBuilder<'_>,
    ) -> Result<()> {
        let this = meta.current_operator()?;
        // The join probes the build side in the pipeline of its probe side.
        meta.add_operator(current, Arc::clone(&this))?;
        let Some(last_pipeline) = meta.pipelines().last().copied() else {
            return internal_err!("A meta pipeline without pipelines");
        };
        {
            // The build side sinks into the join, in a child meta pipeline.
            let (mut build, mut build_pipeline) =
                meta.create_child_meta_pipeline(current, Arc::clone(&this))?;
            // The probe streams of a join lowered from a DataFusion join are
            // created as soon as the execution starts, in a pipeline that
            // depends on nothing.
            if self.probe_plan.is_some() {
                let init_pipeline =
                    build.create_union_pipeline(&build_pipeline, false)?;
                build.set_source(
                    &init_pipeline,
                    Arc::new(ProbeStreamsInit {
                        schema: Arc::new(Schema::empty()),
                    }),
                )?;
            }
            build.build(&mut build_pipeline, self.build_side())?;
        }
        // The probe side continues the current pipeline.
        meta.build(current, self.probe_side())?;
        // The build rows emitted after probing, in a pipeline that runs once
        // all the pipelines of the probe side completed.
        if self.spec.emits_final_build_rows() {
            meta.create_child_pipeline(current, this, &last_pipeline)?;
        }
        Ok(())
    }
}

/// The source of a pipeline of the build meta pipeline of a [`LoopJoin`]
/// lowered from a DataFusion join, which depends on nothing, so that it runs
/// as soon as the execution starts. It produces no rows: its sink, the join,
/// creates the streams of its DataFusion probe input when it creates the
/// local sink state of the pipeline (see [`LoopJoin::create_probe_streams`]),
/// as DataFusion does when it executes the join, also when the build
/// pipelines wait for pipeline breakers of the build side.
#[derive(Debug)]
struct ProbeStreamsInit {
    schema: SchemaRef,
}

impl PhysicalOperator for ProbeStreamsInit {
    fn name(&self) -> &str {
        "ProbeStreamsInit"
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

    fn get_data(
        &self,
        _ctx: &ExecutionContext,
        _gstate: &dyn GlobalSourceState,
        _lstate: &mut dyn LocalSourceState,
    ) -> Result<SourceResult> {
        Ok(SourceResult::Finished)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::ops::Range;

    use arrow::array::{
        Decimal128Array, DictionaryArray, Float64Array, Int32Array, Int64Array,
        ListArray, StringArray, StringViewArray, TimestampNanosecondArray,
    };
    use arrow::datatypes::{Field, Int32Type, TimeUnit};
    use datafusion_execution::TaskContext;
    use datafusion_expr::Operator;
    use datafusion_physical_expr::PhysicalExpr;
    use datafusion_physical_expr::expressions::{BinaryExpr, col, lit};

    use crate::explain::format_operator_tree;
    use crate::operators::union::test_support::{
        TestSource, UNREGISTERED_STORE, assert_lowered_plan_byte_eq,
        assert_lowered_plan_fails_like_datafusion, count, pending_plan, run,
        unregistered_store_scan, values, values_chunk, values_schema,
    };
    use crate::test_utils::test_session_config;
    use crate::window::test_support::{
        assert_native_parity, assert_native_parity_with_config, lower,
    };

    const ALL_JOIN_TYPES: [JoinType; 10] = [
        JoinType::Inner,
        JoinType::Left,
        JoinType::Right,
        JoinType::Full,
        JoinType::LeftSemi,
        JoinType::RightSemi,
        JoinType::LeftAnti,
        JoinType::RightAnti,
        JoinType::LeftMark,
        JoinType::RightMark,
    ];

    fn task_ctx() -> Arc<TaskContext> {
        Arc::new(TaskContext::default().with_session_config(test_session_config()))
    }

    // ---------------------------------------------------------------------
    // Test data
    // ---------------------------------------------------------------------

    /// The schema of the `side` (`"l"` or `"r"`) input of the plans of the
    /// tests: a unique `id` and columns of many types, with NULLs (and
    /// `-0.0` and `NaN`).
    fn data_schema(side: &str) -> SchemaRef {
        let field = |name: &str, data_type: DataType, nullable: bool| {
            Field::new(format!("{side}{name}"), data_type, nullable)
        };
        Arc::new(Schema::new(vec![
            field("id", DataType::Int64, false),
            field("k", DataType::Int32, true),
            field("f", DataType::Float64, true),
            field("s", DataType::Utf8, true),
            field("v", DataType::Utf8View, true),
            field("d", DataType::Decimal128(12, 2), true),
            field(
                "t",
                DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
                true,
            ),
            field(
                "l",
                DataType::List(Arc::new(Field::new_list_field(DataType::Int32, true))),
                true,
            ),
            field(
                "e",
                DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
                true,
            ),
        ]))
    }

    /// The rows `ids` of the `side` input. `skew` changes the values, so
    /// that the two sides differ.
    fn data_batch(side: &str, ids: Range<i64>, skew: i64) -> RecordBatch {
        let words = ["alpha", "beta", "a string longer than twelve bytes", ""];
        let k = Int32Array::from_iter(
            ids.clone()
                .map(|i| (i % 11 != 3).then(|| ((i * 7 + skew) % 23) as i32 - 5)),
        );
        let f = Float64Array::from_iter(ids.clone().map(|i| match (i + skew) % 13 {
            0 => None,
            1 => Some(-0.0),
            2 => Some(0.0),
            3 => Some(f64::NAN),
            r => Some(f64::from(r as i32 % 5) / 4.0),
        }));
        let s = StringArray::from_iter(
            ids.clone()
                .map(|i| (i % 9 != 5).then(|| words[((i + skew) % 4) as usize])),
        );
        let v = StringViewArray::from_iter(ids.clone().map(|i| {
            (i % 10 != 7).then(|| format!("a fairly long view string {}", (i + skew) % 6))
        }));
        let d = Decimal128Array::from_iter(
            ids.clone()
                .map(|i| (i % 8 != 6).then(|| i128::from((i * 3 + skew) % 7) * 125)),
        )
        .with_precision_and_scale(12, 2)
        .unwrap();
        let t = TimestampNanosecondArray::from_iter(ids.clone().map(|i| {
            (i % 12 != 2)
                .then(|| 1_700_000_000_000_000_000 + (i + skew) % 5 * 1_000_000_007)
        }))
        .with_timezone("UTC");
        let l =
            ListArray::from_iter_primitive::<Int32Type, _, _>(ids.clone().map(|i| {
                (i % 7 != 4).then(|| vec![Some(((i + skew) % 3) as i32), None])
            }));
        let e = ids
            .clone()
            .map(|i| (i % 6 != 1).then(|| words[((i * 5 + skew) % 4) as usize]))
            .collect::<DictionaryArray<Int32Type>>();
        let id = Int64Array::from_iter_values(ids);
        RecordBatch::try_new(
            data_schema(side),
            vec![
                Arc::new(id),
                Arc::new(k),
                Arc::new(f),
                Arc::new(s),
                Arc::new(v),
                Arc::new(d),
                Arc::new(t),
                Arc::new(l),
                Arc::new(e),
            ],
        )
        .unwrap()
    }

    /// A scan of the `side` input with `rows` rows in skewed partitions (one
    /// of them empty).
    fn data_scan(side: &str, rows: i64, skew: i64) -> Arc<dyn ExecutionPlan> {
        let (a, b) = (rows / 2, rows * 3 / 4);
        let partitions = vec![
            vec![data_batch(side, 0..a, skew)],
            vec![],
            vec![
                data_batch(side, a..b, skew),
                data_batch(side, b..rows, skew),
            ],
        ];
        MemorySourceConfig::try_new_exec(&partitions, data_schema(side), None).unwrap()
    }

    /// A join filter over the columns `columns` of the test inputs (e.g.
    /// `["lk", "rk"]`), with the expression `make` builds on the filter's
    /// schema.
    fn join_filter(
        columns: &[&str],
        make: impl Fn(&Schema) -> Arc<dyn PhysicalExpr>,
    ) -> JoinFilter {
        let (left, right) = (data_schema("l"), data_schema("r"));
        let (fields, indices): (Vec<_>, Vec<_>) = columns
            .iter()
            .map(|name| {
                let (schema, side) = match &name[..1] {
                    "l" => (&left, JoinSide::Left),
                    _ => (&right, JoinSide::Right),
                };
                let index = schema.index_of(name).unwrap();
                (schema.field(index).clone(), ColumnIndex { index, side })
            })
            .unzip();
        let schema = Arc::new(Schema::new(fields));
        JoinFilter::new(make(&schema), indices, schema)
    }

    fn compare(left: &str, op: Operator, right: &str) -> JoinFilter {
        join_filter(&[left, right], |schema| {
            Arc::new(BinaryExpr::new(
                col(left, schema).unwrap(),
                op,
                col(right, schema).unwrap(),
            ))
        })
    }

    /// The filters of the tests: integers, floats with `-0.0` and `NaN`,
    /// strings or a sum, a constant and a filter without any column.
    fn filters() -> Vec<Option<JoinFilter>> {
        vec![
            None,
            Some(compare("lk", Operator::Lt, "rk")),
            Some(compare("lf", Operator::LtEq, "rf")),
            Some(join_filter(&["ls", "rs", "lid", "rid"], |schema| {
                let equal = BinaryExpr::new(
                    col("ls", schema).unwrap(),
                    Operator::Eq,
                    col("rs", schema).unwrap(),
                );
                let sum = BinaryExpr::new(
                    col("lid", schema).unwrap(),
                    Operator::Plus,
                    col("rid", schema).unwrap(),
                );
                let large = BinaryExpr::new(Arc::new(sum), Operator::Gt, lit(600_i64));
                Arc::new(BinaryExpr::new(
                    Arc::new(equal),
                    Operator::Or,
                    Arc::new(large),
                ))
            })),
            Some(join_filter(&[], |_| lit(true))),
        ]
    }

    fn nested_loop_plan(
        join_type: JoinType,
        filter: Option<JoinFilter>,
        projection: Option<Vec<usize>>,
        left_rows: i64,
        right_rows: i64,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let left = Arc::new(CoalescePartitionsExec::new(data_scan("l", left_rows, 0)));
        let right = data_scan("r", right_rows, 3);
        Ok(Arc::new(NestedLoopJoinExec::try_new(
            left, right, filter, &join_type, projection,
        )?))
    }

    fn cross_plan(left_rows: i64, right_rows: i64) -> Result<Arc<dyn ExecutionPlan>> {
        let left = Arc::new(CoalescePartitionsExec::new(data_scan("l", left_rows, 0)));
        let right = data_scan("r", right_rows, 3);
        Ok(Arc::new(CrossJoinExec::new(left, right)))
    }

    /// Assert that the plan `make` creates produces the same rows natively
    /// as in DataFusion, with `joins` native nested loop joins and
    /// `crosses` native cross joins.
    async fn assert_plan(
        make: impl Fn() -> Result<Arc<dyn ExecutionPlan>>,
        joins: usize,
        crosses: usize,
    ) -> Arc<dyn PhysicalOperator> {
        let root = assert_lowered_plan_byte_eq(make, lower, task_ctx(), false).await;
        let tree = format_operator_tree(&root);
        assert_eq!(count::<PhysicalNestedLoopJoin>(&root), joins, "{tree}");
        assert_eq!(count::<PhysicalCrossJoin>(&root), crosses, "{tree}");
        root
    }

    // ---------------------------------------------------------------------
    // Join semantics
    // ---------------------------------------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn all_join_types_match_datafusion() {
        for join_type in ALL_JOIN_TYPES {
            for (index, filter) in filters().into_iter().enumerate() {
                for (left_rows, right_rows) in [(150, 400), (0, 400), (150, 0), (1, 1)] {
                    let make = || {
                        nested_loop_plan(
                            join_type,
                            filter.clone(),
                            None,
                            left_rows,
                            right_rows,
                        )
                    };
                    let root =
                        assert_lowered_plan_byte_eq(make, lower, task_ctx(), false).await;
                    assert_eq!(
                        count::<PhysicalNestedLoopJoin>(&root),
                        1,
                        "{join_type} join, filter {index}, {left_rows} x {right_rows} rows"
                    );
                }
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn projections_of_all_join_types() {
        for join_type in ALL_JOIN_TYPES {
            let columns = nested_loop_plan(join_type, None, None, 0, 0)
                .unwrap()
                .schema()
                .fields()
                .len();
            for projection in [vec![], vec![columns - 1, 0], vec![2, 1, columns - 1]] {
                let filter = Some(compare("lk", Operator::GtEq, "rk"));
                let make = || {
                    nested_loop_plan(
                        join_type,
                        filter.clone(),
                        Some(projection.clone()),
                        120,
                        300,
                    )
                };
                assert_plan(make, 1, 0).await;
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn cross_joins_match_datafusion() {
        for (left_rows, right_rows) in [
            (30, 400),
            (0, 400),
            (30, 0),
            (0, 0),
            (1, 3000),
            (3000, 1),
            (5, 5),
        ] {
            assert_plan(|| cross_plan(left_rows, right_rows), 0, 1).await;
        }
    }

    // ---------------------------------------------------------------------
    // SQL
    // ---------------------------------------------------------------------

    /// Assert that `sql` matches DataFusion with `joins` native nested loop
    /// joins and `crosses` native cross joins.
    async fn assert_joins(sql: &str, joins: usize, crosses: usize) {
        let root = assert_native_parity(sql).await;
        let tree = format_operator_tree(&root);
        assert_eq!(
            count::<PhysicalNestedLoopJoin>(&root),
            joins,
            "query: {sql}\nlowered: {tree}"
        );
        assert_eq!(
            count::<PhysicalCrossJoin>(&root),
            crosses,
            "query: {sql}\nlowered: {tree}"
        );
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_join_types() {
        for sql in [
            // Inner, Left (planned as Right), Right (planned as Left), Full.
            "SELECT t1.id, t2.k FROM t1 JOIN t2 ON t1.a < t2.k WHERE t1.id < 20",
            "SELECT t1.id, t2.k, t2.name FROM t1 LEFT JOIN t2 ON t1.a < t2.k WHERE t1.id < 20",
            "SELECT t1.id, t2.k FROM t1 RIGHT JOIN t2 ON t1.a < t2.k AND t1.id < 20",
            "SELECT t1.id, t2.k FROM t1 FULL JOIN t2 ON t1.a < t2.k AND t1.id < 20",
            // Semi and anti joins of both sides.
            "SELECT t2.k FROM t2 WHERE EXISTS (SELECT 1 FROM t1 WHERE t1.a < t2.k AND t1.id < 20)",
            "SELECT t2.k FROM t2 WHERE NOT EXISTS (SELECT 1 FROM t1 WHERE t1.a < t2.k AND t1.id < 20)",
            "SELECT t1.id FROM t1 WHERE EXISTS (SELECT 1 FROM t2 WHERE t1.a < t2.k) AND t1.id < 100",
            "SELECT t1.id FROM t1 WHERE NOT EXISTS (SELECT 1 FROM t2 WHERE t1.a < t2.k) AND t1.id < 100",
            // A mark join.
            "SELECT t1.id FROM t1 WHERE t1.id < 30 AND (EXISTS (SELECT 1 FROM t2 WHERE t1.a < t2.k) OR t1.a = 3)",
            // Filters of other types.
            "SELECT t1.id, t2.k FROM t1 JOIN t2 ON t1.c < CAST(t2.w AS DOUBLE) WHERE t1.id < 50",
            "SELECT t1.id, t2.name FROM t1 JOIN t2 ON t1.d < t2.name OR t1.h IS NULL WHERE t1.id < 40",
            "SELECT t1.id, t2.k FROM t1 JOIN t2 ON t1.a + t2.k > 20 AND t1.e WHERE t1.id < 60",
            // Every column of `t1`.
            "SELECT * FROM t1 JOIN t2 ON t1.b < t2.w WHERE t1.id < 15",
            "SELECT * FROM t2 LEFT JOIN t1 ON t1.b > t2.w AND t1.id < 10",
            // Aggregates over the join, empty sides.
            "SELECT COUNT(*), SUM(t1.a) FROM t1 JOIN t2 ON t1.a < t2.k",
            "SELECT t1.id, t2.k FROM t_empty t1 JOIN t2 ON t1.a < t2.k",
            "SELECT t1.id, t2.k FROM t_empty t1 RIGHT JOIN t2 ON t1.a < t2.k",
            "SELECT t1.id, t2.k FROM t1 LEFT JOIN t2 ON t1.a < t2.k AND t2.k > 1000 WHERE t1.id < 50",
            // A union on the build side.
            "SELECT t1.id, u.k FROM t1 JOIN (SELECT k FROM t2 UNION ALL SELECT a FROM t1 WHERE id < 5) u ON t1.a < u.k WHERE t1.id < 30",
        ] {
            assert_joins(sql, 1, 0).await;
        }
        // Several joins: those on the probe side of another one are left to
        // DataFusion with that probe side.
        for sql in [
            "SELECT a.id, b.k, c.k FROM t1 a JOIN t2 b ON a.a < b.k JOIN t2 c ON b.w > c.w WHERE a.id < 4",
            "SELECT a.k, b.k, c.k FROM t2 a CROSS JOIN t2 b CROSS JOIN t2 c WHERE a.w < -20 AND b.w < 0",
        ] {
            let root = assert_native_parity(sql).await;
            let joins = count::<PhysicalNestedLoopJoin>(&root)
                + count::<PhysicalCrossJoin>(&root);
            assert!(
                joins >= 1,
                "query: {sql}
lowered: {}",
                format_operator_tree(&root)
            );
        }
        for sql in [
            "SELECT a.k, b.k FROM t2 a CROSS JOIN t2 b WHERE a.w < 0",
            "SELECT t1.id, t2.k FROM t1, t2 WHERE t1.id < 3",
            "SELECT * FROM t1 CROSS JOIN t2 WHERE t1.id < 2",
            "SELECT t2.k, COUNT(*), MAX(t1.id + t2.w) FROM t1 CROSS JOIN t2 WHERE t1.a > 40 GROUP BY t2.k",
            "SELECT t1.id, t2.k FROM t_empty t1 CROSS JOIN t2",
            "SELECT a.id, b.id FROM t1 a CROSS JOIN t_empty b",
        ] {
            assert_joins(sql, 0, 1).await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_partitionings() {
        for target_partitions in [1, 3, 16] {
            let config = test_session_config().with_target_partitions(target_partitions);
            for sql in [
                "SELECT t1.id, t2.k FROM t1 LEFT JOIN t2 ON t1.a < t2.k WHERE t1.id < 20",
                "SELECT t1.id, t2.k FROM t1 FULL JOIN t2 ON t1.a < t2.k AND t1.id < 20",
                "SELECT t2.k FROM t2 WHERE NOT EXISTS (SELECT 1 FROM t1 WHERE t1.a < t2.k AND t1.id < 20)",
                "SELECT a.k, b.k FROM t2 a CROSS JOIN t2 b WHERE a.w < 0",
            ] {
                assert_native_parity_with_config(sql, config.clone()).await;
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_ordered_and_limited_results() {
        for sql in [
            "SELECT t1.id, t2.k FROM t1 JOIN t2 ON t1.a < t2.k WHERE t1.id < 20 ORDER BY t1.id, t2.k",
            "SELECT a.k, a.w, b.w FROM t2 a CROSS JOIN t2 b ORDER BY a.w, b.w, a.k LIMIT 77",
        ] {
            assert_native_parity(sql).await;
        }
    }

    // ---------------------------------------------------------------------
    // Execution
    // ---------------------------------------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn output_chunks_have_at_most_2048_rows() -> Result<()> {
        let all_pairs = join_filter(&["lid", "rid"], |schema| {
            Arc::new(BinaryExpr::new(
                col("lid", schema).unwrap(),
                Operator::GtEq,
                Arc::new(BinaryExpr::new(
                    col("rid", schema).unwrap(),
                    Operator::Minus,
                    lit(100_000_i64),
                )),
            ))
        });
        let plans = [
            (cross_plan(3, 5000)?, 15_000),
            (cross_plan(5000, 3)?, 15_000),
            (
                nested_loop_plan(
                    JoinType::Inner,
                    Some(all_pairs.clone()),
                    None,
                    7,
                    3000,
                )?,
                21_000,
            ),
            (
                nested_loop_plan(JoinType::Full, Some(all_pairs), None, 2500, 3)?,
                7500,
            ),
        ];
        for (plan, rows) in plans {
            let root = lower(&plan)?;
            for threads in [1, 4] {
                let batches = run(Arc::clone(&root), threads).await?;
                assert_eq!(
                    batches.iter().map(RecordBatch::num_rows).sum::<usize>(),
                    rows
                );
                assert!(
                    batches
                        .iter()
                        .all(|batch| batch.num_rows() <= DEFAULT_VECTOR_SIZE),
                    "{:?}",
                    batches
                        .iter()
                        .map(RecordBatch::num_rows)
                        .collect::<Vec<_>>()
                );
            }
        }
        Ok(())
    }

    /// A join of `build` and `probe`, sources of the values of
    /// [`values_schema`], on `build.v < probe.v`.
    fn values_join(
        join_type: JoinType,
        build: Vec<DataChunk>,
        probe: Vec<DataChunk>,
    ) -> Result<PhysicalNestedLoopJoin> {
        let schema = values_schema();
        let filter_schema = Arc::new(Schema::new(vec![
            Field::new("l", DataType::Int64, false),
            Field::new("r", DataType::Int64, false),
        ]));
        let filter = JoinFilter::new(
            Arc::new(BinaryExpr::new(
                col("l", &filter_schema)?,
                Operator::Lt,
                col("r", &filter_schema)?,
            )),
            vec![
                ColumnIndex {
                    index: 0,
                    side: JoinSide::Left,
                },
                ColumnIndex {
                    index: 0,
                    side: JoinSide::Right,
                },
            ],
            filter_schema,
        );
        PhysicalNestedLoopJoin::try_new(
            Arc::new(TestSource::with_schema(Arc::clone(&schema), build)),
            Arc::new(TestSource::with_schema(schema, probe)),
            Some(filter),
            join_type,
            None,
        )
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn only_the_selected_rows_of_chunks_are_joined() -> Result<()> {
        use crate::chunk::SelectionVector;
        // Build: the even values of 0..20 (selected); probe: 10..30, of
        // which 15..25 are selected.
        let build = values_chunk(0..20).with_selection(SelectionVector::from(
            (0..20).step_by(2).collect::<Vec<u32>>(),
        ))?;
        let probe = values_chunk(10..30)
            .with_selection(SelectionVector::from((5..15).collect::<Vec<u32>>()))?;
        for (join_type, expected) in [
            // Probe values 15..25 with the even build values below them.
            (JoinType::RightSemi, (15..25).collect::<Vec<i64>>()),
            (JoinType::LeftSemi, (0..20).step_by(2).collect()),
            (JoinType::LeftAnti, vec![]),
            (JoinType::RightAnti, vec![]),
        ] {
            let join = values_join(join_type, vec![build.clone()], vec![probe.clone()])?;
            for threads in [1, 3] {
                let mut output = values(&run(Arc::new(join_ref(&join)?), threads).await?);
                output.sort_unstable();
                assert_eq!(output, expected, "{join_type} join on {threads} threads");
            }
        }
        // All pairs: 10 build values below each of the probe values.
        let join = values_join(JoinType::Inner, vec![build], vec![probe])?;
        let output = run(Arc::new(join), 4).await?;
        let pairs = output.iter().map(RecordBatch::num_rows).sum::<usize>();
        let expected = (15..25)
            .map(|p| (0..20).step_by(2).filter(|&b| b < p).count())
            .sum::<usize>();
        assert_eq!(pairs, expected);
        Ok(())
    }

    /// A copy of `join`, to execute it again.
    fn join_ref(join: &PhysicalNestedLoopJoin) -> Result<PhysicalNestedLoopJoin> {
        PhysicalNestedLoopJoin::try_new(
            Arc::clone(join.build_side()),
            Arc::clone(join.probe_side()),
            join.filter().cloned(),
            join.join_type(),
            join.projection().map(<[usize]>::to_vec),
        )
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn joins_can_be_executed_again() -> Result<()> {
        let plan = nested_loop_plan(
            JoinType::Full,
            Some(compare("lk", Operator::Lt, "rk")),
            None,
            150,
            400,
        )?;
        let root = lower(&plan)?;
        let first = run(Arc::clone(&root), 4).await?;
        let second = run(root, 4).await?;
        let rows = |batches: &[RecordBatch]| {
            batches.iter().map(RecordBatch::num_rows).sum::<usize>()
        };
        assert!(rows(&first) > 0);
        assert_eq!(rows(&first), rows(&second));
        Ok(())
    }

    #[test]
    fn pipelines_of_loop_joins() -> Result<()> {
        let plan = nested_loop_plan(
            JoinType::Left,
            Some(compare("lk", Operator::Lt, "rk")),
            None,
            10,
            10,
        )?;
        let pipelines = crate::explain::format_pipelines(&lower(&plan)?)?;
        assert_eq!(
            pipelines,
            "MetaPipeline 0 (sink: ResultSink) children: [1]\n\
             \x20 Pipeline 0: MemoryScanSource -> PhysicalNestedLoopJoin -> ResultSink depends on meta pipelines [1]\n\
             \x20 Pipeline 3: PhysicalNestedLoopJoin -> ResultSink depends on pipelines [0]\n\
             MetaPipeline 1 (sink: PhysicalNestedLoopJoin)\n\
             \x20 Pipeline 1: MemoryScanSource -> PhysicalNestedLoopJoin\n\
             \x20 Pipeline 2: ProbeStreamsInit -> PhysicalNestedLoopJoin\n"
        );
        // Without final build rows, there is no scan of the build rows.
        let plan = cross_plan(10, 10)?;
        let pipelines = crate::explain::format_pipelines(&lower(&plan)?)?;
        assert!(
            !pipelines.contains(": PhysicalCrossJoin -> ResultSink"),
            "{pipelines}"
        );
        Ok(())
    }

    // ---------------------------------------------------------------------
    // Reading the probe side
    // ---------------------------------------------------------------------

    /// A memory scan of the column `x` (`INT`) with the values `values`,
    /// below the projection `10 / x AS q`.
    fn ten_over(values: Vec<i32>) -> Result<Arc<dyn ExecutionPlan>> {
        let schema = Arc::new(Schema::new(vec![Field::new("x", DataType::Int32, true)]));
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![Arc::new(Int32Array::from(values))],
        )?;
        let scan =
            MemorySourceConfig::try_new_exec(&[vec![batch]], Arc::clone(&schema), None)?;
        let quotient = Arc::new(BinaryExpr::new(
            lit(10_i32),
            Operator::Divide,
            col("x", &schema)?,
        ));
        Ok(Arc::new(
            datafusion_physical_plan::projection::ProjectionExec::try_new(
                vec![(quotient as Arc<dyn PhysicalExpr>, "q".to_string())],
                scan,
            )?,
        ))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn nested_loop_joins_read_their_probe_side_without_build_rows() {
        // Like DataFusion, also without build rows, a nested loop join reads
        // its probe side (and fails on `10 / 0`), and a cross join does not.
        for join_type in ALL_JOIN_TYPES {
            let make = || {
                let left = Arc::new(CoalescePartitionsExec::new(data_scan("l", 0, 0)));
                Ok(Arc::new(NestedLoopJoinExec::try_new(
                    left,
                    ten_over(vec![1, 0])?,
                    None,
                    &join_type,
                    None,
                )?) as Arc<dyn ExecutionPlan>)
            };
            let root = assert_lowered_plan_fails_like_datafusion(
                make,
                lower,
                test_session_config(),
                "Divide by zero",
            )
            .await;
            assert_eq!(count::<PhysicalNestedLoopJoin>(&root), 1, "{join_type}");
        }
        let make = || {
            let left = Arc::new(CoalescePartitionsExec::new(data_scan("l", 0, 0)));
            Ok(Arc::new(CrossJoinExec::new(left, ten_over(vec![1, 0])?))
                as Arc<dyn ExecutionPlan>)
        };
        assert_plan(make, 0, 1).await;
        // With build rows, both fail.
        let make = || {
            let left = Arc::new(CoalescePartitionsExec::new(data_scan("l", 3, 0)));
            Ok(Arc::new(CrossJoinExec::new(left, ten_over(vec![1, 0])?))
                as Arc<dyn ExecutionPlan>)
        };
        let root = assert_lowered_plan_fails_like_datafusion(
            make,
            lower,
            test_session_config(),
            "Divide by zero",
        )
        .await;
        assert_eq!(count::<PhysicalCrossJoin>(&root), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn probe_streams_are_created_without_build_rows() {
        // DataFusion creates the streams of its probe side when it executes
        // the join, also when it then does not read them: a scan of an object
        // store that is not registered fails, so does the native join.
        let make = || {
            let left = Arc::new(CoalescePartitionsExec::new(data_scan("l", 0, 0)));
            let right = unregistered_store_scan(data_schema("r"))?;
            Ok(Arc::new(CrossJoinExec::new(left, right)) as Arc<dyn ExecutionPlan>)
        };
        let root = assert_lowered_plan_fails_like_datafusion(
            make,
            lower,
            test_session_config(),
            UNREGISTERED_STORE,
        )
        .await;
        assert_eq!(count::<PhysicalCrossJoin>(&root), 1);
        let make = || {
            let left = Arc::new(CoalescePartitionsExec::new(data_scan("l", 0, 0)));
            let right = unregistered_store_scan(data_schema("r"))?;
            Ok(Arc::new(NestedLoopJoinExec::try_new(
                left,
                right,
                None,
                &JoinType::Inner,
                None,
            )?) as Arc<dyn ExecutionPlan>)
        };
        let root = assert_lowered_plan_fails_like_datafusion(
            make,
            lower,
            test_session_config(),
            UNREGISTERED_STORE,
        )
        .await;
        assert_eq!(count::<PhysicalNestedLoopJoin>(&root), 1);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn probe_sides_with_pipeline_breakers_are_bridged() -> Result<()> {
        // The probe side is a cross join, a pipeline breaker: DataFusion reads
        // it only once the build side is collected, so it is left to it.
        let inner = cross_plan(3, 40)?;
        let left = Arc::new(CoalescePartitionsExec::new(data_scan("l", 10, 0)));
        let plan: Arc<dyn ExecutionPlan> = Arc::new(CrossJoinExec::new(left, inner));
        let root = lower(&plan)?;
        let join = root.downcast_ref::<PhysicalCrossJoin>().unwrap();
        assert!(
            join.probe_side().is::<BridgeOperator>(),
            "{}",
            format_operator_tree(&root)
        );
        assert_plan(
            || {
                let left = Arc::new(CoalescePartitionsExec::new(data_scan("l", 10, 0)));
                Ok(Arc::new(CrossJoinExec::new(left, cross_plan(3, 40)?))
                    as Arc<dyn ExecutionPlan>)
            },
            0,
            1,
        )
        .await;
        // The build side stays native.
        let make = || {
            let left = Arc::new(CoalescePartitionsExec::new(cross_plan(3, 4)?));
            let right = data_scan("r", 10, 0);
            Ok(Arc::new(CrossJoinExec::new(left, right)) as Arc<dyn ExecutionPlan>)
        };
        assert_plan(make, 0, 2).await;
        Ok(())
    }

    // ---------------------------------------------------------------------
    // Lowering
    // ---------------------------------------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn joins_that_differ_from_datafusion_are_left_to_it() -> Result<()> {
        // A filter that can fail on some pairs (`lk / rk`).
        let fallible = join_filter(&["lk", "rk"], |schema| {
            Arc::new(BinaryExpr::new(
                Arc::new(BinaryExpr::new(
                    col("lk", schema).unwrap(),
                    Operator::Divide,
                    col("rk", schema).unwrap(),
                )),
                Operator::Gt,
                lit(0_i32),
            ))
        });
        let plan =
            nested_loop_plan(JoinType::RightSemi, Some(fallible.clone()), None, 10, 10)?;
        assert!(PhysicalNestedLoopJoin::try_lower(&plan, lower)?.is_none());
        // Dictionaries with narrow keys.
        let schema = Arc::new(Schema::new(vec![Field::new(
            "n",
            DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
            true,
        )]));
        let narrow = MemorySourceConfig::try_new_exec(&[vec![]], schema, None)?;
        let plan: Arc<dyn ExecutionPlan> = Arc::new(CrossJoinExec::new(
            Arc::new(CoalescePartitionsExec::new(data_scan("l", 3, 0))),
            narrow,
        ));
        assert!(PhysicalCrossJoin::try_lower(&plan, lower)?.is_none());
        // A left input of several partitions, which DataFusion fails to
        // execute, and a right input without partitions, of which DataFusion
        // executes nothing.
        let plan: Arc<dyn ExecutionPlan> = Arc::new(CrossJoinExec::new(
            data_scan("l", 3, 0),
            data_scan("r", 3, 0),
        ));
        assert!(PhysicalCrossJoin::try_lower(&plan, lower)?.is_none());
        let empty = MemorySourceConfig::try_new_exec(&[], data_schema("r"), None)?;
        let plan: Arc<dyn ExecutionPlan> = Arc::new(NestedLoopJoinExec::try_new(
            Arc::new(CoalescePartitionsExec::new(data_scan("l", 3, 0))),
            empty,
            None,
            &JoinType::Left,
            None,
        )?);
        assert!(PhysicalNestedLoopJoin::try_lower(&plan, lower)?.is_none());
        // A probe input whose streams cannot be created on a copy: a cross
        // join is left to DataFusion, a nested loop join does not create them.
        let pending = pending_plan(data_schema("r"), 2)?;
        let plan: Arc<dyn ExecutionPlan> = Arc::new(CrossJoinExec::new(
            Arc::new(CoalescePartitionsExec::new(data_scan("l", 3, 0))),
            Arc::clone(&pending),
        ));
        assert!(PhysicalCrossJoin::try_lower(&plan, lower)?.is_none());
        let plan: Arc<dyn ExecutionPlan> = Arc::new(NestedLoopJoinExec::try_new(
            Arc::new(CoalescePartitionsExec::new(data_scan("l", 3, 0))),
            pending,
            None,
            &JoinType::Left,
            None,
        )?);
        let join = PhysicalNestedLoopJoin::try_lower(&plan, lower)?.unwrap();
        assert!(join.probe_plan.is_none());
        // Such joins behave like DataFusion's (bridged): here, both divide by
        // zero.
        let make =
            || nested_loop_plan(JoinType::Full, Some(fallible.clone()), None, 20, 30);
        let root = assert_lowered_plan_fails_like_datafusion(
            make,
            lower,
            test_session_config(),
            "Divide by zero",
        )
        .await;
        assert_eq!(count::<PhysicalNestedLoopJoin>(&root), 0);
        Ok(())
    }

    // ---------------------------------------------------------------------
    // Memory and limits
    // ---------------------------------------------------------------------

    /// A task context with a memory pool of `limit` bytes, which can spill
    /// (to the temporary directory of the OS) if `spill`.
    fn limited_context(limit: usize, spill: bool) -> Result<Arc<TaskContext>> {
        use datafusion_execution::disk_manager::{DiskManagerBuilder, DiskManagerMode};
        use datafusion_execution::runtime_env::RuntimeEnvBuilder;
        let mut runtime = RuntimeEnvBuilder::new().with_memory_limit(limit, 1.0);
        if !spill {
            runtime = runtime.with_disk_manager_builder(
                DiskManagerBuilder::default().with_mode(DiskManagerMode::Disabled),
            );
        }
        Ok(Arc::new(
            TaskContext::default()
                .with_session_config(test_session_config())
                .with_runtime(runtime.build_arc()?),
        ))
    }

    /// Assert that the plan `make` creates fails with an error containing
    /// `message` in DataFusion and natively, both executed with the task
    /// context `context` creates.
    async fn assert_both_fail(
        make: impl Fn() -> Result<Arc<dyn ExecutionPlan>>,
        context: impl Fn() -> Result<Arc<TaskContext>>,
        message: &str,
    ) -> Result<()> {
        use datafusion_physical_plan::collect;
        let error = collect(make()?, context()?)
            .await
            .expect_err("DataFusion fails");
        assert!(error.to_string().contains(message), "DataFusion: {error}");
        let plan = make()?;
        let root = lower(&plan)?;
        let exec = Arc::new(crate::bridge::PipelineRootExec::try_new(root, plan)?);
        let error = collect(exec, context()?)
            .await
            .expect_err("the native join fails");
        assert!(error.to_string().contains(message), "native: {error}");
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn build_sides_are_charged_to_the_memory_pool() -> Result<()> {
        // A cross join does not spill: in a small pool, both fail.
        let cross = || cross_plan(2000, 3);
        assert_both_fail(
            cross,
            || limited_context(64 * 1024, true),
            "Resources exhausted",
        )
        .await?;
        // DataFusion's nested loop join spills its build side when it can:
        // both succeed (the native join keeps its build side in memory).
        let filter = Some(compare("lk", Operator::Lt, "rk"));
        let join = || nested_loop_plan(JoinType::Left, filter.clone(), None, 2000, 30);
        assert_lowered_plan_byte_eq(
            join,
            lower,
            limited_context(64 * 1024, true)?,
            false,
        )
        .await;
        // Without temporary files, it cannot: both fail.
        assert_both_fail(
            join,
            || limited_context(64 * 1024, false),
            "Resources exhausted",
        )
        .await?;
        // With enough memory, both succeed.
        assert_lowered_plan_byte_eq(
            cross,
            lower,
            limited_context(64 * 1024 * 1024, false)?,
            false,
        )
        .await;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_limit_above_the_join_stops_it() -> Result<()> {
        use datafusion_physical_plan::collect;
        use datafusion_physical_plan::limit::GlobalLimitExec;
        let plan: Arc<dyn ExecutionPlan> = Arc::new(GlobalLimitExec::new(
            Arc::new(CoalescePartitionsExec::new(cross_plan(40, 20_000)?)),
            0,
            Some(10),
        ));
        let root = lower(&plan)?;
        assert_eq!(count::<PhysicalCrossJoin>(&root), 1);
        let exec = Arc::new(crate::bridge::PipelineRootExec::try_new(root, plan)?);
        let batches = collect(Arc::clone(&exec) as _, task_ctx()).await?;
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 10);
        let metrics = exec.execution_metrics().expect("executed");
        let (_, join) = metrics
            .iter()
            .find(|(operator, _)| operator.name() == "PhysicalCrossJoin")
            .expect("the metrics of the join");
        // Far fewer than the 800,000 rows of the join.
        assert!(join.output_rows() < 100_000, "{}", join.output_rows());
        Ok(())
    }
}
