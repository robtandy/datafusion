// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! `PhysicalHashJoin`: parallel shared-build hash join.
//!
//! [`PhysicalHashJoin`] replaces DataFusion's `HashJoinExec`, in both of its
//! partition modes, with the semantics of `HashJoinExec`: its join types,
//! null equality, residual filter, `NOT IN` null awareness and embedded
//! projection, and exactly its output schema. Like `HashJoinExec`, it builds
//! the hash table on its first input (the *left* side, DataFusion's build
//! side) and probes it with its second input (the *right* side). The names
//! of the join types are DataFusion's: e.g. a `Left` join emits the build
//! rows without a match (DuckDB, which builds on the right side, calls that
//! a `RIGHT` join).
//!
//! # Pipelines
//!
//! Like DuckDB's `PhysicalHashJoin`, the join is a sink, an operator and a
//! source:
//!
//! ```text
//! MetaPipeline 0 (sink: ...) children: [1]
//!   Pipeline 0: probe side -> PhysicalHashJoin -> ... depends on meta pipelines [1]
//!   Pipeline 3: PhysicalHashJoin -> ... depends on pipelines [0]
//! MetaPipeline 1 (sink: PhysicalHashJoin)
//!   Pipeline 1: build side -> PhysicalHashJoin
//!   Pipeline 2: JoinHashTableBuild -> PhysicalHashJoin depends on pipelines [1]
//! ```
//!
//! * **Build** (the join as sink, pipeline 1): every thread evaluates and
//!   hashes the join keys of its build rows and scatters them, with the
//!   build columns the join needs, into a thread-local
//!   [`PartitionedTupleData`]; `combine` moves its blocks into the global
//!   partitions without copying any row (see [`JoinHashTable`]).
//! * **Parallel finalize** (pipeline 2): the threads of a
//!   `JoinHashTableBuild` pipeline build the pointer tables of the radix
//!   partitions in parallel, one partition at a time; `finalize` assembles
//!   the shared, read-only [`JoinHashTable`]. A small build side gets a single
//!   table, and every thread thousands of rows (see
//!   [`JoinHashTableBuilder::max_threads`]); a single integer key whose range
//!   the build computes is indexed without hashing (a dense index for a narrow
//!   range, DuckDB's perfect hash join, otherwise a filter of the range of the
//!   keys, see [`JoinHashTableBuilder::with_key_range`]). Like DataFusion, it
//!   evaluates the build keys on an empty build side. A canceled join (whose
//!   output is not needed any more) does neither, nor builds its table. If the
//!   join cannot produce any output (e.g. an inner join with an empty build
//!   side), `finalize` returns [`SinkFinalize::NoOutputPossible`], so that
//!   the probe side is not executed, as DataFusion does not read it either.
//!   DataFusion still creates the streams of its probe side, so a join
//!   lowered from a `HashJoinExec` creates the DataFusion streams of that
//!   probe input too, without polling them, to raise the same errors (e.g.
//!   of a scan of an object store that is not registered), unless creating
//!   them cannot fail (e.g. of scans of memory or of local files, see
//!   [`PhysicalHashJoin::create_probe_streams`]).
//! * **Probe** (the join as operator, pipeline 0): all threads of the probe
//!   pipeline pull chunks directly from the probe side and probe the shared
//!   table concurrently: neither side is shuffled
//!   ([`PhysicalHashJoin::try_lower`] removes the `RepartitionExec`s and
//!   `CoalescePartitionsExec`s that DataFusion puts below the join). The
//!   output keeps the order of the probe side; a probe chunk whose matches
//!   do not fit in a chunk of [`DEFAULT_VECTOR_SIZE`] rows is continued with
//!   [`OperatorResult::HaveMoreOutput`]. The probe keys of a chunk with a
//!   selection are taken from its selected rows, but its other columns are not
//!   compacted: only the probe rows of the output are taken from them, and the
//!   `RightSemi`, `RightAnti` and `RightMark` joins (whose output has no build
//!   columns) output the columns of the probe chunk themselves, with a
//!   selection of the probe rows they emit (see
//!   [`PhysicalHashJoin::start_chunk`]).
//! * **Final build rows** (the join as source, pipeline 3): the `Left`,
//!   `Full`, `LeftSemi`, `LeftAnti` and `LeftMark` joins track the build
//!   rows that found a match in an atomic bitset, and once the probe
//!   pipeline completed, all threads scan the build rows in parallel to emit
//!   the unmatched ones (the matched ones for `LeftSemi`, all of them with
//!   their mark for `LeftMark`).
//!
//! The probe side of a join may have pipeline breakers of its own, e.g. the
//! build side of a join below (as in the joins of TPC-H, which DataFusion
//! plans as chains of joins whose probe side is the next join). DataFusion
//! reads the probe side of a join only once its build side is collected, so
//! the pipelines of these breakers wait for the build side of the join
//! (they depend on its meta pipeline, see
//! [`MetaPipelineBuilder::build_with_dependency`]): if the join cannot
//! produce output, or is canceled (e.g. by a limit above) before its build
//! side completes, they are canceled before they start. For
//! `t1 JOIN (t2 JOIN t3 ON ...) ON ...`:
//!
//! ```text
//! MetaPipeline 0 (sink: ...) children: [1, 2]
//!   Pipeline 0: Scan(t3) -> PhysicalHashJoin -> PhysicalHashJoin -> ... depends on meta pipelines [1, 2]
//! MetaPipeline 1 (sink: PhysicalHashJoin)
//!   Pipeline 1: Scan(t1) -> PhysicalHashJoin
//!   Pipeline 2: JoinHashTableBuild -> PhysicalHashJoin depends on pipelines [1]
//! MetaPipeline 2 (sink: PhysicalHashJoin)
//!   Pipeline 3: Scan(t2) -> PhysicalHashJoin depends on meta pipelines [1]
//!   Pipeline 4: JoinHashTableBuild -> PhysicalHashJoin depends on pipelines [3] depends on meta pipelines [1]
//! ```
//!
//! # Dynamic filters
//!
//! DataFusion pushes a dynamic filter of the probe keys of a join into the
//! nodes of its probe side that can prune rows with it (e.g. Parquet scans,
//! which skip whole files and row groups), and sets it once the build side is
//! collected (see `HashJoinExec::dynamic_expressions_produced`). A
//! [`PhysicalHashJoin`] lowered from such a join sets the filter to the bounds
//! of its build keys (`key >= min AND key <= max`, or `false` if no build row
//! can match) when its build side is finalized, and marks it complete (see
//! [`PhysicalHashJoin::dynamic_filter_predicate`]): DataFusion's filter also
//! checks the membership of the keys (an `IN` list or a hash table lookup),
//! so the probe side of a native join may see more rows than DataFusion's,
//! which the join drops (they cannot match). The result is the same as long as
//! no expression of the probe side can fail on them, which
//! [`PhysicalHashJoin::try_lower`] checks.
//!
//! # Semantics
//!
//! Join keys are equal with DataFusion's join key equality (see
//! [`RowMatcher`](crate::tuple_data::RowMatcher)): `-0.0 = +0.0`, `NaN`s are
//! equal to themselves, and NULLs are equal only with
//! [`NullEquality::NullEqualsNull`] (`IS NOT DISTINCT FROM`). The residual
//! filter is evaluated, like in DataFusion, on the pairs of rows whose keys
//! are equal. A null-aware join (`NOT IN`) follows SQL's three-valued logic:
//!
//! * `LeftAnti` (`build.key NOT IN (probe keys)`) emits the unmatched build
//!   rows, but none if a probe key is NULL, and no row with a NULL key
//!   unless the probe side is empty;
//! * `LeftMark` marks every build row with `build.key IN (probe keys)`:
//!   `true` if it matched, otherwise NULL if a probe key is NULL or (with a
//!   non-empty probe side) its own key is NULL, and `false` otherwise;
//! * `RightAnti` (`probe.key NOT IN (build keys)`) emits the unmatched probe
//!   rows, but none if a build key is NULL, and no row with a NULL key unless
//!   the build side is empty.
//!
//! [`PhysicalHashJoin::try_lower`] lowers the null-aware joins of a single
//! `NOT IN` key without filter (the correlated and multi-column forms are
//! left to DataFusion).

use std::any::Any;
use std::fmt::Write;
use std::ops::Range;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};

use arrow::array::{
    Array, ArrayRef, BooleanArray, RecordBatch, RecordBatchOptions, UInt32Array,
    new_null_array,
};
use arrow::buffer::NullBuffer;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow_select::take::take;
use datafusion_common::ScalarValue;
use datafusion_common::cast::as_boolean_array;
use datafusion_common::tree_node::{
    Transformed, TransformedResult, TreeNode, TreeNodeRecursion,
};
use datafusion_common::{
    DataFusionError, JoinSide, JoinType, NullEquality, Result, internal_datafusion_err,
    internal_err, not_impl_err, plan_err,
};
use datafusion_datasource::file_scan_config::FileScanConfig;
use datafusion_datasource::memory::MemorySourceConfig;
use datafusion_datasource::source::DataSourceExec;
use datafusion_datasource_csv::source::CsvSource;
use datafusion_datasource_parquet::source::ParquetSource;
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_execution::object_store::ObjectStoreUrl;
use datafusion_expr::Operator;
use datafusion_functions_aggregate_common::min_max::{max_batch, min_batch};
use datafusion_physical_expr::expressions::{
    BinaryExpr, CaseExpr, CastExpr, Column, DynamicFilterPhysicalExpr, InListExpr,
    IsNotNullExpr, IsNullExpr, Literal, NotExpr, TryCastExpr, lit,
};
use datafusion_physical_expr::utils::conjunction;
use datafusion_physical_expr::{PhysicalExpr, PhysicalExprRef};
use datafusion_physical_expr_common::utils::evaluate_expressions_to_arrays;
use datafusion_physical_plan::aggregates::{AggregateExec, AggregateMode};
use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion_physical_plan::coop::CooperativeExec;
use datafusion_physical_plan::empty::EmptyExec;
use datafusion_physical_plan::execution_plan::reset_plan_states;
use datafusion_physical_plan::filter::FilterExec;
use datafusion_physical_plan::joins::utils::{
    ColumnIndex, JoinFilter, build_join_schema,
};
use datafusion_physical_plan::joins::{
    CrossJoinExec, HashJoinExec, NestedLoopJoinExec, PartitionMode,
};
use datafusion_physical_plan::limit::{GlobalLimitExec, LocalLimitExec};
use datafusion_physical_plan::memory::LazyMemoryExec;
use datafusion_physical_plan::placeholder_row::PlaceholderRowExec;
use datafusion_physical_plan::projection::ProjectionExec;
use datafusion_physical_plan::repartition::RepartitionExec;
use datafusion_physical_plan::sorts::sort::SortExec;
use datafusion_physical_plan::sorts::sort_preserving_merge::SortPreservingMergeExec;
use datafusion_physical_plan::union::{InterleaveExec, UnionExec};
use datafusion_physical_plan::{
    ExecutionPlan, ExecutionPlanProperties, Partitioning, displayable,
};
use parking_lot::Mutex;

use crate::chunk::{DEFAULT_VECTOR_SIZE, DataChunk, Gather, SelectionVector};
use crate::join::hash_table::{
    ChainScan, JoinHashTable, JoinHashTableBuilder, KeyRange, SMALL_BUILD_ROWS,
    hash_join_keys, radix_bits_for_threads,
};
use crate::operator::{
    ExecutionContext, GlobalOperatorState, GlobalSinkState, GlobalSourceState,
    LocalSinkState, LocalSourceState, OperatorResult, OperatorState, PhysicalOperator,
    SinkFinalize, SinkResult, SourceResult, TaskInterrupt, downcast_state,
    downcast_state_mut,
};
use crate::operators::adaptive_filter::is_infallible;
use crate::pipeline::{MetaPipelineBuilder, PipelineBuilder};
use crate::tuple_data::layout::has_run_end_encoded;
use crate::tuple_data::{PartitionedTupleData, RowLocation, TupleDataLayout};

/// A hash join that builds a shared hash table on its left input and probes
/// it with its right input, with the semantics of DataFusion's
/// `HashJoinExec`.
///
/// See the [module documentation](self) for its pipelines and semantics.
#[derive(Debug)]
pub struct PhysicalHashJoin {
    /// The build side (the left input) and the probe side (the right input).
    children: Vec<Arc<dyn PhysicalOperator>>,
    spec: JoinSpec,
    /// The probe input of the `HashJoinExec` the join was lowered from, whose
    /// streams `HashJoinExec::execute` creates right away, before it reads
    /// its build side (and also if it then does not read its probe side): the
    /// join creates them too (see [`Self::create_probe_streams`]), to raise
    /// the same errors.
    probe_plan: Option<Arc<dyn ExecutionPlan>>,
    /// Whether creating the streams of `probe_plan` cannot fail (see
    /// [`probe_streams_cannot_fail`]), if the local file system is registered
    /// in the runtime (which it is by default): the join then creates none.
    probe_streams_cannot_fail: bool,
    /// The dynamic filters of the `HashJoinExec` the join was lowered from
    /// (see `ExecutionPlan::dynamic_expressions_produced`), which nodes of
    /// its probe side read: the join sets them to the bounds of its build keys
    /// (see [`Self::dynamic_filter_predicate`]), and marks them complete,
    /// when its build side is finalized.
    dynamic_filters: Vec<Arc<DynamicFilterPhysicalExpr>>,
}

/// Everything about a join that does not depend on its inputs, only on
/// their schemas.
#[derive(Debug)]
struct JoinSpec {
    join_type: JoinType,
    /// The join keys: `(build key, probe key)`.
    on: Vec<(PhysicalExprRef, PhysicalExprRef)>,
    filter: Option<JoinFilter>,
    null_equality: NullEquality,
    null_aware: bool,
    projection: Option<Vec<usize>>,
    /// The schema of the join without its projection.
    join_schema: SchemaRef,
    /// The output schema.
    schema: SchemaRef,
    /// The input column of every output column (or the mark column).
    column_indices: Vec<ColumnIndex>,
    /// The layout of the build rows: the build keys, then the payload
    /// columns, plus the hash of the keys.
    layout: Arc<TupleDataLayout>,
    /// The schema of the probe keys.
    key_schema: SchemaRef,
    /// The build columns stored after the keys, in this order.
    payload: Vec<usize>,
    /// For every column of the build side, its column in the layout, if the
    /// join needs it: a key that is that column, or a payload column.
    build_columns: Vec<Option<usize>>,
    /// Whether DataFusion evaluates the build keys only once its whole build
    /// side is collected, when it builds its hash table. It evaluates them on
    /// every build batch as it collects them only to compute their bounds:
    /// for a dynamic filter (see [`PhysicalHashJoin::try_lower`], which
    /// clears this for the joins that have one), or for a single key of an
    /// integer type, a candidate for its perfect hash join (`ArrayMap`).
    defers_build_key_errors: bool,
    /// Whether the join has a single key of an integer type (see
    /// [`KeyRange::supports`]), whose range the build computes, so that the
    /// [`JoinHashTable`] indexes the keys without hashing them (see
    /// [`JoinHashTableBuilder::with_key_range`]).
    tracks_key_range: bool,
    /// For every key, whether the build computes its bounds (its smallest and
    /// largest value), for the dynamic filters of the join (see
    /// [`PhysicalHashJoin::dynamic_filter_predicate`]): the keys whose values
    /// are ordered like they are equal (see [`has_exact_bounds`]) of a join
    /// with dynamic filters.
    bounded_keys: Vec<bool>,
    /// The probe columns of the probe keys, if all of them are columns: the
    /// join then only compacts these columns of a probe chunk with a selection
    /// (see [`PhysicalHashJoin::start_chunk`]).
    probe_key_columns: Option<Vec<usize>>,
    /// Whether the join compacts the probe chunks with a selection, also if
    /// its probe keys are columns: if it outputs (or filters on) probe columns
    /// with run-end encoded values, which Arrow's `take` encodes again (runs
    /// of equal values are merged), so that taking the rows of the compacted
    /// chunk, as DataFusion takes the rows of its filtered batches, can differ
    /// from taking them from the chunk directly.
    compacts_probe_chunks: bool,
}

impl JoinSpec {
    #[expect(clippy::too_many_arguments)]
    fn try_new(
        build_schema: &SchemaRef,
        probe_schema: &SchemaRef,
        on: Vec<(PhysicalExprRef, PhysicalExprRef)>,
        filter: Option<JoinFilter>,
        join_type: JoinType,
        projection: Option<Vec<usize>>,
        null_equality: NullEquality,
        null_aware: bool,
    ) -> Result<Self> {
        if on.is_empty() {
            return plan_err!("PhysicalHashJoin requires at least one join key");
        }
        if null_aware
            && !matches!(
                join_type,
                JoinType::LeftAnti | JoinType::LeftMark | JoinType::RightAnti
            )
        {
            return plan_err!(
                "A null-aware PhysicalHashJoin must be a LeftAnti, LeftMark or RightAnti join, got {join_type}"
            );
        }
        // The `NOT IN` of a single key without filter. With several keys (a
        // multi-column `NOT IN`, or correlation keys) or a filter (a
        // correlated `NOT IN`), whether a NULL makes `IN` UNKNOWN depends on
        // the pairs of rows it is compared with, which the join does not
        // implement; nor does it implement `NOT IN` with NULLs that are equal.
        if null_aware
            && (on.len() != 1
                || filter.is_some()
                || null_equality != NullEquality::NullEqualsNothing)
        {
            return not_impl_err!(
                "PhysicalHashJoin implements the null-aware join of a single key, without filter and with NullEqualsNothing, got {} keys, {} filter and {null_equality:?}",
                on.len(),
                if filter.is_some() { "a" } else { "no" }
            );
        }
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

        let mut key_fields = Vec::with_capacity(on.len());
        for (i, (build_key, probe_key)) in on.iter().enumerate() {
            let build_type = build_key.data_type(build_schema)?;
            let probe_type = probe_key.data_type(probe_schema)?;
            if build_type != probe_type {
                return plan_err!(
                    "PhysicalHashJoin requires join keys of the same type, key {i} has types {build_type} and {probe_type}"
                );
            }
            key_fields.push(Field::new(format!("key{i}"), build_type, true));
        }
        let num_keys = key_fields.len();

        // The filter reads columns of the inputs, one per field of its
        // intermediate batch.
        if let Some(filter) = &filter {
            let columns = filter.column_indices();
            if columns.len() != filter.schema().fields().len() {
                return plan_err!(
                    "The filter of a PhysicalHashJoin reads {} columns, but its schema has {} fields",
                    columns.len(),
                    filter.schema().fields().len()
                );
            }
            if let Some(column) = columns.iter().find(|column| match column.side {
                JoinSide::Left => column.index >= build_schema.fields().len(),
                JoinSide::Right => column.index >= probe_schema.fields().len(),
                JoinSide::None => true,
            }) {
                return plan_err!(
                    "The filter of a PhysicalHashJoin reads column {} of side {:?}, which its input does not have",
                    column.index,
                    column.side
                );
            }
        }

        // The build columns the join needs: those it outputs and those its
        // filter reads.
        let mut needed = column_indices
            .iter()
            .chain(filter.iter().flat_map(|filter| filter.column_indices()))
            .filter(|column| column.side == JoinSide::Left)
            .map(|column| column.index)
            .collect::<Vec<_>>();
        needed.sort_unstable();
        needed.dedup();
        let num_build_columns = build_schema.fields().len();
        if let Some(&column) = needed.iter().find(|&&c| c >= num_build_columns) {
            return internal_err!(
                "Build column {column} out of bounds for a build side with {num_build_columns} columns"
            );
        }
        // A key that is a column of the build side stores that column.
        let mut build_columns = vec![None; num_build_columns];
        for (key, (build_key, _)) in on.iter().enumerate() {
            if let Some(column) = build_key.downcast_ref::<Column>()
                && column.index() < num_build_columns
                && build_columns[column.index()].is_none()
            {
                build_columns[column.index()] = Some(key);
            }
        }
        let payload = needed
            .into_iter()
            .filter(|&column| build_columns[column].is_none())
            .collect::<Vec<_>>();
        let mut fields = key_fields.clone();
        for (position, &column) in payload.iter().enumerate() {
            build_columns[column] = Some(num_keys + position);
            fields.push(build_schema.field(column).clone());
        }
        let layout = Arc::new(TupleDataLayout::try_new(
            Arc::new(Schema::new(fields)),
            true,
        )?);
        let defers_build_key_errors = match on.as_slice() {
            [(key, _)] => !matches!(
                key.data_type(build_schema)?,
                DataType::Int8
                    | DataType::Int16
                    | DataType::Int32
                    | DataType::Int64
                    | DataType::UInt8
                    | DataType::UInt16
                    | DataType::UInt32
                    | DataType::UInt64
            ),
            _ => true,
        };
        let tracks_key_range = match on.as_slice() {
            [(key, _)] => KeyRange::supports(&key.data_type(build_schema)?),
            _ => false,
        };
        let num_probe_columns = probe_schema.fields().len();
        let probe_key_columns = on
            .iter()
            .map(|(_, probe_key)| {
                probe_key
                    .downcast_ref::<Column>()
                    .map(Column::index)
                    .filter(|&index| index < num_probe_columns)
            })
            .collect::<Option<Vec<_>>>();
        let compacts_probe_chunks = column_indices
            .iter()
            .chain(filter.iter().flat_map(|filter| filter.column_indices()))
            .filter(|column| column.side == JoinSide::Right)
            .any(|column| {
                probe_schema
                    .fields()
                    .get(column.index)
                    .is_none_or(|field| has_run_end_encoded(field.data_type()))
            });
        Ok(Self {
            join_type,
            bounded_keys: vec![false; on.len()],
            on,
            filter,
            null_equality,
            null_aware,
            projection,
            join_schema,
            schema,
            column_indices,
            layout,
            key_schema: Arc::new(Schema::new(key_fields)),
            payload,
            build_columns,
            defers_build_key_errors,
            tracks_key_range,
            probe_key_columns,
            compacts_probe_chunks,
        })
    }

    fn num_keys(&self) -> usize {
        self.on.len()
    }

    /// The layout column of build column `column`.
    fn build_column(&self, column: usize) -> Result<usize> {
        self.build_columns
            .get(column)
            .copied()
            .flatten()
            .ok_or_else(|| {
                internal_datafusion_err!(
                    "Build column {column} is not stored by the join"
                )
            })
    }

    /// Whether the join emits build rows once the probe side is exhausted
    /// (DataFusion's `need_produce_result_in_final`), and therefore tracks
    /// the build rows that found a match.
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

    /// Whether the join emits build rows without a match, so that it keeps
    /// the build rows that cannot match (with a NULL key under
    /// `NullEqualsNothing`).
    fn emits_unmatched_build_rows(&self) -> bool {
        matches!(
            self.join_type,
            JoinType::Left | JoinType::Full | JoinType::LeftAnti | JoinType::LeftMark
        )
    }

    /// Whether the join emits a row for a probe row without a match (`Right`,
    /// `Full`, `RightAnti` and `RightMark`, which emits every probe row).
    fn emits_unmatched_probe_rows(&self) -> bool {
        matches!(
            self.join_type,
            JoinType::Right | JoinType::Full | JoinType::RightAnti | JoinType::RightMark
        )
    }

    /// The rows whose keys are all valid, `None` if all are or if NULL keys
    /// can match (DataFusion's `matchable_join_keys`).
    fn valid_keys(&self, keys: &[ArrayRef]) -> Option<NullBuffer> {
        match self.null_equality {
            NullEquality::NullEqualsNothing => {
                let nulls = keys
                    .iter()
                    .map(|key| key.logical_nulls())
                    .collect::<Vec<_>>();
                NullBuffer::union_many(nulls.iter().map(Option::as_ref))
                    .filter(|nulls| nulls.null_count() > 0)
            }
            NullEquality::NullEqualsNull => None,
        }
    }
}

impl PhysicalHashJoin {
    /// A join of `build` (DataFusion's left input) and `probe` (its right
    /// input) on the keys `on` (`(build key, probe key)`, evaluated on the
    /// rows of the respective input), with the semantics of a `HashJoinExec`
    /// with the same arguments. The output has the schema of that
    /// `HashJoinExec`.
    #[expect(clippy::too_many_arguments)]
    pub fn try_new(
        build: Arc<dyn PhysicalOperator>,
        probe: Arc<dyn PhysicalOperator>,
        on: Vec<(PhysicalExprRef, PhysicalExprRef)>,
        filter: Option<JoinFilter>,
        join_type: JoinType,
        projection: Option<Vec<usize>>,
        null_equality: NullEquality,
        null_aware: bool,
    ) -> Result<Self> {
        let spec = JoinSpec::try_new(
            &build.schema(),
            &probe.schema(),
            on,
            filter,
            join_type,
            projection,
            null_equality,
            null_aware,
        )?;
        Ok(Self {
            children: vec![build, probe],
            spec,
            probe_plan: None,
            probe_streams_cannot_fail: true,
            dynamic_filters: vec![],
        })
    }

    /// Lower a DataFusion `HashJoinExec` to a [`PhysicalHashJoin`] with the
    /// same semantics and schema. `lower_input` lowers the inputs of the
    /// join. Returns `Ok(None)` for any other node, and for the joins that
    /// are left to DataFusion:
    ///
    /// * a join with a limit (`fetch`): DataFusion limits every output
    ///   partition of the join separately, which an engine without
    ///   partitions cannot reproduce (a planner that knows that a limit
    ///   above the join makes its `fetch` redundant can lower the join
    ///   without it, `HashJoinExec::with_fetch(None)`);
    /// * a join that produces a dynamic filter (see
    ///   `ExecutionPlan::dynamic_expressions_produced`) whose probe input has
    ///   a node that evaluates an expression that may fail (see
    ///   `evaluates_only_infallible_expressions`). DataFusion attaches the
    ///   filter to a join when a node of its probe side, such as a Parquet
    ///   scan, reads it. Once the build side is collected, DataFusion sets
    ///   the filter to a predicate of the build keys (bounds, an `IN` list or
    ///   a hash table lookup, depending on the size of the build side), with
    ///   which that node skips probe rows (or whole files and row groups) that
    ///   cannot match, so that the nodes above it never evaluate their
    ///   expressions on those rows (nor the probe sides of the joins whose
    ///   build side it empties). A [`PhysicalHashJoin`] sets the filter to the
    ///   bounds of its build keys only (see
    ///   [`Self::dynamic_filter_predicate`]): its probe side sees more rows,
    ///   and the join drops the extra ones (they cannot match), which makes no
    ///   difference unless an expression of the probe side fails on them
    ///   (like a division by zero). Joins without a
    ///   dynamic filter are lowered like any other, e.g. with
    ///   `datafusion.optimizer.enable_join_dynamic_filter_pushdown` off, or
    ///   over scans that do not read it (e.g. of memory);
    /// * a join with a prepared build side, whose left input is not
    ///   executed, and the joins DataFusion does not execute: in
    ///   `PartitionMode::Auto`, in `PartitionMode::CollectLeft` with several
    ///   build partitions, and in `PartitionMode::Partitioned` with different
    ///   numbers of build and probe partitions; and the joins without probe
    ///   partitions, which have no output partitions, so that DataFusion
    ///   executes neither of their inputs (nor evaluates their keys);
    /// * a `Partitioned` join of several partitions whose join type produces
    ///   no output without build rows (`empty_build_side_produces_empty_result`),
    ///   unless its probe side is partitioned directly below the join: a
    ///   partition of DataFusion's join whose build partition is empty does not
    ///   read its probe partition, so the nodes between the join and the
    ///   repartitioning of its probe side do not run on the rows of that
    ///   partition (and raise no errors on them);
    /// * a join whose probe input cannot be copied with its own state (see
    ///   `reset_plan_states`): with a leaf other than a scan of files or of
    ///   memory, or of no rows (e.g. a `StreamingTableExec`, whose partitions
    ///   may open a file or take a channel when they are executed), or with a
    ///   node that is not copied. `HashJoinExec::execute` creates the streams
    ///   of its probe input right away, which the native join does too, on
    ///   such a copy, to raise the same errors before it reads its probe
    ///   side;
    /// * a join whose residual filter or probe keys can fail on some rows
    ///   (conservatively, any expression but columns, literals, comparisons,
    ///   boolean operators, wrapping arithmetic, divisions by constants
    ///   other than 0 and -1, and casts that cannot fail): DataFusion
    ///   evaluates the filter on up to `batch_size` candidate pairs of a
    ///   probe batch, and the probe keys on whole probe batches, before it
    ///   emits any of their rows, while a native join evaluates them chunk by
    ///   chunk and emits as it goes, so that it could stop (e.g. under a
    ///   limit) before it evaluates them on rows on which DataFusion fails;
    /// * a join whose single build key of an integer type can fail on some
    ///   rows (e.g. `1 / k`), or with a dynamic filter, whose build keys can
    ///   fail on some rows or have a type whose bounds may fail to be computed
    ///   (but numbers, temporal types, booleans, strings and binaries, see
    ///   `has_bounds`): DataFusion evaluates such keys on every
    ///   whole build batch as it collects them (to compute their bounds, for
    ///   its perfect hash join or for its dynamic filter), also on an empty
    ///   one, while a native join evaluates them chunk by chunk, so that it
    ///   could stop (e.g. canceled by a limit above) before it reaches the
    ///   rows on which DataFusion fails;
    /// * a join whose probe keys fail without any row (e.g. `1 / 0`):
    ///   DataFusion evaluates them on every batch of its probe side, also on
    ///   an empty one, which a native join, called with the non-empty chunks
    ///   of its probe side, cannot reproduce;
    /// * a null-aware join with several keys or a filter (a correlated or
    ///   multi-column `NOT IN`), or with `NullEqualsNull`;
    /// * a join whose build side (or build keys) has dictionaries with keys of
    ///   8 or 16 bits, or run-end encoded arrays with run ends of 16 bits:
    ///   DataFusion concatenates its whole build side (all its columns),
    ///   which overflows the keys or run ends if they cannot hold all its
    ///   distinct values or rows, while the row store does not; and the row
    ///   store encodes the dictionary columns it gathers again, with the
    ///   distinct values of the gathered rows (which more than 128 or 256 of
    ///   them in a chunk overflow), while DataFusion takes them from its whole
    ///   build side;
    /// * a join whose keys have different types on the two sides, or whose
    ///   build columns the [`TupleDataLayout`] cannot store.
    ///
    /// The inputs of the join are lowered without the nodes directly below
    /// the join that only distribute the rows of the inputs over partitions:
    /// the join has a single build side shared by all threads, which read the
    /// probe side directly. These are the `CoalescePartitionsExec`s without a
    /// limit and the `RepartitionExec`s that do not preserve the order and
    /// evaluate nothing but join keys of their side that cannot fail (round
    /// robin, or hash partitioning by such join keys), which the join
    /// evaluates on all rows too. Other `RepartitionExec`s are kept, so that
    /// DataFusion evaluates their expressions as their rows arrive (and
    /// raises their errors right away, while the join may raise errors of its
    /// build keys only once its build side is complete).
    ///
    /// The probe side of a join may have pipeline breakers, e.g. a join or
    /// an aggregation (as in the chains of joins of TPC-H): DataFusion reads
    /// the probe side only once the build side is collected (not if it never
    /// completes, e.g. before a limit above cancels the join, nor if the join
    /// cannot produce output), so their pipelines wait for the build side of
    /// the join (see [`MetaPipelineBuilder::build_with_dependency`]), and do
    /// not compute (nor raise errors) where DataFusion does not.
    pub fn try_lower<F>(
        plan: &Arc<dyn ExecutionPlan>,
        mut lower_input: F,
    ) -> Result<Option<Self>>
    where
        F: FnMut(&Arc<dyn ExecutionPlan>) -> Result<Arc<dyn PhysicalOperator>>,
    {
        let Some(join) = plan.downcast_ref::<HashJoinExec>() else {
            return Ok(None);
        };
        // The partitions DataFusion executes the join with.
        let partitions = |input: &Arc<dyn ExecutionPlan>| {
            input.output_partitioning().partition_count()
        };
        let executable = partitions(join.right()) > 0
            && match join.partition_mode() {
                PartitionMode::CollectLeft => partitions(join.left()) == 1,
                PartitionMode::Partitioned => {
                    partitions(join.left()) == partitions(join.right())
                }
                PartitionMode::Auto => false,
            };
        if !executable
            || join.fetch().is_some()
            || has_prepared_build(plan)
            || !probe_streams_can_be_replayed(join.right())
        {
            return Ok(None);
        }
        // The dynamic filters that nodes of the probe side read and prune
        // rows with (in DataFusion): the native join sets them to the bounds
        // of its build keys only, which changes which rows the probe side
        // evaluates its expressions on, so these must not fail on any row.
        let Some(dynamic_filters) = dynamic_filters(plan) else {
            return Ok(None);
        };
        if !dynamic_filters.is_empty()
            && !evaluates_only_infallible_expressions(join.right())
        {
            return Ok(None);
        }
        let mut spec = match JoinSpec::try_new(
            &join.left().schema(),
            &join.right().schema(),
            join.on().to_vec(),
            join.filter().cloned(),
            *join.join_type(),
            join.projection.as_deref().map(<[usize]>::to_vec),
            join.null_equality(),
            join.null_aware,
        ) {
            Ok(spec) => spec,
            // Unsupported types or null-aware joins, or keys of different
            // types.
            Err(DataFusionError::NotImplemented(_) | DataFusionError::Plan(_)) => {
                return Ok(None);
            }
            Err(e) => return Err(e),
        };
        if spec.schema != plan.schema() {
            return internal_err!(
                "PhysicalHashJoin would have schema {:?}, but the HashJoinExec has schema {:?}",
                spec.schema,
                plan.schema()
            );
        }
        let build_keys = join.on().iter().map(|(build, _)| build).collect::<Vec<_>>();
        let probe_keys = join.on().iter().map(|(_, probe)| probe).collect::<Vec<_>>();
        // For its dynamic filter, DataFusion computes the bounds of the build
        // keys (their minimum and maximum) as it collects its build side: it
        // evaluates them on every build batch (see
        // `JoinSpec::defers_build_key_errors`), and accumulates them with the
        // `MIN` and `MAX` aggregates, which fail on the types they do not
        // support. Only keys of the types they support are lowered.
        let build_schema = join.left().schema();
        if !dynamic_filters.is_empty() {
            spec.defers_build_key_errors = false;
            if !build_keys.iter().all(|key| {
                key.data_type(&build_schema)
                    .is_ok_and(|data_type| has_bounds(&data_type))
            }) {
                return Ok(None);
            }
            // The join sets its dynamic filters to the bounds of the keys that
            // have exact bounds (see `dynamic_filter_predicate`).
            spec.bounded_keys = build_keys
                .iter()
                .map(|key| {
                    key.data_type(&build_schema)
                        .is_ok_and(|data_type| has_exact_bounds(&data_type))
                })
                .collect();
        }
        // A partition of a `Partitioned` join whose build partition cannot
        // produce output does not read its probe partition, whose nodes then
        // do not run on its rows. All threads of a native join read the whole
        // probe side: the same rows only if the probe side is partitioned
        // directly below the join, by a repartitioning that reads all of its
        // input (if any partition reads it), which the native join reads.
        if *join.partition_mode() == PartitionMode::Partitioned
            && partitions(join.right()) > 1
            && join.join_type().empty_build_side_produces_empty_result()
            && Arc::ptr_eq(
                without_redistribution(join.right(), &probe_keys),
                join.right(),
            )
        {
            return Ok(None);
        }
        // DataFusion evaluates the residual filter on up to `batch_size`
        // candidate pairs of a probe batch at once, and the probe keys on
        // whole probe batches (also empty ones), before it emits any of their
        // rows. A native join evaluates them chunk by chunk (and never on an
        // empty chunk) and emits as it goes, so that, e.g. under a limit, it
        // could stop before it evaluates them on rows on which DataFusion
        // fails: a filter or probe keys that can fail on some rows (or on no
        // rows, e.g. `1 / 0`) are left to DataFusion. (Lowering evaluates no
        // expression: an expression may change its state when evaluated.)
        let probe_schema = join.right().schema();
        if join
            .filter()
            .is_some_and(|filter| can_fail(filter.expression(), filter.schema()))
            || probe_keys.iter().any(|key| can_fail(key, &probe_schema))
        {
            return Ok(None);
        }
        // DataFusion concatenates its whole build side (all its columns, also
        // those the join does not need), which fails if a dictionary needs
        // more keys, or run-end encoded values more run ends, than their types
        // hold; the row store does not. And it encodes the dictionaries it
        // gathers (for the output or the filter) again, with the distinct
        // values of the gathered rows, so that more of them than keys of 8
        // bits hold in a chunk fail, where DataFusion takes them from its
        // whole build side. Joins with such columns on the build side (or such
        // build keys) are left to DataFusion.
        if join
            .left()
            .schema()
            .fields()
            .iter()
            .chain(spec.layout.schema().fields())
            .any(|field| concatenation_can_overflow(field.data_type()))
        {
            return Ok(None);
        }
        // DataFusion evaluates the build keys that it does not defer (see
        // `JoinSpec::defers_build_key_errors`) on every build batch as it
        // collects it (to compute their bounds): on whole batches, also on
        // empty ones, while a native join evaluates them on the non-empty
        // chunks of its build side, so that it could stop (e.g. when a limit
        // above cancels the join) before it evaluates them on rows on which
        // DataFusion fails. Such keys that can fail are left to DataFusion.
        if !spec.defers_build_key_errors
            && build_keys.iter().any(|key| can_fail(key, &build_schema))
        {
            return Ok(None);
        }
        // The pipeline breakers of a native probe side (e.g. the build side
        // of a join below) wait for the build side of the join, see
        // `build_pipelines`.
        let mut children = Vec::with_capacity(2);
        for (input, keys) in [(join.left(), build_keys), (join.right(), probe_keys)] {
            let input = without_redistribution(input, &keys);
            let lowered = lower_input(input)?;
            if lowered.schema() != input.schema() {
                return internal_err!(
                    "The lowered input {} of a HashJoinExec has schema {:?}, expected {:?}",
                    lowered.name(),
                    lowered.schema(),
                    input.schema()
                );
            }
            children.push(lowered);
        }
        Ok(Some(Self {
            children,
            spec,
            probe_streams_cannot_fail: probe_streams_cannot_fail(join.right()),
            probe_plan: Some(Arc::clone(join.right())),
            dynamic_filters,
        }))
    }

    /// Create the streams of all partitions of the probe input of the
    /// `HashJoinExec` the join was lowered from and drop them without polling
    /// them, as `HashJoinExec::execute` creates the probe stream of its
    /// partition before it reads its build side (and also if it then does not
    /// read its probe side): the same errors (e.g. of a scan of an object
    /// store that is not registered), nothing read.
    ///
    /// The streams are those of a copy of the probe input with its own state
    /// (see `reset_plan_states`, and [`probe_streams_can_be_replayed`]):
    /// creating them can start tasks that already read (e.g. those of a
    /// `CoalescePartitionsExec`), which must not take, for instance, the
    /// channels of a `RepartitionExec` of the probe input that the probe side
    /// still reads.
    ///
    /// The streams of probe inputs whose streams are created without failing
    /// (see [`probe_streams_cannot_fail`]), e.g. of scans of memory and of
    /// local files, are not created (nor is the input copied): with the local
    /// file system registered in the runtime, creating them would not fail.
    fn create_probe_streams(&self, ctx: &ExecutionContext) -> Result<()> {
        let Some(probe) = &self.probe_plan else {
            return Ok(());
        };
        if self.probe_streams_cannot_fail
            && ctx
                .task_ctx
                .runtime_env()
                .object_store(ObjectStoreUrl::local_filesystem())
                .is_ok()
        {
            return Ok(());
        }
        let probe = reset_plan_states(Arc::clone(probe))?;
        for partition in 0..probe.output_partitioning().partition_count() {
            drop(probe.execute(partition, Arc::clone(&ctx.task_ctx))?);
        }
        Ok(())
    }

    /// Finalize the build side: build the hash table and set the dynamic
    /// filters (see [`Self::dynamic_filter_predicate`]), unless the join was
    /// canceled or its build keys failed.
    fn finalize_build(
        &self,
        ctx: &ExecutionContext,
        state: &JoinState,
    ) -> Result<SinkFinalize> {
        let spec = &self.spec;
        // The output of a canceled join is not needed any more: its build
        // pipelines stopped or did not run, and the pipelines that read its
        // output are skipped. DataFusion does not collect the build side of a
        // join that is not read, so the join neither evaluates its keys nor
        // builds its table (nor reserves its memory).
        if ctx.registry().is_sink_finished(self) {
            return Ok(SinkFinalize::NoOutputPossible);
        }
        // An error of the build keys that DataFusion raises once the whole
        // build side is collected (and not if the join is canceled before).
        if let Some(error) = state.deferred_build_error.lock().take() {
            return Err(error);
        }
        let build_empty = state.build_rows.load(Ordering::Relaxed) == 0;
        // DataFusion evaluates the build keys on its whole build side, also
        // when it is empty, which raises the errors of keys that fail without
        // any row (e.g. `1 / 0`).
        if build_empty {
            let batch = RecordBatch::new_empty(self.build_side().schema());
            evaluate_expressions_to_arrays(
                spec.on.iter().map(|(build, _)| build),
                &batch,
            )?;
        }
        let table = state.finish_build(spec)?;
        if !self.dynamic_filters.is_empty()
            && let Some(predicate) =
                self.dynamic_filter_predicate(&table, &state.key_stats.lock())
        {
            for filter in &self.dynamic_filters {
                filter.update(Arc::clone(&predicate))?;
            }
        }
        // Like DataFusion, which then does not read its probe side either.
        if (build_empty && spec.join_type.empty_build_side_produces_empty_result())
            || (table.is_empty() && spec.join_type.empty_map_produces_empty_result())
        {
            return Ok(SinkFinalize::NoOutputPossible);
        }
        Ok(SinkFinalize::Ready)
    }

    /// The predicate of the probe keys that the dynamic filters of the join are
    /// set to once its build side is collected: like DataFusion's (without its
    /// `IN` lists and hash table lookups), `key >= min AND key <= max` for every
    /// key with exact bounds (see [`JoinSpec::bounded_keys`]), with `OR key IS
    /// NULL` if the NULL keys of the build side can match (under
    /// `NullEqualsNull`), and `false` if no build row can match. The nodes of
    /// the probe side that read the filters (e.g. Parquet scans) then skip the
    /// files, row groups, pages and rows whose keys no build key equals; the
    /// join drops these rows anyway, and its probe side evaluates only
    /// expressions that cannot fail (see [`Self::try_lower`]).
    ///
    /// `None` (the filters stay `true`) if no key has bounds, and for the joins
    /// that emit their probe rows without a match, and the null-aware ones,
    /// whose probe rows count also if they do not match.
    fn dynamic_filter_predicate(
        &self,
        table: &JoinHashTable,
        stats: &BuildKeyStats,
    ) -> Option<PhysicalExprRef> {
        let spec = &self.spec;
        if spec.null_aware || spec.emits_unmatched_probe_rows() {
            return None;
        }
        if table.is_empty() {
            return Some(lit(false));
        }
        let null_keys_match = spec.null_equality == NullEquality::NullEqualsNull;
        let predicates = spec
            .on
            .iter()
            .zip(&stats.bounds)
            .zip(&spec.bounded_keys)
            .filter(|((_, bounds), bounded)| **bounded && !bounds.failed)
            .filter_map(|(((_, probe_key), bounds), _)| {
                let range = bounds.min_max.as_ref().map(|(min, max)| {
                    let at_least = BinaryExpr::new(
                        Arc::clone(probe_key),
                        Operator::GtEq,
                        lit(min.clone()),
                    );
                    let at_most = BinaryExpr::new(
                        Arc::clone(probe_key),
                        Operator::LtEq,
                        lit(max.clone()),
                    );
                    Arc::new(BinaryExpr::new(
                        Arc::new(at_least),
                        Operator::And,
                        Arc::new(at_most),
                    )) as PhysicalExprRef
                });
                let is_null = (null_keys_match && bounds.has_null).then(|| {
                    Arc::new(IsNullExpr::new(Arc::clone(probe_key))) as PhysicalExprRef
                });
                match (range, is_null) {
                    (Some(range), Some(is_null)) => {
                        Some(Arc::new(BinaryExpr::new(range, Operator::Or, is_null)) as _)
                    }
                    (range, is_null) => range.or(is_null),
                }
            })
            .collect::<Vec<_>>();
        (!predicates.is_empty()).then(|| conjunction(predicates))
    }

    /// The build side (DataFusion's left input).
    pub fn build_side(&self) -> &Arc<dyn PhysicalOperator> {
        &self.children[0]
    }

    /// The probe side (DataFusion's right input).
    pub fn probe_side(&self) -> &Arc<dyn PhysicalOperator> {
        &self.children[1]
    }

    /// The join type.
    pub fn join_type(&self) -> JoinType {
        self.spec.join_type
    }

    /// The join keys: `(build key, probe key)`.
    pub fn on(&self) -> &[(PhysicalExprRef, PhysicalExprRef)] {
        &self.spec.on
    }

    /// The residual filter.
    pub fn filter(&self) -> Option<&JoinFilter> {
        self.spec.filter.as_ref()
    }

    /// How NULL keys are compared.
    pub fn null_equality(&self) -> NullEquality {
        self.spec.null_equality
    }

    /// Whether the join has `NOT IN` semantics.
    pub fn null_aware(&self) -> bool {
        self.spec.null_aware
    }

    /// The output columns, as indices of the columns of the join without
    /// projection.
    pub fn projection(&self) -> Option<&[usize]> {
        self.spec.projection.as_deref()
    }

    /// The global state of the join in the execution of `ctx`.
    fn join_state(&self, ctx: &ExecutionContext) -> Result<Arc<JoinState>> {
        join_state(ctx, self)
    }

    /// Probe the table with the next rows of `probe`, writing the output to
    /// `output`.
    fn probe_next(&self, state: &mut ProbeState, output: &mut DataChunk) -> Result<()> {
        let spec = &self.spec;
        let ProbeState {
            table,
            current,
            scan: chain_scan,
            probe_rows,
            matches,
            ..
        } = state;
        let table: &JoinHashTable = table;
        let Some(chunk) = current.as_mut() else {
            return internal_err!("PhysicalHashJoin has no probe chunk");
        };
        // Without a filter, a probe row has a match if and only if it has a
        // chain: the joins that emit every probe row at most once (and no
        // build column) emit all of their output for the chunk at once,
        // without walking the chains.
        if spec.filter.is_none()
            && matches!(
                spec.join_type,
                JoinType::RightSemi | JoinType::RightAnti | JoinType::RightMark
            )
        {
            let rows = chain_scan.skip_rows(DEFAULT_VECTOR_SIZE);
            *output = match spec.join_type {
                JoinType::RightMark => {
                    let marks = rows.clone().map(|row| chain_scan.has_matches(row));
                    self.marked_probe_chunk(chunk, rows, marks)?
                }
                join_type => {
                    let semi = join_type == JoinType::RightSemi;
                    let rows = rows
                        .filter(|&row| {
                            if chain_scan.has_matches(row) {
                                semi
                            } else {
                                !semi && chunk.emits_unmatched(row)
                            }
                        })
                        .map(|row| row as u32)
                        .collect();
                    self.selected_probe_chunk(chunk, rows)?
                }
            };
            if chain_scan.is_done() {
                *current = None;
            }
            return Ok(());
        }
        let unmatched_cost = usize::from(spec.emits_unmatched_probe_rows());
        let scan = chain_scan.next_batch(
            table,
            DEFAULT_VECTOR_SIZE,
            unmatched_cost,
            probe_rows,
            matches,
        );
        if let Some(filter) = &spec.filter {
            self.apply_filter(filter, table, chunk, probe_rows, matches)?;
        }
        table.mark_matched(matches);

        let matched_in_progress = chunk.matched_in_progress;
        chunk.matched_in_progress = scan.in_progress
            && (probe_rows
                .last()
                .is_some_and(|&row| row as usize == scan.end)
                || (scan.end == scan.start && matched_in_progress));
        // Whether the probe row `row`, whose matches all were returned
        // when `next` reached the first pair after it, found a match.
        let finished_matched = |row: usize, first: usize, next: usize| -> bool {
            next > first || (row == scan.start && matched_in_progress)
        };
        match spec.join_type {
            JoinType::Inner | JoinType::Left => {
                // The probe rows of the pairs are taken by the output (no copy
                // of them), and their buffer is reused for the next batch.
                let mut rows = std::mem::take(probe_rows);
                chunk.to_physical(&mut rows);
                let probe = UInt32Array::from(rows);
                *output = self.output_chunk(
                    table,
                    probe.len(),
                    Some((matches.as_slice(), None)),
                    Some((&chunk.columns, &probe)),
                    None,
                )?;
                *probe_rows = into_indices(probe);
            }
            JoinType::Right | JoinType::Full => {
                // The matches of every probe row, followed by the probe row
                // with NULLs if it finished without a match.
                let mut output_probe = Vec::with_capacity(probe_rows.len());
                let mut output_build = Vec::with_capacity(probe_rows.len());
                let mut padded = false;
                let mut next = 0;
                for row in scan.start..scan.end {
                    let first = next;
                    while next < probe_rows.len() && probe_rows[next] as usize == row {
                        output_probe.push(row as u32);
                        output_build.push(Some(next as u32));
                        next += 1;
                    }
                    if !finished_matched(row, first, next) {
                        output_probe.push(row as u32);
                        output_build.push(None);
                        padded = true;
                    }
                }
                for (pair, &row) in probe_rows.iter().enumerate().skip(next) {
                    output_probe.push(row);
                    output_build.push(Some(pair as u32));
                }
                let build_take = padded.then(|| UInt32Array::from(output_build));
                chunk.to_physical(&mut output_probe);
                let probe = UInt32Array::from(output_probe);
                *output = self.output_chunk(
                    table,
                    probe.len(),
                    Some((matches.as_slice(), build_take.as_ref())),
                    Some((&chunk.columns, &probe)),
                    None,
                )?;
            }
            JoinType::RightSemi | JoinType::RightAnti | JoinType::RightMark => {
                let mut rows = Vec::with_capacity(scan.end - scan.start);
                let mut marks = Vec::new();
                let mut next = 0;
                for row in scan.start..scan.end {
                    let first = next;
                    while next < probe_rows.len() && probe_rows[next] as usize == row {
                        next += 1;
                    }
                    let matched = finished_matched(row, first, next);
                    match spec.join_type {
                        JoinType::RightSemi if matched => rows.push(row as u32),
                        JoinType::RightAnti if !matched && chunk.emits_unmatched(row) => {
                            rows.push(row as u32)
                        }
                        JoinType::RightMark => marks.push(matched),
                        _ => {}
                    }
                }
                *output = match spec.join_type {
                    JoinType::RightMark => {
                        self.marked_probe_chunk(chunk, scan.start..scan.end, marks)?
                    }
                    _ => self.selected_probe_chunk(chunk, rows)?,
                };
            }
            JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark => {
                // The build rows are emitted once the probe side is exhausted.
            }
        }
        if chain_scan.is_done() {
            *current = None;
        }
        Ok(())
    }

    /// Keep the pairs of probe rows and build rows (`probe_rows[i]` and
    /// `matches[i]`) for which the residual filter is true.
    fn apply_filter(
        &self,
        filter: &JoinFilter,
        table: &JoinHashTable,
        chunk: &ProbeChunk,
        probe_rows: &mut Vec<u32>,
        matches: &mut Vec<RowLocation>,
    ) -> Result<()> {
        let num_pairs = probe_rows.len();
        if num_pairs == 0 {
            return Ok(());
        }
        let build_columns = filter
            .column_indices()
            .iter()
            .filter(|column| column.side == JoinSide::Left)
            .map(|column| self.spec.build_column(column.index))
            .collect::<Result<Vec<_>>>()?;
        let mut build_arrays = table
            .collection()
            .gather_columns(matches, &build_columns)?
            .into_iter();
        let mut physical_rows = probe_rows.clone();
        chunk.to_physical(&mut physical_rows);
        let probe_indices = UInt32Array::from(physical_rows);
        let columns = filter
            .column_indices()
            .iter()
            .map(|column| match column.side {
                JoinSide::Left => build_arrays.next().ok_or_else(|| {
                    internal_datafusion_err!("Missing build column of the join filter")
                }),
                JoinSide::Right => take_rows(chunk.column(column.index)?, &probe_indices),
                JoinSide::None => {
                    internal_err!("A join filter cannot read the mark column")
                }
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
        let mut kept = 0;
        for pair in 0..num_pairs {
            if mask.is_valid(pair) && mask.value(pair) {
                probe_rows[kept] = probe_rows[pair];
                matches[kept] = matches[pair];
                kept += 1;
            }
        }
        probe_rows.truncate(kept);
        matches.truncate(kept);
        Ok(())
    }

    /// An output chunk of `num_rows` rows:
    ///
    /// * the build columns are gathered from the rows at `build.0`, and
    ///   taken at the indices `build.1` (into `build.0`) if given, which are
    ///   NULL for the rows without a build row; NULL if `build` is `None` or
    ///   has no rows;
    /// * the probe columns are taken from the rows `probe.1` of the (physical)
    ///   probe columns `probe.0`; NULL if `probe` is `None`;
    /// * the mark column is `mark`.
    fn output_chunk(
        &self,
        table: &JoinHashTable,
        num_rows: usize,
        build: Option<(&[RowLocation], Option<&UInt32Array>)>,
        probe: Option<(&[ArrayRef], &UInt32Array)>,
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
        // No build row at all (e.g. the unmatched probe rows of a `Right`
        // join with an empty build side): the build columns are NULL.
        let build = build.filter(|(locations, _)| !locations.is_empty());
        let build_columns = match build {
            Some(_) => spec
                .column_indices
                .iter()
                .filter(|column| column.side == JoinSide::Left)
                .map(|column| spec.build_column(column.index))
                .collect::<Result<Vec<_>>>()?,
            None => vec![],
        };
        let mut build_arrays = match build {
            Some((locations, take_indices)) => {
                let gathered = table
                    .collection()
                    .gather_columns(locations, &build_columns)?;
                match take_indices {
                    Some(indices) => gathered
                        .iter()
                        .map(|array| Ok(take(array.as_ref(), indices, None)?))
                        .collect::<Result<Vec<_>>>()?,
                    None => gathered,
                }
            }
            None => vec![],
        }
        .into_iter();
        let columns = spec
            .column_indices
            .iter()
            .zip(spec.schema.fields())
            .map(|(column, field)| match column.side {
                JoinSide::Left => match build {
                    Some(_) => build_arrays.next().ok_or_else(|| {
                        internal_datafusion_err!("Missing build column of the join")
                    }),
                    None => Ok(new_null_array(field.data_type(), num_rows)),
                },
                JoinSide::Right => match probe {
                    Some((columns, indices)) => match columns.get(column.index) {
                        Some(array) => take_rows(array, indices),
                        None => internal_err!("Missing probe column of the join"),
                    },
                    None => Ok(new_null_array(field.data_type(), num_rows)),
                },
                JoinSide::None => mark.map(Arc::clone).ok_or_else(|| {
                    internal_datafusion_err!("Missing mark column of the join")
                }),
            })
            .collect::<Result<Vec<_>>>()?;
        DataChunk::try_new(Arc::clone(&spec.schema), columns, None)
    }

    /// The output chunk of the probe rows `rows` (logical rows of `chunk`) of
    /// a join whose output has probe columns only (`RightSemi` and
    /// `RightAnti`): the probe columns of `chunk` (no copy), with a selection
    /// of the physical rows of `rows` (none if they are all physical rows, in
    /// order). The columns are taken when the chunk is compacted, e.g. by the
    /// operator above, as DataFusion takes them from its probe batches.
    fn selected_probe_chunk(
        &self,
        chunk: &ProbeChunk,
        mut rows: Vec<u32>,
    ) -> Result<DataChunk> {
        let spec = &self.spec;
        if rows.is_empty() {
            return Ok(DataChunk::empty(Arc::clone(&spec.schema)));
        }
        if spec.schema.fields().is_empty() {
            return DataChunk::try_new_without_columns(
                Arc::clone(&spec.schema),
                rows.len(),
            );
        }
        let columns = spec
            .column_indices
            .iter()
            .map(|column| match column.side {
                JoinSide::Right => chunk.column(column.index).map(Arc::clone),
                _ => {
                    internal_err!("A {} join only outputs probe columns", spec.join_type)
                }
            })
            .collect::<Result<Vec<_>>>()?;
        chunk.to_physical(&mut rows);
        let selection = chunk.selection_of(rows);
        DataChunk::try_new(Arc::clone(&spec.schema), columns, selection)
    }

    /// The output chunk of a `RightMark` join of the probe rows `rows` of
    /// `chunk`, whose marks are `marks` (one per row of `rows`): like
    /// [`Self::selected_probe_chunk`], the probe columns of `chunk` (no copy)
    /// with a selection of the physical rows of `rows`, and a mark column with
    /// a value for every physical row (`false` for the rows not in `rows`).
    fn marked_probe_chunk(
        &self,
        chunk: &ProbeChunk,
        rows: Range<usize>,
        marks: impl IntoIterator<Item = bool>,
    ) -> Result<DataChunk> {
        let spec = &self.spec;
        if rows.is_empty() {
            return Ok(DataChunk::empty(Arc::clone(&spec.schema)));
        }
        if spec.schema.fields().is_empty() {
            return DataChunk::try_new_without_columns(
                Arc::clone(&spec.schema),
                rows.len(),
            );
        }
        let mut physical_rows = rows.map(|row| row as u32).collect::<Vec<_>>();
        chunk.to_physical(&mut physical_rows);
        let mut physical_marks = vec![false; chunk.physical_len];
        for (&row, mark) in physical_rows.iter().zip(marks) {
            physical_marks[row as usize] = mark;
        }
        let mark: ArrayRef = Arc::new(BooleanArray::from(physical_marks));
        let columns = spec
            .column_indices
            .iter()
            .map(|column| match column.side {
                JoinSide::Right => chunk.column(column.index).map(Arc::clone),
                JoinSide::None => Ok(Arc::clone(&mark)),
                JoinSide::Left => {
                    internal_err!("A RightMark join does not output build columns")
                }
            })
            .collect::<Result<Vec<_>>>()?;
        let selection = chunk.selection_of(physical_rows);
        DataChunk::try_new(Arc::clone(&spec.schema), columns, selection)
    }

    /// Start probing the rows of `input`, resetting `scan` to the scan of
    /// their matches. Returns `None` if the rows cannot produce any output
    /// and do not need to be looked up.
    ///
    /// The keys are evaluated on the logical rows only: a key of a row that a
    /// filter removed must not raise an error. Probe keys that are columns
    /// (see [`JoinSpec::probe_key_columns`]) are taken from the logical rows of
    /// the chunk, whose other columns are not compacted: the join only takes
    /// the probe rows that it outputs from them (or, for `RightSemi`,
    /// `RightAnti` and `RightMark` joins, selects them, see
    /// [`Self::selected_probe_chunk`]), not the rows that match nothing.
    fn start_chunk(
        &self,
        state: &mut ProbeState,
        input: &DataChunk,
    ) -> Result<Option<ProbeChunk>> {
        let ProbeState {
            join,
            table,
            scan,
            hashes,
            ..
        } = state;
        let table: &JoinHashTable = table;
        let spec = &self.spec;
        let num_rows = input.len();
        let (columns, selection, keys) =
            match (input.selection(), &spec.probe_key_columns) {
                (Some(selection), Some(key_columns))
                    if !spec.compacts_probe_chunks && !input.columns().is_empty() =>
                {
                    let gather = Gather::new(
                        selection,
                        input.physical_len(),
                        key_columns
                            .iter()
                            .map(|&column| input.column(column).data_type()),
                    );
                    let keys = key_columns
                        .iter()
                        .map(|&column| gather.column(input.column(column)))
                        .collect::<Result<Vec<_>>>()?;
                    (input.columns().to_vec(), Some(selection.clone()), keys)
                }
                _ => {
                    let batch = input.compact()?;
                    let keys = evaluate_expressions_to_arrays(
                        spec.on.iter().map(|(_, probe)| probe),
                        &batch,
                    )?;
                    let (_, columns, _) = batch.into_parts();
                    (columns, None, keys)
                }
            };
        let physical_len = columns.first().map_or(num_rows, |column| column.len());
        let mut exclude_null_keys = false;
        if spec.null_aware {
            match spec.join_type {
                JoinType::LeftAnti | JoinType::LeftMark => {
                    // `NULL NOT IN (empty)` is true: only probe chunks with
                    // rows count.
                    join.probe_has_rows.store(true, Ordering::Relaxed);
                    if keys[0].logical_null_count() > 0 {
                        join.probe_has_null_key.store(true, Ordering::Relaxed);
                    }
                    // With a NULL probe key, no build row is emitted: every
                    // unmatched one is UNKNOWN.
                    if spec.join_type == JoinType::LeftAnti
                        && join.probe_has_null_key.load(Ordering::Relaxed)
                    {
                        return Ok(None);
                    }
                }
                JoinType::RightAnti => {
                    // With a NULL build key, no probe row is emitted; with a
                    // non-empty build side, no probe row with a NULL key.
                    if join.build_null_key_rows.load(Ordering::Relaxed) > 0 {
                        return Ok(None);
                    }
                    exclude_null_keys = join.build_rows.load(Ordering::Relaxed) > 0;
                }
                _ => {}
            }
        }
        let valid = spec.valid_keys(&keys);
        let excluded = if exclude_null_keys {
            valid.clone()
        } else {
            None
        };
        // Like DataFusion, hash the keys of all probe rows, also if no build
        // row can match them: hashing fails for the types it does not
        // support. (A dense index looks up keys of integer types, whose
        // hashing cannot fail, without their hashes.)
        if table.is_empty() || table.needs_hashes() {
            hash_join_keys(&keys, num_rows, hashes)?;
        } else {
            hashes.clear();
        }
        if table.is_empty() {
            scan.reset_without_matches(num_rows);
        } else {
            let keys = DataChunk::try_new(Arc::clone(&spec.key_schema), keys, None)?;
            scan.reset(table, &keys, hashes, valid.as_ref())?;
            if spec.filter.is_none()
                && matches!(
                    spec.join_type,
                    JoinType::LeftSemi | JoinType::LeftAnti | JoinType::LeftMark
                )
            {
                // These joins only mark the build rows that match. Without a
                // filter, a probe row that finds a chain marks all its rows
                // (their keys are equal): a chain whose first row is marked
                // is (being) marked already.
                scan.skip_chains(|head| table.is_matched_at(head));
            }
        }
        Ok(Some(ProbeChunk {
            columns,
            selection,
            physical_len,
            matched_in_progress: false,
            excluded,
        }))
    }

    /// The final build rows of the next block of rows claimed by this
    /// thread, at most [`DEFAULT_VECTOR_SIZE`] of them, `None` once all
    /// blocks are scanned.
    fn next_final_rows(
        &self,
        global: &FinalScanState,
        local: &mut FinalScanLocalState,
    ) -> Result<Option<DataChunk>> {
        let spec = &self.spec;
        let table = global.table.as_ref();
        let collection = table.collection();
        loop {
            let block = match local.block {
                Some(block) => block,
                None => {
                    let block = global.next_block.fetch_add(1, Ordering::Relaxed);
                    if block >= collection.num_blocks() {
                        return Ok(None);
                    }
                    local.block = Some(block);
                    local.row = 0;
                    block
                }
            };
            let len = collection.block_len(block);
            let first_row = collection.row_index(RowLocation::new(block, 0));
            let end = (local.row + DEFAULT_VECTOR_SIZE).min(len);
            let rows = local.row..end;
            local.row = end;
            if end >= len {
                local.block = None;
            }
            let locations = rows
                .clone()
                .map(|row| RowLocation::new(block, row))
                .collect::<Vec<_>>();
            let matched = rows
                .map(|row| table.is_matched(first_row + row))
                .collect::<Vec<_>>();
            // For `NOT IN`: whether the key of a row is NULL.
            let null_keys = if spec.null_aware {
                table.null_keys(&locations)?
            } else {
                vec![]
            };
            let probe_has_rows = global.probe_has_rows;
            let probe_has_null_key = global.probe_has_null_key;
            let (emitted, mark): (Vec<RowLocation>, Option<ArrayRef>) =
                match spec.join_type {
                    JoinType::LeftSemi => (select(&locations, &matched, |m| m), None),
                    JoinType::LeftAnti if spec.null_aware => {
                        let emit = matched
                            .iter()
                            .zip(&null_keys)
                            .map(|(&matched, &null_key)| {
                                // Not TRUE (matched) nor UNKNOWN (a NULL
                                // on either side).
                                !(matched
                                    || probe_has_null_key
                                    || (null_key && probe_has_rows))
                            })
                            .collect::<Vec<_>>();
                        (select(&locations, &emit, |e| e), None)
                    }
                    JoinType::Left | JoinType::Full | JoinType::LeftAnti => {
                        (select(&locations, &matched, |m| !m), None)
                    }
                    JoinType::LeftMark => {
                        let marks = if spec.null_aware {
                            matched
                                .iter()
                                .zip(&null_keys)
                                .map(|(&matched, &null_key)| {
                                    if matched {
                                        Some(true)
                                    } else if probe_has_null_key
                                        || (null_key && probe_has_rows)
                                    {
                                        None
                                    } else {
                                        Some(false)
                                    }
                                })
                                .collect::<BooleanArray>()
                        } else {
                            BooleanArray::from(matched)
                        };
                        (locations, Some(Arc::new(marks)))
                    }
                    join_type => {
                        return internal_err!(
                            "A {join_type} join does not emit build rows at the end"
                        );
                    }
                };
            if emitted.is_empty() {
                continue;
            }
            let chunk = self.output_chunk(
                table,
                emitted.len(),
                Some((&emitted, None)),
                None,
                mark.as_ref(),
            )?;
            return Ok(Some(chunk));
        }
    }
}

/// The entries of `values` for which `keep` of the corresponding flag is
/// true.
fn select(
    values: &[RowLocation],
    flags: &[bool],
    keep: impl Fn(bool) -> bool,
) -> Vec<RowLocation> {
    values
        .iter()
        .zip(flags)
        .filter(|&(_, &flag)| keep(flag))
        .map(|(&value, _)| value)
        .collect()
}

/// The buffer of the indices `indices`, to reuse it (a new one if the buffer is
/// shared, e.g. by a slice).
fn into_indices(indices: UInt32Array) -> Vec<u32> {
    let (_, values, _) = indices.into_parts();
    let mut indices = values.into_inner().into_vec().unwrap_or_default();
    indices.clear();
    indices
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

/// Whether creating the streams of all partitions of `plan`, the probe input of
/// a `HashJoinExec`, cannot fail (so that a native join need not create them,
/// see [`PhysicalHashJoin::create_probe_streams`]):
///
/// * its leaves are scans of memory, generators or no rows, or Parquet (without
///   virtual columns or decryption) and CSV scans of the local file system,
///   whose object store is registered by default (unlike, e.g., the store of a
///   scan of `s3://`, which may not be);
/// * its other nodes create their streams without failing, but for failing to
///   create the streams of their inputs: projections, filters,
///   repartitionings, coalescings of partitions, hash joins (of valid
///   partitionings, without a prepared build), nested loop and cross joins,
///   aggregations, sorts, merges, unions, limits and cooperative nodes.
///
/// Creating the streams of such inputs reads nothing, but creating the streams
/// of a copy of a deep input (as the joins below a join, in a chain of joins)
/// starts the tasks that, e.g., coalesce partitions, which read until they are
/// dropped: not creating them saves that work, the copy, and the pipeline that
/// creates them.
fn probe_streams_cannot_fail(plan: &Arc<dyn ExecutionPlan>) -> bool {
    let children = plan.children();
    if children.is_empty() {
        return plan.is::<EmptyExec>()
            || plan.is::<PlaceholderRowExec>()
            || plan.is::<LazyMemoryExec>()
            || plan.downcast_ref::<DataSourceExec>().is_some_and(|scan| {
                let source = scan.data_source();
                source.is::<MemorySourceConfig>()
                    || source
                        .downcast_ref::<FileScanConfig>()
                        .is_some_and(local_scan_streams_cannot_fail)
            });
    }
    let partitions =
        |plan: &Arc<dyn ExecutionPlan>| plan.output_partitioning().partition_count();
    let cannot_fail = plan.is::<ProjectionExec>()
        || plan.is::<FilterExec>()
        || plan.is::<RepartitionExec>()
        || plan.is::<CooperativeExec>()
        || plan.is::<AggregateExec>()
        || plan.is::<SortExec>()
        || plan.is::<SortPreservingMergeExec>()
        || plan.is::<UnionExec>()
        || plan.is::<InterleaveExec>()
        || plan.is::<GlobalLimitExec>()
        || plan.is::<LocalLimitExec>()
        || plan.is::<CrossJoinExec>()
        || plan.is::<NestedLoopJoinExec>()
        || plan
            .downcast_ref::<CoalescePartitionsExec>()
            .is_some_and(|coalesce| partitions(coalesce.input()) > 0)
        || plan.downcast_ref::<HashJoinExec>().is_some_and(|join| {
            !has_prepared_build(plan)
                && match join.partition_mode() {
                    PartitionMode::CollectLeft => partitions(join.left()) == 1,
                    PartitionMode::Partitioned => {
                        partitions(join.left()) == partitions(join.right())
                    }
                    PartitionMode::Auto => false,
                }
        });
    cannot_fail && children.into_iter().all(probe_streams_cannot_fail)
}

/// Whether the streams of the scan of files `config` cannot fail to be
/// created: a scan of the local file system, by a Parquet source without
/// virtual columns or decryption properties (which are checked when its
/// streams are created), or by a CSV source.
fn local_scan_streams_cannot_fail(config: &FileScanConfig) -> bool {
    let source = config.file_source();
    config.object_store_url == ObjectStoreUrl::local_filesystem()
        && (source.is::<CsvSource>()
            || (source.table_schema().virtual_columns().is_empty()
                && source
                    .downcast_ref::<ParquetSource>()
                    .is_some_and(|parquet| {
                        parquet
                            .table_parquet_options()
                            .crypto
                            .file_decryption
                            .is_none()
                    })))
}

/// Whether evaluating `expr` on a batch of `schema` can fail on some rows.
/// Conservatively, only expressions built of columns, literals, comparisons,
/// boolean operators, `IS [NOT] NULL`, `CASE`, `IN` lists, `TRY_CAST`,
/// arithmetic that wraps on overflow, divisions by constants other than 0 and
/// -1 and casts that cannot fail cannot fail.
fn can_fail(expr: &PhysicalExprRef, schema: &Schema) -> bool {
    if expr.is::<Column>() || expr.is::<Literal>() {
        return false;
    }
    // A result of a dictionary with narrow keys, or with narrow run ends, is
    // encoded from all the rows of the batch, and fails if they have more
    // distinct values (or rows) than the keys (or run ends) hold, however
    // the values are computed (e.g. by a `TRY_CAST`).
    let narrow_result = expr
        .data_type(schema)
        .map_or(true, |data_type| concatenation_can_overflow(&data_type));
    let infallible = !narrow_result
        && (expr.is::<NotExpr>()
            || expr.is::<IsNullExpr>()
            || expr.is::<IsNotNullExpr>()
            || expr.is::<CaseExpr>()
            || expr.is::<InListExpr>()
            || expr.is::<TryCastExpr>()
            || expr
                .downcast_ref::<BinaryExpr>()
                .is_some_and(|binary| binary_cannot_fail(binary, schema))
            || expr
                .downcast_ref::<CastExpr>()
                .is_some_and(|cast| cast_cannot_fail(cast, schema)));
    !infallible || expr.children().iter().any(|child| can_fail(child, schema))
}

/// Whether the operation of `binary` (not its operands) cannot fail.
fn binary_cannot_fail(binary: &BinaryExpr, schema: &Schema) -> bool {
    let numeric =
        |data_type: &DataType| data_type.is_integer() || data_type.is_floating();
    let Ok(data_type) = binary.data_type(schema) else {
        return false;
    };
    match binary.op() {
        Operator::Eq
        | Operator::NotEq
        | Operator::Lt
        | Operator::LtEq
        | Operator::Gt
        | Operator::GtEq
        | Operator::IsDistinctFrom
        | Operator::IsNotDistinctFrom
        | Operator::And
        | Operator::Or => true,
        // Unless it fails on overflow (not the default), integer arithmetic
        // wraps.
        Operator::Plus | Operator::Minus | Operator::Multiply => {
            let wrapping = BinaryExpr::new(
                Arc::clone(binary.left()),
                *binary.op(),
                Arc::clone(binary.right()),
            );
            numeric(&data_type) && *binary == wrapping
        }
        Operator::Divide | Operator::Modulo => {
            data_type.is_floating()
                || (data_type.is_integer()
                    && binary
                        .right()
                        .downcast_ref::<Literal>()
                        .is_some_and(|literal| safe_integer_divisor(literal.value())))
        }
        _ => false,
    }
}

/// Whether integers can be divided by `value` without failing: by any value
/// but 0 (division by zero) and -1 (overflow of the smallest value), or NULL.
fn safe_integer_divisor(value: &ScalarValue) -> bool {
    match value {
        ScalarValue::Int8(v) => v.is_none_or(|v| v != 0 && v != -1),
        ScalarValue::Int16(v) => v.is_none_or(|v| v != 0 && v != -1),
        ScalarValue::Int32(v) => v.is_none_or(|v| v != 0 && v != -1),
        ScalarValue::Int64(v) => v.is_none_or(|v| v != 0 && v != -1),
        ScalarValue::UInt8(v) => v.is_none_or(|v| v != 0),
        ScalarValue::UInt16(v) => v.is_none_or(|v| v != 0),
        ScalarValue::UInt32(v) => v.is_none_or(|v| v != 0),
        ScalarValue::UInt64(v) => v.is_none_or(|v| v != 0),
        _ => false,
    }
}

/// Whether the cast of `cast` (not its operand) cannot fail: a safe cast
/// (to NULL on failure), or to a type that can represent every value of the
/// input type: a wider integer, a float, a string, or the values of a
/// dictionary.
fn cast_cannot_fail(cast: &CastExpr, schema: &Schema) -> bool {
    // A safe cast casts the values that do not fit to NULL (its results of
    // narrow encodings are checked by `can_fail`).
    if cast.cast_options().safe {
        return true;
    }
    let Ok(from) = cast.expr().data_type(schema) else {
        return false;
    };
    let to = cast.cast_type();
    if &from == to {
        return true;
    }
    if let DataType::Dictionary(_, values) = &from {
        return values.as_ref() == to;
    }
    let bits = |data_type: &DataType| data_type.primitive_width().map(|bytes| bytes * 8);
    match to {
        _ if to.is_integer() && from.is_integer() => {
            // A wider integer type, signed if the input is.
            bits(to) > bits(&from)
                && (to.is_signed_integer() || from.is_unsigned_integer())
        }
        DataType::Float32 | DataType::Float64 => from.is_integer() || from.is_floating(),
        DataType::Utf8 | DataType::LargeUtf8 | DataType::Utf8View => {
            from.is_integer()
                || from.is_floating()
                || matches!(from, DataType::Boolean | DataType::Utf8)
        }
        _ => false,
    }
}

/// The dynamic filters that `plan` produces (see
/// `ExecutionPlan::dynamic_expressions_produced`), `None` if one of them is
/// not a [`DynamicFilterPhysicalExpr`].
fn dynamic_filters(
    plan: &Arc<dyn ExecutionPlan>,
) -> Option<Vec<Arc<DynamicFilterPhysicalExpr>>> {
    plan.dynamic_expressions_produced()
        .into_iter()
        .map(|filter| {
            let filter: Arc<dyn Any + Send + Sync> = filter;
            filter.downcast::<DynamicFilterPhysicalExpr>().ok()
        })
        .collect()
}

/// Whether DataFusion computes the bounds of the build keys of a join of
/// `data_type` (for its dynamic filter) with its `MIN` and `MAX` aggregates
/// without failing: numbers, dates, times, durations and intervals, booleans,
/// strings and binaries, and dictionaries of these.
fn has_bounds(data_type: &DataType) -> bool {
    match data_type {
        DataType::Dictionary(_, values) => has_bounds(values),
        data_type => {
            data_type.is_primitive()
                || matches!(
                    data_type,
                    DataType::Boolean
                        | DataType::Utf8
                        | DataType::LargeUtf8
                        | DataType::Utf8View
                        | DataType::Binary
                        | DataType::LargeBinary
                        | DataType::BinaryView
                        | DataType::FixedSizeBinary(_)
                )
        }
    }
}

/// Whether the values of `data_type` are ordered like they are equal as join
/// keys, so that every key equal to a build key is between the smallest and
/// the largest build key (which [`PhysicalHashJoin::dynamic_filter_predicate`]
/// prunes the probe side with): integers, decimals, dates, times, timestamps,
/// durations, booleans, strings and binaries. Not floating point numbers, of
/// which `-0.0` and `+0.0` are equal keys, but ordered (like the `NaN`s with
/// different signs and payloads) by Arrow's comparisons and statistics, nor
/// dictionaries, intervals and nested values.
fn has_exact_bounds(data_type: &DataType) -> bool {
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

/// Whether every node of `plan` (the probe input of a join that produces a
/// dynamic filter), and every node below it, only evaluates expressions that
/// cannot fail, whatever rows they are evaluated on (see [`is_infallible`]):
/// then which rows of `plan` the dynamic filter prunes makes no difference
/// but to which rows reach the join.
///
/// In DataFusion, the rows that a node of `plan` prunes with the filter (once
/// it is set, see [`PhysicalHashJoin::try_lower`]) are not evaluated by the
/// nodes above that node, nor by the probe sides of the joins whose build
/// side the pruning empties, while the probe side of a native join, which
/// sets the filter to the bounds of its build keys only (see
/// [`PhysicalHashJoin::dynamic_filter_predicate`]), evaluates those of them
/// between the bounds.
///
/// Only the nodes whose expressions are known qualify (see
/// [`evaluated_expressions`]), and the aggregations of aggregate functions
/// that cannot fail (see [`aggregate_functions_cannot_fail`]): DataFusion
/// pushes a dynamic filter through an aggregation on its group keys only, so
/// that it prunes whole groups, which the native engine aggregates too. The
/// dynamic filters that nodes of `plan` read (which their producers may set
/// during the execution) are classified by their children, see
/// [`is_infallible_with_dynamic_filters`].
fn evaluates_only_infallible_expressions(plan: &Arc<dyn ExecutionPlan>) -> bool {
    evaluated_expressions(plan).is_some_and(|expressions| {
        expressions
            .iter()
            .all(|(expr, schema)| is_infallible_with_dynamic_filters(expr, schema))
    }) && aggregate_functions_cannot_fail(plan)
        && plan
            .children()
            .into_iter()
            .all(evaluates_only_infallible_expressions)
}

/// Whether the aggregate functions of `plan`, if it is an aggregation, cannot
/// fail on the values they aggregate: counts, minimums and maximums, first and
/// last values, boolean conjunctions and disjunctions, sums (which wrap on
/// overflow), and averages of integers and floating point numbers (the
/// average of decimals fails if it overflows the precision of its result).
fn aggregate_functions_cannot_fail(plan: &Arc<dyn ExecutionPlan>) -> bool {
    let Some(aggregation) = plan.downcast_ref::<AggregateExec>() else {
        return true;
    };
    // The schema of the input rows, also of an aggregation of partial states.
    let schema = aggregation.input_schema();
    aggregation
        .aggr_expr()
        .iter()
        .all(|aggregate| match aggregate.fun().name() {
            "count" | "min" | "max" | "first_value" | "last_value" | "bool_and"
            | "bool_or" | "sum" => true,
            "avg" => aggregate.expressions().iter().all(|argument| {
                argument.data_type(&schema).is_ok_and(|data_type| {
                    data_type.is_integer() || data_type.is_floating()
                })
            }),
            _ => false,
        })
}

/// The expressions that `plan` itself evaluates on rows, each with the
/// schema of the rows it is evaluated on, but the dynamic filters that a join
/// or an aggregation produces (which the nodes that read them evaluate).
/// `None` if they are not known: for a node with expressions that is not a
/// hash join, an aggregation, a scan of files, a filter, a projection, a
/// repartitioning or a sort (e.g. a window aggregation, or an extension
/// node).
fn evaluated_expressions(
    plan: &Arc<dyn ExecutionPlan>,
) -> Option<Vec<(PhysicalExprRef, SchemaRef)>> {
    if let Some(join) = plan.downcast_ref::<HashJoinExec>() {
        let (left, right) = (join.left().schema(), join.right().schema());
        let keys = join.on().iter().flat_map(|(left_key, right_key)| {
            [
                (Arc::clone(left_key), Arc::clone(&left)),
                (Arc::clone(right_key), Arc::clone(&right)),
            ]
        });
        let filter = join
            .filter()
            .map(|filter| (Arc::clone(filter.expression()), Arc::clone(filter.schema())));
        return Some(keys.chain(filter).collect());
    }
    if let Some(aggregate) = plan.downcast_ref::<AggregateExec>() {
        let mut expressions = aggregate.group_expr().input_exprs();
        // The aggregations of input rows evaluate the arguments (and the
        // orderings and filters) of their aggregates; the aggregations of
        // partial states (`Final`, `FinalPartitioned`, `PartialReduce`) only
        // read the states, by position.
        if matches!(
            aggregate.mode(),
            AggregateMode::Partial
                | AggregateMode::Single
                | AggregateMode::SinglePartitioned
        ) {
            for aggregate in aggregate.aggr_expr() {
                let all = aggregate.all_expressions();
                expressions.extend(all.args);
                expressions.extend(all.order_by_exprs);
            }
            expressions.extend(aggregate.filter_expr().iter().flatten().cloned());
        }
        let input = aggregate.input().schema();
        return Some(
            expressions
                .into_iter()
                .map(|expr| (expr, Arc::clone(&input)))
                .collect(),
        );
    }
    if let Some(scan) = plan.downcast_ref::<DataSourceExec>()
        && let Some(config) = scan.data_source().downcast_ref::<FileScanConfig>()
    {
        // The filter (e.g. the predicate of a Parquet scan) and the
        // projection of a file source are in terms of the table schema.
        let source = config.file_source();
        let projection = source
            .projection()
            .into_iter()
            .flat_map(|projection| projection.iter().map(|expr| Arc::clone(&expr.expr)));
        let schema = source.table_schema().table_schema();
        return Some(
            source
                .filter()
                .into_iter()
                .chain(projection)
                .map(|expr| (expr, Arc::clone(schema)))
                .collect(),
        );
    }
    // Any other node: the expressions that it evaluates or updates.
    let mut expressions = vec![];
    plan.apply_expressions(&mut |expr| {
        expressions.push(Arc::clone(expr));
        Ok(TreeNodeRecursion::Continue)
    })
    .ok()?;
    if expressions.is_empty() {
        return Some(vec![]);
    }
    // These evaluate their expressions on the rows of their input (a `TopK`
    // also updates its dynamic filter, a predicate of its sort keys).
    let on_input = plan.is::<FilterExec>()
        || plan.is::<ProjectionExec>()
        || plan.is::<RepartitionExec>()
        || plan.is::<SortExec>()
        || plan.is::<SortPreservingMergeExec>();
    match plan.children().as_slice() {
        [input] if on_input => Some(
            expressions
                .into_iter()
                .map(|expr| (expr, input.schema()))
                .collect(),
        ),
        _ => None,
    }
}

/// Whether evaluating `expr` on rows of `schema` cannot fail (see
/// [`is_infallible`]), the dynamic filters of `expr` included: a dynamic
/// filter evaluates a predicate of its children, the keys of the node that
/// produces it (bounds, an `IN` list or a hash table lookup of them, for a
/// join; comparisons of its sort keys, for a `TopK`), or `true`, so it cannot
/// fail if its children cannot.
fn is_infallible_with_dynamic_filters(expr: &PhysicalExprRef, schema: &Schema) -> bool {
    // The columns of `expr` must be those of `schema`, the schema of the rows
    // that `expr` is evaluated on.
    let other_columns = expr.exists(|expr| {
        Ok(expr.downcast_ref::<Column>().is_some_and(|column| {
            schema
                .fields()
                .get(column.index())
                .is_none_or(|field| field.name() != column.name())
        }))
    });
    if other_columns.unwrap_or(true) {
        return false;
    }
    let without_dynamic_filters = Arc::clone(expr).transform_up(|expr| {
        let Some(filter) = expr.downcast_ref::<DynamicFilterPhysicalExpr>() else {
            return Ok(Transformed::no(expr));
        };
        // Evaluates its children (and nothing that can fail).
        let children = filter.children().into_iter().map(|child| {
            Arc::new(IsNotNullExpr::new(Arc::clone(child))) as PhysicalExprRef
        });
        Ok(Transformed::yes(conjunction(children)))
    });
    without_dynamic_filters
        .data()
        .is_ok_and(|expr| is_infallible(&expr, schema))
}

/// Whether a `HashJoinExec` has a prepared build side attached, whose left
/// input is a placeholder that is not executed. (The prepared build is only
/// visible in the display of the node.)
fn has_prepared_build(plan: &Arc<dyn ExecutionPlan>) -> bool {
    displayable(plan.as_ref())
        .one_line()
        .to_string()
        .contains(", prepared_build=")
}

/// `plan`, an input of a join whose join keys on its side are `keys`,
/// without the nodes that only distribute its rows over partitions:
/// `CoalescePartitionsExec`s without a limit, and `RepartitionExec`s that do
/// not preserve the order and only evaluate join keys (see
/// [`partitions_by_keys`]).
fn without_redistribution<'a>(
    plan: &'a Arc<dyn ExecutionPlan>,
    keys: &[&PhysicalExprRef],
) -> &'a Arc<dyn ExecutionPlan> {
    let mut plan = plan;
    loop {
        if let Some(repartition) = plan.downcast_ref::<RepartitionExec>()
            && !repartition.preserve_order()
            && partitions_by_keys(
                repartition.partitioning(),
                keys,
                &repartition.input().schema(),
            )
        {
            plan = repartition.input();
        } else if let Some(coalesce) = plan.downcast_ref::<CoalescePartitionsExec>()
            && coalesce.fetch().is_none()
            // Without input partitions, DataFusion fails to execute it.
            && coalesce.input().output_partitioning().partition_count() > 0
        {
            plan = coalesce.input();
        } else {
            return plan;
        }
    }
}

/// Whether distributing the rows of an input of a join (of `schema`) with
/// `partitioning` evaluates nothing but join keys of that input (`keys`)
/// that cannot fail (see [`can_fail`]): round robin partitioning, or hash
/// partitioning by such keys, which the join evaluates on all rows too. Any
/// other expression, e.g. a division by zero, is left to DataFusion: a
/// repartitioning evaluates it on every batch as it arrives, and the errors
/// of the build keys of the join may only be raised once its whole build
/// side is collected (see `JoinSpec::defers_build_key_errors`).
fn partitions_by_keys(
    partitioning: &Partitioning,
    keys: &[&PhysicalExprRef],
    schema: &Schema,
) -> bool {
    match partitioning {
        Partitioning::RoundRobinBatch(_) => true,
        Partitioning::Hash(exprs, _) => exprs.iter().all(|expr| {
            keys.iter().any(|key| key.as_ref() == expr.as_ref())
                && !can_fail(expr, schema)
        }),
        _ => false,
    }
}

/// Whether concatenating arrays of `data_type`, as DataFusion concatenates
/// its build side, can overflow the type for realistic numbers of rows: with
/// dictionary keys of 8 or 16 bits (more distinct values than they hold), or
/// run ends of 16 bits (more rows than they count). (The 32-bit offsets of
/// strings or lists overflow beyond 2 GiB of data, which is not considered.)
fn concatenation_can_overflow(data_type: &DataType) -> bool {
    match data_type {
        DataType::Dictionary(keys, values) => {
            matches!(
                keys.as_ref(),
                DataType::Int8 | DataType::UInt8 | DataType::Int16 | DataType::UInt16
            ) || concatenation_can_overflow(values)
        }
        DataType::RunEndEncoded(run_ends, values) => {
            matches!(run_ends.data_type(), DataType::Int16)
                || concatenation_can_overflow(values.data_type())
        }
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::ListView(field)
        | DataType::LargeListView(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => concatenation_can_overflow(field.data_type()),
        DataType::Struct(fields) => fields
            .iter()
            .any(|field| concatenation_can_overflow(field.data_type())),
        DataType::Union(fields, _) => fields
            .iter()
            .any(|(_, field)| concatenation_can_overflow(field.data_type())),
        _ => false,
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

/// The global state of a join, shared by all its roles: its global sink
/// state.
struct JoinState {
    /// The build rows, until the hash table is built.
    build: Mutex<BuildPhase>,
    /// The hash table, once the join is finalized.
    table: OnceLock<Arc<JoinHashTable>>,
    radix_bits: u32,
    /// The number of build rows, and of build rows with a NULL key (under
    /// `NullEqualsNothing`).
    build_rows: AtomicUsize,
    build_null_key_rows: AtomicUsize,
    /// What the probe side of a null-aware `LeftAnti` or `LeftMark` join
    /// saw: rows, and a NULL key.
    probe_has_rows: AtomicBool,
    probe_has_null_key: AtomicBool,
    /// Whether the streams of the DataFusion probe input were created (see
    /// [`PhysicalHashJoin::create_probe_streams`]), or the error of creating
    /// them.
    probe_streams: OnceLock<std::result::Result<(), Arc<DataFusionError>>>,
    /// The first error of evaluating the build keys, if DataFusion raises it
    /// only once the whole build side is collected (see
    /// `JoinSpec::defers_build_key_errors`): in `finalize`.
    deferred_build_error: Mutex<Option<DataFusionError>>,
    /// The range and the bounds of the build keys, combined by the threads of
    /// the build pipelines.
    key_stats: Mutex<BuildKeyStats>,
    /// The memory of the build rows.
    reservation: MemoryReservation,
}

impl GlobalSinkState for JoinState {}

/// What the build computes about the build keys as it collects the rows (see
/// [`JoinSpec::tracks_key_range`] and [`JoinSpec::bounded_keys`]): the range
/// of a single integer key, for the [`JoinHashTable`], and the bounds of the
/// keys, for the dynamic filters.
#[derive(Debug, Default)]
struct BuildKeyStats {
    /// The range of the valid values of the single key.
    range: Option<KeyRange>,
    /// The bounds of every key (of the bounded ones).
    bounds: Vec<KeyBounds>,
}

/// The bounds of a build key.
#[derive(Debug, Default, Clone)]
struct KeyBounds {
    /// The smallest and the largest valid value, once one was seen.
    min_max: Option<(ScalarValue, ScalarValue)>,
    /// Whether a NULL value was seen.
    has_null: bool,
    /// Whether the bounds could not be computed (then the key has none).
    failed: bool,
}

impl BuildKeyStats {
    fn new(spec: &JoinSpec) -> Self {
        Self {
            range: None,
            bounds: vec![KeyBounds::default(); spec.num_keys()],
        }
    }

    /// Add the build keys `keys` (one array per key, of the rows of a chunk).
    fn update(&mut self, spec: &JoinSpec, keys: &[ArrayRef]) {
        if spec.tracks_key_range
            && let Some(range) = keys
                .first()
                .and_then(|key| KeyRange::of_array(key.as_ref()))
        {
            self.range = Some(self.range.map_or(range, |current| current.union(range)));
        }
        for ((bounds, key), _) in self
            .bounds
            .iter_mut()
            .zip(keys)
            .zip(&spec.bounded_keys)
            .filter(|(_, bounded)| **bounded)
        {
            bounds.update(key);
        }
    }

    /// Add the statistics of the keys of other build rows.
    fn merge(&mut self, other: BuildKeyStats) {
        self.range = match (self.range, other.range) {
            (Some(range), Some(other)) => Some(range.union(other)),
            (range, other) => range.or(other),
        };
        for (bounds, other) in self.bounds.iter_mut().zip(other.bounds) {
            bounds.merge(other);
        }
    }
}

impl KeyBounds {
    fn update(&mut self, key: &ArrayRef) {
        if self.failed {
            return;
        }
        self.has_null |= key.logical_null_count() > 0;
        // The `MIN` and `MAX` aggregates of DataFusion (which ignore NULLs).
        match (min_batch(key), max_batch(key)) {
            (Ok(min), Ok(max)) if !min.is_null() && !max.is_null() => {
                self.add(min, max);
            }
            (Ok(_), Ok(_)) => {}
            _ => self.failed = true,
        }
    }

    fn add(&mut self, min: ScalarValue, max: ScalarValue) {
        let Some((current_min, current_max)) = &mut self.min_max else {
            self.min_max = Some((min, max));
            return;
        };
        match (min.partial_cmp(current_min), max.partial_cmp(current_max)) {
            (Some(min_order), Some(max_order)) => {
                if min_order.is_lt() {
                    *current_min = min;
                }
                if max_order.is_gt() {
                    *current_max = max;
                }
            }
            _ => self.failed = true,
        }
    }

    fn merge(&mut self, other: KeyBounds) {
        self.has_null |= other.has_null;
        self.failed |= other.failed;
        if let Some((min, max)) = other.min_max
            && !self.failed
        {
            self.add(min, max);
        }
    }
}

enum BuildPhase {
    /// The rows combined by the threads of the build pipelines.
    Collecting(Box<BuildRows>),
    /// The pointer tables are being built.
    Building(Arc<JoinHashTableBuilder>),
    /// The table is built.
    Done,
}

/// The build rows combined by the threads of the build pipelines.
struct BuildRows {
    data: PartitionedTupleData,
    /// The reservation of the memory of the hash table that will index the
    /// rows.
    index_reservation: MemoryReservation,
}

impl JoinState {
    /// Start building the hash table of the combined build rows, unless it
    /// is being built already.
    fn start_build(&self, spec: &JoinSpec) -> Result<Arc<JoinHashTableBuilder>> {
        let mut build = self.build.lock();
        match std::mem::replace(&mut *build, BuildPhase::Done) {
            BuildPhase::Collecting(rows) => {
                let BuildRows {
                    data,
                    index_reservation,
                } = *rows;
                // A small build side gets a single table, built by one thread.
                let mut builder = JoinHashTableBuilder::try_new(
                    data,
                    spec.num_keys(),
                    spec.null_equality,
                    index_reservation,
                )?
                .single_partition_up_to(SMALL_BUILD_ROWS);
                // A single integer key with a range is indexed without hashing.
                if let Some(range) = self.key_stats.lock().range {
                    builder = builder.with_key_range(range)?;
                }
                let builder = Arc::new(builder);
                *build = BuildPhase::Building(Arc::clone(&builder));
                Ok(builder)
            }
            BuildPhase::Building(builder) => {
                *build = BuildPhase::Building(Arc::clone(&builder));
                Ok(builder)
            }
            BuildPhase::Done => internal_err!("The hash table of the join was built"),
        }
    }

    /// Build the tables that were not built yet, and publish the hash table.
    fn finish_build(&self, spec: &JoinSpec) -> Result<Arc<JoinHashTable>> {
        let builder = self.start_build(spec)?;
        *self.build.lock() = BuildPhase::Done;
        let builder = Arc::try_unwrap(builder).map_err(|_| {
            internal_datafusion_err!("The hash table of the join is still being built")
        })?;
        let table = Arc::new(builder.finish(spec.emits_final_build_rows())?);
        self.table
            .set(Arc::clone(&table))
            .map_err(|_| internal_datafusion_err!("The join was finalized twice"))?;
        Ok(table)
    }

    fn table(&self) -> Result<Arc<JoinHashTable>> {
        self.table.get().map(Arc::clone).ok_or_else(|| {
            internal_datafusion_err!("The hash table of the join is not built")
        })
    }
}

/// The state of a thread of a build pipeline.
#[derive(Default)]
struct BuildLocalState {
    /// The rows of the thread, created on its first chunk.
    data: Option<PartitionedTupleData>,
    hashes: Vec<u64>,
    /// The memory of `data` reserved so far.
    reserved: usize,
    /// The range and the bounds of the build keys of the thread.
    key_stats: BuildKeyStats,
}

impl LocalSinkState for BuildLocalState {}

/// The state of a thread of the probe pipeline.
struct ProbeState {
    join: Arc<JoinState>,
    table: Arc<JoinHashTable>,
    /// The probe chunk being joined, until all its output was emitted.
    current: Option<ProbeChunk>,
    /// The matches of the rows of the current probe chunk.
    scan: ChainScan,
    /// The hashes of the keys of the current probe chunk.
    hashes: Vec<u64>,
    /// The pairs of the current batch: probe rows and matching build rows.
    probe_rows: Vec<u32>,
    matches: Vec<RowLocation>,
}

impl OperatorState for ProbeState {}

/// A probe chunk being joined. Its (logical) rows are the probe rows of the
/// [`ChainScan`] of their matches.
struct ProbeChunk {
    /// The physical columns of the chunk.
    columns: Vec<ArrayRef>,
    /// The physical row of every probe row, if the chunk was not compacted
    /// (see [`PhysicalHashJoin::start_chunk`]).
    selection: Option<SelectionVector>,
    /// The number of physical rows of the columns.
    physical_len: usize,
    /// Whether the probe row in progress of the scan of the matches found a
    /// match in a previous batch.
    matched_in_progress: bool,
    /// For a null-aware `RightAnti` join with a non-empty build side: the
    /// rows with a valid key (the others are never emitted).
    excluded: Option<NullBuffer>,
}

impl ProbeChunk {
    /// Whether the unmatched probe row `row` is emitted (by a `RightAnti`
    /// join).
    fn emits_unmatched(&self, row: usize) -> bool {
        self.excluded
            .as_ref()
            .is_none_or(|valid| valid.is_valid(row))
    }

    /// The physical column `index`.
    fn column(&self, index: usize) -> Result<&ArrayRef> {
        self.columns.get(index).ok_or_else(|| {
            internal_datafusion_err!("Missing probe column {index} of the join")
        })
    }

    /// Replace the probe rows `rows` by their physical rows.
    fn to_physical(&self, rows: &mut [u32]) {
        if let Some(selection) = &self.selection {
            let physical = selection.indices();
            for row in rows {
                *row = physical[*row as usize];
            }
        }
    }

    /// The selection of the physical rows `rows` of the columns: `None` if
    /// they are all physical rows, in order.
    fn selection_of(&self, rows: Vec<u32>) -> Option<SelectionVector> {
        let all = rows.len() == self.physical_len
            && rows.iter().enumerate().all(|(i, &row)| row as usize == i);
        (!all).then(|| SelectionVector::from(rows))
    }
}

/// The global state of the scan of the final build rows.
struct FinalScanState {
    table: Arc<JoinHashTable>,
    /// The next block of build rows to scan.
    next_block: AtomicUsize,
    probe_has_rows: bool,
    probe_has_null_key: bool,
}

impl GlobalSourceState for FinalScanState {
    fn max_threads(&self) -> usize {
        self.table.collection().num_blocks().max(1)
    }
}

/// The state of a thread of the scan of the final build rows.
#[derive(Default)]
struct FinalScanLocalState {
    /// The block being scanned, and the next row to scan in it.
    block: Option<usize>,
    row: usize,
}

impl LocalSourceState for FinalScanLocalState {}

impl PhysicalOperator for PhysicalHashJoin {
    fn name(&self) -> &str {
        "PhysicalHashJoin"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.spec.schema)
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.children
    }

    fn description(&self) -> String {
        let spec = &self.spec;
        let on = spec
            .on
            .iter()
            .map(|(build, probe)| format!("({build}, {probe})"))
            .collect::<Vec<_>>()
            .join(", ");
        let mut description =
            format!("PhysicalHashJoin: join_type={}, on=[{on}]", spec.join_type);
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
        if spec.null_equality == NullEquality::NullEqualsNull {
            description.push_str(", NullsEqual: true");
        }
        if spec.null_aware {
            description.push_str(", null_aware");
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
        let radix_bits = radix_bits_for_threads(ctx.num_threads);
        let data =
            PartitionedTupleData::try_new(Arc::clone(&self.spec.layout), radix_bits)?;
        let pool = ctx.task_ctx.memory_pool();
        let reservation = MemoryConsumer::new("PhysicalHashJoin").register(pool);
        // The hash table reserves its memory before it allocates it.
        let index_reservation = MemoryConsumer::new("JoinHashTable").register(pool);
        Ok(Arc::new(JoinState {
            build: Mutex::new(BuildPhase::Collecting(Box::new(BuildRows {
                data,
                index_reservation,
            }))),
            table: OnceLock::new(),
            radix_bits,
            build_rows: AtomicUsize::new(0),
            build_null_key_rows: AtomicUsize::new(0),
            probe_has_rows: AtomicBool::new(false),
            probe_has_null_key: AtomicBool::new(false),
            probe_streams: OnceLock::new(),
            deferred_build_error: Mutex::new(None),
            key_stats: Mutex::new(BuildKeyStats::new(&self.spec)),
            reservation,
        }))
    }

    fn get_local_sink_state(
        &self,
        ctx: &ExecutionContext,
        gstate: &dyn GlobalSinkState,
    ) -> Result<Box<dyn LocalSinkState>> {
        let state = downcast_state::<JoinState>(gstate.as_any(), self.name())?;
        // `HashJoinExec::execute` creates the streams of its probe side right
        // away, after the stream of its build side and before reading either:
        // the first thread of a build pipeline creates them once its source
        // is created, while the other threads wait, before any of them reads
        // anything, so that the join raises the same errors, also if its build
        // side never completes.
        state
            .probe_streams
            .get_or_init(|| self.create_probe_streams(ctx).map_err(Arc::new))
            .as_ref()
            .map_err(|e| DataFusionError::Shared(Arc::clone(e)))?;
        Ok(Box::new(BuildLocalState {
            key_stats: BuildKeyStats::new(&self.spec),
            ..Default::default()
        }))
    }

    fn sink(
        &self,
        _ctx: &ExecutionContext,
        input: &DataChunk,
        gstate: &dyn GlobalSinkState,
        lstate: &mut dyn LocalSinkState,
    ) -> Result<SinkResult> {
        let spec = &self.spec;
        let state = downcast_state::<JoinState>(gstate.as_any(), self.name())?;
        let local =
            downcast_state_mut::<BuildLocalState>(lstate.as_any_mut(), self.name())?;
        // Once the build keys failed, and DataFusion raises that only when
        // the build side is complete, the build rows are not needed any more.
        if spec.defers_build_key_errors && state.deferred_build_error.lock().is_some() {
            return Ok(SinkResult::NeedMoreInput);
        }
        // Evaluate the keys on the logical rows only.
        let batch = input.compact()?;
        let num_rows = batch.num_rows();
        let keys = match evaluate_expressions_to_arrays(
            spec.on.iter().map(|(build, _)| build),
            &batch,
        ) {
            Ok(keys) => keys,
            Err(e) if spec.defers_build_key_errors => {
                state.deferred_build_error.lock().get_or_insert(e);
                return Ok(SinkResult::NeedMoreInput);
            }
            Err(e) => return Err(e),
        };
        state.build_rows.fetch_add(num_rows, Ordering::Relaxed);
        local.key_stats.update(spec, &keys);
        let valid = spec.valid_keys(&keys);
        if let Some(valid) = &valid {
            state
                .build_null_key_rows
                .fetch_add(valid.null_count(), Ordering::Relaxed);
        }
        hash_join_keys(&keys, num_rows, &mut local.hashes)?;
        let mut columns = keys;
        columns.extend(
            spec.payload
                .iter()
                .map(|&column| Arc::clone(batch.column(column))),
        );
        let mut chunk =
            DataChunk::try_new(Arc::clone(spec.layout.schema()), columns, None)?;
        // The rows that cannot match are only needed by joins that emit the
        // unmatched build rows.
        if let Some(valid) = valid.filter(|_| !spec.emits_unmatched_build_rows()) {
            let selection = SelectionVector::from(
                valid
                    .valid_indices()
                    .map(|row| row as u32)
                    .collect::<Vec<_>>(),
            );
            if selection.is_empty() {
                return Ok(SinkResult::NeedMoreInput);
            }
            let hashes = selection
                .iter()
                .map(|row| local.hashes[row])
                .collect::<Vec<_>>();
            local.hashes = hashes;
            chunk = chunk.with_selection(selection)?;
        }
        let data = match &mut local.data {
            Some(data) => data,
            None => local.data.insert(PartitionedTupleData::try_new(
                Arc::clone(&spec.layout),
                state.radix_bits,
            )?),
        };
        data.append_chunk(&chunk, &local.hashes)?;
        let size = data.memory_size();
        if size > local.reserved {
            state.reservation.try_grow(size - local.reserved)?;
            local.reserved = size;
        }
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
        state
            .key_stats
            .lock()
            .merge(std::mem::take(&mut local.key_stats));
        let Some(data) = local.data.take() else {
            return Ok(());
        };
        match &mut *state.build.lock() {
            BuildPhase::Collecting(rows) => rows.data.combine(data),
            _ => internal_err!("PhysicalHashJoin received build rows after its build"),
        }
    }

    fn finalize(
        &self,
        ctx: &ExecutionContext,
        gstate: &dyn GlobalSinkState,
    ) -> Result<SinkFinalize> {
        let state = downcast_state::<JoinState>(gstate.as_any(), self.name())?;
        let result = self.finalize_build(ctx, state);
        // The dynamic filters are set (if the table was built, see
        // `finalize_build`): they will not change any more, whatever the build
        // side turned out to be.
        for filter in &self.dynamic_filters {
            filter.mark_complete();
        }
        result
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
        let join = self.join_state(ctx)?;
        let table = join.table()?;
        Ok(Box::new(ProbeState {
            join,
            table,
            current: None,
            scan: ChainScan::default(),
            hashes: Vec::new(),
            probe_rows: Vec::new(),
            matches: Vec::new(),
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
            // A new input chunk.
            match self.start_chunk(state, input)? {
                Some(chunk) => state.current = Some(chunk),
                None => return Ok(OperatorResult::NeedMoreInput),
            }
        }
        self.probe_next(state, output)?;
        Ok(match state.current {
            Some(_) => OperatorResult::HaveMoreOutput,
            None => OperatorResult::NeedMoreInput,
        })
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
        let join = self.join_state(ctx)?;
        Ok(Arc::new(FinalScanState {
            table: join.table()?,
            next_block: AtomicUsize::new(0),
            probe_has_rows: join.probe_has_rows.load(Ordering::Relaxed),
            probe_has_null_key: join.probe_has_null_key.load(Ordering::Relaxed),
        }))
    }

    fn get_local_source_state(
        &self,
        _ctx: &ExecutionContext,
        _gstate: &dyn GlobalSourceState,
    ) -> Result<Box<dyn LocalSourceState>> {
        Ok(Box::new(FinalScanLocalState::default()))
    }

    fn get_data(
        &self,
        _ctx: &ExecutionContext,
        gstate: &dyn GlobalSourceState,
        lstate: &mut dyn LocalSourceState,
    ) -> Result<SourceResult> {
        let global = downcast_state::<FinalScanState>(gstate.as_any(), self.name())?;
        let local =
            downcast_state_mut::<FinalScanLocalState>(lstate.as_any_mut(), self.name())?;
        Ok(match self.next_final_rows(global, local)? {
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
        // The join probes the hash table in the pipeline of its probe side.
        meta.add_operator(current, Arc::clone(&this))?;
        let Some(last_pipeline) = meta.pipelines().last().copied() else {
            return internal_err!("A meta pipeline without pipelines");
        };
        let build_meta = {
            // The build side sinks into the join, in a child meta pipeline,
            // whose last pipeline builds the pointer tables in parallel.
            let (mut build, mut build_pipeline) =
                meta.create_child_meta_pipeline(current, Arc::clone(&this))?;
            let build_meta = build.id();
            let finalize_pipeline =
                build.create_union_pipeline(&build_pipeline, false)?;
            build.set_source(
                &finalize_pipeline,
                Arc::new(JoinHashTableBuild::new(Arc::clone(&this))),
            )?;
            // The probe streams of a join lowered from a `HashJoinExec` are
            // created as soon as the execution starts, in a pipeline that
            // depends on nothing (unless creating them cannot fail).
            if self.probe_plan.is_some() && !self.probe_streams_cannot_fail {
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
            for pipeline in build.pipelines() {
                if pipeline != finalize_pipeline {
                    build.add_dependency(&finalize_pipeline, &pipeline)?;
                }
            }
            build_meta
        };
        // The probe side continues the current pipeline. DataFusion reads it
        // only once the build side is collected: the pipelines of its
        // pipeline breakers (e.g. the build side of a join below) wait for
        // the build side to be finalized, so that they do not run (nor fail)
        // if the join cannot produce output, or is canceled before.
        meta.build_with_dependency(current, self.probe_side(), build_meta)?;
        // The build rows emitted after probing, in a pipeline that runs once
        // all the pipelines of the probe side completed.
        if self.spec.emits_final_build_rows() {
            meta.create_child_pipeline(current, this, &last_pipeline)?;
        }
        Ok(())
    }
}

/// The source of the pipeline that builds the pointer tables of a
/// [`PhysicalHashJoin`] in parallel, once all its build rows are combined:
/// every call of `get_data` builds the table of one radix partition (or the
/// dense index of an integer key, see [`JoinHashTableBuilder::with_key_range`])
/// and yields, as building a large partition takes a while, until all
/// partitions are claimed. It runs on as many threads as the builder has work
/// for (see [`JoinHashTableBuilder::max_threads`]): one for a small build
/// side. It produces no rows; the join, the sink of the pipeline, assembles the
/// [`JoinHashTable`] when it is finalized.
#[derive(Debug)]
struct JoinHashTableBuild {
    /// The join (a [`PhysicalHashJoin`]).
    join: Arc<dyn PhysicalOperator>,
    schema: SchemaRef,
}

impl JoinHashTableBuild {
    fn new(join: Arc<dyn PhysicalOperator>) -> Self {
        Self {
            join,
            schema: Arc::new(Schema::empty()),
        }
    }
}

/// The global state of a [`JoinHashTableBuild`].
struct JoinHashTableBuildState {
    builder: Arc<JoinHashTableBuilder>,
}

impl GlobalSourceState for JoinHashTableBuildState {
    fn max_threads(&self) -> usize {
        self.builder.max_threads()
    }
}

impl PhysicalOperator for JoinHashTableBuild {
    fn name(&self) -> &str {
        "JoinHashTableBuild"
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
        ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalSourceState>> {
        let Some(join) = self.join.downcast_ref::<PhysicalHashJoin>() else {
            return internal_err!(
                "JoinHashTableBuild builds the table of a PhysicalHashJoin"
            );
        };
        let builder = join_state(ctx, join)?.start_build(&join.spec)?;
        Ok(Arc::new(JoinHashTableBuildState { builder }))
    }

    fn get_data(
        &self,
        _ctx: &ExecutionContext,
        gstate: &dyn GlobalSourceState,
        _lstate: &mut dyn LocalSourceState,
    ) -> Result<SourceResult> {
        let state =
            downcast_state::<JoinHashTableBuildState>(gstate.as_any(), self.name())?;
        Ok(if state.builder.build_next_partition()? {
            // Let other tasks run between the partitions.
            SourceResult::Blocked(TaskInterrupt::yield_now())
        } else {
            SourceResult::Finished
        })
    }
}

/// The source of a pipeline of the build meta pipeline of a
/// [`PhysicalHashJoin`] lowered from a `HashJoinExec` whose probe streams may
/// fail to be created (see [`probe_streams_cannot_fail`]), which depends on
/// nothing, so that it runs as soon as the execution starts. It produces no
/// rows: its sink, the join, creates the streams of its DataFusion probe
/// input when it creates the local sink state of the pipeline (see
/// [`PhysicalHashJoin::create_probe_streams`]), as `HashJoinExec::execute`
/// does right away, also when the build pipelines wait for pipeline breakers
/// of the build side.
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

    use std::fmt;
    use std::ops::Range;
    use std::time::Duration;

    use arrow::array::{
        AsArray, Decimal128Array, Float64Array, Int32Array, Int64Array, StringArray,
        StringViewArray, TimestampNanosecondArray,
    };
    use arrow::compute::filter_record_batch;
    use arrow::datatypes::{DataType, Int64Type, TimeUnit};
    use datafusion::prelude::{ParquetReadOptions, SessionConfig, SessionContext};
    use datafusion_datasource::PartitionedFile;
    use datafusion_datasource::file_scan_config::FileScanConfigBuilder;
    use datafusion_datasource::memory::MemorySourceConfig;
    use datafusion_datasource::source::DataSourceExec;
    use datafusion_datasource_csv::source::CsvSource;
    use datafusion_execution::SendableRecordBatchStream;
    use datafusion_execution::TaskContext;
    use datafusion_execution::memory_pool::{GreedyMemoryPool, MemoryPool};
    use datafusion_execution::object_store::{
        DefaultObjectStoreRegistry, ObjectStoreRegistry, ObjectStoreUrl,
    };
    use datafusion_execution::runtime_env::RuntimeEnvBuilder;
    use datafusion_expr::Operator;
    use datafusion_physical_expr::expressions::{BinaryExpr, col, lit};
    use datafusion_physical_expr::{LexOrdering, PhysicalExpr, PhysicalSortExpr};
    use datafusion_physical_optimizer::PhysicalOptimizerRule;
    use datafusion_physical_optimizer::filter_pushdown::FilterPushdown;
    use datafusion_physical_plan::collect;
    use datafusion_physical_plan::empty::EmptyExec;
    use datafusion_physical_plan::filter::FilterExec;
    use datafusion_physical_plan::joins::HashJoinExecBuilder;
    use datafusion_physical_plan::projection::ProjectionExec;
    use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
    use datafusion_physical_plan::streaming::{PartitionStream, StreamingTableExec};
    use datafusion_physical_plan::union::UnionExec;
    use futures::StreamExt;

    use crate::bridge::{BridgeOperator, PipelineRootExec, bridged_plans};
    use crate::executor::Executor;
    use crate::explain::{format_operator_tree, format_pipelines};
    use crate::join::hash_table::hashes_are_forced_to_collide;
    use crate::operators::union::PhysicalUnion;
    use crate::operators::union::test_support::{
        AfterBatch, TestSource, UNREGISTERED_STORE, assert_lowered_plan_byte_eq,
        assert_lowered_plan_fails_like_datafusion, batch_plan, count, pending_plan,
        unregistered_store_scan,
    };
    use crate::sources::test_harness::write_parquet;
    use crate::test_utils::{
        EngineOutput, assert_byte_eq_with_datafusion_with_config, compare_outputs,
        datafusion_context, register_test_tables, run_sql, test_session_config,
    };

    // ---------------------------------------------------------------------
    // Lowering
    // ---------------------------------------------------------------------

    /// Lower `plan` with native hash joins, standing in for the planner:
    ///
    /// * `HashJoinExec`s with [`PhysicalHashJoin::try_lower`], and unions
    ///   with [`PhysicalUnion::try_lower`];
    /// * `CoalescePartitionsExec`s without a fetch and the `RepartitionExec`s
    ///   that do not preserve the order and do not evaluate any expression
    ///   that can fail (round robin, or hash partitioning by columns) are
    ///   removed (the pipeline engine merges the partitions of their input
    ///   anyway);
    /// * `ProjectionExec`s and `FilterExec`s (without a fetch) with
    ///   [`TestMap`]s, minimal stand-ins for the native operators, so that the
    ///   joins below them are lowered too;
    /// * scans with the native sources;
    /// * every other node is bridged, with its subtree.
    fn lower(plan: &Arc<dyn ExecutionPlan>) -> Result<Arc<dyn PhysicalOperator>> {
        if let Some(join) = PhysicalHashJoin::try_lower(plan, lower)? {
            return Ok(Arc::new(join));
        }
        if let Some(union) = PhysicalUnion::try_lower(plan, lower)? {
            return Ok(Arc::new(union));
        }
        if let Some(coalesce) = plan.downcast_ref::<CoalescePartitionsExec>()
            && coalesce.fetch().is_none()
            && coalesce.input().output_partitioning().partition_count() > 0
        {
            return lower(coalesce.input());
        }
        if let Some(repartition) = plan.downcast_ref::<RepartitionExec>()
            && !repartition.preserve_order()
            && match repartition.partitioning() {
                Partitioning::RoundRobinBatch(_) => true,
                Partitioning::Hash(exprs, _) => {
                    exprs.iter().all(|expr| expr.is::<Column>())
                }
                _ => false,
            }
        {
            return lower(repartition.input());
        }
        if let Some(projection) = plan.downcast_ref::<ProjectionExec>() {
            let exprs = projection
                .expr()
                .iter()
                .map(|expr| Arc::clone(&expr.expr))
                .collect::<Vec<_>>();
            let schema = projection.schema();
            return Ok(Arc::new(TestMap {
                name: "TestProjection",
                children: vec![lower(projection.input())?],
                schema: Arc::clone(&schema),
                map: Box::new(move |batch| {
                    let columns = exprs
                        .iter()
                        .map(|expr| expr.evaluate(batch)?.into_array(batch.num_rows()))
                        .collect::<Result<Vec<_>>>()?;
                    let options =
                        RecordBatchOptions::new().with_row_count(Some(batch.num_rows()));
                    Ok(RecordBatch::try_new_with_options(
                        Arc::clone(&schema),
                        columns,
                        &options,
                    )?)
                }),
            }));
        }
        if let Some(filter) = plan.downcast_ref::<FilterExec>()
            && filter.fetch().is_none()
        {
            let predicate = Arc::clone(filter.predicate());
            let projection = filter.projection().as_deref().map(<[usize]>::to_vec);
            let schema = filter.schema();
            return Ok(Arc::new(TestMap {
                name: "TestFilter",
                children: vec![lower(filter.input())?],
                schema: Arc::clone(&schema),
                map: Box::new(move |batch| {
                    let mask = predicate.evaluate(batch)?.into_array(batch.num_rows())?;
                    let filtered = filter_record_batch(batch, as_boolean_array(&mask)?)?;
                    let columns = match &projection {
                        Some(projection) => projection
                            .iter()
                            .map(|&i| Arc::clone(filtered.column(i)))
                            .collect(),
                        None => filtered.columns().to_vec(),
                    };
                    let options = RecordBatchOptions::new()
                        .with_row_count(Some(filtered.num_rows()));
                    Ok(RecordBatch::try_new_with_options(
                        Arc::clone(&schema),
                        columns,
                        &options,
                    )?)
                }),
            }));
        }
        if let Some(source) = crate::sources::try_create_scan_source(plan)? {
            return Ok(source);
        }
        Ok(Arc::new(BridgeOperator::new(Arc::clone(plan))))
    }

    type MapFn = Box<dyn Fn(&RecordBatch) -> Result<RecordBatch> + Send + Sync>;

    /// A streaming operator that maps the compacted input chunks with `map`.
    struct TestMap {
        name: &'static str,
        children: Vec<Arc<dyn PhysicalOperator>>,
        schema: SchemaRef,
        map: MapFn,
    }

    impl fmt::Debug for TestMap {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            f.debug_struct(self.name).finish_non_exhaustive()
        }
    }

    impl PhysicalOperator for TestMap {
        fn name(&self) -> &str {
            self.name
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
            *output = DataChunk::from_record_batch((self.map)(&input.compact()?)?);
            Ok(OperatorResult::NeedMoreInput)
        }
    }

    // ---------------------------------------------------------------------
    // Differential checks
    // ---------------------------------------------------------------------

    async fn with_timeout<T>(future: impl Future<Output = T>) -> T {
        // With `force_hash_collisions`, all keys collide, which makes the
        // joins of both engines quadratic in the number of distinct keys
        // (and the native build single-threaded: all rows are in one
        // partition).
        let secs = if hashes_are_forced_to_collide() {
            600
        } else {
            60
        };
        tokio::time::timeout(Duration::from_secs(secs), future)
            .await
            .expect("the execution did not complete")
    }

    /// Assert that `sql` produces byte-identical results on DataFusion and on
    /// the pipeline engine, both through the `PipelineQueryPlanner` and with
    /// the plan lowered by [`lower`], and that `lowered` hash joins are
    /// lowered to a [`PhysicalHashJoin`]. `setup` registers the tables.
    /// Returns the lowered operator tree.
    async fn assert_join_parity_with_setup<F>(
        sql: &str,
        config: SessionConfig,
        setup: F,
        lowered: usize,
    ) -> Arc<dyn PhysicalOperator>
    where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        assert_byte_eq_with_datafusion_with_config(sql, config.clone(), &setup).await;

        let ctx = datafusion_context(config);
        setup(&ctx).await.expect("setup");
        let expected = with_timeout(run_sql(&ctx, sql)).await.expect("DataFusion");
        // A new plan: executing a plan changes the state of some nodes.
        let plan = ctx
            .sql(sql)
            .await
            .expect("planning")
            .create_physical_plan()
            .await
            .expect("physical planning");
        let root = lower(&plan).expect("lowering");
        let tree = format_operator_tree(&root);
        assert_eq!(
            count::<PhysicalHashJoin>(&root),
            lowered,
            "query: {sql}\nlowered: {tree}"
        );
        let exec: Arc<dyn ExecutionPlan> =
            Arc::new(PipelineRootExec::try_new(Arc::clone(&root), plan).unwrap());
        let batches = with_timeout(collect(Arc::clone(&exec), ctx.task_ctx()))
            .await
            .unwrap_or_else(|e| {
                panic!(
                    "Failed to run the lowered plan: {e}\nquery: {sql}\nlowered: {tree}"
                )
            });
        let actual = EngineOutput {
            schema: exec.schema(),
            plan: exec,
            batches,
            ordered: expected.ordered,
        };
        if let Err(message) = compare_outputs(&expected, &actual) {
            panic!("Results differ: {message}\nquery: {sql}\nlowered: {tree}");
        }
        root
    }

    /// [`assert_join_parity_with_setup`] with the default test tables, for a
    /// query with one native join.
    async fn assert_join_parity(sql: &str) -> Arc<dyn PhysicalOperator> {
        assert_join_parity_with_config(sql, test_session_config(), 1).await
    }

    async fn assert_join_parity_with_config(
        sql: &str,
        config: SessionConfig,
        lowered: usize,
    ) -> Arc<dyn PhysicalOperator> {
        let setup = async |ctx: &SessionContext| register_test_tables(ctx);
        assert_join_parity_with_setup(sql, config, setup, lowered).await
    }

    /// A configuration with which DataFusion plans `Partitioned` hash joins.
    fn partitioned_config() -> SessionConfig {
        let mut config = test_session_config();
        let optimizer = &mut config.options_mut().optimizer;
        optimizer.hash_join_single_partition_threshold = 0;
        optimizer.hash_join_single_partition_threshold_rows = 0;
        config
    }

    fn task_ctx() -> Arc<TaskContext> {
        Arc::new(TaskContext::default().with_session_config(test_session_config()))
    }

    // ---------------------------------------------------------------------
    // Test data
    // ---------------------------------------------------------------------

    /// The schema of the `side` (`"l"` or `"r"`) input of the plans of the
    /// tests: keys of many types (with duplicates, NULLs, `-0.0` and `NaN`)
    /// and a unique `id`.
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
        ]))
    }

    /// The rows `ids` of the `side` input. `skew` changes the distribution
    /// of the keys, so that the two sides do not match one to one.
    fn data_batch(side: &str, ids: Range<i64>, skew: i64) -> RecordBatch {
        let words = ["alpha", "beta", "a string longer than twelve bytes", ""];
        let key = |i: i64| ((i * 7 + skew) % 23) as i32 - 5;
        let k = Int32Array::from_iter(ids.clone().map(|i| (i % 11 != 3).then(|| key(i))));
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
            ],
        )
        .unwrap()
    }

    /// A scan of the `side` input with `rows` rows in skewed partitions (one
    /// of them empty), or of no rows.
    fn data_scan(side: &str, rows: i64, skew: i64) -> Arc<dyn ExecutionPlan> {
        let partitions = if rows == 0 {
            vec![vec![], vec![]]
        } else {
            let a = rows / 2;
            let b = rows * 3 / 4;
            vec![
                vec![data_batch(side, 0..a, skew)],
                vec![],
                vec![
                    data_batch(side, a..b, skew),
                    data_batch(side, b..rows, skew),
                ],
            ]
        };
        MemorySourceConfig::try_new_exec(&partitions, data_schema(side), None).unwrap()
    }

    /// A residual filter `l<left> <op> r<right>`.
    fn join_filter(left: &str, op: Operator, right: &str) -> JoinFilter {
        let left_schema = data_schema("l");
        let right_schema = data_schema("r");
        let (l, r) = (
            left_schema.index_of(&format!("l{left}")).unwrap(),
            right_schema.index_of(&format!("r{right}")).unwrap(),
        );
        let schema = Arc::new(Schema::new(vec![
            left_schema.field(l).clone(),
            right_schema.field(r).clone(),
        ]));
        let expression = Arc::new(BinaryExpr::new(
            col(left_schema.field(l).name(), &schema).unwrap(),
            op,
            col(right_schema.field(r).name(), &schema).unwrap(),
        ));
        JoinFilter::new(
            expression,
            vec![
                ColumnIndex {
                    index: l,
                    side: JoinSide::Left,
                },
                ColumnIndex {
                    index: r,
                    side: JoinSide::Right,
                },
            ],
            schema,
        )
    }

    /// The arguments of a hash join of the test inputs.
    #[derive(Debug, Clone)]
    struct JoinCase {
        join_type: JoinType,
        /// The key columns, e.g. `["k", "s"]` for `lk = rk AND ls = rs`.
        keys: Vec<&'static str>,
        filter: Option<(&'static str, Operator, &'static str)>,
        projection: Option<Vec<usize>>,
        mode: PartitionMode,
        null_equality: NullEquality,
        null_aware: bool,
        left_rows: i64,
        right_rows: i64,
    }

    impl JoinCase {
        fn new(join_type: JoinType, keys: Vec<&'static str>) -> Self {
            Self {
                join_type,
                keys,
                filter: None,
                projection: None,
                mode: PartitionMode::CollectLeft,
                null_equality: NullEquality::NullEqualsNothing,
                null_aware: false,
                left_rows: 300,
                right_rows: 500,
            }
        }

        fn plan(&self) -> Result<Arc<dyn ExecutionPlan>> {
            let left = data_scan("l", self.left_rows, 0);
            let right = data_scan("r", self.right_rows, 3);
            let left_keys = self
                .keys
                .iter()
                .map(|key| col(&format!("l{key}"), &left.schema()))
                .collect::<Result<Vec<_>>>()?;
            let right_keys = self
                .keys
                .iter()
                .map(|key| col(&format!("r{key}"), &right.schema()))
                .collect::<Result<Vec<_>>>()?;
            let (left, right): (Arc<dyn ExecutionPlan>, Arc<dyn ExecutionPlan>) =
                match self.mode {
                    PartitionMode::Partitioned => (
                        Arc::new(RepartitionExec::try_new(
                            left,
                            Partitioning::Hash(left_keys.clone(), 4),
                        )?),
                        Arc::new(RepartitionExec::try_new(
                            right,
                            Partitioning::Hash(right_keys.clone(), 4),
                        )?),
                    ),
                    _ => (Arc::new(CoalescePartitionsExec::new(left)), right),
                };
            let on = left_keys.into_iter().zip(right_keys).collect();
            let filter = self
                .filter
                .map(|(left, op, right)| join_filter(left, op, right));
            Ok(Arc::new(HashJoinExec::try_new(
                left,
                right,
                on,
                filter,
                &self.join_type,
                self.projection.clone(),
                self.mode,
                self.null_equality,
                self.null_aware,
            )?))
        }
    }

    /// Assert that the join of `case` produces the same rows as DataFusion's,
    /// with a native join.
    async fn assert_case(case: &JoinCase) {
        let make = || case.plan();
        let root = assert_lowered_plan_byte_eq(make, lower, task_ctx(), false).await;
        assert_eq!(count::<PhysicalHashJoin>(&root), 1, "{case:?}");
    }

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

    // ---------------------------------------------------------------------
    // Join semantics
    // ---------------------------------------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn all_join_types_match_datafusion() {
        for join_type in ALL_JOIN_TYPES {
            for null_equality in [
                NullEquality::NullEqualsNothing,
                NullEquality::NullEqualsNull,
            ] {
                for mode in [PartitionMode::CollectLeft, PartitionMode::Partitioned] {
                    for filter in [None, Some(("id", Operator::Lt, "id"))] {
                        let case = JoinCase {
                            filter,
                            mode,
                            null_equality,
                            ..JoinCase::new(join_type, vec!["k"])
                        };
                        assert_case(&case).await;
                    }
                }
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn keys_of_many_types_and_several_columns() {
        let keys: [Vec<&'static str>; 8] = [
            vec!["f"],
            vec!["s"],
            vec!["v"],
            vec!["d"],
            vec!["t"],
            vec!["k", "s"],
            vec!["f", "d", "t"],
            vec!["v", "k", "f"],
        ];
        for keys in keys {
            for join_type in [
                JoinType::Inner,
                JoinType::Full,
                JoinType::LeftSemi,
                JoinType::RightAnti,
                JoinType::LeftMark,
            ] {
                for null_equality in [
                    NullEquality::NullEqualsNothing,
                    NullEquality::NullEqualsNull,
                ] {
                    let case = JoinCase {
                        null_equality,
                        ..JoinCase::new(join_type, keys.clone())
                    };
                    assert_case(&case).await;
                }
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn residual_filters_and_projections() {
        let filters = [
            ("id", Operator::Lt, "id"),
            // A filter on the keys themselves.
            ("f", Operator::LtEq, "f"),
            ("v", Operator::NotEq, "v"),
            ("d", Operator::Gt, "d"),
        ];
        for join_type in ALL_JOIN_TYPES {
            for filter in filters {
                let case = JoinCase {
                    filter: Some(filter),
                    ..JoinCase::new(join_type, vec!["k"])
                };
                assert_case(&case).await;
            }
            // Projections, including no columns at all and repeated ones.
            let num_columns = HashJoinExec::try_new(
                data_scan("l", 1, 0),
                data_scan("r", 1, 0),
                vec![(
                    col("lk", &data_schema("l")).unwrap(),
                    col("rk", &data_schema("r")).unwrap(),
                )],
                None,
                &join_type,
                None,
                PartitionMode::CollectLeft,
                NullEquality::NullEqualsNothing,
                false,
            )
            .unwrap()
            .schema()
            .fields()
            .len();
            for projection in [vec![], vec![num_columns - 1, 0], vec![1, 1, 0]] {
                let case = JoinCase {
                    projection: Some(projection),
                    filter: Some(("s", Operator::NotEq, "s")),
                    ..JoinCase::new(join_type, vec!["k"])
                };
                assert_case(&case).await;
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn empty_build_and_probe_sides() {
        for join_type in ALL_JOIN_TYPES {
            for (left_rows, right_rows) in [(0, 500), (300, 0), (0, 0)] {
                for filter in [None, Some(("id", Operator::Gt, "id"))] {
                    let case = JoinCase {
                        filter,
                        left_rows,
                        right_rows,
                        ..JoinCase::new(join_type, vec!["k", "s"])
                    };
                    assert_case(&case).await;
                }
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn null_aware_anti_and_mark_joins() {
        for join_type in [JoinType::LeftAnti, JoinType::LeftMark, JoinType::RightAnti] {
            // Keys with and without NULLs on either side (`id` has none), and
            // empty sides.
            for key in ["k", "id"] {
                for (left_rows, right_rows) in [(300, 500), (0, 500), (300, 0), (40, 7)] {
                    let case = JoinCase {
                        null_aware: true,
                        left_rows,
                        right_rows,
                        ..JoinCase::new(join_type, vec![key])
                    };
                    assert_case(&case).await;
                }
            }
        }
    }

    /// A build side with one key of `duplicates` rows, probed by `probe_rows`
    /// rows of that key, so that every probe row has more matches than fit in
    /// a chunk.
    fn duplicates_plan(
        join_type: JoinType,
        duplicates: i64,
        probe_rows: i64,
        filter: bool,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let schema = |side: &str| {
            Arc::new(Schema::new(vec![
                Field::new(format!("{side}k"), DataType::Int64, true),
                Field::new(format!("{side}v"), DataType::Int64, false),
            ]))
        };
        let batch = |side: &str, keys: Vec<Option<i64>>| {
            let values = Int64Array::from_iter_values(0..keys.len() as i64);
            RecordBatch::try_new(
                schema(side),
                vec![Arc::new(Int64Array::from(keys)), Arc::new(values)],
            )
            .unwrap()
        };
        // Key 7 many times, a few other keys and NULLs.
        let build_keys = (0..duplicates)
            .map(|i| Some(if i % 100 == 99 { i } else { 7 }))
            .chain([None, Some(8)])
            .collect::<Vec<_>>();
        let build = MemorySourceConfig::try_new_exec(
            &[vec![batch("l", build_keys)]],
            schema("l"),
            None,
        )?;
        let probe_keys = (0..probe_rows)
            .map(|i| match i % 5 {
                0 => None,
                1 => Some(8),
                2 => Some(1),
                _ => Some(7),
            })
            .collect::<Vec<_>>();
        // The probe side is sorted by `rv`: DataFusion then keeps the order of
        // the probe side for all joins that do not emit build rows at the end
        // (it emits the unmatched probe rows of a `Right` join after the
        // matches of a batch otherwise).
        let ordering = LexOrdering::new(vec![PhysicalSortExpr::new_default(col(
            "rv",
            &schema("r"),
        )?)])
        .unwrap();
        let probe = DataSourceExec::from_data_source(
            MemorySourceConfig::try_new(
                &[vec![batch("r", probe_keys)]],
                schema("r"),
                None,
            )?
            .try_with_sort_information(vec![ordering])?,
        );
        let filter = filter.then(|| {
            let schema = Arc::new(Schema::new(vec![
                Field::new("lv", DataType::Int64, false),
                Field::new("rv", DataType::Int64, false),
            ]));
            // `lv % 3 <> rv % 2`
            let modulo = |name: &str, n: i64| -> Arc<dyn PhysicalExpr> {
                Arc::new(BinaryExpr::new(
                    col(name, &schema).unwrap(),
                    Operator::Modulo,
                    lit(n),
                ))
            };
            JoinFilter::new(
                Arc::new(BinaryExpr::new(
                    modulo("lv", 3),
                    Operator::NotEq,
                    modulo("rv", 2),
                )),
                vec![
                    ColumnIndex {
                        index: 1,
                        side: JoinSide::Left,
                    },
                    ColumnIndex {
                        index: 1,
                        side: JoinSide::Right,
                    },
                ],
                Arc::clone(&schema),
            )
        });
        let on = vec![(col("lk", &build.schema())?, col("rk", &probe.schema())?)];
        Ok(Arc::new(HashJoinExec::try_new(
            build,
            probe,
            on,
            filter,
            &join_type,
            None,
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )?))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn duplicate_build_keys_span_several_chunks() -> Result<()> {
        for join_type in ALL_JOIN_TYPES {
            for filter in [false, true] {
                // Every probe row of key 7 has 4950 matches.
                let make = || duplicates_plan(join_type, 5000, 13, filter);
                let root =
                    assert_lowered_plan_byte_eq(make, lower, task_ctx(), false).await;
                // The output chunks have at most 2048 rows.
                for threads in [1, 3] {
                    let batches = with_timeout(Executor::collect(
                        Arc::clone(&root),
                        task_ctx(),
                        threads,
                    ))
                    .await?;
                    assert!(
                        batches.iter().all(|b| b.num_rows() <= DEFAULT_VECTOR_SIZE),
                        "{join_type}"
                    );
                    if join_type == JoinType::Inner && !filter {
                        assert!(batches.len() >= 10, "{}", batches.len());
                        let rows =
                            batches.iter().map(RecordBatch::num_rows).sum::<usize>();
                        // 4 probe rows of key 7, 3 of key 8.
                        assert_eq!(rows, 4 * 4950 + 3, "{threads} threads");
                    }
                }
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn single_thread_keeps_the_probe_order() -> Result<()> {
        // With a single thread, the output follows the probe order, and the
        // matches of a probe row the order of the build side.
        for join_type in [
            JoinType::Inner,
            JoinType::Right,
            JoinType::RightSemi,
            JoinType::RightAnti,
            JoinType::RightMark,
        ] {
            let plan = duplicates_plan(join_type, 300, 2000, false)?;
            let expected = collect(Arc::clone(&plan), task_ctx()).await?;
            let plan = duplicates_plan(join_type, 300, 2000, false)?;
            let root = lower(&plan)?;
            let actual = with_timeout(Executor::collect(root, task_ctx(), 1)).await?;
            let column = |batches: &[RecordBatch], name: &str| {
                batches
                    .iter()
                    .flat_map(|batch| {
                        let column = batch.column(batch.schema().index_of(name).unwrap());
                        column
                            .as_primitive::<Int64Type>()
                            .iter()
                            .collect::<Vec<_>>()
                    })
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                column(&actual, "rv"),
                column(&expected, "rv"),
                "{join_type}"
            );
            if join_type == JoinType::Inner {
                assert_eq!(column(&actual, "lv"), column(&expected, "lv"));
            }
        }
        Ok(())
    }

    // ---------------------------------------------------------------------
    // SQL
    // ---------------------------------------------------------------------

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_join_types() {
        for sql in [
            "SELECT t1.id, t2.name FROM t1 JOIN t2 ON t1.a = t2.k",
            "SELECT * FROM t1 JOIN t2 ON t1.a = t2.k",
            "SELECT t1.id, t2.w FROM t1 LEFT JOIN t2 ON t1.a = t2.k AND t2.w > 100",
            "SELECT t1.id, t2.* FROM t2 LEFT JOIN t1 ON t1.a = t2.k",
            "SELECT t2.k, t1.d FROM t1 RIGHT JOIN t2 ON t1.a = t2.k",
            "SELECT t1.id, t2.k FROM t1 FULL OUTER JOIN t2 ON t1.a = t2.k",
            "SELECT t1.*, t2.* FROM t1 FULL JOIN t2 ON t1.a = t2.k AND t1.b < t2.w",
            "SELECT t2.k, t2.name FROM t2 WHERE k IN (SELECT a FROM t1 WHERE e)",
            "SELECT t1.id FROM t1 WHERE a IN (SELECT k FROM t2)",
            "SELECT t2.k FROM t2 WHERE NOT EXISTS (SELECT 1 FROM t1 WHERE t1.a = t2.k)",
            "SELECT t1.id, t1.h FROM t1 WHERE NOT EXISTS (SELECT 1 FROM t2 WHERE t1.a = t2.k)",
            "SELECT t2.k, EXISTS (SELECT 1 FROM t1 WHERE t1.a = t2.k) OR t2.w > 5 AS m FROM t2",
            "SELECT t1.id, EXISTS (SELECT 1 FROM t2 WHERE t1.a = t2.k) AS m FROM t1",
            "SELECT a.id, b.id FROM t1 a JOIN t1 b ON a.id = b.id",
            "SELECT a.k, b.name FROM t2 a JOIN t2 b ON a.k = b.k",
        ] {
            assert_join_parity(sql).await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_keys_filters_and_null_equality() {
        for sql in [
            // Multi-column keys of many types.
            "SELECT x.id, y.id FROM t1 x JOIN t1 y ON x.c = y.c AND x.d = y.d WHERE x.id < 600",
            "SELECT x.id, y.id FROM t1 x JOIN t1 y ON x.h = y.h AND x.g = y.g",
            "SELECT x.id, y.id FROM t1 x JOIN t1 y ON x.ts = y.ts AND x.f = y.f",
            "SELECT x.id, y.c FROM t1 x JOIN t1 y ON x.c = y.c AND x.e = y.e WHERE y.id % 7 = 0",
            "SELECT x.id, y.id FROM t1 x LEFT JOIN t1 y ON x.b = y.b AND x.a = y.a",
            // Keys that are expressions.
            "SELECT t1.id, t2.k FROM t1 JOIN t2 ON t1.a + 1 = t2.k * 2",
            "SELECT t1.id, t2.name FROM t1 JOIN t2 ON upper(t1.d) = upper(t2.name)",
            // Residual filters.
            "SELECT t1.id, t2.k FROM t1 JOIN t2 ON t1.a = t2.k AND t1.b > t2.w",
            "SELECT t1.id, t2.k FROM t1 LEFT JOIN t2 ON t1.a = t2.k AND t1.c < CAST(t2.w AS DOUBLE)",
            "SELECT t2.k FROM t2 WHERE EXISTS (SELECT 1 FROM t1 WHERE t1.a = t2.k AND t1.b > t2.w)",
            "SELECT t2.k FROM t2 WHERE NOT EXISTS (SELECT 1 FROM t1 WHERE t1.a = t2.k AND t1.b > t2.w)",
            // IS NOT DISTINCT FROM.
            "SELECT t1.id, t2.k FROM t1 JOIN t2 ON t1.a IS NOT DISTINCT FROM t2.k",
            "SELECT t1.id, t2.k FROM t1 FULL JOIN t2 ON t1.a IS NOT DISTINCT FROM t2.k",
            "SELECT x.id, y.id FROM t1 x JOIN t1 y ON x.c IS NOT DISTINCT FROM y.c AND x.d IS NOT DISTINCT FROM y.d WHERE x.id < 300",
        ] {
            assert_join_parity(sql).await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_not_in() {
        for sql in [
            "SELECT t2.k FROM t2 WHERE k NOT IN (SELECT a FROM t1)",
            "SELECT t1.id FROM t1 WHERE a NOT IN (SELECT k FROM t2)",
            "SELECT t1.id FROM t1 WHERE a NOT IN (SELECT k FROM t2 WHERE k IS NOT NULL)",
            "SELECT t1.id FROM t1 WHERE id NOT IN (SELECT w FROM t2 WHERE w IS NOT NULL)",
            "SELECT t1.id FROM t1 WHERE a NOT IN (SELECT k FROM t2) OR b > 5000",
            "SELECT t2.k FROM t2 WHERE k NOT IN (SELECT a FROM t1 WHERE a IS NULL)",
            "SELECT t2.k FROM t2 WHERE k NOT IN (SELECT a FROM t1 WHERE a > 1000)",
        ] {
            assert_join_parity(sql).await;
        }
        // DataFusion removes a `NOT IN` of no rows; correlated and
        // multi-column `NOT IN`s are left to DataFusion.
        for sql in [
            "SELECT t2.k FROM t2 WHERE k NOT IN (SELECT a FROM t1 WHERE false)",
            "SELECT t1.id FROM t1 WHERE a NOT IN (SELECT k FROM t2 WHERE t2.w > t1.b)",
            "SELECT t1.id FROM t1 WHERE (a, b) NOT IN (SELECT k, w FROM t2)",
        ] {
            assert_join_parity_with_config(sql, test_session_config(), 0).await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_empty_inputs() {
        for sql in [
            "SELECT t1.id, t2.k FROM t1 JOIN t2 ON t1.a = t2.k WHERE t2.w > 1000000",
            "SELECT * FROM t_empty JOIN t2 ON t_empty.a = t2.k",
            "SELECT * FROM t2 LEFT JOIN t_empty ON t_empty.a = t2.k",
            "SELECT * FROM t_empty RIGHT JOIN t2 ON t_empty.a = t2.k",
            "SELECT * FROM t_empty FULL JOIN t2 ON t_empty.a = t2.k",
            "SELECT * FROM t1 FULL JOIN t_empty ON t_empty.a = t1.a",
            "SELECT t2.* FROM t2 WHERE k IN (SELECT a FROM t_empty)",
            "SELECT t2.* FROM t2 WHERE NOT EXISTS (SELECT 1 FROM t_empty WHERE t_empty.a = t2.k)",
            "SELECT t2.* FROM t2 WHERE k NOT IN (SELECT a FROM t_empty)",
            "SELECT t_empty.* FROM t_empty WHERE a NOT IN (SELECT k FROM t2)",
        ] {
            assert_join_parity(sql).await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn integer_keys_of_narrow_and_wide_ranges() -> Result<()> {
        // A single integer key: of a narrow range, indexed densely, and of a
        // wide range, in hashed tables whose key filter rejects the probe keys
        // outside of the range and of the buckets of the build keys. With
        // duplicates, NULLs and probe keys outside of the range, for all join
        // types, with and without a filter, with NULLs equal or not.
        let schema = |side: &str| {
            Arc::new(Schema::new(vec![
                Field::new(format!("{side}k"), DataType::Int64, true),
                Field::new(format!("{side}v"), DataType::Int64, false),
            ]))
        };
        let batch = |side: &str, keys: Vec<Option<i64>>| {
            let values = Int64Array::from_iter_values(0..keys.len() as i64);
            RecordBatch::try_new(
                schema(side),
                vec![Arc::new(Int64Array::from(keys)), Arc::new(values)],
            )
            .unwrap()
        };
        for scale in [1, 1_000_000_007] {
            let build_keys = (0..500)
                .map(|i| (i % 13 != 5).then(|| (i % 97 - 20) * scale))
                .collect::<Vec<_>>();
            let probe_keys = (0..2000)
                .map(|i| (i % 17 != 3).then(|| (i % 211 - 60) * scale + i % 2))
                .collect::<Vec<_>>();
            for join_type in ALL_JOIN_TYPES {
                for null_equality in [
                    NullEquality::NullEqualsNothing,
                    NullEquality::NullEqualsNull,
                ] {
                    for with_filter in [false, true] {
                        let make = || -> Result<Arc<dyn ExecutionPlan>> {
                            let build = MemorySourceConfig::try_new_exec(
                                &[
                                    vec![batch("l", build_keys[..200].to_vec())],
                                    vec![batch("l", build_keys[200..].to_vec())],
                                ],
                                schema("l"),
                                None,
                            )?;
                            let probe = MemorySourceConfig::try_new_exec(
                                &[
                                    vec![batch("r", probe_keys[..1500].to_vec())],
                                    vec![batch("r", probe_keys[1500..].to_vec())],
                                ],
                                schema("r"),
                                None,
                            )?;
                            let on = vec![(
                                col("lk", &build.schema())?,
                                col("rk", &probe.schema())?,
                            )];
                            let filter = with_filter.then(|| {
                                let schema = Arc::new(Schema::new(vec![
                                    Field::new("lv", DataType::Int64, false),
                                    Field::new("rv", DataType::Int64, false),
                                ]));
                                JoinFilter::new(
                                    Arc::new(BinaryExpr::new(
                                        col("lv", &schema).unwrap(),
                                        Operator::Lt,
                                        col("rv", &schema).unwrap(),
                                    )),
                                    vec![
                                        ColumnIndex {
                                            index: 1,
                                            side: JoinSide::Left,
                                        },
                                        ColumnIndex {
                                            index: 1,
                                            side: JoinSide::Right,
                                        },
                                    ],
                                    schema,
                                )
                            });
                            Ok(Arc::new(HashJoinExec::try_new(
                                Arc::new(CoalescePartitionsExec::new(build)),
                                probe,
                                on,
                                filter,
                                &join_type,
                                None,
                                PartitionMode::CollectLeft,
                                null_equality,
                                false,
                            )?))
                        };
                        let root =
                            assert_lowered_plan_byte_eq(make, lower, task_ctx(), false)
                                .await;
                        assert_eq!(count::<PhysicalHashJoin>(&root), 1, "{join_type}");
                    }
                }
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_multiple_joins_and_partitionings() {
        for (sql, lowered) in [
            // A join below the probe side of a join (inner, `Right` as
            // DataFusion plans these `LEFT JOIN`s, or `Full`): both are
            // native, the build side of the join below waits for the build
            // side of the join above (see `PhysicalHashJoin::try_lower`).
            (
                "SELECT t1.id, x.name, y.w FROM t1 JOIN t2 x ON t1.a = x.k JOIN t2 y ON t1.a = y.k AND x.w = y.w",
                2,
            ),
            (
                "SELECT t1.id, x.name FROM t1 LEFT JOIN t2 x ON t1.a = x.k LEFT JOIN t2 y ON x.w = y.w",
                2,
            ),
            (
                "SELECT t1.id, x.name, y.w FROM t1 JOIN t2 x ON t1.a = x.k FULL JOIN t2 y ON x.w = y.w",
                2,
            ),
            // Chains of three and four joins, of every side.
            (
                "SELECT t1.id, x.name, y.w, z.k FROM t1 JOIN t2 x ON t1.a = x.k JOIN t2 y ON x.w = y.w JOIN t2 z ON y.k = z.k",
                3,
            ),
            (
                "SELECT t1.id, x.name, y.w, z.k FROM t1 LEFT JOIN t2 x ON t1.a = x.k RIGHT JOIN t2 y ON x.w = y.w FULL JOIN t2 z ON y.k = z.k AND z.w > 10",
                3,
            ),
            (
                "SELECT a.id, b.id, c.id, d.id FROM t1 a JOIN t1 b ON a.id = b.id + 1 JOIN t1 c ON b.id = c.id + 1 JOIN t1 d ON c.id = d.id + 1 WHERE a.id % 3 = 0",
                3,
            ),
            (
                "SELECT t2.k, t2.name FROM t2 WHERE k IN (SELECT a FROM t1 WHERE a IN (SELECT k FROM t2 WHERE w > 3)) AND NOT EXISTS (SELECT 1 FROM t1 WHERE t1.a = t2.k AND t1.b > t2.w)",
                3,
            ),
            (
                "SELECT t1.id, u.k, v.name, w.w FROM t1 JOIN t2 u ON t1.a = u.k JOIN (SELECT k, name FROM t2 WHERE w % 2 = 0) v ON u.k = v.k JOIN t2 w ON v.name = w.name AND t1.b < w.w + 5000 JOIN t2 x ON w.k = x.k",
                4,
            ),
            // A join below the build side of a join: both are native.
            (
                "SELECT t1.id, u.name FROM t1 JOIN (SELECT x.k, x.name FROM t2 x JOIN t2 y ON x.k = y.k AND x.w = y.w) u ON t1.a = u.k",
                2,
            ),
        ] {
            assert_join_parity_with_config(sql, test_session_config(), lowered).await;
        }
        for target_partitions in [1, 3, 16] {
            let config = test_session_config().with_target_partitions(target_partitions);
            for sql in [
                "SELECT t1.id, t2.name FROM t1 JOIN t2 ON t1.a = t2.k",
                "SELECT t1.id, t2.k FROM t1 FULL OUTER JOIN t2 ON t1.a = t2.k",
                "SELECT t2.k FROM t2 WHERE k NOT IN (SELECT a FROM t1)",
            ] {
                assert_join_parity_with_config(sql, config.clone(), 1).await;
            }
        }
        // `Partitioned` joins, whose inputs DataFusion repartitions by hash.
        for sql in [
            "SELECT t1.id, t2.name FROM t1 JOIN t2 ON t1.a = t2.k",
            "SELECT t1.id, t2.w FROM t1 LEFT JOIN t2 ON t1.a = t2.k AND t2.w > 100",
            "SELECT t1.id, t2.k FROM t1 FULL OUTER JOIN t2 ON t1.a = t2.k",
            "SELECT t2.k FROM t2 WHERE NOT EXISTS (SELECT 1 FROM t1 WHERE t1.a = t2.k)",
            "SELECT t1.id FROM t1 WHERE a IN (SELECT k FROM t2)",
            "SELECT x.id, y.id FROM t1 x JOIN t1 y ON x.c = y.c AND x.d = y.d WHERE x.id < 600",
        ] {
            let ctx = datafusion_context(partitioned_config());
            register_test_tables(&ctx).unwrap();
            let plan = ctx
                .sql(sql)
                .await
                .unwrap()
                .create_physical_plan()
                .await
                .unwrap();
            let display = displayable(plan.as_ref()).indent(true).to_string();
            assert!(display.contains("mode=Partitioned"), "{display}");
            let root = assert_join_parity_with_config(sql, partitioned_config(), 1).await;
            assert!(
                bridged_plans(&root)
                    .iter()
                    .all(|plan| !plan.is::<RepartitionExec>()),
                "{}",
                format_operator_tree(&root)
            );
        }
    }

    /// The tables of TPC-H that the joins of [`TPCH_JOIN_CHAINS`] read, small
    /// and with keys spread so that every chain has matches.
    const TPCH_TABLES: [(&str, &str); 8] = [
        (
            "region",
            "SELECT column1 AS r_regionkey, column2 AS r_name FROM (VALUES \
             (0, 'AFRICA'), (1, 'AMERICA'), (2, 'ASIA'), (3, 'EUROPE'), (4, 'MIDDLE EAST'))",
        ),
        (
            "nation",
            "SELECT column1 AS n_nationkey, column2 AS n_name, column3 AS n_regionkey FROM (VALUES \
             (0, 'ALGERIA', 0), (1, 'ARGENTINA', 1), (2, 'BRAZIL', 1), (3, 'CANADA', 1), \
             (4, 'EGYPT', 4), (5, 'ETHIOPIA', 0), (6, 'FRANCE', 3), (7, 'GERMANY', 3), \
             (8, 'INDIA', 2), (9, 'INDONESIA', 2), (10, 'IRAN', 4), (11, 'IRAQ', 4), \
             (12, 'JAPAN', 2), (13, 'JORDAN', 4), (14, 'KENYA', 0), (15, 'MOROCCO', 0), \
             (16, 'MOZAMBIQUE', 0), (17, 'PERU', 1), (18, 'CHINA', 2), (19, 'ROMANIA', 3), \
             (20, 'SAUDI ARABIA', 4), (21, 'VIETNAM', 2), (22, 'RUSSIA', 3), \
             (23, 'UNITED KINGDOM', 3), (24, 'UNITED STATES', 1))",
        ),
        (
            "supplier",
            "SELECT value AS s_suppkey, concat('Supplier#', value) AS s_name, \
             value % 25 AS s_nationkey, CAST(value * 37 % 1000 AS DECIMAL(15, 2)) AS s_acctbal \
             FROM generate_series(1, 50)",
        ),
        (
            "customer",
            "SELECT value AS c_custkey, concat('Customer#', value) AS c_name, \
             value % 25 AS c_nationkey, CAST(value * 13 % 997 AS DECIMAL(15, 2)) AS c_acctbal, \
             CASE value % 5 WHEN 0 THEN 'BUILDING' WHEN 1 THEN 'AUTOMOBILE' \
             WHEN 2 THEN 'MACHINERY' WHEN 3 THEN 'HOUSEHOLD' ELSE 'FURNITURE' END AS c_mktsegment \
             FROM generate_series(1, 150)",
        ),
        (
            "part",
            "SELECT value AS p_partkey, \
             concat(CASE value % 3 WHEN 0 THEN 'forest green ' WHEN 1 THEN 'navy ' ELSE 'red ' END, value) AS p_name, \
             CASE value % 4 WHEN 0 THEN 'ECONOMY ANODIZED STEEL' WHEN 1 THEN 'STANDARD POLISHED BRASS' \
             WHEN 2 THEN 'PROMO BRUSHED BRASS' ELSE 'LARGE PLATED TIN' END AS p_type, \
             value % 20 + 1 AS p_size FROM generate_series(1, 100)",
        ),
        (
            "partsupp",
            "SELECT value / 4 + 1 AS ps_partkey, (value / 4 + value % 4 * 13) % 50 + 1 AS ps_suppkey, \
             CAST(value * 17 % 500 + 1 AS DECIMAL(15, 2)) AS ps_supplycost FROM generate_series(0, 399)",
        ),
        (
            "orders",
            "SELECT value AS o_orderkey, value * 7 % 150 + 1 AS o_custkey, \
             CASE value % 3 WHEN 0 THEN 'F' WHEN 1 THEN 'O' ELSE 'P' END AS o_orderstatus, \
             arrow_cast(arrow_cast(8035 + value * 11 % 2400, 'Int32'), 'Date32') AS o_orderdate \
             FROM generate_series(1, 600)",
        ),
        (
            "lineitem",
            "SELECT value / 4 + 1 AS l_orderkey, value * 13 % 100 + 1 AS l_partkey, \
             (value / 4 + value % 4 * 13) * 13 % 50 + 1 AS l_suppkey, value % 4 + 1 AS l_linenumber, \
             CAST(value % 50 + 1 AS DECIMAL(15, 2)) AS l_quantity, \
             CAST(value * 31 % 10000 AS DECIMAL(15, 2)) AS l_extendedprice, \
             CAST(value % 11 AS DECIMAL(15, 2)) AS l_discount, \
             CASE value % 4 WHEN 0 THEN 'R' WHEN 1 THEN 'A' ELSE 'N' END AS l_returnflag, \
             arrow_cast(arrow_cast(8035 + value * 3 % 2500, 'Int32'), 'Date32') AS l_shipdate, \
             arrow_cast(arrow_cast(8045 + value * 3 % 2500, 'Int32'), 'Date32') AS l_commitdate, \
             arrow_cast(arrow_cast(8035 + value * 3 % 2500 + CASE WHEN value % 5 = 0 THEN 20 ELSE 5 END, \
             'Int32'), 'Date32') AS l_receiptdate \
             FROM generate_series(0, 2399)",
        ),
    ];

    /// The joins of TPC-H queries 2, 3, 5, 7, 8, 9, 10 and 21, chains of 3 to
    /// 8 tables whose probe sides are joins (with the predicates of the
    /// queries, adapted to the data of [`TPCH_TABLES`], and the joined
    /// columns instead of the aggregations): DataFusion plans them as chains
    /// of `HashJoinExec`s, the probe side of every join is the next join.
    const TPCH_JOIN_CHAINS: [(&str, &str); 8] = [
        (
            "Q2",
            "SELECT s_acctbal, s_name, n_name, p_partkey, ps_supplycost \
             FROM part, supplier, partsupp, nation, region \
             WHERE p_partkey = ps_partkey AND s_suppkey = ps_suppkey AND p_size > 5 \
             AND p_type LIKE '%BRASS' AND s_nationkey = n_nationkey \
             AND n_regionkey = r_regionkey AND r_name = 'EUROPE'",
        ),
        (
            "Q3",
            "SELECT l_orderkey, l_extendedprice, l_discount, o_orderdate \
             FROM customer, orders, lineitem \
             WHERE c_mktsegment = 'BUILDING' AND c_custkey = o_custkey \
             AND l_orderkey = o_orderkey AND o_orderdate < DATE '1995-03-15' \
             AND l_shipdate > DATE '1995-03-15'",
        ),
        (
            "Q5",
            "SELECT n_name, l_extendedprice, l_discount \
             FROM customer, orders, lineitem, supplier, nation, region \
             WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey \
             AND l_suppkey = s_suppkey AND c_nationkey = s_nationkey \
             AND s_nationkey = n_nationkey AND n_regionkey = r_regionkey \
             AND r_name = 'AMERICA' AND o_orderdate >= DATE '1993-01-01' \
             AND o_orderdate < DATE '1997-01-01'",
        ),
        (
            "Q7",
            "SELECT n1.n_name AS supp_nation, n2.n_name AS cust_nation, l_shipdate, l_extendedprice \
             FROM supplier, lineitem, orders, customer, nation n1, nation n2 \
             WHERE s_suppkey = l_suppkey AND o_orderkey = l_orderkey AND c_custkey = o_custkey \
             AND s_nationkey = n1.n_nationkey AND c_nationkey = n2.n_nationkey \
             AND ((n1.n_regionkey = 3 AND n2.n_name = 'BRAZIL') \
             OR (n1.n_name = 'BRAZIL' AND n2.n_regionkey = 3)) \
             AND l_shipdate BETWEEN DATE '1993-01-01' AND DATE '1997-12-31'",
        ),
        (
            "Q8",
            "SELECT o_orderdate, l_extendedprice, l_discount, n2.n_name AS nation \
             FROM part, supplier, lineitem, orders, customer, nation n1, nation n2, region \
             WHERE p_partkey = l_partkey AND s_suppkey = l_suppkey AND l_orderkey = o_orderkey \
             AND o_custkey = c_custkey AND c_nationkey = n1.n_nationkey \
             AND n1.n_regionkey = r_regionkey AND r_name = 'AMERICA' \
             AND s_nationkey = n2.n_nationkey \
             AND o_orderdate BETWEEN DATE '1993-01-01' AND DATE '1997-12-31' \
             AND p_type = 'ECONOMY ANODIZED STEEL'",
        ),
        (
            "Q9",
            "SELECT n_name AS nation, o_orderdate, l_extendedprice, l_discount, ps_supplycost, l_quantity \
             FROM part, supplier, lineitem, partsupp, orders, nation \
             WHERE s_suppkey = l_suppkey AND ps_suppkey = l_suppkey AND ps_partkey = l_partkey \
             AND p_partkey = l_partkey AND o_orderkey = l_orderkey \
             AND s_nationkey = n_nationkey AND p_name LIKE '%green%'",
        ),
        (
            "Q10",
            "SELECT c_custkey, c_name, l_extendedprice, l_discount, c_acctbal, n_name \
             FROM customer, orders, lineitem, nation \
             WHERE c_custkey = o_custkey AND l_orderkey = o_orderkey \
             AND o_orderdate >= DATE '1993-10-01' AND o_orderdate < DATE '1995-01-01' \
             AND l_returnflag = 'R' AND c_nationkey = n_nationkey",
        ),
        (
            "Q21",
            "SELECT s_name, l1.l_orderkey FROM supplier, lineitem l1, orders, nation \
             WHERE s_suppkey = l1.l_suppkey AND o_orderkey = l1.l_orderkey \
             AND o_orderstatus = 'F' AND l1.l_receiptdate > l1.l_commitdate \
             AND EXISTS (SELECT * FROM lineitem l2 WHERE l2.l_orderkey = l1.l_orderkey \
             AND l2.l_suppkey <> l1.l_suppkey) \
             AND NOT EXISTS (SELECT * FROM lineitem l3 WHERE l3.l_orderkey = l1.l_orderkey \
             AND l3.l_suppkey <> l1.l_suppkey AND l3.l_receiptdate > l3.l_commitdate) \
             AND s_nationkey = n_nationkey \
             AND n_name IN ('SAUDI ARABIA', 'GERMANY', 'PERU', 'CHINA', 'FRANCE')",
        ),
    ];

    /// Register the tables of [`TPCH_TABLES`] in `ctx`, as `MemTable`s.
    async fn register_tpch_memtables(ctx: &SessionContext) -> Result<()> {
        for (table, query) in TPCH_TABLES {
            ctx.sql(&format!("CREATE TABLE {table} AS {query}"))
                .await?
                .collect()
                .await?;
        }
        Ok(())
    }

    /// Write the tables of [`TPCH_TABLES`] as Parquet files to `dir`, every
    /// table in three files of row groups of 64 rows, sorted by its first
    /// column, so that the dynamic filters of the joins prune row groups.
    async fn write_tpch_parquet(dir: &std::path::Path) -> Result<()> {
        let ctx = SessionContext::new();
        register_tpch_memtables(&ctx).await?;
        let properties = ::parquet::file::properties::WriterProperties::builder()
            .set_max_row_group_row_count(Some(64))
            .build();
        for (table, _) in TPCH_TABLES {
            let df = ctx
                .sql(&format!("SELECT * FROM {table} ORDER BY 1"))
                .await?;
            let schema = Arc::clone(df.schema().inner());
            let batches = df.collect().await?;
            let batch = arrow::compute::concat_batches(&schema, &batches)?;
            let rows = batch.num_rows();
            for (file, start) in [0, rows / 3, rows * 2 / 3].into_iter().enumerate() {
                let end = [rows / 3, rows * 2 / 3, rows][file];
                let path = dir.join(table).join(format!("part-{file}.parquet"));
                let slice = batch.slice(start, end - start);
                write_parquet(
                    &path,
                    Arc::clone(&schema),
                    &[slice],
                    Some(properties.clone()),
                )?;
            }
        }
        Ok(())
    }

    /// The number of `HashJoinExec`s of `plan`.
    fn count_hash_joins(plan: &Arc<dyn ExecutionPlan>) -> usize {
        usize::from(plan.is::<HashJoinExec>())
            + plan
                .children()
                .into_iter()
                .map(count_hash_joins)
                .sum::<usize>()
    }

    /// Assert that the chain of joins `sql` produces byte-identical results on
    /// DataFusion and on the pipeline engine (see
    /// [`assert_join_parity_with_setup`]), and runs natively from top to
    /// bottom: every `HashJoinExec` of DataFusion's plan is lowered to a
    /// [`PhysicalHashJoin`], and no `BridgeOperator` is left. Returns the
    /// number of rows of the result.
    async fn assert_native_join_chain<F>(
        name: &str,
        sql: &str,
        config: SessionConfig,
        setup: F,
    ) -> usize
    where
        F: AsyncFn(&SessionContext) -> Result<()>,
    {
        let ctx = datafusion_context(config.clone());
        setup(&ctx).await.expect("setup");
        let plan = ctx
            .sql(sql)
            .await
            .expect("planning")
            .create_physical_plan()
            .await
            .expect("physical planning");
        let joins = count_hash_joins(&plan);
        assert!(joins >= 2, "{name}: {joins} joins");
        let root = assert_join_parity_with_setup(sql, config, setup, joins).await;
        let bridged = bridged_plans(&root);
        assert!(
            bridged.is_empty(),
            "{name} falls back to BridgeOperator: {}",
            format_operator_tree(&root)
        );
        let expected = run_sql(&ctx, sql).await.expect("DataFusion");
        count_rows(&expected.batches)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_tpch_join_chains_over_memtables() {
        let setup = async |ctx: &SessionContext| register_tpch_memtables(ctx).await;
        for (name, sql) in TPCH_JOIN_CHAINS {
            let rows =
                assert_native_join_chain(name, sql, test_session_config(), &setup).await;
            assert!(rows > 0, "{name} has no rows");
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_tpch_join_chains_over_parquet() {
        // DataFusion attaches dynamic filters to the joins, which the Parquet
        // scans of their probe sides read: the native joins leave them `true`.
        let dir = tempfile::tempdir().unwrap();
        write_tpch_parquet(dir.path()).await.unwrap();
        let setup = async |ctx: &SessionContext| {
            for (table, _) in TPCH_TABLES {
                let path = dir.path().join(table);
                let path = path.to_str().expect("a UTF-8 path");
                ctx.register_parquet(table, path, ParquetReadOptions::default())
                    .await?;
            }
            Ok(())
        };
        for pushdown_filters in [false, true] {
            let mut config = test_session_config();
            config.options_mut().execution.parquet.pushdown_filters = pushdown_filters;
            for (name, sql) in TPCH_JOIN_CHAINS {
                let ctx = datafusion_context(config.clone());
                setup(&ctx).await.unwrap();
                let plan = ctx
                    .sql(sql)
                    .await
                    .unwrap()
                    .create_physical_plan()
                    .await
                    .unwrap();
                assert!(
                    has_dynamic_filters(&plan),
                    "{name}: {}",
                    displayable(plan.as_ref()).indent(true)
                );
                let rows =
                    assert_native_join_chain(name, sql, config.clone(), &setup).await;
                assert!(rows > 0, "{name} has no rows");
            }
        }
    }

    /// Whether a `HashJoinExec` of `plan` produces a dynamic filter.
    fn has_dynamic_filters(plan: &Arc<dyn ExecutionPlan>) -> bool {
        (plan.is::<HashJoinExec>() && !plan.dynamic_expressions_produced().is_empty())
            || plan.children().into_iter().any(has_dynamic_filters)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_custom_tables() {
        // Keys of the same value with different signs of zero, NaNs, and a
        // probe side whose chunks all match many build rows.
        let setup = async |ctx: &SessionContext| {
            register_test_tables(ctx)?;
            ctx.sql(
                "CREATE TABLE floats AS SELECT * FROM (VALUES \
                 (1, CAST(0.0 AS DOUBLE)), (2, -0.0), (3, 'NaN'), (4, NULL), (5, 1.5), (6, -0.0)) AS v(id, x)",
            )
            .await?
            .collect()
            .await?;
            ctx.sql(
                "CREATE TABLE dups AS SELECT value % 3 AS k, value AS v FROM generate_series(1, 6000)",
            )
            .await?
            .collect()
            .await?;
            Ok(())
        };
        for (sql, lowered) in [
            (
                "SELECT a.id, b.id FROM floats a JOIN floats b ON a.x = b.x",
                1,
            ),
            (
                "SELECT a.id, b.id FROM floats a FULL JOIN floats b ON a.x = b.x",
                1,
            ),
            (
                "SELECT a.id, b.id FROM floats a JOIN floats b ON a.x IS NOT DISTINCT FROM b.x",
                1,
            ),
            (
                "SELECT a.id FROM floats a WHERE x NOT IN (SELECT x FROM floats WHERE id > 4)",
                1,
            ),
            ("SELECT d.v, t2.name FROM dups d JOIN t2 ON d.k = t2.k", 1),
            (
                "SELECT t2.name, d.v FROM t2 LEFT JOIN dups d ON d.k = t2.k AND d.v % 7 = 0",
                1,
            ),
        ] {
            assert_join_parity_with_setup(sql, test_session_config(), &setup, lowered)
                .await;
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn differential_unions_on_either_side() {
        for sql in [
            // A union on the build side: two pipelines sink into the join.
            "SELECT t1.id, u.k FROM t1 JOIN (SELECT k FROM t2 UNION ALL SELECT k FROM t2 WHERE w > 0) u ON t1.a = u.k",
            "SELECT t1.id, u.k FROM t1 RIGHT JOIN (SELECT k FROM t2 UNION ALL SELECT k + 1 FROM t2) u ON t1.a = u.k",
            // A union on the probe side: two probe pipelines, both before the
            // scan of the unmatched build rows.
            "SELECT x.id, t2.name FROM (SELECT id, a FROM t1 UNION ALL SELECT id, a FROM t1 WHERE a > 0) x FULL JOIN t2 ON x.a = t2.k",
            "SELECT t2.k FROM t2 WHERE k NOT IN (SELECT a FROM t1 UNION ALL SELECT a + 100 FROM t1 WHERE e)",
        ] {
            let root = assert_join_parity(sql).await;
            assert_eq!(
                count::<PhysicalUnion>(&root),
                1,
                "{}",
                format_operator_tree(&root)
            );
        }
    }

    /// A plan of `join_type` of two scans of `Dictionary(Int32, Utf8)` keys,
    /// with different dictionaries in every batch.
    fn dictionary_plan(
        join_type: JoinType,
        null_equality: NullEquality,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        use arrow::array::DictionaryArray;
        use arrow::datatypes::Int32Type;
        let schema = |side: &str| {
            Arc::new(Schema::new(vec![
                Field::new(
                    format!("{side}k"),
                    DataType::Dictionary(
                        Box::new(DataType::Int32),
                        Box::new(DataType::Utf8),
                    ),
                    true,
                ),
                Field::new(format!("{side}v"), DataType::Int64, false),
            ]))
        };
        let batch = |side: &str, offset: usize, rows: usize| {
            let words = ["x", "y", "a longer dictionary value", "z", ""];
            let keys = (0..rows)
                .map(|i| ((i + offset) % 6 != 5).then(|| words[(i * 3 + offset) % 5]))
                .collect::<DictionaryArray<Int32Type>>();
            let values =
                Int64Array::from_iter_values((offset..offset + rows).map(|v| v as i64));
            RecordBatch::try_new(schema(side), vec![Arc::new(keys), Arc::new(values)])
                .unwrap()
        };
        let scan = |side: &str, offsets: [usize; 2]| {
            MemorySourceConfig::try_new_exec(
                &[
                    vec![batch(side, offsets[0], 40)],
                    vec![batch(side, offsets[1], 25)],
                ],
                schema(side),
                None,
            )
        };
        let left = scan("l", [0, 7])?;
        let right = scan("r", [3, 11])?;
        let on = vec![(col("lk", &left.schema())?, col("rk", &right.schema())?)];
        Ok(Arc::new(HashJoinExec::try_new(
            Arc::new(CoalescePartitionsExec::new(left)),
            right,
            on,
            None,
            &join_type,
            None,
            PartitionMode::CollectLeft,
            null_equality,
            false,
        )?))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn dictionary_keys_are_compared_by_value() {
        for join_type in ALL_JOIN_TYPES {
            for null_equality in [
                NullEquality::NullEqualsNothing,
                NullEquality::NullEqualsNull,
            ] {
                let make = || dictionary_plan(join_type, null_equality);
                let root =
                    assert_lowered_plan_byte_eq(make, lower, task_ctx(), false).await;
                assert_eq!(count::<PhysicalHashJoin>(&root), 1);
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn only_the_selected_rows_of_chunks_are_joined() -> Result<()> {
        // Chunks with selections: every third build row, the probe rows in
        // reverse order without every fourth one.
        let build = data_batch("l", 0..900, 0);
        let probe = data_batch("r", 0..1500, 1);
        let build_chunk = DataChunk::from_record_batch(build.clone()).with_selection(
            SelectionVector::from((0..900).step_by(3).collect::<Vec<u32>>()),
        )?;
        let probe_chunk = DataChunk::from_record_batch(probe.clone()).with_selection(
            SelectionVector::from(
                (0..1500).rev().filter(|i| i % 4 != 0).collect::<Vec<u32>>(),
            ),
        )?;
        for join_type in ALL_JOIN_TYPES {
            let native = join_of(
                Arc::new(TestSource::with_schema(
                    data_schema("l"),
                    vec![build_chunk.clone()],
                )),
                Arc::new(TestSource::with_schema(
                    data_schema("r"),
                    vec![probe_chunk.clone()],
                )),
                join_type,
            )?;
            let batches = with_timeout(Executor::collect(native, task_ctx(), 4)).await?;
            // DataFusion's join of the selected rows.
            let scan = |chunk: &DataChunk| {
                let batch = chunk.compact().unwrap();
                MemorySourceConfig::try_new_exec(
                    &[vec![batch.clone()]],
                    batch.schema(),
                    None,
                )
                .unwrap()
            };
            let left = scan(&build_chunk);
            let right = scan(&probe_chunk);
            let on = vec![(col("lk", &left.schema())?, col("rk", &right.schema())?)];
            let plan: Arc<dyn ExecutionPlan> = Arc::new(HashJoinExec::try_new(
                left,
                right,
                on,
                None,
                &join_type,
                None,
                PartitionMode::CollectLeft,
                NullEquality::NullEqualsNothing,
                false,
            )?);
            let expected = EngineOutput {
                schema: plan.schema(),
                batches: collect(Arc::clone(&plan), task_ctx()).await?,
                plan: Arc::clone(&plan),
                ordered: false,
            };
            let actual = EngineOutput {
                schema: plan.schema(),
                batches,
                plan,
                ordered: false,
            };
            if let Err(message) = compare_outputs(&expected, &actual) {
                panic!("{join_type}: {message}");
            }
        }
        Ok(())
    }

    /// A streaming operator that passes its input chunks on unchanged, and
    /// records them.
    #[derive(Debug)]
    struct Recorder {
        children: Vec<Arc<dyn PhysicalOperator>>,
        chunks: Arc<Mutex<Vec<DataChunk>>>,
    }

    impl PhysicalOperator for Recorder {
        fn name(&self) -> &str {
            "Recorder"
        }

        fn schema(&self) -> SchemaRef {
            self.children[0].schema()
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
            self.chunks.lock().push(input.clone());
            *output = input.clone();
            Ok(OperatorResult::NeedMoreInput)
        }
    }

    /// DataFusion's join of the (logical) rows of `build` and `probe` (chunks
    /// of the test inputs) on `l<key> = r<key>`, with the filter `filter`.
    async fn datafusion_join_of_chunks(
        build: &DataChunk,
        probe: &DataChunk,
        join_type: JoinType,
        key: &str,
        filter: Option<JoinFilter>,
    ) -> Result<EngineOutput> {
        let scan = |chunk: &DataChunk| -> Result<Arc<dyn ExecutionPlan>> {
            let batch = chunk.compact()?;
            Ok(MemorySourceConfig::try_new_exec(
                &[vec![batch.clone()]],
                batch.schema(),
                None,
            )?)
        };
        let (left, right) = (scan(build)?, scan(probe)?);
        let on = vec![(
            col(&format!("l{key}"), &left.schema())?,
            col(&format!("r{key}"), &right.schema())?,
        )];
        let plan: Arc<dyn ExecutionPlan> = Arc::new(HashJoinExec::try_new(
            left,
            right,
            on,
            filter,
            &join_type,
            None,
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )?);
        Ok(EngineOutput {
            schema: plan.schema(),
            batches: collect(Arc::clone(&plan), task_ctx()).await?,
            plan,
            ordered: false,
        })
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn probe_chunks_with_a_selection_are_not_compacted() -> Result<()> {
        // Probe chunks with and without a selection (of two thirds of their
        // rows), with keys of integers (a dense index) and strings (hashed
        // tables), with and without a filter. The joins whose output only has
        // probe columns (and a mark) select the probe rows: their output has
        // the probe columns themselves, with a selection; the others take the
        // probe rows of their output from the columns.
        let build = DataChunk::from_record_batch(data_batch("l", 0..60, 0));
        let probe = data_batch("r", 0..900, 1);
        let rows = (0..900).filter(|i| i % 3 != 1).collect::<Vec<u32>>();
        let probes = [
            DataChunk::from_record_batch(probe.clone()),
            DataChunk::from_record_batch(probe.clone())
                .with_selection(SelectionVector::from(rows))?,
        ];
        for probe_chunk in &probes {
            for key in ["k", "s"] {
                for filter in [None, Some(("id", Operator::Lt, "id"))] {
                    for join_type in ALL_JOIN_TYPES {
                        let case = format!(
                            "{join_type} on {key}, filter {filter:?}, selection {}",
                            probe_chunk.selection().is_some()
                        );
                        let on = vec![(
                            col(&format!("l{key}"), &data_schema("l"))?,
                            col(&format!("r{key}"), &data_schema("r"))?,
                        )];
                        let join_filter =
                            filter.map(|(left, op, right)| join_filter(left, op, right));
                        let join = Arc::new(PhysicalHashJoin::try_new(
                            Arc::new(TestSource::with_schema(
                                data_schema("l"),
                                vec![build.clone()],
                            )),
                            Arc::new(TestSource::with_schema(
                                data_schema("r"),
                                vec![probe_chunk.clone()],
                            )),
                            on,
                            join_filter.clone(),
                            join_type,
                            None,
                            NullEquality::NullEqualsNothing,
                            false,
                        )?);
                        let chunks = Arc::new(Mutex::new(vec![]));
                        let recorder = Arc::new(Recorder {
                            children: vec![join],
                            chunks: Arc::clone(&chunks),
                        });
                        let batches =
                            with_timeout(Executor::collect(recorder, task_ctx(), 3))
                                .await?;
                        let expected = datafusion_join_of_chunks(
                            &build,
                            probe_chunk,
                            join_type,
                            key,
                            join_filter,
                        )
                        .await?;
                        let actual = EngineOutput {
                            schema: expected.schema.clone(),
                            batches,
                            plan: Arc::clone(&expected.plan),
                            ordered: false,
                        };
                        if let Err(message) = compare_outputs(&expected, &actual) {
                            panic!("{case}: {message}");
                        }
                        let selects = matches!(
                            join_type,
                            JoinType::RightSemi
                                | JoinType::RightAnti
                                | JoinType::RightMark
                        );
                        let chunks = chunks.lock();
                        assert!(chunks.iter().any(|chunk| !chunk.is_empty()), "{case}");
                        for chunk in chunks.iter().filter(|chunk| !chunk.is_empty()) {
                            // The first columns of the output of the other
                            // joins are build columns.
                            let shared = (0..probe.num_columns())
                                .all(|i| Arc::ptr_eq(chunk.column(i), probe.column(i)));
                            assert_eq!(shared, selects, "{case}");
                            // A selection, unless all rows are selected.
                            if selects {
                                assert!(
                                    chunk.selection().is_some()
                                        || chunk.len() == probe.num_rows(),
                                    "{case}"
                                );
                            }
                        }
                    }
                }
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn semi_anti_and_mark_joins_of_large_probe_chunks() -> Result<()> {
        // A probe chunk of more than `DEFAULT_VECTOR_SIZE` rows: the joins that
        // select their probe rows still output chunks of at most that many.
        let build = DataChunk::from_record_batch(data_batch("l", 0..50, 0));
        let probe = DataChunk::from_record_batch(data_batch("r", 0..5000, 1));
        for join_type in [
            JoinType::RightSemi,
            JoinType::RightAnti,
            JoinType::RightMark,
        ] {
            let join = join_of(
                Arc::new(TestSource::with_schema(
                    data_schema("l"),
                    vec![build.clone()],
                )),
                Arc::new(TestSource::with_schema(
                    data_schema("r"),
                    vec![probe.clone()],
                )),
                join_type,
            )?;
            let batches = with_timeout(Executor::collect(join, task_ctx(), 2)).await?;
            assert!(
                batches.iter().all(|b| b.num_rows() <= DEFAULT_VECTOR_SIZE),
                "{join_type}"
            );
            let expected =
                datafusion_join_of_chunks(&build, &probe, join_type, "k", None).await?;
            let actual = EngineOutput {
                schema: expected.schema.clone(),
                batches,
                plan: Arc::clone(&expected.plan),
                ordered: false,
            };
            if let Err(message) = compare_outputs(&expected, &actual) {
                panic!("{join_type}: {message}");
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn payload_columns_of_all_types() {
        // A self join on `i32` of a batch with columns of (almost) all Arrow
        // types: the build rows are restored exactly from their rows. (Only
        // the types of the Arrow row format, with which the harness compares
        // the rows bit for bit: it sorts the rows of the other types by their
        // display, which does not tell apart NaNs with different payloads.)
        let all_types = crate::tuple_data::test_data::all_types_batch(300);
        // Joins of dictionaries with keys of 8 or 16 bits are left to
        // DataFusion (see
        // `build_dictionaries_with_narrow_keys_are_left_to_datafusion`): the
        // same values, in dictionaries with keys of 32 bits.
        let (fields, columns): (Vec<_>, Vec<_>) = all_types
            .schema()
            .fields()
            .iter()
            .zip(all_types.columns())
            .map(|(field, column)| match field.data_type() {
                DataType::Dictionary(_, values)
                    if concatenation_can_overflow(field.data_type()) =>
                {
                    let wide =
                        DataType::Dictionary(Box::new(DataType::Int32), values.clone());
                    let column = arrow::compute::cast(column, &wide).unwrap();
                    (field.as_ref().clone().with_data_type(wide), column)
                }
                _ => (field.as_ref().clone(), Arc::clone(column)),
            })
            .unzip();
        let all_types =
            RecordBatch::try_new(Arc::new(Schema::new(fields)), columns).unwrap();
        let schema = all_types.schema();
        assert!(
            !schema
                .fields()
                .iter()
                .any(|field| concatenation_can_overflow(field.data_type()))
        );
        let (supported, others): (Vec<usize>, Vec<usize>) = (0..all_types.num_columns())
            .partition(|&i| {
                let field =
                    arrow_row::SortField::new(schema.field(i).data_type().clone());
                arrow_row::RowConverter::supports_fields(&[field])
            });
        assert!(supported.len() > 40, "{}", supported.len());
        // The other types, with the key, without floats (whose NaNs the
        // display does not tell apart).
        let key = schema.index_of("i32").unwrap();
        let others = std::iter::once(key)
            .chain(others.into_iter().filter(|&i| {
                !matches!(
                    schema.field(i).data_type(),
                    DataType::Float16 | DataType::Float32 | DataType::Float64
                )
            }))
            .collect::<Vec<_>>();
        assert!(others.len() >= 2, "{others:?}");
        payload_columns_of(all_types.project(&supported).unwrap(), true).await;
        payload_columns_of(all_types.project(&others).unwrap(), false).await;
    }

    /// The rows of `batches`, displayed, sorted.
    fn displayed_rows(batches: &[RecordBatch]) -> Vec<String> {
        use arrow::util::display::{ArrayFormatter, FormatOptions};
        let options = FormatOptions::default().with_null("NULL");
        let mut rows = batches
            .iter()
            .flat_map(|batch| {
                let formatters = batch
                    .columns()
                    .iter()
                    .map(|column| ArrayFormatter::try_new(column.as_ref(), &options))
                    .collect::<std::result::Result<Vec<_>, _>>()
                    .unwrap();
                (0..batch.num_rows())
                    .map(|row| {
                        formatters
                            .iter()
                            .map(|formatter| formatter.value(row).to_string())
                            .collect::<Vec<_>>()
                            .join("|")
                    })
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();
        rows.sort_unstable();
        rows
    }

    /// Joins of `batch` with itself on `i32`, compared with DataFusion byte
    /// for byte if `bytes`, otherwise by the display of their rows.
    async fn payload_columns_of(batch: RecordBatch, bytes: bool) {
        let make = |join_type: JoinType| {
            let batch = batch.clone();
            move || -> Result<Arc<dyn ExecutionPlan>> {
                let scan = |batch: &RecordBatch| {
                    let half = batch.num_rows() / 2;
                    MemorySourceConfig::try_new_exec(
                        &[
                            vec![batch.slice(0, half)],
                            vec![batch.slice(half, batch.num_rows() - half)],
                        ],
                        batch.schema(),
                        None,
                    )
                };
                let left = scan(&batch)?;
                let right = scan(&batch.slice(50, 250))?;
                let on =
                    vec![(col("i32", &left.schema())?, col("i32", &right.schema())?)];
                Ok(Arc::new(HashJoinExec::try_new(
                    Arc::new(CoalescePartitionsExec::new(left)),
                    right,
                    on,
                    None,
                    &join_type,
                    None,
                    PartitionMode::CollectLeft,
                    NullEquality::NullEqualsNull,
                    false,
                )?))
            }
        };
        for join_type in [
            JoinType::Inner,
            JoinType::Left,
            JoinType::Full,
            JoinType::LeftSemi,
            JoinType::LeftMark,
        ] {
            if bytes {
                let root = assert_lowered_plan_byte_eq(
                    make(join_type),
                    lower,
                    task_ctx(),
                    false,
                )
                .await;
                assert_eq!(count::<PhysicalHashJoin>(&root), 1);
            } else {
                let make = make(join_type);
                let expected = collect(make().unwrap(), task_ctx()).await.unwrap();
                let root = lower(&make().unwrap()).unwrap();
                assert_eq!(count::<PhysicalHashJoin>(&root), 1);
                let actual = with_timeout(Executor::collect(root, task_ctx(), 4))
                    .await
                    .unwrap();
                assert_eq!(actual[0].schema(), expected[0].schema());
                assert_eq!(displayed_rows(&actual), displayed_rows(&expected));
            }
        }
    }

    // ---------------------------------------------------------------------
    // Pipelines and execution
    // ---------------------------------------------------------------------

    fn source(schema: SchemaRef, batches: Vec<RecordBatch>) -> Arc<TestSource> {
        Arc::new(TestSource::with_schema(
            schema,
            batches
                .into_iter()
                .map(DataChunk::from_record_batch)
                .collect(),
        ))
    }

    fn join_of(
        build: Arc<dyn PhysicalOperator>,
        probe: Arc<dyn PhysicalOperator>,
        join_type: JoinType,
    ) -> Result<Arc<dyn PhysicalOperator>> {
        join_on_key(build, probe, join_type, "k")
    }

    /// A join of the test inputs `build` and `probe` on `l<key> = r<key>`.
    fn join_on_key(
        build: Arc<dyn PhysicalOperator>,
        probe: Arc<dyn PhysicalOperator>,
        join_type: JoinType,
        key: &str,
    ) -> Result<Arc<dyn PhysicalOperator>> {
        let on = vec![(
            col(&format!("l{key}"), &build.schema())?,
            col(&format!("r{key}"), &probe.schema())?,
        )];
        Ok(Arc::new(PhysicalHashJoin::try_new(
            build,
            probe,
            on,
            None,
            join_type,
            None,
            NullEquality::NullEqualsNothing,
            false,
        )?))
    }

    #[test]
    fn pipelines_of_a_hash_join() -> Result<()> {
        let build = source(data_schema("l"), vec![]);
        let probe = source(data_schema("r"), vec![]);
        let inner = join_of(
            Arc::clone(&build) as _,
            Arc::clone(&probe) as _,
            JoinType::Inner,
        )?;
        assert_eq!(
            format_pipelines(&inner)?,
            "MetaPipeline 0 (sink: ResultSink) children: [1]\n  \
             Pipeline 0: TestSource -> PhysicalHashJoin -> ResultSink depends on meta pipelines [1]\n\
             MetaPipeline 1 (sink: PhysicalHashJoin)\n  \
             Pipeline 1: TestSource -> PhysicalHashJoin\n  \
             Pipeline 2: JoinHashTableBuild -> PhysicalHashJoin depends on pipelines [1]\n"
        );
        // The unmatched build rows are emitted after the probe pipeline.
        let left = join_of(build as _, probe as _, JoinType::Left)?;
        assert_eq!(
            format_pipelines(&left)?,
            "MetaPipeline 0 (sink: ResultSink) children: [1]\n  \
             Pipeline 0: TestSource -> PhysicalHashJoin -> ResultSink depends on meta pipelines [1]\n  \
             Pipeline 3: PhysicalHashJoin -> ResultSink depends on pipelines [0]\n\
             MetaPipeline 1 (sink: PhysicalHashJoin)\n  \
             Pipeline 1: TestSource -> PhysicalHashJoin\n  \
             Pipeline 2: JoinHashTableBuild -> PhysicalHashJoin depends on pipelines [1]\n"
        );
        let join = left.downcast_ref::<PhysicalHashJoin>().unwrap();
        assert!(join.is_sink() && join.is_source() && join.parallel_operator());
        assert_eq!(join.join_type(), JoinType::Left);
        assert_eq!(join.on().len(), 1);
        assert!(join.filter().is_none() && join.projection().is_none());
        assert!(!join.null_aware());
        assert_eq!(join.null_equality(), NullEquality::NullEqualsNothing);
        assert_eq!(
            join.description(),
            "PhysicalHashJoin: join_type=Left, on=[(lk@1, rk@1)]"
        );
        assert!(!inner.is_source());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn the_probe_side_is_not_executed_without_possible_output() -> Result<()> {
        // DataFusion does not read the probe side if the build side has no
        // (matchable) rows and the join cannot produce output.
        let null_keys = {
            let mut batch = data_batch("l", 0..10, 0);
            let mut columns = batch.columns().to_vec();
            columns[1] = Arc::new(Int32Array::new_null(10));
            batch = RecordBatch::try_new(data_schema("l"), columns)?;
            batch
        };
        for (build_batches, join_type, executed) in [
            (vec![], JoinType::Inner, false),
            (vec![], JoinType::LeftAnti, false),
            (vec![], JoinType::RightSemi, false),
            (vec![], JoinType::Right, true),
            (vec![], JoinType::RightAnti, true),
            (vec![null_keys.clone()], JoinType::Inner, false),
            (vec![null_keys.clone()], JoinType::LeftSemi, false),
            (vec![null_keys.clone()], JoinType::Left, true),
            (vec![null_keys], JoinType::LeftAnti, true),
        ] {
            let build = source(data_schema("l"), build_batches);
            let probe = source(
                data_schema("r"),
                vec![data_batch("r", 0..100, 0), data_batch("r", 100..200, 0)],
            );
            let root = join_of(build as _, Arc::clone(&probe) as _, join_type)?;
            with_timeout(Executor::collect(root, task_ctx(), 4)).await?;
            assert_eq!(probe.produced() > 0, executed, "{join_type}");
        }
        Ok(())
    }

    #[test]
    fn pipelines_of_joins_on_the_probe_side() -> Result<()> {
        // `t1 JOIN (t2 JOIN t3)`: the build side of the inner join (meta
        // pipeline 2) waits for the build side of the outer join (1).
        let inner = join_of(
            source(data_schema("l"), vec![]) as _,
            source(data_schema("r"), vec![]) as _,
            JoinType::Inner,
        )?;
        let outer = join_of(
            source(data_schema("l"), vec![]) as _,
            inner,
            JoinType::Inner,
        )?;
        assert_eq!(
            format_pipelines(&outer)?,
            "MetaPipeline 0 (sink: ResultSink) children: [1, 2]\n  \
             Pipeline 0: TestSource -> PhysicalHashJoin -> PhysicalHashJoin -> ResultSink depends on meta pipelines [1, 2]\n\
             MetaPipeline 1 (sink: PhysicalHashJoin)\n  \
             Pipeline 1: TestSource -> PhysicalHashJoin\n  \
             Pipeline 2: JoinHashTableBuild -> PhysicalHashJoin depends on pipelines [1]\n\
             MetaPipeline 2 (sink: PhysicalHashJoin)\n  \
             Pipeline 3: TestSource -> PhysicalHashJoin depends on meta pipelines [1]\n  \
             Pipeline 4: JoinHashTableBuild -> PhysicalHashJoin depends on pipelines [3] depends on meta pipelines [1]\n"
        );
        // The scan of the unmatched build rows of an inner `Full` join, in the
        // meta pipeline of the outer join, runs after its probe pipeline.
        let inner = join_of(
            source(data_schema("l"), vec![]) as _,
            source(data_schema("r"), vec![]) as _,
            JoinType::Full,
        )?;
        let outer = join_of(
            source(data_schema("l"), vec![]) as _,
            inner,
            JoinType::Inner,
        )?;
        assert_eq!(
            format_pipelines(&outer)?,
            "MetaPipeline 0 (sink: ResultSink) children: [1, 2]\n  \
             Pipeline 0: TestSource -> PhysicalHashJoin -> PhysicalHashJoin -> ResultSink depends on meta pipelines [1, 2]\n  \
             Pipeline 5: PhysicalHashJoin -> PhysicalHashJoin -> ResultSink depends on pipelines [0] depends on meta pipelines [1]\n\
             MetaPipeline 1 (sink: PhysicalHashJoin)\n  \
             Pipeline 1: TestSource -> PhysicalHashJoin\n  \
             Pipeline 2: JoinHashTableBuild -> PhysicalHashJoin depends on pipelines [1]\n\
             MetaPipeline 2 (sink: PhysicalHashJoin)\n  \
             Pipeline 3: TestSource -> PhysicalHashJoin depends on meta pipelines [1]\n  \
             Pipeline 4: JoinHashTableBuild -> PhysicalHashJoin depends on pipelines [3] depends on meta pipelines [1]\n"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn pipeline_breakers_of_the_probe_side_wait_for_the_build_side() -> Result<()> {
        // A join of an empty build side and of the probe side `inner`, a
        // join of two non-empty inputs. If the outer join cannot produce
        // output without build rows, DataFusion does not read its probe side,
        // and so neither input of `inner`: the build pipelines of `inner`
        // wait for the build side of the outer join, and are canceled.
        for (join_type, executed) in [
            (JoinType::Inner, false),
            (JoinType::LeftSemi, false),
            (JoinType::RightSemi, false),
            (JoinType::LeftMark, false),
            (JoinType::Right, true),
            (JoinType::RightAnti, true),
            (JoinType::Full, true),
        ] {
            for build_rows in [0, 50] {
                let nested_build =
                    source(data_schema("l"), vec![data_batch("l", 0..100, 0)]);
                let nested_probe =
                    source(data_schema("r"), vec![data_batch("r", 0..100, 0)]);
                let inner = join_of(
                    Arc::clone(&nested_build) as _,
                    Arc::clone(&nested_probe) as _,
                    JoinType::Inner,
                )?;
                let batches = if build_rows == 0 {
                    vec![]
                } else {
                    vec![data_batch("l", 0..build_rows, 0)]
                };
                let root =
                    join_of(source(data_schema("l"), batches) as _, inner, join_type)?;
                with_timeout(Executor::collect(root, task_ctx(), 4)).await?;
                let executed = executed || build_rows > 0;
                assert_eq!(nested_build.produced() > 0, executed, "{join_type}");
                assert_eq!(nested_probe.produced() > 0, executed, "{join_type}");
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn joins_can_be_executed_again() -> Result<()> {
        // The build side with 6000 rows in many blocks and partitions.
        let build_batches = (0..6)
            .map(|i| data_batch("l", i * 1000..(i + 1) * 1000, 0))
            .collect::<Vec<_>>();
        let probe_batches = (0..3)
            .map(|i| data_batch("r", i * 700..(i + 1) * 700, 1))
            .collect::<Vec<_>>();
        for join_type in [JoinType::Inner, JoinType::Full, JoinType::LeftMark] {
            let build = source(data_schema("l"), build_batches.clone());
            let probe = source(data_schema("r"), probe_batches.clone());
            let root = join_of(build as _, probe as _, join_type)?;
            let mut previous = None;
            for threads in [1, 4, 4] {
                let batches = with_timeout(Executor::collect(
                    Arc::clone(&root),
                    task_ctx(),
                    threads,
                ))
                .await?;
                let rows = batches.iter().map(RecordBatch::num_rows).sum::<usize>();
                assert!(rows > 0);
                if let Some(previous) = previous {
                    assert_eq!(rows, previous, "{join_type}, {threads} threads");
                }
                previous = Some(rows);
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn a_limit_above_the_join_stops_it() -> Result<()> {
        use crate::operators::limit::PhysicalLimit;
        // Every probe row matches many build rows; the `Left` join also has
        // a pipeline of the unmatched build rows, which the limit skips.
        let build_batches = (0..4)
            .map(|i| data_batch("l", i * 1000..(i + 1) * 1000, 0))
            .collect::<Vec<_>>();
        let probe_batches = (0..50)
            .map(|i| data_batch("r", i * 100..(i + 1) * 100, 1))
            .collect::<Vec<_>>();
        for join_type in [JoinType::Inner, JoinType::Left, JoinType::RightMark] {
            for preserve_order in [false, true] {
                let probe = source(data_schema("r"), probe_batches.clone());
                let join = join_of(
                    source(data_schema("l"), build_batches.clone()) as _,
                    Arc::clone(&probe) as _,
                    join_type,
                )?;
                let limit: Arc<dyn PhysicalOperator> = Arc::new(
                    PhysicalLimit::new(join, 0, Some(10))
                        .with_preserve_order(preserve_order),
                );
                let batches =
                    with_timeout(Executor::collect(limit, task_ctx(), 4)).await?;
                let rows = batches.iter().map(RecordBatch::num_rows).sum::<usize>();
                assert_eq!(rows, 10, "{join_type}");
                assert!(probe.produced() < 50, "{join_type}: {}", probe.produced());
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn the_build_side_is_charged_to_the_memory_pool() -> Result<()> {
        use datafusion_execution::runtime_env::RuntimeEnvBuilder;
        let build_batches = (0..10)
            .map(|i| data_batch("l", i * 1000..(i + 1) * 1000, 0))
            .collect::<Vec<_>>();
        let join = || {
            join_of(
                source(data_schema("l"), build_batches.clone()) as _,
                source(data_schema("r"), vec![data_batch("r", 0..100, 0)]) as _,
                JoinType::Inner,
            )
        };
        let limited = |limit: usize| -> Result<Arc<TaskContext>> {
            let runtime = RuntimeEnvBuilder::new()
                .with_memory_limit(limit, 1.0)
                .build_arc()?;
            Ok(Arc::new(TaskContext::default().with_runtime(runtime)))
        };
        let err = with_timeout(Executor::collect(join()?, limited(64 * 1024)?, 4))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("PhysicalHashJoin"), "{err}");
        let batches =
            with_timeout(Executor::collect(join()?, limited(64 * 1024 * 1024)?, 4))
                .await?;
        assert!(!batches.is_empty());
        Ok(())
    }

    /// A null-aware `LeftAnti` join (`k NOT IN`) of a build side with the key
    /// 1 and a probe side with a NULL key, with the residual filter `false`.
    fn not_in_with_a_filter() -> Result<Arc<dyn ExecutionPlan>> {
        let schema = |side: &str| {
            Arc::new(Schema::new(vec![
                Field::new(format!("{side}k"), DataType::Int32, true),
                Field::new(format!("{side}v"), DataType::Int64, false),
            ]))
        };
        let scan = |side: &str, key: Option<i32>| {
            let batch = RecordBatch::try_new(
                schema(side),
                vec![
                    Arc::new(Int32Array::from(vec![key])),
                    Arc::new(Int64Array::from(vec![10])),
                ],
            )
            .unwrap();
            MemorySourceConfig::try_new_exec(&[vec![batch]], schema(side), None).unwrap()
        };
        let (build, probe) = (scan("l", Some(1)), scan("r", None));
        let filter_schema =
            Arc::new(Schema::new(vec![Field::new("rv", DataType::Int64, false)]));
        let filter = JoinFilter::new(
            Arc::new(BinaryExpr::new(
                col("rv", &filter_schema)?,
                Operator::Lt,
                lit(0_i64),
            )),
            vec![ColumnIndex {
                index: 1,
                side: JoinSide::Right,
            }],
            filter_schema,
        );
        let on = vec![(col("lk", &build.schema())?, col("rk", &probe.schema())?)];
        Ok(Arc::new(HashJoinExec::try_new(
            build,
            probe,
            on,
            Some(filter),
            &JoinType::LeftAnti,
            None,
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            true,
        )?))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn null_aware_joins_it_does_not_implement_are_refused() -> Result<()> {
        // `1 NOT IN (SELECT k FROM probe WHERE false)` is true: DataFusion
        // keeps the build row.
        let plan = not_in_with_a_filter()?;
        let expected = collect(Arc::clone(&plan), task_ctx()).await?;
        let rows = expected.iter().map(RecordBatch::num_rows).sum::<usize>();
        assert_eq!(rows, 1);
        // The join is left to DataFusion...
        let root =
            assert_lowered_plan_byte_eq(not_in_with_a_filter, lower, task_ctx(), false)
                .await;
        assert_eq!(count::<PhysicalHashJoin>(&root), 0);
        // ... and cannot be created directly either: with a filter, several
        // keys or NULLs that are equal.
        let join = plan.downcast_ref::<HashJoinExec>().unwrap();
        let try_new = |on: Vec<(PhysicalExprRef, PhysicalExprRef)>,
                       filter: Option<JoinFilter>,
                       null_equality| {
            PhysicalHashJoin::try_new(
                Arc::new(BridgeOperator::new(Arc::clone(join.left()))),
                Arc::new(BridgeOperator::new(Arc::clone(join.right()))),
                on,
                filter,
                JoinType::LeftAnti,
                None,
                null_equality,
                true,
            )
        };
        let on = join.on().to_vec();
        let mut two_keys = on.clone();
        two_keys.extend(on.clone());
        for (on, filter, null_equality) in [
            (
                on.clone(),
                join.filter().cloned(),
                NullEquality::NullEqualsNothing,
            ),
            (two_keys, None, NullEquality::NullEqualsNothing),
            (on.clone(), None, NullEquality::NullEqualsNull),
        ] {
            let err = try_new(on, filter, null_equality).unwrap_err();
            assert!(matches!(err, DataFusionError::NotImplemented(_)), "{err}");
        }
        // Without filter, the build row is removed by the NULL probe key, by
        // the native join too.
        let native = try_new(on, None, NullEquality::NullEqualsNothing)?;
        let batches =
            with_timeout(Executor::collect(Arc::new(native), task_ctx(), 4)).await?;
        assert_eq!(batches.iter().map(RecordBatch::num_rows).sum::<usize>(), 0);
        Ok(())
    }

    /// A join on `k` whose probe side is repartitioned by `1 / d`, with `d`
    /// zero in every row: evaluating the repartitioning fails.
    fn join_with_a_failing_repartition() -> Result<Arc<dyn ExecutionPlan>> {
        let schema = |side: &str| {
            Arc::new(Schema::new(vec![
                Field::new(format!("{side}k"), DataType::Int32, true),
                Field::new(format!("{side}d"), DataType::Int32, false),
            ]))
        };
        let scan = |side: &str| {
            let batch = RecordBatch::try_new(
                schema(side),
                vec![
                    Arc::new(Int32Array::from(vec![1, 2, 3])),
                    Arc::new(Int32Array::from(vec![0, 0, 0])),
                ],
            )
            .unwrap();
            MemorySourceConfig::try_new_exec(&[vec![batch]], schema(side), None).unwrap()
        };
        let probe = scan("r");
        let inverse = Arc::new(BinaryExpr::new(
            lit(1_i32),
            Operator::Divide,
            col("rd", &probe.schema())?,
        ));
        let probe = Arc::new(RepartitionExec::try_new(
            probe,
            Partitioning::Hash(vec![inverse], 4),
        )?);
        let build = scan("l");
        let on = vec![(col("lk", &build.schema())?, col("rk", &probe.schema())?)];
        Ok(Arc::new(HashJoinExec::try_new(
            build,
            probe,
            on,
            None,
            &JoinType::Inner,
            None,
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )?))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn repartitions_that_can_fail_are_kept() -> Result<()> {
        // The repartitioning is left to DataFusion, which fails like the plan.
        let root = assert_lowered_plan_fails_like_datafusion(
            join_with_a_failing_repartition,
            lower,
            test_session_config(),
            "Divide by zero",
        )
        .await;
        let join = root
            .downcast_ref::<PhysicalHashJoin>()
            .expect("a native join");
        let bridged = join.probe_side().downcast_ref::<BridgeOperator>().unwrap();
        assert!(bridged.plan().is::<RepartitionExec>());

        // A repartitioning by the join keys (or round robin) is removed: the
        // join evaluates its keys on all rows too.
        let plan = join_with_a_failing_repartition()?;
        let join = plan.downcast_ref::<HashJoinExec>().unwrap();
        let probe = join.right().downcast_ref::<RepartitionExec>().unwrap();
        for partitioning in [
            Partitioning::Hash(vec![col("rk", &probe.schema())?], 4),
            Partitioning::RoundRobinBatch(3),
        ] {
            let repartition =
                RepartitionExec::try_new(Arc::clone(probe.input()), partitioning)?;
            let plan = join
                .builder()
                .with_new_children(vec![Arc::clone(join.left()), Arc::new(repartition)])?
                .build_exec()?;
            let native = PhysicalHashJoin::try_lower(&plan, |input| {
                Ok(Arc::new(BridgeOperator::new(Arc::clone(input))))
            })?
            .expect("a native join");
            let bridged = native
                .probe_side()
                .downcast_ref::<BridgeOperator>()
                .unwrap();
            assert!(bridged.plan().is::<DataSourceExec>());
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn dynamic_filters_over_fallible_probe_sides_are_left_to_datafusion()
    -> Result<()> {
        // The probe side `(k, d)` of `(1, 1), (2, 0)` is a Parquet file, whose
        // scan reads the dynamic filter of the join by `k`. Once DataFusion has
        // collected the build key 1, the scan skips the row of key 2 (with
        // `pushdown_filters`) before `1 / d` divides by its zero, while the
        // probe side of a native join, which sets the filter to the bounds of
        // its build keys only, could read rows that DataFusion's filter skips
        // (here, a key 2 of a build side of the keys 1 and 3), and divide by
        // zero.
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("p.parquet");
        let probe = RecordBatch::try_from_iter([
            ("k", Arc::new(Int64Array::from(vec![1, 2])) as ArrayRef),
            ("d", Arc::new(Int64Array::from(vec![1, 0])) as ArrayRef),
        ])?;
        write_parquet(&path, probe.schema(), &[probe], None)?;
        let path = path.to_str().expect("a UTF-8 path").to_string();
        let setup = async |ctx: &SessionContext| {
            let build = RecordBatch::try_from_iter([(
                "k",
                Arc::new(Int64Array::from(vec![1])) as ArrayRef,
            )])?;
            ctx.register_batch("b", build)?;
            ctx.register_parquet("p", &path, ParquetReadOptions::default())
                .await
        };
        let sql = "SELECT b.k, x.inv FROM b JOIN (SELECT k, 1 / d AS inv FROM p) x ON b.k = x.k";
        let mut config = test_session_config();
        config.options_mut().execution.parquet.pushdown_filters = true;
        // `b` is the build side, although it is not smaller than `p`.
        config.options_mut().optimizer.join_reordering = false;

        let ctx = datafusion_context(config.clone());
        setup(&ctx).await?;
        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        let join = find_hash_join(&plan).expect("a hash join");
        assert_eq!(join.dynamic_expressions_produced().len(), 1);
        let expected = run_sql(&ctx, sql).await?;
        assert_eq!(count_rows(&expected.batches), 1);
        // The join is left to DataFusion, with the same result.
        assert_join_parity_with_setup(sql, config.clone(), &setup, 0).await;

        // DataFusion computes `1 / d` in the scan above. The same join over a
        // `ProjectionExec` of `1 / d`, below which `FilterPushdown` pushes the
        // dynamic filter of every new plan into the scan.
        let scan = ctx.sql("SELECT k, d FROM p").await?;
        let scan = scan.create_physical_plan().await?;
        let build = ctx.sql("SELECT k FROM b").await?;
        let build = build.create_physical_plan().await?;
        let options = config.options().clone();
        let make_plan = || -> Result<Arc<dyn ExecutionPlan>> {
            let schema = scan.schema();
            let one_over_d = Arc::new(BinaryExpr::new(
                lit(1_i64),
                Operator::Divide,
                col("d", &schema)?,
            ));
            let probe: Arc<dyn ExecutionPlan> = Arc::new(ProjectionExec::try_new(
                [
                    (col("k", &schema)?, "k".to_string()),
                    (one_over_d as _, "inv".to_string()),
                ],
                Arc::clone(&scan),
            )?);
            let on = vec![(col("k", &build.schema())?, col("k", &probe.schema())?)];
            let join = HashJoinExec::try_new(
                Arc::clone(&build),
                probe,
                on,
                None,
                &JoinType::Inner,
                None,
                PartitionMode::CollectLeft,
                NullEquality::NullEqualsNothing,
                false,
            )?;
            FilterPushdown::new_post_optimization().optimize(Arc::new(join), &options)
        };
        let plan = make_plan()?;
        assert_eq!(plan.dynamic_expressions_produced().len(), 1);
        let probe = Arc::clone(plan.children()[1]);
        assert!(probe.is::<ProjectionExec>(), "{probe:?}");
        let batches = collect(make_plan()?, ctx.task_ctx()).await?;
        assert_eq!(count_rows(&batches), 1);
        let root =
            assert_lowered_plan_byte_eq(make_plan, lower, ctx.task_ctx(), false).await;
        assert!(root.is::<BridgeOperator>());

        // Without the dynamic filter, the join is native and both engines
        // divide by zero.
        config
            .options_mut()
            .optimizer
            .enable_join_dynamic_filter_pushdown = false;
        let ctx = datafusion_context(config);
        setup(&ctx).await?;
        let error = run_sql(&ctx, sql).await.expect_err("DataFusion fails");
        assert!(error.to_string().contains("Divide by zero"), "{error}");
        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        let root = lower(&plan)?;
        assert_eq!(count::<PhysicalHashJoin>(&root), 1);
        let exec = Arc::new(PipelineRootExec::try_new(root, plan)?);
        let error = with_timeout(collect(exec, ctx.task_ctx()))
            .await
            .expect_err("the pipeline engine fails");
        assert!(error.to_string().contains("Divide by zero"), "{error}");
        Ok(())
    }

    /// Write the Parquet table `p` of `(k, d, s, m)` to `dir`: two files of
    /// 2,000 rows in row groups of 100 rows, sorted by `k`, so that the
    /// dynamic filter of a join by `k` prunes most row groups (and with
    /// `pushdown_filters`, rows).
    fn write_parquet_table_p(dir: &std::path::Path) -> Result<()> {
        let properties = ::parquet::file::properties::WriterProperties::builder()
            .set_max_row_group_row_count(Some(100))
            .build();
        for (file, keys) in [("a", 0..2000), ("b", 2000..4000)] {
            let k = Int64Array::from_iter_values(keys.clone());
            let d = Int64Array::from_iter(
                keys.clone().map(|k| (k % 7 != 3).then_some(k % 5)),
            );
            let s = StringArray::from_iter_values(
                keys.clone().map(|k| format!("s{}", k % 13)),
            );
            let m =
                Decimal128Array::from_iter_values(keys.map(|k| i128::from(k % 89) * 25))
                    .with_precision_and_scale(15, 2)?;
            let batch = RecordBatch::try_from_iter([
                ("k", Arc::new(k) as ArrayRef),
                ("d", Arc::new(d) as ArrayRef),
                ("s", Arc::new(s) as ArrayRef),
                ("m", Arc::new(m) as ArrayRef),
            ])?;
            let path = dir.join("p").join(format!("{file}.parquet"));
            write_parquet(&path, batch.schema(), &[batch], Some(properties.clone()))?;
        }
        Ok(())
    }

    /// Register the Parquet table `p` of [`write_parquet_table_p`] in `dir`,
    /// and the memory table `b` of the keys `10, 20, ..., 300` (with a NULL
    /// key and a duplicate), with `w`.
    async fn register_tables_p_and_b(
        ctx: &SessionContext,
        dir: &std::path::Path,
    ) -> Result<()> {
        let keys = (1..=30)
            .map(|i| Some(i * 10))
            .chain([None, Some(150)])
            .collect::<Vec<_>>();
        let w = (0..keys.len() as i64).collect::<Vec<_>>();
        let build = RecordBatch::try_from_iter([
            ("k", Arc::new(Int64Array::from(keys)) as ArrayRef),
            ("w", Arc::new(Int64Array::from(w)) as ArrayRef),
        ])?;
        ctx.register_batch("b", build)?;
        let path = dir.join("p");
        ctx.register_parquet(
            "p",
            path.to_str().expect("a UTF-8 path"),
            ParquetReadOptions::default(),
        )
        .await
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn joins_that_produce_dynamic_filters_are_native() -> Result<()> {
        // Joins of the memory table `b` (the build side) with the Parquet
        // table `p`, whose scans read the dynamic filters of the joins. The
        // probe sides only evaluate expressions that cannot fail, so the joins
        // are native: they set the filters to the bounds of their build keys
        // (`10` and `300`), with which the scans skip the row groups of the
        // other keys, and mark them complete.
        let dir = tempfile::tempdir()?;
        write_parquet_table_p(dir.path())?;
        let setup =
            async |ctx: &SessionContext| register_tables_p_and_b(ctx, dir.path()).await;
        let queries = [
            "SELECT b.k, b.w, p.d, p.s FROM b JOIN p ON b.k = p.k",
            "SELECT b.w, x.v FROM b JOIN (SELECT k, d + 1 AS v FROM p WHERE s <> 's3') x ON b.k = x.k",
            "SELECT b.k, p.d FROM b JOIN p ON b.k = p.k AND p.d < b.w WHERE p.s LIKE 's1%'",
            "SELECT b.k, b.w, p.d FROM b LEFT JOIN p ON b.k = p.k AND p.d > 1",
            "SELECT b.k, b.w FROM b WHERE b.k IN (SELECT k FROM p WHERE s <> 's2')",
            "SELECT b.k, b.w FROM b WHERE EXISTS (SELECT 1 FROM p WHERE p.k = b.k AND p.d = 2)",
            "SELECT b.k, b.w FROM b WHERE NOT EXISTS (SELECT 1 FROM p WHERE p.k = b.k AND p.d = 4)",
            // No build rows: the join does not read its probe side.
            "SELECT b.k, p.d FROM b JOIN p ON b.k = p.k WHERE b.w > 1000",
        ];
        for pushdown_filters in [false, true] {
            let mut config = test_session_config();
            config.options_mut().execution.parquet.pushdown_filters = pushdown_filters;
            // `b` is the build side, although it is not smaller than `p`.
            config.options_mut().optimizer.join_reordering = false;
            for sql in queries {
                let ctx = datafusion_context(config.clone());
                setup(&ctx).await?;
                let plan = ctx.sql(sql).await?.create_physical_plan().await?;
                let join = find_hash_join(&plan).expect("a hash join");
                let filters = dynamic_filters(&join).expect("dynamic filters");
                assert_eq!(filters.len(), 1, "{sql}");
                assert!(!filters[0].is_complete(), "{sql}");
                // Native from top to bottom, also the Parquet scan.
                let root = lower(&plan)?;
                assert_eq!(count::<PhysicalHashJoin>(&root), 1, "{sql}");
                assert!(
                    bridged_plans(&root).is_empty(),
                    "{sql}\n{}",
                    format_operator_tree(&root)
                );
                let scan = find_parquet_scan(&plan).expect("a Parquet scan");
                let exec = Arc::new(PipelineRootExec::try_new(root, plan)?);
                with_timeout(collect(exec, ctx.task_ctx())).await?;
                assert!(filters[0].is_complete(), "{sql}");
                let filter = filters[0].current()?.to_string();
                if sql.contains("b.w > 1000") {
                    // No build row can match: the probe side is not read.
                    assert_eq!(filter, "false", "{sql}");
                } else {
                    assert_eq!(filter, "k@0 >= 10 AND k@0 <= 300", "{sql}");
                    let pruned = pruned(&scan, "files_ranges_pruned_statistics")
                        + pruned(&scan, "row_groups_pruned_statistics");
                    assert!(pruned > 0, "{sql}");
                }

                assert_join_parity_with_setup(sql, config.clone(), &setup, 1).await;
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn dynamic_filters_are_set_to_the_bounds_of_the_build_keys() -> Result<()> {
        // Joins of memory tables with the Parquet table `p`, whose scans read
        // the dynamic filters of the joins: they are set to the bounds of the
        // build keys with exact bounds (not floats), keep the probe rows with a
        // NULL key if NULL build keys match them (`IS NOT DISTINCT FROM`), and
        // are `false` if no build row can match.
        let dir = tempfile::tempdir()?;
        write_parquet_table_p(dir.path())?;
        let setup = async |ctx: &SessionContext| {
            register_tables_p_and_b(ctx, dir.path()).await?;
            let c = RecordBatch::try_from_iter([
                (
                    "s",
                    Arc::new(StringViewArray::from(vec![Some("s3"), Some("s11"), None]))
                        as ArrayRef,
                ),
                (
                    "x",
                    Arc::new(Float64Array::from(vec![1.0, -0.0, 4.0])) as ArrayRef,
                ),
            ])?;
            ctx.register_batch("c", c)?;
            Ok(())
        };
        let mut config = test_session_config();
        // The memory tables are the build sides.
        config.options_mut().optimizer.join_reordering = false;
        for (sql, expected) in [
            (
                "SELECT b.w, p.k FROM b JOIN p ON b.k = p.k AND b.w = p.d",
                "k@0 >= 10 AND k@0 <= 300 AND d@1 >= 0 AND d@1 <= 31",
            ),
            (
                "SELECT c.x, p.k FROM c JOIN p ON c.s = p.s",
                "s@1 >= s11 AND s@1 <= s3",
            ),
            (
                "SELECT c.s, p.k FROM c JOIN p ON c.x = CAST(p.d AS DOUBLE)",
                "true",
            ),
            (
                "SELECT b.w, p.k FROM b JOIN p ON b.k IS NOT DISTINCT FROM p.d",
                "d@1 >= 10 AND d@1 <= 300 OR d@1 IS NULL",
            ),
            // The build row of `w = 30` has a NULL key: the probe side of the
            // inner join is not read, the scan of the `Left` join reads nothing.
            (
                "SELECT b.w, p.k FROM b JOIN p ON b.k = p.k WHERE b.w = 30",
                "false",
            ),
            (
                "SELECT b.w, p.d FROM b LEFT JOIN p ON b.k = p.k WHERE b.w = 30",
                "false",
            ),
        ] {
            let ctx = datafusion_context(config.clone());
            setup(&ctx).await?;
            let plan = ctx.sql(sql).await?.create_physical_plan().await?;
            let join = find_hash_join(&plan).unwrap_or_else(|| {
                panic!("{sql}\n{}", displayable(plan.as_ref()).indent(true))
            });
            let filters = dynamic_filters(&join).expect("dynamic filters");
            assert_eq!(filters.len(), 1, "{sql}");
            let root = lower(&plan)?;
            assert_eq!(count::<PhysicalHashJoin>(&root), 1, "{sql}");
            let exec = Arc::new(PipelineRootExec::try_new(root, plan)?);
            with_timeout(collect(exec, ctx.task_ctx())).await?;
            assert!(filters[0].is_complete(), "{sql}");
            assert_eq!(filters[0].current()?.to_string(), expected, "{sql}");
            assert_join_parity_with_setup(sql, config.clone(), &setup, 1).await;
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn probe_sides_that_dynamic_filters_cannot_change() -> Result<()> {
        // Which probe inputs of a join with a dynamic filter only evaluate
        // expressions that cannot fail (see
        // `evaluates_only_infallible_expressions`), on the plans of joins of
        // `b` with `p` (or `b`).
        let dir = tempfile::tempdir()?;
        write_parquet_table_p(dir.path())?;
        let mut config = test_session_config();
        config.options_mut().execution.parquet.pushdown_filters = true;
        config.options_mut().optimizer.join_reordering = false;
        let ctx = datafusion_context(config);
        register_tables_p_and_b(&ctx, dir.path()).await?;
        for (probe, infallible) in [
            // A scan of memory or of a Parquet file (whose predicate has the
            // dynamic filter of the join).
            ("SELECT k FROM b", true),
            ("SELECT k, d FROM p", true),
            // Filters and projections.
            (
                "SELECT k, d + 1 AS d FROM p WHERE s = 's1' OR d IS NULL",
                true,
            ),
            ("SELECT k, d / 2 AS d FROM p WHERE s LIKE '%7'", true),
            (
                "SELECT k, d FROM p WHERE NOT (d = 1) AND k IN (1, 2, 3)",
                true,
            ),
            ("SELECT k, 10 / d AS d FROM p", false),
            ("SELECT k, d FROM p WHERE 10 / d > 1", false),
            ("SELECT k, CAST(s AS INT) AS d FROM p", false),
            ("SELECT k, upper(s) AS d FROM p", false),
            // Aggregations: of input rows, and of partial states.
            ("SELECT k, COUNT(*) AS d FROM p GROUP BY k", true),
            ("SELECT k, MAX(d + 1) AS d FROM p GROUP BY k", true),
            ("SELECT k, SUM(10 / d) AS d FROM p GROUP BY k", false),
            (
                "SELECT k, SUM(d) AS d, SUM(m) AS m, MIN(s) AS s FROM p GROUP BY k",
                true,
            ),
            // The average of decimals may overflow.
            ("SELECT k, AVG(m) AS d FROM p GROUP BY k", false),
            ("SELECT k, string_agg(s, ',') AS d FROM p GROUP BY k", false),
            // Joins: their keys and filters.
            ("SELECT p.k, b.w AS d FROM p JOIN b ON p.k = b.k + 1", true),
            (
                "SELECT p.k, b.w AS d FROM p JOIN b ON p.k = b.k AND p.d < b.w",
                true,
            ),
            (
                "SELECT p.k, b.w AS d FROM p JOIN b ON p.k = b.k AND 10 / p.d < b.w",
                false,
            ),
            // Sorts.
            ("SELECT k, d FROM p ORDER BY d, k", true),
            ("SELECT k, d FROM p ORDER BY 10 / d, k", false),
            // A window aggregation, whose expressions are not classified.
            (
                "SELECT k, ROW_NUMBER() OVER (ORDER BY k) AS d FROM p",
                false,
            ),
        ] {
            let plan = ctx.sql(probe).await?.create_physical_plan().await?;
            assert_eq!(
                evaluates_only_infallible_expressions(&plan),
                infallible,
                "{probe}\n{}",
                displayable(plan.as_ref()).indent(true)
            );
        }
        // The probe input of a join with a dynamic filter, with the filter in
        // the predicate of its scan.
        let sql =
            "SELECT b.k, x.d FROM b JOIN (SELECT k, d + 1 AS d FROM p) x ON b.k = x.k";
        let plan = ctx.sql(sql).await?.create_physical_plan().await?;
        let join = find_hash_join(&plan).expect("a hash join");
        assert_eq!(join.dynamic_expressions_produced().len(), 1);
        let probe = Arc::clone(join.children()[1]);
        assert!(evaluates_only_infallible_expressions(&probe));
        Ok(())
    }

    fn int32_k_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, true)]))
    }

    /// A scan of a single partition of `values` of `k` (without any batch if
    /// there are none).
    fn int32_k_scan(values: Vec<Option<i32>>) -> Result<Arc<dyn ExecutionPlan>> {
        int32_k_batches(if values.is_empty() {
            vec![]
        } else {
            vec![values]
        })
    }

    /// A scan of a single partition with a batch of `k` per element of
    /// `batches`, also the empty ones.
    fn int32_k_batches(batches: Vec<Vec<Option<i32>>>) -> Result<Arc<dyn ExecutionPlan>> {
        let schema = int32_k_schema();
        let batches = batches
            .into_iter()
            .map(|values| {
                let values = Arc::new(Int32Array::from(values));
                RecordBatch::try_new(Arc::clone(&schema), vec![values])
            })
            .collect::<std::result::Result<Vec<_>, _>>()?;
        Ok(MemorySourceConfig::try_new_exec(&[batches], schema, None)?)
    }

    /// A join of `build` and `probe` (of [`int32_k_schema`]) on `1 / 0 = k`:
    /// its build key fails without any row.
    fn join_on_one_over_zero(
        build: Arc<dyn ExecutionPlan>,
        probe: Arc<dyn ExecutionPlan>,
        join_type: JoinType,
        mode: PartitionMode,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let one_over_zero =
            Arc::new(BinaryExpr::new(lit(1_i32), Operator::Divide, lit(0_i32)));
        let on = vec![(one_over_zero as _, col("k", &probe.schema())?)];
        Ok(Arc::new(HashJoinExec::try_new(
            build,
            probe,
            on,
            None,
            &join_type,
            None,
            mode,
            NullEquality::NullEqualsNothing,
            false,
        )?))
    }

    /// A join of `build` and `probe` (of [`int32_k_schema`]) on `(1 / 0, k) =
    /// (k, k)`: its first build key fails without any row, and DataFusion
    /// evaluates its keys once its build side is collected (see
    /// `JoinSpec::defers_build_key_errors`).
    fn join_on_one_over_zero_and_k(
        build: Arc<dyn ExecutionPlan>,
        probe: Arc<dyn ExecutionPlan>,
        join_type: JoinType,
        mode: PartitionMode,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let one_over_zero =
            Arc::new(BinaryExpr::new(lit(1_i32), Operator::Divide, lit(0_i32)));
        let (k, probe_k) = (col("k", &build.schema())?, col("k", &probe.schema())?);
        let on = vec![(one_over_zero as _, Arc::clone(&probe_k)), (k, probe_k)];
        Ok(Arc::new(HashJoinExec::try_new(
            build,
            probe,
            on,
            None,
            &join_type,
            None,
            mode,
            NullEquality::NullEqualsNothing,
            false,
        )?))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn empty_build_sides_evaluate_their_keys() -> Result<()> {
        // DataFusion evaluates the build keys on its (concatenated) build
        // side, also when it is empty: a key that fails without any row fails
        // the join, whether it can produce output or not. A single key of an
        // integer type, which DataFusion also evaluates on every (empty)
        // build batch, is left to DataFusion.
        for join_type in ALL_JOIN_TYPES {
            for mode in [PartitionMode::CollectLeft, PartitionMode::Partitioned] {
                for probe in [vec![], vec![Some(1), None]] {
                    for single_key in [true, false] {
                        let make_plan = || {
                            let join = if single_key {
                                join_on_one_over_zero
                            } else {
                                join_on_one_over_zero_and_k
                            };
                            join(
                                int32_k_scan(vec![])?,
                                int32_k_scan(probe.clone())?,
                                join_type,
                                mode,
                            )
                        };
                        let root = assert_lowered_plan_fails_like_datafusion(
                            make_plan,
                            lower,
                            test_session_config(),
                            "Divide by zero",
                        )
                        .await;
                        assert_eq!(
                            root.is::<PhysicalHashJoin>(),
                            !single_key,
                            "{join_type} {mode:?}"
                        );
                    }
                }
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn canceled_joins_do_not_evaluate_their_build_keys() -> Result<()> {
        use crate::operators::limit::PhysicalLimit;
        // A limit of one row over a union of a scan of that row and a join
        // whose build side never produces anything. Once the limit has its
        // row, the join is not needed: DataFusion never collects its build
        // side, so neither engine evaluates its build key `1 / 0`.
        let make_plan = || -> Result<Arc<dyn ExecutionPlan>> {
            let join = join_on_one_over_zero_and_k(
                pending_plan(int32_k_schema(), 1)?,
                int32_k_scan(vec![Some(1)])?,
                JoinType::RightSemi,
                PartitionMode::CollectLeft,
            )?;
            let union = UnionExec::try_new(vec![join, int32_k_scan(vec![Some(42)])?])?;
            Ok(Arc::new(
                CoalescePartitionsExec::new(union).with_fetch(Some(1)),
            ))
        };
        let lower_limit = |plan: &Arc<dyn ExecutionPlan>| match PhysicalLimit::try_lower(
            plan, lower,
        )? {
            Some(limit) => Ok(Arc::new(limit) as Arc<dyn PhysicalOperator>),
            None => lower(plan),
        };
        let root =
            assert_lowered_plan_byte_eq(make_plan, lower_limit, task_ctx(), false).await;
        assert!(root.is::<PhysicalLimit>());
        assert_eq!(count::<PhysicalHashJoin>(&root), 1);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn joins_whose_probe_keys_fail_without_rows_are_left_to_datafusion()
    -> Result<()> {
        // DataFusion evaluates the probe key `1 / 0` on every probe batch: an
        // empty one fails the join, a probe side without batches does not. A
        // native join only sees the non-empty chunks of its probe side, so
        // the join is left to DataFusion.
        let join =
            |probe_batches: Vec<Vec<Option<i32>>>| -> Result<Arc<dyn ExecutionPlan>> {
                let build = int32_k_scan(vec![Some(1)])?;
                let probe = int32_k_batches(probe_batches)?;
                let one_over_zero =
                    Arc::new(BinaryExpr::new(lit(1_i32), Operator::Divide, lit(0_i32)));
                let on = vec![(col("k", &build.schema())?, one_over_zero as _)];
                Ok(Arc::new(HashJoinExec::try_new(
                    build,
                    probe,
                    on,
                    None,
                    &JoinType::Inner,
                    None,
                    PartitionMode::CollectLeft,
                    NullEquality::NullEqualsNothing,
                    false,
                )?))
            };
        let root = assert_lowered_plan_fails_like_datafusion(
            || join(vec![vec![]]),
            lower,
            test_session_config(),
            "Divide by zero",
        )
        .await;
        assert!(root.is::<BridgeOperator>());
        let root =
            assert_lowered_plan_byte_eq(|| join(vec![]), lower, task_ctx(), false).await;
        assert!(root.is::<BridgeOperator>());
        Ok(())
    }

    /// A `Partitioned` join of 4 partitions of build keys `[1]` and probe rows
    /// `(k, d)` of `(1, 1), (2, 0)` by `k`, with a projection of `k, 1 / d`
    /// above the repartitioning of the probe side, or below it.
    fn partitioned_join_over_one_over_d(
        join_type: JoinType,
        projection_below_repartition: bool,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let by_k = |input: Arc<dyn ExecutionPlan>| -> Result<Arc<dyn ExecutionPlan>> {
            let k = col("k", &input.schema())?;
            Ok(Arc::new(RepartitionExec::try_new(
                input,
                Partitioning::Hash(vec![k], 4),
            )?))
        };
        let one_over_d =
            |input: Arc<dyn ExecutionPlan>| -> Result<Arc<dyn ExecutionPlan>> {
                let schema = input.schema();
                let inverse = Arc::new(BinaryExpr::new(
                    lit(1_i32),
                    Operator::Divide,
                    col("d", &schema)?,
                ));
                Ok(Arc::new(ProjectionExec::try_new(
                    [
                        (col("k", &schema)?, "k".to_string()),
                        (inverse as _, "inv".to_string()),
                    ],
                    input,
                )?))
            };
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int32, true),
            Field::new("d", DataType::Int32, true),
        ]));
        let rows = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from(vec![1, 2])),
                Arc::new(Int32Array::from(vec![1, 0])),
            ],
        )?;
        let probe: Arc<dyn ExecutionPlan> =
            MemorySourceConfig::try_new_exec(&[vec![rows]], schema, None)?;
        let probe = if projection_below_repartition {
            by_k(one_over_d(probe)?)?
        } else {
            one_over_d(by_k(probe)?)?
        };
        let build = by_k(int32_k_scan(vec![Some(1)])?)?;
        let on = vec![(col("k", &build.schema())?, col("k", &probe.schema())?)];
        Ok(Arc::new(HashJoinExec::try_new(
            build,
            probe,
            on,
            None,
            &join_type,
            None,
            PartitionMode::Partitioned,
            NullEquality::NullEqualsNothing,
            false,
        )?))
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn partitioned_joins_read_no_probe_partitions_of_empty_build_partitions()
    -> Result<()> {
        // DataFusion does not read the probe partition of the empty build
        // partition of the row `(2, 0)` (unless all hashes collide, then all
        // rows are in one partition), so `1 / d` above the repartitioning
        // does not divide by its zero: the join is left to DataFusion.
        let above = || partitioned_join_over_one_over_d(JoinType::Inner, false);
        let root = if hashes_are_forced_to_collide() {
            assert_lowered_plan_fails_like_datafusion(
                above,
                lower,
                test_session_config(),
                "Divide by zero",
            )
            .await
        } else {
            assert_eq!(count_rows(&collect(above()?, task_ctx()).await?), 1);
            assert_lowered_plan_byte_eq(above, lower, task_ctx(), false).await
        };
        assert!(root.is::<BridgeOperator>());

        // Below the repartitioning, which reads all rows in DataFusion, and
        // with a join type that reads all probe partitions, both engines
        // divide by zero: the joins are native.
        for (join_type, projection_below_repartition) in [
            (JoinType::Inner, true),
            (JoinType::Right, false),
            (JoinType::RightAnti, false),
        ] {
            let root = assert_lowered_plan_fails_like_datafusion(
                || {
                    partitioned_join_over_one_over_d(
                        join_type,
                        projection_below_repartition,
                    )
                },
                lower,
                test_session_config(),
                "Divide by zero",
            )
            .await;
            assert!(root.is::<PhysicalHashJoin>(), "{join_type}");
        }
        Ok(())
    }

    /// A join of `build` and `probe` (of [`int32_k_schema`]) on `k = k`.
    fn join_on_k(
        build: Arc<dyn ExecutionPlan>,
        probe: Arc<dyn ExecutionPlan>,
        join_type: JoinType,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let on = vec![(col("k", &build.schema())?, col("k", &probe.schema())?)];
        Ok(Arc::new(HashJoinExec::try_new(
            build,
            probe,
            on,
            None,
            &join_type,
            None,
            PartitionMode::CollectLeft,
            NullEquality::NullEqualsNothing,
            false,
        )?))
    }

    /// A plan of a single partition of `partition`.
    fn partition_plan(
        partition: impl PartitionStream + 'static,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let schema = Arc::clone(partition.schema());
        Ok(Arc::new(StreamingTableExec::try_new(
            schema,
            vec![Arc::new(partition)],
            None,
            vec![],
            false,
            None,
        )?))
    }

    /// A partition that produces `batch`, then signals `consumed` once it is
    /// polled again (its batch was consumed), and then never produces
    /// anything again.
    #[derive(Debug)]
    struct SignalAfterBatch {
        batch: RecordBatch,
        consumed: Arc<tokio::sync::Notify>,
    }

    impl PartitionStream for SignalAfterBatch {
        fn schema(&self) -> &SchemaRef {
            self.batch.schema_ref()
        }

        fn execute(&self, _ctx: Arc<TaskContext>) -> SendableRecordBatchStream {
            let consumed = Arc::clone(&self.consumed);
            let after = futures::stream::once(async move {
                consumed.notify_one();
                futures::future::pending::<Result<RecordBatch>>().await
            });
            let stream = futures::stream::iter([Ok(self.batch.clone())]).chain(after);
            Box::pin(RecordBatchStreamAdapter::new(self.batch.schema(), stream))
        }
    }

    /// A partition that produces `batch` once `ready` is signaled.
    #[derive(Debug)]
    struct BatchWhenSignaled {
        batch: RecordBatch,
        ready: Arc<tokio::sync::Notify>,
    }

    impl PartitionStream for BatchWhenSignaled {
        fn schema(&self) -> &SchemaRef {
            self.batch.schema_ref()
        }

        fn execute(&self, _ctx: Arc<TaskContext>) -> SendableRecordBatchStream {
            let ready = Arc::clone(&self.ready);
            let batch = self.batch.clone();
            let stream = futures::stream::once(async move {
                ready.notified().await;
                Ok(batch)
            });
            Box::pin(RecordBatchStreamAdapter::new(self.batch.schema(), stream))
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn canceled_joins_do_not_build_their_table() -> Result<()> {
        use crate::operators::limit::PhysicalLimit;
        // A limit of one row over a union of a join whose build side produces
        // 20,000 rows and then nothing, and of a branch that produces its row
        // once those rows are consumed. Once the limit has its row, the join
        // is not needed: DataFusion never builds its hash table, and in a
        // pool of 500,000 bytes, its index would not fit next to its rows.
        let make_plan = || -> Result<Arc<dyn ExecutionPlan>> {
            let consumed = Arc::new(tokio::sync::Notify::new());
            let rows = (0..20_000).map(Some).collect::<Vec<_>>();
            let build = partition_plan(SignalAfterBatch {
                batch: RecordBatch::try_new(
                    int32_k_schema(),
                    vec![Arc::new(Int32Array::from(rows))],
                )?,
                consumed: Arc::clone(&consumed),
            })?;
            let join =
                join_on_k(build, int32_k_scan(vec![Some(1)])?, JoinType::RightSemi)?;
            let row = partition_plan(BatchWhenSignaled {
                batch: RecordBatch::try_new(
                    int32_k_schema(),
                    vec![Arc::new(Int32Array::from(vec![42]))],
                )?,
                ready: consumed,
            })?;
            let union = UnionExec::try_new(vec![join, row])?;
            Ok(Arc::new(
                CoalescePartitionsExec::new(union).with_fetch(Some(1)),
            ))
        };
        let lower_limit = |plan: &Arc<dyn ExecutionPlan>| match PhysicalLimit::try_lower(
            plan, lower,
        )? {
            Some(limit) => Ok(Arc::new(limit) as Arc<dyn PhysicalOperator>),
            None => lower(plan),
        };
        let task_ctx = |pool: Arc<dyn MemoryPool>| -> Result<Arc<TaskContext>> {
            let runtime = RuntimeEnvBuilder::new()
                .with_memory_pool(pool)
                .build_arc()?;
            Ok(Arc::new(
                TaskContext::default()
                    .with_session_config(test_session_config())
                    .with_runtime(runtime),
            ))
        };
        // The canceled join collects its rows, but reserves no memory for its
        // table.
        let peaks = Arc::new(PeakPool::default());
        let root = assert_lowered_plan_byte_eq(
            make_plan,
            lower_limit,
            task_ctx(Arc::clone(&peaks) as _)?,
            false,
        )
        .await;
        assert!(root.is::<PhysicalLimit>());
        assert_eq!(count::<PhysicalHashJoin>(&root), 1);
        assert!(peaks.peak("PhysicalHashJoin") > 0);
        assert_eq!(peaks.peak("JoinHashTable"), 0);
        // In a pool in which everything else fits, but not the index of
        // 20,000 rows (at least 20 bytes per row: a slot of 8 bytes for every
        // other row and a chain link of 4 bytes), both engines return 42.
        let limit = peaks.sum_of_peaks() + 200_000;
        assert_lowered_plan_byte_eq(
            make_plan,
            lower_limit,
            task_ctx(Arc::new(GreedyMemoryPool::new(limit)))?,
            false,
        )
        .await;
        Ok(())
    }

    /// A CSV scan of `k` of a file in an object store that is not registered:
    /// creating its stream fails.
    fn unregistered_csv_scan() -> Result<Arc<dyn ExecutionPlan>> {
        let config = FileScanConfigBuilder::new(
            ObjectStoreUrl::parse("unregistered://store")?,
            Arc::new(CsvSource::new(int32_k_schema())),
        )
        .with_file(PartitionedFile::new("data.csv", 10))
        .build();
        Ok(DataSourceExec::from_data_source(config))
    }

    /// A partition of `k` that counts the streams created of it, and records
    /// whether one is polled (it never produces anything).
    #[derive(Debug)]
    struct TrackedPartition {
        schema: SchemaRef,
        created: Arc<AtomicUsize>,
        polled: Arc<AtomicBool>,
    }

    impl PartitionStream for TrackedPartition {
        fn schema(&self) -> &SchemaRef {
            &self.schema
        }

        fn execute(&self, _ctx: Arc<TaskContext>) -> SendableRecordBatchStream {
            self.created.fetch_add(1, Ordering::Relaxed);
            let polled = Arc::clone(&self.polled);
            let stream = futures::stream::once(async move {
                polled.store(true, Ordering::Relaxed);
                futures::future::pending::<Result<RecordBatch>>().await
            });
            Box::pin(RecordBatchStreamAdapter::new(
                Arc::clone(&self.schema),
                stream,
            ))
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn joins_without_output_create_their_probe_streams() -> Result<()> {
        // `HashJoinExec::execute` creates the stream of its probe side, also
        // when its build side is empty and it does not read it: creating the
        // stream of a scan of an object store that is not registered fails,
        // whether the scan is native (CSV) or bridged (JSON).
        type MakePlan = fn() -> Result<Arc<dyn ExecutionPlan>>;
        let probes: [MakePlan; 2] = [unregistered_csv_scan, || {
            unregistered_store_scan(int32_k_schema())
        }];
        for probe in probes {
            for join_type in [JoinType::Inner, JoinType::LeftSemi, JoinType::RightSemi] {
                let make_plan = || join_on_k(int32_k_scan(vec![])?, probe()?, join_type);
                let root = assert_lowered_plan_fails_like_datafusion(
                    make_plan,
                    lower,
                    test_session_config(),
                    UNREGISTERED_STORE,
                )
                .await;
                assert!(root.is::<PhysicalHashJoin>(), "{join_type}");
            }
        }

        // The streams are created, but not read: reading a file that does not
        // exist would fail.
        let make_plan =
            || join_on_k(int32_k_scan(vec![])?, missing_csv_scan()?, JoinType::Inner);
        let root = assert_lowered_plan_byte_eq(make_plan, lower, task_ctx(), false).await;
        assert!(root.is::<PhysicalHashJoin>());

        // A probe input whose streams cannot be created twice safely is left
        // to DataFusion, which creates them once, and does not poll them.
        let created = Arc::new(AtomicUsize::new(0));
        let polled = Arc::new(AtomicBool::new(false));
        let make_plan = || {
            let probe = partition_plan(TrackedPartition {
                schema: int32_k_schema(),
                created: Arc::clone(&created),
                polled: Arc::clone(&polled),
            })?;
            join_on_k(int32_k_scan(vec![])?, probe, JoinType::Inner)
        };
        let root = assert_lowered_plan_byte_eq(make_plan, lower, task_ctx(), false).await;
        assert!(root.is::<BridgeOperator>());
        assert_eq!(created.load(Ordering::Relaxed), 2);
        assert!(!polled.load(Ordering::Relaxed));
        Ok(())
    }

    /// A CSV scan of `k` of a local file that does not exist: its stream is
    /// created, but reading it fails.
    fn missing_csv_scan() -> Result<Arc<dyn ExecutionPlan>> {
        let config = FileScanConfigBuilder::new(
            ObjectStoreUrl::local_filesystem(),
            Arc::new(CsvSource::new(int32_k_schema())),
        )
        .with_file(PartitionedFile::new("/does/not/exist/data.csv", 10))
        .build();
        Ok(DataSourceExec::from_data_source(config))
    }

    /// A partition of `k` that ends without producing anything after `delay`.
    #[derive(Debug)]
    struct EmptyAfter {
        schema: SchemaRef,
        delay: Duration,
    }

    impl PartitionStream for EmptyAfter {
        fn schema(&self) -> &SchemaRef {
            &self.schema
        }

        fn execute(&self, _ctx: Arc<TaskContext>) -> SendableRecordBatchStream {
            let delay = self.delay;
            let stream = futures::stream::once(tokio::time::sleep(delay))
                .filter_map(|()| futures::future::ready(None::<Result<RecordBatch>>));
            Box::pin(RecordBatchStreamAdapter::new(
                Arc::clone(&self.schema),
                stream,
            ))
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn probe_sides_with_pipeline_breakers_wait_for_the_build_side() -> Result<()> {
        // An inner join whose build side ends empty after a while, over a join
        // on `(1 / 0, k) = (k, k)` whose build side is not empty, and whose
        // keys DataFusion evaluates once its build side is collected.
        // DataFusion reads the probe side of the outer join only once its
        // build side is collected: never, as it is empty, so it never collects
        // the build side of the inner join (nor divides by zero). Both joins
        // are native: the pipelines of the build side of the inner join wait
        // for the build side of the outer join, and are canceled with it.
        let make_plan = |join_type: JoinType| -> Result<Arc<dyn ExecutionPlan>> {
            let nested = join_on_one_over_zero_and_k(
                int32_k_scan(vec![Some(1)])?,
                int32_k_scan(vec![Some(1)])?,
                JoinType::RightSemi,
                PartitionMode::CollectLeft,
            )?;
            let build = partition_plan(EmptyAfter {
                schema: int32_k_schema(),
                delay: Duration::from_millis(200),
            })?;
            join_on_k(build, nested, join_type)
        };
        for join_type in [JoinType::Inner, JoinType::LeftSemi, JoinType::RightSemi] {
            let root = assert_lowered_plan_byte_eq(
                || make_plan(join_type),
                lower,
                task_ctx(),
                false,
            )
            .await;
            let join = root
                .downcast_ref::<PhysicalHashJoin>()
                .expect("a native join");
            assert!(join.probe_side().is::<PhysicalHashJoin>(), "{join_type}");
        }

        // A `Right` join reads its probe side also if its build side is
        // empty, once it is collected: both engines divide by zero then.
        for join_type in [JoinType::Right, JoinType::RightAnti, JoinType::Full] {
            let root = assert_lowered_plan_fails_like_datafusion(
                || make_plan(join_type),
                lower,
                test_session_config(),
                "Divide by zero",
            )
            .await;
            let join = root.downcast_ref::<PhysicalHashJoin>().unwrap();
            assert!(join.probe_side().is::<PhysicalHashJoin>(), "{join_type}");
        }
        Ok(())
    }

    /// Assert that `execution` neither completes nor fails for a while (then
    /// drop it).
    async fn assert_pending(execution: impl Future<Output = Result<Vec<RecordBatch>>>) {
        if let Ok(result) =
            tokio::time::timeout(Duration::from_millis(500), execution).await
        {
            panic!("The execution completed: {:?}", result.map(|b| b.len()));
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn probe_streams_are_created_before_the_build_side_is_read() -> Result<()> {
        // `HashJoinExec::execute` creates the stream of its probe side right
        // away: with a build side that never produces anything, creating the
        // stream of a scan of an object store that is not registered fails
        // the join at once.
        let make_plan = || {
            join_on_k(
                pending_plan(int32_k_schema(), 1)?,
                unregistered_csv_scan()?,
                JoinType::Inner,
            )
        };
        let root = assert_lowered_plan_fails_like_datafusion(
            make_plan,
            lower,
            test_session_config(),
            UNREGISTERED_STORE,
        )
        .await;
        assert!(root.is::<PhysicalHashJoin>());

        // A probe side that can be created (but would fail when read) is not
        // read: neither engine completes, nor fails.
        let make_plan = || {
            join_on_k(
                pending_plan(int32_k_schema(), 1)?,
                missing_csv_scan()?,
                JoinType::Inner,
            )
        };
        assert_pending(collect(make_plan()?, task_ctx())).await;
        let plan = make_plan()?;
        let root = lower(&plan)?;
        assert!(root.is::<PhysicalHashJoin>());
        let exec = Arc::new(PipelineRootExec::try_new(root, plan)?);
        assert_pending(collect(exec, task_ctx())).await;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn probe_streams_are_created_before_pipeline_breakers_of_the_build_side()
    -> Result<()> {
        // The build side of the join is a native join whose own build side
        // never produces anything. DataFusion creates the stream of the probe
        // side right away, and fails: so does the native join, before its
        // build pipelines (which wait for the nested build side) start.
        let make_plan = || {
            let nested = join_on_k(
                pending_plan(int32_k_schema(), 1)?,
                int32_k_scan(vec![Some(1)])?,
                JoinType::RightSemi,
            )?;
            join_on_k(nested, unregistered_csv_scan()?, JoinType::Inner)
        };
        let root = assert_lowered_plan_fails_like_datafusion(
            make_plan,
            lower,
            test_session_config(),
            UNREGISTERED_STORE,
        )
        .await;
        let join = root.downcast_ref::<PhysicalHashJoin>().unwrap();
        assert!(join.build_side().is::<PhysicalHashJoin>());
        // The pipeline that creates the probe streams depends on nothing.
        let pipelines = format_pipelines(&root)?;
        assert!(
            pipelines.contains("Pipeline 3: ProbeStreamsInit -> PhysicalHashJoin\n"),
            "{pipelines}"
        );
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn probe_streams_are_created_on_a_copy_of_the_probe_input() -> Result<()> {
        // Creating the streams of a `CoalescePartitionsExec` starts tasks that
        // read its input: here, the channels of a `RepartitionExec` by a
        // literal, which the native join reads (bridged). The streams created
        // before the build side is read are those of a copy with its own
        // state, so that the probe side still reads all of its rows.
        let make_plan = || {
            let schema = int32_k_schema();
            let repartition = RepartitionExec::try_new(
                int32_k_scan(vec![Some(1), Some(2), Some(3)])?,
                Partitioning::Hash(vec![lit(7_i32)], 32),
            )?;
            let probe = Arc::new(CoalescePartitionsExec::new(Arc::new(repartition)));
            assert_eq!(probe.schema(), schema);
            join_on_k(
                int32_k_scan(vec![Some(1), Some(2), Some(3)])?,
                probe,
                JoinType::Inner,
            )
        };
        let batches = collect(make_plan()?, task_ctx()).await?;
        assert_eq!(count_rows(&batches), 3);
        let root = assert_lowered_plan_byte_eq(make_plan, lower, task_ctx(), false).await;
        let join = root.downcast_ref::<PhysicalHashJoin>().unwrap();
        let probe = join.probe_side().downcast_ref::<BridgeOperator>().unwrap();
        assert!(probe.plan().is::<RepartitionExec>());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn probe_streams_that_cannot_fail_are_not_created() -> Result<()> {
        // The probe inputs whose streams are created without failing: scans of
        // memory, of no rows and of local files, below nodes that create their
        // streams without failing.
        let dir = tempfile::tempdir()?;
        write_parquet_table_p(dir.path())?;
        let ctx = datafusion_context(test_session_config());
        register_tables_p_and_b(&ctx, dir.path()).await?;
        for (sql, cannot_fail) in [
            ("SELECT k FROM b", true),
            ("SELECT k, d FROM p WHERE s <> 's1'", true),
            (
                "SELECT p.k, COUNT(*) FROM p JOIN b ON p.k = b.k GROUP BY p.k ORDER BY p.k",
                true,
            ),
            ("SELECT k FROM b UNION ALL SELECT k FROM p", true),
            ("SELECT k FROM p ORDER BY k LIMIT 10", true),
            // A window, which is not known to create its streams without failing.
            ("SELECT k, ROW_NUMBER() OVER (ORDER BY k) FROM p", false),
        ] {
            let plan = ctx.sql(sql).await?.create_physical_plan().await?;
            assert_eq!(
                probe_streams_cannot_fail(&plan),
                cannot_fail,
                "{sql}\n{}",
                displayable(plan.as_ref()).indent(true)
            );
        }
        let no_partitions =
            MemorySourceConfig::try_new_exec(&[], int32_k_schema(), None)?;
        let empty_after = EmptyAfter {
            schema: int32_k_schema(),
            delay: Duration::ZERO,
        };
        for (name, plan, cannot_fail) in [
            (
                "scan of an unregistered store",
                unregistered_csv_scan()?,
                false,
            ),
            (
                "JSON scan",
                unregistered_store_scan(int32_k_schema())?,
                false,
            ),
            ("scan of a local file", missing_csv_scan()?, true),
            ("streaming table", partition_plan(empty_after)?, false),
            (
                "coalescing of no partitions",
                Arc::new(CoalescePartitionsExec::new(no_partitions)) as _,
                false,
            ),
            (
                "empty",
                Arc::new(EmptyExec::new(int32_k_schema())) as _,
                true,
            ),
        ] {
            assert_eq!(probe_streams_cannot_fail(&plan), cannot_fail, "{name}");
        }

        // A join of scans of memory has no pipeline that creates its probe
        // streams.
        let root = lower(&JoinCase::new(JoinType::Inner, vec!["k"]).plan()?)?;
        let pipelines = format_pipelines(&root)?;
        assert!(!pipelines.contains("ProbeStreamsInit"), "{pipelines}");

        // In a runtime without the local file system, creating the streams of
        // a scan of local files fails: the native join creates them, and fails
        // like DataFusion, also without build rows (and without reading its
        // probe side).
        let registry = DefaultObjectStoreRegistry::new();
        registry.deregister_store(ObjectStoreUrl::local_filesystem().as_ref())?;
        let runtime = RuntimeEnvBuilder::new()
            .with_object_store_registry(Arc::new(registry))
            .build_arc()?;
        let task_ctx = Arc::new(
            TaskContext::default()
                .with_session_config(test_session_config())
                .with_runtime(runtime),
        );
        let make_plan =
            || join_on_k(int32_k_scan(vec![])?, missing_csv_scan()?, JoinType::Inner);
        let error = collect(make_plan()?, Arc::clone(&task_ctx))
            .await
            .expect_err("DataFusion fails");
        assert!(
            error.to_string().contains("No suitable object store"),
            "{error}"
        );
        let plan = make_plan()?;
        let root = lower(&plan)?;
        assert!(root.is::<PhysicalHashJoin>());
        let exec = Arc::new(PipelineRootExec::try_new(root, plan)?);
        let native = with_timeout(collect(exec, task_ctx))
            .await
            .expect_err("the pipeline engine fails");
        assert!(
            native.to_string().contains("No suitable object store"),
            "{native}"
        );
        Ok(())
    }

    /// A partition that produces `batch` after `delay`.
    #[derive(Debug)]
    struct DelayedBatch {
        batch: RecordBatch,
        delay: Duration,
    }

    impl PartitionStream for DelayedBatch {
        fn schema(&self) -> &SchemaRef {
            self.batch.schema_ref()
        }

        fn execute(&self, _ctx: Arc<TaskContext>) -> SendableRecordBatchStream {
            let (batch, delay) = (self.batch.clone(), self.delay);
            let stream = futures::stream::once(async move {
                tokio::time::sleep(delay).await;
                Ok(batch)
            });
            Box::pin(RecordBatchStreamAdapter::new(self.batch.schema(), stream))
        }
    }

    /// A limit of one row over a union of `branch` (of [`int32_k_schema`])
    /// and of a branch that produces the row 42 after 200 ms: once the limit
    /// has that row, `branch` is canceled.
    fn limit_over_delayed_42(
        branch: Arc<dyn ExecutionPlan>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let row = partition_plan(DelayedBatch {
            batch: RecordBatch::try_new(
                int32_k_schema(),
                vec![Arc::new(Int32Array::from(vec![42]))],
            )?,
            delay: Duration::from_millis(200),
        })?;
        let union = UnionExec::try_new(vec![branch, row])?;
        Ok(Arc::new(
            CoalescePartitionsExec::new(union).with_fetch(Some(1)),
        ))
    }

    /// [`lower`], also of a limit at the root.
    fn lower_with_limit(
        plan: &Arc<dyn ExecutionPlan>,
    ) -> Result<Arc<dyn PhysicalOperator>> {
        use crate::operators::limit::PhysicalLimit;
        match PhysicalLimit::try_lower(plan, lower)? {
            Some(limit) => Ok(Arc::new(limit)),
            None => lower(plan),
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn probe_sides_wait_for_the_build_side_of_every_join_type() -> Result<()> {
        // A join, whose build side never produces anything, over a join whose
        // build keys `(1 / k, k)` divide by zero, under a limit that is
        // reached after 200 ms. DataFusion never reads the probe side of the
        // outer join, as its build side never completes, so it never collects
        // the build side of the inner join: neither engine fails. With a
        // single build key `1 / k`, which DataFusion evaluates on every build
        // batch as it collects them, the inner join is left to DataFusion.
        let make_plan = |join_type: JoinType, keys: usize| {
            let k = col("k", &int32_k_schema())?;
            let one_over_k = Arc::new(BinaryExpr::new(
                lit(1_i32),
                Operator::Divide,
                Arc::clone(&k),
            ));
            let build = int32_k_scan(vec![Some(0)])?;
            let probe = int32_k_scan(vec![Some(1)])?;
            let probe_k = col("k", &probe.schema())?;
            let mut on = vec![(one_over_k as _, Arc::clone(&probe_k))];
            if keys == 2 {
                on.push((k, probe_k));
            }
            let nested: Arc<dyn ExecutionPlan> = Arc::new(HashJoinExec::try_new(
                build,
                probe,
                on,
                None,
                &JoinType::RightSemi,
                None,
                PartitionMode::CollectLeft,
                NullEquality::NullEqualsNothing,
                false,
            )?);
            // Only the first column (`k` of either side), for the union with
            // the row 42.
            let build = pending_plan(int32_k_schema(), 1)?;
            let on = vec![(col("k", &build.schema())?, col("k", &nested.schema())?)];
            let join = HashJoinExec::try_new(
                build,
                nested,
                on,
                None,
                &join_type,
                Some(vec![0]),
                PartitionMode::CollectLeft,
                NullEquality::NullEqualsNothing,
                false,
            )?;
            limit_over_delayed_42(Arc::new(join))
        };
        for join_type in ALL_JOIN_TYPES {
            for keys in [1, 2] {
                let root = assert_lowered_plan_byte_eq(
                    || make_plan(join_type, keys),
                    lower_with_limit,
                    task_ctx(),
                    false,
                )
                .await;
                assert_eq!(count::<PhysicalHashJoin>(&root), keys, "{join_type}");
            }
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn build_key_errors_are_raised_when_datafusion_raises_them() -> Result<()> {
        // A `RightSemi` join on `(1 / k, k)` whose build side produces `k = 0`
        // (and then nothing), under a limit that is reached after 200 ms.
        // DataFusion evaluates these keys once its whole build side is
        // collected: never, so neither engine fails.
        let join = |keys: usize, build_ends: bool| -> Result<Arc<dyn ExecutionPlan>> {
            let zero = RecordBatch::try_new(
                int32_k_schema(),
                vec![Arc::new(Int32Array::from(vec![0]))],
            )?;
            let after = if build_ends {
                AfterBatch::End
            } else {
                AfterBatch::Idle
            };
            let build = batch_plan(zero, after)?;
            let probe = int32_k_scan(vec![Some(1)])?;
            let (k, probe_k) = (col("k", &build.schema())?, col("k", &probe.schema())?);
            let one_over_k = Arc::new(BinaryExpr::new(
                lit(1_i32),
                Operator::Divide,
                Arc::clone(&k),
            ));
            let mut on = vec![(one_over_k as _, Arc::clone(&probe_k))];
            if keys == 2 {
                on.push((k, probe_k));
            }
            Ok(Arc::new(HashJoinExec::try_new(
                build,
                probe,
                on,
                None,
                &JoinType::RightSemi,
                None,
                PartitionMode::CollectLeft,
                NullEquality::NullEqualsNothing,
                false,
            )?))
        };
        let root = assert_lowered_plan_byte_eq(
            || limit_over_delayed_42(join(2, false)?),
            lower_with_limit,
            task_ctx(),
            false,
        )
        .await;
        assert_eq!(count::<PhysicalHashJoin>(&root), 1);

        // Once the build side is complete, both engines fail.
        let root = assert_lowered_plan_fails_like_datafusion(
            || join(2, true),
            lower,
            test_session_config(),
            "Divide by zero",
        )
        .await;
        assert!(root.is::<PhysicalHashJoin>());

        // The bounds of a single key of an integer type (a candidate for a
        // perfect hash join) are computed on every build batch, as it is
        // collected: both engines fail right away (the join is left to
        // DataFusion, see `eagerly_evaluated_build_keys_that_can_fail`).
        let root = assert_lowered_plan_fails_like_datafusion(
            || limit_over_delayed_42(join(1, false)?),
            lower_with_limit,
            test_session_config(),
            "Divide by zero",
        )
        .await;
        assert_eq!(count::<PhysicalHashJoin>(&root), 0);
        Ok(())
    }

    /// A memory pool without limit that signals `signal` when a reservation
    /// of the consumer `consumer` grows for the first time.
    #[derive(Debug)]
    struct SignalingPool {
        consumer: &'static str,
        signal: Arc<tokio::sync::Notify>,
        signaled: AtomicBool,
        reserved: AtomicUsize,
    }

    impl fmt::Display for SignalingPool {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "SignalingPool")
        }
    }

    impl MemoryPool for SignalingPool {
        fn name(&self) -> &str {
            "SignalingPool"
        }

        fn grow(&self, reservation: &MemoryReservation, additional: usize) {
            if reservation.consumer().name() == self.consumer
                && !self.signaled.swap(true, Ordering::Relaxed)
            {
                self.signal.notify_one();
            }
            self.reserved.fetch_add(additional, Ordering::Relaxed);
        }

        fn shrink(&self, _reservation: &MemoryReservation, shrink: usize) {
            self.reserved.fetch_sub(shrink, Ordering::Relaxed);
        }

        fn try_grow(
            &self,
            reservation: &MemoryReservation,
            additional: usize,
        ) -> Result<()> {
            self.grow(reservation, additional);
            Ok(())
        }

        fn reserved(&self) -> usize {
            self.reserved.load(Ordering::Relaxed)
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn build_keys_of_joins_with_dynamic_filters_are_evaluated_eagerly() -> Result<()>
    {
        // A `RightSemi` join on `(1 / k, k)` whose build side produces `k = 0`
        // (and then nothing), over a Parquet scan of `k`, under a limit that
        // is reached after 200 ms. Without a dynamic filter, DataFusion
        // evaluates these keys once its whole build side is collected: never,
        // so neither engine fails. With the dynamic filter of the join, which
        // the scan reads, DataFusion computes the bounds of the keys as it
        // collects the build side, and divides by zero right away: the join is
        // left to DataFusion.
        let dir = tempfile::tempdir()?;
        let path = dir.path().join("k.parquet");
        let rows = RecordBatch::try_new(
            int32_k_schema(),
            vec![Arc::new(Int32Array::from(vec![1, 2]))],
        )?;
        write_parquet(&path, int32_k_schema(), &[rows], None)?;
        let ctx = datafusion_context(test_session_config());
        let path = path.to_str().expect("a UTF-8 path");
        ctx.register_parquet("probe", path, ParquetReadOptions::default())
            .await?;
        let scan = ctx.sql("SELECT k FROM probe").await?;
        let scan = scan.create_physical_plan().await?;
        assert_eq!(scan.schema(), int32_k_schema());
        let options = test_session_config().options().clone();
        let make_plan = |dynamic_filter: bool| -> Result<Arc<dyn ExecutionPlan>> {
            let zero = RecordBatch::try_new(
                int32_k_schema(),
                vec![Arc::new(Int32Array::from(vec![0]))],
            )?;
            let build = batch_plan(zero, AfterBatch::Idle)?;
            let k = col("k", &build.schema())?;
            let probe_k = col("k", &scan.schema())?;
            let one_over_k = Arc::new(BinaryExpr::new(
                lit(1_i32),
                Operator::Divide,
                Arc::clone(&k),
            ));
            let on = vec![(one_over_k as _, Arc::clone(&probe_k)), (k, probe_k)];
            let join: Arc<dyn ExecutionPlan> = Arc::new(HashJoinExec::try_new(
                build,
                Arc::clone(&scan),
                on,
                None,
                &JoinType::RightSemi,
                None,
                PartitionMode::CollectLeft,
                NullEquality::NullEqualsNothing,
                false,
            )?);
            let join = if dynamic_filter {
                FilterPushdown::new_post_optimization().optimize(join, &options)?
            } else {
                join
            };
            assert_eq!(
                join.dynamic_expressions_produced().len(),
                usize::from(dynamic_filter)
            );
            limit_over_delayed_42(join)
        };
        let root = assert_lowered_plan_byte_eq(
            || make_plan(false),
            lower_with_limit,
            task_ctx(),
            false,
        )
        .await;
        assert_eq!(count::<PhysicalHashJoin>(&root), 1);
        let root = assert_lowered_plan_fails_like_datafusion(
            || make_plan(true),
            lower_with_limit,
            test_session_config(),
            "Divide by zero",
        )
        .await;
        assert_eq!(count::<PhysicalHashJoin>(&root), 0);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn eagerly_evaluated_build_keys_that_can_fail_are_left_to_datafusion()
    -> Result<()> {
        // A `RightSemi` join on `1 / k`, whose build side is a single batch of
        // 500,000 rows of `k = 1` but row 300,000 (`k = 0`), under a limit of
        // one row over a union whose other branch produces 42 as soon as the
        // native join reserves memory for its first build rows, on one thread.
        // DataFusion evaluates the key on the whole batch as it collects it,
        // and divides by zero, while a native join, which evaluates it chunk
        // by chunk, would be canceled before it reaches row 300,000.
        let make_plan =
            |signal: Arc<tokio::sync::Notify>| -> Result<Arc<dyn ExecutionPlan>> {
                let k = Int32Array::from_iter_values(
                    (0..500_000).map(|i| i32::from(i != 300_000)),
                );
                let rows = RecordBatch::try_new(int32_k_schema(), vec![Arc::new(k)])?;
                let build: Arc<dyn ExecutionPlan> = MemorySourceConfig::try_new_exec(
                    &[vec![rows]],
                    int32_k_schema(),
                    None,
                )?;
                let probe = int32_k_scan(vec![Some(1)])?;
                let one_over_k = Arc::new(BinaryExpr::new(
                    lit(1_i32),
                    Operator::Divide,
                    col("k", &build.schema())?,
                ));
                let on = vec![(one_over_k as _, col("k", &probe.schema())?)];
                let join: Arc<dyn ExecutionPlan> = Arc::new(HashJoinExec::try_new(
                    build,
                    probe,
                    on,
                    None,
                    &JoinType::RightSemi,
                    None,
                    PartitionMode::CollectLeft,
                    NullEquality::NullEqualsNothing,
                    false,
                )?);
                let row = partition_plan(BatchWhenSignaled {
                    batch: RecordBatch::try_new(
                        int32_k_schema(),
                        vec![Arc::new(Int32Array::from(vec![42]))],
                    )?,
                    ready: signal,
                })?;
                let union = UnionExec::try_new(vec![join, row])?;
                Ok(Arc::new(
                    CoalescePartitionsExec::new(union).with_fetch(Some(1)),
                ))
            };
        let run = |native: bool| async move {
            let signal = Arc::new(tokio::sync::Notify::new());
            let pool = Arc::new(SignalingPool {
                consumer: "PhysicalHashJoin",
                signal: Arc::clone(&signal),
                signaled: AtomicBool::new(false),
                reserved: AtomicUsize::new(0),
            });
            let runtime = RuntimeEnvBuilder::new()
                .with_memory_pool(pool)
                .build_arc()?;
            let task_ctx = Arc::new(
                TaskContext::default()
                    .with_session_config(test_session_config().with_target_partitions(1))
                    .with_runtime(runtime),
            );
            let plan = make_plan(signal)?;
            if native {
                let root = lower_with_limit(&plan)?;
                assert_eq!(count::<PhysicalHashJoin>(&root), 0);
                let exec = Arc::new(PipelineRootExec::try_new(root, plan)?);
                with_timeout(collect(exec, task_ctx)).await
            } else {
                with_timeout(collect(plan, task_ctx)).await
            }
        };
        for native in [false, true] {
            let error = run(native).await.expect_err("the join fails");
            assert!(error.to_string().contains("Divide by zero"), "{error}");
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn residual_filters_that_can_fail_are_left_to_datafusion() -> Result<()> {
        // 5,000 build rows of key 7, all with `v = 1` but row 3,000 (`v = 0`),
        // a probe row of key 7, and the filter `1 / v > 0`, under a limit of
        // one row, on a single thread. DataFusion evaluates the filter on all
        // 5,000 candidate pairs (fewer than `batch_size`) before it emits any
        // row, and divides by zero, while a native join, which emits as it
        // goes, would stop after its first chunk.
        let make_plan = |fallible: bool| -> Result<Arc<dyn ExecutionPlan>> {
            let schema = Arc::new(Schema::new(vec![
                Field::new("k", DataType::Int32, true),
                Field::new("v", DataType::Int32, true),
            ]));
            let rows = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int32Array::from(vec![7; 5000])),
                    Arc::new(Int32Array::from_iter_values(
                        (0..5000).map(|i| i32::from(i != 3000)),
                    )),
                ],
            )?;
            let build: Arc<dyn ExecutionPlan> =
                MemorySourceConfig::try_new_exec(&[vec![rows]], schema, None)?;
            let probe = int32_k_scan(vec![Some(7)])?;
            let filter_schema =
                Arc::new(Schema::new(vec![Field::new("v", DataType::Int32, true)]));
            let v = col("v", &filter_schema)?;
            let checked = if fallible {
                Arc::new(BinaryExpr::new(lit(1_i32), Operator::Divide, v)) as _
            } else {
                v
            };
            let filter = JoinFilter::new(
                Arc::new(BinaryExpr::new(checked, Operator::Gt, lit(0_i32))),
                vec![ColumnIndex {
                    index: 1,
                    side: JoinSide::Left,
                }],
                filter_schema,
            );
            let on = vec![(col("k", &build.schema())?, col("k", &probe.schema())?)];
            let join = Arc::new(HashJoinExec::try_new(
                build,
                probe,
                on,
                Some(filter),
                &JoinType::Inner,
                None,
                PartitionMode::CollectLeft,
                NullEquality::NullEqualsNothing,
                false,
            )?);
            Ok(Arc::new(GlobalLimitExec::new(join, 0, Some(1))))
        };
        let config = test_session_config()
            .with_target_partitions(1)
            .with_batch_size(8192);
        let root = assert_lowered_plan_fails_like_datafusion(
            || make_plan(true),
            lower_with_limit,
            config.clone(),
            "Divide by zero",
        )
        .await;
        assert_eq!(count::<PhysicalHashJoin>(&root), 0);

        // A filter that cannot fail (`v > 0`): the join is native.
        let task_ctx = Arc::new(TaskContext::default().with_session_config(config));
        let root = assert_lowered_plan_byte_eq(
            || make_plan(false),
            lower_with_limit,
            task_ctx,
            false,
        )
        .await;
        assert_eq!(count::<PhysicalHashJoin>(&root), 1);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn casts_to_narrow_dictionaries_are_left_to_datafusion() -> Result<()> {
        // 5,000 build rows of key 7, whose strings `v` are the same 3,000
        // times, then all different, a probe row of key 7, and the filter
        // `(TRY_)CAST(v AS Dictionary(Int8, Utf8)) IS NOT NULL` under a limit
        // of one row, on one thread. DataFusion evaluates the filter on all
        // 5,000 candidate pairs at once: more distinct values than keys of 8
        // bits hold, also for casts that cast what does not fit to NULL. A
        // native join, which emits as it goes, would stop after its first
        // chunk (of a single value).
        let make_plan = |try_cast: bool| -> Result<Arc<dyn ExecutionPlan>> {
            let schema = Arc::new(Schema::new(vec![
                Field::new("k", DataType::Int32, true),
                Field::new("v", DataType::Utf8, true),
            ]));
            let rows = RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int32Array::from(vec![7; 5000])),
                    Arc::new(StringArray::from_iter_values((0..5000).map(|i| {
                        if i < 3000 {
                            "same".to_string()
                        } else {
                            format!("v{i}")
                        }
                    }))),
                ],
            )?;
            let build: Arc<dyn ExecutionPlan> =
                MemorySourceConfig::try_new_exec(&[vec![rows]], schema, None)?;
            let probe = int32_k_scan(vec![Some(7)])?;
            let filter_schema =
                Arc::new(Schema::new(vec![Field::new("v", DataType::Utf8, true)]));
            let v = col("v", &filter_schema)?;
            let dictionary =
                DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8));
            let cast: PhysicalExprRef = if try_cast {
                Arc::new(TryCastExpr::new(v, dictionary))
            } else {
                let options = arrow::compute::CastOptions {
                    safe: true,
                    ..Default::default()
                };
                Arc::new(CastExpr::new(v, dictionary, Some(options)))
            };
            let filter = JoinFilter::new(
                Arc::new(IsNotNullExpr::new(cast)),
                vec![ColumnIndex {
                    index: 1,
                    side: JoinSide::Left,
                }],
                filter_schema,
            );
            let on = vec![(col("k", &build.schema())?, col("k", &probe.schema())?)];
            let join = Arc::new(HashJoinExec::try_new(
                build,
                probe,
                on,
                Some(filter),
                &JoinType::Inner,
                None,
                PartitionMode::CollectLeft,
                NullEquality::NullEqualsNothing,
                false,
            )?);
            Ok(Arc::new(GlobalLimitExec::new(join, 0, Some(1))))
        };
        let config = test_session_config()
            .with_target_partitions(1)
            .with_batch_size(8192);
        for try_cast in [true, false] {
            let task_ctx =
                Arc::new(TaskContext::default().with_session_config(config.clone()));
            let error = collect(make_plan(try_cast)?, task_ctx).await.unwrap_err();
            let message = error.to_string();
            assert!(
                message.contains("Dictionary key bigger than the key type"),
                "{message}"
            );
            let root = assert_lowered_plan_fails_like_datafusion(
                || make_plan(try_cast),
                lower_with_limit,
                config.clone(),
                &message,
            )
            .await;
            assert_eq!(count::<PhysicalHashJoin>(&root), 0, "{try_cast}");
        }
        Ok(())
    }

    /// A probe key that is 1 (for all rows) when it is evaluated the first
    /// time, 2 the second time, and so on.
    #[derive(Debug)]
    struct Sequence {
        evaluations: Arc<std::sync::atomic::AtomicI32>,
    }

    impl fmt::Display for Sequence {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "sequence()")
        }
    }

    impl PartialEq for Sequence {
        fn eq(&self, other: &Self) -> bool {
            Arc::ptr_eq(&self.evaluations, &other.evaluations)
        }
    }

    impl Eq for Sequence {}

    impl std::hash::Hash for Sequence {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            Arc::as_ptr(&self.evaluations).hash(state);
        }
    }

    impl PhysicalExpr for Sequence {
        fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
            Ok(DataType::Int32)
        }

        fn nullable(&self, _input_schema: &Schema) -> Result<bool> {
            Ok(false)
        }

        fn evaluate(
            &self,
            batch: &RecordBatch,
        ) -> Result<datafusion_expr::ColumnarValue> {
            let value = self.evaluations.fetch_add(1, Ordering::Relaxed) + 1;
            Ok(datafusion_expr::ColumnarValue::Array(Arc::new(
                Int32Array::from(vec![value; batch.num_rows()]),
            )))
        }

        fn children(&self) -> Vec<&Arc<dyn PhysicalExpr>> {
            vec![]
        }

        fn with_new_children(
            self: Arc<Self>,
            _children: Vec<Arc<dyn PhysicalExpr>>,
        ) -> Result<Arc<dyn PhysicalExpr>> {
            Ok(self)
        }

        fn fmt_sql(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "sequence()")
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn lowering_does_not_evaluate_expressions() -> Result<()> {
        // The probe key of the probe row is the next value of a sequence,
        // which matches the build key 1 if it is evaluated once, by the
        // execution. Lowering must not evaluate it (the join, whose key is not
        // known to never fail, is left to DataFusion).
        // Every plan has a sequence of its own.
        let make_plan =
            || -> Result<(Arc<dyn ExecutionPlan>, Arc<std::sync::atomic::AtomicI32>)> {
                let evaluations = Arc::new(std::sync::atomic::AtomicI32::new(0));
                let build = int32_k_scan(vec![Some(1)])?;
                let probe = int32_k_scan(vec![Some(5)])?;
                let sequence = Arc::new(Sequence {
                    evaluations: Arc::clone(&evaluations),
                });
                let on = vec![(col("k", &build.schema())?, sequence as _)];
                let join = HashJoinExec::try_new(
                    build,
                    probe,
                    on,
                    None,
                    &JoinType::Inner,
                    None,
                    PartitionMode::CollectLeft,
                    NullEquality::NullEqualsNothing,
                    false,
                )?;
                Ok((Arc::new(join), evaluations))
            };
        let (plan, evaluations) = make_plan()?;
        let root = lower(&plan)?;
        assert_eq!(evaluations.load(Ordering::Relaxed), 0);
        assert!(root.is::<BridgeOperator>());
        let (plan, _) = make_plan()?;
        assert_eq!(count_rows(&collect(plan, task_ctx()).await?), 1);
        let root = assert_lowered_plan_byte_eq(
            || make_plan().map(|(plan, _)| plan),
            lower,
            task_ctx(),
            false,
        )
        .await;
        assert!(root.is::<BridgeOperator>());
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn build_keys_that_datafusion_evaluates_on_empty_batches() -> Result<()> {
        // The single build key `1 / 0` of type `Int32`, whose bounds
        // DataFusion computes on every build batch: on the empty batch the
        // build side produces (before it produces nothing else), so that
        // DataFusion fails right away, before the limit is reached. A native
        // join does not see empty chunks: the join is left to DataFusion.
        let make_plan = || {
            let empty = RecordBatch::new_empty(int32_k_schema());
            let join = join_on_one_over_zero(
                batch_plan(empty, AfterBatch::Idle)?,
                int32_k_scan(vec![Some(1)])?,
                JoinType::RightSemi,
                PartitionMode::CollectLeft,
            )?;
            limit_over_delayed_42(join)
        };
        let root = assert_lowered_plan_fails_like_datafusion(
            make_plan,
            lower_with_limit,
            test_session_config(),
            "Divide by zero",
        )
        .await;
        assert_eq!(count::<PhysicalHashJoin>(&root), 0);
        Ok(())
    }

    #[test]
    fn expressions_that_can_fail() -> Result<()> {
        let schema = Schema::new(vec![
            Field::new("i", DataType::Int32, true),
            Field::new("f", DataType::Float64, true),
            Field::new("s", DataType::Utf8, true),
        ]);
        let (i, f, s) = (col("i", &schema)?, col("f", &schema)?, col("s", &schema)?);
        let binary = |l: &PhysicalExprRef, op, r: &PhysicalExprRef| -> PhysicalExprRef {
            Arc::new(BinaryExpr::new(Arc::clone(l), op, Arc::clone(r)))
        };
        let cast = |e: &PhysicalExprRef, to: DataType| -> PhysicalExprRef {
            Arc::new(CastExpr::new(Arc::clone(e), to, None))
        };
        for (expr, fails) in [
            (Arc::clone(&i), false),
            (binary(&i, Operator::Lt, &lit(3_i32)), false),
            (binary(&i, Operator::Plus, &lit(1_i32)), false),
            (binary(&f, Operator::Divide, &f), false),
            (binary(&i, Operator::Modulo, &lit(7_i32)), false),
            (binary(&i, Operator::Divide, &lit(-1_i32)), true),
            (binary(&i, Operator::Divide, &i), true),
            (binary(&lit(1_i32), Operator::Divide, &lit(0_i32)), true),
            (
                Arc::new(
                    BinaryExpr::new(Arc::clone(&i), Operator::Plus, lit(1_i32))
                        .with_fail_on_overflow(true),
                ),
                true,
            ),
            (cast(&i, DataType::Int64), false),
            (cast(&i, DataType::Float64), false),
            (cast(&i, DataType::Utf8), false),
            (cast(&i, DataType::Int16), true),
            (cast(&s, DataType::Int32), true),
            (
                Arc::new(TryCastExpr::new(Arc::clone(&s), DataType::Int32)),
                false,
            ),
            (
                Arc::new(TryCastExpr::new(
                    Arc::clone(&s),
                    DataType::Dictionary(
                        Box::new(DataType::Int8),
                        Box::new(DataType::Utf8),
                    ),
                )),
                true,
            ),
            (
                Arc::new(CastExpr::new(
                    Arc::clone(&s),
                    DataType::Dictionary(
                        Box::new(DataType::Int16),
                        Box::new(DataType::Utf8),
                    ),
                    Some(arrow::compute::CastOptions {
                        safe: true,
                        ..Default::default()
                    }),
                )),
                true,
            ),
            (
                Arc::new(TryCastExpr::new(
                    Arc::clone(&s),
                    DataType::Dictionary(
                        Box::new(DataType::Int32),
                        Box::new(DataType::Utf8),
                    ),
                )),
                false,
            ),
            (
                Arc::new(Sequence {
                    evaluations: Arc::new(std::sync::atomic::AtomicI32::new(0)),
                }),
                true,
            ),
            (
                binary(
                    &binary(&i, Operator::Gt, &lit(0_i32)),
                    Operator::And,
                    &binary(&lit(1_i32), Operator::Divide, &i),
                ),
                true,
            ),
        ] {
            assert_eq!(can_fail(&expr, &schema), fails, "{expr}");
        }
        Ok(())
    }

    /// A build side of `k` (`Int32`) and `d` (`Dictionary(Int8, Utf8)`): two
    /// batches of 100 rows, of keys `0..200`, whose dictionaries hold the 100
    /// distinct values of their rows: together more than keys of 8 bits hold.
    fn narrow_dictionary_build() -> Result<Arc<dyn ExecutionPlan>> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int32, true),
            Field::new(
                "d",
                DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
                true,
            ),
        ]));
        let batch = |start: i32| -> Result<RecordBatch> {
            let values = (start..start + 100)
                .map(|v| format!("v{v}"))
                .collect::<Vec<_>>();
            let d = values
                .iter()
                .map(String::as_str)
                .collect::<arrow::array::DictionaryArray<arrow::datatypes::Int8Type>>();
            Ok(RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int32Array::from_iter_values(start..start + 100)),
                    Arc::new(d),
                ],
            )?)
        };
        Ok(MemorySourceConfig::try_new_exec(
            &[vec![batch(0)?, batch(100)?]],
            schema,
            None,
        )?)
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn build_dictionaries_with_narrow_keys_are_left_to_datafusion() -> Result<()> {
        // A `Partitioned` inner join of 4 partitions, of the 200 build rows
        // with the probe keys `0..200`: in DataFusion, every partition holds
        // fewer distinct values of `d` than keys of 8 bits, while the row
        // store would encode all 200 of a chunk again, as output column or
        // only for the filter `d <> 'x'`.
        let partitioned = |filter_only: bool| -> Result<Arc<dyn ExecutionPlan>> {
            let by_k = |input: Arc<dyn ExecutionPlan>| -> Result<Arc<dyn ExecutionPlan>> {
                let k = col("k", &input.schema())?;
                Ok(Arc::new(RepartitionExec::try_new(
                    input,
                    Partitioning::Hash(vec![k], 4),
                )?))
            };
            let build = by_k(narrow_dictionary_build()?)?;
            let probe = by_k(int32_k_scan((0..200).map(Some).collect())?)?;
            let on = vec![(col("k", &build.schema())?, col("k", &probe.schema())?)];
            let (filter, projection) = if filter_only {
                let schema = Arc::new(Schema::new(vec![build.schema().field(1).clone()]));
                let not_x = Arc::new(BinaryExpr::new(
                    col("d", &schema)?,
                    Operator::NotEq,
                    lit("x"),
                ));
                let column = ColumnIndex {
                    index: 1,
                    side: JoinSide::Left,
                };
                (
                    Some(JoinFilter::new(not_x, vec![column], schema)),
                    Some(vec![0, 2]),
                )
            } else {
                (None, None)
            };
            Ok(Arc::new(HashJoinExec::try_new(
                build,
                probe,
                on,
                filter,
                &JoinType::Inner,
                projection,
                PartitionMode::Partitioned,
                NullEquality::NullEqualsNothing,
                false,
            )?))
        };
        for filter_only in [false, true] {
            let root = if hashes_are_forced_to_collide() {
                // All rows are in one partition: its build side does not fit.
                assert_lowered_plan_fails_like_datafusion(
                    || partitioned(filter_only),
                    lower,
                    test_session_config(),
                    "Dictionary key bigger than the key type",
                )
                .await
            } else {
                let batches = collect(partitioned(filter_only)?, task_ctx()).await?;
                assert_eq!(count_rows(&batches), 200);
                assert_lowered_plan_byte_eq(
                    || partitioned(filter_only),
                    lower,
                    task_ctx(),
                    false,
                )
                .await
            };
            assert_eq!(count::<PhysicalHashJoin>(&root), 0);
        }

        // `CollectLeft` joins probing the key 0 only: DataFusion fails to
        // concatenate the build batches, whose dictionaries need more keys
        // than 8 bits hold together, while the row store would not; also if
        // the join does not need the dictionary at all (projected away, or a
        // `RightSemi` join).
        for (join_type, projection) in [
            (JoinType::Inner, None),
            (JoinType::Inner, Some(vec![0])),
            (JoinType::RightSemi, None),
        ] {
            let collect_left = || -> Result<Arc<dyn ExecutionPlan>> {
                let build = narrow_dictionary_build()?;
                let probe = int32_k_scan(vec![Some(0)])?;
                let on = vec![(col("k", &build.schema())?, col("k", &probe.schema())?)];
                Ok(Arc::new(HashJoinExec::try_new(
                    build,
                    probe,
                    on,
                    None,
                    &join_type,
                    projection.clone(),
                    PartitionMode::CollectLeft,
                    NullEquality::NullEqualsNothing,
                    false,
                )?))
            };
            let error = collect(collect_left()?, task_ctx()).await.unwrap_err();
            let message = error.to_string();
            assert!(
                message.contains("Dictionary key bigger than the key type"),
                "{message}"
            );
            let root = assert_lowered_plan_fails_like_datafusion(
                collect_left,
                lower,
                test_session_config(),
                &message,
            )
            .await;
            assert_eq!(count::<PhysicalHashJoin>(&root), 0, "{join_type}");
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn build_run_ends_of_16_bits_are_left_to_datafusion() -> Result<()> {
        // Two build batches of 20,000 rows with a column run-end encoded with
        // run ends of 16 bits: DataFusion fails to concatenate them (40,000
        // rows), while the row store would not.
        let make_plan = || -> Result<Arc<dyn ExecutionPlan>> {
            let batch = |start: i32| -> Result<RecordBatch> {
                let run_ends = arrow::array::Int16Array::from_iter_values(
                    (1..=200).map(|run| run * 100),
                );
                let values = StringArray::from_iter_values(
                    (0..200).map(|run| format!("r{}", start + run)),
                );
                let runs =
                    arrow::array::RunArray::<arrow::datatypes::Int16Type>::try_new(
                        &run_ends, &values,
                    )?;
                let schema = Arc::new(Schema::new(vec![
                    Field::new("k", DataType::Int32, true),
                    Field::new("r", runs.data_type().clone(), true),
                ]));
                Ok(RecordBatch::try_new(
                    schema,
                    vec![
                        Arc::new(Int32Array::from_iter_values(start..start + 20_000)),
                        Arc::new(runs),
                    ],
                )?)
            };
            let batches = vec![batch(0)?, batch(20_000)?];
            let schema = batches[0].schema();
            let build: Arc<dyn ExecutionPlan> =
                MemorySourceConfig::try_new_exec(&[batches], schema, None)?;
            join_on_k(build, int32_k_scan(vec![Some(0)])?, JoinType::Inner)
        };
        let root = assert_lowered_plan_fails_like_datafusion(
            make_plan,
            lower,
            test_session_config(),
            "Run end index bigger than the run-end type",
        )
        .await;
        assert_eq!(count::<PhysicalHashJoin>(&root), 0);
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn fallible_build_repartitionings_are_kept() -> Result<()> {
        // A `RightSemi` join on `(1 / k, k) = (k, k)` whose build side is
        // repartitioned by its keys, and produces `k = 0` and then nothing,
        // under a limit reached after 200 ms. The repartitioning evaluates
        // `1 / k` on the row as it arrives, so that DataFusion fails right
        // away: it is kept (bridged), although the join itself evaluates its
        // keys too (and defers their errors until its build side is
        // complete).
        for mode in [PartitionMode::Partitioned, PartitionMode::CollectLeft] {
            let make_plan = || {
                let zero = RecordBatch::try_new(
                    int32_k_schema(),
                    vec![Arc::new(Int32Array::from(vec![0]))],
                )?;
                let source = batch_plan(zero, AfterBatch::Idle)?;
                let k = col("k", &int32_k_schema())?;
                let one_over_k: PhysicalExprRef = Arc::new(BinaryExpr::new(
                    lit(1_i32),
                    Operator::Divide,
                    Arc::clone(&k),
                ));
                let build_keys = vec![one_over_k, k];
                let repartitioned: Arc<dyn ExecutionPlan> =
                    Arc::new(RepartitionExec::try_new(
                        source,
                        Partitioning::Hash(build_keys.clone(), 4),
                    )?);
                let probe = int32_k_scan(vec![Some(1)])?;
                let probe_k = col("k", &probe.schema())?;
                let (build, probe): (Arc<dyn ExecutionPlan>, Arc<dyn ExecutionPlan>) =
                    match mode {
                        PartitionMode::Partitioned => (
                            repartitioned,
                            Arc::new(RepartitionExec::try_new(
                                probe,
                                Partitioning::Hash(
                                    vec![Arc::clone(&probe_k), Arc::clone(&probe_k)],
                                    4,
                                ),
                            )?),
                        ),
                        _ => {
                            (Arc::new(CoalescePartitionsExec::new(repartitioned)), probe)
                        }
                    };
                let on = build_keys
                    .into_iter()
                    .zip([Arc::clone(&probe_k), probe_k])
                    .collect();
                let join = HashJoinExec::try_new(
                    build,
                    probe,
                    on,
                    None,
                    &JoinType::RightSemi,
                    None,
                    mode,
                    NullEquality::NullEqualsNothing,
                    false,
                )?;
                limit_over_delayed_42(Arc::new(join))
            };
            let root = assert_lowered_plan_fails_like_datafusion(
                make_plan,
                lower_with_limit,
                test_session_config(),
                "Divide by zero",
            )
            .await;
            assert_eq!(count::<PhysicalHashJoin>(&root), 1, "{mode:?}");
        }
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn joins_without_probe_partitions_are_left_to_datafusion() -> Result<()> {
        // A join without probe partitions has no output partitions:
        // DataFusion executes neither of its inputs, nor evaluates its build
        // key `1 / 0` on the build rows.
        let no_partitions = || -> Arc<dyn ExecutionPlan> {
            Arc::new(EmptyExec::new(int32_k_schema()).with_partitions(0))
        };
        for mode in [PartitionMode::CollectLeft, PartitionMode::Partitioned] {
            let make_plan = || {
                let build = match mode {
                    PartitionMode::CollectLeft => int32_k_scan(vec![Some(1), Some(2)])?,
                    _ => no_partitions(),
                };
                join_on_one_over_zero(build, no_partitions(), JoinType::Inner, mode)
            };
            let root =
                assert_lowered_plan_byte_eq(make_plan, lower, task_ctx(), false).await;
            assert!(root.is::<BridgeOperator>(), "{mode:?}");
        }
        Ok(())
    }

    /// The first `HashJoinExec` of `plan`, depth first.
    fn find_hash_join(plan: &Arc<dyn ExecutionPlan>) -> Option<Arc<dyn ExecutionPlan>> {
        if plan.is::<HashJoinExec>() {
            return Some(Arc::clone(plan));
        }
        plan.children().into_iter().find_map(find_hash_join)
    }

    /// The first scan of files of `plan`.
    fn find_parquet_scan(
        plan: &Arc<dyn ExecutionPlan>,
    ) -> Option<Arc<dyn ExecutionPlan>> {
        if plan
            .downcast_ref::<DataSourceExec>()
            .is_some_and(|scan| scan.data_source().is::<FileScanConfig>())
        {
            return Some(Arc::clone(plan));
        }
        plan.children().into_iter().find_map(find_parquet_scan)
    }

    /// The number of items pruned by the scan of files `scan` (native or not),
    /// by its pruning metric `name`.
    fn pruned(scan: &Arc<dyn ExecutionPlan>, name: &str) -> usize {
        use datafusion_physical_plan::metrics::MetricValue;
        match scan.metrics().and_then(|metrics| metrics.sum_by_name(name)) {
            Some(MetricValue::PruningMetrics {
                pruning_metrics, ..
            }) => pruning_metrics.pruned(),
            Some(other) => other.as_usize(),
            None => 0,
        }
    }

    fn count_rows(batches: &[RecordBatch]) -> usize {
        batches.iter().map(RecordBatch::num_rows).sum()
    }

    /// A memory pool without limit that records the peak memory reserved by
    /// every consumer (by name).
    #[derive(Debug, Default)]
    struct PeakPool {
        /// By consumer: the memory reserved now, and its peak.
        reserved: Mutex<std::collections::HashMap<String, (usize, usize)>>,
    }

    impl PeakPool {
        fn record(&self, reservation: &MemoryReservation, grown: usize, shrunk: usize) {
            let mut reserved = self.reserved.lock();
            let (now, peak) = reserved
                .entry(reservation.consumer().name().to_string())
                .or_default();
            *now = *now + grown - shrunk;
            *peak = (*peak).max(*now);
        }

        /// The sum of the peaks of all consumers: at least the peak of their
        /// total.
        fn sum_of_peaks(&self) -> usize {
            self.reserved.lock().values().map(|&(_, peak)| peak).sum()
        }

        fn peak(&self, consumer: &str) -> usize {
            self.reserved
                .lock()
                .get(consumer)
                .map_or(0, |&(_, peak)| peak)
        }
    }

    impl fmt::Display for PeakPool {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            write!(f, "PeakPool")
        }
    }

    impl MemoryPool for PeakPool {
        fn name(&self) -> &str {
            "PeakPool"
        }

        fn grow(&self, reservation: &MemoryReservation, additional: usize) {
            self.record(reservation, additional, 0);
        }

        fn shrink(&self, reservation: &MemoryReservation, shrink: usize) {
            self.record(reservation, 0, shrink);
        }

        fn try_grow(
            &self,
            reservation: &MemoryReservation,
            additional: usize,
        ) -> Result<()> {
            self.grow(reservation, additional);
            Ok(())
        }

        fn reserved(&self) -> usize {
            self.reserved.lock().values().map(|&(now, _)| now).sum()
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn hash_tables_that_do_not_fit_the_memory_pool_are_not_allocated() -> Result<()>
    {
        let build_batches = (0..20)
            .map(|i| data_batch("l", i * 1000..(i + 1) * 1000, 0))
            .collect::<Vec<_>>();
        // A join on the 23 values of `k`, which get a dense index, and on the
        // strings `s`, which get hashed tables.
        for (key, dense) in [("k", true), ("s", false)] {
            // A single thread, so that the memory used is deterministic.
            let run = |pool: Arc<dyn MemoryPool>| {
                let build_batches = build_batches.clone();
                async move {
                    let join = join_on_key(
                        source(data_schema("l"), build_batches) as _,
                        source(data_schema("r"), vec![data_batch("r", 0..100, 0)]) as _,
                        JoinType::Left,
                        key,
                    )?;
                    let runtime = RuntimeEnvBuilder::new()
                        .with_memory_pool(pool)
                        .build_arc()?;
                    let task_ctx = Arc::new(TaskContext::default().with_runtime(runtime));
                    with_timeout(Executor::collect(join, task_ctx, 1)).await
                }
            };
            let peaks = Arc::new(PeakPool::default());
            run(Arc::clone(&peaks) as _).await?;
            let (rows, index) =
                (peaks.peak("PhysicalHashJoin"), peaks.peak("JoinHashTable"));
            if dense {
                // The entries of the 23 keys, the chains and the matched rows.
                assert_eq!(index, (23 + 20_000) * 8 + 20_000_usize.div_ceil(64) * 8);
            } else {
                // Tables of at least twice the rows, chains and matched rows.
                assert!(index >= 20_000 * 3 * 8 + 20_000 / 8, "{index}");
            }
            assert!(rows > index, "{rows}");
            assert_eq!(peaks.reserved(), 0);
            // The rows fit, the hash table does not: it is not allocated.
            let err = run(Arc::new(GreedyMemoryPool::new(rows + index / 2)))
                .await
                .unwrap_err();
            assert!(err.to_string().contains("JoinHashTable"), "{key}: {err}");
            let err = run(Arc::new(GreedyMemoryPool::new(rows + index - 1)))
                .await
                .unwrap_err();
            assert!(err.to_string().contains("JoinHashTable"), "{key}: {err}");
            // Both fit.
            run(Arc::new(GreedyMemoryPool::new(rows + index))).await?;
        }
        Ok(())
    }

    #[test]
    fn lowering_of_hash_joins() -> Result<()> {
        let try_lower = |plan: Arc<dyn ExecutionPlan>| {
            PhysicalHashJoin::try_lower(&plan, |input| {
                Ok(Arc::new(BridgeOperator::new(Arc::clone(input))))
            })
        };
        // The redistribution below the join is removed.
        for mode in [PartitionMode::CollectLeft, PartitionMode::Partitioned] {
            let case = JoinCase {
                mode,
                filter: Some(("id", Operator::Lt, "id")),
                projection: Some(vec![0, 9]),
                ..JoinCase::new(JoinType::Full, vec!["k", "s"])
            };
            let plan = case.plan()?;
            let join = try_lower(Arc::clone(&plan))?.expect("a join");
            assert_eq!(join.schema(), plan.schema());
            assert_eq!(join.projection(), Some([0, 9].as_slice()));
            assert!(join.filter().is_some());
            for (input, side) in [(join.build_side(), "l"), (join.probe_side(), "r")] {
                let bridged = input.downcast_ref::<BridgeOperator>().unwrap();
                assert!(
                    bridged.plan().is::<DataSourceExec>(),
                    "{side}: {}",
                    bridged.plan().name()
                );
            }
            assert!(
                join.description()
                    .contains("filter=lid@0 < rid@1, projection=[lid@0, rf@9]"),
                "{}",
                join.description()
            );
        }

        // Joins left to DataFusion.
        let join = |case: JoinCase| -> Result<HashJoinExec> {
            Ok(case
                .plan()?
                .downcast_ref::<HashJoinExec>()
                .unwrap()
                .clone_for_test())
        };
        let base = || JoinCase::new(JoinType::Inner, vec!["k"]);
        let with_fetch = join(base())?.builder().with_fetch(Some(10)).build_exec()?;
        assert!(try_lower(with_fetch)?.is_none());
        let auto = join(base())?
            .builder()
            .with_partition_mode(PartitionMode::Auto)
            .build_exec()?;
        assert!(try_lower(auto)?.is_none());
        let correlated = JoinCase {
            null_aware: true,
            filter: Some(("id", Operator::Lt, "id")),
            ..JoinCase::new(JoinType::LeftAnti, vec!["k"])
        };
        assert!(try_lower(correlated.plan()?)?.is_none());
        let multi_column = join(JoinCase::new(JoinType::LeftAnti, vec!["k", "s"]))?
            .builder()
            .with_null_aware(true)
            .with_null_aware_value_keys(2)
            .build_exec()?;
        assert!(try_lower(multi_column)?.is_none());
        let null_equal = JoinCase {
            null_aware: true,
            null_equality: NullEquality::NullEqualsNull,
            ..JoinCase::new(JoinType::LeftAnti, vec!["k"])
        };
        assert!(try_lower(null_equal.plan()?)?.is_none());
        // Partitions DataFusion does not execute: a `CollectLeft` join of
        // several build partitions, a `Partitioned` join of different numbers
        // of partitions.
        let left = data_scan("l", 10, 0);
        let right = data_scan("r", 10, 0);
        let on = vec![(col("lk", &left.schema())?, col("rk", &right.schema())?)];
        for (mode, right) in [
            (PartitionMode::CollectLeft, Arc::clone(&right)),
            (
                PartitionMode::Partitioned,
                Arc::new(CoalescePartitionsExec::new(right)) as _,
            ),
        ] {
            let invalid = HashJoinExecBuilder::new(
                Arc::clone(&left),
                right,
                on.clone(),
                JoinType::Inner,
            )
            .with_partition_mode(mode)
            .build_exec()?;
            assert!(try_lower(invalid)?.is_none(), "{mode}");
        }
        // Keys of different types.
        let left = data_scan("l", 10, 0);
        let right = data_scan("r", 10, 0);
        let on = vec![(col("lk", &left.schema())?, col("rid", &right.schema())?)];
        let mixed = HashJoinExecBuilder::new(
            Arc::new(CoalescePartitionsExec::new(left)),
            right,
            on,
            JoinType::Inner,
        )
        .with_partition_mode(PartitionMode::CollectLeft)
        .build_exec()?;
        assert!(try_lower(mixed)?.is_none());
        // Other nodes.
        let mut called = false;
        let scan = data_scan("l", 10, 0);
        assert!(
            PhysicalHashJoin::try_lower(&scan, |_| {
                called = true;
                internal_err!("not called")
            })?
            .is_none()
        );
        assert!(!called);

        // The lowered inputs must have the schemas of the inputs.
        let err = PhysicalHashJoin::try_lower(&base().plan()?, |_| {
            Ok(source(data_schema("x"), vec![]) as _)
        })
        .unwrap_err();
        assert!(err.to_string().contains("has schema"), "{err}");
        Ok(())
    }

    /// `HashJoinExec` is not `Clone`; tests rebuild it from its builder.
    trait CloneForTest {
        fn clone_for_test(&self) -> HashJoinExec;
    }

    impl CloneForTest for HashJoinExec {
        fn clone_for_test(&self) -> HashJoinExec {
            self.builder().build().unwrap()
        }
    }

    #[test]
    fn take_rows_slices_contiguous_rows() -> Result<()> {
        let array: ArrayRef = Arc::new(Int64Array::from_iter_values(0..10));
        let rows = |indices: Vec<u32>| take_rows(&array, &UInt32Array::from(indices));
        let values = |array: ArrayRef| {
            array
                .as_primitive::<Int64Type>()
                .values()
                .iter()
                .copied()
                .collect::<Vec<_>>()
        };
        let all = rows((0..10).collect())?;
        assert!(Arc::ptr_eq(&all, &array));
        assert_eq!(values(rows(vec![3, 4, 5])?), vec![3, 4, 5]);
        assert_eq!(values(rows(vec![5, 3, 3])?), vec![5, 3, 3]);
        assert_eq!(values(rows(vec![9, 0])?), vec![9, 0]);
        assert_eq!(values(rows(vec![])?), Vec::<i64>::new());
        let with_null = take_rows(&array, &UInt32Array::from(vec![Some(1), None]))?;
        assert_eq!(with_null.null_count(), 1);
        Ok(())
    }

    #[test]
    fn join_specs_check_their_arguments() {
        let build = data_schema("l");
        let probe = data_schema("r");
        let on =
            |l: &str, r: &str| vec![(col(l, &build).unwrap(), col(r, &probe).unwrap())];
        let spec = |on, join_type, null_aware| {
            JoinSpec::try_new(
                &build,
                &probe,
                on,
                None,
                join_type,
                None,
                NullEquality::NullEqualsNothing,
                null_aware,
            )
        };
        assert!(spec(vec![], JoinType::Inner, false).is_err());
        assert!(spec(on("lk", "rid"), JoinType::Inner, false).is_err());
        assert!(spec(on("lk", "rk"), JoinType::Inner, true).is_err());
        let spec = spec(on("lk", "rk"), JoinType::LeftAnti, true).unwrap();
        // The key column is stored once, and only the output columns are
        // stored too.
        assert_eq!(spec.layout.num_columns(), build.fields().len());
        assert_eq!(spec.build_column(1).unwrap(), 0);
        assert_eq!(spec.payload, vec![0, 2, 3, 4, 5, 6]);
        let inner = JoinSpec::try_new(
            &build,
            &probe,
            on("lk", "rk"),
            Some(join_filter("v", Operator::Eq, "v")),
            JoinType::RightSemi,
            None,
            NullEquality::NullEqualsNull,
            false,
        )
        .unwrap();
        assert_eq!(inner.payload, vec![4]);
        assert!(inner.build_column(0).is_err());
        // Filters must read columns that the inputs have.
        let valid = join_filter("v", Operator::Eq, "v");
        for (index, side) in [
            (7, JoinSide::Left),
            (7, JoinSide::Right),
            (0, JoinSide::None),
        ] {
            let mut columns = valid.column_indices().to_vec();
            columns[usize::from(side != JoinSide::Left)] = ColumnIndex { index, side };
            let filter = JoinFilter::new(
                Arc::clone(valid.expression()),
                columns,
                Arc::clone(valid.schema()),
            );
            let err = JoinSpec::try_new(
                &build,
                &probe,
                on("lk", "rk"),
                Some(filter),
                JoinType::Inner,
                None,
                NullEquality::NullEqualsNothing,
                false,
            )
            .unwrap_err();
            assert!(matches!(err, DataFusionError::Plan(_)), "{err}");
        }
        let missing_column = JoinFilter::new(
            Arc::clone(valid.expression()),
            valid.column_indices()[..1].to_vec(),
            Arc::clone(valid.schema()),
        );
        assert!(
            JoinSpec::try_new(
                &build,
                &probe,
                on("lk", "rk"),
                Some(missing_column),
                JoinType::Inner,
                None,
                NullEquality::NullEqualsNothing,
                false,
            )
            .is_err()
        );
        assert!(!inner.emits_final_build_rows() && !inner.emits_unmatched_build_rows());
    }
}
