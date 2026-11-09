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
//!   the shared, read-only [`JoinHashTable`]. Like DataFusion, it evaluates
//!   the build keys on an empty build side. A canceled join (whose output is
//!   not needed any more) does neither, nor builds its table. If the join
//!   cannot produce any output (e.g. an inner join with an empty build
//!   side), `finalize` returns [`SinkFinalize::NoOutputPossible`], so that
//!   the probe side is not executed, as DataFusion does not read it either.
//!   DataFusion still creates the streams of its probe side, so a join
//!   lowered from a `HashJoinExec` creates the DataFusion streams of that
//!   probe input too, without polling them, to raise the same errors (e.g.
//!   of a scan of an object store that is not registered).
//! * **Probe** (the join as operator, pipeline 0): all threads of the probe
//!   pipeline pull chunks directly from the probe side and probe the shared
//!   table concurrently: neither side is shuffled
//!   ([`PhysicalHashJoin::try_lower`] removes the `RepartitionExec`s and
//!   `CoalescePartitionsExec`s that DataFusion puts below the join). The
//!   output keeps the order of the probe side; a probe chunk whose matches
//!   do not fit in a chunk of [`DEFAULT_VECTOR_SIZE`] rows is continued with
//!   [`OperatorResult::HaveMoreOutput`].
//! * **Final build rows** (the join as source, pipeline 3): the `Left`,
//!   `Full`, `LeftSemi`, `LeftAnti` and `LeftMark` joins track the build
//!   rows that found a match in an atomic bitset, and once the probe
//!   pipeline completed, all threads scan the build rows in parallel to emit
//!   the unmatched ones (the matched ones for `LeftSemi`, all of them with
//!   their mark for `LeftMark`).
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

use std::fmt::Write;
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
use datafusion_common::{
    DataFusionError, JoinSide, JoinType, NullEquality, Result, internal_datafusion_err,
    internal_err, not_impl_err, plan_err,
};
use datafusion_datasource::file_scan_config::FileScanConfig;
use datafusion_datasource::memory::MemorySourceConfig;
use datafusion_datasource::source::DataSourceExec;
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_expr::Operator;
use datafusion_physical_expr::expressions::{
    BinaryExpr, CaseExpr, CastExpr, Column, InListExpr, IsNotNullExpr, IsNullExpr,
    Literal, NotExpr, TryCastExpr,
};
use datafusion_physical_expr::{PhysicalExpr, PhysicalExprRef};
use datafusion_physical_expr_common::utils::evaluate_expressions_to_arrays;
use datafusion_physical_plan::coalesce_partitions::CoalescePartitionsExec;
use datafusion_physical_plan::empty::EmptyExec;
use datafusion_physical_plan::execution_plan::reset_plan_states;
use datafusion_physical_plan::joins::utils::{
    ColumnIndex, JoinFilter, build_join_schema,
};
use datafusion_physical_plan::joins::{HashJoinExec, PartitionMode};
use datafusion_physical_plan::memory::LazyMemoryExec;
use datafusion_physical_plan::placeholder_row::PlaceholderRowExec;
use datafusion_physical_plan::repartition::RepartitionExec;
use datafusion_physical_plan::{
    ExecutionPlan, ExecutionPlanProperties, Partitioning, displayable,
};
use parking_lot::Mutex;

use crate::bridge::BridgeOperator;
use crate::chunk::{DEFAULT_VECTOR_SIZE, DataChunk, SelectionVector};
use crate::join::hash_table::{
    ChainScan, JoinHashTable, JoinHashTableBuilder, hash_join_keys,
    radix_bits_for_threads,
};
use crate::operator::{
    ExecutionContext, GlobalOperatorState, GlobalSinkState, GlobalSourceState,
    LocalSinkState, LocalSourceState, OperatorResult, OperatorState, PhysicalOperator,
    SinkFinalize, SinkResult, SourceResult, TaskInterrupt, downcast_state,
    downcast_state_mut,
};
use crate::pipeline::{MetaPipelineBuilder, PipelineBuilder};
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
    /// for a dynamic filter (which the joins lowered to native joins do not
    /// have), or for a single key of an integer type, a candidate for its
    /// perfect hash join (`ArrayMap`).
    defers_build_key_errors: bool,
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
        Ok(Self {
            join_type,
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
    ///   `ExecutionPlan::dynamic_expressions_produced`), which DataFusion
    ///   attaches to a join when a node of its probe side, such as a Parquet
    ///   scan, reads the filter. Once the build side is collected, DataFusion
    ///   sets the filter to a predicate of the build keys (bounds, an `IN`
    ///   list or a hash table lookup, depending on the size of the build
    ///   side), with which that node skips probe rows (or whole row groups)
    ///   that cannot match, so the nodes between it and the join never
    ///   evaluate their expressions on those rows. A [`PhysicalHashJoin`]
    ///   leaves the filter `true`, so they would see all rows (and could
    ///   raise errors on them, like a division by zero). Joins without one
    ///   are lowered, e.g. with
    ///   `datafusion.optimizer.enable_join_dynamic_filter_pushdown` off;
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
    /// * a join whose single build key of an integer type fails without any
    ///   row (e.g. `1 / 0`): DataFusion evaluates such a key on every build
    ///   batch as it collects them (to compute its bounds), also on an empty
    ///   one, which a native join does not see;
    /// * a join whose probe keys fail without any row (e.g. `1 / 0`):
    ///   DataFusion evaluates them on every batch of its probe side, also on
    ///   an empty one, which a native join, called with the non-empty chunks
    ///   of its probe side, cannot reproduce;
    /// * a null-aware join with several keys or a filter (a correlated or
    ///   multi-column `NOT IN`), or with `NullEqualsNull`;
    /// * a join that stores build columns of dictionaries with keys of 8 or
    ///   16 bits: the row store encodes the dictionary columns it gathers
    ///   again, with the distinct values of the gathered rows (which more
    ///   than 128 or 256 of them in a chunk overflow), while DataFusion takes
    ///   them from its whole (concatenated) build side, whose concatenation
    ///   overflows the keys if they cannot hold all its distinct values;
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
    /// The probe side of a join is left to DataFusion (bridged) if its
    /// lowered form has pipeline breakers, e.g. a join or an aggregation:
    /// their pipelines run independently of the build side of this join,
    /// while DataFusion reads the probe side only once the build side is
    /// collected (not if it never completes, e.g. before a limit above
    /// cancels the join, nor if the join cannot produce output), so that they
    /// would compute (and raise errors) where DataFusion does not. A bridged
    /// probe side is executed by DataFusion when the native join reads it.
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
            || !plan.dynamic_expressions_produced().is_empty()
            || !probe_streams_can_be_replayed(join.right())
        {
            return Ok(None);
        }
        let spec = match JoinSpec::try_new(
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
        // DataFusion evaluates the probe keys on every probe batch, also on an
        // empty one, so that keys that fail without any row (e.g. `1 / 0`)
        // fail the join as soon as its probe side produces a batch. The native
        // join evaluates them on the (non-empty) chunks of its probe side.
        let empty_probe = RecordBatch::new_empty(join.right().schema());
        if evaluate_expressions_to_arrays(probe_keys.iter().copied(), &empty_probe)
            .is_err()
        {
            return Ok(None);
        }
        // DataFusion evaluates the residual filter on up to `batch_size`
        // candidate pairs of a probe batch at once, and the probe keys on
        // whole probe batches, before it emits any of their rows. A native
        // join evaluates them chunk by chunk and emits as it goes, so that,
        // e.g. under a limit, it could stop before it evaluates them on rows
        // on which DataFusion fails: a filter or probe keys that can fail on
        // some rows are left to DataFusion.
        let probe_schema = join.right().schema();
        if join
            .filter()
            .is_some_and(|filter| can_fail(filter.expression(), filter.schema()))
            || probe_keys.iter().any(|key| can_fail(key, &probe_schema))
        {
            return Ok(None);
        }
        // DataFusion evaluates the build keys that it does not defer (see
        // `JoinSpec::defers_build_key_errors`) on every build batch, also on
        // an empty one, which the native join, called with the non-empty
        // chunks of its build side, does not see: keys that fail without any
        // row are left to DataFusion.
        // The row store encodes the dictionary columns it gathers (for the
        // output or the filter) again, with the distinct values of the
        // gathered rows, so that more distinct values than keys of 8 bits
        // hold in a chunk fail, where DataFusion takes them from its whole
        // build side; and DataFusion fails to concatenate a build side with
        // more distinct values than keys of 8 or 16 bits hold, where the row
        // store does not. Joins that store such dictionaries of the build
        // side are left to DataFusion.
        if spec
            .layout
            .schema()
            .fields()
            .iter()
            .any(|field| has_narrow_dictionary_keys(field.data_type()))
        {
            return Ok(None);
        }
        let empty_build = RecordBatch::new_empty(join.left().schema());
        if !spec.defers_build_key_errors
            && evaluate_expressions_to_arrays(build_keys.iter().copied(), &empty_build)
                .is_err()
        {
            return Ok(None);
        }
        let mut children = Vec::with_capacity(2);
        for (input, keys) in [(join.left(), build_keys), (join.right(), probe_keys)] {
            let input = without_redistribution(input, &keys);
            let mut lowered = lower_input(input)?;
            if lowered.schema() != input.schema() {
                return internal_err!(
                    "The lowered input {} of a HashJoinExec has schema {:?}, expected {:?}",
                    lowered.name(),
                    lowered.schema(),
                    input.schema()
                );
            }
            // DataFusion reads the probe side only once the build side is
            // collected: not if the build side never completes (e.g. before a
            // limit above cancels the join), nor if the join cannot produce
            // output. The pipelines of the pipeline breakers of a native probe
            // side (e.g. the build side of a join below) do not wait for the
            // build side, so they would compute (and raise errors) where
            // DataFusion does not: such a probe side is left to DataFusion,
            // which executes it only once the native join reads it.
            if children.len() == 1 && has_pipeline_breakers(&lowered) {
                lowered = Arc::new(BridgeOperator::new(Arc::clone(input)));
            }
            children.push(lowered);
        }
        Ok(Some(Self {
            children,
            spec,
            probe_plan: Some(Arc::clone(join.right())),
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
        let unmatched_cost = usize::from(spec.emits_unmatched_probe_rows());
        let scan = chain_scan.next_batch(
            table,
            DEFAULT_VECTOR_SIZE,
            unmatched_cost,
            probe_rows,
            matches,
        );
        if let Some(filter) = &spec.filter {
            self.apply_filter(filter, table, &chunk.batch, probe_rows, matches)?;
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
                let probe = UInt32Array::from(probe_rows.clone());
                *output = self.output_chunk(
                    table,
                    probe.len(),
                    Some((matches.as_slice(), None)),
                    Some((&chunk.batch, &probe)),
                    None,
                )?;
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
                let probe = UInt32Array::from(output_probe);
                *output = self.output_chunk(
                    table,
                    probe.len(),
                    Some((matches.as_slice(), build_take.as_ref())),
                    Some((&chunk.batch, &probe)),
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
                        JoinType::RightMark => {
                            rows.push(row as u32);
                            marks.push(matched);
                        }
                        _ => {}
                    }
                }
                let mark = (spec.join_type == JoinType::RightMark)
                    .then(|| Arc::new(BooleanArray::from(marks)) as ArrayRef);
                let probe = UInt32Array::from(rows);
                *output = self.output_chunk(
                    table,
                    probe.len(),
                    None,
                    Some((&chunk.batch, &probe)),
                    mark.as_ref(),
                )?;
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
        batch: &RecordBatch,
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
        let probe_indices = UInt32Array::from(probe_rows.clone());
        let columns = filter
            .column_indices()
            .iter()
            .map(|column| match column.side {
                JoinSide::Left => build_arrays.next().ok_or_else(|| {
                    internal_datafusion_err!("Missing build column of the join filter")
                }),
                JoinSide::Right => take_rows(batch.column(column.index), &probe_indices),
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
    /// * the probe columns are taken from the rows `probe.1` of `probe.0`;
    ///   NULL if `probe` is `None`;
    /// * the mark column is `mark`.
    fn output_chunk(
        &self,
        table: &JoinHashTable,
        num_rows: usize,
        build: Option<(&[RowLocation], Option<&UInt32Array>)>,
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
                    Some((batch, indices)) => {
                        take_rows(batch.column(column.index), indices)
                    }
                    None => Ok(new_null_array(field.data_type(), num_rows)),
                },
                JoinSide::None => mark.map(Arc::clone).ok_or_else(|| {
                    internal_datafusion_err!("Missing mark column of the join")
                }),
            })
            .collect::<Result<Vec<_>>>()?;
        DataChunk::try_new(Arc::clone(&spec.schema), columns, None)
    }

    /// Start probing the rows of `input`, resetting `scan` to the scan of
    /// their matches. Returns `None` if the rows cannot produce any output
    /// and do not need to be looked up.
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
        // Evaluate the keys on the logical rows only: a key of a row that a
        // filter removed must not raise an error.
        let batch = input.compact()?;
        let num_rows = batch.num_rows();
        let keys = evaluate_expressions_to_arrays(
            spec.on.iter().map(|(_, probe)| probe),
            &batch,
        )?;
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
        // support.
        hash_join_keys(&keys, num_rows, hashes)?;
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
            batch,
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

/// Whether evaluating `expr` on a batch of `schema` can fail on some rows.
/// Conservatively, only expressions built of columns, literals, comparisons,
/// boolean operators, `IS [NOT] NULL`, `CASE`, `IN` lists, `TRY_CAST`,
/// arithmetic that wraps on overflow, divisions by constants other than 0 and
/// -1 and casts that cannot fail cannot fail.
fn can_fail(expr: &PhysicalExprRef, schema: &Schema) -> bool {
    let infallible = expr.is::<Column>()
        || expr.is::<Literal>()
        || expr.is::<NotExpr>()
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
            .is_some_and(|cast| cast_cannot_fail(cast, schema));
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

/// Whether `operator`, or an operator below it, is a pipeline breaker (a
/// sink, e.g. a join or an aggregation), whose input runs in pipelines of its
/// own. A bridged plan is a source: DataFusion executes it when it is read.
fn has_pipeline_breakers(operator: &Arc<dyn PhysicalOperator>) -> bool {
    operator.is_sink() || operator.children().iter().any(has_pipeline_breakers)
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

/// Whether `data_type` is or has a dictionary with keys of 8 or 16 bits.
fn has_narrow_dictionary_keys(data_type: &DataType) -> bool {
    match data_type {
        DataType::Dictionary(keys, values) => {
            matches!(
                keys.as_ref(),
                DataType::Int8 | DataType::UInt8 | DataType::Int16 | DataType::UInt16
            ) || has_narrow_dictionary_keys(values)
        }
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::ListView(field)
        | DataType::LargeListView(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => has_narrow_dictionary_keys(field.data_type()),
        DataType::Struct(fields) => fields
            .iter()
            .any(|field| has_narrow_dictionary_keys(field.data_type())),
        DataType::Union(fields, _) => fields
            .iter()
            .any(|(_, field)| has_narrow_dictionary_keys(field.data_type())),
        DataType::RunEndEncoded(_, values) => {
            has_narrow_dictionary_keys(values.data_type())
        }
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
    /// The memory of the build rows.
    reservation: MemoryReservation,
}

impl GlobalSinkState for JoinState {}

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
                let builder = Arc::new(JoinHashTableBuilder::try_new(
                    data,
                    spec.num_keys(),
                    spec.null_equality,
                    index_reservation,
                )?);
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

/// A probe chunk being joined.
struct ProbeChunk {
    /// The (logical) rows of the chunk.
    batch: RecordBatch,
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
        Ok(Box::new(BuildLocalState::default()))
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
        let spec = &self.spec;
        let state = downcast_state::<JoinState>(gstate.as_any(), self.name())?;
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
        // Like DataFusion, which then does not read its probe side either.
        if (build_empty && spec.join_type.empty_build_side_produces_empty_result())
            || (table.is_empty() && spec.join_type.empty_map_produces_empty_result())
        {
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
        {
            // The build side sinks into the join, in a child meta pipeline,
            // whose last pipeline builds the pointer tables in parallel.
            let (mut build, mut build_pipeline) =
                meta.create_child_meta_pipeline(current, Arc::clone(&this))?;
            let finalize_pipeline =
                build.create_union_pipeline(&build_pipeline, false)?;
            build.set_source(
                &finalize_pipeline,
                Arc::new(JoinHashTableBuild::new(Arc::clone(&this))),
            )?;
            // The probe streams of a join lowered from a `HashJoinExec` are
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
            for pipeline in build.pipelines() {
                if pipeline != finalize_pipeline {
                    build.add_dependency(&finalize_pipeline, &pipeline)?;
                }
            }
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

/// The source of the pipeline that builds the pointer tables of a
/// [`PhysicalHashJoin`] in parallel, once all its build rows are combined:
/// every call of `get_data` builds the table of one radix partition (and
/// yields, as building a large partition takes a while), until all
/// partitions are claimed. It produces no rows; the join, the sink of the
/// pipeline, assembles the [`JoinHashTable`] when it is finalized.
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
        self.builder.num_partitions_with_rows().max(1)
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
/// [`PhysicalHashJoin`] lowered from a `HashJoinExec`, which depends on
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
    use datafusion_execution::object_store::ObjectStoreUrl;
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

    use crate::bridge::{BridgeOperator, PipelineRootExec};
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
    async fn differential_multiple_joins_and_partitionings() {
        for (sql, lowered) in [
            // A join below the probe side of a join (inner, `Right` as
            // DataFusion plans these `LEFT JOIN`s, or `Full`): the probe side
            // is left to DataFusion, which reads it only once the build side
            // is collected (see `PhysicalHashJoin::try_lower`).
            (
                "SELECT t1.id, x.name, y.w FROM t1 JOIN t2 x ON t1.a = x.k JOIN t2 y ON t1.a = y.k AND x.w = y.w",
                1,
            ),
            (
                "SELECT t1.id, x.name FROM t1 LEFT JOIN t2 x ON t1.a = x.k LEFT JOIN t2 y ON x.w = y.w",
                1,
            ),
            (
                "SELECT t1.id, x.name, y.w FROM t1 JOIN t2 x ON t1.a = x.k FULL JOIN t2 y ON x.w = y.w",
                1,
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
                crate::bridge::bridged_plans(&root)
                    .iter()
                    .all(|plan| !plan.is::<RepartitionExec>()),
                "{}",
                format_operator_tree(&root)
            );
        }
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
                    if has_narrow_dictionary_keys(field.data_type()) =>
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
                .any(|field| has_narrow_dictionary_keys(field.data_type()))
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
        let on = vec![(col("lk", &build.schema())?, col("rk", &probe.schema())?)];
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
    async fn joins_that_produce_dynamic_filters_are_left_to_datafusion() -> Result<()> {
        // The probe side `(k, d)` of `(1, 1), (2, 0)` is a Parquet file, whose
        // scan reads the dynamic filter of the join by `k`. Once DataFusion has
        // collected the build key 1, the scan skips the row of key 2 (with
        // `pushdown_filters`) before `1 / d` divides by its zero.
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
    async fn probe_sides_with_pipeline_breakers_are_left_to_datafusion() -> Result<()> {
        // An inner join whose build side ends empty after a while, over a join
        // whose non-empty build side has the key `1 / 0`. DataFusion reads
        // the probe side of the outer join only once its build side is
        // collected: never, as it is empty, so it never collects the build
        // side of the inner join. The pipelines of that build side would not
        // wait for the outer build side: the probe side is left to DataFusion.
        let make_plan = |join_type: JoinType| -> Result<Arc<dyn ExecutionPlan>> {
            let nested = join_on_one_over_zero(
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
        let root = assert_lowered_plan_byte_eq(
            || make_plan(JoinType::Inner),
            lower,
            task_ctx(),
            false,
        )
        .await;
        let join = root
            .downcast_ref::<PhysicalHashJoin>()
            .expect("a native join");
        let probe = join.probe_side().downcast_ref::<BridgeOperator>().unwrap();
        assert!(probe.plan().is::<HashJoinExec>());

        // A `Right` join reads its probe side also if its build side is
        // empty, once it is collected: both engines divide by zero then.
        let root = assert_lowered_plan_fails_like_datafusion(
            || make_plan(JoinType::Right),
            lower,
            test_session_config(),
            "Divide by zero",
        )
        .await;
        let join = root.downcast_ref::<PhysicalHashJoin>().unwrap();
        assert!(join.probe_side().is::<BridgeOperator>());
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
        // A `RightAnti` join, whose build side never produces anything, over
        // a join whose build key `1 / k` divides by zero, under a limit that
        // is reached after 200 ms. DataFusion never reads the probe side of
        // the outer join, as its build side never completes, so it never
        // collects the build side of the inner join: neither engine fails.
        let make_plan = || {
            let one_over_k = Arc::new(BinaryExpr::new(
                lit(1_i32),
                Operator::Divide,
                col("k", &int32_k_schema())?,
            ));
            let build = int32_k_scan(vec![Some(0)])?;
            let probe = int32_k_scan(vec![Some(1)])?;
            let on = vec![(one_over_k as _, col("k", &probe.schema())?)];
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
            let join = join_on_k(
                pending_plan(int32_k_schema(), 1)?,
                nested,
                JoinType::RightAnti,
            )?;
            limit_over_delayed_42(join)
        };
        let root =
            assert_lowered_plan_byte_eq(make_plan, lower_with_limit, task_ctx(), false)
                .await;
        assert_eq!(count::<PhysicalHashJoin>(&root), 1);
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
        // collected: both engines fail right away.
        let root = assert_lowered_plan_fails_like_datafusion(
            || limit_over_delayed_42(join(1, false)?),
            lower_with_limit,
            test_session_config(),
            "Divide by zero",
        )
        .await;
        assert_eq!(count::<PhysicalHashJoin>(&root), 1);
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
            Ok(Arc::new(
                datafusion_physical_plan::limit::GlobalLimitExec::new(join, 0, Some(1)),
            ))
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

        // A `CollectLeft` join probing the key 0 only: DataFusion fails to
        // concatenate the build batches, whose dictionaries need more keys
        // than 8 bits hold together, while the row store would not.
        let collect_left = || {
            join_on_k(
                narrow_dictionary_build()?,
                int32_k_scan(vec![Some(0)])?,
                JoinType::Inner,
            )
        };
        let error = collect(collect_left()?, task_ctx()).await.unwrap_err();
        let message = error.to_string();
        let root = assert_lowered_plan_fails_like_datafusion(
            collect_left,
            lower,
            test_session_config(),
            &message,
        )
        .await;
        assert_eq!(count::<PhysicalHashJoin>(&root), 0, "{message}");
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
        // A single thread, so that the memory used is deterministic.
        let run = |pool: Arc<dyn MemoryPool>| {
            let build_batches = build_batches.clone();
            async move {
                let join = join_of(
                    source(data_schema("l"), build_batches) as _,
                    source(data_schema("r"), vec![data_batch("r", 0..100, 0)]) as _,
                    JoinType::Left,
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
        let (rows, index) = (peaks.peak("PhysicalHashJoin"), peaks.peak("JoinHashTable"));
        // Tables of at least twice the rows, chains and matched rows.
        assert!(index >= 20_000 * 3 * 8 + 20_000 / 8, "{index}");
        assert!(rows > index, "{rows}");
        assert_eq!(peaks.reserved(), 0);
        // The rows fit, the hash table does not: it is not allocated.
        let err = run(Arc::new(GreedyMemoryPool::new(rows + index / 2)))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("JoinHashTable"), "{err}");
        let err = run(Arc::new(GreedyMemoryPool::new(rows + index - 1)))
            .await
            .unwrap_err();
        assert!(err.to_string().contains("JoinHashTable"), "{err}");
        // Both fit.
        run(Arc::new(GreedyMemoryPool::new(rows + index))).await?;
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
