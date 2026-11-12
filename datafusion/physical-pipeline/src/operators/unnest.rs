// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`PhysicalUnnest`]: the unnesting of lists and structs (`UnnestExec`).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, FixedSizeListArray, Int64Array, LargeListArray,
    LargeListViewArray, ListArray, ListViewArray, PrimitiveArray, RecordBatch, Scalar,
    StructArray, UInt32Array, new_empty_array,
};
use arrow::compute::kernels::length::length;
use arrow::compute::kernels::zip::zip;
use arrow::compute::{cast, is_not_null, sum, take};
use arrow::datatypes::{DataType, Int64Type, SchemaRef};
use arrow_ord::cmp::{eq, lt};
use datafusion_common::utils::apply_parent_nulls;
use datafusion_common::{
    Result, UnnestOptions, exec_datafusion_err, exec_err, internal_err,
};
use datafusion_execution::memory_pool::{MemoryConsumer, MemoryReservation};
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::unnest::{ListUnnest, UnnestExec};

use crate::chunk::{DEFAULT_VECTOR_SIZE, DataChunk};
use crate::operator::{
    ExecutionContext, GlobalOperatorState, OperatorResult, OperatorState,
    PhysicalOperator, downcast_state_mut,
};

/// Unnests the lists and structs of its input, as `UnnestExec` does.
///
/// Every list column (an `ARRAY`, a large list, a fixed-size list or a list
/// view) that is unnested turns every row into as many rows as its longest
/// list has values: a list that is shorter is padded with nulls, and the
/// columns that are not unnested repeat the value of the row. A column may be
/// unnested at several depths (`unnest(unnest(x))` or `recursive := true`),
/// level by level, as DuckDB does. Every struct column that is unnested
/// becomes the columns of its fields (which are null in the rows where the
/// struct is null). How a `NULL` list, and an empty list, is unnested is
/// set by the `NullHandling` of the options: into no row (`Drop`), one row of
/// nulls (`Preserve`, for `NULL` lists only, the default), or one row of
/// nulls for both (`PreserveAndExpandEmpty`).
///
/// The operator is a parallel streaming operator: every thread unnests its
/// chunks, and emits their rows in chunks of at most
/// [`DEFAULT_VECTOR_SIZE`] rows. If the number of rows that every row expands
/// into is known in advance (no column is unnested more than one level deep),
/// the rows of a chunk are unnested in windows of at most
/// [`DEFAULT_VECTOR_SIZE`] rows, also those of a row that expands into many
/// rows; otherwise all rows of the chunk are unnested at once, level by
/// level, as DataFusion does. The memory of the indices that the rows are
/// taken with, and an estimate of the memory of the rows, are reserved in the
/// memory pool of the session before they are allocated, and the memory of
/// the unnested rows until they are emitted: an unnest fails with
/// `ResourcesExhausted` before it allocates what the pool cannot hold.
#[derive(Debug)]
pub struct PhysicalUnnest {
    /// The input, the only child.
    children: Vec<Arc<dyn PhysicalOperator>>,
    schema: SchemaRef,
    /// The unnests of list columns: the index of the column in the input, and
    /// the depth of the unnest.
    lists: Vec<ListUnnest>,
    /// The indices of the struct columns in the input that are unnested.
    structs: HashSet<usize>,
    options: UnnestOptions,
}

impl PhysicalUnnest {
    /// Unnest the list columns `lists` and the struct columns `structs` of
    /// `input`, with `options`, into rows of the schema `schema`.
    pub fn try_new(
        input: Arc<dyn PhysicalOperator>,
        lists: Vec<ListUnnest>,
        structs: Vec<usize>,
        schema: SchemaRef,
        options: UnnestOptions,
    ) -> Result<Self> {
        let input_schema = input.schema();
        let columns = input_schema.fields().len();
        for list in &lists {
            if list.index_in_input_schema >= columns || list.depth == 0 {
                return internal_err!(
                    "PhysicalUnnest cannot unnest column {} at depth {}",
                    list.index_in_input_schema,
                    list.depth
                );
            }
        }
        for &column in &structs {
            if !matches!(
                input_schema.fields().get(column).map(|f| f.data_type()),
                Some(DataType::Struct(_))
            ) {
                return internal_err!(
                    "PhysicalUnnest cannot unnest column {column}: it is not a struct"
                );
            }
        }
        Ok(Self {
            children: vec![input],
            schema,
            lists,
            structs: structs.into_iter().collect(),
            options,
        })
    }

    /// Lower an `UnnestExec` to a [`PhysicalUnnest`], whose input is
    /// `lower_input` of the input of the node. Returns `Ok(None)` for any
    /// other node, without calling `lower_input`.
    pub fn try_lower<F>(
        plan: &Arc<dyn ExecutionPlan>,
        lower_input: F,
    ) -> Result<Option<Self>>
    where
        F: FnOnce(&Arc<dyn ExecutionPlan>) -> Result<Arc<dyn PhysicalOperator>>,
    {
        let Some(unnest) = plan.downcast_ref::<UnnestExec>() else {
            return Ok(None);
        };
        let input = lower_input(unnest.input())?;
        Self::try_new(
            input,
            unnest.list_column_indices().to_vec(),
            unnest.struct_column_indices().to_vec(),
            plan.schema(),
            unnest.options().clone(),
        )
        .map(Some)
    }

    /// The numbers of rows into which the rows of `batch` expand, if they
    /// are known before unnesting: if no column is unnested more than one
    /// level deep (and a list column is unnested at all).
    fn output_lengths(&self, batch: &RecordBatch) -> Result<Option<Int64Array>> {
        if self.lists.is_empty() || self.lists.iter().any(|list| list.depth != 1) {
            return Ok(None);
        }
        let lists = self
            .lists
            .iter()
            .map(|list| Arc::clone(batch.column(list.index_in_input_schema)))
            .collect::<Vec<_>>();
        let lengths = longest_lengths(&lists, &self.options)?;
        Ok(Some(lengths.as_primitive::<Int64Type>().clone()))
    }
}

/// The rows of the chunk that a thread unnests.
struct Unnesting {
    /// The rows of the chunk.
    input: RecordBatch,
    /// If no column is unnested more than one level deep: the numbers of
    /// rows into which the rows of `input` expand, which are unnested in
    /// windows of at most [`DEFAULT_VECTOR_SIZE`] rows (see
    /// [`PhysicalUnnest::next_window`]), from the row `offset` of the
    /// expansion of the row `row` of `input` on.
    lengths: Option<Int64Array>,
    row: usize,
    offset: i64,
    /// Otherwise: whether all rows of `input` are unnested (at once), the
    /// unnested rows that are not emitted yet, and the first of them.
    unnested: bool,
    output: Option<RecordBatch>,
    emitted: usize,
}

/// The state of a thread of a [`PhysicalUnnest`]: the chunk it unnests, and
/// the memory of its unnested rows.
struct UnnestState {
    unnesting: Option<Unnesting>,
    reservation: MemoryReservation,
}

impl OperatorState for UnnestState {}

impl PhysicalOperator for PhysicalUnnest {
    fn name(&self) -> &str {
        "PhysicalUnnest"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.children
    }

    fn description(&self) -> String {
        let lists = self
            .lists
            .iter()
            .map(|list| format!("{}@depth={}", list.index_in_input_schema, list.depth))
            .collect::<Vec<_>>()
            .join(", ");
        let mut structs = self.structs.iter().copied().collect::<Vec<_>>();
        structs.sort_unstable();
        format!(
            "PhysicalUnnest: lists=[{lists}], structs={structs:?}, null_handling={:?}",
            self.options.null_handling
        )
    }

    fn get_operator_state(
        &self,
        ctx: &ExecutionContext,
        _gstate: &dyn GlobalOperatorState,
    ) -> Result<Box<dyn OperatorState>> {
        let reservation =
            MemoryConsumer::new("PhysicalUnnest").register(ctx.task_ctx.memory_pool());
        Ok(Box::new(UnnestState {
            unnesting: None,
            reservation,
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
        let state = downcast_state_mut::<UnnestState>(lstate.as_any_mut(), self.name())?;
        // A chunk that is not being unnested is a new input chunk.
        let unnesting = match &mut state.unnesting {
            Some(unnesting) => unnesting,
            None => {
                let batch = input.compact()?;
                let lengths = self.output_lengths(&batch)?;
                state.unnesting.insert(Unnesting {
                    input: batch,
                    lengths,
                    row: 0,
                    offset: 0,
                    unnested: false,
                    output: None,
                    emitted: 0,
                })
            }
        };
        if unnesting.lengths.is_some() {
            if let Some(window) = self.next_window(unnesting, &state.reservation)? {
                *output = DataChunk::from_record_batch(window);
                if unnesting.row < unnesting.input.num_rows() {
                    return Ok(OperatorResult::HaveMoreOutput);
                }
            }
            state.unnesting = None;
            return Ok(OperatorResult::NeedMoreInput);
        }
        if !unnesting.unnested {
            unnesting.unnested = true;
            // Reserves the memory of every level before it allocates it.
            let reservation = &state.reservation;
            let unnested = unnest_batch(
                &unnesting.input,
                &self.schema,
                &self.lists,
                &self.structs,
                &self.options,
                &mut |bytes| reservation.try_grow(bytes),
            )?;
            reservation.try_resize(
                unnested
                    .as_ref()
                    .map_or(0, RecordBatch::get_array_memory_size),
            )?;
            unnesting.output = unnested;
        }
        if let Some(batch) = &unnesting.output {
            let len = (batch.num_rows() - unnesting.emitted).min(DEFAULT_VECTOR_SIZE);
            *output = DataChunk::from_record_batch(batch.slice(unnesting.emitted, len));
            unnesting.emitted += len;
            if unnesting.emitted < batch.num_rows() {
                return Ok(OperatorResult::HaveMoreOutput);
            }
        }
        state.unnesting = None;
        state.reservation.free();
        Ok(OperatorResult::NeedMoreInput)
    }
}

impl PhysicalUnnest {
    /// The next window of the rows into which the rows of `unnesting` expand,
    /// if no column is unnested more than one level deep: at most
    /// [`DEFAULT_VECTOR_SIZE`] rows, which may expand several rows of the
    /// input, or a part of the expansion of a row (a row that expands into
    /// many rows is unnested in several windows). `None` if no row is left.
    ///
    /// The memory of the indices that the window takes its rows with, and an
    /// estimate of the memory of its rows, are reserved in `reservation`
    /// before they are allocated; then the memory of the rows of the window,
    /// which the operator emits.
    fn next_window(
        &self,
        unnesting: &mut Unnesting,
        reservation: &MemoryReservation,
    ) -> Result<Option<RecordBatch>> {
        let Some(lengths) = &unnesting.lengths else {
            return internal_err!("PhysicalUnnest unnests windows of known lengths");
        };
        let input = &unnesting.input;
        // The parts of the expansions of the rows of the window: the row,
        // and the rows of its expansion.
        let mut spans = vec![];
        let mut rows = 0;
        while unnesting.row < input.num_rows() && rows < DEFAULT_VECTOR_SIZE {
            let length = lengths.value(unnesting.row);
            let take =
                (length - unnesting.offset).min((DEFAULT_VECTOR_SIZE - rows) as i64);
            if take > 0 {
                spans.push((unnesting.row, unnesting.offset, unnesting.offset + take));
                rows += take as usize;
                unnesting.offset += take;
            }
            if unnesting.offset >= length {
                unnesting.row += 1;
                unnesting.offset = 0;
            }
        }
        if rows == 0 {
            return Ok(None);
        }
        let repeated = (0..input.num_columns())
            .filter(|&column| {
                !self
                    .lists
                    .iter()
                    .any(|list| list.index_in_input_schema == column)
            })
            .map(|column| input.column(column))
            .collect::<Vec<_>>();
        let values = self
            .lists
            .iter()
            .map(|list| list_values(input.column(list.index_in_input_schema)))
            .collect::<Result<Vec<_>>>()?;
        let estimate = size_of::<i64>() * rows * (1 + self.lists.len())
            + repeated
                .iter()
                .copied()
                .chain(values.iter())
                .map(|array| bytes_per_row(array) * rows)
                .sum::<usize>();
        reservation.try_resize(estimate)?;

        // The rows of the columns that are not unnested, repeated.
        let mut indices = UInt32Array::builder(rows);
        for &(row, from, to) in &spans {
            for _ in from..to {
                indices.append_value(row as u32);
            }
        }
        let indices = indices.finish();
        // The values of every unnested list, padded with nulls.
        let mut unnests = Vec::with_capacity(self.lists.len());
        for (list, values) in self.lists.iter().zip(&values) {
            let array = input.column(list.index_in_input_schema);
            let mut positions = PrimitiveArray::<Int64Type>::builder(rows);
            for &(row, from, to) in &spans {
                let (start, end) = match array.is_null(row) {
                    true => (0, 0),
                    false => list_offsets(array, row)?,
                };
                for offset in from..to {
                    if start + offset < end {
                        positions.append_value(start + offset);
                    } else {
                        positions.append_null();
                    }
                }
            }
            unnests.push(take(values.as_ref(), &positions.finish(), None)?);
        }
        let mut columns = Vec::with_capacity(self.schema.fields().len());
        for (index, column) in input.columns().iter().enumerate() {
            let mut replaced = false;
            for (list, unnest) in self.lists.iter().zip(&unnests) {
                if list.index_in_input_schema == index {
                    columns.push(Arc::clone(unnest));
                    replaced = true;
                }
            }
            if !replaced {
                columns.push(take(column.as_ref(), &indices, None)?);
            }
        }
        let window = flatten_structs(&columns, &self.schema, &self.structs)?;
        reservation.try_resize(window.get_array_memory_size())?;
        Ok(Some(window))
    }
}

/// The memory of `array` per row, rounded up.
fn bytes_per_row(array: &ArrayRef) -> usize {
    array.get_array_memory_size().div_ceil(array.len().max(1))
}

/// The values of the lists of the list array `list`.
fn list_values(list: &ArrayRef) -> Result<ArrayRef> {
    Ok(match list.data_type() {
        DataType::List(_) => Arc::clone(list.as_list::<i32>().values()),
        DataType::LargeList(_) => Arc::clone(list.as_list::<i64>().values()),
        DataType::FixedSizeList(_, _) => Arc::clone(list.as_fixed_size_list().values()),
        DataType::ListView(_) => Arc::clone(list.as_list_view::<i32>().values()),
        DataType::LargeListView(_) => Arc::clone(list.as_list_view::<i64>().values()),
        other => return exec_err!("Invalid unnest datatype {other}"),
    })
}

/// The offsets of the first value of the list of the row `row` of the list
/// array `list`, and after its last value.
fn list_offsets(list: &ArrayRef, row: usize) -> Result<(i64, i64)> {
    Ok(match list.data_type() {
        DataType::List(_) => list.as_list::<i32>().value_range(row),
        DataType::LargeList(_) => list.as_list::<i64>().value_range(row),
        DataType::FixedSizeList(_, _) => list.as_fixed_size_list().value_range(row),
        DataType::ListView(_) => list.as_list_view::<i32>().value_range(row),
        DataType::LargeListView(_) => list.as_list_view::<i64>().value_range(row),
        other => return exec_err!("Invalid unnest datatype {other}"),
    })
}

/// Unnest the rows of `batch` into rows of `schema`: the list columns
/// `lists`, level by level from the deepest, and then the struct columns
/// `structs`. Returns `None` if the rows expand into no row. `reserve`
/// reserves the memory of every level before it is allocated (see
/// [`unnest_level`]).
///
/// This is the unnesting of DataFusion's `UnnestExec` (`build_batch`).
fn unnest_batch(
    batch: &RecordBatch,
    schema: &SchemaRef,
    lists: &[ListUnnest],
    structs: &HashSet<usize>,
    options: &UnnestOptions,
    reserve: &mut dyn FnMut(usize) -> Result<()>,
) -> Result<Option<RecordBatch>> {
    if lists.is_empty() {
        // The fields of the structs, with the nulls of their structs.
        reserve(batch.num_rows().div_ceil(8) * schema.fields().len())?;
        return flatten_structs(batch.columns(), schema, structs).map(Some);
    }
    let depth = lists.iter().map(|list| list.depth).max().unwrap_or(1);
    // The columns of the input, with the values repeated by the levels
    // unnested so far, and the unnested lists of every unnest.
    let mut columns = batch.columns().to_vec();
    let mut unnested: HashMap<ListUnnest, ArrayRef> = HashMap::with_capacity(lists.len());
    for level in (1..=depth).rev() {
        let Some(repeated) =
            unnest_level(&columns, lists, &mut unnested, level, options, reserve)?
        else {
            return Ok(None);
        };
        columns = repeated;
    }
    // Replace every unnested column by its unnests, in the order of the
    // unnests (e.g. `unnest(unnest(x))` before `unnest(x)`).
    let mut by_column: HashMap<usize, Vec<(usize, ArrayRef)>> = HashMap::new();
    for (order, list) in lists.iter().enumerate() {
        let Some(array) = unnested.get(list).map(Arc::clone) else {
            return internal_err!(
                "PhysicalUnnest did not unnest column {} at depth {}",
                list.index_in_input_schema,
                list.depth
            );
        };
        by_column
            .entry(list.index_in_input_schema)
            .or_default()
            .push((order, array));
    }
    let columns = columns
        .into_iter()
        .enumerate()
        .flat_map(|(index, column)| match by_column.remove(&index) {
            Some(mut unnests) => {
                unnests.sort_by_key(|(order, _)| *order);
                unnests.into_iter().map(|(_, array)| array).collect()
            }
            None => vec![column],
        })
        .collect::<Vec<_>>();
    flatten_structs(&columns, schema, structs).map(Some)
}

/// Unnest one level, `level`, of the list columns of `columns` (the columns
/// of the input, with the values that the levels unnested so far repeated):
/// the unnests whose depth is `level` start at this level, from the column of
/// the input, and those that started at a level above continue with their
/// lists of the level above (in `unnested`). The lists of this level of every
/// unnest replace those in `unnested`. Returns the columns repeated for the
/// rows of this level, `None` if there is none.
///
/// This is `list_unnest_at_level` of DataFusion's `UnnestExec`.
fn unnest_level(
    columns: &[ArrayRef],
    lists: &[ListUnnest],
    unnested: &mut HashMap<ListUnnest, ArrayRef>,
    level: usize,
    options: &UnnestOptions,
    reserve: &mut dyn FnMut(usize) -> Result<()>,
) -> Result<Option<Vec<ArrayRef>>> {
    let mut arrays = vec![];
    let mut unnests = vec![];
    for list in lists {
        let array = match list.depth.cmp(&level) {
            std::cmp::Ordering::Equal => Arc::clone(&columns[list.index_in_input_schema]),
            std::cmp::Ordering::Greater => match unnested.get(list) {
                Some(array) => Arc::clone(array),
                None => {
                    return internal_err!(
                        "PhysicalUnnest did not unnest column {} above level {level}",
                        list.index_in_input_schema
                    );
                }
            },
            std::cmp::Ordering::Less => continue,
        };
        arrays.push(array);
        unnests.push(*list);
    }
    let lengths = longest_lengths(&arrays, options)?
        .as_primitive::<Int64Type>()
        .clone();
    let total = if lengths.is_empty() {
        0
    } else {
        sum(&lengths).ok_or_else(|| {
            exec_datafusion_err!("Failed to calculate the total unnested length")
        })? as usize
    };
    if total == 0 {
        return Ok(None);
    }
    // A column that a level below unnests, or that no level unnests, is
    // repeated; the others are replaced by their unnests (they are not read
    // again).
    let repeated = (0..columns.len())
        .map(|index| {
            let needed_below = lists
                .iter()
                .any(|list| list.index_in_input_schema == index && list.depth < level);
            let unnested_column =
                lists.iter().any(|list| list.index_in_input_schema == index);
            needed_below || !unnested_column
        })
        .collect::<Vec<_>>();
    // The indices of the level, and the arrays it takes with them.
    let values = arrays.iter().map(list_values).collect::<Result<Vec<_>>>()?;
    reserve(
        size_of::<i64>() * total * (1 + arrays.len())
            + columns
                .iter()
                .zip(&repeated)
                .filter(|(_, repeated)| **repeated)
                .map(|(column, _)| column)
                .chain(&values)
                .map(|array| bytes_per_row(array) * total)
                .sum::<usize>(),
    )?;
    for (array, list) in arrays.iter().zip(&unnests) {
        unnested.insert(*list, unnest_list(array, &lengths, total)?);
    }
    let indices = repeated_indices(&lengths, total);
    columns
        .iter()
        .zip(repeated)
        .map(|(column, repeated)| match repeated {
            true => Ok(take(column.as_ref(), &indices, None)?),
            false => Ok(new_empty_array(column.data_type())),
        })
        .collect::<Result<Vec<_>>>()
        .map(Some)
}

/// For every row of `lists`, the length of its longest list, which is the
/// number of rows into which the row expands. A `NULL` list counts as one
/// value if the options preserve nulls, an empty list as one value if they
/// expand empty lists.
fn longest_lengths(lists: &[ArrayRef], options: &UnnestOptions) -> Result<ArrayRef> {
    let null_length = Scalar::new(Int64Array::from_value(
        i64::from(options.preserve_nulls()),
        1,
    ));
    let zero = Scalar::new(Int64Array::from_value(0, 1));
    let one = Scalar::new(Int64Array::from_value(1, 1));
    let lengths = lists
        .iter()
        .map(|list| {
            let lengths = cast(&length(list)?, &DataType::Int64)?;
            let mut lengths = zip(&is_not_null(&lengths)?, &lengths, &null_length)?;
            if options.expand_empty_as_null() {
                lengths = zip(&eq(&lengths, &zero)?, &one, &lengths)?;
            }
            Ok(lengths)
        })
        .collect::<Result<Vec<_>>>()?;
    let Some((first, others)) = lengths.split_first() else {
        return internal_err!("PhysicalUnnest unnests no list");
    };
    others
        .iter()
        .try_fold(Arc::clone(first), |longest, lengths| {
            Ok(zip(&lt(&longest, lengths)?, lengths, &longest)?)
        })
}

/// The values of the lists of `list`, every list padded with nulls to its
/// length in `lengths` (`total` values in all).
fn unnest_list(
    list: &ArrayRef,
    lengths: &PrimitiveArray<Int64Type>,
    total: usize,
) -> Result<ArrayRef> {
    fn unnest<L: ListOffsets>(
        list: &L,
        lengths: &PrimitiveArray<Int64Type>,
        total: usize,
    ) -> Result<ArrayRef> {
        let mut indices = PrimitiveArray::<Int64Type>::builder(total);
        for row in 0..list.len() {
            let mut values = 0;
            if !list.is_null(row) {
                let (start, end) = list.value_range(row);
                values = end - start;
                for index in start..end {
                    indices.append_value(index);
                }
            }
            for _ in values..lengths.value(row) {
                indices.append_null();
            }
        }
        Ok(take(list.values().as_ref(), &indices.finish(), None)?)
    }
    match list.data_type() {
        DataType::List(_) => unnest(list.as_list::<i32>(), lengths, total),
        DataType::LargeList(_) => unnest(list.as_list::<i64>(), lengths, total),
        DataType::FixedSizeList(_, _) => {
            unnest(list.as_fixed_size_list(), lengths, total)
        }
        DataType::ListView(_) => unnest(list.as_list_view::<i32>(), lengths, total),
        DataType::LargeListView(_) => unnest(list.as_list_view::<i64>(), lengths, total),
        other => exec_err!("Invalid unnest datatype {other}"),
    }
}

/// The offsets of the values of the lists of a list array.
trait ListOffsets: Array {
    fn values(&self) -> &ArrayRef;

    /// The offsets of the first value of the list of `row`, and after its
    /// last value.
    fn value_range(&self, row: usize) -> (i64, i64);
}

impl ListOffsets for ListArray {
    fn values(&self) -> &ArrayRef {
        ListArray::values(self)
    }

    fn value_range(&self, row: usize) -> (i64, i64) {
        let offsets = self.value_offsets();
        (offsets[row].into(), offsets[row + 1].into())
    }
}

impl ListOffsets for LargeListArray {
    fn values(&self) -> &ArrayRef {
        LargeListArray::values(self)
    }

    fn value_range(&self, row: usize) -> (i64, i64) {
        let offsets = self.value_offsets();
        (offsets[row], offsets[row + 1])
    }
}

impl ListOffsets for FixedSizeListArray {
    fn values(&self) -> &ArrayRef {
        FixedSizeListArray::values(self)
    }

    fn value_range(&self, row: usize) -> (i64, i64) {
        let start = self.value_offset_at(row) as i64;
        (start, start + self.value_length() as i64)
    }
}

impl ListOffsets for ListViewArray {
    fn values(&self) -> &ArrayRef {
        ListViewArray::values(self)
    }

    fn value_range(&self, row: usize) -> (i64, i64) {
        let offset = self.value_offsets()[row] as i64;
        (offset, offset + self.value_sizes()[row] as i64)
    }
}

impl ListOffsets for LargeListViewArray {
    fn values(&self) -> &ArrayRef {
        LargeListViewArray::values(self)
    }

    fn value_range(&self, row: usize) -> (i64, i64) {
        let offset = self.value_offsets()[row];
        (offset, offset + self.value_sizes()[row])
    }
}

/// The indices of the rows that the rows of the columns that are not
/// unnested repeat: every row `lengths[row]` times (`total` in all).
fn repeated_indices(
    lengths: &PrimitiveArray<Int64Type>,
    total: usize,
) -> PrimitiveArray<Int64Type> {
    let mut indices = PrimitiveArray::<Int64Type>::builder(total);
    for (row, &length) in lengths.values().iter().enumerate() {
        for _ in 0..length {
            indices.append_value(row as i64);
        }
    }
    indices.finish()
}

/// `columns`, with the struct columns `structs` replaced by the columns of
/// their fields (null where the struct is null), as rows of `schema`.
fn flatten_structs(
    columns: &[ArrayRef],
    schema: &SchemaRef,
    structs: &HashSet<usize>,
) -> Result<RecordBatch> {
    let mut flattened = Vec::with_capacity(schema.fields().len());
    for (index, column) in columns.iter().enumerate() {
        if !structs.contains(&index) {
            flattened.push(Arc::clone(column));
            continue;
        }
        let Some(fields) = column.as_any().downcast_ref::<StructArray>() else {
            return internal_err!(
                "PhysicalUnnest expects column {index} to be a struct, got {}",
                column.data_type()
            );
        };
        for field in fields.columns() {
            flattened.push(apply_parent_nulls(field, fields.nulls())?);
        }
    }
    let rows = columns.first().map_or(0, |column| column.len());
    let options = arrow::array::RecordBatchOptions::new().with_row_count(Some(rows));
    Ok(RecordBatch::try_new_with_options(
        Arc::clone(schema),
        flattened,
        &options,
    )?)
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::array::{Int32Array, ListBuilder, StringBuilder, StructBuilder};
    use arrow::datatypes::{Field, Schema};
    use datafusion::datasource::MemTable;
    use datafusion::prelude::SessionContext;
    use datafusion_common::NullHandling;

    use crate::operators::lowering_tests::{
        assert_lowered, assert_resources_exhausted, count, lowered,
        pipeline_context_with_memory_limit,
    };
    use crate::test_utils::{register_test_tables, test_session_config};

    /// `nested`: `id`, a list of integers `l` (with nulls, empty lists and
    /// lists of nulls), a list of lists of strings `ll`, and a struct `st`
    /// (with nulls), in three partitions; `big`: 20 rows whose lists of 1000
    /// values expand into 20000 rows.
    fn register_nested(ctx: &SessionContext) -> Result<()> {
        register_test_tables(ctx)?;
        let schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new(
                "l",
                DataType::List(Arc::new(Field::new_list_field(DataType::Int32, true))),
                true,
            ),
            Field::new(
                "ll",
                DataType::List(Arc::new(Field::new_list_field(
                    DataType::List(Arc::new(Field::new_list_field(DataType::Utf8, true))),
                    true,
                ))),
                true,
            ),
            Field::new(
                "st",
                DataType::Struct(
                    vec![
                        Field::new("x", DataType::Int32, true),
                        Field::new("y", DataType::Utf8, true),
                    ]
                    .into(),
                ),
                true,
            ),
        ]));
        let batch = |ids: std::ops::Range<i64>| -> Result<RecordBatch> {
            let mut l = ListBuilder::new(arrow::array::Int32Builder::new());
            let mut ll = ListBuilder::new(ListBuilder::new(StringBuilder::new()));
            let mut st = StructBuilder::from_fields(
                vec![
                    Field::new("x", DataType::Int32, true),
                    Field::new("y", DataType::Utf8, true),
                ],
                0,
            );
            for id in ids.clone() {
                match id % 6 {
                    0 => l.append_null(),
                    1 => l.append(true),
                    _ => {
                        for v in 0..(id % 5) {
                            l.values()
                                .append_option((v != 2).then_some((id * v) as i32));
                        }
                        l.append(true);
                    }
                }
                if id % 7 == 3 {
                    ll.append_null();
                } else {
                    for inner in 0..(id % 3) {
                        if inner == 1 && id % 2 == 0 {
                            ll.values().append_null();
                            continue;
                        }
                        for v in 0..(id % 4) {
                            ll.values().values().append_value(format!("s{id}-{v}"));
                        }
                        ll.values().append(true);
                    }
                    ll.append(true);
                }
                st.field_builder::<arrow::array::Int32Builder>(0)
                    .unwrap()
                    .append_option((id % 4 != 1).then_some(id as i32));
                st.field_builder::<StringBuilder>(1)
                    .unwrap()
                    .append_option((id % 3 != 2).then(|| format!("y{id}")));
                st.append(id % 9 != 4);
            }
            Ok(RecordBatch::try_new(
                Arc::clone(&schema),
                vec![
                    Arc::new(Int64Array::from_iter_values(ids)),
                    Arc::new(l.finish()),
                    Arc::new(ll.finish()),
                    Arc::new(st.finish()),
                ],
            )?)
        };
        let table = MemTable::try_new(
            Arc::clone(&schema),
            vec![
                vec![batch(0..3000)?, batch(3000..3100)?],
                vec![],
                vec![batch(5000..5050)?],
            ],
        )?;
        ctx.register_table("nested", Arc::new(table))?;

        let big_schema = Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int32, false),
            Field::new(
                "values",
                DataType::List(Arc::new(Field::new_list_field(DataType::Utf8, true))),
                true,
            ),
        ]));
        let mut values = ListBuilder::new(StringBuilder::new());
        for id in 0..20 {
            for v in 0..1000 {
                values.values().append_value(format!("v{id}-{v}"));
            }
            values.append(true);
        }
        let big = RecordBatch::try_new(
            Arc::clone(&big_schema),
            vec![
                Arc::new(Int32Array::from_iter_values(0..20)),
                Arc::new(values.finish()),
            ],
        )?;
        ctx.register_table(
            "big",
            Arc::new(MemTable::try_new(big_schema, vec![vec![big]])?),
        )?;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn unnests_are_native() {
        for sql in [
            "SELECT unnest([1, 2, 3]) AS x",
            "SELECT unnest(struct(1 AS x, 'a' AS y))",
            "SELECT id, unnest(l) AS v FROM nested",
            "SELECT id, unnest(l) AS v, unnest(ll) AS w FROM nested",
            "SELECT id, unnest(ll) AS w FROM nested WHERE id % 5 = 1",
            "SELECT id, unnest(unnest(ll)) AS s FROM nested",
            "SELECT id, unnest(unnest(ll)) AS s, unnest(ll) AS w, unnest(l) AS v \
             FROM nested",
            "SELECT id, unnest(st) FROM nested",
            "SELECT unnest(st), unnest(l) FROM nested",
            "SELECT v, COUNT(*) AS n FROM (SELECT unnest(l) AS v FROM nested) GROUP BY v",
            "SELECT id, unnest(make_array(a, b)) AS ab FROM t1",
            "SELECT * FROM unnest([1, NULL, 3], ['a', 'b'])",
            "SELECT id, unnest(values) AS v FROM big",
            "SELECT COUNT(*), COUNT(DISTINCT v) FROM (SELECT unnest(values) AS v FROM big)",
            "SELECT unnest(range(1, id + 1)) AS r FROM (SELECT id FROM big WHERE id < 5)",
        ] {
            for target_partitions in [1, 4] {
                let config =
                    test_session_config().with_target_partitions(target_partitions);
                let root = assert_lowered(sql, config, async |ctx: &SessionContext| {
                    register_nested(ctx)
                })
                .await;
                assert!(count::<PhysicalUnnest>(&root) > 0, "{sql}");
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn nulls_and_empty_lists_are_unnested_as_their_options_say() {
        // The options of SQL's `unnest` preserve nulls; the data frame API
        // also drops them, or expands empty lists into nulls.
        for null_handling in [
            NullHandling::Preserve,
            NullHandling::Drop,
            NullHandling::PreserveAndExpandEmpty,
        ] {
            let options = UnnestOptions {
                null_handling,
                recursions: vec![],
            };
            for columns in [vec!["l"], vec!["l", "ll"], vec!["st", "l"]] {
                let (expected, actual) = {
                    let run = |ctx: SessionContext| {
                        let options = options.clone();
                        let columns = columns.clone();
                        async move {
                            register_nested(&ctx)?;
                            let df = ctx
                                .table("nested")
                                .await?
                                .unnest_columns_with_options(&columns, options)?;
                            let plan = df.clone().create_physical_plan().await?;
                            let batches = df.collect().await?;
                            Ok::<_, datafusion_common::DataFusionError>((plan, batches))
                        }
                    };
                    let expected =
                        run(crate::test_utils::datafusion_context(test_session_config()))
                            .await
                            .unwrap();
                    let actual =
                        run(crate::test_utils::pipeline_context(test_session_config()))
                            .await
                            .unwrap();
                    (expected, actual)
                };
                let output =
                    |(plan, batches): (Arc<dyn ExecutionPlan>, Vec<RecordBatch>)| {
                        crate::test_utils::EngineOutput {
                            schema: plan.schema(),
                            plan,
                            batches,
                            ordered: false,
                        }
                    };
                let actual = output(actual);
                let root = actual
                    .plan
                    .downcast_ref::<crate::bridge::PipelineRootExec>()
                    .unwrap()
                    .operator();
                assert!(crate::bridge::bridged_plans(root).is_empty());
                assert_eq!(count::<PhysicalUnnest>(root), 1);
                if let Err(message) =
                    crate::test_utils::compare_outputs(&output(expected), &actual)
                {
                    panic!("{null_handling:?} {columns:?}: {message}");
                }
            }
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn unnested_rows_reserve_memory() -> Result<()> {
        let sql = "SELECT id, unnest(values) AS v FROM big";
        let ctx = pipeline_context_with_memory_limit(10_000);
        register_nested(&ctx)?;
        let root = lowered(&ctx, sql).await?;
        assert_eq!(count::<PhysicalUnnest>(&root), 1);
        assert_resources_exhausted(&ctx, sql).await;
        Ok(())
    }

    /// `nulls`: a row whose list `l` has 1,000,000 nulls, and whose list of
    /// lists `ll` has a list of 1,000,000 nulls. Its arrays hold almost no
    /// memory, its unnests are 1,000,000 rows.
    fn register_lists_of_nulls(ctx: &SessionContext) -> Result<()> {
        use arrow::array::{ListArray, NullArray};
        use arrow::buffer::OffsetBuffer;

        let nulls = 1_000_000;
        let null_field = Arc::new(Field::new_list_field(DataType::Null, true));
        let l = ListArray::new(
            Arc::clone(&null_field),
            OffsetBuffer::from_lengths([nulls]),
            Arc::new(NullArray::new(nulls)),
            None,
        );
        let inner_field = Arc::new(Field::new_list_field(l.data_type().clone(), true));
        let ll = ListArray::new(
            Arc::clone(&inner_field),
            OffsetBuffer::from_lengths([1]),
            Arc::new(l.clone()),
            None,
        );
        let schema = Arc::new(Schema::new(vec![
            Field::new("l", l.data_type().clone(), true),
            Field::new("ll", ll.data_type().clone(), true),
        ]));
        let batch =
            RecordBatch::try_new(Arc::clone(&schema), vec![Arc::new(l), Arc::new(ll)])?;
        assert!(batch.get_array_memory_size() < 1024);
        ctx.register_table(
            "nulls",
            Arc::new(MemTable::try_new(schema, vec![vec![batch]])?),
        )?;
        Ok(())
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn the_indices_of_unnests_reserve_memory_before_they_are_allocated()
    -> Result<()> {
        // A list of a million nulls holds almost no memory, but its unnest
        // takes its rows with indices of a million entries (in windows of
        // `DEFAULT_VECTOR_SIZE` rows, also of a single row): their memory is
        // reserved before they are allocated, in one level or in several.
        for sql in [
            "SELECT unnest(l) AS v FROM nulls",
            "SELECT unnest(unnest(ll)) AS v FROM nulls",
        ] {
            let ctx = pipeline_context_with_memory_limit(1024);
            register_lists_of_nulls(&ctx)?;
            let root = lowered(&ctx, sql).await?;
            assert_eq!(count::<PhysicalUnnest>(&root), 1, "{sql}");
            assert_resources_exhausted(&ctx, sql).await;
            // Without a limit, the rows of the unnest (the projection below
            // it, of lists of nulls, which the native projection does not
            // gather, is left to DataFusion).
            let counted = format!("SELECT COUNT(*) AS n FROM ({sql})");
            let (expected, actual) = crate::test_utils::run_engines(
                &counted,
                test_session_config(),
                async |ctx: &SessionContext| register_lists_of_nulls(ctx),
            )
            .await?;
            if let Err(message) = crate::test_utils::compare_outputs(&expected, &actual) {
                panic!("{counted}: {message}");
            }
            let root = actual
                .plan
                .downcast_ref::<crate::bridge::PipelineRootExec>()
                .unwrap()
                .operator();
            assert_eq!(count::<PhysicalUnnest>(root), 1, "{sql}");
        }
        // The windows of a single row hold at most `DEFAULT_VECTOR_SIZE` rows.
        let ctx = crate::test_utils::pipeline_context(test_session_config());
        register_lists_of_nulls(&ctx)?;
        let batches = ctx
            .sql("SELECT unnest(l) AS v FROM nulls")
            .await?
            .collect()
            .await?;
        assert_eq!(
            batches.iter().map(RecordBatch::num_rows).sum::<usize>(),
            1_000_000
        );
        assert!(
            batches
                .iter()
                .all(|batch| batch.num_rows() <= DEFAULT_VECTOR_SIZE)
        );
        Ok(())
    }
}
