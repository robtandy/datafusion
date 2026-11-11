// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`PhysicalUnnest`]: the unnesting of lists and structs (`UnnestExec`).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, AsArray, FixedSizeListArray, Int64Array, LargeListArray,
    LargeListViewArray, ListArray, ListViewArray, PrimitiveArray, RecordBatch, Scalar,
    StructArray, new_null_array,
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
/// the rows of a chunk are unnested a few at a time, so that every step
/// unnests at most about [`DEFAULT_VECTOR_SIZE`] rows (but a single row that
/// expands into more); otherwise all rows of the chunk are unnested at once.
/// The memory of the unnested rows is reserved in the memory pool of the
/// session until they are emitted.
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
    /// The next row of `input` to unnest.
    row: usize,
    /// The numbers of rows into which the rows of `input` expand, if known.
    lengths: Option<Int64Array>,
    /// The unnested rows that are not emitted yet, and the first of them.
    output: Option<RecordBatch>,
    offset: usize,
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
                    row: 0,
                    lengths,
                    output: None,
                    offset: 0,
                })
            }
        };
        loop {
            if let Some(batch) = &unnesting.output {
                let rows = batch.num_rows() - unnesting.offset;
                let len = rows.min(DEFAULT_VECTOR_SIZE);
                *output =
                    DataChunk::from_record_batch(batch.slice(unnesting.offset, len));
                unnesting.offset += len;
                if unnesting.offset == batch.num_rows() {
                    unnesting.output = None;
                    state.reservation.free();
                }
                if unnesting.output.is_some()
                    || unnesting.row < unnesting.input.num_rows()
                {
                    return Ok(OperatorResult::HaveMoreOutput);
                }
                state.unnesting = None;
                return Ok(OperatorResult::NeedMoreInput);
            }
            let remaining = unnesting.input.num_rows() - unnesting.row;
            if remaining == 0 {
                state.unnesting = None;
                return Ok(OperatorResult::NeedMoreInput);
            }
            // The next rows of the chunk, about `DEFAULT_VECTOR_SIZE` rows
            // once unnested if the lengths are known (at least one row).
            let rows = match &unnesting.lengths {
                Some(lengths) => rows_expanding_into(
                    &lengths.values()[unnesting.row..],
                    DEFAULT_VECTOR_SIZE,
                ),
                None => remaining,
            };
            let rows_to_unnest = unnesting.input.slice(unnesting.row, rows);
            let lengths = unnesting
                .lengths
                .as_ref()
                .map(|lengths| lengths.slice(unnesting.row, rows));
            unnesting.row += rows;
            let unnested = unnest_batch(
                &rows_to_unnest,
                &self.schema,
                &self.lists,
                &self.structs,
                &self.options,
                lengths.as_ref(),
            )?;
            if let Some(batch) = unnested {
                state
                    .reservation
                    .try_resize(batch.get_array_memory_size())?;
                unnesting.output = Some(batch);
                unnesting.offset = 0;
            }
        }
    }
}

/// The number of the first rows of `lengths` (the numbers of rows into which
/// rows expand) that expand into at most `rows` rows, but at least one.
fn rows_expanding_into(lengths: &[i64], rows: usize) -> usize {
    let limit = rows as i64;
    let mut expanded = 0i64;
    for (row, &length) in lengths.iter().enumerate() {
        if row > 0 && expanded + length > limit {
            return row;
        }
        expanded += length;
    }
    lengths.len()
}

/// Unnest the rows of `batch` into rows of `schema`: the list columns
/// `lists`, level by level from the deepest, and then the struct columns
/// `structs`. `lengths` are the numbers of rows into which the rows expand,
/// if known (see [`PhysicalUnnest::output_lengths`]). Returns `None` if the
/// rows expand into no row.
///
/// This is the unnesting of DataFusion's `UnnestExec` (`build_batch`).
fn unnest_batch(
    batch: &RecordBatch,
    schema: &SchemaRef,
    lists: &[ListUnnest],
    structs: &HashSet<usize>,
    options: &UnnestOptions,
    lengths: Option<&PrimitiveArray<Int64Type>>,
) -> Result<Option<RecordBatch>> {
    if lists.is_empty() {
        return flatten_structs(batch.columns(), schema, structs).map(Some);
    }
    let depth = lists.iter().map(|list| list.depth).max().unwrap_or(1);
    // The columns of the input, with the values repeated by the levels
    // unnested so far, and the unnested lists of every unnest.
    let mut columns = batch.columns().to_vec();
    let mut unnested: HashMap<ListUnnest, ArrayRef> = HashMap::with_capacity(lists.len());
    for level in (1..=depth).rev() {
        // The lengths are only known in advance for a single level.
        let lengths = if depth == 1 { lengths } else { None };
        let Some(repeated) =
            unnest_level(&columns, lists, &mut unnested, level, options, lengths)?
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
    lengths: Option<&PrimitiveArray<Int64Type>>,
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
    let lengths = match lengths {
        Some(lengths) => lengths.clone(),
        None => longest_lengths(&arrays, options)?
            .as_primitive::<Int64Type>()
            .clone(),
    };
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
    for (array, list) in arrays.iter().zip(&unnests) {
        unnested.insert(*list, unnest_list(array, &lengths, total)?);
    }
    let indices = repeated_indices(&lengths, total);
    columns
        .iter()
        .enumerate()
        .map(|(index, column)| {
            // A column that a level below unnests, or that no level unnests,
            // is repeated; the others are replaced by their unnests.
            let needed_below = lists
                .iter()
                .any(|list| list.index_in_input_schema == index && list.depth < level);
            let unnested_column =
                lists.iter().any(|list| list.index_in_input_schema == index);
            if needed_below || !unnested_column {
                Ok(take(column.as_ref(), &indices, None)?)
            } else {
                Ok(new_null_array(column.data_type(), column.len()))
            }
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
                let (start, end) = list.offsets(row);
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
    fn offsets(&self, row: usize) -> (i64, i64);
}

impl ListOffsets for ListArray {
    fn values(&self) -> &ArrayRef {
        ListArray::values(self)
    }

    fn offsets(&self, row: usize) -> (i64, i64) {
        let offsets = self.value_offsets();
        (offsets[row].into(), offsets[row + 1].into())
    }
}

impl ListOffsets for LargeListArray {
    fn values(&self) -> &ArrayRef {
        LargeListArray::values(self)
    }

    fn offsets(&self, row: usize) -> (i64, i64) {
        let offsets = self.value_offsets();
        (offsets[row], offsets[row + 1])
    }
}

impl ListOffsets for FixedSizeListArray {
    fn values(&self) -> &ArrayRef {
        FixedSizeListArray::values(self)
    }

    fn offsets(&self, row: usize) -> (i64, i64) {
        let start = self.value_offset_at(row) as i64;
        (start, start + self.value_length() as i64)
    }
}

impl ListOffsets for ListViewArray {
    fn values(&self) -> &ArrayRef {
        ListViewArray::values(self)
    }

    fn offsets(&self, row: usize) -> (i64, i64) {
        let offset = self.value_offsets()[row] as i64;
        (offset, offset + self.value_sizes()[row] as i64)
    }
}

impl ListOffsets for LargeListViewArray {
    fn values(&self) -> &ArrayRef {
        LargeListViewArray::values(self)
    }

    fn offsets(&self, row: usize) -> (i64, i64) {
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

    #[test]
    fn rows_are_unnested_a_few_at_a_time() {
        assert_eq!(rows_expanding_into(&[1, 1, 1], 2), 2);
        assert_eq!(rows_expanding_into(&[5, 1, 1], 2), 1);
        assert_eq!(rows_expanding_into(&[0, 0, 3], 2), 2);
        assert_eq!(rows_expanding_into(&[1, 1], 2048), 2);
        assert_eq!(rows_expanding_into(&[3000], 2048), 1);
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
}
