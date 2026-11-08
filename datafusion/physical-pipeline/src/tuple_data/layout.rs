// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`TupleDataLayout`]: the fixed-width row layout of a
//! [`TupleDataCollection`](super::TupleDataCollection).
//!
//! Every row of a collection has the same width, so that a row is addressed
//! by the index of its block and its index in the block (a
//! [`RowLocation`](super::RowLocation)). A row consists of
//!
//! ```text
//! +----------------+----------+----------+-----+--------------+-----------------+
//! | validity bytes | column 0 | column 1 | ... | column n - 1 | hash (optional) |
//! +----------------+----------+----------+-----+--------------+-----------------+
//! ```
//!
//! * **validity**: one bit per column (bit `i % 8` of byte `i / 8`), set if
//!   the value of column `i` is valid (not null). The validity of a column
//!   is the *logical* validity of its Arrow array ([`Array::logical_nulls`]).
//!   Every dictionary column has one more bit, after the bits of the
//!   columns, set if its key is valid: a valid key can refer to a null value
//!   (also in a non-nullable field), so that the keys are restored exactly.
//! * **fixed-width columns** ([`ColumnStorage::Boolean`],
//!   [`ColumnStorage::Fixed`]) are stored inline in their Arrow byte
//!   representation (booleans as one byte, `0` or `1`). The bytes of null
//!   values are zero.
//! * **variable-length columns** ([`ColumnStorage::VarLen`]: `Utf8`,
//!   `LargeUtf8`, `Utf8View`, `Binary`, `LargeBinary` and `BinaryView`) use
//!   a 16 byte slot, like Arrow's views and DuckDB's `string_t`. Values of at
//!   most [`MAX_INLINE_LEN`] bytes are stored inline, longer values are
//!   stored in the heap of the block that holds the row:
//!
//!   ```text
//!   len <= 12: | len: u32 | data, zero padded (12 bytes)    |
//!   len >  12: | len: u32 | prefix (4 bytes) | offset: u64  |
//!   ```
//!
//!   Null values have length `0`.
//! * **nested columns** ([`ColumnStorage::Encoded`]: structs, lists, maps,
//!   unions, run-end encoded arrays) are encoded with the Arrow row format
//!   ([`arrow_row`]) and the encoding is stored like a variable-length value
//!   (also for null values, so that their exact encoding is preserved).
//!   What the row format decoder does not preserve is restored when values
//!   are gathered, so that the gathered arrays have exactly the original
//!   type: nested dictionaries (decoded to their values, dictionary encoded
//!   again with valid keys of null values in non-nullable fields and null
//!   keys otherwise) and the child fields of run-end encoded arrays
//!   (decoded with Arrow's default fields).
//! * **dictionary columns** are stored as their values: a
//!   `Dictionary(K, V)` column uses the storage of `V`, and is dictionary
//!   encoded again when it is gathered (with null keys exactly where the
//!   keys were null).
//! * the optional **hash** is a `u64` (little endian), e.g. the hash of the
//!   join or group keys of the row, computed once when the row is inserted.
//!
//! The layout is shared (`Arc<TupleDataLayout>`) by all collections that
//! store the same kind of rows, e.g. by all partitions of a
//! [`PartitionedTupleData`](super::PartitionedTupleData) and by all threads
//! of a sink.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{
    Array, ArrayData, ArrayRef, AsArray, UInt32Array, UInt64Array, make_array,
    new_null_array,
};
use arrow::buffer::{BooleanBuffer, NullBuffer, ScalarBuffer};
use arrow::compute::{CastOptions, cast_with_options, take};
use arrow::datatypes::{
    DataType, Field, FieldRef, IntervalUnit, Schema, SchemaRef, UnionMode,
};
use arrow_row::{RowConverter, RowParser, SortField};
use datafusion_common::float_semantics::has_float_leaf;
use datafusion_common::hash_utils::RandomState;
use datafusion_common::{
    DataFusionError, Result, exec_datafusion_err, internal_datafusion_err, internal_err,
    not_impl_err,
};

/// The width in bytes of the slot of a variable-length (or row encoded)
/// value.
pub const VAR_LEN_SLOT_WIDTH: usize = 16;

/// Variable-length values of at most this many bytes are stored inline in
/// their slot, longer values are stored in the heap.
pub const MAX_INLINE_LEN: usize = 12;

/// The width in bytes of the optional hash slot.
pub const HASH_WIDTH: usize = 8;

/// The target size in bytes of the row buffer of a block (DuckDB's default
/// block size).
pub const DEFAULT_BLOCK_SIZE: usize = 256 * 1024;

/// The maximum number of rows in a block: the row index of a
/// [`RowLocation`](super::RowLocation) has
/// [`RowLocation::ROW_BITS`](super::RowLocation::ROW_BITS) bits.
pub const MAX_ROWS_PER_BLOCK: usize = 1 << 16;

/// How the values of a column are stored in a row.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColumnStorage {
    /// `DataType::Null`: no value, only the validity bit (which is never
    /// set).
    Null,
    /// `DataType::Boolean`: one byte, `0` or `1`.
    Boolean,
    /// A fixed-width value of the given number of bytes, stored inline in
    /// its Arrow representation: integers, floats, decimals, dates, times,
    /// timestamps, durations, intervals and fixed size binaries.
    Fixed(usize),
    /// A variable-length string or binary value, in a
    /// [`VAR_LEN_SLOT_WIDTH`] byte slot.
    VarLen,
    /// A nested value encoded with the Arrow row format, stored like a
    /// [`ColumnStorage::VarLen`] value.
    Encoded,
}

impl ColumnStorage {
    /// The width in bytes of the slot of a value in a row.
    pub fn width(&self) -> usize {
        match self {
            Self::Null => 0,
            Self::Boolean => 1,
            Self::Fixed(width) => *width,
            Self::VarLen | Self::Encoded => VAR_LEN_SLOT_WIDTH,
        }
    }

    /// Returns `true` if values of this storage can be stored in the heap.
    pub fn uses_heap(&self) -> bool {
        matches!(self, Self::VarLen | Self::Encoded)
    }
}

/// The layout of one column in a row of a [`TupleDataLayout`].
#[derive(Debug)]
pub struct ColumnLayout {
    /// The data type of the column.
    data_type: DataType,
    /// The data type of the stored values: the value type of a dictionary,
    /// otherwise `data_type`.
    value_type: DataType,
    storage: ColumnStorage,
    /// The byte offset of the value slot in the row.
    offset: usize,
    /// The validity bit of the keys of a dictionary column.
    key_validity_bit: Option<usize>,
    /// The row format codec of a [`ColumnStorage::Encoded`] column.
    codec: Option<RowCodec>,
}

impl ColumnLayout {
    fn try_new(data_type: &DataType, offset: usize) -> Result<Self> {
        // Before anything is built for the type: Arrow cannot create rows
        // (e.g. a null row) of a union without fields.
        if has_union_without_fields(data_type) {
            return not_impl_err!(
                "TupleDataLayout does not support unions without fields: {data_type}"
            );
        }
        if has_shared_run_end_encoded(data_type) {
            return not_impl_err!(
                "TupleDataLayout does not support run-end encoded values in list views or \
                 dense unions, which can share them: {data_type}"
            );
        }
        if has_nested_dictionary_of_run_end_encoded(data_type) {
            return not_impl_err!(
                "TupleDataLayout does not support dictionaries of run-end encoded values \
                 inside other types: {data_type}"
            );
        }
        let value_type = match data_type {
            DataType::Dictionary(_, value_type) => {
                if matches!(value_type.as_ref(), DataType::Dictionary(..)) {
                    return not_impl_err!(
                        "TupleDataLayout does not support nested dictionaries: {data_type}"
                    );
                }
                value_type.as_ref().clone()
            }
            data_type => data_type.clone(),
        };
        let storage = match &value_type {
            DataType::Null => ColumnStorage::Null,
            DataType::Boolean => ColumnStorage::Boolean,
            DataType::Utf8
            | DataType::LargeUtf8
            | DataType::Utf8View
            | DataType::Binary
            | DataType::LargeBinary
            | DataType::BinaryView => ColumnStorage::VarLen,
            DataType::List(_)
            | DataType::LargeList(_)
            | DataType::ListView(_)
            | DataType::LargeListView(_)
            | DataType::FixedSizeList(..)
            | DataType::Struct(_)
            | DataType::Map(..)
            | DataType::Union(..)
            | DataType::RunEndEncoded(..) => ColumnStorage::Encoded,
            value_type => match fixed_width(value_type) {
                Some(width) => ColumnStorage::Fixed(width),
                None => {
                    return not_impl_err!(
                        "TupleDataLayout does not support the data type {data_type}"
                    );
                }
            },
        };
        let codec = match storage {
            ColumnStorage::Encoded => Some(RowCodec::try_new(&value_type)?),
            _ => None,
        };
        Ok(Self {
            data_type: data_type.clone(),
            value_type,
            storage,
            offset,
            key_validity_bit: None,
            codec,
        })
    }

    /// The data type of the column.
    pub fn data_type(&self) -> &DataType {
        &self.data_type
    }

    /// The data type of the stored values: the value type of a dictionary
    /// column, otherwise [`Self::data_type`].
    pub fn value_type(&self) -> &DataType {
        &self.value_type
    }

    /// Returns `true` if the column is dictionary encoded.
    pub fn is_dictionary(&self) -> bool {
        matches!(self.data_type, DataType::Dictionary(..))
    }

    /// How the values of the column are stored.
    pub fn storage(&self) -> ColumnStorage {
        self.storage
    }

    /// The byte offset of the value slot of the column in a row.
    pub fn offset(&self) -> usize {
        self.offset
    }

    /// The width in bytes of the value slot of the column in a row.
    pub fn width(&self) -> usize {
        self.storage.width()
    }

    /// The index of the bit of the validity bytes of a row that is set if
    /// the key of the value of a dictionary column is valid (`None` for
    /// other columns): a dictionary can have valid keys of null values, also
    /// in a non-nullable field, so the validity of the value does not tell
    /// whether its key is null.
    pub fn key_validity_bit(&self) -> Option<usize> {
        self.key_validity_bit
    }

    /// The row format codec of a [`ColumnStorage::Encoded`] column.
    pub(crate) fn codec(&self) -> Option<&RowCodec> {
        self.codec.as_ref()
    }
}

/// Encodes and decodes the values of a [`ColumnStorage::Encoded`] column
/// with the Arrow row format.
#[derive(Debug)]
pub(crate) struct RowCodec {
    /// Encodes the values of the column.
    pub(crate) converter: RowConverter,
    /// Decodes the values encoded by `converter`, with `Int64` run ends (see
    /// [`with_wide_run_ends`]): the row format encodes run-end encoded
    /// arrays by their values, whatever their run end type.
    decoder: RowConverter,
    /// Parses the rows of `decoder`.
    parser: RowParser,
    /// The value type of the column.
    value_type: DataType,
    /// The value type with every dictionary replaced by its values, and
    /// `Int64` run ends.
    flat_type: DataType,
    /// Whether two values can be equal for DataFusion although their row
    /// encodings differ, so that they must be compared by their decoded
    /// values: the row format distinguishes `-0.0` and `+0.0`, and the nulls
    /// of different variants of a union.
    pub(crate) compare_decoded: bool,
}

impl RowCodec {
    fn try_new(value_type: &DataType) -> Result<Self> {
        let fields = vec![SortField::new(value_type.clone())];
        if !RowConverter::supports_fields(&fields) {
            return not_impl_err!(
                "TupleDataLayout does not support the data type {value_type}"
            );
        }
        let converter = RowConverter::new(fields)?;
        let wide_type = with_wide_run_ends(value_type);
        let decoder = RowConverter::new(vec![SortField::new(wide_type.clone())])?;
        let parser = decoder.parser();
        let codec = Self {
            converter,
            decoder,
            parser,
            value_type: value_type.clone(),
            flat_type: without_dictionaries(&wide_type),
            compare_decoded: has_float_leaf(value_type) || has_union(value_type),
        };
        // Reject the types whose values cannot be restored exactly (or
        // without dictionaries, to compare them) now rather than when they
        // are gathered or compared.
        let round_trip = codec
            .converter
            .convert_columns(&[new_null_array(value_type, 1)])
            .map_err(DataFusionError::from)
            .and_then(|rows| {
                let encoded = || rows.iter().map(|row| row.data());
                Ok([
                    (codec.decode(encoded())?, value_type),
                    (codec.decode_flat(encoded())?, &codec.flat_type),
                ])
            });
        match round_trip {
            Ok(decoded) => {
                for (values, data_type) in decoded {
                    if !has_type(&values.to_data(), data_type) {
                        return not_impl_err!(
                            "TupleDataLayout does not support the data type {value_type}: \
                             its values are restored as {}",
                            values.data_type()
                        );
                    }
                }
                Ok(codec)
            }
            Err(e) => not_impl_err!(
                "TupleDataLayout does not support the data type {value_type}: {e}"
            ),
        }
    }

    /// Decode values encoded with [`Self::converter`] into an array of
    /// exactly the value type of the column.
    pub(crate) fn decode<'a>(
        &self,
        encoded: impl IntoIterator<Item = &'a [u8]>,
    ) -> Result<ArrayRef> {
        self.decode_as(encoded, &self.value_type)
    }

    /// Like [`Self::decode`], but with every dictionary replaced by its
    /// values and `Int64` run ends, e.g. to compare values: unlike values of
    /// the value type, any values can be decoded together (e.g. values from
    /// collections with different dictionaries, whose dictionaries do not
    /// need to fit the keys, or more run-end encoded values than the run end
    /// type counts).
    pub(crate) fn decode_flat<'a>(
        &self,
        encoded: impl IntoIterator<Item = &'a [u8]>,
    ) -> Result<ArrayRef> {
        self.decode_as(encoded, &self.flat_type)
    }

    /// `values` (of the value type of the column) with every dictionary
    /// replaced by its values, exactly as [`Self::decode_flat`] decodes
    /// them.
    pub(crate) fn flatten(&self, values: &ArrayRef) -> Result<ArrayRef> {
        let rows = self.converter.convert_columns(&[Arc::clone(values)])?;
        self.decode_flat(rows.iter().map(|row| row.data()))
    }

    fn decode_as<'a>(
        &self,
        encoded: impl IntoIterator<Item = &'a [u8]>,
        data_type: &DataType,
    ) -> Result<ArrayRef> {
        let rows = encoded.into_iter().map(|bytes| self.parser.parse(bytes));
        let values =
            self.decoder.convert_rows(rows)?.pop().ok_or_else(|| {
                internal_datafusion_err!("RowConverter returned no column")
            })?;
        let data = values.to_data();
        if has_type(&data, data_type) {
            return Ok(values);
        }
        Ok(make_array(restore(data, data_type, true, None)?))
    }
}

/// Cast options that fail instead of producing nulls.
pub(crate) fn strict_cast_options() -> CastOptions<'static> {
    CastOptions {
        safe: false,
        ..Default::default()
    }
}

/// The data types of the children of an [`ArrayData`] of type `data_type`,
/// and whether they are nullable.
fn child_data_types(data_type: &DataType) -> Vec<(&DataType, bool)> {
    fn field(field: &FieldRef) -> (&DataType, bool) {
        (field.data_type(), field.is_nullable())
    }
    match data_type {
        DataType::List(f)
        | DataType::LargeList(f)
        | DataType::ListView(f)
        | DataType::LargeListView(f)
        | DataType::FixedSizeList(f, _)
        | DataType::Map(f, _) => vec![field(f)],
        DataType::Struct(fields) => fields.iter().map(field).collect(),
        DataType::Union(fields, _) => fields.iter().map(|(_, f)| field(f)).collect(),
        DataType::RunEndEncoded(run_ends, values) => vec![field(run_ends), field(values)],
        DataType::Dictionary(_, values) => vec![(values.as_ref(), true)],
        _ => vec![],
    }
}

/// Whether `data` has the type `data_type` and its children, recursively,
/// the types of the fields of `data_type`.
fn has_type(data: &ArrayData, data_type: &DataType) -> bool {
    data.data_type() == data_type
        && data
            .child_data()
            .iter()
            .zip(child_data_types(data_type))
            .all(|(child, (data_type, _))| has_type(child, data_type))
}

/// Restore the data type `data_type` of values decoded by the row format
/// (`data`), whose values are `nullable` if they are in a nullable field, and
/// of which only the rows in `relevant` matter (`None`: all rows).
///
/// The row format decoder does not restore
///
/// * dictionaries, which it decodes to their values: they are dictionary
///   encoded again (see [`dictionary_encode`]). In nullable fields null
///   values get null keys. In non-nullable fields every key is valid: null
///   values that matter get the key of a null value, and the rows that do
///   not matter (masked by a null parent, or of an inactive union variant,
///   which the decoder decodes as nulls) the key of any value, so that they
///   do not need a dictionary entry of their own;
/// * the child fields of run-end encoded arrays, for which it uses Arrow's
///   default fields, and their run end types (see [`RowCodec`], they are
///   decoded as `Int64`): the run ends must fit their type;
/// * the types of the children of unions, which keep their declared type.
///
/// Everything else (the physical layout and the buffers) is kept.
fn restore(
    data: ArrayData,
    data_type: &DataType,
    nullable: bool,
    relevant: Option<&BooleanBuffer>,
) -> Result<ArrayData> {
    if has_type(&data, data_type) {
        return Ok(data);
    }
    if let (DataType::Dictionary(..), false) = (
        data.data_type(),
        matches!(data_type, DataType::Dictionary(..)),
    ) {
        // E.g. a child of a dense union without rows, which the decoder
        // creates with the type of its field.
        let dictionary = make_array(data);
        let dictionary = dictionary.as_any_dictionary();
        let values = take(dictionary.values().as_ref(), dictionary.keys(), None)?;
        return restore(values.to_data(), data_type, nullable, relevant);
    }
    if let DataType::Dictionary(key_type, value_type) = data_type {
        let values = make_array(restore(data, value_type, true, relevant)?);
        let (key_nulls, any_key) = if nullable {
            let valid = match (values.logical_nulls(), relevant) {
                (None, None) => None,
                (Some(nulls), None) => Some(nulls.into_inner()),
                (None, Some(relevant)) => Some(relevant.clone()),
                (Some(nulls), Some(relevant)) => Some(nulls.inner() & relevant),
            };
            (valid.map(NullBuffer::new), None)
        } else {
            (None, relevant.map(|relevant| !relevant))
        };
        return Ok(dictionary_encode(
            &values,
            key_nulls,
            any_key.as_ref(),
            key_type,
            data_type,
        )?
        .to_data());
    }
    let child_types = child_data_types(data_type);
    if !same_shape(data.data_type(), data_type)
        || child_types.len() != data.child_data().len()
    {
        return internal_err!(
            "Cannot restore the data type {data_type} of an array of type {}",
            data.data_type()
        );
    }
    let run_end_encoded = matches!(data_type, DataType::RunEndEncoded(..));
    let children = data
        .child_data()
        .iter()
        .zip(child_types)
        .zip(relevant_child_rows(&data, relevant))
        .enumerate()
        .map(|(i, ((child, (data_type, nullable)), relevant))| {
            if run_end_encoded && i == 0 {
                narrow_run_ends(child, data_type)
            } else {
                restore(child.clone(), data_type, nullable, relevant.as_ref())
            }
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(data
        .into_builder()
        .data_type(data_type.clone())
        .child_data(children)
        .build()?)
}

/// The run ends `run_ends` of a run-end encoded array (decoded as `Int64`)
/// as `data_type`, if they fit it.
fn narrow_run_ends(run_ends: &ArrayData, data_type: &DataType) -> Result<ArrayData> {
    if run_ends.data_type() == data_type {
        return Ok(run_ends.clone());
    }
    let run_ends = make_array(run_ends.clone());
    let narrowed = cast_with_options(&run_ends, data_type, &strict_cast_options())
        .map_err(|e| {
            exec_datafusion_err!(
                "Too many run-end encoded values for run ends of type {data_type}: {e}"
            )
        })?;
    Ok(narrowed.to_data())
}

/// The rows of the children of `data` that matter (`None`: all rows), if the
/// rows of `data` in `relevant` matter (`None`: all rows): the children of
/// null rows do not matter, nor the rows of the children of sparse unions
/// that belong to another variant.
fn relevant_child_rows(
    data: &ArrayData,
    relevant: Option<&BooleanBuffer>,
) -> Vec<Option<BooleanBuffer>> {
    let children = data.child_data();
    let (offset, len) = (data.offset(), data.len());
    let is_relevant = |i: usize| relevant.is_none_or(|r| r.value(i));
    // Whether the children of row `i` of `data` matter.
    let children_matter = |i: usize| is_relevant(i) && data.is_valid(i);
    // Whether all children of all rows matter.
    let all = relevant.is_none() && data.null_count() == 0;
    // The child rows in the ranges of the rows whose children matter.
    let ranges = |child: &ArrayData, ranges: &mut dyn Iterator<Item = (usize, usize)>| {
        let mut rows = vec![false; child.len()];
        for (i, (start, end)) in ranges.enumerate() {
            if children_matter(i) {
                rows[start..end].iter_mut().for_each(|row| *row = true);
            }
        }
        Some(BooleanBuffer::from(rows))
    };
    match data.data_type() {
        DataType::Struct(_) if !all => children
            .iter()
            .map(|child| {
                Some(BooleanBuffer::collect_bool(child.len(), |j| {
                    j.checked_sub(offset)
                        .is_some_and(|i| i < len && children_matter(i))
                }))
            })
            .collect(),
        DataType::FixedSizeList(_, size) if !all => {
            let size = (*size).max(1) as usize;
            vec![Some(BooleanBuffer::collect_bool(children[0].len(), |j| {
                (j / size)
                    .checked_sub(offset)
                    .is_some_and(|i| i < len && children_matter(i))
            }))]
        }
        DataType::List(_) | DataType::Map(..) if !all => {
            let offsets = data.buffer::<i32>(0);
            let mut bounds = offsets.windows(2).map(|w| (w[0] as usize, w[1] as usize));
            vec![ranges(&children[0], &mut bounds)]
        }
        DataType::LargeList(_) if !all => {
            let offsets = data.buffer::<i64>(0);
            let mut bounds = offsets.windows(2).map(|w| (w[0] as usize, w[1] as usize));
            vec![ranges(&children[0], &mut bounds)]
        }
        DataType::ListView(_) if !all => {
            let (offsets, sizes) = (data.buffer::<i32>(0), data.buffer::<i32>(1));
            let mut bounds = offsets
                .iter()
                .zip(sizes)
                .take(len)
                .map(|(&o, &s)| (o as usize, (o + s) as usize));
            vec![ranges(&children[0], &mut bounds)]
        }
        DataType::LargeListView(_) if !all => {
            let (offsets, sizes) = (data.buffer::<i64>(0), data.buffer::<i64>(1));
            let mut bounds = offsets
                .iter()
                .zip(sizes)
                .take(len)
                .map(|(&o, &s)| (o as usize, (o + s) as usize));
            vec![ranges(&children[0], &mut bounds)]
        }
        DataType::Union(fields, UnionMode::Sparse) => {
            let type_ids = data.buffer::<i8>(0);
            fields
                .iter()
                .zip(children)
                .map(|((type_id, _), child)| {
                    Some(BooleanBuffer::collect_bool(child.len(), |j| {
                        j.checked_sub(offset).is_some_and(|i| {
                            i < len && is_relevant(i) && type_ids[i] == type_id
                        })
                    }))
                })
                .collect()
        }
        DataType::Union(fields, UnionMode::Dense) if relevant.is_some() => {
            let (type_ids, offsets) = (data.buffer::<i8>(0), data.buffer::<i32>(1));
            fields
                .iter()
                .zip(children)
                .map(|((type_id, _), child)| {
                    let mut rows = vec![false; child.len()];
                    for i in
                        (0..len).filter(|&i| is_relevant(i) && type_ids[i] == type_id)
                    {
                        if let Some(row) = rows.get_mut(offsets[i] as usize) {
                            *row = true;
                        }
                    }
                    Some(BooleanBuffer::from(rows))
                })
                .collect()
        }
        DataType::RunEndEncoded(..) if relevant.is_some() => {
            let run_ends: Vec<usize> = match children[0].data_type() {
                DataType::Int16 => children[0]
                    .buffer::<i16>(0)
                    .iter()
                    .map(|&e| e as usize)
                    .collect(),
                DataType::Int32 => children[0]
                    .buffer::<i32>(0)
                    .iter()
                    .map(|&e| e as usize)
                    .collect(),
                _ => children[0]
                    .buffer::<i64>(0)
                    .iter()
                    .map(|&e| e as usize)
                    .collect(),
            };
            // A run matters if one of its rows in the array matters.
            let mut start = 0;
            let runs = run_ends
                .iter()
                .map(|&end| {
                    let rows = start.max(offset)..end.min(offset + len);
                    start = end;
                    rows.into_iter().any(|row| is_relevant(row - offset))
                })
                .collect::<Vec<_>>();
            let mut values = vec![false; children[1].len()];
            values
                .iter_mut()
                .zip(runs)
                .for_each(|(value, run)| *value = run);
            vec![None, Some(BooleanBuffer::from(values))]
        }
        _ => vec![None; children.len()],
    }
}

/// Whether arrays of types `a` and `b` have the same physical layout, not
/// counting their children: the types are equal up to their children (and
/// the names, nullability and metadata of their fields).
fn same_shape(a: &DataType, b: &DataType) -> bool {
    match (a, b) {
        (DataType::List(_), DataType::List(_))
        | (DataType::LargeList(_), DataType::LargeList(_))
        | (DataType::ListView(_), DataType::ListView(_))
        | (DataType::LargeListView(_), DataType::LargeListView(_)) => true,
        (DataType::FixedSizeList(_, a), DataType::FixedSizeList(_, b)) => a == b,
        (DataType::Map(_, a), DataType::Map(_, b)) => a == b,
        (DataType::Struct(a), DataType::Struct(b)) => a.len() == b.len(),
        (DataType::Union(a, a_mode), DataType::Union(b, b_mode)) => {
            a_mode == b_mode && a.iter().map(|(id, _)| id).eq(b.iter().map(|(id, _)| id))
        }
        // The run ends are decoded as `Int64`, see `restore`.
        (DataType::RunEndEncoded(..), DataType::RunEndEncoded(..)) => true,
        (a, b) => a == b,
    }
}

/// Dictionary encode `values` as an array of type `data_type`
/// (`Dictionary(key_type, values.data_type())`): the dictionary holds the
/// distinct values in order of first occurrence (distinct by their bits,
/// so `-0.0` and `0.0` stay distinct), including null values that have a
/// valid key. The rows in `key_nulls` have a null key, and the rows in
/// `any_key` (which do not matter) the valid key of any value: of the first
/// value, which is the value of the first such row if there is no other.
pub(crate) fn dictionary_encode(
    values: &ArrayRef,
    key_nulls: Option<NullBuffer>,
    any_key: Option<&BooleanBuffer>,
    key_type: &DataType,
    data_type: &DataType,
) -> Result<ArrayRef> {
    let key_nulls = key_nulls.filter(|n| n.null_count() > 0);
    let converter = RowConverter::new(vec![SortField::new(values.data_type().clone())])?;
    let rows = converter.convert_columns(&[Arc::clone(values)])?;
    let mut keys_of_rows =
        HashMap::with_capacity_and_hasher(values.len(), RandomState::default());
    let mut distinct = Vec::new();
    let mut any = None;
    let keys = (0..values.len())
        .map(|i| {
            if key_nulls.as_ref().is_some_and(|n| n.is_null(i)) {
                return 0;
            }
            if any_key.is_some_and(|rows| rows.value(i)) {
                any.get_or_insert(i);
                return 0;
            }
            *keys_of_rows.entry(rows.row(i)).or_insert_with(|| {
                distinct.push(i as u32);
                (distinct.len() - 1) as u64
            })
        })
        .collect::<Vec<u64>>();
    if let (Some(row), true) = (any, distinct.is_empty()) {
        distinct.push(row as u32);
    }
    let dictionary = take(values.as_ref(), &UInt32Array::from(distinct), None)?;
    new_dictionary(keys, key_nulls, &dictionary, key_type, data_type)
}

/// The dictionary array of type `data_type` (`Dictionary(key_type, _)`)
/// with the keys `keys` (null in `key_nulls`) of the values `dictionary`.
pub(crate) fn new_dictionary(
    keys: Vec<u64>,
    key_nulls: Option<NullBuffer>,
    dictionary: &ArrayRef,
    key_type: &DataType,
    data_type: &DataType,
) -> Result<ArrayRef> {
    let keys: ArrayRef = Arc::new(UInt64Array::new(ScalarBuffer::from(keys), key_nulls));
    // Fails if the keys do not fit `key_type`.
    let keys =
        cast_with_options(&keys, key_type, &strict_cast_options()).map_err(|e| {
            exec_datafusion_err!(
                "Cannot dictionary encode {} distinct values with keys of type {key_type}: {e}",
                dictionary.len()
            )
        })?;
    let data = keys
        .to_data()
        .into_builder()
        .data_type(data_type.clone())
        .child_data(vec![dictionary.to_data()])
        .build()?;
    Ok(make_array(data))
}

/// `data_type` with every dictionary replaced by its value type.
fn without_dictionaries(data_type: &DataType) -> DataType {
    match data_type {
        DataType::Dictionary(_, value_type) => without_dictionaries(value_type),
        data_type => map_children(data_type, &without_dictionaries),
    }
}

/// `data_type` with `Int64` run ends for every run-end encoded array (at any
/// depth). The row format decodes the values of all rows of a nested type
/// into one array (e.g. the values of all lists), whose run ends could
/// overflow a narrower type although the run ends of every row fit it; they
/// are narrowed when the values are restored, see [`restore`].
fn with_wide_run_ends(data_type: &DataType) -> DataType {
    match data_type {
        DataType::RunEndEncoded(run_ends, values) => DataType::RunEndEncoded(
            Arc::new(run_ends.as_ref().clone().with_data_type(DataType::Int64)),
            Arc::new(
                values
                    .as_ref()
                    .clone()
                    .with_data_type(with_wide_run_ends(values.data_type())),
            ),
        ),
        data_type => map_children(data_type, &with_wide_run_ends),
    }
}

/// `data_type` with the data types of its children (of the fields of a
/// nested type, or the value type of a dictionary) mapped by `f`.
fn map_children(data_type: &DataType, f: &impl Fn(&DataType) -> DataType) -> DataType {
    let field = |field: &FieldRef| -> FieldRef {
        Arc::new(field.as_ref().clone().with_data_type(f(field.data_type())))
    };
    match data_type {
        DataType::List(f) => DataType::List(field(f)),
        DataType::LargeList(f) => DataType::LargeList(field(f)),
        DataType::ListView(f) => DataType::ListView(field(f)),
        DataType::LargeListView(f) => DataType::LargeListView(field(f)),
        DataType::FixedSizeList(f, size) => DataType::FixedSizeList(field(f), *size),
        DataType::Map(f, sorted) => DataType::Map(field(f), *sorted),
        DataType::Struct(fields) => DataType::Struct(fields.iter().map(field).collect()),
        DataType::Union(fields, mode) => DataType::Union(
            fields
                .iter()
                .map(|(type_id, f)| (type_id, field(f)))
                .collect(),
            *mode,
        ),
        DataType::RunEndEncoded(run_ends, values) => {
            DataType::RunEndEncoded(field(run_ends), field(values))
        }
        DataType::Dictionary(key_type, value_type) => {
            DataType::Dictionary(key_type.clone(), Box::new(f(value_type)))
        }
        data_type => data_type.clone(),
    }
}

/// Whether `data_type` is or contains (at any depth) a union without
/// fields, which cannot have any row: every row of a union has the type id
/// of one of its fields.
fn has_union_without_fields(data_type: &DataType) -> bool {
    match data_type {
        DataType::Union(fields, _) => {
            fields.is_empty()
                || fields
                    .iter()
                    .any(|(_, f)| has_union_without_fields(f.data_type()))
        }
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::ListView(field)
        | DataType::LargeListView(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => has_union_without_fields(field.data_type()),
        DataType::Struct(fields) => fields
            .iter()
            .any(|f| has_union_without_fields(f.data_type())),
        DataType::RunEndEncoded(_, values) => {
            has_union_without_fields(values.data_type())
        }
        DataType::Dictionary(_, values) => has_union_without_fields(values),
        _ => false,
    }
}

/// Whether `data_type` is or contains a run-end encoded array at any depth.
pub(crate) fn has_run_end_encoded(data_type: &DataType) -> bool {
    matches!(data_type, DataType::RunEndEncoded(..))
        || child_data_types(data_type)
            .into_iter()
            .any(|(child, _)| has_run_end_encoded(child))
}

/// Whether `data_type` contains, below its top level, a dictionary whose
/// values contain a run-end encoded array. Unlike the dictionary of a
/// column, which is gathered once per distinct value, the row format decodes
/// such a dictionary to one value per row, and the run ends of these values
/// can overflow (e.g. `Int16` run ends count at most 32,767 values) although
/// the dictionary can hold them.
fn has_nested_dictionary_of_run_end_encoded(data_type: &DataType) -> bool {
    child_data_types(data_type)
        .into_iter()
        .any(|(child, _)| match child {
            DataType::Dictionary(_, values) => has_run_end_encoded(values),
            child => has_nested_dictionary_of_run_end_encoded(child),
        })
}

/// Whether `data_type` has (at any depth) values that several rows can
/// share and that contain a run-end encoded array: the items of a list view
/// (whose views can overlap), or the children of a dense union (whose
/// offsets can repeat). The row format decodes the shared values once per
/// row, and their run ends can overflow (e.g. 2,048 views of the same 32
/// values with `Int16` run ends) although the shared values fit them.
fn has_shared_run_end_encoded(data_type: &DataType) -> bool {
    let shared = match data_type {
        DataType::ListView(item) | DataType::LargeListView(item) => {
            has_run_end_encoded(item.data_type())
        }
        DataType::Union(fields, UnionMode::Dense) => fields
            .iter()
            .any(|(_, field)| has_run_end_encoded(field.data_type())),
        _ => false,
    };
    shared
        || child_data_types(data_type)
            .into_iter()
            .any(|(child, _)| has_shared_run_end_encoded(child))
}

/// Whether `data_type` is a union or contains a union at any depth.
fn has_union(data_type: &DataType) -> bool {
    match data_type {
        DataType::Union(..) => true,
        DataType::List(field)
        | DataType::LargeList(field)
        | DataType::ListView(field)
        | DataType::LargeListView(field)
        | DataType::FixedSizeList(field, _)
        | DataType::Map(field, _) => has_union(field.data_type()),
        DataType::Struct(fields) => fields.iter().any(|f| has_union(f.data_type())),
        DataType::RunEndEncoded(_, values) => has_union(values.data_type()),
        DataType::Dictionary(_, values) => has_union(values),
        _ => false,
    }
}

/// The width in bytes of a value of a fixed-width type, `None` if the type
/// is not a fixed-width type.
fn fixed_width(data_type: &DataType) -> Option<usize> {
    Some(match data_type {
        DataType::Int8 | DataType::UInt8 => 1,
        DataType::Int16 | DataType::UInt16 | DataType::Float16 => 2,
        DataType::Int32
        | DataType::UInt32
        | DataType::Float32
        | DataType::Date32
        | DataType::Time32(_)
        | DataType::Decimal32(..)
        | DataType::Interval(IntervalUnit::YearMonth) => 4,
        DataType::Int64
        | DataType::UInt64
        | DataType::Float64
        | DataType::Date64
        | DataType::Time64(_)
        | DataType::Timestamp(..)
        | DataType::Duration(_)
        | DataType::Decimal64(..)
        | DataType::Interval(IntervalUnit::DayTime) => 8,
        DataType::Decimal128(..) | DataType::Interval(IntervalUnit::MonthDayNano) => 16,
        DataType::Decimal256(..) => 32,
        DataType::FixedSizeBinary(width) => usize::try_from(*width).ok()?,
        _ => return None,
    })
}

/// The layout of the rows of a
/// [`TupleDataCollection`](super::TupleDataCollection): the offset and
/// storage of every column, and the width of a row. See the
/// [module documentation](self).
#[derive(Debug)]
pub struct TupleDataLayout {
    schema: SchemaRef,
    columns: Vec<ColumnLayout>,
    validity_bytes: usize,
    /// The validity bytes of a row in which all values are valid.
    all_valid: Vec<u8>,
    hash_offset: Option<usize>,
    row_width: usize,
    rows_per_block: usize,
}

impl TupleDataLayout {
    /// Create the layout of rows with the columns of `schema`, plus a `u64`
    /// hash slot if `with_hash` is `true`.
    ///
    /// The schema is kept as is (including field names, nullability and
    /// metadata): it is the schema of the batches gathered from collections
    /// with this layout.
    ///
    /// Returns a `NotImplemented` error if a data type is not supported, see
    /// [`Self::supports_data_type`].
    pub fn try_new(schema: SchemaRef, with_hash: bool) -> Result<Self> {
        let num_columns = schema.fields().len();
        // One validity bit per column, and one more for the keys of every
        // dictionary column.
        let num_dictionaries = schema
            .fields()
            .iter()
            .filter(|f| matches!(f.data_type(), DataType::Dictionary(..)))
            .count();
        let validity_bits = num_columns + num_dictionaries;
        let validity_bytes = validity_bits.div_ceil(8);
        let mut all_valid = vec![0xFF_u8; validity_bytes];
        if !validity_bits.is_multiple_of(8)
            && let Some(last) = all_valid.last_mut()
        {
            *last = (1_u8 << (validity_bits % 8)) - 1;
        }
        let mut row_width = validity_bytes;
        let mut columns = Vec::with_capacity(num_columns);
        let mut key_validity_bits = num_columns..;
        for field in schema.fields() {
            let mut column = ColumnLayout::try_new(field.data_type(), row_width)?;
            if column.is_dictionary() {
                column.key_validity_bit = key_validity_bits.next();
            }
            row_width = row_width.checked_add(column.width()).ok_or_else(|| {
                internal_datafusion_err!("TupleDataLayout row width overflow")
            })?;
            columns.push(column);
        }
        let hash_offset = with_hash.then_some(row_width);
        if with_hash {
            row_width += HASH_WIDTH;
        }
        let rows_per_block = match row_width {
            0 => MAX_ROWS_PER_BLOCK,
            width => (DEFAULT_BLOCK_SIZE / width).clamp(1, MAX_ROWS_PER_BLOCK),
        };
        Ok(Self {
            schema,
            columns,
            validity_bytes,
            all_valid,
            hash_offset,
            row_width,
            rows_per_block,
        })
    }

    /// Create the layout of rows with columns of the given data types (all
    /// nullable, named `c0`, `c1`, ...), plus a `u64` hash slot if
    /// `with_hash` is `true`.
    pub fn try_new_from_types(
        data_types: impl IntoIterator<Item = DataType>,
        with_hash: bool,
    ) -> Result<Self> {
        let fields = data_types
            .into_iter()
            .enumerate()
            .map(|(i, data_type)| Field::new(format!("c{i}"), data_type, true))
            .collect::<Vec<_>>();
        Self::try_new(Arc::new(Schema::new(fields)), with_hash)
    }

    /// Set the maximum number of rows in a block of the collections with
    /// this layout (by default, as many rows as fit in
    /// [`DEFAULT_BLOCK_SIZE`] bytes).
    ///
    /// `rows_per_block` must be in `1..=MAX_ROWS_PER_BLOCK`.
    pub fn with_rows_per_block(mut self, rows_per_block: usize) -> Result<Self> {
        if !(1..=MAX_ROWS_PER_BLOCK).contains(&rows_per_block) {
            return internal_err!(
                "rows_per_block must be in 1..={MAX_ROWS_PER_BLOCK}, got {rows_per_block}"
            );
        }
        self.rows_per_block = rows_per_block;
        Ok(self)
    }

    /// Returns `true` if columns of type `data_type` can be stored in a
    /// [`TupleDataCollection`](super::TupleDataCollection).
    ///
    /// All types are supported except dictionaries of dictionaries,
    /// dictionaries of run-end encoded values inside other types, and
    /// run-end encoded values in list views or dense unions (the row format
    /// decodes these values once per row, whose run ends can overflow
    /// although the shared values fit them), types with a union without
    /// fields (at any depth: such a union cannot have rows), and nested types
    /// whose values the Arrow row format cannot encode or restore exactly.
    pub fn supports_data_type(data_type: &DataType) -> bool {
        ColumnLayout::try_new(data_type, 0).is_ok()
    }

    /// The schema of the rows.
    pub fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    /// The number of columns.
    pub fn num_columns(&self) -> usize {
        self.columns.len()
    }

    /// The layouts of the columns.
    pub fn columns(&self) -> &[ColumnLayout] {
        &self.columns
    }

    /// The layout of column `i`.
    ///
    /// # Panics
    ///
    /// Panics if `i >= self.num_columns()`.
    pub fn column(&self, i: usize) -> &ColumnLayout {
        &self.columns[i]
    }

    /// The number of validity bytes at the start of a row.
    pub fn validity_bytes(&self) -> usize {
        self.validity_bytes
    }

    /// The validity bytes of a row in which all values are valid.
    pub(crate) fn all_valid(&self) -> &[u8] {
        &self.all_valid
    }

    /// The byte offset of the hash slot in a row, if the rows have one.
    pub fn hash_offset(&self) -> Option<usize> {
        self.hash_offset
    }

    /// Returns `true` if the rows have a hash slot.
    pub fn has_hash(&self) -> bool {
        self.hash_offset.is_some()
    }

    /// The width in bytes of a row.
    pub fn row_width(&self) -> usize {
        self.row_width
    }

    /// The maximum number of rows in a block.
    pub fn rows_per_block(&self) -> usize {
        self.rows_per_block
    }

    /// Returns `true` if values of some column can be stored in the heap.
    pub fn has_heap(&self) -> bool {
        self.columns.iter().any(|c| c.storage.uses_heap())
    }

    /// Returns `true` if rows with this layout and rows with the `other`
    /// layout are stored identically, i.e. the columns have the same data
    /// types and both or none have a hash slot. Field names and metadata
    /// are not compared.
    pub fn is_compatible(&self, other: &TupleDataLayout) -> bool {
        std::ptr::eq(self, other)
            || (self.has_hash() == other.has_hash()
                && self.columns.len() == other.columns.len()
                && self
                    .columns
                    .iter()
                    .zip(&other.columns)
                    .all(|(a, b)| a.data_type == b.data_type))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::datatypes::{Fields, TimeUnit, UnionFields, UnionMode};

    fn layout(types: Vec<DataType>, with_hash: bool) -> TupleDataLayout {
        TupleDataLayout::try_new_from_types(types, with_hash).unwrap()
    }

    #[test]
    fn offsets_and_widths() {
        let layout = layout(
            vec![
                DataType::Boolean,
                DataType::Int32,
                DataType::Utf8,
                DataType::Null,
                DataType::Decimal128(20, 3),
                DataType::Timestamp(TimeUnit::Nanosecond, Some("UTC".into())),
                DataType::FixedSizeBinary(3),
                DataType::Float16,
                DataType::Interval(IntervalUnit::MonthDayNano),
            ],
            true,
        );
        // 9 columns: 2 validity bytes.
        assert_eq!(layout.validity_bytes(), 2);
        assert_eq!(layout.all_valid(), &[0xFF, 0x01]);
        let storages = layout
            .columns()
            .iter()
            .map(|c| (c.offset(), c.storage()))
            .collect::<Vec<_>>();
        assert_eq!(
            storages,
            vec![
                (2, ColumnStorage::Boolean),
                (3, ColumnStorage::Fixed(4)),
                (7, ColumnStorage::VarLen),
                (23, ColumnStorage::Null),
                (23, ColumnStorage::Fixed(16)),
                (39, ColumnStorage::Fixed(8)),
                (47, ColumnStorage::Fixed(3)),
                (50, ColumnStorage::Fixed(2)),
                (52, ColumnStorage::Fixed(16)),
            ]
        );
        assert_eq!(layout.hash_offset(), Some(68));
        assert!(layout.has_hash());
        assert_eq!(layout.row_width(), 76);
        assert_eq!(layout.rows_per_block(), DEFAULT_BLOCK_SIZE / 76);
        assert!(layout.has_heap());
        assert_eq!(layout.num_columns(), 9);
        assert_eq!(layout.column(2).width(), VAR_LEN_SLOT_WIDTH);
    }

    #[test]
    fn layout_without_columns_or_hash() {
        let empty = layout(vec![], false);
        assert_eq!(empty.validity_bytes(), 0);
        assert_eq!(empty.row_width(), 0);
        assert_eq!(empty.rows_per_block(), MAX_ROWS_PER_BLOCK);
        assert!(!empty.has_heap());
        assert!(!empty.has_hash());

        let hash_only = layout(vec![], true);
        assert_eq!(hash_only.hash_offset(), Some(0));
        assert_eq!(hash_only.row_width(), HASH_WIDTH);

        let eight = layout(vec![DataType::Int8; 8], false);
        assert_eq!(eight.validity_bytes(), 1);
        assert_eq!(eight.all_valid(), &[0xFF]);
        assert_eq!(eight.row_width(), 9);
        assert!(!eight.has_heap());
    }

    #[test]
    fn dictionaries_and_nested_types() {
        let dictionary =
            DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Float64));
        let list = DataType::new_list(DataType::Int32, true);
        let list_of_dictionaries = DataType::new_list(
            DataType::Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            true,
        );
        let layout = layout(
            vec![
                dictionary.clone(),
                list.clone(),
                list_of_dictionaries.clone(),
            ],
            false,
        );

        let column = layout.column(0);
        assert!(column.is_dictionary());
        assert_eq!(column.data_type(), &dictionary);
        assert_eq!(column.value_type(), &DataType::Float64);
        assert_eq!(column.storage(), ColumnStorage::Fixed(8));
        assert!(column.codec().is_none());
        // The key validity bit follows the validity bits of the columns.
        assert_eq!(column.key_validity_bit(), Some(3));
        assert_eq!(layout.validity_bytes(), 1);
        assert_eq!(layout.all_valid(), &[0b1111]);
        let wide = TupleDataLayout::try_new_from_types(
            (0..8).map(|_| dictionary.clone()),
            false,
        )
        .unwrap();
        assert_eq!(wide.validity_bytes(), 2);
        assert_eq!(wide.all_valid(), &[0xFF, 0xFF]);
        assert_eq!(wide.column(7).key_validity_bit(), Some(15));

        let column = layout.column(1);
        assert_eq!(column.storage(), ColumnStorage::Encoded);
        let codec = column.codec().unwrap();
        assert!(!codec.compare_decoded);
        assert_eq!(column.key_validity_bit(), None);
        assert_eq!(layout.column(2).value_type(), &list_of_dictionaries);

        let floats =
            DataType::Struct(vec![Field::new("f", DataType::Float32, true)].into());
        let layout = TupleDataLayout::try_new_from_types([floats], false).unwrap();
        assert!(layout.column(0).codec().unwrap().compare_decoded);
    }

    #[test]
    fn supported_data_types() {
        let union = DataType::Union(
            UnionFields::try_new(
                [0, 1],
                [
                    Field::new("a", DataType::Int32, true),
                    Field::new("b", DataType::Utf8, true),
                ],
            )
            .unwrap(),
            UnionMode::Dense,
        );
        for data_type in [
            DataType::Null,
            DataType::Boolean,
            DataType::Decimal32(9, 2),
            DataType::Decimal256(76, 10),
            DataType::Interval(IntervalUnit::DayTime),
            DataType::Duration(TimeUnit::Millisecond),
            DataType::Utf8View,
            DataType::BinaryView,
            DataType::LargeBinary,
            DataType::new_large_list(DataType::Utf8, true),
            DataType::new_fixed_size_list(DataType::Float64, 2, true),
            DataType::Map(
                Arc::new(Field::new(
                    "entries",
                    DataType::Struct(
                        vec![
                            Field::new("key", DataType::Utf8, false),
                            Field::new("value", DataType::Int32, true),
                        ]
                        .into(),
                    ),
                    false,
                )),
                false,
            ),
            union,
            DataType::Dictionary(Box::new(DataType::UInt16), Box::new(DataType::Utf8)),
        ] {
            assert!(
                TupleDataLayout::supports_data_type(&data_type),
                "{data_type} should be supported"
            );
        }

        let nested_dictionary = DataType::Dictionary(
            Box::new(DataType::Int8),
            Box::new(DataType::Dictionary(
                Box::new(DataType::Int8),
                Box::new(DataType::Utf8),
            )),
        );
        assert!(!TupleDataLayout::supports_data_type(&nested_dictionary));
        let err =
            TupleDataLayout::try_new_from_types([nested_dictionary], false).unwrap_err();
        assert!(err.to_string().contains("nested dictionaries"), "{err}");
        assert!(!TupleDataLayout::supports_data_type(
            &DataType::FixedSizeBinary(-1)
        ));
    }

    #[test]
    fn unions_without_fields_are_not_supported() {
        let union = |fields: Vec<Field>, mode| {
            let type_ids = (0..fields.len() as i8).collect::<Vec<_>>();
            DataType::Union(UnionFields::try_new(type_ids, fields).unwrap(), mode)
        };
        for mode in [UnionMode::Sparse, UnionMode::Dense] {
            let empty = union(vec![], mode);
            let item = || Arc::new(Field::new("item", empty.clone(), true));
            let unsupported = [
                empty.clone(),
                // A union with an (inactive) union child without fields.
                union(
                    vec![
                        Field::new("a", DataType::Int32, true),
                        Field::new("b", empty.clone(), true),
                    ],
                    mode,
                ),
                DataType::List(item()),
                DataType::LargeListView(item()),
                DataType::FixedSizeList(item(), 2),
                DataType::FixedSizeList(item(), 0),
                DataType::Struct(vec![Field::new("u", empty.clone(), true)].into()),
                DataType::Map(
                    Arc::new(Field::new(
                        "entries",
                        DataType::Struct(
                            vec![
                                Field::new("k", DataType::Utf8, false),
                                Field::new("v", empty.clone(), true),
                            ]
                            .into(),
                        ),
                        false,
                    )),
                    false,
                ),
                DataType::RunEndEncoded(
                    Arc::new(Field::new("run_ends", DataType::Int32, false)),
                    item(),
                ),
                DataType::Dictionary(Box::new(DataType::Int8), Box::new(empty.clone())),
            ];
            for data_type in unsupported {
                // Neither panics.
                assert!(
                    !TupleDataLayout::supports_data_type(&data_type),
                    "{data_type}"
                );
                let err = TupleDataLayout::try_new_from_types([data_type.clone()], false)
                    .unwrap_err();
                assert!(
                    err.to_string().contains("unions without fields"),
                    "{data_type}: {err}"
                );
            }
            // Unions of nulls and of structs without fields are fine.
            for data_type in [
                union(vec![Field::new("n", DataType::Null, true)], mode),
                union(
                    vec![Field::new("s", DataType::Struct(Fields::empty()), true)],
                    mode,
                ),
            ] {
                assert!(
                    TupleDataLayout::supports_data_type(&data_type),
                    "{data_type}"
                );
            }
        }
    }

    #[test]
    fn nested_dictionaries_of_run_end_encoded_values_are_not_supported() {
        let run_end_encoded = DataType::RunEndEncoded(
            Arc::new(Field::new("run_ends", DataType::Int16, false)),
            Arc::new(Field::new("values", DataType::Int32, true)),
        );
        let dictionary = |values: DataType| {
            DataType::Dictionary(Box::new(DataType::Int8), Box::new(values))
        };
        let structs =
            |child: DataType| DataType::Struct(vec![Field::new("c", child, true)].into());
        let list = |item: DataType| DataType::new_list(item, true);
        let union = |child: DataType| {
            DataType::Union(
                UnionFields::try_new([0], [Field::new("c", child, true)]).unwrap(),
                UnionMode::Sparse,
            )
        };
        for data_type in [
            structs(dictionary(run_end_encoded.clone())),
            list(dictionary(run_end_encoded.clone())),
            union(dictionary(run_end_encoded.clone())),
            structs(dictionary(structs(run_end_encoded.clone()))),
            dictionary(structs(dictionary(run_end_encoded.clone()))),
        ] {
            assert!(
                !TupleDataLayout::supports_data_type(&data_type),
                "{data_type}"
            );
            let err = TupleDataLayout::try_new_from_types([data_type.clone()], false)
                .unwrap_err();
            assert!(
                err.to_string()
                    .contains("dictionaries of run-end encoded values"),
                "{data_type}: {err}"
            );
        }
        // Dictionaries of columns are gathered once per distinct value.
        for data_type in [
            dictionary(run_end_encoded.clone()),
            dictionary(structs(run_end_encoded.clone())),
            structs(run_end_encoded.clone()),
            list(dictionary(DataType::Utf8)),
        ] {
            assert!(
                TupleDataLayout::supports_data_type(&data_type),
                "{data_type}"
            );
        }
    }

    #[test]
    fn list_views_of_run_end_encoded_values_are_not_supported() {
        use arrow::array::{Float64Array, GenericListViewArray, Int16Array, RunArray};
        use arrow::buffer::ScalarBuffer;
        use arrow::datatypes::{Int16Type, Schema};

        // 2,048 views of the same 32 values (one run), more values than
        // `Int16` run ends count once the row format decodes every view.
        let runs = RunArray::<Int16Type>::try_new(
            &Int16Array::from(vec![32]),
            &Float64Array::from(vec![1.5]),
        )
        .unwrap();
        let item = Arc::new(Field::new("item", runs.data_type().clone(), true));
        let views = GenericListViewArray::<i32>::try_new(
            Arc::clone(&item),
            ScalarBuffer::from(vec![0; 2048]),
            ScalarBuffer::from(vec![32; 2048]),
            Arc::new(runs),
            None,
        )
        .unwrap();
        let schema = Schema::new(vec![Field::new("v", views.data_type().clone(), true)]);
        let err = TupleDataLayout::try_new(Arc::new(schema), false).unwrap_err();
        assert!(
            err.to_string().contains("in list views or dense unions"),
            "{err}"
        );

        let run_end_encoded = item.data_type().clone();
        let structs =
            |child: DataType| DataType::Struct(vec![Field::new("c", child, true)].into());
        let list_view =
            |item: DataType| DataType::ListView(Arc::new(Field::new("item", item, true)));
        let large_list_view = |item: DataType| {
            DataType::LargeListView(Arc::new(Field::new("item", item, true)))
        };
        let dictionary = |values: DataType| {
            DataType::Dictionary(Box::new(DataType::Int8), Box::new(values))
        };
        for data_type in [
            list_view(run_end_encoded.clone()),
            large_list_view(run_end_encoded.clone()),
            list_view(structs(run_end_encoded.clone())),
            list_view(DataType::new_list(run_end_encoded.clone(), true)),
            structs(list_view(run_end_encoded.clone())),
            DataType::new_list(large_list_view(run_end_encoded.clone()), true),
            dictionary(list_view(run_end_encoded.clone())),
        ] {
            assert!(
                !TupleDataLayout::supports_data_type(&data_type),
                "{data_type}"
            );
            let err = TupleDataLayout::try_new_from_types([data_type.clone()], false)
                .unwrap_err();
            assert!(
                err.to_string().contains("in list views or dense unions"),
                "{data_type}: {err}"
            );
        }
        for data_type in [
            DataType::new_list(run_end_encoded.clone(), true),
            list_view(DataType::Float64),
            list_view(structs(DataType::Float64)),
        ] {
            assert!(
                TupleDataLayout::supports_data_type(&data_type),
                "{data_type}"
            );
        }
    }

    #[test]
    fn dense_unions_of_run_end_encoded_values_are_not_supported() {
        use arrow::array::{
            FixedSizeListArray, Float64Array, Int16Array, ListArray, RunArray, UnionArray,
        };
        use arrow::buffer::OffsetBuffer;
        use arrow::datatypes::{Int16Type, Schema};

        // A list of 2,048 entries of a dense union that all refer to the same
        // 32 run-end encoded values, more values than `Int16` run ends count
        // once the row format decodes every entry.
        let runs = RunArray::<Int16Type>::try_new(
            &Int16Array::from(vec![32]),
            &Float64Array::from(vec![1.5]),
        )
        .unwrap();
        let run_end_encoded = runs.data_type().clone();
        let item = Arc::new(Field::new("item", run_end_encoded.clone(), true));
        let lists = FixedSizeListArray::try_new(item, 32, Arc::new(runs), None).unwrap();
        let union_fields = |child: DataType| {
            UnionFields::try_new([0], [Field::new("c", child, true)]).unwrap()
        };
        let union = UnionArray::try_new(
            union_fields(lists.data_type().clone()),
            vec![0; 2048].into(),
            Some(vec![0; 2048].into()),
            vec![Arc::new(lists.clone())],
        )
        .unwrap();
        let list = ListArray::new(
            Arc::new(Field::new("item", union.data_type().clone(), true)),
            OffsetBuffer::from_lengths([2048]),
            Arc::new(union),
            None,
        );
        assert!(!TupleDataLayout::supports_data_type(list.data_type()));
        let schema = Schema::new(vec![Field::new("l", list.data_type().clone(), true)]);
        let err = TupleDataLayout::try_new(Arc::new(schema), false).unwrap_err();
        assert!(
            err.to_string().contains("in list views or dense unions"),
            "{err}"
        );

        let dense =
            |child: DataType| DataType::Union(union_fields(child), UnionMode::Dense);
        let sparse =
            |child: DataType| DataType::Union(union_fields(child), UnionMode::Sparse);
        let structs =
            |child: DataType| DataType::Struct(vec![Field::new("c", child, true)].into());
        let runs_of = |values: DataType| {
            DataType::RunEndEncoded(
                Arc::new(Field::new("run_ends", DataType::Int16, false)),
                Arc::new(Field::new("values", values, true)),
            )
        };
        for data_type in [
            dense(run_end_encoded.clone()),
            dense(structs(run_end_encoded.clone())),
            structs(dense(DataType::new_list(run_end_encoded.clone(), true))),
            DataType::Dictionary(
                Box::new(DataType::Int8),
                Box::new(dense(run_end_encoded.clone())),
            ),
            runs_of(dense(run_end_encoded.clone())),
        ] {
            assert!(
                !TupleDataLayout::supports_data_type(&data_type),
                "{data_type}"
            );
            let err = TupleDataLayout::try_new_from_types([data_type.clone()], false)
                .unwrap_err();
            assert!(
                err.to_string().contains("in list views or dense unions"),
                "{data_type}: {err}"
            );
        }
        // Sparse unions do not share values, and a dense union without
        // run-end encoded values can.
        for data_type in [
            sparse(run_end_encoded.clone()),
            DataType::new_list(sparse(lists.data_type().clone()), true),
            dense(DataType::Float64),
            runs_of(dense(DataType::Float64)),
        ] {
            assert!(
                TupleDataLayout::supports_data_type(&data_type),
                "{data_type}"
            );
        }
    }

    #[test]
    fn rows_per_block_and_compatibility() {
        let a = layout(vec![DataType::Int64, DataType::Utf8], true);
        assert_eq!(a.rows_per_block(), DEFAULT_BLOCK_SIZE / (1 + 8 + 16 + 8));
        let a = a.with_rows_per_block(3).unwrap();
        assert_eq!(a.rows_per_block(), 3);
        assert!(
            layout(vec![DataType::Int64], false)
                .with_rows_per_block(0)
                .is_err()
        );
        assert!(
            layout(vec![DataType::Int64], false)
                .with_rows_per_block(MAX_ROWS_PER_BLOCK + 1)
                .is_err()
        );
        // A huge row still has one row per block.
        let huge = layout(vec![DataType::FixedSizeBinary(1 << 20)], false);
        assert_eq!(huge.rows_per_block(), 1);
        // Tiny rows are limited by the row index bits.
        let tiny = layout(vec![DataType::Boolean], false);
        assert_eq!(tiny.rows_per_block(), MAX_ROWS_PER_BLOCK);

        let b = TupleDataLayout::try_new(
            Arc::new(Schema::new(vec![
                Field::new("x", DataType::Int64, false),
                Field::new("y", DataType::Utf8, true),
            ])),
            true,
        )
        .unwrap();
        assert!(a.is_compatible(&b));
        assert!(a.is_compatible(&a));
        assert!(!a.is_compatible(&layout(vec![DataType::Int64, DataType::Utf8], false)));
        assert!(
            !a.is_compatible(&layout(vec![DataType::Int64, DataType::Utf8View], true))
        );
        assert!(!a.is_compatible(&layout(vec![DataType::Int64], true)));
    }
}
