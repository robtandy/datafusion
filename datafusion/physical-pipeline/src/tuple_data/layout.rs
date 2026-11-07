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
//! * **validity**: `ceil(n / 8)` bytes with one bit per column (bit `i % 8`
//!   of byte `i / 8`), set if the value of column `i` is valid (not null).
//!   The validity of a column is the *logical* validity of its Arrow array
//!   ([`Array::logical_nulls`](arrow::array::Array::logical_nulls)).
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
//! * **dictionary columns** are stored as their values: a
//!   `Dictionary(K, V)` column uses the storage of `V`, and is dictionary
//!   encoded again when it is gathered.
//! * the optional **hash** is a `u64` (little endian), e.g. the hash of the
//!   join or group keys of the row, computed once when the row is inserted.
//!
//! The layout is shared (`Arc<TupleDataLayout>`) by all collections that
//! store the same kind of rows, e.g. by all partitions of a
//! [`PartitionedTupleData`](super::PartitionedTupleData) and by all threads
//! of a sink.

use std::sync::Arc;

use arrow::compute::can_cast_types;
use arrow::datatypes::{
    DataType, Field, FieldRef, Fields, IntervalUnit, Schema, SchemaRef, UnionFields,
};
use arrow_row::{RowConverter, RowParser, SortField};
use datafusion_common::{Result, internal_datafusion_err, internal_err, not_impl_err};

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
    /// The row format codec of a [`ColumnStorage::Encoded`] column.
    codec: Option<RowCodec>,
}

impl ColumnLayout {
    fn try_new(data_type: &DataType, offset: usize) -> Result<Self> {
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

    /// The row format codec of a [`ColumnStorage::Encoded`] column.
    pub(crate) fn codec(&self) -> Option<&RowCodec> {
        self.codec.as_ref()
    }
}

/// Encodes and decodes the values of a [`ColumnStorage::Encoded`] column
/// with the Arrow row format.
#[derive(Debug)]
pub(crate) struct RowCodec {
    pub(crate) converter: RowConverter,
    pub(crate) parser: RowParser,
    /// Whether the decoded values must be cast to the value type of the
    /// column: the row format decodes dictionaries to their values.
    pub(crate) needs_cast: bool,
}

impl RowCodec {
    fn try_new(value_type: &DataType) -> Result<Self> {
        let fields = vec![SortField::new(value_type.clone())];
        if !RowConverter::supports_fields(&fields) {
            return not_impl_err!(
                "TupleDataLayout does not support the data type {value_type}"
            );
        }
        let decoded_type = without_dictionaries(value_type);
        let needs_cast = &decoded_type != value_type;
        if needs_cast && !can_cast_types(&decoded_type, value_type) {
            return not_impl_err!(
                "TupleDataLayout does not support the data type {value_type}: \
                 its dictionaries cannot be restored from {decoded_type}"
            );
        }
        let converter = RowConverter::new(fields)?;
        let parser = converter.parser();
        Ok(Self {
            converter,
            parser,
            needs_cast,
        })
    }
}

/// The data type the Arrow row format decodes values of `data_type` to:
/// `data_type` with every dictionary replaced by its value type.
fn without_dictionaries(data_type: &DataType) -> DataType {
    let field = |field: &FieldRef| -> FieldRef {
        Arc::new(
            field
                .as_ref()
                .clone()
                .with_data_type(without_dictionaries(field.data_type())),
        )
    };
    match data_type {
        DataType::Dictionary(_, value_type) => without_dictionaries(value_type),
        DataType::List(f) => DataType::List(field(f)),
        DataType::LargeList(f) => DataType::LargeList(field(f)),
        DataType::ListView(f) => DataType::ListView(field(f)),
        DataType::LargeListView(f) => DataType::LargeListView(field(f)),
        DataType::FixedSizeList(f, size) => DataType::FixedSizeList(field(f), *size),
        DataType::Map(f, sorted) => DataType::Map(field(f), *sorted),
        DataType::Struct(fields) => {
            DataType::Struct(fields.iter().map(field).collect::<Fields>())
        }
        DataType::Union(fields, mode) => DataType::Union(
            fields
                .iter()
                .map(|(type_id, f)| (type_id, field(f)))
                .collect::<UnionFields>(),
            *mode,
        ),
        DataType::RunEndEncoded(run_ends, values) => {
            DataType::RunEndEncoded(Arc::clone(run_ends), field(values))
        }
        data_type => data_type.clone(),
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
        let validity_bytes = num_columns.div_ceil(8);
        let mut all_valid = vec![0xFF_u8; validity_bytes];
        if !num_columns.is_multiple_of(8)
            && let Some(last) = all_valid.last_mut()
        {
            *last = (1_u8 << (num_columns % 8)) - 1;
        }
        let mut row_width = validity_bytes;
        let mut columns = Vec::with_capacity(num_columns);
        for field in schema.fields() {
            let column = ColumnLayout::try_new(field.data_type(), row_width)?;
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
    /// All types are supported except nested dictionaries, dictionaries
    /// inside nested types that Arrow cannot cast to (e.g. inside unions),
    /// and types that the Arrow row format does not support.
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

    use arrow::datatypes::{TimeUnit, UnionMode};

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

        let column = layout.column(1);
        assert_eq!(column.storage(), ColumnStorage::Encoded);
        let codec = column.codec().unwrap();
        assert!(!codec.needs_cast);

        let codec = layout.column(2).codec().unwrap();
        assert!(codec.needs_cast);
        assert_eq!(
            without_dictionaries(&list_of_dictionaries),
            DataType::new_list(DataType::Utf8, true)
        );
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
