// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Row-major tuple storage backing the stateful sinks (hash join, hash
//! aggregate, sort), modeled after DuckDB's `TupleDataCollection`.
//!
//! * [`TupleDataLayout`] ([`layout`]): the fixed-width row layout for a
//!   schema: validity bytes, inline fixed-width values, 16 byte slots for
//!   variable-length and nested values (inline or in a heap), and an
//!   optional precomputed `u64` hash.
//! * [`TupleDataCollection`] ([`collection`]): block-allocated rows plus
//!   per-block heaps, with a vectorized scatter
//!   ([`TupleDataCollection::append_chunk`], which respects the
//!   [`SelectionVector`](crate::chunk::SelectionVector) of a chunk without
//!   compacting it), a vectorized gather
//!   ([`TupleDataCollection::gather_columns`],
//!   [`TupleDataCollection::scan_chunk`]) that reconstructs Arrow arrays
//!   with exactly the original data types and values, and an O(1) per block
//!   [`TupleDataCollection::combine`] for `Sink::combine`. Every row is
//!   addressed by a 48-bit [`RowLocation`].
//! * [`PartitionedTupleData`] ([`partitioned`]): `2^radix_bits`
//!   collections, one per radix partition of the row hashes (the upper
//!   `radix_bits` bits of the hash), so that thread-local partitions can be
//!   merged into global partitions without copying and each partition can
//!   be finalized independently.
//! * [`RowMatcher`] ([`row_matcher`]): vectorized comparison of the keys of
//!   probe rows with the keys of candidate rows, with DataFusion's join key
//!   equality (`=` or `IS NOT DISTINCT FROM`, `-0.0 = +0.0`).
//!
//! ```text
//!                  append_chunk(chunk, hashes)
//!   DataChunk     ---------------------------->   TupleDataCollection
//!   (+ selection)        scatter rows             +---------------------+
//!                                                 | block 0: rows, heap |
//!   RecordBatch   <----------------------------   | block 1: rows, heap |
//!                 gather_columns(locations, ids)  | ...                 |
//!                                                 +---------------------+
//! ```

pub mod collection;
pub mod layout;
pub mod partitioned;
pub mod row_matcher;

pub use collection::{RowLocation, TupleDataCollection, TupleDataScanState};
pub use layout::{ColumnLayout, ColumnStorage, TupleDataLayout};
pub use partitioned::{PartitionedRowLocation, PartitionedTupleData};
pub use row_matcher::{PreparedProbe, RowMatcher};

/// Test data shared by the tests of the tuple data modules.
#[cfg(test)]
pub(crate) mod test_data {
    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow::array::{
        Array, ArrayRef, BinaryArray, BinaryViewArray, BooleanArray, Date32Array,
        Date64Array, Decimal32Array, Decimal64Array, Decimal128Array, Decimal256Array,
        DictionaryArray, DurationMicrosecondArray, DurationMillisecondArray,
        DurationNanosecondArray, DurationSecondArray, FixedSizeBinaryArray,
        FixedSizeListArray, Float32Array, Float64Array, Int8Array, Int16Array,
        Int32Array, Int32Builder, Int64Array, IntervalDayTimeArray,
        IntervalMonthDayNanoArray, IntervalYearMonthArray, LargeBinaryArray,
        LargeListBuilder, LargeStringArray, ListArray, ListBuilder, ListViewArray,
        MapBuilder, NullArray, RecordBatch, RunArray, StringArray, StringBuilder,
        StringDictionaryBuilder, StringViewArray, StructArray, Time32MillisecondArray,
        Time32SecondArray, Time64MicrosecondArray, Time64NanosecondArray,
        TimestampMicrosecondArray, TimestampMillisecondArray, TimestampNanosecondArray,
        TimestampSecondArray, UInt8Array, UInt16Array, UInt32Array, UInt64Array,
        UnionArray,
    };
    use arrow::buffer::{NullBuffer, ScalarBuffer};
    use arrow::compute::cast;
    use arrow::datatypes::{
        DataType, Field, Fields, Int8Type, Int16Type, Int32Type, IntervalDayTime,
        IntervalMonthDayNano, Schema, UInt16Type, UnionFields, UnionMode, i256,
    };

    use arrow_row::{RowConverter, SortField};

    use crate::chunk::{DataChunk, SelectionVector};

    /// Special float values: zeros of both signs, NaNs with different signs
    /// and payloads, infinities and extremes.
    pub(crate) fn special_f64(i: usize) -> f64 {
        const VALUES: [f64; 12] = [
            0.0,
            -0.0,
            1.5,
            -2.25,
            f64::NAN,
            f64::INFINITY,
            f64::NEG_INFINITY,
            f64::MIN_POSITIVE,
            f64::MAX,
            f64::MIN,
            1e-300,
            123_456.789,
        ];
        match i % 14 {
            12 => -f64::NAN,
            13 => f64::from_bits(f64::NAN.to_bits() | 0xBEEF),
            j => VALUES[j],
        }
    }

    fn string(i: usize) -> String {
        // Lengths 0 to 40, i.e. inline and heap values.
        let len = (i * 7) % 41;
        (0..len)
            .map(|j| char::from(b'a' + ((i + j) % 26) as u8))
            .collect()
    }

    /// `Some(value(i))` for most rows, `None` every `null_every` rows.
    fn opt<T>(i: usize, null_every: usize, value: impl Fn(usize) -> T) -> Option<T> {
        (i % null_every != null_every - 1).then(|| value(i))
    }

    fn metadata(key: &str) -> HashMap<String, String> {
        HashMap::from([(key.to_string(), format!("{key} value"))])
    }

    /// A batch of `n` rows with columns of (almost) all Arrow data types,
    /// with nulls, special float values, inline and heap strings, field and
    /// schema metadata.
    pub(crate) fn all_types_batch(n: usize) -> RecordBatch {
        let mut fields = vec![];
        let mut columns: Vec<ArrayRef> = vec![];
        let mut add = |field: Field, column: ArrayRef| {
            fields.push(field);
            columns.push(column);
        };
        let rows = 0..n;
        let int = |i: usize| i as i64 * 7_919 - 50_000;

        add(
            Field::new("bool", DataType::Boolean, true),
            Arc::new(
                rows.clone()
                    .map(|i| opt(i, 7, |i| i % 3 == 0))
                    .collect::<BooleanArray>(),
            ),
        );
        add(
            Field::new("i8", DataType::Int8, true),
            Arc::new(
                rows.clone()
                    .map(|i| opt(i, 5, |i| int(i) as i8))
                    .collect::<Int8Array>(),
            ),
        );
        add(
            Field::new("i16", DataType::Int16, false),
            Arc::new(rows.clone().map(|i| int(i) as i16).collect::<Int16Array>()),
        );
        add(
            Field::new("i32", DataType::Int32, true).with_metadata(metadata("i32")),
            Arc::new(
                rows.clone()
                    .map(|i| {
                        opt(i, 4, |i| if i % 9 == 0 { i32::MIN } else { int(i) as i32 })
                    })
                    .collect::<Int32Array>(),
            ),
        );
        add(
            Field::new("i64", DataType::Int64, true),
            Arc::new(
                rows.clone()
                    .map(|i| opt(i, 6, |i| if i % 11 == 0 { i64::MAX } else { int(i) }))
                    .collect::<Int64Array>(),
            ),
        );
        add(
            Field::new("u8", DataType::UInt8, true),
            Arc::new(
                rows.clone()
                    .map(|i| opt(i, 3, |i| i as u8))
                    .collect::<UInt8Array>(),
            ),
        );
        add(
            Field::new("u16", DataType::UInt16, true),
            Arc::new(
                rows.clone()
                    .map(|i| opt(i, 8, |i| (i * 300) as u16))
                    .collect::<UInt16Array>(),
            ),
        );
        add(
            Field::new("u32", DataType::UInt32, true),
            Arc::new(
                rows.clone()
                    .map(|i| opt(i, 9, |i| (i as u32).wrapping_mul(2_654_435_761)))
                    .collect::<UInt32Array>(),
            ),
        );
        add(
            Field::new("u64", DataType::UInt64, true),
            Arc::new(
                rows.clone()
                    .map(|i| opt(i, 10, |i| u64::MAX - i as u64))
                    .collect::<UInt64Array>(),
            ),
        );
        let f64s = rows
            .clone()
            .map(|i| opt(i, 15, special_f64))
            .collect::<Float64Array>();
        add(
            Field::new("f16", DataType::Float16, true),
            cast(&f64s, &DataType::Float16).unwrap(),
        );
        add(
            Field::new("f32", DataType::Float32, true),
            Arc::new(
                rows.clone()
                    .map(|i| {
                        opt(i, 13, |i| match i % 5 {
                            0 => f32::from_bits(f32::NAN.to_bits() | 3),
                            _ => special_f64(i) as f32,
                        })
                    })
                    .collect::<Float32Array>(),
            ),
        );
        add(Field::new("f64", DataType::Float64, true), Arc::new(f64s));
        add(
            Field::new("dec32", DataType::Decimal32(9, 2), true),
            Arc::new(
                rows.clone()
                    .map(|i| opt(i, 5, |i| (int(i) % 999_999_999) as i32))
                    .collect::<Decimal32Array>()
                    .with_precision_and_scale(9, 2)
                    .unwrap(),
            ),
        );
        add(
            Field::new("dec64", DataType::Decimal64(18, 3), true),
            Arc::new(
                rows.clone()
                    .map(|i| opt(i, 6, |i| int(i) * 1_000_003))
                    .collect::<Decimal64Array>()
                    .with_precision_and_scale(18, 3)
                    .unwrap(),
            ),
        );
        add(
            Field::new("dec128", DataType::Decimal128(38, 10), true),
            Arc::new(
                rows.clone()
                    .map(|i| opt(i, 7, |i| i128::from(int(i)) * 10_i128.pow(20) + 17))
                    .collect::<Decimal128Array>()
                    .with_precision_and_scale(38, 10)
                    .unwrap(),
            ),
        );
        add(
            Field::new("dec256", DataType::Decimal256(76, 5), true),
            Arc::new(
                rows.clone()
                    .map(|i| {
                        opt(i, 8, |i| {
                            i256::from_i128(i128::from(int(i)))
                                .wrapping_mul(i256::from_i128(10_i128.pow(30)))
                        })
                    })
                    .collect::<Decimal256Array>()
                    .with_precision_and_scale(76, 5)
                    .unwrap(),
            ),
        );
        add(
            Field::new("date32", DataType::Date32, true),
            Arc::new(
                rows.clone()
                    .map(|i| opt(i, 4, |i| int(i) as i32))
                    .collect::<Date32Array>(),
            ),
        );
        add(
            Field::new("date64", DataType::Date64, true),
            Arc::new(
                rows.clone()
                    .map(|i| opt(i, 5, |i| int(i) * 86_400_000))
                    .collect::<Date64Array>(),
            ),
        );
        add(
            Field::new(
                "time32s",
                DataType::Time32(arrow::datatypes::TimeUnit::Second),
                true,
            ),
            Arc::new(
                rows.clone()
                    .map(|i| opt(i, 6, |i| (i * 37 % 86_400) as i32))
                    .collect::<Time32SecondArray>(),
            ),
        );
        add(
            Field::new(
                "time32ms",
                DataType::Time32(arrow::datatypes::TimeUnit::Millisecond),
                true,
            ),
            Arc::new(
                rows.clone()
                    .map(|i| opt(i, 7, |i| (i * 3_701 % 86_400_000) as i32))
                    .collect::<Time32MillisecondArray>(),
            ),
        );
        add(
            Field::new(
                "time64us",
                DataType::Time64(arrow::datatypes::TimeUnit::Microsecond),
                true,
            ),
            Arc::new(
                rows.clone()
                    .map(|i| opt(i, 8, |i| i as i64 * 1_000_037))
                    .collect::<Time64MicrosecondArray>(),
            ),
        );
        add(
            Field::new(
                "time64ns",
                DataType::Time64(arrow::datatypes::TimeUnit::Nanosecond),
                true,
            ),
            Arc::new(
                rows.clone()
                    .map(|i| opt(i, 9, |i| i as i64 * 1_000_000_007))
                    .collect::<Time64NanosecondArray>(),
            ),
        );
        add(
            Field::new(
                "ts_s",
                DataType::Timestamp(arrow::datatypes::TimeUnit::Second, None),
                true,
            ),
            Arc::new(
                rows.clone()
                    .map(|i| opt(i, 5, int))
                    .collect::<TimestampSecondArray>(),
            ),
        );
        add(
            Field::new(
                "ts_ms",
                DataType::Timestamp(
                    arrow::datatypes::TimeUnit::Millisecond,
                    Some("UTC".into()),
                ),
                true,
            ),
            Arc::new(
                rows.clone()
                    .map(|i| opt(i, 6, |i| int(i) * 1_001))
                    .collect::<TimestampMillisecondArray>()
                    .with_timezone("UTC"),
            ),
        );
        add(
            Field::new(
                "ts_us",
                DataType::Timestamp(
                    arrow::datatypes::TimeUnit::Microsecond,
                    Some("+02:00".into()),
                ),
                true,
            ),
            Arc::new(
                rows.clone()
                    .map(|i| opt(i, 7, |i| int(i) * 1_000_001))
                    .collect::<TimestampMicrosecondArray>()
                    .with_timezone("+02:00"),
            ),
        );
        add(
            Field::new(
                "ts_ns",
                DataType::Timestamp(arrow::datatypes::TimeUnit::Nanosecond, None),
                true,
            ),
            Arc::new(
                rows.clone()
                    .map(|i| opt(i, 8, |i| int(i) * 1_000_000_009))
                    .collect::<TimestampNanosecondArray>(),
            ),
        );
        add(
            Field::new(
                "dur_s",
                DataType::Duration(arrow::datatypes::TimeUnit::Second),
                true,
            ),
            Arc::new(
                rows.clone()
                    .map(|i| opt(i, 5, int))
                    .collect::<DurationSecondArray>(),
            ),
        );
        add(
            Field::new(
                "dur_ms",
                DataType::Duration(arrow::datatypes::TimeUnit::Millisecond),
                true,
            ),
            Arc::new(
                rows.clone()
                    .map(|i| opt(i, 6, |i| -int(i)))
                    .collect::<DurationMillisecondArray>(),
            ),
        );
        add(
            Field::new(
                "dur_us",
                DataType::Duration(arrow::datatypes::TimeUnit::Microsecond),
                true,
            ),
            Arc::new(
                rows.clone()
                    .map(|i| opt(i, 7, |i| int(i) * 3))
                    .collect::<DurationMicrosecondArray>(),
            ),
        );
        add(
            Field::new(
                "dur_ns",
                DataType::Duration(arrow::datatypes::TimeUnit::Nanosecond),
                true,
            ),
            Arc::new(
                rows.clone()
                    .map(|i| opt(i, 8, |i| int(i) * 5))
                    .collect::<DurationNanosecondArray>(),
            ),
        );
        add(
            Field::new(
                "iv_ym",
                DataType::Interval(arrow::datatypes::IntervalUnit::YearMonth),
                true,
            ),
            Arc::new(
                rows.clone()
                    .map(|i| opt(i, 5, |i| int(i) as i32))
                    .collect::<IntervalYearMonthArray>(),
            ),
        );
        add(
            Field::new(
                "iv_dt",
                DataType::Interval(arrow::datatypes::IntervalUnit::DayTime),
                true,
            ),
            Arc::new(
                rows.clone()
                    .map(|i| {
                        opt(i, 6, |i| IntervalDayTime::new(i as i32, -(i as i32) * 1000))
                    })
                    .collect::<IntervalDayTimeArray>(),
            ),
        );
        add(
            Field::new(
                "iv_mdn",
                DataType::Interval(arrow::datatypes::IntervalUnit::MonthDayNano),
                true,
            ),
            Arc::new(
                rows.clone()
                    .map(|i| {
                        opt(i, 7, |i| {
                            IntervalMonthDayNano::new(
                                i as i32,
                                2 - i as i32,
                                int(i) * 1_000,
                            )
                        })
                    })
                    .collect::<IntervalMonthDayNanoArray>(),
            ),
        );
        add(
            Field::new("fsb3", DataType::FixedSizeBinary(3), true),
            Arc::new(
                FixedSizeBinaryArray::try_from_sparse_iter_with_size(
                    rows.clone()
                        .map(|i| opt(i, 4, |i| [i as u8, (i * 3) as u8, (i * 7) as u8])),
                    3,
                )
                .unwrap(),
            ),
        );
        let strings = rows
            .clone()
            .map(|i| opt(i, 6, string))
            .collect::<StringArray>();
        add(
            Field::new("utf8", DataType::Utf8, true).with_metadata(metadata("utf8")),
            Arc::new(strings.clone()),
        );
        add(
            Field::new("large_utf8", DataType::LargeUtf8, true),
            Arc::new(
                rows.clone()
                    .map(|i| opt(i, 5, |i| string(i + 3)))
                    .collect::<LargeStringArray>(),
            ),
        );
        add(
            Field::new("utf8_view", DataType::Utf8View, true),
            Arc::new(
                rows.clone()
                    .map(|i| opt(i, 7, |i| string(i + 5)))
                    .collect::<StringViewArray>(),
            ),
        );
        add(
            Field::new("binary", DataType::Binary, true),
            Arc::new(
                rows.clone()
                    .map(|i| opt(i, 4, |i| string(i + 1).into_bytes()))
                    .collect::<BinaryArray>(),
            ),
        );
        add(
            Field::new("large_binary", DataType::LargeBinary, true),
            Arc::new(
                rows.clone()
                    .map(|i| opt(i, 3, |i| string(i + 2).into_bytes()))
                    .collect::<LargeBinaryArray>(),
            ),
        );
        add(
            Field::new("binary_view", DataType::BinaryView, true),
            Arc::new(
                rows.clone()
                    .map(|i| opt(i, 8, |i| string(i + 9).into_bytes()))
                    .collect::<BinaryViewArray>(),
            ),
        );
        add(
            Field::new("null", DataType::Null, true),
            Arc::new(NullArray::new(n)),
        );
        add(
            Field::new(
                "dict_utf8",
                DataType::Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
                true,
            ),
            Arc::new(
                rows.clone()
                    .map(|i| opt(i, 5, |i| string(i % 6 + 2)))
                    .collect::<Vec<_>>()
                    .iter()
                    .map(Option::as_deref)
                    .collect::<DictionaryArray<Int8Type>>(),
            ),
        );
        let dict_values = Arc::new(Int64Array::from(vec![
            Some(7),
            None,
            Some(-1),
            Some(i64::MIN),
        ]));
        add(
            Field::new(
                "dict_i64",
                DataType::Dictionary(
                    Box::new(DataType::UInt16),
                    Box::new(DataType::Int64),
                ),
                true,
            ),
            Arc::new(
                DictionaryArray::<UInt16Type>::try_new(
                    rows.clone()
                        .map(|i| opt(i, 6, |i| (i % 4) as u16))
                        .collect(),
                    dict_values,
                )
                .unwrap(),
            ),
        );
        let list_values = ListArray::from_iter_primitive::<Int32Type, _, _>(vec![
            Some(vec![Some(1), None]),
            Some(vec![]),
            None,
        ]);
        add(
            Field::new(
                "dict_list",
                DataType::Dictionary(
                    Box::new(DataType::Int16),
                    Box::new(list_values.data_type().clone()),
                ),
                true,
            ),
            Arc::new(
                DictionaryArray::<Int16Type>::try_new(
                    rows.clone()
                        .map(|i| opt(i, 7, |i| (i % 3) as i16))
                        .collect(),
                    Arc::new(list_values),
                )
                .unwrap(),
            ),
        );
        let struct_fields = Fields::from(vec![
            Field::new("a", DataType::Int32, true).with_metadata(metadata("a")),
            Field::new("b", DataType::Utf8View, true),
            Field::new("c", DataType::Float64, false),
        ]);
        add(
            Field::new("struct", DataType::Struct(struct_fields.clone()), true),
            Arc::new(
                StructArray::try_new(
                    struct_fields,
                    vec![
                        Arc::new(
                            rows.clone()
                                .map(|i| opt(i, 3, |i| i as i32))
                                .collect::<Int32Array>(),
                        ),
                        Arc::new(
                            rows.clone()
                                .map(|i| opt(i, 4, |i| string(i + 7)))
                                .collect::<StringViewArray>(),
                        ),
                        Arc::new(rows.clone().map(special_f64).collect::<Float64Array>()),
                    ],
                    Some(NullBuffer::from(
                        rows.clone().map(|i| i % 5 != 2).collect::<Vec<_>>(),
                    )),
                )
                .unwrap(),
            ),
        );
        add(
            Field::new("list", DataType::new_list(DataType::Int32, true), true),
            Arc::new(ListArray::from_iter_primitive::<Int32Type, _, _>(
                rows.clone().map(|i| {
                    opt(i, 5, |i| {
                        (0..i % 4)
                            .map(|j| (j != 1).then_some((i * j) as i32))
                            .collect::<Vec<_>>()
                    })
                }),
            )),
        );
        let mut large_list = LargeListBuilder::new(StringBuilder::new());
        for i in rows.clone() {
            if i % 6 == 5 {
                large_list.append_null();
            } else {
                for j in 0..i % 3 {
                    large_list.values().append_value(string(i + j));
                }
                large_list.append(true);
            }
        }
        add(
            Field::new(
                "large_list",
                DataType::new_large_list(DataType::Utf8, true),
                true,
            ),
            Arc::new(large_list.finish()),
        );
        let item = Arc::new(Field::new_list_field(DataType::Float64, true));
        add(
            Field::new(
                "fixed_list",
                DataType::FixedSizeList(Arc::clone(&item), 2),
                true,
            ),
            Arc::new(
                FixedSizeListArray::try_new(
                    item,
                    2,
                    Arc::new(
                        (0..2 * n)
                            .map(|i| opt(i, 7, special_f64))
                            .collect::<Float64Array>(),
                    ),
                    Some(NullBuffer::from(
                        rows.clone().map(|i| i % 4 != 3).collect::<Vec<_>>(),
                    )),
                )
                .unwrap(),
            ),
        );
        let mut map = MapBuilder::new(None, StringBuilder::new(), Int32Builder::new());
        for i in rows.clone() {
            if i % 7 == 6 {
                map.append(false).unwrap();
            } else {
                for j in 0..i % 3 {
                    map.keys().append_value(format!("k{j}"));
                    map.values()
                        .append_option((j != 2).then_some((i + j) as i32));
                }
                map.append(true).unwrap();
            }
        }
        let map = map.finish();
        add(
            Field::new("map", map.data_type().clone(), true),
            Arc::new(map),
        );
        let mut list_dict = ListBuilder::new(StringDictionaryBuilder::<Int32Type>::new());
        for i in rows.clone() {
            if i % 5 == 4 {
                list_dict.append_null();
            } else {
                for j in 0..i % 3 {
                    list_dict.values().append_value(string(j + 1));
                }
                list_dict.append(true);
            }
        }
        let list_dict = list_dict.finish();
        add(
            Field::new("list_dict", list_dict.data_type().clone(), true),
            Arc::new(list_dict),
        );
        let list_view = ListArray::from_iter_primitive::<Int32Type, _, _>(
            rows.clone()
                .map(|i| opt(i, 4, |i| vec![Some(i as i32); i % 3])),
        );
        let list_view = ListViewArray::from(list_view);
        add(
            Field::new("list_view", list_view.data_type().clone(), true),
            Arc::new(list_view),
        );
        let union_fields = UnionFields::try_new(
            [3, 5],
            [
                Field::new("int", DataType::Int32, true),
                Field::new("str", DataType::Utf8, true),
            ],
        )
        .unwrap();
        let type_ids = rows
            .clone()
            .map(|i| if i % 3 == 0 { 5 } else { 3 })
            .collect::<ScalarBuffer<i8>>();
        add(
            Field::new(
                "sparse_union",
                DataType::Union(union_fields.clone(), UnionMode::Sparse),
                true,
            ),
            Arc::new(
                UnionArray::try_new(
                    union_fields.clone(),
                    type_ids.clone(),
                    None,
                    vec![
                        Arc::new(
                            rows.clone()
                                .map(|i| opt(i, 4, |i| i as i32))
                                .collect::<Int32Array>(),
                        ),
                        Arc::new(
                            rows.clone()
                                .map(|i| opt(i, 5, string))
                                .collect::<StringArray>(),
                        ),
                    ],
                )
                .unwrap(),
            ),
        );
        let dense_offsets = rows
            .clone()
            .map(|i| if i % 3 == 0 { i / 3 } else { i - i / 3 - 1 } as i32)
            .collect::<ScalarBuffer<i32>>();
        let num_str = n.div_ceil(3);
        add(
            Field::new(
                "dense_union",
                DataType::Union(union_fields.clone(), UnionMode::Dense),
                true,
            ),
            Arc::new(
                UnionArray::try_new(
                    union_fields,
                    type_ids,
                    Some(dense_offsets),
                    vec![
                        Arc::new(
                            (0..n - num_str)
                                .map(|i| opt(i, 3, |i| -(i as i32)))
                                .collect::<Int32Array>(),
                        ),
                        Arc::new(
                            (0..num_str)
                                .map(|i| opt(i, 4, |i| string(i + 11)))
                                .collect::<StringArray>(),
                        ),
                    ],
                )
                .unwrap(),
            ),
        );
        let run_ends = (1..=n.div_ceil(3))
            .map(|i| (i * 3).min(n) as i32)
            .collect::<Int32Array>();
        let run_values = (0..run_ends.len())
            .map(|i| opt(i, 3, string))
            .collect::<StringArray>();
        let ree = RunArray::<Int32Type>::try_new(&run_ends, &run_values).unwrap();
        add(
            Field::new("ree", ree.data_type().clone(), true),
            Arc::new(ree),
        );

        let schema = Schema::new_with_metadata(fields, metadata("schema"));
        RecordBatch::try_new(Arc::new(schema), columns).unwrap()
    }

    /// A run-end encoded array with custom child fields: other names,
    /// nullability and metadata than Arrow's default fields.
    pub(crate) fn custom_run_end_encoded(
        values: ArrayRef,
        run_ends: &[i32],
        nullable: bool,
    ) -> ArrayRef {
        let data_type = DataType::RunEndEncoded(
            Arc::new(
                Field::new("ends", DataType::Int32, false)
                    .with_metadata(metadata("ends")),
            ),
            Arc::new(
                Field::new("samples", values.data_type().clone(), nullable)
                    .with_metadata(metadata("samples")),
            ),
        );
        let data = arrow::array::ArrayData::builder(data_type)
            .len(run_ends.last().map_or(0, |&end| end as usize))
            .add_child_data(Int32Array::from(run_ends.to_vec()).to_data())
            .add_child_data(values.to_data())
            .build()
            .unwrap();
        arrow::array::make_array(data)
    }

    /// Arrays of rarely used types and type combinations.
    pub(crate) fn exotic_arrays() -> Vec<(&'static str, ArrayRef, bool)> {
        use arrow::array::{
            Int16Array, LargeListArray, LargeListViewArray, MapArray, StringArray,
        };
        use arrow::buffer::OffsetBuffer;
        use arrow::datatypes::UInt64Type;

        let views = [
            Some("a long string value!"),
            None,
            Some("b"),
            Some("a long string value!"),
        ]
        .into_iter()
        .collect::<DictionaryArray<Int8Type>>();
        let views = DictionaryArray::<Int8Type>::try_new(
            views.keys().clone(),
            cast(views.values(), &DataType::Utf8View).unwrap(),
        )
        .unwrap();
        let halves = cast(
            &Float32Array::from(vec![Some(-0.0), None, Some(0.0)]),
            &DataType::Float16,
        )
        .unwrap();
        let map_fields = Fields::from(vec![
            Field::new("k", DataType::Utf8, false),
            Field::new("v", DataType::Null, true),
        ]);
        let inner_lists = ListArray::new(
            Arc::new(Field::new_list_field(DataType::Utf8View, true)),
            OffsetBuffer::from_lengths([2, 0, 1, 2]),
            Arc::new(StringViewArray::from(vec![
                Some("a value longer than twelve"),
                None,
                Some("x"),
                Some("a value longer than twelve"),
                None,
            ])),
            None,
        );
        let mut arrays: Vec<(&'static str, ArrayRef, bool)> = vec![
            (
                "empty struct",
                Arc::new(StructArray::new_empty_fields(
                    4,
                    Some(NullBuffer::from(vec![true, false, true, false])),
                )),
                true,
            ),
            (
                "list of nulls",
                Arc::new(ListArray::new(
                    Arc::new(Field::new_list_field(DataType::Null, true)),
                    OffsetBuffer::from_lengths([2, 0, 2, 1]),
                    Arc::new(NullArray::new(5)),
                    Some(NullBuffer::from(vec![true, false, true, true])),
                )),
                true,
            ),
            (
                "fixed size list of nulls",
                Arc::new(FixedSizeListArray::new(
                    Arc::new(Field::new_list_field(DataType::Null, true)),
                    3,
                    Arc::new(NullArray::new(12)),
                    None,
                )),
                false,
            ),
            ("dictionary of views", Arc::new(views), true),
            (
                "dictionary with u64 keys of binaries",
                Arc::new(
                    DictionaryArray::<UInt64Type>::try_new(
                        UInt64Array::from(vec![Some(0), Some(1), None, Some(0)]),
                        Arc::new(BinaryArray::from(vec![Some(b"xy".as_ref()), None])),
                    )
                    .unwrap(),
                ),
                true,
            ),
            (
                "non-nullable dictionary of decimals",
                Arc::new(
                    DictionaryArray::<Int8Type>::try_new(
                        Int8Array::from(vec![0, 1, 0, 1]),
                        Arc::new(
                            Decimal256Array::from(vec![Some(i256::from_i128(5)), None])
                                .with_precision_and_scale(40, 3)
                                .unwrap(),
                        ),
                    )
                    .unwrap(),
                ),
                false,
            ),
            (
                "dictionary of half floats",
                Arc::new(
                    DictionaryArray::<Int8Type>::try_new(
                        Int8Array::from(vec![Some(0), Some(2), None, Some(1)]),
                        halves,
                    )
                    .unwrap(),
                ),
                true,
            ),
            (
                "non-nullable dictionary of nulls",
                Arc::new(
                    DictionaryArray::<Int8Type>::try_new(
                        Int8Array::from(vec![0, 0, 0, 0]),
                        Arc::new(NullArray::new(1)),
                    )
                    .unwrap(),
                ),
                false,
            ),
            (
                "large list view",
                Arc::new(LargeListViewArray::from(
                    LargeListArray::from_iter_primitive::<Int32Type, _, _>(vec![
                        Some(vec![Some(1), None]),
                        None,
                        Some(vec![]),
                        Some(vec![Some(1), None]),
                    ]),
                )),
                true,
            ),
            (
                "union with a single field",
                Arc::new(
                    UnionArray::try_new(
                        UnionFields::try_new(
                            [5],
                            [Field::new("only", DataType::Utf8, true)],
                        )
                        .unwrap(),
                        vec![5, 5, 5, 5].into(),
                        None,
                        vec![Arc::new(StringArray::from(vec![
                            Some("a"),
                            None,
                            Some("a"),
                            None,
                        ]))],
                    )
                    .unwrap(),
                ),
                false,
            ),
            (
                "run-end encoded floats with int16 run ends",
                Arc::new(
                    RunArray::<Int16Type>::try_new(
                        &Int16Array::from(vec![1, 2, 4]),
                        &Float64Array::from(vec![Some(-0.0), None, Some(0.0)]),
                    )
                    .unwrap(),
                ),
                true,
            ),
            (
                "sorted map of nulls",
                Arc::new(
                    MapArray::try_new(
                        Arc::new(Field::new(
                            "entries",
                            DataType::Struct(map_fields.clone()),
                            false,
                        )),
                        OffsetBuffer::from_lengths([1, 0, 1, 1]),
                        StructArray::try_new(
                            map_fields,
                            vec![
                                Arc::new(StringArray::from(vec!["a", "b", "a"])),
                                Arc::new(NullArray::new(3)),
                            ],
                            None,
                        )
                        .unwrap(),
                        Some(NullBuffer::from(vec![true, false, true, true])),
                        true,
                    )
                    .unwrap(),
                ),
                true,
            ),
            (
                "non-nullable lists of lists of views",
                Arc::new(ListArray::new(
                    Arc::new(Field::new("inner", inner_lists.data_type().clone(), false)),
                    OffsetBuffer::from_lengths([1, 1, 1, 1]),
                    Arc::new(inner_lists),
                    None,
                )),
                false,
            ),
        ];
        // Degenerate types: fixed size lists of size 0, structs without
        // fields in other types, and unions and run-end encoded arrays of
        // nulls.
        let item = |data_type: DataType| Arc::new(Field::new("item", data_type, true));
        let empty_structs = |nulls: Vec<bool>| -> ArrayRef {
            Arc::new(StructArray::new_empty_fields(
                nulls.len(),
                Some(NullBuffer::from(nulls)),
            ))
        };
        let empty_struct_type = DataType::Struct(Fields::empty());
        let union_fields = UnionFields::try_new(
            [0, 1],
            [
                Field::new("s", empty_struct_type.clone(), true),
                Field::new("n", DataType::Null, true),
            ],
        )
        .unwrap();
        arrays.extend([
            (
                "fixed size lists of size 0",
                Arc::new(
                    FixedSizeListArray::try_new_with_length(
                        item(DataType::Int32),
                        0,
                        Arc::new(Int32Array::from(Vec::<i32>::new())),
                        Some(NullBuffer::from(vec![true, false, true, true])),
                        4,
                    )
                    .unwrap(),
                ) as ArrayRef,
                true,
            ),
            (
                "lists of structs without fields",
                Arc::new(ListArray::new(
                    item(empty_struct_type.clone()),
                    OffsetBuffer::from_lengths([2, 0, 3, 1]),
                    empty_structs(vec![true, false, true, true, false, true]),
                    Some(NullBuffer::from(vec![true, false, true, true])),
                )),
                true,
            ),
            (
                "dictionary of structs without fields",
                Arc::new(
                    DictionaryArray::<Int8Type>::try_new(
                        Int8Array::from(vec![Some(0), None, Some(1), Some(0)]),
                        empty_structs(vec![true, false]),
                    )
                    .unwrap(),
                ),
                true,
            ),
            (
                "dense union of a struct without fields and of nulls",
                Arc::new(
                    UnionArray::try_new(
                        union_fields,
                        vec![0, 1, 0, 0].into(),
                        Some(vec![0, 0, 1, 2].into()),
                        vec![
                            empty_structs(vec![true, false, true]),
                            Arc::new(NullArray::new(1)),
                        ],
                    )
                    .unwrap(),
                ),
                false,
            ),
            (
                "run-end encoded nulls",
                Arc::new(
                    RunArray::<Int16Type>::try_new(
                        &Int16Array::from(vec![2, 4]),
                        &NullArray::new(2),
                    )
                    .unwrap(),
                ),
                true,
            ),
        ]);
        arrays.extend(
            dictionaries_of_values_without_nulls()
                .into_iter()
                .zip([
                    "dictionary with null keys of run-end encoded values",
                    "dictionary with null keys of a union without nullable fields",
                ])
                .map(|(array, name)| (name, array, true)),
        );
        arrays
    }

    /// Dictionaries of 2048 rows of `FixedSizeList<RunEndEncoded<Int16, _>,
    /// 32>` values: expanded to one value per row, their run-end encoded
    /// values would have 65,536 rows, more than `Int16` run ends can count.
    /// With one value referenced by every row, two values and null keys, and
    /// floats (`-0.0` and `+0.0` are equal for DataFusion, so they are
    /// compared by value).
    pub(crate) fn dictionaries_of_run_end_encoded_lists() -> Vec<ArrayRef> {
        const ROWS: usize = 2048;
        let lists = |run_ends: Vec<i16>, values: ArrayRef| -> ArrayRef {
            let runs =
                RunArray::<Int16Type>::try_new(&Int16Array::from(run_ends), &values)
                    .unwrap();
            let item = Arc::new(Field::new("item", runs.data_type().clone(), true));
            Arc::new(FixedSizeListArray::try_new(item, 32, Arc::new(runs), None).unwrap())
        };
        let dictionary = |keys: Int8Array, values: ArrayRef| -> ArrayRef {
            Arc::new(DictionaryArray::<Int8Type>::try_new(keys, values).unwrap())
        };
        vec![
            dictionary(
                Int8Array::from(vec![0; ROWS]),
                lists(vec![32], Arc::new(Int32Array::from(vec![10]))),
            ),
            dictionary(
                (0..ROWS)
                    .map(|i| (i % 5 != 4).then_some(i8::from(i % 3 == 0)))
                    .collect(),
                lists(
                    vec![32, 48, 64],
                    Arc::new(Int32Array::from(vec![10, 10, 20])),
                ),
            ),
            dictionary(
                (0..ROWS)
                    .map(|i| (i % 7 != 6).then_some((i % 3) as i8))
                    .collect(),
                lists(
                    vec![32, 64, 80, 96],
                    Arc::new(Float32Array::from(vec![-0.0, 0.0, 1.5, f32::NAN])),
                ),
            ),
        ]
    }

    /// Dictionaries with null keys of run-end encoded values (with a
    /// non-nullable values field) and of unions without nullable fields,
    /// which Arrow's `take` cannot take null indices of.
    pub(crate) fn dictionaries_of_values_without_nulls() -> Vec<ArrayRef> {
        let runs = arrow::array::make_array(
            arrow::array::ArrayData::builder(DataType::RunEndEncoded(
                Arc::new(Field::new("run_ends", DataType::Int32, false)),
                Arc::new(Field::new("values", DataType::Int32, false)),
            ))
            .len(2)
            .add_child_data(Int32Array::from(vec![1, 2]).to_data())
            .add_child_data(Int32Array::from(vec![10, 20]).to_data())
            .build()
            .unwrap(),
        );
        let unions = UnionArray::try_new(
            UnionFields::try_new(
                [0, 1],
                [
                    Field::new("i", DataType::Int32, false),
                    Field::new("s", DataType::Utf8, false),
                ],
            )
            .unwrap(),
            vec![0, 1, 0].into(),
            None,
            vec![
                Arc::new(Int32Array::from(vec![10, 0, 30])),
                Arc::new(StringArray::from(vec!["", "twenty", ""])),
            ],
        )
        .unwrap();
        [runs, Arc::new(unions)]
            .into_iter()
            .map(|values| -> ArrayRef {
                Arc::new(
                    DictionaryArray::<Int8Type>::try_new(
                        Int8Array::from(vec![None, Some(0), Some(1)]),
                        values,
                    )
                    .unwrap(),
                )
            })
            .collect()
    }

    /// `batch` as a chunk with the selection `selection`.
    pub(crate) fn chunk_with_selection(
        batch: &RecordBatch,
        selection: Vec<u32>,
    ) -> DataChunk {
        DataChunk::from_record_batch(batch.clone())
            .with_selection(SelectionVector::from(selection))
            .unwrap()
    }

    /// Whether `data_type` contains a dictionary below its top level.
    fn has_nested_dictionary(data_type: &DataType) -> bool {
        fn has_dictionary(data_type: &DataType) -> bool {
            matches!(data_type, DataType::Dictionary(..))
                || has_nested_dictionary(data_type)
        }
        match data_type {
            DataType::List(f)
            | DataType::LargeList(f)
            | DataType::ListView(f)
            | DataType::LargeListView(f)
            | DataType::FixedSizeList(f, _)
            | DataType::Map(f, _) => has_dictionary(f.data_type()),
            DataType::Struct(fields) => {
                fields.iter().any(|f| has_dictionary(f.data_type()))
            }
            DataType::Union(fields, _) => {
                fields.iter().any(|(_, f)| has_dictionary(f.data_type()))
            }
            DataType::RunEndEncoded(_, values) => has_dictionary(values.data_type()),
            DataType::Dictionary(_, values) => has_dictionary(values),
            _ => false,
        }
    }

    /// Assert that `actual` is identical to `expected`: same schema
    /// (including metadata), same data types, valid and equal arrays
    /// (logically equal for nested dictionaries), and byte-identical
    /// canonical Arrow IPC serializations.
    pub(crate) fn assert_identical(actual: &RecordBatch, expected: &RecordBatch) {
        assert_eq!(actual.schema(), expected.schema(), "schemas differ");
        assert_eq!(actual.num_rows(), expected.num_rows(), "row counts differ");
        for (i, (a, e)) in actual.columns().iter().zip(expected.columns()).enumerate() {
            let name = expected.schema().field(i).name().clone();
            assert_eq!(a.data_type(), e.data_type(), "data types of {name} differ");
            // E.g. the children of nested arrays have the types of their fields.
            if let Err(e) = a.to_data().validate_full() {
                panic!("{name} is not valid: {e}");
            }
            assert_eq!(a.len(), e.len(), "lengths of {name} differ");
            let nulls =
                |array: &ArrayRef| array.logical_nulls().filter(|n| n.null_count() > 0);
            assert!(nulls(a) == nulls(e), "nulls of {name} differ");
            // The keys of nested dictionaries are not restored exactly (the
            // row format decodes them to their values): their values are
            // compared through the canonical form below.
            if has_nested_dictionary(e.data_type()) {
                continue;
            }
            if a.to_data() != e.to_data() {
                let options =
                    arrow::util::display::FormatOptions::default().with_null("NULL");
                let format = |array: &ArrayRef| {
                    let formatter = arrow::util::display::ArrayFormatter::try_new(
                        array.as_ref(),
                        &options,
                    )
                    .unwrap();
                    (0..array.len().min(20))
                        .map(|i| formatter.value(i).to_string())
                        .collect::<Vec<_>>()
                };
                panic!(
                    "values of {name} differ\nactual:   {:?}\nexpected: {:?}",
                    format(a),
                    format(e)
                );
            }
        }
        // The canonical form normalizes the physical layout (e.g. the bytes
        // under null slots) through the row format, so it is only compared
        // for the columns that the row format supports. It decodes
        // dictionaries, so its fields must be nullable: a non-nullable
        // dictionary field can have null values (the schemas are compared
        // above).
        let schema = expected.schema();
        let supported = (0..schema.fields().len())
            .filter(|&i| {
                let field = SortField::new(schema.field(i).data_type().clone());
                RowConverter::supports_fields(&[field])
            })
            .collect::<Vec<_>>();
        let bytes = |batch: &RecordBatch, columns: &[usize]| {
            let batch = batch.project(columns).unwrap();
            let fields = batch
                .schema()
                .fields()
                .iter()
                .map(|f| f.as_ref().clone().with_nullable(true))
                .collect::<Vec<_>>();
            let schema = Arc::new(Schema::new_with_metadata(
                fields,
                batch.schema().metadata().clone(),
            ));
            let batch = RecordBatch::try_new_with_options(
                Arc::clone(&schema),
                batch.columns().to_vec(),
                &arrow::array::RecordBatchOptions::new()
                    .with_row_count(Some(batch.num_rows())),
            )
            .unwrap();
            crate::test_utils::canonical_ipc_bytes(&schema, &[batch], true).unwrap()
        };
        if bytes(actual, &supported) != bytes(expected, &supported) {
            for &i in &supported {
                assert!(
                    bytes(actual, &[i]) == bytes(expected, &[i]),
                    "canonical IPC bytes of {} differ",
                    schema.field(i).name()
                );
            }
            panic!("canonical IPC bytes differ");
        }
    }

    #[test]
    fn all_types_batch_is_valid() {
        let batch = all_types_batch(50);
        assert_eq!(batch.num_rows(), 50);
        assert!(batch.num_columns() > 45);
        for column in batch.columns() {
            column.to_data().validate_full().unwrap();
        }
    }
}
