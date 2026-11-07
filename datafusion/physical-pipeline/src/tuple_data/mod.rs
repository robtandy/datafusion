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

    /// `batch` as a chunk with the selection `selection`.
    pub(crate) fn chunk_with_selection(
        batch: &RecordBatch,
        selection: Vec<u32>,
    ) -> DataChunk {
        DataChunk::from_record_batch(batch.clone())
            .with_selection(SelectionVector::from(selection))
            .unwrap()
    }

    /// Assert that `actual` is identical to `expected`: same schema
    /// (including metadata), same data types, logically equal arrays, and
    /// byte-identical canonical Arrow IPC serializations.
    pub(crate) fn assert_identical(actual: &RecordBatch, expected: &RecordBatch) {
        assert_eq!(actual.schema(), expected.schema(), "schemas differ");
        assert_eq!(actual.num_rows(), expected.num_rows(), "row counts differ");
        for (i, (a, e)) in actual.columns().iter().zip(expected.columns()).enumerate() {
            let name = expected.schema().field(i).name().clone();
            assert_eq!(a.data_type(), e.data_type(), "data types of {name} differ");
            assert_eq!(a.len(), e.len(), "lengths of {name} differ");
            let nulls =
                |array: &ArrayRef| array.logical_nulls().filter(|n| n.null_count() > 0);
            assert!(nulls(a) == nulls(e), "nulls of {name} differ");
            // Dictionaries are compared by their values: a valid key of a
            // null value becomes a null key.
            let decoded = |array: &ArrayRef| match array.data_type() {
                DataType::Dictionary(_, value_type) => cast(array, value_type).unwrap(),
                _ => Arc::clone(array),
            };
            let (a, e) = (&decoded(a), &decoded(e));
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
        // for the columns that the row format supports.
        let schema = expected.schema();
        let supported = (0..schema.fields().len())
            .filter(|&i| {
                let field = SortField::new(schema.field(i).data_type().clone());
                RowConverter::supports_fields(&[field])
            })
            .collect::<Vec<_>>();
        let bytes = |batch: &RecordBatch, columns: &[usize]| {
            let batch = batch.project(columns).unwrap();
            crate::test_utils::canonical_ipc_bytes(&batch.schema(), &[batch], true)
                .unwrap()
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
