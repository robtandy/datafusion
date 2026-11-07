// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`RowMatcher`]: vectorized comparison of the keys of probe rows (in a
//! [`DataChunk`]) with the keys of candidate rows in a
//! [`TupleDataCollection`] (DuckDB's `RowMatcher`).
//!
//! This is the key comparison of hash joins and hash aggregations: after a
//! hash table lookup found a candidate row for a probe row (a row with the
//! same hash, or the same hash salt), the matcher checks whether their keys
//! are actually equal.
//!
//! Equality follows DataFusion's join key equality
//! (`equal_rows_arr` / `JoinKeyComparator`) exactly:
//!
//! * two valid values are equal if their Arrow values are equal; floats are
//!   compared by their bits after normalizing `-0.0` to `+0.0`, so
//!   `-0.0 = +0.0` and a `NaN` equals a `NaN` with the same bits;
//!   dictionaries compare their values; nested values are compared value by
//!   value (nested nulls are equal to each other, nested floats are
//!   normalized too);
//! * a null and a valid value are never equal;
//! * two nulls are equal with [`NullEquality::NullEqualsNull`]
//!   (`IS NOT DISTINCT FROM`) and not equal with
//!   [`NullEquality::NullEqualsNothing`] (`=`).

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, make_comparator};
use arrow::buffer::NullBuffer;
use arrow::compute::SortOptions;
use arrow::datatypes::DataType;
use datafusion_common::float_semantics::normalize_floats;
use datafusion_common::{NullEquality, Result, internal_datafusion_err, internal_err};

use super::collection::{
    RowLocation, RowRef, TupleDataCollection, decode_dictionary, fixed_values,
    gather_values,
};
use super::layout::{ColumnStorage, TupleDataLayout};
use crate::chunk::DataChunk;

/// Compares the key columns of probe rows with the key columns of candidate
/// rows stored in a [`TupleDataCollection`]. See the
/// [module documentation](self).
///
/// Column `c` of the probe chunk is compared with column `c` of the layout,
/// for every key column `c` (as in DuckDB, e.g. the probe chunk holds the
/// join keys and the layout starts with the join keys).
#[derive(Debug, Clone)]
pub struct RowMatcher {
    layout: Arc<TupleDataLayout>,
    key_columns: Vec<usize>,
    null_equality: NullEquality,
}

impl RowMatcher {
    /// A matcher that compares the columns `key_columns` of probe chunks with
    /// the same columns of rows with layout `layout`.
    pub fn try_new(
        layout: Arc<TupleDataLayout>,
        key_columns: Vec<usize>,
        null_equality: NullEquality,
    ) -> Result<Self> {
        if let Some(&column) = key_columns.iter().find(|&&c| c >= layout.num_columns()) {
            return internal_err!(
                "Key column {column} out of bounds for a TupleDataLayout with {} columns",
                layout.num_columns()
            );
        }
        Ok(Self {
            layout,
            key_columns,
            null_equality,
        })
    }

    /// The layout of the candidate rows.
    pub fn layout(&self) -> &Arc<TupleDataLayout> {
        &self.layout
    }

    /// The compared columns.
    pub fn key_columns(&self) -> &[usize] {
        &self.key_columns
    }

    /// How nulls are compared.
    pub fn null_equality(&self) -> NullEquality {
        self.null_equality
    }

    /// Compare the keys of the probe rows in `sel` with the keys of their
    /// candidate rows, and keep in `sel` only the rows whose keys are equal.
    ///
    /// * `sel` holds logical row indices of `probe` (relative to its
    ///   selection, if it has one). On return it holds the rows that match
    ///   their candidate, in their original order.
    /// * `candidates[row]` is the location of the candidate row in
    ///   `collection` for probe row `row` (as in DuckDB, the candidates are
    ///   indexed by probe row, so only the entries of rows in `sel` are
    ///   read).
    /// * the rows that do not match are appended to `no_match`, if given.
    ///
    /// Returns the number of matching rows.
    pub fn match_rows(
        &self,
        probe: &DataChunk,
        sel: &mut Vec<u32>,
        candidates: &[RowLocation],
        collection: &TupleDataCollection,
        mut no_match: Option<&mut Vec<u32>>,
    ) -> Result<usize> {
        if sel.is_empty() {
            return Ok(0);
        }
        if !collection.layout().is_compatible(&self.layout) {
            return internal_err!(
                "The TupleDataCollection does not have the layout of the RowMatcher"
            );
        }
        for &column in &self.key_columns {
            let probe_type = probe.schema().fields().get(column).map(|f| f.data_type());
            let key_type = self.layout.column(column).data_type();
            if probe_type != Some(key_type) {
                return internal_err!(
                    "Probe column {column} has type {probe_type:?}, but the key has type {key_type}"
                );
            }
        }
        let num_rows = probe.len().min(candidates.len());
        if let Some(&row) = sel.iter().find(|&&row| row as usize >= num_rows) {
            return internal_err!(
                "Probe row {row} out of bounds for {} probe rows and {} candidates",
                probe.len(),
                candidates.len()
            );
        }
        let mut rows = sel
            .iter()
            .map(|&row| collection.row_ref(candidates[row as usize]))
            .collect::<Result<Vec<_>>>()?;
        let physical = probe.selection().map(|s| s.indices());
        for &column in &self.key_columns {
            if sel.is_empty() {
                break;
            }
            let mut matches = ColumnMatches {
                sel: &mut *sel,
                rows: &mut rows,
                no_match: no_match.as_deref_mut(),
                physical,
                column,
                null_equals_null: self.null_equality == NullEquality::NullEqualsNull,
            };
            self.match_column(column, probe.column(column), &mut matches)?;
        }
        Ok(sel.len())
    }

    fn match_column(
        &self,
        column: usize,
        array: &ArrayRef,
        matches: &mut ColumnMatches<'_, '_>,
    ) -> Result<()> {
        let layout = self.layout.column(column);
        let values = if layout.is_dictionary() {
            decode_dictionary(array)?
        } else {
            Arc::clone(array)
        };
        let nulls = values.logical_nulls().filter(|n| n.null_count() > 0);
        let nulls = nulls.as_ref();
        let offset = layout.offset();
        match layout.storage() {
            // Both sides are always null.
            ColumnStorage::Null => matches.retain(nulls, |_, _, _| true),
            ColumnStorage::Boolean => {
                let values = values.as_boolean();
                matches
                    .retain(nulls, |_, i, row| values.value(i) == (row.row[offset] != 0))
            }
            ColumnStorage::Fixed(width) => {
                let bytes = fixed_values(&values, width);
                let bytes = bytes.as_slice();
                match layout.value_type() {
                    DataType::Float16 => matches.retain(nulls, |_, i, row| {
                        float_bits_eq::<2>(&bytes[i * 2..], &row.row[offset..])
                    }),
                    DataType::Float32 => matches.retain(nulls, |_, i, row| {
                        float_bits_eq::<4>(&bytes[i * 4..], &row.row[offset..])
                    }),
                    DataType::Float64 => matches.retain(nulls, |_, i, row| {
                        float_bits_eq::<8>(&bytes[i * 8..], &row.row[offset..])
                    }),
                    _ => matches.retain(nulls, |_, i, row| {
                        bytes[i * width..(i + 1) * width]
                            == row.row[offset..offset + width]
                    }),
                }
            }
            ColumnStorage::VarLen => {
                macro_rules! retain_bytes {
                    ($values:expr, $as_bytes:expr) => {{
                        let values = $values;
                        matches.retain(nulls, |_, i, row| {
                            $as_bytes(values.value(i)) == row.var_len(offset)
                        })
                    }};
                }
                match values.data_type() {
                    DataType::Utf8 => {
                        retain_bytes!(values.as_string::<i32>(), str::as_bytes)
                    }
                    DataType::LargeUtf8 => {
                        retain_bytes!(values.as_string::<i64>(), str::as_bytes)
                    }
                    DataType::Utf8View => {
                        retain_bytes!(values.as_string_view(), str::as_bytes)
                    }
                    DataType::Binary => {
                        retain_bytes!(values.as_binary::<i32>(), identity)
                    }
                    DataType::LargeBinary => {
                        retain_bytes!(values.as_binary::<i64>(), identity)
                    }
                    DataType::BinaryView => {
                        retain_bytes!(values.as_binary_view(), identity)
                    }
                    data_type => {
                        return internal_err!(
                            "Unexpected variable-length data type {data_type}"
                        );
                    }
                }
            }
            ColumnStorage::Encoded => {
                let codec = layout.codec().ok_or_else(|| {
                    internal_datafusion_err!("Encoded column without a row codec")
                })?;
                if codec.has_float_leaf {
                    // The row format distinguishes `-0.0` and `+0.0`: compare
                    // the normalized values, like `JoinKeyComparator`.
                    let candidates = gather_values(layout, column, matches.rows)?;
                    let comparator = make_comparator(
                        normalize_floats(&values).as_ref(),
                        normalize_floats(&candidates).as_ref(),
                        SortOptions::default(),
                    )?;
                    matches.retain(nulls, |entry, i, _| comparator(i, entry).is_eq())
                } else {
                    // Equal values have equal row encodings.
                    let encoded = codec.converter.convert_columns(&[values])?;
                    matches.retain(nulls, |_, i, row| {
                        encoded.row(i).data() == row.var_len(offset)
                    })
                }
            }
        }
        Ok(())
    }
}

#[inline]
fn identity(bytes: &[u8]) -> &[u8] {
    bytes
}

/// Whether the native-endian floats of `W` bytes at the start of `a` and
/// `b` are equal after normalizing `-0.0` to `+0.0` (DataFusion's
/// `normalize_float`), i.e. `-0.0 == +0.0` and `NaN`s are equal if they
/// have the same bits.
#[inline]
fn float_bits_eq<const W: usize>(a: &[u8], b: &[u8]) -> bool {
    #[inline]
    fn normalized<const W: usize>(bytes: &[u8]) -> u64 {
        let mut value = [0_u8; 8];
        value[..W].copy_from_slice(&bytes[..W]);
        let bits = u64::from_ne_bytes(value);
        let negative_zero = if cfg!(target_endian = "little") {
            1_u64 << (W * 8 - 1)
        } else {
            1_u64 << 63
        };
        if bits == negative_zero { 0 } else { bits }
    }
    normalized::<W>(a) == normalized::<W>(b)
}

/// The state of the comparison of one key column: the probe rows (`sel`)
/// and their candidate rows (`rows`) that matched all previous columns.
struct ColumnMatches<'a, 'r> {
    sel: &'a mut Vec<u32>,
    rows: &'a mut Vec<RowRef<'r>>,
    no_match: Option<&'a mut Vec<u32>>,
    /// The selection of the probe chunk.
    physical: Option<&'a [u32]>,
    column: usize,
    null_equals_null: bool,
}

impl ColumnMatches<'_, '_> {
    /// Keep the entries whose values are equal, where `nulls` are the
    /// logical nulls of the probe column and `equal(entry, i, row)` compares
    /// the valid value of physical probe row `i` with the valid value of
    /// candidate `row`, the candidate of entry `entry`.
    #[inline]
    fn retain(
        &mut self,
        nulls: Option<&NullBuffer>,
        mut equal: impl FnMut(usize, usize, &RowRef<'_>) -> bool,
    ) {
        let mut kept = 0;
        for entry in 0..self.sel.len() {
            let row = self.sel[entry];
            let candidate = self.rows[entry];
            let i = match self.physical {
                Some(physical) => physical[row as usize] as usize,
                None => row as usize,
            };
            let probe_valid = nulls.is_none_or(|n| n.is_valid(i));
            let equal = match (probe_valid, candidate.is_valid(self.column)) {
                (true, true) => equal(entry, i, &candidate),
                (false, false) => self.null_equals_null,
                _ => false,
            };
            if equal {
                self.sel[kept] = row;
                self.rows[kept] = candidate;
                kept += 1;
            } else if let Some(no_match) = self.no_match.as_deref_mut() {
                no_match.push(row);
            }
        }
        self.sel.truncate(kept);
        self.rows.truncate(kept);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::array::{
        DictionaryArray, Float32Array, Float64Array, Int32Array, ListArray, RecordBatch,
        StringArray, StructArray,
    };
    use arrow::datatypes::{Field, Fields, Float64Type, Int8Type};
    use datafusion_physical_plan::joins::utils::JoinKeyComparator;

    use crate::tuple_data::test_data::{all_types_batch, chunk_with_selection};

    /// Store `build` in a collection and return it with the locations of its
    /// rows.
    fn store(build: &RecordBatch) -> (TupleDataCollection, Vec<RowLocation>) {
        let layout = TupleDataLayout::try_new(build.schema(), false)
            .unwrap()
            .with_rows_per_block(7)
            .unwrap();
        let mut collection = TupleDataCollection::new(Arc::new(layout));
        let locations = collection
            .append_chunk(&DataChunk::from_record_batch(build.clone()), None)
            .unwrap();
        (collection, locations)
    }

    /// Match every logical probe row `i` against build row `pairs[i]`,
    /// returning the matching probe rows and the non-matching probe rows.
    fn match_pairs(
        build: &RecordBatch,
        probe: &DataChunk,
        build_rows: &[usize],
        key_columns: Vec<usize>,
        null_equality: NullEquality,
    ) -> (Vec<u32>, Vec<u32>) {
        let (collection, locations) = store(build);
        let matcher = RowMatcher::try_new(
            Arc::clone(collection.layout()),
            key_columns,
            null_equality,
        )
        .unwrap();
        let candidates = build_rows.iter().map(|&r| locations[r]).collect::<Vec<_>>();
        let mut sel = (0..probe.len() as u32).collect::<Vec<_>>();
        let mut no_match = vec![];
        let count = matcher
            .match_rows(
                probe,
                &mut sel,
                &candidates,
                &collection,
                Some(&mut no_match),
            )
            .unwrap();
        assert_eq!(count, sel.len());
        no_match.sort_unstable();
        (sel, no_match)
    }

    fn batch(columns: Vec<(&str, ArrayRef)>) -> RecordBatch {
        RecordBatch::try_from_iter(columns).unwrap()
    }

    #[test]
    fn equal_and_unequal_keys() {
        let build = batch(vec![
            (
                "k",
                Arc::new(Int32Array::from(vec![1, 2, 3, 4])) as ArrayRef,
            ),
            (
                "s",
                Arc::new(StringArray::from(vec![
                    "a",
                    "bb",
                    "a long string value",
                    "d",
                ])),
            ),
        ]);
        let probe = batch(vec![
            (
                "k",
                Arc::new(Int32Array::from(vec![1, 2, 3, 4, 3])) as ArrayRef,
            ),
            (
                "s",
                Arc::new(StringArray::from(vec![
                    "a",
                    "bx",
                    "a long string value",
                    "a long string valuf",
                    "a long string value",
                ])),
            ),
        ]);
        let probe = DataChunk::from_record_batch(probe);
        // Rows 0, 2 and 4 match; row 1 differs in `s`, row 3 has another key.
        let (matched, unmatched) = match_pairs(
            &build,
            &probe,
            &[0, 1, 2, 2, 2],
            vec![0, 1],
            NullEquality::NullEqualsNothing,
        );
        assert_eq!(matched, vec![0, 2, 4]);
        assert_eq!(unmatched, vec![1, 3]);
        let (matched, _) = match_pairs(
            &build,
            &probe,
            &[0, 1, 2, 3, 2],
            vec![0, 1],
            NullEquality::NullEqualsNothing,
        );
        assert_eq!(matched, vec![0, 2, 4]);
        // Only the first column.
        let (matched, _) = match_pairs(
            &build,
            &probe,
            &[0, 1, 2, 3, 2],
            vec![0],
            NullEquality::NullEqualsNothing,
        );
        assert_eq!(matched, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn hash_collisions_are_not_matches() {
        // All rows have the same hash: the matcher compares the keys, not the
        // hashes.
        let build = batch(vec![(
            "k",
            Arc::new(StringArray::from(vec!["collision a", "collision b"])) as ArrayRef,
        )]);
        let layout = Arc::new(TupleDataLayout::try_new(build.schema(), true).unwrap());
        let mut collection = TupleDataCollection::new(Arc::clone(&layout));
        let locations = collection
            .append_chunk(
                &DataChunk::from_record_batch(build.clone()),
                Some(&[42, 42]),
            )
            .unwrap();
        let probe = DataChunk::from_record_batch(batch(vec![(
            "k",
            Arc::new(StringArray::from(vec![
                "collision b",
                "collision a",
                "collision c",
            ])) as ArrayRef,
        )]));
        let matcher =
            RowMatcher::try_new(layout, vec![0], NullEquality::NullEqualsNull).unwrap();
        let candidates = vec![locations[0], locations[0], locations[1]];
        let mut sel = vec![0, 1, 2];
        let mut no_match = vec![];
        let count = matcher
            .match_rows(
                &probe,
                &mut sel,
                &candidates,
                &collection,
                Some(&mut no_match),
            )
            .unwrap();
        assert_eq!(count, 1);
        assert_eq!(sel, vec![1]);
        assert_eq!(no_match, vec![0, 2]);
        // The non-matching rows can be compared with the next candidates.
        let candidates = vec![locations[1], locations[0], locations[0]];
        let mut sel = no_match;
        let count = matcher
            .match_rows(&probe, &mut sel, &candidates, &collection, None)
            .unwrap();
        assert_eq!(count, 1);
        assert_eq!(sel, vec![0]);
    }

    #[test]
    fn null_equality() {
        let build = batch(vec![(
            "k",
            Arc::new(Int32Array::from(vec![Some(1), None])) as ArrayRef,
        )]);
        let probe = DataChunk::from_record_batch(batch(vec![(
            "k",
            Arc::new(Int32Array::from(vec![None, None, Some(1), Some(1)])) as ArrayRef,
        )]));
        // (NULL, 1), (NULL, NULL), (1, NULL), (1, 1).
        let pairs = [0, 1, 1, 0];
        let (matched, unmatched) = match_pairs(
            &build,
            &probe,
            &pairs,
            vec![0],
            NullEquality::NullEqualsNothing,
        );
        assert_eq!(matched, vec![3]);
        assert_eq!(unmatched, vec![0, 1, 2]);
        let (matched, unmatched) = match_pairs(
            &build,
            &probe,
            &pairs,
            vec![0],
            NullEquality::NullEqualsNull,
        );
        assert_eq!(matched, vec![1, 3]);
        assert_eq!(unmatched, vec![0, 2]);
    }

    /// An array of `data_type` (a float type) with the values with the
    /// given bits.
    fn floats(data_type: &DataType, bits: &[u64]) -> ArrayRef {
        match data_type {
            DataType::Float16 => {
                let bits = bits.iter().map(|&b| b as u16).collect::<Vec<_>>();
                let data = arrow::array::ArrayData::builder(DataType::Float16)
                    .len(bits.len())
                    .add_buffer(arrow::buffer::Buffer::from_slice_ref(&bits))
                    .build()
                    .unwrap();
                arrow::array::make_array(data)
            }
            DataType::Float32 => Arc::new(Float32Array::from_iter_values(
                bits.iter().map(|&b| f32::from_bits(b as u32)),
            )),
            _ => Arc::new(Float64Array::from_iter_values(
                bits.iter().map(|&b| f64::from_bits(b)),
            )),
        }
    }

    #[test]
    fn float_normalization() {
        // The bits of 0.0, -0.0, NaN, -NaN, 1.5 and a NaN with a payload.
        let f16 = [0, 0x8000, 0x7E00, 0xFE00, 0x3E00, 0x7E01];
        let f32 = [
            0,
            0x8000_0000,
            0x7FC0_0000,
            0xFFC0_0000,
            0x3FC0_0000,
            0x7FC0_0001,
        ];
        let f64 = [
            0,
            0x8000_0000_0000_0000,
            0x7FF8_0000_0000_0000,
            0xFFF8_0000_0000_0000,
            0x3FF8_0000_0000_0000,
            0x7FF8_0000_0000_0001,
        ];
        for (data_type, bits) in [
            (DataType::Float16, f16),
            (DataType::Float32, f32),
            (DataType::Float64, f64),
        ] {
            let [
                zero,
                negative_zero,
                nan,
                negative_nan,
                one_and_a_half,
                payload_nan,
            ] = bits;
            let build = batch(vec![(
                "f",
                floats(
                    &data_type,
                    &[zero, negative_zero, nan, nan, one_and_a_half, payload_nan],
                ),
            )]);
            let probe = DataChunk::from_record_batch(batch(vec![(
                "f",
                floats(
                    &data_type,
                    &[negative_zero, zero, nan, negative_nan, one_and_a_half, nan],
                ),
            )]));
            // -0.0 = 0.0, 0.0 = -0.0, NaN = NaN, 1.5 = 1.5, but NaNs with
            // different bits differ.
            let (matched, unmatched) = match_pairs(
                &build,
                &probe,
                &[0, 1, 2, 3, 4, 5],
                vec![0],
                NullEquality::NullEqualsNothing,
            );
            assert_eq!(matched, vec![0, 1, 2, 4], "{data_type}");
            assert_eq!(unmatched, vec![3, 5], "{data_type}");

            // DataFusion agrees.
            let comparator = JoinKeyComparator::new(
                &[Arc::clone(build.column(0))],
                &[Arc::clone(probe.column(0))],
                &[SortOptions::default()],
                NullEquality::NullEqualsNothing,
            )
            .unwrap();
            let expected = (0..6)
                .filter(|&i| comparator.is_equal(i, i))
                .map(|i| i as u32)
                .collect::<Vec<_>>();
            assert_eq!(matched, expected, "{data_type}");
        }
    }

    #[test]
    fn probe_selection_and_dictionaries() {
        let dictionary: DictionaryArray<Int8Type> =
            vec![Some("x"), None, Some("y"), Some("a long dictionary value")]
                .into_iter()
                .collect();
        let build = batch(vec![("d", Arc::new(dictionary) as ArrayRef)]);
        // Another dictionary with the values in another order.
        let probe_dictionary: DictionaryArray<Int8Type> = vec![
            Some("a long dictionary value"),
            Some("y"),
            None,
            Some("x"),
            Some("z"),
        ]
        .into_iter()
        .collect();
        let probe = batch(vec![("d", Arc::new(probe_dictionary) as ArrayRef)]);
        // Logical probe rows: 0 -> physical 3 ("x"), 1 -> 2 (NULL),
        // 2 -> 0 ("a long ..."), 3 -> 4 ("z").
        let probe = chunk_with_selection(&probe, vec![3, 2, 0, 4]);
        let (matched, unmatched) = match_pairs(
            &build,
            &probe,
            &[0, 1, 3, 2],
            vec![0],
            NullEquality::NullEqualsNull,
        );
        assert_eq!(matched, vec![0, 1, 2]);
        assert_eq!(unmatched, vec![3]);
        let (matched, _) = match_pairs(
            &build,
            &probe,
            &[0, 1, 3, 2],
            vec![0],
            NullEquality::NullEqualsNothing,
        );
        assert_eq!(matched, vec![0, 2]);
    }

    #[test]
    fn nested_keys() {
        let list = |values: Vec<Option<Vec<Option<f64>>>>| -> ArrayRef {
            Arc::new(ListArray::from_iter_primitive::<Float64Type, _, _>(values))
        };
        let build = batch(vec![(
            "l",
            list(vec![
                Some(vec![Some(-0.0), None]),
                Some(vec![Some(1.0)]),
                None,
                Some(vec![]),
            ]),
        )]);
        let probe = DataChunk::from_record_batch(batch(vec![(
            "l",
            list(vec![
                Some(vec![Some(0.0), None]),
                Some(vec![Some(1.0), Some(2.0)]),
                None,
                Some(vec![]),
                Some(vec![Some(-0.0), Some(f64::NAN)]),
            ]),
        )]));
        // Nested -0.0 equals 0.0 and nested nulls are equal.
        let (matched, _) = match_pairs(
            &build,
            &probe,
            &[0, 1, 2, 3, 0],
            vec![0],
            NullEquality::NullEqualsNothing,
        );
        assert_eq!(matched, vec![0, 3]);
        let (matched, _) = match_pairs(
            &build,
            &probe,
            &[0, 1, 2, 3, 0],
            vec![0],
            NullEquality::NullEqualsNull,
        );
        assert_eq!(matched, vec![0, 2, 3]);

        // Without floats, the row encodings are compared.
        let fields = Fields::from(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Utf8, true),
        ]);
        let structs = |a: Vec<Option<i32>>, b: Vec<Option<&str>>| -> ArrayRef {
            Arc::new(
                StructArray::try_new(
                    fields.clone(),
                    vec![
                        Arc::new(Int32Array::from(a)),
                        Arc::new(StringArray::from(b)),
                    ],
                    None,
                )
                .unwrap(),
            )
        };
        let build = batch(vec![(
            "s",
            structs(
                vec![Some(1), None, Some(3)],
                vec![Some("x"), Some("y"), None],
            ),
        )]);
        let probe = DataChunk::from_record_batch(batch(vec![(
            "s",
            structs(
                vec![Some(1), None, Some(3), Some(1)],
                vec![Some("x"), Some("y"), Some("z"), Some("y")],
            ),
        )]));
        let (matched, _) = match_pairs(
            &build,
            &probe,
            &[0, 1, 2, 0],
            vec![0],
            NullEquality::NullEqualsNothing,
        );
        assert_eq!(matched, vec![0, 1]);
    }

    /// The matcher agrees with DataFusion's `JoinKeyComparator` on all pairs
    /// of rows of a batch with all data types.
    #[test]
    fn matches_join_key_comparator() {
        let all = all_types_batch(40);
        // Every row is compared with a few shifted rows, with duplicated
        // rows so that equal keys exist.
        let build =
            arrow::compute::concat_batches(&all.schema(), &[all.clone(), all.clone()])
                .unwrap();
        let probe_rows = (0..80_u32).rev().collect::<Vec<_>>();
        let probe = chunk_with_selection(&build, probe_rows.clone());
        let compacted = probe.compact().unwrap();
        for column in 0..all.num_columns() {
            let data_type = all.schema().field(column).data_type().clone();
            for null_equality in [
                NullEquality::NullEqualsNothing,
                NullEquality::NullEqualsNull,
            ] {
                // DataFusion can compare all the types.
                let comparator = JoinKeyComparator::new(
                    &[Arc::clone(build.column(column))],
                    &[Arc::clone(compacted.column(column))],
                    &[SortOptions::default()],
                    null_equality,
                )
                .unwrap();
                for shift in [0, 1, 40, 41, 17] {
                    let pairs = (0..80).map(|i| (i + shift) % 80).collect::<Vec<_>>();
                    let (matched, _) =
                        match_pairs(&build, &probe, &pairs, vec![column], null_equality);
                    let expected = (0..80)
                        .filter(|&i| comparator.is_equal(pairs[i], i))
                        .map(|i| i as u32)
                        .collect::<Vec<_>>();
                    assert_eq!(
                        matched, expected,
                        "column {column} ({data_type}), {null_equality:?}, shift {shift}"
                    );
                }
            }
        }
    }

    #[test]
    fn multi_column_keys_match_join_key_comparator() {
        let all = all_types_batch(30);
        let build =
            arrow::compute::concat_batches(&all.schema(), &[all.clone(), all.clone()])
                .unwrap();
        let probe = DataChunk::from_record_batch(build.clone());
        let key_columns = vec![0, 3, 11, 33, 34, 37, 41, 44];
        let arrays = |batch: &RecordBatch| {
            key_columns
                .iter()
                .map(|&c| Arc::clone(batch.column(c)))
                .collect::<Vec<_>>()
        };
        for null_equality in [
            NullEquality::NullEqualsNothing,
            NullEquality::NullEqualsNull,
        ] {
            let comparator = JoinKeyComparator::new(
                &arrays(&build),
                &arrays(&build),
                &vec![SortOptions::default(); key_columns.len()],
                null_equality,
            )
            .unwrap();
            for shift in [0, 30, 1] {
                let pairs = (0..60).map(|i| (i + shift) % 60).collect::<Vec<_>>();
                let (matched, _) = match_pairs(
                    &build,
                    &probe,
                    &pairs,
                    key_columns.clone(),
                    null_equality,
                );
                let expected = (0..60)
                    .filter(|&i| comparator.is_equal(pairs[i], i))
                    .map(|i| i as u32)
                    .collect::<Vec<_>>();
                assert_eq!(matched, expected, "{null_equality:?}, shift {shift}");
                if shift == 30 {
                    // Equal rows match unless they have null keys.
                    assert!(!matched.is_empty());
                }
            }
        }
    }

    /// A hash join built from the parts of this module: the build side is
    /// scattered into thread-local partitions, which are combined into a
    /// single collection; a hash table maps hashes to rows; the matcher
    /// compares the probe rows (with a selection) with their candidates.
    #[test]
    fn hash_join_with_partitioned_build() {
        use std::collections::HashMap;

        use datafusion_common::hash_utils::{RandomState, create_hashes};

        use crate::tuple_data::{PartitionedTupleData, TupleDataScanState};

        let build = all_types_batch(400);
        let probe = all_types_batch(300);
        // bool, f64 (with -0.0 and NaNs) and dict_utf8 keys, with nulls and
        // many duplicates.
        let key_columns = [0, 11, 41];
        let random_state = RandomState::with_seed(3);
        // The hashes of the keys, the first columns of `batch`.
        let hash = |batch: &RecordBatch| {
            let keys = batch.columns()[..key_columns.len()].to_vec();
            let mut hashes = vec![0; batch.num_rows()];
            create_hashes(&keys, &random_state, &mut hashes).unwrap();
            hashes
        };
        // The layout starts with the keys, followed by the payload.
        let mut build_columns = key_columns.to_vec();
        build_columns.extend([34, 3, 48, 44]);
        let build = build.project(&build_columns).unwrap();
        let layout = Arc::new(
            TupleDataLayout::try_new(build.schema(), true)
                .unwrap()
                .with_rows_per_block(16)
                .unwrap(),
        );

        // Build: three "threads" scatter chunks into their partitions.
        let build_hashes = hash(&build);
        let mut global = PartitionedTupleData::try_new(Arc::clone(&layout), 3).unwrap();
        for thread in 0..3 {
            let mut local =
                PartitionedTupleData::try_new(Arc::clone(&layout), 3).unwrap();
            for offset in (thread * 50..400).step_by(150) {
                let chunk = DataChunk::from_record_batch(build.slice(offset, 50));
                local
                    .append_chunk(&chunk, &build_hashes[offset..offset + 50])
                    .unwrap();
            }
            global.combine(local).unwrap();
        }
        let (collection, _) = global.into_collection().unwrap();
        let mut table: HashMap<u64, Vec<RowLocation>> = HashMap::new();
        let mut state = TupleDataScanState::new().with_column_ids(vec![]);
        while collection.scan_chunk(&mut state).unwrap().is_some() {
            let hashes = collection.gather_hashes(state.locations()).unwrap();
            for (&hash, &location) in hashes.iter().zip(state.locations()) {
                table.entry(hash).or_default().push(location);
            }
        }

        // Probe the rows of a selection, one candidate per row and round.
        let selection = (0..300_u32)
            .filter(|i| i % 4 != 1)
            .rev()
            .collect::<Vec<_>>();
        let probe_keys = probe.project(&key_columns).unwrap();
        let probe_chunk = chunk_with_selection(&probe_keys, selection.clone());
        let probe_hashes = hash(&probe_chunk.compact().unwrap());
        let matcher = RowMatcher::try_new(
            Arc::clone(&layout),
            (0..key_columns.len()).collect(),
            NullEquality::NullEqualsNothing,
        )
        .unwrap();
        let empty = vec![];
        let candidate_lists = probe_hashes
            .iter()
            .map(|hash| table.get(hash).unwrap_or(&empty))
            .collect::<Vec<_>>();
        let mut pairs = vec![];
        let mut candidates = vec![RowLocation::default(); selection.len()];
        for round in 0.. {
            let mut sel = (0..selection.len() as u32)
                .filter(|&row| candidate_lists[row as usize].len() > round)
                .collect::<Vec<_>>();
            if sel.is_empty() {
                break;
            }
            for &row in &sel {
                candidates[row as usize] = candidate_lists[row as usize][round];
            }
            matcher
                .match_rows(&probe_chunk, &mut sel, &candidates, &collection, None)
                .unwrap();
            pairs.extend(sel.iter().map(|&row| (row, candidates[row as usize])));
        }
        assert!(pairs.len() > selection.len(), "1:N matches");

        // The same pairs as a nested loop join with DataFusion's key
        // comparison.
        let compacted = probe_chunk.compact().unwrap();
        let comparator = JoinKeyComparator::new(
            build.project(&[0, 1, 2]).unwrap().columns(),
            compacted.columns(),
            &[SortOptions::default(); 3],
            NullEquality::NullEqualsNothing,
        )
        .unwrap();
        let mut expected = vec![];
        for row in 0..selection.len() {
            for build_row in 0..400 {
                if comparator.is_equal(build_row, row) {
                    expected.push((row as u32, build_row));
                }
            }
        }
        // Identify the build rows of the matches by their payload.
        let locations = pairs.iter().map(|&(_, l)| l).collect::<Vec<_>>();
        let payload = collection
            .gather_projected(&locations, &[3, 4, 5, 6])
            .unwrap();
        let indices = arrow::array::UInt32Array::from(
            expected.iter().map(|&(_, b)| b as u32).collect::<Vec<_>>(),
        );
        let mut actual = pairs
            .iter()
            .enumerate()
            .map(|(i, &(row, _))| (row, payload.slice(i, 1)))
            .collect::<Vec<_>>();
        let expected_payload = arrow::compute::take_record_batch(
            &build.project(&[3, 4, 5, 6]).unwrap(),
            &indices,
        )
        .unwrap();
        let mut expected = expected
            .iter()
            .enumerate()
            .map(|(i, &(row, _))| (row, expected_payload.slice(i, 1)))
            .collect::<Vec<_>>();
        let key = |(row, batch): &(u32, RecordBatch)| {
            (
                *row,
                arrow::util::pretty::pretty_format_batches(std::slice::from_ref(batch))
                    .unwrap()
                    .to_string(),
            )
        };
        actual.sort_by_cached_key(key);
        expected.sort_by_cached_key(key);
        assert_eq!(actual.len(), expected.len());
        for (a, e) in actual.iter().zip(&expected) {
            assert_eq!(a.0, e.0);
            assert_eq!(a.1, e.1);
        }
    }

    #[test]
    fn invalid_inputs_are_errors() {
        let build = batch(vec![(
            "k",
            Arc::new(Int32Array::from(vec![1, 2])) as ArrayRef,
        )]);
        let (collection, locations) = store(&build);
        let layout = Arc::clone(collection.layout());
        assert!(
            RowMatcher::try_new(
                Arc::clone(&layout),
                vec![1],
                NullEquality::NullEqualsNull
            )
            .is_err()
        );
        let matcher = RowMatcher::try_new(
            Arc::clone(&layout),
            vec![0],
            NullEquality::NullEqualsNull,
        )
        .unwrap();
        assert_eq!(matcher.key_columns(), &[0]);
        assert_eq!(matcher.null_equality(), NullEquality::NullEqualsNull);
        assert!(matcher.layout().is_compatible(&layout));
        let probe = DataChunk::from_record_batch(build.clone());

        // Empty selections match nothing.
        let mut sel = vec![];
        assert_eq!(
            matcher
                .match_rows(&probe, &mut sel, &locations, &collection, None)
                .unwrap(),
            0
        );
        // Out of bounds probe rows and candidates.
        let mut sel = vec![2];
        assert!(
            matcher
                .match_rows(&probe, &mut sel, &locations, &collection, None)
                .is_err()
        );
        let mut sel = vec![0];
        assert!(
            matcher
                .match_rows(
                    &probe,
                    &mut sel,
                    &[RowLocation::new(5, 0)],
                    &collection,
                    None
                )
                .is_err()
        );
        // A probe column with another type.
        let other = DataChunk::from_record_batch(batch(vec![(
            "k",
            Arc::new(StringArray::from(vec!["1", "2"])) as ArrayRef,
        )]));
        let mut sel = vec![0];
        let err = matcher
            .match_rows(&other, &mut sel, &locations, &collection, None)
            .unwrap_err();
        assert!(err.to_string().contains("type"), "{err}");
    }
}
