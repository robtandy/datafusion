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
//!   value (nested nulls are equal to each other, also the nulls of
//!   different union variants, and nested floats are normalized too);
//! * a null and a valid value are never equal;
//! * two nulls are equal with [`NullEquality::NullEqualsNull`]
//!   (`IS NOT DISTINCT FROM`) and not equal with
//!   [`NullEquality::NullEqualsNothing`] (`=`).
//!
//! Nested keys are compared by their row encodings, which are equal exactly
//! when the values are equal, except for keys with floats or unions: the row
//! format distinguishes `-0.0` from `+0.0` and the nulls of different union
//! variants. These keys are compared with Arrow's comparator instead, like
//! `JoinKeyComparator` does, on their decoded values with normalized floats
//! and with dictionaries replaced by their values (which is how dictionaries
//! compare), so that candidates of collections with different dictionaries
//! can be compared together.
//!
//! A probe chunk is usually compared with several candidates per row (e.g.
//! the rows of a hash chain, one per round), so its keys are prepared once
//! ([`RowMatcher::prepare`], DuckDB's `TupleDataVectorFormat`) and then
//! compared with [`RowMatcher::match_prepared`] in every round.

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, AsArray, make_comparator};
use arrow::buffer::{BooleanBuffer, Buffer, NullBuffer};
use arrow::compute::SortOptions;
use arrow::datatypes::DataType;
use arrow_row::Rows;
use datafusion_common::float_semantics::normalize_floats;
use datafusion_common::hash_utils::RandomState;
use datafusion_common::{NullEquality, Result, internal_datafusion_err, internal_err};

use super::collection::{
    DictionaryValues, RowLocation, RowRef, TupleDataCollection, fixed_values,
};
use super::layout::{ColumnLayout, ColumnStorage, TupleDataLayout};
use crate::chunk::{DataChunk, SelectionVector};

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

/// The key columns of a probe chunk, prepared for
/// [`RowMatcher::match_prepared`] by [`RowMatcher::prepare`]: nested values
/// are encoded with the row format (or have their floats normalized; the
/// values of dictionaries once per value, without expanding them to one
/// value per row), and the logical nulls are computed, once per probe chunk.
#[derive(Debug)]
pub struct PreparedProbe {
    /// The number of (logical) rows of the probe chunk.
    len: usize,
    /// The selection of the probe chunk.
    selection: Option<SelectionVector>,
    /// One entry per key column of the matcher.
    columns: Vec<ProbeColumn>,
}

impl PreparedProbe {
    /// The number of (logical) rows of the probe chunk.
    pub fn len(&self) -> usize {
        self.len
    }

    /// Returns `true` if the probe chunk has no rows.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

#[derive(Debug)]
struct ProbeColumn {
    /// The logical nulls of the key, `None` if all keys are valid.
    nulls: Option<NullBuffer>,
    /// For dictionaries, the index in the values of `data` of the value of
    /// every row, see [`DictionaryValues`].
    indices: Option<Vec<usize>>,
    data: ProbeData,
}

/// The values of a probe key column (by physical row, or for dictionaries,
/// the values of the dictionary).
#[derive(Debug)]
enum ProbeData {
    Null,
    Boolean(BooleanBuffer),
    /// The bytes of fixed-width values (`width` bytes per row).
    Fixed(Buffer),
    /// A string or binary array.
    VarLen(ArrayRef),
    /// Nested values encoded with the row format of the column: equal
    /// values have equal encodings.
    Encoded(Rows),
    /// Nested values whose row encodings can differ although they are equal
    /// (see `RowCodec::compare_decoded`), compared with Arrow's comparator
    /// like `JoinKeyComparator` does, with `-0.0` normalized to `+0.0`.
    Compared(ArrayRef),
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

    /// Prepare the key columns of `probe` for [`Self::match_prepared`].
    ///
    /// The key columns of `probe` must have the data types of the key
    /// columns of the layout.
    pub fn prepare(&self, probe: &DataChunk) -> Result<PreparedProbe> {
        for &column in &self.key_columns {
            let probe_type = probe.schema().fields().get(column).map(|f| f.data_type());
            let key_type = self.layout.column(column).data_type();
            if probe_type != Some(key_type) {
                return internal_err!(
                    "Probe column {column} has type {probe_type:?}, but the key has type {key_type}"
                );
            }
        }
        // A chunk without rows can be cleared, i.e. have no columns.
        let columns = if probe.is_empty() {
            vec![]
        } else {
            self.key_columns
                .iter()
                .map(|&column| {
                    ProbeColumn::try_new(self.layout.column(column), probe.column(column))
                })
                .collect::<Result<_>>()?
        };
        Ok(PreparedProbe {
            len: probe.len(),
            selection: probe.selection().cloned(),
            columns,
        })
    }

    /// Compare the keys of the probe rows in `sel` with the keys of their
    /// candidate rows, and keep in `sel` only the rows whose keys are equal.
    ///
    /// This is [`Self::prepare`] followed by [`Self::match_prepared`]; use
    /// these to compare the rows of a probe chunk with several candidates.
    pub fn match_rows(
        &self,
        probe: &DataChunk,
        sel: &mut Vec<u32>,
        candidates: &[RowLocation],
        collection: &TupleDataCollection,
        no_match: Option<&mut Vec<u32>>,
    ) -> Result<usize> {
        if sel.is_empty() {
            return Ok(0);
        }
        let probe = self.prepare(probe)?;
        self.match_prepared(&probe, sel, candidates, collection, no_match)
    }

    /// Compare the keys of the probe rows in `sel` with the keys of their
    /// candidate rows, and keep in `sel` only the rows whose keys are equal.
    ///
    /// * `probe` holds the keys of the probe chunk, see [`Self::prepare`].
    /// * `sel` holds logical row indices of the probe chunk (relative to its
    ///   selection, if it has one). On return it holds the rows that match
    ///   their candidate, in their original order.
    /// * `candidates[row]` is the location of the candidate row in
    ///   `collection` for probe row `row` (as in DuckDB, the candidates are
    ///   indexed by probe row, so only the entries of rows in `sel` are
    ///   read).
    /// * the rows that do not match are appended to `no_match`, if given.
    ///
    /// Returns the number of matching rows.
    pub fn match_prepared(
        &self,
        probe: &PreparedProbe,
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
        if probe.columns.len() != self.key_columns.len() && !probe.is_empty() {
            return internal_err!(
                "The PreparedProbe was not prepared by this RowMatcher"
            );
        }
        let num_rows = probe.len.min(candidates.len());
        if let Some(&row) = sel.iter().find(|&&row| row as usize >= num_rows) {
            return internal_err!(
                "Probe row {row} out of bounds for {} probe rows and {} candidates",
                probe.len,
                candidates.len()
            );
        }
        let mut rows = collection
            .row_refs_of(sel.iter().map(|&row| candidates[row as usize]), sel.len())?;
        let physical = probe.selection.as_ref().map(SelectionVector::indices);
        for (&column, probe_column) in self.key_columns.iter().zip(&probe.columns) {
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
            let layout = self.layout.column(column);
            match probe_column.indices.as_deref() {
                None => match_column(layout, probe_column, &mut matches, |i| i)?,
                Some(indices) => {
                    match_column(layout, probe_column, &mut matches, |i| indices[i])?
                }
            }
        }
        Ok(sel.len())
    }
}

impl ProbeColumn {
    fn try_new(layout: &ColumnLayout, array: &ArrayRef) -> Result<Self> {
        let (values, indices) = if layout.is_dictionary() {
            // Encode (or flatten) only the values that the rows refer to.
            let used_only = matches!(layout.storage(), ColumnStorage::Encoded);
            let DictionaryValues { values, indices } =
                DictionaryValues::try_new(array, used_only)?;
            (values, Some(indices))
        } else {
            (Arc::clone(array), None)
        };
        // The logical nulls of a dictionary are the null keys and the keys of
        // null values (the value of a null key is any value).
        let nulls = array.logical_nulls().filter(|n| n.null_count() > 0);
        let data = match layout.storage() {
            ColumnStorage::Null => ProbeData::Null,
            ColumnStorage::Boolean => {
                ProbeData::Boolean(values.as_boolean().values().clone())
            }
            ColumnStorage::Fixed(width) => ProbeData::Fixed(fixed_values(&values, width)),
            ColumnStorage::VarLen => ProbeData::VarLen(values),
            ColumnStorage::Encoded => {
                let codec = layout.codec().ok_or_else(|| {
                    internal_datafusion_err!("Encoded column without a row codec")
                })?;
                if codec.compare_decoded {
                    // The row format distinguishes `-0.0` and `+0.0`, and
                    // the nulls of different union variants. The values are
                    // compared without dictionaries, as the candidates.
                    ProbeData::Compared(normalize_floats(&codec.flatten(&values)?))
                } else {
                    ProbeData::Encoded(codec.converter.convert_columns(&[values])?)
                }
            }
        };
        Ok(Self {
            nulls,
            indices,
            data,
        })
    }
}

/// Compare a key column (with layout `layout`) of the probe rows with their
/// candidates, where the value of physical probe row `i` is value `index(i)`
/// of `probe`.
fn match_column(
    layout: &ColumnLayout,
    probe: &ProbeColumn,
    matches: &mut ColumnMatches<'_, '_>,
    index: impl Fn(usize) -> usize + Copy,
) -> Result<()> {
    let nulls = probe.nulls.as_ref();
    let offset = layout.offset();
    match &probe.data {
        // Both sides are always null.
        ProbeData::Null => matches.retain(nulls, |_, _, _| true),
        ProbeData::Boolean(values) => matches.retain(nulls, |_, i, row| {
            values.value(index(i)) == (row.row[offset] != 0)
        }),
        ProbeData::Fixed(bytes) => {
            let bytes = bytes.as_slice();
            match layout.value_type() {
                DataType::Float16 => matches.retain(nulls, |_, i, row| {
                    float_bits_eq::<2>(&bytes[index(i) * 2..], &row.row[offset..])
                }),
                DataType::Float32 => matches.retain(nulls, |_, i, row| {
                    float_bits_eq::<4>(&bytes[index(i) * 4..], &row.row[offset..])
                }),
                DataType::Float64 => matches.retain(nulls, |_, i, row| {
                    float_bits_eq::<8>(&bytes[index(i) * 8..], &row.row[offset..])
                }),
                _ => match layout.width() {
                    1 => matches.retain_fixed::<1>(nulls, bytes, offset, index),
                    2 => matches.retain_fixed::<2>(nulls, bytes, offset, index),
                    4 => matches.retain_fixed::<4>(nulls, bytes, offset, index),
                    8 => matches.retain_fixed::<8>(nulls, bytes, offset, index),
                    16 => matches.retain_fixed::<16>(nulls, bytes, offset, index),
                    32 => matches.retain_fixed::<32>(nulls, bytes, offset, index),
                    width => matches.retain(nulls, |_, i, row| {
                        let v = index(i);
                        bytes[v * width..(v + 1) * width]
                            == row.row[offset..offset + width]
                    }),
                },
            }
        }
        ProbeData::VarLen(values) => {
            macro_rules! retain_bytes {
                ($values:expr, $as_bytes:expr) => {{
                    let values = $values;
                    matches.retain(nulls, |_, i, row| {
                        $as_bytes(values.value(index(i))) == row.var_len(offset)
                    })
                }};
            }
            match values.data_type() {
                DataType::Utf8 => retain_bytes!(values.as_string::<i32>(), str::as_bytes),
                DataType::LargeUtf8 => {
                    retain_bytes!(values.as_string::<i64>(), str::as_bytes)
                }
                DataType::Utf8View => {
                    retain_bytes!(values.as_string_view(), str::as_bytes)
                }
                DataType::Binary => retain_bytes!(values.as_binary::<i32>(), identity),
                DataType::LargeBinary => {
                    retain_bytes!(values.as_binary::<i64>(), identity)
                }
                DataType::BinaryView => retain_bytes!(values.as_binary_view(), identity),
                data_type => {
                    return internal_err!(
                        "Unexpected variable-length data type {data_type}"
                    );
                }
            }
        }
        // Equal values have equal row encodings.
        ProbeData::Encoded(encoded) => matches.retain(nulls, |_, i, row| {
            encoded.row(index(i)).data() == row.var_len(offset)
        }),
        // Compare the (normalized) values, like `JoinKeyComparator`.
        ProbeData::Compared(values) => {
            // Without dictionaries: the candidates can come from collections
            // with different dictionaries that would not fit the keys
            // together.
            let codec = layout.codec().ok_or_else(|| {
                internal_datafusion_err!("Encoded column without a row codec")
            })?;
            // Decode every distinct candidate value once: e.g. the candidates
            // of the rows of a dictionary often have the same values, which
            // could be too large for their type once per candidate (e.g.
            // nested run-end encoded values).
            let mut distinct = Vec::new();
            let mut indices_of_encodings = HashMap::with_capacity_and_hasher(
                matches.rows.len(),
                RandomState::default(),
            );
            let candidates = matches
                .rows
                .iter()
                .map(|row| {
                    let encoding = row.var_len(offset);
                    *indices_of_encodings.entry(encoding).or_insert_with(|| {
                        distinct.push(encoding);
                        distinct.len() - 1
                    })
                })
                .collect::<Vec<_>>();
            let comparator = make_comparator(
                values.as_ref(),
                normalize_floats(&codec.decode_flat(distinct)?).as_ref(),
                SortOptions::default(),
            )?;
            matches.retain(nulls, |entry, i, _| {
                comparator(index(i), candidates[entry]).is_eq()
            })
        }
    }
    Ok(())
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

    /// [`Self::retain`] for fixed-width values of `W` bytes (`bytes` holds
    /// the probe values, the value of physical probe row `i` is value
    /// `index(i)`; the candidate values are at `offset`).
    #[inline]
    fn retain_fixed<const W: usize>(
        &mut self,
        nulls: Option<&NullBuffer>,
        bytes: &[u8],
        offset: usize,
        index: impl Fn(usize) -> usize,
    ) {
        self.retain(nulls, |_, i, row| {
            let v = index(i);
            let probe: &[u8; W] = bytes[v * W..(v + 1) * W].try_into().unwrap();
            let candidate: &[u8; W] = row.row[offset..offset + W].try_into().unwrap();
            probe == candidate
        })
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
        // The non-matching rows can be compared with the next candidates,
        // with keys that are prepared once.
        let prepared = matcher.prepare(&probe).unwrap();
        assert_eq!(prepared.len(), 3);
        assert!(!prepared.is_empty());
        let candidates = vec![locations[1], locations[0], locations[0]];
        let mut sel = no_match;
        let mut no_match = vec![];
        let count = matcher
            .match_prepared(
                &prepared,
                &mut sel,
                &candidates,
                &collection,
                Some(&mut no_match),
            )
            .unwrap();
        assert_eq!(count, 1);
        assert_eq!(sel, vec![0]);
        assert_eq!(no_match, vec![2]);
        let candidates = vec![locations[0], locations[0], locations[1]];
        let mut sel = no_match;
        let count = matcher
            .match_prepared(&prepared, &mut sel, &candidates, &collection, None)
            .unwrap();
        assert_eq!(count, 0);
        assert!(sel.is_empty());
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
                    .add_buffer(Buffer::from_slice_ref(&bits))
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

    /// Assert that matching probe row `i` of `probe` with row `pairs[i]` of
    /// `build` agrees with DataFusion's `JoinKeyComparator`, in both null
    /// equality modes, and return the matching rows of both modes.
    fn assert_matches_join_key_comparator(
        build: &ArrayRef,
        probe: &ArrayRef,
        pairs: &[usize],
    ) -> [Vec<u32>; 2] {
        let build_batch = batch(vec![("k", Arc::clone(build))]);
        let probe_chunk =
            DataChunk::from_record_batch(batch(vec![("k", Arc::clone(probe))]));
        [
            NullEquality::NullEqualsNothing,
            NullEquality::NullEqualsNull,
        ]
        .map(|null_equality| {
            let (matched, _) =
                match_pairs(&build_batch, &probe_chunk, pairs, vec![0], null_equality);
            let comparator = JoinKeyComparator::new(
                &[Arc::clone(build)],
                &[Arc::clone(probe)],
                &[SortOptions::default()],
                null_equality,
            )
            .unwrap();
            let expected = (0..pairs.len())
                .filter(|&i| comparator.is_equal(pairs[i], i))
                .map(|i| i as u32)
                .collect::<Vec<_>>();
            assert_eq!(matched, expected, "{null_equality:?}");
            matched
        })
    }

    /// An element of a union: its type id and its value.
    type UnionElement<'a> = (i8, Option<&'a str>);

    /// A sparse `Union<i: Int32, s: Utf8>` of the given `(type id, value)`
    /// elements: type id 0 is the `Int32` variant, 1 the `Utf8` variant, and
    /// `None` is a null value of the variant.
    fn sparse_union(elements: &[UnionElement<'_>]) -> arrow::array::UnionArray {
        let fields = arrow::datatypes::UnionFields::try_new(
            [0, 1],
            [
                Field::new("i", DataType::Int32, true),
                Field::new("s", DataType::Utf8, true),
            ],
        )
        .unwrap();
        let ints = elements
            .iter()
            .map(|&(t, v)| v.filter(|_| t == 0).map(|v| v.parse::<i32>().unwrap()))
            .collect::<Int32Array>();
        let strings = elements
            .iter()
            .map(|&(t, v)| v.filter(|_| t == 1))
            .collect::<StringArray>();
        arrow::array::UnionArray::try_new(
            fields,
            elements.iter().map(|&(t, _)| t).collect(),
            None,
            vec![Arc::new(ints), Arc::new(strings)],
        )
        .unwrap()
    }

    /// A `List<Union<i: Int32, s: Utf8>>` array, see [`sparse_union`].
    fn union_lists(lists: &[Option<Vec<UnionElement<'_>>>]) -> ArrayRef {
        let elements = lists
            .iter()
            .flatten()
            .flatten()
            .copied()
            .collect::<Vec<_>>();
        let union = sparse_union(&elements);
        let field = Arc::new(Field::new_list_field(union.data_type().clone(), true));
        let offsets = arrow::buffer::OffsetBuffer::from_lengths(
            lists.iter().map(|l| l.as_ref().map_or(0, Vec::len)),
        );
        let nulls =
            NullBuffer::from(lists.iter().map(Option::is_some).collect::<Vec<_>>());
        Arc::new(ListArray::new(field, offsets, Arc::new(union), Some(nulls)))
    }

    #[test]
    fn nested_union_nulls_of_different_variants_are_equal() {
        const INT: i8 = 0;
        const STR: i8 = 1;
        let build = union_lists(&[
            Some(vec![(INT, Some("1")), (INT, None)]),
            Some(vec![(STR, None)]),
            Some(vec![(INT, Some("1"))]),
            Some(vec![(INT, None)]),
            None,
            Some(vec![]),
            Some(vec![(STR, Some("x")), (INT, None), (STR, None)]),
        ]);
        let probe = union_lists(&[
            // A null element of the other variant.
            Some(vec![(INT, Some("1")), (STR, None)]),
            Some(vec![(INT, None)]),
            // The same value in another variant.
            Some(vec![(STR, Some("1"))]),
            Some(vec![(INT, Some("2"))]),
            None,
            Some(vec![]),
            Some(vec![(STR, Some("x")), (STR, None), (INT, None)]),
        ]);
        let pairs = [0, 1, 2, 3, 4, 5, 6];
        let [nothing, null] = assert_matches_join_key_comparator(&build, &probe, &pairs);
        // Valid lists whose null elements have different variants are equal.
        assert_eq!(nothing, vec![0, 1, 5, 6]);
        assert_eq!(null, vec![0, 1, 4, 5, 6]);

        // A struct with a union field, and a union key at the top level.
        let structs = |elements: &[(i8, Option<&str>)]| -> ArrayRef {
            Arc::new(
                StructArray::try_new(
                    vec![Field::new(
                        "u",
                        sparse_union(elements).data_type().clone(),
                        true,
                    )]
                    .into(),
                    vec![Arc::new(sparse_union(elements))],
                    None,
                )
                .unwrap(),
            )
        };
        let build = structs(&[(INT, None), (STR, Some("a")), (INT, Some("3"))]);
        let probe = structs(&[(STR, None), (STR, Some("a")), (STR, Some("3"))]);
        let [nothing, null] =
            assert_matches_join_key_comparator(&build, &probe, &[0, 1, 2]);
        assert_eq!(nothing, vec![0, 1]);
        assert_eq!(null, vec![0, 1]);
        let build: ArrayRef = Arc::new(sparse_union(&[(INT, None), (STR, Some("a"))]));
        let probe: ArrayRef = Arc::new(sparse_union(&[(STR, None), (STR, Some("a"))]));
        let [nothing, null] = assert_matches_join_key_comparator(&build, &probe, &[0, 1]);
        assert_eq!(nothing, vec![1]);
        assert_eq!(null, vec![0, 1]);
    }

    /// Nested key types with nulls at all levels, sampled with repetitions
    /// so that equal keys exist, agree with `JoinKeyComparator` on all
    /// pairs. Covers the row encoding comparison (nested types without
    /// floats and unions) and the comparator fallback (with unions).
    #[test]
    fn sampled_nested_keys_match_join_key_comparator() {
        use arrow::array::{
            FixedSizeListArray, Int32Builder, ListBuilder, ListViewArray, MapBuilder,
            StringBuilder, StringDictionaryBuilder, UnionArray,
        };
        use arrow::buffer::OffsetBuffer;
        use arrow::datatypes::{Int32Type, UnionFields};

        const INT: i8 = 0;
        const STR: i8 = 1;
        // Lists of integers, with different bytes under null elements.
        let ints = Int32Array::new(
            vec![1, 77, 1, 0, 5, -3, 9].into(),
            Some(NullBuffer::from(vec![
                true, false, true, false, true, false, false,
            ])),
        );
        let int_lists: ArrayRef = Arc::new(ListArray::new(
            Arc::new(Field::new_list_field(DataType::Int32, true)),
            OffsetBuffer::from_lengths([2, 2, 0, 1, 2, 0]),
            Arc::new(ints),
            Some(NullBuffer::from(vec![true, true, true, true, true, false])),
        ));
        // Structs with a nested list, null structs with different children.
        let fields = Fields::from(vec![
            Field::new("i", DataType::Int32, true),
            Field::new("s", DataType::Utf8, true),
            Field::new("l", DataType::new_list(DataType::Int32, true), true),
        ]);
        let structs: ArrayRef = Arc::new(
            StructArray::try_new(
                fields,
                vec![
                    Arc::new(Int32Array::from(vec![
                        Some(1),
                        None,
                        Some(1),
                        Some(2),
                        None,
                        Some(1),
                    ])),
                    Arc::new(StringArray::from(vec![
                        Some("a"),
                        Some("a"),
                        None,
                        Some("b"),
                        None,
                        Some("a"),
                    ])),
                    int_lists.slice(0, 6),
                ],
                Some(NullBuffer::from(vec![true, true, true, false, false, true])),
            )
            .unwrap(),
        );
        // Maps of strings to lists.
        let mut maps = MapBuilder::new(
            None,
            StringBuilder::new(),
            ListBuilder::new(Int32Builder::new()),
        );
        for i in 0..6 {
            for j in 0..i % 3 {
                maps.keys().append_value(format!("k{j}"));
                if (i + j) % 4 == 0 {
                    maps.values().append_null();
                } else {
                    maps.values().values().append_option((j != 1).then_some(j));
                    maps.values().append(true);
                }
            }
            maps.append(i != 4).unwrap();
        }
        let maps: ArrayRef = Arc::new(maps.finish());
        // Fixed size lists and list views of dictionaries.
        let fixed: ArrayRef = Arc::new(
            FixedSizeListArray::try_new(
                Arc::new(Field::new_list_field(DataType::Utf8, true)),
                2,
                Arc::new(StringArray::from(vec![
                    Some("x"),
                    None,
                    Some("x"),
                    None,
                    None,
                    None,
                    Some("y"),
                    Some("x"),
                    None,
                    Some("z"),
                    Some("x"),
                    None,
                ])),
                Some(NullBuffer::from(vec![true, true, true, true, false, true])),
            )
            .unwrap(),
        );
        let mut dictionaries =
            ListBuilder::new(StringDictionaryBuilder::<Int8Type>::new());
        for i in 0..6 {
            for j in 0..i % 3 {
                dictionaries
                    .values()
                    .append_option((j != 1).then(|| format!("d{}", i % 2)));
            }
            dictionaries.append(i != 2);
        }
        let views: ArrayRef = Arc::new(ListViewArray::from(dictionaries.finish()));
        // Lists of sparse and dense unions with nulls of both variants.
        let union_fields = UnionFields::try_new(
            [INT, STR],
            [
                Field::new("i", DataType::Int32, true),
                Field::new("s", DataType::Utf8, true),
            ],
        )
        .unwrap();
        let elements = [
            (INT, Some("1")),
            (INT, None),
            (STR, None),
            (STR, Some("1")),
            (INT, None),
            (STR, None),
            (INT, Some("1")),
            (STR, None),
            (INT, None),
            (STR, Some("a")),
        ];
        let sparse = sparse_union(&elements);
        let dense_type_ids = elements
            .iter()
            .map(|&(t, _)| t)
            .collect::<arrow::buffer::ScalarBuffer<i8>>();
        let mut dense_offsets = vec![];
        let (mut ints, mut strings) = (vec![], vec![]);
        for &(t, v) in &elements {
            if t == INT {
                dense_offsets.push(ints.len() as i32);
                ints.push(v.map(|v| v.parse::<i32>().unwrap()));
            } else {
                dense_offsets.push(strings.len() as i32);
                strings.push(v);
            }
        }
        let dense = UnionArray::try_new(
            union_fields.clone(),
            dense_type_ids,
            Some(dense_offsets.into()),
            vec![
                Arc::new(Int32Array::from(ints)),
                Arc::new(StringArray::from(strings)),
            ],
        )
        .unwrap();
        let lists_of = |union: UnionArray| -> ArrayRef {
            Arc::new(ListArray::new(
                Arc::new(Field::new_list_field(union.data_type().clone(), true)),
                OffsetBuffer::from_lengths([2, 1, 1, 2, 0, 1, 1, 2]),
                Arc::new(union),
                Some(NullBuffer::from(vec![
                    true, true, true, true, true, false, true, true,
                ])),
            ))
        };
        let mut union_maps =
            MapBuilder::new(None, StringBuilder::new(), Int32Builder::new());
        for i in 0..6 {
            for j in 0..i % 3 {
                union_maps.keys().append_value(format!("k{j}"));
                union_maps.values().append_option((i % 2 == 0).then_some(j));
            }
            union_maps.append(true).unwrap();
        }
        let union_maps = union_maps.finish();
        let unions_in_maps: ArrayRef = {
            // The same map, with union values.
            let entries = union_maps.entries();
            let values = entries.column(1).as_primitive::<Int32Type>();
            let union = UnionArray::try_new(
                union_fields,
                (0..values.len())
                    .map(|i| if i % 2 == 0 { INT } else { STR })
                    .collect(),
                None,
                vec![
                    Arc::new(values.clone()),
                    Arc::new(StringArray::from(vec![None::<&str>; values.len()])),
                ],
            )
            .unwrap();
            let entry_fields = Fields::from(vec![
                Field::new("keys", DataType::Utf8, false),
                Field::new("values", union.data_type().clone(), true),
            ]);
            let entries = StructArray::try_new(
                entry_fields,
                vec![Arc::clone(entries.column(0)), Arc::new(union)],
                None,
            )
            .unwrap();
            Arc::new(
                arrow::array::MapArray::try_new(
                    Arc::new(Field::new("entries", entries.data_type().clone(), false)),
                    union_maps.offsets().clone(),
                    entries,
                    union_maps.nulls().cloned(),
                    false,
                )
                .unwrap(),
            )
        };

        let keys = [
            ("lists", int_lists, false),
            ("structs", structs, false),
            ("maps", maps, false),
            ("fixed size lists", fixed, false),
            ("list views of dictionaries", views, false),
            ("lists of sparse unions", lists_of(sparse), true),
            ("lists of dense unions", lists_of(dense), true),
            ("maps of unions", unions_in_maps, true),
        ];
        for (name, values, compare_decoded) in keys {
            let layout =
                TupleDataLayout::try_new_from_types([values.data_type().clone()], false)
                    .unwrap();
            assert_eq!(
                layout.column(0).codec().unwrap().compare_decoded,
                compare_decoded,
                "{name}"
            );
            // Sample the values with repetitions.
            let sample = |seed: u64| {
                let indices = (0..3 * values.len() as u64)
                    .map(|i| {
                        let hash = (i + seed).wrapping_mul(0x9E37_79B9_7F4A_7C15);
                        ((hash >> 32) % values.len() as u64) as u32
                    })
                    .collect::<arrow::array::UInt32Array>();
                arrow::compute::take(values.as_ref(), &indices, None).unwrap()
            };
            let (build, probe) = (sample(1), sample(1000));
            let mut equal_pairs = 0;
            for shift in 0..build.len() {
                let pairs = (0..probe.len())
                    .map(|i| (i + shift) % build.len())
                    .collect::<Vec<_>>();
                let [nothing, _] =
                    assert_matches_join_key_comparator(&build, &probe, &pairs);
                equal_pairs += nothing.len();
            }
            assert!(equal_pairs > 0, "{name}: no equal keys");
        }
    }

    #[test]
    fn dictionaries_of_nested_keys() {
        use arrow::array::Int8Array;

        const INT: i8 = 0;
        const STR: i8 = 1;
        // Dictionaries of unions and of lists of unions, with null keys and
        // null values: the values are compared, the unions by value.
        let unions: ArrayRef = Arc::new(sparse_union(&[
            (INT, None),
            (STR, None),
            (STR, Some("a")),
            (INT, Some("1")),
        ]));
        let lists = union_lists(&[
            Some(vec![(INT, None)]),
            Some(vec![(STR, None)]),
            None,
            Some(vec![(STR, Some("a"))]),
        ]);
        // The expected matches for `NullEqualsNothing` and `NullEqualsNull`.
        let cases = [
            // [NULL(i), "a", NULL, 1, NULL(s), NULL(i)] and
            // [NULL(s), "a", NULL, 1, NULL(i), NULL].
            (unions, vec![1, 3], vec![0, 1, 2, 3, 4, 5]),
            // [[NULL(i)], NULL, NULL, ["a"], [NULL(s)], [NULL(i)]] and
            // [[NULL(s)], NULL, NULL, ["a"], [NULL(i)], NULL]: lists with
            // nulls of different variants are equal.
            (lists, vec![0, 3, 4], vec![0, 1, 2, 3, 4, 5]),
        ];
        for (values, expected_nothing, expected_null) in cases {
            let dictionary = |keys: Vec<Option<i8>>| -> ArrayRef {
                Arc::new(
                    DictionaryArray::<Int8Type>::try_new(
                        Int8Array::from(keys),
                        Arc::clone(&values),
                    )
                    .unwrap(),
                )
            };
            let build =
                dictionary(vec![Some(0), Some(2), None, Some(3), Some(1), Some(0)]);
            let probe = dictionary(vec![Some(1), Some(2), None, Some(3), Some(0), None]);
            let [nothing, null] =
                assert_matches_join_key_comparator(&build, &probe, &[0, 1, 2, 3, 4, 2]);
            assert_eq!(nothing, expected_nothing);
            assert_eq!(null, expected_null);
        }
    }

    #[test]
    fn exotic_keys_match_join_key_comparator() {
        for (name, keys, _) in crate::tuple_data::test_data::exotic_arrays() {
            for shift in 0..keys.len() {
                let pairs = (0..keys.len())
                    .map(|i| (i + shift) % keys.len())
                    .collect::<Vec<_>>();
                let [_, null] = assert_matches_join_key_comparator(&keys, &keys, &pairs);
                if shift == 0 {
                    // Every row is IS NOT DISTINCT FROM itself.
                    assert_eq!(null.len(), keys.len(), "{name}");
                }
            }
        }
    }

    #[test]
    fn nested_dictionary_keys_of_different_batches() {
        use arrow::array::{Int8Array, UnionArray};
        use arrow::buffer::OffsetBuffer;
        use arrow::datatypes::UnionFields;

        // Keys with float dictionaries in structs, lists and unions: the
        // candidates come from two batches with disjoint dictionaries that
        // together have more values than Int8 keys can address, which must
        // not matter for comparing them.
        let dictionary = |keys: Vec<i8>, values: Vec<f64>| -> ArrayRef {
            Arc::new(
                DictionaryArray::<Int8Type>::try_new(
                    Int8Array::from(keys),
                    Arc::new(Float64Array::from(values)),
                )
                .unwrap(),
            )
        };
        let structs = |values: ArrayRef| -> ArrayRef {
            Arc::new(
                StructArray::try_new(
                    vec![Field::new("d", values.data_type().clone(), true)].into(),
                    vec![values],
                    None,
                )
                .unwrap(),
            )
        };
        let lists = |values: ArrayRef| -> ArrayRef {
            Arc::new(ListArray::new(
                Arc::new(Field::new_list_field(values.data_type().clone(), true)),
                OffsetBuffer::from_lengths(std::iter::repeat_n(1, values.len())),
                values,
                None,
            ))
        };
        let unions = |values: ArrayRef| -> ArrayRef {
            let n = values.len();
            Arc::new(
                UnionArray::try_new(
                    UnionFields::try_new(
                        [0, 1],
                        [
                            Field::new("d", values.data_type().clone(), true),
                            Field::new("i", DataType::Int32, true),
                        ],
                    )
                    .unwrap(),
                    vec![0; n].into(),
                    None,
                    vec![values, Arc::new(Int32Array::from(vec![1; n]))],
                )
                .unwrap(),
            )
        };
        let keys = (0..=127).collect::<Vec<i8>>();
        for wrap in [structs, lists, unions] {
            let first = wrap(dictionary(keys.clone(), (0..128).map(f64::from).collect()));
            let second = wrap(dictionary(
                keys.clone(),
                (128..256).map(f64::from).collect(),
            ));
            // -0.0 equals 0.0.
            let probe = wrap(dictionary(vec![0; 256], vec![-0.0]));
            let layout = Arc::new(
                TupleDataLayout::try_new_from_types([first.data_type().clone()], false)
                    .unwrap(),
            );
            let mut collection = TupleDataCollection::new(Arc::clone(&layout));
            let mut candidates = vec![];
            for build in [&first, &second] {
                let batch = RecordBatch::try_new(
                    Arc::clone(layout.schema()),
                    vec![Arc::clone(build)],
                )
                .unwrap();
                candidates.extend(
                    collection
                        .append_chunk(&DataChunk::from_record_batch(batch), None)
                        .unwrap(),
                );
            }
            let probe_chunk = DataChunk::from_record_batch(
                RecordBatch::try_new(
                    Arc::clone(layout.schema()),
                    vec![Arc::clone(&probe)],
                )
                .unwrap(),
            );
            for null_equality in [
                NullEquality::NullEqualsNothing,
                NullEquality::NullEqualsNull,
            ] {
                let matcher =
                    RowMatcher::try_new(Arc::clone(&layout), vec![0], null_equality)
                        .unwrap();
                let mut sel = (0..256).collect::<Vec<u32>>();
                matcher
                    .match_rows(&probe_chunk, &mut sel, &candidates, &collection, None)
                    .unwrap();
                // Probe row `i` is compared with build row `i`.
                let comparators = [&first, &second].map(|build| {
                    JoinKeyComparator::new(
                        &[Arc::clone(build)],
                        &[Arc::clone(&probe)],
                        &[SortOptions::default()],
                        null_equality,
                    )
                    .unwrap()
                });
                let expected = (0..256)
                    .filter(|&i| comparators[i / 128].is_equal(i % 128, i))
                    .map(|i| i as u32)
                    .collect::<Vec<_>>();
                let name = first.data_type();
                assert_eq!(expected, vec![0], "{name} {null_equality:?}");
                assert_eq!(sel, expected, "{name} {null_equality:?}");
            }
        }
    }

    /// Randomized keys with float dictionaries (with nulls, `-0.0` and
    /// `NaN`) in structs, lists and sparse unions, whose candidates come
    /// from two batches with different dictionaries, agree with
    /// `JoinKeyComparator` on all pairs.
    #[test]
    fn randomized_dictionary_keys_of_different_batches() {
        use arrow::array::{Int8Array, UnionArray};
        use arrow::buffer::{OffsetBuffer, ScalarBuffer};
        use arrow::datatypes::UnionFields;
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        fn dictionary(rng: &mut StdRng, n: usize) -> ArrayRef {
            let size = rng.random_range(1..6);
            let values = (0..size)
                .map(|_| match rng.random_range(0..8) {
                    0 => None,
                    1 => Some(-0.0),
                    2 => Some(0.0),
                    3 => Some(f64::NAN),
                    _ => Some(f64::from(rng.random_range(0..4))),
                })
                .collect::<Float64Array>();
            let keys = (0..n)
                .map(|_| (!rng.random_bool(0.1)).then(|| rng.random_range(0..size) as i8))
                .collect::<Int8Array>();
            Arc::new(
                DictionaryArray::<Int8Type>::try_new(keys, Arc::new(values)).unwrap(),
            )
        }
        fn keys(rng: &mut StdRng, case: usize, n: usize) -> ArrayRef {
            let nulls = |rng: &mut StdRng| {
                NullBuffer::from(
                    (0..n).map(|_| !rng.random_bool(0.15)).collect::<Vec<_>>(),
                )
            };
            match case {
                0 => {
                    let values = dictionary(rng, n);
                    let fields = vec![Field::new("d", values.data_type().clone(), true)];
                    Arc::new(
                        StructArray::try_new(
                            fields.into(),
                            vec![values],
                            Some(nulls(rng)),
                        )
                        .unwrap(),
                    )
                }
                1 => {
                    let lengths =
                        (0..n).map(|_| rng.random_range(0..3)).collect::<Vec<_>>();
                    let values = dictionary(rng, lengths.iter().sum());
                    Arc::new(ListArray::new(
                        Arc::new(Field::new_list_field(values.data_type().clone(), true)),
                        OffsetBuffer::from_lengths(lengths),
                        values,
                        Some(nulls(rng)),
                    ))
                }
                _ => {
                    let values = dictionary(rng, n);
                    let fields = UnionFields::try_new(
                        [0, 1],
                        [
                            Field::new("d", values.data_type().clone(), true),
                            Field::new("i", DataType::Int32, true),
                        ],
                    )
                    .unwrap();
                    let type_ids = (0..n)
                        .map(|_| i8::from(rng.random_bool(0.3)))
                        .collect::<ScalarBuffer<i8>>();
                    let ints = (0..n)
                        .map(|_| (!rng.random_bool(0.3)).then(|| rng.random_range(0..2)))
                        .collect::<Int32Array>();
                    Arc::new(
                        UnionArray::try_new(
                            fields,
                            type_ids,
                            None,
                            vec![values, Arc::new(ints)],
                        )
                        .unwrap(),
                    )
                }
            }
        }

        for seed in 0..100 {
            let mut rng = StdRng::seed_from_u64(seed);
            let case = rng.random_range(0..3);
            let n = rng.random_range(1..30);
            let builds = [keys(&mut rng, case, n), keys(&mut rng, case, n)];
            let probe = keys(&mut rng, case, 2 * n);
            let layout = Arc::new(
                TupleDataLayout::try_new_from_types(
                    [builds[0].data_type().clone()],
                    false,
                )
                .unwrap(),
            );
            let mut collection = TupleDataCollection::new(Arc::clone(&layout));
            let mut locations = vec![];
            for build in &builds {
                let batch = RecordBatch::try_new(
                    Arc::clone(layout.schema()),
                    vec![Arc::clone(build)],
                )
                .unwrap();
                locations.extend(
                    collection
                        .append_chunk(&DataChunk::from_record_batch(batch), None)
                        .unwrap(),
                );
            }
            let probe_chunk = DataChunk::from_record_batch(
                RecordBatch::try_new(
                    Arc::clone(layout.schema()),
                    vec![Arc::clone(&probe)],
                )
                .unwrap(),
            );
            for null_equality in [
                NullEquality::NullEqualsNothing,
                NullEquality::NullEqualsNull,
            ] {
                let matcher =
                    RowMatcher::try_new(Arc::clone(&layout), vec![0], null_equality)
                        .unwrap();
                let prepared = matcher.prepare(&probe_chunk).unwrap();
                let comparators = [&builds[0], &builds[1]].map(|build| {
                    JoinKeyComparator::new(
                        &[Arc::clone(build)],
                        &[Arc::clone(&probe)],
                        &[SortOptions::default()],
                        null_equality,
                    )
                    .unwrap()
                });
                // Probe row `i` is compared with build row `(i + shift) % 2n`.
                for shift in 0..2 * n {
                    let pairs = (0..2 * n)
                        .map(|i| (i + shift) % (2 * n))
                        .collect::<Vec<_>>();
                    let candidates =
                        pairs.iter().map(|&row| locations[row]).collect::<Vec<_>>();
                    let mut sel = (0..2 * n as u32).collect::<Vec<_>>();
                    matcher
                        .match_prepared(
                            &prepared,
                            &mut sel,
                            &candidates,
                            &collection,
                            None,
                        )
                        .unwrap();
                    let expected = (0..2 * n)
                        .filter(|&i| comparators[pairs[i] / n].is_equal(pairs[i] % n, i))
                        .map(|i| i as u32)
                        .collect::<Vec<_>>();
                    assert_eq!(
                        sel, expected,
                        "seed {seed}, case {case}, {null_equality:?}, shift {shift}"
                    );
                }
            }
        }
    }

    #[test]
    fn dictionary_keys_with_null_keys_of_values_without_nulls() {
        use crate::tuple_data::test_data::dictionaries_of_values_without_nulls;

        for keys in dictionaries_of_values_without_nulls() {
            for shift in 0..keys.len() {
                let pairs = (0..keys.len())
                    .map(|i| (i + shift) % keys.len())
                    .collect::<Vec<_>>();
                let [nothing, null] =
                    assert_matches_join_key_comparator(&keys, &keys, &pairs);
                if shift == 0 {
                    // The null key only matches with `IS NOT DISTINCT FROM`.
                    assert_eq!(nothing, vec![1, 2]);
                    assert_eq!(null, vec![0, 1, 2]);
                }
            }
            // A probe chunk whose selection excludes the null key.
            let build = batch(vec![("k", Arc::clone(&keys))]);
            let probe = chunk_with_selection(&build, vec![2, 1]);
            let (matched, unmatched) = match_pairs(
                &build,
                &probe,
                &[2, 0],
                vec![0],
                NullEquality::NullEqualsNothing,
            );
            assert_eq!(matched, vec![0]);
            assert_eq!(unmatched, vec![1]);
        }
    }

    #[test]
    fn dictionary_keys_of_run_end_encoded_lists() {
        use crate::tuple_data::test_data::dictionaries_of_run_end_encoded_lists;

        for keys in dictionaries_of_run_end_encoded_lists() {
            let len = keys.len();
            for shift in [0, 1, 3, 1000] {
                let pairs = (0..len).map(|i| (i + shift) % len).collect::<Vec<_>>();
                let [_, null] = assert_matches_join_key_comparator(&keys, &keys, &pairs);
                if shift == 0 {
                    // Every row is `IS NOT DISTINCT FROM` itself.
                    assert_eq!(null.len(), len);
                }
            }
            // Probe slices with fewer rows than dictionary values (some keys
            // null), so that only the values that they refer to are encoded.
            for (offset, len) in [(0, 1), (1, 2), (5, 2)] {
                let probe = keys.slice(offset, len);
                for shift in [0, 1, 2] {
                    let pairs = (0..len).map(|i| offset + i + shift).collect::<Vec<_>>();
                    assert_matches_join_key_comparator(&keys, &probe, &pairs);
                }
            }
            // A probe chunk whose selection excludes almost all rows.
            let build = batch(vec![("k", Arc::clone(&keys))]);
            let probe = chunk_with_selection(&build, vec![1, 0]);
            let build_rows = [0, 3];
            let comparator = JoinKeyComparator::new(
                &[Arc::clone(&keys)],
                &[Arc::clone(&keys)],
                &[SortOptions::default()],
                NullEquality::NullEqualsNothing,
            )
            .unwrap();
            let expected = [1, 0]
                .into_iter()
                .zip(build_rows)
                .enumerate()
                .filter(|&(_, (probe_row, build_row))| {
                    comparator.is_equal(build_row, probe_row)
                })
                .map(|(row, _)| row as u32)
                .collect::<Vec<_>>();
            let (matched, _) = match_pairs(
                &build,
                &probe,
                &build_rows,
                vec![0],
                NullEquality::NullEqualsNothing,
            );
            assert_eq!(matched, expected);
        }
    }

    #[test]
    fn run_end_encoded_keys_with_custom_fields() {
        use crate::tuple_data::test_data::custom_run_end_encoded;

        // Compared with their row encodings.
        let strings = |values: Vec<Option<&str>>, run_ends: &[i32]| {
            custom_run_end_encoded(Arc::new(StringArray::from(values)), run_ends, true)
        };
        // [a, a, NULL, NULL, b, b] and [a, b, b, NULL, NULL, NULL].
        let build = strings(vec![Some("a"), None, Some("b")], &[2, 4, 6]);
        let probe = strings(vec![Some("a"), Some("b"), None], &[1, 3, 6]);
        let [nothing, null] =
            assert_matches_join_key_comparator(&build, &probe, &[1, 4, 5, 2, 3, 0]);
        assert_eq!(nothing, vec![0, 1, 2]);
        assert_eq!(null, vec![0, 1, 2, 3, 4]);

        // With floats, compared by value: -0.0 = 0.0 and NaN = NaN.
        let floats = |values: Vec<Option<f64>>, run_ends: &[i32]| {
            custom_run_end_encoded(Arc::new(Float64Array::from(values)), run_ends, true)
        };
        // [-0.0, -0.0, NULL, NaN, NaN] and [0.0, NaN, NaN, NULL, NULL].
        let build = floats(vec![Some(-0.0), None, Some(f64::NAN)], &[2, 3, 5]);
        let probe = floats(vec![Some(0.0), Some(f64::NAN), None], &[1, 3, 5]);
        let [nothing, null] =
            assert_matches_join_key_comparator(&build, &probe, &[1, 3, 4, 2, 0]);
        assert_eq!(nothing, vec![0, 1, 2]);
        assert_eq!(null, vec![0, 1, 2, 3]);
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
        let prepared = matcher.prepare(&probe_chunk).unwrap();
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
                .match_prepared(&prepared, &mut sel, &candidates, &collection, None)
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

        // Keys prepared by another matcher.
        let no_keys = RowMatcher::try_new(
            Arc::clone(&layout),
            vec![],
            NullEquality::NullEqualsNull,
        )
        .unwrap();
        let prepared = no_keys.prepare(&probe).unwrap();
        let mut sel = vec![0];
        let err = matcher
            .match_prepared(&prepared, &mut sel, &locations, &collection, None)
            .unwrap_err();
        assert!(err.to_string().contains("not prepared by this"), "{err}");
        // Without key columns, all rows match.
        let mut sel = vec![1, 0];
        assert_eq!(
            no_keys
                .match_prepared(&prepared, &mut sel, &locations, &collection, None)
                .unwrap(),
            2
        );
        assert_eq!(sel, vec![1, 0]);

        // A cleared probe chunk has no rows.
        let cleared = matcher.prepare(&DataChunk::empty(build.schema())).unwrap();
        assert!(cleared.is_empty());
        let mut sel = vec![];
        assert_eq!(
            matcher
                .match_prepared(&cleared, &mut sel, &locations, &collection, None)
                .unwrap(),
            0
        );
        let mut sel = vec![0];
        assert!(
            matcher
                .match_prepared(&cleared, &mut sel, &locations, &collection, None)
                .is_err()
        );
    }
}
