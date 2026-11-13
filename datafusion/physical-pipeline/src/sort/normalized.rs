// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Normalized keys: sorting and comparing sort keys by fixed-width integers.
//!
//! The sort keys that are ordered by their bytes, the keys of several sort
//! keys (or of a key of a nested or encoded type) in the Arrow row format,
//! and the values of a single string or binary key, are sorted by their
//! *normalized keys*, as DuckDB sorts its rows: the first 8, 16, 24 or 32
//! bytes of every key as big-endian 64-bit words, padded with zeros, next to
//! the position of its row. If the normalized keys of two keys differ, they
//! order them like their bytes do: at the first byte in which the normalized
//! keys differ, either both keys have a byte, or the key that ends there is a
//! prefix of the other one (whose byte is not a zero of the padding), which
//! comes after it. So the sorts compare inline integers rather than looking
//! up the rows and calling `memcmp` on their bytes, and compare the bytes of
//! two keys only if their normalized keys are equal and do not decide:
//!
//! * encoded rows of the same length that fit into the normalized key (e.g.
//!   the 18 bytes of two `Int64` keys in 24) never compare their bytes, and
//!   the position of the row fills the last 4 bytes of the normalized key if
//!   they are padding: such keys are sorted as plain words (see [`Key`]);
//! * strings and binary values of at most 16 bytes are ordered by their
//!   first 16 bytes and their length, and so are their positions;
//! * longer keys compare their remaining bytes if their normalized keys are
//!   equal.
//!
//! [`compare_bytes`] compares the keys of the runs that the merge compares
//! (see [`super::merge`]) in the same way, word by word, inline.

use std::cmp::Ordering;

use arrow::array::{Array, AsArray};
use arrow::buffer::NullBuffer;
use arrow::compute::SortOptions;
use arrow::datatypes::DataType;
use arrow_row::{RowConverter, Rows};
use datafusion_common::Result;

use super::ReservedMemory;

/// The most 64-bit words of the normalized keys of encoded rows (32 bytes).
const MAX_WORDS: usize = 4;

/// The bytes of a normalized key that hold the position of a row, if they
/// are padding of the bytes of the keys.
const POSITION_BYTES: usize = size_of::<u32>();

/// The most words that [`compare_bytes`] compares inline before it compares
/// the remaining bytes with `memcmp`.
const INLINE_WORDS: usize = 4;

/// The bytes of the normalized keys of strings and binary values, whose
/// lengths decide among the values of at most as many bytes.
const VALUE_PREFIX_BYTES: usize = 16;

/// A normalized key of `N` words: ordered by its words, lexicographically,
/// and compared by `<` without branches, as the sorts compare (`sort_unstable`
/// and `select_nth_unstable` compare with `<` only), so that the sorts can
/// partition without mispredicting branches.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Key<const N: usize>([u64; N]);

impl<const N: usize> Ord for Key<N> {
    #[inline]
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.cmp(&other.0)
    }
}

impl<const N: usize> PartialOrd for Key<N> {
    #[inline]
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }

    #[inline]
    fn lt(&self, other: &Self) -> bool {
        let mut less = false;
        for (a, b) in self.0.iter().zip(&other.0).rev() {
            less = (a < b) | ((a == b) & less);
        }
        less
    }
}

/// The first bytes of `bytes` (all of them if there are fewer than 8) as the
/// leading bytes of a big-endian word, padded with zeros.
#[inline]
fn partial_word(bytes: &[u8]) -> u64 {
    bytes
        .iter()
        .take(8)
        .enumerate()
        .fold(0, |word, (i, &byte)| {
            word | (u64::from(byte) << (56 - 8 * i))
        })
}

/// The normalized key of `bytes`: its first `8 * N` bytes as `N` big-endian
/// words, padded with zeros.
#[inline]
fn normalized_key<const N: usize>(bytes: &[u8]) -> [u64; N] {
    let mut key = [0; N];
    let mut rest = bytes;
    for word in &mut key {
        match rest.split_first_chunk::<8>() {
            Some((chunk, tail)) => {
                *word = u64::from_be_bytes(*chunk);
                rest = tail;
            }
            None => {
                *word = partial_word(rest);
                break;
            }
        }
    }
    key
}

/// The order of the bytes `a` and `b`, as `a.cmp(b)`: compared as big-endian
/// words, inline, for their first `8 * INLINE_WORDS` bytes, rather than with
/// a call of `memcmp` (only longer bytes call it for the rest).
#[inline]
pub(crate) fn compare_bytes(a: &[u8], b: &[u8]) -> Ordering {
    let (mut a, mut b) = (a, b);
    for _ in 0..INLINE_WORDS {
        match (a.split_first_chunk::<8>(), b.split_first_chunk::<8>()) {
            (Some((word_a, rest_a)), Some((word_b, rest_b))) => {
                let (word_a, word_b) =
                    (u64::from_be_bytes(*word_a), u64::from_be_bytes(*word_b));
                if word_a != word_b {
                    return word_a.cmp(&word_b);
                }
                (a, b) = (rest_a, rest_b);
            }
            _ => {
                // One of them ends in its next word: unless the padded words
                // differ, it is a prefix of the other one.
                return partial_word(a)
                    .cmp(&partial_word(b))
                    .then(a.len().cmp(&b.len()));
            }
        }
    }
    a.cmp(b)
}

/// The first `len` rows of `rows` (all rows if there are not as many), sorted
/// by their bytes with normalized keys: their positions in `rows`, and, if
/// `copy` is the converter that encoded them, a copy of the sorted rows.
///
/// The normalized keys, the positions and the copy are reserved in `working`
/// before they are allocated.
pub(crate) fn sort_rows(
    rows: &Rows,
    len: usize,
    copy: Option<&RowConverter>,
    working: &mut ReservedMemory<'_>,
) -> Result<(Vec<u32>, Option<Rows>)> {
    let len = len.min(rows.num_rows());
    if len == 0 {
        return Ok((vec![], copy.map(|converter| converter.empty_rows(0, 0))));
    }
    let (min_len, max_len) =
        rows.lengths().fold((usize::MAX, 0), |(min, max), row_len| {
            (min.min(row_len), max.max(row_len))
        });
    let same_len = (min_len == max_len).then_some(max_len);
    let sort = RowSort {
        rows,
        len,
        copy,
        same_len,
    };
    match same_len {
        Some(row_len) if row_len + POSITION_BYTES <= 8 * MAX_WORDS => {
            match (row_len + POSITION_BYTES).div_ceil(8) {
                1 => sort.packed::<1>(working),
                2 => sort.packed::<2>(working),
                3 => sort.packed::<3>(working),
                _ => sort.packed::<4>(working),
            }
        }
        _ => match max_len.div_ceil(8) {
            0 | 1 => sort.with_positions::<1>(working),
            2 => sort.with_positions::<2>(working),
            3 => sort.with_positions::<3>(working),
            _ => sort.with_positions::<4>(working),
        },
    }
}

/// A sort of encoded rows, see [`sort_rows`].
struct RowSort<'a> {
    rows: &'a Rows,
    /// The number of rows to keep.
    len: usize,
    copy: Option<&'a RowConverter>,
    /// The length of all rows, if they have the same length.
    same_len: Option<usize>,
}

impl RowSort<'_> {
    /// Sort rows of the same length that leave room for their position in
    /// their normalized keys of `N` words, which hold all of their bytes:
    /// the keys are sorted as words.
    fn packed<const N: usize>(
        &self,
        working: &mut ReservedMemory<'_>,
    ) -> Result<(Vec<u32>, Option<Rows>)> {
        let row_len = self.same_len.unwrap_or_default();
        working.try_grow(self.rows.num_rows() * size_of::<Key<N>>())?;
        let mut keys = self
            .rows
            .iter()
            .zip(0_u64..)
            .map(|(row, position)| {
                let mut key = normalized_key::<N>(row.data());
                key[N - 1] |= position;
                Key(key)
            })
            .collect::<Vec<_>>();
        if self.len < keys.len() {
            keys.select_nth_unstable(self.len - 1);
            keys.truncate(self.len);
        }
        keys.sort_unstable();
        working.try_grow(self.len * size_of::<u32>())?;
        let order = keys.iter().map(|key| key.0[N - 1] as u32).collect();
        let copy = self
            .copy
            .map(|converter| {
                copy_of_keys(converter, keys.iter().map(|key| &key.0), row_len, working)
            })
            .transpose()?;
        Ok((order, copy))
    }

    /// Sort rows by their normalized keys of `N` words, next to their
    /// positions: rows whose keys are equal compare their bytes, unless the
    /// keys hold all bytes of all rows (of the same length).
    fn with_positions<const N: usize>(
        &self,
        working: &mut ReservedMemory<'_>,
    ) -> Result<(Vec<u32>, Option<Rows>)> {
        let rows = self.rows;
        let complete = self.same_len.filter(|&row_len| row_len <= 8 * N);
        working.try_grow(rows.num_rows() * size_of::<(Key<N>, u32)>())?;
        let mut keys = rows
            .iter()
            .zip(0_u32..)
            .map(|(row, position)| (Key(normalized_key::<N>(row.data())), position))
            .collect::<Vec<_>>();
        let compare = |a: &(Key<N>, u32), b: &(Key<N>, u32)| match a.0.cmp(&b.0) {
            Ordering::Equal if complete.is_none() => {
                let (row_a, row_b) = (rows.row(a.1 as usize), rows.row(b.1 as usize));
                row_a.data().cmp(row_b.data())
            }
            ordering => ordering,
        };
        if self.len < keys.len() {
            keys.select_nth_unstable_by(self.len - 1, compare);
            keys.truncate(self.len);
        }
        keys.sort_unstable_by(compare);
        working.try_grow(self.len * size_of::<u32>())?;
        let order = keys
            .iter()
            .map(|&(_, position)| position)
            .collect::<Vec<_>>();
        let copy = match (self.copy, complete) {
            (None, _) => None,
            (Some(converter), Some(row_len)) => Some(copy_of_keys(
                converter,
                keys.iter().map(|(key, _)| &key.0),
                row_len,
                working,
            )?),
            (Some(converter), None) => {
                Some(copy_of_rows(converter, rows, &order, working)?)
            }
        };
        Ok((order, copy))
    }
}

/// The rows (of `row_len` bytes, encoded by `converter`) that the normalized
/// keys `keys` hold, in order. Their memory is reserved in `working` before
/// it is allocated.
fn copy_of_keys<'a, const N: usize>(
    converter: &RowConverter,
    keys: impl ExactSizeIterator<Item = &'a [u64; N]>,
    row_len: usize,
    working: &mut ReservedMemory<'_>,
) -> Result<Rows> {
    let len = keys.len();
    working.try_grow(len * row_len + (len + 1) * size_of::<usize>())?;
    let parser = converter.parser();
    let mut rows = converter.empty_rows(len, len * row_len);
    let mut bytes = [0; 8 * MAX_WORDS];
    for key in keys {
        for (chunk, word) in bytes.chunks_exact_mut(8).zip(key) {
            chunk.copy_from_slice(&word.to_be_bytes());
        }
        rows.push(parser.parse(&bytes[..row_len]));
    }
    Ok(rows)
}

/// The rows `order` of `rows` (encoded by `converter`), in order. Their
/// memory is reserved in `working` before it is allocated.
fn copy_of_rows(
    converter: &RowConverter,
    rows: &Rows,
    order: &[u32],
    working: &mut ReservedMemory<'_>,
) -> Result<Rows> {
    let data_len = order.iter().map(|&i| rows.row_len(i as usize)).sum();
    working.try_grow(data_len + (order.len() + 1) * size_of::<usize>())?;
    let mut sorted = converter.empty_rows(order.len(), data_len);
    for &i in order {
        sorted.push(rows.row(i as usize));
    }
    Ok(sorted)
}

/// Sort the values of a single string or binary sort key (of a
/// `FixedSizeBinary` type too) with `options` by their normalized keys: the
/// positions of the first `limit` values (all of them if `None`) in their
/// order, with the nulls first or last. `None` for values of other types.
///
/// What the sort allocates is at most `KERNEL_SORT_BYTES_PER_ROW` bytes per
/// value, as for the sort kernel.
pub(crate) fn sort_values(
    values: &dyn Array,
    options: SortOptions,
    limit: Option<usize>,
) -> Option<Vec<u32>> {
    let nulls = values.logical_nulls();
    let sort = ValueSort {
        len: values.len(),
        nulls: nulls.as_ref(),
        options,
        limit,
    };
    Some(match values.data_type() {
        DataType::Utf8 => {
            let values = values.as_string::<i32>();
            sort.sort(|i| values.value(i).as_bytes())
        }
        DataType::LargeUtf8 => {
            let values = values.as_string::<i64>();
            sort.sort(|i| values.value(i).as_bytes())
        }
        DataType::Utf8View => {
            let values = values.as_string_view();
            sort.sort(|i| values.value(i).as_bytes())
        }
        DataType::Binary => {
            let values = values.as_binary::<i32>();
            sort.sort(|i| values.value(i))
        }
        DataType::LargeBinary => {
            let values = values.as_binary::<i64>();
            sort.sort(|i| values.value(i))
        }
        DataType::BinaryView => {
            let values = values.as_binary_view();
            sort.sort(|i| values.value(i))
        }
        DataType::FixedSizeBinary(_) => {
            let values = values.as_fixed_size_binary();
            sort.sort(|i| values.value(i))
        }
        _ => return None,
    })
}

/// A sort of the values of a single sort key, see [`sort_values`].
struct ValueSort<'a> {
    len: usize,
    nulls: Option<&'a NullBuffer>,
    options: SortOptions,
    limit: Option<usize>,
}

impl ValueSort<'_> {
    /// Sort the values `value(i)`, see [`sort_values`].
    fn sort<'v>(&self, value: impl Fn(usize) -> &'v [u8]) -> Vec<u32> {
        let is_valid = |i: usize| self.nulls.is_none_or(|nulls| nulls.is_valid(i));
        let null_count = self.nulls.map_or(0, NullBuffer::null_count);
        let limit = self.limit.unwrap_or(self.len).min(self.len);
        // The sorted values that are kept.
        let valid_limit = match self.options.nulls_first {
            true => limit.saturating_sub(null_count),
            false => limit,
        }
        .min(self.len - null_count);
        let max_len = (0..self.len)
            .filter(|&i| is_valid(i))
            .map(|i| value(i).len())
            .max()
            .unwrap_or(0);
        let sorted = if valid_limit == 0 {
            vec![]
        } else if max_len <= 8 {
            self.packed::<2>(&is_valid, &value, valid_limit)
        } else if max_len <= VALUE_PREFIX_BYTES {
            self.packed::<3>(&is_valid, &value, valid_limit)
        } else {
            self.with_suffixes(&is_valid, &value, valid_limit)
        };
        let nulls = (0..self.len).filter(|&i| !is_valid(i)).map(|i| i as u32);
        let mut order = Vec::with_capacity(limit);
        if self.options.nulls_first {
            order.extend(nulls.take(limit));
            order.extend(sorted);
        } else {
            order.extend(sorted);
            order.extend(nulls.take(limit - valid_limit));
        }
        order
    }

    /// Sort the valid values (of at most `8 * (N - 1)` bytes) by their bytes
    /// in the first `N - 1` words of their normalized keys, and by their
    /// length and their position in the last word (descending keys invert
    /// both): the positions of the first `len` values.
    fn packed<'v, const N: usize>(
        &self,
        is_valid: &impl Fn(usize) -> bool,
        value: &impl Fn(usize) -> &'v [u8],
        len: usize,
    ) -> Vec<u32> {
        let descending = self.options.descending;
        let mut keys = (0..self.len)
            .filter(|&i| is_valid(i))
            .map(|i| {
                let bytes = value(i);
                // The last word of the key is padding: the values are shorter.
                let mut key = normalized_key::<N>(bytes);
                let mut value_len = bytes.len() as u32;
                if descending {
                    key.iter_mut().for_each(|word| *word = !*word);
                    value_len = !value_len;
                }
                key[N - 1] = (u64::from(value_len) << 32) | i as u64;
                Key(key)
            })
            .collect::<Vec<_>>();
        if len < keys.len() {
            keys.select_nth_unstable(len - 1);
            keys.truncate(len);
        }
        keys.sort_unstable();
        keys.iter().map(|key| key.0[N - 1] as u32).collect()
    }

    /// Sort the valid values by their first 16 bytes and their length (up to
    /// 17, for values that are longer), and values longer than 16 bytes
    /// whose first 16 bytes are equal by the rest of their bytes: the
    /// positions of the first `len` values.
    fn with_suffixes<'v>(
        &self,
        is_valid: &impl Fn(usize) -> bool,
        value: &impl Fn(usize) -> &'v [u8],
        len: usize,
    ) -> Vec<u32> {
        let mut keys = (0..self.len)
            .filter(|&i| is_valid(i))
            .map(|i| {
                let bytes = value(i);
                let value_len = bytes.len().min(VALUE_PREFIX_BYTES + 1) as u32;
                (Key(normalized_key::<2>(bytes)), value_len, i as u32)
            })
            .collect::<Vec<_>>();
        let ascending = |a: &(Key<2>, u32, u32), b: &(Key<2>, u32, u32)| match a
            .0
            .cmp(&b.0)
            .then(a.1.cmp(&b.1))
        {
            Ordering::Equal if a.1 as usize > VALUE_PREFIX_BYTES => compare_bytes(
                &value(a.2 as usize)[VALUE_PREFIX_BYTES..],
                &value(b.2 as usize)[VALUE_PREFIX_BYTES..],
            ),
            ordering => ordering,
        };
        let descending = self.options.descending;
        let compare = |a: &(Key<2>, u32, u32), b: &(Key<2>, u32, u32)| match descending {
            true => ascending(b, a),
            false => ascending(a, b),
        };
        if len < keys.len() {
            keys.select_nth_unstable_by(len - 1, compare);
            keys.truncate(len);
        }
        keys.sort_unstable_by(compare);
        keys.iter().map(|&(_, _, position)| position).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use std::sync::Arc;

    use arrow::array::{
        ArrayRef, BinaryArray, FixedSizeBinaryArray, Int32Array, Int64Array, StringArray,
        StringViewArray,
    };
    use arrow::compute::{cast, sort_to_indices};
    use arrow_ord::ord::make_comparator;
    use arrow_row::SortField;
    use datafusion_execution::memory_pool::{
        MemoryConsumer, MemoryPool, UnboundedMemoryPool,
    };
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};

    /// Random bytes of up to `max_len` bytes from a small alphabet with
    /// zeros, so that many share prefixes and end in zeros.
    fn random_bytes(rng: &mut StdRng, max_len: usize) -> Vec<u8> {
        let len = rng.random_range(0..=max_len);
        (0..len)
            .map(|_| [0, 1, 0x7f, 0x80, 0xff][rng.random_range(0..5)])
            .collect()
    }

    #[test]
    fn normalized_keys_and_inline_comparisons_order_like_the_bytes() {
        let mut rng = StdRng::seed_from_u64(3);
        for _ in 0..20_000 {
            let max_len = [3, 9, 17, 40][rng.random_range(0..4)];
            let a = random_bytes(&mut rng, max_len);
            let mut b = random_bytes(&mut rng, max_len);
            if rng.random_bool(0.3) {
                // A prefix of the other, or with a common prefix.
                b = [&a[..rng.random_range(0..=a.len())], &b].concat();
            }
            let expected = a.cmp(&b);
            assert_eq!(compare_bytes(&a, &b), expected, "{a:?} {b:?}");
            // Normalized keys that differ order like the bytes.
            let (key_a, key_b) = (normalized_key::<2>(&a), normalized_key::<2>(&b));
            if key_a != key_b {
                assert_eq!(key_a.cmp(&key_b), expected, "{a:?} {b:?}");
            }
            let (key_a, key_b) =
                (Key(normalized_key::<3>(&a)), Key(normalized_key::<3>(&b)));
            assert_eq!(key_a < key_b, key_a.0 < key_b.0, "{a:?} {b:?}");
            assert_eq!(key_a.cmp(&key_b), key_a.0.cmp(&key_b.0));
            if key_a != key_b {
                assert_eq!(key_a.cmp(&key_b), expected, "{a:?} {b:?}");
            }
            // Keys that hold all bytes are equal only for equal bytes of the
            // same length.
            if a.len() <= 24 && a.len() == b.len() {
                assert_eq!(key_a == key_b, a == b);
            }
        }
    }

    fn working_memory() -> MemoryConsumer {
        MemoryConsumer::new("normalized keys")
    }

    /// The encoded rows of `columns` sorted by `sort_rows`, which must order
    /// them like sorting by their bytes, keeping the first `len`.
    fn assert_sorts_rows_like_their_bytes(columns: &[ArrayRef], len: usize) {
        let pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
        let reservation = working_memory().register(&pool);
        let fields = columns
            .iter()
            .map(|column| SortField::new(column.data_type().clone()))
            .collect();
        let converter = RowConverter::new(fields).unwrap();
        let rows = converter.convert_columns(columns).unwrap();
        let mut expected = (0..rows.num_rows()).collect::<Vec<_>>();
        expected.sort_by(|&a, &b| rows.row(a).cmp(&rows.row(b)));
        expected.truncate(len);
        let mut working = ReservedMemory::new(&reservation);
        let (order, copy) =
            sort_rows(&rows, len, Some(&converter), &mut working).unwrap();
        let types = columns
            .iter()
            .map(|column| column.data_type().to_string())
            .collect::<Vec<_>>();
        assert_eq!(order.len(), expected.len(), "{types:?}");
        let copy = copy.unwrap();
        assert_eq!(copy.num_rows(), order.len());
        for (i, (&position, &expected)) in order.iter().zip(&expected).enumerate() {
            // The same keys in the same order (rows with equal keys may come
            // in any order), and the copy holds them.
            assert_eq!(
                rows.row(position as usize),
                rows.row(expected),
                "{types:?}: row {i}"
            );
            assert_eq!(copy.row(i), rows.row(expected), "{types:?}: row {i}");
        }
        // Every row comes once.
        let mut positions = order.clone();
        positions.sort_unstable();
        positions.dedup();
        assert_eq!(positions.len(), order.len());
        // The positions and the copy stay reserved until `working` drops.
        assert!(reservation.size() > 0);
        drop(working);
        assert_eq!(reservation.size(), 0);
        let (order, copy) =
            sort_rows(&rows, len, None, &mut ReservedMemory::new(&reservation)).unwrap();
        assert_eq!(order.len(), expected.len());
        assert!(copy.is_none());
    }

    #[test]
    fn sorts_rows_like_their_bytes() {
        let mut rng = StdRng::seed_from_u64(5);
        let n = 1000;
        let int64 = |rng: &mut StdRng, range: i64| -> ArrayRef {
            Arc::new(Int64Array::from_iter((0..n).map(|_| {
                rng.random_bool(0.9)
                    .then(|| rng.random_range(-range..range))
            })))
        };
        let int32 = |rng: &mut StdRng| -> ArrayRef {
            Arc::new(Int32Array::from_iter(
                (0..n).map(|_| rng.random_bool(0.9).then(|| rng.random_range(-3..3))),
            ))
        };
        let strings = |rng: &mut StdRng, max_len: usize| -> ArrayRef {
            let values = (0..n)
                .map(|_| {
                    rng.random_bool(0.9).then(|| {
                        let len = rng.random_range(0..=max_len);
                        (0..len)
                            .map(|_| ['a', 'b', '\0'][rng.random_range(0..3)])
                            .collect::<String>()
                    })
                })
                .collect::<Vec<_>>();
            Arc::new(StringArray::from(values))
        };
        for _ in 0..4 {
            let cases: Vec<Vec<ArrayRef>> = vec![
                // 5 bytes: one word with the position.
                vec![int32(&mut rng)],
                // 10 and 18 bytes: two and three words with the position.
                vec![int32(&mut rng), int32(&mut rng)],
                vec![int64(&mut rng, 4), int64(&mut rng, 1 << 40)],
                // 27 bytes: four words with the position; 30 bytes: four words
                // without it; 36 bytes: longer than the keys.
                vec![int64(&mut rng, 3), int64(&mut rng, 3), int64(&mut rng, 3)],
                vec![
                    int64(&mut rng, 3),
                    int64(&mut rng, 3),
                    int64(&mut rng, 3),
                    int32(&mut rng),
                ],
                vec![
                    int64(&mut rng, 2),
                    int64(&mut rng, 2),
                    int64(&mut rng, 2),
                    int64(&mut rng, 2),
                ],
                // Rows of different lengths.
                vec![strings(&mut rng, 3)],
                vec![strings(&mut rng, 40), int32(&mut rng)],
                vec![int32(&mut rng), strings(&mut rng, 12)],
            ];
            for columns in &cases {
                for len in [1, 7, 500, n, n + 1] {
                    assert_sorts_rows_like_their_bytes(columns, len);
                }
            }
        }
    }

    #[test]
    fn sorts_values_like_the_sort_kernel() {
        let mut rng = StdRng::seed_from_u64(9);
        let all_options = [
            SortOptions::new(false, false),
            SortOptions::new(false, true),
            SortOptions::new(true, false),
            SortOptions::new(true, true),
        ];
        let data_types = [
            DataType::Utf8,
            DataType::LargeUtf8,
            DataType::Utf8View,
            DataType::Binary,
            DataType::LargeBinary,
            DataType::BinaryView,
        ];
        for max_len in [0, 3, 8, 9, 16, 17, 40] {
            for _ in 0..3 {
                let n = rng.random_range(0..600);
                let null_share = [0.0, 0.2, 1.0][rng.random_range(0..3)];
                let strings = (0..n)
                    .map(|_| {
                        (!rng.random_bool(null_share)).then(|| {
                            let len = rng.random_range(0..=max_len);
                            (0..len)
                                .map(|_| {
                                    ['a', 'b', '\0', '\u{e9}'][rng.random_range(0..4)]
                                })
                                .collect::<String>()
                        })
                    })
                    .collect::<Vec<_>>();
                let strings: ArrayRef = Arc::new(StringArray::from(strings));
                let mut arrays = data_types
                    .iter()
                    .map(|data_type| cast(&strings, data_type).unwrap())
                    .collect::<Vec<_>>();
                // A fixed-size binary of the first byte, sliced.
                let fixed = FixedSizeBinaryArray::try_from_sparse_iter_with_size(
                    (0..n).map(|i| {
                        strings.is_valid(i).then(|| {
                            let value = strings.as_string::<i32>().value(i).as_bytes();
                            [value.first().copied().unwrap_or(0), 7]
                        })
                    }),
                    2,
                )
                .unwrap();
                arrays.push(Arc::new(fixed));
                if n > 2 {
                    arrays.push(strings.slice(1, n - 2));
                    arrays.push(
                        cast(&strings, &DataType::Utf8View).unwrap().slice(2, n - 2),
                    );
                }
                for values in &arrays {
                    for options in all_options {
                        let compare =
                            make_comparator(values.as_ref(), values.as_ref(), options)
                                .unwrap();
                        for limit in [None, Some(0), Some(1), Some(n / 3), Some(n + 5)] {
                            let order =
                                sort_values(values.as_ref(), options, limit).unwrap();
                            let expected =
                                sort_to_indices(values.as_ref(), Some(options), limit)
                                    .unwrap();
                            let case =
                                format!("{} {options:?} {limit:?}", values.data_type());
                            assert_eq!(order.len(), expected.len(), "{case}");
                            for (&a, &b) in order.iter().zip(expected.values()) {
                                assert_eq!(
                                    compare(a as usize, b as usize),
                                    Ordering::Equal,
                                    "{case}"
                                );
                            }
                            let mut positions = order.clone();
                            positions.sort_unstable();
                            positions.dedup();
                            assert_eq!(positions.len(), order.len(), "{case}");
                        }
                    }
                }
            }
        }
        // Other types are left to the sort kernel.
        let ints = Int32Array::from(vec![1, 2]);
        assert!(sort_values(&ints, SortOptions::default(), None).is_none());
        let binary = BinaryArray::from(vec![b"b".as_ref(), b"a"]);
        assert_eq!(
            sort_values(&binary, SortOptions::default(), None),
            Some(vec![1, 0])
        );
        let views = StringViewArray::from(vec!["b", "a"]);
        assert_eq!(
            sort_values(&views, SortOptions::default(), None),
            Some(vec![1, 0])
        );
    }
}
