// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Shared radix-partitioning primitives for morsel `GROUP BY` and `HashJoin`.
//!
//! Hash-bit layout (64-bit `foldhash` / `mix_u64` values):
//! ```text
//! 63      57 56     49 48     41 40     33 32                             0
//! +---------+---------+---------+---------+-------------------------------+
//! | h2 (7b) | L0 (<=8)| L1 (<=8)| L2 (<=8)| h1 bucket / bloom block (33b) |
//! +---------+---------+---------+---------+-------------------------------+
//! ```
//! Middle bit windows `[33..57)` stay completely disjoint from `hashbrown`'s
//! top-7-bit `h2` SIMD control tag (`[57..64)`) and low `h1` bucket bits
//! (`[0..33)`), so 64-bit hashes are passed unchanged into `hashbrown` and
//! open-addressing / chained join tables without entropy collapse.

use arrow::array::UInt32Array;
use arrow::buffer::ScalarBuffer;

pub(super) const RADIX_MAX_HIGH_BIT: u32 = 57;
pub(super) const RADIX_MIN_LOW_BIT: u32 = 33;
pub(super) const DEFAULT_RADIX_BITS: u32 = 6; // 64 partitions
pub(super) const RECURSIVE_RADIX_BITS: u32 = 4; // 16 sub-partitions per recursive split
pub(super) const MAX_RADIX_DEPTH: usize = 2;

/// Extract radix partition index from a middle bit window disjoint from
/// `hashbrown`'s `h2` (`[57..64)`) and low `h1` bucket bits (`[0..33)`).
#[inline]
pub(super) fn radix_partition(h: u64, shift: u32, radix_bits: u32) -> usize {
    debug_assert!(
        radix_bits == 0
            || (shift >= RADIX_MIN_LOW_BIT && shift + radix_bits <= RADIX_MAX_HIGH_BIT)
    );
    if radix_bits == 0 {
        return 0;
    }
    ((h >> shift) & ((1u64 << radix_bits) - 1)) as usize
}

/// Level-0 shift for `radix_bits` in `[0..=8]`.
#[inline]
pub(super) const fn level0_shift(radix_bits: u32) -> u32 {
    if radix_bits == 0 {
        RADIX_MAX_HIGH_BIT
    } else {
        RADIX_MAX_HIGH_BIT - radix_bits
    }
}

/// Next lower disjoint bit-window shift for recursive radix re-partitioning,
/// returning `None` if the bit budget `[33..57)` would be exceeded.
#[inline]
pub(super) const fn next_level_shift(
    prev_shift: u32,
    next_radix_bits: u32,
) -> Option<u32> {
    if next_radix_bits == 0 {
        return None;
    }
    match prev_shift.checked_sub(next_radix_bits) {
        Some(shift) if shift >= RADIX_MIN_LOW_BIT => Some(shift),
        _ => None,
    }
}

/// Compute a single-pass O(N) counting-sort permutation grouping row indices
/// `0..hashes.len()` by `radix_partition(hashes[i], shift, radix_bits)`.
///
/// Returns `(permutation_indices, ranges)` where `ranges[p] = (offset, len)`
/// gives the contiguous slice in the permuted batch belonging to partition `p`.
pub(super) fn radix_permute_indices(
    hashes: &[u64],
    shift: u32,
    radix_bits: u32,
) -> (UInt32Array, Vec<(usize, usize)>) {
    let num_parts = 1usize << radix_bits;
    let mut counts = vec![0usize; num_parts];
    for &h in hashes {
        counts[radix_partition(h, shift, radix_bits)] += 1;
    }

    let mut offsets = vec![0usize; num_parts];
    let mut ranges = Vec::with_capacity(num_parts);
    let mut cursor = 0usize;
    for p in 0..num_parts {
        let len = counts[p];
        offsets[p] = cursor;
        ranges.push((cursor, len));
        cursor += len;
    }

    let mut perm = vec![0u32; hashes.len()];
    for (row_idx, &h) in hashes.iter().enumerate() {
        let p = radix_partition(h, shift, radix_bits);
        let dst = offsets[p];
        perm[dst] = row_idx as u32;
        offsets[p] = dst + 1;
    }

    (UInt32Array::new(ScalarBuffer::from(perm), None), ranges)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::morsel::hash_join::mix_u64;

    #[test]
    fn radix_bit_windows_preserve_h1_and_h2_uniformity() {
        let r0 = DEFAULT_RADIX_BITS;
        let s0 = level0_shift(r0);
        let r1 = RECURSIVE_RADIX_BITS;
        let s1 = next_level_shift(s0, r1).expect("L1 shift");
        let r2 = RECURSIVE_RADIX_BITS;
        let s2 = next_level_shift(s1, r2).expect("L2 shift");
        assert!(s2 >= RADIX_MIN_LOW_BIT);
        assert_eq!(s0 + r0, RADIX_MAX_HIGH_BIT);

        let n = 65_536u64;
        let num_parts = 1usize << r0;
        let mut h2_seen = vec![[false; 128]; num_parts];
        let mut h1_seen = vec![[false; 64]; num_parts];
        let mut part_counts = vec![0usize; num_parts];

        for k in 0..n {
            let h = mix_u64(k);
            let p = radix_partition(h, s0, r0);
            part_counts[p] += 1;
            h2_seen[p][(h >> 57) as usize] = true;
            h1_seen[p][(h & 63) as usize] = true;
        }

        // Every partition receives close to n / 64 keys and observes nearly all
        // 128 h2 control tags and all 64 low h1 buckets.
        let expected = (n as usize) / num_parts;
        for p in 0..num_parts {
            assert!(
                part_counts[p] > expected / 2 && part_counts[p] < expected * 2,
                "partition {p} imbalanced: {}",
                part_counts[p]
            );
            let distinct_h2 = h2_seen[p].iter().filter(|&&b| b).count();
            let distinct_h1 = h1_seen[p].iter().filter(|&&b| b).count();
            assert!(
                distinct_h2 >= 120,
                "partition {p} lost h2 entropy: {distinct_h2}/128"
            );
            assert_eq!(
                distinct_h1, 64,
                "partition {p} lost h1 entropy: {distinct_h1}/64"
            );
        }
    }

    #[test]
    fn radix_permute_groups_all_rows_by_partition() {
        let shift = level0_shift(DEFAULT_RADIX_BITS);
        let hashes: Vec<u64> = (0..1000u64).map(mix_u64).collect();
        let (perm, ranges) = radix_permute_indices(&hashes, shift, DEFAULT_RADIX_BITS);
        assert_eq!(perm.len(), hashes.len());
        let mut seen = vec![false; hashes.len()];
        for (p, &(offset, len)) in ranges.iter().enumerate() {
            for i in offset..offset + len {
                let row = perm.value(i) as usize;
                assert!(!seen[row]);
                seen[row] = true;
                assert_eq!(radix_partition(hashes[row], shift, DEFAULT_RADIX_BITS), p);
            }
        }
        assert!(seen.into_iter().all(|b| b));
    }
}
