// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Native join operators.
//!
//! * [`hash_table`]: [`JoinHashTable`], the shared, radix-partitioned build
//!   side of hash joins, with one open-addressed pointer table of salted
//!   entries per partition, built in parallel.
//! * [`hash_join`]: [`PhysicalHashJoin`], DataFusion's `HashJoinExec` as a
//!   pipeline sink (build), operator (probe) and source (final build rows),
//!   built by all threads into one shared table and probed by all threads
//!   without shuffling either side.
//! * [`nested_loop`]: nested loop and cross joins.
//! * [`merge_join`]: [`PhysicalMergeJoin`], DataFusion's `SortMergeJoinExec`
//!   where its order is required (otherwise a [`PhysicalHashJoin`]).
//! * [`symmetric_hash_join`]: [`PhysicalSymmetricHashJoin`], DataFusion's
//!   `SymmetricHashJoinExec` of two inputs that never end.

pub mod hash_join;
pub mod hash_table;
pub mod merge_join;
pub mod nested_loop;
pub mod symmetric_hash_join;

pub use hash_join::PhysicalHashJoin;
pub use hash_table::{JoinHashTable, JoinHashTableBuilder};
pub use merge_join::PhysicalMergeJoin;
pub use symmetric_hash_join::PhysicalSymmetricHashJoin;
