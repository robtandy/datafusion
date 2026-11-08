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

pub mod hash_join;
pub mod hash_table;
pub mod nested_loop;

pub use hash_join::PhysicalHashJoin;
pub use hash_table::{JoinHashTable, JoinHashTableBuilder};
