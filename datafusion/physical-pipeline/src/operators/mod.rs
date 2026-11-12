// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Native streaming operators.

pub mod adaptive_filter;
pub mod analyze;
pub mod asof_join;
pub mod async_func;
pub mod caching;
pub mod data_sink;
pub mod filter;
pub mod limit;
#[cfg(test)]
pub(crate) mod lowering_tests;
pub mod materialized_cte;
pub mod partial_sort;
pub mod partitioned_topk;
pub(crate) mod piecewise_merge_join;
pub mod placeholder_row;
pub mod projection;
pub mod range_join;
pub mod recursive_cte;
pub mod union;
pub mod unnest;
