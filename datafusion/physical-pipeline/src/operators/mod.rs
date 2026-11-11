// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Native streaming operators.

pub mod adaptive_filter;
pub mod caching;
pub mod filter;
pub mod limit;
#[cfg(test)]
pub(crate) mod lowering_tests;
pub mod placeholder_row;
pub mod projection;
pub mod union;
