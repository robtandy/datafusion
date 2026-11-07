// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Native scan sources (memory, Parquet, CSV, empty) that hand out
//! `DEFAULT_VECTOR_SIZE`-row morsels to all pipeline threads.
//!
//! Placeholder: implemented in Wave 2. Until then every scan runs through a
//! [`BridgeOperator`](crate::bridge::BridgeOperator).
