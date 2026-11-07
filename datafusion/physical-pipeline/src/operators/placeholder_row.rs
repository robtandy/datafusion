// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! `PhysicalPlaceholderRow`: source for `PlaceholderRowExec`.
//!
//! DataFusion plans a `PlaceholderRowExec` for a relation that produces a
//! single row whose values are never read: the input of a projection
//! without a `FROM` clause (`SELECT 1, 'hello'`), the result of a global
//! aggregation without aggregate functions, and the input of the constants
//! that replace an aggregation answered from statistics
//! (`SELECT COUNT(*) FROM t`). It produces one row per output partition.
//!
//! [`PhysicalPlaceholderRow`] is the native source for it. Its schema is
//! almost always empty: it then produces chunks without columns that only
//! carry a row count, exactly like `PlaceholderRowExec`. For a schema with
//! fields, `PlaceholderRowExec` emits `NULL` columns of type `Null`, which
//! do not match its own schema. The chunks of this source always have the
//! schema of the operator, as the pipeline engine requires: nullable fields
//! are `NULL`, non-nullable fields hold the default value of their type
//! (`0`, `''`, ...), so that the rows are valid for the operators above,
//! which never read them.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use arrow::array::{ArrayRef, new_null_array};
use arrow::datatypes::{Field, SchemaRef};
use datafusion_common::{Result, ScalarValue};
use datafusion_physical_plan::placeholder_row::PlaceholderRowExec;
use datafusion_physical_plan::{ExecutionPlan, ExecutionPlanProperties};

use crate::chunk::{DEFAULT_VECTOR_SIZE, DataChunk};
use crate::operator::{
    ExecutionContext, GlobalSourceState, LocalSourceState, PhysicalOperator,
    SourceResult, downcast_state,
};

/// A source that produces `rows` placeholder rows (one by default).
///
/// See the [module documentation](self) for the values of the rows.
#[derive(Debug)]
pub struct PhysicalPlaceholderRow {
    schema: SchemaRef,
    rows: usize,
}

impl PhysicalPlaceholderRow {
    /// A source of a single row with schema `schema`.
    ///
    /// Fails if the default value of the type of a non-nullable field cannot
    /// be created.
    pub fn try_new(schema: SchemaRef) -> Result<Self> {
        let source = Self { schema, rows: 1 };
        // Fail now rather than during the execution.
        source.chunk(1)?;
        Ok(source)
    }

    /// Produce `rows` rows instead of one. A `PlaceholderRowExec` with `n`
    /// partitions produces `n` rows, one per partition.
    pub fn with_rows(mut self, rows: usize) -> Self {
        self.rows = rows;
        self
    }

    /// Lower a DataFusion `PlaceholderRowExec` to a source that produces its
    /// rows, one per output partition. Returns `Ok(None)` for any other node,
    /// and for a `PlaceholderRowExec` whose rows this source cannot create
    /// (see [`Self::try_new`]), which can be executed by DataFusion instead.
    pub fn try_lower(plan: &Arc<dyn ExecutionPlan>) -> Result<Option<Self>> {
        if !plan.is::<PlaceholderRowExec>() {
            return Ok(None);
        }
        let rows = plan.output_partitioning().partition_count();
        Ok(Self::try_new(plan.schema())
            .ok()
            .map(|source| source.with_rows(rows)))
    }

    /// The number of rows this source produces.
    pub fn rows(&self) -> usize {
        self.rows
    }

    /// A chunk of `len` placeholder rows.
    fn chunk(&self, len: usize) -> Result<DataChunk> {
        if self.schema.fields().is_empty() {
            return DataChunk::try_new_without_columns(Arc::clone(&self.schema), len);
        }
        let columns = self
            .schema
            .fields()
            .iter()
            .map(|field| placeholder_column(field, len))
            .collect::<Result<Vec<_>>>()?;
        DataChunk::try_new(Arc::clone(&self.schema), columns, None)
    }
}

/// `len` placeholder values for `field`: `NULL`s if it is nullable, the
/// default value of its type otherwise.
fn placeholder_column(field: &Field, len: usize) -> Result<ArrayRef> {
    if field.is_nullable() {
        Ok(new_null_array(field.data_type(), len))
    } else {
        ScalarValue::new_default(field.data_type())?.to_array_of_size(len)
    }
}

/// The global source state of a [`PhysicalPlaceholderRow`].
#[derive(Debug)]
struct PlaceholderRowState {
    /// The number of rows handed out so far (possibly more than `rows`).
    next: AtomicUsize,
    rows: usize,
}

impl GlobalSourceState for PlaceholderRowState {
    fn max_threads(&self) -> usize {
        self.rows.div_ceil(DEFAULT_VECTOR_SIZE).max(1)
    }
}

impl PhysicalOperator for PhysicalPlaceholderRow {
    fn name(&self) -> &str {
        "PhysicalPlaceholderRow"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &[]
    }

    fn description(&self) -> String {
        if self.rows == 1 {
            "PhysicalPlaceholderRow".to_string()
        } else {
            format!("PhysicalPlaceholderRow: rows={}", self.rows)
        }
    }

    fn is_source(&self) -> bool {
        true
    }

    fn parallel_source(&self) -> bool {
        true
    }

    fn get_global_source_state(
        &self,
        _ctx: &ExecutionContext,
    ) -> Result<Arc<dyn GlobalSourceState>> {
        Ok(Arc::new(PlaceholderRowState {
            next: AtomicUsize::new(0),
            rows: self.rows,
        }))
    }

    fn get_data(
        &self,
        _ctx: &ExecutionContext,
        gstate: &dyn GlobalSourceState,
        _lstate: &mut dyn LocalSourceState,
    ) -> Result<SourceResult> {
        let state = downcast_state::<PlaceholderRowState>(gstate.as_any(), self.name())?;
        let start = state.next.fetch_add(DEFAULT_VECTOR_SIZE, Ordering::Relaxed);
        if start >= state.rows {
            return Ok(SourceResult::Finished);
        }
        let len = (state.rows - start).min(DEFAULT_VECTOR_SIZE);
        Ok(SourceResult::Chunk(self.chunk(len)?))
    }
}
