// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

use std::hash::Hash;
use std::sync::Arc;

use arrow::array::types::{
    Int8Type, Int16Type, Int32Type, Int64Type, UInt8Type, UInt16Type, UInt32Type,
    UInt64Type,
};
use arrow::array::{
    Array, ArrayRef, ArrowPrimitiveType, BooleanArray, Int64Array, NullBufferBuilder,
    PrimitiveArray, RecordBatch, UInt32Array,
};
use arrow::buffer::BooleanBuffer;
use arrow::datatypes::{DataType, SchemaRef};
use datafusion_common::cast::as_boolean_array;
use datafusion_common::{HashMap, Result, ScalarValue, internal_err};
use datafusion_execution::memory_pool::MemoryReservation;
use datafusion_expr::logical_plan::Aggregate;
use datafusion_expr::{ColumnarValue, Expr};
use datafusion_physical_expr::PhysicalExpr;
use tokio::sync::mpsc;

use super::batch::{BatchView, Selection};
use super::count::count_overflow;
use super::metrics::{self, RunMetrics};
use super::{Operator, apply_operators, filter_batch};

const MERGE_YIELD_INTERVAL: usize = 4096;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum IntKeyType {
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
}

impl IntKeyType {
    pub(super) fn from_data_type(data_type: &DataType) -> Option<Self> {
        match data_type {
            DataType::Int8 => Some(Self::Int8),
            DataType::Int16 => Some(Self::Int16),
            DataType::Int32 => Some(Self::Int32),
            DataType::Int64 => Some(Self::Int64),
            DataType::UInt8 => Some(Self::UInt8),
            DataType::UInt16 => Some(Self::UInt16),
            DataType::UInt32 => Some(Self::UInt32),
            DataType::UInt64 => Some(Self::UInt64),
            _ => None,
        }
    }
}

pub(super) struct GroupKeySpec<'a> {
    pub(super) expr: &'a Expr,
    pub(super) column_index: usize,
    pub(super) key_type: IntKeyType,
}

pub(super) fn group_by_integer_column(
    aggregate: &Aggregate,
) -> Result<std::result::Result<GroupKeySpec<'_>, String>> {
    match aggregate.group_expr.as_slice() {
        [Expr::GroupingSet(_)] => Ok(Err(
            "GROUPING SETS, ROLLUP, and CUBE are not supported in native morsel aggregates"
                .into(),
        )),
        [expr] => {
            let mut unaliased = expr;
            while let Expr::Alias(alias) = unaliased {
                unaliased = &alias.expr;
            }
            let Expr::Column(column) = unaliased else {
                return Ok(Err(
                    "GROUP BY key must be a single integer column".into(),
                ));
            };
            let column_index = aggregate.input.schema().index_of_column(column)?;
            let data_type = aggregate.input.schema().field(column_index).data_type();
            let Some(key_type) = IntKeyType::from_data_type(data_type) else {
                return Ok(Err(format!(
                    "unsupported GROUP BY key type {data_type}: only a single integer column is supported"
                )));
            };
            Ok(Ok(GroupKeySpec {
                expr,
                column_index,
                key_type,
            }))
        }
        _ => Ok(Err(
            "only a single GROUP BY integer column is supported in native morsel aggregates"
                .into(),
        )),
    }
}

fn estimated_next_table_bytes<K>(current_alloc: usize) -> usize {
    // hashbrown doubles bucket count on rehash; include control bytes and alignment slack
    // so the reservation covers old + new table buffers coexisting during reallocation.
    let min_alloc = 8 * (size_of::<(K, i64)>() + 1) + 32;
    current_alloc.saturating_mul(2).max(min_alloc)
}

struct GroupCountMap<T: ArrowPrimitiveType>
where
    T::Native: Eq + Hash,
{
    counts: HashMap<T::Native, i64>,
    null_count: Option<i64>,
    reservation: MemoryReservation,
}

impl<T: ArrowPrimitiveType> GroupCountMap<T>
where
    T::Native: Eq + Hash,
{
    fn new(reservation: MemoryReservation) -> Self {
        Self {
            counts: HashMap::new(),
            null_count: None,
            reservation,
        }
    }

    fn add_null(&mut self, rows: i64) -> Result<()> {
        if rows == 0 {
            return Ok(());
        }
        let updated = match self.null_count {
            Some(existing) => existing.checked_add(rows).ok_or_else(count_overflow)?,
            None => rows,
        };
        self.null_count = Some(updated);
        Ok(())
    }

    #[inline]
    fn add_key(&mut self, key: T::Native, delta: i64) -> Result<()> {
        if let Some(count) = self.counts.get_mut(&key) {
            *count = count.checked_add(delta).ok_or_else(count_overflow)?;
            return Ok(());
        }
        self.insert_new(key, delta)
    }

    #[cold]
    fn insert_new(&mut self, key: T::Native, count: i64) -> Result<()> {
        if self.counts.len() == self.counts.capacity() {
            let next_bytes =
                estimated_next_table_bytes::<T::Native>(self.counts.allocation_size());
            self.reservation.try_grow(next_bytes)?;
            self.counts.insert(key, count);
            let actual = self.counts.allocation_size();
            self.reservation.try_resize(actual)?;
        } else {
            self.counts.insert(key, count);
            let actual = self.counts.allocation_size();
            if actual != self.reservation.size() {
                self.reservation.try_resize(actual)?;
            }
        }
        Ok(())
    }

    fn add_array(&mut self, array: &PrimitiveArray<T>) -> Result<()> {
        let len = array.len();
        if len == 0 {
            return Ok(());
        }
        let null_count = array.null_count();
        if null_count == 0 {
            for &key in array.values().iter() {
                self.add_key(key, 1)?;
            }
            return Ok(());
        }
        if null_count == len {
            let rows = i64::try_from(len).map_err(|_| count_overflow())?;
            return self.add_null(rows);
        }
        let values = array.values();
        let nulls = array.nulls().unwrap();
        let mut null_rows: i64 = 0;
        for (i, &key) in values.iter().enumerate() {
            if nulls.is_valid(i) {
                self.add_key(key, 1)?;
            } else {
                null_rows = null_rows.checked_add(1).ok_or_else(count_overflow)?;
            }
        }
        self.add_null(null_rows)
    }

    fn add_array_selected(
        &mut self,
        array: &PrimitiveArray<T>,
        selected: &BooleanBuffer,
    ) -> Result<()> {
        let null_count = array.null_count();
        if null_count == array.len() {
            let rows =
                i64::try_from(selected.count_set_bits()).map_err(|_| count_overflow())?;
            return self.add_null(rows);
        }
        let values = array.values();
        if null_count == 0 {
            for i in selected.set_indices() {
                self.add_key(values[i], 1)?;
            }
            return Ok(());
        }
        let nulls = array.nulls().unwrap();
        let mut null_rows: i64 = 0;
        for i in selected.set_indices() {
            if nulls.is_valid(i) {
                self.add_key(values[i], 1)?;
            } else {
                null_rows = null_rows.checked_add(1).ok_or_else(count_overflow)?;
            }
        }
        self.add_null(null_rows)
    }

    fn add_array_indices(
        &mut self,
        array: &PrimitiveArray<T>,
        indices: &UInt32Array,
    ) -> Result<()> {
        if indices.is_empty() {
            return Ok(());
        }
        let null_count = array.null_count();
        if null_count == array.len() {
            let rows = i64::try_from(indices.len()).map_err(|_| count_overflow())?;
            return self.add_null(rows);
        }
        let values = array.values();
        if null_count == 0 {
            for &idx in indices.values() {
                self.add_key(values[idx as usize], 1)?;
            }
            return Ok(());
        }
        let nulls = array.nulls().unwrap();
        let mut null_rows: i64 = 0;
        for &idx in indices.values() {
            let i = idx as usize;
            if nulls.is_valid(i) {
                self.add_key(values[i], 1)?;
            } else {
                null_rows = null_rows.checked_add(1).ok_or_else(count_overflow)?;
            }
        }
        self.add_null(null_rows)
    }

    fn add_array_masked(
        &mut self,
        array: &PrimitiveArray<T>,
        mask: &BooleanArray,
    ) -> Result<()> {
        debug_assert_eq!(array.len(), mask.len());
        let selected_count = mask.true_count();
        if selected_count == 0 {
            return Ok(());
        }
        if selected_count == array.len() {
            return self.add_array(array);
        }
        match mask.nulls() {
            Some(nulls) => {
                // Clear undefined value bits at null mask positions before set_indices().
                let effective = mask.values() & nulls.inner();
                self.add_array_selected(array, &effective)
            }
            None => self.add_array_selected(array, mask.values()),
        }
    }

    async fn merge_from(
        &mut self,
        mut other: Self,
        tx: &mpsc::Sender<RecordBatch>,
    ) -> Result<bool> {
        if let Some(n) = other.null_count.take() {
            self.add_null(n)?;
        }
        if other.counts.is_empty() {
            return Ok(true);
        }
        if self.counts.is_empty() {
            std::mem::swap(&mut self.counts, &mut other.counts);
            std::mem::swap(&mut self.reservation, &mut other.reservation);
            return Ok(true);
        }
        if other.counts.len() > self.counts.len() {
            std::mem::swap(&mut self.counts, &mut other.counts);
            std::mem::swap(&mut self.reservation, &mut other.reservation);
        }
        let mut processed = 0usize;
        for (key, count) in other.counts.drain() {
            self.add_key(key, count)?;
            processed += 1;
            if processed.is_multiple_of(MERGE_YIELD_INTERVAL) {
                if tx.is_closed() {
                    return Ok(false);
                }
                tokio::task::yield_now().await;
            }
        }
        Ok(true)
    }

    async fn emit_batches(
        self,
        schema: &SchemaRef,
        result_operators: &[Operator],
        batch_size: usize,
        tx: &mpsc::Sender<RecordBatch>,
        measurement: Option<&RunMetrics>,
    ) -> Result<()> {
        let batch_size = batch_size.max(1);
        let table_reservation = self.reservation;
        let mut null_count = self.null_count;
        let mut iter = self.counts.into_iter();
        while !tx.is_closed() {
            let remaining = iter.len() + usize::from(null_count.is_some());
            if remaining == 0 {
                break;
            }
            let rows = remaining.min(batch_size);
            let has_null = null_count.is_some();
            let null_bytes = if has_null { rows.div_ceil(8) } else { 0 };
            let batch_bytes = rows
                .saturating_mul(size_of::<T::Native>() + size_of::<i64>())
                .saturating_add(null_bytes);
            let batch_reservation = table_reservation.new_empty();
            batch_reservation.try_grow(batch_bytes)?;

            let mut keys = Vec::with_capacity(rows);
            let mut counts = Vec::with_capacity(rows);
            let nulls = if let Some(n) = null_count.take() {
                keys.push(T::Native::default());
                counts.push(n);
                let mut builder = NullBufferBuilder::new(rows);
                builder.append_null();
                builder.append_n_non_nulls(rows - 1);
                builder.finish()
            } else {
                None
            };
            for (k, c) in (&mut iter).take(rows - keys.len()) {
                keys.push(k);
                counts.push(c);
            }
            let key_array: ArrayRef =
                Arc::new(PrimitiveArray::<T>::new(keys.into(), nulls));
            let count_array: ArrayRef = Arc::new(Int64Array::from(counts));
            let batch =
                RecordBatch::try_new(Arc::clone(schema), vec![key_array, count_array])?;
            let batch = apply_operators(result_operators, batch)?;

            if batch.num_rows() > 0 {
                let sent = if let Some(run) = measurement {
                    metrics::send(run, None, tx, batch).await
                } else {
                    tx.send(batch).await.is_ok()
                };
                drop(batch_reservation);
                if !sent {
                    break;
                }
            } else {
                drop(batch_reservation);
            }
            if remaining > rows {
                tokio::task::yield_now().await;
            }
        }
        drop(iter);
        drop(table_reservation);
        Ok(())
    }
}

macro_rules! dispatch_group_map {
    ($self:expr, $map:ident => $body:expr) => {
        match $self {
            GroupCountMapKind::Int8($map) => $body,
            GroupCountMapKind::Int16($map) => $body,
            GroupCountMapKind::Int32($map) => $body,
            GroupCountMapKind::Int64($map) => $body,
            GroupCountMapKind::UInt8($map) => $body,
            GroupCountMapKind::UInt16($map) => $body,
            GroupCountMapKind::UInt32($map) => $body,
            GroupCountMapKind::UInt64($map) => $body,
        }
    };
}

macro_rules! downcast_primitive {
    ($array:expr, $ty:ty) => {
        $array
            .as_any()
            .downcast_ref::<PrimitiveArray<$ty>>()
            .ok_or_else(|| {
                datafusion_common::internal_datafusion_err!(
                    "Morsel GROUP BY key array type mismatch: expected {:?}, got {:?}",
                    <$ty as ArrowPrimitiveType>::DATA_TYPE,
                    $array.data_type()
                )
            })?
    };
}

enum GroupCountMapKind {
    Int8(GroupCountMap<Int8Type>),
    Int16(GroupCountMap<Int16Type>),
    Int32(GroupCountMap<Int32Type>),
    Int64(GroupCountMap<Int64Type>),
    UInt8(GroupCountMap<UInt8Type>),
    UInt16(GroupCountMap<UInt16Type>),
    UInt32(GroupCountMap<UInt32Type>),
    UInt64(GroupCountMap<UInt64Type>),
}

pub(super) struct GroupCountState(GroupCountMapKind);

impl GroupCountState {
    pub(super) fn new(key_type: IntKeyType, reservation: MemoryReservation) -> Self {
        Self(match key_type {
            IntKeyType::Int8 => GroupCountMapKind::Int8(GroupCountMap::new(reservation)),
            IntKeyType::Int16 => {
                GroupCountMapKind::Int16(GroupCountMap::new(reservation))
            }
            IntKeyType::Int32 => {
                GroupCountMapKind::Int32(GroupCountMap::new(reservation))
            }
            IntKeyType::Int64 => {
                GroupCountMapKind::Int64(GroupCountMap::new(reservation))
            }
            IntKeyType::UInt8 => {
                GroupCountMapKind::UInt8(GroupCountMap::new(reservation))
            }
            IntKeyType::UInt16 => {
                GroupCountMapKind::UInt16(GroupCountMap::new(reservation))
            }
            IntKeyType::UInt32 => {
                GroupCountMapKind::UInt32(GroupCountMap::new(reservation))
            }
            IntKeyType::UInt64 => {
                GroupCountMapKind::UInt64(GroupCountMap::new(reservation))
            }
        })
    }

    pub(super) fn add_column(&mut self, array: &ArrayRef) -> Result<()> {
        match &mut self.0 {
            GroupCountMapKind::Int8(m) => {
                m.add_array(downcast_primitive!(array, Int8Type))
            }
            GroupCountMapKind::Int16(m) => {
                m.add_array(downcast_primitive!(array, Int16Type))
            }
            GroupCountMapKind::Int32(m) => {
                m.add_array(downcast_primitive!(array, Int32Type))
            }
            GroupCountMapKind::Int64(m) => {
                m.add_array(downcast_primitive!(array, Int64Type))
            }
            GroupCountMapKind::UInt8(m) => {
                m.add_array(downcast_primitive!(array, UInt8Type))
            }
            GroupCountMapKind::UInt16(m) => {
                m.add_array(downcast_primitive!(array, UInt16Type))
            }
            GroupCountMapKind::UInt32(m) => {
                m.add_array(downcast_primitive!(array, UInt32Type))
            }
            GroupCountMapKind::UInt64(m) => {
                m.add_array(downcast_primitive!(array, UInt64Type))
            }
        }
    }

    fn add_column_selected(
        &mut self,
        array: &ArrayRef,
        bits: &BooleanBuffer,
    ) -> Result<()> {
        match &mut self.0 {
            GroupCountMapKind::Int8(m) => {
                m.add_array_selected(downcast_primitive!(array, Int8Type), bits)
            }
            GroupCountMapKind::Int16(m) => {
                m.add_array_selected(downcast_primitive!(array, Int16Type), bits)
            }
            GroupCountMapKind::Int32(m) => {
                m.add_array_selected(downcast_primitive!(array, Int32Type), bits)
            }
            GroupCountMapKind::Int64(m) => {
                m.add_array_selected(downcast_primitive!(array, Int64Type), bits)
            }
            GroupCountMapKind::UInt8(m) => {
                m.add_array_selected(downcast_primitive!(array, UInt8Type), bits)
            }
            GroupCountMapKind::UInt16(m) => {
                m.add_array_selected(downcast_primitive!(array, UInt16Type), bits)
            }
            GroupCountMapKind::UInt32(m) => {
                m.add_array_selected(downcast_primitive!(array, UInt32Type), bits)
            }
            GroupCountMapKind::UInt64(m) => {
                m.add_array_selected(downcast_primitive!(array, UInt64Type), bits)
            }
        }
    }

    fn add_column_indices(
        &mut self,
        array: &ArrayRef,
        indices: &UInt32Array,
    ) -> Result<()> {
        match &mut self.0 {
            GroupCountMapKind::Int8(m) => {
                m.add_array_indices(downcast_primitive!(array, Int8Type), indices)
            }
            GroupCountMapKind::Int16(m) => {
                m.add_array_indices(downcast_primitive!(array, Int16Type), indices)
            }
            GroupCountMapKind::Int32(m) => {
                m.add_array_indices(downcast_primitive!(array, Int32Type), indices)
            }
            GroupCountMapKind::Int64(m) => {
                m.add_array_indices(downcast_primitive!(array, Int64Type), indices)
            }
            GroupCountMapKind::UInt8(m) => {
                m.add_array_indices(downcast_primitive!(array, UInt8Type), indices)
            }
            GroupCountMapKind::UInt16(m) => {
                m.add_array_indices(downcast_primitive!(array, UInt16Type), indices)
            }
            GroupCountMapKind::UInt32(m) => {
                m.add_array_indices(downcast_primitive!(array, UInt32Type), indices)
            }
            GroupCountMapKind::UInt64(m) => {
                m.add_array_indices(downcast_primitive!(array, UInt64Type), indices)
            }
        }
    }

    pub(super) fn add_view(&mut self, view: &BatchView, key_index: usize) -> Result<()> {
        if view.is_empty() {
            return Ok(());
        }
        let col = view.batch().column(key_index);
        match view.selection() {
            Selection::All => self.add_column(col),
            Selection::Mask { bits, selected } => {
                if *selected == view.batch().num_rows() {
                    self.add_column(col)
                } else {
                    self.add_column_selected(col, bits)
                }
            }
            Selection::Indices(indices) => self.add_column_indices(col, indices),
        }
    }

    fn add_column_masked(&mut self, array: &ArrayRef, mask: &BooleanArray) -> Result<()> {
        match &mut self.0 {
            GroupCountMapKind::Int8(m) => {
                m.add_array_masked(downcast_primitive!(array, Int8Type), mask)
            }
            GroupCountMapKind::Int16(m) => {
                m.add_array_masked(downcast_primitive!(array, Int16Type), mask)
            }
            GroupCountMapKind::Int32(m) => {
                m.add_array_masked(downcast_primitive!(array, Int32Type), mask)
            }
            GroupCountMapKind::Int64(m) => {
                m.add_array_masked(downcast_primitive!(array, Int64Type), mask)
            }
            GroupCountMapKind::UInt8(m) => {
                m.add_array_masked(downcast_primitive!(array, UInt8Type), mask)
            }
            GroupCountMapKind::UInt16(m) => {
                m.add_array_masked(downcast_primitive!(array, UInt16Type), mask)
            }
            GroupCountMapKind::UInt32(m) => {
                m.add_array_masked(downcast_primitive!(array, UInt32Type), mask)
            }
            GroupCountMapKind::UInt64(m) => {
                m.add_array_masked(downcast_primitive!(array, UInt64Type), mask)
            }
        }
    }

    pub(super) fn add_filtered_batch(
        &mut self,
        batch: &RecordBatch,
        predicate: &Arc<dyn PhysicalExpr>,
        key_index: usize,
    ) -> Result<()> {
        match predicate.evaluate(batch)? {
            ColumnarValue::Scalar(ScalarValue::Boolean(Some(true))) => {
                self.add_column(batch.column(key_index))
            }
            ColumnarValue::Scalar(ScalarValue::Boolean(Some(false) | None)) => Ok(()),
            ColumnarValue::Scalar(_) => {
                internal_err!("Cannot create filter_array from non-boolean predicates")
            }
            ColumnarValue::Array(array) => {
                let Ok(mask) = as_boolean_array(&array) else {
                    return internal_err!(
                        "Cannot create filter_array from non-boolean predicates"
                    );
                };
                if mask.len() == batch.num_rows() {
                    self.add_column_masked(batch.column(key_index), mask)
                } else {
                    let filtered = filter_batch(batch, mask, Some(&[key_index]))?;
                    if filtered.num_rows() > 0 {
                        self.add_column(filtered.column(0))?;
                    }
                    Ok(())
                }
            }
        }
    }

    pub(super) async fn merge_from(
        &mut self,
        other: Self,
        tx: &mpsc::Sender<RecordBatch>,
    ) -> Result<bool> {
        match (&mut self.0, other.0) {
            (GroupCountMapKind::Int8(a), GroupCountMapKind::Int8(b)) => {
                a.merge_from(b, tx).await
            }
            (GroupCountMapKind::Int16(a), GroupCountMapKind::Int16(b)) => {
                a.merge_from(b, tx).await
            }
            (GroupCountMapKind::Int32(a), GroupCountMapKind::Int32(b)) => {
                a.merge_from(b, tx).await
            }
            (GroupCountMapKind::Int64(a), GroupCountMapKind::Int64(b)) => {
                a.merge_from(b, tx).await
            }
            (GroupCountMapKind::UInt8(a), GroupCountMapKind::UInt8(b)) => {
                a.merge_from(b, tx).await
            }
            (GroupCountMapKind::UInt16(a), GroupCountMapKind::UInt16(b)) => {
                a.merge_from(b, tx).await
            }
            (GroupCountMapKind::UInt32(a), GroupCountMapKind::UInt32(b)) => {
                a.merge_from(b, tx).await
            }
            (GroupCountMapKind::UInt64(a), GroupCountMapKind::UInt64(b)) => {
                a.merge_from(b, tx).await
            }
            _ => internal_err!("Morsel GROUP BY worker state key type mismatch"),
        }
    }
}

#[derive(Debug)]
pub(super) struct GroupCountSink {
    schema: SchemaRef,
    pub(super) key_index: usize,
    pub(super) key_type: IntKeyType,
    pub(super) key_name: String,
    #[cfg(test)]
    pub(super) finalize_gate: Option<super::tests::FinalizeGate>,
}

impl GroupCountSink {
    pub(super) fn new(
        schema: SchemaRef,
        key_index: usize,
        key_type: IntKeyType,
        key_name: String,
    ) -> Self {
        Self {
            schema,
            key_index,
            key_type,
            key_name,
            #[cfg(test)]
            finalize_gate: None,
        }
    }

    pub(super) fn new_state(&self, reservation: MemoryReservation) -> GroupCountState {
        GroupCountState::new(self.key_type, reservation)
    }

    pub(super) fn finalize(&self, combined: GroupCountState) -> FinalizedGroupCounts {
        FinalizedGroupCounts {
            schema: Arc::clone(&self.schema),
            state: combined,
        }
    }
}

pub(super) struct FinalizedGroupCounts {
    schema: SchemaRef,
    state: GroupCountState,
}

impl FinalizedGroupCounts {
    pub(super) async fn emit(
        self,
        result_operators: &[Operator],
        batch_size: usize,
        tx: mpsc::Sender<RecordBatch>,
        measurement: Option<&RunMetrics>,
    ) -> Result<()> {
        let schema = self.schema;
        dispatch_group_map!(self.state.0, map => {
            map.emit_batches(&schema, result_operators, batch_size, &tx, measurement).await
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{
        Int8Array, Int16Array, Int32Array, Int64Array, UInt8Array, UInt16Array,
        UInt32Array, UInt64Array,
    };
    use arrow::buffer::NullBuffer;
    use arrow::datatypes::{Field, Schema};
    use datafusion_common_runtime::SpawnedTask;
    use datafusion_execution::memory_pool::{
        GreedyMemoryPool, MemoryConsumer, MemoryPool, UnboundedMemoryPool,
    };

    fn test_reservation(pool: &Arc<dyn MemoryPool>) -> MemoryReservation {
        MemoryConsumer::new("test_group_count").register(pool)
    }

    #[test]
    fn checked_overflow_on_add_key_and_add_null() {
        let pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
        let mut map = GroupCountMap::<Int32Type>::new(test_reservation(&pool));
        map.add_key(7, i64::MAX).unwrap();
        let err = map.add_key(7, 1).unwrap_err();
        assert!(err.to_string().contains("exceeds Int64"), "{err}");

        map.add_null(i64::MAX).unwrap();
        let err = map.add_null(1).unwrap_err();
        assert!(err.to_string().contains("exceeds Int64"), "{err}");
    }

    #[tokio::test]
    async fn checked_overflow_on_merge() {
        let pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
        let (tx, _rx) = mpsc::channel(2);

        let mut a = GroupCountMap::<Int32Type>::new(test_reservation(&pool));
        let mut b = GroupCountMap::<Int32Type>::new(test_reservation(&pool));
        a.add_key(1, i64::MAX).unwrap();
        b.add_key(1, 1).unwrap();
        let err = a.merge_from(b, &tx).await.unwrap_err();
        assert!(err.to_string().contains("exceeds Int64"), "{err}");

        let mut a = GroupCountMap::<Int32Type>::new(test_reservation(&pool));
        let mut b = GroupCountMap::<Int32Type>::new(test_reservation(&pool));
        a.add_null(i64::MAX).unwrap();
        b.add_null(1).unwrap();
        let err = a.merge_from(b, &tx).await.unwrap_err();
        assert!(err.to_string().contains("exceeds Int64"), "{err}");
    }

    #[test]
    fn sliced_array_and_masked_nulls_with_set_bits() -> Result<()> {
        let pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
        let mut map = GroupCountMap::<Int32Type>::new(test_reservation(&pool));

        // Base array: [99, 10, NULL, 10, 20, NULL, 88]
        // Sliced (offset 1, len 5): [10, NULL, 10, 20, NULL]
        let base_keys = Int32Array::from(vec![
            Some(99),
            Some(10),
            None,
            Some(10),
            Some(20),
            None,
            Some(88),
        ]);
        let keys = base_keys.slice(1, 5);

        // Base mask with set value bits under nulls (offset 1, len 5):
        // values:   [true, true, true, true, false, true, true]
        // validity: [true, true, true, false, true, true, true]
        // sliced:   idx 0 -> true(valid) => key 10
        //           idx 1 -> true(valid) => key NULL
        //           idx 2 -> true(NULL!) => MUST BE IGNORED (key 10 not counted!)
        //           idx 3 -> false(valid) => ignored
        //           idx 4 -> true(valid) => key NULL
        let base_mask = BooleanArray::new(
            BooleanBuffer::from(vec![true, true, true, true, false, true, true]),
            Some(NullBuffer::from(vec![
                true, true, true, false, true, true, true,
            ])),
        );
        let mask = base_mask.slice(1, 5);

        map.add_array_masked(&keys, &mask)?;
        assert_eq!(map.counts.len(), 1);
        assert_eq!(map.counts.get(&10), Some(&1));
        assert_eq!(map.null_count, Some(2));
        Ok(())
    }

    #[tokio::test]
    async fn memory_reservation_tracks_growth_merge_peak_and_cleans_up() -> Result<()> {
        // Measure exact size of one 1024-key worker map first.
        let unbounded: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
        let mut probe = GroupCountMap::<Int64Type>::new(test_reservation(&unbounded));
        for k in 0..1024i64 {
            probe.add_key(k, 1)?;
        }
        let one_worker_bytes = probe.counts.allocation_size();
        assert_eq!(probe.reservation.size(), one_worker_bytes);
        assert!(one_worker_bytes > 0);
        drop(probe);
        assert_eq!(unbounded.reserved(), 0);

        // 1. Accumulation fails cleanly when budget is smaller than one worker map.
        let tight: Arc<dyn MemoryPool> =
            Arc::new(GreedyMemoryPool::new(one_worker_bytes / 2));
        {
            let mut m = GroupCountMap::<Int64Type>::new(test_reservation(&tight));
            let mut hit_oom = false;
            for k in 0..1024i64 {
                if let Err(err) = m.add_key(k, 1) {
                    assert!(
                        matches!(
                            err,
                            datafusion_common::DataFusionError::ResourcesExhausted(_)
                        ),
                        "{err}"
                    );
                    hit_oom = true;
                    break;
                }
            }
            assert!(hit_oom, "expected ResourcesExhausted during accumulation");
        }
        assert_eq!(tight.reserved(), 0);

        // 2. Merge peak fails cleanly when two disjoint 1024-key maps fit simultaneously,
        //    but growing the merged destination during merge exceeds the pool budget.
        //    Each map holds ~one_worker_bytes; during merge, `a` must double (rehash peak
        //    = a_old + a_new + b_alive ~ 4 * one_worker_bytes).
        let merge_budget = one_worker_bytes * 3 + one_worker_bytes / 2;
        let merge_pool: Arc<dyn MemoryPool> =
            Arc::new(GreedyMemoryPool::new(merge_budget));
        let (tx, _rx) = mpsc::channel(2);
        {
            let mut a = GroupCountMap::<Int64Type>::new(test_reservation(&merge_pool));
            let mut b = GroupCountMap::<Int64Type>::new(test_reservation(&merge_pool));
            for k in 0..1024i64 {
                a.add_key(k, 1)?;
            }
            for k in 1024..2048i64 {
                b.add_key(k, 1)?;
            }
            assert_eq!(merge_pool.reserved(), 2 * one_worker_bytes);
            let err = a.merge_from(b, &tx).await.unwrap_err();
            assert!(
                matches!(
                    err,
                    datafusion_common::DataFusionError::ResourcesExhausted(_)
                ),
                "expected ResourcesExhausted at merge peak, got: {err}"
            );
        }
        assert_eq!(merge_pool.reserved(), 0);

        // 3. Emit succeeds with sufficient budget and releases all memory on completion.
        let ample: Arc<dyn MemoryPool> =
            Arc::new(GreedyMemoryPool::new(one_worker_bytes * 16));
        let (tx, mut rx) = mpsc::channel(4);
        let schema = Arc::new(Schema::new(vec![
            Field::new("k", DataType::Int64, true),
            Field::new("count", DataType::Int64, false),
        ]));
        let mut a = GroupCountMap::<Int64Type>::new(test_reservation(&ample));
        let mut b = GroupCountMap::<Int64Type>::new(test_reservation(&ample));
        for k in 0..100i64 {
            a.add_key(k, 2)?;
        }
        a.add_null(3)?;
        for k in 50..150i64 {
            b.add_key(k, 5)?;
        }
        b.add_null(4)?;
        assert!(a.merge_from(b, &tx).await?);
        assert!(ample.reserved() > 0);
        let emit_task = SpawnedTask::spawn(async move {
            a.emit_batches(&schema, &[], 32, &tx, None).await
        });
        let mut total_rows = 0;
        let mut batches = 0;
        while let Some(batch) = rx.recv().await {
            assert!(batch.num_rows() <= 32);
            total_rows += batch.num_rows();
            batches += 1;
        }
        emit_task.join_unwind().await.unwrap()?;
        assert_eq!(total_rows, 151); // 150 non-null + 1 NULL
        assert_eq!(batches, 5); // 32 * 4 + 23
        assert_eq!(ample.reserved(), 0);
        Ok(())
    }

    #[tokio::test]
    async fn all_eight_integer_types_accumulate_merge_and_emit() -> Result<()> {
        let pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
        let cases: Vec<(IntKeyType, DataType, ArrayRef)> = vec![
            (
                IntKeyType::Int8,
                DataType::Int8,
                Arc::new(Int8Array::from(vec![
                    Some(i8::MIN),
                    None,
                    Some(i8::MAX),
                    Some(i8::MIN),
                ])),
            ),
            (
                IntKeyType::Int16,
                DataType::Int16,
                Arc::new(Int16Array::from(vec![
                    Some(i16::MIN),
                    None,
                    Some(i16::MAX),
                    Some(i16::MIN),
                ])),
            ),
            (
                IntKeyType::Int32,
                DataType::Int32,
                Arc::new(Int32Array::from(vec![
                    Some(i32::MIN),
                    None,
                    Some(i32::MAX),
                    Some(i32::MIN),
                ])),
            ),
            (
                IntKeyType::Int64,
                DataType::Int64,
                Arc::new(Int64Array::from(vec![
                    Some(i64::MIN),
                    None,
                    Some(i64::MAX),
                    Some(i64::MIN),
                ])),
            ),
            (
                IntKeyType::UInt8,
                DataType::UInt8,
                Arc::new(UInt8Array::from(vec![
                    Some(0),
                    None,
                    Some(u8::MAX),
                    Some(0),
                ])),
            ),
            (
                IntKeyType::UInt16,
                DataType::UInt16,
                Arc::new(UInt16Array::from(vec![
                    Some(0),
                    None,
                    Some(u16::MAX),
                    Some(0),
                ])),
            ),
            (
                IntKeyType::UInt32,
                DataType::UInt32,
                Arc::new(UInt32Array::from(vec![
                    Some(0),
                    None,
                    Some(u32::MAX),
                    Some(0),
                ])),
            ),
            (
                IntKeyType::UInt64,
                DataType::UInt64,
                Arc::new(UInt64Array::from(vec![
                    Some(0),
                    None,
                    Some(u64::MAX),
                    Some(0),
                ])),
            ),
        ];

        for (key_type, dt, array) in cases {
            let schema = Arc::new(Schema::new(vec![
                Field::new("k", dt.clone(), true),
                Field::new("n", DataType::Int64, false),
            ]));
            let sink = GroupCountSink::new(Arc::clone(&schema), 0, key_type, "k".into());
            let mut w0 = sink.new_state(test_reservation(&pool));
            let mut w1 = sink.new_state(test_reservation(&pool));
            w0.add_column(&array)?;
            w1.add_column(&array)?;
            let (tx, mut rx) = mpsc::channel(4);
            assert!(w0.merge_from(w1, &tx).await?);
            let finalized = sink.finalize(w0);
            let emit =
                SpawnedTask::spawn(async move { finalized.emit(&[], 2, tx, None).await });
            let mut rows = 0;
            let mut total_count = 0i64;
            while let Some(batch) = rx.recv().await {
                assert_eq!(batch.schema(), schema);
                assert!(batch.num_rows() <= 2);
                rows += batch.num_rows();
                let counts = batch
                    .column(1)
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap();
                total_count += counts.values().iter().sum::<i64>();
            }
            emit.join_unwind().await.unwrap()?;
            assert_eq!(rows, 3, "type={dt}"); // min/0, max, NULL
            assert_eq!(total_count, 8, "type={dt}");
            assert_eq!(pool.reserved(), 0, "type={dt}");
        }
        Ok(())
    }

    #[test]
    fn group_count_add_view_all_mask_and_indices_on_sliced_batch() -> Result<()> {
        let pool: Arc<dyn MemoryPool> = Arc::new(UnboundedMemoryPool::default());
        let schema = Arc::new(Schema::new(vec![Field::new("k", DataType::Int32, true)]));
        let base = RecordBatch::try_new(
            schema,
            vec![Arc::new(Int32Array::from(vec![
                Some(99),
                Some(10),
                None,
                Some(10),
                Some(20),
                None,
                Some(88),
            ]))],
        )?;
        let sliced = base.slice(1, 5); // [10, NULL, 10, 20, NULL]
        let mask = BooleanArray::from(vec![
            Some(true),
            Some(true),
            None,
            Some(false),
            Some(true),
        ]);
        let view_mask = BatchView::from_boolean_array(sliced.clone(), &mask)?;
        let view_idx = view_mask.clone().with_indices_selection();

        let mut s_mask = GroupCountState::new(IntKeyType::Int32, test_reservation(&pool));
        let mut s_idx = GroupCountState::new(IntKeyType::Int32, test_reservation(&pool));
        s_mask.add_view(&view_mask, 0)?;
        s_idx.add_view(&view_idx, 0)?;
        let GroupCountMapKind::Int32(m1) = &s_mask.0 else {
            unreachable!()
        };
        let GroupCountMapKind::Int32(m2) = &s_idx.0 else {
            unreachable!()
        };
        assert_eq!(m1.counts, m2.counts);
        assert_eq!(m1.null_count, m2.null_count);
        assert_eq!(m1.counts.get(&10), Some(&1));
        assert_eq!(m1.null_count, Some(2));
        Ok(())
    }
}
