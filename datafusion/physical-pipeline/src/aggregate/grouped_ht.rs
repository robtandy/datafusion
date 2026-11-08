// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`GroupedHashTable`]: the hash table of a grouped aggregation, mapping the
//! group keys to the aggregate states of their groups.
//!
//! It is the hash table of DataFusion's hash aggregation (its
//! `AggregateHashTable`): the group keys are interned by DataFusion's
//! [`GroupValues`] (the implementation specialized for the group key types,
//! which hashes every key once and assigns consecutive group indices), and
//! the aggregate states are kept by one [`GroupsAccumulator`] per aggregate
//! (or a [`GroupsAccumulatorAdapter`] over its [`Accumulator`]s, for the
//! aggregates without a `GroupsAccumulator`), indexed by group index. So
//! the groups, the grouping of `NULL`s, `-0.0` / `NaN` and the aggregate
//! values are exactly those of DataFusion.
//!
//! A [`PhysicalHashAggregate`](super::PhysicalHashAggregate) uses a table in
//! two places:
//!
//! * every thread aggregates the rows of its input chunks into its own
//!   table ([`GroupedHashTable::update`], DataFusion's partial aggregation);
//! * every radix partition is finalized by merging the partial states of
//!   its groups from all threads into a new table
//!   ([`GroupedHashTable::merge`]) and evaluating it
//!   ([`GroupedHashTable::evaluate`], DataFusion's final aggregation).
//!
//! [`Accumulator`]: datafusion_expr::Accumulator

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BooleanArray, new_null_array};
use arrow::compute::prep_null_mask_filter;
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion_common::{Result, ScalarValue};
use datafusion_expr::{EmitTo, GroupsAccumulator};
use datafusion_physical_expr::GroupsAccumulatorAdapter;
use datafusion_physical_expr::aggregate::AggregateFunctionExpr;
use datafusion_physical_plan::aggregates::evaluate_group_by;
use datafusion_physical_plan::aggregates::group_values::{GroupValues, new_group_values};
use datafusion_physical_plan::aggregates::order::GroupClustering;

use super::hash_aggregate::AggregateSpec;

/// The group keys and the aggregate states of a set of groups, emitted by a
/// [`GroupedHashTable`]: row `i` of every array belongs to the same group.
#[derive(Debug, Clone)]
pub(crate) struct GroupedStates {
    /// The group columns (the columns of `AggregateSpec::group_schema`).
    pub(crate) keys: Vec<ArrayRef>,
    /// For every aggregate, its state columns (`GroupsAccumulator::state`).
    pub(crate) states: Vec<Vec<ArrayRef>>,
}

impl GroupedStates {
    /// The number of groups.
    pub(crate) fn num_groups(&self) -> usize {
        self.keys.first().map_or(0, |keys| keys.len())
    }
}

/// Maps group keys to group indices, and keeps the aggregate states of the
/// groups. See the [module documentation](self).
pub(crate) struct GroupedHashTable {
    group_values: Box<dyn GroupValues>,
    accumulators: Vec<Box<dyn GroupsAccumulator>>,
    /// The group index of every row of the batch being aggregated.
    group_indices: Vec<usize>,
}

impl std::fmt::Debug for GroupedHashTable {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("GroupedHashTable")
            .field("groups", &self.num_groups())
            .field("accumulators", &self.accumulators.len())
            .finish()
    }
}

impl GroupedHashTable {
    /// A table without groups for the group keys of `group_schema` and the
    /// aggregates `aggregates`.
    pub(crate) fn try_new(
        group_schema: &SchemaRef,
        aggregates: &[Arc<AggregateFunctionExpr>],
    ) -> Result<Self> {
        Ok(Self {
            group_values: new_group_values(
                Arc::clone(group_schema),
                &GroupClustering::None,
            )?,
            accumulators: aggregates
                .iter()
                .map(create_groups_accumulator)
                .collect::<Result<_>>()?,
            group_indices: vec![],
        })
    }

    /// The table of the thread-local aggregation of `spec`.
    pub(crate) fn try_new_update(spec: &AggregateSpec) -> Result<Self> {
        let aggregates = spec
            .update
            .iter()
            .map(|update| Arc::clone(&update.expr))
            .collect::<Vec<_>>();
        Self::try_new(&spec.group_schema, &aggregates)
    }

    /// The table of the final aggregation of `spec`.
    pub(crate) fn try_new_merge(spec: &AggregateSpec) -> Result<Self> {
        Self::try_new(&spec.group_schema, &spec.merge)
    }

    /// The number of groups.
    pub(crate) fn num_groups(&self) -> usize {
        self.group_values.len()
    }

    /// Whether the table has no groups.
    pub(crate) fn is_empty(&self) -> bool {
        self.group_values.is_empty()
    }

    /// Aggregate the input rows of `batch`, like DataFusion's partial (or
    /// single) aggregation: the group keys of every grouping set of every
    /// row are interned, and the accumulator of every aggregate is updated
    /// with its arguments for the rows that pass its filter.
    pub(crate) fn update(
        &mut self,
        spec: &AggregateSpec,
        batch: &RecordBatch,
    ) -> Result<()> {
        // The keys of every grouping set, e.g. `(a, b)`, `(a, NULL)` and
        // `(NULL, NULL)` for `ROLLUP (a, b)`.
        let grouping_sets = evaluate_group_by(&spec.group_by, batch)?;
        let arguments = spec
            .update
            .iter()
            .map(|update| update.evaluate_filtered(batch))
            .collect::<Result<Vec<_>>>()?;
        for keys in &grouping_sets {
            self.group_values.intern(keys, &mut self.group_indices)?;
            let num_groups = self.group_values.len();
            for (accumulator, arguments) in self.accumulators.iter_mut().zip(&arguments) {
                // The group indices of the rows that pass the filter.
                let selected = arguments.selection.as_ref().and_then(|selection| {
                    compact_group_indices(&self.group_indices, selection)
                });
                accumulator.update_batch(
                    &arguments.values,
                    selected.as_deref().unwrap_or(&self.group_indices),
                    None,
                    num_groups,
                )?;
            }
        }
        Ok(())
    }

    /// Merge partial states into the table, like DataFusion's final
    /// aggregation.
    pub(crate) fn merge(&mut self, states: &GroupedStates) -> Result<()> {
        self.group_values
            .intern(&states.keys, &mut self.group_indices)?;
        let num_groups = self.group_values.len();
        for (accumulator, state) in self.accumulators.iter_mut().zip(&states.states) {
            accumulator.merge_batch(state, &self.group_indices, num_groups)?;
        }
        Ok(())
    }

    /// Emit the keys and states of all groups, and reset the table. Returns
    /// `None` if the table has no groups.
    pub(crate) fn emit_states(&mut self) -> Result<Option<GroupedStates>> {
        if self.is_empty() {
            return Ok(None);
        }
        let keys = self.group_values.emit(EmitTo::All)?;
        let states = self
            .accumulators
            .iter_mut()
            .map(|accumulator| accumulator.state(EmitTo::All))
            .collect::<Result<Vec<_>>>()?;
        Ok(Some(GroupedStates { keys, states }))
    }

    /// Evaluate all groups: their keys followed by the final value of every
    /// aggregate, the columns of the output of the aggregation. Resets the
    /// table. Returns `None` if the table has no groups.
    pub(crate) fn evaluate(&mut self) -> Result<Option<Vec<ArrayRef>>> {
        if self.is_empty() {
            return Ok(None);
        }
        let mut columns = self.group_values.emit(EmitTo::All)?;
        for accumulator in &mut self.accumulators {
            columns.push(accumulator.evaluate(EmitTo::All)?);
        }
        Ok(Some(columns))
    }

    /// If the table has no groups, add the groups of the grouping sets in
    /// which all group columns are `NULL` (e.g. the grand total of a
    /// `ROLLUP`), which exist even without any input row, like DataFusion's
    /// partial aggregation of an empty input does (`init_empty_grouping_sets`):
    /// the accumulators are updated with no rows.
    pub(crate) fn add_empty_grouping_sets(&mut self, spec: &AggregateSpec) -> Result<()> {
        let group_by = &spec.group_by;
        if !group_by.has_grouping_set() || !self.is_empty() {
            return Ok(());
        }
        let num_exprs = group_by.expr().len();
        let grouping_id_type = spec.group_schema.field(num_exprs).data_type();
        let mut ordinals = HashMap::<&[bool], usize>::new();
        let mut added = false;
        for group in group_by.groups() {
            let ordinal = ordinals.entry(group.as_slice()).or_insert(0);
            let current = *ordinal;
            *ordinal += 1;
            if !group.iter().all(|&is_null| is_null) {
                continue;
            }
            let mut keys = spec
                .group_schema
                .fields()
                .iter()
                .take(num_exprs)
                .map(|field| new_null_array(field.data_type(), 1))
                .collect::<Vec<_>>();
            let grouping_id = grouping_id(group, current);
            keys.push(
                ScalarValue::UInt64(Some(grouping_id))
                    .cast_to(grouping_id_type)?
                    .to_array()?,
            );
            self.group_values.intern(&keys, &mut self.group_indices)?;
            added = true;
        }
        if added {
            let num_groups = self.group_values.len();
            for (accumulator, update) in self.accumulators.iter_mut().zip(&spec.update) {
                let values = update
                    .arguments
                    .iter()
                    .map(|argument| {
                        Ok(new_null_array(&argument.data_type(&spec.input_schema)?, 0))
                    })
                    .collect::<Result<Vec<_>>>()?;
                accumulator.update_batch(&values, &[], None, num_groups)?;
            }
        }
        Ok(())
    }
}

/// A [`GroupsAccumulator`] for `expr`: its own if it has one, otherwise a
/// [`GroupsAccumulatorAdapter`] over its [`Accumulator`]s, like DataFusion
/// (`create_group_accumulator`).
///
/// [`Accumulator`]: datafusion_expr::Accumulator
fn create_groups_accumulator(
    expr: &Arc<AggregateFunctionExpr>,
) -> Result<Box<dyn GroupsAccumulator>> {
    if expr.groups_accumulator_supported() {
        expr.create_groups_accumulator()
    } else {
        let expr = Arc::clone(expr);
        Ok(Box::new(GroupsAccumulatorAdapter::new(move || {
            expr.create_accumulator()
        })))
    }
}

/// The group indices of the rows for which `filter` is `true`, or `None` if
/// it is `true` for all rows (DataFusion's `compact_group_indices`).
fn compact_group_indices(
    group_indices: &[usize],
    filter: &BooleanArray,
) -> Option<Vec<usize>> {
    let filter = if filter.null_count() > 0 {
        prep_null_mask_filter(filter)
    } else {
        filter.clone()
    };
    let mask = filter.values();
    if mask.count_set_bits() == group_indices.len() {
        return None;
    }
    Some(mask.set_indices().map(|row| group_indices[row]).collect())
}

/// The value of the `__grouping_id` column of the grouping set `group` (its
/// `ordinal`-th occurrence, from 0) in DataFusion's encoding
/// (`group_id_array`): the low bits flag the `NULL` group columns (the last
/// column in the lowest bit), the high bits hold the ordinal of a repeated
/// grouping set. The type of the column is wide enough for both.
fn grouping_id(group: &[bool], ordinal: usize) -> u64 {
    let semantic_id = group
        .iter()
        .fold(0u64, |id, &is_null| (id << 1) | u64::from(is_null));
    if ordinal == 0 {
        semantic_id
    } else {
        semantic_id | ((ordinal as u64) << group.len())
    }
}
