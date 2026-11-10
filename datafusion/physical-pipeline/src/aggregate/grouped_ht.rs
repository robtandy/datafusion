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
//!   ([`GroupedHashTable::merge`]), aggregating the input rows that
//!   bypassed the thread-local tables into it
//!   ([`GroupedHashTable::update_rows`], of rows evaluated by
//!   [`evaluate_rows`]), and evaluating it ([`GroupedHashTable::evaluate`],
//!   DataFusion's final aggregation).
//!
//! [`Accumulator`]: datafusion_expr::Accumulator

use std::collections::HashMap;
use std::sync::Arc;

use arrow::array::{Array, ArrayRef, BooleanArray, UInt32Array, new_null_array};
use arrow::compute::{filter, prep_null_mask_filter, take};
use arrow::datatypes::{DataType, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion_common::cast::as_boolean_array;
use datafusion_common::{Result, ScalarValue, internal_err, not_impl_err};
use datafusion_expr::{EmitTo, GroupsAccumulator};
use datafusion_physical_expr::GroupsAccumulatorAdapter;
use datafusion_physical_expr::aggregate::AggregateFunctionExpr;
use datafusion_physical_plan::aggregates::evaluate_group_by;
use datafusion_physical_plan::aggregates::group_values::{GroupValues, new_group_values};
use datafusion_physical_plan::aggregates::order::GroupClustering;

use super::hash_aggregate::{AggregateSpec, literal_value};
use crate::chunk::DEFAULT_VECTOR_SIZE;

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

    /// The groups at `indices`.
    pub(crate) fn take(&self, indices: &UInt32Array) -> Result<Self> {
        let take_all = |arrays: &[ArrayRef]| {
            arrays
                .iter()
                .map(|array| Ok(take(array.as_ref(), indices, None)?))
                .collect::<Result<Vec<_>>>()
        };
        Ok(Self {
            keys: take_all(&self.keys)?,
            states: self
                .states
                .iter()
                .map(|state| take_all(state))
                .collect::<Result<_>>()?,
        })
    }
}

/// The group keys and the aggregate arguments of the input rows of `batch`
/// (of the input of `spec`, an aggregation without grouping sets), aligned
/// with its rows, for [`GroupedHashTable::update_rows`]: the group columns,
/// followed by the arguments of every aggregate that are not literals (see
/// [`literal_value`]) and, for an aggregate with a filter, the filter result
/// (see [`AggregateUpdate::evaluate_row_aligned`]).
///
/// [`AggregateUpdate::evaluate_row_aligned`]: super::hash_aggregate::AggregateUpdate::evaluate_row_aligned
pub(crate) fn evaluate_rows(
    spec: &AggregateSpec,
    batch: &RecordBatch,
) -> Result<Vec<ArrayRef>> {
    let mut grouping_sets = evaluate_group_by(&spec.group_by, batch)?;
    if grouping_sets.len() != 1 {
        return internal_err!(
            "Only the input rows of an aggregation without grouping sets can be \
             evaluated on their own, got {} grouping sets",
            grouping_sets.len()
        );
    }
    let mut columns = grouping_sets.swap_remove(0);
    for update in &spec.update {
        let (values, selection) = update.evaluate_row_aligned(batch)?;
        columns.extend(values);
        if let Some(selection) = selection {
            columns.push(Arc::new(selection));
        }
    }
    Ok(columns)
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

    /// Aggregate input rows evaluated by [`evaluate_rows`] (`columns`, of the
    /// input of `spec`) into the table, like [`Self::update`]: the group keys
    /// of every row are interned, and the accumulator of every aggregate is
    /// updated with its arguments for the rows that pass its filter.
    ///
    /// The final table of a partition aggregates the rows that bypassed the
    /// thread-local tables so, with the accumulators of the final
    /// aggregation: only for aggregations whose final aggregates are those
    /// that consume the input rows (see `AggregateSpec::bypass`).
    pub(crate) fn update_rows(
        &mut self,
        spec: &AggregateSpec,
        columns: &[ArrayRef],
    ) -> Result<()> {
        let num_keys = spec.group_by.expr().len();
        let expected = num_keys
            + spec
                .update
                .iter()
                .map(|update| {
                    let stored = update
                        .arguments
                        .iter()
                        .filter(|argument| literal_value(argument).is_none())
                        .count();
                    stored + usize::from(update.filter.is_some())
                })
                .sum::<usize>();
        if columns.len() != expected || self.accumulators.len() != spec.update.len() {
            return internal_err!(
                "PhysicalHashAggregate expected {expected} columns of input rows for \
                 {} accumulators, got {} columns for {}",
                spec.update.len(),
                columns.len(),
                self.accumulators.len()
            );
        }
        let (keys, mut rest) = columns.split_at(num_keys);
        let num_rows = keys.first().map_or(0, |keys| keys.len());
        // The arguments that are literals are rebuilt for the rows, like the
        // regular path evaluates them on chunks of at most
        // `DEFAULT_VECTOR_SIZE` rows: literals of variable length (strings,
        // lists...) for as many rows at most, so that their arrays do not
        // overflow their offsets where DataFusion's (of its batches) do not.
        let variable_literals = spec.update.iter().any(|update| {
            update.arguments.iter().any(|argument| {
                literal_value(argument).is_some_and(|value| {
                    let data_type = value.data_type();
                    !(data_type.is_primitive()
                        || matches!(data_type, DataType::Boolean | DataType::Null))
                })
            })
        });
        if variable_literals && num_rows > DEFAULT_VECTOR_SIZE {
            for offset in (0..num_rows).step_by(DEFAULT_VECTOR_SIZE) {
                let len = DEFAULT_VECTOR_SIZE.min(num_rows - offset);
                let slice = columns
                    .iter()
                    .map(|column| column.slice(offset, len))
                    .collect::<Vec<_>>();
                self.update_rows(spec, &slice)?;
            }
            return Ok(());
        }
        self.group_values.intern(keys, &mut self.group_indices)?;
        let num_groups = self.group_values.len();
        for (accumulator, update) in self.accumulators.iter_mut().zip(&spec.update) {
            let values = update
                .arguments
                .iter()
                .map(|argument| match literal_value(argument) {
                    Some(value) => value.to_array_of_size(num_rows),
                    None => {
                        let (value, tail) = rest.split_at(1);
                        rest = tail;
                        Ok(Arc::clone(&value[0]))
                    }
                })
                .collect::<Result<Vec<_>>>()?;
            let values = values.as_slice();
            let selection = match update.filter {
                Some(_) => {
                    let (selection, tail) = rest.split_at(1);
                    rest = tail;
                    Some(as_boolean_array(&selection[0])?)
                }
                None => None,
            };
            // The group indices of the rows that pass the filter.
            let selected = selection.and_then(|selection| {
                compact_group_indices(&self.group_indices, selection)
                    .map(|indices| (selection, indices))
            });
            match selected {
                None => accumulator.update_batch(
                    values,
                    &self.group_indices,
                    None,
                    num_groups,
                )?,
                Some((selection, indices)) => {
                    // The arguments of the rows that pass the filter.
                    let values = values
                        .iter()
                        .map(|value| Ok(filter(value.as_ref(), selection)?))
                        .collect::<Result<Vec<_>>>()?;
                    accumulator.update_batch(&values, &indices, None, num_groups)?
                }
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

    /// Emit the keys and states of all groups, and reset the table, which
    /// can then aggregate more rows. Returns `None` if the table has no
    /// groups.
    pub(crate) fn emit_states(&mut self) -> Result<Option<GroupedStates>> {
        let num_groups = self.num_groups();
        if num_groups == 0 {
            return Ok(None);
        }
        let keys = self.group_values.emit(EmitTo::All)?;
        // Some `GroupValues` keep their hash table on `emit(EmitTo::All)`:
        // clear it, keeping room for as many groups.
        self.group_values.clear_shrink(num_groups);
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
        let max_ordinal = max_duplicate_ordinal(group_by.groups());
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
            let grouping_id = grouping_id(group, current, max_ordinal)?;
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
/// `ordinal`-th occurrence, from 0) of grouping sets whose repeated sets have
/// ordinals up to `max_ordinal`, exactly like DataFusion's `group_id_array`:
/// the low bits flag the `NULL` group columns (the last column in the lowest
/// bit), the high bits hold the ordinal of a repeated grouping set. Fails
/// like `group_id_array` if the two do not fit into 64 bits. (The type of the
/// column, `Aggregate::grouping_id_type`, is wide enough for both if they
/// do.)
fn grouping_id(group: &[bool], ordinal: usize, max_ordinal: usize) -> Result<u64> {
    let n = group.len();
    if n > 64 {
        return not_impl_err!(
            "Grouping sets with more than 64 columns are not supported"
        );
    }
    let ordinal_bits = usize::BITS as usize - max_ordinal.leading_zeros() as usize;
    let total_bits = n + ordinal_bits;
    if total_bits > 64 {
        return not_impl_err!(
            "Grouping sets with {n} columns and a maximum duplicate ordinal of \
             {max_ordinal} require {total_bits} bits, which exceeds 64"
        );
    }
    let semantic_id = group
        .iter()
        .fold(0u64, |id, &is_null| (id << 1) | u64::from(is_null));
    // A zero ordinal needs no high bits (and `n` may be 64).
    Ok(if ordinal == 0 {
        semantic_id
    } else {
        semantic_id | ((ordinal as u64) << n)
    })
}

/// The highest ordinal of a repeated grouping set: `0` if no grouping set is
/// repeated, `k - 1` if one appears `k` times (DataFusion's
/// `max_duplicate_ordinal`).
fn max_duplicate_ordinal(groups: &[Vec<bool>]) -> usize {
    let mut counts = HashMap::<&[bool], usize>::new();
    for group in groups {
        *counts.entry(group.as_slice()).or_insert(0) += 1;
    }
    counts.into_values().max().unwrap_or(0).saturating_sub(1)
}

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::array::{AsArray, Int32Array, StringArray};
    use arrow::datatypes::{DataType, Field, Int64Type, Schema};
    use datafusion::functions_aggregate::count::count_udaf;
    use datafusion_common::DataFusionError;
    use datafusion_physical_expr::aggregate::AggregateExprBuilder;
    use datafusion_physical_expr::expressions::{col, lit};
    use datafusion_physical_plan::aggregates::{
        AggregateExec, AggregateMode, PhysicalGroupBy,
    };
    use datafusion_physical_plan::empty::EmptyExec;

    use super::super::PhysicalHashAggregate;
    use crate::bridge::BridgeOperator;

    fn input_schema() -> SchemaRef {
        Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, true),
            Field::new("b", DataType::Utf8, true),
        ]))
    }

    /// `GROUP BY GROUPING SETS (groups)` over columns `a` and `b`.
    fn grouping_sets(groups: Vec<Vec<bool>>) -> Result<PhysicalGroupBy> {
        let schema = input_schema();
        Ok(PhysicalGroupBy::new(
            vec![
                (col("a", &schema)?, "a".into()),
                (col("b", &schema)?, "b".into()),
            ],
            vec![
                (lit(ScalarValue::Int32(None)), "a".into()),
                (lit(ScalarValue::Utf8(None)), "b".into()),
            ],
            groups,
            true,
        ))
    }

    /// The `AggregateSpec` of `COUNT(a)` grouped by `group_by`.
    fn spec(group_by: PhysicalGroupBy) -> Result<Arc<AggregateSpec>> {
        let schema = input_schema();
        let count = AggregateExprBuilder::new(count_udaf(), vec![col("a", &schema)?])
            .schema(Arc::clone(&schema))
            .alias("count(a)")
            .build()?;
        let aggregate = AggregateExec::try_new(
            AggregateMode::Single,
            group_by,
            vec![Arc::new(count)],
            vec![None],
            Arc::new(EmptyExec::new(Arc::clone(&schema))),
            schema,
        )?;
        let input = Arc::new(BridgeOperator::new(Arc::clone(aggregate.input())));
        let aggregate = PhysicalHashAggregate::try_new_single_stage(&aggregate, input)?;
        Ok(aggregate.spec())
    }

    fn batch() -> Result<RecordBatch> {
        Ok(RecordBatch::try_new(
            input_schema(),
            vec![
                Arc::new(Int32Array::from(vec![Some(1), None, Some(1)])),
                Arc::new(StringArray::from(vec![Some("x"), Some("y"), None])),
            ],
        )?)
    }

    #[test]
    fn grouping_ids_are_those_of_datafusion() -> Result<()> {
        // With repeated grouping sets, the ordinals go into the high bits.
        let group_sets = [
            vec![vec![false, false], vec![true, false], vec![true, true]],
            vec![
                vec![false, true],
                vec![true, true],
                vec![false, true],
                vec![true, true],
                vec![false, true],
            ],
        ];
        for groups in group_sets {
            let group_by = grouping_sets(groups.clone())?;
            let spec = spec(group_by.clone())?;
            let grouping_id_type = spec.group_schema.field(2).data_type().clone();
            let expected = evaluate_group_by(&group_by, &batch()?)?;
            let mut ordinals = HashMap::<&[bool], usize>::new();
            for (group, keys) in groups.iter().zip(&expected) {
                let ordinal = ordinals.entry(group).or_insert(0);
                let max_ordinal = max_duplicate_ordinal(&groups);
                let id =
                    ScalarValue::UInt64(Some(grouping_id(group, *ordinal, max_ordinal)?))
                        .cast_to(&grouping_id_type)?;
                *ordinal += 1;
                assert_eq!(keys[2].data_type(), &grouping_id_type);
                assert_eq!(ScalarValue::try_from_array(&keys[2], 0)?, id, "{group:?}");
            }
        }
        Ok(())
    }

    #[test]
    fn grouping_ids_fit_into_64_bits() -> Result<()> {
        let all_null = |n: usize| vec![true; n];
        // 64 columns and no repeated grouping set: all 64 bits are columns.
        assert_eq!(grouping_id(&all_null(64), 0, 0)?, u64::MAX);
        // 63 columns and the ordinal 1 of a repeated grouping set.
        assert_eq!(grouping_id(&all_null(63), 1, 1)?, u64::MAX);
        assert_eq!(grouping_id(&all_null(63), 0, 1)?, u64::MAX >> 1);
        // More bits fail like DataFusion, also for the first occurrence.
        for (n, ordinal, max_ordinal, message) in [
            (
                64,
                0,
                1,
                "64 columns and a maximum duplicate ordinal of 1 require 65 bits",
            ),
            (
                62,
                0,
                4,
                "62 columns and a maximum duplicate ordinal of 4 require 65 bits",
            ),
            (
                65,
                0,
                0,
                "Grouping sets with more than 64 columns are not supported",
            ),
        ] {
            let error = grouping_id(&all_null(n), ordinal, max_ordinal).unwrap_err();
            assert!(
                matches!(error, DataFusionError::NotImplemented(_)),
                "{error}"
            );
            assert!(error.to_string().contains(message), "{error}");
        }
        assert_eq!(max_duplicate_ordinal(&[]), 0);
        assert_eq!(
            max_duplicate_ordinal(&[vec![true], vec![false], vec![true], vec![true]]),
            2
        );
        Ok(())
    }

    #[test]
    fn empty_grouping_sets_are_added_to_empty_tables() -> Result<()> {
        let group_by = grouping_sets(vec![
            vec![false, false],
            vec![true, true],
            vec![false, true],
            vec![true, true],
        ])?;
        let spec = spec(group_by)?;
        let mut table = GroupedHashTable::try_new_update(&spec)?;
        table.add_empty_grouping_sets(&spec)?;
        // One group per occurrence of the grouping set without columns.
        assert_eq!(table.num_groups(), 2);
        let states = table.emit_states()?.expect("groups");
        assert!(states.keys[0].is_null(0) && states.keys[1].is_null(0));
        assert_eq!(
            states.states[0][0].as_primitive::<Int64Type>().values(),
            &[0, 0]
        );
        assert!(table.is_empty());

        // Tables with groups get none.
        table.update(&spec, &batch()?)?;
        let groups = table.num_groups();
        table.add_empty_grouping_sets(&spec)?;
        assert_eq!(table.num_groups(), groups);

        // Neither do aggregations without grouping sets.
        let schema = input_schema();
        let spec = self::spec(PhysicalGroupBy::new_single(vec![(
            col("a", &schema)?,
            "a".into(),
        )]))?;
        let mut table = GroupedHashTable::try_new_update(&spec)?;
        table.add_empty_grouping_sets(&spec)?;
        assert!(table.is_empty());
        assert!(table.emit_states()?.is_none());
        assert!(table.evaluate()?.is_none());
        Ok(())
    }

    #[test]
    fn tables_can_aggregate_again_after_emitting_their_states() -> Result<()> {
        // Several group columns (DataFusion's `GroupValuesColumn`) and one.
        let schema = input_schema();
        for group_by in [
            PhysicalGroupBy::new_single(vec![
                (col("a", &schema)?, "a".into()),
                (col("b", &schema)?, "b".into()),
            ]),
            PhysicalGroupBy::new_single(vec![(col("b", &schema)?, "b".into())]),
        ] {
            let spec = spec(group_by)?;
            let mut table = GroupedHashTable::try_new_update(&spec)?;
            table.update(&spec, &batch()?)?;
            let groups = table.num_groups();
            let first = table.emit_states()?.expect("groups");
            assert!(table.is_empty());
            // The same keys again, twice: the same groups, counted again.
            table.update(&spec, &batch()?)?;
            table.update(&spec, &batch()?)?;
            assert_eq!(table.num_groups(), groups);
            let second = table.emit_states()?.expect("groups");
            let counts = |states: &GroupedStates| {
                states.states[0][0]
                    .as_primitive::<Int64Type>()
                    .values()
                    .iter()
                    .sum::<i64>()
            };
            assert_eq!(counts(&second), 2 * counts(&first));
        }
        Ok(())
    }

    #[test]
    fn rows_with_literal_arguments_are_aggregated_in_chunks() -> Result<()> {
        // `COUNT('...') GROUP BY a`, whose literal argument is not stored with
        // the rows, but rebuilt for chunks of their rows.
        let schema = input_schema();
        let count = AggregateExprBuilder::new(count_udaf(), vec![lit("a literal")])
            .schema(Arc::clone(&schema))
            .alias("count('a literal')")
            .build()?;
        let aggregate = AggregateExec::try_new(
            AggregateMode::Single,
            PhysicalGroupBy::new_single(vec![(col("a", &schema)?, "a".into())]),
            vec![Arc::new(count)],
            vec![None],
            Arc::new(EmptyExec::new(Arc::clone(&schema))),
            Arc::clone(&schema),
        )?;
        let input = Arc::new(BridgeOperator::new(Arc::clone(aggregate.input())));
        let spec = PhysicalHashAggregate::try_new_single_stage(&aggregate, input)?.spec();
        let rows = 5_000;
        let batch = RecordBatch::try_new(
            Arc::clone(&schema),
            vec![
                Arc::new(Int32Array::from_iter_values((0..rows).map(|i| i % 3))),
                Arc::new(StringArray::from_iter_values(
                    (0..rows).map(|i| format!("{i}")),
                )),
            ],
        )?;
        let columns = evaluate_rows(&spec, &batch)?;
        // The key only.
        assert_eq!(columns.len(), 1);
        let mut table = GroupedHashTable::try_new_merge(&spec)?;
        table.update_rows(&spec, &columns)?;
        let output = table.evaluate()?.expect("groups");
        let mut counts = output[0]
            .as_primitive::<arrow::datatypes::Int32Type>()
            .values()
            .iter()
            .zip(output[1].as_primitive::<Int64Type>().values())
            .map(|(&key, &count)| (key, count))
            .collect::<Vec<_>>();
        counts.sort_unstable();
        assert_eq!(counts, [(0, 1_667), (1, 1_667), (2, 1_666)]);
        Ok(())
    }

    #[test]
    fn filters_select_the_group_indices_of_their_rows() {
        let group_indices = (0..6).collect::<Vec<_>>();
        let all = BooleanArray::from(vec![true; 6]);
        assert_eq!(compact_group_indices(&group_indices, &all), None);
        let some = BooleanArray::from(vec![
            Some(true),
            None,
            Some(false),
            Some(true),
            None,
            Some(true),
        ]);
        assert_eq!(
            compact_group_indices(&group_indices, &some),
            Some(vec![0, 3, 5])
        );
        let none = BooleanArray::from(vec![false; 6]);
        assert_eq!(compact_group_indices(&group_indices, &none), Some(vec![]));
    }
}
