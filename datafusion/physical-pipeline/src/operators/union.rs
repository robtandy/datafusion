// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! `PhysicalUnion`: parallel `UNION ALL` over several child pipelines.
//!
//! [`PhysicalUnion`] replaces DataFusion's `UnionExec` and `InterleaveExec`.
//! Like DuckDB's `PhysicalUnion`, it does not move any data: it only builds
//! pipelines (see [`MetaPipelineBuilder::build_union`]). The first input
//! continues the pipeline of the operators above the union, and every other
//! input gets a union pipeline that shares these operators and the sink.
//! Unless the order of the inputs must be kept
//! ([`PhysicalUnion::with_preserve_order`]), the pipelines of all inputs run
//! concurrently, each on all worker threads, so the chunks of all inputs are
//! pushed through the operators above the union in parallel, without any
//! channel between the inputs and their consumer.
//!
//! The schema of the union is the schema of `UnionExec`: the field names of
//! the first input, a field is nullable if it is nullable in any input, and
//! the field and schema metadata of all inputs are merged. The chunks of an
//! input whose fields differ from the fields of the union (in their names,
//! nullability or metadata: the data types must be equal) are relabeled with
//! the fields of the union by a `UnionCoercion` operator, without copying any
//! data, like the `ProjectionExec` that `UnionExec` puts above such an input.
//! Like that projection, it keeps the schema metadata of the input: the
//! batches of DataFusion's union carry the schema metadata of the input they
//! come from, not the merged schema metadata of the union.

use std::sync::Arc;

use arrow::datatypes::{Schema, SchemaRef};
use datafusion_common::metadata::merge_metadata_for_union;
use datafusion_common::{Result, exec_err, internal_err, plan_err};
use datafusion_physical_plan::ExecutionPlan;
use datafusion_physical_plan::union::{InterleaveExec, UnionExec};

use crate::chunk::DataChunk;
use crate::operator::{
    ExecutionContext, GlobalOperatorState, OperatorResult, OperatorState,
    PhysicalOperator,
};
use crate::pipeline::{MetaPipelineBuilder, PipelineBuilder};

/// `UNION ALL` of any number of inputs.
///
/// See the [module documentation](self) for its pipelines and schema.
#[derive(Debug)]
pub struct PhysicalUnion {
    /// The inputs. An input whose fields are not the fields of the union is
    /// wrapped in a `UnionCoercion`.
    children: Vec<Arc<dyn PhysicalOperator>>,
    schema: SchemaRef,
    preserve_order: bool,
}

impl PhysicalUnion {
    /// The union of `inputs`, with the schema that DataFusion's `UnionExec`
    /// has for inputs with the schemas of `inputs`.
    pub fn try_new(inputs: Vec<Arc<dyn PhysicalOperator>>) -> Result<Self> {
        let schema = union_schema(&inputs)?;
        Self::try_new_with_schema(inputs, schema)
    }

    /// The union of `inputs`, with schema `schema` (e.g. the schema of the
    /// `UnionExec` being lowered). Every input must have the data types of
    /// `schema`; the field names, nullability and metadata may differ.
    ///
    /// The chunks of an input keep its schema metadata, see the
    /// [module documentation](self).
    pub fn try_new_with_schema(
        inputs: Vec<Arc<dyn PhysicalOperator>>,
        schema: SchemaRef,
    ) -> Result<Self> {
        if inputs.is_empty() {
            return internal_err!("PhysicalUnion requires at least one input");
        }
        let children = inputs
            .into_iter()
            .enumerate()
            .map(|(index, input)| coerce_input(index, input, &schema))
            .collect::<Result<Vec<_>>>()?;
        Ok(Self {
            children,
            schema,
            preserve_order: false,
        })
    }

    /// If `preserve_order` is `true`, the pipelines of every input only start
    /// once all pipelines of the inputs before it completed, so that the
    /// output comes in input order (if the operators above the union keep
    /// it): all rows of the first input, then all rows of the second one,
    /// and so on. Otherwise all inputs are executed concurrently.
    pub fn with_preserve_order(mut self, preserve_order: bool) -> Self {
        self.preserve_order = preserve_order;
        self
    }

    /// Lower a DataFusion `UnionExec` or `InterleaveExec` to a
    /// [`PhysicalUnion`] with the same schema. `lower_input` lowers the
    /// inputs of the node. Returns `Ok(None)` for any other node.
    ///
    /// `InterleaveExec` combines hash partitioned inputs partition by
    /// partition; the pipeline engine has no partitions, so it is a union of
    /// its inputs too.
    pub fn try_lower<F>(
        plan: &Arc<dyn ExecutionPlan>,
        lower_input: F,
    ) -> Result<Option<Self>>
    where
        F: FnMut(&Arc<dyn ExecutionPlan>) -> Result<Arc<dyn PhysicalOperator>>,
    {
        let inputs = if let Some(union) = plan.downcast_ref::<UnionExec>() {
            union.inputs()
        } else if let Some(interleave) = plan.downcast_ref::<InterleaveExec>() {
            interleave.inputs()
        } else {
            return Ok(None);
        };
        let inputs = inputs.iter().map(lower_input).collect::<Result<Vec<_>>>()?;
        Self::try_new_with_schema(inputs, plan.schema()).map(Some)
    }

    /// Whether the inputs are executed one after the other, in order, see
    /// [`Self::with_preserve_order`].
    pub fn preserve_order(&self) -> bool {
        self.preserve_order
    }
}

impl PhysicalOperator for PhysicalUnion {
    fn name(&self) -> &str {
        "PhysicalUnion"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.children
    }

    fn description(&self) -> String {
        if self.preserve_order {
            "PhysicalUnion: preserve_order=true".to_string()
        } else {
            "PhysicalUnion".to_string()
        }
    }

    fn build_pipelines(
        &self,
        current: &mut PipelineBuilder,
        meta: &mut MetaPipelineBuilder<'_>,
    ) -> Result<()> {
        meta.build_union(current, &self.children, self.preserve_order)
    }
}

/// The schema of DataFusion's `UnionExec` for inputs with the schemas of
/// `inputs` (`union_schema` in `datafusion_physical_plan::union`): the names
/// of the fields of the first input; the first nullable field among the
/// inputs (or the field of the first input if none is nullable), with the
/// merged metadata of the fields of all inputs; the merged metadata of all
/// schemas.
fn union_schema(inputs: &[Arc<dyn PhysicalOperator>]) -> Result<SchemaRef> {
    let Some(first) = inputs.first() else {
        return internal_err!("PhysicalUnion requires at least one input");
    };
    let first_schema = first.schema();
    let schemas = inputs
        .iter()
        .map(|input| input.schema())
        .collect::<Vec<_>>();
    if schemas
        .iter()
        .all(|schema| Arc::ptr_eq(schema, &first_schema) || schema == &first_schema)
    {
        return Ok(first_schema);
    }
    let num_fields = first_schema.fields().len();
    for (index, schema) in schemas.iter().enumerate() {
        check_field_count(index, schema, num_fields)?;
    }
    let fields = (0..num_fields)
        .map(|i| {
            let field = schemas
                .iter()
                .find(|schema| schema.field(i).is_nullable())
                .unwrap_or(&first_schema)
                .field(i);
            let metadata =
                merge_metadata_for_union(schemas.iter().map(|s| s.field(i).metadata()));
            field
                .clone()
                .with_metadata(metadata)
                .with_name(first_schema.field(i).name())
        })
        .collect::<Vec<_>>();
    let metadata = merge_metadata_for_union(schemas.iter().map(|s| s.metadata()));
    Ok(Arc::new(Schema::new_with_metadata(fields, metadata)))
}

fn check_field_count(index: usize, schema: &Schema, expected: usize) -> Result<()> {
    let num_fields = schema.fields().len();
    if num_fields != expected {
        return exec_err!(
            "PhysicalUnion requires all inputs to have the same number of fields as the \
             union ({expected}), but input {index} has {num_fields} fields"
        );
    }
    Ok(())
}

/// `input`, the input at `index` of a union with schema `schema`, wrapped in
/// a [`UnionCoercion`] if its fields are not the fields of `schema`.
///
/// Like `UnionExec`, an input whose schema differs only in its metadata is
/// not changed: DataFusion's union passes the batches of its inputs through,
/// and the projection that coerces the fields of an input (`coerce_schema` in
/// `datafusion_physical_plan::union`) keeps the schema metadata of the input
/// too.
fn coerce_input(
    index: usize,
    input: Arc<dyn PhysicalOperator>,
    schema: &SchemaRef,
) -> Result<Arc<dyn PhysicalOperator>> {
    let input_schema = input.schema();
    if input_schema.fields() == schema.fields() {
        return Ok(input);
    }
    check_field_count(index, &input_schema, schema.fields().len())?;
    for (i, (input_field, field)) in input_schema
        .fields()
        .iter()
        .zip(schema.fields())
        .enumerate()
    {
        if input_field.data_type() != field.data_type() {
            return plan_err!(
                "PhysicalUnion requires all inputs to have the data types of the union; \
                 column {i} of input {index} has type {}, but the union expects {}",
                input_field.data_type(),
                field.data_type()
            );
        }
    }
    Ok(Arc::new(UnionCoercion {
        children: vec![input],
        schema: Arc::new(Schema::new_with_metadata(
            schema.fields().clone(),
            input_schema.metadata().clone(),
        )),
    }))
}

/// Relabels the chunks of a union input with the fields of the union: same
/// columns (and selection), but the field names, nullability and metadata of
/// the union. No data is copied.
#[derive(Debug)]
struct UnionCoercion {
    children: Vec<Arc<dyn PhysicalOperator>>,
    schema: SchemaRef,
}

impl PhysicalOperator for UnionCoercion {
    fn name(&self) -> &str {
        "UnionCoercion"
    }

    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.children
    }

    fn execute(
        &self,
        _ctx: &ExecutionContext,
        input: &DataChunk,
        output: &mut DataChunk,
        _gstate: &dyn GlobalOperatorState,
        _lstate: &mut dyn OperatorState,
    ) -> Result<OperatorResult> {
        // The fields of a coerced input differ, so it has columns, and
        // `input` is not empty.
        *output = DataChunk::try_new(
            Arc::clone(&self.schema),
            input.columns().to_vec(),
            input.selection().cloned(),
        )?;
        Ok(OperatorResult::NeedMoreInput)
    }
}
