// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! [`RoutedPartitionExec`] and [`PhysicalRoutedPartition`]: the rows of one
//! output partition of a hash or range `RepartitionExec`, which the planner
//! splits a repartitioning into to keep its partitions apart (see
//! `partitions_of` in `partial_sort.rs`).

use std::fmt;
use std::sync::Arc;

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use datafusion_common::tree_node::TreeNodeRecursion;
use datafusion_common::{Result, internal_err};
use datafusion_execution::{SendableRecordBatchStream, TaskContext};
use datafusion_physical_expr::Partitioning;
use datafusion_physical_expr_common::physical_expr::PhysicalExpr;
use datafusion_physical_plan::metrics::Time;
use datafusion_physical_plan::repartition::BatchPartitioner;
use datafusion_physical_plan::stream::RecordBatchStreamAdapter;
use datafusion_physical_plan::{
    ChildrenPropertiesMode, DisplayAs, DisplayFormatType, ExecutionPlan,
    ExecutionPlanProperties, PlanProperties, ReplaceChildrenOptions,
    apply_expression_roots,
};
use futures::StreamExt;

use crate::chunk::DataChunk;
use crate::operator::{
    ExecutionContext, GlobalOperatorState, OperatorResult, OperatorState,
    PhysicalOperator, downcast_state_mut,
};

/// The partitioner of `partitioning` (hash or range), which routes the rows
/// of a batch to the partitions of the repartitioning as DataFusion's
/// `RepartitionExec` does.
fn partitioner(partitioning: &Partitioning) -> Result<BatchPartitioner> {
    BatchPartitioner::try_new(partitioning.clone(), Time::new(), 0, 1)
}

/// The rows of every partition of `input` that the hash or range
/// repartitioning `partitioning` routes to its output partition `partition`,
/// in their order: partition `p` of the node has the rows of partition `p`
/// of `input` that a `RepartitionExec` with `partitioning` sends to
/// `partition`. The planner lowers it to a [`PhysicalRoutedPartition`].
#[derive(Debug)]
pub(crate) struct RoutedPartitionExec {
    input: Arc<dyn ExecutionPlan>,
    partitioning: Partitioning,
    partition: usize,
    properties: Arc<PlanProperties>,
}

impl RoutedPartitionExec {
    /// The rows of `input` that `partitioning`, a hash or range
    /// partitioning, routes to its partition `partition`.
    pub(crate) fn try_new(
        input: Arc<dyn ExecutionPlan>,
        partitioning: Partitioning,
        partition: usize,
    ) -> Result<Self> {
        if !matches!(
            partitioning,
            Partitioning::Hash(..) | Partitioning::Range(_)
        ) || partition >= partitioning.partition_count()
        {
            return internal_err!(
                "RoutedPartitionExec routes rows to partition {partition} of {partitioning}"
            );
        }
        let properties = PlanProperties::new(
            input.equivalence_properties().clone(),
            Partitioning::UnknownPartitioning(
                input.output_partitioning().partition_count(),
            ),
            input.pipeline_behavior(),
            input.boundedness(),
        );
        Ok(Self {
            input,
            partitioning,
            partition,
            properties: Arc::new(properties),
        })
    }

    /// The expressions that route the rows.
    fn exprs(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        match &self.partitioning {
            Partitioning::Hash(exprs, _) => exprs.clone(),
            Partitioning::Range(range) => range
                .ordering()
                .iter()
                .map(|key| Arc::clone(&key.expr))
                .collect(),
            _ => vec![],
        }
    }
}

impl DisplayAs for RoutedPartitionExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "RoutedPartitionExec: partitioning={}, partition={}",
            self.partitioning, self.partition
        )
    }
}

impl ExecutionPlan for RoutedPartitionExec {
    fn name(&self) -> &str {
        "RoutedPartitionExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn apply_expressions(
        &self,
        f: &mut dyn FnMut(&Arc<dyn PhysicalExpr>) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        apply_expression_roots(&self.exprs(), f)
    }

    fn replace_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
        _options: ReplaceChildrenOptions,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let [input] = <[_; 1]>::try_from(children).map_err(|children| {
            datafusion_common::internal_datafusion_err!(
                "RoutedPartitionExec has one child, got {}",
                children.len()
            )
        })?;
        Ok(Arc::new(Self::try_new(
            input,
            self.partitioning.clone(),
            self.partition,
        )?))
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        self.replace_children(
            children,
            ReplaceChildrenOptions::new(ChildrenPropertiesMode::Recompute),
        )
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let mut partitioner = partitioner(&self.partitioning)?;
        let target = self.partition;
        let input = self.input.execute(partition, context)?;
        let routed = input.filter_map(move |batch| {
            let routed = batch.and_then(|batch| route(&mut partitioner, batch, target));
            futures::future::ready(routed.transpose())
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema(),
            routed,
        )))
    }
}

/// The rows of `batch` that `partitioner` routes to `partition`, if any.
fn route(
    partitioner: &mut BatchPartitioner,
    batch: RecordBatch,
    partition: usize,
) -> Result<Option<RecordBatch>> {
    let mut routed = None;
    for part in partitioner.partition_iter(batch)? {
        let (index, rows) = part?;
        if index == partition && rows.num_rows() > 0 {
            routed = Some(rows);
        }
    }
    Ok(routed)
}

/// Keeps the rows of its input that a hash or range repartitioning routes to
/// one of its output partitions, as [`RoutedPartitionExec`] does (with
/// DataFusion's `BatchPartitioner`), on all threads of its pipeline. It holds
/// no rows.
#[derive(Debug)]
pub struct PhysicalRoutedPartition {
    children: Vec<Arc<dyn PhysicalOperator>>,
    partitioning: Partitioning,
    partition: usize,
}

impl PhysicalRoutedPartition {
    /// Lower a [`RoutedPartitionExec`] to a [`PhysicalRoutedPartition`] of
    /// `lower_input` of its input. Returns `Ok(None)` for any other node,
    /// without calling `lower_input`.
    pub(crate) fn try_lower<F>(
        plan: &Arc<dyn ExecutionPlan>,
        lower_input: F,
    ) -> Result<Option<Self>>
    where
        F: FnOnce(&Arc<dyn ExecutionPlan>) -> Result<Arc<dyn PhysicalOperator>>,
    {
        let Some(routed) = plan.downcast_ref::<RoutedPartitionExec>() else {
            return Ok(None);
        };
        Ok(Some(Self {
            children: vec![lower_input(&routed.input)?],
            partitioning: routed.partitioning.clone(),
            partition: routed.partition,
        }))
    }
}

/// The partitioner of a thread of a [`PhysicalRoutedPartition`].
struct RoutingState(BatchPartitioner);

impl OperatorState for RoutingState {}

impl PhysicalOperator for PhysicalRoutedPartition {
    fn name(&self) -> &str {
        "PhysicalRoutedPartition"
    }

    fn schema(&self) -> SchemaRef {
        self.children[0].schema()
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.children
    }

    fn description(&self) -> String {
        format!(
            "PhysicalRoutedPartition: partitioning={}, partition={}",
            self.partitioning, self.partition
        )
    }

    fn get_operator_state(
        &self,
        _ctx: &ExecutionContext,
        _gstate: &dyn GlobalOperatorState,
    ) -> Result<Box<dyn OperatorState>> {
        Ok(Box::new(RoutingState(partitioner(&self.partitioning)?)))
    }

    fn execute(
        &self,
        _ctx: &ExecutionContext,
        input: &DataChunk,
        output: &mut DataChunk,
        _gstate: &dyn GlobalOperatorState,
        lstate: &mut dyn OperatorState,
    ) -> Result<OperatorResult> {
        let state = downcast_state_mut::<RoutingState>(lstate.as_any_mut(), self.name())?;
        let batch = input.compact()?;
        let rows = route(&mut state.0, batch, self.partition)?
            .unwrap_or_else(|| RecordBatch::new_empty(self.schema()));
        *output = DataChunk::from_record_batch(rows);
        Ok(OperatorResult::NeedMoreInput)
    }
}
