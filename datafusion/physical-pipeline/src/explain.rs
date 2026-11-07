// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! `EXPLAIN` and metrics support for [`PipelineRootExec`].
//!
//! * The [`DisplayAs`] implementation of [`PipelineRootExec`] renders the
//!   lowered operator tree.
//! * [`format_operator_tree`] and [`format_pipelines`] render an operator
//!   tree and the pipelines it decomposes into.
//! * [`root_metrics`] provides the metrics reported by
//!   [`ExecutionPlan::metrics`](datafusion_physical_plan::ExecutionPlan::metrics)
//!   for `EXPLAIN ANALYZE`.

use std::fmt::{self, Write};
use std::pin::Pin;
use std::sync::Arc;
use std::task::{Context, Poll};
use std::time::Duration;

use arrow::array::RecordBatch;
use arrow::datatypes::SchemaRef;
use datafusion_common::Result;
use datafusion_execution::{RecordBatchStream, SendableRecordBatchStream};
use datafusion_physical_plan::metrics::{BaselineMetrics, MetricsSet};
use datafusion_physical_plan::{DisplayAs, DisplayFormatType};
use futures::{Stream, StreamExt};

use crate::bridge::PipelineRootExec;
use crate::executor::{ExecutionMetrics, PipelineStream};
use crate::operator::PhysicalOperator;
use crate::pipeline::PipelineGraph;

impl DisplayAs for PipelineRootExec {
    fn fmt_as(&self, t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match t {
            DisplayFormatType::Default | DisplayFormatType::Verbose => {
                write!(
                    f,
                    "PipelineRootExec: root={}",
                    format_operator_tree(self.operator())
                )
            }
            DisplayFormatType::TreeRender => {
                writeln!(f, "root={}", format_operator_tree(self.operator()))
            }
        }
    }
}

/// Render an operator tree on one line, e.g.
/// `Projection(BridgeOperator: plan=DataSourceExec: ...)`.
pub fn format_operator_tree(operator: &Arc<dyn PhysicalOperator>) -> String {
    let mut out = String::new();
    write_operator_tree(operator, &mut out);
    out
}

fn write_operator_tree(operator: &Arc<dyn PhysicalOperator>, out: &mut String) {
    out.push_str(&operator.description());
    if operator.children().is_empty() {
        return;
    }
    out.push('(');
    for (i, child) in operator.children().iter().enumerate() {
        if i > 0 {
            out.push_str(", ");
        }
        write_operator_tree(child, out);
    }
    out.push(')');
}

/// Render the pipelines that the operator tree `root` decomposes into, one
/// line per meta pipeline and pipeline. The root pipeline ends in a
/// placeholder `ResultSink`.
pub fn format_pipelines(root: &Arc<dyn PhysicalOperator>) -> Result<String> {
    let sink: Arc<dyn PhysicalOperator> = Arc::new(ResultSink {
        children: vec![Arc::clone(root)],
    });
    Ok(PipelineGraph::try_new(sink)?.to_string())
}

/// Render the per-operator metrics of an execution, one line per operator.
pub fn format_execution_metrics(metrics: &ExecutionMetrics) -> String {
    let mut out = String::new();
    for (operator, metrics) in metrics.iter() {
        let _ = writeln!(
            out,
            "{}: output_rows={}, output_chunks={}, sink_rows={}, elapsed={:?}",
            operator.name(),
            metrics.output_rows(),
            metrics.output_chunks(),
            metrics.sink_rows(),
            Duration::from_nanos(metrics.elapsed_nanos())
        );
    }
    out
}

/// The metrics of a [`PipelineRootExec`]: `output_rows` of the root and the
/// total `elapsed_compute` of all operators of its executions.
pub fn root_metrics(exec: &PipelineRootExec) -> MetricsSet {
    exec.metrics_set().clone_inner()
}

/// Wrap the output stream of an execution so that it records the baseline
/// metrics of `exec`.
pub(crate) fn instrument_output_stream(
    exec: &PipelineRootExec,
    stream: PipelineStream,
) -> SendableRecordBatchStream {
    let baseline = BaselineMetrics::new(exec.metrics_set(), 0);
    Box::pin(InstrumentedStream {
        schema: RecordBatchStream::schema(&stream),
        execution: Arc::clone(stream.metrics()),
        inner: stream,
        baseline,
        reported: false,
    })
}

struct InstrumentedStream {
    schema: SchemaRef,
    inner: PipelineStream,
    execution: Arc<ExecutionMetrics>,
    baseline: BaselineMetrics,
    reported: bool,
}

impl InstrumentedStream {
    fn report_elapsed(&mut self) {
        if self.reported {
            return;
        }
        self.reported = true;
        let nanos: u64 = self
            .execution
            .iter()
            .map(|(_, metrics)| metrics.elapsed_nanos())
            .sum();
        self.baseline
            .elapsed_compute()
            .add_duration(Duration::from_nanos(nanos));
    }
}

impl Stream for InstrumentedStream {
    type Item = Result<RecordBatch>;

    fn poll_next(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = self.get_mut();
        let poll = this.inner.poll_next_unpin(cx);
        if matches!(poll, Poll::Ready(None | Some(Err(_)))) {
            this.report_elapsed();
        }
        this.baseline.record_poll(poll)
    }
}

impl RecordBatchStream for InstrumentedStream {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }
}

/// Placeholder sink used to render the pipelines of an operator tree.
#[derive(Debug)]
struct ResultSink {
    children: Vec<Arc<dyn PhysicalOperator>>,
}

impl PhysicalOperator for ResultSink {
    fn name(&self) -> &str {
        "ResultSink"
    }

    fn schema(&self) -> SchemaRef {
        self.children[0].schema()
    }

    fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.children
    }

    fn is_sink(&self) -> bool {
        true
    }
}
