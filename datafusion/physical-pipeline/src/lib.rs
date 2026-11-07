// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

#![doc(
    html_logo_url = "https://raw.githubusercontent.com/apache/datafusion/19fe44cf2f30cbdd63d4a4f52c74055163c6cc38/docs/logos/standalone_logo/logo_original.svg",
    html_favicon_url = "https://raw.githubusercontent.com/apache/datafusion/19fe44cf2f30cbdd63d4a4f52c74055163c6cc38/docs/logos/standalone_logo/logo_original.svg"
)]
#![cfg_attr(docsrs, feature(doc_cfg))]
// Make sure fast / cheap clones on Arc are explicit:
// https://github.com/apache/datafusion/issues/11143
#![cfg_attr(not(test), deny(clippy::clone_on_ref_ptr))]
#![cfg_attr(test, allow(clippy::needless_pass_by_value))]

//! A push-based, DuckDB-inspired pipeline execution engine for DataFusion.
//!
//! Queries are planned by DataFusion as usual; [`PipelineQueryPlanner`]
//! lowers the resulting [`ExecutionPlan`](datafusion_physical_plan::ExecutionPlan)
//! into a tree of [`PhysicalOperator`]s, which is decomposed into
//! [`Pipeline`](pipeline::Pipeline)s and executed by the [`Executor`]:
//! worker threads pull [`DataChunk`]s from the source of a pipeline and push
//! them through its operators into its sink.
//!
//! See `docs/architecture.md` in this crate for the design.
//!
//! # Crate layout
//!
//! * [`chunk`] - [`DataChunk`] and [`SelectionVector`].
//! * [`operator`] - the [`PhysicalOperator`] trait and its states.
//! * [`pipeline`] - decomposition of operator trees into pipelines.
//! * [`executor`] - [`PipelineExecutor`](executor::PipelineExecutor) and
//!   [`Executor`].
//! * [`bridge`] - [`BridgeOperator`] and [`PipelineRootExec`], the bridges
//!   between DataFusion `ExecutionPlan`s and the pipeline engine.
//! * [`planner`] - [`PipelineQueryPlanner`].
//! * [`explain`] - `EXPLAIN` and metrics support.
//! * [`tuple_data`], [`sources`], [`operators`], [`join`], [`aggregate`],
//!   [`sort`] and [`window`] - native operators.

// Allows `test_utils` to refer to this crate by name, so that it can also be
// compiled as part of an integration test, see `test_utils`.
#[cfg(test)]
extern crate self as datafusion_physical_pipeline;

pub mod aggregate;
pub mod bridge;
pub mod chunk;
pub mod executor;
pub mod explain;
pub mod join;
pub mod operator;
pub mod operators;
pub mod pipeline;
pub mod planner;
pub mod sort;
pub mod sources;
#[cfg(test)]
pub mod test_utils;
pub mod tuple_data;
pub mod window;

pub use bridge::{BridgeOperator, PipelineRootExec};
pub use chunk::{DEFAULT_VECTOR_SIZE, DataChunk, SelectionVector};
pub use executor::{CollectorSink, Executor, PipelineExecutor};
pub use operator::{
    ExecutionContext, GlobalOperatorState, GlobalSinkState, GlobalSourceState,
    LocalSinkState, LocalSourceState, OperatorFinalizeResult, OperatorMetrics,
    OperatorResult, OperatorState, PhysicalOperator, SinkFinalize, SinkResult,
    SourceResult, TaskInterrupt,
};
pub use pipeline::{MetaPipeline, MetaPipelineBuilder, Pipeline, PipelineBuilder};
pub use planner::PipelineQueryPlanner;
