// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Decomposition of a [`PhysicalOperator`] tree into [`Pipeline`]s.
//!
//! A [`Pipeline`] is a chain `source -> operators... -> sink` that is
//! executed by pushing [`DataChunk`](crate::chunk::DataChunk)s from the
//! source through the operators into the sink, without materializing
//! intermediate results.
//!
//! Pipelines are grouped into [`MetaPipeline`]s: all pipelines of a meta
//! pipeline share the same sink (e.g. the two inputs of a `UNION ALL` that
//! feeds an aggregate), and the sink is finalized once all of them completed.
//! A meta pipeline has child meta pipelines for the pipeline breakers below
//! it (e.g. the build side of a hash join); a pipeline that reads from a
//! pipeline breaker depends on the child meta pipeline of that breaker and
//! only starts once that child is finalized.
//!
//! The decomposition is driven by [`PhysicalOperator::build_pipelines`],
//! modelled after DuckDB's `PhysicalOperator::BuildPipelines`, through the
//! [`MetaPipelineBuilder`] API.

use std::fmt::{self, Display};
use std::sync::Arc;

use datafusion_common::{Result, internal_datafusion_err, internal_err};

use crate::operator::{PhysicalOperator, operator_key};

/// Index of a [`Pipeline`] in a [`PipelineGraph`].
pub type PipelineId = usize;

/// Index of a [`MetaPipeline`] in a [`PipelineGraph`].
pub type MetaPipelineId = usize;

/// A chain of operators executed as one unit: `source -> operators -> sink`.
#[derive(Debug)]
pub struct Pipeline {
    id: PipelineId,
    meta_pipeline: MetaPipelineId,
    source: Arc<dyn PhysicalOperator>,
    operators: Vec<Arc<dyn PhysicalOperator>>,
    sink: Arc<dyn PhysicalOperator>,
    dependencies: Vec<PipelineId>,
    meta_dependencies: Vec<MetaPipelineId>,
}

impl Pipeline {
    /// The id of this pipeline.
    pub fn id(&self) -> PipelineId {
        self.id
    }

    /// The meta pipeline this pipeline belongs to.
    pub fn meta_pipeline(&self) -> MetaPipelineId {
        self.meta_pipeline
    }

    /// The source of the pipeline.
    pub fn source(&self) -> &Arc<dyn PhysicalOperator> {
        &self.source
    }

    /// The streaming operators of the pipeline, in execution order (from the
    /// source to the sink).
    pub fn operators(&self) -> &[Arc<dyn PhysicalOperator>] {
        &self.operators
    }

    /// The sink of the pipeline (the sink of its meta pipeline).
    pub fn sink(&self) -> &Arc<dyn PhysicalOperator> {
        &self.sink
    }

    /// Pipelines of the same meta pipeline that must complete before this
    /// pipeline starts.
    pub fn dependencies(&self) -> &[PipelineId] {
        &self.dependencies
    }

    /// Child meta pipelines that must be finalized before this pipeline
    /// starts.
    pub fn meta_dependencies(&self) -> &[MetaPipelineId] {
        &self.meta_dependencies
    }
}

impl Display for Pipeline {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.source.name())?;
        for operator in &self.operators {
            write!(f, " -> {}", operator.name())?;
        }
        write!(f, " -> {}", self.sink.name())
    }
}

/// A set of pipelines that share the same sink.
#[derive(Debug)]
pub struct MetaPipeline {
    id: MetaPipelineId,
    sink: Arc<dyn PhysicalOperator>,
    pipelines: Vec<PipelineId>,
    children: Vec<MetaPipelineId>,
    parent: Option<MetaPipelineId>,
}

impl MetaPipeline {
    /// The id of this meta pipeline.
    pub fn id(&self) -> MetaPipelineId {
        self.id
    }

    /// The sink shared by all pipelines of this meta pipeline.
    pub fn sink(&self) -> &Arc<dyn PhysicalOperator> {
        &self.sink
    }

    /// The pipelines of this meta pipeline, in creation order. The first one
    /// is the base pipeline.
    pub fn pipelines(&self) -> &[PipelineId] {
        &self.pipelines
    }

    /// The child meta pipelines (one per pipeline breaker below this one).
    pub fn children(&self) -> &[MetaPipelineId] {
        &self.children
    }

    /// The parent meta pipeline, `None` for the root.
    pub fn parent(&self) -> Option<MetaPipelineId> {
        self.parent
    }
}

/// The pipelines and meta pipelines of a query.
///
/// Meta pipeline `0` is the root: its sink is the sink the graph was built
/// for (e.g. a [`CollectorSink`](crate::executor::CollectorSink)).
#[derive(Debug)]
pub struct PipelineGraph {
    pipelines: Vec<Pipeline>,
    meta_pipelines: Vec<MetaPipeline>,
}

impl PipelineGraph {
    /// Decompose the operator tree below `sink` into pipelines.
    ///
    /// `sink` must be a sink with exactly one child; it becomes the sink of
    /// the root meta pipeline.
    pub fn try_new(sink: Arc<dyn PhysicalOperator>) -> Result<Self> {
        if !sink.is_sink() {
            return internal_err!(
                "The root of a pipeline graph must be a sink, got {}",
                sink.name()
            );
        }
        let [input] = sink.children() else {
            return internal_err!(
                "The root sink of a pipeline graph must have exactly one child, {} has {}",
                sink.name(),
                sink.children().len()
            );
        };
        let input = Arc::clone(input);
        let mut graph = GraphBuilder::default();
        let (root, mut base) = graph.new_meta_pipeline(sink, None);
        MetaPipelineBuilder {
            graph: &mut graph,
            meta: root,
        }
        .build(&mut base, &input)?;
        graph.finish()
    }

    /// All pipelines, indexed by [`PipelineId`].
    pub fn pipelines(&self) -> &[Pipeline] {
        &self.pipelines
    }

    /// The pipeline with id `id`.
    pub fn pipeline(&self, id: PipelineId) -> &Pipeline {
        &self.pipelines[id]
    }

    /// All meta pipelines, indexed by [`MetaPipelineId`].
    pub fn meta_pipelines(&self) -> &[MetaPipeline] {
        &self.meta_pipelines
    }

    /// The meta pipeline with id `id`.
    pub fn meta_pipeline(&self, id: MetaPipelineId) -> &MetaPipeline {
        &self.meta_pipelines[id]
    }

    /// The root meta pipeline.
    pub fn root(&self) -> &MetaPipeline {
        &self.meta_pipelines[0]
    }

    /// All distinct operators of the graph (sources, operators and sinks),
    /// each listed once, in pipeline order.
    pub fn operators(&self) -> Vec<Arc<dyn PhysicalOperator>> {
        let mut seen = std::collections::HashSet::new();
        let mut operators = vec![];
        for pipeline in &self.pipelines {
            let all = std::iter::once(&pipeline.source)
                .chain(&pipeline.operators)
                .chain(std::iter::once(&pipeline.sink));
            for operator in all {
                if seen.insert(operator_key(operator.as_ref())) {
                    operators.push(Arc::clone(operator));
                }
            }
        }
        operators
    }
}

impl Display for PipelineGraph {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for meta in &self.meta_pipelines {
            write!(f, "MetaPipeline {} (sink: {})", meta.id, meta.sink.name())?;
            if !meta.children.is_empty() {
                write!(f, " children: {:?}", meta.children)?;
            }
            writeln!(f)?;
            for &id in &meta.pipelines {
                let pipeline = &self.pipelines[id];
                write!(f, "  Pipeline {id}: {pipeline}")?;
                if !pipeline.dependencies.is_empty() {
                    write!(f, " depends on pipelines {:?}", pipeline.dependencies)?;
                }
                if !pipeline.meta_dependencies.is_empty() {
                    write!(
                        f,
                        " depends on meta pipelines {:?}",
                        pipeline.meta_dependencies
                    )?;
                }
                writeln!(f)?;
            }
        }
        Ok(())
    }
}

/// Handle to a pipeline under construction, see [`MetaPipelineBuilder`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PipelineBuilder {
    id: PipelineId,
}

impl PipelineBuilder {
    /// The id the pipeline will have in the [`PipelineGraph`].
    pub fn id(&self) -> PipelineId {
        self.id
    }
}

/// A pipeline under construction. `operators` are stored from the sink side
/// to the source side, the order in which `build_pipelines` visits them.
struct PipelineState {
    meta: MetaPipelineId,
    source: Option<Arc<dyn PhysicalOperator>>,
    operators: Vec<Arc<dyn PhysicalOperator>>,
    dependencies: Vec<PipelineId>,
    meta_dependencies: Vec<MetaPipelineId>,
}

struct MetaPipelineState {
    sink: Arc<dyn PhysicalOperator>,
    pipelines: Vec<PipelineId>,
    children: Vec<MetaPipelineId>,
    parent: Option<MetaPipelineId>,
}

#[derive(Default)]
struct GraphBuilder {
    pipelines: Vec<PipelineState>,
    meta_pipelines: Vec<MetaPipelineState>,
    /// The operators whose `build_pipelines` is running, innermost last.
    visiting: Vec<Arc<dyn PhysicalOperator>>,
}

impl GraphBuilder {
    fn new_meta_pipeline(
        &mut self,
        sink: Arc<dyn PhysicalOperator>,
        parent: Option<MetaPipelineId>,
    ) -> (MetaPipelineId, PipelineBuilder) {
        let meta = self.meta_pipelines.len();
        self.meta_pipelines.push(MetaPipelineState {
            sink,
            pipelines: vec![],
            children: vec![],
            parent,
        });
        let base = self.new_pipeline(meta);
        (meta, base)
    }

    fn new_pipeline(&mut self, meta: MetaPipelineId) -> PipelineBuilder {
        let id = self.pipelines.len();
        self.pipelines.push(PipelineState {
            meta,
            source: None,
            operators: vec![],
            dependencies: vec![],
            meta_dependencies: vec![],
        });
        self.meta_pipelines[meta].pipelines.push(id);
        PipelineBuilder { id }
    }

    fn finish(self) -> Result<PipelineGraph> {
        let meta_pipelines = self
            .meta_pipelines
            .into_iter()
            .enumerate()
            .map(|(id, meta)| MetaPipeline {
                id,
                sink: meta.sink,
                pipelines: meta.pipelines,
                children: meta.children,
                parent: meta.parent,
            })
            .collect::<Vec<_>>();
        let pipelines = self
            .pipelines
            .into_iter()
            .enumerate()
            .map(|(id, state)| {
                let sink = Arc::clone(&meta_pipelines[state.meta].sink);
                let Some(source) = state.source else {
                    return internal_err!(
                        "Pipeline {id} into {} has no source",
                        sink.name()
                    );
                };
                let mut operators = state.operators;
                operators.reverse();
                Ok(Pipeline {
                    id,
                    meta_pipeline: state.meta,
                    source,
                    operators,
                    sink,
                    dependencies: state.dependencies,
                    meta_dependencies: state.meta_dependencies,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(PipelineGraph {
            pipelines,
            meta_pipelines,
        })
    }
}

/// Builds the pipelines of one [`MetaPipeline`].
///
/// A `MetaPipelineBuilder` is handed to
/// [`PhysicalOperator::build_pipelines`] together with the pipeline under
/// construction (`current`). An implementation typically adds itself to
/// `current` (as source or operator), creates child meta pipelines or extra
/// pipelines as needed, and recurses into its children with
/// [`Self::build`].
///
/// The operator whose `build_pipelines` is running is available as
/// [`Self::current_operator`], so implementations do not need an `Arc` of
/// themselves.
pub struct MetaPipelineBuilder<'a> {
    graph: &'a mut GraphBuilder,
    meta: MetaPipelineId,
}

impl MetaPipelineBuilder<'_> {
    /// The id of the meta pipeline being built.
    pub fn id(&self) -> MetaPipelineId {
        self.meta
    }

    /// The sink of the meta pipeline being built.
    pub fn sink(&self) -> Arc<dyn PhysicalOperator> {
        Arc::clone(&self.graph.meta_pipelines[self.meta].sink)
    }

    /// The operator whose [`PhysicalOperator::build_pipelines`] is running.
    pub fn current_operator(&self) -> Result<Arc<dyn PhysicalOperator>> {
        self.graph.visiting.last().map(Arc::clone).ok_or_else(|| {
            internal_datafusion_err!("No operator is being added to a pipeline")
        })
    }

    /// Continue building `current` with `operator`: calls
    /// `operator.build_pipelines(current, self)`.
    pub fn build(
        &mut self,
        current: &mut PipelineBuilder,
        operator: &Arc<dyn PhysicalOperator>,
    ) -> Result<()> {
        self.check_pipeline(current)?;
        self.graph.visiting.push(Arc::clone(operator));
        let result = operator.build_pipelines(current, self);
        self.graph.visiting.pop();
        result
    }

    /// Set the source of `current`.
    pub fn set_source(
        &mut self,
        current: &PipelineBuilder,
        source: Arc<dyn PhysicalOperator>,
    ) -> Result<()> {
        self.check_pipeline(current)?;
        if !source.is_source() {
            return internal_err!(
                "{} cannot be the source of a pipeline: it is not a source",
                source.name()
            );
        }
        let pipeline = &mut self.graph.pipelines[current.id];
        if let Some(existing) = &pipeline.source {
            return internal_err!(
                "Pipeline {} already has source {}, cannot set {}",
                current.id,
                existing.name(),
                source.name()
            );
        }
        pipeline.source = Some(source);
        Ok(())
    }

    /// Add a streaming operator to `current`. Operators must be added from
    /// the sink side to the source side.
    pub fn add_operator(
        &mut self,
        current: &PipelineBuilder,
        operator: Arc<dyn PhysicalOperator>,
    ) -> Result<()> {
        self.check_pipeline(current)?;
        let pipeline = &mut self.graph.pipelines[current.id];
        if pipeline.source.is_some() {
            return internal_err!(
                "Cannot add operator {} to pipeline {}: its source is already set",
                operator.name(),
                current.id
            );
        }
        pipeline.operators.push(operator);
        Ok(())
    }

    /// Create a child meta pipeline whose sink is `sink`, and make `current`
    /// depend on it: `current` only starts once the child meta pipeline is
    /// finalized.
    ///
    /// Returns the builder of the child meta pipeline and its (empty) base
    /// pipeline, into which the input of `sink` is typically built with
    /// [`Self::build`].
    pub fn create_child_meta_pipeline(
        &mut self,
        current: &PipelineBuilder,
        sink: Arc<dyn PhysicalOperator>,
    ) -> Result<(MetaPipelineBuilder<'_>, PipelineBuilder)> {
        self.check_pipeline(current)?;
        if !sink.is_sink() {
            return internal_err!(
                "{} cannot be the sink of a meta pipeline: it is not a sink",
                sink.name()
            );
        }
        let (child, base) = self.graph.new_meta_pipeline(sink, Some(self.meta));
        self.graph.meta_pipelines[self.meta].children.push(child);
        self.graph.pipelines[current.id]
            .meta_dependencies
            .push(child);
        Ok((
            MetaPipelineBuilder {
                graph: &mut *self.graph,
                meta: child,
            },
            base,
        ))
    }

    /// Create a pipeline for another input of a union-like operator.
    ///
    /// The new pipeline belongs to this meta pipeline, has the operators
    /// that were added to `current` so far (the operators above the union)
    /// and inherits all dependencies of `current`. Its source is set by
    /// building the other input into it. If `order_matters`, the new
    /// pipeline only starts after `current` completed.
    pub fn create_union_pipeline(
        &mut self,
        current: &PipelineBuilder,
        order_matters: bool,
    ) -> Result<PipelineBuilder> {
        self.check_pipeline(current)?;
        let union = self.graph.new_pipeline(self.meta);
        let current_state = &self.graph.pipelines[current.id];
        let operators = current_state.operators.clone();
        let mut dependencies = current_state.dependencies.clone();
        let meta_dependencies = current_state.meta_dependencies.clone();
        if order_matters {
            dependencies.push(current.id);
        }
        let state = &mut self.graph.pipelines[union.id];
        state.operators = operators;
        state.dependencies = dependencies;
        state.meta_dependencies = meta_dependencies;
        Ok(union)
    }

    /// Create a pipeline whose source is `source`, an operator of `current`
    /// that also produces data once `current` is complete (e.g. the
    /// unmatched build rows of a right / full outer hash join).
    ///
    /// `current` must be fully built (its source set). The new pipeline
    /// belongs to this meta pipeline, has the operators of `current` that
    /// come after `source`, and depends on `current` and on all pipelines of
    /// this meta pipeline created after `last_pipeline` (typically the last
    /// pipeline of [`Self::pipelines`] before the input of `source` was
    /// built).
    pub fn create_child_pipeline(
        &mut self,
        current: &PipelineBuilder,
        source: Arc<dyn PhysicalOperator>,
        last_pipeline: &PipelineBuilder,
    ) -> Result<PipelineBuilder> {
        self.check_pipeline(current)?;
        self.check_pipeline(last_pipeline)?;
        if !source.is_source() {
            return internal_err!(
                "{} cannot be the source of a pipeline: it is not a source",
                source.name()
            );
        }
        let current_state = &self.graph.pipelines[current.id];
        if current_state.source.is_none() {
            return internal_err!(
                "Pipeline {} must be complete before creating a child pipeline",
                current.id
            );
        }
        let source_key = operator_key(source.as_ref());
        let Some(position) = current_state
            .operators
            .iter()
            .position(|op| operator_key(op.as_ref()) == source_key)
        else {
            return internal_err!(
                "{} is not an operator of pipeline {}",
                source.name(),
                current.id
            );
        };
        // Operators are stored sink side first: keep the ones above `source`.
        let operators = current_state.operators[..position].to_vec();

        let siblings = &self.graph.meta_pipelines[self.meta].pipelines;
        let start = siblings
            .iter()
            .position(|&id| id == last_pipeline.id)
            .map_or(siblings.len(), |p| p + 1);
        let mut dependencies = vec![current.id];
        dependencies.extend(
            siblings[start..]
                .iter()
                .copied()
                .filter(|&id| id != current.id),
        );

        let child = self.graph.new_pipeline(self.meta);
        let state = &mut self.graph.pipelines[child.id];
        state.source = Some(source);
        state.operators = operators;
        state.dependencies = dependencies;
        Ok(child)
    }

    /// Make `pipeline` start only after `dependency` (another pipeline of
    /// this meta pipeline) completed.
    pub fn add_dependency(
        &mut self,
        pipeline: &PipelineBuilder,
        dependency: &PipelineBuilder,
    ) -> Result<()> {
        self.check_pipeline(pipeline)?;
        self.check_pipeline(dependency)?;
        if pipeline == dependency {
            return internal_err!("Pipeline {} cannot depend on itself", pipeline.id);
        }
        let dependencies = &mut self.graph.pipelines[pipeline.id].dependencies;
        if !dependencies.contains(&dependency.id) {
            dependencies.push(dependency.id);
        }
        Ok(())
    }

    /// The pipelines of this meta pipeline created so far, in creation
    /// order.
    pub fn pipelines(&self) -> Vec<PipelineBuilder> {
        self.graph.meta_pipelines[self.meta]
            .pipelines
            .iter()
            .map(|&id| PipelineBuilder { id })
            .collect()
    }

    /// Make the current operator the source of `current`.
    pub fn build_source(&mut self, current: &mut PipelineBuilder) -> Result<()> {
        let operator = self.current_operator()?;
        self.set_source(current, operator)
    }

    /// Add the current operator to `current` as a streaming operator and
    /// continue with its single child.
    pub fn build_streaming_operator(
        &mut self,
        current: &mut PipelineBuilder,
    ) -> Result<()> {
        let operator = self.current_operator()?;
        let child = single_child(operator.as_ref())?;
        self.add_operator(current, operator)?;
        self.build(current, &child)
    }

    /// Make the current operator, a sink, the source of `current`, and build
    /// its single child into a new child meta pipeline that sinks into it.
    pub fn build_sink(&mut self, current: &mut PipelineBuilder) -> Result<()> {
        let operator = self.current_operator()?;
        let child = single_child(operator.as_ref())?;
        self.set_source(current, Arc::clone(&operator))?;
        let (mut child_meta, mut child_pipeline) =
            self.create_child_meta_pipeline(current, operator)?;
        child_meta.build(&mut child_pipeline, &child)
    }

    /// The default pipeline construction of
    /// [`PhysicalOperator::build_pipelines`]: sinks use
    /// [`Self::build_sink`], leaves [`Self::build_source`] and operators
    /// with one child [`Self::build_streaming_operator`].
    pub fn build_default(&mut self, current: &mut PipelineBuilder) -> Result<()> {
        let operator = self.current_operator()?;
        if operator.is_sink() {
            self.build_sink(current)
        } else if operator.children().is_empty() {
            self.build_source(current)
        } else if operator.children().len() == 1 {
            self.build_streaming_operator(current)
        } else {
            internal_err!(
                "{} has {} children and must implement build_pipelines",
                operator.name(),
                operator.children().len()
            )
        }
    }

    fn check_pipeline(&self, pipeline: &PipelineBuilder) -> Result<()> {
        match self.graph.pipelines.get(pipeline.id) {
            Some(state) if state.meta == self.meta => Ok(()),
            Some(state) => internal_err!(
                "Pipeline {} belongs to meta pipeline {}, not {}",
                pipeline.id,
                state.meta,
                self.meta
            ),
            None => internal_err!("Unknown pipeline {}", pipeline.id),
        }
    }
}

fn single_child(operator: &dyn PhysicalOperator) -> Result<Arc<dyn PhysicalOperator>> {
    match operator.children() {
        [child] => Ok(Arc::clone(child)),
        children => internal_err!(
            "{} must have exactly one child, it has {}",
            operator.name(),
            children.len()
        ),
    }
}
