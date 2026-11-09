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
//! The pipelines of a meta pipeline can also be made to wait for a meta
//! pipeline that is not their child (see
//! [`MetaPipelineBuilder::build_with_dependency`], DuckDB's
//! `MetaPipeline::AddRecursiveDependencies`): the pipeline breakers of the
//! probe side of a hash join only run once its build side is finalized, as
//! the join reads its probe side only then.
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

    /// Meta pipelines that must be finalized before this pipeline starts:
    /// the child meta pipelines of the pipeline breakers it reads from, and
    /// the meta pipelines it waits for without reading from them, e.g. the
    /// build side of a hash join whose probe side this pipeline is part of
    /// (see [`MetaPipelineBuilder::build_with_dependency`]).
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
    /// Building the pipeline started (it was passed to
    /// [`MetaPipelineBuilder::build`], or got a source or an operator).
    started: bool,
    /// For a union pipeline whose order matters: the pipeline it was created
    /// from. Once building it starts, it depends on that pipeline and on all
    /// pipelines created after it that started building, i.e. the pipelines
    /// of the inputs of the union before it.
    ordered_after: Option<PipelineId>,
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

    /// Whether meta pipeline `meta` is below meta pipeline `ancestor` (one of
    /// its descendants, not `ancestor` itself).
    fn is_below(&self, meta: MetaPipelineId, ancestor: MetaPipelineId) -> bool {
        let mut parent = self.meta_pipelines.get(meta).and_then(|meta| meta.parent);
        while let Some(id) = parent {
            if id == ancestor {
                return true;
            }
            parent = self.meta_pipelines[id].parent;
        }
        false
    }

    fn new_pipeline(&mut self, meta: MetaPipelineId) -> PipelineBuilder {
        let id = self.pipelines.len();
        self.pipelines.push(PipelineState {
            meta,
            source: None,
            operators: vec![],
            dependencies: vec![],
            meta_dependencies: vec![],
            started: false,
            ordered_after: None,
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
        self.start_pipeline(current.id);
        self.graph.visiting.push(Arc::clone(operator));
        let result = operator.build_pipelines(current, self);
        self.graph.visiting.pop();
        result
    }

    /// Continue building `current` with `operator`, like [`Self::build`], and
    /// make every pipeline that this continues or creates wait for the meta
    /// pipeline `dependency` to be finalized: `current`, the pipelines that
    /// it adds to this meta pipeline, and all pipelines of the meta pipelines
    /// that it creates for the pipeline breakers below `operator`,
    /// recursively (DuckDB's `MetaPipeline::AddRecursiveDependencies`).
    ///
    /// A hash join builds its probe side with the child meta pipeline of its
    /// build side as `dependency`: DataFusion reads the probe side of a join
    /// only once its build side is complete, so the pipeline breakers of the
    /// probe side (e.g. the build side of a join below) must not run before.
    /// If the join then cannot produce output, or is canceled, the pipelines
    /// of its probe side are skipped, and the meta pipelines whose output
    /// only these read are canceled before they start: their sinks are
    /// finalized without any input, as DataFusion does not execute them
    /// either (see [`crate::operator::SinkFinalize::NoOutputPossible`]).
    ///
    /// `dependency` must be a meta pipeline below this one whose pipelines
    /// are built, e.g. a child meta pipeline created with
    /// [`Self::create_child_meta_pipeline`] (whose builder was dropped).
    pub fn build_with_dependency(
        &mut self,
        current: &mut PipelineBuilder,
        operator: &Arc<dyn PhysicalOperator>,
        dependency: MetaPipelineId,
    ) -> Result<()> {
        self.check_pipeline(current)?;
        if !self.graph.is_below(dependency, self.meta) {
            return internal_err!(
                "The pipelines of meta pipeline {} cannot wait for meta pipeline {dependency}: it is not below it",
                self.meta
            );
        }
        // Pipelines get increasing ids: the pipelines created while building
        // `operator` are those of this meta pipeline that it creates and all
        // pipelines of the meta pipelines that it creates, which are below
        // this one (only builders of this meta pipeline and of the meta
        // pipelines below it can be used meanwhile).
        let created = self.graph.pipelines.len();
        self.build(current, operator)?;
        let pipelines =
            std::iter::once(current.id).chain(created..self.graph.pipelines.len());
        for pipeline in pipelines {
            let dependencies = &mut self.graph.pipelines[pipeline].meta_dependencies;
            if !dependencies.contains(&dependency) {
                dependencies.push(dependency);
            }
        }
        Ok(())
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
        self.start_pipeline(current.id);
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
        self.start_pipeline(current.id);
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

    /// Create a pipeline for the second input of a union-like operator with
    /// two inputs (see [`Self::build_union`], which also handles more inputs).
    ///
    /// The new pipeline belongs to this meta pipeline, has the operators
    /// that were added to `current` so far (the operators above the union)
    /// and inherits all dependencies of `current`. Create it before building
    /// the first input into `current`, then build the second input into it.
    ///
    /// If `order_matters`, the new pipeline only starts after all pipelines
    /// of the first input completed: `current`, and the pipelines created
    /// while building the first input (e.g. by a nested union).
    pub fn create_union_pipeline(
        &mut self,
        current: &PipelineBuilder,
        order_matters: bool,
    ) -> Result<PipelineBuilder> {
        self.check_pipeline(current)?;
        let union = self.graph.new_pipeline(self.meta);
        let current_state = &self.graph.pipelines[current.id];
        let operators = current_state.operators.clone();
        let dependencies = current_state.dependencies.clone();
        let meta_dependencies = current_state.meta_dependencies.clone();
        let state = &mut self.graph.pipelines[union.id];
        state.operators = operators;
        state.dependencies = dependencies;
        state.meta_dependencies = meta_dependencies;
        state.ordered_after = order_matters.then_some(current.id);
        Ok(union)
    }

    /// Build the pipelines of a union-like operator over `inputs`: the first
    /// input continues `current`, every other input gets a union pipeline
    /// (see [`Self::create_union_pipeline`]) that shares the operators above
    /// the union and the sink.
    ///
    /// If `order_matters`, the pipelines of every input only start after all
    /// pipelines of the inputs before it completed, so that the output of the
    /// inputs is not interleaved and comes in input order (if the sink and
    /// the other operators preserve it).
    pub fn build_union(
        &mut self,
        current: &mut PipelineBuilder,
        inputs: &[Arc<dyn PhysicalOperator>],
        order_matters: bool,
    ) -> Result<()> {
        self.check_pipeline(current)?;
        let Some((first, others)) = inputs.split_first() else {
            return internal_err!("A union needs at least one input");
        };
        // Create the pipelines of the other inputs while `current` has
        // exactly the operators above the union.
        let pipelines = others
            .iter()
            .map(|_| self.create_union_pipeline(current, false))
            .collect::<Result<Vec<_>>>()?;
        // The pipelines of the inputs built so far.
        let mut previous = vec![current.id];
        let created = self.graph.meta_pipelines[self.meta].pipelines.len();
        self.build(current, first)?;
        previous.extend(self.pipelines_created_since(created));
        for (mut pipeline, input) in pipelines.into_iter().zip(others) {
            if order_matters {
                for &dependency in &previous {
                    self.depend_on(pipeline.id, dependency);
                }
            }
            let created = self.graph.meta_pipelines[self.meta].pipelines.len();
            self.build(&mut pipeline, input)?;
            previous.push(pipeline.id);
            previous.extend(self.pipelines_created_since(created));
        }
        Ok(())
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
        state.started = true;
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
        self.depend_on(pipeline.id, dependency.id);
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

    /// Mark `pipeline` as started. If it is an ordered union pipeline, it
    /// now depends on the pipelines of the inputs of the union before it.
    fn start_pipeline(&mut self, pipeline: PipelineId) {
        let state = &mut self.graph.pipelines[pipeline];
        if state.started {
            return;
        }
        state.started = true;
        let Some(created_from) = state.ordered_after.take() else {
            return;
        };
        let siblings = &self.graph.meta_pipelines[self.meta].pipelines;
        let position = siblings
            .iter()
            .position(|&id| id == pipeline)
            .unwrap_or(siblings.len());
        let created_after = siblings[position..]
            .iter()
            .copied()
            .filter(|&id| id != pipeline && self.graph.pipelines[id].started)
            .collect::<Vec<_>>();
        self.depend_on(pipeline, created_from);
        for dependency in created_after {
            self.depend_on(pipeline, dependency);
        }
    }

    /// The pipelines of this meta pipeline at positions `start..`.
    fn pipelines_created_since(&self, start: usize) -> Vec<PipelineId> {
        self.graph.meta_pipelines[self.meta].pipelines[start..].to_vec()
    }

    /// Make `pipeline` depend on `dependency` (both of this meta pipeline).
    fn depend_on(&mut self, pipeline: PipelineId, dependency: PipelineId) {
        let dependencies = &mut self.graph.pipelines[pipeline].dependencies;
        if pipeline != dependency && !dependencies.contains(&dependency) {
            dependencies.push(dependency);
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

#[cfg(test)]
mod tests {
    use super::*;

    use arrow::datatypes::{Schema, SchemaRef};

    /// What a [`TestOperator`] does in [`PhysicalOperator::build_pipelines`].
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    enum Role {
        /// [`MetaPipelineBuilder::build_default`]: a source (without
        /// children), a pipeline breaker (a sink), or a streaming operator.
        Default,
        /// Like a hash join: an operator of the pipeline of its second child
        /// (its probe side), and the sink of its first child (its build side),
        /// whose meta pipeline its second child waits for.
        Join,
        /// Like a join, but its second child waits for the meta pipeline
        /// `MetaPipelineId` instead.
        JoinWaitingFor(MetaPipelineId),
    }

    #[derive(Debug)]
    struct TestOperator {
        name: &'static str,
        children: Vec<Arc<dyn PhysicalOperator>>,
        sink: bool,
        role: Role,
    }

    fn operator(
        name: &'static str,
        children: Vec<Arc<dyn PhysicalOperator>>,
        sink: bool,
        role: Role,
    ) -> Arc<dyn PhysicalOperator> {
        Arc::new(TestOperator {
            name,
            children,
            sink,
            role,
        })
    }

    fn source(name: &'static str) -> Arc<dyn PhysicalOperator> {
        operator(name, vec![], false, Role::Default)
    }

    fn breaker(
        name: &'static str,
        input: Arc<dyn PhysicalOperator>,
    ) -> Arc<dyn PhysicalOperator> {
        operator(name, vec![input], true, Role::Default)
    }

    fn join(
        name: &'static str,
        build: Arc<dyn PhysicalOperator>,
        probe: Arc<dyn PhysicalOperator>,
    ) -> Arc<dyn PhysicalOperator> {
        operator(name, vec![build, probe], true, Role::Join)
    }

    impl PhysicalOperator for TestOperator {
        fn name(&self) -> &str {
            self.name
        }

        fn schema(&self) -> SchemaRef {
            Arc::new(Schema::empty())
        }

        fn children(&self) -> &[Arc<dyn PhysicalOperator>] {
            &self.children
        }

        fn is_source(&self) -> bool {
            self.children.is_empty() || self.sink
        }

        fn is_sink(&self) -> bool {
            self.sink
        }

        fn build_pipelines(
            &self,
            current: &mut PipelineBuilder,
            meta: &mut MetaPipelineBuilder<'_>,
        ) -> Result<()> {
            let dependency = match self.role {
                Role::Default => return meta.build_default(current),
                Role::Join => None,
                Role::JoinWaitingFor(dependency) => Some(dependency),
            };
            let this = meta.current_operator()?;
            meta.add_operator(current, Arc::clone(&this))?;
            let (mut build, mut build_pipeline) =
                meta.create_child_meta_pipeline(current, this)?;
            let build_meta = build.id();
            build.build(&mut build_pipeline, &self.children[0])?;
            meta.build_with_dependency(
                current,
                &self.children[1],
                dependency.unwrap_or(build_meta),
            )
        }
    }

    fn graph(root: Arc<dyn PhysicalOperator>) -> Result<PipelineGraph> {
        PipelineGraph::try_new(breaker("Root", root))
    }

    #[test]
    fn pipeline_breakers_of_the_probe_side_wait_for_the_build_side() -> Result<()> {
        // `Outer` joins `a` with the probe side `Inner`, which joins the
        // aggregation of `b` with the aggregation of `c`. All pipelines of
        // the meta pipelines created for the probe side of `Outer` (those of
        // `Inner` and of the aggregations) wait for its build side (meta
        // pipeline 1); those created for the probe side of `Inner` (the
        // aggregation of `c`) also for the build side of `Inner` (meta
        // pipeline 2).
        let inner = join(
            "Inner",
            breaker("AggregateB", source("b")),
            breaker("AggregateC", source("c")),
        );
        let outer = join("Outer", source("a"), inner);
        let graph = graph(outer)?;
        assert_eq!(
            graph.to_string(),
            "MetaPipeline 0 (sink: Root) children: [1, 2, 4]\n  \
             Pipeline 0: AggregateC -> Inner -> Outer -> Root depends on meta pipelines [1, 2, 4]\n\
             MetaPipeline 1 (sink: Outer)\n  \
             Pipeline 1: a -> Outer\n\
             MetaPipeline 2 (sink: Inner) children: [3]\n  \
             Pipeline 2: AggregateB -> Inner depends on meta pipelines [3, 1]\n\
             MetaPipeline 3 (sink: AggregateB)\n  \
             Pipeline 3: b -> AggregateB depends on meta pipelines [1]\n\
             MetaPipeline 4 (sink: AggregateC)\n  \
             Pipeline 4: c -> AggregateC depends on meta pipelines [2, 1]\n"
        );
        // The meta pipelines created before the probe side do not wait.
        assert!(graph.pipeline(1).meta_dependencies().is_empty());
        assert_eq!(graph.meta_pipeline(4).parent(), Some(0));
        Ok(())
    }

    #[test]
    fn pipelines_wait_only_for_meta_pipelines_below_theirs() {
        // Waiting for the meta pipeline of the join itself (or one above it,
        // or one that does not exist) would wait for itself.
        for dependency in [0, 7] {
            let join = operator(
                "Join",
                vec![source("a"), breaker("Aggregate", source("b"))],
                true,
                Role::JoinWaitingFor(dependency),
            );
            let err = graph(join).unwrap_err();
            assert!(
                err.to_string().contains(&format!(
                    "cannot wait for meta pipeline {dependency}: it is not below it"
                )),
                "{err}"
            );
        }
    }
}
