// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: See the NOTICE file distributed with this work for additional information regarding copyright ownership
// SPDX-FileContributor: Licensed to the Apache Software Foundation (ASF) under one or more contributor license agreements

//! Structural Common Subplan Elimination (CSE) analysis for [`super::MorselQueryPlanner`].
//!
//! Detects duplicate pure (non-volatile) subplans across the logical query tree
//! (including uncorrelated scalar subqueries), covering both:
//! 1. Exact structural matches ignoring `SubqueryAlias` and output aliases
//!    (e.g. TPC-H Q15's inlined `revenue0` aggregate), and
//! 2. Projection-subsumed `Join` subtrees whose scans/filters/joins are
//!    identical and differ only in column pruning (e.g. TPC-H Q11's
//!    `partsupp ⋈ supplier ⋈ nation`).

use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use arrow::datatypes::DataType;
use datafusion_common::tree_node::{Transformed, TreeNode, TreeNodeRecursion};
use datafusion_common::{
    Column, DFSchema, JoinConstraint, JoinType, NullEquality, ScalarValue, TableReference,
};
use datafusion_expr::expr::Sort as SortExpr;
use datafusion_expr::logical_plan::{
    Distinct, FetchType, MaterializedCteId, MaterializedCtePlan, SkipType,
};
use datafusion_expr::{Expr, LogicalPlan, Operator, Subquery};
use indexmap::IndexMap;

use crate::datasource::source_as_provider;

/// Interned identifier for an exact canonical pure logical subplan.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ExactId(usize);

/// Interned identifier for a projection-stripped join relation tree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct JoinRelId(usize);

/// Provenance of a column inside a projection-stripped join subtree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct ColOrigin {
    leaf_idx: usize,
    src_col: usize,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ExactNode {
    TableScan {
        provider_ptr: usize,
        table_name: TableReference,
        projection: Vec<usize>,
        filters: Vec<Expr>,
        fetch: Option<usize>,
        skip: Option<usize>,
        output_types: Vec<(DataType, bool)>,
    },
    EmptyRelation {
        produce_one_row: bool,
        output_types: Vec<(DataType, bool)>,
    },
    Values {
        rows: Vec<Vec<Expr>>,
        output_types: Vec<(DataType, bool)>,
    },
    Filter {
        input: ExactId,
        predicate: Expr,
    },
    Projection {
        input: ExactId,
        exprs: Vec<Expr>,
        output_types: Vec<(DataType, bool)>,
    },
    Aggregate {
        input: ExactId,
        group_expr: Vec<Expr>,
        aggr_expr: Vec<Expr>,
        output_types: Vec<(DataType, bool)>,
    },
    DistinctAll {
        input: ExactId,
    },
    Join {
        left: ExactId,
        right: ExactId,
        join_type: JoinType,
        join_constraint: JoinConstraint,
        null_equality: NullEquality,
        null_aware: bool,
        on: Vec<(Expr, Expr)>,
        filter: Option<Expr>,
    },
    Sort {
        input: ExactId,
        expr: Vec<SortExpr>,
        fetch: Option<usize>,
    },
    Limit {
        input: ExactId,
        skip: usize,
        fetch: Option<usize>,
    },
    MaterializedCteScan {
        scope_id: usize,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum JoinRelNode {
    TableScan {
        leaf_idx: usize,
        provider_ptr: usize,
        table_name: TableReference,
        filters: Vec<Expr>,
        fetch: Option<usize>,
        skip: Option<usize>,
    },
    OpaqueLeaf {
        leaf_idx: usize,
        exact_id: ExactId,
    },
    Filter {
        input: JoinRelId,
        predicate: Expr,
    },
    Join {
        left: JoinRelId,
        right: JoinRelId,
        join_type: JoinType,
        join_constraint: JoinConstraint,
        null_equality: NullEquality,
        null_aware: bool,
        on: Vec<(Expr, Expr)>,
        filter: Option<Expr>,
    },
}

#[derive(Debug, Clone)]
struct JoinRelInfo {
    rel_id: JoinRelId,
    provenance: Vec<ColOrigin>,
    output_types: Vec<(DataType, bool)>,
    has_internal_filter: bool,
}

#[derive(Debug, Clone)]
struct NodeAnalysis {
    exact_id: ExactId,
    has_blocking: bool,
    has_join: bool,
    has_filter: bool,
    join_rel: Option<JoinRelInfo>,
}

type JoinRelSuperset<'plan> = (&'plan LogicalPlan, HashMap<usize, Option<Vec<usize>>>);

/// Assignment for one logical node participating in a deduplicated CSE stage.
#[derive(Debug, Clone)]
pub(super) struct CseOccurrence<'plan> {
    pub(super) group_id: usize,
    pub(super) representative: &'plan LogicalPlan,
    /// When `Some`, selects columns from `representative`'s output schema to
    /// produce this occurrence's narrower output (projection-subsumed Join CSE).
    pub(super) col_mapping: Option<Vec<usize>>,
}

/// Read-only CSE lookup table built once per `MorselQueryPlanner::try_lower`.
#[derive(Debug, Default)]
pub(super) struct CsePlan<'plan> {
    /// Keyed by `node as *const LogicalPlan as usize`.
    occurrences: HashMap<usize, CseOccurrence<'plan>>,
}

impl<'plan> CsePlan<'plan> {
    pub(super) fn analyze(root: &'plan LogicalPlan) -> Self {
        let mut analyzer = Analyzer::default();
        analyzer.analyze_plan(root);
        let occurrences = analyzer.select_cse_groups(root);
        Self { occurrences }
    }

    pub(super) fn lookup(&self, node: &LogicalPlan) -> Option<&CseOccurrence<'plan>> {
        self.occurrences.get(&plan_ptr(node))
    }
}

#[derive(Default)]
#[allow(clippy::allow_attributes, clippy::mutable_key_type)]
struct Analyzer<'plan> {
    exact_intern: HashMap<ExactNode, ExactId>,
    join_rel_intern: HashMap<JoinRelNode, JoinRelId>,
    nodes: HashMap<usize, NodeAnalysis>,
    /// Canonical `ExactId` per unique uncorrelated scalar subquery (or `None` if impure).
    subquery_canon: HashMap<Subquery, Option<ExactId>>,
    /// First-seen borrowed AST root for each unique uncorrelated scalar subquery.
    subquery_roots: HashMap<Subquery, &'plan LogicalPlan>,
    cte_scopes: HashMap<MaterializedCteId, usize>,
    next_cte_scope: usize,
}

fn schema_types(schema: &DFSchema) -> Vec<(DataType, bool)> {
    schema
        .as_arrow()
        .fields()
        .iter()
        .map(|f| (f.data_type().clone(), f.is_nullable()))
        .collect()
}

fn plan_ptr(plan: &LogicalPlan) -> usize {
    std::ptr::from_ref(plan).addr()
}

fn strip_alias(mut expr: &Expr) -> &Expr {
    while let Expr::Alias(alias) = expr {
        expr = &alias.expr;
    }
    expr
}

fn is_trivially_true_filter(expr: &Expr) -> bool {
    match strip_alias(expr) {
        Expr::Literal(ScalarValue::Boolean(Some(true)), _) => true,
        Expr::BinaryExpr(bin) if bin.op == Operator::And => {
            is_trivially_true_filter(&bin.left) && is_trivially_true_filter(&bin.right)
        }
        _ => false,
    }
}

fn is_column_only_projection(proj: &datafusion_expr::logical_plan::Projection) -> bool {
    proj.expr
        .iter()
        .all(|e| matches!(strip_alias(e), Expr::Column(_)))
}

fn unwrap_aliases_and_col_projs(mut plan: &LogicalPlan) -> &LogicalPlan {
    loop {
        match plan {
            LogicalPlan::SubqueryAlias(alias) => plan = &alias.input,
            LogicalPlan::Projection(proj) if is_column_only_projection(proj) => {
                plan = &proj.input;
            }
            _ => return plan,
        }
    }
}

/// Borrow every top-level expression directly attached to `plan` with lifetime `'plan`.
fn for_each_plan_expr<'plan>(plan: &'plan LogicalPlan, mut f: impl FnMut(&'plan Expr)) {
    match plan {
        LogicalPlan::Filter(filter) => f(&filter.predicate),
        LogicalPlan::Projection(proj) => {
            for e in &proj.expr {
                f(e);
            }
        }
        LogicalPlan::Aggregate(agg) => {
            for e in &agg.group_expr {
                f(e);
            }
            for e in &agg.aggr_expr {
                f(e);
            }
        }
        LogicalPlan::Join(join) => {
            for (l, r) in &join.on {
                f(l);
                f(r);
            }
            if let Some(filter) = &join.filter {
                f(filter);
            }
        }
        LogicalPlan::Sort(sort) => {
            for s in &sort.expr {
                f(&s.expr);
            }
        }
        LogicalPlan::TableScan(scan) => {
            for e in &scan.filters {
                f(e);
            }
        }
        LogicalPlan::Values(values) => {
            for row in &values.values {
                for e in row {
                    f(e);
                }
            }
        }
        _ => {}
    }
}

/// Borrow every uncorrelated `Expr::ScalarSubquery` directly inside `plan`'s expressions.
fn for_each_scalar_subquery<'plan>(
    plan: &'plan LogicalPlan,
    mut f: impl FnMut(&'plan Subquery),
) {
    for_each_plan_expr(plan, |expr| {
        let _ = expr.apply(|e| {
            if let Expr::ScalarSubquery(sq) = e
                && sq.outer_ref_columns.is_empty()
            {
                f(sq);
            }
            Ok(TreeNodeRecursion::Continue)
        });
    });
}

/// Returns true if `plan` contains any non-trivial `Filter`, `TableScan` filter,
/// or `Join` filter — used to detect selective outer join siblings.
fn subtree_has_filter(plan: &LogicalPlan) -> bool {
    let mut found = false;
    let _ = plan.apply(|node| {
        match node {
            LogicalPlan::Filter(filter)
                if !is_trivially_true_filter(&filter.predicate) =>
            {
                found = true;
                return Ok(TreeNodeRecursion::Stop);
            }
            LogicalPlan::TableScan(scan)
                if scan.filters.iter().any(|f| !is_trivially_true_filter(f)) =>
            {
                found = true;
                return Ok(TreeNodeRecursion::Stop);
            }
            LogicalPlan::Join(join)
                if join
                    .filter
                    .as_ref()
                    .is_some_and(|f| !is_trivially_true_filter(f)) =>
            {
                found = true;
                return Ok(TreeNodeRecursion::Stop);
            }
            LogicalPlan::MaterializedCte(MaterializedCtePlan::Cte(_)) => {
                return Ok(TreeNodeRecursion::Jump);
            }
            _ => {}
        }
        Ok(TreeNodeRecursion::Continue)
    });
    found
}

impl<'plan> Analyzer<'plan> {
    fn intern_exact(&mut self, node: ExactNode) -> ExactId {
        if let Some(&id) = self.exact_intern.get(&node) {
            return id;
        }
        let id = ExactId(self.exact_intern.len());
        self.exact_intern.insert(node, id);
        id
    }

    fn intern_join_rel(&mut self, node: JoinRelNode) -> JoinRelId {
        if let Some(&id) = self.join_rel_intern.get(&node) {
            return id;
        }
        let id = JoinRelId(self.join_rel_intern.len());
        self.join_rel_intern.insert(node, id);
        id
    }

    fn ensure_scalar_subquery_analyzed(&mut self, sq: &'plan Subquery) {
        if self.subquery_canon.contains_key(sq) {
            return;
        }
        let sub_root: &'plan LogicalPlan = sq.subquery.as_ref();
        self.subquery_roots.entry(sq.clone()).or_insert(sub_root);
        let res = self.analyze_plan(sub_root).map(|a| a.exact_id);
        self.subquery_canon.insert(sq.clone(), res);
    }

    /// Canonicalize an expression by checking purity (no volatile functions, no
    /// correlated outer refs), stripping `Expr::Alias`, replacing uncorrelated
    /// scalar subqueries with canonical placeholders, and rewriting every
    /// `Expr::Column` via `resolve_col`.
    ///
    /// Callers must call `ensure_scalar_subquery_analyzed` on all subqueries in
    /// `expr` beforehand so `self.subquery_canon` is populated.
    fn canonicalize_expr(
        &self,
        expr: &Expr,
        resolve_col: &dyn Fn(&Column) -> Option<String>,
    ) -> Option<Expr> {
        let mut valid = true;
        let transformed = expr
            .clone()
            .transform_up(|e| {
                if !valid {
                    return Ok(Transformed::no(e));
                }
                if e.is_volatile_node() {
                    valid = false;
                    return Ok(Transformed::no(e));
                }
                match e {
                    Expr::OuterReferenceColumn(_, _)
                    | Expr::Exists(_)
                    | Expr::InSubquery(_)
                    | Expr::Placeholder(_) => {
                        valid = false;
                        Ok(Transformed::no(e))
                    }
                    Expr::Alias(alias) => Ok(Transformed::yes(*alias.expr)),
                    Expr::ScalarSubquery(sq) => {
                        if let Some(Some(sq_id)) = self.subquery_canon.get(&sq) {
                            Ok(Transformed::yes(Expr::Column(Column::new_unqualified(
                                format!("__sq_{}", sq_id.0),
                            ))))
                        } else {
                            valid = false;
                            Ok(Transformed::no(Expr::ScalarSubquery(sq)))
                        }
                    }
                    Expr::Column(col) => {
                        if let Some(canon) = resolve_col(&col) {
                            Ok(Transformed::yes(Expr::Column(Column::new_unqualified(
                                canon,
                            ))))
                        } else {
                            valid = false;
                            Ok(Transformed::no(Expr::Column(col)))
                        }
                    }
                    other => Ok(Transformed::no(other)),
                }
            })
            .ok()?;
        valid.then_some(transformed.data)
    }

    fn canonicalize_expr_with_schema(
        &self,
        expr: &Expr,
        schema: &DFSchema,
    ) -> Option<Expr> {
        self.canonicalize_expr(expr, &|col| {
            schema
                .index_of_column(col)
                .ok()
                .map(|idx| format!("@{idx}"))
        })
    }

    fn analyze_plan(&mut self, plan: &'plan LogicalPlan) -> Option<NodeAnalysis> {
        let ptr = plan_ptr(plan);
        if let Some(existing) = self.nodes.get(&ptr) {
            return Some(existing.clone());
        }

        // Always analyze reachable uncorrelated scalar subqueries on this node
        // first, even if this node itself turns out to be impure (e.g. volatile).
        let mut sqs: Vec<&'plan Subquery> = Vec::new();
        for_each_scalar_subquery(plan, |sq| sqs.push(sq));
        for sq in sqs {
            self.ensure_scalar_subquery_analyzed(sq);
        }

        let analysis = match plan {
            LogicalPlan::SubqueryAlias(alias) => {
                // Transparent: analyze input, do NOT register alias itself in `self.nodes`.
                return self.analyze_plan(&alias.input);
            }
            LogicalPlan::MaterializedCte(MaterializedCtePlan::Cte(cte)) => {
                let scans = cte.scans().ok()?;
                if scans.is_empty() {
                    return self.analyze_plan(cte.continuation());
                }
                let _ = self.analyze_plan(cte.body());
                let scope_id = self.next_cte_scope;
                self.next_cte_scope += 1;
                let prev = self.cte_scopes.insert(cte.id, scope_id);
                let cont = self.analyze_plan(cte.continuation());
                match prev {
                    Some(old) => {
                        self.cte_scopes.insert(cte.id, old);
                    }
                    None => {
                        self.cte_scopes.remove(&cte.id);
                    }
                }
                return cont;
            }
            LogicalPlan::MaterializedCte(MaterializedCtePlan::Scan(scan)) => {
                let scope_id = *self.cte_scopes.get(&scan.id)?;
                let exact_id =
                    self.intern_exact(ExactNode::MaterializedCteScan { scope_id });
                NodeAnalysis {
                    exact_id,
                    has_blocking: false,
                    has_join: false,
                    has_filter: false,
                    join_rel: None,
                }
            }
            LogicalPlan::TableScan(scan) => {
                if !scan.statistics_requests.is_empty() {
                    return None;
                }
                let provider = source_as_provider(&scan.source).ok()?;
                let provider_ptr = Arc::as_ptr(&provider).cast::<()>().addr();
                let full_schema = DFSchema::try_from_qualified_schema(
                    scan.table_name.clone(),
                    &scan.source.schema(),
                )
                .ok()?;
                let mut filters = Vec::with_capacity(scan.filters.len());
                for f in &scan.filters {
                    filters.push(self.canonicalize_expr_with_schema(f, &full_schema)?);
                }
                let projection = match &scan.projection {
                    Some(p) => p.clone(),
                    None => (0..scan.source.schema().fields().len()).collect(),
                };
                let has_filter = filters.iter().any(|f| !is_trivially_true_filter(f));
                let exact_id = self.intern_exact(ExactNode::TableScan {
                    provider_ptr,
                    table_name: scan.table_name.clone(),
                    projection,
                    filters,
                    fetch: scan.fetch,
                    skip: scan.skip,
                    output_types: schema_types(&scan.projected_schema),
                });
                NodeAnalysis {
                    exact_id,
                    has_blocking: false,
                    has_join: false,
                    has_filter,
                    join_rel: None,
                }
            }
            LogicalPlan::EmptyRelation(empty) => {
                let exact_id = self.intern_exact(ExactNode::EmptyRelation {
                    produce_one_row: empty.produce_one_row,
                    output_types: schema_types(&empty.schema),
                });
                NodeAnalysis {
                    exact_id,
                    has_blocking: false,
                    has_join: false,
                    has_filter: false,
                    join_rel: None,
                }
            }
            LogicalPlan::Values(values) => {
                let mut rows = Vec::with_capacity(values.values.len());
                for row in &values.values {
                    let mut canon_row = Vec::with_capacity(row.len());
                    for expr in row {
                        canon_row.push(
                            self.canonicalize_expr_with_schema(expr, &values.schema)?,
                        );
                    }
                    rows.push(canon_row);
                }
                let exact_id = self.intern_exact(ExactNode::Values {
                    rows,
                    output_types: schema_types(&values.schema),
                });
                NodeAnalysis {
                    exact_id,
                    has_blocking: false,
                    has_join: false,
                    has_filter: false,
                    join_rel: None,
                }
            }
            LogicalPlan::Filter(filter) => {
                let child = self.analyze_plan(&filter.input);
                let pred = self.canonicalize_expr_with_schema(
                    &filter.predicate,
                    filter.input.schema(),
                );
                let (pred, child) = (pred?, child?);
                let has_filter = child.has_filter || !is_trivially_true_filter(&pred);
                let exact_id = self.intern_exact(ExactNode::Filter {
                    input: child.exact_id,
                    predicate: pred,
                });
                NodeAnalysis {
                    exact_id,
                    has_blocking: child.has_blocking,
                    has_join: child.has_join,
                    has_filter,
                    join_rel: None,
                }
            }
            LogicalPlan::Projection(proj) => {
                let child = self.analyze_plan(&proj.input)?;
                let mut exprs = Vec::with_capacity(proj.expr.len());
                for e in &proj.expr {
                    exprs.push(
                        self.canonicalize_expr_with_schema(e, proj.input.schema())?,
                    );
                }
                let output_types = schema_types(&proj.schema);
                let exact_id = self.intern_exact(ExactNode::Projection {
                    input: child.exact_id,
                    exprs,
                    output_types: output_types.clone(),
                });
                let join_rel = if is_column_only_projection(proj)
                    && matches!(unwrap_aliases_and_col_projs(plan), LogicalPlan::Join(_))
                {
                    let mut next_leaf = 0;
                    self.analyze_join_rel(plan, &mut next_leaf).map(
                        |(rel_id, provenance, has_internal_filter)| JoinRelInfo {
                            rel_id,
                            provenance,
                            output_types,
                            has_internal_filter,
                        },
                    )
                } else {
                    None
                };
                NodeAnalysis {
                    exact_id,
                    has_blocking: child.has_blocking,
                    has_join: child.has_join,
                    has_filter: child.has_filter,
                    join_rel,
                }
            }
            LogicalPlan::Aggregate(agg) => {
                let child = self.analyze_plan(&agg.input)?;
                let mut group_expr = Vec::with_capacity(agg.group_expr.len());
                let mut aggr_expr = Vec::with_capacity(agg.aggr_expr.len());
                for e in &agg.group_expr {
                    group_expr
                        .push(self.canonicalize_expr_with_schema(e, agg.input.schema())?);
                }
                for e in &agg.aggr_expr {
                    aggr_expr
                        .push(self.canonicalize_expr_with_schema(e, agg.input.schema())?);
                }
                let exact_id = self.intern_exact(ExactNode::Aggregate {
                    input: child.exact_id,
                    group_expr,
                    aggr_expr,
                    output_types: schema_types(&agg.schema),
                });
                NodeAnalysis {
                    exact_id,
                    has_blocking: true,
                    has_join: child.has_join,
                    has_filter: child.has_filter,
                    join_rel: None,
                }
            }
            LogicalPlan::Distinct(Distinct::All(input)) => {
                let child = self.analyze_plan(input)?;
                let exact_id = self.intern_exact(ExactNode::DistinctAll {
                    input: child.exact_id,
                });
                NodeAnalysis {
                    exact_id,
                    has_blocking: true,
                    has_join: child.has_join,
                    has_filter: child.has_filter,
                    join_rel: None,
                }
            }
            LogicalPlan::Join(join) => {
                let left = self.analyze_plan(&join.left);
                let right = self.analyze_plan(&join.right);
                let (left, right) = (left?, right?);
                let mut on = Vec::with_capacity(join.on.len());
                for (l, r) in &join.on {
                    let cl = self.canonicalize_expr_with_schema(l, join.left.schema())?;
                    let cr =
                        self.canonicalize_expr_with_schema(r, join.right.schema())?;
                    on.push((cl, cr));
                }
                let filter = match &join.filter {
                    Some(f) => {
                        let left_schema = join.left.schema();
                        let right_schema = join.right.schema();
                        Some(self.canonicalize_expr(f, &|col| match (
                            left_schema.index_of_column(col).ok(),
                            right_schema.index_of_column(col).ok(),
                        ) {
                            (Some(idx), None) => Some(format!("L@{idx}")),
                            (None, Some(idx)) => Some(format!("R@{idx}")),
                            _ => None,
                        })?)
                    }
                    None => None,
                };
                let has_filter = left.has_filter
                    || right.has_filter
                    || filter
                        .as_ref()
                        .is_some_and(|f| !is_trivially_true_filter(f));
                let exact_id = self.intern_exact(ExactNode::Join {
                    left: left.exact_id,
                    right: right.exact_id,
                    join_type: join.join_type,
                    join_constraint: join.join_constraint,
                    null_equality: join.null_equality,
                    null_aware: join.null_aware,
                    on,
                    filter,
                });
                let mut next_leaf = 0;
                let join_rel = self.analyze_join_rel(plan, &mut next_leaf).map(
                    |(rel_id, provenance, has_internal_filter)| JoinRelInfo {
                        rel_id,
                        provenance,
                        output_types: schema_types(&join.schema),
                        has_internal_filter,
                    },
                );
                NodeAnalysis {
                    exact_id,
                    has_blocking: left.has_blocking || right.has_blocking,
                    has_join: true,
                    has_filter,
                    join_rel,
                }
            }
            LogicalPlan::Sort(sort) => {
                let child = self.analyze_plan(&sort.input)?;
                let mut exprs = Vec::with_capacity(sort.expr.len());
                for s in &sort.expr {
                    let ce =
                        self.canonicalize_expr_with_schema(&s.expr, sort.input.schema())?;
                    exprs.push(SortExpr {
                        expr: ce,
                        asc: s.asc,
                        nulls_first: s.nulls_first,
                    });
                }
                let exact_id = self.intern_exact(ExactNode::Sort {
                    input: child.exact_id,
                    expr: exprs,
                    fetch: sort.fetch,
                });
                NodeAnalysis {
                    exact_id,
                    has_blocking: true,
                    has_join: child.has_join,
                    has_filter: child.has_filter,
                    join_rel: None,
                }
            }
            LogicalPlan::Limit(limit) => {
                let child = self.analyze_plan(&limit.input)?;
                let SkipType::Literal(skip) = limit.get_skip_type().ok()? else {
                    return None;
                };
                let FetchType::Literal(fetch) = limit.get_fetch_type().ok()? else {
                    return None;
                };
                let exact_id = self.intern_exact(ExactNode::Limit {
                    input: child.exact_id,
                    skip,
                    fetch,
                });
                NodeAnalysis {
                    exact_id,
                    has_blocking: true,
                    has_join: child.has_join,
                    has_filter: child.has_filter,
                    join_rel: None,
                }
            }
            _ => return None,
        };

        self.nodes.insert(ptr, analysis.clone());
        Some(analysis)
    }

    /// Analyze a join subtree stripping `SubqueryAlias`, column-only
    /// `Projection`, and `TableScan.projection`, returning
    /// `(JoinRelId, output_provenance, has_internal_filter)`.
    fn analyze_join_rel(
        &mut self,
        plan: &'plan LogicalPlan,
        next_leaf: &mut usize,
    ) -> Option<(JoinRelId, Vec<ColOrigin>, bool)> {
        match plan {
            LogicalPlan::SubqueryAlias(alias) => {
                self.analyze_join_rel(&alias.input, next_leaf)
            }
            LogicalPlan::Projection(proj) if is_column_only_projection(proj) => {
                let (child_rel, child_prov, has_filter) =
                    self.analyze_join_rel(&proj.input, next_leaf)?;
                let in_schema = proj.input.schema();
                let mut out_prov = Vec::with_capacity(proj.expr.len());
                for e in &proj.expr {
                    let Expr::Column(col) = strip_alias(e) else {
                        return None;
                    };
                    let idx = in_schema.index_of_column(col).ok()?;
                    out_prov.push(*child_prov.get(idx)?);
                }
                Some((child_rel, out_prov, has_filter))
            }
            LogicalPlan::TableScan(scan) => {
                if !scan.statistics_requests.is_empty() {
                    return None;
                }
                let provider = source_as_provider(&scan.source).ok()?;
                let provider_ptr = Arc::as_ptr(&provider).cast::<()>().addr();
                let leaf_idx = *next_leaf;
                *next_leaf += 1;
                let full_schema = DFSchema::try_from_qualified_schema(
                    scan.table_name.clone(),
                    &scan.source.schema(),
                )
                .ok()?;
                let mut filters = Vec::with_capacity(scan.filters.len());
                for f in &scan.filters {
                    let cf = self.canonicalize_expr(f, &|col| {
                        full_schema
                            .index_of_column(col)
                            .ok()
                            .map(|src_col| format!("@p_{leaf_idx}_{src_col}"))
                    })?;
                    filters.push(cf);
                }
                let has_filter = filters.iter().any(|f| !is_trivially_true_filter(f));
                let rel_id = self.intern_join_rel(JoinRelNode::TableScan {
                    leaf_idx,
                    provider_ptr,
                    table_name: scan.table_name.clone(),
                    filters,
                    fetch: scan.fetch,
                    skip: scan.skip,
                });
                let num_cols = scan.source.schema().fields().len();
                let out_prov: Vec<ColOrigin> = match &scan.projection {
                    Some(proj) => proj
                        .iter()
                        .map(|&src_col| ColOrigin { leaf_idx, src_col })
                        .collect(),
                    None => (0..num_cols)
                        .map(|src_col| ColOrigin { leaf_idx, src_col })
                        .collect(),
                };
                Some((rel_id, out_prov, has_filter))
            }
            LogicalPlan::Filter(filter) => {
                let (child_rel, child_prov, child_filt) =
                    self.analyze_join_rel(&filter.input, next_leaf)?;
                let in_schema = filter.input.schema();
                let predicate = self.canonicalize_expr(&filter.predicate, &|col| {
                    let idx = in_schema.index_of_column(col).ok()?;
                    let orig = child_prov.get(idx)?;
                    Some(format!("@p_{}_{}", orig.leaf_idx, orig.src_col))
                })?;
                let has_filter = child_filt || !is_trivially_true_filter(&predicate);
                let rel_id = self.intern_join_rel(JoinRelNode::Filter {
                    input: child_rel,
                    predicate,
                });
                Some((rel_id, child_prov, has_filter))
            }
            LogicalPlan::Join(join) => {
                let (left_rel, left_prov, left_filt) =
                    self.analyze_join_rel(&join.left, next_leaf)?;
                let (right_rel, right_prov, right_filt) =
                    self.analyze_join_rel(&join.right, next_leaf)?;
                let left_schema = join.left.schema();
                let right_schema = join.right.schema();
                let mut on = Vec::with_capacity(join.on.len());
                for (l, r) in &join.on {
                    let cl = self.canonicalize_expr(l, &|col| {
                        let idx = left_schema.index_of_column(col).ok()?;
                        let orig = left_prov.get(idx)?;
                        Some(format!("@p_{}_{}", orig.leaf_idx, orig.src_col))
                    })?;
                    let cr = self.canonicalize_expr(r, &|col| {
                        let idx = right_schema.index_of_column(col).ok()?;
                        let orig = right_prov.get(idx)?;
                        Some(format!("@p_{}_{}", orig.leaf_idx, orig.src_col))
                    })?;
                    on.push((cl, cr));
                }
                let filter = match &join.filter {
                    Some(f) => Some(self.canonicalize_expr(f, &|col| match (
                        left_schema.index_of_column(col).ok(),
                        right_schema.index_of_column(col).ok(),
                    ) {
                        (Some(idx), None) => {
                            let orig = left_prov.get(idx)?;
                            Some(format!("@p_{}_{}", orig.leaf_idx, orig.src_col))
                        }
                        (None, Some(idx)) => {
                            let orig = right_prov.get(idx)?;
                            Some(format!("@p_{}_{}", orig.leaf_idx, orig.src_col))
                        }
                        _ => None,
                    })?),
                    None => None,
                };
                let out_prov = match join.join_type {
                    JoinType::Inner
                    | JoinType::Left
                    | JoinType::Right
                    | JoinType::Full => {
                        let mut v =
                            Vec::with_capacity(left_prov.len() + right_prov.len());
                        v.extend_from_slice(&left_prov);
                        v.extend_from_slice(&right_prov);
                        v
                    }
                    JoinType::LeftSemi | JoinType::LeftAnti => left_prov,
                    JoinType::RightSemi | JoinType::RightAnti => right_prov,
                    JoinType::LeftMark | JoinType::RightMark => return None,
                };
                let has_filter = left_filt
                    || right_filt
                    || filter
                        .as_ref()
                        .is_some_and(|f| !is_trivially_true_filter(f));
                let rel_id = self.intern_join_rel(JoinRelNode::Join {
                    left: left_rel,
                    right: right_rel,
                    join_type: join.join_type,
                    join_constraint: join.join_constraint,
                    null_equality: join.null_equality,
                    null_aware: join.null_aware,
                    on,
                    filter,
                });
                Some((rel_id, out_prov, has_filter))
            }
            other => {
                let info = self.nodes.get(&plan_ptr(other))?;
                let leaf_idx = *next_leaf;
                *next_leaf += 1;
                let exact_id = info.exact_id;
                let has_filter = info.has_filter;
                let rel_id =
                    self.intern_join_rel(JoinRelNode::OpaqueLeaf { leaf_idx, exact_id });
                let num_cols = other.schema().fields().len();
                let out_prov = (0..num_cols)
                    .map(|src_col| ColOrigin { leaf_idx, src_col })
                    .collect();
                Some((rel_id, out_prov, has_filter))
            }
        }
    }

    /// Iteratively select maximal CSE groups top-down (handling nested multi-level
    /// CSE cleanly by re-counting through each already-selected group's
    /// representative exactly once per round).
    fn select_cse_groups(
        &self,
        root: &'plan LogicalPlan,
    ) -> HashMap<usize, CseOccurrence<'plan>> {
        let mut selected: HashMap<usize, CseOccurrence<'plan>> = HashMap::new();
        let mut next_group_id = 0usize;

        loop {
            let mut round = RoundCounter::default();
            round.walk(root, self, &selected, false, false);

            let mut exact_eligible: HashSet<ExactId> = HashSet::new();
            for (&exact_id, occs) in &round.exact_occs {
                if occs.len() < 2 {
                    continue;
                }
                let first_info = &self.nodes[&plan_ptr(occs[0].plan)];
                if !(first_info.has_blocking || first_info.has_join) {
                    continue;
                }
                if !first_info.has_blocking {
                    let any_under_filtered_join =
                        occs.iter().any(|o| o.under_filtered_outer_join);
                    if !first_info.has_filter && any_under_filtered_join {
                        continue;
                    }
                }
                exact_eligible.insert(exact_id);
            }

            let mut join_rel_eligible: HashSet<JoinRelId> = HashSet::new();
            for (&rel_id, occs) in &round.join_rel_occs {
                if occs.len() < 2 {
                    continue;
                }
                let first_exact = self.nodes[&plan_ptr(occs[0].plan)].exact_id;
                if occs
                    .iter()
                    .all(|o| self.nodes[&plan_ptr(o.plan)].exact_id == first_exact)
                    && exact_eligible.contains(&first_exact)
                {
                    continue;
                }
                let first_rel = self.nodes[&plan_ptr(occs[0].plan)]
                    .join_rel
                    .as_ref()
                    .unwrap();
                let any_under_filtered_join =
                    occs.iter().any(|o| o.under_filtered_outer_join);
                if !first_rel.has_internal_filter && any_under_filtered_join {
                    continue;
                }
                join_rel_eligible.insert(rel_id);
            }

            if exact_eligible.is_empty() && join_rel_eligible.is_empty() {
                break;
            }

            // Collect ONLY unshadowed maximal occurrences (stopping at the first
            // eligible ancestor) so partially-overlapping nested subplans wait
            // until the next round when their outer parent is resolved.
            let mut maximal = MaximalCollector {
                analyzer: self,
                already_selected: &selected,
                exact_eligible: &exact_eligible,
                join_rel_eligible: &join_rel_eligible,
                exact_maximal: IndexMap::new(),
                join_rel_maximal: IndexMap::new(),
                visited_groups: HashSet::new(),
                visited_subqueries: HashSet::new(),
            };
            maximal.walk(root, false);
            let exact_maximal = maximal.exact_maximal;
            let join_rel_maximal = maximal.join_rel_maximal;

            let mut added_any = false;
            for occs in exact_maximal.values() {
                if occs.len() < 2 {
                    continue;
                }
                let group_id = next_group_id;
                next_group_id += 1;
                let rep = occs[0];
                for &p in occs {
                    selected.insert(
                        plan_ptr(p),
                        CseOccurrence {
                            group_id,
                            representative: rep,
                            col_mapping: None,
                        },
                    );
                }
                added_any = true;
            }

            for occs in join_rel_maximal.values() {
                if occs.len() < 2 {
                    continue;
                }
                let Some((rep, mappings)) = self.find_join_rel_superset(occs) else {
                    continue;
                };
                let group_id = next_group_id;
                next_group_id += 1;
                for &p in occs {
                    let ptr = plan_ptr(p);
                    selected.insert(
                        ptr,
                        CseOccurrence {
                            group_id,
                            representative: rep,
                            col_mapping: mappings.get(&ptr).cloned().flatten(),
                        },
                    );
                }
                added_any = true;
            }

            if !added_any {
                break;
            }
        }

        selected
    }

    fn find_join_rel_superset(
        &self,
        occs: &[&'plan LogicalPlan],
    ) -> Option<JoinRelSuperset<'plan>> {
        let mut candidates: Vec<usize> = (0..occs.len()).collect();
        candidates.sort_by(|&a, &b| {
            let la = self.nodes[&plan_ptr(occs[a])]
                .join_rel
                .as_ref()
                .unwrap()
                .provenance
                .len();
            let lb = self.nodes[&plan_ptr(occs[b])]
                .join_rel
                .as_ref()
                .unwrap()
                .provenance
                .len();
            lb.cmp(&la)
        });

        for rep_idx in candidates {
            let rep_plan = occs[rep_idx];
            let rep_rel = self.nodes[&plan_ptr(rep_plan)].join_rel.as_ref()?;
            let mut rep_col_lookup: HashMap<ColOrigin, usize> =
                HashMap::with_capacity(rep_rel.provenance.len());
            for (idx, &orig) in rep_rel.provenance.iter().enumerate() {
                rep_col_lookup.entry(orig).or_insert(idx);
            }

            let mut mappings: HashMap<usize, Option<Vec<usize>>> =
                HashMap::with_capacity(occs.len());
            let mut all_covered = true;
            for &occ_plan in occs {
                let occ_ptr = plan_ptr(occ_plan);
                let occ_rel = self.nodes[&occ_ptr].join_rel.as_ref()?;
                let mut col_map = Vec::with_capacity(occ_rel.provenance.len());
                let mut is_identity =
                    occ_rel.provenance.len() == rep_rel.provenance.len();
                for (out_idx, (&orig, occ_ty)) in occ_rel
                    .provenance
                    .iter()
                    .zip(occ_rel.output_types.iter())
                    .enumerate()
                {
                    let Some(&src_idx) = rep_col_lookup.get(&orig) else {
                        all_covered = false;
                        break;
                    };
                    if rep_rel.output_types[src_idx] != *occ_ty {
                        all_covered = false;
                        break;
                    }
                    if src_idx != out_idx {
                        is_identity = false;
                    }
                    col_map.push(src_idx);
                }
                if !all_covered {
                    break;
                }
                mappings.insert(occ_ptr, (!is_identity).then_some(col_map));
            }
            if all_covered {
                return Some((rep_plan, mappings));
            }
        }
        None
    }
}

#[derive(Clone, Copy)]
struct OccEntry<'plan> {
    plan: &'plan LogicalPlan,
    under_filtered_outer_join: bool,
}

#[derive(Default)]
#[allow(clippy::allow_attributes, clippy::mutable_key_type)]
struct RoundCounter<'plan> {
    exact_occs: HashMap<ExactId, Vec<OccEntry<'plan>>>,
    join_rel_occs: HashMap<JoinRelId, Vec<OccEntry<'plan>>>,
    visited_groups: HashSet<usize>,
    visited_subqueries: HashSet<Subquery>,
}

impl<'plan> RoundCounter<'plan> {
    fn walk(
        &mut self,
        plan: &'plan LogicalPlan,
        analyzer: &Analyzer<'plan>,
        already_selected: &HashMap<usize, CseOccurrence<'plan>>,
        under_filtered_outer_join: bool,
        in_join_rel_chain: bool,
    ) {
        if let LogicalPlan::SubqueryAlias(alias) = plan {
            self.walk(
                &alias.input,
                analyzer,
                already_selected,
                under_filtered_outer_join,
                in_join_rel_chain,
            );
            return;
        }
        if let LogicalPlan::MaterializedCte(MaterializedCtePlan::Cte(cte)) = plan {
            if cte.scans().is_ok_and(|s| !s.is_empty()) {
                self.walk(cte.body(), analyzer, already_selected, false, false);
            }
            self.walk(
                cte.continuation(),
                analyzer,
                already_selected,
                under_filtered_outer_join,
                false,
            );
            return;
        }

        let ptr = plan_ptr(plan);
        if let Some(occ) = already_selected.get(&ptr) {
            if self.visited_groups.insert(occ.group_id) {
                self.walk_children_with_chain(
                    occ.representative,
                    analyzer,
                    already_selected,
                    false,
                    false,
                );
            }
            return;
        }

        let mut next_in_join_rel_chain = false;
        if let Some(info) = analyzer.nodes.get(&ptr) {
            self.exact_occs
                .entry(info.exact_id)
                .or_default()
                .push(OccEntry {
                    plan,
                    under_filtered_outer_join,
                });
            if !in_join_rel_chain && let Some(jr) = &info.join_rel {
                self.join_rel_occs
                    .entry(jr.rel_id)
                    .or_default()
                    .push(OccEntry {
                        plan,
                        under_filtered_outer_join,
                    });
                if matches!(plan, LogicalPlan::Projection(_)) {
                    next_in_join_rel_chain = true;
                }
            } else if in_join_rel_chain && matches!(plan, LogicalPlan::Projection(_)) {
                next_in_join_rel_chain = true;
            }
        }

        self.walk_children_with_chain(
            plan,
            analyzer,
            already_selected,
            under_filtered_outer_join,
            next_in_join_rel_chain,
        );
    }

    fn walk_children_with_chain(
        &mut self,
        plan: &'plan LogicalPlan,
        analyzer: &Analyzer<'plan>,
        already_selected: &HashMap<usize, CseOccurrence<'plan>>,
        under_filtered_outer_join: bool,
        next_in_join_rel_chain: bool,
    ) {
        let mut sqs: Vec<&'plan Subquery> = Vec::new();
        for_each_scalar_subquery(plan, |sq| sqs.push(sq));
        for sq in sqs {
            if self.visited_subqueries.insert(sq.clone())
                && let Some(&sub_root) = analyzer.subquery_roots.get(sq)
            {
                self.walk(sub_root, analyzer, already_selected, false, false);
            }
        }

        match plan {
            LogicalPlan::Join(join) => {
                let left_has_filt = subtree_has_filter(&join.left);
                let right_has_filt = subtree_has_filter(&join.right);
                self.walk(
                    &join.left,
                    analyzer,
                    already_selected,
                    under_filtered_outer_join || right_has_filt,
                    false,
                );
                self.walk(
                    &join.right,
                    analyzer,
                    already_selected,
                    under_filtered_outer_join || left_has_filt,
                    false,
                );
            }
            LogicalPlan::Aggregate(agg) => {
                self.walk(&agg.input, analyzer, already_selected, false, false);
            }
            LogicalPlan::Distinct(Distinct::All(input)) => {
                self.walk(input, analyzer, already_selected, false, false);
            }
            LogicalPlan::Sort(sort) => {
                self.walk(&sort.input, analyzer, already_selected, false, false);
            }
            LogicalPlan::Limit(limit) => {
                self.walk(&limit.input, analyzer, already_selected, false, false);
            }
            LogicalPlan::Projection(proj) => {
                self.walk(
                    &proj.input,
                    analyzer,
                    already_selected,
                    under_filtered_outer_join,
                    next_in_join_rel_chain,
                );
            }
            LogicalPlan::Filter(filter) => {
                self.walk(
                    &filter.input,
                    analyzer,
                    already_selected,
                    under_filtered_outer_join,
                    false,
                );
            }
            _ => {}
        }
    }
}

#[allow(clippy::allow_attributes, clippy::mutable_key_type)]
struct MaximalCollector<'a, 'plan> {
    analyzer: &'a Analyzer<'plan>,
    already_selected: &'a HashMap<usize, CseOccurrence<'plan>>,
    exact_eligible: &'a HashSet<ExactId>,
    join_rel_eligible: &'a HashSet<JoinRelId>,
    exact_maximal: IndexMap<ExactId, Vec<&'plan LogicalPlan>>,
    join_rel_maximal: IndexMap<JoinRelId, Vec<&'plan LogicalPlan>>,
    visited_groups: HashSet<usize>,
    visited_subqueries: HashSet<Subquery>,
}

impl<'plan> MaximalCollector<'_, 'plan> {
    fn walk(&mut self, plan: &'plan LogicalPlan, in_join_rel_chain: bool) {
        if let LogicalPlan::SubqueryAlias(alias) = plan {
            self.walk(&alias.input, in_join_rel_chain);
            return;
        }
        if let LogicalPlan::MaterializedCte(MaterializedCtePlan::Cte(cte)) = plan {
            if cte.scans().is_ok_and(|s| !s.is_empty()) {
                self.walk(cte.body(), false);
            }
            self.walk(cte.continuation(), false);
            return;
        }

        let ptr = plan_ptr(plan);
        if let Some(occ) = self.already_selected.get(&ptr) {
            if self.visited_groups.insert(occ.group_id) {
                self.walk_children(occ.representative, false);
            }
            return;
        }

        let mut next_in_join_rel_chain = false;
        if let Some(info) = self.analyzer.nodes.get(&ptr) {
            if self.exact_eligible.contains(&info.exact_id) {
                self.exact_maximal
                    .entry(info.exact_id)
                    .or_default()
                    .push(plan);
                return;
            }
            if !in_join_rel_chain
                && let Some(jr) = &info.join_rel
                && self.join_rel_eligible.contains(&jr.rel_id)
            {
                self.join_rel_maximal
                    .entry(jr.rel_id)
                    .or_default()
                    .push(plan);
                return;
            }
            if matches!(plan, LogicalPlan::Projection(_))
                && (in_join_rel_chain || info.join_rel.is_some())
            {
                next_in_join_rel_chain = true;
            }
        }

        self.walk_children(plan, next_in_join_rel_chain);
    }

    fn walk_children(&mut self, plan: &'plan LogicalPlan, next_in_join_rel_chain: bool) {
        let mut sqs: Vec<&'plan Subquery> = Vec::new();
        for_each_scalar_subquery(plan, |sq| sqs.push(sq));
        for sq in sqs {
            if self.visited_subqueries.insert(sq.clone())
                && let Some(&sub_root) = self.analyzer.subquery_roots.get(sq)
            {
                self.walk(sub_root, false);
            }
        }

        match plan {
            LogicalPlan::Join(join) => {
                self.walk(&join.left, false);
                self.walk(&join.right, false);
            }
            LogicalPlan::Aggregate(agg) => self.walk(&agg.input, false),
            LogicalPlan::Distinct(Distinct::All(input)) => self.walk(input, false),
            LogicalPlan::Sort(sort) => self.walk(&sort.input, false),
            LogicalPlan::Limit(limit) => self.walk(&limit.input, false),
            LogicalPlan::Projection(proj) => {
                self.walk(&proj.input, next_in_join_rel_chain)
            }
            LogicalPlan::Filter(filter) => self.walk(&filter.input, false),
            _ => {}
        }
    }
}
