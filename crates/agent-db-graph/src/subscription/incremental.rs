//! Phase 3A: Incremental view maintenance types and operator states.
//!
//! Supports a single linear plan shape: ScanNodes → [Expand] → [Filter]* → Project
//! with ungrouped count(*) as the only aggregation.

use rustc_hash::{FxHashMap, FxHashSet};
use smallvec::SmallVec;

use crate::query_lang::ast::{CompOp, Direction as AstDirection, Literal};
use crate::query_lang::executor::{
    edge_matches_type_standalone, edge_visible_standalone, node_matches_props_standalone,
};
use crate::query_lang::planner::{
    AggregateFunction, ExecutionPlan, PlanStep, RBoolExpr, RExpr, SlotIdx, TemporalViewport,
};
use crate::query_lang::types::Value;
use crate::structures::{EdgeId, Graph, GraphEdge, GraphNode, NodeId};

use super::delta::GraphDelta;
use super::trigger::{compile_trigger_set, TriggerSet};

// ---------------------------------------------------------------------------
// RowId — Structural Binding Identity
// ---------------------------------------------------------------------------

/// Canonical row identity = the sorted binding tuple itself.
/// NOT a hash. This is the structural identity of a row.
/// Two rows are the same iff they bind the same entities in the same slots.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct RowId {
    /// Sorted by slot index. This IS the identity, not a lossy projection.
    slots: SmallVec<[(SlotIdx, BoundEntityId); 4]>,
}

impl RowId {
    pub fn new(mut slots: SmallVec<[(SlotIdx, BoundEntityId); 4]>) -> Self {
        slots.sort_by_key(|(idx, _)| *idx);
        Self { slots }
    }

    pub fn slots(&self) -> &[(SlotIdx, BoundEntityId)] {
        &self.slots
    }

    /// Get the bound entity for a specific slot, if present.
    pub fn get(&self, slot: SlotIdx) -> Option<&BoundEntityId> {
        self.slots
            .iter()
            .find(|(idx, _)| *idx == slot)
            .map(|(_, id)| id)
    }

    /// Extend this RowId with additional slot bindings.
    pub fn extend(&self, extra: &[(SlotIdx, BoundEntityId)]) -> Self {
        let mut slots = self.slots.clone();
        slots.extend_from_slice(extra);
        slots.sort_by_key(|(idx, _)| *idx);
        Self { slots }
    }
}

/// A bound entity in a query result row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BoundEntityId {
    Node(NodeId),
    Edge(EdgeId),
}

// ---------------------------------------------------------------------------
// RowDelta — Row-level Change Propagation
// ---------------------------------------------------------------------------

/// A row-level change carrying the full binding context.
/// Downstream operators receive the binding payload directly —
/// no shared-store lookup required.
#[derive(Debug, Clone)]
pub enum RowDelta {
    /// A new row entered the operator's output.
    /// `row_id` IS the binding (structural identity = binding tuple).
    Insert { row_id: RowId },
    /// A row left the operator's output.
    Delete { row_id: RowId },
}

impl RowDelta {
    pub fn row_id(&self) -> &RowId {
        match self {
            RowDelta::Insert { row_id } | RowDelta::Delete { row_id } => row_id,
        }
    }

    pub fn insert(row_id: RowId) -> Self {
        RowDelta::Insert { row_id }
    }

    pub fn delete(row_id: RowId) -> Self {
        RowDelta::Delete { row_id }
    }

    pub fn is_insert(&self) -> bool {
        matches!(self, RowDelta::Insert { .. })
    }
}

// ---------------------------------------------------------------------------
// ActiveRows — Subscription Row Tracking
// ---------------------------------------------------------------------------

// ActiveRows removed — cached_output.rows is the sole source of truth.

// ---------------------------------------------------------------------------
// IncrementalPlan — Plan Classification
// ---------------------------------------------------------------------------

/// Whether the subscription can be maintained incrementally or needs full rerun.
#[derive(Debug, Clone)]
pub enum MaintenanceStrategy {
    Incremental,
    FullRerun { reason: String },
}

/// A classified execution plan for subscription maintenance.
#[derive(Debug, Clone)]
pub struct IncrementalPlan {
    pub plan: ExecutionPlan,
    pub strategy: MaintenanceStrategy,
    pub trigger_set: TriggerSet,
}

impl IncrementalPlan {
    /// Analyze a plan and determine maintenance strategy.
    ///
    /// Supported shape: ScanNodes → [Expand(range=None)] → [Filter]* → Project
    /// Supported aggregation: ungrouped count(*) only
    pub fn analyze(plan: ExecutionPlan) -> Self {
        let trigger_set = compile_trigger_set(&plan);
        let strategy = Self::classify(&plan);
        Self {
            plan,
            strategy,
            trigger_set,
        }
    }

    fn classify(plan: &ExecutionPlan) -> MaintenanceStrategy {
        if plan.steps.is_empty() {
            return MaintenanceStrategy::FullRerun {
                reason: "empty plan".to_string(),
            };
        }

        // Step 0 must be ScanNodes.
        if !matches!(plan.steps[0], PlanStep::ScanNodes { .. }) {
            return MaintenanceStrategy::FullRerun {
                reason: "first step is not ScanNodes".to_string(),
            };
        }

        // Check for exactly one ScanNodes step.
        let scan_count = plan
            .steps
            .iter()
            .filter(|s| matches!(s, PlanStep::ScanNodes { .. }))
            .count();
        if scan_count != 1 {
            return MaintenanceStrategy::FullRerun {
                reason: "multi-pattern plan".to_string(),
            };
        }

        // Check Expand steps.
        let expand_steps: Vec<(usize, &PlanStep)> = plan
            .steps
            .iter()
            .enumerate()
            .filter(|(_, s)| matches!(s, PlanStep::Expand { .. }))
            .collect();

        if expand_steps.len() > 1 {
            return MaintenanceStrategy::FullRerun {
                reason: "multiple expand steps".to_string(),
            };
        }

        if let Some((idx, _step)) = expand_steps.first() {
            // Expand must be step[1].
            if *idx != 1 {
                return MaintenanceStrategy::FullRerun {
                    reason: "expand is not immediately after scan".to_string(),
                };
            }
            // Both single-hop (range=None) and variable-length (range=Some)
            // are now supported incrementally.
        }

        // Remaining steps (after scan + optional expand) must all be Filter.
        let first_filter_idx = if expand_steps.is_empty() { 1 } else { 2 };
        for step in &plan.steps[first_filter_idx..] {
            if !matches!(step, PlanStep::Filter(_)) {
                return MaintenanceStrategy::FullRerun {
                    reason: "unsupported step after expand".to_string(),
                };
            }
        }

        // Check aggregation: single function allowed (grouped or ungrouped).
        if !plan.aggregations.is_empty() {
            if plan.aggregations.len() != 1 {
                return MaintenanceStrategy::FullRerun {
                    reason: "multiple aggregations".to_string(),
                };
            }
        }

        // Check for FuncPredicate in filters — not supported incrementally.
        for step in &plan.steps {
            if let PlanStep::Filter(expr) = step {
                if contains_func_predicate(expr) {
                    return MaintenanceStrategy::FullRerun {
                        reason: "filter contains function predicate".to_string(),
                    };
                }
            }
        }

        // All temporal viewports are now supported incrementally.
        // PointInTime/Range: edge visibility is checked via edge_visible_standalone
        // which already handles these viewports. EdgeSuperseded deltas carry
        // old/new valid_until so we can determine if an edge enters/exits the viewport.

        MaintenanceStrategy::Incremental
    }
}

// ---------------------------------------------------------------------------
// Operator States
// ---------------------------------------------------------------------------

/// ScanNodes operator state.
pub struct ScanState {
    pub var: SlotIdx,
    pub label_discs: Vec<u8>,
    pub prop_filters: Vec<(String, Literal)>,
    pub active_nodes: FxHashSet<NodeId>,
}

impl ScanState {
    /// Initialize from a ScanNodes plan step, populating active_nodes from the graph.
    pub fn init(
        var: SlotIdx,
        labels: &[String],
        props: &[(String, Literal)],
        graph: &Graph,
    ) -> Self {
        let label_discs: Vec<u8> = labels
            .iter()
            .filter_map(|l| label_to_discriminant(l))
            .collect();
        let prop_filters = props.to_vec();

        let mut active_nodes = FxHashSet::default();

        // Populate initial active set.
        // Use type index when labels are provided for O(matching) instead of O(all nodes).
        if labels.is_empty() {
            for node in graph.nodes() {
                if node_matches_props_standalone(node, &prop_filters, graph) {
                    active_nodes.insert(node.id);
                }
            }
        } else {
            for label in labels {
                for node in graph.get_nodes_by_type(label) {
                    if node_matches_props_standalone(node, &prop_filters, graph) {
                        active_nodes.insert(node.id);
                    }
                }
            }
        }

        Self {
            var,
            label_discs,
            prop_filters,
            active_nodes,
        }
    }

    /// Process a graph delta through this scan operator.
    /// Returns row deltas for downstream consumption.
    pub fn apply_delta(&mut self, delta: &GraphDelta, graph: &Graph) -> SmallVec<[RowDelta; 2]> {
        let mut out = SmallVec::new();

        match delta {
            GraphDelta::NodeAdded {
                node_id,
                node_type_disc,
                ..
            } => {
                // Check type filter.
                if !self.label_discs.is_empty() && !self.label_discs.contains(node_type_disc) {
                    return out;
                }
                // Check property filter.
                if let Some(node) = graph.get_node(*node_id) {
                    if node_matches_props_standalone(node, &self.prop_filters, graph) {
                        self.active_nodes.insert(*node_id);
                        let row_id = RowId::new(smallvec::smallvec![(
                            self.var,
                            BoundEntityId::Node(*node_id)
                        )]);
                        out.push(RowDelta::insert(row_id));
                    }
                }
            },
            GraphDelta::NodeRemoved { node_id, .. } => {
                if self.active_nodes.remove(node_id) {
                    let row_id = RowId::new(smallvec::smallvec![(
                        self.var,
                        BoundEntityId::Node(*node_id)
                    )]);
                    out.push(RowDelta::delete(row_id));
                }
            },
            // Edge deltas don't affect scan state.
            _ => {},
        }

        out
    }
}

/// Single-hop Expand operator state.
///
/// # Upstream lineage assumption
///
/// `from_var` always refers to the single preceding ScanState's `var`.
/// ExpandState checks `scan_state.active_nodes.contains(&from_node)` to
/// determine relevance. This is sound because:
/// 1. The plan is a single linear chain (validated at subscription time).
/// 2. There is no join — the only source of `from_var` bindings is the ScanState.
/// 3. ScanState maintains `active_nodes` as the authoritative set of bound source nodes.
pub struct ExpandState {
    pub from_var: SlotIdx,
    pub edge_var: Option<SlotIdx>,
    pub to_var: SlotIdx,
    pub edge_type_filter: Option<String>,
    pub direction: AstDirection,
    /// Forward index: from_node → [(edge_id, to_node)]
    pub expansions: FxHashMap<NodeId, SmallVec<[(EdgeId, NodeId); 4]>>,
}

impl ExpandState {
    /// Initialize from an Expand plan step, populating expansions for all active scan nodes.
    pub fn init(
        from_var: SlotIdx,
        edge_var: Option<SlotIdx>,
        to_var: SlotIdx,
        edge_type: &Option<String>,
        direction: &AstDirection,
        scan_state: &ScanState,
        graph: &Graph,
        viewport: &TemporalViewport,
        txn_cutoff: Option<u64>,
    ) -> Self {
        let mut state = Self {
            from_var,
            edge_var,
            to_var,
            edge_type_filter: edge_type.clone(),
            direction: direction.clone(),
            expansions: FxHashMap::default(),
        };

        // Populate initial expansions from active scan nodes.
        for &node_id in &scan_state.active_nodes {
            state.expand_from_node(node_id, graph, viewport, txn_cutoff);
        }

        state
    }

    /// Expand from a single source node, adding matching edges to the expansion index.
    /// Clears any existing expansions for this node first to prevent duplicates.
    fn expand_from_node(
        &mut self,
        from_node: NodeId,
        graph: &Graph,
        viewport: &TemporalViewport,
        txn_cutoff: Option<u64>,
    ) {
        let edges = match self.direction {
            AstDirection::Out => graph.get_edges_from(from_node),
            AstDirection::In => graph.get_edges_to(from_node),
        };

        // Clear existing expansions for this node to prevent duplicates
        // when re-expanding (e.g., after upstream re-insert).
        let entry = self.expansions.entry(from_node).or_default();
        entry.clear();

        for edge in edges {
            if !edge_visible_standalone(edge, viewport, txn_cutoff) {
                continue;
            }
            if !edge_matches_type_standalone(edge, &self.edge_type_filter) {
                continue;
            }
            let to_node = match self.direction {
                AstDirection::Out => edge.target,
                AstDirection::In => edge.source,
            };
            entry.push((edge.id, to_node));
        }
    }

    /// Build a RowId for an expanded row.
    pub(crate) fn make_row_id(&self, from_node: NodeId, edge_id: EdgeId, to_node: NodeId) -> RowId {
        let mut slots: SmallVec<[(SlotIdx, BoundEntityId); 4]> = smallvec::smallvec![
            (self.from_var, BoundEntityId::Node(from_node)),
            (self.to_var, BoundEntityId::Node(to_node)),
        ];
        if let Some(ev) = self.edge_var {
            slots.push((ev, BoundEntityId::Edge(edge_id)));
        }
        RowId::new(slots)
    }

    /// Process upstream scan deltas (new/removed source nodes).
    pub fn apply_upstream_deltas(
        &mut self,
        scan_deltas: &[RowDelta],
        graph: &Graph,
        viewport: &TemporalViewport,
        txn_cutoff: Option<u64>,
    ) -> Vec<RowDelta> {
        let mut out = Vec::new();

        for delta in scan_deltas {
            match delta {
                RowDelta::Insert { row_id } => {
                    // Extract from_node from the scan row.
                    if let Some(BoundEntityId::Node(from_node)) = row_id.get(self.from_var) {
                        let from_node = *from_node;
                        self.expand_from_node(from_node, graph, viewport, txn_cutoff);
                        // Emit inserts for all expansions.
                        if let Some(exps) = self.expansions.get(&from_node) {
                            for &(eid, to_node) in exps {
                                out.push(RowDelta::insert(
                                    self.make_row_id(from_node, eid, to_node),
                                ));
                            }
                        }
                    }
                },
                RowDelta::Delete { row_id } => {
                    // Remove all expansions for the deleted source node.
                    if let Some(BoundEntityId::Node(from_node)) = row_id.get(self.from_var) {
                        let from_node = *from_node;
                        if let Some(exps) = self.expansions.remove(&from_node) {
                            for (eid, to_node) in exps {
                                out.push(RowDelta::delete(
                                    self.make_row_id(from_node, eid, to_node),
                                ));
                            }
                        }
                    }
                },
            }
        }

        out
    }

    /// Process a direct edge delta (edge added/removed/superseded/mutated).
    pub fn apply_edge_delta(
        &mut self,
        delta: &GraphDelta,
        scan_state: &ScanState,
        graph: &Graph,
        viewport: &TemporalViewport,
        txn_cutoff: Option<u64>,
    ) -> Vec<RowDelta> {
        let mut out = Vec::new();

        match delta {
            GraphDelta::EdgeAdded {
                edge_id,
                source,
                target,
                ..
            } => {
                let (from_node, to_node) = match self.direction {
                    AstDirection::Out => (*source, *target),
                    AstDirection::In => (*target, *source),
                };
                if !scan_state.active_nodes.contains(&from_node) {
                    return out;
                }
                // Check visibility and type match.
                if let Some(edge) = graph.get_edge(*edge_id) {
                    if edge_visible_standalone(edge, viewport, txn_cutoff)
                        && edge_matches_type_standalone(edge, &self.edge_type_filter)
                    {
                        self.expansions
                            .entry(from_node)
                            .or_default()
                            .push((*edge_id, to_node));
                        out.push(RowDelta::insert(
                            self.make_row_id(from_node, *edge_id, to_node),
                        ));
                    }
                }
            },
            GraphDelta::EdgeRemoved {
                edge_id,
                source,
                target,
                ..
            } => {
                let (from_node, to_node) = match self.direction {
                    AstDirection::Out => (*source, *target),
                    AstDirection::In => (*target, *source),
                };
                if let Some(exps) = self.expansions.get_mut(&from_node) {
                    let before_len = exps.len();
                    exps.retain(|(eid, _)| eid != edge_id);
                    if exps.len() < before_len {
                        out.push(RowDelta::delete(
                            self.make_row_id(from_node, *edge_id, to_node),
                        ));
                    }
                }
            },
            GraphDelta::EdgeSuperseded {
                edge_id,
                source,
                target,
                old_valid_until,
                ..
            } => {
                let (from_node, to_node) = match self.direction {
                    AstDirection::Out => (*source, *target),
                    AstDirection::In => (*target, *source),
                };
                if !scan_state.active_nodes.contains(&from_node) {
                    return out;
                }

                // Determine old/new visibility under the current viewport.
                // Re-read the edge from graph to get full state for visibility check.
                let was_in = if let Some(edge) = graph.get_edge(*edge_id) {
                    // Simulate old state: temporarily check with old_valid_until
                    match viewport {
                        TemporalViewport::ActiveOnly => old_valid_until.is_none(),
                        TemporalViewport::All => true,
                        TemporalViewport::PointInTime(ts) => {
                            let from = edge.valid_from.unwrap_or(0);
                            let old_end = *old_valid_until;
                            from <= *ts && old_end.map_or(true, |u| u > *ts)
                        },
                        TemporalViewport::Range(t1, t2) => {
                            let from = edge.valid_from.unwrap_or(0);
                            let old_end = *old_valid_until;
                            from <= *t2 && old_end.map_or(true, |u| u >= *t1)
                        },
                    }
                } else {
                    false
                };

                let is_in = if let Some(edge) = graph.get_edge(*edge_id) {
                    edge_visible_standalone(edge, viewport, txn_cutoff)
                } else {
                    false
                };

                if was_in && !is_in {
                    // Edge left the viewport → removal.
                    if let Some(exps) = self.expansions.get_mut(&from_node) {
                        let before_len = exps.len();
                        exps.retain(|(eid, _)| eid != edge_id);
                        if exps.len() < before_len {
                            out.push(RowDelta::delete(
                                self.make_row_id(from_node, *edge_id, to_node),
                            ));
                        }
                    }
                } else if !was_in && is_in {
                    // Edge entered the viewport → addition.
                    self.expansions
                        .entry(from_node)
                        .or_default()
                        .push((*edge_id, to_node));
                    out.push(RowDelta::insert(
                        self.make_row_id(from_node, *edge_id, to_node),
                    ));
                }
            },
            GraphDelta::EdgeMutated {
                edge_id,
                source,
                target,
                ..
            } => {
                let (from_node, to_node) = match self.direction {
                    AstDirection::Out => (*source, *target),
                    AstDirection::In => (*target, *source),
                };
                // If this edge is in our expansions, emit Delete+Insert to force
                // downstream re-evaluation (filter/projection values may have changed).
                if let Some(exps) = self.expansions.get(&from_node) {
                    if exps.iter().any(|(eid, _)| eid == edge_id) {
                        let rid = self.make_row_id(from_node, *edge_id, to_node);
                        out.push(RowDelta::delete(rid.clone()));
                        out.push(RowDelta::insert(rid));
                    }
                }
            },
            _ => {}, // Node deltas handled by scan state.
        }

        out
    }
}

/// Filter operator state.
pub struct FilterState {
    pub filter: RBoolExpr,
    pub passed: FxHashSet<RowId>,
}

impl FilterState {
    pub fn init(filter: RBoolExpr) -> Self {
        Self {
            filter,
            passed: FxHashSet::default(),
        }
    }

    /// Initialize with pre-populated passed set from initial execution.
    pub fn init_with_rows(filter: RBoolExpr, passed_rows: FxHashSet<RowId>) -> Self {
        Self {
            filter,
            passed: passed_rows,
        }
    }

    /// Process upstream deltas through this filter.
    /// Requires graph access to evaluate property expressions.
    pub fn apply_deltas(&mut self, upstream_deltas: &[RowDelta], graph: &Graph) -> Vec<RowDelta> {
        let mut out = Vec::new();

        for delta in upstream_deltas {
            match delta {
                RowDelta::Insert { row_id } => {
                    if evaluate_filter_for_row(&self.filter, row_id, graph) {
                        self.passed.insert(row_id.clone());
                        out.push(RowDelta::insert(row_id.clone()));
                    }
                },
                RowDelta::Delete { row_id } => {
                    if self.passed.remove(row_id) {
                        out.push(RowDelta::delete(row_id.clone()));
                    }
                },
            }
        }

        out
    }
}

/// Ungrouped count(*) aggregation state (kept for backward compatibility).
pub struct CountState {
    pub count: i64,
}

impl CountState {
    pub fn new(initial_count: i64) -> Self {
        Self {
            count: initial_count,
        }
    }

    /// Apply row deltas and return updated count.
    pub fn apply_deltas(&mut self, deltas: &[RowDelta]) -> i64 {
        for d in deltas {
            match d {
                RowDelta::Insert { .. } => self.count += 1,
                RowDelta::Delete { .. } => self.count -= 1,
            }
        }
        self.count
    }
}

// ---------------------------------------------------------------------------
// Advanced Aggregation States
// ---------------------------------------------------------------------------

/// Unified aggregation state for all supported functions.
pub enum AggregationState {
    Count(CountState),
    Sum(SumState),
    Avg(AvgState),
    MinMax(MinMaxState),
    Collect(CollectState),
}

impl AggregationState {
    /// Create from an aggregate function and initial row set.
    pub fn init(
        function: &AggregateFunction,
        input_expr: &RExpr,
        initial_rows: &[RowId],
        graph: &Graph,
    ) -> Self {
        match function {
            AggregateFunction::Count => {
                AggregationState::Count(CountState::new(initial_rows.len() as i64))
            },
            AggregateFunction::Sum => {
                let mut sum = 0.0f64;
                for row in initial_rows {
                    sum += extract_f64(input_expr, row, graph);
                }
                AggregationState::Sum(SumState {
                    sum,
                    input_expr: input_expr.clone(),
                })
            },
            AggregateFunction::Avg => {
                let mut sum = 0.0f64;
                for row in initial_rows {
                    sum += extract_f64(input_expr, row, graph);
                }
                AggregationState::Avg(AvgState {
                    sum,
                    count: initial_rows.len() as i64,
                    input_expr: input_expr.clone(),
                })
            },
            AggregateFunction::Min => {
                let mut values = FxHashMap::default();
                for row in initial_rows {
                    let val = evaluate_expr_for_row_or_null(input_expr, row, graph);
                    values.insert(row.clone(), val);
                }
                let current = find_extreme(&values, true);
                AggregationState::MinMax(MinMaxState {
                    is_min: true,
                    input_expr: input_expr.clone(),
                    values,
                    current_extreme: current,
                })
            },
            AggregateFunction::Max => {
                let mut values = FxHashMap::default();
                for row in initial_rows {
                    let val = evaluate_expr_for_row_or_null(input_expr, row, graph);
                    values.insert(row.clone(), val);
                }
                let current = find_extreme(&values, false);
                AggregationState::MinMax(MinMaxState {
                    is_min: false,
                    input_expr: input_expr.clone(),
                    values,
                    current_extreme: current,
                })
            },
            AggregateFunction::Collect => {
                let mut items = FxHashMap::default();
                for row in initial_rows {
                    let val = evaluate_expr_for_row_or_null(input_expr, row, graph);
                    items.insert(row.clone(), val);
                }
                AggregationState::Collect(CollectState {
                    input_expr: input_expr.clone(),
                    items,
                })
            },
        }
    }

    /// Apply row deltas and return the current aggregate as a Value.
    pub fn apply_deltas(&mut self, deltas: &[RowDelta], graph: &Graph) -> Value {
        match self {
            AggregationState::Count(cs) => Value::Int(cs.apply_deltas(deltas)),
            AggregationState::Sum(s) => s.apply_deltas(deltas, graph),
            AggregationState::Avg(a) => a.apply_deltas(deltas, graph),
            AggregationState::MinMax(mm) => mm.apply_deltas(deltas, graph),
            AggregationState::Collect(c) => c.apply_deltas(deltas, graph),
        }
    }

    /// Get current aggregate value without applying any deltas.
    pub fn current_value(&self) -> Value {
        match self {
            AggregationState::Count(cs) => Value::Int(cs.count),
            AggregationState::Sum(s) => Value::Float(s.sum),
            AggregationState::Avg(a) => {
                if a.count == 0 {
                    Value::Null
                } else {
                    Value::Float(a.sum / a.count as f64)
                }
            },
            AggregationState::MinMax(mm) => mm.current_extreme.clone().unwrap_or(Value::Null),
            AggregationState::Collect(c) => Value::List(c.items.values().cloned().collect()),
        }
    }
}

/// Sum accumulator.
pub struct SumState {
    pub sum: f64,
    pub input_expr: RExpr,
}

impl SumState {
    fn apply_deltas(&mut self, deltas: &[RowDelta], graph: &Graph) -> Value {
        for d in deltas {
            match d {
                RowDelta::Insert { row_id } => {
                    self.sum += extract_f64(&self.input_expr, row_id, graph);
                },
                RowDelta::Delete { row_id } => {
                    self.sum -= extract_f64(&self.input_expr, row_id, graph);
                },
            }
        }
        Value::Float(self.sum)
    }
}

/// Average accumulator (maintains running sum + count).
pub struct AvgState {
    pub sum: f64,
    pub count: i64,
    pub input_expr: RExpr,
}

impl AvgState {
    fn apply_deltas(&mut self, deltas: &[RowDelta], graph: &Graph) -> Value {
        for d in deltas {
            match d {
                RowDelta::Insert { row_id } => {
                    self.sum += extract_f64(&self.input_expr, row_id, graph);
                    self.count += 1;
                },
                RowDelta::Delete { row_id } => {
                    self.sum -= extract_f64(&self.input_expr, row_id, graph);
                    self.count -= 1;
                },
            }
        }
        if self.count == 0 {
            Value::Null
        } else {
            Value::Float(self.sum / self.count as f64)
        }
    }
}

/// Min/Max accumulator. Stores all values for O(n) re-scan on extreme deletion.
pub struct MinMaxState {
    pub is_min: bool,
    pub input_expr: RExpr,
    pub values: FxHashMap<RowId, Value>,
    pub current_extreme: Option<Value>,
}

impl MinMaxState {
    fn apply_deltas(&mut self, deltas: &[RowDelta], graph: &Graph) -> Value {
        for d in deltas {
            match d {
                RowDelta::Insert { row_id } => {
                    let val = evaluate_expr_for_row_or_null(&self.input_expr, row_id, graph);
                    self.values.insert(row_id.clone(), val.clone());
                    // Update extreme if new value beats it.
                    self.current_extreme = match &self.current_extreme {
                        None => Some(val),
                        Some(cur) => {
                            if self.is_min {
                                if val.partial_cmp(cur) == Some(std::cmp::Ordering::Less) {
                                    Some(val)
                                } else {
                                    self.current_extreme.clone()
                                }
                            } else {
                                if val.partial_cmp(cur) == Some(std::cmp::Ordering::Greater) {
                                    Some(val)
                                } else {
                                    self.current_extreme.clone()
                                }
                            }
                        },
                    };
                },
                RowDelta::Delete { row_id } => {
                    if let Some(removed_val) = self.values.remove(row_id) {
                        // If we removed the current extreme, rescan.
                        if self.current_extreme.as_ref() == Some(&removed_val) {
                            self.current_extreme = find_extreme(&self.values, self.is_min);
                        }
                    }
                },
            }
        }
        self.current_extreme.clone().unwrap_or(Value::Null)
    }
}

/// Collect accumulator.
pub struct CollectState {
    pub input_expr: RExpr,
    pub items: FxHashMap<RowId, Value>,
}

impl CollectState {
    fn apply_deltas(&mut self, deltas: &[RowDelta], graph: &Graph) -> Value {
        for d in deltas {
            match d {
                RowDelta::Insert { row_id } => {
                    let val = evaluate_expr_for_row_or_null(&self.input_expr, row_id, graph);
                    self.items.insert(row_id.clone(), val);
                },
                RowDelta::Delete { row_id } => {
                    self.items.remove(row_id);
                },
            }
        }
        Value::List(self.items.values().cloned().collect())
    }
}

/// Extract a float value from an expression evaluated against a RowId.
fn extract_f64(expr: &RExpr, row_id: &RowId, graph: &Graph) -> f64 {
    match evaluate_expr_for_row_or_null(expr, row_id, graph) {
        Value::Float(f) => f,
        Value::Int(i) => i as f64,
        _ => 0.0,
    }
}

/// Find the min or max value in a map.
fn find_extreme(values: &FxHashMap<RowId, Value>, is_min: bool) -> Option<Value> {
    values
        .values()
        .filter(|v| !v.is_null())
        .cloned()
        .reduce(|best, v| {
            let ord = v.partial_cmp(&best);
            if is_min {
                if ord == Some(std::cmp::Ordering::Less) {
                    v
                } else {
                    best
                }
            } else {
                if ord == Some(std::cmp::Ordering::Greater) {
                    v
                } else {
                    best
                }
            }
        })
}

// ---------------------------------------------------------------------------
// Grouped Aggregation
// ---------------------------------------------------------------------------

/// Hash-safe group key wrapping Vec<Value>.
/// Uses f64::to_bits() for float hashing (consistent with executor).
#[derive(Debug, Clone)]
pub struct GroupKey(pub Vec<Value>);

impl PartialEq for GroupKey {
    fn eq(&self, other: &Self) -> bool {
        if self.0.len() != other.0.len() {
            return false;
        }
        self.0
            .iter()
            .zip(other.0.iter())
            .all(|(a, b)| match (a, b) {
                (Value::Float(fa), Value::Float(fb)) => fa.to_bits() == fb.to_bits(),
                _ => a == b,
            })
    }
}

impl Eq for GroupKey {}

impl std::hash::Hash for GroupKey {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.0.len().hash(state);
        for v in &self.0 {
            std::mem::discriminant(v).hash(state);
            match v {
                Value::String(s) => s.hash(state),
                Value::Int(i) => i.hash(state),
                Value::Float(f) => f.to_bits().hash(state),
                Value::Bool(b) => b.hash(state),
                Value::Null => 0u8.hash(state),
                Value::List(items) => items.len().hash(state),
                Value::Map(m) => m.len().hash(state),
            }
        }
    }
}

/// Per-group entry holding the accumulator and member rows.
pub struct GroupEntry {
    pub accumulator: AggregationState,
    pub member_rows: FxHashSet<RowId>,
}

/// Grouped aggregation state.
pub struct GroupedAggregationState {
    pub group_by_exprs: Vec<RExpr>,
    pub agg_function: AggregateFunction,
    pub agg_input_expr: RExpr,
    pub groups: FxHashMap<GroupKey, GroupEntry>,
}

/// Result of a grouped aggregation delta: which groups were added, removed, or changed.
pub struct GroupedDelta {
    /// Groups that appeared (new group key + aggregate value).
    pub inserted_groups: Vec<(GroupKey, Value)>,
    /// Groups that disappeared (all members removed).
    pub deleted_groups: Vec<GroupKey>,
    /// Groups whose aggregate value changed (old value removed, new value emitted).
    pub updated_groups: Vec<(GroupKey, Value)>,
}

impl GroupedAggregationState {
    /// Initialize from plan and initial rows.
    pub fn init(
        group_by_exprs: Vec<RExpr>,
        agg_function: AggregateFunction,
        agg_input_expr: RExpr,
        initial_rows: &[RowId],
        graph: &Graph,
    ) -> Self {
        let mut groups: FxHashMap<GroupKey, GroupEntry> = FxHashMap::default();

        for row in initial_rows {
            let gk = extract_group_key(&group_by_exprs, row, graph);
            let entry = groups.entry(gk).or_insert_with(|| GroupEntry {
                accumulator: AggregationState::init(&agg_function, &agg_input_expr, &[], graph),
                member_rows: FxHashSet::default(),
            });
            // Apply a synthetic insert to the accumulator.
            let delta = [RowDelta::insert(row.clone())];
            entry.accumulator.apply_deltas(&delta, graph);
            entry.member_rows.insert(row.clone());
        }

        Self {
            group_by_exprs,
            agg_function,
            agg_input_expr,
            groups,
        }
    }

    /// Apply row deltas and return group-level changes.
    pub fn apply_deltas(&mut self, deltas: &[RowDelta], graph: &Graph) -> GroupedDelta {
        let mut inserted = Vec::new();
        let mut deleted = Vec::new();
        let mut updated = Vec::new();

        for delta in deltas {
            match delta {
                RowDelta::Insert { row_id } => {
                    let gk = extract_group_key(&self.group_by_exprs, row_id, graph);
                    let is_new = !self.groups.contains_key(&gk);
                    let entry = self.groups.entry(gk.clone()).or_insert_with(|| GroupEntry {
                        accumulator: AggregationState::init(
                            &self.agg_function,
                            &self.agg_input_expr,
                            &[],
                            graph,
                        ),
                        member_rows: FxHashSet::default(),
                    });
                    let single = [RowDelta::insert(row_id.clone())];
                    let val = entry.accumulator.apply_deltas(&single, graph);
                    entry.member_rows.insert(row_id.clone());

                    if is_new {
                        inserted.push((gk, val));
                    } else {
                        updated.push((gk, val));
                    }
                },
                RowDelta::Delete { row_id } => {
                    let gk = extract_group_key(&self.group_by_exprs, row_id, graph);
                    if let Some(entry) = self.groups.get_mut(&gk) {
                        let single = [RowDelta::delete(row_id.clone())];
                        let val = entry.accumulator.apply_deltas(&single, graph);
                        entry.member_rows.remove(row_id);

                        if entry.member_rows.is_empty() {
                            self.groups.remove(&gk);
                            deleted.push(gk);
                        } else {
                            updated.push((gk, val));
                        }
                    }
                },
            }
        }

        GroupedDelta {
            inserted_groups: inserted,
            deleted_groups: deleted,
            updated_groups: updated,
        }
    }
}

/// Extract a group key from a row's bindings.
fn extract_group_key(exprs: &[RExpr], row_id: &RowId, graph: &Graph) -> GroupKey {
    let values: Vec<Value> = exprs
        .iter()
        .map(|expr| evaluate_expr_for_row_or_null(expr, row_id, graph))
        .collect();
    GroupKey(values)
}

// ---------------------------------------------------------------------------
// Filter evaluation for incremental mode
// ---------------------------------------------------------------------------

/// Evaluate a boolean filter expression against a RowId's bindings.
/// This is a simplified version that reads properties from the graph
/// using the entity IDs in the RowId.
pub(crate) fn evaluate_filter_for_row(expr: &RBoolExpr, row_id: &RowId, graph: &Graph) -> bool {
    match evaluate_bool_for_row(expr, row_id, graph) {
        Ok(v) => v,
        Err(_) => false,
    }
}

fn evaluate_bool_for_row(expr: &RBoolExpr, row_id: &RowId, graph: &Graph) -> Result<bool, ()> {
    match expr {
        RBoolExpr::Comparison(lhs, op, rhs) => {
            let l = evaluate_expr_for_row(lhs, row_id, graph)?;
            let r = evaluate_expr_for_row(rhs, row_id, graph)?;
            Ok(compare_values_simple(&l, op, &r))
        },
        RBoolExpr::IsNull(e) => {
            let v = evaluate_expr_for_row(e, row_id, graph)?;
            Ok(v.is_null())
        },
        RBoolExpr::IsNotNull(e) => {
            let v = evaluate_expr_for_row(e, row_id, graph)?;
            Ok(!v.is_null())
        },
        RBoolExpr::And(a, b) => Ok(
            evaluate_bool_for_row(a, row_id, graph)? && evaluate_bool_for_row(b, row_id, graph)?
        ),
        RBoolExpr::Or(a, b) => Ok(
            evaluate_bool_for_row(a, row_id, graph)? || evaluate_bool_for_row(b, row_id, graph)?
        ),
        RBoolExpr::Not(inner) => Ok(!evaluate_bool_for_row(inner, row_id, graph)?),
        RBoolExpr::Paren(inner) => evaluate_bool_for_row(inner, row_id, graph),
        RBoolExpr::FuncPredicate(name, args) => {
            // For now, function predicates in incremental mode fall through to false.
            // Full support would require porting all predicate functions.
            let _ = (name, args);
            Err(())
        },
    }
}

/// Evaluate an expression for a row, returning Null on unsupported expressions.
pub(crate) fn evaluate_expr_for_row_or_null(expr: &RExpr, row_id: &RowId, graph: &Graph) -> Value {
    evaluate_expr_for_row(expr, row_id, graph).unwrap_or(Value::Null)
}

fn evaluate_expr_for_row(expr: &RExpr, row_id: &RowId, graph: &Graph) -> Result<Value, ()> {
    match expr {
        RExpr::Literal(lit) => Ok(literal_to_value_simple(lit)),
        RExpr::Var(slot) => match row_id.get(*slot) {
            Some(BoundEntityId::Node(id)) => {
                if let Some(node) = graph.get_node(*id) {
                    Ok(Value::String(node.label()))
                } else {
                    Ok(Value::Null)
                }
            },
            Some(BoundEntityId::Edge(id)) => {
                if let Some(edge) = graph.get_edge(*id) {
                    Ok(Value::String(format!("{:?}", edge.edge_type)))
                } else {
                    Ok(Value::Null)
                }
            },
            None => Ok(Value::Null),
        },
        RExpr::Property(slot, prop) => match row_id.get(*slot) {
            Some(BoundEntityId::Node(id)) => {
                if let Some(node) = graph.get_node(*id) {
                    Ok(node_property_value_simple(node, prop, graph))
                } else {
                    Ok(Value::Null)
                }
            },
            Some(BoundEntityId::Edge(id)) => {
                if let Some(edge) = graph.get_edge(*id) {
                    Ok(edge_property_value_simple(edge, prop))
                } else {
                    Ok(Value::Null)
                }
            },
            None => Ok(Value::Null),
        },
        RExpr::FuncCall(..) => Err(()), // Function calls not supported in incremental filter eval
        RExpr::Star => Ok(Value::Null),
    }
}

/// Node property extraction — delegates to the executor's standalone version.
pub(crate) fn node_property_value_simple(node: &GraphNode, prop: &str, _graph: &Graph) -> Value {
    crate::query_lang::executor::node_property_value_standalone(node, prop)
}

/// Simplified edge property extraction (mirrors executor's edge_property_value).
pub(crate) fn edge_property_value_simple(edge: &GraphEdge, prop: &str) -> Value {
    match prop {
        "id" => Value::Int(edge.id as i64),
        "weight" => Value::Float(edge.weight as f64),
        "confidence" => Value::Float(edge.confidence as f64),
        "source" => Value::Int(edge.source as i64),
        "target" => Value::Int(edge.target as i64),
        "created_at" => Value::Int(edge.created_at as i64),
        "updated_at" => Value::Int(edge.updated_at as i64),
        "observation_count" => Value::Int(edge.observation_count as i64),
        _ => Value::Null,
    }
}

fn literal_to_value_simple(lit: &Literal) -> Value {
    match lit {
        Literal::String(s) => Value::String(s.clone()),
        Literal::Int(i) => Value::Int(*i),
        Literal::Float(f) => Value::Float(*f),
        Literal::Bool(b) => Value::Bool(*b),
        Literal::Null => Value::Null,
    }
}

fn compare_values_simple(lhs: &Value, op: &CompOp, rhs: &Value) -> bool {
    match op {
        CompOp::Eq => value_eq_loose_simple(lhs, rhs),
        CompOp::Neq => !value_eq_loose_simple(lhs, rhs),
        CompOp::Lt => lhs.partial_cmp(rhs) == Some(std::cmp::Ordering::Less),
        CompOp::Gt => lhs.partial_cmp(rhs) == Some(std::cmp::Ordering::Greater),
        CompOp::Lte => matches!(
            lhs.partial_cmp(rhs),
            Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
        ),
        CompOp::Gte => matches!(
            lhs.partial_cmp(rhs),
            Some(std::cmp::Ordering::Greater | std::cmp::Ordering::Equal)
        ),
        CompOp::Contains => match (lhs, rhs) {
            (Value::String(haystack), Value::String(needle)) => {
                haystack.to_lowercase().contains(&needle.to_lowercase())
            },
            _ => false,
        },
        CompOp::StartsWith => match (lhs, rhs) {
            (Value::String(haystack), Value::String(prefix)) => {
                haystack.to_lowercase().starts_with(&prefix.to_lowercase())
            },
            _ => false,
        },
    }
}

fn value_eq_loose_simple(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::String(sa), Value::String(sb)) => sa.eq_ignore_ascii_case(sb),
        (Value::Int(ia), Value::Int(ib)) => ia == ib,
        (Value::Float(fa), Value::Float(fb)) => (fa - fb).abs() < f64::EPSILON,
        (Value::Int(i), Value::Float(f)) | (Value::Float(f), Value::Int(i)) => {
            (*i as f64 - f).abs() < f64::EPSILON
        },
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::Null, Value::Null) => true,
        _ => false,
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Check if a boolean expression contains any FuncPredicate nodes.
fn contains_func_predicate(expr: &RBoolExpr) -> bool {
    match expr {
        RBoolExpr::FuncPredicate(_, _) => true,
        RBoolExpr::And(a, b) | RBoolExpr::Or(a, b) => {
            contains_func_predicate(a) || contains_func_predicate(b)
        },
        RBoolExpr::Not(inner) | RBoolExpr::Paren(inner) => contains_func_predicate(inner),
        _ => false,
    }
}

/// Map MinnsQL label strings to NodeType discriminants.
fn label_to_discriminant(label: &str) -> Option<u8> {
    match label.to_lowercase().as_str() {
        "agent" => Some(0),
        "event" => Some(1),
        "context" => Some(2),
        "concept" => Some(3),
        "goal" => Some(4),
        "episode" => Some(5),
        "memory" => Some(6),
        "strategy" => Some(7),
        "tool" => Some(8),
        "result" => Some(9),
        "claim" => Some(10),
        "person" | "location" | "organization" | "thing" => Some(3),
        _ => None,
    }
}
