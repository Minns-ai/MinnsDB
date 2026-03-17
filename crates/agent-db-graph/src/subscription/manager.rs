//! Subscription manager: lifecycle, delta processing, and result delivery.

use rustc_hash::FxHashMap;
use smallvec::SmallVec;
use tokio::sync::broadcast;

use crate::ontology::OntologyRegistry;
use crate::query_lang::executor::Executor;
use crate::query_lang::planner::{ExecutionPlan, PlanStep, SlotIdx};
use crate::query_lang::types::{QueryOutput, Value};
use crate::structures::Graph;

use super::delta::{DeltaBatch, GraphDelta};
use super::diff::{build_cached_output, diff_outputs, CachedOutput};
use super::incremental::{
    AggregationState, BoundEntityId, ExpandState, FilterState,
    GroupedAggregationState, IncrementalPlan, MaintenanceStrategy, RowDelta, RowId, ScanState,
};
use super::varlength::VarLengthExpandState;

pub type SubscriptionId = u64;

/// Update emitted when a subscription's results change.
#[derive(Debug)]
pub struct SubscriptionUpdate {
    pub subscription_id: SubscriptionId,
    pub inserts: Vec<(RowId, Vec<Value>)>,
    pub deletes: Vec<RowId>,
    pub count: Option<i64>,
    pub was_full_rerun: bool,
}

/// Per-subscription state.
pub struct SubscriptionState {
    pub id: SubscriptionId,
    pub incremental_plan: IncrementalPlan,
    pub scan_state: ScanState,
    pub expand_state: Option<ExpandState>,
    pub varlength_state: Option<VarLengthExpandState>,
    pub filter_states: Vec<FilterState>,
    pub aggregation_state: Option<AggregationState>,
    pub grouped_state: Option<GroupedAggregationState>,
    pub cached_output: CachedOutput,
    pub last_generation: u64,
}

/// Manages all active subscriptions.
pub struct SubscriptionManager {
    subscriptions: FxHashMap<SubscriptionId, SubscriptionState>,
    next_id: SubscriptionId,
    delta_rx: broadcast::Receiver<DeltaBatch>,
    /// Per-subscription buffered updates, populated by drain_and_process.
    pending_updates: FxHashMap<SubscriptionId, Vec<SubscriptionUpdate>>,
    /// Maximum number of active subscriptions (0 = unlimited).
    max_subscriptions: usize,
}

impl SubscriptionManager {
    pub fn new(delta_rx: broadcast::Receiver<DeltaBatch>) -> Self {
        Self {
            subscriptions: FxHashMap::default(),
            next_id: 1,
            delta_rx,
            pending_updates: FxHashMap::default(),
            max_subscriptions: 100,
        }
    }

    /// Set the maximum number of allowed subscriptions.
    pub fn set_max_subscriptions(&mut self, max: usize) {
        self.max_subscriptions = max;
    }

    /// Subscribe to a query plan. Returns the subscription ID and initial results.
    pub fn subscribe(
        &mut self,
        plan: ExecutionPlan,
        graph: &Graph,
        ontology: &OntologyRegistry,
    ) -> Result<(SubscriptionId, QueryOutput), crate::query_lang::types::QueryError> {
        if self.max_subscriptions > 0 && self.subscriptions.len() >= self.max_subscriptions {
            return Err(crate::query_lang::types::QueryError::ExecutionError(
                format!(
                    "Subscription limit reached ({} active, max {})",
                    self.subscriptions.len(),
                    self.max_subscriptions
                ),
            ));
        }

        let id = self.next_id;
        self.next_id += 1;

        let incremental_plan = IncrementalPlan::analyze(plan.clone());

        // Execute initial query to get current results.
        let (output, binding_rows) =
            Executor::execute_with_bindings(graph, ontology, plan.clone())?;

        // Build cached output from bindings.
        let cached_output = build_cached_output(&output, &binding_rows);

        // Initialize operator states.
        let ops = init_operator_states(&plan, graph, &binding_rows);

        // Initialize aggregation state if aggregation present.
        let initial_row_ids: Vec<RowId> = binding_rows
            .iter()
            .map(|b| RowId::new(b.clone()))
            .collect();

        let is_incremental = matches!(incremental_plan.strategy, MaintenanceStrategy::Incremental);
        let has_agg = !plan.aggregations.is_empty();
        let has_group_by = !plan.group_by_keys.is_empty();

        let aggregation_state = if is_incremental && has_agg && !has_group_by {
            let agg = &plan.aggregations[0];
            Some(AggregationState::init(
                &agg.function,
                &agg.input_expr,
                &initial_row_ids,
                graph,
            ))
        } else {
            None
        };

        let grouped_state = if is_incremental && has_agg && has_group_by {
            let agg = &plan.aggregations[0];
            // Resolve group-by keys to RExprs. Group-by keys reference projection aliases,
            // so we need to find the corresponding projection expressions.
            let group_exprs: Vec<_> = plan
                .group_by_keys
                .iter()
                .filter_map(|key| {
                    plan.projections
                        .iter()
                        .find(|p| &p.alias == key)
                        .map(|p| p.expr.clone())
                })
                .collect();
            Some(GroupedAggregationState::init(
                group_exprs,
                agg.function.clone(),
                agg.input_expr.clone(),
                &initial_row_ids,
                graph,
            ))
        } else {
            None
        };

        let state = SubscriptionState {
            id,
            incremental_plan,
            scan_state: ops.scan_state,
            expand_state: ops.expand_state,
            varlength_state: ops.varlength_state,
            filter_states: ops.filter_states,
            aggregation_state,
            grouped_state,
            cached_output,
            last_generation: graph.generation,
        };

        self.subscriptions.insert(id, state);

        Ok((id, output))
    }

    /// Remove a subscription and any pending updates.
    pub fn unsubscribe(&mut self, id: SubscriptionId) -> bool {
        self.pending_updates.remove(&id);
        self.subscriptions.remove(&id).is_some()
    }

    /// Process a single delta batch against all subscriptions.
    pub fn process_batch(
        &mut self,
        batch: &DeltaBatch,
        graph: &Graph,
        ontology: &OntologyRegistry,
    ) -> Vec<SubscriptionUpdate> {
        let mut updates = Vec::new();

        // Collect subscription IDs to process (avoid borrow issues).
        let sub_ids: Vec<SubscriptionId> = self.subscriptions.keys().copied().collect();

        for sub_id in sub_ids {
            let sub = match self.subscriptions.get_mut(&sub_id) {
                Some(s) => s,
                None => continue,
            };

            // Fast rejection via trigger set.
            if !sub.incremental_plan.trigger_set.overlaps(batch) {
                sub.last_generation = batch.generation_range.1;
                continue;
            }

            // Check for generation gap → force full rerun.
            let has_gap = batch.generation_range.0 > sub.last_generation + 1;

            let update = match (&sub.incremental_plan.strategy, has_gap) {
                (MaintenanceStrategy::FullRerun { .. }, _) | (_, true) => {
                    process_full_rerun(sub, graph, ontology)
                }
                (MaintenanceStrategy::Incremental, false) => {
                    // Check for NodeMerged → force targeted rerun.
                    // We check unconditionally for any NodeMerged delta because
                    // the absorbed node may have already been removed from operator
                    // state by a preceding NodeRemoved batch in the same drain cycle.
                    let has_merge = batch.deltas.iter().any(|d| {
                        matches!(d, GraphDelta::NodeMerged { .. })
                    });

                    if has_merge {
                        process_full_rerun(sub, graph, ontology)
                    } else {
                        process_incremental(sub, batch, graph)
                    }
                }
            };

            sub.last_generation = batch.generation_range.1;

            if let Some(u) = update {
                updates.push(u);
            }
        }

        updates
    }

    /// Drain all pending deltas from the broadcast channel and process them.
    /// Updates are buffered per-subscription in `pending_updates`.
    /// Returns the total number of updates processed.
    pub fn drain_and_process(
        &mut self,
        graph: &Graph,
        ontology: &OntologyRegistry,
    ) -> usize {
        let mut all_updates = Vec::new();
        let mut lagged = false;

        loop {
            match self.delta_rx.try_recv() {
                Ok(batch) => {
                    let updates = self.process_batch(&batch, graph, ontology);
                    all_updates.extend(updates);
                }
                Err(broadcast::error::TryRecvError::Empty) => break,
                Err(broadcast::error::TryRecvError::Closed) => break,
                Err(broadcast::error::TryRecvError::Lagged(n)) => {
                    tracing::warn!(
                        "Subscription broadcast lagged by {} messages — forcing full rerun for all subscriptions",
                        n
                    );
                    lagged = true;
                }
            }
        }

        if lagged {
            let sub_ids: Vec<SubscriptionId> = self.subscriptions.keys().copied().collect();
            for sub_id in sub_ids {
                if let Some(sub) = self.subscriptions.get_mut(&sub_id) {
                    if let Some(update) = process_full_rerun(sub, graph, ontology) {
                        all_updates.push(update);
                    }
                }
            }
        }

        let count = all_updates.len();

        // Buffer updates per-subscription (move, not clone).
        for update in all_updates {
            self.pending_updates
                .entry(update.subscription_id)
                .or_default()
                .push(update);
        }

        count
    }

    /// Take all pending updates for a specific subscription.
    pub fn take_pending(&mut self, id: SubscriptionId) -> Vec<SubscriptionUpdate> {
        self.pending_updates.remove(&id).unwrap_or_default()
    }

    /// Take all pending updates across all subscriptions. Used for testing.
    pub fn take_all_pending(&mut self) -> Vec<SubscriptionUpdate> {
        let mut all = Vec::new();
        for (_, updates) in self.pending_updates.drain() {
            all.extend(updates);
        }
        all
    }

    /// Get the number of active subscriptions.
    pub fn subscription_count(&self) -> usize {
        self.subscriptions.len()
    }

    /// Get a reference to a subscription's state.
    pub fn get_subscription(&self, id: SubscriptionId) -> Option<&SubscriptionState> {
        self.subscriptions.get(&id)
    }

    /// List all active subscriptions with their ID, cached row count, and strategy.
    pub fn list_subscriptions(&self) -> Vec<(SubscriptionId, usize, String)> {
        self.subscriptions
            .iter()
            .map(|(&id, state)| {
                let strategy = match &state.incremental_plan.strategy {
                    MaintenanceStrategy::Incremental => "incremental".to_string(),
                    MaintenanceStrategy::FullRerun { reason } => {
                        format!("full_rerun: {}", reason)
                    }
                };
                (id, state.cached_output.rows.len(), strategy)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Processing helpers
// ---------------------------------------------------------------------------

fn process_full_rerun(
    sub: &mut SubscriptionState,
    graph: &Graph,
    ontology: &OntologyRegistry,
) -> Option<SubscriptionUpdate> {
    let result = Executor::execute_with_bindings(graph, ontology, sub.incremental_plan.plan.clone());
    let (new_output, new_bindings) = match result {
        Ok(r) => r,
        Err(_) => return None,
    };

    // Build new cached output using structural RowIds from bindings.
    let new_cached = build_cached_output(&new_output, &new_bindings);

    // Diff old vs new.
    let diff = diff_outputs(&sub.cached_output, &new_cached.rows);

    // Rebuild operator states from new bindings.
    let ops = init_operator_states(&sub.incremental_plan.plan, graph, &new_bindings);
    sub.scan_state = ops.scan_state;
    sub.expand_state = ops.expand_state;
    sub.varlength_state = ops.varlength_state;
    sub.filter_states = ops.filter_states;

    // Re-init aggregation state from new bindings on full rerun.
    if sub.aggregation_state.is_some() && !sub.incremental_plan.plan.aggregations.is_empty() {
        let agg = &sub.incremental_plan.plan.aggregations[0];
        let new_row_ids: Vec<RowId> = new_bindings
            .iter()
            .map(|b| RowId::new(b.clone()))
            .collect();
        sub.aggregation_state = Some(AggregationState::init(
            &agg.function,
            &agg.input_expr,
            &new_row_ids,
            graph,
        ));
    }
    let count = sub.aggregation_state.as_ref().map(|agg| {
        match agg.current_value() {
            Value::Int(i) => i,
            Value::Float(f) => f as i64,
            _ => 0,
        }
    });

    sub.cached_output = new_cached;

    if diff.inserts.is_empty() && diff.deletes.is_empty() {
        return None;
    }

    Some(SubscriptionUpdate {
        subscription_id: sub.id,
        inserts: diff.inserts,
        deletes: diff.deletes,
        count,
        was_full_rerun: true,
    })
}

fn process_incremental(
    sub: &mut SubscriptionState,
    batch: &DeltaBatch,
    graph: &Graph,
) -> Option<SubscriptionUpdate> {
    let viewport = &sub.incremental_plan.plan.temporal_viewport;
    let txn_cutoff = sub.incremental_plan.plan.transaction_cutoff;

    // Partition deltas into node and edge categories.
    let mut node_deltas = Vec::new();
    let mut edge_deltas = Vec::new();
    for delta in &batch.deltas {
        match delta {
            GraphDelta::NodeAdded { .. } | GraphDelta::NodeRemoved { .. } => {
                node_deltas.push(delta);
            }
            GraphDelta::EdgeAdded { .. }
            | GraphDelta::EdgeRemoved { .. }
            | GraphDelta::EdgeSuperseded { .. }
            | GraphDelta::EdgeMutated { .. } => {
                edge_deltas.push(delta);
            }
            GraphDelta::NodeMerged { .. } => {
                // Should have been caught above and sent to full rerun.
            }
        }
    }

    // Phase 1: Scan state processes node deltas.
    let mut scan_row_deltas: Vec<RowDelta> = Vec::new();
    for delta in &node_deltas {
        let deltas = sub.scan_state.apply_delta(delta, graph);
        scan_row_deltas.extend(deltas.into_iter());
    }

    // Phase 2: Expand state (single-hop or variable-length) processes upstream + edge deltas.
    let operator_deltas = if let Some(expand_state) = &mut sub.expand_state {
        let mut expand_deltas =
            expand_state.apply_upstream_deltas(&scan_row_deltas, graph, viewport, txn_cutoff);
        for delta in &edge_deltas {
            let ed = expand_state.apply_edge_delta(
                delta,
                &sub.scan_state,
                graph,
                viewport,
                txn_cutoff,
            );
            expand_deltas.extend(ed);
        }
        expand_deltas
    } else if let Some(vl_state) = &mut sub.varlength_state {
        let mut vl_deltas =
            vl_state.apply_upstream_deltas(&scan_row_deltas, graph, viewport, txn_cutoff);
        for delta in &edge_deltas {
            let ed = vl_state.apply_edge_delta(
                delta,
                &sub.scan_state,
                graph,
                viewport,
                txn_cutoff,
            );
            vl_deltas.extend(ed);
        }
        vl_deltas
    } else {
        scan_row_deltas
    };

    // Phase 3: Filter states process operator deltas.
    let mut filtered_deltas = operator_deltas;
    for filter_state in &mut sub.filter_states {
        filtered_deltas = filter_state.apply_deltas(&filtered_deltas, graph);
    }

    if filtered_deltas.is_empty() {
        return None;
    }

    // Phase 4: Project and update cached output.
    let mut inserts = Vec::new();
    let mut deletes = Vec::new();

    for delta in &filtered_deltas {
        match delta {
            RowDelta::Insert { row_id } => {
                let values = project_row(row_id, &sub.incremental_plan.plan, graph);
                sub.cached_output.rows.insert(row_id.clone(), values.clone());
                inserts.push((row_id.clone(), values));
            }
            RowDelta::Delete { row_id } => {
                sub.cached_output.rows.remove(row_id);
                deletes.push(row_id.clone());
            }
        }
    }

    // Update aggregation state (ungrouped or grouped).
    let count = if let Some(grouped) = &mut sub.grouped_state {
        let _group_delta = grouped.apply_deltas(&filtered_deltas, graph);
        // For grouped aggregation, count = number of groups.
        Some(grouped.groups.len() as i64)
    } else if let Some(agg) = &mut sub.aggregation_state {
        let val = agg.apply_deltas(&filtered_deltas, graph);
        Some(match val {
            Value::Int(i) => i,
            Value::Float(f) => f as i64,
            _ => 0,
        })
    } else {
        None
    };

    Some(SubscriptionUpdate {
        subscription_id: sub.id,
        inserts,
        deletes,
        count,
        was_full_rerun: false,
    })
}

/// Project a row's values using the plan's projections.
fn project_row(row_id: &RowId, plan: &ExecutionPlan, graph: &Graph) -> Vec<Value> {
    plan.projections
        .iter()
        .map(|proj| evaluate_projection_expr(&proj.expr, row_id, graph))
        .collect()
}

/// Evaluate a projection expression against a RowId's bindings.
/// Delegates to the incremental expression evaluator.
fn evaluate_projection_expr(expr: &crate::query_lang::planner::RExpr, row_id: &RowId, graph: &Graph) -> Value {
    super::incremental::evaluate_expr_for_row_or_null(expr, row_id, graph)
}

/// Operator states initialized from a plan.
struct OperatorStates {
    scan_state: ScanState,
    expand_state: Option<ExpandState>,
    varlength_state: Option<VarLengthExpandState>,
    filter_states: Vec<FilterState>,
}

/// Initialize operator states from a plan and initial bindings.
fn init_operator_states(
    plan: &ExecutionPlan,
    graph: &Graph,
    _binding_rows: &[SmallVec<[(SlotIdx, BoundEntityId); 4]>],
) -> OperatorStates {
    let mut scan_state = ScanState {
        var: 0,
        label_discs: vec![],
        prop_filters: vec![],
        active_nodes: rustc_hash::FxHashSet::default(),
    };
    let mut expand_state = None;
    let mut varlength_state = None;
    let mut filter_states = Vec::new();

    for step in plan.steps.iter() {
        match step {
            PlanStep::ScanNodes { var, labels, props } => {
                scan_state = ScanState::init(*var, labels, props, graph);
            }
            PlanStep::Expand {
                from_var,
                edge_var,
                to_var,
                edge_type,
                direction,
                range,
            } => {
                if let Some((min_hops, max_hops)) = range {
                    // Variable-length expansion.
                    varlength_state = Some(VarLengthExpandState::init(
                        *from_var,
                        *to_var,
                        edge_type,
                        direction,
                        *min_hops,
                        *max_hops,
                        &scan_state,
                        graph,
                        &plan.temporal_viewport,
                        plan.transaction_cutoff,
                    ));
                } else {
                    // Single-hop expansion.
                    expand_state = Some(ExpandState::init(
                        *from_var,
                        *edge_var,
                        *to_var,
                        edge_type,
                        direction,
                        &scan_state,
                        graph,
                        &plan.temporal_viewport,
                        plan.transaction_cutoff,
                    ));
                }
            }
            PlanStep::Filter(expr) => {
                filter_states.push(FilterState::init(expr.clone()));
            }
        }
    }

    // Populate filter passed sets from initial binding data.
    if !filter_states.is_empty() {
        let mut current_rows: Vec<RowId> = scan_state
            .active_nodes
            .iter()
            .map(|&nid| RowId::new(smallvec::smallvec![(scan_state.var, BoundEntityId::Node(nid))]))
            .collect();

        if let Some(es) = &expand_state {
            let mut expanded = Vec::new();
            for row_id in &current_rows {
                if let Some(BoundEntityId::Node(from_node)) = row_id.get(es.from_var) {
                    if let Some(exps) = es.expansions.get(from_node) {
                        for &(eid, to_node) in exps {
                            expanded.push(es.make_row_id(*from_node, eid, to_node));
                        }
                    }
                }
            }
            current_rows = expanded;
        } else if let Some(vls) = &varlength_state {
            let mut expanded = Vec::new();
            for row_id in &current_rows {
                if let Some(BoundEntityId::Node(from_node)) = row_id.get(vls.from_var) {
                    if let Some(targets) = vls.reachability.get(from_node) {
                        for &target in targets {
                            expanded.push(RowId::new(smallvec::smallvec![
                                (vls.from_var, BoundEntityId::Node(*from_node)),
                                (vls.to_var, BoundEntityId::Node(target)),
                            ]));
                        }
                    }
                }
            }
            current_rows = expanded;
        }

        for filter_state in &mut filter_states {
            let mut passed_rows = Vec::new();
            for row_id in &current_rows {
                if super::incremental::evaluate_filter_for_row(&filter_state.filter, row_id, graph) {
                    filter_state.passed.insert(row_id.clone());
                    passed_rows.push(row_id.clone());
                }
            }
            current_rows = passed_rows;
        }
    }

    OperatorStates {
        scan_state,
        expand_state,
        varlength_state,
        filter_states,
    }
}
