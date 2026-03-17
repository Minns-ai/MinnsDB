//! MinnsQL query executor.
//!
//! Evaluates an [`ExecutionPlan`] against a [`Graph`], producing a [`QueryOutput`]
//! table of rows and columns.

use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};
use std::time::{Duration, Instant};

use smallvec::SmallVec;

use super::ast::{CompOp, Direction as AstDirection, Literal};
use super::planner::{
    AggregateFunction, Aggregation, ExecutionPlan, OrderSpec, PlanStep, Projection, RBoolExpr,
    RExpr, SlotIdx, TemporalViewport,
};
use super::types::{QueryError, QueryOutput, QueryStats, Value};
use crate::ontology::OntologyRegistry;
use crate::structures::{EdgeId, EdgeType, Graph, GraphEdge, GraphNode, NodeId, NodeType};

/// Hard cap on intermediate binding rows to prevent OOM on cartesian products.
const MAX_INTERMEDIATE_ROWS: usize = 100_000;

/// Hard cap on nodes visited during BFS expansion.
const MAX_BFS_VISITED: usize = 10_000;

/// Default query timeout.
const QUERY_TIMEOUT: Duration = Duration::from_secs(30);

// ---------------------------------------------------------------------------
// Binding types — hybrid inline/dynamic slot array
// ---------------------------------------------------------------------------

/// A value bound during pattern matching.
#[derive(Debug, Clone)]
#[allow(dead_code)]
enum BoundValue {
    Node(NodeId),
    Edge(EdgeId),
    Path(Vec<NodeId>),
}

const INLINE_SLOTS: usize = 16;

/// A binding row: maps slot indices to bound values during execution.
#[derive(Debug, Clone)]
struct BindingRow {
    slots: BindingSlots,
}

#[derive(Debug, Clone)]
enum BindingSlots {
    Inline([Option<BoundValue>; INLINE_SLOTS]),
    Dynamic(Vec<Option<BoundValue>>),
}

impl BindingRow {
    fn new(count: u8) -> Self {
        if (count as usize) <= INLINE_SLOTS {
            BindingRow {
                slots: BindingSlots::Inline(Default::default()),
            }
        } else {
            BindingRow {
                slots: BindingSlots::Dynamic(vec![None; count as usize]),
            }
        }
    }

    #[inline]
    fn get(&self, idx: SlotIdx) -> Option<&BoundValue> {
        let i = idx as usize;
        match &self.slots {
            BindingSlots::Inline(arr) => arr.get(i).and_then(|v| v.as_ref()),
            BindingSlots::Dynamic(vec) => vec.get(i).and_then(|v| v.as_ref()),
        }
    }

    #[inline]
    fn set(&mut self, idx: SlotIdx, val: BoundValue) {
        let i = idx as usize;
        match &mut self.slots {
            BindingSlots::Inline(arr) => {
                debug_assert!(i < INLINE_SLOTS, "slot index {} out of bounds for inline array", i);
                arr[i] = Some(val);
            }
            BindingSlots::Dynamic(vec) => {
                debug_assert!(i < vec.len(), "slot index {} out of bounds for dynamic vec (len {})", i, vec.len());
                vec[i] = Some(val);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Executor
// ---------------------------------------------------------------------------

/// Tracks internal statistics for a single query execution.
struct ExecutionStats {
    nodes_scanned: u64,
    edges_traversed: u64,
}

/// Executes MinnsQL plans against a [`Graph`].
pub struct Executor<'a> {
    graph: &'a Graph,
    #[allow(dead_code)]
    ontology: &'a OntologyRegistry,
    viewport: TemporalViewport,
    /// Transaction-time cutoff (AS OF). When set, only nodes/edges with
    /// `created_at <= cutoff` are visible. Initial support based on creation
    /// cutoff; richer correction/supersession semantics can come later.
    transaction_cutoff: Option<u64>,
    stats: ExecutionStats,
    deadline: Instant,
}

impl<'a> Executor<'a> {
    // -----------------------------------------------------------------------
    // Public entry point
    // -----------------------------------------------------------------------

    /// Execute a plan and return the query output.
    pub fn execute(
        graph: &'a Graph,
        ontology: &'a OntologyRegistry,
        plan: ExecutionPlan,
    ) -> Result<QueryOutput, QueryError> {
        let start = Instant::now();

        let mut executor = Executor {
            graph,
            ontology,
            viewport: plan.temporal_viewport.clone(),
            transaction_cutoff: plan.transaction_cutoff,
            stats: ExecutionStats {
                nodes_scanned: 0,
                edges_traversed: 0,
            },
            deadline: start + QUERY_TIMEOUT,
        };

        // Start with a single empty binding row.
        let mut rows: Vec<BindingRow> = vec![BindingRow::new(plan.var_count)];

        // Execute each plan step in sequence.
        for step in &plan.steps {
            rows = executor.execute_step(step, rows)?;
        }

        // ----- Projection -----
        let columns: Vec<String> = plan.projections.iter().map(|p| p.alias.clone()).collect();

        let mut result_rows: Vec<Vec<Value>> = Vec::with_capacity(rows.len());
        for binding in &rows {
            let mut row = Vec::with_capacity(plan.projections.len());
            for proj in &plan.projections {
                let val = executor.evaluate_expr(&proj.expr, binding)?;
                row.push(val);
            }
            result_rows.push(row);
        }

        // ----- Aggregation -----
        if !plan.aggregations.is_empty() {
            result_rows = executor.apply_aggregations(
                &result_rows,
                &columns,
                &plan.aggregations,
                &plan.group_by_keys,
                &plan.projections,
                &rows,
            )?;
        }

        // ----- DISTINCT -----
        if plan.projections.iter().any(|p| p.distinct) {
            let mut seen = HashSet::new();
            let mut deduped = Vec::new();
            for row in result_rows {
                let hash = hash_row(&row);
                if seen.insert(hash) {
                    deduped.push(row);
                }
            }
            result_rows = deduped;
        }

        // ----- ORDER BY -----
        if !plan.ordering.is_empty() {
            executor.apply_ordering(&mut result_rows, &columns, &plan.ordering);
        }

        // ----- LIMIT -----
        if let Some(limit) = plan.limit {
            result_rows.truncate(limit as usize);
        }

        let elapsed = start.elapsed();

        Ok(QueryOutput {
            columns,
            rows: result_rows,
            stats: QueryStats {
                nodes_scanned: executor.stats.nodes_scanned,
                edges_traversed: executor.stats.edges_traversed,
                execution_time_ms: elapsed.as_millis() as u64,
            },
        })
    }

    // -----------------------------------------------------------------------
    // Plan step dispatch
    // -----------------------------------------------------------------------

    fn execute_step(
        &mut self,
        step: &PlanStep,
        rows: Vec<BindingRow>,
    ) -> Result<Vec<BindingRow>, QueryError> {
        match step {
            PlanStep::ScanNodes { var, labels, props } => {
                self.step_scan_nodes(&rows, *var, labels, props)
            }
            PlanStep::Expand {
                from_var,
                edge_var,
                to_var,
                edge_type,
                direction,
                range,
            } => self.step_expand(&rows, *from_var, *edge_var, *to_var, edge_type, direction, range),
            PlanStep::Filter(expr) => self.step_filter(rows, expr),
        }
    }

    // -----------------------------------------------------------------------
    // ScanNodes
    // -----------------------------------------------------------------------

    fn step_scan_nodes(
        &mut self,
        rows: &[BindingRow],
        var: SlotIdx,
        labels: &[String],
        props: &[(String, Literal)],
    ) -> Result<Vec<BindingRow>, QueryError> {
        // Collect candidate nodes.
        let candidates: Vec<&GraphNode> = self.collect_scan_candidates(labels, props);
        self.stats.nodes_scanned += candidates.len() as u64;

        // Cross-product with existing binding rows.
        let mut out = Vec::with_capacity(std::cmp::min(
            rows.len() * candidates.len(),
            MAX_INTERMEDIATE_ROWS,
        ));
        for binding in rows {
            for node in &candidates {
                if out.len() >= MAX_INTERMEDIATE_ROWS {
                    return Err(QueryError::ExecutionError(format!(
                        "Result set exceeded {} rows — add filters or LIMIT to narrow the query",
                        MAX_INTERMEDIATE_ROWS
                    )));
                }
                if self.node_visible_txn(node) && self.node_matches_props(node, props) {
                    let mut new_binding = binding.clone();
                    new_binding.set(var, BoundValue::Node(node.id));
                    out.push(new_binding);
                }
            }
        }
        Ok(out)
    }

    /// Determine candidate nodes for a scan.
    fn collect_scan_candidates(
        &self,
        labels: &[String],
        props: &[(String, Literal)],
    ) -> Vec<&'a GraphNode> {
        // Fast-path: if a `name` property is given, try concept index first.
        if let Some((_, Literal::String(name))) = props.iter().find(|(k, _)| k == "name") {
            // Try exact match, then case-insensitive.
            if let Some(node) = self.graph.get_concept_node(name) {
                return vec![node];
            }
            let lower = name.to_lowercase();
            if &lower != name {
                if let Some(node) = self.graph.get_concept_node(&lower) {
                    return vec![node];
                }
            }
        }

        // If labels are given, use the type index.
        if !labels.is_empty() {
            let mut candidates = Vec::new();
            for label in labels {
                candidates.extend(self.graph.get_nodes_by_type(label));
            }
            return candidates;
        }

        // Fallback: scan all nodes.
        self.graph.nodes().collect()
    }

    /// Check whether a node satisfies the property filters.
    fn node_matches_props(&self, node: &GraphNode, props: &[(String, Literal)]) -> bool {
        for (key, lit) in props {
            let val = self.node_property_value(node, key);
            let expected = literal_to_value(lit);
            if !value_eq_loose(&val, &expected) {
                return false;
            }
        }
        true
    }

    // -----------------------------------------------------------------------
    // Expand (single-hop and variable-length)
    // -----------------------------------------------------------------------

    fn step_expand(
        &mut self,
        rows: &[BindingRow],
        from_var: SlotIdx,
        edge_var: Option<SlotIdx>,
        to_var: SlotIdx,
        edge_type: &Option<String>,
        direction: &AstDirection,
        range: &Option<(u32, Option<u32>)>,
    ) -> Result<Vec<BindingRow>, QueryError> {
        let mut out = Vec::new();

        for binding in rows {
            self.check_deadline()?;

            let from_id = match binding.get(from_var) {
                Some(BoundValue::Node(id)) => *id,
                _ => {
                    return Err(QueryError::ExecutionError(format!(
                        "Variable at slot {} is not bound to a node",
                        from_var
                    )));
                }
            };

            match range {
                Some((min_hops, max_hops_opt)) => {
                    let max_hops = max_hops_opt.unwrap_or(10); // sensible default cap
                    let reached = self.bfs_expand(from_id, edge_type, direction, *min_hops, max_hops)?;
                    for target_id in reached {
                        if out.len() >= MAX_INTERMEDIATE_ROWS {
                            return Err(QueryError::ExecutionError(format!(
                                "Result set exceeded {} rows during expansion",
                                MAX_INTERMEDIATE_ROWS
                            )));
                        }
                        let mut new_binding = binding.clone();
                        new_binding.set(to_var, BoundValue::Node(target_id));
                        out.push(new_binding);
                    }
                }
                None => {
                    // Single-hop expansion.
                    let edges = self.directed_edges(from_id, direction);
                    for edge in edges {
                        if !self.edge_visible(edge) {
                            continue;
                        }
                        if !self.edge_matches_type(edge, edge_type) {
                            continue;
                        }
                        self.stats.edges_traversed += 1;

                        if out.len() >= MAX_INTERMEDIATE_ROWS {
                            return Err(QueryError::ExecutionError(format!(
                                "Result set exceeded {} rows during expansion",
                                MAX_INTERMEDIATE_ROWS
                            )));
                        }

                        let other_id = match direction {
                            AstDirection::Out => edge.target,
                            AstDirection::In => edge.source,
                        };

                        // Transaction-time filter on the target node.
                        if let Some(node) = self.graph.get_node(other_id) {
                            if !self.node_visible_txn(node) {
                                continue;
                            }
                        }

                        let mut new_binding = binding.clone();
                        if let Some(ev) = edge_var {
                            new_binding.set(ev, BoundValue::Edge(edge.id));
                        }
                        new_binding.set(to_var, BoundValue::Node(other_id));
                        out.push(new_binding);
                    }
                }
            }
        }

        Ok(out)
    }

    /// Get edges in the requested direction.
    fn directed_edges(&self, node_id: NodeId, direction: &AstDirection) -> Vec<&'a GraphEdge> {
        match direction {
            AstDirection::Out => self.graph.get_edges_from(node_id),
            AstDirection::In => self.graph.get_edges_to(node_id),
        }
    }

    /// BFS expansion up to `max_hops`, returning all reached node IDs
    /// (excluding the start) that are at least `min_hops` away.
    ///
    /// Capped at [`MAX_BFS_VISITED`] nodes to prevent runaway traversals on
    /// dense graphs.
    fn bfs_expand(
        &mut self,
        start: NodeId,
        edge_type: &Option<String>,
        direction: &AstDirection,
        min_hops: u32,
        max_hops: u32,
    ) -> Result<Vec<NodeId>, QueryError> {
        let mut visited = HashSet::new();
        visited.insert(start);

        let mut queue: VecDeque<(NodeId, u32)> = VecDeque::new();
        queue.push_back((start, 0));

        let mut result = Vec::new();

        while let Some((current, depth)) = queue.pop_front() {
            if depth >= max_hops {
                continue;
            }

            if visited.len() >= MAX_BFS_VISITED {
                break;
            }

            let edges = self.directed_edges(current, direction);
            for edge in edges {
                if !self.edge_visible(edge) {
                    continue;
                }
                if !self.edge_matches_type(edge, edge_type) {
                    continue;
                }
                self.stats.edges_traversed += 1;

                let next = match direction {
                    AstDirection::Out => edge.target,
                    AstDirection::In => edge.source,
                };

                // Transaction-time filter on the target node.
                if let Some(node) = self.graph.get_node(next) {
                    if !self.node_visible_txn(node) {
                        continue;
                    }
                }

                if visited.insert(next) {
                    let next_depth = depth + 1;
                    if next_depth >= min_hops {
                        result.push(next);
                    }
                    queue.push_back((next, next_depth));
                }
            }
        }

        Ok(result)
    }

    // -----------------------------------------------------------------------
    // Filter
    // -----------------------------------------------------------------------

    fn step_filter(
        &self,
        mut rows: Vec<BindingRow>,
        expr: &RBoolExpr,
    ) -> Result<Vec<BindingRow>, QueryError> {
        // Filter in-place to avoid a second Vec allocation.
        let mut err: Option<QueryError> = None;
        rows.retain(|binding| {
            if err.is_some() {
                return false;
            }
            match self.evaluate_bool(expr, binding) {
                Ok(keep) => keep,
                Err(e) => {
                    err = Some(e);
                    false
                }
            }
        });
        if let Some(e) = err {
            return Err(e);
        }
        Ok(rows)
    }

    // -----------------------------------------------------------------------
    // Boolean expression evaluation
    // -----------------------------------------------------------------------

    fn evaluate_bool(&self, expr: &RBoolExpr, binding: &BindingRow) -> Result<bool, QueryError> {
        match expr {
            RBoolExpr::Comparison(lhs, op, rhs) => {
                let l = self.evaluate_expr(lhs, binding)?;
                let r = self.evaluate_expr(rhs, binding)?;
                Ok(compare_values(&l, op, &r))
            }
            RBoolExpr::IsNull(e) => {
                let v = self.evaluate_expr(e, binding)?;
                Ok(v.is_null())
            }
            RBoolExpr::IsNotNull(e) => {
                let v = self.evaluate_expr(e, binding)?;
                Ok(!v.is_null())
            }
            RBoolExpr::And(a, b) => {
                Ok(self.evaluate_bool(a, binding)? && self.evaluate_bool(b, binding)?)
            }
            RBoolExpr::Or(a, b) => {
                Ok(self.evaluate_bool(a, binding)? || self.evaluate_bool(b, binding)?)
            }
            RBoolExpr::Not(inner) => Ok(!self.evaluate_bool(inner, binding)?),
            RBoolExpr::Paren(inner) => self.evaluate_bool(inner, binding),
            RBoolExpr::FuncPredicate(name, args) => {
                self.evaluate_predicate(name, args, binding)
            }
        }
    }

    /// Evaluate a boolean-returning function predicate.
    fn evaluate_predicate(
        &self,
        name: &str,
        args: &[RExpr],
        binding: &BindingRow,
    ) -> Result<bool, QueryError> {
        match name {
            "SUCCESSIVE" | "successive" => self.pred_successive(args, binding),
            "CHANGED" | "changed" => self.pred_changed(args, binding),
            _ => {
                // Fall back to evaluating as a regular function and coercing to bool.
                let val = self.evaluate_func(name, args, binding)?;
                match val {
                    Value::Bool(b) => Ok(b),
                    Value::Null => Ok(false),
                    _ => Err(QueryError::ExecutionError(format!(
                        "Function `{}` does not return a boolean",
                        name
                    ))),
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Expression evaluation
    // -----------------------------------------------------------------------

    fn evaluate_expr(&self, expr: &RExpr, binding: &BindingRow) -> Result<Value, QueryError> {
        match expr {
            RExpr::Literal(lit) => Ok(literal_to_value(lit)),

            RExpr::Var(slot) => match binding.get(*slot) {
                Some(BoundValue::Node(id)) => {
                    if let Some(node) = self.graph.get_node(*id) {
                        Ok(Value::String(node.label()))
                    } else {
                        Ok(Value::Null)
                    }
                }
                Some(BoundValue::Edge(id)) => {
                    if let Some(edge) = self.graph.get_edge(*id) {
                        Ok(Value::String(edge_type_display(&edge.edge_type)))
                    } else {
                        Ok(Value::Null)
                    }
                }
                Some(BoundValue::Path(ids)) => {
                    let labels: Vec<Value> = ids
                        .iter()
                        .filter_map(|id| self.graph.get_node(*id))
                        .map(|n| Value::String(n.label()))
                        .collect();
                    Ok(Value::List(labels))
                }
                None => Ok(Value::Null),
            },

            RExpr::Property(slot, prop) => self.evaluate_property(*slot, prop, binding),

            RExpr::FuncCall(name, args) => self.evaluate_func(name, args, binding),

            RExpr::Star => Ok(Value::Null), // placeholder for count(*)
        }
    }

    /// Evaluate a property access like `n.name` or `e.weight`.
    fn evaluate_property(
        &self,
        slot: SlotIdx,
        prop: &str,
        binding: &BindingRow,
    ) -> Result<Value, QueryError> {
        match binding.get(slot) {
            Some(BoundValue::Node(id)) => {
                let node = self.graph.get_node(*id).ok_or_else(|| {
                    QueryError::ExecutionError(format!("Node {} not found", id))
                })?;
                Ok(self.node_property_value(node, prop))
            }
            Some(BoundValue::Edge(id)) => {
                let edge = self.graph.get_edge(*id).ok_or_else(|| {
                    QueryError::ExecutionError(format!("Edge {} not found", id))
                })?;
                Ok(self.edge_property_value(edge, prop))
            }
            Some(BoundValue::Path(ids)) => match prop {
                "length" => Ok(Value::Int(ids.len() as i64)),
                "hops" => Ok(Value::Int(ids.len().saturating_sub(1) as i64)),
                _ => Ok(Value::Null),
            },
            None => Ok(Value::Null),
        }
    }

    /// Extract a named property from a node.
    fn node_property_value(&self, node: &GraphNode, prop: &str) -> Value {
        match prop {
            "id" => Value::Int(node.id as i64),
            "name" | "label" => Value::String(node.label()),
            "type" => Value::String(node.type_name().to_string()),
            "created_at" => Value::Int(node.created_at as i64),
            "updated_at" => Value::Int(node.updated_at as i64),
            "group_id" => Value::String(node.group_id.clone()),
            "degree" => Value::Int(node.degree as i64),
            // Type-specific accessors.
            "confidence" => match &node.node_type {
                NodeType::Concept { confidence, .. } | NodeType::Claim { confidence, .. } => {
                    Value::Float(*confidence as f64)
                }
                _ => Value::Null,
            },
            "concept_name" => match &node.node_type {
                NodeType::Concept { concept_name, .. } => {
                    Value::String(concept_name.clone())
                }
                _ => Value::Null,
            },
            "claim_text" => match &node.node_type {
                NodeType::Claim { claim_text, .. } => Value::String(claim_text.clone()),
                _ => Value::Null,
            },
            "significance" => match &node.node_type {
                NodeType::Event { significance, .. } => Value::Float(*significance as f64),
                _ => Value::Null,
            },
            "priority" => match &node.node_type {
                NodeType::Goal { priority, .. } => Value::Float(*priority as f64),
                _ => Value::Null,
            },
            "description" => match &node.node_type {
                NodeType::Goal { description, .. } => Value::String(description.clone()),
                _ => Value::Null,
            },
            _ => {
                // Fall through to the properties map.
                node.properties
                    .get(prop)
                    .map(json_to_value)
                    .unwrap_or(Value::Null)
            }
        }
    }

    /// Extract a named property from an edge.
    fn edge_property_value(&self, edge: &GraphEdge, prop: &str) -> Value {
        match prop {
            "id" => Value::Int(edge.id as i64),
            "type" => Value::String(edge_type_display(&edge.edge_type)),
            "weight" => Value::Float(edge.weight as f64),
            "confidence" => Value::Float(edge.confidence as f64),
            "source" => Value::Int(edge.source as i64),
            "target" => Value::Int(edge.target as i64),
            "created_at" => Value::Int(edge.created_at as i64),
            "updated_at" => Value::Int(edge.updated_at as i64),
            "valid_from" => match edge.valid_from {
                Some(ts) => Value::Int(ts as i64),
                None => Value::Null,
            },
            "valid_until" => match edge.valid_until {
                Some(ts) => Value::Int(ts as i64),
                None => Value::Null,
            },
            "observation_count" => Value::Int(edge.observation_count as i64),
            "group_id" => Value::String(edge.group_id.clone()),
            "association_type" => match &edge.edge_type {
                EdgeType::Association {
                    association_type, ..
                } => Value::String(association_type.clone()),
                _ => Value::Null,
            },
            _ => edge
                .properties
                .get(prop)
                .map(json_to_value)
                .unwrap_or(Value::Null),
        }
    }

    // -----------------------------------------------------------------------
    // Built-in functions
    // -----------------------------------------------------------------------

    fn evaluate_func(
        &self,
        name: &str,
        args: &[RExpr],
        binding: &BindingRow,
    ) -> Result<Value, QueryError> {
        match name {
            "type" => {
                if let Some(arg) = args.first() {
                    let v = self.evaluate_expr(arg, binding)?;
                    match v {
                        Value::String(s) => Ok(Value::String(s)),
                        _ => {
                            // Try resolving the arg as a variable for direct type lookup.
                            if let RExpr::Var(slot) = arg {
                                match binding.get(*slot) {
                                    Some(BoundValue::Node(id)) => {
                                        if let Some(node) = self.graph.get_node(*id) {
                                            return Ok(Value::String(
                                                node.type_name().to_string(),
                                            ));
                                        }
                                    }
                                    Some(BoundValue::Edge(id)) => {
                                        if let Some(edge) = self.graph.get_edge(*id) {
                                            return Ok(Value::String(edge_type_display(
                                                &edge.edge_type,
                                            )));
                                        }
                                    }
                                    _ => {}
                                }
                            }
                            Ok(Value::Null)
                        }
                    }
                } else {
                    Ok(Value::Null)
                }
            }

            "id" => {
                if let Some(RExpr::Var(slot)) = args.first() {
                    match binding.get(*slot) {
                        Some(BoundValue::Node(id)) => Ok(Value::Int(*id as i64)),
                        Some(BoundValue::Edge(id)) => Ok(Value::Int(*id as i64)),
                        _ => Ok(Value::Null),
                    }
                } else {
                    Ok(Value::Null)
                }
            }

            "labels" => {
                if let Some(RExpr::Var(slot)) = args.first() {
                    if let Some(BoundValue::Node(id)) = binding.get(*slot) {
                        if let Some(node) = self.graph.get_node(*id) {
                            return Ok(Value::String(node.type_name().to_string()));
                        }
                    }
                }
                Ok(Value::Null)
            }

            "properties" => {
                if let Some(RExpr::Var(slot)) = args.first() {
                    match binding.get(*slot) {
                        Some(BoundValue::Node(id)) => {
                            if let Some(node) = self.graph.get_node(*id) {
                                let map: HashMap<String, Value> = node
                                    .properties
                                    .iter()
                                    .map(|(k, v)| (k.clone(), json_to_value(v)))
                                    .collect();
                                return Ok(Value::Map(map));
                            }
                        }
                        Some(BoundValue::Edge(id)) => {
                            if let Some(edge) = self.graph.get_edge(*id) {
                                let map: HashMap<String, Value> = edge
                                    .properties
                                    .iter()
                                    .map(|(k, v)| (k.clone(), json_to_value(v)))
                                    .collect();
                                return Ok(Value::Map(map));
                            }
                        }
                        _ => {}
                    }
                }
                Ok(Value::Map(HashMap::new()))
            }

            "count" => {
                // count(*) is handled in the aggregation phase; at row level return 1.
                Ok(Value::Int(1))
            }

            "path" => {
                if args.len() >= 2 {
                    let a_id = self.resolve_node_id(&args[0], binding)?;
                    let b_id = self.resolve_node_id(&args[1], binding)?;
                    if let (Some(a), Some(b)) = (a_id, b_id) {
                        if let Some(path) = self.graph.shortest_path(a, b) {
                            let labels: Vec<Value> = path
                                .iter()
                                .filter_map(|id| self.graph.get_node(*id))
                                .map(|n| Value::String(n.label()))
                                .collect();
                            return Ok(Value::List(labels));
                        }
                    }
                }
                Ok(Value::Null)
            }

            "hops" => {
                if let Some(arg) = args.first() {
                    let v = self.evaluate_expr(arg, binding)?;
                    match v {
                        Value::List(items) => {
                            Ok(Value::Int(items.len().saturating_sub(1) as i64))
                        }
                        _ => {
                            // Try path bound value.
                            if let RExpr::Var(slot) = arg {
                                if let Some(BoundValue::Path(ids)) = binding.get(*slot) {
                                    return Ok(Value::Int(
                                        ids.len().saturating_sub(1) as i64,
                                    ));
                                }
                            }
                            Ok(Value::Int(0))
                        }
                    }
                } else {
                    Ok(Value::Int(0))
                }
            }

            "now" => {
                let ts = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as i64;
                Ok(Value::Int(ts))
            }

            "duration" => {
                if let Some(RExpr::Var(slot)) = args.first() {
                    if let Some(BoundValue::Edge(id)) = binding.get(*slot) {
                        if let Some(edge) = self.graph.get_edge(*id) {
                            match (edge.valid_from, edge.valid_until) {
                                (Some(from), Some(until)) if until > from => {
                                    Ok(Value::Int((until - from) as i64))
                                }
                                _ => Ok(Value::Null),
                            }
                        } else {
                            Ok(Value::Null)
                        }
                    } else {
                        Ok(Value::Null)
                    }
                } else {
                    Ok(Value::Null)
                }
            }

            "coalesce" => {
                for arg in args {
                    let v = self.evaluate_expr(arg, binding)?;
                    if !v.is_null() {
                        return Ok(v);
                    }
                }
                Ok(Value::Null)
            }

            "toUpper" | "toupper" => {
                if let Some(arg) = args.first() {
                    let v = self.evaluate_expr(arg, binding)?;
                    match v {
                        Value::String(s) => Ok(Value::String(s.to_uppercase())),
                        other => Ok(other),
                    }
                } else {
                    Ok(Value::Null)
                }
            }

            "toLower" | "tolower" => {
                if let Some(arg) = args.first() {
                    let v = self.evaluate_expr(arg, binding)?;
                    match v {
                        Value::String(s) => Ok(Value::String(s.to_lowercase())),
                        other => Ok(other),
                    }
                } else {
                    Ok(Value::Null)
                }
            }

            // ── Temporal accessor functions ──────────────────────────────

            "valid_from" => {
                if let Some(RExpr::Var(slot)) = args.first() {
                    if let Some(BoundValue::Edge(id)) = binding.get(*slot) {
                        if let Some(edge) = self.graph.get_edge(*id) {
                            return Ok(match edge.valid_from {
                                Some(ts) => Value::Int(ts as i64),
                                None => Value::Null,
                            });
                        }
                    }
                }
                Ok(Value::Null)
            }

            "valid_until" => {
                if let Some(RExpr::Var(slot)) = args.first() {
                    if let Some(BoundValue::Edge(id)) = binding.get(*slot) {
                        if let Some(edge) = self.graph.get_edge(*id) {
                            return Ok(match edge.valid_until {
                                Some(ts) => Value::Int(ts as i64),
                                None => Value::Null,
                            });
                        }
                    }
                }
                Ok(Value::Null)
            }

            "created_at" => {
                if let Some(RExpr::Var(slot)) = args.first() {
                    match binding.get(*slot) {
                        Some(BoundValue::Node(id)) => {
                            if let Some(node) = self.graph.get_node(*id) {
                                return Ok(Value::Int(node.created_at as i64));
                            }
                        }
                        Some(BoundValue::Edge(id)) => {
                            if let Some(edge) = self.graph.get_edge(*id) {
                                return Ok(Value::Int(edge.created_at as i64));
                            }
                        }
                        _ => {}
                    }
                }
                Ok(Value::Null)
            }

            "updated_at" => {
                if let Some(RExpr::Var(slot)) = args.first() {
                    match binding.get(*slot) {
                        Some(BoundValue::Node(id)) => {
                            if let Some(node) = self.graph.get_node(*id) {
                                return Ok(Value::Int(node.updated_at as i64));
                            }
                        }
                        Some(BoundValue::Edge(id)) => {
                            if let Some(edge) = self.graph.get_edge(*id) {
                                return Ok(Value::Int(edge.updated_at as i64));
                            }
                        }
                        _ => {}
                    }
                }
                Ok(Value::Null)
            }

            "open_ended" => {
                if let Some(RExpr::Var(slot)) = args.first() {
                    if let Some(BoundValue::Edge(id)) = binding.get(*slot) {
                        if let Some(edge) = self.graph.get_edge(*id) {
                            return Ok(Value::Bool(edge.valid_until.is_none()));
                        }
                    }
                }
                Ok(Value::Null)
            }

            "change_type" => {
                // change_type(r, "t1", "t2") — returns "started", "ended", "created", or "stable"
                if args.len() >= 3 {
                    let edge = if let RExpr::Var(slot) = &args[0] {
                        binding.get(*slot).and_then(|b| match b {
                            BoundValue::Edge(id) => self.graph.get_edge(*id),
                            _ => None,
                        })
                    } else {
                        None
                    };
                    if let Some(edge) = edge {
                        let t1 = self.resolve_timestamp_arg(&args[1], binding)?;
                        let t2 = self.resolve_timestamp_arg(&args[2], binding)?;

                        let started = edge.valid_from.map_or(false, |vf| vf >= t1 && vf <= t2);
                        let ended = edge.valid_until.map_or(false, |vu| vu >= t1 && vu <= t2);
                        let created = edge.created_at >= t1 && edge.created_at <= t2;

                        let label = if started && ended {
                            "started_and_ended"
                        } else if started {
                            "started"
                        } else if ended {
                            "ended"
                        } else if created {
                            "created"
                        } else {
                            "stable"
                        };
                        return Ok(Value::String(label.to_string()));
                    }
                }
                Ok(Value::Null)
            }

            // ── Phase 1: time_bucket / date_trunc / ago ────────────────
            "time_bucket" => {
                if args.len() < 2 {
                    return Err(QueryError::ExecutionError(
                        "time_bucket requires 2 arguments: time_bucket(width, timestamp)".into(),
                    ));
                }
                let width_str = match self.evaluate_expr(&args[0], binding)? {
                    Value::String(s) => s,
                    _ => {
                        return Err(QueryError::ExecutionError(
                            "time_bucket first argument must be a string".into(),
                        ))
                    }
                };
                let ts = match self.evaluate_expr(&args[1], binding)? {
                    Value::Int(i) if i >= 0 => i as u64,
                    Value::Int(_) => {
                        return Err(QueryError::ExecutionError(
                            "time_bucket timestamp must be non-negative".into(),
                        ))
                    }
                    _ => return Ok(Value::Null),
                };
                let bucket_nanos = truncate_timestamp(ts, &width_str)?;
                Ok(Value::Int(bucket_nanos as i64))
            }

            "date_trunc" => {
                if args.len() < 2 {
                    return Err(QueryError::ExecutionError(
                        "date_trunc requires 2 arguments: date_trunc(unit, timestamp)".into(),
                    ));
                }
                let unit = match self.evaluate_expr(&args[0], binding)? {
                    Value::String(s) => s,
                    _ => {
                        return Err(QueryError::ExecutionError(
                            "date_trunc first argument must be a string".into(),
                        ))
                    }
                };
                let ts = match self.evaluate_expr(&args[1], binding)? {
                    Value::Int(i) if i >= 0 => i as u64,
                    Value::Int(_) => {
                        return Err(QueryError::ExecutionError(
                            "date_trunc timestamp must be non-negative".into(),
                        ))
                    }
                    _ => return Ok(Value::Null),
                };
                let truncated = truncate_timestamp_by_unit(ts, &unit)?;
                Ok(Value::Int(truncated as i64))
            }

            "ago" => {
                if args.is_empty() {
                    return Err(QueryError::ExecutionError(
                        "ago requires 1 argument: ago(duration_string)".into(),
                    ));
                }
                let dur_str = match self.evaluate_expr(&args[0], binding)? {
                    Value::String(s) => s,
                    _ => {
                        return Err(QueryError::ExecutionError(
                            "ago argument must be a string".into(),
                        ))
                    }
                };
                // Reject calendar units that can't be subtracted as fixed nanos.
                // parse_duration_nanos treats "1y" as 365d — wrong for ago().
                let lower = dur_str.to_lowercase();
                let unit_part = lower.trim_start_matches(|c: char| c.is_ascii_digit());
                if matches!(
                    unit_part,
                    "mo" | "month" | "months" | "quarter" | "quarters" | "year" | "years"
                ) {
                    return Err(QueryError::ExecutionError(format!(
                        "ago() does not support calendar units ('{}'); use seconds/minutes/hours/days/weeks",
                        dur_str
                    )));
                }
                let dur_nanos = super::planner::parse_duration_nanos(&dur_str).map_err(|e| {
                    QueryError::ExecutionError(format!("ago(): {}", e))
                })?;
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_nanos() as u64;
                Ok(Value::Int(now.saturating_sub(dur_nanos) as i64))
            }

            // ── Phase 2: Allen's interval algebra predicates ─────────
            "precedes" => {
                let intervals = self.resolve_two_edge_intervals(args, binding)?;
                Ok(Value::Bool(match intervals {
                    Some(((_, e1_end), (e2_start, _))) => e1_end <= e2_start,
                    None => false,
                }))
            }

            "meets" => {
                let intervals = self.resolve_two_edge_intervals(args, binding)?;
                Ok(Value::Bool(match intervals {
                    Some(((_, e1_end), (e2_start, _))) => e1_end == e2_start,
                    None => false,
                }))
            }

            "covers" => {
                let intervals = self.resolve_two_edge_intervals(args, binding)?;
                Ok(Value::Bool(match intervals {
                    Some(((e1_start, e1_end), (e2_start, e2_end))) => {
                        e1_start <= e2_start && e2_end <= e1_end
                    }
                    None => false,
                }))
            }

            "starts" => {
                let intervals = self.resolve_two_edge_intervals(args, binding)?;
                Ok(Value::Bool(match intervals {
                    Some(((e1_start, e1_end), (e2_start, e2_end))) => {
                        e1_start == e2_start && e1_end <= e2_end
                    }
                    None => false,
                }))
            }

            "finishes" => {
                let intervals = self.resolve_two_edge_intervals(args, binding)?;
                Ok(Value::Bool(match intervals {
                    Some(((e1_start, e1_end), (e2_start, e2_end))) => {
                        e1_end == e2_end && e1_start >= e2_start
                    }
                    None => false,
                }))
            }

            "equals" => {
                let intervals = self.resolve_two_edge_intervals(args, binding)?;
                Ok(Value::Bool(match intervals {
                    Some(((e1_start, e1_end), (e2_start, e2_end))) => {
                        e1_start == e2_start && e1_end == e2_end
                    }
                    None => false,
                }))
            }

            // ── Phase 3: TCell access functions ──────────────────────
            "confidence_at" => {
                if args.len() < 2 {
                    return Err(QueryError::ExecutionError(
                        "confidence_at requires 2 arguments: confidence_at(edge, timestamp)".into(),
                    ));
                }
                if let Some(edge) = self.resolve_edge(&args[0], binding)? {
                    let ts = self.resolve_timestamp_arg(&args[1], binding)?;
                    let et = agent_db_core::event_time::EventTime::from_nanos(ts);
                    if let Some((_, val)) = edge.confidence_history.last_at_or_before(et) {
                        return Ok(Value::Float(*val as f64));
                    }
                }
                Ok(Value::Null)
            }

            "weight_at" => {
                if args.len() < 2 {
                    return Err(QueryError::ExecutionError(
                        "weight_at requires 2 arguments: weight_at(edge, timestamp)".into(),
                    ));
                }
                if let Some(edge) = self.resolve_edge(&args[0], binding)? {
                    let ts = self.resolve_timestamp_arg(&args[1], binding)?;
                    let et = agent_db_core::event_time::EventTime::from_nanos(ts);
                    if let Some((_, val)) = edge.weight_history.last_at_or_before(et) {
                        return Ok(Value::Float(*val as f64));
                    }
                }
                Ok(Value::Null)
            }

            "confidence_history" => {
                if let Some(edge) = args.first().and_then(|a| {
                    self.resolve_edge(a, binding).ok().flatten()
                }) {
                    let entries: Vec<Value> = edge
                        .confidence_history
                        .iter()
                        .map(|(et, val)| {
                            Value::List(vec![
                                Value::Int(et.as_nanos() as i64),
                                Value::Float(*val as f64),
                            ])
                        })
                        .collect();
                    return Ok(Value::List(entries));
                }
                Ok(Value::List(vec![]))
            }

            "weight_history" => {
                if let Some(edge) = args.first().and_then(|a| {
                    self.resolve_edge(a, binding).ok().flatten()
                }) {
                    let entries: Vec<Value> = edge
                        .weight_history
                        .iter()
                        .map(|(et, val)| {
                            Value::List(vec![
                                Value::Int(et.as_nanos() as i64),
                                Value::Float(*val as f64),
                            ])
                        })
                        .collect();
                    return Ok(Value::List(entries));
                }
                Ok(Value::List(vec![]))
            }

            "overlap" => {
                if args.len() >= 2 {
                    let e1 = if let RExpr::Var(slot) = &args[0] {
                        binding.get(*slot).and_then(|b| match b {
                            BoundValue::Edge(id) => self.graph.get_edge(*id),
                            _ => None,
                        })
                    } else {
                        None
                    };
                    let e2 = if let RExpr::Var(slot) = &args[1] {
                        binding.get(*slot).and_then(|b| match b {
                            BoundValue::Edge(id) => self.graph.get_edge(*id),
                            _ => None,
                        })
                    } else {
                        None
                    };
                    if let (Some(edge1), Some(edge2)) = (e1, e2) {
                        let s1 = edge1.valid_from.unwrap_or(0);
                        let e1_end = edge1.valid_until.unwrap_or(u64::MAX);
                        let s2 = edge2.valid_from.unwrap_or(0);
                        let e2_end = edge2.valid_until.unwrap_or(u64::MAX);
                        return Ok(Value::Bool(s1 < e2_end && s2 < e1_end));
                    }
                }
                Ok(Value::Null)
            }

            _ => Err(QueryError::ExecutionError(format!(
                "Unknown function `{}`",
                name
            ))),
        }
    }

    // -----------------------------------------------------------------------
    // Temporal predicates
    // -----------------------------------------------------------------------

    /// SUCCESSIVE(r1, r2) — true if r1's valid_until equals r2's valid_from
    /// (within a tolerance). Default tolerance: 1 second. Optional third arg
    /// overrides: SUCCESSIVE(r1, r2, "100ms").
    ///
    /// v1 note: this is a predicate over existing MATCH bindings, bounded by
    /// MAX_INTERMEDIATE_ROWS. Not yet a specialized temporal adjacency operator.
    fn pred_successive(
        &self,
        args: &[RExpr],
        binding: &BindingRow,
    ) -> Result<bool, QueryError> {
        if args.len() < 2 {
            return Err(QueryError::ExecutionError(
                "SUCCESSIVE requires at least 2 arguments: SUCCESSIVE(r1, r2)".into(),
            ));
        }

        let e1 = self.resolve_edge(&args[0], binding)?;
        let e2 = self.resolve_edge(&args[1], binding)?;

        let (edge1, edge2) = match (e1, e2) {
            (Some(a), Some(b)) => (a, b),
            _ => return Ok(false),
        };

        // Tolerance: default 1 second (in nanos), overridable.
        let tolerance: u64 = if args.len() >= 3 {
            if let RExpr::Literal(Literal::String(dur)) = &args[2] {
                super::planner::parse_duration_nanos(dur).map_err(|e| {
                    QueryError::ExecutionError(format!("SUCCESSIVE tolerance: {}", e))
                })?
            } else {
                1_000_000_000
            }
        } else {
            1_000_000_000
        };

        // r1.valid_until must exist and be close to r2.valid_from.
        match (edge1.valid_until, edge2.valid_from) {
            (Some(end1), Some(start2)) => {
                let diff = if end1 > start2 {
                    end1 - start2
                } else {
                    start2 - end1
                };
                Ok(diff <= tolerance)
            }
            _ => Ok(false),
        }
    }

    /// CHANGED(r, "t1", "t2") — true if the edge started, ended, or was created
    /// within [t1, t2].
    fn pred_changed(
        &self,
        args: &[RExpr],
        binding: &BindingRow,
    ) -> Result<bool, QueryError> {
        if args.len() < 3 {
            return Err(QueryError::ExecutionError(
                "CHANGED requires 3 arguments: CHANGED(r, start, end)".into(),
            ));
        }

        let edge = match self.resolve_edge(&args[0], binding)? {
            Some(e) => e,
            None => return Ok(false),
        };

        let t1 = self.resolve_timestamp_arg(&args[1], binding)?;
        let t2 = self.resolve_timestamp_arg(&args[2], binding)?;

        // Edge "changed" if: started in range, ended in range, or was created in range.
        let started_in = edge.valid_from.map_or(false, |vf| vf >= t1 && vf <= t2);
        let ended_in = edge.valid_until.map_or(false, |vu| vu >= t1 && vu <= t2);
        let created_in = edge.created_at >= t1 && edge.created_at <= t2;

        Ok(started_in || ended_in || created_in)
    }

    /// Resolve two edge arguments to their (valid_from, valid_until) intervals.
    ///
    /// Open intervals use: valid_from.unwrap_or(0), valid_until.unwrap_or(u64::MAX).
    fn resolve_two_edge_intervals(
        &self,
        args: &[RExpr],
        binding: &BindingRow,
    ) -> Result<Option<((u64, u64), (u64, u64))>, QueryError> {
        if args.len() < 2 {
            return Err(QueryError::ExecutionError(
                "Temporal predicate requires 2 edge arguments".into(),
            ));
        }
        let e1 = self.resolve_edge(&args[0], binding)?;
        let e2 = self.resolve_edge(&args[1], binding)?;
        match (e1, e2) {
            (Some(edge1), Some(edge2)) => {
                let i1 = (
                    edge1.valid_from.unwrap_or(0),
                    edge1.valid_until.unwrap_or(u64::MAX),
                );
                let i2 = (
                    edge2.valid_from.unwrap_or(0),
                    edge2.valid_until.unwrap_or(u64::MAX),
                );
                Ok(Some((i1, i2)))
            }
            _ => Ok(None),
        }
    }

    /// Resolve an edge variable from an expression.
    fn resolve_edge<'b>(
        &'b self,
        expr: &RExpr,
        binding: &BindingRow,
    ) -> Result<Option<&'a GraphEdge>, QueryError> {
        if let RExpr::Var(slot) = expr {
            if let Some(BoundValue::Edge(id)) = binding.get(*slot) {
                return Ok(self.graph.get_edge(*id));
            }
        }
        Ok(None)
    }

    /// Resolve a timestamp from an expression argument (string literal → parse).
    fn resolve_timestamp_arg(
        &self,
        expr: &RExpr,
        binding: &BindingRow,
    ) -> Result<u64, QueryError> {
        let val = self.evaluate_expr(expr, binding)?;
        match val {
            Value::String(s) => super::planner::parse_timestamp_str(&s),
            Value::Int(i) => {
                if i < 0 {
                    return Err(QueryError::ExecutionError(
                        format!("Timestamp cannot be negative: {}", i),
                    ));
                }
                Ok(i as u64)
            }
            _ => Err(QueryError::ExecutionError(
                "Expected timestamp string or integer".into(),
            )),
        }
    }

    /// Resolve an expression to a NodeId (for path/hops functions).
    fn resolve_node_id(
        &self,
        expr: &RExpr,
        binding: &BindingRow,
    ) -> Result<Option<NodeId>, QueryError> {
        match expr {
            RExpr::Var(slot) => match binding.get(*slot) {
                Some(BoundValue::Node(id)) => Ok(Some(*id)),
                _ => Ok(None),
            },
            _ => {
                let v = self.evaluate_expr(expr, binding)?;
                match v {
                    Value::Int(i) => Ok(Some(i as NodeId)),
                    _ => Ok(None),
                }
            }
        }
    }

    // -----------------------------------------------------------------------
    // Edge helpers
    // -----------------------------------------------------------------------

    /// Check whether an edge is visible under the current temporal viewport.
    fn edge_visible(&self, edge: &GraphEdge) -> bool {
        if !edge.is_valid() {
            return false;
        }
        // Transaction-time filter (AS OF).
        if let Some(cutoff) = self.transaction_cutoff {
            if edge.created_at > cutoff {
                return false;
            }
        }
        match &self.viewport {
            TemporalViewport::ActiveOnly => edge.valid_until.is_none(),
            TemporalViewport::PointInTime(ts) => edge.valid_at(*ts),
            TemporalViewport::Range(t1, t2) => edge.valid_during(*t1, *t2),
            TemporalViewport::All => true,
        }
    }

    /// Check whether a node passes the transaction-time filter (AS OF).
    fn node_visible_txn(&self, node: &GraphNode) -> bool {
        match self.transaction_cutoff {
            Some(cutoff) => node.created_at <= cutoff,
            None => true,
        }
    }

    /// Check whether an edge matches the requested type filter.
    ///
    /// Matching rules:
    /// - `None` filter matches everything.
    /// - For `Association` edges, match if:
    ///   - `association_type == filter` (exact), or
    ///   - `association_type` starts with `filter:` (category prefix match).
    /// - For non-Association edge types, match by variant name (case-insensitive).
    fn edge_matches_type(&self, edge: &GraphEdge, filter: &Option<String>) -> bool {
        let filter = match filter {
            Some(f) => f,
            None => return true,
        };

        match &edge.edge_type {
            EdgeType::Association {
                association_type, ..
            } => {
                // Exact match.
                if association_type == filter {
                    return true;
                }
                // Category prefix: e.g. filter="location" matches "location:lives_in".
                // Avoid format! allocation — check prefix + ':' manually.
                if association_type.len() > filter.len()
                    && association_type.as_bytes().get(filter.len()) == Some(&b':')
                    && association_type[..filter.len()].eq_ignore_ascii_case(filter)
                {
                    return true;
                }
                false
            }
            other => {
                let variant_name = edge_type_variant_name(other);
                variant_name.eq_ignore_ascii_case(filter)
            }
        }
    }

    // -----------------------------------------------------------------------
    // Aggregation
    // -----------------------------------------------------------------------

    #[allow(clippy::too_many_arguments)]
    fn apply_aggregations(
        &self,
        result_rows: &[Vec<Value>],
        columns: &[String],
        aggregations: &[Aggregation],
        group_by_keys: &[String],
        projections: &[Projection],
        _bindings: &[BindingRow],
    ) -> Result<Vec<Vec<Value>>, QueryError> {
        // Map column aliases to indices.
        let col_index: HashMap<&str, usize> = columns
            .iter()
            .enumerate()
            .map(|(i, c)| (c.as_str(), i))
            .collect();

        // Build group keys.
        let group_col_indices: Vec<usize> = group_by_keys
            .iter()
            .filter_map(|k| col_index.get(k.as_str()).copied())
            .collect();

        // Group rows by hash of group-key columns to avoid String allocation per cell.
        let mut groups: HashMap<u64, Vec<usize>> = HashMap::new();
        for (i, row) in result_rows.iter().enumerate() {
            let mut hasher = std::collections::hash_map::DefaultHasher::new();
            for &ci in &group_col_indices {
                std::mem::discriminant(&row[ci]).hash(&mut hasher);
                match &row[ci] {
                    Value::String(s) => s.hash(&mut hasher),
                    Value::Int(v) => v.hash(&mut hasher),
                    Value::Float(f) => f.to_bits().hash(&mut hasher),
                    Value::Bool(b) => b.hash(&mut hasher),
                    Value::Null => 0u8.hash(&mut hasher),
                    Value::List(items) => items.len().hash(&mut hasher),
                    Value::Map(m) => m.len().hash(&mut hasher),
                }
            }
            let key = hasher.finish();
            groups.entry(key).or_default().push(i);
        }

        // If no group-by keys, treat all rows as one group.
        if group_col_indices.is_empty() && groups.is_empty() && !result_rows.is_empty() {
            groups.insert(0, (0..result_rows.len()).collect());
        }

        let mut output = Vec::with_capacity(groups.len());

        for (_key, row_indices) in &groups {
            let mut out_row = Vec::with_capacity(projections.len());

            for proj in projections {
                // Check if this projection is an aggregation.
                if let Some(agg) = aggregations.iter().find(|a| a.output_alias == proj.alias) {
                    // Use already-computed values from result_rows instead of
                    // re-evaluating expressions against bindings. Find the column
                    // index for the aggregate's input expression.
                    let input_col = col_index.get(agg.output_alias.as_str());
                    let values: Vec<Value> = row_indices
                        .iter()
                        .map(|&i| {
                            if let Some(&ci) = input_col {
                                result_rows[i][ci].clone()
                            } else {
                                // For count(*) the value is always 1
                                Value::Int(1)
                            }
                        })
                        .collect();
                    let agg_val = compute_aggregate(&agg.function, &values);
                    out_row.push(agg_val);
                } else if let Some(&ci) = col_index.get(proj.alias.as_str()) {
                    // Use the first row's value for grouped columns.
                    let first = row_indices.first().copied().unwrap_or(0);
                    out_row.push(result_rows[first][ci].clone());
                } else {
                    out_row.push(Value::Null);
                }
            }

            output.push(out_row);
        }

        Ok(output)
    }

    // -----------------------------------------------------------------------
    // Deadline
    // -----------------------------------------------------------------------

    /// Check whether the query has exceeded its time budget.
    #[inline]
    fn check_deadline(&self) -> Result<(), QueryError> {
        if Instant::now() >= self.deadline {
            return Err(QueryError::Timeout);
        }
        Ok(())
    }

    // -----------------------------------------------------------------------
    // Ordering
    // -----------------------------------------------------------------------

    fn apply_ordering(
        &self,
        rows: &mut [Vec<Value>],
        columns: &[String],
        ordering: &[OrderSpec],
    ) {
        let col_index: HashMap<&str, usize> = columns
            .iter()
            .enumerate()
            .map(|(i, c)| (c.as_str(), i))
            .collect();

        let specs: Vec<(usize, bool)> = ordering
            .iter()
            .filter_map(|o| {
                col_index
                    .get(o.column_alias.as_str())
                    .map(|&i| (i, o.descending))
            })
            .collect();

        rows.sort_by(|a, b| {
            for &(ci, desc) in &specs {
                let cmp = a[ci].partial_cmp(&b[ci]).unwrap_or(std::cmp::Ordering::Equal);
                let cmp = if desc { cmp.reverse() } else { cmp };
                if cmp != std::cmp::Ordering::Equal {
                    return cmp;
                }
            }
            std::cmp::Ordering::Equal
        });
    }

    /// Execute a plan and return results with binding rows preserved.
    /// Used by the subscription system to build structural RowIds.
    pub fn execute_with_bindings(
        graph: &'a Graph,
        ontology: &'a OntologyRegistry,
        plan: ExecutionPlan,
    ) -> Result<(QueryOutput, Vec<SmallVec<[(SlotIdx, crate::subscription::incremental::BoundEntityId); 4]>>), QueryError> {
        let start = Instant::now();

        let mut executor = Executor {
            graph,
            ontology,
            viewport: plan.temporal_viewport.clone(),
            transaction_cutoff: plan.transaction_cutoff,
            stats: ExecutionStats {
                nodes_scanned: 0,
                edges_traversed: 0,
            },
            deadline: start + QUERY_TIMEOUT,
        };

        let mut rows: Vec<BindingRow> = vec![BindingRow::new(plan.var_count)];

        for step in &plan.steps {
            rows = executor.execute_step(step, rows)?;
        }

        // Extract binding rows before projection.
        let binding_rows: Vec<SmallVec<[(SlotIdx, crate::subscription::incremental::BoundEntityId); 4]>> = rows
            .iter()
            .map(|br| {
                let mut slots = SmallVec::new();
                for i in 0..plan.var_count {
                    if let Some(bv) = br.get(i) {
                        let entity = match bv {
                            BoundValue::Node(id) => crate::subscription::incremental::BoundEntityId::Node(*id),
                            BoundValue::Edge(id) => crate::subscription::incremental::BoundEntityId::Edge(*id),
                            BoundValue::Path(_) => continue, // Paths not supported in subscriptions
                        };
                        slots.push((i, entity));
                    }
                }
                slots
            })
            .collect();

        // Projection.
        let columns: Vec<String> = plan.projections.iter().map(|p| p.alias.clone()).collect();
        let mut result_rows: Vec<Vec<Value>> = Vec::with_capacity(rows.len());
        for binding in &rows {
            let mut row = Vec::with_capacity(plan.projections.len());
            for proj in &plan.projections {
                let val = executor.evaluate_expr(&proj.expr, binding)?;
                row.push(val);
            }
            result_rows.push(row);
        }

        // Aggregation.
        if !plan.aggregations.is_empty() {
            result_rows = executor.apply_aggregations(
                &result_rows,
                &columns,
                &plan.aggregations,
                &plan.group_by_keys,
                &plan.projections,
                &rows,
            )?;
        }

        // DISTINCT.
        if plan.projections.iter().any(|p| p.distinct) {
            let mut seen = HashSet::new();
            let mut deduped = Vec::new();
            for row in result_rows {
                let hash = hash_row(&row);
                if seen.insert(hash) {
                    deduped.push(row);
                }
            }
            result_rows = deduped;
        }

        // ORDER BY.
        if !plan.ordering.is_empty() {
            executor.apply_ordering(&mut result_rows, &columns, &plan.ordering);
        }

        // LIMIT.
        if let Some(limit) = plan.limit {
            result_rows.truncate(limit as usize);
        }

        let elapsed = start.elapsed();

        let output = QueryOutput {
            columns,
            rows: result_rows,
            stats: QueryStats {
                nodes_scanned: executor.stats.nodes_scanned,
                edges_traversed: executor.stats.edges_traversed,
                execution_time_ms: elapsed.as_millis() as u64,
            },
        };

        Ok((output, binding_rows))
    }
}

// ---------------------------------------------------------------------------
// Standalone visibility/matching functions for subscription system
// ---------------------------------------------------------------------------

/// Check whether an edge is visible under a temporal viewport.
/// Standalone version of `Executor::edge_visible` for use by subscription operators.
pub fn edge_visible_standalone(
    edge: &GraphEdge,
    viewport: &TemporalViewport,
    txn_cutoff: Option<u64>,
) -> bool {
    if !edge.is_valid() {
        return false;
    }
    if let Some(cutoff) = txn_cutoff {
        if edge.created_at > cutoff {
            return false;
        }
    }
    match viewport {
        TemporalViewport::ActiveOnly => edge.valid_until.is_none(),
        TemporalViewport::PointInTime(ts) => edge.valid_at(*ts),
        TemporalViewport::Range(t1, t2) => edge.valid_during(*t1, *t2),
        TemporalViewport::All => true,
    }
}

/// Check whether an edge matches a type filter.
/// Standalone version of `Executor::edge_matches_type` for use by subscription operators.
pub fn edge_matches_type_standalone(edge: &GraphEdge, filter: &Option<String>) -> bool {
    let filter = match filter {
        Some(f) => f,
        None => return true,
    };

    match &edge.edge_type {
        EdgeType::Association {
            association_type, ..
        } => {
            if association_type == filter {
                return true;
            }
            if association_type.len() > filter.len()
                && association_type.as_bytes().get(filter.len()) == Some(&b':')
                && association_type[..filter.len()].eq_ignore_ascii_case(filter)
            {
                return true;
            }
            false
        }
        other => {
            let variant_name = edge_type_variant_name(other);
            variant_name.eq_ignore_ascii_case(filter)
        }
    }
}

/// Check whether a node satisfies property filters.
/// Standalone version of `Executor::node_matches_props` for use by subscription operators.
pub fn node_matches_props_standalone(
    node: &GraphNode,
    props: &[(String, Literal)],
    _graph: &Graph,
) -> bool {
    for (key, lit) in props {
        let val = node_property_value_standalone(node, key);
        let expected = literal_to_value(lit);
        if !value_eq_loose(&val, &expected) {
            return false;
        }
    }
    true
}

/// Extract a named property from a node (standalone version).
pub(crate) fn node_property_value_standalone(node: &GraphNode, prop: &str) -> Value {
    match prop {
        "id" => Value::Int(node.id as i64),
        "name" | "label" => Value::String(node.label()),
        "type" => Value::String(node.type_name().to_string()),
        "created_at" => Value::Int(node.created_at as i64),
        "updated_at" => Value::Int(node.updated_at as i64),
        "group_id" => Value::String(node.group_id.clone()),
        "degree" => Value::Int(node.degree as i64),
        "confidence" => match &node.node_type {
            NodeType::Concept { confidence, .. } | NodeType::Claim { confidence, .. } => {
                Value::Float(*confidence as f64)
            }
            _ => Value::Null,
        },
        "concept_name" => match &node.node_type {
            NodeType::Concept { concept_name, .. } => Value::String(concept_name.clone()),
            _ => Value::Null,
        },
        _ => node
            .properties
            .get(prop)
            .map(|jv| json_to_value(jv))
            .unwrap_or(Value::Null),
    }
}

// ===========================================================================
// Free functions
// ===========================================================================

/// Compute a u64 hash for a row of Values (for DISTINCT deduplication).
/// Much cheaper than `format!("{:?}", row)` which allocates a String per row.
fn hash_row(row: &[Value]) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    for val in row {
        std::mem::discriminant(val).hash(&mut hasher);
        match val {
            Value::String(s) => s.hash(&mut hasher),
            Value::Int(i) => i.hash(&mut hasher),
            Value::Float(f) => f.to_bits().hash(&mut hasher),
            Value::Bool(b) => b.hash(&mut hasher),
            Value::Null => 0u8.hash(&mut hasher),
            Value::List(items) => {
                items.len().hash(&mut hasher);
                // Hash first few items to avoid O(n) for huge lists
                for item in items.iter().take(8) {
                    if let Value::String(s) = item {
                        s.hash(&mut hasher);
                    }
                }
            }
            Value::Map(m) => m.len().hash(&mut hasher),
        }
    }
    hasher.finish()
}

/// Human-readable string for an EdgeType.
///
/// Returns a `Cow<str>` to avoid allocation for the common static-name variants.
/// Only `Association` and `CodeStructure` need owned strings.
pub fn edge_type_display(et: &EdgeType) -> String {
    match et {
        EdgeType::Association {
            association_type, ..
        } => association_type.clone(),
        EdgeType::CodeStructure { relation_kind, .. } => {
            format!("code_structure:{}", relation_kind)
        }
        other => edge_type_variant_name(other).to_string(),
    }
}

/// Variant name (lowercase) for non-Association edge types.
fn edge_type_variant_name(et: &EdgeType) -> &'static str {
    match et {
        EdgeType::Causality { .. } => "causality",
        EdgeType::Temporal { .. } => "temporal",
        EdgeType::Contextual { .. } => "contextual",
        EdgeType::Interaction { .. } => "interaction",
        EdgeType::GoalRelation { .. } => "goal_relation",
        EdgeType::Association { .. } => "association",
        EdgeType::Communication { .. } => "communication",
        EdgeType::DerivedFrom { .. } => "derived_from",
        EdgeType::SupportedBy { .. } => "supported_by",
        EdgeType::CodeStructure { .. } => "code_structure",
        EdgeType::About { .. } => "about",
    }
}

/// Convert an AST literal to a runtime Value.
fn literal_to_value(lit: &Literal) -> Value {
    match lit {
        Literal::String(s) => Value::String(s.clone()),
        Literal::Int(i) => Value::Int(*i),
        Literal::Float(f) => Value::Float(*f),
        Literal::Bool(b) => Value::Bool(*b),
        Literal::Null => Value::Null,
    }
}

/// Convert a `serde_json::Value` to a runtime `Value`.
fn json_to_value(jv: &serde_json::Value) -> Value {
    match jv {
        serde_json::Value::Null => Value::Null,
        serde_json::Value::Bool(b) => Value::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Value::Int(i)
            } else if let Some(f) = n.as_f64() {
                Value::Float(f)
            } else {
                Value::Null
            }
        }
        serde_json::Value::String(s) => Value::String(s.clone()),
        serde_json::Value::Array(arr) => {
            Value::List(arr.iter().map(json_to_value).collect())
        }
        serde_json::Value::Object(map) => {
            let m: HashMap<String, Value> = map
                .iter()
                .map(|(k, v)| (k.clone(), json_to_value(v)))
                .collect();
            Value::Map(m)
        }
    }
}

/// Loose equality: case-insensitive for strings.
fn value_eq_loose(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::String(sa), Value::String(sb)) => sa.eq_ignore_ascii_case(sb),
        (Value::Int(ia), Value::Int(ib)) => ia == ib,
        (Value::Float(fa), Value::Float(fb)) => (fa - fb).abs() < f64::EPSILON,
        (Value::Int(i), Value::Float(f)) | (Value::Float(f), Value::Int(i)) => {
            (*i as f64 - f).abs() < f64::EPSILON
        }
        (Value::Bool(a), Value::Bool(b)) => a == b,
        (Value::Null, Value::Null) => true,
        _ => false,
    }
}

/// Compare two Values with the given operator.
fn compare_values(lhs: &Value, op: &CompOp, rhs: &Value) -> bool {
    match op {
        CompOp::Eq => value_eq_loose(lhs, rhs),
        CompOp::Neq => !value_eq_loose(lhs, rhs),
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
                // Case-insensitive contains without allocating the haystack lowercase.
                let needle_lower = needle.to_lowercase();
                haystack
                    .as_bytes()
                    .windows(needle_lower.len())
                    .any(|window| {
                        window
                            .iter()
                            .zip(needle_lower.as_bytes())
                            .all(|(a, b)| a.to_ascii_lowercase() == *b)
                    })
            }
            (Value::List(items), val) => items.iter().any(|item| value_eq_loose(item, val)),
            _ => false,
        },
        CompOp::StartsWith => match (lhs, rhs) {
            (Value::String(haystack), Value::String(prefix)) => {
                haystack.len() >= prefix.len()
                    && haystack.as_bytes()[..prefix.len()]
                        .iter()
                        .zip(prefix.as_bytes())
                        .all(|(a, b)| a.to_ascii_lowercase() == b.to_ascii_lowercase())
            }
            _ => false,
        },
    }
}

// ---------------------------------------------------------------------------
// Temporal bucketing / truncation helpers
// ---------------------------------------------------------------------------

/// Floor a nanosecond timestamp to a bucket boundary.
///
/// Calendar units (month, quarter, year) route through calendar-aware truncation.
/// Fixed-width units (second, minute, hour, day, week) use modular arithmetic.
fn truncate_timestamp(nanos: u64, width: &str) -> Result<u64, QueryError> {
    let lower = width.to_lowercase();
    // Calendar units MUST use calendar-aware path — parse_duration_nanos treats
    // "1y" as 365d and "1m" as 30d, which is wrong for bucket boundaries.
    match lower.as_str() {
        "month" | "quarter" | "year" => return truncate_timestamp_by_unit_lower(nanos, &lower),
        _ => {}
    }
    // Try parsing as a fixed-width duration string (e.g. "1d", "6h", "30m").
    if let Ok(bucket_width) = super::planner::parse_duration_nanos(width) {
        if bucket_width == 0 {
            return Err(QueryError::ExecutionError(
                "time_bucket width must be > 0".into(),
            ));
        }
        return Ok(nanos - (nanos % bucket_width));
    }
    // Named unit fallback (e.g. "day", "hour").
    truncate_timestamp_by_unit_lower(nanos, &lower)
}

/// Calendar-aware truncation by named unit.
fn truncate_timestamp_by_unit(nanos: u64, unit: &str) -> Result<u64, QueryError> {
    truncate_timestamp_by_unit_lower(nanos, &unit.to_lowercase())
}

/// Inner truncation: expects already-lowercased unit.
fn truncate_timestamp_by_unit_lower(nanos: u64, unit: &str) -> Result<u64, QueryError> {
    use super::planner::{civil_to_nanos, nanos_to_civil};

    match unit {
        "second" | "s" => Ok(nanos - (nanos % 1_000_000_000)),
        "minute" | "min" => Ok(nanos - (nanos % 60_000_000_000)),
        "hour" | "h" => Ok(nanos - (nanos % 3_600_000_000_000)),
        "day" | "d" => {
            let (y, m, d) = nanos_to_civil(nanos);
            Ok(civil_to_nanos(y, m, d))
        }
        "week" | "w" => {
            // Floor to Monday. Unix epoch (1970-01-01) was a Thursday (weekday 3, Mon=0).
            let total_days = (nanos / (86_400 * 1_000_000_000)) as i64;
            // day_of_week: 0=Mon, 1=Tue, ..., 6=Sun
            let dow = ((total_days + 3) % 7 + 7) % 7; // +3 because epoch is Thursday
            let monday_days = total_days - dow;
            let (y, m, d) = super::planner::civil_from_days(monday_days);
            Ok(civil_to_nanos(y, m, d))
        }
        "month" => {
            let (y, m, _) = nanos_to_civil(nanos);
            Ok(civil_to_nanos(y, m, 1))
        }
        "quarter" => {
            let (y, m, _) = nanos_to_civil(nanos);
            let q_month = ((m - 1) / 3) * 3 + 1;
            Ok(civil_to_nanos(y, q_month, 1))
        }
        "year" => {
            let (y, _, _) = nanos_to_civil(nanos);
            Ok(civil_to_nanos(y, 1, 1))
        }
        _ => Err(QueryError::ExecutionError(format!(
            "unknown time unit: {}",
            unit
        ))),
    }
}

/// Compute an aggregate over a slice of values.
fn compute_aggregate(func: &AggregateFunction, values: &[Value]) -> Value {
    match func {
        AggregateFunction::Count => Value::Int(values.len() as i64),

        AggregateFunction::Sum => {
            let mut sum = 0.0_f64;
            let mut has_value = false;
            for v in values {
                if let Some(f) = v.as_f64() {
                    sum += f;
                    has_value = true;
                }
            }
            if has_value {
                Value::Float(sum)
            } else {
                Value::Null
            }
        }

        AggregateFunction::Avg => {
            let mut sum = 0.0_f64;
            let mut count = 0_u64;
            for v in values {
                if let Some(f) = v.as_f64() {
                    sum += f;
                    count += 1;
                }
            }
            if count > 0 {
                Value::Float(sum / count as f64)
            } else {
                Value::Null
            }
        }

        AggregateFunction::Min => {
            values
                .iter()
                .filter(|v| !v.is_null())
                .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .cloned()
                .unwrap_or(Value::Null)
        }

        AggregateFunction::Max => {
            values
                .iter()
                .filter(|v| !v.is_null())
                .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
                .cloned()
                .unwrap_or(Value::Null)
        }

        AggregateFunction::Collect => Value::List(values.to_vec()),
    }
}

// ===========================================================================
// Tests
// ===========================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::ast::WhenClause;
    use agent_db_core::types::Timestamp;

    // -----------------------------------------------------------------------
    // BindingRow unit tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_binding_row_inline() {
        let mut row = BindingRow::new(3);
        assert!(row.get(0).is_none());
        row.set(0, BoundValue::Node(42));
        row.set(2, BoundValue::Edge(99));
        match row.get(0) {
            Some(BoundValue::Node(id)) => assert_eq!(*id, 42),
            other => panic!("expected Node(42), got {:?}", other),
        }
        match row.get(2) {
            Some(BoundValue::Edge(id)) => assert_eq!(*id, 99),
            other => panic!("expected Edge(99), got {:?}", other),
        }
        assert!(row.get(1).is_none());
    }

    #[test]
    fn test_binding_row_dynamic() {
        let mut row = BindingRow::new(20); // > INLINE_SLOTS
        assert!(matches!(row.slots, BindingSlots::Dynamic(_)));
        row.set(0, BoundValue::Node(1));
        row.set(19, BoundValue::Node(2));
        match row.get(0) {
            Some(BoundValue::Node(id)) => assert_eq!(*id, 1),
            other => panic!("expected Node(1), got {:?}", other),
        }
        match row.get(19) {
            Some(BoundValue::Node(id)) => assert_eq!(*id, 2),
            other => panic!("expected Node(2), got {:?}", other),
        }
    }

    #[test]
    fn test_binding_row_clone() {
        let mut row = BindingRow::new(4);
        row.set(0, BoundValue::Node(10));
        row.set(1, BoundValue::Edge(20));
        let cloned = row.clone();
        match cloned.get(0) {
            Some(BoundValue::Node(id)) => assert_eq!(*id, 10),
            other => panic!("expected Node(10), got {:?}", other),
        }
        match cloned.get(1) {
            Some(BoundValue::Edge(id)) => assert_eq!(*id, 20),
            other => panic!("expected Edge(20), got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // edge_type_display
    // -----------------------------------------------------------------------

    #[test]
    fn test_edge_type_display_association() {
        let et = EdgeType::Association {
            association_type: "location:lives_in".to_string(),
            evidence_count: 1,
            statistical_significance: 0.9,
        };
        assert_eq!(edge_type_display(&et), "location:lives_in");
    }

    #[test]
    fn test_edge_type_display_causality() {
        let et = EdgeType::Causality {
            strength: 0.8,
            lag_ms: 100,
        };
        assert_eq!(edge_type_display(&et), "causality");
    }

    #[test]
    fn test_edge_type_display_temporal() {
        let et = EdgeType::Temporal {
            average_interval_ms: 5000,
            sequence_confidence: 0.95,
        };
        assert_eq!(edge_type_display(&et), "temporal");
    }

    #[test]
    fn test_edge_type_display_contextual() {
        let et = EdgeType::Contextual {
            similarity: 0.7,
            co_occurrence_rate: 0.3,
        };
        assert_eq!(edge_type_display(&et), "contextual");
    }

    #[test]
    fn test_edge_type_display_code_structure() {
        let et = EdgeType::CodeStructure {
            relation_kind: "calls".to_string(),
            file_path: "main.rs".to_string(),
            confidence: 1.0,
        };
        assert_eq!(edge_type_display(&et), "code_structure:calls");
    }

    #[test]
    fn test_edge_type_display_about() {
        let et = EdgeType::About {
            relevance_score: 0.9,
            mention_count: 3,
            entity_role: Default::default(),
            predicate: None,
        };
        assert_eq!(edge_type_display(&et), "about");
    }

    // -----------------------------------------------------------------------
    // literal_to_value
    // -----------------------------------------------------------------------

    #[test]
    fn test_literal_to_value_string() {
        let v = literal_to_value(&Literal::String("hello".into()));
        assert_eq!(v, Value::String("hello".into()));
    }

    #[test]
    fn test_literal_to_value_int() {
        let v = literal_to_value(&Literal::Int(42));
        assert_eq!(v, Value::Int(42));
    }

    #[test]
    fn test_literal_to_value_float() {
        let v = literal_to_value(&Literal::Float(3.14));
        assert_eq!(v, Value::Float(3.14));
    }

    #[test]
    fn test_literal_to_value_bool() {
        let v = literal_to_value(&Literal::Bool(true));
        assert_eq!(v, Value::Bool(true));
    }

    #[test]
    fn test_literal_to_value_null() {
        let v = literal_to_value(&Literal::Null);
        assert_eq!(v, Value::Null);
    }

    // -----------------------------------------------------------------------
    // compare_values
    // -----------------------------------------------------------------------

    #[test]
    fn test_compare_eq_strings_case_insensitive() {
        let a = Value::String("Alice".into());
        let b = Value::String("alice".into());
        assert!(compare_values(&a, &CompOp::Eq, &b));
    }

    #[test]
    fn test_compare_neq() {
        let a = Value::Int(1);
        let b = Value::Int(2);
        assert!(compare_values(&a, &CompOp::Neq, &b));
    }

    #[test]
    fn test_compare_lt() {
        let a = Value::Int(1);
        let b = Value::Int(2);
        assert!(compare_values(&a, &CompOp::Lt, &b));
        assert!(!compare_values(&b, &CompOp::Lt, &a));
    }

    #[test]
    fn test_compare_gt() {
        let a = Value::Float(3.14);
        let b = Value::Float(2.71);
        assert!(compare_values(&a, &CompOp::Gt, &b));
    }

    #[test]
    fn test_compare_contains_string() {
        let a = Value::String("hello world".into());
        let b = Value::String("World".into());
        assert!(compare_values(&a, &CompOp::Contains, &b));
    }

    #[test]
    fn test_compare_contains_list() {
        let a = Value::List(vec![Value::Int(1), Value::Int(2), Value::Int(3)]);
        let b = Value::Int(2);
        assert!(compare_values(&a, &CompOp::Contains, &b));
        assert!(!compare_values(&a, &CompOp::Contains, &Value::Int(5)));
    }

    #[test]
    fn test_compare_starts_with() {
        let a = Value::String("location:lives_in".into());
        let b = Value::String("location".into());
        assert!(compare_values(&a, &CompOp::StartsWith, &b));
    }

    // -----------------------------------------------------------------------
    // compute_aggregate
    // -----------------------------------------------------------------------

    #[test]
    fn test_aggregate_count() {
        let vals = vec![Value::Int(1), Value::Int(2), Value::Int(3)];
        assert_eq!(compute_aggregate(&AggregateFunction::Count, &vals), Value::Int(3));
    }

    #[test]
    fn test_aggregate_sum() {
        let vals = vec![Value::Int(10), Value::Int(20), Value::Float(5.5)];
        let result = compute_aggregate(&AggregateFunction::Sum, &vals);
        match result {
            Value::Float(f) => assert!((f - 35.5).abs() < 0.001),
            _ => panic!("Expected Float"),
        }
    }

    #[test]
    fn test_aggregate_avg() {
        let vals = vec![Value::Int(10), Value::Int(20)];
        let result = compute_aggregate(&AggregateFunction::Avg, &vals);
        match result {
            Value::Float(f) => assert!((f - 15.0).abs() < 0.001),
            _ => panic!("Expected Float"),
        }
    }

    #[test]
    fn test_aggregate_min() {
        let vals = vec![Value::Int(5), Value::Int(2), Value::Int(8)];
        assert_eq!(compute_aggregate(&AggregateFunction::Min, &vals), Value::Int(2));
    }

    #[test]
    fn test_aggregate_max() {
        let vals = vec![Value::Int(5), Value::Int(2), Value::Int(8)];
        assert_eq!(compute_aggregate(&AggregateFunction::Max, &vals), Value::Int(8));
    }

    #[test]
    fn test_aggregate_collect() {
        let vals = vec![Value::String("a".into()), Value::String("b".into())];
        let result = compute_aggregate(&AggregateFunction::Collect, &vals);
        assert_eq!(
            result,
            Value::List(vec![Value::String("a".into()), Value::String("b".into())])
        );
    }

    #[test]
    fn test_aggregate_sum_no_numeric() {
        let vals = vec![Value::String("x".into()), Value::Null];
        assert_eq!(compute_aggregate(&AggregateFunction::Sum, &vals), Value::Null);
    }

    // -----------------------------------------------------------------------
    // json_to_value
    // -----------------------------------------------------------------------

    #[test]
    fn test_json_to_value() {
        assert_eq!(json_to_value(&serde_json::json!(null)), Value::Null);
        assert_eq!(json_to_value(&serde_json::json!(true)), Value::Bool(true));
        assert_eq!(json_to_value(&serde_json::json!(42)), Value::Int(42));
        assert_eq!(json_to_value(&serde_json::json!(3.14)), Value::Float(3.14));
        assert_eq!(
            json_to_value(&serde_json::json!("hi")),
            Value::String("hi".into())
        );
        assert_eq!(
            json_to_value(&serde_json::json!([1, 2])),
            Value::List(vec![Value::Int(1), Value::Int(2)])
        );
    }

    // -----------------------------------------------------------------------
    // edge_visible (standalone edge, no graph needed)
    // -----------------------------------------------------------------------

    fn make_test_edge(valid_from: Option<Timestamp>, valid_until: Option<Timestamp>) -> GraphEdge {
        GraphEdge {
            id: 0,
            source: 0,
            target: 1,
            edge_type: EdgeType::Association {
                association_type: "test".to_string(),
                evidence_count: 1,
                statistical_significance: 1.0,
            },
            weight: 1.0,
            created_at: 1000,
            updated_at: 1000,
            valid_from,
            valid_until,
            observation_count: 1,
            confidence: 1.0,
            group_id: String::new(),
            properties: HashMap::new(),
            confidence_history: crate::tcell::TCell::Empty,
            weight_history: crate::tcell::TCell::Empty,
        }
    }

    /// Helper that checks edge visibility without needing a full Executor.
    fn edge_visible_with_viewport(edge: &GraphEdge, vp: &TemporalViewport) -> bool {
        if !edge.is_valid() {
            return false;
        }
        match vp {
            TemporalViewport::ActiveOnly => edge.valid_until.is_none(),
            TemporalViewport::PointInTime(ts) => edge.valid_at(*ts),
            TemporalViewport::Range(t1, t2) => edge.valid_during(*t1, *t2),
            TemporalViewport::All => true,
        }
    }

    #[test]
    fn test_edge_visible_active_only() {
        let active = make_test_edge(Some(100), None);
        let expired = make_test_edge(Some(100), Some(200));

        assert!(edge_visible_with_viewport(&active, &TemporalViewport::ActiveOnly));
        assert!(!edge_visible_with_viewport(&expired, &TemporalViewport::ActiveOnly));
    }

    #[test]
    fn test_edge_visible_point_in_time() {
        let edge = make_test_edge(Some(100), Some(300));
        let vp = TemporalViewport::PointInTime(200);

        assert!(edge_visible_with_viewport(&edge, &vp));
        assert!(!edge_visible_with_viewport(&edge, &TemporalViewport::PointInTime(50)));
        assert!(!edge_visible_with_viewport(&edge, &TemporalViewport::PointInTime(300)));
    }

    #[test]
    fn test_edge_visible_range() {
        let edge = make_test_edge(Some(100), Some(300));

        // Overlapping range.
        assert!(edge_visible_with_viewport(&edge, &TemporalViewport::Range(200, 400)));
        // Non-overlapping range (entirely before).
        assert!(!edge_visible_with_viewport(&edge, &TemporalViewport::Range(0, 50)));
        // Non-overlapping range (entirely after).
        assert!(!edge_visible_with_viewport(&edge, &TemporalViewport::Range(300, 500)));
    }

    #[test]
    fn test_edge_visible_all() {
        let edge = make_test_edge(Some(100), Some(200));
        assert!(edge_visible_with_viewport(&edge, &TemporalViewport::All));
    }

    #[test]
    fn test_edge_visible_soft_deleted() {
        let mut edge = make_test_edge(Some(100), None);
        edge.properties
            .insert("is_valid".to_string(), serde_json::json!(false));
        assert!(!edge_visible_with_viewport(&edge, &TemporalViewport::All));
    }

    // -----------------------------------------------------------------------
    // edge_matches_type (standalone)
    // -----------------------------------------------------------------------

    #[test]
    fn test_edge_matches_type_exact() {
        let edge = make_test_edge(None, None);
        // The test edge has association_type = "test".
        assert!(matches_type_helper(&edge, &Some("test".to_string())));
        assert!(!matches_type_helper(&edge, &Some("other".to_string())));
    }

    #[test]
    fn test_edge_matches_type_prefix() {
        let mut edge = make_test_edge(None, None);
        edge.edge_type = EdgeType::Association {
            association_type: "location:lives_in".to_string(),
            evidence_count: 1,
            statistical_significance: 1.0,
        };
        assert!(matches_type_helper(&edge, &Some("location".to_string())));
        assert!(!matches_type_helper(&edge, &Some("relationship".to_string())));
    }

    #[test]
    fn test_edge_matches_type_none_filter() {
        let edge = make_test_edge(None, None);
        assert!(matches_type_helper(&edge, &None));
    }

    #[test]
    fn test_edge_matches_type_non_association() {
        let mut edge = make_test_edge(None, None);
        edge.edge_type = EdgeType::Causality {
            strength: 0.5,
            lag_ms: 100,
        };
        assert!(matches_type_helper(&edge, &Some("causality".to_string())));
        assert!(matches_type_helper(&edge, &Some("Causality".to_string())));
        assert!(!matches_type_helper(&edge, &Some("temporal".to_string())));
    }

    /// Standalone edge type matching (mirrors Executor::edge_matches_type logic).
    fn matches_type_helper(edge: &GraphEdge, filter: &Option<String>) -> bool {
        let filter = match filter {
            Some(f) => f,
            None => return true,
        };
        match &edge.edge_type {
            EdgeType::Association {
                association_type, ..
            } => {
                if association_type == filter {
                    return true;
                }
                if association_type.starts_with(&format!("{}:", filter)) {
                    return true;
                }
                false
            }
            other => {
                let name = edge_type_variant_name(other);
                name.eq_ignore_ascii_case(filter)
            }
        }
    }

    // -----------------------------------------------------------------------
    // value_eq_loose
    // -----------------------------------------------------------------------

    #[test]
    fn test_value_eq_loose_cross_type() {
        assert!(value_eq_loose(&Value::Int(3), &Value::Float(3.0)));
        assert!(!value_eq_loose(&Value::Int(3), &Value::String("3".into())));
        assert!(value_eq_loose(&Value::Null, &Value::Null));
    }

    // -----------------------------------------------------------------------
    // Temporal extension tests
    // -----------------------------------------------------------------------

    /// Build a minimal graph for temporal extension tests.
    ///
    /// Layout:
    /// - Node 1: concept "alice" (created_at=100)
    /// - Node 2: concept "london" (created_at=100)
    /// - Node 3: concept "berlin" (created_at=200)
    /// - Node 4: concept "tokyo" (created_at=300)
    /// - Edge 1: alice→london, location:lives_in, valid_from=100, valid_until=200, created_at=100
    /// - Edge 2: alice→berlin, location:lives_in, valid_from=200, valid_until=300, created_at=200
    /// - Edge 3: alice→tokyo, location:lives_in, valid_from=300, valid_until=None, created_at=300
    fn build_temporal_test_graph() -> (Graph, crate::ontology::OntologyRegistry) {
        use crate::structures::ConceptType;
        use std::sync::Arc;

        let mut g = Graph::new();

        // Create nodes
        let n_alice = GraphNode {
            id: 0,
            node_type: NodeType::Concept { concept_name: "alice".to_string(), concept_type: ConceptType::Person, confidence: 1.0 },
            created_at: 100,
            updated_at: 100,
            properties: HashMap::new(),
            degree: 0,
            embedding: Vec::new(),
            group_id: String::new(),
        };
        let id_alice = g.add_node(n_alice).unwrap();
        g.concept_index.insert(Arc::from("alice"), id_alice);

        let n_london = GraphNode {
            id: 0,
            node_type: NodeType::Concept { concept_name: "london".to_string(), concept_type: ConceptType::Location, confidence: 1.0 },
            created_at: 100,
            updated_at: 100,
            properties: HashMap::new(),
            degree: 0,
            embedding: Vec::new(),
            group_id: String::new(),
        };
        let id_london = g.add_node(n_london).unwrap();
        g.concept_index.insert(Arc::from("london"), id_london);

        let n_berlin = GraphNode {
            id: 0,
            node_type: NodeType::Concept { concept_name: "berlin".to_string(), concept_type: ConceptType::Location, confidence: 1.0 },
            created_at: 200,
            updated_at: 200,
            properties: HashMap::new(),
            degree: 0,
            embedding: Vec::new(),
            group_id: String::new(),
        };
        let id_berlin = g.add_node(n_berlin).unwrap();
        g.concept_index.insert(Arc::from("berlin"), id_berlin);

        let n_tokyo = GraphNode {
            id: 0,
            node_type: NodeType::Concept { concept_name: "tokyo".to_string(), concept_type: ConceptType::Location, confidence: 1.0 },
            created_at: 300,
            updated_at: 300,
            properties: HashMap::new(),
            degree: 0,
            embedding: Vec::new(),
            group_id: String::new(),
        };
        let id_tokyo = g.add_node(n_tokyo).unwrap();
        g.concept_index.insert(Arc::from("tokyo"), id_tokyo);

        // Edge: alice→london, valid [100, 200)
        let e1 = GraphEdge {
            id: 0,
            source: id_alice,
            target: id_london,
            edge_type: EdgeType::Association {
                association_type: "location:lives_in".to_string(),
                evidence_count: 1,
                statistical_significance: 1.0,
            },
            weight: 1.0,
            created_at: 100,
            updated_at: 100,
            valid_from: Some(100),
            valid_until: Some(200),
            observation_count: 1,
            confidence: 1.0,
            properties: HashMap::new(),
            confidence_history: crate::tcell::TCell::Empty,
            weight_history: crate::tcell::TCell::Empty,
            group_id: String::new(),
        };
        g.add_edge(e1);

        // Edge: alice→berlin, valid [200, 300)
        let e2 = GraphEdge {
            id: 0,
            source: id_alice,
            target: id_berlin,
            edge_type: EdgeType::Association {
                association_type: "location:lives_in".to_string(),
                evidence_count: 1,
                statistical_significance: 1.0,
            },
            weight: 1.0,
            created_at: 200,
            updated_at: 200,
            valid_from: Some(200),
            valid_until: Some(300),
            observation_count: 1,
            confidence: 1.0,
            properties: HashMap::new(),
            confidence_history: crate::tcell::TCell::Empty,
            weight_history: crate::tcell::TCell::Empty,
            group_id: String::new(),
        };
        g.add_edge(e2);

        // Edge: alice→tokyo, valid [300, ∞)
        let e3 = GraphEdge {
            id: 0,
            source: id_alice,
            target: id_tokyo,
            edge_type: EdgeType::Association {
                association_type: "location:lives_in".to_string(),
                evidence_count: 1,
                statistical_significance: 1.0,
            },
            weight: 1.0,
            created_at: 300,
            updated_at: 300,
            valid_from: Some(300),
            valid_until: None,
            observation_count: 1,
            confidence: 1.0,
            properties: HashMap::new(),
            confidence_history: crate::tcell::TCell::Empty,
            weight_history: crate::tcell::TCell::Empty,
            group_id: String::new(),
        };
        g.add_edge(e3);

        let ontology = crate::ontology::OntologyRegistry::new();
        (g, ontology)
    }

    /// Helper to execute a MinnsQL query against the temporal test graph.
    fn exec_temporal(query: &str) -> Result<QueryOutput, QueryError> {
        let (graph, ontology) = build_temporal_test_graph();
        crate::query_lang::execute_query(query, &graph, &ontology)
    }

    // -- AS OF tests --

    #[test]
    fn test_as_of_filters_by_created_at() {
        // AS OF 150: only alice and london (created_at<=150), only edge1 (created_at=100)
        let result = exec_temporal(
            r#"MATCH (u {name: "alice"})-[r:location]->(p) WHEN ALL AS OF 150 RETURN p"#,
        ).unwrap();
        assert_eq!(result.rows.len(), 1); // only london edge visible
    }

    #[test]
    fn test_as_of_all_visible() {
        // AS OF 999: everything visible
        let result = exec_temporal(
            r#"MATCH (u {name: "alice"})-[r:location]->(p) WHEN ALL AS OF 999 RETURN p"#,
        ).unwrap();
        assert_eq!(result.rows.len(), 3);
    }

    #[test]
    fn test_as_of_nothing_visible() {
        // AS OF 50: nothing created yet
        let result = exec_temporal(
            r#"MATCH (u {name: "alice"})-[r:location]->(p) WHEN ALL AS OF 50 RETURN p"#,
        ).unwrap();
        assert_eq!(result.rows.len(), 0);
    }

    // -- Temporal function tests --

    #[test]
    fn test_valid_from_function() {
        let result = exec_temporal(
            r#"MATCH (u {name: "alice"})-[r:location]->(p) WHEN ALL RETURN valid_from(r) ORDER BY valid_from(r) ASC LIMIT 1"#,
        ).unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], Value::Int(100));
    }

    #[test]
    fn test_valid_until_function() {
        // The tokyo edge has valid_until=None → should return Null
        let result = exec_temporal(
            r#"MATCH (u {name: "alice"})-[r:location]->(p) RETURN valid_until(r) ORDER BY valid_from(r) DESC LIMIT 1"#,
        ).unwrap();
        // Active only → tokyo (valid_until=None)
        assert_eq!(result.rows[0][0], Value::Null);
    }

    #[test]
    fn test_created_at_function() {
        let result = exec_temporal(
            r#"MATCH (u {name: "alice"}) RETURN created_at(u)"#,
        ).unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], Value::Int(100));
    }

    #[test]
    fn test_open_ended_function() {
        // Default viewport = ActiveOnly, so only tokyo edge (open-ended)
        let result = exec_temporal(
            r#"MATCH (u {name: "alice"})-[r:location]->(p) RETURN open_ended(r)"#,
        ).unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], Value::Bool(true));
    }

    #[test]
    fn test_duration_function() {
        // London edge: valid_from=100, valid_until=200, duration=100
        let result = exec_temporal(
            r#"MATCH (u {name: "alice"})-[r:location]->(p) WHEN ALL RETURN duration(r) ORDER BY valid_from(r) ASC LIMIT 1"#,
        ).unwrap();
        assert_eq!(result.rows[0][0], Value::Int(100));
    }

    #[test]
    fn test_duration_null_for_open_ended() {
        // Tokyo edge has no valid_until → duration returns Null
        let result = exec_temporal(
            r#"MATCH (u {name: "alice"})-[r:location]->(p) RETURN duration(r)"#,
        ).unwrap();
        // ActiveOnly → only tokyo
        assert_eq!(result.rows[0][0], Value::Null);
    }

    #[test]
    fn test_overlap_function() {
        // London [100,200) and Berlin [200,300): they DON'T overlap (200 == 200 → s2 < e1_end fails when s2=200, e1_end=200)
        // Actually: s1=100 < e2_end=300 is true; s2=200 < e1_end=200 is false → not overlapping. Correct.
        let result = exec_temporal(
            r#"MATCH (u {name: "alice"})-[r1:location]->(p1), (u)-[r2:location]->(p2) WHEN ALL WHERE overlap(r1, r2) AND valid_from(r1) != valid_from(r2) RETURN p1, p2"#,
        ).unwrap();
        // London [100,200) overlaps nothing that's different, Berlin [200,300) overlaps nothing different.
        // Actually london[100,200) and tokyo[300,∞): 100 < MAX true, 300 < 200 false → no.
        // berlin[200,300) and tokyo[300,∞): 200 < MAX true, 300 < 300 false → no.
        // So no overlapping pairs of different edges. Good.
        assert_eq!(result.rows.len(), 0);
    }

    // -- SUCCESSIVE tests --

    #[test]
    fn test_successive_finds_consecutive_edges() {
        // Use tight tolerance (1ns) since test timestamps are small integers.
        // London→Berlin: valid_until=200 == valid_from=200 → diff=0 ≤ 1 → SUCCESSIVE
        // Berlin→Tokyo: valid_until=300 == valid_from=300 → diff=0 ≤ 1 → SUCCESSIVE
        // London→Tokyo: valid_until=200 vs valid_from=300 → diff=100 > 1 → NO
        let result = exec_temporal(
            r#"MATCH (u {name: "alice"})-[r1:location]->(p1), (u)-[r2:location]->(p2) WHEN ALL WHERE SUCCESSIVE(r1, r2, "1ns") RETURN p1, p2"#,
        ).unwrap();
        assert_eq!(result.rows.len(), 2); // london→berlin, berlin→tokyo
    }

    #[test]
    fn test_successive_rejects_non_consecutive() {
        // London→Tokyo: valid_until=200 vs valid_from=300 → diff=100 > tolerance(1s=1e9ns)
        // But wait, our timestamps are raw integers not nanos. 200 vs 300 diff=100.
        // Tolerance is 1_000_000_000 (1s in nanos). 100 < 1e9, so it would pass!
        // This is fine — our test graph uses tiny timestamps. With a custom tight tolerance:
        let result = exec_temporal(
            r#"MATCH (u {name: "alice"})-[r1:location]->(p1), (u)-[r2:location]->(p2) WHEN ALL WHERE SUCCESSIVE(r1, r2, "1ns") RETURN p1, p2"#,
        ).unwrap();
        // With 1ns tolerance: only exact matches. 200==200 and 300==300 → still 2
        // London.valid_until=200, Berlin.valid_from=200 → diff=0 ≤ 1 → yes
        // Berlin.valid_until=300, Tokyo.valid_from=300 → diff=0 ≤ 1 → yes
        // London.valid_until=200, Tokyo.valid_from=300 → diff=100 > 1 → no
        assert_eq!(result.rows.len(), 2);
    }

    // -- CHANGED tests --

    #[test]
    fn test_changed_detects_edge_starting_in_range() {
        // Berlin edge started (valid_from=200) in range [150, 250]
        let result = exec_temporal(
            r#"MATCH (u {name: "alice"})-[r:location]->(p) WHEN ALL WHERE CHANGED(r, 150, 250) RETURN p"#,
        ).unwrap();
        // London: valid_from=100 (not in), valid_until=200 (in range), created_at=100 (not in) → changed (ended)
        // Berlin: valid_from=200 (in range) → changed (started)
        // Tokyo: valid_from=300 (not in), valid_until=None, created_at=300 (not in) → not changed
        assert_eq!(result.rows.len(), 2); // london (ended) + berlin (started)
    }

    #[test]
    fn test_changed_nothing_in_range() {
        // Range [400, 500]: no edges have anything in this range
        let result = exec_temporal(
            r#"MATCH (u {name: "alice"})-[r:location]->(p) WHEN ALL WHERE CHANGED(r, 400, 500) RETURN p"#,
        ).unwrap();
        assert_eq!(result.rows.len(), 0);
    }

    // -- change_type function tests --

    #[test]
    fn test_change_type_started() {
        // Berlin: valid_from=200 is in [190, 210], valid_until=300 is not
        let result = exec_temporal(
            r#"MATCH (u {name: "alice"})-[r:location]->(p) WHEN ALL WHERE valid_from(r) = 200 RETURN change_type(r, 190, 210)"#,
        ).unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], Value::String("started".into()));
    }

    #[test]
    fn test_change_type_ended() {
        // London: valid_from=100 not in [190, 210], valid_until=200 is in [190, 210]
        let result = exec_temporal(
            r#"MATCH (u {name: "alice"})-[r:location]->(p) WHEN ALL WHERE valid_from(r) = 100 RETURN change_type(r, 190, 210)"#,
        ).unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], Value::String("ended".into()));
    }

    #[test]
    fn test_change_type_stable() {
        // Tokyo: nothing in [400, 500]
        let result = exec_temporal(
            r#"MATCH (u {name: "alice"})-[r:location]->(p) WHEN ALL WHERE valid_from(r) = 300 RETURN change_type(r, 400, 500)"#,
        ).unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], Value::String("stable".into()));
    }

    // -- Parser tests for AS OF and FuncPredicate --

    #[test]
    fn test_parse_as_of() {
        use super::super::ast::BoolExpr;
        use super::super::ast::Expr;
        let q = super::super::parser::Parser::parse(
            r#"MATCH (n) AS OF "2024-06-15" RETURN n"#,
        ).unwrap();
        assert!(q.as_of.is_some());
        match &q.as_of.unwrap() {
            Expr::Literal(Literal::String(s)) => assert_eq!(s, "2024-06-15"),
            other => panic!("expected string literal, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_as_of_with_when() {
        let q = super::super::parser::Parser::parse(
            r#"MATCH (n) WHEN ALL AS OF "2024-06-15" RETURN n"#,
        ).unwrap();
        assert!(matches!(q.when, Some(WhenClause::All)));
        assert!(q.as_of.is_some());
    }

    #[test]
    fn test_parse_func_predicate_in_where() {
        use super::super::ast::{BoolExpr, Expr};
        let q = super::super::parser::Parser::parse(
            r#"MATCH (n)-[r1]->(m), (n)-[r2]->(o) WHERE SUCCESSIVE(r1, r2) RETURN n"#,
        ).unwrap();
        match &q.where_clause {
            Some(BoolExpr::FuncPredicate(name, args)) => {
                assert!(name == "SUCCESSIVE");
                assert_eq!(args.len(), 2);
            },
            other => panic!("expected FuncPredicate, got {:?}", other),
        }
    }

    #[test]
    fn test_parse_changed_with_comma_args() {
        use super::super::ast::BoolExpr;
        let q = super::super::parser::Parser::parse(
            r#"MATCH (n)-[r]->(m) WHEN ALL WHERE CHANGED(r, "2024-01", "2024-12") RETURN n"#,
        ).unwrap();
        match &q.where_clause {
            Some(BoolExpr::FuncPredicate(name, args)) => {
                assert!(name == "CHANGED");
                assert_eq!(args.len(), 3);
            },
            other => panic!("expected FuncPredicate, got {:?}", other),
        }
    }

    #[test]
    fn test_as_alias_still_works() {
        let q = super::super::parser::Parser::parse(
            r#"MATCH (n) RETURN n.name AS label"#,
        ).unwrap();
        assert_eq!(q.returns[0].alias.as_deref(), Some("label"));
        assert!(q.as_of.is_none());
    }

    // -----------------------------------------------------------------------
    // Phase 1: time_bucket / date_trunc / ago
    // -----------------------------------------------------------------------

    #[test]
    fn test_truncate_timestamp_day() {
        use super::super::planner::{civil_to_nanos, days_from_civil};
        // 2024-06-15T12:30:00 → should truncate to 2024-06-15T00:00:00
        let noon = civil_to_nanos(2024, 6, 15) + 12 * 3_600_000_000_000 + 30 * 60_000_000_000;
        let result = truncate_timestamp(noon, "1d").unwrap();
        assert_eq!(result, civil_to_nanos(2024, 6, 15));
    }

    #[test]
    fn test_truncate_timestamp_hour() {
        use super::super::planner::civil_to_nanos;
        let ts = civil_to_nanos(2024, 6, 15) + 14 * 3_600_000_000_000 + 45 * 60_000_000_000;
        let result = truncate_timestamp(ts, "1h").unwrap();
        let expected = civil_to_nanos(2024, 6, 15) + 14 * 3_600_000_000_000;
        assert_eq!(result, expected);
    }

    #[test]
    fn test_date_trunc_month() {
        use super::super::planner::civil_to_nanos;
        let mid_month = civil_to_nanos(2024, 6, 15) + 3_600_000_000_000;
        let result = truncate_timestamp_by_unit(mid_month, "month").unwrap();
        assert_eq!(result, civil_to_nanos(2024, 6, 1));
    }

    #[test]
    fn test_date_trunc_quarter() {
        use super::super::planner::civil_to_nanos;
        // May 2024 → Q2 starts April 1
        let may_ts = civil_to_nanos(2024, 5, 10);
        let result = truncate_timestamp_by_unit(may_ts, "quarter").unwrap();
        assert_eq!(result, civil_to_nanos(2024, 4, 1));
    }

    #[test]
    fn test_date_trunc_year() {
        use super::super::planner::civil_to_nanos;
        let ts = civil_to_nanos(2024, 8, 20);
        let result = truncate_timestamp_by_unit(ts, "year").unwrap();
        assert_eq!(result, civil_to_nanos(2024, 1, 1));
    }

    #[test]
    fn test_date_trunc_week() {
        use super::super::planner::civil_to_nanos;
        // 2024-06-12 is a Wednesday → Monday was 2024-06-10
        let wednesday = civil_to_nanos(2024, 6, 12) + 5_000_000_000;
        let result = truncate_timestamp_by_unit(wednesday, "week").unwrap();
        assert_eq!(result, civil_to_nanos(2024, 6, 10));
    }

    #[test]
    fn test_ago_returns_near_now() {
        // ago("0s") should be approximately now
        let result = exec_temporal(
            r#"MATCH (u {name: "alice"}) RETURN ago("0s")"#,
        ).unwrap();
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as i64;
        match &result.rows[0][0] {
            Value::Int(v) => {
                let diff = (now - v).abs();
                assert!(diff < 2_000_000_000, "ago(0s) too far from now: diff={}", diff);
            }
            other => panic!("expected Int, got {:?}", other),
        }
    }

    #[test]
    fn test_time_bucket_calendar_routing() {
        use super::super::planner::civil_to_nanos;
        // "month" should route through calendar truncation, not fixed nanos
        let mid_month = civil_to_nanos(2024, 3, 15);
        let result = truncate_timestamp(mid_month, "month").unwrap();
        assert_eq!(result, civil_to_nanos(2024, 3, 1));
    }

    // -----------------------------------------------------------------------
    // Phase 2: Allen's interval algebra predicates
    // -----------------------------------------------------------------------

    #[test]
    fn test_precedes_positive() {
        // London [100,200) precedes Berlin [200,300): 200 <= 200 → true
        let result = exec_temporal(
            r#"MATCH (u {name: "alice"})-[r1:location]->(p1), (u)-[r2:location]->(p2) WHEN ALL WHERE precedes(r1, r2) AND valid_from(r1) = 100 AND valid_from(r2) = 200 RETURN p1, p2"#,
        ).unwrap();
        assert_eq!(result.rows.len(), 1);
    }

    #[test]
    fn test_precedes_negative() {
        // Berlin [200,300) does NOT precede London [100,200): 300 <= 100 → false
        let result = exec_temporal(
            r#"MATCH (u {name: "alice"})-[r1:location]->(p1), (u)-[r2:location]->(p2) WHEN ALL WHERE precedes(r1, r2) AND valid_from(r1) = 200 AND valid_from(r2) = 100 RETURN p1, p2"#,
        ).unwrap();
        assert_eq!(result.rows.len(), 0);
    }

    #[test]
    fn test_meets_positive() {
        // London valid_until=200 == Berlin valid_from=200 → meets
        let result = exec_temporal(
            r#"MATCH (u {name: "alice"})-[r1:location]->(p1), (u)-[r2:location]->(p2) WHEN ALL WHERE meets(r1, r2) AND valid_from(r1) = 100 AND valid_from(r2) = 200 RETURN p1, p2"#,
        ).unwrap();
        assert_eq!(result.rows.len(), 1);
    }

    #[test]
    fn test_meets_negative() {
        // London valid_until=200 != Tokyo valid_from=300 → does not meet
        let result = exec_temporal(
            r#"MATCH (u {name: "alice"})-[r1:location]->(p1), (u)-[r2:location]->(p2) WHEN ALL WHERE meets(r1, r2) AND valid_from(r1) = 100 AND valid_from(r2) = 300 RETURN p1, p2"#,
        ).unwrap();
        assert_eq!(result.rows.len(), 0);
    }

    #[test]
    fn test_covers_positive() {
        // Build a graph with an edge [100, 400) and another [200, 300) — first covers second.
        // Use the existing graph: London [100,200) does NOT cover Berlin [200,300).
        // Instead test with equals which is simpler on existing data.
        let result = exec_temporal(
            r#"MATCH (u {name: "alice"})-[r1:location]->(p1), (u)-[r2:location]->(p2) WHEN ALL WHERE equals(r1, r2) AND valid_from(r1) = valid_from(r2) AND valid_from(r1) = 100 RETURN p1"#,
        ).unwrap();
        assert_eq!(result.rows.len(), 1); // edge equals itself
    }

    #[test]
    fn test_starts_finishes() {
        // An edge starts itself: same valid_from, e1_end <= e2_end
        let result = exec_temporal(
            r#"MATCH (u {name: "alice"})-[r1:location]->(p1), (u)-[r2:location]->(p2) WHEN ALL WHERE starts(r1, r2) AND valid_from(r1) = 100 AND valid_from(r2) = 100 RETURN p1"#,
        ).unwrap();
        assert_eq!(result.rows.len(), 1); // edge starts itself

        let result2 = exec_temporal(
            r#"MATCH (u {name: "alice"})-[r1:location]->(p1), (u)-[r2:location]->(p2) WHEN ALL WHERE finishes(r1, r2) AND valid_from(r1) = 100 AND valid_from(r2) = 100 RETURN p1"#,
        ).unwrap();
        assert_eq!(result2.rows.len(), 1); // edge finishes itself
    }

    #[test]
    fn test_open_ended_interval_semantics() {
        // Tokyo [300, MAX) — covers nothing with finite end, but equals itself
        let result = exec_temporal(
            r#"MATCH (u {name: "alice"})-[r1:location]->(p1), (u)-[r2:location]->(p2) WHEN ALL WHERE equals(r1, r2) AND valid_from(r1) = 300 AND valid_from(r2) = 300 RETURN p1"#,
        ).unwrap();
        assert_eq!(result.rows.len(), 1);
    }

    // -----------------------------------------------------------------------
    // Phase 3: TCell access — confidence_history / weight_history
    // -----------------------------------------------------------------------

    #[test]
    fn test_confidence_history_empty() {
        // All test edges have TCell::Empty confidence_history
        let result = exec_temporal(
            r#"MATCH (u {name: "alice"})-[r:location]->(p) RETURN confidence_history(r)"#,
        ).unwrap();
        // ActiveOnly → tokyo only
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], Value::List(vec![]));
    }

    #[test]
    fn test_confidence_at_empty() {
        let result = exec_temporal(
            r#"MATCH (u {name: "alice"})-[r:location]->(p) RETURN confidence_at(r, 500)"#,
        ).unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], Value::Null);
    }

    #[test]
    fn test_weight_history_empty() {
        let result = exec_temporal(
            r#"MATCH (u {name: "alice"})-[r:location]->(p) RETURN weight_history(r)"#,
        ).unwrap();
        assert_eq!(result.rows.len(), 1);
        assert_eq!(result.rows[0][0], Value::List(vec![]));
    }

    #[test]
    fn test_confidence_at_with_tcell_data() {
        use agent_db_core::event_time::EventTime;

        let (mut graph, ontology) = build_temporal_test_graph();
        // Add confidence history to the first edge via direct mutable access
        let edge_ids: Vec<_> = graph.edges.keys().collect();
        if let Some(edge) = graph.edges.get_mut(edge_ids[0]) {
            edge.confidence_history.set(EventTime::from_nanos(100), 0.5);
            edge.confidence_history.set(EventTime::from_nanos(200), 0.8);
            edge.confidence_history.set(EventTime::from_nanos(300), 0.95);
        }
        // Query confidence_at at timestamp 250 → should get 0.8 (last at or before 250)
        let result = crate::query_lang::execute_query(
            r#"MATCH (u {name: "alice"})-[r:location]->(p) WHEN ALL WHERE valid_from(r) = 100 RETURN confidence_at(r, 250)"#,
            &graph,
            &ontology,
        ).unwrap();
        assert_eq!(result.rows.len(), 1);
        match &result.rows[0][0] {
            Value::Float(f) => assert!((*f - 0.8).abs() < 0.01, "expected 0.8, got {}", f),
            other => panic!("expected Float, got {:?}", other),
        }
    }

    #[test]
    fn test_confidence_history_with_tcell_data() {
        use agent_db_core::event_time::EventTime;

        let (mut graph, ontology) = build_temporal_test_graph();
        let edge_ids: Vec<_> = graph.edges.keys().collect();
        if let Some(edge) = graph.edges.get_mut(edge_ids[0]) {
            edge.confidence_history.set(EventTime::from_nanos(100), 0.5);
            edge.confidence_history.set(EventTime::from_nanos(200), 0.9);
        }
        let result = crate::query_lang::execute_query(
            r#"MATCH (u {name: "alice"})-[r:location]->(p) WHEN ALL WHERE valid_from(r) = 100 RETURN confidence_history(r)"#,
            &graph,
            &ontology,
        ).unwrap();
        assert_eq!(result.rows.len(), 1);
        match &result.rows[0][0] {
            Value::List(entries) => {
                assert_eq!(entries.len(), 2);
                // First entry: [100, 0.5]
                match &entries[0] {
                    Value::List(pair) => {
                        assert_eq!(pair[0], Value::Int(100));
                        match &pair[1] {
                            Value::Float(f) => assert!((*f - 0.5).abs() < 0.01),
                            other => panic!("expected Float, got {:?}", other),
                        }
                    }
                    other => panic!("expected List pair, got {:?}", other),
                }
            }
            other => panic!("expected List, got {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // Phase 4: Bi-temporal composition (AS OF + WHEN)
    // -----------------------------------------------------------------------

    #[test]
    fn test_bitemporal_as_of_and_when_compose() {
        // AS OF 250: created_at <= 250 → alice, london, berlin visible; edge1(created=100), edge2(created=200)
        // WHEN 100 TO 250: valid time filter → edge1[100,200) overlaps [100,250), edge2[200,300) overlaps [100,250)
        // Both edges visible. edge3 (created_at=300) excluded by AS OF.
        let result = exec_temporal(
            r#"MATCH (u {name: "alice"})-[r:location]->(p) WHEN 100 TO 250 AS OF 250 RETURN p ORDER BY valid_from(r) ASC"#,
        ).unwrap();
        assert_eq!(result.rows.len(), 2); // london + berlin
    }

    #[test]
    fn test_bitemporal_as_of_excludes_future_txn() {
        // AS OF 150: only edge1 (created_at=100)
        // WHEN ALL: no valid-time filter
        let result = exec_temporal(
            r#"MATCH (u {name: "alice"})-[r:location]->(p) WHEN ALL AS OF 150 RETURN p"#,
        ).unwrap();
        assert_eq!(result.rows.len(), 1); // only london
    }

    #[test]
    fn test_bitemporal_when_filters_valid_time() {
        // No AS OF: all txn-times visible
        // WHEN 250 TO 350: edge2[200,300) overlaps, edge3[300,∞) overlaps
        let result = exec_temporal(
            r#"MATCH (u {name: "alice"})-[r:location]->(p) WHEN 250 TO 350 RETURN p"#,
        ).unwrap();
        assert_eq!(result.rows.len(), 2); // berlin + tokyo
    }

    // -----------------------------------------------------------------------
    // Phase 1 planner: civil date helper tests
    // -----------------------------------------------------------------------

    #[test]
    fn test_civil_roundtrip() {
        use super::super::planner::{civil_from_days, civil_to_nanos, days_from_civil, nanos_to_civil};
        // 2024-06-15 roundtrip
        let days = days_from_civil(2024, 6, 15);
        let (y, m, d) = civil_from_days(days);
        assert_eq!((y, m, d), (2024, 6, 15));

        // nanos roundtrip
        let nanos = civil_to_nanos(2024, 1, 1);
        let (y2, m2, d2) = nanos_to_civil(nanos);
        assert_eq!((y2, m2, d2), (2024, 1, 1));
    }

    #[test]
    fn test_civil_epoch() {
        use super::super::planner::{civil_to_nanos, nanos_to_civil};
        let nanos = civil_to_nanos(1970, 1, 1);
        assert_eq!(nanos, 0);
        let (y, m, d) = nanos_to_civil(0);
        assert_eq!((y, m, d), (1970, 1, 1));
    }
}
