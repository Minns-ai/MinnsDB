//! Temporal Reachability (Taint Propagation) — causal chain discovery.
//!
//! Follows edges only forward in time, tracking `(origin_source, arrival_time)`
//! per reached node. Useful for:
//! - Causal chain discovery: "what downstream effects did event X trigger?"
//! - Information diffusion: "how far did a signal propagate?"
//! - Taint analysis: "which nodes were influenced by a compromised source?"
//!
//! The algorithm uses a BFS-like priority queue ordered by arrival time,
//! ensuring that each node records the earliest possible arrival.

use crate::structures::{Graph, NodeId};
use crate::GraphResult;
use agent_db_core::types::Timestamp;
use std::cmp::Reverse;
use std::collections::{BinaryHeap, HashMap};

/// Configuration for temporal reachability analysis.
#[derive(Debug, Clone)]
pub struct TemporalReachabilityConfig {
    /// Maximum number of hops from the source (0 = unlimited).
    pub max_hops: usize,

    /// Only follow edges created at or after this timestamp (0 = no lower bound).
    pub time_start: Timestamp,

    /// Only follow edges created at or before this timestamp (0 = no upper bound).
    pub time_end: Timestamp,

    /// Minimum edge weight to traverse (edges below this are ignored).
    pub min_edge_weight: f32,
}

impl Default for TemporalReachabilityConfig {
    fn default() -> Self {
        Self {
            max_hops: 0,
            time_start: 0,
            time_end: 0,
            min_edge_weight: 0.0,
        }
    }
}

/// A single reachability record for a visited node.
#[derive(Debug, Clone)]
pub struct ReachabilityRecord {
    /// The original source node that initiated the propagation.
    pub origin: NodeId,

    /// The earliest arrival time at this node.
    pub arrival_time: Timestamp,

    /// Number of hops from the origin to reach this node.
    pub hops: usize,

    /// The predecessor node on the shortest-time path from origin.
    pub predecessor: Option<NodeId>,
}

/// Result of temporal reachability analysis.
#[derive(Debug, Clone)]
pub struct TemporalReachabilityResult {
    /// Per-node reachability records (only nodes reachable from the source).
    pub reachable: HashMap<NodeId, ReachabilityRecord>,

    /// Nodes ordered by arrival time (earliest first).
    pub arrival_order: Vec<NodeId>,

    /// Maximum depth (hops) reached during propagation.
    pub max_depth: usize,

    /// Total number of edges traversed.
    pub edges_traversed: usize,
}

/// Temporal reachability / taint propagation algorithm.
pub struct TemporalReachability {
    config: TemporalReachabilityConfig,
}

impl TemporalReachability {
    /// Create with default config.
    pub fn new() -> Self {
        Self {
            config: TemporalReachabilityConfig::default(),
        }
    }

    /// Create with custom config.
    pub fn with_config(config: TemporalReachabilityConfig) -> Self {
        Self { config }
    }

    /// Run temporal reachability from a single source node.
    ///
    /// The algorithm explores outgoing edges in arrival-time order, only
    /// following edges whose `created_at >= predecessor's arrival_time`
    /// (temporal monotonicity). Each node records the earliest arrival.
    pub fn propagate(
        &self,
        graph: &Graph,
        source: NodeId,
    ) -> GraphResult<TemporalReachabilityResult> {
        let source_node = graph.get_node(source).ok_or_else(|| {
            crate::GraphError::NodeNotFound(format!("source node {} not found", source))
        })?;

        let mut reachable: HashMap<NodeId, ReachabilityRecord> = HashMap::new();
        let mut arrival_order: Vec<NodeId> = Vec::new();
        let mut edges_traversed: usize = 0;
        let mut max_depth: usize = 0;

        // Seed: the source itself arrives at its own created_at timestamp
        let source_arrival = source_node.created_at;
        reachable.insert(
            source,
            ReachabilityRecord {
                origin: source,
                arrival_time: source_arrival,
                hops: 0,
                predecessor: None,
            },
        );
        arrival_order.push(source);

        // Min-heap ordered by (arrival_time, node_id) for determinism
        // Entry: (arrival_time, hops, node_id)
        let mut frontier: BinaryHeap<Reverse<(Timestamp, usize, NodeId)>> = BinaryHeap::new();
        frontier.push(Reverse((source_arrival, 0, source)));

        while let Some(Reverse((current_arrival, current_hops, current_node))) = frontier.pop() {
            // Skip if we already found a better (earlier) arrival for this node
            if let Some(record) = reachable.get(&current_node) {
                if record.arrival_time < current_arrival {
                    continue;
                }
            }

            // Hop limit check
            if self.config.max_hops > 0 && current_hops >= self.config.max_hops {
                continue;
            }

            // Explore outgoing edges
            for edge in graph.get_edges_from(current_node) {
                edges_traversed += 1;

                // Temporal monotonicity: edge must be created at or after current arrival
                if edge.created_at < current_arrival {
                    continue;
                }

                // Time window filter
                if self.config.time_start > 0 && edge.created_at < self.config.time_start {
                    continue;
                }
                if self.config.time_end > 0 && edge.created_at > self.config.time_end {
                    continue;
                }

                // Weight filter
                if edge.weight < self.config.min_edge_weight {
                    continue;
                }

                let target = edge.target;
                let new_arrival = edge.created_at;
                let new_hops = current_hops + 1;

                // Only update if this is the earliest arrival at the target
                let dominated = reachable
                    .get(&target)
                    .is_some_and(|r| r.arrival_time <= new_arrival);

                if !dominated {
                    let is_new = !reachable.contains_key(&target);
                    reachable.insert(
                        target,
                        ReachabilityRecord {
                            origin: source,
                            arrival_time: new_arrival,
                            hops: new_hops,
                            predecessor: Some(current_node),
                        },
                    );
                    if is_new {
                        arrival_order.push(target);
                    }
                    if new_hops > max_depth {
                        max_depth = new_hops;
                    }
                    frontier.push(Reverse((new_arrival, new_hops, target)));
                }
            }
        }

        Ok(TemporalReachabilityResult {
            reachable,
            arrival_order,
            max_depth,
            edges_traversed,
        })
    }

    /// Run temporal reachability from multiple source nodes simultaneously.
    ///
    /// Each node records which original source reached it first (earliest
    /// arrival wins). Useful for multi-source taint analysis.
    pub fn propagate_multi(
        &self,
        graph: &Graph,
        sources: &[NodeId],
    ) -> GraphResult<TemporalReachabilityResult> {
        let mut reachable: HashMap<NodeId, ReachabilityRecord> = HashMap::new();
        let mut arrival_order: Vec<NodeId> = Vec::new();
        let mut edges_traversed: usize = 0;
        let mut max_depth: usize = 0;

        let mut frontier: BinaryHeap<Reverse<(Timestamp, usize, NodeId, NodeId)>> =
            BinaryHeap::new();

        // Seed all sources
        for &src in sources {
            let src_node = graph.get_node(src).ok_or_else(|| {
                crate::GraphError::NodeNotFound(format!("source node {} not found", src))
            })?;
            let arrival = src_node.created_at;

            let dominated = reachable
                .get(&src)
                .is_some_and(|r| r.arrival_time <= arrival);

            if !dominated {
                let is_new = !reachable.contains_key(&src);
                reachable.insert(
                    src,
                    ReachabilityRecord {
                        origin: src,
                        arrival_time: arrival,
                        hops: 0,
                        predecessor: None,
                    },
                );
                if is_new {
                    arrival_order.push(src);
                }
                frontier.push(Reverse((arrival, 0, src, src)));
            }
        }

        while let Some(Reverse((current_arrival, current_hops, current_node, origin))) =
            frontier.pop()
        {
            // Skip if we already found a better arrival
            if let Some(record) = reachable.get(&current_node) {
                if record.arrival_time < current_arrival {
                    continue;
                }
                // If same arrival but different origin, skip (first-come wins)
                if record.arrival_time == current_arrival && record.origin != origin {
                    continue;
                }
            }

            if self.config.max_hops > 0 && current_hops >= self.config.max_hops {
                continue;
            }

            for edge in graph.get_edges_from(current_node) {
                edges_traversed += 1;

                if edge.created_at < current_arrival {
                    continue;
                }
                if self.config.time_start > 0 && edge.created_at < self.config.time_start {
                    continue;
                }
                if self.config.time_end > 0 && edge.created_at > self.config.time_end {
                    continue;
                }
                if edge.weight < self.config.min_edge_weight {
                    continue;
                }

                let target = edge.target;
                let new_arrival = edge.created_at;
                let new_hops = current_hops + 1;

                let dominated = reachable
                    .get(&target)
                    .is_some_and(|r| r.arrival_time <= new_arrival);

                if !dominated {
                    let is_new = !reachable.contains_key(&target);
                    reachable.insert(
                        target,
                        ReachabilityRecord {
                            origin,
                            arrival_time: new_arrival,
                            hops: new_hops,
                            predecessor: Some(current_node),
                        },
                    );
                    if is_new {
                        arrival_order.push(target);
                    }
                    if new_hops > max_depth {
                        max_depth = new_hops;
                    }
                    frontier.push(Reverse((new_arrival, new_hops, target, origin)));
                }
            }
        }

        Ok(TemporalReachabilityResult {
            reachable,
            arrival_order,
            max_depth,
            edges_traversed,
        })
    }

    /// Reconstruct the causal path from origin to a specific target node.
    /// Returns None if the target was not reached.
    pub fn causal_path(
        &self,
        result: &TemporalReachabilityResult,
        target: NodeId,
    ) -> Option<Vec<NodeId>> {
        let record = result.reachable.get(&target)?;
        let mut path = Vec::new();
        let mut current = target;

        path.push(current);
        while let Some(pred) = result.reachable.get(&current).and_then(|r| r.predecessor) {
            path.push(pred);
            current = pred;
            if current == record.origin {
                break;
            }
        }

        path.reverse();
        Some(path)
    }
}

impl Default for TemporalReachability {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::{EdgeType, GraphEdge, GraphNode, NodeType};

    fn make_node_at(event_id: u128, ts: Timestamp) -> GraphNode {
        let mut node = GraphNode::new(NodeType::Event {
            event_id,
            event_type: "test".into(),
            significance: 0.5,
        });
        node.created_at = ts;
        node.updated_at = ts;
        node
    }

    fn make_temporal_edge(source: NodeId, target: NodeId, ts: Timestamp, weight: f32) -> GraphEdge {
        let mut edge = GraphEdge::new(
            source,
            target,
            EdgeType::Temporal {
                average_interval_ms: 100,
                sequence_confidence: 0.9,
            },
            weight,
        );
        edge.created_at = ts;
        edge.updated_at = ts;
        edge
    }

    #[test]
    fn test_linear_chain() {
        // A(t=100) -> B(t=200) -> C(t=300), each edge at its target's time
        let mut graph = Graph::new();
        let a = graph.add_node(make_node_at(1, 100)).unwrap();
        let b = graph.add_node(make_node_at(2, 200)).unwrap();
        let c = graph.add_node(make_node_at(3, 300)).unwrap();

        graph.add_edge(make_temporal_edge(a, b, 200, 1.0));
        graph.add_edge(make_temporal_edge(b, c, 300, 1.0));

        let tr = TemporalReachability::new();
        let result = tr.propagate(&graph, a).unwrap();

        assert_eq!(result.reachable.len(), 3); // a, b, c all reachable
        assert_eq!(result.max_depth, 2);

        let rec_b = &result.reachable[&b];
        assert_eq!(rec_b.arrival_time, 200);
        assert_eq!(rec_b.hops, 1);
        assert_eq!(rec_b.predecessor, Some(a));

        let rec_c = &result.reachable[&c];
        assert_eq!(rec_c.arrival_time, 300);
        assert_eq!(rec_c.hops, 2);
        assert_eq!(rec_c.predecessor, Some(b));
    }

    #[test]
    fn test_backward_edge_blocked() {
        // A(t=100) -> B(t=200), but edge created at t=50 (before A's arrival)
        let mut graph = Graph::new();
        let a = graph.add_node(make_node_at(1, 100)).unwrap();
        let b = graph.add_node(make_node_at(2, 200)).unwrap();

        graph.add_edge(make_temporal_edge(a, b, 50, 1.0)); // backwards in time

        let tr = TemporalReachability::new();
        let result = tr.propagate(&graph, a).unwrap();

        // B should NOT be reachable because edge is before A's arrival
        assert_eq!(result.reachable.len(), 1); // only A
        assert!(!result.reachable.contains_key(&b));
    }

    #[test]
    fn test_max_hops_limit() {
        // A -> B -> C -> D, limit to 2 hops
        let mut graph = Graph::new();
        let a = graph.add_node(make_node_at(1, 100)).unwrap();
        let b = graph.add_node(make_node_at(2, 200)).unwrap();
        let c = graph.add_node(make_node_at(3, 300)).unwrap();
        let d = graph.add_node(make_node_at(4, 400)).unwrap();

        graph.add_edge(make_temporal_edge(a, b, 200, 1.0));
        graph.add_edge(make_temporal_edge(b, c, 300, 1.0));
        graph.add_edge(make_temporal_edge(c, d, 400, 1.0));

        let tr = TemporalReachability::with_config(TemporalReachabilityConfig {
            max_hops: 2,
            ..Default::default()
        });
        let result = tr.propagate(&graph, a).unwrap();

        assert!(result.reachable.contains_key(&a));
        assert!(result.reachable.contains_key(&b));
        assert!(result.reachable.contains_key(&c));
        assert!(!result.reachable.contains_key(&d)); // beyond 2 hops
    }

    #[test]
    fn test_diamond_earliest_arrival() {
        //     B(t=200)
        //    / \
        // A(t=100) D(t=400)
        //    \ /
        //     C(t=300)
        // A->B at t=200, A->C at t=300
        // B->D at t=350, C->D at t=400
        // D should be reached via B (earlier: t=350)
        let mut graph = Graph::new();
        let a = graph.add_node(make_node_at(1, 100)).unwrap();
        let b = graph.add_node(make_node_at(2, 200)).unwrap();
        let c = graph.add_node(make_node_at(3, 300)).unwrap();
        let d = graph.add_node(make_node_at(4, 400)).unwrap();

        graph.add_edge(make_temporal_edge(a, b, 200, 1.0));
        graph.add_edge(make_temporal_edge(a, c, 300, 1.0));
        graph.add_edge(make_temporal_edge(b, d, 350, 1.0));
        graph.add_edge(make_temporal_edge(c, d, 400, 1.0));

        let tr = TemporalReachability::new();
        let result = tr.propagate(&graph, a).unwrap();

        let rec_d = &result.reachable[&d];
        assert_eq!(rec_d.arrival_time, 350); // via B, not C
        assert_eq!(rec_d.predecessor, Some(b));
    }

    #[test]
    fn test_weight_filter() {
        let mut graph = Graph::new();
        let a = graph.add_node(make_node_at(1, 100)).unwrap();
        let b = graph.add_node(make_node_at(2, 200)).unwrap();
        let c = graph.add_node(make_node_at(3, 300)).unwrap();

        graph.add_edge(make_temporal_edge(a, b, 200, 0.1)); // weak
        graph.add_edge(make_temporal_edge(a, c, 300, 0.8)); // strong

        let tr = TemporalReachability::with_config(TemporalReachabilityConfig {
            min_edge_weight: 0.5,
            ..Default::default()
        });
        let result = tr.propagate(&graph, a).unwrap();

        assert!(!result.reachable.contains_key(&b)); // filtered out
        assert!(result.reachable.contains_key(&c)); // passes filter
    }

    #[test]
    fn test_multi_source() {
        // Two sources: A(t=100), B(t=200) both reach C(t=300)
        // A->C at t=250, B->C at t=300
        // A reaches C earlier
        let mut graph = Graph::new();
        let a = graph.add_node(make_node_at(1, 100)).unwrap();
        let b = graph.add_node(make_node_at(2, 200)).unwrap();
        let c = graph.add_node(make_node_at(3, 300)).unwrap();

        graph.add_edge(make_temporal_edge(a, c, 250, 1.0));
        graph.add_edge(make_temporal_edge(b, c, 300, 1.0));

        let tr = TemporalReachability::new();
        let result = tr.propagate_multi(&graph, &[a, b]).unwrap();

        let rec_c = &result.reachable[&c];
        assert_eq!(rec_c.origin, a); // A arrived first
        assert_eq!(rec_c.arrival_time, 250);
    }

    #[test]
    fn test_causal_path_reconstruction() {
        let mut graph = Graph::new();
        let a = graph.add_node(make_node_at(1, 100)).unwrap();
        let b = graph.add_node(make_node_at(2, 200)).unwrap();
        let c = graph.add_node(make_node_at(3, 300)).unwrap();

        graph.add_edge(make_temporal_edge(a, b, 200, 1.0));
        graph.add_edge(make_temporal_edge(b, c, 300, 1.0));

        let tr = TemporalReachability::new();
        let result = tr.propagate(&graph, a).unwrap();
        let path = tr.causal_path(&result, c).unwrap();

        assert_eq!(path, vec![a, b, c]);
    }

    #[test]
    fn test_empty_graph_source_not_found() {
        let graph = Graph::new();
        let tr = TemporalReachability::new();
        let result = tr.propagate(&graph, 999);
        assert!(result.is_err());
    }

    #[test]
    fn test_isolated_node() {
        let mut graph = Graph::new();
        let a = graph.add_node(make_node_at(1, 100)).unwrap();

        let tr = TemporalReachability::new();
        let result = tr.propagate(&graph, a).unwrap();

        assert_eq!(result.reachable.len(), 1);
        assert_eq!(result.max_depth, 0);
        assert_eq!(result.edges_traversed, 0);
    }

    #[test]
    fn test_time_window_filter() {
        let mut graph = Graph::new();
        let a = graph.add_node(make_node_at(1, 100)).unwrap();
        let b = graph.add_node(make_node_at(2, 200)).unwrap();
        let c = graph.add_node(make_node_at(3, 500)).unwrap();

        graph.add_edge(make_temporal_edge(a, b, 200, 1.0));
        graph.add_edge(make_temporal_edge(b, c, 500, 1.0)); // outside window

        let tr = TemporalReachability::with_config(TemporalReachabilityConfig {
            time_end: 400, // cut off before c
            ..Default::default()
        });
        let result = tr.propagate(&graph, a).unwrap();

        assert!(result.reachable.contains_key(&b));
        assert!(!result.reachable.contains_key(&c)); // outside time window
    }
}
