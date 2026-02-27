//! Temporal views of the graph.
//!
//! Provides snapshot and rolling-window views that filter nodes and edges
//! by their `created_at` timestamp without copying data.

use crate::structures::{Direction, EdgeId, Graph, GraphEdge, GraphNode, NodeId};
use agent_db_core::types::Timestamp;

// ============================================================================
// GraphAtSnapshot — point-in-time view
// ============================================================================

/// Snapshot view of the graph at a specific point in time.
///
/// Only nodes and edges with `created_at <= cutoff` are visible.
/// Edge visibility additionally requires both endpoint nodes to be visible.
pub struct GraphAtSnapshot<'a> {
    graph: &'a Graph,
    cutoff: Timestamp,
}

impl<'a> GraphAtSnapshot<'a> {
    pub fn new(graph: &'a Graph, cutoff: Timestamp) -> Self {
        Self { graph, cutoff }
    }

    pub fn cutoff(&self) -> Timestamp {
        self.cutoff
    }

    pub fn get_node(&self, node_id: NodeId) -> Option<&'a GraphNode> {
        self.graph
            .get_node(node_id)
            .filter(|n| n.created_at <= self.cutoff)
    }

    pub fn get_edge(&self, edge_id: EdgeId) -> Option<&'a GraphEdge> {
        self.graph.get_edge(edge_id).filter(|e| {
            e.created_at <= self.cutoff
                && self.get_node(e.source).is_some()
                && self.get_node(e.target).is_some()
        })
    }

    /// Outgoing neighbors visible at the snapshot time.
    pub fn get_neighbors(&self, node_id: NodeId) -> Vec<NodeId> {
        if self.get_node(node_id).is_none() {
            return Vec::new();
        }
        self.graph
            .get_edges_from(node_id)
            .into_iter()
            .filter(|e| e.created_at <= self.cutoff)
            .filter_map(|e| self.get_node(e.target).map(|_| e.target))
            .collect()
    }

    /// Direction-aware neighbors visible at the snapshot time.
    pub fn neighbors_directed(&self, node_id: NodeId, direction: Direction) -> Vec<NodeId> {
        if self.get_node(node_id).is_none() {
            return Vec::new();
        }
        match direction {
            Direction::Out => self.get_neighbors(node_id),
            Direction::In => self
                .graph
                .get_edges_to(node_id)
                .into_iter()
                .filter(|e| e.created_at <= self.cutoff)
                .filter_map(|e| self.get_node(e.source).map(|_| e.source))
                .collect(),
            Direction::Both => {
                let mut seen = std::collections::HashSet::new();
                let mut result = Vec::new();
                for n in self.get_neighbors(node_id) {
                    if seen.insert(n) {
                        result.push(n);
                    }
                }
                for e in self.graph.get_edges_to(node_id) {
                    if e.created_at <= self.cutoff {
                        if let Some(_node) = self.get_node(e.source) {
                            if seen.insert(e.source) {
                                result.push(e.source);
                            }
                        }
                    }
                }
                result
            },
        }
    }

    /// Number of nodes visible at the snapshot time.
    pub fn node_count(&self) -> usize {
        self.graph
            .temporal_index
            .range(..=self.cutoff)
            .flat_map(|(_, ids)| ids.iter())
            .filter(|&&nid| self.graph.nodes.contains_key(&nid))
            .count()
    }

    /// Nodes created in `[start, end]`, clamped by the snapshot cutoff.
    pub fn nodes_in_range(&self, start: Timestamp, end: Timestamp) -> Vec<&'a GraphNode> {
        let clamped_end = end.min(self.cutoff);
        if start > clamped_end {
            return Vec::new();
        }
        self.graph
            .temporal_index
            .range(start..=clamped_end)
            .flat_map(|(_, ids)| ids.iter())
            .filter_map(|&nid| self.graph.nodes.get(&nid))
            .collect()
    }
}

// ============================================================================
// RollingWindow — time-range view
// ============================================================================

/// Rolling time-window view of the graph.
///
/// Only nodes and edges with `created_at` in `[start, end]` are visible.
/// Edge visibility additionally requires both endpoint nodes to be visible.
pub struct RollingWindow<'a> {
    graph: &'a Graph,
    start: Timestamp,
    end: Timestamp,
}

impl<'a> RollingWindow<'a> {
    pub fn new(graph: &'a Graph, start: Timestamp, end: Timestamp) -> Self {
        Self { graph, start, end }
    }

    pub fn start(&self) -> Timestamp {
        self.start
    }

    pub fn end(&self) -> Timestamp {
        self.end
    }

    fn is_visible_node(&self, node: &GraphNode) -> bool {
        node.created_at >= self.start && node.created_at <= self.end
    }

    fn is_visible_edge(&self, edge: &GraphEdge) -> bool {
        edge.created_at >= self.start
            && edge.created_at <= self.end
            && self
                .graph
                .get_node(edge.source)
                .is_some_and(|n| self.is_visible_node(n))
            && self
                .graph
                .get_node(edge.target)
                .is_some_and(|n| self.is_visible_node(n))
    }

    /// All visible nodes in the window.
    pub fn nodes(&self) -> Vec<&'a GraphNode> {
        self.graph
            .temporal_index
            .range(self.start..=self.end)
            .flat_map(|(_, ids)| ids.iter())
            .filter_map(|&nid| self.graph.nodes.get(&nid))
            .collect()
    }

    /// All visible edges in the window.
    pub fn edges(&self) -> Vec<&'a GraphEdge> {
        self.graph
            .edges
            .values()
            .filter(|e| self.is_visible_edge(e))
            .collect()
    }

    pub fn node_count(&self) -> usize {
        self.graph
            .temporal_index
            .range(self.start..=self.end)
            .flat_map(|(_, ids)| ids.iter())
            .filter(|&&nid| self.graph.nodes.contains_key(&nid))
            .count()
    }

    pub fn edge_count(&self) -> usize {
        self.graph
            .edges
            .values()
            .filter(|e| self.is_visible_edge(e))
            .count()
    }

    /// Outgoing neighbors visible in the window.
    pub fn get_neighbors(&self, node_id: NodeId) -> Vec<NodeId> {
        if !self
            .graph
            .get_node(node_id)
            .is_some_and(|n| self.is_visible_node(n))
        {
            return Vec::new();
        }
        self.graph
            .get_edges_from(node_id)
            .into_iter()
            .filter(|e| self.is_visible_edge(e))
            .map(|e| e.target)
            .collect()
    }
}

// ============================================================================
// Convenience methods on Graph
// ============================================================================

impl Graph {
    /// Create a snapshot view of the graph at a specific point in time.
    pub fn at(&self, timestamp: Timestamp) -> GraphAtSnapshot<'_> {
        GraphAtSnapshot::new(self, timestamp)
    }

    /// Create a rolling window view using `latest_timestamp()` as logical "now".
    /// `duration_ns` is the window width in nanoseconds.
    pub fn rolling(&self, duration_ns: u64) -> RollingWindow<'_> {
        let end = self.latest_timestamp().unwrap_or(0);
        let start = end.saturating_sub(duration_ns);
        RollingWindow::new(self, start, end)
    }

    /// Create an explicit time-window view with inclusive bounds `[start, end]`.
    pub fn window(&self, start: Timestamp, end: Timestamp) -> RollingWindow<'_> {
        RollingWindow::new(self, start, end)
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::{EdgeType, GraphEdge, GraphNode, NodeType};
    use std::collections::HashMap;

    fn make_node(id: NodeId, created_at: Timestamp) -> GraphNode {
        GraphNode {
            id,
            node_type: NodeType::Event {
                event_id: id as u128,
                event_type: "test".to_string(),
                significance: 0.5,
            },
            created_at,
            updated_at: created_at,
            properties: HashMap::new(),
            degree: 0,
        }
    }

    fn make_edge(source: NodeId, target: NodeId, created_at: Timestamp) -> GraphEdge {
        GraphEdge {
            id: 0,
            source,
            target,
            edge_type: EdgeType::Causality {
                strength: 0.8,
                lag_ms: 100,
            },
            weight: 1.0,
            created_at,
            updated_at: created_at,
            valid_from: None,
            valid_until: None,
            observation_count: 1,
            confidence: 0.9,
            properties: HashMap::new(),
        }
    }

    /// Build a graph: n1(t=100) -> n2(t=200) -> n3(t=300), edge12(t=150), edge23(t=250)
    fn build_temporal_graph() -> Graph {
        let mut g = Graph::new();
        let mut n1 = make_node(0, 100);
        n1.id = 0;
        let mut n2 = make_node(0, 200);
        n2.id = 0;
        let mut n3 = make_node(0, 300);
        n3.id = 0;

        // Use add_node so IDs and indices are set properly
        // We need to set created_at before adding
        n1.created_at = 100;
        n1.updated_at = 100;
        let id1 = g.add_node(n1).unwrap();

        n2.created_at = 200;
        n2.updated_at = 200;
        let id2 = g.add_node(n2).unwrap();

        n3.created_at = 300;
        n3.updated_at = 300;
        let id3 = g.add_node(n3).unwrap();

        let e12 = make_edge(id1, id2, 150);
        g.add_edge(e12);

        let e23 = make_edge(id2, id3, 250);
        g.add_edge(e23);

        g
    }

    // ── GraphAtSnapshot tests ──

    #[test]
    fn snapshot_node_visible() {
        let g = build_temporal_graph();
        let snap = g.at(200);
        // Nodes at t=100 and t=200 should be visible
        assert!(snap.get_node(1).is_some());
        assert!(snap.get_node(2).is_some());
        // Node at t=300 should NOT be visible
        assert!(snap.get_node(3).is_none());
    }

    #[test]
    fn snapshot_node_invisible() {
        let g = build_temporal_graph();
        let snap = g.at(50);
        assert!(snap.get_node(1).is_none());
        assert!(snap.get_node(2).is_none());
    }

    #[test]
    fn snapshot_edge_filtered() {
        let g = build_temporal_graph();
        let snap = g.at(200);
        // Edge 1 (t=150) connects n1(t=100) and n2(t=200) — both visible at t=200
        assert!(snap.get_edge(1).is_some());
        // Edge 2 (t=250) is after cutoff
        assert!(snap.get_edge(2).is_none());
    }

    #[test]
    fn snapshot_neighbors() {
        let g = build_temporal_graph();
        let snap = g.at(200);
        let neighbors = snap.get_neighbors(1);
        assert_eq!(neighbors, vec![2]);
        // n2's outgoing to n3: edge at t=250 > cutoff, so no neighbors
        let neighbors2 = snap.get_neighbors(2);
        assert!(neighbors2.is_empty());
    }

    #[test]
    fn snapshot_node_count() {
        let g = build_temporal_graph();
        assert_eq!(g.at(100).node_count(), 1);
        assert_eq!(g.at(200).node_count(), 2);
        assert_eq!(g.at(300).node_count(), 3);
        assert_eq!(g.at(50).node_count(), 0);
    }

    #[test]
    fn snapshot_nodes_in_range() {
        let g = build_temporal_graph();
        let snap = g.at(300);
        let nodes = snap.nodes_in_range(150, 250);
        assert_eq!(nodes.len(), 1); // Only n2 (t=200) is in [150,250]
    }

    #[test]
    fn snapshot_directed_neighbors() {
        let g = build_temporal_graph();
        let snap = g.at(300);
        // n2 has outgoing to n3, incoming from n1
        let out = snap.neighbors_directed(2, Direction::Out);
        assert_eq!(out, vec![3]);
        let inc = snap.neighbors_directed(2, Direction::In);
        assert_eq!(inc, vec![1]);
        let both = snap.neighbors_directed(2, Direction::Both);
        assert_eq!(both.len(), 2);
    }

    // ── RollingWindow tests ──

    #[test]
    fn rolling_window_nodes() {
        let g = build_temporal_graph();
        let w = g.window(150, 250);
        let nodes = w.nodes();
        assert_eq!(nodes.len(), 1); // Only n2 (t=200)
    }

    #[test]
    fn rolling_window_edges() {
        let g = build_temporal_graph();
        // Window [100, 300]: all nodes visible, edges at t=150 and t=250
        let w = g.window(100, 300);
        let edges = w.edges();
        assert_eq!(edges.len(), 2);
    }

    #[test]
    fn rolling_window_node_count() {
        let g = build_temporal_graph();
        assert_eq!(g.window(100, 300).node_count(), 3);
        assert_eq!(g.window(200, 200).node_count(), 1);
        assert_eq!(g.window(400, 500).node_count(), 0);
    }

    #[test]
    fn rolling_window_edge_count() {
        let g = build_temporal_graph();
        // Only edge12 (t=150) has both endpoints in [100,200]
        assert_eq!(g.window(100, 200).edge_count(), 1);
    }

    #[test]
    fn rolling_window_neighbors() {
        let g = build_temporal_graph();
        let w = g.window(100, 300);
        let neighbors = w.get_neighbors(1);
        assert_eq!(neighbors, vec![2]);
    }

    // ── Convenience method tests ──

    #[test]
    fn graph_at_convenience() {
        let g = build_temporal_graph();
        let snap = g.at(200);
        assert_eq!(snap.cutoff(), 200);
        assert_eq!(snap.node_count(), 2);
    }

    #[test]
    fn graph_rolling_convenience() {
        let g = build_temporal_graph();
        // latest_timestamp = 300, rolling(100) => window [200, 300]
        let w = g.rolling(100);
        assert_eq!(w.start(), 200);
        assert_eq!(w.end(), 300);
        assert_eq!(w.node_count(), 2); // n2 (t=200), n3 (t=300)
    }

    #[test]
    fn graph_window_convenience() {
        let g = build_temporal_graph();
        let w = g.window(100, 200);
        assert_eq!(w.start(), 100);
        assert_eq!(w.end(), 200);
    }
}
