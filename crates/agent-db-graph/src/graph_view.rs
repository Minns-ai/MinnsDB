//! GraphView trait: composable, zero-copy graph views.
//!
//! Any type that implements GraphView can be used with the traversal
//! engine, algorithms, and analytics — including filtered views,
//! time-windowed views, and sharded snapshots.

use crate::structures::{Direction, EdgeId, Graph, GraphEdge, GraphNode, NodeId};

/// Core trait for read-only graph access.
///
/// Implementations include:
/// - `Graph` (full graph, direct access)
/// - `GraphAtSnapshot<'a>` (point-in-time filter)
/// - `RollingWindow<'a>` (time-range filter)
/// - `EdgeTypeFilter<'a, G>` (edge-type filter, composes with any G)
/// - `NodeSubgraph<'a, G>` (node-subset filter, composes with any G)
///
/// All methods have default implementations that delegate through
/// the primitive methods (get_node, get_edge, neighbors), so most
/// implementors only need to implement the core methods.
pub trait GraphView {
    /// Get a node by ID, or None if not visible in this view.
    fn view_node(&self, node_id: NodeId) -> Option<&GraphNode>;

    /// Get an edge by ID, or None if not visible in this view.
    fn view_edge(&self, edge_id: EdgeId) -> Option<&GraphEdge>;

    /// Get outgoing edge IDs for a node visible in this view.
    fn out_edge_ids(&self, node_id: NodeId) -> Box<dyn Iterator<Item = EdgeId> + '_>;

    /// Get incoming edge IDs for a node visible in this view.
    fn in_edge_ids(&self, node_id: NodeId) -> Box<dyn Iterator<Item = EdgeId> + '_>;

    /// Number of visible nodes.
    fn view_node_count(&self) -> usize;

    /// Number of visible edges.
    fn view_edge_count(&self) -> usize;

    /// Iterator over all visible node IDs.
    fn view_node_ids(&self) -> Box<dyn Iterator<Item = NodeId> + '_>;

    // === Derived methods with default implementations ===

    /// Get outgoing neighbors of a node.
    fn view_neighbors(&self, node_id: NodeId) -> Vec<NodeId> {
        self.out_edge_ids(node_id)
            .filter_map(|eid| self.view_edge(eid).map(|e| e.target))
            .collect()
    }

    /// Get incoming neighbors of a node.
    fn view_incoming_neighbors(&self, node_id: NodeId) -> Vec<NodeId> {
        self.in_edge_ids(node_id)
            .filter_map(|eid| self.view_edge(eid).map(|e| e.source))
            .collect()
    }

    /// Direction-aware neighbors.
    fn view_neighbors_directed(&self, node_id: NodeId, direction: Direction) -> Vec<NodeId> {
        match direction {
            Direction::Out => self.view_neighbors(node_id),
            Direction::In => self.view_incoming_neighbors(node_id),
            Direction::Both => {
                let mut seen = std::collections::HashSet::new();
                let mut result = Vec::new();
                for n in self.view_neighbors(node_id) {
                    if seen.insert(n) {
                        result.push(n);
                    }
                }
                for n in self.view_incoming_neighbors(node_id) {
                    if seen.insert(n) {
                        result.push(n);
                    }
                }
                result
            },
        }
    }

    /// Get edges from a source node.
    fn view_edges_from(&self, source: NodeId) -> Vec<&GraphEdge> {
        self.out_edge_ids(source)
            .filter_map(|eid| self.view_edge(eid))
            .collect()
    }

    /// Get edges to a target node.
    fn view_edges_to(&self, target: NodeId) -> Vec<&GraphEdge> {
        self.in_edge_ids(target)
            .filter_map(|eid| self.view_edge(eid))
            .collect()
    }

    /// Get edge between two specific nodes.
    fn view_edge_between(&self, source: NodeId, target: NodeId) -> Option<&GraphEdge> {
        self.out_edge_ids(source)
            .filter_map(|eid| self.view_edge(eid))
            .find(|e| e.target == target)
    }

    /// Check if a node exists in this view.
    fn has_node(&self, node_id: NodeId) -> bool {
        self.view_node(node_id).is_some()
    }
}

// ── Implementation for Graph ────────────────────────────────────────────

impl GraphView for Graph {
    #[inline]
    fn view_node(&self, node_id: NodeId) -> Option<&GraphNode> {
        self.nodes.get(node_id)
    }

    #[inline]
    fn view_edge(&self, edge_id: EdgeId) -> Option<&GraphEdge> {
        self.edges.get(edge_id)
    }

    fn out_edge_ids(&self, node_id: NodeId) -> Box<dyn Iterator<Item = EdgeId> + '_> {
        match self.adjacency_out.get(node_id) {
            Some(adj) => Box::new(adj.iter().copied()),
            None => Box::new(std::iter::empty()),
        }
    }

    fn in_edge_ids(&self, node_id: NodeId) -> Box<dyn Iterator<Item = EdgeId> + '_> {
        match self.adjacency_in.get(node_id) {
            Some(adj) => Box::new(adj.iter().copied()),
            None => Box::new(std::iter::empty()),
        }
    }

    fn view_node_count(&self) -> usize {
        self.nodes.len()
    }

    fn view_edge_count(&self) -> usize {
        self.edges.len()
    }

    fn view_node_ids(&self) -> Box<dyn Iterator<Item = NodeId> + '_> {
        Box::new(self.nodes.keys())
    }
}

// ── Implementation for GraphAtSnapshot ──────────────────────────────────

impl<'a> GraphView for crate::temporal_view::GraphAtSnapshot<'a> {
    fn view_node(&self, node_id: NodeId) -> Option<&GraphNode> {
        self.get_node(node_id)
    }

    fn view_edge(&self, edge_id: EdgeId) -> Option<&GraphEdge> {
        self.get_edge(edge_id)
    }

    fn out_edge_ids(&self, node_id: NodeId) -> Box<dyn Iterator<Item = EdgeId> + '_> {
        if self.get_node(node_id).is_none() {
            return Box::new(std::iter::empty());
        }
        let cutoff = self.cutoff();
        let graph = self.graph();
        match graph.adjacency_out.get(node_id) {
            Some(adj) => {
                Box::new(adj.iter().copied().filter(move |&eid| {
                    graph.edges.get(eid).is_some_and(|e| e.created_at <= cutoff)
                }))
            },
            None => Box::new(std::iter::empty()),
        }
    }

    fn in_edge_ids(&self, node_id: NodeId) -> Box<dyn Iterator<Item = EdgeId> + '_> {
        if self.get_node(node_id).is_none() {
            return Box::new(std::iter::empty());
        }
        let cutoff = self.cutoff();
        let graph = self.graph();
        match graph.adjacency_in.get(node_id) {
            Some(adj) => {
                Box::new(adj.iter().copied().filter(move |&eid| {
                    graph.edges.get(eid).is_some_and(|e| e.created_at <= cutoff)
                }))
            },
            None => Box::new(std::iter::empty()),
        }
    }

    fn view_node_count(&self) -> usize {
        self.node_count()
    }

    fn view_edge_count(&self) -> usize {
        // Count edges visible at the snapshot time
        let cutoff = self.cutoff();
        self.graph()
            .edges
            .values()
            .filter(|e| {
                e.created_at <= cutoff
                    && self.get_node(e.source).is_some()
                    && self.get_node(e.target).is_some()
            })
            .count()
    }

    fn view_node_ids(&self) -> Box<dyn Iterator<Item = NodeId> + '_> {
        let cutoff = self.cutoff();
        Box::new(
            self.graph()
                .temporal_index
                .range(..=cutoff)
                .flat_map(|(_, ids)| ids.iter().copied())
                .filter(move |&nid| self.graph().nodes.contains_key(nid)),
        )
    }
}

// ── Implementation for RollingWindow ────────────────────────────────────

impl<'a> GraphView for crate::temporal_view::RollingWindow<'a> {
    fn view_node(&self, node_id: NodeId) -> Option<&GraphNode> {
        self.graph()
            .get_node(node_id)
            .filter(|n| n.created_at >= self.start() && n.created_at <= self.end())
    }

    fn view_edge(&self, edge_id: EdgeId) -> Option<&GraphEdge> {
        self.graph().get_edge(edge_id).filter(|e| {
            e.created_at >= self.start()
                && e.created_at <= self.end()
                && self.view_node(e.source).is_some()
                && self.view_node(e.target).is_some()
        })
    }

    fn out_edge_ids(&self, node_id: NodeId) -> Box<dyn Iterator<Item = EdgeId> + '_> {
        // Source node must be visible in the window
        if self.view_node(node_id).is_none() {
            return Box::new(std::iter::empty());
        }
        let start = self.start();
        let end = self.end();
        let graph = self.graph();
        match graph.adjacency_out.get(node_id) {
            Some(adj) => Box::new(adj.iter().copied().filter(move |&eid| {
                graph.edges.get(eid).is_some_and(|e| {
                    e.created_at >= start
                        && e.created_at <= end
                        && graph
                            .get_node(e.target)
                            .is_some_and(|n| n.created_at >= start && n.created_at <= end)
                })
            })),
            None => Box::new(std::iter::empty()),
        }
    }

    fn in_edge_ids(&self, node_id: NodeId) -> Box<dyn Iterator<Item = EdgeId> + '_> {
        // Target node (node_id) must be visible in the window
        if self.view_node(node_id).is_none() {
            return Box::new(std::iter::empty());
        }
        let start = self.start();
        let end = self.end();
        let graph = self.graph();
        match graph.adjacency_in.get(node_id) {
            Some(adj) => Box::new(adj.iter().copied().filter(move |&eid| {
                graph.edges.get(eid).is_some_and(|e| {
                    e.created_at >= start
                        && e.created_at <= end
                        && graph
                            .get_node(e.source)
                            .is_some_and(|n| n.created_at >= start && n.created_at <= end)
                })
            })),
            None => Box::new(std::iter::empty()),
        }
    }

    fn view_node_count(&self) -> usize {
        self.node_count()
    }

    fn view_edge_count(&self) -> usize {
        self.edge_count()
    }

    fn view_node_ids(&self) -> Box<dyn Iterator<Item = NodeId> + '_> {
        let start = self.start();
        let end = self.end();
        Box::new(
            self.graph()
                .temporal_index
                .range(start..=end)
                .flat_map(|(_, ids)| ids.iter().copied())
                .filter(move |&nid| self.graph().nodes.contains_key(nid)),
        )
    }
}

// ── Composable Filters ──────────────────────────────────────────────────

/// Edge-type filter view: only shows edges of specified types.
pub struct EdgeTypeFilter<'a, G: GraphView> {
    inner: &'a G,
    allowed_discriminants:
        std::collections::HashSet<std::mem::Discriminant<crate::structures::EdgeType>>,
}

impl<'a, G: GraphView> EdgeTypeFilter<'a, G> {
    pub fn new(
        inner: &'a G,
        allowed_discriminants: std::collections::HashSet<
            std::mem::Discriminant<crate::structures::EdgeType>,
        >,
    ) -> Self {
        Self {
            inner,
            allowed_discriminants,
        }
    }
}

impl<'a, G: GraphView> GraphView for EdgeTypeFilter<'a, G> {
    fn view_node(&self, node_id: NodeId) -> Option<&GraphNode> {
        self.inner.view_node(node_id)
    }

    fn view_edge(&self, edge_id: EdgeId) -> Option<&GraphEdge> {
        self.inner.view_edge(edge_id).filter(|e| {
            self.allowed_discriminants
                .contains(&std::mem::discriminant(&e.edge_type))
        })
    }

    fn out_edge_ids(&self, node_id: NodeId) -> Box<dyn Iterator<Item = EdgeId> + '_> {
        Box::new(
            self.inner
                .out_edge_ids(node_id)
                .filter(|&eid| self.view_edge(eid).is_some()),
        )
    }

    fn in_edge_ids(&self, node_id: NodeId) -> Box<dyn Iterator<Item = EdgeId> + '_> {
        Box::new(
            self.inner
                .in_edge_ids(node_id)
                .filter(|&eid| self.view_edge(eid).is_some()),
        )
    }

    fn view_node_count(&self) -> usize {
        self.inner.view_node_count()
    }

    fn view_edge_count(&self) -> usize {
        // Must count only edges matching the filter
        self.inner
            .view_node_ids()
            .flat_map(|nid| self.out_edge_ids(nid))
            .count()
    }

    fn view_node_ids(&self) -> Box<dyn Iterator<Item = NodeId> + '_> {
        self.inner.view_node_ids()
    }
}

/// Node-subgraph view: only shows a specific set of nodes and their edges.
pub struct NodeSubgraph<'a, G: GraphView> {
    inner: &'a G,
    visible_nodes: std::collections::HashSet<NodeId>,
}

impl<'a, G: GraphView> NodeSubgraph<'a, G> {
    pub fn new(inner: &'a G, visible_nodes: std::collections::HashSet<NodeId>) -> Self {
        Self {
            inner,
            visible_nodes,
        }
    }
}

impl<'a, G: GraphView> GraphView for NodeSubgraph<'a, G> {
    fn view_node(&self, node_id: NodeId) -> Option<&GraphNode> {
        if self.visible_nodes.contains(&node_id) {
            self.inner.view_node(node_id)
        } else {
            None
        }
    }

    fn view_edge(&self, edge_id: EdgeId) -> Option<&GraphEdge> {
        self.inner.view_edge(edge_id).filter(|e| {
            self.visible_nodes.contains(&e.source) && self.visible_nodes.contains(&e.target)
        })
    }

    fn out_edge_ids(&self, node_id: NodeId) -> Box<dyn Iterator<Item = EdgeId> + '_> {
        if !self.visible_nodes.contains(&node_id) {
            return Box::new(std::iter::empty());
        }
        Box::new(
            self.inner
                .out_edge_ids(node_id)
                .filter(|&eid| self.view_edge(eid).is_some()),
        )
    }

    fn in_edge_ids(&self, node_id: NodeId) -> Box<dyn Iterator<Item = EdgeId> + '_> {
        if !self.visible_nodes.contains(&node_id) {
            return Box::new(std::iter::empty());
        }
        Box::new(
            self.inner
                .in_edge_ids(node_id)
                .filter(|&eid| self.view_edge(eid).is_some()),
        )
    }

    fn view_node_count(&self) -> usize {
        self.visible_nodes
            .iter()
            .filter(|&&nid| self.inner.view_node(nid).is_some())
            .count()
    }

    fn view_edge_count(&self) -> usize {
        self.visible_nodes
            .iter()
            .flat_map(|&nid| self.out_edge_ids(nid))
            .count()
    }

    fn view_node_ids(&self) -> Box<dyn Iterator<Item = NodeId> + '_> {
        Box::new(
            self.visible_nodes
                .iter()
                .copied()
                .filter(|&nid| self.inner.view_node(nid).is_some()),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::{EdgeType, GraphEdge, GraphNode, NodeType};

    fn make_test_graph() -> Graph {
        let mut g = Graph::new();
        let n1 = GraphNode::new(NodeType::Agent {
            agent_id: 1,
            agent_type: "test".to_string(),
            capabilities: vec![],
        });
        let n2 = GraphNode::new(NodeType::Event {
            event_id: 100,
            event_type: "action".to_string(),
            significance: 0.5,
        });
        let n3 = GraphNode::new(NodeType::Event {
            event_id: 200,
            event_type: "observation".to_string(),
            significance: 0.3,
        });
        let id1 = g.add_node(n1).unwrap();
        let id2 = g.add_node(n2).unwrap();
        let id3 = g.add_node(n3).unwrap();

        g.add_edge(GraphEdge::new(
            id1,
            id2,
            EdgeType::Causality {
                strength: 0.8,
                lag_ms: 10,
            },
            1.0,
        ));
        g.add_edge(GraphEdge::new(
            id2,
            id3,
            EdgeType::Temporal {
                average_interval_ms: 50,
                sequence_confidence: 0.9,
            },
            0.5,
        ));
        g.add_edge(GraphEdge::new(
            id1,
            id3,
            EdgeType::Contextual {
                similarity: 0.7,
                co_occurrence_rate: 0.3,
            },
            0.6,
        ));
        g
    }

    #[test]
    fn test_graph_view_for_graph() {
        let g = make_test_graph();
        assert_eq!(g.view_node_count(), 3);
        assert_eq!(g.view_edge_count(), 3);
        assert!(g.view_node(1).is_some());
        assert!(g.view_node(99).is_none());
        assert!(g.view_edge(1).is_some());

        let neighbors = g.view_neighbors(1);
        assert_eq!(neighbors.len(), 2); // edges to 2 and 3
    }

    #[test]
    fn test_graph_view_neighbors_directed() {
        let g = make_test_graph();
        let out = g.view_neighbors_directed(1, Direction::Out);
        assert_eq!(out.len(), 2);
        let inc = g.view_neighbors_directed(2, Direction::In);
        assert_eq!(inc.len(), 1);
        assert_eq!(inc[0], 1);
    }

    #[test]
    fn test_edge_type_filter() {
        let g = make_test_graph();
        let mut allowed = std::collections::HashSet::new();
        allowed.insert(std::mem::discriminant(&EdgeType::Causality {
            strength: 0.0,
            lag_ms: 0,
        }));
        let filtered = EdgeTypeFilter::new(&g, allowed);

        // Only 1 causality edge should be visible
        let neighbors = filtered.view_neighbors(1);
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0], 2);
    }

    #[test]
    fn test_node_subgraph() {
        let g = make_test_graph();
        let visible: std::collections::HashSet<NodeId> = [1, 2].into_iter().collect();
        let subgraph = NodeSubgraph::new(&g, visible);

        assert_eq!(subgraph.view_node_count(), 2);
        assert!(subgraph.view_node(1).is_some());
        assert!(subgraph.view_node(3).is_none());

        // Only edges between visible nodes
        let neighbors = subgraph.view_neighbors(1);
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0], 2);
    }

    #[test]
    fn test_composable_filters() {
        let g = make_test_graph();
        // First filter to subgraph {1, 2, 3}, then filter edges to causality only
        let visible: std::collections::HashSet<NodeId> = [1, 2, 3].into_iter().collect();
        let subgraph = NodeSubgraph::new(&g, visible);

        let mut allowed = std::collections::HashSet::new();
        allowed.insert(std::mem::discriminant(&EdgeType::Causality {
            strength: 0.0,
            lag_ms: 0,
        }));
        let filtered = EdgeTypeFilter::new(&subgraph, allowed);

        // Only the causality edge from 1 -> 2
        let neighbors = filtered.view_neighbors(1);
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0], 2);
    }

    #[test]
    fn test_view_edge_between() {
        let g = make_test_graph();
        assert!(g.view_edge_between(1, 2).is_some());
        assert!(g.view_edge_between(2, 1).is_none()); // directed
    }

    #[test]
    fn test_view_has_node() {
        let g = make_test_graph();
        assert!(g.has_node(1));
        assert!(!g.has_node(99));
    }
}
