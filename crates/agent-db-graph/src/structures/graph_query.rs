// crates/agent-db-graph/src/structures/graph_query.rs
//
// Graph query methods: getters, directed queries, filtered queries,
// bi-temporal queries, shortest_path, nodes_in_time_range.

use agent_db_core::types::{AgentId, ContextHash, EventId, Timestamp};
use std::collections::{HashMap, HashSet, VecDeque};

use super::edge::GraphEdge;
use super::graph::Graph;
use super::node::{node_type_discriminant_from_name, GraphNode};
use super::types::{Direction, EdgeId, NodeId};

impl Graph {
    /// Get node by ID
    pub fn get_node(&self, node_id: NodeId) -> Option<&GraphNode> {
        self.nodes.get(node_id)
    }

    /// Get edge by ID
    pub fn get_edge(&self, edge_id: EdgeId) -> Option<&GraphEdge> {
        self.edges.get(edge_id)
    }

    /// Get node by agent ID
    pub fn get_agent_node(&self, agent_id: AgentId) -> Option<&GraphNode> {
        self.agent_index
            .get(&agent_id)
            .and_then(|&node_id| self.nodes.get(node_id))
    }

    /// Get node by event ID
    pub fn get_event_node(&self, event_id: EventId) -> Option<&GraphNode> {
        self.event_index
            .get(&event_id)
            .and_then(|&node_id| self.nodes.get(node_id))
    }

    /// Get node by context hash
    pub fn get_context_node(&self, context_hash: ContextHash) -> Option<&GraphNode> {
        self.context_index
            .get(&context_hash)
            .and_then(|&node_id| self.nodes.get(node_id))
    }

    /// Get node by goal ID
    pub fn get_goal_node(&self, goal_id: u64) -> Option<&GraphNode> {
        self.goal_index
            .get(&goal_id)
            .and_then(|&node_id| self.nodes.get(node_id))
    }

    /// Get node by episode ID
    pub fn get_episode_node(&self, episode_id: u64) -> Option<&GraphNode> {
        self.episode_index
            .get(&episode_id)
            .and_then(|&node_id| self.nodes.get(node_id))
    }

    /// Get node by memory ID
    pub fn get_memory_node(&self, memory_id: u64) -> Option<&GraphNode> {
        self.memory_index
            .get(&memory_id)
            .and_then(|&node_id| self.nodes.get(node_id))
    }

    /// Get node by strategy ID
    pub fn get_strategy_node(&self, strategy_id: u64) -> Option<&GraphNode> {
        self.strategy_index
            .get(&strategy_id)
            .and_then(|&node_id| self.nodes.get(node_id))
    }

    /// Get node by tool name
    pub fn get_tool_node(&self, tool_name: &str) -> Option<&GraphNode> {
        self.tool_index
            .get(tool_name)
            .and_then(|&node_id| self.nodes.get(node_id))
    }

    /// Get node by result key
    pub fn get_result_node(&self, result_key: &str) -> Option<&GraphNode> {
        self.result_index
            .get(result_key)
            .and_then(|&node_id| self.nodes.get(node_id))
    }

    /// Get mutable reference to node by ID
    pub fn get_node_mut(&mut self, node_id: NodeId) -> Option<&mut GraphNode> {
        self.nodes.get_mut(node_id)
    }

    /// Get edge between two nodes (source -> target)
    pub fn get_edge_between(&self, source: NodeId, target: NodeId) -> Option<&GraphEdge> {
        // Use adjacency list to find the edge efficiently
        if let Some(edge_ids) = self.adjacency_out.get(source) {
            for &edge_id in edge_ids {
                if let Some(edge) = self.edges.get(edge_id) {
                    if edge.target == target {
                        return Some(edge);
                    }
                }
            }
        }
        None
    }

    /// Get mutable reference to edge between two nodes
    pub fn get_edge_between_mut(
        &mut self,
        source: NodeId,
        target: NodeId,
    ) -> Option<&mut GraphEdge> {
        // Find the edge ID first
        let edge_id_opt = self.adjacency_out.get(source).and_then(|edge_ids| {
            edge_ids.iter().find_map(|&edge_id| {
                self.edges
                    .get(edge_id)
                    .filter(|edge| edge.target == target)
                    .map(|_| edge_id)
            })
        });

        if let Some(edge_id) = edge_id_opt {
            self.edges.get_mut(edge_id)
        } else {
            None
        }
    }

    /// Get neighbors of a node (outgoing edges)
    pub fn get_neighbors(&self, node_id: NodeId) -> Vec<NodeId> {
        match self.adjacency_out.get(node_id) {
            Some(edges) => edges
                .iter()
                .filter_map(|&edge_id| self.edges.get(edge_id).map(|edge| edge.target))
                .collect(),
            None => Vec::new(),
        }
    }

    /// Get incoming neighbors of a node
    pub fn get_incoming_neighbors(&self, node_id: NodeId) -> Vec<NodeId> {
        match self.adjacency_in.get(node_id) {
            Some(edges) => edges
                .iter()
                .filter_map(|&edge_id| self.edges.get(edge_id).map(|edge| edge.source))
                .collect(),
            None => Vec::new(),
        }
    }

    /// Get all edges from a source node
    pub fn get_edges_from(&self, source: NodeId) -> Vec<&GraphEdge> {
        self.adjacency_out
            .get(source)
            .map(|edge_ids| {
                edge_ids
                    .iter()
                    .filter_map(|&edge_id| self.edges.get(edge_id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get all edges to a target node
    pub fn get_edges_to(&self, target: NodeId) -> Vec<&GraphEdge> {
        self.adjacency_in
            .get(target)
            .map(|edge_ids| {
                edge_ids
                    .iter()
                    .filter_map(|&edge_id| self.edges.get(edge_id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get the outgoing adjacency list for a node without collecting into Vec.
    /// Returns None if the node has no outgoing edges.
    #[inline]
    pub fn adjacency_out_ref(
        &self,
        node_id: NodeId,
    ) -> Option<&crate::structures::adj_list::AdjList> {
        self.adjacency_out.get(node_id)
    }

    /// Get the incoming adjacency list for a node without collecting into Vec.
    /// Returns None if the node has no incoming edges.
    #[inline]
    pub fn adjacency_in_ref(
        &self,
        node_id: NodeId,
    ) -> Option<&crate::structures::adj_list::AdjList> {
        self.adjacency_in.get(node_id)
    }

    /// Get node by claim ID
    pub fn get_claim_node(&self, claim_id: u64) -> Option<&GraphNode> {
        self.claim_index
            .get(&claim_id)
            .and_then(|&node_id| self.nodes.get(node_id))
    }

    /// Get node by concept name
    pub fn get_concept_node(&self, concept_name: &str) -> Option<&GraphNode> {
        self.concept_index
            .get(concept_name)
            .and_then(|&node_id| self.nodes.get(node_id))
    }

    /// Iterate over all nodes in the graph.
    pub fn nodes(&self) -> impl Iterator<Item = &GraphNode> {
        self.nodes.values()
    }

    /// Find a concept node by its qualified_name (used for code dedup).
    pub fn find_concept_node(&self, qualified_name: &str) -> Option<NodeId> {
        self.concept_index.get(qualified_name).copied()
    }

    /// Get node count
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get edge count
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Get all node IDs
    pub fn node_ids(&self) -> Vec<NodeId> {
        self.nodes.keys().collect()
    }

    /// Current generation counter (monotonically increasing on every mutation).
    /// Used by cache layers to detect stale entries.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Get all nodes of a specific type.
    /// Accepts human-readable names ("Agent", "Event", etc.) and resolves them
    /// to u8 discriminant keys internally.
    pub fn get_nodes_by_type(&self, type_name: &str) -> Vec<&GraphNode> {
        let disc = match node_type_discriminant_from_name(type_name) {
            Some(d) => d,
            None => return Vec::new(),
        };
        self.type_index
            .get(&disc)
            .map(|set| {
                set.iter()
                    .filter_map(|&node_id| self.nodes.get(node_id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Query nodes within a timestamp range `[start, end]` (inclusive).
    /// Uses the BTree temporal index for O(log N + K) lookups.
    pub fn nodes_in_time_range(&self, start: Timestamp, end: Timestamp) -> Vec<&GraphNode> {
        self.temporal_index
            .range(start..=end)
            .flat_map(|(_, ids)| ids.iter())
            .filter_map(|&nid| self.nodes.get(nid))
            .collect()
    }

    /// Find shortest path between two nodes
    pub fn shortest_path(&self, start: NodeId, end: NodeId) -> Option<Vec<NodeId>> {
        if start == end {
            return Some(vec![start]);
        }

        let mut queue = VecDeque::new();
        let mut visited = HashSet::new();
        let mut parent = HashMap::new();

        queue.push_back(start);
        visited.insert(start);

        while let Some(current) = queue.pop_front() {
            for neighbor in self.get_neighbors(current) {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    parent.insert(neighbor, current);
                    queue.push_back(neighbor);

                    if neighbor == end {
                        // Reconstruct path
                        let mut path = vec![end];
                        let mut current = end;

                        while let Some(&prev) = parent.get(&current) {
                            path.push(prev);
                            current = prev;
                        }

                        path.reverse();
                        return Some(path);
                    }
                }
            }
        }

        None // No path found
    }

    // ========================================================================
    // Direction-aware queries
    // ========================================================================

    /// Get neighbors in the specified direction, deduped for `Both`.
    pub fn neighbors_directed(&self, node_id: NodeId, direction: Direction) -> Vec<NodeId> {
        match direction {
            Direction::Out => self.get_neighbors(node_id),
            Direction::In => self.get_incoming_neighbors(node_id),
            Direction::Both => {
                let mut seen = HashSet::new();
                let mut result = Vec::new();
                for n in self.get_neighbors(node_id) {
                    if seen.insert(n) {
                        result.push(n);
                    }
                }
                for n in self.get_incoming_neighbors(node_id) {
                    if seen.insert(n) {
                        result.push(n);
                    }
                }
                result
            },
        }
    }

    /// Get incident edges in the specified direction, deduped by EdgeId for `Both`.
    pub fn edges_directed(&self, node_id: NodeId, direction: Direction) -> Vec<&GraphEdge> {
        match direction {
            Direction::Out => self.get_edges_from(node_id),
            Direction::In => self.get_edges_to(node_id),
            Direction::Both => {
                let mut seen = HashSet::new();
                let mut result = Vec::new();
                for edge in self.get_edges_from(node_id) {
                    if seen.insert(edge.id) {
                        result.push(edge);
                    }
                }
                for edge in self.get_edges_to(node_id) {
                    if seen.insert(edge.id) {
                        result.push(edge);
                    }
                }
                result
            },
        }
    }

    // ========================================================================
    // Soft-delete filtered queries
    // ========================================================================

    /// Get valid (non-soft-deleted) edges from a source node.
    pub fn get_valid_edges_from(&self, source: NodeId) -> Vec<&GraphEdge> {
        self.get_edges_from(source)
            .into_iter()
            .filter(|e| e.is_valid())
            .collect()
    }

    /// Get valid (non-soft-deleted) edges to a target node.
    pub fn get_valid_edges_to(&self, target: NodeId) -> Vec<&GraphEdge> {
        self.get_edges_to(target)
            .into_iter()
            .filter(|e| e.is_valid())
            .collect()
    }

    /// Get valid neighbors (outgoing, filtering soft-deleted edges).
    pub fn get_valid_neighbors(&self, node_id: NodeId) -> Vec<NodeId> {
        self.adjacency_out
            .get(node_id)
            .map(|edge_ids| {
                edge_ids
                    .iter()
                    .filter_map(|&edge_id| {
                        self.edges
                            .get(edge_id)
                            .filter(|e| e.is_valid())
                            .map(|e| e.target)
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get valid incoming neighbors (filtering soft-deleted edges).
    pub fn get_valid_incoming_neighbors(&self, node_id: NodeId) -> Vec<NodeId> {
        self.adjacency_in
            .get(node_id)
            .map(|edge_ids| {
                edge_ids
                    .iter()
                    .filter_map(|&edge_id| {
                        self.edges
                            .get(edge_id)
                            .filter(|e| e.is_valid())
                            .map(|e| e.source)
                    })
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get invalidated edges from a node (for temporal queries like "what changed?").
    pub fn get_invalidated_edges_from(&self, source: NodeId) -> Vec<&GraphEdge> {
        self.get_edges_from(source)
            .into_iter()
            .filter(|e| !e.is_valid())
            .collect()
    }

    // ── Bi-temporal queries ─────────────────────────────────────────────

    /// Get valid edges from a source that were true at a specific real-world time.
    ///
    /// Combines soft-delete check (`is_valid`) with valid-time check.
    pub fn get_edges_valid_at(&self, source: NodeId, point_in_time: Timestamp) -> Vec<&GraphEdge> {
        self.get_edges_from(source)
            .into_iter()
            .filter(|e| e.is_valid() && e.valid_at(point_in_time))
            .collect()
    }

    /// Get valid edges from a source whose fact was true during a time range.
    pub fn get_edges_valid_during(
        &self,
        source: NodeId,
        range_start: Timestamp,
        range_end: Timestamp,
    ) -> Vec<&GraphEdge> {
        self.get_edges_from(source)
            .into_iter()
            .filter(|e| e.is_valid() && e.valid_during(range_start, range_end))
            .collect()
    }

    /// Get all edges (any source) that represent currently-valid facts.
    pub fn get_all_currently_valid_edges(&self, now: Timestamp) -> Vec<&GraphEdge> {
        self.edges
            .values()
            .filter(|e| e.is_currently_valid_fact(now))
            .collect()
    }

    /// The latest timestamp in the temporal index (max created_at across all nodes).
    /// Returns `None` if the graph is empty.
    pub fn latest_timestamp(&self) -> Option<Timestamp> {
        self.temporal_index.keys().next_back().copied()
    }
}
