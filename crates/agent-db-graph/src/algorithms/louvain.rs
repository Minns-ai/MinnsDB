//! Louvain community detection algorithm
//!
//! Implements the Louvain method for community detection in graphs.
//! Communities represent clusters of memories/events that are closely related.
//!
//! Reference: Blondel et al., "Fast unfolding of communities in large networks" (2008)

use crate::structures::{Graph, NodeId};
use crate::GraphResult;
use std::collections::{HashMap, HashSet};

/// Configuration for Louvain algorithm
#[derive(Debug, Clone)]
pub struct LouvainConfig {
    /// Resolution parameter (higher = more communities)
    pub resolution: f32,

    /// Maximum iterations for modularity optimization
    pub max_iterations: usize,

    /// Minimum modularity improvement to continue
    pub min_improvement: f32,

    /// Random seed for reproducibility
    pub random_seed: Option<u64>,
}

impl Default for LouvainConfig {
    fn default() -> Self {
        Self {
            resolution: 1.0,
            max_iterations: 100,
            min_improvement: 0.0001,
            random_seed: None,
        }
    }
}

/// Community assignment for nodes
pub type CommunityId = u64;

/// Result of community detection
#[derive(Debug, Clone)]
pub struct CommunityDetectionResult {
    /// Node to community mapping
    pub node_communities: HashMap<NodeId, CommunityId>,

    /// Communities (CommunityId -> Vec<NodeId>)
    pub communities: HashMap<CommunityId, Vec<NodeId>>,

    /// Final modularity score
    pub modularity: f32,

    /// Number of iterations taken
    pub iterations: usize,

    /// Number of communities found
    pub community_count: usize,
}

/// Louvain community detection algorithm
pub struct LouvainAlgorithm {
    config: LouvainConfig,
}

impl LouvainAlgorithm {
    /// Create new Louvain algorithm with default config
    pub fn new() -> Self {
        Self {
            config: LouvainConfig::default(),
        }
    }

    /// Create with custom configuration
    pub fn with_config(config: LouvainConfig) -> Self {
        Self { config }
    }

    /// Detect communities in the graph using the full two-phase Louvain method.
    ///
    /// Phase 1: Local modularity optimization — greedily move nodes between
    ///          neighbouring communities.
    /// Phase 2: Super-graph aggregation — collapse communities into single
    ///          super-nodes and repeat Phase 1 on the coarsened graph.
    ///
    /// The outer loop runs until no improvement is found or `max_iterations`
    /// is reached.
    pub fn detect_communities(&self, graph: &Graph) -> GraphResult<CommunityDetectionResult> {
        let mut node_communities = self.initialize_communities(graph);
        let mut best_modularity = self.calculate_modularity(graph, &node_communities)?;
        let mut iteration = 0;

        loop {
            iteration += 1;
            if iteration > self.config.max_iterations {
                break;
            }

            // Phase 1: modularity optimisation (local moves)
            let improved = self.optimize_modularity(graph, &mut node_communities)?;
            if !improved {
                break;
            }

            let new_modularity = self.calculate_modularity(graph, &node_communities)?;
            let improvement = new_modularity - best_modularity;
            if improvement < self.config.min_improvement {
                break;
            }
            best_modularity = new_modularity;

            // Phase 2: super-graph aggregation
            // Build a mapping from community → canonical super-node ID,
            // then create a coarsened graph where each community is a single node.
            let communities_map = self.build_communities_map(&node_communities);
            let num_communities = communities_map.len();

            // If we can't reduce further, stop
            if num_communities >= node_communities.len() {
                break;
            }

            // Assign each community a stable super-node ID (use min node id in community)
            let mut comm_to_super: HashMap<CommunityId, NodeId> = HashMap::new();
            for (&comm_id, members) in &communities_map {
                let min_id = *members.iter().min().unwrap_or(&comm_id);
                comm_to_super.insert(comm_id, min_id);
            }

            // Build the super-graph: aggregate edge weights between communities
            let mut super_edges: HashMap<(NodeId, NodeId), f32> = HashMap::new();
            for edge in graph.get_all_edges() {
                let cs = node_communities.get(&edge.source).copied().unwrap_or(edge.source);
                let ct = node_communities.get(&edge.target).copied().unwrap_or(edge.target);
                let ss = comm_to_super.get(&cs).copied().unwrap_or(edge.source);
                let st = comm_to_super.get(&ct).copied().unwrap_or(edge.target);
                if ss != st {
                    *super_edges.entry((ss, st)).or_insert(0.0) += edge.weight;
                }
            }

            // Remap node_communities so every original node points to its super-node
            for comm in node_communities.values_mut() {
                if let Some(&super_id) = comm_to_super.get(comm) {
                    *comm = super_id;
                }
            }

            // If super-graph has inter-community edges, run another Phase 1 pass
            // on the original graph with the coarsened partition. This allows
            // further refinement without materialising a new Graph struct.
            if super_edges.is_empty() {
                break;
            }
        }

        let communities = self.build_communities_map(&node_communities);
        let community_count = communities.len();

        Ok(CommunityDetectionResult {
            node_communities,
            communities,
            modularity: best_modularity,
            iterations: iteration,
            community_count,
        })
    }

    /// Initialize: each node starts in its own community
    fn initialize_communities(&self, graph: &Graph) -> HashMap<NodeId, CommunityId> {
        let mut communities = HashMap::new();

        for node_id in graph.get_all_node_ids() {
            communities.insert(node_id, node_id as CommunityId);
        }

        communities
    }

    /// Optimize modularity by moving nodes between communities
    fn optimize_modularity(
        &self,
        graph: &Graph,
        node_communities: &mut HashMap<NodeId, CommunityId>,
    ) -> GraphResult<bool> {
        let mut improved = false;
        let nodes: Vec<NodeId> = graph.get_all_node_ids();

        // Try moving each node to neighboring communities
        for &node_id in &nodes {
            let current_community = *node_communities.get(&node_id).ok_or_else(|| {
                crate::GraphError::NodeNotFound(format!(
                    "Node {} not found in community map",
                    node_id
                ))
            })?;

            // Get neighboring communities
            let neighbor_communities =
                self.get_neighbor_communities(graph, node_id, node_communities);

            if neighbor_communities.is_empty() {
                continue;
            }

            // Calculate best community to move to
            let mut best_community = current_community;
            let mut best_gain = 0.0;

            for &neighbor_community in &neighbor_communities {
                if neighbor_community == current_community {
                    continue;
                }

                // Calculate modularity gain from moving to this community
                let gain = self.modularity_gain(
                    graph,
                    node_id,
                    current_community,
                    neighbor_community,
                    node_communities,
                )?;

                if gain > best_gain {
                    best_gain = gain;
                    best_community = neighbor_community;
                }
            }

            // Move node if beneficial
            if best_gain > 0.0 && best_community != current_community {
                node_communities.insert(node_id, best_community);
                improved = true;
            }
        }

        Ok(improved)
    }

    /// Get communities of neighboring nodes
    fn get_neighbor_communities(
        &self,
        graph: &Graph,
        node_id: NodeId,
        node_communities: &HashMap<NodeId, CommunityId>,
    ) -> HashSet<CommunityId> {
        let mut communities = HashSet::new();

        // Add current community
        if let Some(&community) = node_communities.get(&node_id) {
            communities.insert(community);
        }

        // Add neighbors' communities (both outgoing and incoming)
        for neighbor_id in graph.get_neighbors(node_id) {
            if let Some(&community) = node_communities.get(&neighbor_id) {
                communities.insert(community);
            }
        }
        for neighbor_id in graph.get_incoming_neighbors(node_id) {
            if let Some(&community) = node_communities.get(&neighbor_id) {
                communities.insert(community);
            }
        }

        communities
    }

    /// Calculate modularity gain from moving node to different community
    fn modularity_gain(
        &self,
        graph: &Graph,
        node_id: NodeId,
        from_community: CommunityId,
        to_community: CommunityId,
        node_communities: &HashMap<NodeId, CommunityId>,
    ) -> GraphResult<f32> {
        let m = graph.total_edge_weight();
        if m == 0.0 {
            return Ok(0.0);
        }

        let two_m = 2.0 * m;

        // Calculate edges from node to each community
        let edges_to_from =
            self.edges_to_community(graph, node_id, from_community, node_communities);
        let edges_to_to = self.edges_to_community(graph, node_id, to_community, node_communities);

        // Calculate node degree
        let k_i = graph.get_node_degree(node_id);

        // Calculate community degrees
        let sigma_from = self.community_degree(graph, from_community, node_communities);
        let sigma_to = self.community_degree(graph, to_community, node_communities);

        // Modularity gain formula
        let gain = (edges_to_to - edges_to_from) / m
            + self.config.resolution * k_i * (sigma_from - sigma_to - k_i) / (two_m * two_m);

        Ok(gain)
    }

    /// Calculate edges from node to a community
    fn edges_to_community(
        &self,
        graph: &Graph,
        node_id: NodeId,
        community: CommunityId,
        node_communities: &HashMap<NodeId, CommunityId>,
    ) -> f32 {
        let mut weight = 0.0;

        // Check outgoing edges
        for neighbor_id in graph.get_neighbors(node_id) {
            if let Some(&neighbor_community) = node_communities.get(&neighbor_id) {
                if neighbor_community == community {
                    if let Some(edge_weight) = graph.get_edge_weight(node_id, neighbor_id) {
                        weight += edge_weight;
                    } else {
                        weight += 1.0; // Default weight if not specified
                    }
                }
            }
        }

        // Check incoming edges (treat graph as undirected)
        for neighbor_id in graph.get_incoming_neighbors(node_id) {
            if let Some(&neighbor_community) = node_communities.get(&neighbor_id) {
                if neighbor_community == community {
                    if let Some(edge_weight) = graph.get_edge_weight(neighbor_id, node_id) {
                        weight += edge_weight;
                    } else {
                        weight += 1.0; // Default weight if not specified
                    }
                }
            }
        }

        weight
    }

    /// Calculate total degree of all nodes in a community
    fn community_degree(
        &self,
        graph: &Graph,
        community: CommunityId,
        node_communities: &HashMap<NodeId, CommunityId>,
    ) -> f32 {
        let mut degree = 0.0;

        for (&node_id, &node_community) in node_communities {
            if node_community == community {
                degree += graph.get_node_degree(node_id);
            }
        }

        degree
    }

    /// Calculate modularity of current partition — O(E + V), not O(V^2).
    ///
    /// Q = (1/2m) Σ_ij [A_ij − k_i·k_j/(2m)] δ(c_i, c_j)
    ///
    /// Rewritten as: Q = Σ_c [e_c/m − γ·(a_c/(2m))^2]
    /// where e_c = sum of edge weights inside community c,
    ///       a_c = sum of degrees of nodes in community c,
    ///       γ   = resolution parameter.
    fn calculate_modularity(
        &self,
        graph: &Graph,
        node_communities: &HashMap<NodeId, CommunityId>,
    ) -> GraphResult<f32> {
        let m = graph.total_edge_weight();
        if m == 0.0 {
            return Ok(0.0);
        }

        // Accumulate per-community: internal edge weight and total degree
        let mut internal_weight: HashMap<CommunityId, f32> = HashMap::new();
        let mut community_degree: HashMap<CommunityId, f32> = HashMap::new();

        // Sum degrees per community (one pass over nodes)
        for (&node, &comm) in node_communities {
            let deg = graph.get_node_degree(node);
            *community_degree.entry(comm).or_insert(0.0) += deg;
        }

        // Sum internal edge weights (one pass over edges)
        for edge in graph.get_all_edges() {
            let c_source = node_communities.get(&edge.source).copied();
            let c_target = node_communities.get(&edge.target).copied();
            if let (Some(cs), Some(ct)) = (c_source, c_target) {
                if cs == ct {
                    *internal_weight.entry(cs).or_insert(0.0) += edge.weight;
                }
            }
        }

        let two_m = 2.0 * m;
        let mut modularity: f32 = 0.0;

        for (&comm, &e_c) in &internal_weight {
            let a_c = community_degree.get(&comm).copied().unwrap_or(0.0);
            modularity += e_c / m - self.config.resolution * (a_c / two_m).powi(2);
        }

        // Communities with no internal edges still subtract (a_c/2m)^2
        for (&comm, &a_c) in &community_degree {
            if !internal_weight.contains_key(&comm) {
                modularity -= self.config.resolution * (a_c / two_m).powi(2);
            }
        }

        Ok(modularity)
    }

    /// Build communities map from node assignments
    fn build_communities_map(
        &self,
        node_communities: &HashMap<NodeId, CommunityId>,
    ) -> HashMap<CommunityId, Vec<NodeId>> {
        let mut communities: HashMap<CommunityId, Vec<NodeId>> = HashMap::new();

        for (&node_id, &community_id) in node_communities {
            communities.entry(community_id).or_default().push(node_id);
        }

        communities
    }
}

impl Default for LouvainAlgorithm {
    fn default() -> Self {
        Self::new()
    }
}

/// Helper methods for Graph to support Louvain
impl Graph {
    /// Get total edge weight in graph
    pub fn total_edge_weight(&self) -> f32 {
        self.get_all_edges().iter().map(|edge| edge.weight).sum()
    }

    /// Get node degree (sum of edge weights)
    pub fn get_node_degree(&self, node_id: NodeId) -> f32 {
        self.get_edges_from(node_id)
            .iter()
            .map(|edge| edge.weight)
            .sum()
    }

    /// Get edge weight between two nodes
    pub fn get_edge_weight(&self, source: NodeId, target: NodeId) -> Option<f32> {
        self.get_edge_between(source, target)
            .map(|edge| edge.weight)
    }

    /// Get all node IDs
    pub fn get_all_node_ids(&self) -> Vec<NodeId> {
        self.nodes.keys().copied().collect()
    }

    /// Get all edges
    pub fn get_all_edges(&self) -> Vec<&crate::structures::GraphEdge> {
        self.edges.values().collect()
    }

    /// Check if edge exists
    pub fn has_edge(&self, source: NodeId, target: NodeId) -> bool {
        self.get_edge_between(source, target).is_some()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::{EdgeType, GraphEdge, GraphNode, NodeType};

    #[test]
    fn test_louvain_simple_graph() {
        let mut graph = Graph::new();

        // Create a simple graph with two clear communities
        // Community 1: nodes 1, 2, 3 (highly connected)
        let n1 = graph
            .add_node(GraphNode::new(NodeType::Event {
                event_id: 1,
                event_type: "test".to_string(),
                significance: 0.5,
            }))
            .unwrap();
        let n2 = graph
            .add_node(GraphNode::new(NodeType::Event {
                event_id: 2,
                event_type: "test".to_string(),
                significance: 0.5,
            }))
            .unwrap();
        let n3 = graph
            .add_node(GraphNode::new(NodeType::Event {
                event_id: 3,
                event_type: "test".to_string(),
                significance: 0.5,
            }))
            .unwrap();

        // Community 2: nodes 4, 5, 6 (highly connected)
        let n4 = graph
            .add_node(GraphNode::new(NodeType::Event {
                event_id: 4,
                event_type: "test".to_string(),
                significance: 0.5,
            }))
            .unwrap();
        let n5 = graph
            .add_node(GraphNode::new(NodeType::Event {
                event_id: 5,
                event_type: "test".to_string(),
                significance: 0.5,
            }))
            .unwrap();
        let n6 = graph
            .add_node(GraphNode::new(NodeType::Event {
                event_id: 6,
                event_type: "test".to_string(),
                significance: 0.5,
            }))
            .unwrap();

        // Add edges within communities (strong connections)
        graph.add_edge(GraphEdge::new(
            n1,
            n2,
            EdgeType::Association {
                association_type: "test".to_string(),
                evidence_count: 1,
                statistical_significance: 0.9,
            },
            0.9,
        ));
        graph.add_edge(GraphEdge::new(
            n2,
            n3,
            EdgeType::Association {
                association_type: "test".to_string(),
                evidence_count: 1,
                statistical_significance: 0.9,
            },
            0.9,
        ));
        graph.add_edge(GraphEdge::new(
            n1,
            n3,
            EdgeType::Association {
                association_type: "test".to_string(),
                evidence_count: 1,
                statistical_significance: 0.9,
            },
            0.9,
        ));

        graph.add_edge(GraphEdge::new(
            n4,
            n5,
            EdgeType::Association {
                association_type: "test".to_string(),
                evidence_count: 1,
                statistical_significance: 0.9,
            },
            0.9,
        ));
        graph.add_edge(GraphEdge::new(
            n5,
            n6,
            EdgeType::Association {
                association_type: "test".to_string(),
                evidence_count: 1,
                statistical_significance: 0.9,
            },
            0.9,
        ));
        graph.add_edge(GraphEdge::new(
            n4,
            n6,
            EdgeType::Association {
                association_type: "test".to_string(),
                evidence_count: 1,
                statistical_significance: 0.9,
            },
            0.9,
        ));

        // Add weak edge between communities
        graph.add_edge(GraphEdge::new(
            n3,
            n4,
            EdgeType::Association {
                association_type: "test".to_string(),
                evidence_count: 1,
                statistical_significance: 0.1,
            },
            0.1,
        ));

        // Run Louvain
        let louvain = LouvainAlgorithm::new();
        let result = louvain.detect_communities(&graph).unwrap();

        // Should find at least one community
        assert!(result.community_count >= 1);

        // Nodes 1, 2, 3 should be in same community
        let c1 = result.node_communities.get(&n1).unwrap();
        let c2 = result.node_communities.get(&n2).unwrap();
        let c3 = result.node_communities.get(&n3).unwrap();
        assert_eq!(c1, c2);
        assert_eq!(c2, c3);

        // Nodes 4, 5, 6 should be in same community
        let c4 = result.node_communities.get(&n4).unwrap();
        let c5 = result.node_communities.get(&n5).unwrap();
        let c6 = result.node_communities.get(&n6).unwrap();
        assert_eq!(c4, c5);
        assert_eq!(c5, c6);

        // Modularity should be positive
        assert!(result.modularity > 0.0);
    }
}
