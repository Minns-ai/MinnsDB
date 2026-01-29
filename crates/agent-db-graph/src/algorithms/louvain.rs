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

    /// Detect communities in the graph
    pub fn detect_communities(&self, graph: &Graph) -> GraphResult<CommunityDetectionResult> {
        // Phase 1: Initialize - each node in its own community
        let mut node_communities = self.initialize_communities(graph);
        let mut best_modularity = self.calculate_modularity(graph, &node_communities)?;
        let mut iteration = 0;

        loop {
            iteration += 1;

            if iteration > self.config.max_iterations {
                break;
            }

            // Phase 1: Modularity optimization
            let improved = self.optimize_modularity(graph, &mut node_communities)?;

            if !improved {
                break;
            }

            // Calculate new modularity
            let new_modularity = self.calculate_modularity(graph, &node_communities)?;
            let improvement = new_modularity - best_modularity;

            if improvement < self.config.min_improvement {
                break;
            }

            best_modularity = new_modularity;

            // Phase 2: Could aggregate communities and repeat, but for now keep it simple
            // Full implementation would create super-graph and recurse
        }

        // Build communities map
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
            let current_community = *node_communities
                .get(&node_id)
                .ok_or_else(|| {
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

        // Add neighbors' communities
        for neighbor_id in graph.get_neighbors(node_id) {
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

    /// Calculate modularity of current partition
    fn calculate_modularity(
        &self,
        graph: &Graph,
        node_communities: &HashMap<NodeId, CommunityId>,
    ) -> GraphResult<f32> {
        let m = graph.total_edge_weight();
        if m == 0.0 {
            return Ok(0.0);
        }

        let two_m = 2.0 * m;
        let mut modularity = 0.0;

        // For each pair of nodes in the same community
        for (&node_i, &community_i) in node_communities {
            for (&node_j, &community_j) in node_communities {
                if community_i != community_j {
                    continue;
                }

                // A_ij - (k_i * k_j) / (2m)
                let a_ij = if let Some(weight) = graph.get_edge_weight(node_i, node_j) {
                    weight
                } else {
                    0.0
                };

                let k_i = graph.get_node_degree(node_i);
                let k_j = graph.get_node_degree(node_j);

                modularity += a_ij - (k_i * k_j) / two_m;
            }
        }

        Ok(modularity / two_m)
    }

    /// Build communities map from node assignments
    fn build_communities_map(
        &self,
        node_communities: &HashMap<NodeId, CommunityId>,
    ) -> HashMap<CommunityId, Vec<NodeId>> {
        let mut communities: HashMap<CommunityId, Vec<NodeId>> = HashMap::new();

        for (&node_id, &community_id) in node_communities {
            communities
                .entry(community_id)
                .or_insert_with(Vec::new)
                .push(node_id);
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
        let n1 = graph.add_node(GraphNode::new(NodeType::Event {
            event_id: 1,
            event_type: "test".to_string(),
            significance: 0.5,
        }));
        let n2 = graph.add_node(GraphNode::new(NodeType::Event {
            event_id: 2,
            event_type: "test".to_string(),
            significance: 0.5,
        }));
        let n3 = graph.add_node(GraphNode::new(NodeType::Event {
            event_id: 3,
            event_type: "test".to_string(),
            significance: 0.5,
        }));

        // Community 2: nodes 4, 5, 6 (highly connected)
        let n4 = graph.add_node(GraphNode::new(NodeType::Event {
            event_id: 4,
            event_type: "test".to_string(),
            significance: 0.5,
        }));
        let n5 = graph.add_node(GraphNode::new(NodeType::Event {
            event_id: 5,
            event_type: "test".to_string(),
            significance: 0.5,
        }));
        let n6 = graph.add_node(GraphNode::new(NodeType::Event {
            event_id: 6,
            event_type: "test".to_string(),
            significance: 0.5,
        }));

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
