//! Graph analytics with focus on agent learning metrics
//!
//! Unlike generic graph analytics (baseline system-style), this focuses on:
//! - Learning performance metrics
//! - Memory formation rates
//! - Strategy success trends
//! - Agent improvement over time

use crate::algorithms::{CentralityMeasures, LouvainAlgorithm};
use crate::structures::{Graph, NodeId};
use crate::GraphResult;
use std::collections::HashSet;

/// Comprehensive graph analytics
pub struct GraphAnalytics<'a> {
    graph: &'a Graph,
}

impl<'a> GraphAnalytics<'a> {
    /// Create analytics from graph reference
    pub fn new(graph: &'a Graph) -> Self {
        Self { graph }
    }

    /// Create analytics from reference (alias for new)
    pub fn from_ref(graph: &'a Graph) -> Self {
        Self { graph }
    }

    /// Calculate all metrics at once
    pub fn calculate_all_metrics(&self) -> GraphResult<GraphMetrics> {
        Ok(GraphMetrics {
            // Basic stats
            node_count: self.graph.node_count(),
            edge_count: self.graph.edge_count(),

            // Connectivity
            connected_components: self.count_components()?,
            largest_component_size: self.largest_component_size()?,
            average_path_length: self.average_path_length()?,
            diameter: self.diameter()?,

            // Clustering
            clustering_coefficient: self.global_clustering_coefficient()?,
            average_clustering: self.average_clustering_coefficient()?,

            // Centrality (top nodes only for performance)
            most_central_nodes: self.top_centrality_nodes(10)?,

            // Community structure
            modularity: self.modularity()?,
            community_count: self.community_count()?,

            // EventGraphDB-specific learning metrics
            learning_metrics: self.calculate_learning_metrics()?,
        })
    }

    /// Count connected components
    fn count_components(&self) -> GraphResult<usize> {
        let nodes = self.graph.get_all_node_ids();
        let mut visited = HashSet::new();
        let mut components = 0;

        for &node_id in &nodes {
            if !visited.contains(&node_id) {
                // Start DFS from this node
                self.dfs_mark(&self.graph, node_id, &mut visited);
                components += 1;
            }
        }

        Ok(components)
    }

    /// DFS marking for component detection
    fn dfs_mark(&self, graph: &Graph, start: NodeId, visited: &mut HashSet<NodeId>) {
        let mut stack = vec![start];

        while let Some(node_id) = stack.pop() {
            if visited.contains(&node_id) {
                continue;
            }

            visited.insert(node_id);

            for neighbor in graph.get_neighbors(node_id) {
                if !visited.contains(&neighbor) {
                    stack.push(neighbor);
                }
            }
        }
    }

    /// Find size of largest connected component
    fn largest_component_size(&self) -> GraphResult<usize> {
        let nodes = self.graph.get_all_node_ids();
        let mut visited = HashSet::new();
        let mut max_size = 0;

        for &node_id in &nodes {
            if !visited.contains(&node_id) {
                let component_nodes = self.get_component_nodes(&self.graph, node_id);
                let size = component_nodes.len();
                max_size = max_size.max(size);

                visited.extend(component_nodes);
            }
        }

        Ok(max_size)
    }

    /// Get all nodes in a component
    fn get_component_nodes(&self, graph: &Graph, start: NodeId) -> HashSet<NodeId> {
        let mut visited = HashSet::new();
        let mut stack = vec![start];

        while let Some(node_id) = stack.pop() {
            if visited.contains(&node_id) {
                continue;
            }

            visited.insert(node_id);

            for neighbor in graph.get_neighbors(node_id) {
                if !visited.contains(&neighbor) {
                    stack.push(neighbor);
                }
            }
        }

        visited
    }

    /// Calculate average path length (sample-based for large graphs)
    fn average_path_length(&self) -> GraphResult<f32> {
        let nodes = self.graph.get_all_node_ids();
        let n = nodes.len();

        if n <= 1 {
            return Ok(0.0);
        }

        // For large graphs, sample pairs
        let sample_size = if n > 100 { 100 } else { n };
        let mut total_distance = 0.0;
        let mut path_count = 0;

        for i in 0..sample_size.min(nodes.len()) {
            for j in i + 1..sample_size.min(nodes.len()) {
                if let Some(path) = self.graph.shortest_path(nodes[i], nodes[j]) {
                    total_distance += (path.len() - 1) as f32;
                    path_count += 1;
                }
            }
        }

        if path_count == 0 {
            return Ok(0.0);
        }

        Ok(total_distance / path_count as f32)
    }

    /// Calculate graph diameter (maximum shortest path)
    fn diameter(&self) -> GraphResult<u32> {
        let nodes = self.graph.get_all_node_ids();
        let mut max_distance = 0;

        // Sample for large graphs
        let sample_size = if nodes.len() > 50 { 50 } else { nodes.len() };

        for i in 0..sample_size {
            for j in i + 1..sample_size {
                if let Some(path) = self.graph.shortest_path(nodes[i], nodes[j]) {
                    let distance = (path.len() - 1) as u32;
                    max_distance = max_distance.max(distance);
                }
            }
        }

        Ok(max_distance)
    }

    /// Global clustering coefficient
    fn global_clustering_coefficient(&self) -> GraphResult<f32> {
        let nodes = self.graph.get_all_node_ids();
        let mut triangles = 0;
        let mut triplets = 0;

        for &node_id in &nodes {
            let neighbors: Vec<NodeId> = self.graph.get_neighbors(node_id);

            // Count triangles (3-cliques)
            for i in 0..neighbors.len() {
                for j in i + 1..neighbors.len() {
                    triplets += 1;
                    if self.graph.has_edge(neighbors[i], neighbors[j]) {
                        triangles += 1;
                    }
                }
            }
        }

        if triplets == 0 {
            return Ok(0.0);
        }

        Ok((3 * triangles) as f32 / triplets as f32)
    }

    /// Average clustering coefficient
    fn average_clustering_coefficient(&self) -> GraphResult<f32> {
        let nodes = self.graph.get_all_node_ids();

        if nodes.is_empty() {
            return Ok(0.0);
        }

        let mut total_clustering = 0.0;

        for &node_id in &nodes {
            let clustering = self.local_clustering_coefficient(node_id)?;
            total_clustering += clustering;
        }

        Ok(total_clustering / nodes.len() as f32)
    }

    /// Local clustering coefficient for a node
    fn local_clustering_coefficient(&self, node_id: NodeId) -> GraphResult<f32> {
        let neighbors = self.graph.get_neighbors(node_id);
        let k = neighbors.len();

        if k < 2 {
            return Ok(0.0);
        }

        let mut edges_between_neighbors = 0;

        for i in 0..neighbors.len() {
            for j in i + 1..neighbors.len() {
                if self.graph.has_edge(neighbors[i], neighbors[j]) {
                    edges_between_neighbors += 1;
                }
            }
        }

        let max_possible = k * (k - 1) / 2;
        Ok(edges_between_neighbors as f32 / max_possible as f32)
    }

    /// Calculate modularity using Louvain
    fn modularity(&self) -> GraphResult<f32> {
        let louvain = LouvainAlgorithm::new();
        let result = louvain.detect_communities(&self.graph)?;
        Ok(result.modularity)
    }

    /// Get community count
    fn community_count(&self) -> GraphResult<usize> {
        let louvain = LouvainAlgorithm::new();
        let result = louvain.detect_communities(&self.graph)?;
        Ok(result.community_count)
    }

    /// Get top central nodes
    fn top_centrality_nodes(&self, n: usize) -> GraphResult<Vec<(NodeId, f32)>> {
        let centrality = CentralityMeasures::new();
        let all_centralities = centrality.all_centralities(&self.graph)?;
        Ok(all_centralities.top_combined(n))
    }

    /// Calculate learning-specific metrics for EventGraphDB
    fn calculate_learning_metrics(&self) -> GraphResult<LearningMetrics> {
        // Count different node types
        let event_nodes = self.graph.get_nodes_by_type("Event");
        let context_nodes = self.graph.get_nodes_by_type("Context");
        let concept_nodes = self.graph.get_nodes_by_type("Concept");

        // Calculate success rates from edge properties
        let edges = self.graph.get_all_edges();
        let mut total_success = 0;
        let mut total_failure = 0;

        for edge in &edges {
            total_success += edge.get_success_count();
            total_failure += edge.get_failure_count();
        }

        let total_outcomes = total_success + total_failure;
        let overall_success_rate = if total_outcomes > 0 {
            total_success as f32 / total_outcomes as f32
        } else {
            0.0
        };

        // Estimate memory formation rate (contexts with high connection count)
        let strong_memories = context_nodes
            .iter()
            .filter(|node| node.degree > 5) // Arbitrary threshold
            .count();

        // Pattern count (concept nodes)
        let pattern_count = concept_nodes.len();

        Ok(LearningMetrics {
            total_events: event_nodes.len(),
            unique_contexts: context_nodes.len(),
            learned_patterns: pattern_count,
            strong_memories: strong_memories,
            overall_success_rate,
            total_successful_actions: total_success,
            total_failed_actions: total_failure,
            average_edge_weight: if !edges.is_empty() {
                edges.iter().map(|e| e.weight).sum::<f32>() / edges.len() as f32
            } else {
                0.0
            },
        })
    }
}

/// Comprehensive graph metrics
#[derive(Debug, Clone)]
pub struct GraphMetrics {
    // Basic stats
    pub node_count: usize,
    pub edge_count: usize,

    // Connectivity
    pub connected_components: usize,
    pub largest_component_size: usize,
    pub average_path_length: f32,
    pub diameter: u32,

    // Clustering
    pub clustering_coefficient: f32,
    pub average_clustering: f32,

    // Centrality
    pub most_central_nodes: Vec<(NodeId, f32)>,

    // Community structure
    pub modularity: f32,
    pub community_count: usize,

    // Learning metrics (EventGraphDB-specific)
    pub learning_metrics: LearningMetrics,
}

/// Learning performance metrics
#[derive(Debug, Clone)]
pub struct LearningMetrics {
    /// Total events processed
    pub total_events: usize,

    /// Unique contexts encountered
    pub unique_contexts: usize,

    /// Number of learned patterns
    pub learned_patterns: usize,

    /// Number of strong memories (high degree contexts)
    pub strong_memories: usize,

    /// Overall action success rate
    pub overall_success_rate: f32,

    /// Total successful actions
    pub total_successful_actions: u32,

    /// Total failed actions
    pub total_failed_actions: u32,

    /// Average edge weight (pattern strength)
    pub average_edge_weight: f32,
}

/// Helper trait to add node_count and edge_count to Graph
impl Graph {
    /// Get node count
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Get edge count
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }
}

// Graph cloning removed to avoid expensive analytics copies.

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structures::{EdgeType, GraphEdge, GraphNode};
    use crate::NodeType;

    #[test]
    fn test_connected_components() {
        let mut graph = Graph::new();

        // Create two separate components
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
        let n4 = graph.add_node(GraphNode::new(NodeType::Event {
            event_id: 4,
            event_type: "test".to_string(),
            significance: 0.5,
        }));

        // Component 1: n1-n2
        graph.add_edge(GraphEdge::new(
            n1,
            n2,
            EdgeType::Temporal {
                average_interval_ms: 100,
                sequence_confidence: 0.9,
            },
            0.9,
        ));

        // Component 2: n3-n4
        graph.add_edge(GraphEdge::new(
            n3,
            n4,
            EdgeType::Temporal {
                average_interval_ms: 100,
                sequence_confidence: 0.9,
            },
            0.9,
        ));

        let analytics = GraphAnalytics::new(&graph);
        let components = analytics.count_components().unwrap();

        assert_eq!(components, 2);
    }
}
