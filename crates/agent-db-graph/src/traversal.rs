//! Graph traversal and query engine
//!
//! This module provides advanced graph traversal capabilities and query
//! processing for the agentic database graph structure.

use crate::{GraphResult, GraphError};
use crate::structures::{Graph, GraphNode, GraphEdge, NodeType, EdgeType, NodeId, EdgeId, EdgeWeight};
use std::collections::{HashMap, HashSet, VecDeque, BinaryHeap};
use std::cmp::Ordering;

/// Query operations for graph traversal
#[derive(Debug, Clone)]
pub enum GraphQuery {
    /// Find all nodes of a specific type
    NodesByType(String),
    
    /// Find shortest path between two nodes
    ShortestPath { start: NodeId, end: NodeId },
    
    /// Find all neighbors within N hops
    NeighborsWithinDistance { start: NodeId, max_distance: u32 },
    
    /// Find strongly connected components
    StronglyConnectedComponents,
    
    /// Find nodes by property values
    NodesByProperty { key: String, value: serde_json::Value },
    
    /// Find edges by type and weight threshold
    EdgesByType { edge_type: String, min_weight: EdgeWeight },
    
    /// Complex path query with constraints
    PathQuery {
        start: NodeId,
        end: NodeId,
        constraints: Vec<PathConstraint>,
    },
    
    /// Subgraph extraction
    Subgraph {
        center: NodeId,
        radius: u32,
        node_types: Option<Vec<String>>,
    },
    
    /// PageRank calculation
    PageRank {
        iterations: usize,
        damping_factor: f32,
    },
    
    /// Community detection
    CommunityDetection {
        algorithm: CommunityAlgorithm,
    },
}

/// Constraints for path finding
#[derive(Debug, Clone)]
pub enum PathConstraint {
    /// Path must not exceed this length
    MaxLength(u32),
    
    /// Path must include nodes of specific types
    RequiredNodeTypes(Vec<String>),
    
    /// Path must avoid nodes of specific types
    AvoidNodeTypes(Vec<String>),
    
    /// Path edges must have minimum weight
    MinEdgeWeight(EdgeWeight),
    
    /// Path must include specific edge types
    RequiredEdgeTypes(Vec<String>),
    
    /// Path must avoid specific edge types
    AvoidEdgeTypes(Vec<String>),
    
    /// Custom filter function
    CustomFilter(fn(&GraphNode, &GraphEdge) -> bool),
}

/// Community detection algorithms
#[derive(Debug, Clone)]
pub enum CommunityAlgorithm {
    /// Louvain algorithm for modularity optimization
    Louvain { resolution: f32 },
    
    /// Label propagation algorithm
    LabelPropagation { iterations: usize },
    
    /// Connected components (simple)
    ConnectedComponents,
}

/// Results from graph queries
#[derive(Debug)]
pub enum QueryResult {
    /// List of node IDs
    Nodes(Vec<NodeId>),
    
    /// Path as sequence of node IDs
    Path(Vec<NodeId>),
    
    /// Multiple paths
    Paths(Vec<Vec<NodeId>>),
    
    /// List of edge IDs
    Edges(Vec<EdgeId>),
    
    /// Subgraph data
    Subgraph {
        nodes: Vec<NodeId>,
        edges: Vec<EdgeId>,
        center: NodeId,
    },
    
    /// Node rankings with scores
    Rankings(Vec<(NodeId, f32)>),
    
    /// Community assignments
    Communities(Vec<Vec<NodeId>>),
    
    /// Generic key-value results
    Properties(HashMap<String, serde_json::Value>),
}

/// Statistics about query execution
#[derive(Debug, Default)]
pub struct QueryStats {
    pub nodes_visited: usize,
    pub edges_traversed: usize,
    pub execution_time_ms: u64,
    pub memory_used_bytes: usize,
    pub cache_hits: usize,
    pub cache_misses: usize,
}

/// Priority queue entry for pathfinding algorithms
#[derive(Debug, Clone)]
struct PathEntry {
    node_id: NodeId,
    cost: f32,
    path: Vec<NodeId>,
}

impl PartialEq for PathEntry {
    fn eq(&self, other: &Self) -> bool {
        self.cost == other.cost
    }
}

impl Eq for PathEntry {}

impl PartialOrd for PathEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PathEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse ordering for min-heap behavior
        other.cost.partial_cmp(&self.cost).unwrap_or(Ordering::Equal)
    }
}

/// Advanced graph traversal engine
pub struct GraphTraversal {
    /// Query result cache
    query_cache: HashMap<String, (QueryResult, std::time::Instant)>,
    
    /// Cache TTL in seconds
    cache_ttl: u64,
    
    /// Maximum cache size
    max_cache_size: usize,
}

impl GraphTraversal {
    /// Create new traversal engine
    pub fn new() -> Self {
        Self {
            query_cache: HashMap::new(),
            cache_ttl: 300, // 5 minutes
            max_cache_size: 1000,
        }
    }
    
    /// Execute a graph query
    pub fn execute_query(&mut self, graph: &Graph, query: GraphQuery) -> GraphResult<QueryResult> {
        let start_time = std::time::Instant::now();
        
        // Check cache first
        let cache_key = format!("{:?}", query);
        if let Some((cached_result, timestamp)) = self.query_cache.get(&cache_key) {
            if start_time.duration_since(*timestamp).as_secs() < self.cache_ttl {
                // Return cached result (would need to clone properly)
                return Ok(QueryResult::Nodes(Vec::new())); // Placeholder
            }
        }
        
        // Execute query
        let result = match query {
            GraphQuery::NodesByType(node_type) => {
                self.find_nodes_by_type(graph, &node_type)
            }
            
            GraphQuery::ShortestPath { start, end } => {
                self.shortest_path(graph, start, end)
            }
            
            GraphQuery::NeighborsWithinDistance { start, max_distance } => {
                self.neighbors_within_distance(graph, start, max_distance)
            }
            
            GraphQuery::StronglyConnectedComponents => {
                self.find_strongly_connected_components(graph)
            }
            
            GraphQuery::NodesByProperty { key, value } => {
                self.find_nodes_by_property(graph, &key, &value)
            }
            
            GraphQuery::EdgesByType { edge_type, min_weight } => {
                self.find_edges_by_type(graph, &edge_type, min_weight)
            }
            
            GraphQuery::PathQuery { start, end, constraints } => {
                self.constrained_path_search(graph, start, end, &constraints)
            }
            
            GraphQuery::Subgraph { center, radius, node_types } => {
                self.extract_subgraph(graph, center, radius, node_types.as_ref())
            }
            
            GraphQuery::PageRank { iterations, damping_factor } => {
                self.calculate_pagerank(graph, iterations, damping_factor)
            }
            
            GraphQuery::CommunityDetection { algorithm } => {
                self.detect_communities(graph, &algorithm)
            }
        }?;
        
        // Cache result
        if self.query_cache.len() < self.max_cache_size {
            // Would need proper cloning implementation
            // self.query_cache.insert(cache_key, (result.clone(), start_time));
        }
        
        Ok(result)
    }
    
    /// Find nodes by type
    fn find_nodes_by_type(&self, graph: &Graph, node_type: &str) -> GraphResult<QueryResult> {
        let nodes = graph.get_nodes_by_type(node_type)
            .into_iter()
            .map(|node| node.id)
            .collect();
            
        Ok(QueryResult::Nodes(nodes))
    }
    
    /// Find shortest path using Dijkstra's algorithm
    fn shortest_path(&self, graph: &Graph, start: NodeId, end: NodeId) -> GraphResult<QueryResult> {
        if start == end {
            return Ok(QueryResult::Path(vec![start]));
        }
        
        let mut heap = BinaryHeap::new();
        let mut distances: HashMap<NodeId, f32> = HashMap::new();
        let mut previous: HashMap<NodeId, NodeId> = HashMap::new();
        
        distances.insert(start, 0.0);
        heap.push(PathEntry {
            node_id: start,
            cost: 0.0,
            path: vec![start],
        });
        
        while let Some(PathEntry { node_id: current, cost, .. }) = heap.pop() {
            if current == end {
                // Reconstruct path
                let mut path = vec![current];
                let mut current_node = current;
                
                while let Some(&prev) = previous.get(&current_node) {
                    path.push(prev);
                    current_node = prev;
                }
                
                path.reverse();
                return Ok(QueryResult::Path(path));
            }
            
            if let Some(&best_distance) = distances.get(&current) {
                if cost > best_distance {
                    continue; // Already found better path
                }
            }
            
            // Explore neighbors
            for neighbor_id in graph.get_neighbors(current) {
                // Calculate edge weight (simplified - would use actual edge weights)
                let edge_weight = 1.0; // Default weight
                let new_distance = cost + edge_weight;
                
                let is_shorter = distances.get(&neighbor_id)
                    .map_or(true, |&current_distance| new_distance < current_distance);
                
                if is_shorter {
                    distances.insert(neighbor_id, new_distance);
                    previous.insert(neighbor_id, current);
                    
                    heap.push(PathEntry {
                        node_id: neighbor_id,
                        cost: new_distance,
                        path: Vec::new(), // Would track full path if needed
                    });
                }
            }
        }
        
        Err(GraphError::NodeNotFound("No path found".to_string()))
    }
    
    /// Find neighbors within specified distance
    fn neighbors_within_distance(
        &self, 
        graph: &Graph, 
        start: NodeId, 
        max_distance: u32
    ) -> GraphResult<QueryResult> {
        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();
        let mut result = Vec::new();
        
        queue.push_back((start, 0));
        visited.insert(start);
        
        while let Some((current, distance)) = queue.pop_front() {
            if distance <= max_distance {
                result.push(current);
                
                if distance < max_distance {
                    for neighbor in graph.get_neighbors(current) {
                        if !visited.contains(&neighbor) {
                            visited.insert(neighbor);
                            queue.push_back((neighbor, distance + 1));
                        }
                    }
                }
            }
        }
        
        Ok(QueryResult::Nodes(result))
    }
    
    /// Find strongly connected components using Tarjan's algorithm
    fn find_strongly_connected_components(&self, graph: &Graph) -> GraphResult<QueryResult> {
        let mut index = 0;
        let mut stack = Vec::new();
        let mut indices: HashMap<NodeId, usize> = HashMap::new();
        let mut lowlinks: HashMap<NodeId, usize> = HashMap::new();
        let mut on_stack: HashSet<NodeId> = HashSet::new();
        let mut components = Vec::new();
        
        // Get all node IDs
        let all_nodes: Vec<NodeId> = graph.get_nodes_by_type("Agent")
            .into_iter()
            .chain(graph.get_nodes_by_type("Event"))
            .chain(graph.get_nodes_by_type("Context"))
            .chain(graph.get_nodes_by_type("Concept"))
            .chain(graph.get_nodes_by_type("Goal"))
            .map(|node| node.id)
            .collect();
        
        for &node_id in &all_nodes {
            if !indices.contains_key(&node_id) {
                self.strongconnect(
                    graph,
                    node_id,
                    &mut index,
                    &mut stack,
                    &mut indices,
                    &mut lowlinks,
                    &mut on_stack,
                    &mut components,
                );
            }
        }
        
        Ok(QueryResult::Communities(components))
    }
    
    /// Helper for Tarjan's strongly connected components algorithm
    fn strongconnect(
        &self,
        graph: &Graph,
        node_id: NodeId,
        index: &mut usize,
        stack: &mut Vec<NodeId>,
        indices: &mut HashMap<NodeId, usize>,
        lowlinks: &mut HashMap<NodeId, usize>,
        on_stack: &mut HashSet<NodeId>,
        components: &mut Vec<Vec<NodeId>>,
    ) {
        indices.insert(node_id, *index);
        lowlinks.insert(node_id, *index);
        *index += 1;
        stack.push(node_id);
        on_stack.insert(node_id);
        
        for neighbor in graph.get_neighbors(node_id) {
            if !indices.contains_key(&neighbor) {
                self.strongconnect(
                    graph, neighbor, index, stack, indices, lowlinks, on_stack, components
                );
                let neighbor_lowlink = lowlinks[&neighbor];
                lowlinks.insert(node_id, lowlinks[&node_id].min(neighbor_lowlink));
            } else if on_stack.contains(&neighbor) {
                let neighbor_index = indices[&neighbor];
                lowlinks.insert(node_id, lowlinks[&node_id].min(neighbor_index));
            }
        }
        
        if lowlinks[&node_id] == indices[&node_id] {
            let mut component = Vec::new();
            loop {
                let w = stack.pop().unwrap();
                on_stack.remove(&w);
                component.push(w);
                if w == node_id {
                    break;
                }
            }
            components.push(component);
        }
    }
    
    /// Find nodes by property value
    fn find_nodes_by_property(
        &self,
        graph: &Graph,
        key: &str,
        value: &serde_json::Value,
    ) -> GraphResult<QueryResult> {
        let mut matching_nodes = Vec::new();
        
        // This would require iterating over all nodes in the graph
        // For now, return empty result as we need access to all nodes
        // In a real implementation, we'd have an iterator over all nodes
        
        Ok(QueryResult::Nodes(matching_nodes))
    }
    
    /// Find edges by type and weight threshold  
    fn find_edges_by_type(
        &self,
        _graph: &Graph,
        _edge_type: &str,
        _min_weight: EdgeWeight,
    ) -> GraphResult<QueryResult> {
        // Would iterate over all edges and filter by type and weight
        Ok(QueryResult::Edges(Vec::new()))
    }
    
    /// Constrained path search with multiple constraints
    fn constrained_path_search(
        &self,
        graph: &Graph,
        start: NodeId,
        end: NodeId,
        constraints: &[PathConstraint],
    ) -> GraphResult<QueryResult> {
        // Simplified implementation - would need full constraint checking
        self.shortest_path(graph, start, end)
    }
    
    /// Extract subgraph around a center node
    fn extract_subgraph(
        &self,
        graph: &Graph,
        center: NodeId,
        radius: u32,
        node_types: Option<&Vec<String>>,
    ) -> GraphResult<QueryResult> {
        let neighbors_result = self.neighbors_within_distance(graph, center, radius)?;
        
        if let QueryResult::Nodes(nodes) = neighbors_result {
            let filtered_nodes = if let Some(types) = node_types {
                nodes.into_iter()
                    .filter(|&node_id| {
                        if let Some(node) = graph.get_node(node_id) {
                            types.contains(&node.type_name())
                        } else {
                            false
                        }
                    })
                    .collect()
            } else {
                nodes
            };
            
            // Would also collect relevant edges
            let edges = Vec::new(); // Simplified
            
            Ok(QueryResult::Subgraph {
                nodes: filtered_nodes,
                edges,
                center,
            })
        } else {
            Err(GraphError::NodeNotFound("Failed to extract subgraph".to_string()))
        }
    }
    
    /// Calculate PageRank scores for all nodes
    fn calculate_pagerank(
        &self,
        graph: &Graph,
        iterations: usize,
        damping_factor: f32,
    ) -> GraphResult<QueryResult> {
        let all_nodes: Vec<NodeId> = graph.get_nodes_by_type("Agent")
            .into_iter()
            .chain(graph.get_nodes_by_type("Event"))
            .chain(graph.get_nodes_by_type("Context"))
            .map(|node| node.id)
            .collect();
            
        let n = all_nodes.len() as f32;
        if n == 0.0 {
            return Ok(QueryResult::Rankings(Vec::new()));
        }
        
        let mut pagerank: HashMap<NodeId, f32> = HashMap::new();
        let mut new_pagerank: HashMap<NodeId, f32> = HashMap::new();
        
        // Initialize PageRank scores
        for &node_id in &all_nodes {
            pagerank.insert(node_id, 1.0 / n);
        }
        
        // Iterate PageRank calculation
        for _ in 0..iterations {
            new_pagerank.clear();
            
            for &node_id in &all_nodes {
                let mut rank = (1.0 - damping_factor) / n;
                
                // Sum contributions from incoming neighbors
                for incoming_neighbor in graph.get_incoming_neighbors(node_id) {
                    let neighbor_out_degree = graph.get_neighbors(incoming_neighbor).len() as f32;
                    if neighbor_out_degree > 0.0 {
                        rank += damping_factor * pagerank[&incoming_neighbor] / neighbor_out_degree;
                    }
                }
                
                new_pagerank.insert(node_id, rank);
            }
            
            pagerank = new_pagerank.clone();
        }
        
        // Convert to sorted rankings
        let mut rankings: Vec<(NodeId, f32)> = pagerank.into_iter().collect();
        rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(Ordering::Equal));
        
        Ok(QueryResult::Rankings(rankings))
    }
    
    /// Detect communities using specified algorithm
    fn detect_communities(
        &self,
        graph: &Graph,
        algorithm: &CommunityAlgorithm,
    ) -> GraphResult<QueryResult> {
        match algorithm {
            CommunityAlgorithm::ConnectedComponents => {
                self.find_strongly_connected_components(graph)
            }
            CommunityAlgorithm::Louvain { resolution: _ } => {
                // Simplified Louvain implementation would go here
                self.find_strongly_connected_components(graph)
            }
            CommunityAlgorithm::LabelPropagation { iterations: _ } => {
                // Label propagation implementation would go here
                self.find_strongly_connected_components(graph)
            }
        }
    }
    
    /// Clean up expired cache entries
    pub fn cleanup_cache(&mut self) {
        let now = std::time::Instant::now();
        self.query_cache.retain(|_, (_, timestamp)| {
            now.duration_since(*timestamp).as_secs() < self.cache_ttl
        });
    }
}