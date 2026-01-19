//! Graph traversal and query engine
//!
//! This module provides advanced graph traversal capabilities and query
//! processing for the agentic database graph structure.

use crate::{GraphResult, GraphError};
use crate::structures::{Graph, GraphNode, GraphEdge, NodeType, NodeId, EdgeId, EdgeWeight};
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
    #[allow(dead_code)]
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
        if let Some((_cached_result, timestamp)) = self.query_cache.get(&cache_key) {
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
        _graph: &Graph,
        _key: &str,
        _value: &serde_json::Value,
    ) -> GraphResult<QueryResult> {
        let matching_nodes = Vec::new();
        
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
        _constraints: &[PathConstraint],
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

    // ========================================
    // Policy Guide Queries
    // ========================================
    // Methods to help agents decide "what usually works next"
    // based on graph patterns and historical success rates

    /// Get next step suggestions based on current context and last action
    ///
    /// Returns actions that have historically worked well in similar contexts,
    /// ranked by success probability
    pub fn get_next_step_suggestions(
        &self,
        graph: &Graph,
        current_context_hash: agent_db_core::types::ContextHash,
        last_action_node: Option<NodeId>,
        limit: usize,
    ) -> GraphResult<Vec<ActionSuggestion>> {
        let mut suggestions = Vec::new();

        // Strategy 1: If we have a last action, find successful continuations
        if let Some(action_node_id) = last_action_node {
            let continuations = self.get_successful_continuations(graph, action_node_id)?;
            suggestions.extend(continuations);
        }

        // Strategy 2: Find actions that work well in this context
        let context_actions = self.get_actions_for_context(graph, current_context_hash)?;
        suggestions.extend(context_actions);

        // Remove dead ends
        let dead_ends = self.get_dead_ends(graph, current_context_hash)?;
        suggestions.retain(|s| !dead_ends.contains(&s.action_name));

        // Sort by success probability (descending)
        suggestions.sort_by(|a, b| {
            b.success_probability
                .partial_cmp(&a.success_probability)
                .unwrap_or(Ordering::Equal)
        });

        // Deduplicate by action name, keeping highest probability
        let mut seen = HashSet::new();
        suggestions.retain(|s| seen.insert(s.action_name.clone()));

        // Return top-k
        Ok(suggestions.into_iter().take(limit).collect())
    }

    /// Find actions that have successfully followed a given action
    ///
    /// Looks at graph edges from the action node to find actions that
    /// have been performed next and succeeded
    pub fn get_successful_continuations(
        &self,
        graph: &Graph,
        from_action_node: NodeId,
    ) -> GraphResult<Vec<ActionSuggestion>> {
        let mut suggestions = Vec::new();

        // Get all outgoing edges from this action node
        let neighbors = graph.get_neighbors(from_action_node);

        for neighbor_id in neighbors {
            // Get edge weight by finding the edge between these nodes
            if let Some(edge_weight) = self.get_edge_weight_between(graph, from_action_node, neighbor_id) {
                // Only consider Event nodes (which contain action information)
                if let Some(node) = graph.get_node(neighbor_id) {
                    if let NodeType::Event { event_type, .. } = &node.node_type {
                        // Higher edge weight = more successful pattern
                        let success_probability = edge_weight.min(1.0);

                        // Count evidence (how many times this pattern occurred)
                        let evidence_count = self.count_pattern_occurrences(
                            graph,
                            from_action_node,
                            neighbor_id,
                        )?;

                        suggestions.push(ActionSuggestion {
                            action_name: event_type.clone(),
                            action_node_id: neighbor_id,
                            success_probability,
                            evidence_count,
                            reasoning: format!(
                                "This action has followed the previous action {} times with {:.1}% success rate",
                                evidence_count,
                                success_probability * 100.0
                            ),
                        });
                    }
                }
            }
        }

        Ok(suggestions)
    }

    /// Get actions that work well in a specific context
    ///
    /// Finds actions that have been successful in similar contexts
    fn get_actions_for_context(
        &self,
        graph: &Graph,
        context_hash: agent_db_core::types::ContextHash,
    ) -> GraphResult<Vec<ActionSuggestion>> {
        let mut suggestions = Vec::new();

        // Find context nodes matching or similar to this context
        let context_nodes = self.find_context_nodes(graph, context_hash)?;

        for context_node_id in context_nodes {
            // Get actions connected to this context
            let neighbors = graph.get_neighbors(context_node_id);

            for neighbor_id in neighbors {
                if let Some(node) = graph.get_node(neighbor_id) {
                    if let NodeType::Event { event_type, .. } = &node.node_type {
                        // Get edge weight (success rate in this context)
                        let success_probability = self
                            .get_edge_weight_between(graph, context_node_id, neighbor_id)
                            .unwrap_or(0.5);

                        let evidence_count = self.count_pattern_occurrences(
                            graph,
                            context_node_id,
                            neighbor_id,
                        )?;

                        suggestions.push(ActionSuggestion {
                            action_name: event_type.clone(),
                            action_node_id: neighbor_id,
                            success_probability,
                            evidence_count,
                            reasoning: format!(
                                "This action has worked well in similar contexts ({} times, {:.1}% success)",
                                evidence_count,
                                success_probability * 100.0
                            ),
                        });
                    }
                }
            }
        }

        Ok(suggestions)
    }

    /// Get actions that are known to fail in this context (dead ends)
    ///
    /// Returns action names that should be avoided based on actual failure rates
    pub fn get_dead_ends(
        &self,
        graph: &Graph,
        context_hash: agent_db_core::types::ContextHash,
    ) -> GraphResult<Vec<String>> {
        let mut dead_ends = Vec::new();

        // Find context nodes
        let context_nodes = self.find_context_nodes(graph, context_hash)?;

        for context_node_id in context_nodes {
            let neighbors = graph.get_neighbors(context_node_id);

            for neighbor_id in neighbors {
                if let Some(node) = graph.get_node(neighbor_id) {
                    if let NodeType::Event { event_type, .. } = &node.node_type {
                        // Get the actual edge to check failure rate
                        if let Some(edge) = graph.get_edge_between(context_node_id, neighbor_id) {
                            let success_count = edge.get_success_count();
                            let failure_count = edge.get_failure_count();
                            let total = success_count + failure_count;
                            
                            // Dead end if:
                            // 1. Low success rate (< 20%) AND at least 3 observations
                            // 2. OR failure rate > 70% (strong negative signal)
                            if total >= 3 {
                                let failure_rate = failure_count as f32 / total as f32;
                                let success_rate = success_count as f32 / total as f32;
                                
                                if failure_rate > 0.7 || success_rate < 0.2 {
                                    dead_ends.push(event_type.clone());
                                }
                            } else if let Some(success_rate) = edge.get_success_rate() {
                                // Use computed success rate if available
                                if success_rate < 0.2 {
                                    dead_ends.push(event_type.clone());
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(dead_ends)
    }

    // Helper methods

    /// Find context nodes matching or similar to the given context hash
    fn find_context_nodes(
        &self,
        graph: &Graph,
        context_hash: agent_db_core::types::ContextHash,
    ) -> GraphResult<Vec<NodeId>> {
        let mut matching_nodes = Vec::new();

        // Use graph method to get context node directly
        if let Some(context_node) = graph.get_context_node(context_hash) {
            matching_nodes.push(context_node.id);
        }

        Ok(matching_nodes)
    }

    /// Get edge weight between two nodes, using success rate if available
    fn get_edge_weight_between(
        &self,
        graph: &Graph,
        from_node: NodeId,
        to_node: NodeId,
    ) -> Option<EdgeWeight> {
        // Find the edge between these nodes
        if let Some(edge) = graph.get_edge_between(from_node, to_node) {
            // Prefer success rate if we have enough evidence
            if let Some(success_rate) = edge.get_success_rate() {
                let total_observations = edge.get_success_count() + edge.get_failure_count();
                
                // If we have at least 3 observations, use success rate
                // Otherwise, blend with edge weight (prior)
                if total_observations >= 3 {
                    Some(success_rate)
                } else {
                    // Bayesian prior: blend success rate with edge weight
                    let prior_weight = edge.weight;
                    let evidence_weight = total_observations as f32;
                    let prior_strength = 2.0; // Equivalent to 2 prior observations
                    
                    let blended = (success_rate * evidence_weight + prior_weight * prior_strength) 
                        / (evidence_weight + prior_strength);
                    Some(blended.clamp(0.0, 1.0))
                }
            } else {
                // No success/failure data yet, use edge weight
                Some(edge.weight)
            }
        } else {
            None
        }
    }

    /// Count how many times a pattern (edge) has occurred
    fn count_pattern_occurrences(
        &self,
        graph: &Graph,
        from_node: NodeId,
        to_node: NodeId,
    ) -> GraphResult<u32> {
        if let Some(edge) = graph.get_edge_between(from_node, to_node) {
            // Use observation_count which tracks all occurrences
            Ok(edge.observation_count)
        } else {
            Ok(0)
        }
    }
}

/// Suggestion for next action based on historical patterns
#[derive(Debug, Clone)]
pub struct ActionSuggestion {
    /// Name of the suggested action
    pub action_name: String,

    /// Node ID of the action in the graph
    pub action_node_id: NodeId,

    /// Probability of success (0.0 to 1.0)
    pub success_probability: f32,

    /// Number of times this pattern has been observed
    pub evidence_count: u32,

    /// Human-readable explanation of why this action is suggested
    pub reasoning: String,
}