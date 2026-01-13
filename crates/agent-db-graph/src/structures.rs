//! Core graph data structures
//!
//! Implements the graph data structures used for modeling relationships
//! between agents, events, and contexts in the agentic database.

use agent_db_core::types::{EventId, AgentId, Timestamp, ContextHash};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::hash::{Hash, Hasher};

/// Unique identifier for graph nodes
pub type NodeId = u64;

/// Unique identifier for graph edges  
pub type EdgeId = u64;

/// Weight type for edges (can represent similarity, causality strength, etc.)
pub type EdgeWeight = f32;

/// Core graph node representing entities in the system
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphNode {
    /// Unique node identifier
    pub id: NodeId,
    
    /// Node type and associated data
    pub node_type: NodeType,
    
    /// Creation timestamp
    pub created_at: Timestamp,
    
    /// Last update timestamp
    pub updated_at: Timestamp,
    
    /// Node properties for additional metadata
    pub properties: HashMap<String, serde_json::Value>,
    
    /// Cached degree for performance
    pub degree: u32,
}

/// Types of nodes in the graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum NodeType {
    /// Agent node representing an intelligent agent
    Agent {
        agent_id: AgentId,
        agent_type: String,
        capabilities: Vec<String>,
    },
    
    /// Event node representing a specific event
    Event {
        event_id: EventId,
        event_type: String,
        significance: f32, // 0.0 to 1.0
    },
    
    /// Context node representing a specific environmental context
    Context {
        context_hash: ContextHash,
        context_type: String,
        frequency: u32, // How often this context appears
    },
    
    /// Concept node representing abstract concepts learned from patterns
    Concept {
        concept_name: String,
        concept_type: ConceptType,
        confidence: f32, // 0.0 to 1.0
    },
    
    /// Goal node representing agent objectives
    Goal {
        goal_id: u64,
        description: String,
        priority: f32,
        status: GoalStatus,
    },
}

/// Types of abstract concepts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ConceptType {
    /// Learned behavioral pattern
    BehaviorPattern,
    /// Causal relationship pattern  
    CausalPattern,
    /// Temporal pattern
    TemporalPattern,
    /// Contextual association
    ContextualAssociation,
    /// Goal-oriented strategy
    Strategy,
}

/// Goal status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GoalStatus {
    Active,
    Completed,
    Failed,
    Paused,
}

/// Graph edge representing relationships between nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphEdge {
    /// Unique edge identifier
    pub id: EdgeId,
    
    /// Source node
    pub source: NodeId,
    
    /// Target node
    pub target: NodeId,
    
    /// Edge type and metadata
    pub edge_type: EdgeType,
    
    /// Edge weight/strength
    pub weight: EdgeWeight,
    
    /// Creation timestamp
    pub created_at: Timestamp,
    
    /// Last update timestamp
    pub updated_at: Timestamp,
    
    /// Number of times this relationship has been observed
    pub observation_count: u32,
    
    /// Confidence in this relationship (0.0 to 1.0)
    pub confidence: f32,
    
    /// Additional edge properties
    pub properties: HashMap<String, serde_json::Value>,
}

/// Types of relationships between nodes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EdgeType {
    /// Causal relationship (A causes B)
    Causality {
        strength: f32, // How strong the causal link is
        lag_ms: u64,   // Average time lag between cause and effect
    },
    
    /// Temporal sequence (A happens before B)
    Temporal {
        average_interval_ms: u64,
        sequence_confidence: f32,
    },
    
    /// Spatial/contextual relationship
    Contextual {
        similarity: f32,          // Context similarity score
        co_occurrence_rate: f32,  // How often they occur together
    },
    
    /// Agent interaction
    Interaction {
        interaction_type: InteractionType,
        frequency: u32,
        success_rate: f32,
    },
    
    /// Goal hierarchy (parent-child, dependency)
    GoalRelation {
        relation_type: GoalRelationType,
        dependency_strength: f32,
    },
    
    /// Learned association (inferred relationship)
    Association {
        association_type: String,
        evidence_count: u32,
        statistical_significance: f32,
    },
    
    /// Communication channel
    Communication {
        bandwidth: f32,           // Information flow rate
        reliability: f32,         // Communication success rate
        protocol: String,         // Communication method
    },
}

/// Types of agent interactions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum InteractionType {
    Collaboration,
    Competition,
    Coordination,
    InformationExchange,
    ResourceSharing,
    Communication,
    Conflict,
}

/// Types of goal relationships
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GoalRelationType {
    SubGoal,      // Target is a subgoal of source
    Dependency,   // Target must complete before source
    Conflict,     // Goals are mutually exclusive
    Support,      // Goals are mutually supportive
}

/// Graph structure with optimized storage and indexing
#[derive(Debug)]
pub struct Graph {
    /// All nodes in the graph
    nodes: HashMap<NodeId, GraphNode>,
    
    /// All edges in the graph
    edges: HashMap<EdgeId, GraphEdge>,
    
    /// Adjacency list for fast traversal (outgoing edges)
    adjacency_out: HashMap<NodeId, Vec<EdgeId>>,
    
    /// Reverse adjacency list (incoming edges)
    adjacency_in: HashMap<NodeId, Vec<EdgeId>>,
    
    /// Index by node type for fast filtering
    type_index: HashMap<String, HashSet<NodeId>>,
    
    /// Spatial index for context nodes
    context_index: HashMap<ContextHash, NodeId>,
    
    /// Agent index for quick agent lookup
    agent_index: HashMap<AgentId, NodeId>,
    
    /// Event index for event-node mapping
    event_index: HashMap<EventId, NodeId>,
    
    /// Next available IDs
    next_node_id: NodeId,
    next_edge_id: EdgeId,
    
    /// Statistics
    stats: GraphStats,
}

/// Graph statistics for monitoring and optimization
#[derive(Debug, Clone, Default)]
pub struct GraphStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub avg_degree: f32,
    pub max_degree: u32,
    pub component_count: usize,
    pub largest_component_size: usize,
    pub clustering_coefficient: f32,
    pub last_updated: Timestamp,
}

impl GraphNode {
    /// Create a new graph node
    pub fn new(node_type: NodeType) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as Timestamp;
            
        Self {
            id: 0, // Will be set by graph when added
            node_type,
            created_at: timestamp,
            updated_at: timestamp,
            properties: HashMap::new(),
            degree: 0,
        }
    }
    
    /// Get the type name of this node
    pub fn type_name(&self) -> String {
        match &self.node_type {
            NodeType::Agent { .. } => "Agent".to_string(),
            NodeType::Event { .. } => "Event".to_string(),
            NodeType::Context { .. } => "Context".to_string(),
            NodeType::Concept { .. } => "Concept".to_string(),
            NodeType::Goal { .. } => "Goal".to_string(),
        }
    }
    
    /// Update the node's timestamp
    pub fn touch(&mut self) {
        self.updated_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as Timestamp;
    }
}

impl GraphEdge {
    /// Create a new graph edge
    pub fn new(source: NodeId, target: NodeId, edge_type: EdgeType, weight: EdgeWeight) -> Self {
        let timestamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as Timestamp;
            
        Self {
            id: 0, // Will be set by graph when added
            source,
            target,
            edge_type,
            weight,
            created_at: timestamp,
            updated_at: timestamp,
            observation_count: 1,
            confidence: 0.5, // Start with medium confidence
            properties: HashMap::new(),
        }
    }
    
    /// Strengthen the edge based on repeated observation
    pub fn strengthen(&mut self, weight_delta: EdgeWeight) {
        self.observation_count += 1;
        self.weight = (self.weight + weight_delta).min(1.0).max(0.0);
        
        // Increase confidence based on observations
        let confidence_boost = (self.observation_count as f32).ln() * 0.1;
        self.confidence = (self.confidence + confidence_boost).min(1.0);
        
        self.touch();
    }
    
    /// Update the edge's timestamp
    pub fn touch(&mut self) {
        self.updated_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as Timestamp;
    }
}

impl Graph {
    /// Create a new empty graph
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            adjacency_out: HashMap::new(),
            adjacency_in: HashMap::new(),
            type_index: HashMap::new(),
            context_index: HashMap::new(),
            agent_index: HashMap::new(),
            event_index: HashMap::new(),
            next_node_id: 1,
            next_edge_id: 1,
            stats: GraphStats::default(),
        }
    }
    
    /// Add a node to the graph
    pub fn add_node(&mut self, mut node: GraphNode) -> NodeId {
        let node_id = self.next_node_id;
        self.next_node_id += 1;
        
        node.id = node_id;
        
        // Update type index
        let type_name = node.type_name();
        self.type_index.entry(type_name).or_insert_with(HashSet::new).insert(node_id);
        
        // Update specialized indices
        match &node.node_type {
            NodeType::Agent { agent_id, .. } => {
                self.agent_index.insert(*agent_id, node_id);
            }
            NodeType::Event { event_id, .. } => {
                self.event_index.insert(*event_id, node_id);
            }
            NodeType::Context { context_hash, .. } => {
                self.context_index.insert(*context_hash, node_id);
            }
            _ => {}
        }
        
        // Initialize adjacency lists
        self.adjacency_out.insert(node_id, Vec::new());
        self.adjacency_in.insert(node_id, Vec::new());
        
        self.nodes.insert(node_id, node);
        self.update_stats();
        
        node_id
    }
    
    /// Add an edge to the graph
    pub fn add_edge(&mut self, mut edge: GraphEdge) -> Option<EdgeId> {
        // Verify both nodes exist
        if !self.nodes.contains_key(&edge.source) || !self.nodes.contains_key(&edge.target) {
            return None;
        }
        
        let edge_id = self.next_edge_id;
        self.next_edge_id += 1;
        
        edge.id = edge_id;
        
        // Update adjacency lists
        self.adjacency_out.get_mut(&edge.source)?.push(edge_id);
        self.adjacency_in.get_mut(&edge.target)?.push(edge_id);
        
        // Update node degrees
        if let Some(source_node) = self.nodes.get_mut(&edge.source) {
            source_node.degree += 1;
            source_node.touch();
        }
        if let Some(target_node) = self.nodes.get_mut(&edge.target) {
            target_node.degree += 1;
            target_node.touch();
        }
        
        self.edges.insert(edge_id, edge);
        self.update_stats();
        
        Some(edge_id)
    }
    
    /// Get neighbors of a node (outgoing edges)
    pub fn get_neighbors(&self, node_id: NodeId) -> Vec<NodeId> {
        self.adjacency_out
            .get(&node_id)
            .unwrap_or(&Vec::new())
            .iter()
            .filter_map(|&edge_id| {
                self.edges.get(&edge_id).map(|edge| edge.target)
            })
            .collect()
    }
    
    /// Get incoming neighbors of a node
    pub fn get_incoming_neighbors(&self, node_id: NodeId) -> Vec<NodeId> {
        self.adjacency_in
            .get(&node_id)
            .unwrap_or(&Vec::new())
            .iter()
            .filter_map(|&edge_id| {
                self.edges.get(&edge_id).map(|edge| edge.source)
            })
            .collect()
    }
    
    /// Get all nodes of a specific type
    pub fn get_nodes_by_type(&self, type_name: &str) -> Vec<&GraphNode> {
        self.type_index
            .get(type_name)
            .unwrap_or(&HashSet::new())
            .iter()
            .filter_map(|&node_id| self.nodes.get(&node_id))
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
    
    /// Update graph statistics
    fn update_stats(&mut self) {
        self.stats.node_count = self.nodes.len();
        self.stats.edge_count = self.edges.len();
        
        if !self.nodes.is_empty() {
            let total_degree: u32 = self.nodes.values().map(|n| n.degree).sum();
            self.stats.avg_degree = total_degree as f32 / self.nodes.len() as f32;
            self.stats.max_degree = self.nodes.values().map(|n| n.degree).max().unwrap_or(0);
        }
        
        self.stats.last_updated = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as Timestamp;
    }
    
    /// Get graph statistics
    pub fn stats(&self) -> &GraphStats {
        &self.stats
    }
    
    /// Get node by ID
    pub fn get_node(&self, node_id: NodeId) -> Option<&GraphNode> {
        self.nodes.get(&node_id)
    }
    
    /// Get edge by ID
    pub fn get_edge(&self, edge_id: EdgeId) -> Option<&GraphEdge> {
        self.edges.get(&edge_id)
    }
    
    /// Get node by agent ID
    pub fn get_agent_node(&self, agent_id: AgentId) -> Option<&GraphNode> {
        self.agent_index.get(&agent_id).and_then(|&node_id| self.nodes.get(&node_id))
    }
    
    /// Get node by event ID
    pub fn get_event_node(&self, event_id: EventId) -> Option<&GraphNode> {
        self.event_index.get(&event_id).and_then(|&node_id| self.nodes.get(&node_id))
    }
    
    /// Get node by context hash
    pub fn get_context_node(&self, context_hash: ContextHash) -> Option<&GraphNode> {
        self.context_index.get(&context_hash).and_then(|&node_id| self.nodes.get(&node_id))
    }
    
    /// Get mutable reference to node by ID
    pub fn get_node_mut(&mut self, node_id: NodeId) -> Option<&mut GraphNode> {
        self.nodes.get_mut(&node_id)
    }
}