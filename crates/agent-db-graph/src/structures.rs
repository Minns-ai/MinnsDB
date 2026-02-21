//! Core graph data structures
//!
//! Implements the graph data structures used for modeling relationships
//! between agents, events, and contexts in the agentic database.

use agent_db_core::types::{AgentId, ContextHash, EventId, Timestamp};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};

/// Unique identifier for graph nodes
pub type NodeId = u64;

/// Unique identifier for graph edges
pub type EdgeId = u64;

/// Weight type for edges (can represent similarity, causality strength, etc.)
pub type EdgeWeight = f32;

/// Unique identifier for goal buckets (semantic partitions)
pub type GoalBucketId = u64;

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

    /// Episode node representing a coherent experience sequence
    Episode {
        episode_id: u64,
        agent_id: AgentId,
        session_id: u64,
        outcome: String,
    },

    /// Memory node representing a stored experience
    Memory {
        memory_id: u64,
        agent_id: AgentId,
        session_id: u64,
    },

    /// Strategy node representing a reusable plan
    Strategy {
        strategy_id: u64,
        agent_id: AgentId,
        name: String,
    },

    /// Tool node representing external tools or resources
    Tool {
        tool_name: String,
        tool_type: String,
    },

    /// Result node representing outputs or artifacts
    Result {
        result_key: String,
        result_type: String,
        summary: String,
    },

    /// Claim node representing semantic memory (derived atomic facts)
    Claim {
        claim_id: u64,
        claim_text: String,
        confidence: f32,
        source_event_id: EventId,
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
    // ── NER-derived concept types ──
    /// Person entity (NER label: PERSON, PER)
    Person,
    /// Organization entity (NER label: ORG)
    Organization,
    /// Location/place entity (NER label: LOC, GPE)
    Location,
    /// Product/brand entity (NER label: PRODUCT)
    Product,
    /// Date/time entity (NER label: DATE, TIME)
    DateTime,
    /// Event entity (NER label: EVENT)
    Event,
    /// Miscellaneous named entity (NER label: MISC, NORP, WORK_OF_ART, etc.)
    NamedEntity,
}

impl ConceptType {
    /// Map a NER label string to a `ConceptType`.
    pub fn from_ner_label(label: &str) -> Self {
        match label.to_uppercase().as_str() {
            "PERSON" | "PER" => ConceptType::Person,
            "ORG" | "ORGANIZATION" => ConceptType::Organization,
            "LOC" | "GPE" | "FAC" | "LOCATION" => ConceptType::Location,
            "PRODUCT" | "BRAND" => ConceptType::Product,
            "DATE" | "TIME" => ConceptType::DateTime,
            "EVENT" => ConceptType::Event,
            "MISC" | "NORP" | "WORK_OF_ART" | "LAW" | "LANGUAGE" => ConceptType::NamedEntity,
            _ => ConceptType::ContextualAssociation,
        }
    }
}

/// Goal status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GoalStatus {
    Active,
    Completed,
    Failed,
    Paused,
}

impl NodeType {
    /// Numeric discriminant for compact header storage.
    pub fn discriminant(&self) -> u8 {
        match self {
            NodeType::Agent { .. } => 0,
            NodeType::Event { .. } => 1,
            NodeType::Context { .. } => 2,
            NodeType::Concept { .. } => 3,
            NodeType::Goal { .. } => 4,
            NodeType::Episode { .. } => 5,
            NodeType::Memory { .. } => 6,
            NodeType::Strategy { .. } => 7,
            NodeType::Tool { .. } => 8,
            NodeType::Result { .. } => 9,
            NodeType::Claim { .. } => 10,
        }
    }

    /// Importance signal for this node type (0.0–1.0).
    /// Higher = more important to keep in memory.
    pub fn signal(&self) -> f32 {
        match self {
            NodeType::Agent { .. } => 1.0,
            NodeType::Goal { priority, .. } => 0.8 + (*priority * 0.2).min(0.2),
            NodeType::Strategy { .. } => 0.7,
            NodeType::Memory { .. } => 0.6,
            NodeType::Tool { .. } => 0.6,
            NodeType::Concept { confidence, .. } => 0.4 + (*confidence * 0.3),
            NodeType::Claim { confidence, .. } => 0.3 + (*confidence * 0.3),
            NodeType::Episode { .. } => 0.2,
            NodeType::Event { significance, .. } => 0.1 + (*significance * 0.2),
            NodeType::Context { .. } => 0.15,
            NodeType::Result { .. } => 0.1,
        }
    }

    /// Eviction tier for hot/cold cache management.
    pub fn eviction_tier(&self) -> crate::graph_store::EvictionTier {
        use crate::graph_store::EvictionTier;
        match self {
            NodeType::Agent { .. } | NodeType::Goal { .. } => EvictionTier::Protected,
            NodeType::Strategy { .. } | NodeType::Memory { .. } | NodeType::Tool { .. } => {
                EvictionTier::Important
            },
            NodeType::Concept { .. } | NodeType::Claim { .. } => EvictionTier::Standard,
            NodeType::Context { .. }
            | NodeType::Episode { .. }
            | NodeType::Event { .. }
            | NodeType::Result { .. } => EvictionTier::Ephemeral,
        }
    }

    /// Derive the goal bucket for storage partitioning.
    /// Agent-scoped nodes use agent_id; global nodes use 0.
    pub fn goal_bucket(&self) -> GoalBucketId {
        match self {
            NodeType::Agent { agent_id, .. }
            | NodeType::Episode { agent_id, .. }
            | NodeType::Memory { agent_id, .. }
            | NodeType::Strategy { agent_id, .. } => *agent_id,
            NodeType::Event { event_id, .. } => (*event_id % 1024) as GoalBucketId,
            _ => 0, // Context, Concept, Goal, Tool, Result, Claim → default bucket
        }
    }
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
        similarity: f32,         // Context similarity score
        co_occurrence_rate: f32, // How often they occur together
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
        bandwidth: f32,   // Information flow rate
        reliability: f32, // Communication success rate
        protocol: String, // Communication method
    },

    /// Claim derivation (claim derived from source event/context)
    DerivedFrom {
        extraction_confidence: f32,
        extraction_timestamp: u64,
    },

    /// Evidence support (claim supported by text span/entity)
    SupportedBy {
        evidence_strength: f32,
        span_offset: (usize, usize), // start, end byte offsets
    },

    /// Semantic relationship (claim is about an entity/concept)
    About {
        relevance_score: f32,
        mention_count: u32,
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
    SubGoal,    // Target is a subgoal of source
    Dependency, // Target must complete before source
    Conflict,   // Goals are mutually exclusive
    Support,    // Goals are mutually supportive
}

/// Graph structure with optimized storage and indexing
#[derive(Debug)]
pub struct Graph {
    /// All nodes in the graph
    pub(crate) nodes: HashMap<NodeId, GraphNode>,

    /// All edges in the graph
    pub(crate) edges: HashMap<EdgeId, GraphEdge>,

    /// Adjacency list for fast traversal (outgoing edges)
    pub(crate) adjacency_out: HashMap<NodeId, Vec<EdgeId>>,

    /// Reverse adjacency list (incoming edges)
    pub(crate) adjacency_in: HashMap<NodeId, Vec<EdgeId>>,

    /// Index by node type for fast filtering
    pub(crate) type_index: HashMap<String, HashSet<NodeId>>,

    /// Spatial index for context nodes
    pub(crate) context_index: HashMap<ContextHash, NodeId>,

    /// Agent index for quick agent lookup
    pub(crate) agent_index: HashMap<AgentId, NodeId>,

    /// Event index for event-node mapping
    pub(crate) event_index: HashMap<EventId, NodeId>,

    /// Goal index for goal-node mapping
    pub(crate) goal_index: HashMap<u64, NodeId>,

    /// Episode index for episode-node mapping
    pub(crate) episode_index: HashMap<u64, NodeId>,

    /// Memory index for memory-node mapping
    pub(crate) memory_index: HashMap<u64, NodeId>,

    /// Strategy index for strategy-node mapping
    pub(crate) strategy_index: HashMap<u64, NodeId>,

    /// Tool index for tool-node mapping
    pub(crate) tool_index: HashMap<String, NodeId>,

    /// Result index for result-node mapping
    pub(crate) result_index: HashMap<String, NodeId>,

    /// Claim index for claim-node mapping
    pub(crate) claim_index: HashMap<u64, NodeId>,

    /// Concept index for concept-node mapping (by name)
    pub(crate) concept_index: HashMap<String, NodeId>,

    /// BM25 full-text search index
    pub(crate) bm25_index: crate::indexing::Bm25Index,

    /// Next available IDs
    pub(crate) next_node_id: NodeId,
    pub(crate) next_edge_id: EdgeId,

    /// Statistics
    pub(crate) stats: GraphStats,

    /// Maximum number of nodes allowed (enforced at add_node)
    pub(crate) max_graph_size: usize,
}

/// Graph statistics for monitoring and optimization
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
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
            NodeType::Episode { .. } => "Episode".to_string(),
            NodeType::Memory { .. } => "Memory".to_string(),
            NodeType::Strategy { .. } => "Strategy".to_string(),
            NodeType::Tool { .. } => "Tool".to_string(),
            NodeType::Result { .. } => "Result".to_string(),
            NodeType::Claim { .. } => "Claim".to_string(),
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
        self.weight = (self.weight + weight_delta).clamp(0.0, 1.0);

        // Increase confidence based on observations
        let confidence_boost = (self.observation_count as f32).ln() * 0.1;
        self.confidence = (self.confidence + confidence_boost).min(1.0);

        self.touch();
    }

    /// Weaken the edge based on negative outcomes
    pub fn weaken(&mut self, weight_delta: EdgeWeight) {
        self.observation_count += 1;
        self.weight = (self.weight - weight_delta).clamp(0.0, 1.0);

        // Decrease confidence on failures (but more slowly than increase)
        let confidence_penalty = (self.observation_count as f32).ln() * 0.05;
        self.confidence = (self.confidence - confidence_penalty).max(0.0);

        self.touch();
    }

    /// Record a success outcome for this edge
    pub fn record_success(&mut self) {
        let success_count = self
            .properties
            .get("success_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;

        self.properties.insert(
            "success_count".to_string(),
            serde_json::Value::Number((success_count + 1).into()),
        );

        self.update_success_rate();
        self.touch();
    }

    /// Record a failure outcome for this edge
    pub fn record_failure(&mut self) {
        let failure_count = self
            .properties
            .get("failure_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32;

        self.properties.insert(
            "failure_count".to_string(),
            serde_json::Value::Number((failure_count + 1).into()),
        );

        self.update_success_rate();
        self.touch();
    }

    /// Update success rate based on success/failure counts
    fn update_success_rate(&mut self) {
        let success_count = self
            .properties
            .get("success_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as f32;

        let failure_count = self
            .properties
            .get("failure_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as f32;

        let total = success_count + failure_count;
        if total > 0.0 {
            let success_rate = success_count / total;
            self.properties.insert(
                "success_rate".to_string(),
                serde_json::Value::Number(
                    serde_json::Number::from_f64(success_rate as f64)
                        .unwrap_or(serde_json::Number::from(0)),
                ),
            );
        }
    }

    /// Get success rate from properties (0.0 to 1.0)
    pub fn get_success_rate(&self) -> Option<f32> {
        self.properties
            .get("success_rate")
            .and_then(|v| v.as_f64())
            .map(|r| r as f32)
    }

    /// Get success count
    pub fn get_success_count(&self) -> u32 {
        self.properties
            .get("success_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32
    }

    /// Get failure count
    pub fn get_failure_count(&self) -> u32 {
        self.properties
            .get("failure_count")
            .and_then(|v| v.as_u64())
            .unwrap_or(0) as u32
    }

    /// Update the edge's timestamp
    pub fn touch(&mut self) {
        self.updated_at = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos() as Timestamp;
    }
}

impl Default for Graph {
    fn default() -> Self {
        Self::new()
    }
}

impl Graph {
    /// Create a new empty graph with default max size (1,000,000 nodes)
    pub fn new() -> Self {
        Self::with_max_size(1_000_000)
    }

    /// Create a new empty graph with a specific max node capacity
    pub fn with_max_size(max_graph_size: usize) -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            adjacency_out: HashMap::new(),
            adjacency_in: HashMap::new(),
            type_index: HashMap::new(),
            context_index: HashMap::new(),
            agent_index: HashMap::new(),
            event_index: HashMap::new(),
            goal_index: HashMap::new(),
            episode_index: HashMap::new(),
            memory_index: HashMap::new(),
            strategy_index: HashMap::new(),
            tool_index: HashMap::new(),
            result_index: HashMap::new(),
            claim_index: HashMap::new(),
            concept_index: HashMap::new(),
            bm25_index: crate::indexing::Bm25Index::new(),
            next_node_id: 1,
            next_edge_id: 1,
            stats: GraphStats::default(),
            max_graph_size,
        }
    }

    /// Add a node to the graph.
    ///
    /// Returns `GraphError::CapacityExceeded` when `nodes.len() >= max_graph_size`.
    /// Let the graph pruner free space before retrying.
    pub fn add_node(&mut self, mut node: GraphNode) -> crate::GraphResult<NodeId> {
        if self.nodes.len() >= self.max_graph_size {
            return Err(crate::GraphError::CapacityExceeded(format!(
                "graph has {} nodes (max {})",
                self.nodes.len(),
                self.max_graph_size
            )));
        }

        let node_id = self.next_node_id;
        self.next_node_id += 1;

        node.id = node_id;

        // Update type index
        let type_name = node.type_name();
        self.type_index
            .entry(type_name)
            .or_default()
            .insert(node_id);

        // Update specialized indices
        match &node.node_type {
            NodeType::Agent { agent_id, .. } => {
                self.agent_index.insert(*agent_id, node_id);
            },
            NodeType::Event { event_id, .. } => {
                self.event_index.insert(*event_id, node_id);
            },
            NodeType::Context { context_hash, .. } => {
                self.context_index.insert(*context_hash, node_id);
            },
            NodeType::Goal { goal_id, .. } => {
                self.goal_index.insert(*goal_id, node_id);
            },
            NodeType::Episode { episode_id, .. } => {
                self.episode_index.insert(*episode_id, node_id);
            },
            NodeType::Memory { memory_id, .. } => {
                self.memory_index.insert(*memory_id, node_id);
            },
            NodeType::Strategy { strategy_id, .. } => {
                self.strategy_index.insert(*strategy_id, node_id);
            },
            NodeType::Tool { tool_name, .. } => {
                self.tool_index.insert(tool_name.clone(), node_id);
            },
            NodeType::Result { result_key, .. } => {
                self.result_index.insert(result_key.clone(), node_id);
            },
            NodeType::Claim { claim_id, .. } => {
                self.claim_index.insert(*claim_id, node_id);
            },
            NodeType::Concept { concept_name, .. } => {
                self.concept_index.insert(concept_name.clone(), node_id);
            },
        }

        // Initialize adjacency lists
        self.adjacency_out.insert(node_id, Vec::new());
        self.adjacency_in.insert(node_id, Vec::new());

        // Index text content with BM25 for full-text search
        let mut text_parts = Vec::new();

        // Extract searchable text from NodeType fields
        match &node.node_type {
            NodeType::Claim { claim_text, .. } => {
                text_parts.push(claim_text.as_str());
            },
            NodeType::Goal { description, .. } => {
                text_parts.push(description.as_str());
            },
            NodeType::Strategy { name, .. } => {
                text_parts.push(name.as_str());
            },
            NodeType::Result { summary, .. } => {
                text_parts.push(summary.as_str());
            },
            NodeType::Concept { concept_name, .. } => {
                text_parts.push(concept_name.as_str());
            },
            NodeType::Tool { tool_name, .. } => {
                text_parts.push(tool_name.as_str());
            },
            NodeType::Episode { outcome, .. } => {
                text_parts.push(outcome.as_str());
            },
            _ => {}, // Other node types don't have text in NodeType
        }

        // Extract searchable text from common property keys
        for (key, value) in &node.properties {
            let key_lower = key.to_lowercase();
            if key_lower.contains("text")
                || key_lower.contains("description")
                || key_lower.contains("content")
                || key_lower.contains("name")
                || key_lower.contains("summary")
                || key_lower == "data"
            {
                if let Some(text) = value.as_str() {
                    text_parts.push(text);
                }
            }
        }

        // Index combined text if available
        if !text_parts.is_empty() {
            let combined_text = text_parts.join(" ");
            self.bm25_index.index_document(node_id, &combined_text);
        }

        self.nodes.insert(node_id, node);
        self.update_stats();

        Ok(node_id)
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
            .filter_map(|&edge_id| self.edges.get(&edge_id).map(|edge| edge.target))
            .collect()
    }

    /// Get incoming neighbors of a node
    pub fn get_incoming_neighbors(&self, node_id: NodeId) -> Vec<NodeId> {
        self.adjacency_in
            .get(&node_id)
            .unwrap_or(&Vec::new())
            .iter()
            .filter_map(|&edge_id| self.edges.get(&edge_id).map(|edge| edge.source))
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
        self.agent_index
            .get(&agent_id)
            .and_then(|&node_id| self.nodes.get(&node_id))
    }

    /// Get node by event ID
    pub fn get_event_node(&self, event_id: EventId) -> Option<&GraphNode> {
        self.event_index
            .get(&event_id)
            .and_then(|&node_id| self.nodes.get(&node_id))
    }

    /// Get node by context hash
    pub fn get_context_node(&self, context_hash: ContextHash) -> Option<&GraphNode> {
        self.context_index
            .get(&context_hash)
            .and_then(|&node_id| self.nodes.get(&node_id))
    }

    /// Get node by goal ID
    pub fn get_goal_node(&self, goal_id: u64) -> Option<&GraphNode> {
        self.goal_index
            .get(&goal_id)
            .and_then(|&node_id| self.nodes.get(&node_id))
    }

    /// Get node by episode ID
    pub fn get_episode_node(&self, episode_id: u64) -> Option<&GraphNode> {
        self.episode_index
            .get(&episode_id)
            .and_then(|&node_id| self.nodes.get(&node_id))
    }

    /// Get node by memory ID
    pub fn get_memory_node(&self, memory_id: u64) -> Option<&GraphNode> {
        self.memory_index
            .get(&memory_id)
            .and_then(|&node_id| self.nodes.get(&node_id))
    }

    /// Get node by strategy ID
    pub fn get_strategy_node(&self, strategy_id: u64) -> Option<&GraphNode> {
        self.strategy_index
            .get(&strategy_id)
            .and_then(|&node_id| self.nodes.get(&node_id))
    }

    /// Get node by tool name
    pub fn get_tool_node(&self, tool_name: &str) -> Option<&GraphNode> {
        self.tool_index
            .get(tool_name)
            .and_then(|&node_id| self.nodes.get(&node_id))
    }

    /// Get node by result key
    pub fn get_result_node(&self, result_key: &str) -> Option<&GraphNode> {
        self.result_index
            .get(result_key)
            .and_then(|&node_id| self.nodes.get(&node_id))
    }

    /// Get mutable reference to node by ID
    pub fn get_node_mut(&mut self, node_id: NodeId) -> Option<&mut GraphNode> {
        self.nodes.get_mut(&node_id)
    }

    /// Get edge between two nodes (source -> target)
    pub fn get_edge_between(&self, source: NodeId, target: NodeId) -> Option<&GraphEdge> {
        // Use adjacency list to find the edge efficiently
        if let Some(edge_ids) = self.adjacency_out.get(&source) {
            for &edge_id in edge_ids {
                if let Some(edge) = self.edges.get(&edge_id) {
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
        let edge_id_opt = self.adjacency_out.get(&source).and_then(|edge_ids| {
            edge_ids.iter().find_map(|&edge_id| {
                self.edges
                    .get(&edge_id)
                    .filter(|edge| edge.target == target)
                    .map(|_| edge_id)
            })
        });

        if let Some(edge_id) = edge_id_opt {
            self.edges.get_mut(&edge_id)
        } else {
            None
        }
    }

    /// Get all edges from a source node
    pub fn get_edges_from(&self, source: NodeId) -> Vec<&GraphEdge> {
        self.adjacency_out
            .get(&source)
            .map(|edge_ids| {
                edge_ids
                    .iter()
                    .filter_map(|&edge_id| self.edges.get(&edge_id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get all edges to a target node
    pub fn get_edges_to(&self, target: NodeId) -> Vec<&GraphEdge> {
        self.adjacency_in
            .get(&target)
            .map(|edge_ids| {
                edge_ids
                    .iter()
                    .filter_map(|&edge_id| self.edges.get(&edge_id))
                    .collect()
            })
            .unwrap_or_default()
    }

    /// Get node by claim ID
    pub fn get_claim_node(&self, claim_id: u64) -> Option<&GraphNode> {
        self.claim_index
            .get(&claim_id)
            .and_then(|&node_id| self.nodes.get(&node_id))
    }

    /// Get node by concept name
    pub fn get_concept_node(&self, concept_name: &str) -> Option<&GraphNode> {
        self.concept_index
            .get(concept_name)
            .and_then(|&node_id| self.nodes.get(&node_id))
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
        self.nodes.keys().copied().collect()
    }

    // ========================================================================
    // remove_node — clean removal from all indices + edges + BM25
    // ========================================================================

    /// Remove a node and all its incident edges from the graph.
    ///
    /// Cleans up: nodes map, type_index, all 11 specialized indices,
    /// bm25_index, outgoing edges, incoming edges, adjacency lists,
    /// degree updates on neighbors, and stats.
    pub fn remove_node(&mut self, node_id: NodeId) -> Option<GraphNode> {
        let node = self.nodes.remove(&node_id)?;

        // Remove from type index
        let type_name = node.type_name();
        if let Some(set) = self.type_index.get_mut(&type_name) {
            set.remove(&node_id);
            if set.is_empty() {
                self.type_index.remove(&type_name);
            }
        }

        // Remove from specialized indices
        match &node.node_type {
            NodeType::Agent { agent_id, .. } => {
                self.agent_index.remove(agent_id);
            },
            NodeType::Event { event_id, .. } => {
                self.event_index.remove(event_id);
            },
            NodeType::Context { context_hash, .. } => {
                self.context_index.remove(context_hash);
            },
            NodeType::Goal { goal_id, .. } => {
                self.goal_index.remove(goal_id);
            },
            NodeType::Episode { episode_id, .. } => {
                self.episode_index.remove(episode_id);
            },
            NodeType::Memory { memory_id, .. } => {
                self.memory_index.remove(memory_id);
            },
            NodeType::Strategy { strategy_id, .. } => {
                self.strategy_index.remove(strategy_id);
            },
            NodeType::Tool { tool_name, .. } => {
                self.tool_index.remove(tool_name);
            },
            NodeType::Result { result_key, .. } => {
                self.result_index.remove(result_key);
            },
            NodeType::Claim { claim_id, .. } => {
                self.claim_index.remove(claim_id);
            },
            NodeType::Concept { concept_name, .. } => {
                self.concept_index.remove(concept_name);
            },
        }

        // Remove from BM25 index
        self.bm25_index.remove_document(node_id);

        // Collect edge IDs to remove (outgoing + incoming)
        let outgoing_edge_ids = self.adjacency_out.remove(&node_id).unwrap_or_default();
        let incoming_edge_ids = self.adjacency_in.remove(&node_id).unwrap_or_default();

        // Remove outgoing edges and update targets' incoming adjacency + degree
        for edge_id in &outgoing_edge_ids {
            if let Some(edge) = self.edges.remove(edge_id) {
                if edge.target != node_id {
                    if let Some(in_list) = self.adjacency_in.get_mut(&edge.target) {
                        in_list.retain(|&eid| eid != *edge_id);
                    }
                    if let Some(target) = self.nodes.get_mut(&edge.target) {
                        target.degree = target.degree.saturating_sub(1);
                    }
                }
            }
        }

        // Remove incoming edges and update sources' outgoing adjacency + degree
        for edge_id in &incoming_edge_ids {
            if let Some(edge) = self.edges.remove(edge_id) {
                if edge.source != node_id {
                    if let Some(out_list) = self.adjacency_out.get_mut(&edge.source) {
                        out_list.retain(|&eid| eid != *edge_id);
                    }
                    if let Some(source) = self.nodes.get_mut(&edge.source) {
                        source.degree = source.degree.saturating_sub(1);
                    }
                }
            }
        }

        self.update_stats();
        Some(node)
    }

    // ========================================================================
    // merge_nodes — merge absorbed into survivor
    // ========================================================================

    /// Merge two nodes of the same type variant. The survivor keeps its data
    /// with a signal boost; all edges from the absorbed node are redirected
    /// to the survivor (strengthening existing, skipping self-loops).
    /// Returns the updated survivor node.
    pub fn merge_nodes(
        &mut self,
        survivor_id: NodeId,
        absorbed_id: NodeId,
    ) -> Result<GraphNode, String> {
        // Verify both exist
        if !self.nodes.contains_key(&survivor_id) {
            return Err(format!("Survivor node {} not found", survivor_id));
        }
        if !self.nodes.contains_key(&absorbed_id) {
            return Err(format!("Absorbed node {} not found", absorbed_id));
        }

        // Same variant check
        {
            let survivor = &self.nodes[&survivor_id];
            let absorbed = &self.nodes[&absorbed_id];
            if std::mem::discriminant(&survivor.node_type)
                != std::mem::discriminant(&absorbed.node_type)
            {
                return Err(format!(
                    "Cannot merge different node types: {} vs {}",
                    survivor.type_name(),
                    absorbed.type_name()
                ));
            }
        }

        // Collect absorbed edges before mutation
        let absorbed_out = self
            .adjacency_out
            .get(&absorbed_id)
            .cloned()
            .unwrap_or_default();
        let absorbed_in = self
            .adjacency_in
            .get(&absorbed_id)
            .cloned()
            .unwrap_or_default();

        // Redirect outgoing edges: absorbed -> X becomes survivor -> X
        // First pass: gather info we need (targets, weights, existing edges)
        let mut out_redirects: Vec<(EdgeId, NodeId, f32)> = Vec::new(); // (edge_id, target, weight)
        for edge_id in &absorbed_out {
            if let Some(edge) = self.edges.get(edge_id) {
                let target = edge.target;
                if target == survivor_id || target == absorbed_id {
                    continue;
                }
                out_redirects.push((*edge_id, target, edge.weight));
            }
        }

        for (edge_id, target, weight) in out_redirects {
            // Check if survivor already has an edge to this target
            let existing_eid = self
                .adjacency_out
                .get(&survivor_id)
                .and_then(|ids| {
                    ids.iter()
                        .find(|&&eid| self.edges.get(&eid).is_some_and(|e| e.target == target))
                })
                .copied();

            if let Some(existing_eid) = existing_eid {
                // Strengthen existing edge
                if let Some(existing_edge) = self.edges.get_mut(&existing_eid) {
                    existing_edge.strengthen(weight * 0.5);
                }
            } else {
                // Redirect edge to survivor
                if let Some(edge) = self.edges.get_mut(&edge_id) {
                    edge.source = survivor_id;
                }
                self.adjacency_out
                    .entry(survivor_id)
                    .or_default()
                    .push(edge_id);
            }
        }

        // Redirect incoming edges: X -> absorbed becomes X -> survivor
        let mut in_redirects: Vec<(EdgeId, NodeId, f32)> = Vec::new();
        for edge_id in &absorbed_in {
            if let Some(edge) = self.edges.get(edge_id) {
                let source = edge.source;
                if source == survivor_id || source == absorbed_id {
                    continue;
                }
                in_redirects.push((*edge_id, source, edge.weight));
            }
        }

        for (edge_id, source, weight) in in_redirects {
            // Check if survivor already has an incoming edge from this source
            let existing_eid = self
                .adjacency_in
                .get(&survivor_id)
                .and_then(|ids| {
                    ids.iter()
                        .find(|&&eid| self.edges.get(&eid).is_some_and(|e| e.source == source))
                })
                .copied();

            if let Some(existing_eid) = existing_eid {
                // Strengthen existing edge
                if let Some(existing_edge) = self.edges.get_mut(&existing_eid) {
                    existing_edge.strengthen(weight * 0.5);
                }
            } else {
                // Redirect edge to survivor
                if let Some(edge) = self.edges.get_mut(&edge_id) {
                    edge.target = survivor_id;
                }
                self.adjacency_in
                    .entry(survivor_id)
                    .or_default()
                    .push(edge_id);
            }
        }

        // Boost survivor's signal via properties
        if let Some(survivor) = self.nodes.get_mut(&survivor_id) {
            let current_boost = survivor
                .properties
                .get("merge_boost")
                .and_then(|v| v.as_f64())
                .unwrap_or(0.0);
            survivor.properties.insert(
                "merge_boost".to_string(),
                serde_json::Value::from(current_boost + 0.1),
            );
            survivor.touch();
        }

        // Remove absorbed node (this cleans up all remaining references)
        self.remove_node(absorbed_id);

        // Update survivor degree
        if let Some(survivor) = self.nodes.get_mut(&survivor_id) {
            let out_count = self.adjacency_out.get(&survivor_id).map_or(0, |v| v.len());
            let in_count = self.adjacency_in.get(&survivor_id).map_or(0, |v| v.len());
            survivor.degree = (out_count + in_count) as u32;
        }

        Ok(self.nodes[&survivor_id].clone())
    }

    // ========================================================================
    // insert_existing_node / insert_existing_edge — preserves original IDs
    // ========================================================================

    /// Insert a node preserving its original ID. Updates all indices.
    /// Advances next_node_id if needed.
    pub fn insert_existing_node(&mut self, node: GraphNode) {
        let node_id = node.id;

        // Update type index
        let type_name = node.type_name();
        self.type_index
            .entry(type_name)
            .or_default()
            .insert(node_id);

        // Update specialized indices
        match &node.node_type {
            NodeType::Agent { agent_id, .. } => {
                self.agent_index.insert(*agent_id, node_id);
            },
            NodeType::Event { event_id, .. } => {
                self.event_index.insert(*event_id, node_id);
            },
            NodeType::Context { context_hash, .. } => {
                self.context_index.insert(*context_hash, node_id);
            },
            NodeType::Goal { goal_id, .. } => {
                self.goal_index.insert(*goal_id, node_id);
            },
            NodeType::Episode { episode_id, .. } => {
                self.episode_index.insert(*episode_id, node_id);
            },
            NodeType::Memory { memory_id, .. } => {
                self.memory_index.insert(*memory_id, node_id);
            },
            NodeType::Strategy { strategy_id, .. } => {
                self.strategy_index.insert(*strategy_id, node_id);
            },
            NodeType::Tool { tool_name, .. } => {
                self.tool_index.insert(tool_name.clone(), node_id);
            },
            NodeType::Result { result_key, .. } => {
                self.result_index.insert(result_key.clone(), node_id);
            },
            NodeType::Claim { claim_id, .. } => {
                self.claim_index.insert(*claim_id, node_id);
            },
            NodeType::Concept { concept_name, .. } => {
                self.concept_index.insert(concept_name.clone(), node_id);
            },
        }

        // BM25 indexing
        let mut text_parts = Vec::new();
        match &node.node_type {
            NodeType::Claim { claim_text, .. } => text_parts.push(claim_text.as_str()),
            NodeType::Goal { description, .. } => text_parts.push(description.as_str()),
            NodeType::Strategy { name, .. } => text_parts.push(name.as_str()),
            NodeType::Result { summary, .. } => text_parts.push(summary.as_str()),
            NodeType::Concept { concept_name, .. } => text_parts.push(concept_name.as_str()),
            NodeType::Tool { tool_name, .. } => text_parts.push(tool_name.as_str()),
            NodeType::Episode { outcome, .. } => text_parts.push(outcome.as_str()),
            _ => {},
        }
        for (key, value) in &node.properties {
            let key_lower = key.to_lowercase();
            if key_lower.contains("text")
                || key_lower.contains("description")
                || key_lower.contains("content")
                || key_lower.contains("name")
                || key_lower.contains("summary")
                || key_lower == "data"
            {
                if let Some(text) = value.as_str() {
                    text_parts.push(text);
                }
            }
        }
        if !text_parts.is_empty() {
            let combined = text_parts.join(" ");
            self.bm25_index.index_document(node_id, &combined);
        }

        // Initialize adjacency if needed
        self.adjacency_out.entry(node_id).or_default();
        self.adjacency_in.entry(node_id).or_default();

        // Advance next_node_id past this ID
        if node_id >= self.next_node_id {
            self.next_node_id = node_id + 1;
        }

        self.nodes.insert(node_id, node);
        self.update_stats();
    }

    /// Insert an edge preserving its original ID. Both endpoints must exist.
    /// Advances next_edge_id if needed.
    pub fn insert_existing_edge(&mut self, edge: GraphEdge) -> Option<EdgeId> {
        if !self.nodes.contains_key(&edge.source) || !self.nodes.contains_key(&edge.target) {
            return None;
        }

        let edge_id = edge.id;

        self.adjacency_out
            .entry(edge.source)
            .or_default()
            .push(edge_id);
        self.adjacency_in
            .entry(edge.target)
            .or_default()
            .push(edge_id);

        if let Some(source) = self.nodes.get_mut(&edge.source) {
            source.degree += 1;
        }
        if let Some(target) = self.nodes.get_mut(&edge.target) {
            target.degree += 1;
        }

        if edge_id >= self.next_edge_id {
            self.next_edge_id = edge_id + 1;
        }

        self.edges.insert(edge_id, edge);
        self.update_stats();

        Some(edge_id)
    }
}
