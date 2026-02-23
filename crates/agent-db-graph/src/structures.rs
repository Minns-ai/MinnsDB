//! Core graph data structures
//!
//! Implements the graph data structures used for modeling relationships
//! between agents, events, and contexts in the agentic database.

use crate::intern::Interner;
use agent_db_core::types::{AgentId, ContextHash, EventId, Timestamp};
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};
use smallvec::SmallVec;
use std::collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque};
use std::sync::Arc;

/// Unique identifier for graph nodes
pub type NodeId = u64;

/// Unique identifier for graph edges
pub type EdgeId = u64;

/// Weight type for edges (can represent similarity, causality strength, etc.)
pub type EdgeWeight = f32;

/// Traversal direction for directed graph queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Direction {
    /// Follow outgoing edges only.
    Out,
    /// Follow incoming edges only.
    In,
    /// Follow both outgoing and incoming edges.
    Both,
}

/// Depth specification for traversal queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Depth {
    /// Exactly N hops.
    Fixed(u32),
    /// Between min and max hops (inclusive).
    Range(u32, u32),
    /// No depth limit (bounded by budgets or graph size).
    Unbounded,
}

impl Depth {
    /// Maximum depth bound. `None` for `Unbounded`.
    pub fn max_depth(&self) -> Option<u32> {
        match self {
            Depth::Fixed(n) => Some(*n),
            Depth::Range(_, max) => Some(*max),
            Depth::Unbounded => None,
        }
    }

    /// Minimum depth bound. `0` for `Unbounded`.
    pub fn min_depth(&self) -> u32 {
        match self {
            Depth::Fixed(n) => *n,
            Depth::Range(min, _) => *min,
            Depth::Unbounded => 0,
        }
    }

    /// Validate depth specification. Returns error if Range has min > max.
    pub fn validate(&self) -> crate::GraphResult<()> {
        match self {
            Depth::Range(min, max) if min > max => Err(crate::GraphError::InvalidQuery(format!(
                "Depth range min ({}) > max ({})",
                min, max
            ))),
            _ => Ok(()),
        }
    }
}

// ============================================================================
// Adaptive Adjacency List
// ============================================================================

/// Threshold at which adjacency list is promoted from Small to Large.
const ADJ_LARGE_THRESHOLD: usize = 1024;

/// Adaptive adjacency list that selects the optimal representation
/// based on edge count.
///
/// - `Empty`: zero edges, no allocation
/// - `One(EdgeId)`: single edge, 8 bytes inline
/// - `Small(SmallVec<[EdgeId; 8]>)`: 2–1024 edges, inline up to 8
/// - `Large(BTreeSet<EdgeId>)`: 1025+ edges, O(log n) operations
#[derive(Debug, Clone, Default)]
pub enum AdjList {
    #[default]
    Empty,
    One(EdgeId),
    Small(SmallVec<[EdgeId; 8]>),
    Large(BTreeSet<EdgeId>),
}

impl AdjList {
    pub fn new() -> Self {
        AdjList::Empty
    }

    pub fn push(&mut self, edge_id: EdgeId) {
        *self = match std::mem::take(self) {
            AdjList::Empty => AdjList::One(edge_id),
            AdjList::One(existing) => {
                let mut sv = SmallVec::new();
                sv.push(existing);
                sv.push(edge_id);
                AdjList::Small(sv)
            },
            AdjList::Small(mut sv) => {
                sv.push(edge_id);
                if sv.len() > ADJ_LARGE_THRESHOLD {
                    AdjList::Large(sv.into_iter().collect())
                } else {
                    AdjList::Small(sv)
                }
            },
            AdjList::Large(mut set) => {
                set.insert(edge_id);
                AdjList::Large(set)
            },
        };
    }

    pub fn retain<F>(&mut self, mut f: F)
    where
        F: FnMut(&EdgeId) -> bool,
    {
        match self {
            AdjList::Empty => {},
            AdjList::One(eid) => {
                if !f(eid) {
                    *self = AdjList::Empty;
                }
            },
            AdjList::Small(sv) => {
                sv.retain(|eid| f(eid));
                match sv.len() {
                    0 => *self = AdjList::Empty,
                    1 => {
                        let eid = sv[0];
                        *self = AdjList::One(eid);
                    },
                    _ => {},
                }
            },
            AdjList::Large(set) => {
                set.retain(|eid| f(eid));
            },
        }
    }

    pub fn iter(&self) -> AdjListIter<'_> {
        match self {
            AdjList::Empty => AdjListIter::Empty,
            AdjList::One(eid) => AdjListIter::One(std::iter::once(eid)),
            AdjList::Small(sv) => AdjListIter::Small(sv.iter()),
            AdjList::Large(set) => AdjListIter::Large(set.iter()),
        }
    }

    pub fn len(&self) -> usize {
        match self {
            AdjList::Empty => 0,
            AdjList::One(_) => 1,
            AdjList::Small(sv) => sv.len(),
            AdjList::Large(set) => set.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        matches!(self, AdjList::Empty)
    }

    pub fn contains(&self, edge_id: &EdgeId) -> bool {
        match self {
            AdjList::Empty => false,
            AdjList::One(eid) => eid == edge_id,
            AdjList::Small(sv) => sv.contains(edge_id),
            AdjList::Large(set) => set.contains(edge_id),
        }
    }

    /// Build an AdjList from a Vec, picking the optimal variant.
    fn from_vec(ids: Vec<EdgeId>) -> Self {
        match ids.len() {
            0 => AdjList::Empty,
            1 => AdjList::One(ids[0]),
            n if n > ADJ_LARGE_THRESHOLD => AdjList::Large(ids.into_iter().collect()),
            _ => AdjList::Small(SmallVec::from_vec(ids)),
        }
    }
}

/// Custom Serialize: writes AdjList as a flat sequence of EdgeIds.
/// This is backward-compatible with SmallVec<[EdgeId; 8]> serialization.
impl Serialize for AdjList {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        use serde::ser::SerializeSeq;
        let len = self.len();
        let mut seq = serializer.serialize_seq(Some(len))?;
        for eid in self.iter() {
            seq.serialize_element(eid)?;
        }
        seq.end()
    }
}

/// Custom Deserialize: reads a sequence of EdgeIds and picks the optimal variant.
/// Backward-compatible with SmallVec<[EdgeId; 8]> deserialization.
impl<'de> Deserialize<'de> for AdjList {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let ids: Vec<EdgeId> = Vec::deserialize(deserializer)?;
        Ok(AdjList::from_vec(ids))
    }
}

/// Iterator over AdjList entries.
pub enum AdjListIter<'a> {
    Empty,
    One(std::iter::Once<&'a EdgeId>),
    Small(std::slice::Iter<'a, EdgeId>),
    Large(std::collections::btree_set::Iter<'a, EdgeId>),
}

impl<'a> Iterator for AdjListIter<'a> {
    type Item = &'a EdgeId;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            AdjListIter::Empty => None,
            AdjListIter::One(it) => it.next(),
            AdjListIter::Small(it) => it.next(),
            AdjListIter::Large(it) => it.next(),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self {
            AdjListIter::Empty => (0, Some(0)),
            AdjListIter::One(it) => it.size_hint(),
            AdjListIter::Small(it) => it.size_hint(),
            AdjListIter::Large(it) => it.size_hint(),
        }
    }
}

impl<'a> ExactSizeIterator for AdjListIter<'a> {}

impl<'a> IntoIterator for &'a AdjList {
    type Item = &'a EdgeId;
    type IntoIter = AdjListIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Unique identifier for goal buckets (semantic partitions)
pub type GoalBucketId = u64;

/// Maximum number of shards. Agent/event IDs are modded by this value
/// so the shard count stays bounded regardless of entity cardinality.
pub const NUM_SHARDS: u64 = 256;

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

/// Number of distinct `NodeType` variants. Keep in sync with `NodeType::discriminant()`.
pub const NODE_TYPE_COUNT: usize = 11;

/// Map a type name string (e.g. "Agent", "Event") to the corresponding discriminant.
/// Returns `None` for unknown names.
pub fn node_type_discriminant_from_name(name: &str) -> Option<u8> {
    match name {
        "Agent" => Some(0),
        "Event" => Some(1),
        "Context" => Some(2),
        "Concept" => Some(3),
        "Goal" => Some(4),
        "Episode" => Some(5),
        "Memory" => Some(6),
        "Strategy" => Some(7),
        "Tool" => Some(8),
        "Result" => Some(9),
        "Claim" => Some(10),
        _ => None,
    }
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
    /// All IDs are modded by [`NUM_SHARDS`] so the shard count stays bounded.
    pub fn goal_bucket(&self) -> GoalBucketId {
        match self {
            NodeType::Agent { agent_id, .. }
            | NodeType::Episode { agent_id, .. }
            | NodeType::Memory { agent_id, .. }
            | NodeType::Strategy { agent_id, .. } => *agent_id % NUM_SHARDS,
            NodeType::Event { event_id, .. } => (*event_id as u64) % NUM_SHARDS,
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
    pub(crate) nodes: FxHashMap<NodeId, GraphNode>,

    /// All edges in the graph
    pub(crate) edges: FxHashMap<EdgeId, GraphEdge>,

    /// Adjacency list for fast traversal (outgoing edges)
    pub(crate) adjacency_out: FxHashMap<NodeId, AdjList>,

    /// Reverse adjacency list (incoming edges)
    pub(crate) adjacency_in: FxHashMap<NodeId, AdjList>,

    /// Index by node type discriminant (u8) for O(1) type-filtered queries.
    /// Key is `NodeType::discriminant()` (0–10), not a heap-allocated string.
    pub(crate) type_index: FxHashMap<u8, HashSet<NodeId>>,

    /// Temporal index for efficient time-range queries.
    /// Maps `created_at` timestamp to the nodes created at that instant.
    pub(crate) temporal_index: BTreeMap<Timestamp, SmallVec<[NodeId; 4]>>,

    /// Monotonically increasing generation counter. Incremented on every
    /// structural mutation (add/remove node or edge, merge). Used by the
    /// query cache to detect staleness without manual invalidation.
    pub(crate) generation: u64,

    /// Spatial index for context nodes
    pub(crate) context_index: FxHashMap<ContextHash, NodeId>,

    /// Agent index for quick agent lookup
    pub(crate) agent_index: FxHashMap<AgentId, NodeId>,

    /// Event index for event-node mapping
    pub(crate) event_index: FxHashMap<EventId, NodeId>,

    /// Goal index for goal-node mapping
    pub(crate) goal_index: FxHashMap<u64, NodeId>,

    /// Episode index for episode-node mapping
    pub(crate) episode_index: FxHashMap<u64, NodeId>,

    /// Memory index for memory-node mapping
    pub(crate) memory_index: FxHashMap<u64, NodeId>,

    /// Strategy index for strategy-node mapping
    pub(crate) strategy_index: FxHashMap<u64, NodeId>,

    /// Tool index for tool-node mapping (interned keys)
    pub(crate) tool_index: HashMap<Arc<str>, NodeId>,

    /// Result index for result-node mapping (interned keys)
    pub(crate) result_index: HashMap<Arc<str>, NodeId>,

    /// Claim index for claim-node mapping
    pub(crate) claim_index: FxHashMap<u64, NodeId>,

    /// Concept index for concept-node mapping (interned keys)
    pub(crate) concept_index: HashMap<Arc<str>, NodeId>,

    /// BM25 full-text search index
    pub(crate) bm25_index: crate::indexing::Bm25Index,

    /// String interner for deduplicating repeated string values
    pub(crate) interner: Interner,

    /// Next available IDs
    pub(crate) next_node_id: NodeId,
    pub(crate) next_edge_id: EdgeId,

    /// Statistics
    pub(crate) stats: GraphStats,

    /// Maximum number of nodes allowed (enforced at add_node)
    pub(crate) max_graph_size: usize,

    // ── Delta tracking for incremental persistence ──
    /// Nodes added or modified since last persist
    pub(crate) dirty_nodes: HashSet<NodeId>,

    /// Edges added or modified since last persist
    pub(crate) dirty_edges: HashSet<EdgeId>,

    /// Nodes deleted since last persist (need disk cleanup)
    pub(crate) deleted_nodes: HashSet<NodeId>,

    /// Edges deleted since last persist (need disk cleanup)
    pub(crate) deleted_edges: HashSet<EdgeId>,

    /// Whether adjacency metadata blob needs re-persisting
    pub(crate) adjacency_dirty: bool,
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

    /// Get the type name of this node as a static string (zero allocation).
    pub fn type_name(&self) -> &'static str {
        match &self.node_type {
            NodeType::Agent { .. } => "Agent",
            NodeType::Event { .. } => "Event",
            NodeType::Context { .. } => "Context",
            NodeType::Concept { .. } => "Concept",
            NodeType::Goal { .. } => "Goal",
            NodeType::Episode { .. } => "Episode",
            NodeType::Memory { .. } => "Memory",
            NodeType::Strategy { .. } => "Strategy",
            NodeType::Tool { .. } => "Tool",
            NodeType::Result { .. } => "Result",
            NodeType::Claim { .. } => "Claim",
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
            nodes: FxHashMap::default(),
            edges: FxHashMap::default(),
            adjacency_out: FxHashMap::default(),
            adjacency_in: FxHashMap::default(),
            type_index: FxHashMap::default(),
            temporal_index: BTreeMap::new(),
            generation: 0,
            context_index: FxHashMap::default(),
            agent_index: FxHashMap::default(),
            event_index: FxHashMap::default(),
            goal_index: FxHashMap::default(),
            episode_index: FxHashMap::default(),
            memory_index: FxHashMap::default(),
            strategy_index: FxHashMap::default(),
            tool_index: HashMap::new(),
            result_index: HashMap::new(),
            claim_index: FxHashMap::default(),
            concept_index: HashMap::new(),
            bm25_index: crate::indexing::Bm25Index::new(),
            interner: Interner::new(),
            next_node_id: 1,
            next_edge_id: 1,
            stats: GraphStats::default(),
            max_graph_size,
            dirty_nodes: HashSet::new(),
            dirty_edges: HashSet::new(),
            deleted_nodes: HashSet::new(),
            deleted_edges: HashSet::new(),
            adjacency_dirty: false,
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

        // Update type index (u8 discriminant key)
        self.type_index
            .entry(node.node_type.discriminant())
            .or_default()
            .insert(node_id);

        // Update temporal index
        self.temporal_index
            .entry(node.created_at)
            .or_default()
            .push(node_id);

        // Bump generation for cache invalidation
        self.generation += 1;

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
                let key = self.interner.intern(tool_name);
                self.tool_index.insert(key, node_id);
            },
            NodeType::Result { result_key, .. } => {
                let key = self.interner.intern(result_key);
                self.result_index.insert(key, node_id);
            },
            NodeType::Claim { claim_id, .. } => {
                self.claim_index.insert(*claim_id, node_id);
            },
            NodeType::Concept { concept_name, .. } => {
                let key = self.interner.intern(concept_name);
                self.concept_index.insert(key, node_id);
            },
        }

        // Initialize adjacency lists
        self.adjacency_out.insert(node_id, AdjList::new());
        self.adjacency_in.insert(node_id, AdjList::new());

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
        self.dirty_nodes.insert(node_id);
        self.adjacency_dirty = true;
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

        let source = edge.source;
        let target = edge.target;
        self.edges.insert(edge_id, edge);
        self.dirty_edges.insert(edge_id);
        self.dirty_nodes.insert(source);
        self.dirty_nodes.insert(target);
        self.adjacency_dirty = true;

        // Bump generation for cache invalidation
        self.generation += 1;

        self.update_stats();

        Some(edge_id)
    }

    /// Get neighbors of a node (outgoing edges)
    pub fn get_neighbors(&self, node_id: NodeId) -> Vec<NodeId> {
        match self.adjacency_out.get(&node_id) {
            Some(edges) => edges
                .iter()
                .filter_map(|&edge_id| self.edges.get(&edge_id).map(|edge| edge.target))
                .collect(),
            None => Vec::new(),
        }
    }

    /// Get incoming neighbors of a node
    pub fn get_incoming_neighbors(&self, node_id: NodeId) -> Vec<NodeId> {
        match self.adjacency_in.get(&node_id) {
            Some(edges) => edges
                .iter()
                .filter_map(|&edge_id| self.edges.get(&edge_id).map(|edge| edge.source))
                .collect(),
            None => Vec::new(),
        }
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
                    .filter_map(|&node_id| self.nodes.get(&node_id))
                    .collect()
            })
            .unwrap_or_default()
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

    /// Current generation counter (monotonically increasing on every mutation).
    /// Used by cache layers to detect stale entries.
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Query nodes within a timestamp range `[start, end]` (inclusive).
    /// Uses the BTree temporal index for O(log N + K) lookups.
    pub fn nodes_in_time_range(&self, start: Timestamp, end: Timestamp) -> Vec<&GraphNode> {
        self.temporal_index
            .range(start..=end)
            .flat_map(|(_, ids)| ids.iter())
            .filter_map(|&nid| self.nodes.get(&nid))
            .collect()
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

    /// The latest timestamp in the temporal index (max created_at across all nodes).
    /// Returns `None` if the graph is empty.
    pub fn latest_timestamp(&self) -> Option<Timestamp> {
        self.temporal_index.keys().next_back().copied()
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

        // Remove from type index (u8 discriminant key)
        let disc = node.node_type.discriminant();
        if let Some(set) = self.type_index.get_mut(&disc) {
            set.remove(&node_id);
            if set.is_empty() {
                self.type_index.remove(&disc);
            }
        }

        // Remove from temporal index
        if let Some(ids) = self.temporal_index.get_mut(&node.created_at) {
            ids.retain(|nid| *nid != node_id);
            if ids.is_empty() {
                self.temporal_index.remove(&node.created_at);
            }
        }

        // Bump generation for cache invalidation
        self.generation += 1;

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
                self.tool_index.remove(tool_name.as_str());
            },
            NodeType::Result { result_key, .. } => {
                self.result_index.remove(result_key.as_str());
            },
            NodeType::Claim { claim_id, .. } => {
                self.claim_index.remove(claim_id);
            },
            NodeType::Concept { concept_name, .. } => {
                self.concept_index.remove(concept_name.as_str());
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
                self.deleted_edges.insert(*edge_id);
                self.dirty_edges.remove(edge_id);
                if edge.target != node_id {
                    if let Some(in_list) = self.adjacency_in.get_mut(&edge.target) {
                        in_list.retain(|eid| *eid != *edge_id);
                    }
                    if let Some(target) = self.nodes.get_mut(&edge.target) {
                        target.degree = target.degree.saturating_sub(1);
                        self.dirty_nodes.insert(edge.target);
                    }
                }
            }
        }

        // Remove incoming edges and update sources' outgoing adjacency + degree
        for edge_id in &incoming_edge_ids {
            if let Some(edge) = self.edges.remove(edge_id) {
                self.deleted_edges.insert(*edge_id);
                self.dirty_edges.remove(edge_id);
                if edge.source != node_id {
                    if let Some(out_list) = self.adjacency_out.get_mut(&edge.source) {
                        out_list.retain(|eid| *eid != *edge_id);
                    }
                    if let Some(source) = self.nodes.get_mut(&edge.source) {
                        source.degree = source.degree.saturating_sub(1);
                        self.dirty_nodes.insert(edge.source);
                    }
                }
            }
        }

        // Track deletion for delta persistence
        self.deleted_nodes.insert(node_id);
        self.dirty_nodes.remove(&node_id);
        self.adjacency_dirty = true;

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

        // Update type index (u8 discriminant key)
        self.type_index
            .entry(node.node_type.discriminant())
            .or_default()
            .insert(node_id);

        // Update temporal index
        self.temporal_index
            .entry(node.created_at)
            .or_default()
            .push(node_id);

        // Bump generation for cache invalidation
        self.generation += 1;

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
                let key = self.interner.intern(tool_name);
                self.tool_index.insert(key, node_id);
            },
            NodeType::Result { result_key, .. } => {
                let key = self.interner.intern(result_key);
                self.result_index.insert(key, node_id);
            },
            NodeType::Claim { claim_id, .. } => {
                self.claim_index.insert(*claim_id, node_id);
            },
            NodeType::Concept { concept_name, .. } => {
                let key = self.interner.intern(concept_name);
                self.concept_index.insert(key, node_id);
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
        self.dirty_nodes.insert(node_id);
        self.adjacency_dirty = true;
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

        let source = edge.source;
        let target = edge.target;
        self.edges.insert(edge_id, edge);
        self.dirty_edges.insert(edge_id);
        self.dirty_nodes.insert(source);
        self.dirty_nodes.insert(target);
        self.adjacency_dirty = true;
        self.update_stats();

        Some(edge_id)
    }

    // ========================================================================
    // Delta persistence helpers
    // ========================================================================

    /// Whether there are any pending changes to persist.
    pub fn has_pending_changes(&self) -> bool {
        !self.dirty_nodes.is_empty()
            || !self.dirty_edges.is_empty()
            || !self.deleted_nodes.is_empty()
            || !self.deleted_edges.is_empty()
            || self.adjacency_dirty
    }

    /// Clear all dirty tracking state after a successful persist.
    pub fn clear_dirty(&mut self) {
        self.dirty_nodes.clear();
        self.dirty_edges.clear();
        self.deleted_nodes.clear();
        self.deleted_edges.clear();
        self.adjacency_dirty = false;
    }
}
