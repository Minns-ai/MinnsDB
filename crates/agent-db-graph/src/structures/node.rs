// crates/agent-db-graph/src/structures/node.rs
//
// GraphNode struct, NodeType enum, ConceptType enum, GoalStatus enum, and related impls.

use agent_db_core::types::{AgentId, ContextHash, EventId, Timestamp};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::types::{GoalBucketId, NodeId, NUM_SHARDS, truncate_str};

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

    /// Partition key for multi-tenant isolation.
    /// All queries scope to this value when set. Empty string means global/unscoped.
    #[serde(default)]
    pub group_id: String,

    /// Node properties for additional metadata
    pub properties: HashMap<String, serde_json::Value>,

    /// Cached degree for performance
    pub degree: u32,

    /// Optional embedding vector for semantic similarity search.
    /// Empty if not yet embedded.
    #[serde(default)]
    pub embedding: Vec<f32>,
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
    // ── Code-derived concept types ──
    /// Function or method (code-derived)
    Function,
    /// Class, struct, or type (code-derived)
    Class,
    /// Module, package, or crate (code-derived)
    Module,
    /// Variable or constant (code-derived)
    Variable,
    /// Trait or interface (code-derived)
    Interface,
    /// Enum type (code-derived)
    Enum,
    /// Type alias (code-derived)
    TypeAlias,
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
            "FUNCTION" | "METHOD" | "FUNC" => ConceptType::Function,
            "CLASS" | "STRUCT" | "TYPE" => ConceptType::Class,
            "INTERFACE" | "TRAIT" => ConceptType::Interface,
            "ENUM" => ConceptType::Enum,
            "TYPE_ALIAS" | "TYPEDEF" => ConceptType::TypeAlias,
            "MODULE" | "PACKAGE" | "CRATE" | "NAMESPACE" => ConceptType::Module,
            "VARIABLE" | "CONST" | "VAR" | "PARAM" => ConceptType::Variable,
            _ => ConceptType::ContextualAssociation,
        }
    }
}

/// Goal status
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
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
            group_id: String::new(),
            properties: HashMap::new(),
            degree: 0,
            embedding: Vec::new(),
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

    /// Return a human-readable label for this node.
    pub fn label(&self) -> String {
        match &self.node_type {
            NodeType::Agent {
                agent_type,
                agent_id,
                ..
            } => {
                if agent_type.is_empty() {
                    format!("Agent {}", agent_id)
                } else {
                    agent_type.clone()
                }
            },
            NodeType::Event {
                event_id,
                event_type,
                ..
            } => {
                if event_type.is_empty() {
                    format!("Event {}", event_id)
                } else {
                    truncate_str(event_type, 40)
                }
            },
            NodeType::Context { context_hash, .. } => {
                format!("Context {}", context_hash)
            },
            NodeType::Concept { concept_name, .. } => concept_name.clone(),
            NodeType::Goal { description, .. } => truncate_str(description, 60),
            NodeType::Episode { episode_id, .. } => format!("Episode {}", episode_id),
            NodeType::Memory { memory_id, .. } => format!("Memory {}", memory_id),
            NodeType::Strategy { name, .. } => name.clone(),
            NodeType::Tool { tool_name, .. } => tool_name.clone(),
            NodeType::Result { result_key, .. } => result_key.clone(),
            NodeType::Claim { claim_text, .. } => truncate_str(claim_text, 60),
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
