// crates/agent-db-graph/src/structures/edge.rs
//
// GraphEdge struct, EdgeType enum, InteractionType, GoalRelationType, and related impls.

use agent_db_core::types::Timestamp;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::types::{EdgeId, EdgeWeight, NodeId};

/// Graph edge representing relationships between nodes.
///
/// Supports a **bi-temporal** model:
///
/// | Dimension       | Fields                    | Meaning |
/// |-----------------|---------------------------|---------|
/// | Transaction time| `created_at`, `updated_at`| When the edge entered/was modified in the database |
/// | Valid time      | `valid_from`, `valid_until`| When the fact was true in the real world |
///
/// Example: "Alice worked at Google from 2020-2023" → `valid_from=2020`, `valid_until=2023`,
/// but `created_at` is whenever it was ingested into the graph.
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

    // ── Transaction time (when the edge was recorded) ────────────────
    /// Creation timestamp (transaction time — when this edge entered the database)
    pub created_at: Timestamp,

    /// Last update timestamp (transaction time — last modification in the database)
    pub updated_at: Timestamp,

    // ── Valid time (when the fact was true in the real world) ─────────
    /// Start of real-world validity period (nanos since epoch).
    /// `None` means "valid since the beginning of time" (open start).
    #[serde(default)]
    pub valid_from: Option<Timestamp>,

    /// End of real-world validity period (nanos since epoch).
    /// `None` means "still valid now" (open end / currently true).
    #[serde(default)]
    pub valid_until: Option<Timestamp>,

    /// Number of times this relationship has been observed
    pub observation_count: u32,

    /// Confidence in this relationship (0.0 to 1.0)
    pub confidence: f32,

    /// Partition key for multi-tenant isolation.
    /// Matches the source node's `group_id`. Empty string means global/unscoped.
    #[serde(default)]
    pub group_id: String,

    /// Additional edge properties
    pub properties: HashMap<String, serde_json::Value>,

    /// Temporal history of confidence values (optional, populated by temporal API).
    /// When empty, `self.confidence` is the canonical value.
    #[serde(default, skip_serializing_if = "crate::tcell::TCell::is_empty")]
    pub confidence_history: crate::tcell::TCell<f32>,

    /// Temporal history of weight values (optional).
    /// When empty, `self.weight` is the canonical value.
    #[serde(default, skip_serializing_if = "crate::tcell::TCell::is_empty")]
    pub weight_history: crate::tcell::TCell<f32>,
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

    /// Code structural relationship (from AST analysis)
    CodeStructure {
        /// Relationship kind: "contains", "imports", "field_of", "returns", "calls", etc.
        relation_kind: String,
        /// Source file path
        file_path: String,
        /// Extraction confidence (1.0 for safe structural, <1.0 for heuristic)
        confidence: f32,
    },

    /// Semantic relationship (claim is about an entity/concept)
    About {
        relevance_score: f32,
        mention_count: u32,
        /// Role of the entity in the claim's SPO triple (Subject/Object/Mentioned)
        #[serde(default)]
        entity_role: crate::claims::types::EntityRole,
        /// Predicate linking subject to object (only set for Subject/Object roles)
        #[serde(default)]
        predicate: Option<String>,
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
            valid_from: None,
            valid_until: None,
            observation_count: 1,
            confidence: 0.5, // Start with medium confidence
            group_id: String::new(),
            properties: HashMap::new(),
            confidence_history: crate::tcell::TCell::Empty,
            weight_history: crate::tcell::TCell::Empty,
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

    /// Get confidence at a specific time, falling back to the snapshot value.
    pub fn confidence_at(&self, time: agent_db_core::event_time::EventTime) -> f32 {
        self.confidence_history
            .last_at_or_before(time)
            .map(|(_, v)| *v)
            .unwrap_or(self.confidence)
    }

    /// Update confidence with temporal tracking.
    pub fn set_confidence_temporal(
        &mut self,
        time: agent_db_core::event_time::EventTime,
        value: f32,
    ) {
        self.confidence_history.set(time, value);
        self.confidence = value; // keep snapshot in sync
    }

    /// Get weight at a specific time, falling back to the snapshot value.
    pub fn weight_at(&self, time: agent_db_core::event_time::EventTime) -> f32 {
        self.weight_history
            .last_at_or_before(time)
            .map(|(_, v)| *v)
            .unwrap_or(self.weight)
    }

    /// Update weight with temporal tracking.
    pub fn set_weight_temporal(&mut self, time: agent_db_core::event_time::EventTime, value: f32) {
        self.weight_history.set(time, value);
        self.weight = value; // keep snapshot in sync
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

    /// Check if this edge is currently valid (not soft-deleted).
    ///
    /// Edges without an `is_valid` property are treated as valid (backward compatible).
    pub fn is_valid(&self) -> bool {
        self.properties
            .get("is_valid")
            .and_then(|v| v.as_bool())
            .unwrap_or(true)
    }

    /// Soft-delete this edge by marking it invalid.
    pub fn invalidate(&mut self, reason: &str) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos() as u64;
        self.properties
            .insert("is_valid".to_string(), serde_json::json!(false));
        self.properties
            .insert("invalidated_at".to_string(), serde_json::json!(now));
        self.properties
            .insert("invalidated_reason".to_string(), serde_json::json!(reason));
    }

    /// Get the invalidation timestamp, if this edge has been soft-deleted.
    pub fn invalidated_at(&self) -> Option<u64> {
        self.properties
            .get("invalidated_at")
            .and_then(|v| v.as_u64())
    }

    // ── Bi-temporal helpers ──────────────────────────────────────────────

    /// Check if this edge's fact was valid at a given real-world timestamp.
    ///
    /// Uses half-open interval `[valid_from, valid_until)`:
    /// - `valid_from == None` → valid since beginning of time
    /// - `valid_until == None` → still valid now (open-ended)
    pub fn valid_at(&self, point_in_time: Timestamp) -> bool {
        if let Some(from) = self.valid_from {
            if point_in_time < from {
                return false;
            }
        }
        if let Some(until) = self.valid_until {
            if point_in_time >= until {
                return false;
            }
        }
        true
    }

    /// Check if this edge's valid-time range overlaps with a query range.
    ///
    /// Useful for "what was true during 2022?" queries.
    pub fn valid_during(&self, range_start: Timestamp, range_end: Timestamp) -> bool {
        // Edge start must be before range end
        if let Some(from) = self.valid_from {
            if from >= range_end {
                return false;
            }
        }
        // Edge end must be after range start
        if let Some(until) = self.valid_until {
            if until <= range_start {
                return false;
            }
        }
        true
    }

    /// Set the valid-time window for this edge.
    pub fn set_valid_time(&mut self, from: Option<Timestamp>, until: Option<Timestamp>) {
        self.valid_from = from;
        self.valid_until = until;
    }

    /// Check if this edge represents a currently-valid fact
    /// (valid_until is None or in the future).
    pub fn is_currently_valid_fact(&self, now: Timestamp) -> bool {
        self.is_valid() && self.valid_at(now)
    }
}
