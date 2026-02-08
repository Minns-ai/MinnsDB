//! Core event structures and types

use agent_db_core::types::*;
use serde::{Deserialize, Serialize};
use serde_with::{serde_as, DisplayFromStr, IfIsHumanReadable, PickFirst};
use std::collections::HashMap;
use std::time::Duration;

/// Complete event structure with all metadata
#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    /// Unique event identifier (auto-generated if not provided)
    #[serde(default = "generate_event_id")]
    #[serde_as(as = "DisplayFromStr")]
    pub id: EventId,

    /// High-precision timestamp (auto-generated if not provided)
    #[serde(default = "current_timestamp")]
    #[serde_as(as = "DisplayFromStr")]
    pub timestamp: Timestamp,

    /// Agent that generated this event
    #[serde_as(as = "IfIsHumanReadable<PickFirst<(_, DisplayFromStr)>>")]
    pub agent_id: AgentId,

    /// Agent type classification (e.g., "coding-assistant", "data-analyst")
    pub agent_type: AgentType,

    /// Session identifier for grouping
    #[serde_as(as = "IfIsHumanReadable<PickFirst<(_, DisplayFromStr)>>")]
    pub session_id: SessionId,

    /// Type and payload of the event
    pub event_type: EventType,

    /// Parent events in causality chain (optional - system may auto-populate)
    #[serde(default)]
    #[serde_as(as = "Vec<DisplayFromStr>")]
    pub causality_chain: Vec<EventId>,

    /// Environmental context (optional - uses minimal defaults if not provided)
    #[serde(default)]
    pub context: EventContext,

    /// Additional metadata (optional - system may auto-populate)
    #[serde(default)]
    pub metadata: HashMap<String, MetadataValue>,

    /// Size of context in bytes (for semantic memory promotion threshold)
    #[serde(default)]
    #[serde_as(as = "IfIsHumanReadable<PickFirst<(_, DisplayFromStr)>>")]
    pub context_size_bytes: usize,

    /// Pointer to segment storage for large contexts
    /// Format: "segment://{bucket}/{key}" or None for inline storage
    #[serde(default)]
    pub segment_pointer: Option<String>,
}

/// Different types of events the system can handle
#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventType {
    /// Agent actions and decisions
    Action {
        action_name: String,
        parameters: serde_json::Value,
        outcome: ActionOutcome,
        #[serde_as(as = "IfIsHumanReadable<PickFirst<(_, DisplayFromStr)>>")]
        duration_ns: u64,
    },

    /// Environmental observations
    Observation {
        observation_type: String,
        data: serde_json::Value,
        #[serde_as(as = "IfIsHumanReadable<PickFirst<(_, DisplayFromStr)>>")]
        confidence: f32,
        source: String,
    },

    /// Cognitive processes
    Cognitive {
        process_type: CognitiveType,
        input: serde_json::Value,
        output: serde_json::Value,
        reasoning_trace: Vec<String>,
    },

    /// Communication events
    Communication {
        message_type: String,
        #[serde_as(as = "IfIsHumanReadable<PickFirst<(_, DisplayFromStr)>>")]
        sender: AgentId,
        #[serde_as(as = "IfIsHumanReadable<PickFirst<(_, DisplayFromStr)>>")]
        recipient: AgentId,
        content: serde_json::Value,
    },

    /// Learning telemetry events (retrieved/used/outcome)
    Learning { event: LearningEvent },

    /// Explicit context event for semantic distillation
    Context {
        /// Raw text content (use segment_pointer in Event for large content)
        text: String,
        /// Type: "conversation", "document", "transcript", etc.
        context_type: String,
        /// Optional: language hint for NER ("en", "es", etc.)
        language: Option<String>,
    },
}

/// Explicit learning telemetry events (no inference)
#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LearningEvent {
    MemoryRetrieved {
        query_id: String,
        #[serde_as(as = "Vec<IfIsHumanReadable<PickFirst<(_, DisplayFromStr)>>>")]
        memory_ids: Vec<u64>,
    },
    MemoryUsed {
        query_id: String,
        #[serde_as(as = "IfIsHumanReadable<PickFirst<(_, DisplayFromStr)>>")]
        memory_id: u64,
    },
    StrategyServed {
        query_id: String,
        #[serde_as(as = "Vec<IfIsHumanReadable<PickFirst<(_, DisplayFromStr)>>>")]
        strategy_ids: Vec<u64>,
    },
    StrategyUsed {
        query_id: String,
        #[serde_as(as = "IfIsHumanReadable<PickFirst<(_, DisplayFromStr)>>")]
        strategy_id: u64,
    },
    Outcome {
        query_id: String,
        success: bool,
    },
}

/// Outcome of an action event
#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionOutcome {
    Success {
        result: serde_json::Value,
    },
    Failure {
        error: String,
        #[serde_as(as = "IfIsHumanReadable<PickFirst<(_, DisplayFromStr)>>")]
        error_code: u32,
    },
    Partial {
        result: serde_json::Value,
        issues: Vec<String>,
    },
}

/// Types of cognitive processes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CognitiveType {
    GoalFormation,
    Planning,
    Reasoning,
    MemoryRetrieval,
    LearningUpdate,
}

/// Environmental context at the time of event
#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EventContext {
    /// Environment state snapshot
    pub environment: EnvironmentState,

    /// Active goals
    pub active_goals: Vec<Goal>,

    /// Available resources
    pub resources: ResourceState,

    /// Context fingerprint for fast matching
    /// If not provided during deserialization, it will be auto-computed
    #[serde(default = "default_fingerprint")]
    #[serde_as(as = "IfIsHumanReadable<PickFirst<(_, DisplayFromStr)>>")]
    pub fingerprint: ContextHash,

    /// Context embeddings for similarity
    pub embeddings: Option<Vec<f32>>,
}

/// Default fingerprint (will be recomputed after deserialization)
fn default_fingerprint() -> ContextHash {
    0
}

/// Environment state variables
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct EnvironmentState {
    /// Key-value environment variables
    pub variables: HashMap<String, serde_json::Value>,

    /// Spatial context if applicable
    pub spatial: Option<SpatialContext>,

    /// Temporal context
    pub temporal: TemporalContext,
}

/// Goal information
#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Goal {
    #[serde_as(as = "IfIsHumanReadable<PickFirst<(_, DisplayFromStr)>>")]
    pub id: GoalId,
    pub description: String,
    #[serde_as(as = "IfIsHumanReadable<PickFirst<(_, DisplayFromStr)>>")]
    pub priority: f32,
    #[serde_as(as = "Option<IfIsHumanReadable<PickFirst<(_, DisplayFromStr)>>>")]
    pub deadline: Option<Timestamp>,
    #[serde_as(as = "IfIsHumanReadable<PickFirst<(_, DisplayFromStr)>>")]
    pub progress: f32,
    #[serde_as(as = "Vec<IfIsHumanReadable<PickFirst<(_, DisplayFromStr)>>>")]
    pub subgoals: Vec<GoalId>,
}

/// Resource availability state
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ResourceState {
    /// Available computational resources
    pub computational: ComputationalResources,

    /// Available external resources
    pub external: HashMap<String, ResourceAvailability>,
}

/// Spatial context information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpatialContext {
    pub location: (f64, f64, f64), // x, y, z coordinates
    pub bounds: Option<BoundingBox>,
    pub reference_frame: String,
}

/// Bounding box for spatial context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BoundingBox {
    pub min: (f64, f64, f64),
    pub max: (f64, f64, f64),
}

/// Temporal context information
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct TemporalContext {
    /// Time of day effects
    pub time_of_day: Option<TimeOfDay>,

    /// Active deadlines
    pub deadlines: Vec<Deadline>,

    /// Temporal patterns
    pub patterns: Vec<TemporalPattern>,
}

/// Computational resource availability
#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputationalResources {
    #[serde_as(as = "IfIsHumanReadable<PickFirst<(_, DisplayFromStr)>>")]
    pub cpu_percent: f32,
    #[serde_as(as = "IfIsHumanReadable<PickFirst<(_, DisplayFromStr)>>")]
    pub memory_bytes: u64,
    #[serde_as(as = "IfIsHumanReadable<PickFirst<(_, DisplayFromStr)>>")]
    pub storage_bytes: u64,
    #[serde_as(as = "IfIsHumanReadable<PickFirst<(_, DisplayFromStr)>>")]
    pub network_bandwidth: u64,
}

/// External resource availability
#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAvailability {
    pub available: bool,
    #[serde_as(as = "IfIsHumanReadable<PickFirst<(_, DisplayFromStr)>>")]
    pub capacity: f32,
    #[serde_as(as = "IfIsHumanReadable<PickFirst<(_, DisplayFromStr)>>")]
    pub current_usage: f32,
    #[serde_as(as = "Option<IfIsHumanReadable<PickFirst<(_, DisplayFromStr)>>>")]
    pub estimated_cost: Option<f32>,
}

/// Time of day information
#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeOfDay {
    #[serde_as(as = "IfIsHumanReadable<PickFirst<(_, DisplayFromStr)>>")]
    pub hour: u8,
    #[serde_as(as = "IfIsHumanReadable<PickFirst<(_, DisplayFromStr)>>")]
    pub minute: u8,
    pub timezone: String,
}

/// Deadline information
#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Deadline {
    #[serde_as(as = "IfIsHumanReadable<PickFirst<(_, DisplayFromStr)>>")]
    pub goal_id: GoalId,
    #[serde_as(as = "IfIsHumanReadable<PickFirst<(_, DisplayFromStr)>>")]
    pub timestamp: Timestamp,
    #[serde_as(as = "IfIsHumanReadable<PickFirst<(_, DisplayFromStr)>>")]
    pub priority: f32,
}

/// Temporal pattern information
#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalPattern {
    pub pattern_name: String,
    pub frequency: Duration,
    #[serde_as(as = "IfIsHumanReadable<PickFirst<(_, DisplayFromStr)>>")]
    pub phase: f32,
}

/// Extensible metadata value
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetadataValue {
    String(String),
    Integer(i64),
    Float(f64),
    Boolean(bool),
    Json(serde_json::Value),
}

impl Event {
    /// Create a new event with current timestamp
    pub fn new(
        agent_id: AgentId,
        agent_type: AgentType,
        session_id: SessionId,
        event_type: EventType,
        context: EventContext,
    ) -> Self {
        Self {
            id: generate_event_id(),
            timestamp: current_timestamp(),
            agent_id,
            agent_type,
            session_id,
            event_type,
            causality_chain: Vec::new(),
            context,
            metadata: HashMap::new(),
            context_size_bytes: 0,
            segment_pointer: None,
        }
    }

    /// Add parent event to causality chain
    pub fn with_parent(mut self, parent_id: EventId) -> Self {
        self.causality_chain.push(parent_id);
        self
    }

    /// Add metadata to event
    pub fn with_metadata(mut self, key: String, value: MetadataValue) -> Self {
        self.metadata.insert(key, value);
        self
    }

    /// Get event size in bytes (for storage calculations)
    pub fn size_bytes(&self) -> usize {
        // Rough estimate - would be more precise with actual serialization
        std::mem::size_of::<Self>()
            + self.causality_chain.len() * std::mem::size_of::<EventId>()
            + self.metadata.len() * 64 // rough estimate for metadata
    }

    /// Check if event references specific parent
    pub fn has_parent(&self, parent_id: EventId) -> bool {
        self.causality_chain.contains(&parent_id)
    }
}

impl EventContext {
    /// Create new context with computed fingerprint
    pub fn new(
        environment: EnvironmentState,
        active_goals: Vec<Goal>,
        resources: ResourceState,
    ) -> Self {
        let mut context = Self {
            environment,
            active_goals,
            resources,
            fingerprint: 0,
            embeddings: None,
        };
        context.fingerprint = context.compute_fingerprint();
        context
    }

    /// Compute deterministic context fingerprint for fast matching
    ///
    /// This fingerprint is **stable across processes and languages**.
    /// Based on semantic context only (goals, environment) - NOT runtime stats.
    ///
    /// # Algorithm (for cross-language implementations)
    ///
    /// 1. **Canonicalization**: Sort all maps by key, order all arrays deterministically
    /// 2. **Serialization**: Convert to bytes in stable order:
    ///    - Environment variables: sorted by key, JSON canonical form
    ///    - Goals: sorted by id, then serialize id + priority (f32 as bits)
    ///    - External resources: sorted by name, serialize name + availability flag
    /// 3. **Hashing**: BLAKE3 hash of concatenated bytes
    /// 4. **Output**: First 8 bytes as u64 (little-endian)
    ///
    /// # Example (pseudocode for other languages)
    /// ```text
    /// context_bytes = []
    ///
    /// // Environment variables (sorted)
    /// for key in sorted(environment.variables.keys()):
    ///     context_bytes += key.as_bytes()
    ///     context_bytes += canonical_json(environment.variables[key]).as_bytes()
    ///
    /// // Goals (sorted by id)
    /// for goal in sorted(active_goals, key=lambda g: g.id):
    ///     context_bytes += goal.id.to_bytes(8, 'little')
    ///     context_bytes += float_to_bits(goal.priority).to_bytes(4, 'little')
    ///
    /// // External resources (sorted by name)
    /// for name in sorted(resources.external.keys()):
    ///     context_bytes += name.as_bytes()
    ///     context_bytes += [1 if resources.external[name].available else 0]
    ///
    /// // Hash with BLAKE3
    /// hash = blake3(context_bytes)
    /// fingerprint = u64_from_le_bytes(hash[0..8])
    /// ```
    pub fn compute_fingerprint(&self) -> ContextHash {
        let mut canonical_bytes = Vec::new();

        // 1. Environment variables (sorted by key for determinism)
        let mut env_keys: Vec<&String> = self.environment.variables.keys().collect();
        env_keys.sort();
        for key in env_keys {
            canonical_bytes.extend_from_slice(key.as_bytes());
            // Use canonical JSON serialization for values
            if let Some(value) = self.environment.variables.get(key) {
                // Serialize value to canonical JSON (sorted keys, no whitespace)
                let json_str = serde_json::to_string(value).unwrap_or_default();
                canonical_bytes.extend_from_slice(json_str.as_bytes());
            }
        }

        // 2. Active goals (sorted by ID for determinism)
        let mut sorted_goals = self.active_goals.clone();
        sorted_goals.sort_by_key(|g| g.id);
        for goal in sorted_goals {
            canonical_bytes.extend_from_slice(&goal.id.to_le_bytes());
            // Use bits representation for stable f32 hashing
            canonical_bytes.extend_from_slice(&goal.priority.to_bits().to_le_bytes());
        }

        // 3. External resources (sorted by name for determinism)
        let mut resource_names: Vec<&String> = self.resources.external.keys().collect();
        resource_names.sort();
        for name in resource_names {
            canonical_bytes.extend_from_slice(name.as_bytes());
            if let Some(availability) = self.resources.external.get(name) {
                // Only include availability flag (primary semantic indicator)
                canonical_bytes.push(if availability.available { 1 } else { 0 });
            }
        }

        // NOTE: Computational resources (CPU/memory/storage/network) are NOT included
        // They are runtime stats, not semantic context for pattern matching

        // 4. Hash with BLAKE3 (deterministic, fast, cross-platform)
        let hash = blake3::hash(&canonical_bytes);

        // 5. Take first 8 bytes as u64 (little-endian)
        let bytes: [u8; 8] = hash.as_bytes()[0..8].try_into().unwrap();
        u64::from_le_bytes(bytes)
    }

    /// Calculate similarity to another context (0.0 to 1.0)
    pub fn similarity(&self, other: &EventContext) -> f32 {
        if let (Some(embed1), Some(embed2)) = (&self.embeddings, &other.embeddings) {
            // Cosine similarity if embeddings available
            cosine_similarity(embed1, embed2)
        } else {
            // Fallback to simple fingerprint comparison
            if self.fingerprint == other.fingerprint {
                1.0
            } else {
                0.0
            }
        }
    }

    /// Calculate comprehensive similarity using multi-component matching
    /// This is a general-purpose algorithm that works across domains
    pub fn comprehensive_similarity(
        &self,
        other: &EventContext,
        weights: Option<&ContextSimilarityWeights>,
    ) -> f32 {
        // Fast path: exact fingerprint match
        if self.fingerprint == other.fingerprint {
            return 1.0;
        }

        // Fast path: embeddings available (most accurate)
        if let (Some(emb1), Some(emb2)) = (&self.embeddings, &other.embeddings) {
            let emb_sim = cosine_similarity(emb1, emb2);
            // Use embeddings as primary, other components as refinement
            let component_sim = self.compute_component_similarity(other, weights);
            return (0.7 * emb_sim + 0.3 * component_sim).clamp(0.0, 1.0);
        }

        // Compute component-wise similarity
        self.compute_component_similarity(other, weights)
    }

    /// Compute similarity across all context components
    fn compute_component_similarity(
        &self,
        other: &EventContext,
        weights: Option<&ContextSimilarityWeights>,
    ) -> f32 {
        let default_weights = ContextSimilarityWeights::default();
        let weights = weights.unwrap_or(&default_weights);

        weights.environment * self.environment_similarity(&other.environment)
            + weights.goals * self.goals_similarity(&other.active_goals)
            + weights.resources * self.resources_similarity(&other.resources)
            + weights.temporal * self.temporal_similarity(&other.environment.temporal)
            + weights.spatial * self.spatial_similarity(&other.environment.spatial)
            + weights.embeddings * self.embeddings_similarity(&other.embeddings)
    }

    /// Environment variables similarity (Jaccard + value similarity)
    fn environment_similarity(&self, other: &EnvironmentState) -> f32 {
        let vars1 = &self.environment.variables;
        let vars2 = &other.variables;

        if vars1.is_empty() && vars2.is_empty() {
            return 1.0;
        }

        let all_keys: std::collections::HashSet<_> = vars1.keys().chain(vars2.keys()).collect();

        if all_keys.is_empty() {
            return 1.0;
        }

        let mut matching_score = 0.0;
        let total_keys = all_keys.len() as f32;

        for key in all_keys {
            match (vars1.get(key), vars2.get(key)) {
                (Some(v1), Some(v2)) => {
                    matching_score += self.value_similarity(v1, v2);
                },
                _ => {
                    // Missing key = different
                    matching_score += 0.0;
                },
            }
        }

        (matching_score / total_keys).clamp(0.0, 1.0)
    }

    /// Goals similarity (priority distribution + description similarity)
    fn goals_similarity(&self, other_goals: &[Goal]) -> f32 {
        let goals1 = &self.active_goals;
        let goals2 = other_goals;

        if goals1.is_empty() && goals2.is_empty() {
            return 1.0;
        }
        if goals1.is_empty() || goals2.is_empty() {
            return 0.0;
        }

        // Priority distribution similarity
        let priority_sim = self.priority_distribution_similarity(goals1, goals2);

        // Description similarity (if available)
        let desc_sim = self.goal_description_similarity(goals1, goals2);

        // Weighted combination
        0.6 * priority_sim + 0.4 * desc_sim
    }

    /// Compare priority distributions using histogram intersection
    fn priority_distribution_similarity(&self, goals1: &[Goal], goals2: &[Goal]) -> f32 {
        // Create priority buckets: [0.0-0.2), [0.2-0.4), [0.4-0.6), [0.6-0.8), [0.8-1.0]
        let mut hist1 = [0; 5];
        let mut hist2 = [0; 5];

        for goal in goals1 {
            let bucket = ((goal.priority * 5.0) as usize).min(4);
            hist1[bucket] += 1;
        }

        for goal in goals2 {
            let bucket = ((goal.priority * 5.0) as usize).min(4);
            hist2[bucket] += 1;
        }

        // Histogram intersection (normalized)
        let intersection: usize = hist1.iter().zip(hist2.iter()).map(|(a, b)| a.min(b)).sum();
        let union: usize = hist1.iter().zip(hist2.iter()).map(|(a, b)| a.max(b)).sum();

        if union == 0 {
            1.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Goal description similarity (simple word overlap)
    fn goal_description_similarity(&self, goals1: &[Goal], goals2: &[Goal]) -> f32 {
        if goals1.is_empty() || goals2.is_empty() {
            return 0.0;
        }

        // Simple word-based similarity
        let words1: std::collections::HashSet<_> = goals1
            .iter()
            .flat_map(|g| g.description.split_whitespace())
            .map(|w| w.to_lowercase())
            .collect();

        let words2: std::collections::HashSet<_> = goals2
            .iter()
            .flat_map(|g| g.description.split_whitespace())
            .map(|w| w.to_lowercase())
            .collect();

        if words1.is_empty() && words2.is_empty() {
            return 1.0;
        }

        let intersection = words1.intersection(&words2).count();
        let union = words1.union(&words2).count();

        if union == 0 {
            1.0
        } else {
            intersection as f32 / union as f32
        }
    }

    /// Resources similarity (normalized distance metrics)
    fn resources_similarity(&self, other: &ResourceState) -> f32 {
        let comp1 = &self.resources.computational;
        let comp2 = &other.computational;

        // CPU similarity (normalized distance)
        let cpu_sim = 1.0 - ((comp1.cpu_percent - comp2.cpu_percent).abs() / 100.0).min(1.0);

        // Memory similarity (normalized, assuming max 1TB)
        let max_memory = 1_000_000_000_000.0;
        let memory_diff = (comp1.memory_bytes as f64 - comp2.memory_bytes as f64).abs();
        let memory_sim = 1.0 - (memory_diff / max_memory).min(1.0) as f32;

        // External resources (Jaccard similarity of keys)
        let ext1: std::collections::HashSet<_> = self.resources.external.keys().collect();
        let ext2: std::collections::HashSet<_> = other.external.keys().collect();
        let ext_sim = if ext1.is_empty() && ext2.is_empty() {
            1.0
        } else {
            let intersection = ext1.intersection(&ext2).count();
            let union = ext1.union(&ext2).count();
            if union == 0 {
                1.0
            } else {
                intersection as f32 / union as f32
            }
        };

        // Weighted average
        0.4 * cpu_sim + 0.3 * memory_sim + 0.3 * ext_sim
    }

    /// Temporal similarity
    fn temporal_similarity(&self, other: &TemporalContext) -> f32 {
        let temp1 = &self.environment.temporal;

        // Deadline urgency similarity
        let deadline_sim = if temp1.deadlines.is_empty() && other.deadlines.is_empty() {
            1.0
        } else if temp1.deadlines.is_empty() || other.deadlines.is_empty() {
            0.5
        } else {
            // Simple: compare count and urgency
            1.0 - ((temp1.deadlines.len() as f32 - other.deadlines.len() as f32).abs() / 10.0)
                .min(1.0)
        };

        // Time-of-day similarity
        let time_sim = match (&temp1.time_of_day, &other.time_of_day) {
            (Some(t1), Some(t2)) => {
                let hour_diff = ((t1.hour as i32 - t2.hour as i32).abs())
                    .min(24 - (t1.hour as i32 - t2.hour as i32).abs());
                1.0 - (hour_diff as f32 / 12.0).min(1.0) // 12 hours apart = 0 similarity
            },
            (None, None) => 1.0,
            _ => 0.5, // One missing = neutral
        };

        // Pattern similarity (simple count comparison)
        let pattern_sim = if temp1.patterns.is_empty() && other.patterns.is_empty() {
            1.0
        } else {
            let count_diff = (temp1.patterns.len() as f32 - other.patterns.len() as f32).abs();
            1.0 - (count_diff / 10.0).min(1.0)
        };

        // Weighted combination
        0.4 * deadline_sim + 0.2 * time_sim + 0.4 * pattern_sim
    }

    /// Spatial similarity
    fn spatial_similarity(&self, other: &Option<SpatialContext>) -> f32 {
        match (&self.environment.spatial, other) {
            (None, None) => 1.0,                      // Both missing = same
            (Some(_), None) | (None, Some(_)) => 0.0, // One missing = different
            (Some(s1), Some(s2)) => {
                // Euclidean distance normalized
                let dist = ((s1.location.0 - s2.location.0).powi(2)
                    + (s1.location.1 - s2.location.1).powi(2)
                    + (s1.location.2 - s2.location.2).powi(2))
                .sqrt();

                // Estimate max distance from bounds if available
                let max_dist = 1000.0; // Default max distance
                (1.0 - (dist / max_dist).min(1.0)) as f32
            },
        }
    }

    /// Embeddings similarity
    fn embeddings_similarity(&self, other: &Option<Vec<f32>>) -> f32 {
        match (&self.embeddings, other) {
            (Some(e1), Some(e2)) => cosine_similarity(e1, e2),
            _ => 0.0, // Can't compute without embeddings
        }
    }

    /// Value similarity for JSON values (domain-agnostic)
    fn value_similarity(&self, v1: &serde_json::Value, v2: &serde_json::Value) -> f32 {
        match (v1, v2) {
            // Numbers: normalized distance
            (serde_json::Value::Number(n1), serde_json::Value::Number(n2)) => {
                if let (Some(f1), Some(f2)) = (n1.as_f64(), n2.as_f64()) {
                    let diff = (f1 - f2).abs();
                    let max_val = f1.abs().max(f2.abs()).max(1.0);
                    (1.0 - (diff / max_val).min(1.0)) as f32
                } else {
                    0.0
                }
            },
            // Strings: Jaro-Winkler similarity (simplified to exact match for now)
            (serde_json::Value::String(s1), serde_json::Value::String(s2)) => {
                if s1 == s2 {
                    1.0
                } else {
                    // Simple word overlap
                    let words1: std::collections::HashSet<_> = s1.split_whitespace().collect();
                    let words2: std::collections::HashSet<_> = s2.split_whitespace().collect();
                    let intersection = words1.intersection(&words2).count();
                    let union = words1.union(&words2).count();
                    if union == 0 {
                        1.0
                    } else {
                        intersection as f32 / union as f32
                    }
                }
            },
            // Booleans: exact match
            (serde_json::Value::Bool(b1), serde_json::Value::Bool(b2)) => {
                if b1 == b2 {
                    1.0
                } else {
                    0.0
                }
            },
            // Arrays: Jaccard similarity
            (serde_json::Value::Array(a1), serde_json::Value::Array(a2)) => {
                if a1.is_empty() && a2.is_empty() {
                    1.0
                } else {
                    // Simple: compare lengths and some elements
                    let len_sim = 1.0
                        - ((a1.len() as f32 - a2.len() as f32).abs()
                            / (a1.len().max(a2.len()) as f32).max(1.0));
                    len_sim * 0.5 // Simplified
                }
            },
            // Objects: recursive similarity
            (serde_json::Value::Object(o1), serde_json::Value::Object(o2)) => {
                if o1.is_empty() && o2.is_empty() {
                    1.0
                } else {
                    // Key overlap
                    let keys1: std::collections::HashSet<_> = o1.keys().collect();
                    let keys2: std::collections::HashSet<_> = o2.keys().collect();
                    let key_sim = if keys1.is_empty() && keys2.is_empty() {
                        1.0
                    } else {
                        let intersection = keys1.intersection(&keys2).count();
                        let union = keys1.union(&keys2).count();
                        if union == 0 {
                            1.0
                        } else {
                            intersection as f32 / union as f32
                        }
                    };
                    key_sim * 0.7 // Simplified
                }
            },
            // Different types = not similar
            _ => 0.0,
        }
    }
}

/// Weights for context similarity components
#[derive(Debug, Clone)]
pub struct ContextSimilarityWeights {
    pub environment: f32,
    pub goals: f32,
    pub resources: f32,
    pub temporal: f32,
    pub spatial: f32,
    pub embeddings: f32,
}

impl Default for ContextSimilarityWeights {
    fn default() -> Self {
        Self {
            environment: 0.25,
            goals: 0.30,
            resources: 0.15,
            temporal: 0.10,
            spatial: 0.05,
            embeddings: 0.15,
        }
    }
}

impl ContextSimilarityWeights {
    /// Normalize weights to sum to 1.0
    pub fn normalize(&mut self) {
        let sum = self.environment
            + self.goals
            + self.resources
            + self.temporal
            + self.spatial
            + self.embeddings;
        if sum > 0.0 {
            self.environment /= sum;
            self.goals /= sum;
            self.resources /= sum;
            self.temporal /= sum;
            self.spatial /= sum;
            self.embeddings /= sum;
        }
    }
}

// ============================================================================
// Default Implementations for Simple Integration
// ============================================================================

impl Default for ComputationalResources {
    fn default() -> Self {
        Self {
            cpu_percent: 0.0,     // Unknown/not tracked
            memory_bytes: 0,      // Unknown/not tracked
            storage_bytes: 0,     // Unknown/not tracked
            network_bandwidth: 0, // Unknown/not tracked
        }
    }
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        dot_product / (norm_a * norm_b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn test_event_creation() {
        let context = create_test_context();
        let event = Event::new(
            123,                      // agent_id
            "test_agent".to_string(), // agent_type
            456,                      // session_id
            EventType::Action {
                action_name: "test_action".to_string(),
                parameters: json!({"x": 10, "y": 20}),
                outcome: ActionOutcome::Success {
                    result: json!({"success": true}),
                },
                duration_ns: 1_000_000,
            },
            context,
        );

        assert_eq!(event.agent_id, 123);
        assert_eq!(event.session_id, 456);
        assert!(!event.id.to_string().is_empty());
        assert!(event.timestamp > 0);
    }

    #[test]
    fn test_causality_chain() {
        let parent_id = generate_event_id();
        let context = create_test_context();

        let event = Event::new(
            123,
            "test_agent".to_string(), // agent_type
            456,
            EventType::Cognitive {
                process_type: CognitiveType::Reasoning,
                input: json!({"problem": "navigation"}),
                output: json!({"solution": "path_found"}),
                reasoning_trace: vec!["step1".to_string()],
            },
            context,
        )
        .with_parent(parent_id);

        assert!(event.has_parent(parent_id));
        assert_eq!(event.causality_chain.len(), 1);
    }

    #[test]
    fn test_event_serialization() {
        let context = create_test_context();
        let event = Event::new(
            123,
            "test_agent".to_string(), // agent_type
            456,
            EventType::Observation {
                observation_type: "sensor_reading".to_string(),
                data: json!({"temperature": 23.5}),
                confidence: 0.95,
                source: "temp_sensor_1".to_string(),
            },
            context,
        );

        // Test JSON serialization
        let json_str = serde_json::to_string(&event).unwrap();
        let deserialized: Event = serde_json::from_str(&json_str).unwrap();
        assert_eq!(event.id, deserialized.id);

        // Test binary serialization
        // Note: bincode 1.x does not support DeserializeAny, which is used by serde_json::Value.
        // We skip binary serialization for events containing JSON values in this test,
        // or we would need to use a different binary format like postcard or messagepack.
        /*
        let binary = bincode::serialize(&event).expect("Bincode serialization failed");
        let deserialized_binary: Event = bincode::deserialize(&binary).expect("Bincode deserialization failed");
        assert_eq!(event.id, deserialized_binary.id);
        */
    }

    #[test]
    fn test_context_fingerprinting() {
        let context1 = create_test_context();
        let context2 = create_test_context();
        let context3 = create_different_context();

        println!("Context 1 fingerprint: {}", context1.fingerprint);
        println!("Context 2 fingerprint: {}", context2.fingerprint);
        println!("Context 3 fingerprint: {}", context3.fingerprint);

        // Same contexts should have same fingerprint
        assert_eq!(context1.fingerprint, context2.fingerprint);

        // Different contexts should have different fingerprints (most likely)
        assert_ne!(context1.fingerprint, context3.fingerprint);
    }

    fn create_test_context() -> EventContext {
        let mut context = EventContext::new(
            EnvironmentState {
                variables: {
                    let mut vars = HashMap::new();
                    vars.insert("temperature".to_string(), json!(23.5));
                    vars.insert("location".to_string(), json!("test_room"));
                    vars
                },
                spatial: Some(SpatialContext {
                    location: (0.0, 0.0, 0.0),
                    bounds: None,
                    reference_frame: "world".to_string(),
                }),
                temporal: TemporalContext {
                    time_of_day: Some(TimeOfDay {
                        hour: 14,
                        minute: 30,
                        timezone: "UTC".to_string(),
                    }),
                    deadlines: Vec::new(),
                    patterns: Vec::new(),
                },
            },
            vec![Goal {
                id: 1,
                description: "Test goal".to_string(),
                priority: 0.5,
                deadline: None,
                progress: 0.0,
                subgoals: Vec::new(),
            }],
            ResourceState {
                computational: ComputationalResources {
                    cpu_percent: 50.0,
                    memory_bytes: 1024 * 1024,
                    storage_bytes: 1024 * 1024 * 1024,
                    network_bandwidth: 1000,
                },
                external: HashMap::new(),
            },
        );
        context.fingerprint = context.compute_fingerprint();
        context
    }

    fn create_different_context() -> EventContext {
        let mut context = EventContext::new(
            EnvironmentState {
                variables: {
                    let mut vars = HashMap::new();
                    vars.insert("temperature".to_string(), json!(18.0));
                    vars.insert("location".to_string(), json!("different_room"));
                    vars
                },
                spatial: Some(SpatialContext {
                    location: (10.0, 5.0, 0.0),
                    bounds: None,
                    reference_frame: "world".to_string(),
                }),
                temporal: TemporalContext {
                    time_of_day: Some(TimeOfDay {
                        hour: 9,
                        minute: 15,
                        timezone: "UTC".to_string(),
                    }),
                    deadlines: Vec::new(),
                    patterns: Vec::new(),
                },
            },
            Vec::new(),
            ResourceState {
                computational: ComputationalResources {
                    cpu_percent: 85.0,
                    memory_bytes: 512 * 1024,
                    storage_bytes: 500 * 1024 * 1024,
                    network_bandwidth: 100,
                },
                external: HashMap::new(),
            },
        );
        context.fingerprint = context.compute_fingerprint();
        context
    }
}
