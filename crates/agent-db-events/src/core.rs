//! Core event structures and types

use agent_db_core::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::Duration;

/// Complete event structure with all metadata
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    /// Unique event identifier
    pub id: EventId,
    
    /// High-precision timestamp
    pub timestamp: Timestamp,
    
    /// Agent that generated this event
    pub agent_id: AgentId,
    
    /// Agent type classification (e.g., "coding-assistant", "data-analyst")
    pub agent_type: AgentType,
    
    /// Session identifier for grouping
    pub session_id: SessionId,
    
    /// Type and payload of the event
    pub event_type: EventType,
    
    /// Parent events in causality chain
    pub causality_chain: Vec<EventId>,
    
    /// Environmental context
    pub context: EventContext,
    
    /// Additional metadata
    pub metadata: HashMap<String, MetadataValue>,
}

/// Different types of events the system can handle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EventType {
    /// Agent actions and decisions
    Action {
        action_name: String,
        parameters: serde_json::Value,
        outcome: ActionOutcome,
        duration_ns: u64,
    },
    
    /// Environmental observations
    Observation {
        observation_type: String,
        data: serde_json::Value,
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
        sender: AgentId,
        recipient: AgentId,
        content: serde_json::Value,
    },
}

/// Outcome of an action event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ActionOutcome {
    Success { result: serde_json::Value },
    Failure { error: String, error_code: u32 },
    Partial { result: serde_json::Value, issues: Vec<String> },
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventContext {
    /// Environment state snapshot
    pub environment: EnvironmentState,
    
    /// Active goals
    pub active_goals: Vec<Goal>,
    
    /// Available resources
    pub resources: ResourceState,
    
    /// Context fingerprint for fast matching
    pub fingerprint: ContextHash,
    
    /// Context embeddings for similarity
    pub embeddings: Option<Vec<f32>>,
}

/// Environment state variables
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnvironmentState {
    /// Key-value environment variables
    pub variables: HashMap<String, serde_json::Value>,
    
    /// Spatial context if applicable
    pub spatial: Option<SpatialContext>,
    
    /// Temporal context
    pub temporal: TemporalContext,
}

/// Goal information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Goal {
    pub id: GoalId,
    pub description: String,
    pub priority: f32,
    pub deadline: Option<Timestamp>,
    pub progress: f32,
    pub subgoals: Vec<GoalId>,
}

/// Resource availability state
#[derive(Debug, Clone, Serialize, Deserialize)]
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
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalContext {
    /// Time of day effects
    pub time_of_day: Option<TimeOfDay>,
    
    /// Active deadlines
    pub deadlines: Vec<Deadline>,
    
    /// Temporal patterns
    pub patterns: Vec<TemporalPattern>,
}

/// Computational resource availability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComputationalResources {
    pub cpu_percent: f32,
    pub memory_bytes: u64,
    pub storage_bytes: u64,
    pub network_bandwidth: u64,
}

/// External resource availability
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceAvailability {
    pub available: bool,
    pub capacity: f32,
    pub current_usage: f32,
    pub estimated_cost: Option<f32>,
}

/// Time of day information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeOfDay {
    pub hour: u8,
    pub minute: u8,
    pub timezone: String,
}

/// Deadline information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Deadline {
    pub goal_id: GoalId,
    pub timestamp: Timestamp,
    pub priority: f32,
}

/// Temporal pattern information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalPattern {
    pub pattern_name: String,
    pub frequency: Duration,
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
    pub fn new(environment: EnvironmentState, active_goals: Vec<Goal>, resources: ResourceState) -> Self {
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
    
    /// Compute context fingerprint for fast matching
    pub fn compute_fingerprint(&self) -> ContextHash {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};
        
        let mut hasher = DefaultHasher::new();
        
        // Hash environment variables (simplified)
        for (key, _value) in &self.environment.variables {
            key.hash(&mut hasher);
            // Note: Would need to implement Hash for serde_json::Value or convert to string
        }
        
        // Hash active goals
        for goal in &self.active_goals {
            goal.id.hash(&mut hasher);
            goal.priority.to_bits().hash(&mut hasher);
        }
        
        // Hash resource state
        self.resources.computational.cpu_percent.to_bits().hash(&mut hasher);
        self.resources.computational.memory_bytes.hash(&mut hasher);
        
        hasher.finish()
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
            123, // agent_id
            456, // session_id
            EventType::Action {
                action_name: "test_action".to_string(),
                parameters: json!({"x": 10, "y": 20}),
                outcome: ActionOutcome::Success { 
                    result: json!({"success": true}) 
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
            456,
            EventType::Cognitive {
                process_type: CognitiveType::Reasoning,
                input: json!({"problem": "navigation"}),
                output: json!({"solution": "path_found"}),
                reasoning_trace: vec!["step1".to_string()],
            },
            context,
        ).with_parent(parent_id);
        
        assert!(event.has_parent(parent_id));
        assert_eq!(event.causality_chain.len(), 1);
    }
    
    #[test]
    fn test_event_serialization() {
        let context = create_test_context();
        let event = Event::new(
            123,
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
        let binary = bincode::serialize(&event).unwrap();
        let deserialized_binary: Event = bincode::deserialize(&binary).unwrap();
        assert_eq!(event.id, deserialized_binary.id);
    }
    
    #[test]
    fn test_context_fingerprinting() {
        let context1 = create_test_context();
        let context2 = create_test_context();
        let context3 = create_different_context();
        
        // Same contexts should have same fingerprint
        assert_eq!(context1.fingerprint, context2.fingerprint);
        
        // Different contexts should have different fingerprints (most likely)
        assert_ne!(context1.fingerprint, context3.fingerprint);
    }
    
    fn create_test_context() -> EventContext {
        EventContext::new(
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
        )
    }
    
    fn create_different_context() -> EventContext {
        EventContext::new(
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
        )
    }
}