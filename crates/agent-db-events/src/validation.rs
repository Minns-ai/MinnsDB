//! Event validation logic

use crate::core::{Event, EventContext, EventType};
use agent_db_core::error::{DatabaseError, DatabaseResult};
use agent_db_core::types::EventId;

/// Basic event validator implementation
pub struct BasicEventValidator {
    /// Maximum allowed event size in bytes
    max_event_size: usize,

    /// Maximum causality chain length
    max_causality_depth: usize,
}

impl BasicEventValidator {
    /// Create a new validator with default settings
    pub fn new() -> Self {
        Self {
            max_event_size: 1024 * 1024, // 1MB
            max_causality_depth: 100,    // Prevent very deep chains
        }
    }

    /// Create validator with custom settings
    pub fn with_limits(max_event_size: usize, max_causality_depth: usize) -> Self {
        Self {
            max_event_size,
            max_causality_depth,
        }
    }

    /// Validate event size constraints
    fn validate_size(&self, event: &Event) -> DatabaseResult<()> {
        let size = event.size_bytes();
        if size > self.max_event_size {
            return Err(DatabaseError::validation(format!(
                "Event size {} exceeds maximum {}",
                size, self.max_event_size
            )));
        }
        Ok(())
    }

    /// Validate event content and structure
    fn validate_content(&self, event: &Event) -> DatabaseResult<()> {
        // Validate agent ID
        if event.agent_id == 0 {
            return Err(DatabaseError::validation("Agent ID cannot be zero"));
        }

        // Validate timestamp
        if event.timestamp == 0 {
            return Err(DatabaseError::validation("Timestamp cannot be zero"));
        }

        // Validate event type specific content
        match &event.event_type {
            EventType::Action {
                action_name,
                duration_ns,
                ..
            } => {
                if action_name.is_empty() {
                    return Err(DatabaseError::validation("Action name cannot be empty"));
                }
                if *duration_ns == 0 {
                    return Err(DatabaseError::validation(
                        "Action duration must be positive",
                    ));
                }
            },
            EventType::Observation {
                observation_type,
                confidence,
                ..
            } => {
                if observation_type.is_empty() {
                    return Err(DatabaseError::validation(
                        "Observation type cannot be empty",
                    ));
                }
                if *confidence < 0.0 || *confidence > 1.0 {
                    return Err(DatabaseError::validation(
                        "Confidence must be between 0.0 and 1.0",
                    ));
                }
            },
            EventType::Cognitive {
                reasoning_trace, ..
            } => {
                if reasoning_trace.is_empty() {
                    return Err(DatabaseError::validation(
                        "Cognitive events must have reasoning trace",
                    ));
                }
            },
            EventType::Communication {
                sender, recipient, ..
            } => {
                if sender == recipient {
                    return Err(DatabaseError::validation(
                        "Sender and recipient cannot be the same",
                    ));
                }
            },
            EventType::Learning { .. } => {
                // Learning telemetry is validated by schema; no extra constraints.
            },
            EventType::Context {
                text, context_type, ..
            } => {
                if text.is_empty() {
                    return Err(DatabaseError::validation("Context text cannot be empty"));
                }
                if context_type.is_empty() {
                    return Err(DatabaseError::validation("Context type cannot be empty"));
                }
            },
            EventType::Conversation {
                speaker, content, ..
            } => {
                if speaker.is_empty() {
                    return Err(DatabaseError::validation(
                        "Conversation speaker cannot be empty",
                    ));
                }
                if content.is_empty() {
                    return Err(DatabaseError::validation(
                        "Conversation content cannot be empty",
                    ));
                }
            },
            EventType::CodeReview {
                review_id,
                body,
                repository,
                ..
            } => {
                if review_id.is_empty() {
                    return Err(DatabaseError::validation(
                        "CodeReview review_id cannot be empty",
                    ));
                }
                if body.is_empty() {
                    return Err(DatabaseError::validation("CodeReview body cannot be empty"));
                }
                if repository.is_empty() {
                    return Err(DatabaseError::validation(
                        "CodeReview repository cannot be empty",
                    ));
                }
            },
            EventType::CodeFile {
                file_path, content, ..
            } => {
                if file_path.is_empty() {
                    return Err(DatabaseError::validation(
                        "CodeFile file_path cannot be empty",
                    ));
                }
                if content.is_empty() {
                    return Err(DatabaseError::validation(
                        "CodeFile content cannot be empty",
                    ));
                }
            },
        }

        Ok(())
    }

    /// Validate goal structure
    fn validate_goals(&self, context: &EventContext) -> DatabaseResult<()> {
        for goal in &context.active_goals {
            if goal.description.is_empty() {
                return Err(DatabaseError::validation(
                    "Goal description cannot be empty",
                ));
            }

            if goal.priority < 0.0 || goal.priority > 1.0 {
                return Err(DatabaseError::validation(
                    "Goal priority must be between 0.0 and 1.0",
                ));
            }

            if goal.progress < 0.0 || goal.progress > 1.0 {
                return Err(DatabaseError::validation(
                    "Goal progress must be between 0.0 and 1.0",
                ));
            }
        }
        Ok(())
    }

    /// Validate resource constraints
    fn validate_resources(&self, context: &EventContext) -> DatabaseResult<()> {
        let comp = &context.resources.computational;

        if comp.cpu_percent < 0.0 || comp.cpu_percent > 100.0 {
            return Err(DatabaseError::validation(
                "CPU percentage must be between 0.0 and 100.0",
            ));
        }

        for (name, resource) in &context.resources.external {
            if resource.capacity < 0.0 {
                return Err(DatabaseError::validation(format!(
                    "Resource '{}' capacity cannot be negative",
                    name
                )));
            }

            if resource.current_usage < 0.0 || resource.current_usage > resource.capacity {
                return Err(DatabaseError::validation(format!(
                    "Resource '{}' usage {} exceeds capacity {}",
                    name, resource.current_usage, resource.capacity
                )));
            }
        }

        Ok(())
    }
}

impl Default for BasicEventValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl BasicEventValidator {
    /// Validate event structure and content
    pub fn validate_event(&self, event: &Event) -> DatabaseResult<()> {
        // Check basic structure
        self.validate_size(event)?;
        self.validate_content(event)?;

        // Validate context
        self.validate_context(&event.context)?;

        // Validate causality chain length
        if event.causality_chain.len() > self.max_causality_depth {
            return Err(DatabaseError::validation(format!(
                "Causality chain length {} exceeds maximum {}",
                event.causality_chain.len(),
                self.max_causality_depth
            )));
        }

        Ok(())
    }

    /// Validate causality chain
    pub fn validate_causality(&self, chain: &[EventId]) -> DatabaseResult<()> {
        if chain.len() > self.max_causality_depth {
            return Err(DatabaseError::validation(format!(
                "Causality chain length {} exceeds maximum {}",
                chain.len(),
                self.max_causality_depth
            )));
        }

        // Check for duplicates in causality chain
        let mut seen = std::collections::HashSet::new();
        for &event_id in chain {
            if !seen.insert(event_id) {
                return Err(DatabaseError::validation(format!(
                    "Duplicate event ID {} in causality chain",
                    event_id
                )));
            }
        }

        Ok(())
    }

    /// Validate event context
    pub fn validate_context(&self, context: &EventContext) -> DatabaseResult<()> {
        // Validate goals
        self.validate_goals(context)?;

        // Validate resources
        self.validate_resources(context)?;

        // Validate temporal context
        if let Some(time_of_day) = &context.environment.temporal.time_of_day {
            if time_of_day.hour > 23 {
                return Err(DatabaseError::validation("Hour must be between 0 and 23"));
            }
            if time_of_day.minute > 59 {
                return Err(DatabaseError::validation("Minute must be between 0 and 59"));
            }
        }

        // Validate spatial context
        if let Some(spatial) = &context.environment.spatial {
            if let Some(bounds) = &spatial.bounds {
                // Check that location is within bounds
                let loc = spatial.location;
                if loc.0 < bounds.min.0
                    || loc.0 > bounds.max.0
                    || loc.1 < bounds.min.1
                    || loc.1 > bounds.max.1
                    || loc.2 < bounds.min.2
                    || loc.2 > bounds.max.2
                {
                    return Err(DatabaseError::validation(
                        "Location is outside defined bounds",
                    ));
                }
            }
        }

        Ok(())
    }
}

/// Validator that can check against known event IDs
pub struct ContextualEventValidator {
    basic: BasicEventValidator,
    known_events: std::collections::HashSet<EventId>,
}

impl ContextualEventValidator {
    /// Create new contextual validator
    pub fn new() -> Self {
        Self {
            basic: BasicEventValidator::new(),
            known_events: std::collections::HashSet::new(),
        }
    }

    /// Add known event ID for causality validation
    pub fn add_known_event(&mut self, event_id: EventId) {
        self.known_events.insert(event_id);
    }

    /// Remove known event ID
    pub fn remove_known_event(&mut self, event_id: &EventId) {
        self.known_events.remove(event_id);
    }

    /// Check if all parents in causality chain exist
    pub fn validate_causality_existence(&self, chain: &[EventId]) -> DatabaseResult<()> {
        for &parent_id in chain {
            if !self.known_events.contains(&parent_id) {
                return Err(DatabaseError::CausalityViolation {
                    event_id: 0, // Would be filled in by caller
                    parent_id,
                });
            }
        }
        Ok(())
    }
}

impl Default for ContextualEventValidator {
    fn default() -> Self {
        Self::new()
    }
}

impl ContextualEventValidator {
    pub fn validate_event(&self, event: &Event) -> DatabaseResult<()> {
        // First run basic validation
        self.basic.validate_event(event)?;

        // Then check causality existence if we have the context
        if !self.known_events.is_empty() {
            self.validate_causality_existence(&event.causality_chain)?;
        }

        Ok(())
    }

    pub fn validate_causality(&self, chain: &[EventId]) -> DatabaseResult<()> {
        self.basic.validate_causality(chain)?;
        self.validate_causality_existence(chain)?;
        Ok(())
    }

    pub fn validate_context(&self, context: &EventContext) -> DatabaseResult<()> {
        self.basic.validate_context(context)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{ActionOutcome, EventType};
    use agent_db_core::types::generate_event_id;
    use serde_json::json;

    fn create_valid_event() -> Event {
        Event::new(
            123,                      // agent_id
            "test_agent".to_string(), // agent_type
            456,                      // session_id
            EventType::Action {
                action_name: "test_action".to_string(),
                parameters: json!({"x": 10}),
                outcome: ActionOutcome::Success {
                    result: json!({"success": true}),
                },
                duration_ns: 1_000_000,
            },
            create_valid_context(),
        )
    }

    fn create_valid_context() -> EventContext {
        use crate::core::*;
        use std::collections::HashMap;

        EventContext::new(
            EnvironmentState {
                variables: HashMap::new(),
                spatial: None,
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
                description: "Valid goal".to_string(),
                priority: 0.5,
                deadline: None,
                progress: 0.3,
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

    #[test]
    fn test_valid_event_passes() {
        let validator = BasicEventValidator::new();
        let event = create_valid_event();

        assert!(validator.validate_event(&event).is_ok());
    }

    #[test]
    fn test_zero_agent_id_fails() {
        let validator = BasicEventValidator::new();
        let mut event = create_valid_event();
        event.agent_id = 0;

        assert!(validator.validate_event(&event).is_err());
    }

    #[test]
    fn test_empty_action_name_fails() {
        let validator = BasicEventValidator::new();
        let mut event = create_valid_event();
        event.event_type = EventType::Action {
            action_name: String::new(),
            parameters: json!({}),
            outcome: ActionOutcome::Success {
                result: json!(true),
            },
            duration_ns: 1000,
        };

        assert!(validator.validate_event(&event).is_err());
    }

    #[test]
    fn test_invalid_confidence_fails() {
        let validator = BasicEventValidator::new();
        let mut event = create_valid_event();
        event.event_type = EventType::Observation {
            observation_type: "test".to_string(),
            data: json!({}),
            confidence: 1.5, // Invalid - should be <= 1.0
            source: "test".to_string(),
        };

        assert!(validator.validate_event(&event).is_err());
    }

    #[test]
    fn test_causality_chain_validation() {
        let validator = BasicEventValidator::new();
        let parent1 = generate_event_id();
        let parent2 = generate_event_id();

        let chain = vec![parent1, parent2];
        assert!(validator.validate_causality(&chain).is_ok());

        // Test duplicate detection
        let invalid_chain = vec![parent1, parent1];
        assert!(validator.validate_causality(&invalid_chain).is_err());
    }

    #[test]
    fn test_contextual_validator() {
        let mut validator = ContextualEventValidator::new();
        let parent_id = generate_event_id();

        // Add known event
        validator.add_known_event(parent_id);

        // Create event with valid parent
        let event = create_valid_event().with_parent(parent_id);
        assert!(validator.validate_event(&event).is_ok());

        // Create event with unknown parent
        let unknown_parent = generate_event_id();
        let invalid_event = create_valid_event().with_parent(unknown_parent);
        assert!(validator.validate_event(&invalid_event).is_err());
    }

    #[test]
    fn test_resource_validation() {
        let validator = BasicEventValidator::new();
        let mut context = create_valid_context();

        // Test invalid CPU percentage
        context.resources.computational.cpu_percent = 150.0;
        assert!(validator.validate_context(&context).is_err());

        // Reset and test invalid resource usage
        context.resources.computational.cpu_percent = 50.0;
        context.resources.external.insert(
            "test_resource".to_string(),
            crate::core::ResourceAvailability {
                available: true,
                capacity: 100.0,
                current_usage: 150.0, // Exceeds capacity
                estimated_cost: None,
            },
        );
        assert!(validator.validate_context(&context).is_err());
    }

    #[test]
    fn test_temporal_validation() {
        let validator = BasicEventValidator::new();
        let mut context = create_valid_context();

        // Test invalid hour
        context.environment.temporal.time_of_day = Some(crate::core::TimeOfDay {
            hour: 25, // Invalid
            minute: 30,
            timezone: "UTC".to_string(),
        });

        assert!(validator.validate_context(&context).is_err());
    }
}
