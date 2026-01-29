//! Graph inference from events
//!
//! This module implements the core intelligence of the agentic database:
//! automatically inferring relationships between agents, events, and contexts
//! based on patterns in the event stream.

use crate::structures::{
    ConceptType, EdgeType, EdgeWeight, GoalStatus, Graph, GraphEdge, GraphNode, InteractionType,
    NodeId, NodeType,
};
use crate::{GraphError, GraphResult};
use agent_db_core::types::{AgentId, ContextHash, EventId, Timestamp};
use agent_db_events::core::MetadataValue;
use agent_db_events::{ActionOutcome, Event, EventContext, EventType};
use serde_json::json;
use std::collections::{HashMap, HashSet, VecDeque};
use std::time::{SystemTime, UNIX_EPOCH};

/// Configuration for inference algorithms
#[derive(Debug, Clone)]
pub struct InferenceConfig {
    /// Minimum confidence threshold for creating relationships
    pub min_confidence_threshold: f32,

    /// Time window for causal inference (nanoseconds)
    pub causality_time_window: u64,

    /// Minimum co-occurrence count for pattern detection
    pub min_co_occurrence_count: u32,

    /// Weight decay factor for temporal relationships
    pub temporal_decay_factor: f32,

    /// Similarity threshold for context relationships
    pub context_similarity_threshold: f32,

    /// Maximum events to consider in batch processing
    pub batch_size: usize,
}

/// Statistics about inference operations
#[derive(Debug, Clone, Default)]
pub struct InferenceStats {
    pub events_processed: u64,
    pub relationships_created: u64,
    pub nodes_created: u64,
    pub patterns_detected: u64,
    pub processing_time_ms: u64,
    pub last_inference_time: Timestamp,
}

/// Temporal pattern detected in event sequences
#[derive(Debug, Clone)]
pub struct TemporalPattern {
    pub pattern_name: String,
    pub event_sequence: Vec<String>, // Event type sequence
    pub average_interval: u64,       // Average time between events
    pub confidence: f32,             // Pattern confidence
    pub occurrence_count: u32,       // How often observed
}

/// Contextual association between events/agents
#[derive(Debug, Clone)]
pub struct ContextualAssociation {
    pub association_name: String,
    pub entities: Vec<EntityReference>,
    pub strength: f32,
    pub evidence_count: u32,
    pub last_observed: Timestamp,
}

/// Reference to an entity in the graph
#[derive(Debug, Clone)]
pub enum EntityReference {
    Agent(AgentId),
    Event(EventId),
    Context(ContextHash),
    Node(NodeId),
}

/// Core inference engine for the agentic database
pub struct GraphInference {
    graph: Graph,
    config: InferenceConfig,
    stats: InferenceStats,

    /// Temporal buffer for causal analysis
    temporal_buffer: VecDeque<Event>,

    /// Context similarity cache
    context_similarity_cache: HashMap<(ContextHash, ContextHash), f32>,

    /// Detected patterns
    temporal_patterns: Vec<TemporalPattern>,
    contextual_associations: Vec<ContextualAssociation>,
}

impl Default for InferenceConfig {
    fn default() -> Self {
        Self {
            min_confidence_threshold: 0.3,
            causality_time_window: 10_000_000_000, // 10 seconds
            min_co_occurrence_count: 3,
            temporal_decay_factor: 0.95,
            context_similarity_threshold: 0.7,
            batch_size: 1000,
        }
    }
}

impl GraphInference {
    /// Create a new inference engine with default configuration
    pub fn new() -> Self {
        Self::with_config(InferenceConfig::default())
    }

    /// Create inference engine with custom configuration
    pub fn with_config(config: InferenceConfig) -> Self {
        Self {
            graph: Graph::new(),
            config,
            stats: InferenceStats::default(),
            temporal_buffer: VecDeque::new(),
            context_similarity_cache: HashMap::new(),
            temporal_patterns: Vec::new(),
            contextual_associations: Vec::new(),
        }
    }

    /// Process a new event and update the graph
    pub fn process_event(&mut self, event: Event) -> GraphResult<Vec<NodeId>> {
        let start_time = SystemTime::now();
        let mut created_nodes = Vec::new();
        tracing::info!(
            "Inference start event_id={} agent_id={} session_id={}",
            event.id,
            event.agent_id,
            event.session_id
        );

        // Ensure agent node exists
        tracing::info!("Inference ensure_agent_node event_id={}", event.id);
        let agent_node_id = self.ensure_agent_node(event.agent_id, &event)?;
        created_nodes.push(agent_node_id);

        // Create event node
        tracing::info!("Inference create_event_node event_id={}", event.id);
        let event_node_id = self.create_event_node(&event)?;
        created_nodes.push(event_node_id);

        // Ensure context node exists
        tracing::info!("Inference ensure_context_node event_id={}", event.id);
        let context_node_id = self.ensure_context_node(&event.context)?;
        if context_node_id != event_node_id {
            created_nodes.push(context_node_id);
        }

        // Create basic relationships
        tracing::info!(
            "Inference create_agent_event_relationship event_id={}",
            event.id
        );
        self.create_agent_event_relationship(agent_node_id, event_node_id, &event)?;
        tracing::info!(
            "Inference create_event_context_relationship event_id={}",
            event.id
        );
        self.create_event_context_relationship(event_node_id, context_node_id, &event)?;

        // Create goal/tool/result relationships for richer learning graph
        tracing::info!(
            "Inference attach_goal_tool_result_relationships event_id={}",
            event.id
        );
        self.attach_goal_tool_result_relationships(agent_node_id, event_node_id, &event)?;

        // Add to temporal buffer for causal analysis
        tracing::info!("Inference update_temporal_buffer event_id={}", event.id);
        self.temporal_buffer.push_back(event.clone());
        if self.temporal_buffer.len() > self.config.batch_size {
            self.temporal_buffer.pop_front();
        }

        // Perform causal inference
        tracing::info!("Inference infer_causal_relationships event_id={}", event.id);
        self.infer_causal_relationships(&event)?;

        // Infer contextual relationships
        tracing::info!(
            "Inference infer_contextual_relationships event_id={}",
            event.id
        );
        self.infer_contextual_relationships(&event)?;

        // Update statistics
        self.stats.events_processed += 1;
        if let Ok(duration) = start_time.elapsed() {
            self.stats.processing_time_ms += duration.as_millis() as u64;
        }

        // Periodically clean up buffer and detect patterns
        if self.stats.events_processed % 100 == 0 {
            tracing::info!("Inference detect_patterns event_id={}", event.id);
            self.detect_patterns()?;
            self.cleanup_old_associations();
        }
        tracing::info!(
            "Inference done event_id={} nodes_created={}",
            event.id,
            created_nodes.len()
        );

        Ok(created_nodes)
    }

    /// Batch process multiple events efficiently
    pub fn process_events(&mut self, events: Vec<Event>) -> GraphResult<InferenceResults> {
        let start_time = SystemTime::now();
        let mut results = InferenceResults::default();

        for event in events {
            match self.process_event(event) {
                Ok(nodes) => {
                    results.nodes_created.extend(nodes);
                    results.events_processed += 1;
                },
                Err(e) => {
                    results.errors.push(e);
                },
            }
        }

        // Perform batch analysis
        self.detect_patterns()?;
        results.patterns_detected = self.temporal_patterns.len();
        results.associations_found = self.contextual_associations.len();

        if let Ok(duration) = start_time.elapsed() {
            results.processing_time_ms = duration.as_millis() as u64;
        }

        Ok(results)
    }

    /// Get the current graph
    pub fn graph(&self) -> &Graph {
        &self.graph
    }

    /// Get mutable reference to graph
    pub fn graph_mut(&mut self) -> &mut Graph {
        &mut self.graph
    }

    /// Get inference statistics
    pub fn stats(&self) -> &InferenceStats {
        &self.stats
    }

    /// Get detected temporal patterns
    pub fn get_temporal_patterns(&self) -> &[TemporalPattern] {
        &self.temporal_patterns
    }

    /// Get contextual associations
    pub fn get_contextual_associations(&self) -> &[ContextualAssociation] {
        &self.contextual_associations
    }

    // Private helper methods

    /// Ensure agent node exists in graph
    fn ensure_agent_node(&mut self, agent_id: AgentId, event: &Event) -> GraphResult<NodeId> {
        if let Some(node) = self.graph.get_agent_node(agent_id) {
            Ok(node.id)
        } else {
            // Create new agent node
            let agent_type = match &event.event_type {
                EventType::Action { .. } => "action_agent",
                EventType::Observation { .. } => "observer_agent",
                EventType::Communication { .. } => "communicator_agent",
                EventType::Cognitive { .. } => "cognitive_agent",
                EventType::Learning { .. } => "learning_agent",
                EventType::Context { .. } => "context_agent",
            };

            let node = GraphNode::new(NodeType::Agent {
                agent_id,
                agent_type: agent_type.to_string(),
                capabilities: vec!["event_generation".to_string()],
            });

            let node_id = self.graph.add_node(node);
            self.stats.nodes_created += 1;
            Ok(node_id)
        }
    }

    /// Create event node
    fn create_event_node(&mut self, event: &Event) -> GraphResult<NodeId> {
        let event_type_name = match &event.event_type {
            EventType::Action { action_name, .. } => action_name.clone(),
            EventType::Observation {
                observation_type, ..
            } => observation_type.clone(),
            EventType::Communication { message_type, .. } => message_type.clone(),
            EventType::Cognitive { process_type, .. } => format!("{:?}", process_type),
            EventType::Learning { .. } => "LearningTelemetry".to_string(),
            EventType::Context { context_type, .. } => format!("Context:{}", context_type),
        };

        let significance = self.calculate_event_significance(event);

        let mut node = GraphNode::new(NodeType::Event {
            event_id: event.id,
            event_type: event_type_name,
            significance,
        });

        let tool_names = self.extract_tool_names(event);
        let goal_ids: Vec<u64> = event
            .context
            .active_goals
            .iter()
            .map(|goal| goal.id)
            .collect();

        node.properties
            .insert("agent_id".to_string(), json!(event.agent_id));
        node.properties
            .insert("session_id".to_string(), json!(event.session_id));
        node.properties
            .insert("agent_type".to_string(), json!(event.agent_type));
        node.properties.insert(
            "event_type".to_string(),
            json!(self.event_type_label(event)),
        );
        node.properties
            .insert("tool_names".to_string(), json!(tool_names));
        node.properties
            .insert("goal_ids".to_string(), json!(goal_ids));

        let node_id = self.graph.add_node(node);
        self.stats.nodes_created += 1;
        Ok(node_id)
    }

    /// Ensure context node exists
    fn ensure_context_node(&mut self, context: &EventContext) -> GraphResult<NodeId> {
        if let Some(node) = self.graph.get_context_node(context.fingerprint) {
            let node_id = node.id;
            // Update frequency
            if let Some(mut_node) = self.graph.get_node_mut(node_id) {
                if let NodeType::Context {
                    ref mut frequency, ..
                } = &mut mut_node.node_type
                {
                    *frequency += 1;
                }
                mut_node.touch();
            }
            Ok(node_id)
        } else {
            let context_type = self.classify_context(context);

            let node = GraphNode::new(NodeType::Context {
                context_hash: context.fingerprint,
                context_type,
                frequency: 1,
            });

            let node_id = self.graph.add_node(node);
            self.stats.nodes_created += 1;
            Ok(node_id)
        }
    }

    /// Create relationship between agent and event
    fn create_agent_event_relationship(
        &mut self,
        agent_id: NodeId,
        event_id: NodeId,
        event: &Event,
    ) -> GraphResult<()> {
        let interaction_type = match &event.event_type {
            EventType::Action { .. } => InteractionType::Coordination,
            EventType::Observation { .. } => InteractionType::InformationExchange,
            EventType::Communication { .. } => InteractionType::Communication,
            EventType::Cognitive { .. } => InteractionType::Coordination,
            EventType::Learning { .. } => InteractionType::Coordination,
            EventType::Context { .. } => InteractionType::InformationExchange,
        };

        let edge = GraphEdge::new(
            agent_id,
            event_id,
            EdgeType::Interaction {
                interaction_type,
                frequency: 1,
                success_rate: self.calculate_success_rate(&event.event_type),
            },
            0.8, // Strong relationship weight
        );

        self.graph.add_edge(edge);
        self.stats.relationships_created += 1;
        Ok(())
    }

    /// Create relationship between event and context
    fn create_event_context_relationship(
        &mut self,
        event_id: NodeId,
        context_id: NodeId,
        _event: &Event,
    ) -> GraphResult<()> {
        let edge = GraphEdge::new(
            event_id,
            context_id,
            EdgeType::Contextual {
                similarity: 1.0, // Perfect similarity since event occurred in this context
                co_occurrence_rate: 1.0,
            },
            0.9, // Very strong contextual relationship
        );

        self.graph.add_edge(edge);
        self.stats.relationships_created += 1;
        Ok(())
    }

    /// Attach goal, tool, and result relationships to enrich the graph
    fn attach_goal_tool_result_relationships(
        &mut self,
        agent_node_id: NodeId,
        event_node_id: NodeId,
        event: &Event,
    ) -> GraphResult<()> {
        let goal_ids: Vec<_> = event.context.active_goals.iter().collect();
        for goal in goal_ids {
            let goal_node_id = self.ensure_goal_node(goal)?;
            self.add_or_strengthen_association(
                event_node_id,
                goal_node_id,
                "ContributesToGoal",
                0.8,
                json!({
                    "event_id": event.id.to_string(),
                    "agent_id": event.agent_id,
                    "session_id": event.session_id,
                    "goal_id": goal.id,
                    "goal_priority": goal.priority,
                }),
            );
            self.add_or_strengthen_association(
                agent_node_id,
                goal_node_id,
                "ResponsibleForGoal",
                0.6,
                json!({
                    "agent_id": event.agent_id,
                    "session_id": event.session_id,
                    "goal_id": goal.id,
                }),
            );
        }

        for tool_name in self.extract_tool_names(event) {
            let tool_node_id = self.ensure_tool_node(&tool_name)?;
            self.add_or_strengthen_association(
                event_node_id,
                tool_node_id,
                "UsesTool",
                0.7,
                json!({
                    "event_id": event.id.to_string(),
                    "agent_id": event.agent_id,
                    "session_id": event.session_id,
                    "tool_name": tool_name,
                }),
            );
        }

        if let Some((result_type, summary)) = self.extract_result_summary(event) {
            let result_key = format!("event:{}:{}", event.id, result_type);
            let result_node_id = self.ensure_result_node(&result_key, &result_type, &summary)?;
            self.add_or_strengthen_association(
                event_node_id,
                result_node_id,
                "ProducesResult",
                0.7,
                json!({
                    "event_id": event.id.to_string(),
                    "agent_id": event.agent_id,
                    "session_id": event.session_id,
                    "result_type": result_type,
                }),
            );
        }

        Ok(())
    }

    fn ensure_goal_node(&mut self, goal: &agent_db_events::core::Goal) -> GraphResult<NodeId> {
        if let Some(node_id) = self.graph.get_goal_node(goal.id).map(|node| node.id) {
            if let Some(existing) = self.graph.get_node_mut(node_id) {
                existing
                    .properties
                    .insert("priority".to_string(), json!(goal.priority));
                existing
                    .properties
                    .insert("progress".to_string(), json!(goal.progress));
                if let Some(deadline) = goal.deadline {
                    existing
                        .properties
                        .insert("deadline".to_string(), json!(deadline));
                }
                if let NodeType::Goal {
                    description,
                    priority,
                    status,
                    ..
                } = &mut existing.node_type
                {
                    *description = goal.description.clone();
                    *priority = goal.priority;
                    *status = if goal.progress >= 1.0 {
                        GoalStatus::Completed
                    } else {
                        GoalStatus::Active
                    };
                }
                existing.touch();
            }
            Ok(node_id)
        } else {
            let status = if goal.progress >= 1.0 {
                GoalStatus::Completed
            } else {
                GoalStatus::Active
            };

            let mut node = GraphNode::new(NodeType::Goal {
                goal_id: goal.id,
                description: goal.description.clone(),
                priority: goal.priority,
                status,
            });
            node.properties
                .insert("progress".to_string(), json!(goal.progress));
            if let Some(deadline) = goal.deadline {
                node.properties
                    .insert("deadline".to_string(), json!(deadline));
            }

            let node_id = self.graph.add_node(node);
            self.stats.nodes_created += 1;
            Ok(node_id)
        }
    }

    fn ensure_tool_node(&mut self, tool_name: &str) -> GraphResult<NodeId> {
        if let Some(node_id) = self.graph.get_tool_node(tool_name).map(|node| node.id) {
            if let Some(existing) = self.graph.get_node_mut(node_id) {
                let count = existing
                    .properties
                    .get("usage_count")
                    .and_then(|value| value.as_u64())
                    .unwrap_or(0)
                    .saturating_add(1);
                existing
                    .properties
                    .insert("usage_count".to_string(), json!(count));
                existing.touch();
            }
            Ok(node_id)
        } else {
            let mut node = GraphNode::new(NodeType::Tool {
                tool_name: tool_name.to_string(),
                tool_type: "external".to_string(),
            });
            node.properties.insert("usage_count".to_string(), json!(1));
            let node_id = self.graph.add_node(node);
            self.stats.nodes_created += 1;
            Ok(node_id)
        }
    }

    fn ensure_result_node(
        &mut self,
        result_key: &str,
        result_type: &str,
        summary: &str,
    ) -> GraphResult<NodeId> {
        if let Some(node) = self.graph.get_result_node(result_key) {
            Ok(node.id)
        } else {
            let mut node = GraphNode::new(NodeType::Result {
                result_key: result_key.to_string(),
                result_type: result_type.to_string(),
                summary: summary.to_string(),
            });
            node.properties
                .insert("summary".to_string(), json!(summary));
            let node_id = self.graph.add_node(node);
            self.stats.nodes_created += 1;
            Ok(node_id)
        }
    }

    fn add_or_strengthen_association(
        &mut self,
        source: NodeId,
        target: NodeId,
        association_type: &str,
        weight: EdgeWeight,
        properties: serde_json::Value,
    ) {
        if let Some(edge) = self.graph.get_edge_between_mut(source, target) {
            edge.strengthen(weight * 0.1);
            return;
        }

        let mut edge = GraphEdge::new(
            source,
            target,
            EdgeType::Association {
                association_type: association_type.to_string(),
                evidence_count: 1,
                statistical_significance: weight,
            },
            weight,
        );
        edge.properties.insert("details".to_string(), properties);
        self.graph.add_edge(edge);
        self.stats.relationships_created += 1;
    }

    fn extract_tool_names(&self, event: &Event) -> Vec<String> {
        let mut tools: HashSet<String> = HashSet::new();

        for key in ["tool", "tool_name", "tools", "tool_used"] {
            if let Some(value) = event.metadata.get(key) {
                self.collect_tools_from_metadata(value, &mut tools);
            }
        }

        if let EventType::Action { parameters, .. } = &event.event_type {
            self.collect_tools_from_json(parameters, &mut tools);
        }

        let mut list: Vec<String> = tools.into_iter().collect();
        list.sort();
        list
    }

    fn collect_tools_from_metadata(&self, value: &MetadataValue, tools: &mut HashSet<String>) {
        match value {
            MetadataValue::String(name) => {
                if !name.trim().is_empty() {
                    tools.insert(name.trim().to_string());
                }
            },
            MetadataValue::Json(json) => {
                self.collect_tools_from_json(json, tools);
            },
            _ => {},
        }
    }

    fn collect_tools_from_json(&self, value: &serde_json::Value, tools: &mut HashSet<String>) {
        match value {
            serde_json::Value::String(name) => {
                if !name.trim().is_empty() {
                    tools.insert(name.trim().to_string());
                }
            },
            serde_json::Value::Array(items) => {
                for item in items {
                    self.collect_tools_from_json(item, tools);
                }
            },
            serde_json::Value::Object(map) => {
                for key in ["tool", "tool_name", "tools", "tool_used"] {
                    if let Some(value) = map.get(key) {
                        self.collect_tools_from_json(value, tools);
                    }
                }
            },
            _ => {},
        }
    }

    fn extract_result_summary(&self, event: &Event) -> Option<(String, String)> {
        match &event.event_type {
            EventType::Action { outcome, .. } => match outcome {
                ActionOutcome::Success { result } => Some((
                    "action_success".to_string(),
                    self.format_json_summary(result),
                )),
                ActionOutcome::Failure { error, .. } => {
                    Some(("action_failure".to_string(), self.truncate_summary(error)))
                },
                ActionOutcome::Partial { result, issues } => {
                    let mut summary = self.format_json_summary(result);
                    if !issues.is_empty() {
                        summary.push_str(" | issues: ");
                        summary.push_str(&issues.join(", "));
                    }
                    Some((
                        "action_partial".to_string(),
                        self.truncate_summary(&summary),
                    ))
                },
            },
            EventType::Observation { data, .. } => {
                Some(("observation".to_string(), self.format_json_summary(data)))
            },
            EventType::Cognitive { output, .. } => Some((
                "cognitive_output".to_string(),
                self.format_json_summary(output),
            )),
            EventType::Communication { content, .. } => Some((
                "communication".to_string(),
                self.format_json_summary(content),
            )),
            EventType::Learning { event } => Some((
                "learning_telemetry".to_string(),
                self.format_json_summary(&serde_json::json!(event)),
            )),
            EventType::Context { text, .. } => {
                Some(("context".to_string(), self.truncate_summary(text)))
            },
        }
    }

    fn format_json_summary(&self, value: &serde_json::Value) -> String {
        self.truncate_summary(&value.to_string())
    }

    fn truncate_summary(&self, value: &str) -> String {
        let max_len = 200;
        if value.len() <= max_len {
            value.to_string()
        } else {
            format!("{}...", &value[..max_len])
        }
    }

    fn event_type_label(&self, event: &Event) -> String {
        match &event.event_type {
            EventType::Action { .. } => "Action".to_string(),
            EventType::Observation { .. } => "Observation".to_string(),
            EventType::Communication { .. } => "Communication".to_string(),
            EventType::Cognitive { .. } => "Cognitive".to_string(),
            EventType::Learning { .. } => "Learning".to_string(),
            EventType::Context { .. } => "Context".to_string(),
        }
    }

    /// Infer causal relationships from temporal patterns
    fn infer_causal_relationships(&mut self, current_event: &Event) -> GraphResult<()> {
        let current_time = current_event.timestamp;
        tracing::info!("Causal inference start event_id={}", current_event.id);

        // Collect causal candidates first to avoid borrowing issues
        let mut causal_candidates = Vec::new();
        for previous_event in self.temporal_buffer.iter().rev() {
            if previous_event.id == current_event.id {
                continue;
            }

            let time_diff = current_time.saturating_sub(previous_event.timestamp);

            if time_diff <= self.config.causality_time_window && time_diff > 0 {
                causal_candidates.push((previous_event.clone(), time_diff));
            }
        }

        // Now process causal candidates
        for (previous_event, time_diff) in causal_candidates {
            let causal_strength = self.calculate_causal_strength(&previous_event, current_event);

            if causal_strength > self.config.min_confidence_threshold {
                // Create causal relationship
                if let (Some(prev_node), Some(curr_node)) = (
                    self.graph.get_event_node(previous_event.id),
                    self.graph.get_event_node(current_event.id),
                ) {
                    let edge = GraphEdge::new(
                        prev_node.id,
                        curr_node.id,
                        EdgeType::Causality {
                            strength: causal_strength,
                            lag_ms: time_diff / 1_000_000, // Convert to milliseconds
                        },
                        causal_strength,
                    );

                    self.graph.add_edge(edge);
                    self.stats.relationships_created += 1;
                }
            }
        }
        tracing::info!("Causal inference done event_id={}", current_event.id);
        Ok(())
    }

    /// Infer contextual relationships
    fn infer_contextual_relationships(&mut self, current_event: &Event) -> GraphResult<()> {
        tracing::info!("Contextual inference start event_id={}", current_event.id);
        // Collect context pairs first to avoid borrowing issues
        let mut context_pairs = Vec::new();
        for previous_event in self.temporal_buffer.iter() {
            if previous_event.id == current_event.id {
                continue;
            }
            context_pairs.push((previous_event.context.clone(), previous_event.id));
        }

        // Now calculate similarities
        for (prev_context, prev_event_id) in context_pairs {
            let similarity =
                self.calculate_context_similarity(&prev_context, &current_event.context);

            if similarity > self.config.context_similarity_threshold {
                // Create contextual association
                if let (Some(prev_node), Some(curr_node)) = (
                    self.graph.get_event_node(prev_event_id),
                    self.graph.get_event_node(current_event.id),
                ) {
                    let edge = GraphEdge::new(
                        prev_node.id,
                        curr_node.id,
                        EdgeType::Contextual {
                            similarity,
                            co_occurrence_rate: 1.0,
                        },
                        similarity * 0.7, // Weight based on similarity
                    );

                    self.graph.add_edge(edge);
                    self.stats.relationships_created += 1;
                }
            }
        }
        tracing::info!("Contextual inference done event_id={}", current_event.id);
        Ok(())
    }

    /// Detect temporal and behavioral patterns
    fn detect_patterns(&mut self) -> GraphResult<()> {
        // Detect temporal patterns in event sequences
        self.detect_temporal_patterns()?;

        // Detect contextual associations
        self.detect_contextual_associations()?;

        self.stats.patterns_detected += 1;
        Ok(())
    }

    /// Detect temporal patterns in event sequences
    fn detect_temporal_patterns(&mut self) -> GraphResult<()> {
        if self.temporal_buffer.len() < 2 {
            return Ok(()); // Need at least 2 events for pattern
        }

        let events: Vec<_> = self.temporal_buffer.iter().collect();

        // Look for repeating sequences of event types with variable lengths (2 to 4)
        for sequence_length in 2..=4 {
            if events.len() < sequence_length {
                continue;
            }

            for window in events.windows(sequence_length) {
                let event_types: Vec<String> = window
                    .iter()
                    .map(|e| match &e.event_type {
                        EventType::Action { action_name, .. } => action_name.clone(),
                        EventType::Observation {
                            observation_type, ..
                        } => observation_type.clone(),
                        EventType::Communication { message_type, .. } => message_type.clone(),
                        EventType::Cognitive { process_type, .. } => format!("{:?}", process_type),
                        EventType::Learning { .. } => "LearningTelemetry".to_string(),
                        EventType::Context { context_type, .. } => {
                            format!("Context:{}", context_type)
                        },
                    })
                    .collect();

                // Calculate average interval
                let mut total_interval = 0u64;
                for i in 1..window.len() {
                    total_interval += window[i].timestamp.saturating_sub(window[i - 1].timestamp);
                }
                let avg_interval = total_interval / (window.len() - 1) as u64;

                // Check if this pattern already exists or create new one
                let pattern_name = event_types.join("->");
                if let Some(existing) = self
                    .temporal_patterns
                    .iter_mut()
                    .find(|p| p.pattern_name == pattern_name)
                {
                    existing.occurrence_count += 1;
                    existing.confidence = (existing.confidence * 0.95 + 0.05).min(1.0);
                } else {
                    self.temporal_patterns.push(TemporalPattern {
                        pattern_name,
                        event_sequence: event_types,
                        average_interval: avg_interval,
                        confidence: 0.2, // Lower initial confidence for new patterns
                        occurrence_count: 1,
                    });
                }
            }
        }

        Ok(())
    }

    /// Detect contextual associations
    fn detect_contextual_associations(&mut self) -> GraphResult<()> {
        // Group events by similar contexts
        let mut context_groups: HashMap<ContextHash, Vec<&Event>> = HashMap::new();

        for event in &self.temporal_buffer {
            context_groups
                .entry(event.context.fingerprint)
                .or_insert_with(Vec::new)
                .push(event);
        }

        // Look for associations between contexts
        for (context_hash, events) in context_groups {
            if events.len() >= self.config.min_co_occurrence_count as usize {
                let association_name = format!("context_{}", context_hash);

                let entities: Vec<EntityReference> = events
                    .iter()
                    .map(|e| EntityReference::Event(e.id))
                    .collect();

                self.contextual_associations.push(ContextualAssociation {
                    association_name,
                    entities,
                    strength: (events.len() as f32 / self.temporal_buffer.len() as f32).min(1.0),
                    evidence_count: events.len() as u32,
                    last_observed: events.iter().map(|e| e.timestamp).max().unwrap_or(0),
                });
            }
        }

        Ok(())
    }

    /// Clean up old associations that are no longer relevant
    fn cleanup_old_associations(&mut self) {
        let current_time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap()
            .as_nanos() as Timestamp;

        let cutoff_time = current_time.saturating_sub(24 * 60 * 60 * 1_000_000_000); // 24 hours

        self.contextual_associations
            .retain(|assoc| assoc.last_observed > cutoff_time);
    }

    /// Calculate event significance based on various factors
    fn calculate_event_significance(&self, event: &Event) -> f32 {
        let mut significance = 0.5; // Base significance

        // Factor in event type
        significance += match &event.event_type {
            EventType::Action { .. } => 0.2,
            EventType::Communication { .. } => 0.3,
            EventType::Cognitive { .. } => 0.1,
            EventType::Observation { .. } => 0.1,
            EventType::Learning { .. } => 0.05,
            EventType::Context { .. } => 0.15,
        };

        // Factor in causality chain length
        significance += (event.causality_chain.len() as f32 * 0.05).min(0.3);

        // Factor in context complexity (more goals/resources = higher significance)
        significance += (event.context.active_goals.len() as f32 * 0.02).min(0.2);

        significance.min(1.0)
    }

    /// Calculate success rate from event type
    fn calculate_success_rate(&self, event_type: &EventType) -> f32 {
        match event_type {
            EventType::Action { outcome, .. } => match outcome {
                ActionOutcome::Success { .. } => 1.0,
                ActionOutcome::Partial { .. } => 0.7,
                ActionOutcome::Failure { .. } => 0.0,
            },
            _ => 0.8, // Default success rate for non-actions
        }
    }

    /// Calculate causal strength between two events
    fn calculate_causal_strength(&mut self, prev_event: &Event, curr_event: &Event) -> f32 {
        let mut strength = 0.0;

        // Explicit causality chain is the strongest evidence
        if curr_event.causality_chain.contains(&prev_event.id) {
            return 1.0;
        }

        // Same agent increases causal likelihood
        if prev_event.agent_id == curr_event.agent_id {
            strength += 0.4;
        }

        // Context similarity increases causal likelihood
        let context_sim =
            self.calculate_context_similarity(&prev_event.context, &curr_event.context);
        strength += context_sim * 0.3;

        // Action success leading to another action suggests causality
        if let EventType::Action { outcome, .. } = &prev_event.event_type {
            if matches!(outcome, ActionOutcome::Success { .. }) {
                strength += 0.3;
            }
        }

        strength.min(1.0)
    }

    /// Calculate context similarity
    fn calculate_context_similarity(&mut self, ctx1: &EventContext, ctx2: &EventContext) -> f32 {
        // Check cache first
        let cache_key = if ctx1.fingerprint < ctx2.fingerprint {
            (ctx1.fingerprint, ctx2.fingerprint)
        } else {
            (ctx2.fingerprint, ctx1.fingerprint)
        };

        if let Some(&cached) = self.context_similarity_cache.get(&cache_key) {
            return cached;
        }

        // Calculate similarity
        let mut similarity = 0.0;

        // Fingerprint exact match
        if ctx1.fingerprint == ctx2.fingerprint {
            similarity = 1.0;
        } else {
            // Goal similarity
            let goal_similarity =
                self.calculate_goal_similarity(&ctx1.active_goals, &ctx2.active_goals);
            similarity += goal_similarity * 0.4;

            // Resource similarity
            let resource_similarity =
                self.calculate_resource_similarity(&ctx1.resources, &ctx2.resources);
            similarity += resource_similarity * 0.3;

            // Environment similarity
            similarity += 0.3; // Simplified - would compare environment variables
        }

        // Cache result
        self.context_similarity_cache.insert(cache_key, similarity);

        similarity
    }

    /// Calculate goal similarity
    fn calculate_goal_similarity(
        &self,
        goals1: &[agent_db_events::Goal],
        goals2: &[agent_db_events::Goal],
    ) -> f32 {
        if goals1.is_empty() && goals2.is_empty() {
            return 1.0;
        }

        if goals1.is_empty() || goals2.is_empty() {
            return 0.0;
        }

        let mut matches = 0;
        for goal1 in goals1 {
            for goal2 in goals2 {
                if goal1.id == goal2.id {
                    matches += 1;
                    break;
                }
            }
        }

        matches as f32 / goals1.len().max(goals2.len()) as f32
    }

    /// Calculate resource similarity
    fn calculate_resource_similarity(
        &self,
        res1: &agent_db_events::ResourceState,
        res2: &agent_db_events::ResourceState,
    ) -> f32 {
        // Simplified resource similarity based on computational resources
        let cpu_diff = (res1.computational.cpu_percent - res2.computational.cpu_percent).abs();
        let cpu_similarity = 1.0 - (cpu_diff / 100.0).min(1.0);

        let memory_ratio = res1
            .computational
            .memory_bytes
            .min(res2.computational.memory_bytes) as f32
            / res1
                .computational
                .memory_bytes
                .max(res2.computational.memory_bytes) as f32;

        (cpu_similarity + memory_ratio) / 2.0
    }

    /// Classify context type based on context characteristics
    fn classify_context(&self, context: &EventContext) -> String {
        // High-pressure context
        if context.active_goals.iter().any(|g| g.priority > 0.8) {
            return "high_pressure".to_string();
        }

        // Resource-constrained context
        if context.resources.computational.cpu_percent > 80.0 {
            return "resource_constrained".to_string();
        }

        // Multi-goal context
        if context.active_goals.len() > 3 {
            return "multi_goal".to_string();
        }

        "normal".to_string()
    }

    // ========================================
    // Reinforcement Loop
    // ========================================
    // Methods to strengthen/weaken patterns based on outcomes

    /// Reinforce patterns based on episode outcome
    ///
    /// Strengthens successful patterns and weakens failure patterns
    /// based on the episode's events and final outcome
    pub async fn reinforce_patterns(
        &mut self,
        episode: &crate::episodes::Episode,
        success: bool,
        metrics: Option<EpisodeMetrics>,
    ) -> GraphResult<ReinforcementResult> {
        let mut result = ReinforcementResult::default();

        // Determine reinforcement strength based on success and metrics
        let reinforcement_strength =
            self.calculate_reinforcement_strength(success, metrics.as_ref());

        if success {
            // Strengthen successful paths in the episode
            result.patterns_strengthened = self
                .strengthen_successful_paths(episode, reinforcement_strength)
                .await?;
        } else {
            // Weaken failure paths in the episode
            result.patterns_weakened = self
                .weaken_failure_paths(episode, reinforcement_strength)
                .await?;
        }

        // Update pattern confidence scores
        result.patterns_updated = self
            .update_pattern_confidence(&episode.events, success)
            .await?;

        // Consolidate highly successful patterns into skills
        if success && reinforcement_strength > 0.8 {
            result.skills_consolidated = self.consolidate_patterns(episode).await?;
        }

        Ok(result)
    }

    /// Calculate how much to reinforce based on success and metrics
    fn calculate_reinforcement_strength(
        &self,
        success: bool,
        metrics: Option<&EpisodeMetrics>,
    ) -> f32 {
        let base_strength = if success { 0.1 } else { -0.1 };

        if let Some(m) = metrics {
            // Adjust based on metrics
            let time_factor = if m.duration_seconds < m.expected_duration_seconds {
                1.2 // Faster than expected = stronger reinforcement
            } else {
                0.8 // Slower than expected = weaker reinforcement
            };

            let quality_factor = m.quality_score.unwrap_or(1.0);

            base_strength * time_factor * quality_factor
        } else {
            base_strength
        }
    }

    /// Strengthen edges along successful paths in the episode
    async fn strengthen_successful_paths(
        &mut self,
        episode: &crate::episodes::Episode,
        strength: f32,
    ) -> GraphResult<usize> {
        let mut strengthened_count = 0;

        // Iterate through event pairs in the episode
        for window in episode.events.windows(2) {
            if let [event1_id, event2_id] = window {
                // Find nodes corresponding to these events
                if let (Some(node1), Some(node2)) = (
                    self.graph.get_event_node(*event1_id),
                    self.graph.get_event_node(*event2_id),
                ) {
                    // Strengthen the edge between them
                    if self.strengthen_edge(node1.id, node2.id, strength).await? {
                        strengthened_count += 1;
                    }
                }
            }
        }

        Ok(strengthened_count)
    }

    /// Weaken edges along failure paths in the episode
    async fn weaken_failure_paths(
        &mut self,
        episode: &crate::episodes::Episode,
        strength: f32,
    ) -> GraphResult<usize> {
        let mut weakened_count = 0;

        // Iterate through event pairs in the episode
        for window in episode.events.windows(2) {
            if let [event1_id, event2_id] = window {
                // Find nodes corresponding to these events
                if let (Some(node1), Some(node2)) = (
                    self.graph.get_event_node(*event1_id),
                    self.graph.get_event_node(*event2_id),
                ) {
                    // Weaken the edge between them
                    if self.weaken_edge(node1.id, node2.id, strength.abs()).await? {
                        weakened_count += 1;
                    }
                }
            }
        }

        Ok(weakened_count)
    }

    /// Update edge weight between two nodes (strengthen)
    async fn strengthen_edge(
        &mut self,
        from_node: NodeId,
        to_node: NodeId,
        delta: f32,
    ) -> GraphResult<bool> {
        // Try to find existing edge
        if let Some(edge) = self.graph.get_edge_between_mut(from_node, to_node) {
            // Edge exists - strengthen it
            edge.strengthen(delta);
            edge.record_success();
            Ok(true)
        } else {
            // Create new edge if it doesn't exist
            let mut edge = GraphEdge::new(
                from_node,
                to_node,
                EdgeType::Temporal {
                    average_interval_ms: 1000,
                    sequence_confidence: delta.max(0.1),
                },
                delta.max(0.1), // Ensure positive initial weight
            );
            edge.record_success();
            self.graph.add_edge(edge);
            Ok(true)
        }
    }

    /// Update edge weight between two nodes (weaken)
    async fn weaken_edge(
        &mut self,
        from_node: NodeId,
        to_node: NodeId,
        delta: f32,
    ) -> GraphResult<bool> {
        // Try to find existing edge
        if let Some(edge) = self.graph.get_edge_between_mut(from_node, to_node) {
            // Edge exists - weaken it
            edge.weaken(delta);
            edge.record_failure();
            Ok(true)
        } else {
            // Create new edge with low weight if it doesn't exist
            let mut edge = GraphEdge::new(
                from_node,
                to_node,
                EdgeType::Temporal {
                    average_interval_ms: 1000,
                    sequence_confidence: 0.1,
                },
                0.1, // Low initial weight for failures
            );
            edge.record_failure();
            self.graph.add_edge(edge);
            Ok(true)
        }
    }

    /// Update confidence scores for patterns involving these events
    async fn update_pattern_confidence(
        &mut self,
        event_ids: &[EventId],
        success: bool,
    ) -> GraphResult<usize> {
        let mut updated_count = 0;
        let confidence_delta = if success { 0.05 } else { -0.05 };

        // Get event types for the events in this episode to match patterns
        let mut episode_event_types = Vec::new();
        for id in event_ids {
            if let Some(node) = self.graph.get_event_node(*id) {
                if let NodeType::Event { event_type, .. } = &node.node_type {
                    episode_event_types.push(event_type.clone());
                }
            }
        }

        if episode_event_types.is_empty() {
            return Ok(0);
        }

        // Update temporal patterns that match subsequences of the episode
        for pattern in &mut self.temporal_patterns {
            let mut matches = false;

            // A pattern matches if its event_sequence is a subsequence of the episode
            if pattern.event_sequence.len() <= episode_event_types.len() {
                for window in episode_event_types.windows(pattern.event_sequence.len()) {
                    if window == pattern.event_sequence {
                        matches = true;
                        break;
                    }
                }
            }

            if matches {
                pattern.confidence = (pattern.confidence + confidence_delta).clamp(0.0, 1.0);
                if success {
                    pattern.occurrence_count += 1;
                }
                updated_count += 1;
            }
        }

        Ok(updated_count)
    }

    /// Consolidate highly successful patterns into reusable skills
    async fn consolidate_patterns(
        &mut self,
        _episode: &crate::episodes::Episode,
    ) -> GraphResult<usize> {
        let mut consolidated = 0;

        // Find patterns that appear frequently and successfully in this episode
        let high_confidence_patterns: Vec<&TemporalPattern> = self
            .temporal_patterns
            .iter()
            .filter(|p| p.confidence > 0.8 && p.occurrence_count > 10)
            .collect();

        for pattern in high_confidence_patterns {
            // Create a "strategy" concept node for this pattern (skill)
            let skill_node = GraphNode::new(NodeType::Concept {
                concept_name: format!("skill_{}", pattern.pattern_name),
                concept_type: ConceptType::Strategy,
                confidence: pattern.confidence,
            });

            self.graph.add_node(skill_node);
            consolidated += 1;
        }

        Ok(consolidated)
    }

    /// Get reinforcement statistics
    pub fn get_reinforcement_stats(&self) -> ReinforcementStats {
        ReinforcementStats {
            total_patterns: self.temporal_patterns.len(),
            high_confidence_patterns: self
                .temporal_patterns
                .iter()
                .filter(|p| p.confidence > 0.7)
                .count(),
            low_confidence_patterns: self
                .temporal_patterns
                .iter()
                .filter(|p| p.confidence < 0.3)
                .count(),
            average_confidence: if !self.temporal_patterns.is_empty() {
                self.temporal_patterns
                    .iter()
                    .map(|p| p.confidence)
                    .sum::<f32>()
                    / self.temporal_patterns.len() as f32
            } else {
                0.0
            },
        }
    }
}

/// Metrics for episode evaluation
#[derive(Debug, Clone)]
pub struct EpisodeMetrics {
    /// How long the episode took (seconds)
    pub duration_seconds: f32,

    /// Expected duration (seconds)
    pub expected_duration_seconds: f32,

    /// Quality score (0.0 to 1.0), if available
    pub quality_score: Option<f32>,

    /// Custom metrics
    pub custom_metrics: HashMap<String, f32>,
}

/// Result of reinforcement operation
#[derive(Debug, Default)]
pub struct ReinforcementResult {
    /// Number of patterns strengthened
    pub patterns_strengthened: usize,

    /// Number of patterns weakened
    pub patterns_weakened: usize,

    /// Number of patterns updated
    pub patterns_updated: usize,

    /// Number of skills consolidated
    pub skills_consolidated: usize,
}

/// Statistics about reinforcement learning
#[derive(Debug, Clone)]
pub struct ReinforcementStats {
    /// Total number of patterns being tracked
    pub total_patterns: usize,

    /// Patterns with confidence > 0.7
    pub high_confidence_patterns: usize,

    /// Patterns with confidence < 0.3
    pub low_confidence_patterns: usize,

    /// Average confidence across all patterns
    pub average_confidence: f32,
}

/// Results from batch inference processing
#[derive(Debug, Default)]
pub struct InferenceResults {
    pub events_processed: u64,
    pub nodes_created: Vec<NodeId>,
    pub patterns_detected: usize,
    pub associations_found: usize,
    pub processing_time_ms: u64,
    pub errors: Vec<GraphError>,
}
