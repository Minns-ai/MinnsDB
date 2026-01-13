//! Graph inference from events
//!
//! This module implements the core intelligence of the agentic database:
//! automatically inferring relationships between agents, events, and contexts
//! based on patterns in the event stream.

use crate::{GraphResult, GraphError};
use crate::structures::{
    Graph, GraphNode, GraphEdge, NodeType, EdgeType, NodeId, EdgeWeight,
    ConceptType, GoalStatus, InteractionType, GoalRelationType,
};
use agent_db_core::types::{AgentId, EventId, Timestamp, ContextHash};
use agent_db_events::{Event, EventType, ActionOutcome, EventContext};
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
        
        // Ensure agent node exists
        let agent_node_id = self.ensure_agent_node(event.agent_id, &event)?;
        created_nodes.push(agent_node_id);
        
        // Create event node
        let event_node_id = self.create_event_node(&event)?;
        created_nodes.push(event_node_id);
        
        // Ensure context node exists
        let context_node_id = self.ensure_context_node(&event.context)?;
        if context_node_id != event_node_id {
            created_nodes.push(context_node_id);
        }
        
        // Create basic relationships
        self.create_agent_event_relationship(agent_node_id, event_node_id, &event)?;
        self.create_event_context_relationship(event_node_id, context_node_id, &event)?;
        
        // Add to temporal buffer for causal analysis
        self.temporal_buffer.push_back(event.clone());
        if self.temporal_buffer.len() > self.config.batch_size {
            self.temporal_buffer.pop_front();
        }
        
        // Perform causal inference
        self.infer_causal_relationships(&event)?;
        
        // Infer contextual relationships
        self.infer_contextual_relationships(&event)?;
        
        // Update statistics
        self.stats.events_processed += 1;
        if let Ok(duration) = start_time.elapsed() {
            self.stats.processing_time_ms += duration.as_millis() as u64;
        }
        
        // Periodically clean up buffer and detect patterns
        if self.stats.events_processed % 100 == 0 {
            self.detect_patterns()?;
            self.cleanup_old_associations();
        }
        
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
                }
                Err(e) => {
                    results.errors.push(e);
                }
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
            EventType::Observation { observation_type, .. } => observation_type.clone(),
            EventType::Communication { message_type, .. } => message_type.clone(),
            EventType::Cognitive { process_type, .. } => format!("{:?}", process_type),
        };
        
        let significance = self.calculate_event_significance(event);
        
        let node = GraphNode::new(NodeType::Event {
            event_id: event.id,
            event_type: event_type_name,
            significance,
        });
        
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
                if let NodeType::Context { ref mut frequency, .. } = &mut mut_node.node_type {
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
    
    /// Infer causal relationships from temporal patterns
    fn infer_causal_relationships(&mut self, current_event: &Event) -> GraphResult<()> {
        let current_time = current_event.timestamp;
        
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
                    self.graph.get_event_node(current_event.id)
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
        
        Ok(())
    }
    
    /// Infer contextual relationships
    fn infer_contextual_relationships(&mut self, current_event: &Event) -> GraphResult<()> {
        // Collect context pairs first to avoid borrowing issues
        let mut context_pairs = Vec::new();
        for previous_event in self.temporal_buffer.iter() {
            if previous_event.id == current_event.id {
                continue;
            }
            context_pairs.push((
                previous_event.context.clone(),
                previous_event.id,
            ));
        }
        
        // Now calculate similarities
        for (prev_context, prev_event_id) in context_pairs {
            let similarity = self.calculate_context_similarity(
                &prev_context,
                &current_event.context,
            );
            
            if similarity > self.config.context_similarity_threshold {
                // Create contextual association
                if let (Some(prev_node), Some(curr_node)) = (
                    self.graph.get_event_node(prev_event_id),
                    self.graph.get_event_node(current_event.id)
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
        if self.temporal_buffer.len() < 3 {
            return Ok(()); // Need at least 3 events for pattern
        }
        
        // Look for repeating sequences of event types
        let events: Vec<_> = self.temporal_buffer.iter().collect();
        let sequence_length = 3; // Start with 3-event patterns
        
        for window in events.windows(sequence_length) {
            let event_types: Vec<String> = window.iter()
                .map(|e| match &e.event_type {
                    EventType::Action { action_name, .. } => action_name.clone(),
                    EventType::Observation { observation_type, .. } => observation_type.clone(),
                    EventType::Communication { message_type, .. } => message_type.clone(),
                    EventType::Cognitive { process_type, .. } => format!("{:?}", process_type),
                })
                .collect();
            
            // Calculate average interval
            let mut total_interval = 0u64;
            for i in 1..window.len() {
                total_interval += window[i].timestamp.saturating_sub(window[i-1].timestamp);
            }
            let avg_interval = total_interval / (window.len() - 1) as u64;
            
            // Check if this pattern already exists or create new one
            let pattern_name = event_types.join("->");
            if let Some(existing) = self.temporal_patterns.iter_mut()
                .find(|p| p.pattern_name == pattern_name) {
                existing.occurrence_count += 1;
                existing.confidence = (existing.confidence * 0.9 + 0.1).min(1.0);
            } else {
                self.temporal_patterns.push(TemporalPattern {
                    pattern_name,
                    event_sequence: event_types,
                    average_interval: avg_interval,
                    confidence: 0.3,
                    occurrence_count: 1,
                });
            }
        }
        
        Ok(())
    }
    
    /// Detect contextual associations
    fn detect_contextual_associations(&mut self) -> GraphResult<()> {
        // Group events by similar contexts
        let mut context_groups: HashMap<ContextHash, Vec<&Event>> = HashMap::new();
        
        for event in &self.temporal_buffer {
            context_groups.entry(event.context.fingerprint)
                .or_insert_with(Vec::new)
                .push(event);
        }
        
        // Look for associations between contexts
        for (context_hash, events) in context_groups {
            if events.len() >= self.config.min_co_occurrence_count as usize {
                let association_name = format!("context_{}", context_hash);
                
                let entities: Vec<EntityReference> = events.iter()
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
        
        self.contextual_associations.retain(|assoc| assoc.last_observed > cutoff_time);
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
            EventType::Action { outcome, .. } => {
                match outcome {
                    ActionOutcome::Success { .. } => 1.0,
                    ActionOutcome::Partial { .. } => 0.7,
                    ActionOutcome::Failure { .. } => 0.0,
                }
            }
            _ => 0.8, // Default success rate for non-actions
        }
    }
    
    /// Calculate causal strength between two events
    fn calculate_causal_strength(&mut self, prev_event: &Event, curr_event: &Event) -> f32 {
        let mut strength = 0.0;
        
        // Same agent increases causal likelihood
        if prev_event.agent_id == curr_event.agent_id {
            strength += 0.4;
        }
        
        // Context similarity increases causal likelihood
        let context_sim = self.calculate_context_similarity(&prev_event.context, &curr_event.context);
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
            let goal_similarity = self.calculate_goal_similarity(&ctx1.active_goals, &ctx2.active_goals);
            similarity += goal_similarity * 0.4;
            
            // Resource similarity 
            let resource_similarity = self.calculate_resource_similarity(&ctx1.resources, &ctx2.resources);
            similarity += resource_similarity * 0.3;
            
            // Environment similarity
            similarity += 0.3; // Simplified - would compare environment variables
        }
        
        // Cache result
        self.context_similarity_cache.insert(cache_key, similarity);
        
        similarity
    }
    
    /// Calculate goal similarity
    fn calculate_goal_similarity(&self, goals1: &[agent_db_events::Goal], goals2: &[agent_db_events::Goal]) -> f32 {
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
    fn calculate_resource_similarity(&self, res1: &agent_db_events::ResourceState, res2: &agent_db_events::ResourceState) -> f32 {
        // Simplified resource similarity based on computational resources
        let cpu_diff = (res1.computational.cpu_percent - res2.computational.cpu_percent).abs();
        let cpu_similarity = 1.0 - (cpu_diff / 100.0).min(1.0);
        
        let memory_ratio = res1.computational.memory_bytes.min(res2.computational.memory_bytes) as f32 /
                          res1.computational.memory_bytes.max(res2.computational.memory_bytes) as f32;
        
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