//! Scoped inference engine using agent_type + session_id
//!
//! Provides more intelligent relationship inference by considering
//! agent types and session boundaries for contextual grouping.

use crate::{GraphResult, GraphError};
use crate::structures::{Graph, GraphNode, GraphEdge, NodeType, EdgeType, NodeId};
use crate::inference::{GraphInference, InferenceConfig};
use crate::event_ordering::{EventOrderingEngine, OrderingConfig};
use agent_db_core::types::{AgentId, AgentType, SessionId, EventId, Timestamp, current_timestamp};
use agent_db_events::Event;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Scoped inference engine that groups agents by type and session
pub struct ScopedInferenceEngine {
    /// Per-scope inference engines
    scoped_engines: Arc<RwLock<HashMap<InferenceScope, Arc<RwLock<GraphInference>>>>>,
    
    /// Cross-scope pattern detector
    cross_scope_patterns: Arc<RwLock<CrossScopePatterns>>,
    
    /// Event ordering per scope
    scope_ordering: Arc<RwLock<HashMap<InferenceScope, Arc<EventOrderingEngine>>>>,
    
    /// Configuration
    config: ScopedInferenceConfig,
}

/// Scope for inference - combination of agent type and session
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct InferenceScope {
    /// Agent type (e.g., "coding-assistant", "data-analyst")
    pub agent_type: AgentType,
    
    /// Session identifier
    pub session_id: SessionId,
}

/// Configuration for scoped inference
#[derive(Debug, Clone)]
pub struct ScopedInferenceConfig {
    /// Base inference configuration
    pub inference_config: InferenceConfig,
    
    /// Event ordering configuration
    pub ordering_config: OrderingConfig,
    
    /// Enable cross-scope pattern detection
    pub enable_cross_scope_patterns: bool,
    
    /// Maximum scopes to maintain simultaneously
    pub max_scopes: usize,
    
    /// Enable agent type clustering
    pub enable_agent_type_clustering: bool,
    
    /// Session timeout (auto-cleanup inactive sessions)
    pub session_timeout_hours: u64,
}

impl Default for ScopedInferenceConfig {
    fn default() -> Self {
        Self {
            inference_config: InferenceConfig::default(),
            ordering_config: OrderingConfig::default(),
            enable_cross_scope_patterns: true,
            max_scopes: 1000,
            enable_agent_type_clustering: true,
            session_timeout_hours: 24,
        }
    }
}

/// Enhanced event with agent type information
#[derive(Debug, Clone)]
pub struct ScopedEvent {
    /// Original event
    pub event: Event,
    
    /// Agent type for this event
    pub agent_type: AgentType,
    
    /// Processing priority (higher = process first)
    pub priority: f32,
    
    /// Additional scope metadata
    pub scope_metadata: ScopeMetadata,
}

/// Metadata about the scope context
#[derive(Debug, Clone)]
pub struct ScopeMetadata {
    /// Project or workspace identifier
    pub workspace_id: Option<String>,
    
    /// User identifier (for user-scoped sessions)
    pub user_id: Option<String>,
    
    /// Environment (dev, staging, prod)
    pub environment: Option<String>,
    
    /// Custom tags for grouping
    pub tags: Vec<String>,
}

impl ScopedInferenceEngine {
    /// Create a new scoped inference engine
    pub async fn new(config: ScopedInferenceConfig) -> GraphResult<Self> {
        Ok(Self {
            scoped_engines: Arc::new(RwLock::new(HashMap::new())),
            cross_scope_patterns: Arc::new(RwLock::new(CrossScopePatterns::new())),
            scope_ordering: Arc::new(RwLock::new(HashMap::new())),
            config,
        })
    }
    
    /// Process a scoped event
    pub async fn process_scoped_event(&self, scoped_event: ScopedEvent) -> GraphResult<ScopedInferenceResult> {
        let scope = InferenceScope {
            agent_type: scoped_event.agent_type.clone(),
            session_id: scoped_event.event.session_id,
        };
        
        // Get or create inference engine for this scope
        let inference_engine = self.get_or_create_scope_engine(&scope).await?;
        
        // Get or create event ordering for this scope
        let ordering_engine = self.get_or_create_scope_ordering(&scope).await;
        
        // Process event through ordering first
        let ordering_result = ordering_engine.process_event(scoped_event.event.clone()).await?;
        
        let mut result = ScopedInferenceResult {
            scope: scope.clone(),
            nodes_created: Vec::new(),
            relationships_discovered: 0,
            patterns_detected: Vec::new(),
            cross_scope_patterns: Vec::new(),
            processing_time_ms: 0,
        };
        
        let start_time = std::time::Instant::now();
        
        // Process all ready events in correct order
        for ready_event in ordering_result.ready_events {
            let scope_result = {
                let mut engine = inference_engine.write().await;
                engine.process_event(ready_event.clone())?
            };
            
            result.nodes_created.extend(scope_result);
            result.relationships_discovered += 1;
            
            // Extract patterns for this scope
            let scope_patterns = {
                let engine = inference_engine.read().await;
                engine.get_temporal_patterns()
                    .iter()
                    .map(|p| p.pattern_name.clone())
                    .collect::<Vec<_>>()
            };
            result.patterns_detected = scope_patterns;
        }
        
        // Check for cross-scope patterns if enabled
        if self.config.enable_cross_scope_patterns {
            let cross_patterns = self.detect_cross_scope_patterns(&scope, &scoped_event).await;
            result.cross_scope_patterns = cross_patterns;
        }
        
        result.processing_time_ms = start_time.elapsed().as_millis() as u64;
        
        Ok(result)
    }
    
    /// Query within a specific scope
    pub async fn query_scope(&self, scope: &InferenceScope, query: ScopeQuery) -> GraphResult<ScopeQueryResult> {
        let inference_engine = {
            let engines = self.scoped_engines.read().await;
            engines.get(scope)
                .ok_or_else(|| GraphError::OperationError(format!("Scope not found: {:?}", scope)))?
                .clone()
        };
        
        match query {
            ScopeQuery::AgentCollaboration => {
                // Find collaboration patterns within this scope
                let collaboration_patterns = self.find_agent_collaboration_patterns(&inference_engine).await?;
                Ok(ScopeQueryResult::CollaborationPatterns(collaboration_patterns))
            }
            
            ScopeQuery::WorkflowSequences => {
                // Find common workflow sequences in this scope
                let workflows = self.find_workflow_sequences(&inference_engine).await?;
                Ok(ScopeQueryResult::WorkflowSequences(workflows))
            }
            
            ScopeQuery::AgentTypes => {
                // Get all agent types in this scope
                let agent_types = self.get_scope_agent_types(&inference_engine).await?;
                Ok(ScopeQueryResult::AgentTypes(agent_types))
            }
            
            ScopeQuery::SessionMetrics => {
                // Get metrics for this session
                let metrics = self.calculate_session_metrics(&inference_engine).await?;
                Ok(ScopeQueryResult::SessionMetrics(metrics))
            }
        }
    }
    
    /// Get cross-scope insights (patterns that appear across different scopes)
    pub async fn get_cross_scope_insights(&self) -> CrossScopeInsights {
        let patterns = self.cross_scope_patterns.read().await;
        
        CrossScopeInsights {
            common_agent_patterns: patterns.get_common_patterns_by_agent_type(),
            session_similarities: patterns.get_similar_sessions(),
            global_workflows: patterns.get_global_workflow_patterns(),
        }
    }
    
    /// Get statistics for all scopes
    pub async fn get_scope_statistics(&self) -> ScopeStatistics {
        let engines = self.scoped_engines.read().await;
        
        let mut stats_by_agent_type: HashMap<AgentType, AgentTypeStats> = HashMap::new();
        let mut total_sessions = 0;
        let mut total_agents = 0;
        
        for (scope, engine) in engines.iter() {
            total_sessions += 1;
            
            let graph_stats = {
                let inference = engine.read().await;
                inference.graph().stats().clone()
            };
            
            let agent_type_stat = stats_by_agent_type
                .entry(scope.agent_type.clone())
                .or_insert_with(|| AgentTypeStats::new(scope.agent_type.clone()));
            
            agent_type_stat.session_count += 1;
            agent_type_stat.total_events += graph_stats.edge_count;
            agent_type_stat.total_relationships += graph_stats.edge_count;
            
            // Count unique agents in this scope
            let agent_nodes = {
                let inference = engine.read().await;
                inference.graph().get_nodes_by_type("Agent").len()
            };
            total_agents += agent_nodes;
            agent_type_stat.unique_agents += agent_nodes;
        }
        
        let most_active_agent_type = stats_by_agent_type
            .iter()
            .max_by_key(|(_, stats)| stats.total_events)
            .map(|(agent_type, _)| agent_type.clone());

        ScopeStatistics {
            total_scopes: engines.len(),
            total_sessions,
            total_agents,
            stats_by_agent_type,
            most_active_agent_type,
        }
    }
    
    /// Clean up inactive scopes
    pub async fn cleanup_inactive_scopes(&self) -> GraphResult<CleanupReport> {
        let mut report = CleanupReport {
            scopes_removed: 0,
            memory_freed_bytes: 0,
        };
        
        let cutoff_time = current_timestamp() - (self.config.session_timeout_hours * 3600 * 1_000_000_000);
        let mut to_remove = Vec::new();
        
        {
            let engines = self.scoped_engines.read().await;
            for (scope, engine) in engines.iter() {
                let last_activity = {
                    let inference = engine.read().await;
                    inference.graph().stats().last_updated
                };
                
                if last_activity < cutoff_time {
                    to_remove.push(scope.clone());
                }
            }
        }
        
        // Remove inactive scopes
        if !to_remove.is_empty() {
            let mut engines = self.scoped_engines.write().await;
            let mut ordering = self.scope_ordering.write().await;
            
            for scope in to_remove {
                engines.remove(&scope);
                ordering.remove(&scope);
                report.scopes_removed += 1;
                report.memory_freed_bytes += 1024 * 1024; // Estimate 1MB per scope
            }
        }
        
        Ok(report)
    }
    
    /// Get or create inference engine for a scope
    async fn get_or_create_scope_engine(&self, scope: &InferenceScope) -> GraphResult<Arc<RwLock<GraphInference>>> {
        {
            let engines = self.scoped_engines.read().await;
            if let Some(engine) = engines.get(scope) {
                return Ok(engine.clone());
            }
        }
        
        // Create new scope engine
        let mut engines = self.scoped_engines.write().await;
        
        // Check again after acquiring write lock (double-checked locking)
        if let Some(engine) = engines.get(scope) {
            return Ok(engine.clone());
        }
        
        // Check scope limits
        if engines.len() >= self.config.max_scopes {
            return Err(GraphError::OperationError("Maximum scopes reached".to_string()));
        }
        
        let new_engine = Arc::new(RwLock::new(
            GraphInference::with_config(self.config.inference_config.clone())
        ));
        
        engines.insert(scope.clone(), new_engine.clone());
        Ok(new_engine)
    }
    
    /// Get or create event ordering for a scope
    async fn get_or_create_scope_ordering(&self, scope: &InferenceScope) -> Arc<EventOrderingEngine> {
        {
            let ordering = self.scope_ordering.read().await;
            if let Some(engine) = ordering.get(scope) {
                return engine.clone();
            }
        }
        
        let mut ordering = self.scope_ordering.write().await;
        
        // Double-checked locking
        if let Some(engine) = ordering.get(scope) {
            return engine.clone();
        }
        
        let new_engine = Arc::new(EventOrderingEngine::new(self.config.ordering_config.clone()));
        ordering.insert(scope.clone(), new_engine.clone());
        new_engine
    }
    
    /// Detect patterns that span across different scopes
    async fn detect_cross_scope_patterns(&self, current_scope: &InferenceScope, event: &ScopedEvent) -> Vec<CrossScopePattern> {
        let mut patterns = Vec::new();
        
        // Find similar agent types in different sessions
        if self.config.enable_agent_type_clustering {
            let similar_scopes = self.find_similar_scopes(current_scope).await;
            
            for similar_scope in similar_scopes {
                if let Some(pattern) = self.compare_scope_patterns(current_scope, &similar_scope).await {
                    patterns.push(pattern);
                }
            }
        }
        
        patterns
    }
    
    /// Find scopes with similar agent types
    async fn find_similar_scopes(&self, target_scope: &InferenceScope) -> Vec<InferenceScope> {
        let engines = self.scoped_engines.read().await;
        
        engines.keys()
            .filter(|scope| {
                // Same agent type but different session
                scope.agent_type == target_scope.agent_type && scope.session_id != target_scope.session_id
            })
            .cloned()
            .collect()
    }
    
    /// Compare patterns between two scopes
    async fn compare_scope_patterns(&self, scope1: &InferenceScope, scope2: &InferenceScope) -> Option<CrossScopePattern> {
        // Implementation would compare patterns between scopes
        // For now, return a placeholder
        Some(CrossScopePattern {
            pattern_name: format!("common_{}_{}", scope1.agent_type, scope2.agent_type),
            scopes: vec![scope1.clone(), scope2.clone()],
            similarity: 0.75,
            frequency: 1,
        })
    }
    
    /// Find agent collaboration patterns within a scope
    async fn find_agent_collaboration_patterns(&self, engine: &Arc<RwLock<GraphInference>>) -> GraphResult<Vec<CollaborationPattern>> {
        let inference = engine.read().await;
        let graph = inference.graph();
        
        // Find agent nodes and their interactions
        let agent_nodes = graph.get_nodes_by_type("Agent");
        let mut patterns = Vec::new();
        
        for agent1 in &agent_nodes {
            for agent2 in &agent_nodes {
                if agent1.id != agent2.id {
                    // Check if there's a path between these agents
                    if let Some(path) = graph.shortest_path(agent1.id, agent2.id) {
                        patterns.push(CollaborationPattern {
                            agents: vec![agent1.id, agent2.id],
                            interaction_count: 1,
                            collaboration_strength: 0.5,
                            common_tasks: vec!["task_collaboration".to_string()],
                        });
                    }
                }
            }
        }
        
        Ok(patterns)
    }
    
    /// Find workflow sequences within a scope
    async fn find_workflow_sequences(&self, engine: &Arc<RwLock<GraphInference>>) -> GraphResult<Vec<WorkflowSequence>> {
        let inference = engine.read().await;
        let patterns = inference.get_temporal_patterns();
        
        Ok(patterns.iter().map(|p| WorkflowSequence {
            name: p.pattern_name.clone(),
            steps: vec![], // Would extract actual steps from pattern
            frequency: 1,
            average_duration_ms: 1000,
            success_rate: 0.9,
        }).collect())
    }
    
    /// Get agent types in a scope
    async fn get_scope_agent_types(&self, engine: &Arc<RwLock<GraphInference>>) -> GraphResult<Vec<ScopeAgentType>> {
        let inference = engine.read().await;
        let graph = inference.graph();
        let agent_nodes = graph.get_nodes_by_type("Agent");
        
        let mut agent_types = HashMap::new();
        for node in agent_nodes {
            if let NodeType::Agent { agent_type, capabilities, .. } = &node.node_type {
                let scope_agent_type = agent_types
                    .entry(agent_type.clone())
                    .or_insert_with(|| ScopeAgentType {
                        agent_type: agent_type.clone(),
                        count: 0,
                        capabilities: capabilities.clone(),
                        average_activity: 0.0,
                    });
                scope_agent_type.count += 1;
            }
        }
        
        Ok(agent_types.into_values().collect())
    }
    
    /// Calculate session metrics
    async fn calculate_session_metrics(&self, engine: &Arc<RwLock<GraphInference>>) -> GraphResult<SessionMetrics> {
        let inference = engine.read().await;
        let stats = inference.graph().stats();
        
        Ok(SessionMetrics {
            total_events: stats.edge_count,
            unique_agents: stats.node_count,
            collaboration_index: stats.avg_degree,
            session_duration_hours: 1.0, // Would calculate from timestamps
            productivity_score: 0.8,
        })
    }
}

// Supporting types

#[derive(Debug)]
pub struct ScopedInferenceResult {
    pub scope: InferenceScope,
    pub nodes_created: Vec<NodeId>,
    pub relationships_discovered: u64,
    pub patterns_detected: Vec<String>,
    pub cross_scope_patterns: Vec<CrossScopePattern>,
    pub processing_time_ms: u64,
}

#[derive(Debug)]
pub enum ScopeQuery {
    AgentCollaboration,
    WorkflowSequences,
    AgentTypes,
    SessionMetrics,
}

#[derive(Debug)]
pub enum ScopeQueryResult {
    CollaborationPatterns(Vec<CollaborationPattern>),
    WorkflowSequences(Vec<WorkflowSequence>),
    AgentTypes(Vec<ScopeAgentType>),
    SessionMetrics(SessionMetrics),
}

#[derive(Debug)]
pub struct CollaborationPattern {
    pub agents: Vec<NodeId>,
    pub interaction_count: u64,
    pub collaboration_strength: f32,
    pub common_tasks: Vec<String>,
}

#[derive(Debug)]
pub struct WorkflowSequence {
    pub name: String,
    pub steps: Vec<String>,
    pub frequency: u64,
    pub average_duration_ms: u64,
    pub success_rate: f32,
}

#[derive(Debug)]
pub struct ScopeAgentType {
    pub agent_type: String,
    pub count: u64,
    pub capabilities: Vec<String>,
    pub average_activity: f32,
}

#[derive(Debug)]
pub struct SessionMetrics {
    pub total_events: usize,
    pub unique_agents: usize,
    pub collaboration_index: f32,
    pub session_duration_hours: f64,
    pub productivity_score: f32,
}

#[derive(Debug)]
pub struct CrossScopePattern {
    pub pattern_name: String,
    pub scopes: Vec<InferenceScope>,
    pub similarity: f32,
    pub frequency: u64,
}

#[derive(Debug)]
struct CrossScopePatterns {
    patterns: HashMap<String, Vec<CrossScopePattern>>,
}

impl CrossScopePatterns {
    fn new() -> Self {
        Self {
            patterns: HashMap::new(),
        }
    }
    
    fn get_common_patterns_by_agent_type(&self) -> HashMap<AgentType, Vec<String>> {
        HashMap::new() // Placeholder
    }
    
    fn get_similar_sessions(&self) -> Vec<(SessionId, SessionId, f32)> {
        Vec::new() // Placeholder
    }
    
    fn get_global_workflow_patterns(&self) -> Vec<String> {
        Vec::new() // Placeholder
    }
}

#[derive(Debug)]
pub struct CrossScopeInsights {
    pub common_agent_patterns: HashMap<AgentType, Vec<String>>,
    pub session_similarities: Vec<(SessionId, SessionId, f32)>,
    pub global_workflows: Vec<String>,
}

#[derive(Debug)]
pub struct ScopeStatistics {
    pub total_scopes: usize,
    pub total_sessions: usize,
    pub total_agents: usize,
    pub stats_by_agent_type: HashMap<AgentType, AgentTypeStats>,
    pub most_active_agent_type: Option<AgentType>,
}

#[derive(Debug)]
pub struct AgentTypeStats {
    pub agent_type: AgentType,
    pub session_count: usize,
    pub unique_agents: usize,
    pub total_events: usize,
    pub total_relationships: usize,
}

impl AgentTypeStats {
    fn new(agent_type: AgentType) -> Self {
        Self {
            agent_type,
            session_count: 0,
            unique_agents: 0,
            total_events: 0,
            total_relationships: 0,
        }
    }
}

#[derive(Debug)]
pub struct CleanupReport {
    pub scopes_removed: usize,
    pub memory_freed_bytes: u64,
}