//! Integration layer for graph operations
//!
//! This module provides a unified interface that integrates graph structures,
//! inference, and traversal capabilities with the storage layer.
//!
//! **NEW:** Full self-evolution pipeline integration:
//! - Automatic episode detection from event streams
//! - Memory formation from significant episodes
//! - Strategy extraction from successful experiences
//! - Reinforcement learning from outcomes
//! - Policy guide queries for action suggestions

use crate::{GraphResult, GraphError};
use crate::structures::{Graph, GraphNode, GraphEdge, NodeId, GraphStats};
use crate::inference::{GraphInference, InferenceConfig, InferenceStats, InferenceResults, EpisodeMetrics};
use crate::traversal::{GraphTraversal, GraphQuery, QueryResult, ActionSuggestion};
use crate::event_ordering::{EventOrderingEngine, OrderingConfig, OrderingResult};
use crate::episodes::{EpisodeDetector, EpisodeDetectorConfig, Episode, EpisodeId, EpisodeOutcome};
use crate::memory::{MemoryFormation, MemoryFormationConfig, Memory, MemoryId, MemoryStats};
use crate::strategies::{StrategyExtractor, StrategyExtractionConfig, Strategy, StrategyId, StrategyStats};
use agent_db_events::Event;
use agent_db_storage::{StorageEngine, StorageResult};
use agent_db_core::types::{AgentId, ContextHash};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// Configuration for the integrated graph engine
#[derive(Debug, Clone)]
pub struct GraphEngineConfig {
    /// Configuration for inference algorithms
    pub inference_config: InferenceConfig,

    /// Configuration for event ordering
    pub ordering_config: OrderingConfig,

    /// Configuration for scoped inference
    pub scoped_inference_config: crate::scoped_inference::ScopedInferenceConfig,

    /// Configuration for episode detection
    pub episode_config: EpisodeDetectorConfig,

    /// Configuration for memory formation
    pub memory_config: MemoryFormationConfig,

    /// Configuration for strategy extraction
    pub strategy_config: StrategyExtractionConfig,

    /// Enable automatic pattern detection
    pub auto_pattern_detection: bool,

    /// Enable automatic episode detection
    pub auto_episode_detection: bool,

    /// Enable automatic memory formation
    pub auto_memory_formation: bool,

    /// Enable automatic strategy extraction
    pub auto_strategy_extraction: bool,

    /// Enable automatic reinforcement learning
    pub auto_reinforcement_learning: bool,

    /// Batch size for processing events
    pub batch_size: usize,

    /// Enable graph persistence to storage
    pub enable_persistence: bool,

    /// Interval for persisting graph state (in events processed)
    pub persistence_interval: u64,

    /// Maximum graph size before cleanup
    pub max_graph_size: usize,

    /// Enable query caching
    pub enable_query_cache: bool,
}

/// Results from graph operations
#[derive(Debug)]
pub struct GraphOperationResult {
    pub nodes_created: Vec<NodeId>,
    pub relationships_discovered: u64,
    pub patterns_detected: Vec<String>,
    pub processing_time_ms: u64,
    pub errors: Vec<GraphError>,
}

/// Comprehensive graph engine that integrates all graph capabilities
///
/// **Self-Evolution Pipeline:**
/// 1. Events → Episode Detection → Episodes
/// 2. Episodes → Memory Formation → Memories
/// 3. Episodes → Strategy Extraction → Strategies
/// 4. Outcomes → Reinforcement Learning → Pattern Updates
/// 5. Context → Policy Guide → Action Suggestions
pub struct GraphEngine {
    /// Core inference engine
    inference: Arc<RwLock<GraphInference>>,

    /// Graph traversal engine
    traversal: Arc<RwLock<GraphTraversal>>,

    /// Event ordering engine for handling concurrent events
    event_ordering: Arc<EventOrderingEngine>,

    /// Scoped inference engine
    scoped_inference: Arc<crate::scoped_inference::ScopedInferenceEngine>,

    /// Episode detector - automatically detects episode boundaries
    episode_detector: Arc<RwLock<EpisodeDetector>>,

    /// Memory formation - creates memories from significant episodes
    memory_formation: Arc<RwLock<MemoryFormation>>,

    /// Strategy extractor - extracts reusable strategies from successful episodes
    strategy_extractor: Arc<RwLock<StrategyExtractor>>,

    /// Event storage for episode processing
    event_store: Arc<RwLock<HashMap<agent_db_core::types::EventId, Event>>>,

    /// Optional storage engine for persistence
    storage: Option<Arc<StorageEngine>>,

    /// Configuration
    config: GraphEngineConfig,

    /// Operation statistics
    stats: Arc<RwLock<GraphEngineStats>>,

    /// Event processing buffer
    event_buffer: Arc<RwLock<Vec<Event>>>,

    /// Last persistence checkpoint
    last_persistence: Arc<RwLock<u64>>,
}

/// Statistics for the graph engine
#[derive(Debug)]
pub struct GraphEngineStats {
    pub total_events_processed: u64,
    pub total_nodes_created: u64,
    pub total_relationships_created: u64,
    pub total_patterns_detected: u64,
    pub total_queries_executed: u64,
    pub average_processing_time_ms: f64,
    pub cache_hit_rate: f32,
    pub last_operation_time: std::time::Instant,

    // Self-evolution stats
    pub total_episodes_detected: u64,
    pub total_memories_formed: u64,
    pub total_strategies_extracted: u64,
    pub total_reinforcements_applied: u64,
}

impl Default for GraphEngineConfig {
    fn default() -> Self {
        Self {
            inference_config: InferenceConfig::default(),
            ordering_config: OrderingConfig::default(),
            scoped_inference_config: crate::scoped_inference::ScopedInferenceConfig::default(),
            episode_config: EpisodeDetectorConfig::default(),
            memory_config: MemoryFormationConfig::default(),
            strategy_config: StrategyExtractionConfig::default(),
            auto_pattern_detection: true,
            auto_episode_detection: true,
            auto_memory_formation: true,
            auto_strategy_extraction: true,
            auto_reinforcement_learning: true,
            batch_size: 100,
            enable_persistence: true,
            persistence_interval: 1000,
            max_graph_size: 1_000_000,
            enable_query_cache: true,
        }
    }
}

impl GraphEngine {
    /// Create a new graph engine with default configuration
    pub async fn new() -> GraphResult<Self> {
        Self::with_config(GraphEngineConfig::default()).await
    }
    
    /// Create a graph engine with custom configuration
    pub async fn with_config(config: GraphEngineConfig) -> GraphResult<Self> {
        let inference = Arc::new(RwLock::new(
            GraphInference::with_config(config.inference_config.clone())
        ));

        let traversal = Arc::new(RwLock::new(GraphTraversal::new()));

        let event_ordering = Arc::new(EventOrderingEngine::new(config.ordering_config.clone()));

        let scoped_inference = Arc::new(
            crate::scoped_inference::ScopedInferenceEngine::new(config.scoped_inference_config.clone()).await?
        );

        // Initialize self-evolution components
        // Note: EpisodeDetector requires an Arc<Graph> but doesn't actually use it for detection
        // It uses event-based heuristics instead. We provide an empty graph for API compatibility.
        let graph_for_episodes = Arc::new(Graph::new());
        let episode_detector = Arc::new(RwLock::new(
            EpisodeDetector::new(graph_for_episodes, config.episode_config.clone())
        ));

        let memory_formation = Arc::new(RwLock::new(
            MemoryFormation::new(config.memory_config.clone())
        ));

        let strategy_extractor = Arc::new(RwLock::new(
            StrategyExtractor::new(config.strategy_config.clone())
        ));

        Ok(Self {
            inference,
            traversal,
            event_ordering,
            scoped_inference,
            episode_detector,
            memory_formation,
            strategy_extractor,
            event_store: Arc::new(RwLock::new(HashMap::new())),
            storage: None,
            config,
            stats: Arc::new(RwLock::new(GraphEngineStats::default())),
            event_buffer: Arc::new(RwLock::new(Vec::new())),
            last_persistence: Arc::new(RwLock::new(0)),
        })
    }
    
    /// Create a graph engine with storage integration
    pub async fn with_storage(
        config: GraphEngineConfig,
        storage: Arc<StorageEngine>
    ) -> GraphResult<Self> {
        let mut engine = Self::with_config(config).await?;
        engine.storage = Some(storage);
        Ok(engine)
    }
    
    /// Process a single event and update the graph
    ///
    /// **Automatic Self-Evolution Pipeline:**
    /// 1. Event ordering and graph construction
    /// 2. Episode detection from event stream
    /// 3. Memory formation from significant episodes
    /// 4. Strategy extraction from successful episodes
    /// 5. Reinforcement learning from outcomes
    pub async fn process_event(&self, event: Event) -> GraphResult<GraphOperationResult> {
        let start_time = std::time::Instant::now();
        let mut result = GraphOperationResult {
            nodes_created: Vec::new(),
            relationships_discovered: 0,
            patterns_detected: Vec::new(),
            processing_time_ms: 0,
            errors: Vec::new(),
        };

        // Store event for episode processing
        {
            let mut store = self.event_store.write().await;
            store.insert(event.id, event.clone());
        }

        // Step 1: Order the event (handles out-of-order arrival)
        let ordering_result = self.event_ordering.process_event(event.clone()).await?;

        // Step 2: Process all ready events through graph construction
        for ready_event in ordering_result.ready_events {
            // Add to processing buffer
            {
                let mut buffer = self.event_buffer.write().await;
                buffer.push(ready_event.clone());
            }

            // Process through inference engine
            match self.inference.write().await.process_event(ready_event.clone()) {
                Ok(nodes) => {
                    result.nodes_created.extend(nodes);
                }
                Err(e) => {
                    result.errors.push(e);
                }
            }

            // Step 3: Self-Evolution Pipeline - Episode Detection
            if self.config.auto_episode_detection {
                if let Some(completed_episode_id) = self.episode_detector.write().await.process_event(&ready_event) {
                    result.patterns_detected.push(format!("episode_completed_{}", completed_episode_id));

                    // Get the completed episode (collect to owned Vec to avoid borrow issues)
                    let episodes: Vec<Episode> = self.episode_detector.read().await.get_completed_episodes().to_vec();
                    if let Some(episode) = episodes.iter().find(|e| e.id == completed_episode_id) {
                        // Update stats
                        self.stats.write().await.total_episodes_detected += 1;

                        // Step 4: Memory Formation
                        if self.config.auto_memory_formation {
                            self.process_episode_for_memory(episode).await?;
                        }

                        // Step 5: Strategy Extraction
                        if self.config.auto_strategy_extraction {
                            self.process_episode_for_strategy(episode).await?;
                        }

                        // Step 6: Reinforcement Learning
                        if self.config.auto_reinforcement_learning {
                            self.process_episode_for_reinforcement(episode).await?;
                        }
                    }
                }
            }
        }

        // Update ordering statistics
        if ordering_result.reordering_occurred {
            result.patterns_detected.push("event_reordering_occurred".to_string());
        }

        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.total_events_processed += 1;
            stats.total_nodes_created += result.nodes_created.len() as u64;
            stats.last_operation_time = std::time::Instant::now();

            // Update average processing time
            let processing_time = start_time.elapsed().as_millis() as f64;
            stats.average_processing_time_ms =
                (stats.average_processing_time_ms * (stats.total_events_processed as f64 - 1.0) + processing_time)
                / stats.total_events_processed as f64;
        }

        // Check if we need to process batch
        let should_process_batch = {
            let buffer = self.event_buffer.read().await;
            buffer.len() >= self.config.batch_size
        };

        if should_process_batch {
            self.process_batch().await?;
        }

        // Check if we need to persist
        if self.config.enable_persistence {
            let stats = self.stats.read().await;
            let last_persistence = *self.last_persistence.read().await;

            if stats.total_events_processed - last_persistence >= self.config.persistence_interval {
                drop(stats);
                self.persist_graph_state().await?;
            }
        }

        result.processing_time_ms = start_time.elapsed().as_millis() as u64;
        Ok(result)
    }
    
    /// Process multiple events in batch
    pub async fn process_events(&self, events: Vec<Event>) -> GraphResult<GraphOperationResult> {
        let start_time = std::time::Instant::now();
        let mut combined_result = GraphOperationResult {
            nodes_created: Vec::new(),
            relationships_discovered: 0,
            patterns_detected: Vec::new(),
            processing_time_ms: 0,
            errors: Vec::new(),
        };
        
        // Add all events to buffer
        {
            let mut buffer = self.event_buffer.write().await;
            buffer.extend(events.clone());
        }
        
        // Process through inference engine
        match self.inference.write().await.process_events(events) {
            Ok(inference_results) => {
                combined_result.nodes_created = inference_results.nodes_created;
                combined_result.patterns_detected = (0..inference_results.patterns_detected)
                    .map(|i| format!("pattern_{}", i))
                    .collect();
                combined_result.relationships_discovered = inference_results.events_processed;
            }
            Err(e) => {
                combined_result.errors.push(e);
            }
        }
        
        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.total_events_processed += combined_result.nodes_created.len() as u64;
            stats.total_nodes_created += combined_result.nodes_created.len() as u64;
            stats.total_patterns_detected += combined_result.patterns_detected.len() as u64;
            stats.last_operation_time = std::time::Instant::now();
        }
        
        combined_result.processing_time_ms = start_time.elapsed().as_millis() as u64;
        Ok(combined_result)
    }
    
    /// Execute a graph query
    pub async fn execute_query(&self, query: GraphQuery) -> GraphResult<QueryResult> {
        let start_time = std::time::Instant::now();
        
        // Get read access to the graph through inference engine
        let graph = {
            let inference = self.inference.read().await;
            // We need a way to get a reference to the graph
            // For now, we'll execute queries directly through traversal
        };
        
        let result = {
            let inference = self.inference.read().await;
            let mut traversal = self.traversal.write().await;
            traversal.execute_query(inference.graph(), query)?
        };
        
        // Update query statistics
        {
            let mut stats = self.stats.write().await;
            stats.total_queries_executed += 1;
            
            let query_time = start_time.elapsed().as_millis() as f64;
            // Would update query timing statistics here
        }
        
        Ok(result)
    }
    
    /// Get current graph statistics
    pub async fn get_graph_stats(&self) -> GraphStats {
        let inference = self.inference.read().await;
        inference.graph().stats().clone()
    }
    
    /// Get engine statistics
    pub async fn get_engine_stats(&self) -> GraphEngineStats {
        let stats = self.stats.read().await;
        GraphEngineStats {
            total_events_processed: stats.total_events_processed,
            total_nodes_created: stats.total_nodes_created,
            total_relationships_created: stats.total_relationships_created,
            total_patterns_detected: stats.total_patterns_detected,
            total_queries_executed: stats.total_queries_executed,
            average_processing_time_ms: stats.average_processing_time_ms,
            cache_hit_rate: stats.cache_hit_rate,
            last_operation_time: stats.last_operation_time,
            total_episodes_detected: stats.total_episodes_detected,
            total_memories_formed: stats.total_memories_formed,
            total_strategies_extracted: stats.total_strategies_extracted,
            total_reinforcements_applied: stats.total_reinforcements_applied,
        }
    }
    
    /// Get detected patterns
    pub async fn get_patterns(&self) -> Vec<String> {
        let inference = self.inference.read().await;
        inference.get_temporal_patterns()
            .iter()
            .map(|p| p.pattern_name.clone())
            .collect()
    }
    
    /// Force pattern detection on current data
    pub async fn detect_patterns(&self) -> GraphResult<Vec<String>> {
        // Process any buffered events first
        self.process_batch().await?;
        
        // Get patterns
        let patterns = self.get_patterns().await;
        
        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.total_patterns_detected += patterns.len() as u64;
        }
        
        Ok(patterns)
    }
    
    /// Get graph size and health metrics
    pub async fn get_health_metrics(&self) -> GraphHealthMetrics {
        let graph_stats = self.get_graph_stats().await;
        let engine_stats = self.get_engine_stats().await;
        
        GraphHealthMetrics {
            node_count: graph_stats.node_count,
            edge_count: graph_stats.edge_count,
            average_degree: graph_stats.avg_degree,
            largest_component_size: graph_stats.largest_component_size,
            events_processed: engine_stats.total_events_processed,
            processing_rate: if engine_stats.average_processing_time_ms > 0.0 {
                1000.0 / engine_stats.average_processing_time_ms
            } else {
                0.0
            },
            memory_usage_estimate: graph_stats.node_count * 1000 + graph_stats.edge_count * 500, // Rough estimate
            is_healthy: graph_stats.node_count < self.config.max_graph_size 
                && engine_stats.average_processing_time_ms < 100.0,
        }
    }
    
    /// Clean up old data and optimize graph
    pub async fn cleanup(&self) -> GraphResult<()> {
        // Clean up old associations
        // This would involve removing old temporal patterns and low-confidence relationships
        
        // Clean up query cache
        {
            let mut traversal = self.traversal.write().await;
            traversal.cleanup_cache();
        }
        
        Ok(())
    }
    
    /// Persist current graph state to storage
    async fn persist_graph_state(&self) -> GraphResult<()> {
        if let Some(ref storage) = self.storage {
            // Would serialize graph state and store it
            // For now, just update the persistence checkpoint
            let stats = self.stats.read().await;
            *self.last_persistence.write().await = stats.total_events_processed;
        }
        Ok(())
    }
    
    /// Process buffered events in batch
    async fn process_batch(&self) -> GraphResult<()> {
        let events = {
            let mut buffer = self.event_buffer.write().await;
            let events = buffer.clone();
            buffer.clear();
            events
        };
        
        if !events.is_empty() {
            let mut inference = self.inference.write().await;
            let _results = inference.process_events(events)?;
        }
        
        Ok(())
    }
    
    /// Flush all buffered events (useful for shutdown or testing)
    pub async fn flush_all_buffers(&self) -> GraphResult<()> {
        // Flush the event ordering buffers
        let buffered_events = self.event_ordering.flush_all_buffers().await?;
        
        // Process any remaining buffered events
        for event in buffered_events {
            let _ = self.process_event(event).await; // Ignore errors during shutdown
        }
        
        Ok(())
    }
    
    /// Get scoped inference statistics
    pub async fn get_scoped_inference_stats(&self) -> crate::scoped_inference::ScopeStatistics {
        self.scoped_inference.get_scope_statistics().await
    }

    /// Query events in a specific scope
    pub async fn query_events_in_scope(
        &self,
        scope: &crate::scoped_inference::InferenceScope,
        query: crate::scoped_inference::ScopeQuery
    ) -> GraphResult<crate::scoped_inference::ScopeQueryResult> {
        self.scoped_inference.query_scope(scope, query).await
    }

    /// Get cross-scope relationships
    pub async fn get_cross_scope_relationships(&self) -> crate::scoped_inference::CrossScopeInsights {
        self.scoped_inference.get_cross_scope_insights().await
    }

    // ============================================================================
    // Self-Evolution Pipeline Methods
    // ============================================================================

    /// Process episode for memory formation
    async fn process_episode_for_memory(&self, episode: &Episode) -> GraphResult<()> {
        let mut memory_formation = self.memory_formation.write().await;

        if let Some(memory_id) = memory_formation.form_memory(episode) {
            self.stats.write().await.total_memories_formed += 1;
            // Memory formed successfully
        }

        Ok(())
    }

    /// Process episode for strategy extraction
    async fn process_episode_for_strategy(&self, episode: &Episode) -> GraphResult<()> {
        // Only extract from successful episodes
        if episode.outcome != Some(EpisodeOutcome::Success) {
            return Ok(());
        }

        // Get events for this episode
        let events: Vec<Event> = {
            let store = self.event_store.read().await;
            episode.events.iter()
                .filter_map(|event_id| store.get(event_id).cloned())
                .collect()
        };

        if !events.is_empty() {
            let mut strategy_extractor = self.strategy_extractor.write().await;
            if let Some(_strategy_id) = strategy_extractor.extract_from_episode(episode, &events)? {
                self.stats.write().await.total_strategies_extracted += 1;
            }
        }

        Ok(())
    }

    /// Process episode for reinforcement learning
    async fn process_episode_for_reinforcement(&self, episode: &Episode) -> GraphResult<()> {
        // Determine success/failure
        let success = matches!(episode.outcome, Some(EpisodeOutcome::Success));

        // Calculate duration from events
        let duration_seconds = {
            let store = self.event_store.read().await;
            if let (Some(start_event), Some(end_event_id)) = (
                store.get(&episode.start_event),
                episode.end_event,
            ) {
                if let Some(end_event) = store.get(&end_event_id) {
                    let duration_ns = end_event.timestamp.saturating_sub(start_event.timestamp);
                    (duration_ns as f32) / 1_000_000_000.0
                } else {
                    1.0 // Default
                }
            } else {
                1.0 // Default
            }
        };

        // Calculate metrics
        let metrics = EpisodeMetrics {
            duration_seconds,
            expected_duration_seconds: 5.0, // Default expectation
            quality_score: Some(episode.significance),
            custom_metrics: HashMap::new(),
        };

        // Apply reinforcement
        let mut inference = self.inference.write().await;
        let _result = inference.reinforce_patterns(episode, success, Some(metrics)).await?;

        self.stats.write().await.total_reinforcements_applied += 1;

        Ok(())
    }

    // ============================================================================
    // Self-Evolution Query Methods
    // ============================================================================

    /// Get policy guide suggestions for what action to take next
    ///
    /// **Returns:** Action suggestions ranked by success probability
    pub async fn get_next_action_suggestions(
        &self,
        context_hash: ContextHash,
        last_action_node: Option<NodeId>,
        limit: usize,
    ) -> GraphResult<Vec<ActionSuggestion>> {
        let inference = self.inference.read().await;
        let graph = inference.graph();
        let traversal = self.traversal.read().await;

        traversal.get_next_step_suggestions(graph, context_hash, last_action_node, limit)
    }

    /// Get all memories for an agent
    pub async fn get_agent_memories(&self, agent_id: AgentId, limit: usize) -> Vec<Memory> {
        self.memory_formation.read().await
            .retrieve_by_agent(agent_id, limit)
            .into_iter()
            .cloned()
            .collect()
    }

    /// Retrieve memories by context similarity
    pub async fn retrieve_memories_by_context(
        &self,
        context: &agent_db_events::core::EventContext,
        limit: usize,
    ) -> Vec<Memory> {
        self.memory_formation.write().await.retrieve_by_context(context, limit)
    }

    /// Get all strategies for an agent
    pub async fn get_agent_strategies(&self, agent_id: AgentId, limit: usize) -> Vec<Strategy> {
        let extractor = self.strategy_extractor.read().await;
        extractor.get_agent_strategies(agent_id, limit)
            .into_iter()
            .cloned()
            .collect()
    }

    /// Get strategies applicable to a context
    pub async fn get_strategies_for_context(
        &self,
        context_hash: ContextHash,
        limit: usize,
    ) -> Vec<Strategy> {
        let extractor = self.strategy_extractor.read().await;
        extractor.get_strategies_for_context(context_hash, limit)
            .into_iter()
            .cloned()
            .collect()
    }

    /// Get all completed episodes
    pub async fn get_completed_episodes(&self) -> Vec<Episode> {
        self.episode_detector.read().await.get_completed_episodes().to_vec()
    }

    /// Get memory statistics
    pub async fn get_memory_stats(&self) -> MemoryStats {
        self.memory_formation.read().await.get_stats()
    }

    /// Get strategy statistics
    pub async fn get_strategy_stats(&self) -> StrategyStats {
        self.strategy_extractor.read().await.get_stats()
    }

    /// Get reinforcement learning statistics
    pub async fn get_reinforcement_stats(&self) -> crate::inference::ReinforcementStats {
        self.inference.read().await.get_reinforcement_stats()
    }

    /// Manually update strategy outcome (for external feedback)
    pub async fn update_strategy_outcome(
        &self,
        strategy_id: StrategyId,
        success: bool,
    ) -> GraphResult<()> {
        self.strategy_extractor.write().await.update_strategy_outcome(strategy_id, success)
    }

    /// Force memory decay (for testing or periodic cleanup)
    pub async fn decay_memories(&self) {
        self.memory_formation.write().await.apply_decay();
    }
}

/// Health metrics for the graph engine
#[derive(Debug, Clone)]
pub struct GraphHealthMetrics {
    pub node_count: usize,
    pub edge_count: usize,
    pub average_degree: f32,
    pub largest_component_size: usize,
    pub events_processed: u64,
    pub processing_rate: f64, // events per second
    pub memory_usage_estimate: usize, // bytes
    pub is_healthy: bool,
}

impl GraphEngineStats {
    fn new() -> Self {
        Self {
            total_events_processed: 0,
            total_nodes_created: 0,
            total_relationships_created: 0,
            total_patterns_detected: 0,
            total_queries_executed: 0,
            average_processing_time_ms: 0.0,
            cache_hit_rate: 0.0,
            last_operation_time: std::time::Instant::now(),
            total_episodes_detected: 0,
            total_memories_formed: 0,
            total_strategies_extracted: 0,
            total_reinforcements_applied: 0,
        }
    }
}

impl Default for GraphEngineStats {
    fn default() -> Self {
        Self::new()
    }
}