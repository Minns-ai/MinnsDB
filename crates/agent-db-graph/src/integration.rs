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
use crate::structures::{
    Graph, GraphNode, GraphEdge, NodeId, GraphStats, NodeType, EdgeType, EdgeWeight, GoalStatus,
};
use crate::inference::{GraphInference, InferenceConfig, EpisodeMetrics};
use crate::traversal::{GraphTraversal, GraphQuery, QueryResult, ActionSuggestion};
use crate::event_ordering::{EventOrderingEngine, OrderingConfig};
use crate::episodes::{EpisodeDetector, EpisodeDetectorConfig, Episode, EpisodeOutcome};
use crate::memory::{MemoryFormationConfig, Memory, MemoryStats};
use crate::strategies::{
    StrategyExtractionConfig, Strategy, StrategyId, StrategyStats, StrategySimilarityQuery,
};
use crate::stores::{InMemoryMemoryStore, InMemoryStrategyStore, MemoryStore, StrategyStore};
use crate::transitions::{TransitionModel, TransitionModelConfig};
use crate::indexing::{IndexManager, IndexType};
use crate::algorithms::{LouvainAlgorithm, CentralityMeasures};
use crate::analytics::GraphAnalytics;
use agent_db_events::{Event, EventType};
use agent_db_events::core::LearningEvent;
use serde_json::json;
use agent_db_storage::{StorageEngine};
use agent_db_core::types::{AgentId, AgentType, ContextHash, EventId, SessionId};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{timeout, Duration};

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

#[derive(Debug, Clone)]
struct DecisionTrace {
    memory_ids: Vec<u64>,
    memory_used: Vec<u64>,
    strategy_ids: Vec<u64>,
    strategy_used: Vec<u64>,
    last_updated: std::time::Instant,
}

/// Comprehensive graph engine that integrates all graph capabilities
///
/// **Self-Evolution Pipeline:**
/// 1. Events → Episode Detection → Episodes
/// 2. Episodes → Memory Formation → Memories
/// 3. Episodes → Strategy Extraction → Strategies
/// 4. Outcomes → Reinforcement Learning → Pattern Updates
/// 5. Context → Policy Guide → Action Suggestions
///
/// **NEW: Advanced Graph Features (2026-01-15):**
/// - Property indexing for 100-1000x faster queries
/// - Louvain community detection for memory clustering
/// - Centrality measures for action importance ranking
/// - Graph analytics for learning metrics
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

    /// Memory store - retrieval substrate
    memory_store: Arc<RwLock<InMemoryMemoryStore>>,

    /// Strategy store - policy substrate
    strategy_store: Arc<RwLock<InMemoryStrategyStore>>,

    /// Transition model - procedural memory spine
    transition_model: Arc<RwLock<TransitionModel>>,

    /// Event storage for episode processing
    event_store: Arc<RwLock<HashMap<agent_db_core::types::EventId, Event>>>,

    /// Learning decision traces keyed by query_id
    decision_traces: Arc<RwLock<HashMap<String, DecisionTrace>>>,

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

    // ========== NEW: Advanced Graph Features ==========

    /// Index manager for fast property queries
    index_manager: Arc<RwLock<IndexManager>>,

    /// Louvain algorithm for community detection
    louvain: Arc<LouvainAlgorithm>,

    /// Centrality measures for importance ranking
    centrality: Arc<CentralityMeasures>,
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

        let memory_store = Arc::new(RwLock::new(
            InMemoryMemoryStore::new(config.memory_config.clone())
        ));

        let strategy_store = Arc::new(RwLock::new(
            InMemoryStrategyStore::new(config.strategy_config.clone())
        ));

        let transition_model = Arc::new(RwLock::new(
            TransitionModel::new(TransitionModelConfig::default())
        ));

        // Initialize advanced graph features
        let index_manager = Arc::new(RwLock::new(IndexManager::new()));

        // Auto-create common indexes for fast lookups
        {
            let mut idx_mgr = index_manager.write().await;

            // Context hash index (exact match, high frequency)
            idx_mgr.create_index(
                "context_hash_idx".to_string(),
                "context_hash".to_string(),
                IndexType::Hash,
            )?;

            // Agent type index (exact match)
            idx_mgr.create_index(
                "agent_type_idx".to_string(),
                "agent_type".to_string(),
                IndexType::Hash,
            )?;

            // Event type index (exact match)
            idx_mgr.create_index(
                "event_type_idx".to_string(),
                "event_type".to_string(),
                IndexType::Hash,
            )?;

            // Significance index (range queries)
            idx_mgr.create_index(
                "significance_idx".to_string(),
                "significance".to_string(),
                IndexType::BTree,
            )?;
        }

        let louvain = Arc::new(LouvainAlgorithm::new());
        let centrality = Arc::new(CentralityMeasures::new());

        Ok(Self {
            inference,
            traversal,
            event_ordering,
            scoped_inference,
            episode_detector,
            memory_store,
            strategy_store,
            transition_model,
            event_store: Arc::new(RwLock::new(HashMap::new())),
            decision_traces: Arc::new(RwLock::new(HashMap::new())),
            storage: None,
            config,
            stats: Arc::new(RwLock::new(GraphEngineStats::default())),
            event_buffer: Arc::new(RwLock::new(Vec::new())),
            last_persistence: Arc::new(RwLock::new(0)),
            index_manager,
            louvain,
            centrality,
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
        tracing::info!(
            "GraphEngine process_event start id={} agent_id={} session_id={} type={}",
            event.id,
            event.agent_id,
            event.session_id,
            Self::event_type_name(&event.event_type)
        );
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
        tracing::info!("Ordering event {}", event.id);
        let ordering_result = self.event_ordering.process_event(event.clone()).await?;
        if !ordering_result.issues.is_empty() {
            for issue in &ordering_result.issues {
                tracing::warn!("Ordering issue event_id={} issue={:?}", event.id, issue);
            }
        }

        // Step 2: Process all ready events through graph construction
        for ready_event in ordering_result.ready_events {
            // Add to processing buffer
            {
                let mut buffer = self.event_buffer.write().await;
                buffer.push(ready_event.clone());
            }

            // Process through inference engine
            tracing::info!("Inference processing event {}", ready_event.id);
            tracing::info!("Acquiring inference write lock for event {}", ready_event.id);
            let nodes_result = {
                let mut inference = self.inference.write().await;
                inference.process_event(ready_event.clone())
            };
            match nodes_result {
                Ok(nodes) => {
                    result.nodes_created.extend(nodes.clone());

                    // Auto-index newly created nodes
                    tracing::info!("Auto-index start event_id={} nodes={}", ready_event.id, nodes.len());
                    self.auto_index_nodes(&nodes).await?;
                    tracing::info!("Auto-index done event_id={}", ready_event.id);
                }
                Err(e) => {
                    result.errors.push(e);
                }
            }

            // Process through scoped inference engine (session + agent_type isolation)
            let scoped_event = crate::scoped_inference::ScopedEvent {
                event: ready_event.clone(),
                agent_type: ready_event.agent_type.clone(),
                priority: 0.0,
                scope_metadata: crate::scoped_inference::ScopeMetadata {
                    workspace_id: None,
                    user_id: None,
                    environment: None,
                    tags: Vec::new(),
                },
            };
            tracing::info!("Scoped inference processing event {}", ready_event.id);
            if let Err(e) = self.scoped_inference.process_scoped_event(scoped_event).await {
                result.errors.push(e);
            }

            if let Err(e) = self.handle_learning_event(&ready_event).await {
                result.errors.push(e);
            }

            // Step 3: Self-Evolution Pipeline - Episode Detection
            if self.config.auto_episode_detection {
                // Check for completed episodes (must drop write lock before acquiring read lock)
                let episode_update = {
                    self.episode_detector.write().await.process_event(&ready_event)
                };

                if let Some(episode_update) = episode_update {
                    let (episode_id, is_correction) = match episode_update {
                        crate::episodes::EpisodeUpdate::Completed(id) => (id, false),
                        crate::episodes::EpisodeUpdate::Corrected(id) => (id, true),
                    };
                    result.patterns_detected.push(format!(
                        "{}_{}",
                        if is_correction { "episode_corrected" } else { "episode_completed" },
                        episode_id
                    ));

                    let episodes: Vec<Episode> = self.episode_detector.read().await.get_completed_episodes().to_vec();
                    if let Some(episode) = episodes.iter().find(|e| e.id == episode_id) {
                        if !is_correction {
                            self.stats.write().await.total_episodes_detected += 1;
                        }

                        if self.config.auto_memory_formation {
                            self.process_episode_for_memory(episode).await?;
                        }

                        if self.config.auto_strategy_extraction {
                            self.process_episode_for_strategy(episode).await?;
                        }

                        if self.config.auto_reinforcement_learning {
                            if is_correction {
                                self.update_transition_model(episode).await?;
                            } else {
                                self.process_episode_for_reinforcement(episode).await?;
                            }
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

            // Run Louvain community detection periodically (every 1000 events)
            if stats.total_events_processed % 1000 == 0 {
                drop(stats); // Release stats lock before async operation
                if let Err(e) = self.run_community_detection().await {
                    result.errors.push(e);
                } else {
                    result.patterns_detected.push("louvain_communities_updated".to_string());
                }
            }
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
        tracing::info!(
            "GraphEngine process_event done id={} nodes_created={} patterns_detected={} errors={} duration_ms={}",
            event.id,
            result.nodes_created.len(),
            result.patterns_detected.len(),
            result.errors.len(),
            result.processing_time_ms
        );
        Ok(result)
    }

    fn event_type_name(event_type: &EventType) -> &'static str {
        match event_type {
            EventType::Action { .. } => "Action",
            EventType::Observation { .. } => "Observation",
            EventType::Cognitive { .. } => "Cognitive",
            EventType::Communication { .. } => "Communication",
            EventType::Learning { .. } => "Learning",
        }
    }

    async fn handle_learning_event(&self, event: &Event) -> GraphResult<()> {
        let EventType::Learning { event: learning_event } = &event.event_type else {
            return Ok(());
        };

        let now = std::time::Instant::now();
        match learning_event {
            LearningEvent::MemoryRetrieved { query_id, memory_ids } => {
                let mut traces = self.decision_traces.write().await;
                let trace = traces.entry(query_id.clone()).or_insert(DecisionTrace {
                    memory_ids: Vec::new(),
                    memory_used: Vec::new(),
                    strategy_ids: Vec::new(),
                    strategy_used: Vec::new(),
                    last_updated: now,
                });
                trace.memory_ids = memory_ids.clone();
                trace.last_updated = now;
                tracing::info!(
                    "Learning telemetry memory_retrieved query_id={} count={}",
                    query_id,
                    memory_ids.len()
                );
            }
            LearningEvent::MemoryUsed { query_id, memory_id } => {
                let mut traces = self.decision_traces.write().await;
                let trace = traces.entry(query_id.clone()).or_insert(DecisionTrace {
                    memory_ids: Vec::new(),
                    memory_used: Vec::new(),
                    strategy_ids: Vec::new(),
                    strategy_used: Vec::new(),
                    last_updated: now,
                });
                if !trace.memory_used.contains(memory_id) {
                    trace.memory_used.push(*memory_id);
                }
                trace.last_updated = now;
                tracing::info!(
                    "Learning telemetry memory_used query_id={} memory_id={}",
                    query_id,
                    memory_id
                );
            }
            LearningEvent::StrategyServed { query_id, strategy_ids } => {
                let mut traces = self.decision_traces.write().await;
                let trace = traces.entry(query_id.clone()).or_insert(DecisionTrace {
                    memory_ids: Vec::new(),
                    memory_used: Vec::new(),
                    strategy_ids: Vec::new(),
                    strategy_used: Vec::new(),
                    last_updated: now,
                });
                trace.strategy_ids = strategy_ids.clone();
                trace.last_updated = now;
                tracing::info!(
                    "Learning telemetry strategy_served query_id={} count={}",
                    query_id,
                    strategy_ids.len()
                );
            }
            LearningEvent::StrategyUsed { query_id, strategy_id } => {
                let mut traces = self.decision_traces.write().await;
                let trace = traces.entry(query_id.clone()).or_insert(DecisionTrace {
                    memory_ids: Vec::new(),
                    memory_used: Vec::new(),
                    strategy_ids: Vec::new(),
                    strategy_used: Vec::new(),
                    last_updated: now,
                });
                if !trace.strategy_used.contains(strategy_id) {
                    trace.strategy_used.push(*strategy_id);
                }
                trace.last_updated = now;
                tracing::info!(
                    "Learning telemetry strategy_used query_id={} strategy_id={}",
                    query_id,
                    strategy_id
                );
            }
            LearningEvent::Outcome { query_id, success } => {
                tracing::info!(
                    "Learning telemetry outcome query_id={} success={}",
                    query_id,
                    success
                );
                self.apply_learning_outcome(query_id, *success).await?;
            }
        }

        Ok(())
    }

    async fn apply_learning_outcome(&self, query_id: &str, success: bool) -> GraphResult<()> {
        let trace = {
            let mut traces = self.decision_traces.write().await;
            traces.remove(query_id)
        };

        let Some(trace) = trace else {
            return Ok(());
        };

        if !trace.memory_used.is_empty() {
            let mut store = self.memory_store.write().await;
            for memory_id in &trace.memory_used {
                let applied = store.apply_outcome(*memory_id, success);
                tracing::info!(
                    "Learning outcome applied to memory_id={} success={} applied={}",
                    memory_id,
                    success,
                    applied
                );
            }
        }

        if !trace.strategy_used.is_empty() {
            let mut store = self.strategy_store.write().await;
            for strategy_id in &trace.strategy_used {
                let updated = store.update_strategy_outcome(*strategy_id, success).is_ok();
                tracing::info!(
                    "Learning outcome applied to strategy_id={} success={} updated={}",
                    strategy_id,
                    success,
                    updated
                );
            }
        }

        Ok(())
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
        let _graph = {
            let _inference = self.inference.read().await;
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
            
            let _query_time = start_time.elapsed().as_millis() as f64;
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
        if let Some(ref _storage) = self.storage {
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
            tracing::info!("Acquiring inference write lock for batch processing");
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
        let mut memory_store = self.memory_store.write().await;

        tracing::info!(
            "Memory formation start episode_id={} agent_id={} session_id={} significance={:.3} outcome={:?}",
            episode.id,
            episode.agent_id,
            episode.session_id,
            episode.significance,
            episode.outcome
        );
        if let Some(upsert) = memory_store.store_episode(episode) {
            if upsert.is_new {
                self.stats.write().await.total_memories_formed += 1;
            }
            if let Some(memory) = memory_store.get_memory(upsert.id) {
                drop(memory_store);
                let outputs = crate::contracts::build_episode_record(episode, &[]);
                tracing::info!(
                    "Learning outputs (memory): episode_id={} goal_bucket_id={} behavior_signature={}",
                    outputs.episode_id,
                    outputs.goal_bucket_id,
                    outputs.behavior_signature
                );
                tracing::info!(
                    "Memory formed id={} episode_id={} strength={:.3} relevance={:.3} context_hash={}",
                    upsert.id,
                    episode.id,
                    memory.strength,
                    memory.relevance_score,
                    memory.context.fingerprint
                );
                self.attach_memory_to_graph(episode, &memory).await?;
            }
        } else {
            tracing::info!(
                "Memory formation skipped episode_id={} (not eligible)",
                episode.id
            );
        }

        Ok(())
    }

    /// Process episode for strategy extraction
    async fn process_episode_for_strategy(&self, episode: &Episode) -> GraphResult<()> {
        // Only extract from successful episodes
        if episode.outcome != Some(EpisodeOutcome::Success) {
            tracing::info!(
                "Strategy extraction skipped episode_id={} outcome={:?}",
                episode.id,
                episode.outcome
            );
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
            let outputs = crate::contracts::build_learning_outputs(episode, &events);
            tracing::info!(
                "Learning outputs (strategy): episode_id={} goal_bucket_id={} transitions={}",
                outputs.episode_record.episode_id,
                outputs.episode_record.goal_bucket_id,
                outputs.abstract_trace.transitions.len()
            );
            let mut strategy_store = self.strategy_store.write().await;
            tracing::info!(
                "Strategy extraction start episode_id={} events={}",
                episode.id,
                events.len()
            );
            if let Some(upsert) = strategy_store.store_episode(episode, &events)? {
                if upsert.is_new {
                    self.stats.write().await.total_strategies_extracted += 1;
                }
                if let Some(strategy) = strategy_store.get_strategy(upsert.id) {
                    drop(strategy_store);
                    tracing::info!(
                        "Strategy formed id={} episode_id={} quality={:.3} success_count={} failure_count={}",
                        upsert.id,
                        episode.id,
                        strategy.quality_score,
                        strategy.success_count,
                        strategy.failure_count
                    );
                    self.attach_strategy_to_graph(episode, &strategy).await?;
                }
            } else {
                tracing::info!(
                    "Strategy extraction produced no strategy for episode_id={}",
                    episode.id
                );
            }
        } else {
            tracing::info!(
                "Strategy extraction skipped episode_id={} (no events)",
                episode.id
            );
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

        self.update_transition_model(episode).await?;

        self.stats.write().await.total_reinforcements_applied += 1;

        Ok(())
    }

    async fn update_transition_model(&self, episode: &Episode) -> GraphResult<()> {
        let should_update = matches!(
            episode.outcome,
            Some(EpisodeOutcome::Success) | Some(EpisodeOutcome::Failure)
        );
        if !should_update {
            return Ok(());
        }

        let success = matches!(episode.outcome, Some(EpisodeOutcome::Success));
        let events: Vec<Event> = {
            let store = self.event_store.read().await;
            episode
                .events
                .iter()
                .filter_map(|event_id| store.get(event_id).cloned())
                .collect()
        };

        if events.is_empty() {
            return Ok(());
        }

        let outputs = crate::contracts::build_learning_outputs(episode, &events);
        let mut model = self.transition_model.write().await;
        model.update_from_trace(
            outputs.episode_record.goal_bucket_id,
            &outputs.abstract_trace,
            outputs.episode_record.episode_id,
            success,
        );
        tracing::info!(
            "Transition model updated episode_id={} goal_bucket_id={} transitions={} success={}",
            outputs.episode_record.episode_id,
            outputs.episode_record.goal_bucket_id,
            outputs.abstract_trace.transitions.len(),
            success
        );

        Ok(())
    }

    // ============================================================================
    // Self-Evolution Query Methods
    // ============================================================================

    /// Get policy guide suggestions for what action to take next
    ///
    /// **Returns:** Action suggestions ranked by success probability and centrality
    pub async fn get_next_action_suggestions(
        &self,
        context_hash: ContextHash,
        last_action_node: Option<NodeId>,
        limit: usize,
    ) -> GraphResult<Vec<ActionSuggestion>> {
        let inference = self.inference.read().await;
        let graph = inference.graph();
        let traversal = self.traversal.read().await;

        let mut suggestions = traversal.get_next_step_suggestions(graph, context_hash, last_action_node, limit * 2)?;

        // Calculate centrality scores for ranking
        let centrality_scores = self.centrality.all_centralities(graph)?;

        // Re-rank suggestions using combined score: success_probability * centrality
        for suggestion in &mut suggestions {
            let combined_score = centrality_scores.combined_score(suggestion.action_node_id);

            // Blend success probability (60%) with centrality importance (40%)
            let original_prob = suggestion.success_probability;
            suggestion.success_probability = (original_prob * 0.6) + (combined_score * 0.4);
        }

        // Re-sort by updated success probability
        suggestions.sort_by(|a, b| {
            b.success_probability
                .partial_cmp(&a.success_probability)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Return top-k after centrality ranking
        Ok(suggestions.into_iter().take(limit).collect())
    }

    /// Get all memories for an agent
    pub async fn get_agent_memories(&self, agent_id: AgentId, limit: usize) -> Vec<Memory> {
        self.memory_store.read().await
            .get_agent_memories(agent_id, limit)
    }

    /// Retrieve memories by context similarity
    pub async fn retrieve_memories_by_context(
        &self,
        context: &agent_db_events::core::EventContext,
        limit: usize,
    ) -> Vec<Memory> {
        self.memory_store.write().await.retrieve_by_context(context, limit)
    }

    /// Retrieve memories by context similarity with optional filtering
    pub async fn retrieve_memories_by_context_similar(
        &self,
        context: &agent_db_events::core::EventContext,
        limit: usize,
        min_similarity: f32,
        agent_id: Option<AgentId>,
        session_id: Option<SessionId>,
    ) -> Vec<Memory> {
        self.memory_store
            .write()
            .await
            .retrieve_by_context_similar(context, limit, min_similarity, agent_id, session_id)
    }

    /// Get all strategies for an agent
    pub async fn get_agent_strategies(&self, agent_id: AgentId, limit: usize) -> Vec<Strategy> {
        let extractor = self.strategy_store.read().await;
        extractor.get_agent_strategies(agent_id, limit)
    }

    /// Get strategies applicable to a context
    pub async fn get_strategies_for_context(
        &self,
        context_hash: ContextHash,
        limit: usize,
    ) -> Vec<Strategy> {
        let extractor = self.strategy_store.read().await;
        extractor.get_strategies_for_context(context_hash, limit)
    }

    /// Find strategies similar to a graph signature
    pub async fn get_similar_strategies(
        &self,
        query: StrategySimilarityQuery,
    ) -> Vec<(Strategy, f32)> {
        let extractor = self.strategy_store.read().await;
        extractor.find_similar_strategies(query)
    }

    /// Get all completed episodes
    pub async fn get_completed_episodes(&self) -> Vec<Episode> {
        self.episode_detector.read().await.get_completed_episodes().to_vec()
    }

    /// Get memory statistics
    pub async fn get_memory_stats(&self) -> MemoryStats {
        self.memory_store.read().await.get_stats()
    }

    /// Get strategy statistics
    pub async fn get_strategy_stats(&self) -> StrategyStats {
        self.strategy_store.read().await.get_stats()
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
        self.strategy_store.write().await.update_strategy_outcome(strategy_id, success)
    }

    /// Force memory decay (for testing or periodic cleanup)
    pub async fn decay_memories(&self) {
        self.memory_store.write().await.apply_decay();
    }

    // ============================================================================
    // Advanced Graph Features - Helper Methods
    // ============================================================================

    /// Auto-index newly created nodes
    async fn auto_index_nodes(&self, node_ids: &[NodeId]) -> GraphResult<()> {
        tracing::info!("auto_index_nodes acquiring inference read lock (nodes={})", node_ids.len());
        let inference = match timeout(Duration::from_secs(2), self.inference.read()).await {
            Ok(lock) => lock,
            Err(_) => {
                tracing::info!("auto_index_nodes timeout acquiring inference read lock");
                return Err(GraphError::OperationError(
                    "auto_index_nodes timeout acquiring inference read lock".to_string(),
                ));
            }
        };
        tracing::info!("auto_index_nodes acquired inference read lock");
        let graph = inference.graph();
        tracing::info!("auto_index_nodes acquiring index manager write lock");
        let mut idx_mgr = self.index_manager.write().await;
        tracing::info!("auto_index_nodes acquired index manager write lock");

        for &node_id in node_ids {
            if let Some(node) = graph.get_node(node_id) {
                tracing::info!("auto_index_nodes indexing node_id={}", node_id);
                // Index common properties
                for (key, value) in &node.properties {
                    // Track property queries for auto-indexing
                    idx_mgr.record_property_query(key);

                    // Find or auto-create index for this property
                    if let Some(index) = idx_mgr.find_index_for_property(key) {
                        index.insert(node_id, value);
                    }
                }

                // Index node type
                let node_type_str = node.type_name();
                let node_type_value = serde_json::json!(node_type_str);
                if let Some(index) = idx_mgr.find_index_for_property("node_type") {
                    index.insert(node_id, &node_type_value);
                }
            }
        }

        tracing::info!("auto_index_nodes finished");
        Ok(())
    }

    /// Run Louvain community detection and update memory clusters
    async fn run_community_detection(&self) -> GraphResult<()> {
        tracing::info!("Acquiring inference write lock for community detection");
        let mut inference = self.inference.write().await;
        let graph = inference.graph_mut();

        // Detect communities using Louvain
        let communities = self.louvain.detect_communities(graph)?;

        for (node_id, community_id) in communities.node_communities {
            if let Some(node) = graph.get_node_mut(node_id) {
                node.properties.insert("community_id".to_string(), json!(community_id));
                node.touch();
            }
        }

        Ok(())
    }

    /// Get graph analytics including learning metrics
    pub async fn get_analytics(&self) -> GraphResult<crate::analytics::GraphMetrics> {
        let inference = self.inference.read().await;
        let graph = inference.graph();

        let analytics = GraphAnalytics::from_ref(graph);
        analytics.calculate_all_metrics()
    }

    /// Get property index statistics
    pub async fn get_index_stats(&self) -> Vec<crate::indexing::IndexStats> {
        let idx_mgr = self.index_manager.read().await;
        idx_mgr.get_all_stats().into_values().collect()
    }

    /// Manually trigger community detection
    pub async fn detect_communities(&self) -> GraphResult<crate::algorithms::CommunityDetectionResult> {
        let inference = self.inference.read().await;
        let graph = inference.graph();

        self.louvain.detect_communities(graph)
    }

    /// Get centrality scores for all nodes
    pub async fn get_all_centrality_scores(&self) -> GraphResult<crate::algorithms::AllCentralities> {
        let inference = self.inference.read().await;
        let graph = inference.graph();

        self.centrality.all_centralities(graph)
    }

    /// Get graph structure for visualization
    pub async fn get_graph_structure(
        &self,
        limit: usize,
        session_id: Option<SessionId>,
        agent_type: Option<AgentType>,
    ) -> GraphStructure {
        let event_store = self.event_store.read().await;

        if let Some(session_id) = session_id {
            let agent_type = agent_type.and_then(|value| {
                if value.trim().is_empty() {
                    None
                } else {
                    Some(value)
                }
            });

            if let Some(agent_type) = agent_type {
                let scope = crate::scoped_inference::InferenceScope {
                    agent_type: agent_type.clone(),
                    session_id,
                };

                if let Ok(scope_engine) = self.scoped_inference.get_scope_engine(&scope).await {
                    let inference = scope_engine.read().await;
                    let graph = inference.graph();
                    return Self::build_graph_structure_from_events(
                        graph,
                        event_store
                            .iter()
                            .filter(|(_, event)| event.session_id == session_id && event.agent_type == agent_type),
                        limit,
                    );
                }

                return GraphStructure {
                    nodes: Vec::new(),
                    edges: Vec::new(),
                };
            }

            // Fallback: session-only filtering on the global graph
            let inference = self.inference.read().await;
            let graph = inference.graph();
            return Self::build_graph_structure_from_events(
                graph,
                event_store.iter().filter(|(_, event)| event.session_id == session_id),
                limit,
            );
        }

        let inference = self.inference.read().await;
        let graph = inference.graph();
        Self::build_graph_structure_from_events(graph, event_store.iter(), limit)
    }

    /// Get graph structure centered around a context hash
    pub async fn get_graph_structure_for_context(
        &self,
        context_hash: ContextHash,
        limit: usize,
        session_id: Option<SessionId>,
        agent_type: Option<AgentType>,
    ) -> GraphStructure {
        let event_store = self.event_store.read().await;

        if let Some(session_id) = session_id {
            let agent_type = agent_type.and_then(|value| {
                if value.trim().is_empty() {
                    None
                } else {
                    Some(value)
                }
            });

            if let Some(agent_type) = agent_type {
                let scope = crate::scoped_inference::InferenceScope {
                    agent_type: agent_type.clone(),
                    session_id,
                };

                if let Ok(scope_engine) = self.scoped_inference.get_scope_engine(&scope).await {
                    let inference = scope_engine.read().await;
                    let graph = inference.graph();
                    return Self::build_context_graph_structure(
                        graph,
                        context_hash,
                        event_store
                            .iter()
                            .filter(|(_, event)| event.session_id == session_id && event.agent_type == agent_type),
                        limit,
                    );
                }

                return GraphStructure {
                    nodes: Vec::new(),
                    edges: Vec::new(),
                };
            }

            let inference = self.inference.read().await;
            let graph = inference.graph();
            return Self::build_context_graph_structure(
                graph,
                context_hash,
                event_store.iter().filter(|(_, event)| event.session_id == session_id),
                limit,
            );
        }

        let inference = self.inference.read().await;
        let graph = inference.graph();
        Self::build_context_graph_structure(graph, context_hash, event_store.iter(), limit)
    }

    fn build_graph_structure_from_events<'a, I>(
        graph: &Graph,
        events: I,
        limit: usize,
    ) -> GraphStructure
    where
        I: Iterator<Item = (&'a EventId, &'a Event)>,
    {
        let mut nodes: HashMap<NodeId, GraphNodeData> = HashMap::new();
        let mut edges: Vec<GraphEdgeData> = Vec::new();

        for (event_id, event) in events.take(limit) {
            if let Some(node) = graph.get_event_node(*event_id) {
                let label = match &event.event_type {
                    EventType::Action { action_name, .. } => action_name.clone(),
                    EventType::Observation { observation_type, .. } => observation_type.clone(),
                    EventType::Cognitive { process_type, .. } => format!("{:?}", process_type),
                    EventType::Learning { .. } => "Learning".to_string(),
                    _ => format!("Event {}", event.id),
                };

                nodes.insert(
                    node.id,
                    Self::build_graph_node_data(node, Some(label)),
                );

                // Get edges from this node
                let outgoing_edges = graph.get_edges_from(node.id);
                for edge in outgoing_edges.into_iter().take(5) {
                    let edge_type = match &edge.edge_type {
                        EdgeType::Temporal { .. } => "Temporal",
                        EdgeType::Causality { .. } => "Causal",
                        EdgeType::Contextual { .. } => "Contextual",
                        EdgeType::Interaction { .. } => "Interaction",
                        EdgeType::Association { .. } => "Association",
                        EdgeType::GoalRelation { .. } => "GoalRelation",
                        EdgeType::Communication { .. } => "Communication",
                    };

                    edges.push(GraphEdgeData {
                        id: edge.id,
                        from: edge.source,
                        to: edge.target,
                        edge_type: edge_type.to_string(),
                        weight: edge.weight,
                        confidence: edge.confidence,
                    });

                    if let Some(target_node) = graph.get_node(edge.target) {
                        nodes.entry(target_node.id)
                            .or_insert_with(|| Self::build_graph_node_data(target_node, None));
                    }
                }
            }
        }

        GraphStructure {
            nodes: nodes.into_values().collect(),
            edges,
        }
    }

    fn build_graph_node_data(node: &GraphNode, label_override: Option<String>) -> GraphNodeData {
        let (node_type, label) = match &node.node_type {
            NodeType::Event { event_type, .. } => (format!("Event::{}", event_type), label_override),
            NodeType::Context { .. } => ("Context".to_string(), label_override),
            NodeType::Agent { .. } => ("Agent".to_string(), label_override),
            NodeType::Goal { description, .. } => ("Goal".to_string(), Some(description.clone())),
            NodeType::Episode { episode_id, .. } => ("Episode".to_string(), Some(format!("Episode {}", episode_id))),
            NodeType::Memory { memory_id, .. } => ("Memory".to_string(), Some(format!("Memory {}", memory_id))),
            NodeType::Strategy { name, .. } => ("Strategy".to_string(), Some(name.clone())),
            NodeType::Tool { tool_name, .. } => ("Tool".to_string(), Some(tool_name.clone())),
            NodeType::Result { summary, .. } => ("Result".to_string(), Some(summary.clone())),
            NodeType::Concept { concept_name, .. } => ("Concept".to_string(), Some(concept_name.clone())),
        };

        GraphNodeData {
            id: node.id,
            label,
            node_type,
            created_at: node.created_at,
            properties: serde_json::to_value(&node.properties)
                .unwrap_or_else(|_| serde_json::json!({})),
        }
    }

    fn build_context_graph_structure<'a, I>(
        graph: &Graph,
        context_hash: ContextHash,
        events: I,
        limit: usize,
    ) -> GraphStructure
    where
        I: Iterator<Item = (&'a EventId, &'a Event)>,
    {
        let mut event_labels: HashMap<EventId, String> = HashMap::new();
        for (event_id, event) in events.take(limit) {
            let label = match &event.event_type {
                EventType::Action { action_name, .. } => action_name.clone(),
                EventType::Observation { observation_type, .. } => observation_type.clone(),
                EventType::Cognitive { process_type, .. } => format!("{:?}", process_type),
                EventType::Learning { .. } => "Learning".to_string(),
                _ => format!("Event {}", event.id),
            };
            event_labels.insert(*event_id, label);
        }

        let context_node = match graph.get_context_node(context_hash) {
            Some(node) => node,
            None => {
                return GraphStructure {
                    nodes: Vec::new(),
                    edges: Vec::new(),
                }
            }
        };

        let mut nodes: HashMap<NodeId, GraphNodeData> = HashMap::new();
        let mut edges: Vec<GraphEdgeData> = Vec::new();

        nodes.insert(
            context_node.id,
            Self::build_graph_node_data(context_node, None),
        );

        let mut edge_count = 0usize;
        let mut candidate_edges = Vec::new();
        candidate_edges.extend(graph.get_edges_from(context_node.id));
        candidate_edges.extend(graph.get_edges_to(context_node.id));

        for edge in candidate_edges {
            if edge_count >= limit {
                break;
            }

            let edge_type = match &edge.edge_type {
                EdgeType::Temporal { .. } => "Temporal",
                EdgeType::Causality { .. } => "Causal",
                EdgeType::Contextual { .. } => "Contextual",
                EdgeType::Interaction { .. } => "Interaction",
                EdgeType::Association { .. } => "Association",
                EdgeType::GoalRelation { .. } => "GoalRelation",
                EdgeType::Communication { .. } => "Communication",
            };

            edges.push(GraphEdgeData {
                id: edge.id,
                from: edge.source,
                to: edge.target,
                edge_type: edge_type.to_string(),
                weight: edge.weight,
                confidence: edge.confidence,
            });
            edge_count += 1;

            for node_id in [edge.source, edge.target] {
                if nodes.contains_key(&node_id) {
                    continue;
                }

                if let Some(node) = graph.get_node(node_id) {
                    let label = if let NodeType::Event { event_id, .. } = node.node_type {
                        event_labels.get(&event_id).cloned()
                    } else {
                        None
                    };
                    nodes.insert(node.id, Self::build_graph_node_data(node, label));
                }
            }
        }

        GraphStructure {
            nodes: nodes.into_values().collect(),
            edges,
        }
    }

    async fn attach_memory_to_graph(&self, episode: &Episode, memory: &Memory) -> GraphResult<()> {
        let mut inference = self.inference.write().await;
        let graph = inference.graph_mut();

        let episode_node_id = Self::ensure_episode_node(graph, episode);
        let memory_node_id = Self::ensure_memory_node(graph, memory);

        Self::add_or_strengthen_association(
            graph,
            episode_node_id,
            memory_node_id,
            "FormedMemory",
            0.8,
            json!({
                "episode_id": episode.id,
                "memory_id": memory.id,
                "agent_id": memory.agent_id,
                "session_id": memory.session_id,
            }),
        );

        if let Some(context_node) = graph.get_context_node(memory.context.fingerprint) {
            Self::add_or_strengthen_association(
                graph,
                memory_node_id,
                context_node.id,
                "MemoryContext",
                0.7,
                json!({
                    "context_hash": memory.context.fingerprint,
                }),
            );
        }

        for goal in &episode.context.active_goals {
            let goal_node_id = Self::ensure_goal_node(graph, goal);
            Self::add_or_strengthen_association(
                graph,
                memory_node_id,
                goal_node_id,
                "MemoryGoal",
                0.6,
                json!({
                    "goal_id": goal.id,
                }),
            );
        }

        for event_id in episode.events.iter().take(5) {
            if let Some(event_node) = graph.get_event_node(*event_id) {
                Self::add_or_strengthen_association(
                    graph,
                    episode_node_id,
                    event_node.id,
                    "EpisodeContainsEvent",
                    0.4,
                    json!({
                        "event_id": event_id.to_string(),
                    }),
                );
            }
        }

        Ok(())
    }

    async fn attach_strategy_to_graph(&self, episode: &Episode, strategy: &Strategy) -> GraphResult<()> {
        let mut inference = self.inference.write().await;
        let graph = inference.graph_mut();

        let episode_node_id = Self::ensure_episode_node(graph, episode);
        let strategy_node_id = Self::ensure_strategy_node(graph, strategy);

        Self::add_or_strengthen_association(
            graph,
            episode_node_id,
            strategy_node_id,
            "DerivedStrategy",
            0.8,
            json!({
                "episode_id": episode.id,
                "strategy_id": strategy.id,
            }),
        );

        for goal in &episode.context.active_goals {
            let goal_node_id = Self::ensure_goal_node(graph, goal);
            Self::add_or_strengthen_association(
                graph,
                strategy_node_id,
                goal_node_id,
                "StrategyGoal",
                0.7,
                json!({
                    "goal_id": goal.id,
                }),
            );
        }

        let tool_names = strategy
            .metadata
            .get("tool_names")
            .and_then(|value| serde_json::from_str::<Vec<String>>(value).ok())
            .unwrap_or_default();
        for tool_name in tool_names {
            let tool_node_id = Self::ensure_tool_node(graph, &tool_name);
            Self::add_or_strengthen_association(
                graph,
                strategy_node_id,
                tool_node_id,
                "StrategyUsesTool",
                0.6,
                json!({
                    "tool_name": tool_name,
                }),
            );
        }

        let result_types = strategy
            .metadata
            .get("result_types")
            .and_then(|value| serde_json::from_str::<Vec<String>>(value).ok())
            .unwrap_or_default();
        for result_type in result_types {
            let result_key = format!("strategy:{}:{}", strategy.id, result_type);
            let result_node_id = Self::ensure_result_node(graph, &result_key, &result_type, &result_type);
            Self::add_or_strengthen_association(
                graph,
                strategy_node_id,
                result_node_id,
                "StrategyProducesResult",
                0.6,
                json!({
                    "result_type": result_type,
                }),
            );
        }

        Ok(())
    }

    fn ensure_episode_node(graph: &mut Graph, episode: &Episode) -> NodeId {
        if let Some(node) = graph.get_episode_node(episode.id) {
            node.id
        } else {
        let outcome = episode
            .outcome
            .as_ref()
            .map(|value| format!("{:?}", value))
            .unwrap_or_else(|| "Unknown".to_string());
            let mut node = GraphNode::new(NodeType::Episode {
                episode_id: episode.id,
                agent_id: episode.agent_id,
                session_id: episode.session_id,
                outcome: outcome.clone(),
            });
            node.properties.insert("outcome".to_string(), json!(outcome));
            node.properties.insert("event_count".to_string(), json!(episode.events.len()));
            node.properties.insert("significance".to_string(), json!(episode.significance));
            node.properties.insert("salience_score".to_string(), json!(episode.salience_score));
            graph.add_node(node)
        }
    }

    fn ensure_memory_node(graph: &mut Graph, memory: &Memory) -> NodeId {
        if let Some(node) = graph.get_memory_node(memory.id) {
            node.id
        } else {
            let mut node = GraphNode::new(NodeType::Memory {
                memory_id: memory.id,
                agent_id: memory.agent_id,
                session_id: memory.session_id,
            });
            node.properties.insert("strength".to_string(), json!(memory.strength));
            node.properties.insert("relevance_score".to_string(), json!(memory.relevance_score));
            node.properties.insert("context_hash".to_string(), json!(memory.context.fingerprint));
            node.properties.insert("formed_at".to_string(), json!(memory.formed_at));
            node.properties.insert("memory_type".to_string(), json!(format!("{:?}", memory.memory_type)));
            graph.add_node(node)
        }
    }

    fn ensure_strategy_node(graph: &mut Graph, strategy: &Strategy) -> NodeId {
        if let Some(node) = graph.get_strategy_node(strategy.id) {
            node.id
        } else {
            let mut node = GraphNode::new(NodeType::Strategy {
                strategy_id: strategy.id,
                agent_id: strategy.agent_id,
                name: strategy.name.clone(),
            });
            node.properties.insert("quality_score".to_string(), json!(strategy.quality_score));
            node.properties.insert("success_count".to_string(), json!(strategy.success_count));
            node.properties.insert("failure_count".to_string(), json!(strategy.failure_count));
            node.properties.insert("version".to_string(), json!(strategy.version));
            node.properties.insert("strategy_type".to_string(), json!(format!("{:?}", strategy.strategy_type)));
            node.properties.insert("support_count".to_string(), json!(strategy.support_count));
            node.properties.insert("expected_success".to_string(), json!(strategy.expected_success));
            node.properties.insert("expected_cost".to_string(), json!(strategy.expected_cost));
            node.properties.insert("expected_value".to_string(), json!(strategy.expected_value));
            node.properties.insert("confidence".to_string(), json!(strategy.confidence));
            node.properties.insert("goal_bucket_id".to_string(), json!(strategy.goal_bucket_id));
            node.properties.insert("behavior_signature".to_string(), json!(strategy.behavior_signature));
            node.properties.insert("precondition".to_string(), json!(strategy.precondition));
            node.properties.insert("action_hint".to_string(), json!(strategy.action_hint));
            graph.add_node(node)
        }
    }

    fn ensure_goal_node(graph: &mut Graph, goal: &agent_db_events::core::Goal) -> NodeId {
        if let Some(node) = graph.get_goal_node(goal.id) {
            node.id
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
            node.properties.insert("progress".to_string(), json!(goal.progress));
            if let Some(deadline) = goal.deadline {
                node.properties.insert("deadline".to_string(), json!(deadline));
            }
            graph.add_node(node)
        }
    }

    fn ensure_tool_node(graph: &mut Graph, tool_name: &str) -> NodeId {
        if let Some(node) = graph.get_tool_node(tool_name) {
            node.id
        } else {
            let mut node = GraphNode::new(NodeType::Tool {
                tool_name: tool_name.to_string(),
                tool_type: "external".to_string(),
            });
            node.properties.insert("usage_count".to_string(), json!(1));
            graph.add_node(node)
        }
    }

    fn ensure_result_node(graph: &mut Graph, result_key: &str, result_type: &str, summary: &str) -> NodeId {
        if let Some(node) = graph.get_result_node(result_key) {
            node.id
        } else {
            let mut node = GraphNode::new(NodeType::Result {
                result_key: result_key.to_string(),
                result_type: result_type.to_string(),
                summary: summary.to_string(),
            });
            node.properties.insert("summary".to_string(), json!(summary));
            graph.add_node(node)
        }
    }

    fn add_or_strengthen_association(
        graph: &mut Graph,
        source: NodeId,
        target: NodeId,
        association_type: &str,
        weight: EdgeWeight,
        properties: serde_json::Value,
    ) {
        if let Some(edge) = graph.get_edge_between_mut(source, target) {
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
        graph.add_edge(edge);
    }

    /// Get the most recent events from the in-memory event store
    pub async fn get_recent_events(&self, limit: usize) -> Vec<Event> {
        let store = self.event_store.read().await;
        let mut events: Vec<Event> = store.values().cloned().collect();
        events.sort_by_key(|event| std::cmp::Reverse(event.timestamp));
        events.into_iter().take(limit).collect()
    }
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct GraphStructure {
    pub nodes: Vec<GraphNodeData>,
    pub edges: Vec<GraphEdgeData>,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct GraphNodeData {
    pub id: u64,
    pub label: Option<String>,
    pub node_type: String,
    pub created_at: u64,
    pub properties: serde_json::Value,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct GraphEdgeData {
    pub id: u64,
    pub from: u64,
    pub to: u64,
    pub edge_type: String,
    pub weight: f32,
    pub confidence: f32,
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