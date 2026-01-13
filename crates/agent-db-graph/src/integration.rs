//! Integration layer for graph operations
//!
//! This module provides a unified interface that integrates graph structures,
//! inference, and traversal capabilities with the storage layer.

use crate::{GraphResult, GraphError};
use crate::structures::{Graph, GraphNode, GraphEdge, NodeId, GraphStats};
use crate::inference::{GraphInference, InferenceConfig, InferenceStats, InferenceResults};
use crate::traversal::{GraphTraversal, GraphQuery, QueryResult};
use crate::event_ordering::{EventOrderingEngine, OrderingConfig, OrderingResult};
use agent_db_events::Event;
use agent_db_storage::{StorageEngine, StorageResult};
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
    
    /// Enable automatic pattern detection
    pub auto_pattern_detection: bool,
    
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
pub struct GraphEngine {
    /// Core inference engine
    inference: Arc<RwLock<GraphInference>>,
    
    /// Graph traversal engine
    traversal: Arc<RwLock<GraphTraversal>>,
    
    /// Event ordering engine for handling concurrent events
    event_ordering: Arc<EventOrderingEngine>,
    
    /// Scoped inference engine
    scoped_inference: Arc<crate::scoped_inference::ScopedInferenceEngine>,
    
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
}

impl Default for GraphEngineConfig {
    fn default() -> Self {
        Self {
            inference_config: InferenceConfig::default(),
            ordering_config: OrderingConfig::default(),
            scoped_inference_config: crate::scoped_inference::ScopedInferenceConfig::default(),
            auto_pattern_detection: true,
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
            crate::scoped_inference::ScopedInferenceEngine::new(config.scoped_inference_config.clone())
        );
        
        Ok(Self {
            inference,
            traversal,
            event_ordering,
            scoped_inference,
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
    pub async fn process_event(&self, event: Event) -> GraphResult<GraphOperationResult> {
        let start_time = std::time::Instant::now();
        let mut result = GraphOperationResult {
            nodes_created: Vec::new(),
            relationships_discovered: 0,
            patterns_detected: Vec::new(),
            processing_time_ms: 0,
            errors: Vec::new(),
        };
        
        // Step 1: Order the event (handles out-of-order arrival)
        let ordering_result = self.event_ordering.process_event(event.clone()).await?;
        
        // Step 2: Process all ready events (may include previously buffered ones)
        for ready_event in ordering_result.ready_events {
            // Add to processing buffer
            {
                let mut buffer = self.event_buffer.write().await;
                buffer.push(ready_event.clone());
            }
            
            // Process through inference engine
            match self.inference.write().await.process_event(ready_event) {
                Ok(nodes) => {
                    result.nodes_created.extend(nodes);
                }
                Err(e) => {
                    result.errors.push(e);
                }
            }
        }
        
        // Step 3: Update statistics with ordering info
        if ordering_result.reordering_occurred {
            // Track that reordering happened for debugging
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
        self.scoped_inference.get_statistics().await
    }
    
    /// Query events in a specific scope
    pub async fn query_events_in_scope(&self, scope: &crate::scoped_inference::InferenceScope) -> GraphResult<Vec<Event>> {
        self.scoped_inference.query_events_in_scope(scope).await
    }
    
    /// Get cross-scope relationships
    pub async fn get_cross_scope_relationships(&self) -> GraphResult<Vec<String>> {
        self.scoped_inference.get_cross_scope_relationships().await
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
        }
    }
}

impl Default for GraphEngineStats {
    fn default() -> Self {
        Self::new()
    }
}