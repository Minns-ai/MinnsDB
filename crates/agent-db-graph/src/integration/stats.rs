use super::*;

impl GraphEngine {
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

    /// Get live aggregate counts from all stores
    pub async fn get_store_metrics(&self) -> StoreMetrics {
        // Memory stats
        let memory_stats = {
            let store = self.memory_store.read().await;
            store.get_stats()
        };

        // Strategy stats
        let strategy_stats = {
            let store = self.strategy_store.read().await;
            store.get_stats()
        };

        // Claim stats
        let (claim_count, claim_embeddings_indexed) = match &self.claim_store {
            Some(store) => (store.count().unwrap_or(0), store.vector_index_size()),
            None => (0, 0),
        };

        // Graph stats
        let graph_stats = self.get_graph_stats().await;

        StoreMetrics {
            memories: MemoryMetrics {
                total: memory_stats.total_memories,
                episodic: memory_stats.episodic_count,
                semantic: memory_stats.semantic_count,
                schema: memory_stats.schema_count,
                avg_strength: memory_stats.avg_strength,
                avg_access_count: memory_stats.avg_access_count,
                agents_with_memories: memory_stats.agents_with_memories,
                unique_contexts: memory_stats.unique_contexts,
            },
            strategies: StrategyMetrics {
                total: strategy_stats.total_strategies,
                high_quality: strategy_stats.high_quality_strategies,
                avg_quality: strategy_stats.average_quality,
                agents_with_strategies: strategy_stats.agents_with_strategies,
            },
            claims: ClaimMetrics {
                total: claim_count,
                embeddings_indexed: claim_embeddings_indexed,
            },
            graph: GraphMetricsSummary {
                nodes: graph_stats.node_count,
                edges: graph_stats.edge_count,
                avg_degree: graph_stats.avg_degree,
                largest_component: graph_stats.largest_component_size,
            },
        }
    }

    /// Get detected patterns
    pub async fn get_patterns(&self) -> Vec<String> {
        let inference = self.inference.read().await;
        inference
            .get_temporal_patterns()
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
}

/// Health metrics for the graph engine
#[derive(Debug, Clone)]
pub struct GraphHealthMetrics {
    pub node_count: usize,
    pub edge_count: usize,
    pub average_degree: f32,
    pub largest_component_size: usize,
    pub events_processed: u64,
    pub processing_rate: f64,         // events per second
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

/// Live aggregate metrics from all stores
#[derive(Debug, Clone, serde::Serialize)]
pub struct StoreMetrics {
    pub memories: MemoryMetrics,
    pub strategies: StrategyMetrics,
    pub claims: ClaimMetrics,
    pub graph: GraphMetricsSummary,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct MemoryMetrics {
    pub total: usize,
    pub episodic: usize,
    pub semantic: usize,
    pub schema: usize,
    pub avg_strength: f32,
    pub avg_access_count: u32,
    pub agents_with_memories: usize,
    pub unique_contexts: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct StrategyMetrics {
    pub total: usize,
    pub high_quality: usize,
    pub avg_quality: f32,
    pub agents_with_strategies: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct ClaimMetrics {
    pub total: usize,
    pub embeddings_indexed: usize,
}

#[derive(Debug, Clone, serde::Serialize)]
pub struct GraphMetricsSummary {
    pub nodes: usize,
    pub edges: usize,
    pub avg_degree: f32,
    pub largest_component: usize,
}

impl Default for GraphEngineStats {
    fn default() -> Self {
        Self::new()
    }
}
