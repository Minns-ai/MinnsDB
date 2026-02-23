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

mod constructor;
mod event_processing;
mod export_import;
mod graph_analytics;
mod graph_building;
mod lifecycle;
mod persistence;
mod pipeline;
mod queries;
mod stats;
mod world_model;
mod planning;
pub(crate) mod planning_llm_adapter;
mod execution;

pub use stats::{
    ClaimMetrics, GraphHealthMetrics, GraphMetricsSummary, MemoryMetrics, StoreMetrics,
    StrategyMetrics,
};

use crate::algorithms::{CentralityMeasures, LabelPropagationAlgorithm, LouvainAlgorithm, RandomWalker, TemporalReachability};
use crate::analytics::GraphAnalytics;
use crate::episodes::{Episode, EpisodeDetector, EpisodeDetectorConfig, EpisodeOutcome};
use crate::event_ordering::{EventOrderingEngine, OrderingConfig};
use crate::indexing::{IndexManager, IndexType};
use crate::inference::{EpisodeMetrics, GraphInference, InferenceConfig};
use crate::memory::{Memory, MemoryFormationConfig, MemoryStats};
use crate::stores::{
    InMemoryMemoryStore, InMemoryStrategyStore, MemoryStore, RedbMemoryStore, RedbStrategyStore,
    StrategyStore,
};
use crate::strategies::{
    Strategy, StrategyExtractionConfig, StrategyId, StrategySimilarityQuery, StrategyStats,
};
use crate::structures::{
    EdgeId, EdgeType, EdgeWeight, GoalStatus, Graph, GraphEdge, GraphNode, GraphStats, NodeId,
    NodeType,
};
use crate::transitions::{TransitionModel, TransitionModelConfig};
use agent_db_world_model::{EbmWorldModel, WorldModelConfig, WorldModelCritic};
use agent_db_planning::{PlanningConfig, WorldModelMode};
use agent_db_planning::orchestrator::PlanningOrchestrator;
use crate::traversal::{ActionSuggestion, GraphQuery, GraphTraversal, QueryResult};
use crate::{GraphError, GraphResult};
use agent_db_core::types::{AgentId, AgentType, ContextHash, EventId, SessionId};
use agent_db_events::core::LearningEvent;
use agent_db_events::{Event, EventType};
use agent_db_storage::{table_names, BatchOperation, RedbBackend, RedbConfig};
use serde_json::json;
use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{timeout, Duration};

/// Type alias for the memory store type used in the engine
type MemoryStoreType = Arc<RwLock<Box<dyn MemoryStore>>>;

/// Type alias for the strategy store type used in the engine
type StrategyStoreType = Arc<RwLock<Box<dyn StrategyStore>>>;

/// Storage backend type
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StorageBackend {
    /// In-memory only (fast, no persistence)
    InMemory,
    /// Persistent with redb (LRU cache + durable storage)
    Persistent,
}

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

    /// Interval for persisting graph state (in events processed)
    pub persistence_interval: u64,

    /// Maximum graph size before cleanup
    pub max_graph_size: usize,

    /// Enable Louvain community detection
    pub enable_louvain: bool,

    /// Interval for running community detection (in events processed)
    pub louvain_interval: u64,

    /// Community detection algorithm to use ("louvain" or "label_propagation")
    pub community_algorithm: String,

    /// Enable query caching
    pub enable_query_cache: bool,

    // ========== Persistent Storage Configuration ==========
    /// Storage backend type (InMemory or Persistent)
    pub storage_backend: StorageBackend,

    /// Path to redb database file (only used when storage_backend = Persistent)
    pub redb_path: PathBuf,

    /// Cache size for redb backend (MB of RAM for hot data)
    pub redb_cache_size_mb: usize,

    /// Maximum memories to keep in LRU cache (0 = unlimited)
    pub memory_cache_size: usize,

    /// Maximum strategies to keep in LRU cache (0 = unlimited)
    pub strategy_cache_size: usize,

    // ========== Semantic Memory Configuration ==========
    /// Enable semantic memory with claim extraction
    pub enable_semantic_memory: bool,

    /// Number of NER worker threads
    pub ner_workers: usize,

    /// External NER service URL
    pub ner_service_url: String,

    /// NER service request timeout (milliseconds)
    pub ner_request_timeout_ms: u64,

    /// Optional NER model name to request
    pub ner_model: Option<String>,

    /// Path to NER feature storage (redb)
    pub ner_storage_path: Option<PathBuf>,

    /// Context size threshold for automatic NER extraction (bytes)
    pub ner_promotion_threshold: usize,

    /// Number of claim extraction worker threads
    pub claim_workers: usize,

    /// Path to claim storage (redb)
    pub claim_storage_path: Option<PathBuf>,

    /// OpenAI API key for claim extraction
    pub openai_api_key: Option<String>,

    /// LLM model for claim extraction
    pub llm_model: String,

    /// Minimum confidence threshold for accepting claims
    pub claim_min_confidence: f32,

    /// Maximum claims to extract per context
    pub claim_max_per_input: usize,

    /// Number of embedding generation worker threads
    pub embedding_workers: usize,

    /// Enable automatic embedding generation for claims
    pub enable_embedding_generation: bool,

    // ========== 10x/100x: Consolidation + Refinement ==========
    /// Configuration for memory consolidation
    pub consolidation_config: crate::consolidation::ConsolidationConfig,

    /// Configuration for LLM refinement
    pub refinement_config: crate::refinement::RefinementConfig,

    /// How many episodes between consolidation passes
    pub consolidation_interval: u64,

    // ========== Maintenance ==========
    /// Configuration for the periodic background maintenance loop
    pub maintenance_config: crate::maintenance::MaintenanceConfig,

    // ========== Graph Pruning ==========
    /// Configuration for streaming graph pruning (merge/delete dead nodes)
    pub pruning_config: crate::graph_pruning::GraphPruningConfig,

    /// Maximum transition episodes to keep in memory
    pub max_transition_episodes: usize,

    /// Minimum transition count to survive pruning
    pub min_transition_count: u64,

    // ========== Bounded Memory Caps ==========
    /// Maximum events to keep in the in-memory event_store (ring-buffer cap)
    pub max_event_store_size: usize,

    /// Maximum age (seconds) for decision traces before TTL eviction
    pub max_decision_trace_age_secs: u64,

    // ========== World Model (Shadow Mode) ==========
    /// Enable the energy-based world model (shadow mode: train + score + log only)
    pub enable_world_model: bool,
    /// World model configuration (embed_dim, learning_rate, etc.)
    pub world_model_config: WorldModelConfig,
    /// Graduated activation mode for the world model
    pub world_model_mode: WorldModelMode,

    // ========== Planning Engine ==========
    /// Planning engine configuration (generation, selection, repair thresholds)
    pub planning_config: PlanningConfig,

    /// API key for planning LLM (separate from claim extraction)
    pub planning_llm_api_key: Option<String>,

    /// LLM provider for planning ("openai" or "anthropic")
    pub planning_llm_provider: String,
}

impl GraphEngineConfig {
    /// Effective world model mode (backward compat: enable_world_model=true → Shadow)
    pub fn effective_world_model_mode(&self) -> WorldModelMode {
        if self.world_model_mode != WorldModelMode::Disabled {
            self.world_model_mode
        } else if self.enable_world_model {
            WorldModelMode::Shadow
        } else {
            WorldModelMode::Disabled
        }
    }
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
pub(crate) struct DecisionTrace {
    memory_ids: Vec<u64>,
    memory_used: Vec<u64>,
    strategy_ids: Vec<u64>,
    strategy_used: Vec<u64>,
    claim_ids: Vec<u64>,
    claims_used: Vec<u64>,
    last_updated: std::time::Instant,
}

/// Comprehensive graph engine that integrates all graph capabilities
///
/// **Self-Evolution Pipeline:**
/// 1. Events -> Episode Detection -> Episodes
/// 2. Episodes -> Memory Formation -> Memories
/// 3. Episodes -> Strategy Extraction -> Strategies
/// 4. Outcomes -> Reinforcement Learning -> Pattern Updates
/// 5. Context -> Policy Guide -> Action Suggestions
///
/// **NEW: Advanced Graph Features (2026-01-15):**
/// - Property indexing for 100-1000x faster queries
/// - Louvain community detection for memory clustering
/// - Centrality measures for action importance ranking
/// - Graph analytics for learning metrics
pub struct GraphEngine {
    /// Core inference engine
    pub(crate) inference: Arc<RwLock<GraphInference>>,

    /// Graph traversal engine
    pub(crate) traversal: Arc<RwLock<GraphTraversal>>,

    /// Event ordering engine for handling concurrent events
    pub(crate) event_ordering: Arc<EventOrderingEngine>,

    /// Scoped inference engine.
    /// **Intentionally ephemeral**: scoped inference caches are derived from the
    /// graph and rebuilt on demand. Persisting them would create staleness issues
    /// since they depend on the current graph topology.
    pub(crate) scoped_inference: Arc<crate::scoped_inference::ScopedInferenceEngine>,

    /// Episode detector - automatically detects episode boundaries
    pub(crate) episode_detector: Arc<RwLock<EpisodeDetector>>,

    /// Memory store - retrieval substrate (trait object for flexibility)
    pub(crate) memory_store: Arc<RwLock<Box<dyn MemoryStore>>>,

    /// Strategy store - policy substrate (trait object for flexibility)
    pub(crate) strategy_store: Arc<RwLock<Box<dyn StrategyStore>>>,

    /// Transition model - procedural memory spine
    pub(crate) transition_model: Arc<RwLock<TransitionModel>>,

    /// Event storage for episode processing.
    /// **Intentionally ephemeral**: events are transient inputs consumed by the episode
    /// pipeline. They are ring-buffered and evicted by the maintenance loop, so
    /// persisting them would add I/O cost with no benefit (episodes capture the
    /// durable representation of event sequences).
    pub(crate) event_store: Arc<RwLock<HashMap<agent_db_core::types::EventId, Event>>>,

    /// Insertion order for event_store ring-buffer eviction
    pub(crate) event_store_order: Arc<RwLock<VecDeque<agent_db_core::types::EventId>>>,

    /// Learning decision traces keyed by query_id (lock-free with DashMap for performance).
    /// **Intentionally ephemeral**: decision traces are short-lived feedback records
    /// linking a policy-guide query to its outcome. They expire via TTL sweep and
    /// exist only to close the reinforcement loop within a single session.
    pub(crate) decision_traces: Arc<dashmap::DashMap<String, DecisionTrace>>,

    /// Redb backend for graph persistence (shared with memory/strategy stores)
    pub(crate) redb_backend: Option<Arc<RedbBackend>>,

    /// Unified graph store (cold / source of truth) using structures.rs types
    pub(crate) graph_store: Option<Arc<RwLock<crate::redb_graph_store::RedbGraphStore>>>,

    /// Configuration
    pub config: GraphEngineConfig,

    /// Operation statistics
    pub(crate) stats: Arc<RwLock<GraphEngineStats>>,

    /// Event processing buffer
    pub(crate) event_buffer: Arc<RwLock<Vec<Event>>>,

    /// Last persistence checkpoint
    pub(crate) last_persistence: Arc<RwLock<u64>>,

    // ========== NEW: Advanced Graph Features ==========
    /// Index manager for fast property queries
    pub(crate) index_manager: Arc<RwLock<IndexManager>>,

    /// Louvain algorithm for community detection
    pub(crate) louvain: Arc<LouvainAlgorithm>,

    /// Centrality measures for importance ranking
    pub(crate) centrality: Arc<CentralityMeasures>,

    /// Random walk engine for PersonalizedPageRank
    pub(crate) random_walker: Arc<RandomWalker>,

    /// Temporal reachability for causal chain discovery
    pub(crate) temporal_reachability: Arc<TemporalReachability>,

    /// Label propagation for community detection
    pub(crate) label_propagation: Arc<LabelPropagationAlgorithm>,

    // ========== Semantic Memory (Optional) ==========
    /// NER extraction queue (optional, when semantic memory is enabled)
    pub(crate) ner_queue: Option<Arc<agent_db_ner::NerExtractionQueue>>,

    /// NER feature storage (optional, when semantic memory is enabled)
    pub(crate) ner_store: Option<Arc<agent_db_ner::NerFeatureStore>>,

    /// Claim extraction queue (optional, when semantic memory is enabled)
    pub(crate) claim_queue: Option<Arc<crate::claims::ClaimExtractionQueue>>,

    /// Claim storage (optional, when semantic memory is enabled)
    pub(crate) claim_store: Option<Arc<crate::claims::ClaimStore>>,

    /// LLM client for claim extraction (optional, when semantic memory is enabled)
    #[allow(dead_code)]
    pub(crate) llm_client: Option<Arc<dyn crate::claims::LlmClient>>,

    /// Embedding queue for semantic search (optional, when semantic memory is enabled)
    pub(crate) embedding_queue: Option<Arc<crate::claims::EmbeddingQueue>>,

    /// Embedding client (optional, when semantic memory is enabled)
    pub(crate) embedding_client: Option<Arc<dyn crate::claims::EmbeddingClient>>,

    // ========== 10x/100x: Consolidation + Refinement ==========
    /// Consolidation engine for memory hierarchy
    pub(crate) consolidation_engine: Arc<RwLock<crate::consolidation::ConsolidationEngine>>,

    /// Refinement engine for LLM-enhanced summaries
    pub(crate) refinement_engine: Option<Arc<crate::refinement::RefinementEngine>>,

    /// Counter for triggering periodic consolidation
    pub(crate) episodes_since_consolidation: Arc<RwLock<u64>>,

    // ========== World Model (Shadow Mode) ==========
    /// Energy-based world model for predictive coding (optional, shadow mode)
    pub(crate) world_model: Option<Arc<RwLock<EbmWorldModel>>>,

    // ========== Planning Engine ==========
    /// Planning orchestrator for strategy/action generation (optional)
    pub(crate) planning_orchestrator: Option<PlanningOrchestrator>,
    /// Strategy generator (mock for now, LLM later)
    pub(crate) strategy_generator: Option<Arc<dyn agent_db_planning::StrategyGenerator>>,
    /// Action generator (mock for now, LLM later)
    pub(crate) action_generator: Option<Arc<dyn agent_db_planning::ActionGenerator>>,

    // ========== Execution State ==========
    /// Active plan executions keyed by execution ID
    pub(crate) active_executions: Arc<dashmap::DashMap<u64, Arc<RwLock<execution::ExecutionState>>>>,
    /// Monotonic counter for generating unique execution IDs
    pub(crate) next_execution_id: Arc<std::sync::atomic::AtomicU64>,
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
            persistence_interval: 50,
            max_graph_size: 1_000_000,
            enable_louvain: true,
            louvain_interval: 1000,
            community_algorithm: "louvain".to_string(),
            enable_query_cache: true,
            // Storage configuration defaults
            storage_backend: StorageBackend::InMemory,
            redb_path: PathBuf::from("./agent_db_data/graph.redb"),
            redb_cache_size_mb: 128,    // 128MB cache by default
            memory_cache_size: 10_000,  // Keep 10K memories in RAM
            strategy_cache_size: 5_000, // Keep 5K strategies in RAM
            // Semantic memory defaults (disabled by default)
            enable_semantic_memory: false,
            ner_workers: 2,
            ner_service_url: "http://localhost:8081/ner".to_string(),
            ner_request_timeout_ms: 5_000,
            ner_model: None,
            ner_storage_path: Some(PathBuf::from("./agent_db_data/ner_features.redb")),
            ner_promotion_threshold: 1024, // 1KB threshold
            claim_workers: 4,
            claim_storage_path: Some(PathBuf::from("./agent_db_data/claims.redb")),
            openai_api_key: None,
            llm_model: "gpt-4o-mini".to_string(),
            claim_min_confidence: 0.7,
            claim_max_per_input: 10,
            embedding_workers: 2,
            enable_embedding_generation: true,
            // Consolidation + Refinement
            consolidation_config: crate::consolidation::ConsolidationConfig::default(),
            refinement_config: crate::refinement::RefinementConfig::default(),
            consolidation_interval: 10, // Run consolidation every 10 episodes
            // Maintenance
            maintenance_config: crate::maintenance::MaintenanceConfig::default(),
            // Graph pruning
            pruning_config: crate::graph_pruning::GraphPruningConfig::default(),
            max_transition_episodes: 10_000,
            min_transition_count: 2,
            // Bounded memory caps
            max_event_store_size: 50_000,
            max_decision_trace_age_secs: 3600, // 1 hour
            // World Model (shadow mode — disabled by default)
            enable_world_model: false,
            world_model_config: WorldModelConfig::default(),
            world_model_mode: WorldModelMode::Disabled,
            // Planning engine (disabled by default)
            planning_config: PlanningConfig::default(),
            planning_llm_api_key: None,
            planning_llm_provider: "openai".to_string(),
        }
    }
}

// Visualization types

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
