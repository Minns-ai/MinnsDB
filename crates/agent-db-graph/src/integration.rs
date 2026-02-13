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

use crate::algorithms::{CentralityMeasures, LouvainAlgorithm};
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
use crate::traversal::{ActionSuggestion, GraphQuery, GraphTraversal, QueryResult};
use crate::{GraphError, GraphResult};
use agent_db_core::types::{AgentId, AgentType, ContextHash, EventId, SessionId};
use agent_db_events::core::LearningEvent;
use agent_db_events::{Event, EventType};
use agent_db_storage::{table_names, BatchOperation, RedbBackend, RedbConfig, StorageEngine};
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

    /// Enable graph persistence to storage
    pub enable_persistence: bool,

    /// Interval for persisting graph state (in events processed)
    pub persistence_interval: u64,

    /// Maximum graph size before cleanup
    pub max_graph_size: usize,

    /// Enable Louvain community detection
    pub enable_louvain: bool,

    /// Interval for running community detection (in events processed)
    pub louvain_interval: u64,

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
    pub(crate) inference: Arc<RwLock<GraphInference>>,

    /// Graph traversal engine
    pub(crate) traversal: Arc<RwLock<GraphTraversal>>,

    /// Event ordering engine for handling concurrent events
    pub(crate) event_ordering: Arc<EventOrderingEngine>,

    /// Scoped inference engine
    pub(crate) scoped_inference: Arc<crate::scoped_inference::ScopedInferenceEngine>,

    /// Episode detector - automatically detects episode boundaries
    pub(crate) episode_detector: Arc<RwLock<EpisodeDetector>>,

    /// Memory store - retrieval substrate (trait object for flexibility)
    pub(crate) memory_store: Arc<RwLock<Box<dyn MemoryStore>>>,

    /// Strategy store - policy substrate (trait object for flexibility)
    pub(crate) strategy_store: Arc<RwLock<Box<dyn StrategyStore>>>,

    /// Transition model - procedural memory spine
    pub(crate) transition_model: Arc<RwLock<TransitionModel>>,

    /// Event storage for episode processing
    pub(crate) event_store: Arc<RwLock<HashMap<agent_db_core::types::EventId, Event>>>,

    /// Insertion order for event_store ring-buffer eviction
    pub(crate) event_store_order: Arc<RwLock<VecDeque<agent_db_core::types::EventId>>>,

    /// Learning decision traces keyed by query_id (lock-free with DashMap for performance)
    pub(crate) decision_traces: Arc<dashmap::DashMap<String, DecisionTrace>>,

    /// Optional storage engine for persistence
    pub(crate) storage: Option<Arc<StorageEngine>>,

    /// Redb backend for graph persistence (shared with memory/strategy stores)
    pub(crate) redb_backend: Option<Arc<RedbBackend>>,

    /// Unified graph store (cold / source of truth) using structures.rs types
    pub(crate) graph_store: Option<Arc<RwLock<crate::redb_graph_store::RedbGraphStore>>>,

    /// Configuration
    pub(crate) config: GraphEngineConfig,

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
            enable_louvain: true,
            louvain_interval: 1000,
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
        // Apply Free Tier overrides if requested via environment
        let config = if std::env::var("SERVICE_PROFILE").unwrap_or_default() == "free" {
            tracing::info!("Applying Free Tier resource limits (0.25 CPU, 768MB RAM cap, No NER)");
            let mut free_config = config;
            free_config.enable_louvain = false;
            free_config.redb_cache_size_mb = 64;
            free_config.memory_cache_size = 1000;
            free_config.strategy_cache_size = 500;
            free_config.ner_workers = 0; // Disable NER workers for free tier
            free_config.claim_workers = 1;
            free_config.embedding_workers = 1;
            free_config
        } else {
            config
        };

        let inference = Arc::new(RwLock::new(GraphInference::with_config(
            config.inference_config.clone(),
        )));

        let traversal = Arc::new(RwLock::new(GraphTraversal::new()));

        let event_ordering = Arc::new(EventOrderingEngine::new(config.ordering_config.clone()));

        let scoped_inference = Arc::new(
            crate::scoped_inference::ScopedInferenceEngine::new(
                config.scoped_inference_config.clone(),
            )
            .await?,
        );

        // Initialize self-evolution components
        // Note: EpisodeDetector requires an Arc<Graph> but doesn't actually use it for detection
        // It uses event-based heuristics instead. We provide an empty graph for API compatibility.
        let graph_for_episodes = Arc::new(Graph::new());
        let episode_detector = Arc::new(RwLock::new(EpisodeDetector::new(
            graph_for_episodes,
            config.episode_config.clone(),
        )));

        // Initialize stores based on storage backend configuration
        let (memory_store, strategy_store, redb_backend): (
            MemoryStoreType,
            StrategyStoreType,
            Option<Arc<RedbBackend>>,
        ) = match config.storage_backend {
            StorageBackend::InMemory => {
                tracing::info!("Initializing with InMemory storage backend");
                let mem = Arc::new(RwLock::new(Box::new(InMemoryMemoryStore::new(
                    config.memory_config.clone(),
                )) as Box<dyn MemoryStore>));
                let strat = Arc::new(RwLock::new(Box::new(InMemoryStrategyStore::new(
                    config.strategy_config.clone(),
                )) as Box<dyn StrategyStore>));
                (mem, strat, None)
            },
            StorageBackend::Persistent => {
                tracing::info!(
                    "Initializing with Persistent storage backend (redb) at {:?}",
                    config.redb_path
                );

                // Initialize redb backend
                let redb_config = RedbConfig {
                    data_path: config.redb_path.clone(),
                    cache_size_bytes: config.redb_cache_size_mb * 1024 * 1024,
                    repair_on_open: false,
                };
                let backend = Arc::new(RedbBackend::open(redb_config).map_err(|e| {
                    GraphError::OperationError(format!("Failed to open redb: {:?}", e))
                })?);

                // Create memory store with LRU cache
                let mut mem_store = RedbMemoryStore::new(
                    backend.clone(),
                    config.memory_config.clone(),
                    config.memory_cache_size.max(1000), // Minimum 1000 entries
                );
                mem_store.initialize().map_err(|e| {
                    GraphError::OperationError(format!(
                        "Failed to initialize memory store: {:?}",
                        e
                    ))
                })?;

                // Create strategy store with LRU cache
                let mut strat_store = RedbStrategyStore::new(
                    backend.clone(),
                    config.strategy_config.clone(),
                    config.strategy_cache_size.max(500), // Minimum 500 entries
                );
                strat_store.initialize().map_err(|e| {
                    GraphError::OperationError(format!(
                        "Failed to initialize strategy store: {:?}",
                        e
                    ))
                })?;

                let mem = Arc::new(RwLock::new(Box::new(mem_store) as Box<dyn MemoryStore>));
                let strat = Arc::new(RwLock::new(Box::new(strat_store) as Box<dyn StrategyStore>));

                tracing::info!(
                    "Persistent storage initialized: memory_cache={}, strategy_cache={}",
                    config.memory_cache_size,
                    config.strategy_cache_size
                );

                (mem, strat, Some(backend))
            },
        };

        let transition_model = Arc::new(RwLock::new(TransitionModel::new(
            TransitionModelConfig::default(),
        )));

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

        // Initialize semantic memory components if enabled
        let (ner_queue, ner_store) = if config.enable_semantic_memory {
            tracing::info!("Initializing semantic memory with NER extraction");

            // Create NER extractor (external service)
            let extractor = Arc::new(
                agent_db_ner::NerServiceExtractor::new(agent_db_ner::NerServiceConfig {
                    base_url: config.ner_service_url.clone(),
                    request_timeout_ms: config.ner_request_timeout_ms,
                    model: config.ner_model.clone(),
                    max_retries: 3,
                    retry_delay_ms: 100,
                })
                .map_err(|e| {
                    GraphError::OperationError(format!("Failed to initialize NER client: {}", e))
                })?,
            );

            // Create NER extraction queue
            let queue = Arc::new(agent_db_ner::NerExtractionQueue::new(
                extractor,
                config.ner_workers,
            ));

            // Create NER storage
            let store = if let Some(path) = &config.ner_storage_path {
                let store = agent_db_ner::NerFeatureStore::new(path).map_err(|e| {
                    GraphError::OperationError(format!("Failed to initialize NER storage: {}", e))
                })?;
                Some(Arc::new(store))
            } else {
                None
            };

            tracing::info!(
                "Semantic memory initialized with {} NER workers",
                config.ner_workers
            );
            (Some(queue), store)
        } else {
            (None, None)
        };

        // Initialize claim extraction components if semantic memory is enabled
        let (claim_queue, claim_store, llm_client, embedding_client) = if config
            .enable_semantic_memory
        {
            tracing::info!("Initializing claim extraction pipeline");

            // Create claim store
            let store = if let Some(path) = &config.claim_storage_path {
                let store = crate::claims::ClaimStore::new(path).map_err(|e| {
                    GraphError::OperationError(format!("Failed to initialize claim storage: {}", e))
                })?;
                Some(Arc::new(store))
            } else {
                None
            };

            // Create LLM client
            let client: Arc<dyn crate::claims::LlmClient> =
                if let Some(key) = &config.openai_api_key {
                    Arc::new(crate::claims::OpenAiClient::new(
                        key.clone(),
                        config.llm_model.clone(),
                    ))
                } else {
                    Arc::new(crate::claims::MockClient::new())
                };

            // Create embedding client
            let embedding_client: Arc<dyn crate::claims::EmbeddingClient> =
                if let Some(key) = &config.openai_api_key {
                    Arc::new(crate::claims::OpenAiEmbeddingClient::new(
                        key.clone(),
                        "text-embedding-3-small".to_string(),
                    ))
                } else {
                    Arc::new(crate::claims::MockEmbeddingClient::new(384))
                };

            // Create claim extraction queue
            let queue = if let Some(ref store) = store {
                let extraction_config = crate::claims::ClaimExtractionConfig {
                    max_claims_per_input: config.claim_max_per_input,
                    min_confidence: config.claim_min_confidence,
                    min_evidence_length: 10,
                    enable_dedup: true,
                    maintenance_config: config.maintenance_config.clone(),
                };

                let queue = Arc::new(crate::claims::ClaimExtractionQueue::new(
                    client.clone(),
                    embedding_client.clone(),
                    store.clone(),
                    config.claim_workers,
                    extraction_config,
                ));
                Some(queue)
            } else {
                None
            };

            tracing::info!(
                "Claim extraction initialized with {} workers",
                config.claim_workers
            );
            (queue, store, Some(client), Some(embedding_client))
        } else {
            (None, None, None, None)
        };

        // Initialize embedding generation components if semantic memory is enabled
        let (embedding_queue, embedding_client) =
            if config.enable_semantic_memory && config.enable_embedding_generation {
                tracing::info!("Initializing embedding generation pipeline");

                // Use the client created above if available, otherwise create a new one
                let client: Arc<dyn crate::claims::EmbeddingClient> =
                    if let Some(ref client) = embedding_client {
                        client.clone()
                    } else if let Some(key) = &config.openai_api_key {
                        Arc::new(crate::claims::OpenAiEmbeddingClient::new(
                            key.clone(),
                            "text-embedding-3-small".to_string(),
                        ))
                    } else {
                        Arc::new(crate::claims::MockEmbeddingClient::new(384))
                    };

                // Create embedding queue if claim store is available
                let queue = if let Some(ref store) = claim_store {
                    let queue = Arc::new(crate::claims::EmbeddingQueue::new(
                        client.clone(),
                        store.clone(),
                        config.embedding_workers,
                    ));
                    Some(queue)
                } else {
                    None
                };

                tracing::info!(
                    "Embedding generation initialized with {} workers",
                    config.embedding_workers
                );
                (queue, Some(client))
            } else {
                (None, embedding_client)
            };

        // 10x/100x: Build consolidation + refinement before config is moved
        let consolidation_engine_arc =
            Arc::new(RwLock::new(crate::consolidation::ConsolidationEngine::new(
                config.consolidation_config.clone(),
                100_000,
            )));
        let refinement_engine_arc = {
            let mut refine_config = config.refinement_config.clone();
            if config.openai_api_key.is_some() {
                refine_config.enable_llm_refinement = true;
            }
            if config.openai_api_key.is_some() || config.enable_semantic_memory {
                refine_config.enable_summary_embedding = true;
            }
            let engine = crate::refinement::RefinementEngine::new(
                refine_config,
                config.openai_api_key.clone(),
            );
            Some(Arc::new(engine))
        };

        // Initialize RedbGraphStore alongside the existing backend
        let graph_store = if let Some(ref backend) = redb_backend {
            let store = crate::redb_graph_store::RedbGraphStore::new(
                backend.clone(),
                8, // max loaded partitions
            );
            tracing::info!("RedbGraphStore initialized for unified graph persistence");
            Some(Arc::new(RwLock::new(store)))
        } else {
            None
        };

        let engine = Self {
            inference,
            traversal,
            event_ordering,
            scoped_inference,
            episode_detector,
            memory_store,
            strategy_store,
            transition_model,
            event_store: Arc::new(RwLock::new(HashMap::new())),
            event_store_order: Arc::new(RwLock::new(VecDeque::new())),
            decision_traces: Arc::new(dashmap::DashMap::new()),
            storage: None,
            redb_backend,
            graph_store,
            config,
            stats: Arc::new(RwLock::new(GraphEngineStats::default())),
            event_buffer: Arc::new(RwLock::new(Vec::new())),
            last_persistence: Arc::new(RwLock::new(0)),
            index_manager,
            louvain,
            centrality,
            ner_queue,
            ner_store,
            claim_queue,
            claim_store,
            llm_client,
            embedding_queue,
            embedding_client,
            // 10x/100x: Consolidation + Refinement — built BEFORE config is moved
            consolidation_engine: consolidation_engine_arc,
            refinement_engine: refinement_engine_arc,
            episodes_since_consolidation: Arc::new(RwLock::new(0)),
        };

        // Restore graph state from redb if available
        match engine.restore_graph_state().await {
            Ok((nodes, edges)) if nodes > 0 || edges > 0 => {
                tracing::info!("Restored graph from disk: {} nodes, {} edges", nodes, edges);
            },
            Ok(_) => {},
            Err(e) => {
                tracing::warn!("Failed to restore graph state (starting fresh): {}", e);
            },
        }

        Ok(engine)
    }

    /// Create a graph engine with storage integration
    pub async fn with_storage(
        config: GraphEngineConfig,
        storage: Arc<StorageEngine>,
    ) -> GraphResult<Self> {
        let mut engine = Self::with_config(config).await?;
        engine.storage = Some(storage);
        Ok(engine)
    }

    /// Get reference to claim store (if semantic memory is enabled)
    pub fn claim_store(&self) -> Option<&Arc<crate::claims::ClaimStore>> {
        self.claim_store.as_ref()
    }

    /// Retrieve memories using hierarchical search (Schema > Semantic > Episodic)
    pub async fn retrieve_memories_hierarchical(
        &self,
        context: &agent_db_events::core::EventContext,
        limit: usize,
        min_similarity: f32,
        agent_id: Option<AgentId>,
    ) -> Vec<Memory> {
        let mut store = self.memory_store.write().await;
        store.retrieve_hierarchical(context, limit, min_similarity, agent_id)
    }

    /// Manually trigger a consolidation pass
    pub async fn run_consolidation(&self) -> crate::consolidation::ConsolidationResult {
        let mut store = self.memory_store.write().await;
        let mut engine = self.consolidation_engine.write().await;
        engine.run_consolidation(store.as_mut())
    }

    /// Process a single event and update the graph
    ///
    /// **Automatic Self-Evolution Pipeline:**
    /// 1. Event ordering and graph construction
    /// 2. Episode detection from event stream
    /// 3. Memory formation from significant episodes
    /// 4. Strategy extraction from successful episodes
    /// 5. Reinforcement learning from outcomes
    ///
    /// **Semantic Memory Control:**
    /// - If `enable_semantic` is Some(true), semantic memory will be processed for this event
    /// - If `enable_semantic` is Some(false), semantic memory will be skipped for this event
    /// - If `enable_semantic` is None, falls back to `config.enable_semantic_memory`
    pub async fn process_event_with_options(
        &self,
        event: Event,
        enable_semantic: Option<bool>,
    ) -> GraphResult<GraphOperationResult> {
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

        // Store event for episode processing (ring-buffer capped)
        {
            let mut store = self.event_store.write().await;
            let mut order = self.event_store_order.write().await;
            store.insert(event.id, event.clone());
            order.push_back(event.id);
            // Evict oldest events when over cap
            let cap = self.config.max_event_store_size;
            while store.len() > cap {
                if let Some(old_id) = order.pop_front() {
                    store.remove(&old_id);
                } else {
                    break;
                }
            }
        }

        // Extract NER features if semantic memory is enabled
        // Check per-request override first, then fall back to config
        let should_extract_semantic = enable_semantic.unwrap_or(self.config.enable_semantic_memory);

        if should_extract_semantic {
            // Claims pipeline calls NER internally and uses the result for
            // entity overlap scoring, LLM prompt grounding, and entity attachment.
            self.extract_claims_async(&event).await;
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
            tracing::info!(
                "Acquiring inference write lock for event {}",
                ready_event.id
            );
            let nodes_result = {
                let mut inference = self.inference.write().await;
                inference.process_event(ready_event.clone())
            };
            match nodes_result {
                Ok(nodes) => {
                    result.nodes_created.extend(nodes.clone());

                    // Auto-index newly created nodes
                    tracing::info!(
                        "Auto-index start event_id={} nodes={}",
                        ready_event.id,
                        nodes.len()
                    );
                    self.auto_index_nodes(&nodes).await?;
                    tracing::info!("Auto-index done event_id={}", ready_event.id);
                },
                Err(e) => {
                    result.errors.push(e);
                },
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
            if let Err(e) = self
                .scoped_inference
                .process_scoped_event(scoped_event)
                .await
            {
                result.errors.push(e);
            }

            if let Err(e) = self.handle_learning_event(&ready_event).await {
                result.errors.push(e);
            }

            // Step 3: Self-Evolution Pipeline - Episode Detection
            if self.config.auto_episode_detection {
                // Check for completed episodes (must drop write lock before acquiring read lock)
                let episode_update = {
                    self.episode_detector
                        .write()
                        .await
                        .process_event(&ready_event)
                };

                if let Some(episode_update) = episode_update {
                    let (episode_id, is_correction) = match episode_update {
                        crate::episodes::EpisodeUpdate::Completed(id) => (id, false),
                        crate::episodes::EpisodeUpdate::Corrected(id) => (id, true),
                    };
                    result.patterns_detected.push(format!(
                        "{}_{}",
                        if is_correction {
                            "episode_corrected"
                        } else {
                            "episode_completed"
                        },
                        episode_id
                    ));

                    let episodes: Vec<Episode> = self
                        .episode_detector
                        .read()
                        .await
                        .get_completed_episodes()
                        .to_vec();
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
            result
                .patterns_detected
                .push("event_reordering_occurred".to_string());
        }

        // Update statistics
        {
            let mut stats = self.stats.write().await;
            stats.total_events_processed += 1;
            stats.total_nodes_created += result.nodes_created.len() as u64;
            stats.last_operation_time = std::time::Instant::now();

            // Update average processing time
            let processing_time = start_time.elapsed().as_millis() as f64;
            stats.average_processing_time_ms = (stats.average_processing_time_ms
                * (stats.total_events_processed as f64 - 1.0)
                + processing_time)
                / stats.total_events_processed as f64;

            // Run Louvain community detection periodically
            if self.config.enable_louvain
                && stats.total_events_processed % self.config.louvain_interval == 0
            {
                drop(stats); // Release stats lock before async operation
                if let Err(e) = self.run_community_detection().await {
                    result.errors.push(e);
                } else {
                    result
                        .patterns_detected
                        .push("louvain_communities_updated".to_string());
                }
            } else {
                drop(stats); // Always release the lock
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

        // Persist graph to redb when the backend is available
        if self.redb_backend.is_some() {
            let stats = self.stats.read().await;
            let last_persistence = *self.last_persistence.read().await;

            if stats.total_events_processed - last_persistence >= self.config.persistence_interval {
                drop(stats);
                if let Err(e) = self.persist_graph_state().await {
                    tracing::warn!("Graph persistence failed (will retry next interval): {}", e);
                }
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
            EventType::Context { .. } => "Context",
        }
    }

    async fn handle_learning_event(&self, event: &Event) -> GraphResult<()> {
        let EventType::Learning {
            event: learning_event,
        } = &event.event_type
        else {
            return Ok(());
        };

        let now = std::time::Instant::now();
        match learning_event {
            LearningEvent::MemoryRetrieved {
                query_id,
                memory_ids,
            } => {
                // DashMap provides lock-free concurrent access
                let mut trace =
                    self.decision_traces
                        .entry(query_id.clone())
                        .or_insert(DecisionTrace {
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
            },
            LearningEvent::MemoryUsed {
                query_id,
                memory_id,
            } => {
                // DashMap provides lock-free concurrent access
                let mut trace =
                    self.decision_traces
                        .entry(query_id.clone())
                        .or_insert(DecisionTrace {
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
            },
            LearningEvent::StrategyServed {
                query_id,
                strategy_ids,
            } => {
                // DashMap provides lock-free concurrent access
                let mut trace =
                    self.decision_traces
                        .entry(query_id.clone())
                        .or_insert(DecisionTrace {
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
            },
            LearningEvent::StrategyUsed {
                query_id,
                strategy_id,
            } => {
                // DashMap provides lock-free concurrent access
                let mut trace =
                    self.decision_traces
                        .entry(query_id.clone())
                        .or_insert(DecisionTrace {
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
            },
            LearningEvent::Outcome { query_id, success } => {
                tracing::info!(
                    "Learning telemetry outcome query_id={} success={}",
                    query_id,
                    success
                );
                self.apply_learning_outcome(query_id, *success).await?;
            },
        }

        Ok(())
    }

    async fn apply_learning_outcome(&self, query_id: &str, success: bool) -> GraphResult<()> {
        // DashMap provides lock-free concurrent access
        let trace = self.decision_traces.remove(query_id).map(|(_, v)| v);

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

    /// Process a single event (convenience wrapper that uses config default for semantic memory)
    pub async fn process_event(&self, event: Event) -> GraphResult<GraphOperationResult> {
        self.process_event_with_options(event, None).await
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
            },
            Err(e) => {
                combined_result.errors.push(e);
            },
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
        {
            let _inference = self.inference.read().await;
            // We need a way to get a reference to the graph
            // For now, we'll execute queries directly through traversal
        }

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

    /// Search nodes using BM25 full-text search
    ///
    /// Returns a list of (NodeId, score) tuples ranked by relevance
    pub async fn search_bm25(&self, query: &str, limit: usize) -> Vec<(u64, f32)> {
        let inference = self.inference.read().await;
        inference.graph().bm25_index.search(query, limit)
    }

    /// Get a node by ID for search results
    pub async fn get_node(&self, node_id: u64) -> Option<crate::structures::GraphNode> {
        let inference = self.inference.read().await;
        inference.graph().nodes.get(&node_id).cloned()
    }

    /// Search claims by semantic similarity for hybrid search
    ///
    /// Returns a list of (NodeId, score) tuples for claims ranked by semantic similarity
    pub async fn search_claims_semantic(
        &self,
        query: &str,
        limit: usize,
        min_similarity: f32,
    ) -> crate::GraphResult<Vec<(u64, f32)>> {
        // Check if embedding client is available
        let embedding_client = match &self.embedding_client {
            Some(c) => c,
            None => return Ok(vec![]), // No semantic search available
        };

        let claim_store = match &self.claim_store {
            Some(s) => s,
            None => return Ok(vec![]), // No claim store available
        };

        // Generate embedding for query
        let request = crate::claims::EmbeddingRequest {
            text: query.to_string(),
            context: None,
        };

        let response = embedding_client.embed(request).await.map_err(|e| {
            crate::GraphError::OperationError(format!("Failed to generate query embedding: {}", e))
        })?;

        // Search for similar claims
        let similar_claims = claim_store
            .find_similar(&response.embedding, limit, min_similarity)
            .map_err(|e| {
                crate::GraphError::OperationError(format!("Failed to search claims: {}", e))
            })?;

        // Convert claim IDs to node IDs
        let inference = self.inference.read().await;
        let graph = inference.graph();

        let mut results = Vec::new();
        for (claim_id, similarity) in similar_claims {
            if let Some(&node_id) = graph.claim_index.get(&claim_id) {
                results.push((node_id, similarity));
            }
        }

        Ok(results)
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

    /// Persist current graph state to redb storage.
    ///
    /// Serializes every node, edge, and graph metadata into an atomic
    /// `write_batch` so the on-disk representation is always consistent.
    /// Returns the number of nodes and edges persisted.
    pub async fn persist_graph_state(&self) -> GraphResult<(usize, usize)> {
        let backend = match self.redb_backend {
            Some(ref b) => b.clone(),
            None => {
                // No persistent backend — nothing to do
                return Ok((0, 0));
            },
        };

        let inference = self.inference.read().await;
        let graph = inference.graph();

        let mut ops: Vec<BatchOperation> = Vec::with_capacity(
            graph.nodes.len() + graph.edges.len() + 1, // +1 for metadata
        );

        // Serialize each node: key = "n" + node_id (big-endian)
        for (id, node) in &graph.nodes {
            let mut key = Vec::with_capacity(9);
            key.push(b'n');
            key.extend_from_slice(&id.to_be_bytes());
            let value = bincode::serialize(node).map_err(|e| {
                GraphError::OperationError(format!("Failed to serialize node {}: {}", id, e))
            })?;
            ops.push(BatchOperation::Put {
                table_name: table_names::GRAPH_NODES.to_string(),
                key,
                value,
            });
        }

        // Serialize each edge: key = "e" + edge_id (big-endian)
        for (id, edge) in &graph.edges {
            let mut key = Vec::with_capacity(9);
            key.push(b'e');
            key.extend_from_slice(&id.to_be_bytes());
            let value = bincode::serialize(edge).map_err(|e| {
                GraphError::OperationError(format!("Failed to serialize edge {}: {}", id, e))
            })?;
            ops.push(BatchOperation::Put {
                table_name: table_names::GRAPH_EDGES.to_string(),
                key,
                value,
            });
        }

        // Persist adjacency lists and graph metadata in a single metadata blob.
        // This keeps the batch atomic and avoids needing extra tables.
        #[derive(serde::Serialize, serde::Deserialize)]
        struct GraphMeta {
            adjacency_out: HashMap<NodeId, Vec<EdgeId>>,
            adjacency_in: HashMap<NodeId, Vec<EdgeId>>,
            next_node_id: NodeId,
            next_edge_id: EdgeId,
            stats: GraphStats,
        }

        let meta = GraphMeta {
            adjacency_out: graph.adjacency_out.clone(),
            adjacency_in: graph.adjacency_in.clone(),
            next_node_id: graph.next_node_id,
            next_edge_id: graph.next_edge_id,
            stats: graph.stats.clone(),
        };
        let meta_value = bincode::serialize(&meta).map_err(|e| {
            GraphError::OperationError(format!("Failed to serialize graph metadata: {}", e))
        })?;
        ops.push(BatchOperation::Put {
            table_name: table_names::GRAPH_ADJACENCY.to_string(),
            key: b"__meta__".to_vec(),
            value: meta_value,
        });

        let node_count = graph.nodes.len();
        let edge_count = graph.edges.len();

        // Release the read lock before the blocking I/O
        drop(inference);

        backend.write_batch(ops).map_err(|e| {
            GraphError::OperationError(format!("Failed to persist graph state: {:?}", e))
        })?;

        // Update persistence checkpoint
        let engine_stats = self.stats.read().await;
        *self.last_persistence.write().await = engine_stats.total_events_processed;

        tracing::info!(
            "Graph persisted: {} nodes, {} edges written to redb",
            node_count,
            edge_count
        );

        Ok((node_count, edge_count))
    }

    /// Restore graph state from redb on startup.
    ///
    /// Reads all persisted nodes, edges, and metadata and rebuilds the
    /// in-memory `Graph`, including all secondary indexes.
    pub async fn restore_graph_state(&self) -> GraphResult<(usize, usize)> {
        let backend = match self.redb_backend {
            Some(ref b) => b.clone(),
            None => return Ok((0, 0)),
        };

        // Load metadata first — if it doesn't exist, there's nothing to restore
        #[derive(serde::Serialize, serde::Deserialize)]
        struct GraphMeta {
            adjacency_out: HashMap<NodeId, Vec<EdgeId>>,
            adjacency_in: HashMap<NodeId, Vec<EdgeId>>,
            next_node_id: NodeId,
            next_edge_id: EdgeId,
            stats: GraphStats,
        }

        let meta: GraphMeta = match backend
            .get(table_names::GRAPH_ADJACENCY, b"__meta__")
            .map_err(|e| {
                GraphError::OperationError(format!("Failed to read graph metadata: {:?}", e))
            })? {
            Some(m) => m,
            None => {
                tracing::info!("No persisted graph state found — starting fresh");
                return Ok((0, 0));
            },
        };

        // Load all nodes (keys prefixed with 'n')
        let raw_nodes: Vec<(Vec<u8>, GraphNode)> = backend
            .scan_prefix(table_names::GRAPH_NODES, b"n")
            .map_err(|e| {
                GraphError::OperationError(format!("Failed to scan graph nodes: {:?}", e))
            })?;

        // Load all edges (keys prefixed with 'e')
        let raw_edges: Vec<(Vec<u8>, GraphEdge)> = backend
            .scan_prefix(table_names::GRAPH_EDGES, b"e")
            .map_err(|e| {
                GraphError::OperationError(format!("Failed to scan graph edges: {:?}", e))
            })?;

        let node_count = raw_nodes.len();
        let edge_count = raw_edges.len();

        // Rebuild the graph under the inference write lock
        let mut inference = self.inference.write().await;
        let graph = inference.graph_mut();

        // Clear existing data
        graph.nodes.clear();
        graph.edges.clear();
        graph.adjacency_out = meta.adjacency_out;
        graph.adjacency_in = meta.adjacency_in;
        graph.next_node_id = meta.next_node_id;
        graph.next_edge_id = meta.next_edge_id;
        graph.stats = meta.stats;

        // Clear all secondary indexes before rebuilding
        graph.type_index.clear();
        graph.context_index.clear();
        graph.agent_index.clear();
        graph.event_index.clear();
        graph.goal_index.clear();
        graph.episode_index.clear();
        graph.memory_index.clear();
        graph.strategy_index.clear();
        graph.tool_index.clear();
        graph.result_index.clear();
        graph.claim_index.clear();
        graph.concept_index.clear();

        // Insert nodes and rebuild indexes
        for (_key, node) in raw_nodes {
            let node_id = node.id;

            // Rebuild type index
            let type_name = node.type_name();
            graph
                .type_index
                .entry(type_name)
                .or_default()
                .insert(node_id);

            // Rebuild specialized indexes
            match &node.node_type {
                NodeType::Agent { agent_id, .. } => {
                    graph.agent_index.insert(*agent_id, node_id);
                },
                NodeType::Event { event_id, .. } => {
                    graph.event_index.insert(*event_id, node_id);
                },
                NodeType::Context { context_hash, .. } => {
                    graph.context_index.insert(*context_hash, node_id);
                },
                NodeType::Goal { goal_id, .. } => {
                    graph.goal_index.insert(*goal_id, node_id);
                },
                NodeType::Episode { episode_id, .. } => {
                    graph.episode_index.insert(*episode_id, node_id);
                },
                NodeType::Memory { memory_id, .. } => {
                    graph.memory_index.insert(*memory_id, node_id);
                },
                NodeType::Strategy { strategy_id, .. } => {
                    graph.strategy_index.insert(*strategy_id, node_id);
                },
                NodeType::Tool { tool_name, .. } => {
                    graph.tool_index.insert(tool_name.clone(), node_id);
                },
                NodeType::Result { result_key, .. } => {
                    graph.result_index.insert(result_key.clone(), node_id);
                },
                NodeType::Claim { claim_id, .. } => {
                    graph.claim_index.insert(*claim_id, node_id);
                },
                NodeType::Concept { concept_name, .. } => {
                    graph.concept_index.insert(concept_name.clone(), node_id);
                },
            }

            // Rebuild BM25 index
            let mut text_parts = Vec::new();
            match &node.node_type {
                NodeType::Claim { claim_text, .. } => text_parts.push(claim_text.as_str()),
                NodeType::Goal { description, .. } => text_parts.push(description.as_str()),
                NodeType::Strategy { name, .. } => text_parts.push(name.as_str()),
                NodeType::Result { summary, .. } => text_parts.push(summary.as_str()),
                NodeType::Concept { concept_name, .. } => text_parts.push(concept_name.as_str()),
                NodeType::Tool { tool_name, .. } => text_parts.push(tool_name.as_str()),
                NodeType::Episode { outcome, .. } => text_parts.push(outcome.as_str()),
                _ => {},
            }
            for (key, value) in &node.properties {
                let key_lower = key.to_lowercase();
                if key_lower.contains("text")
                    || key_lower.contains("description")
                    || key_lower.contains("content")
                    || key_lower.contains("name")
                    || key_lower.contains("summary")
                    || key_lower == "data"
                {
                    if let Some(text) = value.as_str() {
                        text_parts.push(text);
                    }
                }
            }
            if !text_parts.is_empty() {
                let combined_text = text_parts.join(" ");
                graph.bm25_index.index_document(node_id, &combined_text);
            }

            graph.nodes.insert(node_id, node);
        }

        // Insert edges
        for (_key, edge) in raw_edges {
            graph.edges.insert(edge.id, edge);
        }

        tracing::info!(
            "Graph restored from redb: {} nodes, {} edges",
            node_count,
            edge_count
        );

        Ok((node_count, edge_count))
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

    /// Graceful shutdown: flush all buffers, drain queues, and sync storage.
    ///
    /// Call this before dropping the engine to ensure all in-flight work is
    /// committed to redb. After this returns the process can exit safely.
    pub async fn shutdown(&self) {
        tracing::info!("GraphEngine shutdown initiated — flushing buffers");

        // 1. Flush event ordering buffers (process any queued out-of-order events)
        if let Err(e) = self.flush_all_buffers().await {
            tracing::warn!("Error flushing event buffers during shutdown: {}", e);
        }

        // 2. Process any pending claim extraction jobs by dropping the queue sender.
        //    Workers will drain remaining items and exit when the channel closes.
        //    We can't drop Arc fields, but the server dropping AppState after this
        //    will trigger cleanup. Log the intent so operators know.
        if self.claim_queue.is_some() {
            tracing::info!("Claim extraction queue will drain on drop");
        }
        if self.embedding_queue.is_some() {
            tracing::info!("Embedding queue will drain on drop");
        }
        if self.ner_queue.is_some() {
            tracing::info!("NER extraction queue will drain on drop");
        }

        // 3. Persist graph state to redb before shutdown
        if self.redb_backend.is_some() {
            match self.persist_graph_state().await {
                Ok((n, e)) => {
                    tracing::info!("Graph persisted on shutdown: {} nodes, {} edges", n, e)
                },
                Err(e) => tracing::warn!("Failed to persist graph during shutdown: {}", e),
            }
        }

        // 4. Sync storage engine if present (flushes any buffered segment writes)
        if let Some(ref storage) = self.storage {
            if let Err(e) = storage.sync().await {
                tracing::warn!("Error syncing storage engine during shutdown: {}", e);
            }
        }

        tracing::info!("GraphEngine shutdown complete — all buffers flushed, graph persisted");
    }

    /// Spawn a background maintenance loop that periodically runs:
    /// 1. Memory decay (age out stale memories)
    /// 2. Strategy pruning (remove weak + merge near-duplicates)
    ///
    /// The loop respects `maintenance_config.interval_secs`.
    /// Pass `0` to disable.
    ///
    /// The returned `JoinHandle` can be used to abort the loop on shutdown.
    pub fn start_maintenance_loop(self: &Arc<Self>) -> Option<tokio::task::JoinHandle<()>> {
        let interval_secs = self.config.maintenance_config.interval_secs;
        if interval_secs == 0 {
            tracing::info!("Maintenance loop disabled (interval_secs=0)");
            return None;
        }

        let engine = Arc::clone(self);
        let handle = tokio::spawn(async move {
            let mut ticker = tokio::time::interval(tokio::time::Duration::from_secs(interval_secs));
            // Skip the first immediate tick
            ticker.tick().await;

            tracing::info!("Maintenance loop started (interval={}s)", interval_secs);

            loop {
                ticker.tick().await;
                tracing::debug!("Maintenance pass starting");

                // 1. Memory decay
                engine.memory_store.write().await.apply_decay();

                // 2. Strategy pruning
                let mc = &engine.config.maintenance_config;
                let pruned = engine.strategy_store.write().await.prune_strategies(
                    mc.strategy_min_confidence,
                    mc.strategy_min_support,
                    mc.strategy_max_stale_hours,
                );

                if pruned > 0 {
                    tracing::info!("Maintenance: pruned {} strategies", pruned);
                }

                // 3. Graph pruning (streaming, bounded, via RedbGraphStore)
                if let Some(ref graph_store) = engine.graph_store {
                    let now = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap()
                        .as_nanos() as u64;

                    let pruner = crate::graph_pruning::GraphPruner::new(
                        engine.config.pruning_config.clone(),
                    );

                    let mut store = graph_store.write().await;
                    let mut inference = engine.inference.write().await;
                    let graph = inference.graph_mut();

                    match pruner.prune_full_graph(graph, &mut store, now) {
                        Ok(result) => {
                            if result.nodes_merged > 0 || result.nodes_deleted > 0 {
                                tracing::info!(
                                    "Maintenance: graph pruning merged={} deleted={} scanned={} stopped_early={}",
                                    result.nodes_merged,
                                    result.nodes_deleted,
                                    result.total_headers_scanned,
                                    result.stopped_early,
                                );
                            }
                        },
                        Err(e) => {
                            tracing::warn!("Graph pruning failed: {}", e);
                        },
                    }
                }

                // 4. Transition model cleanup
                {
                    let mut tm = engine.transition_model.write().await;
                    let ep_count_before = tm.episode_count();
                    tm.cleanup_oldest_episodes(engine.config.max_transition_episodes);
                    let ep_cleaned = ep_count_before - tm.episode_count();

                    tm.prune_weak_transitions(engine.config.min_transition_count);

                    if ep_cleaned > 0 {
                        tracing::info!(
                            "Maintenance: transition cleanup episodes_removed={}",
                            ep_cleaned,
                        );
                    }
                }

                // 5. Event store ring-buffer eviction (safety net)
                {
                    let mut store = engine.event_store.write().await;
                    let mut order = engine.event_store_order.write().await;
                    let cap = engine.config.max_event_store_size;
                    let mut evicted = 0usize;
                    while store.len() > cap {
                        if let Some(old_id) = order.pop_front() {
                            store.remove(&old_id);
                            evicted += 1;
                        } else {
                            break;
                        }
                    }
                    if evicted > 0 {
                        tracing::info!("Maintenance: event_store evicted {} entries", evicted);
                    }
                }

                // 6. Decision trace TTL sweep
                {
                    let max_age =
                        std::time::Duration::from_secs(engine.config.max_decision_trace_age_secs);
                    let before = engine.decision_traces.len();
                    engine
                        .decision_traces
                        .retain(|_, trace| trace.last_updated.elapsed() < max_age);
                    let swept = before - engine.decision_traces.len();
                    if swept > 0 {
                        tracing::info!(
                            "Maintenance: decision_traces TTL swept {} stale entries",
                            swept
                        );
                    }
                }

                // 7. Inference memory caps (context cache + temporal patterns)
                {
                    let mut inf = engine.inference.write().await;
                    inf.enforce_memory_caps();
                }

                // 8. Claim store maintenance (expire stale, cap vector index, purge disk)
                if let Some(ref claim_store) = engine.claim_store {
                    // Expire claims past their TTL
                    match claim_store.expire_stale_claims() {
                        Ok(expired) if expired > 0 => {
                            tracing::info!("Maintenance: expired {} stale claims", expired);
                        },
                        Err(e) => {
                            tracing::warn!("Maintenance: claim expiry failed: {}", e);
                        },
                        _ => {},
                    }

                    // Enforce vector index cap
                    let max_idx = mc.max_vector_index_size;
                    if max_idx > 0 {
                        if let Err(e) = claim_store.enforce_vector_index_cap(max_idx) {
                            tracing::warn!("Maintenance: vector index cap failed: {}", e);
                        }
                    }

                    // Purge inactive claims from disk
                    if mc.purge_inactive_claims {
                        match claim_store.purge_inactive_claims() {
                            Ok(purged) if purged > 0 => {
                                tracing::info!(
                                    "Maintenance: purged {} inactive claims from disk",
                                    purged
                                );
                            },
                            Err(e) => {
                                tracing::warn!("Maintenance: claim purge failed: {}", e);
                            },
                            _ => {},
                        }
                    }
                }

                tracing::debug!("Maintenance pass complete");
            }
        });

        Some(handle)
    }

    /// Get scoped inference statistics
    pub async fn get_scoped_inference_stats(&self) -> crate::scoped_inference::ScopeStatistics {
        self.scoped_inference.get_scope_statistics().await
    }

    /// Query events in a specific scope
    pub async fn query_events_in_scope(
        &self,
        scope: &crate::scoped_inference::InferenceScope,
        query: crate::scoped_inference::ScopeQuery,
    ) -> GraphResult<crate::scoped_inference::ScopeQueryResult> {
        self.scoped_inference.query_scope(scope, query).await
    }

    /// Get cross-scope relationships
    pub async fn get_cross_scope_relationships(
        &self,
    ) -> crate::scoped_inference::CrossScopeInsights {
        self.scoped_inference.get_cross_scope_insights().await
    }

    // ============================================================================
    // Self-Evolution Pipeline Methods
    // ============================================================================

    /// Process episode for memory formation
    async fn process_episode_for_memory(&self, episode: &Episode) -> GraphResult<()> {
        // Load events for summary generation
        let events: Vec<agent_db_events::core::Event> = {
            let store = self.event_store.read().await;
            episode
                .events
                .iter()
                .filter_map(|event_id| store.get(event_id).cloned())
                .collect()
        };

        let mut memory_store = self.memory_store.write().await;

        tracing::info!(
            "Memory formation start episode_id={} agent_id={} session_id={} significance={:.3} outcome={:?}",
            episode.id,
            episode.agent_id,
            episode.session_id,
            episode.significance,
            episode.outcome
        );
        if let Some(upsert) = memory_store.store_episode(episode, &events) {
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
                    "Memory formed id={} episode_id={} strength={:.3} relevance={:.3} context_hash={} tier={:?}",
                    upsert.id,
                    episode.id,
                    memory.strength,
                    memory.relevance_score,
                    memory.context.fingerprint,
                    memory.tier
                );
                self.attach_memory_to_graph(episode, &memory).await?;

                // Fire-and-forget: async LLM refinement + embedding
                if let Some(ref refinement) = self.refinement_engine {
                    let memory_id = upsert.id;
                    let store_ref = self.memory_store.clone();
                    let refinement_ref = refinement.clone();
                    let embedding_client = self.embedding_client.clone();
                    tokio::spawn(async move {
                        if let Err(e) = refinement_ref
                            .refine_and_embed_memory(
                                memory_id,
                                &store_ref,
                                embedding_client.as_ref(),
                            )
                            .await
                        {
                            tracing::warn!("Memory refinement failed for {}: {}", memory_id, e);
                        }
                    });
                }
            }

            // Check if we should run consolidation
            let should_consolidate = {
                let mut counter = self.episodes_since_consolidation.write().await;
                *counter += 1;
                *counter >= self.config.consolidation_interval
            };
            if should_consolidate {
                *self.episodes_since_consolidation.write().await = 0;
                let store_ref = self.memory_store.clone();
                let engine_ref = self.consolidation_engine.clone();
                tokio::spawn(async move {
                    let mut store = store_ref.write().await;
                    let mut engine = engine_ref.write().await;
                    let result = engine.run_consolidation(store.as_mut());
                    if result.semantic_created > 0 || result.schema_created > 0 {
                        tracing::info!(
                            "Consolidation pass: {} semantic created, {} schemas created, {} episodes consolidated",
                            result.semantic_created,
                            result.schema_created,
                            result.consolidated_episode_ids.len()
                        );
                    }
                });
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
            episode
                .events
                .iter()
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
            if let (Some(start_event), Some(end_event_id)) =
                (store.get(&episode.start_event), episode.end_event)
            {
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
        let _result = inference
            .reinforce_patterns(episode, success, Some(metrics))
            .await?;

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

        let mut suggestions = traversal.get_next_step_suggestions(
            graph,
            context_hash,
            last_action_node,
            limit * 2,
        )?;

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
        self.memory_store
            .read()
            .await
            .get_agent_memories(agent_id, limit)
    }

    /// Retrieve memories by context similarity
    pub async fn retrieve_memories_by_context(
        &self,
        context: &agent_db_events::core::EventContext,
        limit: usize,
    ) -> Vec<Memory> {
        self.memory_store
            .write()
            .await
            .retrieve_by_context(context, limit)
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
        self.memory_store.write().await.retrieve_by_context_similar(
            context,
            limit,
            min_similarity,
            agent_id,
            session_id,
        )
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
        self.episode_detector
            .read()
            .await
            .get_completed_episodes()
            .to_vec()
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
        self.strategy_store
            .write()
            .await
            .update_strategy_outcome(strategy_id, success)
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
        tracing::info!(
            "auto_index_nodes acquiring inference read lock (nodes={})",
            node_ids.len()
        );
        let inference = match timeout(Duration::from_secs(2), self.inference.read()).await {
            Ok(lock) => lock,
            Err(_) => {
                tracing::info!("auto_index_nodes timeout acquiring inference read lock");
                return Err(GraphError::OperationError(
                    "auto_index_nodes timeout acquiring inference read lock".to_string(),
                ));
            },
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
                node.properties
                    .insert("community_id".to_string(), json!(community_id));
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
    pub async fn detect_communities(
        &self,
    ) -> GraphResult<crate::algorithms::CommunityDetectionResult> {
        let inference = self.inference.read().await;
        let graph = inference.graph();

        self.louvain.detect_communities(graph)
    }

    /// Get centrality scores for all nodes
    pub async fn get_all_centrality_scores(
        &self,
    ) -> GraphResult<crate::algorithms::AllCentralities> {
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
                        event_store.iter().filter(|(_, event)| {
                            event.session_id == session_id && event.agent_type == agent_type
                        }),
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
                event_store
                    .iter()
                    .filter(|(_, event)| event.session_id == session_id),
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
                        event_store.iter().filter(|(_, event)| {
                            event.session_id == session_id && event.agent_type == agent_type
                        }),
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
                event_store
                    .iter()
                    .filter(|(_, event)| event.session_id == session_id),
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
                    EventType::Observation {
                        observation_type, ..
                    } => observation_type.clone(),
                    EventType::Cognitive { process_type, .. } => format!("{:?}", process_type),
                    EventType::Learning { .. } => "Learning".to_string(),
                    _ => format!("Event {}", event.id),
                };

                nodes.insert(node.id, Self::build_graph_node_data(node, Some(label)));

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
                        EdgeType::DerivedFrom { .. } => "DerivedFrom",
                        EdgeType::SupportedBy { .. } => "SupportedBy",
                        EdgeType::About { .. } => "About",
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
                        nodes
                            .entry(target_node.id)
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
            NodeType::Event { event_type, .. } => {
                (format!("Event::{}", event_type), label_override)
            },
            NodeType::Context { .. } => ("Context".to_string(), label_override),
            NodeType::Agent { .. } => ("Agent".to_string(), label_override),
            NodeType::Goal { description, .. } => ("Goal".to_string(), Some(description.clone())),
            NodeType::Episode { episode_id, .. } => (
                "Episode".to_string(),
                Some(format!("Episode {}", episode_id)),
            ),
            NodeType::Memory { memory_id, .. } => {
                ("Memory".to_string(), Some(format!("Memory {}", memory_id)))
            },
            NodeType::Strategy { name, .. } => ("Strategy".to_string(), Some(name.clone())),
            NodeType::Tool { tool_name, .. } => ("Tool".to_string(), Some(tool_name.clone())),
            NodeType::Result { summary, .. } => ("Result".to_string(), Some(summary.clone())),
            NodeType::Concept { concept_name, .. } => {
                ("Concept".to_string(), Some(concept_name.clone()))
            },
            NodeType::Claim { claim_text, .. } => ("Claim".to_string(), Some(claim_text.clone())),
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
                EventType::Observation {
                    observation_type, ..
                } => observation_type.clone(),
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
            },
        };

        let mut nodes: HashMap<NodeId, GraphNodeData> = HashMap::new();
        let mut edges: Vec<GraphEdgeData> = Vec::new();

        nodes.insert(
            context_node.id,
            Self::build_graph_node_data(context_node, None),
        );

        let mut candidate_edges = Vec::new();
        candidate_edges.extend(graph.get_edges_from(context_node.id));
        candidate_edges.extend(graph.get_edges_to(context_node.id));

        for (edge_count, edge) in candidate_edges.into_iter().enumerate() {
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
                EdgeType::DerivedFrom { .. } => "DerivedFrom",
                EdgeType::SupportedBy { .. } => "SupportedBy",
                EdgeType::About { .. } => "About",
            };

            edges.push(GraphEdgeData {
                id: edge.id,
                from: edge.source,
                to: edge.target,
                edge_type: edge_type.to_string(),
                weight: edge.weight,
                confidence: edge.confidence,
            });

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

    async fn attach_strategy_to_graph(
        &self,
        episode: &Episode,
        strategy: &Strategy,
    ) -> GraphResult<()> {
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
            let result_node_id =
                Self::ensure_result_node(graph, &result_key, &result_type, &result_type);
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
            node.properties
                .insert("outcome".to_string(), json!(outcome));
            node.properties
                .insert("event_count".to_string(), json!(episode.events.len()));
            node.properties
                .insert("significance".to_string(), json!(episode.significance));
            node.properties
                .insert("salience_score".to_string(), json!(episode.salience_score));
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
            node.properties
                .insert("strength".to_string(), json!(memory.strength));
            node.properties
                .insert("relevance_score".to_string(), json!(memory.relevance_score));
            node.properties.insert(
                "context_hash".to_string(),
                json!(memory.context.fingerprint),
            );
            node.properties
                .insert("formed_at".to_string(), json!(memory.formed_at));
            node.properties.insert(
                "memory_type".to_string(),
                json!(format!("{:?}", memory.memory_type)),
            );
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
            node.properties
                .insert("quality_score".to_string(), json!(strategy.quality_score));
            node.properties
                .insert("success_count".to_string(), json!(strategy.success_count));
            node.properties
                .insert("failure_count".to_string(), json!(strategy.failure_count));
            node.properties
                .insert("version".to_string(), json!(strategy.version));
            node.properties.insert(
                "strategy_type".to_string(),
                json!(format!("{:?}", strategy.strategy_type)),
            );
            node.properties
                .insert("support_count".to_string(), json!(strategy.support_count));
            node.properties.insert(
                "expected_success".to_string(),
                json!(strategy.expected_success),
            );
            node.properties
                .insert("expected_cost".to_string(), json!(strategy.expected_cost));
            node.properties
                .insert("expected_value".to_string(), json!(strategy.expected_value));
            node.properties
                .insert("confidence".to_string(), json!(strategy.confidence));
            node.properties
                .insert("goal_bucket_id".to_string(), json!(strategy.goal_bucket_id));
            node.properties.insert(
                "behavior_signature".to_string(),
                json!(strategy.behavior_signature),
            );
            node.properties
                .insert("precondition".to_string(), json!(strategy.precondition));
            node.properties
                .insert("action_hint".to_string(), json!(strategy.action_hint));
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
            node.properties
                .insert("progress".to_string(), json!(goal.progress));
            if let Some(deadline) = goal.deadline {
                node.properties
                    .insert("deadline".to_string(), json!(deadline));
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

    fn ensure_result_node(
        graph: &mut Graph,
        result_key: &str,
        result_type: &str,
        summary: &str,
    ) -> NodeId {
        if let Some(node) = graph.get_result_node(result_key) {
            node.id
        } else {
            let mut node = GraphNode::new(NodeType::Result {
                result_key: result_key.to_string(),
                result_type: result_type.to_string(),
                summary: summary.to_string(),
            });
            node.properties
                .insert("summary".to_string(), json!(summary));
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
