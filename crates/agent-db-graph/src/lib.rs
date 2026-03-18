//! Graph operations and inference for agent database
//!
//! This crate provides graph construction and analysis capabilities
//! for the agent database, inferring relationships from event patterns.
//!
#![allow(
    clippy::too_many_arguments,
    clippy::large_enum_variant,
    clippy::type_complexity,
    clippy::approx_constant
)]
//! ## New Features (2026-01-15)
//! - Property indexing for 100-1000x faster queries
//! - Louvain community detection for memory clustering
//! - Centrality measures for identifying important actions
//! - Parallel processing for 4-8x speedup
//! - Learning-focused analytics

pub mod error {
    //! Graph-specific error types
    use thiserror::Error;

    #[derive(Error, Debug)]
    pub enum GraphError {
        #[error("Node not found: {0}")]
        NodeNotFound(String),

        #[error("Edge not found: {0}")]
        EdgeNotFound(String),

        #[error("Cycle detected in graph")]
        CycleDetected,

        #[error("Graph operation error: {0}")]
        OperationError(String),

        #[error("Invalid query: {0}")]
        InvalidQuery(String),

        #[error("Invalid operation: {0}")]
        InvalidOperation(String),

        #[error("Graph capacity exceeded: {0}")]
        CapacityExceeded(String),
    }

    pub type GraphResult<T> = Result<T, GraphError>;
}

pub mod catalog;
pub mod contracts;
pub mod decision_trace;
pub mod episodes;
pub mod event_ordering;
pub mod inference;
pub mod integration;
pub mod integration_claims;
pub mod learning;
pub mod memory;
pub mod scoped_inference;
pub mod stores;
pub mod strategies;
pub mod structures;
pub mod transitions;
pub mod traversal;

// Code intelligence
pub mod code_graph;

// New advanced graph features (2026-01-15)
pub mod algorithms;
pub mod analytics;
pub mod indexing;

// Phase 5B: Graph Persistence (2026-01-20)
pub mod compression;
pub mod graph_store;
pub mod redb_graph_store;

// Semantic Memory (2026-01-22)
pub mod claims;

// 10x/100x: Memory Consolidation + LLM Refinement
pub mod consolidation;
pub mod refinement;

// Export/Import (streaming binary v2 format)
pub mod export;
pub mod wire_v2;

// Event content extraction helpers
pub mod event_content;

// Temporal views (snapshot, rolling window)
pub mod temporal_view;

// GraphView trait: composable, zero-copy graph views
pub mod graph_view;

// TCell: time-indexed property container
pub mod tcell;

// Dense Vec storage for O(1) node/edge access
pub mod slot_vec;

// String interning pool
pub mod intern;

// Multi-signal retrieval for memories and strategies
pub mod retrieval;

// Natural Language to Graph Query pipeline
pub mod nlq;

// Dynamic structured memory templates (ledgers, trees, state machines, preferences)
pub mod structured_memory;

// Production-grade NLP primitives (tokenizer, POS tagger, chunker, frames)
pub mod nlp;

// Unified LLM client abstraction
pub mod llm_client;

// Conversation ingestion layer (NL → structured memory)
pub mod conversation;

// Memory audit trail (ADD/UPDATE/DELETE tracking)
pub mod memory_audit;

// LLM-driven memory update classification (ADD/UPDATE/DELETE/NONE)
pub mod memory_classifier;

// In-memory goal store with BM25-backed fast deduplication
pub mod goal_store;

// Resilient metadata normalization for structured memory auto-detection
pub mod metadata_normalize;

// Canonical domain registry for temporal state predicates
pub mod domain_schema;

// OWL/RDFS-aligned ontology layer (replaces hardcoded domain checks)
pub mod ontology;

// Self-expanding ontology: discovery, inference, and LLM-assisted evolution
pub mod ontology_evolution;

// Community summaries for graph communities
pub mod community_summary;

// Active Retrieval Testing (ART) — validates embedding retrievability
pub mod active_retrieval_test;

// Community-enriched context injection for formation pipelines
pub mod context_enrichment;

// Background maintenance (decay, pruning, dedup)
pub mod maintenance;

// Re-export commonly used items
pub use catalog::{EpisodeCatalog, EpisodeRecord, RedbEpisodeCatalog};
pub use consolidation::{
    ConsolidationConfig, ConsolidationEngine, ConsolidationResult, StrategyEvolution,
};
pub use decision_trace::{
    DecisionTrace, DecisionTraceStore, OutcomeSignal, RedbDecisionTraceStore,
};
pub use episodes::{Episode, EpisodeDetector, EpisodeDetectorConfig, EpisodeId, EpisodeOutcome};
pub use error::{GraphError, GraphResult};
pub use graph_view::{EdgeTypeFilter, GraphView, NodeSubgraph};
pub use inference::{
    ContextualAssociation, EntityReference, EpisodeMetrics, GraphInference, InferenceConfig,
    InferenceResults, InferenceStats, ReinforcementResult, ReinforcementStats, TemporalPattern,
};
pub use integration::{
    ClaimMetrics, GraphEngine, GraphEngineConfig, GraphMetricsSummary, GraphOperationResult,
    MemoryMetrics, StoreMetrics, StrategyMetrics,
};
pub use learning::{LearningStatsStore, MotifStats, RedbLearningStatsStore, TransitionStats};
pub use maintenance::{MaintenanceConfig, MaintenanceResult};
pub use memory::{
    ConsolidationStatus, Memory, MemoryFormation, MemoryFormationConfig, MemoryId, MemoryStats,
    MemoryTier, MemoryType,
};
pub use refinement::{RefinementConfig, RefinementEngine};
pub use stores::{
    InMemoryMemoryStore, InMemoryStrategyStore, MemoryStore, RedbMemoryStore, RedbStrategyStore,
    StrategyStore,
};
pub use strategies::{
    ContextPattern, PlaybookBranch, PlaybookStep, ReasoningStep, Strategy,
    StrategyExtractionConfig, StrategyExtractor, StrategyId, StrategyStats, StrategyType,
};
pub use structures::{
    AdjList, ConceptType, Depth, Direction, EdgeId, EdgeType, EdgeWeight, GoalRelationType,
    GoalStatus, Graph, GraphEdge, GraphNode, GraphStats, InteractionType, NodeId, NodeType,
};
pub use temporal_view::{GraphAtSnapshot, RollingWindow};
pub use traversal::{
    ActionSuggestion, BfsIter, CancelHandle, CommunityAlgorithm, DfsIter, DijkstraIter,
    DirectedBfsIter, DirectedDfsIter, DirectedDijkstraIter, EdgeFilterExpr, GraphQuery,
    GraphTraversal, Instruction, NodeFilterExpr, PathConstraint, QueryContext, QueryResult,
    QueryStats, StreamingQuery, TraversalRequest, TraversalSpec,
};

// New advanced graph features
pub use algorithms::{
    AllCentralities, CentralityMeasures, CommunityDetectionResult, CommunityPrepData,
    LabelPropagationAlgorithm, LabelPropagationConfig, LabelPropagationResult, LouvainAlgorithm,
    LouvainConfig, ParallelGraphAlgorithms, ProcessResult, RandomWalkConfig, RandomWalkResult,
    RandomWalker, ReachabilityRecord, TemporalReachability, TemporalReachabilityConfig,
    TemporalReachabilityResult, WalkPath,
};
pub use analytics::{GraphAnalytics, GraphMetrics, LearningMetrics};
pub use indexing::{IndexManager, IndexStats, IndexType, PropertyIndex};

// Phase 5B: Graph Persistence
pub use compression::{CompressedAdjacencyList, CompressionStats};
pub use graph_store::{
    BucketInfo, EvictionTier, GraphPath, GraphStore, GraphStoreError, InMemoryGraphStore,
    NodeFilter, NodeHeader, ShardedGraphStore, Subgraph, NODE_HEADER_BYTES,
};
pub use redb_graph_store::RedbGraphStore;

// Graph Pruning (streaming, bounded)
pub mod graph_pruning;
pub use graph_pruning::{GraphPruner, GraphPruningConfig, PruneResult};

// MinnsQL query language
pub mod query_lang;

// Subscription system (delta capture + trigger set compilation)
pub mod subscription;

// Export/Import
pub use export::{ExportError, ImportError, ImportMode, ImportStats};
pub use stores::{build_memory_index_ops, build_strategy_index_ops};

// Semantic Memory
pub use claims::{
    ClaimExtractionRequest, ClaimExtractionResult, ClaimId, ClaimStatus, ClaimType, DerivedClaim,
    EvidenceSpan, RejectedClaim, RejectionReason, SourceRole, TemporalType, ThreadId,
};

// Multi-signal retrieval
pub use retrieval::{
    apply_reranking, compute_importance, importance_modulated_decay_score, ImportanceDecayConfig,
    ImportanceDecayParams, LlmReranker, MemoryRetrievalConfig, MemoryRetrievalPipeline,
    MemoryRetrievalQuery, RerankedItem, Reranker, RerankerConfig, StrategyRetrievalConfig,
    StrategyRetrievalPipeline, StrategyRetrievalQuery,
};

// Memory audit trail
pub use memory_audit::{MemoryAuditEntry, MemoryAuditLog, MemoryMutationType, MutationActor};

// Memory update classifier
pub use memory_classifier::{
    classify_memory_updates, ClassificationResult, ClassifiedOperation, MemoryAction,
};

// Goal store
pub use goal_store::{GoalDedupDecision, GoalEntry, GoalStore};

// Community Summaries
pub use community_summary::{CommunitySummary, CommunitySummaryConfig};

// Context Enrichment
pub use context_enrichment::EnrichmentConfig;

// Natural Language Query
pub use nlq::{
    intent::QueryIntent, ConversationContext, ConversationExchange, NlqPagination, NlqPipeline,
    NlqResponse,
};

// Structured Memory
pub use structured_memory::{
    LedgerDirection, LedgerEntry, MemoryProvenance, MemoryTemplate, PreferenceItem,
    StateTransition, StructuredMemoryStore,
};

// Metadata Normalization
pub use metadata_normalize::{
    AliasConfig, MetadataNormalizer, MetadataRole, NormalizedMetadata, ResolutionMethod,
};

// Conversation Ingestion
pub use conversation::{
    gather_memory_context, ingest_incremental, ingest_with_llm_incremental, CompactionResult,
    ConversationIngest, ConversationMessage, ConversationRollingSummary, ConversationSession,
    ConversationState, GoalPlaybook, IngestOptions, IngestResult, MemoryContextEntry,
    MemorySummary, NameRegistry, StrategySummary,
};
