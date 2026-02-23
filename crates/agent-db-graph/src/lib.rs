//! Graph operations and inference for agent database
//!
//! This crate provides graph construction and analysis capabilities
//! for the agent database, inferring relationships from event patterns.
//!
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

// String interning pool
pub mod intern;

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

// Export/Import
pub use export::{ExportError, ImportError, ImportMode, ImportStats};
pub use stores::{build_memory_index_ops, build_strategy_index_ops};

// Semantic Memory
pub use claims::{
    ClaimExtractionRequest, ClaimExtractionResult, ClaimId, ClaimStatus, ClaimType, DerivedClaim,
    EvidenceSpan, RejectedClaim, RejectionReason, ThreadId,
};
