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
pub mod integration_ner;
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

// Re-export commonly used items
pub use catalog::{EpisodeCatalog, EpisodeRecord, RedbEpisodeCatalog};
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
pub use memory::{
    Memory, MemoryFormation, MemoryFormationConfig, MemoryId, MemoryStats, MemoryType,
};
pub use stores::{
    InMemoryMemoryStore, InMemoryStrategyStore, MemoryStore, RedbMemoryStore, RedbStrategyStore,
    StrategyStore,
};
pub use strategies::{
    ContextPattern, ReasoningStep, Strategy, StrategyExtractionConfig, StrategyExtractor,
    StrategyId, StrategyStats,
};
pub use structures::{
    ConceptType, EdgeId, EdgeType, EdgeWeight, GoalRelationType, GoalStatus, Graph, GraphEdge,
    GraphNode, GraphStats, InteractionType, NodeId, NodeType,
};
pub use traversal::{
    ActionSuggestion, CommunityAlgorithm, GraphQuery, GraphTraversal, PathConstraint, QueryResult,
    QueryStats,
};

// New advanced graph features
pub use algorithms::{
    AllCentralities, CentralityMeasures, CommunityDetectionResult, CommunityPrepData,
    LouvainAlgorithm, LouvainConfig, ParallelGraphAlgorithms, ProcessResult,
};
pub use analytics::{GraphAnalytics, GraphMetrics, LearningMetrics};
pub use indexing::{IndexManager, IndexStats, IndexType, PropertyIndex};

// Phase 5B: Graph Persistence
pub use compression::{CompressedAdjacencyList, CompressionStats};
pub use graph_store::{
    BucketInfo, GraphEdge as PersistentGraphEdge, GraphEdgeType as PersistentGraphEdgeType,
    GraphNode as PersistentGraphNode, GraphNodeType as PersistentGraphNodeType, GraphPath,
    GraphStore, GraphStoreError, InMemoryGraphStore, Subgraph,
};
pub use redb_graph_store::RedbGraphStore;

// Semantic Memory
pub use claims::{
    ClaimExtractionRequest, ClaimExtractionResult, ClaimId, ClaimStatus, DerivedClaim,
    EvidenceSpan, RejectedClaim, RejectionReason, ThreadId,
};
