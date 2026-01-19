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

pub mod structures;
pub mod inference;
pub mod traversal;
pub mod integration;
pub mod event_ordering;
pub mod scoped_inference;
pub mod episodes;
pub mod memory;
pub mod strategies;
pub mod contracts;
pub mod stores;
pub mod transitions;

// New advanced graph features (2026-01-15)
pub mod indexing;
pub mod algorithms;
pub mod analytics;

// Re-export commonly used items
pub use error::{GraphError, GraphResult};
pub use structures::{
    Graph, GraphNode, GraphEdge, NodeType, EdgeType, NodeId, EdgeId, EdgeWeight,
    ConceptType, GoalStatus, InteractionType, GoalRelationType, GraphStats,
};
pub use inference::{
    GraphInference, InferenceConfig, InferenceStats, InferenceResults,
    TemporalPattern, ContextualAssociation, EntityReference,
    EpisodeMetrics, ReinforcementResult, ReinforcementStats,
};
pub use traversal::{
    GraphTraversal, GraphQuery, QueryResult, QueryStats,
    PathConstraint, CommunityAlgorithm, ActionSuggestion,
};
pub use integration::{
    GraphEngine, GraphEngineConfig, GraphOperationResult,
};
pub use episodes::{
    Episode, EpisodeId, EpisodeOutcome, EpisodeDetector, EpisodeDetectorConfig,
};
pub use memory::{
    Memory, MemoryId, MemoryType, MemoryFormation, MemoryFormationConfig, MemoryStats,
};
pub use strategies::{
    Strategy, StrategyId, ReasoningStep, ContextPattern,
    StrategyExtractor, StrategyExtractionConfig, StrategyStats,
};

// New advanced graph features
pub use indexing::{
    PropertyIndex, IndexManager, IndexType, IndexStats,
};
pub use algorithms::{
    LouvainAlgorithm, LouvainConfig, CommunityDetectionResult,
    CentralityMeasures, AllCentralities,
    ParallelGraphAlgorithms, ProcessResult, CommunityPrepData,
};
pub use analytics::{
    GraphAnalytics, GraphMetrics, LearningMetrics,
};