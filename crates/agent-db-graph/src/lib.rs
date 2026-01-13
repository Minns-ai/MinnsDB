//! Graph operations and inference for agent database
//! 
//! This crate provides graph construction and analysis capabilities
//! for the agent database, inferring relationships from event patterns.

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
    }
    
    pub type GraphResult<T> = Result<T, GraphError>;
}

pub mod structures;
pub mod inference;
pub mod traversal;
pub mod integration;
pub mod event_ordering;
pub mod scoped_inference;

// Re-export commonly used items
pub use error::{GraphError, GraphResult};
pub use structures::{
    Graph, GraphNode, GraphEdge, NodeType, EdgeType, NodeId, EdgeId, EdgeWeight,
    ConceptType, GoalStatus, InteractionType, GoalRelationType, GraphStats,
};
pub use inference::{
    GraphInference, InferenceConfig, InferenceStats, InferenceResults,
    TemporalPattern, ContextualAssociation, EntityReference,
};
pub use traversal::{
    GraphTraversal, GraphQuery, QueryResult, QueryStats,
    PathConstraint, CommunityAlgorithm,
};
pub use integration::{
    GraphEngine, GraphEngineConfig, GraphOperationResult,
};