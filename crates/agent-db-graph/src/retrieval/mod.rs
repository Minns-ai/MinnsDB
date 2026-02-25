//! Multi-signal retrieval for memories and strategies.
//!
//! Orchestrates scoring across multiple signals (semantic, BM25, context,
//! temporal, graph proximity, etc.) and fuses them via Reciprocal Rank Fusion.
//! Pure functions — no store ownership; the caller supplies candidates.

mod fusion;
mod memory_retrieval;
mod strategy_retrieval;
mod temporal;

pub use fusion::multi_list_rrf;
pub use memory_retrieval::{MemoryRetrievalConfig, MemoryRetrievalPipeline, MemoryRetrievalQuery};
pub use strategy_retrieval::{
    StrategyRetrievalConfig, StrategyRetrievalPipeline, StrategyRetrievalQuery,
};
pub use temporal::temporal_decay_score;
