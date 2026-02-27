//! Multi-signal retrieval for memories and strategies.
//!
//! Orchestrates scoring across multiple signals (semantic, BM25, context,
//! temporal, graph proximity, etc.) and fuses them via Reciprocal Rank Fusion.
//! Pure functions — no store ownership; the caller supplies candidates.

mod fusion;
mod memory_retrieval;
pub mod reranker;
mod strategy_retrieval;
mod temporal;

pub use fusion::multi_list_rrf;
pub use memory_retrieval::{MemoryRetrievalConfig, MemoryRetrievalPipeline, MemoryRetrievalQuery};
pub use reranker::{apply_reranking, LlmReranker, RerankedItem, Reranker, RerankerConfig};
pub use strategy_retrieval::{
    StrategyRetrievalConfig, StrategyRetrievalPipeline, StrategyRetrievalQuery,
};
pub use temporal::{
    compute_importance, importance_modulated_decay_score, temporal_decay_score,
    ImportanceDecayConfig, ImportanceDecayParams,
};
