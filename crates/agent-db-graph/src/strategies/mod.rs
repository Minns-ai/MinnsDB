// crates/agent-db-graph/src/strategies/mod.rs
//
// Strategy Extraction Module
//
// Extracts generalizable strategies from successful episodes and reasoning traces,
// enabling agents to reuse proven approaches in similar contexts.

mod context;
mod distiller;
mod extractor;
mod reasoning;
mod retrieval;
mod scoring;
mod signatures;
mod storage;
pub mod synthesis;
pub mod types;

#[cfg(test)]
mod tests;

// ── Re-exports ──

// Types
pub use types::{
    compute_goal_bucket_id_from_ids, ContextPattern, PlaybookBranch, PlaybookStep, ReasoningStep,
    Strategy, StrategyExtractionConfig, StrategyId, StrategySimilarityQuery, StrategyStats,
    StrategyType, StrategyUpsert,
};

// Extractor
pub use extractor::StrategyExtractor;

// Synthesis (public free function)
pub use synthesis::synthesize_strategy_summary;
