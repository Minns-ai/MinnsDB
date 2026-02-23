//! Generative strategy/action planning engine for EventGraphDB.
//!
//! This crate implements the **Generator** side of the Generator + Critic
//! architecture. It provides:
//!
//! - **Strategy generation**: Produce K candidate strategies for a goal
//! - **Action generation**: Produce N candidate next-actions under a strategy
//! - **Validation**: JSON schema validation before scoring
//! - **Selection**: Accept/revise/reject/experimental based on critic scores
//! - **Orchestration**: End-to-end pipeline coordination
//! - **Repair**: Revise plans based on prediction errors during execution
//!
//! # Design Principle: Traits First
//!
//! All boundary traits are defined here. Concrete implementations (LLM-based,
//! mock, etc.) implement these traits. This enables:
//! - Mock-based testing from day one
//! - Swappable generator backends (LLM, rule-based, trained)
//! - Clean dependency boundaries (no networking deps in this crate)

pub mod llm_client;
pub mod llm_generator;
pub mod mock_generator;
pub mod orchestrator;
pub mod prompts;
pub mod repair;
pub mod selector;
pub mod types;
pub mod validation;

// Re-export primary types at crate root.
pub use types::{
    ActionGenerationRequest, Criteria, EventContext, GeneratedActionPlan, GeneratedStep,
    GeneratedStrategyPlan, GenerationMode, MemoryContext, PlanningConfig, PlanningError, RiskFlag,
    RiskSeverity, ScoredAction, ScoredCandidate, SelectionDecision, SelectionDecisionKind,
    StepKind, StrategyContext, StrategyGenerationRequest, TransitionHint, ValidationError,
    ValidationSeverity, WorldModelMode,
};

use agent_db_world_model::{
    CriticReport, EventFeatures, MemoryFeatures, PolicyFeatures, PredictionErrorReport,
    StrategyFeatures,
};

// ────────────────────────────────────────────────────────────────
// Boundary traits
// ────────────────────────────────────────────────────────────────

/// Context provider — reads memories, strategies, events from GraphEngine.
///
/// This trait decouples planning from the graph engine's internals.
pub trait PlanningContextProvider: Send + Sync {
    /// Retrieve relevant memories for a context.
    fn get_memories_for_context(
        &self,
        context_fingerprint: u64,
        limit: usize,
    ) -> Vec<MemoryFeatures>;

    /// Retrieve similar strategies for a goal.
    fn get_strategies_for_goal(&self, goal_bucket_id: u64, limit: usize) -> Vec<StrategyFeatures>;

    /// Get recent events for the session.
    fn get_recent_events(&self, session_id: u64, limit: usize) -> Vec<EventFeatures>;

    /// Build policy features from current state.
    fn build_policy_features(
        &self,
        goal_description: &str,
        context_fingerprint: u64,
    ) -> PolicyFeatures;
}

/// Strategy generator — produces K candidate strategies for a goal.
///
/// v1: LLM-based. Trait allows mock/deterministic implementations for testing.
#[async_trait::async_trait]
pub trait StrategyGenerator: Send + Sync {
    /// Generate K candidate strategies for a goal.
    async fn generate(
        &self,
        request: StrategyGenerationRequest,
    ) -> Result<Vec<GeneratedStrategyPlan>, PlanningError>;

    /// Revise a candidate using critic diagnostics.
    async fn revise(
        &self,
        candidate: &GeneratedStrategyPlan,
        diagnostics: &CriticReport,
        request: &StrategyGenerationRequest,
    ) -> Result<Vec<GeneratedStrategyPlan>, PlanningError>;
}

/// Action generator — produces N candidate next-actions under a strategy.
#[async_trait::async_trait]
pub trait ActionGenerator: Send + Sync {
    /// Generate N candidate actions for the current step.
    async fn generate(
        &self,
        request: ActionGenerationRequest,
    ) -> Result<Vec<GeneratedActionPlan>, PlanningError>;

    /// Generate repair action after prediction error spike.
    async fn repair(
        &self,
        current_step: &GeneratedStep,
        error_report: &PredictionErrorReport,
        request: &ActionGenerationRequest,
    ) -> Result<Vec<GeneratedActionPlan>, PlanningError>;
}

/// Execution sink — emits actions and records results.
///
/// Decouples planning from the actual execution mechanism.
pub trait ExecutionSink: Send + Sync {
    /// Emit an action for execution. Returns an execution ID.
    fn emit_action(&self, action: &GeneratedActionPlan) -> Result<u64, PlanningError>;

    /// Record the result of an executed action.
    fn record_result(
        &self,
        execution_id: u64,
        event_features: &EventFeatures,
    ) -> Result<(), PlanningError>;
}

/// JSON schema validation for generator output.
///
/// Applied before any scoring — rejects malformed plans early.
pub trait PlanValidator: Send + Sync {
    /// Validate a strategy plan. Returns errors if invalid.
    fn validate_strategy(&self, plan: &GeneratedStrategyPlan) -> Vec<ValidationError>;

    /// Validate an action plan. Returns errors if invalid.
    fn validate_action(&self, plan: &GeneratedActionPlan) -> Vec<ValidationError>;
}
