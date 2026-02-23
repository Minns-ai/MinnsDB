//! Structured output types for the planning engine.
//!
//! These types define the shape of generated strategies, actions, and
//! associated metadata. They are designed for structured LLM output
//! (JSON schema constrained) and programmatic consumption.

use serde::{Deserialize, Serialize};

use agent_db_world_model::CriticReport;

// ────────────────────────────────────────────────────────────────
// Enums
// ────────────────────────────────────────────────────────────────

/// Risk severity levels.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Step kind — typed, not free-text.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StepKind {
    /// Execute an action.
    Action,
    /// Wait for / check observation.
    Observation,
    /// Branch point.
    Decision,
    /// Check precondition / postcondition.
    Validation,
    /// Error handling step.
    Recovery,
}

// ────────────────────────────────────────────────────────────────
// Criteria (structured success/failure checks)
// ────────────────────────────────────────────────────────────────

/// Structured success/failure criteria.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Criteria {
    pub description: String,
    /// Type of check: "event_type_match", "value_range", "timeout", etc.
    pub check_type: String,
    /// Parameters for the check (schema depends on check_type).
    pub parameters: serde_json::Value,
}

// ────────────────────────────────────────────────────────────────
// Strategy plan
// ────────────────────────────────────────────────────────────────

/// A single step in a generated strategy plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedStep {
    pub step_number: u32,
    pub step_kind: StepKind,
    /// Action identifier (not free text).
    pub action_type: String,
    /// Structured action parameters.
    pub parameters: serde_json::Value,
    /// Human-readable description (optional).
    pub description: Option<String>,
    pub precondition: Option<Criteria>,
    pub success_criteria: Option<Criteria>,
    pub failure_criteria: Option<Criteria>,
    pub skip_if: Option<Criteria>,
    /// Maximum retries before failing (default: 0).
    pub max_retries: u32,
    /// Timeout in milliseconds (None = no timeout).
    pub timeout_ms: Option<u64>,
    /// Conditional branches.
    pub branches: Vec<StepBranch>,
    /// Recovery action identifier if this step fails.
    pub recovery: Option<String>,
}

/// A conditional branch within a step.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepBranch {
    pub condition: Criteria,
    pub goto_step: u32,
}

/// Full structured strategy plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedStrategyPlan {
    pub goal_bucket_id: u64,
    pub goal_description: String,
    pub steps: Vec<GeneratedStep>,
    pub preconditions: Vec<Criteria>,
    pub stop_conditions: Vec<Criteria>,
    pub fallback_steps: Vec<GeneratedStep>,
    pub risk_flags: Vec<RiskFlag>,
    pub assumptions: Vec<String>,
    /// Generator's self-assessed confidence (0.0–1.0).
    pub confidence: f32,
    /// Human-readable rationale for the plan.
    pub rationale: Option<String>,
}

/// A risk flag identified by the generator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFlag {
    pub description: String,
    pub severity: RiskSeverity,
    pub mitigation: Option<String>,
}

// ────────────────────────────────────────────────────────────────
// Action plan
// ────────────────────────────────────────────────────────────────

/// Expected event after action execution.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedEvent {
    pub event_type: String,
    pub expected_outcome: String,
    pub expected_significance: f32,
}

/// Structured action plan.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedActionPlan {
    pub action_type: String,
    pub parameters: serde_json::Value,
    pub preconditions: Vec<Criteria>,
    pub expected_event: ExpectedEvent,
    pub success_criteria: Option<Criteria>,
    pub failure_criteria: Option<Criteria>,
    pub timeout_ms: Option<u64>,
    pub max_retries: u32,
    pub fallback_action: Option<String>,
    pub risk_flags: Vec<RiskFlag>,
    /// Generator's self-assessed confidence (0.0–1.0).
    pub confidence: f32,
}

// ────────────────────────────────────────────────────────────────
// Selector output
// ────────────────────────────────────────────────────────────────

/// Selector decision for a scored candidate.
#[derive(Debug, Clone)]
pub enum SelectionDecision {
    /// Accept this plan as-is.
    Accept(GeneratedStrategyPlan),
    /// Send back to generator with diagnostics for revision.
    Revise {
        candidate: GeneratedStrategyPlan,
        diagnostics: CriticReport,
    },
    /// Accept as experimental (novel but not impossible).
    Experimental(GeneratedStrategyPlan),
    /// Reject entirely.
    Reject {
        reason: String,
        diagnostics: CriticReport,
    },
}

// ────────────────────────────────────────────────────────────────
// Scored output (used by GraphEngine integration)
// ────────────────────────────────────────────────────────────────

/// A strategy candidate with its critic score attached.
#[derive(Debug, Clone)]
pub struct ScoredCandidate {
    pub plan: GeneratedStrategyPlan,
    pub report: CriticReport,
    pub decision: SelectionDecisionKind,
}

/// Simplified enum variant for `ScoredCandidate` (avoids cloning the full plan again).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SelectionDecisionKind {
    Accept,
    Revise,
    Experimental,
    Reject,
}

/// A scored action plan.
#[derive(Debug, Clone)]
pub struct ScoredAction {
    pub plan: GeneratedActionPlan,
    pub energy: f32,
    pub feasibility: f32,
}

// ────────────────────────────────────────────────────────────────
// Generator context (human-readable, for LLM input)
// ────────────────────────────────────────────────────────────────

/// Simplified memory context for generator input (human-readable).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContext {
    pub summary: String,
    pub tier: String,
    pub strength: f32,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub takeaway: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub causal_note: Option<String>,
}

/// Simplified strategy context for generator input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyContext {
    pub name: String,
    pub summary: String,
    pub quality_score: f32,
    pub when_to_use: String,
}

/// Simplified event context for generator input.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventContext {
    pub event_type: String,
    pub action_name: Option<String>,
    pub outcome: Option<String>,
    pub timestamp: u64,
}

/// Empirical success rate for a state→action→next_state transition from the Markov model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TransitionHint {
    pub from_state: String,
    pub action: String,
    pub to_state: String,
    pub success_rate: f32,
    pub observation_count: u64,
}

// ────────────────────────────────────────────────────────────────
// Requests
// ────────────────────────────────────────────────────────────────

/// Request to generate strategy candidates.
#[derive(Debug, Clone)]
pub struct StrategyGenerationRequest {
    pub goal_description: String,
    pub goal_bucket_id: u64,
    pub context_fingerprint: u64,
    pub relevant_memories: Vec<MemoryContext>,
    pub similar_strategies: Vec<StrategyContext>,
    pub recent_events: Vec<EventContext>,
    pub transition_hints: Vec<TransitionHint>,
    pub constraints: Vec<String>,
    pub k_candidates: usize,
}

/// Request to generate action candidates.
#[derive(Debug, Clone)]
pub struct ActionGenerationRequest {
    pub strategy: GeneratedStrategyPlan,
    pub current_step_index: usize,
    pub recent_events: Vec<EventContext>,
    pub context_fingerprint: u64,
    pub n_candidates: usize,
}

// ────────────────────────────────────────────────────────────────
// Validation
// ────────────────────────────────────────────────────────────────

/// A validation error found in a generated plan.
#[derive(Debug, Clone)]
pub struct ValidationError {
    pub field: String,
    pub message: String,
    pub severity: ValidationSeverity,
}

/// Severity of a validation error.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ValidationSeverity {
    /// Proceed with caution.
    Warning,
    /// Reject this candidate.
    Error,
}

// ────────────────────────────────────────────────────────────────
// Errors
// ────────────────────────────────────────────────────────────────

/// Planning errors.
#[derive(Debug, thiserror::Error)]
pub enum PlanningError {
    #[error("generation failed: {0}")]
    GenerationFailed(String),

    #[error("validation failed: {} errors", .0.len())]
    ValidationFailed(Vec<ValidationError>),

    #[error("all candidates rejected: {0}")]
    AllCandidatesRejected(String),

    #[error("LLM timeout")]
    LlmTimeout,

    #[error("LLM error: {0}")]
    LlmError(String),

    #[error("world model not warmed up")]
    NotWarmedUp,
}

// Display for ValidationError (needed since it's inside PlanningError)
impl std::fmt::Display for ValidationError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "[{:?}] {}: {}", self.severity, self.field, self.message)
    }
}

// ────────────────────────────────────────────────────────────────
// Configuration
// ────────────────────────────────────────────────────────────────

/// Configuration for the planning engine.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanningConfig {
    // Master switches
    pub enable_world_model: bool,
    pub enable_strategy_generation: bool,
    pub enable_action_generation: bool,
    pub repair_enabled: bool,

    // Rollout modes
    pub world_model_mode: WorldModelMode,
    pub generation_mode: GenerationMode,

    // Selector thresholds (z-score based)
    pub accept_z: f32,
    pub revise_z: f32,
    pub reject_z: f32,
    pub repair_z: f32,
    pub min_confidence: f32,

    // Generation
    pub strategy_candidates_k: usize,
    pub action_candidates_n: usize,
    pub max_revision_rounds: usize,

    // LLM
    pub llm_model: String,
    pub llm_temperature: f32,
    pub llm_max_tokens: u32,
}

impl Default for PlanningConfig {
    fn default() -> Self {
        Self {
            enable_world_model: false,
            enable_strategy_generation: false,
            enable_action_generation: false,
            repair_enabled: false,
            world_model_mode: WorldModelMode::Disabled,
            generation_mode: GenerationMode::Disabled,
            accept_z: 1.0,
            revise_z: 2.0,
            reject_z: 3.0,
            repair_z: 2.0,
            min_confidence: 0.3,
            strategy_candidates_k: 3,
            action_candidates_n: 2,
            max_revision_rounds: 2,
            llm_model: "gpt-4o-mini".to_string(),
            llm_temperature: 0.4,
            llm_max_tokens: 2048,
        }
    }
}

/// World model operating mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WorldModelMode {
    Disabled,
    /// Train + score + log only, zero impact.
    Shadow,
    /// Score, expose via API, no selection.
    ScoringOnly,
    /// Score + rerank candidates.
    ScoringAndReranking,
    /// Score + rerank + repair + learn.
    Full,
}

/// Generation operating mode.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GenerationMode {
    Disabled,
    /// Generate candidates, log them, don't score.
    GenerateOnly,
    /// Generate + score, log recommended selection.
    GenerateAndScore,
    /// Generate + score + select, don't execute.
    GenerateScoreAndSelect,
    /// Generate + score + select + execute.
    Full,
}
