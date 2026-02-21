# Plan: Energy-Based Predictive Coding World Model + Generative Strategy/Action Planner

## Context

EventGraphDB has ~40% of the foundation for an intelligent agent system: Bayesian transition dynamics, 3-tier memory consolidation, strategy extraction with quality metrics, episode detection with prediction errors, and a rich property graph. But it is **reactive only** — it records what happened and extracts patterns, but cannot **generate plans**, **predict outcomes**, or **detect genuine novelty**.

The goal: a **Generator + Critic** architecture where:
- A **Generator** (LLM or structured) produces candidate strategies and actions for new goals
- A **Critic** (energy-based predictive coding world model) scores, reranks, and validates those candidates
- **Bidirectional predictive coding** flows top-down (planning constraints) and bottom-up (error propagation + learning)
- The system **learns from prediction errors** to improve over time

This is NOT reinforcement learning (v1). It is a **planning + world-model** system. RL is future work.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                     GENERATOR (GenAI)                        │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │  Strategy     │  │  Action      │  │  Plan        │      │
│  │  Generator    │  │  Generator   │  │  Repairer    │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
└─────────┼──────────────────┼──────────────────┼─────────────┘
          │ K candidates     │ N candidates     │ revised plan
          ▼                  ▼                  ▲
┌─────────────────────────────────────────────────────────────┐
│                  SELECTOR / ORCHESTRATOR                      │
│  accept · revise · reject · mark-experimental                │
│  Routes diagnostics back to Generator for revision           │
└─────────┬──────────────────┬──────────────────┬─────────────┘
          │ selected         │ selected         │ diagnostics
          ▼                  ▼                  ▲
┌─────────────────────────────────────────────────────────────┐
│              CRITIC (EBM Predictive Coding World Model)      │
│                                                              │
│  TOP-DOWN (planning/evaluation)    BOTTOM-UP (learning)      │
│  ┌────────┐                        ┌────────┐               │
│  │ Policy │──predicts──►           │ Events │──error──►     │
│  └────────┘            │           └────────┘          │    │
│  ┌────────┐            ▼           ┌────────┐          ▼    │
│  │Strategy│──predicts──►           │ Memory │──error──►     │
│  └────────┘            │           └────────┘          │    │
│  ┌────────┐            ▼           ┌────────┐          ▼    │
│  │ Memory │──predicts──►           │Strategy│──error──►     │
│  └────────┘            │           └────────┘          │    │
│  ┌────────┐            ▼           ┌────────┐          ▼    │
│  │ Events │◄──expected──           │ Policy │◄──update──    │
│  └────────┘                        └────────┘               │
│                                                              │
│  Outputs: total energy, per-layer energies, novelty score,   │
│           mismatch attribution, confidence + support         │
└─────────────────────────────────────────────────────────────┘
          │                                     ▲
          ▼                                     │
┌─────────────────────────────────────────────────────────────┐
│                      EXECUTOR                                │
│  Emits actions → captures outcomes → feeds back to Critic    │
└─────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────┐
│                      LEARNER                                 │
│  Completed episodes + prediction errors → train world model  │
│  Update novelty calibration, energy statistics               │
│  Persist world model weights                                 │
└─────────────────────────────────────────────────────────────┘
```

---

## Bidirectional Predictive Coding (Explicit)

The predictive coding world model is **bidirectional**. Top-down only is NOT sufficient.

### Top-Down Flow (Planning & Evaluation)

Higher layers predict lower layers. Used to **evaluate and constrain** generated strategies/actions and provide expected compatibility profiles.

```
Policy state (goal + context + resources)
  │
  ├─ scores → which Strategy candidates are compatible
  │             E_strategy(candidate | policy) → compatibility score
  │
  ├─ scores → which Memory patterns are compatible with a strategy
  │             E_memory(memory | strategy) → compatibility score
  │
  └─ scores → which Event patterns are compatible with memory context
                E_event(event | memory) → compatibility score
```

**Use cases**: Score candidate strategies before execution. Rank action candidates by expected compatibility. Flag strategies that don't fit the current policy.

**v1 scope**: The critic **scores and ranks** — it does not decode/generate concrete memory or event objects from embeddings. "Predict" means "compute expected compatibility profile", not "generate objects".

### Bottom-Up Flow (Validation, Repair, Learning)

Reality pushes upward via prediction errors. Used for **error propagation**, novelty detection, repair triggers, and learning.

```
Actual events arrive
  │
  ├─ compare: E_event(actual_event | active_memory_context)
  │   → Event prediction error (ε_event)
  │
  ├─ propagate: ε_event feeds into Memory layer
  │   → Memory prediction error (ε_memory) — "memories don't explain this"
  │
  ├─ propagate: ε_memory feeds into Strategy layer
  │   → Strategy prediction error (ε_strategy) — "strategy didn't predict this"
  │
  └─ propagate: ε_strategy feeds into Policy layer
      → Policy confidence update
```

**Use cases**: Detect genuinely novel situations. Trigger action/strategy repair. Provide training signal for world model. Credit assignment (which layer failed).

---

## v1 Interface Spec (Frozen Traits + Core Types)

### Design Principle: Traits First

All boundary traits are defined and frozen before any implementation begins. This prevents integration churn and enables mock-based testing from day one.

---

### Crate: `ml/agent-db-world-model`

```rust
// ============================================================
// types.rs — Core types for world model
// ============================================================

/// Which layer has the highest mismatch
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MismatchLayer {
    Event,
    Memory,
    Strategy,
    Policy,
    None,
}

/// Full critic report for a scored configuration
#[derive(Debug, Clone)]
pub struct CriticReport {
    /// Total free energy (lower = more compatible)
    pub total_energy: f32,
    /// Per-layer energies
    pub policy_strategy_energy: f32,
    pub strategy_memory_energy: f32,
    pub memory_event_energy: f32,
    /// Novelty as z-score over running energy statistics
    pub novelty_z: f32,
    /// Whether this exceeds the novelty threshold
    pub is_novel: bool,
    /// Which layer has the highest energy (mismatch attribution)
    pub mismatch_layer: MismatchLayer,
    /// Confidence in this score (see definition below)
    pub confidence: f32,
    /// How many similar configurations the model has been trained on
    pub support_count: u64,
}

/// Confidence definition (v1):
///   confidence = min(
///     warmup_factor,          // 0.0 until warmup_episodes seen, then 1.0
///     support_factor,         // min(1.0, support_count / min_support)
///     stability_factor,       // 1.0 - (energy_variance / max_variance).clamp(0,1)
///   )
/// This prevents over-trusting scores in sparse regions.

/// Bottom-up prediction error report for a single observed event
#[derive(Debug, Clone)]
pub struct PredictionErrorReport {
    pub event_energy: f32,
    pub memory_energy: f32,
    pub strategy_energy: f32,
    /// Per-layer z-scores
    pub event_z: f32,
    pub memory_z: f32,
    pub strategy_z: f32,
    /// Overall surprise
    pub total_z: f32,
    pub mismatch_layer: MismatchLayer,
}

/// A training tuple assembled from a completed episode.
/// See "Training Tuple Construction Rules" section for assembly rules.
#[derive(Debug, Clone)]
pub struct TrainingTuple {
    pub event_features: EventFeatures,
    pub memory_features: MemoryFeatures,
    pub strategy_features: StrategyFeatures,
    pub policy_features: PolicyFeatures,
    /// Was this a real (positive) or corrupted (negative) example?
    pub is_positive: bool,
    /// Weight for this example (salience_score of source episode)
    pub weight: f32,
}

/// Feature vectors extracted from domain objects (not embeddings yet)
#[derive(Debug, Clone)]
pub struct EventFeatures {
    pub event_type_hash: u64,
    pub action_name_hash: u64,
    pub context_fingerprint: u64,
    pub outcome_success: f32,     // 1.0 = success, 0.0 = failure, 0.5 = partial
    pub significance: f32,
    pub temporal_delta_ns: f64,   // time since previous event in episode
    pub duration_ns: f64,
}

#[derive(Debug, Clone)]
pub struct MemoryFeatures {
    pub tier: u8,                 // 0=Episodic, 1=Semantic, 2=Schema
    pub strength: f32,
    pub access_count: u32,
    pub context_fingerprint: u64,
    pub goal_bucket_id: u64,
}

#[derive(Debug, Clone)]
pub struct StrategyFeatures {
    pub quality_score: f32,
    pub expected_success: f32,
    pub expected_value: f32,
    pub confidence: f32,
    pub goal_bucket_id: u64,
    pub behavior_signature_hash: u64,
}

#[derive(Debug, Clone)]
pub struct PolicyFeatures {
    pub goal_count: u32,
    pub top_goal_priority: f32,
    pub resource_cpu_percent: f32,
    pub resource_memory_bytes: u64,
    pub context_fingerprint: u64,
}

// ============================================================
// lib.rs — WorldModelCritic trait (the frozen boundary)
// ============================================================

/// The world model critic scores configurations and computes prediction errors.
/// This is the primary integration trait — GraphEngine depends on this.
pub trait WorldModelCritic: Send + Sync {
    /// Score a full (policy, strategy, memory, event) configuration.
    /// Used for top-down evaluation of generated candidates.
    fn score(
        &self,
        policy: &PolicyFeatures,
        strategy: &StrategyFeatures,
        memory: &MemoryFeatures,
        event: &EventFeatures,
    ) -> CriticReport;

    /// Score a strategy candidate against a policy (no event/memory needed).
    /// Used when ranking strategy candidates before execution.
    fn score_strategy(
        &self,
        policy: &PolicyFeatures,
        strategy: &StrategyFeatures,
    ) -> CriticReport;

    /// Compute bottom-up prediction error for an observed event.
    /// Used during execution for validation and repair triggers.
    fn prediction_error(
        &self,
        observed_event: &EventFeatures,
        memory_context: &MemoryFeatures,
        active_strategy: &StrategyFeatures,
        active_policy: &PolicyFeatures,
    ) -> PredictionErrorReport;

    /// Is the model trained enough to produce reliable scores?
    fn is_warmed_up(&self) -> bool;

    /// Submit a training tuple (called by learner after episode completion)
    fn submit_training(&mut self, tuple: TrainingTuple);

    /// Run a training step (batch of pending tuples). Returns avg loss.
    fn train_step(&mut self) -> f32;

    /// Get current energy statistics for monitoring
    fn energy_stats(&self) -> EnergyStats;
}

#[derive(Debug, Clone, Default)]
pub struct EnergyStats {
    pub running_mean: f32,
    pub running_variance: f32,
    pub total_scored: u64,
    pub total_trained: u64,
    pub avg_loss: f32,
    pub is_warmed_up: bool,
}

/// Persistence: serialize/deserialize world model state to bytes.
/// Uses existing agent_db_storage::serialize_versioned pattern.
pub trait WorldModelStateStore: Send + Sync {
    fn save(&self, model: &dyn WorldModelCritic) -> Result<Vec<u8>, String>;
    fn load(&self, bytes: &[u8]) -> Result<Box<dyn WorldModelCritic>, String>;
}
```

---

### Crate: `ml/agent-db-planning`

```rust
// ============================================================
// types.rs — Structured output formats
// ============================================================

/// Risk severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RiskSeverity {
    Low,
    Medium,
    High,
    Critical,
}

/// Step kind — typed, not free-text
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StepKind {
    Action,          // Execute an action
    Observation,     // Wait for / check observation
    Decision,        // Branch point
    Validation,      // Check precondition / postcondition
    Recovery,        // Error handling step
}

/// Structured success/failure criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Criteria {
    pub description: String,
    pub check_type: String,          // e.g. "event_type_match", "value_range", "timeout"
    pub parameters: serde_json::Value,
}

/// A single step in a generated strategy plan
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneratedStep {
    pub step_number: u32,
    pub step_kind: StepKind,
    pub action_type: String,                    // action identifier (not free text)
    pub parameters: serde_json::Value,          // structured action params
    pub description: Option<String>,            // human-readable (optional)
    pub precondition: Option<Criteria>,
    pub success_criteria: Option<Criteria>,
    pub failure_criteria: Option<Criteria>,
    pub skip_if: Option<Criteria>,
    pub max_retries: u32,                       // default: 0
    pub timeout_ms: Option<u64>,
    pub branches: Vec<StepBranch>,
    pub recovery: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StepBranch {
    pub condition: Criteria,
    pub goto_step: u32,
}

/// Full structured strategy plan
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
    pub confidence: f32,
    pub rationale: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskFlag {
    pub description: String,
    pub severity: RiskSeverity,
    pub mitigation: Option<String>,
}

/// Expected event after action execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExpectedEvent {
    pub event_type: String,
    pub expected_outcome: String,
    pub expected_significance: f32,
}

/// Structured action plan
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
    pub confidence: f32,
}

/// Selector decision
#[derive(Debug, Clone)]
pub enum SelectionDecision {
    Accept(GeneratedStrategyPlan),
    Revise {
        candidate: GeneratedStrategyPlan,
        diagnostics: CriticReport,
    },
    Experimental(GeneratedStrategyPlan),
    Reject {
        reason: String,
        diagnostics: CriticReport,
    },
}

// ============================================================
// lib.rs — Frozen boundary traits
// ============================================================

use agent_db_world_model::{
    CriticReport, EventFeatures, MemoryFeatures, PolicyFeatures,
    PredictionErrorReport, StrategyFeatures, WorldModelCritic,
};

/// Context provider — reads memories, strategies, events from GraphEngine.
/// This trait decouples planning from the graph engine's internals.
pub trait PlanningContextProvider: Send + Sync {
    /// Retrieve relevant memories for a context
    fn get_memories_for_context(
        &self,
        context_fingerprint: u64,
        limit: usize,
    ) -> Vec<MemoryFeatures>;

    /// Retrieve similar strategies for a goal
    fn get_strategies_for_goal(
        &self,
        goal_bucket_id: u64,
        limit: usize,
    ) -> Vec<StrategyFeatures>;

    /// Get recent events for the session
    fn get_recent_events(
        &self,
        session_id: u64,
        limit: usize,
    ) -> Vec<EventFeatures>;

    /// Build policy features from current state
    fn build_policy_features(
        &self,
        goal_description: &str,
        context_fingerprint: u64,
    ) -> PolicyFeatures;
}

/// Strategy generator — produces K candidate strategies for a goal.
/// v1: LLM-based. Trait allows mock/deterministic implementations for testing.
#[async_trait]
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

#[derive(Debug, Clone)]
pub struct StrategyGenerationRequest {
    pub goal_description: String,
    pub goal_bucket_id: u64,
    pub context_fingerprint: u64,
    pub relevant_memories: Vec<MemoryContext>,
    pub similar_strategies: Vec<StrategyContext>,
    pub recent_events: Vec<EventContext>,
    pub constraints: Vec<String>,
    pub k_candidates: usize,
}

/// Simplified context structs for generator input (human-readable, not raw features)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryContext {
    pub summary: String,
    pub tier: String,
    pub strength: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyContext {
    pub name: String,
    pub summary: String,
    pub quality_score: f32,
    pub when_to_use: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventContext {
    pub event_type: String,
    pub action_name: Option<String>,
    pub outcome: Option<String>,
    pub timestamp: u64,
}

/// Action generator — produces N candidate next-actions under a strategy.
#[async_trait]
pub trait ActionGenerator: Send + Sync {
    async fn generate(
        &self,
        request: ActionGenerationRequest,
    ) -> Result<Vec<GeneratedActionPlan>, PlanningError>;

    /// Generate repair action after prediction error spike
    async fn repair(
        &self,
        current_step: &GeneratedStep,
        error_report: &PredictionErrorReport,
        request: &ActionGenerationRequest,
    ) -> Result<Vec<GeneratedActionPlan>, PlanningError>;
}

#[derive(Debug, Clone)]
pub struct ActionGenerationRequest {
    pub strategy: GeneratedStrategyPlan,
    pub current_step_index: usize,
    pub recent_events: Vec<EventContext>,
    pub context_fingerprint: u64,
    pub n_candidates: usize,
}

/// Execution sink — emits actions and records results.
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
/// Applied before any scoring — rejects malformed plans early.
pub trait PlanValidator: Send + Sync {
    /// Validate a strategy plan. Returns errors if invalid.
    fn validate_strategy(&self, plan: &GeneratedStrategyPlan) -> Vec<ValidationError>;

    /// Validate an action plan. Returns errors if invalid.
    fn validate_action(&self, plan: &GeneratedActionPlan) -> Vec<ValidationError>;
}

#[derive(Debug, Clone)]
pub struct ValidationError {
    pub field: String,
    pub message: String,
    pub severity: ValidationSeverity,
}

#[derive(Debug, Clone, Copy)]
pub enum ValidationSeverity {
    Warning,  // proceed with caution
    Error,    // reject this candidate
}

/// Planning errors
#[derive(Debug)]
pub enum PlanningError {
    GenerationFailed(String),
    ValidationFailed(Vec<ValidationError>),
    AllCandidatesRejected(String),
    LlmTimeout,
    LlmError(String),
    NotWarmedUp,
}
```

---

### Integration Hooks on GraphEngine

```rust
// In crates/agent-db-graph/src/integration/mod.rs (additions)

impl GraphEngine {
    /// Score a strategy candidate using the world model critic.
    /// Returns None if world model is disabled or not warmed up.
    pub async fn score_strategy(
        &self,
        policy: &PolicyFeatures,
        strategy: &StrategyFeatures,
    ) -> Option<CriticReport> { ... }

    /// Compute prediction error for an observed event (bottom-up).
    pub async fn compute_prediction_error(
        &self,
        event: &EventFeatures,
        memory: &MemoryFeatures,
        strategy: &StrategyFeatures,
        policy: &PolicyFeatures,
    ) -> Option<PredictionErrorReport> { ... }

    /// Generate strategy candidates for a goal (delegates to planning engine).
    pub async fn generate_strategies(
        &self,
        request: StrategyGenerationRequest,
    ) -> Result<Vec<ScoredCandidate>, PlanningError> { ... }

    /// Generate action candidates for a strategy step.
    pub async fn generate_actions(
        &self,
        request: ActionGenerationRequest,
    ) -> Result<Vec<ScoredAction>, PlanningError> { ... }

    /// Get world model statistics for monitoring.
    pub async fn get_world_model_stats(&self) -> Option<EnergyStats> { ... }
}

/// A strategy candidate with its critic score attached
#[derive(Debug, Clone)]
pub struct ScoredCandidate {
    pub plan: GeneratedStrategyPlan,
    pub report: CriticReport,
    pub decision: SelectionDecision,
}

#[derive(Debug, Clone)]
pub struct ScoredAction {
    pub plan: GeneratedActionPlan,
    pub energy: f32,
    pub feasibility: f32,  // from TransitionModel
}
```

---

## Training Tuple Construction Rules

**This is the most critical spec for training quality.** Noisy tuples → noisy critic → broken generator loop.

### Positive Tuple Assembly

A positive tuple is constructed from a **completed episode with known outcome**:

```
For each completed episode:
  1. EVENT features: from the episode's most significant event
     - event_type_hash: hash of event_type variant name
     - action_name_hash: hash of action_name (0 if not Action type)
     - context_fingerprint: episode.context_signature
     - outcome_success: 1.0 if episode succeeded, 0.0 if failed, 0.5 if partial
     - significance: episode.significance
     - temporal_delta_ns: median inter-event gap within episode
     - duration_ns: total episode duration

  2. MEMORY features: from the best matching memory for this episode's context
     - Query: retrieve_memories_by_context(episode.context_signature, 1)
     - If no memory exists: use default features (tier=0, strength=0.5, access=1)
     - tier, strength, access_count, context_fingerprint, goal_bucket_id from Memory

  3. STRATEGY features: from the strategy extracted from this episode (if any)
     - If strategy exists: use its quality_score, expected_success, etc.
     - If no strategy: use default features (quality=0.5, confidence=0.0)
     - goal_bucket_id, behavior_signature_hash from Strategy

  4. POLICY features: from the episode's context at start
     - goal_count: episode.context.active_goals.len()
     - top_goal_priority: max goal priority (or 0.5 if no goals)
     - resource_cpu_percent, resource_memory_bytes from context
     - context_fingerprint: episode.context_signature

  5. weight = episode.salience_score (prioritizes surprising/important episodes)
  6. is_positive = true
```

### Negative Tuple Assembly (Corruption Strategies)

For each positive tuple, generate `negatives_per_positive` (default: 4) corruptions:

| Corruption | How | What it teaches |
|-----------|-----|-----------------|
| **Temporal** | Replace event features with features from a random different episode | "This event doesn't belong in this context" |
| **Context** | Replace context_fingerprint in event+memory with a random different fingerprint | "This context is wrong for this strategy" |
| **Strategy** | Replace strategy features with features from a different goal_bucket | "This strategy doesn't fit this policy/goal" |
| **Outcome** | Flip outcome_success (1.0→0.0, 0.0→1.0) | "This outcome was unexpected given the strategy" |

Each corruption produces one negative tuple with `is_positive = false`, same weight as the positive.

### Assembly Rules

1. **Only assemble tuples from episodes with ≥ 3 events** (too-short episodes are noisy)
2. **Only use episodes where we have at least a memory OR strategy match** (pure defaults are uninformative)
3. **Deduplicate**: don't assemble tuples for the same (context_fingerprint, goal_bucket) pair more than 3x per training batch
4. **Temporal ordering**: older episodes first in batch (curriculum: learn common patterns before rare ones)

---

## Threshold & Selection Logic

### Use Novelty Z-Score for Decisions (Not Raw Energy)

Raw energy values drift during training. All accept/revise/reject decisions use **novelty_z** (z-score normalized energy) and **confidence**.

```rust
fn select(candidates: &[ScoredCandidate], config: &PlanningConfig) -> SelectionDecision {
    // Sort by total_energy (ascending = most compatible first)
    let best = candidates.iter().min_by(|a, b| {
        a.report.total_energy.partial_cmp(&b.report.total_energy).unwrap()
    });

    let report = &best.report;

    // If not warmed up, accept best candidate with a warning
    if !report.confidence > 0.0 {
        return Accept(best) // low confidence = can't reject
    }

    // Decision based on novelty z-score
    if report.novelty_z < config.accept_z {           // default: 1.0σ
        Accept(best)
    } else if report.novelty_z < config.revise_z {    // default: 2.0σ
        Revise { candidate: best, diagnostics: report }
    } else if report.novelty_z < config.reject_z {    // default: 3.0σ
        Experimental(best)  // novel but not impossible
    } else {
        Reject { reason: format!("z={:.1}", report.novelty_z), diagnostics: report }
    }
}
```

### Confidence Gate

If `confidence < min_confidence` (default: 0.3), the selector always accepts the best candidate regardless of energy — the model doesn't have enough evidence to reject. This prevents the world model from blocking viable strategies in sparse regions.

---

## Runtime Pipelines

### A. New Goal → Strategy Generation

```
on_new_goal(goal, context):
  1. policy = context_provider.build_policy_features(goal, context)
  2. memories = context_provider.get_memories_for_context(context, 20)
  3. strategies = context_provider.get_strategies_for_goal(goal_bucket, 10)
  4. events = context_provider.get_recent_events(session, 50)

  5. candidates = strategy_generator.generate(request{goal, memories, strategies, events, K=5})

  6. For each candidate:
     a. errors = plan_validator.validate_strategy(candidate)
        if errors.has_errors(): discard candidate
     b. strategy_features = extract_features(candidate)
        report = critic.score_strategy(policy, strategy_features)
        candidate.report = report

  7. decision = selector.select(scored_candidates, config)

  8. Match decision:
     Accept(plan) → emit to execution
     Revise{plan, diag} → strategy_generator.revise(plan, diag) → re-score → re-select (max 2 rounds)
     Experimental(plan) → emit with experimental flag
     Reject{reason} → log, return error to caller
```

### B. Strategy → Action Generation

```
on_strategy_step(strategy, step_index, context):
  1. actions = action_generator.generate(request{strategy, step_index, events, N=3})
  2. For each action:
     a. errors = plan_validator.validate_action(action)
     b. event_features = extract_predicted_features(action)
        energy = critic.score(policy, strategy, memory, event_features).total_energy
        feasibility = transition_model.posterior_success(action_as_transition)
        action.score = energy * 0.6 + (1.0 - feasibility) * 0.4
  3. Select lowest-score action, execute
```

### C. Execution Validation & Repair (Bidirectional)

```
on_event_observed(event, active_strategy, active_policy):
  1. event_features = extract_features(event)
  2. memory_features = get_active_memory_context()
  3. strategy_features = extract_features(active_strategy)

  4. error = critic.prediction_error(event_features, memory_features, strategy_features, policy)

  5. Log training example (always, for learning)

  6. If error.total_z > repair_threshold (default: 2.0σ):
     a. If error.mismatch_layer == Event:
        repair_actions = action_generator.repair(current_step, error, request)
        execute best repair action
     b. If error.mismatch_layer == Strategy OR 3 consecutive action repairs:
        revised = strategy_generator.revise(active_strategy, error_as_diagnostics)
        re-score → accept or abort
     c. If error.mismatch_layer == Policy:
        log "goal may need revision", continue (policy repair is future work)
```

### D. Learning / Training Loop

```
on_training_trigger():  // every N events or on consolidation
  1. Assemble training tuples from completed episodes (see rules above)
  2. Submit tuples to critic: critic.submit_training(tuple)
  3. Run training: avg_loss = critic.train_step()
  4. Log energy_stats for monitoring
  5. Persist world model state
```

---

## Feature Flags / Rollout Safety

```rust
pub struct PlanningConfig {
    // Master switches
    pub enable_world_model: bool,           // default: false
    pub enable_strategy_generation: bool,   // default: false
    pub enable_action_generation: bool,     // default: false
    pub repair_enabled: bool,               // default: false

    // Rollout modes
    pub world_model_mode: WorldModelMode,
    pub generation_mode: GenerationMode,

    // Selector thresholds (z-score based)
    pub accept_z: f32,                      // default: 1.0
    pub revise_z: f32,                      // default: 2.0
    pub reject_z: f32,                      // default: 3.0
    pub repair_z: f32,                      // default: 2.0
    pub min_confidence: f32,                // default: 0.3

    // Generation
    pub strategy_candidates_k: usize,       // default: 3
    pub action_candidates_n: usize,         // default: 2
    pub max_revision_rounds: usize,         // default: 2

    // Training
    pub warmup_episodes: u64,               // default: 100
    pub training_batch_size: usize,         // default: 64
    pub train_every_n_events: usize,        // default: 50
    pub learning_rate: f32,                 // default: 0.01
    pub contrastive_margin: f32,            // default: 1.0
    pub negatives_per_positive: usize,      // default: 4

    // LLM
    pub llm_model: String,                  // default: "gpt-4o-mini"
    pub llm_temperature: f32,               // default: 0.4
    pub llm_max_tokens: u32,                // default: 2048
}

pub enum WorldModelMode {
    Disabled,
    Shadow,                   // train + score + log only, zero impact
    ScoringOnly,              // score, expose via API, no selection
    ScoringAndReranking,      // score + rerank candidates
    Full,                     // score + rerank + repair + learn
}

pub enum GenerationMode {
    Disabled,
    GenerateOnly,             // generate candidates, log them, don't score
    GenerateAndScore,         // generate + score, log recommended selection
    GenerateScoreAndSelect,   // generate + score + select, don't execute
    Full,                     // generate + score + select + execute
}
```

**Key mode: `GenerateAndScore`** — generates candidates, scores them, **logs the recommended selection** but does NOT feed the executor. This is the safest way to validate the system against current behavior.

---

## Crate Layout

```
EventGraphDB/
├── ml/
│   ├── agent-db-world-model/
│   │   ├── Cargo.toml
│   │   └── src/
│   │       ├── lib.rs               # WorldModelCritic trait, public re-exports
│   │       ├── types.rs             # CriticReport, features, configs, EnergyStats
│   │       ├── encoders.rs          # EmbeddingTable, LinearLayer, 4 encoders
│   │       ├── energy.rs            # BilinearEnergy, total_free_energy, layer_energies
│   │       ├── training.rs          # Contrastive training, negative sampling, gradients
│   │       ├── scoring.rs           # score(), score_strategy(), prediction_error()
│   │       ├── persistence.rs       # Serialize/deserialize weights
│   │       └── ebm.rs              # EbmWorldModel — concrete impl of WorldModelCritic
│   │
│   └── agent-db-planning/
│       ├── Cargo.toml
│       └── src/
│           ├── lib.rs               # Traits: StrategyGenerator, ActionGenerator, etc.
│           ├── types.rs             # GeneratedStrategyPlan, GeneratedActionPlan, etc.
│           ├── validation.rs        # JSON schema validation, PlanValidator impl
│           ├── selector.rs          # Accept/revise/reject/experimental logic
│           ├── orchestrator.rs      # End-to-end pipeline coordination
│           ├── prompts.rs           # System prompts for LLM calls
│           ├── llm_generator.rs     # LLM-based StrategyGenerator + ActionGenerator
│           ├── mock_generator.rs    # Deterministic mock for testing
│           └── repair.rs            # PlanRepairer: action repair + strategy repair
│
├── crates/
│   ├── agent-db-core/
│   ├── agent-db-events/
│   ├── agent-db-storage/
│   ├── agent-db-graph/              # Modified: optional deps on ml/ crates
│   └── agent-db-ner/
├── server/
└── Cargo.toml                       # Add ml/* to workspace
```

### Dependency Graph

```
agent-db-planning
  ├── depends on → agent-db-world-model (for CriticReport, traits, feature types)
  ├── depends on → agent-db-core (for AgentId, Timestamp, etc.)
  └── depends on → agent-db-events (for Event, EventType)
  // NO reqwest — LLM HTTP calls via injected trait impl

agent-db-world-model
  ├── depends on → agent-db-core (for types)
  └── depends on → agent-db-storage (for serialize_versioned)
  // NO reqwest, NO heavy deps

agent-db-graph (modified, optional features)
  ├── optional dep → agent-db-world-model
  └── optional dep → agent-db-planning
  // Concrete LLM impls (OpenAI, Anthropic) stay in agent-db-graph
  // since it already has reqwest + the LlmClient infrastructure
```

**Note on LLM trait**: The existing `LlmClient` in `claims/llm_client.rs` is claim-specific. For planning, we define a more general `StrategyGenerator` trait in `agent-db-planning`. The concrete LLM adapter (which calls OpenAI/Anthropic) lives in `agent-db-graph` where `reqwest` already exists, and is injected into the planning engine at construction time. This keeps the planning crate free of networking dependencies.

---

## Implementation Phases (Revised Order)

### Phase 1: Types + Traits + World Model Crate (Critic)

**Goal**: Frozen interfaces + working scorer + persistence.

1. Create `ml/agent-db-world-model/` crate
2. Implement `types.rs`: all feature structs, CriticReport, PredictionErrorReport, EnergyStats
3. Implement `lib.rs`: `WorldModelCritic` trait, `WorldModelStateStore` trait
4. Implement `encoders.rs`: EmbeddingTable, LinearLayer, 4 encoders
5. Implement `energy.rs`: BilinearEnergy, total_free_energy, layer_energies
6. Implement `training.rs`: contrastive training, negative sampling, manual gradient descent
7. Implement `scoring.rs`: score(), score_strategy(), prediction_error()
8. Implement `persistence.rs`: serialize/deserialize weights
9. Implement `ebm.rs`: `EbmWorldModel` — concrete impl of WorldModelCritic
10. Unit tests for all components

### Phase 2: Planning Types + Selector + Mock Generator

**Goal**: End-to-end planning pipeline with deterministic candidates (no LLM yet).

1. Create `ml/agent-db-planning/` crate
2. Implement `types.rs`: GeneratedStrategyPlan, GeneratedActionPlan, all sub-types
3. Implement `lib.rs`: all boundary traits (StrategyGenerator, ActionGenerator, PlanningContextProvider, ExecutionSink, PlanValidator)
4. Implement `validation.rs`: JSON schema validation, constraint checking
5. Implement `selector.rs`: accept/revise/reject/experimental using novelty z-score + confidence
6. Implement `mock_generator.rs`: deterministic mock that returns fixed candidates
7. Implement `orchestrator.rs`: full pipeline (generate → validate → score → select)
8. Unit tests: mock generator → validator → scorer → selector → decision

### Phase 3: Shadow Mode Integration

**Goal**: World model trains on real data, zero impact on existing pipeline.

1. Add optional world-model dep to `agent-db-graph`
2. Add `WorldModel` field to `GraphEngine` (behind `enable_world_model` flag)
3. On `process_event()`: extract EventFeatures, compute energy, log
4. On episode completion: assemble training tuples (per construction rules), submit to critic
5. On consolidation: run `train_step()`
6. Persist/restore weights on lifecycle events (startup, shutdown, export/import)
7. Add `GET /api/world-model/stats` endpoint
8. Integration tests: process 100+ events → model trains → energy decreases for repeated patterns

### Phase 4: LLM Generator Adapter

**Goal**: Real strategy generation via LLM.

1. Implement `prompts.rs`: system prompts for strategy/action generation
2. Implement `llm_generator.rs`: LlmStrategyGenerator + LlmActionGenerator
   - Uses the existing `reqwest` + OpenAI/Anthropic infrastructure from `agent-db-graph`
   - Injected via trait at construction time
3. Implement `repair.rs`: revision/repair using critic diagnostics
4. Integration tests with mock LLM (record/replay HTTP responses)

### Phase 5: Scoring + Reranking Mode

**Goal**: Generate → Score → Log recommended selection (no execution).

1. Wire planning engine into `GraphEngine`
2. Implement `generate_strategies()` and `generate_actions()` on GraphEngine
3. Add `POST /api/planning/generate-strategy` endpoint
4. Add `POST /api/planning/score` endpoint (EBM scoring only)
5. Add `GenerateAndScore` mode: generates, scores, logs, but doesn't select
6. Integration tests: goal → candidates → scored → logged

### Phase 6: Execution + Repair Loop

**Goal**: Full bidirectional flow.

1. Implement execution validation: predicted vs actual comparison
2. Implement action repair loop
3. Implement strategy repair escalation
4. Wire bottom-up error propagation during live execution
5. End-to-end tests: goal → plan → execute → error → repair → learn

---

## Existing Code to Reuse

| Component | Location | Reuse |
|-----------|----------|-------|
| `LlmClient` trait + HTTP infra | `claims/llm_client.rs` | Pattern for LLM adapter; concrete OpenAI/Anthropic clients |
| `RefinementEngine.call_llm()` | `refinement.rs` | Pattern for structured JSON LLM calls |
| `TransitionModel` posteriors | `transitions.rs` | Action feasibility scoring |
| `Episode.prediction_error` / `salience_score` | `episodes.rs` | Training tuple weighting |
| `Strategy` struct (76 fields) | `strategies.rs` | Ground truth features for encoder calibration |
| `ContextHash` (BLAKE3) | `agent_db_core` | Embedding table keys |
| `serialize_versioned` / `deserialize_versioned` | `agent_db_storage` | Weight persistence |
| `retrieve_memories_by_context()` | `integration/queries.rs` | PlanningContextProvider impl |
| `get_similar_strategies()` | `integration/queries.rs` | PlanningContextProvider impl |
| `get_next_action_suggestions()` | `integration/queries.rs` | Baseline for action generation |

---

## Infrastructure

### CPU Only — No GPU

| Resource | Requirement |
|----------|-------------|
| **World model params** | ~30K (120 KB of f32) |
| **Scoring latency** | <10 μs per configuration |
| **Training batch** | <1 ms for 100 examples |
| **World model memory** | <1 MB |
| **LLM calls** | External API (OpenAI/Anthropic) — generation only, not scoring |
| **New Rust deps** | `rand` (already present) for world-model; `async-trait` for planning traits |
| **Python/CUDA/ONNX** | **Not required** |

---

## Verification

### Unit Tests (per crate)

**agent-db-world-model**:
- Encoder output dimensions correct for each layer
- BilinearEnergy computes correctly (manual example with known weights)
- Contrastive loss: positive energy < negative energy after N training steps
- Negative sampling produces valid corruptions per strategy
- Serialization roundtrip preserves all weights exactly
- Novelty z-score flags outliers (>2σ)
- Confidence computation: 0.0 before warmup, rises with support
- Bottom-up error: high for mismatched layers, low for matching

**agent-db-planning**:
- GeneratedStrategyPlan serializes/deserializes correctly
- Validation catches: empty steps, missing action_type, negative step numbers
- Selector: accepts at z < 1.0, revises at 1.0-2.0, experimental at 2.0-3.0, rejects at > 3.0
- Selector: accepts all when confidence < min_confidence (sparse region)
- Mock generator → validator → scorer → selector produces deterministic results
- Revision loop terminates within max_revision_rounds
- Orchestrator handles LLM timeout/failure gracefully

### Integration Tests

- Shadow mode: 100 events → model trains → energy stats populated → zero pipeline impact
- Full scoring: goal → generate 3 strategies → score → select lowest energy
- Repair: execute → inject unexpected event → repair triggered → revised plan scored
- Persistence: train model → export/import → weights preserved → scoring identical
- Novelty: train on A → present B → high z → train on B → z decreases
- GenerateAndScore mode: generates, scores, logs, does NOT execute

### Performance

- Single scoring: <100 μs
- Training batch of 100: <10 ms
- Total memory with world model: <2 MB
- No regression on `process_event()` in shadow mode

---

## Future Work (Not v1)

- **Reinforcement Learning**: Policy gradient on action selection using prediction errors as reward
- **Internal Generator**: Replace LLM with trained autoregressive strategy generator
- **Active Inference**: Choose actions to minimize expected free energy (exploration via epistemic value)
- **Dreamer-style Imagination**: Generate imagined trajectories for offline planning
- **Multi-Agent**: Cross-agent world models, shared energy landscapes
- **Policy Repair**: Currently policy layer is observe-only; future: automated goal revision
