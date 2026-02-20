# Patent Freedom-to-Operate (FTO) Analysis
## EventGraphDB — Experiential Learning System for Autonomous AI Agents

**Document Classification:** Patent-Level Technical Disclosure
**Date:** 2026-02-19 (updated: hierarchical predictive coding architecture)
**System:** EventGraphDB v1.0
**Crate Under Analysis:** `agent-db-graph` (core learning engine)

---

## Table of Contents

1. [System Overview & Architecture](#1-system-overview--architecture)
2. [Event Processing Pipeline](#2-event-processing-pipeline)
3. [Episode Boundary Detection](#3-episode-boundary-detection)
4. [Memory Formation & Hierarchy](#4-memory-formation--hierarchy)
5. [Memory Consolidation Engine](#5-memory-consolidation-engine)
6. [Strategy Extraction Pipeline](#6-strategy-extraction-pipeline)
7. [Two-Stage Strategy Synthesis (Template + LLM Distillation)](#7-two-stage-strategy-synthesis-template--llm-distillation)
8. [Executable Playbook Construction](#8-executable-playbook-construction)
9. [Strategy & Playbook Data Structures](#9-strategy--playbook-data-structures)
10. [Strategy Search & Retrieval Pipeline](#10-strategy-search--retrieval-pipeline)
11. [Bayesian Scoring & Ranking Algorithms](#11-bayesian-scoring--ranking-algorithms)
12. [Markov Transition Model (Policy Guide)](#12-markov-transition-model-policy-guide)
13. [Strategy Evolution & Supersession](#13-strategy-evolution--supersession)
14. [Contrastive Motif Distillation](#14-contrastive-motif-distillation)
15. [Claim Deduplication & Contradiction Detection](#15-claim-deduplication--contradiction-detection)
16. [Decision Trace Auditing](#16-decision-trace-auditing)
17. [Storage Architecture (Hot/Cold LRU + Persistent B+Tree)](#17-storage-architecture)
18. [Background Maintenance Engine](#18-background-maintenance-engine)
19. [Claim Outcome Feedback Loop](#19-claim-outcome-feedback-loop)
20. [Hierarchical Predictive Coding Engine](#20-hierarchical-predictive-coding-engine)
21. [Appendix A: Fully Populated Strategy Example (Before & After LLM Distillation)](#appendix-a)
22. [Appendix B: Embedding Gap Analysis](#appendix-b)
23. [Appendix C: Prior Art Comparison & Architectural Differentiation](#appendix-c)

---

## 1. System Overview & Architecture

### 1.1 Purpose

EventGraphDB is an experiential learning database for autonomous AI agents. It observes an agent's raw event stream and automatically forms episodic memories, extracts reusable strategies with executable playbooks, builds a Markov transition model for next-action prediction, and consolidates knowledge into a three-tier memory hierarchy — all without human annotation.

### 1.2 Crate Architecture

```
┌─────────────────┐   ┌──────────────────┐   ┌──────────────────┐
│ agent-db-core   │   │ agent-db-events  │   │ agent-db-storage │
│ (types, utils,  │   │ (Event, Context, │   │ (RedbBackend,    │
│  timestamps)    │   │  EventType)      │   │  LRU cache)      │
└────────┬────────┘   └────────┬─────────┘   └────────┬─────────┘
         │                     │                       │
         └─────────────────────┼───────────────────────┘
                               │
                    ┌──────────▼──────────┐
                    │  agent-db-graph     │
                    │  (Core Engine)      │
                    │                     │
                    │  ┌───────────────┐  │
                    │  │ Episodes      │  │
                    │  │ Memory        │  │
                    │  │ Strategies    │  │
                    │  │ Consolidation │  │
                    │  │ Transitions   │  │
                    │  │ Refinement    │  │
                    │  │ Maintenance   │  │
                    │  │ Claims        │  │
                    │  │ DecisionTrace │  │
                    │  └───────────────┘  │
                    └──────────┬──────────┘
                               │
                    ┌──────────▼──────────┐
                    │     server          │
                    │  (Axum REST API)    │
                    └─────────────────────┘
```

### 1.3 Core Data Flow

```
Raw Event Stream
      │
      ▼
Episode Boundary Detection (multi-signal state machine)
      │
      ▼
Episode Completion
      │
      ├──► Memory Formation (summary, takeaway, causal note)
      │         │
      │         ▼
      │    LLM Refinement (async, fire-and-forget)
      │         │
      │         ▼
      │    Embedding Generation (async)
      │
      ├──► Strategy Extraction (eligibility gate → playbook → scoring)
      │         │
      │         ▼
      │    LLM Distillation (async, rewrites 5 fields)
      │         │
      │         ▼
      │    Embedding Generation (async)
      │
      ├──► Transition Model Update (Markov/MDP)
      │
      └──► Decision Trace Logging (retrieved→used→outcome)
                │
                └──► Claim Outcome Feedback (success/failure → re-rank + avoidance generation)

Periodic:
      ├──► Memory Consolidation (Episodic→Semantic→Schema)
      ├──► Strategy Pruning & Merging
      └──► Maintenance (decay, cleanup, dedup)
```

---

## 2. Event Processing Pipeline

### 2.1 Event Types

The system processes six event types:

| EventType | Description | Fields |
|---|---|---|
| `Action` | Agent performs an action | `action_name`, `outcome: Success{result}/Failure{error}/Partial{issues}` |
| `Observation` | Agent observes external data | `observation_type`, `data` |
| `Cognitive` | Internal reasoning | `process_type: Planning/Reasoning/GoalFormation/Evaluation/LearningUpdate`, `content` |
| `Context` | External input received | `context_type`, `text` |
| `Communication` | Agent sends/receives messages | `message_type`, `content` |
| `Learning` | Explicit learning signal | `MemoryRetrieved`, `MemoryUsed`, `StrategyServed`, `StrategyUsed`, `ClaimRetrieved`, `ClaimUsed`, `Outcome` |

### 2.2 Event Context

Every event carries an `EventContext` containing:
- `environment`: key-value variables (env, region, user_id, etc.)
- `active_goals`: `Vec<Goal>` with `id`, `description`, `priority`
- `available_resources`: `Vec<Resource>` with name and type
- `fingerprint`: deterministic hash of environment + goals + resources
- `goal_bucket_id`: BLAKE3 hash of sorted goal IDs (priority excluded)
- `embeddings: Option<Vec<f32>>`: optional context embedding vector

### 2.3 Learning Outputs Contract

When an episode completes, the system produces a `LearningOutputs` bundle:

```rust
pub struct LearningOutputs {
    pub episode_record: EpisodeRecord,
    pub abstract_trace: AbstractTrace,
    pub context_features: ContextFeatures,
    pub outcome_signals: Vec<OutcomeSignal>,
}
```

The `AbstractTrace` reduces raw events into state-action-state transitions:

```rust
pub struct AbstractTransition {
    pub state: String,   // e.g. "Observe:test_results"
    pub action: String,  // e.g. "Act:deploy_service"
    pub next_state: String, // e.g. "Act:health_check"
}
```

The `behavior_signature` is computed by building a skeleton of event types (`Observe`, `Think:Planning`, `Act:run_tests`, etc.), joining with `>`, and hashing.

---

## 3. Episode Boundary Detection

### 3.1 Multi-Signal State Machine

The `EpisodeDetector` maintains one active episode per agent and uses a multi-signal approach to detect boundaries:

**Episode Start Signals:**
1. No active episode exists for this agent → start
2. Cognitive event with configurable start types: `GoalFormation`, `Planning`, `LearningUpdate`

**Episode End Signals (checked in order):**
1. **Feedback event** — explicit `LearningUpdate` cognitive event → immediate end
2. **Context shift** — context similarity drops below `context_shift_threshold` (default 0.4)
3. **Time gap** — elapsed time since last event exceeds `max_time_gap_ns` (default 1 hour)
4. **Outcome event** — `ActionOutcome::Success` or `ActionOutcome::Failure` after `min_events_before_outcome_end` (default 2) events
5. **Consecutive outcomes** — 2+ consecutive outcome events without intervening non-outcome events
6. **Periodic consolidation** — time since last consolidation exceeds `consolidation_interval_ns`

### 3.2 Significance Calculation

Each event receives a significance score via `calculate_significance()`:

```
significance = base_weight
             + goal_signal
             + chain_depth_signal
             + duration_signal
             + novelty_signal
             + event_type_signal
```

| Signal | Weight | Formula |
|---|---|---|
| Base | 0.30 | Fixed baseline |
| Goal alignment | +0.15 | If event has active goals |
| Chain depth | +0.10 | If event has parent chain > 2 |
| Duration | +0.08 | If action duration > 1000ms |
| Event type novelty | varies | `1/(1 + log₂(seen_count))` for event type string |
| Context novelty | varies | `1/(1 + log₂(seen_count))` for context hash |

**Episode-level significance** is updated incrementally with each event:

```
episode.significance = 0.7 × max(episode.significance, event_significance)
                     + 0.3 × weighted_avg(all event significances)
```

### 3.3 Late Event Correction

Events arriving within `late_event_window_ns` (default 5 seconds) after an episode completes are retroactively appended to the most recent completed episode if the context signature matches. The episode's significance and outcome are recalculated, and downstream stores (memory, strategy, transition model) are notified via `EpisodeUpdate::Corrected`.

### 3.4 Prediction Error & Salience

On episode completion:

```
prediction_error = |expected_quality - actual_quality|
  where expected_quality = 0.5 (neutral prior)
        actual_quality = 1.0 (Success), 0.3 (Failure), 0.6 (Partial), 0.4 (Interrupted)

salience_score = 0.4 × prediction_error + 0.3 × significance + 0.3 × goal_relevance
```

Higher prediction error (more surprising outcomes) produces stronger memories — this implements a prediction-error-weighted learning signal.

---

## 4. Memory Formation & Hierarchy

### 4.1 Memory Structure

```rust
pub struct Memory {
    pub id: MemoryId,
    pub agent_id: AgentId,
    pub session_id: SessionId,
    pub episode_id: EpisodeId,
    pub summary: String,           // LLM-refinable
    pub takeaway: String,          // LLM-refinable
    pub causal_note: String,       // LLM-refinable
    pub summary_embedding: Vec<f32>, // Populated async by RefinementEngine
    pub tier: MemoryTier,          // Episodic | Semantic | Schema
    pub consolidated_from: Vec<MemoryId>,
    pub schema_id: Option<MemoryId>,
    pub consolidation_status: ConsolidationStatus, // Active | Consolidated | Archived
    pub context: EventContext,
    pub memory_type: MemoryType,   // Episodic | Negative | Working | Semantic
    pub outcome: Option<EpisodeOutcome>,
    pub strength: f32,
    pub relevance_score: f32,
    pub access_count: u32,
    pub prediction_error: f32,
    pub salience_score: f32,
}
```

### 4.2 Memory Formation from Episodes

`MemoryFormation::form_memory()` synthesizes three natural-language fields:

1. **`synthesize_memory_summary()`** — walks events for goals, context, actions, observations, and outcome; concatenates into a structured narrative
2. **`synthesize_takeaway()`** — finds the pivotal action (last success or last failure) and frames it as a lesson
3. **`synthesize_causal_note()`** — analyzes the causal chain of action outcomes to explain why the episode succeeded or failed

Initial strength is computed from episode significance weighted by prediction error:

```
strength = significance × (1.0 + prediction_error × 0.3)
```

### 4.3 Three-Tier Memory Hierarchy

| Tier | Description | Source |
|---|---|---|
| **Episodic** | Raw individual experiences | Direct from episodes |
| **Semantic** | Generalized knowledge from multiple episodes | Consolidation Phase 1 |
| **Schema** | Reusable mental models from multiple semantic memories | Consolidation Phase 2 |

### 4.4 Memory Retrieval

**Context-based retrieval** (`retrieve_by_context`): exact match on context fingerprint hash.

**Similarity-based retrieval** (`retrieve_by_context_similar`): embedding cosine similarity with structured fallback (environment 0.4, goals 0.3, resources 0.3).

**Hierarchical retrieval** (`retrieve_hierarchical`): multi-signal scoring with tier boost:

```
sim = max(fingerprint_match, embedding_cosine, bucket_match)

tier_boost = {
    Schema:   +0.30
    Semantic: +0.15
    Episodic: +0.00
}

final_score = sim + tier_boost
```

Where `bucket_match` = 0.5 if the memory shares the same goal bucket (same set of active goals), enabling cross-context retrieval within the same goal-set equivalence class.

---

## 5. Memory Consolidation Engine

### 5.1 Two-Phase Consolidation

**Phase 1: Episodic → Semantic**

1. Group all Active Episodic memories by `goal_bucket_id`
2. For each bucket with ≥ `episodic_threshold` (default 3) unconsolidated memories:
   - Synthesize a Semantic memory: average strength, average relevance, merged takeaways/causal notes
   - Strength boost: `avg_strength × 1.1` (semantic memories are slightly stronger)
   - Mark source memories as `Consolidated` with decay factor `post_consolidation_decay` (default 0.3)

**Phase 2: Semantic → Schema**

Three grouping modes:

| Mode | Algorithm |
|---|---|
| `ExactFingerprint` | Group by identical context fingerprint hash |
| `EmbeddingCentroid` (default) | Greedy centroid clustering: seed by strongest memory, running mean centroid, dual threshold (vs centroid AND vs seed) |
| `EmbeddingMutual` | Complete-link clustering: every member must meet `schema_similarity_threshold` (default 0.80) against every other member |

**Centroid clustering algorithm:**

```
1. Sort candidate semantic memories by strength (descending)
2. For each candidate (seed):
   a. Initialize centroid = seed.embedding
   b. For each remaining candidate:
      - Compute cosine_similarity(candidate.embedding, centroid)
      - Compute cosine_similarity(candidate.embedding, seed.embedding)
      - If BOTH ≥ schema_similarity_threshold → add to group, update centroid as running mean
   c. If group.size ≥ semantic_threshold → create Schema memory
3. Schema strength boost: avg_strength × 1.2
```

### 5.2 Embedding Source Priority

For consolidation similarity computation:

```
fn best_embedding(memory) -> Vec<f32> {
    if !memory.summary_embedding.is_empty() {
        return memory.summary_embedding;  // Prefer LLM-refined embedding
    }
    memory.context.embeddings  // Fall back to context embedding
}
```

---

## 6. Strategy Extraction Pipeline

### 6.1 Extraction Trigger

Strategy extraction occurs on every completed episode via `StrategyExtractor::extract_from_episode()`.

### 6.2 Idempotency

If the episode has already been processed (tracked via `episode_index: HashMap<EpisodeId, StrategyId>`), the system performs an **outcome correction** instead of creating a new strategy — adjusting success/failure counts on the existing strategy and recalculating Bayesian metrics.

### 6.3 Strategy Type Classification

```
EpisodeOutcome::Success     → StrategyType::Positive    ("do this")
EpisodeOutcome::Failure     → StrategyType::Constraint  ("avoid this")
EpisodeOutcome::Partial     → StrategyType::Positive
EpisodeOutcome::Interrupted → StrategyType::Constraint
```

### 6.4 Eligibility Gate

Before extraction, every episode must pass an eligibility score threshold:

```
eligibility = w_novelty      × novelty
            + w_outcome_util  × outcome_utility
            + w_difficulty    × difficulty
            + w_reuse_potential × reuse_potential
            - w_redundancy    × redundancy
```

| Factor | Weight | Formula |
|---|---|---|
| Novelty | 0.25 | `1 / (1 + context_count)` — how often this (agent, goal_bucket, context_hash) has been seen |
| Outcome Utility | 0.25 | Success=1.0, Partial=0.7, Failure=0.8, Interrupted=0.4 |
| Difficulty | 0.20 | `min(events.len() / 10, 1.0)` — longer episodes = harder tasks |
| Reuse Potential | 0.20 | `min(bucket_count / 10, 1.0)` — how often this goal bucket recurs |
| Redundancy | 0.10 | 1.0 if identical behavior_signature exists in this bucket, else 0.0 |

**Threshold:** `eligibility_threshold = 0.5` (default). Episodes below this are rejected.

### 6.5 Goal Bucket Computation

Strategies are partitioned by goal-set equivalence class using BLAKE3:

```
fn compute_goal_bucket_id(goal_ids: &[u64]) -> u64 {
    if goal_ids.is_empty() { return 0; }
    let mut sorted = goal_ids.to_vec();
    sorted.sort();
    let canonical_bytes = sorted.flat_map(|id| id.to_le_bytes());
    let hash = blake3::hash(&canonical_bytes);
    u64::from_le_bytes(hash[0..8])
}
```

**Priority is deliberately excluded** so that minor priority changes do not split strategies into separate buckets.

### 6.6 Behavior Signature

A deterministic hash of the event skeleton:

```
skeleton = events.map(|e| match e.event_type {
    Observation => "Observe",
    Cognitive(t) => "Think:{t:?}",
    Action(name) => "Act:{name}:{tool}",
    Communication => "Communicate",
    Learning => "Learn",
    Context(t) => "Context:{t}",
})

behavior_signature = hex(hash(skeleton.join(">")))
```

### 6.7 Strategy Deduplication via Signature

Each strategy has a `strategy_signature = hash(precondition | action_hint | strategy_type)`. On storage:

1. If a strategy with the same signature exists → **merge**: increment support_count, add success/failure counts, recalculate Bayesian expected_success
2. If no matching signature → **create new** strategy and index it

---

## 7. Two-Stage Strategy Synthesis (Template + LLM Distillation)

### 7.1 Stage 1: Deterministic Template Synthesis (Immediate)

Six synthesis functions run at extraction time with zero external dependencies:

#### 7.1.1 `synthesize_strategy_summary()`

Produces a structured natural-language summary by walking the episode's events:

```
"DO this when applicable. When: {goals}. Context: {env_vars}.
 Steps: 1. validate_config -> {"valid": true} → 2. Think (Planning) → 3. run_tests [partial] → ...
 Success looks like: {success_indicators}.
 Avoid: {failure_patterns}.
 This episode {outcome} ({n} events, significance {s}%)."
```

Steps are capped at 8 in the summary. Each action includes its outcome inline (`-> result`, `[FAILED: error]`, `[partial]`).

#### 7.1.2 `synthesize_when_to_use()`

For Positive strategies:
```
"Use when the goal is: {goals}. Context matches: {env_vars}. Best for episodes with significance ≥ {s}%"
```

For Constraint strategies:
```
"Watch out when the goal is: {goals}. Applies when the agent is about to repeat a pattern that previously failed"
```

#### 7.1.3 `synthesize_when_not_to_use()`

For Positive strategies: cites fragile conditions (failed actions in episode), low significance, and context divergence warnings.

For Constraint strategies:
```
"Safe to ignore when a newer strategy explicitly supersedes this constraint, or when the failure pattern is no longer applicable to the current context."
```

#### 7.1.4 `synthesize_failure_modes()`

Scans events for `ActionOutcome::Failure` and `ActionOutcome::Partial`, producing entries like:
```
"Action 'health_check' can fail: connection timeout after 30s"
"Action 'run_tests' partially succeeds with issues: ["flaky test: test_auth_timeout"]"
```

Capped at 5 failure modes.

#### 7.1.5 `synthesize_counterfactual()`

Generates a what-if analysis based on outcome and failure points:
- Success with failures: "Despite failures at X, the episode recovered. Skipping those steps could have been faster."
- Failure with failures: "An alternative approach for X might have avoided the failure."
- Failure without failures: "Episode failed without clear action errors; the approach may need rethinking."

#### 7.1.6 `build_playbook()` — See Section 8

### 7.2 Stage 2: LLM Distillation (Async, Fire-and-Forget)

After the template is stored, the `RefinementEngine` asynchronously rewrites 5 fields using an LLM.

#### 7.2.1 LLM Configuration

```rust
pub struct RefinementConfig {
    pub enable_llm_refinement: bool,    // default: false
    pub enable_summary_embedding: bool, // default: true
    pub model: String,                  // default: "gpt-4o-mini"
    pub max_tokens: u32,                // default: 512
    pub temperature: f32,               // default: 0.3
}
```

#### 7.2.2 System Prompt

```
You are an expert at codifying AI agent strategies into reusable playbooks.
Given raw strategy data, produce a JSON response with:
1. "summary": A clear 2-3 sentence description of this strategy
2. "when_to_use": Specific conditions where this strategy applies (1-2 sentences)
3. "when_not_to_use": Conditions where this strategy should NOT be used (1-2 sentences)
4. "failure_modes": Array of known failure modes, each as a short sentence (max 3)
5. "counterfactual": What would have happened differently with an alternative approach (1 sentence)

Be specific. Reference actual action names, context patterns, and outcomes.
Output ONLY valid JSON, no markdown fences.
```

#### 7.2.3 User Prompt (Constructed from Template Fields)

```
Raw strategy data:

Summary: {strategy.summary}
When to use: {strategy.when_to_use}
When not to use: {strategy.when_not_to_use}
Action hint: {strategy.action_hint}
Precondition: {strategy.precondition}
Success rate: {expected_success * 100}%
Quality: {quality_score}
Failure patterns: {failure_patterns:?}
Playbook steps: {playbook.len()}
```

#### 7.2.4 LLM API Call

```
POST https://api.openai.com/v1/chat/completions
Authorization: Bearer {api_key}
Content-Type: application/json

{
    "model": "gpt-4o-mini",
    "messages": [
        {"role": "system", "content": STRATEGY_REFINEMENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt}
    ],
    "temperature": 0.3,
    "max_tokens": 512,
    "response_format": {"type": "json_object"}
}
```

#### 7.2.5 Fields Overwritten by LLM

| Field | Before (Template) | After (LLM) |
|---|---|---|
| `summary` | Mechanical step listing | Concise 2-3 sentence narrative |
| `when_to_use` | Raw goal/env listing | Specific applicability conditions |
| `when_not_to_use` | Generic warnings | Targeted exclusion criteria |
| `failure_modes` | Raw error strings | Actionable failure descriptions |
| `counterfactual` | Template what-if | Specific alternative suggestion |

#### 7.2.6 Fields NOT Overwritten (Remain Template/Algorithmic)

| Field | Source | Unchanged |
|---|---|---|
| `playbook` | `build_playbook()` | Yes |
| `reasoning_steps` | `extract_reasoning_steps()` | Yes |
| `context_patterns` | `extract_context_patterns()` | Yes |
| `precondition` | `extract_behavior_skeleton()` | Yes |
| `action_hint` | `extract_behavior_skeleton()` | Yes |
| All Bayesian metrics | `derive_calibrated_metrics()` | Yes |
| `behavior_signature` | Hash of event skeleton | Yes |
| `metadata` (goal_ids, tool_names, etc.) | `extract_graph_signature()` | Yes |
| `success_indicators` | Pattern matching on outcomes | Yes |
| `failure_patterns` | Pattern matching on failures | Yes |

#### 7.2.7 Embedding Generation

After LLM refinement (or using the template summary if LLM is unavailable):

```rust
if config.enable_summary_embedding {
    let embedding = embedding_client.embed(EmbeddingRequest {
        text: updated.summary,
        context: None,
    }).await;
    updated.summary_embedding = embedding;
}
```

#### 7.2.8 Memory Refinement (Same Pipeline)

Memories undergo the same two-stage process with a different system prompt:

```
You are an expert at distilling AI agent experiences into concise, actionable knowledge.
Given raw event data from an agent's episode, produce a JSON response with:
1. "summary": A clear 2-3 sentence narrative of what happened (no jargon, no IDs)
2. "takeaway": The single most important lesson from this experience (1 sentence)
3. "causal_note": Why the outcome was what it was — identify the key causal factors (1-2 sentences)
```

Refined memory fields: `summary`, `takeaway`, `causal_note`. Then `summary_embedding` is generated from the refined summary.

---

## 8. Executable Playbook Construction

### 8.1 Playbook Generation Algorithm

`build_playbook()` walks the episode's raw events and produces an ordered sequence of `PlaybookStep` structs:

| Event Type | Playbook Step |
|---|---|
| `Action{name, outcome}` | `"Execute '{name}'"` with outcome-dependent recovery/branches |
| `Context{type, text}` | `"Receive [{type}]: {text}"` with `skip_if: "No input available"` |
| `Observation{type}` | `"Observe [{type}]"` |
| `Cognitive{process_type}` | `"Think ({process_type:?})"` |
| `Communication`, `Learning` | Skipped (not actionable) |

**Capped at 12 steps** maximum.

### 8.2 Branching and Recovery Logic

For each `Action` event, the playbook injects:

| Action Outcome | Injection |
|---|---|
| `Failure{error}` | `recovery: "On failure ({error}): retry or use alternative approach"` |
| `Partial{issues}` | `recovery: "On partial success: address {issues}"` |
| `Success` + Constraint strategy | `branch: {condition: "If this pattern appears", action: "Skip '{name}' and use alternative"}` |

### 8.3 PlaybookStep Data Structure

```rust
pub struct PlaybookStep {
    pub step: u32,                      // 1-indexed sequence number
    pub action: String,                 // What to do
    pub condition: String,              // Prerequisite condition
    pub skip_if: String,                // When to skip this step
    pub branches: Vec<PlaybookBranch>,  // Conditional branches
    pub recovery: String,               // Fallback on failure
    pub step_id: String,                // Unique ID for non-linear navigation
    pub next_step_id: Option<String>,   // Override sequential ordering
}

pub struct PlaybookBranch {
    pub condition: String,              // Trigger condition
    pub action: String,                 // Alternative action
    pub next_step_id: Option<String>,   // Jump target
}
```

### 8.4 Reasoning Step Generalization

`extract_reasoning_steps()` processes cognitive events to produce generalized, reusable reasoning steps:

1. **Path abstraction**: File paths → `<file_path>`
2. **Error abstraction**: Error messages → `<error_type>: <error_message>`
3. **URL abstraction**: URLs → `<url>`
4. **Number abstraction**: Large numbers (>100) → `<number>`
5. **Structure tagging**: Reasoning patterns tagged as `[IF-THEN]`, `[ERROR-HANDLING]`, `[DECOMPOSE]`, `[VERIFY]`, `[SEARCH]`

Applicability classification:
- `"general"` — tagged with abstract pattern (reusable across domains)
- `"contextual"` — contains placeholders (parameterized)
- `"specific"` — concrete, no abstraction applied

---

## 9. Strategy & Playbook Data Structures

### 9.1 Complete Strategy Struct

```rust
pub struct Strategy {
    // ── Identity ──
    pub id: StrategyId,                    // u64
    pub name: String,                      // e.g. "strategy_1001_ep_7"

    // ── LLM-Retrievable Fields (overwritten by Stage 2 distillation) ──
    pub summary: String,
    pub when_to_use: String,
    pub when_not_to_use: String,
    pub failure_modes: Vec<String>,
    pub playbook: Vec<PlaybookStep>,       // NOT overwritten by LLM
    pub counterfactual: String,
    pub supersedes: Vec<StrategyId>,
    pub applicable_domains: Vec<String>,
    pub lineage_depth: u32,
    pub summary_embedding: Vec<f32>,       // Populated async

    // ── Core Fields ──
    pub agent_id: AgentId,
    pub reasoning_steps: Vec<ReasoningStep>,
    pub context_patterns: Vec<ContextPattern>,
    pub success_indicators: Vec<String>,
    pub failure_patterns: Vec<String>,
    pub quality_score: f32,                // success / (success + failure)
    pub success_count: u32,
    pub failure_count: u32,
    pub support_count: u32,
    pub strategy_type: StrategyType,       // Positive | Constraint
    pub precondition: String,
    pub action_hint: String,
    pub expected_success: f32,             // Bayesian posterior
    pub expected_cost: f32,
    pub expected_value: f32,               // +/- depending on type
    pub confidence: f32,                   // Exponential saturation
    pub contradictions: Vec<String>,
    pub goal_bucket_id: u64,
    pub behavior_signature: String,
    pub source_episodes: Vec<Episode>,     // Not persisted
    pub created_at: Timestamp,
    pub last_used: Timestamp,
    pub metadata: HashMap<String, String>, // Jaccard search index

    // ── Phase 1 Upgrades ──
    pub self_judged_quality: Option<f32>,
    pub source_outcomes: Vec<EpisodeOutcome>,
    pub version: u32,
    pub parent_strategy: Option<StrategyId>,
}
```

### 9.2 Supporting Structures

```rust
pub struct ReasoningStep {
    pub description: String,        // Generalized step (may contain placeholders)
    pub applicability: String,      // "general" | "contextual" | "specific"
    pub expected_outcome: Option<String>,
    pub sequence_order: usize,
}

pub struct ContextPattern {
    pub environment_type: Option<String>,
    pub task_type: Option<String>,
    pub resource_constraints: Vec<String>,
    pub goal_characteristics: Vec<String>,
    pub match_confidence: f32,
}

pub enum StrategyType {
    Positive,    // "do this" — success-correlated
    Constraint,  // "avoid this" — failure-correlated
}
```

### 9.3 Metadata Map (Jaccard Search Index)

The `metadata: HashMap<String, String>` stores JSON-encoded feature sets used for similarity search:

```json
{
    "goal_ids": "[42, 99]",
    "tool_names": "[\"validate_config\", \"run_tests\", \"deploy_service\"]",
    "result_types": "[\"deployment_success\", \"test_report\"]",
    "goal_bucket_id": "14298376502",
    "behavior_signature": "a3f7c2e1d904b6",
    "strategy_signature": "e8b2f1a09c3d"
}
```

---

## 10. Strategy Search & Retrieval Pipeline

### 10.1 Three Search Modes

| Endpoint | Method | Purpose |
|---|---|---|
| `/api/strategies/agent/:agent_id` | GET | All strategies for a specific agent |
| `/api/strategies/similar` | POST | Similarity search by feature signature |
| `/api/suggestions` | GET | Next-action suggestions (Policy Guide / Markov) |

### 10.2 Mode A: Agent-Based Retrieval

**Path:** `API → StrategyStore::get_agent_strategies → Agent Index Scan → Strategy Load`

1. Prefix-scan the `strategy_feature_postings` redb table with `agent_id` as key prefix
2. Key format: `[agent_id (8 bytes BE) | strategy_id (8 bytes BE)]` → value: `quality_score: f32`
3. For each matching key, extract `strategy_id` from bytes `[8..16]`
4. Load each strategy through the Hot/Cold LRU cache

### 10.3 Mode B: Similarity-Based Retrieval

**Path:** `API → StrategySimilarityQuery → Candidate Selection → Weighted Jaccard → Threshold + Ranking`

**Query structure:**
```rust
pub struct StrategySimilarityQuery {
    pub goal_ids: Vec<u64>,
    pub tool_names: Vec<String>,
    pub result_types: Vec<String>,
    pub context_hash: Option<ContextHash>,
    pub agent_id: Option<AgentId>,
    pub min_score: f32,     // default 0.2
    pub limit: usize,
}
```

#### Step 1: Goal Bucket Computation

```
goal_bucket_id = BLAKE3(sort(goal_ids).flat_map(to_le_bytes))[0..8] as u64
```

#### Step 2: Tiered Candidate Narrowing

```
if context_hash is provided:
    candidates = context_index[context_hash]        // Exact context match
else if goal_bucket_id != 0:
    candidates = goal_bucket_index[goal_bucket_id]  // Same goal-set bucket
else:
    candidates = ALL strategy IDs                   // Full scan fallback
```

Optional agent filter: `strategy.agent_id == query.agent_id`

#### Step 3: Feature Extraction

For each candidate, parse the `metadata` map:
```
goal_ids    = JSON::parse(strategy.metadata["goal_ids"])    → HashSet<u64>
tool_names  = JSON::parse(strategy.metadata["tool_names"])  → HashSet<String>
result_types = JSON::parse(strategy.metadata["result_types"]) → HashSet<String>
```

#### Step 4: Weighted Jaccard Similarity

```
Jaccard(A, B) = |A ∩ B| / |A ∪ B|

score = (goal_weight × Jaccard(query_goals, strategy_goals)
       + tool_weight × Jaccard(query_tools, strategy_tools)
       + result_weight × Jaccard(query_results, strategy_results))
       / weight_sum
```

| Feature | Weight | Rationale |
|---|---|---|
| Goals | **0.5** | Goal alignment is strongest signal |
| Tools | **0.3** | Tool availability constrains executability |
| Results | **0.2** | Result type is secondary signal |

If a query feature set is empty, its weight is set to 0.0 and excluded from `weight_sum`.

#### Step 5: Threshold + Ranking

1. Discard candidates with `score < min_score` (default 0.2)
2. Sort descending by score
3. Return top `limit` results as `Vec<(Strategy, f32)>`

### 10.4 Mode C: Policy Guide (Next-Action Suggestions)

See Section 12 (Markov Transition Model).

### 10.5 Current Retrieval Modality Gap

**Embedding-based strategy search is plumbed but not active:**

- The `Strategy` struct has `summary_embedding: Vec<f32>` (populated by `RefinementEngine`)
- The field comment reads: `"Embedding of the summary text for semantic retrieval"`
- **No code path performs cosine similarity over strategy embeddings**
- The `find_similar_strategies()` method uses only Jaccard over discrete feature sets

By contrast, **memory retrieval does use embeddings** — `retrieve_hierarchical()` computes `cosine_similarity(query_emb, m_emb)` with tier boosting.

---

## 11. Bayesian Scoring & Ranking Algorithms

### 11.1 Quality Score (Frequentist)

```
quality_score = success_count / (success_count + failure_count)
```

Updated on each outcome event.

### 11.2 Expected Success (Bayesian Posterior with Laplace Smoothing)

```
expected_success = (success_count + α) / (support_count + α + β)
```

Default priors: `α = 1.0`, `β = 1.0` (uniform Laplace smoothing). This prevents extreme estimates from small samples — a strategy with 1 success / 0 failures gets `expected_success = 2/3 ≈ 0.67`, not `1.0`.

### 11.3 Expected Value (Signed Utility)

```
expected_value = {
    Positive:   +expected_success    // "do this" — positive utility
    Constraint: -expected_success    // "avoid this" — negative utility
}
```

This sign convention allows strategies to be ranked on a single axis where positive values encourage actions and negative values discourage them.

### 11.4 Confidence (Exponential Saturation)

```
confidence = 1 - e^(-support_count / 3)
```

| Support Count | Confidence |
|---|---|
| 1 | 0.283 |
| 3 | 0.632 |
| 5 | 0.811 |
| 10 | 0.964 |
| 17 | 0.997 |

Asymptotically approaches 1.0 — requires ~10 episodes to reach high confidence.

### 11.5 Initial Quality with Prediction Error Weighting

At extraction time:
```
quality_with_prediction = significance × (1.0 + prediction_error × 0.3)
```

Surprising outcomes (high prediction error) produce higher initial quality scores, biasing the system toward remembering novel experiences.

### 11.6 Transition Model Posterior (Markov/MDP)

```
P(success | state, action, next_state) =
    (prior_success + success_count) / (prior_success + prior_failure + success_count + failure_count)
```

Default priors: `prior_success = 1.0`, `prior_failure = 3.0` (skeptical prior — requires evidence before recommending an action).

---

## 12. Markov Transition Model (Policy Guide)

### 12.1 Architecture

The `TransitionModel` maintains per-goal-bucket transition statistics:

```rust
pub struct TransitionModel {
    config: TransitionModelConfig,
    buckets: HashMap<u64, HashMap<TransitionKey, TransitionStats>>,
    episode_transitions: HashMap<EpisodeId, Vec<TransitionKey>>,
    episode_outcomes: HashMap<EpisodeId, bool>,
    episode_goal_bucket: HashMap<EpisodeId, u64>,
}

struct TransitionKey {
    state: String,       // e.g. "Observe:test_results"
    action: String,      // e.g. "Act:deploy_service"
    next_state: String,  // e.g. "Act:health_check"
}

pub struct TransitionStats {
    pub count: u64,
    pub success_count: u64,
    pub failure_count: u64,
}
```

### 12.2 Idempotent Update

`update_from_trace()` is idempotent — if the same episode is re-processed:
1. Decrement counts for the previous outcome
2. Increment counts for the new outcome

This supports late event corrections without double-counting.

### 12.3 Posterior Computation

```
P(success) = (prior_success + success_count) / (prior_success + prior_failure + success_count + failure_count)
```

Clamped to `[0.0, 1.0]`.

### 12.4 Policy Guide API

`GET /api/suggestions?context_hash={hash}&last_action_node={id}&limit={n}`

Returns ranked next actions with:
- `action_name`: the suggested action
- `success_probability`: Bayesian posterior
- `evidence_count`: number of historical transitions
- `reasoning`: explanation of why this action is suggested

### 12.5 Maintenance

- `cleanup_oldest_episodes(keep)`: Removes oldest episodes by ID, decrementing their transition counts
- `prune_weak_transitions(min_count)`: Removes transitions with `count < min_count`, then removes empty buckets

---

## 13. Strategy Evolution & Supersession

### 13.1 Detection

`StrategyEvolution::detect_supersession()` identifies when a new strategy makes an older one obsolete:

**Conditions for supersession:**
1. Same goal bucket (`goal_bucket_id` matches)
2. Same behavior signature (`behavior_signature` matches)
3. New strategy has higher quality score
4. New strategy has higher support count

### 13.2 Merge on Duplicate Signature

When `store_strategy()` finds an existing strategy with the same `strategy_signature`:

```rust
existing.support_count += new.support_count;
existing.success_count += new.success_count;
existing.failure_count += new.failure_count;
existing.last_used = now();
existing.source_episodes.append(new.source_episodes);
existing.source_outcomes.append(new.source_outcomes);

// Recalculate Bayesian metrics
existing.expected_success = (success + α) / (support + α + β);
existing.confidence = 1 - e^(-support / 3);
```

---

## 14. Contrastive Motif Distillation

### 14.1 Purpose

The contrastive distiller identifies **statistically significant action patterns (motifs)** that discriminate successful from failed episodes within the same goal bucket. These become distilled strategies.

### 14.2 Trigger

Runs every `distill_every` (default 5) episodes per (agent, goal_bucket) pair.

### 14.3 Three Motif Classes

| Class | Extraction | Example |
|---|---|---|
| **Transition** | Consecutive event pairs `(event_i, event_{i+1})` | `"Act:validate_config→Act:run_tests"` |
| **Anchor** | Window of ±k events around each event | `"Think:Planning±Act:validate_config"` |
| **Macro** | N-grams of length 3–6 over event skeleton | `"Act:validate→Act:run_tests→Act:deploy"` |

### 14.4 Log-Odds Lift

For each motif, compute lift (enrichment in success vs. failure episodes):

```
p_s = (success_count + α) / (total_success_episodes + α + β)
p_f = (failure_count + α) / (total_failure_episodes + α + β)

lift = ln(p_s / p_f)    // log-odds ratio
uplift = p_s - p_f       // absolute difference
```

**Thresholds:** `min_lift ≥ 1.0`, `min_uplift ≥ 0.10`

### 14.5 Holdout Validation with Drift Detection

The episode cache is split into halves for validation:

1. Train on first half, compute lift
2. Validate on second half, compute lift
3. **Drift detection:** If `train_lift - validate_lift > drift_max_drop` (default 0.25) → reject motif

### 14.6 Conflict Resolution

When both a Positive motif and a Constraint motif exist for the same pattern, the one with higher absolute lift wins, provided the margin exceeds `conflict_margin` (default 0.1).

### 14.7 Distilled Strategy Output

Motifs that pass validation become strategies with:
- `strategy_type`: Positive or Constraint based on lift sign
- `action_hint`: "Motif: {motif_key}" or "Avoid motif: {motif_key}"
- `expected_success`: Bayesian posterior from motif stats
- `when_to_use` / `when_not_to_use`: generated from lift statistics
- `playbook: Vec::new()` — distiller strategies are motif-level, no step playbook

---

## 15. Claim Deduplication & Contradiction Detection

### 15.0 Claim Type Taxonomy

Each claim is typed to control its decay rate and retrieval behavior:

| Type | Half-life | Description |
|---|---|---|
| `Preference` | 30 days | Personal taste — "I like X" |
| `Fact` | 365 days | Objective statement — "X costs $10" |
| `Belief` | 14 days | Uncertain opinion — "I think X will work" |
| `Intention` | 3 days | Desired future action — "I plan to Y" |
| `Capability` | 180 days | System/agent ability — "The system supports X" |
| `Avoidance` | 14 days | Negative lesson — "Don't do X", auto-generated from repeated failures (see Section 19) |

The `Avoidance` type is automatically generated when the claim outcome feedback loop (Section 19) detects a claim that has accumulated ≥ 2 negative outcomes. It reuses the source claim's embedding and entities for retrieval continuity.

### 15.1 Dedup Pipeline

`check_claim_dedup()` runs on every new claim insertion:

1. Find the most similar existing claim via vector index (`find_similar(embedding, k=1, min_threshold=contradiction_threshold)`)
2. Load the existing claim text
3. Check for contradiction (negation flip)
4. If similarity ≥ `claim_dedup_threshold` (default 0.92) → **Duplicate** (merge)
5. If similarity ≥ `claim_contradiction_threshold` (default 0.85) AND negation detected → **Contradiction** (supersede)
6. Otherwise → **NewClaim**

### 15.2 Negation-Based Contradiction Detection

`is_contradiction()` uses a 23-word negation lexicon:

```
NEGATION_WORDS = ["not", "no", "don't", "doesn't", "never", "neither", "nor",
    "isn't", "aren't", "wasn't", "weren't", "won't", "wouldn't", "couldn't",
    "shouldn't", "cannot", "can't", "haven't", "hasn't", "hadn't", "didn't",
    "dislike", "hate", "refuse", "reject"]
```

**Algorithm:** Tokenize both claims, check if exactly one contains a negation word (XOR). This catches:
- "I like X" vs "I do not like X" → contradiction
- "I like X" vs "I like Y" → not contradiction (same polarity)
- "I don't like X" vs "I don't like Y" → not contradiction (both negative)

---

## 16. Decision Trace Auditing

### 16.1 Purpose

Full lifecycle tracking: what was retrieved → what was used → what outcome resulted.

### 16.2 Data Structure

```rust
pub struct DecisionTrace {
    pub query_id: String,
    pub agent_id: AgentId,
    pub session_id: SessionId,
    pub retrieved_memory_ids: Vec<MemoryId>,
    pub retrieved_strategy_ids: Vec<StrategyId>,
    pub retrieved_claim_ids: Vec<ClaimId>,
    pub used_memory_ids: Vec<MemoryId>,
    pub used_strategy_ids: Vec<StrategyId>,
    pub used_claim_ids: Vec<ClaimId>,
    pub outcome: Option<OutcomeSignal>,
    pub policy_version: String,
    pub started_at: Timestamp,
    pub closed_at: Option<Timestamp>,
}
```

### 16.3 Lifecycle

1. **Start**: Agent retrieves memories/strategies/claims → log retrieved IDs
2. **Mark used**: Agent applies a memory/strategy/claim → add to used list
3. **Close**: Episode completes → log outcome signal with success/failure
4. **Claim feedback**: For each used claim, record outcome → update Bayesian score → optionally generate avoidance claim (see Section 19)

### 16.4 Storage

- Main table: `decision_trace` — `query_id → DecisionTrace`
- Agent index: `outcome_signals` — `(agent_id, timestamp, query_id) → ()` for range queries

---

## 17. Storage Architecture

### 17.1 Hot/Cold LRU Cache

```
┌──────────────┐     Cache Miss     ┌─────────────────┐
│  LRU Cache   │ ─────────────────► │   redb Tables    │
│  (In-Memory) │ ◄───────────────── │  (On-Disk B+Tree)│
│  HashMap     │     Load + Cache   │                  │
└──────────────┘                    └─────────────────┘
```

- Cache bounded by `max_cache_size`
- Eviction: remove entry with oldest `last_accessed` timestamp
- On cache miss: load from redb, insert into cache, evict LRU if full

### 17.2 Redb Index Tables (Strategy)

| Table | Key Format | Value | Purpose |
|---|---|---|---|
| `strategy_records` | `strategy_id (8B BE)` | `Strategy (bincode)` | Primary storage |
| `strategy_by_bucket` | `[goal_bucket_id (8B) \| strategy_id (8B)]` | `()` | Goal bucket index |
| `strategy_by_signature` | `[signature_hash (8B) \| strategy_id (8B)]` | `()` | Dedup index |
| `strategy_feature_postings` | `[agent_id (8B) \| strategy_id (8B)]` | `quality_score: f32` | Agent index |

### 17.3 Redb Index Tables (Memory)

| Table | Key Format | Value | Purpose |
|---|---|---|---|
| `memory_records` | `memory_id (8B BE)` | `Memory (bincode)` | Primary storage |
| `memory_by_context` | `[context_hash (8B) \| memory_id (8B)]` | `()` | Context index |
| `memory_by_agent` | `[agent_id (8B) \| memory_id (8B)]` | `()` | Agent index |
| `memory_by_goal_bucket` | `[goal_bucket_id (8B) \| memory_id (8B)]` | `()` | Goal bucket index |

### 17.4 Composite Key Encoding

All multi-part keys use big-endian byte encoding with `0xFF` separators where needed:

```
agent_index_key = agent_id.to_be_bytes() ++ timestamp.to_be_bytes() ++ 0xFF ++ query_id.as_bytes()
```

---

## 18. Background Maintenance Engine

### 18.1 Configuration

```rust
pub struct MaintenanceConfig {
    pub interval_secs: u64,              // default: 300 (5 minutes)
    pub strategy_min_confidence: f32,    // default: 0.15
    pub strategy_min_support: u32,       // default: 1
    pub strategy_max_stale_hours: f32,   // default: 72.0 (3 days)
    pub claim_dedup_threshold: f32,      // default: 0.92
    pub claim_contradiction_threshold: f32, // default: 0.85
    pub max_vector_index_size: usize,    // default: 50_000
    pub purge_inactive_claims: bool,     // default: true
}
```

### 18.2 Maintenance Pass

Each pass performs:
1. **Memory decay** — age out stale memories below forget threshold
2. **Strategy pruning** — remove strategies below `min_confidence` AND below `min_support`, or stale beyond `max_stale_hours`
3. **Strategy merging** — merge near-duplicate strategies by behavior signature
4. **Graph node cap** — enforce per-bucket node limits (oldest Event nodes evicted first)
5. **Transition cleanup** — `cleanup_oldest_episodes(keep)` + `prune_weak_transitions(min_count)`

### 18.3 Result Tracking

```rust
pub struct MaintenanceResult {
    pub memories_decayed: bool,
    pub strategies_pruned: usize,
    pub graph_nodes_merged: usize,
    pub graph_nodes_deleted: usize,
    pub graph_headers_scanned: usize,
    pub graph_pruning_stopped_early: bool,
    pub transition_episodes_cleaned: usize,
    pub transition_entries_pruned_pass: bool,
}
```

---

## 19. Claim Outcome Feedback Loop

### 19.1 Purpose

Claims are extracted from events and stored with embeddings, BM25 indexes, and HNSW — but without feedback, they are static knowledge. The claim outcome feedback loop closes this gap: when an agent retrieves a claim and uses it in a decision, the outcome (success/failure) feeds back to the claim. Claims that consistently help agents get boosted in retrieval. Claims that consistently hurt get demoted. Repeated failures auto-generate "avoidance" claims so agents learn what NOT to do.

### 19.2 Outcome Storage (Zero Migration)

Outcome counts are stored in the **existing `metadata: HashMap<String, String>`** on `DerivedClaim`, avoiding any bincode serialization breakage:

```rust
pub const META_POSITIVE_OUTCOMES: &str = "_positive_outcomes";
pub const META_NEGATIVE_OUTCOMES: &str = "_negative_outcomes";
```

Methods on `DerivedClaim`:
- `positive_outcomes() → u32` — parse from metadata, default 0
- `negative_outcomes() → u32` — parse from metadata, default 0
- `total_outcomes() → u32` — sum of positive + negative
- `record_outcome(success: bool)` — increment the appropriate counter
- `outcome_score() → f32` — Bayesian-smoothed score (see §19.3)

### 19.3 Piecewise Outcome Score (Bayesian → Q-Value)

The outcome score uses a **piecewise function** that selects the scoring regime based on evidence quantity:

```
if total_outcomes < Q_KICK_IN (5):
    outcome_score = (positive + 1) / (total + 2)        // Bayesian (Laplace)
else:
    outcome_score = Q_value                               // EMA Q-value
```

**Phase 1 — Bayesian (< 5 outcomes):** Beta-Binomial posterior with uniform prior. Stable, prior-dominated, resistant to noise from small samples. A single failure drops the score to 0.33, not 0.0.

**Phase 2 — Q-value (≥ 5 outcomes):** Exponential moving average updated on every outcome:

```
Q_new = Q_old + α × (r − Q_old)       where α = 0.3, r ∈ {0.0, 1.0}
```

The Q-value accumulates from outcome #1 but is not *read* until the threshold, giving the EMA time to warm up. Initial Q = 0.5 (neutral).

**Why piecewise?** Each regime is optimal for its evidence range:

| Property | Bayesian (< 5 outcomes) | Q-value (≥ 5 outcomes) |
|---|---|---|
| Noise robustness | Excellent (prior absorbs outliers) | Moderate (α smooths but doesn't prevent swings) |
| Adaptation to shift | Slow (all history weighted equally) | Fast (recent outcomes dominate exponentially) |
| Appropriate when | Small sample — stability matters more than speed | Enough evidence — responsiveness matters more than caution |

**Score evolution under distribution shift** (10 successes then 5 consecutive failures):

| After | Bayesian (if used) | Q-value (actual) |
|---|---|---|
| 10 successes | 0.917 | ~0.97 |
| +1 failure | 0.846 | ~0.68 |
| +3 failures | 0.733 | ~0.33 |
| +5 failures | 0.647 | ~0.17 |

The Q-value drops to 0.17 after 5 failures — a 4x faster response than Bayesian (0.647). This means the system adapts within 3-5 outcomes when the environment changes, while still being conservative during the initial evidence-gathering phase.

**Data stored per claim** (all in existing `metadata: HashMap<String, String>`, zero migration):

| Key | Type | Purpose |
|---|---|---|
| `_positive_outcomes` | u32 (as string) | Lifetime positive count (lossless, auditable) |
| `_negative_outcomes` | u32 (as string) | Lifetime negative count (lossless, auditable) |
| `_q_value` | f32 (as string) | EMA Q-value (responsive scoring) |

The counters provide full auditability. The Q-value provides responsiveness. Both are maintained on every outcome; the piecewise function chooses which to read.

### 19.4 Outcome-Aware Retrieval Scoring

The claim retrieval score incorporates outcome history via a multiplicative factor:

```
base = similarity × (0.6 + 0.4 × temporal_weight)
multiplier = 0.6 + 0.8 × outcome_score        // maps [0..1] → [0.6..1.4]
score = base × multiplier
```

**Why multiplicative, not additive?** Additive scoring (as in MemRL: `(1−λ)·sim + λ·Q`) allows a high-Q but irrelevant memory to outrank a relevant but unproven one. Multiplicative scoring ensures that **relevance (similarity) is always a hard gate**: a claim with similarity = 0.1 can at most score `0.1 × 1.4 = 0.14`, regardless of how successful it was in other contexts. This is correct for knowledge retrieval — facts about REST APIs should never appear in a query about DOM manipulation, no matter how useful they were elsewhere.

The multiplier maps the piecewise `outcome_score` (§19.3) to a retrieval factor:

| Regime | Outcome History | outcome_score | multiplier | Effect |
|---|---|---|---|---|
| Bayesian | No outcomes | 0.500 | **1.000** | Neutral (backward-compatible) |
| Bayesian | 1 negative, 0 positive | 0.333 | **0.867** | 13% penalty (mild) |
| Bayesian | 3 positive | 0.800 | **1.240** | 24% boost |
| Q-value | All positive (converged) | →1.000 | **→1.400** | 40% boost |
| Q-value | All negative (converged) | →0.000 | **→0.600** | 40% penalty |
| Q-value | 10 success then 5 failure | ~0.170 | **~0.736** | 26% penalty (fast adaptation) |

The neutral case (`multiplier = 1.0`) ensures all existing tests and retrieval behavior are unchanged for claims without outcome data.

### 19.5 Learning Telemetry Events

Two new `LearningEvent` variants track claim usage in the decision trace pipeline:

```rust
LearningEvent::ClaimRetrieved {
    query_id: String,
    claim_ids: Vec<u64>,
}

LearningEvent::ClaimUsed {
    query_id: String,
    claim_id: u64,
}
```

These follow the same pattern as the existing `MemoryRetrieved`/`MemoryUsed` and `StrategyServed`/`StrategyUsed` variants. The `DecisionTrace` accumulates claim IDs alongside memory and strategy IDs.

### 19.6 Outcome Application Pipeline

When a `LearningEvent::Outcome { query_id, success }` arrives:

```
1. Remove DecisionTrace for query_id (DashMap lock-free)
2. For each used memory   → store.apply_outcome(memory_id, success)
3. For each used strategy → store.update_strategy_outcome(strategy_id, success)
4. For each used claim    → claim_store.record_outcome(claim_id, success)
      │
      └─► If !success AND updated.negative_outcomes() >= 2:
            Generate avoidance claim (see §19.7)
```

`ClaimStore::record_outcome()` follows the same read-modify-write-commit pattern as `add_support` and `mark_accessed`: begin write txn → read claim → call `claim.record_outcome(success)` → serialize → insert → commit. Returns the updated claim for downstream avoidance generation.

### 19.7 Automatic Avoidance Claim Generation

When a claim accumulates **≥ 2 negative outcomes** from a failure event, the system automatically generates a new `ClaimType::Avoidance` claim:

```
avoidance_text = "Avoid relying on: {source_claim_text} — this has led to repeated failures"
```

The avoidance claim:
- Inherits the source claim's **embedding** (so it appears in the same semantic neighborhood during retrieval)
- Inherits the source claim's **entities** (for entity-based search continuity)
- Has `ClaimType::Avoidance` with a 14-day half-life (same as Belief — mistakes should fade but not quickly)
- Has `confidence = 0.5` (moderate — it is auto-generated, not human-validated)
- Gets its own unique `ClaimId` from the store's ID counter

**Effect on retrieval:** When an agent queries the same semantic space, the avoidance claim appears alongside the original (demoted) claim. The agent sees both "X" (with a low retrieval score) and "Avoid relying on X" (fresh, with neutral outcome score), steering it toward alternative approaches.

### 19.8 Convergence Properties

The feedback loop has several desirable convergence properties:

1. **Bayesian smoothing prevents oscillation**: A single bad outcome doesn't destroy a claim's ranking. With the `(pos+1)/(total+2)` prior, the system requires sustained evidence to make large ranking changes.

2. **Avoidance generation is bounded**: Avoidance claims are only generated on failure events (not retroactively), and only when `negative_outcomes >= 2`. A claim that fails once gets a mild penalty; only repeated failures trigger avoidance.

3. **Temporal decay provides forgetting**: Both the original claim and its avoidance claim decay via `temporal_weight()`. If the failure pattern becomes irrelevant over time, both the demoted claim and its avoidance warning naturally fade from retrieval results.

4. **Positive reinforcement compounds**: Successful claims get boosted retrieval scores, making them more likely to be retrieved and used again, generating more positive outcomes — a virtuous cycle bounded by the `multiplier ≤ 1.4` ceiling.

---

## 20. Hierarchical Predictive Coding Engine {#20-hierarchical-predictive-coding-engine}

### 20.1 Motivation & Neuroscience Foundation

Sections 4–5 describe EventGraphDB's three-tier memory hierarchy (Episodic → Semantic → Schema) with **bottom-up** consolidation: raw events form episodes, episodes generalize into semantic patterns, and semantic patterns distill into abstract schemas. This is half of the brain's memory architecture.

In biological neural systems, **each cortical layer also generates top-down predictions** about the layer below it. Learning is driven not by the data itself but by **prediction errors** — the mismatch between what a higher layer expected and what it actually observed. This is the **predictive coding** framework (Rao & Ballard, 1999; Friston, 2005).

Key neuroscience findings that motivate this design:

- **Small prediction errors** → edit/reinforce existing memories (reconsolidation)
- **Large prediction errors** → create entirely new episodic memories (surprise-driven encoding)
- **Precision weighting** → reliable signals drive larger updates than noisy ones
- **~80% of cortical computation** is top-down prediction, not bottom-up processing
- Hippocampal replay generates **fictive prediction errors** that train the neocortex offline

### 20.2 Architecture Overview

```
┌─────────────────────────────────────────────────────────────┐
│                PREDICTIVE CODING ENGINE                      │
│            (Python / GPU co-processor service)               │
│                                                              │
│  ┌──────────────────────────────────────────────────┐       │
│  │ Schema Layer (L2) — Abstract generative model     │       │
│  │   predicts ──▶ expected Semantic patterns         │       │
│  │   ◀── PE from Semantic layer (bottom-up)          │       │
│  │   Model: Lightweight Transformer encoder          │       │
│  └────────────────────┬─────────────────────────────┘       │
│                       │ top-down predictions                 │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────┐       │
│  │ Semantic Layer (L1) — Pattern generative model    │       │
│  │   predicts ──▶ expected Episodic outcomes         │       │
│  │   ◀── PE from Episodic layer (bottom-up)          │       │
│  │   Model: MLP with attention over context          │       │
│  └────────────────────┬─────────────────────────────┘       │
│                       │ top-down predictions                 │
│                       ▼                                      │
│  ┌──────────────────────────────────────────────────┐       │
│  │ Episodic Layer (L0) — Ground truth from events    │       │
│  │   No prediction (leaf layer)                      │       │
│  │   Emits PE = |actual - predicted_by_L1|           │       │
│  └──────────────────────────────────────────────────┘       │
│                                                              │
│  ┌──────────────────────────────────────────────────┐       │
│  │ gRPC / IPC Interface ◀──▶ Rust core (redb)       │       │
│  └──────────────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────────────┘
```

The predictive coding engine runs as a **GPU co-processor** alongside the Rust core. The Rust side owns storage, retrieval, and the event pipeline. The Python/GPU side owns the generative models that produce top-down predictions and compute prediction errors.

### 20.3 Inter-Process Architecture (Rust ↔ Python)

The Rust core and Python GPU process communicate via a lightweight IPC channel:

```
┌──────────────────────┐         ┌────────────────────────┐
│  Rust Core (redb)    │  gRPC   │  Python GPU Service     │
│                      │◀───────▶│                        │
│  • Event ingestion   │         │  • Predictive models   │
│  • Claim storage     │         │  • PE computation      │
│  • HNSW index        │         │  • Precision tracking  │
│  • Retrieval scoring │         │  • Model updates       │
│  • Consolidation     │         │  • Replay generation   │
│    decisions         │         │                        │
└──────────────────────┘         └────────────────────────┘

Protocol:
  Rust → Python:  PredictionRequest { layer, claim_embedding, context, metadata }
  Python → Rust:  PredictionResponse { prediction_error, precision, action }

Actions:
  CONSOLIDATE  — PE is low, higher layer already understands this
  RETAIN       — PE is medium, keep as-is, update model
  PROMOTE      — PE is high, this is genuinely novel, promote to next tier
  PRUNE        — precision is high AND PE is near-zero, redundant memory
```

**Why Python + GPU instead of pure Rust?**

The generative models at each layer require neural network inference (matrix multiplications, attention, gradient updates). While Rust ML frameworks exist (burn, candle, tch-rs), the Python ecosystem provides:

- **PyTorch** with mature GPU kernels (CUDA/ROCm/Metal)
- **Pre-trained model zoo** — CLIP, sentence-transformers, small LLMs for schema generation
- **Fast iteration** — model architecture changes without recompiling the Rust core
- **Existing tooling** — TensorBoard, Weights & Biases for monitoring prediction error convergence

The Rust core remains the system of record. The Python service is stateless (can be restarted without data loss) and advisory (the Rust side makes final consolidation decisions based on PE signals).

### 20.4 Layer-by-Layer Prediction Models

#### 20.4.1 Semantic → Episodic Prediction (L1 → L0)

**What it predicts:** Given a semantic claim (generalized pattern), predict the expected outcome of the next episodic instance matching that pattern.

**Model architecture:**

```
Input:
  semantic_claim_embedding    (768-dim, from claim store)
  context_embedding           (768-dim, from current episode context)
  temporal_features           (8-dim: time_of_day, day_of_week, recency, etc.)
  historical_outcome_stats    (4-dim: success_rate, variance, trend, n_outcomes)

Model:
  concat(all inputs) → Linear(1548, 512) → GELU → Linear(512, 256) → GELU
  → Prediction head: Linear(256, 1) → sigmoid  (predicted success probability)
  → Precision head:  Linear(256, 1) → softplus (predicted precision/confidence)

Output:
  predicted_success: f32       (0.0 to 1.0)
  predicted_precision: f32     (> 0, higher = more confident)
```

**Prediction error computation:**

```
PE_episodic = |actual_outcome - predicted_success|
weighted_PE = PE_episodic × predicted_precision

IF weighted_PE < τ_low (0.15):
    → CONSOLIDATE: this episode is well-predicted, safe to merge into semantic
ELIF weighted_PE > τ_high (0.70):
    → PROMOTE: this is highly surprising, create/update semantic pattern
ELSE:
    → RETAIN: keep as episodic, update L1 model
```

**Training:** Online SGD after each episode outcome. Loss = MSE(predicted_success, actual_outcome) + calibration_loss(predicted_precision, observed_PE_variance).

#### 20.4.2 Schema → Semantic Prediction (L2 → L1)

**What it predicts:** Given an abstract schema (rule/principle), predict the distribution of semantic patterns that should exist beneath it.

**Model architecture:**

```
Input:
  schema_embedding           (768-dim)
  domain_context             (768-dim, e.g., "ATS applications" or "API integration")

Model:
  Transformer encoder (2 layers, 4 heads, d_model=256)
  Input: [schema_token, context_token, K learnable query tokens]

  Output per query token:
    → predicted_claim_embedding (768-dim, via linear projection)
    → predicted_confidence_range (2-dim: min, max)
    → predicted_success_rate (1-dim)
    → predicted_count (1-dim: how many semantic claims expected)

Output:
  K predicted semantic claim prototypes
  aggregate statistics (expected success rate, expected count, confidence bounds)
```

**Prediction error computation:**

```
actual_semantic_claims = claim_store.query_by_parent_schema(schema_id)

// Distributional PE: compare predicted prototypes to actual claims
PE_schema = mean(min_distance(predicted_prototypes, actual_claim_embeddings))
         + |predicted_count - actual_count| / max(predicted_count, 1)
         + |predicted_success_rate - actual_mean_success_rate|

IF PE_schema < τ_low (0.20):
    → Schema is accurate, semantic claims beneath it are redundant
    → Consider PRUNING semantic claims that are fully captured by schema
ELIF PE_schema > τ_high (0.60):
    → Schema is stale or too simplistic
    → Trigger schema REFINEMENT (split, update, or create sub-schemas)
ELSE:
    → Schema partially accurate, update L2 model
```

**Training:** Batch update during maintenance windows (§18). Collects all schema→semantic pairs, trains for N epochs with cosine-annealing LR schedule.

### 20.5 Precision-Weighted Learning

Each claim accumulates a **precision** estimate — the inverse variance of its prediction errors over time:

```
Metadata keys (stored in existing HashMap<String, String>):
  _pe_running_mean     — exponential moving average of PE
  _pe_running_var      — exponential moving average of PE²  (for variance)
  _precision           — 1.0 / max(pe_running_var, ε)

Update rule (after each outcome):
  pe_mean_new = (1 - β) × pe_mean_old + β × current_PE
  pe_var_new  = (1 - β) × pe_var_old  + β × (current_PE - pe_mean_new)²
  precision   = 1.0 / max(pe_var_new, 0.01)

  where β = 0.2 (precision EMA rate)
```

**Effect on retrieval scoring** (extends §19.4):

```
retrieval_score = base × outcome_multiplier × precision_multiplier

where:
  base               = similarity × (0.6 + 0.4 × temporal_weight)
  outcome_multiplier = 0.6 + 0.8 × outcome_score           // [0.6, 1.4]
  precision_multiplier = 0.7 + 0.3 × min(precision / 10, 1) // [0.7, 1.0]
```

**Effect on Q-value learning rate** (extends §19.3):

```
adaptive_alpha = Q_ALPHA × min(precision / 5.0, 2.0)    // [0, 0.6]

// High precision (reliable claim) → learns faster (up to 2× Q_ALPHA)
// Low precision (noisy claim) → learns slower (approaches 0)
```

### 20.6 PE-Gated Consolidation

The existing consolidation engine (§5) uses time-based thresholds to promote memories up the hierarchy. Predictive coding replaces this with **PE-gated consolidation**:

```
ALGORITHM: PE-Gated Consolidation Pass

FOR each episodic claim E:
    1. Query L1 model: predicted_outcome = L1.predict(E.context, E.parent_semantic)
    2. Compute PE = |E.actual_outcome - predicted_outcome|
    3. Weight by precision: weighted_PE = PE × L1.precision(E.context)

    IF weighted_PE < τ_consolidate (0.15):
        // L1 already understands this experience
        IF E.age > min_age_for_consolidation:
            Merge E into parent semantic claim (update stats, don't create new)
            Mark E for pruning

    ELIF weighted_PE > τ_surprise (0.70):
        // This is genuinely novel — L1 can't predict it
        IF similar_high_PE_episodes_count(E) >= 3:
            // Pattern of surprises → create new semantic claim
            Promote E to new Semantic claim
            Retrain L1 to accommodate new pattern
        ELSE:
            // Isolated surprise — keep as episodic, wait for more evidence
            Tag E as "high_PE_anomaly"
            Retain in episodic tier

    ELSE:
        // Medium PE — L1 partially understands
        Update L1 model with this example (online SGD)
        Retain E in episodic tier

FOR each semantic claim S:
    1. Query L2 model: predicted_distribution = L2.predict(S.parent_schema)
    2. Compute PE_schema (distributional mismatch, see §20.4.2)

    IF PE_schema > τ_schema_drift (0.60):
        // Schema is stale — trigger refinement
        IF S represents a consistent sub-pattern not in schema:
            Propose schema split or sub-schema creation
        ELSE:
            Flag schema for re-distillation (§7)
```

### 20.7 Predictive Replay (Offline Learning)

During maintenance windows (§18), the predictive coding engine performs **predictive replay** — analogous to hippocampal replay during sleep:

```
ALGORITHM: Predictive Replay

1. Sample N claims from the store, prioritized by:
   - High PE (most informative for model training)
   - Recent access (likely relevant to current tasks)
   - Low precision (model is uncertain, needs more training)

2. For each sampled claim:
   a. Generate L1 prediction (semantic → episodic)
   b. Compute current PE against stored outcome
   c. Compare to previous PE for this claim

   IF current_PE < previous_PE × 0.5:
       // Model has learned this — reduce replay priority
       claim.replay_priority *= 0.8
   ELIF current_PE > previous_PE:
       // Model is getting WORSE on this claim — flag anomaly
       claim.anomaly_flag = true
       claim.replay_priority *= 1.5

   d. Update model weights via gradient step (SGD/Adam)
   e. Store updated PE in claim metadata

3. Aggregate statistics:
   - mean_PE across all layers (should decrease over time)
   - precision histogram (should shift rightward over time)
   - anomaly_count (should remain low)

4. Emit ReplayComplete event with statistics for monitoring
```

### 20.8 Active Information Seeking

When prediction error is high for a particular context, the agent has a **quantified measure of its own ignorance**. This enables active inference — the agent can act to reduce uncertainty rather than just maximize reward:

```
ALGORITHM: Uncertainty-Driven Claim Retrieval

Input: query context C, top-K candidate claims

1. Standard retrieval: rank claims by retrieval_score (§19.4 + §20.5)
2. For each candidate claim, compute expected PE:
   expected_PE = L1.predict_uncertainty(claim, C)

3. Compute information gain for each claim:
   info_gain = expected_PE × (1 - claim.precision)
   // High expected PE + low precision = maximum learning opportunity

4. Final score blends exploitation and exploration:
   score = (1 - ε) × retrieval_score + ε × info_gain

   where ε adapts based on overall system confidence:
   ε = max(0.05, 0.3 × (1 - mean_precision_across_claims))
   // High overall precision → low ε (exploit)
   // Low overall precision  → high ε (explore)

5. Return top-K by final score
```

This replaces the epsilon-greedy proposal in Appendix C.5 with a principled, precision-aware exploration mechanism derived from the free energy principle.

### 20.9 Neural Network Specifications

#### 20.9.1 L1 Model (Semantic → Episodic)

| Property | Value |
|---|---|
| Architecture | 3-layer MLP with residual connections |
| Input dimension | 1548 (768 + 768 + 8 + 4) |
| Hidden dimensions | 512, 256 |
| Output | success_probability (1), precision (1) |
| Parameters | ~1.1M |
| Activation | GELU |
| Training | Online SGD, lr=1e-3 with cosine decay |
| Inference latency | < 1ms on GPU, < 5ms on CPU |
| Framework | PyTorch (CUDA/ROCm/Metal) |

#### 20.9.2 L2 Model (Schema → Semantic)

| Property | Value |
|---|---|
| Architecture | 2-layer Transformer encoder + linear heads |
| Input dimension | 768 × 2 (schema + context) + K query tokens |
| d_model | 256 |
| Heads | 4 |
| Query tokens (K) | 8 (predicted semantic prototypes) |
| Parameters | ~2.4M |
| Training | Batch, during maintenance, Adam lr=5e-4 |
| Inference latency | < 5ms on GPU, < 20ms on CPU |
| Framework | PyTorch (CUDA/ROCm/Metal) |

#### 20.9.3 Deployment Options

| Configuration | L1 | L2 | Use Case |
|---|---|---|---|
| GPU co-processor | CUDA inference | CUDA inference + training | Production, high-throughput agents |
| CPU-only | ONNX Runtime | ONNX Runtime | Edge deployment, single-agent |
| Cloud offload | API call to inference service | API call | Multi-agent fleet, shared models |
| Disabled | Fall back to Q-value only (§19) | Fall back to time-based consolidation (§5) | Minimal deployment, no GPU available |

The system degrades gracefully: without the Python GPU service, the Rust core continues to operate using the existing Q-value / Bayesian outcome tracking (§19). The predictive coding engine is an **enhancement layer**, not a hard dependency.

### 20.10 Training Data Pipeline

```
Events (Rust core)
    │
    ▼
LearningEvent::Outcome { success, reward }
    │
    ├──▶ Rust: update Q-value, counters (existing §19 path)
    │
    └──▶ Python: training example for L1/L2
         │
         Format:  TrainingExample {
             claim_id: u64,
             claim_embedding: [f32; 768],
             context_embedding: [f32; 768],
             temporal_features: [f32; 8],
             outcome: f32,          // 0.0 or 1.0
             reward: f32,           // continuous signal if available
             tier: MemoryTier,      // which layer produced this claim
             parent_id: Option<u64> // parent claim in hierarchy
         }
         │
         ▼
    Training buffer (ring buffer, 10K examples)
         │
         ├──▶ Online update: L1 after each example
         └──▶ Batch update: L2 every maintenance window
```

### 20.11 Convergence Properties

1. **PE decreases over time**: As models learn, predictions improve, PE shrinks. This is measurable and serves as the primary health metric for the predictive coding engine.

2. **Precision increases over time**: Reliable claims accumulate high precision, concentrating retrieval on trustworthy knowledge.

3. **Consolidation becomes selective**: Early in learning, most experiences are surprising (high PE) and retained as episodic. As models improve, routine experiences are consolidated efficiently while only genuine novelty stays in the episodic tier.

4. **Graceful cold-start**: With no training data, L1/L2 models output uniform predictions (PE ≈ 0.5 for all claims). This maps to the same behavior as the Bayesian prior in §19 — neutral, neither boosting nor penalizing. As data accumulates past Q_KICK_IN (5 outcomes), the models begin differentiating.

5. **Catastrophic forgetting protection**: The neural models are small (< 5M parameters combined) and trained on a sliding window of recent examples. If models degrade, the Rust-side Q-values and Bayesian counters remain the authoritative signal. The predictive coding engine can be retrained from scratch using historical outcome data stored in redb without losing any information.

### 20.12 Prior Art Differentiation

| System | Memory Hierarchy | Top-Down Prediction | PE-Gated Consolidation | Precision Weighting |
|---|---|---|---|---|
| **EventGraphDB (this work)** | 3-tier (Episodic/Semantic/Schema) | Neural generative model per layer | Yes — PE thresholds gate promotion/pruning | Per-claim adaptive precision via PE variance |
| **MemRL (2024)** | Flat episodic buffer | None | None — all memories equal | None — fixed EMA rate |
| **ERL (2024)** | 2-tier (Experience/Reflection) | None | Time-based | None |
| **VERSES AXIOM (2025)** | Hierarchical generative model | Yes (active inference) | Implicit via free energy | Yes (precision on sensory channels) |
| **Predictive Coding Networks** | Per-layer latent states | Yes (core mechanism) | N/A (continuous signals) | Yes (precision weighting) |

**Key novelty**: EventGraphDB is the first system to apply hierarchical predictive coding to a **persistent, non-parametric knowledge store** (claims in a database) rather than to transient neural activations. The combination of PE-gated consolidation, precision-weighted retrieval, and graceful degradation to non-neural scoring (§19) is novel.

---

## Appendix A: Fully Populated Strategy Example (Before & After LLM Distillation) {#appendix-a}

### A.1 Stage 1 Output (Template Synthesis Only)

```json
{
  "id": 42,
  "name": "strategy_1001_ep_7",
  "strategy_type": "Positive",

  "summary": "DO this when applicable. When: deploy payments-service to production; ensure service health after deployment. Context: intent=deploy, env=production, region=us-east-1. Steps: 1. Receive [user_request]: Deploy payments-service v2.3.1 to production → 2. Think (Planning) → 3. validate_config -> {\"valid\": true} → 4. run_tests [partial] → 5. Observe [test_results]: {\"passed\": 47, \"failed\": 1} → 6. Think (Evaluation) → 7. deploy_service -> {\"status\": \"deployed\"} → 8. health_check [FAILED: connection timeout after 30s]. Success looks like: Action 'deploy_service' succeeded; Action 'validate_config' succeeded. Avoid: Action 'health_check' failed: connection timeout after 30s. This episode succeeded (8 events, significance 78%).",

  "when_to_use": "Use when the goal is: deploy payments-service to production; ensure service health after deployment. Context matches: env=production, region=us-east-1. Best for episodes with significance ≥ 78%",

  "when_not_to_use": "Some actions failed during this episode; the strategy is fragile under error conditions. Do not use when the context significantly differs from the original goals",

  "failure_modes": [
    "Action 'health_check' can fail: connection timeout after 30s",
    "Action 'run_tests' partially succeeds with issues: [\"flaky test: test_auth_timeout\"]"
  ],

  "counterfactual": "Despite failures at 'health_check' (connection timeout after 30s), the episode recovered. Skipping those steps could have been faster.",

  "summary_embedding": [],

  "playbook": [
    {"step": 1, "action": "Receive [user_request]: Deploy payments-service v2.3.1 to production", "condition": "", "skip_if": "No input available", "branches": [], "recovery": "", "step_id": "", "next_step_id": null},
    {"step": 2, "action": "Think (Planning)", "condition": "", "skip_if": "", "branches": [], "recovery": "", "step_id": "", "next_step_id": null},
    {"step": 3, "action": "Execute 'validate_config'", "condition": "", "skip_if": "", "branches": [], "recovery": "", "step_id": "", "next_step_id": null},
    {"step": 4, "action": "Execute 'run_tests'", "condition": "", "skip_if": "", "branches": [], "recovery": "On partial success: address [\"flaky test: test_auth_timeout\"]", "step_id": "", "next_step_id": null},
    {"step": 5, "action": "Observe [test_results]", "condition": "", "skip_if": "", "branches": [], "recovery": "", "step_id": "", "next_step_id": null},
    {"step": 6, "action": "Think (Evaluation)", "condition": "", "skip_if": "", "branches": [], "recovery": "", "step_id": "", "next_step_id": null},
    {"step": 7, "action": "Execute 'deploy_service'", "condition": "", "skip_if": "", "branches": [], "recovery": "", "step_id": "", "next_step_id": null},
    {"step": 8, "action": "Execute 'health_check'", "condition": "", "skip_if": "", "branches": [], "recovery": "On failure (connection timeout after 30s): retry or use alternative approach", "step_id": "", "next_step_id": null}
  ],

  "reasoning_steps": [
    {"description": "[VERIFY] Check config file at <file_path> for required fields", "applicability": "general", "expected_outcome": "validation_result", "sequence_order": 0},
    {"description": "[IF-THEN] If tests pass then proceed to deploy, otherwise abort", "applicability": "general", "expected_outcome": null, "sequence_order": 1},
    {"description": "Execute deploy_service with rollback flag enabled", "applicability": "contextual", "expected_outcome": null, "sequence_order": 2},
    {"description": "[VERIFY] Validate health check returns 200 within timeout", "applicability": "general", "expected_outcome": "validation_result", "sequence_order": 3}
  ],

  "context_patterns": [
    {"environment_type": "general", "task_type": "deployment", "resource_constraints": [], "goal_characteristics": ["deploy payments-service to production"], "match_confidence": 0.85}
  ],

  "success_indicators": ["Action 'deploy_service' succeeded", "Action 'validate_config' succeeded"],
  "failure_patterns": ["Action 'health_check' failed: connection timeout after 30s"],

  "quality_score": 0.82,
  "success_count": 14,
  "failure_count": 3,
  "support_count": 17,
  "expected_success": 0.789,
  "expected_cost": 8.0,
  "expected_value": 0.789,
  "confidence": 0.9965,

  "precondition": "goal_bucket=14298376502 contexts=6",
  "action_hint": "repeat sequence: Receive > Think:Planning > Act:validate_config > Act:run_tests > Observe > Think:Evaluation > Act:deploy_service > Act:health_check",

  "goal_bucket_id": 14298376502,
  "behavior_signature": "a3f7c2e1d904b6",

  "agent_id": 1001,
  "created_at": 1739500000,
  "last_used": 1739650000,

  "metadata": {
    "goal_ids": "[42, 99]",
    "tool_names": "[\"validate_config\", \"run_tests\", \"deploy_service\", \"health_check\"]",
    "result_types": "[\"deployment_success\", \"test_report\"]",
    "goal_bucket_id": "14298376502",
    "behavior_signature": "a3f7c2e1d904b6",
    "strategy_signature": "e8b2f1a09c3d"
  },

  "supersedes": [],
  "applicable_domains": ["deploy payments-service to", "ensure service health after"],
  "lineage_depth": 0,
  "contradictions": [],

  "self_judged_quality": 0.85,
  "source_outcomes": ["Success", "Success", "Failure", "Success", "Success"],
  "version": 1,
  "parent_strategy": null
}
```

### A.2 Stage 2 Output (After LLM Distillation + Embedding)

The following 5 fields are **overwritten** by the LLM. All other fields remain identical.

```json
{
  "summary": "A production deployment strategy that validates config, runs integration tests, deploys the service, and performs health checks. It recovers from transient health check failures but is fragile when the test suite has flaky tests.",

  "when_to_use": "Use when deploying a microservice to production where validate_config, run_tests, deploy_service, and health_check tools are available and the target has health check endpoints.",

  "when_not_to_use": "Avoid when deploying to environments without health check endpoints or when the integration test suite has known instability that would block the pipeline.",

  "failure_modes": [
    "health_check times out when the target host is slow to start (30s default timeout)",
    "run_tests produces flaky results from test_auth_timeout, causing false partial outcomes",
    "deploy_service may succeed but health_check fails if load balancer hasn't registered the new instance"
  ],

  "counterfactual": "Adding a retry with exponential backoff to health_check would have avoided the transient timeout failure and completed the deployment without manual intervention.",

  "summary_embedding": [0.0231, -0.0142, 0.0387, 0.0019, -0.0256, "... (1536 dimensions)"]
}
```

### A.3 Example Constraint Strategy

```json
{
  "id": 43,
  "name": "constraint_1001_ep_9",
  "strategy_type": "Constraint",

  "summary": "Avoid deploying without running integration tests first. Skipping tests correlated with 80% failure rate in this goal bucket.",

  "when_to_use": "Watch out when the goal is: deploy payments-service to production. Applies when the agent is about to repeat a pattern that previously failed",

  "when_not_to_use": "Safe to ignore when a newer strategy explicitly supersedes this constraint, or when the failure pattern is no longer applicable to the current context.",

  "playbook": [
    {
      "step": 1,
      "action": "Execute 'deploy_service'",
      "branches": [
        {"condition": "If this pattern appears", "action": "Skip 'deploy_service' and use alternative", "next_step_id": null}
      ],
      "recovery": ""
    }
  ],

  "action_hint": "avoid sequence: Act:deploy_service",
  "expected_success": 0.40,
  "expected_value": -0.40,
  "quality_score": 0.20,
  "success_count": 1,
  "failure_count": 4,
  "support_count": 5,
  "confidence": 0.811
}
```

---

## Appendix B: Embedding Gap Analysis {#appendix-b}

### B.1 Current State

| Component | Embedding Used for Retrieval? | Embedding Populated? |
|---|---|---|
| **Memory** retrieval | **Yes** — cosine similarity in `retrieve_hierarchical()` with tier boost | Yes — via `RefinementEngine` |
| **Strategy** retrieval | **No** — Jaccard over discrete feature sets only | Yes — via `RefinementEngine` |
| **Claim** retrieval | **Yes** — vector index in `ClaimStore::find_similar()` with outcome-aware re-ranking (§19.4) | Yes — at insertion time |

### B.2 Implication

The `Strategy.summary_embedding` field is populated asynchronously but never used in any search path. Strategy retrieval relies entirely on:
1. Exact context hash lookup
2. Goal bucket index lookup
3. Weighted Jaccard similarity over `{goal_ids, tool_names, result_types}`

Embedding-based semantic search for strategies is architecturally prepared but not yet activated.

---

## Appendix C: Prior Art Comparison & Architectural Differentiation {#appendix-c}

This appendix compares EventGraphDB's claim outcome feedback loop (Section 19) with the state of the art in self-evolving agent memory systems as of February 2026. It identifies where the system aligns with, diverges from, or extends contemporary research, and documents known architectural gaps as future work.

### C.1 Prior Art Survey

| System | Reference | Date | Core Mechanism |
|---|---|---|---|
| **MemRL** | Chen et al., arXiv:2601.03192 | Jan 2026 | Two-phase retrieval (similarity + Q-value), EMA Q-update, non-parametric episodic memory |
| **ERL** | arXiv:2602.13949 | Feb 2026 | Experience-reflection-consolidation loop, explicit self-reflection text, distillation into base policy |
| **HiPER** | arXiv:2602.16165 | Feb 2026 | Hierarchical credit assignment (planning + execution + switching advantages), segment-level reward estimation |
| **Metacognitive Position** | OpenReview:4KhDd0Ozqe | 2025 | Intrinsic metacognitive learning: self-assessment, metacognitive planning, metacognitive evaluation |
| **EventGraphDB** | This system | Feb 2026 | Bayesian outcome scoring on claims, outcome-aware retrieval re-ranking, automatic avoidance claim generation |

### C.2 Detailed Comparison: Memory Re-Ranking

**MemRL** uses a composite score that blends semantic similarity with learned Q-values:

```
score(s, zᵢ, eᵢ) = (1 − λ) · sim_norm(s, zᵢ) + λ · Q_norm(zᵢ, eᵢ)
```

Where `λ ∈ [0, 1]` is a tunable exploration-exploitation parameter, and both similarity and Q-values are z-score normalized.

**EventGraphDB** uses a piecewise Bayesian→Q hybrid with multiplicative composition:

```
base = similarity × (0.6 + 0.4 × temporal_weight)

if total_outcomes < 5:
    outcome_score = (pos + 1) / (total + 2)    // Bayesian
else:
    outcome_score = Q_value                      // EMA (α = 0.3)

multiplier = 0.6 + 0.8 × outcome_score
score = base × multiplier
```

**Key differences:**

| Aspect | MemRL | EventGraphDB |
|---|---|---|
| Score composition | Additive blend (similarity + Q) | **Multiplicative** (base × outcome factor) — relevance is always a hard gate |
| Low-evidence regime | Q from first outcome (noisy) | **Bayesian prior** until 5 outcomes (stable) |
| High-evidence regime | EMA Q-value | **Same**: EMA Q-value (piecewise switch at threshold) |
| Normalization | Z-score across all candidates | Per-claim piecewise score |
| Exploration-exploitation | Explicit λ parameter | Not yet implemented (see §C.5) |
| Update rule | EMA only: `Q ← Q + α(r − Q)` | **Both**: counters (lossless) AND EMA Q (responsive) |
| Convergence | Exponential to expected reward | Bayesian for small N, then exponential |
| Memory overhead | 1 float per entry | 2 integer counters + 1 float in existing metadata map |
| Relevance guarantee | Weak (high Q can override low similarity) | **Strong** (multiplicative: sim=0.1 → max score 0.14) |

**Architectural notes:**

1. **Multiplicative > additive for knowledge retrieval.** MemRL's additive score allows a high-Q but irrelevant memory to outrank a relevant but unproven one. MemRL compensates with Phase A similarity gating (binary cutoff), but EventGraphDB's multiplicative composition provides a smooth gradient where relevance always matters proportionally.

2. **Piecewise resolves the Bayesian vs. EMA trade-off.** Pure Bayesian is too slow to adapt under distribution shift (see §19.3 table: still at 0.647 after 5 consecutive failures vs. Q at 0.17). Pure EMA is too noisy with 1-2 outcomes. The piecewise function uses each where it's strongest.

3. **Lossless counters + Q-value is strictly more informative** than Q-value alone. The lifetime counters enable post-hoc auditability ("this claim had 47 successes and 12 failures") while the Q-value enables responsive scoring. MemRL's EMA-only approach loses the ability to reconstruct outcome history.

### C.3 Credit Assignment Granularity

All three memory-focused systems (MemRL, ERL, EventGraphDB) currently use **trajectory-level credit assignment**: when an outcome arrives, all memories/claims used in that decision receive the same reward signal.

**MemRL:** "Memories actually injected into the context receive updates based on the final task reward."

**ERL:** "No intermediate credit assignment occurs; only terminal outcomes propagate."

**EventGraphDB:** All `claims_used` in a `DecisionTrace` receive the same `success: bool`.

**What the research says is better:**

HiPER (arXiv:2602.16165) demonstrates **hierarchical credit assignment** with three levels:

1. **Execution advantage** — per-action within a subgoal segment
2. **Planning advantage** — per-subgoal at segment boundaries
3. **Switching advantage** — decision to change subgoals vs. continue

The recent "Segment Policy Optimization" work (arXiv:2505.23564) and "Agentic RL with Implicit Step Rewards" (arXiv:2509.19199) show that step-level reward estimation using process reward models can significantly improve over trajectory-level assignment in sparse-reward agent environments.

**Gap in EventGraphDB:** The system assigns identical outcomes to all claims used in a decision. If 3 claims are used and 1 caused the failure, all 3 are penalized equally. This is documented as a known limitation.

**Proposed future work — Differential claim attribution:**

```
For each claim_i used in a failed decision:
  score_i = retrieval_similarity(claim_i) × claim_i.outcome_score()
  attribution_i = score_i / sum(all score_j)

  // High-attribution claims get stronger penalty
  claim_i.record_weighted_outcome(success, weight=attribution_i)
```

This approximation uses the claim's own retrieval score as a proxy for causal responsibility — claims that ranked higher and contributed more to the decision receive proportionally larger outcome updates. This is a lightweight alternative to training a full process reward model.

### C.4 Reflection vs. Avoidance Generation

**ERL's reflection mechanism:**
1. Agent fails → generates explicit text reflection: "I failed because X, I should instead do Y"
2. Reflection guides a second attempt
3. If second attempt succeeds, reflection is stored as a memory entry
4. Successful behavior is distilled into the base policy via supervised learning

**EventGraphDB's avoidance generation:**
1. Claim fails ≥ 2 times → auto-generate: "Avoid relying on: {claim_text} — this has led to repeated failures"
2. Avoidance claim inherits source embedding + entities
3. Avoidance claim appears in same retrieval neighborhood as original
4. No second-attempt retry or distillation step

**Key differences:**

| Aspect | ERL Reflection | EventGraphDB Avoidance |
|---|---|---|
| Content generation | LLM generates causal analysis | Template-based ("Avoid relying on: X") |
| Trigger | Every failure (with retry) | Only after ≥ 2 cumulative failures |
| Retry mechanism | Yes — reflection guides second attempt | No — avoidance only affects future retrievals |
| Distillation | Supervised loss trains base policy | None — pure retrieval-time effect |
| Storage | Overwritten per episode (latest reflection wins) | Additive (new claim per trigger, decays via half-life) |

**Gap in EventGraphDB:** Avoidance claims are template-generated, not LLM-generated. They carry no causal analysis ("why did this fail?") — only a warning ("don't use this"). An LLM-generated reflection would be more informative.

**Proposed future work — LLM-enhanced avoidance claims:**

When avoidance generation triggers, optionally invoke the `RefinementEngine` to generate a richer avoidance text:

```
System: "A claim was used in agent decisions that repeatedly failed.
         Analyze why and generate a concise avoidance rule."
User:   "Claim: {claim_text}
         Used in decisions: {query contexts}
         Outcomes: {failure descriptions}"
Output: "Avoid {specific scenario}: {causal explanation}. Instead: {alternative}."
```

This would transform avoidance claims from warnings into actionable negative knowledge, similar to ERL's reflections.

### C.5 Exploration-Exploitation Balance

**MemRL** explicitly controls exploration via the `λ` parameter:
- `λ → 0`: Pure semantic similarity (exploration — try relevant but unproven memories)
- `λ → 1`: Pure Q-value exploitation (use what worked before)

**EventGraphDB** addresses exploration via **precision-aware active information seeking** (§20.8). Rather than random epsilon-greedy exploration or a fixed λ blend, the system uses prediction error and precision from the hierarchical predictive coding engine to quantify ignorance and adaptively balance exploration/exploitation:

```
score = (1 - ε) × retrieval_score + ε × info_gain
where:
  info_gain = expected_PE × (1 - precision)
  ε = max(0.05, 0.3 × (1 - mean_precision_across_claims))
```

This is principled exploration derived from the free energy principle: the agent explores where its predictions are most uncertain (high expected PE, low precision) and exploits where its model is reliable. Unlike MemRL's static λ, EventGraphDB's ε adapts automatically as the agent's overall confidence changes — high early exploration that naturally anneals as the system learns.

**Fallback** (without GPU predictive coding engine): The Bayesian prior `(pos+1)/(total+2)` provides implicit conservatism (new claims start neutral at 0.5), and the Q-value phase (§19.3) responds to distribution shift. This is less sophisticated but functional without neural models.

### C.6 Memory Rewriting vs. Re-Ranking

**MemRL** can **rewrite memory content** — when a better strategy is discovered, the memory entry is updated in place.

**ERL** creates **new reflection entries** that supersede (but don't delete) old ones.

**EventGraphDB** does **neither** — it only adjusts retrieval ranking via outcome metadata. The original claim text is never modified. Avoidance claims are additive (new entries), not rewrites.

**Trade-off:** Re-ranking is safer (no information loss, no risk of corrupting good claims) but slower to adapt. Rewriting is faster but can destroy useful information if the outcome signal is noisy.

**EventGraphDB's position is intentionally conservative:** claim text represents extracted facts from observed events and should not be retroactively modified by outcome signals. The outcome metadata layer is a separate concern from the truth of the claim itself. A claim like "API X uses REST" remains factually true even if relying on it led to a bad outcome — the outcome penalizes the claim's *usefulness*, not its *truthfulness*.

### C.7 Metacognitive Gap

The OpenReview position paper (4KhDd0Ozqe) argues that truly self-improving agents require **intrinsic metacognitive learning** — the ability to evaluate and adapt their own learning processes.

EventGraphDB currently has:
- **Metacognitive knowledge:** Partial — outcome scores reflect "how useful is this claim?" but not "is my learning process working?"
- **Metacognitive planning:** None — the system does not decide what to learn next
- **Metacognitive evaluation:** None — the system does not evaluate whether the feedback loop is improving agent performance over time

**Proposed future work — Learning loop health metrics:**

```rust
struct FeedbackLoopHealth {
    /// Fraction of recent decisions where outcome-boosted claims were used
    exploitation_rate: f32,
    /// Fraction of recent decisions that used claims with no prior outcomes
    exploration_rate: f32,
    /// Moving average of decision success rate
    rolling_success_rate: f32,
    /// Trend: is success rate improving over time?
    success_rate_trend: f32,
    /// Number of avoidance claims generated in last N decisions
    avoidance_generation_rate: f32,
}
```

These metrics would enable the system to detect when the feedback loop is stagnating (all exploitation, no exploration), degrading (success rate declining despite outcome-aware ranking), or thrashing (excessive avoidance generation suggesting unstable claim quality).

### C.8 Summary of Gaps & Differentiation

| Capability | EventGraphDB (Current) | State of Art | Priority |
|---|---|---|---|
| Outcome-aware retrieval | **Piecewise Bayesian→Q** with multiplicative composition | MemRL: additive Q-value blend with z-norm | **Implemented** — arguably superior for knowledge retrieval (see §C.2) |
| Credit assignment | Trajectory-level (all claims equal) | HiPER: hierarchical; SPO: segment-level | **High** — differential attribution via retrieval-score weighting |
| Avoidance/reflection | Template-based avoidance claims | ERL: LLM-generated causal reflections | **Medium** — add optional LLM refinement to avoidance claims |
| Exploration | Precision-aware active information seeking (§20.8) | MemRL: tunable λ blend | **Designed** — free-energy-principled exploration via PE-driven ε adaptation |
| Memory rewriting | Re-rank only (text immutable) | MemRL: in-place rewrite | **Intentional divergence** — text immutability is a design choice |
| Metacognition | PE convergence metrics, precision histograms (§20.7, §20.11) | Position paper: metacognitive planning + evaluation | **Designed** — PE trends and precision shifts quantify learning health |
| Top-down prediction | **Hierarchical predictive coding** with neural generative models per layer (§20) | VERSES AXIOM: active inference; PCNs: per-layer latents | **Designed** — first application to persistent non-parametric knowledge store |
| PE-gated consolidation | Surprise-driven promotion/pruning (§20.6) | None in MemRL/ERL | **Designed** — novel mechanism, no prior art in agent memory systems |
| Precision-weighted retrieval | Per-claim adaptive precision from PE variance (§20.5) | None | **Designed** — extends outcome multiplier with reliability signal |
| Temporal decay | Type-specific half-life (`2^(-age/half_life)`) | Not addressed in MemRL/ERL | **Advantage** — EventGraphDB's temporal decay is more sophisticated |
| Persistent storage | redb + bincode + HNSW | Typically in-memory only | **Advantage** — production-grade persistence with zero-migration metadata |

---

*End of Patent FTO Analysis*
