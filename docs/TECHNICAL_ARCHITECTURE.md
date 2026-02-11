# EventGraphDB — Technical Architecture

**Confidential — EIC Application Reference Document**
**Version:** 2.0 · February 2026

---

## 1. Executive Summary

EventGraphDB is a **purpose-built cognitive memory engine for autonomous AI agents**, implemented as a high-performance Rust system. Unlike conventional databases that store records for later retrieval, EventGraphDB transforms raw agent telemetry into an evolving knowledge graph that enables agents to **learn from experience, extract reusable strategies, and build semantic understanding** — in real time.

The system implements a biologically-inspired memory architecture across three processing layers:

| Layer | Analogy | Function |
|-------|---------|----------|
| **Episodic Memory** | Hippocampus | Records individual experiences with full context |
| **Semantic Memory** | Neocortex | Generalises knowledge across similar experiences |
| **Procedural Memory** | Cerebellum | Extracts reusable action sequences and strategies |

Built in Rust with zero-copy serialisation and lock-free data paths where possible, the system sustains **sub-millisecond event ingestion** and provides an HTTP REST API for integration with any agent framework.

---

## 2. System Architecture

### 2.1 Crate Dependency Graph

```
                          ┌──────────────────────┐
                          │   eventgraphdb-server │   (HTTP API — Axum)
                          └──────────┬───────────┘
                                     │
                          ┌──────────▼───────────┐
                          │   agent-db-graph      │   (Core Intelligence)
                          │  ┌─────────────────┐  │
                          │  │ episodes         │  │
                          │  │ memory           │  │
                          │  │ strategies       │  │
                          │  │ claims (semantic)│  │
                          │  │ consolidation    │  │
                          │  │ refinement (LLM) │  │
                          │  │ maintenance      │  │
                          │  │ inference        │  │
                          │  │ algorithms       │  │
                          │  │ graph_store      │  │
                          │  │ indexing          │  │
                          │  └─────────────────┘  │
                          └──┬──────────┬─────────┘
                             │          │
               ┌─────────────▼──┐  ┌────▼───────────┐
               │ agent-db-events│  │ agent-db-ner    │
               │ (Event Model)  │  │ (NER Service)   │
               └──────┬────────┘  └────┬────────────┘
                      │                │
               ┌──────▼────────────────▼──┐
               │      agent-db-core       │   (Types, Traits, Config)
               └──────────┬──────────────┘
                          │
               ┌──────────▼──────────────┐
               │    agent-db-storage     │   (redb Persistence)
               └─────────────────────────┘
```

### 2.2 Runtime Architecture

```
  SDK (TypeScript/Python)
        │ HTTP POST /api/events
        ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                    Axum HTTP Server                          │
  │  CorsLayer · JSON validation · serde_with flexible deser.   │
  └────────────────────────┬─────────────────────────────────────┘
                           │
  ┌────────────────────────▼─────────────────────────────────────┐
  │                   GraphEngine (Arc)                          │
  │                                                              │
  │  ┌────────────┐  ┌────────────────┐  ┌────────────────────┐ │
  │  │ Event      │  │ Semantic       │  │ Self-Evolution     │ │
  │  │ Ordering   │  │ Memory Pipeline│  │ Pipeline           │ │
  │  │            │  │                │  │                    │ │
  │  │ watermark  │  │ NER → LLM →   │  │ Episode Detection  │ │
  │  │ reordering │  │ Embedding →   │  │ → Memory Formation │ │
  │  │ causality  │  │ Validation →  │  │ → Strategy Extract │ │
  │  │ buffering  │  │ Dedup → Store │  │ → Reinforcement    │ │
  │  └────────────┘  └────────────────┘  └────────────────────┘ │
  │                                                              │
  │  ┌────────────┐  ┌────────────────┐  ┌────────────────────┐ │
  │  │ Graph      │  │ Consolidation  │  │ Background         │ │
  │  │ Inference  │  │ Engine         │  │ Maintenance        │ │
  │  │            │  │                │  │                    │ │
  │  │ Temporal   │  │ Episodic →    │  │ Memory Decay       │ │
  │  │ Causal     │  │ Semantic →    │  │ Strategy Pruning   │ │
  │  │ Contextual │  │ Schema        │  │ Claim Expiry       │ │
  │  └────────────┘  └────────────────┘  └────────────────────┘ │
  │                                                              │
  │  ┌──────────────────────────────────────────────────────────┐│
  │  │              Persistence Layer (redb)                    ││
  │  │  Memories │ Strategies │ Claims │ Graph │ Episodes │ NER ││
  │  │         ACID transactions · bincode serialisation        ││
  │  └──────────────────────────────────────────────────────────┘│
  └──────────────────────────────────────────────────────────────┘
```

---

## 3. Event Model & Ingestion

### 3.1 Event Schema

Every interaction with the system is modelled as a structured event with nanosecond-precision timestamps:

```rust
struct Event {
    id: u128,                    // Globally unique event ID
    timestamp: u128,             // Nanosecond-precision timestamp
    agent_id: u64,               // Agent identifier
    agent_type: String,          // Agent classification (e.g. "coding-assistant")
    session_id: u64,             // Session boundary
    event_type: EventType,       // Discriminated union (see below)
    causality_chain: Vec<u128>,  // Parent event IDs (DAG)
    context: EventContext,       // Environment snapshot
    metadata: Map<String, Value>,
    context_size_bytes: usize,   // Used for semantic promotion threshold
}
```

**Event Types (discriminated union):**

| Variant | Purpose | Key Fields |
|---------|---------|------------|
| `Action` | Agent decisions & tool calls | `action_name`, `parameters`, `outcome`, `duration_ns` |
| `Observation` | Sensory inputs | `observation_type`, `data`, `confidence`, `source` |
| `Cognitive` | Internal reasoning | `process_type` (Planning/Reflection/Evaluation/...), `reasoning_trace` |
| `Communication` | Inter-agent messaging | `sender`, `recipient`, `content` |
| `Learning` | Explicit feedback signals | `MemoryRetrieved` / `MemoryUsed` / `OutcomeReported` |
| `Context` | Raw text for semantic distillation | `text`, `context_type`, `language` |

### 3.2 Context Fingerprinting

Each event carries an `EventContext` containing environment variables, active goals, and resource availability. The system computes a **deterministic 64-bit fingerprint** (FNV-1a) enabling O(1) context deduplication:

**Algorithm:**
1. **Canonicalise** — sort environment variables alphabetically, sort goals by ID, sort resources alphabetically
2. **Serialise** — produce canonical byte representation (no whitespace, deterministic key order)
3. **Hash** — apply FNV-1a over the canonical bytes → `u64` fingerprint

This fingerprint is cross-language compatible (identical results in Rust, JavaScript, Python) and serves as the partitioning key for episode grouping and memory retrieval.

### 3.3 Flexible Deserialisation Layer

The ingestion layer accepts JSON payloads from SDKs that may serialise numeric IDs inconsistently (as native numbers or string representations). The system uses a `serde_with` strategy stack:

```rust
#[serde_as(as = "IfIsHumanReadable<PickFirst<(_, DisplayFromStr)>>")]
pub agent_id: u64,
```

- **`PickFirst<(_, DisplayFromStr)>`** — attempts native deserialisation first (zero overhead); falls back to parsing from string representation
- **`IfIsHumanReadable`** — applies flexible parsing only for JSON; binary formats (`bincode`) use native deserialisation directly

This yields **zero performance overhead** on the happy path while maintaining SDK compatibility.

### 3.4 Event Ordering Engine

Events from distributed sources may arrive out of order. The ordering engine reconstructs correct temporal and causal ordering:

| Parameter | Default | Purpose |
|-----------|---------|---------|
| `reorder_window_ms` | 5,000 | Maximum time to wait for late events |
| `watermark_window_ms` | 5,000 | Late event acceptance window |
| `max_clock_skew_ms` | 10,000 | Clock drift tolerance between sources |
| `strict_causality` | true | Enforce parent-before-child ordering |

**Algorithm:**
1. Per-agent `BTreeMap<Timestamp, Event>` buffers maintain sorted order
2. A global watermark advances when all agents' minimum timestamps progress
3. Events below the watermark are released as "ready" in correct order
4. Causality violations (child arriving before parent) are logged and buffered

### 3.5 High-Throughput Event Buffer

For burst ingestion, a `VecDeque`-backed buffer with configurable auto-flush:

- **Auto-flush by size**: flush when `N` events buffered
- **Auto-flush by time**: flush after `Duration` elapsed
- **Backpressure**: configurable drop-on-full vs error-on-full policy

---

## 4. Graph Construction & Inference Engine

### 4.1 Knowledge Graph Model

The graph is a heterogeneous property graph with typed nodes and weighted edges:

**Node Types:**

| Node Type | Represents | Key Properties |
|-----------|-----------|----------------|
| `Agent` | AI agent instance | `agent_id`, `capabilities[]` |
| `Event` | A recorded event | `event_id`, `significance` |
| `Context` | Environmental state | `context_hash`, `frequency` |
| `Concept` | Learned abstraction | `concept_name`, `concept_type`, `confidence` |
| `Goal` | Agent objective | `description`, `priority`, `status` |
| `Episode` | Coherent experience | `episode_id`, `outcome` |
| `Memory` | Stored experience | `memory_id`, `agent_id` |
| `Strategy` | Reusable plan | `strategy_id`, `name` |
| `Claim` | Semantic fact | `claim_text`, `confidence` |

**ConceptType hierarchy** (includes NER-derived types):

```
ConceptType
├── BehaviorPattern    ← inferred from event sequences
├── CausalPattern      ← inferred from outcome correlations
├── TemporalPattern    ← inferred from timing regularities
├── ContextualAssociation ← inferred from co-occurrence
├── Strategy           ← extracted from successful episodes
├── Person             ← NER label: PERSON/PER
├── Organization       ← NER label: ORG
├── Location           ← NER label: LOC/GPE
├── Product            ← NER label: PRODUCT
├── DateTime           ← NER label: DATE/TIME
├── Event              ← NER label: EVENT
└── NamedEntity        ← NER label: MISC/NORP/WORK_OF_ART
```

**Edge Types:**

| Edge Type | Semantics |
|-----------|-----------|
| `CausalLink` | Event A causally precedes Event B |
| `TemporalSequence` | Events in temporal order |
| `ContextSimilarity` | Contexts share structural overlap |
| `InteractionLink` | Agent-to-agent communication |
| `GoalRelation` | Goal hierarchy (prerequisite, parallel, conflicting) |
| `PatternMembership` | Node belongs to pattern/community |
| `EntityMention` | Claim references an entity |

### 4.2 Automatic Inference

When an event arrives, the `GraphInference` engine applies five inference passes:

#### 4.2.1 Causal Inference
- **Input**: Event with `causality_chain` and timestamp
- **Algorithm**: Within a configurable time window (`causality_time_window`), if two events from the same agent occur sequentially with matching context, a `CausalLink` edge is created with weight proportional to `1 / Δt` (temporal decay factor)
- **Confidence gating**: Only relationships above `min_confidence_threshold` are persisted

#### 4.2.2 Temporal Pattern Detection
- **Algorithm**: Sliding window over event sequences detects recurring motifs (e.g., `Action → Observation → Cognitive → Action`)
- **Output**: `TemporalPattern { event_sequence, average_interval, confidence, occurrence_count }`
- **Threshold**: Minimum `min_co_occurrence_count` observations required

#### 4.2.3 Context-Based Association
- **Algorithm**: Context fingerprints are compared using cosine similarity over feature vectors extracted from environment variables, goals, and resources
- **Threshold**: `context_similarity_threshold` (default 0.7)
- **Output**: `ContextSimilarity` edges between events sharing similar contexts

#### 4.2.4 Batch Inference
- Events are accumulated and processed in configurable `batch_size` for amortised pattern detection

#### 4.2.5 Scoped Inference
- Inference is scoped by `(agent_type, session_id)` for isolation
- Cross-scope patterns are optionally detected for knowledge transfer between agent types

### 4.3 Property Indexing (100–1000× Query Acceleration)

Three index types provide fast property-based queries:

| Index Type | Implementation | Use Case |
|------------|---------------|----------|
| **B-Tree** | `BTreeMap<IndexKey, Vec<NodeId>>` | Range queries, ordered scans |
| **Hash** | `HashMap<IndexKey, Vec<NodeId>>` | Exact-match lookups |
| **Full-Text** | Inverted index with BM25 scoring | Text search across node properties |

**BM25 Implementation:**
- Term frequency: `tf(t,d) = f(t,d) / |d|`
- Inverse document frequency: `idf(t) = ln((N - n(t) + 0.5) / (n(t) + 0.5) + 1)`
- Scoring: `score = Σ idf(t) · (tf · (k₁ + 1)) / (tf + k₁ · (1 - b + b · |d| / avgdl))`
- Parameters: `k₁ = 1.2`, `b = 0.75`

**Reciprocal Rank Fusion** combines results from multiple index types:
```
RRF_score(d) = Σ 1 / (k + rank_i(d))    where k = 60
```

---

## 5. Graph Algorithms

### 5.1 Centrality Measures

Three centrality algorithms identify important nodes (key actions, critical contexts, influential agents):

#### Degree Centrality
```
C_D(v) = deg(v) / (N - 1)
```

#### Betweenness Centrality
```
C_B(v) = Σ_{s≠v≠t} σ_st(v) / σ_st
```
Where `σ_st` = number of shortest paths from `s` to `t`, and `σ_st(v)` = paths through `v`.

Implementation uses all-pairs BFS shortest path enumeration.

#### PageRank
Iterative power method:
```
PR(v) = (1 - d) / N + d · Σ_{u→v} PR(u) / L(u)
```
- Damping factor `d = 0.85`
- Convergence threshold `ε = 1e-6`
- Maximum 100 iterations

### 5.2 Community Detection — Louvain Algorithm

**Reference:** Blondel et al., "Fast unfolding of communities in large networks" (2008)

The algorithm groups related memories, events, and strategies into communities:

**Phase 1 — Local Modularity Optimisation:**
```
ΔQ = [Σ_in + 2k_{i,in}] / 2m - ([Σ_tot + k_i] / 2m)² 
    - [Σ_in / 2m - (Σ_tot / 2m)² - (k_i / 2m)²]
```

**Phase 2 — Graph Coarsening:**
Each community becomes a super-node; repeat Phase 1 until no improvement.

**Configuration:**
| Parameter | Default | Purpose |
|-----------|---------|---------|
| `resolution` | 1.0 | Higher = more communities |
| `max_iterations` | 100 | Convergence limit |
| `min_improvement` | 0.0001 | Modularity improvement threshold |

### 5.3 Parallel Processing

Graph algorithms leverage `rayon` for CPU parallelism:
- Parallel degree centrality computation
- Parallel multi-source BFS
- Parallel PageRank (partitioned node updates)

Typical speedup: **4–8× on 8-core machines** for graphs > 1,000 nodes.

---

## 6. Self-Evolution Pipeline

The core innovation: transforming raw event streams into evolving agent intelligence.

```
Events → Episode Detection → Memory Formation → Strategy Extraction
                                    ↓                      ↓
                            Consolidation          Reinforcement
                            (Episodic →            (Success/Failure
                             Semantic →             Weighting)
                             Schema)
```

### 6.1 Episode Detection

Episodes are coherent sequences of events forming a unit of experience (analogous to hippocampal episodic encoding).

**Boundary Detection Rules:**
1. **Goal formation** → new episode begins
2. **Goal completion/failure** → episode ends
3. **Session boundary** → episode ends
4. **Context change** (fingerprint delta > threshold) → potential episode boundary

**Significance Scoring:**
```
significance = base
             + (goal_events × 0.15)
             + (prediction_error × 0.2)
             + (novelty_bonus)
```

Where:
- `novelty_bonus = max(0, 1 - context_frequency / 10) × 0.2`
- `prediction_error = |expected_outcome - actual_outcome|`
- `salience_score = surprise + outcome_importance + goal_relevance`

Episodes below `min_significance` (default 0.3) do not trigger memory formation.

### 6.2 Memory Formation

When an episode completes with sufficient significance, a `Memory` is formed:

```rust
struct Memory {
    // Identity
    id: MemoryId,
    agent_id: AgentId,
    episode_id: EpisodeId,
    
    // LLM-Retrievable Content
    summary: String,         // Natural language narrative
    takeaway: String,        // Key lesson learned
    causal_note: String,     // Why the outcome occurred
    summary_embedding: Vec<f32>,  // For semantic search
    
    // Hierarchy
    tier: MemoryTier,        // Episodic | Semantic | Schema
    consolidation_status: ConsolidationStatus,  // Active | Consolidated | Archived
    consolidated_from: Vec<MemoryId>,  // Source memories (for higher tiers)
    
    // Retrieval Metadata
    context: EventContext,   // When this memory applies
    strength: f32,           // [0, 1] — decays over time
    relevance_score: f32,    // [0, 1] — contextual match quality
    access_count: u32,       // Retrieval strengthening
    outcome: EpisodeOutcome, // Success | Failure | Partial | Timeout | Unknown
    memory_type: MemoryType, // Episodic{significance} | Negative{severity, pattern} | Working | Semantic
}
```

**Strength Decay Model:**
```
strength(t) = initial_strength × e^(-λ × Δt_hours)
```
Where `λ = decay_rate_per_hour` (default 0.001). Each access boosts strength by `strength_boost_per_access` (default 0.1).

**Summary Synthesis (template-based, no LLM required):**

The system synthesises a natural language summary by iterating through episode events:
1. Extract action names, outcomes, durations from `Action` events
2. Extract observation types and data snippets from `Observation` events
3. Extract reasoning traces from `Cognitive` events
4. Truncate long fields to 80 characters
5. Compose: `"Episode in session {session}: {action_descriptions}. Observed: {observations}. Reasoning: {reasoning}. Outcome: {outcome}."`

### 6.3 Memory Consolidation Engine

Implements the three-tier memory hierarchy inspired by complementary learning systems theory:

```
            ┌──────────────────────────────────┐
            │         Schema Tier              │
            │  Reusable mental models          │
            │  ("When facing X, do Y")         │
            └──────────────┬───────────────────┘
                           ▲ M+ semantic memories overlap
            ┌──────────────┴───────────────────┐
            │        Semantic Tier             │
            │  Generalised knowledge           │
            │  ("Actions A→B→C succeed 80%")   │
            └──────────────┬───────────────────┘
                           ▲ N+ episodic memories share goal bucket
            ┌──────────────┴───────────────────┐
            │        Episodic Tier             │
            │  Raw individual experiences      │
            │  ("On Jan 5, I tried X and...")  │
            └──────────────────────────────────┘
```

**Consolidation Algorithm:**

**Phase 1: Episodic → Semantic**
1. Collect all `Active` episodic memories
2. Group by `goal_bucket_id` (derived from context fingerprint)
3. For each group with ≥ `episodic_threshold` (default 3) members:
   - Calculate success rate across group
   - Synthesise combined summary from constituent summaries
   - Create new `Semantic` tier memory
   - Mark constituent episodics as `Consolidated` with accelerated decay (`post_consolidation_decay × strength`)

**Phase 2: Semantic → Schema**
1. Collect all `Active` semantic memories
2. Group by overlapping goal buckets
3. For each group with ≥ `semantic_threshold` (default 3) members:
   - Synthesise a schema-level summary
   - Create new `Schema` tier memory
   - Mark constituent semantics as `Consolidated`

### 6.4 LLM Refinement Pipeline

After template-based summary generation, an optional async LLM refinement pass produces higher-quality natural language:

```
Template Summary ─→ LLM (gpt-4o-mini) ─→ Refined Summary
                                         ─→ Takeaway
                                         ─→ Causal Note
                                         ─→ Embedding (text-embedding-3-small)
```

**Memory refinement prompt** produces:
- `summary`: 2–3 sentence narrative (no jargon, no IDs)
- `takeaway`: Single most important lesson
- `causal_note`: Key causal factors explaining the outcome

**Strategy refinement prompt** produces:
- `summary`: Strategy description
- `when_to_use`: Applicability conditions
- `when_not_to_use`: Negative conditions
- `failure_modes`: Known failure scenarios
- `counterfactual`: Alternative approach analysis

### 6.5 Strategy Extraction

Strategies are generalised action plans extracted from successful episodes:

```rust
struct Strategy {
    // Identity & Metadata
    id: StrategyId,
    name: String,
    agent_id: AgentId,
    
    // LLM-Retrievable
    summary: String,
    when_to_use: String,
    when_not_to_use: String,
    failure_modes: Vec<String>,
    playbook: Vec<PlaybookStep>,     // Executable step sequence
    counterfactual: String,
    supersedes: Vec<StrategyId>,     // Version lineage
    applicable_domains: Vec<String>,
    lineage_depth: u32,
    summary_embedding: Vec<f32>,
    
    // Machine-Readable
    reasoning_steps: Vec<ReasoningStep>,
    context_patterns: Vec<ContextPattern>,
    quality_score: f32,              // [0, 1]
    confidence: f32,                 // Bayesian posterior
    success_count: u32,
    failure_count: u32,
    expected_success: f32,           // Posterior success probability
    expected_cost: f32,              // Action cost estimate
    expected_value: f32,             // Value = P(success) × reward - cost
    goal_bucket_id: u64,
    behavior_signature: String,      // Hex-encoded hash of action sequence
}
```

**Extraction Algorithm — Contrastive Motif Distillation:**

1. **Abstract Trace Construction**: Convert episode events into state-action-state transitions
   ```
   Event sequence: [Action(plan), Context(msg), Action(execute)]
   Abstract trace: [S₀ →plan→ S₁ →msg→ S₂ →execute→ S₃]
   ```

2. **Motif Extraction**: Three motif classes:
   - **Transition motifs**: Individual `(state, action, next_state)` triples
   - **Anchor motifs**: `(context_hash, action)` pairs showing context-action correlation
   - **Macro motifs**: Length-3 subsequences of the action sequence

3. **Contrastive Distillation**: Compare motifs from successful vs failed episodes
   ```
   lift(m) = P(success | motif m present) / P(success | motif m absent)
   uplift(m) = P(success | m) - P(success | ¬m)
   ```

4. **Strategy Formation**: Motifs with `lift > 1.0` and `uplift > 0` become positive strategies; motifs correlated with failure become `Constraint` strategies

5. **Bayesian Confidence Update**:
   ```
   posterior_α = prior_success + success_count
   posterior_β = prior_failure + failure_count
   P(success) = α / (α + β)
   ```
   With priors `(α₀=1, β₀=3)` — conservative until evidence accumulates.

6. **Deduplication**: Strategies with identical `behavior_signature` (hash of action sequence) are merged, incrementing `support_count`.

### 6.6 Reinforcement Learning

Explicit learning telemetry events close the feedback loop:

```
LearningEvent::MemoryRetrieved  → records which memories were considered
LearningEvent::MemoryUsed       → marks which memories influenced the decision
LearningEvent::OutcomeReported  → propagates success/failure signal
```

This feeds the `DecisionTraceStore`, which maintains an auditable record of `retrieved → used → outcome` for each decision:

```rust
struct DecisionTrace {
    query_id: String,
    retrieved_memory_ids: Vec<MemoryId>,
    retrieved_strategy_ids: Vec<StrategyId>,
    used_memory_ids: Vec<MemoryId>,       // Subset of retrieved
    used_strategy_ids: Vec<StrategyId>,   // Subset of retrieved
    outcome: Option<OutcomeSignal>,
    policy_version: String,
}
```

### 6.7 Markov Decision Process (MDP) Transition Model

The system maintains a full Markov chain over agent behaviour, enabling probabilistic reasoning about action sequences and outcome prediction.

#### 6.7.1 State Abstraction

Raw events are abstracted into a finite state alphabet for Markov modelling:

| Event Type | Abstract State | Abstract Action |
|-----------|---------------|-----------------|
| `Action{action_name}` | `"Act"` | `action_name` |
| `Observation{type}` | `"Observe"` | `"Observe:{type}"` |
| `Cognitive{process_type}` | `"Think:{process_type}"` | `"Think:{process_type}"` |
| `Communication{message_type}` | `"Communicate"` | `"Comm:{message_type}"` |
| `Context{context_type}` | `"Context:{context_type}"` | `"Context:{context_type}"` |
| `Learning` | `"Learn"` | `"Learn"` |

#### 6.7.2 Abstract Trace Construction

When an episode completes, its event sequence is converted into an abstract trace using a sliding window of size 2:

```
Episode events: [Action("plan"), Context("msg"), Action("execute"), Observation("result")]

States:   ["Act", "Context:msg", "Act", "Observe"]
Actions:  ["Context:msg", "execute", "Observe:result"]

Transitions:
  (Act,         Context:msg,       Context:msg)      → S₀ →a₀→ S₁
  (Context:msg, execute,           Act)               → S₁ →a₁→ S₂
  (Act,         Observe:result,    Observe)            → S₂ →a₂→ S₃
```

Each transition is a `(state, action, next_state)` triple forming the edges of the Markov chain.

#### 6.7.3 Behavior Signature

A deterministic hash of the action skeleton provides a unique identifier for each behavioral pattern:

```
skeleton = ["Act:plan", "Context:msg", "Act:execute", "Observe"]
signature = hex(FNV-1a("Act:plan>Context:msg>Act:execute>Observe"))
```

This signature is used for strategy deduplication — identical behavior sequences map to the same strategy.

#### 6.7.4 Goal-Bucketed Transition Table

Transition statistics are partitioned by **goal bucket** (derived from the episode's active goals), creating separate Markov models per task domain:

```
TransitionModel {
    buckets: HashMap<GoalBucketId, HashMap<(state, action, next_state), TransitionStats>>
}
```

This means the system learns that `Action("plan") → Observation("result")` may have different success rates when pursuing goal A vs goal B.

#### 6.7.5 Bayesian Posterior Updates

Each transition triple maintains Bayesian-updated success statistics:

```rust
struct TransitionStats {
    count: u64,           // Total observations
    success_count: u64,   // Successful episodes containing this transition
    failure_count: u64,   // Failed episodes containing this transition
}
```

**Posterior success probability (Beta-Binomial conjugate):**
```
α = prior_success + success_count     (prior_success = 1.0)
β = prior_failure + failure_count     (prior_failure = 3.0)

P(success | state, action, next_state) = α / (α + β)
```

The conservative prior `(α₀=1, β₀=3)` means a transition starts with a 25% assumed success rate, requiring multiple positive observations to build confidence. The posterior is clamped to `[0.0, 1.0]`.

**Example evolution:**

| Observations | α | β | P(success) |
|-------------|---|---|------------|
| 0 success, 0 failure | 1.0 | 3.0 | 0.25 |
| 1 success, 0 failure | 2.0 | 3.0 | 0.40 |
| 3 success, 0 failure | 4.0 | 3.0 | 0.57 |
| 5 success, 1 failure | 6.0 | 4.0 | 0.60 |
| 10 success, 2 failure | 11.0 | 5.0 | 0.69 |

#### 6.7.6 Late Correction Handling

When an episode is revised (e.g., outcome changes from unknown to success), the transition model performs a **corrective update**:

1. Check if a previous recording exists for this `episode_id`
2. If the previous outcome or goal bucket differs:
   - **Decrement** counts for the old outcome in the old bucket (saturating subtraction)
   - **Increment** counts for the new outcome in the new bucket
3. If the outcome and bucket are identical → skip (idempotent)

This ensures the Markov model remains consistent even when episode outcomes are retroactively corrected.

#### 6.7.7 Persistent Learning Stats Store

Transition statistics and motif statistics are durably stored using `redb` with composite keys:

**Transition key encoding:**
```
[GoalBucketId(8 bytes)][0xFF][state bytes][0xFF][action bytes][0xFF][next_state bytes]
```

**Prefix scan capability:** Querying all transitions from a given state uses a prefix:
```
[GoalBucketId(8)][0xFF][state bytes][0xFF]
```

This enables efficient lookup of "given I'm in state S, what actions are available and what are their success probabilities?"

**Motif key encoding:**
```
[GoalBucketId(8 bytes)][0xFF][motif_id bytes]
```

**Motif statistics** (from the contrastive distiller) are stored alongside transitions:

```rust
struct MotifStats {
    success_count: u32,    // Episodes with this motif that succeeded
    failure_count: u32,    // Episodes with this motif that failed
    lift: f32,             // P(success|motif) / P(success|¬motif)  — >1.0 = beneficial
    uplift: f32,           // P(success|motif) - P(success|¬motif)  — additive improvement
}
```

#### 6.7.8 Composite Path Probability

The system can estimate the probability of success for an entire action sequence by composing individual transition probabilities along the Markov chain:

```
P(success | S₀ →a₀→ S₁ →a₁→ S₂ →a₂→ S₃) ≈ ∏ᵢ P(success | Sᵢ, aᵢ, Sᵢ₊₁)
```

This enables the **policy guide** (`GET /api/suggestions`) to rank candidate action sequences by their estimated probability of success, weighted by expected value.

---

## 7. Semantic Memory Pipeline (Claims System)

### 7.1 Architecture Overview

The semantic memory pipeline extracts atomic facts ("claims") from unstructured text, validates them against source evidence, and stores them for vector-based retrieval:

```
Event (text) ──→ NER Service ──→ LLM Extraction ──→ Evidence Grounding
                                                           │
              Claim Dedup ◄──── Embedding ◄──── Validation │
                   │                                       │
                   ▼                                       ▼
              Claim Store ◄─────────────────────── Graph Linking
              (redb + vector index)
```

### 7.2 Named Entity Recognition (NER)

An external NER service (spaCy-based, containerised) provides sentence-level entity extraction:

```
Input:  "John works at Google in London"
Output: [
  SentenceEntities {
    text: "John works at Google in London",
    entities: [
      EntitySpan { text: "John",   label: "PERSON",  start: 0,  end: 4  },
      EntitySpan { text: "Google", label: "ORG",     start: 18, end: 24 },
      EntitySpan { text: "London", label: "LOC",     start: 28, end: 34 }
    ]
  }
]
```

**Sentence splitting** handles abbreviations (Mr., Dr., U.S., etc.) to avoid false boundaries.

The NER output serves three purposes:
1. **LLM prompt grounding**: Entity labels are passed to the LLM as structured hints
2. **Evidence validation**: Entity overlap between claims and source sentences validates grounding
3. **Graph linking**: NER labels map to `ConceptType` for typed graph node creation

### 7.3 LLM Claim Extraction

The LLM receives raw text plus NER-labeled entities and extracts atomic claims:

**Prompt design:**
```
System: "You are a claim extractor. You receive text and NER entities.
         Use NER labels for classification:
         - PERSON/ORG entities → likely 'fact' or 'preference'
         - PRODUCT entities → likely 'preference' or 'capability'
         
         Classify each claim: fact | preference | belief | intention | capability
         Identify the subject entity."
         
User:   "Text: John said he loves Nike shoes and plans to buy Adidas.

         NER entities detected:
         John [PERSON]
         Nike [PRODUCT]
         Adidas [PRODUCT]
         
         Extract claims with evidence:"
```

**Output structure:**
```json
{
  "claims": [
    {
      "claim_text": "John loves Nike shoes",
      "claim_type": "preference",
      "subject_entity": "John",
      "evidence_spans": [{"start_offset": 0, "end_offset": 30}],
      "confidence": 0.95
    }
  ]
}
```

### 7.4 Evidence Grounding & Validation

Each LLM-extracted claim is validated against the source text using a multi-signal scoring function:

```
support_score = w₁ · cosine_similarity(E_claim, E_sentence)
              + w₂ · entity_overlap(NER_claim, NER_sentence)
              + w₃ · exact_entity_presence(entities_in_text)
```

**Weights:** `w₁ = 0.5` (semantic), `w₂ = 0.3` (NER overlap), `w₃ = 0.2` (literal match)

Where:
- `E_claim`, `E_sentence` = embedding vectors from `text-embedding-3-small`
- `entity_overlap = |E_claim ∩ E_sentence| / max(1, |E_claim|)`
- `exact_entity_presence = 1 - (missing_entities / max(1, total_entities))`

Claims scoring below `min_confidence` are rejected with a documented reason (`BelowConfidenceThreshold`, `NoEvidence`, `ContradictoryEvidence`).

### 7.5 Claim Type System & Temporal Decay

Each claim is classified into a semantic type with an associated temporal half-life:

| Claim Type | Half-Life | Use Case |
|-----------|-----------|----------|
| `Intention` | 3 days | "I want to buy X" — quickly stale |
| `Belief` | 14 days | "I think X will work" — needs validation |
| `Preference` | 30 days | "I like X" — can change |
| `Capability` | 180 days | "System supports X" — stable |
| `Fact` | 365 days | "X costs $10" — very stable |

**Temporal weight decay:**
```
w(t) = 2^(-Δt / half_life)
```

**Retrieval scoring blends similarity with freshness:**
```
final_score = 0.7 × cosine_similarity + 0.3 × temporal_weight
```

### 7.6 Claim Deduplication & Contradiction Resolution

At insertion time, each new claim is checked against existing claims:

1. **Vector similarity search** against the in-memory index (top-K candidates)
2. If `similarity > dedup_threshold` (default 0.92):
   - **Negation detection**: Word-level scan for negation markers (`not`, `don't`, `never`, `hate`, `refuse`, etc.)
   - If exactly one claim is negated → **Contradiction**: old claim marked `Superseded`, new claim stored
   - If both or neither negated → **Duplicate**: increment `support_count` on existing, skip new

### 7.7 Entity Resolution

Entity strings are normalised for consistent graph linking:

```
"The Google"  → "google"
"A Nike shoe" → "nike shoe"
"JOHN"        → "john"
```

**Normalisation rules:**
1. Lowercase
2. Strip determiners (`the`, `a`, `an`)
3. Trim whitespace and trailing punctuation

### 7.8 In-Memory Vector Index

Claims are indexed for fast approximate nearest-neighbour search:

**Structure:** Flat array of L2-normalised embedding vectors

**Insertion:**
```rust
fn upsert(id, embedding):
    magnitude = sqrt(Σ x²)
    normalized = embedding / magnitude
    store(id, normalized)
```

**Search:** Dot-product scan (cosine similarity = dot product for unit vectors)
```rust
fn find_similar(query, top_k, min_similarity):
    normalize(query)
    scores = entries.map(|e| dot(query, e.embedding))
    filter(scores > min_similarity)
    sort_descending()
    return top_k
```

**Claim lifecycle management:** When a claim transitions from `Active` to `Dormant`/`Superseded`/`Rejected`, it is removed from the in-memory index but retained in persistent storage for audit.

---

## 8. Graph Persistence Layer

### 8.1 Storage Engine

The persistence layer uses **redb** — a pure-Rust embedded ACID database with built-in copy-on-write B-tree storage:

| Property | Details |
|----------|---------|
| **ACID Guarantees** | Full transaction support with automatic crash recovery |
| **Serialisation** | `bincode` (compact binary, ~10× smaller than JSON) |
| **Compression** | Delta-encoded adjacency lists (50–70% size reduction) |
| **Partitioning** | Goal-bucket semantic sharding |
| **Cache** | LRU partition loading with monotonic counter for deterministic eviction |

### 8.2 Hierarchical Key Design (Dgraph-inspired)

Keys follow a composite structure for efficient range scans:

```
[TypeByte(1)][GoalBucket(8)][NodeID(8)] = 17 bytes per key
```

| Type Byte | Meaning |
|-----------|---------|
| `0x01` | Node metadata |
| `0x02` | Forward adjacency (A → [B, C, D]) |
| `0x03` | Reverse adjacency (backlinks) |
| `0x04` | Edge metadata |
| `0x05` | Partition statistics |

This enables efficient prefix scans: all nodes in a goal bucket share a common key prefix.

### 8.3 Delta-Encoded Adjacency Lists

Adjacency lists are compressed using delta encoding:

```
Raw:        [100, 101, 105, 106, 110, 200, 201]
Compressed: base=100, deltas=[+1, +4, +1, +4, +90, +1]
```

For sequential node IDs (common in practice), deltas are small integers requiring fewer bytes, yielding **50–70% compression**.

### 8.4 LRU Partition Management

Graph partitions (goal buckets) are loaded on demand with LRU eviction:

```
max_loaded_partitions = 10 (default)
```

A monotonic atomic counter (`AtomicU64`) provides deterministic access ordering, avoiding wall-clock timing issues in high-throughput scenarios.

### 8.5 Persistent Stores

| Store | Tables | Cache Size |
|-------|--------|-----------|
| Memory Store | `memories`, `memory_counter` | 10,000 items (~20 MB) |
| Strategy Store | `strategies`, `strategy_counter` | 5,000 items (~15 MB) |
| Claim Store | `claims`, `claim_counter` | Full in-memory vector index |
| Episode Catalog | `episode_catalog`, `episode_counter` | — |
| Decision Trace | `decision_traces` | — |
| Learning Stats | `learning_transitions`, `learning_motifs` | — |
| Graph Store | `graph_nodes`, `graph_adjacency`, `graph_edges` | LRU partitions |

---

## 9. Background Maintenance

A background `tokio::spawn` task runs periodic housekeeping:

| Task | Interval | Algorithm |
|------|----------|-----------|
| **Memory Decay** | 5 min | Exponential decay: `strength × e^(-λΔt)`. Memories below `forget_threshold` are pruned. |
| **Strategy Pruning** | 5 min | Remove strategies with `confidence < 0.15 AND support < 1 AND stale > 72h` |
| **Strategy Merging** | 5 min | Jaccard similarity on `action_hint` tokens. Merge pairs with `J > 0.6`: combine counts, mark superseded. |
| **Claim Expiry** | 5 min | Claims past `expires_at` timestamp marked `Dormant` and removed from vector index |

**Jaccard Similarity for Strategy Merging:**
```
J(A, B) = |tokens(A) ∩ tokens(B)| / |tokens(A) ∪ tokens(B)|
```

---

## 10. API Surface

### 10.1 REST Endpoints

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/api/events` | Ingest event (full pipeline) |
| `POST` | `/api/events/simple` | Simplified event ingestion |
| `GET` | `/api/events` | List events |
| `GET` | `/api/episodes` | List detected episodes |
| `GET` | `/api/memories/agent/:id` | Retrieve agent memories |
| `POST` | `/api/memories/context` | Context-based memory retrieval |
| `GET` | `/api/strategies/agent/:id` | Retrieve agent strategies |
| `POST` | `/api/strategies/similar` | Similarity-based strategy search |
| `GET` | `/api/suggestions` | Policy guide action suggestions |
| `GET` | `/api/graph` | Full graph snapshot |
| `GET` | `/api/graph/context` | Subgraph for a context |
| `GET` | `/api/stats` | Aggregate metrics (memories, strategies, claims, graph) |
| `GET` | `/api/analytics` | Full graph analytics (centrality, clustering, communities) |
| `GET` | `/api/communities` | Louvain community detection results |
| `GET` | `/api/centrality` | Node centrality rankings |
| `GET` | `/api/indexes` | Property index statistics |
| `POST` | `/api/search` | Multi-index search with rank fusion |
| `GET` | `/api/claims` | List claims |
| `GET` | `/api/claims/:id` | Get claim details |
| `POST` | `/api/claims/search` | Semantic claim search (vector + temporal) |
| `POST` | `/api/embeddings/process` | Process pending embedding queue |

### 10.2 Semantic Memory Activation

Semantic memory extraction is controlled per-request:

```json
POST /api/events?enable_semantic=true
```

When `enable_semantic=true`:
1. NER extraction runs on the event text
2. LLM claim extraction uses NER-labeled entities
3. Claims are validated, deduplicated, and stored
4. Graph nodes are created and linked

Event types processed for semantic extraction:

| Event Type | Text Source |
|-----------|-------------|
| `Context` | `text` field (primary use case) |
| `Action` | `action_name: outcome (parameters)` |
| `Observation` | `observation_type: data` |
| `Cognitive` | `reasoning_trace` joined |
| `Communication` | `message_type: content` |
| `Learning` | Learning event payload |

---

## 11. Deployment Architecture

### 11.1 Container Specification

```dockerfile
# Multi-stage Rust build
FROM rust:1.86-slim AS builder    # Compile with optimisations
FROM debian:bookworm-slim AS runtime  # Minimal runtime (~50 MB)
```

### 11.2 Runtime Configuration

| Environment Variable | Purpose |
|---------------------|---------|
| `SERVER_PORT` | HTTP port (default 3000) |
| `SERVER_HOST` | Bind address (default 0.0.0.0) |
| `OPENAI_API_KEY` | LLM + Embedding API key |
| `OPENAI_MODEL` | LLM model (default gpt-4o-mini) |
| `EMBEDDING_MODEL` | Embedding model (default text-embedding-3-small) |
| `NER_SERVICE_URL` | NER service endpoint |

### 11.3 Data Persistence

All data is stored in a single redb database file (`./data/eventgraph.redb`), persisted via Docker volume mount. redb provides:
- ACID transactions with automatic crash recovery
- No separate WAL configuration needed (built into redb)
- Copy-on-write B-tree storage (no compaction pauses)

### 11.4 Graceful Shutdown

The server handles `SIGTERM`/`SIGINT` signals:
1. Stop accepting new connections
2. Drain in-flight requests (configurable timeout)
3. Flush pending maintenance tasks
4. Close redb database (ensures durability)

---

## 12. Performance Characteristics

| Metric | Value | Conditions |
|--------|-------|------------|
| Event ingestion | < 1 ms | Without semantic pipeline |
| Event ingestion | < 500 ms | With full semantic pipeline (NER + LLM + embedding) |
| Memory retrieval | < 1 ms | Context-hash lookup |
| Strategy retrieval | < 1 ms | Agent-ID lookup |
| Claim vector search | < 10 ms | 100K claims, top-10 |
| Graph traversal (BFS) | < 5 ms | 10K nodes |
| Community detection | < 100 ms | 10K nodes |
| Centrality (PageRank) | < 50 ms | 10K nodes |
| Binary serialisation | ~10× | Smaller than JSON (bincode) |
| Adjacency compression | 50–70% | Delta encoding on sequential IDs |
| Memory footprint | ~50 MB | Base + 10K memories + 5K strategies |

---

## 13. Innovation Summary

### 13.1 Key Differentiators

1. **Biologically-Inspired Memory Hierarchy**: Three-tier consolidation (Episodic → Semantic → Schema) mirrors human memory systems, enabling generalisation from experience

2. **Contrastive Motif Distillation**: Novel algorithm for extracting reusable strategies by comparing successful vs failed episodes — not just replaying success, but understanding *why* actions succeed

3. **Evidence-Grounded Semantic Memory**: Claims are validated against source text using a multi-signal scoring function (semantic similarity + NER overlap + literal presence), preventing hallucinated knowledge

4. **Temporal Knowledge Decay**: Claim types with biologically-inspired half-lives ensure the system prioritises fresh, relevant knowledge while gracefully retiring stale information

5. **Contradiction-Aware Deduplication**: The system detects when new evidence contradicts existing knowledge (negation detection) and performs temporal supersession rather than naive deduplication

6. **Zero-Copy High-Performance Rust Core**: Sub-millisecond event ingestion with ACID durability, enabling real-time deployment in latency-sensitive agent workflows

### 13.2 Technology Readiness Level

| Component | TRL | Status |
|-----------|-----|--------|
| Event ingestion & ordering | 8 | Production-ready |
| Graph construction & inference | 7 | Validated in operational environment |
| Episode detection & memory formation | 7 | Validated |
| Strategy extraction | 7 | Validated |
| Semantic memory (claims pipeline) | 6 | System demonstrated in relevant environment |
| Memory consolidation | 5 | Component validated |
| LLM refinement pipeline | 5 | Component validated |
| Background maintenance | 6 | System demonstrated |

---

*Document generated from EventGraphDB source code analysis, February 2026.*
