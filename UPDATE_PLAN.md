# EventGraphDB Update Plan: Adding ReasoningBank-Style Features

## Current State Analysis

### What EventGraphDB Currently Has ✅

**Event Structure:**
- `Event` with Action, Cognitive, Observation, Communication types
- `EventContext` with environment, goals, resources, fingerprint
- `ActionOutcome` (Success, Failure, Partial)
- `CausalityChain` for linking events
- `reasoning_trace: Vec<String>` in Cognitive events
- Context similarity matching (`EventContext.similarity()`)

**Graph Inference:**
- Pattern discovery (`detect_patterns()`)
- Temporal patterns (`TemporalPattern`)
- Contextual associations (`ContextualAssociation`)
- Causal relationship inference
- Graph structure with nodes and edges
- Context similarity cache

**Storage:**
- Event persistence (`StorageEngine`)
- Event retrieval by ID (`retrieve_event()`)
- Query by session/time range (`query_events()`)
- WAL for durability
- Compression support

**Traits (Defined but Not Implemented):**
- `MemoryEngine` trait with `retrieve_memories()` signature
- `GraphEngine` trait for graph operations

---

## What Needs to Be Added/Changed

### 1. Episode Detection (Automatic) ❌

**Current:** No episode concept exists

**Need:** Automatic episode boundary detection from event structure:
- Root events (empty `causality_chain`) → episode start
- Goal formation (`CognitiveType::GoalFormation`) → episode start
- Goal completion (terminal `ActionOutcome`) → episode end
- Context shifts (significant `context.fingerprint` change) → episode boundary
- Terminal events (no children in causality graph) → episode end

**Implementation:**
```rust
// New module: crates/agent-db-graph/src/episodes.rs
pub struct Episode {
    id: EpisodeId,
    agent_id: AgentId,
    start_event: EventId,
    end_event: Option<EventId>,
    events: Vec<EventId>,
    context_signature: ContextHash,
    outcome: Option<EpisodeOutcome>,
}

pub struct EpisodeDetector {
    active_episodes: HashMap<AgentId, Episode>,
    graph: Arc<Graph>,
}

impl EpisodeDetector {
    fn process_event(&mut self, event: &Event) -> Option<EpisodeId>;
    fn is_episode_start(&self, event: &Event) -> bool;
    fn is_episode_end(&self, event: &Event) -> bool;
    fn has_context_shift(&self, event: &Event, last_context: &EventContext) -> bool;
}
```

**Changes Required:**
- Add `Episode` and `EpisodeId` types to `agent-db-core`
- Create episode detection module
- Integrate with graph inference pipeline
- Store episodes separately or as graph nodes

---

### 2. Memory Retrieval Implementation ❌

**Current:** `MemoryEngine` trait exists but not implemented

**Need:** Actual `retrieve_memories(context, k)` function that:
- Queries graph for similar contexts
- Returns episodic memories (past episodes)
- Returns semantic patterns (generalized from graph)
- Returns procedural memories (next-step action sequences)
- Returns negative memories (what not to do)

**Implementation:**
```rust
// New module: crates/agent-db-graph/src/memory.rs
pub enum Memory {
    Episodic {
        episode_id: EpisodeId,
        description: String,
        context: EventContext,
        relevance_score: f32,
    },
    Semantic {
        pattern: Pattern,
        applicability: ContextPattern,
        confidence: f32,
    },
    Procedural {
        action_sequence: Vec<Action>,
        context: EventContext,
        success_probability: f32,
    },
    Negative {
        action: Action,
        context: EventContext,
        failure_reason: String,
    },
}

pub struct MemoryEngine {
    graph: Arc<Graph>,
    episodes: Arc<EpisodeStore>,
    context_matcher: ContextMatcher,
}

impl MemoryEngine {
    async fn retrieve_memories(
        &self,
        context: &EventContext,
        limit: usize,
    ) -> DatabaseResult<Vec<Memory>>;
    
    async fn form_memory(&mut self, events: &[Event]) -> DatabaseResult<MemoryId>;
}
```

**Changes Required:**
- Implement `MemoryEngine` trait
- Create `Memory` enum structure
- Build context-based query system
- Integrate with graph and episodes

---

### 3. Policy Guide Queries ❌

**Current:** Graph has patterns but no "next step" queries

**Need:** `get_next_step_suggestions(context, last_action)` that:
- Queries graph: "After this action in this context, what usually works?"
- Returns high-success-probability next actions
- Avoids known dead ends (negative memories)
- Uses graph edge weights as success probability

**Implementation:**
```rust
// Add to crates/agent-db-graph/src/traversal.rs
pub struct ActionSuggestion {
    action_type: String,
    context: EventContext,
    success_probability: f32,
    evidence_count: u32,
    reasoning: String,
}

impl GraphTraversal {
    pub async fn get_next_step_suggestions(
        &self,
        current_context: &EventContext,
        last_action: &Event,
        limit: usize,
    ) -> GraphResult<Vec<ActionSuggestion>>;
    
    pub async fn get_successful_continuations(
        &self,
        from_node: NodeId,
        context: &EventContext,
    ) -> GraphResult<Vec<ActionSuggestion>>;
    
    pub async fn get_dead_ends(
        &self,
        context: &EventContext,
    ) -> GraphResult<Vec<Action>>;
}
```

**Changes Required:**
- Add policy guide queries to `GraphTraversal`
- Use graph edge weights for success probability
- Query for successful paths from current state
- Filter out known failure patterns

---

### 4. Reinforcement Loop ❌

**Current:** Graph discovers patterns but doesn't strengthen/weaken them

**Need:** `reinforce_patterns(episode, success, metrics)` that:
- Strengthens successful patterns (increase edge weights)
- Weakens failure patterns (decrease edge weights)
- Updates pattern confidence scores
- Consolidates repeated successful patterns into "skills"

**Implementation:**
```rust
// Add to crates/agent-db-graph/src/inference.rs
impl GraphInference {
    pub async fn reinforce_patterns(
        &mut self,
        episode_id: EpisodeId,
        success: bool,
        metrics: EpisodeMetrics,
    ) -> GraphResult<()>;
    
    fn strengthen_successful_paths(&mut self, episode: &Episode);
    fn weaken_failure_paths(&mut self, episode: &Episode);
    fn update_edge_weights(&mut self, edge_ids: &[EdgeId], delta: f32);
    fn consolidate_patterns(&mut self, pattern: &TemporalPattern);
}
```

**Changes Required:**
- Add reinforcement methods to `GraphInference`
- Update edge weights based on outcomes
- Track pattern success rates
- Consolidate successful patterns

---

### 5. Strategy Extraction ❌

**Current:** Graph finds patterns but doesn't extract as reusable strategies

**Need:** Extract strategies from graph patterns:
- Generalize reasoning traces from Cognitive events
- Create context patterns
- Store as retrievable strategies
- Link strategies back to original events

**Implementation:**
```rust
// New module: crates/agent-db-graph/src/strategies.rs
pub struct Strategy {
    id: StrategyId,
    name: String,
    generalizable_steps: Vec<String>,
    context_pattern: ContextPattern,
    success_indicators: Vec<String>,
    failure_patterns: Vec<String>,
    self_judged_quality: f32,
    evidence_count: u32,
    related_events: Vec<EventId>,
}

pub struct StrategyStore {
    strategies: HashMap<StrategyId, Strategy>,
    context_index: HashMap<ContextHash, Vec<StrategyId>>,
    graph: Arc<Graph>,
}

impl StrategyStore {
    pub fn extract_from_pattern(
        &mut self,
        pattern: &TemporalPattern,
        graph: &Graph,
        events: &[Event],
    ) -> Strategy;
    
    pub async fn retrieve_strategies(
        &self,
        context: &EventContext,
        similarity_threshold: f32,
        limit: usize,
    ) -> Vec<Strategy>;
    
    fn generalize_reasoning_trace(&self, traces: &[Vec<String>]) -> Vec<String>;
}
```

**Changes Required:**
- Create strategy extraction module
- Implement generalization logic
- Build strategy storage and retrieval
- Integrate with graph pattern detection

---

### 6. Self-Judgment (Optional Enhancement) ⚠️

**Current:** No self-judgment mechanism

**Need:** Add `self_judgment: Option<f32>` to Cognitive events OR infer quality from outcomes

**Implementation Options:**

**Option A: Add to Event Structure**
```rust
// Modify crates/agent-db-events/src/core.rs
EventType::Cognitive {
    process_type: CognitiveType,
    input: serde_json::Value,
    output: serde_json::Value,
    reasoning_trace: Vec<String>,
    self_judgment: Option<f32>, // NEW
}
```

**Option B: Infer from Outcomes**
```rust
// Infer quality from linked Action outcomes
fn infer_quality(cognitive_event: &Event, graph: &Graph) -> f32 {
    // Find linked Action events via causality
    // Use ActionOutcome to infer quality
    // Success → high quality, Failure → low quality
}
```

**Changes Required:**
- Either add field to Cognitive event type
- Or implement inference from outcomes
- Use in strategy extraction

---

### 7. Memory Types Structure ❌

**Current:** No memory structure exists

**Need:** Define `Memory` enum with different types

**Implementation:**
```rust
// In crates/agent-db-graph/src/memory.rs (already defined above)
pub enum Memory {
    Episodic { ... },
    Semantic { ... },
    Procedural { ... },
    Negative { ... },
}
```

**Changes Required:**
- Define memory types
- Implement formation logic for each type
- Implement retrieval logic for each type

---

## Implementation Priority

### Phase 1: Core Functionality
1. **Episode Detection** - Foundation for everything else
2. **Memory Retrieval** - Core user-facing API
3. **Memory Types** - Structure for different memory kinds

### Phase 2: Learning & Guidance
4. **Policy Guide Queries** - "What usually works next?"
5. **Reinforcement Loop** - Strengthen/weaken patterns
6. **Strategy Extraction** - Extract reusable strategies

### Phase 3: Enhancement
7. **Self-Judgment** - Optional quality scoring

---

## Integration Points

### Where Changes Go

**New Modules:**
- `crates/agent-db-graph/src/episodes.rs` - Episode detection
- `crates/agent-db-graph/src/memory.rs` - Memory implementation
- `crates/agent-db-graph/src/strategies.rs` - Strategy extraction

**Modified Modules:**
- `crates/agent-db-graph/src/inference.rs` - Add reinforcement
- `crates/agent-db-graph/src/traversal.rs` - Add policy queries
- `crates/agent-db-events/src/core.rs` - Optional: add self_judgment

**New Types:**
- `crates/agent-db-core/src/types.rs` - EpisodeId, StrategyId, MemoryId

---

## API Surface Changes

### New Public APIs

```rust
// Episode management (automatic, but queryable)
async fn get_episode(episode_id: EpisodeId) -> Option<Episode>;
async fn get_agent_episodes(agent_id: AgentId, limit: usize) -> Vec<Episode>;

// Memory retrieval
async fn retrieve_memories(
    context: &EventContext,
    limit: usize,
) -> DatabaseResult<Vec<Memory>>;

// Policy guide
async fn get_next_step_suggestions(
    context: &EventContext,
    last_action: &Event,
    limit: usize,
) -> DatabaseResult<Vec<ActionSuggestion>>;

// Strategy retrieval
async fn retrieve_strategies(
    context: &EventContext,
    similarity_threshold: f32,
    limit: usize,
) -> DatabaseResult<Vec<Strategy>>;

// Reinforcement (called automatically on episode completion)
async fn reinforce_episode(
    episode_id: EpisodeId,
    success: bool,
    metrics: EpisodeMetrics,
) -> DatabaseResult<()>;
```

---

## Testing Requirements

### Unit Tests
- Episode detection logic
- Memory formation
- Strategy extraction
- Policy query accuracy

### Integration Tests
- End-to-end: event → episode → memory → retrieval
- Reinforcement loop effectiveness
- Strategy generalization accuracy

### Performance Tests
- Memory retrieval latency (<50ms target)
- Policy query latency (<10ms target)
- Episode detection overhead (<1ms per event)

---

## Migration Notes

### Backward Compatibility
- All changes are additive
- Existing event structure unchanged (unless adding self_judgment)
- Existing graph inference continues to work
- New features are opt-in via new API calls

### Data Migration
- No data migration needed
- Episodes computed on-the-fly from existing events
- Strategies extracted from existing graph patterns
- Memories formed from existing event history

---

## Success Criteria

### Functional
- ✅ Episodes automatically detected from events
- ✅ Memories retrieved by context in <50ms
- ✅ Policy suggestions have >70% success rate
- ✅ Strategies generalize correctly from patterns
- ✅ Reinforcement improves pattern quality over time

### Performance
- ✅ Episode detection: <1ms per event
- ✅ Memory retrieval: <50ms for top 10
- ✅ Policy queries: <10ms
- ✅ Strategy extraction: <100ms per pattern

### Quality
- ✅ Episode boundaries detected correctly (>90% accuracy)
- ✅ Memory relevance scores accurate (>85% precision)
- ✅ Policy suggestions useful (>70% agent adoption)
- ✅ Strategies reusable across contexts (>60% applicability)
