# EventGraphDB API Reference

**Complete API documentation for the integrated self-evolution system**

Version: 0.1.0 (MVP)
Last Updated: 2026-01-14

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [GraphEngine API](#graphengine-api)
3. [Event Processing](#event-processing)
4. [Memory Queries](#memory-queries)
5. [Strategy Queries](#strategy-queries)
6. [Policy Guide](#policy-guide)
7. [Episode Queries](#episode-queries)
8. [Statistics & Monitoring](#statistics--monitoring)
9. [Configuration](#configuration)
10. [Examples](#examples)

---

## Quick Start

### Installation

```toml
[dependencies]
agent-db-graph = "0.1.0"
agent-db-events = "0.1.0"
agent-db-core = "0.1.0"
tokio = { version = "1.28", features = ["full"] }
serde_json = "1.0"
```

### Basic Usage

```rust
use agent_db_graph::{GraphEngine, GraphEngineConfig};
use agent_db_events::{Event, EventType, EventContext};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create integrated engine (all features automatic)
    let engine = GraphEngine::new().await?;

    // Process events - self-evolution happens automatically
    for event in your_events {
        engine.process_event(event).await?;
    }

    // Query learned knowledge
    let memories = engine.get_agent_memories(agent_id, 10).await;
    let strategies = engine.get_agent_strategies(agent_id, 10).await;
    let suggestions = engine.get_next_action_suggestions(context_hash, None, 5).await?;

    Ok(())
}
```

---

## GraphEngine API

### Creating a GraphEngine

#### `GraphEngine::new()`

Create a new GraphEngine with default configuration (all automatic features enabled).

```rust
pub async fn new() -> GraphResult<Self>
```

**Example:**
```rust
let engine = GraphEngine::new().await?;
```

**Default Configuration:**
- Auto episode detection: ✅ Enabled
- Auto memory formation: ✅ Enabled
- Auto strategy extraction: ✅ Enabled
- Auto reinforcement learning: ✅ Enabled

---

#### `GraphEngine::with_config()`

Create a GraphEngine with custom configuration.

```rust
pub async fn with_config(config: GraphEngineConfig) -> GraphResult<Self>
```

**Example:**
```rust
let config = GraphEngineConfig {
    auto_episode_detection: true,
    auto_memory_formation: true,
    auto_strategy_extraction: false,  // Disable strategy extraction
    auto_reinforcement_learning: true,
    ..GraphEngineConfig::default()
};

let engine = GraphEngine::with_config(config).await?;
```

---

#### `GraphEngine::with_storage()`

Create a GraphEngine with persistent storage.

```rust
pub async fn with_storage(
    config: GraphEngineConfig,
    storage: Arc<StorageEngine>
) -> GraphResult<Self>
```

**Example:**
```rust
use agent_db_storage::{StorageEngine, StorageConfig};

let storage_config = StorageConfig {
    data_dir: "./data".to_string(),
    ..StorageConfig::default()
};
let storage = Arc::new(StorageEngine::new(storage_config).await?);

let engine = GraphEngine::with_storage(
    GraphEngineConfig::default(),
    storage
).await?;
```

---

## Event Processing

### `process_event()`

Process a single event through the complete self-evolution pipeline.

```rust
pub async fn process_event(&self, event: Event) -> GraphResult<GraphOperationResult>
```

**What Happens Automatically:**
1. Event ordering (handles out-of-order events)
2. Graph construction (nodes & edges created)
3. Episode detection (identifies episode boundaries)
4. Memory formation (creates memories from significant episodes)
5. Strategy extraction (extracts strategies from successful episodes)
6. Reinforcement learning (updates pattern strengths)

**Parameters:**
- `event`: Event - The event to process

**Returns:**
- `GraphOperationResult` - Contains nodes created, patterns detected, processing time

**Example:**
```rust
use agent_db_events::{Event, EventType, ActionOutcome};

let event = Event {
    id: generate_event_id(),
    timestamp: current_timestamp(),
    agent_id: 1,
    agent_type: "ai-debugger".to_string(),
    session_id: 42,
    event_type: EventType::Action {
        action_name: "fix_null_error".to_string(),
        parameters: json!({"fix_type": "add_null_check"}),
        outcome: ActionOutcome::Success { result: json!("fixed") },
        duration_ns: 1_500_000_000,
    },
    causality_chain: vec![],
    context: your_context,
    metadata: HashMap::new(),
};

let result = engine.process_event(event).await?;
println!("Created {} nodes", result.nodes_created.len());
println!("Detected {} patterns", result.patterns_detected.len());
```

---

### `process_events()`

Process multiple events in batch.

```rust
pub async fn process_events(&self, events: Vec<Event>) -> GraphResult<GraphOperationResult>
```

**Example:**
```rust
let events = vec![event1, event2, event3];
let result = engine.process_events(events).await?;
```

---

## Memory Queries

### `get_agent_memories()`

Retrieve all memories for a specific agent.

```rust
pub async fn get_agent_memories(&self, agent_id: AgentId, limit: usize) -> Vec<Memory>
```

**Parameters:**
- `agent_id`: AgentId - The agent whose memories to retrieve
- `limit`: usize - Maximum number of memories to return

**Returns:**
- `Vec<Memory>` - List of memories, sorted by relevance

**Example:**
```rust
let memories = engine.get_agent_memories(1, 10).await;

for memory in memories {
    println!("Memory {}: strength={:.2}, accessed={} times",
        memory.id,
        memory.strength,
        memory.access_count
    );
}
```

**Memory Structure:**
```rust
pub struct Memory {
    pub id: MemoryId,
    pub agent_id: AgentId,
    pub episode_id: Option<EpisodeId>,
    pub context_hash: ContextHash,
    pub memory_type: MemoryType,
    pub strength: f32,              // Current memory strength (0.0 to 1.0)
    pub relevance_score: f32,       // Relevance to current context
    pub access_count: u32,          // Number of times accessed
    pub created_at: Timestamp,
    pub last_accessed: Timestamp,
    pub metadata: HashMap<String, String>,
}
```

---

### `retrieve_memories_by_context()`

Retrieve memories that match a given context (context-based retrieval).

```rust
pub async fn retrieve_memories_by_context(
    &self,
    context: &EventContext,
    limit: usize,
) -> Vec<Memory>
```

**Parameters:**
- `context`: &EventContext - The context to match against
- `limit`: usize - Maximum number of memories to return

**Returns:**
- `Vec<Memory>` - Memories sorted by context similarity and strength

**Example:**
```rust
use agent_db_events::EventContext;

let current_context = EventContext {
    fingerprint: 12345,
    environment: env_state,
    active_goals: goals,
    resources: resources,
    embeddings: None,
};

let relevant_memories = engine.retrieve_memories_by_context(&current_context, 5).await;

for memory in relevant_memories {
    println!("Relevant memory: strength={:.2}, relevance={:.2}",
        memory.strength,
        memory.relevance_score
    );
}
```

---

### `get_memory_stats()`

Get statistics about the memory system.

```rust
pub async fn get_memory_stats(&self) -> MemoryStats
```

**Returns:**
```rust
pub struct MemoryStats {
    pub total_memories: usize,
    pub avg_strength: f32,
    pub avg_access_count: f32,
    pub agents_with_memories: usize,
    pub unique_contexts: usize,
}
```

**Example:**
```rust
let stats = engine.get_memory_stats().await;
println!("Total memories: {}", stats.total_memories);
println!("Average strength: {:.2}", stats.avg_strength);
println!("Average access count: {:.2}", stats.avg_access_count);
```

---

### `decay_memories()`

Manually trigger memory decay (for testing or periodic cleanup).

```rust
pub async fn decay_memories(&self)
```

**Example:**
```rust
// Force decay of all memories
engine.decay_memories().await;
```

---

## Strategy Queries

### `get_agent_strategies()`

Retrieve all strategies learned by a specific agent.

```rust
pub async fn get_agent_strategies(&self, agent_id: AgentId, limit: usize) -> Vec<Strategy>
```

**Parameters:**
- `agent_id`: AgentId - The agent whose strategies to retrieve
- `limit`: usize - Maximum number of strategies to return

**Returns:**
- `Vec<Strategy>` - Strategies sorted by quality score

**Example:**
```rust
let strategies = engine.get_agent_strategies(1, 10).await;

for strategy in strategies {
    println!("Strategy '{}': quality={:.2}",
        strategy.name,
        strategy.quality_score
    );

    for (i, step) in strategy.reasoning_steps.iter().enumerate() {
        println!("  Step {}: {}", i + 1, step.description);
    }
}
```

**Strategy Structure:**
```rust
pub struct Strategy {
    pub id: StrategyId,
    pub name: String,
    pub agent_id: AgentId,
    pub reasoning_steps: Vec<ReasoningStep>,
    pub context_patterns: Vec<ContextPattern>,
    pub success_indicators: Vec<String>,
    pub failure_patterns: Vec<String>,
    pub quality_score: f32,        // 0.0 to 1.0
    pub success_count: u32,
    pub failure_count: u32,
    pub source_episodes: Vec<Episode>,
    pub created_at: Timestamp,
    pub last_used: Timestamp,
    pub metadata: HashMap<String, String>,
}

pub struct ReasoningStep {
    pub description: String,
    pub applicability: String,
    pub expected_outcome: Option<String>,
    pub sequence_order: usize,
}
```

---

### `get_strategies_for_context()`

Retrieve strategies applicable to a specific context.

```rust
pub async fn get_strategies_for_context(
    &self,
    context_hash: ContextHash,
    limit: usize,
) -> Vec<Strategy>
```

**Parameters:**
- `context_hash`: ContextHash - The context hash to match
- `limit`: usize - Maximum number of strategies to return

**Returns:**
- `Vec<Strategy>` - Strategies applicable to this context, sorted by quality

**Example:**
```rust
let context_hash = 12345;  // From your EventContext.fingerprint
let strategies = engine.get_strategies_for_context(context_hash, 5).await;

for strategy in strategies {
    println!("Applicable strategy: {}", strategy.name);
    println!("  Quality: {:.1}%", strategy.quality_score * 100.0);
    println!("  Used {} times", strategy.success_count + strategy.failure_count);
}
```

---

### `get_strategy_stats()`

Get statistics about strategy extraction.

```rust
pub async fn get_strategy_stats(&self) -> StrategyStats
```

**Returns:**
```rust
pub struct StrategyStats {
    pub total_strategies: usize,
    pub high_quality_strategies: usize,   // quality > 0.8
    pub agents_with_strategies: usize,
    pub average_quality: f32,
}
```

**Example:**
```rust
let stats = engine.get_strategy_stats().await;
println!("Total strategies: {}", stats.total_strategies);
println!("High quality: {}", stats.high_quality_strategies);
println!("Average quality: {:.2}", stats.average_quality);
```

---

### `update_strategy_outcome()`

Update a strategy's quality based on external feedback.

```rust
pub async fn update_strategy_outcome(
    &self,
    strategy_id: StrategyId,
    success: bool,
) -> GraphResult<()>
```

**Parameters:**
- `strategy_id`: StrategyId - The strategy to update
- `success`: bool - Whether the strategy application was successful

**Example:**
```rust
// Agent used strategy 42 and it worked
engine.update_strategy_outcome(42, true).await?;

// Agent used strategy 99 and it failed
engine.update_strategy_outcome(99, false).await?;
```

---

## Policy Guide

### `get_next_action_suggestions()` ⭐

**The main "What should I do next?" method.**

Retrieves action suggestions based on past patterns and success rates.

```rust
pub async fn get_next_action_suggestions(
    &self,
    context_hash: ContextHash,
    last_action_node: Option<NodeId>,
    limit: usize,
) -> GraphResult<Vec<ActionSuggestion>>
```

**Parameters:**
- `context_hash`: ContextHash - Current context fingerprint
- `last_action_node`: Option<NodeId> - Previous action (if any)
- `limit`: usize - Maximum number of suggestions

**Returns:**
- `Vec<ActionSuggestion>` - Suggested actions ranked by success probability

**Example:**
```rust
let context_hash = current_context.fingerprint;
let suggestions = engine.get_next_action_suggestions(context_hash, None, 5).await?;

for (i, suggestion) in suggestions.iter().enumerate() {
    println!("{}. {} - {:.1}% success ({} observations)",
        i + 1,
        suggestion.action_name,
        suggestion.success_probability * 100.0,
        suggestion.evidence_count
    );
    println!("   Reasoning: {}", suggestion.reasoning);
}
```

**Output Example:**
```
1. add_null_check - 94.2% success (47 observations)
   Reasoning: This action has worked well in similar contexts (47 times, 94.2% success)

2. add_type_guard - 87.5% success (24 observations)
   Reasoning: This action has followed the previous action 24 times with 87.5% success rate

3. refactor_function - 72.1% success (15 observations)
   Reasoning: This action has worked well in similar contexts (15 times, 72.1% success)
```

**ActionSuggestion Structure:**
```rust
pub struct ActionSuggestion {
    pub action_name: String,
    pub action_node_id: NodeId,
    pub success_probability: f32,    // 0.0 to 1.0
    pub evidence_count: u32,         // Number of observations
    pub reasoning: String,           // Human-readable explanation
}
```

---

## Episode Queries

### `get_completed_episodes()`

Retrieve all completed episodes.

```rust
pub async fn get_completed_episodes(&self) -> Vec<Episode>
```

**Returns:**
- `Vec<Episode>` - List of all completed episodes

**Example:**
```rust
let episodes = engine.get_completed_episodes().await;

for episode in episodes {
    println!("Episode {}: {} events, significance={:.2}",
        episode.id,
        episode.events.len(),
        episode.significance
    );

    if let Some(outcome) = episode.outcome {
        println!("  Outcome: {:?}", outcome);
    }
}
```

**Episode Structure:**
```rust
pub struct Episode {
    pub id: EpisodeId,
    pub agent_id: AgentId,
    pub start_event: EventId,
    pub end_event: Option<EventId>,
    pub events: Vec<EventId>,
    pub context_signature: ContextHash,
    pub outcome: Option<EpisodeOutcome>,
    pub significance: f32,           // 0.0 to 1.0
    pub metadata: HashMap<String, String>,
}

pub enum EpisodeOutcome {
    Success,
    Failure,
    Partial,
    Interrupted,
}
```

---

## Statistics & Monitoring

### `get_engine_stats()`

Get comprehensive statistics about the engine's operation.

```rust
pub async fn get_engine_stats(&self) -> GraphEngineStats
```

**Returns:**
```rust
pub struct GraphEngineStats {
    // Core processing stats
    pub total_events_processed: u64,
    pub total_nodes_created: u64,
    pub total_relationships_created: u64,
    pub total_patterns_detected: u64,
    pub total_queries_executed: u64,
    pub average_processing_time_ms: f64,
    pub cache_hit_rate: f32,
    pub last_operation_time: std::time::Instant,

    // Self-evolution stats
    pub total_episodes_detected: u64,
    pub total_memories_formed: u64,
    pub total_strategies_extracted: u64,
    pub total_reinforcements_applied: u64,
}
```

**Example:**
```rust
let stats = engine.get_engine_stats().await;

println!("📊 Engine Statistics:");
println!("  Events processed:      {}", stats.total_events_processed);
println!("  Episodes detected:     {}", stats.total_episodes_detected);
println!("  Memories formed:       {}", stats.total_memories_formed);
println!("  Strategies extracted:  {}", stats.total_strategies_extracted);
println!("  Reinforcements:        {}", stats.total_reinforcements_applied);
println!("  Avg processing time:   {:.2}ms", stats.average_processing_time_ms);
```

---

### `get_graph_stats()`

Get statistics about the knowledge graph structure.

```rust
pub async fn get_graph_stats(&self) -> GraphStats
```

**Returns:**
```rust
pub struct GraphStats {
    pub node_count: usize,
    pub edge_count: usize,
    pub avg_degree: f32,
    pub largest_component_size: usize,
}
```

**Example:**
```rust
let stats = engine.get_graph_stats().await;

println!("📈 Graph Statistics:");
println!("  Nodes: {}", stats.node_count);
println!("  Edges: {}", stats.edge_count);
println!("  Avg degree: {:.2}", stats.avg_degree);
```

---

### `get_reinforcement_stats()`

Get statistics about the reinforcement learning system.

```rust
pub async fn get_reinforcement_stats(&self) -> ReinforcementStats
```

**Returns:**
```rust
pub struct ReinforcementStats {
    pub total_patterns: usize,
    pub high_confidence_patterns: usize,    // confidence > 0.7
    pub low_confidence_patterns: usize,     // confidence < 0.3
    pub average_confidence: f32,
}
```

**Example:**
```rust
let stats = engine.get_reinforcement_stats().await;

println!("💪 Reinforcement Statistics:");
println!("  Total patterns:        {}", stats.total_patterns);
println!("  High confidence:       {}", stats.high_confidence_patterns);
println!("  Average confidence:    {:.2}", stats.average_confidence);
```

---

### `get_health_metrics()`

Get health metrics for monitoring system status.

```rust
pub async fn get_health_metrics(&self) -> GraphHealthMetrics
```

**Returns:**
```rust
pub struct GraphHealthMetrics {
    pub node_count: usize,
    pub edge_count: usize,
    pub average_degree: f32,
    pub largest_component_size: usize,
    pub events_processed: u64,
    pub processing_rate: f64,           // events per second
    pub memory_usage_estimate: usize,   // bytes
    pub is_healthy: bool,
}
```

**Example:**
```rust
let health = engine.get_health_metrics().await;

if health.is_healthy {
    println!("✅ System healthy");
    println!("  Processing rate: {:.0} events/sec", health.processing_rate);
} else {
    println!("⚠️ System issues detected");
}
```

---

## Configuration

### `GraphEngineConfig`

Complete configuration structure for the GraphEngine.

```rust
pub struct GraphEngineConfig {
    // Component configurations
    pub inference_config: InferenceConfig,
    pub ordering_config: OrderingConfig,
    pub scoped_inference_config: ScopedInferenceConfig,
    pub episode_config: EpisodeDetectorConfig,
    pub memory_config: MemoryFormationConfig,
    pub strategy_config: StrategyExtractionConfig,

    // Automatic feature toggles
    pub auto_pattern_detection: bool,          // default: true
    pub auto_episode_detection: bool,          // default: true
    pub auto_memory_formation: bool,           // default: true
    pub auto_strategy_extraction: bool,        // default: true
    pub auto_reinforcement_learning: bool,     // default: true

    // Performance settings
    pub batch_size: usize,                     // default: 100
    pub max_graph_size: usize,                 // default: 1_000_000

    // Persistence settings
    pub enable_persistence: bool,              // default: true
    pub persistence_interval: u64,             // default: 1000 events
    pub enable_query_cache: bool,              // default: true
}
```

---

### `EpisodeDetectorConfig`

Configuration for episode detection.

```rust
pub struct EpisodeDetectorConfig {
    pub min_significance_threshold: f32,       // default: 0.3
    pub context_shift_threshold: f32,          // default: 0.5
    pub max_time_gap_ns: u64,                  // default: 5_000_000_000
    pub min_events_per_episode: usize,         // default: 2
    pub consolidation_interval_ns: u64,        // default: 10_000_000_000
}
```

---

### `MemoryFormationConfig`

Configuration for memory formation.

```rust
pub struct MemoryFormationConfig {
    pub min_significance: f32,                 // default: 0.5
    pub initial_strength: f32,                 // default: 0.7
    pub decay_rate_per_hour: f32,              // default: 0.1
    pub access_strength_boost: f32,            // default: 0.1
    pub max_strength: f32,                     // default: 1.0
    pub forget_threshold: f32,                 // default: 0.1
}
```

---

### `StrategyExtractionConfig`

Configuration for strategy extraction.

```rust
pub struct StrategyExtractionConfig {
    pub min_significance: f32,                 // default: 0.6
    pub min_success_rate: f32,                 // default: 0.7
    pub min_occurrences: u32,                  // default: 3
    pub max_strategies_per_agent: usize,       // default: 100
}
```

---

## Examples

### Example 1: AI Code Debugger

```rust
use agent_db_graph::GraphEngine;
use agent_db_events::*;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize engine
    let engine = GraphEngine::new().await?;

    // Simulate debugging session
    let agent_id = 1;
    let session_id = 42;
    let base_time = current_timestamp();

    // Event 1: Cognitive - Analyze error
    let event1 = Event {
        id: generate_event_id(),
        timestamp: base_time,
        agent_id,
        agent_type: "ai-debugger".to_string(),
        session_id,
        event_type: EventType::Cognitive {
            process_type: CognitiveType::Reasoning,
            input: json!({"error": "TypeError: undefined"}),
            output: json!({"analysis": "null reference"}),
            reasoning_trace: vec![
                "Analyze error type".to_string(),
                "Identify null reference issue".to_string(),
                "Consider null check vs optional chaining".to_string(),
                "Choose null check for clarity".to_string(),
            ],
        },
        causality_chain: vec![],
        context: create_debug_context(),
        metadata: HashMap::new(),
    };

    // Event 2: Action - Apply fix
    let event2 = Event {
        id: generate_event_id(),
        timestamp: base_time + 1_000_000_000,
        agent_id,
        agent_type: "ai-debugger".to_string(),
        session_id,
        event_type: EventType::Action {
            action_name: "apply_null_check".to_string(),
            parameters: json!({"location": "line 42"}),
            outcome: ActionOutcome::Success {
                result: json!({"tests_pass": true})
            },
            duration_ns: 500_000_000,
        },
        causality_chain: vec![event1.id],
        context: create_debug_context(),
        metadata: HashMap::new(),
    };

    // Process events - automatic learning
    engine.process_event(event1).await?;
    engine.process_event(event2).await?;

    // Query learned knowledge
    let strategies = engine.get_agent_strategies(agent_id, 10).await;
    println!("Agent learned {} strategies", strategies.len());

    for strategy in strategies {
        println!("\nStrategy: {}", strategy.name);
        println!("Quality: {:.1}%", strategy.quality_score * 100.0);
        for step in &strategy.reasoning_steps {
            println!("  - {}", step.description);
        }
    }

    // Get suggestions for next similar error
    let context_hash = create_debug_context().fingerprint;
    let suggestions = engine.get_next_action_suggestions(context_hash, None, 5).await?;

    println!("\nRecommended actions for next error:");
    for (i, suggestion) in suggestions.iter().enumerate() {
        println!("{}. {} ({:.1}% success)",
            i + 1,
            suggestion.action_name,
            suggestion.success_probability * 100.0
        );
    }

    Ok(())
}
```

---

### Example 2: Continuous Learning Loop

```rust
use agent_db_graph::GraphEngine;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let engine = GraphEngine::new().await?;
    let agent_id = 1;

    // Main agent loop
    loop {
        // 1. Get current context
        let context = get_current_context();

        // 2. Retrieve relevant memories
        let memories = engine.retrieve_memories_by_context(&context, 5).await;
        println!("Retrieved {} relevant memories", memories.len());

        // 3. Get action suggestions
        let suggestions = engine.get_next_action_suggestions(
            context.fingerprint,
            None,
            5
        ).await?;

        // 4. Execute best suggestion
        if let Some(best_action) = suggestions.first() {
            println!("Executing: {}", best_action.action_name);
            let event = execute_action(best_action, &context);

            // 5. Record event - automatic learning happens
            engine.process_event(event).await?;
        }

        // 6. Periodically check stats
        if should_log_stats() {
            let stats = engine.get_engine_stats().await;
            println!("Episodes: {}, Memories: {}, Strategies: {}",
                stats.total_episodes_detected,
                stats.total_memories_formed,
                stats.total_strategies_extracted
            );
        }

        tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;
    }
}
```

---

### Example 3: Custom Configuration

```rust
use agent_db_graph::{GraphEngine, GraphEngineConfig};
use agent_db_graph::{EpisodeDetectorConfig, MemoryFormationConfig, StrategyExtractionConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Custom configuration for specific use case
    let config = GraphEngineConfig {
        // Episode detection - more sensitive
        episode_config: EpisodeDetectorConfig {
            min_significance_threshold: 0.2,     // Lower threshold
            max_time_gap_ns: 10_000_000_000,     // Longer gap allowed
            ..EpisodeDetectorConfig::default()
        },

        // Memory formation - longer retention
        memory_config: MemoryFormationConfig {
            decay_rate_per_hour: 0.05,           // Slower decay
            min_significance: 0.4,               // Lower bar for memory
            ..MemoryFormationConfig::default()
        },

        // Strategy extraction - higher quality bar
        strategy_config: StrategyExtractionConfig {
            min_success_rate: 0.8,               // Only high-success strategies
            min_occurrences: 5,                  // Need more evidence
            ..StrategyExtractionConfig::default()
        },

        // Feature toggles
        auto_episode_detection: true,
        auto_memory_formation: true,
        auto_strategy_extraction: true,
        auto_reinforcement_learning: true,

        // Performance tuning
        batch_size: 200,                         // Larger batches
        max_graph_size: 5_000_000,              // Bigger graph allowed

        ..GraphEngineConfig::default()
    };

    let engine = GraphEngine::with_config(config).await?;

    // Use engine as normal...

    Ok(())
}
```

---

## Error Handling

All async methods return `GraphResult<T>` which is `Result<T, GraphError>`.

**Common Errors:**
```rust
pub enum GraphError {
    NodeNotFound(String),
    EdgeNotFound(String),
    CycleDetected,
    OperationError(String),
    InvalidQuery(String),
}
```

**Example:**
```rust
match engine.process_event(event).await {
    Ok(result) => println!("Success: {} nodes created", result.nodes_created.len()),
    Err(GraphError::OperationError(msg)) => eprintln!("Operation failed: {}", msg),
    Err(e) => eprintln!("Error: {:?}", e),
}
```

---

## Performance Tips

1. **Batch Processing**: Use `process_events()` for multiple events
2. **Configure Batch Size**: Adjust `batch_size` based on your workload
3. **Limit Queries**: Use `limit` parameter to cap result sizes
4. **Monitor Health**: Regularly check `get_health_metrics()`
5. **Tune Thresholds**: Adjust detection thresholds based on your domain

---

## Thread Safety

All `GraphEngine` methods are async and thread-safe. You can safely:
- Share `GraphEngine` across threads using `Arc`
- Call methods concurrently from multiple tasks
- Process events in parallel

**Example:**
```rust
let engine = Arc::new(GraphEngine::new().await?);

let handles: Vec<_> = events.chunks(100).map(|chunk| {
    let engine = Arc::clone(&engine);
    let chunk = chunk.to_vec();

    tokio::spawn(async move {
        for event in chunk {
            engine.process_event(event).await?;
        }
        Ok::<_, GraphError>(())
    })
}).collect();

for handle in handles {
    handle.await??;
}
```

---

## See Also

- [TECHNICAL_WRITEUP.md](TECHNICAL_WRITEUP.md) - Deep technical details
- [PROGRESS_REPORT.md](PROGRESS_REPORT.md) - Development progress
- [MVP_SPECIFICATION.md](MVP_SPECIFICATION.md) - Requirements and targets
- [UPDATE_PLAN.md](UPDATE_PLAN.md) - ReasoningBank integration

---

**Questions or issues?** Check the [README.md](README.md) for support information.
