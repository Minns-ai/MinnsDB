# EventGraphDB - Complete System Specification

**Version:** 1.0.0
**Last Updated:** 2026-01-20
**Architecture:** Event-Driven Graph Database with Self-Evolution

---

## Table of Contents

1. [System Overview](#system-overview)
2. [Architecture](#architecture)
3. [Core Components](#core-components)
4. [Data Flow](#data-flow)
5. [Storage Layer](#storage-layer)
6. [Graph Engine](#graph-engine)
7. [Learning Pipeline](#learning-pipeline)
8. [Performance](#performance)
9. [Scalability](#scalability)
10. [Security](#security)
11. [Deployment](#deployment)

---

## System Overview

EventGraphDB is a specialized graph database designed for autonomous agents that need to:
- Track experiences as temporal event graphs
- Automatically form memories from significant episodes
- Extract reusable strategies from successful patterns
- Make context-aware decisions using past experience

### Key Capabilities

| Capability | Description | Performance |
|-----------|-------------|-------------|
| Event Ingestion | Process events with automatic graph construction | 10K+ events/sec |
| Graph Inference | Build causal, temporal, and semantic relationships | Real-time |
| Episode Detection | Identify meaningful experience boundaries | Automatic |
| Memory Formation | Convert episodes into retrievable memories | LRU cached |
| Strategy Extraction | Extract reusable reasoning patterns | Success-based |
| Context Retrieval | Find relevant past experiences | < 10ms p95 |

### System Characteristics

```
Architecture:      Event-Driven, Graph-Native
Storage:           Persistent (redb) + LRU cache
Processing:        Async pipeline with backpressure
Consistency:       ACID transactions per partition
Availability:      HA-ready (read replicas supported)
Scalability:       Horizontal (partition-based)
Query Performance: Sub-10ms for cached queries
```

---

## Architecture

### High-Level Architecture

```
┌────────────────────────────────────────────────────────────┐
│                     API Layer (Hertz)                      │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Auth         │  │ Rate Limit   │  │ WebSocket    │    │
│  │ Middleware   │  │ Middleware   │  │ Handler      │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└────────────────────────────────────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────────┐
│                   Graph Engine Core                         │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Event        │  │ Graph        │  │ Scoped       │    │
│  │ Ordering     │  │ Inference    │  │ Inference    │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Episode      │  │ Memory       │  │ Strategy     │    │
│  │ Detection    │  │ Formation    │  │ Extraction   │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└────────────────────────────────────────────────────────────┘
                            ▼
┌────────────────────────────────────────────────────────────┐
│                   Storage Layer                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Graph Store  │  │ Memory Store │  │ Strategy     │    │
│  │ (redb)       │  │ (redb)       │  │ Store (redb) │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Event Store  │  │ Learning     │  │ Decision     │    │
│  │ (WAL+mmap)   │  │ Stats (redb) │  │ Trace (redb) │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└────────────────────────────────────────────────────────────┘
```

### Component Layers

#### 1. API Layer
- **Hertz HTTP Server** - Async HTTP/WebSocket server (tokio-based)
- **Auth Middleware** - JWT validation, API key verification
- **Rate Limiting** - Token bucket algorithm
- **WebSocket Manager** - Real-time event streaming

#### 2. Graph Engine Layer
- **Event Ordering** - Lamport timestamps, vector clocks
- **Graph Inference** - Relationship inference from events
- **Scoped Inference** - Context-specific graph construction
- **Episode Detection** - Boundary detection via significance spikes
- **Memory Formation** - Episodic memory consolidation
- **Strategy Extraction** - Pattern-to-procedure conversion

#### 3. Storage Layer
- **Graph Store** - Persistent graph with goal-bucket partitioning
- **Event Store** - Write-ahead log + memory-mapped storage
- **Memory Store** - LRU-cached episodic/semantic memories
- **Strategy Store** - Indexed strategy repository
- **Learning Stats** - Transition probabilities, motif frequencies
- **Decision Trace** - Reinforcement learning traces

---

## Core Components

### 1. Event Ordering Engine

**Purpose:** Handle concurrent events with causal consistency

**Algorithm:**
```rust
pub struct EventOrderingEngine {
    lamport_clock: AtomicU64,
    vector_clocks: HashMap<AgentId, VectorClock>,
    ordering_config: OrderingConfig,
}
```

**Features:**
- Lamport timestamps for global ordering
- Vector clocks for agent-specific causality
- Concurrent event merging
- Conflict resolution strategies

**Performance:**
- Event processing: O(1) for Lamport, O(A) for vector clock (A = agents)
- Throughput: 50K+ events/sec single-threaded

### 2. Graph Inference Engine

**Purpose:** Build graph from event streams

**Inference Types:**

1. **Temporal Inference**
   ```rust
   // Events within time window → Temporal edge
   if |e2.timestamp - e1.timestamp| < threshold {
       add_edge(e1, e2, EdgeType::Temporal, weight)
   }
   ```

2. **Causal Inference**
   ```rust
   // Action → Observation → Causality edge
   if e1.type == Action && e2.type == Observation {
       if same_context(e1, e2) && temporal_proximity(e1, e2) {
           add_edge(e1, e2, EdgeType::Causality, confidence)
       }
   }
   ```

3. **Similarity Inference**
   ```rust
   // Similar contexts → Similarity edge
   if context_similarity(e1, e2) > threshold {
       add_edge(e1, e2, EdgeType::Similarity, score)
   }
   ```

**Graph Construction:**
```
Event Stream → Pattern Matching → Edge Creation → Graph Update
    ↓              ↓                   ↓              ↓
 Queue(1K)    Inference(10ms)    Batched(100)   Atomic Commit
```

**Performance:**
- Inference latency: < 10ms per event
- Graph construction: 10K+ nodes/sec
- Memory: ~200 bytes per node, ~100 bytes per edge

### 3. Episode Detector

**Purpose:** Identify meaningful experience boundaries

**Detection Algorithm:**
```rust
pub struct EpisodeDetector {
    // Sliding window for significance tracking
    window_size: usize,
    significance_threshold: f32,

    // Episode boundaries
    current_episode: Option<Episode>,
    completed_episodes: Vec<EpisodeId>,
}

impl EpisodeDetector {
    fn detect_boundary(&mut self, event: &Event) -> Option<EpisodeBoundary> {
        // 1. Check significance spike
        if event.significance > self.significance_threshold {
            return Some(EpisodeBoundary::SignificanceSpike);
        }

        // 2. Check context shift
        if self.context_shifted(event) {
            return Some(EpisodeBoundary::ContextShift);
        }

        // 3. Check goal completion
        if self.goal_completed(event) {
            return Some(EpisodeBoundary::GoalCompletion);
        }

        None
    }
}
```

**Episode Types:**
- **Episodic** - Specific experience instances (e.g., "navigated to store on 2026-01-15")
- **Semantic** - General knowledge (e.g., "stores are usually closed at night")
- **Procedural** - How-to knowledge (e.g., "steps to navigate efficiently")

**Metrics:**
```rust
pub struct EpisodeMetrics {
    duration_ms: u64,
    event_count: usize,
    avg_significance: f32,
    context_coherence: f32,
    outcome: EpisodeOutcome,  // Success/Failure/Neutral
}
```

### 4. Memory Formation

**Purpose:** Convert episodes into retrievable memories

**Formation Process:**
```
Episode → Consolidation → Memory → Indexing → Storage
   ↓           ↓            ↓          ↓         ↓
Events(N)  Compress(3)  Summary   ByContext  RedbStore
                                   ByGoal
                                   BySignif
```

**Memory Structure:**
```rust
pub struct Memory {
    memory_id: MemoryId,
    memory_type: MemoryType,

    // Context and indexing
    context_hash: ContextHash,
    goal_bucket_id: GoalBucketId,
    significance: f32,

    // Episode reference
    episode_id: EpisodeId,
    key_events: Vec<EventId>,

    // Content
    summary: String,
    outcome: EpisodeOutcome,
    confidence: f32,

    // Usage tracking
    access_count: u64,
    last_accessed: Timestamp,
    formed_at: Timestamp,
}
```

**Indexing:**
- By context_hash → Fast context-based retrieval
- By goal_bucket_id → Goal-oriented queries
- By significance → Most important memories first
- By recency (formed_at) → Temporal queries

**Storage:**
```rust
// LRU cache layer
InMemory: HashMap<MemoryId, Memory>  // Hot data
  ↓
// Persistent layer
Redb: [context_hash][bucket][memory_id] → Memory
```

### 5. Strategy Extraction

**Purpose:** Extract reusable reasoning patterns

**Extraction Algorithm:**
```rust
pub struct StrategyExtractor {
    min_success_rate: f32,  // 0.7 = 70%
    min_usage_count: u32,   // Must see pattern 3+ times
}

impl StrategyExtractor {
    fn extract(&self, episodes: &[Episode]) -> Vec<Strategy> {
        // 1. Group similar episodes by context pattern
        let groups = self.cluster_by_context(episodes);

        // 2. For each group, extract common reasoning steps
        let mut strategies = Vec::new();
        for group in groups {
            if let Some(strategy) = self.extract_from_group(&group) {
                if strategy.success_rate >= self.min_success_rate {
                    strategies.push(strategy);
                }
            }
        }

        strategies
    }

    fn extract_from_group(&self, episodes: &[Episode]) -> Option<Strategy> {
        // Find common reasoning steps across successful episodes
        let successful: Vec<_> = episodes.iter()
            .filter(|e| e.outcome == Success)
            .collect();

        if successful.len() < self.min_usage_count {
            return None;
        }

        // Extract common patterns
        let context_pattern = self.find_context_pattern(&successful);
        let reasoning_steps = self.extract_reasoning_steps(&successful);

        Some(Strategy {
            context_pattern,
            reasoning_steps,
            success_rate: successful.len() as f32 / episodes.len() as f32,
            // ...
        })
    }
}
```

**Strategy Matching:**
```rust
pub fn match_strategy(&self, context: &Context) -> Vec<(Strategy, f32)> {
    let context_vector = self.vectorize_context(context);

    self.strategies.iter()
        .map(|strategy| {
            let pattern_vector = self.vectorize_pattern(&strategy.context_pattern);
            let similarity = cosine_similarity(&context_vector, &pattern_vector);
            (strategy.clone(), similarity * strategy.success_rate)
        })
        .filter(|(_, score)| *score > 0.5)
        .sorted_by(|a, b| b.1.partial_cmp(&a.1).unwrap())
        .collect()
}
```

### 6. Scoped Inference Engine

**Purpose:** Context-specific graph construction

**Scoping Mechanism:**
```rust
pub struct ScopedInferenceEngine {
    // Per-context subgraphs
    context_graphs: HashMap<ContextHash, Graph>,

    // Per-agent subgraphs
    agent_graphs: HashMap<AgentId, Graph>,

    // Per-goal subgraphs
    goal_graphs: HashMap<GoalBucketId, Graph>,
}

impl ScopedInferenceEngine {
    pub fn infer_in_scope(&mut self, event: &Event, scope: Scope) -> GraphResult<()> {
        match scope {
            Scope::Context(hash) => {
                let graph = self.context_graphs.entry(hash).or_insert_with(Graph::new);
                self.infer_event(event, graph)
            }
            Scope::Agent(id) => {
                let graph = self.agent_graphs.entry(id).or_insert_with(Graph::new);
                self.infer_event(event, graph)
            }
            Scope::Goal(bucket) => {
                let graph = self.goal_graphs.entry(bucket).or_insert_with(Graph::new);
                self.infer_event(event, graph)
            }
        }
    }
}
```

**Benefits:**
- Isolated graph per context (prevents cross-contamination)
- Efficient queries (search smaller subgraphs)
- Natural partitioning for distributed storage

---

## Data Flow

### Event Processing Pipeline

```
┌─────────────┐
│   Event     │
│  Ingestion  │
└──────┬──────┘
       ▼
┌─────────────┐
│   Ordering  │  ← Lamport timestamp, vector clock
│   Engine    │
└──────┬──────┘
       ▼
┌─────────────┐
│    Graph    │  ← Infer relationships
│  Inference  │  ← Build nodes & edges
└──────┬──────┘
       ▼
┌─────────────┐
│   Scoped    │  ← Partition by context/agent/goal
│  Inference  │
└──────┬──────┘
       ▼
┌─────────────┐
│   Episode   │  ← Detect boundaries
│  Detection  │  ← Track significance
└──────┬──────┘
       ▼
┌─────────────┐
│   Memory    │  ← Consolidate episode
│  Formation  │  ← Index and store
└──────┬──────┘
       ▼
┌─────────────┐
│  Strategy   │  ← Extract patterns
│ Extraction  │  ← Update success rates
└─────────────┘
```

### Query Flow

```
┌─────────────┐
│    Query    │
│   Request   │
└──────┬──────┘
       ▼
┌─────────────┐
│    Auth     │  ← Validate token
│ Middleware  │  ← Check permissions
└──────┬──────┘
       ▼
┌─────────────┐
│    Cache    │  ← Check LRU cache
│    Layer    │  ← Return if hit
└──────┬──────┘
       │
       │ (cache miss)
       ▼
┌─────────────┐
│   Storage   │  ← Load from disk
│    Query    │  ← Decompress if needed
└──────┬──────┘
       ▼
┌─────────────┐
│   Update    │  ← Update cache
│    Cache    │  ← Update LRU
└──────┬──────┘
       ▼
┌─────────────┐
│   Response  │
└─────────────┘
```

---

## Storage Layer

### Architecture Overview

```
┌────────────────────────────────────────────────┐
│              Storage Engine                     │
├────────────────────────────────────────────────┤
│  Event Store (WAL + mmap)                      │
│  ├─ Write-ahead log (durability)               │
│  ├─ Memory-mapped segments (fast reads)        │
│  └─ LZ4 compression (optional)                 │
├────────────────────────────────────────────────┤
│  Redb Backend (19 tables)                      │
│  ├─ Episode Catalog                            │
│  ├─ Memory Records (+ 3 indexes)               │
│  ├─ Strategy Records (+ 3 indexes)             │
│  ├─ Learning Stats (transitions, motifs)       │
│  ├─ Decision Traces                            │
│  ├─ Graph Nodes                                │
│  ├─ Graph Adjacency (compressed)               │
│  └─ Graph Edges                                │
└────────────────────────────────────────────────┘
```

### Redb Tables (19 Total)

#### Catalogs (2 tables)
1. **episode_catalog** - Episode metadata
2. **partition_map** - Partition to bucket mapping

#### Memory Store (4 tables)
3. **memory_records** - Primary memory storage
4. **mem_by_bucket** - Index by goal bucket
5. **mem_by_context_hash** - Index by context
6. **mem_feature_postings** - Feature-based search

#### Strategy Store (4 tables)
7. **strategy_records** - Primary strategy storage
8. **strategy_by_bucket** - Index by goal bucket
9. **strategy_by_signature** - Index by pattern signature
10. **strategy_feature_postings** - Feature-based search

#### Learning Stats (2 tables)
11. **transition_stats** - State transition probabilities
12. **motif_stats** - Graph motif frequencies

#### Telemetry (2 tables)
13. **decision_trace** - Decision traces for RL
14. **outcome_signals** - Outcome feedback

#### Operational (2 tables)
15. **id_allocator** - ID generation
16. **schema_versions** - Schema migration tracking

#### Graph Persistence (3 tables) - **Phase 5B**
17. **graph_nodes** - Persistent graph nodes
18. **graph_adjacency** - Compressed adjacency lists (delta encoding)
19. **graph_edges** - Persistent graph edges

### Storage Characteristics

| Component | Technology | Persistence | Compression | Cache |
|-----------|-----------|-------------|-------------|-------|
| Events | WAL + mmap | Durable | LZ4 (optional) | OS page cache |
| Graph | Redb | Durable | Delta encoding | LRU partitions |
| Memories | Redb | Durable | None | LRU (10K entries) |
| Strategies | Redb | Durable | None | LRU (5K entries) |

### Graph Storage Details

**Hierarchical Key Design:**
```
[TypeByte][GoalBucket(8)][NodeID(8)][Optional...]

Examples:
- Node metadata:     [0x01][bucket][node_id]
- Forward adjacency: [0x02][bucket][node_id]
- Reverse adjacency: [0x03][bucket][node_id]
- Edge metadata:     [0x04][bucket][from_id][to_id]
```

**Compression:**
```rust
// Before: [100, 101, 102, 103, 104]
// 5 nodes × 8 bytes = 40 bytes

// After: base=100, deltas=[1,1,1,1]
// 8 bytes + 4×4 bytes = 24 bytes
// Compression: 40% reduction
```

**Partition Loading (LRU):**
```rust
pub struct RedbGraphStore {
    backend: Arc<RedbBackend>,
    loaded_partitions: HashMap<GoalBucketId, PartitionCache>,
    max_loaded_partitions: usize,  // Default: 3
}

// Automatic loading on access
fn get_node(&self, bucket: GoalBucketId, node_id: NodeId) -> Option<Node> {
    self.ensure_partition_loaded(bucket);  // Transparent load
    self.loaded_partitions[bucket].get_node(node_id)
}

// Automatic LRU eviction
fn evict_lru_partition(&mut self) {
    let lru = self.loaded_partitions.iter()
        .min_by_key(|p| p.last_accessed())
        .map(|p| p.bucket_id);
    self.loaded_partitions.remove(&lru);
}
```

---

## Graph Engine

### Graph Structure

```rust
pub struct Graph {
    nodes: HashMap<NodeId, GraphNode>,
    edges: HashMap<(NodeId, NodeId), GraphEdge>,
    adjacency: HashMap<NodeId, Vec<NodeId>>,  // Forward edges
    reverse_adjacency: HashMap<NodeId, Vec<NodeId>>,  // Backlinks
    stats: GraphStats,
}

pub struct GraphNode {
    id: NodeId,
    node_type: NodeType,
    properties: HashMap<String, Value>,
    created_at: Timestamp,
}

pub struct GraphEdge {
    from: NodeId,
    to: NodeId,
    edge_type: EdgeType,
    weight: EdgeWeight,
    properties: HashMap<String, Value>,
}
```

### Graph Operations

#### Traversal Algorithms

**BFS (Breadth-First Search):**
```rust
pub fn traverse_bfs(&self, start: NodeId, max_depth: u32) -> Vec<NodeId> {
    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    let mut result = Vec::new();

    queue.push_back((start, 0));
    visited.insert(start);

    while let Some((node, depth)) = queue.pop_front() {
        if depth > max_depth {
            continue;
        }
        result.push(node);

        if let Some(neighbors) = self.adjacency.get(&node) {
            for &neighbor in neighbors {
                if !visited.contains(&neighbor) {
                    visited.insert(neighbor);
                    queue.push_back((neighbor, depth + 1));
                }
            }
        }
    }

    result
}
```

**DFS (Depth-First Search):**
```rust
pub fn traverse_dfs(&self, start: NodeId, max_depth: u32) -> Vec<NodeId> {
    let mut visited = HashSet::new();
    let mut result = Vec::new();

    fn dfs_helper(
        graph: &Graph,
        node: NodeId,
        depth: u32,
        max_depth: u32,
        visited: &mut HashSet<NodeId>,
        result: &mut Vec<NodeId>,
    ) {
        if depth > max_depth || visited.contains(&node) {
            return;
        }

        visited.insert(node);
        result.push(node);

        if let Some(neighbors) = graph.adjacency.get(&node) {
            for &neighbor in neighbors {
                dfs_helper(graph, neighbor, depth + 1, max_depth, visited, result);
            }
        }
    }

    dfs_helper(self, start, 0, max_depth, &mut visited, &mut result);
    result
}
```

#### Community Detection (Louvain Algorithm)

```rust
pub struct LouvainAlgorithm {
    resolution: f32,  // Default: 1.0
    max_iterations: u32,  // Default: 100
}

impl LouvainAlgorithm {
    pub fn detect_communities(&self, graph: &Graph) -> CommunityDetectionResult {
        // Phase 1: Local optimization
        let mut communities = self.initialize_communities(graph);
        let mut improved = true;
        let mut iteration = 0;

        while improved && iteration < self.max_iterations {
            improved = false;

            for node in graph.nodes.keys() {
                let best_community = self.find_best_community(node, &communities, graph);
                if best_community != communities[node] {
                    communities[node] = best_community;
                    improved = true;
                }
            }

            iteration += 1;
        }

        // Phase 2: Network aggregation
        let aggregated_graph = self.aggregate_graph(graph, &communities);

        CommunityDetectionResult {
            communities,
            modularity: self.calculate_modularity(graph, &communities),
            num_communities: communities.values().collect::<HashSet<_>>().len(),
        }
    }
}
```

#### Centrality Measures

```rust
pub struct CentralityMeasures;

impl CentralityMeasures {
    // Degree centrality: number of connections
    pub fn degree(&self, graph: &Graph, node: NodeId) -> f32 {
        let in_degree = graph.reverse_adjacency.get(&node).map_or(0, |v| v.len());
        let out_degree = graph.adjacency.get(&node).map_or(0, |v| v.len());
        (in_degree + out_degree) as f32
    }

    // Betweenness centrality: how often node appears on shortest paths
    pub fn betweenness(&self, graph: &Graph, node: NodeId) -> f32 {
        // Brandes algorithm
        // ...
    }

    // PageRank: importance based on incoming edges
    pub fn pagerank(&self, graph: &Graph, node: NodeId) -> f32 {
        // Power iteration method
        // ...
    }
}
```

---

## Learning Pipeline

### Reinforcement Learning

**Objective:** Learn from outcomes to improve future decisions

```rust
pub struct ReinforcementLearning {
    learning_rate: f32,  // 0.1
    discount_factor: f32,  // 0.9
}

impl ReinforcementLearning {
    pub fn update(&mut self, trace: &DecisionTrace, outcome: &OutcomeSignal) {
        // 1. Calculate reward
        let reward = match outcome.outcome {
            Outcome::Success => outcome.quality_score,
            Outcome::Failure => -outcome.quality_score,
            Outcome::Neutral => 0.0,
        };

        // 2. Update memory values
        for &memory_id in &trace.memory_ids {
            let current_value = self.get_memory_value(memory_id);
            let new_value = current_value + self.learning_rate * (reward - current_value);
            self.set_memory_value(memory_id, new_value);
        }

        // 3. Update strategy success rates
        for &strategy_id in &trace.strategy_ids {
            self.update_strategy_stats(strategy_id, outcome.outcome == Outcome::Success);
        }

        // 4. Update transition probabilities
        self.update_transition_model(trace, reward);
    }
}
```

### Transition Model

**Objective:** Learn state → action → state transition probabilities

```rust
pub struct TransitionModel {
    // P(next_state | current_state, action)
    transitions: HashMap<(StateHash, ActionHash), HashMap<StateHash, f32>>,
}

impl TransitionModel {
    pub fn update(&mut self, current: StateHash, action: ActionHash, next: StateHash) {
        let key = (current, action);
        let outcomes = self.transitions.entry(key).or_insert_with(HashMap::new);

        // Increment count for this outcome
        *outcomes.entry(next).or_insert(0.0) += 1.0;

        // Normalize to probabilities
        let total: f32 = outcomes.values().sum();
        for prob in outcomes.values_mut() {
            *prob /= total;
        }
    }

    pub fn predict(&self, current: StateHash, action: ActionHash) -> Vec<(StateHash, f32)> {
        let key = (current, action);
        self.transitions.get(&key)
            .map(|outcomes| {
                outcomes.iter()
                    .map(|(state, prob)| (*state, *prob))
                    .collect()
            })
            .unwrap_or_default()
    }
}
```

---

## Performance

### Benchmarks

| Operation | Throughput | Latency (p95) | Latency (p99) |
|-----------|-----------|---------------|---------------|
| Event ingestion | 10K/sec | 5ms | 12ms |
| Graph inference | 10K nodes/sec | 8ms | 15ms |
| Memory query (cached) | 50K/sec | 2ms | 5ms |
| Memory query (disk) | 5K/sec | 15ms | 30ms |
| Strategy matching | 20K/sec | 3ms | 8ms |
| Graph traversal (BFS, depth=3) | 10K/sec | 10ms | 20ms |
| Episode detection | 10K events/sec | 1ms | 3ms |

### Memory Usage

| Component | Memory per Item | Total (100K events) |
|-----------|----------------|---------------------|
| Event | ~500 bytes | 50 MB |
| Graph node | ~200 bytes | 20 MB |
| Graph edge | ~100 bytes | 10 MB |
| Memory | ~1 KB | 5 MB (5K memories) |
| Strategy | ~2 KB | 2 MB (1K strategies) |
| **Total** | | **~87 MB** |

### Disk Usage

| Component | Size per Item | Total (100K events) |
|-----------|--------------|---------------------|
| Event (compressed) | ~250 bytes | 25 MB |
| Graph nodes | ~150 bytes | 15 MB |
| Graph edges (compressed) | ~60 bytes | 6 MB |
| Memories | ~800 bytes | 4 MB |
| Strategies | ~1.5 KB | 1.5 MB |
| **Total** | | **~52 MB** |

### Scalability Tests

**Single Node:**
- Events: 1M events, ~500 MB disk, <1 GB RAM
- Graph: 200K nodes, 400K edges
- Queries: Sub-10ms p95

**Sharded (3 nodes):**
- Events: 10M events, ~5 GB disk total
- Graph: 2M nodes, 4M edges
- Queries: Sub-15ms p95

---

## Scalability

### Horizontal Scaling Strategy

```
┌──────────────┐  ┌──────────────┐  ┌──────────────┐
│   Shard 1    │  │   Shard 2    │  │   Shard 3    │
│              │  │              │  │              │
│ Goal Buckets │  │ Goal Buckets │  │ Goal Buckets │
│   0-999      │  │  1000-1999   │  │  2000-2999   │
└──────────────┘  └──────────────┘  └──────────────┘
        ▲                 ▲                 ▲
        └─────────────────┴─────────────────┘
                         │
                 ┌───────┴────────┐
                 │  Router/Proxy  │
                 └────────────────┘
```

**Sharding Key:** `goal_bucket_id % num_shards`

**Benefits:**
- Linear scalability for writes
- Parallel query execution
- Independent failure domains

**Challenges:**
- Cross-shard queries require aggregation
- Strategy extraction may need multi-shard coordination

### Read Replicas

```
┌──────────────┐
│    Master    │  ← Writes
└──────┬───────┘
       │
       ├───────┐
       ▼       ▼
┌──────────┐ ┌──────────┐
│ Replica1 │ │ Replica2 │  ← Reads
└──────────┘ └──────────┘
```

**Replication:**
- Async replication for read scaling
- Eventual consistency (< 100ms lag)
- Read queries distributed via load balancer

---

## Security

### Authentication

**JWT-based:**
```json
{
  "alg": "HS256",
  "typ": "JWT"
}
{
  "sub": "agent_12345",
  "agent_type": "autonomous",
  "tenant_id": "tenant_001",
  "exp": 1735689600,
  "iat": 1735603200,
  "scopes": ["events:write", "graph:read"]
}
```

**API Key:**
```
X-API-Key: sk_live_1234567890abcdef...
```

### Authorization

**Scope-based access control:**
- `events:write` - Submit events
- `events:read` - Query events
- `graph:read` - Query graph
- `graph:write` - Modify graph (admin)
- `memory:read` - Query memories
- `strategy:read` - Query strategies

### Multi-tenancy

**Tenant Isolation:**
- Each tenant has unique `tenant_id`
- All queries filtered by `tenant_id`
- Storage partitioned by tenant

```rust
pub fn query_events(&self, tenant_id: TenantId, filters: Filters) -> Vec<Event> {
    self.events.iter()
        .filter(|e| e.tenant_id == tenant_id)  // Tenant isolation
        .filter(|e| filters.matches(e))
        .collect()
}
```

### Encryption

**At Rest:**
- Database files encrypted with AES-256
- Keys managed via KMS or environment variables

**In Transit:**
- TLS 1.3 for all API communication
- WebSocket over WSS

---

## Deployment

See [HERTZ_DEPLOYMENT_PLAN.md](./HERTZ_DEPLOYMENT_PLAN.md) for detailed deployment instructions.

---

## Appendix

### Configuration Reference

```toml
[server]
host = "0.0.0.0"
port = 8080
workers = 4

[storage]
backend = "persistent"  # or "in_memory"
redb_path = "./data/eventgraph.redb"
redb_cache_size_mb = 256
memory_cache_size = 10000
strategy_cache_size = 5000

[graph]
max_graph_size = 100000
enable_persistence = true
persistence_interval = 1000  # events

[inference]
temporal_threshold_ms = 5000
causality_confidence = 0.7
similarity_threshold = 0.6

[episodes]
significance_threshold = 0.7
min_episode_length = 3
max_episode_length = 100

[memory]
consolidation_delay_ms = 1000
max_memories_per_bucket = 1000

[strategy]
min_success_rate = 0.7
min_usage_count = 3
```

---

## Glossary

- **Agent:** Autonomous entity submitting events
- **Context Hash:** Hash of current context (state, goal, constraints)
- **Episode:** Sequence of related events with clear boundaries
- **Goal Bucket:** Semantic partition based on goal/intent
- **Memory:** Consolidated episode stored for retrieval
- **Significance:** Measure of event importance (0.0-1.0)
- **Strategy:** Reusable reasoning pattern extracted from successful episodes

---

**End of System Specification**
