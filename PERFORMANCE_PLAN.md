# EventGraphDB Performance & Pathfinding Plan

> Lessons learned from reference implementation reference implementation, applied to EventGraphDB's architecture.
> NOT a copy — these are patterns adapted to our event-sourced, agent-memory graph model.

---

## Table of Contents

1. [Phase 1: Fix Broken Fundamentals](#phase-1-fix-broken-fundamentals)
2. [Phase 2: Pathfinding Implementation](#phase-2-pathfinding-implementation)
3. [Phase 3: Cache & Memory Management Overhaul](#phase-3-cache--memory-management-overhaul)
4. [Phase 4: Batch Operations & Storage Pipeline](#phase-4-batch-operations--storage-pipeline)
5. [Phase 5: Enable Parallelism](#phase-5-enable-parallelism)
6. [Phase 6: Data Structure Optimizations](#phase-6-data-structure-optimizations)
7. [Phase 7: Streaming & Push-Down Optimizations](#phase-7-streaming--push-down-optimizations)

---

## Phase 1: Fix Broken Fundamentals

> Priority: **CRITICAL** — These are bugs, not enhancements.

### 1.1 Fix Dijkstra Edge Weights (Bug)

**Problem:** `traversal.rs:329` uses hardcoded `weight = 1.0` instead of the actual `edge.weight` field from `GraphEdge.weight: f32`.

**File:** `crates/agent-db-graph/src/traversal.rs` (lines 284–350)

**What to change:**
- In the `shortest_path()` method, when iterating neighbors, look up the edge between `current` and `neighbor`
- Use `edge.weight` (or a cost derived from edge type) instead of `1.0`
- For Causality edges: cost = `1.0 - strength` (stronger = cheaper to traverse)
- For Temporal edges: cost = `1.0 - sequence_confidence`
- For Contextual edges: cost = `1.0 - similarity`
- Default fallback: use `edge.weight` directly

**Impact:** Without this fix, shortest_path returns topologically-shortest paths, not semantically-cheapest paths. Every downstream feature that relies on pathfinding (causal chain analysis, strategy extraction) is degraded.

**Estimated scope:** ~30 lines changed in 1 file.

---

### 1.2 Fix Query Cache (Broken)

**Problem:** Query cache infrastructure exists (`traversal.rs:178–270`) but is disabled — line 213 returns a placeholder instead of cached results. Cache insertion at line 266 is commented out.

**File:** `crates/agent-db-graph/src/traversal.rs`

**What to change:**
- Re-enable cache lookup: actually return cached `QueryResult` when key matches and TTL (300s) hasn't expired
- Re-enable cache insertion: after computing a result, store it with `Instant::now()` timestamp
- Add cache invalidation: clear relevant entries when graph mutations occur (add_node, add_edge, delete_node, delete_edge)
- Enforce max_entries (1000) with LRU eviction instead of arbitrary clearing

**Impact:** Repeated identical queries (common during consolidation and inference passes) re-execute full traversals unnecessarily. This is free performance left on the table.

**Estimated scope:** ~40 lines changed in 1 file.

---

### 1.3 Fix Constrained Path Search (Stub)

**Problem:** `traversal.rs:496–505` accepts `PathConstraint` parameters but ignores them entirely, just calling `shortest_path()`.

**File:** `crates/agent-db-graph/src/traversal.rs`

**Constraints to implement:**
```
PathConstraint::MaxLength(u32)        — bound search depth
PathConstraint::RequiredNodeTypes(Vec) — path must include these types
PathConstraint::AvoidNodeTypes(Vec)    — skip these node types during traversal
PathConstraint::MinEdgeWeight(f32)     — prune weak edges
PathConstraint::RequiredEdgeTypes(Vec) — only traverse these edge types
PathConstraint::AvoidEdgeTypes(Vec)    — never traverse these edge types
PathConstraint::CustomFilter(fn)       — user-defined predicate
```

**Approach:** Modified Dijkstra that:
1. Prunes neighbors by AvoidNodeTypes, AvoidEdgeTypes, MinEdgeWeight before relaxation
2. Bounds search depth by MaxLength
3. Post-filters complete paths for RequiredNodeTypes/RequiredEdgeTypes
4. Applies CustomFilter at each expansion step

**Impact:** Constrained pathfinding is essential for causal chain analysis ("find me the causal path from event A to outcome B that goes through at least one Cognitive node"). Without it, the inference engine can't do targeted reasoning.

**Estimated scope:** ~80 lines changed in 1 file.

---

## Phase 2: Pathfinding Implementation

> Priority: **HIGH** — Currently returns empty vec. This is a core graph operation.

### Why Implement Pathfinding?

EventGraphDB is fundamentally a **causal event graph**. Pathfinding answers the questions the system was built for:

1. **Causal chain reconstruction:** "What sequence of events led to this outcome?" — requires path from trigger event to result node, weighted by causality strength
2. **Memory retrieval justification:** "Why is this memory relevant?" — the path from current context to a retrieved memory through semantic/contextual edges IS the explanation
3. **Strategy validation:** "Does this strategy still make sense?" — trace the path from strategy's historical success through the graph to check if conditions still hold
4. **Contradiction resolution:** "Why do these claims conflict?" — find divergent paths from shared evidence nodes to opposing claims
5. **Agent collaboration:** "How did agent A's action affect agent B?" — cross-agent paths through shared events
6. **Debugging & observability:** Without pathfinding, operators can't audit why the system made a decision

**Bottom line:** An event graph without pathfinding is a database without queries. It stores relationships but can't explain them.

### 2.1 Define GraphPath Structure

**File:** `crates/agent-db-graph/src/structures.rs`

**Add:**
```rust
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphPath {
    pub nodes: Vec<NodeId>,
    pub edges: Vec<EdgeId>,
    pub total_cost: f32,
    pub path_type: PathType,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PathType {
    Shortest,           // Dijkstra minimum cost
    AllShortest,        // All paths with same minimum cost
    Constrained,        // Subject to PathConstraints
    KShortest(usize),   // Top-K shortest paths (Yen's algorithm)
}
```

**Estimated scope:** ~25 lines added.

---

### 2.2 Implement Weighted Dijkstra with Edge Costs

**File:** `crates/agent-db-graph/src/traversal.rs`

**Algorithm:** Standard Dijkstra using BinaryHeap (already exists, just fix weights per 1.1).

**Edge cost function:**
```rust
fn edge_cost(edge: &GraphEdge) -> f32 {
    match &edge.edge_type {
        EdgeType::Causality { strength, .. }        => 1.0 - strength,
        EdgeType::Temporal { sequence_confidence, ..} => 1.0 - sequence_confidence,
        EdgeType::Contextual { similarity, .. }      => 1.0 - similarity,
        EdgeType::Semantic { similarity, .. }         => 1.0 - similarity,
        EdgeType::Sequential { confidence, .. }       => 1.0 - confidence,
        EdgeType::SimilarityLink { similarity, .. }   => 1.0 - similarity,
        EdgeType::Reinforcement { reward, .. }        => 1.0 - reward.max(0.0),
        EdgeType::HierarchicalMemory { level, .. }    => *level as f32,
    }
    .max(0.001)  // Prevent zero-cost cycles
}
```

**Estimated scope:** ~20 lines added/changed.

---

### 2.3 Implement A* for Goal-Directed Search

**File:** `crates/agent-db-graph/src/traversal.rs`

**Why A* over plain Dijkstra:** For large graphs with a known target, A* prunes the search space dramatically using a heuristic. In our case:
- **Heuristic for semantic edges:** embedding distance between current node and target (if embeddings available)
- **Heuristic for temporal edges:** timestamp difference normalized by average edge interval
- **Fallback heuristic:** 0.0 (degrades to Dijkstra)

**Algorithm:**
```
fn a_star_path(graph, start, end, heuristic_fn) -> Option<GraphPath>:
    open_set = BinaryHeap with (f_score, node_id)
    g_score = HashMap { start: 0.0 }
    came_from = HashMap {}

    while let Some((_, current)) = open_set.pop():
        if current == end: return reconstruct_path(came_from)
        for neighbor in graph.get_neighbors(current):
            edge = graph.get_edge_between(current, neighbor)
            tentative_g = g_score[current] + edge_cost(edge)
            if tentative_g < g_score.get(neighbor, INF):
                came_from[neighbor] = (current, edge.id)
                g_score[neighbor] = tentative_g
                f_score = tentative_g + heuristic_fn(neighbor, end)
                open_set.push((f_score, neighbor))
    None
```

**Estimated scope:** ~60 lines added.

---

### 2.4 Implement K-Shortest Paths (Yen's Algorithm)

**File:** `crates/agent-db-graph/src/traversal.rs`

**Why:** For causal analysis, the single shortest path isn't always the most informative. Multiple alternative paths reveal:
- Redundant causal chains (resilience)
- Alternative explanations for an outcome
- Hidden dependencies

**Algorithm (Yen's):**
1. Find shortest path P1 using Dijkstra
2. For k = 2..K:
   - For each spur node in P(k-1):
     - Remove edges in previous paths that share the same root
     - Find spur path from spur node to target
     - Combine root + spur as candidate
   - Select lowest-cost candidate as Pk

**Cap K at 10** to prevent combinatorial explosion. Return `Vec<GraphPath>`.

**Estimated scope:** ~100 lines added.

---

### 2.5 Implement Bidirectional Search

**File:** `crates/agent-db-graph/src/traversal.rs`

**Why:** For large graphs, bidirectional BFS/Dijkstra explores ~half the nodes. EventGraphDB already stores both `forward_edges` and `reverse_edges` in PartitionCache, so backward traversal is O(1) per step.

**Algorithm:**
- Forward frontier from `start`, backward frontier from `end`
- Alternate expansion between frontiers
- Terminate when frontiers meet
- Reconstruct path from meeting point

**Estimated scope:** ~70 lines added.

---

### 2.6 Implement find_paths() in RedbGraphStore

**File:** `crates/agent-db-graph/src/redb_graph_store.rs` (line 1087–1096, currently returns `Vec::new()`)

**What to implement:**
```rust
fn find_paths(
    &self,
    bucket: GoalBucketId,
    from: NodeId,
    to: NodeId,
    max_depth: u32,
) -> Result<Vec<GraphPath>, GraphStoreError> {
    self.ensure_partition_loaded(bucket)?;
    let partition = self.loaded_partitions.get(&bucket).unwrap();

    // Depth-limited BFS on the partition's forward_edges
    // Use partition.edge_metadata for edge weights
    // Respect max_depth as hard cutoff
    // Return up to 10 paths (K-shortest within partition)
}
```

**Cross-partition pathfinding:** If from and to are in different buckets, load both partitions and bridge via shared node types (Context, Concept, Goal are in bucket 0).

**Estimated scope:** ~80 lines changed.

---

## Phase 3: Cache & Memory Management Overhaul

> Priority: **HIGH** — Directly impacts query latency and memory footprint.

### 3.1 Replace Context Similarity Cache with LRU

**Problem:** `inference.rs:286–289` clears the ENTIRE cache when size exceeds 10K entries. This causes periodic latency spikes as the cache rebuilds from zero.

**File:** `crates/agent-db-graph/src/inference.rs`

**Design note:** They use `quick_cache::Cache` with weighted entries and lifecycle hooks for eviction.

**What to change:**
- Replace `HashMap<(ContextHash, ContextHash), f32>` with an LRU cache
- Options (in order of preference):
  1. Use `dashmap` (already a dependency!) with a separate LRU tracking layer
  2. Add `lru` crate (lightweight, 0 dependencies)
  3. Add `quick-cache` crate (more features, concurrent)
- Evict oldest 10% when at capacity instead of clearing all
- Add hit/miss counters for observability

**Estimated scope:** ~30 lines changed, possibly 1 new dependency.

---

### 3.2 Add TTL + LRU to Query Cache

**Problem:** Query cache in `traversal.rs:178` uses bare `HashMap` with no working eviction.

**File:** `crates/agent-db-graph/src/traversal.rs`

**What to change:**
- Replace `HashMap<String, (QueryResult, Instant)>` with proper LRU
- TTL: 300 seconds (already defined, just not enforced)
- Max entries: 1000 (already defined, just not enforced)
- Add cache invalidation on graph mutations
- Track hit rate for tuning

**Estimated scope:** ~50 lines changed.

---

### 3.3 Batch Memory/Strategy Store Operations

**Problem:** `RedbMemoryStore.store_episode()` (stores.rs:373–464) calls `persist_memory()` individually for each memory. Each persist is a separate redb transaction with separate index updates.

**Design note:** Batch writes use a single atomic transaction across all operations.

**File:** `crates/agent-db-graph/src/stores.rs`

**What to change:**
- Collect all `BatchOperation`s during episode processing
- Single `backend.write_batch(ops)` call at the end
- Same pattern for `RedbStrategyStore.store_episode()` (stores.rs:1164–1188)
- Same pattern for consolidation writes (consolidation.rs mark_consolidated calls)

**Impact:** Reduces transaction overhead from O(N) to O(1) per episode. For an episode with 10 memories, this is ~10x fewer disk syncs.

**Estimated scope:** ~60 lines changed across 2 files.

---

### 3.4 Weighted LRU for Partition Cache

**Current:** `RedbGraphStore.loaded_partitions` uses monotonic counter for LRU with flat eviction.

**Design note:** Weighted caching accounts for partition SIZE, not just recency. A partition with 50K nodes costs more memory than one with 100 nodes.

**File:** `crates/agent-db-graph/src/redb_graph_store.rs`

**What to change:**
- Track partition memory footprint: `nodes.len() * ~200B + edges * ~100B`
- Eviction score = `age_ticks * memory_weight`
- Set a total memory budget (e.g., 256MB) instead of flat partition count (currently max 8)
- Keep more small partitions, fewer large ones

**Estimated scope:** ~40 lines changed.

---

## Phase 4: Batch Operations & Storage Pipeline

> Priority: **MEDIUM-HIGH** — Reduces I/O overhead across the board.

### 4.1 Batch Node Reads in Traversal

**Problem:** BFS/DFS in traversal.rs reads nodes one at a time during expansion.

**Design note:** `BATCH_SIZE = 1000`, accumulate RecordIds, then `multi_get()` in one storage call.

**File:** `crates/agent-db-graph/src/traversal.rs`

**What to change:**
- In BFS: collect all neighbors from current frontier level into a batch
- Fetch all neighbor nodes in single partition cache lookup
- Add `get_nodes_batch(ids: &[NodeId])` to `GraphStore` trait and implementations
- For `RedbGraphStore`: single `ensure_partition_loaded()` + batch HashMap lookups

```rust
// GraphStore trait addition
fn get_nodes_batch(
    &self,
    bucket: GoalBucketId,
    ids: &[NodeId]
) -> Result<Vec<Option<GraphNode>>, GraphStoreError>;
```

**Impact:** Eliminates per-node function call overhead during traversal. For BFS with fanout of 10, each level goes from 10 individual lookups to 1 batch lookup.

**Estimated scope:** ~50 lines across 3 files (trait + 2 impls).

---

### 4.2 Batch Consolidation Writes

**Problem:** `consolidation.rs` calls `store.store_consolidated_memory()` and `store.mark_consolidated()` individually for each memory in a consolidation group.

**File:** `crates/agent-db-graph/src/consolidation.rs`

**What to change:**
- Add `store_consolidated_memories_batch(memories: Vec<Memory>)` to MemoryStore trait
- Add `mark_consolidated_batch(ids: Vec<(MemoryId, MemoryId, MemoryTier, f32)>)` to MemoryStore trait
- Implement in `RedbMemoryStore` using single `write_batch()`
- Collect all operations per consolidation pass, commit once

**Impact:** A single consolidation pass touching 15 memories goes from ~30 transactions to 1.

**Estimated scope:** ~80 lines across 2 files.

---

### 4.3 Atomic Event Buffer Batch

**Problem:** `EventBuffer.add_batch()` (buffer.rs:128–133) just loops over `add()`. If it fails mid-batch, partial events are buffered.

**File:** `crates/agent-db-events/src/buffer.rs`

**What to change:**
- Check capacity for entire batch before inserting any
- Either accept all events or reject all (atomic semantics)
- For `RingEventBuffer`: reserve space, then memcpy in bulk

**Estimated scope:** ~30 lines changed.

---

## Phase 5: Enable Parallelism

> Priority: **HIGH** — Single biggest performance win available.

### 5.1 Enable Rayon

**Problem:** `algorithms/parallel.rs:3–11` has Rayon disabled due to Rust 1.80+ requirement. Rust 1.80 has been stable since **July 2024**. We should be well past that now.

**File:** `crates/agent-db-graph/Cargo.toml` + `crates/agent-db-graph/src/algorithms/parallel.rs`

**What to change:**
1. Uncomment rayon dependency in Cargo.toml
2. Set `rust-version = "1.80"` in Cargo.toml (or verify MSRV)
3. Replace sequential iterators with `par_iter()` in:
   - PageRank computation (matrix multiply step)
   - Community detection (modularity calculation)
   - Connected components (parallel union-find)
   - Centrality measures (betweenness centrality is embarrassingly parallel)

**Expected impact:** 4–8x speedup on 8-core machines for graph algorithm passes. reference implementation uses similar patterns for their vector index operations.

**Estimated scope:** ~100 lines changed across 2 files.

---

### 5.2 Concurrent Bucket Consolidation

**Problem:** Consolidation engine processes all goal buckets sequentially (consolidation.rs:107–153). Each bucket's consolidation is completely independent.

**Design note:** Two-phase write model — enqueue lock-free, apply in background.

**File:** `crates/agent-db-graph/src/consolidation.rs`

**What to change:**
- Use `rayon::par_iter()` over goal bucket groups
- Each bucket's episodic→semantic consolidation runs independently
- Collect results, then merge consolidated memories into store (write phase)
- Schema consolidation (cross-bucket) remains sequential

**Constraints:**
- MemoryStore trait uses `&mut self` — need to split read phase (immutable) from write phase (mutable)
- Read: `list_all_memories()` (takes &self)
- Compute: group + synthesize (pure computation, no store access)
- Write: `store_consolidated_memory()` + `mark_consolidated()` (takes &mut self)

**Estimated scope:** ~60 lines changed.

---

### 5.3 Parallel Graph Maintenance

**File:** `crates/agent-db-graph/src/maintenance.rs`

**What to change:**
- Memory decay, strategy pruning, and claim maintenance are independent tasks
- Run them in parallel using `rayon::join()` or `tokio::join!`
- Graph node cap enforcement remains sequential (needs consistent state)

**Estimated scope:** ~30 lines changed.

---

## Phase 6: Data Structure Optimizations

> Priority: **MEDIUM** — Reduces allocation pressure and improves cache locality.

### 6.1 SmallVec for Low-Degree Adjacency

**Design note:** `ArraySet<N>` uses stack allocation for small neighbor sets. Most nodes in agent graphs have < 8 neighbors.

**File:** `crates/agent-db-graph/src/redb_graph_store.rs`

**What to change:**
- Replace `HashMap<NodeId, Vec<EdgeId>>` in PartitionCache with `HashMap<NodeId, SmallVec<[EdgeId; 8]>>`
- Add `smallvec` dependency to Cargo.toml
- Nodes with ≤ 8 edges: zero heap allocation for adjacency
- Nodes with > 8 edges: transparent fallback to heap Vec

**Impact:** Reduces memory allocations during partition loading. For a partition with 1000 nodes where 80% have ≤ 8 edges, this eliminates ~800 Vec heap allocations.

**Estimated scope:** ~15 lines changed, 1 new dependency.

---

### 6.2 Roaring Bitmaps for Index Posting Lists

**Design note:** Uses `Ids64` enum that promotes to `RoaringTreemap` for large ID sets. Roaring bitmaps give excellent compression and O(1) intersection/union.

**File:** `crates/agent-db-graph/src/indexing.rs`

**What to change:**
- Replace `Vec<NodeId>` in property index posting lists with `RoaringBitmap`
- Add `roaring` crate dependency
- Index queries with AND/OR/NOT conditions become bitmap operations instead of sorted-merge joins

**Impact:** For a property index with 10K entries per value, intersection of two index lookups goes from O(N log N) merge-sort to O(N/64) bitwise AND.

**Estimated scope:** ~50 lines changed, 1 new dependency.

---

### 6.3 Arc::try_unwrap in Event Pipeline

**Design note:** Avoids unnecessary clones when refcount == 1.

**File:** `crates/agent-db-graph/src/integration.rs`

**Where to apply:**
- Event processing pipeline where events flow through buffer → graph → memory → consolidation
- When passing `Arc<Event>` through pipeline stages, use `Arc::try_unwrap()` to move data instead of cloning when possible

```rust
// Instead of:
let event_data = arc_event.clone();

// Use:
let event_data = match Arc::try_unwrap(arc_event) {
    Ok(event) => event,
    Err(arc) => (*arc).clone(),
};
```

**Impact:** Eliminates deep clones of Event structs (which contain `HashMap<String, MetadataValue>`, `Vec<EventId>`, and `EventContext`). Each avoided clone saves ~500 bytes of allocation.

**Estimated scope:** ~20 lines changed.

---

### 6.4 Compact NodeHeader Binary Read

**Problem:** `graph_store.rs:40–86` deserializes full `NodeHeader` struct for eviction scoring. Could read just the fields needed for scoring without full deserialization.

**Design note:** Uses lightweight `MemoryReporter` trait for tracking without full struct access.

**File:** `crates/agent-db-graph/src/graph_store.rs`

**What to change:**
- Define a fixed binary layout for NodeHeader (currently uses bincode which has overhead)
- Read only the scoring fields: `signal(4B) + tier(1B) + degree(4B) + updated_at(8B)` = 17 bytes
- Skip `node_id`, `node_type_discriminant`, `created_at`, `goal_bucket` during scoring

**Impact:** During maintenance passes that scan all headers, reduces deserialization work by ~60%.

**Estimated scope:** ~40 lines changed.

---

## Phase 7: Streaming & Push-Down Optimizations

> Priority: **MEDIUM** — Important for large graph operations.

### 7.1 Iterator-Based Traversal

**Problem:** `traversal.rs` collects all results into `Vec` before returning. For subgraph extraction with radius=5 on a dense graph, this materializes thousands of nodes.

**Design note:** Uses `async_stream::try_stream!` with backpressure. Results stream lazily.

**File:** `crates/agent-db-graph/src/traversal.rs`

**What to change:**
- Add iterator-based variants of key traversal methods:
```rust
fn bfs_iter<'a>(
    &'a self,
    graph: &'a Graph,
    start: NodeId,
    max_depth: u32,
) -> impl Iterator<Item = (NodeId, u32)> + 'a
```
- Consumers can `.take(N)` for early termination
- Existing Vec-based methods call `.collect()` on iterator variant (backward compat)

**Impact:** Subgraph extraction with `LIMIT 10` stops after finding 10 nodes instead of expanding the entire radius.

**Estimated scope:** ~80 lines added.

---

### 7.2 Push-Down Filters to Storage Layer

**Problem:** Queries like "get all Event nodes updated after timestamp X" load all nodes, deserialize them, then filter in Rust.

**Design note:** `pre_skip` pushed to storage layer to skip I/O for discarded rows.

**File:** `crates/agent-db-graph/src/redb_graph_store.rs`

**What to change:**
- Add filtered scan methods to RedbGraphStore:
```rust
fn scan_nodes_filtered(
    &self,
    bucket: GoalBucketId,
    filter: &NodeFilter,
) -> Result<Vec<GraphNode>, GraphStoreError>

pub enum NodeFilter {
    ByType(NodeType),
    UpdatedAfter(Timestamp),
    MinDegree(u32),
    Combined(Vec<NodeFilter>),
}
```
- For `ByType`: use key prefix scan (NodeType byte is in key)
- For `UpdatedAfter`: scan headers first (cheap), then load matching full nodes
- For `MinDegree`: check header degree field before full load

**Impact:** For "get all Events in last hour" on a partition with 10K nodes but only 50 recent events, this reduces deserialization from 10K to 50.

**Estimated scope:** ~60 lines added.

---

### 7.3 Prefetch Hints for Partition Loading

**Design note:** Uses OS page cache hints (prefetch flag) for range scans.

**File:** `crates/agent-db-graph/src/redb_graph_store.rs`

**What to change:**
- When `ensure_partition_loaded()` detects a cache miss, hint the next likely partitions
- Heuristic: if loading bucket X for agent A, preload bucket 0 (global) since cross-references are likely
- Use `tokio::spawn` for async background preloading (non-blocking)

**Estimated scope:** ~30 lines added.

---

## Implementation Order & Dependencies

```
Phase 1 (Fixes)  ──────────────────────────────────────────────────
  1.1 Fix Dijkstra weights        [no deps]           ██ 1 day
  1.2 Fix query cache              [no deps]           ██ 1 day
  1.3 Fix constrained path         [depends on 1.1]    ███ 1.5 days

Phase 2 (Pathfinding)  ────────────────────────────────────────────
  2.1 Define GraphPath             [no deps]           █ 0.5 day
  2.2 Weighted Dijkstra            [depends on 1.1]    █ 0.5 day
  2.3 A* search                    [depends on 2.2]    ██ 1 day
  2.4 K-shortest paths             [depends on 2.2]    ██ 1 day
  2.5 Bidirectional search         [depends on 2.2]    ██ 1 day
  2.6 find_paths() in RedbStore    [depends on 2.1-2.5]███ 1.5 days

Phase 3 (Caching)  ────────────────────────────────────────────────
  3.1 LRU context similarity       [no deps]           ██ 1 day
  3.2 LRU query cache              [depends on 1.2]    █ 0.5 day
  3.3 Batch store operations       [no deps]           ███ 1.5 days
  3.4 Weighted partition cache     [no deps]           ██ 1 day

Phase 4 (Batching)  ───────────────────────────────────────────────
  4.1 Batch node reads             [no deps]           ██ 1 day
  4.2 Batch consolidation writes   [depends on 3.3]    ██ 1 day
  4.3 Atomic event buffer          [no deps]           █ 0.5 day

Phase 5 (Parallelism)  ────────────────────────────────────────────
  5.1 Enable Rayon                 [no deps]           ██ 1 day
  5.2 Concurrent consolidation     [depends on 5.1]    ███ 1.5 days
  5.3 Parallel maintenance         [depends on 5.1]    █ 0.5 day

Phase 6 (Data Structures)  ────────────────────────────────────────
  6.1 SmallVec adjacency           [no deps]           █ 0.5 day
  6.2 Roaring bitmap indexes       [no deps]           ██ 1 day
  6.3 Arc::try_unwrap pipeline     [no deps]           █ 0.5 day
  6.4 Compact NodeHeader read      [no deps]           █ 0.5 day

Phase 7 (Streaming)  ──────────────────────────────────────────────
  7.1 Iterator-based traversal     [no deps]           ██ 1 day
  7.2 Push-down filters            [no deps]           ██ 1 day
  7.3 Prefetch hints               [no deps]           █ 0.5 day
```

---

## New Dependencies

| Crate | Version | Purpose | Size Impact |
|-------|---------|---------|-------------|
| `smallvec` | 1.x | Stack-allocated small vecs | ~15KB |
| `roaring` | 0.10.x | Compressed bitmap indexes | ~200KB |
| `rayon` | 1.x | Data parallelism (uncomment) | ~300KB |

**Note:** `dashmap 5.5` is already a dependency and can be leveraged for concurrent caches. `lru` or `quick-cache` may be needed if dashmap alone isn't sufficient for LRU semantics.

---

## Files Touched (Summary)

| File | Phases | Type of Change |
|------|--------|----------------|
| `structures.rs` | 2.1 | Add GraphPath, PathType |
| `traversal.rs` | 1.1, 1.2, 1.3, 2.2–2.5, 3.2, 4.1, 7.1 | Major: pathfinding + caching |
| `redb_graph_store.rs` | 2.6, 3.4, 4.1, 7.2, 7.3 | Storage-level pathfinding + filters |
| `graph_store.rs` | 4.1, 6.4 | Trait additions + header optimization |
| `inference.rs` | 3.1 | Cache replacement |
| `stores.rs` | 3.3, 4.2 | Batch operations |
| `consolidation.rs` | 4.2, 5.2 | Batch writes + parallelism |
| `maintenance.rs` | 5.3 | Parallel maintenance tasks |
| `algorithms/parallel.rs` | 5.1 | Enable Rayon |
| `indexing.rs` | 6.2 | Roaring bitmap posting lists |
| `integration.rs` | 6.3 | Arc::try_unwrap optimization |
| `buffer.rs` | 4.3 | Atomic batch operations |
| `compression.rs` | — | No changes needed (already good) |
| `Cargo.toml` | 5.1, 6.1, 6.2 | New dependencies |

---

## Testing Strategy

Each phase should include:

1. **Unit tests** for new algorithms (pathfinding correctness, cache eviction behavior)
2. **Property-based tests** for pathfinding (path cost ≤ brute-force enumeration)
3. **Benchmarks** using `criterion` crate:
   - Before/after for each phase
   - Graph sizes: 100, 1K, 10K, 100K nodes
   - Measure: latency (p50, p95, p99), throughput (ops/sec), memory RSS
4. **Integration tests** for cross-cutting changes (batch operations, parallel consolidation)
5. **Regression tests** to ensure existing behavior isn't broken

---

## Success Metrics

| Metric | Current (estimated) | Target |
|--------|-------------------|--------|
| Shortest path (1K nodes) | N/A (broken weights) | < 1ms |
| Shortest path (100K nodes) | N/A | < 50ms |
| K-shortest paths (K=5, 10K nodes) | N/A (not implemented) | < 20ms |
| Query cache hit rate | 0% (disabled) | > 60% |
| Consolidation pass (100 memories) | ~100 transactions | 1 transaction |
| PageRank (10K nodes) | Sequential only | 4x faster (parallel) |
| Partition load time | Full deserialization | 40% faster (push-down) |
| Memory per low-degree node | 1 Vec alloc + 1 HashMap entry | Stack-only (SmallVec) |
| Context similarity cache spike | Full clear every 10K ops | Smooth LRU eviction |
