# EventGraphDB Graph Layer Improvement Plan

## Reference Sources
- **reference graph engine** (`ref/grapgh/reference graph engine/`) — Temporal graph DB, Rust, adaptive data structures, 35+ algorithms
- **reference implementation** (`ref/reference_impl/`) — Document-graph DB, Rust, hierarchical key encoding, recursive traversal

---

## Phase 1: Quick Wins (High Impact, Low Effort) — COMPLETE

### 1.1 Replace String-Based Type Index with u8 Discriminant — DONE
- **File**: `crates/agent-db-graph/src/structures.rs`
- **Implemented**: `type_index: FxHashMap<u8, HashSet<NodeId>>` using `NodeType::discriminant()` as key
- **Impact**: O(1) lookup via u8 discriminant instead of string hashing

### 1.2 Fix Query Cache Key Generation — DONE
- **File**: `crates/agent-db-graph/src/traversal.rs`
- **Implemented**: `query_cache_key()` computes u64 hash via `DefaultHasher` — no Debug string allocation
- **Impact**: Eliminates allocation on every cache lookup

### 1.3 Auto-Invalidate Cache on Mutations — DONE
- **Files**: `structures.rs`, `traversal.rs`
- **Implemented**: `generation: u64` counter in Graph, incremented by `add_node/add_edge/remove_node`; `QueryCache::check_generation()` auto-clears on mismatch
- **Impact**: Eliminates stale query results

### 1.4 Add Temporal BTree Index on created_at — DONE
- **File**: `crates/agent-db-graph/src/structures.rs`
- **Implemented**: `temporal_index: BTreeMap<Timestamp, SmallVec<[NodeId; 4]>>` with `nodes_in_time_range()` range query
- **Impact**: O(log N + K) time-range queries

### 1.5 Implement Brandes Betweenness Centrality — DONE
- **File**: `crates/agent-db-graph/src/algorithms/centrality.rs`
- **Implemented**: Brandes' algorithm — BFS + dependency accumulation, O(V*E), with normalization
- **Impact**: 1000x faster than naive O(V^3) for large graphs

---

## Phase 2: Adaptive Data Structures (from reference graph engine) — COMPLETE

### 2.1 Adaptive AdjList Enum — DONE
- **File**: `crates/agent-db-graph/src/structures.rs`
- **Implemented**: `AdjList { Empty | One(EdgeId) | Small(SmallVec<[EdgeId; 8]>) | Large(BTreeSet<EdgeId>) }`
- Threshold: `ADJ_LARGE_THRESHOLD = 1024`, auto-downgrade in `retain()`
- Custom serde (flat sequence, backward-compatible), `AdjListIter` enum iterator

### 2.2 String Interning Pool — DONE
- **File**: `crates/agent-db-graph/src/intern.rs`
- **Implemented**: `Interner { pool: FxHashSet<Arc<str>> }` with `intern()`, `remove()`, `len()`
- tool_index, result_index, concept_index now `HashMap<Arc<str>, NodeId>`

### 2.3 Direction-Encoded ReDB Keys — DONE
- **File**: `crates/agent-db-graph/src/redb_graph_store.rs`
- **Implemented**: Forward/reverse prefix keys for single prefix-scan adjacency traversal
- Backward-compatible fallback to legacy adjacency list path

### 2.4 Delta Persistence — DONE
- **File**: `crates/agent-db-graph/src/integration/persistence.rs`
- **Implemented**: `dirty_nodes/dirty_edges/deleted_nodes/deleted_edges` + `adjacency_dirty` tracking
- `persist_graph_delta()` writes only changed items; falls back to full persist at >50% dirty ratio
- `clear_dirty()` after successful persist; `has_pending_changes()` for checking

---

## Phase 3: Graph Algorithms (from reference graph engine) — COMPLETE

### 3.1 True Louvain Community Detection — DONE
- **File**: `crates/agent-db-graph/src/algorithms/louvain.rs`
- **Implemented**: Modularity optimization with iterative phase 1 (local moves) + phase 2 (aggregation)
- Config: resolution, max_iterations, min_improvement, random_seed

### 3.2 Temporal Reachability (Taint Propagation) — DONE
- **File**: `crates/agent-db-graph/src/algorithms/temporal_reachability.rs`
- **Implemented**: BFS with min-heap ordered by arrival time, temporal monotonicity constraint
- `propagate()` single-source, `propagate_multi()` multi-source, `causal_path()` reconstruction
- Config: max_hops, time_start/time_end window, min_edge_weight

### 3.3 Label Propagation Community Detection — DONE
- **File**: `crates/agent-db-graph/src/algorithms/label_propagation.rs`
- **Implemented**: O(V+E) majority-label adoption (Raghavan et al. 2007)
- Config: max_iterations

### 3.4 Triangle Counting & Clustering Coefficients — DONE
- **File**: `crates/agent-db-graph/src/algorithms/centrality.rs`
- **Implemented**: Local + global clustering coefficients

### 3.5 Random Walk Infrastructure — DONE
- **File**: `crates/agent-db-graph/src/algorithms/random_walk.rs`
- **Implemented**: Configurable walk_length, restart_probability, num_walks, weighted selection
- Built-in xoshiro256** PRNG for deterministic seeded walks (no `rand` dependency)
- `personalized_pagerank()` via random walks with restart, `walk_all()` for corpus-wide walks

---

## Phase 4: Query API & Architecture (from reference implementation) — COMPLETE

### 4.1 Direction-Aware Traversal API — DONE
- **Files**: `structures.rs`, `traversal.rs`, `temporal_view.rs`
- **Implemented**: `Direction { Out, In, Both }` + `Depth { Fixed(u32), Range(u32,u32), Unbounded }`
- `GraphQuery::DirectedTraversal`, `Graph::neighbors_directed()`, `Graph::edges_directed()`

### 4.2 Recursive Instructions — DONE
- **File**: `crates/agent-db-graph/src/traversal.rs`
- **Implemented**: `Instruction { Collect, Path { max_paths }, Shortest(NodeId) }`
- `TraversalRequest` with composable `NodeFilterExpr` / `EdgeFilterExpr`
- Dispatcher: `execute_collect()`, `execute_paths()`, `execute_shortest()`

### 4.3 Streaming Query Results — DONE
- **File**: `crates/agent-db-graph/src/traversal.rs`
- **Implemented**: `StreamingQuery<I: Iterator>` with `next_batch()` batched iteration
- `QueryContext` with `is_done()` — respects limit + cancellation for early termination

### 4.4 Sharded Graph Storage — DONE
- **File**: `crates/agent-db-graph/src/graph_store.rs`
- **Implemented**: Goal-bucket partitioning via `GoalBucketId`, per-shard `parking_lot::RwLock<BucketShard>`
- `load_partition()`, `unload_partition()`, `is_partition_loaded()`, `get_partition_stats()`

### 4.5 Rolling Time Windows — DONE
- **File**: `crates/agent-db-graph/src/temporal_view.rs`
- **Implemented**: `GraphAtSnapshot` (frozen at cutoff) + `RollingWindow` (start..end range)
- Convenience: `graph.at(timestamp)`, `graph.rolling(duration_ns)`, `graph.window(start, end)`

---

## Standards
- No TODOs, no stubs — every function fully implemented
- All new code has comprehensive tests
- Brandes algorithm tested against naive O(V^3) for correctness
- Adaptive structures tested at all tier transitions
- Cache invalidation tested for all mutation paths
- 431+ tests pass across workspace
