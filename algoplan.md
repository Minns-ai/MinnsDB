# Wire Up 3 Graph Algorithms into GraphEngine + Server

## Context
Three graph algorithms (Random Walk/PPR, Temporal Reachability, Label Propagation) exist as tested standalone modules but have zero callers. This plan wires them into `GraphEngine` and exposes them via REST endpoints, following the exact patterns used by the existing Louvain and Centrality integrations.

---

## Step 1: Add fields + config to GraphEngine

**File:** `crates/agent-db-graph/src/integration/mod.rs`

- Add `community_algorithm: String` field to `GraphEngineConfig` (after `louvain_interval` line 132)
- Add default `"louvain".to_string()` in `impl Default` (after line 458)
- Add 3 new fields to `GraphEngine` struct (after `centrality` line 367):
  ```
  random_walker: Arc<RandomWalker>
  temporal_reachability: Arc<TemporalReachability>
  label_propagation: Arc<LabelPropagationAlgorithm>
  ```

## Step 2: Initialize in constructor

**File:** `crates/agent-db-graph/src/integration/constructor.rs`

- After line 163 (`let centrality = ...`), add:
  ```rust
  let random_walker = Arc::new(RandomWalker::new());
  let temporal_reachability = Arc::new(TemporalReachability::new());
  let label_propagation = Arc::new(LabelPropagationAlgorithm::new());
  ```
- Add 3 fields to struct literal (after `centrality,` line 440)

**Note on TemporalReachability field vs per-call:** The stored `Arc<TemporalReachability>` is used for default-config calls (`causal_path`). Methods that need caller-specified config (`temporal_reachability_from` with `max_hops`) create a local instance — `TemporalReachability::new()` is zero-alloc, so this is cheap.

## Step 3: Add integration methods

**File:** `crates/agent-db-graph/src/integration/graph_analytics.rs`

Add 4 new `pub async` methods to `impl GraphEngine`:

### 3a. `personalized_pagerank`
```rust
pub async fn personalized_pagerank(
    &self, source: NodeId,
) -> GraphResult<HashMap<NodeId, f64>>
```
- `inference.read()` → `self.random_walker.personalized_pagerank(graph, source)`

### 3b. `temporal_reachability_from`
```rust
pub async fn temporal_reachability_from(
    &self, source: NodeId, max_hops: usize,
) -> GraphResult<TemporalReachabilityResult>
```
- If `max_hops > 0`: create local `TemporalReachability::with_config(...)` with that limit
- If `max_hops == 0`: use `self.temporal_reachability` (default config, unlimited)
- `inference.read()` → `tr.propagate(graph, source)`
- Validate `max_hops <= 1000` to prevent accidental huge traversals

### 3c. `causal_path`
```rust
pub async fn causal_path(
    &self, source: NodeId, target: NodeId,
) -> GraphResult<Option<Vec<NodeId>>>
```
- Uses stored `self.temporal_reachability` (default config)
- `inference.read()` → `self.temporal_reachability.propagate(graph, source)` → `.causal_path(&result, target)`

### 3d. `detect_communities_with_algorithm`
```rust
pub async fn detect_communities_with_algorithm(
    &self, algorithm: Option<&str>,
) -> GraphResult<CommunityDetectionResult>
```
- **Algorithm string normalization:** lowercase + trim, then match:
  - `"louvain"` → Louvain
  - `"label_propagation" | "label-propagation" | "labelprop" | "lp"` → Label Propagation
  - Unknown → return `GraphError::InvalidQuery` with accepted values list
- **Precedence:** explicit param > `config.community_algorithm` > `"louvain"` fallback
- Converts `LabelPropagationResult` → `CommunityDetectionResult`:
  - `modularity: -1.0` (sentinel: "not computed", distinct from valid 0.0)
  - Logs which algorithm was used via `tracing::info!`

Also modify `run_community_detection()` (line 53) to respect `config.community_algorithm` using the same normalization.

## Step 4: Add PPR boost in action suggestions

**File:** `crates/agent-db-graph/src/integration/queries.rs`

In `get_next_action_suggestions` (after centrality re-ranking, line 179):

- Only activate when `last_action_node.is_some()`
- Compute PPR from the last action node (using already-held `inference.read()` lock)
- **Rank-based normalization**: sort PPR scores across the candidate set, map to [0.0, 1.0] rank percentiles — avoids scale incompatibility with success_probability
- Blend: `existing * 0.9 + ppr_rank_score * 0.1`
- Re-sort after blending
- Wrap in `if let Ok(ppr_scores) = ...` to gracefully degrade on errors (no panic if node missing)

**Lock scope:** The read lock is already held for the full method. PPR computation is synchronous on `&Graph` — no additional lock acquisition. If profiling later shows contention, the lock can be narrowed, but this matches the existing pattern.

## Step 5: Add server response/query types

**File:** `server/src/models.rs`

Add after existing `CentralityScoresResponse` (line 436):

### Query types (Deserialize)
```rust
PprQuery { source_node_id: u64, limit: Option<usize>, min_score: Option<f64> }
ReachabilityQuery { source: u64, max_hops: Option<usize>, max_results: Option<usize> }
CausalPathQuery { source: u64, target: u64 }
CommunitiesQuery { algorithm: Option<String> }
```

### Response types (Serialize)
```rust
PprResponse { source_node_id: u64, algorithm: String, scores: Vec<PprNodeScore> }
PprNodeScore { node_id: u64, score: f64 }

ReachabilityResponse { source_node_id: u64, reachable_count: usize, max_depth: usize,
                       edges_traversed: usize, reachable: Vec<ReachabilityNodeResponse> }
ReachabilityNodeResponse { node_id: u64, origin: u64, arrival_time: u64,
                           hops: usize, predecessor: Option<u64> }

CausalPathResponse { source: u64, target: u64, found: bool, path: Option<Vec<u64>> }
```

Also add `algorithm: String` field to existing `CommunitiesResponse` to echo which algorithm was resolved.

## Step 6: Add server handlers

**File:** `server/src/handlers/analytics.rs`

### 6a. `get_ppr` (new)
- `GET /api/ppr?source_node_id=123&limit=50&min_score=0.001`
- Calls `engine.personalized_pagerank(source_node_id)`
- Applies `min_score` filter, sorts descending, caps at `limit` (default 100)
- Includes `tracing::info!` with source, result count, duration

### 6b. `get_reachability` (new)
- `GET /api/reachability?source=123&max_hops=5&max_results=200`
- Calls `engine.temporal_reachability_from(source, max_hops.unwrap_or(0))`
- Caps response at `max_results` (default 500)
- Includes timing trace

### 6c. `get_causal_path` (new)
- `GET /api/causal-path?source=123&target=456`
- Calls `engine.causal_path(source, target)`
- Handle `source == target` → return `path: Some(vec![source])`, `found: true`

### 6d. `get_communities` (modify)
- Accept `Query<CommunitiesQuery>` alongside `State`
- Call `detect_communities_with_algorithm(params.algorithm.as_deref())`
- Include `algorithm` field in response (echo resolved algorithm name)
- On `GraphError::InvalidQuery` → return 400 with accepted values

## Step 7: Register routes

**File:** `server/src/routes.rs`

Add after line 69 (centrality route):
```
.route("/api/ppr", get(handlers::get_ppr))
.route("/api/reachability", get(handlers::get_reachability))
.route("/api/causal-path", get(handlers::get_causal_path))
```

Existing `/api/communities` route stays — handler signature change is backward-compatible.

## Step 8: Add LabelPropagation re-exports

**File:** `crates/agent-db-graph/src/lib.rs`

Add `LabelPropagationAlgorithm, LabelPropagationConfig, LabelPropagationResult` to the `pub use algorithms::{ ... }` block.

## Step 9: Server config env var

**File:** `server/src/config.rs`

After line 61 (`louvain_interval` reading), add:
```rust
config.community_algorithm = env::var("COMMUNITY_ALGORITHM")
    .unwrap_or_else(|_| "louvain".to_string());
info!("  Community algorithm: {}", config.community_algorithm);
```

---

## Verification

1. `cargo test --workspace` — all existing 431+ tests pass
2. `cargo build -p server` — server compiles with new endpoints
3. Endpoint checks:
   - `GET /api/ppr?source_node_id=1` → PPR scores, sorted descending
   - `GET /api/ppr?source_node_id=999999` → empty scores (missing node = error)
   - `GET /api/reachability?source=1&max_hops=3` → reachable nodes with hops/arrival times
   - `GET /api/reachability?source=1&max_hops=0` → unlimited (defined behavior)
   - `GET /api/causal-path?source=1&target=5` → causal path or `found: false`
   - `GET /api/causal-path?source=1&target=1` → `path: [1]`, `found: true`
   - `GET /api/communities?algorithm=label_propagation` → LP communities with `algorithm: "label_propagation"` in response
   - `GET /api/communities?algorithm=invalid` → 400 error with accepted values
   - `GET /api/communities` → works as before (backward-compatible, uses config default)
   - `COMMUNITY_ALGORITHM=label_propagation` env var changes default

## Files Modified (9 total)

| File | Changes |
|------|---------|
| `integration/mod.rs` | +1 config field, +3 struct fields, +1 default |
| `integration/constructor.rs` | +3 Arc initializations, +3 struct literal fields |
| `integration/graph_analytics.rs` | +4 new methods, modify `run_community_detection` |
| `integration/queries.rs` | PPR rank-based boost in `get_next_action_suggestions` |
| `lib.rs` | +3 LabelPropagation re-exports |
| `server/src/models.rs` | +4 query structs, +5 response structs, +1 field to CommunitiesResponse |
| `server/src/handlers/analytics.rs` | +3 new handlers, modify `get_communities` |
| `server/src/routes.rs` | +3 new route lines |
| `server/src/config.rs` | +1 env var reading |
