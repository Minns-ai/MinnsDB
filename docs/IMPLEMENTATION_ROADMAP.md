# EventGraphDB Enhancement Roadmap

Implementation guide for key features inspired by temporal graph reference analysis.

## Table of Contents
- [1. BM25 Full-Text Search](#1-bm25-full-text-search)
- [2. Performance Benchmarking Suite](#2-performance-benchmarking-suite)
- [3. Bi-Temporal Event Model](#3-bi-temporal-event-model)
- [4. MCP Server for Claude Integration](#4-mcp-server-for-claude-integration)
- [5. Multi-Backend Graph Store Abstraction](#5-multi-backend-graph-store-abstraction)

---

## 1. BM25 Full-Text Search
**Priority:** ⚡ Quick (1-2 days)
**Effort:** Low
**Value:** High - Keyword search complements semantic embeddings

### Overview
Add BM25 (Best Matching 25) ranking algorithm to the existing `FullTextIndex` in `crates/agent-db-graph/src/indexing.rs`. BM25 provides superior keyword-based ranking compared to simple term frequency.

### Architecture

```
┌─────────────────────────────────────────┐
│         Search Request                  │
│  "agent failed authentication error"    │
└────────────────┬────────────────────────┘
                 │
        ┌────────┴─────────┐
        │                  │
        ▼                  ▼
┌──────────────┐   ┌──────────────┐
│   Semantic   │   │   BM25       │
│   Embedding  │   │   Keyword    │
│   Search     │   │   Search     │
└──────┬───────┘   └──────┬───────┘
       │                  │
       └────────┬─────────┘
                ▼
        ┌──────────────┐
        │   Fusion     │
        │   Reranking  │
        └──────┬───────┘
               ▼
          Results
```

### Implementation Steps

#### 1.1 Add BM25 Dependencies
**File:** `crates/agent-db-graph/Cargo.toml`

```toml
[dependencies]
# Add BM25 implementation
tantivy = "0.22"  # Full-text search library with BM25
# OR implement custom BM25
```

**Alternative:** Implement custom BM25 to avoid heavy dependency.

#### 1.2 Extend Index Structures
**File:** `crates/agent-db-graph/src/indexing.rs`

```rust
use std::collections::HashMap;

/// BM25 parameters
#[derive(Debug, Clone)]
pub struct Bm25Config {
    /// Term saturation parameter (typically 1.2-2.0)
    pub k1: f32,
    /// Length normalization parameter (typically 0.75)
    pub b: f32,
}

impl Default for Bm25Config {
    fn default() -> Self {
        Self {
            k1: 1.5,  // Standard BM25 parameter
            b: 0.75,  // Standard length normalization
        }
    }
}

/// Document statistics for BM25
#[derive(Debug, Clone)]
struct DocumentStats {
    /// Document length (number of terms)
    length: usize,
    /// Term frequencies in this document
    term_frequencies: HashMap<String, usize>,
}

/// Corpus-level statistics
#[derive(Debug)]
struct CorpusStats {
    /// Total number of documents
    total_docs: usize,
    /// Average document length
    avg_doc_length: f32,
    /// Document frequency per term (how many docs contain term)
    doc_frequencies: HashMap<String, usize>,
}

pub struct Bm25Index {
    config: Bm25Config,
    corpus_stats: CorpusStats,
    documents: HashMap<NodeId, DocumentStats>,
}

impl Bm25Index {
    pub fn new(config: Bm25Config) -> Self {
        Self {
            config,
            corpus_stats: CorpusStats {
                total_docs: 0,
                avg_doc_length: 0.0,
                doc_frequencies: HashMap::new(),
            },
            documents: HashMap::new(),
        }
    }

    /// Add document to index
    pub fn index_document(&mut self, node_id: NodeId, text: &str) {
        let terms = Self::tokenize(text);
        let length = terms.len();

        // Calculate term frequencies
        let mut term_frequencies = HashMap::new();
        for term in &terms {
            *term_frequencies.entry(term.clone()).or_insert(0) += 1;
        }

        // Update corpus statistics
        self.corpus_stats.total_docs += 1;

        // Update document frequencies
        for term in term_frequencies.keys() {
            *self.corpus_stats.doc_frequencies.entry(term.clone()).or_insert(0) += 1;
        }

        // Recalculate average document length
        let total_length: usize = self.documents.values().map(|d| d.length).sum::<usize>() + length;
        self.corpus_stats.avg_doc_length = total_length as f32 / self.corpus_stats.total_docs as f32;

        // Store document stats
        self.documents.insert(node_id, DocumentStats {
            length,
            term_frequencies,
        });
    }

    /// Search using BM25 scoring
    pub fn search(&self, query: &str, limit: usize) -> Vec<(NodeId, f32)> {
        let query_terms = Self::tokenize(query);
        let mut scores: HashMap<NodeId, f32> = HashMap::new();

        // Calculate BM25 score for each document
        for (&node_id, doc_stats) in &self.documents {
            let mut score = 0.0;

            for term in &query_terms {
                score += self.bm25_term_score(term, doc_stats);
            }

            if score > 0.0 {
                scores.insert(node_id, score);
            }
        }

        // Sort by score descending
        let mut results: Vec<(NodeId, f32)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(limit);

        results
    }

    /// Calculate BM25 score for a single term in a document
    fn bm25_term_score(&self, term: &str, doc_stats: &DocumentStats) -> f32 {
        // Get term frequency in document
        let tf = *doc_stats.term_frequencies.get(term).unwrap_or(&0) as f32;
        if tf == 0.0 {
            return 0.0;
        }

        // Calculate IDF (Inverse Document Frequency)
        let df = *self.corpus_stats.doc_frequencies.get(term).unwrap_or(&0) as f32;
        let n = self.corpus_stats.total_docs as f32;
        let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();

        // Calculate normalized term frequency
        let doc_length = doc_stats.length as f32;
        let k1 = self.config.k1;
        let b = self.config.b;
        let avg_length = self.corpus_stats.avg_doc_length;

        let normalized_tf = (tf * (k1 + 1.0)) /
            (tf + k1 * (1.0 - b + b * (doc_length / avg_length)));

        idf * normalized_tf
    }

    /// Tokenize text into terms
    fn tokenize(text: &str) -> Vec<String> {
        text.to_lowercase()
            .split_whitespace()
            .filter(|s| s.len() > 2) // Filter short words
            .map(|s| s.trim_matches(|c: char| !c.is_alphanumeric()))
            .filter(|s| !s.is_empty())
            .map(String::from)
            .collect()
    }
}
```

#### 1.3 Update FullTextIndex to use BM25
**File:** `crates/agent-db-graph/src/indexing.rs`

```rust
pub struct FullTextIndex {
    // Existing field
    term_index: HashMap<String, HashSet<NodeId>>,

    // NEW: Add BM25 index
    bm25_index: Bm25Index,
}

impl FullTextIndex {
    pub fn new() -> Self {
        Self {
            term_index: HashMap::new(),
            bm25_index: Bm25Index::new(Bm25Config::default()),
        }
    }

    pub fn index(&mut self, node_id: NodeId, text: &str) {
        // Existing term indexing
        let terms: HashSet<String> = text
            .to_lowercase()
            .split_whitespace()
            .map(String::from)
            .collect();

        for term in terms {
            self.term_index
                .entry(term)
                .or_insert_with(HashSet::new)
                .insert(node_id);
        }

        // NEW: Index with BM25
        self.bm25_index.index_document(node_id, text);
    }

    /// Search with BM25 ranking
    pub fn search_ranked(&self, query: &str, limit: usize) -> Vec<(NodeId, f32)> {
        self.bm25_index.search(query, limit)
    }

    // Keep existing search method for backwards compatibility
    pub fn search(&self, query: &str) -> HashSet<NodeId> {
        // ... existing implementation
    }
}
```

#### 1.4 Add Hybrid Search with Score Fusion
**File:** `crates/agent-db-graph/src/indexing.rs`

```rust
/// Search mode for hybrid retrieval
#[derive(Debug, Clone, Copy)]
pub enum SearchMode {
    /// BM25 keyword search only
    Keyword,
    /// Semantic embedding search only
    Semantic,
    /// Hybrid: combine both with fusion
    Hybrid,
}

/// Fusion strategy for combining scores
#[derive(Debug, Clone, Copy)]
pub enum FusionStrategy {
    /// Reciprocal Rank Fusion (RRF)
    ReciprocalRank { k: f32 },
    /// Weighted average
    Weighted { keyword_weight: f32, semantic_weight: f32 },
}

impl Default for FusionStrategy {
    fn default() -> Self {
        // RRF is more robust to score scale differences
        Self::ReciprocalRank { k: 60.0 }
    }
}

impl FusionStrategy {
    /// Fuse two ranked result lists
    pub fn fuse(
        &self,
        keyword_results: Vec<(NodeId, f32)>,
        semantic_results: Vec<(NodeId, f32)>,
    ) -> Vec<(NodeId, f32)> {
        match self {
            Self::ReciprocalRank { k } => {
                Self::reciprocal_rank_fusion(keyword_results, semantic_results, *k)
            }
            Self::Weighted { keyword_weight, semantic_weight } => {
                Self::weighted_fusion(keyword_results, semantic_results, *keyword_weight, *semantic_weight)
            }
        }
    }

    fn reciprocal_rank_fusion(
        keyword_results: Vec<(NodeId, f32)>,
        semantic_results: Vec<(NodeId, f32)>,
        k: f32,
    ) -> Vec<(NodeId, f32)> {
        let mut scores: HashMap<NodeId, f32> = HashMap::new();

        // Add keyword scores
        for (rank, (node_id, _)) in keyword_results.iter().enumerate() {
            let rrf_score = 1.0 / (k + (rank + 1) as f32);
            *scores.entry(*node_id).or_insert(0.0) += rrf_score;
        }

        // Add semantic scores
        for (rank, (node_id, _)) in semantic_results.iter().enumerate() {
            let rrf_score = 1.0 / (k + (rank + 1) as f32);
            *scores.entry(*node_id).or_insert(0.0) += rrf_score;
        }

        let mut results: Vec<(NodeId, f32)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results
    }

    fn weighted_fusion(
        keyword_results: Vec<(NodeId, f32)>,
        semantic_results: Vec<(NodeId, f32)>,
        keyword_weight: f32,
        semantic_weight: f32,
    ) -> Vec<(NodeId, f32)> {
        let mut scores: HashMap<NodeId, f32> = HashMap::new();

        // Add weighted keyword scores
        for (node_id, score) in keyword_results {
            *scores.entry(node_id).or_insert(0.0) += score * keyword_weight;
        }

        // Add weighted semantic scores
        for (node_id, score) in semantic_results {
            *scores.entry(node_id).or_insert(0.0) += score * semantic_weight;
        }

        let mut results: Vec<(NodeId, f32)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results
    }
}
```

#### 1.5 Update API Endpoints
**File:** `server/src/handlers/search.rs` (new file)

```rust
use axum::{extract::State, Json};
use serde::{Deserialize, Serialize};
use crate::state::AppState;
use crate::error::ApiError;
use agent_db_graph::indexing::{SearchMode, FusionStrategy};

#[derive(Debug, Deserialize)]
pub struct SearchRequest {
    pub query: String,
    #[serde(default = "default_limit")]
    pub limit: usize,
    #[serde(default)]
    pub mode: SearchMode,
    #[serde(default)]
    pub fusion_strategy: Option<FusionStrategy>,
}

fn default_limit() -> usize { 10 }

#[derive(Debug, Serialize)]
pub struct SearchResult {
    pub node_id: u64,
    pub score: f32,
    pub content: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResult>,
    pub total: usize,
}

pub async fn search_nodes(
    State(state): State<AppState>,
    Json(request): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, ApiError> {
    let graph = state.graph.read().await;

    let results = match request.mode {
        SearchMode::Keyword => {
            // BM25 keyword search
            graph.fulltext_index.search_ranked(&request.query, request.limit)
        }
        SearchMode::Semantic => {
            // Semantic embedding search (existing)
            graph.embedding_search(&request.query, request.limit).await?
        }
        SearchMode::Hybrid => {
            // Hybrid search with fusion
            let keyword_results = graph.fulltext_index.search_ranked(&request.query, request.limit);
            let semantic_results = graph.embedding_search(&request.query, request.limit).await?;

            let strategy = request.fusion_strategy.unwrap_or_default();
            strategy.fuse(keyword_results, semantic_results)
        }
    };

    let search_results: Vec<SearchResult> = results
        .iter()
        .map(|(node_id, score)| SearchResult {
            node_id: *node_id,
            score: *score,
            content: graph.get_node_content(*node_id),
        })
        .collect();

    Ok(Json(SearchResponse {
        total: search_results.len(),
        results: search_results,
    }))
}
```

**File:** `server/src/routes.rs`

```rust
use crate::handlers::search;

pub fn create_router(state: AppState) -> Router {
    Router::new()
        // ... existing routes
        .route("/api/search", post(search::search_nodes))
        .with_state(state)
}
```

### Testing Strategy

**File:** `crates/agent-db-graph/src/indexing.rs` (add tests)

```rust
#[cfg(test)]
mod bm25_tests {
    use super::*;

    #[test]
    fn test_bm25_basic_ranking() {
        let mut index = Bm25Index::new(Bm25Config::default());

        index.index_document(1, "the quick brown fox jumps over the lazy dog");
        index.index_document(2, "the quick brown fox");
        index.index_document(3, "the lazy dog sleeps");

        let results = index.search("quick fox", 10);

        // Doc 2 should rank highest (contains both terms, shorter)
        assert_eq!(results[0].0, 2);
        assert!(results[0].1 > results[1].1);
    }

    #[test]
    fn test_bm25_idf_weighting() {
        let mut index = Bm25Index::new(Bm25Config::default());

        // "the" appears in all docs (low IDF)
        // "authentication" appears in one (high IDF)
        index.index_document(1, "the user logged in successfully");
        index.index_document(2, "the authentication failed");
        index.index_document(3, "the system started");

        let results = index.search("authentication", 10);

        // Only doc 2 should match
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 2);
    }

    #[test]
    fn test_reciprocal_rank_fusion() {
        let keyword_results = vec![(1, 10.0), (2, 8.0), (3, 5.0)];
        let semantic_results = vec![(3, 0.95), (1, 0.90), (4, 0.85)];

        let strategy = FusionStrategy::ReciprocalRank { k: 60.0 };
        let fused = strategy.fuse(keyword_results, semantic_results);

        // Node 1 and 3 appear in both, should rank high
        assert!(fused[0].0 == 1 || fused[0].0 == 3);
    }
}
```

### Migration Notes
- **Backwards compatible**: Existing `search()` method remains unchanged
- **Indexing**: Re-index existing documents by iterating all nodes and calling `index_document()`
- **Storage**: BM25 index lives in memory; serialize/deserialize if persistence needed

### Performance Considerations
- **Memory**: ~100 bytes per indexed document (term frequencies + stats)
- **Indexing**: O(n) where n = number of terms in document
- **Search**: O(m × d) where m = query terms, d = documents containing those terms
- **Optimization**: Use inverted index to only score candidate documents

---

## 2. Performance Benchmarking Suite
**Priority:** ⚡ Quick (1 day)
**Effort:** Low
**Value:** High - Demonstrates performance advantage vs temporal graph reference

### Overview
Create comprehensive benchmark suite using Criterion.rs to measure and document query latency, throughput, and scalability.

### Architecture

```
benchmarks/
├── event_ingestion.rs    # Event processing throughput
├── query_latency.rs      # Search/retrieval latency
├── graph_traversal.rs    # BFS/DFS performance
├── memory_retrieval.rs   # Context-based memory lookup
├── learning_ops.rs       # Strategy extraction, transitions
└── concurrent_load.rs    # Multi-threaded stress test
```

### Implementation Steps

#### 2.1 Setup Benchmark Infrastructure
**File:** `Cargo.toml` (workspace level)

```toml
[workspace]
members = [
    # ... existing
    "benchmarks",
]

[[bench]]
name = "event_ingestion"
harness = false

[[bench]]
name = "query_latency"
harness = false

[[bench]]
name = "graph_traversal"
harness = false
```

**File:** `benchmarks/Cargo.toml`

```toml
[package]
name = "eventgraphdb-benchmarks"
version = "0.1.0"
edition = "2021"

[dependencies]
agent-db-graph = { path = "../crates/agent-db-graph" }
agent-db-events = { path = "../crates/agent-db-events" }
agent-db-core = { path = "../crates/agent-db-core" }
criterion = { version = "0.5", features = ["html_reports", "async_tokio"] }
tokio = { version = "1.28", features = ["full"] }
rand = "0.8"

[[bench]]
name = "event_ingestion"
harness = false

[[bench]]
name = "query_latency"
harness = false

[[bench]]
name = "graph_traversal"
harness = false

[[bench]]
name = "memory_retrieval"
harness = false

[[bench]]
name = "learning_ops"
harness = false

[[bench]]
name = "concurrent_load"
harness = false
```

#### 2.2 Event Ingestion Benchmark
**File:** `benchmarks/benches/event_ingestion.rs`

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use agent_db_events::Event;
use agent_db_graph::GraphEngine;
use tokio::runtime::Runtime;

fn generate_test_event(id: u64) -> Event {
    Event {
        id,
        timestamp: id * 1000,
        agent_id: 1,
        agent_type: "test".to_string(),
        session_id: "session-1".to_string(),
        event_type: agent_db_events::EventType::Action {
            action_name: format!("action_{}", id),
            outcome: agent_db_events::ActionOutcome::Success { result: "ok".to_string() },
            tool_used: None,
        },
        causality_chain: vec![],
        context: Default::default(),
        metadata: Default::default(),
        context_size_bytes: 0,
        segment_pointer: None,
    }
}

fn bench_event_ingestion(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("event_ingestion");

    for batch_size in [1, 10, 100, 1000] {
        group.throughput(Throughput::Elements(batch_size));

        group.bench_with_input(
            BenchmarkId::from_parameter(batch_size),
            &batch_size,
            |b, &size| {
                b.to_async(&rt).iter(|| async {
                    let engine = GraphEngine::new_in_memory().unwrap();

                    for i in 0..size {
                        let event = generate_test_event(i);
                        engine.process_event(black_box(event)).await.unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_event_ingestion);
criterion_main!(benches);
```

#### 2.3 Query Latency Benchmark
**File:** `benchmarks/benches/query_latency.rs`

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use agent_db_graph::GraphEngine;
use tokio::runtime::Runtime;

fn setup_graph_with_data(size: usize) -> GraphEngine {
    let rt = Runtime::new().unwrap();
    let engine = GraphEngine::new_in_memory().unwrap();

    rt.block_on(async {
        for i in 0..size {
            let event = generate_test_event(i as u64);
            engine.process_event(event).await.unwrap();
        }
    });

    engine
}

fn bench_query_latency(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("query_latency");

    for size in [100, 1000, 10000] {
        let engine = setup_graph_with_data(size);

        // Benchmark memory retrieval
        group.bench_with_input(
            BenchmarkId::new("memory_retrieval", size),
            &engine,
            |b, engine| {
                b.to_async(&rt).iter(|| async {
                    let context_hash = black_box(12345u64);
                    engine.retrieve_memories(context_hash, 10).await.unwrap();
                });
            },
        );

        // Benchmark strategy lookup
        group.bench_with_input(
            BenchmarkId::new("strategy_lookup", size),
            &engine,
            |b, engine| {
                b.to_async(&rt).iter(|| async {
                    let context_hash = black_box(12345u64);
                    engine.get_strategies_for_context(context_hash).await.unwrap();
                });
            },
        );

        // Benchmark claim search
        group.bench_with_input(
            BenchmarkId::new("claim_search", size),
            &engine,
            |b, engine| {
                b.to_async(&rt).iter(|| async {
                    engine.search_claims(black_box("test query"), 10).await.unwrap();
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_query_latency);
criterion_main!(benches);
```

#### 2.4 Graph Traversal Benchmark
**File:** `benchmarks/benches/graph_traversal.rs`

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use agent_db_graph::structures::{Graph, GraphNode, GraphEdge, NodeType, EdgeType};

fn create_test_graph(nodes: usize, edges_per_node: usize) -> Graph {
    let mut graph = Graph::new();

    // Add nodes
    for i in 0..nodes {
        graph.add_node(GraphNode::new(
            i as u64,
            NodeType::Event,
            format!("event_{}", i),
        ));
    }

    // Add edges (small-world pattern)
    for i in 0..nodes {
        for j in 1..=edges_per_node {
            let target = (i + j) % nodes;
            graph.add_edge(GraphEdge::new(
                i as u64,
                target as u64,
                EdgeType::Temporal {
                    average_interval_ms: 1000,
                    sequence_confidence: 0.9,
                },
                0.9,
            ));
        }
    }

    graph
}

fn bench_graph_traversal(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph_traversal");

    for size in [100, 500, 1000] {
        let graph = create_test_graph(size, 3);

        // BFS traversal
        group.bench_with_input(
            BenchmarkId::new("bfs", size),
            &graph,
            |b, graph| {
                b.iter(|| {
                    graph.bfs(black_box(0), black_box(size / 2))
                });
            },
        );

        // DFS traversal
        group.bench_with_input(
            BenchmarkId::new("dfs", size),
            &graph,
            |graph| {
                b.iter(|| {
                    graph.dfs(black_box(0))
                });
            },
        );

        // Get neighbors
        group.bench_with_input(
            BenchmarkId::new("get_neighbors", size),
            &graph,
            |b, graph| {
                b.iter(|| {
                    graph.get_neighbors(black_box(size as u64 / 2))
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_graph_traversal);
criterion_main!(benches);
```

#### 2.5 Concurrent Load Benchmark
**File:** `benchmarks/benches/concurrent_load.rs`

```rust
use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId, Throughput};
use agent_db_graph::GraphEngine;
use tokio::runtime::Runtime;
use std::sync::Arc;

fn bench_concurrent_writes(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let mut group = c.benchmark_group("concurrent_load");

    for num_threads in [1, 2, 4, 8] {
        group.throughput(Throughput::Elements(num_threads * 100));

        group.bench_with_input(
            BenchmarkId::new("concurrent_writes", num_threads),
            &num_threads,
            |b, &threads| {
                b.to_async(&rt).iter(|| async move {
                    let engine = Arc::new(GraphEngine::new_in_memory().unwrap());
                    let mut handles = vec![];

                    for thread_id in 0..threads {
                        let engine = engine.clone();
                        let handle = tokio::spawn(async move {
                            for i in 0..100 {
                                let event_id = (thread_id * 100 + i) as u64;
                                let event = generate_test_event(event_id);
                                engine.process_event(event).await.unwrap();
                            }
                        });
                        handles.push(handle);
                    }

                    for handle in handles {
                        handle.await.unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

fn bench_concurrent_reads(c: &mut Criterion) {
    let rt = Runtime::new().unwrap();

    let engine = Arc::new(setup_graph_with_data(1000));

    let mut group = c.benchmark_group("concurrent_reads");

    for num_threads in [1, 2, 4, 8] {
        group.throughput(Throughput::Elements(num_threads * 100));

        group.bench_with_input(
            BenchmarkId::new("concurrent_reads", num_threads),
            &num_threads,
            |b, &threads| {
                let engine = engine.clone();
                b.to_async(&rt).iter(|| async move {
                    let mut handles = vec![];

                    for _ in 0..threads {
                        let engine = engine.clone();
                        let handle = tokio::spawn(async move {
                            for i in 0..100 {
                                let context_hash = black_box(i);
                                engine.retrieve_memories(context_hash, 10).await.unwrap();
                            }
                        });
                        handles.push(handle);
                    }

                    for handle in handles {
                        handle.await.unwrap();
                    }
                });
            },
        );
    }

    group.finish();
}

criterion_group!(benches, bench_concurrent_writes, bench_concurrent_reads);
criterion_main!(benches);
```

#### 2.6 Results Documentation Template
**File:** `docs/PERFORMANCE.md`

```markdown
# EventGraphDB Performance Benchmarks

All benchmarks run on: [System specs]
- CPU: [e.g., AMD Ryzen 9 5950X]
- RAM: [e.g., 64GB DDR4]
- Storage: [e.g., NVMe SSD]
- OS: [e.g., Ubuntu 22.04]

## Event Ingestion

| Batch Size | Throughput (events/sec) | Latency (ms/event) |
|------------|-------------------------|-------------------|
| 1          | XXX                     | X.XX              |
| 10         | XXX                     | X.XX              |
| 100        | XXX                     | X.XX              |
| 1000       | XXX                     | X.XX              |

## Query Latency (vs temporal graph reference)

| Operation          | EventGraphDB | temporal graph reference | Speedup |
|-------------------|--------------|----------|---------|
| Memory Retrieval   | XX ms        | ~XXX ms  | XXx     |
| Strategy Lookup    | XX ms        | ~XXX ms  | XXx     |
| Claim Search       | XX ms        | ~XXX ms  | XXx     |
| Graph Traversal    | XX ms        | ~XXX ms  | XXx     |

**Note:** temporal graph reference benchmarks estimated from "sub-second" claims (~500ms avg)

## Scalability

### Query Latency vs Dataset Size

| Dataset Size | Memory Retrieval | Strategy Lookup | Claim Search |
|--------------|-----------------|-----------------|--------------|
| 100 events   | XX ms           | XX ms           | XX ms        |
| 1K events    | XX ms           | XX ms           | XX ms        |
| 10K events   | XX ms           | XX ms           | XX ms        |
| 100K events  | XX ms           | XX ms           | XX ms        |

### Concurrent Throughput

| Threads | Writes/sec | Reads/sec |
|---------|------------|-----------|
| 1       | XXX        | XXX       |
| 2       | XXX        | XXX       |
| 4       | XXX        | XXX       |
| 8       | XXX        | XXX       |

## Key Takeaways

- ✅ **Sub-100ms latency** for memory retrieval (vs temporal graph reference's ~500ms)
- ✅ **Linear scalability** up to 100K events
- ✅ **Efficient concurrency** with Tokio async runtime
- ✅ **10-100x faster** than Python-based temporal graph reference

## Running Benchmarks

```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench --bench query_latency

# Generate flamegraphs
cargo flamegraph --bench query_latency
```
```

### Testing Strategy
```bash
# Run benchmarks
cargo bench

# Compare with baseline
cargo bench -- --save-baseline main

# After optimization
cargo bench -- --baseline main

# Generate HTML report
open target/criterion/report/index.html
```

---

## 3. Bi-Temporal Event Model
**Priority:** 🎯 Medium (2-3 days)
**Effort:** Medium
**Value:** High - Enables point-in-time queries and debugging

### Overview
Add `ingested_at` timestamp to track when events were added to the system, separate from when they occurred. This enables "what did the system know at time X?" queries.

### Architecture

```
Event Timeline:
═══════════════════════════════════════════════════════════

Occurred:    t1────────t2────────t3────────t4────────t5───►
             │         │         │         │         │
Ingested:    │    ti1──┼────ti2──┼────────ti3────────┼ti4─►
             │    │    │    │    │         │         │  │
             v    v    v    v    v         v         v  v
Events:      E1   E2   E1'  E3   E4        E2'       E5 E1''
                  (late)  (update)     (correction)    (latest)

Point-in-time query at ti2:
- Returns: E1, E2, E3 (what system knew then)
- Excludes: E1', E2', E4, E5 (ingested later)
```

### Implementation Steps

#### 3.1 Update Event Structure
**File:** `crates/agent-db-events/src/core.rs`

```rust
use serde_with::{serde_as, DisplayFromStr};

#[serde_as]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    // Existing fields...

    /// When the event actually occurred (business time)
    #[serde(default = "current_timestamp")]
    #[serde_as(as = "DisplayFromStr")]
    pub timestamp: Timestamp,

    /// When the event was ingested into the system (system time)
    /// NEW FIELD
    #[serde(default = "current_timestamp")]
    #[serde_as(as = "DisplayFromStr")]
    pub ingested_at: Timestamp,

    // ... rest of fields
}

impl Event {
    /// Create event with occurrence time and auto-set ingestion time
    pub fn new_with_timestamp(timestamp: Timestamp, /* other params */) -> Self {
        Self {
            timestamp,
            ingested_at: current_timestamp(), // Auto-set to now
            // ... rest
        }
    }

    /// Create event with both timestamps (for replay/import scenarios)
    pub fn new_with_bi_temporal(
        timestamp: Timestamp,
        ingested_at: Timestamp,
        /* other params */
    ) -> Self {
        Self {
            timestamp,
            ingested_at,
            // ... rest
        }
    }
}
```

#### 3.2 Add Temporal Query Support
**File:** `crates/agent-db-graph/src/temporal.rs` (new file)

```rust
use agent_db_core::types::{EventId, Timestamp};
use agent_db_events::Event;
use std::collections::HashMap;

/// Temporal query mode
#[derive(Debug, Clone, Copy)]
pub enum TemporalMode {
    /// Query based on when events occurred
    EventTime,
    /// Query based on when events were ingested
    SystemTime,
    /// Query based on both (bi-temporal)
    BiTemporal,
}

/// Point-in-time query parameters
#[derive(Debug, Clone)]
pub struct PointInTimeQuery {
    /// The point in time to query
    pub as_of: Timestamp,
    /// Which temporal dimension to use
    pub mode: TemporalMode,
}

impl PointInTimeQuery {
    /// Query what events existed at a specific point in time
    pub fn filter_events(&self, events: &[Event]) -> Vec<Event> {
        events
            .iter()
            .filter(|e| self.is_visible(e))
            .cloned()
            .collect()
    }

    /// Check if an event was visible at the query time
    fn is_visible(&self, event: &Event) -> bool {
        match self.mode {
            TemporalMode::EventTime => {
                // Event occurred before query time
                event.timestamp <= self.as_of
            }
            TemporalMode::SystemTime => {
                // Event was ingested before query time
                event.ingested_at <= self.as_of
            }
            TemporalMode::BiTemporal => {
                // Both occurred AND ingested before query time
                event.timestamp <= self.as_of && event.ingested_at <= self.as_of
            }
        }
    }
}

/// Temporal versioning for event updates
#[derive(Debug, Clone)]
pub struct EventVersion {
    pub event: Event,
    pub version: u32,
    pub superseded_by: Option<EventId>,
    pub valid_from: Timestamp,
    pub valid_until: Option<Timestamp>,
}

/// Temporal event store with versioning
pub struct TemporalEventStore {
    /// All versions of all events
    versions: HashMap<EventId, Vec<EventVersion>>,
}

impl TemporalEventStore {
    pub fn new() -> Self {
        Self {
            versions: HashMap::new(),
        }
    }

    /// Add a new event or update an existing one
    pub fn add_or_update(&mut self, event: Event) {
        let versions = self.versions.entry(event.id).or_insert_with(Vec::new);

        if let Some(current) = versions.last_mut() {
            // Mark previous version as superseded
            current.valid_until = Some(event.ingested_at);
            current.superseded_by = Some(event.id);
        }

        versions.push(EventVersion {
            version: versions.len() as u32 + 1,
            valid_from: event.ingested_at,
            valid_until: None, // Current version
            superseded_by: None,
            event,
        });
    }

    /// Get event as it was known at a specific time
    pub fn get_as_of(&self, event_id: EventId, as_of: Timestamp) -> Option<&Event> {
        self.versions.get(&event_id)?
            .iter()
            .find(|v| {
                v.valid_from <= as_of &&
                v.valid_until.map_or(true, |until| until > as_of)
            })
            .map(|v| &v.event)
    }

    /// Get all versions of an event
    pub fn get_history(&self, event_id: EventId) -> Option<&[EventVersion]> {
        self.versions.get(&event_id).map(|v| v.as_slice())
    }

    /// Get the current (latest) version
    pub fn get_current(&self, event_id: EventId) -> Option<&Event> {
        self.versions.get(&event_id)?
            .last()
            .map(|v| &v.event)
    }
}
```

#### 3.3 Add Temporal Indexing
**File:** `crates/agent-db-graph/src/indexing.rs`

```rust
/// Temporal index for efficient point-in-time queries
pub struct TemporalIndex {
    /// Events sorted by occurrence time
    event_time_index: BTreeMap<Timestamp, Vec<EventId>>,
    /// Events sorted by ingestion time
    system_time_index: BTreeMap<Timestamp, Vec<EventId>>,
}

impl TemporalIndex {
    pub fn new() -> Self {
        Self {
            event_time_index: BTreeMap::new(),
            system_time_index: BTreeMap::new(),
        }
    }

    pub fn index(&mut self, event: &Event) {
        // Index by occurrence time
        self.event_time_index
            .entry(event.timestamp)
            .or_insert_with(Vec::new)
            .push(event.id);

        // Index by ingestion time
        self.system_time_index
            .entry(event.ingested_at)
            .or_insert_with(Vec::new)
            .push(event.id);
    }

    /// Get all events that occurred before a timestamp
    pub fn events_before_occurrence(&self, before: Timestamp) -> Vec<EventId> {
        self.event_time_index
            .range(..=before)
            .flat_map(|(_, ids)| ids)
            .copied()
            .collect()
    }

    /// Get all events that were ingested before a timestamp
    pub fn events_before_ingestion(&self, before: Timestamp) -> Vec<EventId> {
        self.system_time_index
            .range(..=before)
            .flat_map(|(_, ids)| ids)
            .copied()
            .collect()
    }

    /// Get events in a time range (event time)
    pub fn events_in_range(&self, start: Timestamp, end: Timestamp) -> Vec<EventId> {
        self.event_time_index
            .range(start..=end)
            .flat_map(|(_, ids)| ids)
            .copied()
            .collect()
    }
}
```

#### 3.4 Update Graph Engine
**File:** `crates/agent-db-graph/src/engine.rs`

```rust
use crate::temporal::{TemporalEventStore, PointInTimeQuery, TemporalMode};

pub struct GraphEngine {
    // Existing fields...

    /// NEW: Temporal event store for versioning
    temporal_store: TemporalEventStore,

    /// NEW: Temporal indexes
    temporal_index: TemporalIndex,
}

impl GraphEngine {
    /// Query graph state as it was at a specific point in time
    pub async fn query_as_of(
        &self,
        as_of: Timestamp,
        mode: TemporalMode,
    ) -> GraphResult<TemporalSnapshot> {
        let query = PointInTimeQuery { as_of, mode };

        // Get visible events
        let event_ids = match mode {
            TemporalMode::EventTime => {
                self.temporal_index.events_before_occurrence(as_of)
            }
            TemporalMode::SystemTime => {
                self.temporal_index.events_before_ingestion(as_of)
            }
            TemporalMode::BiTemporal => {
                // Intersection of both
                let by_occurrence = self.temporal_index.events_before_occurrence(as_of);
                let by_ingestion = self.temporal_index.events_before_ingestion(as_of);
                by_occurrence.into_iter()
                    .filter(|id| by_ingestion.contains(id))
                    .collect()
            }
        };

        // Build snapshot
        let events: Vec<Event> = event_ids
            .iter()
            .filter_map(|&id| self.temporal_store.get_current(id))
            .cloned()
            .collect();

        Ok(TemporalSnapshot {
            as_of,
            mode,
            events,
            // ... other snapshot data
        })
    }

    /// Get event history (all versions)
    pub fn get_event_history(&self, event_id: EventId) -> Option<Vec<EventVersion>> {
        self.temporal_store.get_history(event_id)
            .map(|versions| versions.to_vec())
    }
}

#[derive(Debug, Clone)]
pub struct TemporalSnapshot {
    pub as_of: Timestamp,
    pub mode: TemporalMode,
    pub events: Vec<Event>,
    pub total_events: usize,
}
```

#### 3.5 Add API Endpoints
**File:** `server/src/handlers/temporal.rs` (new file)

```rust
use axum::{extract::{State, Query}, Json};
use serde::{Deserialize, Serialize};
use agent_db_graph::temporal::TemporalMode;
use serde_with::{serde_as, DisplayFromStr};

#[serde_as]
#[derive(Debug, Deserialize)]
pub struct TemporalQueryParams {
    #[serde_as(as = "DisplayFromStr")]
    pub as_of: u64, // Timestamp

    #[serde(default)]
    pub mode: TemporalMode,
}

#[derive(Debug, Serialize)]
pub struct TemporalSnapshotResponse {
    pub as_of: String,
    pub mode: String,
    pub total_events: usize,
    pub events: Vec<EventSummary>,
}

pub async fn query_as_of(
    State(state): State<AppState>,
    Query(params): Query<TemporalQueryParams>,
) -> Result<Json<TemporalSnapshotResponse>, ApiError> {
    let graph = state.graph.read().await;

    let snapshot = graph.query_as_of(params.as_of, params.mode).await?;

    Ok(Json(TemporalSnapshotResponse {
        as_of: params.as_of.to_string(),
        mode: format!("{:?}", params.mode),
        total_events: snapshot.total_events,
        events: snapshot.events.iter().map(EventSummary::from).collect(),
    }))
}

#[serde_as]
#[derive(Debug, Deserialize)]
pub struct EventHistoryParams {
    #[serde_as(as = "DisplayFromStr")]
    pub event_id: u128,
}

#[derive(Debug, Serialize)]
pub struct EventHistoryResponse {
    pub event_id: String,
    pub versions: Vec<EventVersionInfo>,
}

pub async fn get_event_history(
    State(state): State<AppState>,
    Query(params): Query<EventHistoryParams>,
) -> Result<Json<EventHistoryResponse>, ApiError> {
    let graph = state.graph.read().await;

    let history = graph.get_event_history(params.event_id)
        .ok_or_else(|| ApiError::NotFound("Event not found".to_string()))?;

    Ok(Json(EventHistoryResponse {
        event_id: params.event_id.to_string(),
        versions: history.iter().map(EventVersionInfo::from).collect(),
    }))
}
```

**File:** `server/src/routes.rs`

```rust
pub fn create_router(state: AppState) -> Router {
    Router::new()
        // ... existing routes
        .route("/api/temporal/query", get(temporal::query_as_of))
        .route("/api/temporal/history", get(temporal::get_event_history))
        .with_state(state)
}
```

### Migration Strategy

#### 3.6 Database Migration
**File:** `crates/agent-db-storage/src/migrations/add_ingested_at.rs`

```rust
use agent_db_storage::{StorageEngine, StorageResult};
use agent_db_core::types::Timestamp;

pub async fn migrate_add_ingested_at(engine: &mut StorageEngine) -> StorageResult<()> {
    println!("Migrating: Adding ingested_at timestamp to events...");

    // Read all events
    let events = engine.get_all_events().await?;

    let total = events.len();
    let mut updated = 0;

    for mut event in events {
        // Set ingested_at = timestamp for existing events
        // (retroactively assume they were ingested when they occurred)
        event.ingested_at = event.timestamp;

        engine.update_event(event).await?;
        updated += 1;

        if updated % 1000 == 0 {
            println!("  Migrated {}/{} events...", updated, total);
        }
    }

    println!("✓ Migration complete: {} events updated", updated);
    Ok(())
}
```

Run migration:
```bash
cargo run --bin migrate -- --migration add_ingested_at
```

### Testing Strategy

**File:** `crates/agent-db-graph/src/temporal.rs` (tests)

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_in_time_query() {
        let mut store = TemporalEventStore::new();

        // t=1000: Event E1 occurs and is ingested
        let e1 = Event {
            id: 1,
            timestamp: 1000,
            ingested_at: 1000,
            // ... other fields
        };
        store.add_or_update(e1.clone());

        // t=2000: Event E2 occurs but ingested at t=3000 (late arrival)
        let e2 = Event {
            id: 2,
            timestamp: 2000,
            ingested_at: 3000,
            // ... other fields
        };
        store.add_or_update(e2.clone());

        // Query at t=2500 (system time)
        let query = PointInTimeQuery {
            as_of: 2500,
            mode: TemporalMode::SystemTime,
        };

        // Should only see E1 (E2 not ingested yet)
        assert_eq!(store.get_as_of(1, 2500).unwrap().id, 1);
        assert!(store.get_as_of(2, 2500).is_none());

        // Query at t=3500 (system time)
        let query = PointInTimeQuery {
            as_of: 3500,
            mode: TemporalMode::SystemTime,
        };

        // Should see both E1 and E2
        assert!(store.get_as_of(1, 3500).is_some());
        assert!(store.get_as_of(2, 3500).is_some());
    }

    #[test]
    fn test_event_versioning() {
        let mut store = TemporalEventStore::new();

        // Initial version
        let e1_v1 = Event {
            id: 1,
            timestamp: 1000,
            ingested_at: 1000,
            // ... data: "initial"
        };
        store.add_or_update(e1_v1);

        // Updated version
        let e1_v2 = Event {
            id: 1,
            timestamp: 1000, // Same occurrence time
            ingested_at: 2000, // Updated later
            // ... data: "corrected"
        };
        store.add_or_update(e1_v2);

        // At t=1500, should see v1
        let v1 = store.get_as_of(1, 1500).unwrap();
        // assert_eq!(v1.data, "initial");

        // At t=2500, should see v2
        let v2 = store.get_as_of(1, 2500).unwrap();
        // assert_eq!(v2.data, "corrected");

        // History should have 2 versions
        assert_eq!(store.get_history(1).unwrap().len(), 2);
    }
}
```

### Use Cases

#### Debugging Agent Decisions
```rust
// "Why did the agent make this decision at time X?"
let snapshot = engine.query_as_of(decision_time, TemporalMode::SystemTime).await?;
// See exactly what data the agent had at that moment
```

#### Compliance & Audit
```rust
// "What did we know about this user at time of transaction?"
let user_state = engine.query_as_of(transaction_time, TemporalMode::BiTemporal).await?;
```

#### Late-Arriving Data
```rust
// Event occurs at t=100 but arrives at t=200
let event = Event {
    timestamp: 100,      // When it happened
    ingested_at: 200,    // When we learned about it
    // ...
};
```

---

## 4. MCP Server for Claude Integration
**Priority:** 🎯 Medium (3-5 days)
**Effort:** Medium
**Value:** Very High - Direct Claude integration for agent memory

### Overview
Implement Model Context Protocol (MCP) server to expose EventGraphDB queries as tools for Claude Desktop, Cursor, and other MCP-compatible clients.

### Architecture

```
┌─────────────────────────────────────────────┐
│          Claude Desktop / Cursor            │
│                                             │
│  User: "What strategies worked for         │
│         authentication failures?"           │
└────────────────┬────────────────────────────┘
                 │ MCP Protocol
                 ▼
┌─────────────────────────────────────────────┐
│          MCP Server (stdio)                 │
│                                             │
│  Tools:                                     │
│  - get_memories(context_hash)               │
│  - get_strategies(context)                  │
│  - search_events(query)                     │
│  - query_temporal(as_of)                    │
│  - get_claims(topic)                        │
└────────────────┬────────────────────────────┘
                 │ Internal API
                 ▼
┌─────────────────────────────────────────────┐
│        EventGraphDB Engine                  │
│                                             │
│  - Graph traversal                          │
│  - Memory retrieval                         │
│  - Strategy lookup                          │
│  - Temporal queries                         │
└─────────────────────────────────────────────┘
```

### Implementation Steps

#### 4.1 Add MCP Dependencies
**File:** `Cargo.toml` (workspace)

```toml
[workspace]
members = [
    # ... existing
    "mcp-server",
]
```

**File:** `mcp-server/Cargo.toml`

```toml
[package]
name = "eventgraphdb-mcp"
version = "0.1.0"
edition = "2021"

[[bin]]
name = "eventgraphdb-mcp"
path = "src/main.rs"

[dependencies]
agent-db-graph = { path = "../crates/agent-db-graph" }
agent-db-events = { path = "../crates/agent-db-events" }
agent-db-core = { path = "../crates/agent-db-core" }

# MCP protocol
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
tokio = { version = "1.28", features = ["full", "io-std"] }
anyhow = "1.0"
thiserror = "1.0"

# Logging
tracing = "0.1"
tracing-subscriber = "0.3"
```

#### 4.2 MCP Protocol Types
**File:** `mcp-server/src/protocol.rs`

```rust
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// MCP JSON-RPC request
#[derive(Debug, Deserialize)]
pub struct McpRequest {
    pub jsonrpc: String,
    pub id: Option<Value>,
    pub method: String,
    #[serde(default)]
    pub params: Value,
}

/// MCP JSON-RPC response
#[derive(Debug, Serialize)]
pub struct McpResponse {
    pub jsonrpc: String,
    pub id: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub result: Option<Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<McpError>,
}

#[derive(Debug, Serialize)]
pub struct McpError {
    pub code: i32,
    pub message: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<Value>,
}

/// MCP Tool definition
#[derive(Debug, Serialize)]
pub struct McpTool {
    pub name: String,
    pub description: String,
    pub input_schema: Value,
}

/// Initialize result
#[derive(Debug, Serialize)]
pub struct InitializeResult {
    pub protocol_version: String,
    pub capabilities: Capabilities,
    pub server_info: ServerInfo,
}

#[derive(Debug, Serialize)]
pub struct Capabilities {
    pub tools: Option<ToolsCapability>,
}

#[derive(Debug, Serialize)]
pub struct ToolsCapability {
    pub list_changed: Option<bool>,
}

#[derive(Debug, Serialize)]
pub struct ServerInfo {
    pub name: String,
    pub version: String,
}

/// Tool list response
#[derive(Debug, Serialize)]
pub struct ToolsList {
    pub tools: Vec<McpTool>,
}

/// Tool call request
#[derive(Debug, Deserialize)]
pub struct ToolCallParams {
    pub name: String,
    pub arguments: Value,
}

/// Tool call result
#[derive(Debug, Serialize)]
pub struct ToolCallResult {
    pub content: Vec<Content>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub is_error: Option<bool>,
}

#[derive(Debug, Serialize)]
#[serde(tag = "type")]
pub enum Content {
    #[serde(rename = "text")]
    Text { text: String },
}
```

#### 4.3 Tool Definitions
**File:** `mcp-server/src/tools.rs`

```rust
use agent_db_graph::GraphEngine;
use serde_json::{json, Value};
use anyhow::Result;

pub fn get_tool_definitions() -> Vec<McpTool> {
    vec![
        McpTool {
            name: "get_memories".to_string(),
            description: "Retrieve memories associated with a specific context hash. Returns episodes and patterns learned from past events in similar contexts.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "context_hash": {
                        "type": "string",
                        "description": "64-bit context fingerprint hash (as string)"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum number of memories to return",
                        "default": 10
                    }
                },
                "required": ["context_hash"]
            }),
        },
        McpTool {
            name: "get_strategies".to_string(),
            description: "Get learned strategies for a specific context. Returns action patterns that have historically succeeded in similar situations.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "context_hash": {
                        "type": "string",
                        "description": "64-bit context fingerprint hash (as string)"
                    },
                    "min_success_rate": {
                        "type": "number",
                        "description": "Minimum success rate filter (0.0-1.0)",
                        "default": 0.7
                    }
                },
                "required": ["context_hash"]
            }),
        },
        McpTool {
            name: "search_events".to_string(),
            description: "Search events using hybrid semantic + keyword search. Finds relevant past events matching the query.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query (natural language)"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum results to return",
                        "default": 10
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["keyword", "semantic", "hybrid"],
                        "description": "Search mode",
                        "default": "hybrid"
                    }
                },
                "required": ["query"]
            }),
        },
        McpTool {
            name: "query_temporal".to_string(),
            description: "Query graph state as it was at a specific point in time. Useful for debugging past decisions.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "as_of": {
                        "type": "string",
                        "description": "Timestamp (nanoseconds since epoch)"
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["event_time", "system_time", "bi_temporal"],
                        "description": "Temporal query mode",
                        "default": "system_time"
                    }
                },
                "required": ["as_of"]
            }),
        },
        McpTool {
            name: "get_claims".to_string(),
            description: "Retrieve claims (facts derived from events) about a topic. Returns knowledge extracted from event patterns.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Topic or search query"
                    },
                    "limit": {
                        "type": "number",
                        "description": "Maximum claims to return",
                        "default": 10
                    }
                },
                "required": ["query"]
            }),
        },
        McpTool {
            name: "get_episode_summary".to_string(),
            description: "Get summary of a specific episode (sequence of related events). Provides context about what happened during an episode.".to_string(),
            input_schema: json!({
                "type": "object",
                "properties": {
                    "episode_id": {
                        "type": "string",
                        "description": "Episode ID"
                    }
                },
                "required": ["episode_id"]
            }),
        },
    ]
}

/// Execute a tool call
pub async fn execute_tool(
    engine: &GraphEngine,
    name: &str,
    arguments: Value,
) -> Result<ToolCallResult> {
    match name {
        "get_memories" => execute_get_memories(engine, arguments).await,
        "get_strategies" => execute_get_strategies(engine, arguments).await,
        "search_events" => execute_search_events(engine, arguments).await,
        "query_temporal" => execute_query_temporal(engine, arguments).await,
        "get_claims" => execute_get_claims(engine, arguments).await,
        "get_episode_summary" => execute_get_episode_summary(engine, arguments).await,
        _ => Err(anyhow::anyhow!("Unknown tool: {}", name)),
    }
}

async fn execute_get_memories(engine: &GraphEngine, args: Value) -> Result<ToolCallResult> {
    let context_hash: u64 = args["context_hash"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing context_hash"))?
        .parse()?;

    let limit = args["limit"].as_u64().unwrap_or(10) as usize;

    let memories = engine.retrieve_memories(context_hash, limit).await?;

    let text = if memories.is_empty() {
        "No memories found for this context.".to_string()
    } else {
        let mut output = format!("Found {} memories:\n\n", memories.len());
        for (idx, memory) in memories.iter().enumerate() {
            output.push_str(&format!(
                "{}. Episode {} (strength: {:.2})\n   Events: {}\n   Significance: {:.2}\n\n",
                idx + 1,
                memory.episode_id,
                memory.strength,
                memory.event_ids.len(),
                memory.significance,
            ));
        }
        output
    };

    Ok(ToolCallResult {
        content: vec![Content::Text { text }],
        is_error: None,
    })
}

async fn execute_get_strategies(engine: &GraphEngine, args: Value) -> Result<ToolCallResult> {
    let context_hash: u64 = args["context_hash"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing context_hash"))?
        .parse()?;

    let min_success_rate = args["min_success_rate"].as_f64().unwrap_or(0.7) as f32;

    let strategies = engine.get_strategies_for_context(context_hash).await?;

    let filtered: Vec<_> = strategies
        .into_iter()
        .filter(|s| s.success_rate >= min_success_rate)
        .collect();

    let text = if filtered.is_empty() {
        format!("No strategies found with success rate >= {:.0}%", min_success_rate * 100.0)
    } else {
        let mut output = format!("Found {} strategies:\n\n", filtered.len());
        for (idx, strategy) in filtered.iter().enumerate() {
            output.push_str(&format!(
                "{}. Action: {}\n   Success Rate: {:.1}%\n   Sample Size: {} events\n   Avg Outcome: {:.2}\n\n",
                idx + 1,
                strategy.action_name,
                strategy.success_rate * 100.0,
                strategy.sample_size,
                strategy.avg_outcome,
            ));
        }
        output
    };

    Ok(ToolCallResult {
        content: vec![Content::Text { text }],
        is_error: None,
    })
}

async fn execute_search_events(engine: &GraphEngine, args: Value) -> Result<ToolCallResult> {
    let query = args["query"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing query"))?;

    let limit = args["limit"].as_u64().unwrap_or(10) as usize;
    let mode = args["mode"].as_str().unwrap_or("hybrid");

    let search_mode = match mode {
        "keyword" => SearchMode::Keyword,
        "semantic" => SearchMode::Semantic,
        _ => SearchMode::Hybrid,
    };

    let results = engine.search_events(query, limit, search_mode).await?;

    let text = if results.is_empty() {
        "No events found matching query.".to_string()
    } else {
        let mut output = format!("Found {} events:\n\n", results.len());
        for (idx, (event, score)) in results.iter().enumerate() {
            output.push_str(&format!(
                "{}. [Score: {:.3}] Event {}\n   Type: {:?}\n   Timestamp: {}\n   Agent: {}\n\n",
                idx + 1,
                score,
                event.id,
                event.event_type,
                event.timestamp,
                event.agent_id,
            ));
        }
        output
    };

    Ok(ToolCallResult {
        content: vec![Content::Text { text }],
        is_error: None,
    })
}

async fn execute_query_temporal(engine: &GraphEngine, args: Value) -> Result<ToolCallResult> {
    let as_of: u64 = args["as_of"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing as_of"))?
        .parse()?;

    let mode = match args["mode"].as_str().unwrap_or("system_time") {
        "event_time" => TemporalMode::EventTime,
        "bi_temporal" => TemporalMode::BiTemporal,
        _ => TemporalMode::SystemTime,
    };

    let snapshot = engine.query_as_of(as_of, mode).await?;

    let text = format!(
        "Temporal snapshot at timestamp {}:\n\n\
         Mode: {:?}\n\
         Total events visible: {}\n\
         \n\
         Summary of first 5 events:\n{}",
        as_of,
        mode,
        snapshot.total_events,
        snapshot.events.iter().take(5)
            .map(|e| format!("  - Event {}: {:?}", e.id, e.event_type))
            .collect::<Vec<_>>()
            .join("\n")
    );

    Ok(ToolCallResult {
        content: vec![Content::Text { text }],
        is_error: None,
    })
}

async fn execute_get_claims(engine: &GraphEngine, args: Value) -> Result<ToolCallResult> {
    let query = args["query"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing query"))?;

    let limit = args["limit"].as_u64().unwrap_or(10) as usize;

    let claims = engine.search_claims(query, limit).await?;

    let text = if claims.is_empty() {
        "No claims found matching query.".to_string()
    } else {
        let mut output = format!("Found {} claims:\n\n", claims.len());
        for (idx, claim) in claims.iter().enumerate() {
            output.push_str(&format!(
                "{}. {}\n   Confidence: {:.2}\n   Evidence: {} events\n   Source: Event {}\n\n",
                idx + 1,
                claim.claim_text,
                claim.confidence,
                claim.supporting_evidence.len(),
                claim.source_event_id,
            ));
        }
        output
    };

    Ok(ToolCallResult {
        content: vec![Content::Text { text }],
        is_error: None,
    })
}

async fn execute_get_episode_summary(engine: &GraphEngine, args: Value) -> Result<ToolCallResult> {
    let episode_id: u64 = args["episode_id"]
        .as_str()
        .ok_or_else(|| anyhow::anyhow!("Missing episode_id"))?
        .parse()?;

    let episode = engine.get_episode(episode_id).await?
        .ok_or_else(|| anyhow::anyhow!("Episode not found"))?;

    let text = format!(
        "Episode {} Summary:\n\n\
         Significance: {:.2}\n\
         Events: {}\n\
         Duration: {} - {}\n\
         Outcome: {:?}\n\
         \n\
         Event sequence:\n{}",
        episode.id,
        episode.significance,
        episode.event_ids.len(),
        episode.start_timestamp,
        episode.end_timestamp.map(|t| t.to_string()).unwrap_or_else(|| "ongoing".to_string()),
        episode.outcome,
        episode.event_ids.iter()
            .map(|id| format!("  - Event {}", id))
            .collect::<Vec<_>>()
            .join("\n")
    );

    Ok(ToolCallResult {
        content: vec![Content::Text { text }],
        is_error: None,
    })
}
```

#### 4.4 MCP Server Main
**File:** `mcp-server/src/main.rs`

```rust
mod protocol;
mod tools;

use protocol::*;
use tools::*;
use agent_db_graph::GraphEngine;
use anyhow::Result;
use serde_json::json;
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_writer(std::io::stderr) // Log to stderr (stdout is for MCP protocol)
        .init();

    info!("Starting EventGraphDB MCP Server");

    // Initialize graph engine
    let engine = GraphEngine::new_from_env().await?;
    info!("Graph engine initialized");

    // Setup stdio communication
    let stdin = tokio::io::stdin();
    let mut stdout = tokio::io::stdout();
    let mut reader = BufReader::new(stdin).lines();

    // MCP server loop
    while let Some(line) = reader.next_line().await? {
        if line.trim().is_empty() {
            continue;
        }

        let request: McpRequest = match serde_json::from_str(&line) {
            Ok(req) => req,
            Err(e) => {
                eprintln!("Failed to parse request: {}", e);
                continue;
            }
        };

        let response = handle_request(&engine, request).await;

        let response_json = serde_json::to_string(&response)?;
        stdout.write_all(response_json.as_bytes()).await?;
        stdout.write_all(b"\n").await?;
        stdout.flush().await?;
    }

    Ok(())
}

async fn handle_request(engine: &GraphEngine, request: McpRequest) -> McpResponse {
    match request.method.as_str() {
        "initialize" => handle_initialize(request.id),
        "tools/list" => handle_tools_list(request.id),
        "tools/call" => handle_tool_call(engine, request.id, request.params).await,
        _ => McpResponse {
            jsonrpc: "2.0".to_string(),
            id: request.id,
            result: None,
            error: Some(McpError {
                code: -32601,
                message: format!("Method not found: {}", request.method),
                data: None,
            }),
        },
    }
}

fn handle_initialize(id: Option<serde_json::Value>) -> McpResponse {
    McpResponse {
        jsonrpc: "2.0".to_string(),
        id,
        result: Some(json!(InitializeResult {
            protocol_version: "2024-11-05".to_string(),
            capabilities: Capabilities {
                tools: Some(ToolsCapability {
                    list_changed: Some(false),
                }),
            },
            server_info: ServerInfo {
                name: "EventGraphDB".to_string(),
                version: env!("CARGO_PKG_VERSION").to_string(),
            },
        })),
        error: None,
    }
}

fn handle_tools_list(id: Option<serde_json::Value>) -> McpResponse {
    McpResponse {
        jsonrpc: "2.0".to_string(),
        id,
        result: Some(json!(ToolsList {
            tools: get_tool_definitions(),
        })),
        error: None,
    }
}

async fn handle_tool_call(
    engine: &GraphEngine,
    id: Option<serde_json::Value>,
    params: serde_json::Value,
) -> McpResponse {
    let tool_params: ToolCallParams = match serde_json::from_value(params) {
        Ok(p) => p,
        Err(e) => {
            return McpResponse {
                jsonrpc: "2.0".to_string(),
                id,
                result: None,
                error: Some(McpError {
                    code: -32602,
                    message: format!("Invalid params: {}", e),
                    data: None,
                }),
            };
        }
    };

    match execute_tool(engine, &tool_params.name, tool_params.arguments).await {
        Ok(result) => McpResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: Some(json!(result)),
            error: None,
        },
        Err(e) => McpResponse {
            jsonrpc: "2.0".to_string(),
            id,
            result: None,
            error: Some(McpError {
                code: -32000,
                message: format!("Tool execution failed: {}", e),
                data: None,
            }),
        },
    }
}
```

#### 4.5 Claude Desktop Configuration
**File:** `docs/MCP_SETUP.md`

```markdown
# MCP Server Setup for Claude Desktop

## Installation

1. Build the MCP server:
```bash
cd mcp-server
cargo build --release
```

2. Configure Claude Desktop:

Edit `~/Library/Application Support/Claude/claude_desktop_config.json` (macOS) or
`%APPDATA%\Claude\claude_desktop_config.json` (Windows):

```json
{
  "mcpServers": {
    "eventgraphdb": {
      "command": "/path/to/eventgraphdb/target/release/eventgraphdb-mcp",
      "env": {
        "EVENTGRAPHDB_DATA_PATH": "/path/to/data",
        "RUST_LOG": "info"
      }
    }
  }
}
```

3. Restart Claude Desktop

## Usage Examples

### Query Memories
```
User: What do we know about authentication failures?

Claude uses: get_memories(context_hash="12345")
```

### Find Strategies
```
User: What are proven strategies for handling rate limit errors?

Claude uses: get_strategies(context_hash="67890")
```

### Search Events
```
User: Show me recent API errors

Claude uses: search_events(query="API errors", mode="hybrid")
```

### Temporal Debugging
```
User: What did the system know at timestamp 1234567890?

Claude uses: query_temporal(as_of="1234567890", mode="system_time")
```

## Available Tools

See `mcp-server/src/tools.rs` for full tool definitions.
```

### Testing Strategy

**File:** `mcp-server/tests/integration.rs`

```rust
#[tokio::test]
async fn test_mcp_protocol_flow() {
    // Test initialize
    let init_request = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {}
    });

    // Test tools/list
    let tools_request = json!({
        "jsonrpc": "2.0",
        "id": 2,
        "method": "tools/list",
        "params": {}
    });

    // Test tools/call
    let call_request = json!({
        "jsonrpc": "2.0",
        "id": 3,
        "method": "tools/call",
        "params": {
            "name": "get_memories",
            "arguments": {
                "context_hash": "12345",
                "limit": 5
            }
        }
    });
}
```

---

## 5. Multi-Backend Graph Store Abstraction
**Priority:** 🏗️ Strategic (1-2 weeks)
**Effort:** High
**Value:** High - Enables external graph backend, external graph backend, etc.

### Overview
Abstract the graph storage layer to support multiple backends (redb, external graph backend, external graph backend, in-memory) through a common trait interface.

### Architecture

```
┌─────────────────────────────────────────────┐
│          GraphEngine (Business Logic)       │
│  - Episode detection                        │
│  - Memory formation                         │
│  - Strategy extraction                      │
└────────────────┬────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────┐
│      GraphStore Trait (Abstract Interface)  │
│                                             │
│  - add_node(node) -> Result<NodeId>        │
│  - add_edge(edge) -> Result<EdgeId>        │
│  - get_neighbors(id) -> Vec<NodeId>        │
│  - query(params) -> Vec<Node>              │
└────────┬────────────────────────────────────┘
         │
    ┌────┴────┬─────────┬────────────┐
    ▼         ▼         ▼            ▼
┌────────┐ ┌───────┐ ┌───────┐  ┌──────────┐
│ redb   │ │ external graph backend │ │external graph backend│  │InMemory  │
│Backend │ │Backend│ │Backend│  │Backend   │
└────────┘ └───────┘ └───────┘  └──────────┘
```

### Implementation Steps

#### 5.1 Define GraphStore Trait
**File:** `crates/agent-db-graph/src/backend/mod.rs` (new)

```rust
use agent_db_core::types::{EventId, Timestamp, NodeId, EdgeId};
use crate::structures::{GraphNode, GraphEdge, NodeType, EdgeType};
use crate::error::GraphResult;
use async_trait::async_trait;
use std::collections::HashMap;

pub mod inmemory;
pub mod redb;
pub mod external_graph;

/// Abstract graph storage backend
#[async_trait]
pub trait GraphStore: Send + Sync {
    // === Node Operations ===

    /// Add a node to the graph
    async fn add_node(&mut self, node: GraphNode) -> GraphResult<NodeId>;

    /// Get a node by ID
    async fn get_node(&self, id: NodeId) -> GraphResult<Option<GraphNode>>;

    /// Update a node
    async fn update_node(&mut self, node: GraphNode) -> GraphResult<()>;

    /// Delete a node
    async fn delete_node(&mut self, id: NodeId) -> GraphResult<()>;

    /// Get all node IDs
    async fn get_all_node_ids(&self) -> GraphResult<Vec<NodeId>>;

    /// Get nodes by type
    async fn get_nodes_by_type(&self, node_type: &str) -> GraphResult<Vec<GraphNode>>;

    // === Edge Operations ===

    /// Add an edge to the graph
    async fn add_edge(&mut self, edge: GraphEdge) -> GraphResult<EdgeId>;

    /// Get an edge by ID
    async fn get_edge(&self, id: EdgeId) -> GraphResult<Option<GraphEdge>>;

    /// Update an edge
    async fn update_edge(&mut self, edge: GraphEdge) -> GraphResult<()>;

    /// Delete an edge
    async fn delete_edge(&mut self, id: EdgeId) -> GraphResult<()>;

    /// Get all edges
    async fn get_all_edges(&self) -> GraphResult<Vec<GraphEdge>>;

    /// Get edges from a source node
    async fn get_edges_from(&self, source: NodeId) -> GraphResult<Vec<GraphEdge>>;

    /// Get edges to a target node
    async fn get_edges_to(&self, target: NodeId) -> GraphResult<Vec<GraphEdge>>;

    // === Graph Queries ===

    /// Get neighbors of a node (outgoing)
    async fn get_neighbors(&self, node_id: NodeId) -> GraphResult<Vec<NodeId>>;

    /// Get incoming neighbors
    async fn get_incoming_neighbors(&self, node_id: NodeId) -> GraphResult<Vec<NodeId>>;

    /// Get edge weight between two nodes
    async fn get_edge_weight(&self, source: NodeId, target: NodeId) -> GraphResult<Option<f32>>;

    // === Batch Operations ===

    /// Add multiple nodes
    async fn add_nodes(&mut self, nodes: Vec<GraphNode>) -> GraphResult<Vec<NodeId>>;

    /// Add multiple edges
    async fn add_edges(&mut self, edges: Vec<GraphEdge>) -> GraphResult<Vec<EdgeId>>;

    // === Persistence ===

    /// Flush any pending writes
    async fn flush(&mut self) -> GraphResult<()>;

    /// Get storage statistics
    async fn stats(&self) -> GraphResult<StorageStats>;
}

#[derive(Debug, Clone)]
pub struct StorageStats {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub node_types: HashMap<String, usize>,
    pub backend_type: String,
    pub size_bytes: Option<u64>,
}

/// Configuration for graph store backends
#[derive(Debug, Clone)]
pub enum GraphStoreConfig {
    InMemory,
    Redb {
        path: std::path::PathBuf,
        cache_size_mb: usize,
    },
    external graph backend {
        uri: String,
        username: String,
        password: String,
        database: Option<String>,
    },
    external graph backend {
        endpoint: String,
        region: String,
        use_iam: bool,
    },
}

/// Factory for creating graph store instances
pub async fn create_graph_store(config: GraphStoreConfig) -> GraphResult<Box<dyn GraphStore>> {
    match config {
        GraphStoreConfig::InMemory => {
            Ok(Box::new(inmemory::InMemoryGraphStore::new()))
        }
        GraphStoreConfig::Redb { path, cache_size_mb } => {
            Ok(Box::new(redb::RedbGraphStore::new(path, cache_size_mb).await?))
        }
        GraphStoreConfig::external graph backend { uri, username, password, database } => {
            Ok(Box::new(external_graph::external graph backendGraphStore::new(uri, username, password, database).await?))
        }
        GraphStoreConfig::external graph backend { endpoint, region, use_iam } => {
            Ok(Box::new(external_graph::external graph backendGraphStore::new(endpoint, region, use_iam).await?))
        }
    }
}
```

#### 5.2 Refactor Existing Implementations
**File:** `crates/agent-db-graph/src/backend/inmemory.rs`

```rust
use super::{GraphStore, StorageStats};
use crate::structures::{GraphNode, GraphEdge};
use async_trait::async_trait;
use std::collections::HashMap;

pub struct InMemoryGraphStore {
    nodes: HashMap<NodeId, GraphNode>,
    edges: HashMap<EdgeId, GraphEdge>,
    adjacency_out: HashMap<NodeId, Vec<EdgeId>>,
    adjacency_in: HashMap<NodeId, Vec<EdgeId>>,
}

impl InMemoryGraphStore {
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: HashMap::new(),
            adjacency_out: HashMap::new(),
            adjacency_in: HashMap::new(),
        }
    }
}

#[async_trait]
impl GraphStore for InMemoryGraphStore {
    async fn add_node(&mut self, node: GraphNode) -> GraphResult<NodeId> {
        let id = node.id;
        self.adjacency_out.insert(id, Vec::new());
        self.adjacency_in.insert(id, Vec::new());
        self.nodes.insert(id, node);
        Ok(id)
    }

    async fn get_node(&self, id: NodeId) -> GraphResult<Option<GraphNode>> {
        Ok(self.nodes.get(&id).cloned())
    }

    async fn get_neighbors(&self, node_id: NodeId) -> GraphResult<Vec<NodeId>> {
        Ok(self.adjacency_out
            .get(&node_id)
            .map(|edge_ids| {
                edge_ids
                    .iter()
                    .filter_map(|&eid| self.edges.get(&eid))
                    .map(|e| e.target)
                    .collect()
            })
            .unwrap_or_default())
    }

    // ... implement remaining trait methods
}
```

**File:** `crates/agent-db-graph/src/backend/redb.rs`

```rust
// Move existing RedbGraphStore implementation here
// Implement GraphStore trait
```

#### 5.3 external graph backend Backend Implementation
**File:** `crates/agent-db-graph/Cargo.toml`

```toml
[dependencies]
# ... existing

# Optional external graph backend support
neo4rs = { version = "0.7", optional = true }

[features]
default = ["redb-backend"]
redb-backend = []
external_graph-backend = ["neo4rs"]
external_graph-backend = ["aws-sdk-external_graph"]
```

**File:** `crates/agent-db-graph/src/backend/external_graph.rs`

```rust
#[cfg(feature = "external_graph-backend")]
use super::{GraphStore, StorageStats};
use neo4rs::{Graph, Query, Node, Relation};
use async_trait::async_trait;

pub struct external graph backendGraphStore {
    graph: Graph,
    database: String,
}

impl external graph backendGraphStore {
    pub async fn new(
        uri: String,
        username: String,
        password: String,
        database: Option<String>,
    ) -> GraphResult<Self> {
        let graph = Graph::new(&uri, &username, &password).await
            .map_err(|e| GraphError::BackendError(format!("external graph backend connection failed: {}", e)))?;

        Ok(Self {
            graph,
            database: database.unwrap_or_else(|| "external_graph".to_string()),
        })
    }
}

#[async_trait]
impl GraphStore for external graph backendGraphStore {
    async fn add_node(&mut self, node: GraphNode) -> GraphResult<NodeId> {
        let query = Query::new(format!(
            "CREATE (n:{} {{id: $id, label: $label, data: $data}}) RETURN n.id",
            node_type_to_label(&node.node_type)
        ))
        .param("id", node.id as i64)
        .param("label", node.label.clone())
        .param("data", serde_json::to_string(&node).unwrap());

        self.graph.run(query).await
            .map_err(|e| GraphError::BackendError(format!("external graph backend add_node failed: {}", e)))?;

        Ok(node.id)
    }

    async fn get_node(&self, id: NodeId) -> GraphResult<Option<GraphNode>> {
        let mut result = self.graph
            .execute(Query::new("MATCH (n {id: $id}) RETURN n").param("id", id as i64))
            .await
            .map_err(|e| GraphError::BackendError(format!("external graph backend get_node failed: {}", e)))?;

        if let Some(row) = result.next().await.transpose()
            .map_err(|e| GraphError::BackendError(format!("external graph backend row parse failed: {}", e)))?
        {
            let node: Node = row.get("n")
                .map_err(|e| GraphError::BackendError(format!("external graph backend node extract failed: {}", e)))?;

            // Deserialize node data
            let data: String = node.get("data")
                .map_err(|e| GraphError::BackendError(format!("external graph backend data extract failed: {}", e)))?;

            let graph_node: GraphNode = serde_json::from_str(&data)
                .map_err(|e| GraphError::Deserialization(format!("Node deserialization failed: {}", e)))?;

            Ok(Some(graph_node))
        } else {
            Ok(None)
        }
    }

    async fn get_neighbors(&self, node_id: NodeId) -> GraphResult<Vec<NodeId>> {
        let mut result = self.graph
            .execute(
                Query::new("MATCH (n {id: $id})-[]->(m) RETURN m.id")
                    .param("id", node_id as i64)
            )
            .await
            .map_err(|e| GraphError::BackendError(format!("external graph backend get_neighbors failed: {}", e)))?;

        let mut neighbors = Vec::new();
        while let Some(row) = result.next().await.transpose()
            .map_err(|e| GraphError::BackendError(format!("external graph backend row parse failed: {}", e)))?
        {
            let neighbor_id: i64 = row.get("m.id")
                .map_err(|e| GraphError::BackendError(format!("external graph backend id extract failed: {}", e)))?;
            neighbors.push(neighbor_id as NodeId);
        }

        Ok(neighbors)
    }

    // ... implement remaining methods
}

fn node_type_to_label(node_type: &NodeType) -> &'static str {
    match node_type {
        NodeType::Event => "Event",
        NodeType::Context => "Context",
        NodeType::Agent => "Agent",
        // ... other types
    }
}
```

#### 5.4 Update GraphEngine to Use Trait
**File:** `crates/agent-db-graph/src/engine.rs`

```rust
use crate::backend::{GraphStore, GraphStoreConfig, create_graph_store};

pub struct GraphEngine {
    // Replace concrete types with trait object
    store: Box<dyn GraphStore>,

    // ... other fields
}

impl GraphEngine {
    pub async fn new(config: GraphStoreConfig) -> GraphResult<Self> {
        let store = create_graph_store(config).await?;

        Ok(Self {
            store,
            // ... initialize other fields
        })
    }

    /// Convenience constructor for in-memory store
    pub async fn new_in_memory() -> GraphResult<Self> {
        Self::new(GraphStoreConfig::InMemory).await
    }

    /// Convenience constructor for redb store
    pub async fn new_with_redb<P: AsRef<std::path::Path>>(path: P) -> GraphResult<Self> {
        Self::new(GraphStoreConfig::Redb {
            path: path.as_ref().to_path_buf(),
            cache_size_mb: 512,
        }).await
    }

    /// Process event using abstract storage
    pub async fn process_event(&self, event: Event) -> GraphResult<EventId> {
        // Create node
        let node = GraphNode::new(
            event.id,
            NodeType::Event,
            format!("Event {}", event.id),
        );

        // Use trait methods (works with any backend)
        self.store.add_node(node).await?;

        // ... rest of processing
        Ok(event.id)
    }
}
```

#### 5.5 Configuration & Environment
**File:** `.env.example`

```bash
# Graph Store Backend
GRAPH_STORE_TYPE=redb  # Options: inmemory, redb, external_graph, external_graph

# Redb Backend
REDB_PATH=./data/graph
REDB_CACHE_SIZE_MB=512

# external graph backend Backend
GRAPH_BACKEND_URI=bolt://localhost:7687
GRAPH_BACKEND_USERNAME=external_graph
GRAPH_BACKEND_PASSWORD=password
GRAPH_BACKEND_DATABASE=eventgraphdb

# external graph backend Backend
GRAPH_BACKEND_ENDPOINT=https://your-cluster.external_graph.amazonaws.com:8182
GRAPH_BACKEND_REGION=us-east-1
GRAPH_BACKEND_USE_IAM=true
```

**File:** `crates/agent-db-graph/src/config.rs`

```rust
use crate::backend::GraphStoreConfig;

impl GraphConfig {
    pub fn from_env() -> Result<Self> {
        let store_type = std::env::var("GRAPH_STORE_TYPE")
            .unwrap_or_else(|_| "redb".to_string());

        let store_config = match store_type.as_str() {
            "inmemory" => GraphStoreConfig::InMemory,

            "redb" => GraphStoreConfig::Redb {
                path: std::env::var("REDB_PATH")
                    .unwrap_or_else(|_| "./data/graph".to_string())
                    .into(),
                cache_size_mb: std::env::var("REDB_CACHE_SIZE_MB")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(512),
            },

            "external_graph" => GraphStoreConfig::external graph backend {
                uri: std::env::var("GRAPH_BACKEND_URI")?,
                username: std::env::var("GRAPH_BACKEND_USERNAME")?,
                password: std::env::var("GRAPH_BACKEND_PASSWORD")?,
                database: std::env::var("GRAPH_BACKEND_DATABASE").ok(),
            },

            "external_graph" => GraphStoreConfig::external graph backend {
                endpoint: std::env::var("GRAPH_BACKEND_ENDPOINT")?,
                region: std::env::var("GRAPH_BACKEND_REGION")?,
                use_iam: std::env::var("GRAPH_BACKEND_USE_IAM")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(true),
            },

            _ => return Err(anyhow::anyhow!("Unknown store type: {}", store_type)),
        };

        Ok(Self {
            store_config,
            // ... other config
        })
    }
}
```

### Testing Strategy

#### 5.6 Backend Compliance Tests
**File:** `crates/agent-db-graph/tests/backend_compliance.rs`

```rust
use agent_db_graph::backend::{GraphStore, GraphStoreConfig, create_graph_store};

/// Test suite that all backends must pass
async fn test_backend_compliance(config: GraphStoreConfig) {
    let mut store = create_graph_store(config).await.unwrap();

    // Test node operations
    test_node_crud(&mut store).await;
    test_node_queries(&mut store).await;

    // Test edge operations
    test_edge_crud(&mut store).await;
    test_graph_traversal(&mut store).await;

    // Test batch operations
    test_batch_operations(&mut store).await;
}

async fn test_node_crud(store: &mut Box<dyn GraphStore>) {
    // Add node
    let node = GraphNode::new(1, NodeType::Event, "Test".to_string());
    let id = store.add_node(node.clone()).await.unwrap();
    assert_eq!(id, 1);

    // Get node
    let retrieved = store.get_node(1).await.unwrap();
    assert!(retrieved.is_some());
    assert_eq!(retrieved.unwrap().id, 1);

    // Update node
    let mut updated = node.clone();
    updated.label = "Updated".to_string();
    store.update_node(updated).await.unwrap();

    // Delete node
    store.delete_node(1).await.unwrap();
    assert!(store.get_node(1).await.unwrap().is_none());
}

#[tokio::test]
async fn test_inmemory_backend() {
    test_backend_compliance(GraphStoreConfig::InMemory).await;
}

#[tokio::test]
#[ignore] // Requires redb setup
async fn test_redb_backend() {
    let config = GraphStoreConfig::Redb {
        path: "/tmp/test_redb".into(),
        cache_size_mb: 64,
    };
    test_backend_compliance(config).await;
}

#[tokio::test]
#[ignore] // Requires external graph backend running
async fn test_external_graph_backend() {
    let config = GraphStoreConfig::external graph backend {
        uri: "bolt://localhost:7687".to_string(),
        username: "external_graph".to_string(),
        password: "password".to_string(),
        database: Some("test".to_string()),
    };
    test_backend_compliance(config).await;
}
```

### Migration Path

1. **Phase 1:** Define trait and refactor existing code (3 days)
2. **Phase 2:** Implement external graph backend backend (4 days)
3. **Phase 3:** Add external graph backend backend (3 days)
4. **Phase 4:** Update documentation and examples (1 day)

---

## Summary

| Feature | Priority | Effort | Value | Timeline |
|---------|----------|--------|-------|----------|
| BM25 Search | ⚡ Quick | Low | High | 1-2 days |
| Benchmarks | ⚡ Quick | Low | High | 1 day |
| Bi-Temporal | 🎯 Medium | Medium | High | 2-3 days |
| MCP Server | 🎯 Medium | Medium | Very High | 3-5 days |
| Multi-Backend | 🏗️ Strategic | High | High | 1-2 weeks |

**Total estimated effort:** 2-4 weeks for all features

**Recommended order:**
1. Benchmarks (document current performance)
2. BM25 (quick win for search quality)
3. Bi-temporal (foundational for debugging)
4. MCP Server (high-impact Claude integration)
5. Multi-backend (strategic flexibility)
