# EventGraphDB Graph v2: Deep Investigation of Reference Graph Systems

## Research Date: 2026-02-27

This document consolidates deep research on 6 reference graph systems, temporal graph
techniques from academia, and a full audit of EventGraphDB's current capabilities —
producing a ranked list of improvements to make our temporal graph system better.

---

## Part 1: Systems Investigated

### 1.1 Microsoft GraphRAG

**Repository**: https://github.com/microsoft/graphrag
**Paper**: "From Local to Global" (arXiv 2404.16130)

#### Architecture
- Python pipeline that transforms unstructured text into a structured knowledge graph
- Produces 6 artifact types: Document, TextUnit, Entity, Relationship, Covariate, Community Report
- LLM caching layer makes indexing idempotent and crash-resilient
- Factory pattern for 7 extensible subsystems (LLM, input, cache, logging, storage, vectors, workflows)

#### Indexing Pipeline (6 phases)
1. **Compose TextUnits**: Chunk documents (default 1200 tokens)
2. **Document Processing**: Bidirectional linkages between Documents and TextUnits
3. **Graph Extraction** (3 parallel processes):
   - Entity & Relationship Extraction via LLM (tuples: name, type, description)
   - Entity & Relationship Summarization (consolidate descriptions from multiple chunks)
   - Claim Extraction (optional, disabled by default)
   - **Gleaning**: Logit bias of 100 forces yes/no on "were entities missed?" — triggers re-extraction
4. **Graph Augmentation**: Hierarchical Leiden community detection
5. **Community Summarization**: Bottom-up LLM summaries at each hierarchy level
6. **Text Embedding**: Vectors for TextUnits, entities, community reports (default: LanceDB, 3072D)

#### Community Detection (Leiden Algorithm)
- Applied hierarchically via `graspologic` library
- Level 0: Fewest, largest communities (root-level generic groupings)
- Higher levels: Many smaller, specific communities
- Recursion within each community until `max_cluster_size` (default: 10)
- Each level provides mutually exclusive, collectively exhaustive coverage
- Scale example: 8,564 nodes → 34 communities at Level 0, up to 2,142 at finest level

#### Community Report Structure
```json
{
  "title": "<descriptive_name>",
  "summary": "<executive_summary>",
  "rating": 7.5,
  "rating_explanation": "<single_sentence>",
  "findings": [
    {"summary": "<insight>", "explanation": "<multi_paragraph>"}
  ]
}
```
- 5-10 key insights per report
- Data grounding via `[Data: <dataset> (record ids)]` references
- Generated bottom-up: leaf communities use edge-prominence prioritization,
  higher levels incorporate sub-community summaries

#### Query Modes

**Global Search (Map-Reduce)**:
1. Select community reports from a hierarchy level
2. Map phase: Shuffle reports, segment into ~8K token chunks, parallel LLM extraction
   of key points with Description + Importance Score (0-100)
3. Reduce phase: Sort by importance, pack into context window, generate final answer
- Cost: ~610K tokens per query, hundreds of API calls
- Dynamic Community Selection (newer): LLM prunes irrelevant communities first → 77% cost reduction

**Local Search**:
1. Embed query, find top-K nearest entities by cosine similarity (2x oversample)
2. Expand to connected entities, relationships, covariates, community reports
3. Rank: in-network relationships (both endpoints retrieved) prioritized over out-of-network
4. Fill context with proportional token budgets: 50% text chunks, 10% community reports, 40% entities/relationships

**DRIFT Search (Dynamic Reasoning and Inference with Flexible Traversal)**:
1. Primer: Match query against top-K community reports, generate broad answer + follow-up questions
2. Follow-Up: Local search to refine each follow-up question
3. Output: Synthesize all intermediate answers hierarchically
- Outperforms Local Search 78% on comprehensiveness, 81% on diversity

#### Entity Resolution
- **Acknowledged gap**: Exact name match only, no fuzzy matching or coreference resolution
- Previous ER step was removed due to quality issues
- Leiden communities partially compensate by clustering related/duplicate entities

#### Auto Prompt Tuning
- Generates domain-adapted prompts for extraction, summarization, community reports
- Entity type discovery: LLM identifies domain-appropriate types automatically
- Few-shot selection: embed text units, project to lower dimension, select k-nearest to centroid

---

### 1.2 LightRAG

**Repository**: https://github.com/HKUDS/LightRAG
**Paper**: EMNLP 2025 (arXiv 2410.05779)

#### Core Insight
Get most of GraphRAG's benefits without community detection, Leiden clustering,
or map-reduce summarization. Uses graph-enhanced text indexing with dual-level retrieval.

#### R-P-D Indexing Pipeline

**R — Recognize (Entity/Relationship Extraction)**:
- Documents chunked (default 1200 tokens)
- LLM extracts entities and relationships per chunk
- Entity types: Person, Creature, Organization, Location, Event, Concept, Method, Content, Data, Artifact, NaturalObject
- Gleaning loop: configurable re-prompts to catch missed entities

**P — Profile (Key-Value Generation)**:
- Each entity: key = name, value = summarized description
- Each relation: keys = multiple keywords (local + global themes from connected entities), value = summarized description
- When descriptions exceed token limits, LLM map-reduce summarization

**D — Deduplicate (Entity Resolution)**:
- Merge identical entities by name normalization (lowercase, strip spaces)
- Entity type conflicts resolved by frequency voting
- Relationship descriptions concatenated and re-summarized
- Source ID tracking (KEEP all vs FIFO drop oldest)

#### Dual-Level Retrieval
Query → LLM extracts two keyword types:
- **Low-level keywords**: specific entities, proper nouns, technical jargon (the "who/what")
- **High-level keywords**: overarching concepts, themes, core intent (the "about what")

**Low-level retrieval**: Match against entity embeddings → expand to 1-hop graph neighbors
**High-level retrieval**: Match against relation embeddings (carry global-theme keywords)

Both levels pull values from KV store to assemble context for generation.

#### Incremental Indexing (Key Advantage)
- New documents processed through same R-P-D pipeline
- New graph merged with existing graph by union of node/edge sets
- Dedup merges by name, descriptions appended and re-summarized
- New embeddings generated only for new/modified entities/relations
- **No community restructuring, no re-indexing, no re-summarization**
- Cache-first reconstruction: `rebuild_knowledge_from_chunks()` from cached LLM results

#### Six Query Modes
| Mode | How It Works |
|------|-------------|
| Naive | Vector similarity on text chunks, no graph |
| Local | Entity-focused: vector search → graph expansion |
| Global | Relationship-focused: vector search on relations across entire KG |
| Hybrid | Combines local + global (default) |
| Mix | Parallel KG retrieval + vector retrieval |
| Bypass | Direct LLM, no retrieval |

#### Storage Backends
- KV: JSON files, PostgreSQL, Redis, MongoDB
- Vector: NanoVectorDB, pgvector, Milvus, Chroma, Faiss, MongoDB, external vector service
- Graph: NetworkX, external graph backend, PostgreSQL AGE, Memgraph

#### Performance vs GraphRAG
| Metric | GraphRAG | LightRAG |
|--------|----------|----------|
| Tokens per query | ~610,000 | <100 |
| API calls per query | Hundreds | 1 |
| Incremental update | ~13.99M tokens (full rebuild) | Extract-only |
| Win rate (overall) | Baseline | 54-56% wins across 4 datasets |
| Latency | ~120ms | ~80ms |

---

### 1.3 nano-graphrag

**Repository**: https://github.com/gusye1234/nano-graphrag

- Clean-room GraphRAG reimplementation in ~1,100 lines of code
- Retains core pipeline: chunking → LLM extraction → Leiden communities → community reports → search
- **Key simplification**: Global search uses top-K central communities only (default K=512) instead of map-reduce over ALL communities
- Fully async, fully typed
- Supports NetworkX + external graph backend for graph, nano-vectordb + hnswlib + milvus-lite + faiss for vectors
- Communities re-computed on every insert (limitation vs LightRAG)

---

### 1.4 temporal graph reference

**Repository**: https://example.com/reference
**Paper**: arXiv 2501.13956

#### Bi-Temporal Model (Key Innovation)
Every edge carries four timestamps:
- `t_created` / `t_expired` — Transaction Time: when facts enter/leave the database
- `t_valid` / `t_invalid` — Event Time: when facts were true in the real world

This enables queries like "What was true in March 2022?" separately from "What was in the DB in March 2022?"

#### Hybrid Retrieval Pipeline
Three parallel search channels fused:
1. `phi_cos`: Cosine similarity on 1024D embeddings
2. `phi_bm25`: Okapi BM25 full-text search
3. `phi_bfs`: Breadth-first graph traversal

Reranking with: RRF, MMR, episode-mention frequency, node distance from centroid, cross-encoder LLM scoring.

#### Entity Resolution
1. Embed entity name into 1024D vector space
2. Parallel full-text search on existing entity names and summaries
3. LLM comparison of candidates with episode context
4. When duplicates found: generate updated name and summary
5. Constrained through predefined query templates to reduce hallucinations

#### Edge Invalidation
- LLM-based semantic contradiction detection
- Temporally overlapping contradictions: sets `t_invalid = t_valid` of invalidating edge
- Non-lossy: invalidated edges remain for historical queries

---

### 1.5 baseline system

**Repository**: https://github.com/ChihayaAine/baseline system
**Paper**: arXiv 2601.18642

#### Core Decay Equation
```
v_i(t) = v_i(0) * exp(-λ_i * (t - τ_i)^β_i)
```

Decay rate adapts to importance:
```
λ_i = λ_base * exp(-μ * I_i(t))
```
Important memories decay SLOWER (lower λ).

Shape parameter β differs by tier:
- Long-term Memory: β = 0.8 (sub-linear, gradual decay)
- Short-term Memory: β = 1.2 (super-linear, rapid decay)

#### Three-Component Importance Score
```
I_i(t) = α * rel(c_i, Q_t) + β * f_i/(1+f_i) + γ * recency(τ_i, t)
```
- Semantic relevance to recent context
- Saturating frequency function (diminishing returns on access count)
- Temporal recency via exponential decay

#### Memory Consolidation
Memories within temporal-semantic clusters are fused:
```
C_k = {m_i : sim(c_i, c_k) > θ_fusion AND |τ_i - τ_k| < T_window}
```
Fused memories get reduced decay: `λ_fused = λ_base / (1 + log(|C_k|))`

#### Conflict Resolution (4 categories via LLM)
- **Compatible**: Reduce importance by redundancy penalty
- **Contradictory**: Competitive dynamics favoring newer information
- **Subsumes/Subsumed**: LLM-guided merging
- **Independent**: No interaction

#### Results
- 82.1% critical fact retention at 55% storage
- vs baseline 78.4% at 100% storage after 30 days

---

### 1.6 Baseline

**Repository**: https://github.com/prior workai/prior work
**Paper**: arXiv 2504.19413

- GPT-4o-mini extracts salient memories from conversation summary + recent 10 messages
- Update Classification: ADD/UPDATE/DELETE/NOOP via LLM function-calling against top-10 similar memories
- Graph Memory (graph memory baseline): Directed labeled graph with entities as nodes, relationships as edges
- No automatic temporal decay — memories persist indefinitely
- Contradiction detection marks relationships as invalid

---

## Part 2: Temporal Graph Techniques from Academia

### 2.1 Temporal Knowledge Graph Completion (TKGC)

**Most relevant for EventGraphDB**:

**DE-SimplE** (AAAI 2020): Entity embeddings as functions of time:
```
e(τ) = f(entity, τ)
```
No per-timestamp embedding vectors needed. Proven fully expressive.
Lightweight and fits in-memory systems perfectly.

**Time2Vec** (arXiv 1907.05321): Standard approach for encoding continuous time:
```
t2v(τ)[i] = ω_i * τ + φ_i           (for i=0, linear term)
t2v(τ)[i] = sin(ω_i * τ + φ_i)      (for i≥1, periodic terms)
```
Linear term captures non-periodic progression; sinusoidal terms capture periodic patterns (daily, weekly).
Model-agnostic — can concatenate with any existing embedding.

**Not recommended**: TTransE (too simple), HyTE (memory explosion with per-timestamp hyperplanes),
TNTComplEx (expensive tensor decomposition), RE-GCN (requires GPU training).

### 2.2 Temporal Graph Networks (TGN)

**Architecture** (Twitter Research, arXiv 2006.10637):
1. **Memory Module**: Per-node state vector `s_i(t)` compressing all past interactions
2. **Message Function**: On event between nodes i,j: `m_i(t) = msg(s_i(t-), s_j(t-), Δt, e_ij)`
3. **Message Aggregator**: Last (most recent) or Mean (average all)
4. **Memory Updater**: GRU: `s_i(t) = GRU(s_i(t-), agg_msg_i(t))`
5. **Embedding Module**: Temporal graph attention over neighborhood

**Key insight for EventGraphDB**: The per-node memory state pattern is directly applicable
without the full neural architecture. Use exponential moving average instead of GRU:
```rust
struct NodeTemporalState {
    state: Vec<f32>,
    last_updated: Timestamp,
    interaction_count: u32,
}
```

### 2.3 Event-Based Temporal Graph Models

**TGAT**: Attention weights depend on time gap between interactions (via Random Fourier Features)
**DyRep**: Models WHEN interactions happen via temporal point process
**JODIE**: Learns embedding TRAJECTORIES — predicts where embedding will be in the future:
```
e_u(t + Δ) = e_u(t) + Δ * W * e_u(t)
```

**UTG Framework finding**: Performance advantage of event-based models stems primarily from
leveraging joint neighborhood structural features (common neighbors), NOT from continuous-time
representation itself. Snapshot models with good structural features can be competitive.

### 2.4 Temporal Community Detection

**Incremental Label Propagation** (most practical):
- Each node adopts majority label from neighbors
- New nodes adopt plurality community of their neighbors
- O(E) per iteration, converges in 3-5 iterations
- temporal graph reference uses this approach

**Incremental Louvain** (external graph backend approach):
- Seed with previous community assignments
- Only local re-optimization on new edges
- Better quality but slower

**Community Evolution Events to track**:
- Growth/Shrink, Split, Merge, Birth/Death, Migration

### 2.5 Graph Summarization Over Time

**MoSSo** (Incremental Lossless Summarization):
- Supernodes (node clusters) + superedges + corrections
- Processes each change in <0.1ms
- 7 orders of magnitude faster than batch methods

**Practical approach**: Rolling graph summaries per entity neighborhood
(extending our existing conversation rolling summaries to graph level).

---

## Part 3: EventGraphDB Current Capabilities Audit

### What We Already Have

| Category | Feature | Location | Status |
|----------|---------|----------|--------|
| **Temporal** | Snapshot view (point-in-time filtering) | temporal_view.rs | Complete |
| **Temporal** | Rolling window (time-range view) | temporal_view.rs | Complete |
| **Temporal** | BTreeMap temporal index | structures.rs:685 | Complete |
| **Temporal** | Exponential decay scoring | retrieval/temporal.rs | Complete |
| **Temporal** | Temporal reachability (causal chains) | algorithms/temporal_reachability.rs | Complete |
| **Temporal** | Soft-delete edges with audit | structures.rs:1000+ | Phase 18 |
| **Community** | Louvain (two-phase modularity) | algorithms/louvain.rs | Complete |
| **Community** | Label Propagation (O(V+E)) | algorithms/label_propagation.rs | Complete |
| **Graph Algo** | Random Walk / PPR (xoshiro256** PRNG) | algorithms/random_walk.rs | Complete |
| **Graph Algo** | Betweenness Centrality (Brandes) | algorithms/centrality.rs | Complete |
| **Graph Algo** | Degree Centrality (normalized) | algorithms/centrality.rs | Complete |
| **Retrieval** | 6-signal RRF fusion | retrieval/ | Complete |
| **Retrieval** | BM25 keyword search | retrieval/memory_retrieval.rs | Complete |
| **Retrieval** | Semantic similarity | retrieval/memory_retrieval.rs | Complete |
| **Retrieval** | Context similarity | retrieval/memory_retrieval.rs | Complete |
| **Retrieval** | PPR proximity | retrieval/memory_retrieval.rs | Complete |
| **Retrieval** | Temporal decay signal | retrieval/memory_retrieval.rs | Complete |
| **Retrieval** | Tier boost (Schema > Semantic > Episodic) | retrieval/memory_retrieval.rs | Complete |
| **Retrieval** | LLM reranking (top-K + alpha blend) | retrieval/reranker.rs | Phase 16 |
| **Memory** | 3-tier consolidation (Episodic→Semantic→Schema) | consolidation.rs | Complete |
| **Memory** | LLM refinement (summary, embedding) | refinement.rs | Complete |
| **Memory** | LLM update classification (ADD/UPDATE/DELETE) | memory_classifier.rs | Phase 16 |
| **Memory** | Audit trail (append-only log) | memory_audit.rs | Phase 16 |
| **Memory** | Memory TTL / expiration | memory/mod.rs | Phase 19 |
| **Memory** | Category auto-tagging | claims/llm_client.rs | Phase 19 |
| **Claims** | Role-aware LLM extraction (User/Assistant/System) | claims/llm_client.rs | Phase 16 |
| **Claims** | NER + geometric validation | claims/extractor.rs | Complete |
| **Claims** | Dedup / contradiction detection | maintenance.rs | Complete |
| **Claims** | Custom instructions + includes/excludes | claims/extractor.rs | Phase 19 |
| **Claims** | Few-shot examples | claims/llm_client.rs | Phase 19 |
| **Claims** | Piecewise outcome scoring (Bayesian→EMA) | claims/types.rs | Complete |
| **Entity** | Fuzzy matching (Jaro-Winkler ≥ 0.85) | claims/extractor.rs | Complete |
| **Entity** | NER-labeled entities on claims | claims/types.rs | Complete |
| **Entity** | NLQ entity extraction (code, quoted, typed, capitalized) | nlq/entity.rs | Complete |
| **NLQ** | 10 intent types + 8 aggregate metrics | nlq/intent.rs | Complete |
| **NLQ** | Fuzzy intent classification | nlq/intent.rs | Phase 9 |
| **NLQ** | Multi-hop, compound, negation queries | nlq/ | Phase 9 |
| **Conversation** | Rolling summary (per-turn LLM refresh) | conversation/compaction.rs | Phase 18 |
| **Conversation** | Compaction (facts/goals/procedural) | conversation/compaction.rs | Phase 17 |
| **Persistence** | ReDB with goal-bucket sharding | redb_graph_store.rs | Complete |
| **Persistence** | Delta persistence (dirty tracking) | structures.rs | Phase 5 |
| **Structure** | 11 NodeType variants | structures.rs:327-407 | Complete |
| **Structure** | 10 EdgeType variants (Causality, Temporal, etc.) | structures.rs:577-642 | Complete |
| **Structure** | Adaptive adjacency lists (Empty→One→Small→Large) | structures.rs | Phase 5 |
| **Structure** | String interning pool | intern.rs | Phase 5 |

### Graph Structure Detail

**GraphNode** fields: id, node_type (11 variants), created_at, updated_at, properties, degree

**GraphEdge** fields: id, source, target, edge_type (10 variants), weight, created_at, updated_at,
observation_count, confidence, properties + soft-delete (is_valid, invalidated_at, invalidated_reason)

**Temporal Index**: `BTreeMap<Timestamp, Vec<NodeId>>` for O(log n) range queries

---

## Part 4: Gap Analysis and Ranked Improvements

### ALREADY STRONG (No Action Needed)

- Multi-signal retrieval (6 signals via RRF — most systems have 2-3)
- Community detection (Louvain + Label Propagation — LightRAG has neither)
- Temporal reachability with causal monotonicity
- 3-tier memory consolidation with LLM refinement
- LLM-driven memory lifecycle (ADD/UPDATE/DELETE)
- Soft-delete edges with audit trail
- Role-aware claim extraction with NER grounding
- Piecewise outcome scoring (Bayesian → EMA transition)

### GAPS WORTH CLOSING

---

### Tier 1: High Impact, Low Effort

#### Improvement 1: Importance-Modulated Temporal Decay
**Source**: baseline system

**Current**: Fixed exponential decay with half-life per ClaimType.
`score(t) = e^(-λ*age)` where λ is fixed.

**Upgrade**: Decay rate adapts to importance:
```
λ_i = λ_base * exp(-μ * I_i(t))
```
Importance I_i combines:
- Semantic relevance to recent context: `α * cosine_sim(memory, recent_query)`
- Access frequency with saturation: `β * freq / (1 + freq)`
- Temporal recency: `γ * exp(-δ * age)`

**Why**: A frequently-accessed, high-relevance memory decays much slower than a rarely-used one.
Current system treats all memories of same ClaimType identically. baseline system achieves 82% fact
retention at 55% storage vs baseline 78% at 100%.

**Where to change**: `retrieval/temporal.rs` (~30 lines), `retrieval/memory_retrieval.rs` (~20 lines)

---

#### Improvement 2: Bi-Temporal Edge Model
**Source**: temporal graph reference

**Current**: Edges have `created_at`, `updated_at`, soft-delete (`is_valid`, `invalidated_at`).
This is transaction time only.

**Upgrade**: Add event time (when the fact was true in reality):
```rust
// On GraphEdge (via properties or first-class fields)
valid_from: Option<Timestamp>,   // when fact became true in real world
valid_until: Option<Timestamp>,  // when fact stopped being true
```

**Why**: "Alice worked at Google from 2020-2023" has different valid time than when recorded.
Queries like "What was true in March 2022?" need valid time. Currently we can only answer
"What was in the database at time X?"

**Where to change**: `structures.rs` (~20 lines), `temporal_view.rs` (~40 lines)

---

#### Improvement 3: Statement Temporal Typing
**Source**: OpenAI Cookbook temporal agent pattern

**Current**: All facts get same decay within their ClaimType.

**Upgrade**: Classify statements into temporal categories:
```rust
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TemporalType {
    Static,     // "Paris is the capital of France" — never decay
    Dynamic,    // "Alice lives in NYC" — decay and supersede
    Atemporal,  // "2+2=4" — mathematical truth, permanent
}
```

LLM classifies during extraction. Decay function uses temporal_type to select curve:
- Static/Atemporal: λ = 0 (no decay)
- Dynamic: normal decay with type-specific half-life

**Why**: Static facts being decayed wastes retrieval quality. Dynamic facts not decaying
fast enough causes stale information.

**Where to change**: `claims/types.rs` (~15 lines), `claims/llm_client.rs` (~10 lines),
`retrieval/temporal.rs` (~15 lines)

---

### Tier 2: High Impact, Medium Effort

#### Improvement 4: Community Summaries
**Source**: Microsoft GraphRAG

**Current**: Louvain and Label Propagation detect communities but produce only
node→community mappings. No summaries generated.

**Upgrade**: For each community, generate a structured summary:
```rust
pub struct CommunitySummary {
    pub id: u64,
    pub title: String,
    pub summary: String,
    pub summary_embedding: Vec<f32>,
    pub members: Vec<NodeId>,
    pub member_count: usize,
    pub rating: f32,           // importance 0-10
    pub findings: Vec<String>, // key insights
    pub created_at: Timestamp,
    pub last_updated: Timestamp,
}
```

After community detection, LLM summarizes each community's members.
Store summaries + embeddings for retrieval.
Add community summary as a 7th retrieval signal in RRF fusion.

**Why**: Communities without summaries are only useful for clustering. With summaries,
they enable corpus-wide queries ("What are the main themes?") that no single memory can answer.
GraphRAG's community summaries are their key differentiator over traditional RAG.

**Where to change**: New `community_summary.rs` (~200 lines), retrieval pipeline (~50 lines),
integration wiring (~30 lines)

---

#### Improvement 5: Graph-Traversal Retrieval Signal
**Source**: temporal graph reference, GraphRAG Local Search

**Current**: 6 retrieval signals (BM25, semantic, context, PPR, temporal, tier).
None use the entity-relationship graph structure directly.

**Upgrade**: Add BFS neighborhood expansion as a parallel retrieval signal:
1. From query entities (already resolved in NLQ pipeline), BFS expand 1-2 hops
2. Score traversed memories by hop distance and edge weight
3. Fuse as 7th signal in RRF

**Why**: Graph structure captures relationships that embedding similarity misses.
Two memories about the same entity should be retrieved together even if their text
is dissimilar. temporal graph reference's BFS signal is their 3rd strongest retrieval channel.

**Where to change**: New `retrieval/graph_traversal.rs` (~80 lines),
`retrieval/memory_retrieval.rs` (~30 lines)

---

#### Improvement 6: Entity Alias Timeline
**Source**: temporal graph reference entity resolution + temporal ER research

**Current**: Entity resolution uses Jaro-Winkler fuzzy matching on normalized names.
No tracking of name changes over time.

**Upgrade**:
```rust
pub struct EntityAlias {
    pub canonical_id: NodeId,
    pub alias: String,
    pub valid_from: Timestamp,
    pub valid_until: Option<Timestamp>,
}
```

Track entity identity across name changes. "Jon" in January and "Jonathan" in March
may be the same person. Temporal co-occurrence patterns help disambiguate.

**Where to change**: New alias store (~100 lines), extractor modifications (~40 lines)

---

### Tier 3: Medium Impact, Higher Effort

#### Improvement 7: Relation Embeddings for Global Queries
**Source**: LightRAG

**Current**: No way to answer corpus-wide questions without scanning all memories.

**Upgrade**: Each relationship edge gets embedded with global-theme keywords.
High-level queries match against relation embeddings, bypassing community detection.

Much cheaper than GraphRAG's community reports. Works incrementally (new edges
get embedded, no rebuild). LightRAG uses <100 tokens/query vs GraphRAG's 610K.

**Tradeoff**: Less comprehensive than community summaries for truly global questions.

**Where to change**: Edge creation pipeline (~150 lines new code)

---

#### Improvement 8: Incremental Community Updates

**Current**: Community detection runs on full graph each time.

**Upgrade**: Track `dirty_nodes: HashSet<NodeId>`. On edge add/remove, mark endpoints
as dirty. Only re-optimize labels for dirty nodes and their 1-hop neighbors.

**Where to change**: `algorithms/louvain.rs` (~30 lines), `algorithms/label_propagation.rs` (~30 lines)

---

#### Improvement 9: baseline system-Style Conflict Resolution

**Current**: Memory consolidation merges by theme/goal similarity. No explicit conflict detection.

**Upgrade**: Before merging, LLM classifies relationship:
- **Compatible**: Reduce importance by redundancy penalty
- **Contradictory**: Competitive dynamics favoring newer info
- **Subsumes/Subsumed**: LLM-guided merging
- **Independent**: No interaction

**Where to change**: `consolidation.rs` (~100 lines)

---

#### Improvement 10: DRIFT-Style Adaptive Search

**Current**: NLQ pipeline classifies intent and routes to single execution path.

**Upgrade**: Three-phase adaptive search for complex queries:
1. Primer: Match against community summaries for broad context
2. Follow-Up: Local search to refine with specific entities
3. Synthesis: Combine all intermediate answers hierarchically

DRIFT outperformed local search 78% on comprehensiveness.

**Tradeoff**: Multiple LLM calls per query. Only valuable for complex queries.

**Where to change**: New multi-phase query mode (~200 lines new code)

---

## Part 5: What NOT to Adopt

| Technique | Why Skip |
|-----------|----------|
| **GraphRAG map-reduce global search** | 610K tokens/query. Community summaries (#4) give 80% value at 1% cost |
| **Full TGN neural architecture** | Too heavy for in-memory Rust. Importance-modulated decay (#1) captures key insight |
| **Tensor decomposition TKGC** | Designed for large-scale KG completion, not agent memory retrieval |
| **Per-timestamp embeddings (HyTE)** | Memory explosion. DE-SimplE time-as-function approach better for our scale |
| **GraphRAG auto prompt tuning** | Our few-shot examples (Phase 19) + custom instructions cover this more simply |
| **Full bi-temporal SQL model** | Overkill. Adding valid_from/valid_until to edges (#2) is sufficient |
| **LightRAG's query routing** | Not automatic (user selects mode). Our NLQ intent classifier is better |
| **nano-graphrag's approach** | No advantages over our existing architecture. Less capable |

---

## Part 6: Implementation Roadmap

### Phase A — Quick Wins (1-2 days)
1. Importance-modulated decay (~50 lines)
2. Statement temporal typing (~40 lines)
3. Bi-temporal edge fields (~60 lines)

### Phase B — Medium Effort (2-3 days)
4. Graph-traversal retrieval signal (~110 lines)
5. Entity alias timeline (~140 lines)

### Phase C — Larger Features (3-5 days)
6. Community summaries with embeddings (~280 lines)
7. Incremental community updates (~60 lines)

### Phase D — Optional Advanced (3-5 days)
8. Relation embeddings for global queries (~150 lines)
9. baseline system conflict resolution (~100 lines)
10. DRIFT-style adaptive search (~200 lines)

**Total estimated new code**: ~1,190 lines across all 10 improvements.

---

## Part 7: Sources

### Systems & Repositories
- [Microsoft GraphRAG](https://github.com/microsoft/graphrag) — Community hierarchy + map-reduce
- [LightRAG](https://github.com/HKUDS/LightRAG) — Dual-level retrieval without communities
- [nano-graphrag](https://github.com/gusye1234/nano-graphrag) — Minimal GraphRAG in 1100 LOC
- [temporal graph reference](https://example.com/reference) — Bi-temporal agent memory KG
- [baseline system](https://github.com/ChihayaAine/baseline system) — Adaptive forgetting for LLM memory
- [baseline](https://github.com/prior workai/prior work) — LLM-driven memory lifecycle

### Papers
- GraphRAG: "From Local to Global" (arXiv 2404.16130)
- LightRAG: EMNLP 2025 Findings (arXiv 2410.05779)
- temporal graph reference: arXiv 2501.13956
- baseline system: arXiv 2601.18642
- TGN: Temporal Graph Networks (arXiv 2006.10637)
- Time2Vec: Learning Time Representations (arXiv 1907.05321)
- DE-SimplE: Diachronic Embeddings (arXiv 1907.03143)
- RE-GCN: Recurrent Evolution GCN (arXiv 2104.10353)
- ChronoR: Rotation-Based Temporal KG Embedding (AAAI 2021)
- BoxTE: Box Embeddings for TKGC (AAAI 2022)
- UTG: Unified Temporal Graph Models (arXiv 2407.12269)
- MoSSo: Incremental Lossless Graph Summarization (arXiv 2006.09935)
- PathRAG: arXiv 2502.14902
- TKGC Survey (IJCAI 2023, arXiv 2308.02457)
- Temporal Community Detection Survey (Springer 2023)
- Temporal Interaction Graph Survey (IJCAI 2025, arXiv 2505.04461)

### Benchmarks
- LightRAG beats GraphRAG 54-56% overall across 4 datasets
- LightRAG uses <100 tokens per query vs GraphRAG's 610,000
- baseline system retains 82.1% critical facts at 55% storage vs baseline 78.4% at 100%
- GraphRAG DRIFT beats local search 78% on comprehensiveness, 81% on diversity
- GraphRAG Dynamic Community Selection reduces cost by 77% with comparable quality
- PathRAG achieves 59.93% win rate vs GraphRAG, 57.09% vs LightRAG, with 40% less tokens
