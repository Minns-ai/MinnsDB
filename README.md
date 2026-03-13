<p align="center">
  <img src="img/minnsFish.png" alt="MinnsDB" width="400" />
</p>

<h1 align="center">MinnsDB</h1>

<p align="center">
  <strong>A graph database that learns from conversations.</strong><br>
  Send messages in. Ask questions in plain English. Get answers from the graph.
</p>

<p align="center">
  Pure Rust &middot; Single binary &middot; Embedded storage &middot; No GPU &middot; 800+ tests
</p>

---

## 30-second demo

```bash
cargo run --release -p minnsdb-server
# → Listening on http://0.0.0.0:3000
```

```bash
# 1. Send in a conversation
curl -X POST http://localhost:3000/api/conversations/ingest \
  -H 'Content-Type: application/json' \
  -d '{
    "agent_id": 1,
    "sessions": [{
      "session_id": "s1",
      "messages": [
        {"role": "user", "content": "I just moved from London to Berlin for a new job at Stripe"},
        {"role": "assistant", "content": "Congrats on the move and the new role at Stripe!"},
        {"role": "user", "content": "Thanks! I prefer working remotely though"},
        {"role": "assistant", "content": "Good to know. Remote work is great in Berlin."}
      ]
    }]
  }'

# 2. Ask questions in natural language
curl -X POST http://localhost:3000/api/nlq \
  -H 'Content-Type: application/json' \
  -d '{"query": "Where does the user live?", "agent_id": 1}'
# → {"answer": "Berlin", "confidence": 0.95, "intent": "State"}

curl -X POST http://localhost:3000/api/nlq \
  -H 'Content-Type: application/json' \
  -d '{"query": "What are their work preferences?", "agent_id": 1}'
# → {"answer": "prefers remote work", "confidence": 0.9, "intent": "Preference"}
```

Behind that single ingest, MinnsDB ran LLM compaction to extract facts (`lives_in: Berlin`, `employed_at: Stripe`, `prefers: remote work`), created typed graph edges with temporal validity (the old `London` edge gets a `valid_until` timestamp — never deleted, just superseded), and indexed everything for natural language queries. No schema definition needed.

---

## Why not just use X?

| | Vector DB | Graph DB (external graph backend) | KV + embeddings | **MinnsDB** |
|---|---|---|---|---|
| Tracks relationships | No | Yes | No | **Yes** — typed edges with temporal validity |
| Forgets stale info | No | No | No | **Yes** — half-life decay per claim type |
| Handles contradictions | Last write wins | Last write wins | Last write wins | **Confidence scoring** — claims compete |
| Understands time | No | Manual | No | **Built-in** — every edge has `valid_from`/`valid_until` |
| Learns from outcomes | No | No | No | **Yes** — Q-learning on edge weights |
| Accepts raw conversation | No | No | No | **Yes** — LLM compaction extracts facts automatically |
| External deps | Vector service | JVM + Cypher | Redis/Postgres | **None** — single binary, embedded ReDB |

---

## How it works

### Primary flow: Conversations in, answers out

```
  POST /api/conversations/ingest          POST /api/nlq
  POST /api/messages                      POST /api/conversations/query
         │                                         │
         ▼                                         ▼
  ┌─────────────────┐                    ┌──────────────────┐
  │  LLM COMPACTION  │                    │  NLQ PIPELINE     │
  │                  │                    │                   │
  │  Raw messages →  │                    │  Question →       │
  │  Facts extracted │                    │  Intent classify  │
  │  Goals detected  │                    │  Entity extract   │
  │  Summaries built │                    │  Graph projection │
  └────────┬─────────┘                    │  Answer synthesis │
           ▼                              └──────────────────┘
  ┌─────────────────┐                              ▲
  │  GRAPH EDGES     │                              │
  │                  │                    walks edges with
  │  Typed edges     │                    temporal filtering
  │  Temporal        │──────────────────────────────┘
  │  validity        │
  │  OWL/RDFS        │
  │  enforcement     │
  └────────┬─────────┘
           ▼
  ┌─────────────────┐     ┌───────────────┐     ┌───────────────┐
  │ EPISODE          │     │  MEMORY        │     │  STRATEGY      │
  │ DETECTION        │     │  FORMATION     │     │  EXTRACTION    │
  │                  │     │  [WIP]         │     │  [WIP]         │
  │ Temporal gaps    │     │ Consolidation  │     │ Pattern mining │
  │ Context switch   │     │ Half-life decay│     │ Q-value RL     │
  │ Salience scoring │     │ Schema tiers   │     │                │
  └─────────┬────────┘     └───────────────┘     └───────────────┘
            ▼
  ┌─────────────────┐
  │ CLAIMS            │
  │ EXTRACTION        │
  │                   │
  │ Subject/pred/obj  │
  │ BM25 + vector     │
  │ indexing           │
  └───────────────────┘
```

### Conversation ingestion

Two ways to send data in:

**Batch ingest** (`POST /api/conversations/ingest`) — send full sessions with multiple messages. The LLM compaction pass extracts facts, goals, and procedural summaries, then creates typed graph edges automatically. Pre-creates participant entity nodes and runs the full pipeline.

**Streaming messages** (`POST /api/messages`) — send individual messages as they arrive. Messages are buffered per conversation (default buffer size: 6, timeout: 30s). When the buffer fills or times out, compaction runs automatically with a rolling summary for context continuity.

Both require an LLM API key (`LLM_API_KEY` or `OPENAI_API_KEY`).

### Natural language queries

The NLQ pipeline resolves questions against the graph in 7 steps:

1. **Intent classification** — categorizes the question (State, Preference, EntitySummary, Numeric, RelationshipPath, FindNeighbors, FindPath, TemporalChain, Ranking, Aggregation, etc.)
2. **Entity extraction** — 6 strategies: code identifiers, quoted strings, hash IDs, keyword-typed, capitalized words, dedup
3. **Entity resolution** — exact index → BM25 → fuzzy → n-gram matching
4. **Template matching** — maps intent + entities to a graph query template
5. **Query building + validation** — instantiates query, validates node existence, auto-repairs
6. **Graph projection** — walks edges with temporal filtering. For state queries: finds latest edge where `valid_until = None`
7. **Answer formatting** — graph results → human-readable answer with confidence score

Conversation-aware query types:
- **State** — "Where does Alice live?" → walks `location:lives_in` edges, returns current
- **Preference** — "What does Bob prefer?" → walks `preference:*` edges
- **EntitySummary** — "Tell me about Alice" → aggregates all current edges for entity
- **Numeric** — "What's the net balance?" → walks `financial:*` edges, computes sum
- **RelationshipPath** — "How are Alice and Bob connected?" → shortest path through relationship edges

### Graph as source of truth

The graph is the primary data structure — not a secondary index. State queries walk edges with temporal filtering. "Where does Alice live?" finds the latest `location:lives_in` edge where `valid_until = None`. When Alice moves, the old edge gets closed with a timestamp. Never deleted. Full history, always.

Edge types follow `{category}:{predicate}`:
```
location:lives_in     relationship:colleague    financial:payment
work:employed_at      preference:prefers        routine:morning
health:condition      education:studies_at      state:status
```

Property behaviors are defined in OWL/RDFS Turtle files (`data/ontology/*.ttl`), not hardcoded:

- **`owl:FunctionalProperty`** — single-valued, new value supersedes old (e.g., `location:lives_in`)
- **`owl:SymmetricProperty`** — bidirectional (e.g., `relationship:colleague`)
- **`owl:TransitiveProperty`** — transitive closure (e.g., geographic `containedIn`)
- **`eg:appendOnly`** — immutable history, no conflict detection (e.g., `financial:payment`)
- **`eg:cascadeDependents`** — changing location invalidates dependent routines/commute

Sub-property expansion is automatic — querying `relationship` also matches `colleague`, `friend`, `sibling`, `spouse`.

### Advanced: Direct event API

For programmatic graph building beyond conversations, MinnsDB accepts 8 structured event types via `POST /api/events`:

| Type | What it captures |
|------|-----------------|
| `Action` | Agent actions with outcome (Success/Failure/Partial) and duration |
| `Observation` | Environmental observations with confidence and source |
| `Cognitive` | Goal formation, planning, reasoning, memory retrieval |
| `Communication` | Agent-to-agent messages |
| `Learning` | Telemetry: memory retrieved/used, strategy served/used, outcomes |
| `Conversation` | Multi-turn dialogue (classified into transaction/state_change/relationship/preference) |
| `CodeReview` | PR review actions with file/line context |
| `CodeFile` | Source file snapshots for AST parsing |

Events flow through the same pipeline: Graph Construction → Episode Detection → Claims Extraction → Memory Formation → Strategy Extraction.

---

## What's stable, what's WIP

| Component | Status | |
|-----------|--------|-|
| Conversation ingestion + LLM compaction | **Stable** | Multi-turn compaction, fact/goal/procedural extraction, rolling summaries, buffered streaming |
| Natural language queries (NLQ) | **Stable** | 10+ intent types, 6 entity extraction strategies, graph projection with structured memory fallback |
| Graph construction + typed edges + temporal validity | **Stable** | NER entity extraction, ontology enforcement, `valid_from`/`valid_until` on all edges |
| Episode detection | **Stable** | Sliding window, temporal gap + context switch boundaries, salience scoring |
| Claims extraction + hybrid search | **Stable** | LLM-driven subject/predicate/object facts, BM25 + vector search, confidence scoring |
| Graph algorithms | **Stable** | Louvain communities, label propagation, personalized PageRank, betweenness/closeness centrality, temporal reachability, causal paths |
| Code intelligence | **Stable** | Tree-sitter AST parsing for Rust, Python, TypeScript, JavaScript, Go |
| Structured memory | **Stable** | Ledgers, state machines, preference vectors, tree structures — all backed by graph edges |
| Workflows | **Stable** | DAG-based workflow engine with step state tracking, feedback, diff-based updates |
| Persistence + export/import | **Stable** | ReDB (20 tables), delta writes, streaming binary v2 export/import |
| Write lanes + read gate | **Stable** | Sharded write pipeline with backpressure, semaphore-based read concurrency |
| Memory formation | **WIP** | Episodic/semantic/schema tiers implemented. Consolidation heuristics and tier promotion under active development |
| Strategy extraction | **WIP** | Pattern mining and context-action mappings implemented. RL on edge weights is experimental |
| Energy-based world model | **WIP** | ~30K param EBM critic, contrastive learning. CPU-only (<100us/score). Enable with `ENABLE_WORLD_MODEL=true` |
| LLM planning | **WIP** | Generator + critic architecture. Strategy/action candidate generation with validation and repair |

---

## Quick start

### From source

```bash
git clone https://github.com/your-org/minnsdb.git
cd minnsdb
cargo build --release
cargo run --release -p minnsdb-server
# → http://0.0.0.0:3000
```

### Docker

```bash
docker compose up -d
# → http://localhost:3000
```

Two service profiles: `normal` (256MB cache, 1M max nodes, Louvain enabled) and `free` (64MB cache, 50K max nodes).

```bash
# Build free tier
docker build --build-arg SERVICE_PROFILE=free -t minnsdb:free .
```

### Rust SDK

```rust
use agent_db_graph::{GraphEngine, GraphEngineConfig};

let engine = GraphEngine::new().await?;

// Conversation ingestion is the primary interface — but you can also
// work directly with the graph engine for programmatic use:
let event = Event {
    event_type: EventType::Action {
        action_name: "fix_null_error".to_string(),
        parameters: json!({"fix": "add_null_check"}),
        outcome: ActionOutcome::Success {
            result: json!({"tests_pass": true})
        },
        duration_ns: 1_500_000_000,
    },
    ..Default::default()
};

engine.process_event(event).await?;
```

---

## REST API

**Base URL:** `http://localhost:3000`

Full documentation: **[docs.minns.ai](https://docs.minns.ai)**

60+ endpoints across these groups:

| Group | Key Endpoints |
|-------|--------------|
| **Conversations** | `POST /api/conversations/ingest` — batch ingest sessions (requires LLM)<br>`POST /api/messages` — single message with buffering + auto-compaction |
| **Queries** | `POST /api/nlq` — natural language query against graph<br>`POST /api/conversations/query` — conversation-style query<br>`POST /api/search` — unified keyword/semantic/hybrid search<br>`POST /api/claims/search` — semantic claim search<br>`POST /api/code/search` — structural code search |
| **Events** | `POST /api/events` — full pipeline processing<br>`POST /api/events/simple` — simplified event<br>`POST /api/events/state-change` — typed state transitions<br>`POST /api/events/transaction` — typed financial transactions<br>`POST /api/events/code-file` — source file AST analysis<br>`POST /api/events/code-review` — code review events<br>`GET /api/events` — list recent events<br>`GET /api/episodes` — list completed episodes |
| **Memory & Strategy** | `GET /api/memories/agent/:id` — agent memories<br>`POST /api/memories/context` — context similarity search<br>`GET /api/strategies/agent/:id` — learned strategies<br>`GET /api/suggestions` — next-action recommendations |
| **Graph & Analytics** | `GET /api/graph` — graph structure<br>`GET /api/stats` — system statistics<br>`GET /api/analytics` — components, clustering, modularity<br>`GET /api/communities` — Louvain / label propagation<br>`GET /api/centrality` — PageRank, betweenness, closeness<br>`GET /api/causal-path` — causal paths between nodes<br>`GET /api/reachability` — temporal reachability |
| **Structured Memory** | `POST /api/structured-memory` — upsert<br>`POST .../ledger/:key/append` — append ledger entry<br>`POST .../state/:key/transition` — state machine transition<br>`POST .../preference/:key/update` — preference vector update<br>`POST .../tree/:key/add-child` — tree structure |
| **Workflows** | `POST /api/workflows` — create DAG workflow<br>`POST .../steps/:step_id/transition` — step state transition<br>`POST .../feedback` — workflow feedback |
| **Agents** | `POST /api/agents/register` — register agent node<br>`GET /api/agents` — list agents by group |
| **Planning** | `POST /api/planning/strategies` — generate candidates [WIP]<br>`POST /api/planning/plan` — full planning pipeline [WIP] |
| **Admin** | `POST /api/admin/export` — streaming binary export<br>`POST /api/admin/import` — import (replace/merge)<br>`GET /api/health` — health + write lane + read gate metrics |

---

## Architecture

```
minnsdb/
├── crates/
│   ├── agent-db-core/       # Type aliases: EventId, AgentId, Timestamp, NodeId, ContextHash
│   ├── agent-db-events/     # Event struct, 8 EventType variants, EventContext, ActionOutcome
│   ├── agent-db-storage/    # ReDB backend (20 tables), WAL, versioned serialization
│   ├── agent-db-graph/      # GraphEngine — the main orchestrator
│   │                        # Conversation compaction, NLQ pipeline, graph projection
│   │                        # Episode detection, memory formation, strategy extraction
│   │                        # Claims (LLM extraction, BM25+vector search)
│   │                        # Algorithms: Louvain, PageRank, centrality, reachability
│   │                        # Code graph, workflow engine, structured memory
│   │                        # Ontology (OWL/RDFS from TTL), graph pruning
│   ├── agent-db-ast/        # Tree-sitter AST parsing: Rust, Python, TS, JS, Go
│   └── agent-db-ner/        # HTTP client to external NER service
├── ml/
│   ├── agent-db-world-model/  # Energy-based model critic (~30K params) [WIP]
│   └── agent-db-planning/     # LLM-based generator [WIP]
├── server/                  # Axum HTTP server, 60+ endpoints
│   ├── src/handlers/        # 18 handler modules
├── data/ontology/           # 9 OWL/RDFS Turtle files defining property behaviors
├── examples/                # 6 Rust examples (basic → end-to-end pipeline)
└── tests/                   # Integration test suite
```

### Key internals

**GraphEngine** is the central orchestrator. It holds: the graph (`Arc<RwLock<GraphInference>>`), memory store, strategy store, structured memory store, claim store, optional world model, optional NER client, and optional LLM client. Lock ordering is documented: `inference.read → drop → structured_memory.write → drop → inference.write`.

**Graph** uses `SlotVec` (arena allocation, O(1) insert/lookup by NodeId). 11 node types. Bi-temporal edges with transaction time and valid time. Bounded at configurable max size with automatic pruning.

**Write lanes** shard incoming events by session_id hash across N lanes (default: `num_cpus/2`, clamped 2-8). Each lane has bounded capacity (default: 128) and processes sequentially. Per-lane p50/p95/p99 latency tracking.

**Read gate** is a semaphore-based permit system (default: `num_cpus * 2` concurrent reads) with latency tracking.

**Persistence** uses ReDB with 20 tables: graph nodes/edges/adjacency, memories (4 indexes), strategies (3 indexes), transition/motif stats, decision traces, outcome signals, claims, world model state, schema versions.

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `SERVER_HOST` | `0.0.0.0` | Bind address |
| `SERVER_PORT` | `3000` | Bind port |
| `RUST_LOG` | `info` | Log level |
| `SERVICE_PROFILE` | `normal` | `normal` or `free` (controls cache sizes + limits) |
| `LLM_API_KEY` | - | OpenAI-compatible key (required for conversation ingest + claims) |
| `LLM_MODEL` | `gpt-4o-mini` | LLM model for claims extraction |
| `NLQ_HINT_MODEL` | `gpt-4o-mini` | LLM model for NLQ hints |
| `SYNTHESIS_MODEL` | `gpt-4.1` | LLM model for answer synthesis |
| `WRITE_LANE_COUNT` | `num_cpus/2` | Write lane concurrency (clamped 2-8) |
| `WRITE_LANE_CAPACITY` | `128` | Per-lane queue depth |
| `READ_GATE_PERMITS` | `num_cpus*2` | Concurrent read permits |
| `REDB_CACHE_SIZE_MB` | `256` | ReDB page cache (64 in free profile) |
| `NER_SERVICE_URL` | `http://localhost:8081/ner` | External NER service (full URL) |
| `NER_REQUEST_TIMEOUT_MS` | `5000` | NER timeout |
| `PLANNING_LLM_API_KEY` | falls back to `LLM_API_KEY` | Separate key for planning |
| `ENABLE_WORLD_MODEL` | `false` | Enable energy-based world model [WIP] |
| `WORLD_MODEL_MODE` | `Disabled` | `Shadow` / `ScoringOnly` / `Full` |
| `ENABLE_STRATEGY_GENERATION` | `false` | Enable LLM strategy generation [WIP] |
| `ENABLE_LOUVAIN` | `true` | Background community detection |
| `LOUVAIN_INTERVAL` | `1000` | Events between Louvain runs |

Full reference at [docs.minns.ai](https://docs.minns.ai).

---

## Development

```bash
cargo build                     # Build all crates
cargo test --workspace          # 800+ tests
cargo clippy --workspace        # Lint
cargo fmt --all                 # Format
```

The `Makefile` has additional targets: `bench`, `bench-baseline`, `bench-compare`, `perf-test`, `memory-test`, `flamegraph`, `load-test`, `stress-test`, `audit`, `pre-release`. Run `make help` to see all.

### Examples

Six Rust examples in `examples/`:

```bash
cargo run --example basic_usage
cargo run --example graph_integration
cargo run --example scoped_inference
cargo run --example concurrent_events
cargo run --example end_to_end_demo
```

---

## License

MinnsDB is licensed under the **Business Source License 1.1 (BSL)**. It converts to the **AGPL v3.0 with a linking exception** after a few years. The linking exception means you are not required to open-source your own code if you use MinnsDB. You only need to contribute back changes to MinnsDB itself.
