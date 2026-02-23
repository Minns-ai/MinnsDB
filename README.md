# EventGraphDB

**An event-driven contextual graph database with memory-inspired learning, graph algorithms, and generative planning.**

## Overview

EventGraphDB is a specialized database that captures, learns from, and enhances agent behavior through contextual memory formation. Events flow in, a knowledge graph grows, memories consolidate, strategies emerge, and the system tells your agent what to do next.

### Key Features

- **Event-First Architecture** - All knowledge derives from immutable event streams
- **Memory-Inspired** - Episodic/Semantic/Schema tiers with consolidation and decay
- **Contextual Graph** - Relationships emerge from actual agent experiences
- **Graph Algorithms** - Louvain communities, temporal reachability, PageRank, centrality, random walks
- **Generative Planning** - LLM-based strategy/action generation with energy-based world model scoring
- **Semantic Search** - BM25 + vector hybrid search over claims and entities
- **NER Pipeline** - Named entity recognition with claim extraction and embedding
- **Bounded Sharding** - Per-goal-bucket partitions with capped shard count (256 max)
- **Delta Persistence** - Only dirty nodes/edges written to disk on save
- **Streaming Queries** - Batched iteration with early termination
- **Pure Rust** - No Python, no GPU, no external ML frameworks

## Quick Start

### Prerequisites

- Rust 1.70+
- NER service running (see [NER Setup](#ner-service))
- LLM API key (optional, for claim extraction + planning)

### Installation

```bash
git clone https://github.com/your-org/eventgraphdb.git
cd eventgraphdb
cargo build --release
```

### Environment

```bash
cp .env.example .env
# Edit .env with your settings:
#   NER_SERVICE_URL=http://192.168.1.100:8081/ner  (full URL with protocol)
#   LLM_API_KEY=sk-your-key                         (optional)
```

### Run the Server

```bash
cargo run --release -p eventgraphdb-server
# Listening on http://0.0.0.0:8080
```

### Docker

```bash
docker compose up -d
```

### Basic Rust Usage

```rust
use agent_db_graph::{GraphEngine, GraphEngineConfig};
use agent_db_events::{Event, EventType, ActionOutcome};

let engine = GraphEngine::new().await?;

let event = Event {
    id: generate_event_id(),
    timestamp: current_timestamp(),
    agent_id: 1,
    agent_type: "ai-debugger".to_string(),
    session_id: 42,
    event_type: EventType::Action {
        action_name: "fix_null_error".to_string(),
        parameters: json!({"fix": "add_null_check"}),
        outcome: ActionOutcome::Success {
            result: json!({"tests_pass": true})
        },
        duration_ns: 1_500_000_000,
    },
    causality_chain: vec![],
    context: your_context,
    metadata: HashMap::new(),
    ..Default::default()
};

engine.process_event(event).await?;

// Query what the system learned
let memories = engine.get_agent_memories(1, 10).await;
let strategies = engine.get_agent_strategies(1, 10).await;
let suggestions = engine.get_next_action_suggestions(context_hash, None, 5).await?;
```

## Architecture

```
SDK / Agent ──HTTP──▶ REST API Server (axum)
                           │
                     ┌─────▼──────┐
                     │ GraphEngine │ ◀── Central Hub
                     └─────┬──────┘
           ┌───────┬───────┼───────┬──────────┐
           ▼       ▼       ▼       ▼          ▼
        Events   Graph   Memory  Strategy   Planning
        Pipeline Store   System  Extractor  (LLM+EBM)
           │       │       │       │          │
           └───────┴───────┴───────┴──────────┘
                           │
                     ┌─────▼──────┐
                     │   ReDB     │ ◀── Persistent Storage
                     │  (embedded)│
                     └────────────┘
```

### Data Flow

```
Events → Graph Construction → Episode Detection → Memory Formation → Strategy Extraction
  ↑                                                                         ↓
  └── Action Suggestions ← Context Matching ← Memory Retrieval ← Patterns ─┘
```

### Crate Structure

```
eventgraphdb/
├── crates/
│   ├── agent-db-core/      # Core types: AgentId, Timestamp, EventId, etc.
│   ├── agent-db-events/    # Event, EventType, EventContext, ActionOutcome
│   ├── agent-db-storage/   # ReDB backend, versioned serialization
│   ├── agent-db-graph/     # Graph engine, algorithms, memory, strategies
│   └── agent-db-ner/       # NER service client, entity extraction
├── ml/
│   ├── agent-db-world-model/  # Energy-based model (~30K params)
│   └── agent-db-planning/     # LLM generator + critic architecture
├── server/                 # Axum REST API server
└── examples/               # Usage examples
```

## REST API

**Base URL:** `http://localhost:8080`

See **[API_REFERENCE.md](API_REFERENCE.md)** for complete endpoint documentation with request/response examples.

### Endpoint Summary

| Method | Path | Description |
|--------|------|-------------|
| **Events** | | |
| `POST` | `/api/events` | Process full event through pipeline |
| `POST` | `/api/events/simple` | Simplified event submission |
| `GET` | `/api/events` | List recent events |
| `GET` | `/api/episodes` | List completed episodes |
| **Memories** | | |
| `GET` | `/api/memories/agent/:agent_id` | Get agent memories |
| `POST` | `/api/memories/context` | Find memories by context similarity |
| **Strategies** | | |
| `GET` | `/api/strategies/agent/:agent_id` | Get agent strategies |
| `POST` | `/api/strategies/similar` | Find similar strategies |
| `GET` | `/api/suggestions` | Get next-action suggestions |
| **Graph** | | |
| `GET` | `/api/graph` | Get graph structure |
| `GET` | `/api/graph/context` | Get graph for context |
| `POST` | `/api/graph/persist` | Force persist to disk |
| `GET` | `/api/stats` | Engine statistics |
| **Analytics** | | |
| `GET` | `/api/analytics` | Graph analytics (components, clustering, modularity) |
| `GET` | `/api/communities` | Community detection (Louvain / Label Propagation) |
| `GET` | `/api/centrality` | Node centrality scores (degree, betweenness, pagerank) |
| `GET` | `/api/ppr` | Personalized PageRank from source node |
| `GET` | `/api/reachability` | Temporal reachability from source |
| `GET` | `/api/causal-path` | Causal path between two nodes |
| `GET` | `/api/indexes` | Index performance stats |
| **Search** | | |
| `POST` | `/api/search` | Unified search (keyword/semantic/hybrid) |
| **Claims** | | |
| `GET` | `/api/claims` | List active claims |
| `GET` | `/api/claims/:id` | Get claim by ID |
| `POST` | `/api/claims/search` | Semantic claim search |
| `POST` | `/api/embeddings/process` | Process pending embeddings |
| **Planning** | | |
| `POST` | `/api/planning/strategies` | Generate strategy candidates |
| `POST` | `/api/planning/actions` | Generate action candidates |
| `POST` | `/api/planning/plan` | Full planning pipeline for a goal |
| `POST` | `/api/planning/execute` | Start execution tracking |
| `POST` | `/api/planning/validate` | Validate event against prediction |
| `GET` | `/api/world-model/stats` | World model statistics |
| **Admin** | | |
| `POST` | `/api/admin/export` | Export database (streaming binary) |
| `POST` | `/api/admin/import` | Import database (replace/merge) |
| `GET` | `/api/health` | Health check |

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NER_SERVICE_URL` | `http://localhost:8081/ner` | NER service URL (full URL with protocol) |
| `NER_REQUEST_TIMEOUT_MS` | `5000` | NER request timeout |
| `LLM_API_KEY` | - | OpenAI-compatible API key for claims + planning |
| `LLM_MODEL` | `gpt-4o-mini` | LLM model name |
| `PLANNING_LLM_API_KEY` | falls back to `LLM_API_KEY` | Separate key for planning |
| `PLANNING_LLM_PROVIDER` | `openai` | LLM provider |
| `SERVER_HOST` | `0.0.0.0` | Server bind address |
| `SERVER_PORT` | `8080` | Server port |
| `RUST_LOG` | `info` | Log level |
| `REDB_CACHE_SIZE_MB` | `256` | ReDB cache size |
| `MEMORY_CACHE_SIZE` | `10000` | In-memory cache size |
| `STRATEGY_CACHE_SIZE` | `5000` | Strategy cache size |
| `ENABLE_WORLD_MODEL` | `false` | Enable energy-based world model |
| `WORLD_MODEL_MODE` | `Disabled` | Shadow/ScoringOnly/ScoringAndReranking/Full |
| `ENABLE_STRATEGY_GENERATION` | `false` | Enable LLM strategy generation |
| `ENABLE_ACTION_GENERATION` | `false` | Enable LLM action generation |
| `REPAIR_ENABLED` | `false` | Enable auto-repair on prediction error |

### NER Service

The NER service runs separately (GPU-accelerated). The `NER_SERVICE_URL` must be a full URL with protocol:

```
NER_SERVICE_URL=http://192.168.1.100:8081/ner
```

## Development

```bash
# Build
cargo build

# Test (431+ tests across workspace)
cargo test --workspace

# Clippy
cargo clippy --workspace

# Format
cargo fmt --all
```

## Documentation

- **[API_REFERENCE.md](API_REFERENCE.md)** - Complete REST API with request/response schemas
- [TECHNICAL_SPECIFICATION.md](TECHNICAL_SPECIFICATION.md) - System architecture
- [MVP_SPECIFICATION.md](MVP_SPECIFICATION.md) - Requirements and targets
- [Gplan.md](Gplan.md) - Graph layer improvement plan (Phases 1-4 complete)

## License

MIT License - see [LICENSE](LICENSE) for details.
