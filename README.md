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
- **Code Intelligence** - Tree-sitter AST parsing, code graph building, structural code search
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
│   ├── agent-db-ast/       # Tree-sitter AST parsing (Rust, Python, TS, Go)
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
| **Code Intelligence** | | |
| `POST` | `/api/events/code-file` | Submit source file for AST analysis |
| `POST` | `/api/events/code-review` | Submit code review event |
| `POST` | `/api/code/search` | Search code entities by name/kind/file |
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

## Building a Claude Code Memory Plugin

EventGraphDB can serve as a persistent memory backend for Claude Code via the [Model Context Protocol (MCP)](https://modelcontextprotocol.io). This gives Claude long-term memory that persists across conversations — it remembers your codebase, decisions, preferences, bugs, and patterns.

### How It Works

```
Claude Code ──MCP──▶ MCP Server (thin wrapper)
                           │
                     HTTP calls to
                     EventGraphDB API
                           │
              ┌────────────┼────────────┐
              ▼            ▼            ▼
         store_memory  recall_memory  index_code
              │            │            │
              ▼            ▼            ▼
         POST /api/    POST /api/   POST /api/
         events/simple  claims/search events/code-file
              │            │            │
              └────────────┼────────────┘
                           ▼
                    EventGraphDB
                  (graph, claims,
                   episodes, decay)
```

1. Claude Code calls MCP tools like `store_memory`, `recall_memory`, `index_code`
2. Your MCP server translates these into EventGraphDB REST API calls
3. EventGraphDB processes them through the full pipeline — graph construction, claim extraction, episode detection, memory consolidation
4. When Claude recalls, it gets semantically ranked results with confidence scores and temporal decay built in

### What Makes This Different From Flat File Memory

| Feature | Flat files (CLAUDE.md) | EventGraphDB |
|---------|----------------------|--------------|
| Storage | Plain text, manual | Structured graph with relationships |
| Search | Substring / none | BM25 + vector hybrid semantic search |
| Decay | Never expires | Half-life decay (facts: 2yr, bugs: 90d) |
| Relationships | None | Entity graph with typed edges |
| Code awareness | None | AST-parsed function/struct/trait graph |
| Cross-session | Single file | Full event history with episodes |
| Conflicting info | Last write wins | Claim confidence + temporal scoring |

### MCP Server Setup

You need a thin MCP server that wraps EventGraphDB's REST API. This can be an HTTP server (for teams) or a stdio process (for local use).

#### Option A: HTTP Transport (recommended for teams)

Create a `.mcp.json` in your project root:

```json
{
  "mcpServers": {
    "eventgraphdb": {
      "type": "http",
      "url": "http://localhost:3001/mcp"
    }
  }
}
```

Or add it via CLI:

```bash
claude mcp add --transport http eventgraphdb http://localhost:3001/mcp
```

#### Option B: Stdio Transport (local dev)

```bash
claude mcp add --transport stdio eventgraphdb -- node ./mcp-server/index.js
```

### MCP Tool Definitions

Your MCP server should expose these tools. Each one maps to one or more EventGraphDB API calls.

#### `store_memory` — Remember something

Stores a fact, decision, preference, or observation. Maps to `POST /api/events/simple`.

```json
{
  "name": "store_memory",
  "description": "Store a memory that persists across conversations. Use for facts, decisions, user preferences, bug reports, architectural choices, or anything worth remembering.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "content": {
        "type": "string",
        "description": "What to remember (a clear, specific statement)"
      },
      "category": {
        "type": "string",
        "enum": ["fact", "preference", "decision", "bug", "pattern", "api_contract"],
        "description": "Type of memory"
      },
      "tags": {
        "type": "array",
        "items": { "type": "string" },
        "description": "Tags for categorization (e.g. ['auth', 'backend', 'rust'])"
      }
    },
    "required": ["content"]
  }
}
```

**MCP server implementation** — translates to:

```bash
POST /api/events/simple
{
  "agent_id": 1,
  "agent_type": "claude-code",
  "session_id": <session_hash>,
  "event_type": "observation",
  "content": "User prefers async/await over callback patterns",
  "enable_semantic": true
}
```

EventGraphDB then extracts claims, detects entities, builds graph relationships, and indexes everything for semantic search — automatically.

#### `recall_memory` — Search for memories

Retrieves relevant memories using hybrid semantic search. Maps to `POST /api/claims/search` and `POST /api/search`.

```json
{
  "name": "recall_memory",
  "description": "Search persistent memory for relevant information. Returns memories ranked by relevance and recency. Use before making decisions, to check for prior context, or when the user references something from a previous session.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "query": {
        "type": "string",
        "description": "What to search for (natural language)"
      },
      "category": {
        "type": "string",
        "enum": ["fact", "preference", "decision", "bug", "pattern", "api_contract"],
        "description": "Filter by memory type"
      },
      "limit": {
        "type": "integer",
        "description": "Max results (default: 10)"
      }
    },
    "required": ["query"]
  }
}
```

**MCP server implementation** — translates to:

```bash
POST /api/claims/search
{
  "query": "authentication error handling",
  "limit": 10,
  "min_confidence": 0.3
}
```

Returns claims ranked by `combined_score` (semantic similarity * confidence * temporal decay).

#### `index_code` — Parse and remember code structure

Submits a source file for AST analysis. EventGraphDB parses it with tree-sitter, extracts functions/structs/traits/enums, builds a code graph, and extracts structural claims. Maps to `POST /api/events/code-file`.

```json
{
  "name": "index_code",
  "description": "Index a source file into persistent memory. Parses the AST to extract functions, structs, traits, enums, and their relationships. Use when exploring a new codebase or after significant refactors.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "file_path": {
        "type": "string",
        "description": "Path to the source file"
      },
      "content": {
        "type": "string",
        "description": "Full source code content"
      },
      "language": {
        "type": "string",
        "description": "Language (auto-detected from extension if omitted)"
      },
      "repository": {
        "type": "string",
        "description": "Repository name for scoping"
      }
    },
    "required": ["file_path", "content"]
  }
}
```

**MCP server implementation** — translates to:

```bash
POST /api/events/code-file
{
  "agent_id": 1,
  "agent_type": "claude-code",
  "session_id": <session_hash>,
  "file_path": "src/auth/login.rs",
  "content": "<file contents>",
  "language": "rust",
  "repository": "my-app",
  "enable_ast": true,
  "enable_semantic": true
}
```

This creates Concept nodes for every function, struct, trait, and enum in the file — with edges for contains, imports, field_of, returns, and implements relationships. Claude can later search these with `search_code`.

#### `search_code` — Find code entities

Searches the code graph by name, kind, language, or file path. Maps to `POST /api/code/search`.

```json
{
  "name": "search_code",
  "description": "Search indexed code entities (functions, structs, traits, enums). Use to find function signatures, locate definitions, or understand code structure.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "name": {
        "type": "string",
        "description": "Name substring to search for"
      },
      "kind": {
        "type": "string",
        "enum": ["function", "class", "enum", "interface", "module", "variable", "typealias"],
        "description": "Filter by entity kind"
      },
      "file_path": {
        "type": "string",
        "description": "Filter by file path substring"
      },
      "language": {
        "type": "string",
        "description": "Filter by language"
      }
    }
  }
}
```

#### `store_review` — Remember code review feedback

Stores a code review comment with file/line context. Maps to `POST /api/events/code-review`.

```json
{
  "name": "store_review",
  "description": "Store a code review observation — bug found, pattern noted, improvement suggested. Attached to specific file and line range.",
  "inputSchema": {
    "type": "object",
    "properties": {
      "body": {
        "type": "string",
        "description": "Review comment text"
      },
      "file_path": {
        "type": "string",
        "description": "File being reviewed"
      },
      "line_range": {
        "type": "array",
        "items": { "type": "integer" },
        "description": "[start_line, end_line]"
      },
      "action": {
        "type": "string",
        "enum": ["comment", "approve", "request_changes"],
        "description": "Review action type"
      },
      "repository": {
        "type": "string",
        "description": "Repository name"
      }
    },
    "required": ["body", "repository"]
  }
}
```

### Example MCP Server (Node.js)

A minimal MCP server that wraps EventGraphDB:

```javascript
import { McpServer } from "@modelcontextprotocol/sdk/server/mcp.js";
import { StreamableHTTPServerTransport } from "@modelcontextprotocol/sdk/server/streamableHttp.js";
import express from "express";
import { z } from "zod";

const EVENTGRAPHDB_URL = process.env.EVENTGRAPHDB_URL || "http://localhost:8080";
const AGENT_ID = 1;
const SESSION_ID = Date.now();

const server = new McpServer({
  name: "eventgraphdb-memory",
  version: "1.0.0",
});

// store_memory tool
server.tool(
  "store_memory",
  "Store a memory that persists across conversations",
  {
    content: z.string().describe("What to remember"),
    category: z.enum(["fact", "preference", "decision", "bug", "pattern"]).optional(),
    tags: z.array(z.string()).optional(),
  },
  async ({ content, category, tags }) => {
    const res = await fetch(`${EVENTGRAPHDB_URL}/api/events/simple`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        agent_id: AGENT_ID,
        agent_type: "claude-code",
        session_id: SESSION_ID,
        event_type: "observation",
        content: `[${category || "fact"}] ${content}` + (tags ? ` #${tags.join(" #")}` : ""),
        enable_semantic: true,
      }),
    });
    const data = await res.json();
    return { content: [{ type: "text", text: `Stored (event_id: ${data.event_id})` }] };
  }
);

// recall_memory tool
server.tool(
  "recall_memory",
  "Search persistent memory for relevant information",
  {
    query: z.string().describe("What to search for"),
    limit: z.number().optional().default(10),
  },
  async ({ query, limit }) => {
    const res = await fetch(`${EVENTGRAPHDB_URL}/api/claims/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ query, limit, min_confidence: 0.3 }),
    });
    const data = await res.json();
    const memories = (data.claims || [])
      .map((c, i) => `${i + 1}. [${c.claim_type}] ${c.subject}: ${c.claim} (confidence: ${c.confidence?.toFixed(2)})`)
      .join("\n");
    return { content: [{ type: "text", text: memories || "No memories found." }] };
  }
);

// index_code tool
server.tool(
  "index_code",
  "Index a source file into persistent memory via AST parsing",
  {
    file_path: z.string().describe("Path to the source file"),
    content: z.string().describe("Full source code"),
    language: z.string().optional(),
    repository: z.string().optional(),
  },
  async ({ file_path, content, language, repository }) => {
    const res = await fetch(`${EVENTGRAPHDB_URL}/api/events/code-file`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        agent_id: AGENT_ID,
        agent_type: "claude-code",
        session_id: SESSION_ID,
        file_path,
        content,
        language,
        repository,
        enable_ast: true,
        enable_semantic: true,
      }),
    });
    const data = await res.json();
    return {
      content: [{ type: "text", text: `Indexed ${file_path}: ${data.nodes_created} nodes created` }],
    };
  }
);

// search_code tool
server.tool(
  "search_code",
  "Search indexed code entities",
  {
    name: z.string().optional(),
    kind: z.string().optional(),
    file_path: z.string().optional(),
    language: z.string().optional(),
  },
  async (params) => {
    const res = await fetch(`${EVENTGRAPHDB_URL}/api/code/search`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ ...params, limit: 20 }),
    });
    const data = await res.json();
    const results = (data.entities || [])
      .map((e) => `${e.kind} ${e.qualified_name} (${e.file_path}:${e.line_range?.[0] || "?"})${e.signature ? "\n  " + e.signature : ""}`)
      .join("\n");
    return { content: [{ type: "text", text: results || "No code entities found." }] };
  }
);

// Start HTTP transport
const app = express();
const transport = new StreamableHTTPServerTransport({ sessionIdGenerator: undefined });
app.use("/mcp", transport.requestHandler);
server.connect(transport);

const port = process.env.MCP_PORT || 3001;
app.listen(port, () => console.log(`EventGraphDB MCP server on port ${port}`));
```

Run it:

```bash
npm install @modelcontextprotocol/sdk express zod
EVENTGRAPHDB_URL=http://localhost:8080 node mcp-server.js
```

### Usage In Practice

Once configured, Claude Code automatically discovers and uses the tools:

**Storing memories** — Claude calls `store_memory` when it learns something worth persisting:
```
User: "We always use sqlx for database access, never diesel"
Claude: [calls store_memory] Noted — I'll remember that for future sessions.
```

**Recalling context** — Claude calls `recall_memory` when it needs prior context:
```
User: "Set up the database layer for the new service"
Claude: [calls recall_memory("database library preference")]
        → "preference: Use sqlx for database access, not diesel (confidence: 0.95)"
        I see from our history that you prefer sqlx. I'll set that up.
```

**Indexing code** — Claude calls `index_code` to build structural understanding:
```
User: "Learn the auth module"
Claude: [reads src/auth/*.rs]
        [calls index_code for each file]
        Indexed 4 files — found 12 functions, 3 structs, 2 traits, and their relationships.
```

**Searching code** — Claude queries the structural code graph:
```
User: "What functions handle token validation?"
Claude: [calls search_code(name: "token", kind: "function")]
        → Function auth::tokens::validate_token (src/auth/tokens.rs:45)
        → Function auth::middleware::check_token (src/auth/middleware.rs:12)
```

### Memory Lifecycle

EventGraphDB doesn't just store memories — it manages them over time:

```
New memory stored
    ↓
Claim extracted with type + confidence
    ↓
Indexed for semantic search (BM25 + embeddings)
    ↓
Linked to entity graph (relationships emerge)
    ↓
Half-life decay over time:
  Facts: 730 days       Preferences: 365 days
  Code patterns: 365d   API contracts: 180 days
  Bug fixes: 90 days    Intentions: 30 days
    ↓
Low-confidence claims fade naturally
Reinforced claims stay strong
Contradicting claims compete on score
```

### Hooks Integration

You can use Claude Code [hooks](https://docs.anthropic.com/en/docs/claude-code/hooks) to automatically index code on file changes:

```json
// .claude/settings.json
{
  "hooks": {
    "PostToolUse": [
      {
        "matcher": "Write|Edit",
        "command": "curl -s -X POST http://localhost:8080/api/events/code-file -H 'Content-Type: application/json' -d \"$(jq -n --arg fp \"$TOOL_INPUT_FILE_PATH\" --rawfile content \"$TOOL_INPUT_FILE_PATH\" '{agent_id:1, agent_type:\"claude-code\", session_id:1, file_path:$fp, content:$content, enable_ast:true}')\""
      }
    ]
  }
}
```

This automatically re-indexes any file Claude writes or edits, keeping the code graph up to date.

## Documentation

- **[API_REFERENCE.md](API_REFERENCE.md)** - Complete REST API with request/response schemas
- [TECHNICAL_SPECIFICATION.md](TECHNICAL_SPECIFICATION.md) - System architecture
- [MVP_SPECIFICATION.md](MVP_SPECIFICATION.md) - Requirements and targets
- [Gplan.md](Gplan.md) - Graph layer improvement plan (Phases 1-4 complete)

## License

MIT License - see [LICENSE](LICENSE) for details.
