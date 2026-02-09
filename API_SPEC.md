# EventGraphDB API Specification

EventGraphDB is a self-evolving agentic database that transforms discrete events into a semantic graph. This specification details the REST API for ingesting events, retrieving memories, querying strategies, and searching the learned graph.

## Base URL
Default: `http://localhost:3000`

## Core Concepts
- **Events**: Atomic units of interaction (Action, Observation, Communication, Context, Cognitive, Learning).
- **Episodes**: Automatically detected sequences of related events.
- **Memories**: Long-term storage of significant episodes — available in three tiers (Episodic → Semantic → Schema).
- **Strategies**: Learned behavioral patterns with executable playbooks, failure modes, and counterfactual analysis.
- **Claims**: Atomic facts extracted from events via semantic memory (NER + LLM + embeddings).
- **Consolidation**: Background process that promotes episodic memories into generalized semantic and schema memories.
- **Refinement**: Optional LLM-powered post-processing that polishes summaries and generates embeddings for semantic retrieval.

---

## 1. Events

### Post Event
Ingests a new event into the graph engine for processing, memory formation, strategy extraction, and optional semantic claim extraction.

**Endpoint:** `POST /api/events`

**Request Body:**
```json
{
  "event": {
    "id": "u128-uuid-string",
    "timestamp": 1234567890,
    "agent_id": 1,
    "agent_type": "assistant",
    "session_id": 101,
    "event_type": {
      "Action": {
        "action_name": "search_docs",
        "parameters": { "query": "rust async" },
        "outcome": { "Success": { "result": "Found 12 results" } }
      }
    },
    "causality_chain": [],
    "context": {
      "fingerprint": 987654321,
      "active_goals": [
        {
          "id": 12345,
          "description": "Help user debug code",
          "priority": 0.9,
          "progress": 0.4
        }
      ],
      "resources": {
        "computational": {
          "cpu_percent": 0,
          "memory_bytes": 0,
          "storage_bytes": 0,
          "network_bandwidth": 0
        },
        "external": {}
      }
    },
    "metadata": {}
  },
  "enable_semantic": true
}
```

> **`enable_semantic`**: When `true`, the event text is extracted and sent through the claims pipeline (NER → LLM claim extraction → embedding). This applies to **all** event types — Action, Observation, Context, Communication, Cognitive, and Learning. Events with low NER confidence are gated by the `ner_promotion_threshold`.

**Response:**
```json
{
  "success": true,
  "nodes_created": 5,
  "patterns_detected": 1,
  "processing_time_ms": 45
}
```

### Post Simple Event
A convenience endpoint that accepts a flattened event structure.

**Endpoint:** `POST /api/events/simple`

**Request Body:**
```json
{
  "agent_id": 1,
  "session_id": 101,
  "event_type": "Action",
  "action_name": "search_docs",
  "content": "Searching documentation for Rust async patterns",
  "outcome": "Success",
  "enable_semantic": true,
  "metadata": {}
}
```

**Response:** Same as `POST /api/events`.

---

## 2. Memories

Memories represent learned experiences from significant episodes. They are organized into a **three-tier hierarchy**:

| Tier | Description | Created By |
|------|-------------|------------|
| **Episodic** | Specific experiences (what happened in one episode) | Automatic — after each significant episode |
| **Semantic** | Generalized knowledge (patterns across 3+ similar episodes) | Consolidation engine |
| **Schema** | Reusable mental models (high-level principles from 3+ semantics) | Consolidation engine |

### Memory Response Object

```json
{
  "id": 5,
  "agent_id": 2001,
  "session_id": 1770628389617,

  "summary": "Agent helped user find and book a movie. Searched docs for 'rust async', received 12 results successfully. Goal 'Help the user find and book a movie' was active at priority 1.00.",
  "takeaway": "Repeating the action sequence search_docs → assistant_message → semantic_extraction led to a successful outcome.",
  "causal_note": "The episode succeeded. Key factor: the action 'search_docs' produced the expected result.",

  "tier": "Episodic",
  "consolidation_status": "Active",

  "strength": 0.999,
  "relevance_score": 0.784,
  "access_count": 0,
  "formed_at": "1770628412031346421",
  "last_accessed": "1770628412031346421",
  "context_hash": 3677117734126165000,
  "context": {
    "environment": {
      "variables": {
        "user_id": "jonathan@example.com",
        "intent_type": "inform"
      }
    },
    "active_goals": [
      {
        "id": 738332,
        "description": "Help the user find and book a movie",
        "priority": 1,
        "progress": 0
      }
    ],
    "resources": { "computational": { "cpu_percent": 0 }, "external": {} },
    "fingerprint": 3677117734126165000
  },
  "outcome": "Success",
  "memory_type": "Episodic"
}
```

#### LLM-Retrievable Fields

| Field | Type | Description |
|-------|------|-------------|
| `summary` | `string` | Natural language summary of what happened — readable by both humans and LLMs. |
| `takeaway` | `string` | The single most important lesson from this experience. |
| `causal_note` | `string` | Why it succeeded or failed — identifies the key causal factors. |
| `tier` | `string` | One of `"Episodic"`, `"Semantic"`, or `"Schema"`. |
| `consolidation_status` | `string` | One of `"Active"`, `"Consolidated"`, or `"Archived"`. |
| `schema_id` | `u64?` | If consolidated into a schema, the parent schema memory ID. |
| `consolidated_from` | `u64[]` | For Semantic/Schema memories, the IDs of memories that were merged. |

> **Note:** `summary`, `takeaway`, and `causal_note` are auto-generated from event data at creation time. When LLM refinement is enabled (`refinement_config.enable_llm_refinement: true`), these fields are asynchronously polished by gpt-4o-mini for higher quality prose.

### Get Agent Memories
Retrieves long-term memories for a specific agent, sorted by strength.

**Endpoint:** `GET /api/memories/agent/:agent_id?limit=10`

**Response:** `MemoryResponse[]`

### Get Contextual Memories
Finds memories similar to a provided environmental context.

**Endpoint:** `POST /api/memories/context`

**Request Body:**
```json
{
  "context": {
    "fingerprint": 987654321,
    "active_goals": [],
    "resources": { "computational": {}, "external": {} },
    "environment": {
      "variables": { "intent_type": "search" }
    }
  },
  "limit": 5,
  "min_similarity": 0.8,
  "agent_id": 2001,
  "session_id": null
}
```

**Response:** `MemoryResponse[]` — ordered by relevance to the provided context.

---

## 3. Strategies

Strategies are learned behavioral patterns. Each strategy includes an LLM-readable summary, executable playbook, known failure modes, and counterfactual analysis.

### Strategy Response Object

```json
{
  "id": 4,
  "name": "strategy_2001_ep_16",
  "agent_id": 2001,

  "summary": "Positive strategy for goal bucket 703385. Observed action sequence: search_docs → assistant_message → semantic_extraction. Expected success: 80.0%, expected cost: 4.0. Succeeded 1 time(s) out of 1 attempt(s).",
  "when_to_use": "Use when: goal bucket is 703385, context matches one of 4 known contexts, and the agent needs to perform search_docs.",
  "when_not_to_use": "Avoid when: context is novel (no matching fingerprint), goal bucket differs from 703385, or prior attempts with this sequence have failed recently.",
  "failure_modes": [
    "search_docs returns empty results → recovery: broaden search query",
    "API timeout on semantic_extraction → recovery: retry with exponential backoff"
  ],
  "playbook": [
    {
      "step": 1,
      "action": "search_docs",
      "condition": "goal_bucket == 703385",
      "skip_if": "",
      "recovery": "If no results, broaden the query terms",
      "branches": []
    },
    {
      "step": 2,
      "action": "assistant_message",
      "condition": "",
      "skip_if": "search returned 0 results",
      "recovery": "",
      "branches": [
        {
          "condition": "results.length > 10",
          "action": "summarize_results first, then present"
        }
      ]
    },
    {
      "step": 3,
      "action": "semantic_extraction",
      "condition": "",
      "skip_if": "",
      "recovery": "Retry with backoff",
      "branches": []
    }
  ],
  "counterfactual": "If the agent had skipped search_docs and gone straight to assistant_message, success probability would be lower (estimated ~40% based on non-search episodes).",
  "supersedes": [],
  "applicable_domains": [],

  "quality_score": 0.725,
  "success_count": 1,
  "failure_count": 0,
  "reasoning_steps": [],
  "strategy_type": "Positive",
  "support_count": 1,
  "expected_success": 0.8,
  "expected_cost": 4.0,
  "expected_value": 0.8,
  "confidence": 0.736,
  "goal_bucket_id": 703385,
  "behavior_signature": "d7a047db5e05c126",
  "precondition": "goal_bucket=703385 contexts=4",
  "action_hint": "repeat sequence: Act:search_docs > Context:assistant_message > Context:semantic_extraction"
}
```

#### LLM-Retrievable Fields

| Field | Type | Description |
|-------|------|-------------|
| `summary` | `string` | Natural language summary of the strategy. |
| `when_to_use` | `string` | Specific conditions where this strategy applies. |
| `when_not_to_use` | `string` | Conditions where this strategy should NOT be used. |
| `failure_modes` | `string[]` | Known failure modes with recovery hints. |
| `playbook` | `PlaybookStep[]` | Executable steps with branching and recovery logic. |
| `counterfactual` | `string` | What would have happened with a different approach. |
| `supersedes` | `u64[]` | IDs of strategies this one replaces (version lineage). |
| `applicable_domains` | `string[]` | Cross-domain applicability tags. |

#### PlaybookStep Object

| Field | Type | Description |
|-------|------|-------------|
| `step` | `u32` | Step number (1-indexed). |
| `action` | `string` | What to do at this step. |
| `condition` | `string` | Prerequisite condition (when to execute). |
| `skip_if` | `string` | Condition under which this step should be skipped. |
| `recovery` | `string` | Recovery instruction if this step fails. |
| `branches` | `PlaybookBranch[]` | Conditional alternative actions. |

#### PlaybookBranch Object

| Field | Type | Description |
|-------|------|-------------|
| `condition` | `string` | The condition that triggers this branch. |
| `action` | `string` | What to do when this condition is met. |

### Get Agent Strategies
Retrieves strategies for a specific agent.

**Endpoint:** `GET /api/strategies/agent/:agent_id?limit=10`

**Response:** `StrategyResponse[]`

### Find Similar Strategies
Finds strategies matching a multi-dimensional similarity query.

**Endpoint:** `POST /api/strategies/similar`

**Request Body:**
```json
{
  "goal_ids": [703385],
  "tool_names": ["search_docs"],
  "result_types": ["Success"],
  "context_hash": 3677117734126165,
  "agent_id": 2001,
  "min_score": 0.3,
  "limit": 5
}
```

**Response:** `SimilarStrategyResponse[]` — includes all `StrategyResponse` fields plus a `score` field.

### Get Action Suggestions (Policy Guide)
Returns the best next action based on the current context and learned strategies.

**Endpoint:** `GET /api/suggestions?context_hash=987654321&limit=3`

**Response:**
```json
[
  {
    "action_name": "read_file",
    "success_probability": 0.92,
    "evidence_count": 15,
    "reasoning": "Observed high success rate in similar coding contexts."
  }
]
```

---

## 4. Semantic Memory (Claims)

Claims are atomic facts extracted from events via the NER → LLM → Embedding pipeline. When `enable_semantic: true` is sent with an event, the pipeline extracts structured claims from all event types (Action, Observation, Context, Communication, Cognitive, Learning).

### Search Claims
Performs a semantic similarity search over extracted claims using vector embeddings.

**Endpoint:** `POST /api/claims/search`

**Request Body:**
```json
{
  "query_text": "Who is the project manager?",
  "top_k": 3,
  "min_similarity": 0.75
}
```

**Response:**
```json
[
  {
    "claim_id": 1,
    "subject": "Alice",
    "predicate": "is",
    "object": "project manager",
    "confidence": 0.95,
    "source_event_id": "123456789",
    "similarity": 0.89
  }
]
```

> **Implementation detail:** Claims are indexed in an in-memory vector index with L2-normalized embeddings for fast dot-product similarity search — not brute-force O(n).

---

## 5. Graph & Analytics

### Get Graph Structure
Returns a subset of the graph for visualization.

**Endpoint:** `GET /api/graph?limit=100&session_id=101`

### Get Analytics
Returns high-level graph metrics and learning progress.

**Endpoint:** `GET /api/analytics`

**Response:**
```json
{
  "node_count": 1250,
  "edge_count": 4800,
  "community_count": 12,
  "learning_metrics": {
    "overall_success_rate": 0.88,
    "learned_patterns": 42
  }
}
```

### Get System Stats
Returns live aggregate counts of all stores — memories, strategies, and claims.

**Endpoint:** `GET /api/stats`

**Response:**
```json
{
  "uptime_seconds": 3600,
  "events_processed": 1042,
  "active_episodes": 3,
  "stores": {
    "memory": {
      "total_memories": 87,
      "episodic_count": 65,
      "active_agents": 4
    },
    "strategy": {
      "total_strategies": 23,
      "positive_count": 18,
      "negative_count": 5,
      "active_agents": 4
    },
    "claims": {
      "total_claims": 156,
      "verified_claims": 42,
      "pending_claims": 114,
      "indexed_embeddings": 89
    }
  }
}
```

---

## 6. Health

### Health Check
**Endpoint:** `GET /api/health`

**Response:**
```json
{
  "status": "ok",
  "version": "0.1.0",
  "uptime_seconds": 3600
}
```

---

## 7. Configuration & Architecture

### Memory Consolidation
The consolidation engine runs automatically in the background (configurable interval). It:
1. Groups episodic memories by agent + context similarity.
2. When 3+ episodic memories share patterns → synthesizes a **Semantic** memory.
3. When 3+ semantic memories share patterns → synthesizes a **Schema** memory.
4. Consolidated episodic memories are marked `"Consolidated"` and decay faster, while their semantic/schema parents persist.

### LLM Refinement
When enabled, newly formed memories and strategies are asynchronously sent to `gpt-4o-mini` for:
- Polished natural language summaries
- Refined causal analysis
- Embedding generation for semantic retrieval

This is **non-blocking** — the template summary is available immediately, and the LLM-refined version replaces it in the background.

### Numeric Field Flexibility
All numeric ID fields (agent_id, session_id, goal_id, timestamps, etc.) accept **both JSON numbers and JSON strings**. This ensures compatibility with JavaScript SDKs that may serialize large numbers as strings.

```json
// Both are accepted:
{ "agent_id": 2001 }
{ "agent_id": "2001" }
```

### Semantic Event Processing
When `enable_semantic: true`, the following event types have their text extracted for claim processing:

| Event Type | Text Extracted From |
|------------|-------------------|
| `Context` | `content` field |
| `Action` | Action name + parameters + outcome result |
| `Observation` | Observation source + data |
| `Communication` | Sender → receiver: content |
| `Cognitive` | Cognitive type description |
| `Learning` | Outcome query + success status |

Events with NER confidence below the `ner_promotion_threshold` (default: 0.5) are filtered out to reduce noise.

---

## 8. Docker Deployment

```bash
docker run -d \
  -p 3000:3000 \
  -e OPENAI_API_KEY=sk-... \
  -v eventgraphdb_data:/data \
  eventgraphdb:latest
```

- **`OPENAI_API_KEY`**: Required for semantic memory (claims extraction + embeddings) and LLM refinement.
- **`-v eventgraphdb_data:/data`**: Persists all memories, strategies, claims, and graph data across container restarts and image upgrades.
- Graceful shutdown is handled automatically — `SIGTERM` / `Ctrl+C` flushes all in-flight data before exit.
