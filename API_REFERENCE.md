# MinnsDB API Reference

**Complete REST API documentation for SDK integration**

Version: 0.5.0
Last Updated: 2026-03-18

---

## Table of Contents

1. [Architecture](#architecture)
2. [Overview](#overview)
3. [Core Types](#core-types)
4. [Events](#events)
5. [Memories](#memories)
6. [Strategies](#strategies)
7. [Graph](#graph)
8. [Analytics](#analytics)
9. [Search](#search)
10. [Natural Language Query (NLQ)](#natural-language-query-nlq)
11. [MinnsQL Structured Query](#minnsql-structured-query)
12. [Reactive Subscriptions](#reactive-subscriptions)
13. **[Temporal Tables](#temporal-tables)** *(new in 0.5)*
14. [Claims](#claims)
15. [Structured Memory](#structured-memory)
16. [Conversation Ingestion](#conversation-ingestion)
17. [Single Message Ingestion](#single-message-ingestion)
18. [Workflows](#workflows)
19. [Agent Registry](#agent-registry)
20. [Ontology Evolution](#ontology-evolution)
21. **[WASM Agent Modules](#wasm-agent-modules)** *(new in 0.5)*
22. [Planning & World Model](#planning--world-model)
23. [Code Intelligence](#code-intelligence)
24. [Admin](#admin)
25. [Health](#health)
26. [Error Handling](#error-handling)
27. [Configuration](#configuration)

---

## Architecture

MinnsDB has a unified pipeline. All data — structured events and conversations — flows through the same graph engine.

### Unified Event Pipeline

Every event flows through the full pipeline:

```
Event --> Graph Construction --> Episode Detection --> Memory Formation --> Strategy Extraction
                                       |
                                       +--> Reinforcement Learning (edge weights, Q-values)
                                       +--> World Model Training (EBM contrastive learning)
                                       +--> Claims Extraction (LLM-driven entity/fact extraction)
```

**Entry points:**
- `POST /api/events`, `POST /api/events/simple`, `POST /api/events/state-change`, `POST /api/events/transaction` -- structured events with pre-populated metadata
- `POST /api/conversations/ingest` -- raw conversation text (converted to events, then enriched via LLM compaction)

**What it produces:**
- Graph nodes and edges (concepts, actions, observations, relationships)
- Episodes (bounded sequences of related events)
- Episodic, Semantic, and Schema memories (consolidation over time)
- Strategies (playbooks extracted from successful/failed episodes)
- Reinforcement signals (transition model, success/failure posteriors)

**Entity resolution:** Graph `concept_index` (`HashMap<Arc<str>, NodeId>`) -- node IDs assigned by the graph engine during inference.

### Conversation Ingestion

Conversations go through the same event pipeline as structured events, with an additional LLM compaction step that extracts structured knowledge from raw text.

```
ConversationIngest --> Bridge (raw Conversation events) --> Full Event Pipeline
                                                                 |
                                                                 +--> Graph nodes + episodes
                                                                 +--> LLM Compaction (inline)
                                                                        +--> LLM Cascade (facts, goals, summaries)
                                                                        +--> NER Financial Extraction (PAYER/AMOUNT/PAYEE)
                                                                        +--> write_fact_to_graph()
                                                                               +--> state:location, state:work edges
                                                                               +--> financial:payment edges
                                                                               +--> relationship:* edges
                                                                               +--> preference:* edges
```

**Bridge** (`ingest_to_events`): Converts each message into an `EventType::Conversation` event with `category: "message"`. These are intentionally lightweight -- the raw text is preserved, and all semantic extraction happens in compaction.

**LLM Compaction** (`run_compaction`): Runs inline before the response returns. Per-batch extraction with rolling context:
1. **LLM Cascade** (3-call pipeline): Entity extraction, relationship discovery, structured fact production. Each fact includes subject, predicate, object, category, temporal signals, dependencies, and financial amounts.
2. **NER Financial Extraction** (parallel via `tokio::join!`): Specialized accounting NER model with PAYER/AMOUNT/PAYEE labels extracts financial triplets directly. Deduped against LLM facts by normalized (subject, object, category).
3. **Two-phase graph writes**: Single-valued facts (location, work) written first to establish entity state, then multi-valued facts (routines, preferences) with auto-stamped `depends_on` from the updated graph state.

**Requires a configured LLM client.** Without it, the endpoint returns an error -- raw conversation events without compaction contain no structured information.

### Structured Events vs Conversations

| Aspect | Structured Events | Conversations |
|--------|-------------------|---------------|
| **Entry point** | `POST /api/events/*` | `POST /api/conversations/ingest` |
| **Metadata** | Pre-populated by caller | Empty (extracted by LLM compaction) |
| **Fact extraction** | Metadata normalization (alias + LLM fallback) | LLM cascade + NER |
| **State tracking** | Via metadata keys (`entity`, `new_state`, `amount`) | Via compaction and `write_fact_to_graph` |
| **LLM required** | No | Yes |
| **Graph edges** | Created by state tracking pipeline | Created by compaction |

Both paths create the same graph edge types (`state:*`, `financial:payment`, `relationship:*`, `preference:*`) and are queried identically at query time.

### Query-Time Graph Projection

The NLQ pipeline (`POST /api/nlq`) queries the graph directly:

```
Question --> LLM Intent Classifier --> Graph Projection (source of truth)
                  |                          +--> Entity state (successor-state walk)
                  |                          +--> Financial balances (net balance computation)
                  |                          +--> Temporal views (valid_until filtering)
                  |                          +--> Community detection (DRIFT search)
                  |
                  +--> Claims + BM25 + Embedding fusion (RRF)
                  +--> Memory Store (related_memories)
                  +--> Strategy Store (related_strategies)
```

The graph is the source of truth. Query-time projections walk edges dynamically rather than reading pre-computed values, so answers always reflect the latest state.


---

## Overview

**Base URL:** `http://localhost:8080`
**Content-Type:** `application/json` (except binary export/import)

All IDs are numeric:
- `AgentId`, `SessionId`, `NodeId`, `MemoryId`, `StrategyId`, `GoalBucketId` = `u64`
- `EventId` = `u128`
- `ContextHash` = `u64`
- `Timestamp` = `u64` (nanoseconds since Unix epoch)

---

## Core Types

### Event

```json
{
  "id": 340282366920938463463374607431768211455,
  "timestamp": 1735603200000000000,
  "agent_id": 1,
  "agent_type": "ai-debugger",
  "session_id": 42,
  "event_type": { "Action": { ... } },
  "causality_chain": [],
  "context": { ... },
  "metadata": {},
  "context_size_bytes": 0,
  "segment_pointer": null,
  "is_code": false
}
```

### EventType Variants

**Action:**
```json
{
  "Action": {
    "action_name": "fix_null_error",
    "parameters": {"fix": "add_null_check"},
    "outcome": {"Success": {"result": {"tests_pass": true}}},
    "duration_ns": 1500000000
  }
}
```

**Observation:**
```json
{
  "Observation": {
    "observation_type": "error_detected",
    "data": {"error": "TypeError"},
    "confidence": 0.95,
    "source": "runtime"
  }
}
```

**Cognitive:**
```json
{
  "Cognitive": {
    "process_type": "Reasoning",
    "input": {"error": "TypeError: undefined"},
    "output": {"analysis": "null reference"},
    "reasoning_trace": ["Analyze error type", "Identify root cause"]
  }
}
```

**Communication:**
```json
{
  "Communication": {
    "message_type": "request",
    "sender": 1,
    "recipient": 2,
    "content": {"message": "help needed"}
  }
}
```

**Learning (explicit telemetry):**
```json
{
  "Learning": {
    "event": {
      "MemoryUsed": {"query_id": "q1", "memory_id": 42}
    }
  }
}
```

Learning event sub-types:
- `MemoryRetrieved { query_id, memory_ids: [u64] }`
- `MemoryUsed { query_id, memory_id }`
- `StrategyServed { query_id, strategy_ids: [u64] }`
- `StrategyUsed { query_id, strategy_id }`
- `Outcome { query_id, success: bool }`
- `ClaimRetrieved { query_id, claim_ids: [u64] }`
- `ClaimUsed { query_id, claim_id }`

**Context (for NER processing):**
```json
{
  "Context": {
    "text": "Meeting with John at Google HQ on Friday",
    "context_type": "conversation",
    "language": "en"
  }
}
```

### ActionOutcome

```json
// Success
{"Success": {"result": {"tests_pass": true}}}

// Failure
{"Failure": {"error": "timeout", "error_code": 408}}

// Partial
{"Partial": {"result": {"partial": true}, "issues": ["incomplete"]}}
```

### CognitiveType

One of: `"GoalFormation"`, `"Planning"`, `"Reasoning"`, `"MemoryRetrieval"`, `"LearningUpdate"`

### EventContext

```json
{
  "environment": {
    "variables": {"lang": "rust", "project": "minnsdb"},
    "spatial": null,
    "temporal": {"time_of_day": null, "deadlines": [], "patterns": []}
  },
  "active_goals": [
    {
      "id": 1,
      "description": "Fix null reference bug",
      "priority": 0.9,
      "deadline": null,
      "progress": 0.0,
      "subgoals": []
    }
  ],
  "resources": {
    "computational": {"cpu_percent": 50.0, "memory_bytes": 8000000000, "storage_bytes": 100000000000, "network_bandwidth": 1000000},
    "external": {}
  },
  "fingerprint": 0,
  "goal_bucket_id": 0,
  "embeddings": null
}
```

`fingerprint` and `goal_bucket_id` are auto-computed if set to 0.

### Event.is_code

Set `"is_code": true` on events containing source code. This activates:
- **Code tokenizer:** camelCase/PascalCase splitting, snake_case splitting, qualified name recursion, operator tokenization
- **Code BM25 indexing:** Separate code-aware BM25 index with `search_code()` and `search_mixed()` queries
- **Code concept types:** `Function`, `Class`, `Module`, `Variable` node types extracted from code events
- **NLQ code routing:** Entity extraction recognizes code identifiers; BM25 searches route through the code index

```json
{
  "is_code": true,
  "event_type": {
    "Context": {
      "text": "fn process_event(&self, event: Event) -> Result<()> { ... }",
      "context_type": "code",
      "language": "rust"
    }
  }
}
```

The field defaults to `false` and is backward-compatible (older events without it are treated as natural language).

---

## Events

### POST /api/events

Process a full event through the pipeline (graph construction, episode detection, memory formation, strategy extraction, reinforcement learning).

**Request:**
```json
{
  "event": {
    "agent_id": 1,
    "agent_type": "ai-debugger",
    "session_id": 42,
    "event_type": {
      "Action": {
        "action_name": "fix_null_error",
        "parameters": {"fix": "add_null_check"},
        "outcome": {"Success": {"result": {"tests_pass": true}}},
        "duration_ns": 1500000000
      }
    },
    "causality_chain": [],
    "context": { "...": "..." },
    "metadata": {}
  },
  "enable_semantic": false
}
```

Set `enable_semantic: true` to run NER + claim extraction + embedding generation.

**Response (200):**
```json
{
  "success": true,
  "event_id": 340282366920938463463374607431768211455,
  "nodes_created": 2,
  "patterns_detected": 1,
  "processing_time_ms": 12
}
```

### POST /api/events/simple

Simplified event submission for quick integration.

**Request:**
```json
{
  "agent_id": 1,
  "agent_type": "ai-debugger",
  "session_id": 42,
  "action": "fix_null_error",
  "data": {"fix": "add_null_check", "result": "tests pass"},
  "success": true,
  "enable_semantic": false
}
```

**Response:** Same as `POST /api/events`.

### POST /api/events/state-change

Typed state-change event submission. The server maps the fields into event metadata with canonical keys (`entity`, `new_state`, `old_state`) and processes through the pipeline. The pipeline's metadata normalizer detects state changes automatically and updates structured memory state machines.

**Request:**
```json
{
  "agent_id": 1,
  "agent_type": "workflow-engine",
  "session_id": 42,
  "entity": "Order-123",
  "new_state": "shipped",
  "old_state": "processing",
  "trigger": "warehouse_confirmation",
  "extra_metadata": {"warehouse": "US-West"},
  "enable_semantic": false
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `agent_id` | integer | yes | Agent identifier |
| `agent_type` | string | yes | Agent type label |
| `session_id` | integer | yes | Session identifier |
| `entity` | string | yes | Entity whose state is changing |
| `new_state` | string | yes | Target state |
| `old_state` | string | no | Previous state (for history tracking) |
| `trigger` | string | no | What caused the transition |
| `extra_metadata` | object | no | Additional key-value pairs (canonical keys `entity`, `new_state`, `old_state` are ignored if duplicated here) |
| `enable_semantic` | boolean | no | Enable NER + claim extraction |

**Response:** Same as `POST /api/events`.

### POST /api/events/transaction

Typed transaction event submission. Maps fields into event metadata with canonical keys (`from`, `to`, `amount`, `direction`, `description`, `transaction`) and processes through the pipeline. The pipeline's metadata normalizer detects transactions and auto-appends to structured memory ledgers.

**Request:**
```json
{
  "agent_id": 1,
  "agent_type": "payment-service",
  "session_id": 42,
  "from": "Alice",
  "to": "Bob",
  "amount": 25.0,
  "direction": "Credit",
  "description": "Payment for services",
  "extra_metadata": {"invoice_id": "INV-456"},
  "enable_semantic": false
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `agent_id` | integer | yes | Agent identifier |
| `agent_type` | string | yes | Agent type label |
| `session_id` | integer | yes | Session identifier |
| `from` | string | yes | Source entity |
| `to` | string | yes | Destination entity |
| `amount` | number | yes | Transaction amount (must be finite) |
| `direction` | string | no | `"Credit"` (adds) or `"Debit"` (subtracts). Default: Credit |
| `description` | string | no | Human-readable description |
| `extra_metadata` | object | no | Additional key-value pairs (canonical keys `from`, `to`, `amount`, `direction`, `description`, `transaction` are ignored if duplicated here) |
| `enable_semantic` | boolean | no | Enable NER + claim extraction |

**Response:** Same as `POST /api/events`.

### GET /api/events?limit=10

**Response (200):**
```json
[
  {
    "id": 123456,
    "timestamp": 1735603200000000000,
    "agent_id": 1,
    "agent_type": "ai-debugger",
    "session_id": 42,
    "event_type": { "Action": { "..." : "..." } },
    "causality_chain": [],
    "context": { "..." : "..." },
    "metadata": {}
  }
]
```

### GET /api/episodes?limit=10

**Response (200):**
```json
[
  {
    "id": 1,
    "agent_id": 1,
    "event_count": 5,
    "significance": 0.87,
    "outcome": "Success"
  }
]
```

---

## Memories

### GET /api/memories/agent/:agent_id?limit=10

Get memories for a specific agent.

**Response (200):**
```json
[
  {
    "id": 42,
    "agent_id": 1,
    "session_id": 42,
    "summary": "Successfully fixed null reference by adding null check",
    "takeaway": "Always add null checks before dereferencing optional values",
    "causal_note": "The fix worked because the error was a simple null dereference",
    "tier": "Episodic",
    "consolidation_status": "Active",
    "schema_id": null,
    "consolidated_from": [],
    "strength": 0.85,
    "relevance_score": 0.92,
    "access_count": 3,
    "formed_at": 1735603200000000000,
    "last_accessed": 1735689600000000000,
    "context_hash": 12345,
    "context": { "..." : "..." },
    "outcome": "Success",
    "memory_type": "Episodic"
  }
]
```

**Memory tiers:** `"Episodic"` (raw experience), `"Semantic"` (generalized), `"Schema"` (reusable model)

**Consolidation status:** `"Active"`, `"Consolidated"`, `"Archived"`

### POST /api/memories/context

Find memories similar to a given context using cosine similarity.

**Request:**
```json
{
  "context": {
    "environment": { "variables": {"lang": "rust"}, "spatial": null, "temporal": {"time_of_day": null, "deadlines": [], "patterns": []} },
    "active_goals": [{"id": 1, "description": "Fix bug", "priority": 0.9, "deadline": null, "progress": 0.0, "subgoals": []}],
    "resources": { "computational": {"cpu_percent": 50.0, "memory_bytes": 8000000000, "storage_bytes": 100000000000, "network_bandwidth": 1000000}, "external": {} },
    "fingerprint": 0,
    "goal_bucket_id": 0,
    "embeddings": null
  },
  "limit": 10,
  "min_similarity": 0.6,
  "agent_id": 1,
  "session_id": null
}
```

**Response:** Same shape as `GET /api/memories/agent/:agent_id`.

---

## Strategies

### GET /api/strategies/agent/:agent_id?limit=10

Get strategies learned by a specific agent.

**Response (200):**
```json
[
  {
    "id": 1,
    "name": "null-check-fix",
    "agent_id": 1,
    "summary": "Add null checks before dereferencing optional values",
    "when_to_use": "When encountering null/undefined reference errors",
    "when_not_to_use": "When the value is guaranteed to exist by type system",
    "failure_modes": ["Over-aggressive null checking can hide real bugs"],
    "playbook": [
      {
        "step": 1,
        "action": "Identify the null reference location",
        "condition": "Error message contains null/undefined",
        "skip_if": "",
        "recovery": "Check stack trace for source",
        "branches": []
      },
      {
        "step": 2,
        "action": "Add null check or optional chaining",
        "condition": "Location identified",
        "skip_if": "Already has null check",
        "recovery": "Try alternative fix pattern",
        "branches": []
      }
    ],
    "counterfactual": "Without null check, the error would propagate to callers",
    "supersedes": [],
    "applicable_domains": ["debugging", "error-handling"],
    "quality_score": 0.92,
    "success_count": 15,
    "failure_count": 1,
    "reasoning_steps": [
      {
        "description": "Analyze error type",
        "applicability": "TypeError or NullPointerException",
        "expected_outcome": "Root cause identified",
        "sequence_order": 0
      }
    ],
    "strategy_type": "Positive",
    "support_count": 8,
    "expected_success": 0.94,
    "expected_cost": 0.1,
    "expected_value": 0.85,
    "confidence": 0.88,
    "goal_bucket_id": 1,
    "behavior_signature": "analyze→fix→verify",
    "precondition": "null reference error detected",
    "action_hint": "Add null check at error location"
  }
]
```

**Strategy types:** `"Positive"` (do this), `"Constraint"` (avoid this)

### POST /api/strategies/similar

Find strategies matching criteria using multi-dimensional similarity.

**Request:**
```json
{
  "goal_ids": [1, 2],
  "tool_names": ["compiler", "debugger"],
  "result_types": ["fix", "patch"],
  "context_hash": 12345,
  "agent_id": 1,
  "limit": 10,
  "min_score": 0.2
}
```

**Response (200):**
```json
[
  {
    "score": 0.87,
    "id": 1,
    "name": "null-check-fix",
    "...": "...same fields as strategy..."
  }
]
```

### GET /api/suggestions?context_hash=12345&limit=5

Get next-action suggestions based on historical patterns.

**Query Parameters:**
- `context_hash` (required): Current context fingerprint
- `last_action_node` (optional): Previous action node ID
- `limit` (optional, default 10): Max suggestions

**Response (200):**
```json
[
  {
    "action_name": "add_null_check",
    "success_probability": 0.94,
    "evidence_count": 47,
    "reasoning": "This action has worked well in similar contexts (47 times, 94.2% success)"
  },
  {
    "action_name": "add_type_guard",
    "success_probability": 0.87,
    "evidence_count": 24,
    "reasoning": "This action has followed the previous action 24 times with 87.5% success rate"
  }
]
```

---

## Graph

### GET /api/graph?limit=100

Get graph structure for visualization.

**Query Parameters:**
- `limit` (default 10): Max nodes
- `session_id` (optional): Filter by session
- `agent_type` (optional): Filter by agent type

**Response (200):**
```json
{
  "nodes": [
    {
      "id": 1,
      "label": "fix_null_error",
      "node_type": "Action",
      "created_at": 1735603200000000000,
      "properties": {"action_name": "fix_null_error", "success": true}
    }
  ],
  "edges": [
    {
      "id": 1,
      "from": 1,
      "to": 2,
      "edge_type": "Temporal",
      "weight": 1.0,
      "confidence": 0.95
    }
  ]
}
```

### GET /api/graph/context?context_hash=12345&limit=100

Same as above but filtered by context hash.

### POST /api/graph/persist

Force flush in-memory graph state to ReDB.

**Response (200):**
```json
{
  "success": true,
  "nodes_persisted": 1500,
  "edges_persisted": 3200
}
```

### GET /api/stats

**Response (200):**
```json
{
  "total_events_processed": 5000,
  "total_nodes_created": 12000,
  "total_episodes_detected": 450,
  "total_memories_formed": 120,
  "total_strategies_extracted": 35,
  "total_reinforcements_applied": 890,
  "average_processing_time_ms": 8.5,
  "stores": { "...": "..." }
}
```

---

## Analytics

### GET /api/analytics

Full graph analytics.

**Response (200):**
```json
{
  "node_count": 12000,
  "edge_count": 28000,
  "connected_components": 15,
  "largest_component_size": 11200,
  "average_path_length": 4.2,
  "diameter": 12,
  "clustering_coefficient": 0.34,
  "average_clustering": 0.28,
  "modularity": 0.65,
  "community_count": 8,
  "learning_metrics": {
    "total_events": 5000,
    "unique_contexts": 120,
    "learned_patterns": 89,
    "strong_memories": 45,
    "overall_success_rate": 0.78,
    "average_edge_weight": 1.35
  }
}
```

### GET /api/communities?algorithm=louvain

Detect communities. Algorithm: `"louvain"` or `"label_propagation"`.

**Response (200):**
```json
{
  "communities": [
    {"community_id": 0, "size": 450, "node_ids": [1, 2, 3, "..."]},
    {"community_id": 1, "size": 320, "node_ids": [100, 101, "..."]}
  ],
  "modularity": 0.65,
  "iterations": 12,
  "community_count": 8,
  "algorithm": "louvain"
}
```

### GET /api/centrality

Node centrality scores sorted by combined score.

**Response (200):**
```json
[
  {
    "node_id": 42,
    "degree": 0.15,
    "betweenness": 0.08,
    "closeness": 0.45,
    "eigenvector": 0.12,
    "pagerank": 0.003,
    "combined": 0.82
  }
]
```

### GET /api/ppr?source_node_id=42&limit=100&min_score=0.001

Personalized PageRank from a source node using random walks with restart.

**Response (200):**
```json
{
  "source_node_id": 42,
  "algorithm": "personalized_pagerank",
  "scores": [
    {"node_id": 43, "score": 0.15},
    {"node_id": 44, "score": 0.08}
  ]
}
```

### GET /api/reachability?source=42&max_hops=5&max_results=500

Temporal reachability: which nodes can be reached from source following edges forward in time.

**Response (200):**
```json
{
  "source_node_id": 42,
  "reachable_count": 150,
  "max_depth": 5,
  "edges_traversed": 320,
  "reachable": [
    {
      "node_id": 43,
      "origin": 42,
      "arrival_time": 1735603201000000000,
      "hops": 1,
      "predecessor": 42
    }
  ]
}
```

### GET /api/causal-path?source=42&target=99

Find the causal path between two nodes via temporal predecessor chain.

**Response (200):**
```json
{
  "source": 42,
  "target": 99,
  "found": true,
  "path": [42, 55, 67, 99]
}
```

### GET /api/indexes

Property index performance statistics.

**Response (200):**
```json
[
  {
    "insert_count": 12000,
    "query_count": 5000,
    "range_query_count": 200,
    "hit_count": 4800,
    "miss_count": 200,
    "last_accessed": 1735603200000000000
  }
]
```

---

## Search

### POST /api/search

Unified search with three modes.

**Request:**
```json
{
  "query": "null reference error fix",
  "mode": "Hybrid",
  "limit": 10,
  "fusion_strategy": "RRF"
}
```

**Modes:** `"Keyword"` (BM25), `"Semantic"` (embedding similarity), `"Hybrid"` (both fused)

**Fusion strategies (for Hybrid):** `"RRF"` (Reciprocal Rank Fusion), `"Linear"`, `"Max"`

**Response (200):**
```json
{
  "results": [
    {
      "node_id": 42,
      "score": 0.95,
      "node_type": "Action",
      "properties": {"action_name": "fix_null_error", "success": true}
    }
  ],
  "mode": "Hybrid",
  "total": 1
}
```

---

## Natural Language Query (NLQ)

Ask questions about the graph in plain English. The pipeline classifies intent, resolves entities, builds a graph query, executes it, and returns a human-readable answer.

When `ENABLE_NLQ_HINT=true`, an LLM advisory classifier runs alongside the rule-based classifier. It improves routing accuracy for ambiguous queries (especially structured memory types) while gracefully falling back to rules on any LLM failure.

### POST /api/nlq

**Request:**
```json
{
  "question": "What are the neighbors of Alice?",
  "limit": 10,
  "offset": 0,
  "session_id": "user-session-1"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `question` | string | yes | Natural language question |
| `limit` | integer | no | Pagination limit |
| `offset` | integer | no | Pagination offset |
| `session_id` | string | no | Session ID for conversational context (follow-up questions) |

**Response (200):**
```json
{
  "answer": "Neighbors of Alice: Bob (Association)",
  "intent": "FindNeighbors",
  "entities_resolved": [
    {
      "text": "Alice",
      "node_id": 1,
      "node_type": "Concept",
      "confidence": 0.95
    }
  ],
  "confidence": 0.87,
  "result_count": 1,
  "execution_time_ms": 5,
  "query_used": "NeighborsWithinDistance { source: 1, max_distance: 1, direction: Both }",
  "explanation": [
    "Intent classified as FindNeighbors",
    "1 entity mention(s) extracted",
    "Resolved 'Alice' -> Concept #1 (0.95)",
    "Matched template 'neighbors_any'",
    "Built query: NeighborsWithinDistance",
    "Query validated successfully"
  ],
  "total_count": 1
}
```

**Supported intent types:**
- `FindNeighbors` — "Who connects to X?", "neighbors of X"
- `FindPath` — "shortest path from A to B"
- `FilteredTraversal` — "What tools did agent X use?"
- `Subgraph` — "Show me everything about X"
- `TemporalChain` — "What happened after event X?"
- `Ranking` — "Most important nodes", "top nodes by PageRank"
- `SimilaritySearch` — "Find similar to X"
- `Aggregate` — "How many nodes?", "average degree", "sum of amount between A and B"
- `StructuredMemoryQuery` — "What is the balance between A and B?", "current state of X"

**Conversational follow-ups:** When `session_id` is provided, the pipeline maintains context (up to 5 exchanges). Follow-ups like "What about Bob?" or "Show them" resolve pronouns and entity substitutions using the previous exchange.

**Compound queries:** Questions containing " and " or " or " are split into sub-queries and merged (intersection/union).

**Negation:** "not connected to X" inverts the result set.

---

## MinnsQL Structured Query

Execute structured graph queries using MinnsQL, a Cypher-inspired query language with built-in temporal semantics.

### POST /api/query

**Request:**
```json
{
  "query": "MATCH (a:Person)-[r:location]->(b) RETURN a.name, b.name",
  "group_id": "tenant-1"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | yes | MinnsQL query (max 4096 bytes) |
| `group_id` | string | no | Multi-tenant partition key |

**Response (200):**
```json
{
  "columns": ["a.name", "b.name"],
  "rows": [["alice", "london"], ["bob", "berlin"]],
  "stats": {
    "nodes_scanned": 150,
    "edges_traversed": 42,
    "execution_time_ms": 3
  }
}
```

**Error responses:**
- `400` — Parse error (includes position and message)
- `504` — Query execution timed out (30s limit)
- `500` — Internal execution error

### MinnsQL Language Reference

MinnsQL is a unified query language that spans both graph and table operations. It supports graph pattern matching (MATCH), table queries (FROM), table-to-table and graph-to-table JOINs, DDL (CREATE/DROP TABLE), and DML (INSERT/UPDATE/DELETE).

#### Graph Query Syntax

```sql
MATCH (variable:Label {prop: "value"})-[edge:TYPE]->(target)
WHERE condition
RETURN expression AS alias
ORDER BY alias ASC|DESC
LIMIT n
```

#### Table Query Syntax (FROM)

```sql
FROM table_name
WHERE table_name.column = value
RETURN table_name.col1, table_name.col2
ORDER BY table_name.col1 DESC
LIMIT n
```

Column references in FROM queries must be qualified with the table name: `table.column`.

#### Table-to-Table JOIN

```sql
FROM orders
JOIN customers ON orders.customer_id = customers.id
WHERE customers.region = "EU"
RETURN customers.name, orders.amount
```

Multi-way joins:
```sql
FROM orders
JOIN customers ON orders.customer_id = customers.id
JOIN shipments ON shipments.order_id = orders.id
WHERE shipments.status = "delivered"
RETURN customers.name, orders.amount, shipments.tracking
```

All joins are inner equi-joins. Each JOIN extends the result row with columns from the joined table.

#### Graph-to-Table JOIN

```sql
MATCH (n:Person)
JOIN orders ON orders.node = n
RETURN n.name, orders.amount
```

The join condition links a table's NodeRef column to a graph variable.

#### DDL — CREATE TABLE / DROP TABLE

```sql
CREATE TABLE orders (
  id Int64 PRIMARY KEY,
  customer String NOT NULL,
  amount Float64,
  status String,
  node NodeRef REFERENCES GRAPH
)
```

Column types: `String`, `Int64`, `Float64`, `Bool`, `Timestamp`, `Json`, `NodeRef`

Constraints: `PRIMARY KEY`, `NOT NULL`, `UNIQUE`, `REFERENCES GRAPH` (NodeRef foreign key to graph)

```sql
DROP TABLE orders
```

#### DML — INSERT / UPDATE / DELETE

**Insert** (positional or with column names):
```sql
INSERT INTO orders VALUES (1, "Alice", 99.99, "pending", 42)
INSERT INTO orders (id, customer, amount) VALUES (1, "Alice", 99.99), (2, "Bob", 50.00)
```

**Update** (SET with WHERE on any column):
```sql
UPDATE orders SET status = "shipped", amount = 105.0 WHERE id = 1
UPDATE orders SET status = "cancelled" WHERE status = "pending" AND amount < 10.0
```

WHERE in UPDATE supports: `=`, `!=`, `<`, `>`, `<=`, `>=`, `CONTAINS`, `STARTS WITH`, `AND`, `OR`, `NOT`, `IS NULL`, `IS NOT NULL`. Only literal values are supported in SET assignments.

**Delete** (WHERE on any column):
```sql
DELETE FROM orders WHERE id = 1
DELETE FROM orders WHERE status = "cancelled"
```

#### Temporal Clauses (Tables and Graph)

Temporal clauses work on both graph edges and table rows. Tables use bi-temporal versioning: every UPDATE creates a new version and closes the old one.

#### Temporal Clauses

**WHEN** — Filter edges by valid-time (real-world validity period):

```sql
-- Only currently active edges (default)
MATCH (a)-[r]->(b) RETURN a

-- All edges regardless of temporal validity
MATCH (a)-[r]->(b) WHEN ALL RETURN a

-- Edges valid at a specific point
MATCH (a)-[r]->(b) WHEN "2024-06-15" RETURN a

-- Edges valid during a range
MATCH (a)-[r]->(b) WHEN "2024-01-01" TO "2024-06-01" RETURN a

-- Edges valid in the last N duration
MATCH (a)-[r]->(b) WHEN LAST "30d" RETURN a
```

**AS OF** — Filter by transaction-time (when data was recorded). Only nodes/edges with `created_at <= cutoff` are visible:

```sql
-- See the graph as it was known at a specific time
MATCH (a)-[r]->(b) WHEN ALL AS OF "2024-06-01T00:00:00Z" RETURN a
```

**Bi-temporal composition** — WHEN and AS OF can be combined. AS OF filters by when data was recorded; WHEN filters by when it was valid in the real world:

```sql
MATCH (a)-[r]->(b) WHEN "2024-01-01" TO "2024-03-01" AS OF "2024-06-01" RETURN a
```

#### Aggregation and GROUP BY

Non-aggregate expressions in RETURN automatically become group keys:

```sql
MATCH (a:Person)-[r]->(b)
RETURN a.name, count(*) AS cnt
ORDER BY cnt DESC
LIMIT 10
```

Aggregate functions: `count(*)`, `sum(expr)`, `avg(expr)`, `min(expr)`, `max(expr)`, `collect(expr)`.

#### Variable-Length Paths

```sql
-- 1 to 3 hops
MATCH (a)-[*1..3]->(b) RETURN a, b

-- Unbounded (capped at 10 hops)
MATCH (a)-[*1..]->(b) RETURN a, b
```

#### Built-in Functions

**Scalar functions:**

| Function | Args | Returns | Description |
|----------|------|---------|-------------|
| `type(x)` | node or edge | String | Node type or edge type name |
| `id(x)` | node or edge | Int | Internal ID |
| `labels(n)` | node | String | Node type name |
| `properties(x)` | node or edge | Map | All properties as a map |
| `now()` | — | Int | Current time (nanoseconds since epoch) |
| `coalesce(a, b, ...)` | any | any | First non-null argument |
| `toUpper(s)` | string | String | Uppercase |
| `toLower(s)` | string | String | Lowercase |
| `path(a, b)` | two nodes | List | Shortest path between nodes |
| `hops(p)` | path or list | Int | Number of hops in a path |

**Temporal accessor functions:**

| Function | Args | Returns | Description |
|----------|------|---------|-------------|
| `valid_from(r)` | edge | Int or Null | Start of valid-time period (nanos) |
| `valid_until(r)` | edge | Int or Null | End of valid-time period (nanos, Null = open-ended) |
| `created_at(x)` | node or edge | Int | Transaction-time creation timestamp |
| `updated_at(x)` | node or edge | Int | Transaction-time last update |
| `duration(r)` | edge | Int or Null | `valid_until - valid_from` (Null if open-ended) |
| `open_ended(r)` | edge | Bool | True if `valid_until` is Null |
| `change_type(r, t1, t2)` | edge, timestamp, timestamp | String | One of: "started", "ended", "started_and_ended", "created", "stable" |

**Time bucketing functions:**

| Function | Args | Returns | Description |
|----------|------|---------|-------------|
| `time_bucket(width, ts)` | string, Int | Int | Floor timestamp to bucket boundary. Width: `"1d"`, `"6h"`, `"30m"`, `"month"`, `"quarter"`, `"year"` |
| `date_trunc(unit, ts)` | string, Int | Int | Calendar-aware truncation. Units: `"second"`, `"minute"`, `"hour"`, `"day"`, `"week"`, `"month"`, `"quarter"`, `"year"` |
| `ago(duration)` | string | Int | `now() - duration`. Accepts: `"30s"`, `"5m"`, `"2h"`, `"7d"`, `"1w"`. Rejects calendar units (month/year) |

Example — bucketed time series:
```sql
MATCH (a:Person)-[r:location]->(b)
WHEN ALL
RETURN time_bucket("1d", valid_from(r)) AS day, count(*) AS moves
ORDER BY day
```

**Temporal join predicates (Allen's interval algebra):**

| Predicate | Condition | Description |
|-----------|-----------|-------------|
| `overlap(r1, r2)` | intervals overlap | True if the two edges' valid-time periods overlap |
| `precedes(r1, r2)` | r1.end ≤ r2.start | r1 ends before or when r2 starts |
| `meets(r1, r2)` | r1.end = r2.start | r1 ends exactly when r2 starts |
| `covers(r1, r2)` | r1 contains r2 | r1's interval fully contains r2's |
| `starts(r1, r2)` | same start, r1 ≤ r2 | r1 starts with r2 but ends sooner or same |
| `finishes(r1, r2)` | same end, r1 ≥ r2 | r1 finishes with r2 but starts later or same |
| `equals(r1, r2)` | identical intervals | Same valid_from and valid_until |

Open-ended edges (`valid_until = Null`) are treated as extending to infinity.

Example:
```sql
MATCH (a)-[r1]->(b), (a)-[r2]->(c)
WHEN ALL
WHERE meets(r1, r2)
RETURN a.name, type(r1), type(r2)
```

**TCell history functions (confidence/weight over time):**

| Function | Args | Returns | Description |
|----------|------|---------|-------------|
| `confidence_at(r, ts)` | edge, timestamp | Float or Null | Confidence value at or before the given time |
| `weight_at(r, ts)` | edge, timestamp | Float or Null | Weight value at or before the given time |
| `confidence_history(r)` | edge | List | All `[timestamp, value]` pairs, chronological |
| `weight_history(r)` | edge | List | All `[timestamp, value]` pairs, chronological |

**Boolean predicates (used in WHERE):**

| Predicate | Args | Description |
|-----------|------|-------------|
| `SUCCESSIVE(r1, r2)` | two edges, optional tolerance | True if r1.valid_until ≈ r2.valid_from (default tolerance: 1s) |
| `SUCCESSIVE(r1, r2, "100ms")` | two edges + duration | With custom tolerance |
| `CHANGED(r, t1, t2)` | edge, start, end | True if edge started, ended, or was created in [t1, t2] |

**Comparison operators:** `=`, `!=`, `<`, `>`, `<=`, `>=`, `CONTAINS`, `STARTS WITH`, `IS NULL`, `IS NOT NULL`

**Boolean logic:** `AND`, `OR`, `NOT`, parentheses for grouping

---

## Reactive Subscriptions

Register live MinnsQL queries that automatically receive incremental updates as the graph changes. The reactive engine captures deltas at mutation time (node/edge add, remove, supersede, merge) and maintains operator state per subscription for efficient incremental view maintenance.

Two delivery modes: **REST polling** for simple integrations and **WebSocket streaming** for real-time push.

### Architecture

```
Graph Mutation (add_node, add_edge, supersede, merge, ...)
        │
        ▼
  DeltaBatch (broadcast channel)
        │
        ▼
  SubscriptionManager
        │
        ├── Trigger Set fast rejection (node types, edge types)
        ├── Incremental maintenance (ScanState → ExpandState → FilterState → AggregationState)
        │   OR full rerun + structural diff (for complex cases)
        │
        ▼
  Per-subscription pending updates (inserts / deletes)
        │
        ├── REST poll endpoint (client pulls)
        └── WebSocket push (server pushes every ~100ms)
```

**Maintenance strategies:**
- **Incremental** — operator states track scan/expand/filter/aggregation. Only delta-affected rows are reprocessed. Used for simple scan + expand + filter + aggregation queries.
- **Full rerun** — re-executes the full query and diffs against cached output using structural `RowId` identity. Used when: variable-length paths, `NodeMerged` deltas, generation gaps, or complex query patterns.

**Trigger set optimization:** Each subscription compiles a trigger set from its execution plan (node type discriminants, edge type strings). Incoming delta batches are rejected in O(1) if they can't affect the subscription.

**Performance:** Background task processes deltas every 50ms (configurable via `SUBSCRIPTION_INTERVAL_MS`). Max 100 concurrent subscriptions (configurable). Zero overhead when no subscriptions are active (`delta_tx = None`).

### POST /api/subscriptions

Create a new subscription. Returns the initial result set and the maintenance strategy chosen.

**Request:**
```json
{
  "query": "MATCH (a:Agent)-[e:KNOWS]->(b:Agent) RETURN a.name, b.name, e.weight",
  "group_id": "tenant-1"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `query` | string | yes | MinnsQL query (max 4096 bytes). Same syntax as `POST /api/query` |
| `group_id` | string | no | Multi-tenant partition key |

**Response (200):**
```json
{
  "subscription_id": 7,
  "initial": {
    "columns": ["a.name", "b.name", "e.weight"],
    "rows": [["Alice", "Bob", 0.8], ["Bob", "Charlie", 0.6]]
  },
  "strategy": "incremental"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `subscription_id` | integer | Unique ID for this subscription |
| `initial.columns` | string[] | Column names from the RETURN clause |
| `initial.rows` | any[][] | Initial matching rows |
| `strategy` | string | `"incremental"` or `"full_rerun: <reason>"` |

**Error responses:**
- `400` — Parse or plan error (includes message)
- `500` — Internal error

### GET /api/subscriptions

List all active subscriptions.

**Response (200):**
```json
{
  "subscriptions": [
    {
      "subscription_id": 7,
      "query": "MATCH (a:Agent)-[e:KNOWS]->(b:Agent) RETURN a.name, b.name",
      "strategy": "incremental",
      "cached_row_count": 42
    }
  ]
}
```

### GET /api/subscriptions/:id/poll

Poll for pending updates since the last poll (or since subscription creation).

**Response (200):**
```json
{
  "updates": [
    {
      "subscription_id": 7,
      "inserts": [["Alice", "Diana", 0.9]],
      "deletes": [["Bob", "Charlie", 0.6]],
      "count": null,
      "was_full_rerun": false
    }
  ]
}
```

| Field | Type | Description |
|-------|------|-------------|
| `inserts` | any[][] | New rows matching the query |
| `deletes` | any[][] | Rows that no longer match |
| `count` | integer or null | For count-only aggregations, the new total |
| `was_full_rerun` | boolean | `true` if the update was computed via full rerun + diff rather than incremental maintenance |

If no updates are pending, `updates` is an empty array.

### DELETE /api/subscriptions/:id

Unsubscribe and release all operator state for the given subscription.

**Response (200):**
```json
{
  "unsubscribed": true
}
```

**Error responses:**
- `404` — Subscription ID not found

### GET /api/subscriptions/ws

Upgrade to a WebSocket connection for real-time streaming. All subscriptions created over this connection are automatically cleaned up on disconnect.

**Wire protocol (JSON text frames):**

#### Client → Server messages

**Subscribe:**
```json
{
  "type": "subscribe",
  "query": "MATCH (a:Agent)-[e:KNOWS]->(b:Agent) RETURN a.name, b.name",
  "request_id": "req-123"
}
```
`request_id` is optional — if provided, it is echoed back in the `subscribed` response for client-side correlation.

**Unsubscribe:**
```json
{
  "type": "unsubscribe",
  "subscription_id": 7
}
```

**Ping:**
```json
{
  "type": "ping"
}
```

#### Server → Client messages

**Subscribed (response to subscribe):**
```json
{
  "type": "subscribed",
  "subscription_id": 7,
  "request_id": "req-123",
  "initial": {
    "columns": ["a.name", "b.name"],
    "rows": [["Alice", "Bob"], ["Bob", "Charlie"]]
  }
}
```

**Update (pushed when graph changes affect a subscription):**
```json
{
  "type": "update",
  "subscription_id": 7,
  "inserts": [["Alice", "Diana"]],
  "deletes": [["Bob", "Eve"]],
  "count": null
}
```

**Unsubscribed:**
```json
{
  "type": "unsubscribed",
  "subscription_id": 7
}
```

**Error:**
```json
{
  "type": "error",
  "message": "Parse error at position 15: expected RETURN",
  "request_id": "req-123"
}
```

**Pong (response to ping):**
```json
{
  "type": "pong"
}
```

#### Connection behavior

- **Poll interval:** Updates are pushed approximately every 100ms when changes are pending.
- **Heartbeat:** Server sends a ping every 30s for keep-alive.
- **Cleanup on disconnect:** All subscriptions created over the connection are automatically unsubscribed.
- **Multiple subscriptions:** A single WebSocket connection can hold multiple concurrent subscriptions.

### Delta types

The reactive engine captures these atomic graph mutations:

| Delta | Trigger | Description |
|-------|---------|-------------|
| `NodeAdded` | `add_node()` | New node created, includes type discriminant |
| `NodeRemoved` | `remove_node()` | Node deleted, cascades to `EdgeRemoved` for incident edges |
| `EdgeAdded` | `add_edge()` | New edge created, includes edge type tag |
| `EdgeRemoved` | `remove_edge()` | Edge deleted |
| `EdgeSuperseded` | temporal supersession | Edge's `valid_until` set (no longer active) |
| `EdgeMutated` | `invalidate_edge()` | Edge properties changed |
| `NodeMerged` | `merge_nodes()` | Two nodes merged into one (forces full rerun) |

### Subscription examples

**Track agent relationships in real time:**
```
POST /api/subscriptions
{
  "query": "MATCH (a:Agent)-[e:relationship]->(b:Agent) RETURN a.name, b.name, type(e)"
}
```

**Monitor active locations:**
```
POST /api/subscriptions
{
  "query": "MATCH (a:Agent)-[e:location]->(b:Concept) RETURN a.name, b.name"
}
```

**Aggregation — count events by type:**
```
POST /api/subscriptions
{
  "query": "MATCH (n:Event) RETURN type(n), count(*) AS cnt"
}
```

**Variable-length path monitoring (uses full rerun strategy):**
```
POST /api/subscriptions
{
  "query": "MATCH (a:Agent)-[*1..3]->(b:Agent) RETURN a.name, b.name"
}
```

### MinnsQL SUBSCRIBE syntax

The query language also supports inline subscription statements:

```sql
-- Create a subscription
SUBSCRIBE MATCH (a:Agent)-[e:KNOWS]->(b:Agent) RETURN a.name, b.name

-- Cancel a subscription by ID
UNSUBSCRIBE 7
```

These are parsed as `Statement::Subscribe(Query)` and `Statement::Unsubscribe(id)` respectively, and can be used via `POST /api/query` when subscription support is wired through the query endpoint.

---

## Temporal Tables

MinnsDB includes a full temporal relational table engine alongside its graph database. Tables support bi-temporal versioning, bulk import, graph linking via NodeRef columns, and full CRUD operations through both the REST API and MinnsQL.

### Architecture

Tables use a page-based row store with 8KB slotted pages, blake3 page checksums, custom binary row encoding with O(1) column access, and in-memory indexes rebuilt from page data on startup. Every row is bi-temporal: `valid_from`/`valid_until` (real-world validity) and `created_at` (transaction time). Updates create new versions and close old ones — no data is overwritten.

### REST API Endpoints

#### POST /api/tables — Create table

**Request:**
```json
{
  "name": "orders",
  "columns": [
    {"name": "id", "col_type": "Int64", "nullable": false},
    {"name": "customer", "col_type": "String", "nullable": false},
    {"name": "amount", "col_type": "Float64", "nullable": true},
    {"name": "node", "col_type": "NodeRef", "nullable": true}
  ],
  "constraints": [
    {"PrimaryKey": ["id"]}
  ]
}
```

**Response (201):**
```json
{"table_id": 1, "name": "orders"}
```

#### GET /api/tables — List tables

**Response (200):**
```json
[
  {"table_id": 1, "name": "orders", "columns": [...], "constraints": [...]},
  {"table_id": 2, "name": "customers", "columns": [...], "constraints": [...]}
]
```

#### GET /api/tables/:name/schema — Get schema

**Response (200):**
```json
{"table_id": 1, "name": "orders", "columns": [...], "constraints": [...]}
```

#### DELETE /api/tables/:name — Drop table

**Response (200):**
```json
{"table_id": 1, "dropped": true}
```

#### POST /api/tables/:name/rows — Insert row(s)

**Single row:**
```json
{
  "group_id": 0,
  "values": [1, "Alice", 99.99, null]
}
```

**Response (201):**
```json
{"row_id": 1, "version_id": 1}
```

**Batch (array):**
```json
[
  {"group_id": 0, "values": [1, "Alice", 99.99, null]},
  {"group_id": 0, "values": [2, "Bob", 50.00, null]}
]
```

#### PUT /api/tables/:name/rows/:id — Update row

```json
{
  "group_id": 0,
  "values": [1, "Alice Updated", 105.0, null]
}
```

**Response (200):**
```json
{"old_version_id": 1, "new_version_id": 2}
```

Updates create a new version. The old version's `valid_until` is set to the current timestamp. Both versions are retained for temporal queries.

#### DELETE /api/tables/:name/rows/:id — Soft-delete row

```json
{"group_id": 0}
```

**Response (200):**
```json
{"version_id": 1}
```

Soft-deletes close the row's `valid_until`. The row remains queryable through `WHEN ALL` and `AS OF` temporal queries until removed by compaction.

#### GET /api/tables/:name/rows — Scan rows

**Query parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `when` | string | `"active"` | `"active"` (current rows) or `"all"` (all versions) |
| `as_of` | u64 | — | Point-in-time timestamp (nanoseconds) |
| `group_id` | u64 | `0` | Multi-tenant group |
| `limit` | usize | all | Max rows to return |
| `offset` | usize | `0` | Skip first N rows |

**Response (200):**
```json
{
  "count": 5,
  "rows": [
    {
      "row_id": 1, "version_id": 2, "group_id": 0,
      "valid_from": 1710700000000000000,
      "valid_until": null,
      "values": [1, "Alice", 105.0, null]
    }
  ]
}
```

#### GET /api/tables/:name/by-node/:node_id — Rows referencing a graph node

Returns all active rows in the table where a NodeRef column points to the given graph node ID.

**Query parameters:** `group_id` (optional)

#### POST /api/tables/:name/compact — Trigger compaction

Reclaims space from old closed versions past the retention window (default: 7 days). At least one version per row is always retained.

**Response (200):**
```json
{"versions_removed": 42, "pages_compacted": 3}
```

#### GET /api/tables/:name/stats — Table statistics

**Response (200):**
```json
{
  "name": "orders",
  "active_rows": 150,
  "total_versions": 200,
  "pages": 5,
  "generation": 200
}
```

### MinnsQL for Tables

All table operations are also available through the unified `POST /api/query` endpoint using MinnsQL syntax. See the [MinnsQL Language Reference](#minnsql-language-reference) section for the full grammar including CREATE TABLE, INSERT INTO, UPDATE, DELETE FROM, FROM queries, and JOINs.

### Column Types

| Type | Size | Description |
|------|------|-------------|
| `String` | variable | UTF-8 text |
| `Int64` | 8 bytes | 64-bit signed integer |
| `Float64` | 8 bytes | 64-bit IEEE 754 float |
| `Bool` | 1 byte | true/false |
| `Timestamp` | 8 bytes | Nanoseconds since Unix epoch |
| `Json` | variable | Arbitrary nested data (stored as MessagePack internally) |
| `NodeRef` | 8 bytes | Reference to a graph node ID |

### Multi-Tenancy

Every table row has an implicit `group_id` (u64). All queries, inserts, updates, and deletes are scoped by group_id. Unique constraints are enforced within a group, not globally. Different groups can have the same primary key values.

### Temporal Model

| Dimension | Fields | Meaning |
|-----------|--------|---------|
| Transaction time | `created_at` | When the row version was written to the database |
| Valid time | `valid_from`, `valid_until` | When the fact was true in the real world |

- `valid_until = null` means the row is currently active
- UPDATE: closes old version (`valid_until = now`), inserts new version
- DELETE: closes the version (`valid_until = now`), no new version
- `WHEN ALL` queries return all versions including closed ones
- `AS OF <timestamp>` queries return the state at a specific point in time

### Storage Details

- **8KB slotted pages** with blake3 checksums (validated on load)
- **Custom binary row format** with null bitmap, fixed-width inline values, variable-length offset table
- **O(1) column access** — read any column without parsing the entire row
- **Dead slot reuse** — deleted slots are reused for new inserts without compaction
- **Predicate pushdown** — `WHERE id = N` on primary key uses O(1) index lookup
- **Row size limit** — a single row must fit within one 8KB page (~8100 bytes max)
- **Persistence** — pages stored as raw 8KB blobs in ReDB, indexes rebuilt on startup

---

## Claims

Claims are semantic assertions extracted by the NER + LLM pipeline from events.

### GET /api/claims?limit=10&event_id=123

List active claims, optionally filtered by source event.

**Response (200):** Array of claim objects.
```json
[
  {
    "claim_id": 1,
    "claim_text": "The user prefers null checks over optional chaining",
    "confidence": 0.92,
    "source_event_id": 123456,
    "similarity": null,
    "evidence_spans": [
      {"start_offset": 10, "end_offset": 45, "text_snippet": "prefers null checks"}
    ],
    "support_count": 3,
    "status": "Active",
    "created_at": 1735603200000000000,
    "last_accessed": 1735689600000000000,
    "claim_type": "Preference",
    "subject_entity": "user",
    "expires_at": null,
    "temporal_weight": 1.0,
    "superseded_by": null,
    "entities": [
      {"text": "null checks", "label": "CONCEPT"}
    ]
  }
]
```

**Claim types:** `"Preference"`, `"Fact"`, `"Belief"`, `"Intention"`, `"Capability"`, `"Avoidance"`, `"CodePattern"`, `"ApiContract"`, `"BugFix"`

Note: `GET /api/claims` returns a flat array. `POST /api/claims/search` returns a grouped response — see below.

### GET /api/claims/:id

Get a single claim by ID. Same response shape as list item.

### POST /api/claims/search

Semantic search for claims using embeddings. Results are grouped by subject entity.

**Request:**
```json
{
  "query_text": "error handling preferences",
  "top_k": 10,
  "min_similarity": 0.7
}
```

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query_text` | string | required | Natural language search query |
| `top_k` | integer | 10 | Max results to return |
| `min_similarity` | float | 0.7 | Minimum cosine similarity threshold |

**Response (200):**
```json
{
  "groups": [
    {
      "subject": "authentication",
      "claims": [
        {
          "claim_id": 42,
          "claim_text": "The auth module uses JWT tokens with 24h expiry",
          "claim_type": "ApiContract",
          "confidence": 0.92,
          "similarity": 0.85,
          "source_event_id": 1234,
          "support_count": 3,
          "status": "Active",
          "temporal_weight": 0.97,
          "subject_entity": "authentication",
          "entities": [{"text": "JWT", "label": "TECHNOLOGY"}],
          "evidence_spans": [],
          "created_at": 1709312400000000000,
          "last_accessed": 1709398800000000000,
          "expires_at": null,
          "superseded_by": null
        }
      ]
    }
  ],
  "ungrouped": [
    {
      "claim_id": 55,
      "claim_text": "We prefer returning Result over panicking",
      "claim_type": "CodePattern",
      "confidence": 0.88,
      "similarity": 0.79,
      "source_event_id": 1240,
      "support_count": 1,
      "status": "Active",
      "temporal_weight": 0.99,
      "subject_entity": null,
      "entities": [],
      "evidence_spans": [],
      "created_at": 1709312400000000000,
      "last_accessed": 1709312400000000000,
      "expires_at": null,
      "superseded_by": null
    }
  ],
  "total_results": 2
}
```

- `groups` — claims grouped by `subject_entity` (e.g. "authentication", "database", "error_handling")
- `ungrouped` — claims with no `subject_entity`
- `total_results` — total count across both groups and ungrouped

Claims are ranked by `similarity` (cosine similarity to query embedding), weighted by `confidence` and `temporal_weight` (half-life decay).

### POST /api/embeddings/process?limit=10

Process pending claims to generate embeddings (batch processing).

**Response (200):**
```json
{
  "claims_processed": 5,
  "success": true
}
```

---

## Structured Memory

Templated data structures (ledgers, trees, state machines, preference lists) for domain-specific memory patterns that don't fit the general graph model.

### POST /api/structured-memory

Upsert a structured memory template.

**Request:**
```json
{
  "key": "ledger:1:2",
  "template": {
    "Ledger": {
      "entries": [],
      "balance": 0.0,
      "provenance": "Manual"
    }
  }
}
```

**Template types:**

```json
// Ledger — append-only double-entry accounting
{ "Ledger": { "entries": [], "balance": 0.0, "provenance": "Manual" } }

// StateMachine — entity state tracking
{ "StateMachine": { "current_state": "idle", "history": [], "provenance": "Manual" } }

// PreferenceList — ranked items
{ "PreferenceList": { "ranked_items": [], "provenance": "Manual" } }

// Tree — hierarchical data
{ "Tree": { "nodes": {}, "provenance": "Manual" } }
```

**Provenance:** `"Manual"`, `"EpisodePipeline"`, `"NlqUpsert"`

**Response (200):**
```json
{ "success": true, "key": "ledger:1:2" }
```

### GET /api/structured-memory?prefix=ledger

List all keys, optionally filtered by prefix.

**Response (200):**
```json
{ "keys": ["ledger:1:2", "ledger:3:4"], "count": 2 }
```

### GET /api/structured-memory/:key

Get a structured memory by key.

**Response (200):**
```json
{
  "key": "ledger:1:2",
  "template": { "Ledger": { "entries": [...], "balance": 50.0, "provenance": "Manual" } }
}
```

### DELETE /api/structured-memory/:key

Remove a structured memory.

**Response (200):**
```json
{ "success": true, "key": "ledger:1:2" }
```

### POST /api/structured-memory/ledger/:key/append

Append a ledger entry. The balance is recomputed on every append.

**Request:**
```json
{
  "amount": 25.0,
  "description": "Payment for services",
  "direction": "Credit"
}
```

**`direction`:** `"Credit"` (adds) or `"Debit"` (subtracts)

**Response (200):**
```json
{ "success": true, "balance": 75.0 }
```

### GET /api/structured-memory/ledger/:key/balance

**Response (200):**
```json
{ "key": "ledger:1:2", "balance": 75.0 }
```

### POST /api/structured-memory/state/:key/transition

Transition a state machine to a new state.

**Request:**
```json
{
  "new_state": "active",
  "trigger": "user_login"
}
```

**Response (200):**
```json
{ "success": true, "new_state": "active" }
```

### GET /api/structured-memory/state/:key/current

**Response (200):**
```json
{ "key": "state:42", "current_state": "active" }
```

### POST /api/structured-memory/preference/:key/update

Update or insert a preference ranking.

**Request:**
```json
{
  "item": "pizza",
  "rank": 1,
  "score": 9.5
}
```

**Response (200):**
```json
{ "success": true }
```

### POST /api/structured-memory/tree/:key/add-child

Add a child node to a tree.

**Request:**
```json
{
  "parent": "root",
  "child": "category-a"
}
```

**Response (200):**
```json
{ "success": true }
```

---

## Conversation Ingestion

Ingest multi-session conversations into the graph. Each message is converted to a `Conversation` event and processed through the full event pipeline. An inline LLM compaction step then extracts structured facts (locations, relationships, preferences, financial transactions) and writes them as graph edges.

A configured LLM client is required (`LLM_API_KEY` or `OPENAI_API_KEY`). When NER is enabled (`enable_semantic_memory`), a specialized accounting NER model runs in parallel with the LLM cascade to extract PAYER/AMOUNT/PAYEE triplets for financial messages.

### Incremental Ingestion

The server maintains a persistent `ConversationState` per `case_id` across API calls. This ensures:

- **Stable entity IDs:** The `ConversationState` (participant tracking) is preserved across calls. Entity resolution uses the graph `concept_index` — Alice always maps to the same Concept node regardless of which batch she first appeared in.
- **Idempotency:** Every processed message is tracked by `(case_id, session_id, message_index)`. Re-ingesting the same message in a later call is a no-op (0 messages processed).
- **Participant accumulation:** Known participants grow across calls, so "split among all" always resolves to the full set.

States are capped at 1,000 per server instance (LRU eviction by fewest processed messages).

**SDK pattern for incremental ingestion:**
```
// Call 1: First batch of messages
POST /api/conversations/ingest { "case_id": "trip_2024", "sessions": [...batch1...] }
// → Alice=1, Bob=2

// Call 2: More messages arrive later
POST /api/conversations/ingest { "case_id": "trip_2024", "sessions": [...batch2...] }
// → Alice still=1, Bob still=2, Charlie=3 (new)
// → Duplicate messages from batch1 are skipped automatically
```

### POST /api/conversations/ingest

Ingest one or more conversation sessions. Each message is converted to a Conversation event and processed through the event pipeline, then LLM compaction extracts facts and writes them as graph edges. Idempotent per `case_id` — re-ingesting the same `(case_id, session_id, message_index)` tuple is a no-op.

**Request:**
```json
{
  "case_id": "trip_expenses_2024",
  "sessions": [
    {
      "session_id": "session_01",
      "topic": "Dinner expenses",
      "messages": [
        { "role": "user", "content": "Alice: Paid €179 for museum - split with Bob" },
        { "role": "user", "content": "Bob: Paid €107 for dinner - split among all" },
        { "role": "user", "content": "The weather was lovely today!" },
        { "role": "assistant", "content": "Sounds like a great trip!" }
      ],
      "contains_fact": false,
      "fact_id": null,
      "fact_quote": null
    },
    {
      "session_id": "session_02",
      "topic": "Moving to NYC",
      "messages": [
        { "role": "user", "content": "I live in Alfama, Lisbon." },
        { "role": "user", "content": "I'm moving to Lower Manhattan, NYC." },
        { "role": "user", "content": "Johnny Fisher works with Christopher Peterson." }
      ]
    }
  ],
  "include_assistant_facts": false
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `case_id` | string | no | Case identifier; auto-generated UUID if omitted |
| `sessions` | array | yes | One or more conversation sessions |
| `sessions[].session_id` | string | yes | Unique session identifier |
| `sessions[].topic` | string | no | Topic label for context |
| `sessions[].messages` | array | yes | Ordered messages (`role` + `content`) |
| `sessions[].messages[].role` | string | yes | `"user"` or `"assistant"` |
| `sessions[].messages[].content` | string | yes | Message text |
| `sessions[].contains_fact` | bool | no | Benchmark metadata (ignored during ingestion) |
| `sessions[].fact_id` | string | no | Benchmark metadata (ignored during ingestion) |
| `sessions[].fact_quote` | string | no | Benchmark metadata (ignored during ingestion) |
| `include_assistant_facts` | bool | no | If `true`, extract facts from assistant messages too (default `false`) |

**Response (200):**
```json
{
  "case_id": "trip_expenses_2024",
  "messages_processed": 7,
  "events_submitted": 8,
  "compaction": {
    "facts_extracted": 5,
    "goals_extracted": 0,
    "goals_deduplicated": 0,
    "procedural_steps": 0,
    "memories_created": false,
    "memories_updated": 0,
    "memories_deleted": 0,
    "playbooks_extracted": 0,
    "llm_success": true
  },
  "rolling_summary_started": true
}
```

**Compaction extracts facts with these categories (written as graph edges):**

| Category | Edge Type | Example Message |
|----------|-----------|-----------------|
| `location` | `state:location` | `"I'm moving to NYC"`, `"I live in Lisbon"` |
| `work` | `state:work` | `"I started a new job at Google"` |
| `financial` | `financial:payment` | `"Alice: Paid €50 for lunch - split with Bob"` |
| `relationship` | `relationship:*` | `"Johnny Fisher works with Christopher Peterson"` |
| `preference` | `preference:*` | `"I love fantasy novels"`, `"My favorite is pasta"` |
| `routine` | `state:routine` | `"I take morning walks in Battery Park"` |

**NER financial extraction** (parallel with LLM cascade): When NER is enabled, the specialized accounting model extracts PAYER/AMOUNT/PAYEE triplets from financial messages and creates `financial:payment` edges. These are deduped against LLM-extracted financial facts.

---

## Single Message Ingestion

Ingest individual messages (user or assistant turns) with automatic buffering and LLM compaction. Messages are buffered per-session until enough context accumulates, then compacted into facts and graph edges.

### POST /api/messages

**Request:**
```json
{
  "role": "user",
  "content": "I just moved from London to Berlin last week",
  "session_id": "session-abc",
  "case_id": "case-123",
  "include_assistant_facts": false
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `role` | string | yes | `"user"` or `"assistant"` |
| `content` | string | yes | Message text |
| `session_id` | string | no | Session ID (auto-generated if omitted) |
| `case_id` | string | no | Case ID for cross-session grouping (auto-generated if omitted) |
| `include_assistant_facts` | bool | no | Extract facts from assistant messages too (default: false) |

**Response (200):**
```json
{
  "case_id": "case-123",
  "session_id": "session-abc",
  "messages_processed": 1,
  "events_submitted": 1,
  "buffered": true,
  "buffer_size": 3,
  "compaction": null
}
```

When the buffer reaches the compaction threshold, `compaction` is populated:
```json
{
  "compaction": {
    "facts_extracted": 2,
    "goals_extracted": 0,
    "goals_deduplicated": 0,
    "procedural_steps": 0,
    "memories_created": 1,
    "memories_updated": 0,
    "memories_deleted": 0,
    "playbooks_extracted": 0,
    "llm_success": true
  }
}
```

---

## Workflows

Multi-step workflows represented as graph structures. Each workflow is a set of steps with dependencies, roles, and state tracking.

### POST /api/workflows

Create a new workflow.

**Request:**
```json
{
  "name": "Deploy Pipeline",
  "intent": "deploy",
  "description": "Standard deployment workflow",
  "steps": [
    {
      "id": "build",
      "role": "ci",
      "task": "Build and test",
      "depends_on": [],
      "inputs": ["source_branch"],
      "outputs": ["build_artifact"]
    },
    {
      "id": "deploy",
      "role": "cd",
      "task": "Deploy to staging",
      "depends_on": ["build"],
      "inputs": ["build_artifact"],
      "outputs": ["deploy_url"]
    }
  ],
  "group_id": "team-1",
  "metadata": {"priority": "high"}
}
```

**Response (200):**
```json
{
  "success": true,
  "workflow_id": 42,
  "workflow_name": "Deploy Pipeline",
  "nodes_created": 3,
  "edges_created": 2,
  "step_node_ids": {"build": 43, "deploy": 44}
}
```

### GET /api/workflows

List workflows.

| Query Parameter | Type | Default | Description |
|-----------------|------|---------|-------------|
| `group_id` | string | — | Filter by group |
| `limit` | integer | 50 | Max workflows to return |

**Response (200):**
```json
{
  "workflows": [
    {
      "workflow_id": 42,
      "name": "Deploy Pipeline",
      "intent": "deploy",
      "step_count": 2,
      "group_id": "team-1",
      "created_at": "2026-03-15T10:00:00Z",
      "active": true
    }
  ],
  "count": 1
}
```

### GET /api/workflows/:id

Get full workflow details including steps.

**Response (200):**
```json
{
  "workflow_id": 42,
  "name": "Deploy Pipeline",
  "intent": "deploy",
  "description": "Standard deployment workflow",
  "group_id": "team-1",
  "created_at": "2026-03-15T10:00:00Z",
  "steps": [
    {
      "node_id": 43,
      "id": "build",
      "role": "ci",
      "task": "Build and test",
      "depends_on": [],
      "inputs": ["source_branch"],
      "outputs": ["build_artifact"],
      "state": null,
      "metadata": {}
    }
  ],
  "metadata": {"priority": "high"}
}
```

### PUT /api/workflows/:id

Update a workflow. Supersedes old step nodes/edges and creates new ones.

**Request:** Same shape as POST, all fields optional.

**Response (200):**
```json
{
  "success": true,
  "workflow_id": 42,
  "nodes_created": 2,
  "nodes_superseded": 2,
  "edges_created": 1,
  "edges_superseded": 1,
  "step_node_ids": {"build": 50, "deploy": 51}
}
```

### DELETE /api/workflows/:id

Soft-delete a workflow (supersedes edges).

**Response (200):**
```json
{
  "success": true,
  "workflow_id": 42,
  "edges_superseded": 3
}
```

### POST /api/workflows/:id/steps/:step_id/transition

Transition a workflow step to a new state.

**Request:**
```json
{
  "state": "completed",
  "result": "Build succeeded, artifact at s3://..."
}
```

**Response (200):**
```json
{
  "success": true,
  "workflow_id": 42,
  "step_id": "build",
  "new_state": "completed"
}
```

### POST /api/workflows/:id/feedback

Attach outcome feedback to a workflow.

**Request:**
```json
{
  "feedback": "Deployment completed with zero downtime",
  "outcome": "success"
}
```

| `outcome` values | Description |
|-------------------|-------------|
| `"success"` | Workflow succeeded |
| `"partial"` | Partial completion |
| `"failure"` | Workflow failed |

**Response (200):**
```json
{
  "success": true,
  "workflow_id": 42,
  "feedback_node_id": 55
}
```

---

## Agent Registry

Register agents and their capabilities for multi-agent coordination.

### POST /api/agents/register

**Request:**
```json
{
  "agent_id": "coder-agent-1",
  "group_id": "team-1",
  "repository": "backend",
  "capabilities": ["code", "test", "review"]
}
```

**Response (200):**
```json
{
  "agent_node_id": 60,
  "repo_node_id": 61,
  "status": "registered"
}
```

### GET /api/agents

| Query Parameter | Type | Required | Description |
|-----------------|------|----------|-------------|
| `group_id` | string | yes | Partition key for agent filtering |

**Response (200):**
```json
{
  "agents": [
    {
      "node_id": 60,
      "agent_id": "coder-agent-1",
      "group_id": "team-1",
      "repositories": ["backend"],
      "capabilities": ["code", "test", "review"],
      "last_seen": 1742025600000000000
    }
  ]
}
```

---

## Ontology Evolution

Manage the OWL/RDFS ontology that drives edge behaviors (symmetry, functional constraints, cascade dependencies). Properties are defined in Turtle (TTL) format and can be discovered automatically from graph patterns.

### GET /api/ontology/properties

List all registered ontology properties.

**Response (200):**
```json
{
  "properties": [
    {
      "property_name": "location:lives_in",
      "domain": "Person",
      "range": "Location",
      "is_symmetric": false,
      "is_functional": true,
      "is_append_only": false,
      "cascade_dependents": ["location:commutes_to"]
    }
  ],
  "count": 12
}
```

### POST /api/ontology/upload

Upload a Turtle ontology file. Properties are registered and cascade dependencies are inferred.

**Request:**
```json
{
  "ttl": "@prefix : <http://minnsdb.dev/ontology/> .\n:lives_in a owl:FunctionalProperty ."
}
```

**Response (200):**
```json
{
  "status": "ok",
  "properties_registered": 3,
  "cascade_properties_updated": 1
}
```

### POST /api/ontology/discover

Run three-phase automatic ontology discovery: behavior inference from graph patterns, hierarchy discovery, and cascade dependency inference.

**Request:** Empty POST body.

**Response (200):**
```json
{
  "proposals_created": 5,
  "proposal_ids": [1, 2, 3, 4, 5],
  "cascade_properties_updated": 2
}
```

### POST /api/ontology/cascade-inference

Run cascade dependency inference only (without full discovery).

**Request:** Empty POST body.

**Response (200):**
```json
{
  "status": "ok",
  "properties_updated": 3
}
```

### GET /api/ontology/observations

List observed predicates and usage statistics from the graph.

**Response (200):**
```json
{
  "observations": [
    {
      "predicate": "location:lives_in",
      "domain": "Person",
      "range": "Location",
      "count": 42,
      "last_seen": 1742025600000000000
    }
  ],
  "stats": {
    "total_predicates": 15,
    "total_observations": 230,
    "timestamp": 1742025600000000000
  }
}
```

### GET /api/ontology/proposals

List ontology evolution proposals (pending review).

**Response (200):**
```json
{
  "proposals": [
    {
      "id": 1,
      "property_name": "works_with",
      "domain": "Person",
      "range": "Person",
      "is_symmetric": true,
      "is_functional": false,
      "status": "pending",
      "confidence": 0.87
    }
  ],
  "count": 1
}
```

### GET /api/ontology/proposals/:id

Get a specific proposal.

### POST /api/ontology/proposals/:id/approve

Approve a proposal and apply it to the ontology registry.

**Response (200):**
```json
{
  "status": "applied",
  "properties_registered": 1
}
```

### POST /api/ontology/proposals/:id/reject

Reject a proposal.

**Response (200):**
```json
{
  "status": "rejected"
}
```

### GET /api/ontology/stats

Ontology evolution statistics.

**Response (200):**
```json
{
  "total_proposals": 12,
  "pending_proposals": 3,
  "approved_proposals": 8,
  "total_properties": 25,
  "evolution_passes": 4
}
```

---

## WASM Agent Modules

MinnsDB includes a sandboxed WASM agent runtime. Developers write agent modules in Rust (or any language that compiles to WASM), upload them to the server, and the modules execute within a strict sandbox with explicit permissions, instruction budgets, memory limits, and usage metering.

### Architecture

- **Runtime**: Wasmtime with Cranelift JIT compilation
- **Sandboxing**: Per-call instruction budget ("life"), epoch-based wall-time limits, 64MB memory cap
- **Data exchange**: MessagePack for all host-to-module communication
- **Permissions**: Modules declare required permissions; admin approves at upload; enforced at runtime
- **Triggers**: Modules can be invoked by HTTP, table mutations, graph deltas, or cron schedules
- **Usage metering**: Every call records life consumed, rows read/written, HTTP requests, etc. — resettable monthly for billing

### REST API Endpoints

#### POST /api/modules — Upload WASM module

**Request:**
```json
{
  "name": "order-processor",
  "wasm_base64": "<base64 encoded WASM bytes>",
  "permissions": [
    "table:orders:read",
    "table:orders:write",
    "table:customers:read",
    "graph:query",
    "http:fetch:api.stripe.com"
  ],
  "group_id": 0
}
```

**Limits:** Max 5MB WASM binary, max 100 modules.

**Response (201):**
```json
{
  "name": "order-processor",
  "module_id": 1,
  "enabled": true,
  "permissions": ["table:orders:read", "table:orders:write", ...],
  "functions": ["process_order", "reconcile"],
  "triggers": 2
}
```

The module is compiled, `__minns_describe__` is called to extract the descriptor (functions, triggers, permissions), and `__minns_init__` is called if exported.

#### GET /api/modules — List modules

**Response (200):**
```json
[
  {"name": "order-processor", "module_id": 1, "enabled": true, "group_id": 0, "uploaded_at": 1710700000000000000}
]
```

#### GET /api/modules/:name — Module info

**Response (200):**
```json
{
  "name": "order-processor",
  "module_id": 1,
  "enabled": true,
  "permissions": ["table:orders:read", ...],
  "functions": ["process_order", "reconcile"],
  "triggers": 2
}
```

#### DELETE /api/modules/:name — Unload module

Removes the module instance, its blob (if not shared), and all its schedules. Usage data is retained until the next persist.

**Response (200):**
```json
{"deleted": true}
```

#### POST /api/modules/:name/call/:function — Call a function

**Request:**
```json
{
  "args_base64": "<base64 encoded MessagePack args>"
}
```

If `args_base64` is omitted, empty args are passed. The function receives the MessagePack bytes and returns MessagePack result bytes.

**Response (200):**
```json
{
  "result_base64": "<base64 encoded MessagePack result>"
}
```

**Error responses:**
- `404` — Module or function not found
- `403` — Permission denied (module tried to access a resource it's not permitted)
- `408` — Life exceeded (instruction budget exhausted) or timeout (wall-time limit)
- `400` — Execution error

#### PUT /api/modules/:name/enable — Enable module

Enables a disabled module. Triggers and schedules resume.

#### PUT /api/modules/:name/disable — Disable module

Stops triggers and schedules. Direct function calls also fail.

#### GET /api/modules/:name/usage — Usage statistics

**Response (200):**
```json
{
  "module_name": "order-processor",
  "total_life_consumed_lo": 450000000000,
  "total_life_consumed_hi": 0,
  "total_calls": 150,
  "total_rows_read": 3000,
  "total_rows_written": 500,
  "total_graph_queries": 50,
  "total_http_requests": 25,
  "total_http_bytes": 102400,
  "total_subscription_events": 0,
  "period_start": 1710700000000000000,
  "last_updated": 1710786400000000000
}
```

`total_life_consumed` is a u128 split across `_lo` (low 64 bits) and `_hi` (high 64 bits). Default life budget is 120 trillion per call (~1 minute of WASM runtime).

#### POST /api/modules/:name/usage/reset — Reset usage (billing period)

Resets all counters to zero and starts a new billing period. Returns the previous period's totals before zeroing.

**Response (200):**
```json
{
  "previous_period": {
    "total_calls": 150,
    "total_rows_read": 3000,
    ...
  },
  "reset": true
}
```

#### GET /api/modules/:name/schedules — List schedules

**Response (200):**
```json
[
  {
    "schedule_id": 1,
    "cron": "0 * * * * *",
    "function": "reconcile",
    "enabled": true,
    "next_run": 1710786400000000000,
    "last_run": 1710786340000000000
  }
]
```

#### POST /api/modules/:name/schedules — Create schedule

```json
{
  "cron": "0 */5 * * * *",
  "function": "reconcile"
}
```

Cron expressions use 6-field format (seconds, minutes, hours, day-of-month, month, day-of-week).

**Response (201):**
```json
{"schedule_id": 1}
```

#### DELETE /api/modules/:name/schedules/:id — Delete schedule

**Response (200):**
```json
{"deleted": true}
```

### Permission System

Modules declare required permissions in their descriptor. The admin reviews and approves at upload time. Every host function checks permissions at runtime.

| Permission | Meaning |
|-----------|---------|
| `table:<name>:read` | Read rows from a specific table |
| `table:<name>:write` | Insert/update/delete rows in a specific table |
| `table:*:read` | Read any table (wildcard) |
| `table:*:write` | Write any table (wildcard) |
| `graph:query` | Execute graph queries (MinnsQL MATCH) |
| `graph:subscribe` | Create graph subscriptions |
| `http:fetch:<domain>` | Make HTTP requests to a specific domain |
| `http:fetch:*` | Make HTTP requests to any external domain |
| `schedule` | Create/modify scheduled tasks |

Wildcard matching: each colon-separated segment is matched independently. `table:*:read` matches `table:orders:read`.

### Sandboxing

| Resource | Limit | Mechanism |
|----------|-------|-----------|
| Instructions | 120T per call (configurable) | Wasmtime instruction metering ("life budget"). Resets per call. |
| Wall time | 30s per call (configurable) | Epoch interruption (10ms ticks) |
| Memory | 64MB linear memory | Wasmtime `StoreLimits` |
| Result size | 1MB per result buffer | Host-side enforcement |
| HTTP responses | 1MB, 5s timeout | Host-side enforcement |
| HTTP domains | Explicit allowlist | Permission check + internal IP blocking |
| Table query rows | 100K max | Host-side enforcement |

**SSRF protection**: Only `http://` and `https://` URLs are allowed. Redirects are disabled. Internal network addresses (`localhost`, `127.0.0.1`, `169.254.169.254`, `10.*`, `172.16.*`, `192.168.*`) are blocked.

### Module ABI

Modules must export:
- `memory` — WASM linear memory
- `__minns_describe__() -> i32` — Returns module descriptor (MessagePack)
- `__minns_init__() -> i32` (optional) — Called once on load
- `__minns_call__(func_id: i32, args_ptr: i32, args_len: i32) -> i32` — Main call entry
- `__minns_on_trigger__(trigger_id: i32, event_ptr: i32, event_len: i32) -> i32` (optional) — Trigger handler
- `__minns_on_schedule__(schedule_id: i32) -> i32` (optional) — Schedule handler
- `alloc(size: i32) -> i32` — Memory allocator for host→module data transfer

### Host Functions

Host functions available to WASM modules (imported from `minns` namespace):

| Function | Signature | Description |
|----------|-----------|-------------|
| `table_get` | `(table_ptr, table_len, row_id, result_ptr) -> i32` | Get a row by ID |
| `table_insert` | `(table_ptr, table_len, row_ptr, row_len) -> i32` | Insert a row |
| `table_delete` | `(table_ptr, table_len, row_id) -> i32` | Delete a row |
| `table_query` | `(query_ptr, query_len, result_ptr) -> i32` | Execute a MinnsQL FROM query |
| `graph_query` | `(query_ptr, query_len, result_ptr) -> i32` | Execute a MinnsQL MATCH query |
| `http_fetch` | `(req_ptr, req_len, result_ptr) -> i32` | Sandboxed HTTP request |
| `log` | `(level, msg_ptr, msg_len)` | Write to server log |
| `result_len` | `() -> i32` | Length of last result buffer |
| `module_id` | `() -> i64` | Current module's ID |
| `group_id` | `() -> i64` | Current module's group_id |

All data is serialised as MessagePack. Return value `0` = success, `-1` = error (error details in `last_result`).

### Trigger Types

Modules register triggers in their descriptor:

| Trigger Type | Fires When | Event Payload |
|-------------|------------|---------------|
| `table_insert` | A row is inserted into a specified table | `{type, table, row_id, version_id}` |
| `table_update` | A row is updated in a specified table | `{type, table, row_id, old_version_id, new_version_id}` |
| `table_delete` | A row is deleted from a specified table | `{type, table, row_id, version_id}` |
| `graph_edge_added` | An edge of a specified type is added to the graph | `{type, source, target, edge_type}` |
| `graph_edge_removed` | An edge of a specified type is removed | `{type, source, target, edge_type}` |
| `graph_node_added` | A node of a specified type is added | `{type, node_id, node_type}` |
| `schedule` | A cron expression fires | Called via `__minns_on_schedule__` |

Triggers fire asynchronously after the mutation commits. All triggers for a module are serialised (one at a time).

### Persistence

Module records, WASM blobs (content-addressed by blake3 hash), usage counters, and schedules are persisted to ReDB. On startup, all modules are recompiled from stored blobs and usage counters are restored. Deleted modules are cleaned up from ReDB on the next persist cycle.

---

## Planning & World Model

These endpoints require `ENABLE_WORLD_MODEL=true` and/or `ENABLE_STRATEGY_GENERATION=true` in the environment.

### POST /api/planning/strategies

Generate strategy candidates for a goal using LLM + world model scoring.

**Request:**
```json
{
  "goal_description": "Fix the authentication timeout bug",
  "goal_bucket_id": 1,
  "context_fingerprint": 12345
}
```

**Response (200):**
```json
{
  "ok": true,
  "candidates": [
    {
      "goal_description": "Fix the authentication timeout bug",
      "steps": 4,
      "confidence": 0.85,
      "total_energy": 0.23,
      "decision": "accepted"
    }
  ]
}
```

### POST /api/planning/actions

Generate action candidates for a specific strategy step.

**Request:**
```json
{
  "goal_description": "Fix the authentication timeout bug",
  "goal_bucket_id": 1,
  "step_index": 0,
  "context_fingerprint": 12345
}
```

**Response (200):**
```json
{
  "ok": true,
  "actions": [
    {
      "action_type": "increase_timeout",
      "confidence": 0.9,
      "energy": 0.15,
      "feasibility": 0.95
    }
  ]
}
```

### POST /api/planning/plan

Full planning pipeline: generates both strategies and actions for a goal.

**Request:**
```json
{
  "goal_description": "Fix the authentication timeout bug",
  "goal_bucket_id": 1,
  "context_fingerprint": 12345,
  "session_id": 42
}
```

**Response (200):**
```json
{
  "ok": true,
  "mode": "Full",
  "goal_description": "Fix the authentication timeout bug",
  "goal_bucket_id": 1,
  "strategy_candidates": [ "..." ],
  "action_candidates": [ "..." ]
}
```

### POST /api/planning/execute

Start execution tracking for a strategy (enables predictive validation).

**Request:**
```json
{
  "goal_description": "Fix the authentication timeout bug",
  "goal_bucket_id": 1,
  "context_fingerprint": 12345,
  "session_id": 42
}
```

**Response (200):**
```json
{
  "ok": true,
  "execution_id": 1
}
```

### POST /api/planning/validate

Validate an event against predicted world state. Triggers repair if prediction error exceeds threshold.

**Request:**
```json
{
  "execution_id": 1,
  "event": { "...full event object..." }
}
```

**Response (200):**
```json
{
  "ok": true,
  "prediction_error": {
    "total_z": 1.5,
    "event_z": 0.8,
    "memory_z": 0.3,
    "strategy_z": 0.4,
    "mismatch_layer": "event"
  },
  "repair_triggered": false,
  "repair_result": null
}
```

When `repair_triggered: true`:
```json
{
  "repair_result": {
    "scope": "strategy",
    "repaired_actions": 2,
    "repaired_strategies": 1
  }
}
```

### GET /api/world-model/stats

**Response (200):**
```json
{
  "enabled": true,
  "mode": "ScoringOnly",
  "running_mean": 0.45,
  "running_variance": 0.12,
  "total_scored": 500,
  "total_trained": 200,
  "avg_loss": 0.03,
  "is_warmed_up": true,
  "planning": {
    "strategy_generation_enabled": true,
    "action_generation_enabled": true
  }
}
```

When `enabled: false`, only `enabled`, `mode`, and `planning` fields are returned.

---

## Code Intelligence

Code intelligence provides AST-based structural understanding of source code. Source files are parsed into typed entities (functions, structs, traits, enums) and relationships (contains, imports, field_of, implements), which are ingested into the graph as `Concept` nodes with `CodeStructure` edges.

**Requires:** The `code-intelligence` Cargo feature and `enable_code_intelligence: true` in `GraphEngineConfig`. Without the feature, the endpoints still accept requests but skip AST parsing — events are processed through the standard pipeline only.

**Supported languages:** Rust (default). Python, TypeScript, JavaScript, Go available via feature flags (`lang-python`, `lang-typescript`, etc.).

### How It Works

```
CodeFile event → Standard Pipeline (graph node, claims, episodes)
                     │
                     └→ AST Parser (tree-sitter)
                          │
                          ├→ Concept nodes (Function, Struct, Enum, Trait, Module, Variable)
                          ├→ CodeStructure edges (contains, imports, field_of, returns, implements)
                          └→ About edges (event → defines → concept)
```

1. A `CodeFile` event is submitted with source code content
2. The event is processed through the standard pipeline (graph construction, episodes, claims)
3. If code intelligence is enabled, the source is parsed via tree-sitter
4. Extracted entities become `Concept` graph nodes (deduped by `qualified_name`)
5. Extracted relationships become `CodeStructure` edges with confidence scores
6. Concept nodes are linked to the source event via `About` edges with `predicate: "defines"`

**Claim extraction for code events** uses an AST-derived summary (entity names, signatures, imports, relationships) instead of raw source code. This keeps the LLM context small and focused on structural facts.

### Re-indexing Behavior

Submitting the same file again performs a full replace — the graph always reflects the latest version of the code:

| Scenario | Behavior |
|----------|----------|
| File unchanged | Existing nodes touched (timestamp updated), no new nodes created |
| Function signature changed | Existing Concept node updated in place (same node ID, properties overwritten) |
| New function added | New Concept node created, linked to event |
| Function deleted from file | Old Concept node **removed** from graph along with all its edges |
| Relationships changed | Old `CodeStructure` edges for this file **removed**, fresh edges added |

Nodes are deduped by `qualified_name` (e.g. `auth::login::authenticate`). When a file is re-indexed:

1. Entities present in both old and new parse → properties updated in place
2. Entities only in new parse → new Concept node created
3. Entities only in old parse (same `file_path`) → node and edges removed
4. All `CodeStructure` edges for this `file_path` are replaced (old removed, new added)

This means you can safely re-index files after every edit without accumulating stale nodes or duplicate edges.

**Claims on re-index:**

Graph nodes and edges are replaced immediately, but claims follow a different lifecycle:

| Scenario | Claim behavior |
|----------|---------------|
| Same claim extracted again | **Deduped** — existing claim gets `add_support()`, new one merged |
| Claim changed (e.g. new signature) | **Superseded** — old claim marked `Superseded`, new claim stored |
| Entity deleted (no new claim generated) | **Decays** — old claim is not actively removed, but fades via half-life decay (CodePattern: 365d, ApiContract: 180d, BugFix: 90d) |

This is by design: claims represent extracted knowledge that may still be relevant even after code changes (e.g. "we used the repository pattern" remains true even if specific functions change). Stale claims naturally lose confidence over time and are eventually purged by the maintenance loop (`purge_inactive_claims`).

### EventType: CodeFile

Submit a source file for AST analysis and graph ingestion.

#### POST /api/events/code-file

**Request:**
```json
{
  "agent_id": 1,
  "agent_type": "code-indexer",
  "session_id": 42,
  "file_path": "src/auth/login.rs",
  "content": "pub fn authenticate(user: &str, pass: &str) -> Result<Token, AuthError> { ... }",
  "language": "rust",
  "repository": "my-app",
  "git_ref": "main",
  "enable_ast": true,
  "enable_semantic": true
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `agent_id` | integer | yes | Agent identifier |
| `agent_type` | string | yes | Agent type label |
| `session_id` | integer | yes | Session identifier |
| `file_path` | string | yes | Path of the source file |
| `content` | string | yes | Full source code content |
| `language` | string | no | Language identifier (`"rust"`, `"python"`, etc.). Auto-detected from extension if omitted. |
| `repository` | string | no | Repository name for scoping |
| `git_ref` | string | no | Git ref (branch, tag, commit) |
| `enable_ast` | boolean | no | Enable AST parsing (default: false) |
| `enable_semantic` | boolean | no | Enable NER + claim extraction (default: false) |

**Response (200):**
```json
{
  "success": true,
  "event_id": 340282366920938463463374607431768211455,
  "nodes_created": 5,
  "patterns_detected": 0,
  "processing_time_ms": 45
}
```

`nodes_created` includes the event node plus any **new** AST-derived concept nodes. Nodes that already existed (matched by `qualified_name`) are updated in place and not counted here. On a re-index of an unchanged file, this will be `1` (the event node only).

**What gets created in the graph:**

| Source Code | Graph Node | ConceptType |
|-------------|-----------|-------------|
| `fn foo()` | Concept | Function |
| `pub fn bar(&self)` (in impl) | Concept | Function (Method) |
| `struct Point` | Concept | Class |
| `enum Color` | Concept | Enum |
| `trait Drawable` | Concept | Interface |
| `mod auth` | Concept | Module |
| `const MAX: usize` | Concept | Variable |
| `type Alias = Vec<u8>` | Concept | TypeAlias |

**Edges created:**

| Relationship | EdgeType | Confidence |
|-------------|----------|-----------|
| Module contains Function | CodeStructure (`contains`) | 1.0 |
| Struct contains Field | CodeStructure (`field_of`) | 1.0 |
| File imports Module | CodeStructure (`imports`) | 1.0 |
| Function returns Type | CodeStructure (`returns`) | 1.0 |
| Parameter has Type | CodeStructure (`parameter_type`) | 1.0 |
| Struct implements Trait | CodeStructure (`implements`) | 0.9 |

### EventType: CodeReview

Submit a code review comment, approval, or change request.

#### POST /api/events/code-review

**Request:**
```json
{
  "agent_id": 1,
  "agent_type": "code-reviewer",
  "session_id": 42,
  "review_id": "PR-123-review-1",
  "action": "comment",
  "body": "This function should handle the null case explicitly to avoid panics in production.",
  "file_path": "src/auth/login.rs",
  "line_range": [42, 50],
  "repository": "my-app",
  "title": "Add null safety to auth module",
  "enable_semantic": true
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `agent_id` | integer | yes | Agent identifier |
| `agent_type` | string | yes | Agent type label |
| `session_id` | integer | yes | Session identifier |
| `review_id` | string | yes | Unique review identifier |
| `action` | string | yes | One of: `"comment"`, `"approve"`, `"request_changes"` |
| `body` | string | yes | Review comment text |
| `file_path` | string | no | File being reviewed |
| `line_range` | [int, int] | no | Line range [start, end] |
| `repository` | string | yes | Repository name |
| `title` | string | no | Review/PR title |
| `enable_semantic` | boolean | no | Enable NER + claim extraction (default: false) |

**Response (200):**
```json
{
  "success": true,
  "event_id": 340282366920938463463374607431768211456,
  "nodes_created": 1,
  "patterns_detected": 0,
  "processing_time_ms": 8
}
```

Code reviews are processed through the standard pipeline. With `enable_semantic: true`, claims are extracted from the review body (e.g., "This function should handle null" becomes a `BugFix` or `CodePattern` claim).

### Code Search

Search for code entities in the graph by name, kind, language, or file path.

#### POST /api/code/search

**Request:**
```json
{
  "name": "authenticate",
  "kind": "function",
  "language": "rust",
  "file_path": "src/auth",
  "limit": 20
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | no | Substring match on entity name |
| `kind` | string | no | Filter by kind: `"function"`, `"class"`, `"enum"`, `"interface"`, `"module"`, `"variable"`, `"typealias"` |
| `language` | string | no | Filter by language |
| `file_path` | string | no | Substring match on file path |
| `limit` | integer | no | Max results (default: 50) |

All filters are optional. Omitting all filters returns all code entities up to the limit.

**Response (200):**
```json
{
  "entities": [
    {
      "name": "authenticate",
      "qualified_name": "auth::login::authenticate",
      "kind": "Function",
      "file_path": "src/auth/login.rs",
      "language": "rust",
      "line_range": [10, 25],
      "signature": "pub fn authenticate(user: &str, pass: &str) -> Result<Token, AuthError>",
      "doc_comment": "Authenticate a user with credentials.",
      "visibility": "pub"
    }
  ],
  "total": 1
}
```

### Code-Specific Claim Types

When claim extraction processes code events, it may produce these additional claim types:

| ClaimType | Half-life | Description |
|-----------|-----------|-------------|
| `CodePattern` | 365 days | Architectural decision or code pattern (e.g., "We use the repository pattern") |
| `ApiContract` | 180 days | API contract or interface specification (e.g., "The endpoint accepts JSON") |
| `BugFix` | 90 days | Bug fix or known issue (e.g., "Fixed null pointer in login flow") |

These are in addition to the standard claim types (Fact, Preference, Belief, Intention, Capability, Avoidance).

### Configuration

Enable code intelligence in `GraphEngineConfig`:

```rust
let mut config = GraphEngineConfig::default();
config.enable_code_intelligence = true;
```

Or via environment/server configuration:

| Config Field | Type | Default | Description |
|-------------|------|---------|-------------|
| `enable_code_intelligence` | bool | false | Enable AST parsing for CodeFile events |

The `code-intelligence` Cargo feature must also be enabled at compile time:

```bash
cargo build --features code-intelligence
```

---

## Admin

### POST /api/admin/export

Export entire database as streaming binary (versioned v2 format).

**Response:** `application/octet-stream` with `Content-Disposition: attachment; filename="minnsdb-export.bin"`

### POST /api/admin/import?mode=replace

Import database from streaming binary.

**Query Parameters:**
- `mode`: `"replace"` (wipe and import) or `"merge"` (upsert). Default: `"replace"`.

**Request Body:** Binary v2 format (from export).

**Response (200):**
```json
{
  "success": true,
  "memories_imported": 120,
  "strategies_imported": 35,
  "graph_nodes_imported": 12000,
  "graph_edges_imported": 28000,
  "total_records": 40155,
  "mode": "replace"
}
```

---

## Health

### GET /api/health

**Response (200):**
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "uptime_seconds": 3600,
  "is_healthy": true,
  "node_count": 12000,
  "edge_count": 28000,
  "processing_rate": 145.5
}
```

### GET /

Returns ASCII endpoint listing.

### GET /docs

Returns link to this document.

---

## Error Handling

All errors return JSON:

```json
{
  "error": "Description of what went wrong"
}
```

HTTP status codes:
- `200` - Success
- `400` - Bad request / validation error
- `404` - Resource not found
- `500` - Internal server error

---

## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `NER_SERVICE_URL` | `http://localhost:8081/ner` | Full URL to NER service (with protocol) |
| `NER_REQUEST_TIMEOUT_MS` | `5000` | NER HTTP timeout in ms |
| `LLM_API_KEY` | - | OpenAI-compatible key for claims + planning |
| `LLM_MODEL` | `gpt-4o-mini` | LLM model name |
| `PLANNING_LLM_API_KEY` | `LLM_API_KEY` | Separate key for planning LLM |
| `PLANNING_LLM_PROVIDER` | `openai` | LLM provider for planning |
| `SERVER_HOST` | `0.0.0.0` | Bind address |
| `SERVER_PORT` | `8080` | Bind port |
| `RUST_LOG` | `info` | Log level |
| `REDB_CACHE_SIZE_MB` | `256` | ReDB page cache size |
| `MEMORY_CACHE_SIZE` | `10000` | In-memory cache entries |
| `STRATEGY_CACHE_SIZE` | `5000` | Strategy cache entries |
| `ENABLE_WORLD_MODEL` | `false` | Enable energy-based world model |
| `WORLD_MODEL_MODE` | `Disabled` | `Shadow`, `ScoringOnly`, `ScoringAndReranking`, `Full`, `Disabled` |
| `ENABLE_STRATEGY_GENERATION` | `false` | Enable LLM strategy generation |
| `ENABLE_ACTION_GENERATION` | `false` | Enable LLM action generation |
| `REPAIR_ENABLED` | `false` | Auto-repair on high prediction error |
| `ENABLE_NLQ_HINT` | `false` | Enable LLM advisory classifier for NLQ intent routing |
| `NLQ_HINT_API_KEY` | `LLM_API_KEY` | API key for NLQ hint classifier (falls back to `LLM_API_KEY`) |
| `NLQ_HINT_PROVIDER` | `openai` | LLM provider for NLQ hint (`openai` or `anthropic`) |
| `NLQ_HINT_MODEL` | `gpt-4o-mini` | Model name for NLQ hint classifier |
| `ENABLE_METADATA_NORMALIZATION` | `false` | Enable LLM fallback for metadata key normalization |
| `METADATA_NORMALIZATION_MODEL` | `NLQ_HINT_MODEL` | Model name for metadata normalization LLM (falls back to `NLQ_HINT_MODEL`) |
| `METADATA_NORMALIZATION_TIMEOUT_MS` | `3000` | Timeout for LLM metadata normalization calls |
| `SUBSCRIPTION_INTERVAL_MS` | `50` | Background subscription processing interval in ms |
| `CORS_ALLOWED_ORIGINS` | - | Comma-separated allowed CORS origins (for WebSocket and REST) |

### World Model Modes

| Mode | Behavior |
|------|----------|
| `Disabled` | No world model scoring |
| `Shadow` | Score events but don't affect ranking |
| `ScoringOnly` | Score and log, no re-ranking |
| `ScoringAndReranking` | Score and re-rank strategy candidates |
| `Full` | Score, re-rank, and trigger repair on prediction error |

### Metadata Normalization

The pipeline includes a multi-tier metadata key normalizer that automatically maps arbitrary metadata keys to canonical roles for structured memory detection (state machines and ledgers). This works transparently — SDKs can send events with any key naming convention and the pipeline will resolve them.

**Resolution tiers (highest priority first):**

| Tier | Method | Example |
|------|--------|---------|
| 1 | Exact canonical match | `"entity"` → Entity, `"amount"` → Amount |
| 2 | Built-in alias match | `"sender"` → From, `"recipient"` → To, `"status"` → NewState |
| 3 | Token-aware stem match | `"order_status"` → NewState (token "status" matches stem) |
| 4 | Bigram similarity (≥0.5) | `"recipent"` (typo) → To (bigram match to "recipient") |
| 5 | LLM fallback (if enabled) | Foreign/custom keys resolved via LLM JSON mapping |

**Canonical roles:** `entity`, `new_state`, `old_state`, `from`, `to`, `amount`, `direction`, `description`

**Type guards:** The `amount` role requires a numeric value — string values like `"ten"` are rejected even if the key matches.

**Custom aliases:** Configure `metadata_alias_config.custom_mappings` to map domain-specific keys (e.g., `"absender"` → `"from"` for German metadata).

When `ENABLE_METADATA_NORMALIZATION=true`, an LLM fallback activates for events where alias resolution finds fewer than 2 roles but the event has 2+ metadata keys. The LLM classifies remaining keys with a 3-second timeout (configurable). Results are cached (LRU, 500 entries) to avoid repeated calls for the same key sets.

---

## SDK Integration Checklist

### Agent Telemetry (Event Pipeline)

Use this path when your agent needs the full learning loop: events → episodes → memories → strategies → reinforcement learning.

1. **Minimal integration:** `POST /api/events/simple` to send events, `GET /api/suggestions` to get next actions
2. **Memory-aware:** Add `POST /api/memories/context` to retrieve relevant memories before acting
3. **Strategy-aware:** Add `GET /api/strategies/agent/:id` and `POST /api/strategies/similar`
4. **Full telemetry:** Use `Learning` events (`MemoryUsed`, `StrategyUsed`, `Outcome`) to close the feedback loop
5. **Semantic search:** Enable `enable_semantic: true` on events, then use `POST /api/search` with Hybrid mode
6. **Natural language queries:** Use `POST /api/nlq` to ask questions in plain English. Pass `session_id` for conversational follow-ups
7. **MinnsQL queries:** Use `POST /api/query` for structured graph queries with temporal filtering, aggregation, and Allen's interval algebra predicates
8. **Reactive subscriptions (REST):** Use `POST /api/subscriptions` to create a live query, then `GET /api/subscriptions/:id/poll` for incremental updates (inserts/deletes). Use `DELETE /api/subscriptions/:id` to unsubscribe
9. **Reactive subscriptions (WebSocket):** Connect to `GET /api/subscriptions/ws` for real-time push. Send `{"type": "subscribe", "query": "..."}` to start, receive `{"type": "update", ...}` as the graph changes
10. **Structured memory:** Use `/api/structured-memory/*` endpoints for domain-specific data (ledgers, state machines, preference lists, trees)
11. **Typed state changes:** Use `POST /api/events/state-change` to emit state transitions — auto-detected and tracked in structured memory state machines
12. **Typed transactions:** Use `POST /api/events/transaction` to emit financial/quantity transactions — auto-detected and appended to structured memory ledgers
13. **Code events:** Set `is_code: true` on events containing source code for code-aware tokenization and indexing
14. **Planning:** Enable world model + strategy generation, use `POST /api/planning/plan` for goal-driven agents

### Conversation Ingestion

Conversations go through the full event pipeline with LLM compaction for fact extraction. Events, episodes, memories, and graph edges are all created.

15. **Conversation ingestion:** Use `POST /api/conversations/ingest` to ingest multi-session conversations. Requires an LLM client. Use the same `case_id` across calls for incremental ingestion with stable entity resolution and automatic deduplication.
16. **Single message ingestion:** Use `POST /api/messages` to send individual messages with automatic buffering and compaction.
17. **Multi-source queries:** Use `POST /api/nlq` to query the graph. Responses include `related_memories` and `related_strategies` from the episodic stores for full context enrichment.
18. **Incremental pattern:** Send messages as they arrive with the same `case_id`. The server preserves conversation state and deduplicates already-processed messages automatically. No client-side state management needed.

### Multi-Agent Coordination

19. **Agent registry:** Use `POST /api/agents/register` to register agents with capabilities, then `GET /api/agents` to discover peers.
20. **Workflows:** Use `POST /api/workflows` to create multi-step workflows with dependency tracking, then transition steps with `/api/workflows/:id/steps/:step_id/transition`.
21. **Ontology evolution:** Use `POST /api/ontology/discover` for automatic property discovery, review proposals with `/api/ontology/proposals`, and upload custom TTL with `POST /api/ontology/upload`.

### Configuration

22. **NLQ hint classifier:** Set `ENABLE_NLQ_HINT=true` for LLM-assisted intent routing (improves structured memory detection)
23. **Metadata normalization:** Alias-based normalization is always active. Set `ENABLE_METADATA_NORMALIZATION=true` for LLM fallback on unrecognized metadata keys
