# EventGraphDB API Reference

**Complete REST API documentation for SDK integration**

Version: 2.1.0
Last Updated: 2026-02-25

---

## Table of Contents

1. [Overview](#overview)
2. [Core Types](#core-types)
3. [Events](#events)
4. [Memories](#memories)
5. [Strategies](#strategies)
6. [Graph](#graph)
7. [Analytics](#analytics)
8. [Search](#search)
9. [Natural Language Query (NLQ)](#natural-language-query-nlq)
10. [Claims](#claims)
11. [Structured Memory](#structured-memory)
12. [Planning & World Model](#planning--world-model)
13. [Admin](#admin)
14. [Health](#health)
15. [Error Handling](#error-handling)
16. [Configuration](#configuration)

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
    "variables": {"lang": "rust", "project": "eventgraphdb"},
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

## Claims

Claims are semantic assertions extracted by the NER + LLM pipeline from events.

### GET /api/claims?limit=10&event_id=123

List active claims, optionally filtered by source event.

**Response (200):**
```json
[
  {
    "claim_id": 1,
    "claim_text": "The user prefers null checks over optional chaining",
    "confidence": 0.92,
    "source_event_id": 123456,
    "similarity": null,
    "evidence_spans": [
      {"start": 10, "end": 45, "text": "prefers null checks"}
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
      {"entity_text": "null checks", "entity_type": "Concept", "confidence": 0.95}
    ]
  }
]
```

**Claim types:** `"Preference"`, `"Fact"`, `"Belief"`, `"Intention"`, `"Capability"`

### GET /api/claims/:id

Get a single claim by ID. Same response shape as list item.

### POST /api/claims/search

Semantic search for claims using embeddings.

**Request:**
```json
{
  "query_text": "error handling preferences",
  "top_k": 10,
  "min_similarity": 0.7
}
```

**Response:** Array of claim objects with `similarity` field populated.

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

## Admin

### POST /api/admin/export

Export entire database as streaming binary (versioned v2 format).

**Response:** `application/octet-stream` with `Content-Disposition: attachment; filename="eventgraphdb-export.bin"`

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

1. **Minimal integration:** `POST /api/events/simple` to send events, `GET /api/suggestions` to get next actions
2. **Memory-aware:** Add `POST /api/memories/context` to retrieve relevant memories before acting
3. **Strategy-aware:** Add `GET /api/strategies/agent/:id` and `POST /api/strategies/similar`
4. **Full telemetry:** Use `Learning` events (`MemoryUsed`, `StrategyUsed`, `Outcome`) to close the feedback loop
5. **Semantic search:** Enable `enable_semantic: true` on events, then use `POST /api/search` with Hybrid mode
6. **Natural language queries:** Use `POST /api/nlq` to ask questions in plain English. Pass `session_id` for conversational follow-ups
7. **Structured memory:** Use `/api/structured-memory/*` endpoints for domain-specific data (ledgers, state machines, preference lists, trees)
8. **Typed state changes:** Use `POST /api/events/state-change` to emit state transitions — auto-detected and tracked in structured memory state machines
9. **Typed transactions:** Use `POST /api/events/transaction` to emit financial/quantity transactions — auto-detected and appended to structured memory ledgers
10. **Code events:** Set `is_code: true` on events containing source code for code-aware tokenization and indexing
11. **Planning:** Enable world model + strategy generation, use `POST /api/planning/plan` for goal-driven agents
12. **NLQ hint classifier:** Set `ENABLE_NLQ_HINT=true` for LLM-assisted intent routing (improves structured memory detection)
13. **Metadata normalization:** Alias-based normalization is always active. Set `ENABLE_METADATA_NORMALIZATION=true` for LLM fallback on unrecognized metadata keys
