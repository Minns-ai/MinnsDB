# EventGraphDB API Reference

**Complete REST API documentation for SDK integration**

Version: 2.0.0
Last Updated: 2026-02-23

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
9. [Claims](#claims)
10. [Planning & World Model](#planning--world-model)
11. [Admin](#admin)
12. [Health](#health)
13. [Error Handling](#error-handling)
14. [Configuration](#configuration)

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
  "segment_pointer": null
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

### World Model Modes

| Mode | Behavior |
|------|----------|
| `Disabled` | No world model scoring |
| `Shadow` | Score events but don't affect ranking |
| `ScoringOnly` | Score and log, no re-ranking |
| `ScoringAndReranking` | Score and re-rank strategy candidates |
| `Full` | Score, re-rank, and trigger repair on prediction error |

---

## SDK Integration Checklist

1. **Minimal integration:** `POST /api/events/simple` to send events, `GET /api/suggestions` to get next actions
2. **Memory-aware:** Add `POST /api/memories/context` to retrieve relevant memories before acting
3. **Strategy-aware:** Add `GET /api/strategies/agent/:id` and `POST /api/strategies/similar`
4. **Full telemetry:** Use `Learning` events (`MemoryUsed`, `StrategyUsed`, `Outcome`) to close the feedback loop
5. **Semantic search:** Enable `enable_semantic: true` on events, then use `POST /api/search` with Hybrid mode
6. **Planning:** Enable world model + strategy generation, use `POST /api/planning/plan` for goal-driven agents
