# EventGraphDB - Complete API Specification

**Version:** 1.0.0
**Last Updated:** 2026-01-20
**Protocol:** HTTP/JSON over WebSocket and REST

---

## Table of Contents

1. [Overview](#overview)
2. [Authentication](#authentication)
3. [Core Endpoints](#core-endpoints)
4. [WebSocket API](#websocket-api)
5. [Data Models](#data-models)
6. [Error Handling](#error-handling)
7. [Rate Limiting](#rate-limiting)
8. [Examples](#examples)

---

## Overview

EventGraphDB provides a unified API for event ingestion, graph queries, memory retrieval, and strategy execution. The API supports both REST and WebSocket protocols.

### Base URL
```
Production: https://eventgraph.hertz.app
Development: http://localhost:8080
```

### Content Type
```
Content-Type: application/json
```

### API Versioning
```
/api/v1/*
```

---

## Authentication

### Auth Middleware (Recommended)

All requests must include authentication in the header:

```http
Authorization: Bearer <JWT_TOKEN>
```

or

```http
X-API-Key: <API_KEY>
```

### JWT Token Structure
```json
{
  "sub": "agent_12345",
  "agent_type": "autonomous",
  "tenant_id": "tenant_001",
  "exp": 1735689600,
  "iat": 1735603200,
  "scopes": ["events:write", "graph:read", "memory:read"]
}
```

### Scopes
- `events:write` - Submit events
- `events:read` - Query events
- `graph:read` - Query graph structure
- `graph:write` - Modify graph (admin only)
- `memory:read` - Query memories
- `memory:write` - Create memories (usually auto-generated)
- `strategy:read` - Query strategies
- `strategy:write` - Create strategies
- `admin:*` - Full access

### Getting a Token
```http
POST /api/v1/auth/token
Content-Type: application/json

{
  "agent_id": "agent_12345",
  "api_key": "sk_live_..."
}
```

**Response:**
```json
{
  "token": "eyJhbGciOiJIUzI1NiIs...",
  "expires_in": 86400,
  "token_type": "Bearer"
}
```

---

## Core Endpoints

### 1. Event Ingestion

#### POST /api/v1/events

Submit a single event to the system.

**Request:**
```http
POST /api/v1/events
Authorization: Bearer <token>
Content-Type: application/json

{
  "event_type": "Cognitive",
  "subtype": "Decision",
  "agent_id": "agent_12345",
  "agent_type": "Autonomous",
  "session_id": "session_67890",
  "timestamp": 1735603200000,
  "context_hash": 123456789,
  "significance": 0.85,
  "metadata": {
    "decision_id": "dec_001",
    "options_considered": 3,
    "chosen_option": "optimize_path",
    "confidence": 0.92
  },
  "content": {
    "problem": "Route optimization required",
    "solution": "Applied A* algorithm with heuristic",
    "outcome": "30% faster route"
  }
}
```

**Response (201 Created):**
```json
{
  "event_id": "evt_abc123",
  "processed_at": 1735603201000,
  "graph_nodes_created": 2,
  "patterns_detected": ["decision_pattern_01"],
  "memory_formed": false,
  "strategy_extracted": false
}
```

**Event Types:**
- `Cognitive` - Thinking, decisions, reasoning
  - Subtypes: `Decision`, `Reasoning`, `Planning`, `Reflection`
- `Action` - Physical or digital actions
  - Subtypes: `Execute`, `Modify`, `Create`, `Delete`
- `Observation` - Sensory input, environment changes
  - Subtypes: `Visual`, `Auditory`, `DataReceived`, `StateChange`
- `Communication` - Agent-to-agent or agent-to-human
  - Subtypes: `Request`, `Response`, `Broadcast`, `Feedback`
- `Learning` - Learning outcomes, insights
  - Subtypes: `Insight`, `SkillAcquired`, `PatternRecognized`, `ErrorCorrected`

#### POST /api/v1/events/batch

Submit multiple events in a single request.

**Request:**
```http
POST /api/v1/events/batch
Authorization: Bearer <token>
Content-Type: application/json

{
  "events": [
    { /* event 1 */ },
    { /* event 2 */ },
    { /* event 3 */ }
  ]
}
```

**Response (201 Created):**
```json
{
  "batch_id": "batch_xyz789",
  "events_processed": 3,
  "events_failed": 0,
  "processing_time_ms": 45,
  "results": [
    { "event_id": "evt_001", "status": "success" },
    { "event_id": "evt_002", "status": "success" },
    { "event_id": "evt_003", "status": "success" }
  ]
}
```

### 2. Graph Queries

#### GET /api/v1/graph/nodes/{node_id}

Retrieve a specific node from the graph.

**Request:**
```http
GET /api/v1/graph/nodes/node_12345
Authorization: Bearer <token>
```

**Response (200 OK):**
```json
{
  "id": "node_12345",
  "node_type": "Action",
  "label": "optimize_route",
  "context_hash": 123456789,
  "created_at": 1735603200000,
  "properties": {
    "algorithm": "A*",
    "execution_time_ms": 15,
    "success": true
  },
  "edges": {
    "outgoing": ["node_12346", "node_12347"],
    "incoming": ["node_12344"]
  }
}
```

#### GET /api/v1/graph/traverse

Traverse the graph from a starting node.

**Request:**
```http
GET /api/v1/graph/traverse?start=node_12345&algorithm=bfs&max_depth=3
Authorization: Bearer <token>
```

**Query Parameters:**
- `start` (required) - Starting node ID
- `algorithm` - `bfs` (breadth-first) or `dfs` (depth-first)
- `max_depth` - Maximum traversal depth (default: 10)
- `node_types` - Filter by node types (comma-separated)
- `context_hash` - Filter by context

**Response (200 OK):**
```json
{
  "start_node": "node_12345",
  "algorithm": "bfs",
  "max_depth": 3,
  "nodes_visited": 15,
  "path": [
    { "node_id": "node_12345", "depth": 0, "type": "Action" },
    { "node_id": "node_12346", "depth": 1, "type": "Observation" },
    { "node_id": "node_12347", "depth": 1, "type": "Decision" }
  ],
  "edges": [
    { "from": "node_12345", "to": "node_12346", "type": "Temporal", "weight": 1.0 }
  ]
}
```

#### POST /api/v1/graph/query

Advanced graph query with filters.

**Request:**
```http
POST /api/v1/graph/query
Authorization: Bearer <token>
Content-Type: application/json

{
  "node_types": ["Cognitive", "Action"],
  "context_hash": 123456789,
  "time_range": {
    "start": 1735603000000,
    "end": 1735603200000
  },
  "significance_threshold": 0.7,
  "max_results": 100
}
```

**Response (200 OK):**
```json
{
  "query_id": "qry_abc123",
  "results": [
    { /* node 1 */ },
    { /* node 2 */ }
  ],
  "total_matches": 47,
  "returned": 100,
  "execution_time_ms": 23
}
```

### 3. Memory Queries

#### GET /api/v1/memories

Retrieve memories formed from episodes.

**Request:**
```http
GET /api/v1/memories?context_hash=123456789&limit=10
Authorization: Bearer <token>
```

**Query Parameters:**
- `context_hash` - Filter by context
- `goal_bucket` - Filter by goal bucket
- `significance_min` - Minimum significance (0.0-1.0)
- `limit` - Max results (default: 50, max: 1000)
- `offset` - Pagination offset

**Response (200 OK):**
```json
{
  "memories": [
    {
      "memory_id": "mem_12345",
      "memory_type": "Episodic",
      "context_hash": 123456789,
      "goal_bucket_id": 42,
      "significance": 0.87,
      "formed_at": 1735603200000,
      "episode_id": "ep_67890",
      "summary": "Successfully optimized route using A* algorithm",
      "key_events": ["evt_001", "evt_002", "evt_003"],
      "outcome": "Success",
      "confidence": 0.92,
      "access_count": 5,
      "last_accessed": 1735689600000
    }
  ],
  "total": 25,
  "limit": 10,
  "offset": 0
}
```

#### GET /api/v1/memories/{memory_id}

Retrieve a specific memory with full details.

**Request:**
```http
GET /api/v1/memories/mem_12345
Authorization: Bearer <token>
```

**Response (200 OK):**
```json
{
  "memory_id": "mem_12345",
  "memory_type": "Episodic",
  "context_hash": 123456789,
  "goal_bucket_id": 42,
  "significance": 0.87,
  "formed_at": 1735603200000,
  "episode": {
    "episode_id": "ep_67890",
    "start_event": "evt_001",
    "end_event": "evt_003",
    "duration_ms": 1500,
    "event_count": 3
  },
  "events": [
    { /* full event 1 */ },
    { /* full event 2 */ },
    { /* full event 3 */ }
  ],
  "patterns": ["decision_under_time_pressure"],
  "related_memories": ["mem_11111", "mem_22222"],
  "related_strategies": ["strat_001"]
}
```

### 4. Strategy Queries

#### GET /api/v1/strategies

Retrieve strategies extracted from successful episodes.

**Request:**
```http
GET /api/v1/strategies?context_pattern=route_optimization&limit=10
Authorization: Bearer <token>
```

**Query Parameters:**
- `context_pattern` - Context pattern to match
- `goal_bucket` - Filter by goal bucket
- `min_success_rate` - Minimum success rate (0.0-1.0)
- `limit` - Max results (default: 50)

**Response (200 OK):**
```json
{
  "strategies": [
    {
      "strategy_id": "strat_001",
      "context_pattern": ["route_needed", "time_constraint"],
      "reasoning_steps": [
        "Analyze current position and destination",
        "Identify constraints (time, fuel, traffic)",
        "Apply A* with traffic-aware heuristic",
        "Validate route feasibility"
      ],
      "success_rate": 0.94,
      "usage_count": 47,
      "avg_outcome_score": 0.89,
      "formed_at": 1735500000000,
      "last_used": 1735603200000,
      "source_episodes": ["ep_001", "ep_002", "ep_003"]
    }
  ],
  "total": 12,
  "limit": 10
}
```

#### POST /api/v1/strategies/suggest

Get strategy suggestions for a given context.

**Request:**
```http
POST /api/v1/strategies/suggest
Authorization: Bearer <token>
Content-Type: application/json

{
  "context": {
    "goal": "optimize_route",
    "constraints": ["time_limited", "fuel_limited"],
    "current_state": "location_A"
  },
  "top_k": 3
}
```

**Response (200 OK):**
```json
{
  "suggestions": [
    {
      "strategy_id": "strat_001",
      "match_score": 0.95,
      "success_rate": 0.94,
      "reasoning_steps": [ /* ... */ ],
      "confidence": 0.92
    },
    {
      "strategy_id": "strat_002",
      "match_score": 0.87,
      "success_rate": 0.89,
      "reasoning_steps": [ /* ... */ ],
      "confidence": 0.85
    }
  ],
  "context_hash": 123456789,
  "query_time_ms": 12
}
```

### 5. Analytics & Metrics

#### GET /api/v1/analytics/stats

Get overall system statistics.

**Request:**
```http
GET /api/v1/analytics/stats
Authorization: Bearer <token>
```

**Response (200 OK):**
```json
{
  "events": {
    "total": 125000,
    "last_24h": 3500,
    "avg_per_hour": 145
  },
  "graph": {
    "nodes": 45000,
    "edges": 89000,
    "avg_degree": 1.98,
    "communities": 142
  },
  "memories": {
    "total": 1250,
    "episodic": 980,
    "semantic": 270,
    "avg_significance": 0.76
  },
  "strategies": {
    "total": 87,
    "avg_success_rate": 0.82,
    "total_usage": 5400
  },
  "storage": {
    "db_size_mb": 1024,
    "cache_hit_rate": 0.94,
    "avg_query_ms": 8.5
  }
}
```

#### GET /api/v1/analytics/agent/{agent_id}

Get analytics for a specific agent.

**Request:**
```http
GET /api/v1/analytics/agent/agent_12345
Authorization: Bearer <token>
```

**Response (200 OK):**
```json
{
  "agent_id": "agent_12345",
  "agent_type": "Autonomous",
  "active_since": 1735000000000,
  "events_submitted": 5432,
  "memories_formed": 87,
  "strategies_used": 23,
  "avg_decision_quality": 0.87,
  "learning_rate": 0.92,
  "top_strategies": [
    { "strategy_id": "strat_001", "usage_count": 45 }
  ],
  "performance_trend": [
    { "date": "2026-01-15", "score": 0.83 },
    { "date": "2026-01-16", "score": 0.85 },
    { "date": "2026-01-17", "score": 0.87 }
  ]
}
```

---

## WebSocket API

### Connection

```javascript
const ws = new WebSocket('ws://localhost:8080/ws');

// Send auth after connection
ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'auth',
    token: 'Bearer <JWT_TOKEN>'
  }));
};
```

### Message Types

#### 1. Event Submission
```json
{
  "type": "submit_event",
  "event": { /* event object */ }
}
```

**Response:**
```json
{
  "type": "event_processed",
  "event_id": "evt_abc123",
  "graph_nodes_created": 2
}
```

#### 2. Real-time Graph Updates
```json
{
  "type": "subscribe_graph",
  "filters": {
    "agent_id": "agent_12345",
    "node_types": ["Cognitive", "Action"]
  }
}
```

**Streaming Response:**
```json
{
  "type": "graph_update",
  "node": { /* node object */ },
  "timestamp": 1735603200000
}
```

#### 3. Memory Notifications
```json
{
  "type": "subscribe_memories",
  "agent_id": "agent_12345"
}
```

**Streaming Response:**
```json
{
  "type": "memory_formed",
  "memory": { /* memory object */ },
  "significance": 0.87
}
```

---

## Data Models

### Event
```typescript
interface Event {
  event_type: 'Cognitive' | 'Action' | 'Observation' | 'Communication' | 'Learning';
  subtype: string;
  agent_id: string;
  agent_type: 'Autonomous' | 'SemiAutonomous' | 'Reactive' | 'Human';
  session_id: string;
  timestamp: number; // Unix timestamp in milliseconds
  context_hash: number;
  significance: number; // 0.0-1.0
  metadata: Record<string, any>;
  content: Record<string, any>;
}
```

### Node
```typescript
interface GraphNode {
  id: string;
  node_type: 'Event' | 'Action' | 'Observation' | 'Cognitive' | 'Communication' | 'Learning' | 'Context' | 'Pattern';
  label: string;
  context_hash: number;
  created_at: number;
  properties: Record<string, any>;
}
```

### Edge
```typescript
interface GraphEdge {
  from: string;
  to: string;
  edge_type: 'Causality' | 'Temporal' | 'Similarity' | 'Containment' | 'Reference' | 'Transition';
  weight: number;
  confidence: number;
  created_at: number;
}
```

### Memory
```typescript
interface Memory {
  memory_id: string;
  memory_type: 'Episodic' | 'Semantic' | 'Procedural';
  context_hash: number;
  goal_bucket_id: number;
  significance: number;
  formed_at: number;
  episode_id: string;
  summary: string;
  key_events: string[];
  outcome: 'Success' | 'Failure' | 'Neutral';
  confidence: number;
  access_count: number;
  last_accessed: number;
}
```

### Strategy
```typescript
interface Strategy {
  strategy_id: string;
  context_pattern: string[];
  reasoning_steps: string[];
  success_rate: number;
  usage_count: number;
  avg_outcome_score: number;
  formed_at: number;
  last_used: number;
  source_episodes: string[];
}
```

---

## Error Handling

### Error Response Format
```json
{
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid event type",
    "details": {
      "field": "event_type",
      "received": "InvalidType",
      "expected": ["Cognitive", "Action", "Observation", "Communication", "Learning"]
    },
    "request_id": "req_abc123",
    "timestamp": 1735603200000
  }
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `UNAUTHORIZED` | 401 | Missing or invalid authentication |
| `FORBIDDEN` | 403 | Insufficient permissions |
| `NOT_FOUND` | 404 | Resource not found |
| `VALIDATION_ERROR` | 400 | Invalid request data |
| `RATE_LIMIT_EXCEEDED` | 429 | Too many requests |
| `INTERNAL_ERROR` | 500 | Server error |
| `DATABASE_ERROR` | 500 | Database operation failed |
| `TIMEOUT` | 504 | Request timeout |

---

## Rate Limiting

### Limits (per API key)

| Tier | Events/min | Queries/min | WebSocket Connections |
|------|-----------|-------------|----------------------|
| Free | 60 | 120 | 1 |
| Pro | 600 | 1200 | 5 |
| Enterprise | Unlimited | Unlimited | Unlimited |

### Rate Limit Headers
```http
X-RateLimit-Limit: 600
X-RateLimit-Remaining: 547
X-RateLimit-Reset: 1735603260
```

### Rate Limit Exceeded Response
```json
{
  "error": {
    "code": "RATE_LIMIT_EXCEEDED",
    "message": "Rate limit exceeded. Retry after 45 seconds.",
    "retry_after": 45,
    "limit": 600,
    "window": "1 minute"
  }
}
```

---

## Examples

### Example 1: Submit Event and Check Memory Formation

```javascript
// 1. Submit event
const response = await fetch('https://eventgraph.hertz.app/api/v1/events', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer <token>',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    event_type: 'Cognitive',
    subtype: 'Decision',
    agent_id: 'agent_12345',
    agent_type: 'Autonomous',
    session_id: 'session_67890',
    timestamp: Date.now(),
    context_hash: 123456789,
    significance: 0.9,
    metadata: { decision_id: 'dec_001' },
    content: { problem: 'Route optimization', solution: 'A* algorithm' }
  })
});

const result = await response.json();
console.log('Event ID:', result.event_id);

// 2. Wait for memory formation (async process)
setTimeout(async () => {
  const memories = await fetch(
    `https://eventgraph.hertz.app/api/v1/memories?context_hash=123456789`,
    {
      headers: { 'Authorization': 'Bearer <token>' }
    }
  );
  const data = await memories.json();
  console.log('Memories formed:', data.memories.length);
}, 2000);
```

### Example 2: Real-time Graph Monitoring via WebSocket

```javascript
const ws = new WebSocket('ws://localhost:8080/ws');

ws.onopen = () => {
  // Authenticate
  ws.send(JSON.stringify({
    type: 'auth',
    token: 'Bearer <token>'
  }));

  // Subscribe to graph updates
  ws.send(JSON.stringify({
    type: 'subscribe_graph',
    filters: {
      agent_id: 'agent_12345',
      node_types: ['Cognitive']
    }
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);

  if (data.type === 'graph_update') {
    console.log('New node:', data.node);
  } else if (data.type === 'memory_formed') {
    console.log('Memory formed:', data.memory);
  }
};
```

### Example 3: Query Similar Strategies

```javascript
const response = await fetch('https://eventgraph.hertz.app/api/v1/strategies/suggest', {
  method: 'POST',
  headers: {
    'Authorization': 'Bearer <token>',
    'Content-Type': 'application/json'
  },
  body: JSON.stringify({
    context: {
      goal: 'optimize_route',
      constraints: ['time_limited'],
      current_state: 'location_A'
    },
    top_k: 5
  })
});

const { suggestions } = await response.json();
console.log('Top strategy:', suggestions[0].strategy_id);
console.log('Success rate:', suggestions[0].success_rate);
console.log('Steps:', suggestions[0].reasoning_steps);
```

---

## API Client Libraries

### JavaScript/TypeScript
```bash
npm install @eventgraph/client
```

```typescript
import { EventGraphClient } from '@eventgraph/client';

const client = new EventGraphClient({
  apiKey: 'sk_live_...',
  baseUrl: 'https://eventgraph.hertz.app'
});

await client.events.submit({
  event_type: 'Cognitive',
  // ...
});
```

### Python
```bash
pip install eventgraph-client
```

```python
from eventgraph import EventGraphClient

client = EventGraphClient(api_key='sk_live_...')
result = client.events.submit({
    'event_type': 'Cognitive',
    # ...
})
```

### Rust
```toml
[dependencies]
eventgraph-client = "1.0"
```

```rust
use eventgraph_client::EventGraphClient;

let client = EventGraphClient::new("sk_live_...");
client.events().submit(event).await?;
```

---

## Changelog

### v2.0.0 (2026-02-23)
- Graph algorithms: Louvain communities, label propagation, temporal reachability, centrality, PPR, random walks
- Planning endpoints: strategy generation, action generation, plan-for-goal, execution tracking, validation
- World model: energy-based scoring with configurable modes
- Search: unified keyword/semantic/hybrid with fusion strategies
- Claims: semantic claim extraction, search, and management
- Analytics: communities, centrality, PageRank, reachability, causal paths
- Admin: binary export/import with replace/merge modes
- Bounded sharding (NUM_SHARDS=256), delta persistence, streaming queries
- See **[API_REFERENCE.md](API_REFERENCE.md)** for complete endpoint documentation

### v1.0.0 (2026-01-20)
- Initial API release
- Event ingestion endpoints
- Graph query endpoints
- Memory and strategy endpoints
- WebSocket support
- Authentication and rate limiting
