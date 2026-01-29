# EventGraphDB API Specification

EventGraphDB is a self-evolving agentic database that transforms discrete events into a semantic graph. This specification details the REST API for ingesting events, retrieving memories, and querying the learned graph.

## Base URL
Default: `http://localhost:3000`

## Core Concepts
- **Events**: Atomic units of interaction (Action, Observation, Communication, etc.).
- **Episodes**: Automatically detected sequences of related events.
- **Memories**: Long-term storage of significant episodes and contexts.
- **Strategies**: Learned behavioral patterns derived from successful outcomes.
- **Claims**: Atomic facts extracted from events via semantic memory (NER + LLM).

---

## 1. Events

### Post Event
Ingests a new event into the graph engine for processing and inference.

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
        "outcome": { "Success": { "result": "..." } }
      }
    },
    "causality_chain": [],
    "context": {
      "fingerprint": 987654321,
      "active_goals": [],
      "resources": { ... }
    },
    "metadata": {}
  },
  "enable_semantic": true
}
```

**Response:**
```json
{
  "success": true,
  "nodes_created": 5,
  "patterns_detected": 1,
  "processing_time_ms": 45
}
```

---

## 2. Memories & Strategies

### Get Agent Memories
Retrieves long-term memories for a specific agent.

**Endpoint:** `GET /api/memories/agent/:agent_id?limit=10`

### Get Contextual Memories
Finds memories similar to a provided environmental context.

**Endpoint:** `POST /api/memories/context`

**Request Body:**
```json
{
  "context": { "fingerprint": 987654321, ... },
  "limit": 5,
  "min_similarity": 0.8
}
```

### Get Action Suggestions
Asks the "Policy Guide" for the next best action based on current context.

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

## 3. Semantic Memory (Claims)

### Search Claims
Performs a semantic search over extracted atomic facts.

**Endpoint:** `POST /api/claims/search`

**Request Body:**
```json
{
  "query_text": "Who is the project manager?",
  "top_k": 3,
  "min_similarity": 0.75
}
```

---

## 4. Graph & Analytics

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
