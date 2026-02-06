# EventGraphDB Master API Specification

**Version:** 1.3.0
**Status:** Comprehensive Type Documentation
**Base URL:** `http://localhost:3000` (Default)

**Specification Source:** This spec is generated from the actual Rust source code:
- Event types: `crates/agent-db-events/src/core.rs`
- API routes: `server/src/routes.rs`
- Request/Response models: `server/src/models.rs`
- Handlers: `server/src/handlers/*.rs`

**Verified Routes:** All endpoint paths verified against `server/src/routes.rs` (no `/v1/` prefix)

---

## 1. Primitive Types Reference

All fields in the API use these core primitive types:

| Type | Description | Format | Example |
| :--- | :--- | :--- | :--- |
| `EventId` | Unique event identifier | `u128` (UUID as 128-bit integer) | `123456789012345678901234567890` |
| `Timestamp` | High-precision timestamp | `u64` (nanoseconds since Unix epoch) | `1738425600000000000` |
| `AgentId` | Agent identifier | `u64` | `1` |
| `AgentType` | Agent classification | `String` | `"movie-bot"` |
| `SessionId` | Session identifier | `u64` | `5001` |
| `GoalId` | Goal identifier | `u64` | `101` |
| `ContextHash` | Context fingerprint | `u64` (computed hash) | `12345678901234` |
| `NodeId` | Graph node identifier | `u64` | `42` |
| `MemoryId` | Memory identifier | `u64` | `789` |

---

## 2. Core Concepts

EventGraphDB is designed for **Agentic Workflows**. Unlike traditional databases, it understands the *intent* (Goals) and *context* of interactions.

| Concept | Description |
| :--- | :--- |
| **Event** | An atomic interaction (e.g., User message, AI action, Sensor reading). |
| **Episode** | A sequence of events forming a coherent task (e.g., a booking flow). |
| **Goal** | The objective of an episode (e.g., `book_movie`, `resolve_issue`). |
| **Memory** | Long-term storage of significant episodes, retrieved by context. |
| **Claim** | Atomic "facts" extracted from events (e.g., "User likes Sci-Fi"). |
| **Strategy** | Learned behavioral patterns that lead to successful goal completion. |

---

## 2. Complete Type Definitions

### 2.1 Event Structure

The `Event` object is the core data structure. All fields and their requirements:

```typescript
{
  // SERVER-GENERATED FIELDS (optional - server will generate if omitted)
  "id": EventId,                    // u128, OPTIONAL, server generates if omitted
  "timestamp": Timestamp,           // u64, OPTIONAL, server generates if omitted

  // REQUIRED FIELDS
  "agent_id": AgentId,              // u64, REQUIRED, agent that generated event
  "agent_type": AgentType,          // String, REQUIRED, e.g. "movie-bot"
  "session_id": SessionId,          // u64, REQUIRED, session identifier
  "event_type": EventType,          // EventType enum, REQUIRED, see section 2.2
  "context": EventContext,          // Object, REQUIRED, see section 2.3

  // OPTIONAL FIELDS (can be omitted or null)
  "causality_chain": EventId[],     // Array of u128, OPTIONAL, defaults to []
  "metadata": {                     // Object, OPTIONAL, defaults to {}
    "key": MetadataValue            // See section 2.6
  },
  "context_size_bytes": usize,      // usize, OPTIONAL, defaults to 0
  "segment_pointer": String | null  // String or null, OPTIONAL, format: "segment://{bucket}/{key}"
}
```

**Field Details:**

| Field | Type | Required | Can be null | Default | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `id` | `u128` | ⚙️ Auto | ❌ No | Auto-generated | Unique event identifier (UUID). Server generates if omitted. |
| `timestamp` | `u64` | ⚙️ Auto | ❌ No | Auto-generated | Nanoseconds since Unix epoch. Server generates if omitted. |
| `agent_id` | `u64` | ✅ Yes | ❌ No | N/A | Agent that generated this event |
| `agent_type` | `String` | ✅ Yes | ❌ No | N/A | Agent classification (e.g., "movie-bot") |
| `session_id` | `u64` | ✅ Yes | ❌ No | N/A | Session identifier for grouping |
| `event_type` | `EventType` | ✅ Yes | ❌ No | N/A | Type and payload (see section 2.2) |
| `causality_chain` | `EventId[]` | ❌ No | ❌ No | `[]` | Parent event IDs in causality chain |
| `context` | `EventContext` | ✅ Yes | ❌ No | N/A | Environmental context (see section 2.3) |
| `metadata` | `HashMap<String, MetadataValue>` | ❌ No | ❌ No | `{}` | Additional metadata |
| `context_size_bytes` | `usize` | ❌ No | ❌ No | `0` | Size of context in bytes |
| `segment_pointer` | `String` or `null` | ❌ No | ✅ Yes | `null` | Pointer to segment storage for large contexts |

---

### 2.2 EventType Variants

EventType is an enum with the following variants. **Exactly ONE** variant must be provided:

#### 2.2.1 Action Event

```json
{
  "Action": {
    "action_name": String,         // REQUIRED, action identifier
    "parameters": JSON,            // REQUIRED, any JSON value (can be null)
    "outcome": ActionOutcome,      // REQUIRED, see below
    "duration_ns": u64             // REQUIRED, execution duration in nanoseconds
  }
}
```

**ActionOutcome** (exactly one variant):
```json
// Success variant
{ "Success": { "result": JSON } }

// Failure variant
{ "Failure": { "error": String, "error_code": u32 } }

// Partial variant
{ "Partial": { "result": JSON, "issues": String[] } }
```

| Field | Type | Required | Can be null | Description |
| :--- | :--- | :--- | :--- | :--- |
| `action_name` | `String` | ✅ Yes | ❌ No | Name of action executed |
| `parameters` | `JSON` | ✅ Yes | ✅ Yes | Action parameters (any JSON) |
| `outcome` | `ActionOutcome` | ✅ Yes | ❌ No | Result of action |
| `duration_ns` | `u64` | ✅ Yes | ❌ No | Execution duration in nanoseconds |

#### 2.2.2 Observation Event

```json
{
  "Observation": {
    "observation_type": String,    // REQUIRED, type of observation
    "data": JSON,                  // REQUIRED, observed data (any JSON)
    "confidence": f32,             // REQUIRED, range: 0.0-1.0
    "source": String               // REQUIRED, data source identifier
  }
}
```

| Field | Type | Required | Can be null | Valid Range | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `observation_type` | `String` | ✅ Yes | ❌ No | Any string | Type of observation |
| `data` | `JSON` | ✅ Yes | ✅ Yes | Any JSON | Observed data |
| `confidence` | `f32` | ✅ Yes | ❌ No | `0.0` to `1.0` | Confidence in observation |
| `source` | `String` | ✅ Yes | ❌ No | Any string | Source identifier |

#### 2.2.3 Cognitive Event

```json
{
  "Cognitive": {
    "process_type": CognitiveType, // REQUIRED, see below
    "input": JSON,                 // REQUIRED, input to cognitive process
    "output": JSON,                // REQUIRED, output from process
    "reasoning_trace": String[]    // REQUIRED, reasoning steps (can be empty)
  }
}
```

**CognitiveType** (exactly one):
- `"GoalFormation"`
- `"Planning"`
- `"Reasoning"`
- `"MemoryRetrieval"`
- `"LearningUpdate"`

| Field | Type | Required | Can be null | Description |
| :--- | :--- | :--- | :--- | :--- |
| `process_type` | `CognitiveType` | ✅ Yes | ❌ No | Type of cognitive process |
| `input` | `JSON` | ✅ Yes | ✅ Yes | Input data |
| `output` | `JSON` | ✅ Yes | ✅ Yes | Output data |
| `reasoning_trace` | `String[]` | ✅ Yes | ❌ No | Steps taken (can be `[]`) |

#### 2.2.4 Communication Event

```json
{
  "Communication": {
    "message_type": String,        // REQUIRED, e.g., "user_message"
    "sender": AgentId,             // REQUIRED, sender agent ID
    "recipient": AgentId,          // REQUIRED, recipient agent ID
    "content": JSON                // REQUIRED, message content
  }
}
```

| Field | Type | Required | Can be null | Description |
| :--- | :--- | :--- | :--- | :--- |
| `message_type` | `String` | ✅ Yes | ❌ No | Type of message |
| `sender` | `u64` | ✅ Yes | ❌ No | Sender agent ID |
| `recipient` | `u64` | ✅ Yes | ❌ No | Recipient agent ID |
| `content` | `JSON` | ✅ Yes | ✅ Yes | Message content |

#### 2.2.5 Learning Event

```json
{
  "Learning": {
    "event": LearningEvent         // REQUIRED, see below
  }
}
```

**LearningEvent** (exactly one variant):

```json
// Memory retrieved
{ "MemoryRetrieved": { "query_id": String, "memory_ids": u64[] } }

// Memory used
{ "MemoryUsed": { "query_id": String, "memory_id": u64 } }

// Strategy served
{ "StrategyServed": { "query_id": String, "strategy_ids": u64[] } }

// Strategy used
{ "StrategyUsed": { "query_id": String, "strategy_id": u64 } }

// Outcome
{ "Outcome": { "query_id": String, "success": bool } }
```

#### 2.2.6 Context Event

```json
{
  "Context": {
    "text": String,                // REQUIRED, raw text content
    "context_type": String,        // REQUIRED, e.g., "conversation"
    "language": String | null      // OPTIONAL, e.g., "en", "es"
  }
}
```

| Field | Type | Required | Can be null | Description |
| :--- | :--- | :--- | :--- | :--- |
| `text` | `String` | ✅ Yes | ❌ No | Raw text content |
| `context_type` | `String` | ✅ Yes | ❌ No | Type: "conversation", "document", "transcript" |
| `language` | `String` or `null` | ❌ No | ✅ Yes | Language hint (e.g., "en", "es") |

---

### 2.3 EventContext Structure

Complete structure of the context object:

```typescript
{
  "environment": EnvironmentState,   // REQUIRED, see section 2.3.1
  "active_goals": Goal[],            // REQUIRED, active goals (can be [])
  "resources": ResourceState,        // REQUIRED, see section 2.3.2
  "fingerprint": ContextHash,        // OPTIONAL, defaults to 0 (server auto-computes if 0)
  "embeddings": f32[] | null         // OPTIONAL, vector embeddings
}
```

| Field | Type | Required | Can be null | Default | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `environment` | `EnvironmentState` | ✅ Yes | ❌ No | N/A | Environment state |
| `active_goals` | `Goal[]` | ✅ Yes | ❌ No | N/A | Active goals (can be `[]`) |
| `resources` | `ResourceState` | ✅ Yes | ❌ No | N/A | Resource availability |
| `fingerprint` | `u64` | ⚙️ Auto | ❌ No | `0` | Server auto-computes if set to `0` or omitted |
| `embeddings` | `f32[]` or `null` | ❌ No | ✅ Yes | `null` | Context embeddings |

#### 2.3.1 EnvironmentState

```typescript
{
  "variables": {                     // REQUIRED, key-value pairs
    "key": JSON                      // Any JSON value
  },
  "spatial": SpatialContext | null,  // OPTIONAL, spatial information
  "temporal": TemporalContext        // REQUIRED, temporal information
}
```

**SpatialContext** (when provided):
```json
{
  "location": [f64, f64, f64],       // REQUIRED, [x, y, z] coordinates
  "bounds": BoundingBox | null,      // OPTIONAL
  "reference_frame": String          // REQUIRED, e.g., "world"
}
```

**BoundingBox** (when provided):
```json
{
  "min": [f64, f64, f64],            // REQUIRED, minimum bounds
  "max": [f64, f64, f64]             // REQUIRED, maximum bounds
}
```

**TemporalContext**:
```json
{
  "time_of_day": TimeOfDay | null,   // OPTIONAL
  "deadlines": Deadline[],           // REQUIRED (can be [])
  "patterns": TemporalPattern[]      // REQUIRED (can be [])
}
```

**TimeOfDay** (when provided):
```json
{
  "hour": u8,                        // REQUIRED, range: 0-23
  "minute": u8,                      // REQUIRED, range: 0-59
  "timezone": String                 // REQUIRED, e.g., "UTC"
}
```

**Deadline**:
```json
{
  "goal_id": GoalId,                 // REQUIRED, u64
  "timestamp": Timestamp,            // REQUIRED, u64
  "priority": f32                    // REQUIRED, range: 0.0-1.0
}
```

**TemporalPattern**:
```json
{
  "pattern_name": String,            // REQUIRED
  "frequency": Duration,             // REQUIRED, nanoseconds
  "phase": f32                       // REQUIRED
}
```

#### 2.3.2 Goal Structure

```json
{
  "id": GoalId,                      // REQUIRED, u64
  "description": String,             // REQUIRED
  "priority": f32,                   // REQUIRED, range: 0.0-1.0
  "deadline": Timestamp | null,      // OPTIONAL
  "progress": f32,                   // REQUIRED, range: 0.0-1.0
  "subgoals": GoalId[]               // REQUIRED (can be [])
}
```

| Field | Type | Required | Can be null | Valid Range | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `id` | `u64` | ✅ Yes | ❌ No | Any u64 | Goal identifier |
| `description` | `String` | ✅ Yes | ❌ No | Any string | Goal description |
| `priority` | `f32` | ✅ Yes | ❌ No | `0.0` to `1.0` | Priority level |
| `deadline` | `u64` or `null` | ❌ No | ✅ Yes | Timestamp | Deadline (optional) |
| `progress` | `f32` | ✅ Yes | ❌ No | `0.0` to `1.0` | Completion progress |
| `subgoals` | `u64[]` | ✅ Yes | ❌ No | Any u64[] | Subgoal IDs (can be `[]`) |

#### 2.3.3 ResourceState

```json
{
  "computational": ComputationalResources,  // REQUIRED
  "external": {                             // REQUIRED (can be {})
    "resource_name": ResourceAvailability
  }
}
```

**ComputationalResources**:
```json
{
  "cpu_percent": f32,                // REQUIRED, range: 0.0-100.0
  "memory_bytes": u64,               // REQUIRED
  "storage_bytes": u64,              // REQUIRED
  "network_bandwidth": u64           // REQUIRED, bytes per second
}
```

| Field | Type | Required | Can be null | Valid Range | Unit |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `cpu_percent` | `f32` | ✅ Yes | ❌ No | `0.0` to `100.0` | Percentage |
| `memory_bytes` | `u64` | ✅ Yes | ❌ No | Any u64 | Bytes |
| `storage_bytes` | `u64` | ✅ Yes | ❌ No | Any u64 | Bytes |
| `network_bandwidth` | `u64` | ✅ Yes | ❌ No | Any u64 | Bytes/second |

**ResourceAvailability**:
```json
{
  "available": bool,                 // REQUIRED
  "capacity": f32,                   // REQUIRED, range: 0.0-1.0
  "current_usage": f32,              // REQUIRED, range: 0.0-1.0
  "estimated_cost": f32 | null       // OPTIONAL
}
```

---

### 2.4 MetadataValue

Metadata can be one of these types:

```json
{ "String": "text value" }
{ "Integer": 123 }
{ "Float": 45.67 }
{ "Boolean": true }
{ "Json": <any JSON value> }
```

---

## 3. Event Endpoints

### `POST /api/events`
Ingests a new event and triggers automatic graph construction, episode detection, and claim extraction.

**Request Body (`ProcessEventRequest`):**

| Field | Type | Required | Can be null | Default | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `event` | `Event` | ✅ Yes | ❌ No | N/A | Complete event object (see section 2.1) |
| `enable_semantic` | `bool` | ❌ No | ❌ No | `false` | Enable semantic processing (NER, claims, embeddings) |

**Example Request (Minimal - Server generates ID/timestamp):**
```json
{
  "event": {
    "agent_id": 1,
    "agent_type": "movie-bot",
    "session_id": 5001,
    "event_type": {
      "Communication": {
        "message_type": "user_message",
        "sender": 0,
        "recipient": 1,
        "content": { "text": "I want to book Interstellar for tonight." }
      }
    },
    "context": {
      "active_goals": [
        {
          "id": 101,
          "description": "book_movie",
          "priority": 0.9,
          "progress": 0.1,
          "subgoals": []
        }
      ],
      "environment": {
        "variables": { "user_id": "user_99" },
        "temporal": {
          "deadlines": [],
          "patterns": []
        }
      },
      "resources": {
        "computational": {
          "cpu_percent": 10.0,
          "memory_bytes": 1024,
          "storage_bytes": 1024,
          "network_bandwidth": 100
        },
        "external": {}
      }
    },
    "metadata": { "user_id": { "String": "user_99" } }
  },
  "enable_semantic": true
}
```

**Example Request (Full - Client provides all fields):**
```json
{
  "event": {
    "id": 123456789012345678901234567890,
    "timestamp": 1738425600000000000,
    "agent_id": 1,
    "agent_type": "movie-bot",
    "session_id": 5001,
    "event_type": {
      "Communication": {
        "message_type": "user_message",
        "sender": 0,
        "recipient": 1,
        "content": { "text": "I want to book Interstellar for tonight." }
      }
    },
    "causality_chain": [],
    "context": {
      "active_goals": [
        {
          "id": 101,
          "description": "book_movie",
          "priority": 0.9,
          "progress": 0.1,
          "deadline": null,
          "subgoals": []
        }
      ],
      "environment": {
        "variables": { "user_id": "user_99" },
        "spatial": null,
        "temporal": {
          "time_of_day": null,
          "deadlines": [],
          "patterns": []
        }
      },
      "resources": {
        "computational": {
          "cpu_percent": 10.0,
          "memory_bytes": 1024,
          "storage_bytes": 1024,
          "network_bandwidth": 100
        },
        "external": {}
      },
      "fingerprint": 0,
      "embeddings": null
    },
    "metadata": { "user_id": { "String": "user_99" } },
    "context_size_bytes": 0,
    "segment_pointer": null
  },
  "enable_semantic": true
}
```

**Response (`ProcessEventResponse`):**

| Field | Type | Description |
| :--- | :--- | :--- |
| `success` | `bool` | Whether processing succeeded |
| `nodes_created` | `usize` | Number of graph nodes created |
| `patterns_detected` | `usize` | Number of patterns detected |
| `processing_time_ms` | `u64` | Processing time in milliseconds |

**Example Response:**
```json
{
  "success": true,
  "nodes_created": 5,
  "patterns_detected": 2,
  "processing_time_ms": 15
}
```

### `GET /api/events`
Retrieves recent events.

**Query Parameters:**

| Parameter | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `limit` | `usize` | ❌ No | `10` | Maximum number of events to return |

**Response:** Array of `Event` objects (see section 2.1)

---

### `GET /api/episodes`
Retrieves completed episodes detected by the system.

**Query Parameters:**

| Parameter | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `limit` | `usize` | ❌ No | `10` | Maximum number of episodes to return |

**Response (`EpisodeResponse[]`):**

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | `u64` | Episode identifier |
| `agent_id` | `u64` | Agent that executed the episode |
| `event_count` | `usize` | Number of events in episode |
| `significance` | `f32` | Significance score (0.0-1.0) |
| `outcome` | `String` or `null` | Episode outcome (e.g., "Success", "Failure") |

---

## 4. Memory Endpoints

### `GET /api/memories/agent/:agent_id`
Retrieves long-term memories for a specific agent.

**Path Parameters:**

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `agent_id` | `u64` | ✅ Yes | Agent identifier |

**Query Parameters:**

| Parameter | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `limit` | `usize` | ❌ No | `10` | Maximum number of memories to return |

**Response (`MemoryResponse[]`):**

| Field | Type | Can be null | Description |
| :--- | :--- | :--- | :--- |
| `id` | `u64` | ❌ No | Memory identifier |
| `agent_id` | `u64` | ❌ No | Agent that formed the memory |
| `session_id` | `u64` | ❌ No | Session identifier |
| `strength` | `f32` | ❌ No | Memory strength (0.0-1.0) |
| `relevance_score` | `f32` | ❌ No | Relevance score (0.0-1.0) |
| `access_count` | `u32` | ❌ No | Number of times accessed |
| `formed_at` | `u64` | ❌ No | Formation timestamp |
| `last_accessed` | `u64` | ❌ No | Last access timestamp |
| `context_hash` | `u64` | ❌ No | Context fingerprint |
| `context` | `EventContext` | ❌ No | Full context object (see section 2.3) |
| `outcome` | `String` | ❌ No | Episode outcome description |
| `memory_type` | `String` | ❌ No | Type: "Episodic", "Working", "Semantic", "Negative" |

---

### `POST /api/memories/context`
Finds history relevant to the *current task* (Goal + Environment).

**Request Body (`ContextMemoriesRequest`):**

| Field | Type | Required | Can be null | Default | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `context` | `EventContext` | ✅ Yes | ❌ No | N/A | Context to match (see section 2.3) |
| `limit` | `usize` | ❌ No | ❌ No | `10` | Maximum results |
| `min_similarity` | `f32` or `null` | ❌ No | ✅ Yes | `0.6` | Minimum similarity threshold (0.0-1.0) |
| `agent_id` | `u64` or `null` | ❌ No | ✅ Yes | `null` | Filter by agent |
| `session_id` | `u64` or `null` | ❌ No | ✅ Yes | `null` | Filter by session |

**Example Request:**
```json
{
  "context": {
    "active_goals": [
      {
        "id": 101,
        "description": "book_movie",
        "priority": 0.9,
        "progress": 0.5,
        "deadline": null,
        "subgoals": []
      }
    ],
    "environment": {
      "variables": { "user_id": "user_99" },
      "spatial": null,
      "temporal": {
        "time_of_day": null,
        "deadlines": [],
        "patterns": []
      }
    },
    "resources": {
      "computational": {
        "cpu_percent": 50.0,
        "memory_bytes": 1024000,
        "storage_bytes": 1024000000,
        "network_bandwidth": 1000
      },
      "external": {}
    },
    "fingerprint": 0,
    "embeddings": null
  },
  "limit": 5,
  "min_similarity": 0.8,
  "agent_id": null,
  "session_id": null
}
```

**Response:** Array of `MemoryResponse` objects (same as above)

---

## 5. Strategy & Policy Endpoints

### `GET /api/strategies/agent/:agent_id`
Retrieves learned strategies for a specific agent.

**Path Parameters:**

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `agent_id` | `u64` | ✅ Yes | Agent identifier |

**Query Parameters:**

| Parameter | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `limit` | `usize` | ❌ No | `10` | Maximum number of strategies to return |

**Response (`StrategyResponse[]`):**

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | `u64` | Strategy identifier |
| `name` | `String` | Strategy name |
| `agent_id` | `u64` | Agent that learned this strategy |
| `quality_score` | `f32` | Quality score (0.0-1.0) |
| `success_count` | `u32` | Number of successes |
| `failure_count` | `u32` | Number of failures |
| `reasoning_steps` | `ReasoningStepResponse[]` | Steps in strategy |
| `strategy_type` | `String` | Strategy classification |
| `support_count` | `u32` | Number of supporting episodes |
| `expected_success` | `f32` | Expected success rate (0.0-1.0) |
| `expected_cost` | `f32` | Expected cost |
| `expected_value` | `f32` | Expected value |
| `confidence` | `f32` | Confidence in strategy (0.0-1.0) |
| `goal_bucket_id` | `u64` | Associated goal bucket |
| `behavior_signature` | `String` | Unique behavior signature |
| `precondition` | `String` | When to apply strategy |
| `action_hint` | `String` | Suggested action |

**ReasoningStepResponse:**
| Field | Type | Description |
| :--- | :--- | :--- |
| `description` | `String` | Step description |
| `sequence_order` | `usize` | Order in sequence |

---

### `POST /api/strategies/similar`
Finds strategies similar to a given set of goals or tools.

**Request Body (`StrategySimilarityRequest`):**

| Field | Type | Required | Can be null | Default | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `goal_ids` | `u64[]` | ❌ No | ❌ No | `[]` | Goal IDs to match |
| `tool_names` | `String[]` | ❌ No | ❌ No | `[]` | Tool names to match |
| `result_types` | `String[]` | ❌ No | ❌ No | `[]` | Result types to match |
| `context_hash` | `u64` or `null` | ❌ No | ✅ Yes | `null` | Context fingerprint |
| `agent_id` | `u64` or `null` | ❌ No | ✅ Yes | `null` | Filter by agent |
| `limit` | `usize` | ❌ No | ❌ No | `10` | Maximum results |
| `min_score` | `f32` or `null` | ❌ No | ✅ Yes | `null` | Minimum similarity score |

**Example Request:**
```json
{
  "goal_ids": [101],
  "tool_names": ["search_api"],
  "result_types": [],
  "context_hash": null,
  "agent_id": null,
  "limit": 5,
  "min_score": 0.7
}
```

**Response (`SimilarStrategyResponse[]`):**

Same as `StrategyResponse` with additional field:

| Field | Type | Description |
| :--- | :--- | :--- |
| `score` | `f32` | Similarity score (0.0-1.0) |
| ... | ... | (plus all fields from `StrategyResponse`) |

---

### `GET /api/suggestions`
The **Policy Guide** endpoint. Asks: "What should the AI do next?"

**Query Parameters:**

| Parameter | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `context_hash` | `u64` | ✅ Yes | N/A | Context fingerprint |
| `last_action_node` | `u64` or `null` | ❌ No | `null` | Last action node |
| `limit` | `usize` | ❌ No | `10` | Maximum suggestions |

**Response (`ActionSuggestionResponse[]`):**

| Field | Type | Description |
| :--- | :--- | :--- |
| `action_name` | `String` | Suggested action name |
| `success_probability` | `f32` | Probability of success (0.0-1.0) |
| `evidence_count` | `u32` | Number of supporting examples |
| `reasoning` | `String` | Explanation for suggestion |

---

## 6. Semantic Memory (Claims)

### `GET /api/claims`
List all extracted facts (Claims).

**Query Parameters:**

| Parameter | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `limit` | `usize` | ❌ No | `10` | Maximum number of claims to return |
| `event_id` | `u128` or `null` | ❌ No | `null` | Filter by source event ID |

**Response (`ClaimResponse[]`):**

| Field | Type | Can be null | Description |
| :--- | :--- | :--- | :--- |
| `claim_id` | `u64` | ❌ No | Claim identifier |
| `claim_text` | `String` | ❌ No | The extracted claim/fact |
| `confidence` | `f32` | ❌ No | Extraction confidence (0.0-1.0) |
| `source_event_id` | `u128` | ❌ No | Source event ID |
| `similarity` | `f32` or `null` | ✅ Yes | Similarity score (only in search results) |
| `evidence_spans` | `EvidenceSpanResponse[]` | ❌ No | Supporting evidence |
| `support_count` | `u32` | ❌ No | Number of supporting instances |
| `status` | `String` | ❌ No | Status: "Active", "Contradicted", "Deprecated" |
| `created_at` | `u64` | ❌ No | Creation timestamp |
| `last_accessed` | `u64` | ❌ No | Last access timestamp |

**EvidenceSpanResponse:**

| Field | Type | Description |
| :--- | :--- | :--- |
| `start_offset` | `usize` | Start position in source text |
| `end_offset` | `usize` | End position in source text |
| `text_snippet` | `String` | Extracted text snippet |

---

### `GET /api/claims/:id`
Retrieve a specific claim by its ID.

**Path Parameters:**

| Parameter | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| `id` | `u64` | ✅ Yes | Claim identifier |

**Response:** Single `ClaimResponse` object (see above)

---

### `POST /api/claims/search`
Semantic search over claims. Use this for "Soft Facts" like preferences.

**Request Body (`ClaimSearchRequest`):**

| Field | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `query_text` | `String` | ✅ Yes | N/A | Natural language query |
| `top_k` | `usize` | ❌ No | `10` | Number of results to return |
| `min_similarity` | `f32` | ❌ No | `0.7` | Minimum similarity threshold (0.0-1.0) |

**Example Request:**
```json
{
  "query_text": "What are the user's favorite genres?",
  "top_k": 3,
  "min_similarity": 0.7
}
```

**Response:** Array of `ClaimResponse` objects with `similarity` field populated

---

### `POST /api/embeddings/process`
Manually trigger embedding generation for pending claims.

**Query Parameters:**

| Parameter | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `limit` | `usize` | ❌ No | `10` | Batch size for processing |

**Response (`EmbeddingProcessResponse`):**

| Field | Type | Description |
| :--- | :--- | :--- |
| `claims_processed` | `usize` | Number of claims processed |
| `success` | `bool` | Whether processing succeeded |

---

## 7. Graph & Analytics

### `GET /api/graph`
Retrieve graph structure (nodes and edges).

**Query Parameters:**

| Parameter | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `limit` | `usize` | ❌ No | `10` | Maximum number of nodes/edges |
| `session_id` | `u64` or `null` | ❌ No | `null` | Filter by session |
| `agent_type` | `String` or `null` | ❌ No | `null` | Filter by agent type |

**Response (`GraphResponse`):**

| Field | Type | Description |
| :--- | :--- | :--- |
| `nodes` | `GraphNodeResponse[]` | Array of graph nodes |
| `edges` | `GraphEdgeResponse[]` | Array of graph edges |

**GraphNodeResponse:**

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | `u64` | Node identifier |
| `label` | `String` | Node label |
| `node_type` | `String` | Type: "Event", "Goal", "Action", "Context", etc. |
| `created_at` | `u64` | Creation timestamp |
| `properties` | `JSON` | Node properties (any JSON object) |

**GraphEdgeResponse:**

| Field | Type | Description |
| :--- | :--- | :--- |
| `id` | `u64` | Edge identifier |
| `from` | `u64` | Source node ID |
| `to` | `u64` | Target node ID |
| `edge_type` | `String` | Type: "CausedBy", "PartOf", "LeadsTo", etc. |
| `weight` | `f32` | Edge weight (0.0-1.0) |
| `confidence` | `f32` | Confidence in edge (0.0-1.0) |

---

### `GET /api/graph/context`
Retrieve a subgraph centered around a specific context.

**Query Parameters:**

| Parameter | Type | Required | Default | Description |
| :--- | :--- | :--- | :--- | :--- |
| `context_hash` | `u64` | ✅ Yes | N/A | Context fingerprint |
| `limit` | `usize` | ❌ No | `10` | Maximum nodes/edges |
| `session_id` | `u64` or `null` | ❌ No | `null` | Filter by session |
| `agent_type` | `String` or `null` | ❌ No | `null` | Filter by agent type |

**Response:** `GraphResponse` (same as above)

---

### `GET /api/stats`
High-level system statistics (total events, nodes, etc.).

**Response (`StatsResponse`):**

| Field | Type | Description |
| :--- | :--- | :--- |
| `total_events_processed` | `u64` | Total events ingested |
| `total_nodes_created` | `u64` | Total graph nodes |
| `total_episodes_detected` | `u64` | Total episodes completed |
| `total_memories_formed` | `u64` | Total memories stored |
| `total_strategies_extracted` | `u64` | Total strategies learned |
| `total_reinforcements_applied` | `u64` | Total reinforcement updates |
| `average_processing_time_ms` | `f64` | Average event processing time |

---

### `GET /api/analytics`
Advanced graph analytics (connected components, modularity).

**Response (`AnalyticsResponse`):**

| Field | Type | Description |
| :--- | :--- | :--- |
| `node_count` | `usize` | Total nodes in graph |
| `edge_count` | `usize` | Total edges in graph |
| `connected_components` | `usize` | Number of connected components |
| `largest_component_size` | `usize` | Size of largest component |
| `average_path_length` | `f32` | Average shortest path length |
| `diameter` | `u32` | Graph diameter |
| `clustering_coefficient` | `f32` | Global clustering coefficient |
| `average_clustering` | `f32` | Average local clustering |
| `modularity` | `f32` | Modularity score |
| `community_count` | `usize` | Number of communities detected |
| `learning_metrics` | `LearningMetricsResponse` | Learning-specific metrics |

**LearningMetricsResponse:**

| Field | Type | Description |
| :--- | :--- | :--- |
| `total_events` | `usize` | Total events processed |
| `unique_contexts` | `usize` | Number of unique contexts |
| `learned_patterns` | `usize` | Number of learned patterns |
| `strong_memories` | `usize` | Number of strong memories |
| `overall_success_rate` | `f32` | Overall success rate (0.0-1.0) |
| `average_edge_weight` | `f32` | Average edge weight |

---

### `GET /api/indexes`
Statistics on property indexes (hits, misses).

**Response:** Map of index name to `IndexStatsResponse`

**IndexStatsResponse:**

| Field | Type | Description |
| :--- | :--- | :--- |
| `insert_count` | `u64` | Number of insertions |
| `query_count` | `u64` | Number of queries |
| `range_query_count` | `u64` | Number of range queries |
| `hit_count` | `u64` | Number of cache hits |
| `miss_count` | `u64` | Number of cache misses |
| `last_accessed` | `u64` | Last access timestamp |

---

### `GET /api/communities`
Returns detected communities (Louvain algorithm results).

**Response (`CommunitiesResponse`):**

| Field | Type | Description |
| :--- | :--- | :--- |
| `communities` | `CommunityResponse[]` | Array of detected communities |
| `modularity` | `f32` | Modularity score |
| `iterations` | `usize` | Number of algorithm iterations |
| `community_count` | `usize` | Total number of communities |

**CommunityResponse:**

| Field | Type | Description |
| :--- | :--- | :--- |
| `community_id` | `u64` | Community identifier |
| `node_ids` | `u64[]` | Node IDs in community |
| `size` | `usize` | Number of nodes |

---

### `GET /api/centrality`
Returns node centrality scores (importance ranking).

**Response:** Array of `CentralityScoresResponse`

**CentralityScoresResponse:**

| Field | Type | Description |
| :--- | :--- | :--- |
| `node_id` | `u64` | Node identifier |
| `degree` | `f32` | Degree centrality |
| `betweenness` | `f32` | Betweenness centrality |
| `closeness` | `f32` | Closeness centrality |
| `eigenvector` | `f32` | Eigenvector centrality |
| `pagerank` | `f32` | PageRank score |
| `combined` | `f32` | Combined centrality score |

---

## 8. System Endpoints

### `GET /api/health`
Check server status and engine health metrics.

**Response (`HealthResponse`):**

| Field | Type | Description |
| :--- | :--- | :--- |
| `status` | `String` | Status: "healthy", "degraded", "unhealthy" |
| `version` | `String` | Server version |
| `uptime_seconds` | `u64` | Server uptime in seconds |
| `is_healthy` | `bool` | Overall health status |
| `node_count` | `usize` | Current graph node count |
| `edge_count` | `usize` | Current graph edge count |
| `processing_rate` | `f64` | Events processed per second |

**Example Response:**
```json
{
  "status": "healthy",
  "version": "1.3.0",
  "uptime_seconds": 3600,
  "is_healthy": true,
  "node_count": 15420,
  "edge_count": 32801,
  "processing_rate": 125.5
}
```

---

### `GET /`
Root endpoint with version and quick-start info.

**Response:** JSON object with server information and quick-start guide

---

### `GET /docs`
Redirects to API documentation.

**Response:** HTTP 302 redirect to documentation URL

---

## 9. Event Types Reference (Deep Dive)

Events are the lifeblood of EventGraphDB. Every event should ideally be wrapped in a **Goal** to allow the system to group them into **Episodes**.

### A. Action Event
**When to use:** Whenever the AI *does* something (calls an API, books a seat, sends an email).
```json
{
  "Action": {
    "action_name": "book_movie_ticket",
    "parameters": { "movie": "Interstellar", "seat": "H12" },
    "outcome": { "Success": { "result": { "confirmation_id": "CONF-123" } } },
    "duration_ns": 1500000
  }
}
```

### B. Cognitive Event
**When to use:** To log the AI's internal "thinking" process. This is crucial for **Strategy Extraction**.
```json
{
  "Cognitive": {
    "process_type": "Reasoning",
    "input": { "user_query": "I want a quiet seat" },
    "output": { "decision": "Select back row" },
    "reasoning_trace": [
      "User requested quiet environment",
      "Back rows have less foot traffic",
      "Checking availability for row Z"
    ]
  }
}
```

### C. Communication Event
**When to use:** For every message sent between the User and the AI.
```json
{
  "Communication": {
    "message_type": "user_message",
    "sender": 0,
    "recipient": 1,
    "content": { "text": "I'd like to book a ticket for tonight." }
  }
}
```

### D. Observation Event
**When to use:** When the AI receives external data that isn't a direct message (e.g., price updates, weather, sensor data).
```json
{
  "Observation": {
    "observation_type": "ticket_availability",
    "data": { "remaining_seats": 5 },
    "confidence": 1.0,
    "source": "Theater-API"
  }
}
```

---

## 10. Important Notes on Field Requirements

### Server-Generated Fields

The server automatically generates certain fields if not provided:

| Field | Auto-Generated | Description |
| :--- | :--- | :--- |
| `event.id` | ✅ Yes | Server generates UUID if omitted |
| `event.timestamp` | ✅ Yes | Server generates current timestamp if omitted |
| `context.fingerprint` | ✅ Yes | Server computes hash if set to `0` or omitted |

**Best Practice:** Omit `id`, `timestamp`, and `fingerprint` fields and let the server generate them. This ensures consistency and reduces client complexity.

---

### Null vs Omitted Fields

Understanding the difference between `null` and omitted fields:

| Scenario | Example | Description |
| :--- | :--- | :--- |
| **Required, Cannot be null** | `"agent_id": 1` | Field MUST be present with a valid value |
| **Server-Generated** | Omit `"id"` → server generates | Field can be omitted, server auto-generates |
| **Optional, Defaults** | Omit `"limit"` → defaults to `10` | Field can be omitted, uses default |
| **Optional, Can be null** | `"deadline": null` | Field can be omitted OR explicitly set to `null` |
| **Optional, Cannot be null** | `"causality_chain": []` | Field can be omitted (defaults to `[]`) but if present, cannot be `null` |

### Field Value Constraints

**Numeric Ranges:**
- `priority`, `progress`, `confidence`: Must be between `0.0` and `1.0`
- `cpu_percent`: Must be between `0.0` and `100.0`
- `hour`: Must be between `0` and `23`
- `minute`: Must be between `0` and `59`

**ID Fields:**
- All ID fields (`EventId`, `AgentId`, `SessionId`, etc.) must be non-negative integers
- `EventId` is a 128-bit integer (UUID)
- All other IDs are 64-bit integers (`u64`)

**Timestamp Fields:**
- All timestamps are `u64` representing nanoseconds since Unix epoch
- Example: `1738425600000000000` = February 1, 2026, 12:00:00 AM UTC
- JavaScript: `Date.now() * 1000000` (multiply milliseconds by 1,000,000)

**String Fields:**
- Cannot be empty strings unless explicitly noted
- `agent_type`, `action_name`, `description`, etc. must be non-empty

**Array Fields:**
- Can be empty arrays `[]` unless explicitly required to have elements
- `active_goals: []` is valid (no active goals)
- `causality_chain: []` is valid (no parent events)

---

## 11. The Role of Goals in Events

Every event listed above should be sent inside an `Event` object that includes a `context` with `active_goals`. This is how the system knows that a `Communication` event and an `Action` event belong to the same "Booking" task.

**Example: Linking a message to a goal**
```json
{
  "event": {
    "event_type": { "Communication": { ... } },
    "context": {
      "active_goals": [
        {
          "id": 101,
          "description": "book_movie",
          "priority": 0.9,
          "progress": 0.5 
        }
      ]
    }
  }
}
```
*   **Goal ID**: Keeps the task unique.
*   **Progress**: Tell the system how close you are to finishing (0.0 to 1.0). When progress hits 1.0, the **Episode** is marked as complete.

---

## 12. Querying Deep Dive: Which Search to Use?

EventGraphDB provides five distinct ways to find data. Choosing the right one is critical for AI performance.

### A. Direct Node Search (The "Hard Fact" Search)
**Endpoint:** `GET /api/graph` or `GET /api/graph/context`
**What it is:** A deterministic lookup for specific nodes and their properties via graph traversal.
**When to use:** For retrieving specific nodes by session, agent type, or context hash.
**Example (Find nodes by session):**
```
GET /api/graph?session_id=5001&limit=10
```
*   **Advice:** Use this for retrieving graph structures. For metadata filtering, use the event metadata system.

### B. Semantic Search (The "Concept" Search)
**Endpoint:** `POST /api/claims/search`  
**What it is:** A vector-based search over "Claims" (facts extracted from text).  
**When to use:** For fuzzy concepts like "User preferences," "Past complaints," or "Vibe."  
**Example (Find Movie Preferences):**
```json
{
  "query_text": "What kind of seating does the user prefer?",
  "top_k": 3,
  "min_similarity": 0.7
}
```
*   **Advice:** Keep the `query_text` natural. Don't include IDs in the string; filter the results in your app code instead.

### C. Context/Fingerprint Search (The "Task" Search)
**Endpoint:** `POST /api/memories/context`  
**What it is:** Finds previous **Episodes** that match a specific environmental state.  
**When to use:** To answer "What did we do last time we were in this exact situation?"  
**Example (Find history for a specific Goal):**
```json
{
  "context": {
    "active_goals": [{ "id": 101, "description": "book_movie" }],
    "environment": { "variables": { "user_id": "user_99" } }
  }
}
```
*   **Advice:** The system uses a `fingerprint` (a hash of the context) to find near-identical situations instantly. Use this to maintain "state" across sessions.

### D. Graph Traversal (The "Relationship" Search)
**Endpoint:** `GET /api/graph` with filters
**What it is:** Retrieves graph structure showing nodes and their connections (edges).
**When to use:** To see relationships between events, goals, actions, and contexts.
**Example (Get graph for a specific session):**
```http
GET /api/graph?session_id=5001&limit=100
```
*   **Advice:** The response includes both nodes and edges, showing how events are connected. Use this to understand event causality and episode formation.

### E. Embedding Search (The "Similarity" Search)
**Endpoint:** `POST /api/strategies/similar`  
**What it is:** Finds learned behaviors (Strategies) that "look like" the current goal.  
**When to use:** When the AI is stuck and needs to find a successful "recipe" from a similar task.  
**Example (Find a strategy for a new goal):**
```json
{
  "goal_ids": [202],
  "tool_names": ["payment_gateway"],
  "min_score": 0.8
}
```
*   **Advice:** This is how you share "wisdom" between different agents. If one agent learned how to handle a payment error, another can find that strategy here.

---

## 13. Query Selection Matrix

| If you want to find... | Use this Search | Accuracy |
| :--- | :--- | :--- |
| **Member # / Email** | **Direct Node Search** | 100% |
| **"Does he like popcorn?"** | **Semantic Search** | Fuzzy |
| **"Where did we leave off?"** | **Context Search** | High |
| **"What's his booking history?"** | **Graph Traversal** | 100% |
| **"How do I solve this error?"** | **Embedding Search** | Fuzzy |

---

## 14. Integration Examples (JavaScript/TypeScript)

### A. Basic Client Setup
```typescript
import axios from 'axios';

const client = axios.create({
  baseURL: 'http://localhost:3000/api',
  headers: { 'Content-Type': 'application/json' }
});
```

### B. Movie Booking Workflow
This example shows how to handle a returning user requesting a booking based on their history.

```javascript
async function handleMovieBooking(userId, userMessage) {
  // 1. Fetch Hard Facts (Member Number) from Graph
  const graphRes = await client.post('/v1/graph/query', {
    node_types: ["User"],
    property_filters: [{ key: "user_id", value: userId, operator: "equals" }]
  });
  const memberId = graphRes.data.results[0]?.properties?.member_number || "Guest";

  // 2. Fetch Soft Facts (Preferences) from Semantic Memory
  const claimsRes = await client.post('/claims/search', {
    query_text: "What are the user's movie and seat preferences?",
    top_k: 3,
    min_similarity: 0.7
  });
  const preferences = claimsRes.data.map(c => c.claim_text).join(", ");

  // 3. Fetch Process History (Last Booking State)
  const contextRes = await client.post('/memories/context', {
    context: {
      active_goals: [{ id: 101, description: "book_movie" }],
      environment: { variables: { user_id: userId } }
    },
    limit: 1
  });
  const lastSession = contextRes.data[0]?.outcome || "No previous history";

  // 4. Generate AI Response (Conceptual)
  console.log(`AI: Welcome back ${memberId}!`);
  console.log(`AI: I remember you like: ${preferences}`);
  console.log(`AI: Last time we were at: ${lastSession}`);

  // 5. Log the new interaction
  await client.post('/events', {
    event: {
      id: Date.now(),
      timestamp: Date.now() * 1000000,
      agent_id: 1,
      agent_type: "movie-bot",
      session_id: 5001,
      event_type: {
        Communication: {
          message_type: "user_message",
          sender: 0,
          recipient: 1,
          content: { text: userMessage }
        }
      },
      context: {
        active_goals: [{ id: 101, description: "book_movie" }],
        environment: { variables: { user_id: userId } }
      },
      metadata: { user_id: { String: userId } }
    },
    enable_semantic: true
  });
}
```

### C. Polling for Suggestions (Policy Guide)
```javascript
async function getNextAction(contextHash) {
  const res = await client.get(`/suggestions?context_hash=${contextHash}&limit=1`);
  if (res.data.length > 0) {
    const suggestion = res.data[0];
    console.log(`Recommended Action: ${suggestion.action_name} (${suggestion.success_probability * 100}% success)`);
    console.log(`Reasoning: ${suggestion.reasoning}`);
  }
}
```

---

## 15. System Limits & Profiles

| Feature | **FREE Profile** | **NORMAL Profile** |
| :--- | :--- | :--- |
| **Max Graph Size** | 50,000 nodes | 1,000,000 nodes |
| **Memory Cache** | 1,000 items | 10,000 items |
| **Strategy Cache** | 500 items | 5,000 items |
| **Redb Cache** | 64 MB | 256 MB |
| **Louvain (Analytics)** | Disabled | Enabled |

### Query Limits
*   **Default Limit**: 10 items.
*   **Max Limit**: 1,000 items.
*   **Semantic Extraction**: Max 10 claims per event.
*   **Claim Confidence**: Min 0.7 score required for storage.

### Graph Query vs. Semantic Search
*   **Use Graph (Direct Lookup)**: For "Hard Facts" (Member Numbers, Emails, IDs).
*   **Use Semantic (Claims Search)**: For "Soft Facts" (Preferences, Mood, Intent).

### Goal-Based Retrieval
Always query memories by **Context/Goal**. This ensures your AI doesn't get confused by history from a different task (e.g., don't show "Refund" history when the user is trying to "Book").

### Metadata Usage
Store any field you need to filter by (like `user_id`) in the `metadata` of the event. This makes it searchable in the graph layer.
