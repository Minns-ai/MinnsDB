# EventGraphDB JavaScript Client

JavaScript/TypeScript client library for EventGraphDB REST API.

## Installation

```bash
npm install eventgraphdb-client
```

## Quick Start

```typescript
import { EventGraphDBClient, Event, EventType } from "eventgraphdb-client";

// Create client
const client = new EventGraphDBClient({
  baseUrl: "http://127.0.0.1:3000",
  timeout: 30000,
});

// Process an event
const event: Event = {
  id: "evt_123",
  timestamp: Date.now() * 1000000, // nanoseconds
  agent_id: 1,
  agent_type: "code-assistant",
  session_id: 42,
  event_type: {
    Action: {
      action_name: "fix_bug",
      parameters: { bug_type: "null_reference" },
      outcome: {
        Success: {
          result: { tests_pass: true },
        },
      },
      duration_ns: 1500000000,
    },
  },
  causality_chain: [],
  context: {
    task_description: "Fix null reference error in user service",
    code_snapshot: "function getUserById(id) { return users[id]; }",
    error_messages: ["TypeError: Cannot read property 'name' of null"],
    recent_actions: ["run_tests", "check_logs"],
    environment_state: {},
  },
  metadata: {},
};

// Process event (automatic learning happens!)
const result = await client.processEvent(event);
console.log(`Created ${result.nodes_created} nodes`);
console.log(`Detected ${result.patterns_detected} patterns`);

// Query learned knowledge
const memories = await client.getAgentMemories(1, 10);
const strategies = await client.getAgentStrategies(1, 10);

// Get action suggestions (Policy Guide)
const suggestions = await client.getActionSuggestions("ctx_abc123", undefined, 5);
console.log("What should I do next?");
suggestions.forEach((s) => {
  console.log(`${s.action_name}: ${(s.success_probability * 100).toFixed(1)}% success`);
  console.log(`  ${s.reasoning}`);
});
```

## API Methods

### Event Processing

#### `processEvent(event: Event): Promise<ProcessEventResponse>`

Process a new event through the EventGraphDB system. This automatically triggers:
- Episode detection
- Memory formation
- Strategy extraction
- Reinforcement learning

**Example:**

```typescript
const result = await client.processEvent(event);
console.log(`Processing time: ${result.processing_time_ms}ms`);
```

### Memory Queries

#### `getAgentMemories(agentId: number, limit?: number): Promise<MemoryResponse[]>`

Get memories for a specific agent, sorted by strength.

**Example:**

```typescript
const memories = await client.getAgentMemories(1, 5);
memories.forEach((m) => {
  console.log(`Memory ${m.id}: strength=${m.strength.toFixed(2)}`);
  console.log(`  Accessed ${m.access_count} times`);
});
```

### Strategy Queries

#### `getAgentStrategies(agentId: number, limit?: number): Promise<StrategyResponse[]>`

Get strategies learned by a specific agent, sorted by quality score.

**Example:**

```typescript
const strategies = await client.getAgentStrategies(1, 5);
strategies.forEach((s) => {
  const successRate = s.success_count / (s.success_count + s.failure_count);
  console.log(`${s.name}: ${(successRate * 100).toFixed(1)}% success rate`);
  s.reasoning_steps.forEach((step) => {
    console.log(`  ${step.sequence_order}. ${step.description}`);
  });
});
```

### Policy Guide

#### `getActionSuggestions(contextHash: string, lastActionNode?: number, limit?: number): Promise<ActionSuggestionResponse[]>`

Get action suggestions based on current context. Returns recommended next actions with success probabilities.

**Example:**

```typescript
const suggestions = await client.getActionSuggestions("ctx_debugging_session", undefined, 5);

console.log("Suggested next actions:");
suggestions.forEach((s, i) => {
  console.log(`${i + 1}. ${s.action_name}`);
  console.log(`   Success probability: ${(s.success_probability * 100).toFixed(1)}%`);
  console.log(`   Evidence: ${s.evidence_count} similar cases`);
  console.log(`   Reasoning: ${s.reasoning}`);
});
```

### Episode Queries

#### `getEpisodes(limit?: number): Promise<EpisodeResponse[]>`

Get completed episodes detected by the system.

**Example:**

```typescript
const episodes = await client.getEpisodes(10);
episodes.forEach((e) => {
  console.log(`Episode ${e.id}: ${e.event_count} events`);
  console.log(`  Significance: ${e.significance.toFixed(2)}`);
  if (e.outcome) {
    console.log(`  Outcome: ${e.outcome}`);
  }
});
```

### System Queries

#### `getStats(): Promise<StatsResponse>`

Get system-wide statistics.

**Example:**

```typescript
const stats = await client.getStats();
console.log(`Total events processed: ${stats.total_events_processed}`);
console.log(`Memories formed: ${stats.total_memories_formed}`);
console.log(`Strategies extracted: ${stats.total_strategies_extracted}`);
console.log(`Average processing time: ${stats.average_processing_time_ms.toFixed(2)}ms`);
```

#### `healthCheck(): Promise<HealthResponse>`

Check system health status.

**Example:**

```typescript
const health = await client.healthCheck();
if (health.is_healthy) {
  console.log(`System healthy: v${health.version}`);
  console.log(`Graph size: ${health.node_count} nodes, ${health.edge_count} edges`);
} else {
  console.log(`System degraded: ${health.status}`);
}
```

## Configuration

```typescript
const client = new EventGraphDBClient({
  baseUrl: "http://127.0.0.1:3000", // Server URL
  timeout: 30000, // Request timeout in ms
  headers: {
    // Custom headers
    "X-Custom-Header": "value",
  },
});
```

## Error Handling

```typescript
import { EventGraphDBError } from "eventgraphdb-client";

try {
  await client.processEvent(event);
} catch (error) {
  if (error instanceof EventGraphDBError) {
    console.error(`Error ${error.statusCode}: ${error.message}`);
    if (error.details) {
      console.error(`Details: ${error.details}`);
    }
  } else {
    console.error("Unexpected error:", error);
  }
}
```

## TypeScript Support

The library is written in TypeScript and includes full type definitions:

```typescript
import type {
  Event,
  EventType,
  ActionEvent,
  ObservationEvent,
  CognitiveEvent,
  ActionOutcome,
  EventContext,
  MemoryResponse,
  StrategyResponse,
  ActionSuggestionResponse,
} from "eventgraphdb-client";
```

## Complete Example: AI Code Assistant

```typescript
import { EventGraphDBClient, Event, createClient } from "eventgraphdb-client";

async function runCodeAssistant() {
  const client = createClient({ baseUrl: "http://localhost:3000" });

  // 1. Process debugging event
  const debugEvent: Event = {
    id: `evt_${Date.now()}`,
    timestamp: Date.now() * 1000000,
    agent_id: 1,
    agent_type: "code-debugger",
    session_id: 42,
    event_type: {
      Action: {
        action_name: "add_null_check",
        parameters: {
          variable: "user",
          location: "getUserName()",
        },
        outcome: {
          Success: {
            result: { tests_pass: true, null_errors: 0 },
          },
        },
        duration_ns: 2000000000,
      },
    },
    causality_chain: [],
    context: {
      task_description: "Fix null reference error",
      code_snapshot: "function getUserName(id) { return users[id].name; }",
      error_messages: ["TypeError: Cannot read property 'name' of null"],
      recent_actions: ["run_tests", "analyze_stack_trace"],
      environment_state: { language: "javascript", framework: "node" },
    },
    metadata: { severity: "high", confidence: 0.9 },
  };

  const result = await client.processEvent(debugEvent);
  console.log(`Event processed in ${result.processing_time_ms}ms`);

  // 2. Check system learning
  const stats = await client.getStats();
  console.log(`\nSystem has learned from ${stats.total_events_processed} events`);
  console.log(`Formed ${stats.total_memories_formed} memories`);
  console.log(`Extracted ${stats.total_strategies_extracted} strategies`);

  // 3. Get action suggestions for similar context
  const suggestions = await client.getActionSuggestions("null_reference_error", undefined, 3);

  console.log("\nWhat should I do when I see a null reference error?");
  suggestions.forEach((s, i) => {
    console.log(`\n${i + 1}. ${s.action_name}`);
    console.log(`   Success rate: ${(s.success_probability * 100).toFixed(1)}%`);
    console.log(`   Based on ${s.evidence_count} similar cases`);
    console.log(`   ${s.reasoning}`);
  });

  // 4. Review learned strategies
  const strategies = await client.getAgentStrategies(1, 3);
  console.log("\nLearned debugging strategies:");
  strategies.forEach((s) => {
    const successRate = (s.success_count / (s.success_count + s.failure_count)) * 100;
    console.log(`\n${s.name} (${successRate.toFixed(1)}% success rate)`);
    s.reasoning_steps.forEach((step) => {
      console.log(`  ${step.sequence_order}. ${step.description}`);
    });
  });
}

runCodeAssistant().catch(console.error);
```

## License

MIT
