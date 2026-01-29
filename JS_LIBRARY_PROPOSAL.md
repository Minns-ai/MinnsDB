# EventGraphDB JavaScript SDK (Proposed)

A lightweight, promise-based client for interacting with EventGraphDB.

## Installation
```bash
npm install event-graph-db-sdk
```

## Basic Usage

```javascript
const { EventGraphClient } = require('event-graph-db-sdk');

const db = new EventGraphClient({
  baseUrl: 'http://localhost:3000',
  agentId: 1,
  agentType: 'coding-assistant'
});

async function main() {
  // 1. Ingest an event
  const result = await db.ingest({
    eventType: {
      Action: {
        action_name: 'git_commit',
        parameters: { message: 'feat: add SDK' },
        outcome: { Success: { result: 'commit hash 123' } }
      }
    },
    context: currentContext // helper to capture OS state
  });

  // 2. Get next best action suggestions
  const suggestions = await db.getSuggestions(currentContext.fingerprint);
  console.log(`Suggested next step: ${suggestions[0].action_name}`);

  // 3. Search semantic memory
  const claims = await db.searchClaims("What were the last project requirements?");
  claims.forEach(c => console.log(`- ${c.claim_text} (Confidence: ${c.confidence})`));
}
```

## API Reference

### `new EventGraphClient(options)`
- `baseUrl`: The URL of the EventGraphDB server.
- `agentId`: Default agent ID for this client instance.
- `agentType`: Default agent type for this client instance.

### `client.ingest(payload, options)`
Wraps `POST /api/events`. Automatically handles timestamping and UUID generation if omitted.
- `payload`: The event structure.
- `options.enableSemantic`: Enable LLM-based claim extraction (default: true).

### `client.getMemories(agentId, limit)`
Wraps `GET /api/memories/agent/:agent_id`. Returns long-term memories for the agent.

### `client.getSuggestions(contextHash, limit)`
Wraps `GET /api/suggestions`. Interacts with the Policy Guide to rank potential next steps.

### `client.searchClaims(query, options)`
Wraps `POST /api/claims/search`. Performs semantic vector search over the graph's extracted claims.

### `client.getGraph(query)`
Wraps `GET /api/graph`. Useful for building custom visualizers or debugging relationships.

---

## Example: Context-Aware Reasoning

```javascript
// Retrieve memories related to the current coding task
const relevantMemories = await db.getMemoriesByContext(currentContext);

if (relevantMemories.length > 0) {
  const topMemory = relevantMemories[0];
  console.log(`I remember a similar situation in session ${topMemory.session_id}`);
  console.log(`Previous outcome was: ${topMemory.outcome}`);
}
```
