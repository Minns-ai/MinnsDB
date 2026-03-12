---
name: recall
description: Query cross-session memory from EventGraphDB
user_invocable: true
---

# /recall — Query Cross-Session Memory

Query the EventGraphDB knowledge graph for information stored across previous sessions.

## Usage

When the user invokes `/recall <query>`, run:

```bash
eventgraph recall "<query>"
```

## Behavior

1. Run `eventgraph recall "<query>"` to search the knowledge graph
2. Present the results organized by type:
   - **Answer**: The direct NLQ response
   - **Entities**: Resolved graph entities
   - **Memories**: Related memories from past sessions
   - **Strategies**: Proven patterns that may apply
3. If the answer has low confidence, suggest refining the query or trying `eventgraph search "<query>" --mode hybrid`
4. If no results found, suggest the user store relevant information with `/learn`

## Examples

```
/recall how does authentication work?
/recall what decisions were made about the database schema?
/recall what patterns have we used for error handling?
```
