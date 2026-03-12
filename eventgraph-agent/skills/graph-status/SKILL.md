---
name: graph-status
description: Show EventGraphDB knowledge graph dashboard
user_invocable: true
---

# /graph-status — Knowledge Graph Dashboard

Display a formatted overview of the EventGraphDB knowledge graph status.

## Usage

When the user invokes `/graph-status`, run:

```bash
eventgraph status
```

## Behavior

1. Run `eventgraph status` to fetch health and statistics
2. Present the dashboard showing:
   - Server health and uptime
   - Graph size (nodes, edges)
   - Knowledge metrics (events, episodes, memories, strategies)
   - Processing performance
3. If the server is unreachable, inform the user and suggest starting it with:
   ```bash
   $HOME/.cargo/bin/cargo run --release -p eventgraphdb-server
   ```
