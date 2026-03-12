# EventGraph Agent

Claude Code plugin for cross-session memory and workflow orchestration powered by EventGraphDB.

## Quick Start

```bash
# 1. Build the CLI
$HOME/.cargo/bin/cargo build --release -p eventgraph-cli

# 2. Start EventGraphDB server
$HOME/.cargo/bin/cargo run --release -p eventgraphdb-server &

# 3. Add the CLI to your PATH (or use full path)
export PATH="$PATH:$(pwd)/target/release"

# 4. Set project scope (optional)
export EVENTGRAPH_GROUP_ID="my-project"
```

## Architecture

```
Claude (reasoning) ──→ eventgraph CLI (I/O) ──→ EventGraphDB server (storage)
```

- **Claude** does all reasoning: decomposes intents into roles, designs workflow structure, decides execution order
- **CLI** is thin I/O: sends workflow JSON, queries graph context, reports step transitions
- **Server** handles graph storage: decomposes workflow JSON into Concept nodes + Association edges with temporal tracking

### Workflow Graph Structure

When you create a workflow, the server stores it as real graph structure:

- **Root node**: `Concept(workflow:<name>, Strategy)` — the workflow itself
- **Step nodes**: `Concept(wf_step:<name>:<step_id>, Strategy)` — one per step
- **Member edges**: `workflow:member_of` from each step → root
- **Dependency edges**: `workflow:depends_on` between steps (carries data flow info)
- **State machines**: `StateMachine` per step for execution tracking (`pending → ready → running → completed | failed`)

Changes are tracked temporally via `valid_from`/`valid_until` on edges.

## CLI Commands

### Memory

```bash
eventgraph recall "how does auth work?"              # NLQ + hybrid search
eventgraph learn "We chose JWT with RS256" -c decision -t auth,security
eventgraph search "database migration" --mode hybrid --limit 10
```

### Workflows

```bash
eventgraph workflow create --file workflow.json      # Create from JSON (or - for stdin)
eventgraph workflow list                             # List all workflows
eventgraph workflow status <workflow_id>             # Show steps and states
eventgraph workflow step-transition <wf_id> <step_id> --state running
eventgraph workflow step-transition <wf_id> <step_id> --state completed --result "Done"
eventgraph workflow update <workflow_id> --file updated.json
eventgraph workflow delete <workflow_id>
```

### Vibe Graphing

```bash
eventgraph vibe-graph "Review PR and write tests"           # Get graph context for design
eventgraph vibe-graph "Review PR and write tests" --json    # JSON output for programmatic use
```

The `/vibe-graph` skill in Claude Code runs the full 3-stage pipeline:
1. **Role Assignment** — Claude determines roles needed
2. **Structure Design** — Claude designs the dependency graph
3. **Semantic Completion** — Claude builds full JSON, server stores as graph nodes+edges

### Graph Queries

```bash
eventgraph query neighbors 42
eventgraph query causal-path 10 50
eventgraph query communities
eventgraph strategies "refactoring a monolith"
eventgraph plan "migrate to PostgreSQL"
```

### Status

```bash
eventgraph status
```

## Skills (Claude Code)

- `/recall` — Query cross-session memory
- `/learn` — Store a learning
- `/graph-status` — Knowledge graph dashboard
- `/vibe-graph` — Design workflow via 3-stage Vibe Graphing pipeline
- `/workflow-run` — Execute a stored workflow step-by-step

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `EVENTGRAPH_URL` | `http://127.0.0.1:3000` | EventGraphDB server URL |
| `EVENTGRAPH_GROUP_ID` | `default` | Project-scoped isolation key |

## Templates

Pre-built workflow templates in `templates/`:

- `code-review.json` — Comprehensive PR review
- `bug-investigation.json` — Systematic bug investigation
- `feature-implementation.json` — End-to-end feature implementation
- `refactoring.json` — Safe refactoring workflow

## Self-Improvement Loop

1. Workflow execution creates events → EventGraphDB's episode detection fires
2. Episodes → memory formation (what worked/failed)
3. Memories → strategy extraction (proven patterns)
4. Future `/vibe-graph` queries strategies → better workflow designs
