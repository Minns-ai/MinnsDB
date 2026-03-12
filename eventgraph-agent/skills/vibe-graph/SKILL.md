---
name: vibe-graph
description: Design a workflow via 3-stage Vibe Graphing pipeline
user_invocable: true
---

# /vibe-graph — Design Workflow via Vibe Graphing

Convert a natural language intent into an executable workflow stored as real graph nodes and edges in EventGraphDB. Uses a 3-stage compilation pipeline (Role Assignment → Structure Design → Semantic Completion).

**Claude does all reasoning. The CLI provides graph context. The server stores the result.**

## Usage

When the user invokes `/vibe-graph <intent>`:

## Step 0: Gather Context

Query the graph for relevant prior knowledge:

```bash
eventgraph vibe-graph "<intent>" --json
```

This returns:
- `graph_context` — NLQ answer, resolved entities, related memories and strategies
- `proven_strategies` — strategies that have worked for similar tasks
- `existing_workflows` — workflows that may be reusable or adaptable

Use this context to inform your design decisions in the stages below.

## Stage 1: Role Assignment

**You (Claude) reason about this — do NOT delegate to the CLI.**

Based on the intent and graph context, determine the roles/agents needed:

1. Decompose the intent into distinct concerns
2. Assign a role to each concern (e.g., `code`, `review`, `test`, `research`, `architect`, `ops`)
3. Consider which roles can work in parallel vs. sequentially

Present the roles to the user in a table:

| Role | Responsibility | Why |
|------|---------------|-----|
| architect | Design the API surface | Ensures consistency with existing patterns |
| code | Implement the changes | Core implementation work |
| test | Write and run tests | Verify correctness |
| review | Review the final changes | Quality gate |

**Wait for user approval before proceeding.** They may add, remove, or modify roles.

## Stage 2: Structure Design

**You (Claude) reason about this — do NOT delegate to the CLI.**

Design the workflow graph:

1. Create a step for each role, with a clear `task` description
2. Define `depends_on` relationships (what must complete before this step starts)
3. Define `inputs` and `outputs` (data flow between steps)
4. Maximize parallelism — only add dependencies where truly needed
5. Incorporate proven strategies from the graph context

Present the structure as a dependency graph:

```
[architect] ──→ [code] ──→ [review]
                  │
                  └──→ [test] ──→ [review]
```

With a table of steps:

| Step ID | Role | Task | Depends On | Inputs | Outputs |
|---------|------|------|------------|--------|---------|
| design | architect | Design API surface matching existing patterns | — | intent | api_spec |
| implement | code | Implement the API changes | design | api_spec | code_changes |
| test | test | Write integration tests for new endpoints | implement | code_changes | test_results |
| review | review | Review implementation and tests | implement, test | code_changes, test_results | approval |

**Wait for user approval before proceeding.** They may restructure dependencies or modify tasks.

## Stage 3: Semantic Completion & Save

After approval, build the complete workflow JSON and save it to EventGraphDB:

```bash
cat <<'WORKFLOW_JSON' | eventgraph workflow create --file - --group_id "$EVENTGRAPH_GROUP_ID"
{
  "name": "<descriptive-name>",
  "intent": "<original intent>",
  "description": "<detailed description of what this workflow accomplishes>",
  "steps": [
    {
      "id": "design",
      "role": "architect",
      "task": "Design API surface matching existing patterns in the codebase",
      "depends_on": [],
      "inputs": ["intent"],
      "outputs": ["api_spec"],
      "metadata": {}
    },
    {
      "id": "implement",
      "role": "code",
      "task": "Implement the API changes based on the approved design",
      "depends_on": ["design"],
      "inputs": ["api_spec"],
      "outputs": ["code_changes"],
      "metadata": {}
    }
  ],
  "metadata": {}
}
WORKFLOW_JSON
```

The server decomposes this into:
- A root **Concept node** (`workflow:<name>`, type `Strategy`)
- A **Concept node** per step (with role, task, inputs, outputs as properties)
- `workflow:member_of` edges from each step → root
- `workflow:depends_on` edges between steps (carrying data flow info)
- A **StateMachine** per step for execution tracking (`pending → ready → running → completed | failed`)

Present the result: workflow ID, node count, edge count, step-to-node-ID mapping.

## After Saving

Offer the user options:
1. **Execute now** — invoke `/workflow-run <workflow_id>`
2. **Execute later** — just save for future use
3. **Modify** — adjust and re-save (uses `eventgraph workflow update`)

## Design Principles

- **Maximize parallelism**: Independent steps should not depend on each other
- **Reuse proven strategies**: If the graph context contains strategies that worked before, incorporate them
- **Keep steps atomic**: Each step should have one clear deliverable
- **Data flows matter**: Be explicit about what each step produces and consumes
- **Claude reasons, CLI stores**: Never ask the CLI to do reasoning — that's your job

## Examples

```
/vibe-graph Review the PR changes and write comprehensive tests
/vibe-graph Investigate the performance regression in the API
/vibe-graph Refactor the authentication module to use OAuth2
/vibe-graph Add a new REST endpoint for user preferences with full test coverage
```
