---
name: workflow-run
description: Execute a stored workflow with step-by-step orchestration
user_invocable: true
---

# /workflow-run — Execute a Stored Workflow

Execute a previously created workflow with step-by-step orchestration, tracking progress through MinnsDB's graph.

## Usage

When the user invokes `/workflow-run <workflow_id>`:

## Execution Loop

1. **Load workflow state**:
   ```bash
   minns workflow status <workflow_id>
   ```
   This shows all steps, their current states, roles, tasks, and dependencies.

2. **Identify ready steps**: Steps whose dependencies are all `completed` and whose own state is `pending` or `ready`.

3. **For each ready step**, transition it to `running`:
   ```bash
   minns workflow step-transition <workflow_id> <step_id> --state running
   ```

4. **Execute the step** based on its role:
   - `code` — Read relevant files, make changes, write code
   - `architect` / `design` — Analyze codebase, produce design decisions
   - `test` — Write tests, run them via Bash
   - `review` — Use the Explore agent to analyze changes
   - `research` — Use the Explore agent for investigation
   - `ops` — Run infrastructure/deployment commands

5. **Report completion**:
   ```bash
   minns workflow step-transition <workflow_id> <step_id> --state completed --result "<summary of what was done>"
   ```
   Or on failure:
   ```bash
   minns workflow step-transition <workflow_id> <step_id> --state failed --result "<what went wrong>"
   ```

6. **Repeat** — check status again and process newly ready steps until all steps are completed or a failure requires intervention.

7. **Capture outcome**:
   ```bash
   minns learn "Workflow completed: <summary>" --category workflow_result --tags workflow
   ```

## Parallel Execution

Steps that are independent (no shared dependencies) can be executed in parallel using subagents:
- Launch an Agent for each ready step
- Wait for all to complete
- Report results

## Error Handling

When a step fails:
1. Report the failure via `step-transition --state failed`
2. Present the error to the user
3. Offer options:
   - **Retry** — transition back to `running` and try again
   - **Skip** — mark as `completed` with a note and continue
   - **Abort** — stop the workflow
   - **Modify** — update the workflow definition and retry

## Examples

```
/workflow-run 42
/workflow-run 1337
```
