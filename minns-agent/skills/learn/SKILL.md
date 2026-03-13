---
name: learn
description: Store a learning into MinnsDB for cross-session recall
user_invocable: true
---

# /learn — Store a Learning

Explicitly store a learning, decision, pattern, or insight into the MinnsDB knowledge graph for recall in future sessions.

## Usage

When the user invokes `/learn <content>`, run:

```bash
minns learn "<content>" --category <detected_category> --tags <relevant_tags>
```

## Behavior

1. Analyze the content to detect the appropriate category:
   - `decision` — architectural or design decisions
   - `pattern` — code patterns or conventions
   - `bug` — bug reports or root causes
   - `architecture` — system architecture details
   - `convention` — team conventions or standards
   - `learning` — general learnings (default)
2. Extract relevant tags from the content and current context (file names, technologies, concepts)
3. Run the `minns learn` command with detected category and tags
4. Confirm storage to the user

## Examples

```
/learn We chose JWT with RS256 for API authentication because it supports key rotation
/learn The user service must be called before the order service due to foreign key constraints
/learn Always use structured logging with correlation IDs in this codebase
```
