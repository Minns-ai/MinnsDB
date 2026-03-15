//! LLM prompt templates for conversation compaction.

// ────────── Main Compaction System Prompt ──────────

pub(crate) fn compaction_system_prompt(categories: &str, category_enum: &str) -> String {
    format!(
        r#"You are an information extraction system. Given a conversation transcript, extract:

1. "facts": Self-contained propositions (cross-message inferences, implicit knowledge).
   Each fact must be understandable on its own without the conversation context.

   Each: {{
     "statement": "self-contained proposition preserving full context",
     "subject": "entity name", "predicate": "relationship/attribute", "object": "value/target",
     "category": "{category_enum}",
     "temporal_signal": "recently|since last week|every morning|null",
     "depends_on": "condition for this fact to hold, or null",
     "is_update": true if this replaces a previous fact in the same category,
     "cardinality_hint": "single|multi|append (only if category is not in the known list)"
   }}

   CRITICAL:
   - FIRST-PERSON REFERENCES: When the speaker refers to themselves (I, me, my, we, our,
     myself, etc. in ANY language), always use "user" as the subject or object name.
     BAD:  "I live in Tokyo" / subject: "I"
     GOOD: "User lives in Tokyo" / subject: "user"
     This applies to ALL languages — use "user" regardless of source language.
   - Extract PROPOSITIONS not bare triplets:
     BAD:  "User lives_in Tokyo"
     GOOD: "User recently moved to Tokyo for a six-month work assignment"
   - Conditional facts MUST include depends_on:
     "User walks in Yoyogi Park on Saturdays" → depends_on: "User lives in Tokyo"
   - State changes MUST set is_update: true
   - Extract ONLY the CURRENT/LATEST state. Do NOT extract superseded historical states.

   Category determines supersession:
{categories}
   - "other": anything else

   If you use a category not in the list above, include "cardinality_hint": "single"|"multi"|"append"
   to indicate: single = one value at a time (newest supersedes all), multi = multiple values coexist, append = never supersede.

   For multi-valued categories (routine, preference, relationship), the predicate must describe
   the SPECIFIC role or type of that fact — never use the category name itself as the predicate.
   Derive the predicate from the context: time-of-day, frequency, activity type, relationship role, etc.

   For "preference" category facts, include "sentiment": a value from -1.0 to 1.0.
   -1.0 = strong dislike, 0.0 = neutral, 1.0 = strong like.

2. "goals": User objectives/intentions detected.
   Each: {{ "description", "status": "active"|"completed"|"abandoned", "owner" }}

3. "procedural_summary": Structured session summary, or null if no procedural content.
   {{ "objective", "progress_status": "completed"|"in_progress"|"blocked"|"abandoned",
     "steps": [{{ "step_number", "action", "result", "outcome": "success"|"failure"|"partial"|"pending" }}],
     "overall_summary", "takeaway" }}

Rules:
- Look for cross-message inferences: facts apparent only by combining multiple messages
- Focus on relationships, preferences, states, and implicit knowledge
- For state changes across sessions, extract the LATEST state only
- Output ONLY valid JSON

Example:
{{
  "facts": [
    {{"statement": "User recently moved to New York for work", "subject": "User", "predicate": "lives_in", "object": "New York", "category": "location", "temporal_signal": "recently", "is_update": true}},
    {{"statement": "User takes morning walks in Battery Park on weekends", "subject": "User", "predicate": "weekend_activity", "object": "Battery Park", "category": "routine", "depends_on": "User lives in New York"}}
  ],
  "goals": [],
  "procedural_summary": null
}}"#
    )
}

// ────────── Per-Turn Extraction Prompt ──────────

/// Per-turn extraction prompt that takes rolling context and graph state.
pub(crate) const TURN_EXTRACTION_PROMPT: &str = r#"You are an information extraction system. Given a single conversation exchange, extract ONLY NEW facts or state changes as self-contained propositions.

PREVIOUSLY ESTABLISHED FACTS (do NOT re-extract these):
{rolling_facts}

CURRENT ENTITY STATES (from the knowledge graph):
{graph_state}

CURRENT EXCHANGE:
{messages}

Extract ONLY new facts or state changes introduced in the current exchange.
Each fact must be a SELF-CONTAINED PROPOSITION — understandable on its own without the conversation.
If a fact changes a previously established state, extract ONLY the NEW value and mark is_update: true.
Use entity names that match existing graph entities when referring to the same thing.

Output format:
{
  "facts": [
    {
      "statement": "self-contained proposition preserving full context",
      "subject": "entity name",
      "predicate": "relationship or attribute",
      "object": "value or target entity",
      "category": "location|routine|preference|relationship|work|financial|health|education|other",
      "temporal_signal": "recently|since last week|used to|every morning|null",
      "depends_on": "condition that must be true for this fact to hold, or null",
      "is_update": true if this replaces a previous fact in the same category
    }
  ]
}

CRITICAL RULES:

1. PROPOSITIONS, NOT TRIPLETS:
   BAD:  {"subject": "User", "predicate": "lives_in", "object": "Tokyo"}
   GOOD: {"statement": "User recently moved to Tokyo for a work assignment", "subject": "User", "predicate": "lives_in", "object": "Tokyo", "temporal_signal": "recently", "is_update": true}

2. CONDITIONAL DEPENDENCIES — location-dependent facts MUST specify depends_on:
   "User explores Yoyogi Park on Saturdays" → depends_on: "User lives in Tokyo"
   "User visits Feira da Ladra flea market on Saturdays" → depends_on: "User lives in Lisbon"
   When the user moves, dependent facts automatically become stale.

3. TEMPORAL SIGNALS — capture when things changed:
   "moved last week" → temporal_signal: "last week"
   "every morning" → temporal_signal: "every morning"
   "used to" → temporal_signal: "used to" (marks historical, not current)
   "since moving" → temporal_signal: "since moving"

4. STATE CHANGES — mark updates explicitly:
   "I moved to NYC" → is_update: true (supersedes previous location)
   "I switched jobs" → is_update: true (supersedes previous work)
   "I started running" → is_update: false (new habit, doesn't replace anything)

5. CATEGORY determines supersession:
   - "location": where someone lives, moved to
   - "routine": daily habits, regular activities
   - "preference": likes, dislikes, favorites
   - "relationship": connections between people
   - "work": job, employer, role
   - "financial": payments, debts, expenses
   - "other": anything else

6. FINANCIAL facts — extract structured payment details:
   - "subject": the payer
   - "predicate": what was paid for
   - "object": amount with currency (e.g. "179 EUR")
   - "amount": numeric amount (e.g. 179.0)
   - "split_with": list of people to split with, or ["all"]

7. SENTIMENT for preference facts:
   When category is "preference", include "sentiment": a value from -1.0 to 1.0.
   -1.0 = strong dislike, 0.0 = neutral, 1.0 = strong like.
   "I love sushi" → sentiment: 0.9
   "I hate early mornings" → sentiment: -0.8
   "Coffee is okay" → sentiment: 0.2

8. FIRST-PERSON NORMALIZATION:
   When the speaker refers to themselves, always use "user" as the entity name.
   "I moved to NYC" → subject: "user", object: "NYC"
   "My sister is Alice" → subject: "user", object: "Alice"
   "Je vis à Paris" → subject: "user", object: "Paris"
   Never use pronouns (I, me, my, we, our) as entity names.

Output ONLY valid JSON."#;

// ────────── 3-Call Cascade Prompts ──────────

pub(crate) const CASCADE_ENTITY_SYSTEM: &str = r#"Extract ALL entity mentions from the conversation exchange.
Entities include: people, places, organizations, things, concepts.
Use existing entity names when referring to the same entity (normalize).
Resolve pronouns when the referent is clear.
When the speaker refers to themselves (I, me, my, we, our, etc.), use "user" as the entity name.

Output ONLY valid JSON:
{ "entities": [{ "name": "entity name", "type": "person|place|organization|thing|concept", "mentions": ["exact text mentions"] }] }"#;

pub(crate) fn cascade_relationship_prompt(categories: &str, category_enum: &str) -> String {
    format!(
        r#"Discover relationships between entities. Focus on NEW or CHANGED relationships not already in graph state. Identify category and state changes.

Categories:
{categories}
- "other": anything else

For multi-valued categories (routine, preference, relationship), the predicate must describe
the SPECIFIC role or type — never use the category name itself as the predicate.
Derive the predicate from the context: time-of-day, frequency, activity type, relationship role, etc.

If you use a category not in the list above, include "cardinality_hint": "single"|"multi"|"append"
to indicate: single = one value at a time (newest supersedes all), multi = multiple values coexist, append = never supersede.

When the speaker refers to themselves, always use "user" as subject or object.

Output ONLY valid JSON:
{{ "relationships": [{{ "subject": "entity", "predicate": "relationship", "object": "value", "category": "{category_enum}", "is_state_change": true/false, "temporal_hint": "recently|since last week|null" }}] }}"#
    )
}

pub(crate) fn cascade_fact_prompt(category_enum: &str) -> String {
    format!(
        r#"Produce self-contained factual propositions from discovered relationships.
Each statement must be understandable without conversation context.
Mark is_update=true for state changes. Add depends_on for location-dependent facts.

For multi-valued categories (routine, preference, relationship), the predicate must describe
the SPECIFIC role or type — never use the category name itself as the predicate.
Derive the predicate from the context: time-of-day, frequency, activity type, relationship role, etc.

If you use a category not in the list above, include "cardinality_hint": "single"|"multi"|"append"
to indicate: single = one value at a time (newest supersedes all), multi = multiple values coexist, append = never supersede.

When the speaker refers to themselves, always use "user" as subject or object.

For "preference" category facts, include "sentiment": -1.0 to 1.0 (-1.0 = strong dislike, 1.0 = strong like).

Output ONLY valid JSON:
{{
  "facts": [
    {{
      "statement": "self-contained proposition preserving full context",
      "subject": "entity name",
      "predicate": "relationship or attribute",
      "object": "value or target entity",
      "category": "{category_enum}",
      "temporal_signal": "recently|since last week|used to|every morning|null",
      "depends_on": "condition that must be true for this fact to hold, or null",
      "is_update": true if this replaces a previous fact in the same category,
      "cardinality_hint": "single|multi|append (only if category is not in the known list)"
    }}
  ]
}}

CRITICAL RULES:
1. PROPOSITIONS, NOT TRIPLETS:
   BAD:  {{"subject": "User", "predicate": "lives_in", "object": "Tokyo"}}
   GOOD: {{"statement": "User recently moved to Tokyo for a work assignment", ...}}
2. CONDITIONAL DEPENDENCIES — location-dependent facts MUST specify depends_on
3. STATE CHANGES — mark is_update: true for superseding facts
4. FINANCIAL facts — include "amount" (numeric) and "split_with" (list) when applicable"#
    )
}

// ────────── Playbook Extraction Prompt ──────────

pub(crate) const PLAYBOOK_SYSTEM_PROMPT: &str = r#"You are a retrospective analysis system. Given a conversation transcript and goals, extract a playbook for each goal:

1. "what_worked": Actions/approaches that succeeded
2. "what_didnt_work": Actions/approaches that failed or were abandoned
3. "lessons_learned": Key takeaways for future attempts
4. "steps_taken": Brief ordered list of steps actually taken
5. "confidence": 0.0-1.0

If prior playbook experience is provided, use it to compare approaches and note what was
done differently this time. Reference prior lessons when relevant.

Output: { "playbooks": [ { "goal_description", "what_worked", "what_didnt_work", "lessons_learned", "steps_taken", "confidence" } ] }
Rules: One playbook per goal. Empty arrays if goal was barely discussed. Be specific, not generic. Output ONLY valid JSON"#;
