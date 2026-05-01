//! LLM-driven conversation compaction.
//!
//! Runs AFTER the existing rule-based pipeline to extract:
//! - **Facts**: Cross-message inferences the rule-based classifier misses
//! - **Goals**: User objectives/intentions embedded in conversation flow
//! - **Procedural summary**: Structured session summary with steps and outcomes
//!
//! Extracted data is converted into Events (Observation, Cognitive, Action)
//! and a procedural Memory, then fed back through the pipeline.

use crate::conversation::graph_projection;
use crate::conversation::types::ConversationIngest;
use crate::conversation::types::ConversationMessage;
use crate::episodes::EpisodeOutcome;
use crate::llm_client::{parse_json_from_llm, LlmClient, LlmRequest};
use crate::memory::{Memory, MemoryTier, MemoryType};
use crate::memory_audit::MutationActor;
use crate::memory_classifier::{
    classify_memory_updates, resolve_target, ClassifiedOperation, MemoryAction,
};
use crate::structures::Graph;
use agent_db_events::core::{ActionOutcome, CognitiveType, EventContext, EventType, MetadataValue};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

/// Truncate a string to at most `max_bytes` bytes, ensuring the cut
/// falls on a valid UTF-8 character boundary.
fn safe_truncate(s: &str, max_bytes: usize) -> &str {
    if s.len() <= max_bytes {
        return s;
    }
    let mut end = max_bytes;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

// ────────── Quality Gate Constants ──────────

/// Minimum transcript length (chars) to attempt playbook extraction.
/// ~3-4 conversational turns minimum.
const MIN_PLAYBOOK_TRANSCRIPT_LEN: usize = 200;

/// Minimum confidence for an extracted playbook to be attached.
const MIN_PLAYBOOK_CONFIDENCE: f32 = 0.4;

// ────────── Rolling Summary ──────────

/// A rolling, incrementally updated summary of an ongoing conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationRollingSummary {
    /// Conversation case identifier.
    pub case_id: String,
    /// Current summary text.
    pub summary: String,
    /// Last update timestamp (nanoseconds since epoch).
    pub last_updated: u64,
    /// How many turns have been summarized so far.
    pub turn_count: u32,
    /// Rough token estimate of the current summary.
    pub token_estimate: u32,
}

const SUMMARY_UPDATE_SYSTEM_PROMPT: &str = r#"You are a conversation summarizer. Given an existing summary and new messages, produce an UPDATED summary that captures all key information.

Rules:
- Preserve all important facts, preferences, goals, and decisions from the existing summary
- Integrate new information from the recent messages
- Keep the summary concise (under 500 words)
- Focus on: facts, preferences, goals, decisions, relationships, state changes
- Drop: greetings, filler, repetition
- Output ONLY the updated summary text, no JSON"#;

/// Update the rolling summary with new conversation messages.
///
/// - First call (no existing summary): "Summarize this conversation"
/// - Subsequent calls: "Existing summary + new messages → updated summary"
///
/// Returns `None` on failure (fail-open).
pub async fn update_rolling_summary(
    llm: &dyn LlmClient,
    existing_summary: Option<&str>,
    new_messages: &[crate::conversation::types::ConversationMessage],
) -> Option<String> {
    if new_messages.is_empty() {
        return existing_summary.map(|s| s.to_string());
    }

    let mut messages_text = String::new();
    for msg in new_messages {
        messages_text.push_str(&msg.role);
        messages_text.push_str(": ");
        messages_text.push_str(&msg.content);
        messages_text.push('\n');
    }

    let user_prompt = if let Some(summary) = existing_summary {
        format!(
            "Existing summary:\n{}\n\nNew messages:\n{}\nProduce updated summary.",
            summary, messages_text
        )
    } else {
        format!("Summarize this conversation:\n{}", messages_text)
    };

    let request = LlmRequest {
        system_prompt: SUMMARY_UPDATE_SYSTEM_PROMPT.to_string(),
        user_prompt,
        temperature: 0.0,
        max_tokens: 1024,
        json_mode: false,
    };

    let response = llm.complete(request).await.ok()?;
    let text = response.content.trim().to_string();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

/// Format transcript using rolling summary + last N messages.
///
/// Instead of the full transcript, this uses the summary as context
/// and appends only the most recent messages for detail.
pub fn format_with_summary(
    summary: &ConversationRollingSummary,
    data: &ConversationIngest,
    recent_count: usize,
) -> String {
    let mut buf = String::new();
    buf.push_str("[Rolling Summary]\n");
    buf.push_str(&summary.summary);
    buf.push_str("\n\n[Recent Messages]\n");

    // Collect all messages in order
    let all_messages: Vec<&crate::conversation::types::ConversationMessage> = data
        .sessions
        .iter()
        .flat_map(|s| s.messages.iter())
        .collect();

    // Take the last `recent_count` messages
    let start = all_messages.len().saturating_sub(recent_count);
    for msg in &all_messages[start..] {
        buf.push_str(&msg.role);
        buf.push_str(": ");
        buf.push_str(&msg.content);
        buf.push('\n');
    }

    buf
}

// ────────── Types ──────────

/// LLM extraction response (deserialized from JSON).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CompactionResponse {
    pub facts: Vec<ExtractedFact>,
    pub goals: Vec<ExtractedGoal>,
    pub procedural_summary: Option<ProceduralSummary>,
}

/// A single extracted fact — a self-contained proposition with context.
///
/// Goes beyond simple (subject, predicate, object) triplets by preserving:
/// - Temporal signals (when things changed, duration, recency)
/// - Conditional dependencies (fact X is only true while condition Y holds)
/// - Supersession markers (this fact replaces a previous one)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedFact {
    /// Self-contained proposition preserving full context.
    /// Should be understandable without reference to the conversation.
    pub statement: String,
    pub subject: String,
    pub predicate: String,
    pub object: String,
    #[serde(default)]
    pub confidence: f32,
    /// Semantic category for state supersession grouping.
    /// Facts with the same entity + category supersede each other (latest wins).
    #[serde(default)]
    pub category: Option<String>,
    /// Numeric amount for financial facts (extracted by LLM).
    #[serde(default)]
    pub amount: Option<f64>,
    /// Who the cost is split with for financial facts.
    /// Contains person names, or ["all"] for split among all participants.
    #[serde(default)]
    pub split_with: Option<Vec<String>>,
    /// Temporal signal detected in the text (e.g., "recently", "since last week", "used to").
    /// Helps the system understand temporal ordering and state transitions.
    #[serde(default)]
    pub temporal_signal: Option<String>,
    /// If this fact is only valid while a condition holds (e.g., "while living in Tokyo").
    /// When the condition becomes false, this fact should be invalidated.
    #[serde(default)]
    pub depends_on: Option<String>,
    /// If true, this fact explicitly supersedes a previous fact in the same category.
    /// E.g., "moved to Tokyo" supersedes previous location facts.
    #[serde(default)]
    pub is_update: Option<bool>,
    /// Cardinality hint for unknown categories: "single"|"multi"|"append".
    /// Used to auto-register new domain slots when the LLM assigns a novel category.
    #[serde(default)]
    pub cardinality_hint: Option<String>,
    /// Sentiment polarity for preference facts (-1.0 = strong dislike, +1.0 = strong like).
    /// Only set when category is "preference".
    #[serde(default)]
    pub sentiment: Option<f32>,
    /// Partition key for multi-tenant isolation.
    /// Not populated by LLM — injected after extraction from the ingest request.
    #[serde(default, skip_serializing)]
    pub group_id: String,
    /// Request-level metadata propagated from the ingest request.
    /// Not populated by LLM — injected after extraction to flow through to graph edges.
    #[serde(default, skip_serializing)]
    pub ingest_metadata: std::collections::HashMap<String, serde_json::Value>,
}

/// A user goal/intention detected in the conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExtractedGoal {
    pub description: String,
    pub status: String,
    pub owner: String,
}

/// Structured procedural summary of the session.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProceduralSummary {
    pub objective: String,
    pub progress_status: String,
    pub steps: Vec<ProceduralStep>,
    pub overall_summary: String,
    pub takeaway: String,
}

/// A single step in a procedural summary.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProceduralStep {
    pub step_number: u32,
    pub action: String,
    pub result: String,
    pub outcome: String,
}

/// Result of the compaction process (returned for logging/response).
#[derive(Debug, Clone, Default, Serialize)]
pub struct CompactionResult {
    pub facts_extracted: usize,
    pub goals_extracted: usize,
    pub goals_deduplicated: usize,
    pub procedural_steps_extracted: usize,
    pub procedural_memory_created: bool,
    pub procedural_memory_id: Option<u64>,
    pub memories_updated: usize,
    pub memories_deleted: usize,
    pub playbooks_extracted: usize,
    pub llm_success: bool,
    pub tokens_used: u32,
}

// ────────── LLM Financial Extraction ──────────

#[derive(Deserialize)]
struct LlmTransaction {
    payer: String,
    payee: String,
    #[serde(default)]
    amount: f64,
}

/// Use LLM to extract financial transactions from text and convert them into `ExtractedFact`s.
///
/// Replaces the NER-based extraction with a structured LLM call that returns
/// payer/payee/amount triples directly.
async fn extract_financial_facts_llm(llm: &dyn LlmClient, text: &str) -> Vec<ExtractedFact> {
    let request = LlmRequest {
        system_prompt:
            "You are a financial transaction parser with access to a settlement solver tool.\n\n\
            Step 1: Extract all transactions from the text as structured triples.\n\
            Step 2: Pass them to the solver tool.\n\n\
            For each transaction, identify:\n\
            - payer: who pays\n\
            - payee: who receives\n\
            - amount: the number (0 if not stated)\n\n\
            IMPORTANT:\n\
            - Only include transactions that ACTUALLY happened\n\
            - If corrected, use corrected amount\n\
            - Payer paying for a group they're in → payer only, not payee\n\n\
            Output as JSON array for the solver:\n\
            [{\"payer\": \"name\", \"payee\": \"name\", \"amount\": number}, ...]"
                .to_string(),
        user_prompt: format!("Extract transactions and compute settlement:\n\n{}", text),
        temperature: 0.0,
        max_tokens: 512,
        json_mode: true,
    };

    let response = match llm.complete(request).await {
        Ok(r) => r,
        Err(e) => {
            tracing::debug!("LLM financial extraction failed (non-fatal): {}", e);
            return Vec::new();
        },
    };

    let parsed = match parse_json_from_llm(&response.content) {
        Some(v) => v,
        None => {
            tracing::debug!("LLM financial extraction returned unparseable JSON");
            return Vec::new();
        },
    };

    let transactions: Vec<LlmTransaction> = match serde_json::from_value(parsed) {
        Ok(t) => t,
        Err(e) => {
            tracing::debug!("LLM financial extraction deserialization failed: {}", e);
            return Vec::new();
        },
    };

    let facts: Vec<ExtractedFact> = transactions
        .into_iter()
        .map(|tx| {
            let amount = if tx.amount != 0.0 {
                Some(tx.amount)
            } else {
                None
            };
            let statement = if let Some(amt) = amount {
                format!("{} paid {} {:.2}", tx.payer, tx.payee, amt)
            } else {
                format!("{} paid {}", tx.payer, tx.payee)
            };
            ExtractedFact {
                statement,
                subject: tx.payer,
                predicate: "paid".to_string(),
                object: tx.payee,
                confidence: 1.0,
                category: Some("financial".to_string()),
                amount,
                split_with: None,
                temporal_signal: None,
                depends_on: None,
                is_update: Some(false),
                cardinality_hint: None,
                sentiment: None,
                group_id: Default::default(),
                ingest_metadata: Default::default(),
            }
        })
        .collect();

    tracing::info!("LLM financial extraction produced {} facts", facts.len(),);

    facts
}

/// Retrospective playbook for a single goal extracted from conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GoalPlaybook {
    pub goal_description: String,
    pub what_worked: Vec<String>,
    pub what_didnt_work: Vec<String>,
    pub lessons_learned: Vec<String>,
    #[serde(default)]
    pub steps_taken: Vec<String>,
    #[serde(default = "default_playbook_confidence")]
    pub confidence: f32,
}

fn default_playbook_confidence() -> f32 {
    0.5
}

/// LLM response containing per-goal playbooks.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlaybookExtractionResponse {
    pub playbooks: Vec<GoalPlaybook>,
}

// ────────── System Prompt ──────────

fn compaction_system_prompt(categories: &str, category_enum: &str) -> String {
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

// ────────── Transcript Formatting ──────────

/// Maximum transcript length sent to the LLM (chars). Keeps the tail.
const MAX_TRANSCRIPT_CHARS: usize = 16_000;

/// Gap in nanoseconds between turn timestamps (1 second).
/// Turn N's facts get `base_ts + N * TURN_GAP + offset * 1000`.
const TURN_GAP: u64 = 1_000_000_000;

// ────────── Per-Turn Processing ──────────

/// A single conversation turn (user+assistant pair) with positional metadata.
#[derive(Debug, Clone)]
pub struct ConversationTurn {
    pub messages: Vec<ConversationMessage>,
    pub session_index: usize,
    pub turn_index: usize,
}

/// Split a `ConversationIngest` into individual turns (user+assistant pairs).
///
/// Each turn contains one user message and the following assistant message (if any).
/// The `turn_index` is a global counter across all sessions, establishing temporal order.
pub fn split_into_turns(data: &ConversationIngest) -> Vec<ConversationTurn> {
    let mut turns = Vec::new();
    let mut global_turn = 0usize;

    for (session_idx, session) in data.sessions.iter().enumerate() {
        let mut i = 0;
        while i < session.messages.len() {
            let mut turn_msgs = Vec::new();

            // Take the current message (should be user)
            turn_msgs.push(session.messages[i].clone());
            i += 1;

            // If next message is assistant, include it in the same turn
            if i < session.messages.len() && session.messages[i].role == "assistant" {
                turn_msgs.push(session.messages[i].clone());
                i += 1;
            }

            turns.push(ConversationTurn {
                messages: turn_msgs,
                session_index: session_idx,
                turn_index: global_turn,
            });
            global_turn += 1;
        }
    }

    turns
}

/// Per-turn extraction prompt that takes rolling context and graph state.
const TURN_EXTRACTION_PROMPT: &str = r#"You are an information extraction system. Given a single conversation exchange, extract ONLY NEW facts or state changes as self-contained propositions.

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

// ────────── 3-Call Cascade Types ──────────

/// Entity discovered in Call 1 of the cascade.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CascadeEntity {
    #[serde(default)]
    name: String,
    #[serde(default, rename = "type")]
    entity_type: String,
    #[serde(default)]
    mentions: Vec<String>,
}

/// Response from Call 1: entity extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CascadeEntitiesResponse {
    #[serde(default)]
    entities: Vec<CascadeEntity>,
}

/// Relationship discovered in Call 2 of the cascade.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CascadeRelationship {
    #[serde(default)]
    subject: String,
    #[serde(default)]
    predicate: String,
    #[serde(default)]
    object: String,
    #[serde(default)]
    category: Option<String>,
    #[serde(default, deserialize_with = "deserialize_bool_lenient")]
    is_state_change: Option<bool>,
    #[serde(default)]
    temporal_hint: Option<String>,
}

/// Response from Call 2: relationship discovery.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct CascadeRelationshipsResponse {
    #[serde(default)]
    relationships: Vec<CascadeRelationship>,
}

// ────────── Cascade Prompts ──────────

const CASCADE_ENTITY_SYSTEM: &str = r#"Extract ALL entity mentions from the conversation exchange.
Entities include: people, places, organizations, things, concepts.
Use existing entity names when referring to the same entity (normalize).
Resolve pronouns when the referent is clear.
When the speaker refers to themselves (I, me, my, we, our, etc.), use "user" as the entity name.

Output ONLY valid JSON:
{ "entities": [{ "name": "entity name", "type": "person|place|organization|thing|concept", "mentions": ["exact text mentions"] }] }"#;

fn cascade_relationship_prompt(categories: &str, category_enum: &str) -> String {
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

fn cascade_fact_prompt(category_enum: &str) -> String {
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

// ────────── Cascade Functions ──────────

/// Format just the turn messages as "role: content\n" without injecting rolling context.
fn format_messages(turn: &ConversationTurn) -> String {
    let mut buf = String::new();
    for msg in &turn.messages {
        buf.push_str(&msg.role);
        buf.push_str(": ");
        buf.push_str(&msg.content);
        buf.push('\n');
    }
    buf
}

/// Convert a CascadeRelationship into a basic ExtractedFact deterministically (no LLM).
/// Used as fallback when Call 3 fails.
fn relationship_to_basic_fact(rel: &CascadeRelationship) -> ExtractedFact {
    let statement = format!("{} {} {}", rel.subject, rel.predicate, rel.object);
    ExtractedFact {
        statement,
        subject: rel.subject.clone(),
        predicate: rel.predicate.clone(),
        object: rel.object.clone(),
        confidence: 0.7,
        category: rel.category.clone(),
        amount: None,
        split_with: None,
        temporal_signal: rel.temporal_hint.clone(),
        depends_on: None,
        is_update: rel.is_state_change,
        cardinality_hint: None,
        sentiment: None,
        group_id: Default::default(),
        ingest_metadata: Default::default(),
    }
}

/// 3-call cascade extraction: entities → relationships → structured facts.
///
/// Falls back to single-call `extract_turn_facts` on early failures.
async fn extract_turn_facts_cascade(
    llm: &dyn LlmClient,
    messages_text: &str,
    rolling_facts: Option<&str>,
    graph_state: &str,
    known_entities: &[String],
    category_block: &str,
    category_enum: &str,
) -> Option<Vec<ExtractedFact>> {
    if messages_text.is_empty() {
        return None;
    }

    // ── Call 1: Entity Extraction ──
    let known_list = if known_entities.is_empty() {
        "(none)".to_string()
    } else {
        known_entities.join(", ")
    };
    let entity_user_prompt = format!(
        "KNOWN ENTITIES: {}\nEXCHANGE:\n{}",
        known_list, messages_text
    );

    let entity_request = LlmRequest {
        system_prompt: CASCADE_ENTITY_SYSTEM.to_string(),
        user_prompt: entity_user_prompt,
        temperature: 0.0,
        max_tokens: 512,
        json_mode: true,
    };

    let entities = match tokio::time::timeout(
        std::time::Duration::from_secs(8),
        llm.complete(entity_request),
    )
    .await
    {
        Ok(Ok(resp)) => {
            match parse_json_from_llm(&resp.content)
                .and_then(|v| serde_json::from_value::<CascadeEntitiesResponse>(v).ok())
            {
                Some(parsed) => {
                    tracing::info!(
                        "CASCADE call 1: extracted {} entities",
                        parsed.entities.len()
                    );
                    parsed.entities
                },
                None => {
                    tracing::warn!("CASCADE call 1: parse failed, falling back to single-call");
                    return extract_turn_facts_single_call_fallback(
                        llm,
                        messages_text,
                        rolling_facts,
                        graph_state,
                    )
                    .await;
                },
            }
        },
        Ok(Err(e)) => {
            tracing::warn!("CASCADE call 1 failed: {}, falling back to single-call", e);
            return extract_turn_facts_single_call_fallback(
                llm,
                messages_text,
                rolling_facts,
                graph_state,
            )
            .await;
        },
        Err(_) => {
            tracing::warn!("CASCADE call 1 timed out, falling back to single-call");
            return extract_turn_facts_single_call_fallback(
                llm,
                messages_text,
                rolling_facts,
                graph_state,
            )
            .await;
        },
    };

    // ── Call 2: Relationship Discovery ──
    let entities_json = serde_json::to_string(&entities).unwrap_or_default();
    let rolling = rolling_facts.unwrap_or("(none)");
    let gs = if graph_state.is_empty() {
        "(none)"
    } else {
        graph_state
    };

    let rel_user_prompt = format!(
        "PREVIOUSLY ESTABLISHED FACTS:\n{}\n\nCURRENT ENTITY STATES:\n{}\n\nENTITIES FOUND:\n{}\n\nEXCHANGE:\n{}",
        rolling, gs, entities_json, messages_text
    );

    let rel_request = LlmRequest {
        system_prompt: cascade_relationship_prompt(category_block, category_enum),
        user_prompt: rel_user_prompt,
        temperature: 0.0,
        max_tokens: 768,
        json_mode: true,
    };

    let relationships = match tokio::time::timeout(
        std::time::Duration::from_secs(10),
        llm.complete(rel_request),
    )
    .await
    {
        Ok(Ok(resp)) => {
            match parse_json_from_llm(&resp.content)
                .and_then(|v| serde_json::from_value::<CascadeRelationshipsResponse>(v).ok())
            {
                Some(parsed) => {
                    tracing::info!(
                        "CASCADE call 2: discovered {} relationships",
                        parsed.relationships.len()
                    );
                    parsed.relationships
                },
                None => {
                    tracing::warn!("CASCADE call 2: parse failed, falling back to single-call");
                    return extract_turn_facts_single_call_fallback(
                        llm,
                        messages_text,
                        rolling_facts,
                        graph_state,
                    )
                    .await;
                },
            }
        },
        Ok(Err(e)) => {
            tracing::warn!("CASCADE call 2 failed: {}, falling back to single-call", e);
            return extract_turn_facts_single_call_fallback(
                llm,
                messages_text,
                rolling_facts,
                graph_state,
            )
            .await;
        },
        Err(_) => {
            tracing::warn!("CASCADE call 2 timed out, falling back to single-call");
            return extract_turn_facts_single_call_fallback(
                llm,
                messages_text,
                rolling_facts,
                graph_state,
            )
            .await;
        },
    };

    if relationships.is_empty() {
        tracing::info!("CASCADE call 2: no relationships found, returning empty");
        return Some(vec![]);
    }

    // ── Call 3: Structured Fact Formation ──
    let rels_json = serde_json::to_string(&relationships).unwrap_or_default();

    let fact_user_prompt = format!(
        "PREVIOUSLY ESTABLISHED FACTS:\n{}\n\nCURRENT ENTITY STATES:\n{}\n\nDISCOVERED RELATIONSHIPS:\n{}\n\nORIGINAL EXCHANGE:\n{}",
        rolling, gs, rels_json, messages_text
    );

    let fact_request = LlmRequest {
        system_prompt: cascade_fact_prompt(category_enum),
        user_prompt: fact_user_prompt,
        temperature: 0.0,
        max_tokens: 1024,
        json_mode: true,
    };

    match tokio::time::timeout(
        std::time::Duration::from_secs(10),
        llm.complete(fact_request),
    )
    .await
    {
        Ok(Ok(resp)) => {
            match parse_json_from_llm(&resp.content) {
                Some(value) => {
                    // Reuse existing LenientFact deserialization
                    #[derive(Deserialize)]
                    struct LenientFact {
                        #[serde(default)]
                        statement: String,
                        #[serde(default)]
                        subject: String,
                        #[serde(default)]
                        predicate: String,
                        #[serde(default)]
                        object: String,
                        #[serde(default, deserialize_with = "deserialize_f32_lenient")]
                        confidence: f32,
                        #[serde(default)]
                        category: Option<String>,
                        #[serde(default)]
                        amount: Option<f64>,
                        #[serde(default)]
                        split_with: Option<Vec<String>>,
                        #[serde(default)]
                        temporal_signal: Option<String>,
                        #[serde(default)]
                        depends_on: Option<String>,
                        #[serde(default, deserialize_with = "deserialize_bool_lenient")]
                        is_update: Option<bool>,
                        #[serde(default)]
                        cardinality_hint: Option<String>,
                        #[serde(default, deserialize_with = "deserialize_f32_option_lenient")]
                        sentiment: Option<f32>,
                    }

                    #[derive(Deserialize)]
                    struct TurnResponse {
                        #[serde(default)]
                        facts: Vec<LenientFact>,
                    }

                    match serde_json::from_value::<TurnResponse>(value) {
                        Ok(parsed) => {
                            let facts: Vec<ExtractedFact> = parsed
                                .facts
                                .into_iter()
                                .filter(|f| !f.statement.is_empty() || !f.subject.is_empty())
                                .map(|f| ExtractedFact {
                                    statement: f.statement,
                                    subject: f.subject,
                                    predicate: f.predicate,
                                    object: f.object,
                                    confidence: if f.confidence > 0.0 {
                                        f.confidence
                                    } else {
                                        0.8
                                    },
                                    category: f.category,
                                    amount: f.amount,
                                    split_with: f.split_with,
                                    temporal_signal: f.temporal_signal,
                                    depends_on: f.depends_on,
                                    is_update: f.is_update,
                                    cardinality_hint: f.cardinality_hint,
                                    sentiment: f.sentiment,
                                    group_id: Default::default(),
                                    ingest_metadata: Default::default(),
                                })
                                .collect();
                            tracing::info!(
                                "CASCADE call 3: produced {} structured facts",
                                facts.len()
                            );
                            Some(facts)
                        },
                        Err(e) => {
                            tracing::warn!(
                                "CASCADE call 3: deser failed ({}), converting relationships deterministically",
                                e
                            );
                            let facts: Vec<ExtractedFact> = relationships
                                .iter()
                                .map(relationship_to_basic_fact)
                                .collect();
                            Some(facts)
                        },
                    }
                },
                None => {
                    tracing::warn!(
                        "CASCADE call 3: JSON parse failed, converting relationships deterministically"
                    );
                    let facts: Vec<ExtractedFact> = relationships
                        .iter()
                        .map(relationship_to_basic_fact)
                        .collect();
                    Some(facts)
                },
            }
        },
        Ok(Err(e)) => {
            tracing::warn!(
                "CASCADE call 3 failed ({}), converting relationships deterministically",
                e
            );
            let facts: Vec<ExtractedFact> = relationships
                .iter()
                .map(relationship_to_basic_fact)
                .collect();
            Some(facts)
        },
        Err(_) => {
            tracing::warn!("CASCADE call 3 timed out, converting relationships deterministically");
            let facts: Vec<ExtractedFact> = relationships
                .iter()
                .map(relationship_to_basic_fact)
                .collect();
            Some(facts)
        },
    }
}

/// Fallback: format a single-call transcript and use `extract_turn_facts`.
/// Used when cascade Call 1 or Call 2 fails.
async fn extract_turn_facts_single_call_fallback(
    llm: &dyn LlmClient,
    messages_text: &str,
    rolling_facts: Option<&str>,
    graph_state: &str,
) -> Option<Vec<ExtractedFact>> {
    let transcript = TURN_EXTRACTION_PROMPT
        .replace("{rolling_facts}", rolling_facts.unwrap_or("(none)"))
        .replace(
            "{graph_state}",
            if graph_state.is_empty() {
                "(none)"
            } else {
                graph_state
            },
        )
        .replace("{messages}", messages_text);
    extract_turn_facts(llm, &transcript).await
}

/// Build graph context string showing current entity states for known entities.
///
/// For each known entity, queries the graph for current state edges (max valid_from per category)
/// and formats them as contextual information for the LLM.
pub fn build_graph_context_for_turn(graph: &Graph, known_entities: &[String]) -> String {
    if known_entities.is_empty() {
        return String::new();
    }

    let mut lines = Vec::new();

    for entity_name in known_entities {
        let facts = graph_projection::collect_entity_facts(graph, entity_name);
        if facts.is_empty() {
            continue;
        }

        // Group by category prefix (state:location, preference:food, etc.)
        let mut by_category: HashMap<String, Vec<&graph_projection::EntityFact>> = HashMap::new();
        for fact in &facts {
            by_category
                .entry(fact.association_type.clone())
                .or_default()
                .push(fact);
        }

        let mut entity_parts = Vec::new();
        for (assoc_type, cat_facts) in &by_category {
            // Find the current one (highest valid_from)
            if let Some(current) = cat_facts
                .iter()
                .filter(|f| f.is_current)
                .max_by_key(|f| f.valid_from.unwrap_or(0))
            {
                entity_parts.push(format!("{}={}", assoc_type, current.target_name));
            }
        }

        if !entity_parts.is_empty() {
            lines.push(format!("{}: {}", entity_name, entity_parts.join(", ")));
        }
    }

    lines.join("\n")
}

/// Format a single turn's transcript with rolling context and graph state for per-turn extraction.
pub fn format_turn_transcript(
    turn: &ConversationTurn,
    rolling_facts: Option<&str>,
    graph_state: &str,
) -> String {
    let mut messages_text = String::new();
    for msg in &turn.messages {
        messages_text.push_str(&msg.role);
        messages_text.push_str(": ");
        messages_text.push_str(&msg.content);
        messages_text.push('\n');
    }

    TURN_EXTRACTION_PROMPT
        .replace("{rolling_facts}", rolling_facts.unwrap_or("(none)"))
        .replace(
            "{graph_state}",
            if graph_state.is_empty() {
                "(none)"
            } else {
                graph_state
            },
        )
        .replace("{messages}", &messages_text)
}

/// Extract facts from a single turn using the per-turn prompt.
///
/// Returns only the facts portion (no goals or procedural summary).
async fn extract_turn_facts(llm: &dyn LlmClient, transcript: &str) -> Option<Vec<ExtractedFact>> {
    if transcript.is_empty() {
        return None;
    }

    let request = LlmRequest {
        system_prompt: String::new(),
        user_prompt: transcript.to_string(),
        temperature: 0.0,
        max_tokens: 1024,
        json_mode: true,
    };

    let response = match llm.complete(request).await {
        Ok(r) => r,
        Err(e) => {
            tracing::warn!("extract_turn_facts LLM call failed: {}", e);
            return None;
        },
    };
    let value = match parse_json_from_llm(&response.content) {
        Some(v) => v,
        None => {
            tracing::warn!(
                "extract_turn_facts JSON parse failed, raw response: {}",
                safe_truncate(&response.content, 300)
            );
            return None;
        },
    };

    // Parse as a partial CompactionResponse (only facts field needed)
    // Use a lenient struct that accepts flexible types from the LLM
    #[derive(Deserialize)]
    struct LenientFact {
        #[serde(default)]
        statement: String,
        #[serde(default)]
        subject: String,
        #[serde(default)]
        predicate: String,
        #[serde(default)]
        object: String,
        #[serde(default, deserialize_with = "deserialize_f32_lenient")]
        confidence: f32,
        #[serde(default)]
        category: Option<String>,
        #[serde(default)]
        amount: Option<f64>,
        #[serde(default)]
        split_with: Option<Vec<String>>,
        #[serde(default)]
        temporal_signal: Option<String>,
        #[serde(default)]
        depends_on: Option<String>,
        #[serde(default, deserialize_with = "deserialize_bool_lenient")]
        is_update: Option<bool>,
        #[serde(default)]
        cardinality_hint: Option<String>,
        #[serde(default, deserialize_with = "deserialize_f32_option_lenient")]
        sentiment: Option<f32>,
    }

    #[derive(Deserialize)]
    struct TurnResponse {
        #[serde(default)]
        facts: Vec<LenientFact>,
    }

    match serde_json::from_value::<TurnResponse>(value.clone()) {
        Ok(parsed) => {
            let facts: Vec<ExtractedFact> = parsed
                .facts
                .into_iter()
                .filter(|f| !f.statement.is_empty() || !f.subject.is_empty())
                .map(|f| ExtractedFact {
                    statement: f.statement,
                    subject: f.subject,
                    predicate: f.predicate,
                    object: f.object,
                    confidence: if f.confidence > 0.0 {
                        f.confidence
                    } else {
                        0.8
                    },
                    category: f.category,
                    amount: f.amount,
                    split_with: f.split_with,
                    temporal_signal: f.temporal_signal,
                    depends_on: f.depends_on,
                    is_update: f.is_update,
                    cardinality_hint: f.cardinality_hint,
                    sentiment: f.sentiment,
                    group_id: Default::default(),
                    ingest_metadata: Default::default(),
                })
                .collect();
            Some(facts)
        },
        Err(e) => {
            let json_str = value.to_string();
            tracing::warn!(
                "extract_turn_facts deserialization failed: {}. JSON: {}",
                e,
                safe_truncate(&json_str, 500)
            );
            None
        },
    }
}

/// Deserialize f32 leniently: accept numbers, strings ("0.9"), or default to 0.8
fn deserialize_f32_lenient<'de, D>(deserializer: D) -> Result<f32, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let val: serde_json::Value = serde::Deserialize::deserialize(deserializer)?;
    match &val {
        serde_json::Value::Number(n) => Ok(n.as_f64().unwrap_or(0.8) as f32),
        serde_json::Value::String(s) => Ok(s.parse::<f32>().unwrap_or(0.8)),
        serde_json::Value::Null => Ok(0.8),
        _ => Ok(0.8),
    }
}

/// Deserialize Option<bool> leniently: accept booleans, strings ("true"/"false"), or null
fn deserialize_bool_lenient<'de, D>(deserializer: D) -> Result<Option<bool>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let val: serde_json::Value = serde::Deserialize::deserialize(deserializer)?;
    match &val {
        serde_json::Value::Bool(b) => Ok(Some(*b)),
        serde_json::Value::String(s) => match s.to_lowercase().as_str() {
            "true" | "yes" | "1" => Ok(Some(true)),
            "false" | "no" | "0" => Ok(Some(false)),
            _ => Ok(None),
        },
        serde_json::Value::Null => Ok(None),
        _ => Ok(None),
    }
}

/// Deserialize Option<f32> leniently: accept numbers, strings ("0.9"), or null.
/// Used for sentiment values where the LLM may return a string instead of a number.
fn deserialize_f32_option_lenient<'de, D>(deserializer: D) -> Result<Option<f32>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let val: serde_json::Value = serde::Deserialize::deserialize(deserializer)?;
    match &val {
        serde_json::Value::Number(n) => Ok(Some(n.as_f64().unwrap_or(0.5) as f32)),
        serde_json::Value::String(s) => Ok(s.parse::<f32>().ok()),
        serde_json::Value::Null => Ok(None),
        _ => Ok(None),
    }
}

/// Format a `ConversationIngest` into a text transcript for the LLM.
pub fn format_transcript(data: &ConversationIngest) -> String {
    let mut buf = String::new();
    for session in &data.sessions {
        for msg in &session.messages {
            buf.push_str(&msg.role);
            buf.push_str(": ");
            buf.push_str(&msg.content);
            buf.push('\n');
        }
    }

    if buf.len() > MAX_TRANSCRIPT_CHARS {
        let start = buf.len() - MAX_TRANSCRIPT_CHARS;
        // Find the next newline after the cut point to avoid splitting a line
        let adjusted = buf[start..]
            .find('\n')
            .map(|p| start + p + 1)
            .unwrap_or(start);
        buf = buf[adjusted..].to_string();
    }

    buf
}

// ────────── LLM Extraction ──────────

/// Call the LLM to extract facts, goals, and procedural summary from a transcript.
///
/// Returns `None` on any failure (fail-open).
pub async fn extract_compaction(
    llm: &dyn LlmClient,
    data: &ConversationIngest,
    category_block: &str,
    category_enum: &str,
) -> Option<CompactionResponse> {
    let transcript = format_transcript(data);
    extract_compaction_from_transcript(llm, &transcript, category_block, category_enum).await
}

/// Extract compaction from a pre-formatted transcript string.
///
/// Returns `None` on any failure (fail-open).
pub async fn extract_compaction_from_transcript(
    llm: &dyn LlmClient,
    transcript: &str,
    category_block: &str,
    category_enum: &str,
) -> Option<CompactionResponse> {
    if transcript.is_empty() {
        return None;
    }

    let request = LlmRequest {
        system_prompt: compaction_system_prompt(category_block, category_enum),
        user_prompt: transcript.to_string(),
        temperature: 0.0,
        max_tokens: 2048,
        json_mode: true,
    };

    let response = llm.complete(request).await.ok()?;
    let value = parse_json_from_llm(&response.content)?;
    serde_json::from_value::<CompactionResponse>(value).ok()
}

// ────────── Playbook Extraction ──────────

const PLAYBOOK_SYSTEM_PROMPT: &str = r#"You are a retrospective analysis system. Given a conversation transcript and goals, extract a playbook for each goal:

1. "what_worked": Actions/approaches that succeeded
2. "what_didnt_work": Actions/approaches that failed or were abandoned
3. "lessons_learned": Key takeaways for future attempts
4. "steps_taken": Brief ordered list of steps actually taken
5. "confidence": 0.0-1.0

If prior playbook experience is provided, use it to compare approaches and note what was
done differently this time. Reference prior lessons when relevant.

Output: { "playbooks": [ { "goal_description", "what_worked", "what_didnt_work", "lessons_learned", "steps_taken", "confidence" } ] }
Rules: One playbook per goal. Empty arrays if goal was barely discussed. Be specific, not generic. Output ONLY valid JSON"#;

/// Call the LLM to extract retrospective playbooks for each goal.
///
/// Returns `None` on any failure (fail-open).
pub async fn extract_playbooks(
    llm: &dyn LlmClient,
    transcript: &str,
    goals: &[ExtractedGoal],
) -> Option<PlaybookExtractionResponse> {
    if goals.is_empty() || transcript.is_empty() {
        return None;
    }

    // Reject trivially short transcripts
    if transcript.len() < MIN_PLAYBOOK_TRANSCRIPT_LEN {
        tracing::info!(
            "Playbook extraction skipped: transcript too short ({} < {} chars)",
            transcript.len(),
            MIN_PLAYBOOK_TRANSCRIPT_LEN
        );
        return None;
    }

    let mut goal_list = String::new();
    for (i, goal) in goals.iter().enumerate() {
        goal_list.push_str(&format!("{}. {}\n", i + 1, goal.description));
    }

    let user_prompt = format!(
        "Transcript:\n{}\n\nGoals to analyze:\n{}",
        transcript, goal_list
    );

    let request = LlmRequest {
        system_prompt: PLAYBOOK_SYSTEM_PROMPT.to_string(),
        user_prompt,
        temperature: 0.0,
        max_tokens: 2048,
        json_mode: true,
    };

    let response = llm.complete(request).await.ok()?;
    let value = parse_json_from_llm(&response.content)?;
    serde_json::from_value::<PlaybookExtractionResponse>(value).ok()
}

/// Attach extracted playbooks to GoalStore entries and graph node properties.
async fn attach_playbooks(
    engine: &crate::integration::GraphEngine,
    playbooks: &[GoalPlaybook],
    _case_id: &str,
) {
    for playbook in playbooks {
        // 1. Find matching goal in GoalStore via BM25
        let goal_store = engine.goal_store.read().await;
        let similar = goal_store.find_similar(&playbook.goal_description, 1);
        let goal_id = match similar.first() {
            Some((id, _score)) => *id,
            None => continue,
        };
        drop(goal_store);

        // 2. Attach playbook to GoalEntry
        let mut goal_store = engine.goal_store.write().await;
        goal_store.attach_playbook(goal_id, playbook.clone());
        drop(goal_store);

        // 3. Set properties on the Goal graph node
        let mut inference = engine.inference.write().await;
        let graph = inference.graph_mut();
        if let Some(&node_id) = graph.goal_index.get(&goal_id) {
            if let Some(node) = graph.get_node_mut(node_id) {
                node.properties.insert(
                    "playbook_what_worked".to_string(),
                    serde_json::json!(playbook.what_worked),
                );
                node.properties.insert(
                    "playbook_what_didnt_work".to_string(),
                    serde_json::json!(playbook.what_didnt_work),
                );
                node.properties.insert(
                    "playbook_lessons_learned".to_string(),
                    serde_json::json!(playbook.lessons_learned),
                );
                node.properties.insert(
                    "playbook_steps_taken".to_string(),
                    serde_json::json!(playbook.steps_taken),
                );
                node.properties.insert(
                    "playbook_confidence".to_string(),
                    serde_json::json!(playbook.confidence),
                );
            }
        }
    }
}

// ────────── Event Conversion ──────────

/// Convert a `CompactionResponse` into pipeline-ready `Event` structs.
pub fn compaction_to_events(
    response: &CompactionResponse,
    case_id: &str,
    agent_id: u64,
    session_id: u64,
    base_ts: u64,
) -> Vec<agent_db_events::Event> {
    let mut events = Vec::new();
    let mut ts_offset: u64 = 0;

    let next_ts = |offset: &mut u64| -> u64 {
        let ts = base_ts + *offset * 1_000;
        *offset += 1;
        ts
    };

    // Facts → Events (state-aware: predicate determines edge type)
    for fact in &response.facts {
        let mut metadata = HashMap::new();
        metadata.insert(
            "compaction_fact".to_string(),
            MetadataValue::String("true".to_string()),
        );
        metadata.insert(
            "case_id".to_string(),
            MetadataValue::String(case_id.to_string()),
        );

        // Include entity + new_state + predicate so the pipeline creates
        // graph edges directly from LLM-extracted facts.
        metadata.insert(
            "entity".to_string(),
            MetadataValue::String(fact.subject.clone()),
        );
        metadata.insert(
            "new_state".to_string(),
            MetadataValue::String(fact.object.clone()),
        );
        metadata.insert(
            "attribute".to_string(),
            MetadataValue::String(fact.predicate.clone()),
        );
        metadata.insert(
            "entity_state".to_string(),
            MetadataValue::String("true".to_string()),
        );
        // Category for semantic supersession grouping
        if let Some(cat) = &fact.category {
            metadata.insert("category".to_string(), MetadataValue::String(cat.clone()));
        }
        // Conditional dependency — this fact is only valid while the condition holds
        if let Some(dep) = &fact.depends_on {
            metadata.insert("depends_on".to_string(), MetadataValue::String(dep.clone()));
        }
        // Explicit state update marker
        if fact.is_update == Some(true) {
            metadata.insert(
                "is_update".to_string(),
                MetadataValue::String("true".to_string()),
            );
        }

        let evt = agent_db_events::Event {
            id: agent_db_core::types::generate_event_id(),
            timestamp: next_ts(&mut ts_offset),
            agent_id,
            agent_type: "conversation_compaction".to_string(),
            session_id,
            event_type: EventType::Observation {
                observation_type: "extracted_fact".to_string(),
                data: serde_json::json!({
                    "statement": fact.statement,
                    "subject": fact.subject,
                    "predicate": fact.predicate,
                    "object": fact.object,
                }),
                confidence: fact.confidence,
                source: "conversation_compaction".to_string(),
            },
            causality_chain: Vec::new(),
            context: EventContext::default(),
            metadata,
            context_size_bytes: 0,
            segment_pointer: None,
            is_code: false,
        };
        events.push(evt);
    }

    // Goals → Cognitive/GoalFormation events
    for goal in &response.goals {
        let mut metadata = HashMap::new();
        metadata.insert(
            "compaction_goal".to_string(),
            MetadataValue::String("true".to_string()),
        );
        metadata.insert(
            "case_id".to_string(),
            MetadataValue::String(case_id.to_string()),
        );

        let evt = agent_db_events::Event {
            id: agent_db_core::types::generate_event_id(),
            timestamp: next_ts(&mut ts_offset),
            agent_id,
            agent_type: "conversation_compaction".to_string(),
            session_id,
            event_type: EventType::Cognitive {
                process_type: CognitiveType::GoalFormation,
                input: serde_json::json!({
                    "description": goal.description,
                    "owner": goal.owner,
                }),
                output: serde_json::json!({
                    "status": goal.status,
                }),
                reasoning_trace: vec!["LLM conversation compaction".to_string()],
            },
            causality_chain: Vec::new(),
            context: EventContext::default(),
            metadata,
            context_size_bytes: 0,
            segment_pointer: None,
            is_code: false,
        };
        events.push(evt);
    }

    // Procedural steps → Action events
    if let Some(ref summary) = response.procedural_summary {
        for step in &summary.steps {
            let mut metadata = HashMap::new();
            metadata.insert(
                "compaction_step".to_string(),
                MetadataValue::String("true".to_string()),
            );
            metadata.insert(
                "case_id".to_string(),
                MetadataValue::String(case_id.to_string()),
            );

            let outcome = map_step_outcome(&step.outcome);

            let evt = agent_db_events::Event {
                id: agent_db_core::types::generate_event_id(),
                timestamp: next_ts(&mut ts_offset),
                agent_id,
                agent_type: "conversation_compaction".to_string(),
                session_id,
                event_type: EventType::Action {
                    action_name: format!("step_{}", step.step_number),
                    parameters: serde_json::json!({
                        "action": step.action,
                        "result": step.result,
                    }),
                    outcome,
                    duration_ns: 0,
                },
                causality_chain: Vec::new(),
                context: EventContext::default(),
                metadata,
                context_size_bytes: 0,
                segment_pointer: None,
                is_code: false,
            };
            events.push(evt);
        }
    }

    events
}

/// Map a step outcome string to an `ActionOutcome`.
fn map_step_outcome(outcome: &str) -> ActionOutcome {
    match outcome {
        "success" => ActionOutcome::Success {
            result: serde_json::json!({"status": "success"}),
        },
        "failure" => ActionOutcome::Failure {
            error: "step failed".to_string(),
            error_code: 1,
        },
        "partial" => ActionOutcome::Partial {
            result: serde_json::json!({"status": "partial"}),
            issues: vec!["partial completion".to_string()],
        },
        _ => ActionOutcome::Success {
            result: serde_json::json!({"status": "pending"}),
        },
    }
}

// ────────── Procedural Memory ──────────

/// Build a procedural `Memory` from a `ProceduralSummary`.
pub fn build_procedural_memory(
    summary: &ProceduralSummary,
    agent_id: u64,
    session_id: u64,
    episode_id: u64,
) -> Memory {
    let outcome = map_progress_to_outcome(&summary.progress_status);

    let mut metadata = HashMap::new();
    metadata.insert("source".to_string(), "compaction".to_string());
    metadata.insert("objective".to_string(), summary.objective.clone());
    metadata.insert(
        "progress_status".to_string(),
        summary.progress_status.clone(),
    );

    Memory {
        id: 0, // Will be assigned by the store
        agent_id,
        session_id,
        episode_id,
        summary: summary.overall_summary.clone(),
        takeaway: summary.takeaway.clone(),
        causal_note: format!(
            "Objective: {}. Status: {}",
            summary.objective, summary.progress_status
        ),
        summary_embedding: Vec::new(),
        tier: MemoryTier::Episodic,
        consolidated_from: Vec::new(),
        schema_id: None,
        consolidation_status: crate::memory::ConsolidationStatus::Active,
        context: EventContext::default(),
        key_events: Vec::new(),
        strength: 0.8,
        relevance_score: 0.8,
        formed_at: agent_db_core::types::current_timestamp(),
        last_accessed: agent_db_core::types::current_timestamp(),
        access_count: 0,
        outcome,
        memory_type: MemoryType::Episodic { significance: 0.8 },
        metadata,
        expires_at: None,
    }
}

/// Map a progress_status string to an `EpisodeOutcome`.
pub fn map_progress_to_outcome(status: &str) -> EpisodeOutcome {
    match status {
        "completed" => EpisodeOutcome::Success,
        "blocked" | "abandoned" => EpisodeOutcome::Failure,
        "in_progress" => EpisodeOutcome::Partial,
        _ => EpisodeOutcome::Partial,
    }
}

// ────────── Goal Dedup Helpers ──────────

/// Filter goals by their classification operations.
///
/// Returns `(approved_goals, dedup_count)` where approved goals are those
/// classified as ADD or UPDATE. Goals classified as DELETE or NONE are filtered out.
pub fn filter_goals_by_classification(
    goals: &[ExtractedGoal],
    goal_ops: &[ClassifiedOperation],
) -> (Vec<ExtractedGoal>, usize) {
    let mut approved = Vec::new();
    let mut dedup_count = 0usize;

    for (i, goal) in goals.iter().enumerate() {
        let action = goal_ops
            .get(i)
            .map(|op| op.action)
            .unwrap_or(MemoryAction::Add); // fallback: keep

        match action {
            MemoryAction::Add | MemoryAction::Update => {
                approved.push(goal.clone());
            },
            MemoryAction::Delete | MemoryAction::None => {
                dedup_count += 1;
            },
        }
    }

    (approved, dedup_count)
}

// ────────── Procedural Memory Helpers ──────────

/// Handle procedural memory with a classification op.
#[allow(clippy::too_many_arguments)]
async fn handle_procedural_memory(
    engine: &crate::integration::GraphEngine,
    summary: &ProceduralSummary,
    proc_op: &ClassifiedOperation,
    similar_refs: &[&Memory],
    case_id: &str,
    agent_id: u64,
    session_id: u64,
    result: &mut CompactionResult,
) {
    let episode_id = procedural_episode_id(case_id);

    match proc_op.action {
        MemoryAction::Add => {
            store_new_procedural_memory(
                engine, summary, case_id, agent_id, session_id, episode_id, result,
            )
            .await;
        },
        MemoryAction::Update => {
            if let Some(target_id) = resolve_target(proc_op, similar_refs) {
                let store = engine.memory_store.read().await;
                let existing = store.get_memory(target_id);
                drop(store);

                if let Some(existing_mem) = existing {
                    let old_summary = existing_mem.summary.clone();
                    let old_takeaway = existing_mem.takeaway.clone();

                    let new_summary_text = proc_op
                        .new_text
                        .as_deref()
                        .unwrap_or(&summary.overall_summary);
                    let mut updated = existing_mem;
                    updated.summary = new_summary_text.to_string();
                    updated.takeaway = summary.takeaway.clone();
                    updated.last_accessed = agent_db_core::types::current_timestamp();

                    let text = format!(
                        "{} {} {}",
                        updated.summary, updated.takeaway, updated.causal_note
                    );

                    let mut store = engine.memory_store.write().await;
                    store.store_consolidated_memory(updated);
                    drop(store);

                    let mut bm25 = engine.memory_bm25_index.write().await;
                    bm25.index_document(target_id, &text);
                    drop(bm25);

                    result.memories_updated += 1;

                    let mut audit = engine.memory_audit_log.write().await;
                    audit.record_update(
                        target_id,
                        &old_summary,
                        new_summary_text,
                        &old_takeaway,
                        &summary.takeaway,
                        MutationActor::LlmClassifier,
                        Some(format!("Compaction UPDATE for case {}", case_id)),
                    );
                }
            }
        },
        MemoryAction::Delete => {
            if let Some(target_id) = resolve_target(proc_op, similar_refs) {
                let store = engine.memory_store.read().await;
                let existing = store.get_memory(target_id);
                drop(store);

                if let Some(existing_mem) = existing {
                    let old_summary = existing_mem.summary.clone();
                    let old_takeaway = existing_mem.takeaway.clone();

                    let mut store = engine.memory_store.write().await;
                    store.delete_memories_batch(vec![target_id]);
                    drop(store);

                    result.memories_deleted += 1;

                    let mut audit = engine.memory_audit_log.write().await;
                    audit.record_delete(
                        target_id,
                        &old_summary,
                        &old_takeaway,
                        MutationActor::LlmClassifier,
                        Some(format!("Compaction DELETE for case {}", case_id)),
                    );
                }
            }
        },
        MemoryAction::None => {
            // Skip — already captured
        },
    }
}

/// Fallback: unconditionally create a new procedural memory (fail-open).
async fn handle_procedural_memory_fallback(
    engine: &crate::integration::GraphEngine,
    summary: &ProceduralSummary,
    case_id: &str,
    agent_id: u64,
    session_id: u64,
    result: &mut CompactionResult,
) {
    let episode_id = procedural_episode_id(case_id);
    store_new_procedural_memory(
        engine, summary, case_id, agent_id, session_id, episode_id, result,
    )
    .await;

    // Record audit trail as fallback
    let actual_id = procedural_memory_id(case_id);
    let mut audit = engine.memory_audit_log.write().await;
    audit.record_add(
        actual_id,
        &summary.overall_summary,
        &summary.takeaway,
        MutationActor::ConversationBridge,
        Some(format!(
            "Compaction procedural memory for case {} (classifier fallback)",
            case_id
        )),
    );
}

/// Store a new procedural memory and record its audit trail.
async fn store_new_procedural_memory(
    engine: &crate::integration::GraphEngine,
    summary: &ProceduralSummary,
    case_id: &str,
    agent_id: u64,
    session_id: u64,
    episode_id: u64,
    result: &mut CompactionResult,
) {
    let memory = build_procedural_memory(summary, agent_id, session_id, episode_id);
    let actual_id = procedural_memory_id(case_id);

    let mut mem_with_id = memory;
    mem_with_id.id = actual_id;

    let text = format!(
        "{} {} {}",
        mem_with_id.summary, mem_with_id.takeaway, mem_with_id.causal_note
    );

    let mut store = engine.memory_store.write().await;
    store.store_consolidated_memory(mem_with_id);
    drop(store);

    let mut bm25 = engine.memory_bm25_index.write().await;
    bm25.index_document(actual_id, &text);
    drop(bm25);

    result.procedural_memory_created = true;
    result.procedural_memory_id = Some(actual_id);

    let mut audit = engine.memory_audit_log.write().await;
    audit.record_add(
        actual_id,
        &summary.overall_summary,
        &summary.takeaway,
        MutationActor::LlmClassifier,
        Some(format!("Compaction ADD for case {}", case_id)),
    );
}

/// Compute a deterministic episode ID for procedural memory.
fn procedural_episode_id(case_id: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    format!("{}:procedural", case_id).hash(&mut hasher);
    hasher.finish()
}

/// Compute a deterministic memory ID for procedural memory.
fn procedural_memory_id(case_id: &str) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    format!("{}:procedural_memory", case_id).hash(&mut hasher);
    hasher.finish()
}

// ────────── Top-Level Entry Point ──────────

/// Run LLM-driven compaction on a conversation ingest.
///
/// This is the top-level entry point called after the rule-based pipeline.
/// It extracts facts, goals, and a procedural summary via a single LLM call,
/// then feeds the results back through the event pipeline and stores a
/// procedural memory.
///
/// Fail-open: returns an empty `CompactionResult` on any failure.
/// Run compaction without prior context (used by batch ingest).
pub async fn run_compaction(
    engine: &crate::integration::GraphEngine,
    data: &ConversationIngest,
    case_id: &str,
) -> CompactionResult {
    run_compaction_with_context(engine, data, case_id, None).await
}

/// Run compaction with optional prior context from a rolling summary.
/// When `prior_summary` is provided, it seeds the rolling facts so the LLM
/// can resolve coreferences (pronouns, "the restaurant", etc.) across buffered
/// single-message batches.
pub async fn run_compaction_with_context(
    engine: &crate::integration::GraphEngine,
    data: &ConversationIngest,
    case_id: &str,
    prior_summary: Option<&str>,
) -> CompactionResult {
    let mut result = CompactionResult::default();

    // Need an LLM client
    let llm = match engine.unified_llm_client() {
        Some(client) => Arc::clone(client),
        None => return result,
    };

    // Derive agent_id and session_id using the same formula as ingest_to_events
    let agent_id: u64 = {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        case_id.hash(&mut hasher);
        hasher.finish() | 0x8000_0000_0000_0000
    };

    let session_id: u64 = {
        use std::hash::{Hash, Hasher};
        let mut hasher = std::collections::hash_map::DefaultHasher::new();
        format!("{}:compaction", case_id).hash(&mut hasher);
        hasher.finish()
    };

    let base_ts: u64 = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_nanos() as u64;

    // ── Phase 1: Per-turn extraction with rolling context + graph state ──
    // Batch adjacent turns in pairs (2 user+assistant exchanges per LLM call)
    // to reduce API calls while maintaining temporal ordering between batches.
    let turns = split_into_turns(data);
    let mut rolling_facts: Vec<String> = match prior_summary {
        Some(s) if !s.is_empty() => vec![format!("[Prior conversation context]\n{}", s)],
        _ => Vec::new(),
    };
    let mut known_entities: HashSet<String> = HashSet::new();
    let mut all_extracted_facts: Vec<ExtractedFact> = Vec::new();

    // Batch turns in pairs: [0,1], [2,3], [4,5], ...
    let batches: Vec<Vec<&ConversationTurn>> = turns
        .iter()
        .collect::<Vec<_>>()
        .chunks(2)
        .map(|c| c.to_vec())
        .collect();

    tracing::info!(
        "COMPACTION per-turn extraction: {} turns in {} batches for case_id={}",
        turns.len(),
        batches.len(),
        case_id
    );

    for batch in &batches {
        let batch_start_idx = batch[0].turn_index;

        // Query graph for current entity states
        let known_entities_vec: Vec<String> = known_entities.iter().cloned().collect();
        let graph_ctx = {
            let inf = engine.inference.read().await;
            build_graph_context_for_turn(inf.graph(), &known_entities_vec)
        };

        let rolling_ctx = if rolling_facts.is_empty() {
            None
        } else {
            Some(rolling_facts.join("\n"))
        };

        // Merge batch turns into a single transcript
        let mut merged_turn = ConversationTurn {
            messages: Vec::new(),
            session_index: batch[0].session_index,
            turn_index: batch_start_idx,
        };
        for turn in batch {
            merged_turn.messages.extend(turn.messages.clone());
        }

        let messages_text = format_messages(&merged_turn);
        let cat_block = engine.domain_registry.prompt_category_block();
        let cat_enum = engine.domain_registry.prompt_category_enum();

        // Extract facts: LLM cascade + LLM financial extraction in parallel
        let (turn_facts, ner_financial_facts) = tokio::join!(
            extract_turn_facts_cascade(
                llm.as_ref(),
                &messages_text,
                rolling_ctx.as_deref(),
                &graph_ctx,
                &known_entities_vec,
                &cat_block,
                &cat_enum,
            ),
            extract_financial_facts_llm(llm.as_ref(), &messages_text)
        );

        let batch_label = match batch.last() {
            Some(last) if batch.len() > 1 => {
                format!("batch {}-{}", batch_start_idx, last.turn_index)
            },
            _ => format!("turn {}", batch_start_idx),
        };

        match &turn_facts {
            None => {
                tracing::warn!(
                    "COMPACTION {} returned None (extraction failed)",
                    batch_label
                );
            },
            Some(f) if f.is_empty() => {
                tracing::info!(
                    "COMPACTION {} extracted 0 facts (empty response)",
                    batch_label
                );
            },
            _ => {},
        }

        // Merge LLM facts with NER financial facts, deduplicating by
        // normalized (subject, object, category=financial).
        let mut facts_combined = turn_facts.unwrap_or_default();
        if !ner_financial_facts.is_empty() {
            let existing_financial: HashSet<(String, String)> = facts_combined
                .iter()
                .filter(|f| {
                    f.category
                        .as_deref()
                        .is_some_and(|c| engine.ontology.is_append_only(c))
                })
                .map(|f| (f.subject.to_lowercase(), f.object.to_lowercase()))
                .collect();

            for nf in ner_financial_facts {
                let key = (nf.subject.to_lowercase(), nf.object.to_lowercase());
                if !existing_financial.contains(&key) {
                    facts_combined.push(nf);
                }
            }
        }

        if !facts_combined.is_empty() {
            let mut facts = facts_combined;
            let turn_base = base_ts + batch_start_idx as u64 * TURN_GAP;

            // Inject group_id and request-level metadata into each fact so they flow to graph edges.
            for fact in facts.iter_mut() {
                if !data.group_id.is_empty() {
                    fact.group_id = data.group_id.clone();
                }
                if !data.metadata.is_empty() {
                    fact.ingest_metadata = data.metadata.clone();
                }
            }

            tracing::info!(
                "COMPACTION {} extracted {} facts, writing directly to graph",
                batch_label,
                facts.len(),
            );

            // ── Two-phase write: single-valued first, then multi-valued ──
            // Single-valued facts (location, work, education) establish entity
            // state. Writing them first ensures that:
            //   1. The depends_on stamp on multi-valued facts sees the correct
            //      current location (not the previous one or nothing)
            //   2. The supersession cascade triggers correctly when a new
            //      location replaces an old one, catching old depends_on edges

            // Phase A: write single-valued facts (they establish state)
            let mut sv_offset = 0u64;
            for fact in facts.iter() {
                let cat = fact.category.as_deref().unwrap_or("");
                if engine.ontology.is_single_valued(cat) {
                    let fact_ts = turn_base + sv_offset * 1_000;
                    engine.write_fact_to_graph(fact, fact_ts).await;
                    engine.detect_edge_conflicts(fact, fact_ts).await;
                    sv_offset += 1;
                }
            }

            // Phase B: stamp depends_on on multi-valued facts using
            // the NOW-UPDATED graph state (after single-valued writes)
            {
                let inf = engine.inference.read().await;
                let graph = inf.graph();
                for fact in facts.iter_mut() {
                    // Skip if already set by the LLM
                    if fact.depends_on.is_some() {
                        continue;
                    }
                    // Skip single-valued categories — they ARE the state
                    let cat = match &fact.category {
                        Some(c) => c.as_str(),
                        None => continue,
                    };
                    if engine.ontology.is_single_valued(cat) {
                        continue;
                    }
                    // For multi-valued facts: check if the subject has a current location.
                    // Normalize the subject name (lowercase, strip articles) to match
                    // the concept_index key, which uses the same normalization.
                    let subject_normalized = fact.subject.to_lowercase();
                    let subject_normalized = subject_normalized.trim();
                    let projected = graph_projection::project_entity_state(
                        graph,
                        subject_normalized,
                        u64::MAX,
                        Some(engine.ontology.as_ref()),
                    );
                    // Find the first single-valued cascade-triggering slot (e.g., location)
                    // to stamp as depends_on context. Ontology-driven — no hardcoded category.
                    let cascade_slot = projected.slots.values().find(|s| {
                        let cat = s.association_type.split(':').next().unwrap_or("");
                        engine.ontology.triggers_cascade(cat)
                    });
                    if let Some(loc_slot) = cascade_slot {
                        let loc_val = loc_slot.value.as_deref().unwrap_or(&loc_slot.target_name);
                        let _cat = loc_slot
                            .association_type
                            .split(':')
                            .next()
                            .unwrap_or("state");
                        let pred = loc_slot.association_type.split(':').nth(1).unwrap_or("in");
                        let dep = format!("{} {} {}", fact.subject, pred, loc_val);
                        tracing::debug!(
                            "COMPACTION: auto-stamped depends_on='{}' on fact '{}'",
                            dep,
                            fact.statement,
                        );
                        fact.depends_on = Some(dep);
                    }
                }
            }

            // Phase C: write multi-valued facts (with correct depends_on)
            let mut mv_offset = sv_offset;
            for fact in facts.iter() {
                let cat = fact.category.as_deref().unwrap_or("");
                if !engine.ontology.is_single_valued(cat) {
                    let fact_ts = turn_base + mv_offset * 1_000;
                    engine.write_fact_to_graph(fact, fact_ts).await;
                    engine.detect_edge_conflicts(fact, fact_ts).await;
                    mv_offset += 1;
                }
            }

            // Update rolling context
            for fact in &facts {
                rolling_facts.push(fact.statement.clone());
                known_entities.insert(fact.subject.clone());
                known_entities.insert(fact.object.clone());
            }
            const MAX_ROLLING_FACTS: usize = 30;
            if rolling_facts.len() > MAX_ROLLING_FACTS {
                rolling_facts.drain(..rolling_facts.len() - MAX_ROLLING_FACTS);
            }

            result.facts_extracted += facts.len();
            all_extracted_facts.extend(facts);
        }
    }

    result.llm_success = true;

    // ── Run community detection so DRIFT search has data ──
    if engine.config.enable_louvain && !all_extracted_facts.is_empty() {
        tracing::info!("COMPACTION: triggering community detection after per-turn extraction");
        if let Err(e) = engine.run_community_detection().await {
            tracing::warn!("COMPACTION: community detection failed: {}", e);
        }
    }

    // ── Embed concept nodes + create claims for extracted facts ──
    // Without this, node vector search and claim hybrid search find nothing
    // for compaction-created artifacts. This must complete BEFORE queries.
    if !all_extracted_facts.is_empty() {
        embed_nodes_and_create_claims(engine, &all_extracted_facts, base_ts).await;
    }

    // ── Active Retrieval Testing (ART): spawn background validation ──
    // Non-blocking — doesn't delay ingest response.
    if engine.config.art_config.enabled {
        if let (Some(ref llm_client), Some(ref embedding_client)) =
            (&engine.unified_llm_client, &engine.embedding_client)
        {
            // Collect recently embedded concept node IDs
            let candidate_node_ids: Vec<u64> = {
                let inf = engine.inference.read().await;
                let graph = inf.graph();
                graph
                    .concept_index
                    .values()
                    .copied()
                    .filter(|&nid| {
                        graph
                            .get_node(nid)
                            .map(|n| !n.embedding.is_empty())
                            .unwrap_or(false)
                    })
                    .collect()
            };

            if !candidate_node_ids.is_empty() {
                let inference = engine.inference.clone();
                let llm = llm_client.clone();
                let embedder = embedding_client.clone();
                let art_config = engine.config.art_config.clone();

                tokio::spawn(async move {
                    let result = crate::active_retrieval_test::run_art_pass(
                        candidate_node_ids,
                        &inference,
                        llm.as_ref(),
                        embedder.as_ref(),
                        &art_config,
                    )
                    .await;
                    tracing::info!(
                        "ART background pass: tested={}, enhanced={}, hits={}, misses={}",
                        result.nodes_tested,
                        result.nodes_enhanced,
                        result.total_hits,
                        result.total_misses,
                    );
                });
            }
        }
    }

    // ── Goals + procedural summary: ONE call at end using full transcript ──
    let full_transcript = format_transcript(data);

    // Enrich with community context if enabled
    let enriched_transcript = if engine.config.enable_context_enrichment {
        let summaries = engine.community_summaries.read().await;
        if summaries.is_empty() {
            full_transcript.clone()
        } else {
            let topic_slice = safe_truncate(&full_transcript, 500);
            let ctx = crate::context_enrichment::community_context_for_topic(
                topic_slice,
                &summaries,
                &engine.config.enrichment_config,
            );
            if ctx.is_empty() {
                full_transcript.clone()
            } else {
                format!("{}\n\n[Knowledge Context]\n{}", full_transcript, ctx)
            }
        }
    } else {
        full_transcript.clone()
    };

    // Extract goals + procedural summary from full transcript
    let goal_extraction = tokio::time::timeout(
        std::time::Duration::from_secs(30),
        extract_compaction_from_transcript(
            llm.as_ref(),
            &enriched_transcript,
            &engine.domain_registry.prompt_category_block(),
            &engine.domain_registry.prompt_category_enum(),
        ),
    )
    .await;

    if let Ok(Some(response)) = goal_extraction {
        result.goals_extracted = response.goals.len();
        result.procedural_steps_extracted = response
            .procedural_summary
            .as_ref()
            .map(|s| s.steps.len())
            .unwrap_or(0);

        // ── Fast goal dedup via GoalStore (no LLM needed) ──
        let mut fast_dedup_count = 0usize;
        let pre_filtered_goals: Vec<ExtractedGoal> = {
            let mut goal_store = engine.goal_store.write().await;
            let mut kept = Vec::new();
            for goal in &response.goals {
                match goal_store.store_or_dedup(
                    &goal.description,
                    &goal.status,
                    &goal.owner,
                    case_id,
                ) {
                    crate::goal_store::GoalDedupDecision::NewGoal => {
                        kept.push(goal.clone());
                    },
                    crate::goal_store::GoalDedupDecision::Duplicate { .. } => {
                        fast_dedup_count += 1;
                    },
                    crate::goal_store::GoalDedupDecision::StatusUpdate {
                        existing_id,
                        new_status,
                    } => {
                        goal_store.update_status(existing_id, new_status);
                        fast_dedup_count += 1;
                    },
                }
            }
            kept
        };
        result.goals_deduplicated += fast_dedup_count;

        let response = CompactionResponse {
            facts: vec![], // Facts already processed per-turn above
            goals: pre_filtered_goals,
            procedural_summary: response.procedural_summary,
        };

        let has_goals = !response.goals.is_empty();
        let has_summary = response.procedural_summary.is_some();

        if has_goals || has_summary {
            let goal_count = response.goals.len();
            let mut classifiable: Vec<String> = response
                .goals
                .iter()
                .map(|g| g.description.clone())
                .collect();
            if let Some(ref summary) = response.procedural_summary {
                classifiable.push(summary.overall_summary.clone());
            }
            let classifiable_refs: Vec<&str> = classifiable.iter().map(|s| s.as_str()).collect();

            let similar_memories = {
                let bm25 = engine.memory_bm25_index.read().await;
                let mut seen_ids = HashSet::new();
                let mut all_memories = Vec::new();

                for item in &classifiable_refs {
                    let hits = bm25.search(item, 10);
                    for (id, _score) in &hits {
                        seen_ids.insert(*id);
                    }
                }
                drop(bm25);

                let store = engine.memory_store.read().await;
                for id in &seen_ids {
                    if let Some(mem) = store.get_memory(*id) {
                        all_memories.push(mem);
                    }
                }
                all_memories
            };

            let similar_refs: Vec<&Memory> = similar_memories.iter().collect();

            let classifier_ctx = if engine.config.enable_context_enrichment {
                let summaries = engine.community_summaries.read().await;
                if summaries.is_empty() {
                    None
                } else {
                    let topic = classifiable.join(" ");
                    let ctx = crate::context_enrichment::community_context_for_topic(
                        safe_truncate(&topic, 500),
                        &summaries,
                        &engine.config.enrichment_config,
                    );
                    if ctx.is_empty() {
                        None
                    } else {
                        Some(ctx)
                    }
                }
            } else {
                None
            };

            let classification = classify_memory_updates(
                llm.as_ref(),
                &classifiable_refs,
                &similar_refs,
                classifier_ctx.as_deref(),
            )
            .await;

            match classification {
                Ok(class_result) => {
                    let goal_ops =
                        &class_result.operations[..goal_count.min(class_result.operations.len())];
                    let proc_ops = if class_result.operations.len() > goal_count {
                        &class_result.operations[goal_count..]
                    } else {
                        &[]
                    };

                    let (approved_goals, dedup_count) =
                        filter_goals_by_classification(&response.goals, goal_ops);
                    result.goals_deduplicated = dedup_count;

                    for (i, op) in goal_ops.iter().enumerate() {
                        if op.action == MemoryAction::Update {
                            if let Some(target_id) = resolve_target(op, &similar_refs) {
                                let store = engine.memory_store.read().await;
                                let existing = store.get_memory(target_id);
                                drop(store);

                                if let Some(existing_mem) = existing {
                                    let goal_desc = response
                                        .goals
                                        .get(i)
                                        .map(|g| g.description.as_str())
                                        .unwrap_or("");
                                    let mut audit = engine.memory_audit_log.write().await;
                                    audit.record_update(
                                        target_id,
                                        &existing_mem.summary,
                                        goal_desc,
                                        &existing_mem.takeaway,
                                        goal_desc,
                                        MutationActor::LlmClassifier,
                                        Some(format!(
                                            "Compaction goal UPDATE for case {}",
                                            case_id
                                        )),
                                    );
                                }
                            }
                        } else if op.action == MemoryAction::Delete {
                            if let Some(target_id) = resolve_target(op, &similar_refs) {
                                let store = engine.memory_store.read().await;
                                let existing = store.get_memory(target_id);
                                drop(store);

                                if let Some(existing_mem) = existing {
                                    let old_summary = existing_mem.summary.clone();
                                    let old_takeaway = existing_mem.takeaway.clone();

                                    let mut store = engine.memory_store.write().await;
                                    store.delete_memories_batch(vec![target_id]);
                                    drop(store);

                                    result.memories_deleted += 1;

                                    let mut audit = engine.memory_audit_log.write().await;
                                    audit.record_delete(
                                        target_id,
                                        &old_summary,
                                        &old_takeaway,
                                        MutationActor::LlmClassifier,
                                        Some(format!(
                                            "Compaction goal DELETE for case {}",
                                            case_id
                                        )),
                                    );
                                }
                            }
                        }
                    }

                    // Process goal events (no facts — those were already processed per-turn)
                    let filtered_response = CompactionResponse {
                        facts: vec![],
                        goals: approved_goals,
                        procedural_summary: response.procedural_summary.clone(),
                    };

                    let goal_base_ts = base_ts + (turns.len() as u64 + 1) * TURN_GAP;
                    let events = compaction_to_events(
                        &filtered_response,
                        case_id,
                        agent_id,
                        session_id,
                        goal_base_ts,
                    );
                    for event in events {
                        if let Err(e) = engine.process_event_with_options(event, Some(true)).await {
                            tracing::info!("COMPACTION goal event pipeline error: {}", e);
                        }
                    }

                    if let Some(ref summary) = response.procedural_summary {
                        if let Some(proc_op) = proc_ops.first() {
                            handle_procedural_memory(
                                engine,
                                summary,
                                proc_op,
                                &similar_refs,
                                case_id,
                                agent_id,
                                session_id,
                                &mut result,
                            )
                            .await;
                        } else {
                            handle_procedural_memory_fallback(
                                engine,
                                summary,
                                case_id,
                                agent_id,
                                session_id,
                                &mut result,
                            )
                            .await;
                        }
                    }
                },
                Err(_) => {
                    let events =
                        compaction_to_events(&response, case_id, agent_id, session_id, base_ts);
                    for event in events {
                        if let Err(e) = engine.process_event_with_options(event, Some(true)).await {
                            tracing::debug!("Compaction event pipeline error: {}", e);
                        }
                    }

                    if let Some(ref summary) = response.procedural_summary {
                        handle_procedural_memory_fallback(
                            engine,
                            summary,
                            case_id,
                            agent_id,
                            session_id,
                            &mut result,
                        )
                        .await;
                    }
                },
            }
        }

        // ── Playbook extraction ──
        if !response.goals.is_empty() {
            let enriched_pb_transcript = if engine.config.enable_context_enrichment {
                let goal_store = engine.goal_store.read().await;
                let mut existing = Vec::new();
                for goal in &response.goals {
                    for (id, _score) in goal_store.find_similar(&goal.description, 3) {
                        if let Some(entry) = goal_store.get(id) {
                            if let Some(ref pb) = entry.playbook {
                                existing.push((entry.description.clone(), pb.clone()));
                            }
                        }
                    }
                }
                drop(goal_store);
                if existing.is_empty() {
                    full_transcript.clone()
                } else {
                    let ctx = crate::context_enrichment::build_playbook_context(
                        &existing,
                        engine.config.enrichment_config.max_similar_playbooks,
                    );
                    format!(
                        "{}\n\n[Prior Playbook Experience]\n{}",
                        full_transcript, ctx
                    )
                }
            } else {
                full_transcript.clone()
            };

            if let Ok(Some(pb_response)) = tokio::time::timeout(
                std::time::Duration::from_secs(30),
                extract_playbooks(llm.as_ref(), &enriched_pb_transcript, &response.goals),
            )
            .await
            {
                let quality_playbooks: Vec<_> = pb_response
                    .playbooks
                    .into_iter()
                    .filter(|pb| {
                        pb.confidence >= MIN_PLAYBOOK_CONFIDENCE && !pb.steps_taken.is_empty()
                    })
                    .collect();
                result.playbooks_extracted = quality_playbooks.len();
                attach_playbooks(engine, &quality_playbooks, case_id).await;
            }
        }
    }

    result
}

// ────────── Rich Edge Text for Embedding ──────────

/// Build rich embedding text for an edge that goes beyond bare triplets.
///
/// Instead of "user location tokyo", produces something like:
/// "User currently lives in Tokyo. (recently moved, depends on: User relocated for work)"
///
/// Uses the original `statement` property (a self-contained natural language
/// proposition) when available, and appends qualifiers: temporal signal,
/// dependency, current/historical status.
fn build_rich_edge_text(
    source: &str,
    association_type: &str,
    target: &str,
    properties: &HashMap<String, serde_json::Value>,
    is_current: bool,
) -> String {
    // Prefer the original statement (richest representation)
    let base = if let Some(serde_json::Value::String(stmt)) = properties.get("statement") {
        stmt.clone()
    } else {
        // Fallback: reconstruct from triplet
        let predicate = association_type
            .split(':')
            .nth(1)
            .unwrap_or(association_type)
            .replace('_', " ");
        format!("{} {} {}", source, predicate, target)
    };

    let mut parts = vec![base];

    // Temporal qualifier
    if let Some(serde_json::Value::String(ts)) = properties.get("temporal_signal") {
        if !ts.is_empty() {
            parts.push(format!("({})", ts));
        }
    }

    // Dependency qualifier — what condition must hold for this fact
    if let Some(serde_json::Value::String(dep)) = properties.get("depends_on") {
        if !dep.is_empty() {
            parts.push(format!("[while: {}]", dep));
        }
    }

    // Current vs historical status
    if is_current {
        parts.push("[current]".to_string());
    } else {
        parts.push("[historical/superseded]".to_string());
    }

    parts.join(" ")
}

// ────────── Post-Compaction Embedding + Claim Creation ──────────

/// Embed all un-embedded concept nodes and create claims from extracted facts.
///
/// This ensures that node vector search and claim hybrid search (BM25+vector)
/// can find compaction-created artifacts. Must complete BEFORE queries execute.
async fn embed_nodes_and_create_claims(
    engine: &crate::integration::GraphEngine,
    facts: &[ExtractedFact],
    base_ts: u64,
) {
    let embedding_client = match &engine.embedding_client {
        Some(ec) => ec.clone(),
        None => {
            tracing::debug!("COMPACTION: no embedding client — skipping node/claim embedding");
            return;
        },
    };

    // 1. Embed un-embedded concept nodes
    let nodes_to_embed: Vec<(u64, String)> = {
        let inference = engine.inference.read().await;
        let graph = inference.graph();
        graph
            .concept_index
            .iter()
            .filter_map(|(name, &nid)| {
                graph.get_node(nid).and_then(|node| {
                    if node.embedding.is_empty() {
                        Some((nid, name.to_string()))
                    } else {
                        None
                    }
                })
            })
            .collect()
    };

    if !nodes_to_embed.is_empty() {
        tracing::info!(
            "COMPACTION: embedding {} concept nodes for vector search",
            nodes_to_embed.len()
        );
        for (nid, text) in &nodes_to_embed {
            match embedding_client
                .embed(crate::claims::EmbeddingRequest {
                    text: text.clone(),
                    context: None,
                })
                .await
            {
                Ok(resp) if !resp.embedding.is_empty() => {
                    let mut inf = engine.inference.write().await;
                    let graph = inf.graph_mut();
                    if let Some(node) = graph.get_node_mut(*nid) {
                        node.embedding = resp.embedding.clone();
                    }
                    graph.node_vector_index.insert(*nid, resp.embedding);
                },
                Ok(_) => {},
                Err(e) => {
                    tracing::debug!("COMPACTION: node embedding failed for '{}': {}", text, e);
                },
            }
        }
    }

    // 2. Embed triplets (subject + predicate + object) for edge vector search.
    // This enables triplet scoring where the query is compared against
    // the full context of each edge, not just individual nodes.
    {
        let edges_to_embed: Vec<(u64, String)> = {
            let inference = engine.inference.read().await;
            let graph = inference.graph();
            let mut edges = Vec::new();
            for (eid, edge) in graph.edges.iter() {
                if let crate::structures::EdgeType::Association {
                    association_type, ..
                } = &edge.edge_type
                {
                    let source_name =
                        crate::conversation::graph_projection::concept_name_of(graph, edge.source)
                            .unwrap_or_default();
                    let target_name =
                        crate::conversation::graph_projection::concept_name_of(graph, edge.target)
                            .unwrap_or_default();
                    if !source_name.is_empty() && !target_name.is_empty() {
                        // Build rich embedding text: statement + qualifiers.
                        // Goes beyond bare triplets — includes context, temporal
                        // signals, and dependency info stored on the edge.
                        let rich_text = build_rich_edge_text(
                            &source_name,
                            association_type,
                            &target_name,
                            &edge.properties,
                            edge.valid_until.is_none(),
                        );
                        edges.push((eid, rich_text));
                    }
                }
            }
            edges
        };

        if !edges_to_embed.is_empty() {
            tracing::info!(
                "COMPACTION: embedding {} edge triplets for vector search",
                edges_to_embed.len()
            );
            for (eid, text) in &edges_to_embed {
                match embedding_client
                    .embed(crate::claims::EmbeddingRequest {
                        text: text.clone(),
                        context: None,
                    })
                    .await
                {
                    Ok(resp) if !resp.embedding.is_empty() => {
                        let mut inf = engine.inference.write().await;
                        let graph = inf.graph_mut();
                        graph.edge_vector_index.insert(*eid, resp.embedding);
                    },
                    Ok(_) => {},
                    Err(e) => {
                        tracing::debug!("COMPACTION: edge embedding failed for '{}': {}", text, e);
                    },
                }
            }
        }
    }

    // 3. Create claims from extracted facts (for claim hybrid search)
    let claim_store = match &engine.claim_store {
        Some(cs) => cs.clone(),
        None => return,
    };

    let mut claims_created = 0usize;
    for (i, fact) in facts.iter().enumerate() {
        // Embed the fact statement
        let embedding = match embedding_client
            .embed(crate::claims::EmbeddingRequest {
                text: fact.statement.clone(),
                context: Some(format!(
                    "{} {} {}",
                    fact.subject, fact.predicate, fact.object
                )),
            })
            .await
        {
            Ok(resp) => resp.embedding,
            Err(_) => vec![], // store claim without embedding — BM25 still works
        };

        let claim_id = match claim_store.next_id() {
            Ok(id) => id,
            Err(_) => continue,
        };

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs();

        let claim_type = match fact.category.as_deref() {
            Some("preference") => crate::claims::ClaimType::Preference,
            Some("intention") | Some("goal") => crate::claims::ClaimType::Intention,
            Some("belief") | Some("opinion") => crate::claims::ClaimType::Belief,
            _ => crate::claims::ClaimType::Fact,
        };

        let claim = crate::claims::DerivedClaim {
            id: claim_id,
            claim_text: fact.statement.clone(),
            supporting_evidence: vec![crate::claims::EvidenceSpan::new(
                0,
                fact.statement.len(),
                &fact.statement,
            )],
            confidence: fact.confidence,
            embedding,
            source_event_id: 0,
            episode_id: None,
            thread_id: None,
            user_id: None,
            workspace_id: None,
            created_at: now,
            last_accessed: now,
            access_count: 0,
            status: crate::claims::ClaimStatus::Active,
            support_count: 1,
            metadata: HashMap::new(),
            claim_type,
            subject_entity: Some(fact.subject.to_lowercase()),
            predicate: Some(fact.predicate.clone()),
            object_entity: Some(fact.object.to_lowercase()),
            expires_at: None,
            superseded_by: None,
            entities: vec![],
            category: fact.category.clone(),
            temporal_type: crate::claims::TemporalType::Dynamic,
            valid_from: Some(base_ts + i as u64 * 1_000),
            valid_until: None,
        };

        if let Ok(()) = claim_store.store(&claim) {
            // ── State anchor stamping ──
            // Mirror the anchor logic from integration_claims.rs so that
            // apply_state_anchor_filter() can detect stale compaction claims.
            if let Some(ref subj) = claim.subject_entity {
                let subj_normalized = subj.to_lowercase();
                let inf = engine.inference.read().await;
                let projected = graph_projection::project_entity_state(
                    inf.graph(),
                    subj_normalized.trim(),
                    u64::MAX,
                    Some(engine.ontology.as_ref()),
                );
                drop(inf);
                let mut anchor_meta: HashMap<String, String> = HashMap::new();
                for slot in projected.slots.values() {
                    let cat = slot.association_type.split(':').next().unwrap_or("");
                    if engine.ontology.is_single_valued(cat) {
                        let key = format!("state_anchor:{}", slot.association_type);
                        let val = slot.value.as_deref().unwrap_or(&slot.target_name);
                        anchor_meta.insert(key, val.to_string());
                    }
                }
                if !anchor_meta.is_empty() {
                    let _ = claim_store.update_metadata(claim.id, &anchor_meta);
                    tracing::debug!(
                        "COMPACTION: stamped {} state anchors on claim {}",
                        anchor_meta.len(),
                        claim.id,
                    );
                }
            }

            // Also add a Claim node to the graph for unified retrieval
            let mut inf = engine.inference.write().await;
            let graph = inf.graph_mut();
            let claim_node =
                crate::structures::GraphNode::new(crate::structures::NodeType::Claim {
                    claim_id,
                    claim_text: fact.statement.clone(),
                    confidence: fact.confidence,
                    source_event_id: 0,
                });
            if let Ok(node_id) = graph.add_node(claim_node) {
                graph.claim_index.insert(claim_id, node_id);
                // Index in BM25
                graph.bm25_index.index_document(node_id, &fact.statement);
            }
            claims_created += 1;
        }
    }

    if claims_created > 0 {
        tracing::info!(
            "COMPACTION: created {} claims from extracted facts for hybrid search",
            claims_created
        );
    }
}

// ────────── Tests ──────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::types::ConversationSession;

    fn make_ingest(messages: Vec<(&str, &str)>) -> ConversationIngest {
        ConversationIngest {
            case_id: Some("test_compaction".to_string()),
            sessions: vec![ConversationSession {
                session_id: "s1".to_string(),
                topic: None,
                messages: messages
                    .into_iter()
                    .map(|(role, content)| ConversationMessage {
                        role: role.to_string(),
                        content: content.to_string(),
                        metadata: Default::default(),
                    })
                    .collect(),
                timestamp: None,
                contains_fact: None,
                fact_id: None,
                fact_quote: None,
                answers: vec![],
            }],
            queries: vec![],
            group_id: Default::default(),
            metadata: Default::default(),
        }
    }

    #[test]
    fn test_split_into_turns() {
        let data = ConversationIngest {
            case_id: Some("test".to_string()),
            sessions: vec![
                ConversationSession {
                    session_id: "s1".to_string(),
                    topic: None,
                    messages: vec![
                        ConversationMessage {
                            role: "user".to_string(),
                            content: "Hi".to_string(),
                            metadata: Default::default(),
                        },
                        ConversationMessage {
                            role: "assistant".to_string(),
                            content: "Hello!".to_string(),
                            metadata: Default::default(),
                        },
                        ConversationMessage {
                            role: "user".to_string(),
                            content: "Where am I?".to_string(),
                            metadata: Default::default(),
                        },
                        ConversationMessage {
                            role: "assistant".to_string(),
                            content: "Lisbon".to_string(),
                            metadata: Default::default(),
                        },
                    ],
                    timestamp: None,
                contains_fact: None,
                    fact_id: None,
                    fact_quote: None,
                    answers: vec![],
                },
                ConversationSession {
                    session_id: "s2".to_string(),
                    topic: None,
                    messages: vec![
                        ConversationMessage {
                            role: "user".to_string(),
                            content: "I moved".to_string(),
                            metadata: Default::default(),
                        },
                        ConversationMessage {
                            role: "assistant".to_string(),
                            content: "Where?".to_string(),
                            metadata: Default::default(),
                        },
                    ],
                    timestamp: None,
                contains_fact: None,
                    fact_id: None,
                    fact_quote: None,
                    answers: vec![],
                },
            ],
            queries: vec![],
            group_id: Default::default(),
            metadata: Default::default(),
        };

        let turns = split_into_turns(&data);
        assert_eq!(turns.len(), 3);
        assert_eq!(turns[0].turn_index, 0);
        assert_eq!(turns[0].session_index, 0);
        assert_eq!(turns[0].messages.len(), 2);
        assert_eq!(turns[1].turn_index, 1);
        assert_eq!(turns[1].session_index, 0);
        assert_eq!(turns[2].turn_index, 2);
        assert_eq!(turns[2].session_index, 1);
    }

    #[test]
    fn test_format_turn_transcript() {
        let turn = ConversationTurn {
            messages: vec![
                ConversationMessage {
                    role: "user".to_string(),
                    content: "I moved to NYC".to_string(),
                    metadata: Default::default(),
                },
                ConversationMessage {
                    role: "assistant".to_string(),
                    content: "Great!".to_string(),
                    metadata: Default::default(),
                },
            ],
            session_index: 0,
            turn_index: 1,
        };

        let transcript = format_turn_transcript(
            &turn,
            Some("User lives in Lisbon"),
            "User: state:location=Lisbon",
        );
        assert!(transcript.contains("User lives in Lisbon"));
        assert!(transcript.contains("state:location=Lisbon"));
        assert!(transcript.contains("user: I moved to NYC"));
    }

    // 1. test_format_transcript
    #[test]
    fn test_format_transcript() {
        let data = make_ingest(vec![
            ("user", "Hello, I want to plan a trip"),
            ("assistant", "Sure! Where would you like to go?"),
            ("user", "I want to visit Japan in April"),
        ]);

        let transcript = format_transcript(&data);
        assert!(transcript.contains("user: Hello, I want to plan a trip\n"));
        assert!(transcript.contains("assistant: Sure! Where would you like to go?\n"));
        assert!(transcript.contains("user: I want to visit Japan in April\n"));
    }

    #[test]
    fn test_format_transcript_truncation() {
        // Create a very long transcript that exceeds MAX_TRANSCRIPT_CHARS
        let long_msg = "x".repeat(20_000);
        let data = make_ingest(vec![("user", &long_msg)]);

        let transcript = format_transcript(&data);
        assert!(transcript.len() <= MAX_TRANSCRIPT_CHARS);
    }

    // 2. test_parse_compaction_response
    #[test]
    fn test_parse_compaction_response() {
        let json = r#"{
            "facts": [
                {
                    "statement": "Alice lives in Paris",
                    "subject": "Alice",
                    "predicate": "lives_in",
                    "object": "Paris",
                    "confidence": 0.9
                }
            ],
            "goals": [
                {
                    "description": "Plan a trip to Japan",
                    "status": "active",
                    "owner": "user"
                }
            ],
            "procedural_summary": {
                "objective": "Plan vacation",
                "progress_status": "in_progress",
                "steps": [
                    {
                        "step_number": 1,
                        "action": "Choose destination",
                        "result": "Selected Japan",
                        "outcome": "success"
                    }
                ],
                "overall_summary": "User is planning a trip to Japan",
                "takeaway": "User prefers travel to Asia"
            }
        }"#;

        let response: CompactionResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.facts.len(), 1);
        assert_eq!(response.facts[0].subject, "Alice");
        assert_eq!(response.goals.len(), 1);
        assert_eq!(response.goals[0].status, "active");
        assert!(response.procedural_summary.is_some());
        let summary = response.procedural_summary.unwrap();
        assert_eq!(summary.steps.len(), 1);
        assert_eq!(summary.progress_status, "in_progress");
    }

    // 3. test_parse_compaction_response_minimal
    #[test]
    fn test_parse_compaction_response_minimal() {
        let json = r#"{
            "facts": [],
            "goals": [],
            "procedural_summary": null
        }"#;

        let response: CompactionResponse = serde_json::from_str(json).unwrap();
        assert!(response.facts.is_empty());
        assert!(response.goals.is_empty());
        assert!(response.procedural_summary.is_none());
    }

    // 4. test_parse_compaction_response_fenced
    #[test]
    fn test_parse_compaction_response_fenced() {
        let fenced = r#"```json
{
    "facts": [{"statement": "Bob is tall", "subject": "Bob", "predicate": "is", "object": "tall", "confidence": 0.8}],
    "goals": [],
    "procedural_summary": null
}
```"#;

        let value = parse_json_from_llm(fenced).unwrap();
        let response: CompactionResponse = serde_json::from_value(value).unwrap();
        assert_eq!(response.facts.len(), 1);
        assert_eq!(response.facts[0].statement, "Bob is tall");
    }

    // 5. test_compaction_to_events_facts
    #[test]
    fn test_compaction_to_events_facts() {
        let response = CompactionResponse {
            facts: vec![
                ExtractedFact {
                    statement: "Alice lives in Paris".to_string(),
                    subject: "Alice".to_string(),
                    predicate: "lives_in".to_string(),
                    object: "Paris".to_string(),
                    confidence: 0.9,
                    category: None,
                    amount: None,
                    split_with: None,
                    temporal_signal: None,
                    depends_on: None,
                    is_update: None,
                    cardinality_hint: None,
                    sentiment: None,
                    group_id: Default::default(),
                    ingest_metadata: Default::default(),
                },
                ExtractedFact {
                    statement: "Bob works at Google".to_string(),
                    subject: "Bob".to_string(),
                    predicate: "works_at".to_string(),
                    object: "Google".to_string(),
                    confidence: 0.85,
                    category: None,
                    amount: None,
                    split_with: None,
                    temporal_signal: None,
                    depends_on: None,
                    is_update: None,
                    cardinality_hint: None,
                    sentiment: None,
                    group_id: Default::default(),
                    ingest_metadata: Default::default(),
                },
                ExtractedFact {
                    statement: "Alice and Bob are friends".to_string(),
                    subject: "Alice".to_string(),
                    predicate: "friends_with".to_string(),
                    object: "Bob".to_string(),
                    confidence: 0.75,
                    category: None,
                    amount: None,
                    split_with: None,
                    temporal_signal: None,
                    depends_on: None,
                    is_update: None,
                    cardinality_hint: None,
                    sentiment: None,
                    group_id: Default::default(),
                    ingest_metadata: Default::default(),
                },
            ],
            goals: vec![],
            procedural_summary: None,
        };

        let events = compaction_to_events(&response, "case1", 100, 200, 1000);
        assert_eq!(events.len(), 3);

        for evt in &events {
            assert_eq!(evt.agent_type, "conversation_compaction");
            match &evt.event_type {
                EventType::Observation {
                    observation_type,
                    source,
                    ..
                } => {
                    assert_eq!(observation_type, "extracted_fact");
                    assert_eq!(source, "conversation_compaction");
                },
                other => panic!("Expected Observation, got {:?}", other),
            }
            assert!(evt.metadata.contains_key("compaction_fact"));
        }
    }

    // 6. test_compaction_to_events_goals
    #[test]
    fn test_compaction_to_events_goals() {
        let response = CompactionResponse {
            facts: vec![],
            goals: vec![
                ExtractedGoal {
                    description: "Plan trip to Japan".to_string(),
                    status: "active".to_string(),
                    owner: "user".to_string(),
                },
                ExtractedGoal {
                    description: "Learn Rust".to_string(),
                    status: "completed".to_string(),
                    owner: "user".to_string(),
                },
            ],
            procedural_summary: None,
        };

        let events = compaction_to_events(&response, "case1", 100, 200, 1000);
        assert_eq!(events.len(), 2);

        for evt in &events {
            match &evt.event_type {
                EventType::Cognitive {
                    process_type,
                    reasoning_trace,
                    ..
                } => {
                    assert_eq!(*process_type, CognitiveType::GoalFormation);
                    assert_eq!(reasoning_trace[0], "LLM conversation compaction");
                },
                other => panic!("Expected Cognitive, got {:?}", other),
            }
            assert!(evt.metadata.contains_key("compaction_goal"));
        }
    }

    // 7. test_compaction_to_events_steps
    #[test]
    fn test_compaction_to_events_steps() {
        let response = CompactionResponse {
            facts: vec![],
            goals: vec![],
            procedural_summary: Some(ProceduralSummary {
                objective: "Deploy app".to_string(),
                progress_status: "completed".to_string(),
                steps: vec![
                    ProceduralStep {
                        step_number: 1,
                        action: "Build Docker image".to_string(),
                        result: "Image built successfully".to_string(),
                        outcome: "success".to_string(),
                    },
                    ProceduralStep {
                        step_number: 2,
                        action: "Push to registry".to_string(),
                        result: "Push failed due to auth".to_string(),
                        outcome: "failure".to_string(),
                    },
                    ProceduralStep {
                        step_number: 3,
                        action: "Retry with credentials".to_string(),
                        result: "Push succeeded".to_string(),
                        outcome: "success".to_string(),
                    },
                ],
                overall_summary: "Deployed app after auth fix".to_string(),
                takeaway: "Always verify registry credentials".to_string(),
            }),
        };

        let events = compaction_to_events(&response, "case1", 100, 200, 1000);
        assert_eq!(events.len(), 3);

        // Check outcome mapping
        match &events[0].event_type {
            EventType::Action {
                action_name,
                outcome,
                ..
            } => {
                assert_eq!(action_name, "step_1");
                assert!(matches!(outcome, ActionOutcome::Success { .. }));
            },
            other => panic!("Expected Action, got {:?}", other),
        }

        match &events[1].event_type {
            EventType::Action { outcome, .. } => {
                assert!(matches!(outcome, ActionOutcome::Failure { .. }));
            },
            other => panic!("Expected Action, got {:?}", other),
        }

        for evt in &events {
            assert!(evt.metadata.contains_key("compaction_step"));
        }
    }

    // 8. test_compaction_to_events_mixed
    #[test]
    fn test_compaction_to_events_mixed() {
        let response = CompactionResponse {
            facts: vec![
                ExtractedFact {
                    statement: "F1".to_string(),
                    subject: "s".to_string(),
                    predicate: "p".to_string(),
                    object: "o".to_string(),
                    confidence: 0.9,
                    category: None,
                    amount: None,
                    split_with: None,
                    temporal_signal: None,
                    depends_on: None,
                    is_update: None,
                    cardinality_hint: None,
                    sentiment: None,
                    group_id: Default::default(),
                    ingest_metadata: Default::default(),
                },
                ExtractedFact {
                    statement: "F2".to_string(),
                    subject: "s".to_string(),
                    predicate: "p".to_string(),
                    object: "o".to_string(),
                    confidence: 0.8,
                    category: None,
                    amount: None,
                    split_with: None,
                    temporal_signal: None,
                    depends_on: None,
                    is_update: None,
                    cardinality_hint: None,
                    sentiment: None,
                    group_id: Default::default(),
                    ingest_metadata: Default::default(),
                },
            ],
            goals: vec![ExtractedGoal {
                description: "G1".to_string(),
                status: "active".to_string(),
                owner: "user".to_string(),
            }],
            procedural_summary: Some(ProceduralSummary {
                objective: "Obj".to_string(),
                progress_status: "completed".to_string(),
                steps: vec![ProceduralStep {
                    step_number: 1,
                    action: "A".to_string(),
                    result: "R".to_string(),
                    outcome: "success".to_string(),
                }],
                overall_summary: "S".to_string(),
                takeaway: "T".to_string(),
            }),
        };

        let events = compaction_to_events(&response, "case1", 100, 200, 1000);
        // 2 facts + 1 goal + 1 step = 4
        assert_eq!(events.len(), 4);

        // Timestamps must be monotonically increasing
        for window in events.windows(2) {
            assert!(
                window[1].timestamp > window[0].timestamp,
                "Timestamps not monotonic: {} <= {}",
                window[1].timestamp,
                window[0].timestamp
            );
        }
    }

    // 9. test_compaction_to_events_empty
    #[test]
    fn test_compaction_to_events_empty() {
        let response = CompactionResponse {
            facts: vec![],
            goals: vec![],
            procedural_summary: None,
        };

        let events = compaction_to_events(&response, "case1", 100, 200, 1000);
        assert!(events.is_empty());
    }

    // 10. test_build_procedural_memory
    #[test]
    fn test_build_procedural_memory() {
        let summary = ProceduralSummary {
            objective: "Deploy app".to_string(),
            progress_status: "completed".to_string(),
            steps: vec![],
            overall_summary: "Successfully deployed the application".to_string(),
            takeaway: "Always test in staging first".to_string(),
        };

        let memory = build_procedural_memory(&summary, 100, 200, 300);

        assert_eq!(memory.summary, "Successfully deployed the application");
        assert_eq!(memory.takeaway, "Always test in staging first");
        assert_eq!(
            memory.causal_note,
            "Objective: Deploy app. Status: completed"
        );
        assert_eq!(memory.tier, MemoryTier::Episodic);
        assert_eq!(memory.strength, 0.8);
        assert_eq!(memory.outcome, EpisodeOutcome::Success);
        assert_eq!(memory.agent_id, 100);
        assert_eq!(memory.session_id, 200);
        assert_eq!(memory.episode_id, 300);
        assert_eq!(
            memory.metadata.get("source").map(|s| s.as_str()),
            Some("compaction")
        );
        assert_eq!(
            memory.metadata.get("objective").map(|s| s.as_str()),
            Some("Deploy app")
        );
        assert_eq!(
            memory.metadata.get("progress_status").map(|s| s.as_str()),
            Some("completed")
        );
    }

    // 11. test_build_procedural_memory_outcome_mapping
    #[test]
    fn test_build_procedural_memory_outcome_mapping() {
        assert_eq!(
            map_progress_to_outcome("completed"),
            EpisodeOutcome::Success
        );
        assert_eq!(map_progress_to_outcome("blocked"), EpisodeOutcome::Failure);
        assert_eq!(
            map_progress_to_outcome("abandoned"),
            EpisodeOutcome::Failure
        );
        assert_eq!(
            map_progress_to_outcome("in_progress"),
            EpisodeOutcome::Partial
        );
        assert_eq!(
            map_progress_to_outcome("unknown_status"),
            EpisodeOutcome::Partial
        );
    }

    // 12. test_compaction_result_has_classifier_fields
    #[test]
    fn test_compaction_result_has_classifier_fields() {
        let result = CompactionResult::default();
        assert_eq!(result.memories_updated, 0);
        assert_eq!(result.memories_deleted, 0);
        assert_eq!(result.facts_extracted, 0);
        assert!(!result.procedural_memory_created);
        assert!(!result.llm_success);
    }

    // 13. test_goals_deduplicated_field_default
    #[test]
    fn test_goals_deduplicated_field_default() {
        let result = CompactionResult::default();
        assert_eq!(result.goals_deduplicated, 0);
    }

    // 14. test_filter_goals_add_keeps
    #[test]
    fn test_filter_goals_add_keeps() {
        use crate::memory_classifier::{ClassifiedOperation, MemoryAction};

        let goals = vec![ExtractedGoal {
            description: "Visit Japan".to_string(),
            status: "active".to_string(),
            owner: "user".to_string(),
        }];
        let ops = vec![ClassifiedOperation {
            action: MemoryAction::Add,
            target_index: None,
            new_text: Some("Visit Japan".to_string()),
            fact_text: "Visit Japan".to_string(),
        }];

        let (approved, dedup) = filter_goals_by_classification(&goals, &ops);
        assert_eq!(approved.len(), 1);
        assert_eq!(approved[0].description, "Visit Japan");
        assert_eq!(dedup, 0);
    }

    // 15. test_filter_goals_none_filters
    #[test]
    fn test_filter_goals_none_filters() {
        use crate::memory_classifier::{ClassifiedOperation, MemoryAction};

        let goals = vec![ExtractedGoal {
            description: "Visit Japan".to_string(),
            status: "active".to_string(),
            owner: "user".to_string(),
        }];
        let ops = vec![ClassifiedOperation {
            action: MemoryAction::None,
            target_index: None,
            new_text: None,
            fact_text: "Visit Japan".to_string(),
        }];

        let (approved, dedup) = filter_goals_by_classification(&goals, &ops);
        assert!(approved.is_empty());
        assert_eq!(dedup, 1);
    }

    // 16. test_filter_goals_mixed
    #[test]
    fn test_filter_goals_mixed() {
        use crate::memory_classifier::{ClassifiedOperation, MemoryAction};

        let goals = vec![
            ExtractedGoal {
                description: "Visit Japan".to_string(),
                status: "active".to_string(),
                owner: "user".to_string(),
            },
            ExtractedGoal {
                description: "Learn Rust".to_string(),
                status: "active".to_string(),
                owner: "user".to_string(),
            },
            ExtractedGoal {
                description: "Buy groceries".to_string(),
                status: "completed".to_string(),
                owner: "user".to_string(),
            },
            ExtractedGoal {
                description: "Read a book".to_string(),
                status: "active".to_string(),
                owner: "user".to_string(),
            },
        ];
        let ops = vec![
            ClassifiedOperation {
                action: MemoryAction::Add,
                target_index: None,
                new_text: Some("Visit Japan".to_string()),
                fact_text: "Visit Japan".to_string(),
            },
            ClassifiedOperation {
                action: MemoryAction::None,
                target_index: None,
                new_text: None,
                fact_text: "Learn Rust".to_string(),
            },
            ClassifiedOperation {
                action: MemoryAction::Delete,
                target_index: Some(0),
                new_text: None,
                fact_text: "Buy groceries".to_string(),
            },
            ClassifiedOperation {
                action: MemoryAction::Update,
                target_index: Some(1),
                new_text: Some("Read a book regularly".to_string()),
                fact_text: "Read a book".to_string(),
            },
        ];

        let (approved, dedup) = filter_goals_by_classification(&goals, &ops);
        // ADD + UPDATE kept, NONE + DELETE filtered
        assert_eq!(approved.len(), 2);
        assert_eq!(approved[0].description, "Visit Japan");
        assert_eq!(approved[1].description, "Read a book");
        assert_eq!(dedup, 2);
    }

    // 17. test_compaction_to_events_with_filtered_goals
    #[test]
    fn test_compaction_to_events_with_filtered_goals() {
        // An empty goals vec produces only fact+step events (no Cognitive)
        let response = CompactionResponse {
            facts: vec![ExtractedFact {
                statement: "F1".to_string(),
                subject: "s".to_string(),
                predicate: "p".to_string(),
                object: "o".to_string(),
                confidence: 0.9,
                category: None,
                amount: None,
                split_with: None,
                temporal_signal: None,
                depends_on: None,
                is_update: None,
                cardinality_hint: None,
                sentiment: None,
                ingest_metadata: Default::default(),
                group_id: Default::default(),
            }],
            goals: vec![], // all goals filtered out
            procedural_summary: Some(ProceduralSummary {
                objective: "Obj".to_string(),
                progress_status: "completed".to_string(),
                steps: vec![ProceduralStep {
                    step_number: 1,
                    action: "A".to_string(),
                    result: "R".to_string(),
                    outcome: "success".to_string(),
                }],
                overall_summary: "S".to_string(),
                takeaway: "T".to_string(),
            }),
        };

        let events = compaction_to_events(&response, "case1", 100, 200, 1000);
        // 1 fact + 0 goals + 1 step = 2
        assert_eq!(events.len(), 2);

        // No Cognitive events
        for evt in &events {
            assert!(
                !matches!(&evt.event_type, EventType::Cognitive { .. }),
                "Expected no Cognitive events when goals are filtered"
            );
        }
    }

    // 18. test_fallback_keeps_all_goals
    #[test]
    fn test_fallback_keeps_all_goals() {
        // When no classification is provided (more goals than ops), extras default to ADD
        use crate::memory_classifier::ClassifiedOperation;

        let goals = vec![
            ExtractedGoal {
                description: "G1".to_string(),
                status: "active".to_string(),
                owner: "user".to_string(),
            },
            ExtractedGoal {
                description: "G2".to_string(),
                status: "active".to_string(),
                owner: "user".to_string(),
            },
            ExtractedGoal {
                description: "G3".to_string(),
                status: "active".to_string(),
                owner: "user".to_string(),
            },
        ];
        // Empty ops — all goals should pass through (fallback to Add)
        let ops: Vec<ClassifiedOperation> = vec![];

        let (approved, dedup) = filter_goals_by_classification(&goals, &ops);
        assert_eq!(approved.len(), 3);
        assert_eq!(dedup, 0);
    }

    // 19. test_fast_goal_dedup_filters_duplicates
    #[test]
    fn test_fast_goal_dedup_filters_duplicates() {
        use crate::goal_store::{GoalDedupDecision, GoalStore};

        let mut store = GoalStore::new();

        // First time: all goals are new
        let goals = vec![
            ExtractedGoal {
                description: "Plan trip to Japan".to_string(),
                status: "active".to_string(),
                owner: "user".to_string(),
            },
            ExtractedGoal {
                description: "Learn Rust programming".to_string(),
                status: "active".to_string(),
                owner: "user".to_string(),
            },
        ];

        let mut new_goals = Vec::new();
        let mut dedup_count = 0usize;
        for goal in &goals {
            match store.store_or_dedup(&goal.description, &goal.status, &goal.owner, "case1") {
                GoalDedupDecision::NewGoal => new_goals.push(goal.clone()),
                GoalDedupDecision::Duplicate { .. } => dedup_count += 1,
                GoalDedupDecision::StatusUpdate {
                    existing_id,
                    new_status,
                } => {
                    store.update_status(existing_id, new_status);
                    dedup_count += 1;
                },
            }
        }
        assert_eq!(new_goals.len(), 2);
        assert_eq!(dedup_count, 0);

        // Second time: same goals → all duplicates
        let mut new_goals2 = Vec::new();
        let mut dedup_count2 = 0usize;
        for goal in &goals {
            match store.store_or_dedup(&goal.description, &goal.status, &goal.owner, "case2") {
                GoalDedupDecision::NewGoal => new_goals2.push(goal.clone()),
                GoalDedupDecision::Duplicate { .. } => dedup_count2 += 1,
                GoalDedupDecision::StatusUpdate {
                    existing_id,
                    new_status,
                } => {
                    store.update_status(existing_id, new_status);
                    dedup_count2 += 1;
                },
            }
        }
        assert!(new_goals2.is_empty(), "All goals should be deduplicated");
        assert_eq!(dedup_count2, 2);
    }

    // 20. test_goal_playbook_serde
    #[test]
    fn test_goal_playbook_serde() {
        let playbook = GoalPlaybook {
            goal_description: "Deploy the app".to_string(),
            what_worked: vec!["Docker build".to_string()],
            what_didnt_work: vec!["Manual deployment".to_string()],
            lessons_learned: vec!["Always use CI/CD".to_string()],
            steps_taken: vec![
                "Build".to_string(),
                "Push".to_string(),
                "Deploy".to_string(),
            ],
            confidence: 0.85,
        };

        let json = serde_json::to_string(&playbook).unwrap();
        let roundtrip: GoalPlaybook = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtrip.goal_description, "Deploy the app");
        assert_eq!(roundtrip.what_worked.len(), 1);
        assert_eq!(roundtrip.what_didnt_work.len(), 1);
        assert_eq!(roundtrip.lessons_learned.len(), 1);
        assert_eq!(roundtrip.steps_taken.len(), 3);
        assert!((roundtrip.confidence - 0.85).abs() < f32::EPSILON);
    }

    // 21. test_playbook_extraction_response_serde
    #[test]
    fn test_playbook_extraction_response_serde() {
        let json = r#"{
            "playbooks": [
                {
                    "goal_description": "Plan trip",
                    "what_worked": ["Booked flights early"],
                    "what_didnt_work": ["Waited too long for hotel"],
                    "lessons_learned": ["Book everything 2 months ahead"],
                    "steps_taken": ["Research", "Book flights", "Find hotel"],
                    "confidence": 0.9
                }
            ]
        }"#;

        let response: PlaybookExtractionResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.playbooks.len(), 1);
        assert_eq!(response.playbooks[0].goal_description, "Plan trip");
        assert_eq!(response.playbooks[0].what_worked[0], "Booked flights early");
    }

    // 22. test_playbook_extraction_response_empty
    #[test]
    fn test_playbook_extraction_response_empty() {
        let json = r#"{"playbooks": []}"#;
        let response: PlaybookExtractionResponse = serde_json::from_str(json).unwrap();
        assert!(response.playbooks.is_empty());
    }

    // 23. test_playbook_extraction_response_partial
    #[test]
    fn test_playbook_extraction_response_partial() {
        // Missing optional fields (steps_taken, confidence) should use defaults
        let json = r#"{
            "playbooks": [
                {
                    "goal_description": "Learn Rust",
                    "what_worked": ["Read the book"],
                    "what_didnt_work": [],
                    "lessons_learned": ["Practice daily"]
                }
            ]
        }"#;

        let response: PlaybookExtractionResponse = serde_json::from_str(json).unwrap();
        assert_eq!(response.playbooks.len(), 1);
        let pb = &response.playbooks[0];
        assert!(
            pb.steps_taken.is_empty(),
            "steps_taken should default to empty"
        );
        assert!(
            (pb.confidence - 0.5).abs() < f32::EPSILON,
            "confidence should default to 0.5"
        );
    }

    // 24. test_compaction_result_playbooks_field
    #[test]
    fn test_compaction_result_playbooks_field() {
        let result = CompactionResult::default();
        assert_eq!(result.playbooks_extracted, 0);
    }

    // ── Rolling Summary Tests ──

    // 25. test_conversation_rolling_summary_serde
    #[test]
    fn test_conversation_rolling_summary_serde() {
        let summary = ConversationRollingSummary {
            case_id: "case_123".to_string(),
            summary: "User wants to plan a trip to Japan".to_string(),
            last_updated: 1_000_000_000,
            turn_count: 5,
            token_estimate: 42,
        };

        let json = serde_json::to_string(&summary).unwrap();
        let roundtrip: ConversationRollingSummary = serde_json::from_str(&json).unwrap();
        assert_eq!(roundtrip.case_id, "case_123");
        assert_eq!(roundtrip.summary, "User wants to plan a trip to Japan");
        assert_eq!(roundtrip.last_updated, 1_000_000_000);
        assert_eq!(roundtrip.turn_count, 5);
        assert_eq!(roundtrip.token_estimate, 42);
    }

    // 26. test_format_with_summary
    #[test]
    fn test_format_with_summary() {
        let summary = ConversationRollingSummary {
            case_id: "test".to_string(),
            summary: "User is planning a trip to Japan in April.".to_string(),
            last_updated: 0,
            turn_count: 3,
            token_estimate: 10,
        };
        let data = make_ingest(vec![
            ("user", "I want to visit Tokyo"),
            ("assistant", "Great choice!"),
            ("user", "What about Kyoto?"),
            ("assistant", "Kyoto is wonderful too"),
            ("user", "Let's add Osaka"),
        ]);

        let result = format_with_summary(&summary, &data, 2);
        assert!(result.contains("[Rolling Summary]"));
        assert!(result.contains("User is planning a trip to Japan in April."));
        assert!(result.contains("[Recent Messages]"));
        // Only last 2 messages
        assert!(result.contains("assistant: Kyoto is wonderful too"));
        assert!(result.contains("user: Let's add Osaka"));
        // Earlier messages should NOT be present
        assert!(!result.contains("I want to visit Tokyo"));
    }

    // 27. test_format_with_summary_few_messages
    #[test]
    fn test_format_with_summary_few_messages() {
        let summary = ConversationRollingSummary {
            case_id: "test".to_string(),
            summary: "Summary here.".to_string(),
            last_updated: 0,
            turn_count: 1,
            token_estimate: 5,
        };
        let data = make_ingest(vec![("user", "Hello"), ("assistant", "Hi there")]);

        // recent_count > actual messages: should include all
        let result = format_with_summary(&summary, &data, 10);
        assert!(result.contains("user: Hello"));
        assert!(result.contains("assistant: Hi there"));
    }

    // 28. test_update_rolling_summary_first_call
    #[tokio::test]
    async fn test_update_rolling_summary_first_call() {
        use crate::llm_client::LlmResponse;

        // Create a simple mock that returns a summary
        struct SimpleLlm;
        #[async_trait::async_trait]
        impl LlmClient for SimpleLlm {
            async fn complete(
                &self,
                req: crate::llm_client::LlmRequest,
            ) -> anyhow::Result<LlmResponse> {
                // First call has no existing summary → prompt starts with "Summarize"
                assert!(req.user_prompt.contains("Summarize this conversation"));
                Ok(LlmResponse {
                    content: "User discussed trip plans to Japan.".to_string(),
                    tokens_used: 10,
                })
            }
            fn model_name(&self) -> &str {
                "test"
            }
        }

        let messages = vec![crate::conversation::types::ConversationMessage {
            role: "user".to_string(),
            content: "I want to go to Japan".to_string(),
            metadata: Default::default(),
        }];

        let result = update_rolling_summary(&SimpleLlm, None, &messages).await;
        assert!(result.is_some());
        assert_eq!(result.unwrap(), "User discussed trip plans to Japan.");
    }
}
