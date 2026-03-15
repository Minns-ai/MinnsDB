//! Core data types for LLM-driven conversation compaction.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ────────── Quality Gate Constants ──────────

/// Minimum transcript length (chars) to attempt playbook extraction.
/// ~3-4 conversational turns minimum.
pub(crate) const MIN_PLAYBOOK_TRANSCRIPT_LEN: usize = 200;

/// Minimum confidence for an extracted playbook to be attached.
pub(crate) const MIN_PLAYBOOK_CONFIDENCE: f32 = 0.4;

// ────────── Extraction Response Types ──────────

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
    pub ingest_metadata: HashMap<String, serde_json::Value>,
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

// ────────── Financial Extraction Types ──────────

#[derive(Deserialize)]
pub(crate) struct LlmTransaction {
    pub payer: String,
    pub payee: String,
    #[serde(default)]
    pub amount: f64,
}

// ────────── Playbook Types ──────────

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

// ────────── Serde Helpers ──────────

/// Deserialize f32 leniently: accept numbers, strings ("0.9"), or default to 0.8
pub(crate) fn deserialize_f32_lenient<'de, D>(deserializer: D) -> Result<f32, D::Error>
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
pub(crate) fn deserialize_bool_lenient<'de, D>(deserializer: D) -> Result<Option<bool>, D::Error>
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
pub(crate) fn deserialize_f32_option_lenient<'de, D>(
    deserializer: D,
) -> Result<Option<f32>, D::Error>
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
