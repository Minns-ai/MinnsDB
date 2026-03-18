//! LLM extraction functions — single-call fact extraction, financial extraction, playbooks.

use super::prompts::{compaction_system_prompt, PLAYBOOK_SYSTEM_PROMPT, TURN_EXTRACTION_PROMPT};
use super::turn_processing::format_transcript;
use super::types::*;
use crate::conversation::types::ConversationIngest;
use crate::llm_client::{parse_json_from_llm, LlmClient, LlmRequest};
use serde::Deserialize;

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

// ────────── Full-Transcript Extraction ──────────

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

// ────────── Per-Turn Extraction ──────────

/// Extract facts from a single turn using the per-turn prompt.
///
/// Returns only the facts portion (no goals or procedural summary).
pub(crate) async fn extract_turn_facts(
    llm: &dyn LlmClient,
    transcript: &str,
) -> Option<Vec<ExtractedFact>> {
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

    match serde_json::from_value::<TurnResponse>(value.clone()) {
        Ok(parsed) => {
            let facts: Vec<ExtractedFact> = parsed
                .facts
                .into_iter()
                .filter(|f| !f.statement.is_empty() || !f.subject.is_empty())
                .map(lenient_to_extracted)
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

/// Fallback: format a single-call transcript and use `extract_turn_facts`.
/// Used when cascade Call 1 or Call 2 fails.
pub(crate) async fn extract_turn_facts_single_call_fallback(
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

// ────────── Financial Extraction ──────────

/// Use LLM to extract financial transactions from text and convert them into `ExtractedFact`s.
///
/// Replaces the NER-based extraction with a structured LLM call that returns
/// payer/payee/amount triples directly.
pub(crate) async fn extract_financial_facts_llm(
    llm: &dyn LlmClient,
    text: &str,
) -> Vec<ExtractedFact> {
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

// ────────── Playbook Extraction ──────────

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

// ────────── Lenient Deserialization ──────────

/// Lenient fact structure for parsing LLM responses that may have non-standard types.
#[derive(Deserialize)]
pub(crate) struct LenientFact {
    #[serde(default)]
    pub statement: String,
    #[serde(default)]
    pub subject: String,
    #[serde(default)]
    pub predicate: String,
    #[serde(default)]
    pub object: String,
    #[serde(default, deserialize_with = "deserialize_f32_lenient")]
    pub confidence: f32,
    #[serde(default)]
    pub category: Option<String>,
    #[serde(default)]
    pub amount: Option<f64>,
    #[serde(default)]
    pub split_with: Option<Vec<String>>,
    #[serde(default)]
    pub temporal_signal: Option<String>,
    #[serde(default)]
    pub depends_on: Option<String>,
    #[serde(default, deserialize_with = "deserialize_bool_lenient")]
    pub is_update: Option<bool>,
    #[serde(default)]
    pub cardinality_hint: Option<String>,
    #[serde(default, deserialize_with = "deserialize_f32_option_lenient")]
    pub sentiment: Option<f32>,
}

#[derive(Deserialize)]
pub(crate) struct TurnResponse {
    #[serde(default)]
    pub facts: Vec<LenientFact>,
}

/// Convert a `LenientFact` into a proper `ExtractedFact`.
pub(crate) fn lenient_to_extracted(f: LenientFact) -> ExtractedFact {
    ExtractedFact {
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
    }
}
