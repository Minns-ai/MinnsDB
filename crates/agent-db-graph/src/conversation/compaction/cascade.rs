//! 3-call cascade extraction: entities → relationships → structured facts.

use super::extraction::{
    extract_turn_facts_single_call_fallback, lenient_to_extracted, TurnResponse,
};
use super::prompts::{cascade_fact_prompt, cascade_relationship_prompt, CASCADE_ENTITY_SYSTEM};
use super::types::ExtractedFact;
use crate::llm_client::{parse_json_from_llm, LlmClient, LlmRequest};
use serde::{Deserialize, Serialize};

// ────────── Cascade Types ──────────

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
    #[serde(default, deserialize_with = "super::types::deserialize_bool_lenient")]
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

// ────────── Helpers ──────────

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

// ────────── Main Cascade Function ──────────

/// 3-call cascade extraction: entities → relationships → structured facts.
///
/// Falls back to single-call `extract_turn_facts` on early failures.
pub(crate) async fn extract_turn_facts_cascade(
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
        Ok(Ok(resp)) => match parse_json_from_llm(&resp.content) {
            Some(value) => match serde_json::from_value::<TurnResponse>(value) {
                Ok(parsed) => {
                    let facts: Vec<ExtractedFact> = parsed
                        .facts
                        .into_iter()
                        .filter(|f| !f.statement.is_empty() || !f.subject.is_empty())
                        .map(|f| lenient_to_extracted(f))
                        .collect();
                    tracing::info!("CASCADE call 3: produced {} structured facts", facts.len());
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
