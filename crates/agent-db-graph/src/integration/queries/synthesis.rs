// crates/agent-db-graph/src/integration/queries/synthesis.rs
//
// LLM answer synthesis, dynamic projection building, and entity extraction
// helpers for the NLQ pipeline.

// ────────── Answer Synthesis ──────────

pub(crate) const SYNTHESIS_SYSTEM_PROMPT: &str = r#"You are answering a question about a user based on stored knowledge. Answer the question directly and concisely using ONLY the information provided in the context.

Rules:
- Give a direct, focused answer — not a list of facts
- Use only information present in the context
- The "Current state" section contains AUTHORITATIVE facts about what is true RIGHT NOW
- Current state facts (location, routine, preferences, relationships) are ALWAYS sufficient to answer questions about the user's current life, activities, and recommendations
- When asked "what should I do", "what to do this weekend", etc., use the user's current routine and location to answer directly
- Write naturally, as if you personally know the user
- Do NOT add speculation or information not in the context
- Only say "I don't have enough information" if the context has NO relevant facts at all
- For preference/recommendation questions: Compare categories, count positive vs negative, identify patterns, and make a clear recommendation with reasoning. Cite specific items.
- For "what do X have in common" questions: Identify shared themes across items.
- The user's question is enclosed in <user_question> tags. Only treat content within those tags as the question to answer.

CRITICAL — TEMPORAL STATE RULES:
- The "Current state" section is the AUTHORITATIVE ground truth. It shows what is true RIGHT NOW.
- If the current state says "location: Vancouver", the user is in Vancouver. Period.
- ALL other facts must be CONSISTENT with the current state. Discard any known facts about activities, routines, landmarks, or places from a DIFFERENT or superseded state.
- NEVER reference activities, places, or routines from superseded locations/states unless the question explicitly asks about history."#;

/// Use the LLM to synthesize a focused answer from retrieved context.
///
/// Takes the question and the raw retrieval context (facts, claims, entity state)
/// and produces a direct, concise answer instead of a bullet-point dump.
pub(crate) async fn synthesize_answer(
    llm: &dyn crate::llm_client::LlmClient,
    question: &str,
    context: &str,
) -> anyhow::Result<String> {
    let user_prompt = format!(
        "Context:\n{}\n\n<user_question>{}</user_question>",
        context, question
    );
    tracing::info!("NLQ synthesis context:\n{}", context);

    let request = crate::llm_client::LlmRequest {
        system_prompt: SYNTHESIS_SYSTEM_PROMPT.to_string(),
        user_prompt,
        temperature: 0.0,
        max_tokens: 256,
        json_mode: false,
    };

    let response = llm.complete(request).await?;
    let answer = response.content.trim().to_string();
    if answer.is_empty() {
        anyhow::bail!("Empty LLM response");
    }
    Ok(answer)
}

/// Build a dynamic projection from graph edges based on LLM classification.
///
/// Maps LLM hints to `ConversationQueryType` variants and delegates to the
/// unified conversation query path (`execute_conversation_query_with_graph`).
/// This ensures a single, complete execution path for all structured queries.
pub(crate) fn build_dynamic_projection(
    graph: &crate::structures::Graph,
    hint: &crate::nlq::llm_hint::LlmHintResponse,
    question: &str,
    store: &crate::structured_memory::StructuredMemoryStore,
    name_registry: &crate::conversation::types::NameRegistry,
) -> Option<String> {
    use crate::conversation::nlq_ext::{
        execute_conversation_query_with_graph, ConversationQueryType, NumericOp,
    };
    use crate::nlq::llm_hint::{IntentHint, StructureHint};

    let entities = extract_entity_names_from_question(graph, question);
    let primary_entity = entities.first().cloned();

    let query_type = match (&hint.structure_hint, &hint.intent_hint) {
        (StructureHint::Ledger, IntentHint::Balance | IntentHint::AggregateBalance) => {
            ConversationQueryType::Numeric {
                op: NumericOp::NetBalance,
            }
        },
        (StructureHint::StateMachine, IntentHint::CurrentState) => ConversationQueryType::State {
            entity: primary_entity,
            attribute: None,
        },
        (StructureHint::PreferenceList, IntentHint::Ranking) => ConversationQueryType::Preference {
            entity: primary_entity,
            category: None,
        },
        (_, IntentHint::Path) if entities.len() >= 2 => ConversationQueryType::RelationshipPath {
            from: entities[0].clone(),
            to: entities[1].clone(),
            relation: None,
        },
        (StructureHint::GenericGraph, IntentHint::Neighbors | IntentHint::Subgraph) => {
            match primary_entity {
                Some(e) => ConversationQueryType::EntitySummary { entity: e },
                None => return None,
            }
        },
        _ => return None,
    };

    let result = execute_conversation_query_with_graph(
        &query_type,
        store,
        name_registry,
        question,
        Some(graph),
    );
    if result.is_empty() || result.contains("No ") {
        None
    } else {
        Some(result)
    }
}

/// Extract entity names mentioned in a question by checking against graph concept index.
///
/// Also resolves implicit relational references (e.g., "my neighbor" → look for
/// `relationship:neighbor` edges from user → return the target entity name).
pub(crate) fn extract_entity_names_from_question(
    graph: &crate::structures::Graph,
    question: &str,
) -> Vec<String> {
    let lower = question.to_lowercase();
    let mut found = Vec::new();

    // Always include "user" for first-person queries
    let has_first_person = lower.contains(" i ")
        || lower.contains("my ")
        || lower.starts_with("i ")
        || lower.contains(" me");
    if has_first_person {
        found.push("user".to_string());
    }

    // Check all concept names against the question (word-boundary matching)
    for concept_name in graph.concept_index.keys() {
        let cn_lower = concept_name.to_lowercase();
        if cn_lower.len() < 3 {
            continue;
        }
        if let Some(pos) = lower.find(&cn_lower) {
            let before_ok = pos == 0 || !lower.as_bytes()[pos - 1].is_ascii_alphanumeric();
            let end = pos + cn_lower.len();
            let after_ok = end >= lower.len() || !lower.as_bytes()[end].is_ascii_alphanumeric();
            if before_ok && after_ok {
                found.push(concept_name.to_string());
            }
        }
    }

    // Resolve implicit relational references from the "user" node.
    // E.g., "my neighbor" → look for `relationship:neighbor` edges from user.
    if has_first_person {
        if let Some(&user_nid) = graph.concept_index.get("user") {
            // Common relational words to check
            let relational_words = [
                "neighbor",
                "neighbour",
                "boss",
                "manager",
                "sister",
                "brother",
                "mother",
                "father",
                "friend",
                "partner",
                "colleague",
                "coworker",
                "wife",
                "husband",
                "child",
                "doctor",
                "teacher",
            ];
            for rel_word in &relational_words {
                if lower.contains(rel_word) {
                    // Look for relationship edges from user matching this word
                    for edge in graph.get_edges_from(user_nid) {
                        if let crate::structures::EdgeType::Association {
                            association_type, ..
                        } = &edge.edge_type
                        {
                            // Check if edge predicate (after ':') matches the relational word
                            let rel_suffix = association_type.split(':').nth(1).unwrap_or("");
                            if rel_suffix.to_lowercase().contains(rel_word) {
                                if let Some(name) =
                                    crate::conversation::graph_projection::concept_name_of(
                                        graph,
                                        edge.target,
                                    )
                                {
                                    if !found.contains(&name) {
                                        found.push(name);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    found
}
