//! LLM hint classifier for NLQ intent routing.
//!
//! Provides an async LLM classifier that maps natural language questions to
//! structure and intent hints.

use crate::llm_client::{parse_json_from_llm, LlmClient, LlmRequest};
use crate::nlq::intent::{QueryIntent, StructuredQueryType};
use serde::{Deserialize, Serialize};

/// What kind of structured memory the query targets.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructureHint {
    Ledger,
    Tree,
    StateMachine,
    PreferenceList,
    GenericGraph,
}

/// What intent the LLM classifies.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum IntentHint {
    /// Pairwise balance between two specific entities
    Balance,
    /// Aggregate balance across ALL entities: "who owes the most", "all balances"
    AggregateBalance,
    CurrentState,
    Children,
    Ranking,
    Path,
    Neighbors,
    Similarity,
    Aggregate,
    Temporal,
    Subgraph,
    Knowledge,
    Unknown,
}

/// Temporal frame: how the query relates to time.
///
/// Drives two downstream decisions in NLQ execution:
/// 1. Whether the temporal-validity filter is applied (superseded edges
///    are dropped only when the frame is `Current`).
/// 2. How the retrieval ranking is *ordered* — for `First`/`Last` the
///    fused list is re-sorted by `valid_from` ascending/descending so
///    the answer LLM sees the earliest or most-recent fact in lead
///    position even when retrieval ranking would have buried it.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemporalFrame {
    /// "where do I live now" — filter to current state
    #[default]
    Current,
    /// "where did I used to live" / "all the places I've worked"
    /// — include historical state, no ordering preference
    Historical,
    /// "which pet did I get first" / "originally" / "最早" / "auparavant"
    /// — include historical state, sort by valid_from ASC
    First,
    /// "what was my most recent job" / "last city I lived in"
    /// — include historical state, sort by valid_from DESC. Distinct from
    /// `Current` because the asked-about state may itself be already ended
    /// (e.g. "last job" when currently unemployed).
    Last,
    /// "how has my routine changed" — compare across time
    Comparative,
    /// "who is my sister" — time-independent facts
    Timeless,
}

/// Response parsed from the LLM's JSON output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmHintResponse {
    pub structure_hint: StructureHint,
    pub intent_hint: IntentHint,
    #[serde(default)]
    pub temporal_frame: TemporalFrame,
}

const SYSTEM_PROMPT: &str = concat!(
    "You classify graph database queries. Output strict JSON with exactly three fields:\n",
    "- \"structure_hint\": one of \"ledger\", \"tree\", \"state_machine\", \"preference_list\", \"generic_graph\"\n",
    "- \"intent_hint\": one of \"balance\", \"aggregate_balance\", \"current_state\", \"children\", \"ranking\", ",
    "\"path\", \"neighbors\", \"similarity\", \"aggregate\", \"temporal\", \"subgraph\", \"knowledge\", \"unknown\"\n",
    "- \"temporal_frame\": one of \"current\", \"historical\", \"first\", \"last\", \"comparative\", \"timeless\"\n\n",
    "Classify the temporal frame by the *semantics* of the question, not by surface keywords. ",
    "Questions in any language (English, French, Chinese, Spanish, German, …) map to these six frames ",
    "based on what's being asked:\n",
    "  - \"current\": asks about the present, currently-true state.\n",
    "      English: \"where do I live now\", \"what's my job\", \"what's my routine\"\n",
    "      French:  \"où est-ce que j'habite\", \"quel est mon travail\"\n",
    "      Chinese: \"我现在住在哪里\", \"我的工作是什么\"\n",
    "  - \"historical\": asks about past state without ordering preference.\n",
    "      English: \"where did I used to live\", \"what was my old job\", \"all the cities I've lived in\"\n",
    "      French:  \"où vivais-je avant\", \"quels sont tous les endroits où j'ai vécu\"\n",
    "      Chinese: \"我以前住在哪里\", \"我住过哪些城市\"\n",
    "  - \"first\": asks for the EARLIEST occurrence of something. Includes history and orders chronologically.\n",
    "      English: \"which pet did I get first\", \"originally where did I work\", \"my first apartment\"\n",
    "      French:  \"quel animal ai-je adopté en premier\", \"à l'origine où travaillais-je\"\n",
    "      Chinese: \"我最早领养的宠物是什么\", \"我最先在哪里工作\"\n",
    "      Spanish: \"qué mascota adopté primero\", \"originalmente dónde trabajaba\"\n",
    "      German:  \"welches Haustier hatte ich zuerst\", \"wo habe ich ursprünglich gearbeitet\"\n",
    "  - \"last\": asks for the MOST-RECENT or final occurrence of something (not necessarily currently active).\n",
    "      English: \"what was my last job\", \"the most recent city I lived in\", \"last pet I adopted\"\n",
    "      French:  \"quel était mon dernier emploi\", \"la dernière ville où j'ai vécu\"\n",
    "      Chinese: \"我最后一份工作是什么\", \"我最近住过的城市\"\n",
    "  - \"comparative\": asks how something has changed across time.\n",
    "      English: \"how has my routine changed\", \"compared to before\"\n",
    "      Chinese: \"我的日常有什么变化\"\n",
    "  - \"timeless\": asks about time-independent facts (identity, family, immutable attributes).\n",
    "      English: \"who is my sister\", \"what's my blood type\", \"when was I born\"\n",
    "      French:  \"qui est ma sœur\", \"quel est mon groupe sanguin\"\n\n",
    "Use \"balance\" for pairwise queries between two specific entities (e.g. \"balance between Alice and Bob\").\n",
    "Use \"aggregate_balance\" for queries across ALL entities (e.g. \"who owes the most\", \"show all balances\", \"biggest debtor\").\n",
    "No markdown fences, no explanation, no other fields.",
);

/// Classify a question into structure + intent hints using the unified LLM client.
///
/// Returns `Ok(None)` if the LLM response cannot be parsed.
pub async fn classify_with_llm(
    client: &dyn LlmClient,
    question: &str,
) -> anyhow::Result<Option<LlmHintResponse>> {
    let request = LlmRequest {
        system_prompt: SYSTEM_PROMPT.to_string(),
        user_prompt: question.to_string(),
        temperature: 0.0,
        max_tokens: 128,
        json_mode: true,
    };

    let response = client.complete(request).await?;
    Ok(parse_hint_response(&response.content))
}

/// Parse LLM response text into a typed hint.
///
/// Strips markdown fences if present. Returns `None` on any parse error.
pub fn parse_hint_response(text: &str) -> Option<LlmHintResponse> {
    // Reuse the unified JSON parser, then try to deserialize
    let value = parse_json_from_llm(text)?;
    serde_json::from_value::<LlmHintResponse>(value).ok()
}

/// Rule-based fallback for temporal frame detection when LLM is unavailable.
///
/// Uses English keyword patterns to classify the temporal intent of a
/// question. **This is a fallback only** — the LLM classifier above is
/// authoritative and works in any language. Reaching this function means
/// the LLM client was unavailable or timed out; correct multilingual
/// classification is the LLM's job.
pub fn detect_temporal_frame(question: &str) -> TemporalFrame {
    let q = question.to_lowercase();

    // First-occurrence patterns. Checked BEFORE Historical because both
    // include past state, but First also orders chronologically.
    // The " first" / " 1st" patterns deliberately omit a trailing space so
    // they match "first?", "first.", "first," etc. at sentence ends.
    if q.contains(" first")
        || q.starts_with("first")
        || q.contains("originally")
        || q.contains("at first")
        || q.contains("initially")
        || q.contains("earliest")
        || q.contains("oldest")
        || q.contains(" 1st")
    {
        return TemporalFrame::First;
    }

    // Last-occurrence patterns. Distinct from Current — could refer to a
    // recent past state that's already ended (e.g. "my last job" when
    // currently unemployed).
    if q.contains("most recent")
        || q.contains("latest")
        || q.contains("last job")
        || q.contains("last city")
        || q.contains("last pet")
        || q.contains("last place")
        || q.contains("last apartment")
        || q.contains("last home")
        || q.contains("the last one")
    {
        return TemporalFrame::Last;
    }

    // Historical patterns
    if q.contains("used to")
        || q.contains("when i lived in")
        || q.contains("back when")
        || q.contains("before i moved")
        || q.contains("did i")
        || q.contains("previously")
        || q.contains("in the past")
        || q.contains("old job")
        || q.contains("former")
        || q.contains("all my")
        || q.contains("every city")
        || q.contains("every place")
        || q.contains("all the places")
        || q.contains("all the cities")
        || q.contains("where did i live")
        || q.contains("where have i lived")
        || q.contains("1st city")
        || q.contains("2nd city")
        || q.contains("3rd city")
        || q.contains("first city")
        || q.contains("second city")
        || q.contains("third city")
        || q.contains("previous city")
        || q.contains("previous location")
        || q.contains("ago")
        || q.contains("last week")
        || q.contains("last month")
        || q.contains("yesterday")
        || q.contains("days ago")
        || q.contains("weeks ago")
        || q.contains("what happened")
        || q.contains("history")
        || q.contains("timeline")
    {
        return TemporalFrame::Historical;
    }

    // Comparative patterns
    if q.contains("how has")
        || q.contains("changed")
        || q.contains("compared to")
        || q.contains("over time")
        || q.contains("difference between")
        || q.contains("evolution of")
    {
        return TemporalFrame::Comparative;
    }

    // Timeless patterns (identity, family, immutable facts)
    if q.contains("who is my")
        || q.contains("birthday")
        || q.contains("blood type")
        || q.contains("sister")
        || q.contains("brother")
        || q.contains("parent")
        || q.contains("mother")
        || q.contains("father")
        || q.contains("maiden name")
        || q.contains("born")
    {
        return TemporalFrame::Timeless;
    }

    TemporalFrame::Current
}

/// Map an LLM hint to a `QueryIntent`.
///
/// Structured memory hints take precedence over intent_hint.
pub fn intent_from_hint(hint: &LlmHintResponse) -> QueryIntent {
    // First check if it's a structured memory hint
    if let Some(structured) = structured_intent_from_hint(&hint.structure_hint) {
        return structured;
    }

    // Otherwise map intent_hint to a generic QueryIntent
    match hint.intent_hint {
        IntentHint::Balance => QueryIntent::StructuredMemoryQuery {
            query_type: StructuredQueryType::LedgerBalance,
        },
        IntentHint::AggregateBalance => QueryIntent::StructuredMemoryQuery {
            query_type: StructuredQueryType::AggregateBalance,
        },
        IntentHint::CurrentState => QueryIntent::StructuredMemoryQuery {
            query_type: StructuredQueryType::CurrentState,
        },
        IntentHint::Children => QueryIntent::StructuredMemoryQuery {
            query_type: StructuredQueryType::TreeChildren,
        },
        IntentHint::Ranking => QueryIntent::Ranking {
            metric: crate::nlq::intent::RankingMetric::PageRank,
        },
        IntentHint::Path => QueryIntent::FindPath {
            algorithm: crate::nlq::intent::PathAlgorithm::Shortest,
        },
        IntentHint::Neighbors => QueryIntent::FindNeighbors {
            direction: crate::structures::Direction::Both,
            edge_hint: None,
        },
        IntentHint::Similarity => QueryIntent::SimilaritySearch,
        IntentHint::Aggregate => QueryIntent::Aggregate {
            metric: crate::nlq::intent::AggregateMetric::Stats,
        },
        IntentHint::Temporal => QueryIntent::TemporalChain {
            direction: crate::nlq::intent::TemporalDirection::After,
        },
        IntentHint::Subgraph => QueryIntent::Subgraph { radius: 2 },
        IntentHint::Knowledge => QueryIntent::KnowledgeQuery,
        IntentHint::Unknown => QueryIntent::Unknown,
    }
}

/// Map a structure hint to a structured memory `QueryIntent`.
///
/// Returns `None` for `GenericGraph` (not a structured memory type).
/// Note: For Ledger, defaults to LedgerBalance. The caller should check
/// `intent_hint` for `AggregateBalance` to distinguish pairwise vs aggregate.
pub fn structured_intent_from_hint(hint: &StructureHint) -> Option<QueryIntent> {
    match hint {
        StructureHint::Ledger => Some(QueryIntent::StructuredMemoryQuery {
            query_type: StructuredQueryType::LedgerBalance,
        }),
        StructureHint::Tree => Some(QueryIntent::StructuredMemoryQuery {
            query_type: StructuredQueryType::TreeChildren,
        }),
        StructureHint::StateMachine => Some(QueryIntent::StructuredMemoryQuery {
            query_type: StructuredQueryType::CurrentState,
        }),
        StructureHint::PreferenceList => Some(QueryIntent::StructuredMemoryQuery {
            query_type: StructuredQueryType::PreferenceRanking,
        }),
        StructureHint::GenericGraph => None,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_valid_json() {
        let json = r#"{"structure_hint": "ledger", "intent_hint": "balance"}"#;
        let result = parse_hint_response(json);
        assert!(result.is_some());
        let hint = result.unwrap();
        assert_eq!(hint.structure_hint, StructureHint::Ledger);
        assert_eq!(hint.intent_hint, IntentHint::Balance);
    }

    #[test]
    fn test_parse_fenced_json() {
        let json = "```json\n{\"structure_hint\": \"tree\", \"intent_hint\": \"children\"}\n```";
        let result = parse_hint_response(json);
        assert!(result.is_some());
        let hint = result.unwrap();
        assert_eq!(hint.structure_hint, StructureHint::Tree);
        assert_eq!(hint.intent_hint, IntentHint::Children);
    }

    #[test]
    fn test_parse_bare_fences() {
        let json = "```\n{\"structure_hint\": \"generic_graph\", \"intent_hint\": \"path\"}\n```";
        let result = parse_hint_response(json);
        assert!(result.is_some());
        assert_eq!(result.unwrap().intent_hint, IntentHint::Path);
    }

    #[test]
    fn test_parse_empty_response() {
        assert!(parse_hint_response("").is_none());
        assert!(parse_hint_response("  ").is_none());
    }

    #[test]
    fn test_parse_invalid_json() {
        assert!(parse_hint_response("not json at all").is_none());
        assert!(parse_hint_response(r#"{"structure_hint": "invalid_val"}"#).is_none());
    }

    #[test]
    fn test_intent_from_hint_structured() {
        let hint = LlmHintResponse {
            structure_hint: StructureHint::StateMachine,
            intent_hint: IntentHint::CurrentState,
            temporal_frame: TemporalFrame::Current,
        };
        let intent = intent_from_hint(&hint);
        assert!(matches!(
            intent,
            QueryIntent::StructuredMemoryQuery {
                query_type: StructuredQueryType::CurrentState
            }
        ));
    }

    #[test]
    fn test_intent_from_hint_generic() {
        let hint = LlmHintResponse {
            structure_hint: StructureHint::GenericGraph,
            intent_hint: IntentHint::Path,
            temporal_frame: TemporalFrame::Current,
        };
        let intent = intent_from_hint(&hint);
        assert!(matches!(intent, QueryIntent::FindPath { .. }));
    }

    #[test]
    fn test_structured_intent_from_hint_generic_graph() {
        assert!(structured_intent_from_hint(&StructureHint::GenericGraph).is_none());
    }

    #[test]
    fn test_structured_intent_from_hint_all_types() {
        assert!(structured_intent_from_hint(&StructureHint::Ledger).is_some());
        assert!(structured_intent_from_hint(&StructureHint::Tree).is_some());
        assert!(structured_intent_from_hint(&StructureHint::StateMachine).is_some());
        assert!(structured_intent_from_hint(&StructureHint::PreferenceList).is_some());
    }

    #[test]
    fn test_parse_knowledge_hint() {
        let json = r#"{"structure_hint": "generic_graph", "intent_hint": "knowledge"}"#;
        let result = parse_hint_response(json);
        assert!(result.is_some());
        assert_eq!(result.unwrap().intent_hint, IntentHint::Knowledge);
    }

    #[test]
    fn test_intent_from_hint_knowledge() {
        let hint = LlmHintResponse {
            structure_hint: StructureHint::GenericGraph,
            intent_hint: IntentHint::Knowledge,
            temporal_frame: TemporalFrame::Current,
        };
        let intent = intent_from_hint(&hint);
        assert!(matches!(intent, QueryIntent::KnowledgeQuery));
    }

    #[test]
    fn test_parse_aggregate_balance_hint() {
        let json = r#"{"structure_hint": "ledger", "intent_hint": "aggregate_balance"}"#;
        let result = parse_hint_response(json);
        assert!(result.is_some());
        let hint = result.unwrap();
        assert_eq!(hint.structure_hint, StructureHint::Ledger);
        assert_eq!(hint.intent_hint, IntentHint::AggregateBalance);
    }

    #[test]
    fn test_temporal_frame_defaults_to_current() {
        let json = r#"{"structure_hint": "generic_graph", "intent_hint": "knowledge"}"#;
        let hint = parse_hint_response(json).unwrap();
        assert_eq!(hint.temporal_frame, TemporalFrame::Current);
    }

    #[test]
    fn test_parse_temporal_frame_from_json() {
        let json = r#"{"structure_hint": "generic_graph", "intent_hint": "knowledge", "temporal_frame": "historical"}"#;
        let hint = parse_hint_response(json).unwrap();
        assert_eq!(hint.temporal_frame, TemporalFrame::Historical);
    }

    #[test]
    fn test_detect_temporal_frame_historical() {
        assert_eq!(
            detect_temporal_frame("Where did I used to live?"),
            TemporalFrame::Historical
        );
        assert_eq!(
            detect_temporal_frame("What did I do back when I was in Tokyo?"),
            TemporalFrame::Historical
        );
        assert_eq!(
            detect_temporal_frame("Before I moved, who were my neighbors?"),
            TemporalFrame::Historical
        );
    }

    #[test]
    fn test_detect_temporal_frame_comparative() {
        assert_eq!(
            detect_temporal_frame("How has my routine changed?"),
            TemporalFrame::Comparative
        );
        assert_eq!(
            detect_temporal_frame("Compared to before, what's different?"),
            TemporalFrame::Comparative
        );
        assert_eq!(
            detect_temporal_frame("How have things changed over time?"),
            TemporalFrame::Comparative
        );
    }

    #[test]
    fn test_detect_temporal_frame_timeless() {
        assert_eq!(
            detect_temporal_frame("Who is my sister?"),
            TemporalFrame::Timeless
        );
        assert_eq!(
            detect_temporal_frame("When is my birthday?"),
            TemporalFrame::Timeless
        );
        assert_eq!(
            detect_temporal_frame("What's my blood type?"),
            TemporalFrame::Timeless
        );
    }

    #[test]
    fn test_detect_temporal_frame_current_default() {
        assert_eq!(
            detect_temporal_frame("Where do I live?"),
            TemporalFrame::Current
        );
        assert_eq!(
            detect_temporal_frame("What's my morning routine?"),
            TemporalFrame::Current
        );
    }

    // First / Last classification is intentionally NOT covered by unit
    // tests here. The production path is the LLM classifier in
    // `classify_with_llm`, which is multilingual and reads semantics;
    // a hardcoded keyword-matching test would only assert that the
    // *fallback* rules return what we encoded into them (tautological)
    // and would suggest an English-only correctness story for what is
    // actually a multilingual problem. Validation is done by running
    // the live classifier against a curated multilingual question set
    // and inspecting the `target=nlq.temporal_frame` tracing line.
}
