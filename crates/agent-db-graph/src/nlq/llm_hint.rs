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
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TemporalFrame {
    /// "where do I live now" — filter to current state
    Current,
    /// "where did I used to live" — include historical state
    Historical,
    /// "how has my routine changed" — compare across time
    Comparative,
    /// "who is my sister" — time-independent facts
    Timeless,
}

impl Default for TemporalFrame {
    fn default() -> Self {
        Self::Current
    }
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
    "- \"temporal_frame\": one of \"current\", \"historical\", \"comparative\", \"timeless\"\n",
    "  - \"current\": query asks about present state (\"where do I live now\", \"what's my routine\")\n",
    "  - \"historical\": query asks about past state (\"where did I used to live\", \"what was my old job\")\n",
    "  - \"comparative\": query compares across time (\"how has my routine changed\", \"compared to before\")\n",
    "  - \"timeless\": query asks about time-independent facts (\"who is my sister\", \"what's my blood type\")\n\n",
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
/// Uses keyword patterns to classify the temporal intent of a question.
pub fn detect_temporal_frame(question: &str) -> TemporalFrame {
    let q = question.to_lowercase();

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
}
