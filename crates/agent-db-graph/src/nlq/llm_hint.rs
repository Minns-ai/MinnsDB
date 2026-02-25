//! LLM hint classifier for NLQ intent routing.
//!
//! Provides an async LLM advisory classifier that can override the rule-based
//! intent classifier when it detects structured memory types or when rules
//! return Unknown.

use crate::nlq::intent::{ClassifiedIntent, QueryIntent, StructuredQueryType};
use async_trait::async_trait;
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
    Balance,
    CurrentState,
    Children,
    Ranking,
    Path,
    Neighbors,
    Similarity,
    Aggregate,
    Temporal,
    Subgraph,
    Unknown,
}

/// Response parsed from the LLM's JSON output.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmHintResponse {
    pub structure_hint: StructureHint,
    pub intent_hint: IntentHint,
}

/// Result of merging rule-based classification with LLM hint.
#[derive(Debug, Clone)]
pub struct MergedClassification {
    pub intent: ClassifiedIntent,
    pub llm_overrode: bool,
    pub raw_hint: Option<LlmHintResponse>,
}

/// Trait for LLM hint clients.
#[async_trait]
pub trait NlqHintClient: Send + Sync {
    /// Classify a question into structure + intent hints.
    /// Returns `Ok(None)` if the LLM response cannot be parsed.
    async fn classify(&self, question: &str) -> anyhow::Result<Option<LlmHintResponse>>;
}

const SYSTEM_PROMPT: &str = concat!(
    "You classify graph database queries. Output strict JSON with exactly two fields:\n",
    "- \"structure_hint\": one of \"ledger\", \"tree\", \"state_machine\", \"preference_list\", \"generic_graph\"\n",
    "- \"intent_hint\": one of \"balance\", \"current_state\", \"children\", \"ranking\", ",
    "\"path\", \"neighbors\", \"similarity\", \"aggregate\", \"temporal\", \"subgraph\", \"unknown\"\n",
    "No markdown fences, no explanation, no other fields.",
);

// ────────── OpenAI client ──────────

pub struct OpenAiHintClient {
    api_key: String,
    model: String,
    http: reqwest::Client,
}

impl OpenAiHintClient {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            api_key,
            model,
            http: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl NlqHintClient for OpenAiHintClient {
    async fn classify(&self, question: &str) -> anyhow::Result<Option<LlmHintResponse>> {
        let body = serde_json::json!({
            "model": self.model,
            "temperature": 0.0,
            "max_tokens": 128,
            "response_format": { "type": "json_object" },
            "messages": [
                { "role": "system", "content": SYSTEM_PROMPT },
                { "role": "user", "content": question }
            ]
        });

        let resp = self
            .http
            .post("https://api.openai.com/v1/chat/completions")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .json(&body)
            .send()
            .await?;

        let json: serde_json::Value = resp.json().await?;
        let text = json["choices"][0]["message"]["content"]
            .as_str()
            .unwrap_or("");

        Ok(parse_hint_response(text))
    }
}

// ────────── Anthropic client ──────────

pub struct AnthropicHintClient {
    api_key: String,
    model: String,
    http: reqwest::Client,
}

impl AnthropicHintClient {
    pub fn new(api_key: String, model: String) -> Self {
        Self {
            api_key,
            model,
            http: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl NlqHintClient for AnthropicHintClient {
    async fn classify(&self, question: &str) -> anyhow::Result<Option<LlmHintResponse>> {
        let body = serde_json::json!({
            "model": self.model,
            "max_tokens": 128,
            "system": SYSTEM_PROMPT,
            "messages": [
                { "role": "user", "content": question }
            ]
        });

        let resp = self
            .http
            .post("https://api.anthropic.com/v1/messages")
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await?;

        let json: serde_json::Value = resp.json().await?;
        let text = json["content"][0]["text"].as_str().unwrap_or("");

        Ok(parse_hint_response(text))
    }
}

// ────────── Parsing ──────────

/// Parse LLM response text into a typed hint.
///
/// Strips markdown fences if present. Returns `None` on any parse error.
pub fn parse_hint_response(text: &str) -> Option<LlmHintResponse> {
    let trimmed = text.trim();
    if trimmed.is_empty() {
        return None;
    }

    // Strip markdown fences
    let json_str = if trimmed.starts_with("```") {
        trimmed
            .trim_start_matches("```json")
            .trim_start_matches("```")
            .trim_end_matches("```")
            .trim()
    } else {
        trimmed
    };

    serde_json::from_str::<LlmHintResponse>(json_str).ok()
}

// ────────── Merge logic ──────────

/// Merge rule-based classification with LLM hint.
///
/// Rules (purely categorical, no confidence scores):
/// 1. No LLM hint → keep rule-based
/// 2. Rule returns Unknown → use LLM's categorical hint
/// 3. LLM says structured memory type (not GenericGraph) AND rules didn't → override
/// 4. Both agree → keep rule-based (has more detail)
/// 5. Disagree on generic intents → keep rule-based (deterministic wins)
pub fn merge_classification(
    rule_based: ClassifiedIntent,
    llm_hint: Option<LlmHintResponse>,
) -> MergedClassification {
    // Rule 1
    let hint = match llm_hint {
        None => {
            return MergedClassification {
                intent: rule_based,
                llm_overrode: false,
                raw_hint: None,
            }
        },
        Some(h) => h,
    };

    // Rule 2: Rules returned Unknown → use LLM
    if matches!(rule_based.intent, QueryIntent::Unknown) {
        let intent = intent_from_hint(&hint);
        return MergedClassification {
            intent: ClassifiedIntent {
                intent,
                negated: rule_based.negated,
            },
            llm_overrode: true,
            raw_hint: Some(hint),
        };
    }

    // Rule 3: LLM says structured memory AND rules didn't
    if !matches!(hint.structure_hint, StructureHint::GenericGraph)
        && !matches!(rule_based.intent, QueryIntent::StructuredMemoryQuery { .. })
    {
        if let Some(intent) = structured_intent_from_hint(&hint.structure_hint) {
            return MergedClassification {
                intent: ClassifiedIntent {
                    intent,
                    negated: rule_based.negated,
                },
                llm_overrode: true,
                raw_hint: Some(hint),
            };
        }
    }

    // Rules 4 & 5: keep rule-based
    MergedClassification {
        intent: rule_based,
        llm_overrode: false,
        raw_hint: Some(hint),
    }
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
        IntentHint::Unknown => QueryIntent::Unknown,
    }
}

/// Map a structure hint to a structured memory `QueryIntent`.
///
/// Returns `None` for `GenericGraph` (not a structured memory type).
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

/// Check if two intents agree on the same broad category.
pub fn intents_agree(rule_based: &QueryIntent, hint: &IntentHint) -> bool {
    matches!(
        (rule_based, hint),
        (QueryIntent::FindNeighbors { .. }, IntentHint::Neighbors)
            | (QueryIntent::FindPath { .. }, IntentHint::Path)
            | (QueryIntent::SimilaritySearch, IntentHint::Similarity)
            | (QueryIntent::Ranking { .. }, IntentHint::Ranking)
            | (QueryIntent::Aggregate { .. }, IntentHint::Aggregate)
            | (QueryIntent::TemporalChain { .. }, IntentHint::Temporal)
            | (QueryIntent::Subgraph { .. }, IntentHint::Subgraph)
            | (
                QueryIntent::StructuredMemoryQuery {
                    query_type: StructuredQueryType::LedgerBalance
                },
                IntentHint::Balance
            )
            | (
                QueryIntent::StructuredMemoryQuery {
                    query_type: StructuredQueryType::CurrentState
                },
                IntentHint::CurrentState
            )
            | (
                QueryIntent::StructuredMemoryQuery {
                    query_type: StructuredQueryType::TreeChildren
                },
                IntentHint::Children
            )
            | (
                QueryIntent::StructuredMemoryQuery {
                    query_type: StructuredQueryType::PreferenceRanking
                },
                IntentHint::Ranking
            )
            | (QueryIntent::Unknown, IntentHint::Unknown)
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nlq::intent::RankingMetric;
    use crate::structures::Direction;

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
    fn test_merge_no_hint() {
        let rule = ClassifiedIntent {
            intent: QueryIntent::FindNeighbors {
                direction: Direction::Both,
                edge_hint: None,
            },
            negated: false,
        };
        let merged = merge_classification(rule, None);
        assert!(!merged.llm_overrode);
        assert!(merged.raw_hint.is_none());
        assert!(matches!(
            merged.intent.intent,
            QueryIntent::FindNeighbors { .. }
        ));
    }

    #[test]
    fn test_merge_unknown_uses_llm() {
        let rule = ClassifiedIntent {
            intent: QueryIntent::Unknown,
            negated: false,
        };
        let hint = LlmHintResponse {
            structure_hint: StructureHint::GenericGraph,
            intent_hint: IntentHint::Neighbors,
        };
        let merged = merge_classification(rule, Some(hint));
        assert!(merged.llm_overrode);
        assert!(matches!(
            merged.intent.intent,
            QueryIntent::FindNeighbors { .. }
        ));
    }

    #[test]
    fn test_merge_llm_structured_override() {
        let rule = ClassifiedIntent {
            intent: QueryIntent::FindNeighbors {
                direction: Direction::Both,
                edge_hint: None,
            },
            negated: false,
        };
        let hint = LlmHintResponse {
            structure_hint: StructureHint::Ledger,
            intent_hint: IntentHint::Balance,
        };
        let merged = merge_classification(rule, Some(hint));
        assert!(merged.llm_overrode);
        assert!(matches!(
            merged.intent.intent,
            QueryIntent::StructuredMemoryQuery {
                query_type: StructuredQueryType::LedgerBalance
            }
        ));
    }

    #[test]
    fn test_merge_both_agree_keeps_rules() {
        let rule = ClassifiedIntent {
            intent: QueryIntent::FindNeighbors {
                direction: Direction::Out,
                edge_hint: Some("knows".to_string()),
            },
            negated: false,
        };
        let hint = LlmHintResponse {
            structure_hint: StructureHint::GenericGraph,
            intent_hint: IntentHint::Neighbors,
        };
        let merged = merge_classification(rule, Some(hint));
        assert!(!merged.llm_overrode);
        // Rule-based kept with more detail (Outgoing direction, edge_hint)
        if let QueryIntent::FindNeighbors {
            direction,
            edge_hint,
        } = &merged.intent.intent
        {
            assert!(matches!(direction, Direction::Out));
            assert_eq!(edge_hint.as_deref(), Some("knows"));
        } else {
            panic!("Expected FindNeighbors");
        }
    }

    #[test]
    fn test_merge_disagree_keeps_rules() {
        let rule = ClassifiedIntent {
            intent: QueryIntent::Ranking {
                metric: RankingMetric::PageRank,
            },
            negated: false,
        };
        let hint = LlmHintResponse {
            structure_hint: StructureHint::GenericGraph,
            intent_hint: IntentHint::Neighbors,
        };
        let merged = merge_classification(rule, Some(hint));
        assert!(!merged.llm_overrode);
        assert!(matches!(merged.intent.intent, QueryIntent::Ranking { .. }));
    }

    #[test]
    fn test_merge_preserves_negation() {
        let rule = ClassifiedIntent {
            intent: QueryIntent::Unknown,
            negated: true,
        };
        let hint = LlmHintResponse {
            structure_hint: StructureHint::Ledger,
            intent_hint: IntentHint::Balance,
        };
        let merged = merge_classification(rule, Some(hint));
        assert!(merged.llm_overrode);
        assert!(merged.intent.negated);
    }

    #[test]
    fn test_intent_from_hint_structured() {
        let hint = LlmHintResponse {
            structure_hint: StructureHint::StateMachine,
            intent_hint: IntentHint::CurrentState,
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
    fn test_intents_agree() {
        assert!(intents_agree(
            &QueryIntent::FindNeighbors {
                direction: Direction::Both,
                edge_hint: None,
            },
            &IntentHint::Neighbors,
        ));
        assert!(!intents_agree(
            &QueryIntent::FindNeighbors {
                direction: Direction::Both,
                edge_hint: None,
            },
            &IntentHint::Path,
        ));
        assert!(intents_agree(&QueryIntent::Unknown, &IntentHint::Unknown));
    }
}
