//! Rule-based intent classification for natural language graph queries.

use crate::structures::Direction;

/// Classified intent of a natural language query.
#[derive(Debug, Clone)]
pub enum QueryIntent {
    /// "Who does Alice owe?" -> find neighbors
    FindNeighbors {
        direction: Direction,
        edge_hint: Option<String>,
    },
    /// "Shortest path from A to B" -> pathfinding
    FindPath { algorithm: PathAlgorithm },
    /// "What tools did agent X use?" -> filtered traversal
    FilteredTraversal {
        node_type_filter: Option<String>,
        edge_type_filter: Option<String>,
    },
    /// "Show me everything about X" -> subgraph extraction
    Subgraph { radius: u32 },
    /// "What happened after event X?" -> temporal/causal chain
    TemporalChain { direction: TemporalDirection },
    /// "What are the most important nodes?" -> ranking
    Ranking { metric: RankingMetric },
    /// "Find similar to X" -> similarity search
    SimilaritySearch,
    /// "How many nodes?" -> aggregation query
    Aggregate { metric: AggregateMetric },
    /// "What is the balance between Alice and Bob?" -> structured memory lookup
    StructuredMemoryQuery { query_type: StructuredQueryType },
    /// Generic fallback
    Unknown,
}

/// Type of structured memory query.
#[derive(Debug, Clone)]
pub enum StructuredQueryType {
    LedgerBalance,
    CurrentState,
    PreferenceRanking,
    TreeChildren,
}

/// Aggregation metric for count/stats queries.
#[derive(Debug, Clone)]
pub enum AggregateMetric {
    NodeCount,
    EdgeCount,
    CountByType(String),
    AverageDegree,
    Stats,
    /// Sum a numeric property across nodes (or edges between entities).
    SumProperty {
        property: String,
        node_type_filter: Option<String>,
    },
    /// Group nodes by a property and count per group.
    GroupByCount {
        group_property: String,
        node_type_filter: Option<String>,
    },
    /// Find min or max of a numeric property.
    MinMaxProperty {
        property: String,
        find_min: bool,
        node_type_filter: Option<String>,
    },
}

/// Intent with negation flag.
#[derive(Debug, Clone)]
pub struct ClassifiedIntent {
    pub intent: QueryIntent,
    pub negated: bool,
}

/// Pathfinding algorithm variant.
#[derive(Debug, Clone)]
pub enum PathAlgorithm {
    Shortest,
    KShortest(usize),
    Bidirectional,
}

/// Temporal direction for causal chain queries.
#[derive(Debug, Clone)]
pub enum TemporalDirection {
    After,
    Before,
    Between,
}

/// Ranking metric.
#[derive(Debug, Clone)]
pub enum RankingMetric {
    PageRank,
    Centrality,
    Degree,
}

/// Classify the intent of a natural language question.
pub fn classify_intent(question: &str) -> QueryIntent {
    let q = question.to_lowercase();
    let words: Vec<&str> = q.split_whitespace().collect();

    // Check for path queries first (most specific)
    if contains_any(
        &q,
        &[
            "shortest path",
            "path from",
            "path between",
            "route from",
            "route to",
        ],
    ) {
        let k = extract_k_value(&q);
        if let Some(k) = k {
            return QueryIntent::FindPath {
                algorithm: PathAlgorithm::KShortest(k),
            };
        }
        if contains_any(&q, &["bidirectional"]) {
            return QueryIntent::FindPath {
                algorithm: PathAlgorithm::Bidirectional,
            };
        }
        return QueryIntent::FindPath {
            algorithm: PathAlgorithm::Shortest,
        };
    }

    // K-shortest paths
    if contains_any(
        &q,
        &[
            "k shortest",
            "k paths",
            "k-shortest",
            "alternative paths",
            "multiple paths",
        ],
    ) {
        let k = extract_k_value(&q).unwrap_or(3);
        return QueryIntent::FindPath {
            algorithm: PathAlgorithm::KShortest(k),
        };
    }

    // ---- Priority 1-4: Structured memory queries (Section B) ----

    // LedgerBalance: "owe", "balance between", "net between", "ledger" + likely 2 entities
    if contains_any(&q, &["balance between", "net between", "ledger"]) {
        return QueryIntent::StructuredMemoryQuery {
            query_type: StructuredQueryType::LedgerBalance,
        };
    }
    if contains_any(&q, &["how much does.*owe", "how much do.*owe"]) {
        return QueryIntent::StructuredMemoryQuery {
            query_type: StructuredQueryType::LedgerBalance,
        };
    }

    // CurrentState: "where is.*now", "current state", "status of", "what state"
    if contains_any(&q, &["current state", "status of", "what state"]) {
        return QueryIntent::StructuredMemoryQuery {
            query_type: StructuredQueryType::CurrentState,
        };
    }
    if contains_any(&q, &["where is.*now", "where's.*now"]) {
        return QueryIntent::StructuredMemoryQuery {
            query_type: StructuredQueryType::CurrentState,
        };
    }

    // PreferenceRanking: "favorite", "preferred", "ranking of", "preference"
    if contains_any(&q, &["favorite", "preferred", "preference", "ranking of"]) {
        return QueryIntent::StructuredMemoryQuery {
            query_type: StructuredQueryType::PreferenceRanking,
        };
    }

    // TreeChildren: "children of", "reports to", "subtree", "who reports under"
    if contains_any(
        &q,
        &["children of", "reports to", "subtree", "reports under"],
    ) {
        return QueryIntent::StructuredMemoryQuery {
            query_type: StructuredQueryType::TreeChildren,
        };
    }

    // ---- Priority 5: Numeric aggregation (SumProperty, GroupByCount, MinMaxProperty) ----

    // "sum", "total of" (but NOT "total number" which is count)
    if contains_any(&q, &["sum of", "total of"]) {
        let property = extract_property_hint(&q).unwrap_or_else(|| "amount".to_string());
        let type_filter = detect_type_filter(&words);
        return QueryIntent::Aggregate {
            metric: AggregateMetric::SumProperty {
                property,
                node_type_filter: type_filter,
            },
        };
    }

    // "group by", "breakdown", "by type", "per" (as aggregation)
    if contains_any(
        &q,
        &["group by", "breakdown by", "per category", "per type"],
    ) {
        let property = extract_property_hint(&q).unwrap_or_else(|| "type".to_string());
        let type_filter = detect_type_filter(&words);
        return QueryIntent::Aggregate {
            metric: AggregateMetric::GroupByCount {
                group_property: property,
                node_type_filter: type_filter,
            },
        };
    }

    // "minimum", "maximum" (numeric aggregation, not ranking)
    // Use exact contains (not fuzzy) to distinguish min vs max,
    // since "maximum" and "minimum" are only 2 edits apart.
    if q.contains("minimum") || q.contains("maximum") {
        let find_min = q.contains("minimum");
        let property = extract_property_hint(&q).unwrap_or_else(|| "value".to_string());
        let type_filter = detect_type_filter(&words);
        return QueryIntent::Aggregate {
            metric: AggregateMetric::MinMaxProperty {
                property,
                find_min,
                node_type_filter: type_filter,
            },
        };
    }

    // ---- Priority 6: Existing aggregation (count, how many, etc.) ----

    // Aggregation queries ("how many", "count", "total") — checked early to beat
    // temporal and type-filter matches on the same words.
    if contains_any(&q, &["how many", "count", "total number"]) {
        let type_filter = detect_type_filter(&words);
        let metric = if let Some(ref type_name) = type_filter {
            AggregateMetric::CountByType(type_name.clone())
        } else if contains_any(&q, &["edge", "edges", "connections", "relationships"]) {
            AggregateMetric::EdgeCount
        } else if contains_any(&q, &["degree", "average degree"]) {
            AggregateMetric::AverageDegree
        } else if contains_any(&q, &["statistics", "stats", "overview"]) {
            AggregateMetric::Stats
        } else {
            AggregateMetric::NodeCount
        };
        return QueryIntent::Aggregate { metric };
    }

    // Temporal / causal chain
    if contains_any(
        &q,
        &[
            "after",
            "then",
            "caused",
            "followed by",
            "next",
            "consequence",
        ],
    ) {
        return QueryIntent::TemporalChain {
            direction: TemporalDirection::After,
        };
    }
    if contains_any(
        &q,
        &["before", "preceded", "prior to", "led to", "previous"],
    ) {
        return QueryIntent::TemporalChain {
            direction: TemporalDirection::Before,
        };
    }
    if contains_any(&q, &["between", "during", "from.*to"]) {
        // Check if this is a time-range query, not a path query
        if contains_any(&q, &["happened", "occurred", "events"]) {
            return QueryIntent::TemporalChain {
                direction: TemporalDirection::Between,
            };
        }
    }

    // Ranking queries (including code-specific patterns)
    if contains_any(
        &q,
        &[
            "important",
            "rank",
            "top",
            "central",
            "most connected",
            "influential",
            "highest",
            "most called",
            "most imported",
            "most used functions",
        ],
    ) {
        let metric = if contains_any(&q, &["pagerank", "page rank"]) {
            RankingMetric::PageRank
        } else if contains_any(&q, &["central", "centrality", "betweenness"]) {
            RankingMetric::Centrality
        } else if contains_any(&q, &["degree", "connected"]) {
            RankingMetric::Degree
        } else {
            RankingMetric::PageRank // default
        };
        return QueryIntent::Ranking { metric };
    }

    // Similarity search (including code-specific patterns)
    if contains_any(
        &q,
        &[
            "similar",
            "like",
            "related to",
            "resembl",
            "closest to",
            "similar functions",
            "similar code",
        ],
    ) {
        return QueryIntent::SimilaritySearch;
    }

    // Subgraph / "everything about"
    if contains_any(
        &q,
        &[
            "everything about",
            "all about",
            "detail",
            "show me",
            "subgraph",
            "neighborhood",
        ],
    ) {
        let radius = extract_number(&q).unwrap_or(2);
        return QueryIntent::Subgraph { radius };
    }

    // Code-specific filtered traversal (functions/classes/methods → Concept nodes)
    if contains_any(&q, &["functions in", "classes in", "methods of"]) {
        return QueryIntent::FilteredTraversal {
            node_type_filter: Some("Concept".to_string()),
            edge_type_filter: None,
        };
    }

    // Filtered traversal with type filters
    if let Some(type_filter) = detect_type_filter(&words) {
        return QueryIntent::FilteredTraversal {
            node_type_filter: Some(type_filter),
            edge_type_filter: None,
        };
    }

    // Neighbor queries (broad match, including code relationships)
    if contains_any(
        &q,
        &[
            "who",
            "what",
            "connected",
            "neighbor",
            "adjacent",
            "link",
            "relate",
            "owe",
            "know",
            "calls",
            "imports",
            "depends on",
            "inherits",
            "implements",
            "references",
        ],
    ) {
        let direction = if contains_any(&q, &["incoming", "who.*to", "from whom"]) {
            Direction::In
        } else if contains_any(&q, &["outgoing", "to whom", "does.*owe", "does.*connect"]) {
            Direction::Out
        } else {
            Direction::Both
        };
        let edge_hint = detect_edge_hint(&q);
        return QueryIntent::FindNeighbors {
            direction,
            edge_hint,
        };
    }

    QueryIntent::Unknown
}

/// Classify intent with negation detection.
pub fn classify_intent_full(question: &str) -> ClassifiedIntent {
    let intent = classify_intent(question);
    let negated = detect_negation(question);
    ClassifiedIntent { intent, negated }
}

/// Detect negation in question text.
fn detect_negation(question: &str) -> bool {
    let q = question.to_lowercase();
    // Match word-boundary negation markers
    for marker in &[" not ", "n't ", " excluding ", " except ", " without "] {
        if q.contains(marker) {
            return true;
        }
    }
    // Also check start of string
    if q.starts_with("not ") {
        return true;
    }
    false
}

/// Detect multi-hop patterns (e.g., "2-hop", "two-hop", "friends of friends").
pub(super) fn detect_multi_hop(q: &str) -> Option<u32> {
    let lower = q.to_lowercase();

    // Match "N-hop" or "N hop"
    for word in lower.split_whitespace() {
        let trimmed = word.trim_matches(|c: char| !c.is_alphanumeric());
        if let Some(prefix) = trimmed.strip_suffix("hop") {
            let num_part = prefix.trim_end_matches('-');
            if let Ok(n) = num_part.parse::<u32>() {
                if (2..=10).contains(&n) {
                    return Some(n);
                }
            }
        }
    }

    // Match written numbers
    let word_nums = [("two", 2u32), ("three", 3), ("four", 4), ("five", 5)];
    for (word, n) in &word_nums {
        let pattern = format!("{}-hop", word);
        if lower.contains(&pattern) {
            return Some(*n);
        }
        let pattern2 = format!("{} hop", word);
        if lower.contains(&pattern2) {
            return Some(*n);
        }
    }

    // Match "friends of friends" (count repeated neighbor words)
    let neighbor_words = ["friends", "neighbors", "neighbours", "connections"];
    for nw in &neighbor_words {
        let parts: Vec<&str> = lower.split(nw).collect();
        if parts.len() >= 3 {
            // e.g., "friends of friends" splits into 3 parts
            return Some((parts.len() - 1) as u32);
        }
    }

    None
}

/// Human-readable display name for an intent.
pub fn intent_display_name(intent: &QueryIntent) -> &'static str {
    match intent {
        QueryIntent::FindNeighbors { .. } => "FindNeighbors",
        QueryIntent::FindPath { .. } => "FindPath",
        QueryIntent::FilteredTraversal { .. } => "FilteredTraversal",
        QueryIntent::Subgraph { .. } => "Subgraph",
        QueryIntent::TemporalChain { .. } => "TemporalChain",
        QueryIntent::Ranking { .. } => "Ranking",
        QueryIntent::SimilaritySearch => "SimilaritySearch",
        QueryIntent::Aggregate { .. } => "Aggregate",
        QueryIntent::StructuredMemoryQuery { .. } => "StructuredMemoryQuery",
        QueryIntent::Unknown => "Unknown",
    }
}

/// Extract a property name hint from the question text (Section D fallback chain).
///
/// 1. Explicit: `"sum of <property>"`, `"total <property>"`, `"group by <property>"`
/// 2. Known property names
pub fn extract_property_hint(question: &str) -> Option<String> {
    let q = question.to_lowercase();

    // Explicit extraction: "sum of <prop>", "total of <prop>", "group by <prop>"
    for prefix in &["sum of ", "total of ", "minimum ", "maximum "] {
        if let Some(rest) = q
            .strip_prefix(prefix)
            .or_else(|| q.find(prefix).map(|pos| &q[pos + prefix.len()..]))
        {
            let word = rest.split_whitespace().next();
            if let Some(w) = word {
                let clean = w.trim_matches(|c: char| !c.is_alphanumeric() && c != '_');
                if !clean.is_empty() && !is_stopword(clean) {
                    return Some(clean.to_string());
                }
            }
        }
    }

    // "group by <prop>"
    if let Some(pos) = q.find("group by ") {
        let rest = &q[pos + 9..];
        if let Some(word) = rest.split_whitespace().next() {
            let clean = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '_');
            if !clean.is_empty() && !is_stopword(clean) {
                return Some(clean.to_string());
            }
        }
    }

    // "<prop> per"
    if let Some(pos) = q.find(" per ") {
        let before = &q[..pos];
        if let Some(word) = before.split_whitespace().last() {
            let clean = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '_');
            if !clean.is_empty() && !is_stopword(clean) {
                return Some(clean.to_string());
            }
        }
    }

    // Known property names (broad set)
    let known = [
        "amount", "weight", "score", "balance", "cost", "price", "value", "count", "quantity",
        "salary", "revenue", "duration", "age", "size",
    ];
    for prop in &known {
        if q.contains(prop) {
            return Some(prop.to_string());
        }
    }

    None
}

fn is_stopword(word: &str) -> bool {
    matches!(
        word,
        "the"
            | "a"
            | "an"
            | "of"
            | "in"
            | "on"
            | "at"
            | "to"
            | "for"
            | "is"
            | "are"
            | "was"
            | "were"
            | "be"
            | "by"
            | "with"
            | "from"
            | "all"
            | "each"
            | "every"
            | "and"
            | "or"
            | "not"
            | "no"
            | "that"
            | "this"
            | "it"
            | "its"
            | "they"
    )
}

/// Check if the text contains any of the given patterns.
/// Falls back to fuzzy matching (Levenshtein distance <= 2) for single-word
/// patterns of length >= 4 when exact match fails.
pub(super) fn contains_any(text: &str, patterns: &[&str]) -> bool {
    patterns.iter().any(|p| {
        if p.contains(".*") {
            // Simple regex-like pattern: split on .* and check both parts exist in order
            let parts: Vec<&str> = p.split(".*").collect();
            if parts.len() == 2 {
                if let Some(pos) = text.find(parts[0]) {
                    text[pos + parts[0].len()..].contains(parts[1])
                } else {
                    false
                }
            } else {
                text.contains(p)
            }
        } else if text.contains(p) {
            true
        } else {
            // Fuzzy fallback: only for single-word patterns >= 5 chars
            if !p.contains(' ') && p.len() >= 5 {
                // Adaptive threshold: max 1 edit for patterns < 7 chars, max 2 for >= 7
                let max_dist = if p.len() >= 7 { 2 } else { 1 };
                text.split_whitespace().any(|word| {
                    // Quick pre-filter: skip if length difference > max_dist
                    if word.len().abs_diff(p.len()) > max_dist {
                        return false;
                    }
                    levenshtein(word, p) <= max_dist
                })
            } else {
                false
            }
        }
    })
}

/// Wagner-Fischer Levenshtein edit distance.
fn levenshtein(a: &str, b: &str) -> usize {
    let a_bytes = a.as_bytes();
    let b_bytes = b.as_bytes();
    let m = a_bytes.len();
    let n = b_bytes.len();
    let mut prev = (0..=n).collect::<Vec<_>>();
    let mut curr = vec![0usize; n + 1];
    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = if a_bytes[i - 1] == b_bytes[j - 1] {
                0
            } else {
                1
            };
            curr[j] = (prev[j] + 1).min(curr[j - 1] + 1).min(prev[j - 1] + cost);
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

/// Extract a numeric K value from the question (e.g., "3 shortest paths" -> 3).
fn extract_k_value(text: &str) -> Option<usize> {
    for word in text.split_whitespace() {
        if let Ok(n) = word.parse::<usize>() {
            if (2..=10).contains(&n) {
                return Some(n);
            }
        }
    }
    None
}

/// Extract a number from text (for radius, depth, etc.)
fn extract_number(text: &str) -> Option<u32> {
    for word in text.split_whitespace() {
        if let Ok(n) = word.parse::<u32>() {
            if (1..=100).contains(&n) {
                return Some(n);
            }
        }
    }
    None
}

/// Detect node type filters from question words.
fn detect_type_filter(words: &[&str]) -> Option<String> {
    for word in words {
        match *word {
            "tools" | "tool" => return Some("Tool".to_string()),
            "strategies" | "strategy" => return Some("Strategy".to_string()),
            "memories" | "memory" => return Some("Memory".to_string()),
            "episodes" | "episode" => return Some("Episode".to_string()),
            "events" | "event" => return Some("Event".to_string()),
            "agents" | "agent" => return Some("Agent".to_string()),
            "concepts" | "concept" => return Some("Concept".to_string()),
            "goals" | "goal" => return Some("Goal".to_string()),
            "claims" | "claim" => return Some("Claim".to_string()),
            _ => {},
        }
    }
    None
}

/// Detect edge type hint from question text.
fn detect_edge_hint(text: &str) -> Option<String> {
    if contains_any(text, &["cause", "caused", "causal"]) {
        Some("Causality".to_string())
    } else if contains_any(text, &["temporal", "sequence", "time"]) {
        Some("Temporal".to_string())
    } else if contains_any(text, &["interact", "collaboration"]) {
        Some("Interaction".to_string())
    } else if contains_any(text, &["context", "contextual"]) {
        Some("Contextual".to_string())
    } else if contains_any(text, &["calls", "invokes"]) {
        Some("Calls".to_string())
    } else if contains_any(text, &["imports", "depends"]) {
        Some("Imports".to_string())
    } else if contains_any(text, &["inherits", "extends"]) {
        Some("Inherits".to_string())
    } else if contains_any(text, &["implements"]) {
        Some("Implements".to_string())
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_classify_intent_neighbors() {
        let intent = classify_intent("Who does Alice connect to?");
        assert!(matches!(intent, QueryIntent::FindNeighbors { .. }));
    }

    #[test]
    fn test_classify_intent_path() {
        let intent = classify_intent("Shortest path from A to B");
        assert!(matches!(
            intent,
            QueryIntent::FindPath {
                algorithm: PathAlgorithm::Shortest
            }
        ));
    }

    #[test]
    fn test_classify_intent_k_shortest() {
        let intent = classify_intent("Find 3 shortest paths from X to Y");
        match intent {
            QueryIntent::FindPath {
                algorithm: PathAlgorithm::KShortest(k),
            } => assert_eq!(k, 3),
            other => panic!("Expected KShortest, got {:?}", other),
        }
    }

    #[test]
    fn test_classify_intent_temporal() {
        let intent = classify_intent("What happened after event X?");
        assert!(matches!(
            intent,
            QueryIntent::TemporalChain {
                direction: TemporalDirection::After
            }
        ));
    }

    #[test]
    fn test_classify_intent_ranking() {
        let intent = classify_intent("Most important nodes");
        assert!(matches!(intent, QueryIntent::Ranking { .. }));
    }

    #[test]
    fn test_classify_intent_similarity() {
        let intent = classify_intent("Find nodes similar to X");
        assert!(matches!(intent, QueryIntent::SimilaritySearch));
    }

    #[test]
    fn test_classify_intent_subgraph() {
        let intent = classify_intent("Show me everything about Alice");
        assert!(matches!(intent, QueryIntent::Subgraph { .. }));
    }

    #[test]
    fn test_classify_intent_filtered() {
        let intent = classify_intent("What tools did agent 1 use?");
        assert!(matches!(
            intent,
            QueryIntent::FilteredTraversal {
                node_type_filter: Some(_),
                ..
            }
        ));
    }

    #[test]
    fn test_classify_intent_unknown() {
        let intent = classify_intent("Hello world");
        assert!(matches!(intent, QueryIntent::Unknown));
    }

    // Enhancement 1: Fuzzy/typo-tolerant tests
    #[test]
    fn test_fuzzy_intent_british_spelling() {
        // "neighbours" should fuzzy-match "neighbor"
        let intent = classify_intent("List neighbours of Alice");
        assert!(matches!(intent, QueryIntent::FindNeighbors { .. }));
    }

    #[test]
    fn test_fuzzy_intent_typo() {
        // "conected" should fuzzy-match "connected"
        let intent = classify_intent("Nodes conected to Alice");
        assert!(matches!(intent, QueryIntent::FindNeighbors { .. }));
    }

    #[test]
    fn test_fuzzy_intent_shortest_typo() {
        // "shortes" should fuzzy-match "shortest"
        let intent = classify_intent("shortes path from A to B");
        assert!(matches!(intent, QueryIntent::FindPath { .. }));
    }

    // Enhancement 3: Aggregation tests
    #[test]
    fn test_classify_intent_aggregate_count() {
        let intent = classify_intent("How many nodes are there?");
        assert!(matches!(
            intent,
            QueryIntent::Aggregate {
                metric: AggregateMetric::NodeCount
            }
        ));
    }

    #[test]
    fn test_classify_intent_aggregate_count_by_type() {
        let intent = classify_intent("How many strategies exist?");
        assert!(matches!(
            intent,
            QueryIntent::Aggregate {
                metric: AggregateMetric::CountByType(_)
            }
        ));
    }

    #[test]
    fn test_classify_intent_aggregate_edges() {
        let intent = classify_intent("Count all edges");
        assert!(matches!(
            intent,
            QueryIntent::Aggregate {
                metric: AggregateMetric::EdgeCount
            }
        ));
    }

    // Enhancement 5: Negation tests
    #[test]
    fn test_detect_negation_not() {
        let ci = classify_intent_full("Who is not connected to Alice?");
        assert!(ci.negated);
    }

    #[test]
    fn test_detect_negation_excluding() {
        let ci = classify_intent_full("Neighbors of Alice excluding Bob");
        assert!(ci.negated);
    }

    #[test]
    fn test_detect_negation_none() {
        let ci = classify_intent_full("Who does Alice know?");
        assert!(!ci.negated);
    }

    // Enhancement 2: Multi-hop detection tests
    #[test]
    fn test_detect_multi_hop_numeric() {
        assert_eq!(detect_multi_hop("2-hop neighbors"), Some(2));
    }

    #[test]
    fn test_detect_multi_hop_written() {
        assert_eq!(detect_multi_hop("three-hop connections"), Some(3));
    }

    #[test]
    fn test_detect_multi_hop_friends_of_friends() {
        assert_eq!(detect_multi_hop("friends of friends of Alice"), Some(2));
    }

    #[test]
    fn test_detect_multi_hop_none() {
        assert_eq!(detect_multi_hop("neighbors of Alice"), None);
    }

    // ===== Structured memory query intent tests =====

    #[test]
    fn test_intent_ledger_balance() {
        let intent = classify_intent("What is the balance between Alice and Bob?");
        assert!(matches!(
            intent,
            QueryIntent::StructuredMemoryQuery {
                query_type: StructuredQueryType::LedgerBalance
            }
        ));
    }

    #[test]
    fn test_intent_how_much_owe() {
        let intent = classify_intent("How much does Alice owe Bob?");
        assert!(matches!(
            intent,
            QueryIntent::StructuredMemoryQuery {
                query_type: StructuredQueryType::LedgerBalance
            }
        ));
    }

    #[test]
    fn test_intent_current_state() {
        let intent = classify_intent("What is the current state of package X?");
        assert!(matches!(
            intent,
            QueryIntent::StructuredMemoryQuery {
                query_type: StructuredQueryType::CurrentState
            }
        ));
    }

    #[test]
    fn test_intent_where_is_now() {
        let intent = classify_intent("Where is package X now?");
        assert!(matches!(
            intent,
            QueryIntent::StructuredMemoryQuery {
                query_type: StructuredQueryType::CurrentState
            }
        ));
    }

    #[test]
    fn test_intent_preference_ranking() {
        let intent = classify_intent("What is Alice's favorite food?");
        assert!(matches!(
            intent,
            QueryIntent::StructuredMemoryQuery {
                query_type: StructuredQueryType::PreferenceRanking
            }
        ));
    }

    #[test]
    fn test_intent_tree_children() {
        let intent = classify_intent("Who are the children of the CEO?");
        assert!(matches!(
            intent,
            QueryIntent::StructuredMemoryQuery {
                query_type: StructuredQueryType::TreeChildren
            }
        ));
    }

    // ===== Numeric aggregation intent tests =====

    #[test]
    fn test_intent_sum_property() {
        let intent = classify_intent("sum of cost across all nodes");
        assert!(matches!(
            intent,
            QueryIntent::Aggregate {
                metric: AggregateMetric::SumProperty { .. }
            }
        ));
    }

    #[test]
    fn test_intent_group_by() {
        let intent = classify_intent("group by type for all nodes");
        assert!(matches!(
            intent,
            QueryIntent::Aggregate {
                metric: AggregateMetric::GroupByCount { .. }
            }
        ));
    }

    #[test]
    fn test_intent_min_max() {
        let intent = classify_intent("What is the maximum score?");
        assert!(matches!(
            intent,
            QueryIntent::Aggregate {
                metric: AggregateMetric::MinMaxProperty {
                    find_min: false,
                    ..
                }
            }
        ));

        let intent = classify_intent("Find the minimum cost");
        assert!(matches!(
            intent,
            QueryIntent::Aggregate {
                metric: AggregateMetric::MinMaxProperty { find_min: true, .. }
            }
        ));
    }

    // ===== Property extraction tests =====

    #[test]
    fn test_extract_property_explicit() {
        assert_eq!(
            extract_property_hint("sum of cost"),
            Some("cost".to_string())
        );
        assert_eq!(
            extract_property_hint("total of revenue"),
            Some("revenue".to_string())
        );
    }

    #[test]
    fn test_extract_property_known() {
        assert_eq!(
            extract_property_hint("show me the amount"),
            Some("amount".to_string())
        );
    }

    #[test]
    fn test_extract_property_none() {
        assert_eq!(extract_property_hint("hello world"), None);
    }

    // ===== Dispatch priority test =====

    #[test]
    fn test_dispatch_priority_owe_goes_to_ledger_not_aggregation() {
        // "How much does Alice owe Bob?" should go to LedgerBalance, NOT SumProperty
        let intent = classify_intent("How much does Alice owe Bob?");
        assert!(
            matches!(
                intent,
                QueryIntent::StructuredMemoryQuery {
                    query_type: StructuredQueryType::LedgerBalance
                }
            ),
            "Expected LedgerBalance, got {:?}",
            intent
        );
    }
}
