//! Intent types for natural language graph queries.
//!
//! Provides the `QueryIntent` enum and related types used by the NLQ pipeline.
//! The unified NLQ pipeline (multi-source RRF) handles all queries without
//! rule-based intent classification.

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
    /// Open-ended knowledge question -> triggers full multi-source retrieval
    KnowledgeQuery,
    /// Generic fallback
    Unknown,
}

/// Type of structured memory query.
#[derive(Debug, Clone)]
pub enum StructuredQueryType {
    LedgerBalance,
    /// Aggregate across ALL ledgers: "who owes the most", "all balances", "biggest debtor"
    AggregateBalance,
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
        QueryIntent::KnowledgeQuery => "KnowledgeQuery",
        QueryIntent::Unknown => "Unknown",
    }
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
