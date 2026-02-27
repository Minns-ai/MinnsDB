//! Community-enriched context injection for formation pipelines.
//!
//! Provides CPU-only helpers that score community summaries and format
//! context strings for injection into LLM prompts. No async, no LLM calls.

use crate::community_summary::CommunitySummary;
use crate::strategies::Strategy;
use std::collections::{HashMap, HashSet};

/// Configuration for context enrichment injection.
#[derive(Debug, Clone)]
pub struct EnrichmentConfig {
    /// Maximum communities to include in context. Default: 3.
    pub max_communities: usize,
    /// Maximum characters for community context string. Default: 2000.
    pub max_context_chars: usize,
    /// Maximum similar strategies to include. Default: 3.
    pub max_similar_strategies: usize,
    /// Maximum similar playbooks to include. Default: 3.
    pub max_similar_playbooks: usize,
}

impl Default for EnrichmentConfig {
    fn default() -> Self {
        Self {
            max_communities: 3,
            max_context_chars: 2000,
            max_similar_strategies: 3,
            max_similar_playbooks: 3,
        }
    }
}

/// Score communities by keyword overlap with a topic string.
///
/// Uses the same tokenization as DRIFT primer: lowercase, split on whitespace,
/// filter tokens >= 3 chars, compute overlap fraction.
///
/// Returns `Vec<(community_id, score)>` sorted descending, capped at `max`.
pub fn score_communities(
    topic: &str,
    summaries: &HashMap<u64, CommunitySummary>,
    max: usize,
) -> Vec<(u64, f32)> {
    if summaries.is_empty() || topic.is_empty() {
        return Vec::new();
    }

    let query_tokens: HashSet<String> = topic
        .to_lowercase()
        .split_whitespace()
        .filter(|w| w.len() >= 3)
        .map(|s| s.to_string())
        .collect();

    if query_tokens.is_empty() {
        return Vec::new();
    }

    let mut scored: Vec<(u64, f32)> = summaries
        .iter()
        .filter_map(|(&id, summary)| {
            let summary_lower = summary.summary.to_lowercase();
            let entity_text = summary
                .key_entities
                .iter()
                .map(|e| e.to_lowercase())
                .collect::<Vec<_>>()
                .join(" ");
            let combined = format!("{} {}", summary_lower, entity_text);

            let overlap = query_tokens
                .iter()
                .filter(|t| combined.contains(t.as_str()))
                .count();

            if overlap > 0 {
                let score = overlap as f32 / query_tokens.len() as f32;
                Some((id, score))
            } else {
                None
            }
        })
        .collect();

    scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(max);
    scored
}

/// Format scored communities into a context string.
///
/// Produces lines like:
/// ```text
/// Related knowledge:
/// - Community 42: summary text (entities: Alice, Bob, Project X)
/// ```
///
/// Truncates at `max_chars` (on a line boundary).
pub fn build_community_context(
    scored: &[(u64, f32)],
    summaries: &HashMap<u64, CommunitySummary>,
    max_chars: usize,
) -> String {
    if scored.is_empty() {
        return String::new();
    }

    let mut result = String::from("Related knowledge:\n");

    for &(id, _score) in scored {
        if let Some(summary) = summaries.get(&id) {
            let entities = if summary.key_entities.is_empty() {
                String::new()
            } else {
                format!(" (entities: {})", summary.key_entities.join(", "))
            };

            let line = format!("- Community {}: {}{}\n", id, summary.summary, entities);

            if result.len() + line.len() > max_chars {
                break;
            }
            result.push_str(&line);
        }
    }

    result
}

/// Format similar strategies into a context string.
///
/// Produces lines like:
/// ```text
/// Similar strategies:
/// 1. summary (quality: 0.85, success: 3, failure: 1)
/// ```
pub fn build_strategy_context(strategies: &[&Strategy], max: usize) -> String {
    if strategies.is_empty() {
        return String::new();
    }

    let mut result = String::from("Similar strategies:\n");
    for (i, strategy) in strategies.iter().take(max).enumerate() {
        let line = format!(
            "{}. {} (quality: {:.2}, success: {}, failure: {})\n",
            i + 1,
            strategy.summary,
            strategy.quality_score,
            strategy.success_count,
            strategy.failure_count,
        );
        result.push_str(&line);
    }
    result
}

/// Format existing playbooks as a context string for playbook extraction.
///
/// Takes pairs of `(goal_description, playbook)` and produces:
/// ```text
/// Prior experience:
/// - Goal: Deploy the app
///   What worked: Docker build, CI/CD pipeline
///   Lessons: Always test before deploying
/// ```
pub fn build_playbook_context(
    playbooks: &[(String, crate::conversation::compaction::GoalPlaybook)],
    max: usize,
) -> String {
    if playbooks.is_empty() {
        return String::new();
    }

    let mut result = String::from("Prior experience:\n");
    for (desc, pb) in playbooks.iter().take(max) {
        result.push_str(&format!("- Goal: {}\n", desc));
        if !pb.what_worked.is_empty() {
            result.push_str(&format!("  What worked: {}\n", pb.what_worked.join(", ")));
        }
        if !pb.lessons_learned.is_empty() {
            result.push_str(&format!("  Lessons: {}\n", pb.lessons_learned.join(", ")));
        }
    }
    result
}

/// Convenience: score communities and format context in one call.
///
/// Returns an empty string if no communities match.
pub fn community_context_for_topic(
    topic: &str,
    summaries: &HashMap<u64, CommunitySummary>,
    config: &EnrichmentConfig,
) -> String {
    let scored = score_communities(topic, summaries, config.max_communities);
    build_community_context(&scored, summaries, config.max_context_chars)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_summary(id: u64, summary: &str, entities: Vec<&str>) -> (u64, CommunitySummary) {
        (
            id,
            CommunitySummary {
                community_id: id,
                summary: summary.to_string(),
                key_entities: entities.into_iter().map(|s| s.to_string()).collect(),
                node_count: 5,
                generated_at: 0,
                token_estimate: 50,
            },
        )
    }

    #[test]
    fn test_score_communities_empty() {
        let summaries = HashMap::new();
        let result = score_communities("some topic", &summaries, 3);
        assert!(result.is_empty());
    }

    #[test]
    fn test_score_communities_empty_topic() {
        let mut summaries = HashMap::new();
        summaries.insert(1, make_summary(1, "test summary", vec!["Alice"]).1);
        let result = score_communities("", &summaries, 3);
        assert!(result.is_empty());
    }

    #[test]
    fn test_score_communities_no_overlap() {
        let mut summaries = HashMap::new();
        summaries.insert(1, make_summary(1, "elephants and giraffes", vec!["Zoo"]).1);
        let result = score_communities("quantum computing research", &summaries, 3);
        assert!(result.is_empty());
    }

    #[test]
    fn test_score_communities_with_match() {
        let mut summaries = HashMap::new();
        summaries.insert(
            1,
            make_summary(1, "project deployment and testing", vec!["Alice", "Bob"]).1,
        );
        summaries.insert(
            2,
            make_summary(2, "database optimization queries", vec!["System"]).1,
        );

        let result = score_communities("deployment testing pipeline", &summaries, 3);
        assert!(!result.is_empty());
        // Community 1 should score higher (2 overlapping tokens: deployment, testing)
        assert_eq!(result[0].0, 1);
    }

    #[test]
    fn test_score_communities_caps_at_max() {
        let mut summaries = HashMap::new();
        for i in 0..10 {
            summaries.insert(
                i,
                make_summary(
                    i,
                    &format!("topic about testing item {}", i),
                    vec!["Entity"],
                )
                .1,
            );
        }
        let result = score_communities("testing items", &summaries, 3);
        assert!(result.len() <= 3);
    }

    #[test]
    fn test_score_communities_entity_match() {
        let mut summaries = HashMap::new();
        summaries.insert(
            1,
            make_summary(1, "some unrelated summary", vec!["ProjectAlpha"]).1,
        );
        // "projectalpha" token length is > 3 so it should match
        let result = score_communities("working on ProjectAlpha deployment", &summaries, 3);
        assert!(!result.is_empty());
    }

    #[test]
    fn test_build_community_context_empty() {
        let summaries = HashMap::new();
        let result = build_community_context(&[], &summaries, 2000);
        assert!(result.is_empty());
    }

    #[test]
    fn test_build_community_context_formats_correctly() {
        let mut summaries = HashMap::new();
        summaries.insert(
            1,
            make_summary(1, "Handles user authentication", vec!["Auth", "Login"]).1,
        );
        let scored = vec![(1, 0.5)];
        let result = build_community_context(&scored, &summaries, 2000);
        assert!(result.contains("Related knowledge:"));
        assert!(result.contains("Community 1:"));
        assert!(result.contains("Handles user authentication"));
        assert!(result.contains("entities: Auth, Login"));
    }

    #[test]
    fn test_build_community_context_respects_max_chars() {
        let mut summaries = HashMap::new();
        for i in 0..10 {
            summaries.insert(
                i,
                make_summary(
                    i,
                    &format!("A very long summary about community number {} with lots of detail to fill up space and test truncation behavior", i),
                    vec!["Entity"],
                ).1,
            );
        }
        let scored: Vec<(u64, f32)> = (0..10).map(|i| (i, 1.0)).collect();
        let result = build_community_context(&scored, &summaries, 300);
        assert!(result.len() <= 350); // some slack for the last complete line
    }

    fn make_strategy(summary: &str, quality: f32, success: u32, failure: u32) -> Strategy {
        use crate::strategies::StrategyType;
        Strategy {
            id: 0,
            name: String::new(),
            summary: summary.to_string(),
            when_to_use: String::new(),
            when_not_to_use: String::new(),
            failure_modes: Vec::new(),
            playbook: Vec::new(),
            counterfactual: String::new(),
            supersedes: Vec::new(),
            applicable_domains: Vec::new(),
            lineage_depth: 0,
            summary_embedding: Vec::new(),
            agent_id: 0,
            reasoning_steps: Vec::new(),
            context_patterns: Vec::new(),
            success_indicators: Vec::new(),
            failure_patterns: Vec::new(),
            quality_score: quality,
            success_count: success,
            failure_count: failure,
            support_count: 0,
            strategy_type: StrategyType::Positive,
            precondition: String::new(),
            action_hint: String::new(),
            expected_success: 0.0,
            expected_cost: 0.0,
            expected_value: 0.0,
            confidence: 0.0,
            contradictions: Vec::new(),
            goal_bucket_id: 0,
            behavior_signature: String::new(),
            source_episodes: Vec::new(),
            created_at: 0,
            last_used: 0,
            metadata: std::collections::HashMap::new(),
            self_judged_quality: None,
            source_outcomes: Vec::new(),
            version: 0,
            parent_strategy: None,
        }
    }

    #[test]
    fn test_build_strategy_context_empty() {
        let result = build_strategy_context(&[], 3);
        assert!(result.is_empty());
    }

    #[test]
    fn test_build_strategy_context_formats() {
        let s1 = make_strategy("Use caching for repeated queries", 0.85, 5, 1);
        let s2 = make_strategy("Batch API calls when possible", 0.72, 3, 2);
        let strategies: Vec<&Strategy> = vec![&s1, &s2];
        let result = build_strategy_context(&strategies, 3);
        assert!(result.contains("Similar strategies:"));
        assert!(result.contains("1. Use caching"));
        assert!(result.contains("2. Batch API"));
        assert!(result.contains("quality: 0.85"));
    }

    #[test]
    fn test_build_strategy_context_caps_at_max() {
        let strats: Vec<Strategy> = (0..5)
            .map(|i| make_strategy(&format!("Strategy {}", i), 0.5, 1, 0))
            .collect();
        let refs: Vec<&Strategy> = strats.iter().collect();
        let result = build_strategy_context(&refs, 2);
        assert!(result.contains("1. Strategy 0"));
        assert!(result.contains("2. Strategy 1"));
        assert!(!result.contains("3. Strategy 2"));
    }

    #[test]
    fn test_build_playbook_context_empty() {
        let result = build_playbook_context(&[], 3);
        assert!(result.is_empty());
    }

    #[test]
    fn test_build_playbook_context_formats() {
        use crate::conversation::compaction::GoalPlaybook;
        let pb = GoalPlaybook {
            goal_description: "Deploy app".to_string(),
            what_worked: vec!["Docker".to_string(), "CI/CD".to_string()],
            what_didnt_work: vec!["Manual deploy".to_string()],
            lessons_learned: vec!["Always test first".to_string()],
            steps_taken: vec![],
            confidence: 0.9,
        };
        let playbooks = vec![("Deploy the application".to_string(), pb)];
        let result = build_playbook_context(&playbooks, 3);
        assert!(result.contains("Prior experience:"));
        assert!(result.contains("Goal: Deploy the application"));
        assert!(result.contains("What worked: Docker, CI/CD"));
        assert!(result.contains("Lessons: Always test first"));
    }

    #[test]
    fn test_community_context_for_topic() {
        let mut summaries = HashMap::new();
        summaries.insert(
            1,
            make_summary(1, "deployment and testing workflows", vec!["CI"]).1,
        );
        let config = EnrichmentConfig::default();
        let result = community_context_for_topic("deployment pipeline", &summaries, &config);
        assert!(result.contains("Related knowledge:"));
        assert!(result.contains("deployment"));
    }

    #[test]
    fn test_community_context_for_topic_no_match() {
        let mut summaries = HashMap::new();
        summaries.insert(1, make_summary(1, "elephants zoo animals", vec!["Zoo"]).1);
        let config = EnrichmentConfig::default();
        let result = community_context_for_topic("quantum computing", &summaries, &config);
        assert!(result.is_empty());
    }
}
