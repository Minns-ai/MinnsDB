//! Conversation turn splitting, formatting, and graph context building.

use crate::conversation::graph_projection;
use crate::conversation::types::{ConversationIngest, ConversationMessage};
use crate::structures::Graph;
use std::collections::HashMap;

use super::prompts::TURN_EXTRACTION_PROMPT;

/// Maximum transcript length sent to the LLM (chars). Keeps the tail.
pub(crate) const MAX_TRANSCRIPT_CHARS: usize = 16_000;

/// Gap in nanoseconds between turn timestamps (1 second).
/// Turn N's facts get `base_ts + N * TURN_GAP + offset * 1000`.
pub(crate) const TURN_GAP: u64 = 1_000_000_000;

// ────────── Turn Data Structure ──────────

/// A single conversation turn (user+assistant pair) with positional metadata.
#[derive(Debug, Clone)]
pub struct ConversationTurn {
    pub messages: Vec<ConversationMessage>,
    pub session_index: usize,
    pub turn_index: usize,
    /// Session timestamp string, if provided (e.g. "2023/05/28").
    pub session_timestamp: Option<String>,
}

// ────────── Turn Splitting ──────────

/// Split a `ConversationIngest` into individual turns (user+assistant pairs).
///
/// Each turn contains one user message and the following assistant message (if any).
/// The `turn_index` is a global counter across all sessions, establishing temporal order.
pub fn split_into_turns(data: &ConversationIngest) -> Vec<ConversationTurn> {
    let mut turns = Vec::new();
    let mut global_turn = 0usize;

    for (session_idx, session) in data.sessions.iter().enumerate() {
        let mut i = 0;
        while i < session.messages.len() {
            let mut turn_msgs = Vec::new();

            // Take the current message (should be user)
            turn_msgs.push(session.messages[i].clone());
            i += 1;

            // If next message is assistant, include it in the same turn
            if i < session.messages.len() && session.messages[i].role == "assistant" {
                turn_msgs.push(session.messages[i].clone());
                i += 1;
            }

            turns.push(ConversationTurn {
                messages: turn_msgs,
                session_index: session_idx,
                turn_index: global_turn,
                session_timestamp: session.timestamp.clone(),
            });
            global_turn += 1;
        }
    }

    turns
}

// ────────── Adaptive Batching ──────────

/// Target token budget per cascade extraction call. Below this threshold
/// the LLM handles context well; above it we start seeing the "forgetting
/// effect" where entities introduced early in the prompt are missed by
/// later cascade steps. Empirically 3k tokens is the safe upper bound for
/// the small extraction models we use (gpt-4o-mini class).
pub(crate) const BATCH_TARGET_TOKENS: usize = 3_000;

/// Crude tokens-per-char ratio for size estimation without tokenising. Good
/// enough for budgeting decisions; off by a constant factor doesn't matter
/// because the target is itself a rule-of-thumb threshold.
const CHARS_PER_TOKEN: usize = 4;

/// Estimate the cascade prompt token cost of a turn — sum of message
/// content lengths divided by CHARS_PER_TOKEN.
fn estimate_turn_tokens(turn: &ConversationTurn) -> usize {
    turn.messages.iter().map(|m| m.content.len()).sum::<usize>() / CHARS_PER_TOKEN
}

/// Adaptive batching of turns by token budget. Replaces the historical
/// `.chunks(2)` hardcoded pairing.
///
/// Behaviour:
/// - Greedy: keep appending turns into the current batch while the running
///   token estimate stays under `BATCH_TARGET_TOKENS`.
/// - When adding a turn would push the batch over budget, seal the current
///   batch and start a new one with that turn.
/// - A single oversize turn (rare but possible if a user pastes a document)
///   becomes its own batch — never split mid-turn, never silently dropped.
/// - Always preserves chronological order across batches — the sequential
///   write phase downstream relies on this for supersession ordering.
///
/// Why this matters: the previous fixed pairing pays a fixed 1-batch cost
/// per ~2 turns regardless of content size. For dense sales-team-style
/// scenarios (20 messages × ~150 tokens) it produces 5+ small batches and
/// 5× the cascade-prompt overhead. For sparse small-talk it under-packs
/// each batch. Token-budget greedy matches the constraint that actually
/// matters (LLM context window + forgetting effect).
pub(crate) fn batch_turns_by_token_budget(
    turns: Vec<ConversationTurn>,
) -> Vec<Vec<ConversationTurn>> {
    let mut batches: Vec<Vec<ConversationTurn>> = Vec::new();
    let mut current: Vec<ConversationTurn> = Vec::new();
    let mut current_tokens = 0usize;

    for turn in turns {
        let turn_tokens = estimate_turn_tokens(&turn);
        // If this turn would push the batch over budget AND we already have
        // something in the batch, seal it off first. A single oversize turn
        // still becomes its own batch via the empty-batch case.
        if !current.is_empty() && current_tokens + turn_tokens > BATCH_TARGET_TOKENS {
            batches.push(std::mem::take(&mut current));
            current_tokens = 0;
        }
        current.push(turn);
        current_tokens += turn_tokens;
    }
    if !current.is_empty() {
        batches.push(current);
    }
    batches
}

// ────────── Formatting ──────────

/// Format just the turn messages as "role: content\n" without injecting rolling context.
pub(crate) fn format_messages(turn: &ConversationTurn) -> String {
    let mut buf = String::new();
    for msg in &turn.messages {
        buf.push_str(&msg.role);
        buf.push_str(": ");
        buf.push_str(&msg.content);
        buf.push('\n');
    }
    buf
}

/// Format a single turn's transcript with rolling context and graph state for per-turn extraction.
pub fn format_turn_transcript(
    turn: &ConversationTurn,
    rolling_facts: Option<&str>,
    graph_state: &str,
) -> String {
    let mut messages_text = String::new();
    for msg in &turn.messages {
        messages_text.push_str(&msg.role);
        messages_text.push_str(": ");
        messages_text.push_str(&msg.content);
        messages_text.push('\n');
    }

    TURN_EXTRACTION_PROMPT
        .replace("{rolling_facts}", rolling_facts.unwrap_or("(none)"))
        .replace(
            "{graph_state}",
            if graph_state.is_empty() {
                "(none)"
            } else {
                graph_state
            },
        )
        .replace("{messages}", &messages_text)
}

/// Format a `ConversationIngest` into a text transcript for the LLM.
pub fn format_transcript(data: &ConversationIngest) -> String {
    let mut buf = String::new();
    for session in &data.sessions {
        for msg in &session.messages {
            buf.push_str(&msg.role);
            buf.push_str(": ");
            buf.push_str(&msg.content);
            buf.push('\n');
        }
    }

    if buf.len() > MAX_TRANSCRIPT_CHARS {
        let start = buf.len() - MAX_TRANSCRIPT_CHARS;
        // Find the next newline after the cut point to avoid splitting a line
        let adjusted = buf[start..]
            .find('\n')
            .map(|p| start + p + 1)
            .unwrap_or(start);
        buf = buf[adjusted..].to_string();
    }

    buf
}

// ────────── Graph Context ──────────

/// Build graph context string showing current entity states for known entities.
///
/// For each known entity, queries the graph for current state edges (max valid_from per category)
/// and formats them as contextual information for the LLM.
pub fn build_graph_context_for_turn(graph: &Graph, known_entities: &[String]) -> String {
    if known_entities.is_empty() {
        return String::new();
    }

    let mut lines = Vec::new();

    for entity_name in known_entities {
        let facts = graph_projection::collect_entity_facts(graph, entity_name);
        if facts.is_empty() {
            continue;
        }

        // Group by category prefix (state:location, preference:food, etc.)
        let mut by_category: HashMap<String, Vec<&graph_projection::EntityFact>> = HashMap::new();
        for fact in &facts {
            by_category
                .entry(fact.association_type.clone())
                .or_default()
                .push(fact);
        }

        let mut entity_parts = Vec::new();
        for (assoc_type, cat_facts) in &by_category {
            // Find the current one (highest valid_from)
            if let Some(current) = cat_facts
                .iter()
                .filter(|f| f.is_current)
                .max_by_key(|f| f.valid_from.unwrap_or(0))
            {
                entity_parts.push(format!("{}={}", assoc_type, current.target_name));
            }
        }

        if !entity_parts.is_empty() {
            lines.push(format!("{}: {}", entity_name, entity_parts.join(", ")));
        }
    }

    lines.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conversation::types::ConversationMessage;

    fn make_turn(turn_index: usize, content: &str) -> ConversationTurn {
        ConversationTurn {
            messages: vec![ConversationMessage {
                role: "user".to_string(),
                content: content.to_string(),
                metadata: Default::default(),
            }],
            session_index: 0,
            turn_index,
            session_timestamp: None,
        }
    }

    /// Many small turns should all fit in one batch.
    #[test]
    fn small_turns_pack_into_single_batch() {
        let turns: Vec<_> = (0..10).map(|i| make_turn(i, "hello world")).collect();
        let batches = batch_turns_by_token_budget(turns);
        assert_eq!(batches.len(), 1);
        assert_eq!(batches[0].len(), 10);
    }

    /// One oversize turn becomes its own batch — never split mid-turn.
    #[test]
    fn oversize_turn_becomes_its_own_batch() {
        let big = "a".repeat(BATCH_TARGET_TOKENS * CHARS_PER_TOKEN * 2);
        let turns = vec![
            make_turn(0, "small"),
            make_turn(1, &big),
            make_turn(2, "small"),
        ];
        let batches = batch_turns_by_token_budget(turns);
        // Expected: [small], [big], [small] — the big turn forces a seal
        // before AND after.
        assert_eq!(batches.len(), 3);
        assert_eq!(batches[0][0].turn_index, 0);
        assert_eq!(batches[1][0].turn_index, 1);
        assert_eq!(batches[2][0].turn_index, 2);
    }

    /// Order is preserved across batches.
    #[test]
    fn batching_preserves_chronological_order() {
        let half_budget_chars = BATCH_TARGET_TOKENS * CHARS_PER_TOKEN / 2;
        let turns: Vec<_> = (0..6)
            .map(|i| make_turn(i, &"x".repeat(half_budget_chars)))
            .collect();
        let batches = batch_turns_by_token_budget(turns);
        let flat: Vec<usize> = batches
            .iter()
            .flat_map(|b| b.iter().map(|t| t.turn_index))
            .collect();
        for window in flat.windows(2) {
            assert!(window[0] < window[1]);
        }
        assert_eq!(flat.len(), 6);
    }

    /// Empty input returns no batches.
    #[test]
    fn empty_input_yields_no_batches() {
        let batches = batch_turns_by_token_budget(Vec::new());
        assert!(batches.is_empty());
    }
}
