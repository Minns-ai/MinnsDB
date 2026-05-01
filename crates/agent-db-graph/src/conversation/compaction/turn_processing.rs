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
