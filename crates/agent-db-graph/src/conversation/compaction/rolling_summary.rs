//! Rolling conversation summary — incrementally updated across turns.

use crate::conversation::types::ConversationIngest;
use crate::llm_client::{LlmClient, LlmRequest};
use serde::{Deserialize, Serialize};

/// A rolling, incrementally updated summary of an ongoing conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConversationRollingSummary {
    /// Conversation case identifier.
    pub case_id: String,
    /// Current summary text.
    pub summary: String,
    /// Last update timestamp (nanoseconds since epoch).
    pub last_updated: u64,
    /// How many turns have been summarized so far.
    pub turn_count: u32,
    /// Rough token estimate of the current summary.
    pub token_estimate: u32,
}

const SUMMARY_UPDATE_SYSTEM_PROMPT: &str = r#"You are a conversation summarizer. Given an existing summary and new messages, produce an UPDATED summary that captures all key information.

Rules:
- Preserve all important facts, preferences, goals, and decisions from the existing summary
- Integrate new information from the recent messages
- Keep the summary concise (under 500 words)
- Focus on: facts, preferences, goals, decisions, relationships, state changes
- Drop: greetings, filler, repetition
- Output ONLY the updated summary text, no JSON"#;

/// Update the rolling summary with new conversation messages.
///
/// - First call (no existing summary): "Summarize this conversation"
/// - Subsequent calls: "Existing summary + new messages → updated summary"
///
/// Returns `None` on failure (fail-open).
pub async fn update_rolling_summary(
    llm: &dyn LlmClient,
    existing_summary: Option<&str>,
    new_messages: &[crate::conversation::types::ConversationMessage],
) -> Option<String> {
    if new_messages.is_empty() {
        return existing_summary.map(|s| s.to_string());
    }

    let mut messages_text = String::new();
    for msg in new_messages {
        messages_text.push_str(&msg.role);
        messages_text.push_str(": ");
        messages_text.push_str(&msg.content);
        messages_text.push('\n');
    }

    let user_prompt = if let Some(summary) = existing_summary {
        format!(
            "Existing summary:\n{}\n\nNew messages:\n{}\nProduce updated summary.",
            summary, messages_text
        )
    } else {
        format!("Summarize this conversation:\n{}", messages_text)
    };

    let request = LlmRequest {
        system_prompt: SUMMARY_UPDATE_SYSTEM_PROMPT.to_string(),
        user_prompt,
        temperature: 0.0,
        max_tokens: 1024,
        json_mode: false,
    };

    let response = llm.complete(request).await.ok()?;
    let text = response.content.trim().to_string();
    if text.is_empty() {
        None
    } else {
        Some(text)
    }
}

/// Format transcript using rolling summary + last N messages.
///
/// Instead of the full transcript, this uses the summary as context
/// and appends only the most recent messages for detail.
pub fn format_with_summary(
    summary: &ConversationRollingSummary,
    data: &ConversationIngest,
    recent_count: usize,
) -> String {
    let mut buf = String::new();
    buf.push_str("[Rolling Summary]\n");
    buf.push_str(&summary.summary);
    buf.push_str("\n\n[Recent Messages]\n");

    // Collect all messages in order
    let all_messages: Vec<&crate::conversation::types::ConversationMessage> = data
        .sessions
        .iter()
        .flat_map(|s| s.messages.iter())
        .collect();

    // Take the last `recent_count` messages
    let start = all_messages.len().saturating_sub(recent_count);
    for msg in &all_messages[start..] {
        buf.push_str(&msg.role);
        buf.push_str(": ");
        buf.push_str(&msg.content);
        buf.push('\n');
    }

    buf
}
