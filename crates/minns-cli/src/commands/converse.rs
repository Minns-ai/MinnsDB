//! `minns converse` — send conversation events to the graph.
//!
//! Supports three modes:
//! - Single message: `minns converse "user said this"`
//! - File: `minns converse --file conversation.jsonl`
//! - Stdin: `cat conversation.jsonl | minns converse --from-stdin`
//!
//! JSONL format (one JSON object per line):
//!   {"role": "user", "content": "How does auth work?"}
//!   {"role": "assistant", "content": "We use JWT with RS256..."}

use anyhow::Result;
use clap::Args;
use colored::Colorize;
use std::collections::HashMap;
use std::io::Read as _;

use crate::client::MinnsDBClient;
use crate::types::{ConversationIngestRequest, MessageInput, SessionInput};

#[derive(Debug, Args)]
pub struct ConverseArgs {
    /// Message content (role defaults to "user")
    pub content: Option<String>,

    /// Message role
    #[arg(long, short, default_value = "user")]
    pub role: String,

    /// Session ID (groups messages into a conversation; auto-generated if omitted)
    #[arg(long)]
    pub session_id: Option<String>,

    /// Case ID for entity resolution continuity
    #[arg(long)]
    pub case_id: Option<String>,

    /// Topic label for this conversation
    #[arg(long)]
    pub topic: Option<String>,

    /// Read JSONL messages from a file
    #[arg(long)]
    pub file: Option<String>,

    /// Read JSONL messages from stdin
    #[arg(long)]
    pub from_stdin: bool,

    /// Process assistant messages for facts too
    #[arg(long)]
    pub include_assistant_facts: bool,
}

pub async fn execute(client: &MinnsDBClient, args: ConverseArgs) -> Result<()> {
    let session_id = args
        .session_id
        .unwrap_or_else(|| uuid::Uuid::new_v4().to_string());

    let messages = if let Some(ref file_path) = args.file {
        // Read JSONL from file
        let content = std::fs::read_to_string(file_path)
            .map_err(|e| anyhow::anyhow!("Failed to read {}: {}", file_path, e))?;
        parse_jsonl(&content)?
    } else if args.from_stdin || args.content.is_none() {
        // Read JSONL from stdin
        let mut buf = String::new();
        std::io::stdin().read_to_string(&mut buf)?;
        if buf.trim().is_empty() {
            anyhow::bail!(
                "No input. Pass a message as argument, --file, or --from-stdin with JSONL."
            );
        }
        // Try JSONL first, fall back to treating entire stdin as a single user message
        match parse_jsonl(&buf) {
            Ok(msgs) if !msgs.is_empty() => msgs,
            _ => vec![MessageInput {
                role: "user".to_string(),
                content: buf.trim().to_string(),
                metadata: HashMap::new(),
            }],
        }
    } else if let Some(content) = args.content {
        // Single message from args
        vec![MessageInput {
            role: args.role.clone(),
            content,
            metadata: HashMap::new(),
        }]
    } else {
        unreachable!()
    };

    if messages.is_empty() {
        anyhow::bail!("No messages to send.");
    }

    let msg_count = messages.len();

    let request = ConversationIngestRequest {
        case_id: args.case_id.clone(),
        sessions: vec![SessionInput {
            session_id: session_id.clone(),
            topic: args.topic.clone(),
            messages,
            contains_fact: None,
            fact_id: None,
            fact_quote: None,
        }],
        include_assistant_facts: args.include_assistant_facts,
        group_id: client.group_id().to_string(),
        metadata: HashMap::new(),
    };

    let resp = client.ingest_conversation(request).await?;

    // Display result
    let case = resp.get("case_id").and_then(|v| v.as_str()).unwrap_or("?");
    let events = resp
        .get("events_submitted")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    println!(
        "{} Sent {} message(s) → {} event(s)",
        "OK".bold().green(),
        msg_count,
        events,
    );
    println!("   {} {}", "case_id:".dimmed(), case);
    println!("   {} {}", "session: ".dimmed(), session_id);

    // Show compaction info if it ran
    if let Some(compaction) = resp.get("compaction").and_then(|v| v.as_object()) {
        let facts = compaction
            .get("facts_extracted")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        let goals = compaction
            .get("goals_extracted")
            .and_then(|v| v.as_u64())
            .unwrap_or(0);
        if facts > 0 || goals > 0 {
            println!(
                "   {} {} facts, {} goals extracted",
                "compaction:".dimmed(),
                facts,
                goals,
            );
        }
    }

    Ok(())
}

/// Parse JSONL (newline-delimited JSON) into messages.
/// Each line should be `{"role": "user"|"assistant", "content": "..."}`.
fn parse_jsonl(input: &str) -> Result<Vec<MessageInput>> {
    let mut messages = Vec::new();
    for (i, line) in input.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        let parsed: serde_json::Value = serde_json::from_str(line)
            .map_err(|e| anyhow::anyhow!("Invalid JSON on line {}: {}", i + 1, e))?;
        let role = parsed
            .get("role")
            .and_then(|v| v.as_str())
            .unwrap_or("user")
            .to_string();
        let content = parsed
            .get("content")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("Missing 'content' on line {}", i + 1))?
            .to_string();
        let metadata = parsed
            .get("metadata")
            .and_then(|v| v.as_object())
            .map(|m| m.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();
        messages.push(MessageInput {
            role,
            content,
            metadata,
        });
    }
    Ok(messages)
}
