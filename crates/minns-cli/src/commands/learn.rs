//! `minns learn` — store a learning into the graph.

use anyhow::Result;
use clap::Args;
use colored::Colorize;
use std::collections::HashMap;
use std::io::Read as _;

use crate::client::MinnsDBClient;
use crate::types::{ConversationIngestRequest, MessageInput, SessionInput};

#[derive(Debug, Args)]
pub struct LearnArgs {
    /// Content to learn (omit to read from stdin)
    pub content: Option<String>,

    /// Category: decision, pattern, bug, architecture, session_summary, etc.
    #[arg(long, short, default_value = "learning")]
    pub category: String,

    /// Comma-separated tags
    #[arg(long, short)]
    pub tags: Option<String>,

    /// Read content from stdin (for hook integration)
    #[arg(long)]
    pub from_stdin: bool,

    /// Source file path (adds file context to the learning)
    #[arg(long)]
    pub source: Option<String>,
}

pub async fn execute(client: &MinnsDBClient, args: LearnArgs) -> Result<()> {
    // Get content from args or stdin
    let content = match args.content {
        Some(c) if !args.from_stdin => c,
        _ => {
            let mut buf = String::new();
            std::io::stdin()
                .read_to_string(&mut buf)
                .map_err(|e| anyhow::anyhow!("Failed to read stdin: {}", e))?;
            if buf.trim().is_empty() {
                anyhow::bail!("No content provided. Pass content as argument or via --from-stdin.");
            }
            buf
        },
    };

    // Build metadata
    let mut metadata: HashMap<String, serde_json::Value> = HashMap::new();
    metadata.insert("category".into(), serde_json::json!(args.category));
    if let Some(ref tags) = args.tags {
        let tag_list: Vec<&str> = tags.split(',').map(|t| t.trim()).collect();
        metadata.insert("tags".into(), serde_json::json!(tag_list));
    }
    if let Some(ref source) = args.source {
        metadata.insert("source_file".into(), serde_json::json!(source));
    }

    // Format as a conversation with the learning as a user fact
    let session_id = uuid::Uuid::new_v4().to_string();
    let message_content = format!("[{}] {}", args.category, content);

    let request = ConversationIngestRequest {
        case_id: None,
        sessions: vec![SessionInput {
            session_id,
            topic: Some(args.category.clone()),
            messages: vec![MessageInput {
                role: "user".into(),
                content: message_content,
                metadata: HashMap::new(),
            }],
            contains_fact: Some(true),
            fact_id: None,
            fact_quote: Some(content.clone()),
        }],
        include_assistant_facts: false,
        group_id: client.group_id().to_string(),
        metadata,
    };

    let resp = client.ingest_conversation(request).await?;

    // Check for success
    if resp.get("error").is_some() {
        let err = resp
            .get("error")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown error");
        anyhow::bail!("Failed to store learning: {}", err);
    }

    println!(
        "{} Stored {} learning: {}",
        "OK".bold().green(),
        args.category.cyan(),
        if content.len() > 80 {
            format!("{}...", &content[..77])
        } else {
            content
        },
    );

    if let Some(tags) = args.tags {
        println!("   {} {}", "Tags:".dimmed(), tags);
    }

    Ok(())
}
