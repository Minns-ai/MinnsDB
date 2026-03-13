//! `minns recall` — query cross-session memory via NLQ + hybrid search.

use anyhow::Result;
use clap::Args;
use colored::Colorize;

use crate::client::MinnsDBClient;

#[derive(Debug, Args)]
pub struct RecallArgs {
    /// Natural language query
    pub query: String,

    /// Include related memories and strategies in output
    #[arg(long, default_value_t = true)]
    pub context: bool,

    /// Show brief output (single-line answer only)
    #[arg(long)]
    pub brief: bool,

    /// Maximum results to return
    #[arg(long)]
    pub limit: Option<usize>,

    /// Also show file-specific results (for hook integration)
    #[arg(long)]
    pub file: Option<String>,
}

pub async fn execute(client: &MinnsDBClient, args: RecallArgs) -> Result<()> {
    // Build query — if a file path is provided, prepend it for context
    let query = if let Some(ref file) = args.file {
        format!("{} (file: {})", args.query, file)
    } else {
        args.query.clone()
    };

    let resp = client.nlq(&query, args.context, args.limit).await?;

    if args.brief {
        // Brief mode: just the answer, for hook integration
        println!("{}", resp.answer);
        return Ok(());
    }

    // Full output
    println!("{}", "Answer".bold().green());
    println!("{}", resp.answer);
    println!();

    if resp.confidence > 0.0 {
        println!(
            "{} {:.0}%  {} {}  {} {}",
            "Confidence:".dimmed(),
            resp.confidence * 100.0,
            "Results:".dimmed(),
            resp.result_count,
            "Intent:".dimmed(),
            resp.intent,
        );
    }

    if !resp.entities_resolved.is_empty() {
        println!();
        println!("{}", "Entities".bold().cyan());
        for entity in &resp.entities_resolved {
            println!(
                "  {} ({}) — node {}",
                entity.text.bold(),
                entity.node_type.dimmed(),
                entity.node_id,
            );
        }
    }

    if !resp.explanation.is_empty() {
        println!();
        println!("{}", "Pipeline".bold().dimmed());
        for step in &resp.explanation {
            println!("  {}", step.dimmed());
        }
    }

    if !resp.related_memories.is_empty() {
        println!();
        println!("{}", "Related Memories".bold().yellow());
        for mem in &resp.related_memories {
            if let Some(summary) = mem.get("summary").and_then(|v| v.as_str()) {
                println!("  - {}", summary);
            }
        }
    }

    if !resp.related_strategies.is_empty() {
        println!();
        println!("{}", "Related Strategies".bold().magenta());
        for strat in &resp.related_strategies {
            if let Some(name) = strat.get("name").and_then(|v| v.as_str()) {
                let summary = strat.get("summary").and_then(|v| v.as_str()).unwrap_or("");
                println!("  - {} — {}", name.bold(), summary);
            }
        }
    }

    Ok(())
}
