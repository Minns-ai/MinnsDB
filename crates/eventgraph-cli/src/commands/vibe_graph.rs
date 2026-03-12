//! `eventgraph vibe-graph` — query the graph for workflow design context.
//!
//! The real 3-stage pipeline lives in the /vibe-graph skill prompt.
//! Claude does the reasoning. This command provides context from the graph
//! (strategies, past workflows, memories) to inform Claude's design.
//!
//! Usage:
//!   eventgraph vibe-graph "Review PR and write tests"       # Get design context
//!   eventgraph vibe-graph "Review PR" --json                # JSON for programmatic use

use anyhow::Result;
use clap::Args;
use colored::Colorize;

use crate::client::EventGraphClient;

#[derive(Debug, Args)]
pub struct VibeGraphArgs {
    /// Natural language intent (e.g., "Review PR and write tests")
    pub intent: String,

    /// Output as JSON (for programmatic use by Claude)
    #[arg(long)]
    pub json: bool,
}

pub async fn execute(client: &EventGraphClient, args: VibeGraphArgs) -> Result<()> {
    // Query graph for relevant context: strategies, memories, past workflows
    let nlq_query = format!(
        "What strategies, patterns, and workflows exist for: {}",
        args.intent
    );
    let resp = client.nlq(&nlq_query, true, Some(10)).await?;

    // Also search for strategies
    let strategies = client
        .find_strategies(&args.intent)
        .await
        .unwrap_or_default();

    // List existing workflows for reference
    let workflows = client
        .list_workflows()
        .await
        .unwrap_or(serde_json::json!({}));

    if args.json {
        println!(
            "{}",
            serde_json::to_string_pretty(&serde_json::json!({
                "intent": args.intent,
                "graph_context": {
                    "answer": resp.answer,
                    "confidence": resp.confidence,
                    "entities": resp.entities_resolved.iter().map(|e| serde_json::json!({
                        "text": e.text,
                        "type": e.node_type,
                        "node_id": e.node_id,
                    })).collect::<Vec<_>>(),
                    "related_memories": resp.related_memories,
                    "related_strategies": resp.related_strategies,
                },
                "proven_strategies": strategies.iter().map(|s| serde_json::json!({
                    "name": s.name,
                    "summary": s.summary,
                    "when_to_use": s.when_to_use,
                    "score": s.score,
                })).collect::<Vec<_>>(),
                "existing_workflows": workflows.get("workflows"),
            }))?
        );
    } else {
        println!(
            "{} {}",
            "Vibe Graph Context for:".bold().green(),
            args.intent.cyan()
        );
        println!();

        // Graph answer
        println!("{}", "Graph Knowledge".bold().yellow());
        println!("  {}", resp.answer);
        println!();

        // Strategies
        if !strategies.is_empty() {
            println!("{}", "Proven Strategies".bold().yellow());
            for s in &strategies {
                println!("  {} (score: {:.2})", s.name.bold(), s.score);
                println!("    {}", s.summary);
                println!("    When to use: {}", s.when_to_use);
            }
            println!();
        }

        // Existing workflows
        if let Some(wfs) = workflows.get("workflows").and_then(|v| v.as_array()) {
            if !wfs.is_empty() {
                println!("{}", "Existing Workflows".bold().yellow());
                for wf in wfs {
                    let name = wf.get("name").and_then(|v| v.as_str()).unwrap_or("?");
                    let id = wf.get("workflow_id").and_then(|v| v.as_u64()).unwrap_or(0);
                    println!("  {} (id: {})", name, id);
                }
                println!();
            }
        }

        println!(
            "{}",
            "Use /vibe-graph skill in Claude Code for the full 3-stage pipeline.".dimmed()
        );
    }

    Ok(())
}
