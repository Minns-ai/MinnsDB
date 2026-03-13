//! Minns CLI — cross-session memory and workflow orchestration for MinnsDB.
//!
//! Usage:
//!   minns recall "how does auth work?"
//!   minns learn "We chose JWT with RS256" --category decision
//!   minns search "database migration" --mode hybrid
//!   minns status
//!   minns workflow create --file workflow.json
//!   minns vibe-graph "Review PR and write tests"
//!   minns scan .
//!   minns code-search --kind function
//!   minns converse "The user prefers dark mode"
//!   minns converse --file conversation.jsonl
//!   minns agents register --repo my-backend
//!   minns agents list
//!   minns init

mod client;
mod commands;
mod config;
#[allow(dead_code)]
mod types;

use anyhow::Result;
use clap::{Parser, Subcommand};
use client::MinnsDBClient;

/// Minns CLI — cross-session memory and workflow orchestration
#[derive(Parser)]
#[command(name = "minns", version, about)]
struct Cli {
    /// MinnsDB server URL
    #[arg(long, env = "MINNS_URL", default_value = "http://127.0.0.1:3000")]
    url: String,

    /// Group ID for project-scoped isolation
    #[arg(long, env = "MINNS_GROUP_ID", default_value = "default")]
    group_id: String,

    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Query cross-session memory (NLQ + hybrid search)
    Recall(commands::recall::RecallArgs),

    /// Store a learning into the graph
    Learn(commands::learn::LearnArgs),

    /// Multi-mode search (keyword/semantic/hybrid)
    Search(commands::search::SearchArgs),

    /// Knowledge graph dashboard
    Status,

    /// Manage workflows (create, list, run, step-complete)
    Workflow(commands::workflow::WorkflowArgs),

    /// Design workflow via 3-stage Vibe Graphing pipeline
    #[command(name = "vibe-graph")]
    VibeGraph(commands::vibe_graph::VibeGraphArgs),

    /// Generate a plan for a goal
    Plan(commands::plan::PlanArgs),

    /// Graph algorithm queries (neighbors, causal-path, communities)
    Query(commands::query::QueryArgs),

    /// Find strategies relevant to a situation
    Strategies {
        /// Describe the situation
        query: String,
    },

    /// Scan a codebase and ingest files into the graph (AST parsing)
    Scan(commands::scan::ScanArgs),

    /// Search code entities in the graph
    #[command(name = "code-search")]
    CodeSearch(commands::code_search::CodeSearchArgs),

    /// Send conversation messages to the graph for processing
    Converse(commands::converse::ConverseArgs),

    /// Agent registry — register and discover agents across repos
    Agents(commands::agents::AgentsArgs),

    /// Initialize a .minns.toml config file in the current directory
    Init {
        /// Group ID for this project
        #[arg(long)]
        group_id: Option<String>,
        /// Agent ID for this Claude instance
        #[arg(long)]
        agent_id: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();

    // Handle init before loading config (it creates the config)
    if let Command::Init { group_id, agent_id } = cli.command {
        return init_config(group_id, agent_id);
    }

    // Load .minns.toml (searches up from CWD)
    let cfg = config::load_config();

    // Resolve: CLI arg > env var > config file > default
    let url = config::resolve(&cli.url, "MINNS_URL", None, "http://127.0.0.1:3000");
    let group_id = config::resolve(
        &cli.group_id,
        "MINNS_GROUP_ID",
        cfg.group_id.as_deref(),
        "default",
    );

    let client = MinnsDBClient::new(url, group_id)?;

    match cli.command {
        Command::Recall(args) => commands::recall::execute(&client, args).await,
        Command::Learn(args) => commands::learn::execute(&client, args).await,
        Command::Search(args) => commands::search::execute(&client, args).await,
        Command::Status => commands::status::execute(&client).await,
        Command::Workflow(args) => commands::workflow::execute(&client, args).await,
        Command::VibeGraph(args) => commands::vibe_graph::execute(&client, args).await,
        Command::Plan(args) => commands::plan::execute(&client, args).await,
        Command::Query(args) => commands::query::execute(&client, args).await,
        Command::Scan(args) => commands::scan::execute(&client, args).await,
        Command::CodeSearch(args) => commands::code_search::execute(&client, args).await,
        Command::Converse(args) => commands::converse::execute(&client, args).await,
        Command::Agents(args) => {
            commands::agents::execute(&client, args, cfg.agent_id.as_deref()).await
        },
        Command::Strategies { query } => {
            let strategies = client.find_strategies(&query).await?;
            if strategies.is_empty() {
                println!("No strategies found for: {}", query);
            } else {
                println!("Strategies: {} strategies found", strategies.len());
                for s in &strategies {
                    println!();
                    println!("  {} (score: {:.2})", s.name, s.score);
                    println!("  {}", s.summary);
                    println!("  When to use: {}", s.when_to_use);
                }
            }
            Ok(())
        },
        Command::Init { .. } => unreachable!(),
    }
}

fn init_config(group_id: Option<String>, agent_id: Option<String>) -> Result<()> {
    let path = config::config_path();

    if path.exists() {
        anyhow::bail!(".minns.toml already exists at {}", path.display());
    }

    let group = group_id.unwrap_or_else(|| {
        // Default to directory name
        std::env::current_dir()
            .ok()
            .and_then(|p| p.file_name().map(|n| n.to_string_lossy().to_string()))
            .unwrap_or_else(|| "default".to_string())
    });

    let agent = agent_id.unwrap_or_else(|| format!("{}-claude", group));

    let content = format!("group_id = \"{group}\"\nagent_id = \"{agent}\"\n");

    std::fs::write(&path, &content)?;
    println!("Created {}", path.display());
    println!();
    println!("  group_id: {}", group);
    println!("  agent_id: {}", agent);

    Ok(())
}
