//! `minns query` — graph algorithm queries (neighbors, causal-path, communities).

use anyhow::Result;
use clap::{Args, Subcommand};
use colored::Colorize;

use crate::client::MinnsDBClient;

#[derive(Debug, Args)]
pub struct QueryArgs {
    #[command(subcommand)]
    pub command: QueryCommand,
}

#[derive(Debug, Subcommand)]
pub enum QueryCommand {
    /// Get neighbors of a node
    Neighbors {
        /// Node ID
        node_id: u64,
    },
    /// Find causal path between two nodes
    CausalPath {
        /// Source node ID
        from: u64,
        /// Target node ID
        to: u64,
    },
    /// Detect communities in the graph
    Communities,
}

pub async fn execute(client: &MinnsDBClient, args: QueryArgs) -> Result<()> {
    match args.command {
        QueryCommand::Neighbors { node_id } => neighbors(client, node_id).await,
        QueryCommand::CausalPath { from, to } => causal_path(client, from, to).await,
        QueryCommand::Communities => communities(client).await,
    }
}

async fn neighbors(client: &MinnsDBClient, node_id: u64) -> Result<()> {
    let resp = client.neighbors(node_id).await?;

    println!(
        "{} {} ({} nodes, {} edges)",
        "Neighbors of".bold().green(),
        node_id.to_string().cyan(),
        resp.nodes.len(),
        resp.edges.len(),
    );
    println!();

    for node in &resp.nodes {
        println!(
            "  {} {} [{}]",
            node.id.to_string().bold(),
            node.label,
            node.node_type.dimmed(),
        );
    }

    if !resp.edges.is_empty() {
        println!();
        println!("{}", "Edges".bold().yellow());
        for edge in &resp.edges {
            println!(
                "  {} --[{}]--> {} (w={:.2}, c={:.2})",
                edge.from, edge.edge_type, edge.to, edge.weight, edge.confidence,
            );
        }
    }

    Ok(())
}

async fn causal_path(client: &MinnsDBClient, from: u64, to: u64) -> Result<()> {
    let resp = client.causal_path(from, to).await?;

    if resp.found {
        let path = resp.path.unwrap_or_default();
        println!(
            "{} {} -> {} ({} hops)",
            "Causal path found:".bold().green(),
            from,
            to,
            path.len().saturating_sub(1),
        );
        println!(
            "  {}",
            path.iter()
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(" → ")
        );
    } else {
        println!(
            "{} No causal path from {} to {}",
            "Not found:".bold().yellow(),
            from,
            to,
        );
    }

    Ok(())
}

async fn communities(client: &MinnsDBClient) -> Result<()> {
    let resp = client.communities().await?;

    println!(
        "{} {} communities (modularity: {:.3}, algorithm: {})",
        "Communities:".bold().green(),
        resp.community_count,
        resp.modularity,
        resp.algorithm,
    );
    println!();

    for community in &resp.communities {
        println!(
            "  Community {} ({} nodes): [{}]",
            community.community_id.to_string().cyan(),
            community.size,
            community
                .node_ids
                .iter()
                .take(10)
                .map(|id| id.to_string())
                .collect::<Vec<_>>()
                .join(", "),
        );
    }

    Ok(())
}
