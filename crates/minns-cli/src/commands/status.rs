//! `minns status` — knowledge graph dashboard.

use anyhow::Result;
use colored::Colorize;

use crate::client::MinnsDBClient;

pub async fn execute(client: &MinnsDBClient) -> Result<()> {
    // Fetch health and stats in parallel
    let (health, stats) = tokio::try_join!(client.health(), client.stats())?;

    println!("{}", "MinnsDB Status".bold().green());
    println!("{}", "═".repeat(50));

    // Server health
    println!(
        "  {} {} (v{}, uptime {}s)",
        "Server:".bold(),
        if health.is_healthy {
            "healthy".green().to_string()
        } else {
            "unhealthy".red().to_string()
        },
        health.version,
        health.uptime_seconds,
    );

    // Graph size
    println!(
        "  {} {} nodes, {} edges",
        "Graph:".bold(),
        health.node_count.to_string().cyan(),
        health.edge_count.to_string().cyan(),
    );

    println!();
    println!("{}", "Knowledge Metrics".bold().yellow());
    println!("{}", "─".repeat(50));
    println!(
        "  Events processed:    {}",
        stats.total_events_processed.to_string().cyan()
    );
    println!(
        "  Episodes detected:   {}",
        stats.total_episodes_detected.to_string().cyan()
    );
    println!(
        "  Memories formed:     {}",
        stats.total_memories_formed.to_string().cyan()
    );
    println!(
        "  Strategies extracted: {}",
        stats.total_strategies_extracted.to_string().cyan()
    );
    println!(
        "  Reinforcements:      {}",
        stats.total_reinforcements_applied.to_string().cyan()
    );
    println!(
        "  Avg processing:      {:.1}ms",
        stats.average_processing_time_ms
    );

    if health.processing_rate > 0.0 {
        println!(
            "  Processing rate:     {:.1} events/s",
            health.processing_rate
        );
    }

    Ok(())
}
