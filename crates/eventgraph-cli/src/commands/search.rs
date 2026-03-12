//! `eventgraph search` — multi-mode search (keyword/semantic/hybrid).

use anyhow::Result;
use clap::Args;
use colored::Colorize;

use crate::client::EventGraphClient;

#[derive(Debug, Args)]
pub struct SearchArgs {
    /// Search query
    pub query: String,

    /// Search mode: keyword, semantic, or hybrid
    #[arg(long, short, default_value = "hybrid")]
    pub mode: String,

    /// Maximum results
    #[arg(long, short, default_value_t = 10)]
    pub limit: usize,
}

pub async fn execute(client: &EventGraphClient, args: SearchArgs) -> Result<()> {
    let resp = client
        .search(&args.query, Some(args.mode.clone()), Some(args.limit))
        .await?;

    println!(
        "{} {} results ({} mode)",
        "Search:".bold().green(),
        resp.total,
        resp.mode.cyan(),
    );
    println!();

    for (i, item) in resp.results.iter().enumerate() {
        let label = item
            .properties
            .get("label")
            .and_then(|v| v.as_str())
            .unwrap_or("(unlabeled)");
        println!(
            "  {}. {} [{}] score={:.3}",
            i + 1,
            label.bold(),
            item.node_type.dimmed(),
            item.score,
        );

        // Show a snippet of properties if available
        if let Some(props) = item.properties.as_object() {
            for key in ["summary", "action", "content", "data"] {
                if let Some(val) = props.get(key).and_then(|v| v.as_str()) {
                    let snippet = if val.len() > 100 {
                        format!("{}...", &val[..97])
                    } else {
                        val.to_string()
                    };
                    println!("     {}: {}", key.dimmed(), snippet);
                    break;
                }
            }
        }
    }

    Ok(())
}
