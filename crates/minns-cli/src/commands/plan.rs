//! `minns plan` — generate a plan for a goal using the planning engine.

use anyhow::Result;
use clap::Args;
use colored::Colorize;

use crate::client::MinnsDBClient;

#[derive(Debug, Args)]
pub struct PlanArgs {
    /// Goal to plan for
    pub goal: String,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

pub async fn execute(client: &MinnsDBClient, args: PlanArgs) -> Result<()> {
    let resp = client.plan(&args.goal).await?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&resp)?);
    } else {
        println!("{} {}", "Plan for:".bold().green(), args.goal.cyan());
        println!();
        println!("{}", serde_json::to_string_pretty(&resp)?);
    }

    Ok(())
}
