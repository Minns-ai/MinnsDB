//! Agent registry commands — register this agent and list agents in the group.

use anyhow::Result;
use clap::Subcommand;
use colored::Colorize;

use crate::client::EventGraphClient;

#[derive(Debug, clap::Args)]
pub struct AgentsArgs {
    #[command(subcommand)]
    pub action: AgentsAction,
}

#[derive(Debug, Subcommand)]
pub enum AgentsAction {
    /// Register this agent with the server
    Register {
        /// Agent ID (defaults to config or auto-generated)
        #[arg(long)]
        agent_id: Option<String>,
        /// Repository name this agent works on
        #[arg(long)]
        repo: Option<String>,
        /// Capabilities (comma-separated, e.g. "code,test,review")
        #[arg(long)]
        capabilities: Option<String>,
    },
    /// List all agents in the group
    List,
}

pub async fn execute(
    client: &EventGraphClient,
    args: AgentsArgs,
    config_agent_id: Option<&str>,
) -> Result<()> {
    match args.action {
        AgentsAction::Register {
            agent_id,
            repo,
            capabilities,
        } => {
            let aid = agent_id
                .or_else(|| config_agent_id.map(|s| s.to_string()))
                .unwrap_or_else(|| format!("{}-claude", client.group_id()));

            let caps: Vec<String> = capabilities
                .map(|c| c.split(',').map(|s| s.trim().to_string()).collect())
                .unwrap_or_default();

            let req = serde_json::json!({
                "agent_id": aid,
                "group_id": client.group_id(),
                "repository": repo,
                "capabilities": caps,
            });

            let resp = client.register_agent(req).await?;

            println!("{}", "Agent registered".green().bold());
            println!("  agent_id:      {}", aid);
            println!("  group_id:      {}", client.group_id());
            if let Some(r) = &repo {
                println!("  repository:    {}", r);
            }
            if let Some(nid) = resp.get("agent_node_id").and_then(|v| v.as_u64()) {
                println!("  node_id:       {}", nid);
            }

            Ok(())
        },
        AgentsAction::List => {
            let resp = client.list_agents().await?;

            let agents = resp
                .get("agents")
                .and_then(|v| v.as_array())
                .cloned()
                .unwrap_or_default();

            if agents.is_empty() {
                println!("No agents registered in group '{}'", client.group_id());
                return Ok(());
            }

            println!(
                "{} {} agent(s) in group '{}'",
                "Agents:".green().bold(),
                agents.len(),
                client.group_id()
            );

            for agent in &agents {
                let aid = agent
                    .get("agent_id")
                    .and_then(|v| v.as_str())
                    .unwrap_or("?");
                let repos = agent
                    .get("repositories")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    })
                    .unwrap_or_default();
                let caps = agent
                    .get("capabilities")
                    .and_then(|v| v.as_array())
                    .map(|arr| {
                        arr.iter()
                            .filter_map(|v| v.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    })
                    .unwrap_or_default();

                println!();
                println!("  {} {}", "agent:".dimmed(), aid.bold());
                if !repos.is_empty() {
                    println!("  {} {}", "repos:".dimmed(), repos);
                }
                if !caps.is_empty() {
                    println!("  {} {}", "caps: ".dimmed(), caps);
                }
            }

            Ok(())
        },
    }
}
