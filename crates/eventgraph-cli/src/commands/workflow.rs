//! `eventgraph workflow` — create, list, run, and manage workflows via the workflow API.

use anyhow::Result;
use clap::{Args, Subcommand};
use colored::Colorize;

use crate::client::EventGraphClient;

#[derive(Debug, Args)]
pub struct WorkflowArgs {
    #[command(subcommand)]
    pub command: WorkflowCommand,
}

#[derive(Debug, Subcommand)]
pub enum WorkflowCommand {
    /// Create a new workflow from a JSON definition
    Create {
        /// Path to workflow JSON file (or - for stdin)
        #[arg(long)]
        file: String,
        /// Group ID for project scoping
        #[arg(long)]
        group_id: Option<String>,
    },
    /// List all workflows
    List,
    /// Show workflow details and step states
    Status {
        /// Workflow ID (node ID)
        workflow_id: u64,
    },
    /// Show workflow step states (alias for status)
    Run {
        /// Workflow ID
        workflow_id: u64,
    },
    /// Transition a workflow step's state
    StepTransition {
        /// Workflow ID
        workflow_id: u64,
        /// Step ID
        step_id: String,
        /// Target state (ready, running, completed, failed)
        #[arg(long)]
        state: String,
        /// Result/output (for completed steps)
        #[arg(long)]
        result: Option<String>,
    },
    /// Update a workflow with a new definition
    Update {
        /// Workflow ID
        workflow_id: u64,
        /// Path to updated workflow JSON
        #[arg(long)]
        file: String,
    },
    /// Delete (soft-delete) a workflow
    Delete {
        /// Workflow ID
        workflow_id: u64,
    },
    /// Attach feedback to a completed workflow
    Feedback {
        /// Workflow ID
        workflow_id: u64,
        /// Feedback text (what worked, what didn't, lessons learned)
        feedback: String,
        /// Overall outcome: success, partial, failure
        #[arg(long)]
        outcome: Option<String>,
    },
}

pub async fn execute(client: &EventGraphClient, args: WorkflowArgs) -> Result<()> {
    match args.command {
        WorkflowCommand::Create { file, group_id } => create(client, &file, group_id).await,
        WorkflowCommand::List => list(client).await,
        WorkflowCommand::Status { workflow_id } => status(client, workflow_id).await,
        WorkflowCommand::Run { workflow_id } => status(client, workflow_id).await,
        WorkflowCommand::StepTransition {
            workflow_id,
            step_id,
            state,
            result,
        } => step_transition(client, workflow_id, &step_id, &state, result).await,
        WorkflowCommand::Update { workflow_id, file } => update(client, workflow_id, &file).await,
        WorkflowCommand::Delete { workflow_id } => delete(client, workflow_id).await,
        WorkflowCommand::Feedback {
            workflow_id,
            feedback: fb_text,
            outcome,
        } => feedback(client, workflow_id, &fb_text, outcome).await,
    }
}

async fn create(client: &EventGraphClient, file: &str, group_id: Option<String>) -> Result<()> {
    let content = if file == "-" {
        let mut buf = String::new();
        std::io::Read::read_to_string(&mut std::io::stdin(), &mut buf)?;
        buf
    } else {
        std::fs::read_to_string(file)
            .map_err(|e| anyhow::anyhow!("Failed to read '{}': {}", file, e))?
    };

    let mut definition: serde_json::Value =
        serde_json::from_str(&content).map_err(|e| anyhow::anyhow!("Invalid JSON: {}", e))?;

    // Inject group_id if provided
    if let Some(gid) = group_id {
        definition
            .as_object_mut()
            .map(|obj| obj.insert("group_id".into(), serde_json::json!(gid)));
    }

    let resp = client.create_workflow(definition).await?;

    let name = resp
        .get("workflow_name")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown");
    let wf_id = resp
        .get("workflow_id")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let nodes = resp
        .get("nodes_created")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let edges = resp
        .get("edges_created")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    println!(
        "{} Created workflow '{}' (id: {})",
        "OK".bold().green(),
        name.cyan(),
        wf_id,
    );
    println!(
        "   {} nodes, {} edges created",
        nodes.to_string().cyan(),
        edges.to_string().cyan(),
    );

    if let Some(step_ids) = resp.get("step_node_ids").and_then(|v| v.as_object()) {
        println!("   {}", "Steps:".bold());
        for (step, nid) in step_ids {
            println!("     {} → node {}", step, nid);
        }
    }

    Ok(())
}

async fn list(client: &EventGraphClient) -> Result<()> {
    let resp = client.list_workflows().await?;

    println!("{}", "Workflows".bold().green());
    println!("{}", "─".repeat(60));

    if let Some(workflows) = resp.get("workflows").and_then(|v| v.as_array()) {
        if workflows.is_empty() {
            println!("  (no workflows found)");
        }
        for wf in workflows {
            let name = wf.get("name").and_then(|v| v.as_str()).unwrap_or("?");
            let id = wf.get("workflow_id").and_then(|v| v.as_u64()).unwrap_or(0);
            let steps = wf.get("step_count").and_then(|v| v.as_u64()).unwrap_or(0);
            let intent = wf.get("intent").and_then(|v| v.as_str()).unwrap_or("");

            println!("  {} [id: {}] ({} steps)", name.bold().cyan(), id, steps,);
            if !intent.is_empty() {
                println!("    {}", intent.dimmed());
            }
        }
    }

    Ok(())
}

async fn status(client: &EventGraphClient, workflow_id: u64) -> Result<()> {
    let resp = client.get_workflow(workflow_id).await?;

    let name = resp.get("name").and_then(|v| v.as_str()).unwrap_or("?");
    let intent = resp.get("intent").and_then(|v| v.as_str());

    println!(
        "{} {} (id: {})",
        "Workflow:".bold().green(),
        name.cyan(),
        workflow_id
    );
    if let Some(intent) = intent {
        println!("  {}", intent.dimmed());
    }
    println!();

    if let Some(steps) = resp.get("steps").and_then(|v| v.as_array()) {
        for step in steps {
            let step_id = step.get("id").and_then(|v| v.as_str()).unwrap_or("?");
            let task = step.get("task").and_then(|v| v.as_str()).unwrap_or("?");
            let role = step.get("role").and_then(|v| v.as_str()).unwrap_or("?");
            let state = step
                .get("state")
                .and_then(|v| v.as_str())
                .unwrap_or("pending");

            let icon = match state {
                "completed" => "✓".green().to_string(),
                "running" => "▶".yellow().to_string(),
                "failed" => "✗".red().to_string(),
                "ready" => "○".cyan().to_string(),
                _ => "·".dimmed().to_string(),
            };

            println!(
                "  {} {} [{}] ({}) — {}",
                icon,
                step_id.bold(),
                state,
                role.dimmed(),
                task,
            );

            // Show dependencies
            if let Some(deps) = step.get("depends_on").and_then(|v| v.as_array()) {
                if !deps.is_empty() {
                    let dep_strs: Vec<&str> = deps.iter().filter_map(|v| v.as_str()).collect();
                    println!("      depends on: {}", dep_strs.join(", ").dimmed());
                }
            }
        }
    }

    Ok(())
}

async fn step_transition(
    client: &EventGraphClient,
    workflow_id: u64,
    step_id: &str,
    state: &str,
    result: Option<String>,
) -> Result<()> {
    let resp = client
        .workflow_step_transition(workflow_id, step_id, state, result.as_deref())
        .await?;

    let success = resp
        .get("success")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    if success {
        println!(
            "{} Step '{}' → {}",
            "OK".bold().green(),
            step_id.cyan(),
            state.bold(),
        );
    } else {
        let err = resp
            .get("error")
            .and_then(|v| v.as_str())
            .unwrap_or("Unknown error");
        anyhow::bail!("Transition failed: {}", err);
    }

    Ok(())
}

async fn update(client: &EventGraphClient, workflow_id: u64, file: &str) -> Result<()> {
    let content = std::fs::read_to_string(file)
        .map_err(|e| anyhow::anyhow!("Failed to read '{}': {}", file, e))?;
    let definition: serde_json::Value =
        serde_json::from_str(&content).map_err(|e| anyhow::anyhow!("Invalid JSON: {}", e))?;

    let resp = client.update_workflow(workflow_id, definition).await?;

    let nodes_created = resp
        .get("nodes_created")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let edges_created = resp
        .get("edges_created")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let nodes_sup = resp
        .get("nodes_superseded")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);
    let edges_sup = resp
        .get("edges_superseded")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    println!(
        "{} Updated workflow {}",
        "OK".bold().green(),
        workflow_id.to_string().cyan(),
    );
    println!(
        "   +{} nodes, +{} edges | -{} nodes, -{} edges superseded",
        nodes_created, edges_created, nodes_sup, edges_sup,
    );

    Ok(())
}

async fn delete(client: &EventGraphClient, workflow_id: u64) -> Result<()> {
    let resp = client.delete_workflow(workflow_id).await?;

    let edges_sup = resp
        .get("edges_superseded")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    println!(
        "{} Deleted workflow {} ({} edges superseded)",
        "OK".bold().green(),
        workflow_id.to_string().cyan(),
        edges_sup,
    );

    Ok(())
}

async fn feedback(
    client: &EventGraphClient,
    workflow_id: u64,
    feedback: &str,
    outcome: Option<String>,
) -> Result<()> {
    let resp = client
        .workflow_feedback(workflow_id, feedback, outcome.as_deref())
        .await?;

    let feedback_nid = resp
        .get("feedback_node_id")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    println!(
        "{} Feedback attached to workflow {} (node: {})",
        "OK".bold().green(),
        workflow_id.to_string().cyan(),
        feedback_nid,
    );

    Ok(())
}
