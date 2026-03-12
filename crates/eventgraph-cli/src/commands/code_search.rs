//! `eventgraph code-search` — search code entities in the graph.
//!
//! Usage:
//!   eventgraph code-search parse_rust             # Find by name
//!   eventgraph code-search --kind function         # All functions
//!   eventgraph code-search --language rust          # All Rust entities
//!   eventgraph code-search --file "src/main*"       # By file pattern

use anyhow::Result;
use clap::Args;
use colored::Colorize;

use crate::client::EventGraphClient;

#[derive(Debug, Args)]
pub struct CodeSearchArgs {
    /// Name pattern to search for (glob-style)
    pub pattern: Option<String>,

    /// Filter by entity kind: function, struct, class, enum, module, etc.
    #[arg(long)]
    pub kind: Option<String>,

    /// Filter by language
    #[arg(long)]
    pub language: Option<String>,

    /// Filter by file path pattern (glob-style)
    #[arg(long)]
    pub file: Option<String>,

    /// Maximum results
    #[arg(long, default_value = "20")]
    pub limit: usize,

    /// Output as JSON
    #[arg(long)]
    pub json: bool,
}

pub async fn execute(client: &EventGraphClient, args: CodeSearchArgs) -> Result<()> {
    let resp = client
        .search_code(
            args.pattern.as_deref(),
            args.kind.as_deref(),
            args.language.as_deref(),
            args.file.as_deref(),
            args.limit,
        )
        .await?;

    if args.json {
        println!("{}", serde_json::to_string_pretty(&resp)?);
        return Ok(());
    }

    let entities = resp
        .get("entities")
        .and_then(|v| v.as_array())
        .cloned()
        .unwrap_or_default();

    let total = resp
        .get("total_matches")
        .and_then(|v| v.as_u64())
        .unwrap_or(0);

    if entities.is_empty() {
        println!("No code entities found.");
        return Ok(());
    }

    println!(
        "{} {} entities (showing {})",
        "Code Search:".bold().green(),
        total,
        entities.len(),
    );
    println!("{}", "─".repeat(70));

    for entity in &entities {
        let name = entity.get("name").and_then(|v| v.as_str()).unwrap_or("?");
        let qualified = entity
            .get("qualified_name")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let kind = entity.get("kind").and_then(|v| v.as_str()).unwrap_or("?");
        let file = entity
            .get("file_path")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let lang = entity
            .get("language")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        let sig = entity.get("signature").and_then(|v| v.as_str());
        let line_range = entity.get("line_range").and_then(|v| v.as_array());
        let vis = entity
            .get("visibility")
            .and_then(|v| v.as_str())
            .unwrap_or("");

        // Kind badge
        let kind_badge = match kind {
            "Function" => "fn".cyan(),
            "Class" | "Struct" => "struct".yellow(),
            "Enum" => "enum".yellow(),
            "Interface" => "trait".magenta(),
            "Module" => "mod".green(),
            "Variable" | "Constant" => "var".dimmed(),
            _ => kind.dimmed(),
        };

        print!("  {} ", kind_badge);
        if !vis.is_empty() {
            print!("{} ", vis.dimmed());
        }
        print!("{}", name.bold());

        if let Some(sig) = sig {
            print!(" {}", sig.dimmed());
        }
        println!();

        // Location
        let loc = if let Some(range) = line_range {
            let start = range.first().and_then(|v| v.as_u64()).unwrap_or(0);
            let end = range.get(1).and_then(|v| v.as_u64()).unwrap_or(0);
            format!("{}:{}-{}", file, start, end)
        } else {
            file.to_string()
        };
        println!("       {} [{}]", loc.dimmed(), lang.dimmed());

        if qualified != name && !qualified.is_empty() {
            println!("       {}", qualified.dimmed());
        }
    }

    Ok(())
}
