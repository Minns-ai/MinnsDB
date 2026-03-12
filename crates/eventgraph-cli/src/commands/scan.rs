//! `eventgraph scan` — scan a codebase and ingest all files into EventGraphDB.
//!
//! Walks a directory recursively, detects language from file extensions,
//! reads each file, and submits it to the code-file endpoint for AST parsing
//! and graph ingestion.
//!
//! Usage:
//!   eventgraph scan .                          # Scan current directory
//!   eventgraph scan src/ --language rust        # Force language
//!   eventgraph scan . --repo my-project         # Tag with repository name
//!   eventgraph scan . --ext rs,py              # Only specific extensions
//!   eventgraph scan . --dry-run                # Show what would be scanned

use anyhow::Result;
use clap::Args;
use colored::Colorize;
use std::path::{Path, PathBuf};

use crate::client::EventGraphClient;

/// Known language extensions (extension → language name).
fn detect_language(path: &Path) -> Option<&'static str> {
    match path.extension()?.to_str()? {
        "rs" => Some("rust"),
        "py" => Some("python"),
        "js" => Some("javascript"),
        "jsx" => Some("javascript"),
        "ts" => Some("typescript"),
        "tsx" => Some("typescript"),
        "go" => Some("go"),
        _ => None,
    }
}

/// Check if a path component should be skipped.
fn is_ignored(component: &str) -> bool {
    matches!(
        component,
        "target"
            | "node_modules"
            | ".git"
            | "__pycache__"
            | ".mypy_cache"
            | "dist"
            | "build"
            | ".next"
            | "vendor"
            | ".venv"
            | "venv"
    )
}

#[derive(Debug, Args)]
pub struct ScanArgs {
    /// Directory to scan
    pub path: String,

    /// Force language (overrides auto-detection)
    #[arg(long)]
    pub language: Option<String>,

    /// Repository name to tag files with
    #[arg(long)]
    pub repo: Option<String>,

    /// Only scan files with these extensions (comma-separated, e.g., "rs,py,ts")
    #[arg(long)]
    pub ext: Option<String>,

    /// Show what would be scanned without sending to server
    #[arg(long)]
    pub dry_run: bool,

    /// Maximum file size in bytes (default: 500KB)
    #[arg(long, default_value = "512000")]
    pub max_size: u64,
}

pub async fn execute(client: &EventGraphClient, args: ScanArgs) -> Result<()> {
    let root = PathBuf::from(&args.path).canonicalize()?;

    // Parse allowed extensions
    let allowed_ext: Option<Vec<String>> = args
        .ext
        .as_ref()
        .map(|e| e.split(',').map(|s| s.trim().to_lowercase()).collect());

    // Collect files to scan
    let mut files: Vec<PathBuf> = Vec::new();
    collect_files(&root, &mut files, &allowed_ext, args.max_size)?;

    if files.is_empty() {
        println!("No supported files found in {}", root.display());
        return Ok(());
    }

    println!(
        "{} Found {} files to scan in {}",
        "Scan:".bold().green(),
        files.len().to_string().cyan(),
        root.display()
    );

    if args.dry_run {
        for f in &files {
            let lang = detect_language(f).unwrap_or("unknown");
            let rel = f.strip_prefix(&root).unwrap_or(f);
            println!("  {} [{}]", rel.display(), lang);
        }
        println!("\n{}", "(dry run — no files sent to server)".dimmed());
        return Ok(());
    }

    let mut total_nodes = 0usize;
    let mut total_files = 0usize;
    let mut errors = 0usize;

    for (i, file_path) in files.iter().enumerate() {
        let rel = file_path.strip_prefix(&root).unwrap_or(file_path);
        let lang = args
            .language
            .as_deref()
            .or_else(|| detect_language(file_path));

        let content = match std::fs::read_to_string(file_path) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("  {} {} ({})", "SKIP".yellow(), rel.display(), e);
                errors += 1;
                continue;
            },
        };

        let file_path_str = rel.to_string_lossy();

        match client
            .submit_code_file(&file_path_str, &content, lang, args.repo.as_deref())
            .await
        {
            Ok(resp) => {
                total_nodes += resp.nodes_created;
                total_files += 1;
                // Progress line
                eprint!(
                    "\r  [{}/{}] {} — {} nodes",
                    i + 1,
                    files.len(),
                    rel.display(),
                    resp.nodes_created,
                );
            },
            Err(e) => {
                eprintln!("\n  {} {} ({})", "ERR".red(), rel.display(), e);
                errors += 1;
            },
        }
    }

    eprintln!(); // newline after progress
    println!();
    println!(
        "{} Scanned {} files, {} nodes created, {} errors",
        "Done.".bold().green(),
        total_files.to_string().cyan(),
        total_nodes.to_string().cyan(),
        errors,
    );

    Ok(())
}

/// Recursively collect files from a directory.
fn collect_files(
    dir: &Path,
    out: &mut Vec<PathBuf>,
    allowed_ext: &Option<Vec<String>>,
    max_size: u64,
) -> Result<()> {
    if !dir.is_dir() {
        // Single file
        if should_include(dir, allowed_ext) {
            if let Ok(meta) = std::fs::metadata(dir) {
                if meta.len() <= max_size {
                    out.push(dir.to_path_buf());
                }
            }
        }
        return Ok(());
    }

    let entries = std::fs::read_dir(dir)?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();

        // Skip ignored directories
        if let Some(name) = path.file_name().and_then(|n| n.to_str()) {
            if is_ignored(name) {
                continue;
            }
            // Skip hidden files/dirs
            if name.starts_with('.') {
                continue;
            }
        }

        if path.is_dir() {
            collect_files(&path, out, allowed_ext, max_size)?;
        } else if should_include(&path, allowed_ext) {
            if let Ok(meta) = std::fs::metadata(&path) {
                if meta.len() <= max_size {
                    out.push(path);
                }
            }
        }
    }

    Ok(())
}

/// Check if a file should be included based on extension filters.
fn should_include(path: &Path, allowed_ext: &Option<Vec<String>>) -> bool {
    let ext = match path.extension().and_then(|e| e.to_str()) {
        Some(e) => e.to_lowercase(),
        None => return false,
    };

    if let Some(allowed) = allowed_ext {
        return allowed.contains(&ext);
    }

    // Default: only known language extensions
    detect_language(path).is_some()
}
