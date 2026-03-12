//! Config file loading for `.eventgraph.toml`.
//!
//! Searches up from CWD for `.eventgraph.toml`. Values from the config
//! file are used as defaults — CLI args and env vars override them.

use serde::Deserialize;
use std::path::PathBuf;

#[derive(Debug, Deserialize, Default)]
pub struct Config {
    pub group_id: Option<String>,
    pub agent_id: Option<String>,
}

/// Search up from `start` for `.eventgraph.toml` and parse it.
/// Returns default config if not found.
pub fn load_config() -> Config {
    match find_config_file() {
        Some(path) => match std::fs::read_to_string(&path) {
            Ok(content) => match toml::from_str(&content) {
                Ok(config) => {
                    tracing_log(&format!("Loaded config from {}", path.display()));
                    config
                },
                Err(e) => {
                    eprintln!("Warning: failed to parse {}: {}", path.display(), e);
                    Config::default()
                },
            },
            Err(_) => Config::default(),
        },
        None => Config::default(),
    }
}

/// Walk up from CWD looking for `.eventgraph.toml`.
fn find_config_file() -> Option<PathBuf> {
    let mut dir = std::env::current_dir().ok()?;
    loop {
        let candidate = dir.join(".eventgraph.toml");
        if candidate.is_file() {
            return Some(candidate);
        }
        if !dir.pop() {
            return None;
        }
    }
}

/// Resolve a value with priority: CLI arg > env var > config file > default.
pub fn resolve(cli_val: &str, env_key: &str, config_val: Option<&str>, default: &str) -> String {
    // clap already handles CLI arg > env var via #[arg(env = "...")].
    // If the CLI value is still the clap default, check config file.
    if cli_val == default {
        if let Ok(env_val) = std::env::var(env_key) {
            if !env_val.is_empty() {
                return env_val;
            }
        }
        if let Some(cfg) = config_val {
            return cfg.to_string();
        }
    }
    cli_val.to_string()
}

fn tracing_log(_msg: &str) {
    // Intentionally quiet — only prints in debug mode
    #[cfg(debug_assertions)]
    eprintln!("[config] {}", _msg);
}

/// Get the config file path (for `eventgraph init`).
pub fn config_path() -> PathBuf {
    std::env::current_dir()
        .unwrap_or_else(|_| PathBuf::from("."))
        .join(".eventgraph.toml")
}

/// Check if a config file exists in CWD or above.
pub fn config_exists() -> bool {
    find_config_file().is_some()
}
