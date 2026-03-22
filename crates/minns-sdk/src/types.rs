//! Shared types between the SDK and the MinnsDB runtime.

use serde::{Deserialize, Serialize};

/// Module descriptor returned by `__minns_describe__`.
/// Tells the runtime what this module is called, what functions it exports,
/// and what permissions it needs.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModuleDescriptor {
    pub name: String,
    pub version: String,
    #[serde(default)]
    pub functions: Vec<FunctionDef>,
    #[serde(default)]
    pub triggers: Vec<TriggerDef>,
    #[serde(default)]
    pub permissions: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionDef {
    pub name: String,
    #[serde(default)]
    pub description: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TriggerDef {
    #[serde(rename = "type")]
    pub trigger_type: String,
    #[serde(default)]
    pub table: Option<String>,
    #[serde(default)]
    pub edge_type: Option<String>,
    #[serde(default)]
    pub node_type: Option<String>,
    #[serde(default)]
    pub cron: Option<String>,
    pub function: String,
}

/// Result from a table query (returned by `table_query` and `graph_query`).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryResult {
    pub columns: Vec<String>,
    pub rows: Vec<Vec<serde_json::Value>>,
}

/// Result from an INSERT operation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InsertResult {
    pub row_id: u64,
    pub version_id: u64,
}

/// Result from an HTTP fetch.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpResponse {
    pub status: u16,
    pub body: Vec<u8>,
}

/// HTTP request sent to `http_fetch`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HttpRequest {
    pub url: String,
    #[serde(default = "default_method")]
    pub method: String,
    #[serde(default)]
    pub headers: std::collections::HashMap<String, String>,
    #[serde(default)]
    pub body: Option<Vec<u8>>,
}

fn default_method() -> String {
    "GET".into()
}
