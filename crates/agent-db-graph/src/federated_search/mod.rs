pub mod http_client;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedHit {
    pub text: String,
    pub score: f32,
    pub source: String,
    pub entity_id: String,
    pub timestamp: Option<String>,
    pub metadata: HashMap<String, serde_json::Value>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedSearchRequest {
    pub query: String,
    pub sources: Vec<String>,
    pub limit: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedSearchResponse {
    pub results: Vec<FederatedHit>,
    pub errors: Vec<FederatedSourceError>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FederatedSourceError {
    pub source: String,
    pub error: String,
}

#[async_trait]
pub trait FederatedSearchProvider: Send + Sync {
    async fn search(
        &self,
        request: &FederatedSearchRequest,
    ) -> Result<FederatedSearchResponse, FederatedError>;
}

#[derive(Debug, thiserror::Error)]
pub enum FederatedError {
    #[error("request failed: {0}")]
    Http(String),
    #[error("timed out after {0}ms")]
    Timeout(u64),
    #[error("deserialization failed: {0}")]
    Deserialize(String),
}

pub fn format_federated_context(hits: &[FederatedHit]) -> String {
    if hits.is_empty() {
        return String::new();
    }
    hits.iter()
        .map(|h| {
            let ts = h.timestamp.as_deref().unwrap_or("unknown time");
            format!("[{} via {}] {}", ts, h.source, h.text)
        })
        .collect::<Vec<_>>()
        .join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_empty() {
        assert_eq!(format_federated_context(&[]), "");
    }

    #[test]
    fn format_single_hit() {
        let hit = FederatedHit {
            text: "the auth bug is fixed".into(),
            score: 0.9,
            source: "src".into(),
            entity_id: "1".into(),
            timestamp: Some("2026-04-20T10:00:00Z".into()),
            metadata: HashMap::new(),
        };
        assert!(format_federated_context(&[hit]).contains("2026-04-20T10:00:00Z via src"));
    }

    #[test]
    fn format_missing_timestamp() {
        let hit = FederatedHit {
            text: "hello".into(),
            score: 0.5,
            source: "src".into(),
            entity_id: "1".into(),
            timestamp: None,
            metadata: HashMap::new(),
        };
        assert!(format_federated_context(&[hit]).contains("unknown time"));
    }
}
