//! The four vector collections MinnsDB indexes: graph nodes, graph edges,
//! claims, and memories. Each collection is an independent [`VectorStore`]
//! (today backed by Qdrant via [`QdrantStore`]) accessed through the trait
//! so call sites stay backend-agnostic.

use std::sync::Arc;

use minns_vectors::{
    Distance, QdrantConfig, QdrantStore, Quantization, ScalarQuantization, VectorError, VectorStore,
};

/// Default embedding dimensionality. Matches `text-embedding-3-small` and the
/// other 1536-d models we currently call against; override for any
/// collection trained on a different model.
pub const DEFAULT_DIM: usize = 1536;

/// Default Qdrant endpoint. Matches the sidecar deployed alongside each
/// MinnsDB instance.
pub const DEFAULT_QDRANT_URL: &str = "http://localhost:6334";

/// Default quantization choice: int8 scalar quantization pinned in RAM.
///
/// 4× memory compression at < 1 % typical recall loss for d ≥ 768 embeddings.
/// Operators who need more aggressive compression (TurboQuant 2-bit at 16×,
/// binary at 32×) override at config time.
fn default_quantization() -> Quantization {
    Quantization::Scalar(ScalarQuantization {
        quantile: None,
        always_ram: true,
    })
}

/// Configuration for the four vector collections opened against one Qdrant
/// endpoint. `collection_prefix` is empty in per-VM deployments; shared
/// deployments use it to namespace tenants on a single server.
#[derive(Debug, Clone)]
pub struct VectorsConfig {
    pub url: String,
    pub api_key: Option<String>,
    pub dim: usize,
    pub distance: Distance,
    pub quantization: Option<Quantization>,
    pub collection_prefix: String,
}

impl Default for VectorsConfig {
    fn default() -> Self {
        Self {
            url: DEFAULT_QDRANT_URL.to_string(),
            api_key: None,
            dim: DEFAULT_DIM,
            distance: Distance::Cosine,
            quantization: Some(default_quantization()),
            collection_prefix: String::new(),
        }
    }
}

impl VectorsConfig {
    pub fn new(url: impl Into<String>, dim: usize) -> Self {
        Self {
            url: url.into(),
            api_key: None,
            dim,
            distance: Distance::Cosine,
            quantization: Some(default_quantization()),
            collection_prefix: String::new(),
        }
    }

    pub fn with_api_key(mut self, api_key: impl Into<String>) -> Self {
        self.api_key = Some(api_key.into());
        self
    }

    pub fn with_distance(mut self, distance: Distance) -> Self {
        self.distance = distance;
        self
    }

    pub fn with_quantization(mut self, quantization: Quantization) -> Self {
        self.quantization = Some(quantization);
        self
    }

    pub fn without_quantization(mut self) -> Self {
        self.quantization = None;
        self
    }

    pub fn with_collection_prefix(mut self, prefix: impl Into<String>) -> Self {
        self.collection_prefix = prefix.into();
        self
    }

    fn collection_for(&self, category: &str) -> String {
        if self.collection_prefix.is_empty() {
            category.to_string()
        } else {
            format!("{}_{}", self.collection_prefix, category)
        }
    }

    fn qdrant_config_for(&self, category: &str) -> QdrantConfig {
        let mut q = QdrantConfig::new(&self.url, self.collection_for(category), self.dim)
            .with_distance(self.distance);
        if let Some(key) = self.api_key.as_deref() {
            q = q.with_api_key(key);
        }
        if let Some(quant) = self.quantization {
            q = q.with_quantization(quant);
        }
        q
    }
}

/// The four collections held by [`GraphEngine`](crate::GraphEngine).
pub struct Vectors {
    pub nodes: Arc<dyn VectorStore>,
    pub edges: Arc<dyn VectorStore>,
    pub claims: Arc<dyn VectorStore>,
    pub memories: Arc<dyn VectorStore>,
}

impl Vectors {
    /// Open all four collections against the configured Qdrant endpoint.
    ///
    /// Collections are opened in parallel so engine boot pays one round-trip
    /// instead of four. A missing or misconfigured backend is surfaced as a
    /// [`VectorError`] from the first collection that fails; the engine
    /// constructor propagates it as a startup error.
    pub async fn open(config: &VectorsConfig) -> Result<Self, VectorError> {
        tracing::info!(
            url = %config.url,
            dim = config.dim,
            prefix = %config.collection_prefix,
            "opening Qdrant vector collections"
        );

        let (nodes, edges, claims, memories) = tokio::try_join!(
            QdrantStore::open(config.qdrant_config_for("nodes")),
            QdrantStore::open(config.qdrant_config_for("edges")),
            QdrantStore::open(config.qdrant_config_for("claims")),
            QdrantStore::open(config.qdrant_config_for("memories")),
        )?;

        Ok(Self {
            nodes: Arc::new(nodes),
            edges: Arc::new(edges),
            claims: Arc::new(claims),
            memories: Arc::new(memories),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn collection_for_with_empty_prefix_returns_category_verbatim() {
        let config = VectorsConfig::new("http://qdrant:6334", DEFAULT_DIM);
        assert_eq!(config.collection_for("nodes"), "nodes");
        assert_eq!(config.collection_for("memories"), "memories");
    }

    #[test]
    fn collection_for_with_prefix_joins_with_underscore() {
        let config = VectorsConfig::new("http://qdrant:6334", DEFAULT_DIM)
            .with_collection_prefix("tenant_42");
        assert_eq!(config.collection_for("nodes"), "tenant_42_nodes");
        assert_eq!(config.collection_for("memories"), "tenant_42_memories");
    }

    #[test]
    fn qdrant_config_threads_through_api_key_and_quantization() {
        let custom = Quantization::Scalar(ScalarQuantization {
            quantile: Some(0.99),
            always_ram: true,
        });
        let config = VectorsConfig::new("http://qdrant:6334", 768)
            .with_api_key("secret")
            .with_quantization(custom);

        let q = config.qdrant_config_for("nodes");
        assert_eq!(q.collection, "nodes");
        assert_eq!(q.dim, 768);
        assert_eq!(q.api_key.as_deref(), Some("secret"));
        assert!(q.quantization.is_some());
    }

    #[test]
    fn default_config_targets_local_qdrant_with_int8_quantization() {
        let c = VectorsConfig::default();
        assert_eq!(c.url, DEFAULT_QDRANT_URL);
        assert_eq!(c.dim, DEFAULT_DIM);
        assert!(c.api_key.is_none());
        assert!(c.collection_prefix.is_empty());
        assert!(matches!(c.distance, Distance::Cosine));
        assert!(matches!(c.quantization, Some(Quantization::Scalar(_))));
    }

    #[test]
    fn without_quantization_disables_default() {
        let c = VectorsConfig::default().without_quantization();
        assert!(c.quantization.is_none());
    }
}
