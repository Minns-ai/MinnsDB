//! Embedding generation and vector similarity for semantic search

use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use tracing::{debug, info};

/// Embedding vector type
pub type Embedding = Vec<f32>;

/// Request to generate embeddings for text
#[derive(Debug, Clone)]
pub struct EmbeddingRequest {
    /// Text to embed
    pub text: String,
    /// Optional context for better embeddings
    pub context: Option<String>,
}

/// Response containing generated embeddings
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    /// The embedding vector
    pub embedding: Embedding,
    /// Model used for embedding
    pub model: String,
    /// Tokens used (for cost tracking)
    pub tokens_used: u64,
}

/// Trait for embedding generation clients
#[async_trait]
pub trait EmbeddingClient: Send + Sync {
    /// Generate embedding for text
    async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse>;

    /// Get embedding dimensionality
    fn dimensions(&self) -> usize;

    /// Get model name
    fn model_name(&self) -> &str;
}

/// OpenAI embedding client
pub struct OpenAiEmbeddingClient {
    api_key: String,
    model: String,
    dimensions: usize,
    client: reqwest::Client,
}

impl OpenAiEmbeddingClient {
    /// Create a new OpenAI embedding client
    ///
    /// Common models:
    /// - text-embedding-3-small (1536 dims, cheaper)
    /// - text-embedding-3-large (3072 dims, better quality)
    /// - text-embedding-ada-002 (1536 dims, legacy)
    pub fn new(api_key: String, model: String) -> Self {
        let dimensions = match model.as_str() {
            "text-embedding-3-large" => 3072,
            "text-embedding-3-small" | "text-embedding-ada-002" => 1536,
            _ => 1536, // Default
        };

        Self {
            api_key,
            model,
            dimensions,
            client: reqwest::Client::new(),
        }
    }
}

#[async_trait]
impl EmbeddingClient for OpenAiEmbeddingClient {
    async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        debug!(
            "Generating OpenAI embedding for text: {} chars",
            request.text.len()
        );

        let input_text = if let Some(context) = request.context {
            format!("{}\n\n{}", context, request.text)
        } else {
            request.text
        };

        let response = self
            .client
            .post("https://api.openai.com/v1/embeddings")
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&serde_json::json!({
                "model": self.model,
                "input": input_text,
            }))
            .send()
            .await?;

        if !response.status().is_success() {
            let status = response.status();
            let body = response.text().await?;
            return Err(anyhow::anyhow!("OpenAI API error {}: {}", status, body));
        }

        let json: serde_json::Value = response.json().await?;

        // Extract embedding
        let embedding: Vec<f32> = serde_json::from_value(json["data"][0]["embedding"].clone())?;

        // Extract token usage
        let tokens_used = json["usage"]["total_tokens"].as_u64().unwrap_or(0);

        info!(
            "Generated embedding with {} dimensions, {} tokens",
            embedding.len(),
            tokens_used
        );

        Ok(EmbeddingResponse {
            embedding,
            model: self.model.clone(),
            tokens_used,
        })
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn model_name(&self) -> &str {
        &self.model
    }
}

/// Anthropic embedding client (placeholder - Anthropic doesn't have embedding API yet)
/// Uses OpenAI as fallback
pub struct AnthropicEmbeddingClient {
    openai_client: OpenAiEmbeddingClient,
}

impl AnthropicEmbeddingClient {
    pub fn new(openai_api_key: String) -> Self {
        Self {
            openai_client: OpenAiEmbeddingClient::new(
                openai_api_key,
                "text-embedding-3-small".to_string(),
            ),
        }
    }
}

#[async_trait]
impl EmbeddingClient for AnthropicEmbeddingClient {
    async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        // Fallback to OpenAI
        self.openai_client.embed(request).await
    }

    fn dimensions(&self) -> usize {
        self.openai_client.dimensions()
    }

    fn model_name(&self) -> &str {
        "openai-fallback"
    }
}

/// Mock embedding client for testing
pub struct MockEmbeddingClient {
    dimensions: usize,
}

impl MockEmbeddingClient {
    pub fn new(dimensions: usize) -> Self {
        Self { dimensions }
    }
}

impl Default for MockEmbeddingClient {
    fn default() -> Self {
        Self::new(384) // Common dimension for small models
    }
}

#[async_trait]
impl EmbeddingClient for MockEmbeddingClient {
    async fn embed(&self, request: EmbeddingRequest) -> Result<EmbeddingResponse> {
        debug!("Mock embedding for text: {} chars", request.text.len());

        // Generate deterministic embedding based on text hash
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        request.text.hash(&mut hasher);
        let hash = hasher.finish();

        // Use hash to seed random-looking but deterministic vector
        let mut embedding = Vec::with_capacity(self.dimensions);
        let mut seed = hash;
        for _ in 0..self.dimensions {
            seed = seed.wrapping_mul(1103515245).wrapping_add(12345);
            let value = ((seed >> 16) & 0x7FFF) as f32 / 32768.0;
            embedding.push(value);
        }

        // Normalize to unit length
        let magnitude: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for v in embedding.iter_mut() {
                *v /= magnitude;
            }
        }

        Ok(EmbeddingResponse {
            embedding,
            model: "mock".to_string(),
            tokens_used: 0,
        })
    }

    fn dimensions(&self) -> usize {
        self.dimensions
    }

    fn model_name(&self) -> &str {
        "mock"
    }
}

/// Vector similarity utilities
pub struct VectorSimilarity;

impl VectorSimilarity {
    /// Calculate cosine similarity between two vectors
    /// Returns value in range [-1, 1] where 1 = identical, 0 = orthogonal, -1 = opposite
    pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let magnitude_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let magnitude_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if magnitude_a == 0.0 || magnitude_b == 0.0 {
            return 0.0;
        }

        dot_product / (magnitude_a * magnitude_b)
    }

    /// Calculate cosine distance (1 - cosine_similarity)
    /// Returns value in range [0, 2] where 0 = identical, 1 = orthogonal, 2 = opposite
    pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
        1.0 - Self::cosine_similarity(a, b)
    }

    /// Calculate Euclidean distance between two vectors
    pub fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
        if a.len() != b.len() {
            return f32::INFINITY;
        }

        a.iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).powi(2))
            .sum::<f32>()
            .sqrt()
    }

    /// Normalize a vector to unit length (L2 normalization)
    pub fn normalize(vector: &mut [f32]) {
        let magnitude: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        if magnitude > 0.0 {
            for v in vector.iter_mut() {
                *v /= magnitude;
            }
        }
    }

    /// Find top-k most similar vectors to query using brute-force search
    ///
    /// Returns indices and similarity scores sorted by descending similarity
    pub fn top_k_similar(query: &[f32], vectors: &[Vec<f32>], k: usize) -> Vec<(usize, f32)> {
        let mut similarities: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(idx, vec)| (idx, Self::cosine_similarity(query, vec)))
            .collect();

        // Sort by similarity (descending)
        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        // Take top k
        similarities.truncate(k);
        similarities
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_mock_embedding_client() {
        let client = MockEmbeddingClient::new(384);
        let request = EmbeddingRequest {
            text: "Test claim text".to_string(),
            context: None,
        };

        let response = client.embed(request).await.unwrap();
        assert_eq!(response.embedding.len(), 384);
        assert_eq!(response.model, "mock");

        // Verify normalization (should be unit vector)
        let magnitude: f32 = response.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((magnitude - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let c = vec![0.0, 1.0, 0.0];
        let d = vec![-1.0, 0.0, 0.0];

        // Identical vectors
        assert!((VectorSimilarity::cosine_similarity(&a, &b) - 1.0).abs() < 0.001);

        // Orthogonal vectors
        assert!(VectorSimilarity::cosine_similarity(&a, &c).abs() < 0.001);

        // Opposite vectors
        assert!((VectorSimilarity::cosine_similarity(&a, &d) + 1.0).abs() < 0.001);
    }

    #[test]
    fn test_normalize() {
        let mut vector = vec![3.0, 4.0];
        VectorSimilarity::normalize(&mut vector);

        let magnitude: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((magnitude - 1.0).abs() < 0.001);
        assert!((vector[0] - 0.6).abs() < 0.001);
        assert!((vector[1] - 0.8).abs() < 0.001);
    }

    #[test]
    fn test_top_k_similar() {
        let query = vec![1.0, 0.0, 0.0];
        let vectors = vec![
            vec![1.0, 0.0, 0.0],  // Identical
            vec![0.9, 0.1, 0.0],  // Very similar
            vec![0.0, 1.0, 0.0],  // Orthogonal
            vec![-1.0, 0.0, 0.0], // Opposite
        ];

        let results = VectorSimilarity::top_k_similar(&query, &vectors, 2);

        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0, 0); // Index 0 should be most similar
        assert!(results[0].1 > 0.99); // Nearly 1.0 similarity
        assert_eq!(results[1].0, 1); // Index 1 should be second
    }
}
