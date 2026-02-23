//! Neural-network-style encoders for transforming raw feature vectors into
//! fixed-size embeddings suitable for energy computation.
//!
//! All computation is pure Rust, CPU-only, no external ML framework required.
//! Total parameter count ~30K (120 KB of f32).

use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::types::{EventFeatures, MemoryFeatures, PolicyFeatures, StrategyFeatures};

// ────────────────────────────────────────────────────────────────
// Building blocks
// ────────────────────────────────────────────────────────────────

/// Hash embedding table: maps u64 hashes to learned embedding vectors.
///
/// Uses modular hashing (`hash % table_size`) to bucket arbitrary u64 keys
/// into a fixed-size table. Each bucket stores a learned `embed_dim`-sized vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingTable {
    /// Shape: [table_size][embed_dim]
    pub weights: Vec<Vec<f32>>,
    pub table_size: usize,
    pub embed_dim: usize,
}

impl EmbeddingTable {
    /// Create a new embedding table with small random weights.
    pub fn new(table_size: usize, embed_dim: usize, rng: &mut impl Rng) -> Self {
        let scale = 1.0 / (embed_dim as f32).sqrt();
        let weights = (0..table_size)
            .map(|_| {
                (0..embed_dim)
                    .map(|_| rng.gen_range(-scale..scale))
                    .collect()
            })
            .collect();
        Self {
            weights,
            table_size,
            embed_dim,
        }
    }

    /// Look up the embedding for a hash value.
    pub fn lookup(&self, hash: u64) -> &[f32] {
        let idx = (hash % self.table_size as u64) as usize;
        &self.weights[idx]
    }

    /// Get a mutable reference to the embedding for a hash value.
    pub fn lookup_mut(&mut self, hash: u64) -> &mut Vec<f32> {
        let idx = (hash % self.table_size as u64) as usize;
        &mut self.weights[idx]
    }
}

/// Simple linear layer: `output = input * W + bias` followed by ReLU.
///
/// Used to project concatenated features into the shared embedding space.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LinearLayer {
    /// Weight matrix, shape [output_dim][input_dim] (row-major).
    pub weights: Vec<Vec<f32>>,
    /// Bias vector, shape [output_dim].
    pub bias: Vec<f32>,
    pub input_dim: usize,
    pub output_dim: usize,
}

impl LinearLayer {
    /// Create a new linear layer with Xavier-uniform initialization.
    pub fn new(input_dim: usize, output_dim: usize, rng: &mut impl Rng) -> Self {
        let scale = (6.0 / (input_dim + output_dim) as f32).sqrt();
        let weights = (0..output_dim)
            .map(|_| {
                (0..input_dim)
                    .map(|_| rng.gen_range(-scale..scale))
                    .collect()
            })
            .collect();
        let bias = vec![0.0; output_dim];
        Self {
            weights,
            bias,
            input_dim,
            output_dim,
        }
    }

    /// Forward pass: `ReLU(input * W^T + bias)`.
    #[allow(clippy::needless_range_loop)]
    pub fn forward(&self, input: &[f32]) -> Vec<f32> {
        debug_assert_eq!(input.len(), self.input_dim, "input dimension mismatch");
        let mut output = Vec::with_capacity(self.output_dim);
        for i in 0..self.output_dim {
            let mut sum = self.bias[i];
            for j in 0..self.input_dim {
                sum += self.weights[i][j] * input[j];
            }
            // ReLU activation
            output.push(sum.max(0.0));
        }
        output
    }

    /// Forward pass without ReLU (for the final projection layer).
    #[allow(clippy::needless_range_loop)]
    pub fn forward_linear(&self, input: &[f32]) -> Vec<f32> {
        debug_assert_eq!(input.len(), self.input_dim, "input dimension mismatch");
        let mut output = Vec::with_capacity(self.output_dim);
        for i in 0..self.output_dim {
            let mut sum = self.bias[i];
            for j in 0..self.input_dim {
                sum += self.weights[i][j] * input[j];
            }
            output.push(sum);
        }
        output
    }
}

// ────────────────────────────────────────────────────────────────
// Layer encoders
// ────────────────────────────────────────────────────────────────

/// Encodes EventFeatures into a fixed-size embedding.
///
/// Input: 2 hash embeddings + 5 scalar features → concatenated → linear → embed_dim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventEncoder {
    pub event_type_emb: EmbeddingTable,
    pub action_name_emb: EmbeddingTable,
    pub context_emb: EmbeddingTable,
    /// Projects concatenated features to embed_dim.
    /// Input dim = 3 * embed_dim + 4 scalars (outcome, significance, log_delta, log_duration).
    pub projection: LinearLayer,
}

impl EventEncoder {
    pub fn new(embed_dim: usize, table_size: usize, rng: &mut impl Rng) -> Self {
        let input_dim = 3 * embed_dim + 4;
        Self {
            event_type_emb: EmbeddingTable::new(table_size, embed_dim, rng),
            action_name_emb: EmbeddingTable::new(table_size, embed_dim, rng),
            context_emb: EmbeddingTable::new(table_size, embed_dim, rng),
            projection: LinearLayer::new(input_dim, embed_dim, rng),
        }
    }

    /// Encode event features into an embed_dim-sized vector.
    pub fn encode(&self, features: &EventFeatures) -> Vec<f32> {
        let embed_dim = self.event_type_emb.embed_dim;
        let mut input = Vec::with_capacity(3 * embed_dim + 4);

        input.extend_from_slice(self.event_type_emb.lookup(features.event_type_hash));
        input.extend_from_slice(self.action_name_emb.lookup(features.action_name_hash));
        input.extend_from_slice(self.context_emb.lookup(features.context_fingerprint));

        // Scalar features (normalized)
        input.push(features.outcome_success);
        input.push(features.significance);
        input.push(log_normalize(features.temporal_delta_ns));
        input.push(log_normalize(features.duration_ns));

        self.projection.forward(&input)
    }
}

/// Encodes MemoryFeatures into a fixed-size embedding.
///
/// Input: 2 hash embeddings + 3 scalar features → concatenated → linear → embed_dim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEncoder {
    pub context_emb: EmbeddingTable,
    pub goal_emb: EmbeddingTable,
    /// Input dim = 2 * embed_dim + 3 scalars (tier_norm, strength, log_access).
    pub projection: LinearLayer,
}

impl MemoryEncoder {
    pub fn new(embed_dim: usize, table_size: usize, rng: &mut impl Rng) -> Self {
        let input_dim = 2 * embed_dim + 3;
        Self {
            context_emb: EmbeddingTable::new(table_size, embed_dim, rng),
            goal_emb: EmbeddingTable::new(table_size, embed_dim, rng),
            projection: LinearLayer::new(input_dim, embed_dim, rng),
        }
    }

    pub fn encode(&self, features: &MemoryFeatures) -> Vec<f32> {
        let embed_dim = self.context_emb.embed_dim;
        let mut input = Vec::with_capacity(2 * embed_dim + 3);

        input.extend_from_slice(self.context_emb.lookup(features.context_fingerprint));
        input.extend_from_slice(self.goal_emb.lookup(features.goal_bucket_id));

        // Scalar features
        input.push(features.tier as f32 / 2.0); // normalize to [0, 1]
        input.push(features.strength);
        input.push((features.access_count as f32 + 1.0).ln());

        self.projection.forward(&input)
    }
}

/// Encodes StrategyFeatures into a fixed-size embedding.
///
/// Input: 2 hash embeddings + 4 scalar features → concatenated → linear → embed_dim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyEncoder {
    pub goal_emb: EmbeddingTable,
    pub behavior_emb: EmbeddingTable,
    /// Input dim = 2 * embed_dim + 4 scalars.
    pub projection: LinearLayer,
}

impl StrategyEncoder {
    pub fn new(embed_dim: usize, table_size: usize, rng: &mut impl Rng) -> Self {
        let input_dim = 2 * embed_dim + 4;
        Self {
            goal_emb: EmbeddingTable::new(table_size, embed_dim, rng),
            behavior_emb: EmbeddingTable::new(table_size, embed_dim, rng),
            projection: LinearLayer::new(input_dim, embed_dim, rng),
        }
    }

    pub fn encode(&self, features: &StrategyFeatures) -> Vec<f32> {
        let embed_dim = self.goal_emb.embed_dim;
        let mut input = Vec::with_capacity(2 * embed_dim + 4);

        input.extend_from_slice(self.goal_emb.lookup(features.goal_bucket_id));
        input.extend_from_slice(self.behavior_emb.lookup(features.behavior_signature_hash));

        input.push(features.quality_score);
        input.push(features.expected_success);
        input.push(features.expected_value);
        input.push(features.confidence);

        self.projection.forward(&input)
    }
}

/// Encodes PolicyFeatures into a fixed-size embedding.
///
/// Input: 1 hash embedding + 4 scalar features → concatenated → linear → embed_dim.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyEncoder {
    pub context_emb: EmbeddingTable,
    /// Input dim = embed_dim + 4 scalars.
    pub projection: LinearLayer,
}

impl PolicyEncoder {
    pub fn new(embed_dim: usize, table_size: usize, rng: &mut impl Rng) -> Self {
        let input_dim = embed_dim + 4;
        Self {
            context_emb: EmbeddingTable::new(table_size, embed_dim, rng),
            projection: LinearLayer::new(input_dim, embed_dim, rng),
        }
    }

    pub fn encode(&self, features: &PolicyFeatures) -> Vec<f32> {
        let embed_dim = self.context_emb.embed_dim;
        let mut input = Vec::with_capacity(embed_dim + 4);

        input.extend_from_slice(self.context_emb.lookup(features.context_fingerprint));

        input.push((features.goal_count as f32 + 1.0).ln());
        input.push(features.top_goal_priority);
        input.push(features.resource_cpu_percent / 100.0);
        input.push(log_normalize(features.resource_memory_bytes as f64));

        self.projection.forward(&input)
    }
}

// ────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────

/// Log-normalize a large positive value to a manageable range.
/// `ln(1 + x / 1e9)` maps nanoseconds to a ~[0, 3] range for typical durations.
fn log_normalize(value: f64) -> f32 {
    ((1.0 + value / 1e9) as f32).ln()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn make_rng() -> StdRng {
        StdRng::seed_from_u64(42)
    }

    #[test]
    fn test_embedding_table_lookup() {
        let mut rng = make_rng();
        let table = EmbeddingTable::new(16, 8, &mut rng);
        let emb = table.lookup(42);
        assert_eq!(emb.len(), 8);
        // Same hash → same embedding
        assert_eq!(table.lookup(42), table.lookup(42));
        // Different hash mod → potentially different embedding
        let emb2 = table.lookup(42 + 16); // same bucket
        assert_eq!(emb, emb2);
    }

    #[test]
    fn test_linear_layer_output_dim() {
        let mut rng = make_rng();
        let layer = LinearLayer::new(10, 5, &mut rng);
        let input = vec![1.0; 10];
        let output = layer.forward(&input);
        assert_eq!(output.len(), 5);
    }

    #[test]
    fn test_linear_layer_relu() {
        let mut rng = make_rng();
        let layer = LinearLayer::new(2, 2, &mut rng);
        let input = vec![0.0; 2];
        let output = layer.forward(&input);
        // With zero input, output = ReLU(bias). Bias is 0, so output = 0.
        assert!(output.iter().all(|&v| v >= 0.0));
    }

    #[test]
    fn test_event_encoder_output_dim() {
        let mut rng = make_rng();
        let encoder = EventEncoder::new(16, 256, &mut rng);
        let features = EventFeatures {
            event_type_hash: 100,
            action_name_hash: 200,
            context_fingerprint: 300,
            outcome_success: 1.0,
            significance: 0.8,
            temporal_delta_ns: 1_000_000_000.0,
            duration_ns: 500_000_000.0,
        };
        let embedding = encoder.encode(&features);
        assert_eq!(embedding.len(), 16);
    }

    #[test]
    fn test_memory_encoder_output_dim() {
        let mut rng = make_rng();
        let encoder = MemoryEncoder::new(16, 256, &mut rng);
        let features = MemoryFeatures {
            tier: 1,
            strength: 0.9,
            access_count: 5,
            context_fingerprint: 123,
            goal_bucket_id: 456,
        };
        let embedding = encoder.encode(&features);
        assert_eq!(embedding.len(), 16);
    }

    #[test]
    fn test_strategy_encoder_output_dim() {
        let mut rng = make_rng();
        let encoder = StrategyEncoder::new(16, 256, &mut rng);
        let features = StrategyFeatures {
            quality_score: 0.85,
            expected_success: 0.9,
            expected_value: 1.0,
            confidence: 0.7,
            goal_bucket_id: 789,
            behavior_signature_hash: 101112,
        };
        let embedding = encoder.encode(&features);
        assert_eq!(embedding.len(), 16);
    }

    #[test]
    fn test_policy_encoder_output_dim() {
        let mut rng = make_rng();
        let encoder = PolicyEncoder::new(16, 256, &mut rng);
        let features = PolicyFeatures {
            goal_count: 3,
            top_goal_priority: 0.9,
            resource_cpu_percent: 50.0,
            resource_memory_bytes: 1_073_741_824,
            context_fingerprint: 999,
        };
        let embedding = encoder.encode(&features);
        assert_eq!(embedding.len(), 16);
    }

    #[test]
    fn test_log_normalize() {
        assert_eq!(log_normalize(0.0), 0.0_f32.ln_1p());
        // 1 second in ns
        let one_sec = log_normalize(1_000_000_000.0);
        assert!(one_sec > 0.0 && one_sec < 2.0);
    }
}
