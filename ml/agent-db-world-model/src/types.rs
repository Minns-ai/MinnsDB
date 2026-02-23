//! Core types for the energy-based predictive coding world model.
//!
//! This module defines all feature vectors, critic reports, prediction error
//! reports, training tuples, energy statistics, and configuration types used
//! throughout the world model crate.

use serde::{Deserialize, Serialize};

// ────────────────────────────────────────────────────────────────
// Mismatch attribution
// ────────────────────────────────────────────────────────────────

/// Which layer has the highest mismatch in a critic evaluation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MismatchLayer {
    Event,
    Memory,
    Strategy,
    Policy,
    None,
}

// ────────────────────────────────────────────────────────────────
// Feature vectors
// ────────────────────────────────────────────────────────────────

/// Feature vector extracted from an event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventFeatures {
    pub event_type_hash: u64,
    pub action_name_hash: u64,
    pub context_fingerprint: u64,
    /// 1.0 = success, 0.0 = failure, 0.5 = partial/unknown
    pub outcome_success: f32,
    pub significance: f32,
    /// Time since previous event in episode (nanoseconds)
    pub temporal_delta_ns: f64,
    /// Total duration of the action/event (nanoseconds)
    pub duration_ns: f64,
}

/// Feature vector extracted from a memory.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryFeatures {
    /// 0 = Episodic, 1 = Semantic, 2 = Schema
    pub tier: u8,
    pub strength: f32,
    pub access_count: u32,
    pub context_fingerprint: u64,
    pub goal_bucket_id: u64,
}

/// Feature vector extracted from a strategy.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyFeatures {
    pub quality_score: f32,
    pub expected_success: f32,
    pub expected_value: f32,
    pub confidence: f32,
    pub goal_bucket_id: u64,
    pub behavior_signature_hash: u64,
}

/// Feature vector extracted from the current policy state.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PolicyFeatures {
    pub goal_count: u32,
    pub top_goal_priority: f32,
    pub resource_cpu_percent: f32,
    pub resource_memory_bytes: u64,
    pub context_fingerprint: u64,
}

// ────────────────────────────────────────────────────────────────
// Critic output
// ────────────────────────────────────────────────────────────────

/// Full critic report for a scored configuration.
///
/// Produced by [`WorldModelCritic::score`] and [`WorldModelCritic::score_strategy`].
/// The `total_energy` is the primary ranking signal — lower means the
/// configuration is more compatible with what the model has learned.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CriticReport {
    /// Total free energy (lower = more compatible).
    pub total_energy: f32,

    /// Energy at the policy→strategy boundary.
    pub policy_strategy_energy: f32,
    /// Energy at the strategy→memory boundary.
    pub strategy_memory_energy: f32,
    /// Energy at the memory→event boundary.
    pub memory_event_energy: f32,

    /// Novelty as z-score over running energy statistics.
    pub novelty_z: f32,
    /// Whether this exceeds the novelty threshold.
    pub is_novel: bool,
    /// Which layer has the highest energy (mismatch attribution).
    pub mismatch_layer: MismatchLayer,

    /// Confidence in this score.
    ///
    /// `confidence = min(warmup_factor, support_factor, stability_factor)`
    /// where:
    ///   - warmup_factor:   0.0 until warmup_episodes seen, then 1.0
    ///   - support_factor:  min(1.0, support_count / min_support)
    ///   - stability_factor: 1.0 - (energy_variance / max_variance).clamp(0, 1)
    pub confidence: f32,
    /// How many similar configurations the model has been trained on.
    pub support_count: u64,
}

// ────────────────────────────────────────────────────────────────
// Bottom-up prediction error
// ────────────────────────────────────────────────────────────────

/// Bottom-up prediction error report for a single observed event.
///
/// Produced by [`WorldModelCritic::prediction_error`].
/// Each per-layer z-score indicates how surprising that layer's energy is
/// relative to the running statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictionErrorReport {
    /// Raw energy at the event→memory boundary.
    pub event_energy: f32,
    /// Raw energy at the memory→strategy boundary.
    pub memory_energy: f32,
    /// Raw energy at the strategy→policy boundary.
    pub strategy_energy: f32,

    /// Per-layer z-scores.
    pub event_z: f32,
    pub memory_z: f32,
    pub strategy_z: f32,

    /// Overall surprise (z-score of total energy).
    pub total_z: f32,
    /// Which layer has the highest z-score.
    pub mismatch_layer: MismatchLayer,
}

// ────────────────────────────────────────────────────────────────
// Training
// ────────────────────────────────────────────────────────────────

/// A training tuple assembled from a completed episode.
///
/// Positive tuples come from real episodes; negative tuples are corrupted
/// versions designed to teach the model what does NOT belong together.
/// See the plan's "Training Tuple Construction Rules" for assembly details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrainingTuple {
    pub event_features: EventFeatures,
    pub memory_features: MemoryFeatures,
    pub strategy_features: StrategyFeatures,
    pub policy_features: PolicyFeatures,
    /// `true` for real (positive) examples, `false` for corrupted (negative) examples.
    pub is_positive: bool,
    /// Weight for this example (typically the salience_score of the source episode).
    pub weight: f32,
}

// ────────────────────────────────────────────────────────────────
// Energy statistics (monitoring)
// ────────────────────────────────────────────────────────────────

/// Running energy statistics for monitoring the world model's state.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct EnergyStats {
    pub running_mean: f32,
    pub running_variance: f32,
    pub total_scored: u64,
    pub total_trained: u64,
    pub avg_loss: f32,
    pub is_warmed_up: bool,
}

// ────────────────────────────────────────────────────────────────
// Per-layer running statistics
// ────────────────────────────────────────────────────────────────

/// Running mean/variance for a single energy layer (Welford's algorithm).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerStats {
    pub count: u64,
    pub mean: f32,
    pub m2: f32,
}

impl Default for LayerStats {
    fn default() -> Self {
        Self {
            count: 0,
            mean: 0.0,
            m2: 0.0,
        }
    }
}

impl LayerStats {
    /// Update running statistics with a new observation (Welford's online algorithm).
    pub fn update(&mut self, value: f32) {
        self.count += 1;
        let delta = value - self.mean;
        self.mean += delta / self.count as f32;
        let delta2 = value - self.mean;
        self.m2 += delta * delta2;
    }

    /// Current variance (population).
    pub fn variance(&self) -> f32 {
        if self.count < 2 {
            0.0
        } else {
            self.m2 / self.count as f32
        }
    }

    /// Standard deviation.
    pub fn std_dev(&self) -> f32 {
        self.variance().sqrt()
    }

    /// Compute z-score for a value relative to this distribution.
    pub fn z_score(&self, value: f32) -> f32 {
        let sd = self.std_dev();
        if sd < 1e-8 {
            0.0
        } else {
            (value - self.mean) / sd
        }
    }
}

// ────────────────────────────────────────────────────────────────
// Configuration
// ────────────────────────────────────────────────────────────────

/// Configuration for the EBM world model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorldModelConfig {
    /// Embedding dimension for each layer's encoder output.
    pub embed_dim: usize,

    /// Size of the hash embedding tables (number of buckets).
    pub embedding_table_size: usize,

    /// Number of episodes needed before the model is considered warmed up.
    pub warmup_episodes: u64,

    /// Minimum support count for full confidence.
    pub min_support: u64,

    /// Maximum variance before stability factor drops to zero.
    pub max_variance: f32,

    /// Default novelty z-score threshold for flagging novel configurations.
    pub novelty_z_threshold: f32,

    // Training hyperparameters
    /// Learning rate for SGD.
    pub learning_rate: f32,
    /// Contrastive margin: positive energy should be at least this much lower than negative.
    pub contrastive_margin: f32,
    /// Number of negative tuples to generate per positive tuple.
    pub negatives_per_positive: usize,
    /// Training batch size.
    pub training_batch_size: usize,
}

impl Default for WorldModelConfig {
    fn default() -> Self {
        Self {
            embed_dim: 16,
            embedding_table_size: 256,
            warmup_episodes: 100,
            min_support: 10,
            max_variance: 100.0,
            novelty_z_threshold: 2.0,
            learning_rate: 0.01,
            contrastive_margin: 1.0,
            negatives_per_positive: 4,
            training_batch_size: 64,
        }
    }
}
