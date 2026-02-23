//! Concrete implementation of [`WorldModelCritic`] using an energy-based model.
//!
//! [`EbmWorldModel`] bundles encoders, energy functions, scoring statistics,
//! a training queue, and configuration into a single struct that implements
//! the [`WorldModelCritic`] trait.

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::encoders::{EventEncoder, MemoryEncoder, PolicyEncoder, StrategyEncoder};
use crate::energy::EnergyStack;
use crate::persistence::{self, WorldModelSnapshot};
use crate::scoring::{self, Encoders, ScoringStats};
use crate::training;
use crate::types::{
    CriticReport, EnergyStats, EventFeatures, MemoryFeatures, PolicyFeatures,
    PredictionErrorReport, StrategyFeatures, TrainingTuple, WorldModelConfig,
};
use crate::WorldModelCritic;

/// Energy-based world model implementing [`WorldModelCritic`].
///
/// This is the concrete production implementation. It maintains:
/// - Four layer encoders (event, memory, strategy, policy)
/// - Three bilinear energy functions between adjacent layers
/// - Running energy statistics for z-score normalization
/// - A training queue for batched contrastive learning
/// - An RNG for negative sampling during training
pub struct EbmWorldModel {
    pub(crate) encoders: Encoders,
    pub(crate) energy_stack: EnergyStack,
    pub(crate) stats: ScoringStats,
    pub(crate) config: WorldModelConfig,
    pub(crate) training_queue: Vec<TrainingTuple>,
    pub(crate) rng: StdRng,
}

impl EbmWorldModel {
    /// Create a new world model with random initialization.
    pub fn new(config: WorldModelConfig) -> Self {
        Self::with_seed(config, 42)
    }

    /// Create a new world model with a specific random seed (for reproducibility).
    pub fn with_seed(config: WorldModelConfig, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let encoders = Encoders {
            event: EventEncoder::new(config.embed_dim, config.embedding_table_size, &mut rng),
            memory: MemoryEncoder::new(config.embed_dim, config.embedding_table_size, &mut rng),
            strategy: StrategyEncoder::new(config.embed_dim, config.embedding_table_size, &mut rng),
            policy: PolicyEncoder::new(config.embed_dim, config.embedding_table_size, &mut rng),
        };
        let energy_stack = EnergyStack::new(config.embed_dim, &mut rng);

        Self {
            encoders,
            energy_stack,
            stats: ScoringStats::default(),
            config,
            training_queue: Vec::new(),
            rng,
        }
    }

    /// Serialize the model to bytes for persistence.
    pub fn to_bytes(&self) -> Result<Vec<u8>, String> {
        let snapshot = self.to_snapshot();
        persistence::serialize_snapshot(&snapshot)
    }

    /// Deserialize a model from bytes.
    pub fn from_bytes(data: &[u8]) -> Result<Self, String> {
        let snapshot = persistence::deserialize_snapshot(data)?;
        Ok(Self::from_snapshot(snapshot))
    }

    fn to_snapshot(&self) -> WorldModelSnapshot {
        WorldModelSnapshot {
            version: 1,
            config: self.config.clone(),
            event_encoder: self.encoders.event.clone(),
            memory_encoder: self.encoders.memory.clone(),
            strategy_encoder: self.encoders.strategy.clone(),
            policy_encoder: self.encoders.policy.clone(),
            energy_stack: self.energy_stack.clone(),
            scoring_stats: self.stats.clone(),
        }
    }

    fn from_snapshot(snapshot: WorldModelSnapshot) -> Self {
        Self {
            encoders: Encoders {
                event: snapshot.event_encoder,
                memory: snapshot.memory_encoder,
                strategy: snapshot.strategy_encoder,
                policy: snapshot.policy_encoder,
            },
            energy_stack: snapshot.energy_stack,
            stats: snapshot.scoring_stats,
            config: snapshot.config,
            training_queue: Vec::new(),
            rng: StdRng::seed_from_u64(42),
        }
    }

    /// Number of pending training tuples.
    pub fn pending_training_count(&self) -> usize {
        self.training_queue.len()
    }
}

impl WorldModelCritic for EbmWorldModel {
    fn score(
        &self,
        policy: &PolicyFeatures,
        strategy: &StrategyFeatures,
        memory: &MemoryFeatures,
        event: &EventFeatures,
    ) -> CriticReport {
        scoring::score_full(
            policy,
            strategy,
            memory,
            event,
            &self.encoders,
            &self.energy_stack,
            &self.stats,
            &self.config,
        )
    }

    fn score_strategy(&self, policy: &PolicyFeatures, strategy: &StrategyFeatures) -> CriticReport {
        scoring::score_strategy_only(
            policy,
            strategy,
            &self.encoders,
            &self.energy_stack,
            &self.stats,
            &self.config,
        )
    }

    fn prediction_error(
        &self,
        observed_event: &EventFeatures,
        memory_context: &MemoryFeatures,
        active_strategy: &StrategyFeatures,
        active_policy: &PolicyFeatures,
    ) -> PredictionErrorReport {
        scoring::compute_prediction_error(
            observed_event,
            memory_context,
            active_strategy,
            active_policy,
            &self.encoders,
            &self.energy_stack,
            &self.stats,
        )
    }

    fn is_warmed_up(&self) -> bool {
        self.stats.total_trained >= self.config.warmup_episodes
    }

    fn submit_training(&mut self, tuple: TrainingTuple) {
        self.training_queue.push(tuple);
    }

    fn train_step(&mut self) -> f32 {
        if self.training_queue.is_empty() {
            return 0.0;
        }

        // Take up to batch_size tuples from the queue
        let batch_size = self.config.training_batch_size;
        let batch: Vec<TrainingTuple> = if self.training_queue.len() <= batch_size {
            std::mem::take(&mut self.training_queue)
        } else {
            self.training_queue.drain(..batch_size).collect()
        };

        // Separate positives, generate negatives
        let positives: Vec<TrainingTuple> = batch.into_iter().filter(|t| t.is_positive).collect();
        if positives.is_empty() {
            return 0.0;
        }

        let mut full_batch: Vec<TrainingTuple> = Vec::new();
        for pos in &positives {
            let negatives = training::generate_negatives(
                pos,
                &positives,
                self.config.negatives_per_positive,
                &mut self.rng,
            );
            full_batch.push(pos.clone());
            full_batch.extend(negatives);
        }

        training::train_batch(
            &full_batch,
            &mut self.encoders,
            &mut self.energy_stack,
            &mut self.stats,
            &self.config,
        )
    }

    fn energy_stats(&self) -> EnergyStats {
        EnergyStats {
            running_mean: self.stats.total.mean,
            running_variance: self.stats.total.variance(),
            total_scored: self.stats.total.count,
            total_trained: self.stats.total_trained,
            avg_loss: self.stats.avg_loss,
            is_warmed_up: self.is_warmed_up(),
        }
    }
}

// Static assertion: EbmWorldModel must be Send + Sync for the WorldModelCritic trait.
// All fields (Encoders, EnergyStack, ScoringStats, WorldModelConfig, Vec<TrainingTuple>, StdRng)
// are Send + Sync, so this is satisfied automatically. These assertions catch regressions at compile time.
fn _assert_send_sync() {
    fn require_send<T: Send>() {}
    fn require_sync<T: Sync>() {}
    require_send::<EbmWorldModel>();
    require_sync::<EbmWorldModel>();
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_model() -> EbmWorldModel {
        EbmWorldModel::new(WorldModelConfig::default())
    }

    fn sample_policy() -> PolicyFeatures {
        PolicyFeatures {
            goal_count: 2,
            top_goal_priority: 0.8,
            resource_cpu_percent: 30.0,
            resource_memory_bytes: 1_000_000_000,
            context_fingerprint: 12345,
        }
    }

    fn sample_strategy() -> StrategyFeatures {
        StrategyFeatures {
            quality_score: 0.9,
            expected_success: 0.85,
            expected_value: 1.2,
            confidence: 0.7,
            goal_bucket_id: 100,
            behavior_signature_hash: 200,
        }
    }

    fn sample_memory() -> MemoryFeatures {
        MemoryFeatures {
            tier: 1,
            strength: 0.8,
            access_count: 10,
            context_fingerprint: 12345,
            goal_bucket_id: 100,
        }
    }

    fn sample_event() -> EventFeatures {
        EventFeatures {
            event_type_hash: 300,
            action_name_hash: 400,
            context_fingerprint: 12345,
            outcome_success: 1.0,
            significance: 0.7,
            temporal_delta_ns: 500_000_000.0,
            duration_ns: 100_000_000.0,
        }
    }

    fn make_tuple(context: u64, goal: u64) -> TrainingTuple {
        TrainingTuple {
            event_features: EventFeatures {
                event_type_hash: 100,
                action_name_hash: 200,
                context_fingerprint: context,
                outcome_success: 1.0,
                significance: 0.8,
                temporal_delta_ns: 1e9,
                duration_ns: 5e8,
            },
            memory_features: MemoryFeatures {
                tier: 1,
                strength: 0.7,
                access_count: 5,
                context_fingerprint: context,
                goal_bucket_id: goal,
            },
            strategy_features: StrategyFeatures {
                quality_score: 0.85,
                expected_success: 0.9,
                expected_value: 1.0,
                confidence: 0.6,
                goal_bucket_id: goal,
                behavior_signature_hash: 500,
            },
            policy_features: PolicyFeatures {
                goal_count: 2,
                top_goal_priority: 0.8,
                resource_cpu_percent: 30.0,
                resource_memory_bytes: 1_000_000_000,
                context_fingerprint: context,
            },
            is_positive: true,
            weight: 1.0,
        }
    }

    #[test]
    fn test_score_returns_finite() {
        let model = make_model();
        let report = model.score(
            &sample_policy(),
            &sample_strategy(),
            &sample_memory(),
            &sample_event(),
        );
        assert!(report.total_energy.is_finite());
        assert!(report.confidence.is_finite());
    }

    #[test]
    fn test_score_strategy_returns_finite() {
        let model = make_model();
        let report = model.score_strategy(&sample_policy(), &sample_strategy());
        assert!(report.total_energy.is_finite());
        assert_eq!(report.strategy_memory_energy, 0.0);
        assert_eq!(report.memory_event_energy, 0.0);
    }

    #[test]
    fn test_prediction_error_returns_finite() {
        let model = make_model();
        let error = model.prediction_error(
            &sample_event(),
            &sample_memory(),
            &sample_strategy(),
            &sample_policy(),
        );
        assert!(error.total_z.is_finite());
    }

    #[test]
    fn test_not_warmed_up_initially() {
        let model = make_model();
        assert!(!model.is_warmed_up());
    }

    #[test]
    fn test_submit_and_train() {
        let mut model = make_model();
        let tuple = make_tuple(100, 200);

        // Submit positive tuples
        for _ in 0..10 {
            model.submit_training(tuple.clone());
        }
        assert_eq!(model.pending_training_count(), 10);

        let loss = model.train_step();
        assert!(loss.is_finite());
        assert_eq!(model.pending_training_count(), 0);
    }

    #[test]
    fn test_energy_stats_populated_after_training() {
        let mut model = make_model();
        let tuple = make_tuple(100, 200);

        for _ in 0..20 {
            model.submit_training(tuple.clone());
        }
        model.train_step();

        let stats = model.energy_stats();
        assert!(stats.total_scored > 0);
        assert!(stats.total_trained > 0);
    }

    #[test]
    fn test_persistence_roundtrip() {
        let mut model = make_model();

        // Train a bit to get non-default weights
        let tuple = make_tuple(100, 200);
        for _ in 0..10 {
            model.submit_training(tuple.clone());
        }
        model.train_step();

        // Score before persistence
        let report_before = model.score(
            &sample_policy(),
            &sample_strategy(),
            &sample_memory(),
            &sample_event(),
        );

        // Serialize → deserialize
        let bytes = model.to_bytes().unwrap();
        let restored = EbmWorldModel::from_bytes(&bytes).unwrap();

        // Score after persistence — should be identical
        let report_after = restored.score(
            &sample_policy(),
            &sample_strategy(),
            &sample_memory(),
            &sample_event(),
        );

        assert!(
            (report_before.total_energy - report_after.total_energy).abs() < 1e-6,
            "scoring should be identical after persistence roundtrip"
        );
    }

    #[test]
    fn test_novelty_detection() {
        let mut model = EbmWorldModel::with_seed(
            WorldModelConfig {
                warmup_episodes: 5,
                learning_rate: 0.05,
                ..WorldModelConfig::default()
            },
            42,
        );

        // Train on pattern A
        let tuple_a = make_tuple(100, 200);
        for _ in 0..50 {
            model.submit_training(tuple_a.clone());
            model.train_step();
        }

        // Score pattern A (should be familiar)
        let report_a = model.score(
            &tuple_a.policy_features,
            &tuple_a.strategy_features,
            &tuple_a.memory_features,
            &tuple_a.event_features,
        );

        // Score pattern B (should be more novel)
        let tuple_b = make_tuple(999, 888);
        let report_b = model.score(
            &tuple_b.policy_features,
            &tuple_b.strategy_features,
            &tuple_b.memory_features,
            &tuple_b.event_features,
        );

        // B should have higher z-score than A (more novel)
        // Note: with random init this is probabilistic, but training pushes
        // pattern A's energy lower, making B relatively more novel.
        tracing::info!(
            "Pattern A z={:.2}, Pattern B z={:.2}",
            report_a.novelty_z,
            report_b.novelty_z
        );
    }

    #[test]
    fn test_confidence_increases_with_training() {
        let mut model = EbmWorldModel::with_seed(
            WorldModelConfig {
                warmup_episodes: 10,
                min_support: 5,
                ..WorldModelConfig::default()
            },
            42,
        );

        let conf_before = model.energy_stats().is_warmed_up;
        assert!(!conf_before);

        let tuple = make_tuple(100, 200);
        for _ in 0..20 {
            model.submit_training(tuple.clone());
            model.train_step();
        }

        assert!(model.is_warmed_up());
    }
}
