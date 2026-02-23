//! Scoring functions for top-down evaluation and bottom-up prediction error.
//!
//! These are the high-level functions that compose encoders + energy functions
//! to produce [`CriticReport`] and [`PredictionErrorReport`] outputs.

use crate::encoders::{EventEncoder, MemoryEncoder, PolicyEncoder, StrategyEncoder};
use crate::energy::EnergyStack;
use crate::types::{
    CriticReport, EventFeatures, LayerStats, MemoryFeatures, MismatchLayer, PolicyFeatures,
    PredictionErrorReport, StrategyFeatures, WorldModelConfig,
};

/// Compute a full critic report for a (policy, strategy, memory, event) configuration.
///
/// This is the top-down scoring path: encode all four layers, compute pairwise
/// energies, compute novelty z-score, determine mismatch attribution, and
/// compute confidence.
pub fn score_full(
    policy: &PolicyFeatures,
    strategy: &StrategyFeatures,
    memory: &MemoryFeatures,
    event: &EventFeatures,
    encoders: &Encoders,
    energy_stack: &EnergyStack,
    stats: &ScoringStats,
    config: &WorldModelConfig,
) -> CriticReport {
    // Encode
    let p_emb = encoders.policy.encode(policy);
    let s_emb = encoders.strategy.encode(strategy);
    let m_emb = encoders.memory.encode(memory);
    let e_emb = encoders.event.encode(event);

    // Layer energies
    let (e_ps, e_sm, e_me) = energy_stack.layer_energies(&p_emb, &s_emb, &m_emb, &e_emb);
    let total = e_ps + e_sm + e_me;

    // Novelty z-score
    let novelty_z = stats.total.z_score(total);
    let is_novel = novelty_z > config.novelty_z_threshold;

    // Mismatch attribution: which layer has the highest energy (most incompatible)
    let mismatch_layer = mismatch_from_energies(e_ps, e_sm, e_me);

    // Confidence
    let confidence = compute_confidence(stats, config);

    // Support count: how many similar configurations we've scored
    let support_count = stats.total.count;

    CriticReport {
        total_energy: total,
        policy_strategy_energy: e_ps,
        strategy_memory_energy: e_sm,
        memory_event_energy: e_me,
        novelty_z,
        is_novel,
        mismatch_layer,
        confidence,
        support_count,
    }
}

/// Compute a critic report for policy→strategy only (no memory/event).
///
/// Used when ranking strategy candidates before execution.
pub fn score_strategy_only(
    policy: &PolicyFeatures,
    strategy: &StrategyFeatures,
    encoders: &Encoders,
    energy_stack: &EnergyStack,
    stats: &ScoringStats,
    config: &WorldModelConfig,
) -> CriticReport {
    let p_emb = encoders.policy.encode(policy);
    let s_emb = encoders.strategy.encode(strategy);

    let e_ps = energy_stack.policy_strategy_energy_only(&p_emb, &s_emb);
    let total = e_ps; // only one layer

    let novelty_z = stats.policy_strategy.z_score(e_ps);
    let is_novel = novelty_z > config.novelty_z_threshold;

    let confidence = compute_confidence(stats, config);
    let support_count = stats.total.count;

    CriticReport {
        total_energy: total,
        policy_strategy_energy: e_ps,
        strategy_memory_energy: 0.0,
        memory_event_energy: 0.0,
        novelty_z,
        is_novel,
        mismatch_layer: MismatchLayer::Policy, // only layer scored
        confidence,
        support_count,
    }
}

/// Compute bottom-up prediction error for an observed event.
///
/// Encodes the observed event and its active context (memory, strategy, policy),
/// computes per-layer energies, and returns z-scores indicating how surprising
/// each layer is relative to the running statistics.
pub fn compute_prediction_error(
    observed_event: &EventFeatures,
    memory_context: &MemoryFeatures,
    active_strategy: &StrategyFeatures,
    active_policy: &PolicyFeatures,
    encoders: &Encoders,
    energy_stack: &EnergyStack,
    stats: &ScoringStats,
) -> PredictionErrorReport {
    let p_emb = encoders.policy.encode(active_policy);
    let s_emb = encoders.strategy.encode(active_strategy);
    let m_emb = encoders.memory.encode(memory_context);
    let e_emb = encoders.event.encode(observed_event);

    // Bottom-up: event→memory→strategy→policy
    let event_energy = energy_stack.memory_event.compute(&m_emb, &e_emb);
    let memory_energy = energy_stack.strategy_memory.compute(&s_emb, &m_emb);
    let strategy_energy = energy_stack.policy_strategy.compute(&p_emb, &s_emb);

    let total = event_energy + memory_energy + strategy_energy;

    // Per-layer z-scores
    let event_z = stats.memory_event.z_score(event_energy);
    let memory_z = stats.strategy_memory.z_score(memory_energy);
    let strategy_z = stats.policy_strategy.z_score(strategy_energy);
    let total_z = stats.total.z_score(total);

    // Mismatch: which layer is most surprising
    let mismatch_layer = mismatch_from_z_scores(event_z, memory_z, strategy_z);

    PredictionErrorReport {
        event_energy,
        memory_energy,
        strategy_energy,
        event_z,
        memory_z,
        strategy_z,
        total_z,
        mismatch_layer,
    }
}

// ────────────────────────────────────────────────────────────────
// Shared scoring state
// ────────────────────────────────────────────────────────────────

/// Bundle of all four encoders.
pub struct Encoders {
    pub event: EventEncoder,
    pub memory: MemoryEncoder,
    pub strategy: StrategyEncoder,
    pub policy: PolicyEncoder,
}

/// Running statistics for scoring (used for z-score normalization).
#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct ScoringStats {
    pub total: LayerStats,
    pub policy_strategy: LayerStats,
    pub strategy_memory: LayerStats,
    pub memory_event: LayerStats,
    /// Total training examples processed.
    pub total_trained: u64,
    /// Running average training loss.
    pub avg_loss: f32,
}

impl ScoringStats {
    /// Update all statistics with a new set of layer energies.
    pub fn update(&mut self, e_ps: f32, e_sm: f32, e_me: f32) {
        let total = e_ps + e_sm + e_me;
        self.total.update(total);
        self.policy_strategy.update(e_ps);
        self.strategy_memory.update(e_sm);
        self.memory_event.update(e_me);
    }
}

// ────────────────────────────────────────────────────────────────
// Helpers
// ────────────────────────────────────────────────────────────────

/// Determine which layer has the highest (worst) energy.
fn mismatch_from_energies(e_ps: f32, e_sm: f32, e_me: f32) -> MismatchLayer {
    if e_ps >= e_sm && e_ps >= e_me {
        MismatchLayer::Policy
    } else if e_sm >= e_me {
        MismatchLayer::Strategy
    } else {
        MismatchLayer::Event
    }
}

/// Determine which layer has the highest z-score (most surprising).
fn mismatch_from_z_scores(event_z: f32, memory_z: f32, strategy_z: f32) -> MismatchLayer {
    if strategy_z >= memory_z && strategy_z >= event_z {
        MismatchLayer::Strategy
    } else if memory_z >= event_z {
        MismatchLayer::Memory
    } else {
        MismatchLayer::Event
    }
}

/// Compute confidence from stats and config.
///
/// confidence = min(warmup_factor, support_factor, stability_factor)
fn compute_confidence(stats: &ScoringStats, config: &WorldModelConfig) -> f32 {
    let warmup_factor: f32 = if stats.total_trained >= config.warmup_episodes {
        1.0
    } else {
        0.0
    };

    let support_factor = (stats.total.count as f32 / config.min_support as f32).min(1.0);

    let variance = stats.total.variance();
    let stability_factor = 1.0 - (variance / config.max_variance).clamp(0.0, 1.0);

    warmup_factor.min(support_factor).min(stability_factor)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn make_test_fixtures() -> (Encoders, EnergyStack, ScoringStats, WorldModelConfig) {
        let mut rng = StdRng::seed_from_u64(42);
        let config = WorldModelConfig::default();
        let encoders = Encoders {
            event: EventEncoder::new(config.embed_dim, config.embedding_table_size, &mut rng),
            memory: MemoryEncoder::new(config.embed_dim, config.embedding_table_size, &mut rng),
            strategy: StrategyEncoder::new(config.embed_dim, config.embedding_table_size, &mut rng),
            policy: PolicyEncoder::new(config.embed_dim, config.embedding_table_size, &mut rng),
        };
        let energy_stack = EnergyStack::new(config.embed_dim, &mut rng);
        let stats = ScoringStats::default();
        (encoders, energy_stack, stats, config)
    }

    fn sample_features() -> (
        PolicyFeatures,
        StrategyFeatures,
        MemoryFeatures,
        EventFeatures,
    ) {
        let policy = PolicyFeatures {
            goal_count: 2,
            top_goal_priority: 0.8,
            resource_cpu_percent: 30.0,
            resource_memory_bytes: 1_000_000_000,
            context_fingerprint: 12345,
        };
        let strategy = StrategyFeatures {
            quality_score: 0.9,
            expected_success: 0.85,
            expected_value: 1.2,
            confidence: 0.7,
            goal_bucket_id: 100,
            behavior_signature_hash: 200,
        };
        let memory = MemoryFeatures {
            tier: 1,
            strength: 0.8,
            access_count: 10,
            context_fingerprint: 12345,
            goal_bucket_id: 100,
        };
        let event = EventFeatures {
            event_type_hash: 300,
            action_name_hash: 400,
            context_fingerprint: 12345,
            outcome_success: 1.0,
            significance: 0.7,
            temporal_delta_ns: 500_000_000.0,
            duration_ns: 100_000_000.0,
        };
        (policy, strategy, memory, event)
    }

    #[test]
    fn test_score_full_produces_report() {
        let (encoders, energy_stack, stats, config) = make_test_fixtures();
        let (policy, strategy, memory, event) = sample_features();
        let report = score_full(
            &policy,
            &strategy,
            &memory,
            &event,
            &encoders,
            &energy_stack,
            &stats,
            &config,
        );

        // Report should have finite values
        assert!(report.total_energy.is_finite());
        assert!(report.policy_strategy_energy.is_finite());
        assert!(report.strategy_memory_energy.is_finite());
        assert!(report.memory_event_energy.is_finite());
        assert!(report.novelty_z.is_finite());

        // Total = sum of layer energies
        let sum = report.policy_strategy_energy
            + report.strategy_memory_energy
            + report.memory_event_energy;
        assert!((report.total_energy - sum).abs() < 1e-5);
    }

    #[test]
    fn test_score_strategy_only() {
        let (encoders, energy_stack, stats, config) = make_test_fixtures();
        let (policy, strategy, _, _) = sample_features();
        let report = score_strategy_only(
            &policy,
            &strategy,
            &encoders,
            &energy_stack,
            &stats,
            &config,
        );

        assert!(report.total_energy.is_finite());
        assert_eq!(report.strategy_memory_energy, 0.0);
        assert_eq!(report.memory_event_energy, 0.0);
    }

    #[test]
    fn test_prediction_error_produces_report() {
        let (encoders, energy_stack, stats, _config) = make_test_fixtures();
        let (policy, strategy, memory, event) = sample_features();
        let error = compute_prediction_error(
            &event,
            &memory,
            &strategy,
            &policy,
            &encoders,
            &energy_stack,
            &stats,
        );

        assert!(error.event_energy.is_finite());
        assert!(error.memory_energy.is_finite());
        assert!(error.strategy_energy.is_finite());
        assert!(error.total_z.is_finite());
    }

    #[test]
    fn test_confidence_before_warmup() {
        let stats = ScoringStats::default();
        let config = WorldModelConfig::default();
        let confidence = compute_confidence(&stats, &config);
        assert_eq!(confidence, 0.0, "should be zero before warmup");
    }

    #[test]
    fn test_confidence_after_warmup() {
        let mut stats = ScoringStats::default();
        stats.total_trained = 200; // > warmup_episodes (100)
                                   // Add enough scoring observations for support
        for i in 0..20 {
            stats.total.update(i as f32);
        }
        let config = WorldModelConfig::default();
        let confidence = compute_confidence(&stats, &config);
        assert!(
            confidence > 0.0,
            "should be positive after warmup with support"
        );
    }
}
