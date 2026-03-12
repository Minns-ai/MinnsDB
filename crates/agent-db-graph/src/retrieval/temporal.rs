//! Temporal decay scoring for retrieval signals.
//!
//! Includes both basic exponential decay and importance-modulated decay where
//! frequently-accessed, highly-relevant memories decay slower.

use agent_db_core::types::Timestamp;

/// Exponential decay score based on age relative to a half-life.
///
/// Returns 1.0 when `timestamp == now`, ~0.5 at `half_life_hours`, and
/// asymptotically approaches 0.0 for very old timestamps.
///
/// If `timestamp > now` (future), returns 1.0. If `half_life_hours <= 0`,
/// returns 1.0 (no decay).
pub fn temporal_decay_score(timestamp: Timestamp, now: Timestamp, half_life_hours: f32) -> f32 {
    if half_life_hours <= 0.0 {
        return 1.0;
    }
    if timestamp >= now {
        return 1.0;
    }
    let age_nanos = (now - timestamp) as f64;
    let age_hours = age_nanos / 3_600_000_000_000.0;
    let lambda = (2.0_f64).ln() / half_life_hours as f64;
    (-lambda * age_hours).exp() as f32
}

/// Parameters for importance-modulated temporal decay.
///
/// The decay rate adapts based on a memory's importance:
/// `λ_eff = λ_base * exp(-mu * importance)`
///
/// High-importance memories decay much slower than low-importance ones,
/// even when they share the same base half-life.
#[derive(Debug, Clone)]
pub struct ImportanceDecayParams {
    /// Access frequency (0.0 = never accessed, 1.0 = max in candidate set).
    pub access_frequency: f32,
    /// Semantic relevance to current query (cosine similarity, 0.0–1.0).
    pub relevance: f32,
    /// Strength/significance of the memory (0.0–1.0).
    pub strength: f32,
}

/// Configuration for importance-modulated decay.
#[derive(Debug, Clone)]
pub struct ImportanceDecayConfig {
    /// Weight for access frequency in importance calculation (β).
    pub access_weight: f32,
    /// Weight for semantic relevance in importance calculation (α).
    pub relevance_weight: f32,
    /// Weight for memory strength in importance calculation (γ).
    pub strength_weight: f32,
    /// Modulation strength: how much importance reduces the decay rate (μ).
    /// Higher values → more dramatic slowdown for important memories.
    pub mu: f32,
}

impl Default for ImportanceDecayConfig {
    fn default() -> Self {
        Self {
            access_weight: 0.3,
            relevance_weight: 0.4,
            strength_weight: 0.3,
            mu: 2.0,
        }
    }
}

/// Compute importance score ∈ [0, 1] from component signals.
///
/// `I = α * relevance + β * freq_saturated + γ * strength`
///
/// Access frequency uses saturation: `freq / (1 + freq)` to prevent
/// runaway values from very high access counts.
pub fn compute_importance(params: &ImportanceDecayParams, config: &ImportanceDecayConfig) -> f32 {
    let freq_saturated = params.access_frequency / (1.0 + params.access_frequency);
    let importance = config.relevance_weight * params.relevance
        + config.access_weight * freq_saturated
        + config.strength_weight * params.strength;
    importance.clamp(0.0, 1.0)
}

/// Importance-modulated temporal decay.
///
/// The effective decay rate adapts to importance:
/// `λ_eff = λ_base * exp(-μ * importance)`
///
/// A frequently-accessed, highly-relevant memory decays much slower
/// than a rarely-used one, even if they share the same base half-life.
///
/// Returns a score ∈ (0, 1] where 1.0 means "just formed" or "no decay".
pub fn importance_modulated_decay_score(
    timestamp: Timestamp,
    now: Timestamp,
    half_life_hours: f32,
    params: &ImportanceDecayParams,
    config: &ImportanceDecayConfig,
) -> f32 {
    if half_life_hours <= 0.0 {
        return 1.0;
    }
    if timestamp >= now {
        return 1.0;
    }

    let importance = compute_importance(params, config);

    // λ_base = ln(2) / half_life
    let lambda_base = (2.0_f64).ln() / half_life_hours as f64;
    // λ_eff = λ_base * exp(-μ * importance) — higher importance → lower effective λ
    let lambda_eff = lambda_base * (-config.mu as f64 * importance as f64).exp();

    let age_nanos = (now - timestamp) as f64;
    let age_hours = age_nanos / 3_600_000_000_000.0;

    (-lambda_eff * age_hours).exp() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    const NANOS_PER_HOUR: u64 = 3_600_000_000_000;

    #[test]
    fn test_temporal_decay_at_now() {
        let now = 1_000_000_000_000u64;
        let score = temporal_decay_score(now, now, 24.0);
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_temporal_decay_at_half_life() {
        let now = 100 * NANOS_PER_HOUR;
        let half_life = 24.0;
        let timestamp = now - (half_life as u64) * NANOS_PER_HOUR;
        let score = temporal_decay_score(timestamp, now, half_life);
        assert!((score - 0.5).abs() < 0.01, "Expected ~0.5, got {}", score);
    }

    #[test]
    fn test_temporal_decay_far_past() {
        let now = 1000 * NANOS_PER_HOUR;
        let timestamp = 0u64;
        let score = temporal_decay_score(timestamp, now, 24.0);
        assert!(score < 0.001, "Expected ~0.0, got {}", score);
    }

    #[test]
    fn test_temporal_decay_future_timestamp() {
        let now = 100 * NANOS_PER_HOUR;
        let future = now + 10 * NANOS_PER_HOUR;
        let score = temporal_decay_score(future, now, 24.0);
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_temporal_decay_zero_half_life() {
        let score = temporal_decay_score(0, 1000, 0.0);
        assert!((score - 1.0).abs() < 1e-6);
    }

    // ── Importance-modulated decay tests ─────────────────────────────────

    #[test]
    fn test_compute_importance_all_zero() {
        let params = ImportanceDecayParams {
            access_frequency: 0.0,
            relevance: 0.0,
            strength: 0.0,
        };
        let config = ImportanceDecayConfig::default();
        assert!((compute_importance(&params, &config)).abs() < 1e-6);
    }

    #[test]
    fn test_compute_importance_all_max() {
        let params = ImportanceDecayParams {
            access_frequency: 1.0,
            relevance: 1.0,
            strength: 1.0,
        };
        let config = ImportanceDecayConfig::default();
        let imp = compute_importance(&params, &config);
        // α*1.0 + β*(1/(1+1)) + γ*1.0 = 0.4 + 0.3*0.5 + 0.3 = 0.85
        assert!((imp - 0.85).abs() < 0.01, "Expected ~0.85, got {}", imp);
    }

    #[test]
    fn test_compute_importance_access_saturation() {
        let config = ImportanceDecayConfig::default();
        // Very high access count — should saturate near 1.0
        let params_high = ImportanceDecayParams {
            access_frequency: 100.0,
            relevance: 0.0,
            strength: 0.0,
        };
        let imp_high = compute_importance(&params_high, &config);
        // β * (100/101) ≈ 0.3 * 0.99 ≈ 0.297
        assert!((imp_high - 0.297).abs() < 0.01);

        // Low access count
        let params_low = ImportanceDecayParams {
            access_frequency: 1.0,
            relevance: 0.0,
            strength: 0.0,
        };
        let imp_low = compute_importance(&params_low, &config);
        // β * (1/2) = 0.3 * 0.5 = 0.15
        assert!((imp_low - 0.15).abs() < 0.01);
    }

    #[test]
    fn test_importance_modulated_decay_at_now() {
        let now = 100 * NANOS_PER_HOUR;
        let params = ImportanceDecayParams {
            access_frequency: 0.5,
            relevance: 0.8,
            strength: 0.7,
        };
        let config = ImportanceDecayConfig::default();
        let score = importance_modulated_decay_score(now, now, 24.0, &params, &config);
        assert!((score - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_importance_modulated_high_importance_decays_slower() {
        let now = 200 * NANOS_PER_HOUR;
        let timestamp = now - 48 * NANOS_PER_HOUR; // 2 days ago
        let config = ImportanceDecayConfig::default();

        // Low importance memory
        let low_params = ImportanceDecayParams {
            access_frequency: 0.0,
            relevance: 0.0,
            strength: 0.0,
        };
        let low_score =
            importance_modulated_decay_score(timestamp, now, 24.0, &low_params, &config);

        // High importance memory
        let high_params = ImportanceDecayParams {
            access_frequency: 1.0,
            relevance: 0.9,
            strength: 0.9,
        };
        let high_score =
            importance_modulated_decay_score(timestamp, now, 24.0, &high_params, &config);

        assert!(
            high_score > low_score,
            "High importance should decay slower: high={} > low={}",
            high_score,
            low_score
        );
        // Low importance at 48h with 24h half-life → ~0.25 (standard decay)
        assert!(
            low_score < 0.3,
            "Low importance should decay normally: {}",
            low_score
        );
        // High importance should retain much more
        assert!(
            high_score > 0.5,
            "High importance should retain more: {}",
            high_score
        );
    }

    #[test]
    fn test_importance_modulated_zero_importance_matches_basic() {
        let now = 200 * NANOS_PER_HOUR;
        let timestamp = now - 24 * NANOS_PER_HOUR; // 1 day ago
        let half_life = 24.0;

        let basic = temporal_decay_score(timestamp, now, half_life);

        let zero_params = ImportanceDecayParams {
            access_frequency: 0.0,
            relevance: 0.0,
            strength: 0.0,
        };
        let config = ImportanceDecayConfig::default();
        let modulated =
            importance_modulated_decay_score(timestamp, now, half_life, &zero_params, &config);

        // With zero importance, exp(-μ*0) = 1, so λ_eff = λ_base → same as basic
        assert!(
            (basic - modulated).abs() < 0.01,
            "Zero importance should match basic: basic={}, modulated={}",
            basic,
            modulated
        );
    }

    #[test]
    fn test_importance_decay_config_default() {
        let config = ImportanceDecayConfig::default();
        assert!((config.access_weight - 0.3).abs() < 1e-6);
        assert!((config.relevance_weight - 0.4).abs() < 1e-6);
        assert!((config.strength_weight - 0.3).abs() < 1e-6);
        assert!((config.mu - 2.0).abs() < 1e-6);
    }
}
