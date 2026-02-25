//! Temporal decay scoring for retrieval signals.

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
}
