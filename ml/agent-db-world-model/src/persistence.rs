//! Serialize/deserialize world model weights.
//!
//! Uses a simple version-envelope format compatible with the
//! `agent_db_storage::versioned` pattern: `[magic: 0x00][version: u8][payload...]`
//!
//! The payload is MessagePack-serialized model state.

use serde::{Deserialize, Serialize};

use crate::encoders::{EventEncoder, MemoryEncoder, PolicyEncoder, StrategyEncoder};
use crate::energy::EnergyStack;
use crate::scoring::ScoringStats;
use crate::types::WorldModelConfig;

/// Magic byte for world model state envelope.
const WM_MAGIC: u8 = 0x00;
/// Current format version.
const WM_VERSION: u8 = 1;

/// Serializable snapshot of the entire world model state.
#[derive(Debug, Serialize, Deserialize)]
pub struct WorldModelSnapshot {
    pub version: u8,
    pub config: WorldModelConfig,
    pub event_encoder: EventEncoder,
    pub memory_encoder: MemoryEncoder,
    pub strategy_encoder: StrategyEncoder,
    pub policy_encoder: PolicyEncoder,
    pub energy_stack: EnergyStack,
    pub scoring_stats: ScoringStats,
}

/// Serialize a world model snapshot to bytes.
pub fn serialize_snapshot(snapshot: &WorldModelSnapshot) -> Result<Vec<u8>, String> {
    let payload =
        rmp_serde::to_vec(snapshot).map_err(|e| format!("serialization failed: {}", e))?;
    let mut out = Vec::with_capacity(2 + payload.len());
    out.push(WM_MAGIC);
    out.push(WM_VERSION);
    out.extend_from_slice(&payload);
    Ok(out)
}

/// Deserialize a world model snapshot from bytes.
pub fn deserialize_snapshot(data: &[u8]) -> Result<WorldModelSnapshot, String> {
    if data.len() < 2 {
        return Err("data too short for world model snapshot".to_string());
    }

    let (version, payload) = if data[0] == WM_MAGIC {
        (data[1], &data[2..])
    } else {
        // Legacy format (no envelope) — try raw msgpack
        (0u8, data)
    };

    match version {
        0 | 1 => rmp_serde::from_slice(payload)
            .map_err(|e| format!("deserialization failed (v{}): {}", version, e)),
        _ => Err(format!("unsupported world model version: {}", version)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::WorldModelConfig;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    #[test]
    fn test_snapshot_roundtrip() {
        let mut rng = StdRng::seed_from_u64(42);
        let config = WorldModelConfig::default();

        let snapshot = WorldModelSnapshot {
            version: WM_VERSION,
            config: config.clone(),
            event_encoder: EventEncoder::new(
                config.embed_dim,
                config.embedding_table_size,
                &mut rng,
            ),
            memory_encoder: MemoryEncoder::new(
                config.embed_dim,
                config.embedding_table_size,
                &mut rng,
            ),
            strategy_encoder: StrategyEncoder::new(
                config.embed_dim,
                config.embedding_table_size,
                &mut rng,
            ),
            policy_encoder: PolicyEncoder::new(
                config.embed_dim,
                config.embedding_table_size,
                &mut rng,
            ),
            energy_stack: EnergyStack::new(config.embed_dim, &mut rng),
            scoring_stats: ScoringStats::default(),
        };

        let bytes = serialize_snapshot(&snapshot).unwrap();
        assert_eq!(bytes[0], WM_MAGIC);
        assert_eq!(bytes[1], WM_VERSION);

        let restored = deserialize_snapshot(&bytes).unwrap();

        // Verify config roundtrip
        assert_eq!(restored.config.embed_dim, config.embed_dim);
        assert_eq!(restored.config.learning_rate, config.learning_rate);

        // Verify weights are identical (spot check one weight)
        assert_eq!(
            snapshot.energy_stack.policy_strategy.weights[0][0],
            restored.energy_stack.policy_strategy.weights[0][0],
        );
    }

    #[test]
    fn test_snapshot_too_short() {
        let result = deserialize_snapshot(&[0x00]);
        assert!(result.is_err());
    }

    #[test]
    fn test_snapshot_unsupported_version() {
        let result = deserialize_snapshot(&[0x00, 99, 0x80]);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("unsupported"));
    }
}
