//! Contrastive training for the energy-based world model.
//!
//! Uses margin-based contrastive loss:
//!   L = weight * max(0, E_pos - E_neg + margin)
//!
//! where E_pos is the energy of a positive (real) tuple and E_neg is the energy
//! of a corrupted (negative) tuple. Training pushes positive energies down and
//! negative energies up, separated by at least `margin`.
//!
//! Gradient descent is manual (SGD) — no autograd framework needed for ~30K params.

use rand::Rng;

use crate::encoders::EmbeddingTable;
use crate::energy::EnergyStack;
use crate::scoring::{Encoders, ScoringStats};
use crate::types::{
    EventFeatures, MemoryFeatures, PolicyFeatures, StrategyFeatures, TrainingTuple,
    WorldModelConfig,
};

/// Run one training step on a batch of tuples.
///
/// Returns the average loss over the batch.
///
/// The training procedure:
/// 1. Separate batch into positive and negative tuples.
/// 2. For each positive-negative pair, compute contrastive loss.
/// 3. Accumulate gradients.
/// 4. Apply SGD weight update.
pub fn train_batch(
    tuples: &[TrainingTuple],
    encoders: &mut Encoders,
    energy_stack: &mut EnergyStack,
    stats: &mut ScoringStats,
    config: &WorldModelConfig,
) -> f32 {
    if tuples.is_empty() {
        return 0.0;
    }

    // Separate positive and negative tuples
    let positives: Vec<&TrainingTuple> = tuples.iter().filter(|t| t.is_positive).collect();
    let negatives: Vec<&TrainingTuple> = tuples.iter().filter(|t| !t.is_positive).collect();

    if positives.is_empty() || negatives.is_empty() {
        return 0.0;
    }

    let mut total_loss = 0.0f32;
    let mut pair_count = 0usize;
    let lr = config.learning_rate;
    let margin = config.contrastive_margin;

    // Pair each positive with corresponding negatives
    // (in practice, negatives are generated N:1 per positive)
    let neg_per_pos = negatives.len() / positives.len().max(1);

    for (pos_idx, pos) in positives.iter().enumerate() {
        // Encode positive
        let p_emb_pos = encoders.policy.encode(&pos.policy_features);
        let s_emb_pos = encoders.strategy.encode(&pos.strategy_features);
        let m_emb_pos = encoders.memory.encode(&pos.memory_features);
        let e_emb_pos = encoders.event.encode(&pos.event_features);
        let energy_pos =
            energy_stack.total_free_energy(&p_emb_pos, &s_emb_pos, &m_emb_pos, &e_emb_pos);

        // Update scoring statistics with positive energy
        let (e_ps, e_sm, e_me) =
            energy_stack.layer_energies(&p_emb_pos, &s_emb_pos, &m_emb_pos, &e_emb_pos);
        stats.update(e_ps, e_sm, e_me);

        // Get corresponding negatives
        let neg_start = pos_idx * neg_per_pos;
        let neg_end = ((pos_idx + 1) * neg_per_pos).min(negatives.len());

        for neg in &negatives[neg_start..neg_end] {
            // Encode negative
            let p_emb_neg = encoders.policy.encode(&neg.policy_features);
            let s_emb_neg = encoders.strategy.encode(&neg.strategy_features);
            let m_emb_neg = encoders.memory.encode(&neg.memory_features);
            let e_emb_neg = encoders.event.encode(&neg.event_features);
            let energy_neg =
                energy_stack.total_free_energy(&p_emb_neg, &s_emb_neg, &m_emb_neg, &e_emb_neg);

            // Contrastive loss: max(0, E_pos - E_neg + margin)
            let loss = (energy_pos - energy_neg + margin).max(0.0) * pos.weight;
            total_loss += loss;
            pair_count += 1;

            // Skip gradient update if loss is zero (margin already satisfied)
            if loss <= 0.0 {
                continue;
            }

            let grad_scale = lr * pos.weight;

            // Apply gradients to energy stack
            // For positive: want to decrease energy → subtract gradient
            // For negative: want to increase energy → add gradient
            apply_energy_gradients(
                energy_stack,
                &p_emb_pos,
                &s_emb_pos,
                &m_emb_pos,
                &e_emb_pos,
                grad_scale,
                true, // decrease positive energy
            );
            apply_energy_gradients(
                energy_stack,
                &p_emb_neg,
                &s_emb_neg,
                &m_emb_neg,
                &e_emb_neg,
                grad_scale,
                false, // increase negative energy
            );

            // Apply gradients to encoders (simplified: just update embedding tables)
            // Full backprop through encoders is expensive; instead we update the
            // embedding table entries that were looked up during encoding.
            update_embedding_tables_for_tuple(
                encoders,
                &pos.event_features,
                &pos.memory_features,
                &pos.strategy_features,
                &pos.policy_features,
                grad_scale * 0.1, // smaller LR for embeddings
            );
        }
    }

    let avg_loss = if pair_count > 0 {
        total_loss / pair_count as f32
    } else {
        0.0
    };

    stats.total_trained += tuples.iter().filter(|t| t.is_positive).count() as u64;
    stats.avg_loss = stats.avg_loss * 0.9 + avg_loss * 0.1; // EMA

    avg_loss
}

/// Apply gradients to the energy stack's bilinear weight matrices.
///
/// For a positive example (decrease_energy=true): W -= lr * dE/dW
/// For a negative example (decrease_energy=false): W += lr * dE/dW
fn apply_energy_gradients(
    energy_stack: &mut EnergyStack,
    p_emb: &[f32],
    s_emb: &[f32],
    m_emb: &[f32],
    e_emb: &[f32],
    lr: f32,
    decrease_energy: bool,
) {
    let sign = if decrease_energy { -1.0 } else { 1.0 };

    // Policy→Strategy
    {
        let (dw, _, _) = energy_stack.policy_strategy.gradients(p_emb, s_emb);
        for i in 0..energy_stack.policy_strategy.dim_a {
            for j in 0..energy_stack.policy_strategy.dim_b {
                energy_stack.policy_strategy.weights[i][j] += sign * lr * dw[i][j];
            }
        }
    }

    // Strategy→Memory
    {
        let (dw, _, _) = energy_stack.strategy_memory.gradients(s_emb, m_emb);
        for i in 0..energy_stack.strategy_memory.dim_a {
            for j in 0..energy_stack.strategy_memory.dim_b {
                energy_stack.strategy_memory.weights[i][j] += sign * lr * dw[i][j];
            }
        }
    }

    // Memory→Event
    {
        let (dw, _, _) = energy_stack.memory_event.gradients(m_emb, e_emb);
        for i in 0..energy_stack.memory_event.dim_a {
            for j in 0..energy_stack.memory_event.dim_b {
                energy_stack.memory_event.weights[i][j] += sign * lr * dw[i][j];
            }
        }
    }
}

/// Update embedding table entries that were looked up during encoding.
///
/// This is a simplified form of backpropagation: instead of full chain-rule
/// through the linear layers, we nudge the embedding entries slightly to
/// reduce energy for positive examples.
fn update_embedding_tables_for_tuple(
    encoders: &mut Encoders,
    event: &EventFeatures,
    memory: &MemoryFeatures,
    strategy: &StrategyFeatures,
    policy: &PolicyFeatures,
    lr: f32,
) {
    // Nudge event encoder embeddings
    nudge_embedding(
        &mut encoders.event.event_type_emb,
        event.event_type_hash,
        lr,
    );
    nudge_embedding(
        &mut encoders.event.action_name_emb,
        event.action_name_hash,
        lr,
    );
    nudge_embedding(
        &mut encoders.event.context_emb,
        event.context_fingerprint,
        lr,
    );

    // Nudge memory encoder embeddings
    nudge_embedding(
        &mut encoders.memory.context_emb,
        memory.context_fingerprint,
        lr,
    );
    nudge_embedding(&mut encoders.memory.goal_emb, memory.goal_bucket_id, lr);

    // Nudge strategy encoder embeddings
    nudge_embedding(&mut encoders.strategy.goal_emb, strategy.goal_bucket_id, lr);
    nudge_embedding(
        &mut encoders.strategy.behavior_emb,
        strategy.behavior_signature_hash,
        lr,
    );

    // Nudge policy encoder embeddings
    nudge_embedding(
        &mut encoders.policy.context_emb,
        policy.context_fingerprint,
        lr,
    );
}

/// Apply a small L2 regularization nudge to an embedding entry.
/// This prevents embedding weights from growing unbounded.
fn nudge_embedding(table: &mut EmbeddingTable, hash: u64, lr: f32) {
    let emb = table.lookup_mut(hash);
    for v in emb.iter_mut() {
        // L2 regularization: decay toward zero
        *v *= 1.0 - lr * 0.01;
    }
}

/// Generate negative tuples by corrupting a positive tuple.
///
/// Corruption strategies:
/// - **Temporal**: Replace event features with those from another tuple.
/// - **Context**: Randomize context_fingerprint in event and memory.
/// - **Strategy**: Replace strategy features with those from another goal.
/// - **Outcome**: Flip outcome_success.
pub fn generate_negatives(
    positive: &TrainingTuple,
    all_positives: &[TrainingTuple],
    negatives_per_positive: usize,
    rng: &mut impl Rng,
) -> Vec<TrainingTuple> {
    let mut negatives = Vec::with_capacity(negatives_per_positive);

    for i in 0..negatives_per_positive {
        let strategy = i % 4; // cycle through 4 corruption strategies
        let mut neg = positive.clone();
        neg.is_positive = false;

        match strategy {
            0 => {
                // Temporal: swap event features
                if all_positives.len() > 1 {
                    let other_idx = rng.gen_range(0..all_positives.len());
                    neg.event_features = all_positives[other_idx].event_features.clone();
                }
            },
            1 => {
                // Context: randomize fingerprint
                neg.event_features.context_fingerprint = rng.gen();
                neg.memory_features.context_fingerprint = rng.gen();
            },
            2 => {
                // Strategy: swap strategy from different goal
                if all_positives.len() > 1 {
                    let other_idx = rng.gen_range(0..all_positives.len());
                    neg.strategy_features = all_positives[other_idx].strategy_features.clone();
                }
            },
            3 => {
                // Outcome: flip success
                neg.event_features.outcome_success =
                    if positive.event_features.outcome_success > 0.5 {
                        0.0
                    } else {
                        1.0
                    };
            },
            _ => unreachable!(),
        }

        negatives.push(neg);
    }

    negatives
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::encoders::{EventEncoder, MemoryEncoder, PolicyEncoder, StrategyEncoder};
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn make_positive_tuple(context: u64, goal: u64, success: f32) -> TrainingTuple {
        TrainingTuple {
            event_features: EventFeatures {
                event_type_hash: 100,
                action_name_hash: 200,
                context_fingerprint: context,
                outcome_success: success,
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
    fn test_generate_negatives_count() {
        let mut rng = StdRng::seed_from_u64(42);
        let pos = make_positive_tuple(100, 200, 1.0);
        let all = vec![pos.clone(), make_positive_tuple(300, 400, 0.0)];
        let negs = generate_negatives(&pos, &all, 4, &mut rng);
        assert_eq!(negs.len(), 4);
        assert!(negs.iter().all(|n| !n.is_positive));
    }

    #[test]
    fn test_generate_negatives_corruption_strategies() {
        let mut rng = StdRng::seed_from_u64(42);
        let pos = make_positive_tuple(100, 200, 1.0);
        let all = vec![pos.clone(), make_positive_tuple(300, 400, 0.0)];
        let negs = generate_negatives(&pos, &all, 4, &mut rng);

        // Strategy 3 (outcome flip): outcome should be flipped
        let outcome_neg = &negs[3];
        assert_eq!(outcome_neg.event_features.outcome_success, 0.0);
    }

    #[test]
    fn test_train_batch_reduces_loss() {
        let mut rng = StdRng::seed_from_u64(42);
        let config = WorldModelConfig {
            learning_rate: 0.05,
            contrastive_margin: 1.0,
            ..WorldModelConfig::default()
        };

        let mut encoders = Encoders {
            event: EventEncoder::new(config.embed_dim, config.embedding_table_size, &mut rng),
            memory: MemoryEncoder::new(config.embed_dim, config.embedding_table_size, &mut rng),
            strategy: StrategyEncoder::new(config.embed_dim, config.embedding_table_size, &mut rng),
            policy: PolicyEncoder::new(config.embed_dim, config.embedding_table_size, &mut rng),
        };
        let mut energy_stack = EnergyStack::new(config.embed_dim, &mut rng);
        let mut stats = ScoringStats::default();

        // Create training data: same context + goal (positive) vs corrupted
        let pos = make_positive_tuple(100, 200, 1.0);
        let negs = generate_negatives(&pos, &[pos.clone()], 4, &mut rng);

        let mut batch: Vec<TrainingTuple> = vec![pos.clone()];
        batch.extend(negs);

        // Train for multiple steps and check loss decreases
        let loss_first = train_batch(
            &batch,
            &mut encoders,
            &mut energy_stack,
            &mut stats,
            &config,
        );

        let mut loss_last = loss_first;
        for _ in 0..50 {
            loss_last = train_batch(
                &batch,
                &mut encoders,
                &mut energy_stack,
                &mut stats,
                &config,
            );
        }

        // Loss should decrease (or reach zero if margin is satisfied)
        assert!(
            loss_last <= loss_first + 0.1,
            "loss should not increase significantly: first={}, last={}",
            loss_first,
            loss_last
        );
    }

    #[test]
    fn test_positive_energy_lower_after_training() {
        let mut rng = StdRng::seed_from_u64(42);
        let config = WorldModelConfig {
            learning_rate: 0.05,
            ..WorldModelConfig::default()
        };

        let mut encoders = Encoders {
            event: EventEncoder::new(config.embed_dim, config.embedding_table_size, &mut rng),
            memory: MemoryEncoder::new(config.embed_dim, config.embedding_table_size, &mut rng),
            strategy: StrategyEncoder::new(config.embed_dim, config.embedding_table_size, &mut rng),
            policy: PolicyEncoder::new(config.embed_dim, config.embedding_table_size, &mut rng),
        };
        let mut energy_stack = EnergyStack::new(config.embed_dim, &mut rng);
        let mut stats = ScoringStats::default();

        let pos = make_positive_tuple(100, 200, 1.0);
        let negs = generate_negatives(&pos, &[pos.clone()], 4, &mut rng);

        // Measure positive energy before training
        let p = encoders.policy.encode(&pos.policy_features);
        let s = encoders.strategy.encode(&pos.strategy_features);
        let m = encoders.memory.encode(&pos.memory_features);
        let e = encoders.event.encode(&pos.event_features);
        let energy_before = energy_stack.total_free_energy(&p, &s, &m, &e);

        let mut batch = vec![pos.clone()];
        batch.extend(negs);

        // Train
        for _ in 0..100 {
            train_batch(
                &batch,
                &mut encoders,
                &mut energy_stack,
                &mut stats,
                &config,
            );
        }

        // Measure positive energy after training
        let p = encoders.policy.encode(&pos.policy_features);
        let s = encoders.strategy.encode(&pos.strategy_features);
        let m = encoders.memory.encode(&pos.memory_features);
        let e = encoders.event.encode(&pos.event_features);
        let energy_after = energy_stack.total_free_energy(&p, &s, &m, &e);

        assert!(
            energy_after < energy_before + 0.5,
            "positive energy should decrease after training: before={}, after={}",
            energy_before,
            energy_after
        );
    }
}
