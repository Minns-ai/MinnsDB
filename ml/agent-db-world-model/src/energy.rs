//! Energy functions for measuring compatibility between layer embeddings.
//!
//! The energy-based model uses bilinear energy functions to score how
//! compatible two layer embeddings are. Lower energy = more compatible.
//! Total free energy is the sum of all pairwise layer energies.

use rand::Rng;
use serde::{Deserialize, Serialize};

// ────────────────────────────────────────────────────────────────
// Bilinear energy
// ────────────────────────────────────────────────────────────────

/// Bilinear energy function: `E(a, b) = -a^T W b`
///
/// Measures compatibility between two embedding vectors via a learned
/// interaction matrix. Lower (more negative) energy means higher compatibility.
///
/// The negative sign means that when `a` and `b` are compatible (as learned
/// from positive training examples), the energy will be low (negative).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BilinearEnergy {
    /// Interaction matrix W, shape [dim_a][dim_b].
    pub weights: Vec<Vec<f32>>,
    pub dim_a: usize,
    pub dim_b: usize,
}

impl BilinearEnergy {
    /// Create a new bilinear energy function with small random weights.
    pub fn new(dim_a: usize, dim_b: usize, rng: &mut impl Rng) -> Self {
        let scale = 1.0 / ((dim_a * dim_b) as f32).sqrt();
        let weights = (0..dim_a)
            .map(|_| (0..dim_b).map(|_| rng.gen_range(-scale..scale)).collect())
            .collect();
        Self {
            weights,
            dim_a,
            dim_b,
        }
    }

    /// Compute energy: `E(a, b) = -a^T W b`.
    ///
    /// Returns a scalar energy value. Lower = more compatible.
    #[allow(clippy::needless_range_loop)]
    pub fn compute(&self, a: &[f32], b: &[f32]) -> f32 {
        debug_assert_eq!(a.len(), self.dim_a);
        debug_assert_eq!(b.len(), self.dim_b);

        let mut energy = 0.0f32;
        for i in 0..self.dim_a {
            let mut row_sum = 0.0f32;
            for j in 0..self.dim_b {
                row_sum += self.weights[i][j] * b[j];
            }
            energy += a[i] * row_sum;
        }
        -energy // negative: compatible pairs → low energy
    }

    /// Compute gradients of energy w.r.t. W, a, and b.
    ///
    /// Returns (dE/dW, dE/da, dE/db).
    /// Since E = -a^T W b:
    ///   dE/dW = -a * b^T  (outer product)
    ///   dE/da = -W * b
    ///   dE/db = -W^T * a
    #[allow(clippy::needless_range_loop)]
    pub fn gradients(&self, a: &[f32], b: &[f32]) -> (Vec<Vec<f32>>, Vec<f32>, Vec<f32>) {
        // dE/dW[i][j] = -a[i] * b[j]
        let dw: Vec<Vec<f32>> = (0..self.dim_a)
            .map(|i| (0..self.dim_b).map(|j| -a[i] * b[j]).collect())
            .collect();

        // dE/da[i] = -sum_j(W[i][j] * b[j])
        let da: Vec<f32> = (0..self.dim_a)
            .map(|i| {
                let mut sum = 0.0f32;
                for j in 0..self.dim_b {
                    sum += self.weights[i][j] * b[j];
                }
                -sum
            })
            .collect();

        // dE/db[j] = -sum_i(W[i][j] * a[i])
        let db: Vec<f32> = (0..self.dim_b)
            .map(|j| {
                let mut sum = 0.0f32;
                for i in 0..self.dim_a {
                    sum += self.weights[i][j] * a[i];
                }
                -sum
            })
            .collect();

        (dw, da, db)
    }
}

// ────────────────────────────────────────────────────────────────
// Energy stack (3 pairwise energies)
// ────────────────────────────────────────────────────────────────

/// The three bilinear energy functions connecting adjacent layers.
///
/// ```text
/// Policy ──[E_ps]──> Strategy ──[E_sm]──> Memory ──[E_me]──> Event
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnergyStack {
    /// Policy→Strategy compatibility.
    pub policy_strategy: BilinearEnergy,
    /// Strategy→Memory compatibility.
    pub strategy_memory: BilinearEnergy,
    /// Memory→Event compatibility.
    pub memory_event: BilinearEnergy,
}

impl EnergyStack {
    pub fn new(embed_dim: usize, rng: &mut impl Rng) -> Self {
        Self {
            policy_strategy: BilinearEnergy::new(embed_dim, embed_dim, rng),
            strategy_memory: BilinearEnergy::new(embed_dim, embed_dim, rng),
            memory_event: BilinearEnergy::new(embed_dim, embed_dim, rng),
        }
    }

    /// Compute all three layer energies.
    ///
    /// Returns `(policy_strategy_energy, strategy_memory_energy, memory_event_energy)`.
    pub fn layer_energies(
        &self,
        policy_emb: &[f32],
        strategy_emb: &[f32],
        memory_emb: &[f32],
        event_emb: &[f32],
    ) -> (f32, f32, f32) {
        let e_ps = self.policy_strategy.compute(policy_emb, strategy_emb);
        let e_sm = self.strategy_memory.compute(strategy_emb, memory_emb);
        let e_me = self.memory_event.compute(memory_emb, event_emb);
        (e_ps, e_sm, e_me)
    }

    /// Total free energy = sum of layer energies.
    pub fn total_free_energy(
        &self,
        policy_emb: &[f32],
        strategy_emb: &[f32],
        memory_emb: &[f32],
        event_emb: &[f32],
    ) -> f32 {
        let (e_ps, e_sm, e_me) =
            self.layer_energies(policy_emb, strategy_emb, memory_emb, event_emb);
        e_ps + e_sm + e_me
    }

    /// Compute layer energies for policy→strategy only (no memory/event).
    pub fn policy_strategy_energy_only(&self, policy_emb: &[f32], strategy_emb: &[f32]) -> f32 {
        self.policy_strategy.compute(policy_emb, strategy_emb)
    }
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
    fn test_bilinear_energy_dimensions() {
        let mut rng = make_rng();
        let energy = BilinearEnergy::new(4, 4, &mut rng);
        let a = vec![1.0; 4];
        let b = vec![1.0; 4];
        let _ = energy.compute(&a, &b); // should not panic
    }

    #[test]
    fn test_bilinear_energy_zero_vectors() {
        let mut rng = make_rng();
        let energy = BilinearEnergy::new(4, 4, &mut rng);
        let zero = vec![0.0; 4];
        assert_eq!(energy.compute(&zero, &zero), 0.0);
    }

    #[test]
    fn test_bilinear_energy_symmetry() {
        // Not symmetric in general since E(a,b) = -a^T W b ≠ -b^T W a
        let mut rng = make_rng();
        let energy = BilinearEnergy::new(4, 4, &mut rng);
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let b = vec![4.0, 3.0, 2.0, 1.0];
        let e_ab = energy.compute(&a, &b);
        let e_ba = energy.compute(&b, &a);
        // In general these are different (unless W is symmetric)
        let _ = (e_ab, e_ba);
    }

    #[test]
    #[allow(clippy::needless_range_loop)]
    fn test_bilinear_gradients_finite_difference() {
        let mut rng = make_rng();
        let energy = BilinearEnergy::new(3, 3, &mut rng);
        let a = vec![1.0, -0.5, 0.3];
        let b = vec![0.2, 0.8, -0.4];

        let (dw, da, db) = energy.gradients(&a, &b);
        let e0 = energy.compute(&a, &b);
        let eps = 1e-4;

        // Check dE/da via finite differences
        for i in 0..3 {
            let mut a_plus = a.clone();
            a_plus[i] += eps;
            let e_plus = energy.compute(&a_plus, &b);
            let numerical = (e_plus - e0) / eps;
            assert!(
                (da[i] - numerical).abs() < 1e-2,
                "da[{}]: analytical={}, numerical={}",
                i,
                da[i],
                numerical
            );
        }

        // Check dE/db via finite differences
        for j in 0..3 {
            let mut b_plus = b.clone();
            b_plus[j] += eps;
            let e_plus = energy.compute(&a, &b_plus);
            let numerical = (e_plus - e0) / eps;
            assert!(
                (db[j] - numerical).abs() < 1e-2,
                "db[{}]: analytical={}, numerical={}",
                j,
                db[j],
                numerical
            );
        }

        // Check dE/dW via finite differences (spot check)
        for i in 0..3 {
            for j in 0..3 {
                let mut energy_perturbed = energy.clone();
                energy_perturbed.weights[i][j] += eps;
                let e_plus = energy_perturbed.compute(&a, &b);
                let numerical = (e_plus - e0) / eps;
                assert!(
                    (dw[i][j] - numerical).abs() < 1e-2,
                    "dW[{}][{}]: analytical={}, numerical={}",
                    i,
                    j,
                    dw[i][j],
                    numerical
                );
            }
        }
    }

    #[test]
    fn test_energy_stack_total() {
        let mut rng = make_rng();
        let stack = EnergyStack::new(4, &mut rng);
        let p = vec![1.0; 4];
        let s = vec![1.0; 4];
        let m = vec![1.0; 4];
        let e = vec![1.0; 4];

        let (e_ps, e_sm, e_me) = stack.layer_energies(&p, &s, &m, &e);
        let total = stack.total_free_energy(&p, &s, &m, &e);
        assert!((total - (e_ps + e_sm + e_me)).abs() < 1e-6);
    }
}
