//! STOMP (Stochastic Trajectory Optimization for Motion Planning).
//!
//! Derivative-free trajectory optimizer that generates K noisy trajectory
//! samples, evaluates their costs, and combines them using probability-weighted
//! averaging. Unlike CHOMP, STOMP does NOT require cost function gradients,
//! making it suitable for non-differentiable costs (binary collision checks).
//!
//! # Algorithm
//!
//! 1. Initialize trajectory (straight-line or seed).
//! 2. Each iteration:
//!    a. Generate K noisy trajectories: ξ_k = ξ + ε_k, where ε ~ N(0, R^{-1}).
//!    b. Evaluate cost S_k = cost(ξ_k) for each sample.
//!    c. Compute probability weights: P_k = exp(-h * S_k) / Σ exp(-h * S_j).
//!    d. Update: δξ = Σ_k P_k * R^{-1} * ε_k.
//!    e. ξ += δξ.
//!    f. Check convergence.
//! 3. Return optimized trajectory.
//!
//! # References
//!
//! - Kalakrishnan et al., "STOMP: Stochastic Trajectory Optimization for Motion Planning", ICRA 2011

use kinetic_core::Trajectory;
use rand::Rng;

use crate::cost::TrajectoryCost;

/// STOMP optimizer configuration.
#[derive(Debug, Clone)]
pub struct STOMPConfig {
    /// Number of noisy trajectory samples per iteration (default: 10).
    pub num_samples: usize,
    /// Noise magnitude for trajectory perturbation (default: 0.1 radians).
    pub noise_magnitude: f64,
    /// Temperature parameter h for probability weighting (default: 10.0).
    /// Higher h → more greedy (low-cost samples dominate).
    pub temperature: f64,
    /// Maximum number of iterations (default: 50).
    pub max_iterations: usize,
    /// Convergence tolerance: stop when relative cost change < this (default: 1e-3).
    pub tolerance: f64,
    /// Joint limits: (lower, upper) per DOF. Waypoints clamped after update.
    pub joint_limits: Option<Vec<(f64, f64)>>,
    /// Number of waypoints for initialization (default: 20).
    pub num_waypoints: usize,
}

impl Default for STOMPConfig {
    fn default() -> Self {
        Self {
            num_samples: 10,
            noise_magnitude: 0.1,
            temperature: 10.0,
            max_iterations: 50,
            tolerance: 1e-3,
            joint_limits: None,
            num_waypoints: 20,
        }
    }
}

/// Result of STOMP optimization.
#[derive(Debug, Clone)]
pub struct STOMPResult {
    /// Optimized trajectory.
    pub trajectory: Trajectory,
    /// Final total cost.
    pub cost: f64,
    /// Cost at each iteration.
    pub cost_history: Vec<f64>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the optimizer converged.
    pub converged: bool,
}

/// STOMP trajectory optimizer.
///
/// Stochastic, derivative-free trajectory optimization. Generates noisy
/// trajectory samples, evaluates costs, and updates via probability-weighted
/// averaging. Works with any cost function — does not need gradients.
pub struct STOMP {
    cost: Box<dyn TrajectoryCost>,
    config: STOMPConfig,
    dof: usize,
}

impl STOMP {
    /// Create a STOMP optimizer.
    pub fn new(cost: Box<dyn TrajectoryCost>, dof: usize, config: STOMPConfig) -> Self {
        Self { cost, config, dof }
    }

    /// Create with default configuration.
    pub fn with_defaults(cost: Box<dyn TrajectoryCost>, dof: usize) -> Self {
        Self::new(cost, dof, STOMPConfig::default())
    }

    /// Plan from start to goal: initialize + optimize.
    pub fn plan(&self, start: &[f64], goal: &[f64]) -> STOMPResult {
        let traj = self.linear_init(start, goal);
        self.optimize(&traj)
    }

    /// Optimize an existing trajectory.
    pub fn optimize(&self, trajectory: &Trajectory) -> STOMPResult {
        let n = trajectory.len();
        assert!(n >= 3, "STOMP requires at least 3 waypoints");

        let interior = n - 2;
        let flat_len = interior * self.dof;

        // Extract interior waypoints as flat array
        let mut xi = self.extract_interior(trajectory);

        // Fixed start/goal
        let start = self.extract_waypoint(trajectory, 0);
        let goal = self.extract_waypoint(trajectory, n - 1);

        // Pre-compute R^{-1} (smoothness-biased covariance)
        // Use the same A^{-1} as CHOMP for smooth noise generation
        let r_inv = self.compute_r_inverse(interior);

        let mut cost_history = Vec::with_capacity(self.config.max_iterations);
        let mut prev_cost = self.eval_cost_from_flat(&start, &xi, &goal, n);
        cost_history.push(prev_cost);

        let mut rng = rand::thread_rng();
        let mut iterations = 0;
        let mut converged = false;

        for _iter in 0..self.config.max_iterations {
            iterations += 1;

            // Step 1: Generate K noisy samples
            let mut noises = Vec::with_capacity(self.config.num_samples);
            let mut sample_costs = Vec::with_capacity(self.config.num_samples);

            for _ in 0..self.config.num_samples {
                // Generate smooth noise: ε = R^{-1/2} * z, where z ~ N(0, I)
                // Approximation: generate white noise then smooth with R^{-1}
                let raw_noise: Vec<f64> = (0..flat_len)
                    .map(|_| rng.gen_range(-1.0..1.0) * self.config.noise_magnitude)
                    .collect();

                // Smooth the noise via R^{-1} multiplication (per-DOF)
                let smooth_noise = self.smooth_noise(&raw_noise, &r_inv, interior);

                // Create noisy trajectory: ξ_k = ξ + ε_k
                let noisy_xi: Vec<f64> = xi
                    .iter()
                    .zip(smooth_noise.iter())
                    .map(|(x, e)| x + e)
                    .collect();

                // Evaluate cost
                let c = self.eval_cost_from_flat(&start, &noisy_xi, &goal, n);
                sample_costs.push(c);
                noises.push(smooth_noise);
            }

            // Step 2: Compute probability weights
            let weights = self.compute_weights(&sample_costs);

            // Step 3: Weighted update: δξ = Σ_k P_k * ε_k
            let mut delta = vec![0.0; flat_len];
            for (k, noise) in noises.iter().enumerate() {
                let w = weights[k];
                for i in 0..flat_len {
                    delta[i] += w * noise[i];
                }
            }

            // Step 4: Apply update
            for i in 0..flat_len {
                xi[i] += delta[i];
            }

            // Clamp to joint limits
            if let Some(ref limits) = self.config.joint_limits {
                for wi in 0..interior {
                    for j in 0..self.dof {
                        let (lo, hi) = limits[j];
                        xi[wi * self.dof + j] = xi[wi * self.dof + j].clamp(lo, hi);
                    }
                }
            }

            // Step 5: Evaluate new cost and check convergence
            let new_cost = self.eval_cost_from_flat(&start, &xi, &goal, n);
            cost_history.push(new_cost);

            let relative_change = if prev_cost.abs() > 1e-12 {
                (prev_cost - new_cost).abs() / prev_cost.abs()
            } else {
                (prev_cost - new_cost).abs()
            };

            if relative_change < self.config.tolerance {
                converged = true;
                prev_cost = new_cost;
                break;
            }

            prev_cost = new_cost;
        }

        let final_traj = self.build_trajectory(&start, &xi, &goal, n);

        STOMPResult {
            trajectory: final_traj,
            cost: prev_cost,
            cost_history,
            iterations,
            converged,
        }
    }

    /// Compute probability weights from sample costs using softmin.
    ///
    /// P_k = exp(-h * S_k) / Σ_j exp(-h * S_j)
    fn compute_weights(&self, costs: &[f64]) -> Vec<f64> {
        let h = self.config.temperature;

        // Find min cost for numerical stability (log-sum-exp trick)
        let min_cost = costs.iter().cloned().fold(f64::INFINITY, f64::min);

        let exp_values: Vec<f64> = costs.iter().map(|c| (-h * (c - min_cost)).exp()).collect();
        let sum: f64 = exp_values.iter().sum();

        if sum > 0.0 {
            exp_values.iter().map(|e| e / sum).collect()
        } else {
            // Uniform fallback
            let w = 1.0 / costs.len() as f64;
            vec![w; costs.len()]
        }
    }

    /// Smooth noise by multiplying with R^{-1} per-DOF column.
    fn smooth_noise(
        &self,
        raw_noise: &[f64],
        r_inv: &[f64],
        interior: usize,
    ) -> Vec<f64> {
        let mut smooth = vec![0.0; raw_noise.len()];

        for j in 0..self.dof {
            for wi in 0..interior {
                let mut val = 0.0;
                for wk in 0..interior {
                    val += r_inv[wi * interior + wk] * raw_noise[wk * self.dof + j];
                }
                smooth[wi * self.dof + j] = val;
            }
        }
        smooth
    }

    /// Compute R^{-1} — smoothness-biased covariance for noise generation.
    ///
    /// Same structure as CHOMP's A^{-1}: inverse of the second-order
    /// finite differencing matrix. This biases noise toward smooth trajectories.
    fn compute_r_inverse(&self, interior: usize) -> Vec<f64> {
        if interior == 0 {
            return vec![];
        }

        let mut r = vec![0.0; interior * interior];
        for i in 0..interior {
            r[i * interior + i] = 6.0;
            if i > 0 { r[i * interior + (i - 1)] = -4.0; }
            if i + 1 < interior { r[i * interior + (i + 1)] = -4.0; }
            if i >= 2 { r[i * interior + (i - 2)] = 1.0; }
            if i + 2 < interior { r[i * interior + (i + 2)] = 1.0; }
        }

        crate::chomp::invert_matrix(&r, interior)
    }

    /// Linear interpolation initialization.
    fn linear_init(&self, start: &[f64], goal: &[f64]) -> Trajectory {
        let n = self.config.num_waypoints;
        let mut traj = Trajectory::with_dof(self.dof);
        for i in 0..n {
            let t = i as f64 / (n - 1) as f64;
            let wp: Vec<f64> = (0..self.dof)
                .map(|j| start[j] + t * (goal[j] - start[j]))
                .collect();
            traj.push_waypoint(&wp);
        }
        traj
    }

    fn extract_interior(&self, trajectory: &Trajectory) -> Vec<f64> {
        let n = trajectory.len();
        let interior = n - 2;
        let mut flat = vec![0.0; interior * self.dof];
        for wi in 0..interior {
            let wp = trajectory.waypoint(wi + 1);
            for j in 0..self.dof {
                flat[wi * self.dof + j] = wp.positions[j];
            }
        }
        flat
    }

    fn extract_waypoint(&self, trajectory: &Trajectory, idx: usize) -> Vec<f64> {
        let wp = trajectory.waypoint(idx);
        (0..self.dof).map(|j| wp.positions[j]).collect()
    }

    fn eval_cost_from_flat(
        &self,
        start: &[f64],
        interior: &[f64],
        goal: &[f64],
        n: usize,
    ) -> f64 {
        let traj = self.build_trajectory(start, interior, goal, n);
        self.cost.evaluate(&traj)
    }

    fn build_trajectory(
        &self,
        start: &[f64],
        interior: &[f64],
        goal: &[f64],
        n: usize,
    ) -> Trajectory {
        let int_count = n - 2;
        let mut traj = Trajectory::with_dof(self.dof);
        traj.push_waypoint(start);
        for wi in 0..int_count {
            let wp: Vec<f64> = (0..self.dof)
                .map(|j| interior[wi * self.dof + j])
                .collect();
            traj.push_waypoint(&wp);
        }
        traj.push_waypoint(goal);
        traj
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::{CompositeCost, ConstantCost, SmoothnessCost, VelocityCost};

    fn make_trajectory(dof: usize, waypoints: &[Vec<f64>]) -> Trajectory {
        let mut traj = Trajectory::with_dof(dof);
        for wp in waypoints {
            traj.push_waypoint(wp);
        }
        traj
    }

    #[test]
    fn stomp_reduces_smoothness_cost() {
        let cost = Box::new(SmoothnessCost::new(1));
        let stomp = STOMP::new(
            cost,
            1,
            STOMPConfig {
                num_samples: 20,
                noise_magnitude: 0.3,
                temperature: 10.0,
                max_iterations: 30,
                num_waypoints: 10,
                ..Default::default()
            },
        );

        // Zigzag trajectory
        let traj = make_trajectory(1, &[
            vec![0.0], vec![2.0], vec![-1.0], vec![1.5], vec![1.0],
        ]);

        let initial_cost = SmoothnessCost::new(1).evaluate(&traj);
        let result = stomp.optimize(&traj);

        assert!(
            result.cost <= initial_cost,
            "STOMP should reduce cost: {} -> {}",
            initial_cost,
            result.cost
        );
    }

    #[test]
    fn stomp_plan_from_start_goal() {
        let cost = Box::new(SmoothnessCost::new(2));
        let stomp = STOMP::new(
            cost,
            2,
            STOMPConfig {
                num_samples: 10,
                max_iterations: 10,
                num_waypoints: 8,
                ..Default::default()
            },
        );

        let result = stomp.plan(&[0.0, 0.0], &[1.0, 1.0]);
        assert_eq!(result.trajectory.len(), 8);
        // Start and goal preserved
        assert!((result.trajectory.position(0, 0) - 0.0).abs() < 1e-10);
        assert!((result.trajectory.position(0, 7) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn stomp_straight_line_low_cost() {
        // Straight line should have near-zero smoothness cost and STOMP shouldn't ruin it
        let cost = Box::new(SmoothnessCost::new(1));
        let stomp = STOMP::new(
            cost,
            1,
            STOMPConfig {
                num_samples: 5,
                noise_magnitude: 0.01, // tiny noise
                max_iterations: 5,
                num_waypoints: 10,
                ..Default::default()
            },
        );

        let result = stomp.plan(&[0.0], &[1.0]);
        // Should still be near-straight
        assert!(
            result.cost < 0.1,
            "Straight-line + tiny noise should have low cost: {}",
            result.cost
        );
    }

    #[test]
    fn stomp_probability_weights_sum_to_one() {
        let cost = Box::new(ConstantCost::new(0.0, 1));
        let stomp = STOMP::with_defaults(cost, 1);

        let costs = vec![1.0, 2.0, 0.5, 3.0, 1.5];
        let weights = stomp.compute_weights(&costs);

        let sum: f64 = weights.iter().sum();
        assert!(
            (sum - 1.0).abs() < 1e-10,
            "Weights should sum to 1: {}",
            sum
        );

        // Lowest cost (0.5) should have highest weight
        let min_idx = 2; // cost = 0.5
        let max_weight_idx = weights
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .unwrap()
            .0;
        assert_eq!(
            max_weight_idx, min_idx,
            "Lowest cost sample should have highest weight"
        );
    }

    #[test]
    fn stomp_probability_weights_equal_for_equal_costs() {
        let cost = Box::new(ConstantCost::new(0.0, 1));
        let stomp = STOMP::with_defaults(cost, 1);

        let costs = vec![1.0, 1.0, 1.0, 1.0];
        let weights = stomp.compute_weights(&costs);

        for w in &weights {
            assert!(
                (w - 0.25).abs() < 1e-10,
                "Equal costs should give equal weights: {}",
                w
            );
        }
    }

    #[test]
    fn stomp_respects_joint_limits() {
        let cost = Box::new(SmoothnessCost::new(1));
        let stomp = STOMP::new(
            cost,
            1,
            STOMPConfig {
                num_samples: 10,
                noise_magnitude: 5.0, // huge noise
                max_iterations: 10,
                joint_limits: Some(vec![(-1.0, 1.0)]),
                num_waypoints: 8,
                ..Default::default()
            },
        );

        let result = stomp.plan(&[0.0], &[0.5]);

        for i in 1..result.trajectory.len() - 1 {
            let val = result.trajectory.position(0, i);
            assert!(
                val >= -1.0 - 1e-10 && val <= 1.0 + 1e-10,
                "Waypoint {} should be within limits: {}",
                i,
                val
            );
        }
    }

    #[test]
    fn stomp_composite_cost() {
        let mut composite = CompositeCost::new(2);
        composite.add("smooth", Box::new(SmoothnessCost::new(2)), 1.0);
        composite.add("vel", Box::new(VelocityCost::new(2)), 0.5);

        let stomp = STOMP::new(
            Box::new(composite),
            2,
            STOMPConfig {
                num_samples: 10,
                max_iterations: 10,
                num_waypoints: 8,
                ..Default::default()
            },
        );

        let result = stomp.plan(&[0.0, 0.0], &[1.0, 1.0]);
        assert!(result.iterations > 0);
        assert!(!result.cost_history.is_empty());
    }

    #[test]
    fn stomp_cost_history_recorded() {
        let cost = Box::new(SmoothnessCost::new(1));
        let stomp = STOMP::new(
            cost,
            1,
            STOMPConfig {
                num_samples: 5,
                max_iterations: 10,
                num_waypoints: 6,
                ..Default::default()
            },
        );

        let traj = make_trajectory(1, &[
            vec![0.0], vec![2.0], vec![-1.0], vec![1.0],
        ]);
        let result = stomp.optimize(&traj);

        assert_eq!(
            result.cost_history.len(),
            result.iterations + 1,
            "Cost history should have iterations+1 entries"
        );
    }

    #[test]
    fn stomp_smooth_noise_preserves_shape() {
        // Smooth noise should have similar magnitude but smoother profile
        let cost = Box::new(ConstantCost::new(0.0, 1));
        let stomp = STOMP::new(cost, 1, STOMPConfig::default());

        let interior = 10;
        let r_inv = stomp.compute_r_inverse(interior);

        // White noise
        let raw: Vec<f64> = (0..interior).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }).collect();

        let smooth = stomp.smooth_noise(&raw, &r_inv, interior);

        // Smooth noise should have smaller second differences than raw
        let raw_roughness: f64 = (1..interior - 1)
            .map(|i| (raw[i - 1] - 2.0 * raw[i] + raw[i + 1]).powi(2))
            .sum();
        let smooth_roughness: f64 = (1..interior - 1)
            .map(|i| (smooth[i - 1] - 2.0 * smooth[i] + smooth[i + 1]).powi(2))
            .sum();

        assert!(
            smooth_roughness < raw_roughness,
            "Smooth noise ({}) should be less rough than raw ({})",
            smooth_roughness,
            raw_roughness
        );
    }
}
