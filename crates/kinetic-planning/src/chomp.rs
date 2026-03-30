//! CHOMP (Covariant Hamiltonian Optimization for Motion Planning).
//!
//! Gradient-based trajectory optimizer that minimizes a composite cost
//! (obstacle + smoothness) using covariant gradient descent. The covariant
//! update uses the inverse of the smoothness metric A to precondition
//! gradients, producing smooth updates that respect the trajectory manifold.
//!
//! # Algorithm
//!
//! 1. Initialize trajectory as straight-line interpolation.
//! 2. Pre-compute A (finite differencing matrix) and A^{-1}.
//! 3. Each iteration:
//!    a. Evaluate total cost c = obstacle_cost + η_smooth * smoothness_cost.
//!    b. Compute gradient g = ∂c/∂ξ (per interior waypoint).
//!    c. Covariant update: ξ -= η * A^{-1} * g.
//!    d. Check convergence: |Δc/c| < tolerance.
//! 4. Return optimized trajectory.
//!
//! # References
//!
//! - Zucker et al., "CHOMP: Covariant Hamiltonian Optimization for Motion Planning", IJRR 2013
//! - Ratliff et al., "CHOMP: Gradient Optimization Techniques for Efficient Motion Planning", ICRA 2009

use kinetic_core::Trajectory;
use rand::Rng;

use crate::cost::TrajectoryCost;

/// How to initialize the trajectory before optimization.
#[derive(Debug, Clone)]
pub enum InitStrategy {
    /// Linear interpolation between start and goal (default).
    /// Produces a straight-line trajectory with `num_waypoints` total points.
    Linear { num_waypoints: usize },
    /// Use an existing trajectory as seed (e.g., from RRT-Connect).
    /// CHOMP optimizes the seed trajectory directly.
    Seed,
    /// Linear interpolation + random perturbation.
    /// `magnitude` controls the noise amplitude (radians).
    Perturbed {
        num_waypoints: usize,
        magnitude: f64,
    },
}

impl Default for InitStrategy {
    fn default() -> Self {
        Self::Linear { num_waypoints: 20 }
    }
}

/// CHOMP optimizer configuration.
#[derive(Debug, Clone)]
pub struct CHOMPConfig {
    /// Learning rate for covariant gradient descent (default: 0.05).
    pub learning_rate: f64,
    /// Maximum number of iterations (default: 100).
    pub max_iterations: usize,
    /// Convergence tolerance: stop when relative cost change < this (default: 1e-4).
    pub tolerance: f64,
    /// Minimum cost improvement to continue (absolute). Prevents oscillation (default: 1e-8).
    pub min_improvement: f64,
    /// Joint limits: (lower, upper) per DOF. Waypoints clamped after each update.
    pub joint_limits: Option<Vec<(f64, f64)>>,
    /// Initialization strategy (default: Linear with 20 waypoints).
    pub init_strategy: InitStrategy,
}

impl Default for CHOMPConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.05,
            max_iterations: 100,
            tolerance: 1e-4,
            min_improvement: 1e-8,
            joint_limits: None,
            init_strategy: InitStrategy::default(),
        }
    }
}

/// Result of CHOMP optimization.
#[derive(Debug, Clone)]
pub struct CHOMPResult {
    /// Optimized trajectory.
    pub trajectory: Trajectory,
    /// Final total cost.
    pub cost: f64,
    /// Cost at each iteration (for convergence analysis).
    pub cost_history: Vec<f64>,
    /// Number of iterations performed.
    pub iterations: usize,
    /// Whether the optimizer converged within tolerance.
    pub converged: bool,
}

/// CHOMP trajectory optimizer.
///
/// Optimizes a trajectory by covariant gradient descent using a composite
/// cost function. The optimizer pre-computes the inverse of the smoothness
/// metric matrix A for efficient covariant updates.
pub struct CHOMP {
    cost: Box<dyn TrajectoryCost>,
    config: CHOMPConfig,
    dof: usize,
}

impl CHOMP {
    /// Create a CHOMP optimizer with a cost function.
    ///
    /// The cost function should include all desired terms (obstacle, smoothness,
    /// constraints) combined via `CompositeCost`.
    pub fn new(cost: Box<dyn TrajectoryCost>, dof: usize, config: CHOMPConfig) -> Self {
        Self { cost, config, dof }
    }

    /// Create with default configuration.
    pub fn with_defaults(cost: Box<dyn TrajectoryCost>, dof: usize) -> Self {
        Self::new(cost, dof, CHOMPConfig::default())
    }

    /// Initialize a trajectory from start/goal using the configured strategy.
    ///
    /// - `Linear`: straight-line interpolation with N waypoints.
    /// - `Seed`: returns the provided seed trajectory unchanged.
    /// - `Perturbed`: linear interpolation + random noise on interior waypoints.
    pub fn init_trajectory(
        &self,
        start: &[f64],
        goal: &[f64],
        seed: Option<&Trajectory>,
    ) -> Trajectory {
        match &self.config.init_strategy {
            InitStrategy::Seed => {
                seed.expect("InitStrategy::Seed requires a seed trajectory")
                    .clone()
            }
            InitStrategy::Linear { num_waypoints } => {
                Self::linear_init(self.dof, start, goal, *num_waypoints)
            }
            InitStrategy::Perturbed {
                num_waypoints,
                magnitude,
            } => {
                let mut traj = Self::linear_init(self.dof, start, goal, *num_waypoints);
                Self::perturb_interior(&mut traj, self.dof, *magnitude, &self.config.joint_limits);
                traj
            }
        }
    }

    /// Plan from start to goal: initialize + optimize.
    ///
    /// Convenience method that combines initialization and optimization.
    pub fn plan(
        &self,
        start: &[f64],
        goal: &[f64],
        seed: Option<&Trajectory>,
    ) -> CHOMPResult {
        let traj = self.init_trajectory(start, goal, seed);
        self.optimize(&traj)
    }

    /// Linear interpolation between start and goal.
    fn linear_init(dof: usize, start: &[f64], goal: &[f64], num_waypoints: usize) -> Trajectory {
        assert!(num_waypoints >= 3, "Need at least 3 waypoints");
        assert_eq!(start.len(), dof);
        assert_eq!(goal.len(), dof);

        let mut traj = Trajectory::with_dof(dof);
        for i in 0..num_waypoints {
            let t = i as f64 / (num_waypoints - 1) as f64;
            let wp: Vec<f64> = (0..dof)
                .map(|j| start[j] + t * (goal[j] - start[j]))
                .collect();
            traj.push_waypoint(&wp);
        }
        traj
    }

    /// Add random perturbation to interior waypoints.
    fn perturb_interior(
        traj: &mut Trajectory,
        dof: usize,
        magnitude: f64,
        limits: &Option<Vec<(f64, f64)>>,
    ) {
        let n = traj.len();
        if n <= 2 {
            return;
        }

        let mut rng = rand::thread_rng();

        // Rebuild the trajectory with perturbed interior
        let start = traj.waypoint(0).positions.to_vec();
        let goal = traj.waypoint(n - 1).positions.to_vec();
        let mut waypoints: Vec<Vec<f64>> = Vec::with_capacity(n);
        waypoints.push(start);

        for i in 1..n - 1 {
            let wp = traj.waypoint(i);
            let mut perturbed: Vec<f64> = (0..dof)
                .map(|j| wp.positions[j] + rng.gen_range(-magnitude..magnitude))
                .collect();

            // Clamp to limits
            if let Some(lims) = limits {
                for j in 0..dof {
                    perturbed[j] = perturbed[j].clamp(lims[j].0, lims[j].1);
                }
            }
            waypoints.push(perturbed);
        }
        waypoints.push(goal);

        *traj = Trajectory::with_dof(dof);
        for wp in &waypoints {
            traj.push_waypoint(wp);
        }
    }

    /// Optimize a trajectory in-place.
    ///
    /// The trajectory must have at least 3 waypoints (start + interior + goal).
    /// Start and goal are fixed; only interior waypoints are optimized.
    pub fn optimize(&self, trajectory: &Trajectory) -> CHOMPResult {
        let n = trajectory.len();
        assert!(n >= 3, "CHOMP requires at least 3 waypoints");

        let interior = n - 2;

        // Pre-compute A^{-1} (smoothness metric inverse)
        let a_inv = self.compute_a_inverse(interior);

        // Extract mutable waypoint data
        let mut waypoints = self.extract_waypoints(trajectory);

        let mut cost_history = Vec::with_capacity(self.config.max_iterations);
        let mut prev_cost = self.eval_cost(trajectory);
        cost_history.push(prev_cost);

        let mut iterations = 0;
        let mut converged = false;

        for _iter in 0..self.config.max_iterations {
            iterations += 1;

            // Build trajectory from current waypoints
            let current_traj = self.build_trajectory(trajectory, &waypoints);

            // Compute gradient
            let grad = self.cost.gradient(&current_traj);
            if grad.is_empty() {
                converged = true;
                break;
            }

            // Covariant update: ξ -= η * A^{-1} * g
            // A^{-1} is (interior × interior) applied per-DOF column
            for j in 0..self.dof {
                for wi in 0..interior {
                    let mut update = 0.0;
                    for wk in 0..interior {
                        update += a_inv[wi * interior + wk] * grad[wk * self.dof + j];
                    }
                    waypoints[wi * self.dof + j] -= self.config.learning_rate * update;
                }
            }

            // Clamp to joint limits
            if let Some(ref limits) = self.config.joint_limits {
                for wi in 0..interior {
                    for j in 0..self.dof {
                        let (lo, hi) = limits[j];
                        waypoints[wi * self.dof + j] =
                            waypoints[wi * self.dof + j].clamp(lo, hi);
                    }
                }
            }

            // Evaluate new cost
            let new_traj = self.build_trajectory(trajectory, &waypoints);
            let new_cost = self.eval_cost(&new_traj);
            cost_history.push(new_cost);

            // Convergence check
            let improvement = prev_cost - new_cost;
            let relative_change = if prev_cost.abs() > 1e-12 {
                improvement.abs() / prev_cost.abs()
            } else {
                improvement.abs()
            };

            if relative_change < self.config.tolerance && improvement >= 0.0 {
                converged = true;
                prev_cost = new_cost;
                break;
            }

            if improvement.abs() < self.config.min_improvement {
                converged = true;
                prev_cost = new_cost;
                break;
            }

            prev_cost = new_cost;
        }

        let final_traj = self.build_trajectory(trajectory, &waypoints);

        CHOMPResult {
            trajectory: final_traj,
            cost: prev_cost,
            cost_history,
            iterations,
            converged,
        }
    }

    /// Compute the inverse of the smoothness metric matrix A.
    ///
    /// A is the (interior × interior) finite-differencing matrix that encodes
    /// the smoothness prior. For second-order smoothness (acceleration):
    ///
    ///   A[i][i] = 6, A[i][i±1] = -4, A[i][i±2] = 1
    ///
    /// This is K^T * K where K is the second-order finite differencing operator.
    /// We pre-compute A^{-1} once and reuse it for all iterations.
    fn compute_a_inverse(&self, interior: usize) -> Vec<f64> {
        if interior == 0 {
            return vec![];
        }

        // Build A = K^T * K where K is the (interior+2 × interior) second-difference matrix
        // For simplicity and robustness, use the tridiagonal approximation:
        // A[i][i] = 2 + 2*alpha, A[i][i±1] = -alpha
        // where alpha=1 gives the standard discrete Laplacian squared
        let mut a = vec![0.0; interior * interior];

        for i in 0..interior {
            // Diagonal: from the three terms that touch waypoint i
            a[i * interior + i] = 6.0;

            // Off-diagonal ±1
            if i > 0 {
                a[i * interior + (i - 1)] = -4.0;
            }
            if i + 1 < interior {
                a[i * interior + (i + 1)] = -4.0;
            }

            // Off-diagonal ±2
            if i >= 2 {
                a[i * interior + (i - 2)] = 1.0;
            }
            if i + 2 < interior {
                a[i * interior + (i + 2)] = 1.0;
            }
        }

        // Invert A using LU decomposition (small matrix, done once)
        invert_matrix(&a, interior)
    }

    /// Extract interior waypoint positions as flat array.
    fn extract_waypoints(&self, trajectory: &Trajectory) -> Vec<f64> {
        let n = trajectory.len();
        let interior = n - 2;
        let mut waypoints = vec![0.0; interior * self.dof];

        for wi in 0..interior {
            let wp = trajectory.waypoint(wi + 1);
            for j in 0..self.dof {
                waypoints[wi * self.dof + j] = wp.positions[j];
            }
        }
        waypoints
    }

    /// Build a full trajectory from start/goal of original + interior waypoints.
    fn build_trajectory(&self, original: &Trajectory, waypoints: &[f64]) -> Trajectory {
        let n = original.len();
        let mut traj = Trajectory::with_dof(self.dof);

        // Start waypoint (fixed)
        let start = original.waypoint(0);
        traj.push_waypoint(&start.positions);

        // Interior waypoints (optimized)
        let interior = n - 2;
        for wi in 0..interior {
            let wp: Vec<f64> = (0..self.dof)
                .map(|j| waypoints[wi * self.dof + j])
                .collect();
            traj.push_waypoint(&wp);
        }

        // Goal waypoint (fixed)
        let goal = original.waypoint(n - 1);
        traj.push_waypoint(&goal.positions);

        traj
    }

    /// Evaluate cost on a trajectory.
    fn eval_cost(&self, trajectory: &Trajectory) -> f64 {
        self.cost.evaluate(trajectory)
    }
}

/// Invert a square matrix using Gauss-Jordan elimination.
///
/// Input: flat row-major matrix of size n×n.
/// Returns: flat row-major inverse of size n×n.
pub(crate) fn invert_matrix(matrix: &[f64], n: usize) -> Vec<f64> {
    assert_eq!(matrix.len(), n * n);

    // Augmented matrix [A | I]
    let mut aug = vec![0.0; n * 2 * n];
    for i in 0..n {
        for j in 0..n {
            aug[i * 2 * n + j] = matrix[i * n + j];
        }
        aug[i * 2 * n + n + i] = 1.0;
    }

    // Forward elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_val = aug[col * 2 * n + col].abs();
        let mut max_row = col;
        for row in (col + 1)..n {
            let val = aug[row * 2 * n + col].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        // Swap rows
        if max_row != col {
            for j in 0..2 * n {
                aug.swap(col * 2 * n + j, max_row * 2 * n + j);
            }
        }

        let pivot = aug[col * 2 * n + col];
        if pivot.abs() < 1e-15 {
            // Singular matrix — return identity as fallback
            let mut identity = vec![0.0; n * n];
            for i in 0..n {
                identity[i * n + i] = 1.0;
            }
            return identity;
        }

        // Scale pivot row
        for j in 0..2 * n {
            aug[col * 2 * n + j] /= pivot;
        }

        // Eliminate column
        for row in 0..n {
            if row == col {
                continue;
            }
            let factor = aug[row * 2 * n + col];
            for j in 0..2 * n {
                aug[row * 2 * n + j] -= factor * aug[col * 2 * n + j];
            }
        }
    }

    // Extract inverse from right half
    let mut inv = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            inv[i * n + j] = aug[i * 2 * n + n + j];
        }
    }
    inv
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cost::{CompositeCost, SmoothnessCost, VelocityCost};

    fn make_trajectory(dof: usize, waypoints: &[Vec<f64>]) -> Trajectory {
        let mut traj = Trajectory::with_dof(dof);
        for wp in waypoints {
            traj.push_waypoint(wp);
        }
        traj
    }

    #[test]
    fn a_inverse_is_identity_product() {
        // Verify A * A^{-1} ≈ I
        let chomp = CHOMP::with_defaults(
            Box::new(SmoothnessCost::new(1)),
            1,
        );
        let n = 5;
        let a_inv = chomp.compute_a_inverse(n);

        // Rebuild A
        let mut a = vec![0.0; n * n];
        for i in 0..n {
            a[i * n + i] = 6.0;
            if i > 0 { a[i * n + (i - 1)] = -4.0; }
            if i + 1 < n { a[i * n + (i + 1)] = -4.0; }
            if i >= 2 { a[i * n + (i - 2)] = 1.0; }
            if i + 2 < n { a[i * n + (i + 2)] = 1.0; }
        }

        // Compute A * A^{-1}
        for i in 0..n {
            for j in 0..n {
                let mut sum = 0.0;
                for k in 0..n {
                    sum += a[i * n + k] * a_inv[k * n + j];
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (sum - expected).abs() < 1e-8,
                    "A*A^-1 [{},{}] = {}, expected {}",
                    i, j, sum, expected
                );
            }
        }
    }

    #[test]
    fn chomp_straight_line_stays_straight() {
        // A straight-line trajectory with only smoothness cost should stay straight
        // (it's already optimal for smoothness)
        let cost = Box::new(SmoothnessCost::new(2));
        let chomp = CHOMP::with_defaults(cost, 2);

        let traj = make_trajectory(2, &[
            vec![0.0, 0.0],
            vec![1.0, 1.0],
            vec![2.0, 2.0],
            vec![3.0, 3.0],
            vec![4.0, 4.0],
        ]);

        let result = chomp.optimize(&traj);
        assert!(result.converged, "Should converge quickly on straight line");
        assert!(result.cost < 1e-8, "Straight line smoothness cost should be ~0: {}", result.cost);
    }

    #[test]
    fn chomp_reduces_cost() {
        // Zigzag trajectory + smoothness cost → CHOMP should reduce cost
        let cost = Box::new(SmoothnessCost::new(1));
        let chomp = CHOMP::new(
            cost,
            1,
            CHOMPConfig {
                learning_rate: 0.01,
                max_iterations: 50,
                ..Default::default()
            },
        );

        let traj = make_trajectory(1, &[
            vec![0.0],
            vec![2.0],   // bump up
            vec![-1.0],  // bump down
            vec![1.5],   // bump up
            vec![1.0],
        ]);

        let initial_cost = SmoothnessCost::new(1).evaluate(&traj);
        let result = chomp.optimize(&traj);

        assert!(
            result.cost < initial_cost,
            "CHOMP should reduce cost: {} -> {}",
            initial_cost,
            result.cost
        );
    }

    #[test]
    fn chomp_composite_cost() {
        // Smoothness + velocity cost
        let mut composite = CompositeCost::new(2);
        composite.add("smooth", Box::new(SmoothnessCost::new(2)), 1.0);
        composite.add("vel", Box::new(VelocityCost::new(2)), 0.5);

        let chomp = CHOMP::new(
            Box::new(composite),
            2,
            CHOMPConfig {
                learning_rate: 0.01,
                max_iterations: 50,
                ..Default::default()
            },
        );

        let traj = make_trajectory(2, &[
            vec![0.0, 0.0],
            vec![3.0, -2.0],
            vec![-1.0, 3.0],
            vec![2.0, 2.0],
        ]);

        let result = chomp.optimize(&traj);
        assert!(result.iterations > 0);
        assert!(!result.cost_history.is_empty());
    }

    #[test]
    fn chomp_respects_joint_limits() {
        let cost = Box::new(SmoothnessCost::new(1));
        let chomp = CHOMP::new(
            cost,
            1,
            CHOMPConfig {
                learning_rate: 0.1,
                max_iterations: 20,
                joint_limits: Some(vec![(-1.0, 1.0)]),
                ..Default::default()
            },
        );

        let traj = make_trajectory(1, &[
            vec![0.0],
            vec![5.0],  // way above upper limit
            vec![0.0],
        ]);

        let result = chomp.optimize(&traj);

        // All interior waypoints should be within limits
        for i in 1..result.trajectory.len() - 1 {
            let wp = result.trajectory.waypoint(i);
            assert!(
                wp.positions[0] >= -1.0 - 1e-10 && wp.positions[0] <= 1.0 + 1e-10,
                "Waypoint {} should be within limits: {}",
                i,
                wp.positions[0]
            );
        }
    }

    #[test]
    fn chomp_cost_history_monotonic_or_converges() {
        let cost = Box::new(SmoothnessCost::new(1));
        let chomp = CHOMP::new(
            cost,
            1,
            CHOMPConfig {
                learning_rate: 0.005, // conservative
                max_iterations: 100,
                ..Default::default()
            },
        );

        let traj = make_trajectory(1, &[
            vec![0.0],
            vec![2.0],
            vec![-1.0],
            vec![3.0],
            vec![0.0],
            vec![1.0],
        ]);

        let result = chomp.optimize(&traj);

        // With conservative learning rate, cost should generally decrease
        if result.cost_history.len() >= 3 {
            let last = result.cost_history.last().unwrap();
            let first = result.cost_history.first().unwrap();
            assert!(
                last <= first,
                "Final cost ({}) should be <= initial ({})",
                last,
                first
            );
        }
    }

    // ─── Initialization strategy tests ───

    #[test]
    fn init_linear_correct_endpoints() {
        let cost = Box::new(SmoothnessCost::new(2));
        let chomp = CHOMP::new(
            cost,
            2,
            CHOMPConfig {
                init_strategy: InitStrategy::Linear { num_waypoints: 10 },
                ..Default::default()
            },
        );

        let start = [0.0, 0.0];
        let goal = [1.0, 2.0];
        let traj = chomp.init_trajectory(&start, &goal, None);

        assert_eq!(traj.len(), 10);
        assert!((traj.position(0, 0) - 0.0).abs() < 1e-10);
        assert!((traj.position(1, 0) - 0.0).abs() < 1e-10);
        assert!((traj.position(0, 9) - 1.0).abs() < 1e-10);
        assert!((traj.position(1, 9) - 2.0).abs() < 1e-10);
    }

    #[test]
    fn init_linear_is_straight() {
        let cost = Box::new(SmoothnessCost::new(1));
        let chomp = CHOMP::new(
            cost,
            1,
            CHOMPConfig {
                init_strategy: InitStrategy::Linear { num_waypoints: 5 },
                ..Default::default()
            },
        );

        let traj = chomp.init_trajectory(&[0.0], &[4.0], None);
        // Should be [0, 1, 2, 3, 4]
        for i in 0..5 {
            assert!(
                (traj.position(0, i) - i as f64).abs() < 1e-10,
                "wp {}: {} != {}",
                i,
                traj.position(0, i),
                i
            );
        }
    }

    #[test]
    fn init_seed_uses_provided_trajectory() {
        let cost = Box::new(SmoothnessCost::new(1));
        let chomp = CHOMP::new(
            cost,
            1,
            CHOMPConfig {
                init_strategy: InitStrategy::Seed,
                ..Default::default()
            },
        );

        let seed = make_trajectory(1, &[vec![0.0], vec![5.0], vec![3.0], vec![1.0]]);
        let traj = chomp.init_trajectory(&[0.0], &[1.0], Some(&seed));

        assert_eq!(traj.len(), 4);
        assert!((traj.position(0, 1) - 5.0).abs() < 1e-10, "Should preserve seed");
    }

    #[test]
    fn init_perturbed_differs_from_linear() {
        let cost = Box::new(SmoothnessCost::new(2));
        let chomp = CHOMP::new(
            cost,
            2,
            CHOMPConfig {
                init_strategy: InitStrategy::Perturbed {
                    num_waypoints: 10,
                    magnitude: 0.5,
                },
                ..Default::default()
            },
        );

        let start = [0.0, 0.0];
        let goal = [1.0, 1.0];
        let traj = chomp.init_trajectory(&start, &goal, None);

        assert_eq!(traj.len(), 10);
        // Start and goal should be exact
        assert!((traj.position(0, 0) - 0.0).abs() < 1e-10);
        assert!((traj.position(0, 9) - 1.0).abs() < 1e-10);

        // Interior should differ from linear (with high probability)
        // With magnitude=0.5, at least one interior point should deviate
        let any_deviated = (1..9).any(|i| {
            let expected = i as f64 / 9.0;
            (traj.position(0, i) - expected).abs() > 0.01
        });
        assert!(any_deviated, "Perturbed init should differ from linear");
    }

    #[test]
    fn plan_convenience_method() {
        let cost = Box::new(SmoothnessCost::new(1));
        let chomp = CHOMP::new(
            cost,
            1,
            CHOMPConfig {
                init_strategy: InitStrategy::Linear { num_waypoints: 10 },
                learning_rate: 0.01,
                max_iterations: 10,
                ..Default::default()
            },
        );

        let result = chomp.plan(&[0.0], &[1.0], None);
        assert_eq!(result.trajectory.len(), 10);
        assert!((result.trajectory.position(0, 0) - 0.0).abs() < 1e-10);
        assert!((result.trajectory.position(0, 9) - 1.0).abs() < 1e-10);
    }

    #[test]
    fn matrix_inverse_identity() {
        // Invert 3×3 identity → should get identity back
        let id = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0];
        let inv = invert_matrix(&id, 3);
        for i in 0..3 {
            for j in 0..3 {
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!(
                    (inv[i * 3 + j] - expected).abs() < 1e-10,
                    "inv[{},{}] = {}, expected {}",
                    i, j, inv[i * 3 + j], expected
                );
            }
        }
    }

    #[test]
    fn matrix_inverse_2x2() {
        // [[2, 1], [1, 3]] → [[3/5, -1/5], [-1/5, 2/5]]
        let m = vec![2.0, 1.0, 1.0, 3.0];
        let inv = invert_matrix(&m, 2);
        assert!((inv[0] - 0.6).abs() < 1e-10);
        assert!((inv[1] - (-0.2)).abs() < 1e-10);
        assert!((inv[2] - (-0.2)).abs() < 1e-10);
        assert!((inv[3] - 0.4).abs() < 1e-10);
    }
}
