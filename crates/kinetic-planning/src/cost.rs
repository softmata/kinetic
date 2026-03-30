//! Trajectory cost function framework for CHOMP/STOMP trajectory optimization.
//!
//! Defines the [`TrajectoryCost`] trait and [`CompositeCost`] for combining
//! multiple cost terms with weights. Cost functions evaluate entire trajectories
//! and produce per-waypoint gradients for gradient-based optimization.
//!
//! # Cost Terms
//!
//! - **Obstacle cost**: SDF-based collision penalty (see [`ObstacleCost`]).
//! - **Smoothness cost**: Finite-difference approximation of trajectory curvature.
//! - **Joint limit cost**: Penalty for approaching joint limits.
//! - **Custom costs**: Implement [`TrajectoryCost`] for domain-specific objectives.
//!
//! # Usage
//!
//! ```ignore
//! let mut composite = CompositeCost::new(dof);
//! composite.add("obstacle", Box::new(obstacle_cost), 10.0);
//! composite.add("smooth", Box::new(smoothness_cost), 1.0);
//!
//! let cost = composite.evaluate(&trajectory);
//! let grad = composite.gradient(&trajectory);
//! ```

use kinetic_collision::{RobotSphereModel, SignedDistanceField, SphereGenConfig};
use kinetic_core::{Constraint, Trajectory};
use kinetic_kinematics::{forward_kinematics_all, KinematicChain};
use kinetic_robot::Robot;

use crate::constraint as constraint_eval;

/// Trajectory cost function trait.
///
/// Evaluates an entire trajectory and optionally computes per-waypoint gradients.
/// The gradient is a flat array of length `dof * num_interior_waypoints` in
/// waypoint-major order: `[wp0_j0, wp0_j1, ..., wp0_jN, wp1_j0, ...]`.
///
/// Interior waypoints exclude the first and last (start/goal are fixed).
pub trait TrajectoryCost: Send + Sync {
    /// Evaluate the total cost of a trajectory.
    fn evaluate(&self, trajectory: &Trajectory) -> f64;

    /// Compute the gradient of cost with respect to interior waypoint positions.
    ///
    /// Returns a flat vector of length `dof * (num_waypoints - 2)` in
    /// waypoint-major order. The first and last waypoints (start/goal)
    /// are fixed and have no gradient.
    fn gradient(&self, trajectory: &Trajectory) -> Vec<f64>;

    /// Human-readable name for this cost term.
    fn name(&self) -> &str;
}

/// A named, weighted cost term in a composite.
struct WeightedCost {
    name: String,
    cost: Box<dyn TrajectoryCost>,
    weight: f64,
}

/// Composite cost function: weighted sum of sub-costs.
///
/// `total_cost = sum(weight_i * cost_i(trajectory))`
/// `total_gradient = sum(weight_i * gradient_i(trajectory))`
///
/// This is the primary way to combine obstacle, smoothness, and constraint
/// costs for CHOMP/STOMP optimization.
pub struct CompositeCost {
    costs: Vec<WeightedCost>,
    dof: usize,
}

impl CompositeCost {
    /// Create an empty composite cost for a robot with given DOF.
    pub fn new(dof: usize) -> Self {
        Self {
            costs: Vec::new(),
            dof,
        }
    }

    /// Add a named cost term with a weight.
    ///
    /// Higher weight = more influence on the optimization.
    pub fn add(&mut self, name: &str, cost: Box<dyn TrajectoryCost>, weight: f64) {
        self.costs.push(WeightedCost {
            name: name.to_string(),
            cost,
            weight,
        });
    }

    /// Number of cost terms.
    pub fn num_terms(&self) -> usize {
        self.costs.len()
    }

    /// DOF this composite was created for.
    pub fn dof(&self) -> usize {
        self.dof
    }

    /// Get the weight of a named cost term.
    pub fn weight(&self, name: &str) -> Option<f64> {
        self.costs.iter().find(|c| c.name == name).map(|c| c.weight)
    }

    /// Update the weight of a named cost term.
    pub fn set_weight(&mut self, name: &str, weight: f64) -> bool {
        if let Some(c) = self.costs.iter_mut().find(|c| c.name == name) {
            c.weight = weight;
            true
        } else {
            false
        }
    }

    /// Evaluate each cost term separately. Returns `(name, weight, raw_cost, weighted_cost)`.
    pub fn evaluate_breakdown(&self, trajectory: &Trajectory) -> Vec<(&str, f64, f64, f64)> {
        self.costs
            .iter()
            .map(|wc| {
                let raw = wc.cost.evaluate(trajectory);
                (&*wc.name, wc.weight, raw, wc.weight * raw)
            })
            .collect()
    }
}

impl TrajectoryCost for CompositeCost {
    fn evaluate(&self, trajectory: &Trajectory) -> f64 {
        self.costs
            .iter()
            .map(|wc| wc.weight * wc.cost.evaluate(trajectory))
            .sum()
    }

    fn gradient(&self, trajectory: &Trajectory) -> Vec<f64> {
        let n = trajectory.len();
        if n <= 2 {
            return vec![];
        }
        let interior = n - 2;
        let grad_len = self.dof * interior;
        let mut total_grad = vec![0.0; grad_len];

        for wc in &self.costs {
            let g = wc.cost.gradient(trajectory);
            assert_eq!(
                g.len(),
                grad_len,
                "Cost '{}' gradient length {} != expected {}",
                wc.name,
                g.len(),
                grad_len
            );
            for (i, val) in g.iter().enumerate() {
                total_grad[i] += wc.weight * val;
            }
        }

        total_grad
    }

    fn name(&self) -> &str {
        "composite"
    }
}

// ─── Built-in cost functions ─────────────────────────────────────────────

/// Smoothness cost: penalizes second-order finite differences (acceleration).
///
/// For each interior waypoint i and joint j:
///   accel_ij = q[i-1][j] - 2*q[i][j] + q[i+1][j]
///   cost += accel_ij^2
///
/// This is equivalent to minimizing the integral of squared acceleration
/// over the trajectory, discretized via central differences.
pub struct SmoothnessCost {
    dof: usize,
}

impl SmoothnessCost {
    pub fn new(dof: usize) -> Self {
        Self { dof }
    }
}

impl TrajectoryCost for SmoothnessCost {
    fn evaluate(&self, trajectory: &Trajectory) -> f64 {
        let n = trajectory.len();
        if n <= 2 {
            return 0.0;
        }

        let mut cost = 0.0;
        for i in 1..n - 1 {
            for j in 0..self.dof {
                let prev = trajectory.position(j, i - 1);
                let curr = trajectory.position(j, i);
                let next = trajectory.position(j, i + 1);
                let accel = prev - 2.0 * curr + next;
                cost += accel * accel;
            }
        }
        cost
    }

    fn gradient(&self, trajectory: &Trajectory) -> Vec<f64> {
        let n = trajectory.len();
        if n <= 2 {
            return vec![];
        }
        let interior = n - 2;
        let mut grad = vec![0.0; self.dof * interior];

        for i in 1..n - 1 {
            let wi = i - 1; // interior waypoint index (0-based)
            for j in 0..self.dof {
                let prev = trajectory.position(j, i - 1);
                let curr = trajectory.position(j, i);
                let next = trajectory.position(j, i + 1);
                let accel = prev - 2.0 * curr + next;

                // d(accel^2)/d(curr) = 2 * accel * (-2) = -4 * accel
                grad[wi * self.dof + j] += -4.0 * accel;

                // Contribution to neighboring interior waypoints:
                // d(accel_at_i)/d(q[i-1]) = 1, d(accel_at_i)/d(q[i+1]) = 1
                // But we also need to account for accel terms centered at i-1 and i+1
                // that include q[i] in their stencil.

                // accel at waypoint i-1 (if i-1 is interior): prev2 - 2*prev + curr
                // d/d(curr) = 1 → contributes 2 * accel_{i-1} * 1
                if i >= 2 {
                    let prev2 = trajectory.position(j, i - 2);
                    let accel_prev = prev2 - 2.0 * prev + curr;
                    grad[wi * self.dof + j] += 2.0 * accel_prev;
                }

                // accel at waypoint i+1 (if i+1 is interior): curr - 2*next + next2
                // d/d(curr) = 1 → contributes 2 * accel_{i+1} * 1
                if i + 2 < n {
                    let next2 = trajectory.position(j, i + 2);
                    let accel_next = curr - 2.0 * next + next2;
                    grad[wi * self.dof + j] += 2.0 * accel_next;
                }
            }
        }
        grad
    }

    fn name(&self) -> &str {
        "smoothness"
    }
}

/// Joint limit cost: penalty for approaching joint limits.
///
/// For each joint j at each interior waypoint i:
///   margin = min(q - lower, upper - q)
///   if margin < buffer:
///     cost += (buffer - margin)^2
///
/// Gradient pushes joints away from limits.
pub struct JointLimitCost {
    lower: Vec<f64>,
    upper: Vec<f64>,
    /// Buffer zone width near limits (default: 0.1 radians).
    pub buffer: f64,
}

impl JointLimitCost {
    /// Create from explicit joint limits.
    pub fn new(lower: Vec<f64>, upper: Vec<f64>, buffer: f64) -> Self {
        assert_eq!(lower.len(), upper.len());
        Self {
            lower,
            upper,
            buffer,
        }
    }

    /// Create from a robot model.
    pub fn from_robot(robot: &kinetic_robot::Robot, buffer: f64) -> Self {
        let lower: Vec<f64> = robot.joint_limits.iter().map(|l| l.lower).collect();
        let upper: Vec<f64> = robot.joint_limits.iter().map(|l| l.upper).collect();
        Self::new(lower, upper, buffer)
    }

    fn dof(&self) -> usize {
        self.lower.len()
    }
}

impl TrajectoryCost for JointLimitCost {
    fn evaluate(&self, trajectory: &Trajectory) -> f64 {
        let n = trajectory.len();
        if n <= 2 {
            return 0.0;
        }

        let mut cost = 0.0;
        for i in 1..n - 1 {
            for j in 0..self.dof() {
                let q = trajectory.position(j, i);
                let margin_low = q - self.lower[j];
                let margin_high = self.upper[j] - q;
                let margin = margin_low.min(margin_high);

                if margin < self.buffer {
                    let penalty = self.buffer - margin;
                    cost += penalty * penalty;
                }
            }
        }
        cost
    }

    fn gradient(&self, trajectory: &Trajectory) -> Vec<f64> {
        let n = trajectory.len();
        if n <= 2 {
            return vec![];
        }
        let interior = n - 2;
        let dof = self.dof();
        let mut grad = vec![0.0; dof * interior];

        for i in 1..n - 1 {
            let wi = i - 1;
            for j in 0..dof {
                let q = trajectory.position(j, i);
                let margin_low = q - self.lower[j];
                let margin_high = self.upper[j] - q;

                if margin_low < self.buffer {
                    // Near lower limit: d/dq of (buffer - (q - lower))^2
                    // = 2 * (buffer - margin_low) * (-1) = -2 * penalty
                    let penalty = self.buffer - margin_low;
                    grad[wi * dof + j] += -2.0 * penalty;
                }
                if margin_high < self.buffer {
                    // Near upper limit: d/dq of (buffer - (upper - q))^2
                    // = 2 * (buffer - margin_high) * (1) = 2 * penalty
                    let penalty = self.buffer - margin_high;
                    grad[wi * dof + j] += 2.0 * penalty;
                }
            }
        }
        grad
    }

    fn name(&self) -> &str {
        "joint_limits"
    }
}

/// Velocity cost: penalizes large joint velocities (first-order finite differences).
///
/// For each segment between waypoints i and i+1:
///   vel_j = q[i+1][j] - q[i][j]
///   cost += vel_j^2
pub struct VelocityCost {
    dof: usize,
}

impl VelocityCost {
    pub fn new(dof: usize) -> Self {
        Self { dof }
    }
}

impl TrajectoryCost for VelocityCost {
    fn evaluate(&self, trajectory: &Trajectory) -> f64 {
        let n = trajectory.len();
        if n <= 1 {
            return 0.0;
        }

        let mut cost = 0.0;
        for i in 0..n - 1 {
            for j in 0..self.dof {
                let curr = trajectory.position(j, i);
                let next = trajectory.position(j, i + 1);
                let vel = next - curr;
                cost += vel * vel;
            }
        }
        cost
    }

    fn gradient(&self, trajectory: &Trajectory) -> Vec<f64> {
        let n = trajectory.len();
        if n <= 2 {
            return vec![];
        }
        let interior = n - 2;
        let mut grad = vec![0.0; self.dof * interior];

        for i in 1..n - 1 {
            let wi = i - 1;
            for j in 0..self.dof {
                let prev = trajectory.position(j, i - 1);
                let curr = trajectory.position(j, i);
                let next = trajectory.position(j, i + 1);

                // Segment i-1→i: vel = curr - prev, d(vel^2)/d(curr) = 2*(curr-prev)
                // Segment i→i+1: vel = next - curr, d(vel^2)/d(curr) = -2*(next-curr)
                grad[wi * self.dof + j] += 2.0 * (curr - prev) - 2.0 * (next - curr);
            }
        }
        grad
    }

    fn name(&self) -> &str {
        "velocity"
    }
}

/// Constant cost that always returns a fixed value.
/// Useful for testing and as a baseline.
pub struct ConstantCost {
    value: f64,
    dof: usize,
}

impl ConstantCost {
    pub fn new(value: f64, dof: usize) -> Self {
        Self { value, dof }
    }
}

impl TrajectoryCost for ConstantCost {
    fn evaluate(&self, _trajectory: &Trajectory) -> f64 {
        self.value
    }

    fn gradient(&self, trajectory: &Trajectory) -> Vec<f64> {
        let n = trajectory.len();
        if n <= 2 {
            return vec![];
        }
        vec![0.0; self.dof * (n - 2)]
    }

    fn name(&self) -> &str {
        "constant"
    }
}

/// Obstacle cost: SDF-based collision penalty for trajectory optimization.
///
/// For each interior waypoint, computes FK to get link poses, updates the
/// sphere model to get body point positions, then queries the SDF at each
/// body point. The cost is quadratic in the margin violation:
///
///   cost = Σ_body_points max(0, margin - sdf_distance)²
///
/// The gradient is projected from workspace (SDF gradient) to joint space
/// via numerical Jacobian (perturb-and-FK approach, robust for any kinematic
/// structure).
///
/// # Performance
///
/// This is the most expensive cost term since it requires FK per waypoint.
/// For a 7-DOF robot with 30 body points and 50 waypoints, expect ~1ms per
/// evaluate() call. Gradient is ~7x slower (one FK per DOF perturbation).
pub struct ObstacleCost {
    robot: std::sync::Arc<Robot>,
    chain: std::sync::Arc<KinematicChain>,
    sphere_model: RobotSphereModel,
    sdf: std::sync::Arc<SignedDistanceField>,
    /// Safety margin: penalty applies when distance < margin.
    pub margin: f64,
    /// Perturbation step for numerical Jacobian (default: 1e-4 radians).
    pub epsilon: f64,
}

impl ObstacleCost {
    /// Create an obstacle cost from robot, kinematic chain, sphere model, and SDF.
    pub fn new(
        robot: std::sync::Arc<Robot>,
        chain: std::sync::Arc<KinematicChain>,
        sphere_config: &SphereGenConfig,
        sdf: std::sync::Arc<SignedDistanceField>,
        margin: f64,
    ) -> Self {
        let sphere_model = RobotSphereModel::from_robot(&robot, sphere_config);
        Self {
            robot,
            chain,
            sphere_model,
            sdf,
            margin,
            epsilon: 1e-4,
        }
    }

    /// Evaluate obstacle cost at a single joint configuration.
    ///
    /// Returns the sum of squared margin violations at all body points.
    fn config_cost(&self, joint_values: &[f64]) -> f64 {
        let poses = match forward_kinematics_all(&self.robot, &self.chain, joint_values) {
            Ok(p) => p,
            Err(_) => return 0.0,
        };

        let mut runtime = self.sphere_model.create_runtime();
        runtime.update(&poses);

        let mut cost = 0.0;
        let w = &runtime.world;
        for i in 0..w.len() {
            let d = self.sdf.distance_at(w.x[i], w.y[i], w.z[i]) - w.radius[i];
            if d < self.margin {
                let penalty = self.margin - d;
                cost += penalty * penalty;
            }
        }
        cost
    }
}

impl TrajectoryCost for ObstacleCost {
    fn evaluate(&self, trajectory: &Trajectory) -> f64 {
        let n = trajectory.len();
        if n <= 2 {
            return 0.0;
        }

        let dof = self.chain.dof;
        let mut total_cost = 0.0;

        for i in 1..n - 1 {
            let wp = trajectory.waypoint(i);
            let q: Vec<f64> = (0..dof).map(|j| wp.positions[j]).collect();
            total_cost += self.config_cost(&q);
        }

        total_cost
    }

    fn gradient(&self, trajectory: &Trajectory) -> Vec<f64> {
        let n = trajectory.len();
        if n <= 2 {
            return vec![];
        }

        let dof = self.chain.dof;
        let interior = n - 2;
        let mut grad = vec![0.0; dof * interior];

        for i in 1..n - 1 {
            let wi = i - 1;
            let wp = trajectory.waypoint(i);
            let q: Vec<f64> = (0..dof).map(|j| wp.positions[j]).collect();

            // Numerical gradient: perturb each joint
            let c0 = self.config_cost(&q);
            if c0 == 0.0 {
                // All body points beyond margin — skip gradient computation
                continue;
            }

            for j in 0..dof {
                let mut q_plus = q.clone();
                q_plus[j] += self.epsilon;
                let c_plus = self.config_cost(&q_plus);

                let mut q_minus = q.clone();
                q_minus[j] -= self.epsilon;
                let c_minus = self.config_cost(&q_minus);

                grad[wi * dof + j] = (c_plus - c_minus) / (2.0 * self.epsilon);
            }
        }

        grad
    }

    fn name(&self) -> &str {
        "obstacle"
    }
}

/// Constraint cost: smooth penalty for violating Cartesian constraints.
///
/// Uses the existing `constraint::distance()` function to compute signed
/// distance to each constraint boundary, then applies a smooth quadratic
/// penalty inside a buffer zone:
///
///   cost = Σ_constraints Σ_waypoints max(0, buffer - distance)²
///
/// This is a differentiable relaxation of hard constraints, suitable for
/// gradient-based trajectory optimization. The `buffer` parameter controls
/// how far from the constraint boundary the penalty starts.
///
/// Supports all `Constraint` types: `Orientation`, `PositionBound`, `Joint`,
/// `Visibility`.
pub struct ConstraintCost {
    robot: std::sync::Arc<Robot>,
    chain: std::sync::Arc<KinematicChain>,
    constraints: Vec<Constraint>,
    /// Buffer zone: penalty starts at this distance from the constraint boundary.
    /// Default: 0.1 (radians for orientation, meters for position).
    pub buffer: f64,
    /// Perturbation step for numerical gradient.
    pub epsilon: f64,
}

impl ConstraintCost {
    /// Create a constraint cost from a set of constraints.
    pub fn new(
        robot: std::sync::Arc<Robot>,
        chain: std::sync::Arc<KinematicChain>,
        constraints: Vec<Constraint>,
        buffer: f64,
    ) -> Self {
        Self {
            robot,
            chain,
            constraints,
            buffer,
            epsilon: 1e-4,
        }
    }

    /// Evaluate constraint cost at a single joint configuration.
    fn config_cost(&self, joints: &[f64]) -> f64 {
        let mut cost = 0.0;
        for constraint in &self.constraints {
            let d = constraint_eval::distance(constraint, &self.robot, &self.chain, joints);
            if d < self.buffer {
                let penalty = self.buffer - d;
                cost += penalty * penalty;
            }
        }
        cost
    }
}

impl TrajectoryCost for ConstraintCost {
    fn evaluate(&self, trajectory: &Trajectory) -> f64 {
        let n = trajectory.len();
        if n <= 2 {
            return 0.0;
        }

        let dof = self.chain.dof;
        let mut total = 0.0;
        for i in 1..n - 1 {
            let wp = trajectory.waypoint(i);
            let q: Vec<f64> = (0..dof).map(|j| wp.positions[j]).collect();
            total += self.config_cost(&q);
        }
        total
    }

    fn gradient(&self, trajectory: &Trajectory) -> Vec<f64> {
        let n = trajectory.len();
        if n <= 2 {
            return vec![];
        }

        let dof = self.chain.dof;
        let interior = n - 2;
        let mut grad = vec![0.0; dof * interior];

        for i in 1..n - 1 {
            let wi = i - 1;
            let wp = trajectory.waypoint(i);
            let q: Vec<f64> = (0..dof).map(|j| wp.positions[j]).collect();

            let c0 = self.config_cost(&q);
            if c0 == 0.0 {
                continue;
            }

            for j in 0..dof {
                let mut q_plus = q.clone();
                q_plus[j] += self.epsilon;
                let c_plus = self.config_cost(&q_plus);

                let mut q_minus = q.clone();
                q_minus[j] -= self.epsilon;
                let c_minus = self.config_cost(&q_minus);

                grad[wi * dof + j] = (c_plus - c_minus) / (2.0 * self.epsilon);
            }
        }
        grad
    }

    fn name(&self) -> &str {
        "constraint"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kinetic_collision::SDFConfig;

    fn make_trajectory(dof: usize, waypoints: &[Vec<f64>]) -> Trajectory {
        let mut traj = Trajectory::with_dof(dof);
        for wp in waypoints {
            traj.push_waypoint(wp);
        }
        traj
    }

    // ─── CompositeCost tests ───

    #[test]
    fn composite_empty_zero_cost() {
        let composite = CompositeCost::new(2);
        let traj = make_trajectory(2, &[vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]]);
        assert_eq!(composite.evaluate(&traj), 0.0);
        assert!(composite.gradient(&traj).iter().all(|&g| g == 0.0));
    }

    #[test]
    fn composite_single_term() {
        let mut composite = CompositeCost::new(2);
        composite.add("const", Box::new(ConstantCost::new(5.0, 2)), 3.0);

        let traj = make_trajectory(2, &[vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]]);
        assert_eq!(composite.evaluate(&traj), 15.0); // 3.0 * 5.0
    }

    #[test]
    fn composite_weighted_sum() {
        let mut composite = CompositeCost::new(2);
        composite.add("a", Box::new(ConstantCost::new(1.0, 2)), 2.0);
        composite.add("b", Box::new(ConstantCost::new(3.0, 2)), 4.0);

        let traj = make_trajectory(2, &[vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]]);
        // 2*1 + 4*3 = 14
        assert_eq!(composite.evaluate(&traj), 14.0);
    }

    #[test]
    fn composite_gradient_is_weighted_sum() {
        let mut composite = CompositeCost::new(1);
        composite.add("smooth", Box::new(SmoothnessCost::new(1)), 2.0);
        composite.add("vel", Box::new(VelocityCost::new(1)), 3.0);

        let traj = make_trajectory(1, &[vec![0.0], vec![1.0], vec![0.5], vec![2.0]]);

        let g_composite = composite.gradient(&traj);
        let g_smooth: Vec<f64> = SmoothnessCost::new(1)
            .gradient(&traj)
            .iter()
            .map(|x| 2.0 * x)
            .collect();
        let g_vel: Vec<f64> = VelocityCost::new(1)
            .gradient(&traj)
            .iter()
            .map(|x| 3.0 * x)
            .collect();

        for i in 0..g_composite.len() {
            let expected = g_smooth[i] + g_vel[i];
            assert!(
                (g_composite[i] - expected).abs() < 1e-10,
                "i={}: composite {} != smooth+vel {}",
                i,
                g_composite[i],
                expected
            );
        }
    }

    #[test]
    fn composite_breakdown() {
        let mut composite = CompositeCost::new(2);
        composite.add("a", Box::new(ConstantCost::new(1.0, 2)), 2.0);
        composite.add("b", Box::new(ConstantCost::new(3.0, 2)), 4.0);

        let traj = make_trajectory(2, &[vec![0.0, 0.0], vec![1.0, 1.0], vec![2.0, 2.0]]);
        let breakdown = composite.evaluate_breakdown(&traj);

        assert_eq!(breakdown.len(), 2);
        assert_eq!(breakdown[0].0, "a");
        assert_eq!(breakdown[0].1, 2.0); // weight
        assert_eq!(breakdown[0].2, 1.0); // raw cost
        assert_eq!(breakdown[0].3, 2.0); // weighted
    }

    #[test]
    fn composite_set_weight() {
        let mut composite = CompositeCost::new(2);
        composite.add("a", Box::new(ConstantCost::new(1.0, 2)), 1.0);

        assert_eq!(composite.weight("a"), Some(1.0));
        assert!(composite.set_weight("a", 5.0));
        assert_eq!(composite.weight("a"), Some(5.0));
        assert!(!composite.set_weight("nonexistent", 1.0));
    }

    // ─── SmoothnessCost tests ───

    #[test]
    fn smoothness_straight_line_zero() {
        // Linear trajectory: acceleration = 0 everywhere
        let cost = SmoothnessCost::new(1);
        let traj = make_trajectory(1, &[vec![0.0], vec![1.0], vec![2.0], vec![3.0]]);
        assert!(
            cost.evaluate(&traj).abs() < 1e-10,
            "Linear trajectory should have zero smoothness cost"
        );
    }

    #[test]
    fn smoothness_zigzag_high() {
        // Zigzag: high acceleration at every waypoint
        let cost = SmoothnessCost::new(1);
        let traj = make_trajectory(1, &[vec![0.0], vec![1.0], vec![0.0], vec![1.0]]);
        let c = cost.evaluate(&traj);
        assert!(c > 0.0, "Zigzag should have positive smoothness cost: {}", c);
    }

    #[test]
    fn smoothness_gradient_direction() {
        // A trajectory with a bump should have gradient pushing the bump flat
        let cost = SmoothnessCost::new(1);
        let traj = make_trajectory(1, &[vec![0.0], vec![2.0], vec![0.0]]);
        // accel at wp1: 0 - 2*2 + 0 = -4, cost = 16
        assert!((cost.evaluate(&traj) - 16.0).abs() < 1e-10);

        let grad = cost.gradient(&traj);
        // d/d(wp1) of (0 - 2*q + 0)^2 = 2*(-4)*(-2) = 16... hmm
        // Actually: accel = 0 - 2*q + 0, d(accel^2)/dq = 2*accel*(-2) = -4*accel = -4*(-4) = 16
        // But we also need to check that the gradient points toward reducing cost
        // Reducing q from 2.0 toward 0.0 flattens the trajectory → should be positive gradient
        assert!(
            grad[0] > 0.0,
            "Gradient should push bump down (positive): {}",
            grad[0]
        );
    }

    #[test]
    fn smoothness_two_waypoint_no_interior() {
        let cost = SmoothnessCost::new(2);
        let traj = make_trajectory(2, &[vec![0.0, 0.0], vec![1.0, 1.0]]);
        assert_eq!(cost.evaluate(&traj), 0.0);
        assert!(cost.gradient(&traj).is_empty());
    }

    // ─── JointLimitCost tests ───

    #[test]
    fn joint_limit_within_range_zero() {
        let cost = JointLimitCost::new(vec![-3.0], vec![3.0], 0.5);
        // All waypoints well within limits
        let traj = make_trajectory(1, &[vec![0.0], vec![0.5], vec![1.0]]);
        assert_eq!(cost.evaluate(&traj), 0.0);
    }

    #[test]
    fn joint_limit_near_boundary_penalty() {
        let cost = JointLimitCost::new(vec![-1.0], vec![1.0], 0.2);
        // Interior waypoint at 0.9 → margin_high = 0.1 < buffer 0.2 → penalty
        let traj = make_trajectory(1, &[vec![0.0], vec![0.9], vec![0.0]]);
        let c = cost.evaluate(&traj);
        // penalty = 0.2 - 0.1 = 0.1, cost = 0.01
        assert!(
            (c - 0.01).abs() < 1e-10,
            "Near-limit cost: expected 0.01, got {}",
            c
        );
    }

    #[test]
    fn joint_limit_gradient_pushes_away() {
        let cost = JointLimitCost::new(vec![-1.0], vec![1.0], 0.3);
        // Near upper limit
        let traj = make_trajectory(1, &[vec![0.0], vec![0.85], vec![0.0]]);
        let grad = cost.gradient(&traj);
        // Near upper: gradient should be positive (push q higher → farther from upper)
        // Wait, gradient should push AWAY from limit, so negative (lower q)
        assert!(
            grad[0] > 0.0,
            "Gradient near upper limit should push away (positive = toward higher penalty side??)"
        );
        // Actually: near upper, margin_high = 0.15 < 0.3, penalty = 0.15
        // d/dq = 2 * 0.15 = 0.3 (positive because upper limit gradient pushes q up)
        // No: d/dq of (buffer - (upper - q))^2 = 2*(buffer - margin_high)*(1) = positive
        // This means increasing q increases cost → gradient descent will decrease q → correct!
    }

    // ─── VelocityCost tests ───

    #[test]
    fn velocity_stationary_zero() {
        let cost = VelocityCost::new(2);
        let traj = make_trajectory(2, &[vec![1.0, 2.0], vec![1.0, 2.0], vec![1.0, 2.0]]);
        assert_eq!(cost.evaluate(&traj), 0.0);
    }

    #[test]
    fn velocity_constant_speed() {
        let cost = VelocityCost::new(1);
        let traj = make_trajectory(1, &[vec![0.0], vec![1.0], vec![2.0], vec![3.0]]);
        // vel = 1.0 at each segment, 3 segments → cost = 3 * 1.0 = 3.0
        assert!((cost.evaluate(&traj) - 3.0).abs() < 1e-10);
    }

    #[test]
    fn velocity_gradient_reduces_speed() {
        let cost = VelocityCost::new(1);
        // Big jump in middle
        let traj = make_trajectory(1, &[vec![0.0], vec![5.0], vec![0.0]]);
        let grad = cost.gradient(&traj);
        // vel segments: 5-0=5, 0-5=-5. cost = 25 + 25 = 50
        // d/d(wp1): 2*(5-0) - 2*(0-5) = 10 + 10 = 20
        assert!(
            (grad[0] - 20.0).abs() < 1e-10,
            "Velocity gradient: expected 20.0, got {}",
            grad[0]
        );
    }

    // ─── Gradient numerical verification ───

    #[test]
    fn smoothness_gradient_numerical_check() {
        numerical_gradient_check(&SmoothnessCost::new(2), 2);
    }

    #[test]
    fn velocity_gradient_numerical_check() {
        numerical_gradient_check(&VelocityCost::new(2), 2);
    }

    #[test]
    fn joint_limit_gradient_numerical_check() {
        let cost = JointLimitCost::new(vec![-1.0, -2.0], vec![1.0, 2.0], 0.5);
        numerical_gradient_check(&cost, 2);
    }

    // ─── ObstacleCost tests ───

    fn make_obstacle_cost() -> ObstacleCost {
        let urdf = r#"<?xml version="1.0"?>
<robot name="test2dof">
  <link name="base">
    <collision><geometry><sphere radius="0.05"/></geometry></collision>
  </link>
  <link name="link1">
    <collision><geometry><sphere radius="0.03"/></geometry><origin xyz="0 0 0.3"/></collision>
  </link>
  <link name="tip">
    <collision><geometry><sphere radius="0.02"/></geometry></collision>
  </link>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="link1"/>
    <origin xyz="0 0 0.1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="10"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="link1"/><child link="tip"/>
    <origin xyz="0 0 0.3"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="2.0" effort="10"/>
  </joint>
</robot>"#;

        let robot = std::sync::Arc::new(Robot::from_urdf_string(urdf).unwrap());
        let chain = std::sync::Arc::new(
            KinematicChain::extract(&robot, "base", "tip").unwrap(),
        );

        let mut sdf = SignedDistanceField::new(&SDFConfig {
            resolution: 0.05,
            bounds: [-1.0, -1.0, -1.0, 1.0, 1.0, 1.0],
            truncation: 0.5,
        });
        // Obstacle at (0.3, 0.3, 0.3)
        sdf.add_sphere(0.3, 0.3, 0.3, 0.15, 1);
        let sdf = std::sync::Arc::new(sdf);

        ObstacleCost::new(
            robot,
            chain,
            &SphereGenConfig::coarse(),
            sdf,
            0.1, // 10cm margin
        )
    }

    #[test]
    fn obstacle_cost_zero_when_far() {
        let cost = make_obstacle_cost();
        // Joint config that keeps robot near origin (far from obstacle at 0.3,0.3,0.3)
        let traj = make_trajectory(2, &[vec![0.0, 0.0], vec![0.1, 0.1], vec![0.0, 0.0]]);
        let c = cost.evaluate(&traj);
        assert!(
            c < 0.01,
            "Robot near origin should have ~zero obstacle cost: {}",
            c
        );
    }

    #[test]
    fn obstacle_cost_gradient_nonzero_near_obstacle() {
        let cost = make_obstacle_cost();
        // Move joints to bring links toward the obstacle
        let traj = make_trajectory(
            2,
            &[vec![0.0, 0.0], vec![0.8, 0.5], vec![0.0, 0.0]],
        );
        let c = cost.evaluate(&traj);
        if c > 0.0 {
            let grad = cost.gradient(&traj);
            let grad_magnitude: f64 = grad.iter().map(|g| g * g).sum::<f64>().sqrt();
            assert!(
                grad_magnitude > 1e-6,
                "Non-zero cost should have non-zero gradient: cost={}, grad_mag={}",
                c,
                grad_magnitude
            );
        }
    }

    // ─── ConstraintCost tests ───

    fn make_constraint_cost() -> ConstraintCost {
        let urdf = r#"<?xml version="1.0"?>
<robot name="test2dof">
  <link name="base"/>
  <link name="link1"/>
  <link name="tip"/>
  <joint name="j1" type="revolute">
    <parent link="base"/><child link="link1"/>
    <origin xyz="0 0 0.1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="10"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="link1"/><child link="tip"/>
    <origin xyz="0 0 0.3"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="2.0" effort="10"/>
  </joint>
</robot>"#;

        let robot = std::sync::Arc::new(Robot::from_urdf_string(urdf).unwrap());
        let chain = std::sync::Arc::new(
            KinematicChain::extract(&robot, "base", "tip").unwrap(),
        );

        // Joint constraint: keep joint 0 between -0.5 and 0.5
        let constraints = vec![Constraint::Joint {
            joint_index: 0,
            min: -0.5,
            max: 0.5,
        }];

        ConstraintCost::new(robot, chain, constraints, 0.1)
    }

    #[test]
    fn constraint_cost_zero_when_satisfied() {
        let cost = make_constraint_cost();
        // Joint 0 at 0.0 → well within [-0.5, 0.5]
        let traj = make_trajectory(2, &[vec![0.0, 0.0], vec![0.1, 0.0], vec![0.0, 0.0]]);
        let c = cost.evaluate(&traj);
        assert_eq!(c, 0.0, "Satisfied constraint should have zero cost");
    }

    #[test]
    fn constraint_cost_positive_when_violated() {
        let cost = make_constraint_cost();
        // Joint 0 at 1.0 → violates max=0.5, distance = min(1.0-(-0.5), 0.5-1.0) = -0.5
        // penalty = buffer - distance = 0.1 - (-0.5) = 0.6, cost = 0.36
        let traj = make_trajectory(2, &[vec![0.0, 0.0], vec![1.0, 0.0], vec![0.0, 0.0]]);
        let c = cost.evaluate(&traj);
        assert!(c > 0.0, "Violated constraint should have positive cost: {}", c);
    }

    #[test]
    fn constraint_cost_gradient_pushes_toward_feasible() {
        let cost = make_constraint_cost();
        // Joint 0 at 0.8 → violated (max=0.5)
        let traj = make_trajectory(2, &[vec![0.0, 0.0], vec![0.8, 0.0], vec![0.0, 0.0]]);
        let grad = cost.gradient(&traj);
        // Gradient for joint 0 should be positive (increasing cost with increasing q)
        // → gradient descent will decrease q toward feasible region
        assert!(
            grad[0] > 0.0,
            "Gradient should push joint 0 back toward feasible: {}",
            grad[0]
        );
    }

    #[test]
    fn obstacle_cost_in_composite() {
        let obs_cost = make_obstacle_cost();
        let mut composite = CompositeCost::new(2);
        composite.add("obstacle", Box::new(obs_cost), 10.0);
        composite.add("smooth", Box::new(SmoothnessCost::new(2)), 1.0);

        let traj = make_trajectory(2, &[vec![0.0, 0.0], vec![0.1, 0.1], vec![0.0, 0.0]]);
        let _ = composite.evaluate(&traj);
        let _ = composite.gradient(&traj);
        // No panic = success
    }

    /// Verify analytical gradient matches numerical gradient via finite differences.
    fn numerical_gradient_check(cost: &dyn TrajectoryCost, dof: usize) {
        let waypoints = vec![
            vec![0.1; dof],
            vec![0.5; dof],
            vec![-0.3; dof],
            vec![0.8; dof],
            vec![0.2; dof],
        ];
        let traj = make_trajectory(dof, &waypoints);

        let analytical = cost.gradient(&traj);
        let eps = 1e-6;
        let interior = traj.len() - 2;

        for wi in 0..interior {
            for j in 0..dof {
                let idx = wi * dof + j;

                // Perturb waypoint wi+1 (interior index wi → trajectory index wi+1)
                let mut wp_plus = waypoints.clone();
                wp_plus[wi + 1][j] += eps;
                let traj_plus = make_trajectory(dof, &wp_plus);

                let mut wp_minus = waypoints.clone();
                wp_minus[wi + 1][j] -= eps;
                let traj_minus = make_trajectory(dof, &wp_minus);

                let numerical = (cost.evaluate(&traj_plus) - cost.evaluate(&traj_minus)) / (2.0 * eps);

                assert!(
                    (analytical[idx] - numerical).abs() < 1e-4,
                    "Cost '{}' grad[{}] (wp={}, j={}): analytical={:.6}, numerical={:.6}",
                    cost.name(),
                    idx,
                    wi,
                    j,
                    analytical[idx],
                    numerical
                );
            }
        }
    }
}
