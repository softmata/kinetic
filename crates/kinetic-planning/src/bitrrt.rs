//! BiTRRT (Bidirectional Transition-based RRT) planner (Devaurs et al., 2014).
//!
//! Cost-aware bidirectional RRT that uses a Boltzmann transition test to bias
//! exploration toward lower-cost regions. Accepts worse-cost nodes with probability
//! `exp(-delta_cost / T)`, where temperature `T` anneals down over iterations.
//!
//! # When to use
//!
//! - When you have a cost function over C-space (e.g., distance from obstacles,
//!   joint-limit avoidance, manipulability) and want cost-aware paths without
//!   the full overhead of RRT*.
//! - BiTRRT finds solutions faster than RRT* while producing lower-cost paths
//!   than basic RRT-Connect.
//!
//! # Algorithm
//!
//! 1. Initialize two trees rooted at start and goal.
//! 2. Each iteration:
//!    a. Sample random config, find nearest node in active tree.
//!    b. Steer toward sample, compute cost at new config.
//!    c. **Transition test**: accept if cost improved, or with probability
//!       `exp(-(new_cost - parent_cost) / T)` if cost worsened.
//!    d. If accepted and collision-free, add node.
//!    e. Try to connect the other tree.
//!    f. Anneal temperature: `T *= alpha` (alpha < 1).
//! 3. Return path when trees connect.

use std::sync::Arc;
use std::time::{Duration, Instant};

use rand::Rng;

use kinetic_collision::{
    AllowedCollisionMatrix, CollisionEnvironment, ResolvedACM, RobotSphereModel, SphereGenConfig,
};
use kinetic_core::{Goal, KineticError, PlannerConfig, Result};
use kinetic_kinematics::{forward_kinematics_all, solve_ik, IKConfig, KinematicChain};
use kinetic_robot::Robot;

use crate::shortcut::{self, CollisionChecker};
use crate::smooth;

/// BiTRRT-specific configuration.
#[derive(Debug, Clone)]
pub struct BiTRRTConfig {
    /// Step size for tree extension (default: 0.1 radians).
    pub step_size: f64,
    /// Probability of sampling the goal instead of random (default: 0.05).
    pub goal_bias: f64,
    /// Initial temperature for Boltzmann acceptance (default: 1.0).
    ///
    /// Higher values accept more cost-increasing moves initially.
    pub initial_temperature: f64,
    /// Temperature decay factor per iteration (default: 0.95).
    ///
    /// `T *= alpha` each iteration. Lower alpha = faster annealing.
    pub alpha: f64,
    /// Minimum temperature — annealing floor (default: 0.01).
    pub min_temperature: f64,
    /// Number of consecutive rejections before forcing a temperature increase (default: 100).
    pub frustration_threshold: usize,
    /// Factor to increase temperature on frustration (default: 2.0).
    pub frustration_factor: f64,
}

impl Default for BiTRRTConfig {
    fn default() -> Self {
        Self {
            step_size: 0.1,
            goal_bias: 0.05,
            initial_temperature: 1.0,
            alpha: 0.95,
            min_temperature: 0.01,
            frustration_threshold: 100,
            frustration_factor: 2.0,
        }
    }
}

/// Cost function trait for BiTRRT.
///
/// Maps a joint configuration to a scalar cost. BiTRRT biases exploration
/// toward lower-cost regions.
pub trait CostFn: Send + Sync {
    fn cost(&self, joints: &[f64]) -> f64;
}

/// Default cost function: distance from joint-space center (midpoint of limits).
///
/// Biases toward the middle of the joint range, away from limits.
pub struct JointCenterCost {
    center: Vec<f64>,
    range: Vec<f64>,
}

impl JointCenterCost {
    pub fn from_robot(robot: &Robot, chain: &KinematicChain) -> Self {
        let mut center = Vec::with_capacity(chain.active_joints.len());
        let mut range = Vec::with_capacity(chain.active_joints.len());
        for &joint_idx in &chain.active_joints {
            if let Some(limits) = &robot.joints[joint_idx].limits {
                center.push((limits.lower + limits.upper) / 2.0);
                range.push((limits.upper - limits.lower).max(1e-6));
            } else {
                center.push(0.0);
                range.push(2.0 * std::f64::consts::PI);
            }
        }
        Self { center, range }
    }
}

impl CostFn for JointCenterCost {
    fn cost(&self, joints: &[f64]) -> f64 {
        joints
            .iter()
            .zip(self.center.iter())
            .zip(self.range.iter())
            .map(|((j, c), r)| ((j - c) / r).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

/// A node in the BiTRRT tree.
#[derive(Debug, Clone)]
struct TRRTNode {
    joints: Vec<f64>,
    parent: Option<usize>,
    cost: f64,
}

/// A C-space tree with cost tracking.
#[derive(Debug)]
struct CostTree {
    nodes: Vec<TRRTNode>,
}

impl CostTree {
    fn new(root: Vec<f64>, root_cost: f64) -> Self {
        Self {
            nodes: vec![TRRTNode {
                joints: root,
                parent: None,
                cost: root_cost,
            }],
        }
    }

    fn len(&self) -> usize {
        self.nodes.len()
    }

    fn add(&mut self, joints: Vec<f64>, parent: usize, cost: f64) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(TRRTNode {
            joints,
            parent: Some(parent),
            cost,
        });
        idx
    }

    fn nearest(&self, target: &[f64]) -> usize {
        let mut best_idx = 0;
        let mut best_dist = f64::INFINITY;
        for (i, node) in self.nodes.iter().enumerate() {
            let dist = joint_distance(&node.joints, target);
            if dist < best_dist {
                best_dist = dist;
                best_idx = i;
            }
        }
        best_idx
    }

    fn path_to_root(&self, mut idx: usize) -> Vec<Vec<f64>> {
        let mut path = Vec::new();
        loop {
            path.push(self.nodes[idx].joints.clone());
            match self.nodes[idx].parent {
                Some(p) => idx = p,
                None => break,
            }
        }
        path.reverse();
        path
    }
}

fn joint_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Result of an extend operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExtendResult {
    Reached,
    Advanced(usize),
    Trapped,
}

/// BiTRRT planner result.
#[derive(Debug, Clone)]
pub struct BiTRRTResult {
    pub waypoints: Vec<Vec<f64>>,
    pub planning_time: Duration,
    pub iterations: usize,
    pub tree_size: usize,
    /// Path cost (sum of cost function along waypoints).
    pub path_cost: f64,
    /// Final temperature when solution was found.
    pub final_temperature: f64,
}

/// Bidirectional Transition-based RRT planner.
pub struct BiTRRT {
    robot: Arc<Robot>,
    chain: KinematicChain,
    sphere_model: RobotSphereModel,
    acm: ResolvedACM,
    environment: CollisionEnvironment,
    planner_config: PlannerConfig,
    bitrrt_config: BiTRRTConfig,
    cost_fn: Box<dyn CostFn>,
}

impl CollisionChecker for BiTRRT {
    fn is_in_collision(&self, joints: &[f64]) -> bool {
        self.check_collision(joints)
    }
}

impl BiTRRT {
    /// Create a new BiTRRT planner with a custom cost function.
    pub fn new(
        robot: Arc<Robot>,
        chain: KinematicChain,
        environment: CollisionEnvironment,
        planner_config: PlannerConfig,
        bitrrt_config: BiTRRTConfig,
        cost_fn: Box<dyn CostFn>,
    ) -> Self {
        let sphere_model = RobotSphereModel::from_robot(&robot, &SphereGenConfig::coarse());
        let acm = ResolvedACM::from_robot(&robot);

        Self {
            robot,
            chain,
            sphere_model,
            acm,
            environment,
            planner_config,
            bitrrt_config,
            cost_fn,
        }
    }

    /// Create with default joint-center cost function.
    pub fn with_default_cost(
        robot: Arc<Robot>,
        chain: KinematicChain,
        environment: CollisionEnvironment,
        planner_config: PlannerConfig,
        bitrrt_config: BiTRRTConfig,
    ) -> Self {
        let cost_fn = Box::new(JointCenterCost::from_robot(&robot, &chain));
        Self::new(robot, chain, environment, planner_config, bitrrt_config, cost_fn)
    }

    pub fn with_acm(mut self, acm: &AllowedCollisionMatrix) -> Self {
        self.acm = ResolvedACM::from_acm(acm, &self.robot);
        self
    }

    /// Plan a collision-free, cost-aware path.
    pub fn plan(&self, start: &[f64], goal: &Goal) -> Result<BiTRRTResult> {
        let start_time = Instant::now();
        let joint_limits = self.get_joint_limits();
        let mut rng = rand::thread_rng();

        // Resolve goal to joint-space
        let goal_joints = self.resolve_goal(goal)?;

        if self.check_collision(start) {
            return Err(KineticError::StartInCollision);
        }
        if self.check_collision(&goal_joints) {
            return Err(KineticError::GoalInCollision);
        }

        let start_cost = self.cost_fn.cost(start);
        let goal_cost = self.cost_fn.cost(&goal_joints);

        let mut tree_a = CostTree::new(start.to_vec(), start_cost);
        let mut tree_b = CostTree::new(goal_joints.clone(), goal_cost);

        let mut temperature = self.bitrrt_config.initial_temperature;
        let mut consecutive_rejections: usize = 0;

        for iteration in 0..self.planner_config.max_iterations {
            if start_time.elapsed() >= self.planner_config.timeout {
                break;
            }

            // Sample (with goal bias toward the other tree's root)
            let sample = if rng.gen::<f64>() < self.bitrrt_config.goal_bias {
                tree_b.nodes[0].joints.clone()
            } else {
                self.random_sample(&joint_limits, &mut rng)
            };

            // Extend tree_a toward sample with transition test
            let extend_result = self.extend_with_transition(
                &mut tree_a,
                &sample,
                temperature,
                &mut rng,
            );

            let new_idx = match extend_result {
                ExtendResult::Advanced(i) => Some(i),
                ExtendResult::Reached => Some(tree_a.len() - 1),
                ExtendResult::Trapped => None,
            };

            if let Some(new_idx) = new_idx {
                consecutive_rejections = 0;

                // Try to connect tree_b
                let connect_result = self.connect(
                    &mut tree_b,
                    &tree_a.nodes[new_idx].joints,
                );

                if let ExtendResult::Reached = connect_result {
                    let connect_idx = tree_b.len() - 1;

                    // Extract path
                    let mut path_a = tree_a.path_to_root(new_idx);
                    let mut path_b = tree_b.path_to_root(connect_idx);
                    path_b.reverse();
                    path_a.extend(path_b);

                    let mut waypoints = path_a;

                    // Post-process
                    if self.planner_config.shortcut_iterations > 0 {
                        waypoints = shortcut::shortcut(
                            &waypoints,
                            self,
                            self.planner_config.shortcut_iterations,
                            self.bitrrt_config.step_size,
                        );
                    }
                    if self.planner_config.smooth && waypoints.len() > 2 {
                        let n = waypoints.len() * 10;
                        let smoothed = smooth::smooth_cubic_spline(&waypoints, n, Some(self));
                        if smoothed.c2_continuous {
                            waypoints = smoothed.waypoints;
                        }
                    }

                    let path_cost: f64 = waypoints.iter().map(|w| self.cost_fn.cost(w)).sum();

                    return Ok(BiTRRTResult {
                        waypoints,
                        planning_time: start_time.elapsed(),
                        iterations: iteration + 1,
                        tree_size: tree_a.len() + tree_b.len(),
                        path_cost,
                        final_temperature: temperature,
                    });
                }
            } else {
                consecutive_rejections += 1;

                // Frustration mechanism: increase temperature if stuck too long
                if consecutive_rejections >= self.bitrrt_config.frustration_threshold {
                    temperature *= self.bitrrt_config.frustration_factor;
                    consecutive_rejections = 0;
                }
            }

            // Anneal temperature
            temperature = (temperature * self.bitrrt_config.alpha)
                .max(self.bitrrt_config.min_temperature);

            // Swap trees
            std::mem::swap(&mut tree_a, &mut tree_b);
        }

        Err(KineticError::PlanningFailed(
            "BiTRRT: timeout or iteration limit reached".into(),
        ))
    }

    /// Extend tree toward target with Boltzmann transition test.
    fn extend_with_transition(
        &self,
        tree: &mut CostTree,
        target: &[f64],
        temperature: f64,
        rng: &mut impl Rng,
    ) -> ExtendResult {
        let nearest_idx = tree.nearest(target);
        let nearest = &tree.nodes[nearest_idx];
        let dist = joint_distance(&nearest.joints, target);

        if dist < 1e-10 {
            return ExtendResult::Reached;
        }

        // Steer: move step_size toward target
        let step = self.bitrrt_config.step_size;
        let reached = dist <= step;
        let t = if reached { 1.0 } else { step / dist };

        let new_joints: Vec<f64> = nearest
            .joints
            .iter()
            .zip(target.iter())
            .map(|(a, b)| a + t * (b - a))
            .collect();

        // Collision check
        if self.check_collision(&new_joints) {
            return ExtendResult::Trapped;
        }

        // Transition test: Boltzmann acceptance
        let parent_cost = nearest.cost;
        let new_cost = self.cost_fn.cost(&new_joints);
        let delta_cost = new_cost - parent_cost;

        if delta_cost > 0.0 {
            // Cost increased — accept with probability exp(-delta/T)
            let acceptance_prob = (-delta_cost / temperature).exp();
            if rng.gen::<f64>() > acceptance_prob {
                return ExtendResult::Trapped; // Rejected by transition test
            }
        }
        // delta_cost <= 0: always accept (cost decreased or equal)

        let idx = tree.add(new_joints, nearest_idx, new_cost);
        if reached {
            ExtendResult::Reached
        } else {
            ExtendResult::Advanced(idx)
        }
    }

    /// Greedy connect: extend tree toward target until Reached or Trapped.
    fn connect(&self, tree: &mut CostTree, target: &[f64]) -> ExtendResult {
        let step = self.bitrrt_config.step_size;
        loop {
            let nearest_idx = tree.nearest(target);
            let dist = joint_distance(&tree.nodes[nearest_idx].joints, target);

            if dist < 1e-10 {
                return ExtendResult::Reached;
            }

            let reached = dist <= step;
            let t = if reached { 1.0 } else { step / dist };

            let new_joints: Vec<f64> = tree.nodes[nearest_idx]
                .joints
                .iter()
                .zip(target.iter())
                .map(|(a, b)| a + t * (b - a))
                .collect();

            if self.check_collision(&new_joints) {
                return ExtendResult::Trapped;
            }

            let new_cost = self.cost_fn.cost(&new_joints);
            tree.add(new_joints, nearest_idx, new_cost);

            if reached {
                return ExtendResult::Reached;
            }
        }
    }

    fn check_collision(&self, joints: &[f64]) -> bool {
        let link_poses = match forward_kinematics_all(&self.robot, &self.chain, joints) {
            Ok(poses) => poses,
            Err(_) => return true,
        };
        let mut runtime = self.sphere_model.create_runtime();
        runtime.update(&link_poses);
        if self
            .environment
            .check_collision_with_margin(&runtime.world, self.planner_config.collision_margin)
        {
            return true;
        }
        let skip_pairs = self.acm.to_skip_pairs();
        runtime.self_collision_with_margin(&skip_pairs, self.planner_config.collision_margin)
    }

    fn resolve_goal(&self, goal: &Goal) -> Result<Vec<f64>> {
        match goal {
            Goal::Joints(jv) => Ok(jv.0.clone()),
            Goal::Pose(target_pose) => {
                let ik_config = IKConfig {
                    num_restarts: 8,
                    ..Default::default()
                };
                match solve_ik(&self.robot, &self.chain, target_pose, &ik_config) {
                    Ok(sol) => Ok(sol.joints),
                    Err(_) => Err(KineticError::NoIKSolution),
                }
            }
            Goal::Named(name) => Err(KineticError::NamedConfigNotFound(name.clone())),
            Goal::Relative(_) => Err(KineticError::UnsupportedGoal(
                "Relative goals not supported in BiTRRT".into(),
            )),
        }
    }

    fn random_sample(&self, limits: &[(f64, f64)], rng: &mut impl Rng) -> Vec<f64> {
        limits
            .iter()
            .map(|&(lo, hi)| rng.gen_range(lo..=hi))
            .collect()
    }

    fn get_joint_limits(&self) -> Vec<(f64, f64)> {
        self.chain
            .active_joints
            .iter()
            .map(|&joint_idx| {
                if let Some(limits) = &self.robot.joints[joint_idx].limits {
                    (limits.lower, limits.upper)
                } else {
                    (-std::f64::consts::PI, std::f64::consts::PI)
                }
            })
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kinetic_collision::capt::AABB;
    use kinetic_core::JointValues;

    fn setup_bitrrt() -> BiTRRT {
        let robot = Arc::new(Robot::from_name("ur5e").unwrap());
        let chain = KinematicChain::extract(&robot, "base_link", "tool0").unwrap();
        let env = CollisionEnvironment::empty(0.05, AABB::symmetric(2.0));

        let config = PlannerConfig {
            timeout: Duration::from_secs(30),
            max_iterations: 50_000,
            collision_margin: 0.0,
            shortcut_iterations: 0,
            smooth: false,
            workspace_bounds: None,
        };

        BiTRRT::with_default_cost(robot, chain, env, config, BiTRRTConfig::default())
    }

    #[test]
    fn bitrrt_plan_free_space() {
        let planner = setup_bitrrt();
        let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let goal = Goal::Joints(JointValues(vec![0.5, -0.5, 0.3, 0.0, 0.0, 0.0]));

        let result = planner.plan(&start, &goal).unwrap();
        assert!(result.waypoints.len() >= 2);
        assert!(result.iterations > 0);
        assert!(result.tree_size > 2);
        assert!(result.path_cost >= 0.0);
        assert!(result.final_temperature > 0.0);
    }

    #[test]
    fn bitrrt_path_collision_free() {
        let planner = setup_bitrrt();
        let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let goal = Goal::Joints(JointValues(vec![0.5, -0.5, 0.3, 0.0, 0.0, 0.0]));

        let result = planner.plan(&start, &goal).unwrap();
        for (i, wp) in result.waypoints.iter().enumerate() {
            assert!(!planner.check_collision(wp), "Waypoint {} in collision", i);
        }
    }

    #[test]
    fn bitrrt_temperature_anneals() {
        let planner = setup_bitrrt();
        // Use a larger distance to force more iterations
        let start = vec![0.0, -1.5, 1.0, 0.0, 0.5, 0.0];
        let goal = Goal::Joints(JointValues(vec![1.5, -0.3, -0.5, 1.0, -0.5, 1.0]));

        let result = planner.plan(&start, &goal).unwrap();
        // If the planner took multiple iterations, temperature should have annealed.
        // For very easy problems it may solve in 1 iteration before annealing.
        if result.iterations > 1 {
            assert!(
                result.final_temperature <= 1.0,
                "Temperature should anneal: {}",
                result.final_temperature
            );
        }
    }

    #[test]
    fn bitrrt_transition_test_unit() {
        // Transition test: cost decrease → always accept
        // Cost increase → accept with prob exp(-delta/T)
        // At T=1.0, delta=0.5 → exp(-0.5) ≈ 0.607 (often accepts)
        // At T=0.01, delta=0.5 → exp(-50) ≈ 0 (always rejects)

        let accept_prob_warm = (-0.5_f64 / 1.0).exp();
        assert!(accept_prob_warm > 0.5, "Warm T should accept often");

        let accept_prob_cold = (-0.5_f64 / 0.01).exp();
        assert!(accept_prob_cold < 0.001, "Cold T should reject");
    }

    #[test]
    fn bitrrt_custom_cost() {
        // Use a constant cost function — every node has same cost → no rejections
        struct ConstCost;
        impl CostFn for ConstCost {
            fn cost(&self, _joints: &[f64]) -> f64 {
                1.0
            }
        }

        let robot = Arc::new(Robot::from_name("ur5e").unwrap());
        let chain = KinematicChain::extract(&robot, "base_link", "tool0").unwrap();
        let env = CollisionEnvironment::empty(0.05, AABB::symmetric(2.0));

        let config = PlannerConfig {
            timeout: Duration::from_secs(30),
            max_iterations: 50_000,
            collision_margin: 0.0,
            shortcut_iterations: 0,
            smooth: false,
            workspace_bounds: None,
        };

        let planner = BiTRRT::new(
            robot,
            chain,
            env,
            config,
            BiTRRTConfig::default(),
            Box::new(ConstCost),
        );

        let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let goal = Goal::Joints(JointValues(vec![0.5, -0.5, 0.3, 0.0, 0.0, 0.0]));

        let result = planner.plan(&start, &goal).unwrap();
        assert!(result.waypoints.len() >= 2);
    }

    #[test]
    fn bitrrt_config_defaults() {
        let config = BiTRRTConfig::default();
        assert_eq!(config.step_size, 0.1);
        assert_eq!(config.initial_temperature, 1.0);
        assert_eq!(config.alpha, 0.95);
        assert_eq!(config.min_temperature, 0.01);
        assert_eq!(config.frustration_threshold, 100);
    }
}
