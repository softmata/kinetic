//! RRT* planner with asymptotic optimality (Karaman & Frazzoli, 2011).
//!
//! Extends RRT with cost tracking and rewiring. After adding a new node,
//! checks if nearby nodes can be reached cheaper through the new node and
//! reparents them. Produces asymptotically optimal paths.
//!
//! # Algorithm
//!
//! 1. Initialize tree rooted at start.
//! 2. Each iteration:
//!    a. Sample random config (with goal bias).
//!    b. Find nearest node, steer toward sample.
//!    c. Find near nodes within `r_n = gamma * (log(n)/n)^(1/d)`.
//!    d. Choose best parent from near nodes (minimum cost-to-come).
//!    e. Add node with best parent.
//!    f. Rewire: for each near node, check if going through new node is cheaper.
//! 3. Return best path to goal once found (continues improving until timeout).

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

/// RRT*-specific configuration.
#[derive(Debug, Clone)]
pub struct RRTStarConfig {
    /// Step size for tree extension (default: 0.1 radians).
    pub step_size: f64,
    /// Goal bias: probability of sampling the goal (default: 0.05).
    pub goal_bias: f64,
    /// Goal tolerance: distance threshold to consider goal reached (default: 0.05 radians).
    pub goal_tolerance: f64,
    /// Gamma constant for near-neighbor radius: r = gamma * (log(n)/n)^(1/d).
    /// Higher values search more neighbors (better quality, slower).
    /// Default: 1.5 (good balance for 6-7 DOF).
    pub gamma: f64,
    /// Whether to continue improving after first solution (anytime behavior).
    /// Default: true.
    pub anytime: bool,
    /// Maximum improvement time after first solution found.
    /// Only used when `anytime` is true. Default: remaining timeout.
    pub improvement_timeout: Option<Duration>,
    /// Enable Informed RRT* ellipsoidal sampling after first solution.
    /// Restricts sampling to prolate hyperspheroid containing potentially better paths.
    /// Default: true.
    pub informed: bool,
}

impl Default for RRTStarConfig {
    fn default() -> Self {
        Self {
            step_size: 0.1,
            goal_bias: 0.05,
            goal_tolerance: 0.05,
            gamma: 1.5,
            anytime: true,
            improvement_timeout: None,
            informed: true,
        }
    }
}

/// Prolate hyperspheroid for Informed RRT* sampling.
///
/// Defined by two foci (start, goal) and a transverse diameter (best cost so far).
/// Samples uniformly from the interior of this ellipsoid in C-space.
struct InformedSampler {
    /// Center of the ellipsoid: midpoint of start and goal.
    center: Vec<f64>,
    /// Half the distance between foci (c_min / 2).
    c_min_half: f64,
    /// Rotation matrix columns (d-dimensional): axes of the ellipsoid.
    /// First column aligns with the start→goal direction.
    rotation: Vec<Vec<f64>>,
    /// Dimensionality.
    dof: usize,
}

impl InformedSampler {
    fn new(start: &[f64], goal: &[f64]) -> Self {
        let dof = start.len();
        let center: Vec<f64> = start.iter().zip(goal.iter()).map(|(a, b)| (a + b) / 2.0).collect();
        let c_min = joint_distance(start, goal);
        let c_min_half = c_min / 2.0;

        // Build rotation: first axis = start→goal direction
        let mut axis1: Vec<f64> = goal.iter().zip(start.iter()).map(|(g, s)| g - s).collect();
        let norm = axis1.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm > 1e-12 {
            for x in &mut axis1 {
                *x /= norm;
            }
        }

        // Build orthonormal basis via Gram-Schmidt
        let mut rotation = vec![axis1];
        for i in 0..dof {
            if rotation.len() >= dof {
                break;
            }
            // Start with unit vector e_i
            let mut v = vec![0.0; dof];
            v[i] = 1.0;

            // Orthogonalize against existing basis
            for basis in &rotation {
                let dot: f64 = v.iter().zip(basis.iter()).map(|(a, b)| a * b).sum();
                for (vj, bj) in v.iter_mut().zip(basis.iter()) {
                    *vj -= dot * bj;
                }
            }

            let v_norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
            if v_norm > 1e-10 {
                for x in &mut v {
                    *x /= v_norm;
                }
                rotation.push(v);
            }
        }

        Self {
            center,
            c_min_half,
            rotation,
            dof,
        }
    }

    /// Sample uniformly from the prolate hyperspheroid with transverse diameter `c_best`.
    fn sample(&self, c_best: f64, limits: &[(f64, f64)], rng: &mut impl Rng) -> Vec<f64> {
        let c_best_half = c_best / 2.0;

        // Semi-axes: first axis = c_best/2, others = sqrt(c_best^2 - c_min^2) / 2
        let r_transverse = if c_best_half > self.c_min_half {
            (c_best_half * c_best_half - self.c_min_half * self.c_min_half).sqrt()
        } else {
            1e-6 // degenerate: just sample along the line
        };

        // Sample from unit ball via: direction * radius^(1/d)
        let mut unit_ball = vec![0.0f64; self.dof];
        let mut norm_sq: f64 = 0.0;
        for x in &mut unit_ball {
            *x = rng.gen_range(-1.0f64..=1.0f64);
            norm_sq += *x * *x;
        }
        let norm = norm_sq.sqrt();
        let radius = rng.gen::<f64>().powf(1.0 / self.dof as f64);
        for x in &mut unit_ball {
            *x = *x / norm * radius;
        }

        // Scale by semi-axes
        let mut scaled = vec![0.0; self.dof];
        scaled[0] = unit_ball[0] * c_best_half;
        for i in 1..self.dof {
            scaled[i] = unit_ball[i] * r_transverse;
        }

        // Rotate and translate to world frame
        let mut result = self.center.clone();
        for i in 0..self.dof {
            for j in 0..self.dof {
                if j < self.rotation.len() {
                    result[i] += self.rotation[j][i] * scaled[j];
                }
            }
        }

        // Clamp to joint limits
        for (i, &(lo, hi)) in limits.iter().enumerate() {
            result[i] = result[i].clamp(lo, hi);
        }

        result
    }
}

/// A node in the RRT* tree with cost tracking.
#[derive(Debug, Clone)]
struct RRTStarNode {
    /// Joint configuration.
    joints: Vec<f64>,
    /// Parent node index (None for root).
    parent: Option<usize>,
    /// Cost-to-come from root.
    cost: f64,
}

/// The RRT* tree with rewiring support.
#[derive(Debug)]
struct RRTStarTree {
    nodes: Vec<RRTStarNode>,
    dof: usize,
}

impl RRTStarTree {
    fn new(root: Vec<f64>) -> Self {
        let dof = root.len();
        Self {
            nodes: vec![RRTStarNode {
                joints: root,
                parent: None,
                cost: 0.0,
            }],
            dof,
        }
    }

    fn len(&self) -> usize {
        self.nodes.len()
    }

    fn add(&mut self, joints: Vec<f64>, parent: usize, cost: f64) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(RRTStarNode {
            joints,
            parent: Some(parent),
            cost,
        });
        idx
    }

    /// Brute-force nearest neighbor (uses squared distance to avoid sqrt).
    fn nearest(&self, target: &[f64]) -> usize {
        let mut best_idx = 0;
        let mut best_dist_sq = f64::INFINITY;

        for (i, node) in self.nodes.iter().enumerate() {
            let dist_sq = joint_distance_sq(&node.joints, target);
            if dist_sq < best_dist_sq {
                best_dist_sq = dist_sq;
                best_idx = i;
            }
        }

        best_idx
    }

    /// Find all nodes within radius of target (uses squared distance).
    fn near(&self, target: &[f64], radius: f64) -> Vec<usize> {
        let radius_sq = radius * radius;
        self.nodes
            .iter()
            .enumerate()
            .filter(|(_, node)| joint_distance_sq(&node.joints, target) <= radius_sq)
            .map(|(i, _)| i)
            .collect()
    }

    /// Compute the near-neighbor radius: gamma * (log(n)/n)^(1/d).
    fn near_radius(&self, gamma: f64) -> f64 {
        let n = self.len() as f64;
        let d = self.dof as f64;
        if n <= 1.0 {
            return f64::INFINITY;
        }
        gamma * (n.ln() / n).powf(1.0 / d)
    }

    /// Rewire: change parent of `node_idx` to `new_parent` with `new_cost`.
    fn rewire(&mut self, node_idx: usize, new_parent: usize, new_cost: f64) {
        let cost_delta = new_cost - self.nodes[node_idx].cost;
        self.nodes[node_idx].parent = Some(new_parent);
        self.nodes[node_idx].cost = new_cost;

        // Propagate cost change to all descendants
        self.propagate_cost_change(node_idx, cost_delta);
    }

    /// Propagate a cost change to all descendants of a node.
    fn propagate_cost_change(&mut self, parent_idx: usize, delta: f64) {
        // Collect children first to avoid borrow issues
        let children: Vec<usize> = self
            .nodes
            .iter()
            .enumerate()
            .filter(|(_, n)| n.parent == Some(parent_idx))
            .map(|(i, _)| i)
            .collect();

        for child_idx in children {
            self.nodes[child_idx].cost += delta;
            self.propagate_cost_change(child_idx, delta);
        }
    }

    /// Extract path from root to a given node index.
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

/// Squared Euclidean distance in joint space (no sqrt — faster for comparisons).
fn joint_distance_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
}

/// Euclidean distance in joint space.
fn joint_distance(a: &[f64], b: &[f64]) -> f64 {
    joint_distance_sq(a, b).sqrt()
}

/// Planning result from RRT*.
#[derive(Debug, Clone)]
pub struct RRTStarResult {
    /// Waypoints (joint configurations) along the path.
    pub waypoints: Vec<Vec<f64>>,
    /// Total planning time.
    pub planning_time: Duration,
    /// Number of RRT* iterations.
    pub iterations: usize,
    /// Total nodes in tree.
    pub tree_size: usize,
    /// Cost of the solution path (joint-space Euclidean length).
    pub path_cost: f64,
    /// Number of rewiring operations performed.
    pub rewire_count: usize,
    /// Whether the solution was improved after initial find (anytime).
    pub improved: bool,
}

/// RRT* planner with asymptotic optimality via rewiring.
///
/// Uses the same SIMD sphere-tree collision checking as RRT-Connect.
/// Produces better paths at the cost of more computation time.
pub struct RRTStar {
    robot: Arc<Robot>,
    chain: KinematicChain,
    sphere_model: RobotSphereModel,
    acm: ResolvedACM,
    environment: CollisionEnvironment,
    planner_config: PlannerConfig,
    rrt_star_config: RRTStarConfig,
}

impl CollisionChecker for RRTStar {
    fn is_in_collision(&self, joints: &[f64]) -> bool {
        self.is_in_collision(joints)
    }
}

impl RRTStar {
    /// Create a new RRT* planner.
    pub fn new(
        robot: Arc<Robot>,
        chain: KinematicChain,
        environment: CollisionEnvironment,
        planner_config: PlannerConfig,
        rrt_star_config: RRTStarConfig,
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
            rrt_star_config,
        }
    }

    /// Create with default configs.
    pub fn from_robot(
        robot: Arc<Robot>,
        chain: KinematicChain,
        environment: CollisionEnvironment,
    ) -> Self {
        Self::new(
            robot,
            chain,
            environment,
            PlannerConfig::default(),
            RRTStarConfig::default(),
        )
    }

    /// Set the ACM for self-collision filtering.
    pub fn with_acm(mut self, acm: &AllowedCollisionMatrix) -> Self {
        self.acm = ResolvedACM::from_acm(acm, &self.robot);
        self
    }

    /// Plan an asymptotically optimal path from start to goal.
    ///
    /// Pipeline: RRT* search → shortcutting → optional smoothing.
    pub fn plan(&self, start: &[f64], goal: &Goal) -> Result<RRTStarResult> {
        let start_time = Instant::now();

        let goal_configs = self.resolve_goal(goal)?;
        if goal_configs.is_empty() {
            return Err(KineticError::GoalUnreachable);
        }

        if self.is_in_collision(start) {
            return Err(KineticError::StartInCollision);
        }

        let mut last_err = KineticError::GoalUnreachable;

        for goal_joints in &goal_configs {
            if self.is_in_collision(goal_joints) {
                continue;
            }

            match self.plan_to_joints(start, goal_joints, start_time) {
                Ok(mut result) => {
                    // Post-process: shortcutting
                    if self.planner_config.shortcut_iterations > 0 {
                        result.waypoints = shortcut::shortcut(
                            &result.waypoints,
                            self,
                            self.planner_config.shortcut_iterations,
                            self.rrt_star_config.step_size,
                        );
                    }

                    // Post-process: smoothing
                    if self.planner_config.smooth && result.waypoints.len() > 2 {
                        let num_output = result.waypoints.len() * 10;
                        let smoothed =
                            smooth::smooth_cubic_spline(&result.waypoints, num_output, Some(self));
                        if smoothed.c2_continuous {
                            result.waypoints = smoothed.waypoints;
                        }
                    }

                    result.planning_time = start_time.elapsed();
                    return Ok(result);
                }
                Err(e) => last_err = e,
            }
        }

        Err(last_err)
    }

    /// Core RRT* planning loop.
    fn plan_to_joints(
        &self,
        start: &[f64],
        goal: &[f64],
        start_time: Instant,
    ) -> Result<RRTStarResult> {
        let mut tree = RRTStarTree::new(start.to_vec());
        let mut rng = rand::thread_rng();
        let joint_limits = self.get_joint_limits();

        let mut best_goal_node: Option<usize> = None;
        let mut best_goal_cost = f64::INFINITY;
        let mut rewire_count: usize = 0;
        let mut first_solution_time: Option<Instant> = None;

        // Informed RRT* sampler (created once, used after first solution)
        let informed_sampler = if self.rrt_star_config.informed {
            Some(InformedSampler::new(start, goal))
        } else {
            None
        };

        for _iteration in 0..self.planner_config.max_iterations {
            // Check timeout
            let elapsed = start_time.elapsed();
            if elapsed > self.planner_config.timeout {
                break;
            }

            // If we have a solution and anytime is enabled, check improvement timeout
            if let Some(first_time) = first_solution_time {
                if !self.rrt_star_config.anytime {
                    break; // Return first solution immediately
                }
                if let Some(imp_timeout) = self.rrt_star_config.improvement_timeout {
                    if first_time.elapsed() > imp_timeout {
                        break;
                    }
                }
            }

            // Sample: goal bias → informed ellipsoidal → uniform random
            let sample = if rng.gen::<f64>() < self.rrt_star_config.goal_bias {
                goal.to_vec()
            } else if let (Some(sampler), true) = (&informed_sampler, best_goal_cost.is_finite()) {
                // Informed RRT*: sample from ellipsoid containing better paths
                sampler.sample(best_goal_cost, &joint_limits, &mut rng)
            } else {
                self.random_sample(&joint_limits, &mut rng)
            };

            // Find nearest node
            let nearest_idx = tree.nearest(&sample);
            let nearest = &tree.nodes[nearest_idx].joints;

            // Steer: compute new node toward sample (limited by step_size)
            let new_joints = self.steer(nearest, &sample);

            // Collision check for new node
            if self.is_in_collision(&new_joints) {
                continue;
            }

            // Find near nodes within RRT* radius
            let radius = tree
                .near_radius(self.rrt_star_config.gamma)
                .min(self.rrt_star_config.step_size * 3.0); // cap radius
            let near_indices = tree.near(&new_joints, radius);

            // Choose best parent from near nodes
            let (best_parent, best_cost) =
                self.choose_best_parent(&tree, &near_indices, nearest_idx, &new_joints);

            // Check edge from best_parent to new node is collision-free
            if !self.is_edge_collision_free(
                &tree.nodes[best_parent].joints,
                &new_joints,
            ) {
                continue;
            }

            // Add node
            let new_idx = tree.add(new_joints.clone(), best_parent, best_cost);

            // Rewire near nodes through new node
            for &near_idx in &near_indices {
                if near_idx == best_parent {
                    continue;
                }

                let cost_through_new =
                    best_cost + joint_distance(&new_joints, &tree.nodes[near_idx].joints);

                if cost_through_new < tree.nodes[near_idx].cost {
                    // Check edge is collision-free
                    if self.is_edge_collision_free(&new_joints, &tree.nodes[near_idx].joints) {
                        tree.rewire(near_idx, new_idx, cost_through_new);
                        rewire_count += 1;
                    }
                }
            }

            // Check if new node is close to goal
            let dist_to_goal = joint_distance(&new_joints, goal);
            if dist_to_goal <= self.rrt_star_config.goal_tolerance {
                let goal_cost = best_cost + dist_to_goal;
                if goal_cost < best_goal_cost {
                    best_goal_cost = goal_cost;
                    best_goal_node = Some(new_idx);

                    if first_solution_time.is_none() {
                        first_solution_time = Some(Instant::now());
                    }
                }
            }
        }

        // Extract best path
        match best_goal_node {
            Some(goal_idx) => {
                let mut waypoints = tree.path_to_root(goal_idx);

                // Append exact goal if not already close enough
                if let Some(last) = waypoints.last() {
                    if joint_distance(last, goal) > 1e-8 {
                        waypoints.push(goal.to_vec());
                    }
                }

                Ok(RRTStarResult {
                    waypoints,
                    planning_time: start_time.elapsed(),
                    iterations: tree.len(),
                    tree_size: tree.len(),
                    path_cost: best_goal_cost,
                    rewire_count,
                    improved: first_solution_time.is_some()
                        && best_goal_node != first_solution_time.map(|_| best_goal_node).flatten(),
                })
            }
            None => Err(KineticError::PlanningTimeout {
                elapsed: start_time.elapsed(),
                iterations: tree.len(),
            }),
        }
    }

    /// Steer from `from` toward `to`, limited by step_size.
    fn steer(&self, from: &[f64], to: &[f64]) -> Vec<f64> {
        let dist = joint_distance(from, to);
        if dist <= self.rrt_star_config.step_size {
            return to.to_vec();
        }

        let ratio = self.rrt_star_config.step_size / dist;
        from.iter()
            .zip(to.iter())
            .map(|(a, b)| a + ratio * (b - a))
            .collect()
    }

    /// Choose best parent from near nodes (minimum cost-to-come).
    fn choose_best_parent(
        &self,
        tree: &RRTStarTree,
        near_indices: &[usize],
        default_parent: usize,
        new_joints: &[f64],
    ) -> (usize, f64) {
        let mut best_parent = default_parent;
        let mut best_cost =
            tree.nodes[default_parent].cost + joint_distance(&tree.nodes[default_parent].joints, new_joints);

        for &near_idx in near_indices {
            let cost = tree.nodes[near_idx].cost
                + joint_distance(&tree.nodes[near_idx].joints, new_joints);

            if cost < best_cost
                && self.is_edge_collision_free(&tree.nodes[near_idx].joints, new_joints)
            {
                best_parent = near_idx;
                best_cost = cost;
            }
        }

        (best_parent, best_cost)
    }

    /// Check if a straight-line edge between two configs is collision-free.
    fn is_edge_collision_free(&self, from: &[f64], to: &[f64]) -> bool {
        let dist = joint_distance(from, to);
        let step = self.rrt_star_config.step_size;
        let num_checks = (dist / step).ceil() as usize;

        if num_checks == 0 {
            return true;
        }

        for i in 1..=num_checks {
            let t = i as f64 / num_checks as f64;
            let interp: Vec<f64> = from
                .iter()
                .zip(to.iter())
                .map(|(a, b)| a + t * (b - a))
                .collect();

            if self.is_in_collision(&interp) {
                return false;
            }
        }

        true
    }

    /// Check if a configuration is in collision.
    fn is_in_collision(&self, joints: &[f64]) -> bool {
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

    /// Resolve a Goal to joint configurations.
    fn resolve_goal(&self, goal: &Goal) -> Result<Vec<Vec<f64>>> {
        match goal {
            Goal::Joints(jv) => Ok(vec![jv.0.clone()]),
            Goal::Pose(target_pose) => {
                let ik_config = IKConfig {
                    num_restarts: 8,
                    ..Default::default()
                };
                match solve_ik(&self.robot, &self.chain, target_pose, &ik_config) {
                    Ok(sol) => Ok(vec![sol.joints]),
                    Err(_) => Err(KineticError::NoIKSolution),
                }
            }
            Goal::Named(name) => Err(KineticError::NamedConfigNotFound(name.clone())),
            Goal::Relative(_) => Err(KineticError::UnsupportedGoal(
                "Relative goals not supported in RRT* planner".into(),
            )),
        }
    }

    /// Sample a random configuration within joint limits.
    fn random_sample(&self, limits: &[(f64, f64)], rng: &mut impl Rng) -> Vec<f64> {
        limits
            .iter()
            .map(|&(lo, hi)| rng.gen_range(lo..=hi))
            .collect()
    }

    /// Get joint limits as (lower, upper) pairs.
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
    use kinetic_collision::SpheresSoA;
    use kinetic_core::JointValues;

    fn setup_rrt_star_free_space() -> RRTStar {
        let robot = Arc::new(Robot::from_name("ur5e").unwrap());
        let chain = KinematicChain::extract(&robot, "base_link", "tool0").unwrap();
        let env = CollisionEnvironment::empty(0.05, AABB::symmetric(2.0));

        let config = PlannerConfig {
            timeout: Duration::from_secs(5),
            max_iterations: 50_000,
            collision_margin: 0.0,
            shortcut_iterations: 0,
            smooth: false,
            workspace_bounds: None,
        };

        RRTStar::new(
            robot,
            chain,
            env,
            config,
            RRTStarConfig {
                anytime: false, // Return first solution for faster tests
                ..Default::default()
            },
        )
    }

    #[test]
    fn rrt_star_free_space_joint_goal() {
        let planner = setup_rrt_star_free_space();

        let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let goal_joints = vec![0.5, -0.5, 0.3, 0.1, -0.2, 0.4];
        let goal = Goal::Joints(JointValues(goal_joints.clone()));

        let result = planner.plan(&start, &goal).unwrap();

        assert!(!result.waypoints.is_empty(), "Path should have waypoints");
        assert!(result.waypoints.len() >= 2, "Path needs start and goal");
        assert!(result.path_cost > 0.0, "Path cost should be positive");
        assert!(result.tree_size > 0, "Tree should have nodes");

        // First waypoint should be start
        let first = &result.waypoints[0];
        for (a, b) in first.iter().zip(start.iter()) {
            assert!((a - b).abs() < 1e-6, "First waypoint should match start");
        }

        // Last waypoint should be near goal
        let last = result.waypoints.last().unwrap();
        let dist = joint_distance(last, &goal_joints);
        assert!(dist < 0.2, "Last waypoint dist to goal: {}", dist);
    }

    #[test]
    fn rrt_star_start_in_collision() {
        let robot = Arc::new(Robot::from_name("ur5e").unwrap());
        let chain = KinematicChain::extract(&robot, "base_link", "tool0").unwrap();

        let mut obstacles = SpheresSoA::new();
        obstacles.push(0.0, 0.0, 0.0, 5.0, 0);
        let env = CollisionEnvironment::build(obstacles, 0.05, AABB::symmetric(10.0));

        let planner = RRTStar::new(
            robot,
            chain,
            env,
            PlannerConfig::default(),
            RRTStarConfig::default(),
        );

        let start = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let goal = Goal::Joints(JointValues(vec![1.0, -1.0, 0.5, 0.0, 0.0, 0.0]));

        match planner.plan(&start, &goal) {
            Err(KineticError::StartInCollision) => {}
            other => panic!("Expected StartInCollision, got {:?}", other),
        }
    }

    #[test]
    fn rrt_star_path_collision_free() {
        let planner = setup_rrt_star_free_space();

        let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let goal = Goal::Joints(JointValues(vec![0.5, -0.5, 0.3, 0.0, 0.0, 0.0]));

        let result = planner.plan(&start, &goal).unwrap();

        for (i, waypoint) in result.waypoints.iter().enumerate() {
            assert!(
                !planner.is_in_collision(waypoint),
                "Waypoint {} is in collision",
                i
            );
        }
    }

    #[test]
    fn rrt_star_rewiring_improves_cost() {
        let robot = Arc::new(Robot::from_name("ur5e").unwrap());
        let chain = KinematicChain::extract(&robot, "base_link", "tool0").unwrap();
        let env = CollisionEnvironment::empty(0.05, AABB::symmetric(2.0));

        let config = PlannerConfig {
            timeout: Duration::from_secs(5),
            max_iterations: 50_000,
            collision_margin: 0.0,
            shortcut_iterations: 0,
            smooth: false,
            workspace_bounds: None,
        };

        // With anytime=true and generous timeout, should find and improve
        let planner = RRTStar::new(
            robot,
            chain,
            env,
            config,
            RRTStarConfig {
                anytime: true,
                improvement_timeout: Some(Duration::from_millis(200)),
                ..Default::default()
            },
        );

        let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let goal = Goal::Joints(JointValues(vec![0.5, -0.5, 0.3, 0.0, 0.0, 0.0]));

        let result = planner.plan(&start, &goal).unwrap();

        // Should have performed some rewiring
        // (not guaranteed in every run, but very likely with 200ms improvement window)
        assert!(result.tree_size > 2, "Tree should have grown");
    }

    #[test]
    fn rrt_star_anytime_off_returns_first() {
        let planner = setup_rrt_star_free_space();

        let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let goal = Goal::Joints(JointValues(vec![0.5, -0.5, 0.3, 0.0, 0.0, 0.0]));

        let result = planner.plan(&start, &goal).unwrap();
        // With anytime=false, should return quickly after first solution
        assert!(result.planning_time < Duration::from_secs(3));
    }

    #[test]
    fn rrt_star_near_radius_decreases_with_tree_size() {
        let tree = RRTStarTree::new(vec![0.0; 6]);
        let gamma = 1.5;

        let r1 = tree.near_radius(gamma);
        // r at n=1 should be very large (infinity)
        assert!(r1 > 100.0, "Radius at n=1 should be large");
    }

    #[test]
    fn rrt_star_steer_within_step() {
        let planner = setup_rrt_star_free_space();
        let from = vec![0.0; 6];
        let to = vec![0.01; 6]; // very close

        let result = planner.steer(&from, &to);
        // Should reach exactly since dist < step_size
        for (a, b) in result.iter().zip(to.iter()) {
            assert!((a - b).abs() < 1e-10);
        }
    }

    #[test]
    fn rrt_star_steer_beyond_step() {
        let planner = setup_rrt_star_free_space();
        let from = vec![0.0; 6];
        let to = vec![1.0; 6]; // far away

        let result = planner.steer(&from, &to);
        let dist = joint_distance(&from, &result);
        assert!(
            (dist - planner.rrt_star_config.step_size).abs() < 1e-10,
            "Steered distance should equal step_size"
        );
    }

    #[test]
    fn informed_sampler_samples_within_ellipsoid() {
        let start = vec![0.0; 6];
        let goal = vec![2.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let sampler = InformedSampler::new(&start, &goal);

        let c_best = 3.0; // transverse diameter > c_min (2.0)
        let limits: Vec<(f64, f64)> = vec![(-10.0, 10.0); 6];
        let mut rng = rand::thread_rng();

        for _ in 0..100 {
            let s = sampler.sample(c_best, &limits, &mut rng);
            // All samples should satisfy the ellipsoid constraint:
            // dist(s, start) + dist(s, goal) <= c_best (within tolerance for clamping)
            let d_start = joint_distance(&s, &start);
            let d_goal = joint_distance(&s, &goal);
            // Allow some slack for clamping to joint limits
            assert!(
                d_start + d_goal <= c_best + 0.5,
                "Sample outside ellipsoid: d_start={}, d_goal={}, sum={}, c_best={}",
                d_start,
                d_goal,
                d_start + d_goal,
                c_best
            );
        }
    }

    #[test]
    fn informed_rrt_star_uses_ellipsoidal_sampling() {
        let robot = Arc::new(Robot::from_name("ur5e").unwrap());
        let chain = KinematicChain::extract(&robot, "base_link", "tool0").unwrap();
        let env = CollisionEnvironment::empty(0.05, AABB::symmetric(2.0));

        let config = PlannerConfig {
            timeout: Duration::from_secs(5),
            max_iterations: 50_000,
            collision_margin: 0.0,
            shortcut_iterations: 0,
            smooth: false,
            workspace_bounds: None,
        };

        let planner = RRTStar::new(
            robot,
            chain,
            env,
            config,
            RRTStarConfig {
                anytime: true,
                informed: true,
                improvement_timeout: Some(Duration::from_millis(100)),
                ..Default::default()
            },
        );

        let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let goal = Goal::Joints(JointValues(vec![0.5, -0.5, 0.3, 0.0, 0.0, 0.0]));

        let result = planner.plan(&start, &goal).unwrap();
        assert!(result.waypoints.len() >= 2);
        assert!(result.path_cost > 0.0);
    }
}
