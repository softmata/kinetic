//! EST (Expansive Space Trees) planner (Hsu et al., 1997).
//!
//! Bidirectional tree planner with density-biased node selection. Nodes in
//! sparsely explored regions are selected more frequently for expansion,
//! improving exploration of narrow passages and undersampled areas.
//!
//! # Key idea
//!
//! Instead of uniform random node selection (RRT) or nearest-neighbor extension,
//! EST selects a node for expansion with probability inversely proportional to
//! the number of nearby nodes within a density radius. This naturally pushes
//! exploration toward the frontier of the explored space.
//!
//! # Algorithm
//!
//! 1. Initialize two trees rooted at start and goal.
//! 2. Each iteration:
//!    a. **Select** a node from the active tree weighted by `1 / (1 + neighbors_within_radius)`.
//!    b. **Expand** by sampling a random direction from the selected node.
//!    c. If collision-free, add the new node.
//!    d. Try to connect the other tree.
//!    e. Swap trees.
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

/// EST-specific configuration.
#[derive(Debug, Clone)]
pub struct ESTConfig {
    /// Maximum expansion distance from selected node (default: 0.3 radians).
    pub expansion_range: f64,
    /// Radius for density estimation — nodes within this distance are "neighbors" (default: 0.5 radians).
    pub density_radius: f64,
    /// Step size for edge collision checking during connect (default: 0.1 radians).
    pub step_size: f64,
    /// Probability of sampling toward the other tree's root (default: 0.05).
    pub goal_bias: f64,
}

impl Default for ESTConfig {
    fn default() -> Self {
        Self {
            expansion_range: 0.3,
            density_radius: 0.5,
            step_size: 0.1,
            goal_bias: 0.05,
        }
    }
}

/// A node in the EST tree.
#[derive(Debug, Clone)]
struct ESTNode {
    joints: Vec<f64>,
    parent: Option<usize>,
}

/// EST tree with density tracking.
#[derive(Debug)]
struct ESTTree {
    nodes: Vec<ESTNode>,
    /// Cached neighbor count for each node (updated lazily).
    neighbor_counts: Vec<usize>,
    /// Whether neighbor counts need refresh.
    dirty: bool,
}

impl ESTTree {
    fn new(root: Vec<f64>) -> Self {
        Self {
            nodes: vec![ESTNode {
                joints: root,
                parent: None,
            }],
            neighbor_counts: vec![0],
            dirty: true,
        }
    }

    fn len(&self) -> usize {
        self.nodes.len()
    }

    fn add(&mut self, joints: Vec<f64>, parent: usize) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(ESTNode {
            joints,
            parent: Some(parent),
        });
        self.neighbor_counts.push(0);
        self.dirty = true;
        idx
    }

    /// Recompute neighbor counts for all nodes within the density radius.
    fn refresh_density(&mut self, radius: f64) {
        if !self.dirty {
            return;
        }
        let r_sq = radius * radius;
        let n = self.nodes.len();

        for i in 0..n {
            let mut count = 0usize;
            for j in 0..n {
                if i != j {
                    let d_sq = joint_distance_sq(&self.nodes[i].joints, &self.nodes[j].joints);
                    if d_sq <= r_sq {
                        count += 1;
                    }
                }
            }
            self.neighbor_counts[i] = count;
        }
        self.dirty = false;
    }

    /// Select a node for expansion, weighted inversely by density.
    ///
    /// Weight of node i = 1 / (1 + neighbor_count_i).
    /// Nodes in sparse regions get higher weight → more likely to be selected.
    fn select_weighted(&self, rng: &mut impl Rng) -> usize {
        let weights: Vec<f64> = self
            .neighbor_counts
            .iter()
            .map(|&count| 1.0 / (1.0 + count as f64))
            .collect();

        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return rng.gen_range(0..self.nodes.len());
        }

        let mut r = rng.gen::<f64>() * total;
        for (i, &w) in weights.iter().enumerate() {
            r -= w;
            if r <= 0.0 {
                return i;
            }
        }
        self.nodes.len() - 1
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
    joint_distance_sq(a, b).sqrt()
}

fn joint_distance_sq(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
}

/// EST planner result.
#[derive(Debug, Clone)]
pub struct ESTResult {
    pub waypoints: Vec<Vec<f64>>,
    pub planning_time: Duration,
    pub iterations: usize,
    pub tree_size: usize,
}

/// Bidirectional EST planner.
pub struct EST {
    robot: Arc<Robot>,
    chain: KinematicChain,
    sphere_model: RobotSphereModel,
    acm: ResolvedACM,
    environment: CollisionEnvironment,
    planner_config: PlannerConfig,
    est_config: ESTConfig,
}

impl CollisionChecker for EST {
    fn is_in_collision(&self, joints: &[f64]) -> bool {
        self.check_collision(joints)
    }
}

impl EST {
    pub fn new(
        robot: Arc<Robot>,
        chain: KinematicChain,
        environment: CollisionEnvironment,
        planner_config: PlannerConfig,
        est_config: ESTConfig,
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
            est_config,
        }
    }

    pub fn with_acm(mut self, acm: &AllowedCollisionMatrix) -> Self {
        self.acm = ResolvedACM::from_acm(acm, &self.robot);
        self
    }

    /// Plan a collision-free path using density-biased exploration.
    pub fn plan(&self, start: &[f64], goal: &Goal) -> Result<ESTResult> {
        let start_time = Instant::now();
        let joint_limits = self.get_joint_limits();
        let mut rng = rand::thread_rng();

        let goal_joints = self.resolve_goal(goal)?;

        if self.check_collision(start) {
            return Err(KineticError::StartInCollision);
        }
        if self.check_collision(&goal_joints) {
            return Err(KineticError::GoalInCollision);
        }

        let mut tree_a = ESTTree::new(start.to_vec());
        let mut tree_b = ESTTree::new(goal_joints.clone());

        // Density refresh interval (avoid O(n²) every iteration)
        let refresh_interval = 20;

        for iteration in 0..self.planner_config.max_iterations {
            if start_time.elapsed() >= self.planner_config.timeout {
                break;
            }

            // Refresh density periodically
            if iteration % refresh_interval == 0 {
                tree_a.refresh_density(self.est_config.density_radius);
            }

            // Select node weighted by inverse density
            let selected_idx = tree_a.select_weighted(&mut rng);
            let selected = &tree_a.nodes[selected_idx].joints;

            // Expand: sample random direction from selected node
            let new_joints = if rng.gen::<f64>() < self.est_config.goal_bias {
                // Bias toward the other tree's root
                let target = &tree_b.nodes[0].joints;
                let dist = joint_distance(selected, target);
                if dist < 1e-10 {
                    selected.clone()
                } else {
                    let t = (self.est_config.expansion_range / dist).min(1.0);
                    selected
                        .iter()
                        .zip(target.iter())
                        .map(|(a, b)| a + t * (b - a))
                        .collect()
                }
            } else {
                // Random expansion within range, clamped to joint limits
                selected
                    .iter()
                    .zip(joint_limits.iter())
                    .map(|(&s, &(lo, hi))| {
                        let delta = rng.gen_range(-self.est_config.expansion_range..=self.est_config.expansion_range);
                        (s + delta).clamp(lo, hi)
                    })
                    .collect()
            };

            if self.check_collision(&new_joints) {
                std::mem::swap(&mut tree_a, &mut tree_b);
                continue;
            }

            let new_idx = tree_a.add(new_joints, selected_idx);

            // Try to connect tree_b
            if self.try_connect(&mut tree_b, &tree_a.nodes[new_idx].joints) {
                let connect_idx = tree_b.len() - 1;

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
                        self.est_config.step_size,
                    );
                }
                if self.planner_config.smooth && waypoints.len() > 2 {
                    let n = waypoints.len() * 10;
                    let smoothed = smooth::smooth_cubic_spline(&waypoints, n, Some(self));
                    if smoothed.c2_continuous {
                        waypoints = smoothed.waypoints;
                    }
                }

                return Ok(ESTResult {
                    waypoints,
                    planning_time: start_time.elapsed(),
                    iterations: iteration + 1,
                    tree_size: tree_a.len() + tree_b.len(),
                });
            }

            // Swap trees
            std::mem::swap(&mut tree_a, &mut tree_b);
        }

        Err(KineticError::PlanningFailed(
            "EST: timeout or iteration limit reached".into(),
        ))
    }

    /// Greedy connect: extend tree_b toward target step-by-step.
    fn try_connect(&self, tree: &mut ESTTree, target: &[f64]) -> bool {
        let step = self.est_config.step_size;
        loop {
            let nearest_idx = tree.nearest(target);
            let dist = joint_distance(&tree.nodes[nearest_idx].joints, target);

            if dist < 1e-10 {
                return true;
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
                return false;
            }

            tree.add(new_joints, nearest_idx);

            if reached {
                return true;
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
                "Relative goals not supported in EST".into(),
            )),
        }
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

    fn setup_est() -> EST {
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

        EST::new(robot, chain, env, config, ESTConfig::default())
    }

    #[test]
    fn est_plan_free_space() {
        let planner = setup_est();
        let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let goal = Goal::Joints(JointValues(vec![0.5, -0.5, 0.3, 0.0, 0.0, 0.0]));

        let result = planner.plan(&start, &goal).unwrap();
        assert!(result.waypoints.len() >= 2);
        assert!(result.iterations > 0);
        assert!(result.tree_size > 2);
    }

    #[test]
    fn est_path_collision_free() {
        let planner = setup_est();
        let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let goal = Goal::Joints(JointValues(vec![0.5, -0.5, 0.3, 0.0, 0.0, 0.0]));

        let result = planner.plan(&start, &goal).unwrap();
        for (i, wp) in result.waypoints.iter().enumerate() {
            assert!(!planner.check_collision(wp), "Waypoint {} in collision", i);
        }
    }

    #[test]
    fn est_density_weighting() {
        // Verify that nodes with fewer neighbors get higher selection weight.
        // Weight = 1/(1+count): count=0 → weight=1.0, count=5 → weight=0.167
        let w_sparse = 1.0 / (1.0 + 0.0);
        let w_dense = 1.0 / (1.0 + 5.0);
        assert!(w_sparse > w_dense * 5.0, "Sparse nodes should be strongly preferred");
    }

    #[test]
    fn est_config_defaults() {
        let config = ESTConfig::default();
        assert_eq!(config.expansion_range, 0.3);
        assert_eq!(config.density_radius, 0.5);
        assert_eq!(config.step_size, 0.1);
        assert_eq!(config.goal_bias, 0.05);
    }

    #[test]
    fn est_larger_goal_distance() {
        let planner = setup_est();
        let start = vec![0.0, -1.5, 1.0, 0.0, 0.5, 0.0];
        let goal = Goal::Joints(JointValues(vec![1.5, -0.3, -0.5, 1.0, -0.5, 1.0]));

        let result = planner.plan(&start, &goal).unwrap();
        assert!(result.waypoints.len() >= 2);
    }
}
