//! Constrained RRT-Connect planner.
//!
//! Extends standard RRT-Connect with constraint projection (OMPL-style).
//! Each random sample and extension step is projected onto the constraint
//! manifold before being added to the tree.

use std::sync::Arc;
use std::time::Instant;

use rand::Rng;

use kinetic_collision::{CollisionEnvironment, ResolvedACM, RobotSphereModel, SphereGenConfig};
use kinetic_core::{Constraint, Goal, KineticError, PlannerConfig, Result};
use kinetic_kinematics::{forward_kinematics_all, solve_ik, IKConfig, KinematicChain};
use kinetic_robot::Robot;

use crate::constraint;
use crate::rrt::RRTConfig;
use crate::shortcut::CollisionChecker;
use crate::smooth;

/// Result of an extend operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExtendResult {
    Reached,
    Advanced,
    Trapped,
}

/// A node in the constrained RRT tree.
#[derive(Debug, Clone)]
struct TreeNode {
    joints: Vec<f64>,
    parent: Option<usize>,
}

/// A C-space tree.
#[derive(Debug)]
struct CSpaceTree {
    nodes: Vec<TreeNode>,
}

impl CSpaceTree {
    fn new(root: Vec<f64>) -> Self {
        Self {
            nodes: vec![TreeNode {
                joints: root,
                parent: None,
            }],
        }
    }

    fn len(&self) -> usize {
        self.nodes.len()
    }

    fn add(&mut self, joints: Vec<f64>, parent: usize) -> usize {
        let idx = self.nodes.len();
        self.nodes.push(TreeNode {
            joints,
            parent: Some(parent),
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

/// Constrained RRT-Connect planner.
///
/// Same algorithm as standard RRT-Connect, but every sample and extension
/// is projected onto the constraint manifold before acceptance.
pub struct ConstrainedRRT {
    robot: Arc<Robot>,
    chain: KinematicChain,
    sphere_model: RobotSphereModel,
    acm: ResolvedACM,
    environment: CollisionEnvironment,
    planner_config: PlannerConfig,
    rrt_config: RRTConfig,
    constraints: Vec<Constraint>,
    /// Max iterations for constraint projection.
    projection_iterations: usize,
}

/// Planning result from constrained RRT.
#[derive(Debug, Clone)]
pub struct ConstrainedPlanningResult {
    /// Waypoints (joint configurations) along the path.
    pub waypoints: Vec<Vec<f64>>,
    /// Total planning time.
    pub planning_time: std::time::Duration,
    /// Number of RRT iterations.
    pub iterations: usize,
    /// Total nodes in both trees.
    pub tree_size: usize,
    /// How many samples were rejected by constraint projection.
    pub projection_rejections: usize,
}

impl ConstrainedRRT {
    /// Create a new constrained RRT-Connect planner.
    pub fn new(
        robot: Arc<Robot>,
        chain: KinematicChain,
        environment: CollisionEnvironment,
        planner_config: PlannerConfig,
        rrt_config: RRTConfig,
        constraints: Vec<Constraint>,
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
            rrt_config,
            constraints,
            projection_iterations: 50,
        }
    }

    /// Set the maximum iterations for constraint projection (default: 50).
    pub fn with_projection_iterations(mut self, iterations: usize) -> Self {
        self.projection_iterations = iterations;
        self
    }

    /// Plan a constrained collision-free path from start to goal.
    pub fn plan(&self, start: &[f64], goal: &Goal) -> Result<ConstrainedPlanningResult> {
        let start_time = Instant::now();

        // Resolve goal to joint configurations
        let goal_configs = self.resolve_goal(goal)?;
        if goal_configs.is_empty() {
            return Err(KineticError::GoalUnreachable);
        }

        // Validate start
        if self.is_in_collision(start) {
            return Err(KineticError::StartInCollision);
        }
        if !constraint::all_satisfied(&self.constraints, &self.robot, &self.chain, start) {
            return Err(KineticError::PlanningFailed(
                "Start configuration violates constraints".into(),
            ));
        }

        let mut last_err = KineticError::GoalUnreachable;

        for goal_joints in &goal_configs {
            if self.is_in_collision(goal_joints) {
                continue;
            }
            // Project goal onto constraints
            let projected_goal = match constraint::project(
                &self.constraints,
                &self.robot,
                &self.chain,
                goal_joints,
                self.projection_iterations,
            ) {
                Some(g) => g,
                None => continue,
            };

            if self.is_in_collision(&projected_goal) {
                continue;
            }

            match self.plan_to_joints(start, &projected_goal, start_time) {
                Ok(mut result) => {
                    // Post-process: shortcutting (with constraint checking)
                    if self.planner_config.shortcut_iterations > 0 {
                        result.waypoints = self.constrained_shortcut(
                            &result.waypoints,
                            self.planner_config.shortcut_iterations,
                        );
                    }

                    // Post-process: smoothing
                    if self.planner_config.smooth && result.waypoints.len() > 2 {
                        let num_output = result.waypoints.len() * 10;
                        let smoothed =
                            smooth::smooth_cubic_spline(&result.waypoints, num_output, Some(self));
                        if smoothed.c2_continuous {
                            // Verify constraints on smoothed path
                            let all_valid = smoothed.waypoints.iter().all(|wp| {
                                constraint::all_satisfied(
                                    &self.constraints,
                                    &self.robot,
                                    &self.chain,
                                    wp,
                                )
                            });
                            if all_valid {
                                result.waypoints = smoothed.waypoints;
                            }
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

    /// Plan to a specific joint-space goal.
    fn plan_to_joints(
        &self,
        start: &[f64],
        goal: &[f64],
        start_time: Instant,
    ) -> Result<ConstrainedPlanningResult> {
        let mut tree_a = CSpaceTree::new(start.to_vec());
        let mut tree_b = CSpaceTree::new(goal.to_vec());
        let mut rng = rand::thread_rng();
        let joint_limits = self.get_joint_limits();
        let mut projection_rejections = 0;

        for iteration in 0..self.planner_config.max_iterations {
            let elapsed = start_time.elapsed();
            if elapsed > self.planner_config.timeout {
                return Err(KineticError::PlanningTimeout {
                    elapsed,
                    iterations: iteration,
                });
            }

            // Sample random configuration (with goal bias)
            let raw_sample = if rng.gen::<f64>() < self.rrt_config.goal_bias {
                goal.to_vec()
            } else {
                self.random_sample(&joint_limits, &mut rng)
            };

            // Project sample onto constraint manifold
            let sample = match constraint::project(
                &self.constraints,
                &self.robot,
                &self.chain,
                &raw_sample,
                self.projection_iterations,
            ) {
                Some(s) => s,
                None => {
                    projection_rejections += 1;
                    continue;
                }
            };

            // Extend tree_a toward projected sample
            let nearest_a = tree_a.nearest(&sample);
            let extend_result = self.extend(&mut tree_a, nearest_a, &sample);

            if extend_result != ExtendResult::Trapped {
                let new_node_a = tree_a.len() - 1;
                let target = &tree_a.nodes[new_node_a].joints;

                let nearest_b = tree_b.nearest(target);
                let connect_result = self.connect(&mut tree_b, nearest_b, target);

                if connect_result == ExtendResult::Reached {
                    let new_node_b = tree_b.len() - 1;
                    let path = self.extract_path(&tree_a, &tree_b, new_node_a, new_node_b);

                    return Ok(ConstrainedPlanningResult {
                        waypoints: path,
                        planning_time: start_time.elapsed(),
                        iterations: iteration + 1,
                        tree_size: tree_a.len() + tree_b.len(),
                        projection_rejections,
                    });
                }
            }

            std::mem::swap(&mut tree_a, &mut tree_b);
        }

        Err(KineticError::PlanningTimeout {
            elapsed: start_time.elapsed(),
            iterations: self.planner_config.max_iterations,
        })
    }

    /// Extend tree one step toward target, with constraint projection.
    fn extend(&self, tree: &mut CSpaceTree, nearest_idx: usize, target: &[f64]) -> ExtendResult {
        let nearest = &tree.nodes[nearest_idx].joints;
        let dist = joint_distance(nearest, target);

        if dist < 1e-10 {
            return ExtendResult::Reached;
        }

        let step = self.rrt_config.step_size;

        let new_joints = if dist <= step {
            target.to_vec()
        } else {
            let ratio = step / dist;
            nearest
                .iter()
                .zip(target.iter())
                .map(|(a, b)| a + ratio * (b - a))
                .collect()
        };

        // Project onto constraint manifold
        let projected = match constraint::project(
            &self.constraints,
            &self.robot,
            &self.chain,
            &new_joints,
            self.projection_iterations,
        ) {
            Some(p) => p,
            None => return ExtendResult::Trapped,
        };

        if self.is_in_collision(&projected) {
            return ExtendResult::Trapped;
        }

        let reached = joint_distance(&projected, target) < 1e-6;
        tree.add(projected, nearest_idx);

        if reached {
            ExtendResult::Reached
        } else {
            ExtendResult::Advanced
        }
    }

    /// Greedily connect tree toward target.
    fn connect(&self, tree: &mut CSpaceTree, nearest_idx: usize, target: &[f64]) -> ExtendResult {
        let mut current_idx = nearest_idx;
        loop {
            match self.extend(tree, current_idx, target) {
                ExtendResult::Reached => return ExtendResult::Reached,
                ExtendResult::Advanced => {
                    current_idx = tree.len() - 1;
                }
                ExtendResult::Trapped => return ExtendResult::Trapped,
            }
        }
    }

    /// Extract path from connected trees.
    fn extract_path(
        &self,
        tree_a: &CSpaceTree,
        tree_b: &CSpaceTree,
        connect_a: usize,
        connect_b: usize,
    ) -> Vec<Vec<f64>> {
        let path_a = tree_a.path_to_root(connect_a);
        let mut path_b = tree_b.path_to_root(connect_b);
        path_b.reverse();

        let mut full_path = path_a;
        if !full_path.is_empty()
            && !path_b.is_empty()
            && joint_distance(full_path.last().unwrap(), &path_b[0]) < 1e-8
        {
            path_b.remove(0);
        }
        full_path.extend(path_b);
        full_path
    }

    /// Check collision.
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

    /// Constrained shortcutting — only accept shortcuts that satisfy constraints.
    fn constrained_shortcut(&self, path: &[Vec<f64>], iterations: usize) -> Vec<Vec<f64>> {
        let mut result = path.to_vec();
        let mut rng = rand::thread_rng();

        for _ in 0..iterations {
            if result.len() <= 2 {
                break;
            }

            let i = rng.gen_range(0..result.len() - 1);
            let j = rng.gen_range(i + 1..result.len());

            // Check if direct path between i and j is valid
            let valid = self.check_constrained_edge(&result[i], &result[j]);
            if valid {
                // Remove intermediate waypoints
                let mut new_path = result[..=i].to_vec();
                new_path.extend_from_slice(&result[j..]);
                result = new_path;
            }
        }

        result
    }

    /// Check if an edge satisfies both collision and constraints.
    fn check_constrained_edge(&self, from: &[f64], to: &[f64]) -> bool {
        let dist = joint_distance(from, to);
        let steps = (dist / (self.rrt_config.step_size * 0.5)).ceil() as usize;
        let steps = steps.max(2);

        for s in 1..steps {
            let t = s as f64 / steps as f64;
            let interp: Vec<f64> = from
                .iter()
                .zip(to.iter())
                .map(|(a, b)| a + t * (b - a))
                .collect();

            if self.is_in_collision(&interp) {
                return false;
            }
            if !constraint::all_satisfied(&self.constraints, &self.robot, &self.chain, &interp) {
                return false;
            }
        }

        true
    }

    fn random_sample(&self, joint_limits: &[(f64, f64)], rng: &mut impl Rng) -> Vec<f64> {
        joint_limits
            .iter()
            .map(|(lo, hi)| rng.gen_range(*lo..=*hi))
            .collect()
    }

    fn get_joint_limits(&self) -> Vec<(f64, f64)> {
        self.chain
            .active_joints
            .iter()
            .map(|&ji| {
                self.robot.joints[ji]
                    .limits
                    .as_ref()
                    .map_or((-std::f64::consts::PI, std::f64::consts::PI), |l| {
                        (l.lower, l.upper)
                    })
            })
            .collect()
    }

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
                "Relative goals not supported in constrained RRT".into(),
            )),
        }
    }
}

impl CollisionChecker for ConstrainedRRT {
    fn is_in_collision(&self, joints: &[f64]) -> bool {
        self.is_in_collision(joints)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kinetic_collision::capt::AABB;
    use kinetic_core::JointValues;

    fn test_setup(constraints: Vec<Constraint>) -> ConstrainedRRT {
        // Use UR5e — no collision geometry means no self-collision issues
        let robot = Robot::from_name("ur5e").unwrap();
        let chain = KinematicChain::extract(&robot, "base_link", "tool0").unwrap();
        let env = CollisionEnvironment::empty(0.05, AABB::symmetric(10.0));

        ConstrainedRRT::new(
            Arc::new(robot),
            chain,
            env,
            PlannerConfig {
                max_iterations: 5000,
                shortcut_iterations: 0,
                smooth: false,
                ..PlannerConfig::default()
            },
            RRTConfig {
                step_size: 0.15,
                goal_bias: 0.1,
            },
            constraints,
        )
    }

    #[test]
    fn constrained_rrt_joint_constraint() {
        // Plan with joint 0 constrained to [-1.0, 1.0]
        let planner = test_setup(vec![Constraint::joint(0, -1.0, 1.0)]);

        let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let goal = Goal::Joints(JointValues(vec![0.5, -0.5, 0.3, 0.1, -0.2, 0.4]));

        let result = planner.plan(&start, &goal).unwrap();
        assert!(result.waypoints.len() >= 2);

        // Verify all waypoints satisfy the constraint
        for wp in &result.waypoints {
            assert!(
                wp[0] >= -1.0 - 1e-6 && wp[0] <= 1.0 + 1e-6,
                "Joint 0 out of constraint bounds: {}",
                wp[0]
            );
        }
    }

    #[test]
    fn constrained_rrt_start_violates() {
        let planner = test_setup(vec![Constraint::joint(0, -0.5, 0.5)]);
        let start = vec![1.0, -1.0, 0.8, 0.0, 0.0, 0.0]; // violates constraint
        let goal = Goal::Joints(JointValues(vec![0.3, -0.5, 0.3, 0.0, 0.0, 0.0]));

        let result = planner.plan(&start, &goal);
        assert!(
            result.is_err(),
            "Should fail when start violates constraints"
        );
    }

    #[test]
    fn constrained_rrt_no_constraints() {
        // With no constraints, should work like regular RRT
        let planner = test_setup(vec![]);

        let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let goal = Goal::Joints(JointValues(vec![0.5, -0.5, 0.3, 0.1, -0.2, 0.4]));

        let result = planner.plan(&start, &goal).unwrap();
        assert!(result.waypoints.len() >= 2);
    }

    #[test]
    fn constrained_rrt_multiple_joint_constraints() {
        let planner = test_setup(vec![
            Constraint::joint(0, -0.8, 0.8),
            Constraint::joint(1, -1.5, -0.3),
        ]);

        let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let goal = Goal::Joints(JointValues(vec![0.5, -0.5, 0.3, 0.1, -0.2, 0.4]));

        let result = planner.plan(&start, &goal).unwrap();

        for wp in &result.waypoints {
            assert!(
                wp[0] >= -0.8 - 1e-6 && wp[0] <= 0.8 + 1e-6,
                "Joint 0 out of bounds: {}",
                wp[0]
            );
            assert!(
                wp[1] >= -1.5 - 1e-6 && wp[1] <= -0.3 + 1e-6,
                "Joint 1 out of bounds: {}",
                wp[1]
            );
        }
    }

    #[test]
    fn constrained_rrt_position_bound() {
        // Constrain EE to stay above z = -0.5 (wide enough for UR5e workspace)
        let planner = test_setup(vec![Constraint::position_bound(
            "tool0",
            kinetic_core::Axis::Z,
            -0.5,
            1.5,
        )]);

        let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let goal = Goal::Joints(JointValues(vec![0.3, -0.8, 0.5, 0.0, 0.0, 0.0]));

        let result = planner.plan(&start, &goal).unwrap();
        assert!(result.waypoints.len() >= 2);
    }

    // ─── Additional coverage tests ───

    #[test]
    fn joint_distance_identical() {
        let a = vec![1.0, 2.0, 3.0];
        assert!((joint_distance(&a, &a)).abs() < 1e-10);
    }

    #[test]
    fn joint_distance_unit_vectors() {
        let a = vec![0.0, 0.0, 0.0];
        let b = vec![3.0, 4.0, 0.0];
        assert!((joint_distance(&a, &b) - 5.0).abs() < 1e-10);
    }

    #[test]
    fn joint_distance_empty() {
        let empty: Vec<f64> = vec![];
        assert!((joint_distance(&empty, &empty)).abs() < 1e-10);
    }

    #[test]
    fn cspace_tree_basic_operations() {
        let mut tree = CSpaceTree::new(vec![0.0, 0.0]);
        assert_eq!(tree.len(), 1);

        let idx = tree.add(vec![1.0, 0.0], 0);
        assert_eq!(idx, 1);
        assert_eq!(tree.len(), 2);

        let idx2 = tree.add(vec![2.0, 0.0], 1);
        assert_eq!(idx2, 2);
        assert_eq!(tree.len(), 3);
    }

    #[test]
    fn cspace_tree_nearest() {
        let mut tree = CSpaceTree::new(vec![0.0, 0.0]);
        tree.add(vec![1.0, 0.0], 0);
        tree.add(vec![5.0, 0.0], 0);

        // Nearest to (0.9, 0.0) should be node 1 (1.0, 0.0)
        assert_eq!(tree.nearest(&[0.9, 0.0]), 1);
        // Nearest to (4.0, 0.0) should be node 2 (5.0, 0.0)
        assert_eq!(tree.nearest(&[4.0, 0.0]), 2);
        // Nearest to (0.0, 0.0) should be root
        assert_eq!(tree.nearest(&[0.0, 0.0]), 0);
    }

    #[test]
    fn cspace_tree_path_to_root() {
        let mut tree = CSpaceTree::new(vec![0.0]);
        tree.add(vec![1.0], 0);
        tree.add(vec![2.0], 1);
        tree.add(vec![3.0], 2);

        let path = tree.path_to_root(3);
        assert_eq!(path.len(), 4);
        assert_eq!(path[0], vec![0.0]);
        assert_eq!(path[1], vec![1.0]);
        assert_eq!(path[2], vec![2.0]);
        assert_eq!(path[3], vec![3.0]);
    }

    #[test]
    fn cspace_tree_path_to_root_of_root() {
        let tree = CSpaceTree::new(vec![42.0]);
        let path = tree.path_to_root(0);
        assert_eq!(path.len(), 1);
        assert_eq!(path[0], vec![42.0]);
    }

    #[test]
    fn constrained_rrt_with_projection_iterations() {
        let planner = test_setup(vec![Constraint::joint(0, -1.0, 1.0)])
            .with_projection_iterations(100);

        let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let goal = Goal::Joints(JointValues(vec![0.5, -0.5, 0.3, 0.1, -0.2, 0.4]));

        let result = planner.plan(&start, &goal).unwrap();
        assert!(result.waypoints.len() >= 2);
    }

    #[test]
    fn constrained_rrt_named_goal_fails() {
        let planner = test_setup(vec![]);
        let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let goal = Goal::Named("nonexistent".to_string());

        let result = planner.plan(&start, &goal);
        assert!(result.is_err());
    }

    #[test]
    fn constrained_rrt_relative_goal_fails() {
        let planner = test_setup(vec![]);
        let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let goal = Goal::Relative(nalgebra::Vector3::new(0.1, 0.0, 0.0));

        let result = planner.plan(&start, &goal);
        assert!(result.is_err());
    }

    #[test]
    fn constrained_planning_result_debug() {
        let result = ConstrainedPlanningResult {
            waypoints: vec![vec![0.0; 6], vec![1.0; 6]],
            planning_time: std::time::Duration::from_millis(10),
            iterations: 100,
            tree_size: 50,
            projection_rejections: 5,
        };
        let debug = format!("{:?}", result);
        assert!(debug.contains("iterations"));
        assert!(debug.contains("projection_rejections"));
        assert!(debug.contains("tree_size"));
    }

    #[test]
    fn constrained_planning_result_clone() {
        let result = ConstrainedPlanningResult {
            waypoints: vec![vec![0.0; 6]],
            planning_time: std::time::Duration::from_millis(1),
            iterations: 10,
            tree_size: 5,
            projection_rejections: 2,
        };
        let cloned = result.clone();
        assert_eq!(cloned.iterations, 10);
        assert_eq!(cloned.projection_rejections, 2);
    }

    #[test]
    fn extend_result_equality() {
        assert_eq!(ExtendResult::Reached, ExtendResult::Reached);
        assert_eq!(ExtendResult::Advanced, ExtendResult::Advanced);
        assert_eq!(ExtendResult::Trapped, ExtendResult::Trapped);
        assert_ne!(ExtendResult::Reached, ExtendResult::Trapped);
        assert_ne!(ExtendResult::Advanced, ExtendResult::Reached);
    }

    #[test]
    fn constrained_rrt_same_start_and_goal() {
        let planner = test_setup(vec![]);
        let joints = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let goal = Goal::Joints(JointValues(joints.clone()));

        let result = planner.plan(&joints, &goal).unwrap();
        // Should succeed quickly since start == goal
        assert!(result.waypoints.len() >= 1);
    }

    #[test]
    fn constrained_rrt_with_shortcutting() {
        let robot = Robot::from_name("ur5e").unwrap();
        let chain = KinematicChain::extract(&robot, "base_link", "tool0").unwrap();
        let env = CollisionEnvironment::empty(0.05, AABB::symmetric(10.0));

        let planner = ConstrainedRRT::new(
            Arc::new(robot),
            chain,
            env,
            PlannerConfig {
                max_iterations: 5000,
                shortcut_iterations: 50,
                smooth: false,
                ..PlannerConfig::default()
            },
            RRTConfig {
                step_size: 0.15,
                goal_bias: 0.1,
            },
            vec![Constraint::joint(0, -1.0, 1.0)],
        );

        let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let goal = Goal::Joints(JointValues(vec![0.5, -0.5, 0.3, 0.1, -0.2, 0.4]));

        let result = planner.plan(&start, &goal).unwrap();
        assert!(result.waypoints.len() >= 2);
        // All waypoints should still satisfy joint constraint
        for wp in &result.waypoints {
            assert!(
                wp[0] >= -1.0 - 1e-6 && wp[0] <= 1.0 + 1e-6,
                "Joint 0 out of bounds after shortcutting: {}",
                wp[0]
            );
        }
    }

    #[test]
    fn constrained_rrt_collision_checker_trait() {
        let planner = test_setup(vec![]);
        // Test the CollisionChecker trait implementation
        let joints = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let is_colliding: bool = crate::shortcut::CollisionChecker::is_in_collision(&planner, &joints);
        // In free space, should not be colliding
        assert!(!is_colliding);
    }
}
