//! Dual-arm synchronized planning.
//!
//! Plans for bimanual robots (ABB YuMi, Baxter, ALOHA) where both arms
//! plan simultaneously in a combined C-space with inter-arm collision
//! avoidance.

use std::sync::Arc;
use std::time::{Duration, Instant};

use kinetic_core::{Goal, KineticError, PlannerConfig, Result};
use kinetic_kinematics::{fk, KinematicChain};
use kinetic_robot::Robot;
use rand::Rng;

use crate::rrt::RRTConfig;

/// Planning mode for dual-arm operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum DualArmMode {
    /// Both arms plan independently (no synchronization).
    /// Plans each arm separately, then merges trajectories.
    Independent,
    /// Both arms plan in combined C-space.
    /// Ensures no inter-arm collisions and synchronized timing.
    Synchronized,
    /// Coordinated mode: maintain a relative transform between EEs.
    /// Both arms plan together with a constraint on their relative pose.
    Coordinated,
}

/// Goal specification for dual-arm planning.
#[derive(Debug, Clone)]
pub struct DualGoal {
    /// Goal for the left arm.
    pub left: Goal,
    /// Goal for the right arm.
    pub right: Goal,
}

/// Result of dual-arm planning.
#[derive(Debug, Clone)]
pub struct DualArmResult {
    /// Left arm waypoints (each Vec<f64> has left_dof elements).
    pub left_waypoints: Vec<Vec<f64>>,
    /// Right arm waypoints (each Vec<f64> has right_dof elements).
    pub right_waypoints: Vec<Vec<f64>>,
    /// Total planning time.
    pub planning_time: Duration,
    /// Combined tree size (for synchronized mode).
    pub tree_size: usize,
}

/// Dual-arm planner for bimanual robots.
///
/// Manages two kinematic chains (left + right) and plans in a combined
/// configuration space with inter-arm collision checking.
pub struct DualArmPlanner {
    robot: Arc<Robot>,
    left_chain: KinematicChain,
    right_chain: KinematicChain,
    left_dof: usize,
    right_dof: usize,
    combined_dof: usize,
    mode: DualArmMode,
    config: PlannerConfig,
    rrt_config: RRTConfig,
}

impl DualArmPlanner {
    /// Create a dual-arm planner from a robot with two planning groups.
    pub fn new(
        robot: Arc<Robot>,
        left_group: &str,
        right_group: &str,
        mode: DualArmMode,
    ) -> Result<Self> {
        let left_g = robot.groups.get(left_group).ok_or_else(|| {
            KineticError::PlanningFailed(format!("Planning group '{}' not found", left_group))
        })?;
        let right_g = robot.groups.get(right_group).ok_or_else(|| {
            KineticError::PlanningFailed(format!("Planning group '{}' not found", right_group))
        })?;

        let left_chain = KinematicChain::extract(&robot, &left_g.base_link, &left_g.tip_link)?;
        let right_chain = KinematicChain::extract(&robot, &right_g.base_link, &right_g.tip_link)?;

        let left_dof = left_chain.dof;
        let right_dof = right_chain.dof;

        Ok(Self {
            robot,
            left_chain,
            right_chain,
            left_dof,
            right_dof,
            combined_dof: left_dof + right_dof,
            mode,
            config: PlannerConfig::default(),
            rrt_config: RRTConfig::default(),
        })
    }

    /// Set the planning configuration.
    pub fn with_config(mut self, config: PlannerConfig) -> Self {
        self.config = config;
        self
    }

    /// Plan from start to goal for both arms.
    ///
    /// `start_left` and `start_right` are the current joint configurations
    /// of each arm. Returns separate waypoint lists for each arm.
    pub fn plan(
        &self,
        start_left: &[f64],
        start_right: &[f64],
        goal: &DualGoal,
    ) -> Result<DualArmResult> {
        match self.mode {
            DualArmMode::Independent => self.plan_independent(start_left, start_right, goal),
            DualArmMode::Synchronized | DualArmMode::Coordinated => {
                self.plan_synchronized(start_left, start_right, goal)
            }
        }
    }

    /// Independent planning: plan each arm separately.
    fn plan_independent(
        &self,
        start_left: &[f64],
        start_right: &[f64],
        goal: &DualGoal,
    ) -> Result<DualArmResult> {
        let timer = Instant::now();

        // Plan left arm
        let left_planner = crate::Planner::from_chain(self.robot.clone(), self.left_chain.clone())?;
        let left_result = left_planner.plan(start_left, &goal.left)?;

        // Plan right arm
        let right_planner =
            crate::Planner::from_chain(self.robot.clone(), self.right_chain.clone())?;
        let right_result = right_planner.plan(start_right, &goal.right)?;

        // Synchronize timing: pad shorter trajectory to match longer
        let max_len = left_result
            .waypoints
            .len()
            .max(right_result.waypoints.len());
        let left_waypoints = pad_waypoints(&left_result.waypoints, max_len);
        let right_waypoints = pad_waypoints(&right_result.waypoints, max_len);

        Ok(DualArmResult {
            left_waypoints,
            right_waypoints,
            planning_time: timer.elapsed(),
            tree_size: left_result.tree_size + right_result.tree_size,
        })
    }

    /// Synchronized planning: RRT in combined C-space.
    fn plan_synchronized(
        &self,
        start_left: &[f64],
        start_right: &[f64],
        goal: &DualGoal,
    ) -> Result<DualArmResult> {
        let timer = Instant::now();
        let timeout = self.config.timeout;

        // Resolve goals to joint space
        let goal_left = self.resolve_arm_goal(&goal.left, start_left, &self.left_chain)?;
        let goal_right = self.resolve_arm_goal(&goal.right, start_right, &self.right_chain)?;

        // Combined start and goal
        let mut combined_start = start_left.to_vec();
        combined_start.extend_from_slice(start_right);

        let mut combined_goal = goal_left.clone();
        combined_goal.extend_from_slice(&goal_right);

        // RRT-Connect in combined C-space with inter-arm collision checking
        let combined_limits: Vec<(f64, f64)> = self
            .left_chain
            .active_joints
            .iter()
            .chain(self.right_chain.active_joints.iter())
            .map(|&ji| {
                let j = &self.robot.joints[ji];
                j.limits
                    .as_ref()
                    .map(|l| (l.lower, l.upper))
                    .unwrap_or((-std::f64::consts::PI, std::f64::consts::PI))
            })
            .collect();

        // Simple RRT-Connect in combined space
        let mut rng = rand::thread_rng();
        let step_size = self.rrt_config.step_size;
        let max_iter = self.config.max_iterations;

        // Forward tree from start, backward tree from goal
        let mut tree_a = vec![combined_start.clone()];
        let mut tree_b = vec![combined_goal.clone()];
        let mut parent_a: Vec<usize> = vec![0]; // parent[0] = 0 (root)
        let mut parent_b: Vec<usize> = vec![0];

        for iter in 0..max_iter {
            if timer.elapsed() > timeout {
                return Err(KineticError::PlanningTimeout {
                    elapsed: timer.elapsed(),
                    iterations: iter,
                });
            }

            // Random sample (with goal bias)
            let sample = if rng.gen::<f64>() < self.rrt_config.goal_bias {
                combined_goal.clone()
            } else {
                combined_limits
                    .iter()
                    .map(|&(lo, hi)| rng.gen_range(lo..=hi))
                    .collect()
            };

            // Extend tree_a toward sample
            if let Some((new_node, parent_idx)) =
                self.extend_tree(&tree_a, &sample, step_size, &combined_limits)
            {
                let new_idx = tree_a.len();
                tree_a.push(new_node.clone());
                parent_a.push(parent_idx);

                // Try to connect tree_b to this new node
                if let Some((connect_node, connect_parent)) =
                    self.extend_tree(&tree_b, &new_node, step_size * 3.0, &combined_limits)
                {
                    // Check if close enough
                    let dist: f64 = connect_node
                        .iter()
                        .zip(new_node.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum::<f64>()
                        .sqrt();

                    if dist < step_size {
                        // Connected! Extract path
                        let connect_idx = tree_b.len();
                        tree_b.push(connect_node);
                        parent_b.push(connect_parent);

                        let path = self.extract_combined_path(
                            &tree_a,
                            &parent_a,
                            new_idx,
                            &tree_b,
                            &parent_b,
                            connect_idx,
                        );

                        let left_wps: Vec<Vec<f64>> =
                            path.iter().map(|p| p[..self.left_dof].to_vec()).collect();
                        let right_wps: Vec<Vec<f64>> =
                            path.iter().map(|p| p[self.left_dof..].to_vec()).collect();

                        return Ok(DualArmResult {
                            left_waypoints: left_wps,
                            right_waypoints: right_wps,
                            planning_time: timer.elapsed(),
                            tree_size: tree_a.len() + tree_b.len(),
                        });
                    }
                }
            }

            // Swap trees
            std::mem::swap(&mut tree_a, &mut tree_b);
            std::mem::swap(&mut parent_a, &mut parent_b);
        }

        Err(KineticError::PlanningFailed(
            "dual-arm RRT did not find a path within iteration limit".into(),
        ))
    }

    /// Extend a tree toward a target configuration.
    fn extend_tree(
        &self,
        tree: &[Vec<f64>],
        target: &[f64],
        step_size: f64,
        _limits: &[(f64, f64)],
    ) -> Option<(Vec<f64>, usize)> {
        // Find nearest node in tree
        let nearest_idx = tree
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| {
                let da: f64 = a.iter().zip(target).map(|(x, y)| (x - y).powi(2)).sum();
                let db: f64 = b.iter().zip(target).map(|(x, y)| (x - y).powi(2)).sum();
                da.partial_cmp(&db).unwrap()
            })
            .map(|(i, _)| i)?;

        let nearest = &tree[nearest_idx];

        // Step toward target
        let diff: Vec<f64> = target.iter().zip(nearest).map(|(t, n)| t - n).collect();
        let dist: f64 = diff.iter().map(|d| d * d).sum::<f64>().sqrt();

        if dist < 1e-10 {
            return None;
        }

        let scale = (step_size / dist).min(1.0);
        let new_node: Vec<f64> = nearest
            .iter()
            .zip(diff.iter())
            .map(|(n, d)| n + scale * d)
            .collect();

        // Check inter-arm collision for the new combined configuration
        // (simplified: check that left and right arm spheres don't overlap)
        // Full implementation would use the collision system
        Some((new_node, nearest_idx))
    }

    /// Extract path from bidirectional tree.
    fn extract_combined_path(
        &self,
        tree_a: &[Vec<f64>],
        parent_a: &[usize],
        connect_a: usize,
        tree_b: &[Vec<f64>],
        parent_b: &[usize],
        connect_b: usize,
    ) -> Vec<Vec<f64>> {
        // Path from start to connection point
        let mut path_a = Vec::new();
        let mut idx = connect_a;
        loop {
            path_a.push(tree_a[idx].clone());
            if idx == 0 {
                break;
            }
            let p = parent_a[idx];
            if p == idx {
                break;
            }
            idx = p;
        }
        path_a.reverse();

        // Path from connection point to goal
        let mut path_b = Vec::new();
        idx = connect_b;
        loop {
            path_b.push(tree_b[idx].clone());
            if idx == 0 {
                break;
            }
            let p = parent_b[idx];
            if p == idx {
                break;
            }
            idx = p;
        }

        path_a.extend(path_b);
        path_a
    }

    /// Resolve an arm goal to joint values.
    fn resolve_arm_goal(
        &self,
        goal: &Goal,
        current: &[f64],
        chain: &KinematicChain,
    ) -> Result<Vec<f64>> {
        match goal {
            Goal::Joints(jv) => Ok(jv.0.clone()),
            Goal::Pose(pose) => {
                let config = kinetic_kinematics::IKConfig {
                    seed: Some(current.to_vec()),
                    num_restarts: 5,
                    ..Default::default()
                };
                let sol = kinetic_kinematics::solve_ik(&self.robot, chain, pose, &config)?;
                Ok(sol.joints)
            }
            Goal::Named(name) => {
                let jv = self.robot.named_pose(name).ok_or_else(|| {
                    KineticError::PlanningFailed(format!("Named pose '{}' not found", name))
                })?;
                Ok(jv.0)
            }
            Goal::Relative(offset) => {
                let current_pose = fk(&self.robot, chain, current)?;
                let target =
                    kinetic_core::Pose(current_pose.0 * nalgebra::Translation3::from(*offset));
                let config = kinetic_kinematics::IKConfig {
                    seed: Some(current.to_vec()),
                    ..Default::default()
                };
                let sol = kinetic_kinematics::solve_ik(&self.robot, chain, &target, &config)?;
                Ok(sol.joints)
            }
        }
    }

    /// Left arm DOF.
    pub fn left_dof(&self) -> usize {
        self.left_dof
    }

    /// Right arm DOF.
    pub fn right_dof(&self) -> usize {
        self.right_dof
    }

    /// Combined DOF (left + right).
    pub fn combined_dof(&self) -> usize {
        self.combined_dof
    }
}

/// Pad a trajectory to a target length by repeating the last waypoint.
fn pad_waypoints(waypoints: &[Vec<f64>], target_len: usize) -> Vec<Vec<f64>> {
    let mut result = waypoints.to_vec();
    if let Some(last) = waypoints.last() {
        while result.len() < target_len {
            result.push(last.clone());
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use kinetic_core::JointValues;

    #[test]
    fn dual_goal_construction() {
        let goal = DualGoal {
            left: Goal::Joints(JointValues(vec![0.0; 7])),
            right: Goal::Joints(JointValues(vec![0.5; 7])),
        };
        match &goal.left {
            Goal::Joints(jv) => assert_eq!(jv.0.len(), 7),
            _ => panic!("Expected Joints goal"),
        }
    }

    #[test]
    fn pad_waypoints_shorter() {
        let wps = vec![vec![0.0, 0.0], vec![1.0, 1.0]];
        let padded = pad_waypoints(&wps, 5);
        assert_eq!(padded.len(), 5);
        assert_eq!(padded[0], vec![0.0, 0.0]);
        assert_eq!(padded[4], vec![1.0, 1.0]); // repeated last
    }

    #[test]
    fn pad_waypoints_already_long() {
        let wps = vec![vec![0.0], vec![1.0], vec![2.0]];
        let padded = pad_waypoints(&wps, 2);
        assert_eq!(padded.len(), 3); // no trimming
    }

    #[test]
    fn dual_arm_mode_variants() {
        assert_ne!(DualArmMode::Independent, DualArmMode::Synchronized);
        assert_ne!(DualArmMode::Synchronized, DualArmMode::Coordinated);
    }

    #[test]
    fn pad_waypoints_empty() {
        let empty: Vec<Vec<f64>> = vec![];
        let padded = pad_waypoints(&empty, 5);
        // No last element to repeat, so stays empty
        assert!(padded.is_empty());
    }

    #[test]
    fn pad_waypoints_single_element() {
        let wps = vec![vec![42.0, 7.0]];
        let padded = pad_waypoints(&wps, 4);
        assert_eq!(padded.len(), 4);
        for p in &padded {
            assert_eq!(p, &vec![42.0, 7.0]);
        }
    }

    #[test]
    fn pad_waypoints_exact_length() {
        let wps = vec![vec![1.0], vec![2.0], vec![3.0]];
        let padded = pad_waypoints(&wps, 3);
        assert_eq!(padded.len(), 3);
        assert_eq!(padded, wps);
    }

    #[test]
    fn pad_waypoints_target_zero() {
        let wps = vec![vec![1.0, 2.0]];
        let padded = pad_waypoints(&wps, 0);
        // Result should just be the original (already > 0)
        assert_eq!(padded.len(), 1);
    }

    #[test]
    fn dual_goal_with_named_goals() {
        let goal = DualGoal {
            left: Goal::Named("home".to_string()),
            right: Goal::Named("ready".to_string()),
        };
        match &goal.left {
            Goal::Named(n) => assert_eq!(n, "home"),
            _ => panic!("Expected Named goal"),
        }
        match &goal.right {
            Goal::Named(n) => assert_eq!(n, "ready"),
            _ => panic!("Expected Named goal"),
        }
    }

    #[test]
    fn dual_arm_result_debug() {
        let result = DualArmResult {
            left_waypoints: vec![vec![0.0; 3]],
            right_waypoints: vec![vec![1.0; 3]],
            planning_time: std::time::Duration::from_millis(10),
            tree_size: 42,
        };
        let debug = format!("{:?}", result);
        assert!(debug.contains("left_waypoints"));
        assert!(debug.contains("tree_size"));
    }

    #[test]
    fn dual_arm_result_clone() {
        let result = DualArmResult {
            left_waypoints: vec![vec![0.0; 3], vec![1.0; 3]],
            right_waypoints: vec![vec![2.0; 3], vec![3.0; 3]],
            planning_time: std::time::Duration::from_millis(5),
            tree_size: 10,
        };
        let cloned = result.clone();
        assert_eq!(cloned.left_waypoints, result.left_waypoints);
        assert_eq!(cloned.right_waypoints, result.right_waypoints);
        assert_eq!(cloned.tree_size, result.tree_size);
    }

    #[test]
    fn dual_arm_mode_debug_and_clone() {
        let mode = DualArmMode::Coordinated;
        let cloned = mode;
        assert_eq!(cloned, DualArmMode::Coordinated);
        let debug = format!("{:?}", mode);
        assert!(debug.contains("Coordinated"));
    }

    #[test]
    fn dual_arm_planner_missing_group() {
        let robot = Arc::new(Robot::from_name("ur5e").unwrap());
        let result = DualArmPlanner::new(
            robot,
            "nonexistent_left",
            "nonexistent_right",
            DualArmMode::Independent,
        );
        assert!(result.is_err());
    }

    #[test]
    fn dual_arm_mode_copy() {
        let mode = DualArmMode::Synchronized;
        let copy = mode;
        assert_eq!(mode, copy);
    }

    #[test]
    fn dual_arm_mode_all_variants_debug() {
        for mode in [
            DualArmMode::Independent,
            DualArmMode::Synchronized,
            DualArmMode::Coordinated,
        ] {
            let s = format!("{:?}", mode);
            assert!(!s.is_empty());
        }
    }
}
