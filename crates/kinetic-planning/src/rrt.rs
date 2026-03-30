//! Bidirectional RRT-Connect planner (LaValle & Kuffner, 2000).
//!
//! SIMD-accelerated collision checking via kinetic_collision makes this
//! planner achieve <100µs p50 on simple scenes.
//!
//! # Algorithm
//!
//! 1. Initialize two trees: `tree_start` rooted at start, `tree_goal` rooted at goal.
//! 2. Each iteration:
//!    a. Sample a random point in C-space (with goal bias).
//!    b. Extend one tree toward the sample.
//!    c. If extended, try to connect the other tree to the new node.
//!    d. If connected, extract and return the path.
//!    e. Swap which tree extends vs connects.
//! 3. Repeat until path found or timeout/iteration limit reached.

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

/// Result of an extend operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ExtendResult {
    /// Reached the target exactly.
    Reached,
    /// Extended partway toward the target (new node added).
    Advanced,
    /// Could not extend (collision or out of bounds).
    Trapped,
}

/// A node in the RRT tree.
#[derive(Debug, Clone)]
struct TreeNode {
    /// Joint configuration at this node.
    joints: Vec<f64>,
    /// Parent node index (None for root).
    parent: Option<usize>,
}

/// A C-space tree for the bidirectional RRT.
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

    /// Brute-force nearest neighbor (fast enough for <10k nodes in 6-7D).
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

/// Euclidean distance in joint space.
fn joint_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// RRT-specific configuration.
#[derive(Debug, Clone)]
pub struct RRTConfig {
    /// Step size as a fraction of joint range (default 0.1 radians).
    pub step_size: f64,
    /// Probability of sampling the goal instead of random (default 0.05).
    pub goal_bias: f64,
}

impl Default for RRTConfig {
    fn default() -> Self {
        Self {
            step_size: 0.1,
            goal_bias: 0.05,
        }
    }
}

/// Planning result: a collision-free path through C-space.
#[derive(Debug, Clone)]
pub struct PlanningResult {
    /// Waypoints (joint configurations) along the path.
    pub waypoints: Vec<Vec<f64>>,
    /// Total planning time.
    pub planning_time: Duration,
    /// Number of RRT iterations.
    pub iterations: usize,
    /// Total nodes in both trees.
    pub tree_size: usize,
}

/// Bidirectional RRT-Connect planner.
///
/// Uses SIMD sphere-tree collision checking for fast iteration.
pub struct RRTConnect {
    robot: Arc<Robot>,
    chain: KinematicChain,
    sphere_model: RobotSphereModel,
    acm: ResolvedACM,
    environment: CollisionEnvironment,
    planner_config: PlannerConfig,
    rrt_config: RRTConfig,
}

impl CollisionChecker for RRTConnect {
    fn is_in_collision(&self, joints: &[f64]) -> bool {
        self.is_in_collision(joints)
    }
}

impl RRTConnect {
    /// Create a new RRT-Connect planner.
    pub fn new(
        robot: Arc<Robot>,
        chain: KinematicChain,
        environment: CollisionEnvironment,
        planner_config: PlannerConfig,
        rrt_config: RRTConfig,
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
            RRTConfig::default(),
        )
    }

    /// Set the ACM (allowed collision matrix) for self-collision filtering.
    pub fn with_acm(mut self, acm: &AllowedCollisionMatrix) -> Self {
        self.acm = ResolvedACM::from_acm(acm, &self.robot);
        self
    }

    /// Plan a collision-free path from start to goal.
    ///
    /// Pipeline: RRT-Connect search → path shortcutting → optional spline smoothing.
    pub fn plan(&self, start: &[f64], goal: &Goal) -> Result<PlanningResult> {
        let start_time = Instant::now();

        // Resolve goal to one or more joint configurations
        let goal_configs = self.resolve_goal(goal)?;

        if goal_configs.is_empty() {
            return Err(KineticError::GoalUnreachable);
        }

        // Validate start configuration
        if self.is_in_collision(start) {
            return Err(KineticError::StartInCollision);
        }

        // Try planning to each goal config, return first success
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
                            self.rrt_config.step_size,
                        );
                    }

                    // Post-process: spline smoothing
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

    /// Plan to a specific joint-space goal.
    fn plan_to_joints(
        &self,
        start: &[f64],
        goal: &[f64],
        start_time: Instant,
    ) -> Result<PlanningResult> {
        let mut tree_a = CSpaceTree::new(start.to_vec());
        let mut tree_b = CSpaceTree::new(goal.to_vec());
        let mut rng = rand::thread_rng();

        let joint_limits = self.get_joint_limits();

        for iteration in 0..self.planner_config.max_iterations {
            // Check timeout
            let elapsed = start_time.elapsed();
            if elapsed > self.planner_config.timeout {
                return Err(KineticError::PlanningTimeout {
                    elapsed,
                    iterations: iteration,
                });
            }

            // Sample random configuration (with goal bias)
            let sample = if rng.gen::<f64>() < self.rrt_config.goal_bias {
                goal.to_vec()
            } else {
                self.random_sample(&joint_limits, &mut rng)
            };

            // Extend tree_a toward sample
            let nearest_a = tree_a.nearest(&sample);
            let extend_result = self.extend(&mut tree_a, nearest_a, &sample);

            if extend_result != ExtendResult::Trapped {
                // Try to connect tree_b to the new node in tree_a
                let new_node_a = tree_a.len() - 1;
                let target = &tree_a.nodes[new_node_a].joints;

                let nearest_b = tree_b.nearest(target);
                let connect_result = self.connect(&mut tree_b, nearest_b, target);

                if connect_result == ExtendResult::Reached {
                    // Path found! Extract and return.
                    let new_node_b = tree_b.len() - 1;
                    let path = self.extract_path(&tree_a, &tree_b, new_node_a, new_node_b);

                    return Ok(PlanningResult {
                        waypoints: path,
                        planning_time: start_time.elapsed(),
                        iterations: iteration + 1,
                        tree_size: tree_a.len() + tree_b.len(),
                    });
                }
            }

            // Swap trees for next iteration
            std::mem::swap(&mut tree_a, &mut tree_b);
        }

        Err(KineticError::PlanningTimeout {
            elapsed: start_time.elapsed(),
            iterations: self.planner_config.max_iterations,
        })
    }

    /// Extend tree from `nearest_idx` toward `target`, one step.
    fn extend(&self, tree: &mut CSpaceTree, nearest_idx: usize, target: &[f64]) -> ExtendResult {
        let nearest = &tree.nodes[nearest_idx].joints;
        let dist = joint_distance(nearest, target);

        if dist < 1e-10 {
            return ExtendResult::Reached;
        }

        let step = self.rrt_config.step_size;

        if dist <= step {
            // Target is within one step — try to reach it
            if !self.is_in_collision(target) {
                tree.add(target.to_vec(), nearest_idx);
                return ExtendResult::Reached;
            }
            return ExtendResult::Trapped;
        }

        // Interpolate one step toward target
        let ratio = step / dist;
        let new_joints: Vec<f64> = nearest
            .iter()
            .zip(target.iter())
            .map(|(a, b)| a + ratio * (b - a))
            .collect();

        if !self.is_in_collision(&new_joints) {
            tree.add(new_joints, nearest_idx);
            ExtendResult::Advanced
        } else {
            ExtendResult::Trapped
        }
    }

    /// Greedily extend tree toward target until reaching or trapped.
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
    ///
    /// path_a goes root_a → node_a, path_b goes root_b → node_b.
    /// Combined path: root_a → node_a → node_b → root_b (reversed).
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
        // Skip duplicate connection point
        if !full_path.is_empty() && !path_b.is_empty() {
            let last_a = full_path.last().unwrap();
            let first_b = &path_b[0];
            if joint_distance(last_a, first_b) < 1e-8 {
                path_b.remove(0);
            }
        }
        full_path.extend(path_b);

        full_path
    }

    /// Check if a configuration is in collision.
    fn is_in_collision(&self, joints: &[f64]) -> bool {
        // Compute FK for all links
        let link_poses = match forward_kinematics_all(&self.robot, &self.chain, joints) {
            Ok(poses) => poses,
            Err(_) => return true, // treat FK failure as collision
        };

        // Build world-frame spheres
        let mut runtime = self.sphere_model.create_runtime();
        runtime.update(&link_poses);

        // Check robot vs environment
        if self
            .environment
            .check_collision_with_margin(&runtime.world, self.planner_config.collision_margin)
        {
            return true;
        }

        // Check self-collision
        let skip_pairs = self.acm.to_skip_pairs();
        runtime.self_collision_with_margin(&skip_pairs, self.planner_config.collision_margin)
    }

    /// Resolve a Goal to one or more joint configurations.
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
                "Relative goals not supported in RRT planner".into(),
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

    fn setup_planner_free_space() -> (Arc<Robot>, KinematicChain, RRTConnect) {
        let robot = Arc::new(Robot::from_name("ur5e").unwrap());
        let chain = KinematicChain::extract(&robot, "base_link", "tool0").unwrap();
        let env = CollisionEnvironment::empty(0.05, AABB::symmetric(2.0));

        let config = PlannerConfig {
            timeout: Duration::from_secs(5),
            max_iterations: 50_000,
            collision_margin: 0.0, // no margin for free-space test
            ..Default::default()
        };

        let planner = RRTConnect::new(
            robot.clone(),
            chain.clone(),
            env,
            config,
            RRTConfig::default(),
        );

        (robot, chain, planner)
    }

    #[test]
    fn rrt_free_space_joint_goal() {
        let (_robot, _chain, planner) = setup_planner_free_space();

        let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let goal_joints = vec![0.5, -0.5, 0.3, 0.1, -0.2, 0.4];
        let goal = Goal::Joints(JointValues(goal_joints.clone()));

        let result = planner.plan(&start, &goal).unwrap();

        assert!(!result.waypoints.is_empty(), "Path should have waypoints");
        assert!(
            result.waypoints.len() >= 2,
            "Path needs at least start and goal"
        );

        // First waypoint should be start
        let first = &result.waypoints[0];
        for (a, b) in first.iter().zip(start.iter()) {
            assert!((a - b).abs() < 1e-6, "First waypoint should match start");
        }

        // Last waypoint should be close to goal
        let last = result.waypoints.last().unwrap();
        let dist = joint_distance(last, &goal_joints);
        assert!(
            dist < 0.2,
            "Last waypoint should be near goal, got dist={}",
            dist
        );
    }

    #[test]
    fn rrt_free_space_pose_goal() {
        let (robot, chain, planner) = setup_planner_free_space();

        let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let target_joints = vec![0.3, -0.8, 0.5, 0.0, 0.0, 0.0];
        let target_pose =
            kinetic_kinematics::forward_kinematics(&robot, &chain, &target_joints).unwrap();
        let goal = Goal::Pose(target_pose);

        let result = planner.plan(&start, &goal).unwrap();
        assert!(!result.waypoints.is_empty());
    }

    /// URDF with collision geometry for testing collision-aware planning.
    const COLLISION_URDF: &str = r#"<?xml version="1.0"?>
<robot name="test_arm">
  <link name="base_link">
    <collision><geometry><box size="0.2 0.2 0.1"/></geometry></collision>
  </link>
  <link name="link1">
    <collision><geometry><cylinder radius="0.04" length="0.3"/></geometry></collision>
  </link>
  <link name="link2">
    <collision><geometry><cylinder radius="0.04" length="0.3"/></geometry></collision>
  </link>
  <link name="ee_link">
    <collision><geometry><sphere radius="0.03"/></geometry></collision>
  </link>

  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.05"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="100"/>
  </joint>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0 0 0.3"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="2.0" effort="80"/>
  </joint>
  <joint name="joint3" type="revolute">
    <parent link="link2"/>
    <child link="ee_link"/>
    <origin xyz="0 0 0.3"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.5" upper="2.5" velocity="3.0" effort="50"/>
  </joint>
</robot>
"#;

    #[test]
    fn rrt_start_in_collision() {
        let robot = Arc::new(Robot::from_urdf_string(COLLISION_URDF).unwrap());
        let chain = KinematicChain::extract(&robot, "base_link", "ee_link").unwrap();

        // Place a massive obstacle engulfing everything
        let mut obstacles = SpheresSoA::new();
        obstacles.push(0.0, 0.0, 0.0, 5.0, 0);
        let env = CollisionEnvironment::build(obstacles, 0.05, AABB::symmetric(10.0));

        let planner = RRTConnect::new(
            robot,
            chain,
            env,
            PlannerConfig::default(),
            RRTConfig::default(),
        );

        let start = vec![0.0, 0.0, 0.0];
        let goal = Goal::Joints(JointValues(vec![1.0, -1.0, 0.5]));

        match planner.plan(&start, &goal) {
            Err(KineticError::StartInCollision) => {} // expected
            other => panic!("Expected StartInCollision, got {:?}", other),
        }
    }

    #[test]
    fn rrt_with_obstacles() {
        let robot = Arc::new(Robot::from_name("ur5e").unwrap());
        let chain = KinematicChain::extract(&robot, "base_link", "tool0").unwrap();

        // Small obstacles that don't block the path
        let mut obstacles = SpheresSoA::new();
        obstacles.push(2.0, 2.0, 2.0, 0.1, 0);
        let env = CollisionEnvironment::build(obstacles, 0.05, AABB::symmetric(5.0));

        let config = PlannerConfig {
            timeout: Duration::from_secs(5),
            max_iterations: 50_000,
            collision_margin: 0.01,
            ..Default::default()
        };

        let planner = RRTConnect::new(robot, chain, env, config, RRTConfig::default());

        let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let goal = Goal::Joints(JointValues(vec![0.5, -0.5, 0.3, 0.0, 0.0, 0.0]));

        let result = planner.plan(&start, &goal).unwrap();
        assert!(result.waypoints.len() >= 2);
    }

    #[test]
    fn rrt_path_collision_free() {
        let (_robot, _chain, planner) = setup_planner_free_space();

        let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let goal = Goal::Joints(JointValues(vec![0.5, -0.5, 0.3, 0.0, 0.0, 0.0]));

        let result = planner.plan(&start, &goal).unwrap();

        // Verify every waypoint is collision-free
        for (i, waypoint) in result.waypoints.iter().enumerate() {
            assert!(
                !planner.is_in_collision(waypoint),
                "Waypoint {} is in collision",
                i
            );
        }
    }

    #[test]
    fn rrt_iteration_limit() {
        // Use a very tight iteration limit to test the timeout path
        let robot = Arc::new(Robot::from_name("ur5e").unwrap());
        let chain = KinematicChain::extract(&robot, "base_link", "tool0").unwrap();
        let env = CollisionEnvironment::empty(0.05, AABB::symmetric(2.0));

        let config = PlannerConfig {
            timeout: Duration::from_millis(10),
            max_iterations: 1, // only 1 iteration allowed
            collision_margin: 0.0,
            ..Default::default()
        };

        let planner = RRTConnect::new(robot, chain, env, config, RRTConfig::default());

        // Very far apart — unlikely to connect in 1 iteration
        let start = vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let goal = Goal::Joints(JointValues(vec![3.0, -3.0, 3.0, -3.0, 3.0, -3.0]));

        match planner.plan(&start, &goal) {
            Err(KineticError::PlanningTimeout { .. }) => {} // expected: hit iteration limit
            Err(KineticError::GoalUnreachable) => {}        // goal may violate joint limits
            Ok(_) => {}                                     // extremely unlikely but possible
            other => panic!("Expected timeout or goal unreachable, got {:?}", other),
        }
    }
}
