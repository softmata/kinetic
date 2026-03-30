//! Unified Planner facade — clean top-level API for motion planning.
//!
//! The `Planner` wraps the RRT-Connect planner (and future planners) behind
//! a single interface that auto-selects the right algorithm based on goal type.
//!
//! # Usage
//!
//! ```ignore
//! use kinetic_planning::Planner;
//! let robot = Robot::from_name("ur5e")?;
//! let planner = Planner::new(&robot)?;
//! let result = planner.plan(&start_joints, &Goal::Joints(goal_joints))?;
//! ```

use std::sync::Arc;

use kinetic_collision::{
    AllowedCollisionMatrix, CollisionEnvironment, ResolvedACM, RobotSphereModel, SphereGenConfig,
};
use kinetic_core::{Goal, JointValues, KineticError, PlannerConfig, Pose, Result};
use kinetic_kinematics::{forward_kinematics_all, forward_kinematics, solve_ik, IKConfig, KinematicChain};
use kinetic_robot::Robot;
use kinetic_scene::Scene;

use crate::bi_rrt_star::BiRRTStar;
use crate::bitrrt::{BiTRRT, BiTRRTConfig, JointCenterCost};
use crate::est::{ESTConfig, EST};
use crate::kpiece::{KPIECEConfig, KPIECE};
use crate::rrt::{RRTConfig, RRTConnect};
use crate::rrt_star::{RRTStar, RRTStarConfig};
use crate::shortcut::{self, CollisionChecker};

/// Planner type selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PlannerType {
    /// Bidirectional RRT-Connect (default, general-purpose).
    RRTConnect,
    /// RRT* — asymptotically optimal path cost reduction.
    RRTStar,
    /// Bidirectional RRT* — faster convergence to optimal.
    BiRRTStar,
    /// Bidirectional informed RRT with transition test.
    BiTRRT,
    /// Expansive Space Tree — good for narrow passages.
    EST,
    /// Kinematic Planning by Interior-Exterior Cell Exploration.
    KPIECE,
    /// Probabilistic Roadmap — multi-query same environment.
    PRM,
    /// Graphs of Convex Sets — globally optimal over pre-computed convex regions.
    GCS,
    /// Auto-select based on goal type and context.
    #[default]
    Auto,
}

/// Unified motion planner facade.
///
/// Wraps RRT-Connect (and future planners) behind a single interface.
/// Auto-detects the kinematic chain from the robot's planning groups or
/// by walking the URDF tree from root to the farthest leaf.
pub struct Planner {
    robot: Arc<Robot>,
    chain: KinematicChain,
    sphere_model: RobotSphereModel,
    acm: ResolvedACM,
    environment: CollisionEnvironment,
    planner_config: PlannerConfig,
    rrt_config: RRTConfig,
    planner_type: PlannerType,
}

impl Planner {
    /// Create a planner for a robot with default configuration.
    ///
    /// Auto-detects the kinematic chain from:
    /// 1. The first planning group (if configured), or
    /// 2. Root link → farthest leaf link.
    ///
    /// Uses empty environment (no obstacles) — add obstacles with `with_environment`.
    pub fn new(robot: &Robot) -> Result<Self> {
        let chain = auto_detect_chain(robot)?;
        Self::with_chain(robot, chain)
    }

    /// Create a planner from an Arc<Robot> and a kinematic chain.
    ///
    /// Useful when the caller already owns an Arc (e.g., dual-arm planner).
    pub fn from_chain(robot: Arc<Robot>, chain: KinematicChain) -> Result<Self> {
        let sphere_model = RobotSphereModel::from_robot(&robot, &SphereGenConfig::coarse());
        let acm = ResolvedACM::from_robot(&robot);
        let environment =
            CollisionEnvironment::empty(0.05, kinetic_collision::capt::AABB::symmetric(10.0));

        Ok(Self {
            robot,
            chain,
            sphere_model,
            acm,
            environment,
            planner_config: PlannerConfig::default(),
            rrt_config: RRTConfig::default(),
            planner_type: PlannerType::default(),
        })
    }

    /// Create a planner with a specific kinematic chain.
    pub fn with_chain(robot: &Robot, chain: KinematicChain) -> Result<Self> {
        let robot = Arc::new(robot.clone());
        let sphere_model = RobotSphereModel::from_robot(&robot, &SphereGenConfig::coarse());
        let acm = ResolvedACM::from_robot(&robot);
        let environment =
            CollisionEnvironment::empty(0.05, kinetic_collision::capt::AABB::symmetric(10.0));

        Ok(Self {
            robot,
            chain,
            sphere_model,
            acm,
            environment,
            planner_config: PlannerConfig::default(),
            rrt_config: RRTConfig::default(),
            planner_type: PlannerType::default(),
        })
    }

    /// Create a planner with a specific environment.
    pub fn with_environment(mut self, environment: CollisionEnvironment) -> Self {
        self.environment = environment;
        self
    }

    /// Create a planner with obstacles from a [`Scene`].
    ///
    /// Extracts environment collision spheres from the scene's objects and
    /// point clouds, building a `CollisionEnvironment` automatically.
    /// Also inherits the scene's ACM (disabled collision pairs from SRDF/config).
    pub fn with_scene(mut self, scene: &Scene) -> Self {
        let spheres = scene.build_environment_spheres();
        self.environment = CollisionEnvironment::build(
            spheres,
            0.05,
            kinetic_collision::capt::AABB::symmetric(10.0),
        );
        self.acm = ResolvedACM::from_acm(scene.acm(), &self.robot);
        self
    }

    /// Create a planner with specific planner config.
    pub fn with_config(mut self, config: PlannerConfig) -> Self {
        self.planner_config = config;
        self
    }

    /// Set the planner type.
    pub fn with_planner_type(mut self, planner_type: PlannerType) -> Self {
        self.planner_type = planner_type;
        self
    }

    /// Set RRT-specific config.
    pub fn with_rrt_config(mut self, config: RRTConfig) -> Self {
        self.rrt_config = config;
        self
    }

    /// Set the ACM for self-collision filtering.
    pub fn with_acm(mut self, acm: &AllowedCollisionMatrix) -> Self {
        self.acm = ResolvedACM::from_acm(acm, &self.robot);
        self
    }

    /// Create a constrained planner from this planner's configuration.
    ///
    /// The returned `ConstrainedRRT` uses the same robot, chain, environment,
    /// and configs, plus the given constraints.
    pub fn with_constraints(
        &self,
        constraints: &[kinetic_core::Constraint],
    ) -> crate::constrained_rrt::ConstrainedRRT {
        crate::constrained_rrt::ConstrainedRRT::new(
            self.robot.clone(),
            self.chain.clone(),
            self.environment.clone(),
            self.planner_config.clone(),
            self.rrt_config.clone(),
            constraints.to_vec(),
        )
    }

    /// Access the underlying robot model.
    pub fn robot(&self) -> &Robot {
        &self.robot
    }

    /// Access the kinematic chain.
    pub fn chain(&self) -> &KinematicChain {
        &self.chain
    }

    /// Plan a collision-free path from start joint configuration to a goal.
    ///
    /// Auto-selects the planning algorithm based on goal type:
    /// - `Goal::Joints` → RRT-Connect
    /// - `Goal::Pose` → IK + RRT-Connect
    /// - `Goal::Named` → Resolve named pose + RRT-Connect
    /// - `Goal::Relative` → FK + offset + IK + RRT-Connect
    pub fn plan(&self, start: &[f64], goal: &Goal) -> Result<PlanningResult> {
        self.plan_with_config(start, goal, self.planner_config.clone())
    }

    /// Plan with a specific planner config (overrides default).
    pub fn plan_with_config(
        &self,
        start: &[f64],
        goal: &Goal,
        config: PlannerConfig,
    ) -> Result<PlanningResult> {
        // Extract chain-DOF joints from full robot config (handles mobile manipulators
        // where robot.dof > chain.dof, e.g., fetch has base joints + arm joints)
        let chain_start = if start.len() == self.chain.dof {
            start.to_vec()
        } else if start.len() > self.chain.dof {
            self.chain.extract_joint_values(start)
        } else {
            return Err(KineticError::DimensionMismatch {
                expected: self.chain.dof,
                got: start.len(),
                context: "start joint values".into(),
            });
        };

        // Validate inputs are finite (NaN/Inf causes RRT to loop forever)
        for (i, &v) in chain_start.iter().enumerate() {
            if !v.is_finite() {
                return Err(KineticError::Other(format!(
                    "start joint {} has non-finite value: {v}", i
                )));
            }
        }
        if let Goal::Joints(jv) = goal {
            for (i, &v) in jv.iter().enumerate() {
                if !v.is_finite() {
                    return Err(KineticError::Other(format!(
                        "goal joint {} has non-finite value: {v}", i
                    )));
                }
            }
            // Check goal is within joint limits
            if self.robot.check_limits(jv).is_err() {
                return Err(KineticError::GoalUnreachable);
            }
        }

        // Resolve the goal to joint-space
        let goal_resolved = self.resolve_goal(goal, &chain_start)?;

        // Capture workspace bounds before moving config into planner
        let workspace_bounds = config.workspace_bounds;

        // Resolve effective planner type (Auto defaults to RRTConnect)
        let effective_type = match self.planner_type {
            PlannerType::Auto => PlannerType::RRTConnect,
            other => other,
        };

        // Dispatch to the selected planner
        let (waypoints, planning_time, iterations, tree_size) = match effective_type {
            PlannerType::RRTConnect | PlannerType::Auto => {
                let rrt = RRTConnect::new(
                    self.robot.clone(),
                    self.chain.clone(),
                    self.environment.clone(),
                    config,
                    self.rrt_config.clone(),
                );
                let r = rrt.plan(&chain_start, &goal_resolved)?;
                (r.waypoints, r.planning_time, r.iterations, r.tree_size)
            }
            PlannerType::RRTStar => {
                let planner = RRTStar::new(
                    self.robot.clone(),
                    self.chain.clone(),
                    self.environment.clone(),
                    config,
                    RRTStarConfig::default(),
                );
                let r = planner.plan(&chain_start, &goal_resolved)?;
                (r.waypoints, r.planning_time, r.iterations, r.tree_size)
            }
            PlannerType::BiRRTStar => {
                let planner = BiRRTStar::new(
                    self.robot.clone(),
                    self.chain.clone(),
                    self.environment.clone(),
                    config,
                    RRTStarConfig::default(),
                );
                let r = planner.plan(&chain_start, &goal_resolved)?;
                (r.waypoints, r.planning_time, r.iterations, r.tree_size)
            }
            PlannerType::BiTRRT => {
                let cost_fn = Box::new(JointCenterCost::from_robot(&self.robot, &self.chain));
                let planner = BiTRRT::new(
                    self.robot.clone(),
                    self.chain.clone(),
                    self.environment.clone(),
                    config,
                    BiTRRTConfig::default(),
                    cost_fn,
                );
                let r = planner.plan(&chain_start, &goal_resolved)?;
                (r.waypoints, r.planning_time, r.iterations, r.tree_size)
            }
            PlannerType::EST => {
                let planner = EST::new(
                    self.robot.clone(),
                    self.chain.clone(),
                    self.environment.clone(),
                    config,
                    ESTConfig::default(),
                );
                let r = planner.plan(&chain_start, &goal_resolved)?;
                (r.waypoints, r.planning_time, r.iterations, r.tree_size)
            }
            PlannerType::KPIECE => {
                let planner = KPIECE::new(
                    self.robot.clone(),
                    self.chain.clone(),
                    self.environment.clone(),
                    config,
                    KPIECEConfig::default(),
                );
                let r = planner.plan(&chain_start, &goal_resolved)?;
                (r.waypoints, r.planning_time, r.iterations, r.tree_size)
            }
            PlannerType::PRM => {
                // PRM requires pre-built roadmap; fall back to RRT-Connect
                let rrt = RRTConnect::new(
                    self.robot.clone(),
                    self.chain.clone(),
                    self.environment.clone(),
                    config,
                    self.rrt_config.clone(),
                );
                let r = rrt.plan(&chain_start, &goal_resolved)?;
                (r.waypoints, r.planning_time, r.iterations, r.tree_size)
            }
            PlannerType::GCS => {
                // GCS requires pre-computed convex decomposition; fall back to RRT-Connect
                let rrt = RRTConnect::new(
                    self.robot.clone(),
                    self.chain.clone(),
                    self.environment.clone(),
                    config,
                    self.rrt_config.clone(),
                );
                let r = rrt.plan(&chain_start, &goal_resolved)?;
                (r.waypoints, r.planning_time, r.iterations, r.tree_size)
            }
        };

        // SAFETY GATE 1: Validate every waypoint against joint limits.
        for (wp_idx, wp) in waypoints.iter().enumerate() {
            let jv = JointValues::from_slice(wp);
            self.robot.check_limits(&jv).map_err(|e| {
                KineticError::PlannerOutputInvalid {
                    waypoint: wp_idx,
                    reason: format!("joint limits: {}", e),
                }
            })?;
        }

        // SAFETY GATE 2: Validate workspace bounds if configured.
        if let Some(bounds) = &workspace_bounds {
            for (wp_idx, wp) in waypoints.iter().enumerate() {
                let ee_pose = forward_kinematics(&self.robot, &self.chain, wp)?;
                let t = ee_pose.translation();
                if t.x < bounds[0] || t.x > bounds[3]
                    || t.y < bounds[1] || t.y > bounds[4]
                    || t.z < bounds[2] || t.z > bounds[5]
                {
                    return Err(KineticError::PlannerOutputInvalid {
                        waypoint: wp_idx,
                        reason: format!(
                            "EE position ({:.3}, {:.3}, {:.3}) outside workspace bounds",
                            t.x, t.y, t.z
                        ),
                    });
                }
            }
        }

        Ok(PlanningResult {
            waypoints,
            planning_time,
            iterations,
            tree_size,
            planner_used: effective_type,
        })
    }

    /// Compute forward kinematics at a joint configuration.
    pub fn fk(&self, joints: &[f64]) -> Result<Pose> {
        forward_kinematics(&self.robot, &self.chain, joints)
    }

    /// Solve IK for a target pose.
    pub fn ik(&self, target: &Pose) -> Result<Vec<f64>> {
        let config = IKConfig {
            num_restarts: 8,
            ..Default::default()
        };
        solve_ik(&self.robot, &self.chain, target, &config).map(|sol| sol.joints)
    }

    /// Check if a joint configuration is in collision.
    pub fn is_in_collision(&self, joints: &[f64]) -> bool {
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

    /// Resolve a Goal to a joint-space Goal suitable for the RRT planner.
    fn resolve_goal(&self, goal: &Goal, current_joints: &[f64]) -> Result<Goal> {
        match goal {
            Goal::Joints(_) => Ok(goal.clone()),
            Goal::Pose(_) => Ok(goal.clone()),

            Goal::Named(name) => {
                if let Some(pose_values) = self.robot.named_pose(name) {
                    Ok(Goal::Joints(pose_values))
                } else {
                    Err(KineticError::NamedConfigNotFound(name.clone()))
                }
            }

            Goal::Relative(offset) => {
                // Compute current EE pose, apply offset, plan to new pose
                let current_pose = forward_kinematics(&self.robot, &self.chain, current_joints)?;
                let offset_iso = nalgebra::Isometry3::translation(offset.x, offset.y, offset.z);
                let target_iso = current_pose.0 * offset_iso;
                Ok(Goal::Pose(Pose(target_iso)))
            }
        }
    }
}

impl CollisionChecker for Planner {
    fn is_in_collision(&self, joints: &[f64]) -> bool {
        self.is_in_collision(joints)
    }
}

/// Planning result from the facade.
#[derive(Debug, Clone)]
pub struct PlanningResult {
    /// Waypoints (joint configurations) along the path.
    pub waypoints: Vec<Vec<f64>>,
    /// Total planning time.
    pub planning_time: std::time::Duration,
    /// Number of planner iterations.
    pub iterations: usize,
    /// Total nodes in trees (RRT-specific).
    pub tree_size: usize,
    /// Which planner was used.
    pub planner_used: PlannerType,
}

impl PlanningResult {
    /// Number of waypoints in the path.
    pub fn num_waypoints(&self) -> usize {
        self.waypoints.len()
    }

    /// Convert to a joint-space path length.
    pub fn path_length(&self) -> f64 {
        shortcut::path_length(&self.waypoints)
    }

    /// Get the first waypoint.
    pub fn start(&self) -> Option<&[f64]> {
        self.waypoints.first().map(|w| w.as_slice())
    }

    /// Get the last waypoint.
    pub fn end(&self) -> Option<&[f64]> {
        self.waypoints.last().map(|w| w.as_slice())
    }
}

/// Auto-detect kinematic chain from robot model.
///
/// Strategy:
/// 1. If robot has planning groups, use the first group's chain.
/// 2. Otherwise, find the chain from root to the farthest leaf link.
fn auto_detect_chain(robot: &Robot) -> Result<KinematicChain> {
    // Try planning groups first
    if let Some((_, group)) = robot.groups.iter().next() {
        return KinematicChain::extract(robot, &group.base_link, &group.tip_link);
    }

    // Fall back: root link → farthest leaf
    if robot.links.is_empty() {
        return Err(KineticError::NoLinks);
    }

    let root_name = &robot.links[0].name;

    // Find leaf links (links with no children)
    let mut has_child = vec![false; robot.links.len()];
    for joint in &robot.joints {
        has_child[joint.parent_link] = true;
    }

    // Find the leaf link that is farthest from root (most joints in chain)
    let mut best_leaf = robot.links.len() - 1;
    let mut best_depth = 0;

    for (i, _link) in robot.links.iter().enumerate() {
        if has_child[i] {
            continue; // not a leaf
        }

        // Count depth by walking to root
        let mut depth = 0;
        let mut current = i;
        while let Some(joint_idx) = robot.links[current].parent_joint {
            depth += 1;
            current = robot.joints[joint_idx].parent_link;
        }

        if depth > best_depth {
            best_depth = depth;
            best_leaf = i;
        }
    }

    let tip_name = &robot.links[best_leaf].name;
    KinematicChain::extract(robot, root_name, tip_name)
}

/// One-line planning convenience function.
///
/// ```ignore
/// let result = kinetic_planning::plan("ur5e", &start_joints, Goal::Joints(goal))?;
/// ```
pub fn plan(robot_name: &str, start: &[f64], goal: &Goal) -> Result<PlanningResult> {
    let robot = Robot::from_name(robot_name)?;
    let planner = Planner::new(&robot)?;
    planner.plan(start, goal)
}

/// Plan with scene-based collision avoidance.
///
/// ```ignore
/// let result = kinetic_planning::plan_with_scene("ur5e", &start, &goal, &scene)?;
/// ```
pub fn plan_with_scene(
    robot_name: &str,
    start: &[f64],
    goal: &Goal,
    scene: &Scene,
) -> Result<PlanningResult> {
    let robot = Robot::from_name(robot_name)?;
    let planner = Planner::new(&robot)?.with_scene(scene);
    planner.plan(start, goal)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kinetic_collision::capt::AABB;
    use kinetic_collision::SpheresSoA;
    use kinetic_core::JointValues;

    #[test]
    fn planner_new_ur5e() {
        let robot = Robot::from_name("ur5e").unwrap();
        let planner = Planner::new(&robot).unwrap();
        assert_eq!(planner.chain().active_joints.len(), 6);
    }

    #[test]
    fn planner_plan_joint_goal() {
        let robot = Robot::from_name("ur5e").unwrap();
        let planner = Planner::new(&robot).unwrap().with_config(PlannerConfig {
            shortcut_iterations: 0,
            smooth: false,
            ..PlannerConfig::default()
        });

        let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let goal = Goal::Joints(JointValues(vec![0.5, -0.5, 0.3, 0.1, -0.2, 0.4]));

        let result = planner.plan(&start, &goal).unwrap();
        assert!(result.num_waypoints() >= 2);
        assert_eq!(result.planner_used, PlannerType::RRTConnect);
    }

    #[test]
    fn planner_plan_pose_goal() {
        let robot = Robot::from_name("ur5e").unwrap();
        let planner = Planner::new(&robot).unwrap().with_config(PlannerConfig {
            shortcut_iterations: 0,
            smooth: false,
            ..PlannerConfig::default()
        });

        let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let target_joints = vec![0.3, -0.8, 0.5, 0.0, 0.0, 0.0];
        let target_pose = planner.fk(&target_joints).unwrap();
        let goal = Goal::Pose(target_pose);

        let result = planner.plan(&start, &goal).unwrap();
        assert!(result.num_waypoints() >= 2);
    }

    #[test]
    fn planner_fk_ik_roundtrip() {
        let robot = Robot::from_name("ur5e").unwrap();
        let planner = Planner::new(&robot).unwrap();

        let joints = vec![0.3, -0.8, 0.5, 0.1, -0.2, 0.3];
        let pose = planner.fk(&joints).unwrap();
        let ik_joints = planner.ik(&pose).unwrap();

        // FK of IK result should match the target pose
        let recovered = planner.fk(&ik_joints).unwrap();
        let pos_err = (pose.translation() - recovered.translation()).norm();
        assert!(
            pos_err < 0.01,
            "FK/IK roundtrip position error: {}",
            pos_err
        );
    }

    #[test]
    fn planner_collision_check() {
        let robot = Robot::from_name("ur5e").unwrap();
        let planner = Planner::new(&robot).unwrap();

        // UR5e has no collision geometry, so nothing should collide
        let joints = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        assert!(!planner.is_in_collision(&joints));
    }

    #[test]
    fn planner_path_length() {
        let robot = Robot::from_name("ur5e").unwrap();
        let planner = Planner::new(&robot).unwrap().with_config(PlannerConfig {
            shortcut_iterations: 0,
            smooth: false,
            ..PlannerConfig::default()
        });

        let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let goal = Goal::Joints(JointValues(vec![0.5, -0.5, 0.3, 0.1, -0.2, 0.4]));

        let result = planner.plan(&start, &goal).unwrap();
        assert!(result.path_length() > 0.0);
    }

    #[test]
    fn planner_start_in_collision() {
        // Use robot with collision geometry
        let robot = Robot::from_urdf_string(COLLISION_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "base_link", "ee_link").unwrap();

        let mut obstacles = SpheresSoA::new();
        obstacles.push(0.0, 0.0, 0.0, 5.0, 0);
        let env = CollisionEnvironment::build(obstacles, 0.05, AABB::symmetric(10.0));

        let planner = Planner::with_chain(&robot, chain)
            .unwrap()
            .with_environment(env);

        let start = vec![0.0, 0.0, 0.0];
        let goal = Goal::Joints(JointValues(vec![1.0, -1.0, 0.5]));

        match planner.plan(&start, &goal) {
            Err(KineticError::StartInCollision) => {}
            other => panic!("Expected StartInCollision, got {:?}", other),
        }
    }

    #[test]
    fn one_line_plan() {
        let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
        let goal = Goal::Joints(JointValues(vec![0.5, -0.5, 0.3, 0.0, 0.0, 0.0]));

        // This should work without any setup beyond the robot name
        let result = plan("ur5e", &start, &goal).unwrap();
        assert!(result.num_waypoints() >= 2);
    }

    #[test]
    fn auto_detect_chain_ur5e() {
        let robot = Robot::from_name("ur5e").unwrap();
        let chain = auto_detect_chain(&robot).unwrap();
        // UR5e has 6 active joints
        assert_eq!(chain.active_joints.len(), 6);
    }

    /// Gap 4: Facade goal resolution dispatch — all 4 Goal variants succeed
    /// for franka_panda (7-DOF robot with "home" named pose).
    #[test]
    fn facade_goal_joints_dispatch() {
        let robot = Robot::from_name("franka_panda").unwrap();
        let planner = Planner::new(&robot).unwrap().with_config(PlannerConfig {
            shortcut_iterations: 0,
            smooth: false,
            ..PlannerConfig::default()
        });

        let start = vec![0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854];
        let goal = Goal::Joints(JointValues(vec![0.2, -0.5, 0.1, -2.0, 0.1, 1.3, 0.5]));

        let result = planner.plan(&start, &goal);
        assert!(result.is_ok(), "Goal::Joints should succeed: {:?}", result.err());
        assert!(result.unwrap().num_waypoints() >= 2);
    }

    #[test]
    fn facade_goal_pose_dispatch() {
        let robot = Robot::from_name("franka_panda").unwrap();
        let planner = Planner::new(&robot).unwrap().with_config(PlannerConfig {
            shortcut_iterations: 0,
            smooth: false,
            ..PlannerConfig::default()
        });

        let start = vec![0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854];
        // Use FK at a mid-range configuration to get a reachable pose
        let mid_joints = vec![0.2, -0.5, 0.1, -2.0, 0.1, 1.3, 0.5];
        let target_pose = planner.fk(&mid_joints).unwrap();
        let goal = Goal::Pose(target_pose);

        let result = planner.plan(&start, &goal);
        assert!(result.is_ok(), "Goal::Pose should succeed: {:?}", result.err());
        assert!(result.unwrap().num_waypoints() >= 2);
    }

    #[test]
    fn facade_goal_named_dispatch() {
        let robot = Robot::from_name("franka_panda").unwrap();
        let planner = Planner::new(&robot).unwrap().with_config(PlannerConfig {
            shortcut_iterations: 0,
            smooth: false,
            ..PlannerConfig::default()
        });

        // Start slightly away from home
        let start = vec![0.2, -0.5, 0.1, -2.0, 0.1, 1.3, 0.5];
        let goal = Goal::Named("home".into());

        let result = planner.plan(&start, &goal);
        assert!(result.is_ok(), "Goal::Named('home') should succeed: {:?}", result.err());
        assert!(result.unwrap().num_waypoints() >= 2);
    }

    #[test]
    fn facade_goal_relative_dispatch() {
        let robot = Robot::from_name("franka_panda").unwrap();
        let planner = Planner::new(&robot).unwrap().with_config(PlannerConfig {
            shortcut_iterations: 0,
            smooth: false,
            ..PlannerConfig::default()
        });

        let start = vec![0.0, -0.7854, 0.0, -2.3562, 0.0, 1.5708, 0.7854];
        // Small relative offset — FK + offset should be reachable
        let goal = Goal::Relative(nalgebra::Vector3::new(0.0, 0.0, -0.02));

        let result = planner.plan(&start, &goal);
        assert!(result.is_ok(), "Goal::Relative should succeed: {:?}", result.err());
        assert!(result.unwrap().num_waypoints() >= 2);
    }

    /// URDF with collision geometry for collision-aware tests.
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
}
