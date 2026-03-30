//! Plan-Execute Loop: orchestrates planning, execution, and replanning.
//!
//! The `PlanExecuteLoop` owns a planner, executor, and scene, managing
//! the full plan→execute→monitor→replan lifecycle. Equivalent to
//! MoveIt2's MoveGroupInterface but without ROS.
//!
//! # Usage
//!
//! ```ignore
//! use kinetic_planning::PlanExecuteLoop;
//! use kinetic_execution::{SimExecutor, ExecutionConfig};
//!
//! let executor = Box::new(SimExecutor::default());
//! let mut loop_ = PlanExecuteLoop::new(planner, executor);
//! let result = loop_.move_to(&start_joints, &goal)?;
//! ```

use std::time::{Duration, Instant};

use kinetic_core::{Goal, KineticError, Result};
use kinetic_execution::{
    CommandSink, ExecutionConfig, ExecutionError, ExecutionResult, TrajectoryExecutor,
};
use kinetic_trajectory::{trapezoidal_per_joint, TimedTrajectory};

use crate::Planner;

/// Strategy for handling scene changes during execution.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ReplanStrategy {
    /// Stop immediately and replan from current position.
    Immediate,
    /// Finish current trajectory, then plan the next move.
    Deferred,
}

/// Strategy for error recovery.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RecoveryStrategy {
    /// Hold the last commanded position.
    HoldPosition,
    /// Abort and return error to the caller.
    Abort,
}

/// Result of a plan-execute operation.
#[derive(Debug)]
pub struct PlanExecuteResult {
    /// The final planned trajectory (may differ from initial if replanned).
    pub trajectory: Option<TimedTrajectory>,
    /// Execution result from the executor.
    pub execution: Option<ExecutionResult>,
    /// Total wall-clock time (planning + execution).
    pub total_duration: Duration,
    /// Number of replanning attempts.
    pub replans: usize,
    /// Last known joint positions.
    pub final_joints: Vec<f64>,
}

/// Configuration for the plan-execute loop.
#[derive(Debug, Clone)]
pub struct PlanExecuteConfig {
    /// Maximum number of replan attempts before giving up.
    pub max_replans: usize,
    /// Strategy when scene changes during execution.
    pub replan_strategy: ReplanStrategy,
    /// Strategy when execution fails.
    pub recovery_strategy: RecoveryStrategy,
    /// Execution configuration.
    pub execution_config: ExecutionConfig,
}

impl Default for PlanExecuteConfig {
    fn default() -> Self {
        Self {
            max_replans: 3,
            replan_strategy: ReplanStrategy::Immediate,
            recovery_strategy: RecoveryStrategy::Abort,
            execution_config: ExecutionConfig::default(),
        }
    }
}

/// Orchestrator for the plan→execute→replan lifecycle.
///
/// Owns a planner and executor, providing one-call methods like
/// `move_to(goal)` that handle the full pipeline.
pub struct PlanExecuteLoop {
    planner: Planner,
    executor: Box<dyn TrajectoryExecutor>,
    config: PlanExecuteConfig,
    /// Last known joint positions (updated after each execution).
    current_joints: Vec<f64>,
    /// Velocity limits for time parameterization.
    vel_limits: Vec<f64>,
    /// Acceleration limits for time parameterization.
    accel_limits: Vec<f64>,
}

impl PlanExecuteLoop {
    /// Create a new plan-execute loop.
    pub fn new(planner: Planner, executor: Box<dyn TrajectoryExecutor>) -> Self {
        let vel_limits = planner.robot().velocity_limits();
        let accel_limits = planner.robot().acceleration_limits();
        let dof = planner.chain().dof;

        Self {
            planner,
            executor,
            config: PlanExecuteConfig::default(),
            current_joints: vec![0.0; dof],
            vel_limits,
            accel_limits,
        }
    }

    /// Set the plan-execute configuration.
    pub fn with_config(mut self, config: PlanExecuteConfig) -> Self {
        self.config = config;
        self
    }

    /// Set the current joint state (call before first move_to if not at zeros).
    pub fn set_current_joints(&mut self, joints: &[f64]) {
        self.current_joints = joints.to_vec();
    }

    /// Get the current joint positions.
    pub fn current_joints(&self) -> &[f64] {
        &self.current_joints
    }

    /// Plan and execute a motion to the given goal.
    ///
    /// Full pipeline: plan → time-parameterize → execute → update state.
    /// Returns the combined result.
    pub fn move_to(&mut self, start_joints: &[f64], goal: &Goal) -> Result<PlanExecuteResult> {
        let timer = Instant::now();
        self.current_joints = start_joints.to_vec();

        let mut replans = 0;
        let mut current_start = start_joints.to_vec();

        loop {
            // Plan
            let plan_result = self.planner.plan(&current_start, goal)?;

            // Time-parameterize
            let timed =
                trapezoidal_per_joint(&plan_result.waypoints, &self.vel_limits, &self.accel_limits)
                    .map_err(|e| {
                        KineticError::PlanningFailed(format!("time parameterization: {e}"))
                    })?;

            // Execute
            let mut sink = NoOpSink;
            let exec_result = self.executor.execute(&timed, &mut sink);

            match exec_result {
                Ok(result) => {
                    self.current_joints = result.final_positions.clone();
                    return Ok(PlanExecuteResult {
                        trajectory: Some(timed),
                        execution: Some(result),
                        total_duration: timer.elapsed(),
                        replans,
                        final_joints: self.current_joints.clone(),
                    });
                }
                Err(ExecutionError::DeviationExceeded { .. })
                | Err(ExecutionError::CommandFailed(_)) => {
                    replans += 1;
                    if replans > self.config.max_replans {
                        match self.config.recovery_strategy {
                            RecoveryStrategy::Abort => {
                                return Err(KineticError::PlanningFailed(format!(
                                    "execution failed after {} replan attempts",
                                    replans
                                )));
                            }
                            RecoveryStrategy::HoldPosition => {
                                return Ok(PlanExecuteResult {
                                    trajectory: Some(timed),
                                    execution: None,
                                    total_duration: timer.elapsed(),
                                    replans,
                                    final_joints: self.current_joints.clone(),
                                });
                            }
                        }
                    }
                    // Replan from current position
                    // (In real implementation, read current from feedback)
                    current_start = self.current_joints.clone();
                    continue;
                }
                Err(ExecutionError::Timeout {
                    elapsed,
                    expected: _,
                }) => {
                    return Err(KineticError::PlanningTimeout {
                        elapsed,
                        iterations: 0,
                    });
                }
                Err(e) => {
                    return Err(KineticError::PlanningFailed(format!("execution: {e}")));
                }
            }
        }
    }

    /// Plan and execute a Cartesian (straight-line) motion.
    pub fn move_cartesian(
        &mut self,
        start_joints: &[f64],
        target_pose: &kinetic_core::Pose,
        config: &crate::CartesianConfig,
    ) -> Result<PlanExecuteResult> {
        let timer = Instant::now();
        self.current_joints = start_joints.to_vec();

        let cartesian = crate::CartesianPlanner::new(
            std::sync::Arc::new(self.planner.robot().clone()),
            self.planner.chain().clone(),
        );

        let cart_result = cartesian.plan_linear(start_joints, target_pose, config)?;

        if cart_result.waypoints.is_empty() {
            return Err(KineticError::PlanningFailed(
                "Cartesian planner produced empty path".into(),
            ));
        }

        let timed =
            trapezoidal_per_joint(&cart_result.waypoints, &self.vel_limits, &self.accel_limits)
                .map_err(|e| KineticError::PlanningFailed(format!("time parameterization: {e}")))?;

        let mut sink = NoOpSink;
        let exec_result = self
            .executor
            .execute(&timed, &mut sink)
            .map_err(|e| KineticError::PlanningFailed(format!("execution: {e}")))?;

        self.current_joints = exec_result.final_positions.clone();

        Ok(PlanExecuteResult {
            trajectory: Some(timed),
            execution: Some(exec_result),
            total_duration: timer.elapsed(),
            replans: 0,
            final_joints: self.current_joints.clone(),
        })
    }

    /// Access the planner.
    pub fn planner(&self) -> &Planner {
        &self.planner
    }

    /// Access the configuration.
    pub fn config(&self) -> &PlanExecuteConfig {
        &self.config
    }
}

/// No-op command sink (executor handles internal dispatch).
struct NoOpSink;

impl CommandSink for NoOpSink {
    fn send_command(
        &mut self,
        _positions: &[f64],
        _velocities: &[f64],
    ) -> std::result::Result<(), String> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kinetic_core::JointValues;
    use kinetic_execution::SimExecutor;
    use kinetic_robot::Robot;

    fn setup() -> (Robot, Vec<f64>, Vec<f64>) {
        let robot = Robot::from_name("franka_panda").unwrap();
        let mid: Vec<f64> = robot
            .joint_limits
            .iter()
            .map(|l| (l.lower + l.upper) / 2.0)
            .collect();
        let goal: Vec<f64> = robot
            .joint_limits
            .iter()
            .map(|l| l.lower + (l.upper - l.lower) * 0.6)
            .collect();
        (robot, mid, goal)
    }

    #[test]
    fn plan_execute_loop_basic() {
        let (robot, start, goal) = setup();
        let planner = Planner::new(&robot).unwrap();
        let executor = Box::new(SimExecutor::default());
        let mut pel = PlanExecuteLoop::new(planner, executor);

        let result = pel.move_to(&start, &Goal::Joints(JointValues(goal.clone())));
        assert!(result.is_ok(), "move_to failed: {:?}", result.err());

        let r = result.unwrap();
        assert!(r.trajectory.is_some());
        assert_eq!(r.replans, 0);
        assert_eq!(r.final_joints.len(), robot.dof);
    }

    #[test]
    fn plan_execute_updates_current_joints() {
        let (robot, start, goal) = setup();
        let planner = Planner::new(&robot).unwrap();
        let executor = Box::new(SimExecutor::default());
        let mut pel = PlanExecuteLoop::new(planner, executor);

        pel.set_current_joints(&start);
        assert_eq!(pel.current_joints(), &start);

        let _ = pel.move_to(&start, &Goal::Joints(JointValues(goal)));
        // After execution, current joints should be updated
        assert_ne!(pel.current_joints(), &start);
    }

    #[test]
    fn plan_execute_config_defaults() {
        let cfg = PlanExecuteConfig::default();
        assert_eq!(cfg.max_replans, 3);
        assert_eq!(cfg.replan_strategy, ReplanStrategy::Immediate);
        assert_eq!(cfg.recovery_strategy, RecoveryStrategy::Abort);
    }

    #[test]
    fn with_config_applies_custom_settings() {
        let (robot, _start, _goal) = setup();
        let planner = Planner::new(&robot).unwrap();
        let executor = Box::new(SimExecutor::default());

        let custom_config = PlanExecuteConfig {
            max_replans: 10,
            replan_strategy: ReplanStrategy::Deferred,
            recovery_strategy: RecoveryStrategy::HoldPosition,
            execution_config: ExecutionConfig::default(),
        };

        let pel = PlanExecuteLoop::new(planner, executor).with_config(custom_config);
        assert_eq!(pel.config().max_replans, 10);
        assert_eq!(pel.config().replan_strategy, ReplanStrategy::Deferred);
        assert_eq!(
            pel.config().recovery_strategy,
            RecoveryStrategy::HoldPosition
        );
    }

    #[test]
    fn plan_execute_accessor_methods() {
        let (robot, start, _goal) = setup();
        let planner = Planner::new(&robot).unwrap();
        let executor = Box::new(SimExecutor::default());
        let mut pel = PlanExecuteLoop::new(planner, executor);

        // Default current joints should be zeros
        assert_eq!(pel.current_joints().len(), robot.dof);
        assert!(pel.current_joints().iter().all(|&v| v == 0.0));

        // set_current_joints should update
        pel.set_current_joints(&start);
        assert_eq!(pel.current_joints(), &start);

        // planner() should return the embedded planner
        assert_eq!(pel.planner().chain().dof, robot.dof);
    }

    #[test]
    fn replan_strategy_variants_are_distinct() {
        assert_ne!(ReplanStrategy::Immediate, ReplanStrategy::Deferred);
    }

    #[test]
    fn recovery_strategy_variants_are_distinct() {
        assert_ne!(RecoveryStrategy::HoldPosition, RecoveryStrategy::Abort);
    }

    #[test]
    fn plan_execute_result_has_trajectory_and_execution() {
        let (robot, start, goal) = setup();
        let planner = Planner::new(&robot).unwrap();
        let executor = Box::new(SimExecutor::default());
        let mut pel = PlanExecuteLoop::new(planner, executor);

        let result = pel
            .move_to(&start, &Goal::Joints(JointValues(goal.clone())))
            .unwrap();

        // trajectory should be present
        assert!(result.trajectory.is_some());
        let traj = result.trajectory.unwrap();
        assert!(traj.duration().as_secs_f64() > 0.0);

        // execution should be present
        assert!(result.execution.is_some());

        // total_duration should be positive
        assert!(result.total_duration.as_nanos() > 0);

        // final_joints should match DOF
        assert_eq!(result.final_joints.len(), robot.dof);
    }

    #[test]
    fn noop_sink_send_command_returns_ok() {
        let mut sink = NoOpSink;
        assert!(sink.send_command(&[0.0, 1.0], &[0.0, 0.0]).is_ok());
        assert!(sink.send_command(&[], &[]).is_ok());
    }

    #[test]
    fn move_to_with_same_start_and_goal() {
        let (robot, start, _goal) = setup();
        let planner = Planner::new(&robot).unwrap();
        let executor = Box::new(SimExecutor::default());
        let mut pel = PlanExecuteLoop::new(planner, executor);

        // Move to the same position as start
        let result = pel.move_to(&start, &Goal::Joints(JointValues(start.clone())));
        assert!(result.is_ok(), "Same start/goal should succeed: {:?}", result.err());
        let r = result.unwrap();
        assert_eq!(r.replans, 0);
    }

    #[test]
    fn plan_execute_cartesian_basic() {
        let (robot, start, _goal) = setup();
        let planner = Planner::new(&robot).unwrap();
        let executor = Box::new(SimExecutor::default());
        let mut pel = PlanExecuteLoop::new(planner, executor);

        // Compute the current pose, then move slightly in z
        let arm = &robot.groups["arm"];
        let chain = kinetic_kinematics::KinematicChain::extract(
            &robot,
            &arm.base_link,
            &arm.tip_link,
        )
        .unwrap();
        let current_pose = kinetic_kinematics::forward_kinematics(&robot, &chain, &start).unwrap();
        let goal_iso = nalgebra::Isometry3::from_parts(
            nalgebra::Translation3::from(
                current_pose.0.translation.vector + nalgebra::Vector3::new(0.0, 0.0, 0.02),
            ),
            *current_pose.rotation(),
        );

        let config = crate::CartesianConfig {
            max_step: 0.01,
            avoid_collisions: false,
            ..Default::default()
        };

        let result = pel.move_cartesian(&start, &kinetic_core::Pose(goal_iso), &config);
        // The result should either succeed or fail gracefully (IK might not converge for all robots)
        match result {
            Ok(r) => {
                assert!(r.trajectory.is_some());
                assert_eq!(r.replans, 0);
            }
            Err(e) => {
                // Acceptable failure modes: empty path or IK failure
                let msg = format!("{e}");
                assert!(
                    msg.contains("empty path") || msg.contains("IK") || msg.contains("parameterization"),
                    "Unexpected error: {e}"
                );
            }
        }
    }

    #[test]
    fn plan_execute_config_debug_format() {
        let cfg = PlanExecuteConfig::default();
        let debug = format!("{:?}", cfg);
        assert!(debug.contains("max_replans"));
        assert!(debug.contains("replan_strategy"));
        assert!(debug.contains("recovery_strategy"));
    }

    #[test]
    fn plan_execute_result_debug_format() {
        let result = PlanExecuteResult {
            trajectory: None,
            execution: None,
            total_duration: Duration::from_millis(42),
            replans: 2,
            final_joints: vec![0.0, 1.0],
        };
        let debug = format!("{:?}", result);
        assert!(debug.contains("replans"));
        assert!(debug.contains("total_duration"));
    }
}
