//! Trajectory execution pipeline for KINETIC.
//!
//! Bridges the gap between motion planning and robot hardware. Provides:
//! - [`TrajectoryExecutor`] trait — abstract execution interface
//! - [`RealTimeExecutor`] — streams commands at precise rates via user callback
//! - [`SimExecutor`] — instant execution for testing (no real timing)
//! - [`LogExecutor`] — records all commands for inspection/replay
//!
//! # Quick Start
//!
//! ```ignore
//! use kinetic_execution::{RealTimeExecutor, ExecutionConfig, CommandSink};
//!
//! // Your hardware callback
//! struct MyRobot;
//! impl CommandSink for MyRobot {
//!     fn send_command(&mut self, positions: &[f64], velocities: &[f64]) -> Result<(), String> {
//!         // Send to your motor controller
//!         Ok(())
//!     }
//! }
//!
//! let executor = RealTimeExecutor::new(ExecutionConfig::default());
//! let result = executor.execute(&timed_trajectory, &mut MyRobot)?;
//! println!("Executed in {:.2}s", result.actual_duration);
//! ```

pub mod controller;
pub mod log;
pub mod realtime;
pub mod sim;
pub mod watchdog;

pub use log::LogExecutor;
pub use realtime::RealTimeExecutor;
pub use sim::SimExecutor;
pub use watchdog::{SafetyWatchdog, WatchdogAction, WatchdogConfig};

use std::time::Duration;

use kinetic_trajectory::TimedTrajectory;

/// User-provided callback for sending joint commands to hardware.
///
/// Implement this trait for your robot driver. The executor calls
/// `send_command` at the configured rate with interpolated positions
/// and velocities.
pub trait CommandSink {
    /// Send a joint command to the robot.
    ///
    /// `positions` and `velocities` have `dof` elements each.
    /// Return `Err` to signal a hardware fault (executor will abort).
    fn send_command(&mut self, positions: &[f64], velocities: &[f64]) -> Result<(), String>;
}

/// Optional feedback from the robot for deviation monitoring.
pub trait FeedbackSource {
    /// Read current actual joint positions from the robot.
    fn read_positions(&self) -> Option<Vec<f64>>;
}

/// Execution state machine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionState {
    /// Not executing.
    Idle,
    /// Currently streaming commands.
    Executing,
    /// Paused — holding last commanded position.
    Paused,
    /// Trajectory completed successfully.
    Completed,
    /// Execution aborted due to error.
    Error,
}

/// Configuration for trajectory execution.
#[derive(Debug, Clone)]
pub struct ExecutionConfig {
    /// Command rate in Hz (default: 500).
    pub rate_hz: f64,
    /// Maximum allowed position deviation before error (radians, default: 0.1).
    pub position_tolerance: f64,
    /// Maximum allowed velocity deviation (rad/s, default: 0.5).
    pub velocity_tolerance: f64,
    /// Execution timeout — abort if trajectory takes longer than expected (default: 2x duration).
    pub timeout_factor: f64,
    /// Joint position limits [(lower, upper)] for pre-execution validation.
    /// If set, all trajectory waypoints are checked before execution starts.
    /// Trajectories with out-of-limits waypoints are rejected.
    pub joint_limits: Option<Vec<(f64, f64)>>,
    /// Per-command timeout in milliseconds (default: 100ms).
    /// If `send_command()` doesn't return within this duration, execution aborts.
    /// Set to 0 to disable (NOT recommended for real robots).
    pub command_timeout_ms: u64,
    /// Whether to require feedback for deviation monitoring (default: false).
    /// When true, `execute_with_feedback()` returns error if feedback source is None.
    /// **Set to true for real robot deployments.**
    pub require_feedback: bool,
    /// Optional watchdog configuration. When set, the executor sends heartbeats
    /// to an external `SafetyWatchdog` every control cycle. If the executor loop
    /// hangs, the watchdog fires independently.
    /// **Set for real robot deployments.** See `watchdog::SafetyWatchdog`.
    pub watchdog: Option<watchdog::WatchdogConfig>,
}

impl ExecutionConfig {
    /// Create a safety-hardened config from a robot model.
    ///
    /// Auto-populates joint limits from the robot, enables feedback requirement,
    /// and configures a safety watchdog. **Use this for real robot deployments.**
    ///
    /// ```ignore
    /// let config = ExecutionConfig::safe(&robot);
    /// let executor = RealTimeExecutor::new(config);
    /// ```
    pub fn safe(robot: &kinetic_robot::Robot) -> Self {
        Self {
            rate_hz: 500.0,
            position_tolerance: 0.05,
            velocity_tolerance: 0.3,
            timeout_factor: 2.0,
            joint_limits: Some(
                robot.joint_limits.iter().map(|l| (l.lower, l.upper)).collect()
            ),
            command_timeout_ms: 50,
            require_feedback: true,
            watchdog: Some(watchdog::WatchdogConfig {
                heartbeat_timeout: std::time::Duration::from_millis(50),
                on_timeout: watchdog::WatchdogAction::ZeroVelocity,
                dof: robot.dof,
            }),
        }
    }

    /// Alias for `safe()`.
    pub fn for_robot(robot: &kinetic_robot::Robot) -> Self {
        Self::safe(robot)
    }
}

impl Default for ExecutionConfig {
    fn default() -> Self {
        Self {
            rate_hz: 500.0,
            position_tolerance: 0.1,
            velocity_tolerance: 0.5,
            timeout_factor: 2.0,
            joint_limits: None,
            command_timeout_ms: 100,
            require_feedback: false,
            watchdog: None,
        }
    }
}

/// Result of executing a trajectory.
#[derive(Debug, Clone)]
pub struct ExecutionResult {
    /// Final execution state.
    pub state: ExecutionState,
    /// Actual execution wall-clock duration.
    pub actual_duration: Duration,
    /// Expected trajectory duration.
    pub expected_duration: Duration,
    /// Maximum position deviation observed (if feedback was available).
    pub max_deviation: Option<f64>,
    /// Number of commands sent.
    pub commands_sent: usize,
    /// Final joint positions (last commanded).
    pub final_positions: Vec<f64>,
}

/// Abstract trajectory execution interface.
///
/// Implement this for custom execution backends (e.g., HORUS topics,
/// ROS2 action clients, direct serial drivers).
pub trait TrajectoryExecutor {
    /// Execute a timed trajectory, sending commands via the given sink.
    ///
    /// Blocks until execution completes, errors, or is aborted.
    fn execute(
        &self,
        trajectory: &TimedTrajectory,
        sink: &mut dyn CommandSink,
    ) -> Result<ExecutionResult, ExecutionError>;
}

/// Execution errors.
#[derive(Debug, thiserror::Error)]
pub enum ExecutionError {
    /// The trajectory is empty or invalid.
    #[error("invalid trajectory: {0}")]
    InvalidTrajectory(String),

    /// Hardware command failed.
    #[error("command send failed: {0}")]
    CommandFailed(String),

    /// Position deviation exceeded tolerance.
    #[error("deviation exceeded: {deviation:.4} > {tolerance:.4} at command {command_index}")]
    DeviationExceeded {
        deviation: f64,
        tolerance: f64,
        command_index: usize,
    },

    /// Execution timed out.
    #[error("execution timed out after {elapsed:?} (expected {expected:?})")]
    Timeout {
        elapsed: Duration,
        expected: Duration,
    },

    /// Execution was aborted by the user.
    #[error("execution aborted")]
    Aborted,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config() {
        let cfg = ExecutionConfig::default();
        assert_eq!(cfg.rate_hz, 500.0);
        assert_eq!(cfg.position_tolerance, 0.1);
        assert_eq!(cfg.timeout_factor, 2.0);
    }

    #[test]
    fn execution_state_transitions() {
        assert_ne!(ExecutionState::Idle, ExecutionState::Executing);
        assert_ne!(ExecutionState::Paused, ExecutionState::Completed);
        assert_eq!(ExecutionState::Error, ExecutionState::Error);
    }

    #[test]
    fn execution_result_construction() {
        let result = ExecutionResult {
            state: ExecutionState::Completed,
            actual_duration: Duration::from_secs(2),
            expected_duration: Duration::from_secs(2),
            max_deviation: Some(0.001),
            commands_sent: 1000,
            final_positions: vec![0.0; 6],
        };
        assert_eq!(result.commands_sent, 1000);
        assert!(result.max_deviation.unwrap() < 0.01);
    }
}
