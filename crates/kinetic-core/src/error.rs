//! Unified error types for the KINETIC stack.

use std::time::Duration;

/// Unified error type for all KINETIC operations.
///
/// Every crate in the stack returns `KineticError` so callers get consistent,
/// actionable error messages without downcasting.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum KineticError {
    /// URDF file could not be parsed.
    #[error("URDF parse error: {0}")]
    UrdfParse(String),

    /// Inverse kinematics did not converge within the iteration budget.
    #[error("IK did not converge after {iterations} iterations (residual: {residual:.6})")]
    IKNotConverged {
        /// Number of iterations executed.
        iterations: usize,
        /// Final position+orientation residual.
        residual: f64,
    },

    /// Motion planner exceeded its time budget.
    #[error("Planning timed out after {elapsed:?} ({iterations} iterations)")]
    PlanningTimeout {
        /// Wall-clock time spent planning.
        elapsed: Duration,
        /// Number of planner iterations completed.
        iterations: usize,
    },

    /// The start configuration is in collision.
    #[error("Start configuration is in collision")]
    StartInCollision,

    /// The goal configuration is in collision.
    #[error("Goal configuration is in collision")]
    GoalInCollision,

    /// The goal is kinematically unreachable.
    #[error("Goal is unreachable")]
    GoalUnreachable,

    /// No valid IK solution exists for the target pose.
    #[error("No valid IK solution found for target pose")]
    NoIKSolution,

    /// A joint value exceeds its URDF limits.
    #[error("Joint '{name}' value {value} outside limits [{min}, {max}]")]
    JointLimitViolation {
        /// Joint name from the URDF.
        name: String,
        /// Offending joint value.
        value: f64,
        /// Lower limit.
        min: f64,
        /// Upper limit.
        max: f64,
    },

    /// Robot configuration file not found.
    #[error("Robot config not found: {0}")]
    RobotConfigNotFound(String),

    /// Cartesian path planner could not complete the full path.
    #[error("Cartesian path only achieved {fraction:.1}% of requested path")]
    CartesianPathIncomplete {
        /// Fraction of the path successfully planned (0.0–100.0).
        fraction: f64,
    },

    /// Collision detected during trajectory execution.
    #[error("Collision detected at waypoint {waypoint_index}")]
    CollisionDetected {
        /// Waypoint index where collision was found.
        waypoint_index: usize,
    },

    /// Trajectory velocity or acceleration limit exceeded.
    #[error("Trajectory limit exceeded at waypoint {waypoint_index}: {detail}")]
    TrajectoryLimitExceeded {
        /// Waypoint index.
        waypoint_index: usize,
        /// Description of which limit was exceeded.
        detail: String,
    },

    /// Robot has no links (empty URDF).
    #[error("Robot has no links")]
    NoLinks,

    /// A named link was not found in the robot model.
    #[error("Link '{0}' not found")]
    LinkNotFound(String),

    /// A named joint was not found in the robot model.
    #[error("Joint '{0}' not found")]
    JointNotFound(String),

    /// A named configuration was not found in the robot config.
    #[error("Named configuration '{0}' not found")]
    NamedConfigNotFound(String),

    /// SRDF file could not be parsed.
    #[error("SRDF parse error: {0}")]
    SrdfParse(String),

    /// Kinematic chain extraction failed.
    #[error("Chain extraction failed: {0}")]
    ChainExtraction(String),

    /// Robot kinematics are incompatible with the requested solver.
    #[error("Incompatible kinematics: {0}")]
    IncompatibleKinematics(String),

    /// Goal type not supported by the requested planner.
    #[error("Unsupported goal type: {0}")]
    UnsupportedGoal(String),

    /// Planning failed (no path found).
    #[error("Planning failed: {0}")]
    PlanningFailed(String),

    /// Array/vector dimension mismatch (e.g., wrong number of joints).
    #[error("Dimension mismatch in {context}: expected {expected}, got {got}")]
    DimensionMismatch {
        expected: usize,
        got: usize,
        context: String,
    },

    /// Servo numerical failure — pseudoinverse failed repeatedly at a singularity.
    ///
    /// Recovery: move the robot away from the singular configuration.
    #[error("Singularity lockup: pseudoinverse failed {consecutive_failures} consecutive times")]
    SingularityLockup { consecutive_failures: usize },

    /// Planner output failed internal validation (safety gate).
    #[error("Planner output invalid at waypoint {waypoint}: {reason}")]
    PlannerOutputInvalid { waypoint: usize, reason: String },

    /// Catch-all for errors that don't fit other variants.
    #[error("{0}")]
    Other(String),
}

/// Convenience type alias used throughout the KINETIC stack.
pub type Result<T> = std::result::Result<T, KineticError>;

impl KineticError {
    /// Whether this error indicates the planner should retry with different parameters.
    pub fn is_retryable(&self) -> bool {
        matches!(
            self,
            KineticError::PlanningTimeout { .. }
                | KineticError::IKNotConverged { .. }
                | KineticError::CartesianPathIncomplete { .. }
        )
    }

    /// Whether this error indicates a problem with the input (not transient).
    pub fn is_input_error(&self) -> bool {
        matches!(
            self,
            KineticError::UrdfParse(_)
                | KineticError::SrdfParse(_)
                | KineticError::JointLimitViolation { .. }
                | KineticError::RobotConfigNotFound(_)
                | KineticError::GoalUnreachable
                | KineticError::NoLinks
                | KineticError::LinkNotFound(_)
                | KineticError::JointNotFound(_)
                | KineticError::NamedConfigNotFound(_)
                | KineticError::ChainExtraction(_)
                | KineticError::IncompatibleKinematics(_)
                | KineticError::UnsupportedGoal(_)
                | KineticError::DimensionMismatch { .. }
        )
    }
}

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    use super::*;

    #[test]
    fn ik_not_converged_display() {
        let e = KineticError::IKNotConverged {
            iterations: 100,
            residual: 0.0023456,
        };
        let msg = format!("{e}");
        assert!(msg.contains("100 iterations"));
        assert!(msg.contains("0.002346")); // 6 decimal places
    }

    #[test]
    fn planning_timeout_display() {
        let e = KineticError::PlanningTimeout {
            elapsed: Duration::from_millis(50),
            iterations: 5000,
        };
        let msg = format!("{e}");
        assert!(msg.contains("50ms"));
        assert!(msg.contains("5000"));
    }

    #[test]
    fn joint_limit_display() {
        let e = KineticError::JointLimitViolation {
            name: "shoulder_pan".into(),
            value: 3.5,
            min: -3.14,
            max: 3.14,
        };
        let msg = format!("{e}");
        assert!(msg.contains("shoulder_pan"));
        assert!(msg.contains("3.5"));
        assert!(msg.contains("-3.14"));
    }

    #[test]
    fn cartesian_incomplete_display() {
        let e = KineticError::CartesianPathIncomplete { fraction: 72.3 };
        let msg = format!("{e}");
        assert!(msg.contains("72.3%"));
    }

    #[test]
    fn retryable_errors() {
        assert!(KineticError::PlanningTimeout {
            elapsed: Duration::from_millis(10),
            iterations: 100,
        }
        .is_retryable());
        assert!(KineticError::IKNotConverged {
            iterations: 50,
            residual: 0.01,
        }
        .is_retryable());
        assert!(!KineticError::StartInCollision.is_retryable());
        assert!(!KineticError::GoalUnreachable.is_retryable());
    }

    #[test]
    fn input_errors() {
        assert!(KineticError::UrdfParse("bad xml".into()).is_input_error());
        assert!(KineticError::GoalUnreachable.is_input_error());
        assert!(!KineticError::PlanningTimeout {
            elapsed: Duration::from_millis(10),
            iterations: 100,
        }
        .is_input_error());
    }

    #[test]
    fn result_type_alias() {
        fn test_fn() -> Result<i32> {
            Ok(42)
        }
        assert_eq!(test_fn().unwrap(), 42);

        fn test_err() -> Result<i32> {
            Err(KineticError::NoIKSolution)
        }
        assert!(test_err().is_err());
    }

    #[test]
    fn collision_detected_display() {
        let e = KineticError::CollisionDetected { waypoint_index: 5 };
        let msg = format!("{e}");
        assert!(msg.contains("waypoint 5"));
    }
}
