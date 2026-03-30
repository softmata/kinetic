//! Trajectory validation for production safety.
//!
//! Validates timed trajectories before execution on real hardware,
//! checking for position jumps, velocity/acceleration limit violations,
//! jerk bounds, and dimensional consistency.
//!
//! # Example
//!
//! ```ignore
//! use kinetic_trajectory::validation::{TrajectoryValidator, ValidationConfig};
//!
//! let validator = TrajectoryValidator::new(
//!     &position_lower, &position_upper,
//!     &velocity_limits, &acceleration_limits,
//!     ValidationConfig::default(),
//! );
//! match validator.validate(&trajectory) {
//!     Ok(()) => println!("Trajectory is safe"),
//!     Err(violations) => eprintln!("{} violations found", violations.len()),
//! }
//! ```

use crate::trapezoidal::TimedTrajectory;

/// Type of trajectory violation detected.
#[derive(Debug, Clone, PartialEq)]
pub enum ViolationType {
    /// Position outside joint limits.
    PositionLimit,
    /// Position jump between consecutive waypoints exceeds threshold.
    PositionJump,
    /// Velocity exceeds joint velocity limit.
    VelocityLimit,
    /// Velocity discontinuity (acceleration exceeds limit between waypoints).
    VelocityDiscontinuity,
    /// Acceleration exceeds joint acceleration limit.
    AccelerationLimit,
    /// Jerk (rate of acceleration change) exceeds limit.
    JerkLimit,
    /// Dimensional inconsistency (wrong number of joints in a waypoint).
    DimensionMismatch,
}

/// A single trajectory violation at a specific waypoint and joint.
#[derive(Debug, Clone)]
pub struct TrajectoryViolation {
    /// Index of the waypoint where the violation occurred.
    pub waypoint_index: usize,
    /// Index of the joint that violated (or 0 for dimensional issues).
    pub joint_index: usize,
    /// Type of violation.
    pub violation_type: ViolationType,
    /// Actual value that triggered the violation.
    pub actual_value: f64,
    /// The limit that was exceeded.
    pub limit_value: f64,
}

/// Configuration for trajectory validation thresholds.
#[derive(Debug, Clone)]
pub struct ValidationConfig {
    /// Safety factor multiplied onto limits for tolerance (default: 1.05 = 5% margin).
    pub safety_factor: f64,
    /// Maximum allowed position jump between consecutive waypoints (radians).
    /// Default: 0.5 rad (~29 degrees).
    pub max_position_jump: f64,
    /// Maximum allowed jerk (rad/s^3). `None` disables jerk checking.
    pub max_jerk: Option<f64>,
}

impl Default for ValidationConfig {
    fn default() -> Self {
        Self {
            safety_factor: 1.05,
            max_position_jump: 0.5,
            max_jerk: None,
        }
    }
}

/// Trajectory validator with per-joint limits.
///
/// Validates `TimedTrajectory` against position, velocity, acceleration,
/// and jerk constraints before execution on hardware.
#[derive(Debug, Clone)]
pub struct TrajectoryValidator {
    /// Per-joint lower position limits.
    position_lower: Vec<f64>,
    /// Per-joint upper position limits.
    position_upper: Vec<f64>,
    /// Per-joint velocity limits (absolute value).
    velocity_limits: Vec<f64>,
    /// Per-joint acceleration limits (absolute value).
    acceleration_limits: Vec<f64>,
    /// Validation configuration.
    config: ValidationConfig,
}

impl TrajectoryValidator {
    /// Create a new validator with per-joint limits.
    ///
    /// All limit arrays must have the same length (DOF).
    pub fn new(
        position_lower: &[f64],
        position_upper: &[f64],
        velocity_limits: &[f64],
        acceleration_limits: &[f64],
        config: ValidationConfig,
    ) -> Self {
        Self {
            position_lower: position_lower.to_vec(),
            position_upper: position_upper.to_vec(),
            velocity_limits: velocity_limits.to_vec(),
            acceleration_limits: acceleration_limits.to_vec(),
            config,
        }
    }

    /// DOF (number of joints) this validator expects.
    pub fn dof(&self) -> usize {
        self.position_lower.len()
    }

    /// Validate a trajectory. Returns `Ok(())` if no violations, or
    /// `Err(violations)` with the complete list of violations found.
    pub fn validate(&self, traj: &TimedTrajectory) -> Result<(), Vec<TrajectoryViolation>> {
        let mut violations = Vec::new();

        // First: dimensional consistency
        if let Err(msg) = traj.validate() {
            violations.push(TrajectoryViolation {
                waypoint_index: 0,
                joint_index: 0,
                violation_type: ViolationType::DimensionMismatch,
                actual_value: 0.0,
                limit_value: 0.0,
            });
            let _ = msg;
        }

        if traj.waypoints.is_empty() {
            if violations.is_empty() {
                return Ok(());
            } else {
                return Err(violations);
            }
        }

        let dof = traj.dof;
        let sf = self.config.safety_factor;

        for (i, wp) in traj.waypoints.iter().enumerate() {
            // Skip if dimensions don't match (already flagged above)
            if wp.positions.len() != dof {
                continue;
            }

            for j in 0..dof.min(self.dof()) {
                // Position limit check
                if wp.positions[j] < self.position_lower[j] - 1e-6
                    || wp.positions[j] > self.position_upper[j] + 1e-6
                {
                    let limit = if wp.positions[j] < self.position_lower[j] {
                        self.position_lower[j]
                    } else {
                        self.position_upper[j]
                    };
                    violations.push(TrajectoryViolation {
                        waypoint_index: i,
                        joint_index: j,
                        violation_type: ViolationType::PositionLimit,
                        actual_value: wp.positions[j],
                        limit_value: limit,
                    });
                }

                // Velocity limit check
                if j < self.velocity_limits.len()
                    && self.velocity_limits[j] > 0.0
                    && wp.velocities.len() > j
                    && wp.velocities[j].abs() > self.velocity_limits[j] * sf
                {
                    violations.push(TrajectoryViolation {
                        waypoint_index: i,
                        joint_index: j,
                        violation_type: ViolationType::VelocityLimit,
                        actual_value: wp.velocities[j].abs(),
                        limit_value: self.velocity_limits[j],
                    });
                }

                // Acceleration limit check
                if j < self.acceleration_limits.len()
                    && self.acceleration_limits[j] > 0.0
                    && wp.accelerations.len() > j
                    && wp.accelerations[j].abs() > self.acceleration_limits[j] * sf
                {
                    violations.push(TrajectoryViolation {
                        waypoint_index: i,
                        joint_index: j,
                        violation_type: ViolationType::AccelerationLimit,
                        actual_value: wp.accelerations[j].abs(),
                        limit_value: self.acceleration_limits[j],
                    });
                }
            }

            // Position jump check (between consecutive waypoints)
            // Time-aware: scale the allowed jump by the time step so sparse
            // trajectories (large dt between waypoints) aren't falsely flagged.
            if i > 0 {
                let prev = &traj.waypoints[i - 1];
                if prev.positions.len() == dof {
                    let dt = wp.time - prev.time;
                    // Scale factor: allow proportionally larger jumps for larger time steps.
                    // Reference dt = 0.01s (100 Hz sampling). At dt=1.0s, allow 100x the jump.
                    let dt_scale = if dt > 1e-12 {
                        (dt / 0.01).max(1.0)
                    } else {
                        1.0
                    };
                    let scaled_limit = self.config.max_position_jump * dt_scale;
                    for j in 0..dof.min(self.dof()) {
                        let jump = (wp.positions[j] - prev.positions[j]).abs();
                        if jump > scaled_limit {
                            violations.push(TrajectoryViolation {
                                waypoint_index: i,
                                joint_index: j,
                                violation_type: ViolationType::PositionJump,
                                actual_value: jump,
                                limit_value: scaled_limit,
                            });
                        }
                    }

                    // Velocity discontinuity check: |dv| / dt should be <= accel limit
                    let dt = wp.time - prev.time;
                    if dt > 1e-12 {
                        for j in 0..dof.min(self.dof()) {
                            if wp.velocities.len() > j && prev.velocities.len() > j {
                                let dv = (wp.velocities[j] - prev.velocities[j]).abs();
                                let implied_accel = dv / dt;
                                if j < self.acceleration_limits.len()
                                    && self.acceleration_limits[j] > 0.0
                                    && implied_accel > self.acceleration_limits[j] * sf * 2.0
                                {
                                    violations.push(TrajectoryViolation {
                                        waypoint_index: i,
                                        joint_index: j,
                                        violation_type: ViolationType::VelocityDiscontinuity,
                                        actual_value: implied_accel,
                                        limit_value: self.acceleration_limits[j],
                                    });
                                }
                            }

                            // Jerk check: |da| / dt
                            if let Some(max_jerk) = self.config.max_jerk {
                                if wp.accelerations.len() > j && prev.accelerations.len() > j {
                                    let da = (wp.accelerations[j] - prev.accelerations[j]).abs();
                                    let implied_jerk = da / dt;
                                    if implied_jerk > max_jerk * sf {
                                        violations.push(TrajectoryViolation {
                                            waypoint_index: i,
                                            joint_index: j,
                                            violation_type: ViolationType::JerkLimit,
                                            actual_value: implied_jerk,
                                            limit_value: max_jerk,
                                        });
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }

        if violations.is_empty() {
            Ok(())
        } else {
            Err(violations)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trapezoidal::{trapezoidal, trapezoidal_per_joint, TimedWaypoint};
    use std::time::Duration;

    fn default_validator(dof: usize) -> TrajectoryValidator {
        TrajectoryValidator::new(
            &vec![-std::f64::consts::PI; dof],
            &vec![std::f64::consts::PI; dof],
            &vec![2.0; dof],
            &vec![4.0; dof],
            ValidationConfig::default(),
        )
    }

    #[test]
    fn valid_trajectory_passes() {
        let path = vec![vec![0.0, 0.0], vec![0.5, 1.0]];
        let traj = trapezoidal(&path, 1.0, 2.0).unwrap();
        let validator = default_validator(2);
        assert!(validator.validate(&traj).is_ok());
    }

    #[test]
    fn valid_per_joint_passes() {
        let path = vec![vec![0.0, 0.0], vec![0.5, 1.0]];
        let traj = trapezoidal_per_joint(&path, &[1.0, 0.5], &[2.0, 1.0]).unwrap();
        let validator = default_validator(2);
        assert!(validator.validate(&traj).is_ok());
    }

    #[test]
    fn empty_trajectory_passes() {
        let traj = TimedTrajectory {
            duration: Duration::ZERO,
            dof: 2,
            waypoints: vec![],
        };
        let validator = default_validator(2);
        assert!(validator.validate(&traj).is_ok());
    }

    #[test]
    fn position_limit_violation() {
        let traj = TimedTrajectory {
            duration: Duration::from_secs_f64(1.0),
            dof: 1,
            waypoints: vec![
                TimedWaypoint {
                    time: 0.0,
                    positions: vec![0.0],
                    velocities: vec![0.0],
                    accelerations: vec![0.0],
                },
                TimedWaypoint {
                    time: 1.0,
                    positions: vec![5.0], // exceeds +-3.14
                    velocities: vec![0.0],
                    accelerations: vec![0.0],
                },
            ],
        };
        let validator = default_validator(1);
        let err = validator.validate(&traj).unwrap_err();
        assert!(err
            .iter()
            .any(|v| v.violation_type == ViolationType::PositionLimit));
    }

    #[test]
    fn velocity_limit_violation() {
        let traj = TimedTrajectory {
            duration: Duration::from_secs_f64(1.0),
            dof: 1,
            waypoints: vec![TimedWaypoint {
                time: 0.0,
                positions: vec![0.0],
                velocities: vec![10.0], // exceeds limit of 2.0
                accelerations: vec![0.0],
            }],
        };
        let validator = default_validator(1);
        let err = validator.validate(&traj).unwrap_err();
        assert!(err
            .iter()
            .any(|v| v.violation_type == ViolationType::VelocityLimit));
    }

    #[test]
    fn acceleration_limit_violation() {
        let traj = TimedTrajectory {
            duration: Duration::from_secs_f64(1.0),
            dof: 1,
            waypoints: vec![TimedWaypoint {
                time: 0.0,
                positions: vec![0.0],
                velocities: vec![0.0],
                accelerations: vec![20.0], // exceeds limit of 4.0
            }],
        };
        let validator = default_validator(1);
        let err = validator.validate(&traj).unwrap_err();
        assert!(err
            .iter()
            .any(|v| v.violation_type == ViolationType::AccelerationLimit));
    }

    #[test]
    fn position_jump_violation() {
        let traj = TimedTrajectory {
            duration: Duration::from_secs_f64(0.01),
            dof: 1,
            waypoints: vec![
                TimedWaypoint {
                    time: 0.0,
                    positions: vec![0.0],
                    velocities: vec![0.0],
                    accelerations: vec![0.0],
                },
                TimedWaypoint {
                    time: 0.01,
                    positions: vec![2.0], // jump of 2.0 > max_position_jump (0.5)
                    velocities: vec![0.0],
                    accelerations: vec![0.0],
                },
            ],
        };
        let validator = default_validator(1);
        let err = validator.validate(&traj).unwrap_err();
        assert!(err
            .iter()
            .any(|v| v.violation_type == ViolationType::PositionJump));
    }

    #[test]
    fn jerk_limit_violation() {
        let traj = TimedTrajectory {
            duration: Duration::from_secs_f64(0.01),
            dof: 1,
            waypoints: vec![
                TimedWaypoint {
                    time: 0.0,
                    positions: vec![0.0],
                    velocities: vec![0.0],
                    accelerations: vec![0.0],
                },
                TimedWaypoint {
                    time: 0.01,
                    positions: vec![0.0],
                    velocities: vec![0.0],
                    accelerations: vec![10.0], // da=10 in dt=0.01 -> jerk=1000
                },
            ],
        };
        let validator = TrajectoryValidator::new(
            &[-std::f64::consts::PI],
            &[std::f64::consts::PI],
            &[2.0],
            &[20.0], // high enough to not trigger accel violation
            ValidationConfig {
                max_jerk: Some(100.0), // 1000 > 100
                ..Default::default()
            },
        );
        let err = validator.validate(&traj).unwrap_err();
        assert!(err
            .iter()
            .any(|v| v.violation_type == ViolationType::JerkLimit));
    }

    #[test]
    fn multiple_violations_reported() {
        let traj = TimedTrajectory {
            duration: Duration::from_secs_f64(1.0),
            dof: 2,
            waypoints: vec![TimedWaypoint {
                time: 0.0,
                positions: vec![5.0, -5.0],   // both out of range
                velocities: vec![10.0, 10.0], // both too fast
                accelerations: vec![0.0, 0.0],
            }],
        };
        let validator = default_validator(2);
        let err = validator.validate(&traj).unwrap_err();
        assert!(
            err.len() >= 4,
            "Expected at least 4 violations (2 pos + 2 vel), got {}",
            err.len()
        );
    }

    #[test]
    fn totp_trajectory_valid() {
        // Single-segment path avoids switching-point acceleration spikes
        let path = vec![vec![0.0, 0.0], vec![0.5, 1.0]];
        let vel = vec![2.0, 2.0];
        let acc = vec![4.0, 4.0];
        let traj = crate::totp::totp(&path, &vel, &acc, 0.01).unwrap();
        // TOTP operates in path-space; per-joint accel can be up to sqrt(dof) * path_accel.
        // For 2-DOF diagonal motion: sqrt(2) * 4.0 ≈ 5.66, so use 6.0.
        let validator = TrajectoryValidator::new(
            &[-std::f64::consts::PI, -std::f64::consts::PI],
            &[std::f64::consts::PI, std::f64::consts::PI],
            &[2.0, 2.0],
            &[6.0, 6.0],
            ValidationConfig::default(),
        );
        assert!(
            validator.validate(&traj).is_ok(),
            "TOTP trajectory should be valid"
        );
    }
}
