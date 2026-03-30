//! Trajectory execution monitoring.
//!
//! Runtime monitoring that compares actual joint positions against a planned
//! trajectory, detecting deviations that require warning or abort.
//!
//! # Example
//!
//! ```ignore
//! use kinetic_trajectory::monitor::{ExecutionMonitor, MonitorConfig, DeviationLevel};
//!
//! let monitor = ExecutionMonitor::new(trajectory, MonitorConfig::default());
//! // During execution loop:
//! match monitor.check(t, &actual_positions) {
//!     DeviationLevel::Normal => { /* continue */ }
//!     DeviationLevel::Warning { .. } => { /* log warning */ }
//!     DeviationLevel::Abort { .. } => { /* stop robot */ }
//! }
//! ```

use std::collections::VecDeque;
use std::time::Duration;

use crate::trapezoidal::TimedTrajectory;

/// Severity of deviation from the planned trajectory.
#[derive(Debug, Clone, PartialEq)]
pub enum DeviationLevel {
    /// Actual positions are within tolerance of the planned trajectory.
    Normal,
    /// Deviation exceeds warning threshold but not abort threshold.
    Warning {
        /// Joint index with the largest deviation.
        joint: usize,
        /// Deviation magnitude (radians).
        deviation: f64,
    },
    /// Deviation exceeds abort threshold — robot should stop.
    Abort {
        /// Joint index with the largest deviation.
        joint: usize,
        /// Deviation magnitude (radians).
        deviation: f64,
    },
}

/// Configuration for execution monitoring.
#[derive(Debug, Clone)]
pub struct MonitorConfig {
    /// Maximum allowed position deviation before abort (radians).
    /// Default: 0.1 rad (~5.7 degrees).
    pub position_tolerance: f64,
    /// Fraction of `position_tolerance` that triggers a warning.
    /// Default: 0.7 (warning at 70% of abort threshold).
    pub warning_fraction: f64,
    /// Number of samples to average for noise filtering.
    /// Set to 1 to disable filtering. Default: 5.
    pub noise_window: usize,
}

impl Default for MonitorConfig {
    fn default() -> Self {
        Self {
            position_tolerance: 0.1,
            warning_fraction: 0.7,
            noise_window: 5,
        }
    }
}

/// Runtime trajectory execution monitor.
///
/// Tracks deviations between planned and actual joint positions during
/// trajectory execution. Uses a sliding window to filter sensor noise
/// before comparing against thresholds.
pub struct ExecutionMonitor {
    trajectory: TimedTrajectory,
    config: MonitorConfig,
    /// Per-joint sliding window of recent deviation magnitudes.
    deviation_history: Vec<VecDeque<f64>>,
}

impl ExecutionMonitor {
    /// Create a new execution monitor for the given trajectory.
    pub fn new(trajectory: TimedTrajectory, config: MonitorConfig) -> Self {
        let dof = trajectory.dof;
        let deviation_history = vec![VecDeque::with_capacity(config.noise_window); dof];
        ExecutionMonitor {
            trajectory,
            config,
            deviation_history,
        }
    }

    /// DOF of the monitored trajectory.
    pub fn dof(&self) -> usize {
        self.trajectory.dof
    }

    /// Check actual positions against the planned trajectory at time `t`.
    ///
    /// Returns the deviation level. The deviation is computed as the
    /// filtered (windowed average) absolute position error per joint,
    /// then compared against warning and abort thresholds.
    pub fn check(&mut self, t: f64, actual_positions: &[f64]) -> DeviationLevel {
        let dof = self.trajectory.dof;

        if actual_positions.len() != dof {
            return DeviationLevel::Abort {
                joint: 0,
                deviation: f64::INFINITY,
            };
        }

        let planned = self.trajectory.sample_at(Duration::from_secs_f64(t));

        // Compute per-joint deviations and update history
        let mut worst_deviation = 0.0_f64;
        let mut worst_joint = 0;

        for (j, &actual_pos) in actual_positions.iter().enumerate().take(dof) {
            let deviation = (actual_pos - planned.positions[j]).abs();

            // Update sliding window
            let history = &mut self.deviation_history[j];
            if history.len() >= self.config.noise_window {
                history.pop_front();
            }
            history.push_back(deviation);

            // Filtered deviation: average of window
            let filtered = if history.is_empty() {
                deviation
            } else {
                history.iter().sum::<f64>() / history.len() as f64
            };

            if filtered > worst_deviation {
                worst_deviation = filtered;
                worst_joint = j;
            }
        }

        let abort_threshold = self.config.position_tolerance;
        let warning_threshold = self.config.position_tolerance * self.config.warning_fraction;

        if worst_deviation >= abort_threshold {
            DeviationLevel::Abort {
                joint: worst_joint,
                deviation: worst_deviation,
            }
        } else if worst_deviation >= warning_threshold {
            DeviationLevel::Warning {
                joint: worst_joint,
                deviation: worst_deviation,
            }
        } else {
            DeviationLevel::Normal
        }
    }

    /// Reset the deviation history (e.g., after a replan).
    pub fn reset(&mut self) {
        for history in &mut self.deviation_history {
            history.clear();
        }
    }

    /// Replace the monitored trajectory (e.g., after replanning).
    pub fn set_trajectory(&mut self, trajectory: TimedTrajectory) {
        self.deviation_history =
            vec![VecDeque::with_capacity(self.config.noise_window); trajectory.dof];
        self.trajectory = trajectory;
    }

    /// Get the raw (unfiltered) per-joint deviation at time `t`.
    pub fn raw_deviations(&self, t: f64, actual_positions: &[f64]) -> Vec<f64> {
        let planned = self.trajectory.sample_at(Duration::from_secs_f64(t));
        actual_positions
            .iter()
            .zip(&planned.positions)
            .map(|(a, p)| (a - p).abs())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trapezoidal::{TimedTrajectory, TimedWaypoint};

    fn simple_trajectory(dof: usize) -> TimedTrajectory {
        // Linear trajectory from 0 to 1 over 1 second, per joint
        TimedTrajectory {
            duration: Duration::from_secs(1),
            dof,
            waypoints: vec![
                TimedWaypoint {
                    time: 0.0,
                    positions: vec![0.0; dof],
                    velocities: vec![1.0; dof],
                    accelerations: vec![0.0; dof],
                },
                TimedWaypoint {
                    time: 1.0,
                    positions: vec![1.0; dof],
                    velocities: vec![1.0; dof],
                    accelerations: vec![0.0; dof],
                },
            ],
        }
    }

    #[test]
    fn normal_tracking() {
        let traj = simple_trajectory(3);
        let mut monitor = ExecutionMonitor::new(traj, MonitorConfig::default());

        // Actual matches planned exactly
        let level = monitor.check(0.5, &[0.5, 0.5, 0.5]);
        assert_eq!(level, DeviationLevel::Normal);
    }

    #[test]
    fn warning_threshold() {
        let config = MonitorConfig {
            position_tolerance: 0.1,
            warning_fraction: 0.7,
            noise_window: 1, // no filtering
        };
        let traj = simple_trajectory(2);
        let mut monitor = ExecutionMonitor::new(traj, config);

        // Deviation of 0.08 > 0.07 (warning) but < 0.1 (abort)
        let level = monitor.check(0.5, &[0.58, 0.5]);
        match level {
            DeviationLevel::Warning { joint, deviation } => {
                assert_eq!(joint, 0);
                assert!((deviation - 0.08).abs() < 1e-6);
            }
            other => panic!("Expected Warning, got {:?}", other),
        }
    }

    #[test]
    fn abort_threshold() {
        let config = MonitorConfig {
            position_tolerance: 0.1,
            warning_fraction: 0.7,
            noise_window: 1,
        };
        let traj = simple_trajectory(2);
        let mut monitor = ExecutionMonitor::new(traj, config);

        // Deviation of 0.15 > 0.1 (abort)
        let level = monitor.check(0.5, &[0.65, 0.5]);
        match level {
            DeviationLevel::Abort { joint, deviation } => {
                assert_eq!(joint, 0);
                assert!((deviation - 0.15).abs() < 1e-6);
            }
            other => panic!("Expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn noise_filtering_prevents_false_abort() {
        let config = MonitorConfig {
            position_tolerance: 0.1,
            warning_fraction: 0.7,
            noise_window: 5,
        };
        let traj = simple_trajectory(1);
        let mut monitor = ExecutionMonitor::new(traj, config);

        // 4 samples on track, 1 noisy spike
        monitor.check(0.1, &[0.1]);
        monitor.check(0.2, &[0.2]);
        monitor.check(0.3, &[0.3]);
        monitor.check(0.4, &[0.4]);
        // Spike: deviation of 0.15 which would abort without filtering
        let level = monitor.check(0.5, &[0.65]);
        // Filtered average: (0 + 0 + 0 + 0 + 0.15) / 5 = 0.03 → Normal
        assert_eq!(level, DeviationLevel::Normal);
    }

    #[test]
    fn sustained_deviation_triggers_abort() {
        let config = MonitorConfig {
            position_tolerance: 0.1,
            warning_fraction: 0.7,
            noise_window: 3,
        };
        let traj = simple_trajectory(1);
        let mut monitor = ExecutionMonitor::new(traj, config);

        // 3 consecutive samples with deviation 0.12
        monitor.check(0.1, &[0.22]);
        monitor.check(0.2, &[0.32]);
        let level = monitor.check(0.3, &[0.42]);
        // All 3 deviations are 0.12 → average is 0.12 > 0.1 → Abort
        match level {
            DeviationLevel::Abort { deviation, .. } => {
                assert!((deviation - 0.12).abs() < 1e-6);
            }
            other => panic!("Expected Abort, got {:?}", other),
        }
    }

    #[test]
    fn wrong_dof_aborts() {
        let traj = simple_trajectory(3);
        let mut monitor = ExecutionMonitor::new(traj, MonitorConfig::default());

        // Wrong number of joints
        let level = monitor.check(0.5, &[0.5, 0.5]);
        assert!(matches!(level, DeviationLevel::Abort { .. }));
    }

    #[test]
    fn reset_clears_history() {
        let config = MonitorConfig {
            position_tolerance: 0.1,
            warning_fraction: 0.7,
            noise_window: 5,
        };
        let traj = simple_trajectory(1);
        let mut monitor = ExecutionMonitor::new(traj, config);

        // Build up deviation history
        for i in 0..5 {
            let t = i as f64 * 0.1;
            monitor.check(t, &[t + 0.12]);
        }

        // Reset should clear
        monitor.reset();

        // After reset, a single on-track sample should be Normal
        let level = monitor.check(0.5, &[0.5]);
        assert_eq!(level, DeviationLevel::Normal);
    }

    #[test]
    fn set_trajectory_resets_state() {
        let config = MonitorConfig {
            position_tolerance: 0.1,
            warning_fraction: 0.7,
            noise_window: 1,
        };
        let traj1 = simple_trajectory(2);
        let mut monitor = ExecutionMonitor::new(traj1, config);

        // Deviation on old trajectory
        let level = monitor.check(0.5, &[0.65, 0.5]);
        assert!(matches!(level, DeviationLevel::Abort { .. }));

        // Set new trajectory where 0.65 is on-track at t=0.5
        let traj2 = TimedTrajectory {
            duration: Duration::from_secs(1),
            dof: 2,
            waypoints: vec![
                TimedWaypoint {
                    time: 0.0,
                    positions: vec![0.3, 0.0],
                    velocities: vec![0.7, 1.0],
                    accelerations: vec![0.0, 0.0],
                },
                TimedWaypoint {
                    time: 1.0,
                    positions: vec![1.0, 1.0],
                    velocities: vec![0.7, 1.0],
                    accelerations: vec![0.0, 0.0],
                },
            ],
        };
        monitor.set_trajectory(traj2);
        let level = monitor.check(0.5, &[0.65, 0.5]);
        assert_eq!(level, DeviationLevel::Normal);
    }

    #[test]
    fn raw_deviations_correct() {
        let traj = simple_trajectory(3);
        let monitor = ExecutionMonitor::new(traj, MonitorConfig::default());

        let devs = monitor.raw_deviations(0.5, &[0.6, 0.5, 0.4]);
        assert!((devs[0] - 0.1).abs() < 1e-6);
        assert!(devs[1].abs() < 1e-6);
        assert!((devs[2] - 0.1).abs() < 1e-6);
    }

    #[test]
    fn configurable_tolerance_at_boundary() {
        let config = MonitorConfig {
            position_tolerance: 0.05,
            warning_fraction: 0.8,
            noise_window: 1,
        };
        let traj = simple_trajectory(1);
        let mut monitor = ExecutionMonitor::new(traj, config);

        // Exactly at warning threshold: 0.05 * 0.8 = 0.04
        let level = monitor.check(0.5, &[0.54]);
        match level {
            DeviationLevel::Warning { joint, deviation } => {
                assert_eq!(joint, 0);
                assert!((deviation - 0.04).abs() < 1e-10);
            }
            other => panic!("Expected Warning, got {:?}", other),
        }

        // Just below warning
        let mut monitor2 = ExecutionMonitor::new(
            simple_trajectory(1),
            MonitorConfig {
                position_tolerance: 0.05,
                warning_fraction: 0.8,
                noise_window: 1,
            },
        );
        let level = monitor2.check(0.5, &[0.539]);
        assert_eq!(level, DeviationLevel::Normal);
    }

    #[test]
    fn identifies_worst_joint() {
        let config = MonitorConfig {
            position_tolerance: 0.1,
            warning_fraction: 0.5,
            noise_window: 1,
        };
        let traj = simple_trajectory(3);
        let mut monitor = ExecutionMonitor::new(traj, config);

        // Joint 1 has worst deviation
        let level = monitor.check(0.5, &[0.5, 0.62, 0.53]);
        match level {
            DeviationLevel::Abort { joint, .. } => assert_eq!(joint, 1),
            other => panic!("Expected Abort on joint 1, got {:?}", other),
        }
    }
}
