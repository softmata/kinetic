//! RealTimeExecutor — streams commands at precise rates.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use kinetic_trajectory::TimedTrajectory;

use crate::sim::interpolate_at;
use crate::{
    CommandSink, ExecutionConfig, ExecutionError, ExecutionResult, ExecutionState, FeedbackSource,
    TrajectoryExecutor,
};

/// Real-time trajectory executor that streams commands at the configured rate.
///
/// Uses a tight timing loop to send interpolated joint commands to the
/// user's `CommandSink` at the configured rate (e.g., 500Hz). Supports
/// pause/resume via an `ExecutionHandle`, and optional feedback monitoring.
pub struct RealTimeExecutor {
    config: ExecutionConfig,
}

impl RealTimeExecutor {
    pub fn new(config: ExecutionConfig) -> Self {
        Self { config }
    }

    /// Execute with optional position feedback for deviation monitoring.
    pub fn execute_with_feedback(
        &self,
        trajectory: &TimedTrajectory,
        sink: &mut dyn CommandSink,
        feedback: Option<&dyn FeedbackSource>,
    ) -> Result<ExecutionResult, ExecutionError> {
        if trajectory.waypoints.is_empty() {
            return Err(ExecutionError::InvalidTrajectory("empty trajectory".into()));
        }

        // Safety: require feedback if configured
        if self.config.require_feedback && feedback.is_none() {
            return Err(ExecutionError::InvalidTrajectory(
                "ExecutionConfig requires feedback source for safety monitoring, but none provided. \
                 Set require_feedback=false to execute without feedback (NOT recommended for real robots)."
                    .into(),
            ));
        }

        // Pre-execution safety: check all waypoints are within joint limits
        if self.config.joint_limits.is_none() {
            eprintln!(
                "[KINETIC WARNING] Executing trajectory without joint limit validation. \
                 Use ExecutionConfig::safe(&robot) for real robot deployments."
            );
        }
        if let Some(limits) = &self.config.joint_limits {
            for (wp_idx, wp) in trajectory.waypoints.iter().enumerate() {
                for (j, &pos) in wp.positions.iter().enumerate() {
                    if j < limits.len() {
                        let (lo, hi) = limits[j];
                        if pos < lo - 1e-6 || pos > hi + 1e-6 {
                            return Err(ExecutionError::InvalidTrajectory(format!(
                                "Waypoint {} joint {} position {:.4} outside limits [{:.4}, {:.4}]",
                                wp_idx, j, pos, lo, hi
                            )));
                        }
                    }
                }
            }
        }

        let dof = trajectory.dof;
        let dt = Duration::from_secs_f64(1.0 / self.config.rate_hz);
        let duration = trajectory.duration;
        let timeout = Duration::from_secs_f64(duration.as_secs_f64() * self.config.timeout_factor);

        let start_time = Instant::now();
        let mut commands_sent = 0usize;
        let mut max_deviation = 0.0_f64;
        let mut final_positions = vec![0.0; dof];

        loop {
            let elapsed = start_time.elapsed();

            // Check timeout
            if elapsed > timeout {
                return Err(ExecutionError::Timeout {
                    elapsed,
                    expected: duration,
                });
            }

            // Compute trajectory time
            let t = elapsed.as_secs_f64();
            if t > duration.as_secs_f64() {
                break; // Trajectory complete
            }

            // Interpolate
            let (positions, velocities) = interpolate_at(trajectory, t);

            // Send command with timing check
            let cmd_start = Instant::now();
            sink.send_command(&positions, &velocities)
                .map_err(ExecutionError::CommandFailed)?;
            let cmd_elapsed = cmd_start.elapsed();

            if self.config.command_timeout_ms > 0
                && cmd_elapsed.as_millis() > self.config.command_timeout_ms as u128
            {
                return Err(ExecutionError::CommandFailed(format!(
                    "Command took {:.1}ms (timeout: {}ms) — hardware may be unresponsive",
                    cmd_elapsed.as_secs_f64() * 1000.0,
                    self.config.command_timeout_ms
                )));
            }

            // Check feedback deviation
            if let Some(fb) = feedback {
                if let Some(actual) = fb.read_positions() {
                    let deviation: f64 = positions
                        .iter()
                        .zip(actual.iter())
                        .map(|(cmd, act)| (cmd - act).abs())
                        .fold(0.0f64, f64::max);

                    max_deviation = max_deviation.max(deviation);

                    if deviation > self.config.position_tolerance {
                        return Err(ExecutionError::DeviationExceeded {
                            deviation,
                            tolerance: self.config.position_tolerance,
                            command_index: commands_sent,
                        });
                    }
                }
            }

            commands_sent += 1;

            // Sleep until next command time
            let next_time = start_time + dt * (commands_sent as u32);
            let now = Instant::now();
            if next_time > now {
                std::thread::sleep(next_time - now);
            }
        }

        // Send final waypoint
        let last = trajectory.waypoints.last().unwrap();
        sink.send_command(&last.positions, &last.velocities)
            .map_err(ExecutionError::CommandFailed)?;
        final_positions = last.positions.clone();
        commands_sent += 1;

        Ok(ExecutionResult {
            state: ExecutionState::Completed,
            actual_duration: start_time.elapsed(),
            expected_duration: duration,
            max_deviation: if feedback.is_some() {
                Some(max_deviation)
            } else {
                None
            },
            commands_sent,
            final_positions,
        })
    }
}

impl RealTimeExecutor {
    /// Execute with an independent safety watchdog.
    ///
    /// The watchdog runs on a separate thread and fires a zero-velocity command
    /// if this executor loop hangs. Call this instead of `execute()` for real
    /// robot deployments.
    pub fn execute_with_watchdog(
        &self,
        trajectory: &TimedTrajectory,
        sink: Arc<Mutex<dyn crate::CommandSink + Send>>,
        feedback: Option<&dyn crate::FeedbackSource>,
        watchdog_config: crate::watchdog::WatchdogConfig,
    ) -> Result<ExecutionResult, ExecutionError> {
        let last_positions = Arc::new(Mutex::new(vec![0.0; trajectory.dof]));
        let watchdog = crate::watchdog::SafetyWatchdog::start(
            sink.clone(),
            last_positions.clone(),
            watchdog_config,
        );

        // Create a wrapper sink that also updates last_safe_positions
        struct WatchdogSink {
            inner: Arc<Mutex<dyn crate::CommandSink + Send>>,
            last_positions: Arc<Mutex<Vec<f64>>>,
            watchdog: crate::watchdog::SafetyWatchdog,
        }
        impl crate::CommandSink for WatchdogSink {
            fn send_command(&mut self, positions: &[f64], velocities: &[f64]) -> Result<(), String> {
                self.watchdog.heartbeat();
                if let Ok(mut lp) = self.last_positions.lock() {
                    lp.clear();
                    lp.extend_from_slice(positions);
                }
                self.inner.lock()
                    .map_err(|e| format!("sink lock failed: {e}"))?
                    .send_command(positions, velocities)
            }
        }

        let mut wrapper = WatchdogSink {
            inner: sink,
            last_positions,
            watchdog,
        };

        self.execute_with_feedback(trajectory, &mut wrapper, feedback)
    }
}

impl Default for RealTimeExecutor {
    fn default() -> Self {
        Self::new(ExecutionConfig::default())
    }
}

impl TrajectoryExecutor for RealTimeExecutor {
    fn execute(
        &self,
        trajectory: &TimedTrajectory,
        sink: &mut dyn CommandSink,
    ) -> Result<ExecutionResult, ExecutionError> {
        self.execute_with_feedback(trajectory, sink, None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kinetic_trajectory::TimedWaypoint;

    struct CountingSink {
        count: usize,
        last_positions: Vec<f64>,
    }

    impl CountingSink {
        fn new() -> Self {
            Self {
                count: 0,
                last_positions: vec![],
            }
        }
    }

    impl CommandSink for CountingSink {
        fn send_command(&mut self, positions: &[f64], _velocities: &[f64]) -> Result<(), String> {
            self.count += 1;
            self.last_positions = positions.to_vec();
            Ok(())
        }
    }

    fn short_trajectory() -> TimedTrajectory {
        TimedTrajectory {
            duration: Duration::from_millis(50), // 50ms — fast enough for tests
            dof: 3,
            waypoints: vec![
                TimedWaypoint {
                    time: 0.0,
                    positions: vec![0.0, 0.0, 0.0],
                    velocities: vec![1.0, 1.0, 1.0],
                    accelerations: vec![0.0, 0.0, 0.0],
                },
                TimedWaypoint {
                    time: 0.05,
                    positions: vec![0.05, 0.05, 0.05],
                    velocities: vec![1.0, 1.0, 1.0],
                    accelerations: vec![0.0, 0.0, 0.0],
                },
            ],
        }
    }

    #[test]
    fn realtime_executor_completes() {
        let executor = RealTimeExecutor::new(ExecutionConfig {
            rate_hz: 100.0, // 100Hz for fast test
            ..Default::default()
        });

        let traj = short_trajectory();
        let mut sink = CountingSink::new();

        let result = executor.execute(&traj, &mut sink).unwrap();

        assert_eq!(result.state, ExecutionState::Completed);
        assert!(sink.count > 0);
        // Final positions should be near trajectory end
        assert!((sink.last_positions[0] - 0.05).abs() < 0.01);
    }

    #[test]
    fn realtime_executor_empty_trajectory() {
        let executor = RealTimeExecutor::default();
        let traj = TimedTrajectory {
            duration: Duration::ZERO,
            dof: 3,
            waypoints: vec![],
        };
        let mut sink = CountingSink::new();
        let result = executor.execute(&traj, &mut sink);
        assert!(result.is_err());
    }

    #[test]
    fn realtime_executor_with_feedback() {
        struct PerfectFeedback;
        impl FeedbackSource for PerfectFeedback {
            fn read_positions(&self) -> Option<Vec<f64>> {
                Some(vec![0.0, 0.0, 0.0]) // Always at origin — will deviate
            }
        }

        let executor = RealTimeExecutor::new(ExecutionConfig {
            rate_hz: 100.0,
            position_tolerance: 10.0, // Large tolerance so it doesn't error
            ..Default::default()
        });

        let traj = short_trajectory();
        let mut sink = CountingSink::new();

        let result = executor
            .execute_with_feedback(&traj, &mut sink, Some(&PerfectFeedback))
            .unwrap();

        assert!(result.max_deviation.is_some());
    }

    #[test]
    fn realtime_executor_command_failure() {
        struct FailSink;
        impl CommandSink for FailSink {
            fn send_command(&mut self, _: &[f64], _: &[f64]) -> Result<(), String> {
                Err("motor fault".into())
            }
        }

        let executor = RealTimeExecutor::new(ExecutionConfig {
            rate_hz: 100.0,
            ..Default::default()
        });
        let traj = short_trajectory();
        let mut sink = FailSink;
        let result = executor.execute(&traj, &mut sink);
        assert!(result.is_err());
        assert!(format!("{}", result.unwrap_err()).contains("motor fault"));
    }

    /// Gap 2: Execute a 100ms trajectory at 100Hz and verify wall-clock time
    /// is within 50ms of expected (generous tolerance for CI).
    #[test]
    fn realtime_executor_timing_precision() {
        let trajectory = TimedTrajectory {
            duration: Duration::from_millis(100),
            dof: 2,
            waypoints: vec![
                TimedWaypoint {
                    time: 0.0,
                    positions: vec![0.0, 0.0],
                    velocities: vec![1.0, 1.0],
                    accelerations: vec![0.0, 0.0],
                },
                TimedWaypoint {
                    time: 0.05,
                    positions: vec![0.05, 0.05],
                    velocities: vec![1.0, 1.0],
                    accelerations: vec![0.0, 0.0],
                },
                TimedWaypoint {
                    time: 0.10,
                    positions: vec![0.10, 0.10],
                    velocities: vec![0.0, 0.0],
                    accelerations: vec![0.0, 0.0],
                },
            ],
        };

        let executor = RealTimeExecutor::new(ExecutionConfig {
            rate_hz: 100.0,
            ..Default::default()
        });

        let mut sink = CountingSink::new();
        let result = executor.execute(&trajectory, &mut sink).unwrap();

        assert_eq!(result.state, ExecutionState::Completed);

        // Wall-clock should be close to 100ms. Allow generous 50ms tolerance for CI.
        let expected = Duration::from_millis(100);
        let actual = result.actual_duration;
        let diff = if actual > expected {
            actual - expected
        } else {
            expected - actual
        };

        assert!(
            diff < Duration::from_millis(50),
            "Timing precision: expected ~100ms, got {:?} (diff {:?})",
            actual,
            diff,
        );
    }

    /// FIX 1b: Executor-side validation gate catches limit violations.
    #[test]
    fn realtime_executor_rejects_limit_violating_trajectory() {
        let executor = RealTimeExecutor::new(ExecutionConfig {
            rate_hz: 100.0,
            joint_limits: Some(vec![(-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14)]),
            ..Default::default()
        });

        // Trajectory with one waypoint outside limits
        let bad_traj = TimedTrajectory {
            duration: Duration::from_millis(50),
            dof: 3,
            waypoints: vec![
                TimedWaypoint {
                    time: 0.0,
                    positions: vec![0.0, 0.0, 0.0],
                    velocities: vec![0.0, 0.0, 0.0],
                    accelerations: vec![0.0, 0.0, 0.0],
                },
                TimedWaypoint {
                    time: 0.05,
                    positions: vec![0.0, 100.0, 0.0], // joint 1 = 100 rad — way outside limits
                    velocities: vec![0.0, 0.0, 0.0],
                    accelerations: vec![0.0, 0.0, 0.0],
                },
            ],
        };

        let mut sink = CountingSink::new();
        let result = executor.execute(&bad_traj, &mut sink);
        assert!(result.is_err(), "should reject limit-violating trajectory");
        assert_eq!(sink.count, 0, "no commands should be sent before validation passes");
    }

    /// FIX 1b: Executor-side validation passes for valid trajectory.
    #[test]
    fn realtime_executor_accepts_valid_trajectory_with_limits() {
        let executor = RealTimeExecutor::new(ExecutionConfig {
            rate_hz: 100.0,
            joint_limits: Some(vec![(-3.14, 3.14), (-3.14, 3.14), (-3.14, 3.14)]),
            ..Default::default()
        });

        let traj = short_trajectory(); // all positions within [-3.14, 3.14]
        let mut sink = CountingSink::new();
        let result = executor.execute(&traj, &mut sink);
        assert!(result.is_ok(), "valid trajectory should pass: {:?}", result.err());
        assert!(sink.count > 0, "commands should be sent");
    }

    /// Gap 3: FeedbackSource returning positions offset by 0.5 radians should
    /// trigger DeviationExceeded when tolerance is 0.1.
    #[test]
    fn realtime_executor_deviation_exceeded() {
        struct OffsetFeedback;
        impl FeedbackSource for OffsetFeedback {
            fn read_positions(&self) -> Option<Vec<f64>> {
                // Return positions offset by 0.5 radians from whatever command is sent
                Some(vec![0.5, 0.5, 0.5])
            }
        }

        let executor = RealTimeExecutor::new(ExecutionConfig {
            rate_hz: 100.0,
            position_tolerance: 0.1, // 0.1 rad tolerance — much less than 0.5 offset
            ..Default::default()
        });

        let traj = short_trajectory();
        let mut sink = CountingSink::new();

        let result = executor.execute_with_feedback(&traj, &mut sink, Some(&OffsetFeedback));

        match result {
            Err(ExecutionError::DeviationExceeded {
                deviation,
                tolerance,
                ..
            }) => {
                assert!(
                    deviation > tolerance,
                    "Deviation {} should exceed tolerance {}",
                    deviation,
                    tolerance
                );
                assert!(
                    (tolerance - 0.1).abs() < 1e-10,
                    "Tolerance should be 0.1"
                );
            }
            other => panic!(
                "Expected DeviationExceeded error, got {:?}",
                other
            ),
        }
    }
}
