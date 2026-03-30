//! LogExecutor — records all commands for inspection and replay.

use std::time::Duration;

use kinetic_trajectory::TimedTrajectory;

use crate::sim::interpolate_at;
use crate::{
    CommandSink, ExecutionConfig, ExecutionError, ExecutionResult, ExecutionState,
    TrajectoryExecutor,
};

/// Recorded command: timestamp, positions, velocities.
#[derive(Debug, Clone)]
pub struct RecordedCommand {
    /// Time in seconds from trajectory start.
    pub time: f64,
    /// Commanded joint positions.
    pub positions: Vec<f64>,
    /// Commanded joint velocities.
    pub velocities: Vec<f64>,
}

/// Executor that records all commands to a log for inspection.
///
/// Useful for verifying command sequences, replaying trajectories,
/// and testing without hardware.
pub struct LogExecutor {
    config: ExecutionConfig,
    log: Vec<RecordedCommand>,
}

impl LogExecutor {
    pub fn new(config: ExecutionConfig) -> Self {
        Self {
            config,
            log: Vec::new(),
        }
    }

    /// Get the recorded command log.
    pub fn commands(&self) -> &[RecordedCommand] {
        &self.log
    }

    /// Take ownership of the log, clearing internal state.
    pub fn take_log(&mut self) -> Vec<RecordedCommand> {
        std::mem::take(&mut self.log)
    }

    /// Clear the log.
    pub fn clear(&mut self) {
        self.log.clear();
    }

    /// Number of recorded commands.
    pub fn len(&self) -> usize {
        self.log.len()
    }

    /// Whether the log is empty.
    pub fn is_empty(&self) -> bool {
        self.log.is_empty()
    }
}

impl Default for LogExecutor {
    fn default() -> Self {
        Self::new(ExecutionConfig::default())
    }
}

impl TrajectoryExecutor for LogExecutor {
    fn execute(
        &self,
        trajectory: &TimedTrajectory,
        _sink: &mut dyn CommandSink,
    ) -> Result<ExecutionResult, ExecutionError> {
        // LogExecutor ignores the external sink and records internally
        if trajectory.waypoints.is_empty() {
            return Err(ExecutionError::InvalidTrajectory("empty trajectory".into()));
        }

        let dof = trajectory.dof;
        let dt = 1.0 / self.config.rate_hz;
        let duration = trajectory.duration.as_secs_f64();
        let num_steps = (duration / dt).ceil() as usize;

        // We need mutable access to self.log, but we're behind &self.
        // Use an internal buffer and copy after.
        let mut log = Vec::with_capacity(num_steps + 1);
        let mut final_positions = vec![0.0; dof];

        for step in 0..=num_steps {
            let t = (step as f64 * dt).min(duration);
            let (positions, velocities) = interpolate_at(trajectory, t);
            log.push(RecordedCommand {
                time: t,
                positions: positions.clone(),
                velocities,
            });
            final_positions = positions;
        }

        let commands_sent = log.len();

        // Store log (we can't mutate self here, so return it in the result)
        // The user should call execute_and_log() instead for mutable access.
        Ok(ExecutionResult {
            state: ExecutionState::Completed,
            actual_duration: Duration::from_secs_f64(duration),
            expected_duration: trajectory.duration,
            max_deviation: None,
            commands_sent,
            final_positions,
        })
    }
}

impl LogExecutor {
    /// Execute and record all commands into the internal log.
    ///
    /// Unlike `execute()` (trait method), this takes `&mut self` and
    /// stores commands in the internal log accessible via `commands()`.
    pub fn execute_and_log(
        &mut self,
        trajectory: &TimedTrajectory,
    ) -> Result<ExecutionResult, ExecutionError> {
        if trajectory.waypoints.is_empty() {
            return Err(ExecutionError::InvalidTrajectory("empty trajectory".into()));
        }

        let dof = trajectory.dof;
        let dt = 1.0 / self.config.rate_hz;
        let duration = trajectory.duration.as_secs_f64();
        let num_steps = (duration / dt).ceil() as usize;
        let mut final_positions = vec![0.0; dof];

        self.log.clear();
        self.log.reserve(num_steps + 1);

        for step in 0..=num_steps {
            let t = (step as f64 * dt).min(duration);
            let (positions, velocities) = interpolate_at(trajectory, t);
            self.log.push(RecordedCommand {
                time: t,
                positions: positions.clone(),
                velocities,
            });
            final_positions = positions;
        }

        Ok(ExecutionResult {
            state: ExecutionState::Completed,
            actual_duration: Duration::from_secs_f64(duration),
            expected_duration: trajectory.duration,
            max_deviation: None,
            commands_sent: self.log.len(),
            final_positions,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kinetic_trajectory::TimedWaypoint;

    fn make_traj() -> TimedTrajectory {
        TimedTrajectory {
            duration: Duration::from_secs_f64(0.5),
            dof: 3,
            waypoints: vec![
                TimedWaypoint {
                    time: 0.0,
                    positions: vec![0.0, 0.0, 0.0],
                    velocities: vec![1.0, 1.0, 1.0],
                    accelerations: vec![0.0, 0.0, 0.0],
                },
                TimedWaypoint {
                    time: 0.5,
                    positions: vec![0.5, 0.5, 0.5],
                    velocities: vec![1.0, 1.0, 1.0],
                    accelerations: vec![0.0, 0.0, 0.0],
                },
            ],
        }
    }

    #[test]
    fn log_executor_records_commands() {
        let mut executor = LogExecutor::new(ExecutionConfig {
            rate_hz: 100.0,
            ..Default::default()
        });

        let traj = make_traj();
        let result = executor.execute_and_log(&traj).unwrap();

        assert_eq!(result.state, ExecutionState::Completed);
        assert!(executor.len() > 0);

        // First command should be near start
        let first = &executor.commands()[0];
        assert!((first.time - 0.0).abs() < 1e-10);
        assert!((first.positions[0] - 0.0).abs() < 0.01);

        // Last command should be near end
        let last = executor.commands().last().unwrap();
        assert!((last.positions[0] - 0.5).abs() < 0.02);
    }

    #[test]
    fn log_executor_take_log() {
        let mut executor = LogExecutor::default();
        let traj = make_traj();
        executor.execute_and_log(&traj).unwrap();

        let log = executor.take_log();
        assert!(!log.is_empty());
        assert!(executor.is_empty()); // cleared after take
    }

    #[test]
    fn log_executor_clear() {
        let mut executor = LogExecutor::default();
        let traj = make_traj();
        executor.execute_and_log(&traj).unwrap();
        assert!(!executor.is_empty());
        executor.clear();
        assert!(executor.is_empty());
    }
}
