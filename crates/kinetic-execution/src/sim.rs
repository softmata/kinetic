//! SimExecutor — instant trajectory execution for testing.

use std::time::Duration;

use kinetic_trajectory::TimedTrajectory;

use crate::{
    CommandSink, ExecutionConfig, ExecutionError, ExecutionResult, ExecutionState,
    TrajectoryExecutor,
};

/// Fast-forward executor that walks through all waypoints instantly.
///
/// No real timing — evaluates the full trajectory in a tight loop.
/// Useful for testing that planning results are executable.
pub struct SimExecutor {
    config: ExecutionConfig,
}

impl SimExecutor {
    pub fn new(config: ExecutionConfig) -> Self {
        Self { config }
    }

    /// Execute without a sink — just validate the trajectory is feasible.
    pub fn validate(
        &self,
        trajectory: &TimedTrajectory,
    ) -> Result<ExecutionResult, ExecutionError> {
        let mut null_sink = NullSink;
        self.execute(trajectory, &mut null_sink)
    }
}

impl Default for SimExecutor {
    fn default() -> Self {
        Self::new(ExecutionConfig::default())
    }
}

impl TrajectoryExecutor for SimExecutor {
    fn execute(
        &self,
        trajectory: &TimedTrajectory,
        sink: &mut dyn CommandSink,
    ) -> Result<ExecutionResult, ExecutionError> {
        if trajectory.waypoints.is_empty() {
            return Err(ExecutionError::InvalidTrajectory("empty trajectory".into()));
        }

        let dof = trajectory.dof;
        let dt = 1.0 / self.config.rate_hz;
        let duration = trajectory.duration.as_secs_f64();
        let num_steps = (duration / dt).ceil() as usize;
        let mut commands_sent = 0;
        let mut final_positions = vec![0.0; dof];

        // Walk through trajectory at configured rate (simulated, no actual sleeping)
        for step in 0..=num_steps {
            let t = (step as f64 * dt).min(duration);

            // Interpolate waypoint at time t
            let (positions, velocities) = interpolate_at(trajectory, t);

            sink.send_command(&positions, &velocities)
                .map_err(ExecutionError::CommandFailed)?;

            final_positions = positions;
            commands_sent += 1;
        }

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

/// Null command sink — discards all commands.
struct NullSink;

impl CommandSink for NullSink {
    fn send_command(&mut self, _positions: &[f64], _velocities: &[f64]) -> Result<(), String> {
        Ok(())
    }
}

/// Interpolate trajectory at a given time.
pub(crate) fn interpolate_at(trajectory: &TimedTrajectory, t: f64) -> (Vec<f64>, Vec<f64>) {
    let wps = &trajectory.waypoints;

    if wps.is_empty() {
        return (vec![], vec![]);
    }

    if wps.len() == 1 || t <= wps[0].time {
        return (wps[0].positions.clone(), wps[0].velocities.clone());
    }

    if t >= wps.last().unwrap().time {
        let last = wps.last().unwrap();
        return (last.positions.clone(), last.velocities.clone());
    }

    // Find surrounding waypoints
    let mut i = 0;
    while i + 1 < wps.len() && wps[i + 1].time <= t {
        i += 1;
    }

    if i + 1 >= wps.len() {
        let last = wps.last().unwrap();
        return (last.positions.clone(), last.velocities.clone());
    }

    let w0 = &wps[i];
    let w1 = &wps[i + 1];
    let dt = w1.time - w0.time;
    let alpha = if dt > 1e-12 { (t - w0.time) / dt } else { 0.0 };

    let positions: Vec<f64> = w0
        .positions
        .iter()
        .zip(w1.positions.iter())
        .map(|(&a, &b)| a + alpha * (b - a))
        .collect();

    let velocities: Vec<f64> = w0
        .velocities
        .iter()
        .zip(w1.velocities.iter())
        .map(|(&a, &b)| a + alpha * (b - a))
        .collect();

    (positions, velocities)
}

#[cfg(test)]
mod tests {
    use super::*;
    use kinetic_trajectory::TimedWaypoint;

    fn make_trajectory(dof: usize, num_wps: usize) -> TimedTrajectory {
        let mut waypoints = Vec::new();
        for i in 0..num_wps {
            let t = i as f64 * 0.1;
            let pos: Vec<f64> = (0..dof).map(|j| t * (j as f64 + 1.0) * 0.1).collect();
            let vel: Vec<f64> = (0..dof).map(|j| (j as f64 + 1.0) * 0.1).collect();
            let acc = vec![0.0; dof];
            waypoints.push(TimedWaypoint {
                time: t,
                positions: pos,
                velocities: vel,
                accelerations: acc,
            });
        }
        TimedTrajectory {
            duration: Duration::from_secs_f64((num_wps - 1) as f64 * 0.1),
            dof,
            waypoints,
        }
    }

    #[test]
    fn sim_executor_completes() {
        let traj = make_trajectory(6, 10);
        let executor = SimExecutor::default();
        let result = executor.validate(&traj).unwrap();
        assert_eq!(result.state, ExecutionState::Completed);
        assert!(result.commands_sent > 0);
        assert_eq!(result.final_positions.len(), 6);
    }

    #[test]
    fn sim_executor_empty_trajectory() {
        let traj = TimedTrajectory {
            duration: Duration::ZERO,
            dof: 6,
            waypoints: vec![],
        };
        let executor = SimExecutor::default();
        let result = executor.validate(&traj);
        assert!(result.is_err());
    }

    #[test]
    fn sim_executor_single_waypoint() {
        let traj = make_trajectory(6, 1);
        let executor = SimExecutor::new(ExecutionConfig {
            rate_hz: 100.0,
            ..Default::default()
        });
        let result = executor.validate(&traj).unwrap();
        assert_eq!(result.state, ExecutionState::Completed);
    }

    #[test]
    fn interpolate_at_endpoints() {
        let traj = make_trajectory(3, 5);
        let (pos_start, _) = interpolate_at(&traj, 0.0);
        let (pos_end, _) = interpolate_at(&traj, 0.4);
        assert_eq!(pos_start, traj.waypoints[0].positions);
        assert_eq!(pos_end, traj.waypoints[4].positions);
    }

    #[test]
    fn interpolate_at_midpoint() {
        let traj = make_trajectory(2, 3); // t=0, 0.1, 0.2
        let (pos, _) = interpolate_at(&traj, 0.05); // midpoint between wp0 and wp1
                                                    // wp0 pos = [0, 0], wp1 pos = [0.01, 0.02], midpoint ≈ [0.005, 0.01]
        assert!((pos[0] - 0.005).abs() < 1e-10);
        assert!((pos[1] - 0.01).abs() < 1e-10);
    }
}
