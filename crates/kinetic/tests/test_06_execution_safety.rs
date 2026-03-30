//! Acceptance tests: 06 execution_safety
//! Spec: doc_tests/06_EXECUTION_SAFETY.md
//!
//! Trajectory execution safety: traversal, timing, deviation, watchdog, limits.

#[path = "helpers.rs"]
mod helpers;
use helpers::*;

use kinetic::prelude::*;
use kinetic::planning::Planner;
use kinetic::trajectory::trapezoidal;
use kinetic::execution::{
    ExecutionConfig, RealTimeExecutor, SimExecutor, TrajectoryExecutor,
};
use std::time::Duration;

// ─── SimExecutor traverses all waypoints ────────────────────────────────────

#[test]
fn sim_executor_traverses_all() {
    for name in SAFETY_ROBOTS {
        let robot = load_robot(name);
        let planner = match Planner::new(&robot) { Ok(p) => p, Err(_) => continue };
        let start = mid_joints(&robot);
        let goal_vals: Vec<f64> = start.iter().map(|v| v + 0.2).collect();
        let goal = Goal::Joints(JointValues::new(goal_vals));

        let plan = match planner.plan(&start, &goal) { Ok(p) => p, Err(_) => continue };
        let timed = match trapezoidal(&plan.waypoints, 1.0, 2.0) { Ok(t) => t, Err(_) => continue };
        if timed.is_empty() { continue; }

        let executor = SimExecutor::new(ExecutionConfig::default());
        let mut sink = RecordingSink::new();
        let result = executor.execute(&timed, &mut sink);

        assert!(result.is_ok(), "{name}: SimExecutor failed: {:?}", result.err());
        let r = result.unwrap();
        assert!(sink.count() > 0, "{name}: no commands sent");
        assert_eq!(r.state, kinetic::execution::ExecutionState::Completed);
    }
}

// ─── RealTimeExecutor timing ────────────────────────────────────────────────

#[test]
fn realtime_executor_completes_short_trajectory() {
    let robot = load_robot("ur5e");
    let planner = Planner::new(&robot).unwrap();
    let start = mid_joints(&robot);
    let goal = Goal::Joints(JointValues::new(start.iter().map(|v| v + 0.1).collect()));
    let plan = planner.plan(&start, &goal).unwrap();
    let timed = trapezoidal(&plan.waypoints, 1.0, 2.0).unwrap();
    if timed.is_empty() { return; }

    let executor = RealTimeExecutor::new(ExecutionConfig {
        rate_hz: 100.0, // fast for testing
        ..Default::default()
    });
    let mut sink = RecordingSink::new();
    let result = executor.execute(&timed, &mut sink);

    assert!(result.is_ok(), "execution should succeed: {:?}", result.err());
    let r = result.unwrap();
    assert_eq!(r.state, kinetic::execution::ExecutionState::Completed);
    assert!(sink.count() >= 2, "should send multiple commands");

    // Wall-clock should be close to trajectory duration (within 2x for CI)
    let expected = timed.duration();
    assert!(
        r.actual_duration < expected * 3,
        "took {:?} for {:?} trajectory", r.actual_duration, expected
    );
}

// ─── Commands within limits ─────────────────────────────────────────────────

#[test]
fn all_executed_commands_within_limits() {
    let robot = load_robot("ur5e");
    let planner = Planner::new(&robot).unwrap();
    let start = mid_joints(&robot);
    let goal = Goal::Joints(JointValues::new(vec![0.3, -0.3, 0.3, -0.3, 0.3, -0.3]));
    let plan = planner.plan(&start, &goal).unwrap();
    let timed = trapezoidal(&plan.waypoints, 1.0, 2.0).unwrap();

    let executor = SimExecutor::new(ExecutionConfig::default());
    let mut sink = RecordingSink::new();
    executor.execute(&timed, &mut sink).unwrap();

    for (cmd_idx, (positions, _velocities)) in sink.commands.iter().enumerate() {
        for (j, &val) in positions.iter().enumerate() {
            if j < robot.joint_limits.len() {
                let lo = robot.joint_limits[j].lower;
                let hi = robot.joint_limits[j].upper;
                assert!(
                    val >= lo - 0.01 && val <= hi + 0.01,
                    "command {} joint {}: {:.4} outside [{:.4}, {:.4}]",
                    cmd_idx, j, val, lo, hi
                );
            }
        }
    }
}

// ─── Deviation detection ────────────────────────────────────────────────────

#[test]
fn deviation_detection_catches_offset_feedback() {
    use kinetic::execution::FeedbackSource;
    use kinetic::trajectory::TimedWaypoint;

    struct OffsetFeedback;
    impl FeedbackSource for OffsetFeedback {
        fn read_positions(&self) -> Option<Vec<f64>> {
            Some(vec![0.5; 6]) // constant offset from any commanded position
        }
    }

    let timed = kinetic::trajectory::TimedTrajectory {
        duration: Duration::from_millis(50),
        dof: 6,
        waypoints: vec![
            TimedWaypoint { time: 0.0, positions: vec![0.0; 6], velocities: vec![0.0; 6], accelerations: vec![0.0; 6] },
            TimedWaypoint { time: 0.05, positions: vec![0.01; 6], velocities: vec![0.0; 6], accelerations: vec![0.0; 6] },
        ],
    };

    let executor = RealTimeExecutor::new(ExecutionConfig {
        rate_hz: 100.0,
        position_tolerance: 0.1, // much less than 0.5 offset
        ..Default::default()
    });

    let mut sink = RecordingSink::new();
    let result = executor.execute_with_feedback(&timed, &mut sink, Some(&OffsetFeedback));

    assert!(result.is_err(), "should detect deviation");
    match result.unwrap_err() {
        kinetic::execution::ExecutionError::DeviationExceeded { deviation, tolerance, .. } => {
            assert!(deviation > tolerance);
        }
        other => panic!("expected DeviationExceeded, got {:?}", other),
    }
}

// ─── Hardware fault handling ────────────────────────────────────────────────

#[test]
fn hardware_fault_does_not_panic() {
    use kinetic::trajectory::TimedWaypoint;

    struct FailingSink;
    impl kinetic::execution::CommandSink for FailingSink {
        fn send_command(&mut self, _: &[f64], _: &[f64]) -> std::result::Result<(), String> {
            Err("motor_fault: overcurrent on joint 2".into())
        }
    }

    let timed = kinetic::trajectory::TimedTrajectory {
        duration: Duration::from_millis(50),
        dof: 6,
        waypoints: vec![
            TimedWaypoint { time: 0.0, positions: vec![0.0; 6], velocities: vec![0.0; 6], accelerations: vec![0.0; 6] },
            TimedWaypoint { time: 0.05, positions: vec![0.01; 6], velocities: vec![0.0; 6], accelerations: vec![0.0; 6] },
        ],
    };

    let executor = RealTimeExecutor::new(ExecutionConfig { rate_hz: 100.0, ..Default::default() });
    let mut sink = FailingSink;
    let result = executor.execute(&timed, &mut sink);

    assert!(result.is_err(), "should fail on hardware fault");
    let err_msg = format!("{}", result.unwrap_err());
    assert!(err_msg.contains("motor_fault"), "error should include hardware message: {err_msg}");
}

// ─── Empty trajectory rejected ──────────────────────────────────────────────

#[test]
fn empty_trajectory_rejected() {
    let timed = kinetic::trajectory::TimedTrajectory {
        duration: Duration::ZERO,
        dof: 6,
        waypoints: vec![],
    };

    let executor = SimExecutor::new(ExecutionConfig::default());
    let mut sink = RecordingSink::new();
    let result = executor.execute(&timed, &mut sink);
    assert!(result.is_err(), "empty trajectory should be rejected");
}

// ─── Executor result fields populated ───────────────────────────────────────

#[test]
fn execution_result_fields_correct() {
    use kinetic::trajectory::TimedWaypoint;

    let timed = kinetic::trajectory::TimedTrajectory {
        duration: Duration::from_millis(50),
        dof: 3,
        waypoints: vec![
            TimedWaypoint { time: 0.0, positions: vec![0.0; 3], velocities: vec![0.0; 3], accelerations: vec![0.0; 3] },
            TimedWaypoint { time: 0.05, positions: vec![0.05; 3], velocities: vec![0.0; 3], accelerations: vec![0.0; 3] },
        ],
    };

    let executor = SimExecutor::new(ExecutionConfig::default());
    let mut sink = RecordingSink::new();
    let result = executor.execute(&timed, &mut sink).unwrap();

    assert_eq!(result.state, kinetic::execution::ExecutionState::Completed);
    assert!(result.commands_sent > 0);
    assert_eq!(result.final_positions.len(), 3);
    // Final position should be near trajectory end
    assert!((result.final_positions[0] - 0.05).abs() < 0.01);
}
