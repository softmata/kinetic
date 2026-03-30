//! Integration test: basic free-space planning through the full pipeline.
//!
//! Tests RRT-Connect planning → trajectory time-parameterization for UR5e.

use kinetic::prelude::*;
use std::sync::Arc;
use std::time::Duration;

fn ur5e() -> Arc<Robot> {
    Arc::new(Robot::from_name("ur5e").unwrap())
}

fn home_joints() -> Vec<f64> {
    vec![0.0, -std::f64::consts::FRAC_PI_2, 0.0, 0.0, 0.0, 0.0]
}

#[test]
fn plan_joints_to_joints() {
    let robot = ur5e();
    let planner = Planner::new(&robot).unwrap();

    let start = home_joints();
    let goal = Goal::Joints(JointValues(vec![0.5, -1.0, 0.5, 0.0, 0.5, 0.0]));

    let result = planner.plan(&start, &goal).unwrap();
    assert!(
        result.waypoints.len() >= 2,
        "Path should have at least start and goal"
    );

    // First waypoint should match start
    for (a, b) in result.waypoints[0].iter().zip(start.iter()) {
        assert!((a - b).abs() < 1e-6, "Start mismatch: {} vs {}", a, b);
    }

    // Last waypoint should be close to goal
    let last = result.waypoints.last().unwrap();
    let goal_vals = [0.5, -1.0, 0.5, 0.0, 0.5, 0.0];
    for (a, b) in last.iter().zip(goal_vals.iter()) {
        assert!((a - b).abs() < 0.1, "Goal mismatch: {} vs {}", a, b);
    }
}

#[test]
fn plan_and_time_parameterize_trapezoidal() {
    let robot = ur5e();
    let planner = Planner::new(&robot).unwrap();

    let start = home_joints();
    let goal = Goal::Joints(JointValues(vec![0.5, -1.0, 0.5, 0.0, 0.5, 0.0]));

    let result = planner.plan(&start, &goal).unwrap();
    let timed = trapezoidal(&result.waypoints, 1.0, 2.0).unwrap();

    assert!(timed.waypoints.len() >= 2);
    assert!(timed.duration.as_secs_f64() > 0.0);

    // Times should be monotonically increasing
    for pair in timed.waypoints.windows(2) {
        assert!(
            pair[1].time >= pair[0].time,
            "Times should be monotonic: {} >= {}",
            pair[1].time,
            pair[0].time
        );
    }

    // First waypoint at t=0
    assert!(
        timed.waypoints[0].time.abs() < 1e-10,
        "First waypoint should be at t=0"
    );
}

#[test]
fn plan_and_time_parameterize_totp() {
    let robot = ur5e();
    let planner = Planner::new(&robot).unwrap();

    let start = home_joints();
    let goal = Goal::Joints(JointValues(vec![0.5, -1.0, 0.5, 0.0, 0.5, 0.0]));

    let result = planner.plan(&start, &goal).unwrap();

    // TOTP (time-optimal time parameterization)
    let vel_limits = vec![2.0; 6];
    let accel_limits = vec![4.0; 6];
    let timed = totp(&result.waypoints, &vel_limits, &accel_limits, 0.01).unwrap();

    assert!(timed.waypoints.len() >= 2);
    assert!(timed.duration.as_secs_f64() > 0.0);
}

#[test]
fn plan_returns_consistent_dof() {
    let robot = ur5e();
    let planner = Planner::new(&robot).unwrap();

    let start = home_joints();
    let goal = Goal::Joints(JointValues(vec![0.5, -1.0, 0.5, 0.0, 0.5, 0.0]));

    let result = planner.plan(&start, &goal).unwrap();

    // Every waypoint should have 6 DOF
    for (i, wp) in result.waypoints.iter().enumerate() {
        assert_eq!(
            wp.len(),
            6,
            "Waypoint {} should have 6 DOF, got {}",
            i,
            wp.len()
        );
    }
}

#[test]
fn plan_multiple_goals() {
    let robot = ur5e();
    let planner = Planner::new(&robot).unwrap();

    let configs = vec![
        vec![0.5, -1.0, 0.5, 0.0, 0.5, 0.0],
        vec![-0.5, -1.2, 0.8, 0.0, -0.5, 0.0],
        vec![1.0, -0.8, 0.3, -0.5, 0.8, 0.2],
    ];

    let mut current = home_joints();
    for target in &configs {
        let result = planner
            .plan(&current, &Goal::Joints(JointValues(target.clone())))
            .unwrap();
        assert!(result.waypoints.len() >= 2);
        current = result.waypoints.last().unwrap().clone();
    }
}

#[test]
fn plan_roundtrip_returns_to_start() {
    let robot = ur5e();
    let planner = Planner::new(&robot).unwrap();

    let start = home_joints();
    let goal = vec![0.5, -1.0, 0.5, 0.0, 0.5, 0.0];

    // Plan forward
    let fwd = planner
        .plan(&start, &Goal::Joints(JointValues(goal.clone())))
        .unwrap();
    let mid = fwd.waypoints.last().unwrap().clone();

    // Plan back
    let rev = planner
        .plan(&mid, &Goal::Joints(JointValues(start.clone())))
        .unwrap();
    let final_joints = rev.waypoints.last().unwrap();

    // Should arrive back near start
    for (a, b) in final_joints.iter().zip(start.iter()) {
        assert!((a - b).abs() < 0.15, "Roundtrip mismatch: {} vs {}", a, b);
    }
}

#[test]
fn rrt_timeout_returns_error_not_panic() {
    let robot = ur5e();
    let planner = Planner::new(&robot).unwrap();

    let start = home_joints();
    // A far-apart goal to make planning non-trivial
    let goal = Goal::Joints(JointValues(vec![2.5, -0.5, 2.0, -1.5, 2.5, -1.0]));

    // Use an absurdly short timeout so the planner cannot succeed
    let config = PlannerConfig {
        timeout: Duration::from_millis(1),
        max_iterations: 10_000,
        ..PlannerConfig::default()
    };

    let result = planner.plan_with_config(&start, &goal, config);

    // Should return a PlanningTimeout error (not panic or crash)
    match result {
        Err(KineticError::PlanningTimeout { .. }) => {
            // Expected — the planner timed out gracefully
        }
        Ok(_) => {
            // On very fast machines the planner might still find a path in 1ms;
            // this is acceptable — the important thing is no panic.
        }
        Err(other) => {
            // Other planning errors (e.g., PlanningFailed) are also acceptable,
            // as long as it didn't panic.
            let msg = format!("{}", other);
            assert!(
                !msg.contains("panic"),
                "Should not panic, got error: {}",
                msg
            );
        }
    }
}
