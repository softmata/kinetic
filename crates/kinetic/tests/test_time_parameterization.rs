//! Integration test: time parameterization methods.
//!
//! Tests all time-parameterization options: trapezoidal, TOTP, S-curve, blending.

use kinetic::prelude::*;
use kinetic::trajectory::{
    blend, blend_sequence, cubic_spline_time, jerk_limited, jerk_limited_per_joint,
    trapezoidal_per_joint,
};
use std::sync::Arc;

fn ur5e() -> Arc<Robot> {
    Arc::new(Robot::from_name("ur5e").unwrap())
}

fn home_joints() -> Vec<f64> {
    vec![0.0, -std::f64::consts::FRAC_PI_2, 0.0, 0.0, 0.0, 0.0]
}

fn plan_path() -> Vec<Vec<f64>> {
    let robot = ur5e();
    let planner = Planner::new(&robot).unwrap();
    let start = home_joints();
    let goal = Goal::Joints(JointValues(vec![0.5, -1.0, 0.5, 0.0, 0.5, 0.0]));
    planner.plan(&start, &goal).unwrap().waypoints
}

#[test]
fn trapezoidal_basic() {
    let path = plan_path();
    let timed = trapezoidal(&path, 1.0, 2.0).unwrap();

    assert!(timed.waypoints.len() >= 2);
    assert!(timed.duration.as_secs_f64() > 0.0);

    // Check monotonic timestamps
    for pair in timed.waypoints.windows(2) {
        assert!(pair[1].time >= pair[0].time);
    }

    // Check velocities are bounded
    for wp in &timed.waypoints {
        for &v in &wp.velocities {
            assert!(
                v.abs() <= 1.0 + 1e-6,
                "Velocity should be bounded by max_vel=1.0: {}",
                v
            );
        }
    }
}

#[test]
fn trapezoidal_per_joint_test() {
    let path = plan_path();
    let vel_limits = vec![2.0; 6];
    let accel_limits = vec![4.0; 6];

    let timed = trapezoidal_per_joint(&path, &vel_limits, &accel_limits).unwrap();

    assert!(timed.waypoints.len() >= 2);
    assert!(timed.duration.as_secs_f64() > 0.0);
}

#[test]
fn totp_basic() {
    let path = plan_path();
    let vel_limits = vec![2.0; 6];
    let accel_limits = vec![4.0; 6];

    let timed = totp(&path, &vel_limits, &accel_limits, 0.01).unwrap();

    assert!(timed.waypoints.len() >= 2);
    assert!(timed.duration.as_secs_f64() > 0.0);

    // TOTP should be faster than or equal to trapezoidal
    let trap_timed = trapezoidal_per_joint(&path, &vel_limits, &accel_limits).unwrap();
    assert!(
        timed.duration <= trap_timed.duration + std::time::Duration::from_millis(100),
        "TOTP should be at least as fast as trapezoidal: {} vs {}",
        timed.duration.as_secs_f64(),
        trap_timed.duration.as_secs_f64()
    );
}

#[test]
fn jerk_limited_basic() {
    let path = plan_path();
    let timed = jerk_limited(&path, 1.0, 2.0, 10.0).unwrap();

    assert!(timed.waypoints.len() >= 2);
    assert!(timed.duration.as_secs_f64() > 0.0);

    // Jerk-limited should produce smooth profiles
    for pair in timed.waypoints.windows(2) {
        assert!(pair[1].time >= pair[0].time);
    }
}

#[test]
fn jerk_limited_per_joint_test() {
    let path = plan_path();
    let vel_limits = vec![2.0; 6];
    let accel_limits = vec![4.0; 6];
    let jerk_limits = vec![20.0; 6];

    let timed = jerk_limited_per_joint(&path, &vel_limits, &accel_limits, &jerk_limits).unwrap();

    assert!(timed.waypoints.len() >= 2);
    assert!(timed.duration.as_secs_f64() > 0.0);
}

#[test]
fn blend_two_trajectories() {
    let robot = ur5e();
    let planner = Planner::new(&robot).unwrap();

    let start = home_joints();
    let mid = vec![0.5, -1.0, 0.5, 0.0, 0.5, 0.0];
    let end = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];

    let path1 = planner
        .plan(&start, &Goal::Joints(JointValues(mid.clone())))
        .unwrap();
    let path2 = planner.plan(&mid, &Goal::Joints(JointValues(end))).unwrap();

    let timed1 = trapezoidal(&path1.waypoints, 1.0, 2.0).unwrap();
    let timed2 = trapezoidal(&path2.waypoints, 1.0, 2.0).unwrap();

    let blended = blend(&timed1, &timed2, 0.2).unwrap();
    assert!(blended.waypoints.len() >= 2);
    assert!(blended.duration.as_secs_f64() > 0.0);
}

#[test]
fn blend_sequence_test() {
    let robot = ur5e();
    let planner = Planner::new(&robot).unwrap();

    let configs = [
        home_joints(),
        vec![0.5, -1.0, 0.5, 0.0, 0.5, 0.0],
        vec![-0.3, -1.2, 0.8, 0.0, -0.3, 0.0],
        vec![0.0, -0.8, 0.3, -0.5, 0.0, 0.2],
    ];

    let mut trajectories = Vec::new();
    for pair in configs.windows(2) {
        let path = planner
            .plan(&pair[0], &Goal::Joints(JointValues(pair[1].clone())))
            .unwrap();
        let timed = trapezoidal(&path.waypoints, 1.0, 2.0).unwrap();
        trajectories.push(timed);
    }

    let blended = blend_sequence(&trajectories, 0.15).unwrap();
    assert!(blended.waypoints.len() >= 2);
    assert!(blended.duration.as_secs_f64() > 0.0);
}

#[test]
fn cubic_spline_basic() {
    let path = plan_path();
    let timed = cubic_spline_time(&path, Some(3.0), None).unwrap();

    assert!(timed.waypoints.len() >= 2);
    // Duration should be approximately 3.0 seconds
    assert!(
        (timed.duration.as_secs_f64() - 3.0).abs() < 0.5,
        "Duration should be ~3.0s: {}",
        timed.duration.as_secs_f64()
    );
}

#[test]
fn timed_trajectory_waypoint_consistency() {
    let path = plan_path();
    let timed = trapezoidal(&path, 1.0, 2.0).unwrap();

    for wp in &timed.waypoints {
        assert_eq!(wp.positions.len(), 6, "Each waypoint should have 6 DOF");
        assert_eq!(
            wp.velocities.len(),
            6,
            "Each waypoint should have 6 velocities"
        );
        assert!(wp.time >= 0.0, "Time should be non-negative");

        for &p in &wp.positions {
            assert!(p.is_finite(), "Position should be finite");
        }
        for &v in &wp.velocities {
            assert!(v.is_finite(), "Velocity should be finite");
        }
    }
}
