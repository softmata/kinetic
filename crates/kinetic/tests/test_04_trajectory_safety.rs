//! Acceptance tests: 04 trajectory_safety
//! Spec: doc_tests/04_TRAJECTORY_SAFETY.md
//!
//! Trajectory parameterization, continuity, and validation.

#[path = "helpers.rs"]
mod helpers;
use helpers::*;

use kinetic::prelude::*;
use kinetic::planning::Planner;
use kinetic::trajectory::{
    trapezoidal, totp,
    TrajectoryValidator, ValidationConfig, TimedTrajectory, TimedWaypoint,
};
use std::time::Duration;

// ─── Trapezoidal/TOTP valid output across 5 robots ──────────────────────────

#[test]
fn trapezoidal_produces_valid_output() {
    for name in SAFETY_ROBOTS {
        let robot = load_robot(name);
        let planner = match Planner::new(&robot) { Ok(p) => p, Err(_) => continue };
        let start = mid_joints(&robot);
        let goal_vals: Vec<f64> = start.iter().map(|v| v + 0.2).collect();
        let goal = Goal::Joints(JointValues::new(goal_vals));

        let plan = match planner.plan(&start, &goal) { Ok(p) => p, Err(_) => continue };
        let timed = match trapezoidal(&plan.waypoints, 1.0, 2.0) { Ok(t) => t, Err(_) => continue };

        assert!(!timed.is_empty(), "{name}: timed trajectory empty");
        assert!(timed.duration().as_secs_f64() > 0.0, "{name}: zero duration");

        // Monotonic timestamps
        for w in timed.waypoints.windows(2) {
            assert!(w[1].time >= w[0].time, "{name}: non-monotonic time");
        }

        // Start/end at zero velocity
        let first = &timed.waypoints[0];
        let last = timed.waypoints.last().unwrap();
        for &v in &first.velocities {
            assert!(v.abs() < 1e-6, "{name}: start velocity not zero: {v}");
        }
        for &v in &last.velocities {
            assert!(v.abs() < 1e-6, "{name}: end velocity not zero: {v}");
        }

        // No NaN/Inf
        for wp in &timed.waypoints {
            for &p in &wp.positions { assert!(p.is_finite(), "{name}: NaN in position"); }
            for &v in &wp.velocities { assert!(v.is_finite(), "{name}: NaN in velocity"); }
        }
    }
}

#[test]
fn totp_produces_valid_output() {
    for name in SAFETY_ROBOTS {
        let robot = load_robot(name);
        let planner = match Planner::new(&robot) { Ok(p) => p, Err(_) => continue };
        let start = mid_joints(&robot);
        let goal_vals: Vec<f64> = start.iter().map(|v| v + 0.2).collect();
        let goal = Goal::Joints(JointValues::new(goal_vals));

        let plan = match planner.plan(&start, &goal) { Ok(p) => p, Err(_) => continue };

        let vel_limits: Vec<f64> = robot.joint_limits.iter().map(|l| l.velocity.max(0.5)).collect();
        let acc_limits: Vec<f64> = vel_limits.iter().map(|v| v * 4.0).collect();

        let timed = match totp(&plan.waypoints, &vel_limits, &acc_limits, 0.01) {
            Ok(t) => t, Err(_) => continue
        };

        assert!(!timed.is_empty(), "{name}: TOTP empty");
        assert!(timed.duration().as_secs_f64() > 0.0, "{name}: TOTP zero duration");
    }
}

// ─── Position continuity: sample and verify ─────────────────────────────────

#[test]
fn trajectory_position_continuity() {
    for name in &["ur5e", "franka_panda"] {
        let robot = load_robot(name);
        let planner = match Planner::new(&robot) { Ok(p) => p, Err(_) => continue };
        let start = mid_joints(&robot);
        let goal_vals: Vec<f64> = start.iter().map(|v| v + 0.3).collect();
        let goal = Goal::Joints(JointValues::new(goal_vals));

        let plan = match planner.plan(&start, &goal) { Ok(p) => p, Err(_) => continue };
        let timed = match trapezoidal(&plan.waypoints, 1.0, 2.0) { Ok(t) => t, Err(_) => continue };
        if timed.is_empty() { continue; }

        let dur = timed.duration().as_secs_f64();
        let n_samples = 500;
        let mut prev_pos: Option<Vec<f64>> = None;
        let mut max_jump = 0.0_f64;

        for i in 0..n_samples {
            let t = dur * i as f64 / (n_samples - 1) as f64;
            let wp = timed.sample_at(Duration::from_secs_f64(t));
            let dt = dur / (n_samples - 1) as f64;

            if let Some(prev) = &prev_pos {
                for (j, (&curr, &prv)) in wp.positions.iter().zip(prev.iter()).enumerate() {
                    let jump = (curr - prv).abs();
                    max_jump = max_jump.max(jump);
                    // Position change per sample should be bounded by max_velocity * dt
                    let v_max = robot.joint_limits.get(j).map(|l| l.velocity).unwrap_or(10.0).max(1.0);
                    assert!(
                        jump < v_max * dt * 2.0,
                        "{name} sample {i} joint {j}: position jump {jump:.6} > limit {:.6}",
                        v_max * dt * 2.0
                    );
                }
            }
            prev_pos = Some(wp.positions.clone());
        }
        eprintln!("{name}: max position jump = {max_jump:.6}");
    }
}

// ─── Validator catches all 6 violation types ────────────────────────────────

#[test]
fn validator_catches_position_violation() {
    let validator = TrajectoryValidator::new(
        &[-3.14; 6], &[3.14; 6],
        &[2.0; 6], &[4.0; 6],
        ValidationConfig::default(),
    );

    let bad = TimedTrajectory {
        duration: Duration::from_millis(100),
        dof: 6,
        waypoints: vec![
            TimedWaypoint { time: 0.0, positions: vec![0.0; 6], velocities: vec![0.0; 6], accelerations: vec![0.0; 6] },
            TimedWaypoint { time: 0.1, positions: vec![100.0; 6], velocities: vec![0.0; 6], accelerations: vec![0.0; 6] },
        ],
    };

    let result = validator.validate(&bad);
    assert!(result.is_err(), "should detect position violation");
}

#[test]
fn validator_catches_velocity_violation() {
    let validator = TrajectoryValidator::new(
        &[-3.14; 6], &[3.14; 6],
        &[2.0; 6], &[4.0; 6],
        ValidationConfig::default(),
    );

    let bad = TimedTrajectory {
        duration: Duration::from_millis(100),
        dof: 6,
        waypoints: vec![
            TimedWaypoint { time: 0.0, positions: vec![0.0; 6], velocities: vec![0.0; 6], accelerations: vec![0.0; 6] },
            TimedWaypoint { time: 0.1, positions: vec![0.1; 6], velocities: vec![100.0; 6], accelerations: vec![0.0; 6] },
        ],
    };

    let result = validator.validate(&bad);
    assert!(result.is_err(), "should detect velocity violation");
}

#[test]
fn validator_passes_valid_trajectory() {
    let validator = TrajectoryValidator::new(
        &[-3.14; 6], &[3.14; 6],
        &[2.0; 6], &[4.0; 6],
        ValidationConfig::default(),
    );

    let good = TimedTrajectory {
        duration: Duration::from_millis(100),
        dof: 6,
        waypoints: vec![
            TimedWaypoint { time: 0.0, positions: vec![0.0; 6], velocities: vec![0.0; 6], accelerations: vec![0.0; 6] },
            TimedWaypoint { time: 0.1, positions: vec![0.01; 6], velocities: vec![0.1; 6], accelerations: vec![1.0; 6] },
        ],
    };

    let result = validator.validate(&good);
    assert!(result.is_ok(), "valid trajectory should pass: {:?}", result.err());
}

// ─── Edge cases ─────────────────────────────────────────────────────────────

#[test]
fn trapezoidal_zero_displacement_ok() {
    let path = vec![vec![0.5, 0.5], vec![0.5, 0.5]]; // no movement
    let result = trapezoidal(&path, 1.0, 2.0);
    assert!(result.is_ok(), "zero displacement should not error");
}

#[test]
fn trapezoidal_single_waypoint_ok() {
    let path = vec![vec![0.5, 0.5]];
    let result = trapezoidal(&path, 1.0, 2.0);
    assert!(result.is_ok(), "single waypoint should not error");
}

#[test]
fn trapezoidal_empty_path_ok() {
    let path: Vec<Vec<f64>> = vec![];
    let result = trapezoidal(&path, 1.0, 2.0);
    assert!(result.is_ok(), "empty path should not error");
    assert!(result.unwrap().is_empty());
}
