//! Integration tests for error recovery during trajectory execution.
//!
//! Tests what happens when execution deviates from the plan:
//! safe stop, recovery to nearest valid config, replanning from
//! mid-trajectory state, and retry with reduced velocity.

use std::sync::Arc;
use std::time::Duration;

use kinetic::prelude::*;
use kinetic_trajectory::monitor::{DeviationLevel, ExecutionMonitor, MonitorConfig};
use kinetic_trajectory::trapezoidal::trapezoidal;
use kinetic_trajectory::validation::{TrajectoryValidator, ValidationConfig};

fn load_robot(name: &str) -> (Arc<Robot>, KinematicChain) {
    let robot = Arc::new(Robot::from_name(name).unwrap());
    let group = robot.groups.values().next().unwrap();
    let chain = KinematicChain::extract(&robot, &group.base_link, &group.tip_link).unwrap();
    (robot, chain)
}

/// Get a safe start config (mid-range of all joints) for the given chain.
fn safe_start(robot: &Robot, chain: &KinematicChain) -> Vec<f64> {
    chain
        .active_joints
        .iter()
        .map(|&ji| {
            let j = &robot.joints[ji];
            match &j.limits {
                Some(l) => (l.lower + l.upper) / 2.0,
                None => 0.0,
            }
        })
        .collect()
}

/// Get a safe goal config offset from start, clamped within joint limits.
fn safe_goal(start: &[f64], robot: &Robot, chain: &KinematicChain, offset: f64) -> Vec<f64> {
    start
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let ji = chain.active_joints[i];
            let j = &robot.joints[ji];
            match &j.limits {
                Some(l) => (s + offset).clamp(l.lower + 0.01, l.upper - 0.01),
                None => s + offset,
            }
        })
        .collect()
}

fn robot_joint_limits(robot: &Robot, chain: &KinematicChain) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut lower = Vec::new();
    let mut upper = Vec::new();
    let mut vel_limits = Vec::new();
    for &ji in &chain.active_joints {
        let j = &robot.joints[ji];
        if let Some(limits) = &j.limits {
            lower.push(limits.lower);
            upper.push(limits.upper);
            vel_limits.push(limits.velocity);
        } else {
            lower.push(-std::f64::consts::PI);
            upper.push(std::f64::consts::PI);
            vel_limits.push(2.0);
        }
    }
    (lower, upper, vel_limits)
}

// ─── Safe stop at arbitrary trajectory point ────────────────────────

#[test]
fn safe_stop_preserves_valid_configuration() {
    let (robot, chain) = load_robot("franka_panda");

    let start = safe_start(&robot, &chain);
    let goal = safe_goal(&start, &robot, &chain, 0.3);
    let traj = trapezoidal(&[start, goal], 1.0, 2.0).unwrap();

    // Simulate execution: stop at an arbitrary mid-point
    let stop_time = traj.duration().as_secs_f64() * 0.37;
    let stopped_state = traj.sample_at(Duration::from_secs_f64(stop_time));

    // Verify stopped state has valid joint positions (within limits)
    let (lower, upper, _) = robot_joint_limits(&robot, &chain);
    for (i, pos) in stopped_state.positions.iter().enumerate() {
        assert!(
            *pos >= lower[i] - 1e-6 && *pos <= upper[i] + 1e-6,
            "Joint {} position {} outside limits [{}, {}]",
            i,
            pos,
            lower[i],
            upper[i]
        );
    }
}

#[test]
fn safe_stop_at_every_waypoint() {
    let (robot, chain) = load_robot("ur5e");

    let start = safe_start(&robot, &chain);
    let mid = safe_goal(&start, &robot, &chain, 0.2);
    let goal = safe_goal(&start, &robot, &chain, 0.4);
    let traj = trapezoidal(&[start, mid, goal], 1.0, 2.0).unwrap();

    let (lower, upper, _) = robot_joint_limits(&robot, &chain);

    // Stop at 10 evenly spaced points along the trajectory
    for i in 0..10 {
        let t = traj.duration().as_secs_f64() * i as f64 / 9.0;
        let state = traj.sample_at(Duration::from_secs_f64(t));

        for (j, pos) in state.positions.iter().enumerate() {
            assert!(
                *pos >= lower[j] - 1e-6 && *pos <= upper[j] + 1e-6,
                "Stop point {}, joint {} position {} outside limits",
                i,
                j,
                pos
            );
        }
        assert_eq!(state.positions.len(), chain.dof);
        assert_eq!(state.velocities.len(), chain.dof);
    }
}

// ─── Recovery from deviated state ───────────────────────────────────

#[test]
fn replan_from_deviated_position() {
    let (robot, chain) = load_robot("franka_panda");

    let start = safe_start(&robot, &chain);
    let goal = safe_goal(&start, &robot, &chain, 0.3);
    let _original_traj = trapezoidal(&[start.clone(), goal.clone()], 1.0, 2.0).unwrap();

    // Simulate: robot deviated to a slightly different configuration
    let deviated_state: Vec<f64> = start
        .iter()
        .enumerate()
        .map(|(i, &s)| s + 0.15 + (i as f64) * 0.01)
        .collect();

    // Replan from deviated state to original goal
    let recovery_traj = trapezoidal(&[deviated_state.clone(), goal.clone()], 1.0, 2.0).unwrap();

    // Verify recovery trajectory starts from deviated state
    let first_wp = &recovery_traj.waypoints[0];
    for (i, (actual, expected)) in first_wp.positions.iter().zip(&deviated_state).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-10,
            "Recovery joint {} should start at deviated position {} but was {}",
            i,
            expected,
            actual
        );
    }

    // Verify recovery trajectory ends at goal
    let last_wp = recovery_traj.waypoints.last().unwrap();
    for (i, (actual, expected)) in last_wp.positions.iter().zip(&goal).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-10,
            "Recovery joint {} should end at goal {} but was {}",
            i,
            expected,
            actual
        );
    }
}

#[test]
fn replan_from_monitor_abort_state() {
    let (robot, chain) = load_robot("franka_panda");

    let start = safe_start(&robot, &chain);
    let goal = safe_goal(&start, &robot, &chain, 0.3);
    let traj = trapezoidal(&[start, goal.clone()], 1.0, 2.0).unwrap();

    let mut monitor = ExecutionMonitor::new(
        traj.clone(),
        MonitorConfig {
            position_tolerance: 0.1,
            warning_fraction: 0.7,
            noise_window: 1,
        },
    );

    // Simulate: execution drifts until abort on joint 0
    let t = traj.duration().as_secs_f64() * 0.5;
    let planned = traj.sample_at(Duration::from_secs_f64(t));
    let mut actual = planned.positions.clone();
    actual[0] += 0.15; // triggers abort

    let level = monitor.check(t, &actual);
    assert!(matches!(level, DeviationLevel::Abort { .. }));

    // On abort: replan from actual position to goal
    let recovery = trapezoidal(&[actual.clone(), goal.clone()], 1.0, 2.0).unwrap();

    // Verify recovery is valid
    assert!(recovery.waypoints.len() >= 2);
    let first = &recovery.waypoints[0];
    assert!((first.positions[0] - actual[0]).abs() < 1e-10);

    // Monitor should accept the new trajectory at t=0 (recovery start)
    monitor.set_trajectory(recovery.clone());
    let level = monitor.check(0.0, &actual);
    assert_eq!(
        level,
        DeviationLevel::Normal,
        "Should be on-track for recovery trajectory"
    );
}

// ─── Retry with reduced velocity ────────────────────────────────────

#[test]
fn retry_with_reduced_velocity_is_slower() {
    let (robot, chain) = load_robot("franka_panda");

    let start = safe_start(&robot, &chain);
    let goal = safe_goal(&start, &robot, &chain, 0.5);

    let fast_traj = trapezoidal(&[start.clone(), goal.clone()], 2.0, 4.0).unwrap();
    let slow_traj = trapezoidal(&[start, goal], 1.0, 2.0).unwrap();

    assert!(
        slow_traj.duration() > fast_traj.duration(),
        "Slow trajectory ({:?}) should be longer than fast ({:?})",
        slow_traj.duration(),
        fast_traj.duration()
    );
}

#[test]
fn reduced_velocity_trajectory_is_valid() {
    let (robot, chain) = load_robot("ur5e");
    let (lower, upper, _) = robot_joint_limits(&robot, &chain);
    let dof = chain.dof;

    let start = safe_start(&robot, &chain);
    let goal = safe_goal(&start, &robot, &chain, 0.3);

    let reduced_vel = 0.5;
    let reduced_accel = 1.0;
    let traj = trapezoidal(&[start, goal], reduced_vel, reduced_accel).unwrap();

    let accel_limits: Vec<f64> = vec![reduced_accel * 1.1; dof];
    let vel_check: Vec<f64> = vec![reduced_vel * 1.1; dof];
    let validator = TrajectoryValidator::new(
        &lower,
        &upper,
        &vel_check,
        &accel_limits,
        ValidationConfig::default(),
    );

    assert!(
        validator.validate(&traj).is_ok(),
        "Reduced-velocity trajectory should pass validation"
    );
}

#[test]
fn progressive_velocity_reduction() {
    let (robot, chain) = load_robot("franka_panda");

    let start = safe_start(&robot, &chain);
    let goal = safe_goal(&start, &robot, &chain, 0.4);

    let mut durations = Vec::new();
    for pct in &[1.0_f64, 0.75, 0.5, 0.25] {
        let vel = 2.0 * pct;
        let acc = 4.0 * pct;
        let traj = trapezoidal(&[start.clone(), goal.clone()], vel, acc).unwrap();
        durations.push(traj.duration());
    }

    for i in 1..durations.len() {
        assert!(
            durations[i] > durations[i - 1],
            "Slower velocity should produce longer trajectory"
        );
    }
}

// ─── Partial execution recovery ─────────────────────────────────────

#[test]
fn recover_from_partial_execution_midpoint() {
    let (robot, chain) = load_robot("xarm7");

    let start = safe_start(&robot, &chain);
    let goal = safe_goal(&start, &robot, &chain, 0.3);
    let traj = trapezoidal(&[start, goal.clone()], 1.0, 2.0).unwrap();

    // Simulate: robot executed 60% of trajectory then stopped
    let stop_t = traj.duration().as_secs_f64() * 0.6;
    let mid_state = traj.sample_at(Duration::from_secs_f64(stop_t));

    // Recovery: plan from current position to original goal
    let recovery = trapezoidal(&[mid_state.positions.clone(), goal.clone()], 1.0, 2.0).unwrap();

    // Recovery should be shorter than original
    assert!(
        recovery.duration() < traj.duration(),
        "Recovery from 60% should be shorter than full trajectory"
    );

    // Recovery end matches original goal
    let recovery_end = recovery.waypoints.last().unwrap();
    for i in 0..chain.dof {
        assert!((recovery_end.positions[i] - goal[i]).abs() < 1e-10);
    }
}

#[test]
fn recover_from_multiple_stop_points() {
    let (robot, chain) = load_robot("franka_panda");

    let start = safe_start(&robot, &chain);
    let goal = safe_goal(&start, &robot, &chain, 0.3);
    let original = trapezoidal(&[start, goal.clone()], 1.0, 2.0).unwrap();

    let mut recovery_durations = Vec::new();
    for pct in &[0.2_f64, 0.4, 0.6, 0.8] {
        let t = original.duration().as_secs_f64() * pct;
        let state = original.sample_at(Duration::from_secs_f64(t));
        let recovery = trapezoidal(&[state.positions.clone(), goal.clone()], 1.0, 2.0).unwrap();
        recovery_durations.push(recovery.duration());
    }

    for i in 1..recovery_durations.len() {
        assert!(
            recovery_durations[i] <= recovery_durations[i - 1],
            "Later stop should produce shorter recovery"
        );
    }
}

// ─── No panics on any failure scenario ──────────────────────────────

#[test]
fn no_panic_replan_from_near_goal() {
    let (robot, chain) = load_robot("franka_panda");

    let start = safe_start(&robot, &chain);
    let goal = safe_goal(&start, &robot, &chain, 0.3);
    let near_goal: Vec<f64> = goal.iter().map(|g| g - 0.001).collect();
    let result = trapezoidal(&[near_goal, goal], 1.0, 2.0);
    assert!(result.is_ok(), "Replan from near-goal should not panic");
}

#[test]
fn no_panic_replan_from_same_position() {
    let (robot, chain) = load_robot("ur5e");

    let pos = safe_start(&robot, &chain);
    let result = trapezoidal(&[pos.clone(), pos], 1.0, 2.0);
    match result {
        Ok(traj) => {
            for wp in &traj.waypoints {
                assert_eq!(wp.positions.len(), chain.dof);
            }
        }
        Err(_) => { /* acceptable */ }
    }
}

#[test]
fn no_panic_replan_from_limit() {
    let (robot, chain) = load_robot("franka_panda");
    let (lower, _, _) = robot_joint_limits(&robot, &chain);

    let goal = safe_start(&robot, &chain);
    let result = trapezoidal(&[lower, goal], 1.0, 2.0);
    assert!(result.is_ok(), "Replan from joint limits should not panic");
}

// ─── Full recovery workflow ─────────────────────────────────────────

#[test]
fn full_abort_replan_resume_workflow() {
    let (robot, chain) = load_robot("franka_panda");

    let start = safe_start(&robot, &chain);
    let goal = safe_goal(&start, &robot, &chain, 0.3);

    // Step 1: Plan original trajectory
    let traj = trapezoidal(&[start, goal.clone()], 2.0, 4.0).unwrap();
    let mut monitor = ExecutionMonitor::new(
        traj.clone(),
        MonitorConfig {
            position_tolerance: 0.1,
            warning_fraction: 0.7,
            noise_window: 1,
        },
    );

    // Step 2: Execute normally for a bit
    let normal_t = traj.duration().as_secs_f64() * 0.3;
    let planned = traj.sample_at(Duration::from_secs_f64(normal_t));
    let level = monitor.check(normal_t, &planned.positions);
    assert_eq!(level, DeviationLevel::Normal);

    // Step 3: Deviation triggers abort
    let abort_t = traj.duration().as_secs_f64() * 0.5;
    let planned = traj.sample_at(Duration::from_secs_f64(abort_t));
    let mut actual = planned.positions.clone();
    actual[0] += 0.15;
    let level = monitor.check(abort_t, &actual);
    assert!(matches!(level, DeviationLevel::Abort { .. }));

    // Step 4: Replan from actual position with reduced velocity
    let recovery = trapezoidal(&[actual.clone(), goal.clone()], 1.0, 2.0).unwrap();

    // Step 5: Resume monitoring with new trajectory
    monitor.set_trajectory(recovery.clone());
    let level = monitor.check(0.0, &actual);
    assert_eq!(level, DeviationLevel::Normal);

    // Step 6: Execute recovery to completion
    let steps = 10;
    let dt = recovery.duration().as_secs_f64() / steps as f64;
    for i in 0..=steps {
        let t = i as f64 * dt;
        let planned = recovery.sample_at(Duration::from_secs_f64(t));
        let level = monitor.check(t, &planned.positions);
        assert_eq!(
            level,
            DeviationLevel::Normal,
            "Recovery step {} should be normal",
            i
        );
    }

    // Step 7: Verify we reached the goal
    let final_state = recovery.sample_at(recovery.duration());
    for (i, (actual, expected)) in final_state.positions.iter().zip(&goal).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-6,
            "Joint {} should reach goal {} but was {}",
            i,
            expected,
            actual
        );
    }
}
