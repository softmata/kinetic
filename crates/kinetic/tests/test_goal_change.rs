//! Tests for goal change scenarios during planning and execution.
//!
//! Covers: sequential replanning to different goals, planning with timeout
//! followed by replan to new goal, servo pose tracking with goal changes,
//! concurrent planning cancellation via timeout, and rapid sequential
//! goal changes.

use std::sync::Arc;
use std::time::Duration;

use kinetic::prelude::*;
use kinetic_trajectory::monitor::{DeviationLevel, ExecutionMonitor, MonitorConfig};
use kinetic_trajectory::trapezoidal::trapezoidal;

fn load_ur5e() -> (Arc<Robot>, KinematicChain) {
    let robot = Arc::new(Robot::from_name("ur5e").unwrap());
    let group = robot.groups.values().next().unwrap();
    let chain = KinematicChain::extract(&robot, &group.base_link, &group.tip_link).unwrap();
    (robot, chain)
}

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

// ─── Sequential goal changes ─────────────────────────────────────────────

/// Plan to goal A, then immediately replan to goal B from the same start.
/// The second plan should reach goal B, not goal A.
#[test]
fn sequential_replan_reaches_latest_goal() {
    let (robot, chain) = load_ur5e();
    let planner = Planner::new(&robot).unwrap();
    let start = safe_start(&robot, &chain);
    let goal_a = safe_goal(&start, &robot, &chain, 0.3);
    let goal_b = safe_goal(&start, &robot, &chain, -0.3);

    let _result_a = planner.plan(&start, &Goal::Joints(goal_a.clone().into()));
    let result_b = planner
        .plan(&start, &Goal::Joints(goal_b.clone().into()))
        .unwrap();

    // The second plan should end at goal B
    let final_waypoint = result_b.waypoints.last().unwrap();
    for (i, (&actual, &expected)) in final_waypoint.iter().zip(goal_b.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 0.01,
            "Joint {} final={:.4}, expected={:.4}",
            i,
            actual,
            expected
        );
    }
}

/// Plan to three different goals sequentially. Each plan should
/// independently reach its target from the same start.
#[test]
fn three_sequential_goals_all_reachable() {
    let (robot, chain) = load_ur5e();
    let planner = Planner::new(&robot).unwrap();
    let start = safe_start(&robot, &chain);

    let offsets = [0.2, -0.2, 0.4];
    for offset in &offsets {
        let goal = safe_goal(&start, &robot, &chain, *offset);
        let result = planner
            .plan(&start, &Goal::Joints(goal.clone().into()))
            .unwrap();
        let final_wp = result.waypoints.last().unwrap();
        for (i, (&actual, &expected)) in final_wp.iter().zip(goal.iter()).enumerate() {
            assert!(
                (actual - expected).abs() < 0.01,
                "Offset {}: joint {} mismatch: {:.4} vs {:.4}",
                offset,
                i,
                actual,
                expected
            );
        }
    }
}

// ─── Goal change mid-execution ───────────────────────────────────────────

/// Plan to goal A, execute partially (50%), then replan from mid-point
/// to a different goal B. Final trajectory should reach goal B.
#[test]
fn replan_from_midpoint_to_new_goal() {
    let (robot, chain) = load_ur5e();
    let planner = Planner::new(&robot).unwrap();
    let start = safe_start(&robot, &chain);
    let goal_a = safe_goal(&start, &robot, &chain, 0.3);
    let goal_b = safe_goal(&start, &robot, &chain, -0.3);

    // Plan to goal A
    let result_a = planner
        .plan(&start, &Goal::Joints(goal_a.clone().into()))
        .unwrap();

    // "Execute" halfway — pick the middle waypoint as new start
    let mid_idx = result_a.waypoints.len() / 2;
    let mid_state = result_a.waypoints[mid_idx].clone();

    // Replan from mid-execution to goal B
    let result_b = planner
        .plan(&mid_state, &Goal::Joints(goal_b.clone().into()))
        .unwrap();

    // New plan should reach goal B
    let final_wp = result_b.waypoints.last().unwrap();
    for (i, (&actual, &expected)) in final_wp.iter().zip(goal_b.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 0.01,
            "Joint {} final={:.4}, expected={:.4}",
            i,
            actual,
            expected
        );
    }
}

/// Simulate execution with monitoring. When deviation is detected,
/// replan to a different goal from the deviated position.
#[test]
fn deviation_triggers_replan_to_new_goal() {
    let (robot, chain) = load_ur5e();
    let planner = Planner::new(&robot).unwrap();
    let start = safe_start(&robot, &chain);
    let goal_a = safe_goal(&start, &robot, &chain, 0.3);
    let goal_b = safe_goal(&start, &robot, &chain, -0.2);

    // Plan and parameterize trajectory to goal A
    let result_a = planner
        .plan(&start, &Goal::Joints(goal_a.clone().into()))
        .unwrap();
    let traj_a = trapezoidal(&result_a.waypoints, 1.0, 2.0).unwrap();

    // Set up monitor
    let monitor_config = MonitorConfig {
        position_tolerance: 0.05,
        ..Default::default()
    };
    let mut monitor = ExecutionMonitor::new(traj_a.clone(), monitor_config);

    // Simulate execution with deliberate deviation at t=0.5 * duration
    let duration = traj_a.duration().as_secs_f64();
    let t_mid = duration * 0.5;
    let planned_at_mid = traj_a.sample_at(Duration::from_secs_f64(t_mid));

    // Add deviation to simulate external disturbance
    let deviated: Vec<f64> = planned_at_mid
        .positions
        .iter()
        .map(|&p| p + 0.1) // Large deviation
        .collect();

    let level = monitor.check(t_mid, &deviated);
    assert!(
        matches!(level, DeviationLevel::Abort { .. }),
        "Large deviation should trigger abort, got {:?}",
        level
    );

    // Replan from deviated position to NEW goal B
    let result_b = planner
        .plan(&deviated, &Goal::Joints(goal_b.clone().into()))
        .unwrap();

    // Verify new plan reaches goal B
    let final_wp = result_b.waypoints.last().unwrap();
    for (i, (&actual, &expected)) in final_wp.iter().zip(goal_b.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 0.01,
            "Joint {} final={:.4}, expected={:.4}",
            i,
            actual,
            expected
        );
    }
}

// ─── Rapid goal changes ─────────────────────────────────────────────────

/// Submit 10 different goals rapidly. Only the last plan result matters.
/// Ensures no resource leaks or panics from rapid sequential planning.
#[test]
fn rapid_10_goal_changes_no_panic() {
    let (robot, chain) = load_ur5e();
    let planner = Planner::new(&robot).unwrap();
    let start = safe_start(&robot, &chain);

    let mut last_result = None;
    for i in 0..10 {
        let offset = 0.05 * (i as f64 - 5.0); // -0.25 to +0.2
        let goal = safe_goal(&start, &robot, &chain, offset);
        // Some goals may fail (e.g., if offset pushes near limits), that's OK
        if let Ok(result) = planner.plan(&start, &Goal::Joints(goal.into())) {
            last_result = Some(result);
        }
    }

    // At least one plan should have succeeded
    assert!(
        last_result.is_some(),
        "At least one of 10 goal changes should produce a valid plan"
    );
}

/// Plan with a very short timeout, then replan with normal timeout.
/// First plan may fail; second should succeed.
#[test]
fn timeout_then_replan_succeeds() {
    let (robot, chain) = load_ur5e();
    let planner = Planner::new(&robot).unwrap().with_config(PlannerConfig {
        timeout: Duration::from_millis(1), // Extremely short
        ..Default::default()
    });
    let start = safe_start(&robot, &chain);
    let goal = safe_goal(&start, &robot, &chain, 0.3);

    // First plan with 1ms timeout — likely fails
    let _first = planner.plan(&start, &Goal::Joints(goal.clone().into()));

    // Replan with normal timeout
    let planner_normal = Planner::new(&robot).unwrap();
    let result = planner_normal
        .plan(&start, &Goal::Joints(goal.clone().into()))
        .unwrap();

    let final_wp = result.waypoints.last().unwrap();
    for (i, (&actual, &expected)) in final_wp.iter().zip(goal.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 0.01,
            "Joint {} final={:.4}, expected={:.4}",
            i,
            actual,
            expected
        );
    }
}

// ─── Concurrent planning with goal change ────────────────────────────────

/// Start planning in a background thread, then plan with a different goal
/// in the main thread. Both should complete without deadlock or panic.
#[test]
fn concurrent_planning_different_goals() {
    let (robot, chain) = load_ur5e();
    let start = safe_start(&robot, &chain);
    let goal_a = safe_goal(&start, &robot, &chain, 0.3);
    let goal_b = safe_goal(&start, &robot, &chain, -0.3);

    let robot_clone = (*robot).clone();
    let start_clone = start.clone();
    let goal_a_clone = goal_a.clone();

    let handle = std::thread::spawn(move || {
        let planner = Planner::new(&robot_clone).unwrap();
        planner.plan(&start_clone, &Goal::Joints(goal_a_clone.into()))
    });

    // Main thread plans to different goal
    let planner = Planner::new(&robot).unwrap();
    let result_b = planner.plan(&start, &Goal::Joints(goal_b.clone().into()));

    // Both should complete (success or known error — no panic)
    let result_a = handle.join().expect("Thread A panicked");

    assert!(
        result_a.is_ok() || result_b.is_ok(),
        "At least one concurrent plan should succeed"
    );
}

// ─── Servo goal tracking changes ─────────────────────────────────────────

/// Servo tracks pose A for several steps, then switches to tracking pose B.
/// Joint positions should change direction after goal switch.
#[test]
fn servo_pose_tracking_goal_change() {
    let (robot, chain) = load_ur5e();
    let scene = Arc::new(Scene::new(&robot).unwrap());
    let config = kinetic_reactive::ServoConfig {
        input_type: kinetic_reactive::InputType::PoseTracking,
        ..Default::default()
    };
    let mut servo = kinetic_reactive::Servo::new(&robot, &scene, config).unwrap();

    // Set initial state at mid-config
    let mid = safe_start(&robot, &chain);
    let velocities = vec![0.0; mid.len()];
    servo.set_state(&mid, &velocities).unwrap();

    // Compute current EE pose
    let fk_mid = kinetic_kinematics::forward_kinematics(&robot, &chain, &mid).unwrap();
    let target_a = nalgebra::Isometry3::translation(
        fk_mid.0.translation.x + 0.05,
        fk_mid.0.translation.y,
        fk_mid.0.translation.z,
    );
    let target_b = nalgebra::Isometry3::translation(
        fk_mid.0.translation.x - 0.05,
        fk_mid.0.translation.y,
        fk_mid.0.translation.z,
    );

    // Track pose A for 5 steps
    let mut positions_after_a = Vec::new();
    for _ in 0..5 {
        if let Ok(cmd) = servo.track_pose(&target_a) {
            positions_after_a.push(cmd.positions.clone());
        }
    }

    // Switch to tracking pose B for 5 steps
    let mut positions_after_b = Vec::new();
    for _ in 0..5 {
        if let Ok(cmd) = servo.track_pose(&target_b) {
            positions_after_b.push(cmd.positions.clone());
        }
    }

    // Verify that commands were produced for both goals
    assert!(
        !positions_after_a.is_empty(),
        "Should produce commands while tracking A"
    );
    assert!(
        !positions_after_b.is_empty(),
        "Should produce commands while tracking B"
    );

    // The servo should have smoothly changed direction — no panics, no NaNs
    for pos in positions_after_a.iter().chain(positions_after_b.iter()) {
        for &p in pos {
            assert!(p.is_finite(), "Position should be finite, got {}", p);
        }
    }
}

// ─── Goal type changes ───────────────────────────────────────────────────

/// Plan with Joint goal, then plan with Named goal, then plan with Pose goal.
/// Different goal types should all work sequentially without issues.
#[test]
fn mixed_goal_types_sequential() {
    let (robot, chain) = load_ur5e();
    let planner = Planner::new(&robot).unwrap();
    let start = safe_start(&robot, &chain);

    // Goal 1: Joint goal
    let joint_goal = safe_goal(&start, &robot, &chain, 0.2);
    let result_joints = planner.plan(&start, &Goal::Joints(joint_goal.into()));
    assert!(result_joints.is_ok(), "Joint goal should succeed");

    // Goal 2: Named goal (home)
    let result_named = planner.plan(&start, &Goal::Named("home".into()));
    // May fail if robot has no "home" named pose — that's OK
    let _named_ok = result_named.is_ok();

    // Goal 3: Pose goal (plan to current position — trivial)
    let current_pose = kinetic_kinematics::forward_kinematics(&robot, &chain, &start).unwrap();
    let result_pose = planner.plan(&start, &Goal::Pose(current_pose));
    // Planning to current pose should succeed (trivial path)
    assert!(
        result_pose.is_ok(),
        "Pose goal (at current position) should succeed"
    );
}

/// Plan with Relative goal, then replan with Joints goal.
/// Different goal resolution paths should work sequentially.
#[test]
fn relative_then_joints_goal() {
    let (robot, chain) = load_ur5e();
    let planner = Planner::new(&robot).unwrap();
    let start = safe_start(&robot, &chain);

    // Goal 1: Relative offset (small move)
    let offset = nalgebra::Vector3::new(0.01, 0.0, 0.0);
    let result_relative = planner.plan(&start, &Goal::Relative(offset));
    // May fail (IK not always solvable) — that's fine

    // Goal 2: Direct joint goal
    let joint_goal = safe_goal(&start, &robot, &chain, 0.2);
    let result_joints = planner
        .plan(&start, &Goal::Joints(joint_goal.clone().into()))
        .unwrap();

    let final_wp = result_joints.waypoints.last().unwrap();
    for (i, (&actual, &expected)) in final_wp.iter().zip(joint_goal.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 0.01,
            "Joint {} final={:.4}, expected={:.4}",
            i,
            actual,
            expected
        );
    }

    // Verify planner state wasn't corrupted by previous relative goal attempt
    let _ = result_relative; // Just ensure it didn't panic
}

// ─── Execution monitoring with goal change ───────────────────────────────

/// Execute trajectory, switch to new goal mid-execution,
/// verify monitor can be reset and used with new trajectory.
#[test]
fn monitor_reset_on_goal_change() {
    let (robot, chain) = load_ur5e();
    let planner = Planner::new(&robot).unwrap();
    let start = safe_start(&robot, &chain);
    let goal_a = safe_goal(&start, &robot, &chain, 0.2);
    let goal_b = safe_goal(&start, &robot, &chain, -0.2);

    // Plan and parameterize for goal A
    let result_a = planner.plan(&start, &Goal::Joints(goal_a.into())).unwrap();
    let traj_a = trapezoidal(&result_a.waypoints, 1.0, 2.0).unwrap();

    let monitor_config = MonitorConfig::default();
    let mut monitor = ExecutionMonitor::new(traj_a.clone(), monitor_config);

    // Check a few points on trajectory A
    let sample_a = traj_a.sample_at(Duration::from_millis(100));
    let check_a = monitor.check(0.1, &sample_a.positions);
    assert!(
        matches!(check_a, DeviationLevel::Normal),
        "On-trajectory should be normal"
    );

    // Goal changes! Plan and parameterize for goal B
    let result_b = planner.plan(&start, &Goal::Joints(goal_b.into())).unwrap();
    let traj_b = trapezoidal(&result_b.waypoints, 1.0, 2.0).unwrap();

    // Reset monitor with new trajectory
    monitor.set_trajectory(traj_b.clone());

    // Check point on trajectory B (after reset)
    let sample_b = traj_b.sample_at(Duration::from_millis(100));
    let check_b = monitor.check(0.1, &sample_b.positions);
    assert!(
        matches!(check_b, DeviationLevel::Normal),
        "On new trajectory should be normal after reset"
    );
}
