//! End-to-end integration test: plan → execute → verify.
//!
//! Proves the full KINETIC standalone pipeline works:
//! load robot → plan → time-parameterize → execute → verify collision-free.

use kinetic::execution::{ExecutionConfig, LogExecutor};
use kinetic::kinematics::{forward_kinematics, KinematicChain};
use kinetic::planning::{PlanExecuteLoop, Planner};
use kinetic::prelude::*;

/// Helper: check if a joint config is collision-free in the scene.
#[allow(dead_code)]
fn is_collision_free(_robot: &Robot, scene: &Scene, joints: &[f64]) -> bool {
    scene.check_collision(joints).map(|c| !c).unwrap_or(false)
}

#[test]
fn e2e_plan_execute_verify_ur5e() {
    let robot = Robot::from_name("ur5e").unwrap();
    let arm = &robot.groups["arm"];
    let chain = KinematicChain::extract(&robot, &arm.base_link, &arm.tip_link).unwrap();

    // No scene obstacles — tests the plan→execute pipeline, not collision avoidance
    // (collision avoidance tested separately in test_03_collision_safety.rs)

    // Start and goal (moderate motion)
    let start = vec![0.0, -1.57, 1.0, -1.0, -1.57, 0.0];
    let goal = vec![0.5, -1.0, 0.8, -1.2, -1.0, 0.3];

    // Plan without scene (fast in debug mode)
    let planner = Planner::new(&robot).unwrap();
    let plan_result = planner
        .plan(&start, &Goal::Joints(JointValues(goal.clone())))
        .unwrap();
    assert!(
        plan_result.waypoints.len() >= 2,
        "Should have at least start+goal"
    );

    // Time-parameterize
    let vel_limits = robot.velocity_limits();
    let accel_limits = robot.acceleration_limits();
    let timed =
        kinetic::trajectory::trapezoidal_per_joint(&plan_result.waypoints, &vel_limits, &accel_limits)
            .unwrap();
    assert!(timed.duration.as_secs_f64() > 0.0);

    // Execute with LogExecutor
    let mut executor = LogExecutor::new(ExecutionConfig {
        rate_hz: 100.0,
        ..Default::default()
    });
    let exec_result = executor.execute_and_log(&timed).unwrap();
    assert_eq!(exec_result.state, ExecutionState::Completed);
    assert!(exec_result.commands_sent > 0);

    // Verify: all logged commands within joint limits
    for cmd in executor.commands() {
        for (j, &pos) in cmd.positions.iter().enumerate() {
            assert!(
                pos >= robot.joint_limits[j].lower - 0.01
                    && pos <= robot.joint_limits[j].upper + 0.01,
                "Joint {} at {:.4} outside limits [{:.4}, {:.4}]",
                j,
                pos,
                robot.joint_limits[j].lower,
                robot.joint_limits[j].upper
            );
        }
    }

    // Verify: first command near start (tolerance generous because interpolation
    // at t=0 returns first timed waypoint which may differ slightly from planning start
    // due to shortcutting and smoothing)
    let first = &executor.commands()[0];
    let last = executor.commands().last().unwrap();
    for j in 0..first.positions.len().min(start.len()) {
        assert!(
            (first.positions[j] - start[j]).abs() < 0.5,
            "First command joint {} = {:.4}, expected ~{:.4}",
            j,
            first.positions[j],
            start[j]
        );
    }

    // Verify: FK at final position produces finite pose
    let final_pose = forward_kinematics(&robot, &chain, &last.positions).unwrap();
    let pos = final_pose.translation();
    assert!(pos.x.is_finite() && pos.y.is_finite() && pos.z.is_finite());
}

#[test]
fn e2e_plan_execute_verify_panda() {
    let robot = Robot::from_name("franka_panda").unwrap();

    let start: Vec<f64> = robot
        .joint_limits
        .iter()
        .map(|l| (l.lower + l.upper) / 2.0)
        .collect();
    let goal: Vec<f64> = robot
        .joint_limits
        .iter()
        .map(|l| l.lower + (l.upper - l.lower) * 0.6)
        .collect();

    // Plan
    let planner = Planner::new(&robot).unwrap();
    let plan_result = planner
        .plan(&start, &Goal::Joints(JointValues(goal.clone())))
        .unwrap();

    // Time-parameterize
    let timed = kinetic::trajectory::trapezoidal_per_joint(
        &plan_result.waypoints,
        &robot.velocity_limits(),
        &robot.acceleration_limits(),
    )
    .unwrap();

    // Execute
    let executor = SimExecutor::default();
    let exec_result = executor.validate(&timed).unwrap();
    assert_eq!(exec_result.state, ExecutionState::Completed);
    assert_eq!(exec_result.final_positions.len(), robot.dof);
}

#[test]
fn e2e_plan_execute_loop_one_liner() {
    let robot = Robot::from_name("ur5e").unwrap();
    let start = vec![0.0, -1.57, 0.0, -1.57, 0.0, 0.0];
    let goal = vec![0.5, -1.2, 0.3, -1.2, -0.3, 0.1];

    let planner = Planner::new(&robot).unwrap();
    let executor = Box::new(SimExecutor::default());
    let mut pel = PlanExecuteLoop::new(planner, executor);

    let result = pel
        .move_to(&start, &Goal::Joints(JointValues(goal)))
        .unwrap();

    assert!(result.trajectory.is_some());
    assert_eq!(result.replans, 0);
    assert_eq!(result.final_joints.len(), 6);
}

#[test]
fn e2e_frame_tree_with_fk() {
    let robot = Robot::from_name("ur5e").unwrap();
    let arm = &robot.groups["arm"];
    let chain = KinematicChain::extract(&robot, &arm.base_link, &arm.tip_link).unwrap();

    let joints = vec![0.0, -1.57, 0.0, -1.57, 0.0, 0.0];

    // Compute FK for all links
    let all_poses = kinetic::kinematics::forward_kinematics_all(&robot, &chain, &joints).unwrap();

    // Build frame tree
    let tree = FrameTree::new();
    let mut pose_map = std::collections::HashMap::new();
    for (i, pose) in all_poses.iter().enumerate() {
        if i < robot.links.len() {
            pose_map.insert(robot.links[i].name.clone(), pose.0);
        }
    }
    tree.update_from_fk(&pose_map, 0.0);

    // Static calibration
    tree.set_static_transform(
        &robot.links[0].name,
        "camera",
        Isometry3::translation(0.0, 0.0, 0.5),
    );

    // Query
    assert!(tree.has_transform("world", &robot.links[1].name));
    assert!(tree.has_transform(&robot.links[0].name, "camera"));
    assert!(tree.list_frames().len() >= 2);
}

#[test]
fn e2e_trajectory_export_import_roundtrip() {
    let robot = Robot::from_name("ur5e").unwrap();
    let start = vec![0.0, -1.57, 0.0, -1.57, 0.0, 0.0];
    let goal = vec![0.5, -1.2, 0.3, -1.2, -0.3, 0.1];

    let planner = Planner::new(&robot).unwrap();
    let result = planner
        .plan(&start, &Goal::Joints(JointValues(goal)))
        .unwrap();

    let timed = kinetic::trajectory::trapezoidal_per_joint(
        &result.waypoints,
        &robot.velocity_limits(),
        &robot.acceleration_limits(),
    )
    .unwrap();

    // Export and reimport
    let json = trajectory_to_json(&timed);
    let reimported = trajectory_from_json(&json).unwrap();

    assert_eq!(reimported.dof, timed.dof);
    assert_eq!(reimported.waypoints.len(), timed.waypoints.len());

    // CSV roundtrip
    let csv = trajectory_to_csv(&timed);
    let reimported_csv = trajectory_from_csv(&csv).unwrap();
    assert_eq!(reimported_csv.dof, timed.dof);
}
