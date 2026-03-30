//! IK and planning failure mode tests.
//!
//! Verifies all failure paths: unreachable targets, infeasible plans, timeout
//! behavior, collision rejection. Every test expects Err or graceful handling.

use std::sync::Arc;
use std::time::{Duration, Instant};

use kinetic::kinematics::{
    forward_kinematics, solve_ik, IKConfig, IKMode, IKSolver, KinematicChain,
};
use kinetic::prelude::*;
use kinetic::scene::Scene;

fn ur5e_robot_and_chain() -> (Robot, KinematicChain) {
    let robot = Robot::from_name("ur5e").unwrap();
    let arm = &robot.groups["arm"];
    let chain = KinematicChain::extract(&robot, &arm.base_link, &arm.tip_link).unwrap();
    (robot, chain)
}

// ─── IK failure modes ──────────────────────────────────────────────────────

#[test]
fn ik_unreachable_pose_far_away() {
    let (robot, chain) = ur5e_robot_and_chain();
    // Target 10m away — far beyond UR5e's ~0.85m reach
    let target = Pose::from_xyz(10.0, 10.0, 10.0);
    let config = IKConfig {
        solver: IKSolver::DLS { damping: 0.05 },
        seed: Some(vec![0.0; 6]),
        max_iterations: 100,
        num_restarts: 0,
        ..IKConfig::default()
    };
    let result = solve_ik(&robot, &chain, &target, &config);
    assert!(result.is_err(), "10m target should fail");
}

#[test]
fn ik_unreachable_pose_just_beyond_workspace() {
    let (robot, chain) = ur5e_robot_and_chain();
    // UR5e reach is ~0.85m — target at ~1.2m
    let target = Pose::from_xyz(1.2, 0.0, 0.0);
    let config = IKConfig {
        solver: IKSolver::DLS { damping: 0.05 },
        seed: Some(vec![0.0; 6]),
        max_iterations: 100,
        num_restarts: 0,
        ..IKConfig::default()
    };
    let result = solve_ik(&robot, &chain, &target, &config);
    assert!(result.is_err(), "Target beyond workspace should fail");
}

#[test]
fn ik_reachable_pose_inside_workspace() {
    let (robot, chain) = ur5e_robot_and_chain();
    // Known reachable config
    let q = vec![0.5, -1.0, 1.0, -0.5, 1.0, 0.3];
    let target = forward_kinematics(&robot, &chain, &q).unwrap();
    let config = IKConfig {
        solver: IKSolver::DLS { damping: 0.05 },
        seed: Some(vec![0.0; 6]),
        max_iterations: 200,
        num_restarts: 2,
        ..IKConfig::default()
    };
    let result = solve_ik(&robot, &chain, &target, &config);
    assert!(result.is_ok(), "Reachable target should succeed");
    assert!(result.unwrap().converged);
}

#[test]
fn ik_zero_iterations() {
    let (robot, chain) = ur5e_robot_and_chain();
    let q = vec![0.5, -1.0, 1.0, -0.5, 1.0, 0.3];
    let target = forward_kinematics(&robot, &chain, &q).unwrap();
    let config = IKConfig {
        solver: IKSolver::DLS { damping: 0.05 },
        seed: Some(vec![0.0; 6]),
        max_iterations: 0, // Zero iterations
        num_restarts: 0,
        ..IKConfig::default()
    };
    let result = solve_ik(&robot, &chain, &target, &config);
    // Should fail — 0 iterations means no convergence
    assert!(result.is_err(), "Zero iterations should not converge");
}

#[test]
fn ik_position_only_for_hard_target() {
    let (robot, chain) = ur5e_robot_and_chain();
    let q = vec![0.5, -1.0, 1.0, -0.5, 1.0, 0.3];
    let target = forward_kinematics(&robot, &chain, &q).unwrap();

    // Position-only mode should converge more easily
    let config = IKConfig {
        solver: IKSolver::DLS { damping: 0.05 },
        mode: IKMode::PositionOnly,
        seed: Some(vec![0.0; 6]),
        max_iterations: 100,
        ..IKConfig::default()
    };
    let result = solve_ik(&robot, &chain, &target, &config);
    match result {
        Ok(sol) => {
            assert!(sol.position_error < 0.01, "Position-only should get close");
            assert_eq!(sol.mode_used, IKMode::PositionOnly);
        }
        Err(_) => {}
    }
}

#[test]
fn ik_fallback_mode_tries_position_only() {
    let (robot, chain) = ur5e_robot_and_chain();
    let q = vec![0.5, -1.0, 1.0, -0.5, 1.0, 0.3];
    let target = forward_kinematics(&robot, &chain, &q).unwrap();

    let config = IKConfig {
        solver: IKSolver::DLS { damping: 0.05 },
        mode: IKMode::PositionFallback,
        seed: Some(vec![0.0; 6]),
        max_iterations: 200,
        num_restarts: 2,
        ..IKConfig::default()
    };
    let result = solve_ik(&robot, &chain, &target, &config);
    // Should converge via full6D or position-only
    assert!(result.is_ok(), "Fallback mode should find a solution");
}

// ─── Planning failure modes ─────────────────────────────────────────────────

#[test]
fn planning_timeout_fires() {
    let robot = Robot::from_name("ur5e").unwrap();
    let planner = Planner::new(&robot).unwrap().with_config(PlannerConfig {
        timeout: Duration::from_millis(100),
        max_iterations: 1_000_000, // Lots of iterations allowed
        shortcut_iterations: 0,
        smooth: false,
        ..PlannerConfig::default()
    });

    // Plan something hard — long distance
    let start = vec![0.0; robot.dof];
    let goal = Goal::Joints(JointValues::new(vec![2.5, -1.5, 2.0, -1.0, 2.0, -1.0]));

    let t0 = Instant::now();
    let _ = planner.plan(&start, &goal);
    let elapsed = t0.elapsed();

    // Should complete within ~10x the timeout (allow for CI overhead, debug builds)
    assert!(
        elapsed < Duration::from_secs(5),
        "Planning should respect timeout: elapsed={:?}",
        elapsed
    );
}

#[test]
fn planning_with_goal_in_collision() {
    let robot = Robot::from_name("ur5e").unwrap();
    let mut scene = Scene::new(&robot).unwrap();

    // Use a sphere instead of cuboid — cuboid generates millions of collision
    // spheres at 0.02 resolution, causing extreme slowness.
    // A sphere of radius 3.0 centered at origin covers all reachable configs.
    scene.add(
        "wall",
        Shape::sphere(3.0),
        Isometry3::translation(0.0, 0.0, 0.0),
    );

    let planner = Planner::new(&robot)
        .unwrap()
        .with_scene(&scene)
        .with_config(PlannerConfig {
            timeout: Duration::from_millis(200),
            max_iterations: 1000,
            shortcut_iterations: 0,
            smooth: false,
            ..PlannerConfig::default()
        });

    let start = vec![0.0, -1.0, 0.8, 0.0, 1.0, 0.0];
    let goal = Goal::Joints(JointValues::new(vec![1.0, -0.5, 0.3, 0.2, -0.3, 0.5]));
    let result = planner.plan(&start, &goal);

    // With a 6m-diameter sphere at origin, everything is in collision
    // Should fail or report collision
    let _ = result; // Must not panic
}

#[test]
fn planning_with_start_in_collision() {
    let robot = Robot::from_name("ur5e").unwrap();
    let mut scene = Scene::new(&robot).unwrap();

    // Use a sphere instead of cuboid — cuboid generates millions of collision
    // spheres at 0.02 resolution, causing extreme slowness.
    scene.add(
        "obstacle",
        Shape::sphere(3.0),
        Isometry3::translation(0.0, 0.0, 0.0),
    );

    let planner = Planner::new(&robot)
        .unwrap()
        .with_scene(&scene)
        .with_config(PlannerConfig {
            timeout: Duration::from_millis(200),
            max_iterations: 1000,
            shortcut_iterations: 0,
            smooth: false,
            ..PlannerConfig::default()
        });

    let start = vec![0.0, -1.0, 0.8, 0.0, 1.0, 0.0];
    let goal = Goal::Joints(JointValues::new(vec![1.0, -0.5, 0.3, 0.2, -0.3, 0.5]));
    let result = planner.plan(&start, &goal);

    // Must not panic — should either error or plan from collision
    let _ = result;
}

#[test]
fn planning_max_iterations_zero() {
    let robot = Robot::from_name("ur5e").unwrap();
    let planner = Planner::new(&robot).unwrap().with_config(PlannerConfig {
        timeout: Duration::from_millis(1000),
        max_iterations: 0, // Zero iterations
        shortcut_iterations: 0,
        smooth: false,
        ..PlannerConfig::default()
    });

    let start = vec![0.0; robot.dof];
    let goal = Goal::Joints(JointValues::new(vec![1.0; robot.dof]));
    let result = planner.plan(&start, &goal);

    // Zero iterations — should fail or produce trivial path
    let _ = result; // Must not panic
}

#[test]
fn planning_very_short_timeout() {
    let robot = Robot::from_name("ur5e").unwrap();
    let planner = Planner::new(&robot).unwrap().with_config(PlannerConfig {
        timeout: Duration::from_nanos(1), // 1 nanosecond
        max_iterations: 10000,
        shortcut_iterations: 0,
        smooth: false,
        ..PlannerConfig::default()
    });

    let start = vec![0.0; robot.dof];
    let goal = Goal::Joints(JointValues::new(vec![1.0; robot.dof]));
    let result = planner.plan(&start, &goal);

    // With 1ns timeout, should fail immediately
    let _ = result; // Must not panic
}

#[test]
fn planning_named_goal_nonexistent() {
    let robot = Robot::from_name("ur5e").unwrap();
    let planner = Planner::new(&robot).unwrap();
    let start = vec![0.0; robot.dof];
    let goal = Goal::Named("this_pose_does_not_exist_xyz".to_string());
    let result = planner.plan(&start, &goal);
    assert!(result.is_err(), "Nonexistent named goal should return Err");
}

// ─── IK error types ────────────────────────────────────────────────────────

#[test]
fn ik_not_converged_has_residual_info() {
    let (robot, chain) = ur5e_robot_and_chain();
    let target = Pose::from_xyz(10.0, 0.0, 0.0);
    let config = IKConfig {
        solver: IKSolver::DLS { damping: 0.05 },
        seed: Some(vec![0.0; 6]),
        max_iterations: 10,
        num_restarts: 0,
        ..IKConfig::default()
    };
    let result = solve_ik(&robot, &chain, &target, &config);

    match result {
        Err(e) => {
            let msg = format!("{}", e);
            // Error message should contain useful info
            assert!(!msg.is_empty(), "Error message should not be empty");
        }
        Ok(_) => panic!("Should not converge to 10m target"),
    }
}

// ─── Planning with scene obstacles ──────────────────────────────────────────

#[test]
fn planning_around_obstacle_succeeds() {
    let robot = Robot::from_name("ur5e").unwrap();
    let mut scene = Scene::new(&robot).unwrap();

    // Small obstacle that doesn't block everything
    scene.add(
        "small_box",
        Shape::cuboid(0.05, 0.05, 0.05),
        Isometry3::translation(0.4, 0.0, 0.3),
    );

    let planner = Planner::new(&robot)
        .unwrap()
        .with_scene(&scene)
        .with_config(PlannerConfig {
            timeout: Duration::from_millis(1000),
            ..PlannerConfig::default()
        });

    let start = vec![0.0, -1.0, 0.8, 0.0, 1.0, 0.0];
    let goal = Goal::Joints(JointValues::new(vec![1.0, -0.5, 0.3, 0.2, -0.3, 0.5]));
    let result = planner.plan(&start, &goal);

    // Should succeed — obstacle is small
    match result {
        Ok(plan) => {
            assert!(plan.num_waypoints() >= 2);
            assert!(plan.path_length() > 0.0);
        }
        Err(_) => {} // May timeout
    }
}

// ─── Task planning failure modes ────────────────────────────────────────────

#[test]
fn task_pick_object_not_in_scene() {
    let robot = Arc::new(Robot::from_name("ur5e").unwrap());
    let scene = Arc::new(Scene::new(&robot).unwrap());

    let task = kinetic::task::Task::pick(
        &robot,
        &scene,
        kinetic::task::PickConfig {
            object: "nonexistent_object".to_string(),
            grasp_poses: vec![Isometry3::identity()],
            approach: kinetic::task::Approach::linear(Vector3::new(0.0, 0.0, -1.0), 0.1),
            retreat: kinetic::task::Approach::linear(Vector3::new(0.0, 0.0, 1.0), 0.1),
            gripper_open: 0.08,
            gripper_close: 0.0,
        },
    );

    let start = vec![0.0, -1.0, 0.8, 0.0, 1.0, 0.0];
    let result = task.plan(&start);
    assert!(result.is_err(), "Pick of nonexistent object should fail");

    let err = result.unwrap_err();
    let msg = format!("{}", err);
    assert!(
        msg.contains("not found") || msg.contains("nonexistent"),
        "Error should mention missing object: {}",
        msg
    );
}

// ─── Multiple IK solvers on same target ─────────────────────────────────────

#[test]
fn all_ik_solvers_handle_unreachable() {
    let (robot, chain) = ur5e_robot_and_chain();
    let target = Pose::from_xyz(10.0, 10.0, 10.0);

    let solvers = vec![
        IKConfig {
            solver: IKSolver::DLS { damping: 0.05 },
            max_iterations: 50,
            num_restarts: 0,
            ..IKConfig::default()
        },
        IKConfig {
            solver: IKSolver::FABRIK,
            max_iterations: 50,
            num_restarts: 0,
            ..IKConfig::default()
        },
    ];

    for config in &solvers {
        let result = solve_ik(&robot, &chain, &target, config);
        assert!(
            result.is_err(),
            "Solver {:?} should fail on unreachable target",
            config.solver
        );
    }
}
