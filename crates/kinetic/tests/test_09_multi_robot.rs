//! Acceptance tests: 09 multi_robot
//! Spec: doc_tests/09_MULTI_ROBOT.md
//!
//! Dual-arm planning, thread safety, concurrent planners.

#[path = "helpers.rs"]
mod helpers;
use helpers::*;

use kinetic::prelude::*;
use kinetic::planning::Planner;
use std::sync::Arc;

// ─── Independent planning isolation: 2 threads, different robots ────────────

#[test]
fn independent_planning_isolation() {
    let handle1 = std::thread::spawn(|| {
        let robot = load_robot("ur5e");
        let planner = Planner::new(&robot).unwrap();
        let start = mid_joints(&robot);
        let goal = Goal::Joints(JointValues::new(start.iter().map(|v| v + 0.2).collect()));
        let result = planner.plan(&start, &goal);
        assert!(result.is_ok(), "Thread 1 (ur5e) failed: {:?}", result.err());
        let plan = result.unwrap();
        assert_eq!(plan.waypoints[0].len(), 6, "UR5e should have 6-DOF waypoints");
        6 // return DOF
    });

    let handle2 = std::thread::spawn(|| {
        let robot = load_robot("xarm6");
        let planner = Planner::new(&robot).unwrap();
        let start = mid_joints(&robot);
        let goal = Goal::Joints(JointValues::new(start.iter().map(|v| v + 0.2).collect()));
        let result = planner.plan(&start, &goal);
        assert!(result.is_ok(), "Thread 2 (xarm6) failed: {:?}", result.err());
        let plan = result.unwrap();
        assert_eq!(plan.waypoints[0].len(), 6, "xArm6 should have 6-DOF waypoints");
        6
    });

    let dof1 = handle1.join().expect("Thread 1 panicked");
    let dof2 = handle2.join().expect("Thread 2 panicked");
    assert_eq!(dof1, 6);
    assert_eq!(dof2, 6);
}

// ─── Multiple concurrent planners on same robot ─────────────────────────────

#[test]
fn concurrent_planners_same_robot() {
    let robot = Arc::new(load_robot("ur5e"));

    let handles: Vec<_> = (0..4).map(|i| {
        let robot = robot.clone();
        std::thread::spawn(move || {
            let planner = Planner::new(&robot).unwrap();
            let start = mid_joints(&robot);
            let offset = 0.1 + i as f64 * 0.05;
            let goal = Goal::Joints(JointValues::new(
                start.iter().map(|v| v + offset).collect()
            ));
            let result = planner.plan(&start, &goal);
            match result {
                Ok(plan) => {
                    assert!(plan.num_waypoints() >= 2, "Thread {i}: plan too short");
                    true
                }
                Err(_) => false,
            }
        })
    }).collect();

    let mut succeeded = 0;
    for h in handles {
        if h.join().expect("thread panicked") {
            succeeded += 1;
        }
    }
    assert!(succeeded >= 3, "at least 3/4 concurrent plans should succeed: {succeeded}/4");
}

// ─── Concurrent FK is thread-safe ───────────────────────────────────────────

#[test]
fn concurrent_fk_thread_safe() {
    let robot = Arc::new(load_robot("ur5e"));

    let handles: Vec<_> = (0..8).map(|seed| {
        let robot = robot.clone();
        std::thread::spawn(move || {
            let chain = load_chain(&robot);
            let joints: Vec<f64> = random_joints(&robot, seed * 111)
                .into_iter().take(chain.dof).collect();
            let result = kinetic::kinematics::forward_kinematics(&robot, &chain, &joints);
            assert!(result.is_ok(), "FK should succeed on thread {seed}");
            let pose = result.unwrap();
            assert!(pose.translation().x.is_finite(), "FK result should be finite on thread {seed}");
        })
    }).collect();

    for h in handles {
        h.join().expect("FK thread panicked");
    }
}

// ─── Concurrent IK is thread-safe ──────────────────────────────────────────

#[test]
fn concurrent_ik_thread_safe() {
    let robot = Arc::new(load_robot("ur5e"));

    let handles: Vec<_> = (0..4).map(|seed| {
        let robot = robot.clone();
        std::thread::spawn(move || {
            let chain = load_chain(&robot);
            let joints: Vec<f64> = random_joints(&robot, seed * 222)
                .into_iter().take(chain.dof).collect();
            let target = kinetic::kinematics::forward_kinematics(&robot, &chain, &joints).unwrap();
            let config = kinetic::kinematics::IKConfig { num_restarts: 2, ..Default::default() };
            // Must not panic regardless of convergence
            let _ = kinetic::kinematics::solve_ik(&robot, &chain, &target, &config);
        })
    }).collect();

    for h in handles {
        h.join().expect("IK thread panicked");
    }
}

// ─── No data races in Robot struct ──────────────────────────────────────────

#[test]
fn robot_shared_across_threads_safely() {
    // Robot is read-only after construction — verify Arc<Robot> is Send+Sync
    let robot: Arc<Robot> = Arc::new(load_robot("franka_panda"));

    let handles: Vec<_> = (0..6).map(|i| {
        let robot = robot.clone();
        std::thread::spawn(move || {
            // Multiple threads reading robot properties concurrently
            let _dof = robot.dof;
            let _name = &robot.name;
            let _limits = &robot.joint_limits;
            let _mid = robot.mid_configuration();
            let _zero = robot.zero_configuration();
            i
        })
    }).collect();

    for h in handles {
        h.join().expect("reader thread panicked");
    }
}
