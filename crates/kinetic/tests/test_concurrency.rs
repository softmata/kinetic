//! Concurrent FK/IK/planning thread-safety tests.
//!
//! Verifies that the major kinetic APIs are safe for concurrent use:
//! parallel FK/IK computation, concurrent planning, parallel trajectory
//! parameterization, and concurrent robot loading.

use std::sync::Arc;
use std::thread;
use std::time::Duration;

use kinetic::kinematics::{forward_kinematics, jacobian, solve_ik, IKConfig, KinematicChain};
use kinetic::prelude::*;
use kinetic::scene::Scene;
use kinetic::trajectory::trapezoidal;

fn ur5e_robot_and_chain() -> (Robot, KinematicChain) {
    let robot = Robot::from_name("ur5e").unwrap();
    let arm = &robot.groups["arm"];
    let chain = KinematicChain::extract(&robot, &arm.base_link, &arm.tip_link).unwrap();
    (robot, chain)
}

// ─── Compile-time Send+Sync assertions ──────────────────────────────────────

#[test]
fn robot_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<Robot>();
}

#[test]
fn kinematic_chain_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<KinematicChain>();
}

#[test]
fn joint_values_is_send_sync() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<JointValues>();
}

// ─── Concurrent FK ──────────────────────────────────────────────────────────

#[test]
fn concurrent_fk_8_threads() {
    let (robot, chain) = ur5e_robot_and_chain();
    let robot = Arc::new(robot);
    let chain = Arc::new(chain);

    let num_threads = 8;
    let iterations_per_thread = 1000;

    let handles: Vec<_> = (0..num_threads)
        .map(|t| {
            let robot = Arc::clone(&robot);
            let chain = Arc::clone(&chain);
            thread::spawn(move || {
                let mut results = Vec::with_capacity(iterations_per_thread);
                for i in 0..iterations_per_thread {
                    let offset = t as f64 * 0.1 + i as f64 * 0.001;
                    let joints = vec![offset, -1.0, 1.0, -0.5, 1.0, 0.3];
                    let pose = forward_kinematics(&robot, &chain, &joints).unwrap();
                    results.push(pose.translation());
                }
                results
            })
        })
        .collect();

    let mut total_results = 0;
    for h in handles {
        let results = h.join().expect("Thread should not panic");
        total_results += results.len();
    }

    assert_eq!(total_results, num_threads * iterations_per_thread);
}

// ─── Concurrent Jacobian ────────────────────────────────────────────────────

#[test]
fn concurrent_jacobian_8_threads() {
    let (robot, chain) = ur5e_robot_and_chain();
    let robot = Arc::new(robot);
    let chain = Arc::new(chain);

    let handles: Vec<_> = (0..8)
        .map(|t| {
            let robot = Arc::clone(&robot);
            let chain = Arc::clone(&chain);
            thread::spawn(move || {
                for i in 0..500 {
                    let offset = t as f64 * 0.1 + i as f64 * 0.002;
                    let joints = vec![offset, -1.0, 1.0, -0.5, 1.0, 0.3];
                    let jac = jacobian(&robot, &chain, &joints).unwrap();
                    assert_eq!(jac.nrows(), 6);
                    assert_eq!(jac.ncols(), 6);
                }
            })
        })
        .collect();

    for h in handles {
        h.join().expect("Thread should not panic");
    }
}

// ─── Concurrent IK ─────────────────────────────────────────────────────────

#[test]
fn concurrent_ik_8_threads() {
    let (robot, chain) = ur5e_robot_and_chain();
    let robot = Arc::new(robot);
    let chain = Arc::new(chain);

    let handles: Vec<_> = (0..8)
        .map(|t| {
            let robot = Arc::clone(&robot);
            let chain = Arc::clone(&chain);
            thread::spawn(move || {
                let mut converged = 0;
                for i in 0..50 {
                    let offset = t as f64 * 0.3 + i as f64 * 0.05;
                    let q = vec![offset, -1.0 + offset * 0.1, 1.0, -0.5, 1.0, 0.3];
                    let target = forward_kinematics(&robot, &chain, &q).unwrap();
                    let config = IKConfig {
                        max_iterations: 100,
                        num_restarts: 1,
                        ..IKConfig::default()
                    };
                    if let Ok(sol) = solve_ik(&robot, &chain, &target, &config) {
                        if sol.converged {
                            converged += 1;
                        }
                    }
                }
                converged
            })
        })
        .collect();

    let mut total_converged = 0;
    for h in handles {
        total_converged += h.join().expect("Thread should not panic");
    }

    eprintln!("Concurrent IK: {total_converged}/400 converged across 8 threads");
    assert!(
        total_converged > 0,
        "At least some IK solves should converge"
    );
}

// ─── Concurrent planning ───────────────────────────────────────────────────

#[test]
fn concurrent_planning_4_threads() {
    let robot = Arc::new(Robot::from_name("ur5e").unwrap());

    let goals = vec![
        vec![0.5, -1.0, 0.5, 0.0, 0.5, 0.0],
        vec![-0.5, -0.8, 0.3, 0.1, 0.3, -0.2],
        vec![1.0, -1.2, 0.8, -0.3, 0.7, 0.4],
        vec![0.0, -0.5, 1.2, 0.2, -0.5, 0.1],
    ];

    let handles: Vec<_> = goals
        .into_iter()
        .map(|goal| {
            let robot = Arc::clone(&robot);
            thread::spawn(move || {
                let planner = Planner::new(&robot).unwrap().with_config(PlannerConfig {
                    timeout: Duration::from_secs(2),
                    ..PlannerConfig::default()
                });
                let start = vec![0.0; robot.dof];
                planner.plan(&start, &Goal::Joints(JointValues::new(goal)))
            })
        })
        .collect();

    let mut successes = 0;
    for h in handles {
        let result = h.join().expect("Thread should not panic");
        if result.is_ok() {
            successes += 1;
        }
    }

    eprintln!("Concurrent planning: {successes}/4 succeeded");
    assert!(successes > 0, "At least one concurrent plan should succeed");
}

// ─── Concurrent planning with shared scene ──────────────────────────────────

#[test]
fn concurrent_planning_shared_scene() {
    let robot = Arc::new(Robot::from_name("ur5e").unwrap());
    let mut scene_mut = Scene::new(&robot).unwrap();
    scene_mut.add(
        "obstacle",
        Shape::cuboid(0.05, 0.05, 0.05),
        Isometry3::translation(0.4, 0.0, 0.3),
    );
    let scene = Arc::new(scene_mut);

    let handles: Vec<_> = (0..4)
        .map(|t| {
            let robot = Arc::clone(&robot);
            let scene = Arc::clone(&scene);
            thread::spawn(move || {
                let planner = Planner::new(&robot)
                    .unwrap()
                    .with_scene(&scene)
                    .with_config(PlannerConfig {
                        timeout: Duration::from_secs(2),
                        ..PlannerConfig::default()
                    });
                let offset = t as f64 * 0.3;
                let start = vec![offset, -1.0, 0.8, 0.0, 1.0, 0.0];
                let goal_joints = vec![offset + 0.5, -0.5, 0.3, 0.2, -0.3, 0.5];
                planner.plan(&start, &Goal::Joints(JointValues::new(goal_joints)))
            })
        })
        .collect();

    for h in handles {
        let _ = h.join().expect("Thread should not panic");
    }
}

// ─── Parallel trajectory parameterization ───────────────────────────────────

#[test]
fn parallel_trajectory_parameterization() {
    let handles: Vec<_> = (0..4)
        .map(|t| {
            thread::spawn(move || {
                let waypoints: Vec<Vec<f64>> = (0..100)
                    .map(|i| {
                        let phase = t as f64 * 0.5;
                        let s = (i as f64 / 100.0 * std::f64::consts::PI + phase).sin();
                        vec![s * 0.5; 6]
                    })
                    .collect();
                trapezoidal(&waypoints, 2.0, 5.0)
            })
        })
        .collect();

    for h in handles {
        let result = h.join().expect("Thread should not panic");
        match result {
            Ok(traj) => {
                assert!(traj.duration() > Duration::ZERO);
                assert!(traj.validate().is_ok());
            }
            Err(e) => {
                eprintln!("Trajectory failed (acceptable): {e}");
            }
        }
    }
}

// ─── Concurrent robot loading ───────────────────────────────────────────────

#[test]
fn concurrent_robot_loading() {
    let names = vec!["ur5e", "franka_panda", "kuka_iiwa7", "xarm6"];

    let handles: Vec<_> = names
        .into_iter()
        .map(|name| {
            thread::spawn(move || {
                let robot = Robot::from_name(name).unwrap();
                assert!(robot.dof > 0);
                robot
            })
        })
        .collect();

    let mut robots = Vec::new();
    for h in handles {
        robots.push(h.join().expect("Thread should not panic"));
    }

    assert_eq!(robots.len(), 4);
    // Verify they loaded different robots
    let names: Vec<&str> = robots.iter().map(|r| r.name.as_str()).collect();
    assert!(names.contains(&"ur5e"));
    assert!(names.contains(&"franka_panda"));
}

// ─── FK consistency across threads ──────────────────────────────────────────

#[test]
fn fk_results_consistent_across_threads() {
    let (robot, chain) = ur5e_robot_and_chain();
    let robot = Arc::new(robot);
    let chain = Arc::new(chain);
    let joints = vec![0.5, -1.0, 1.0, -0.5, 1.0, 0.3];

    // Compute reference result on main thread
    let reference = forward_kinematics(&robot, &chain, &joints).unwrap();
    let ref_t = reference.translation();

    // Compute on 8 threads and compare
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let robot = Arc::clone(&robot);
            let chain = Arc::clone(&chain);
            let joints = joints.clone();
            thread::spawn(move || {
                let pose = forward_kinematics(&robot, &chain, &joints).unwrap();
                pose.translation()
            })
        })
        .collect();

    for h in handles {
        let t = h.join().expect("Thread should not panic");
        assert!(
            (t[0] - ref_t[0]).abs() < 1e-12
                && (t[1] - ref_t[1]).abs() < 1e-12
                && (t[2] - ref_t[2]).abs() < 1e-12,
            "FK result mismatch: {:?} vs {:?}",
            t,
            ref_t
        );
    }
}
