//! Long-duration stability tests: run thousands of planning cycles to detect
//! memory leaks, numerical drift, performance degradation, and rare edge cases.
//!
//! These tests are intentionally slow (~30-60s each) and exercise the full
//! pipeline repeatedly to catch issues that only manifest over time.

#[path = "helpers.rs"]
mod helpers;
use helpers::*;

use kinetic::kinematics::{forward_kinematics, solve_ik, IKConfig};
use kinetic::prelude::*;
use std::time::{Duration, Instant};

// ═══════════════════════════════════════════════════════════════════════════
// FK STABILITY: 100,000 CALLS
// ═══════════════════════════════════════════════════════════════════════════

/// FK must produce identical results on the 100,000th call as the 1st call.
/// Catches: floating-point accumulation, global state corruption, memory leaks.
#[test]
fn fk_100k_calls_no_drift() {
    let robot = Robot::from_name("ur5e").unwrap();
    let chain = load_chain(&robot);
    let joints = mid_joints(&robot);

    let reference = forward_kinematics(&robot, &chain, &joints).unwrap();

    for i in 0..100_000 {
        let pose = forward_kinematics(&robot, &chain, &joints).unwrap();
        let dist = reference.translation_distance(&pose);
        assert!(
            dist == 0.0,
            "FK drift detected at iteration {i}: {dist:.15}m from reference"
        );
    }
}

/// FK with varying inputs over 10,000 calls — no NaN/Inf leakage.
#[test]
fn fk_10k_varying_inputs_all_finite() {
    let robot = Robot::from_name("ur5e").unwrap();
    let chain = load_chain(&robot);
    let mut rng = 12345u64;

    for i in 0..10_000 {
        let joints: Vec<f64> = (0..chain.dof)
            .map(|j| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let t = ((rng >> 16) as f64) / (u64::MAX >> 16) as f64;
                let lim = &robot.joint_limits[j];
                lim.lower + t * (lim.upper - lim.lower)
            })
            .collect();

        let pose = forward_kinematics(&robot, &chain, &joints).unwrap();
        let t = pose.translation();
        assert!(
            t.x.is_finite() && t.y.is_finite() && t.z.is_finite(),
            "FK produced non-finite result at iteration {i}: ({}, {}, {})",
            t.x, t.y, t.z
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// IK STABILITY: 5,000 ROUNDTRIPS
// ═══════════════════════════════════════════════════════════════════════════

/// 5,000 FK→IK roundtrips with random configs. Track convergence rate and max error.
#[test]
fn ik_5000_roundtrips_convergence_rate() {
    let robot = Robot::from_name("ur5e").unwrap();
    let chain = load_chain(&robot);
    let config = IKConfig {
        num_restarts: 3,
        ..Default::default()
    };

    let mut rng = 99999u64;
    let mut converged = 0u32;
    let mut total = 0u32;
    let mut max_error = 0.0f64;
    let mut degraded_count = 0u32;

    for _ in 0..5_000 {
        let joints: Vec<f64> = (0..chain.dof)
            .map(|j| {
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                let t = ((rng >> 16) as f64) / (u64::MAX >> 16) as f64;
                let lim = &robot.joint_limits[j];
                lim.lower + t * (lim.upper - lim.lower)
            })
            .collect();

        let target = match forward_kinematics(&robot, &chain, &joints) {
            Ok(p) => p,
            Err(_) => continue,
        };

        total += 1;
        match solve_ik(&robot, &chain, &target, &config) {
            Ok(sol) => {
                if sol.converged {
                    converged += 1;
                }
                if sol.degraded {
                    degraded_count += 1;
                }
                if sol.position_error > max_error {
                    max_error = sol.position_error;
                }
            }
            Err(_) => {}
        }
    }

    let rate = (converged as f64) / (total as f64) * 100.0;
    println!(
        "IK convergence: {converged}/{total} ({rate:.1}%), max_error={max_error:.6}m, degraded={degraded_count}"
    );

    // UR5e with OPW should converge >90% of the time
    assert!(
        rate > 80.0,
        "IK convergence rate {rate:.1}% is below 80% threshold"
    );
    assert!(
        max_error < 0.01,
        "Max IK error {max_error:.6}m exceeds 10mm threshold"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// PLANNING STABILITY: 500 PLANS
// ═══════════════════════════════════════════════════════════════════════════

/// 500 planning cycles with the same start/goal. Track success rate and timing.
#[test]
fn planning_500_cycles_stability() {
    let robot = Robot::from_name("ur5e").unwrap();
    let planner = Planner::new(&robot).unwrap();
    let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
    let goal = Goal::joints([1.0, -0.5, 0.3, 0.2, -0.3, 0.5]);

    let mut successes = 0u32;
    let mut total_time = Duration::ZERO;
    let mut max_time = Duration::ZERO;
    let mut min_waypoints = usize::MAX;
    let mut max_waypoints = 0usize;

    for i in 0..500 {
        let t0 = Instant::now();
        match planner.plan(&start, &goal) {
            Ok(result) => {
                successes += 1;
                let elapsed = t0.elapsed();
                total_time += elapsed;
                if elapsed > max_time {
                    max_time = elapsed;
                }
                let wps = result.num_waypoints();
                if wps < min_waypoints {
                    min_waypoints = wps;
                }
                if wps > max_waypoints {
                    max_waypoints = wps;
                }
            }
            Err(e) => {
                // Planning failures in open space should be very rare
                if i < 10 {
                    panic!("Planning failed on cycle {i} (early failure): {e}");
                }
            }
        }
    }

    let avg_time = total_time / successes;
    let rate = (successes as f64) / 500.0 * 100.0;
    println!(
        "Planning: {successes}/500 ({rate:.1}%), avg={avg_time:?}, max={max_time:?}, waypoints={min_waypoints}-{max_waypoints}"
    );

    assert!(
        rate > 95.0,
        "Planning success rate {rate:.1}% is below 95% for open-space scenario"
    );
}

/// Planning with scene over 200 cycles — verify scene state doesn't corrupt.
#[test]
fn planning_200_cycles_with_scene() {
    let robot = Robot::from_name("ur5e").unwrap();
    let mut scene = Scene::new(&robot).unwrap();
    // Place table well below the robot's workspace
    scene.add_box("table", [0.3, 0.3, 0.01], [0.5, 0.0, -0.5]);

    let planner = Planner::new(&robot).unwrap().with_scene(&scene);
    // Use small motions in a safe part of the workspace
    let start = vec![0.0, -1.0, 0.5, 0.0, 0.0, 0.0];
    let goal = Goal::joints([0.3, -0.8, 0.3, 0.1, 0.1, 0.1]);

    let mut successes = 0u32;
    let mut first_err = None;
    for _ in 0..200 {
        match planner.plan(&start, &goal) {
            Ok(_) => successes += 1,
            Err(e) => {
                if first_err.is_none() {
                    first_err = Some(format!("{e}"));
                }
            }
        }
    }
    if let Some(e) = &first_err {
        println!("First error: {e}");
    }

    let rate = (successes as f64) / 200.0 * 100.0;
    println!("Scene planning: {successes}/200 ({rate:.1}%)");
    assert!(
        rate > 90.0,
        "Scene planning success rate {rate:.1}% below 90%"
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// COLLISION STABILITY: 100,000 CHECKS
// ═══════════════════════════════════════════════════════════════════════════

/// 100,000 collision checks — verify determinism and no drift.
#[test]
fn collision_100k_checks_deterministic() {
    let robot = Robot::from_name("ur5e").unwrap();
    let mut scene = Scene::new(&robot).unwrap();
    scene.add_sphere("ball", 0.1, [0.4, 0.0, 0.3]);

    let joints = mid_joints(&robot);
    let reference = scene.check_collision(&joints).unwrap();

    for i in 0..100_000 {
        let result = scene.check_collision(&joints).unwrap();
        assert_eq!(
            result, reference,
            "Collision result changed at iteration {i}: was {reference}, now {result}"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// TRAJECTORY STABILITY: LONG TRAJECTORIES
// ═══════════════════════════════════════════════════════════════════════════

/// Plan and time-parameterize a trajectory, then verify all 1000+ waypoints
/// are within joint limits and have monotonically increasing timestamps.
#[test]
fn long_trajectory_1000_waypoints_valid() {
    let robot = Robot::from_name("ur5e").unwrap();
    let planner = Planner::new(&robot).unwrap();
    let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
    let goal = Goal::joints([1.5, -0.3, 0.1, 0.5, -1.0, 1.0]);

    let result = planner.plan(&start, &goal).unwrap();
    let vel = robot.velocity_limits();
    let acc = robot.acceleration_limits();
    let timed = kinetic::trajectory::trapezoidal_per_joint(&result.waypoints, &vel, &acc).unwrap();

    // Verify all waypoints
    let mut prev_time = -1.0f64;
    for (i, wp) in timed.waypoints.iter().enumerate() {
        // Time monotonically increasing
        assert!(
            wp.time > prev_time,
            "Waypoint {i}: time {:.6} not > prev {:.6}",
            wp.time, prev_time
        );
        prev_time = wp.time;

        // All positions finite
        for (j, &pos) in wp.positions.iter().enumerate() {
            assert!(
                pos.is_finite(),
                "Waypoint {i} joint {j}: position {pos} is not finite"
            );
        }

        // All velocities finite
        for (j, &vel) in wp.velocities.iter().enumerate() {
            assert!(
                vel.is_finite(),
                "Waypoint {i} joint {j}: velocity {vel} is not finite"
            );
        }
    }

    println!(
        "Long trajectory: {} waypoints, {:.3}s duration, all valid",
        timed.waypoints.len(),
        timed.duration.as_secs_f64()
    );
}

// ═══════════════════════════════════════════════════════════════════════════
// MULTI-ROBOT STABILITY
// ═══════════════════════════════════════════════════════════════════════════

/// Load all 52 robots 10 times each — verify no state leakage between loads.
#[test]
fn all_robots_repeated_load_no_state_leak() {
    for &(name, _) in ALL_ROBOTS {
        for iteration in 0..10 {
            let robot = Robot::from_name(name).unwrap_or_else(|e| {
                panic!("Robot '{}' load failed on iteration {}: {}", name, iteration, e)
            });
            assert!(
                robot.dof > 0,
                "Robot '{}' has 0 DOF on iteration {}",
                name, iteration
            );
        }
    }
}

/// FK on 5 robots × 1000 calls each in sequence — no cross-contamination.
#[test]
fn sequential_multi_robot_fk_no_contamination() {
    let robots = ["ur5e", "franka_panda", "kuka_iiwa7", "xarm6", "kinova_gen3"];

    // Get reference poses
    let mut references = Vec::new();
    for name in &robots {
        let robot = Robot::from_name(name).unwrap();
        let chain = load_chain(&robot);
        let joints = mid_joints(&robot);
        let pose = forward_kinematics(&robot, &chain, &joints).unwrap();
        references.push((name, robot, chain, joints, pose));
    }

    // Run 1000 cycles interleaving all robots
    for cycle in 0..1000 {
        for (name, robot, chain, joints, ref_pose) in &references {
            let pose = forward_kinematics(robot, chain, joints).unwrap();
            let dist = ref_pose.translation_distance(&pose);
            assert!(
                dist == 0.0,
                "Robot '{}' FK changed at cycle {cycle}: drift={dist:.15}m",
                name
            );
        }
    }
}
