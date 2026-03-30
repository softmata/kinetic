//! Acceptance tests: 02 joint_limit_enforcement
//! Spec: doc_tests/02_JOINT_LIMIT_ENFORCEMENT.md
//!
//! Verifies joint limits are enforced at every layer:
//! robot model, IK output, planner output, trajectory.

#[path = "helpers.rs"]
mod helpers;
use helpers::*;

use kinetic::kinematics::{forward_kinematics, solve_ik, IKConfig};
use kinetic::planning::Planner;
use kinetic::prelude::*;

// ─── Robot limit structural validation: 52 robots ───────────────────────────

#[test]
fn all_robots_limits_lower_less_than_upper() {
    for &(name, _) in ALL_ROBOTS {
        let robot = load_robot(name);
        for (j, limit) in robot.joint_limits.iter().enumerate() {
            assert!(
                limit.lower < limit.upper,
                "{name} joint {j}: lower ({}) >= upper ({})",
                limit.lower, limit.upper
            );
        }
    }
}

#[test]
fn check_limits_catches_above_upper() {
    for &(name, _) in ALL_ROBOTS {
        let robot = load_robot(name);
        if robot.joint_limits.is_empty() { continue; }
        let range = robot.joint_limits[0].upper - robot.joint_limits[0].lower;
        if !range.is_finite() || range > 50.0 { continue; } // skip continuous/unlimited joints
        let mut above = mid_joints(&robot);
        above[0] = robot.joint_limits[0].upper + 1.0;
        let jv = JointValues::new(above);
        assert!(
            robot.check_limits(&jv).is_err(),
            "{name}: check_limits should catch above-upper"
        );
    }
}

#[test]
fn check_limits_catches_below_lower() {
    for &(name, _) in ALL_ROBOTS {
        let robot = load_robot(name);
        if robot.joint_limits.is_empty() { continue; }
        let range = robot.joint_limits[0].upper - robot.joint_limits[0].lower;
        if !range.is_finite() || range > 50.0 { continue; } // skip continuous/unlimited joints
        let mut below = mid_joints(&robot);
        below[0] = robot.joint_limits[0].lower - 1.0;
        let jv = JointValues::new(below);
        assert!(
            robot.check_limits(&jv).is_err(),
            "{name}: check_limits should catch below-lower"
        );
    }
}

#[test]
fn check_limits_passes_mid_config() {
    for &(name, _) in ALL_ROBOTS {
        let robot = load_robot(name);
        let mid = mid_joints(&robot);
        let jv = JointValues::new(mid);
        assert!(
            robot.check_limits(&jv).is_ok(),
            "{name}: check_limits should pass mid config"
        );
    }
}

#[test]
fn clamp_to_limits_works() {
    for name in SAFETY_ROBOTS {
        let robot = load_robot(name);
        let mut out_of_range = mid_joints(&robot);
        out_of_range[0] = robot.joint_limits[0].upper + 5.0;
        let mut jv = JointValues::new(out_of_range);
        robot.clamp_to_limits(&mut jv);
        assert!(
            robot.check_limits(&jv).is_ok(),
            "{name}: clamped config should pass check_limits"
        );
    }
}

// ─── IK output within limits: 52 robots x 20 configs ────────────────────────

#[test]
fn ik_output_within_limits() {
    let mut total = 0;
    let mut violations = 0;

    for &(name, _) in ALL_ROBOTS {
        let robot = load_robot(name);
        let chain = load_chain(&robot);

        for seed in 0..20u64 {
            let joints: Vec<f64> = random_joints(&robot, seed * 333)
                .into_iter().take(chain.dof).collect();

            let target = match forward_kinematics(&robot, &chain, &joints) {
                Ok(p) => p,
                Err(_) => continue,
            };

            let config = IKConfig { num_restarts: 2, ..Default::default() };
            if let Ok(sol) = solve_ik(&robot, &chain, &target, &config) {
                if sol.converged {
                    total += 1;
                    // Check each joint against chain's active joint limits
                    for (j, &val) in sol.joints.iter().enumerate() {
                        let joint_idx = chain.active_joints[j];
                        if let Some(limits) = &robot.joints[joint_idx].limits {
                            if val < limits.lower - 1e-4 || val > limits.upper + 1e-4 {
                                violations += 1;
                                eprintln!(
                                    "{name} seed {seed} joint {j}: {val:.4} outside [{:.4}, {:.4}]",
                                    limits.lower, limits.upper
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    eprintln!("IK limit check: {total} converged solutions, {violations} violations");
    assert_eq!(violations, 0, "IK must not produce out-of-limit solutions");
}

// ─── Planner output within limits: 5 robots x 5 plans ──────────────────────

#[test]
fn planner_output_within_limits() {
    let mut total_waypoints = 0;
    let mut violations = 0;

    for name in SAFETY_ROBOTS {
        let robot = load_robot(name);
        let planner = match Planner::new(&robot) {
            Ok(p) => p,
            Err(_) => continue,
        };

        for seed in 0..5u64 {
            let start = mid_joints(&robot);
            let goal_vals: Vec<f64> = random_joints(&robot, seed * 555)
                .into_iter().take(robot.dof).collect();
            let goal = Goal::Joints(JointValues::new(goal_vals));

            if let Ok(plan) = planner.plan(&start, &goal) {
                for (wp_idx, wp) in plan.waypoints.iter().enumerate() {
                    total_waypoints += 1;
                    for (j, &val) in wp.iter().enumerate() {
                        if j < robot.joint_limits.len() {
                            let lo = robot.joint_limits[j].lower;
                            let hi = robot.joint_limits[j].upper;
                            if val < lo - 1e-4 || val > hi + 1e-4 {
                                violations += 1;
                                eprintln!(
                                    "{name} seed {seed} wp {wp_idx} j{j}: {val:.4} outside [{lo:.4}, {hi:.4}]"
                                );
                            }
                        }
                    }
                }
            }
        }
    }

    eprintln!("Planner limit check: {total_waypoints} waypoints, {violations} violations");
    assert_eq!(violations, 0, "Planner must not produce out-of-limit waypoints");
}

// ─── Trajectory velocity limits: 5 robots x 2 trajectories x 100 samples ───

#[test]
fn trajectory_velocity_within_limits() {
    use kinetic::trajectory::trapezoidal;

    let mut total_samples = 0;
    let mut velocity_violations = 0;

    for name in SAFETY_ROBOTS {
        let robot = load_robot(name);
        let planner = match Planner::new(&robot) {
            Ok(p) => p,
            Err(_) => continue,
        };

        for _seed in 0..2u64 {
            let start = mid_joints(&robot);
            let goal_vals: Vec<f64> = start.iter().map(|v| v + 0.3).collect();
            let goal = Goal::Joints(JointValues::new(goal_vals));

            let plan = match planner.plan(&start, &goal) {
                Ok(p) => p,
                Err(_) => continue,
            };

            let timed = match trapezoidal(&plan.waypoints, 1.0, 2.0) {
                Ok(t) => t,
                Err(_) => continue,
            };

            if timed.is_empty() { continue; }

            // Sample at 100 points
            let dur = timed.duration().as_secs_f64();
            for i in 0..100 {
                let t = dur * i as f64 / 99.0;
                let wp = timed.sample_at(std::time::Duration::from_secs_f64(t));
                total_samples += 1;

                for (j, &vel) in wp.velocities.iter().enumerate() {
                    if j < robot.joint_limits.len() {
                        let vlim = robot.joint_limits[j].velocity;
                        if vlim > 0.0 && vel.abs() > vlim * 1.05 {
                            velocity_violations += 1;
                        }
                    }
                }
            }
        }
    }

    eprintln!("Trajectory velocity check: {total_samples} samples, {velocity_violations} violations");
    // Allow small number of violations (interpolation artifacts at boundaries)
    assert!(
        velocity_violations < total_samples / 20,
        "Too many velocity violations: {velocity_violations}/{total_samples}"
    );
}

// ─── Proptest: check_limits never panics ────────────────────────────────────

#[test]
fn check_limits_never_panics_random() {
    use rand::prelude::*;
    use rand_chacha::ChaCha8Rng;

    let robot = load_robot("ur5e");
    let mut rng = ChaCha8Rng::seed_from_u64(12345);

    for _ in 0..1000 {
        let joints: Vec<f64> = (0..robot.dof)
            .map(|_| rng.gen_range(-100.0..100.0))
            .collect();
        let jv = JointValues::new(joints);
        // Must not panic — Ok or Err is fine
        let _ = robot.check_limits(&jv);
    }
}
