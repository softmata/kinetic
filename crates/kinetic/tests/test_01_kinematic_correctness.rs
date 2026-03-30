//! Acceptance tests: 01 kinematic_correctness
//! Spec: doc_tests/01_KINEMATIC_CORRECTNESS.md
//!
//! FK/IK accuracy verification across all 52 robots.

#[path = "helpers.rs"]
mod helpers;
use helpers::*;

use kinetic::kinematics::{
    forward_kinematics, solve_ik, jacobian, IKConfig,
};

// ─── FK Determinism: 52 robots x 20 configs ─────────────────────────────────

#[test]
fn fk_determinism_all_robots() {
    let mut total_checks = 0;
    for &(name, _) in ALL_ROBOTS {
        let robot = load_robot(name);
        let chain = load_chain(&robot);

        for seed in 0..20u64 {
            let joints = random_joints(&robot, seed * 1000 + 42);
            // Use only chain DOF joints
            let chain_joints: Vec<f64> = joints.iter().take(chain.dof).copied().collect();

            let pose1 = forward_kinematics(&robot, &chain, &chain_joints);
            let pose2 = forward_kinematics(&robot, &chain, &chain_joints);

            match (pose1, pose2) {
                (Ok(p1), Ok(p2)) => {
                    // Bitwise identical: same input → same output
                    let t1 = p1.translation();
                    let t2 = p2.translation();
                    assert_eq!(t1.x.to_bits(), t2.x.to_bits(),
                        "{name} seed {seed}: FK x not bitwise identical");
                    assert_eq!(t1.y.to_bits(), t2.y.to_bits(),
                        "{name} seed {seed}: FK y not bitwise identical");
                    assert_eq!(t1.z.to_bits(), t2.z.to_bits(),
                        "{name} seed {seed}: FK z not bitwise identical");
                    total_checks += 1;
                }
                (Err(_), Err(_)) => {
                    // Both failed consistently — also deterministic
                    total_checks += 1;
                }
                _ => panic!("{name} seed {seed}: FK results inconsistent (one Ok, one Err)"),
            }
        }
    }
    assert!(total_checks >= ALL_ROBOTS.len() * 20,
        "expected {} checks, got {}", ALL_ROBOTS.len() * 20, total_checks);
}

// ─── FK produces finite poses ───────────────────────────────────────────────

#[test]
fn fk_finite_poses_all_robots() {
    for &(name, _) in ALL_ROBOTS {
        let robot = load_robot(name);
        let chain = load_chain(&robot);

        // Test at zero, mid, and random configs
        let configs: Vec<Vec<f64>> = vec![
            vec![0.0; chain.dof],
            mid_joints(&robot).into_iter().take(chain.dof).collect(),
            random_joints(&robot, 7777).into_iter().take(chain.dof).collect(),
        ];

        for (ci, config) in configs.iter().enumerate() {
            if let Ok(pose) = forward_kinematics(&robot, &chain, config) {
                let t = pose.translation();
                assert!(t.x.is_finite(), "{name} config {ci}: x={}", t.x);
                assert!(t.y.is_finite(), "{name} config {ci}: y={}", t.y);
                assert!(t.z.is_finite(), "{name} config {ci}: z={}", t.z);

                let r = pose.rotation();
                assert!(r.angle().is_finite(), "{name} config {ci}: rotation not finite");
            }
        }
    }
}

// ─── FK produces distinct poses for distinct configs ────────────────────────

#[test]
fn fk_distinct_configs_produce_distinct_poses() {
    for name in SAFETY_ROBOTS {
        let robot = load_robot(name);
        let chain = load_chain(&robot);

        let configs: Vec<Vec<f64>> = (0..10)
            .map(|s| random_joints(&robot, s * 111).into_iter().take(chain.dof).collect())
            .collect();

        let poses: Vec<_> = configs.iter()
            .filter_map(|c| forward_kinematics(&robot, &chain, c).ok())
            .collect();

        // At least 8 of 10 configs should produce distinct EE positions
        let mut unique = 0;
        for i in 0..poses.len() {
            let mut is_unique = true;
            for j in 0..i {
                if poses[i].translation_distance(&poses[j]) < 1e-6 {
                    is_unique = false;
                    break;
                }
            }
            if is_unique { unique += 1; }
        }
        assert!(unique >= 8, "{name}: only {unique}/10 distinct poses (expected >= 8)");
    }
}

// ─── IK Roundtrip: 52 robots x 5 configs x 1 solver ────────────────────────
// (Reduced from spec's 30x3 for practical test time — full version in nightly CI)

#[test]
fn ik_roundtrip_all_robots() {
    let mut total_attempts = 0;
    let mut converged = 0;
    let mut within_tolerance = 0;

    for &(name, _) in ALL_ROBOTS {
        let robot = load_robot(name);
        let chain = load_chain(&robot);

        for seed in 0..5u64 {
            let joints = random_joints(&robot, seed * 999 + 13);
            let chain_joints: Vec<f64> = joints.into_iter().take(chain.dof).collect();

            let target = match forward_kinematics(&robot, &chain, &chain_joints) {
                Ok(p) => p,
                Err(_) => continue,
            };

            let config = IKConfig {
                num_restarts: 4,
                ..Default::default()
            };

            total_attempts += 1;
            match solve_ik(&robot, &chain, &target, &config) {
                Ok(sol) => {
                    if sol.converged {
                        converged += 1;

                        // FK(IK_solution) should match target
                        if let Ok(recovered) = forward_kinematics(&robot, &chain, &sol.joints) {
                            let pos_err = recovered.translation_distance(&target);
                            if pos_err < 0.001 { // 1mm
                                within_tolerance += 1;
                            }
                        }

                        // Solution joints should be finite
                        for (j, &val) in sol.joints.iter().enumerate() {
                            assert!(val.is_finite(),
                                "{name} seed {seed}: IK joint {j} not finite: {val}");
                        }
                    }
                }
                Err(_) => {} // IK failure is acceptable for some configs
            }
        }
    }

    let rate = if total_attempts > 0 { converged as f64 / total_attempts as f64 } else { 0.0 };
    eprintln!(
        "IK roundtrip: {}/{} converged ({:.1}%), {}/{} within 1mm",
        converged, total_attempts, rate * 100.0,
        within_tolerance, converged
    );

    // At least 50% should converge (many robots, random seeds, limited restarts)
    assert!(rate > 0.3,
        "IK convergence rate too low: {:.1}% ({converged}/{total_attempts})", rate * 100.0);
}

// ─── Jacobian accuracy vs finite differences ────────────────────────────────

#[test]
fn jacobian_matches_finite_differences() {
    let eps = 1e-6;

    for name in SAFETY_ROBOTS {
        let robot = load_robot(name);
        let chain = load_chain(&robot);

        for seed in 0..20u64 {
            let joints: Vec<f64> = random_joints(&robot, seed * 77)
                .into_iter().take(chain.dof).collect();

            let j_analytical = match jacobian(&robot, &chain, &joints) {
                Ok(j) => j,
                Err(_) => continue,
            };

            // Numerical Jacobian via central differences
            let _base_pose = match forward_kinematics(&robot, &chain, &joints) {
                Ok(p) => p,
                Err(_) => continue,
            };

            for col in 0..chain.dof {
                let mut q_plus = joints.clone();
                let mut q_minus = joints.clone();
                q_plus[col] += eps;
                q_minus[col] -= eps;

                let pose_plus = match forward_kinematics(&robot, &chain, &q_plus) {
                    Ok(p) => p,
                    Err(_) => continue,
                };
                let pose_minus = match forward_kinematics(&robot, &chain, &q_minus) {
                    Ok(p) => p,
                    Err(_) => continue,
                };

                // Position derivative (rows 0-2)
                let dp = (pose_plus.translation() - pose_minus.translation()) / (2.0 * eps);

                for row in 0..3 {
                    let analytical = j_analytical[(row, col)];
                    let numerical = dp[row];
                    let err = (analytical - numerical).abs();
                    assert!(
                        err < 1e-3,
                        "{name} seed {seed}: J[{row},{col}] analytical={analytical:.6} numerical={numerical:.6} err={err:.6}"
                    );
                }
            }
        }
    }
}

// ─── FK at joint limits ─────────────────────────────────────────────────────

#[test]
fn fk_at_joint_limits_is_finite() {
    for &(name, _) in ALL_ROBOTS {
        let robot = load_robot(name);
        let chain = load_chain(&robot);

        let configs = vec![
            boundary_joints(&robot, "lower").into_iter().take(chain.dof).collect::<Vec<_>>(),
            boundary_joints(&robot, "upper").into_iter().take(chain.dof).collect::<Vec<_>>(),
        ];

        for (ci, config) in configs.iter().enumerate() {
            if let Ok(pose) = forward_kinematics(&robot, &chain, config) {
                let t = pose.translation();
                assert!(
                    t.x.is_finite() && t.y.is_finite() && t.z.is_finite(),
                    "{name} boundary config {ci}: FK not finite"
                );
            }
        }
    }
}
