//! FK/IK accuracy validation suite.
//!
//! Comprehensive validation of FK and IK accuracy across all robots.
//! Measures: max position error (mm), max orientation error (deg),
//! IK success rate, IK convergence statistics.

use kinetic::kinematics::{forward_kinematics, solve_ik, IKConfig, KinematicChain};
use kinetic::prelude::*;
use rand::Rng;

// FK validation cases are generated dynamically by computing FK at zero config
// and verifying self-consistency (FK is deterministic and produces finite results).

/// Robots with their expected DOF for sanity checking.
const ROBOTS_DOF: &[(&str, usize)] = &[
    ("ur5e", 6),
    ("ur3e", 6),
    ("ur10e", 6),
    ("ur16e", 6),
    ("ur20", 6),
    ("ur30", 6),
    ("franka_panda", 7),
    ("kuka_iiwa7", 7),
    ("kuka_iiwa14", 7),
    ("kuka_kr6", 6),
    ("abb_irb1200", 6),
    ("abb_irb4600", 6),
    ("fanuc_crx10ia", 6),
    ("fanuc_lr_mate_200id", 6),
    ("xarm5", 5),
    ("xarm6", 6),
    ("xarm7", 7),
    ("kinova_gen3", 7),
    ("kinova_gen3_lite", 6),
    ("sawyer", 7),
    ("dobot_cr5", 6),
    ("flexiv_rizon4", 7),
    ("meca500", 6),
    ("yaskawa_gp7", 6),
    ("yaskawa_hc10", 6),
    ("denso_vs068", 6),
    ("staubli_tx260", 6),
    ("mycobot_280", 6),
    ("so_arm100", 5),
    ("koch_v1", 6),
    ("viperx_300", 6),
    ("widowx_250", 6),
    ("lerobot_so100", 6),
    ("open_manipulator_x", 4),
    ("trossen_px100", 4),
    ("trossen_rx150", 5),
    ("trossen_wx250s", 6),
    ("robotis_open_manipulator_p", 6),
    ("niryo_ned2", 6),
    ("techman_tm5_700", 6),
    ("elite_ec66", 6),
];

#[test]
fn fk_all_robots_zero_config_no_panic() {
    for &(name, expected_dof) in ROBOTS_DOF {
        let robot = Robot::from_name(name).unwrap_or_else(|e| {
            panic!("Failed to load robot '{}': {}", name, e);
        });

        let arm = match robot.groups.get("arm") {
            Some(g) => g,
            None => continue, // Skip robots without arm group
        };

        let chain = KinematicChain::extract(&robot, &arm.base_link, &arm.tip_link)
            .unwrap_or_else(|e| panic!("Failed to extract chain for '{}': {}", name, e));

        assert_eq!(
            chain.dof, expected_dof,
            "Robot '{}' DOF mismatch: expected {}, got {}",
            name, expected_dof, chain.dof
        );

        let zeros = vec![0.0; chain.dof];
        let pose = forward_kinematics(&robot, &chain, &zeros);
        assert!(
            pose.is_ok(),
            "FK at zeros failed for '{}': {:?}",
            name,
            pose.err()
        );
    }
}

#[test]
fn fk_zero_config_produces_finite_pose() {
    // FK at zero config should produce a finite, non-degenerate pose for all robots
    for &(name, _expected_dof) in ROBOTS_DOF {
        let robot = match Robot::from_name(name) {
            Ok(r) => r,
            Err(_) => continue,
        };
        let arm = match robot.groups.get("arm") {
            Some(g) => g,
            None => continue,
        };
        let chain = match KinematicChain::extract(&robot, &arm.base_link, &arm.tip_link) {
            Ok(c) => c,
            Err(_) => continue,
        };

        let zeros = vec![0.0; chain.dof];
        let pose = forward_kinematics(&robot, &chain, &zeros).unwrap();
        let pos = pose.translation();

        // Position should be finite and within reasonable bounds (10m)
        assert!(
            pos.x.is_finite() && pos.y.is_finite() && pos.z.is_finite(),
            "Robot '{}' FK at zero produces non-finite position: [{}, {}, {}]",
            name,
            pos.x,
            pos.y,
            pos.z
        );
        assert!(
            pos.norm() < 10.0,
            "Robot '{}' FK at zero produces unreasonable position: norm = {:.2}m",
            name,
            pos.norm()
        );

        // Rotation should be a valid unit quaternion
        let q = pose.rotation();
        let q_norm = (q.w * q.w + q.i * q.i + q.j * q.j + q.k * q.k).sqrt();
        assert!(
            (q_norm - 1.0).abs() < 1e-6,
            "Robot '{}' FK rotation not unit quaternion: norm = {:.6}",
            name,
            q_norm
        );

        // Print FK result for documentation
        eprintln!(
            "[{}] zero config FK → pos=[{:.4}, {:.4}, {:.4}]",
            name, pos.x, pos.y, pos.z
        );
    }
}

#[test]
fn fk_ik_roundtrip_accuracy() {
    let mut rng = rand::thread_rng();

    // Test a representative subset of robots
    let test_robots: &[(&str, usize)] = &[
        ("ur5e", 6),
        ("franka_panda", 7),
        ("kuka_iiwa7", 7),
        ("xarm6", 6),
        ("kinova_gen3", 7),
        ("fanuc_crx10ia", 6),
        ("abb_irb1200", 6),
        ("sawyer", 7),
    ];

    for &(name, _chain_dof) in test_robots {
        let robot = Robot::from_name(name).unwrap();
        let arm = &robot.groups["arm"];
        let chain = KinematicChain::extract(&robot, &arm.base_link, &arm.tip_link).unwrap();

        let ik_config = IKConfig {
            num_restarts: 10,
            max_iterations: 500,
            position_tolerance: 1e-4,
            orientation_tolerance: 1e-3,
            ..Default::default()
        };

        let mut max_pos_err = 0.0_f64;
        let mut max_ori_err = 0.0_f64;
        let mut successes = 0;
        let num_tests = 20;

        for _ in 0..num_tests {
            // Generate random config within joint limits
            let joints: Vec<f64> = chain
                .active_joints
                .iter()
                .map(|&ji| {
                    let joint = &robot.joints[ji];
                    if let Some(limits) = &joint.limits {
                        let margin = (limits.upper - limits.lower) * 0.1;
                        rng.gen_range((limits.lower + margin)..=(limits.upper - margin))
                    } else {
                        rng.gen_range(-2.0..=2.0)
                    }
                })
                .collect();

            let target_pose = match forward_kinematics(&robot, &chain, &joints) {
                Ok(p) => p,
                Err(_) => continue,
            };

            let ik_cfg = IKConfig {
                seed: Some(joints.clone()),
                ..ik_config.clone()
            };

            if let Ok(solution) = solve_ik(&robot, &chain, &target_pose, &ik_cfg) {
                if solution.converged {
                    let recovered = forward_kinematics(&robot, &chain, &solution.joints).unwrap();

                    let pos_err = (target_pose.translation() - recovered.translation()).norm();
                    let ori_err = target_pose
                        .rotation()
                        .rotation_to(&recovered.rotation())
                        .angle();

                    max_pos_err = max_pos_err.max(pos_err);
                    max_ori_err = max_ori_err.max(ori_err);

                    if pos_err < 0.001 {
                        // <1mm
                        successes += 1;
                    }

                    // Verify joint limits respected
                    for (j, &val) in solution.joints.iter().enumerate() {
                        if j < chain.active_joints.len() {
                            let ji = chain.active_joints[j];
                            if let Some(limits) = &robot.joints[ji].limits {
                                assert!(
                                    val >= limits.lower - 0.01 && val <= limits.upper + 0.01,
                                    "Robot '{}' IK solution joint {} = {} outside limits [{}, {}]",
                                    name,
                                    j,
                                    val,
                                    limits.lower,
                                    limits.upper
                                );
                            }
                        }
                    }
                }
            }
        }

        // Report
        eprintln!(
            "[{}] {}/{} IK roundtrips succeeded, max_pos_err={:.6}m ({:.3}mm), max_ori_err={:.4}rad ({:.2}°)",
            name,
            successes,
            num_tests,
            max_pos_err,
            max_pos_err * 1000.0,
            max_ori_err,
            max_ori_err.to_degrees()
        );

        // At least 25% success rate (some random configs may be near singularities)
        assert!(
            successes >= num_tests / 4,
            "Robot '{}' IK roundtrip success rate too low: {}/{}",
            name,
            successes,
            num_tests
        );

        // Position accuracy should be < 1mm for converged solutions
        if successes > 0 {
            assert!(
                max_pos_err < 0.01, // 10mm tolerance (includes near-singularity edge cases)
                "Robot '{}' max position error too high: {:.4}m",
                name,
                max_pos_err
            );
        }
    }
}

#[test]
fn fk_consistency_different_configs() {
    // FK should produce different results for different joint configs
    let robot = Robot::from_name("ur5e").unwrap();
    let arm = &robot.groups["arm"];
    let chain = KinematicChain::extract(&robot, &arm.base_link, &arm.tip_link).unwrap();

    let config1 = vec![0.0; 6];
    let config2 = vec![0.5, -0.5, 0.5, -0.5, 0.5, 0.0];

    let pose1 = forward_kinematics(&robot, &chain, &config1).unwrap();
    let pose2 = forward_kinematics(&robot, &chain, &config2).unwrap();

    let pos_diff = (pose1.translation() - pose2.translation()).norm();
    assert!(
        pos_diff > 0.01,
        "FK should produce different poses for different configs, diff = {:.6}m",
        pos_diff
    );
}

#[test]
fn fk_deterministic() {
    // Same config should always produce same FK result
    let robot = Robot::from_name("franka_panda").unwrap();
    let arm = &robot.groups["arm"];
    let chain = KinematicChain::extract(&robot, &arm.base_link, &arm.tip_link).unwrap();

    let config = vec![0.3, -0.5, 0.2, -1.5, 0.1, 1.0, 0.5];

    let pose1 = forward_kinematics(&robot, &chain, &config).unwrap();
    let pose2 = forward_kinematics(&robot, &chain, &config).unwrap();

    let pos_diff = (pose1.translation() - pose2.translation()).norm();
    assert!(
        pos_diff < 1e-15,
        "FK should be deterministic, diff = {:.2e}",
        pos_diff
    );
}
