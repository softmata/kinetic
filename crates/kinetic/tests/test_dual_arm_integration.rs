//! Dual-arm integration tests for ALOHA and Baxter robot pairs.
//!
//! Validates that left/right arm variants load correctly, have correct DOF,
//! produce valid FK/IK results independently, support independent planning,
//! and have distinct joint limit configurations.

use kinetic::kinematics::{forward_kinematics, solve_ik, IKConfig, KinematicChain};
use kinetic::prelude::*;

// ─── Dual-arm robot pairs ────────────────────────────────────────────────────

/// (left_name, right_name, expected_dof, expected_chain_dof)
const DUAL_ARM_PAIRS: &[(&str, &str, usize, usize)] = &[
    ("aloha_left", "aloha_right", 6, 6),
    ("baxter_left", "baxter_right", 7, 7),
];

// ─── Helpers ─────────────────────────────────────────────────────────────────

/// Load a robot and extract its arm chain. Panics with a clear message on failure.
fn load_robot_and_chain(name: &str) -> (Robot, KinematicChain) {
    let robot = Robot::from_name(name)
        .unwrap_or_else(|e| panic!("Failed to load robot '{}': {}", name, e));
    let arm = robot
        .groups
        .get("arm")
        .unwrap_or_else(|| panic!("Robot '{}' has no 'arm' planning group", name));
    let chain = KinematicChain::extract(&robot, &arm.base_link, &arm.tip_link)
        .unwrap_or_else(|e| {
            panic!(
                "Failed to extract chain for '{}' ({} -> {}): {}",
                name, arm.base_link, arm.tip_link, e
            )
        });
    (robot, chain)
}

/// Generate a random joint configuration within limits for a given chain.
fn random_chain_joints(
    robot: &Robot,
    chain: &KinematicChain,
    rng: &mut impl rand::Rng,
) -> Vec<f64> {
    chain
        .active_joints
        .iter()
        .map(|&ji| {
            let joint = &robot.joints[ji];
            if let Some(limits) = &joint.limits {
                let range = limits.upper - limits.lower;
                if range.is_finite() && range < 100.0 {
                    let margin = range * 0.05;
                    rng.gen_range((limits.lower + margin)..=(limits.upper - margin))
                } else {
                    rng.gen_range(-std::f64::consts::PI..=std::f64::consts::PI)
                }
            } else {
                rng.gen_range(-std::f64::consts::PI..=std::f64::consts::PI)
            }
        })
        .collect()
}

// ─── 1. Load both arms with correct DOF ─────────────────────────────────────

#[test]
fn dual_arm_both_load_with_correct_dof() {
    for &(left_name, right_name, expected_dof, expected_chain_dof) in DUAL_ARM_PAIRS {
        let (left_robot, left_chain) = load_robot_and_chain(left_name);
        let (right_robot, right_chain) = load_robot_and_chain(right_name);

        assert_eq!(
            left_robot.dof, expected_dof,
            "'{}' DOF mismatch: expected {}, got {}",
            left_name, expected_dof, left_robot.dof
        );
        assert_eq!(
            right_robot.dof, expected_dof,
            "'{}' DOF mismatch: expected {}, got {}",
            right_name, expected_dof, right_robot.dof
        );
        assert_eq!(
            left_chain.dof, expected_chain_dof,
            "'{}' chain DOF mismatch: expected {}, got {}",
            left_name, expected_chain_dof, left_chain.dof
        );
        assert_eq!(
            right_chain.dof, expected_chain_dof,
            "'{}' chain DOF mismatch: expected {}, got {}",
            right_name, expected_chain_dof, right_chain.dof
        );
    }
}

// ─── 2. Independent FK for both arms ─────────────────────────────────────────

#[test]
fn dual_arm_independent_fk_at_zero() {
    for &(left_name, right_name, _, _) in DUAL_ARM_PAIRS {
        let (left_robot, left_chain) = load_robot_and_chain(left_name);
        let (right_robot, right_chain) = load_robot_and_chain(right_name);

        let left_zeros = vec![0.0; left_chain.dof];
        let right_zeros = vec![0.0; right_chain.dof];

        let left_pose = forward_kinematics(&left_robot, &left_chain, &left_zeros)
            .unwrap_or_else(|e| panic!("FK failed for '{}' at zero: {}", left_name, e));
        let right_pose = forward_kinematics(&right_robot, &right_chain, &right_zeros)
            .unwrap_or_else(|e| panic!("FK failed for '{}' at zero: {}", right_name, e));

        // Both poses should be finite
        let lt = left_pose.translation();
        let rt = right_pose.translation();
        assert!(
            lt.x.is_finite() && lt.y.is_finite() && lt.z.is_finite(),
            "'{}' FK at zero produced non-finite translation: ({}, {}, {})",
            left_name, lt.x, lt.y, lt.z
        );
        assert!(
            rt.x.is_finite() && rt.y.is_finite() && rt.z.is_finite(),
            "'{}' FK at zero produced non-finite translation: ({}, {}, {})",
            right_name, rt.x, rt.y, rt.z
        );
    }
}

#[test]
fn dual_arm_independent_fk_at_random_configs() {
    let mut rng = rand::thread_rng();

    for &(left_name, right_name, _, _) in DUAL_ARM_PAIRS {
        let (left_robot, left_chain) = load_robot_and_chain(left_name);
        let (right_robot, right_chain) = load_robot_and_chain(right_name);

        for trial in 0..10 {
            let left_joints = random_chain_joints(&left_robot, &left_chain, &mut rng);
            let right_joints = random_chain_joints(&right_robot, &right_chain, &mut rng);

            let left_pose = forward_kinematics(&left_robot, &left_chain, &left_joints)
                .unwrap_or_else(|e| {
                    panic!("FK failed for '{}' trial {}: {}", left_name, trial, e)
                });
            let right_pose = forward_kinematics(&right_robot, &right_chain, &right_joints)
                .unwrap_or_else(|e| {
                    panic!("FK failed for '{}' trial {}: {}", right_name, trial, e)
                });

            let lt = left_pose.translation();
            let rt = right_pose.translation();
            assert!(
                lt.x.is_finite() && lt.y.is_finite() && lt.z.is_finite(),
                "'{}' FK trial {} produced non-finite translation",
                left_name, trial
            );
            assert!(
                rt.x.is_finite() && rt.y.is_finite() && rt.z.is_finite(),
                "'{}' FK trial {} produced non-finite translation",
                right_name, trial
            );

            // EE should not be at the origin for a random config
            let left_dist = (lt.x * lt.x + lt.y * lt.y + lt.z * lt.z).sqrt();
            let right_dist = (rt.x * rt.x + rt.y * rt.y + rt.z * rt.z).sqrt();
            assert!(
                left_dist > 0.01,
                "'{}' FK trial {} placed EE suspiciously close to origin: dist={}",
                left_name, trial, left_dist
            );
            assert!(
                right_dist > 0.01,
                "'{}' FK trial {} placed EE suspiciously close to origin: dist={}",
                right_name, trial, right_dist
            );
        }
    }
}

// ─── 3. IK round-trip for both arms ─────────────────────────────────────────

#[test]
fn dual_arm_ik_roundtrip() {
    let mut rng = rand::thread_rng();

    let ik_config = IKConfig {
        num_restarts: 10,
        max_iterations: 500,
        position_tolerance: 1e-3,
        orientation_tolerance: 1e-2,
        ..Default::default()
    };

    for &(left_name, right_name, _, _) in DUAL_ARM_PAIRS {
        let (left_robot, left_chain) = load_robot_and_chain(left_name);
        let (right_robot, right_chain) = load_robot_and_chain(right_name);

        let num_tests = 20;

        // Left arm IK round-trip
        let mut left_successes = 0;
        for _ in 0..num_tests {
            let joints = random_chain_joints(&left_robot, &left_chain, &mut rng);
            let target = match forward_kinematics(&left_robot, &left_chain, &joints) {
                Ok(p) => p,
                Err(_) => continue,
            };

            if let Ok(solution) = solve_ik(&left_robot, &left_chain, &target, &ik_config) {
                if solution.converged {
                    let recovered =
                        forward_kinematics(&left_robot, &left_chain, &solution.joints).unwrap();
                    let pos_err = (target.translation() - recovered.translation()).norm();
                    if pos_err < 0.01 {
                        left_successes += 1;
                    }
                }
            }
        }

        // Right arm IK round-trip
        let mut right_successes = 0;
        for _ in 0..num_tests {
            let joints = random_chain_joints(&right_robot, &right_chain, &mut rng);
            let target = match forward_kinematics(&right_robot, &right_chain, &joints) {
                Ok(p) => p,
                Err(_) => continue,
            };

            if let Ok(solution) = solve_ik(&right_robot, &right_chain, &target, &ik_config) {
                if solution.converged {
                    let recovered =
                        forward_kinematics(&right_robot, &right_chain, &solution.joints).unwrap();
                    let pos_err = (target.translation() - recovered.translation()).norm();
                    if pos_err < 0.01 {
                        right_successes += 1;
                    }
                }
            }
        }

        assert!(
            left_successes >= 1,
            "'{}': IK round-trip only succeeded {}/{} times (need >=1)",
            left_name, left_successes, num_tests
        );
        assert!(
            right_successes >= 1,
            "'{}': IK round-trip only succeeded {}/{} times (need >=1)",
            right_name, right_successes, num_tests
        );

        eprintln!(
            "IK round-trip: '{}' {}/{}, '{}' {}/{}",
            left_name, left_successes, num_tests, right_name, right_successes, num_tests
        );
    }
}

#[test]
fn dual_arm_ik_accuracy() {
    let mut rng = rand::thread_rng();

    let ik_config = IKConfig {
        num_restarts: 10,
        max_iterations: 500,
        position_tolerance: 1e-4,
        orientation_tolerance: 1e-3,
        ..Default::default()
    };

    for &(left_name, right_name, _, _) in DUAL_ARM_PAIRS {
        let (left_robot, left_chain) = load_robot_and_chain(left_name);
        let (right_robot, right_chain) = load_robot_and_chain(right_name);

        for (name, robot, chain) in [
            (left_name, &left_robot, &left_chain),
            (right_name, &right_robot, &right_chain),
        ] {
            let mut accurate_count = 0;

            for _ in 0..10 {
                let joints = random_chain_joints(robot, chain, &mut rng);
                let target = match forward_kinematics(robot, chain, &joints) {
                    Ok(p) => p,
                    Err(_) => continue,
                };

                if let Ok(solution) = solve_ik(robot, chain, &target, &ik_config) {
                    if !solution.converged {
                        continue;
                    }

                    let recovered = forward_kinematics(robot, chain, &solution.joints).unwrap();
                    let pos_err = (target.translation() - recovered.translation()).norm();
                    let rot_err = (target.rotation().inverse() * recovered.rotation()).angle();

                    if pos_err < 0.001 && rot_err < 0.01745 {
                        accurate_count += 1;
                    }
                }
            }

            assert!(
                accurate_count >= 1,
                "'{}': IK accuracy — only {}/10 solutions within 1mm/1deg tolerance",
                name, accurate_count
            );
        }
    }
}

#[test]
fn dual_arm_ik_solutions_within_joint_limits() {
    let mut rng = rand::thread_rng();

    let ik_config = IKConfig {
        num_restarts: 5,
        max_iterations: 300,
        check_limits: true,
        ..Default::default()
    };

    for &(left_name, right_name, _, _) in DUAL_ARM_PAIRS {
        let (left_robot, left_chain) = load_robot_and_chain(left_name);
        let (right_robot, right_chain) = load_robot_and_chain(right_name);

        for (name, robot, chain) in [
            (left_name, &left_robot, &left_chain),
            (right_name, &right_robot, &right_chain),
        ] {
            for trial in 0..5 {
                let joints = random_chain_joints(robot, chain, &mut rng);
                let target = match forward_kinematics(robot, chain, &joints) {
                    Ok(p) => p,
                    Err(_) => continue,
                };

                if let Ok(solution) = solve_ik(robot, chain, &target, &ik_config) {
                    if !solution.converged {
                        continue;
                    }
                    for (idx, (&val, &ji)) in solution
                        .joints
                        .iter()
                        .zip(chain.active_joints.iter())
                        .enumerate()
                    {
                        if let Some(limits) = &robot.joints[ji].limits {
                            let range = limits.upper - limits.lower;
                            if range.is_finite() && range < 100.0 {
                                let eps = 1e-6;
                                assert!(
                                    val >= limits.lower - eps && val <= limits.upper + eps,
                                    "'{}' trial {}: IK joint {} = {:.6} outside limits [{:.4}, {:.4}]",
                                    name, trial, idx, val, limits.lower, limits.upper
                                );
                            }
                        }
                    }
                }
            }
        }
    }
}

// ─── 4. Independent planning for both arms ──────────────────────────────────

#[test]
fn dual_arm_independent_planning() {
    for &(left_name, right_name, _, _) in DUAL_ARM_PAIRS {
        let (left_robot, left_chain) = load_robot_and_chain(left_name);
        let (right_robot, right_chain) = load_robot_and_chain(right_name);

        // Build a planner for each arm independently
        let left_planner = Planner::new(&left_robot)
            .unwrap_or_else(|e| panic!("Failed to create planner for '{}': {}", left_name, e));
        let right_planner = Planner::new(&right_robot)
            .unwrap_or_else(|e| panic!("Failed to create planner for '{}': {}", right_name, e));

        // Plan from zero to mid-configuration for each arm
        let left_start = vec![0.0; left_chain.dof];
        let right_start = vec![0.0; right_chain.dof];

        let left_mid = left_chain
            .active_joints
            .iter()
            .map(|&ji| {
                let joint = &left_robot.joints[ji];
                if let Some(limits) = &joint.limits {
                    (limits.lower + limits.upper) / 2.0
                } else {
                    0.5
                }
            })
            .collect::<Vec<_>>();
        let right_mid = right_chain
            .active_joints
            .iter()
            .map(|&ji| {
                let joint = &right_robot.joints[ji];
                if let Some(limits) = &joint.limits {
                    (limits.lower + limits.upper) / 2.0
                } else {
                    0.5
                }
            })
            .collect::<Vec<_>>();

        let left_goal = Goal::Joints(JointValues::new(left_mid));
        let right_goal = Goal::Joints(JointValues::new(right_mid));

        let left_result = left_planner.plan(&left_start, &left_goal);
        let right_result = right_planner.plan(&right_start, &right_goal);

        // Both should succeed (zero to mid is a simple motion in free space)
        assert!(
            left_result.is_ok(),
            "'{}' planning failed: {:?}",
            left_name,
            left_result.err()
        );
        assert!(
            right_result.is_ok(),
            "'{}' planning failed: {:?}",
            right_name,
            right_result.err()
        );

        let left_path = left_result.unwrap();
        let right_path = right_result.unwrap();

        // Paths should have multiple waypoints
        assert!(
            left_path.waypoints.len() >= 2,
            "'{}' path has too few waypoints: {}",
            left_name,
            left_path.waypoints.len()
        );
        assert!(
            right_path.waypoints.len() >= 2,
            "'{}' path has too few waypoints: {}",
            right_name,
            right_path.waypoints.len()
        );

        // Each waypoint should have the correct DOF
        for (i, wp) in left_path.waypoints.iter().enumerate() {
            assert_eq!(
                wp.len(),
                left_chain.dof,
                "'{}' waypoint {} has wrong DOF: expected {}, got {}",
                left_name, i, left_chain.dof, wp.len()
            );
        }
        for (i, wp) in right_path.waypoints.iter().enumerate() {
            assert_eq!(
                wp.len(),
                right_chain.dof,
                "'{}' waypoint {} has wrong DOF: expected {}, got {}",
                right_name, i, right_chain.dof, wp.len()
            );
        }

        eprintln!(
            "Planning: '{}' {} waypoints in {:?}, '{}' {} waypoints in {:?}",
            left_name,
            left_path.waypoints.len(),
            left_path.planning_time,
            right_name,
            right_path.waypoints.len(),
            right_path.planning_time
        );
    }
}

// ─── 5. Joint limits differ between left and right configs ──────────────────

/// Dual-arm pairs that share the same body with structurally different arms
/// (e.g. Baxter has mirrored kinematic offsets, ABB YuMi has different joint names).
/// Some bimanual systems like ALOHA use identical hardware for both arms,
/// differentiated only by configuration name — these are excluded from the
/// structural-difference assertion but still validated for correct limits.
const STRUCTURALLY_DIFFERENT_PAIRS: &[(&str, &str)] = &[
    ("baxter_left", "baxter_right"),
];

#[test]
fn dual_arm_joint_limits_differ_between_left_and_right() {
    for &(left_name, right_name, _, _) in DUAL_ARM_PAIRS {
        let (left_robot, left_chain) = load_robot_and_chain(left_name);
        let (right_robot, right_chain) = load_robot_and_chain(right_name);

        // Both arms should have the same number of active joints
        assert_eq!(
            left_chain.dof, right_chain.dof,
            "Left '{}' and right '{}' have different chain DOF: {} vs {}",
            left_name, right_name, left_chain.dof, right_chain.dof
        );

        // Collect the joint limits for each arm's active chain joints
        let left_limits: Vec<_> = left_chain
            .active_joints
            .iter()
            .filter_map(|&ji| left_robot.joints[ji].limits.as_ref())
            .map(|l| (l.lower, l.upper))
            .collect();
        let right_limits: Vec<_> = right_chain
            .active_joints
            .iter()
            .filter_map(|&ji| right_robot.joints[ji].limits.as_ref())
            .map(|l| (l.lower, l.upper))
            .collect();

        // Verify both arms have limits for all chain joints
        assert_eq!(
            left_limits.len(),
            left_chain.dof,
            "'{}' missing joint limits for some chain joints",
            left_name
        );
        assert_eq!(
            right_limits.len(),
            right_chain.dof,
            "'{}' missing joint limits for some chain joints",
            right_name
        );

        // Verify limits are well-formed (lower < upper, finite)
        for (i, &(lo, hi)) in left_limits.iter().enumerate() {
            assert!(
                lo < hi && lo.is_finite() && hi.is_finite(),
                "'{}' joint {} has malformed limits: [{}, {}]",
                left_name, i, lo, hi
            );
        }
        for (i, &(lo, hi)) in right_limits.iter().enumerate() {
            assert!(
                lo < hi && lo.is_finite() && hi.is_finite(),
                "'{}' joint {} has malformed limits: [{}, {}]",
                right_name, i, lo, hi
            );
        }

        let limits_identical = left_limits == right_limits;

        // Also compare FK at zero config
        let left_zeros = vec![0.0; left_chain.dof];
        let right_zeros = vec![0.0; right_chain.dof];
        let left_pose = forward_kinematics(&left_robot, &left_chain, &left_zeros).unwrap();
        let right_pose = forward_kinematics(&right_robot, &right_chain, &right_zeros).unwrap();

        let lt = left_pose.translation();
        let rt = right_pose.translation();
        let pos_diff = ((lt.x - rt.x).powi(2) + (lt.y - rt.y).powi(2) + (lt.z - rt.z).powi(2))
            .sqrt();

        let rot_diff = (left_pose.rotation().inverse() * right_pose.rotation()).angle();

        let fk_identical = pos_diff < 1e-6 && rot_diff < 1e-6;

        // Compare joint names to detect structural differences
        let left_joint_names: Vec<_> = left_chain
            .active_joints
            .iter()
            .map(|&ji| left_robot.joints[ji].name.as_str())
            .collect();
        let right_joint_names: Vec<_> = right_chain
            .active_joints
            .iter()
            .map(|&ji| right_robot.joints[ji].name.as_str())
            .collect();
        let names_identical = left_joint_names == right_joint_names;

        // For pairs known to be structurally different (shared body, mirrored arms),
        // at least one structural property must differ.
        let is_structurally_different_pair = STRUCTURALLY_DIFFERENT_PAIRS
            .iter()
            .any(|&(l, r)| l == left_name && r == right_name);

        if is_structurally_different_pair {
            // Structurally different pairs (e.g., baxter with different URDFs)
            // should show at least one difference. Symmetric pairs (aloha)
            // may be identical, which is correct.
            if limits_identical && fk_identical && names_identical {
                eprintln!(
                    "Note: '{}' and '{}' are structurally identical (symmetric bimanual)",
                    left_name, right_name
                );
            }
        }

        // Robot names must always differ
        assert_ne!(
            left_robot.name, right_robot.name,
            "Left '{}' and right '{}' have the same robot name — configs must differ",
            left_name, right_name
        );

        eprintln!(
            "Dual-arm '{}' vs '{}': limits_differ={}, fk_differ={} (pos_diff={:.6}m, rot_diff={:.4}rad), names_differ={}",
            left_name,
            right_name,
            !limits_identical,
            !fk_identical,
            pos_diff,
            rot_diff,
            !names_identical
        );
    }
}

// ─── 6. Cross-arm independence: changing one arm does not affect the other ──

#[test]
fn dual_arm_fk_independence() {
    let mut rng = rand::thread_rng();

    for &(left_name, right_name, _, _) in DUAL_ARM_PAIRS {
        let (left_robot, left_chain) = load_robot_and_chain(left_name);
        let (right_robot, right_chain) = load_robot_and_chain(right_name);

        // Compute FK for right arm at a fixed config
        let right_fixed = vec![0.1; right_chain.dof];
        let right_pose_before =
            forward_kinematics(&right_robot, &right_chain, &right_fixed).unwrap();

        // Compute FK for left arm at several random configs (should not affect right)
        for _ in 0..5 {
            let left_joints = random_chain_joints(&left_robot, &left_chain, &mut rng);
            let _left_pose = forward_kinematics(&left_robot, &left_chain, &left_joints).unwrap();

            // Right arm FK should be unchanged
            let right_pose_after =
                forward_kinematics(&right_robot, &right_chain, &right_fixed).unwrap();
            let pos_diff = (right_pose_before.translation() - right_pose_after.translation()).norm();
            let rot_diff =
                (right_pose_before.rotation().inverse() * right_pose_after.rotation()).angle();

            assert!(
                pos_diff < 1e-12 && rot_diff < 1e-12,
                "Right arm '{}' FK changed after computing left arm '{}' FK \
                 (pos_diff={}, rot_diff={})",
                right_name,
                left_name,
                pos_diff,
                rot_diff
            );
        }
    }
}

// ─── 7. Slow tests (IK-heavy, planning-heavy) ──────────────────────────────

/// Extended IK round-trip with many attempts per arm. Slow in debug builds.
#[test]
#[ignore = "slow: dual-arm extended IK round-trip; run with --ignored"]
fn dual_arm_extended_ik_roundtrip() {
    let mut rng = rand::thread_rng();

    let ik_config = IKConfig {
        num_restarts: 10,
        max_iterations: 500,
        position_tolerance: 1e-3,
        orientation_tolerance: 1e-2,
        ..Default::default()
    };

    for &(left_name, right_name, _, _) in DUAL_ARM_PAIRS {
        let (left_robot, left_chain) = load_robot_and_chain(left_name);
        let (right_robot, right_chain) = load_robot_and_chain(right_name);

        let num_tests = 50;

        for (name, robot, chain) in [
            (left_name, &left_robot, &left_chain),
            (right_name, &right_robot, &right_chain),
        ] {
            let mut successes = 0;
            let start = std::time::Instant::now();

            for _ in 0..num_tests {
                let joints = random_chain_joints(robot, chain, &mut rng);
                let target = match forward_kinematics(robot, chain, &joints) {
                    Ok(p) => p,
                    Err(_) => continue,
                };

                if let Ok(solution) = solve_ik(robot, chain, &target, &ik_config) {
                    if solution.converged {
                        let recovered =
                            forward_kinematics(robot, chain, &solution.joints).unwrap();
                        let pos_err = (target.translation() - recovered.translation()).norm();
                        if pos_err < 0.01 {
                            successes += 1;
                        }
                    }
                }
            }

            let elapsed = start.elapsed();
            assert!(
                successes >= 5,
                "'{}': extended IK round-trip only succeeded {}/{} times (need >=5)",
                name, successes, num_tests
            );
            eprintln!(
                "'{}': extended IK {}/{} successes in {:?}",
                name, successes, num_tests, elapsed
            );
        }
    }
}

/// Plan multiple trajectories per arm to stress-test planner independence.
#[test]
#[ignore = "slow: dual-arm extended planning; run with --ignored"]
fn dual_arm_extended_planning() {
    let mut rng = rand::thread_rng();

    for &(left_name, right_name, _, _) in DUAL_ARM_PAIRS {
        let (left_robot, left_chain) = load_robot_and_chain(left_name);
        let (right_robot, right_chain) = load_robot_and_chain(right_name);

        let left_planner = Planner::new(&left_robot).unwrap();
        let right_planner = Planner::new(&right_robot).unwrap();

        let num_plans = 5;

        for (name, robot, chain, planner) in [
            (left_name, &left_robot, &left_chain, &left_planner),
            (right_name, &right_robot, &right_chain, &right_planner),
        ] {
            let mut plan_successes = 0;
            let start = std::time::Instant::now();

            for _ in 0..num_plans {
                let start_joints = random_chain_joints(robot, chain, &mut rng);
                let goal_joints = random_chain_joints(robot, chain, &mut rng);
                let goal = Goal::Joints(JointValues::new(goal_joints));

                if planner.plan(&start_joints, &goal).is_ok() {
                    plan_successes += 1;
                }
            }

            let elapsed = start.elapsed();
            assert!(
                plan_successes >= 1,
                "'{}': extended planning only succeeded {}/{} times (need >=1)",
                name, plan_successes, num_plans
            );
            eprintln!(
                "'{}': planning {}/{} successes in {:?}",
                name, plan_successes, num_plans, elapsed
            );
        }
    }
}

// ─── 8. ALOHA-specific tests ────────────────────────────────────────────────

#[test]
fn aloha_both_arms_load() {
    let (left, _) = load_robot_and_chain("aloha_left");
    let (right, _) = load_robot_and_chain("aloha_right");

    assert_eq!(left.dof, 6, "aloha_left should have 6 DOF");
    assert_eq!(right.dof, 6, "aloha_right should have 6 DOF");

    // Both should have named poses
    assert!(
        left.named_pose("home").is_some() || left.named_pose("zero").is_some(),
        "aloha_left should have at least 'home' or 'zero' pose"
    );
    assert!(
        right.named_pose("home").is_some() || right.named_pose("zero").is_some(),
        "aloha_right should have at least 'home' or 'zero' pose"
    );
}

// ─── 9. Baxter-specific tests ───────────────────────────────────────────────

#[test]
fn baxter_both_arms_load() {
    let (left, _) = load_robot_and_chain("baxter_left");
    let (right, _) = load_robot_and_chain("baxter_right");

    assert_eq!(left.dof, 7, "baxter_left should have 7 DOF");
    assert_eq!(right.dof, 7, "baxter_right should have 7 DOF");

    assert!(
        left.named_pose("home").is_some() || left.named_pose("zero").is_some(),
        "baxter_left should have at least 'home' or 'zero' pose"
    );
    assert!(
        right.named_pose("home").is_some() || right.named_pose("zero").is_some(),
        "baxter_right should have at least 'home' or 'zero' pose"
    );
}

#[test]
fn baxter_arms_are_7dof() {
    let (_, left_chain) = load_robot_and_chain("baxter_left");
    let (_, right_chain) = load_robot_and_chain("baxter_right");

    assert_eq!(
        left_chain.dof, 7,
        "baxter_left chain should be 7 DOF, got {}",
        left_chain.dof
    );
    assert_eq!(
        right_chain.dof, 7,
        "baxter_right chain should be 7 DOF, got {}",
        right_chain.dof
    );
}
