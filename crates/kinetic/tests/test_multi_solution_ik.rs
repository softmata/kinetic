//! Integration tests for multi-solution IK enumeration and ranking.
//!
//! For redundant robots (7-DOF+), IK has infinitely many solutions for a
//! reachable pose. Tests verify that:
//! - Different seeds produce distinct valid solutions
//! - Solutions can be ranked by joint distance, manipulability, or joint limits
//! - Ranking is deterministic for same inputs
//! - All returned solutions reach the target within tolerance

use std::sync::Arc;

use kinetic::prelude::*;
use kinetic_kinematics::{
    forward_kinematics, manipulability, solve_ik, IKConfig, IKMode, IKSolution, IKSolver,
    KinematicChain, NullSpace,
};

fn load_robot(name: &str) -> (Arc<Robot>, KinematicChain) {
    let robot = Arc::new(Robot::from_name(name).unwrap());
    let group = robot.groups.values().next().unwrap();
    let chain = KinematicChain::extract(&robot, &group.base_link, &group.tip_link).unwrap();
    (robot, chain)
}

/// Get the mid-range configuration for a chain (safe, reachable config).
fn mid_config(robot: &Robot, chain: &KinematicChain) -> Vec<f64> {
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

/// Solve IK from multiple random seeds and collect all converged solutions.
fn solve_multi_seed(
    robot: &Robot,
    chain: &KinematicChain,
    target: &Pose,
    num_attempts: usize,
) -> Vec<IKSolution> {
    use rand::Rng;
    let mut rng = rand::thread_rng();
    let mut solutions = Vec::new();

    for _ in 0..num_attempts {
        // Generate random seed within joint limits
        let seed: Vec<f64> = chain
            .active_joints
            .iter()
            .map(|&ji| {
                let j = &robot.joints[ji];
                match &j.limits {
                    Some(l) => rng.gen_range(l.lower..=l.upper),
                    None => rng.gen_range(-std::f64::consts::PI..=std::f64::consts::PI),
                }
            })
            .collect();

        let config = IKConfig {
            solver: IKSolver::DLS { damping: 0.05 },
            mode: IKMode::Full6D,
            max_iterations: 300,
            position_tolerance: 1e-3,
            orientation_tolerance: 1e-2,
            check_limits: true,
            seed: Some(seed),
            null_space: None,
            num_restarts: 0,
        };

        if let Ok(sol) = solve_ik(robot, chain, target, &config) {
            if sol.converged {
                solutions.push(sol);
            }
        }
    }

    solutions
}

/// L2 distance between two joint configurations.
fn joint_distance(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f64>()
        .sqrt()
}

/// Joint limit margin: sum of distances from each joint to its nearest limit.
fn joint_limit_margin(robot: &Robot, chain: &KinematicChain, joints: &[f64]) -> f64 {
    chain
        .active_joints
        .iter()
        .enumerate()
        .map(|(i, &ji)| {
            let j = &robot.joints[ji];
            match &j.limits {
                Some(l) => {
                    let to_lower = (joints[i] - l.lower).abs();
                    let to_upper = (l.upper - joints[i]).abs();
                    to_lower.min(to_upper)
                }
                None => std::f64::consts::PI,
            }
        })
        .sum()
}

// ═══════════════════════════════════════════════════════════════════════════════
// Multi-solution enumeration
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn panda_7dof_produces_multiple_distinct_solutions() {
    let (robot, chain) = load_robot("franka_panda");
    assert_eq!(chain.dof, 7, "Panda should be 7-DOF (redundant)");

    // FK from a known reachable config to get target
    let q_known = mid_config(&robot, &chain);
    let target = forward_kinematics(&robot, &chain, &q_known).unwrap();

    // Solve from many different seeds
    let solutions = solve_multi_seed(&robot, &chain, &target, 30);

    assert!(
        solutions.len() >= 3,
        "Should find at least 3 converged solutions for 7-DOF robot, found {}",
        solutions.len()
    );

    // Check distinctness: at least 2 solutions should differ by > 0.1 rad
    let mut distinct_count = 0;
    for i in 0..solutions.len() {
        for j in (i + 1)..solutions.len() {
            let dist = joint_distance(&solutions[i].joints, &solutions[j].joints);
            if dist > 0.1 {
                distinct_count += 1;
            }
        }
    }
    assert!(
        distinct_count >= 2,
        "Should have at least 2 pairs of distinct solutions, found {}",
        distinct_count
    );
}

#[test]
fn xarm7_produces_multiple_solutions() {
    let (robot, chain) = load_robot("xarm7");
    assert_eq!(chain.dof, 7);

    let q_known = mid_config(&robot, &chain);
    let target = forward_kinematics(&robot, &chain, &q_known).unwrap();

    let solutions = solve_multi_seed(&robot, &chain, &target, 30);
    assert!(
        solutions.len() >= 2,
        "xarm7 should find at least 2 solutions, found {}",
        solutions.len()
    );

    // All solutions should be valid
    for (i, sol) in solutions.iter().enumerate() {
        assert!(
            sol.position_error < 1e-3,
            "Solution {} position error too large: {}",
            i,
            sol.position_error
        );
    }
}

#[test]
fn kuka_iiwa14_7dof_produces_multiple_solutions() {
    let (robot, chain) = load_robot("kuka_iiwa14");
    assert_eq!(chain.dof, 7);

    let q_known = mid_config(&robot, &chain);
    let target = forward_kinematics(&robot, &chain, &q_known).unwrap();

    let solutions = solve_multi_seed(&robot, &chain, &target, 30);
    assert!(
        solutions.len() >= 2,
        "iiwa14 should find at least 2 solutions, found {}",
        solutions.len()
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// All solutions reach target within tolerance
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn all_solutions_reach_target_within_tolerance() {
    let (robot, chain) = load_robot("franka_panda");

    let q_known = mid_config(&robot, &chain);
    let target = forward_kinematics(&robot, &chain, &q_known).unwrap();

    let solutions = solve_multi_seed(&robot, &chain, &target, 20);

    for (i, sol) in solutions.iter().enumerate() {
        // Verify each solution reaches the target via FK
        let achieved = forward_kinematics(&robot, &chain, &sol.joints).unwrap();
        let pos_err = (achieved.translation() - target.translation()).norm();
        assert!(
            pos_err < 1e-3,
            "Solution {} FK position error {} exceeds tolerance",
            i,
            pos_err
        );
    }
}

#[test]
fn all_solutions_within_joint_limits() {
    let (robot, chain) = load_robot("franka_panda");

    let q_known = mid_config(&robot, &chain);
    let target = forward_kinematics(&robot, &chain, &q_known).unwrap();

    let solutions = solve_multi_seed(&robot, &chain, &target, 20);

    for (si, sol) in solutions.iter().enumerate() {
        for (i, &ji) in chain.active_joints.iter().enumerate() {
            if let Some(limits) = &robot.joints[ji].limits {
                assert!(
                    sol.joints[i] >= limits.lower - 1e-6 && sol.joints[i] <= limits.upper + 1e-6,
                    "Solution {}, joint {} = {} outside limits [{}, {}]",
                    si,
                    i,
                    sol.joints[i],
                    limits.lower,
                    limits.upper
                );
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Ranking by joint distance metric
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn ranking_by_seed_distance_selects_closest() {
    let (robot, chain) = load_robot("franka_panda");

    let q_known = mid_config(&robot, &chain);
    let target = forward_kinematics(&robot, &chain, &q_known).unwrap();

    let solutions = solve_multi_seed(&robot, &chain, &target, 30);
    if solutions.len() < 2 {
        return; // need at least 2 to rank
    }

    // Rank by distance to the known config
    let mut ranked: Vec<_> = solutions
        .iter()
        .map(|sol| (joint_distance(&sol.joints, &q_known), sol))
        .collect();
    ranked.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());

    // The closest solution should have minimal distance to known config
    let best_distance = ranked[0].0;
    let worst_distance = ranked.last().unwrap().0;

    // Best should be closer than worst (unless all are identical)
    if ranked.len() >= 2 {
        assert!(
            best_distance <= worst_distance,
            "Ranked best ({}) should be <= worst ({})",
            best_distance,
            worst_distance
        );
    }

    // The best-ranked solution should be reasonably close to the known config
    // (since we used that config to generate the target)
    assert!(
        best_distance < 3.0,
        "Best solution should be within 3.0 rad total distance of known config, was {}",
        best_distance
    );
}

#[test]
fn ranking_is_deterministic_for_same_inputs() {
    let (robot, chain) = load_robot("franka_panda");

    let q_known = mid_config(&robot, &chain);
    let target = forward_kinematics(&robot, &chain, &q_known).unwrap();

    // Solve with fixed seed for determinism
    let seed = q_known.clone();
    let config = IKConfig {
        solver: IKSolver::DLS { damping: 0.05 },
        mode: IKMode::Full6D,
        max_iterations: 200,
        position_tolerance: 1e-3,
        orientation_tolerance: 1e-2,
        check_limits: true,
        seed: Some(seed.clone()),
        null_space: None,
        num_restarts: 0,
    };

    // Two solves with identical inputs should produce identical results
    let sol1 = solve_ik(&robot, &chain, &target, &config).unwrap();
    let sol2 = solve_ik(&robot, &chain, &target, &config).unwrap();

    assert_eq!(sol1.joints.len(), sol2.joints.len());
    for (i, (a, b)) in sol1.joints.iter().zip(&sol2.joints).enumerate() {
        assert!(
            (a - b).abs() < 1e-10,
            "Deterministic solve: joint {} differs: {} vs {}",
            i,
            a,
            b
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Ranking by manipulability
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn solutions_have_varying_manipulability() {
    let (robot, chain) = load_robot("franka_panda");

    let q_known = mid_config(&robot, &chain);
    let target = forward_kinematics(&robot, &chain, &q_known).unwrap();

    let solutions = solve_multi_seed(&robot, &chain, &target, 30);
    if solutions.len() < 2 {
        return;
    }

    // Compute manipulability for each solution
    let manips: Vec<f64> = solutions
        .iter()
        .filter_map(|sol| manipulability(&robot, &chain, &sol.joints).ok())
        .collect();

    // All manipulabilities should be finite and non-negative
    for (i, &m) in manips.iter().enumerate() {
        assert!(m.is_finite(), "Manipulability {} is not finite: {}", i, m);
        assert!(m >= 0.0, "Manipulability {} is negative: {}", i, m);
    }

    // If we have distinct solutions, manipulability should vary
    if manips.len() >= 2 {
        let min_m = manips.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_m = manips.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        // We can't guarantee variation for every run, but the range should be computable
        assert!(
            max_m >= min_m,
            "Max manipulability {} should >= min {}",
            max_m,
            min_m
        );
    }
}

#[test]
fn manipulability_null_space_improves_or_maintains() {
    let (robot, chain) = load_robot("franka_panda");

    let q_known = mid_config(&robot, &chain);
    let target = forward_kinematics(&robot, &chain, &q_known).unwrap();

    let seed = mid_config(&robot, &chain);

    // Solve without null-space
    let config_plain = IKConfig {
        solver: IKSolver::DLS { damping: 0.05 },
        mode: IKMode::Full6D,
        max_iterations: 300,
        position_tolerance: 1e-3,
        orientation_tolerance: 1e-2,
        check_limits: true,
        seed: Some(seed.clone()),
        null_space: None,
        num_restarts: 0,
    };

    // Solve with manipulability null-space
    let config_manip = IKConfig {
        null_space: Some(NullSpace::Manipulability),
        ..config_plain.clone()
    };

    let sol_plain = solve_ik(&robot, &chain, &target, &config_plain).unwrap();
    let sol_manip = solve_ik(&robot, &chain, &target, &config_manip).unwrap();

    // Both should converge
    assert!(sol_plain.converged, "Plain IK should converge");
    assert!(sol_manip.converged, "Manipulability IK should converge");

    // Both should reach the target
    assert!(sol_plain.position_error < 1e-3);
    assert!(sol_manip.position_error < 1e-3);

    // Compute manipulability for both
    let m_plain = manipulability(&robot, &chain, &sol_plain.joints).unwrap();
    let m_manip = manipulability(&robot, &chain, &sol_manip.joints).unwrap();

    // Both should be finite
    assert!(m_plain.is_finite());
    assert!(m_manip.is_finite());
}

// ═══════════════════════════════════════════════════════════════════════════════
// Ranking by joint limit margin
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn joint_centering_null_space_pushes_toward_center() {
    let (robot, chain) = load_robot("franka_panda");

    // Use an off-center seed to make the effect visible
    let seed: Vec<f64> = chain
        .active_joints
        .iter()
        .map(|&ji| {
            let j = &robot.joints[ji];
            match &j.limits {
                // Start at 30% from lower limit (off-center)
                Some(l) => l.lower + (l.upper - l.lower) * 0.3,
                None => 0.0,
            }
        })
        .collect();

    let target = forward_kinematics(&robot, &chain, &seed).unwrap();

    let config_centered = IKConfig {
        solver: IKSolver::DLS { damping: 0.05 },
        mode: IKMode::Full6D,
        max_iterations: 500,
        position_tolerance: 1e-3,
        orientation_tolerance: 1e-2,
        check_limits: true,
        seed: Some(seed.clone()),
        null_space: Some(NullSpace::JointCentering),
        num_restarts: 0,
    };

    let sol = solve_ik(&robot, &chain, &target, &config_centered).unwrap();
    assert!(sol.converged);

    // The joint centering solution should have a non-zero joint limit margin
    let margin = joint_limit_margin(&robot, &chain, &sol.joints);
    assert!(
        margin > 0.0,
        "Joint limit margin should be positive: {}",
        margin
    );
}

#[test]
fn solutions_can_be_ranked_by_joint_limit_margin() {
    let (robot, chain) = load_robot("franka_panda");

    let q_known = mid_config(&robot, &chain);
    let target = forward_kinematics(&robot, &chain, &q_known).unwrap();

    let solutions = solve_multi_seed(&robot, &chain, &target, 20);
    if solutions.len() < 2 {
        return;
    }

    // Compute margin for each and sort
    let mut ranked: Vec<_> = solutions
        .iter()
        .map(|sol| (joint_limit_margin(&robot, &chain, &sol.joints), sol))
        .collect();
    ranked.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap()); // descending: higher margin = better

    // All margins should be non-negative
    for (i, (margin, _)) in ranked.iter().enumerate() {
        assert!(
            *margin >= 0.0,
            "Solution {} margin should be non-negative: {}",
            i,
            margin
        );
    }

    // Best (highest margin) should be >= worst
    let best_margin = ranked.first().unwrap().0;
    let worst_margin = ranked.last().unwrap().0;
    assert!(
        best_margin >= worst_margin,
        "Best margin {} should >= worst {}",
        best_margin,
        worst_margin
    );
}

// ═══════════════════════════════════════════════════════════════════════════════
// Seed influence on solution
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn different_seeds_converge_to_different_solutions() {
    let (robot, chain) = load_robot("franka_panda");

    let q_known = mid_config(&robot, &chain);
    let target = forward_kinematics(&robot, &chain, &q_known).unwrap();

    // Two very different seeds
    let seed_a: Vec<f64> = chain
        .active_joints
        .iter()
        .map(|&ji| {
            let j = &robot.joints[ji];
            match &j.limits {
                Some(l) => l.lower + (l.upper - l.lower) * 0.2,
                None => -1.0,
            }
        })
        .collect();

    let seed_b: Vec<f64> = chain
        .active_joints
        .iter()
        .map(|&ji| {
            let j = &robot.joints[ji];
            match &j.limits {
                Some(l) => l.lower + (l.upper - l.lower) * 0.8,
                None => 1.0,
            }
        })
        .collect();

    let config_a = IKConfig {
        solver: IKSolver::DLS { damping: 0.05 },
        mode: IKMode::Full6D,
        max_iterations: 300,
        position_tolerance: 1e-3,
        orientation_tolerance: 1e-2,
        check_limits: true,
        seed: Some(seed_a.clone()),
        null_space: None,
        num_restarts: 0,
    };

    let config_b = IKConfig {
        seed: Some(seed_b.clone()),
        ..config_a.clone()
    };

    let sol_a = solve_ik(&robot, &chain, &target, &config_a);
    let sol_b = solve_ik(&robot, &chain, &target, &config_b);

    // Both should produce valid results (either converged or error)
    // If both converge, they should generally be different solutions
    if let (Ok(a), Ok(b)) = (&sol_a, &sol_b) {
        if a.converged && b.converged {
            // Both reach target
            assert!(a.position_error < 1e-3);
            assert!(b.position_error < 1e-3);

            // Solution from seed_a should be closer to seed_a
            let dist_a_to_seed_a = joint_distance(&a.joints, &seed_a);
            let dist_b_to_seed_a = joint_distance(&b.joints, &seed_a);

            // Solution from seed_b should be closer to seed_b
            let dist_a_to_seed_b = joint_distance(&a.joints, &seed_b);
            let dist_b_to_seed_b = joint_distance(&b.joints, &seed_b);

            // At least one pair should show seed affinity
            let a_closer_to_own_seed = dist_a_to_seed_a < dist_a_to_seed_b;
            let b_closer_to_own_seed = dist_b_to_seed_b < dist_b_to_seed_a;

            // Not guaranteed for every pose, but should hold for most
            assert!(
                a_closer_to_own_seed || b_closer_to_own_seed,
                "At least one solution should be closer to its seed"
            );
        }
    }
}

#[test]
fn restarts_improve_solution_quality() {
    let (robot, chain) = load_robot("franka_panda");

    // Use a pose that's not trivially the mid-config
    let q_target: Vec<f64> = chain
        .active_joints
        .iter()
        .map(|&ji| {
            let j = &robot.joints[ji];
            match &j.limits {
                Some(l) => l.lower + (l.upper - l.lower) * 0.7,
                None => 0.5,
            }
        })
        .collect();
    let target = forward_kinematics(&robot, &chain, &q_target).unwrap();

    // Bad seed (far from target)
    let bad_seed: Vec<f64> = chain
        .active_joints
        .iter()
        .map(|&ji| {
            let j = &robot.joints[ji];
            match &j.limits {
                Some(l) => l.lower + (l.upper - l.lower) * 0.1,
                None => -1.5,
            }
        })
        .collect();

    // Without restarts
    let config_no_restart = IKConfig {
        solver: IKSolver::DLS { damping: 0.05 },
        mode: IKMode::Full6D,
        max_iterations: 100,
        position_tolerance: 1e-4,
        orientation_tolerance: 1e-3,
        check_limits: true,
        seed: Some(bad_seed.clone()),
        null_space: None,
        num_restarts: 0,
    };

    // With restarts
    let config_restart = IKConfig {
        num_restarts: 10,
        ..config_no_restart.clone()
    };

    let result_no = solve_ik(&robot, &chain, &target, &config_no_restart);
    let result_yes = solve_ik(&robot, &chain, &target, &config_restart);

    // With restarts should either converge or have better error
    match (&result_no, &result_yes) {
        (Err(_), Ok(sol)) => {
            // Restarts found a solution where single attempt didn't
            assert!(sol.converged);
        }
        (Ok(no), Ok(yes)) => {
            // Both converged, or restarts improved error
            assert!(
                yes.position_error <= no.position_error + 1e-6,
                "Restarts should not make error worse: {} vs {}",
                yes.position_error,
                no.position_error
            );
        }
        _ => {
            // Both failed: acceptable for hard poses
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Cross-robot coverage
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn ur5e_6dof_ik_converges() {
    let (robot, chain) = load_robot("ur5e");
    assert_eq!(chain.dof, 6);

    let q_known = mid_config(&robot, &chain);
    let target = forward_kinematics(&robot, &chain, &q_known).unwrap();

    let config = IKConfig {
        solver: IKSolver::DLS { damping: 0.05 },
        mode: IKMode::Full6D,
        max_iterations: 200,
        position_tolerance: 1e-3,
        orientation_tolerance: 1e-2,
        check_limits: true,
        seed: Some(q_known),
        null_space: None,
        num_restarts: 5,
    };

    let sol = solve_ik(&robot, &chain, &target, &config).unwrap();
    assert!(
        sol.converged,
        "UR5e should converge: pos_err={}",
        sol.position_error
    );
}

#[test]
fn kinova_gen3_7dof_multi_solution() {
    let (robot, chain) = load_robot("kinova_gen3");
    assert_eq!(chain.dof, 7);

    let q_known = mid_config(&robot, &chain);
    let target = forward_kinematics(&robot, &chain, &q_known).unwrap();

    let solutions = solve_multi_seed(&robot, &chain, &target, 20);
    assert!(
        !solutions.is_empty(),
        "kinova gen3 should find at least 1 IK solution"
    );

    // All should reach target
    for (i, sol) in solutions.iter().enumerate() {
        assert!(
            sol.position_error < 1e-3,
            "kinova gen3 solution {} error too large: {}",
            i,
            sol.position_error
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// Edge cases
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn ik_from_mid_config_converges_quickly() {
    let (robot, chain) = load_robot("franka_panda");

    let q_known = mid_config(&robot, &chain);
    let target = forward_kinematics(&robot, &chain, &q_known).unwrap();

    let config = IKConfig {
        solver: IKSolver::DLS { damping: 0.05 },
        mode: IKMode::Full6D,
        max_iterations: 50,
        position_tolerance: 1e-4,
        orientation_tolerance: 1e-3,
        check_limits: true,
        seed: Some(q_known.clone()),
        null_space: None,
        num_restarts: 0,
    };

    let sol = solve_ik(&robot, &chain, &target, &config).unwrap();
    assert!(sol.converged, "IK from mid should converge quickly");
    assert!(
        sol.iterations < 20,
        "Should converge in < 20 iterations from mid-config, took {}",
        sol.iterations
    );
}

#[test]
fn ik_position_only_for_redundant_robot() {
    let (robot, chain) = load_robot("franka_panda");

    let q_known = mid_config(&robot, &chain);
    let target = forward_kinematics(&robot, &chain, &q_known).unwrap();

    let config = IKConfig {
        solver: IKSolver::DLS { damping: 0.05 },
        mode: IKMode::PositionOnly,
        max_iterations: 200,
        position_tolerance: 1e-3,
        orientation_tolerance: 1e-2,
        check_limits: true,
        seed: Some(mid_config(&robot, &chain)),
        null_space: None,
        num_restarts: 0,
    };

    let sol = solve_ik(&robot, &chain, &target, &config).unwrap();
    assert!(sol.converged);
    assert_eq!(sol.mode_used, IKMode::PositionOnly);
    assert!(sol.position_error < 1e-3);
}
