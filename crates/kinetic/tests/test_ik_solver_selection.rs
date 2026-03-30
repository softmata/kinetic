//! Tests for IK solver selection and coverage across all robot families.
//!
//! Verifies that:
//! 1. The correct IK solver is auto-selected for each robot geometry
//! 2. OPW is used for 6-DOF spherical wrist robots (UR, xArm6, etc.)
//! 3. Subproblem/Subproblem7DOF is used for compatible structures
//! 4. DLS is the fallback for non-standard robots
//! 5. All 52 robot configs can load and have a valid solver path
//! 6. Solver selection is deterministic

use kinetic::prelude::*;

use kinetic::kinematics::{
    is_opw_compatible, is_subproblem_7dof_compatible, is_subproblem_compatible,
};

/// Helper: load robot and auto-detect kinematic chain.
fn load_robot_chain(name: &str) -> Option<(Robot, KinematicChain)> {
    let robot = Robot::from_name(name).ok()?;
    let chain = KinematicChain::auto_detect(&robot).ok()?;
    Some((robot, chain))
}

/// Helper: determine which solver Auto would select for this robot.
fn auto_solver_for(robot: &Robot, chain: &KinematicChain) -> &'static str {
    if is_opw_compatible(robot, chain) {
        "OPW"
    } else if is_subproblem_compatible(robot, chain) {
        "Subproblem"
    } else if is_subproblem_7dof_compatible(robot, chain) {
        "Subproblem7DOF"
    } else {
        "DLS"
    }
}

// ── All 52 robots load and have a valid solver path ─────────────────

const ALL_ROBOTS: &[&str] = &[
    "abb_irb1200",
    "abb_irb4600",
    "abb_yumi_left",
    "abb_yumi_right",
    "aloha_left",
    "aloha_right",
    "baxter_left",
    "baxter_right",
    "denso_vs068",
    "dobot_cr5",
    "elite_ec66",
    "fanuc_crx10ia",
    "fanuc_lr_mate_200id",
    "fetch",
    "flexiv_rizon4",
    "franka_panda",
    "jaco2_6dof",
    "kinova_gen3",
    "kinova_gen3_lite",
    "koch_v1",
    "kuka_iiwa14",
    "kuka_iiwa7",
    "kuka_kr6",
    "lerobot_so100",
    "meca500",
    "mycobot_280",
    "niryo_ned2",
    "open_manipulator_x",
    "pr2",
    "robotis_open_manipulator_p",
    "sawyer",
    "so_arm100",
    "staubli_tx260",
    "stretch_re2",
    "techman_tm5_700",
    "tiago",
    "trossen_px100",
    "trossen_rx150",
    "trossen_wx250s",
    "ur10e",
    "ur16e",
    "ur20",
    "ur30",
    "ur3e",
    "ur5e",
    "viperx_300",
    "widowx_250",
    "xarm5",
    "xarm6",
    "xarm7",
    "yaskawa_gp7",
    "yaskawa_hc10",
];

#[test]
fn all_robots_load_successfully() {
    let mut failed = Vec::new();
    for &name in ALL_ROBOTS {
        if Robot::from_name(name).is_err() {
            failed.push(name);
        }
    }
    assert!(
        failed.is_empty(),
        "These robots failed to load: {:?}",
        failed
    );
}

#[test]
fn all_robots_have_valid_kinematic_chain() {
    let mut failed = Vec::new();
    for &name in ALL_ROBOTS {
        if let Ok(robot) = Robot::from_name(name) {
            if KinematicChain::auto_detect(&robot).is_err() {
                failed.push(name);
            }
        }
    }
    assert!(
        failed.is_empty(),
        "These robots have no valid kinematic chain: {:?}",
        failed
    );
}

#[test]
fn all_robots_have_a_solver_assigned() {
    // Every robot should resolve to one of: OPW, Subproblem, Subproblem7DOF, DLS
    for &name in ALL_ROBOTS {
        if let Some((robot, chain)) = load_robot_chain(name) {
            let solver = auto_solver_for(&robot, &chain);
            assert!(
                ["OPW", "Subproblem", "Subproblem7DOF", "DLS"].contains(&solver),
                "Robot '{}' has unexpected solver: {}",
                name,
                solver
            );
        }
    }
}

// ── OPW solver selection ────────────────────────────────────────────

const UR_FAMILY: &[&str] = &["ur3e", "ur5e", "ur10e", "ur16e", "ur20", "ur30"];

#[test]
fn ur_family_uses_opw() {
    for &name in UR_FAMILY {
        let (robot, chain) = load_robot_chain(name).unwrap();
        assert!(
            is_opw_compatible(&robot, &chain),
            "UR robot '{}' (DOF={}) should be OPW compatible",
            name,
            chain.dof
        );
        assert_eq!(
            auto_solver_for(&robot, &chain),
            "OPW",
            "UR robot '{}' should auto-select OPW",
            name
        );
    }
}

#[test]
fn xarm6_uses_opw() {
    if let Some((robot, chain)) = load_robot_chain("xarm6") {
        assert!(
            is_opw_compatible(&robot, &chain),
            "xArm6 should be OPW compatible"
        );
    }
}

// ── 7-DOF robots ────────────────────────────────────────────────────

const SEVEN_DOF_ROBOTS: &[&str] = &[
    "franka_panda",
    "kuka_iiwa7",
    "kuka_iiwa14",
    "xarm7",
    "kinova_gen3",
    "sawyer",
    "flexiv_rizon4",
];

#[test]
fn seven_dof_robots_have_correct_dof() {
    for &name in SEVEN_DOF_ROBOTS {
        if let Some((_, chain)) = load_robot_chain(name) {
            assert_eq!(
                chain.dof, 7,
                "Robot '{}' should have 7 DOF, got {}",
                name, chain.dof
            );
        }
    }
}

#[test]
fn seven_dof_robots_use_subproblem7dof_or_dls() {
    for &name in SEVEN_DOF_ROBOTS {
        if let Some((robot, chain)) = load_robot_chain(name) {
            let solver = auto_solver_for(&robot, &chain);
            assert!(
                solver == "Subproblem7DOF" || solver == "DLS",
                "7-DOF robot '{}' should use Subproblem7DOF or DLS, got {}",
                name,
                solver
            );
        }
    }
}

#[test]
fn seven_dof_robots_not_opw_compatible() {
    for &name in SEVEN_DOF_ROBOTS {
        if let Some((robot, chain)) = load_robot_chain(name) {
            assert!(
                !is_opw_compatible(&robot, &chain),
                "7-DOF robot '{}' should NOT be OPW compatible",
                name
            );
        }
    }
}

// ── Solver selection is deterministic ───────────────────────────────

#[test]
fn solver_selection_is_deterministic() {
    for &name in ALL_ROBOTS {
        if let Some((robot, chain)) = load_robot_chain(name) {
            let solver1 = auto_solver_for(&robot, &chain);
            let solver2 = auto_solver_for(&robot, &chain);
            let solver3 = auto_solver_for(&robot, &chain);
            assert_eq!(solver1, solver2, "Solver for '{}' not deterministic", name);
            assert_eq!(solver2, solver3, "Solver for '{}' not deterministic", name);
        }
    }
}

// ── IK actually solves for each solver type ─────────────────────────

#[test]
fn opw_solver_produces_valid_ik_for_ur5e() {
    let (robot, chain) = load_robot_chain("ur5e").unwrap();
    let joints_in = vec![0.5, -1.0, 0.5, 0.0, 0.5, 0.0];
    let target = forward_kinematics(&robot, &chain, &joints_in).unwrap();

    let config = IKConfig {
        solver: IKSolver::OPW,
        ..IKConfig::default()
    }
    .with_seed(joints_in.clone());

    let solution = solve_ik(&robot, &chain, &target, &config).unwrap();
    assert!(
        solution.converged,
        "OPW should converge for UR5e: pos_err={}",
        solution.position_error
    );
    assert!(solution.position_error < 1e-3);
}

#[test]
fn dls_solver_produces_valid_ik_for_ur5e() {
    let (robot, chain) = load_robot_chain("ur5e").unwrap();
    let joints_in = vec![0.5, -1.0, 0.5, 0.0, 0.5, 0.0];
    let target = forward_kinematics(&robot, &chain, &joints_in).unwrap();

    let config = IKConfig::dls().with_seed(joints_in.clone());

    let solution = solve_ik(&robot, &chain, &target, &config).unwrap();
    assert!(
        solution.converged,
        "DLS should converge for UR5e: pos_err={}",
        solution.position_error
    );
    assert!(solution.position_error < 1e-3);
}

#[test]
fn auto_solver_ik_roundtrip_ur5e() {
    let (robot, chain) = load_robot_chain("ur5e").unwrap();
    let joints_in = vec![0.5, -1.0, 0.5, 0.0, 0.5, 0.0];
    let target = forward_kinematics(&robot, &chain, &joints_in).unwrap();

    let config = IKConfig::default().with_seed(joints_in.clone());
    let solution = solve_ik(&robot, &chain, &target, &config).unwrap();
    assert!(solution.converged);

    // Verify the returned joints produce the same pose
    let recovered = forward_kinematics(&robot, &chain, &solution.joints).unwrap();
    let pos_err = (recovered.translation() - target.translation()).norm();
    assert!(pos_err < 1e-3, "FK-IK roundtrip error: {pos_err}");
}

#[test]
fn auto_solver_ik_roundtrip_franka_panda() {
    let (robot, chain) = load_robot_chain("franka_panda").unwrap();
    let mid: Vec<f64> = robot
        .joint_limits
        .iter()
        .take(chain.dof)
        .map(|l| (l.lower + l.upper) / 2.0)
        .collect();
    let target = forward_kinematics(&robot, &chain, &mid).unwrap();

    let config = IKConfig::default().with_seed(mid).with_restarts(3);
    let solution = solve_ik(&robot, &chain, &target, &config).unwrap();
    assert!(
        solution.converged,
        "IK should converge for Panda mid config: pos_err={}",
        solution.position_error
    );
}

#[test]
fn auto_solver_ik_roundtrip_kuka_iiwa7() {
    let (robot, chain) = load_robot_chain("kuka_iiwa7").unwrap();
    let mid: Vec<f64> = robot
        .joint_limits
        .iter()
        .take(chain.dof)
        .map(|l| (l.lower + l.upper) / 2.0)
        .collect();
    let target = forward_kinematics(&robot, &chain, &mid).unwrap();

    let config = IKConfig::default().with_seed(mid).with_restarts(3);
    let solution = solve_ik(&robot, &chain, &target, &config).unwrap();
    assert!(
        solution.converged,
        "IK should converge for KUKA iiwa7 mid config: pos_err={}",
        solution.position_error
    );
}

#[test]
fn auto_solver_ik_roundtrip_xarm7() {
    let (robot, chain) = load_robot_chain("xarm7").unwrap();
    let mid: Vec<f64> = robot
        .joint_limits
        .iter()
        .take(chain.dof)
        .map(|l| (l.lower + l.upper) / 2.0)
        .collect();
    let target = forward_kinematics(&robot, &chain, &mid).unwrap();

    let config = IKConfig::default().with_seed(mid).with_restarts(3);
    let solution = solve_ik(&robot, &chain, &target, &config).unwrap();
    assert!(
        solution.converged,
        "IK should converge for xArm7 mid config: pos_err={}",
        solution.position_error
    );
}

// ── Explicit solver override tests ──────────────────────────────────

#[test]
fn explicit_dls_override_works_for_opw_robot() {
    // Even though UR5e is OPW-compatible, forcing DLS should work
    let (robot, chain) = load_robot_chain("ur5e").unwrap();
    let joints_in = vec![0.5, -1.0, 0.5, 0.0, 0.5, 0.0];
    let target = forward_kinematics(&robot, &chain, &joints_in).unwrap();

    let config = IKConfig::dls().with_seed(joints_in);
    let solution = solve_ik(&robot, &chain, &target, &config).unwrap();
    assert!(solution.converged);
}

#[test]
fn explicit_fabrik_override_works() {
    let (robot, chain) = load_robot_chain("ur5e").unwrap();
    let joints_in = vec![0.5, -1.0, 0.5, 0.0, 0.5, 0.0];
    let target = forward_kinematics(&robot, &chain, &joints_in).unwrap();

    let config = IKConfig::fabrik()
        .with_seed(joints_in)
        .with_max_iterations(200);
    let solution = solve_ik(&robot, &chain, &target, &config).unwrap();
    assert!(
        solution.converged,
        "FABRIK should converge for UR5e: pos_err={}",
        solution.position_error
    );
}

#[test]
fn explicit_opw_override_for_non_opw_robot_still_attempts() {
    // Forcing OPW on a 7-DOF robot: should fail gracefully (error, not panic)
    let (robot, chain) = load_robot_chain("franka_panda").unwrap();
    let mid: Vec<f64> = robot
        .joint_limits
        .iter()
        .take(chain.dof)
        .map(|l| (l.lower + l.upper) / 2.0)
        .collect();
    let target = forward_kinematics(&robot, &chain, &mid).unwrap();

    let config = IKConfig {
        solver: IKSolver::OPW,
        ..IKConfig::default()
    }
    .with_seed(mid);

    // Should return error, not panic
    let result = solve_ik(&robot, &chain, &target, &config);
    // Either it fails to converge (OPW needs 6-DOF) or returns an error
    assert!(
        result.is_err() || !result.as_ref().unwrap().converged,
        "OPW on 7-DOF should fail"
    );
}

// ── Analytical vs numerical produce same solution ───────────────────

#[test]
fn opw_and_dls_produce_equivalent_solutions_for_ur5e() {
    let (robot, chain) = load_robot_chain("ur5e").unwrap();
    let joints_in = vec![0.5, -1.0, 0.5, 0.0, 0.5, 0.0];
    let target = forward_kinematics(&robot, &chain, &joints_in).unwrap();

    let opw_config = IKConfig {
        solver: IKSolver::OPW,
        ..IKConfig::default()
    }
    .with_seed(joints_in.clone());

    let dls_config = IKConfig::dls().with_seed(joints_in);

    let opw_sol = solve_ik(&robot, &chain, &target, &opw_config).unwrap();
    let dls_sol = solve_ik(&robot, &chain, &target, &dls_config).unwrap();

    assert!(opw_sol.converged, "OPW should converge");
    assert!(dls_sol.converged, "DLS should converge");

    // Both should reach the target pose (within tolerance)
    let opw_pose = forward_kinematics(&robot, &chain, &opw_sol.joints).unwrap();
    let dls_pose = forward_kinematics(&robot, &chain, &dls_sol.joints).unwrap();

    let opw_pos_err = (opw_pose.translation() - target.translation()).norm();
    let dls_pos_err = (dls_pose.translation() - target.translation()).norm();

    assert!(
        opw_pos_err < 1e-3,
        "OPW position error too large: {opw_pos_err}"
    );
    assert!(
        dls_pos_err < 1e-3,
        "DLS position error too large: {dls_pos_err}"
    );
}

// ── Coverage: IK on representative robots from each family ──────────

const REPRESENTATIVE_ROBOTS: &[(&str, usize)] = &[
    ("ur5e", 6),
    ("ur3e", 6),
    ("ur10e", 6),
    ("franka_panda", 7),
    ("kuka_iiwa7", 7),
    ("kuka_iiwa14", 7),
    ("xarm6", 6),
    ("xarm7", 7),
    ("kinova_gen3", 7),
    ("sawyer", 7),
    ("abb_irb1200", 6),
    ("fanuc_crx10ia", 6),
    ("denso_vs068", 6),
    ("staubli_tx260", 6),
    ("yaskawa_gp7", 6),
];

#[test]
fn representative_robots_ik_roundtrip() {
    let mut failed = Vec::new();

    for &(name, expected_dof) in REPRESENTATIVE_ROBOTS {
        let (robot, chain) = match load_robot_chain(name) {
            Some(r) => r,
            None => {
                failed.push(format!("{name}: failed to load"));
                continue;
            }
        };

        assert_eq!(
            chain.dof, expected_dof,
            "Robot '{}' expected DOF {}, got {}",
            name, expected_dof, chain.dof
        );

        // Use mid-range config as both seed and FK target
        let mid: Vec<f64> = robot
            .joint_limits
            .iter()
            .take(chain.dof)
            .map(|l| (l.lower + l.upper) / 2.0)
            .collect();

        let target = match forward_kinematics(&robot, &chain, &mid) {
            Ok(p) => p,
            Err(e) => {
                failed.push(format!("{name}: FK failed: {e}"));
                continue;
            }
        };

        let config = IKConfig::default()
            .with_seed(mid)
            .with_restarts(10)
            .with_max_iterations(200);

        match solve_ik(&robot, &chain, &target, &config) {
            Ok(sol) if sol.converged => {
                // Verify solution reaches target
                if let Ok(recovered) = forward_kinematics(&robot, &chain, &sol.joints) {
                    let pos_err = (recovered.translation() - target.translation()).norm();
                    if pos_err > 1e-2 {
                        failed.push(format!("{name}: IK position error too large: {pos_err:.6}"));
                    }
                }
            }
            Ok(sol) => {
                failed.push(format!(
                    "{name}: IK did not converge (pos_err={:.6})",
                    sol.position_error
                ));
            }
            Err(e) => {
                failed.push(format!("{name}: IK error: {e}"));
            }
        }
    }

    assert!(
        failed.is_empty(),
        "IK roundtrip failures:\n{}",
        failed.join("\n")
    );
}

// ── Solver selection summary ────────────────────────────────────────

#[test]
fn print_solver_selection_summary() {
    // This test always passes — it just prints the solver map for debugging
    let mut opw_robots = Vec::new();
    let mut sub6_robots = Vec::new();
    let mut sub7_robots = Vec::new();
    let mut dls_robots = Vec::new();
    let mut load_failed = Vec::new();

    for &name in ALL_ROBOTS {
        match load_robot_chain(name) {
            Some((robot, chain)) => {
                let solver = auto_solver_for(&robot, &chain);
                match solver {
                    "OPW" => opw_robots.push(format!("{name} (DOF={})", chain.dof)),
                    "Subproblem" => sub6_robots.push(format!("{name} (DOF={})", chain.dof)),
                    "Subproblem7DOF" => sub7_robots.push(format!("{name} (DOF={})", chain.dof)),
                    "DLS" => dls_robots.push(format!("{name} (DOF={})", chain.dof)),
                    _ => {}
                }
            }
            None => load_failed.push(name.to_string()),
        }
    }

    eprintln!("=== IK Solver Selection Summary ===");
    eprintln!("OPW ({}):", opw_robots.len());
    for r in &opw_robots {
        eprintln!("  {r}");
    }
    eprintln!("Subproblem 6-DOF ({}):", sub6_robots.len());
    for r in &sub6_robots {
        eprintln!("  {r}");
    }
    eprintln!("Subproblem 7-DOF ({}):", sub7_robots.len());
    for r in &sub7_robots {
        eprintln!("  {r}");
    }
    eprintln!("DLS fallback ({}):", dls_robots.len());
    for r in &dls_robots {
        eprintln!("  {r}");
    }
    if !load_failed.is_empty() {
        eprintln!("Failed to load ({}):", load_failed.len());
        for r in &load_failed {
            eprintln!("  {r}");
        }
    }
}
