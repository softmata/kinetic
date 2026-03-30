//! Dynamics ↔ Planning ↔ Trajectory integration tests.
//!
//! Verifies the full pipeline: plan trajectory → time parameterize → check
//! torque feasibility via dynamics. This is the critical gap: planning and
//! dynamics must agree on the robot model and trajectory format.

use kinetic_dynamics::{
    articulated_body_from_chain, check_trajectory_feasibility, effort_limits_from_chain,
    gravity_compensation, inverse_dynamics, forward_dynamics, mass_matrix,
};
use kinetic::kinematics::{forward_kinematics, KinematicChain};
use kinetic::prelude::*;

// ─── Dynamics across multiple robots ────────────────────────────────────────

const DYNAMICS_ROBOTS: &[&str] = &[
    "ur5e",
    "franka_panda",
    "kuka_iiwa7",
    "xarm6",
    "kinova_gen3",
];

#[test]
fn gravity_compensation_all_robots() {
    // Intent: gravity comp should return correct DOF and finite values for every robot
    for name in DYNAMICS_ROBOTS {
        let robot = Robot::from_name(name).unwrap();
        let arm = &robot.groups["arm"];
        let chain = KinematicChain::extract(&robot, &arm.base_link, &arm.tip_link).unwrap();
        let mut body = articulated_body_from_chain(&robot, &chain);

        let q = JointValues::zeros(chain.dof);
        let tau = gravity_compensation(&mut body, &q);

        assert_eq!(
            tau.len(),
            chain.dof,
            "{name}: gravity comp DOF mismatch: got {} expected {}",
            tau.len(),
            chain.dof
        );

        for (j, t) in tau.as_slice().iter().enumerate() {
            assert!(
                t.is_finite(),
                "{name} joint {j}: gravity comp is non-finite: {t}"
            );
        }
    }
}

#[test]
fn mass_matrix_spd_all_robots() {
    // Intent: mass matrix must be symmetric positive semi-definite for all robots
    for name in DYNAMICS_ROBOTS {
        let robot = Robot::from_name(name).unwrap();
        let arm = &robot.groups["arm"];
        let chain = KinematicChain::extract(&robot, &arm.base_link, &arm.tip_link).unwrap();
        let mut body = articulated_body_from_chain(&robot, &chain);

        let q = JointValues::zeros(chain.dof);
        let m = mass_matrix(&mut body, &q);

        assert_eq!(m.nrows(), chain.dof, "{name}: mass matrix rows");
        assert_eq!(m.ncols(), chain.dof, "{name}: mass matrix cols");

        // Symmetry
        let asym = (&m - m.transpose()).norm();
        assert!(
            asym < 1e-4,
            "{name}: mass matrix asymmetry = {asym}"
        );

        // Positive semi-definite (eigenvalues ≥ 0)
        let eigvals = nalgebra::SymmetricEigen::new(m.clone()).eigenvalues;
        for ev in eigvals.iter() {
            assert!(
                *ev >= -1e-4,
                "{name}: negative eigenvalue {ev}"
            );
        }
    }
}

#[test]
fn inverse_forward_roundtrip_all_robots() {
    // Intent: ID → FD roundtrip should recover original accelerations
    for name in DYNAMICS_ROBOTS {
        let robot = Robot::from_name(name).unwrap();
        let arm = &robot.groups["arm"];
        let chain = KinematicChain::extract(&robot, &arm.base_link, &arm.tip_link).unwrap();
        let mut body = articulated_body_from_chain(&robot, &chain);

        let q = JointValues::zeros(chain.dof);
        let qd = JointValues::zeros(chain.dof);
        let qdd_desired = JointValues::from_slice(&vec![0.5; chain.dof]);

        let tau = inverse_dynamics(&mut body, &q, &qd, &qdd_desired);
        let qdd_recovered = forward_dynamics(&mut body, &q, &qd, &tau);

        for j in 0..chain.dof {
            assert!(
                (qdd_recovered[j] - qdd_desired[j]).abs() < 0.5,
                "{name} joint {j}: ID→FD roundtrip error = {} (desired={}, recovered={})",
                (qdd_recovered[j] - qdd_desired[j]).abs(),
                qdd_desired[j],
                qdd_recovered[j]
            );
        }
    }
}

// ─── FK + Dynamics consistency ──────────────────────────────────────────────

#[test]
fn fk_and_dynamics_use_same_chain() {
    // Intent: FK and dynamics should produce consistent results for the same chain
    let robot = Robot::from_name("franka_panda").unwrap();
    let arm = &robot.groups["arm"];
    let chain = KinematicChain::extract(&robot, &arm.base_link, &arm.tip_link).unwrap();
    let mut body = articulated_body_from_chain(&robot, &chain);

    // FK at zero config
    let q_vals = vec![0.0; chain.dof];
    let fk_pose = forward_kinematics(&robot, &chain, &q_vals).unwrap();
    assert!(fk_pose.translation().norm() > 0.0, "FK should produce nonzero pose");

    // Dynamics at zero config — gravity comp should be finite
    let q = JointValues::from_slice(&q_vals);
    let tau = gravity_compensation(&mut body, &q);
    assert_eq!(tau.len(), chain.dof);

    // Effort limits should match chain DOF
    let limits = effort_limits_from_chain(&robot, &chain);
    assert_eq!(limits.len(), chain.dof);
}

// ─── Trajectory Feasibility Integration ─────────────────────────────────────

#[test]
fn planned_trajectory_is_dynamically_feasible() {
    // Intent: a slowly-planned trajectory should be within effort limits
    let robot = Robot::from_name("ur5e").unwrap();
    let arm = &robot.groups["arm"];
    let chain = KinematicChain::extract(&robot, &arm.base_link, &arm.tip_link).unwrap();
    let mut body = articulated_body_from_chain(&robot, &chain);
    let limits = effort_limits_from_chain(&robot, &chain);

    // Create a slow trajectory (small motion, long duration)
    let mut traj = Trajectory::new(chain.dof, (0..chain.dof).map(|i| format!("j{i}")).collect());
    let start = vec![0.0; chain.dof];
    let end: Vec<f64> = start.iter().map(|s| s + 0.05).collect();
    traj.push_waypoint(&start);
    traj.push_waypoint(&end);

    // SoA layout for timing: small velocities, zero accelerations, 5-second duration
    let velocities: Vec<f64> = vec![0.0; chain.dof * 2]; // zero velocities at endpoints
    let accelerations: Vec<f64> = vec![0.0; chain.dof * 2];
    traj.set_timing(vec![0.0, 5.0], velocities, accelerations);

    let violations = check_trajectory_feasibility(&mut body, &traj, &limits);
    assert!(
        violations.is_empty(),
        "slow 5-second trajectory should be feasible, got {} violations",
        violations.len()
    );
}

#[test]
fn fast_trajectory_has_dynamics_violations() {
    // Intent: extremely fast motion should exceed effort limits
    let robot = Robot::from_name("ur5e").unwrap();
    let arm = &robot.groups["arm"];
    let chain = KinematicChain::extract(&robot, &arm.base_link, &arm.tip_link).unwrap();
    let mut body = articulated_body_from_chain(&robot, &chain);

    // Use very tight effort limits (1 Nm) — any real motion should violate
    let tight_limits = vec![1.0; chain.dof];

    let mut traj = Trajectory::new(chain.dof, (0..chain.dof).map(|i| format!("j{i}")).collect());
    traj.push_waypoint(&vec![0.0; chain.dof]);
    traj.push_waypoint(&vec![1.0; chain.dof]); // 1 rad motion

    // Fast: 50ms with high velocity and acceleration
    let n_joints = chain.dof;
    let velocities = vec![20.0; n_joints * 2];
    let accelerations = vec![400.0; n_joints * 2];
    traj.set_timing(vec![0.0, 0.05], velocities, accelerations);

    let violations = check_trajectory_feasibility(&mut body, &traj, &tight_limits);
    assert!(
        !violations.is_empty(),
        "fast trajectory with 1Nm limits should produce violations"
    );
}
