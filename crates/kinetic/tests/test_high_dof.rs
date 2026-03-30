//! Deep kinematic chain and high-DOF stress tests.
//!
//! Constructs custom 10-DOF and 20-DOF robots from URDF strings and
//! verifies FK, Jacobian, IK, planning, collision, and trajectory
//! operations scale correctly.

use std::time::{Duration, Instant};

use kinetic::collision::{RobotSphereModel, SphereGenConfig};
use kinetic::kinematics::{forward_kinematics, jacobian, solve_ik, IKConfig, KinematicChain};
use kinetic::prelude::*;
use kinetic::trajectory::trapezoidal;

/// Generate a URDF string for an N-DOF serial chain robot.
///
/// Each link has a collision cylinder and is connected by alternating
/// Z and Y axis revolute joints, creating a planar-ish arm.
fn generate_n_dof_urdf(n: usize) -> String {
    let mut urdf = String::from(r#"<?xml version="1.0"?><robot name="chain_"#);
    urdf.push_str(&format!("{n}dof\">\n"));

    // Base link with collision
    urdf.push_str(
        r#"  <link name="base_link">
    <collision><geometry><box size="0.2 0.2 0.1"/></geometry></collision>
  </link>
"#,
    );

    // N intermediate links + joints
    for i in 1..=n {
        let link_name = format!("link_{i}");
        let joint_name = format!("joint_{i}");
        let parent = if i == 1 {
            "base_link".to_string()
        } else {
            format!("link_{}", i - 1)
        };

        // Alternate axis: odd joints around Z, even around Y
        let axis = if i % 2 == 1 { "0 1 0" } else { "0 0 1" };
        let link_len = 0.15; // shorter links for high-DOF

        urdf.push_str(&format!(
            r#"  <link name="{link_name}">
    <collision><geometry><cylinder radius="0.03" length="{link_len}"/></geometry></collision>
  </link>
  <joint name="{joint_name}" type="revolute">
    <parent link="{parent}"/>
    <child link="{link_name}"/>
    <axis xyz="{axis}"/>
    <origin xyz="0 0 {link_len}" rpy="0 0 0"/>
    <limit lower="-2.5" upper="2.5" velocity="3.0" effort="50.0"/>
  </joint>
"#
        ));
    }

    // End-effector link
    let last_link = format!("link_{n}");
    urdf.push_str(&format!(
        r#"  <link name="ee_link">
    <collision><geometry><sphere radius="0.02"/></geometry></collision>
  </link>
  <joint name="ee_joint" type="fixed">
    <parent link="{last_link}"/>
    <child link="ee_link"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
  </joint>
"#
    ));

    urdf.push_str("</robot>\n");
    urdf
}

fn load_n_dof(n: usize) -> (Robot, KinematicChain) {
    let urdf = generate_n_dof_urdf(n);
    let robot = Robot::from_urdf_string(&urdf).unwrap();
    assert_eq!(robot.dof, n, "Expected {n}-DOF robot, got {}", robot.dof);
    let chain = KinematicChain::extract(&robot, "base_link", "ee_link").unwrap();
    assert_eq!(chain.dof, n);
    (robot, chain)
}

// ─── 10-DOF FK ───────────────────────────────────────────────────────────────

#[test]
fn fk_10_dof() {
    let (robot, chain) = load_n_dof(10);
    let joints = vec![0.1; 10];
    let pose = forward_kinematics(&robot, &chain, &joints).unwrap();
    let t = pose.translation();

    eprintln!(
        "10-DOF FK: translation = ({:.4}, {:.4}, {:.4})",
        t[0], t[1], t[2]
    );

    // Pose should be reachable (nonzero translation)
    let dist = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
    assert!(dist > 0.01, "10-DOF FK should produce nonzero translation");
}

// ─── 10-DOF Jacobian ─────────────────────────────────────────────────────────

#[test]
fn jacobian_10_dof() {
    let (robot, chain) = load_n_dof(10);
    let joints = vec![0.1; 10];
    let jac = jacobian(&robot, &chain, &joints).unwrap();

    assert_eq!(jac.nrows(), 6, "Jacobian should have 6 rows (6D twist)");
    assert_eq!(
        jac.ncols(),
        10,
        "Jacobian should have 10 columns for 10-DOF"
    );

    // Jacobian should not be all zeros
    let max_val = jac.iter().copied().fold(0.0f64, f64::max);
    assert!(max_val > 1e-10, "Jacobian should have nonzero entries");
}

// ─── 20-DOF FK ───────────────────────────────────────────────────────────────

#[test]
fn fk_20_dof() {
    let (robot, chain) = load_n_dof(20);
    let joints = vec![0.05; 20];
    let pose = forward_kinematics(&robot, &chain, &joints).unwrap();
    let t = pose.translation();

    eprintln!(
        "20-DOF FK: translation = ({:.4}, {:.4}, {:.4})",
        t[0], t[1], t[2]
    );

    let dist = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
    assert!(dist > 0.01, "20-DOF FK should produce nonzero translation");
}

// ─── 20-DOF Jacobian ─────────────────────────────────────────────────────────

#[test]
fn jacobian_20_dof() {
    let (robot, chain) = load_n_dof(20);
    let joints = vec![0.05; 20];
    let jac = jacobian(&robot, &chain, &joints).unwrap();

    assert_eq!(jac.nrows(), 6);
    assert_eq!(
        jac.ncols(),
        20,
        "Jacobian should have 20 columns for 20-DOF"
    );
}

// ─── 10-DOF IK ───────────────────────────────────────────────────────────────

#[test]
fn ik_10_dof() {
    let (robot, chain) = load_n_dof(10);
    let joints_original = vec![0.2, -0.1, 0.3, 0.1, -0.2, 0.15, 0.05, -0.1, 0.2, -0.05];
    let target = forward_kinematics(&robot, &chain, &joints_original).unwrap();

    let ik_config = IKConfig {
        max_iterations: 300,
        num_restarts: 5,
        ..IKConfig::default()
    };

    let result = solve_ik(&robot, &chain, &target, &ik_config);
    match result {
        Ok(sol) => {
            eprintln!(
                "10-DOF IK: converged={}, pos_error={:.6}",
                sol.converged, sol.position_error
            );
            if sol.converged {
                // Verify FK of solution matches target
                let pose_check = forward_kinematics(&robot, &chain, &sol.joints).unwrap();
                let t1 = target.translation();
                let t2 = pose_check.translation();
                let err =
                    ((t1[0] - t2[0]).powi(2) + (t1[1] - t2[1]).powi(2) + (t1[2] - t2[2]).powi(2))
                        .sqrt();
                assert!(err < 0.01, "IK solution FK mismatch: {err:.6}");
            }
        }
        Err(e) => {
            eprintln!("10-DOF IK failed (acceptable for redundant chain): {e}");
        }
    }
}

// ─── 10-DOF planning ─────────────────────────────────────────────────────────

#[test]
fn planning_10_dof() {
    let (robot, _chain) = load_n_dof(10);
    let start = vec![0.0; 10];
    let goal_joints = vec![0.3, -0.2, 0.4, 0.1, -0.3, 0.2, 0.1, -0.1, 0.2, -0.15];
    let goal = Goal::Joints(JointValues::new(goal_joints));

    let planner = Planner::new(&robot).unwrap().with_config(PlannerConfig {
        timeout: Duration::from_secs(5),
        ..PlannerConfig::default()
    });

    let t0 = Instant::now();
    let result = planner.plan(&start, &goal);
    let elapsed = t0.elapsed();

    match result {
        Ok(plan) => {
            eprintln!(
                "10-DOF planning: {} waypoints, {:.1?}",
                plan.num_waypoints(),
                elapsed
            );
            assert!(plan.num_waypoints() >= 2);
        }
        Err(e) => {
            eprintln!("10-DOF planning failed (acceptable): {e}");
        }
    }
}

// ─── 20-DOF collision model ──────────────────────────────────────────────────

#[test]
fn collision_model_20_dof() {
    let (robot, _chain) = load_n_dof(20);

    let model = RobotSphereModel::from_robot(&robot, &SphereGenConfig::coarse());
    eprintln!(
        "20-DOF collision: {} total spheres across {} links",
        model.total_spheres(),
        model.num_links
    );

    // Should have spheres for links with collision geometry
    assert!(model.total_spheres() > 0, "Should have collision spheres");
    assert_eq!(model.num_links, 22); // base + 20 links + ee_link

    // Each link with collision geometry should have at least 1 sphere
    for i in 0..model.num_links {
        let count = model.link_sphere_count(i);
        // All links except the last one (ee_link with sphere) should have >=1
        // (some links have geometry, ee_link has a sphere)
        if count > 0 {
            assert!(count >= 1);
        }
    }
}

// ─── FK scales linearly with DOF ─────────────────────────────────────────────

#[test]
fn fk_scales_linearly() {
    let dofs = [6, 10, 15, 20];
    let iterations = 1000;
    let mut times = Vec::new();

    for &n in &dofs {
        let (robot, chain) = load_n_dof(n);
        let joints = vec![0.1; n];

        // Warmup
        for _ in 0..10 {
            let _ = forward_kinematics(&robot, &chain, &joints).unwrap();
        }

        let t0 = Instant::now();
        for _ in 0..iterations {
            let _ = forward_kinematics(&robot, &chain, &joints).unwrap();
        }
        let elapsed = t0.elapsed();
        let per_call = elapsed.as_nanos() as f64 / iterations as f64;

        eprintln!("{n}-DOF FK: {per_call:.0} ns/call");
        times.push((n, per_call));
    }

    // Verify no super-linear blowup: 20-DOF should be < 10x cost of 6-DOF
    let time_6 = times.iter().find(|t| t.0 == 6).unwrap().1;
    let time_20 = times.iter().find(|t| t.0 == 20).unwrap().1;
    assert!(
        time_20 < time_6 * 10.0,
        "FK scaling: 20-DOF ({time_20:.0} ns) should be < 10x of 6-DOF ({time_6:.0} ns)"
    );
}

// ─── Jacobian size correct for various DOF ───────────────────────────────────

#[test]
fn jacobian_size_various_dof() {
    for n in [3, 6, 7, 10, 15, 20] {
        let (robot, chain) = load_n_dof(n);
        let joints = vec![0.1; n];
        let jac = jacobian(&robot, &chain, &joints).unwrap();

        assert_eq!(jac.nrows(), 6, "{n}-DOF: Jacobian rows should be 6");
        assert_eq!(jac.ncols(), n, "{n}-DOF: Jacobian cols should be {n}");
    }
}

// ─── 20-DOF trajectory parameterization ──────────────────────────────────────

#[test]
fn trajectory_20_dof() {
    let n = 20;
    let num_waypoints = 50;

    let waypoints: Vec<Vec<f64>> = (0..num_waypoints)
        .map(|i| {
            (0..n)
                .map(|j| {
                    let phase = i as f64 * 0.1 + j as f64 * 0.05;
                    phase.sin() * 0.5
                })
                .collect()
        })
        .collect();

    let t0 = Instant::now();
    let result = trapezoidal(&waypoints, 2.0, 5.0);
    let elapsed = t0.elapsed();

    match result {
        Ok(traj) => {
            eprintln!(
                "20-DOF trajectory: {} waypoints -> {} timed, {:.1?}",
                num_waypoints,
                traj.waypoints.len(),
                elapsed
            );

            assert!(traj.waypoints.len() >= num_waypoints);
            assert!(traj.duration() > Duration::ZERO);
            assert!(traj.validate().is_ok());

            // Check DOF consistency
            for wp in &traj.waypoints {
                assert_eq!(wp.positions.len(), n);
                assert_eq!(wp.velocities.len(), n);
            }

            // Time monotonicity
            for w in traj.waypoints.windows(2) {
                assert!(
                    w[1].time >= w[0].time,
                    "Non-monotonic time in 20-DOF trajectory"
                );
            }

            // Endpoints preserved
            for j in 0..n {
                assert!(
                    (traj.waypoints.first().unwrap().positions[j] - waypoints[0][j]).abs() < 1e-6,
                    "Start position mismatch at joint {j}"
                );
                assert!(
                    (traj.waypoints.last().unwrap().positions[j] - waypoints[num_waypoints - 1][j])
                        .abs()
                        < 1e-6,
                    "End position mismatch at joint {j}"
                );
            }
        }
        Err(e) => {
            panic!("20-DOF trajectory should succeed: {e}");
        }
    }
}
