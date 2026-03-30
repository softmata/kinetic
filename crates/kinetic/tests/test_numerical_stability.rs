//! Numerical stability and extreme value tests.
//!
//! Verifies no NaN/Inf in outputs at extreme joint angles, near-singular
//! configurations, degenerate geometries, and extreme translations.

use std::f64::consts::PI;

use kinetic::collision::{simd, SpheresSoA};
use kinetic::kinematics::{forward_kinematics, jacobian, KinematicChain};
use kinetic::prelude::*;
use kinetic::trajectory::trapezoidal;

fn ur5e() -> Robot {
    Robot::from_name("ur5e").unwrap()
}

fn ur5e_chain(robot: &Robot) -> KinematicChain {
    let arm = &robot.groups["arm"];
    KinematicChain::extract(robot, &arm.base_link, &arm.tip_link).unwrap()
}

fn assert_no_nan_inf(vals: &[f64], context: &str) {
    for (i, &v) in vals.iter().enumerate() {
        assert!(!v.is_nan(), "{context}: NaN at index {i}");
        assert!(!v.is_infinite(), "{context}: Inf at index {i}");
    }
}

// ─── FK with extreme joint angles ────────────────────────────────────────────

#[test]
fn fk_extreme_joint_angles() {
    let robot = ur5e();
    let chain = ur5e_chain(&robot);

    let test_cases: Vec<(&str, Vec<f64>)> = vec![
        ("near-zero", vec![1e-15; 6]),
        ("near-pi", vec![PI - 1e-15; 6]),
        ("large multiples of pi", vec![1000.0 * PI; 6]),
        ("negative large", vec![-999.0 * PI; 6]),
        (
            "mixed extremes",
            vec![1e-15, PI - 1e-15, 1000.0 * PI, -1e-15, PI, 0.0],
        ),
        ("all zeros", vec![0.0; 6]),
        ("all pi", vec![PI; 6]),
        ("all negative pi", vec![-PI; 6]),
    ];

    for (name, joints) in &test_cases {
        let pose = forward_kinematics(&robot, &chain, joints).unwrap();
        let t = pose.translation();
        assert_no_nan_inf(&[t[0], t[1], t[2]], &format!("FK translation ({name})"));

        let rot = pose.rotation();
        let (r, p, y) = rot.euler_angles();
        assert_no_nan_inf(&[r, p, y], &format!("FK rotation ({name})"));

        eprintln!("FK {name}: t=({:.4}, {:.4}, {:.4})", t[0], t[1], t[2]);
    }
}

// ─── Jacobian near singularity ───────────────────────────────────────────────

#[test]
fn jacobian_near_singularity_no_nan() {
    let robot = ur5e();
    let chain = ur5e_chain(&robot);

    // Known singular configurations for UR robots
    let singular_configs = vec![
        ("all-zero (shoulder singularity)", vec![0.0; 6]),
        ("elbow straight", vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
        (
            "wrist singularity",
            vec![0.0, -PI / 2.0, 0.0, 0.0, 0.0, 0.0],
        ),
        (
            "near-singular small",
            vec![1e-10, 1e-10, 1e-10, 1e-10, 1e-10, 1e-10],
        ),
    ];

    for (name, joints) in &singular_configs {
        let jac = jacobian(&robot, &chain, joints).unwrap();

        // No NaN or Inf in Jacobian
        for r in 0..jac.nrows() {
            for c in 0..jac.ncols() {
                let v = jac[(r, c)];
                assert!(!v.is_nan(), "Jacobian NaN at ({r},{c}) for {name}");
                assert!(!v.is_infinite(), "Jacobian Inf at ({r},{c}) for {name}");
            }
        }

        // Check determinant-related quantity
        let jjt = &jac * jac.transpose();
        let det = jjt.determinant();
        eprintln!("Jacobian {name}: det(JJ^T) = {det:.6e}");
        assert!(!det.is_nan(), "JJ^T determinant is NaN for {name}");
    }
}

// ─── FK with tiny link lengths (custom URDF) ────────────────────────────────

#[test]
fn fk_tiny_link_lengths() {
    let tiny_urdf = r#"<?xml version="1.0"?>
<robot name="tiny_arm">
  <link name="base_link"/>
  <link name="link1"/>
  <link name="link2"/>
  <link name="ee_link"/>
  <joint name="j1" type="revolute">
    <parent link="base_link"/><child link="link1"/>
    <axis xyz="0 0 1"/>
    <origin xyz="0 0 1e-10"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="10.0"/>
  </joint>
  <joint name="j2" type="revolute">
    <parent link="link1"/><child link="link2"/>
    <axis xyz="0 1 0"/>
    <origin xyz="0 0 1e-10"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="10.0"/>
  </joint>
  <joint name="j3" type="revolute">
    <parent link="link2"/><child link="ee_link"/>
    <axis xyz="0 1 0"/>
    <origin xyz="0 0 1e-10"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="10.0"/>
  </joint>
</robot>"#;

    let robot = Robot::from_urdf_string(tiny_urdf).unwrap();
    let chain = KinematicChain::extract(&robot, "base_link", "ee_link").unwrap();
    let joints = vec![1.0, 0.5, -0.3];

    let pose = forward_kinematics(&robot, &chain, &joints).unwrap();
    let t = pose.translation();
    assert_no_nan_inf(&[t[0], t[1], t[2]], "FK tiny links");

    // Translation should be very small (links are 1e-10 long)
    let dist = (t[0] * t[0] + t[1] * t[1] + t[2] * t[2]).sqrt();
    assert!(
        dist < 1e-5,
        "Tiny link FK distance should be very small: {dist}"
    );

    // Jacobian should also be valid
    let jac = jacobian(&robot, &chain, &joints).unwrap();
    for r in 0..jac.nrows() {
        for c in 0..jac.ncols() {
            assert!(!jac[(r, c)].is_nan(), "Tiny-link Jacobian NaN at ({r},{c})");
        }
    }
}

// ─── Collision: touching geometries ──────────────────────────────────────────

#[test]
fn collision_touching_geometries() {
    let mut a = SpheresSoA::new();
    let mut b = SpheresSoA::new();

    // Two spheres exactly touching (gap = 0)
    a.push(0.0, 0.0, 0.0, 1.0, 0);
    b.push(2.0, 0.0, 0.0, 1.0, 0);

    let dist = a.signed_distance(0, &b, 0);
    eprintln!("Touching spheres signed distance: {dist}");
    assert!(!dist.is_nan(), "Distance should not be NaN");
    assert!(
        dist.abs() < 1e-10,
        "Touching spheres should have ~0 distance: {dist}"
    );

    // Check overlap detection — touching may or may not count as overlap
    // depending on strict vs non-strict comparison
    let overlaps = a.any_overlap(&b);
    eprintln!("Touching spheres overlap: {overlaps}");
    // Just verify no crash
}

// ─── Collision: penetrating geometries ───────────────────────────────────────

#[test]
fn collision_penetrating_geometries() {
    let mut a = SpheresSoA::new();
    let mut b = SpheresSoA::new();

    // Deep penetration: spheres centered at same point
    a.push(0.0, 0.0, 0.0, 1.0, 0);
    b.push(0.0, 0.0, 0.0, 1.0, 0);

    let dist = a.signed_distance(0, &b, 0);
    eprintln!("Coincident spheres signed distance: {dist}");
    assert!(!dist.is_nan(), "Coincident distance should not be NaN");
    assert!(
        dist < 0.0,
        "Coincident spheres should have negative distance: {dist}"
    );

    assert!(a.any_overlap(&b), "Coincident spheres should overlap");

    // Partial penetration
    let mut c = SpheresSoA::new();
    c.push(0.5, 0.0, 0.0, 1.0, 0);
    let dist2 = a.signed_distance(0, &c, 0);
    assert!(
        dist2 < 0.0,
        "Partially penetrating should have negative distance: {dist2}"
    );
    assert!(a.any_overlap(&c));
}

// ─── Trajectory: near-zero velocity segments ─────────────────────────────────

#[test]
fn trajectory_near_zero_velocity() {
    // Waypoints where consecutive points are nearly identical
    let waypoints: Vec<Vec<f64>> = vec![
        vec![0.0; 6],
        vec![1e-12; 6],                      // nearly zero movement
        vec![2e-12; 6],                      // nearly zero movement
        vec![0.5, 0.3, -0.2, 0.1, 0.0, 0.0], // actual movement
    ];

    let result = trapezoidal(&waypoints, 2.0, 5.0);
    match result {
        Ok(traj) => {
            assert!(
                traj.validate().is_ok(),
                "Near-zero velocity trajectory should validate"
            );
            for wp in &traj.waypoints {
                assert_no_nan_inf(&wp.positions, "trajectory positions");
                assert_no_nan_inf(&wp.velocities, "trajectory velocities");
            }
            // Time should be monotonic
            for w in traj.waypoints.windows(2) {
                assert!(w[1].time >= w[0].time);
            }
        }
        Err(e) => {
            eprintln!("Near-zero trajectory error (acceptable): {e}");
        }
    }
}

// ─── SIMD vs scalar at extreme values ────────────────────────────────────────

#[test]
fn simd_scalar_parity_extreme_values() {
    let test_cases: Vec<(&str, f64, f64, f64, f64, f64, f64, f64, f64)> = vec![
        (
            "very large coords",
            1e6,
            1e6,
            1e6,
            1.0,
            1e6 + 0.5,
            1e6,
            1e6,
            1.0,
        ),
        (
            "very small coords",
            1e-12,
            1e-12,
            1e-12,
            1e-12,
            2e-12,
            1e-12,
            1e-12,
            1e-12,
        ),
        ("very large radii", 0.0, 0.0, 0.0, 1e6, 1.0, 0.0, 0.0, 1e6),
        (
            "very small radii",
            0.0,
            0.0,
            0.0,
            1e-15,
            1e-10,
            0.0,
            0.0,
            1e-15,
        ),
        (
            "mixed extreme",
            1e10,
            -1e10,
            0.0,
            0.001,
            -1e10,
            1e10,
            0.0,
            0.001,
        ),
    ];

    for (name, ax, ay, az, ar, bx, by, bz, br) in &test_cases {
        let mut a = SpheresSoA::new();
        a.push(*ax, *ay, *az, *ar, 0);
        let mut b = SpheresSoA::new();
        b.push(*bx, *by, *bz, *br, 0);

        let simd_coll = simd::any_collision(&a, &b);
        let scalar_coll = simd::scalar::any_collision_scalar(&a, &b);
        assert_eq!(
            simd_coll, scalar_coll,
            "SIMD/scalar collision mismatch for {name}"
        );

        let simd_dist = simd::min_distance(&a, &b);
        let scalar_dist = simd::scalar::min_distance_scalar(&a, &b);

        if simd_dist.is_finite() && scalar_dist.is_finite() {
            let rel_err = if scalar_dist.abs() > 1e-20 {
                (simd_dist - scalar_dist).abs() / scalar_dist.abs()
            } else {
                (simd_dist - scalar_dist).abs()
            };
            assert!(rel_err < 1e-8,
                "SIMD/scalar distance mismatch for {name}: simd={simd_dist}, scalar={scalar_dist}, rel_err={rel_err}");
        }
    }
}

// ─── Pose composition with large translations ────────────────────────────────

#[test]
fn pose_large_translations() {
    let large = Pose::from_xyz(1e6, -1e6, 1e6);
    let small = Pose::from_xyz(0.001, 0.002, 0.003);

    let composed = large.compose(&small);
    let t = composed.translation();
    assert_no_nan_inf(&[t[0], t[1], t[2]], "Large pose composition");

    // Should be approximately large + small translation
    assert!((t[0] - 1e6 - 0.001).abs() < 1e-3);
    assert!((t[1] - (-1e6) - 0.002).abs() < 1e-3);

    // Inverse should cancel
    let inv = large.inverse();
    let identity_ish = large.compose(&inv);
    let t2 = identity_ish.translation();
    assert!(t2[0].abs() < 1e-6, "Inverse should cancel: x={}", t2[0]);
    assert!(t2[1].abs() < 1e-6, "Inverse should cancel: y={}", t2[1]);
    assert!(t2[2].abs() < 1e-6, "Inverse should cancel: z={}", t2[2]);
}

// ─── Quaternion edge cases via Pose ──────────────────────────────────────────

#[test]
fn pose_quaternion_edge_cases() {
    // Identity quaternion
    let p1 = Pose::from_xyz_quat(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0);
    let r1 = p1.rotation();
    assert!(
        !r1.i.is_nan() && !r1.j.is_nan() && !r1.k.is_nan() && !r1.w.is_nan(),
        "Identity quaternion should not have NaN"
    );

    // 180-degree rotation around Z (near-antipodal from identity)
    let p2 = Pose::from_xyz_quat(0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0);
    let r2 = p2.rotation();
    assert!(
        !r2.i.is_nan() && !r2.j.is_nan() && !r2.k.is_nan() && !r2.w.is_nan(),
        "180-deg quaternion should not have NaN"
    );

    // Very small rotation
    let eps = 1e-15;
    let p3 = Pose::from_xyz_quat(0.0, 0.0, 0.0, eps, 0.0, 0.0, 1.0);
    let r3 = p3.rotation();
    assert!(
        !r3.i.is_nan() && !r3.j.is_nan() && !r3.k.is_nan() && !r3.w.is_nan(),
        "Tiny quaternion should not have NaN"
    );

    // Compose identity with identity
    let id = Pose::identity();
    let composed = id.compose(&id);
    let t = composed.translation();
    assert!(t[0].abs() < 1e-15 && t[1].abs() < 1e-15 && t[2].abs() < 1e-15);

    // Rotation distance between near-antipodal
    let dist = p1.rotation_distance(&p2);
    assert!(!dist.is_nan(), "Rotation distance should not be NaN");
    assert!(
        dist > 3.0,
        "180-deg rotation distance should be ~pi: {dist}"
    );
}

// ─── Matrix near-singular via Jacobian pseudoinverse ─────────────────────────

#[test]
fn near_singular_jacobian_pseudoinverse() {
    let robot = ur5e();
    let chain = ur5e_chain(&robot);

    // All-zero config is singular for UR5e
    let joints = vec![0.0; 6];
    let jac = jacobian(&robot, &chain, &joints).unwrap();

    let jjt = &jac * jac.transpose();
    let det = jjt.determinant();
    eprintln!("Near-singular: det(JJ^T) = {det:.6e}");

    // Even near-singular, the damped pseudoinverse should work
    let lambda = 0.01;
    let damped = &jjt + nalgebra::DMatrix::<f64>::identity(6, 6) * (lambda * lambda);
    let damped_det = damped.determinant();
    assert!(
        !damped_det.is_nan(),
        "Damped JJ^T determinant should not be NaN"
    );
    assert!(damped_det > 0.0, "Damped JJ^T should be positive definite");

    // LU decomposition should succeed
    let lu = damped.clone().lu();
    let rhs = nalgebra::DVector::from_element(6, 1.0);
    let solution = lu.solve(&rhs);
    assert!(solution.is_some(), "Damped system should be solvable");
    let sol = solution.unwrap();
    for i in 0..6 {
        assert!(!sol[i].is_nan(), "Solution has NaN at index {i}");
        assert!(!sol[i].is_infinite(), "Solution has Inf at index {i}");
    }
}
