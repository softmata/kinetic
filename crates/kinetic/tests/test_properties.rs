//! Property-based tests using proptest.
//!
//! Verifies invariants with randomized inputs: FK/IK roundtrip, trajectory
//! monotonicity, collision symmetry, scene idempotency, joint clamping, and
//! SIMD-scalar equivalence.

use std::sync::Arc;

use proptest::prelude::*;

use kinetic::collision::{simd, SpheresSoA};
use kinetic::core::Twist;
use kinetic::kinematics::{forward_kinematics, solve_ik, IKConfig, KinematicChain};
use kinetic::prelude::*;
use kinetic::reactive::{Servo, ServoConfig};
use kinetic::scene::Scene;
use kinetic::trajectory::trapezoidal;

fn ur5e() -> Robot {
    Robot::from_name("ur5e").unwrap()
}

fn ur5e_chain(robot: &Robot) -> KinematicChain {
    let arm = &robot.groups["arm"];
    KinematicChain::extract(robot, &arm.base_link, &arm.tip_link).unwrap()
}

/// Generate a random valid joint configuration within UR5e limits.
fn arb_ur5e_joints() -> impl Strategy<Value = Vec<f64>> {
    let robot = ur5e();
    let limits: Vec<(f64, f64)> = robot
        .joint_limits
        .iter()
        .map(|l| (l.lower, l.upper))
        .collect();
    limits
        .into_iter()
        .map(|(lo, hi)| lo..=hi)
        .collect::<Vec<_>>()
        .prop_map(|v| v)
}

// ─── FK→IK→FK roundtrip ─────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn fk_ik_fk_roundtrip(joints in arb_ur5e_joints()) {
        let robot = ur5e();
        let chain = ur5e_chain(&robot);

        let pose = forward_kinematics(&robot, &chain, &joints).unwrap();

        let ik_config = IKConfig {
            max_iterations: 200,
            num_restarts: 3,
            ..IKConfig::default()
        };
        if let Ok(ik_result) = solve_ik(&robot, &chain, &pose, &ik_config) {
            if ik_result.converged {
                let pose2 = forward_kinematics(&robot, &chain, &ik_result.joints).unwrap();
                let t1 = pose.translation();
                let t2 = pose2.translation();
                let pos_err = ((t1[0]-t2[0]).powi(2) + (t1[1]-t2[1]).powi(2) + (t1[2]-t2[2]).powi(2)).sqrt();
                prop_assert!(pos_err < 0.01,
                    "FK->IK->FK position error too large: {:.6}", pos_err);
            }
        }
    }
}

// ─── IK solutions within joint limits ────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn ik_solution_within_limits(joints in arb_ur5e_joints()) {
        let robot = ur5e();
        let chain = ur5e_chain(&robot);
        let pose = forward_kinematics(&robot, &chain, &joints).unwrap();

        let ik_config = IKConfig {
            max_iterations: 100,
            num_restarts: 1,
            ..IKConfig::default()
        };

        if let Ok(ik_result) = solve_ik(&robot, &chain, &pose, &ik_config) {
            if ik_result.converged {
                for (i, &val) in ik_result.joints.iter().enumerate() {
                    let limit = &robot.joint_limits[i];
                    prop_assert!(val >= limit.lower - 1e-6 && val <= limit.upper + 1e-6,
                        "Joint {} = {} outside limits [{}, {}]",
                        i, val, limit.lower, limit.upper);
                }
            }
        }
    }
}

// ─── Trajectory preserves endpoints ──────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn trajectory_preserves_endpoints(
        n in 3usize..20,
        seed in 0u64..1000,
    ) {
        use rand::SeedableRng;
        use rand::Rng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let dof = 6;
        let waypoints: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..dof).map(|_| rng.gen_range(-2.0..2.0)).collect())
            .collect();

        if let Ok(traj) = trapezoidal(&waypoints, 2.0, 5.0) {
            let first = &traj.waypoints.first().unwrap().positions;
            let last = &traj.waypoints.last().unwrap().positions;

            for j in 0..dof {
                prop_assert!((first[j] - waypoints[0][j]).abs() < 1e-6,
                    "First waypoint joint {} mismatch: {} vs {}",
                    j, first[j], waypoints[0][j]);
                prop_assert!((last[j] - waypoints[n-1][j]).abs() < 1e-6,
                    "Last waypoint joint {} mismatch: {} vs {}",
                    j, last[j], waypoints[n-1][j]);
            }
        }
    }
}

// ─── Trajectory time monotonicity ────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn trajectory_time_monotonic(
        n in 3usize..20,
        seed in 0u64..1000,
    ) {
        use rand::SeedableRng;
        use rand::Rng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let dof = 6;
        let waypoints: Vec<Vec<f64>> = (0..n)
            .map(|_| (0..dof).map(|_| rng.gen_range(-2.0..2.0)).collect())
            .collect();

        if let Ok(traj) = trapezoidal(&waypoints, 2.0, 5.0) {
            for w in traj.waypoints.windows(2) {
                prop_assert!(w[1].time >= w[0].time,
                    "Non-monotonic timestamps: {} > {}",
                    w[0].time, w[1].time);
            }
        }
    }
}

// ─── Collision symmetry ──────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn collision_symmetry(
        ax in -5.0f64..5.0, ay in -5.0f64..5.0, az in -5.0f64..5.0, ar in 0.01f64..1.0,
        bx in -5.0f64..5.0, by in -5.0f64..5.0, bz in -5.0f64..5.0, br in 0.01f64..1.0,
    ) {
        let mut a = SpheresSoA::new();
        a.push(ax, ay, az, ar, 0);
        let mut b = SpheresSoA::new();
        b.push(bx, by, bz, br, 0);

        let ab = a.any_overlap(&b);
        let ba = b.any_overlap(&a);
        prop_assert_eq!(ab, ba, "any_overlap not symmetric");

        let dist_ab = a.signed_distance(0, &b, 0);
        let dist_ba = b.signed_distance(0, &a, 0);
        prop_assert!((dist_ab - dist_ba).abs() < 1e-12,
            "signed_distance not symmetric: {} vs {}", dist_ab, dist_ba);
    }
}

// ─── Scene add/remove idempotency ────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn scene_add_remove_idempotent(
        n in 1usize..20,
        seed in 0u64..1000,
    ) {
        use rand::SeedableRng;
        use rand::Rng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let robot = ur5e();
        let mut scene = Scene::new(&robot).unwrap();

        let names: Vec<String> = (0..n).map(|i| format!("obj_{}", i)).collect();
        for name in &names {
            let x: f64 = rng.gen_range(-2.0..2.0);
            let y: f64 = rng.gen_range(-2.0..2.0);
            let z: f64 = rng.gen_range(0.0..2.0);
            let r: f64 = rng.gen_range(0.01..0.2);
            scene.add(name, kinetic::scene::Shape::sphere(r),
                nalgebra::Isometry3::translation(x, y, z));
        }
        prop_assert_eq!(scene.num_objects(), n);

        for name in &names {
            scene.remove(name);
        }
        prop_assert_eq!(scene.num_objects(), 0);

        let env = scene.build_environment_spheres();
        prop_assert_eq!(env.len(), 0, "Environment should be empty after removing all objects");
    }
}

// ─── Joint clamping always within limits ─────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(500))]

    #[test]
    fn joint_clamping_within_limits(
        raw_values in prop::collection::vec(-100.0f64..100.0, 6..=6),
    ) {
        let robot = ur5e();
        let mut joints = JointValues::new(raw_values);
        robot.clamp_to_limits(&mut joints);

        for (i, &val) in joints.as_slice().iter().enumerate() {
            let limit = &robot.joint_limits[i];
            prop_assert!(val >= limit.lower && val <= limit.upper,
                "Joint {} = {} outside [{}, {}] after clamping",
                i, val, limit.lower, limit.upper);
        }
    }
}

// ─── SIMD matches scalar ─────────────────────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn simd_matches_scalar_collision(
        seed in 0u64..10000,
    ) {
        use rand::SeedableRng;
        use rand::Rng;
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

        let n_a = rng.gen_range(1..30usize);
        let n_b = rng.gen_range(1..30usize);

        let mut a = SpheresSoA::new();
        let mut b = SpheresSoA::new();

        for _ in 0..n_a {
            a.push(
                rng.gen_range(-2.0..2.0),
                rng.gen_range(-2.0..2.0),
                rng.gen_range(-2.0..2.0),
                rng.gen_range(0.01..0.5),
                0,
            );
        }
        for _ in 0..n_b {
            b.push(
                rng.gen_range(-2.0..2.0),
                rng.gen_range(-2.0..2.0),
                rng.gen_range(-2.0..2.0),
                rng.gen_range(0.01..0.5),
                0,
            );
        }

        let simd_result = simd::any_collision(&a, &b);
        let scalar_result = simd::scalar::any_collision_scalar(&a, &b);
        prop_assert_eq!(simd_result, scalar_result,
            "SIMD collision != scalar for {}x{} spheres", n_a, n_b);

        let simd_dist = simd::min_distance(&a, &b);
        let scalar_dist = simd::scalar::min_distance_scalar(&a, &b);
        prop_assert!((simd_dist - scalar_dist).abs() < 1e-10,
            "SIMD dist ({}) != scalar dist ({})", simd_dist, scalar_dist);
    }
}

// ─── Servo output within velocity limits ─────────────────────────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(100))]

    #[test]
    fn servo_velocity_within_limits(
        lx in -0.1f64..0.1,
        ly in -0.1f64..0.1,
        lz in -0.1f64..0.1,
    ) {
        let robot = Arc::new(ur5e());
        let scene = Arc::new(Scene::new(&robot).unwrap());
        let vel_limits = vec![1.0; robot.dof];
        let config = ServoConfig {
            velocity_limits: vel_limits.clone(),
            ..Default::default()
        };
        let mut servo = Servo::new(&robot, &scene, config).unwrap();

        let start = vec![0.0, -1.0, 0.8, 0.0, 1.0, 0.0];
        let zeros = vec![0.0; robot.dof];
        servo.set_state(&start, &zeros).unwrap();

        let twist = Twist::new(
            nalgebra::Vector3::new(lx, ly, lz),
            nalgebra::Vector3::new(0.0, 0.0, 0.0),
        );

        if let Ok(cmd) = servo.send_twist(&twist) {
            for (i, &v) in cmd.velocities.iter().enumerate() {
                prop_assert!(v.abs() <= vel_limits[i] + 1e-10,
                    "Joint {} velocity {:.6} exceeds limit {:.6}",
                    i, v, vel_limits[i]);
            }
        }
    }
}

// ─── FK determinism: same inputs always produce same output ──────────────────

proptest! {
    #![proptest_config(ProptestConfig::with_cases(200))]

    #[test]
    fn fk_deterministic(joints in arb_ur5e_joints()) {
        let robot = ur5e();
        let chain = ur5e_chain(&robot);

        let pose1 = forward_kinematics(&robot, &chain, &joints).unwrap();
        let pose2 = forward_kinematics(&robot, &chain, &joints).unwrap();

        let t1 = pose1.translation();
        let t2 = pose2.translation();

        for j in 0..3 {
            prop_assert!((t1[j] - t2[j]).abs() < 1e-15,
                "FK not deterministic: component {}: {} vs {}", j, t1[j], t2[j]);
        }
    }
}
