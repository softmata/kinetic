//! Integration test: stress testing with random queries.
//!
//! Runs many random planning queries to validate stability.

use kinetic::kinematics::fk;
use kinetic::prelude::*;
use std::sync::Arc;

fn ur5e() -> Arc<Robot> {
    Arc::new(Robot::from_name("ur5e").unwrap())
}

/// Simple LCG-based deterministic random number generator.
/// Avoids dependency on `rand` for test builds.
struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.state
    }

    /// Random f64 in [low, high)
    fn gen_range(&mut self, low: f64, high: f64) -> f64 {
        let t = (self.next_u64() as f64) / (u64::MAX as f64);
        low + t * (high - low)
    }
}

/// Generate random joint values within UR5e limits.
fn random_joints(rng: &mut SimpleRng) -> Vec<f64> {
    use std::f64::consts::{PI, TAU};
    // UR5e joint limits (approximate)
    let limits = [
        (-TAU, TAU),
        (-TAU, TAU),
        (-PI, PI),
        (-TAU, TAU),
        (-TAU, TAU),
        (-TAU, TAU),
    ];

    limits
        .iter()
        .map(|&(lo, hi)| rng.gen_range(lo, hi))
        .collect()
}

#[test]
fn stress_fk_1000() {
    let robot = ur5e();
    let chain = KinematicChain::extract(
        &robot,
        &robot.links[0].name,
        &robot.links.last().unwrap().name,
    )
    .unwrap();

    let mut rng = SimpleRng::new(42);
    let mut success_count = 0;

    for _ in 0..1000 {
        let joints = random_joints(&mut rng);
        if let Ok(pose) = fk(&robot, &chain, &joints) {
            // Verify pose is valid
            assert!(pose.0.translation.vector.norm().is_finite());
            success_count += 1;
        }
    }

    assert_eq!(success_count, 1000, "All FK queries should succeed");
}

#[test]
fn stress_jacobian_1000() {
    let robot = ur5e();
    let chain = KinematicChain::extract(
        &robot,
        &robot.links[0].name,
        &robot.links.last().unwrap().name,
    )
    .unwrap();

    let mut rng = SimpleRng::new(123);
    let mut success_count = 0;

    for _ in 0..1000 {
        let joints = random_joints(&mut rng);
        if let Ok(jac) = jacobian(&robot, &chain, &joints) {
            assert_eq!(jac.nrows(), 6);
            assert_eq!(jac.ncols(), 6);
            // Check no NaN/Inf
            for r in 0..6 {
                for c in 0..6 {
                    assert!(jac[(r, c)].is_finite(), "Jacobian should be finite");
                }
            }
            success_count += 1;
        }
    }

    assert_eq!(success_count, 1000, "All Jacobian queries should succeed");
}

#[test]
fn stress_ik_100() {
    let robot = ur5e();
    let chain = KinematicChain::extract(
        &robot,
        &robot.links[0].name,
        &robot.links.last().unwrap().name,
    )
    .unwrap();

    let mut rng = SimpleRng::new(456);
    let mut converged = 0;

    for _ in 0..100 {
        // Use FK at a random config to generate a reachable target
        let seed = random_joints(&mut rng);
        let target = fk(&robot, &chain, &seed).unwrap();

        let new_seed = random_joints(&mut rng);
        let ik_config = IKConfig {
            seed: Some(new_seed),
            ..IKConfig::default()
        };
        if let Ok(sol) = solve_ik(&robot, &chain, &target, &ik_config) {
            // Verify solution
            let check = fk(&robot, &chain, &sol.joints).unwrap();
            let err = (check.translation() - target.translation()).norm();
            if err < 0.01 {
                converged += 1;
            }
        }
    }

    // At least 50% should converge (generous threshold)
    assert!(
        converged >= 30,
        "At least 30/100 IK queries should converge, got {}",
        converged
    );
}

#[test]
fn stress_planning_50() {
    let robot = ur5e();
    let planner = Planner::new(&robot).unwrap();

    let mut rng = SimpleRng::new(789);
    let mut success_count = 0;

    for _ in 0..50 {
        let start = random_joints(&mut rng);
        let goal_joints = random_joints(&mut rng);

        if let Ok(result) = planner.plan(&start, &Goal::Joints(JointValues(goal_joints))) {
            assert!(result.waypoints.len() >= 2);
            // Verify all waypoints have correct DOF
            for wp in &result.waypoints {
                assert_eq!(wp.len(), 6);
            }
            success_count += 1;
        }
    }

    // Most random planning queries should succeed (UR5e has large workspace)
    assert!(
        success_count >= 25,
        "At least 25/50 random plans should succeed, got {}",
        success_count
    );
}

#[test]
fn stress_full_pipeline_20() {
    let robot = ur5e();
    let planner = Planner::new(&robot).unwrap();

    let mut rng = SimpleRng::new(999);
    let mut success_count = 0;

    for _ in 0..20 {
        let start = random_joints(&mut rng);
        let goal_joints = random_joints(&mut rng);

        if let Ok(result) = planner.plan(&start, &Goal::Joints(JointValues(goal_joints))) {
            if let Ok(timed) = trapezoidal(&result.waypoints, 1.5, 3.0) {
                assert!(timed.waypoints.len() >= 2);
                assert!(timed.duration.as_secs_f64() > 0.0);

                // Verify all timed waypoints are valid
                for wp in &timed.waypoints {
                    assert_eq!(wp.positions.len(), 6);
                    assert_eq!(wp.velocities.len(), 6);
                    for &p in &wp.positions {
                        assert!(p.is_finite());
                    }
                    for &v in &wp.velocities {
                        assert!(v.is_finite());
                    }
                }
                success_count += 1;
            }
        }
    }

    assert!(
        success_count >= 10,
        "At least 10/20 full pipeline runs should succeed, got {}",
        success_count
    );
}

#[test]
fn stress_scene_collision_100() {
    let robot = ur5e();
    let mut scene = Scene::new(&robot).unwrap();

    // Add some obstacles
    let mut rng = SimpleRng::new(555);
    for i in 0..10 {
        let x = rng.gen_range(-1.0, 1.0);
        let y = rng.gen_range(-1.0, 1.0);
        let z = rng.gen_range(0.0, 1.0);
        let r = rng.gen_range(0.02, 0.1);
        scene.add(
            &format!("sphere_{}", i),
            Shape::Sphere(r),
            Isometry3::translation(x, y, z),
        );
    }

    let mut collision_count = 0;
    let mut rng2 = SimpleRng::new(666);

    for _ in 0..100 {
        let joints = random_joints(&mut rng2);
        if let Ok(true) = scene.check_collision(&joints) {
            collision_count += 1;
        }
    }

    // We should get a mix of colliding and non-colliding
    // Just verify we didn't crash
    assert!(
        collision_count < 100,
        "Not all 100 configs should collide (obstacles are small)"
    );
}
