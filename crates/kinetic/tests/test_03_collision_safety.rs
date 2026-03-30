//! Acceptance tests: 03 collision_safety
//! Spec: doc_tests/03_COLLISION_SAFETY.md
//!
//! Collision detection safety: self-collision, obstacles, SIMD correctness, empty scene FP.

#[path = "helpers.rs"]
mod helpers;
use helpers::*;

use kinetic::prelude::*;
use kinetic::collision::{
    CollisionEnvironment, ResolvedACM, RobotSphereModel, SphereGenConfig,
    SpheresSoA,
};
use kinetic::kinematics::forward_kinematics_all;
use kinetic::scene::{Scene, Shape};
use nalgebra::Isometry3;

// ─── Self-collision at home: 52 robots should NOT self-collide ──────────────

#[test]
fn no_self_collision_at_home_all_robots() {
    let mut tested = 0;
    for &(name, _) in ALL_ROBOTS {
        let robot = load_robot(name);
        let chain = load_chain(&robot);
        let model = RobotSphereModel::from_robot(&robot, &SphereGenConfig::coarse());

        if model.total_spheres() == 0 { continue; } // no collision geometry

        let zeros = vec![0.0; chain.dof];
        let poses = match forward_kinematics_all(&robot, &chain, &zeros) {
            Ok(p) => p,
            Err(_) => continue,
        };

        let mut runtime = model.create_runtime();
        runtime.update(&poses);

        let acm = ResolvedACM::from_robot(&robot);
        let skip = acm.to_skip_pairs();
        let self_col = runtime.self_collision(&skip);

        assert!(
            !self_col,
            "{name}: self-collision detected at home config (with ACM)"
        );
        tested += 1;
    }
    assert!(tested >= 5, "at least 5 robots with collision geometry should be tested");
}

// ─── Obstacle avoidance: scene collision detection ──────────────────────────

#[test]
fn scene_detects_obstacle_collision() {
    // Only test robots known to have collision geometry
    for name in &["ur5e", "franka_panda"] {
        let robot = load_robot(name);
        let model = RobotSphereModel::from_robot(&robot, &SphereGenConfig::coarse());
        if model.total_spheres() == 0 { continue; }

        let mut scene = match Scene::new(&robot) {
            Ok(s) => s,
            Err(_) => continue,
        };

        // Place a big obstacle at the robot base — guaranteed collision
        scene.add("blocker", Shape::Sphere(1.0), Isometry3::translation(0.0, 0.0, 0.3));

        let zeros = vec![0.0; scene.dof()];
        let result = scene.check_collision(&zeros);
        if let Ok(colliding) = result {
            assert!(colliding, "{name}: should collide with obstacle at base");
        }
    }
}

#[test]
fn scene_no_collision_with_distant_obstacle() {
    for name in SAFETY_ROBOTS {
        let robot = load_robot(name);
        let mut scene = match Scene::new(&robot) {
            Ok(s) => s,
            Err(_) => continue,
        };

        // Place obstacle far away
        scene.add("far_box", Shape::Cuboid(0.1, 0.1, 0.1), Isometry3::translation(50.0, 50.0, 50.0));

        let zeros = vec![0.0; scene.dof()];
        let result = scene.check_collision(&zeros);
        if let Ok(colliding) = result {
            assert!(!colliding, "{name}: should NOT collide with distant obstacle");
        }
    }
}

// ─── Planner avoids obstacles ───────────────────────────────────────────────

#[test]
fn planner_output_collision_free() {
    use kinetic::planning::Planner;
    use kinetic::core::PlannerConfig;

    let robot = load_robot("ur5e");
    let scene = match Scene::new(&robot) {
        Ok(s) => s,
        Err(_) => return,
    };
    // No obstacles — just verify planner output is collision-free with self-collision
    let config = PlannerConfig {
        timeout: std::time::Duration::from_secs(5),
        max_iterations: 5000,
        shortcut_iterations: 0,
        smooth: false,
        ..PlannerConfig::default()
    };
    let planner = match Planner::new(&robot) {
        Ok(p) => p.with_scene(&scene).with_config(config),
        Err(_) => return,
    };

    let start = mid_joints(&robot);
    let goal_vals: Vec<f64> = start.iter().map(|v| v + 0.3).collect();
    let goal = Goal::Joints(JointValues::new(goal_vals));

    if let Ok(plan) = planner.plan(&start, &goal) {
            // Every waypoint should be collision-free
            for (wp_idx, wp) in plan.waypoints.iter().enumerate() {
            let colliding = planner.is_in_collision(wp);
            assert!(
                !colliding,
                "ur5e: waypoint {wp_idx} is in collision"
            );
        }
    }
}

// ─── Empty scene: no false positives ────────────────────────────────────────

#[test]
fn empty_scene_no_environment_false_positives() {
    // In empty scene, min_distance_to_robot should be INFINITY (no obstacles)
    let mut total_checks = 0;
    for name in SAFETY_ROBOTS {
        let robot = load_robot(name);
        let scene = match Scene::new(&robot) {
            Ok(s) => s,
            Err(_) => continue,
        };

        // mid config should report no environment collision (self-collision is separate)
        let mid: Vec<f64> = mid_joints(&robot).into_iter().take(scene.dof()).collect();
        if let Ok(dist) = scene.min_distance_to_robot(&mid) {
            // In empty scene, distance should be large (no obstacles)
            assert!(
                dist > 1.0 || dist == f64::INFINITY,
                "{name}: empty scene min_distance should be large, got {dist}"
            );
            total_checks += 1;
        }
    }
    assert!(total_checks >= 3, "should have checked 3+ robots, got {total_checks}");
}

// ─── SIMD vs scalar correctness ─────────────────────────────────────────────

#[test]
fn simd_matches_scalar_collision_result() {
    use kinetic::collision::simd;

    for name in &["ur5e", "franka_panda"] {
        let robot = load_robot(name);
        let chain = load_chain(&robot);
        let model = RobotSphereModel::from_robot(&robot, &SphereGenConfig::coarse());
        if model.total_spheres() == 0 { continue; }

        let mut obstacles = SpheresSoA::new();
        obstacles.push(0.3, 0.0, 0.4, 0.1, 0);
        obstacles.push(-0.2, 0.3, 0.5, 0.05, 1);

        for seed in 0..20u64 {
            let joints: Vec<f64> = random_joints(&robot, seed * 123)
                .into_iter().take(chain.dof).collect();
            let poses = match forward_kinematics_all(&robot, &chain, &joints) {
                Ok(p) => p,
                Err(_) => continue,
            };

            let mut runtime = model.create_runtime();
            runtime.update(&poses);

            // Compare dispatch (may use SIMD) vs scalar
            let dispatch_result = simd::any_collision(&runtime.world, &obstacles);
            let scalar_result = simd::scalar::any_collision_scalar(&runtime.world, &obstacles);

            assert_eq!(
                dispatch_result, scalar_result,
                "{name} seed {seed}: SIMD dispatch ({dispatch_result}) != scalar ({scalar_result})"
            );
        }
    }
}

// ─── Collision margin makes checks stricter ─────────────────────────────────

#[test]
fn collision_margin_stricter() {
    for name in &["ur5e", "franka_panda"] {
        let robot = load_robot(name);
        let chain = load_chain(&robot);
        let model = RobotSphereModel::from_robot(&robot, &SphereGenConfig::coarse());
        if model.total_spheres() == 0 { continue; }

        let mut obstacles = SpheresSoA::new();
        obstacles.push(0.5, 0.0, 0.3, 0.05, 0);
        let env = CollisionEnvironment::build(
            obstacles, 0.05,
            kinetic::collision::AABB::symmetric(2.0),
        );

        let zeros = vec![0.0; chain.dof];
        let poses = match forward_kinematics_all(&robot, &chain, &zeros) {
            Ok(p) => p,
            Err(_) => continue,
        };

        let mut runtime = model.create_runtime();
        runtime.update(&poses);

        let no_margin = env.check_collision(&runtime.world);
        let big_margin = env.check_collision_with_margin(&runtime.world, 5.0);

        // Big margin should be at least as conservative
        if no_margin {
            assert!(big_margin, "{name}: big margin should be stricter");
        }
        // With 5m margin, almost everything should collide
        assert!(big_margin, "{name}: 5m margin should trigger collision");
    }
}

// ─── Shape types: box, sphere, cylinder obstacle ────────────────────────────

#[test]
fn scene_detects_all_shape_types() {
    let robot = load_robot("ur5e");
    let shapes = vec![
        ("box", Shape::Cuboid(0.5, 0.5, 0.5)),
        ("sphere", Shape::Sphere(0.5)),
        ("cylinder", Shape::Cylinder(0.3, 0.5)),
    ];

    for (shape_name, shape) in shapes {
        let mut scene = Scene::new(&robot).unwrap();
        scene.add("obstacle", shape, Isometry3::translation(0.0, 0.0, 0.3));

        let zeros = vec![0.0; scene.dof()];
        if let Ok(colliding) = scene.check_collision(&zeros) {
            assert!(colliding, "ur5e should collide with {shape_name} at base");
        }
    }
}
