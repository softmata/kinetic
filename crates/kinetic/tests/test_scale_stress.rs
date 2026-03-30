//! Large-scale stress tests for collision, scene, planning, and IK.
//!
//! Exercises the system under heavy load: many obstacles, large pointclouds,
//! batch IK, long trajectories, and all robots loaded simultaneously.

use std::time::{Duration, Instant};

use kinetic::collision::{simd, SpheresSoA};
use kinetic::kinematics::{forward_kinematics, solve_ik, IKConfig, IKSolver, KinematicChain};
use kinetic::prelude::*;
use kinetic::scene::{Octree, OctreeConfig, PointCloudConfig, Scene};
use kinetic::trajectory::trapezoidal;

fn ur5e_robot_and_chain() -> (Robot, KinematicChain) {
    let robot = Robot::from_name("ur5e").unwrap();
    let arm = &robot.groups["arm"];
    let chain = KinematicChain::extract(&robot, &arm.base_link, &arm.tip_link).unwrap();
    (robot, chain)
}

// ─── Many-obstacle scene ────────────────────────────────────────────────────

#[test]
fn scene_100_box_obstacles_collision_check() {
    let robot = Robot::from_name("ur5e").unwrap();
    let mut scene = Scene::new(&robot).unwrap();

    // Add 100 small box obstacles scattered in the workspace
    for i in 0..100 {
        let x = 0.2 + (i % 10) as f64 * 0.08;
        let y = -0.4 + (i / 10) as f64 * 0.08;
        let z = 0.1 + (i % 5) as f64 * 0.1;
        scene.add(
            &format!("box_{i}"),
            Shape::cuboid(0.01, 0.01, 0.01),
            Isometry3::translation(x, y, z),
        );
    }

    assert_eq!(scene.num_objects(), 100);

    let joints = vec![0.5, -1.0, 1.0, -0.5, 1.0, 0.3];
    let n = 100;

    let t0 = Instant::now();
    for _ in 0..n {
        let _ = scene.check_collision(&joints).unwrap();
    }
    let elapsed = t0.elapsed();
    let mean_ms = elapsed.as_millis() / n as u128;

    eprintln!("100 obstacles: {mean_ms}ms mean collision check over {n} calls");
    assert!(
        mean_ms < 500,
        "Collision check with 100 obstacles should be <500ms, got {mean_ms}ms"
    );
}

#[test]
fn scene_500_sphere_obstacles() {
    let robot = Robot::from_name("ur5e").unwrap();
    let mut scene = Scene::new(&robot).unwrap();

    for i in 0..500 {
        let angle = i as f64 * 0.1;
        let r = 0.3 + (i % 20) as f64 * 0.03;
        let x = r * angle.cos();
        let y = r * angle.sin();
        let z = 0.1 + (i % 10) as f64 * 0.05;
        scene.add(
            &format!("sphere_{i}"),
            Shape::sphere(0.01),
            Isometry3::translation(x, y, z),
        );
    }

    assert_eq!(scene.num_objects(), 500);

    // Collision check should still work
    let joints = vec![0.0, -1.0, 0.8, 0.0, 1.0, 0.0];
    let result = scene.check_collision(&joints);
    assert!(result.is_ok(), "Should not panic with 500 obstacles");
}

// ─── Planning in cluttered environment ──────────────────────────────────────

#[test]
fn planning_cluttered_50_obstacles() {
    let robot = Robot::from_name("ur5e").unwrap();
    let mut scene = Scene::new(&robot).unwrap();

    // 50 small obstacles scattered around workspace
    for i in 0..50 {
        let x = 0.2 + (i % 7) as f64 * 0.1;
        let y = -0.3 + (i / 7) as f64 * 0.1;
        let z = 0.2 + (i % 5) as f64 * 0.08;
        scene.add(
            &format!("obs_{i}"),
            Shape::sphere(0.015),
            Isometry3::translation(x, y, z),
        );
    }

    let planner = Planner::new(&robot)
        .unwrap()
        .with_scene(&scene)
        .with_config(PlannerConfig {
            timeout: Duration::from_secs(5),
            ..PlannerConfig::default()
        });

    let start = vec![0.0, -1.0, 0.8, 0.0, 1.0, 0.0];
    let goal = Goal::Joints(JointValues::new(vec![1.0, -0.5, 0.3, 0.2, -0.3, 0.5]));

    let t0 = Instant::now();
    let result = planner.plan(&start, &goal);
    let elapsed = t0.elapsed();

    eprintln!(
        "Cluttered planning (50 obstacles): {:?}, result={}",
        elapsed,
        result.is_ok()
    );

    // Should not hang or OOM — may fail (acceptable), but should finish within timeout
    assert!(
        elapsed < Duration::from_secs(15),
        "Cluttered planning should respect timeout: {:?}",
        elapsed
    );
}

// ─── Pointcloud ingestion ───────────────────────────────────────────────────

#[test]
fn pointcloud_100k_points_ingestion() {
    let robot = Robot::from_name("ur5e").unwrap();
    let mut scene = Scene::new(&robot).unwrap();

    // Generate 100K points on a surface
    let n = 100_000;
    let mut points = Vec::with_capacity(n);
    for i in 0..n {
        let x = 0.3 + (i % 100) as f64 * 0.005;
        let y = -0.25 + (i / 100 % 100) as f64 * 0.005;
        let z = 0.01 + (i / 10000) as f64 * 0.001;
        points.push([x, y, z]);
    }

    let config = PointCloudConfig {
        sphere_radius: 0.005,
        max_points: 100_000,
        ..Default::default()
    };

    let t0 = Instant::now();
    scene.add_pointcloud("scan", &points, config);
    let elapsed = t0.elapsed();

    eprintln!("100K pointcloud ingestion: {:?}", elapsed);
    assert_eq!(scene.num_pointclouds(), 1);
    assert!(
        elapsed < Duration::from_secs(10),
        "100K pointcloud should ingest in <10s, took {:?}",
        elapsed
    );

    // Collision check should still work with pointcloud
    let joints = vec![0.5, -1.0, 1.0, -0.5, 1.0, 0.3];
    let result = scene.check_collision(&joints);
    assert!(result.is_ok());
}

#[test]
fn pointcloud_1m_points_ingestion_timing() {
    let robot = Robot::from_name("ur5e").unwrap();
    let mut scene = Scene::new(&robot).unwrap();

    // 1M points — should be downsampled to max_points
    let n = 1_000_000;
    let mut points = Vec::with_capacity(n);
    for i in 0..n {
        let x = 0.2 + (i % 1000) as f64 * 0.001;
        let y = -0.5 + (i / 1000 % 1000) as f64 * 0.001;
        let z = 0.0 + (i / 1_000_000) as f64 * 0.001;
        points.push([x, y, z]);
    }

    let config = PointCloudConfig {
        sphere_radius: 0.005,
        max_points: 50_000, // Downsample to 50K
        ..Default::default()
    };

    let t0 = Instant::now();
    scene.add_pointcloud("large_scan", &points, config);
    let elapsed = t0.elapsed();

    eprintln!("1M pointcloud (downsampled to 50K): {:?}", elapsed);
    assert_eq!(scene.num_pointclouds(), 1);

    // Should complete, not OOM
    assert!(
        elapsed < Duration::from_secs(30),
        "1M pointcloud should complete in <30s, took {:?}",
        elapsed
    );
}

// ─── Octree construction ────────────────────────────────────────────────────

#[test]
fn octree_100k_points() {
    let config = OctreeConfig {
        resolution: 0.02,
        half_extent: 2.0,
        ray_cast_free_space: false,
        ..Default::default()
    };
    let mut octree = Octree::new(config);

    // Insert 100K points
    let n = 100_000;
    let mut points = Vec::with_capacity(n);
    for i in 0..n {
        let x = -1.0 + (i % 200) as f64 * 0.01;
        let y = -1.0 + (i / 200 % 200) as f64 * 0.01;
        let z = 0.0 + (i / 40000) as f64 * 0.01;
        points.push([x, y, z]);
    }

    let t0 = Instant::now();
    octree.insert_points_occupied(&points);
    let elapsed = t0.elapsed();

    eprintln!(
        "Octree 100K points: {:?}, occupied={}, leaves={}",
        elapsed,
        octree.num_occupied(),
        octree.num_leaves()
    );

    assert!(octree.num_occupied() > 0);
    assert!(
        elapsed < Duration::from_secs(10),
        "Octree 100K insertion should be <10s, took {:?}",
        elapsed
    );
}

// ─── Large SIMD collision ───────────────────────────────────────────────────

#[test]
#[ignore = "benchmark: timing-sensitive, run with --ignored on idle machine"]
fn simd_collision_1000_spheres() {
    // 1000 robot spheres vs 1000 env spheres
    let mut robot_spheres = SpheresSoA::new();
    let mut env_spheres = SpheresSoA::new();

    for i in 0..1000 {
        let f = i as f64 * 0.01;
        robot_spheres.push(f.cos() * 0.5, f.sin() * 0.5, f * 0.001, 0.01, 0);
        env_spheres.push(f.cos() * 0.5 + 2.0, f.sin() * 0.5, f * 0.001, 0.01, 1);
    }

    let n = 100;
    let t0 = Instant::now();
    for _ in 0..n {
        let _ = simd::any_collision(&robot_spheres, &env_spheres);
    }
    let elapsed = t0.elapsed();
    let mean_us = elapsed.as_micros() / n as u128;

    eprintln!("SIMD 1000v1000 spheres: mean={mean_us}μs over {n} calls");

    // Should handle 1M pair-checks efficiently
    assert!(
        mean_us < 100_000,
        "1000v1000 collision should be <100ms, got {mean_us}μs"
    );
}

// ─── Batch IK solving ───────────────────────────────────────────────────────

#[test]
fn batch_ik_1000_random_targets() {
    let (robot, chain) = ur5e_robot_and_chain();

    // Generate 1000 targets via FK from random joint configs
    let n = 1000;
    let mut targets = Vec::with_capacity(n);
    let mut rng_state: u64 = 42;

    for _ in 0..n {
        // Simple LCG pseudo-random
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let q: Vec<f64> = (0..6)
            .map(|_| {
                rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let t = (rng_state >> 33) as f64 / (1u64 << 31) as f64; // [0, 1)
                -std::f64::consts::PI + t * 2.0 * std::f64::consts::PI
            })
            .collect();
        if let Ok(pose) = forward_kinematics(&robot, &chain, &q) {
            targets.push(pose);
        }
    }

    let config = IKConfig {
        solver: IKSolver::DLS { damping: 0.05 },
        max_iterations: 100,
        num_restarts: 1,
        ..IKConfig::default()
    };

    let t0 = Instant::now();
    let mut successes = 0;
    for target in &targets {
        if let Ok(sol) = solve_ik(&robot, &chain, target, &config) {
            if sol.converged {
                successes += 1;
            }
        }
    }
    let elapsed = t0.elapsed();
    let total = targets.len();

    eprintln!(
        "Batch IK: {successes}/{total} converged in {:?} ({:.1}ms/solve)",
        elapsed,
        elapsed.as_millis() as f64 / total as f64
    );

    // Most FK-generated targets should be solvable
    assert!(
        successes > total / 4,
        "At least 25% of FK-generated targets should converge: {successes}/{total}"
    );
}

// ─── Long trajectory parameterization ───────────────────────────────────────

#[test]
fn trajectory_10000_waypoints() {
    let dof = 6;
    let n = 10_000;

    // Generate smooth waypoints (sin wave per joint)
    let waypoints: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            let t = i as f64 / n as f64 * 4.0 * std::f64::consts::PI;
            (0..dof)
                .map(|j| {
                    let phase = j as f64 * 0.5;
                    (t + phase).sin() * 0.5
                })
                .collect()
        })
        .collect();

    let t0 = Instant::now();
    let result = trapezoidal(&waypoints, 2.0, 5.0);
    let elapsed = t0.elapsed();

    eprintln!("10K waypoint trajectory: {:?}", elapsed);

    match result {
        Ok(traj) => {
            // Trapezoidal adds intermediate phase-boundary waypoints
            assert!(
                traj.waypoints.len() >= n,
                "Should have at least {n} waypoints, got {}",
                traj.waypoints.len()
            );
            assert!(traj.duration() > Duration::ZERO);
            assert!(traj.validate().is_ok());
        }
        Err(e) => {
            eprintln!("Trajectory failed (may be expected for zero-distance segments): {e}");
        }
    }

    assert!(
        elapsed < Duration::from_secs(10),
        "10K waypoint trajectory should complete in <10s: {:?}",
        elapsed
    );
}

// ─── All robots loaded ─────────────────────────────────────────────────────

#[test]
fn load_all_52_robots() {
    let names = [
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

    let t0 = Instant::now();
    let mut robots = Vec::new();
    let mut failed = Vec::new();

    for name in &names {
        match Robot::from_name(name) {
            Ok(r) => robots.push(r),
            Err(e) => failed.push(format!("{name}: {e}")),
        }
    }
    let elapsed = t0.elapsed();

    eprintln!(
        "Loaded {}/{} robots in {:?}",
        robots.len(),
        names.len(),
        elapsed
    );

    if !failed.is_empty() {
        eprintln!("Failed: {:?}", failed);
    }

    assert_eq!(robots.len(), names.len(), "All robots should load");
    assert!(
        elapsed < Duration::from_secs(30),
        "Loading all robots should be <30s: {:?}",
        elapsed
    );

    // Verify each has valid DOF and at least one joint
    for r in &robots {
        assert!(r.dof > 0, "Robot {} has 0 DOF", r.name);
        assert!(!r.joints.is_empty(), "Robot {} has no joints", r.name);
    }
}

// ─── Rapid scene add/remove cycles ─────────────────────────────────────────

#[test]
fn rapid_scene_add_remove_1000_cycles() {
    let robot = Robot::from_name("ur5e").unwrap();
    let mut scene = Scene::new(&robot).unwrap();

    let t0 = Instant::now();
    for i in 0..1000 {
        let name = format!("obj_{}", i % 50); // Reuse names to test overwrite
        scene.add(
            &name,
            Shape::sphere(0.02),
            Isometry3::translation(
                0.3 + (i % 10) as f64 * 0.05,
                (i % 7) as f64 * 0.05,
                0.1 + (i % 5) as f64 * 0.04,
            ),
        );

        // Remove every other cycle
        if i % 2 == 1 {
            scene.remove(&name);
        }
    }
    let elapsed = t0.elapsed();

    eprintln!("1000 add/remove cycles: {:?}", elapsed);

    assert!(
        elapsed < Duration::from_secs(5),
        "1000 scene cycles should be <5s: {:?}",
        elapsed
    );

    // Scene should still be functional
    let joints = vec![0.5, -1.0, 1.0, -0.5, 1.0, 0.3];
    let result = scene.check_collision(&joints);
    assert!(result.is_ok());
}
