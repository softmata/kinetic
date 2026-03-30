//! Real-time timing assertion tests.
//!
//! Verifies performance guarantees for FK, Jacobian, IK, collision checking,
//! servo loop, planning timeout, and trajectory sampling. All tests use
//! `std::time::Instant` for wall-clock measurement.
//!
//! Note: These thresholds are for debug builds. Release builds will be faster.
//! Thresholds are set conservatively to avoid CI flakiness while still catching
//! gross regressions.

use std::sync::Arc;
use std::time::{Duration, Instant};

use kinetic::collision::{simd, SpheresSoA};
use kinetic::core::Twist;
use kinetic::kinematics::{forward_kinematics, jacobian, KinematicChain};
use kinetic::prelude::*;
use kinetic::scene::Scene;
use kinetic::trajectory::trapezoidal;

fn ur5e_robot_and_chain() -> (Robot, KinematicChain) {
    let robot = Robot::from_name("ur5e").unwrap();
    let arm = &robot.groups["arm"];
    let chain = KinematicChain::extract(&robot, &arm.base_link, &arm.tip_link).unwrap();
    (robot, chain)
}

// ─── FK timing ──────────────────────────────────────────────────────────────

#[test]
#[ignore = "benchmark: timing-sensitive, run with --ignored on idle machine"]
fn fk_latency_10000_calls() {
    let (robot, chain) = ur5e_robot_and_chain();
    let n = 10_000u128;
    let joints = vec![0.5, -1.0, 1.0, -0.5, 1.0, 0.3];

    // Warmup — first call(s) may initialize caches
    for _ in 0..10 {
        let _ = forward_kinematics(&robot, &chain, &joints).unwrap();
    }

    let mut max_us = 0u128;
    let mut total_us = 0u128;

    for _ in 0..n {
        let t0 = Instant::now();
        let _ = forward_kinematics(&robot, &chain, &joints).unwrap();
        let elapsed = t0.elapsed().as_micros();
        total_us += elapsed;
        if elapsed > max_us {
            max_us = elapsed;
        }
    }

    let mean_us = total_us / n;
    eprintln!("FK: mean={mean_us}μs, max={max_us}μs over {n} calls");

    // Debug build thresholds (generous)
    assert!(mean_us < 500, "FK mean should be <500μs, got {mean_us}μs");
    assert!(max_us < 20_000, "FK max should be <20ms, got {max_us}μs");
}

// ─── Jacobian timing ────────────────────────────────────────────────────────

#[test]
#[ignore = "benchmark: timing-sensitive, run with --ignored on idle machine"]
fn jacobian_latency_10000_calls() {
    let (robot, chain) = ur5e_robot_and_chain();
    let n = 10_000u128;
    let joints = vec![0.5, -1.0, 1.0, -0.5, 1.0, 0.3];

    // Warmup
    for _ in 0..10 {
        let _ = jacobian(&robot, &chain, &joints).unwrap();
    }

    let mut max_us = 0u128;
    let mut total_us = 0u128;

    for _ in 0..n {
        let t0 = Instant::now();
        let _ = jacobian(&robot, &chain, &joints).unwrap();
        let elapsed = t0.elapsed().as_micros();
        total_us += elapsed;
        if elapsed > max_us {
            max_us = elapsed;
        }
    }

    let mean_us = total_us / n;
    eprintln!("Jacobian: mean={mean_us}μs, max={max_us}μs over {n} calls");

    assert!(
        mean_us < 1_000,
        "Jacobian mean should be <1ms, got {mean_us}μs"
    );
    assert!(
        max_us < 20_000,
        "Jacobian max should be <20ms, got {max_us}μs"
    );
}

// ─── Collision check timing ─────────────────────────────────────────────────

#[test]
fn collision_check_latency_1000_calls() {
    let robot = Robot::from_name("ur5e").unwrap();
    let mut scene = Scene::new(&robot).unwrap();

    // Add a small obstacle for realistic collision checking
    scene.add(
        "box",
        Shape::cuboid(0.05, 0.05, 0.05),
        Isometry3::translation(0.4, 0.0, 0.3),
    );

    let joints = vec![0.5, -1.0, 1.0, -0.5, 1.0, 0.3];
    let n = 1_000;

    let mut max_us = 0u128;
    let mut total_us = 0u128;

    for _ in 0..n {
        let t0 = Instant::now();
        let _ = scene.check_collision(&joints).unwrap();
        let elapsed = t0.elapsed().as_micros();
        total_us += elapsed;
        if elapsed > max_us {
            max_us = elapsed;
        }
    }

    let mean_us = total_us / n as u128;
    eprintln!("Collision check: mean={mean_us}μs, max={max_us}μs over {n} calls");

    assert!(
        mean_us < 5_000,
        "Collision mean should be <5ms, got {mean_us}μs"
    );
    assert!(
        max_us < 50_000,
        "Collision max should be <50ms, got {max_us}μs"
    );
}

// ─── SIMD collision kernel timing ───────────────────────────────────────────

#[test]
fn simd_collision_speedup() {
    // Build two sphere arrays (50 vs 50 — realistic robot/env interaction)
    let mut robot_spheres = SpheresSoA::new();
    let mut env_spheres = SpheresSoA::new();

    for i in 0..50 {
        let f = i as f64 * 0.05;
        robot_spheres.push(f, 0.0, 0.0, 0.02, 0);
        env_spheres.push(f + 1.0, 0.0, 0.0, 0.02, 1);
    }

    let n = 1_000;

    // Measure SIMD any_collision (dispatched)
    let t0 = Instant::now();
    for _ in 0..n {
        let _ = simd::any_collision(&robot_spheres, &env_spheres);
    }
    let simd_elapsed = t0.elapsed();

    // Measure scalar any_collision
    let t1 = Instant::now();
    for _ in 0..n {
        let _ = simd::scalar::any_collision_scalar(&robot_spheres, &env_spheres);
    }
    let scalar_elapsed = t1.elapsed();

    let simd_us = simd_elapsed.as_micros();
    let scalar_us = scalar_elapsed.as_micros();
    let tier = simd::detect_simd_tier();

    eprintln!(
        "SIMD collision ({tier}): {simd_us}μs vs scalar: {scalar_us}μs over {n} calls (50v50 spheres)"
    );

    // SIMD should be at least as fast as scalar (on AVX2/NEON, faster)
    // Allow small margin for measurement noise
    assert!(
        simd_us <= scalar_us + scalar_us / 10,
        "SIMD ({simd_us}μs) should not be significantly slower than scalar ({scalar_us}μs)"
    );
}

// ─── Planning timeout accuracy ──────────────────────────────────────────────

#[test]
fn planning_timeout_accuracy() {
    let robot = Robot::from_name("ur5e").unwrap();
    let timeout_ms = 500;

    let planner = Planner::new(&robot).unwrap().with_config(PlannerConfig {
        timeout: Duration::from_millis(timeout_ms),
        max_iterations: 10_000_000,
        shortcut_iterations: 0,
        smooth: false,
        ..PlannerConfig::default()
    });

    // Hard planning problem — long distance with many iterations allowed
    let start = vec![0.0; robot.dof];
    let goal = Goal::Joints(JointValues::new(vec![2.5, -1.5, 2.0, -1.0, 2.0, -1.0]));

    let t0 = Instant::now();
    let _ = planner.plan(&start, &goal);
    let elapsed = t0.elapsed();

    let elapsed_ms = elapsed.as_millis();
    eprintln!("Planning timeout={timeout_ms}ms, actual elapsed={elapsed_ms}ms");

    // Should respect timeout within a generous bound (debug build overhead)
    assert!(
        elapsed < Duration::from_secs(10),
        "Planning should respect {timeout_ms}ms timeout, took {elapsed_ms}ms"
    );
}

// ─── Trajectory sampling timing ─────────────────────────────────────────────

#[test]
#[ignore = "benchmark: timing-sensitive, run with --ignored on idle machine"]
fn trajectory_sampling_latency_100000() {
    let waypoints: Vec<Vec<f64>> = vec![
        vec![0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        vec![0.5, -0.3, 0.2, 0.1, -0.1, 0.4],
        vec![1.0, -1.0, 0.8, -0.5, 1.0, 0.3],
    ];
    let traj = trapezoidal(&waypoints, 2.0, 5.0).unwrap();
    let duration = traj.duration();
    let n = 100_000u128;

    let mut max_us = 0u128;
    let t0 = Instant::now();
    for i in 0..n as u32 {
        let frac = i as f64 / n as f64;
        let t = Duration::from_secs_f64(duration.as_secs_f64() * frac);
        let sample = traj.sample_at(t);
        // Prevent optimizer from eliding
        std::hint::black_box(&sample);
    }
    let total = t0.elapsed();

    // Also measure individual sample timing for max
    for i in 0..1000 {
        let frac = i as f64 / 1000.0;
        let t = Duration::from_secs_f64(duration.as_secs_f64() * frac);
        let t_start = Instant::now();
        let sample = traj.sample_at(t);
        std::hint::black_box(&sample);
        let elapsed = t_start.elapsed().as_micros();
        if elapsed > max_us {
            max_us = elapsed;
        }
    }

    let mean_ns = total.as_nanos() / n as u128;
    eprintln!("Trajectory sample: mean={mean_ns}ns, max={max_us}μs over {n} samples");

    // Each sample should be extremely fast (binary search + lerp)
    assert!(
        mean_ns < 10_000,
        "Sample mean should be <10μs, got {mean_ns}ns"
    );
    assert!(max_us < 1_000, "Sample max should be <1ms, got {max_us}μs");
}

// ─── Servo loop timing ──────────────────────────────────────────────────────

#[test]
fn servo_loop_1000_iterations() {
    let robot = Arc::new(Robot::from_name("ur5e").unwrap());
    let scene = Arc::new(Scene::new(&robot).unwrap());

    let config = kinetic::reactive::ServoConfig {
        rate_hz: 500.0,
        collision_check_hz: 50.0, // Lower frequency to reduce per-tick cost
        ..Default::default()
    };

    let mut servo = kinetic::reactive::Servo::new(&robot, &scene, config).unwrap();

    let twist = Twist::new(
        nalgebra::Vector3::new(0.01, 0.0, 0.0),
        nalgebra::Vector3::new(0.0, 0.0, 0.0),
    );

    let n = 1_000;
    let mut latencies = Vec::with_capacity(n);

    for _ in 0..n {
        let t0 = Instant::now();
        let result = servo.send_twist(&twist);
        let elapsed = t0.elapsed();
        latencies.push(elapsed);

        // Apply the command to update state (if successful)
        if let Ok(cmd) = &result {
            let _ = servo.set_state(&cmd.positions, &cmd.velocities);
        }
    }

    latencies.sort();
    let mean_us = latencies.iter().map(|d| d.as_micros()).sum::<u128>() / n as u128;
    let p50 = latencies[n / 2].as_micros();
    let p99 = latencies[n * 99 / 100].as_micros();
    let max_us = latencies.last().unwrap().as_micros();

    eprintln!("Servo loop ({n} iters): mean={mean_us}μs, p50={p50}μs, p99={p99}μs, max={max_us}μs");

    // Debug build thresholds
    assert!(
        mean_us < 10_000,
        "Servo mean should be <10ms, got {mean_us}μs"
    );
    assert!(p99 < 50_000, "Servo p99 should be <50ms, got {p99}μs");
}

// ─── Servo sustained with scene obstacles ───────────────────────────────────

#[test]
fn servo_sustained_loop_with_scene() {
    let robot = Arc::new(Robot::from_name("ur5e").unwrap());
    let mut scene_mut = Scene::new(&robot).unwrap();

    // Add a few small obstacles for realistic collision checking
    scene_mut.add(
        "box1",
        Shape::cuboid(0.05, 0.05, 0.05),
        Isometry3::translation(0.4, 0.0, 0.3),
    );
    scene_mut.add(
        "box2",
        Shape::cuboid(0.03, 0.03, 0.03),
        Isometry3::translation(-0.3, 0.2, 0.4),
    );
    scene_mut.add(
        "sphere1",
        Shape::sphere(0.05),
        Isometry3::translation(0.0, 0.3, 0.5),
    );

    let scene = Arc::new(scene_mut);

    let config = kinetic::reactive::ServoConfig {
        rate_hz: 500.0,
        collision_check_hz: 100.0,
        ..Default::default()
    };

    let mut servo = kinetic::reactive::Servo::new(&robot, &scene, config).unwrap();

    let twist = Twist::new(
        nalgebra::Vector3::new(0.005, 0.002, 0.0),
        nalgebra::Vector3::new(0.0, 0.0, 0.001),
    );

    let n = 10_000;
    let mut latencies = Vec::with_capacity(n);

    for _ in 0..n {
        let t0 = Instant::now();
        let result = servo.send_twist(&twist);
        let elapsed = t0.elapsed();
        latencies.push(elapsed);

        if let Ok(cmd) = &result {
            let _ = servo.set_state(&cmd.positions, &cmd.velocities);
        }
    }

    latencies.sort();
    let mean_us = latencies.iter().map(|d| d.as_micros()).sum::<u128>() / n as u128;
    let p50 = latencies[n / 2].as_micros();
    let p95 = latencies[n * 95 / 100].as_micros();
    let p99 = latencies[n * 99 / 100].as_micros();
    let max_us = latencies.last().unwrap().as_micros();

    eprintln!(
        "Servo sustained ({n} iters, 3 obstacles): \
         mean={mean_us}μs, p50={p50}μs, p95={p95}μs, p99={p99}μs, max={max_us}μs"
    );

    // With obstacles, collision checks add latency on some ticks
    assert!(
        mean_us < 20_000,
        "Servo+scene mean should be <20ms, got {mean_us}μs"
    );
    assert!(
        p99 < 100_000,
        "Servo+scene p99 should be <100ms, got {p99}μs"
    );
}
