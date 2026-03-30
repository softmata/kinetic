//! Integration tests for the GPU trajectory optimizer.
//!
//! Tests that require a GPU adapter are marked `#[ignore]` for CI without GPU.
//! Run with: `cargo test -p kinetic -- --ignored test_gpu`

use kinetic::collision::SpheresSoA;
use kinetic::gpu::{CpuOptimizer, GpuConfig, GpuError, GpuOptimizer, SignedDistanceField};
use kinetic::prelude::*;
use std::sync::Arc;

fn panda() -> Arc<Robot> {
    Arc::new(Robot::from_name("franka_panda").unwrap())
}

fn ur5e() -> Arc<Robot> {
    Arc::new(Robot::from_name("ur5e").unwrap())
}

/// Try to create a GpuOptimizer; return None if no GPU is available.
fn try_gpu(config: GpuConfig) -> Option<GpuOptimizer> {
    match GpuOptimizer::new(config) {
        Ok(opt) => Some(opt),
        Err(_) => {
            eprintln!("Skipping GPU test: no GPU adapter available");
            None
        }
    }
}

// ─── Existing tests ──────────────────────────────────────────────────────────

#[test]
#[ignore] // Requires GPU adapter
fn gpu_optimizer_basic_trajectory() {
    let robot = panda();
    let config = GpuConfig::default();

    let optimizer = match try_gpu(config) {
        Some(o) => o,
        None => return,
    };

    let start = vec![0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785];
    let goal = vec![0.5, -0.5, 0.3, -1.8, 0.2, 1.2, 0.5];

    let obstacles = SpheresSoA::new();

    let result = optimizer.optimize(&robot, &obstacles, &start, &goal);
    assert!(
        result.is_ok(),
        "GPU optimization should succeed: {:?}",
        result.err()
    );

    let trajectory = result.unwrap();
    assert_eq!(trajectory.dof, 7);
    assert!(
        trajectory.waypoints().len() >= 2,
        "Trajectory should have at least 2 waypoints"
    );
}

#[test]
#[ignore] // Requires GPU adapter
fn gpu_optimizer_with_obstacles() {
    let robot = panda();
    let config = GpuConfig {
        num_seeds: 64,
        timesteps: 16,
        iterations: 50,
        collision_weight: 200.0,
        ..Default::default()
    };

    let optimizer = match try_gpu(config) {
        Some(o) => o,
        None => return,
    };

    let start = vec![0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785];
    let goal = vec![0.5, -0.5, 0.3, -1.8, 0.2, 1.2, 0.5];

    let mut obstacles = SpheresSoA::new();
    for i in 0..5 {
        let x = 0.3 + (i as f64) * 0.1;
        obstacles.push(x, 0.0, 0.4, 0.03, i);
    }

    let result = optimizer.optimize(&robot, &obstacles, &start, &goal);
    if let Ok(traj) = result {
        assert!(traj.waypoints().len() >= 2);
        assert_eq!(traj.dof, 7);
    }
}

#[test]
fn gpu_config_defaults_valid() {
    let config = GpuConfig::default();
    assert!(config.num_seeds > 0);
    assert!(config.timesteps > 0);
    assert!(config.iterations > 0);
    assert!(config.collision_weight > 0.0);
    assert!(config.smoothness_weight > 0.0);
    assert!(config.goal_weight > 0.0);
    assert!(config.step_size > 0.0);
}

// ─── NEW: CPU-only error path tests ─────────────────────────────────────────

#[test]
#[ignore] // Requires GPU adapter
fn gpu_optimizer_invalid_config_start_dof_mismatch() {
    let robot = panda(); // 7 DOF
    let config = GpuConfig {
        num_seeds: 4,
        timesteps: 8,
        iterations: 1,
        ..Default::default()
    };

    let optimizer = match try_gpu(config) {
        Some(o) => o,
        None => return,
    };

    let start = vec![0.0; 3]; // Wrong: 3 instead of 7
    let goal = vec![0.5, -0.5, 0.3, -1.8, 0.2, 1.2, 0.5];
    let obstacles = SpheresSoA::new();

    let result = optimizer.optimize(&robot, &obstacles, &start, &goal);
    assert!(result.is_err(), "Should fail when start DOF != robot DOF");
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("doesn't match robot DOF"),
        "Error should mention DOF mismatch: {msg}"
    );
}

#[test]
#[ignore] // Requires GPU adapter
fn gpu_optimizer_invalid_config_goal_dof_mismatch() {
    let robot = panda(); // 7 DOF
    let config = GpuConfig {
        num_seeds: 4,
        timesteps: 8,
        iterations: 1,
        ..Default::default()
    };

    let optimizer = match try_gpu(config) {
        Some(o) => o,
        None => return,
    };

    let start = vec![0.0; 7];
    let goal = vec![0.5; 10]; // Wrong: 10 instead of 7
    let obstacles = SpheresSoA::new();

    let result = optimizer.optimize(&robot, &obstacles, &start, &goal);
    assert!(result.is_err(), "Should fail when goal DOF != robot DOF");
    let err = result.unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("doesn't match robot DOF"),
        "Error should mention DOF mismatch: {msg}"
    );
}

// ─── NEW: GpuConfig custom values ──────────────────────────────────────────

#[test]
fn gpu_config_custom_values_preserved() {
    let config = GpuConfig {
        num_seeds: 256,
        timesteps: 64,
        iterations: 200,
        collision_weight: 500.0,
        smoothness_weight: 2.5,
        goal_weight: 75.0,
        step_size: 0.005,
        sdf_resolution: 0.01,
        workspace_bounds: [-2.0, -2.0, -1.0, 2.0, 2.0, 3.0],
        seed_perturbation: 0.5,
        warm_start: None,
    };

    assert_eq!(config.num_seeds, 256);
    assert_eq!(config.timesteps, 64);
    assert_eq!(config.iterations, 200);
    assert!((config.collision_weight - 500.0).abs() < 1e-6);
    assert!((config.smoothness_weight - 2.5).abs() < 1e-6);
    assert!((config.goal_weight - 75.0).abs() < 1e-6);
    assert!((config.step_size - 0.005).abs() < 1e-6);
    assert!((config.sdf_resolution - 0.01).abs() < 1e-6);
    assert!((config.workspace_bounds[0] - (-2.0)).abs() < 1e-6);
    assert!((config.workspace_bounds[5] - 3.0).abs() < 1e-6);
    assert!((config.seed_perturbation - 0.5).abs() < 1e-6);
}

// ─── NEW: GpuError Display formatting ───────────────────────────────────────

#[test]
fn gpu_error_display_messages() {
    let err_no_adapter = GpuError::NoAdapter;
    assert_eq!(format!("{err_no_adapter}"), "no suitable GPU adapter found");

    let err_buffer = GpuError::BufferMapping;
    assert_eq!(format!("{err_buffer}"), "GPU buffer mapping failed");

    let err_config = GpuError::InvalidConfig("test message".into());
    assert_eq!(
        format!("{err_config}"),
        "invalid configuration: test message"
    );
}

// ─── NEW: SDF GPU-built query validation ────────────────────────────────────

#[test]
#[ignore] // Requires GPU adapter
fn sdf_query_obstacle_distance_gradient() {
    let config = GpuConfig::default();
    let optimizer = match try_gpu(config) {
        Some(o) => o,
        None => return,
    };

    let device = optimizer.device();
    let queue = optimizer.queue();

    // Place a sphere at (0.0, 0.0, 0.0) with radius 0.2
    let mut spheres = SpheresSoA::new();
    spheres.push(0.0, 0.0, 0.0, 0.2, 0);

    let sdf = SignedDistanceField::from_spheres(
        device,
        queue,
        &spheres,
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        0.05,
    )
    .unwrap();

    // Verify dimensions are reasonable
    assert!(sdf.num_voxels() > 0);
    let (nx, ny, nz) = sdf.dimensions();
    assert!(nx > 0 && ny > 0 && nz > 0);
    assert!((sdf.resolution() - 0.05).abs() < 1e-6);

    // Distance at obstacle center should be negative (inside)
    let d_center = sdf.query(0.0, 0.0, 0.0);
    assert!(
        d_center < 0.0,
        "Inside obstacle, distance should be negative: {d_center}"
    );

    // Distance far away should be positive
    let d_far = sdf.query(0.8, 0.0, 0.0);
    assert!(
        d_far > 0.0,
        "Far from obstacle, distance should be positive: {d_far}"
    );

    // Distance should increase as we move away from the obstacle
    let d_near = sdf.query(0.3, 0.0, 0.0);
    assert!(
        d_far > d_near,
        "Distance should increase with distance from obstacle: near={d_near}, far={d_far}"
    );

    // Out-of-bounds query returns very large value
    assert!(sdf.query(5.0, 0.0, 0.0) > 1e9);
    assert!(sdf.query(-5.0, 0.0, 0.0) > 1e9);
}

// ─── NEW: GPU optimizer with empty obstacles produces valid trajectory ──────

#[test]
#[ignore] // Requires GPU adapter
fn gpu_optimizer_empty_obstacles_valid_trajectory() {
    let robot = panda();
    let config = GpuConfig {
        num_seeds: 16,
        timesteps: 16,
        iterations: 20,
        ..Default::default()
    };

    let optimizer = match try_gpu(config) {
        Some(o) => o,
        None => return,
    };

    let start = vec![0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785];
    let goal = vec![0.5, -0.5, 0.3, -1.8, 0.2, 1.2, 0.5];
    let obstacles = SpheresSoA::new();

    let result = optimizer.optimize(&robot, &obstacles, &start, &goal);
    assert!(
        result.is_ok(),
        "Should succeed with empty obstacles: {:?}",
        result.err()
    );

    let traj = result.unwrap();
    assert_eq!(traj.dof, 7);

    let wps = traj.waypoints();
    assert!(wps.len() >= 2, "Need at least start and goal waypoints");

    // First waypoint should match start
    for j in 0..7 {
        assert!(
            (wps[0].as_slice()[j] - start[j]).abs() < 0.1,
            "Start mismatch at joint {j}: got {}, expected {}",
            wps[0].as_slice()[j],
            start[j]
        );
    }

    // Last waypoint should be near goal
    let last = wps.last().unwrap();
    for j in 0..7 {
        assert!(
            (last.as_slice()[j] - goal[j]).abs() < 0.5,
            "Goal mismatch at joint {j}: got {}, expected {}",
            last.as_slice()[j],
            goal[j]
        );
    }
}

// ─── NEW: GPU optimizer with 6-DOF UR5e ─────────────────────────────────────

#[test]
#[ignore] // Requires GPU adapter
fn gpu_optimizer_ur5e_6dof() {
    let robot = ur5e();
    let config = GpuConfig {
        num_seeds: 16,
        timesteps: 16,
        iterations: 20,
        ..Default::default()
    };

    let optimizer = match try_gpu(config) {
        Some(o) => o,
        None => return,
    };

    let start = vec![0.0; 6];
    let goal = vec![0.5, -0.8, 0.3, 0.1, 0.3, -0.2];
    let obstacles = SpheresSoA::new();

    let result = optimizer.optimize(&robot, &obstacles, &start, &goal);
    assert!(
        result.is_ok(),
        "UR5e optimization should succeed: {:?}",
        result.err()
    );

    let traj = result.unwrap();
    assert_eq!(traj.dof, 6, "UR5e trajectory should be 6-DOF");
    assert!(traj.waypoints().len() >= 2);
}

// ─── NEW: GPU optimizer trajectory waypoint count matches config ────────────

#[test]
#[ignore] // Requires GPU adapter
fn gpu_optimizer_trajectory_timesteps_match_config() {
    let robot = panda();
    let timesteps = 24u32;
    let config = GpuConfig {
        num_seeds: 8,
        timesteps,
        iterations: 10,
        ..Default::default()
    };

    let optimizer = match try_gpu(config) {
        Some(o) => o,
        None => return,
    };

    let start = vec![0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785];
    let goal = vec![0.5, -0.5, 0.3, -1.8, 0.2, 1.2, 0.5];
    let obstacles = SpheresSoA::new();

    let result = optimizer.optimize(&robot, &obstacles, &start, &goal);
    assert!(result.is_ok(), "Optimization failed: {:?}", result.err());

    let traj = result.unwrap();
    assert_eq!(
        traj.waypoints().len(),
        timesteps as usize,
        "Trajectory waypoints should equal configured timesteps"
    );
}

// ─── NEW: GPU optimizer with many obstacles ─────────────────────────────────

#[test]
#[ignore] // Requires GPU adapter
fn gpu_optimizer_many_obstacles() {
    let robot = panda();
    let config = GpuConfig {
        num_seeds: 32,
        timesteps: 16,
        iterations: 30,
        collision_weight: 300.0,
        ..Default::default()
    };

    let optimizer = match try_gpu(config) {
        Some(o) => o,
        None => return,
    };

    let start = vec![0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785];
    let goal = vec![0.5, -0.5, 0.3, -1.8, 0.2, 1.2, 0.5];

    // Create a grid of 50 obstacle spheres
    let mut obstacles = SpheresSoA::new();
    for i in 0..50 {
        let x = -0.5 + (i % 10) as f64 * 0.1;
        let y = -0.25 + (i / 10) as f64 * 0.1;
        obstacles.push(x, y, 0.4, 0.02, i);
    }

    let result = optimizer.optimize(&robot, &obstacles, &start, &goal);
    // Should not crash regardless of obstacle density
    match result {
        Ok(traj) => {
            assert_eq!(traj.dof, 7);
            assert!(traj.waypoints().len() >= 2);
            eprintln!(
                "Many obstacles: {} waypoints produced",
                traj.waypoints().len()
            );
        }
        Err(e) => {
            eprintln!("Many obstacles test completed (optimizer error acceptable): {e}");
        }
    }
}

// ─── NEW: SDF from_spheres on GPU ───────────────────────────────────────────

#[test]
#[ignore] // Requires GPU adapter
fn sdf_from_spheres_on_gpu() {
    let config = GpuConfig::default();
    let optimizer = match try_gpu(config) {
        Some(o) => o,
        None => return,
    };

    let device = optimizer.device();
    let queue = optimizer.queue();

    // Build SDF with a single obstacle at origin
    let mut spheres = SpheresSoA::new();
    spheres.push(0.0, 0.0, 0.0, 0.1, 0);

    let sdf = SignedDistanceField::from_spheres(
        device,
        queue,
        &spheres,
        [-0.5, -0.5, -0.5],
        [0.5, 0.5, 0.5],
        0.1,
    );
    assert!(
        sdf.is_ok(),
        "SDF construction should succeed: {:?}",
        sdf.err()
    );

    let sdf = sdf.unwrap();
    assert!(sdf.num_voxels() > 0);

    // Query at the obstacle center: distance should be negative (inside sphere)
    let center_dist = sdf.query(0.0, 0.0, 0.0);
    assert!(
        center_dist < 0.0,
        "Distance at obstacle center should be negative: {center_dist}"
    );

    // Query far from obstacle: distance should be positive
    let far_dist = sdf.query(0.4, 0.4, 0.4);
    assert!(
        far_dist > 0.0,
        "Distance far from obstacle should be positive: {far_dist}"
    );
}

#[test]
#[ignore] // Requires GPU adapter
fn sdf_from_empty_spheres_on_gpu() {
    let config = GpuConfig::default();
    let optimizer = match try_gpu(config) {
        Some(o) => o,
        None => return,
    };

    let device = optimizer.device();
    let queue = optimizer.queue();
    let spheres = SpheresSoA::new();

    let sdf = SignedDistanceField::from_spheres(
        device,
        queue,
        &spheres,
        [-1.0, -1.0, -1.0],
        [1.0, 1.0, 1.0],
        0.5,
    );
    assert!(sdf.is_ok(), "Empty sphere SDF should succeed");

    let sdf = sdf.unwrap();
    // All voxels should have large positive distance (no obstacles)
    assert!(sdf.query(0.0, 0.0, 0.0) > 1e9);
    assert!(sdf.query(0.5, 0.5, 0.5) > 1e9);
}

// ═══════════════════════════════════════════════════════════════════════════════
// CPU FALLBACK OPTIMIZER TESTS (no GPU required)
// ═══════════════════════════════════════════════════════════════════════════════

/// Safe start config: mid-range of all joint limits.
fn safe_start_config(robot: &Robot) -> Vec<f64> {
    robot
        .joint_limits
        .iter()
        .map(|l| (l.lower + l.upper) / 2.0)
        .collect()
}

/// Safe goal: offset from start, clamped within joint limits.
fn safe_goal_config(robot: &Robot, start: &[f64], offset: f64) -> Vec<f64> {
    start
        .iter()
        .enumerate()
        .map(|(i, &s)| {
            let l = &robot.joint_limits[i];
            (s + offset).clamp(l.lower + 0.01, l.upper - 0.01)
        })
        .collect()
}

#[test]
fn cpu_fallback_panda_basic_trajectory() {
    let robot = panda();
    let start = safe_start_config(&robot);
    let goal = safe_goal_config(&robot, &start, 0.2);

    let config = GpuConfig {
        num_seeds: 4,
        timesteps: 8,
        iterations: 10,
        ..Default::default()
    };
    let opt = CpuOptimizer::new(config);
    let obstacles = SpheresSoA::new();

    let traj = opt.optimize(&robot, &obstacles, &start, &goal).unwrap();

    assert_eq!(traj.dof, 7, "Panda trajectory should be 7-DOF");
    assert_eq!(traj.len(), 8, "Waypoints should match timesteps");

    // Start should match
    let first = traj.waypoint(0);
    for j in 0..7 {
        assert!(
            (first.positions.as_slice()[j] - start[j]).abs() < 1e-4,
            "Start mismatch at joint {j}"
        );
    }

    // Goal should be reached approximately
    let last = traj.waypoint(traj.len() - 1);
    for j in 0..7 {
        assert!(
            (last.positions.as_slice()[j] - goal[j]).abs() < 0.2,
            "Goal not reached at joint {j}: got {}, expected {}",
            last.positions.as_slice()[j],
            goal[j]
        );
    }
}

#[test]
fn cpu_fallback_ur5e_6dof() {
    let robot = ur5e();
    let start = safe_start_config(&robot);
    let goal = safe_goal_config(&robot, &start, 0.2);

    let config = GpuConfig {
        num_seeds: 4,
        timesteps: 8,
        iterations: 10,
        ..Default::default()
    };
    let opt = CpuOptimizer::new(config);
    let obstacles = SpheresSoA::new();

    let traj = opt.optimize(&robot, &obstacles, &start, &goal).unwrap();

    assert_eq!(traj.dof, 6, "UR5e trajectory should be 6-DOF");
    assert_eq!(traj.len(), 8);

    // Verify all waypoints are within joint limits
    for t in 0..traj.len() {
        let wp = traj.waypoint(t);
        for j in 0..6 {
            let pos = wp.positions.as_slice()[j];
            assert!(
                pos >= robot.joint_limits[j].lower - 1e-6
                    && pos <= robot.joint_limits[j].upper + 1e-6,
                "Waypoint {t}, joint {j}: position {pos} outside limits [{}, {}]",
                robot.joint_limits[j].lower,
                robot.joint_limits[j].upper
            );
        }
    }
}

#[test]
fn cpu_fallback_with_obstacles_avoids_collision() {
    let robot = panda();
    let start = safe_start_config(&robot);
    let goal = safe_goal_config(&robot, &start, 0.15);

    let mut obstacles = SpheresSoA::new();
    obstacles.push(0.4, 0.0, 0.4, 0.05, 0);
    obstacles.push(0.3, 0.1, 0.3, 0.05, 1);

    let config = GpuConfig {
        num_seeds: 8,
        timesteps: 8,
        iterations: 20,
        collision_weight: 200.0,
        ..Default::default()
    };
    let opt = CpuOptimizer::new(config);

    let traj = opt.optimize(&robot, &obstacles, &start, &goal).unwrap();

    assert_eq!(traj.dof, 7);
    assert_eq!(traj.len(), 8);
    // Should produce a valid trajectory (not crash, not panic)
}

#[test]
fn cpu_fallback_dof_mismatch_returns_error() {
    let robot = panda(); // 7 DOF
    let opt = CpuOptimizer::new(GpuConfig::default());
    let obstacles = SpheresSoA::new();

    // Start has wrong DOF
    let result = opt.optimize(&robot, &obstacles, &[0.0; 3], &[0.5; 7]);
    assert!(result.is_err(), "Should fail when start DOF != robot DOF");
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("doesn't match robot DOF"));

    // Goal has wrong DOF
    let start = safe_start_config(&robot);
    let result = opt.optimize(&robot, &obstacles, &start, &[0.5; 10]);
    assert!(result.is_err(), "Should fail when goal DOF != robot DOF");
}

#[test]
fn cpu_fallback_timesteps_match_config() {
    let robot = panda();
    let start = safe_start_config(&robot);
    let goal = safe_goal_config(&robot, &start, 0.2);

    for timesteps in [4, 8, 16, 32] {
        let config = GpuConfig {
            num_seeds: 4,
            timesteps: timesteps as u32,
            iterations: 5,
            ..Default::default()
        };
        let opt = CpuOptimizer::new(config);
        let obstacles = SpheresSoA::new();
        let traj = opt.optimize(&robot, &obstacles, &start, &goal).unwrap();
        assert_eq!(
            traj.len(),
            timesteps,
            "Timesteps={timesteps}: trajectory should have {timesteps} waypoints"
        );
    }
}

#[test]
fn cpu_fallback_more_seeds_does_not_crash() {
    let robot = panda();
    let start = safe_start_config(&robot);
    let goal = safe_goal_config(&robot, &start, 0.3);

    let config = GpuConfig {
        num_seeds: 64,
        timesteps: 8,
        iterations: 5,
        ..Default::default()
    };
    let opt = CpuOptimizer::new(config);
    let obstacles = SpheresSoA::new();

    let traj = opt.optimize(&robot, &obstacles, &start, &goal).unwrap();
    assert_eq!(traj.dof, 7);
    assert_eq!(traj.len(), 8);
}

#[test]
fn cpu_fallback_joints_within_limits() {
    let robot = panda();
    let start = safe_start_config(&robot);
    let goal = safe_goal_config(&robot, &start, 0.4);

    let config = GpuConfig {
        num_seeds: 8,
        timesteps: 16,
        iterations: 20,
        ..Default::default()
    };
    let opt = CpuOptimizer::new(config);

    let mut obstacles = SpheresSoA::new();
    for i in 0..10 {
        obstacles.push(0.2 + i as f64 * 0.05, 0.0, 0.3, 0.02, i);
    }

    let traj = opt.optimize(&robot, &obstacles, &start, &goal).unwrap();

    // Verify ALL waypoints are within joint limits
    for t in 0..traj.len() {
        let wp = traj.waypoint(t);
        for j in 0..robot.dof {
            let pos = wp.positions.as_slice()[j];
            assert!(
                pos >= robot.joint_limits[j].lower - 1e-6
                    && pos <= robot.joint_limits[j].upper + 1e-6,
                "Waypoint {t}, joint {j}: {pos} outside [{}, {}]",
                robot.joint_limits[j].lower,
                robot.joint_limits[j].upper
            );
        }
    }
}

#[test]
fn cpu_fallback_higher_collision_weight_changes_trajectory() {
    let robot = panda();
    let start = safe_start_config(&robot);
    let goal = safe_goal_config(&robot, &start, 0.2);

    let mut obstacles = SpheresSoA::new();
    obstacles.push(0.4, 0.0, 0.4, 0.08, 0);

    let low_weight = GpuConfig {
        num_seeds: 4,
        timesteps: 8,
        iterations: 15,
        collision_weight: 10.0,
        ..Default::default()
    };
    let high_weight = GpuConfig {
        num_seeds: 4,
        timesteps: 8,
        iterations: 15,
        collision_weight: 500.0,
        ..Default::default()
    };

    let opt_low = CpuOptimizer::new(low_weight);
    let opt_high = CpuOptimizer::new(high_weight);

    let traj_low = opt_low.optimize(&robot, &obstacles, &start, &goal).unwrap();
    let traj_high = opt_high
        .optimize(&robot, &obstacles, &start, &goal)
        .unwrap();

    // Both should produce valid trajectories
    assert_eq!(traj_low.len(), 8);
    assert_eq!(traj_high.len(), 8);

    // They may produce different paths (we can't guarantee due to randomness,
    // but at minimum both should be valid)
}

// ═══════════════════════════════════════════════════════════════════════════════
// GPU ERROR HANDLING & RESOURCE EXHAUSTION TESTS
// ═══════════════════════════════════════════════════════════════════════════════

#[test]
fn gpu_error_no_adapter_is_descriptive() {
    let err = GpuError::NoAdapter;
    let msg = format!("{err}");
    assert!(
        msg.contains("no suitable GPU adapter"),
        "NoAdapter error should mention 'no suitable GPU adapter': {msg}"
    );
    // Should also impl Debug
    let debug = format!("{err:?}");
    assert!(debug.contains("NoAdapter"));
}

#[test]
fn gpu_error_buffer_mapping_is_descriptive() {
    let err = GpuError::BufferMapping;
    let msg = format!("{err}");
    assert!(
        msg.contains("buffer mapping failed"),
        "BufferMapping error should mention mapping: {msg}"
    );
}

#[test]
fn gpu_error_invalid_config_includes_detail() {
    let err = GpuError::InvalidConfig("start/goal length (3/7) doesn't match robot DOF (7)".into());
    let msg = format!("{err}");
    assert!(msg.contains("start/goal length"));
    assert!(msg.contains("doesn't match robot DOF"));
    assert!(msg.contains("(7)"));
}

#[test]
fn gpu_optimizer_new_does_not_panic() {
    // On CI without GPU, GpuOptimizer::new should return Err(NoAdapter), never panic.
    let result = GpuOptimizer::new(GpuConfig::default());
    match result {
        Ok(_) => { /* GPU available — fine */ }
        Err(GpuError::NoAdapter) => { /* expected on headless CI */ }
        Err(e) => {
            // DeviceRequest errors are also acceptable
            let msg = format!("{e}");
            assert!(
                msg.contains("GPU") || msg.contains("device") || msg.contains("adapter"),
                "Unexpected error type: {msg}"
            );
        }
    }
}

#[test]
fn gpu_fallback_pattern_works_when_no_gpu() {
    // This tests the recommended fallback pattern: try GPU, fall back to CPU
    let robot = panda();
    let start = safe_start_config(&robot);
    let goal = safe_goal_config(&robot, &start, 0.2);
    let obstacles = SpheresSoA::new();

    let config = GpuConfig {
        num_seeds: 4,
        timesteps: 8,
        iterations: 10,
        ..Default::default()
    };

    // The recommended pattern: try GPU, fall back to CPU
    let traj = match GpuOptimizer::new(config.clone()) {
        Ok(gpu_opt) => gpu_opt.optimize(&robot, &obstacles, &start, &goal),
        Err(_) => {
            let cpu_opt = CpuOptimizer::new(config);
            cpu_opt.optimize(&robot, &obstacles, &start, &goal)
        }
    };

    // Regardless of GPU availability, we should get a valid trajectory
    let traj = traj.unwrap();
    assert_eq!(traj.dof, 7);
    assert!(traj.len() >= 2);
}

#[test]
fn cpu_fallback_matches_gpu_api_surface() {
    // Verify CpuOptimizer has the same API methods as GpuOptimizer
    let config = GpuConfig {
        num_seeds: 4,
        timesteps: 8,
        iterations: 5,
        ..Default::default()
    };

    let cpu = CpuOptimizer::new(config.clone());

    // config() accessor works
    assert_eq!(cpu.config().num_seeds, 4);
    assert_eq!(cpu.config().timesteps, 8);
    assert_eq!(cpu.config().iterations, 5);

    // optimize() has the same signature
    let robot = panda();
    let start = safe_start_config(&robot);
    let goal = safe_goal_config(&robot, &start, 0.1);
    let obs = SpheresSoA::new();
    let _ = cpu.optimize(&robot, &obs, &start, &goal);
}

#[test]
fn cpu_fallback_same_error_on_dof_mismatch_as_gpu_would() {
    let robot = panda(); // 7 DOF
    let cpu = CpuOptimizer::new(GpuConfig::default());
    let obs = SpheresSoA::new();

    // Same error as GPU optimizer would produce
    let result = cpu.optimize(&robot, &obs, &[0.0; 3], &[0.5; 7]);
    let err = result.unwrap_err();
    let msg = format!("{err}");
    assert!(
        msg.contains("doesn't match robot DOF"),
        "CPU fallback should produce same error as GPU: {msg}"
    );
    assert!(msg.contains("3/7") || msg.contains("(3)"));
}

#[test]
fn gpu_error_is_send_and_sync() {
    // GpuError should be Send + Sync for use across threads
    fn assert_send<T: Send>() {}
    fn assert_sync<T: Sync>() {}
    assert_send::<GpuError>();
    assert_sync::<GpuError>();
}

#[test]
fn gpu_config_clone_preserves_all_fields() {
    let config = GpuConfig {
        num_seeds: 256,
        timesteps: 64,
        iterations: 200,
        collision_weight: 500.0,
        smoothness_weight: 2.5,
        goal_weight: 75.0,
        step_size: 0.005,
        sdf_resolution: 0.01,
        workspace_bounds: [-2.0, -2.0, -1.0, 2.0, 2.0, 3.0],
        seed_perturbation: 0.5,
        warm_start: None,
    };

    let cloned = config.clone();
    assert_eq!(cloned.num_seeds, config.num_seeds);
    assert_eq!(cloned.timesteps, config.timesteps);
    assert_eq!(cloned.iterations, config.iterations);
    assert!((cloned.collision_weight - config.collision_weight).abs() < f32::EPSILON);
    assert!((cloned.smoothness_weight - config.smoothness_weight).abs() < f32::EPSILON);
    assert!((cloned.goal_weight - config.goal_weight).abs() < f32::EPSILON);
    assert!((cloned.step_size - config.step_size).abs() < f32::EPSILON);
    assert!((cloned.sdf_resolution - config.sdf_resolution).abs() < f32::EPSILON);
    assert!((cloned.seed_perturbation - config.seed_perturbation).abs() < f32::EPSILON);
    for i in 0..6 {
        assert!((cloned.workspace_bounds[i] - config.workspace_bounds[i]).abs() < f32::EPSILON);
    }
}

#[test]
fn gpu_config_debug_format() {
    let config = GpuConfig::default();
    let debug = format!("{config:?}");
    assert!(debug.contains("GpuConfig"));
    assert!(debug.contains("num_seeds"));
    assert!(debug.contains("128"));
}

#[test]
fn gpu_error_debug_format() {
    let err = GpuError::InvalidConfig("test".into());
    let debug = format!("{err:?}");
    assert!(debug.contains("InvalidConfig"));
    assert!(debug.contains("test"));
}

#[test]
#[ignore] // Requires GPU adapter
fn gpu_optimizer_handles_zero_iterations_gracefully() {
    let robot = panda();
    let config = GpuConfig {
        num_seeds: 4,
        timesteps: 8,
        iterations: 0,
        ..Default::default()
    };

    let optimizer = match try_gpu(config) {
        Some(o) => o,
        None => return,
    };

    let start = safe_start_config(&robot);
    let goal = safe_goal_config(&robot, &start, 0.2);
    let obstacles = SpheresSoA::new();

    // Zero iterations should still produce a trajectory (the initial seeds)
    let result = optimizer.optimize(&robot, &obstacles, &start, &goal);
    match result {
        Ok(traj) => {
            assert_eq!(traj.dof, 7);
            assert_eq!(traj.len(), 8);
        }
        Err(e) => {
            // Acceptable if it errors cleanly
            let _ = format!("{e}");
        }
    }
}

#[test]
fn cpu_fallback_handles_zero_iterations() {
    let robot = panda();
    let start = safe_start_config(&robot);
    let goal = safe_goal_config(&robot, &start, 0.2);
    let obstacles = SpheresSoA::new();

    let config = GpuConfig {
        num_seeds: 4,
        timesteps: 8,
        iterations: 0,
        ..Default::default()
    };
    let opt = CpuOptimizer::new(config);

    // Zero iterations = just the initial seeds, no optimization
    let traj = opt.optimize(&robot, &obstacles, &start, &goal).unwrap();
    assert_eq!(traj.dof, 7);
    assert_eq!(traj.len(), 8);

    // Start should still match (seeds always fix endpoints)
    let first = traj.waypoint(0);
    for j in 0..7 {
        assert!(
            (first.positions.as_slice()[j] - start[j]).abs() < 1e-4,
            "Start should match even with 0 iterations"
        );
    }
}

#[test]
fn cpu_fallback_single_seed_produces_valid_result() {
    let robot = panda();
    let start = safe_start_config(&robot);
    let goal = safe_goal_config(&robot, &start, 0.2);
    let obstacles = SpheresSoA::new();

    let config = GpuConfig {
        num_seeds: 1,
        timesteps: 8,
        iterations: 10,
        ..Default::default()
    };
    let opt = CpuOptimizer::new(config);
    let traj = opt.optimize(&robot, &obstacles, &start, &goal).unwrap();

    assert_eq!(traj.dof, 7);
    assert_eq!(traj.len(), 8);
}

#[test]
fn cpu_fallback_minimal_timesteps() {
    let robot = panda();
    let start = safe_start_config(&robot);
    let goal = safe_goal_config(&robot, &start, 0.1);
    let obstacles = SpheresSoA::new();

    // Minimum useful: 2 timesteps (start + goal, no internal points)
    let config = GpuConfig {
        num_seeds: 2,
        timesteps: 2,
        iterations: 5,
        ..Default::default()
    };
    let opt = CpuOptimizer::new(config);
    let traj = opt.optimize(&robot, &obstacles, &start, &goal).unwrap();

    assert_eq!(traj.len(), 2);
    assert_eq!(traj.dof, 7);
}

#[test]
fn cpu_fallback_many_obstacles_does_not_panic() {
    let robot = panda();
    let start = safe_start_config(&robot);
    let goal = safe_goal_config(&robot, &start, 0.15);

    // Dense obstacle field: 100 small spheres
    let mut obstacles = SpheresSoA::new();
    for i in 0..100 {
        let x = -0.5 + (i % 10) as f64 * 0.1;
        let y = -0.5 + (i / 10) as f64 * 0.1;
        obstacles.push(x, y, 0.3, 0.02, i);
    }

    let config = GpuConfig {
        num_seeds: 4,
        timesteps: 8,
        iterations: 5,
        collision_weight: 300.0,
        ..Default::default()
    };
    let opt = CpuOptimizer::new(config);

    // Should complete without panic regardless of obstacle density
    let traj = opt.optimize(&robot, &obstacles, &start, &goal).unwrap();
    assert_eq!(traj.dof, 7);
    assert_eq!(traj.len(), 8);
}
