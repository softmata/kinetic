//! GPU trajectory optimization — parallel-seed trajectory planning via wgpu.
//!
//! Demonstrates GPU-accelerated trajectory optimization:
//! - Load a robot and create obstacle environment
//! - Configure the GPU optimizer with parallel seeds
//! - Run trajectory optimization on the GPU
//! - Print the best trajectory cost and planning time
//!
//! Requires a wgpu-compatible GPU (Vulkan, Metal, or DX12).
//!
//! ```sh
//! cargo run --example gpu_optimize -p kinetic
//! ```

use kinetic::collision::SpheresSoA;
use kinetic::gpu::{GpuConfig, GpuOptimizer};
use kinetic::prelude::*;

fn main() -> kinetic::core::Result<()> {
    println!("=== KINETIC GPU Trajectory Optimization ===\n");

    // 1. Load robot
    let robot = Robot::from_name("franka_panda")?;
    println!("Robot: {} ({} DOF)", robot.name, robot.dof);

    // 2. Define start and goal configurations (within joint limits)
    let start: Vec<f64> = robot
        .joint_limits
        .iter()
        .map(|l| (l.lower + l.upper) / 2.0)
        .collect();
    let goal: Vec<f64> = robot
        .joint_limits
        .iter()
        .map(|l| l.lower + (l.upper - l.lower) * 0.75)
        .collect();

    print!("Start: [");
    for (i, v) in start.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{:.3}", v);
    }
    println!("]");

    print!("Goal:  [");
    for (i, v) in goal.iter().enumerate() {
        if i > 0 {
            print!(", ");
        }
        print!("{:.3}", v);
    }
    println!("]\n");

    // 3. Create obstacle environment (table + two pillars)
    let mut obstacles = SpheresSoA::new();

    // Table surface: line of spheres at z=0.4
    for i in 0..10 {
        for j in 0..10 {
            let x = 0.2 + i as f64 * 0.06;
            let y = -0.3 + j as f64 * 0.06;
            obstacles.push(x, y, 0.4, 0.03, 0);
        }
    }

    // Pillar at (0.4, 0.2, 0.6)
    for k in 0..5 {
        obstacles.push(0.4, 0.2, 0.5 + k as f64 * 0.05, 0.04, 1);
    }

    println!("Environment: {} obstacle spheres", obstacles.len());

    // 4. Configure GPU optimizer
    let config = GpuConfig {
        num_seeds: 64,  // parallel trajectory candidates
        timesteps: 32,  // waypoints per trajectory
        iterations: 50, // gradient descent iterations
        collision_weight: 100.0,
        smoothness_weight: 1.0,
        goal_weight: 50.0,
        step_size: 0.01,
        sdf_resolution: 0.03,
        workspace_bounds: [-1.0, -1.0, -0.5, 1.0, 1.0, 1.5],
        seed_perturbation: 0.3,
        warm_start: None,
    };

    println!(
        "GPU config: {} seeds, {} timesteps, {} iterations",
        config.num_seeds, config.timesteps, config.iterations
    );

    // 5. Initialize GPU and run optimization
    println!("\nInitializing GPU...");
    let start_time = std::time::Instant::now();

    let optimizer = match GpuOptimizer::new(config) {
        Ok(opt) => opt,
        Err(e) => {
            eprintln!("GPU initialization failed: {}", e);
            eprintln!("This example requires a wgpu-compatible GPU (Vulkan, Metal, or DX12).");
            eprintln!("If running in CI or a headless environment, this is expected.");
            return Ok(());
        }
    };

    let init_time = start_time.elapsed();
    println!("GPU initialized in {:?}", init_time);

    println!("Optimizing trajectory...");
    let opt_start = std::time::Instant::now();

    let trajectory = match optimizer.optimize(&robot, &obstacles, &start, &goal) {
        Ok(traj) => traj,
        Err(e) => {
            eprintln!("GPU optimization failed: {}", e);
            return Ok(());
        }
    };

    let opt_time = opt_start.elapsed();

    // 6. Print results
    println!("\n=== Results ===");
    println!("Planning time: {:?}", opt_time);
    println!("Seeds evaluated: {}", optimizer.config().num_seeds);
    println!("Trajectory waypoints: {}", trajectory.len());

    // Show first and last waypoints
    if trajectory.len() >= 2 {
        let first = trajectory.waypoint(0);
        print!("First waypoint: [");
        for (i, v) in first.positions.0.iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{:.3}", v);
        }
        println!("]");

        let last = trajectory.waypoint(trajectory.len() - 1);
        print!("Last waypoint:  [");
        for (i, v) in last.positions.0.iter().enumerate() {
            if i > 0 {
                print!(", ");
            }
            print!("{:.3}", v);
        }
        println!("]");

        // Check goal reaching
        let goal_err: f64 = last
            .positions
            .0
            .iter()
            .zip(goal.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        println!("Goal error (joint-space L2): {:.4} rad", goal_err);
    }

    let total = start_time.elapsed();
    println!("\nTotal time (init + optimize): {:?}", total);

    Ok(())
}
