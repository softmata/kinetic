# GPU Optimization

## What You'll Learn
- Configure GPU-accelerated trajectory optimization with `GpuConfig` presets
- Build an obstacle environment from sphere primitives
- Run parallel-seed optimization on the GPU via `GpuOptimizer`
- Handle GPU unavailability with graceful CPU fallback
- Interpret optimization results (goal error, trajectory cost, timing)

## Prerequisites
- [Motion Planning](../../concepts/motion-planning.md)
- [Collision Detection](../../concepts/collision-detection.md)
- [Planning Basics](planning-basics.md)

## Overview

Kinetic's GPU trajectory optimizer uses wgpu compute shaders to evaluate many
trajectory candidates simultaneously. Instead of the sequential sample-and-extend
approach of RRT, it initializes dozens to hundreds of trajectory seeds, then
applies parallel gradient descent to minimize a cost function that balances
collision avoidance, smoothness, and goal reaching. This tutorial configures the
optimizer, builds an obstacle environment, runs optimization, and inspects the
result.

## Step 1: Load Robot and Define Start/Goal

```rust
use kinetic::collision::SpheresSoA;
use kinetic::gpu::{GpuConfig, GpuOptimizer};
use kinetic::prelude::*;

fn main() -> kinetic::core::Result<()> {
    let robot = Robot::from_name("franka_panda")?;
    println!("Robot: {} ({} DOF)", robot.name, robot.dof);

    let start: Vec<f64> = robot.joint_limits.iter()
        .map(|l| (l.lower + l.upper) / 2.0)
        .collect();
    let goal: Vec<f64> = robot.joint_limits.iter()
        .map(|l| l.lower + (l.upper - l.lower) * 0.75)
        .collect();
```

**What this does:** Loads a 7-DOF Franka Panda and defines start (mid-range of each joint) and goal (75th percentile of each joint's range). Both configurations are guaranteed to be within joint limits.

**Why:** GPU optimization works on any robot kinetic supports. The Panda's 7-DOF redundancy makes it a good test case because there are many possible trajectories between any two configurations. Starting at mid-range ensures a well-conditioned initial configuration.

## Step 2: Build the Obstacle Environment

```rust
    let mut obstacles = SpheresSoA::new();

    // Table surface: grid of spheres at z=0.4
    for i in 0..10 {
        for j in 0..10 {
            let x = 0.2 + i as f64 * 0.06;
            let y = -0.3 + j as f64 * 0.06;
            obstacles.push(x, y, 0.4, 0.03, 0);
        }
    }

    // Pillar obstacle
    for k in 0..5 {
        obstacles.push(0.4, 0.2, 0.5 + k as f64 * 0.05, 0.04, 1);
    }

    println!("Environment: {} obstacle spheres", obstacles.len());
```

**What this does:** Builds a table surface from a 10x10 grid of small spheres at z=0.4, plus a vertical pillar of 5 spheres. Each `push` call adds a sphere with `(x, y, z, radius, group_id)`.

**Why:** The GPU optimizer uses a Signed Distance Field (SDF) constructed from obstacle spheres. The SDF is voxelized on the GPU at the resolution specified by `GpuConfig::sdf_resolution`. A denser sphere environment creates a more detailed SDF, producing smoother collision gradients for the optimizer to follow.

## Step 3: Configure the GPU Optimizer

```rust
    let config = GpuConfig {
        num_seeds: 64,          // parallel trajectory candidates
        timesteps: 32,          // waypoints per trajectory
        iterations: 50,         // gradient descent iterations
        collision_weight: 100.0,
        smoothness_weight: 1.0,
        goal_weight: 50.0,
        step_size: 0.01,
        sdf_resolution: 0.03,
        workspace_bounds: [-1.0, -1.0, -0.5, 1.0, 1.0, 1.5],
        seed_perturbation: 0.3,
        warm_start: None,
    };
```

**What this does:** Configures the optimizer to run 64 trajectory seeds in parallel, each with 32 waypoints, for 50 gradient descent iterations. The cost function weights collision avoidance at 100x, smoothness at 1x, and goal reaching at 50x.

**Why:** The trade-offs are:
- **`num_seeds`**: More seeds explore more of configuration space but use more GPU memory. 64 is a good balance; use 512 for quality-critical offline planning.
- **`timesteps`**: More waypoints produce smoother trajectories. 32 is standard; 48 for complex environments.
- **`iterations`**: More iterations improve convergence. 50 is fast; 200 for high quality.
- **`collision_weight`**: Higher values push harder away from obstacles. Too high causes the optimizer to "flee" obstacles rather than navigate around them.
- **`sdf_resolution`**: Finer resolution (0.01) is more accurate but uses more GPU memory. Coarser (0.05) is faster.

## Step 4: Presets and Initialization

Kinetic provides three presets: `GpuConfig::balanced()` (default), `GpuConfig::speed()` (32 seeds, 24 steps, 30 iters — for real-time replanning), and `GpuConfig::quality()` (512 seeds, 48 steps, 200 iters — for offline planning). Start with a preset and adjust individual parameters only if needed.



```rust
    let optimizer = match GpuOptimizer::new(config) {
        Ok(opt) => opt,
        Err(e) => {
            eprintln!("GPU init failed: {} (needs Vulkan/Metal/DX12)", e);
            return Ok(());  // Fall back to CPU-based RRT planning
        }
    };

    let trajectory = optimizer.optimize(&robot, &obstacles, &start, &goal)?;
```

**What this does:** Initializes wgpu, compiles compute shaders, and runs optimization. Both can fail if no GPU is available.

**Why:** Always handle GPU failure gracefully — CI servers and SSH sessions lack GPU access. Fall back to CPU-based RRT planning for portability.

## Step 5: Inspect Results

```rust
    println!("Trajectory waypoints: {}", trajectory.len());

    if trajectory.len() >= 2 {
        let first = trajectory.waypoint(0);
        let last = trajectory.waypoint(trajectory.len() - 1);

        // Check goal reaching accuracy
        let goal_err: f64 = last.positions.0.iter()
            .zip(goal.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt();
        println!("Goal error (joint-space L2): {:.4} rad", goal_err);
    }

    Ok(())
}
```

**What this does:** Prints the number of trajectory waypoints and computes the L2 distance between the final waypoint and the goal configuration. A goal error below 0.01 rad indicates successful convergence.

**Why:** Unlike RRT which guarantees the exact goal configuration is reached (within tolerance), GPU optimization produces a trajectory that *approaches* the goal as determined by the `goal_weight`. Higher `goal_weight` forces tighter goal reaching at the possible cost of less smooth trajectories. Always check `goal_err` to verify the trajectory is usable.

## Complete Code

See `examples/gpu_optimize.rs` for the full listing combining all steps above.

## What You Learned
- `GpuConfig` controls seeds, timesteps, iterations, cost weights, and SDF resolution
- Three presets: `balanced()` (default), `speed()` (real-time), `quality()` (offline)
- `GpuOptimizer::new(config)` initializes wgpu and compiles compute shaders
- `optimizer.optimize(&robot, &obstacles, &start, &goal)` runs parallel optimization
- Always handle GPU initialization failure for portability
- Goal error should be checked — GPU optimization approaches the goal, not exact
- Obstacle environments are built from `SpheresSoA` which becomes a GPU-side SDF

## Try This
- Compare `GpuConfig::speed()` vs `GpuConfig::quality()` on the same problem — measure planning time and goal error
- Increase `num_seeds` to 256 and observe whether goal error decreases
- Set `warm_start: Some(initial_trajectory)` from a previous RRT solution to seed the optimizer with a known-good path
- Remove all obstacles and verify the optimizer produces a near-straight-line trajectory
- Try `GpuConfig { collision_weight: 0.0, .. }` to disable collision checking and see the difference in trajectory shape

## Next
- [Planning Basics](planning-basics.md) — CPU-based RRT planning for comparison
- [Pick and Place](pick-and-place.md) — using optimized trajectories in a full pipeline
