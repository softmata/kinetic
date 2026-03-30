# Collision Checking

## What You'll Learn
- Build a `RobotSphereModel` from URDF collision geometry
- Update sphere positions from FK poses
- Check self-collision (with adjacent link filtering)
- Check environment collision against obstacle spheres
- Detect the available SIMD tier and benchmark collision speed

## Prerequisites
- [Collision Detection](../../concepts/collision-detection.md)
- [Forward Kinematics](../../concepts/forward-kinematics.md)

## Overview

Kinetic uses bounding sphere models for fast collision detection. Each robot link
is approximated by a set of spheres generated from its URDF collision geometry.
After each FK update, the sphere positions are transformed to world coordinates,
and collision is checked as sphere-sphere overlap — a simple distance comparison.
SIMD acceleration makes this fast enough for real-time planning loops. This
tutorial builds a sphere model, tests several configurations, and benchmarks
collision speed.

## Step 1: Load Robot and Create Sphere Model

```rust
use kinetic::collision::{adjacent_link_pairs, RobotSphereModel, SphereGenConfig, SpheresSoA};
use kinetic::prelude::*;

fn main() -> kinetic::core::Result<()> {
    let robot = Robot::from_urdf_string(THREE_DOF_URDF)?;
    let chain = KinematicChain::extract(&robot, "base_link", "ee_link")?;

    let sphere_config = SphereGenConfig::default();
    let sphere_model = RobotSphereModel::from_robot(&robot, &sphere_config);
    let mut spheres = sphere_model.create_runtime();

    println!("Sphere model: {} total spheres", spheres.world.len());
```

**What this does:** `RobotSphereModel::from_robot` reads each link's `<collision>` geometry from the URDF and fits bounding spheres to approximate the shape. `create_runtime()` produces a mutable `SpheresSoA` (Structure of Arrays) that can be updated with FK poses at runtime.

**Why:** Sphere-based collision is orders of magnitude faster than mesh-based GJK/EPA. The SoA layout enables SIMD vectorization — four or eight sphere-sphere tests execute in a single CPU instruction. The trade-off is approximation quality, controlled by `SphereGenConfig` (more spheres = tighter fit, slower checks).

## Step 2: Update Spheres from FK and Check Self-Collision

```rust
    let q = vec![0.0, 1.0, -0.5];
    let link_poses = forward_kinematics_all(&robot, &chain, &q)?;
    spheres.update(&link_poses);

    let skip_pairs = adjacent_link_pairs(&robot);
    let self_collision = spheres.self_collision(&skip_pairs);
    println!("Self-collision: {}", self_collision);
```

**What this does:** `forward_kinematics_all` computes the pose of every link in the chain (not just the end-effector). `spheres.update()` transforms each sphere from link-local to world coordinates. `self_collision()` checks all sphere pairs, skipping adjacent links that always overlap at the joint.

**Why:** Adjacent links share a joint and their collision geometries overlap by design — checking them would produce false positives. `adjacent_link_pairs` builds the skip list from the URDF's parent-child relationships. Self-collision is critical for robots that can fold back on themselves (7-DOF arms, humanoids).

## Step 3: Check Environment Collision

```rust
    let mut obstacles = SpheresSoA::new();
    obstacles.push(0.5, 0.0, 0.3, 0.1, 0);  // x, y, z, radius, group_id

    let env_collision = spheres.collides_with(&obstacles);
    println!("Environment collision: {}", env_collision);
```

**What this does:** Creates a set of obstacle spheres (here, a single sphere at position (0.5, 0, 0.3) with radius 0.1m) and checks whether any robot sphere overlaps any obstacle sphere.

**Why:** Environment collision is the core check used by motion planners. During RRT expansion, every candidate configuration is tested against the scene's obstacle spheres. Keeping the check fast (sub-microsecond) is what makes real-time planning possible.

## Step 4: Test Multiple Configurations

```rust
    let configs = [
        ("Zero config",    vec![0.0, 0.0, 0.0]),
        ("Bent forward",   vec![0.0, 1.0, -0.5]),
        ("Max bend",       vec![0.0, 1.5, -1.5]),
    ];

    for (name, q) in &configs {
        let link_poses = forward_kinematics_all(&robot, &chain, q)?;
        spheres.update(&link_poses);

        let self_col = spheres.self_collision(&skip_pairs);
        let env_col = spheres.collides_with(&obstacles);

        let ee = forward_kinematics(&robot, &chain, q)?;
        let t = ee.translation();
        println!(
            "  {}: EE=({:.3}, {:.3}, {:.3}) self={} env={}",
            name, t.x, t.y, t.z, self_col, env_col
        );
    }
```

**What this does:** Iterates through several joint configurations, updating the sphere model and checking both self and environment collision for each.

**Why:** This pattern — FK update, sphere update, collision check — is the inner loop of every sampling-based planner. Understanding it helps you debug unexpected planning failures (often caused by unexpected collisions at intermediate configurations).

## Step 5: SIMD Tier and Benchmarking

```rust
    let tier = kinetic::collision::simd::detect_simd_tier();
    println!("SIMD tier: {:?}", tier);

    let start = std::time::Instant::now();
    let iters = 10_000;
    for _ in 0..iters {
        std::hint::black_box(spheres.collides_with(&obstacles));
    }
    let per_check = start.elapsed() / iters as u32;
    println!("Collision check: {:?}/check", per_check);

    Ok(())
}
```

**What this does:** Detects the CPU's SIMD capabilities (SSE4.1, AVX2, AVX-512, NEON) and benchmarks collision checking speed over 10,000 iterations. `black_box` prevents the compiler from optimizing away the computation.

**Why:** Kinetic automatically selects the best SIMD tier at runtime. AVX2 checks 8 sphere pairs per instruction, AVX-512 checks 16. Knowing your tier helps set performance expectations: on AVX2 hardware, a 50-sphere robot vs 100-sphere environment typically takes 200-500 nanoseconds per check.

## Complete Code

```rust
use kinetic::collision::{adjacent_link_pairs, RobotSphereModel, SphereGenConfig, SpheresSoA};
use kinetic::prelude::*;

fn main() -> kinetic::core::Result<()> {
    let robot = Robot::from_urdf_string(THREE_DOF_URDF)?;
    let chain = KinematicChain::extract(&robot, "base_link", "ee_link")?;

    // Build sphere model
    let sphere_model = RobotSphereModel::from_robot(&robot, &SphereGenConfig::default());
    let mut spheres = sphere_model.create_runtime();

    // Test a configuration
    let q = vec![0.0, 1.0, -0.5];
    let link_poses = forward_kinematics_all(&robot, &chain, &q)?;
    spheres.update(&link_poses);

    let skip_pairs = adjacent_link_pairs(&robot);
    println!("Self-collision: {}", spheres.self_collision(&skip_pairs));

    let mut obstacles = SpheresSoA::new();
    obstacles.push(0.5, 0.0, 0.3, 0.1, 0);
    println!("Env collision: {}", spheres.collides_with(&obstacles));

    // Benchmark
    let tier = kinetic::collision::simd::detect_simd_tier();
    println!("SIMD: {:?}", tier);

    let start = std::time::Instant::now();
    for _ in 0..10_000 {
        std::hint::black_box(spheres.collides_with(&obstacles));
    }
    println!("{:?}/check", start.elapsed() / 10_000);

    Ok(())
}
```

## What You Learned
- `RobotSphereModel::from_robot` generates bounding spheres from URDF collision geometry
- `spheres.update(&link_poses)` transforms spheres to world coordinates after FK
- `adjacent_link_pairs` prevents false positive self-collision at joints
- `SpheresSoA` stores obstacles in a cache-friendly Structure of Arrays layout
- SIMD tier is auto-detected; AVX2/AVX-512 provide significant speedups
- A single collision check typically takes 100-500 nanoseconds

## Try This
- Increase `SphereGenConfig` sphere count and observe tighter coverage vs slower checks
- Add 100 obstacle spheres and benchmark how check time scales linearly
- Compare `self_collision` results between a straight and a folded arm configuration
- Use `spheres.min_distance(&obstacles)` instead of `collides_with` to get the closest approach distance

## Next
- [Planning with Obstacles](planning-with-obstacles.md) — using collision checking in a planner
- [Servo Control](servo-control.md) — real-time collision avoidance in control loops
