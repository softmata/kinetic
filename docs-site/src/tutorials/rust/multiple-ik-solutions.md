# Multiple IK Solutions

## What You'll Learn
- Why a single IK target can have multiple valid joint configurations
- Use different seed configurations to discover diverse solutions
- Compare solutions by joint-space distance
- Choose the best solution for your application

## Prerequisites
- [Inverse Kinematics](../../concepts/inverse-kinematics.md)
- [FK and IK Tutorial](fk-and-ik.md)
- [IK Solver Selection](ik-solver-selection.md)

## Overview

A 7-DOF robot has infinite IK solutions for most reachable targets due to its
redundant degree of freedom. Even 6-DOF robots often have 2-8 distinct solutions
(elbow-up vs elbow-down, shoulder-left vs shoulder-right). By seeding the IK
solver with different starting configurations, you can discover multiple distinct
joint configurations that all reach the same end-effector pose. This tutorial
shows how to collect, compare, and choose among them.

## Step 1: Establish a Target Pose

```rust
use kinetic::prelude::*;

const PANDA_URDF: &str = include_str!("panda_urdf.txt");

fn main() -> kinetic::core::Result<()> {
    let robot = Robot::from_urdf_string(PANDA_URDF)?;
    let chain = KinematicChain::extract(&robot, "panda_link0", "panda_link8")?;

    // Create a target by FK from known joints
    let q_target = vec![0.3, -0.5, 0.2, -1.5, 0.1, 1.0, 0.5];
    let target = forward_kinematics(&robot, &chain, &q_target)?;
    let t = target.translation();
    println!("Target pose: ({:.4}, {:.4}, {:.4})", t.x, t.y, t.z);
```

**What this does:** Computes a target pose from known joint values using FK. This guarantees the target is reachable, which is important when testing IK — an unreachable target would cause all seeds to fail.

**Why:** In production, targets come from perception or task planning. Here we use FK to guarantee reachability so we can focus on demonstrating the multi-solution aspect.

## Step 2: Define Diverse Seeds

```rust
    let seeds = [
        robot.mid_configuration().to_vec(),                    // mid-range
        vec![0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0],           // near-zero
        vec![1.0, -1.0, 1.0, -2.0, 0.5, 2.0, -0.5],         // far-positive
        vec![-0.5, 0.5, -0.3, -1.0, -0.5, 0.5, 1.0],        // mixed
    ];
```

**What this does:** Defines four starting joint configurations spread across the robot's workspace. Each seed puts the IK solver in a different region of configuration space.

**Why:** DLS is a gradient-based solver that converges to the nearest local minimum. Different seeds lead to different minima, which correspond to different physical arm configurations (e.g., elbow-up vs elbow-down). The further apart the seeds, the more likely you are to discover genuinely different solutions.

## Step 3: Solve from Each Seed

```rust
    let mut solutions = Vec::new();

    for (i, seed) in seeds.iter().enumerate() {
        let config = IKConfig::dls()
            .with_seed(seed.clone())
            .with_max_iterations(300);

        match solve_ik(&robot, &chain, &target, &config) {
            Ok(sol) => {
                println!(
                    "Solution #{}: {} iters, pos_err={:.2e}",
                    i + 1, sol.iterations, sol.position_error
                );
                print!("  joints: [");
                for (j, &v) in sol.joints.iter().enumerate() {
                    if j > 0 { print!(", "); }
                    print!("{:.3}", v);
                }
                println!("]");
                solutions.push(sol);
            }
            Err(e) => {
                println!("Solution #{}: failed — {}", i + 1, e);
            }
        }
    }
```

**What this does:** Runs IK four times with different seeds. Each successful solve is stored. Failed solves are logged but do not stop the search — not every seed converges.

**Why:** Collecting multiple solutions lets you choose the one that best fits your constraints (shortest travel distance, avoids singularity, maintains elbow clearance, etc.). Failed seeds often mean the solver started too far from any valid solution, which is normal.

## Step 4: Compare Solutions by Distance

```rust
    if solutions.len() >= 2 {
        println!("\n--- Solution distances (L2 in joint space) ---");
        for i in 0..solutions.len() {
            for j in (i + 1)..solutions.len() {
                let dist: f64 = solutions[i].joints.iter()
                    .zip(solutions[j].joints.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                println!("  #{} <-> #{}: {:.4} rad", i + 1, j + 1, dist);
            }
        }
    }

    Ok(())
}
```

**What this does:** Computes pairwise L2 distances between all solution joint vectors. Large distances (> 1.0 rad) indicate genuinely different arm configurations.

**Why:** Two solutions with distance < 0.01 rad are essentially the same configuration found from different starting points. Solutions with distance > 1.0 rad represent distinct physical poses. In pick-and-place, you would pick the solution closest to the current joint state to minimize travel time.

## Complete Code

```rust
use kinetic::prelude::*;

const PANDA_URDF: &str = include_str!("panda_urdf.txt");

fn main() -> kinetic::core::Result<()> {
    let robot = Robot::from_urdf_string(PANDA_URDF)?;
    let chain = KinematicChain::extract(&robot, "panda_link0", "panda_link8")?;

    // Target pose from known joints
    let q_target = vec![0.3, -0.5, 0.2, -1.5, 0.1, 1.0, 0.5];
    let target = forward_kinematics(&robot, &chain, &q_target)?;

    // Solve from four different seeds
    let seeds = [
        robot.mid_configuration().to_vec(),
        vec![0.0, 0.0, 0.0, -1.5, 0.0, 1.5, 0.0],
        vec![1.0, -1.0, 1.0, -2.0, 0.5, 2.0, -0.5],
        vec![-0.5, 0.5, -0.3, -1.0, -0.5, 0.5, 1.0],
    ];

    let mut solutions = Vec::new();
    for (i, seed) in seeds.iter().enumerate() {
        let config = IKConfig::dls().with_seed(seed.clone()).with_max_iterations(300);
        match solve_ik(&robot, &chain, &target, &config) {
            Ok(sol) => { solutions.push(sol); }
            Err(e) => { println!("Seed #{} failed: {}", i + 1, e); }
        }
    }

    // Compare pairwise distances
    for i in 0..solutions.len() {
        for j in (i + 1)..solutions.len() {
            let dist: f64 = solutions[i].joints.iter()
                .zip(solutions[j].joints.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum::<f64>().sqrt();
            println!("#{} <-> #{}: {:.4} rad", i + 1, j + 1, dist);
        }
    }

    Ok(())
}
```

## What You Learned
- The same target pose can have multiple valid joint configurations
- Different seeds guide the iterative solver to different local minima
- `robot.mid_configuration()` provides a safe default seed at the center of joint ranges
- L2 distance in joint space measures how different two solutions are
- Distances > 1.0 rad indicate genuinely distinct arm configurations

## Try This
- Add more seeds (8-10) and count how many unique solutions you find (distance > 0.5 rad)
- Use `IKConfig::dls().with_restarts(10)` instead of manual seeds for automatic random restart exploration
- Pick the solution closest to the current joint state using `min_by` on joint distance
- Try a 6-DOF robot (UR5e) and observe that it typically has fewer distinct solutions than a 7-DOF robot
- Use `solution.condition_number` to filter out near-singular solutions

## Next
- [Collision Checking](collision-checking.md) — verifying solutions are collision-free
- [Planning Basics](planning-basics.md) — using the planner instead of raw IK
