# KINETIC Tutorial: Your First Motion Plan in 5 Lines

## Installation

Add KINETIC to your `Cargo.toml`:

```toml
[dependencies]
kinetic = "0.1"
```

## Quick Start — Rust

```rust
use kinetic::prelude::*;

fn main() -> kinetic::core::Result<()> {
    let robot = Robot::from_name("ur5e")?;
    let planner = Planner::new(&robot)?;
    let start = vec![0.0, -1.57, 0.0, 0.0, 0.0, 0.0];
    let goal = Goal::Joints(JointValues(vec![0.5, -1.0, 0.5, 0.0, 0.5, 0.0]));
    let result = planner.plan(&start, &goal)?;
    println!("Planned {} waypoints in {:?}", result.waypoints.len(), result.planning_time);
    Ok(())
}
```

## Quick Start — Python

```bash
pip install kinetic
```

```python
import kinetic
import numpy as np

robot = kinetic.Robot("ur5e")
planner = kinetic.Planner(robot)
start = np.array([0.0, -1.57, 0.0, 0.0, 0.0, 0.0])
goal = kinetic.Goal.joints(np.array([0.5, -1.0, 0.5, 0.0, 0.5, 0.0]))
traj = planner.plan(start, goal)
print(f"Planned {traj.num_waypoints} waypoints, duration={traj.duration:.3f}s")
```

## Step-by-Step Guide

### 1. Load a Robot

KINETIC ships with built-in configs for common robots:

```rust
// Built-in configs: ur5e, ur10e, franka_panda, kuka_iiwa7, xarm6
let robot = Robot::from_name("ur5e")?;

// Or load from a URDF file
let robot = Robot::from_urdf("/path/to/your/robot.urdf")?;
```

### 2. Forward Kinematics

Compute the end-effector pose for a given joint configuration:

```rust
let chain = KinematicChain::extract(&robot, "base_link", "tool0")?;
let joints = vec![0.0, -1.57, 0.0, 0.0, 0.0, 0.0];
let pose = forward_kinematics(&robot, &chain, &joints)?;
println!("EE position: {:?}", pose.0.translation);
```

### 3. Inverse Kinematics

Find joint values that reach a target pose:

```rust
let config = IKConfig::dls()
    .with_seed(vec![0.0; 6])
    .with_max_iterations(300);

let solution = solve_ik(&robot, &chain, &target_pose, &config)?;
println!("IK solution: {:?}", solution.joints);
```

### 3.1 Choosing an IK Solver

KINETIC provides six IK solver variants. Use this guide to pick the right one for your robot.

#### Solver Comparison

| Solver | Best For | Speed | Accuracy | DOF Requirement |
|--------|----------|-------|----------|-----------------|
| `Auto` | Default — auto-selects based on geometry | Varies | Varies | Any |
| `OPW` | Analytical + refinement for spherical-wrist arms | ~10 µs | Analytical | 6-DOF, spherical wrist |
| `Subproblem` | Analytical decomposition | ~5 µs | Analytical | 6-DOF, general geometry |
| `Subproblem7DOF` | 7-DOF redundant robots | ~50 µs | Analytical | 7-DOF (samples redundant joint) |
| `DLS` | Any robot, reliable fallback | ~200 µs | 1e-6 | Any DOF |
| `FABRIK` | Fast reaching (position + orientation refinement) | ~50 µs | 1e-4 | Any DOF |

#### Decision Flowchart

```
                     ┌──────────────────────────┐
                     │    What's your robot?     │
                     └────────────┬─────────────┘
                                  │
                     ┌────────────▼─────────────┐
                     │ 6-DOF with spherical wrist│
                     │ (UR, Fanuc, ABB, KUKA KR)?│
                     └──────┬──────────────┬─────┘
                        Yes │              │ No
                     ┌──────▼──────┐  ┌────▼────────────┐
                     │   → OPW     │  │ 6-DOF general?  │
                     └─────────────┘  └──┬──────────┬───┘
                                     Yes │          │ No
                              ┌──────────▼──┐  ┌───▼────────────┐
                              │→ Subproblem │  │    7-DOF?      │
                              └─────────────┘  │(Panda, iiwa)   │
                                               └──┬─────────┬──┘
                                               Yes│         │ No
                                    ┌──────────────▼──┐  ┌──▼────────────────┐
                                    │→ Subproblem7DOF │  │Need orientation?  │
                                    └─────────────────┘  └──┬────────────┬───┘
                                                        Yes │            │ No
                                                     ┌──────▼──┐  ┌─────▼──────┐
                                                     │ → DLS   │  │ → FABRIK   │
                                                     └─────────┘  └────────────┘

                           Not sure? → Use Auto (handles all cases)
```

#### Selecting a Solver

```rust
// Recommended: Auto detects your robot's geometry and picks the best solver
let config = IKConfig::default();

// Explicit solver selection
let config = IKConfig::dls()
    .with_seed(vec![0.0; 6])
    .with_max_iterations(300);

let config = IKConfig::fabrik();
let config = IKConfig::opw();
```

**Tips:**
- Start with `Auto` — it picks `OPW` or `Subproblem` when possible, falls back to `DLS`.
- Use `DLS` when you need orientation control on non-standard robots.
- Use `FABRIK` for fast reaching — it uses FABRIK for position convergence, then a small DLS refinement for orientation.
- `OPW` is the fastest and most accurate — but only works for 6-DOF robots with three intersecting wrist axes (UR5e, UR10e, Fanuc, ABB, KUKA KR series).

### 4. Motion Planning

Plan a collision-free path from start to goal:

```rust
let planner = Planner::new(&robot)?;

// Joint-space goal
let goal = Goal::Joints(JointValues(vec![0.5, -1.0, 0.5, 0.0, 0.5, 0.0]));

// Cartesian pose goal
let goal = Goal::Pose(Pose(target_isometry));

// Named goal (from robot config)
let goal = Goal::Named("home".to_string());

let result = planner.plan(&start_joints, &goal)?;
```

### 5. Time Parameterization

Convert a geometric path into a timed trajectory with velocity/acceleration limits:

```rust
// Simple trapezoidal profile
let traj = trapezoidal(&result.waypoints, 1.0, 2.0)?;

// Time-optimal with per-joint limits
let vel_limits = vec![2.175, 2.175, 2.175, 2.610, 2.610, 2.610];
let accel_limits = vec![15.0, 7.5, 10.0, 15.0, 20.0, 20.0];
let traj = totp(&result.waypoints, &vel_limits, &accel_limits, 0.01)?;

// Sample at any time
let wp = traj.sample_at(std::time::Duration::from_secs_f64(0.5));
println!("Positions at t=0.5s: {:?}", wp.positions);
```

### 6. Scene Management

Add collision objects to the planning environment:

```rust
use std::sync::Arc;

let robot = Arc::new(Robot::from_name("ur5e")?);
let mut scene = Scene::new(&robot)?;

// Add a table
let table_pose = Isometry3::from_parts(
    nalgebra::Translation3::new(0.5, 0.0, 0.4),
    UnitQuaternion::identity(),
);
scene.add("table", Shape::Cuboid(0.5, 0.3, 0.01), table_pose);

// Check for collisions
let in_collision = scene.check_collision(&joints)?;
let min_dist = scene.min_distance_to_robot(&joints)?;
```

### 7. Servo Control

Real-time teleoperation with singularity/collision avoidance:

```rust
let robot = Arc::new(Robot::from_name("ur5e")?);
let scene = Arc::new(Scene::new(&robot)?);
let config = ServoConfig::default();

let mut servo = Servo::new(&robot, &scene, config)?;
servo.set_state(&initial_joints, &vec![0.0; 6])?;

// Send Cartesian velocity
let twist = Twist::new(
    Vector3::new(0.05, 0.0, 0.0),  // linear
    Vector3::new(0.0, 0.0, 0.0),   // angular
);
let cmd = servo.send_twist(&twist)?;
println!("Joint command: {:?}", cmd.positions);
```

### 8. GPU-Accelerated Planning

For complex environments or when you need to evaluate hundreds of trajectory candidates simultaneously, KINETIC offers GPU-accelerated trajectory optimization via wgpu compute shaders.

#### When to Use GPU Planning

- **Many parallel seeds** — evaluate 128–1024 trajectory candidates simultaneously
- **Complex environments** — dense obstacle fields where CPU planning is slow
- **Batch optimization** — when you need the globally best trajectory, not just the first feasible one

GPU planning works on any wgpu-compatible GPU: NVIDIA (Vulkan), AMD (Vulkan), Intel (Vulkan), and Apple Silicon (Metal).

#### Basic Usage

```rust
use kinetic::prelude::*;

let robot = Robot::from_name("ur5e")?;
let scene = Scene::new(&robot)?;

// Configure GPU optimizer
let config = GpuConfig {
    num_seeds: 256,      // parallel trajectory candidates
    timesteps: 32,       // waypoints per trajectory
    iterations: 100,     // gradient descent iterations
    ..Default::default()
};

let optimizer = GpuOptimizer::new(config)?;

let start = vec![0.0, -1.57, 0.0, 0.0, 0.0, 0.0];
let goal = vec![1.0, -1.0, 0.5, 0.0, 0.5, 0.0];

let trajectory = optimizer.optimize(
    &robot,
    &scene.build_environment_spheres(),
    &start,
    &goal,
)?;

println!("Optimized trajectory: {} waypoints", trajectory.len());
```

#### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_seeds` | 128 | Parallel trajectory candidates |
| `timesteps` | 32 | Waypoints per trajectory |
| `iterations` | 100 | Gradient descent iterations |
| `collision_weight` | 100.0 | Penalty for obstacle proximity |
| `smoothness_weight` | 1.0 | Penalty for jerky motion |
| `goal_weight` | 50.0 | Penalty for missing the goal |
| `step_size` | 0.01 | Gradient descent step size |
| `sdf_resolution` | 0.02 | Voxel size for signed distance field (m) |
| `seed_perturbation` | 0.3 | Random perturbation magnitude for seed trajectories (rad) |
| `workspace_bounds` | [-1, -1, -0.5, 1, 1, 1.5] | Axis-aligned bounding box [x_min, y_min, z_min, x_max, y_max, z_max] (m) |

## Available Robots

| Robot | DOF | Config Name |
|-------|-----|-------------|
| Universal Robots UR5e | 6 | `ur5e` |
| Universal Robots UR10e | 6 | `ur10e` |
| Franka Emika Panda | 7 | `franka_panda` |
| KUKA iiwa 7 | 7 | `kuka_iiwa7` |
| xArm 6 | 6 | `xarm6` |

## Performance

Run benchmarks with:

```bash
cargo bench -p kinetic
```

See [benchmarks.md](benchmarks.md) for detailed performance numbers.
