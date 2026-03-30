# Planning with Obstacles

## What You'll Learn
- Create a collision `Scene` with box and cylinder obstacles
- Plan collision-free paths around scene objects
- Apply trapezoidal time parameterization to produce executable trajectories
- Inspect timed waypoints with timestamps and velocities

## Prerequisites
- [Collision Detection](../../concepts/collision-detection.md)
- [Motion Planning](../../concepts/motion-planning.md)
- [Trajectory Generation](../../concepts/trajectory-generation.md)

## Overview

Real-world robots operate among tables, walls, and objects. Kinetic's `Scene`
holds geometric obstacles that the planner avoids during path search. After
planning, a raw waypoint path needs time parameterization to become a trajectory
with timestamps, velocities, and accelerations that a controller can execute.
This tutorial builds a scene with three obstacles, plans around them, and applies
a trapezoidal velocity profile.

## Step 1: Load Robot and Create Scene

```rust
use kinetic::prelude::*;

fn main() -> kinetic::core::Result<()> {
    let robot = Robot::from_name("ur5e")?;
    println!("Loaded '{}' ({} DOF)", robot.name, robot.dof);

    let mut scene = Scene::new(&robot)?;

    // Table surface
    scene.add_box("table", [0.4, 0.3, 0.01], [0.5, 0.0, 0.0]);

    // Box obstacle on the table
    scene.add_box("box", [0.05, 0.05, 0.1], [0.4, 0.0, 0.11]);

    // Cylindrical obstacle
    scene.add_cylinder("pipe", 0.03, 0.15, [0.3, 0.1, 0.15]);

    println!("Scene: {} obstacles", scene.num_objects());
```

**What this does:** Creates a `Scene` bound to the UR5e, then adds three geometric primitives. `add_box` takes half-extents `[x, y, z]` and a center position. `add_cylinder` takes radius, half-height, and center.

**Why:** The scene is the planner's view of the world. Every obstacle is checked during RRT expansion — any candidate configuration that puts the robot in collision with a scene object is rejected. Accurate scene geometry is critical for safe motion planning.

## Step 2: Plan Around Obstacles

```rust
    let planner = Planner::new(&robot)?
        .with_scene(&scene)
        .with_config(PlannerConfig::default());

    let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
    let goal = Goal::joints([1.0, -0.5, 0.3, 0.2, -0.3, 0.5]);

    let result = planner.plan(&start, &goal)?;
    println!(
        "Path found: {} waypoints in {:.1?}",
        result.num_waypoints(),
        result.planning_time,
    );
    println!("Path length: {:.3} rad", result.path_length());
```

**What this does:** Attaches the scene to the planner via `.with_scene(&scene)`, then plans a joint-space path from `start` to `goal`. The RRT samples random configurations, checks each for collisions against both the robot's self-collision model and the scene, and grows a tree until it connects start to goal.

**Why:** Without `.with_scene()`, the planner only checks self-collision. With a scene, every candidate node in the RRT tree is validated against all obstacles. The planner automatically inflates collision geometry by a safety margin to account for real-world uncertainty.

## Step 3: Time-Parameterize the Path

```rust
    let max_vel = 2.0;  // rad/s
    let max_acc = 5.0;  // rad/s^2
    let timed = trapezoidal(&result.waypoints, max_vel, max_acc)
        .map_err(KineticError::Other)?;

    println!("Trajectory duration: {:.3?}", timed.duration());
    println!("Timed waypoints: {}", timed.waypoints.len());
```

**What this does:** Applies a trapezoidal velocity profile to the geometric path. Each segment accelerates to `max_vel`, cruises, then decelerates — producing the classic trapezoidal velocity shape. The result is a `TimedTrajectory` where each waypoint has a timestamp.

**Why:** A raw path is just a sequence of joint configurations with no timing information. A controller needs to know *when* to reach each waypoint and *how fast* to move. Trapezoidal parameterization is the simplest profile that respects velocity and acceleration limits. For smoother motion, kinetic also provides `cubic_spline` and `time_optimal` parameterizations.

## Step 4: Inspect the Trajectory

```rust
    if let Some(first) = timed.waypoints.first() {
        println!("  t=0.000s: {:6.3?}", &first.positions[..3]);
    }
    if let Some(last) = timed.waypoints.last() {
        println!("  t={:.3}s: {:6.3?}", last.time, &last.positions[..3]);
    }

    Ok(())
}
```

**What this does:** Prints the first three joint values at the start and end of the trajectory, along with their timestamps.

**Why:** Inspecting boundary waypoints confirms the trajectory starts at `start` and ends at `goal`. In production, you would iterate over `timed.waypoints` to send position commands to a real controller at the prescribed rate.

## Complete Code

```rust
use kinetic::prelude::*;

fn main() -> kinetic::core::Result<()> {
    // 1. Load robot
    let robot = Robot::from_name("ur5e")?;

    // 2. Create scene with obstacles
    let mut scene = Scene::new(&robot)?;
    scene.add_box("table", [0.4, 0.3, 0.01], [0.5, 0.0, 0.0]);
    scene.add_box("box", [0.05, 0.05, 0.1], [0.4, 0.0, 0.11]);
    scene.add_cylinder("pipe", 0.03, 0.15, [0.3, 0.1, 0.15]);

    // 3. Plan around obstacles
    let planner = Planner::new(&robot)?
        .with_scene(&scene)
        .with_config(PlannerConfig::default());

    let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
    let goal = Goal::joints([1.0, -0.5, 0.3, 0.2, -0.3, 0.5]);
    let result = planner.plan(&start, &goal)?;
    println!("Path: {} waypoints in {:.1?}", result.num_waypoints(), result.planning_time);

    // 4. Time-parameterize
    let timed = trapezoidal(&result.waypoints, 2.0, 5.0).map_err(KineticError::Other)?;
    println!("Trajectory: {:.3?}, {} waypoints", timed.duration(), timed.waypoints.len());

    // 5. Inspect endpoints
    if let Some(first) = timed.waypoints.first() {
        println!("  t=0.000s: {:6.3?}", &first.positions[..3]);
    }
    if let Some(last) = timed.waypoints.last() {
        println!("  t={:.3}s: {:6.3?}", last.time, &last.positions[..3]);
    }

    Ok(())
}
```

## What You Learned
- `Scene::new(&robot)` creates a collision environment bound to a robot
- `add_box` and `add_cylinder` place geometric primitives at world-frame positions
- `.with_scene(&scene)` makes the planner collision-aware
- `trapezoidal()` converts a geometric path into a timed trajectory with velocity/acceleration limits
- `TimedTrajectory` waypoints have `.time`, `.positions`, and `.velocities` fields

## Try This
- Move the `box` obstacle directly between start and goal to force a longer detour
- Compare `trapezoidal()` with lower `max_vel` (0.5 rad/s) and observe the longer trajectory duration
- Add more obstacles with `scene.add_sphere("ball", radius, position)` and see how planning time changes
- Use `scene.remove("pipe")` to remove an obstacle and observe the shorter path

## Next
- [Collision Checking](collision-checking.md) — low-level sphere-model collision detection
- [Pick and Place](pick-and-place.md) — full workflow from planning through execution
