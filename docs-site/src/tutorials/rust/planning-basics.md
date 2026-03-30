# Planning Basics

## What You'll Learn
- Plan a joint-space motion with the `plan()` one-liner
- Use the `Planner` builder for fine-grained control
- Plan to a Cartesian pose goal instead of joint angles
- Interpret planning results (waypoints, path length, timing)

## Prerequisites
- [Motion Planning](../../concepts/motion-planning.md)
- [Forward Kinematics](../../concepts/forward-kinematics.md)
- [Robots and URDF](../../concepts/robots-and-urdf.md)

## Overview

Kinetic offers three ways to plan a motion, each trading convenience for control.
The `plan()` free function is a one-liner for quick prototyping. The `Planner`
builder lets you configure the algorithm, attach a collision scene, and set
real-time constraints. Pose goals let you specify a Cartesian target and let the
planner handle IK internally. This tutorial demonstrates all three using a UR5e.

## Step 1: One-Liner Planning

```rust
use kinetic::prelude::*;

fn main() -> kinetic::core::Result<()> {
    let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
    let goal = Goal::joints([1.0, -0.5, 0.3, 0.2, -0.3, 0.5]);

    let result = plan("ur5e", &start, &goal)?;
    println!(
        "One-liner: {} waypoints, {:.1?}, path length {:.3}",
        result.num_waypoints(),
        result.planning_time,
        result.path_length(),
    );
```

**What this does:** Loads the UR5e by name, constructs a planner with default settings, and finds a collision-free path from `start` to `goal` in joint space.

**Why:** The `plan()` free function is the fastest way to get a result. It handles robot loading, planner construction, and solving in a single call. Use it for scripts, tests, and prototyping.

## Step 2: Planner Builder

```rust
    let robot = Robot::from_name("ur5e")?;
    let planner = Planner::new(&robot)?.with_config(PlannerConfig::realtime());

    let result = planner.plan(&start, &goal)?;
    println!(
        "Builder: {} waypoints, {:.1?}, path length {:.3}",
        result.num_waypoints(),
        result.planning_time,
        result.path_length(),
    );
```

**What this does:** Creates a `Planner` explicitly, applying `PlannerConfig::realtime()` which tunes the RRT algorithm for fast (sub-10ms) planning at the cost of path optimality.

**Why:** The builder pattern gives you control over the planning algorithm, timeout, scene, and quality trade-offs. `PlannerConfig::realtime()` uses fewer iterations and a shorter planning horizon, suitable for replanning in servo loops. The default config balances quality and speed.

## Step 3: Pose Goal (Cartesian Target)

```rust
    let target_joints = vec![0.5, -0.8, 0.5, 0.1, -0.1, 0.3];
    let target_pose = planner.fk(&target_joints)?;
    let t = target_pose.translation();
    println!(
        "Planning to Cartesian pose: ({:.3}, {:.3}, {:.3})",
        t.x, t.y, t.z
    );

    let result = planner.plan(&start, &Goal::Pose(target_pose))?;
    println!(
        "Pose goal: {} waypoints, {:.1?}",
        result.num_waypoints(),
        result.planning_time,
    );

    Ok(())
}
```

**What this does:** Computes a Cartesian target from known joint values (via FK), then plans to that pose using `Goal::Pose(...)`. The planner internally solves IK to convert the pose into a joint-space goal before running the RRT.

**Why:** In real applications, goals come from perception (camera-detected object poses) or task specifications (place the tool at position X). `Goal::Pose` lets you plan directly to a world-frame target without manually solving IK first.

## Complete Code

```rust
use kinetic::prelude::*;

fn main() -> kinetic::core::Result<()> {
    println!("=== KINETIC Simple Planning ===\n");

    // --- Method 1: One-liner ---
    let start = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
    let goal = Goal::joints([1.0, -0.5, 0.3, 0.2, -0.3, 0.5]);

    let result = plan("ur5e", &start, &goal)?;
    println!(
        "One-liner: {} waypoints, {:.1?}, path length {:.3}",
        result.num_waypoints(), result.planning_time, result.path_length(),
    );

    // --- Method 2: Planner builder ---
    let robot = Robot::from_name("ur5e")?;
    let planner = Planner::new(&robot)?.with_config(PlannerConfig::realtime());

    let result = planner.plan(&start, &goal)?;
    println!(
        "Builder: {} waypoints, {:.1?}, path length {:.3}",
        result.num_waypoints(), result.planning_time, result.path_length(),
    );

    // --- Method 3: Pose goal ---
    let target_joints = vec![0.5, -0.8, 0.5, 0.1, -0.1, 0.3];
    let target_pose = planner.fk(&target_joints)?;
    let t = target_pose.translation();
    println!("\nPlanning to Cartesian pose: ({:.3}, {:.3}, {:.3})", t.x, t.y, t.z);

    let result = planner.plan(&start, &Goal::Pose(target_pose))?;
    println!("Pose goal: {} waypoints, {:.1?}", result.num_waypoints(), result.planning_time);

    Ok(())
}
```

## What You Learned
- `plan("robot_name", &start, &goal)` is the fastest way to get a path
- `Planner::new(&robot)?.with_config(...)` gives full control over algorithm parameters
- `PlannerConfig::realtime()` optimizes for low-latency planning
- `Goal::joints(...)` targets a specific joint configuration
- `Goal::Pose(...)` targets a Cartesian pose, with IK solved internally
- `PlanResult` reports waypoint count, planning time, and path length

## Try This
- Compare `PlannerConfig::default()` vs `PlannerConfig::realtime()` — measure waypoint count and planning time
- Plan the same start/goal multiple times and observe how RRT's randomness produces different paths
- Try `Goal::Named("home".into())` if the robot has named configurations in its URDF
- Use `result.waypoints` to iterate over the planned joint-space path

## Next
- [Planning with Obstacles](planning-with-obstacles.md) — adding collision objects to the scene
- [Pick and Place](pick-and-place.md) — full planning-to-execution pipeline
