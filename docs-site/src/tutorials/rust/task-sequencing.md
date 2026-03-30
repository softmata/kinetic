# Task Sequencing

## What You'll Learn
- Compose multi-stage manipulation sequences with `Task::sequence`
- Use `Task::move_to`, `Task::gripper`, `Task::pick`, and `Task::place`
- Plan an entire pick-and-place sequence as a single task
- Inspect per-stage trajectories and scene modifications
- Validate the complete solution before execution

## Prerequisites
- [Pick and Place](pick-and-place.md)
- [Grasp Planning](grasp-planning.md)
- [Motion Planning](../../concepts/motion-planning.md)

## Overview

Real manipulation tasks involve multiple stages: move to approach pose, open
gripper, descend, close gripper, retreat, move to place location, release. Rather
than planning each stage manually and stitching them together, kinetic's `Task`
enum lets you declare the sequence declaratively. The task planner handles stage
transitions, joint continuity between stages, and scene modifications (attaching
and detaching objects). This tutorial builds a multi-stage pick-place task from
primitives.

## Step 1: Set Up Robot and Scene

```rust
use std::sync::Arc;
use nalgebra::{Isometry3, Vector3};
use kinetic::prelude::*;
use kinetic_task::{Task, PickConfig, PlaceConfig, Approach};

fn main() -> std::result::Result<(), Box<dyn std::error::Error>> {
    let robot = Arc::new(Robot::from_name("ur5e")?);

    let mut scene = Scene::new(&robot)?;
    scene.add_box("table", [0.6, 0.4, 0.02], [0.5, 0.0, -0.02]);
    scene.add_box("cup", [0.03, 0.03, 0.05], [0.4, 0.1, 0.05]);
    let scene = Arc::new(scene);
```

**What this does:** Creates a UR5e robot and a scene with a table and a `cup` object. Both are wrapped in `Arc` because `Task` variants hold shared references for recursive planning.

**Why:** The scene contains both obstacles (table) and manipulable objects (cup). During pick, the cup will be "attached" to the gripper link in the scene model; during place, it will be "detached" to its target pose. `Arc` enables this shared state across task stages.

## Step 2: Build a Task Sequence

```rust
    let home_goal = Goal::joints([0.0, -1.57, 0.0, -1.57, 0.0, 0.0]);

    let task = Task::sequence(vec![
        // Stage 1: Move to home position
        Task::move_to(&robot, home_goal.clone()),

        // Stage 2: Open gripper
        Task::gripper(0.08),

        // Stage 3: Pick the cup
        Task::pick(&robot, &scene, PickConfig {
            object: "cup".into(),
            grasp_poses: vec![Isometry3::translation(0.4, 0.1, 0.05)],
            approach: Approach::linear(-Vector3::z(), 0.10),
            retreat: Approach::linear(Vector3::z(), 0.05),
            gripper_open: 0.08,
            gripper_close: 0.04,
        }),

        // Stage 4: Move to place location
        Task::move_to(&robot, Goal::joints([0.8, -1.0, 0.5, -1.0, -0.5, 0.2])),

        // Stage 5: Place the cup
        Task::place(&robot, &scene, PlaceConfig {
            object: "cup".into(),
            target_pose: Isometry3::translation(0.5, -0.2, 0.05),
            approach: Approach::linear(-Vector3::z(), 0.08),
            retreat: Approach::linear(Vector3::z(), 0.10),
            gripper_open: 0.08,
        }),

        // Stage 6: Return home
        Task::move_to(&robot, home_goal),
    ]);
```

**What this does:** Declares six stages chained so each starts where the previous ended. `Task::pick` internally generates sub-stages (approach, gripper close, retreat) and handles scene attachment.

**Why:** Declarative composition separates *what* from *how*. You describe intent; the planner handles path planning, IK, and collision checking per sub-stage.

## Step 3: Plan the Complete Task

```rust
    let start_joints = [0.0, -1.57, 0.0, -1.57, 0.0, 0.0];
    let solution = task.plan(&start_joints)?;

    println!("Task planned in {:.1}ms", solution.total_planning_time.as_millis());
    println!("Total duration: {:.3}s", solution.total_duration.as_secs_f64());
    println!("Stages: {}", solution.stages.len());
```

**What this does:** Recursively plans each sub-task, threading the final joint configuration of each stage as the start of the next.

**Why:** Single-call planning ensures joint continuity — stage N ends where stage N+1 begins, preventing discontinuities on real hardware.

## Step 4: Inspect Per-Stage Results

```rust
    for stage in &solution.stages {
        let duration = stage.trajectory.as_ref()
            .map(|t| format!("{:.3}s", t.duration.as_secs_f64()))
            .unwrap_or_else(|| "instant".to_string());

        let info = if let Some(width) = stage.gripper_command {
            format!("gripper → {:.0}mm", width * 1000.0)
        } else if let Some(diff) = &stage.scene_diff {
            format!("{:?}", diff)
        } else {
            format!("{} waypoints", stage.trajectory.as_ref()
                .map(|t| t.waypoints.len()).unwrap_or(0))
        };

        println!("  {}: {} — {}", stage.name, duration, info);
    }
```

**What this does:** Prints each stage's name, duration, and details (gripper width, waypoint count, or scene modification).

**Why:** Verify the plan before execution — check durations, gripper timing, and scene modification order.

## Step 5: Validate and Get Final Configuration

```rust
    // Get the final joint configuration
    if let Some(final_joints) = solution.final_joints() {
        println!("Final joints: {:?}", &final_joints[..3]);
    }

    // Validate all trajectories against joint limits
    let validator = TrajectoryValidator::from_robot(&robot);
    let violations = solution.validate_trajectories(&validator);
    if violations.is_empty() {
        println!("All stages pass joint limit validation");
    } else {
        for (stage_name, viols) in &violations {
            println!("WARNING: {} has {} violations", stage_name, viols.len());
        }
    }

    Ok(())
}
```

**What this does:** Extracts the final joint state and validates all stages against joint/velocity/acceleration limits.

**Why:** `final_joints()` is useful for chaining tasks. `validate_trajectories` catches limit violations across all stages before execution.

## Complete Code

The code above (Steps 1-5) forms the complete listing. Copy them into a single `main()` function. No separate example file exists for this tutorial — it is composed from the `kinetic_task` API.

## What You Learned
- `Task::sequence(vec![...])` chains multiple task stages in order
- `Task::move_to` plans a joint-space motion via RRT
- `Task::gripper(width)` inserts a gripper open/close command
- `Task::pick` and `Task::place` handle approach, grasp/release, and retreat sub-stages
- `Approach::linear(direction, distance)` specifies approach/retreat motion direction and distance
- `task.plan(&start)` plans all stages with automatic joint continuity
- `solution.validate_trajectories()` checks all stages against hardware limits

## Try This
- Add a `Task::cartesian_move` stage for a precise linear approach instead of joint-space planning
- Use `Task::pick_auto` to auto-generate grasps from the object's shape instead of specifying `grasp_poses` manually
- Nest sequences: `Task::sequence(vec![pick_sequence, place_sequence])` for modular task composition
- Apply `apply_scene_diffs(&mut scene, &solution)` to update the scene after execution and verify the cup moved

## Next
- [GPU Optimization](gpu-optimization.md) — GPU-accelerated trajectory optimization
- [Grasp Planning](grasp-planning.md) — generating grasp candidates automatically
