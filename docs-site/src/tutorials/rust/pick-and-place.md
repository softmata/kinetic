# Pick and Place

## What You'll Learn
- Build a complete plan-to-execute pipeline for a UR5e
- Create a collision scene with obstacles
- Plan, time-parameterize, and execute a trajectory
- Verify joint limits and export the trajectory to JSON
- Use `PlanExecuteLoop` for a one-liner workflow

## Prerequisites
- [Planning Basics](planning-basics.md)
- [Planning with Obstacles](planning-with-obstacles.md)
- [Trajectory Generation](../../concepts/trajectory-generation.md)

## Overview

This tutorial walks through kinetic's full workflow from loading a robot to
exporting a trajectory. It covers scene construction, collision-aware planning,
per-joint trapezoidal time parameterization using the robot's actual velocity and
acceleration limits, execution with a logging executor, joint limit validation,
frame tracking, and JSON export. Finally, it shows `PlanExecuteLoop` — a
one-liner that handles planning and execution together.

## Step 1: Load Robot and Build Scene

```rust
use kinetic::prelude::*;

fn main() -> Result<()> {
    let robot = Robot::from_name("ur5e")?;
    println!("Loaded {} ({}DOF)", robot.name, robot.dof);

    let mut scene = Scene::new(&robot)?;
    scene.add_box("table", [0.4, 0.3, 0.02], [0.5, 0.0, -0.02]);
    println!("Scene: {} objects", scene.num_objects());
```

**What this does:** Loads the UR5e from the built-in robot library and creates a scene with a table surface. The table is a thin box positioned just below the XY plane.

**Why:** `Robot::from_name` is the quickest way to load a supported robot — it bundles URDF, joint limits, and collision meshes. The scene ensures the planner avoids the table during path search.

## Step 2: Plan a Collision-Free Path

```rust
    let start = vec![0.0, -1.57, 0.0, -1.57, 0.0, 0.0];
    let goal_joints = vec![1.0, -1.0, 0.5, -1.0, -0.5, 0.3];

    let planner = Planner::new(&robot)?.with_scene(&scene);
    let plan_result = planner.plan(&start, &Goal::joints(goal_joints.clone()))?;
    println!(
        "Planned: {} waypoints in {:.1}ms",
        plan_result.waypoints.len(),
        plan_result.planning_time.as_secs_f64() * 1000.0
    );
```

**What this does:** Plans a joint-space path from `start` to `goal_joints` with scene awareness. The planner runs RRT-Connect to find a collision-free path.

**Why:** The planning result is a geometric path — a sequence of waypoint joint configurations. It has no timing information yet. The waypoint count depends on the scene complexity and path length; the planner inserts waypoints at collision-critical points.

## Step 3: Time-Parameterize with Per-Joint Limits

```rust
    let vel_limits = robot.velocity_limits();
    let accel_limits = robot.acceleration_limits();
    let timed = kinetic::trajectory::trapezoidal_per_joint(
        &plan_result.waypoints,
        &vel_limits,
        &accel_limits,
    ).map_err(|e| KineticError::PlanningFailed(e))?;

    println!(
        "Trajectory: {:.3}s, {} waypoints",
        timed.duration.as_secs_f64(),
        timed.waypoints.len()
    );
```

**What this does:** Applies trapezoidal velocity profiling with per-joint limits. Each joint accelerates to its individual maximum velocity, cruises, then decelerates — respecting the robot's actual hardware limits from the URDF.

**Why:** Unlike the simpler `trapezoidal()` which uses a single velocity limit for all joints, `trapezoidal_per_joint` uses the robot's real limits. Shoulder joints are typically slower (2.175 rad/s on UR5e) while wrist joints are faster (2.61 rad/s). Using real limits produces trajectories that run at full speed without exceeding hardware constraints.

## Step 4: Execute and Log

```rust
    let mut executor = LogExecutor::new(ExecutionConfig {
        rate_hz: 100.0,
        ..Default::default()
    });
    let exec_result = executor.execute_and_log(&timed)
        .map_err(|e| KineticError::PlanningFailed(e.to_string()))?;

    println!(
        "Executed: {:?}, {} commands, {:.3}s",
        exec_result.state, exec_result.commands_sent,
        exec_result.actual_duration.as_secs_f64()
    );
```

**What this does:** Creates a `LogExecutor` that records every joint command at 100 Hz. `execute_and_log` walks the trajectory, interpolating positions at each timestep, and stores all commands for later analysis.

**Why:** `LogExecutor` is a simulation executor for testing and validation. It implements the same `Executor` trait as hardware executors, so you can develop and test your pipeline without a real robot, then swap in a hardware executor for deployment.

## Step 5: Validate Joint Limits

```rust
    let commands = executor.commands();
    let mut all_valid = true;
    for cmd in commands {
        for (j, &pos) in cmd.positions.iter().enumerate() {
            if pos < robot.joint_limits[j].lower - 0.001
                || pos > robot.joint_limits[j].upper + 0.001
            {
                println!("WARNING: joint {} at {:.4} outside limits", j, pos);
                all_valid = false;
            }
        }
    }
    if all_valid {
        println!("All {} commands within joint limits", commands.len());
    }
```

**What this does:** Iterates every logged command and verifies each joint position falls within the robot's limits (with 1 mm tolerance for floating-point rounding).

**Why:** Joint limit validation is a safety check that catches bugs in planning or time parameterization. On real hardware, exceeding joint limits can trigger emergency stops or cause mechanical damage. Always validate before deploying to production.

## Step 6: Export and Frame Tracking

```rust
    // Track coordinate frames (like ROS tf2)
    let tree = FrameTree::new();
    tree.set_static_transform("base_link", "camera", Isometry3::translation(0.0, 0.0, 0.5));

    // Verify final end-effector pose
    let final_pose = kinetic::kinematics::forward_kinematics(
        &robot, &planner.chain(), &exec_result.final_positions,
    )?;
    println!("Final EE: [{:.4}, {:.4}, {:.4}]",
        final_pose.translation().x, final_pose.translation().y, final_pose.translation().z);

    // Export trajectory to JSON
    let json = trajectory_to_json(&timed);
    println!("Exported: {} bytes JSON", json.len());
```

**What this does:** `FrameTree` manages parent-child transform relationships (like ROS tf2). `trajectory_to_json` serializes for visualization or interop.

**Why:** Frame tracking keeps sensor-to-robot calibrations organized. JSON export enables replay, analysis in external tools, and archival.

## Step 7: PlanExecuteLoop One-Liner

```rust
    let planner2 = Planner::new(&robot)?;
    let sim_executor = Box::new(SimExecutor::default());
    let mut pel = PlanExecuteLoop::new(planner2, sim_executor);

    let pel_result = pel.move_to(&start, &Goal::joints(goal_joints))?;
    println!(
        "PlanExecuteLoop: {:.1}ms, {} replans",
        pel_result.total_duration.as_secs_f64() * 1000.0,
        pel_result.replans
    );

    Ok(())
}
```

**What this does:** `PlanExecuteLoop` wraps a planner and executor into a single object. `move_to` plans, time-parameterizes, and executes in one call, with automatic replanning if the first attempt fails or the environment changes.

**Why:** For applications that do not need fine-grained control over each stage, `PlanExecuteLoop` reduces the pipeline to a single method call. It handles retries, replanning on collision, and trajectory validation internally.

## Complete Code

See `examples/plan_and_execute.rs` for the full listing combining all steps above.

## What You Learned
- `Robot::from_name("ur5e")` loads a built-in robot with correct limits
- `trapezoidal_per_joint` respects each joint's individual velocity/acceleration limits
- `LogExecutor` records commands; `FrameTree` tracks frames; `trajectory_to_json` exports
- `PlanExecuteLoop` wraps the full pipeline in a single `move_to` call

## Try This
- Replace `LogExecutor` with `SimExecutor` and compare the execution results
- Add a second motion from `goal` back to `start` and concatenate the trajectories
- Use `TrajectoryValidator` to check velocity and acceleration limits before execution
- Call `planner.fk(&exec_result.final_positions)` to verify the arm reached the goal pose

## Next
- [Task Sequencing](task-sequencing.md) — multi-stage pick-place sequences
- [GPU Optimization](gpu-optimization.md) — GPU-accelerated trajectory optimization
