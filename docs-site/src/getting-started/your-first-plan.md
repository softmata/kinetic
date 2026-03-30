# Your First Plan

Now let's plan a collision-free motion, time-parameterize it, and validate it for execution on a real robot.

## Step 1: Create a Planner

**Rust:**
```rust
use kinetic::prelude::*;

fn main() -> Result<()> {
    let robot = Robot::from_name("ur5e")?;
    let planner = Planner::new(&robot)?;
```

**Python:**
```python
import kinetic
import numpy as np

robot = kinetic.Robot("ur5e")
planner = kinetic.Planner(robot)
```

`Planner::new()` auto-detects the robot's kinematic chain (which joints to plan for) and builds a collision model. You can customize it with `.with_config(PlannerConfig::realtime())` for faster but less optimal plans.

## Step 2: Define Start and Goal

```rust
    let start = vec![0.0, -1.57, 0.0, -1.57, 0.0, 0.0];
    let goal = Goal::joints([1.0, -1.0, 0.5, -1.0, -0.5, 0.3]);
```

```python
start = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
goal = kinetic.Goal.joints(np.array([1.0, -1.0, 0.5, -1.0, -0.5, 0.3]))
```

Goals can be specified in several ways:

| Goal type | Rust | Python | When to use |
|-----------|------|--------|-------------|
| Joint angles | `Goal::joints([...])` | `Goal.joints(array)` | You know the exact configuration |
| Cartesian pose | `Goal::Pose(pose)` | `Goal.pose(matrix_4x4)` | You know where the EE should be |
| Named pose | `Goal::named("home")` | `Goal.named("home")` | Using pre-defined configurations |

## Step 3: Plan

```rust
    let result = planner.plan(&start, &goal)?;
    println!("Path: {} waypoints, {:.1?}", result.num_waypoints(), result.planning_time);
    println!("Path length: {:.3} rad", result.path_length());
```

```python
traj = planner.plan(start, goal)
print(f"Path: {traj.num_waypoints} waypoints, {traj.duration:.3f}s")
```

The planner returns a sequence of joint configurations (waypoints) that form a collision-free path from start to goal.

**What if planning fails?** The planner returns an error with a reason:
- `StartInCollision` — your start configuration collides with something
- `GoalUnreachable` — the goal is outside the robot's workspace
- `PlanningTimeout` — the planner ran out of time (try `PlannerConfig::offline()` for more time)

See [Troubleshooting](../guides/troubleshooting.md) for systematic diagnosis.

## Step 4: Time-Parameterize

The planner outputs a geometric path (positions only). To execute on a real robot, you need velocities and timing:

```rust
    let vel_limits = robot.velocity_limits();
    let acc_limits = robot.acceleration_limits();
    let timed = kinetic::trajectory::trapezoidal_per_joint(
        &result.waypoints, &vel_limits, &acc_limits,
    ).map_err(|e| KineticError::Other(e))?;

    println!("Trajectory: {:.3}s, {} timed waypoints",
             timed.duration.as_secs_f64(), timed.waypoints.len());
```

```python
vel = np.array(robot.velocity_limits)
acc = np.array(robot.acceleration_limits)
timed = traj.time_parameterize("trapezoidal", vel, acc)
print(f"Trajectory: {timed.duration:.3f}s, {timed.num_waypoints} waypoints")
```

The trapezoidal profile accelerates, cruises, and decelerates each joint, respecting its velocity and acceleration limits. Other profiles are available:

| Profile | Best for | Smoothness |
|---------|----------|-----------|
| `trapezoidal` | General use | Velocity-continuous |
| `totp` | Time-optimal | Velocity-continuous |
| `jerk_limited` | Delicate tasks | Acceleration-continuous |
| `cubic_spline` | Smooth motion | C2-continuous |

## Step 5: Validate

Before sending to a real robot, validate the trajectory:

```rust
    // Check every waypoint is within joint limits
    for wp in &timed.waypoints {
        for (j, &pos) in wp.positions.iter().enumerate() {
            assert!(pos >= robot.joint_limits[j].lower - 1e-6);
            assert!(pos <= robot.joint_limits[j].upper + 1e-6);
        }
    }
    println!("All waypoints within joint limits");
```

```python
violations = timed.validate(
    np.array([-6.28] * 6),  # position lower limits
    np.array([6.28] * 6),   # position upper limits
    vel, acc
)
if not violations:
    print("Trajectory is valid")
else:
    print(f"Found {len(violations)} violations")
```

## Step 6: Export

Export the trajectory for your robot controller:

```rust
    let json = trajectory_to_json(&timed);
    std::fs::write("trajectory.json", &json)?;
    println!("Exported {} bytes", json.len());

    Ok(())
}
```

```python
times, positions, velocities = timed.to_numpy()
# times: shape (N,), positions: shape (N, 6), velocities: shape (N, 6)

# Plot with matplotlib
import matplotlib.pyplot as plt
for j in range(6):
    plt.plot(times, positions[:, j], label=f"Joint {j}")
plt.xlabel("Time (s)")
plt.ylabel("Position (rad)")
plt.legend()
plt.savefig("trajectory.png")
```

## Complete Rust Example

```rust
use kinetic::prelude::*;

fn main() -> Result<()> {
    let robot = Robot::from_name("ur5e")?;
    let planner = Planner::new(&robot)?;

    let start = vec![0.0, -1.57, 0.0, -1.57, 0.0, 0.0];
    let goal = Goal::joints([1.0, -1.0, 0.5, -1.0, -0.5, 0.3]);

    let result = planner.plan(&start, &goal)?;
    let vel = robot.velocity_limits();
    let acc = robot.acceleration_limits();
    let timed = kinetic::trajectory::trapezoidal_per_joint(
        &result.waypoints, &vel, &acc,
    ).map_err(|e| KineticError::Other(e))?;

    println!("{} waypoints, {:.3}s trajectory", timed.waypoints.len(), timed.duration.as_secs_f64());
    Ok(())
}
```

## Try This

1. Add an obstacle: create a `Scene`, add a box with `scene.add_box(...)`, and plan with `.with_scene(&scene)` — see [Planning with Obstacles](../tutorials/rust/planning-with-obstacles.md)
2. Try `PlannerConfig::realtime()` vs `PlannerConfig::offline()` — how does planning time and path quality change?
3. Use a `Goal::Pose(...)` instead of `Goal::joints(...)` — kinetic will solve IK internally
4. Export to CSV with `trajectory_to_csv(&timed)` and plot in your favorite tool

## Next

[Next Steps →](next-steps.md)
