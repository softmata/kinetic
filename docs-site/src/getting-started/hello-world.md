# Hello World

Let's plan a robot motion in 5 lines.

## Rust

```rust
use kinetic::prelude::*;

fn main() -> Result<()> {
    let start = [0.0, -1.0, 0.8, 0.0, 0.0, 0.0];
    let goal = Goal::joints([1.0, -0.5, 0.3, 0.2, -0.3, 0.5]);
    let result = plan("ur5e", &start, &goal)?;
    println!("{} waypoints in {:?}", result.num_waypoints(), result.planning_time);
    Ok(())
}
```

## Python

```python
import kinetic
import numpy as np

start = np.array([0.0, -1.0, 0.8, 0.0, 0.0, 0.0])
goal = kinetic.Goal.joints(np.array([1.0, -0.5, 0.3, 0.2, -0.3, 0.5]))
traj = kinetic.plan("ur5e", start, goal)
print(f"{traj.num_waypoints} waypoints, {traj.duration:.3f}s")
```

## What Just Happened?

In those 5 lines, kinetic:

1. **Loaded the UR5e robot** — a 6-DOF industrial arm from Universal Robots. Kinetic parsed its URDF model, extracted joint limits, and built a collision sphere model. (See: [Robots and URDF](../concepts/robots-and-urdf.md))

2. **Defined start and goal** — joint configurations in radians. The UR5e has 6 joints, so each array has 6 values. `start` is where the arm is now; `goal` is where you want it. (See: [Glossary → Joint Configuration](../concepts/glossary.md))

3. **Planned a path** — using RRT-Connect, a sampling-based algorithm that grows two search trees (one from start, one from goal) until they connect. The result is a sequence of collision-free waypoints through joint space. (See: [Motion Planning](../concepts/motion-planning.md))

4. **Time-parameterized the path** — converted the geometric path into a timed trajectory with velocity and acceleration profiles that respect the robot's joint limits. (See: [Trajectory Generation](../concepts/trajectory-generation.md))

5. **Returned the result** — a `PlanningResult` (Rust) or `Trajectory` (Python) containing timed waypoints you can send to a robot controller.

## What Do the Numbers Mean?

- **`start = [0.0, -1.0, 0.8, 0.0, 0.0, 0.0]`** — Joint angles in radians. Joint 1 at 0°, joint 2 at -57°, joint 3 at 46°, etc.
- **`goal = [1.0, -0.5, 0.3, 0.2, -0.3, 0.5]`** — The target joint configuration.
- **`14 waypoints`** — The planner found a path through 14 intermediate configurations.
- **`237ms`** — Planning took 237 milliseconds (this varies — RRT is probabilistic).

## Try This

1. Change the goal to `[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]` (zero configuration) and plan again
2. Try a different robot: replace `"ur5e"` with `"franka_panda"` (7 DOF — the arrays need 7 values)
3. Print all waypoints: `for wp in result.waypoints { println!("{:?}", wp); }`

## Next

[Your First Robot →](your-first-robot.md)
