# Planning in Python

Collision-aware planning, constrained motion, Cartesian straight-line paths,
and error handling.

## Setup

```python
import kinetic
import numpy as np

robot = kinetic.Robot("ur5e")
start = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
```

## Choose a planner

kinetic ships 9 planning algorithms. Pass `planner_type` to `Planner`:

```python
# Default: RRT-Connect (fastest for most problems)
planner = kinetic.Planner(robot)

# Asymptotically optimal (better paths, slower)
planner = kinetic.Planner(robot, planner_type="rrt_star")

# All options: rrt_connect, rrt_star, bi_rrt_star, bitrrt, est, kpiece, prm, gcs
```

Not sure which to use? See the [Planner Selection Guide](../../guides/planner-selection.md).

## Build a scene with obstacles

A `Scene` holds collision objects. Each has a `Shape` and a 4x4 pose matrix:

```python
scene = kinetic.Scene(robot)

# Table (1m x 1m x 2cm)
table_pose = np.eye(4)
table_pose[2, 3] = 0.4
scene.add("table", kinetic.Shape.cuboid(0.5, 0.5, 0.01), table_pose)

# Mug (cylinder r=4cm, h=12cm)
mug_pose = np.eye(4)
mug_pose[0, 3], mug_pose[2, 3] = 0.3, 0.46
scene.add("mug", kinetic.Shape.cylinder(0.04, 0.06), mug_pose)

# Ball (sphere r=5cm)
ball_pose = np.eye(4)
ball_pose[:3, 3] = [0.5, 0.2, 0.5]
scene.add("ball", kinetic.Shape.sphere(0.05), ball_pose)

print(scene.num_objects)  # 3
```

## Collision queries

```python
print(f"In collision: {scene.check_collision(start)}")
print(f"Clearance:    {scene.min_distance(start):.4f} m")
```

## Plan with a scene

Pass the scene to `Planner` -- it avoids all obstacles automatically:

```python
planner = kinetic.Planner(robot, scene=scene, timeout=5.0)

goal = kinetic.Goal.joints(np.array([1.0, -1.2, 0.8, -0.5, 0.7, 0.0]))
traj = planner.plan(start, goal)
print(f"{traj.num_waypoints} waypoints, {traj.duration:.2f}s")
```

Scenes are mutable: `scene.remove("ball")`, `scene.clear()`.

## Constrained planning

Constraints restrict the motion during planning:

```python
# Keep EE level (Z-axis up), 0.1 rad tolerance
keep_level = kinetic.Constraint.orientation(
    link="tool0", axis=[0.0, 0.0, 1.0], tolerance=0.1,
)

# Restrict EE height to [0.45m, 1.2m]
stay_above = kinetic.Constraint.position_bound(
    link="tool0", axis="z", min=0.45, max=1.2,
)

goal = kinetic.Goal.joints(np.array([0.8, -1.0, 0.5, -1.0, 0.8, 0.0]))
traj = planner.plan_constrained(
    start, goal, constraints=[keep_level, stay_above],
)
print(f"Constrained: {traj.num_waypoints} waypoints")
```

Other constraint types:

```python
kinetic.Constraint.joint(joint_index=2, min=-0.5, max=0.5)
kinetic.Constraint.visibility(
    sensor_link="camera_link", target=[0.5, 0.0, 0.4], cone_angle=0.3,
)
```

## Cartesian planning

Move the end-effector in a straight line through task space:

```python
config = kinetic.CartesianConfig(
    max_step=0.005,         # 5mm interpolation step
    jump_threshold=1.4,     # reject large joint jumps
    avoid_collisions=True,
    collision_margin=0.02,  # 2cm safety margin
)

target = robot.fk(np.array([0.6, -1.1, 0.6, -0.8, 0.6, 0.0]))
result = planner.plan_cartesian(start, kinetic.Goal.pose(target), config=config)

print(f"Achieved {result.fraction * 100:.1f}% of the path")
print(f"{result.trajectory.num_waypoints} waypoints")
```

`fraction` is 1.0 when the full straight line is achievable. Less than 1.0
means the robot hit a singularity or collision before reaching the goal.

## Allowed collisions

```python
scene.allow_collision("robot_base", "table")
# plan as usual -- base-table contact is ignored
scene.disallow_collision("robot_base", "table")
```

## Error handling

```python
try:
    traj = planner.plan(start, goal)
except RuntimeError as e:
    print(f"Planning failed: {e}")

try:
    solution = robot.ik(target)
except ValueError as e:
    print(f"IK failed: {e}")
```

## Next

- [Trajectories and NumPy](trajectory-numpy.md) -- export, time-parameterize, plot
- [Servo Control](servo-control.md) -- real-time reactive control
