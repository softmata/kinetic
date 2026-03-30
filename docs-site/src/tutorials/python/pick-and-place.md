# Pick and Place in Python

Full pick-and-place workflow: scene, grasps, attach/detach, and Task composition.

```mermaid
graph LR
    Home[Home] -->|Plan| Pre[Pre-grasp<br/>above object]
    Pre -->|Cartesian| Grasp[Grasp pose]
    Grasp -->|Close gripper| Attach[Attach object]
    Attach -->|Plan| Transport[Transport<br/>to place]
    Transport -->|Cartesian| Place[Place pose]
    Place -->|Open gripper| Detach[Detach object]
    Detach -->|Plan| Home2[Home]

    style Home fill:#4a9eff,color:#fff
    style Grasp fill:#ff6b6b,color:#fff
    style Place fill:#3ddc84,color:#000
    style Home2 fill:#4a9eff,color:#fff
```

## Setup

```python
import kinetic
import numpy as np

robot = kinetic.Robot("ur5e")
home = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
```

## Build the workspace

```python
scene = kinetic.Scene(robot)

table_pose = np.eye(4)
table_pose[2, 3] = 0.4
scene.add("table", kinetic.Shape.cuboid(0.5, 0.5, 0.01), table_pose)

# Object to pick: 5cm x 5cm x 8cm box on the table
box_half = [0.025, 0.025, 0.04]
box_pose = np.eye(4)
box_pose[0, 3], box_pose[2, 3] = 0.4, 0.45
scene.add("box", kinetic.Shape.cuboid(*box_half), box_pose)

# Place location: 30cm to the right
place_pose = np.eye(4)
place_pose[:3, 3] = [0.4, 0.3, 0.45]
```

## Generate grasp candidates

```python
gripper = kinetic.GripperType.parallel(max_opening=0.08, finger_depth=0.03)
generator = kinetic.GraspGenerator(gripper)

grasps = generator.from_shape(
    shape_type="cuboid",
    dimensions=box_half,
    object_pose=box_pose,
    num_candidates=100,
)
print(f"{len(grasps)} grasp candidates")

grasps.sort(key=lambda g: g.quality_score, reverse=True)
best = grasps[0]
print(f"Best quality: {best.quality_score:.3f}")
print(f"Grasp pose:\n{best.pose()}")
```

## Plan approach

```python
planner = kinetic.Planner(robot, scene=scene, timeout=5.0)

# Pre-grasp: 10cm above
pre_grasp = best.pose().copy()
pre_grasp[2, 3] += 0.10

traj_approach = planner.plan(home, kinetic.Goal.pose(pre_grasp))
print(f"Approach: {traj_approach.num_waypoints} wp, {traj_approach.duration:.2f}s")

# Straight-line descent to grasp
approach_end = traj_approach.sample(traj_approach.duration)
config = kinetic.CartesianConfig(max_step=0.002, collision_margin=0.01)
descent = planner.plan_cartesian(
    approach_end, kinetic.Goal.pose(best.pose()), config=config,
)
print(f"Descent: {descent.fraction * 100:.0f}% achieved")
```

## Pick: attach the object

After the gripper closes, attach the object to the robot link:

```python
grasp_tf = np.eye(4)
grasp_tf[2, 3] = -0.04  # object center 4cm below flange

scene.attach("box", kinetic.Shape.cuboid(*box_half), grasp_tf, link_name="tool0")
print(f"Attached: {scene.num_attached}")
```

## Transport

The planner includes the attached object in collision checks:

```python
grasp_joints = descent.trajectory.sample(descent.trajectory.duration)
traj_transport = planner.plan(grasp_joints, kinetic.Goal.pose(place_pose))
print(f"Transport: {traj_transport.duration:.2f}s")
```

## Place: detach the object

```python
scene.detach("box", place_pose)
print(f"Attached: {scene.num_attached}")  # 0
```

## Return home

```python
place_joints = traj_transport.sample(traj_transport.duration)
traj_return = planner.plan(place_joints, kinetic.Goal.joints(home))
print(f"Return: {traj_return.duration:.2f}s")
```

## Compose with Task

For cleaner code, use `Task` to define each stage. `Task.pick` and `Task.place`
handle approach/retreat/gripper automatically:

```python
approach = kinetic.Approach([0, 0, -1], 0.10)  # 10cm straight down
retreat  = kinetic.Approach([0, 0,  1], 0.05)  # 5cm straight up

# Pick: approach, close gripper, retreat
pick = kinetic.Task.pick(
    robot, scene, "box",
    grasp_poses=[best.pose()],
    approach=approach, retreat=retreat,
    gripper_open=0.08, gripper_close=0.0,
)

# Place: approach, open gripper, retreat
place = kinetic.Task.place(
    robot, scene, "box",
    target_pose=place_pose,
    approach=approach, retreat=retreat,
    gripper_open=0.08,
)

# Compose into a full sequence
seq = kinetic.Task.sequence([
    kinetic.Task.move_to(robot, kinetic.Goal.pose(pre_grasp)),
    pick,
    kinetic.Task.move_to(robot, kinetic.Goal.pose(place_pose)),
    place,
    kinetic.Task.move_to(robot, kinetic.Goal.named("home")),
])

sol = seq.plan(home)
print(f"Full sequence: {sol.num_stages} stages, {sol.total_duration:.2f}s")
print(f"Stages: {sol.stage_names}")
```

Individual tasks also work standalone:

```python
t1 = kinetic.Task.move_to(robot, kinetic.Goal.named("home"))
t2 = kinetic.Task.gripper(width=0.08)

sol = t1.plan(home)
print(f"{sol.num_stages} stages, {sol.total_duration:.2f}s")
```

## Verify with LogExecutor

```python
executor = kinetic.LogExecutor(rate_hz=500.0)
executor.execute(traj_approach)
print(f"{executor.num_commands} commands logged")
executor.clear()
```

## Suction gripper

```python
suction = kinetic.GripperType.suction(cup_radius=0.02)
gen = kinetic.GraspGenerator(suction)
grasps = gen.from_shape("cuboid", box_half, box_pose, num_candidates=50)
```

## Next

- [Python Quickstart](quickstart.md) -- review the basics
- [Planning in Python](planning.md) -- constraints and Cartesian paths
