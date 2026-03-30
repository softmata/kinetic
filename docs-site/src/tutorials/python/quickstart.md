# Python Quickstart

Get up and running with kinetic from Python in under 5 minutes: load a robot,
compute FK/IK, and plan your first motion -- all with numpy arrays.

## Install

```bash
pip install kinetic numpy
```

kinetic ships as a native extension (PyO3). No Rust toolchain needed.

## The standard import

```python
import kinetic
import numpy as np
```

All functions that accept or return arrays use `numpy.ndarray`.

## Load a robot

kinetic ships 54 built-in robots. Load one by name:

```python
robot = kinetic.Robot("ur5e")

print(robot.name)        # "ur5e"
print(robot.dof)         # 6
print(robot.joint_names) # ['shoulder_pan', 'shoulder_lift', ...]
```

Other built-ins: `franka_panda`, `kuka_iiwa7`, `kinova_gen3`. Custom URDFs:

```python
robot = kinetic.Robot.from_urdf("/path/to/my_robot.urdf")
```

## Forward kinematics

Pass joint angles as a numpy array, get a 4x4 SE(3) matrix back:

```python
joints = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
pose = robot.fk(joints)

print(pose.shape)  # (4, 4)
print(pose[:3, 3]) # XYZ position of the end-effector

J = robot.jacobian(joints)
print(J.shape)  # (6, 6)
```

## Inverse kinematics

IK finds joint angles that reach a target pose (DLS, 8 random restarts):

```python
target_pose = robot.fk(np.array([0.5, -1.0, 0.5, -0.5, 0.5, 0.0]))
solution = robot.ik(target_pose)
print(solution)  # array of 6 joint values

# With a seed for deterministic results
solution = robot.ik(target_pose, seed=np.zeros(6))
```

For full control, `ik_config` returns a dict with convergence info:

```python
result = robot.ik_config(
    target_pose, solver="auto", max_iterations=300, num_restarts=10,
)
print(result["converged"])       # True
print(result["joints"])          # numpy array
print(result["position_error"])  # meters
```

## Named poses and manipulability

```python
print(robot.named_poses)  # ['home', 'ready', 'zero', ...]
home = robot.named_pose("home")

m = robot.manipulability(joints)
print(f"manipulability = {m:.4f}")  # 0 = singular
```

## Batch operations

Process many configurations at once -- much faster than a Python loop:

```python
configs = np.random.uniform(-1.5, 1.5, size=(100, robot.dof))
poses = robot.batch_fk(configs)     # 100 FK calls in one shot
print(len(poses), poses[0].shape)   # 100, (4, 4)

targets = [robot.fk(c) for c in configs[:10]]
results = robot.batch_ik(targets, solver="auto")
for r in results:
    if r is not None:
        print(f"converged={r['converged']}, error={r['position_error']:.6f}")
```

## Plan a motion

The fastest way to plan -- one function call:

```python
start = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
goal  = kinetic.Goal.joints(np.array([0.5, -1.0, 0.5, -0.5, 0.5, 0.0]))

traj = kinetic.plan("ur5e", start, goal)
print(f"{traj.num_waypoints} waypoints, {traj.duration:.2f}s")
```

For repeated planning, create a `Planner` once and reuse it:

```python
planner = kinetic.Planner(robot)
traj = planner.plan(start, goal)
```

Choose from 9 planner algorithms:

```python
planner = kinetic.Planner(robot, planner_type="rrt_star")  # asymptotically optimal
# Options: rrt_connect (default), rrt_star, bi_rrt_star, bitrrt, est, kpiece, prm, gcs
```

Goals can be joint-space, Cartesian, or named:

```python
goal_pose  = kinetic.Goal.pose(target_pose)  # 4x4 numpy matrix
goal_named = kinetic.Goal.named("home")      # named configuration
```

## Sample the trajectory

Trajectories are time-parameterized. Sample at any time:

```python
for t in np.linspace(0, traj.duration, 50):
    joints_at_t = traj.sample(t)
    print(f"t={t:.3f}s  joints={joints_at_t}")
```

## Using help()

Every class has docstrings with `text_signature`, so built-in help works:

```python
help(kinetic.Robot)
help(kinetic.Robot.ik)
help(kinetic.Planner.plan)
```

## Next

- [Planning in Python](planning.md) -- scenes, constraints, Cartesian planning
- [Trajectories and NumPy](trajectory-numpy.md) -- export, visualize, validate
- [Dynamics](dynamics.md) -- gravity compensation, inverse/forward dynamics
- [GPU Acceleration](gpu-acceleration.md) -- parallel trajectory optimization
- [Hardware Execution](hardware-execution.md) -- deploy to real robots
