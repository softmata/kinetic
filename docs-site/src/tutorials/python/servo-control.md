# Servo Control in Python

High-frequency Cartesian twist and joint-jog commands for teleoperation and
reactive manipulation. Covers the `Servo` controller with collision avoidance
and `RMP` for multi-policy reactive control at 500 Hz.

## Setup

```python
import kinetic
import numpy as np

robot = kinetic.Robot("franka_panda")

scene = kinetic.Scene(robot)
table_pose = np.eye(4)
table_pose[2, 3] = 0.3
scene.add("table", kinetic.Shape.cuboid(0.5, 0.5, 0.01), table_pose)
```

## Create a servo controller

```python
servo = kinetic.Servo(robot, scene, rate_hz=500.0)
```

## Send twist commands

`send_twist` takes a 6D array `[vx, vy, vz, wx, wy, wz]`:

```python
twist = np.array([0.05, 0.0, 0.0, 0.0, 0.0, 0.0])  # 5 cm/s in +X
cmd = servo.send_twist(twist)
print(cmd["positions"])   # new joint positions (numpy array)
print(cmd["velocities"])  # new joint velocities (numpy array)
```

## 500 Hz servo loop

```python
dt = 1.0 / servo.rate_hz
for i in range(1000):  # 2 seconds
    t = i * dt
    vx = 0.03 * np.cos(2 * np.pi * t)
    vy = 0.03 * np.sin(2 * np.pi * t)

    cmd = servo.send_twist(np.array([vx, vy, 0, 0, 0, 0]))

    if i % 100 == 0:
        state = servo.state()
        pos = state["ee_pose"][:3, 3]
        print(f"t={t:.2f}s  EE=[{pos[0]:.3f}, {pos[1]:.3f}, {pos[2]:.3f}]")
```

## Joint jogging and state

```python
cmd = servo.send_joint_jog(joint_index=3, velocity=0.1)

state = servo.state()
print(state["joint_positions"])   # numpy (dof,)
print(state["ee_pose"])           # numpy (4, 4)
print(state["manipulability"])    # float
print(state["near_singularity"])  # bool
print(state["near_collision"])    # bool
```

## Set initial state

```python
home = robot.named_pose("home")
if home is not None:
    servo.set_state(positions=home, velocities=np.zeros(robot.dof))
```

## RMP: multi-policy reactive control

Riemannian Motion Policies combine multiple objectives into one consistent
joint-space command via metric-weighted averaging:

```python
rmp = kinetic.RMP(robot)

target = robot.fk(np.array([0.3, -0.5, 0.2, -1.5, 0.1, 1.0, 0.0]))
rmp.add(kinetic.Policy.reach_target(target, gain=10.0))
rmp.add(kinetic.Policy.avoid_obstacles(scene, influence_distance=0.15, gain=25.0))
rmp.add(kinetic.Policy.joint_limit_avoidance(margin=0.1, gain=15.0))
rmp.add(kinetic.Policy.singularity_avoidance(threshold=0.02, gain=5.0))
rmp.add(kinetic.Policy.avoid_self_collision(gain=20.0))
rmp.add(kinetic.Policy.damping(coefficient=0.5))

print(f"RMP: {rmp.num_policies} policies, {rmp.dof} DOF")
```

## RMP control loop

```python
dt = 0.002  # 500 Hz
q  = np.zeros(robot.dof)
qd = np.zeros(robot.dof)

for step in range(1000):
    cmd = rmp.compute(q, qd, dt)
    q, qd = cmd["positions"], cmd["velocities"]

    if step % 100 == 0:
        ee = robot.fk(q)[:3, 3]
        m = robot.manipulability(q)
        print(f"step {step:4d}  EE=[{ee[0]:.3f}, {ee[1]:.3f}, {ee[2]:.3f}]  "
              f"manip={m:.4f}")
```

## Next

- [Pick and Place](pick-and-place.md) -- full manipulation workflow
- [Trajectories and NumPy](trajectory-numpy.md) -- export and visualize paths
