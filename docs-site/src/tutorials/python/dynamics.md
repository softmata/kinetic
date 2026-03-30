# Dynamics in Python

Gravity compensation, inverse dynamics, forward dynamics, and mass matrix
computation via the featherstone articulated-body algorithm bridge.

## Setup

```python
import kinetic
import numpy as np

robot = kinetic.Robot("franka_panda")
dyn = kinetic.Dynamics(robot)
```

## Gravity compensation

Torques needed to hold the robot stationary at a given configuration
(countering gravity):

```python
q = np.array([0.0, -0.3, 0.0, -1.5, 0.0, 1.2, 0.0])
tau_gravity = dyn.gravity_compensation(q)

print("Gravity torques (Nm):")
for name, t in zip(robot.joint_names, tau_gravity):
    print(f"  {name}: {t:+.3f}")
```

Shoulder joints carry more weight -- you'll see larger torques there.

## Inverse dynamics

Given position, velocity, and acceleration, compute the required torques:

```python
q   = np.array([0.0, -0.3, 0.0, -1.5, 0.0, 1.2, 0.0])
qd  = np.zeros(7)          # zero velocity (static)
qdd = np.zeros(7)          # zero acceleration

tau = dyn.inverse_dynamics(q, qd, qdd)
# With zero velocity and acceleration, this equals gravity compensation
print(f"ID torques: {np.round(tau, 3)}")
```

With non-zero acceleration (e.g., a step response):

```python
qdd = np.array([0, 0, 0, 1.0, 0, 0, 0])  # accelerate joint 4
tau = dyn.inverse_dynamics(q, qd, qdd)
print(f"Torques with acceleration: {np.round(tau, 3)}")
```

## Forward dynamics

Given position, velocity, and applied torques, compute the resulting
accelerations:

```python
tau_applied = dyn.gravity_compensation(q)  # exactly counteract gravity
qdd = dyn.forward_dynamics(q, qd, tau_applied)
print(f"Accelerations: {np.round(qdd, 6)}")
# Should be near-zero (gravity compensated, no external forces)
```

Apply extra torque to see the robot accelerate:

```python
tau_extra = dyn.gravity_compensation(q)
tau_extra[3] += 5.0  # 5 Nm extra on joint 4

qdd = dyn.forward_dynamics(q, qd, tau_extra)
print(f"Joint 4 acceleration: {qdd[3]:.3f} rad/s^2")
```

## Mass matrix

The joint-space inertia matrix M(q) -- symmetric positive-definite:

```python
M = dyn.mass_matrix(q)
print(f"Mass matrix shape: {M.shape}")   # (7, 7) for Panda
print(f"Diagonal (inertias): {np.diag(M).round(3)}")
print(f"Symmetric: {np.allclose(M, M.T)}")
print(f"Positive definite: {np.all(np.linalg.eigvalsh(M) > 0)}")
```

## Trajectory feasibility check

Before executing on real hardware, verify that the required torques stay
within the robot's effort limits:

```python
# Plan and time-parameterize
planner = kinetic.Planner(robot)
start = np.zeros(7)
goal = kinetic.Goal.joints(np.array([0.5, -0.5, 0.3, -1.0, 0.2, 1.0, 0.3]))
traj = planner.plan(start, goal)

# Check torques at each waypoint
times, positions, velocities = traj.to_numpy()
max_torques = np.zeros(robot.dof)

for i in range(len(times)):
    q = positions[i]
    qd = velocities[i]
    # Approximate acceleration from finite differences
    if i == 0 or i == len(times) - 1:
        qdd = np.zeros(robot.dof)
    else:
        dt = times[i+1] - times[i-1]
        qdd = (velocities[i+1] - velocities[i-1]) / dt if dt > 0 else np.zeros(robot.dof)

    tau = dyn.inverse_dynamics(q, qd, qdd)
    max_torques = np.maximum(max_torques, np.abs(tau))

print("Max torques per joint (Nm):")
for name, t in zip(robot.joint_names, max_torques):
    print(f"  {name}: {t:.2f}")
```

## Next

- [GPU Acceleration](gpu-acceleration.md) -- parallel trajectory optimization
- [Hardware Execution](hardware-execution.md) -- deploy to real robots
- [Servo Control](servo-control.md) -- reactive control at 500 Hz
