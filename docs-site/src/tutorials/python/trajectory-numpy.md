# Trajectories and NumPy

Export trajectories to numpy arrays, apply time parameterizations, validate
against joint limits, and plot with matplotlib.

## Setup

```python
import kinetic
import numpy as np

robot = kinetic.Robot("ur5e")
start = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
goal  = kinetic.Goal.joints(np.array([1.0, -0.8, 0.6, -0.5, 0.8, 0.3]))

planner = kinetic.Planner(robot)
traj = planner.plan(start, goal)
```

## Export to numpy arrays

`to_numpy()` returns three arrays -- times, positions, and velocities:

```python
times, positions, velocities = traj.to_numpy()

print(times.shape)       # (N,)   -- timestamps in seconds
print(positions.shape)   # (N, 6) -- joint positions per waypoint
print(velocities.shape)  # (N, 6) -- joint velocities per waypoint
```

## Sample at arbitrary times

`sample()` interpolates at any time. Out-of-range values are clamped:

```python
dt = 0.01
t = 0.0
while t <= traj.duration:
    joints = traj.sample(t)
    print(f"t={t:.3f}  q={np.round(joints, 4)}")
    t += dt
```

## Time parameterization

Assign velocities and accelerations that respect physical limits. Four
profiles are available:

```python
vel_limits = np.array(robot.velocity_limits)
acc_limits = np.array(robot.acceleration_limits)

# Trapezoidal (bang-coast-bang)
traj_trap = traj.time_parameterize("trapezoidal", vel_limits, acc_limits)
print(f"Trapezoidal:  {traj_trap.duration:.3f}s")

# Time-Optimal (TOTP -- fastest possible)
traj_totp = traj.time_parameterize("totp", vel_limits, acc_limits)
print(f"TOTP:         {traj_totp.duration:.3f}s")

# Jerk-limited (smooth acceleration)
jerk_limits = acc_limits * 10.0
traj_smooth = traj.time_parameterize(
    "jerk_limited", vel_limits, acc_limits, jerk_limits=jerk_limits,
)
print(f"Jerk-limited: {traj_smooth.duration:.3f}s")

# Cubic spline (C2 continuous)
traj_spline = traj.time_parameterize("cubic_spline", vel_limits, acc_limits)
print(f"Cubic spline: {traj_spline.duration:.3f}s")
```

## Validate against limits

Returns a list of violation dicts -- empty means valid:

```python
pos_lower = np.full(6, -6.28)
pos_upper = np.full(6,  6.28)

violations = traj_trap.validate(pos_lower, pos_upper, vel_limits, acc_limits)
if not violations:
    print("Trajectory is valid")
else:
    for v in violations:
        print(f"Violation: waypoint {v['waypoint']}, {v['type']}, "
              f"joint {v['joint']}, value {v['value']:.4f}")
```

## Blend two trajectories

Smooth transition between consecutive motions:

```python
goal2 = kinetic.Goal.joints(start)
traj2 = planner.plan(np.array([1.0, -0.8, 0.6, -0.5, 0.8, 0.3]), goal2)

blended = traj.blend(traj2, blend_duration=0.2)
print(f"Blended: {blended.duration:.3f}s, {blended.num_waypoints} waypoints")
```

## Plot with matplotlib

```python
import matplotlib.pyplot as plt

times, positions, velocities = traj_trap.to_numpy()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

for j in range(robot.dof):
    ax1.plot(times, positions[:, j], label=robot.joint_names[j])
ax1.set_ylabel("Position (rad)")
ax1.legend(fontsize=8)
ax1.set_title("UR5e Joint Positions")
ax1.grid(True, alpha=0.3)

for j in range(robot.dof):
    ax2.plot(times, velocities[:, j], label=robot.joint_names[j])
ax2.set_ylabel("Velocity (rad/s)")
ax2.set_xlabel("Time (s)")
ax2.legend(fontsize=8)
ax2.set_title("UR5e Joint Velocities")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("trajectory_plot.png", dpi=150)
plt.show()
```

## Export and import (JSON / CSV)

Save trajectories to files and load them back:

```python
# Export
json_str = traj_trap.to_json()
csv_str = traj_trap.to_csv()

# Write to file
with open("trajectory.json", "w") as f:
    f.write(json_str)
with open("trajectory.csv", "w") as f:
    f.write(csv_str)

# Import
traj_loaded = kinetic.Trajectory.from_json(json_str)
traj_loaded = kinetic.Trajectory.from_csv(csv_str)

print(f"Loaded: {traj_loaded.num_waypoints} wp, {traj_loaded.duration:.3f}s")
```

## Waypoint access as lists

If you need plain Python lists instead of numpy:

```python
waypoints = traj.positions()  # list[list[float]]
for i, wp in enumerate(waypoints[:3]):
    print(f"Waypoint {i}: {wp}")
```

## Next

- [Servo Control](servo-control.md) -- real-time reactive control
- [Pick and Place](pick-and-place.md) -- full manipulation workflow
