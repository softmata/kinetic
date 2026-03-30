# GPU Acceleration in Python

Parallel-seed trajectory optimization and batch collision checking using
wgpu compute shaders. Runs on NVIDIA (Vulkan), AMD (Vulkan), Intel (Vulkan),
and Apple Silicon (Metal). Automatically falls back to CPU if no GPU is found.

## Check GPU availability

```python
import kinetic
import numpy as np

print(f"GPU available: {kinetic.GpuOptimizer.gpu_available()}")
```

## GPU trajectory optimizer

The optimizer runs hundreds of trajectory seeds in parallel and returns the
best one (cuRobo-style):

```python
robot = kinetic.Robot("ur5e")

# Create optimizer (auto-detects GPU, falls back to CPU)
opt = kinetic.GpuOptimizer(robot, preset="balanced")
print(f"Backend: {'GPU' if opt.is_gpu else 'CPU'}")
```

Three presets trade off speed vs quality:

| Preset | Seeds | Iterations | SDF Resolution | Best for |
|--------|-------|-----------|----------------|----------|
| `speed` | 32 | 30 | 5cm | Real-time replanning |
| `balanced` | 128 | 100 | 2cm | General use |
| `quality` | 512 | 200 | 1cm | Offline, quality matters |

```python
opt_fast = kinetic.GpuOptimizer(robot, preset="speed")
opt_quality = kinetic.GpuOptimizer(robot, preset="quality")

# Custom: override specific parameters
opt_custom = kinetic.GpuOptimizer(robot, num_seeds=256, iterations=150)
```

## Optimize a trajectory

```python
start = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
goal  = np.array([1.0, -0.8, 0.6, -0.5, 0.8, 0.3])

traj = opt.optimize(start, goal)
print(f"{traj.num_waypoints} waypoints")
```

## Optimize with obstacles

Pass a scene or raw obstacle spheres:

```python
# Option 1: from a Scene
scene = kinetic.Scene(robot)
table_pose = np.eye(4)
table_pose[2, 3] = 0.4
scene.add("table", kinetic.Shape.cuboid(0.5, 0.5, 0.01), table_pose)

traj = opt.optimize(start, goal, scene=scene)
```

```python
# Option 2: raw obstacle spheres (N, 4) array of [x, y, z, radius]
obstacles = np.array([
    [0.4, 0.0, 0.5, 0.05],  # sphere at (0.4, 0, 0.5) radius 0.05
    [0.3, 0.2, 0.6, 0.08],
])
traj = opt.optimize(start, goal, obstacle_spheres=obstacles)
```

## Time-parameterize the result

The optimizer returns a geometric path. Apply time parameterization before
execution:

```python
vel_limits = np.array(robot.velocity_limits)
acc_limits = np.array(robot.acceleration_limits)

traj_timed = traj.time_parameterize("totp", vel_limits, acc_limits)
print(f"Duration: {traj_timed.duration:.3f}s")
```

## GPU batch collision checking

Check hundreds of configurations at once -- much faster than checking one
at a time:

```python
checker = kinetic.GpuCollisionChecker(robot, scene)
print(f"Backend: {'GPU' if checker.is_gpu else 'CPU'}")

# Generate random configurations
configs = np.random.uniform(-1.5, 1.5, size=(1000, robot.dof))

results = checker.check_batch(configs)
n_collisions = sum(results["in_collision"])
print(f"{n_collisions}/{len(configs)} configs in collision")
print(f"Min clearance: {min(results['min_distances']):.4f} m")
```

Single-configuration check:

```python
colliding, distance = checker.check_single(start)
print(f"In collision: {colliding}, clearance: {distance:.4f} m")
```

## Combining GPU optimization with planning

Use GPU optimization to refine an RRT path:

```python
# Step 1: fast RRT-Connect for initial path
planner = kinetic.Planner(robot, scene=scene)
rrt_traj = planner.plan(start, kinetic.Goal.joints(goal), time_parameterize=False)

# Step 2: GPU refinement with warm-start (future API)
# For now, just compare quality
gpu_traj = opt.optimize(start, goal, scene=scene)
print(f"RRT: {rrt_traj.num_waypoints} waypoints")
print(f"GPU: {gpu_traj.num_waypoints} waypoints")
```

## Next

- [Hardware Execution](hardware-execution.md) -- deploy trajectories to real robots
- [Dynamics](dynamics.md) -- torque feasibility checking
- [Trajectories and NumPy](trajectory-numpy.md) -- export and visualize
