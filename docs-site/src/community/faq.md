# FAQ

Frequently asked questions about kinetic.

## General

### Is kinetic compatible with ROS2?

Kinetic has zero ROS2 dependencies and does not require ROS2 to run. However,
the `horus-kinetic` bridge crate provides HORUS IPC integration, and a
`rmw_horus` bridge is planned for transparent ROS2 topic interop. You can
also use kinetic standalone and publish results to ROS2 topics via your own
bridge node.

### Can I use kinetic from Python only?

Yes. The `kinetic` Python package (via PyO3) exposes Robot, Planner, Scene,
Servo, Task, and Trajectory classes with numpy interop. Install with
`pip install kinetic` or build from source with `maturin develop`.

```python
import kinetic
import numpy as np

robot = kinetic.Robot("ur5e")
planner = kinetic.Planner(robot)
start = np.array([0.0, -1.57, 0.0, 0.0, 0.0, 0.0])
goal = kinetic.Goal.joints(np.array([0.5, -1.0, 0.5, 0.0, 0.5, 0.0]))
traj = planner.plan(start, goal)
```

### Is kinetic production-ready?

Kinetic is used in design partner deployments with real robots. It includes
trajectory validation, safety watchdogs, execution monitoring, and error
recovery mechanisms. See the Production Deployment guide for the full
pre-deployment checklist.

### Does kinetic require a GPU?

No. The GPU optimizer (`kinetic-gpu`) is optional and provides faster
trajectory optimization when a Vulkan/Metal-capable GPU is available.
All core functionality (FK, IK, planning, collision, trajectory) runs
on CPU. The `CpuOptimizer` provides the same API as `GpuOptimizer`
without GPU hardware.

### Does kinetic run on Windows?

Kinetic targets Linux and macOS. Windows support is not actively tested
or maintained. The core math and planning algorithms are platform-agnostic,
but hardware integration (shared memory, real-time scheduling) is
Linux/macOS only.

### How do I visualize trajectories?

Kinetic does not include a built-in visualizer. You can:

1. Export trajectories to CSV/JSON and visualize in Python (matplotlib)
2. Use the `kinetic-viewer` crate for basic 3D rendering
3. Connect to HORUS and use `horus-monitor` for real-time visualization
4. Use `horus-sim3d` for full 3D simulation with physics

### What license is kinetic under?

Apache-2.0. See the LICENSE file in the repository root.

## Robots

### Which robots are supported out of the box?

Kinetic ships with 52 built-in robot configurations covering robots from
Universal Robots, Franka, KUKA, ABB, Fanuc, Yaskawa, xArm, Kinova,
Trossen, and more. See the Supported Robots reference for the full list.

### Can I use a robot not in the built-in list?

Yes. Any robot with a URDF file can be loaded. Create a `kinetic.toml`
config file or load the URDF directly with `Robot::from_urdf()`. See
the Custom Robots guide for step-by-step instructions.

### How accurate is the IK for my robot?

Accuracy depends on the solver and the robot geometry:

- **OPW** (analytical): exact solution for 6-DOF spherical wrist robots
- **Subproblem** (analytical): exact for 6/7-DOF with intersecting wrist axes
- **DLS** (iterative): configurable tolerance, typically <0.1mm position error
- **FABRIK** (iterative): good position accuracy, weaker orientation accuracy

Check `solution.position_error` and `solution.orientation_error` after solving.

### Does kinetic support mobile manipulators?

Yes. Robots like Fetch (8-DOF) and TIAGo (8-DOF) are supported. The
planner auto-extracts the arm chain from the full robot model. Note
that `chain.dof` may be less than `robot.dof` for mobile manipulators.

## Planning

### Which planner should I use?

See the Planner Selection guide for a decision flowchart. Short answer:
RRT-Connect (default) for most tasks, EST/KPIECE for narrow passages,
RRT* for optimal paths, GPU optimizer for complex environments.

### Why does planning fail in cluttered environments?

Sampling-based planners struggle when free space is sparse. Try:

1. Increase timeout and iterations (`PlannerConfig::offline()`)
2. Use EST or KPIECE (better for narrow passages)
3. Reduce collision margin
4. Use the GPU optimizer (SDF-based, handles dense environments better)
5. Decompose the problem into intermediate waypoints

### Can I plan for two arms at once?

Yes. `DualArmPlanner` plans in the combined configuration space of both
arms with inter-arm collision avoidance. Three modes are available:
Independent, Synchronized, and Coordinated.

### What is the difference between `plan()` and `plan_with_scene()`?

`plan()` uses only the robot's self-collision model (no environment obstacles).
`plan_with_scene()` includes scene obstacles (tables, walls, point clouds)
in collision checking.

## Performance

### How fast is FK/IK?

Target performance on a modern x86-64 CPU:

| Operation | Time |
|-----------|------|
| FK (6-DOF) | <1 us |
| Jacobian (6-DOF) | <2 us |
| IK OPW (6-DOF) | <5 us |
| IK DLS (7-DOF) | 100-500 us |
| Collision check (SIMD) | <500 ns |
| RRT-Connect (simple) | <100 us |

### Is kinetic real-time capable?

The core planning and IK functions are deterministic-time with configurable
timeouts. `PlannerConfig::realtime()` uses a 10ms timeout. Servo mode runs
at 500 Hz. Kinetic does not allocate memory in hot paths after initialization.

## Integration

### How does kinetic compare to MoveIt2?

See the From MoveIt2 migration guide for a detailed comparison. Key
differences: kinetic is Rust (not C++), has no ROS dependency, includes
GPU optimization, and uses SIMD collision checking instead of FCL.

### Can I use kinetic with my own physics simulator?

Yes. Kinetic does not include a physics simulator. Plan trajectories
with kinetic and execute them in your simulator via `CommandSink` or
by exporting to CSV/JSON. The `horus-sim3d` simulator uses kinetic
trajectories natively.
