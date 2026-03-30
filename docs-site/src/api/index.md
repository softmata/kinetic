# API Reference

Links to generated Rust and Python API documentation.

## Rust API (cargo doc)

The full Rust API documentation is generated from inline doc comments
using `cargo doc`.

### Generating Locally

```bash
# Generate and open in browser
cargo doc --open

# Generate without opening
cargo doc

# Include private items (for development)
cargo doc --document-private-items

# Generate for a specific crate
cargo doc -p kinetic-planning
```

The generated documentation is written to `target/doc/kinetic/index.html`.

### Crate Index

| Crate | Description | Key Types |
|-------|-------------|-----------|
| [`kinetic`](https://docs.rs/kinetic) | Top-level re-exports and prelude | `prelude::*` |
| [`kinetic-core`](https://docs.rs/kinetic-core) | Shared types and errors | `Pose`, `Goal`, `KineticError`, `PlannerConfig` |
| [`kinetic-robot`](https://docs.rs/kinetic-robot) | Robot model and URDF parsing | `Robot`, `Joint`, `Link`, `RobotConfig` |
| [`kinetic-kinematics`](https://docs.rs/kinetic-kinematics) | FK, IK, Jacobian | `KinematicChain`, `IKSolver`, `IKConfig`, `IKSolution` |
| [`kinetic-collision`](https://docs.rs/kinetic-collision) | SIMD collision detection | `CollisionEnvironment`, `RobotSphereModel`, `AllowedCollisionMatrix` |
| [`kinetic-scene`](https://docs.rs/kinetic-scene) | Planning scene management | `Scene`, `Shape`, `PointCloudConfig` |
| [`kinetic-planning`](https://docs.rs/kinetic-planning) | Motion planners | `Planner`, `PlannerType`, `PlanningResult`, `CartesianPlanner` |
| [`kinetic-trajectory`](https://docs.rs/kinetic-trajectory) | Time parameterization | `TimedTrajectory`, `TrajectoryValidator` |
| [`kinetic-reactive`](https://docs.rs/kinetic-reactive) | Servo and RMP | `Servo`, `ServoConfig`, `RMP` |
| [`kinetic-task`](https://docs.rs/kinetic-task) | Task planning | `Task`, `PickConfig`, `PlaceConfig` |
| [`kinetic-gpu`](https://docs.rs/kinetic-gpu) | GPU optimization | `GpuOptimizer`, `GpuConfig`, `CpuOptimizer` |
| [`kinetic-execution`](https://docs.rs/kinetic-execution) | Trajectory execution | `CommandSink`, `RealTimeExecutor`, `ExecutionConfig` |
| [`kinetic-grasp`](https://docs.rs/kinetic-grasp) | Grasp generation | `GraspGenerator`, `GraspConfig`, `GripperType` |

## Python API

The Python bindings expose kinetic's core functionality with numpy interop.

### Installation

```bash
# From source (requires Rust toolchain)
cd crates/kinetic-python
maturin develop --release

# Or with pip (when published)
pip install kinetic
```

### Module Contents

```python
import kinetic

# Robot & Kinematics
kinetic.Robot(name)                          # Load robot (54 built-in configs)
robot.fk(joints)                             # Forward kinematics → 4x4
robot.ik(target_4x4)                         # Inverse kinematics → joints
robot.ik_config(target, solver="opw", ...)   # IK with full config
robot.jacobian(joints)                       # 6×DOF Jacobian
robot.manipulability(joints)                 # Manipulability index
robot.batch_fk(configs_NxDOF)               # Batch FK → list of 4x4
robot.batch_ik(target_list)                  # Batch IK → list of dicts

# Planning (9 planner types)
kinetic.Planner(robot, planner_type="rrt_star")  # rrt_connect, rrt_star, bi_rrt_star, bitrrt, est, kpiece, prm, gcs
kinetic.Goal.joints(array)                   # Joint-space goal
kinetic.Goal.pose(target_4x4)                # Cartesian pose goal
kinetic.Goal.named(name)                     # Named pose goal
kinetic.Constraint.orientation(...)          # Orientation constraint
kinetic.CartesianConfig()                    # Cartesian planning config
kinetic.DualArmPlanner(robot, "left", "right")  # Dual-arm planning
kinetic.MoveGroup(urdf_string)               # MoveIt-style API

# Trajectory
traj.sample(t)                               # Interpolate at time t
traj.to_numpy()                              # Export (times, pos, vel)
traj.time_parameterize("totp", vel, acc)     # Time parameterization
traj.blend(other, 0.1)                       # Smooth blending
traj.validate(lo, hi, vel, acc)              # Limit validation
traj.to_json() / traj.to_csv()              # Export
Trajectory.from_json(s) / from_csv(s)        # Import

# Scene
kinetic.Scene(robot)                         # Collision scene
kinetic.Shape.cuboid(x, y, z)                # Shapes

# Reactive Control
kinetic.Servo(robot, scene)                  # 500Hz servo
kinetic.RMP(robot)                           # Multi-policy RMP
kinetic.Policy.reach_target(target, gain)    # 6 policy types

# Task Planning
kinetic.Task.move_to(robot, goal)            # Move task
kinetic.Task.pick(robot, scene, ...)         # Pick task
kinetic.Task.place(robot, scene, ...)        # Place task
kinetic.Task.sequence([t1, t2, t3])          # Sequence
kinetic.Task.gripper(width)                  # Gripper command

# Grasp Generation
kinetic.GraspGenerator(gripper_type)         # Grasp candidates
kinetic.GripperType.parallel(0.08, 0.03)     # Parallel jaw

# Dynamics
kinetic.Dynamics(robot)                      # Featherstone bridge
dyn.gravity_compensation(q)                  # Hold torques
dyn.inverse_dynamics(q, qd, qdd)            # ID: τ = M*qdd + C*qd + g
dyn.forward_dynamics(q, qd, tau)             # FD: qdd = M⁻¹(τ - C*qd - g)
dyn.mass_matrix(q)                           # (DOF, DOF) mass matrix

# Execution
kinetic.SimExecutor()                        # Simulated (instant)
kinetic.LogExecutor()                        # Records all commands
kinetic.RealTimeExecutor(rate_hz=500)        # Hardware execution via callback
kinetic.RealTimeExecutor.safe(robot)         # With joint limit validation
kinetic.FrameTree()                          # TF2-like frame tree

# GPU Acceleration
kinetic.GpuOptimizer(robot, preset="balanced")  # Parallel-seed optimization
kinetic.GpuCollisionChecker(robot, scene)    # Batch collision checking
GpuOptimizer.gpu_available()                 # Check GPU availability

# One-liner
kinetic.plan("ur5e", start, goal)            # Plan in one call
```

### Type Stubs (.pyi)

Python type stubs are in `crates/kinetic-python/kinetic.pyi`. These provide
IDE autocompletion and type checking with mypy/pyright.

```bash
# Copy stubs to your project for IDE support
cp crates/kinetic-python/kinetic.pyi /path/to/your/project/
```

### Example

```python
import kinetic
import numpy as np

robot = kinetic.Robot("ur5e")
planner = kinetic.Planner(robot, planner_type="rrt_star")

start = np.array([0.0, -1.57, 0.0, -1.57, 0.0, 0.0])
goal = kinetic.Goal.named("home")
traj = planner.plan(start, goal)

# Sample the trajectory
for t in np.linspace(0, traj.duration, 50):
    joints = traj.sample(t)
    print(f"t={t:.3f}: {joints}")

# Export/import
json_str = traj.to_json()
traj2 = kinetic.Trajectory.from_json(json_str)

# Dynamics
dyn = kinetic.Dynamics(robot)
tau = dyn.gravity_compensation(start)

# GPU optimization
opt = kinetic.GpuOptimizer(robot, preset="speed")
fast_traj = opt.optimize(start, goal.joints)

# Hardware execution
def send(pos, vel):
    my_robot_driver.set_joints(pos)

executor = kinetic.RealTimeExecutor.safe(robot)
result = executor.execute(traj, send)
```
