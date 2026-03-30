# Architecture

Crate structure, dependency layers, and design principles.

## Crate Dependency Diagram

```
                         kinetic  (top-level re-exports + prelude)
                            |
        +-------------------+-------------------+
        |                   |                   |
   kinetic-task      kinetic-gpu        horus-kinetic
        |                   |                   |
        +--------+----------+         (HORUS bridge, optional)
                 |
          kinetic-planning
                 |
     +-----------+-----------+
     |           |           |
kinetic-    kinetic-    kinetic-
reactive    scene       trajectory
     |           |           |
     +-----------+-----------+
                 |
          kinetic-collision
                 |
          kinetic-kinematics
                 |
          kinetic-robot
                 |
          kinetic-core
```

Additional crates not shown in the main dependency chain:

```
kinetic-execution   depends on: kinetic-trajectory, kinetic-robot
kinetic-grasp       depends on: kinetic-kinematics, kinetic-scene
kinetic-python      depends on: all of the above (PyO3 bindings)
kinetic-viewer      depends on: kinetic-robot, kinetic-core
kinetic-dynamics    depends on: kinetic-core, kinetic-robot
```

## Layer Descriptions

### Layer 0: kinetic-core

Foundation types shared by all crates. Zero external dependencies beyond
nalgebra, serde, and thiserror.

**Key types:** `Pose`, `JointValues`, `Trajectory`, `Goal`, `Constraint`,
`KineticError`, `PlannerConfig`, `FrameTree`, `Twist`, `Wrench`

### Layer 1: kinetic-robot

URDF/SRDF/MJCF/SDF parsing and robot model representation. Loads robot
descriptions and configuration files.

**Key types:** `Robot`, `Joint`, `Link`, `JointType`, `RobotConfig`

**Depends on:** kinetic-core, urdf-rs, roxmltree, toml

### Layer 2: kinetic-kinematics

Forward and inverse kinematics. Six IK solvers covering analytical (OPW,
Subproblem) and iterative (DLS, FABRIK) methods. Jacobian computation
and manipulability analysis.

**Key types:** `KinematicChain`, `IKSolver`, `IKConfig`, `IKSolution`

**Key functions:** `forward_kinematics`, `jacobian`, `solve_ik`,
`manipulability`, `solve_ik_batch`

**Depends on:** kinetic-core, kinetic-robot, nalgebra, rand

### Layer 3: kinetic-collision

SIMD-vectorized sphere-based collision detection with CAPT broadphase
acceleration. Generates sphere approximations of robot links from URDF
collision geometry.

**Key types:** `CollisionEnvironment`, `RobotSphereModel`, `SpheresSoA`,
`AllowedCollisionMatrix`, `ResolvedACM`

**Depends on:** kinetic-core, kinetic-robot, parry3d-f64

### Layer 4: kinetic-scene, kinetic-trajectory, kinetic-reactive

Scene management, trajectory processing, and reactive control. These
crates sit at the same layer and may depend on each other.

**kinetic-scene:** `Scene`, `Shape`, `PointCloudConfig`, `Octree`

**kinetic-trajectory:** `TimedTrajectory`, `TrajectoryValidator`, `totp`,
`trapezoidal`, `jerk_limited`, `blend`

**kinetic-reactive:** `Servo`, `ServoConfig`, `RMP`, `JointCommand`

### Layer 5: kinetic-planning

Motion planning algorithms. The `Planner` facade auto-selects the right
algorithm based on goal type and configuration.

**Key types:** `Planner`, `PlannerType`, `PlanningResult`, `CartesianPlanner`,
`DualArmPlanner`, `PlanningPipeline`

**Depends on:** kinetic-core, kinetic-robot, kinetic-kinematics,
kinetic-collision, kinetic-scene

### Layer 6: kinetic-task, kinetic-gpu, kinetic-execution

High-level task planning, GPU optimization, and hardware execution.

**kinetic-task:** `Task`, `PickConfig`, `PlaceConfig`, `Approach`

**kinetic-gpu:** `GpuOptimizer`, `GpuConfig`, `CpuOptimizer`, `SignedDistanceField`

**kinetic-execution:** `CommandSink`, `RealTimeExecutor`, `ExecutionConfig`,
`SafetyWatchdog`

### Top Level: kinetic

Re-exports all sub-crates and provides the `prelude` module for convenience.

### Bridge: horus-kinetic

Optional HORUS integration. Built separately, not part of the main workspace.
Provides `PlannerNode`, `ServoNode`, `SceneNode`.

### Bindings: kinetic-python

Python bindings via PyO3. Built separately, not part of the main workspace.
Wraps all public APIs with numpy interop.

## Design Principles

### 1. Zero-Copy Where Possible

Joint values, trajectories, and poses are passed by reference through
the planning pipeline. Allocations happen during construction, not in
hot loops.

### 2. Fail Fast with Actionable Errors

Every error variant in `KineticError` tells the caller what went wrong
and whether to retry. The `is_retryable()` and `is_input_error()` methods
enable automated recovery strategies.

### 3. Safety Gates

The planner validates its own output before returning. Two safety gates
check every waypoint against joint limits and workspace bounds. This
catches planner bugs before they reach hardware.

### 4. Configuration Defaults That Work

Every config struct has a sensible `Default` implementation. Named presets
(`realtime()`, `offline()`, `safe()`) cover common scenarios. Users only
need to customize what differs from the defaults.

### 5. Layered Independence

Each crate is independently testable and usable. You can use
`kinetic-kinematics` for FK/IK without pulling in the planner.
You can use `kinetic-collision` without pulling in the scene.

### 6. Analytical Before Iterative

The IK solver auto-selection prefers analytical solvers (OPW, Subproblem)
over iterative ones (DLS, FABRIK). Analytical solvers are faster, more
reliable, and return all solutions instead of a single local minimum.

### 7. Hardware Abstraction via Traits

`CommandSink` and `FeedbackSource` are traits, not concrete types.
This allows the same execution code to work with any robot hardware,
simulator, or test double.
