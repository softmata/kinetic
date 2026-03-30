# KINETIC Architecture

## Overview

KINETIC is a Rust-native motion planning stack designed for performance and safety.
It replaces the C++/ROS-based MoveIt2 with a modern, modular architecture that
leverages Rust's type system and SIMD capabilities.

## Crate Hierarchy

```
kinetic (umbrella)
├── kinetic-core        # Shared types: Pose, Goal, JointValues, Constraint, Error
├── kinetic-robot       # URDF parsing, Robot model, TOML config, planning groups
├── kinetic-kinematics  # FK, Jacobian, IK (DLS, FABRIK, OPW), batch FK
├── kinetic-collision   # SIMD sphere collision, CAPT broadphase, ACM
├── kinetic-scene       # Scene graph, obstacle management, attached objects
├── kinetic-planning    # RRT-Connect, Cartesian planning, shortcutting, smoothing
├── kinetic-trajectory  # Time parameterization: TOTP, trapezoidal, S-curve, spline
├── kinetic-reactive    # RMP (Riemannian Motion Policies), servo control
├── kinetic-task        # Task-level planning: pick, place, move sequences
├── kinetic-grasp       # Grasp candidate generation, approach planning
└── kinetic-python      # Python bindings via PyO3 (separate build)
```

## Design Principles

### 1. Zero-Copy Where Possible

Types are designed to minimize allocations:
- `JointValues(Vec<f64>)` wraps a single allocation
- `Pose(Isometry3<f64>)` is stack-allocated (7 f64s)
- FK returns stack-allocated Pose, no heap allocation
- Batch FK operates on flat `&[f64]` slices

### 2. SIMD-Vectorized Collision Detection

The collision system uses Structure-of-Arrays (SoA) layout for SIMD:

```rust
pub struct SpheresSoA {
    pub x: Vec<f64>,
    pub y: Vec<f64>,
    pub z: Vec<f64>,
    pub r: Vec<f64>,
    pub link_id: Vec<usize>,
}
```

This layout enables 4-wide f64 SIMD operations for distance computations,
achieving sub-microsecond collision checks for typical robot configurations.

### 3. Modular Planning Pipeline

```
Goal → [IK if Pose goal] → [RRT-Connect] → [Path Shortcutting] → [Smoothing] → [Time Param] → Trajectory
```

Each stage is independently benchmarked and replaceable:
- **RRT-Connect**: Bidirectional rapidly-exploring random tree
- **Shortcutting**: Iterative random shortcutting with collision checking
- **Smoothing**: B-spline smoothing preserving collision-free guarantee
- **Time Parameterization**: TOTP (time-optimal) or trapezoidal profiles

### 4. Reactive Control via RMP

Riemannian Motion Policies combine multiple competing objectives:

```
RMP = Σ (Metric_i, Policy_i)
```

Policies include:
- **ReachTarget**: Attract end-effector to target pose
- **AvoidObstacles**: Repel from scene collision objects
- **JointLimitAvoidance**: Soft limits near joint boundaries
- **SingularityAvoidance**: Slow near singular configurations
- **Damping**: Velocity damping for stability

The RMP framework naturally resolves conflicts between objectives through
Riemannian metric weighting.

### 5. Scene Management

The Scene struct manages:
- **Static obstacles**: World-frame collision objects
- **Attached objects**: Objects grasped by the robot (move with links)
- **Allowed Collision Matrix**: Disable collision pairs (e.g., adjacent links)

## Key Data Flow

### Planning Pipeline

```
User Request
    │
    ▼
┌──────────────┐
│ Planner::new │ ← Robot model + kinematic chain auto-detection
└──────┬───────┘
       │
       ▼
┌──────────────┐     ┌───────────────┐
│ planner.plan │────►│ RRT-Connect   │ ← Collision checking via Scene
│  (start,goal)│     │ + Shortcutting│
└──────┬───────┘     │ + Smoothing   │
       │             └───────┬───────┘
       ▼                     │
┌──────────────┐             │
│ PlanningResult│◄───────────┘
│  .waypoints  │
│  .time       │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ totp() or    │ ← Time parameterization with velocity/accel limits
│ trapezoidal()│
└──────┬───────┘
       │
       ▼
┌──────────────┐
│TimedTrajectory│ ← Ready for execution: positions, velocities, accelerations
│  .sample_at()│
└──────────────┘
```

### Servo Control Loop

```
                    ┌───────────────────────────┐
                    │     500 Hz Control Loop    │
                    │                            │
Twist command ─────►│ 1. Jacobian pseudoinverse  │
                    │ 2. Collision deceleration  │
                    │ 3. Singularity damping     │
                    │ 4. Velocity limiting       │
                    │ 5. EMA filtering           │
                    │                            │
                    │ Output: JointCommand       │────► Robot
                    └───────────────────────────┘
```

## Performance Architecture

### Why KINETIC is Fast

1. **No ROS overhead**: Direct function calls, no serialization/IPC
2. **f64 precision throughout**: Single code path, no f32↔f64 conversions
3. **SIMD collision**: SoA layout enables vectorized distance checks
4. **Minimal allocations**: Stack-allocated math types via nalgebra
5. **Criterion benchmarks**: Every operation has a benchmark target

### Target Latencies

| Operation | Target | Measured |
|-----------|--------|----------|
| FK (7-DOF) | <1 us | ~324 ns |
| Jacobian (7-DOF) | <2 us | ~540 ns |
| IK DLS (7-DOF) | <500 us | ~10.6 us |
| Self-collision | <200 ns | ~9 ns |
| Env collision (10 obs) | <500 ns | ~507 ns |
| Servo tick | <500 us | ~9.9 us |
| RMP tick (3 policies) | <200 us | ~11 us |
| Trapezoidal (10 wp) | <100 us | ~2.5 us |

## Python Bindings

The `kinetic-python` crate provides PyO3 bindings with numpy interop:

```python
import kinetic
import numpy as np

robot = kinetic.Robot("ur5e")
planner = kinetic.Planner(robot)
traj = planner.plan(start, kinetic.Goal.joints(goal))
times, positions, velocities = traj.to_numpy()
```

Built with maturin: `cd kinetic/crates/kinetic-python && maturin develop`
