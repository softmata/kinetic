# KINETIC

Fast, Rust-native motion planning for robotics. Complete MoveIt2 replacement.

10 IK solvers, 14 planners in Rust (8 surfaced in Python), SIMD collision detection, 52 built-in robots, GPU trajectory optimization, full Python bindings.

**[Documentation](https://softmata.github.io/kinetic/)** | **[API Cheatsheet](https://softmata.github.io/kinetic/reference/api-cheatsheet.html)** | **[Python Quickstart](https://softmata.github.io/kinetic/tutorials/python/quickstart.html)**

## Quick Start

```rust
use kinetic::prelude::*;

// One-liner: load robot, plan, get trajectory
let result = plan("ur5e", &start_joints, &Goal::joints(goal_joints))?;
```

```python
import kinetic
import numpy as np

robot = kinetic.Robot("ur5e")
planner = kinetic.Planner(robot, planner_type="rrt_star")
traj = planner.plan(start, kinetic.Goal.joints(goal))

for t in np.linspace(0, traj.duration, 100):
    joints = traj.sample(t)
```

## Performance vs MoveIt2

| Operation | KINETIC | MoveIt2 | Speedup |
|-----------|---------|---------|---------|
| FK (7-DOF) | 324 ns | 5-10 us | **15-30x** |
| IK (DLS) | 10.6 us | 5 ms | **470x** |
| Self-collision | 9 ns | 50-100 us | **5,000x** |
| Env collision (10 obs) | 507 ns | 50-100 us | **100x** |
| Servo tick | 9.9 us | ~1 ms | **100x** |
| Plan (cluttered) | 420 ms | 1,200 ms | **2.9x** |
| Plan (narrow passage) | 415 ms | 3,000+ ms | **7.2x** |

Run benchmarks: `cargo bench -p kinetic`

## Features

**Kinematics** -- 10 IK solvers (OPW, Paden-Kahan, DLS, FABRIK, SQP, Bio-IK), batch FK/IK, Jacobian, manipulability

**Collision** -- SIMD-vectorized (AVX2/NEON/scalar), CAPT broadphase, 3-tier LOD (sphere/convex/mesh), CCD, SDF

**Planning** -- 14 algorithms: RRT-Connect, RRT\*, BiRRT\*, BiTRRT, EST, KPIECE, PRM, CHOMP, STOMP, GCS, Cartesian, constrained, dual-arm

**Trajectory** -- TOTP (time-optimal), trapezoidal, jerk-limited S-curve, cubic spline, blending, validation

**Reactive** -- 500 Hz servo (twist/jog/pose-tracking), RMP multi-policy blending, collision deceleration

**Task** -- Pick, place, multi-stage sequences with grasp generation

**GPU** -- wgpu parallel-seed optimization (Vulkan/Metal), batch collision, CPU fallback

**Dynamics** -- Featherstone bridge: inverse/forward dynamics, gravity compensation, mass matrix

**Execution** -- RealTimeExecutor (500 Hz), SimExecutor, LogExecutor, safety watchdog

**Python** -- Full PyO3 + numpy bindings, 22 classes, type stubs for IDE autocomplete

**52 Robots** -- UR, Franka, KUKA, ABB, Fanuc, Kinova, xArm, and 45 more

## Crate Architecture

```
kinetic (facade)
├── kinetic-core          Pose, Goal, Trajectory, Constraint, Error
├── kinetic-robot         URDF/MJCF/SDF parsing, 54 robot configs
├── kinetic-kinematics    FK, IK (10 solvers), Jacobian, batch ops
├── kinetic-collision     SIMD spheres, CAPT, ACM, mesh, LOD, CCD
├── kinetic-dynamics      Featherstone: ID, FD, gravity comp, mass matrix
├── kinetic-scene         Obstacles, attached objects, octree, pointcloud
├── kinetic-planning      14 planners, pipeline, cost functions, dual-arm
├── kinetic-trajectory    TOTP, trapezoidal, S-curve, spline, blending
├── kinetic-reactive      Servo 500Hz, RMP 6-policy, EMA/Butterworth
├── kinetic-task          Pick, place, sequence, grasp generation
├── kinetic-execution     RealTime/Sim/Log executors, safety watchdog
├── kinetic-gpu           wgpu optimizer, batch FK, SDF collision
├── kinetic-grasp         Parallel jaw, suction, quality scoring
├── kinetic-python        PyO3 + numpy bindings (22 classes)
├── horus-kinetic         HORUS IPC bridge (optional)
└── kinetic-viewer        3D visualization (optional)
```

## Installation

**Rust:**
```toml
[dependencies]
kinetic = { path = "crates/kinetic" }
```

**Python:**
```bash
cd crates/kinetic-python
maturin develop --release
```

## Examples

```bash
cargo run --example plan_simple -p kinetic
cargo run --example collision_check -p kinetic
cargo run --example servo_loop -p kinetic
cargo run --example grasp_planning -p kinetic
cargo run --example gpu_optimize -p kinetic
```

## Testing

```bash
# Full test suite (1,457+ tests)
cargo test --workspace --exclude kinetic-gpu

# With GPU tests
cargo test --workspace

# Coverage (85% target)
cargo tarpaulin --config tarpaulin.toml
```

## License

Apache-2.0
