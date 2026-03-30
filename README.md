# KINETIC

Fast, Rust-native motion planning for robotics.

KINETIC provides forward/inverse kinematics, SIMD-vectorized collision detection,
and trajectory generation — all in pure Rust with f64 precision throughout.

## Installation

```toml
[dependencies]
kinetic = "0.1"
```

## Quick Start

```rust
use kinetic::prelude::*;

// Load a robot from URDF
let robot = Robot::from_urdf("path/to/panda.urdf")?;
let chain = KinematicChain::extract(&robot, "panda_link0", "panda_link8")?;

// Forward kinematics
let q = vec![0.3, -0.5, 0.2, -1.5, 0.1, 1.0, 0.5];
let pose = forward_kinematics(&robot, &chain, &q)?;

// Inverse kinematics (DLS with adaptive damping)
let config = IKConfig::dls()
    .with_seed(robot.mid_configuration().to_vec())
    .with_max_iterations(300);
let solution = solve_ik(&robot, &chain, &pose, &config)?;
println!("IK converged: {}", solution.converged);
```

## Features

| Feature | Description |
|---------|-------------|
| **Forward Kinematics** | Direct matrix chain, batch FK for planning |
| **Jacobian** | 6xN geometric Jacobian, manipulability index |
| **IK (DLS)** | Damped Least Squares with adaptive damping, null-space optimization |
| **IK (FABRIK)** | Position-level FABRIK with DLS orientation refinement |
| **Collision** | SoA sphere model, SIMD-accelerated (AVX2/NEON/scalar) |
| **CAPT** | Collision-Affording Point Tree for <100ns environment queries |
| **URDF** | Full URDF parsing with joint limits, collision geometry |
| **Config** | TOML config for planning groups, end-effectors, named poses |

## Crate Structure

| Crate | Purpose |
|-------|---------|
| `kinetic` | Facade — re-exports everything via `prelude` |
| `kinetic-core` | Pose, JointValues, Trajectory, Goal, Constraint |
| `kinetic-robot` | URDF parser, Robot model, TOML config |
| `kinetic-kinematics` | FK, Jacobian, IK (DLS, FABRIK) |
| `kinetic-collision` | SIMD sphere collision, CAPT broadphase |

## Performance

All benchmarks on Panda-like 7-DOF robot (AMD Ryzen / Apple M-series):

| Operation | Target | Measured |
|-----------|--------|----------|
| FK (7-DOF) | <1 µs | ~2 µs |
| Jacobian (7-DOF) | <2 µs | ~2.4 µs |
| IK DLS (7-DOF) | <500 µs | ~50 µs |
| IK FABRIK (3-DOF) | <300 µs | ~160 µs |
| Collision self-check (SIMD) | <500 ns | ~97 ns |

Run benchmarks: `cargo bench -p kinetic`

## Examples

```sh
# FK → IK roundtrip demo
cargo run --example hello_fk_ik -p kinetic

# Multiple IK solutions
cargo run --example multiple_ik -p kinetic

# Collision checking
cargo run --example collision_check -p kinetic
```

## Comparison vs MoveIt2

| | KINETIC | MoveIt2 |
|---|---------|---------|
| Language | Rust | C++ |
| Precision | f64 | f64 |
| SIMD collision | AVX2/NEON auto-detect | No |
| Build system | Cargo | CMake + colcon |
| ROS dependency | None | Required |
| Memory safety | Guaranteed | Manual |
| Compile time | ~10s | Minutes |

## Testing

```bash
# Unit tests (~30s)
cargo test --lib --workspace

# Full test suite including integration tests
cargo test --workspace

# Code coverage (80%+ target)
cargo tarpaulin --lib --skip-clean --out html stdout
```

See [docs/testing.md](docs/testing.md) for the full testing guide, test inventory, and coverage details.

## License

Apache-2.0
