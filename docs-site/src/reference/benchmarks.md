# Benchmarks

Performance measurements and comparisons.

## Performance Table

All measurements on AMD Ryzen 9 7950X, DDR5-5200, Ubuntu 24.04,
Rust 1.75+ (release mode, `-C target-cpu=native`).

### Core Operations

| Operation | Robot | Time | Notes |
|-----------|-------|------|-------|
| FK (6-DOF) | UR5e | <1 us | Single end-effector pose |
| FK all links (6-DOF) | UR5e | ~2 us | All link poses |
| FK (7-DOF) | Panda | ~1.2 us | Single end-effector pose |
| Jacobian (6-DOF) | UR5e | ~1.5 us | 6x6 matrix |
| Jacobian (7-DOF) | Panda | ~2 us | 6x7 matrix |
| Manipulability | UR5e | ~3 us | SVD-based |

### Inverse Kinematics

| Solver | Robot | Time | Notes |
|--------|-------|------|-------|
| OPW | UR5e | <5 us | Analytical, closed-form |
| Subproblem | UR5e | <10 us | Analytical, up to 16 solutions |
| Subproblem7DOF | Panda | ~50 us | Sweep + analytical |
| DLS (converge) | UR5e | 100-300 us | 50-100 iterations typical |
| DLS (converge) | Panda | 200-500 us | 7-DOF, more iterations |
| FABRIK | UR5e | 50-150 us | Position-focused |
| Batch IK (100 targets) | UR5e | ~500 us | OPW, amortized |

### Collision Checking

| Operation | Spheres | Time | Notes |
|-----------|---------|------|-------|
| Self-collision (coarse) | ~30 | <500 ns | SIMD AVX2 |
| Self-collision (fine) | ~120 | ~2 us | SIMD AVX2 |
| Env collision (10 obstacles) | ~30 | ~300 ns | CAPT broadphase |
| Env collision (100 obstacles) | ~30 | ~1 us | CAPT broadphase |
| Env collision (1000 obstacles) | ~30 | ~5 us | CAPT broadphase |

### Motion Planning

| Planner | Robot | Environment | Time | Path Quality |
|---------|-------|-------------|------|-------------|
| RRT-Connect | UR5e | Empty | <100 us | Feasible |
| RRT-Connect | UR5e | 10 obstacles | 5-50 ms | Feasible |
| RRT-Connect | Panda | 10 obstacles | 10-80 ms | Feasible |
| RRT* | UR5e | 10 obstacles | 100-500 ms | Near-optimal |
| BiRRT* | UR5e | 10 obstacles | 50-300 ms | Near-optimal |
| EST | UR5e | Narrow passage | 20-100 ms | Feasible |
| Cartesian | UR5e | Empty | <10 ms | Exact path |

### Servo (Reactive Control)

| Operation | Robot | Time | Rate |
|-----------|-------|------|------|
| Twist command | UR5e | ~50 us | 500 Hz capable |
| Joint jog | UR5e | ~20 us | 500 Hz capable |
| Pose tracking | Panda | ~100 us | 500 Hz capable |

### Trajectory Processing

| Operation | Waypoints | Time | Notes |
|-----------|-----------|------|-------|
| Trapezoidal | 50 | <100 us | Per-joint limits |
| TOTP | 50 | 1-5 ms | Time-optimal |
| Jerk-limited | 50 | 200-500 us | S-curve profile |
| Cubic spline | 50 | <50 us | Interpolation only |
| Validation | 50 | ~20 us | Limit checking |

## Comparison vs MoveIt2

Approximate comparisons based on published benchmarks and internal testing.
MoveIt2 measurements from MoveIt2 documentation and community benchmarks.

| Operation | Kinetic | MoveIt2 | Speedup |
|-----------|---------|---------|---------|
| FK (6-DOF) | <1 us | ~5 us | ~5x |
| IK (OPW, 6-DOF) | <5 us | ~50 us (KDL) | ~10x |
| IK (DLS, 7-DOF) | 200-500 us | 500 us-2 ms (KDL) | ~2-4x |
| Collision check | <500 ns | ~5 us (FCL) | ~10x |
| RRT-Connect (simple) | <100 us | 1-10 ms | ~10-100x |
| Servo loop | ~50 us | ~200 us | ~4x |

Speedups come from:
- Rust vs C++ (memory safety without overhead)
- SIMD collision (4-16x throughput per check)
- Analytical IK solvers (OPW, Subproblem) vs iterative KDL
- No ROS middleware overhead

## How to Run Benchmarks

### Prerequisites

```bash
# Install criterion (benchmark framework)
cargo install cargo-criterion  # optional, nicer output

# Build in release mode
cargo build --release
```

### Running

```bash
# All benchmarks
cargo bench

# Specific benchmark group
cargo bench --bench fk_benchmarks
cargo bench --bench ik_benchmarks
cargo bench --bench collision_benchmarks
cargo bench --bench planning_benchmarks

# With HTML report
cargo bench --bench planning_benchmarks -- --output-format=bencher

# Filter by name
cargo bench -- "ur5e"
```

### Results

Criterion generates HTML reports in `target/criterion/`. Open
`target/criterion/report/index.html` for interactive charts.

### Writing Custom Benchmarks

```rust
use criterion::{criterion_group, criterion_main, Criterion};

fn my_benchmark(c: &mut Criterion) {
    let robot = Robot::from_name("ur5e").unwrap();
    let chain = KinematicChain::extract(&robot, "base_link", "tool0").unwrap();
    let joints = vec![0.0, -1.0, 0.8, 0.0, 0.0, 0.0];

    c.bench_function("fk_ur5e", |b| {
        b.iter(|| forward_kinematics(&robot, &chain, &joints))
    });
}

criterion_group!(benches, my_benchmark);
criterion_main!(benches);
```

## Reproducing Results

For reproducible benchmarks:

1. Use `--release` builds with `target-cpu=native`
2. Disable CPU frequency scaling: `sudo cpupower frequency-set -g performance`
3. Close background applications
4. Run each benchmark 3+ times and take the median
5. Report CPU model, RAM speed, OS version, and Rust version
