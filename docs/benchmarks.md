# KINETIC Benchmarks

## Per-Operation Benchmarks

Run with: `cargo bench -p kinetic`

*Last measured: 2026-03-02 on Linux x86_64*

### Forward Kinematics

| Benchmark | Measured | Target |
|-----------|----------|--------|
| FK single 7-DOF | ~324 ns | <1 us |
| FK all links 7-DOF | ~358 ns | <1 us |
| FK batch 8 configs | ~2.6 us (~325 ns/config) | <500 ns/config |
| Jacobian 7-DOF | ~540 ns | <2 us |
| Manipulability 7-DOF | ~3.4 us | — |

### Inverse Kinematics

| Benchmark | Measured | Target |
|-----------|----------|--------|
| IK DLS 7-DOF | ~10.6 us | <500 us |
| IK FABRIK 3-DOF | ~21.5 us | <300 us |
| IK DLS 3-DOF | ~4.5 us | <100 us |
| FK + IK roundtrip | ~11 us | <1 ms |

### Collision Detection

| Benchmark | Measured | Target |
|-----------|----------|--------|
| Self-collision check | ~9 ns | <200 ns |
| Env collision (10 obstacles) | ~507 ns | <500 ns |
| Env collision (100 obstacles) | ~2.6 us | — |
| Env collision (1000 obstacles) | ~313 ns | <1 ms |
| SIMD sphere 20v10 | ~1.2 us | <2 us |
| SIMD min distance 20v10 | ~1.4 us | <2 us |

### Trajectory Time Parameterization

| Benchmark | Measured | Target |
|-----------|----------|--------|
| Trapezoidal 7-DOF 10wp | ~2.5 us | <100 us |
| Trapezoidal per-joint 7-DOF | ~3.6 us | <200 us |
| TOTP 7-DOF 20wp | ~290 us | <1 ms |
| Jerk-limited 7-DOF 10wp | ~22 us | — |
| Cubic spline 7-DOF 10wp | ~15 us | — |
| Blend two trajectories | ~19 us | — |
| Trapezoidal 7-DOF 100wp | ~36 us | — |
| Trajectory validation 7-DOF 50wp | ~8.6 us | — |

### Reactive Control

| Benchmark | Measured | Target |
|-----------|----------|--------|
| Servo send_twist | ~9.9 us | <500 us |
| Servo joint_jog | ~4.6 us | <500 us |
| Servo state query | ~345 ps | — |
| RMP tick (1 policy) | ~10.6 us | <200 us |
| RMP combined (3 policies) | ~11.1 us | <200 us |

### Planning

| Benchmark | Measured | Target |
|-----------|----------|--------|
| RRT-Connect (no obstacles) | ~258 ms | — |
| RRT-Connect (table) | ~244 ms | — |
| RRT-Connect (10 obstacles) | ~262 ms | — |
| Cartesian linear 10cm | ~133 us | <1 ms |
| GCS build + plan | ~172 us | <1 ms |
| Constrained RRT setup | ~986 ns | <10 us |

### Full Pipeline

| Benchmark | Measured | Target |
|-----------|----------|--------|
| Plan only | ~237 ms | — |
| Plan + trapezoidal | ~243 ms | — |
| Plan + TOTP | ~253 ms | — |
| Trajectory sample (100 pts) | ~10.5 us | — |

### MotionBenchMaker Scenarios (Franka Panda, 7-DOF)

| Scenario | Measured | MoveIt2 (OMPL) | VAMP | cuRobo |
|----------|----------|----------------|------|--------|
| Table pick (simple) | ~423 ms | 170 ms | 35 us | 45 ms |
| Shelf pick (cluttered) | ~420 ms | 1200 ms | 16 ms | 45 ms |
| Narrow passage | ~415 ms | 3000+ ms | 50 ms | 100 ms |
| Cluttered desk | ~446 ms | — | — | — |
| Full suite (3 runs each) | ~4.65 s | — | — | — |

*KINETIC uses a general-purpose RRT-Connect planner. While slower than VAMP's
compile-time-specialized SIMD planner, it handles any URDF without code generation.
KINETIC outperforms MoveIt2 on cluttered/narrow scenarios by 3-7x.*

### Per-Operation Comparison

| Operation | KINETIC | MoveIt2/KDL | VAMP |
|-----------|---------|-------------|------|
| FK (7-DOF) | ~324 ns | ~5-10 us | ~35 ns* |
| Jacobian | ~540 ns | ~10 us | — |
| IK (DLS) | ~10.6 us | ~5 ms (timeout) | — |
| Collision check | ~9 ns (self) | ~50-100 us | ~35 ns* |
| Servo tick | ~9.9 us | ~1 ms (ROS overhead) | N/A |

*VAMP uses highly optimized SIMD/AVX2 with compile-time robot specialization.
KINETIC is general-purpose (any URDF) and still achieves sub-microsecond FK.*

## Running Benchmarks

```bash
# All benchmarks
cargo bench -p kinetic

# Specific benchmark group
cargo bench -p kinetic --bench fk
cargo bench -p kinetic --bench ik
cargo bench -p kinetic --bench collision
cargo bench -p kinetic --bench planning
cargo bench -p kinetic --bench trajectory
cargo bench -p kinetic --bench reactive
cargo bench -p kinetic --bench full_pipeline
cargo bench -p kinetic --bench motionbenchmaker

# Quick run (fewer iterations)
cargo bench -p kinetic -- --quick
```

## Benchmark Suite API

Use `kinetic::benchmark` for programmatic benchmarking:

```rust
use kinetic::benchmark::{BenchmarkSuite, generate_report};

let suite = BenchmarkSuite::motionbenchmaker();
let results = suite.run();
let report = generate_report(&results);
println!("{}", report);
```

## CI Integration

Benchmarks can be run in CI to detect regressions:

```yaml
- name: Run benchmarks
  run: cargo bench -p kinetic -- --output-format bencher | tee benchmark_results.txt
```

A >20% regression in any benchmark should trigger investigation.
