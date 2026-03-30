# KINETIC Testing Guide

## Coverage Summary

**Overall: 80.06% line coverage** (5,678/7,092 lines) as of March 2026.

| Crate | Lines Covered | Total | Coverage |
|-------|:---:|:---:|:---:|
| kinetic-core | 316 | 403 | 78.4% |
| kinetic-robot | 391 | 453 | 86.3% |
| kinetic-kinematics | 732 | 1,050 | 69.7% |
| kinetic-collision | 675 | 799 | 84.5% |
| kinetic-scene | 587 | 679 | 86.5% |
| kinetic-trajectory | 1,007 | 1,096 | 91.9% |
| kinetic-planning | 1,032 | 1,377 | 75.0% |
| kinetic-reactive | 340 | 433 | 78.5% |
| kinetic-task | 98 | 253 | 38.7% |
| kinetic-grasp | 282 | 320 | 88.1% |
| kinetic (facade) | 114 | 119 | 95.8% |

## Running Tests

```bash
# All unit tests (fast, ~30s)
cargo test --lib --workspace

# All unit + integration tests (slow, ~5min for planning tests)
cargo test --workspace

# Skip slow/ignored tests
cargo test --workspace -- --skip ik_roundtrip_all_6dof

# Specific crate
cargo test -p kinetic-collision
cargo test -p kinetic-kinematics

# Specific test file (integration tests)
cargo test -p kinetic --test test_goal_change

# Run ignored benchmarks
cargo test -p kinetic --test test_realtime_timing -- --ignored
cargo test -p kinetic --test test_scale_stress -- --ignored
cargo test -p kinetic --test test_all_robots -- --ignored

# Show test output
cargo test -- --nocapture
```

## Code Coverage (cargo-tarpaulin)

```bash
# Install
cargo install cargo-tarpaulin

# Run coverage (unit tests only — integration tests timeout under ptrace)
cargo tarpaulin --lib --skip-clean --out html stdout --timeout 600 \
  --exclude-files "crates/kinetic-gpu/*" "crates/kinetic-python/*" "crates/horus-kinetic/*"

# Using tarpaulin.toml config
cargo tarpaulin --config tarpaulin.toml

# Single crate
cargo tarpaulin -p kinetic-collision --skip-clean --out stdout
```

HTML reports are generated in `coverage/`. Configuration is in `tarpaulin.toml`.

**Note:** Planning tests (RRT-Connect) may timeout under tarpaulin's ptrace
instrumentation (3-5x overhead). Run `--lib` (unit tests only) for reliable
coverage reports.

## Excluded from Coverage

| Exclusion | Reason |
|-----------|--------|
| `kinetic-gpu` | Requires GPU hardware (wgpu compute shaders) |
| `kinetic-python` | PyO3 bindings — tested separately via pytest |
| `horus-kinetic` | HORUS integration — tested in horus workspace |
| Integration tests | Timeout under tarpaulin's ptrace instrumentation |

## Test Architecture

### Unit Tests (co-located with source)

Every crate has `#[cfg(test)] mod tests` blocks co-located with source code.
These test individual functions and data structures in isolation.

| Crate | Test Count | Focus |
|-------|:---:|-------|
| kinetic-core | 75 | JointValues, Pose, Twist, Wrench, Trajectory, Goal, Constraint |
| kinetic-robot | 26 | URDF parsing, config loading, joint limits, SRDF groups |
| kinetic-kinematics | 83 | FK, Jacobian, IK (DLS/FABRIK/OPW/subproblem), chain extraction |
| kinetic-collision | 105 | SIMD sphere-tree, mesh backend, ACM, two-tier, CAPT, SoA |
| kinetic-scene | 71 | Scene graph, add/remove/attach/detach, octree, pointcloud, depth |
| kinetic-trajectory | 90 | Trapezoidal, jerk-limited, spline, TOTP, blend, monitor, validation |
| kinetic-planning | 119 | RRT, IRIS, GCS, constraint projection, Cartesian, shortcut, smooth |
| kinetic-reactive | 16 | Potential fields, servo, filter, damping |
| kinetic-task | 12 | Task sequence, pick/place |
| kinetic-grasp | 94 | Grasp generation, quality scoring, filtering |
| kinetic (facade) | 4 | Benchmark utilities |

### Integration Tests (`crates/kinetic/tests/`)

36 integration test files exercising cross-crate workflows:

| File | Tests | Category | Notes |
|------|:---:|---------|-------|
| `test_plan_simple.rs` | ~5 | Planning | Joint-to-joint planning |
| `test_plan_with_scene.rs` | ~5 | Planning | Collision-aware planning |
| `test_gcs_planning.rs` | ~3 | Planning | Graph of Convex Sets |
| `test_servo_reactive.rs` | ~5 | Reactive | Servo twist/jog control |
| `test_task_planning.rs` | ~4 | Task | Multi-step task sequences |
| `test_e2e_pick_place.rs` | ~3 | E2E | Full pick-and-place workflow |
| `test_e2e_applications.rs` | 5 | E2E | Palletizing, welding, assembly, bin picking |
| `test_goal_change.rs` | 11 | Reactive | Goal change during execution |
| `test_wiring_integration.rs` | ~10 | Integration | Cross-crate API wiring |
| `test_boundary_conditions.rs` | ~8 | Robustness | Edge cases |
| `test_concurrency.rs` | ~5 | Safety | Multi-thread planning |
| `test_dynamic_scene.rs` | ~5 | Scene | Runtime scene modification |
| `test_edge_cases.rs` | ~8 | Robustness | Corner cases |
| `test_error_propagation.rs` | ~10 | Errors | Error type propagation |
| `test_execution_recovery.rs` | ~5 | Recovery | Failure recovery |
| `test_failure_recovery.rs` | ~5 | Recovery | Planning failure handling |
| `test_high_dof.rs` | ~5 | Scale | 7+ DOF robots |
| `test_ik_solver_selection.rs` | ~5 | IK | Solver dispatch |
| `test_multi_solution_ik.rs` | ~5 | IK | Multiple IK solutions |
| `test_multi_robot.rs` | ~5 | Scale | Multiple robots |
| `test_nan_inf_inputs.rs` | ~10 | Robustness | NaN/Inf handling |
| `test_negative_inputs.rs` | ~5 | Robustness | Negative values |
| `test_zero_empty_inputs.rs` | ~8 | Robustness | Zero/empty inputs |
| `test_numerical_stability.rs` | ~5 | Numerical | Floating-point stability |
| `test_properties.rs` | ~5 | Property | Property-based (proptest) |
| `test_scene_density.rs` | ~3 | Scale | Dense scenes |
| `test_stress.rs` | ~5 | Scale | Stress testing |
| `test_time_parameterization.rs` | ~5 | Trajectory | Time parameterization |
| `test_trajectory_execution.rs` | ~5 | Trajectory | Execution monitoring |
| `test_trajectory_scene_edge_cases.rs` | ~5 | Trajectory | Scene+trajectory combos |
| `test_trajectory_splicing.rs` | ~3 | Trajectory | Trajectory splicing |
| `test_uncovered_error_paths.rs` | ~8 | Errors | Error path coverage |
| `test_gpu_optimizer.rs` | ~3 | GPU | GPU optimizer (CPU fallback) |
| `test_all_robots.rs` | 1 | Scale | All 46 robots IK roundtrip (`--ignored`) |
| `test_realtime_timing.rs` | 3 | Benchmark | FK/Jacobian/trajectory latency (`--ignored`) |
| `test_scale_stress.rs` | 1 | Benchmark | SIMD 1000-sphere collision (`--ignored`) |

### Python Tests (`crates/kinetic-python/tests/`)

| File | Tests | Category |
|------|:---:|---------|
| `test_kinetic.py` | 99 | Unit tests: Robot, Goal, Planner, Trajectory, Scene, Servo, errors, NumPy edge cases |
| `test_e2e.py` | 23 | E2E workflows: pick-and-place, obstacle avoidance, trajectory inspection, servo control, multi-robot |

Run with:
```bash
cd crates/kinetic-python
python -m venv .venv && source .venv/bin/activate
pip install numpy pytest
maturin develop --release
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/ -v
```

### Ignored Tests (Benchmarks)

Tests marked `#[ignore]` are timing-sensitive benchmarks that fail under system load
or CI instrumentation. Run them on an idle machine:

```bash
cargo test -p kinetic -- --ignored
```

| Test | Reason |
|------|--------|
| `ik_roundtrip_all_6dof_robots` | 46 robots × 30 IK attempts — too slow for CI |
| `fk_latency_10000_calls` | Timing threshold: <5µs per FK call |
| `jacobian_latency_10000_calls` | Timing threshold: <10µs per Jacobian call |
| `trajectory_sampling_latency_100000` | Timing threshold: <1µs per sample |
| `simd_collision_1000_spheres` | Timing threshold: 1000v1000 <100ms |

## Known Low-Coverage Areas

| Module | Coverage | Reason |
|--------|:---:|--------|
| `subproblem.rs` | 21.2% | Analytical IK for specific robot geometries — only exercised by robots with matching kinematic structures |
| `task/lib.rs` | 38.7% | Pick/place planning requires full scene + grasp + planning pipeline; tested via integration tests but not under tarpaulin |
| `kinetic-core/trajectory.rs` | 71.9% | Iterator/conversion impls for trajectory types; some paths only used by higher-level crates |

## Miri (Undefined Behavior Detection)

```bash
rustup component add miri --toolchain nightly
cargo +nightly miri test -p kinetic-collision --lib simd::tests
cargo +nightly miri test -p kinetic-collision --lib soa::tests
```

Checks unsafe SIMD code in `kinetic-collision` for out-of-bounds access,
use-after-free, uninitialized memory, and alignment violations.

## Benchmarks

```bash
cargo bench -p kinetic
```

Results stored in `target/criterion/`. CI runs benchmarks as report-only
(no regression detection).

## CI Workflow

The CI pipeline (`.github/workflows/kinetic.yml`) runs:

1. **Check & Lint**: `cargo fmt --check`, `cargo clippy`, `cargo build`
2. **Test**: `cargo test --workspace` on Ubuntu + macOS
3. **Coverage**: `cargo tarpaulin --lib` with HTML report artifact
4. **Benchmarks**: `cargo bench` with Criterion results artifact
5. **Documentation**: `cargo doc --no-deps --all-features`
