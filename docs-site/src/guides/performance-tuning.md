# Performance Tuning

Diagnosing and fixing slow planning, IK, or collision checks.

## PlannerConfig Presets

Kinetic ships with three presets covering the most common scenarios.
Start with a preset and adjust individual fields as needed.

| Preset | Timeout | Iterations | Margin | Shortcut | Smooth |
|--------|---------|------------|--------|----------|--------|
| `default()` | 50ms | 10,000 | 0.02m | 100 | Yes |
| `realtime()` | 10ms | 2,000 | 0.01m | 20 | No |
| `offline()` | 500ms | 100,000 | 0.02m | 500 | Yes |

```rust
use kinetic::prelude::*;
use std::time::Duration;

// Use a preset
let planner = Planner::new(&robot)?
    .with_config(PlannerConfig::realtime());

// Or customize individual fields
let config = PlannerConfig {
    timeout: Duration::from_millis(30),
    max_iterations: 5_000,
    collision_margin: 0.015,
    shortcut_iterations: 50,
    smooth: false,
    workspace_bounds: Some([-1.0, -1.0, 0.0, 1.0, 1.0, 2.0]),
};
```

## IKConfig Tuning

IK performance depends on the solver choice and restart count.

**Solver selection guidelines:**

| Solver | Speed | Best For |
|--------|-------|----------|
| OPW | <5 us | 6-DOF spherical-wrist robots (UR, KUKA KR, ABB IRB) |
| Subproblem | <10 us | 6-DOF with intersecting wrist axes |
| Subproblem7DOF | ~50 us | 7-DOF (Panda, KUKA iiwa) |
| DLS | 100-500 us | General purpose, any DOF |
| FABRIK | 50-200 us | Position-only tasks |

**Tuning parameters:**

```rust
use kinetic::prelude::*;

// Fast: analytical solver, no restarts
let config = IKConfig::opw();

// Robust: iterative solver with restarts
let config = IKConfig::dls()
    .with_restarts(8)
    .with_max_iterations(200)
    .with_position_tolerance(1e-3);

// Position-only (3-DOF task, ignore orientation)
let config = IKConfig::position_only();

// Try full 6D, fall back to position-only
let config = IKConfig::with_fallback();
```

**Key tuning knobs:**

- `num_restarts`: Each restart randomizes the seed. More restarts improve
  success rate but cost time. Use 0 for real-time, 4-8 for planning.
- `max_iterations`: Increase for hard-to-reach poses. Decrease for real-time.
- `position_tolerance`: Loosen (1e-3) for speed, tighten (1e-5) for precision.
- `seed`: Providing a good seed (e.g., current joint state) dramatically
  reduces convergence time.

## SIMD Tier Detection

Kinetic auto-detects SIMD capabilities at compile time:

| Tier | Instructions | Collision Speedup |
|------|-------------|-------------------|
| AVX-512 | 16-wide f32 | ~8x |
| AVX2 | 8-wide f32 | ~4x |
| SSE4.1 | 4-wide f32 | ~2x |
| Scalar | Fallback | 1x |

SIMD is used automatically in collision checking. Compile with native
target features to get the best performance:

```bash
RUSTFLAGS="-C target-cpu=native" cargo build --release
```

## Batch Operations

When you need FK or IK for many configurations, use the batch APIs
to amortize overhead and enable parallelism.

```rust
use kinetic::prelude::*;
use kinetic::kinematics::solve_ik_batch;

let robot = Robot::from_name("ur5e")?;
let chain = KinematicChain::extract(&robot, "base_link", "tool0")?;

// Batch FK
let configs: Vec<Vec<f64>> = vec![
    vec![0.0; 6],
    vec![0.1, -0.5, 0.3, 0.0, 0.0, 0.0],
    vec![0.5, -1.0, 0.8, 0.2, -0.3, 0.1],
];
let poses: Vec<Pose> = configs.iter()
    .map(|q| forward_kinematics(&robot, &chain, q).unwrap())
    .collect();

// Batch IK
let solutions = solve_ik_batch(&robot, &chain, &poses, &IKConfig::default());
// Returns Vec<Option<IKSolution>> -- None for failed targets
```

For GPU-accelerated batch FK:

```rust
use kinetic::gpu::batch_fk_gpu;
let results = batch_fk_gpu(&robot, &configs)?;
```

## Collision Resolution vs Speed Trade-off

The `collision_margin` parameter controls the trade-off between safety
and planning speed. Larger margins make the planner more conservative
but reject more configurations, making it harder to find paths.

| collision_margin | Effect |
|-----------------|--------|
| 0.00m | Exact contact only (fast, risky) |
| 0.01m | Tight clearance (production minimum) |
| 0.02m | Default (good balance) |
| 0.05m | Conservative (cluttered environments) |

**Sphere resolution** also affects speed. The `SphereGenConfig` controls
how many spheres approximate each link:

```rust
use kinetic::collision::SphereGenConfig;

SphereGenConfig::coarse();  // ~5 spheres/link (fast, less precise)
SphereGenConfig::default(); // ~10 spheres/link (balanced)
SphereGenConfig::fine();    // ~20 spheres/link (slow, precise)
```

## GPU: When to Use

Use the GPU optimizer when:

- Environment has 50+ obstacles (SDF lookup is O(1) vs O(n) sphere checks)
- You need smooth trajectories without post-processing
- Planning latency budget is 20-100ms with a discrete GPU available

```rust
use kinetic::gpu::{GpuOptimizer, GpuConfig};

// Speed preset: 32 seeds, 30 iterations
let opt = GpuOptimizer::new(GpuConfig::speed())?;

// Quality preset: 512 seeds, 200 iterations
let opt = GpuOptimizer::new(GpuConfig::quality())?;

// Balanced (default): 128 seeds, 100 iterations
let opt = GpuOptimizer::new(GpuConfig::balanced())?;
```

The `CpuOptimizer` provides the same API without requiring a GPU,
useful for testing or environments without GPU access.

## Profiling Checklist

1. Run `cargo bench` to establish baselines
2. Identify the bottleneck: IK, collision, or tree expansion
3. For IK: try analytical solvers (OPW/Subproblem) before DLS
4. For collision: reduce sphere count or loosen margin
5. For planning: reduce iterations, disable smoothing, add workspace bounds
6. For trajectory: use `trapezoidal` (fast) instead of `totp` (optimal)
