# Configuration Reference

All configuration structs, fields, and presets.

## PlannerConfig

Controls motion planner behavior. Used by all planners through the `Planner`
facade.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `timeout` | Duration | 50ms | Maximum planning time before timeout |
| `max_iterations` | usize | 10,000 | Upper bound on sampling/expansion iterations |
| `collision_margin` | f64 | 0.02 | Minimum clearance from obstacles (meters) |
| `shortcut_iterations` | usize | 100 | Number of random-shortcut passes |
| `smooth` | bool | true | Apply B-spline smoothing after shortcutting |
| `workspace_bounds` | `Option<[f64; 6]>` | None | [min_x, min_y, min_z, max_x, max_y, max_z] |

### Presets

**`PlannerConfig::default()`** -- General purpose.
```rust
PlannerConfig {
    timeout: Duration::from_millis(50),
    max_iterations: 10_000,
    collision_margin: 0.02,
    shortcut_iterations: 100,
    smooth: true,
    workspace_bounds: None,
}
```

**`PlannerConfig::realtime()`** -- Low-latency planning (10ms budget).
```rust
PlannerConfig {
    timeout: Duration::from_millis(10),
    max_iterations: 2_000,
    collision_margin: 0.01,
    shortcut_iterations: 20,
    smooth: false,
    workspace_bounds: None,
}
```

**`PlannerConfig::offline()`** -- Thorough offline planning (500ms budget).
```rust
PlannerConfig {
    timeout: Duration::from_millis(500),
    max_iterations: 100_000,
    collision_margin: 0.02,
    shortcut_iterations: 500,
    smooth: true,
    workspace_bounds: None,
}
```

## IKConfig

Controls inverse kinematics solver behavior.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `solver` | IKSolver | Auto | Solver selection (Auto, DLS, FABRIK, OPW, Subproblem, Subproblem7DOF) |
| `mode` | IKMode | Full6D | Full6D, PositionOnly, or PositionFallback |
| `max_iterations` | usize | 100 | Max iterations for iterative solvers |
| `position_tolerance` | f64 | 1e-4 | Position convergence tolerance (meters) |
| `orientation_tolerance` | f64 | 1e-3 | Orientation convergence tolerance (radians) |
| `check_limits` | bool | true | Enforce joint limits |
| `seed` | `Option<Vec<f64>>` | None | Starting configuration (uses mid-config if None) |
| `null_space` | `Option<NullSpace>` | None | Null-space objective for redundant robots |
| `num_restarts` | usize | 0 | Random restart count for escaping local minima |

### Solver Options

| Solver | Constructor | Best For |
|--------|------------|----------|
| Auto | `IKConfig::default()` | Automatic selection based on robot geometry |
| DLS | `IKConfig::dls()` | General purpose, any DOF |
| FABRIK | `IKConfig::fabrik()` | Position-focused tasks |
| OPW | `IKConfig::opw()` | 6-DOF spherical wrist robots |
| Subproblem | Direct construction | 6-DOF intersecting wrist axes |
| Subproblem7DOF | Direct construction | 7-DOF robots |

### Convenience Constructors

```rust
IKConfig::dls()             // DLS with default damping 0.05
IKConfig::fabrik()          // FABRIK solver
IKConfig::opw()             // OPW analytical solver
IKConfig::position_only()   // Position-only mode
IKConfig::with_fallback()   // Try Full6D, fall back to PositionOnly
```

## ServoConfig

Controls the reactive servo controller for teleoperation and tracking.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `rate_hz` | f64 | 500.0 | Control loop rate |
| `input_type` | InputType | Twist | Twist, JointJog, or PoseTracking |
| `collision_check_hz` | f64 | 100.0 | Collision check frequency |
| `singularity_threshold` | f64 | 0.02 | Manipulability threshold |
| `slowdown_distance` | f64 | 0.15 | Distance to start decelerating (meters) |
| `stop_distance` | f64 | 0.03 | Emergency stop distance (meters) |
| `velocity_limits` | `Vec<f64>` | [] | Per-joint vel limits (empty = use robot) |
| `acceleration_limits` | `Vec<f64>` | [] | Per-joint accel limits (empty = defaults) |
| `pose_tracking_gain` | f64 | 5.0 | Proportional gain for pose tracking |
| `singularity_damping` | f64 | 0.05 | Damping for singularity-robust pseudoinverse |
| `max_delta_per_tick` | f64 | 0.02 | Max position change per tick (radians) |

### Presets

**`ServoConfig::teleop()`** -- General teleoperation (joystick, spacemouse).
Uses Twist input, generous collision margins, moderate precision.

**`ServoConfig::tracking()`** -- Pose tracking (following a moving target).
Higher tracking gain (10.0), tighter collision checking (200 Hz),
PoseTracking input mode.

**`ServoConfig::precise()`** -- Precise manipulation (assembly, insertion).
Small movements per tick (0.005 rad), tight singularity avoidance,
fine collision checking (250 Hz).

## GpuConfig

Controls GPU-accelerated trajectory optimization.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `num_seeds` | u32 | 128 | Number of parallel trajectory seeds |
| `timesteps` | u32 | 32 | Waypoints per trajectory |
| `iterations` | u32 | 100 | Gradient descent iterations |
| `collision_weight` | f32 | 100.0 | SDF collision cost weight |
| `smoothness_weight` | f32 | 1.0 | Jerk minimization weight |
| `goal_weight` | f32 | 50.0 | Goal-reaching cost weight |
| `step_size` | f32 | 0.01 | Gradient descent step size |
| `sdf_resolution` | f32 | 0.02 | SDF voxel resolution (meters) |
| `workspace_bounds` | [f32; 6] | [-1,-1,-0.5,1,1,1.5] | SDF workspace bounds |
| `seed_perturbation` | f32 | 0.3 | Random perturbation magnitude (radians) |
| `warm_start` | Option | None | Initial trajectory from RRT |

### Presets

**`GpuConfig::balanced()`** -- Default, good balance of speed and quality.
128 seeds, 32 timesteps, 100 iterations.

**`GpuConfig::speed()`** -- Fast optimization for real-time replanning.
32 seeds, 24 timesteps, 30 iterations, coarse SDF (0.05m).

**`GpuConfig::quality()`** -- High-quality offline optimization.
512 seeds, 48 timesteps, 200 iterations, fine SDF (0.01m).

## ExecutionConfig

Controls trajectory execution on hardware.

### Fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `rate_hz` | f64 | 500.0 | Command streaming rate |
| `position_tolerance` | f64 | 0.1 | Max position deviation (radians) |
| `velocity_tolerance` | f64 | 0.5 | Max velocity deviation (rad/s) |
| `timeout_factor` | f64 | 2.0 | Abort if execution exceeds expected_time * factor |
| `joint_limits` | Option | None | [(lower, upper)] for pre-execution validation |
| `command_timeout_ms` | u64 | 100 | Per-command timeout |
| `require_feedback` | bool | false | Require FeedbackSource for monitoring |
| `watchdog` | Option | None | Safety watchdog configuration |

### Presets

**`ExecutionConfig::default()`** -- Simulation and testing.
Relaxed tolerances, no feedback required, no watchdog.

**`ExecutionConfig::safe(&robot)`** -- Production deployment.
Auto-populates joint limits from URDF, enables feedback requirement,
configures safety watchdog (50ms timeout, ZeroVelocity action).

```rust
let config = ExecutionConfig::safe(&robot);
```
