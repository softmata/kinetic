# KINETIC: The Motion Planning Stack for Rust Robotics

> *"MoveIt took a decade. We have SIMD, wgpu, and hindsight."*

---

## Executive Summary

KINETIC is a complete motion planning stack — not just a planner, but the full pipeline from perception to execution. It is designed to replace MoveIt2 for teams building in Rust, and to leapfrog it for everyone else.

The bet: MoveIt2's architecture was designed in 2011 for single-threaded C++ on ROS1. KINETIC is designed in 2026 for SIMD-vectorized Rust, cross-vendor GPU compute, and zero-copy shared-memory IPC. We don't port MoveIt2 to Rust — we rebuild the stack from research published after MoveIt2's architecture was frozen.

**Three things MoveIt2 cannot do that KINETIC will:**

1. **35-microsecond planning.** VAMP (Kavraki Lab, 2023) proved that SIMD-vectorized collision checking enables RRT-Connect in microseconds on a single CPU core. KINETIC builds on this architecture natively in Rust — the language designed for data-oriented, zero-cost-abstraction, SIMD-friendly code.

2. **Cross-vendor GPU trajectory optimization.** cuRobo achieves 30ms planning with consistent, smooth trajectories — but it requires NVIDIA CUDA and a PyTorch runtime. KINETIC implements the same parallel-seed trajectory optimization through wgpu compute shaders, running on NVIDIA, AMD, Intel, and Apple Silicon.

3. **Reactive motion at IPC speed.** MoveIt Servo runs at ROS2 DDS latency (~500us minimum). KINETIC's reactive layer runs inside a HORUS node at <167ns IPC latency — 3000x faster feedback.

**What KINETIC matches from MoveIt2:** planning scene, collision detection, multiple IK solvers, Cartesian planning, constraint-aware planning, real-time servoing, task planning, grasp generation, perception pipeline, time parameterization, Python bindings, 50+ robot configurations.

**What KINETIC adds beyond MoveIt2:** SIMD-vectorized planning, cross-vendor GPU optimization, RMP-style reactive policies, automatic analytical IK decomposition, convex-set-based globally optimal planning, diffusion-model seeded trajectories, built-in benchmarking against VAMP/cuRobo baselines, and a 5-line API that doesn't require a framework.

---

## The Competitive Landscape (Honest)

| Capability | MoveIt2 | cuRobo | Drake | VAMP | **KINETIC** |
|------------|---------|--------|-------|------|-------------|
| Language | C++ | Python/CUDA | C++ | C++ | **Rust** |
| Framework dependency | ROS2 required | Isaac Sim / PyTorch | Bazel build | Header-only | **None (standalone)** |
| Planning speed (simple) | ~170ms | ~45ms | ~200ms | **35us** | **Target: <100us** |
| Planning speed (cluttered) | ~1200ms | ~45ms | ~500ms | ~16ms | **Target: <20ms** |
| Trajectory quality | Jerky (RRT) | Smooth (optimization) | Optimal (GCS) | Jerky (RRT) | **Smooth (tiered pipeline)** |
| GPU vendor | N/A (CPU) | NVIDIA only | N/A (CPU) | N/A (CPU) | **Any (wgpu)** |
| Reactive control | MoveIt Servo | Planned | MPC | No | **RMP-style reactive** |
| Perception pipeline | Octomap | nvblox/SDF | Hydroelastic | Point tree | **Point tree + SDF** |
| IK solvers | KDL, IKFast, BioIK | cuRobo IK | Drake IK | N/A | **OPW, DLS, FABRIK, analytical decomposition** |
| Task planning | MTC (complex) | No | No | No | **Built-in (clean API)** |
| Grasping | GPD, Dex-Net | Basic | No | No | **Built-in grasp sampler** |
| Robot configs | 100+ | ~12 | ~20 | 3 | **Target: 50+ (V3)** |
| Constraints | Full | Limited | Full | None | **Full** |
| Python bindings | moveit_py (pybind11) | Native Python | pydrake | Python wrapper | **PyO3** |
| Install | `apt install` + config | Docker + CUDA + PyTorch | Bazel | CMake + header | **`cargo add kinetic`** |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         APPLICATION LAYER                               │
│  User code · HORUS nodes · Talos simulator · Python scripts             │
├─────────────────────────────────────────────────────────────────────────┤
│                         TASK LAYER (kinetic_task)                        │
│  Task graph · Pick/Place · Multi-step sequencing · Behavior primitives  │
├──────────────────────────┬──────────────────────────────────────────────┤
│    GLOBAL PLANNING       │        LOCAL / REACTIVE                      │
│   (kinetic_planning)     │       (kinetic_reactive)                     │
│                          │                                              │
│  SIMD RRT-Connect        │  RMP policies (<1ms)                        │
│  Constrained RRT         │  Jacobian-based servo                       │
│  Cartesian planner       │  Singularity avoidance                      │
│  GPU trajectory opt      │  Collision deceleration                     │
│  GCS convex planner      │  Joint limit damping                        │
│  Learned seeders         │  Velocity/acceleration limits               │
├──────────────────────────┴──────────────────────────────────────────────┤
│                      TRAJECTORY LAYER (kinetic_trajectory)              │
│  Time parameterization · TOTP · Trapezoidal · Spline interpolation     │
│  Jerk-limited profiles · Trajectory blending · Waypoint scheduling     │
├─────────────────────────────────────────────────────────────────────────┤
│                      COLLISION LAYER (kinetic_collision)                 │
│  SIMD broadphase · GJK/EPA narrowphase · Signed distance fields        │
│  Self-collision pairs · Attached objects · Allowed collision matrix     │
│  Point tree (<10ns queries) · Mesh-mesh · Convex decomposition         │
├─────────────────────────────────────────────────────────────────────────┤
│                      SCENE LAYER (kinetic_scene)                        │
│  Planning scene · World model · Collision objects (box/sphere/mesh)     │
│  Attached objects · Robot state · Sensor integration                   │
│  Pointcloud → collision world · Depth → SDF                           │
├─────────────────────────────────────────────────────────────────────────┤
│                      KINEMATICS LAYER (kinetic_kinematics)              │
│  FK · Jacobian · DLS · FABRIK · OPW analytical · Subproblem decomp    │
│  Joint limits · Workspace analysis · Manipulability                    │
├─────────────────────────────────────────────────────────────────────────┤
│                      ROBOT LAYER (kinetic_robot)                        │
│  URDF/MJCF/SDF loading · Joint tree · Link geometry · Collision model  │
│  Planning groups · End-effector definitions · Tool transforms          │
│  50+ robot configs (UR, Panda, KUKA, ABB, Fanuc, xArm, Sawyer...)    │
├─────────────────────────────────────────────────────────────────────────┤
│                      CORE TYPES (kinetic_core)                          │
│  Pose (SE3) · Twist · Wrench · Trajectory · Constraint · Error        │
│  SIMD-friendly data layout (SoA) · f64 precision throughout           │
└─────────────────────────────────────────────────────────────────────────┘

         ┌─────────────────┐  ┌────────────────┐  ┌──────────────────┐
         │  horus_kinetic   │  │ kinetic_python  │  │  kinetic_gpu     │
         │  Zero-copy IPC   │  │ PyO3 bindings   │  │  wgpu compute    │
         │  HORUS topics    │  │ numpy interop   │  │  parallel seeds  │
         │  TF integration  │  │ pip installable │  │  GPU collision   │
         └─────────────────┘  └────────────────┘  └──────────────────┘
```

Each layer depends only on layers below it. Bridge crates (`horus_kinetic`, `kinetic_python`, `kinetic_gpu`) sit beside the stack, not inside it. The core stack has **zero** framework dependencies — no HORUS, no async runtime, no GPU driver.

---

## Crate Map

```
kinetic/
├── kinetic_core/               # Core types, SIMD data layouts
├── kinetic_robot/              # Robot model, URDF/MJCF/SDF, planning groups
├── kinetic_kinematics/         # FK, IK (DLS, FABRIK, OPW, analytical decomp)
├── kinetic_collision/          # SIMD collision, point trees, SDF, ACM
├── kinetic_scene/              # Planning scene, world model, perception
├── kinetic_trajectory/         # Time parameterization, profiles, blending
├── kinetic_planning/           # Global planners (SIMD-RRT, Cartesian, constrained)
├── kinetic_reactive/           # Local reactive control (RMP, servo, avoidance)
├── kinetic_task/               # Task planning (pick/place, multi-step sequences)
├── kinetic_grasp/              # Grasp generation (geometric, antipodal, suction)
├── kinetic_gpu/                # wgpu compute (trajectory opt, GPU collision)
├── kinetic_python/             # PyO3 Python bindings
├── horus_kinetic/              # HORUS integration (topics, TF, nodes)
├── kinetic_robots/             # Pre-configured robot packages
├── kinetic/                    # Umbrella crate (re-exports everything)
│
├── examples/
├── tests/
├── benches/
└── robot_configs/              # URDF + config TOML per robot
    ├── franka_panda/
    ├── ur5e/
    ├── ur10e/
    ├── kuka_iiwa7/
    ├── kuka_iiwa14/
    ├── abb_irb1200/
    ├── fanuc_crx10ia/
    ├── xarm6/
    ├── xarm7/
    ├── kinova_gen3/
    ├── sawyer/
    ├── fetch/
    └── ...                     # 50+ robots (see Robot Configs section)
```

**Dependency budget:**
- `kinetic_core`: `nalgebra 0.33`, `thiserror`
- `kinetic_robot`: `kinetic_core`, `urdf-rs`, `serde`, `toml`
- `kinetic_kinematics`: `kinetic_core`, `nalgebra`
- `kinetic_collision`: `kinetic_core`, `parry3d-f64`, `rayon` (optional)
- `kinetic_scene`: `kinetic_core`, `kinetic_collision`, `kinetic_robot`
- `kinetic_trajectory`: `kinetic_core`, `nalgebra`
- `kinetic_planning`: `kinetic_core`, `kinetic_kinematics`, `kinetic_collision`, `kinetic_scene`, `rand`
- `kinetic_reactive`: `kinetic_core`, `kinetic_kinematics`, `kinetic_collision`
- `kinetic_task`: `kinetic_planning`, `kinetic_reactive`, `kinetic_grasp`
- `kinetic_grasp`: `kinetic_core`, `kinetic_collision`, `kinetic_kinematics`
- `kinetic_gpu`: `kinetic_core`, `wgpu`, `bytemuck`
- `kinetic_python`: `kinetic` (all), `pyo3`, `numpy`
- `horus_kinetic`: `kinetic`, `horus`

Critical: **`parry3d-f64`** (not `parry3d`). The f64 variant exists in Dimforge's ecosystem and eliminates the precision concerns of the original spec. This is a hard requirement for robotics-grade collision checking.

---

## API Design

### Level 0: One-Line Planning

```rust
let traj = kinetic::plan("panda", start_joints, Goal::pose(target))?;
```

Uses a pre-configured Panda robot, default scene (self-collision only), default planner (SIMD RRT-Connect + shortcutting). This is the "just make it work" entry point.

### Level 1: Minimal (5 Lines)

```rust
use kinetic::{Robot, Planner, Goal};

let robot = Robot::from_urdf("panda.urdf")?;
let planner = Planner::new(&robot)?;
let trajectory = planner.plan(&start_joints, &Goal::pose(target_pose))?;
let joints_at_halfway = trajectory.sample(0.5);
```

### Level 2: Planning Scene with Obstacles

```rust
use kinetic::{Robot, Planner, Scene, Shape, Goal};

let robot = Robot::from_urdf("ur5e.urdf")?;
let mut scene = Scene::new(&robot);

// Add collision objects
scene.add("table", Shape::box_shape(1.0, 0.6, 0.02), table_pose);
scene.add("bin", Shape::mesh("bin.stl"), bin_pose);
scene.add("wall", Shape::plane(Vec3::X), wall_pose);

// Plan with scene
let planner = Planner::with_scene(&robot, &scene)?;
let traj = planner.plan(&start, &Goal::pose(pick_pose))?;

// Attach object to end-effector (moves with robot, excluded from EE collision)
scene.attach("cup", Shape::cylinder(0.04, 0.12), grasp_offset, "panda_hand");

// Re-plan with attached object
let traj2 = planner.plan(&current, &Goal::pose(place_pose))?;
```

### Level 3: Constrained Planning

```rust
use kinetic::{Planner, Goal, Constraint};

// Keep end-effector upright while moving (waiter carrying a tray)
let constraint = Constraint::orientation(
    "panda_hand",           // link
    Vec3::Z,                // axis to constrain
    0.1,                    // tolerance in radians (~5.7°)
);

let traj = planner
    .with_constraints(&[constraint])
    .plan(&start, &Goal::pose(target))?;

// Cartesian straight-line path
let cartesian_traj = planner.plan_cartesian(
    &start,
    &Goal::pose(target),
    0.01,                   // max step (1cm)
    0.0,                    // jump threshold (0 = disabled)
)?;
```

### Level 4: Reactive Servo Control

```rust
use kinetic::reactive::{Servo, ServoConfig, TwistCommand};

let servo = Servo::new(&robot, &scene, ServoConfig {
    rate_hz: 500.0,
    collision_check_hz: 100.0,
    singularity_threshold: 0.02,
    slowdown_distance: 0.15,      // start slowing 15cm from obstacles
    stop_distance: 0.03,          // hard stop at 3cm
    ..Default::default()
});

// Joystick-style control: send twist commands
servo.send(TwistCommand::linear(0.1, 0.0, 0.0))?;    // 10cm/s in X
servo.send(TwistCommand::angular(0.0, 0.0, 0.5))?;   // rotate 0.5 rad/s around Z

// Joint jog: nudge individual joints
servo.jog_joint(0, 0.01)?;  // joint 0 += 0.01 rad

// Pose tracking: track a moving target
servo.track_pose(moving_target_pose)?;

// Safety: servo automatically decelerates near obstacles and singularities
let state = servo.state();
println!("collision margin: {:.3}m, manipulability: {:.3}",
         state.min_distance, state.manipulability);
```

### Level 5: Task Planning (Pick and Place)

```rust
use kinetic::task::{Task, PickPlace, GraspStrategy};

let task = Task::pick_and_place(
    &robot,
    &scene,
    PickPlace {
        object: "cup",
        grasp: GraspStrategy::top_down(0.08),   // 8cm gripper opening
        pick_pose: cup_pose,
        place_pose: shelf_pose,
        approach_distance: 0.10,                 // 10cm approach
        retreat_distance: 0.05,                  // 5cm retreat
        lift_height: 0.15,                       // 15cm lift after grasp
    },
)?;

// Plan entire sequence: approach → grasp → lift → transfer → place → retreat
let solution = task.plan()?;

// Execute step by step
for stage in solution.stages() {
    println!("{}: {} waypoints, {:.1}ms",
             stage.name, stage.trajectory.len(), stage.duration_ms());
}

// Or execute as one trajectory
let full_traj = solution.as_trajectory()?;
```

### Level 6: Custom Task Graph

```rust
use kinetic::task::{TaskGraph, Stage, Connect, Cartesian, MoveTo};

let mut task = TaskGraph::new(&robot, &scene);

// Build custom multi-step task
let home = task.add(Stage::current_state());
let pre_pick = task.add(Stage::move_to(Goal::named("pre_pick")));
let approach = task.add(Stage::cartesian(Goal::relative(Vec3::new(0.0, 0.0, -0.10))));
let grasp = task.add(Stage::close_gripper(0.04));
let attach = task.add(Stage::attach_object("bolt"));
let lift = task.add(Stage::cartesian(Goal::relative(Vec3::new(0.0, 0.0, 0.15))));
let transfer = task.add(Stage::move_to(Goal::named("pre_place")));
let place_approach = task.add(Stage::cartesian(Goal::relative(Vec3::new(0.0, 0.0, -0.08))));
let release = task.add(Stage::open_gripper(0.08));
let detach = task.add(Stage::detach_object("bolt"));
let retreat = task.add(Stage::cartesian(Goal::relative(Vec3::new(0.0, 0.0, 0.10))));
let return_home = task.add(Stage::move_to(Goal::named("home")));

// Stages execute in order of insertion by default
let solution = task.plan()?;
```

### Level 7: HORUS Integration

```rust
use horus::prelude::*;
use horus_kinetic::{PlannerNode, ServoNode, SceneNode};

// Planning node: subscribes to plan requests, publishes trajectories
#[horus::node]
struct MotionPlanner {
    #[input(topic = "/joint_states")]
    joints: Input<JointState>,

    #[input(topic = "/plan_request")]
    request: Input<PlanRequest>,

    #[output(topic = "/trajectory")]
    trajectory: Output<JointTrajectory>,

    planner: PlannerNode,
}

// Scene node: builds collision world from sensor topics
#[horus::node]
struct WorldModel {
    #[input(topic = "/camera/points")]
    pointcloud: Input<PointCloud>,

    #[output(topic = "/planning_scene")]
    scene: Output<PlanningScene>,

    scene_builder: SceneNode,  // kinetic_scene under the hood
}

// Servo node: real-time reactive control at HORUS IPC speed
#[horus::node]
struct ReactiveController {
    #[input(topic = "/twist_cmd")]
    twist: Input<Twist>,

    #[input(topic = "/joint_states")]
    joints: Input<JointState>,

    #[output(topic = "/joint_command")]
    command: Output<JointCommand>,

    servo: ServoNode,  // kinetic_reactive under the hood
}
```

### Level 8: Python

```python
import kinetic

# One-liner
traj = kinetic.plan("panda", start, kinetic.Goal.pose(target))

# Full pipeline
robot = kinetic.Robot.from_urdf("ur5e.urdf")
scene = kinetic.Scene(robot)
scene.add_box("table", size=[1.0, 0.6, 0.02], pose=table_pose)
scene.add_pointcloud(camera_points, voxel_size=0.01)

planner = kinetic.Planner(robot, scene)
traj = planner.plan(start, kinetic.Goal.pose(target))

# Servo
servo = kinetic.Servo(robot, scene, rate_hz=500)
servo.send_twist([0.1, 0, 0, 0, 0, 0])  # linear + angular

# Pick and place
task = kinetic.PickPlace(robot, scene,
    object="cup", pick=cup_pose, place=shelf_pose)
solution = task.plan()

# Trajectory execution
for t in np.linspace(0, 1, 100):
    joints = traj.sample(t)
```

---

## The SIMD Architecture (KINETIC's Core Advantage)

VAMP (Kavraki Lab, 2023) proved that the bottleneck in sampling-based planning is collision checking, and that SIMD vectorization of collision queries delivers 1000x speedups over scalar code. KINETIC adopts this architecture natively in Rust.

### Key Insight: Collision-Affording Point Trees (CAPT)

Instead of BVH trees (O(log N) traversal per query), CAPT represents the environment as a spatial tree where each node stores the minimum radius a sphere can have and still be collision-free at that position. This enables:

- **<10ns average query time** per point on scenes with thousands of obstacles
- Queries vectorize perfectly across SIMD lanes (8 queries/cycle on AVX2, 16 on AVX-512)
- Entire robot collision check in ~100ns by evaluating all robot spheres simultaneously

### Data Layout: Structure of Arrays (SoA)

```rust
// BAD: Array of Structures (traditional, cache-unfriendly for SIMD)
struct Sphere { x: f64, y: f64, z: f64, radius: f64 }
let spheres: Vec<Sphere>;  // XYRZXYRZ... in memory

// GOOD: Structure of Arrays (SIMD-friendly, cache-friendly)
struct SpheresSoA {
    x: Vec<f64>,       // XXXX... contiguous
    y: Vec<f64>,       // YYYY... contiguous
    z: Vec<f64>,       // ZZZZ... contiguous
    radius: Vec<f64>,  // RRRR... contiguous
}
// 4 spheres checked per AVX2 cycle (8 on AVX-512)
```

All collision geometry in KINETIC uses SoA layout. FK computation outputs joint positions in SoA format. This is the single most important architectural decision — it flows through every layer.

### SIMD Tiers

| Platform | Width | Spheres/Cycle | Approach |
|----------|-------|---------------|----------|
| ARM NEON | 128-bit | 2 | `std::arch::aarch64::*` |
| SSE4.2 | 128-bit | 2 | `std::arch::x86_64::*` |
| AVX2 | 256-bit | 4 | `std::arch::x86_64::*` |
| AVX-512 | 512-bit | 8 | `std::arch::x86_64::*` (where available) |
| WASM SIMD | 128-bit | 2 | `std::arch::wasm32::*` |

Rust's `std::simd` (stabilizing) provides the portable fallback. Architecture-specific intrinsics provide peak performance. Runtime detection (`is_x86_feature_detected!`) selects the widest available path.

### Performance Pipeline

```
Typical 6-DOF planning query (simple scene):

1. RRT-Connect iteration:
   ├── Random sample in C-space:                 ~5 ns
   ├── FK for sampled config (SIMD):             ~50 ns
   ├── Collision check all spheres (SIMD CAPT):  ~100 ns
   ├── Nearest neighbor (k-d tree):              ~80 ns
   └── Total per iteration:                      ~250 ns

2. Typical iterations to solution:                200-400

3. Total planning time:                           50-100 us

4. Path shortcutting (50 iterations):             ~25 us
5. Cubic spline smoothing:                        ~10 us

Total: ~100 us (simple scene)
       ~1-20 ms (cluttered scene with narrow passages)
```

Compare: MoveIt2 RRT-Connect ~170ms (simple), ~1200ms (cluttered).

---

## Collision Detection

### The f64 Requirement

The original spec identified `parry3d`'s f32 precision as a risk. KINETIC uses two backends:

1. **SIMD Sphere-Tree (primary, custom):** SoA-layout sphere approximations of robot links, evaluated with SIMD point-tree queries. This is the hot path — used for 99% of collision checks during planning. f64 throughout.

2. **parry3d-f64 (secondary, exact):** For mesh-mesh distance queries, exact contact points, and distance-to-obstacle computations. Called only when the sphere-tree reports "near collision" (within safety margin). The `parry3d-f64` crate provides the same API as `parry3d` but with f64 internally.

This two-tier approach gives SIMD speed (nanoseconds) for the planning loop while maintaining mesh-exact precision for safety-critical checks.

### Planning Scene

The planning scene is the world model that tracks all collision geometry:

```rust
use kinetic::scene::{Scene, Shape, ObjectState};

let mut scene = Scene::new(&robot);

// Primitive shapes
scene.add("table", Shape::cuboid(1.0, 0.6, 0.02), table_pose);
scene.add("bin", Shape::cylinder(0.15, 0.30), bin_pose);
scene.add("wall", Shape::half_space(Vec3::X, 0.0));  // infinite plane

// Meshes (auto-convex decomposed for collision)
scene.add("engine_block", Shape::mesh("engine.stl"), engine_pose);

// Attach object to robot (carries with end-effector)
scene.attach("bolt", Shape::cylinder(0.005, 0.03), grasp_tf, "panda_hand");
scene.detach("bolt", place_pose);

// Allowed Collision Matrix (skip known-safe pairs)
scene.allow_collision("panda_hand", "bolt");  // grasped object
scene.allow_collision("panda_link0", "table"); // robot base touches table

// Update from sensors
scene.update_from_pointcloud(&points, voxel_size: 0.01);
scene.update_from_depth(&depth_image, &camera_intrinsics, &camera_pose);

// Query
let min_dist = scene.min_distance_to_robot(&joint_values)?;
let contacts = scene.contact_points(&joint_values, margin: 0.02)?;
let in_collision = scene.check_collision(&joint_values)?;
```

### Perception → Collision World Pipeline

MoveIt2 uses Octomap (octree voxelization). KINETIC provides two paths:

**1. Point Tree (CAPT):** Directly ingests point clouds. No voxelization step. Each point is a collision sphere of configurable radius. The CAPT structure enables <10ns queries per robot sphere. This is faster than Octomap for planning because there's no octree traversal.

```rust
scene.add_pointcloud("camera_0", &points, PointCloudConfig {
    sphere_radius: 0.01,    // 1cm collision spheres per point
    max_points: 100_000,    // downsample if larger
    remove_floor: true,     // RANSAC floor removal
    crop_box: Some(workspace_bounds),
});
```

**2. Signed Distance Field (GPU):** For GPU trajectory optimization, a 3D SDF is computed from depth images or point clouds. This lives in `kinetic_gpu` and is optional.

```rust
use kinetic_gpu::sdf::SignedDistanceField;

let sdf = SignedDistanceField::from_depth(
    &depth_image, &camera_intrinsics, &camera_pose,
    resolution: 0.01,  // 1cm voxels
    bounds: workspace_bounds,
)?;

// SDF enables gradient-based trajectory optimization
let distance = sdf.query(point)?;       // signed distance
let gradient = sdf.gradient(point)?;    // distance gradient (for optimization)
```

---

## Kinematics

### Forward Kinematics (SIMD-Vectorized)

FK is the innermost loop of planning. KINETIC computes FK in SoA format to vectorize across SIMD lanes:

```rust
// Single FK call
let ee_pose = robot.fk(&joint_values)?;

// Batch FK (SIMD-vectorized): compute FK for 8 configs at once
let poses = robot.fk_batch(&[config1, config2, config3, ..., config8])?;

// Jacobian (6 x N geometric Jacobian)
let jac = robot.jacobian(&joint_values)?;

// Manipulability (Yoshikawa index)
let m = robot.manipulability(&joint_values)?;
```

**Performance target:** <1us for single 6-DOF FK. <200ns amortized in batch mode.

Source: Extract and refactor 1,098 lines from `horus-sim3d/src/robot/kinematics.rs`, upgrade f32→f64, remove Bevy dependencies.

### Inverse Kinematics

KINETIC provides four IK solvers covering all use cases:

| Solver | DOF | Speed | Use Case |
|--------|-----|-------|----------|
| **OPW Analytical** | 6-DOF spherical wrist | <50us | UR, Panda, ABB, KUKA, Fanuc — covers 90% of industrial arms |
| **Subproblem Decomposition** | Any 6-DOF | <100us | Automatic analytical decomposition (EAIK-inspired). Finds closed-form where possible. |
| **DLS (Damped Least Squares)** | Any | ~500us | General iterative. Handles redundant robots (7+ DOF). |
| **FABRIK** | Any (chain) | ~300us | Fast iterative for chain robots. Good for interactive/visualization. |

Source: DLS and FABRIK from `horus-sim3d/src/robot/ik.rs` (647 lines). OPW is new. Subproblem decomposition is new.

```rust
use kinetic::ik::{IKSolver, IKConfig};

// Automatic solver selection based on robot geometry
let solutions = robot.ik(target_pose, IKConfig::default())?;
// Returns Vec<JointValues> — multiple solutions for analytical solvers

// Specific solver
let solutions = robot.ik(target_pose, IKConfig {
    solver: IKSolver::OPW,
    max_solutions: 8,           // up to 8 analytical solutions for 6-DOF
    check_limits: true,         // filter by joint limits
    check_collision: Some(&scene),  // filter by collision
    ..Default::default()
})?;

// Redundancy resolution for 7-DOF
let solutions = robot.ik(target_pose, IKConfig {
    solver: IKSolver::DLS { damping: 0.05 },
    null_space_objective: Some(NullSpace::manipulability()),  // maximize manipulability
    seed: Some(current_joints),  // start near current config
    ..Default::default()
})?;
```

### Why This Beats MoveIt2's IK

MoveIt2's IK landscape is fragmented and painful:
- **KDL**: Default, slow, often fails. Users universally replace it.
- **IKFast**: Fast analytical, but requires OpenRAVE to generate solver code. OpenRAVE is unmaintained Python 2 software. Generating an IKFast solver is a multi-hour ordeal.
- **BioIK**: Good, but separate ROS package, complex config.
- **TracIK/pick_ik**: Community-maintained, quality varies.

KINETIC ships one `robot.ik()` call that auto-selects the best solver. Analytical IK for common geometries is built-in — no code generation step, no external tools.

---

## Planners

### Tier 1: SIMD RRT-Connect (Default, CPU)

The workhorse. Bidirectional RRT with SIMD-vectorized collision checking.

```rust
use kinetic::planning::{Planner, RRTConfig};

let planner = Planner::rrt_connect(&robot, &scene, RRTConfig {
    max_iterations: 10_000,
    step_size: 0.05,               // 5% of joint range per step
    goal_bias: 0.05,               // 5% probability of sampling goal
    shortcut_iterations: 100,      // path shortcutting passes
    smooth: true,                  // cubic spline smoothing
    timeout: Duration::from_millis(50),
    ..Default::default()
});

let traj = planner.plan(&start, &Goal::pose(target))?;
```

**Performance target:** <100us p50 simple scene, <20ms p50 cluttered.

### Tier 2: Constrained RRT (CPU)

Planning with task-space constraints (keep orientation, follow path, avoid regions):

```rust
use kinetic::planning::constraints::*;

let constraints = vec![
    // Keep gripper upright (e.g., carrying liquid)
    Constraint::orientation("end_effector", Axis::Z, tolerance: 0.05),

    // Stay above table surface
    Constraint::position_bound("end_effector", Axis::Z, min: 0.02, max: f64::MAX),

    // Joint 3 limited range during motion
    Constraint::joint(3, min: -1.0, max: 1.0),

    // Keep point visible to camera (visibility constraint)
    Constraint::visibility("camera_link", target_point, cone_angle: 0.5),
];

let traj = planner
    .with_constraints(&constraints)
    .plan(&start, &Goal::pose(target))?;
```

Implementation: Rejection sampling with constraint projection. Each RRT sample is projected onto the constraint manifold using iterative Jacobian-based correction (same approach as OMPL's constrained planning).

### Tier 3: Cartesian Planner (CPU)

Straight-line end-effector motion in task space. Essential for approach/retreat motions, welding, painting, deburring.

```rust
// Linear path in Cartesian space
let traj = planner.plan_cartesian(&start, &Goal::pose(target), CartesianConfig {
    max_step: 0.005,              // 5mm max step between waypoints
    jump_threshold: 1.4,          // detect joint jumps (IK discontinuities)
    avoid_collisions: true,
    ..Default::default()
})?;

println!("Achieved {:.1}% of path", traj.fraction * 100.0);

// Relative motion (move 10cm in Z, in end-effector frame)
let traj = planner.plan_cartesian_relative(
    &start,
    &Goal::relative(Vec3::new(0.0, 0.0, 0.10)),
    CartesianConfig::default(),
)?;
```

### Tier 4: GPU Trajectory Optimization (Optional)

cuRobo-style parallel-seed trajectory optimization, but on cross-vendor GPU via wgpu:

```rust
use kinetic_gpu::{GpuOptimizer, GpuConfig};

let optimizer = GpuOptimizer::new(&robot, &scene, GpuConfig {
    num_seeds: 128,                // parallel trajectory seeds
    timesteps: 64,                 // waypoints per trajectory
    max_iterations: 100,           // L-BFGS iterations per seed
    collision_weight: 100.0,
    smoothness_weight: 10.0,       // jerk penalty
    ..Default::default()
})?;

let traj = optimizer.optimize(&start, &Goal::joints(goal_joints))?;
// Returns smoothest collision-free trajectory from 128 parallel candidates
```

**Why wgpu, not CUDA:** cuRobo's NVIDIA-only requirement locks out AMD, Intel, and Apple users. wgpu compute shaders run on Vulkan (NVIDIA/AMD/Intel), Metal (Apple), and DX12 (Windows). The performance gap vs raw CUDA is ~20-30% — acceptable for 5x vendor reach.

**Performance target:** <50ms on discrete GPU, <200ms on integrated GPU.

### Tier 5: GCS Convex Planner (CPU, Globally Optimal)

Inspired by Drake's Graphs of Convex Sets (Marcucci et al., Science Robotics 2023). For structured environments where globally optimal paths matter (logistics, manufacturing cells):

```rust
use kinetic::planning::gcs::{ConvexDecomposition, GCSPlanner};

// Pre-compute convex free-space regions (expensive, do once per environment)
let regions = ConvexDecomposition::iris(&scene, IrisConfig {
    num_regions: 50,
    min_volume: 0.001,
})?;

// Plan over convex regions (globally optimal, fast after decomposition)
let planner = GCSPlanner::new(&robot, &regions);
let traj = planner.plan(&start, &goal)?;
```

### Planner Selection Guide

| Scenario | Recommended Planner | Expected Latency |
|----------|-------------------|-----------------|
| Simple free-space motion | SIMD RRT-Connect | <100us |
| Cluttered environment | SIMD RRT-Connect | 1-20ms |
| Keep orientation constraint | Constrained RRT | 5-50ms |
| Straight-line approach/retreat | Cartesian planner | <1ms |
| Smoothest possible trajectory | GPU trajectory optimization | 30-50ms |
| Known static environment | GCS convex planner | 10-100ms (after precompute) |
| Real-time tracking | Reactive servo (not a planner) | <1ms |

---

## Reactive Control (kinetic_reactive)

MoveIt Servo runs as a ROS2 node receiving twist commands over DDS. KINETIC's reactive layer runs in-process with zero serialization overhead.

### Riemannian Motion Policies (RMP)

Inspired by NVIDIA's RMPflow. Multiple competing objectives are combined via Riemannian geometry into a single consistent motion:

```rust
use kinetic::reactive::{RMP, Policy};

let mut rmp = RMP::new(&robot);

// Add policies (each is a "force field" in task space)
rmp.add(Policy::reach_target(target_pose, gain: 10.0));
rmp.add(Policy::avoid_obstacles(&scene, influence: 0.3, gain: 50.0));
rmp.add(Policy::avoid_self_collision(gain: 20.0));
rmp.add(Policy::joint_limit_avoidance(margin: 0.1, gain: 15.0));
rmp.add(Policy::singularity_avoidance(threshold: 0.02, gain: 5.0));
rmp.add(Policy::damping(0.5));  // velocity damping for stability

// Each tick: combine all policies → joint accelerations → integrate → joint command
let command = rmp.compute(&current_joints, dt: 0.002)?; // 500 Hz
```

**Performance target:** <200us per tick. At HORUS IPC speed (167ns), the reactive loop runs at >1kHz with margin.

### Servo Mode

For teleoperation and interactive control:

```rust
use kinetic::reactive::{Servo, ServoConfig};

let servo = Servo::new(&robot, &scene, ServoConfig {
    rate_hz: 500.0,
    input_type: InputType::Twist,        // or JointJog or PoseTracking
    collision_check_hz: 100.0,
    singularity_threshold: 0.02,
    slowdown_distance: 0.15,
    stop_distance: 0.03,
    smoothing_filter: SmoothingFilter::ExponentialMovingAverage { alpha: 0.3 },
    velocity_limits: robot.velocity_limits().scale(0.5),  // 50% of max
    acceleration_limits: robot.acceleration_limits().scale(0.3),
});

// Three input modes (matching MoveIt Servo):
servo.send_twist(twist)?;                    // Cartesian velocity
servo.send_joint_jog(joint_index, delta)?;   // single joint nudge
servo.track_pose(target_pose)?;              // track moving target

// Read state
let state = servo.state();
assert!(state.min_obstacle_distance > 0.03);
```

---

## Trajectory Processing (kinetic_trajectory)

### Time Parameterization

Raw planner output is a geometric path (joint positions only). Time parameterization adds velocities, accelerations, and timing.

```rust
use kinetic::trajectory::{TimeParameterize, Profile};

// Time-Optimal Time Parameterization (TOTP) — fastest possible
let timed = trajectory.time_parameterize(Profile::TimeOptimal {
    velocity_limits: robot.velocity_limits(),
    acceleration_limits: robot.acceleration_limits(),
})?;

// Trapezoidal velocity profile — industrial standard
let timed = trajectory.time_parameterize(Profile::Trapezoidal {
    max_velocity: 1.0,        // rad/s
    max_acceleration: 2.0,    // rad/s^2
})?;

// Jerk-limited (S-curve) — smooth for delicate tasks
let timed = trajectory.time_parameterize(Profile::JerkLimited {
    max_velocity: 1.0,
    max_acceleration: 2.0,
    max_jerk: 10.0,           // rad/s^3
})?;

// Spline interpolation — smooth through waypoints
let timed = trajectory.time_parameterize(Profile::CubicSpline)?;

// Query at any time
let wp = timed.sample_at(Duration::from_secs_f64(1.5));
println!("t=1.5s: pos={:?}, vel={:?}, acc={:?}",
         wp.positions, wp.velocities, wp.accelerations);
```

### Trajectory Blending

Blend between trajectories without stopping (important for continuous motion):

```rust
use kinetic::trajectory::blend;

let blended = blend(&traj1, &traj2, blend_radius: 0.05)?;
// Smooth transition from traj1 to traj2 with 5cm blend radius
```

---

## Task Planning (kinetic_task)

MoveIt Task Constructor uses a complex stage-based architecture with generators, propagators, connectors, and wrappers. KINETIC provides the same capability with a cleaner API.

### Built-in Task Primitives

```rust
use kinetic::task::*;

// Pick
let pick = Task::pick(&robot, &scene, PickConfig {
    object: "cup",
    grasp_poses: vec![top_down, side_grasp],  // multiple grasp options
    approach: Approach::linear(Vec3::NEG_Z, 0.10),
    retreat: Approach::linear(Vec3::Z, 0.05),
    gripper_open: 0.08,
    gripper_close: 0.04,
})?;

// Place
let place = Task::place(&robot, &scene, PlaceConfig {
    object: "cup",
    target_pose: shelf_pose,
    approach: Approach::linear(Vec3::NEG_Z, 0.08),
    retreat: Approach::linear(Vec3::Z, 0.10),
    gripper_open: 0.08,
})?;

// Compose
let full_task = Task::sequence(vec![
    Task::move_to(&robot, Goal::named("home")),
    pick,
    Task::move_to(&robot, Goal::named("transfer_point")),
    place,
    Task::move_to(&robot, Goal::named("home")),
]);

let solution = full_task.plan()?;
```

### Grasp Generation (kinetic_grasp)

```rust
use kinetic::grasp::{GraspGenerator, GripperType};

let generator = GraspGenerator::new(&robot, GripperType::Parallel {
    max_opening: 0.08,
    finger_depth: 0.03,
});

// Generate grasp candidates from object shape
let grasps = generator.from_shape(
    &Shape::cylinder(0.04, 0.12),  // cup
    object_pose,
    GraspConfig {
        num_candidates: 100,
        approach_axis: Vec3::NEG_Z,
        rank_by: GraspMetric::ForceClosureQuality,
        check_collision: Some(&scene),
        check_reachability: Some(&robot),
    },
)?;

// Grasps are ranked by quality — best first
for grasp in &grasps[..5] {
    println!("score: {:.3}, approach: {:?}", grasp.quality, grasp.approach_direction);
}
```

Grasp generation methods:
- **Geometric antipodal:** Parallel jaw grasps from shape geometry
- **Surface sampling:** Sample contact points on mesh surface
- **Top-down:** Vertical approach for bin picking
- **Side grasp:** Horizontal approach for shelf picking
- **Suction:** Center-of-mass based for suction grippers

---

## Robot Configurations (kinetic_robots)

A robot config is a URDF + a TOML file defining planning groups, IK solver selection, collision pairs, and named poses:

```toml
# robot_configs/franka_panda/kinetic.toml
[robot]
name = "franka_panda"
urdf = "panda.urdf"
manufacturer = "Franka Emika"
dof = 7

[planning_group.arm]
chain = ["panda_link0", "panda_link8"]
joints = ["panda_joint1", "panda_joint2", "panda_joint3", "panda_joint4",
          "panda_joint5", "panda_joint6", "panda_joint7"]

[planning_group.hand]
joints = ["panda_finger_joint1", "panda_finger_joint2"]

[end_effector]
name = "hand"
parent_link = "panda_link8"
parent_group = "arm"
tcp_transform = { xyz = [0, 0, 0.1034], rpy = [0, 0, 0] }

[ik]
solver = "opw"  # or "dls", "fabrik", "analytical_decomposition"

[named_poses]
home = [0, -0.785, 0, -2.356, 0, 1.571, 0.785]
ready = [0, -0.5, 0, -2.0, 0, 1.5, 0.785]
extended = [0, 0, 0, 0, 0, 0, 0]

[collision]
self_collision_pairs = "auto"  # auto-generate from URDF adjacency
padding = 0.01                 # 1cm safety padding on all links

[limits]
velocity_scale = 1.0
acceleration_scale = 1.0
```

### Target Robot List (50+ by V3)

**Industrial Arms:**
| Robot | DOF | Manufacturer | IK Solver |
|-------|-----|-------------|-----------|
| Franka Panda | 7 | Franka Emika | DLS (redundant) |
| UR3e | 6 | Universal Robots | OPW |
| UR5e | 6 | Universal Robots | OPW |
| UR10e | 6 | Universal Robots | OPW |
| UR16e | 6 | Universal Robots | OPW |
| UR20 | 6 | Universal Robots | OPW |
| UR30 | 6 | Universal Robots | OPW |
| KUKA iiwa 7 | 7 | KUKA | DLS (redundant) |
| KUKA iiwa 14 | 7 | KUKA | DLS (redundant) |
| KUKA KR6 | 6 | KUKA | OPW |
| ABB IRB 1200 | 6 | ABB | OPW |
| ABB IRB 4600 | 6 | ABB | OPW |
| ABB YuMi | 7+7 | ABB | DLS (dual-arm) |
| Fanuc CRX-10iA | 6 | Fanuc | OPW |
| Fanuc LR Mate 200iD | 6 | Fanuc | OPW |
| Yaskawa GP7 | 6 | Yaskawa | OPW |
| Yaskawa HC10 | 6 | Yaskawa | OPW |
| Denso VS-068 | 6 | Denso | OPW |
| Staubli TX2-60 | 6 | Staubli | OPW |

**Cobot / Research Arms:**
| Robot | DOF | Manufacturer | IK Solver |
|-------|-----|-------------|-----------|
| xArm 5 | 5 | UFACTORY | Analytical Decomp |
| xArm 6 | 6 | UFACTORY | OPW |
| xArm 7 | 7 | UFACTORY | DLS |
| Kinova Gen3 | 7 | Kinova | DLS |
| Kinova Gen3 Lite | 6 | Kinova | OPW |
| Sawyer | 7 | Rethink | DLS |
| Baxter (dual) | 7+7 | Rethink | DLS |
| Dobot CR5 | 6 | Dobot | OPW |
| Flexiv Rizon 4 | 7 | Flexiv | DLS |
| Mecademic Meca500 | 6 | Mecademic | OPW |
| Elephant Robotics myCobot | 6 | Elephant | OPW |

**Mobile Manipulators:**
| Robot | DOF | Manufacturer |
|-------|-----|-------------|
| Fetch | 7 + base | Fetch Robotics |
| Tiago | 7 + base | PAL Robotics |
| Stretch RE2 | 5 + base | Hello Robot |
| PR2 (dual) | 7+7 + base | Willow Garage |

**Open-Source / Education:**
| Robot | DOF | Source |
|-------|-----|-------|
| SO-ARM100 | 5 | HuggingFace LeRobot |
| Koch v1.1 | 6 | HuggingFace LeRobot |
| ALOHA (dual) | 7+7 | Google/Stanford |
| Open Manipulator-X | 4 | ROBOTIS |
| WidowX 250 | 6 | Trossen |
| ViperX 300 | 6 | Trossen |

---

## Honest Comparison: KINETIC V1 vs MoveIt2

| Feature | MoveIt2 | KINETIC V1 | KINETIC V3 (Target) |
|---------|---------|------------|---------------------|
| FK/IK | Yes (KDL, IKFast, BioIK) | **Yes (OPW, DLS, FABRIK, decomp)** | Same + learned |
| Collision | Yes (FCL, f64) | **Yes (SIMD sphere + parry3d-f64)** | Same + GPU SDF |
| Motion planning | 50+ OMPL planners | **SIMD RRT-Connect + shortcut + smooth** | + Constrained, Cartesian, GPU opt, GCS |
| Planning speed | 170ms-5s | **<100us-20ms** | Same |
| Trajectory quality | Jerky (RRT) | Smooth (shortcut + spline) | Optimal (GPU + GCS) |
| Cartesian planning | Yes | **Yes** | Same |
| Constraints | Full (5 types) | Orientation + position | **Full** |
| Planning scene | Yes (FCL world) | **Yes (SIMD point tree + parry3d-f64)** | + Perception pipeline |
| Attached objects | Yes | **Yes** | Same |
| ACM | Yes | **Yes** | Same |
| Perception → collision | Yes (Octomap) | No | **Yes (pointcloud + depth)** |
| Real-time servo | Yes (MoveIt Servo) | **Yes (RMP + Servo)** | Same |
| Task planning | Yes (MTC) | No | **Yes (built-in)** |
| Grasp generation | Yes (GPD, Dex-Net) | No | **Yes (built-in)** |
| Time parameterization | Yes (TOTP, iterative) | **Yes (TOTP, trapez, jerk-limited)** | Same |
| Robot configs | 100+ | 15 | **50+** |
| Python bindings | Yes (moveit_py) | No | **Yes (PyO3)** |
| GPU planning | Via cuRobo plugin | No | **Yes (wgpu, cross-vendor)** |
| Ecosystem maturity | 12+ years | **New** | 2+ years |

**The gap:** V1 closes the technical feature gap for core planning. V3 achieves full parity plus advantages. The ecosystem gap (community, tutorials, production deployments) takes years and users.

---

## Development Phases

### Phase 1: Core (Weeks 1-8) — "Better than `k` crate, competitive with OMPL alone"

| Week | Crate | Deliverable |
|------|-------|-------------|
| 1 | `kinetic_core` | Pose, Trajectory, Constraint, Error types. SoA data layouts. |
| 2 | `kinetic_robot` | URDF/MJCF/SDF loading. Joint tree. Planning groups. Link geometry. 5 robot configs (Panda, UR5e, UR10e, KUKA iiwa7, xArm6). |
| 3-4 | `kinetic_kinematics` | FK (SIMD), Jacobian, DLS, FABRIK, OPW analytical. Extract from horus-sim3d, refactor f32→f64. Benchmarks. |
| 4-5 | `kinetic_collision` | SIMD sphere-tree (CAPT-inspired). Self-collision pairs. parry3d-f64 for exact queries. ACM. Benchmarks. |
| 5-7 | `kinetic_planning` | SIMD RRT-Connect. Path shortcutting. Cubic spline smoothing. Planner facade. Cartesian planner (linear + circular). |
| 7-8 | `kinetic_trajectory` | TOTP, trapezoidal profile, jerk-limited (S-curve), spline interpolation. |
| 8 | `kinetic` | Umbrella crate. Examples. Benchmarks. CI. README. `cargo publish`. |

**Phase 1 success criteria:**
- `cargo add kinetic` works
- 5-line hello world plans a Panda arm in <1ms
- FK <1us, IK (OPW) <50us, Planning <100us (simple), <20ms (cluttered)
- >95% planning success on MotionBenchMaker dataset
- Zero `todo!()`, zero `unimplemented!()`
- Published to crates.io

### Phase 2: Manipulation Stack (Weeks 9-16) — "Replace MoveIt2 for single-arm manipulation"

| Week | Crate | Deliverable |
|------|-------|-------------|
| 9-10 | `kinetic_scene` | Full planning scene. Attached objects. Primitive + mesh collision objects. ACM management. Sensor update API (pointcloud, depth). |
| 10-11 | `kinetic_reactive` | RMP policies. Servo mode (twist, joint jog, pose tracking). Collision deceleration. Singularity avoidance. |
| 12-13 | `kinetic_planning` (expand) | Constrained RRT (orientation, position, joint, visibility). Planning with scene. |
| 14-15 | `kinetic_grasp` | Geometric grasp generation (antipodal, top-down, side). Gripper types (parallel, suction). |
| 15-16 | `kinetic_task` | Task primitives: pick, place, move_to, cartesian. Task sequencing. Pick-and-place pipeline. |
| 16 | `kinetic_robots` | Expand to 30 robot configs. Named poses. Per-robot tuned parameters. |
| 16 | `horus_kinetic` | HORUS bridge. PlannerNode, ServoNode, SceneNode. Topic wiring. |

**Phase 2 success criteria:**
- Pick-and-place demo with Panda + table scene works end-to-end
- Servo mode runs at 500Hz with collision avoidance
- Constrained planning passes standard benchmark (upright orientation through cluttered scene)
- 30 robot configs with validated FK/IK accuracy
- HORUS integration demo with zero-copy topic communication

### Phase 3: Production + Python (Weeks 17-24) — "Production teams can adopt"

| Week | Crate | Deliverable |
|------|-------|-------------|
| 17-19 | `kinetic_python` | Full PyO3 bindings. All core APIs. numpy interop. pip installable. |
| 19-21 | `kinetic_gpu` | wgpu trajectory optimization (parallel seeds). GPU SDF from depth/pointcloud. GPU collision checking. |
| 21-22 | `kinetic_planning` (expand) | GCS convex planner. IK subproblem decomposition (EAIK-inspired). |
| 22-23 | `kinetic_robots` | Expand to 50+ robot configs. Validation suite (FK accuracy vs manufacturer data). |
| 23-24 | Docs + benchmarks | Full documentation. Benchmarks against MoveIt2, VAMP, cuRobo. Migration guide from MoveIt2. |

**Phase 3 success criteria:**
- `pip install kinetic` works
- GPU trajectory optimization <50ms on discrete GPU
- 50+ robot configs, all FK-validated
- Published benchmark comparison: KINETIC vs MoveIt2 vs VAMP
- At least one external user (non-SOFTMATA) has integrated KINETIC

### Phase 4: Leapfrog (Weeks 25-36) — "Things MoveIt2 cannot do"

| Deliverable | Description |
|-------------|-------------|
| Dual-arm planning | Synchronized planning for bimanual robots (ABB YuMi, Baxter, ALOHA) |
| Learned trajectory seeding | Diffusion model generates seed trajectories, GPU optimization refines |
| Whole-body planning | Mobile manipulator planning (base + arm simultaneously) |
| Force/impedance integration | Plan with force constraints (compliant assembly, polishing) |
| `kinetic_task` expansion | Behavior tree integration. Complex multi-step tasks. Error recovery. |
| MoveIt2 ROS2 bridge | Drop-in replacement: KINETIC exposes MoveIt2-compatible ROS2 action servers |

---

## Performance Targets (Benchmarked)

All benchmarks use `criterion` and run in CI. Regressions >20% block merge.

| Operation | Target | Benchmark |
|-----------|--------|-----------|
| FK (6-DOF, single) | <1 us | `benches/fk.rs` |
| FK (6-DOF, batch 8) | <200 ns/config | `benches/fk.rs` |
| Jacobian (6-DOF) | <2 us | `benches/fk.rs` |
| IK — OPW analytical | <50 us | `benches/ik.rs` |
| IK — DLS (convergence) | <500 us | `benches/ik.rs` |
| IK — FABRIK | <300 us | `benches/ik.rs` |
| Collision check (SIMD, 6-DOF + 10 obstacles) | <500 ns | `benches/collision.rs` |
| Distance query (parry3d-f64) | <50 us | `benches/collision.rs` |
| SIMD RRT-Connect (simple scene) | <100 us p50 | `benches/planning.rs` |
| SIMD RRT-Connect (cluttered scene) | <20 ms p50 | `benches/planning.rs` |
| Cartesian planning (20cm line) | <500 us | `benches/planning.rs` |
| RMP reactive tick | <200 us | `benches/reactive.rs` |
| GPU trajectory optimization | <50 ms | `benches/gpu.rs` |
| GCS planning (pre-decomposed) | <100 ms | `benches/planning.rs` |
| Time parameterization (TOTP) | <1 ms | `benches/trajectory.rs` |
| Full pipeline (plan + smooth + TOTP) | <200 us (simple) | `benches/e2e.rs` |

### Benchmark Against Competitors

KINETIC will publish head-to-head benchmarks on the MotionBenchMaker dataset:

| Scenario | MoveIt2 RRT-Connect | VAMP | cuRobo | KINETIC (target) |
|----------|-------------------|------|--------|-----------------|
| Table pick (simple) | 170ms | 35us | 45ms | **<100us** |
| Shelf pick (cluttered) | 1200ms | 16ms | 45ms | **<20ms** |
| Narrow passage | 3000ms+ | 50ms | 100ms | **<50ms** |

---

## Risks

### Why This Will Work

1. **VAMP proved the architecture.** SIMD-vectorized planning achieves microsecond latency on CPU. We are implementing proven research in Rust — the ideal language for SIMD data-oriented code.

2. **The ecosystem code exists.** FK/IK from horus-sim3d (1,745 lines). Collision algorithms from Hephaestus (GJK, EPA, SAT). parry3d-f64 from Dimforge. We are assembling, not inventing.

3. **MoveIt2's pain points are real.** Installation is painful. Configuration is arcane. IKFast requires unmaintained tools. Planning is slow. The Python API is limited. Every MoveIt2 user has war stories.

4. **Rust robotics is growing.** The `k` crate has 40K+ downloads for just FK/IK. ros2-client, r2r, and rclrs bring Rust into the ROS2 ecosystem. KINETIC fills the #1 missing piece.

5. **Internal demand.** Talos needs embedded planning. HORUS needs a manipulation stack. ENKI needs simulation planning. NEXUS needs gym-compatible planning. KINETIC serves four SOFTMATA products.

### Why This Could Fail

1. **Scope.** This spec describes a 36-week project across 14 crates. If velocity drops or priorities shift, the later phases may not ship. Mitigation: Phase 1 is independently useful and publishable.

2. **SIMD performance claims need validation.** VAMP's numbers are from optimized C++ with hand-tuned AVX2. Rust's auto-vectorization may not match hand-written intrinsics without manual SIMD code. Risk: 2-5x slower than VAMP, which is still 100x faster than MoveIt2.

3. **parry3d-f64 maturity.** The f64 variant of parry3d is less battle-tested than the f32 variant. We may encounter bugs. Mitigation: Parry3d is actively maintained by Dimforge; f64 bugs can be reported and fixed upstream.

4. **GPU trajectory optimization in wgpu is unproven.** cuRobo's CUDA kernels are highly optimized. wgpu compute shaders have higher dispatch overhead and less mature tooling. Risk: 2-3x slower than cuRobo. Mitigation: still 10x faster than CPU-only MoveIt2.

5. **Adoption depends on ecosystem momentum.** A technically superior library with zero users loses to a mediocre library with 10,000 users. Mitigation: Python bindings (Phase 3) unlock the largest user base. Robot configs lower adoption friction. MoveIt2 ROS2 bridge (Phase 4) enables gradual migration.

6. **MoveIt2 is not standing still.** PickNik is integrating cuRobo, Drake, and LLMs into MoveIt Pro. By the time KINETIC reaches V3, MoveIt2 may have closed some of the performance gap. Mitigation: KINETIC's architectural advantage (Rust + SIMD + standalone) is structural, not feature-based.

---

## AEGIS Integration

KINETIC produces geometric paths and time-parameterized trajectories. AEGIS closes the loop with feedback control:

```
KINETIC                              AEGIS
───────                              ─────
plan() → geometric path
time_parameterize() → trajectory
                    ──────────►  PID/impedance tracking
                                 EKF state estimation
                                 Safety monitoring
                                 ◄──────────
                    replanning trigger
```

KINETIC's `Trajectory` outputs `(position, velocity, acceleration)` at each timestep. AEGIS's controller tracks these references. KINETIC's reactive layer provides fallback when AEGIS detects deviation beyond threshold.

---

## References

### Core Research

- LaValle & Kuffner, "RRT-Connect" (2000) — primary planning algorithm
- Kalakrishnan et al., "STOMP" (ICRA 2011) — trajectory optimization
- Pieper, "Kinematics of Manipulators Under Computer Control" (1968) — OPW IK
- Marcucci et al., "Motion Planning around Obstacles with Convex Optimization" (Science Robotics, 2023) — GCS
- Cheng et al., "VAMP: Vector-Accelerated Motion Planning" (2023) — SIMD architecture
- Sundaralingam et al., "cuRobo: Parallelized Collision-Free Robot Motion Generation" (2023) — GPU optimization
- Ratliff et al., "RMPflow: A Computational Graph for Automatic Motion Policy Generation" (2020) — reactive policies
- Huang et al., "DiffusionSeeder" (2024) — learned trajectory initialization
- Ostermeier, "EAIK: Analytical IK by Subproblem Decomposition" — automatic analytical IK
- Werner et al., "IKFlow: Generating Diverse IK Solutions" (RA-L 2022) — neural IK diversity
- Harada et al., "RT Core Collision Detection" (ICRA 2025) — hardware-accelerated collision

### Libraries

- [parry3d-f64](https://parry.rs/) — f64 collision detection (Rust)
- [nalgebra](https://nalgebra.org/) — linear algebra (Rust)
- [urdf-rs](https://crates.io/crates/urdf-rs) — URDF parsing (Rust)
- [wgpu](https://wgpu.rs/) — cross-vendor GPU compute (Rust)
- [rayon](https://docs.rs/rayon/) — data parallelism (Rust)

### Existing Code (SOFTMATA)

- `registry_packages/horus-sim3d/src/robot/kinematics.rs` — FK, Jacobian, DH (1,098 lines)
- `registry_packages/horus-sim3d/src/robot/ik.rs` — DLS, FABRIK (647 lines)
- `talos/simulator/crates/hephaestus-contact/` — GJK, EPA, SAT collision
- `talos/simulator/crates/talos-robot/` — URDF/MJCF/SDF loading
