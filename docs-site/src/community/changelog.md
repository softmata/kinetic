# Changelog

All notable changes to kinetic are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `PlannerOutputInvalid` error variant for safety gate failures
- `SingularityLockup` error variant for servo singularity detection
- `DimensionMismatch` error variant with context string
- `IKSolution.degraded` flag for solver fallback detection
- `IKSolution.condition_number` for singularity proximity monitoring
- `ExecutionConfig::safe()` constructor with auto-configured safety
- `SafetyWatchdog` for real-time execution monitoring
- `PlanExecuteLoop` for automatic replanning on failure

## [0.1.0] - 2026-03-29

Initial release of the kinetic motion planning library.

### Added

**Core (`kinetic-core`)**
- `Pose`, `JointValues`, `Trajectory`, `Goal`, `Constraint` types
- `KineticError` with 20+ variants and `is_retryable()`/`is_input_error()`
- `PlannerConfig` with `default()`, `realtime()`, `offline()` presets
- `FrameTree` and `StampedTransform` for coordinate frame management
- `Twist` and `Wrench` spatial vector types

**Robot (`kinetic-robot`)**
- URDF parser with full joint/link/collision geometry support
- SRDF parser for planning groups and collision pairs
- MJCF and SDF loaders
- 52 built-in robot configurations
- `Robot::from_name()`, `Robot::from_urdf()`, `Robot::from_path()`
- `kinetic.toml` configuration format

**Kinematics (`kinetic-kinematics`)**
- Forward kinematics (`forward_kinematics`, `forward_kinematics_all`)
- Jacobian computation (`jacobian`)
- Manipulability index (`manipulability`)
- 6 IK solvers: DLS, FABRIK, OPW, Subproblem, Subproblem7DOF, BioIK
- Cached IK solver for repeated queries
- IKFast codegen support
- Batch IK (`solve_ik_batch`)
- Workspace analysis (`ReachabilityMap`)

**Collision (`kinetic-collision`)**
- SIMD-vectorized sphere-sphere collision (AVX-512/AVX2/SSE4.1)
- CAPT broadphase acceleration
- `RobotSphereModel` with `SphereGenConfig` (coarse/default/fine)
- `CollisionEnvironment` with margin-based checking
- `AllowedCollisionMatrix` for self-collision filtering
- Contact point computation

**Planning (`kinetic-planning`)**
- 8 sampling-based planners: RRT-Connect, RRT*, BiRRT*, BiTRRT, EST, KPIECE, PRM, GCS
- Constrained RRT with orientation/position constraints
- Cartesian planner (linear and arc paths)
- Dual-arm planner (Independent, Synchronized, Coordinated modes)
- Path shortcutting and B-spline smoothing
- `Planner` facade with auto chain detection
- `PlannerType` enum for algorithm selection
- One-line `plan()` convenience function
- `PlanningPipeline` with pre/post-processors and planner racing

**Trajectory (`kinetic-trajectory`)**
- Trapezoidal time parameterization
- TOTP (Time-Optimal Path Parameterization)
- Jerk-limited S-curve profiles
- Cubic spline interpolation
- Trajectory blending
- CSV and JSON import/export
- `TrajectoryValidator` with per-joint limit checking
- `ExecutionMonitor` with deviation detection

**Scene (`kinetic-scene`)**
- Collision object management (add, remove, attach, detach)
- Shape primitives: Cuboid, Cylinder, Sphere, HalfSpace
- Point cloud ingestion with voxel downsampling
- Depth image to point cloud conversion
- Octree spatial indexing
- Outlier removal (statistical and radius)

**Reactive (`kinetic-reactive`)**
- Servo controller (Twist, JointJog, PoseTracking modes)
- `ServoConfig` presets: teleop, tracking, precise
- Riemannian Motion Policies (RMP)
- Collision deceleration
- Singularity avoidance
- Smoothing filters (EMA, Butterworth)

**Task (`kinetic-task`)**
- Pick and Place task planning
- Multi-stage task sequencing
- Grasp generation
- Approach/retreat motion primitives
- Gripper command integration

**GPU (`kinetic-gpu`)**
- cuRobo-style parallel trajectory optimization via wgpu
- Batch FK on GPU
- GPU collision checking with SDF
- `GpuConfig` presets: balanced, speed, quality
- `CpuOptimizer` fallback for non-GPU environments

**Execution (`kinetic-execution`)**
- `CommandSink` trait for hardware drivers
- `FeedbackSource` trait for encoder reading
- `RealTimeExecutor` with 500 Hz command streaming
- `SimExecutor` for testing without hardware
- `LogExecutor` for recording commanded trajectories
- `SafetyWatchdog` with configurable timeout actions

**HORUS Bridge (`horus-kinetic`)**
- `PlannerNode` for motion planning via HORUS IPC
- `ServoNode` for reactive control via HORUS IPC
- `SceneNode` for collision world via HORUS IPC
- Channel-based API for standalone use

**Python (`kinetic-python`)**
- PyO3 bindings with numpy interop
- Robot, Planner, Scene, Servo, Task, Trajectory classes
- One-line `kinetic.plan()` function
- Grasp generation and gripper types
- Frame tree management

### Supported Robots (52)
ABB IRB1200, ABB IRB4600, ABB YuMi (left/right), ALOHA (left/right),
Baxter (left/right), Denso VS068, Dobot CR5, Elite EC66, Fanuc CRX-10iA,
Fanuc LR Mate 200iD, Fetch, Flexiv Rizon4, Franka Panda, Jaco2 6DOF,
Kinova Gen3, Kinova Gen3 Lite, Koch v1, KUKA iiwa7, KUKA iiwa14, KUKA KR6,
LeRobot SO-100, Meca500, myCobot 280, Niryo Ned2, Open Manipulator X,
Open Manipulator P, PR2, Sawyer, SO-ARM100, Staubli TX2-60, Stretch RE2,
Techman TM5-700, TIAGo, Trossen PX100, Trossen RX150, Trossen WX250s,
UR3e, UR5e, UR10e, UR16e, UR20, UR30, ViperX 300, WidowX 250,
xArm5, xArm6, xArm7, Yaskawa GP7, Yaskawa HC10.
