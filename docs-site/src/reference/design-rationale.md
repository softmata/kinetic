# Design Rationale

Why kinetic is designed the way it is.

## Why Rust?

Kinetic is written in Rust for three reasons:

**1. Performance without compromise.** Rust compiles to native code with
zero-cost abstractions. No garbage collector pauses, no interpreter overhead.
FK runs in under 1 microsecond, collision checking in under 500 nanoseconds.
These numbers matter when planning must complete in 10ms.

**2. Memory safety without runtime cost.** The borrow checker eliminates
data races, use-after-free, and buffer overflows at compile time. Motion
planning code is safety-critical -- a memory bug in the planner can cause
a robot to collide. Rust makes entire classes of bugs impossible.

**3. Fearless concurrency.** GPU optimization runs compute shaders while
the CPU handles IK. Dual-arm planning uses parallel tree expansion. Servo
mode runs collision checks asynchronously. Rust's ownership model makes
concurrent code correct by construction.

C++ offers similar performance but without memory safety guarantees.
Python is too slow for real-time planning (100-1000x slower for numerical
code). Rust provides both.

## Why No ROS2 Dependency?

Kinetic deliberately has zero ROS2 dependencies. This is a conscious design
choice, not an oversight.

**1. ROS2 is middleware, not a requirement.** Motion planning is a
mathematical problem: compute a collision-free path from A to B. This
requires no publish-subscribe system, no node graph, no launch files.
Adding ROS2 as a dependency forces every user to install and configure
a middleware stack they may not need.

**2. Deployment flexibility.** Kinetic runs anywhere Rust compiles:
embedded Linux, server-grade workstations, CI pipelines, WebAssembly.
ROS2 constrains deployment to platforms with DDS and ament support.

**3. Testing simplicity.** Unit testing a Rust function that computes
IK requires only `cargo test`. Testing a ROS2 node requires spinning
up executors, creating topics, waiting for discovery. Kinetic's tests
run in milliseconds, not seconds.

**4. Integration is opt-in.** The `horus-kinetic` bridge provides HORUS
IPC integration when needed. A ROS2 bridge (`rmw_horus`) is planned.
Users who need ROS2 integration can add it; users who do not are not
burdened by it.

## Why SIMD Collision Checking?

Traditional collision libraries (FCL, Bullet, HPP-FCL) use mesh-based
representations with BVH trees. Kinetic uses a sphere-based model with
SIMD-vectorized distance computations.

**1. Spheres are fast.** Sphere-sphere distance is one subtraction,
one dot product, and one comparison. SIMD processes 4-16 sphere pairs
per instruction. This is 10-50x faster than GJK/EPA on convex meshes.

**2. Spheres are conservative.** A sphere approximation always
over-estimates the collision volume. This is the safe direction for
motion planning -- the planner may reject valid configurations but
will never accept colliding ones.

**3. Resolution is configurable.** `SphereGenConfig::coarse()` uses
~5 spheres per link for maximum speed. `SphereGenConfig::fine()` uses
~20 for higher accuracy. The user controls the tradeoff.

**4. GPU-compatible.** Sphere collision maps directly to GPU compute
shaders. The same sphere model used for CPU planning is used for
GPU trajectory optimization.

## Why 10 IK Solvers?

Different IK solvers excel at different robot geometries:

**OPW (Ortho-Parallel Wrist):** Analytical closed-form solution for 6-DOF
robots with spherical wrists (UR, ABB IRB, KUKA KR, Fanuc). Returns all
8 solutions in under 5 microseconds. No iterative convergence issues.

**Subproblem decomposition:** Analytical for 6-DOF robots with intersecting
wrist axes. Uses Paden-Kahan subproblems to decompose the 6-DOF IK into
a sequence of 1-DOF and 2-DOF sub-problems. Returns up to 16 solutions.

**Subproblem 7-DOF:** Sweeps the redundant joint and solves analytically
at each sample. Handles 7-DOF robots (Panda, KUKA iiwa) without iterative
convergence issues.

**DLS (Damped Least Squares):** General-purpose iterative solver that works
for any DOF. Handles singularities via damping. The fallback solver.

**FABRIK:** Forward And Backward Reaching IK. Fast for position-only tasks.
Particularly good for long kinematic chains.

**BioIK, IKFast, Cached IK, SQP:** Specialized solvers for specific use
cases (biomechanical models, code-generated solvers, repeated queries,
constrained optimization).

A single IK solver cannot be optimal for all robots. By providing 10
solvers and an auto-selection mechanism, kinetic achieves the best
performance for each robot geometry automatically.

## Competitive Positioning

### vs MoveIt2

MoveIt2 is the established motion planning framework in the ROS ecosystem.
Kinetic provides an alternative for users who want:

- **No ROS dependency** -- kinetic works standalone
- **Faster performance** -- 5-100x faster FK, IK, and collision checking
- **GPU optimization** -- cuRobo-style parallel trajectory optimization
- **Simpler configuration** -- single TOML file vs YAML/SRDF/launch
- **Rust safety** -- memory-safe, data-race-free by construction

MoveIt2 has a larger ecosystem (ROS integration, Rviz visualization, Setup
Assistant) and more community support. Choose MoveIt2 if you are already
invested in ROS2 and need its ecosystem. Choose kinetic if you need
standalone performance, GPU optimization, or Rust safety guarantees.

### vs cuRobo

NVIDIA's cuRobo provides GPU-accelerated motion generation. Kinetic's
`kinetic-gpu` crate provides similar capabilities with key differences:

- **Hardware agnostic** -- wgpu supports Vulkan, Metal, DX12 (not just CUDA)
- **CPU fallback** -- `CpuOptimizer` works without a GPU
- **Full planning stack** -- sampling-based planners + GPU optimization
- **No Isaac Sim dependency** -- standalone library

cuRobo has deeper NVIDIA integration (Isaac Sim, CUDA kernels, TensorRT).
Choose cuRobo if you are fully committed to the NVIDIA ecosystem. Choose
kinetic for cross-platform GPU support and a complete planning stack.

### vs Drake

Drake is a mathematical robotics toolbox with a focus on optimization and
simulation. Kinetic is a focused motion planning library.

- **Faster planning** -- sampling-based planners for real-time use
- **Simpler API** -- `Planner::new(&robot)?.plan(&start, &goal)?`
- **More IK solvers** -- 10 solvers with auto-selection
- **No simulation** -- kinetic does not include a physics engine

Drake excels at optimization-based planning (GCS, trajectory optimization)
and includes a full multibody dynamics simulator. Choose Drake for
mathematical rigor and simulation. Choose kinetic for fast, practical
motion planning on real robots.

### vs OMPL

OMPL provides sampling-based planning algorithms without kinematics or
collision checking. Kinetic bundles everything into one stack.

- **Batteries included** -- FK, IK, collision, planning, trajectory, execution
- **No external deps** -- OMPL requires you to provide collision checking
- **Rust performance** -- faster than OMPL's C++ for equivalent algorithms
- **Integrated GPU** -- OMPL has no GPU support

OMPL has a wider variety of planning algorithms and a longer track record.
Choose OMPL (via MoveIt2) for algorithm diversity. Choose kinetic for an
integrated, zero-dependency planning stack.
