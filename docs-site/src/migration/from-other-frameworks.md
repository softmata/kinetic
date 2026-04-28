# From Other Frameworks

Concept mapping for cuRobo, Drake, OMPL, PyBullet, and Pinocchio users.

## cuRobo

NVIDIA's cuRobo provides GPU-accelerated motion generation using parallel
trajectory optimization. Kinetic's `kinetic-gpu` crate follows the same
architecture (parallel seeds, SDF collision, gradient-based optimization)
but runs on any GPU via wgpu (Vulkan/Metal/DX12), not just CUDA.

| cuRobo | Kinetic |
|--------|---------|
| `MotionGen` | `GpuOptimizer` or `Planner` |
| `RobotConfig` | `Robot` + `kinetic.toml` |
| `WorldConfig` | `Scene` + `CollisionEnvironment` |
| `MotionGenConfig` | `GpuConfig` |
| `CudaRobotModel` | `RobotSphereModel` |
| `plan_single()` | `planner.plan()` |
| `plan_batch()` | `GpuOptimizer::optimize()` |

**Key difference:** cuRobo requires CUDA and Isaac Sim. Kinetic runs on any
platform with a Vulkan/Metal GPU, or falls back to CPU. cuRobo is Python-first;
kinetic is Rust-first with Python bindings.

**Comparison:**
```python
# cuRobo
from curobo.wrap.reacher.motion_gen import MotionGen
motion_gen = MotionGen(MotionGenConfig.load_from_robot_config(robot_cfg))
result = motion_gen.plan_single(start, goal)

# Kinetic
import kinetic
planner = kinetic.Planner(kinetic.Robot("ur5e"))
result = planner.plan(start, kinetic.Goal.joints(goal))
```

## Drake

Drake is a C++ toolbox for model-based robotics with a focus on mathematical
rigor, optimization, and simulation. Kinetic is a focused motion planning
library without a physics simulator.

| Drake | Kinetic |
|-------|---------|
| `MultibodyPlant` | `Robot` |
| `SceneGraph` | `Scene` |
| `InverseKinematics` (program) | `solve_ik()` |
| `KinematicTrajectoryOptimization` | `GpuOptimizer` or `totp()` |
| `GcsTrajectoryOptimization` | `GCS` planner |
| `RigidBodyTree` | `KinematicChain` |
| `CalcJacobian` | `jacobian()` |

**Key difference:** Drake solves motion planning as a mathematical program
(convex optimization, mixed-integer programming). Kinetic uses sampling-based
planners (RRT, EST, KPIECE) for speed and GPU optimization for quality.
Drake includes a full physics simulator; kinetic does not.

**Comparison:**
```python
# Drake
from pydrake.multibody.inverse_kinematics import InverseKinematics
ik = InverseKinematics(plant)
ik.AddPositionConstraint(...)
result = Solve(ik.prog())

# Kinetic
import kinetic
robot = kinetic.Robot("ur5e")
planner = kinetic.Planner(robot)
joints = planner.ik(target_pose)
```

## OMPL

OMPL (Open Motion Planning Library) provides sampling-based planning algorithms.
Kinetic implements many of the same algorithms (RRT, RRT*, PRM, EST, KPIECE)
with Rust performance and integrated kinematics/collision.

| OMPL | Kinetic |
|------|---------|
| `SimpleSetup` | `Planner` |
| `SpaceInformation` | `KinematicChain` + `CollisionEnvironment` |
| `StateSpace` | Joint limits from `Robot` |
| `RRTConnect` | `PlannerType::RRTConnect` |
| `RRTstar` | `PlannerType::RRTStar` |
| `EST` | `PlannerType::EST` |
| `KPIECE1` | `PlannerType::KPIECE` |
| `PRM` | `PlannerType::PRM` |
| `StateValidityChecker` | `planner.is_in_collision()` |

**Key difference:** OMPL is a standalone planning library that requires you
to provide collision checking, FK, and IK externally (usually via MoveIt2).
Kinetic bundles all of these into a single stack with zero external dependencies.

**Comparison:**
```cpp
// OMPL
auto si = std::make_shared<SpaceInformation>(space);
si->setStateValidityChecker(myCollisionChecker);
auto planner = std::make_shared<RRTConnect>(si);
planner->solve(5.0);

// Kinetic
let planner = Planner::new(&robot)?
    .with_planner_type(PlannerType::RRTConnect);
let result = planner.plan(&start, &goal)?;
```

## PyBullet

PyBullet is a Python physics simulator commonly used for robotics research
and reinforcement learning. Kinetic is not a simulator but can replace
PyBullet's IK and motion planning functionality.

| PyBullet | Kinetic |
|----------|---------|
| `loadURDF` | `Robot::from_urdf("file.urdf")` |
| `calculateInverseKinematics` | `solve_ik()` |
| `calculateJacobian` | `jacobian()` |
| `getJointInfo` | `robot.joints` |
| `getLinkState` | `forward_kinematics()` |
| N/A (no planner) | `Planner::plan()` |
| `setCollisionFilter` | `AllowedCollisionMatrix` |

**Key difference:** PyBullet is a simulator with basic IK (damped least squares).
Kinetic is a planning stack with 10 IK solvers, 14 planners (8 surfaced in Python), and GPU optimization.
PyBullet runs in Python; kinetic's core is Rust with Python bindings.

**Comparison:**
```python
# PyBullet
import pybullet as p
robot_id = p.loadURDF("ur5e.urdf")
ik = p.calculateInverseKinematics(robot_id, ee_link, target_pos)

# Kinetic
import kinetic
robot = kinetic.Robot("ur5e")
planner = kinetic.Planner(robot)
ik = planner.ik(target_pose)
```

## Pinocchio

Pinocchio is a C++ rigid-body dynamics library focused on analytical
derivatives and efficient dynamics computations. Kinetic covers kinematics
and planning but not dynamics.

| Pinocchio | Kinetic |
|-----------|---------|
| `Model` | `Robot` |
| `Data` | (computed on the fly) |
| `forwardKinematics` | `forward_kinematics()` |
| `computeJointJacobians` | `jacobian()` |
| `ik` (via proxsuite) | `solve_ik()` |
| `computeMinverse` | N/A (use featherstone crate) |
| `GeometryModel` | `RobotSphereModel` |
| `computeCollisions` (HPP-FCL) | `CollisionEnvironment::check_collision()` |

**Key difference:** Pinocchio focuses on rigid-body dynamics (ABA, RNEA, CRBA)
with analytical derivatives for optimal control. Kinetic focuses on motion
planning with fast collision checking. For dynamics, use the `featherstone`
crate from the Softmata ecosystem.

**Comparison:**
```python
# Pinocchio
import pinocchio as pin
model = pin.buildModelFromUrdf("ur5e.urdf")
data = model.createData()
pin.forwardKinematics(model, data, q)
J = pin.computeJointJacobian(model, data, q, joint_id)

# Kinetic
import kinetic
robot = kinetic.Robot("ur5e")
planner = kinetic.Planner(robot)
pose = planner.fk(q)
```

## Summary Comparison

| Feature | kinetic | MoveIt2 | cuRobo | Drake | OMPL | PyBullet | Pinocchio |
|---------|---------|---------|--------|-------|------|----------|-----------|
| Language | Rust | C++ | Python | C++ | C++ | Python | C++ |
| FK/IK | Built-in | Plugin | Built-in | Built-in | External | Basic | Built-in |
| Planning | 14 (8 in Py) | OMPL | Optimizer | GCS/Opt | Planners | None | None |
| Collision | SIMD spheres | FCL | CUDA SDF | Drake | External | Bullet | HPP-FCL |
| GPU | wgpu | No | CUDA | No | No | No | No |
| Dynamics | No | No | No | Yes | No | Yes | Yes |
| ROS req | No | Yes | No | No | No | No | No |
