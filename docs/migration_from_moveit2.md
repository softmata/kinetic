# Migration Guide: MoveIt2 to KINETIC

## Why Migrate?

| Feature | MoveIt2 | KINETIC |
|---------|---------|---------|
| Language | C++ / ROS 2 | Rust (no ROS required) |
| FK latency | ~5-10 us | <500 ns |
| Planning (simple) | ~170 ms | <100 us - 1s (config dependent) |
| Collision check | ~50-100 us | <500 ns (SIMD) |
| Dependencies | ROS 2, OMPL, FCL, Eigen | nalgebra, parry3d |
| Python bindings | moveit_py (ROS bridge) | Native PyO3 + numpy |
| Safety | C++ memory bugs | Rust memory safety |

## Concept Mapping

### Robot Description

**MoveIt2:**
```yaml
# MoveIt Setup Assistant generates:
# - SRDF (planning groups, disabled collisions)
# - kinematics.yaml (IK solver config)
# - joint_limits.yaml
# - moveit_controllers.yaml
```

**KINETIC:**
```rust
// Load by name — KINETIC ships built-in configs for 52 robots:
let robot = Robot::from_name("ur5e")?;  // No config files needed

// Or load from URDF + SRDF (MoveIt2 configs work directly):
let robot = Robot::from_urdf_srdf("robot.urdf", "robot.srdf")?;
// SRDF provides: planning groups, disabled collision pairs, named poses

// Or just URDF with auto-detected planning groups:
let robot = Robot::from_urdf("path/to/robot.urdf")?;
```

### Planning Scene

**MoveIt2:**
```cpp
auto scene = planning_scene_monitor->getPlanningScene();
moveit_msgs::msg::CollisionObject obj;
obj.id = "table";
obj.primitives.push_back(box_shape);
obj.primitive_poses.push_back(table_pose);
scene->processCollisionObjectMsg(obj);
```

**KINETIC:**
```rust
let mut scene = Scene::new(&robot)?;
scene.add("table", Shape::Cuboid(0.5, 0.3, 0.01), table_pose);
```

### Motion Planning

**MoveIt2:**
```cpp
auto move_group = MoveGroupInterface(node, "manipulator");
move_group.setStartStateToCurrentState();
move_group.setJointValueTarget(target_joints);
auto [success, plan] = move_group.plan();
```

**KINETIC:**
```rust
let planner = Planner::new(&robot)?;
let result = planner.plan(&start_joints, &Goal::Joints(goal))?;
```

### Time Parameterization

**MoveIt2:**
```cpp
trajectory_processing::TimeOptimalTrajectoryGeneration totg;
totg.computeTimeStamps(trajectory, max_velocity, max_acceleration);
```

**KINETIC:**
```rust
let timed = totp(&path, &vel_limits, &accel_limits, 0.01)?;
// Or simpler:
let timed = trapezoidal(&path, 1.0, 2.0)?;
```

### Servo / Teleoperation

**MoveIt2:**
```cpp
// moveit_servo requires ROS 2 topics:
// - /servo_node/delta_twist_cmds
// - /servo_node/joint_states
auto servo = moveit_servo::Servo(node, servo_params);
servo.start();
// Publish twist via ROS topic
```

**KINETIC:**
```rust
let mut servo = Servo::new(&robot, &scene, ServoConfig::default())?;
let cmd = servo.send_twist(&twist)?;
// cmd.positions and cmd.velocities are ready to send to robot
```

### IK Solvers

**MoveIt2:**
```yaml
# kinematics.yaml
manipulator:
  kinematics_solver: kdl_kinematics_plugin/KDLKinematicsPlugin
  kinematics_solver_search_resolution: 0.005
  kinematics_solver_timeout: 0.005
```

**KINETIC:**
```rust
// DLS (Damped Least Squares) — default, works for any robot
let config = IKConfig::dls().with_max_iterations(300);

// FABRIK — faster for simple chains
let config = IKConfig::fabrik();

// OPW — analytical, for 6-DOF industrial robots (UR, KUKA, ABB)
if is_opw_compatible(&robot) {
    let params = OPWParameters::from_robot(&robot)?;
    let solutions = solve_opw(&params, &target)?;
}
```

### Collision Detection

**MoveIt2:**
```cpp
collision_detection::CollisionRequest req;
collision_detection::CollisionResult res;
scene->checkCollision(req, res);
bool in_collision = res.collision;
```

**KINETIC:**
```rust
let in_collision = scene.check_collision(&joints)?;
let min_dist = scene.min_distance_to_robot(&joints)?;
```

## Step-by-Step Migration

### 1. Replace Robot Loading

```rust
// Before (MoveIt2): URDF + SRDF + yaml configs via ROS parameter server
// After (KINETIC):
let robot = Robot::from_name("ur5e")?;  // Built-in (52 robots)
// or with your existing MoveIt2 SRDF:
let robot = Robot::from_urdf_srdf("robot.urdf", "robot.srdf")?;
// or just URDF with auto-detected groups:
let robot = Robot::from_urdf("path/to/robot.urdf")?;
```

### 2. Replace Planning

```rust
let planner = Planner::new(&robot)?;
let goal = Goal::Joints(JointValues(target.to_vec()));
let result = planner.plan(&current_joints, &goal)?;
let traj = trapezoidal(&result.waypoints, 1.0, 2.0)?;
```

### 3. Replace Scene Management

```rust
let mut scene = Scene::new(&Arc::new(robot))?;
scene.add("table", Shape::Cuboid(0.5, 0.3, 0.01), pose);
scene.attach("part", Shape::Sphere(0.02), grasp_tf, "tool0");
```

### 4. Replace Servo

```rust
let mut servo = Servo::new(&robot_arc, &scene_arc, ServoConfig::default())?;
// In your control loop:
let cmd = servo.send_twist(&twist)?;
send_to_robot(cmd.positions, cmd.velocities);
```

## What KINETIC Does NOT Replace

- **Hardware drivers**: KINETIC is planning-only. Use ros2_control or direct drivers.
- **Perception**: KINETIC doesn't do object detection. Feed it collision shapes.
- **Visualization**: Use your own 3D viewer. KINETIC outputs joint trajectories.
- **ROS integration**: KINETIC is ROS-independent. Wrap in a ROS 2 node if needed.

## Python Migration

**MoveIt2 (moveit_py):**
```python
from moveit.core.robot_model import RobotModel
from moveit.planning import MoveItPy

moveit = MoveItPy(node_name="moveit_py")
arm = moveit.get_planning_component("manipulator")
arm.set_goal_state(configuration_name="home")
plan = arm.plan()
```

**KINETIC:**
```python
import kinetic
import numpy as np

robot = kinetic.Robot("ur5e")
planner = kinetic.Planner(robot)
traj = planner.plan(current_joints, kinetic.Goal.joints(goal_joints))
```
