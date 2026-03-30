# API Cheatsheet

Quick reference grouped by task. Rust and Python shown side by side.

## Load Robot

```rust
// Rust
use kinetic::prelude::*;
let robot = Robot::from_name("ur5e")?;
let robot = Robot::from_urdf("path/to/robot.urdf")?;
let robot = Robot::from_urdf_string(urdf_str)?;
println!("DOF: {}", robot.dof);
```

```python
# Python
import kinetic
robot = kinetic.Robot("ur5e")
robot = kinetic.Robot.from_urdf("path/to/robot.urdf")
print(f"DOF: {robot.dof}")
```

## Forward Kinematics

```rust
// Rust
let chain = KinematicChain::extract(&robot, "base_link", "tool0")?;
let pose = forward_kinematics(&robot, &chain, &joints)?;
let all_poses = forward_kinematics_all(&robot, &chain, &joints)?;
println!("Position: {:?}", pose.translation());
```

```python
# Python
pose = robot.fk(joints)           # 4x4 numpy
poses = robot.batch_fk(configs)   # list of 4x4 (N configs)
```

## Inverse Kinematics

```rust
// Rust
let config = IKConfig::default().with_restarts(8);
let solution = solve_ik(&robot, &chain, &target_pose, &config)?;
println!("Converged: {}, error: {:.6}m", solution.converged, solution.position_error);
let joints = solution.joints;
```

```python
# Python
joints = robot.ik(target_pose)
result = robot.ik_config(target, solver="opw", mode="full6d", num_restarts=10)
# result: {'joints': array, 'converged': bool, 'position_error': float, ...}
```

## Batch IK

```rust
// Rust
let results = solve_ik_batch(&robot, &chain, &targets, &config);
// Vec<Option<IKSolution>> -- None for failed targets
```

```python
# Python
results = robot.batch_ik(target_list, solver="auto")
# list of dicts (or None for failures)
```

## Jacobian and Manipulability

```rust
// Rust
let j = jacobian(&robot, &chain, &joints)?;  // 6 x DOF matrix
let m = manipulability(&robot, &chain, &joints)?;
```

```python
# Python
jac = robot.jacobian(joints)       # (6, DOF) numpy
m = robot.manipulability(joints)   # float
```

## Plan

```rust
// Rust
let planner = Planner::new(&robot)?;
let result = planner.plan(&start, &Goal::Joints(goal))?;
let result = planner.plan(&start, &Goal::Pose(target_pose))?;
let result = planner.plan(&start, &Goal::Named("home".into()))?;
let result = planner.plan(&start, &Goal::Relative(offset_vec))?;

// One-liner
let result = kinetic::plan("ur5e", &start, &goal)?;
```

```python
# Python
planner = kinetic.Planner(robot)
goal = kinetic.Goal.joints(goal_array)
goal = kinetic.Goal.pose(position, quaternion)
goal = kinetic.Goal.named("home")
traj = planner.plan(start, goal)
```

## Cartesian / Dual-Arm Planning

```rust
// Rust
let dual = DualArmPlanner::new(&robot, left_chain, right_chain, DualArmMode::Synchronized)?;
let goal = DualGoal { left: Goal::Named("home".into()), right: Goal::Named("home".into()) };
let result = dual.plan(&left_start, &right_start, &goal)?;
```

```python
# Python
dual = kinetic.DualArmPlanner(robot, "left_arm", "right_arm", mode="synchronized")
result = dual.plan(start_left, start_right, goal_left, goal_right)
# result: {'left_trajectory': Trajectory, 'right_trajectory': Trajectory, ...}
```

## Scene

```rust
// Rust
let mut scene = Scene::new(&robot);
scene.add("table", Shape::Cuboid(0.8, 0.6, 0.02), table_pose);
scene.add("bottle", Shape::Cylinder(0.03, 0.15), bottle_pose);
scene.attach("tool", Shape::Sphere(0.02), grasp_tf, "tool0");
scene.remove("bottle");
```

```python
# Python
scene = kinetic.Scene(robot)
scene.add("table", kinetic.Shape.cuboid(0.8, 0.6, 0.02), pose)
scene.attach("tool", kinetic.Shape.sphere(0.02), grasp_tf, "tool0")
```

## Servo

```rust
// Rust
use kinetic::reactive::{Servo, ServoConfig};
let servo = Servo::new(&robot, &scene, ServoConfig::teleop());
let cmd = servo.send_twist(&twist)?;
// cmd.positions, cmd.velocities
```

```python
# Python
servo = kinetic.Servo(robot, scene)
cmd = servo.send_twist(twist)
```

## Trajectory

```rust
// Rust
let timed = trapezoidal(&waypoints, &vel_limits, &accel_limits)?;
let timed = totp(&waypoints, &vel_limits, &accel_limits)?;
trajectory_to_csv_file(&timed, "output.csv")?;
trajectory_to_json_file(&timed, "output.json")?;
let loaded = trajectory_from_csv("output.csv")?;
```

```python
# Python
traj = planner.plan(start, goal)
traj.to_csv("output.csv")
traj.to_json("output.json")
duration = traj.duration
joints_at_t = traj.sample(0.5)
```

## Task Planning

```rust
// Rust
let pick = Task::pick(&robot, &scene, pick_config);
let solution = Task::sequence(vec![Task::move_to(&robot, goal), pick]).plan(&start)?;
```

```python
# Python
t1 = kinetic.Task.move_to(robot, goal)
t2 = kinetic.Task.pick(robot, scene, "cup", grasp_poses, approach, retreat)
t3 = kinetic.Task.place(robot, scene, "cup", target_pose, approach, retreat)
seq = kinetic.Task.sequence([t1, t2, t3])
solution = seq.plan(start_joints)
```

## Dynamics

```rust
// Rust
let mut body = articulated_body_from_chain(&robot, &chain);
let tau = gravity_compensation(&mut body, &q);
let tau = inverse_dynamics(&mut body, &q, &qd, &qdd);
let qdd = forward_dynamics(&mut body, &q, &qd, &tau);
let m = mass_matrix(&mut body, &q);
```

```python
# Python
dyn = kinetic.Dynamics(robot)
tau = dyn.gravity_compensation(q)
tau = dyn.inverse_dynamics(q, qd, qdd)
qdd = dyn.forward_dynamics(q, qd, tau)
M = dyn.mass_matrix(q)   # (DOF, DOF) numpy
```

## GPU Optimization

```rust
// Rust
let opt = GpuOptimizer::new(GpuConfig::balanced())?;
let traj = opt.optimize(&robot, &obstacles, &start, &goal)?;
```

```python
# Python
opt = kinetic.GpuOptimizer(robot, preset="balanced")  # auto GPU/CPU
traj = opt.optimize(start, goal, scene=scene)
print(opt.is_gpu)  # True if GPU available

checker = kinetic.GpuCollisionChecker(robot, scene)
results = checker.check_batch(configs)  # (N, DOF) → {'in_collision', 'min_distances'}
```

## Execution

```rust
// Rust
let executor = RealTimeExecutor::new(ExecutionConfig::safe(&robot));
let result = executor.execute(&timed, &mut my_driver)?;
```

```python
# Python
def send(positions, velocities):
    my_driver.set_joints(positions)

executor = kinetic.RealTimeExecutor.safe(robot, rate_hz=500)
result = executor.execute(traj, send, feedback=read_positions)
# result: {'state', 'actual_duration', 'commands_sent', 'max_deviation', ...}
```
