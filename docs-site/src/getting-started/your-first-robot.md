# Your First Robot

Before planning motions, let's understand how kinetic represents a robot.

## Loading a Robot

Kinetic ships with 54 pre-configured robots. Load one by name:

**Rust:**
```rust
use kinetic::prelude::*;

let robot = Robot::from_name("ur5e")?;
println!("Robot: {} ({} DOF)", robot.name, robot.dof);
println!("Joints: {:?}", robot.joints.iter().map(|j| &j.name).collect::<Vec<_>>());
```

**Python:**
```python
import kinetic

robot = kinetic.Robot("ur5e")
print(f"Robot: {robot.name} ({robot.dof} DOF)")
print(f"Joints: {robot.joint_names}")
```

**Output:**
```
Robot: ur5e (6 DOF)
Joints: ["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
         "wrist_1_joint", "wrist_2_joint", "wrist_3_joint"]
```

## Forward Kinematics

Forward kinematics (FK) answers: *"If the joints are at these angles, where is the end-effector?"*

**Rust:**
```rust
let joints = [0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0]; // UR5e "home" position
let pose = robot.fk(&joints)?;

let t = pose.translation();
println!("End-effector position: x={:.3}, y={:.3}, z={:.3}", t.x, t.y, t.z);

let (roll, pitch, yaw) = pose.rpy();
println!("Orientation (RPY): roll={:.2}°, pitch={:.2}°, yaw={:.2}°",
         roll.to_degrees(), pitch.to_degrees(), yaw.to_degrees());
```

**Python:**
```python
import numpy as np

joints = np.array([0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0])
pose_4x4 = robot.fk(joints)

print(f"End-effector position: {pose_4x4[:3, 3]}")
print(f"Rotation matrix:\n{pose_4x4[:3, :3]}")
```

The result is a **pose** — a position (x, y, z in meters) plus an orientation (rotation matrix or quaternion). This tells you exactly where the robot's tool tip is in 3D space.

## What's Inside a Robot?

A kinetic `Robot` contains:

| Component | What it is | Example |
|-----------|-----------|---------|
| **Joints** | Rotary or linear actuators | `shoulder_pan_joint` (revolute, ±360°) |
| **Links** | Rigid bodies between joints | `upper_arm_link` (0.425m long) |
| **Joint limits** | Min/max angle, velocity, effort | `[-2π, 2π]` rad, 3.14 rad/s |
| **Planning groups** | Named subsets of joints | `"arm"`: all 6 joints |
| **Named poses** | Pre-defined configurations | `"home"`: `[0, -π/2, 0, -π/2, 0, 0]` |
| **Collision geometry** | Simplified shapes for collision | Spheres approximating each link |

All of this comes from the robot's **URDF** (Unified Robot Description Format) file — an XML file that describes the robot's geometry, joints, and limits. Kinetic parses URDF automatically when you call `from_name()`.

## Named Poses

Most robots come with pre-defined poses:

**Rust:**
```rust
if let Some(home) = robot.named_pose("home") {
    let pose = robot.fk(&home)?;
    println!("Home EE position: {:?}", pose.translation());
}
```

**Python:**
```python
home = robot.named_pose("home")
if home is not None:
    pose = robot.fk(home)
    print(f"Home EE position: {pose[:3, 3]}")
```

## Joint Limits

Every joint has physical limits. Kinetic enforces them:

**Rust:**
```rust
for (i, limit) in robot.joint_limits.iter().enumerate() {
    println!("Joint {}: [{:.2}, {:.2}] rad, max vel {:.2} rad/s",
             i, limit.lower, limit.upper, limit.velocity);
}
```

## Loading Custom Robots

If your robot isn't in the built-in list, load it from a URDF file:

**Rust:**
```rust
let robot = Robot::from_urdf("path/to/my_robot.urdf")?;
```

**Python:**
```python
robot = kinetic.Robot.from_urdf("path/to/my_robot.urdf")
```

For MoveIt2 users, kinetic reads SRDF files too:

```rust
let robot = Robot::from_urdf_srdf("my_robot.urdf", "my_robot.srdf")?;
```

See the [Custom Robots](../guides/custom-robots.md) guide for creating a full kinetic configuration.

## Available Robots

See the [Supported Robots](../reference/supported-robots.md) reference for the complete list of 54 built-in robots with manufacturer, DOF, and IK solver.

A quick sample:

| Robot | Manufacturer | DOF | Config name |
|-------|-------------|-----|-------------|
| UR5e | Universal Robots | 6 | `ur5e` |
| Franka Panda | Franka Emika | 7 | `franka_panda` |
| KUKA iiwa7 | KUKA | 7 | `kuka_iiwa7` |
| xArm6 | UFACTORY | 6 | `xarm6` |
| Kinova Gen3 | Kinova | 7 | `kinova_gen3` |

## Try This

1. Load `"franka_panda"` (7 DOF) — note it has one more joint than the UR5e
2. Compute FK at the zero configuration `[0.0; 7]` — where is the Panda's EE?
3. Compare the EE position of two different joint configurations — how far apart are they?
4. Check the joint limits — which joint has the smallest range?

## Next

[Your First Plan →](your-first-plan.md)
