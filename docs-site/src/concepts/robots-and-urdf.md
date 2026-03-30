# Robots and URDF

A robot in kinetic is a kinematic tree of links connected by joints. This page explains how kinetic represents that tree, how it loads robot models from standard file formats, and what additional metadata (planning groups, end-effectors, collision pairs) enriches the model for motion planning.

## The Robot struct

The `Robot` struct is the central data structure in kinetic. Every FK, IK, and planning operation takes a `&Robot`. It contains:

- **Joints** — an ordered list of all joints, including fixed ones. Each joint has a type, an axis, an origin transform, and optional limits.
- **Links** — an ordered list of all links (rigid bodies). Each link has a name and optional geometry.
- **Active joints** — indices of non-fixed joints. The number of active joints is the robot's DOF.
- **Planning groups** — named subsets of joints for planning (e.g., "arm" vs. "gripper").
- **End-effectors** — tool-tip definitions with parent link and grasp frame offset.
- **Named poses** — pre-defined joint configurations like "home" or "tucked".
- **Joint limits** — position, velocity, effort, and acceleration bounds.

```rust
use kinetic::prelude::*;

let robot = Robot::from_name("ur5e")?;
println!("Name: {}", robot.name);        // "ur5e"
println!("DOF: {}", robot.dof);          // 6
println!("Joints: {}", robot.joints.len()); // includes fixed joints
println!("Active: {}", robot.active_joints.len()); // 6
```

## Loading robots from URDF

URDF (Unified Robot Description Format) is an XML format that describes the kinematic and visual properties of a robot. It defines links, joints, origins, axes, and limits. Kinetic parses URDF files into a `Robot`.

```rust
// From a file path
let robot = Robot::from_urdf("path/to/my_robot.urdf")?;

// From an XML string (useful for testing and embedded models)
let robot = Robot::from_urdf_string(r#"
  <robot name="simple">
    <link name="base_link"/>
    <link name="tool"/>
    <joint name="j1" type="revolute">
      <parent link="base_link"/>
      <child link="tool"/>
      <origin xyz="0 0 0.5"/>
      <axis xyz="0 0 1"/>
      <limit lower="-3.14" upper="3.14" velocity="2.0" effort="100"/>
    </joint>
  </robot>
"#)?;
```

A URDF file encodes the kinematic tree using `<link>` and `<joint>` elements. Each `<joint>` connects a parent link to a child link, has a type (revolute, prismatic, continuous, or fixed), an origin transform (where the joint frame sits relative to the parent), an axis of motion, and limits.

## Joint types

Kinetic supports four joint types, matching the URDF spec:

| Type | Motion | DOF | Limits | Typical use |
|------|--------|-----|--------|-------------|
| **Revolute** | Rotates around axis | 1 | Bounded | Most arm joints |
| **Prismatic** | Translates along axis | 1 | Bounded | Linear stages, lifts |
| **Continuous** | Rotates without bounds | 1 | None | Wheels, spindles |
| **Fixed** | No motion | 0 | N/A | Sensor mounts, flanges |

```rust
use kinetic_robot::JointType;

match joint.joint_type {
    JointType::Revolute   => println!("Rotation with limits"),
    JointType::Prismatic  => println!("Translation with limits"),
    JointType::Continuous => println!("Unlimited rotation"),
    JointType::Fixed      => println!("Rigid connection"),
}
```

## The 52 built-in robots

Kinetic ships with 52 pre-configured robots covering major manufacturers and research platforms. Each has a URDF file and a `kinetic.toml` configuration defining planning groups, named poses, IK solver preferences, and collision settings.

Load any built-in robot by name:

```rust
let panda  = Robot::from_name("franka_panda")?;  // 7-DOF research arm
let ur5e   = Robot::from_name("ur5e")?;           // 6-DOF industrial
let xarm7  = Robot::from_name("xarm7")?;          // 7-DOF collaborative
let fetch  = Robot::from_name("fetch")?;           // 8-DOF mobile manipulator
```

Coverage includes Universal Robots (UR3e through UR30), KUKA (iiwa7, iiwa14, KR6), ABB (IRB1200, IRB4600, YuMi), Fanuc (CRX-10iA, LR Mate 200iD), xArm (5/6/7), Franka Panda, Kinova Gen3, Sawyer, Baxter, and many more. See the `robot_configs/` directory for the full list.

## Planning groups and end-effectors

A planning group isolates a kinematic sub-chain for planning. A dual-arm robot might have "left_arm" and "right_arm" groups. A mobile manipulator might have "arm" (the joints that matter for manipulation) and "base" (the mobile platform).

```rust
let robot = Robot::from_name("franka_panda")?;

// Access a planning group
let arm = &robot.groups["arm"];
println!("Base: {}", arm.base_link);  // "panda_link0"
println!("Tip: {}", arm.tip_link);    // "panda_link8"
println!("Joints: {:?}", arm.joint_indices);

// Access end-effectors
if let Some(ee) = robot.end_effectors.get("hand") {
    println!("EE parent: {}", ee.parent_link);
}
```

End-effectors define where the tool is. The `EndEffector` struct stores the parent link, the parent planning group, and an optional grasp-frame offset (`Pose`) from the parent link to the tool center point (TCP).

## Named poses

Named poses are pre-defined joint configurations. Common examples: "home" (a safe starting posture), "tucked" (folded compactly), "ready" (poised for manipulation).

```rust
let robot = Robot::from_name("ur5e")?;

// Look up a named pose
if let Some(home) = robot.named_pose("home") {
    println!("Home joints: {:?}", home.as_slice());
    robot.check_limits(&home)?; // always valid
}

// Built-in utility configurations
let zero = robot.zero_configuration();   // all joints at 0
let mid  = robot.mid_configuration();    // center of joint ranges
```

## SRDF: semantic annotation

SRDF (Semantic Robot Description Format) is an XML companion to URDF that adds planning-level metadata. It originates from MoveIt and is widely used in the ROS ecosystem. Kinetic parses SRDF to set up:

- **Planning groups** — chains of joints or explicit joint lists
- **Disabled collision pairs** — link pairs to skip during self-collision checking
- **End-effectors** — tool-tip definitions
- **Named group states** — pre-defined joint configurations

```rust
// Load URDF + SRDF together
let robot = Robot::from_urdf_srdf(
    "robot.urdf",
    "robot.srdf",
)?;

// Or apply SRDF to an existing robot
use kinetic_robot::srdf::SrdfModel;
let mut robot = Robot::from_urdf("robot.urdf")?;
let srdf = SrdfModel::from_file("robot.srdf")?;
srdf.apply_to_robot(&mut robot)?;
```

A typical SRDF `<disable_collisions>` section lists pairs of links that are always in contact (adjacent links) or geometrically cannot collide (distant links). This builds the Allowed Collision Matrix (ACM) used by collision checkers.

## MJCF and SDF support

Beyond URDF, kinetic also loads models from two other common formats:

- **MJCF** (MuJoCo XML) — the native format for MuJoCo physics simulations. Common in RL research and sim-to-real workflows.
- **SDF** (SDFormat) — the model format used by Gazebo. Supports multiple models, world descriptions, and sensor definitions.

```rust
// Load from MJCF
let robot = Robot::from_mjcf("model.mjcf")?;

// Load from SDF
let robot = Robot::from_sdf("model.sdf")?;

// Auto-detect format from extension
let robot = Robot::from_file("model.urdf")?;  // detects .urdf
let robot = Robot::from_file("model.mjcf")?;  // detects .mjcf
let robot = Robot::from_file("model.sdf")?;   // detects .sdf
```

`Robot::from_file()` inspects the file extension and dispatches to the appropriate loader. For `.xml` files, it tries MJCF first (more common in robotics), falling back to SDF.

## Configuration files

Each built-in robot has a `kinetic.toml` alongside its URDF. This TOML file defines the planning-level configuration that URDF cannot express: planning groups, end-effectors, named poses, IK solver preference (e.g., OPW for 6-DOF spherical wrist robots), and collision settings. When you call `Robot::from_name("ur5e")`, kinetic finds the `robot_configs/ur5e/` directory and loads both the URDF and the TOML config.

If you are bringing your own robot, you can either load just the URDF (the robot will have no planning groups or named poses) or add a TOML config / SRDF file to get the full feature set.

## See Also

- [Glossary](./glossary.md) — definitions of DOF, joint types, planning group, EE, and SRDF
- [Coordinate Frames](./coordinate-frames.md) — how joint origins and link frames form a transform tree
- [Forward Kinematics](./forward-kinematics.md) — computing link poses from joint values
- [Inverse Kinematics](./inverse-kinematics.md) — how IK solver selection depends on robot geometry
- [Motion Planning](./motion-planning.md) — how planning groups define what gets planned
