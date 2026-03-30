# Custom Robots

Loading your own URDF and creating kinetic configurations.

## Overview

Kinetic ships with 52 built-in robot configurations, but you can add any
robot that has a URDF file. This guide walks through the process step by step.

## Step 1: Get a URDF

You need a valid URDF file describing your robot's kinematics. Common sources:

- Robot manufacturer (most provide URDF files)
- ROS robot description packages (e.g., `ur_description`, `franka_description`)
- Export from CAD software (SolidWorks, Fusion 360)
- Create manually for simple robots

The URDF must include joints with `type`, `axis`, `origin`, and `limit` tags.
Collision geometry is optional but improves planning quality.

## Step 2: Create a Configuration Directory

Create a directory under `robot_configs/` named after your robot:

```
robot_configs/
  my_robot/
    kinetic.toml    # Configuration file
    my_robot.urdf   # Robot description
```

If you are working outside the kinetic source tree, you can place the
directory anywhere and load it with `Robot::from_path()`.

## Step 3: Write kinetic.toml

The configuration file defines planning groups, IK solver preferences,
named poses, and collision settings.

```toml
[robot]
name = "my_robot"
urdf = "my_robot.urdf"
dof = 6

# Planning group: defines the kinematic chain
[planning_group.arm]
chain = ["base_link", "tool0"]

# End effector definition (optional)
[end_effector.tool]
parent_link = "tool0"
parent_group = "arm"
tcp_xyz = [0.0, 0.0, 0.05]      # Tool center point offset
tcp_rpy = [0.0, 0.0, 0.0]       # Tool orientation offset (optional)

# IK solver preference
[ik]
solver = "opw"      # "opw" for 6-DOF spherical wrist, "dls" for general
damping = 0.05      # Only used with "dls" solver

# Named joint configurations
[named_poses]
home = [0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0]
zero = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

# Collision settings
[collision]
self_collision_pairs = "auto"     # Auto-detect from URDF geometry
padding = 0.01                    # Extra padding in meters
skip_pairs = [                    # Disable collision between known-safe pairs
    ["link1", "link3"],
]
```

### Section Reference

**`[robot]`** -- Required. Robot identity.

| Field | Type | Description |
|-------|------|-------------|
| `name` | string | Robot name used in `Robot::from_name()` |
| `urdf` | string | Path to URDF file (relative to config directory) |
| `dof` | integer | Degrees of freedom |

**`[planning_group.<name>]`** -- At least one required. Defines kinematic chains.

| Field | Type | Description |
|-------|------|-------------|
| `chain` | [string, string] | [base_link, tip_link] |

**`[ik]`** -- Optional. IK solver configuration.

| Field | Type | Description |
|-------|------|-------------|
| `solver` | string | `"opw"`, `"dls"`, `"fabrik"`, `"subproblem"`, `"subproblem7dof"` |
| `damping` | float | DLS damping factor (default: 0.05) |

**`[named_poses]`** -- Optional. Named joint configurations.

Each key is a pose name, value is an array of joint values matching the DOF.

**`[collision]`** -- Optional. Collision checking settings.

| Field | Type | Description |
|-------|------|-------------|
| `self_collision_pairs` | string | `"auto"` or `"none"` |
| `padding` | float | Extra clearance in meters |
| `skip_pairs` | [[string, string]] | Link pairs to ignore |

## Step 4: Load and Verify

```rust
use kinetic::prelude::*;

// Load from built-in configs (after adding to robot_configs/)
let robot = Robot::from_name("my_robot")?;

// Or load from a path
let robot = Robot::from_path("path/to/my_robot/")?;

// Or load directly from URDF string (no config file)
let robot = Robot::from_urdf_string(urdf_string)?;

// Verify basic info
println!("DOF: {}", robot.dof);
println!("Links: {}", robot.links.len());
println!("Joints: {}", robot.joints.len());
```

## Step 5: Verify FK/IK

```rust
let chain = KinematicChain::extract(&robot, "base_link", "tool0")?;

// Forward kinematics
let home = vec![0.0, -1.5708, 0.0, -1.5708, 0.0, 0.0];
let pose = forward_kinematics(&robot, &chain, &home)?;
println!("EE position: {:?}", pose.translation());

// Inverse kinematics
let config = IKConfig::default().with_restarts(8);
let solution = solve_ik(&robot, &chain, &pose, &config)?;
assert!(solution.converged);
println!("IK residual: {:.6}m", solution.position_error);
```

## Step 6: Test Planning

```rust
let planner = Planner::new(&robot)?;

let start = vec![0.0; 6];
let goal = Goal::Named("home".into());
let result = planner.plan(&start, &goal)?;

println!("Path: {} waypoints in {:?}",
    result.num_waypoints(), result.planning_time);
```

## Choosing the Right IK Solver

| Robot Type | DOF | Recommended Solver |
|-----------|-----|--------------------|
| Spherical wrist (UR, ABB IRB, KUKA KR) | 6 | `opw` |
| Intersecting wrist axes | 6 | `subproblem` |
| Redundant arms (Panda, iiwa, xArm7) | 7 | `subproblem7dof` or `dls` |
| Mobile manipulators (Fetch, TIAGo) | 7-8 | `dls` |
| Low-DOF (4-5 joints) | 4-5 | `dls` |
| Any robot (fallback) | Any | `dls` |

## Tips

- Start with `dls` if unsure. It works for all robots.
- Set `skip_pairs` for links that are physically close but cannot collide
  (reduces false positives in collision checking).
- Use FK to verify your URDF before planning. Wrong link lengths or
  joint axes cause silent errors.
- Named poses must be within joint limits. The planner validates them.
