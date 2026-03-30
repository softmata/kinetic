# Coordinate Frames

Every point on a robot exists in some coordinate frame. The base of the robot defines one frame. Each joint defines another. The end-effector tool tip is yet another. Understanding how these frames relate to each other -- and how to transform points between them -- is the foundation of all robotics computation.

## What is a coordinate frame?

A coordinate frame is an origin point plus three orthogonal axes (X, Y, Z). Any position in 3D space can be described as coordinates relative to a frame. A point at (0.5, 0, 0.3) means "0.5 meters along the frame's X axis and 0.3 meters along its Z axis."

But a point described in one frame has different coordinates in another frame. If the robot's base frame is rotated 90 degrees relative to the world frame, a point at (1, 0, 0) in the base frame is at (0, 1, 0) in the world frame. Coordinate transforms convert between frames.

## Position and orientation

A rigid body in 3D space has six degrees of freedom: three for position (x, y, z) and three for orientation (how it is rotated). Together, position and orientation form a **pose**.

**Position** is straightforward: a 3D vector [x, y, z].

**Orientation** has several representations, each with trade-offs:

| Representation | Parameters | Pros | Cons |
|---------------|-----------|------|------|
| Rotation matrix | 3x3 (9 values) | Intuitive, composes by multiplication | Redundant, can drift from orthogonality |
| Euler angles | 3 values (roll, pitch, yaw) | Human-readable | Gimbal lock, order-dependent |
| Quaternion | 4 values (qx, qy, qz, qw) | Compact, no gimbal lock, smooth interpolation | Less intuitive |
| Axis-angle | axis (3) + angle (1) | Geometric meaning | Singular at zero rotation |

Kinetic uses **unit quaternions** (`UnitQuaternion<f64>` from nalgebra) as the primary rotation representation. Euler angles are supported for input/output convenience, but all internal computation uses quaternions.

## Why quaternions over Euler angles

Euler angles describe rotation as three sequential rotations (e.g., first yaw around Z, then pitch around Y, then roll around X). They are intuitive for small rotations but suffer from **gimbal lock**: when two axes align (e.g., pitch = 90 degrees), one degree of freedom is lost and the system cannot represent arbitrary rotations smoothly.

Quaternions avoid this entirely. A unit quaternion (qx, qy, qz, qw) with magnitude 1 uniquely represents any 3D rotation. Composing two rotations is a quaternion multiplication. Interpolating between two orientations (slerp) produces smooth, constant-speed rotation. No singularities, no ambiguity.

```rust
use kinetic_core::{Pose, UnitQuaternion, Vector3};

// Euler angles: human-readable but susceptible to gimbal lock
let pose = Pose::from_xyz_rpy(1.0, 0.0, 0.5, 0.0, 1.5708, 0.0);

// Quaternion: unambiguous and composable
let pose = Pose::from_xyz_quat(1.0, 0.0, 0.5, 0.0, 0.7071, 0.0, 0.7071);

// Extract either representation from a Pose
let (roll, pitch, yaw) = pose.rpy();
let q: &UnitQuaternion<f64> = pose.rotation();
```

## SE(3): the rigid-body transform group

SE(3) (Special Euclidean group in 3D) is the mathematical group of all rigid-body transforms: rotation plus translation. Every pose lives in SE(3). Two key properties make SE(3) essential:

1. **Composition**: if T_AB transforms from frame A to frame B, and T_BC from B to C, then T_AC = T_AB * T_BC. Chaining transforms is just multiplication.
2. **Invertibility**: every transform has an inverse. If T_AB goes from A to B, then T_AB.inverse() goes from B to A.

SE(3) = SO(3) x R^3, where SO(3) is the group of 3D rotations and R^3 is 3D translation.

## Isometry3 and the Pose type

In kinetic, SE(3) transforms are represented by `nalgebra::Isometry3<f64>`. An `Isometry3` stores a `UnitQuaternion<f64>` (rotation) and a `Translation3<f64>` (position). It guarantees that the rotation component stays a valid rotation -- no scale, shear, or drift.

The `Pose` type wraps `Isometry3<f64>` with ergonomic constructors:

```rust
use kinetic_core::Pose;

// Identity: origin, no rotation
let origin = Pose::identity();

// Translation only
let p = Pose::from_xyz(0.5, 0.0, 0.3);

// Position + Euler angles (radians)
let p = Pose::from_xyz_rpy(0.5, 0.0, 0.3, 0.0, 0.0, 1.5708);

// Position + quaternion (qx, qy, qz, qw)
let p = Pose::from_xyz_quat(0.5, 0.0, 0.3, 0.0, 0.0, 0.7071, 0.7071);

// From a 4x4 homogeneous matrix
let p = Pose::from_matrix(&matrix_4x4);

// Compose: A_to_C = A_to_B * B_to_C
let a_to_c = a_to_b.compose(&b_to_c);

// Inverse: B_to_A from A_to_B
let b_to_a = a_to_b.inverse();
```

`Pose` dereferences to `Isometry3<f64>`, so you can call any nalgebra method on it directly.

## Homogeneous transformation matrices

A 4x4 homogeneous matrix combines rotation and translation into a single matrix:

```
T = | R  t |    R = 3x3 rotation matrix
    | 0  1 |    t = 3x1 translation vector
```

Matrix multiplication chains transforms, matching SE(3) composition. Homogeneous matrices are widely used in textbooks, URDF origins, and DH parameters. Kinetic can convert to/from them:

```rust
let mat: nalgebra::Matrix4<f64> = pose.to_matrix();
let recovered = Pose::from_matrix(&mat);
```

Internally, kinetic prefers `Isometry3` because it cannot accumulate numerical drift (rotation is always a valid quaternion), whereas a 4x4 matrix must be periodically re-orthogonalized.

## The FrameTree

A robot system has many coordinate frames: world, base_link, each joint, each link, camera, tool tip. The `FrameTree` manages these relationships. It is conceptually equivalent to ROS2 TF2 but has no ROS dependency.

```rust
use kinetic_core::{FrameTree, Pose};

let tree = FrameTree::new();

// Register transforms between frames
tree.set_transform("world", "base_link", base_pose.isometry().clone(), 0.0);
tree.set_transform("base_link", "camera", camera_cal.isometry().clone(), 0.0);

// Look up any transform -- chains automatically
let cam_in_world = tree.lookup_transform("world", "camera")?;

// Inverse works too
let world_in_cam = tree.lookup_transform("camera", "world")?;
```

Key features of `FrameTree`:

- **Chain resolution**: looking up "world" to "camera" automatically chains through "base_link" using BFS.
- **Automatic inversion**: if only B-to-A is stored, looking up A-to-B returns the inverse.
- **Static transforms**: calibration transforms (sensor mounts) that never change. Marked with `set_static_transform()` and preserved when dynamic transforms are cleared.
- **FK integration**: `update_from_fk()` populates link poses from forward kinematics results, skipping static transforms.
- **Thread safety**: concurrent reads via `RwLock`. Multiple threads can look up transforms simultaneously.

```rust
// Static calibration (never overwritten by FK)
tree.set_static_transform("base_link", "camera", camera_cal);

// Update from FK results
tree.update_from_fk(&link_poses, timestamp);

// Clear dynamic transforms (static survive)
tree.clear_dynamic();

// List all known frames
let frames = tree.list_frames();
```

## Pose distance metrics

Comparing two poses requires measuring both position difference and orientation difference. Kinetic provides both:

```rust
let a = Pose::from_xyz_rpy(1.0, 0.0, 0.0, 0.0, 0.0, 0.0);
let b = Pose::from_xyz_rpy(1.0, 0.5, 0.0, 0.0, 0.0, 1.57);

// Euclidean distance between translations (meters)
let d_pos = a.translation_distance(&b);  // 0.5

// Angular distance between orientations (radians)
let d_rot = a.rotation_distance(&b);  // ~1.57
```

IK solvers use these metrics to define convergence: a solution is accepted when position error is below `position_tolerance` (default 0.1 mm) and orientation error is below `orientation_tolerance` (default 1 mrad).

## Comparison with ROS2 TF2

If you are familiar with ROS2, kinetic's frame management is the equivalent of `tf2`. The key differences:

| Feature | ROS2 TF2 | Kinetic FrameTree |
|---------|----------|-------------------|
| Transport | Published over DDS topics | In-process, no middleware |
| Time queries | Buffer of timestamped transforms | Single latest transform per pair |
| Thread safety | Callback-based | RwLock for concurrent reads |
| Dependencies | Requires ROS2 runtime | Standalone, no external deps |
| Static transforms | Separate `/tf_static` topic | `set_static_transform()` method |

The FrameTree intentionally keeps one transform per frame pair (the latest) rather than buffering a time history. This matches the needs of real-time motion planning, where you always want the current transform, not one from 200 ms ago. If you need time-history queries, publish transforms to a horus topic and maintain your own buffer.

## See Also

- [Glossary](./glossary.md) — definitions of SE(3), Isometry3, quaternion, Pose, and FrameTree
- [Forward Kinematics](./forward-kinematics.md) — how FK computes link poses that populate the FrameTree
- [Robots and URDF](./robots-and-urdf.md) — how URDF joint origins define the static transform tree
- [Inverse Kinematics](./inverse-kinematics.md) — how IK uses Pose as the target specification
