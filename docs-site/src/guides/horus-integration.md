# HORUS Integration

Using kinetic with the HORUS robotics framework.

## Overview

The `horus-kinetic` crate bridges kinetic's motion planning stack with
HORUS's zero-copy shared-memory IPC. It provides three node types that
can run as HORUS nodes or standalone with channel-based communication.

## Node Architecture

```
                        HORUS IPC (shared memory topics)
                        ================================
                              |           |          |
plan_request -->  [PlannerNode]  --> trajectory
twist_cmd    -->  [ServoNode]    --> joint_cmd
pointcloud   -->  [SceneNode]    --> scene_update
```

### PlannerNode

Receives planning requests, computes collision-free paths, publishes
timed trajectories.

```rust
use horus_kinetic::PlannerNode;

let mut planner = PlannerNode::new("ur5e")?;
// With HORUS IPC: subscribes to "plan_request", publishes to "trajectory"
// Without HORUS: use channel API
```

**Topics:**

| Topic | Direction | Message | Rate |
|-------|-----------|---------|------|
| `plan_request` | Subscribe | `PlanRequest` | On demand |
| `trajectory` | Publish | `TimedTrajectory` | On demand |

### ServoNode

Real-time reactive control. Receives twist or joint jog commands,
outputs joint commands at 500 Hz.

```rust
use horus_kinetic::ServoNode;

let mut servo = ServoNode::new("ur5e")?;
// Subscribes to "twist_cmd" or "joint_jog"
// Publishes "joint_cmd" at 500 Hz
```

**Topics:**

| Topic | Direction | Message | Rate |
|-------|-----------|---------|------|
| `twist_cmd` | Subscribe | `Twist` | Up to 500 Hz |
| `joint_jog` | Subscribe | `JointJog` | Up to 500 Hz |
| `joint_cmd` | Publish | `JointCommand` | 500 Hz |

### SceneNode

Manages the collision environment. Ingests point clouds, depth images,
and explicit obstacle descriptions.

```rust
use horus_kinetic::SceneNode;

let mut scene = SceneNode::new("ur5e")?;
// Subscribes to "pointcloud", "depth_image"
// Publishes "scene_update" on change
```

**Topics:**

| Topic | Direction | Message | Rate |
|-------|-----------|---------|------|
| `pointcloud` | Subscribe | `PointCloud2` | Sensor rate |
| `depth_image` | Subscribe | `DepthImage` | Sensor rate |
| `scene_update` | Publish | `SceneUpdate` | On change |

## With vs Without HORUS

The `horus-kinetic` crate works in two modes:

**With HORUS IPC** (feature `horus-ipc` enabled):
Full HORUS `Node` trait implementations with zero-copy shared-memory
topics. Nodes are discovered and connected by the HORUS runtime.

**Without HORUS** (default):
Channel-based API for standalone use or integration with other transports.

```rust
use horus_kinetic::PlannerNode;

// Standalone mode (no HORUS dependency)
let mut planner = PlannerNode::new("ur5e")?;

// Send a request through the channel API
let request = PlanRequest {
    start: vec![0.0; 6],
    goal: Goal::Named("home".into()),
    ..Default::default()
};
let trajectory = planner.plan(request)?;
```

This design means kinetic works identically in simulation, on standalone
robots, and in full HORUS-networked systems.

## Point Cloud Perception Pipeline

The SceneNode processes raw sensor data into a collision world:

```
Camera/LiDAR --> PointCloud
                    |
              [Downsampling]     -- VoxelGrid or Random
                    |
              [Outlier Removal]  -- Statistical or Radius
                    |
              [Octree Build]     -- Spatial indexing
                    |
              [Sphere Approx]    -- Collision spheres
                    |
              CollisionEnvironment
```

Configure the pipeline:

```rust
use kinetic::scene::{PointCloudConfig, OutlierConfig};

let config = PointCloudConfig {
    voxel_size: 0.01,       // 1cm voxels
    max_points: 50_000,
    outlier: OutlierConfig::Statistical {
        neighbors: 20,
        std_ratio: 2.0,
    },
    ..Default::default()
};

scene.add_pointcloud("camera_0", &points, config);
```

## Building

```bash
# Without HORUS (standalone, channel API)
cargo build -p horus-kinetic

# With HORUS IPC (requires horus to be installed)
cargo build -p horus-kinetic --features horus-ipc
```

The `horus-ipc` feature adds a compile-time dependency on HORUS.
Without it, the crate is fully self-contained and has no HORUS
dependency at all.
