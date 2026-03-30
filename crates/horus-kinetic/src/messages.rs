//! Message types for KINETIC HORUS bridge.
//!
//! These messages are designed to be compatible with HORUS shared-memory
//! topics (Clone + Send + Sync + Serialize + Deserialize).
//!
//! # Topic Naming Convention
//!
//! | Topic                          | Message Type     | Direction   |
//! |--------------------------------|------------------|-------------|
//! | `/kinetic/plan_request`        | PlanRequest      | Subscribe   |
//! | `/kinetic/trajectory`          | TrajectoryMsg    | Publish     |
//! | `/kinetic/servo/twist_cmd`     | TwistMsg         | Subscribe   |
//! | `/kinetic/servo/joint_jog`     | JointJogMsg      | Subscribe   |
//! | `/kinetic/servo/joint_cmd`     | JointCommandMsg  | Publish     |
//! | `/kinetic/servo/state`         | ServoStateMsg    | Publish     |
//! | `/kinetic/scene`               | SceneMsg         | Publish     |
//! | `/kinetic/scene/update`        | SceneUpdateMsg   | Subscribe   |
//! | `/joint_states`                | JointStateMsg    | Subscribe   |

use serde::{Deserialize, Serialize};

/// Topic names used by KINETIC nodes.
///
/// Shared topics use canonical names from softmata-core.
/// Kinetic-specific topics use `kinetic_` prefix (no slashes — HORUS convention).
pub mod topics {
    /// Planning request (kinetic-specific).
    pub const PLAN_REQUEST: &str = softmata_core::conventions::topics::PLAN_REQUEST;
    /// Planned trajectory output (canonical).
    pub const TRAJECTORY: &str = softmata_core::conventions::topics::TRAJECTORY;
    /// Servo twist command (canonical).
    pub const SERVO_TWIST: &str = softmata_core::conventions::topics::TWIST_CMD;
    /// Servo joint jog command (kinetic-specific).
    pub const SERVO_JOINT_JOG: &str = "kinetic_joint_jog";
    /// Servo joint command output (canonical).
    pub const SERVO_JOINT_CMD: &str = softmata_core::conventions::topics::JOINT_CMD;
    /// Servo state feedback (kinetic-specific).
    pub const SERVO_STATE: &str = "kinetic_servo_state";
    /// Scene state (kinetic-specific).
    pub const SCENE: &str = "kinetic_scene";
    /// Scene update commands (kinetic-specific).
    pub const SCENE_UPDATE: &str = "kinetic_scene_update";
    /// Joint states input (canonical).
    pub const JOINT_STATES: &str = softmata_core::conventions::topics::JOINT_STATES;
    /// Point cloud for scene obstacles (canonical).
    pub const POINTCLOUD: &str = softmata_core::conventions::topics::POINTS;
}

// ─── Planning Messages ──────────────────────────────────────────────────────

/// Goal specification for planning requests.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GoalMsg {
    /// Target joint configuration.
    Joints(Vec<f64>),
    /// Target end-effector pose (translation + quaternion [x,y,z,w]).
    Pose {
        position: [f64; 3],
        orientation: [f64; 4],
    },
    /// Named pose from robot configuration.
    Named(String),
}

/// Planning request message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanRequest {
    /// Current joint positions.
    pub start_joints: Vec<f64>,
    /// Goal specification.
    pub goal: GoalMsg,
    /// Request ID for correlating responses.
    pub request_id: u64,
}

/// Waypoint in a trajectory message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WaypointMsg {
    /// Joint positions.
    pub positions: Vec<f64>,
    /// Joint velocities.
    pub velocities: Vec<f64>,
    /// Time from trajectory start in seconds.
    pub time_secs: f64,
}

/// Trajectory response message.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TrajectoryMsg {
    /// Ordered list of waypoints.
    pub waypoints: Vec<WaypointMsg>,
    /// Total trajectory duration in seconds.
    pub duration_secs: f64,
    /// Planning computation time in microseconds.
    pub planning_time_us: u64,
    /// Correlated request ID.
    pub request_id: u64,
    /// Whether planning succeeded.
    pub success: bool,
    /// Error message if planning failed.
    pub error: Option<String>,
}

// ─── Servo Messages ─────────────────────────────────────────────────────────

/// Twist command for Cartesian servo.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TwistMsg {
    /// Linear velocity [vx, vy, vz] in m/s.
    pub linear: [f64; 3],
    /// Angular velocity [wx, wy, wz] in rad/s.
    pub angular: [f64; 3],
}

/// Joint jog command for joint-space servo.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointJogMsg {
    /// Joint velocities in rad/s.
    pub velocities: Vec<f64>,
}

/// Joint command output from servo.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointCommandMsg {
    /// Commanded joint positions.
    pub positions: Vec<f64>,
    /// Commanded joint velocities.
    pub velocities: Vec<f64>,
    /// Timestamp in nanoseconds since epoch.
    pub timestamp_ns: u64,
}

/// Servo state feedback.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServoStateMsg {
    /// Current end-effector position [x, y, z].
    pub ee_position: [f64; 3],
    /// Current end-effector orientation quaternion [x, y, z, w].
    pub ee_orientation: [f64; 4],
    /// Minimum distance to nearest obstacle in meters.
    pub min_obstacle_distance: f64,
    /// Manipulability measure.
    pub manipulability: f64,
    /// Whether near a singularity.
    pub near_singularity: bool,
    /// Whether collision avoidance is active.
    pub collision_avoidance_active: bool,
}

// ─── Joint State Messages ───────────────────────────────────────────────────

/// Joint state feedback (from robot driver).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JointStateMsg {
    /// Joint names.
    pub names: Vec<String>,
    /// Joint positions in radians.
    pub positions: Vec<f64>,
    /// Joint velocities in rad/s.
    pub velocities: Vec<f64>,
    /// Timestamp in nanoseconds since epoch.
    pub timestamp_ns: u64,
}

// ─── Scene Messages ─────────────────────────────────────────────────────────

/// Scene object description.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneObjectMsg {
    /// Object name.
    pub name: String,
    /// Shape description.
    pub shape: ShapeMsg,
    /// Pose: position [x, y, z].
    pub position: [f64; 3],
    /// Pose: orientation quaternion [x, y, z, w].
    pub orientation: [f64; 4],
}

/// Shape description for scene objects.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ShapeMsg {
    /// Box with half-extents [hx, hy, hz].
    Box([f64; 3]),
    /// Cylinder with radius and half-height.
    Cylinder { radius: f64, half_height: f64 },
    /// Sphere with radius.
    Sphere(f64),
}

/// Scene state message (full scene snapshot).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SceneMsg {
    /// All objects in the scene.
    pub objects: Vec<SceneObjectMsg>,
    /// Attached objects (on robot links).
    pub attached: Vec<AttachedObjectMsg>,
}

/// Attached object description.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AttachedObjectMsg {
    /// Object name.
    pub name: String,
    /// Shape description.
    pub shape: ShapeMsg,
    /// Parent link name.
    pub parent_link: String,
}

/// Scene update command.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SceneUpdateMsg {
    /// Add an object to the scene.
    Add(SceneObjectMsg),
    /// Remove an object by name.
    Remove(String),
    /// Attach an object to a robot link.
    Attach {
        object: String,
        link: String,
    },
    /// Detach an object from the robot.
    Detach {
        object: String,
        place_position: [f64; 3],
        place_orientation: [f64; 4],
    },
    /// Clear all objects.
    Clear,
}

// ─── Point Cloud Messages ───────────────────────────────────────────────────

/// Point cloud update message for octree-based collision detection.
///
/// Published on `/kinetic/scene/pointcloud` to update the scene's
/// volumetric occupancy map. The octree uses ray-casting from the
/// sensor origin to clear free space and mark occupied voxels.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PointCloudMsg {
    /// 3D points as flat [x, y, z] triples.
    pub points: Vec<[f64; 3]>,
    /// Sensor origin for ray-casting (used to clear free space along rays).
    pub sensor_origin: [f64; 3],
    /// Optional name for the octree (default: "default").
    /// Multiple named octrees can coexist (e.g., "lidar", "depth_camera").
    pub source_name: Option<String>,
}

// ─── Conversions ─────────────────────────────────────────────────────────────

impl GoalMsg {
    /// Convert to KINETIC Goal type.
    pub fn to_kinetic_goal(&self) -> kinetic_core::Goal {
        match self {
            GoalMsg::Joints(joints) => {
                kinetic_core::Goal::Joints(kinetic_core::JointValues(joints.clone()))
            }
            GoalMsg::Pose {
                position,
                orientation,
            } => {
                let iso = nalgebra::Isometry3::from_parts(
                    nalgebra::Translation3::new(position[0], position[1], position[2]),
                    nalgebra::UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                        orientation[3],
                        orientation[0],
                        orientation[1],
                        orientation[2],
                    )),
                );
                kinetic_core::Goal::Pose(kinetic_core::Pose(iso))
            }
            GoalMsg::Named(name) => kinetic_core::Goal::Named(name.clone()),
        }
    }
}

impl ShapeMsg {
    /// Convert to KINETIC Shape type.
    pub fn to_kinetic_shape(&self) -> kinetic_scene::Shape {
        match self {
            ShapeMsg::Box(half_extents) => {
                kinetic_scene::Shape::Cuboid(half_extents[0], half_extents[1], half_extents[2])
            }
            ShapeMsg::Cylinder {
                radius,
                half_height,
            } => kinetic_scene::Shape::Cylinder(*radius, *half_height),
            ShapeMsg::Sphere(radius) => kinetic_scene::Shape::Sphere(*radius),
        }
    }

    /// Convert from KINETIC Shape type.
    pub fn from_kinetic_shape(shape: &kinetic_scene::Shape) -> Option<Self> {
        match shape {
            kinetic_scene::Shape::Cuboid(hx, hy, hz) => Some(ShapeMsg::Box([*hx, *hy, *hz])),
            kinetic_scene::Shape::Cylinder(r, hh) => Some(ShapeMsg::Cylinder {
                radius: *r,
                half_height: *hh,
            }),
            kinetic_scene::Shape::Sphere(r) => Some(ShapeMsg::Sphere(*r)),
            kinetic_scene::Shape::HalfSpace(_, _) => None, // not serializable as scene object
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn goal_msg_joints_roundtrip() {
        let goal = GoalMsg::Joints(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6]);
        let kinetic_goal = goal.to_kinetic_goal();
        match kinetic_goal {
            kinetic_core::Goal::Joints(jv) => {
                assert_eq!(jv.0.len(), 6);
                assert!((jv.0[0] - 0.1).abs() < 1e-10);
            }
            _ => panic!("Expected Joints goal"),
        }
    }

    #[test]
    fn goal_msg_pose_roundtrip() {
        let goal = GoalMsg::Pose {
            position: [1.0, 2.0, 3.0],
            orientation: [0.0, 0.0, 0.0, 1.0], // identity quaternion
        };
        let kinetic_goal = goal.to_kinetic_goal();
        match kinetic_goal {
            kinetic_core::Goal::Pose(pose) => {
                let t = pose.translation();
                assert!((t.x - 1.0).abs() < 1e-10);
                assert!((t.y - 2.0).abs() < 1e-10);
                assert!((t.z - 3.0).abs() < 1e-10);
            }
            _ => panic!("Expected Pose goal"),
        }
    }

    #[test]
    fn shape_msg_roundtrip() {
        let shapes = vec![
            ShapeMsg::Box([0.1, 0.2, 0.3]),
            ShapeMsg::Cylinder {
                radius: 0.05,
                half_height: 0.1,
            },
            ShapeMsg::Sphere(0.04),
        ];

        for shape in &shapes {
            let kinetic = shape.to_kinetic_shape();
            let back = ShapeMsg::from_kinetic_shape(&kinetic).unwrap();
            let kinetic2 = back.to_kinetic_shape();

            // Verify round-trip by comparing kinetic shapes
            match (&kinetic, &kinetic2) {
                (
                    kinetic_scene::Shape::Cuboid(a, b, c),
                    kinetic_scene::Shape::Cuboid(d, e, f),
                ) => {
                    assert!((a - d).abs() < 1e-10);
                    assert!((b - e).abs() < 1e-10);
                    assert!((c - f).abs() < 1e-10);
                }
                (
                    kinetic_scene::Shape::Cylinder(r1, h1),
                    kinetic_scene::Shape::Cylinder(r2, h2),
                ) => {
                    assert!((r1 - r2).abs() < 1e-10);
                    assert!((h1 - h2).abs() < 1e-10);
                }
                (kinetic_scene::Shape::Sphere(r1), kinetic_scene::Shape::Sphere(r2)) => {
                    assert!((r1 - r2).abs() < 1e-10);
                }
                _ => panic!("Shape type mismatch in round-trip"),
            }
        }
    }

    #[test]
    fn plan_request_serializable() {
        let req = PlanRequest {
            start_joints: vec![0.0; 6],
            goal: GoalMsg::Named("home".into()),
            request_id: 42,
        };
        let json = serde_json::to_string(&req).unwrap();
        let back: PlanRequest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.request_id, 42);
    }

    #[test]
    fn trajectory_msg_serializable() {
        let traj = TrajectoryMsg {
            waypoints: vec![WaypointMsg {
                positions: vec![0.1, 0.2, 0.3],
                velocities: vec![0.0, 0.0, 0.0],
                time_secs: 0.0,
            }],
            duration_secs: 1.5,
            planning_time_us: 500,
            request_id: 1,
            success: true,
            error: None,
        };
        let json = serde_json::to_string(&traj).unwrap();
        let back: TrajectoryMsg = serde_json::from_str(&json).unwrap();
        assert_eq!(back.waypoints.len(), 1);
        assert!(back.success);
    }

    #[test]
    fn scene_update_variants() {
        let updates = vec![
            SceneUpdateMsg::Add(SceneObjectMsg {
                name: "box1".into(),
                shape: ShapeMsg::Box([0.1, 0.1, 0.1]),
                position: [0.5, 0.0, 0.3],
                orientation: [0.0, 0.0, 0.0, 1.0],
            }),
            SceneUpdateMsg::Remove("box1".into()),
            SceneUpdateMsg::Attach {
                object: "cup".into(),
                link: "tool0".into(),
            },
            SceneUpdateMsg::Detach {
                object: "cup".into(),
                place_position: [0.5, 0.0, 0.3],
                place_orientation: [0.0, 0.0, 0.0, 1.0],
            },
            SceneUpdateMsg::Clear,
        ];

        for update in &updates {
            let json = serde_json::to_string(update).unwrap();
            let _back: SceneUpdateMsg = serde_json::from_str(&json).unwrap();
        }
    }

    #[test]
    fn pointcloud_msg_serializable() {
        let msg = PointCloudMsg {
            points: vec![[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]],
            sensor_origin: [0.0, 0.0, 1.0],
            source_name: Some("lidar".into()),
        };
        let json = serde_json::to_string(&msg).unwrap();
        let back: PointCloudMsg = serde_json::from_str(&json).unwrap();
        assert_eq!(back.points.len(), 2);
        assert!((back.sensor_origin[2] - 1.0).abs() < 1e-10);
        assert_eq!(back.source_name.as_deref(), Some("lidar"));
    }

    #[test]
    fn pointcloud_msg_default_source() {
        let msg = PointCloudMsg {
            points: vec![[1.0, 2.0, 3.0]],
            sensor_origin: [0.0, 0.0, 0.0],
            source_name: None,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let back: PointCloudMsg = serde_json::from_str(&json).unwrap();
        assert!(back.source_name.is_none());
    }

    #[test]
    fn topic_names_defined() {
        assert_eq!(topics::PLAN_REQUEST, "plan_request");
        assert_eq!(topics::TRAJECTORY, "trajectory");
        assert_eq!(topics::SERVO_TWIST, "twist_cmd");
        assert_eq!(topics::SERVO_JOINT_JOG, "kinetic_joint_jog");
        assert_eq!(topics::SERVO_JOINT_CMD, "joint_cmd");
        assert_eq!(topics::SERVO_STATE, "kinetic_servo_state");
        assert_eq!(topics::SCENE, "kinetic_scene");
        assert_eq!(topics::SCENE_UPDATE, "kinetic_scene_update");
        assert_eq!(topics::JOINT_STATES, "joint_states");
        assert_eq!(topics::POINTCLOUD, "points");
    }

    #[test]
    fn joint_state_msg_serializable() {
        let msg = JointStateMsg {
            names: vec!["joint1".into(), "joint2".into()],
            positions: vec![0.1, 0.2],
            velocities: vec![0.0, 0.0],
            timestamp_ns: 1_000_000_000,
        };
        let json = serde_json::to_string(&msg).unwrap();
        let back: JointStateMsg = serde_json::from_str(&json).unwrap();
        assert_eq!(back.names.len(), 2);
        assert_eq!(back.positions.len(), 2);
        assert_eq!(back.timestamp_ns, 1_000_000_000);
    }
}
