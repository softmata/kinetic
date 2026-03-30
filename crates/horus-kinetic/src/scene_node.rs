//! SceneNode — wraps KINETIC Scene as a message-driven node.

use std::path::Path;
use std::sync::Arc;

use nalgebra::{Isometry3, Quaternion, Translation3, UnitQuaternion};

use kinetic_robot::Robot;
use kinetic_scene::Scene;

use crate::messages::{
    AttachedObjectMsg, PointCloudMsg, SceneMsg, SceneObjectMsg, SceneUpdateMsg, ShapeMsg,
};

/// Error from SceneNode operations.
#[derive(Debug, thiserror::Error)]
pub enum SceneNodeError {
    #[error("Kinetic error: {0}")]
    Kinetic(#[from] kinetic_core::KineticError),
    #[error("Object not found: {0}")]
    ObjectNotFound(String),
}

/// SceneNode manages the collision world for KINETIC planning.
///
/// Receives scene update commands and publishes scene snapshots.
/// The scene is shared with PlannerNode and ServoNode for collision
/// checking during planning and servo control.
pub struct SceneNode {
    robot: Arc<Robot>,
    scene: Scene,
}

impl SceneNode {
    /// Create a new SceneNode for the named robot.
    pub fn new(robot_name: &str) -> Result<Self, SceneNodeError> {
        let robot = Arc::new(Robot::from_name(robot_name)?);
        let scene = Scene::new(&robot)?;
        Ok(Self { robot, scene })
    }

    /// Create a SceneNode from URDF + SRDF files.
    ///
    /// The SRDF's disabled collision pairs are automatically loaded into
    /// the scene's ACM through `Robot::from_urdf_srdf()`.
    pub fn new_with_srdf(
        urdf_path: impl AsRef<Path>,
        srdf_path: impl AsRef<Path>,
    ) -> Result<Self, SceneNodeError> {
        let robot = Arc::new(Robot::from_urdf_srdf(urdf_path, srdf_path)?);
        let scene = Scene::new(&robot)?;
        Ok(Self { robot, scene })
    }

    /// Create a SceneNode with a pre-loaded Robot model.
    pub fn new_with_robot(robot: Arc<Robot>) -> Result<Self, SceneNodeError> {
        let scene = Scene::new(&robot)?;
        Ok(Self { robot, scene })
    }

    /// Get a reference to the underlying robot.
    pub fn robot(&self) -> &Robot {
        &self.robot
    }

    /// Get a reference to the scene.
    pub fn scene(&self) -> &Scene {
        &self.scene
    }

    /// Get a mutable reference to the scene.
    pub fn scene_mut(&mut self) -> &mut Scene {
        &mut self.scene
    }

    /// Handle a scene update command.
    pub fn handle_update(&mut self, update: &SceneUpdateMsg) -> Result<(), SceneNodeError> {
        match update {
            SceneUpdateMsg::Add(obj) => {
                let shape = obj.shape.to_kinetic_shape();
                let pose = pose_from_msg(&obj.position, &obj.orientation);
                self.scene.add(&obj.name, shape, pose);
            }
            SceneUpdateMsg::Remove(name) => {
                self.scene.remove(name);
            }
            SceneUpdateMsg::Attach { object, link } => {
                // Get object shape from scene before attaching
                let obj = self
                    .scene
                    .get_object(object)
                    .ok_or_else(|| SceneNodeError::ObjectNotFound(object.clone()))?;
                let shape = obj.shape.clone();
                let grasp_transform = Isometry3::identity();
                self.scene.attach(object, shape, grasp_transform, link);
            }
            SceneUpdateMsg::Detach {
                object,
                place_position,
                place_orientation,
            } => {
                let pose = pose_from_msg(place_position, place_orientation);
                self.scene.detach(object, pose);
            }
            SceneUpdateMsg::Clear => {
                self.scene.clear();
            }
        }
        Ok(())
    }

    /// Handle a point cloud message by updating the scene's octree.
    ///
    /// Creates a new octree with default config if one doesn't exist for
    /// the given source name. Uses ray-casting from sensor_origin to
    /// clear free space along rays and mark occupied voxels.
    pub fn handle_pointcloud(&mut self, msg: &PointCloudMsg) {
        let name = msg
            .source_name
            .as_deref()
            .unwrap_or("default");
        self.scene.update_octree(name, &msg.points, &msg.sensor_origin);
    }

    /// Get the number of octrees in the scene.
    pub fn num_octrees(&self) -> usize {
        self.scene.num_octrees()
    }

    /// Generate a scene snapshot message.
    pub fn snapshot(&self) -> SceneMsg {
        let objects = self
            .scene
            .objects_iter()
            .map(|obj| {
                let shape = ShapeMsg::from_kinetic_shape(&obj.shape)
                    .unwrap_or(ShapeMsg::Sphere(0.01));
                let t = obj.pose.translation.vector;
                let q = obj.pose.rotation;
                SceneObjectMsg {
                    name: obj.name.clone(),
                    shape,
                    position: [t.x, t.y, t.z],
                    orientation: [q.i, q.j, q.k, q.w],
                }
            })
            .collect();

        let attached = self
            .scene
            .attached_iter()
            .map(|att| {
                let shape = ShapeMsg::from_kinetic_shape(&att.shape)
                    .unwrap_or(ShapeMsg::Sphere(0.01));
                AttachedObjectMsg {
                    name: att.name.clone(),
                    shape,
                    parent_link: att.parent_link.clone(),
                }
            })
            .collect();

        SceneMsg { objects, attached }
    }
}

/// Create an Isometry3 from position and orientation arrays.
fn pose_from_msg(position: &[f64; 3], orientation: &[f64; 4]) -> Isometry3<f64> {
    Isometry3::from_parts(
        Translation3::new(position[0], position[1], position[2]),
        UnitQuaternion::from_quaternion(Quaternion::new(
            orientation[3], // w
            orientation[0], // x
            orientation[1], // y
            orientation[2], // z
        )),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_node_creation() {
        let node = SceneNode::new("ur5e").unwrap();
        assert_eq!(node.robot().name, "ur5e");
        assert_eq!(node.scene().num_objects(), 0);
    }

    #[test]
    fn scene_node_add_remove() {
        let mut node = SceneNode::new("ur5e").unwrap();

        node.handle_update(&SceneUpdateMsg::Add(SceneObjectMsg {
            name: "box1".into(),
            shape: ShapeMsg::Box([0.1, 0.1, 0.1]),
            position: [0.5, 0.0, 0.3],
            orientation: [0.0, 0.0, 0.0, 1.0],
        }))
        .unwrap();

        assert_eq!(node.scene().num_objects(), 1);
        assert!(node.scene().get_object("box1").is_some());

        node.handle_update(&SceneUpdateMsg::Remove("box1".into()))
            .unwrap();
        assert_eq!(node.scene().num_objects(), 0);
    }

    #[test]
    fn scene_node_snapshot() {
        let mut node = SceneNode::new("ur5e").unwrap();

        node.handle_update(&SceneUpdateMsg::Add(SceneObjectMsg {
            name: "cylinder1".into(),
            shape: ShapeMsg::Cylinder {
                radius: 0.03,
                half_height: 0.06,
            },
            position: [0.4, 0.1, 0.2],
            orientation: [0.0, 0.0, 0.0, 1.0],
        }))
        .unwrap();

        let snapshot = node.snapshot();
        assert_eq!(snapshot.objects.len(), 1);
        assert_eq!(snapshot.objects[0].name, "cylinder1");
        assert_eq!(snapshot.attached.len(), 0);
    }

    #[test]
    fn scene_node_with_robot() {
        let robot = Arc::new(Robot::from_name("ur5e").unwrap());
        let node = SceneNode::new_with_robot(robot).unwrap();
        assert_eq!(node.robot().name, "ur5e");
    }

    #[test]
    fn scene_node_handle_pointcloud() {
        let mut node = SceneNode::new("ur5e").unwrap();
        assert_eq!(node.num_octrees(), 0);

        let msg = PointCloudMsg {
            points: vec![
                [0.5, 0.0, 0.3],
                [0.5, 0.1, 0.3],
                [0.5, -0.1, 0.3],
                [0.6, 0.0, 0.4],
            ],
            sensor_origin: [0.0, 0.0, 1.0],
            source_name: None,
        };

        node.handle_pointcloud(&msg);
        assert_eq!(node.num_octrees(), 1);
    }

    #[test]
    fn scene_node_multiple_pointcloud_sources() {
        let mut node = SceneNode::new("ur5e").unwrap();

        let lidar_msg = PointCloudMsg {
            points: vec![[1.0, 0.0, 0.5]],
            sensor_origin: [0.0, 0.0, 1.5],
            source_name: Some("lidar".into()),
        };

        let depth_msg = PointCloudMsg {
            points: vec![[0.5, 0.5, 0.3]],
            sensor_origin: [0.0, 0.0, 0.8],
            source_name: Some("depth_camera".into()),
        };

        node.handle_pointcloud(&lidar_msg);
        node.handle_pointcloud(&depth_msg);
        assert_eq!(node.num_octrees(), 2);

        // Update existing octree
        let lidar_update = PointCloudMsg {
            points: vec![[1.1, 0.1, 0.5], [1.2, 0.0, 0.6]],
            sensor_origin: [0.0, 0.0, 1.5],
            source_name: Some("lidar".into()),
        };
        node.handle_pointcloud(&lidar_update);
        assert_eq!(node.num_octrees(), 2); // still 2, updated in-place
    }

    #[test]
    fn scene_node_with_srdf() {
        let urdf = r#"<?xml version="1.0"?>
<robot name="test_arm">
  <link name="base_link"/>
  <link name="link1"/>
  <link name="link2"/>
  <link name="ee_link"/>
  <joint name="joint1" type="revolute">
    <parent link="base_link"/><child link="link1"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="100"/>
  </joint>
  <joint name="joint2" type="revolute">
    <parent link="link1"/><child link="link2"/>
    <origin xyz="0 0 0.3"/><axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="2.0" effort="80"/>
  </joint>
  <joint name="joint3" type="revolute">
    <parent link="link2"/><child link="ee_link"/>
    <origin xyz="0 0 0.25"/><axis xyz="0 1 0"/>
    <limit lower="-2.5" upper="2.5" velocity="3.0" effort="50"/>
  </joint>
</robot>"#;

        let srdf = r#"<?xml version="1.0" ?>
<robot name="test_arm">
  <group name="arm">
    <chain base_link="base_link" tip_link="ee_link" />
  </group>
  <disable_collisions link1="base_link" link2="link1" reason="Adjacent" />
  <disable_collisions link1="link1" link2="link2" reason="Adjacent" />
</robot>"#;

        let dir = std::env::temp_dir().join("horus_kinetic_scene_srdf_test");
        std::fs::create_dir_all(&dir).unwrap();
        let urdf_path = dir.join("test.urdf");
        let srdf_path = dir.join("test.srdf");
        std::fs::write(&urdf_path, urdf).unwrap();
        std::fs::write(&srdf_path, srdf).unwrap();

        let node = SceneNode::new_with_srdf(&urdf_path, &srdf_path).unwrap();
        assert_eq!(node.robot().name, "test_arm");
        assert!(node.robot().groups.contains_key("arm"));
        assert_eq!(node.scene().num_objects(), 0);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn scene_node_clear() {
        let mut node = SceneNode::new("ur5e").unwrap();

        for i in 0..5 {
            node.handle_update(&SceneUpdateMsg::Add(SceneObjectMsg {
                name: format!("obj{}", i),
                shape: ShapeMsg::Sphere(0.02),
                position: [0.3 + i as f64 * 0.1, 0.0, 0.2],
                orientation: [0.0, 0.0, 0.0, 1.0],
            }))
            .unwrap();
        }
        assert_eq!(node.scene().num_objects(), 5);

        node.handle_update(&SceneUpdateMsg::Clear).unwrap();
        assert_eq!(node.scene().num_objects(), 0);
    }
}
