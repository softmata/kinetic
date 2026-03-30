//! URDF loading and conversion to Robot.

use std::collections::HashMap;
use std::path::Path;

use kinetic_core::{KineticError, Pose};
use nalgebra::Vector3;

use crate::joint::{Joint, JointLimits, JointType};
use crate::link::{Geometry, GeometryShape, Inertial, Link};
use crate::Robot;

/// Load a Robot from a URDF file path.
pub fn load_urdf(path: impl AsRef<Path>) -> kinetic_core::Result<Robot> {
    let urdf =
        urdf_rs::read_file(path.as_ref()).map_err(|e| KineticError::UrdfParse(format!("{e}")))?;
    convert_urdf(urdf)
}

/// Load a Robot from a URDF XML string.
pub fn load_urdf_string(xml: &str) -> kinetic_core::Result<Robot> {
    let urdf =
        urdf_rs::read_from_string(xml).map_err(|e| KineticError::UrdfParse(format!("{e}")))?;
    convert_urdf(urdf)
}

/// Convert a parsed urdf_rs::Robot into our Robot struct.
fn convert_urdf(urdf: urdf_rs::Robot) -> kinetic_core::Result<Robot> {
    // Build link name → index map
    let mut link_name_to_idx: HashMap<String, usize> = HashMap::new();
    let mut links: Vec<Link> = Vec::with_capacity(urdf.links.len());

    for (i, urdf_link) in urdf.links.iter().enumerate() {
        link_name_to_idx.insert(urdf_link.name.clone(), i);
        links.push(convert_link(urdf_link));
    }

    // Build joints
    let mut joints: Vec<Joint> = Vec::with_capacity(urdf.joints.len());
    let mut active_joints: Vec<usize> = Vec::new();
    let mut joint_limits_vec: Vec<JointLimits> = Vec::new();

    for (i, urdf_joint) in urdf.joints.iter().enumerate() {
        let parent_idx = *link_name_to_idx
            .get(&urdf_joint.parent.link)
            .ok_or_else(|| {
                KineticError::UrdfParse(format!(
                    "Parent link '{}' not found for joint '{}'",
                    urdf_joint.parent.link, urdf_joint.name
                ))
            })?;

        let child_idx = *link_name_to_idx
            .get(&urdf_joint.child.link)
            .ok_or_else(|| {
                KineticError::UrdfParse(format!(
                    "Child link '{}' not found for joint '{}'",
                    urdf_joint.child.link, urdf_joint.name
                ))
            })?;

        let joint = convert_joint(urdf_joint, parent_idx, child_idx);

        // Wire up parent-child relationships in links
        links[parent_idx].child_joints.push(i);
        links[child_idx].parent_joint = Some(i);

        if joint.is_active() {
            active_joints.push(i);
            let limits = joint.limits.clone().unwrap_or_default();
            joint_limits_vec.push(limits);
        }

        joints.push(joint);
    }

    // Find root link (no parent joint)
    let root = links
        .iter()
        .position(|l| l.parent_joint.is_none())
        .ok_or_else(|| KineticError::UrdfParse("No root link found".into()))?;

    let dof = active_joints.len();

    Ok(Robot {
        name: urdf.name.clone(),
        joints,
        links,
        dof,
        root,
        active_joints,
        groups: HashMap::new(),
        end_effectors: HashMap::new(),
        named_poses: HashMap::new(),
        joint_limits: joint_limits_vec,
        ik_preference: None,
        collision_preference: None,
    })
}

fn convert_joint(urdf_joint: &urdf_rs::Joint, parent_idx: usize, child_idx: usize) -> Joint {
    let joint_type = match urdf_joint.joint_type {
        urdf_rs::JointType::Revolute => JointType::Revolute,
        urdf_rs::JointType::Continuous => JointType::Continuous,
        urdf_rs::JointType::Prismatic => JointType::Prismatic,
        urdf_rs::JointType::Fixed => JointType::Fixed,
        // urdf-rs also has Floating and Planar — treat as fixed for planning
        _ => JointType::Fixed,
    };

    let origin = convert_pose(&urdf_joint.origin);

    let axis_xyz = urdf_joint.axis.xyz.0;
    let axis = Vector3::new(axis_xyz[0], axis_xyz[1], axis_xyz[2]);

    let limit = &urdf_joint.limit;
    let limits = match joint_type {
        JointType::Fixed => None,
        JointType::Continuous => {
            // Continuous joints have no position limits; only store velocity/effort.
            Some(JointLimits {
                lower: -f64::MAX,
                upper: f64::MAX,
                velocity: limit.velocity,
                effort: limit.effort,
                acceleration: None,
            })
        }
        _ => Some(JointLimits {
            lower: limit.lower,
            upper: limit.upper,
            velocity: limit.velocity,
            effort: limit.effort,
            acceleration: None,
        }),
    };

    Joint {
        name: urdf_joint.name.clone(),
        joint_type,
        parent_link: parent_idx,
        child_link: child_idx,
        origin,
        axis,
        limits,
    }
}

fn convert_link(urdf_link: &urdf_rs::Link) -> Link {
    let visual_geometry = urdf_link
        .visual
        .iter()
        .filter_map(|v| convert_geometry(&v.geometry, &v.origin))
        .collect();

    let collision_geometry = urdf_link
        .collision
        .iter()
        .filter_map(|c| convert_geometry(&c.geometry, &c.origin))
        .collect();

    // urdf-rs Inertial is not Option — it's Default (mass=0).
    // Treat zero-mass as "no inertial data".
    let inertial = {
        let i = &urdf_link.inertial;
        if i.mass.value > 0.0 {
            let inertia_mat = &i.inertia;
            Some(Inertial {
                mass: i.mass.value,
                origin: convert_pose(&i.origin),
                inertia: [
                    inertia_mat.ixx,
                    inertia_mat.ixy,
                    inertia_mat.ixz,
                    inertia_mat.iyy,
                    inertia_mat.iyz,
                    inertia_mat.izz,
                ],
            })
        } else {
            None
        }
    };

    Link {
        name: urdf_link.name.clone(),
        parent_joint: None,   // Will be filled when processing joints
        child_joints: vec![], // Will be filled when processing joints
        visual_geometry,
        collision_geometry,
        inertial,
    }
}

fn convert_geometry(geom: &urdf_rs::Geometry, origin: &urdf_rs::Pose) -> Option<Geometry> {
    let shape = match geom {
        urdf_rs::Geometry::Box { size } => GeometryShape::Box {
            x: size.0[0],
            y: size.0[1],
            z: size.0[2],
        },
        urdf_rs::Geometry::Cylinder { radius, length } => GeometryShape::Cylinder {
            radius: *radius,
            length: *length,
        },
        urdf_rs::Geometry::Capsule { radius, length } => GeometryShape::Cylinder {
            radius: *radius,
            length: *length,
        },
        urdf_rs::Geometry::Sphere { radius } => GeometryShape::Sphere { radius: *radius },
        urdf_rs::Geometry::Mesh { filename, scale } => GeometryShape::Mesh {
            filename: filename.clone(),
            scale: scale.map(|s| s.0).unwrap_or([1.0, 1.0, 1.0]),
        },
    };

    Some(Geometry {
        shape,
        origin: convert_pose(origin),
    })
}

fn convert_pose(pose: &urdf_rs::Pose) -> Pose {
    let xyz = pose.xyz.0;
    let rpy = pose.rpy.0;
    Pose::from_xyz_rpy(xyz[0], xyz[1], xyz[2], rpy[0], rpy[1], rpy[2])
}

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    use super::*;

    /// Minimal 3-DOF arm URDF for testing.
    const THREE_DOF_URDF: &str = r#"<?xml version="1.0"?>
<robot name="test_3dof">
  <link name="base_link">
    <visual>
      <geometry><box size="0.1 0.1 0.05"/></geometry>
    </visual>
    <collision>
      <geometry><box size="0.1 0.1 0.05"/></geometry>
    </collision>
  </link>

  <link name="link1">
    <visual>
      <geometry><cylinder radius="0.03" length="0.3"/></geometry>
    </visual>
  </link>

  <link name="link2">
    <visual>
      <geometry><cylinder radius="0.025" length="0.25"/></geometry>
    </visual>
  </link>

  <link name="ee_link"/>

  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="100"/>
  </joint>

  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="2.0" effort="80"/>
  </joint>

  <joint name="joint3" type="revolute">
    <parent link="link2"/>
    <child link="ee_link"/>
    <origin xyz="0 0 0.25" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.5" upper="2.5" velocity="3.0" effort="50"/>
  </joint>
</robot>
"#;

    /// Panda-like 7-DOF arm URDF for testing.
    const PANDA_LIKE_URDF: &str = r#"<?xml version="1.0"?>
<robot name="panda_like">
  <link name="panda_link0"/>
  <link name="panda_link1"/>
  <link name="panda_link2"/>
  <link name="panda_link3"/>
  <link name="panda_link4"/>
  <link name="panda_link5"/>
  <link name="panda_link6"/>
  <link name="panda_link7"/>
  <link name="panda_link8"/>
  <link name="panda_hand"/>

  <joint name="panda_joint1" type="revolute">
    <parent link="panda_link0"/>
    <child link="panda_link1"/>
    <origin xyz="0 0 0.333" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.9671" upper="2.9671" velocity="2.175" effort="87"/>
  </joint>

  <joint name="panda_joint2" type="revolute">
    <parent link="panda_link1"/>
    <child link="panda_link2"/>
    <origin xyz="0 0 0" rpy="-1.5708 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.8326" upper="1.8326" velocity="2.175" effort="87"/>
  </joint>

  <joint name="panda_joint3" type="revolute">
    <parent link="panda_link2"/>
    <child link="panda_link3"/>
    <origin xyz="0 -0.316 0" rpy="1.5708 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.9671" upper="2.9671" velocity="2.175" effort="87"/>
  </joint>

  <joint name="panda_joint4" type="revolute">
    <parent link="panda_link3"/>
    <child link="panda_link4"/>
    <origin xyz="0.0825 0 0" rpy="1.5708 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.0718" upper="-0.0698" velocity="2.175" effort="87"/>
  </joint>

  <joint name="panda_joint5" type="revolute">
    <parent link="panda_link4"/>
    <child link="panda_link5"/>
    <origin xyz="-0.0825 0.384 0" rpy="-1.5708 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.9671" upper="2.9671" velocity="2.61" effort="12"/>
  </joint>

  <joint name="panda_joint6" type="revolute">
    <parent link="panda_link5"/>
    <child link="panda_link6"/>
    <origin xyz="0 0 0" rpy="1.5708 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-0.0175" upper="3.7525" velocity="2.61" effort="12"/>
  </joint>

  <joint name="panda_joint7" type="revolute">
    <parent link="panda_link6"/>
    <child link="panda_link7"/>
    <origin xyz="0.088 0 0" rpy="1.5708 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-2.9671" upper="2.9671" velocity="2.61" effort="12"/>
  </joint>

  <joint name="panda_joint8" type="fixed">
    <parent link="panda_link7"/>
    <child link="panda_link8"/>
    <origin xyz="0 0 0.107" rpy="0 0 0"/>
  </joint>

  <joint name="panda_hand_joint" type="fixed">
    <parent link="panda_link8"/>
    <child link="panda_hand"/>
    <origin xyz="0 0 0" rpy="0 0 -0.7854"/>
  </joint>
</robot>
"#;

    /// URDF with mixed joint types (revolute, prismatic, continuous, fixed).
    const MIXED_JOINTS_URDF: &str = r#"<?xml version="1.0"?>
<robot name="mixed_joints">
  <link name="base"/>
  <link name="rotary"/>
  <link name="slider"/>
  <link name="spinner"/>
  <link name="fixed_part"/>

  <joint name="revolute_joint" type="revolute">
    <parent link="base"/>
    <child link="rotary"/>
    <origin xyz="0 0 0.1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1.57" upper="1.57" velocity="1.0" effort="50"/>
  </joint>

  <joint name="prismatic_joint" type="prismatic">
    <parent link="rotary"/>
    <child link="slider"/>
    <origin xyz="0 0 0.2"/>
    <axis xyz="0 0 1"/>
    <limit lower="0" upper="0.5" velocity="0.5" effort="100"/>
  </joint>

  <joint name="continuous_joint" type="continuous">
    <parent link="slider"/>
    <child link="spinner"/>
    <origin xyz="0 0 0.1"/>
    <axis xyz="0 0 1"/>
    <limit velocity="10.0" effort="5"/>
  </joint>

  <joint name="fixed_joint" type="fixed">
    <parent link="spinner"/>
    <child link="fixed_part"/>
    <origin xyz="0 0 0.05"/>
  </joint>
</robot>
"#;

    #[test]
    fn load_3dof_arm() {
        let robot = load_urdf_string(THREE_DOF_URDF).unwrap();
        assert_eq!(robot.name, "test_3dof");
        assert_eq!(robot.dof, 3);
        assert_eq!(robot.links.len(), 4);
        assert_eq!(robot.joints.len(), 3);
        assert_eq!(robot.active_joints.len(), 3);
    }

    #[test]
    fn load_panda_like() {
        let robot = load_urdf_string(PANDA_LIKE_URDF).unwrap();
        assert_eq!(robot.name, "panda_like");
        assert_eq!(robot.dof, 7);
        assert_eq!(robot.links.len(), 10);
        assert_eq!(robot.joints.len(), 9);
        assert_eq!(robot.active_joints.len(), 7);
    }

    #[test]
    fn load_mixed_joints() {
        let robot = load_urdf_string(MIXED_JOINTS_URDF).unwrap();
        assert_eq!(robot.name, "mixed_joints");
        assert_eq!(robot.dof, 3); // revolute + prismatic + continuous
        assert_eq!(robot.joints.len(), 4);
        assert_eq!(robot.active_joints.len(), 3);

        // Check joint types
        assert_eq!(robot.joints[0].joint_type, JointType::Revolute);
        assert_eq!(robot.joints[1].joint_type, JointType::Prismatic);
        assert_eq!(robot.joints[2].joint_type, JointType::Continuous);
        assert_eq!(robot.joints[3].joint_type, JointType::Fixed);
    }

    #[test]
    fn root_link_found() {
        let robot = load_urdf_string(THREE_DOF_URDF).unwrap();
        assert_eq!(robot.root, 0);
        assert_eq!(robot.links[robot.root].name, "base_link");
    }

    #[test]
    fn joint_limits_extracted() {
        let robot = load_urdf_string(THREE_DOF_URDF).unwrap();
        assert_eq!(robot.joint_limits.len(), 3);

        let j1_limits = &robot.joint_limits[0];
        assert!((j1_limits.lower - (-3.14)).abs() < 1e-6);
        assert!((j1_limits.upper - 3.14).abs() < 1e-6);
        assert!((j1_limits.velocity - 2.0).abs() < 1e-6);
    }

    #[test]
    fn panda_joint_limits() {
        let robot = load_urdf_string(PANDA_LIKE_URDF).unwrap();
        // Joint 1 limits
        let j1 = &robot.joint_limits[0];
        assert!((j1.lower - (-2.9671)).abs() < 1e-4);
        assert!((j1.upper - 2.9671).abs() < 1e-4);
        assert!((j1.velocity - 2.175).abs() < 1e-4);
        assert!((j1.effort - 87.0).abs() < 1e-4);
    }

    #[test]
    fn link_parent_child() {
        let robot = load_urdf_string(THREE_DOF_URDF).unwrap();

        // base_link has no parent, one child joint
        assert!(robot.links[0].parent_joint.is_none());
        assert_eq!(robot.links[0].child_joints.len(), 1);

        // link1 has parent joint and one child joint
        assert!(robot.links[1].parent_joint.is_some());
        assert_eq!(robot.links[1].child_joints.len(), 1);

        // ee_link is a leaf
        assert!(robot.links[3].parent_joint.is_some());
        assert!(robot.links[3].child_joints.is_empty());
    }

    #[test]
    fn geometry_extracted() {
        let robot = load_urdf_string(THREE_DOF_URDF).unwrap();
        // base_link has visual and collision geometry
        assert_eq!(robot.links[0].visual_geometry.len(), 1);
        assert_eq!(robot.links[0].collision_geometry.len(), 1);

        // link1 has visual but no collision
        assert_eq!(robot.links[1].visual_geometry.len(), 1);
        assert_eq!(robot.links[1].collision_geometry.len(), 0);
    }

    #[test]
    fn invalid_urdf() {
        let result = load_urdf_string("<invalid>not a urdf</invalid>");
        assert!(result.is_err());
    }
}
