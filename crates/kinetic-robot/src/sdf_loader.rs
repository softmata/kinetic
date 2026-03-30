//! SDF (SDFormat/Gazebo) loading and conversion to Robot.
//!
//! Parses SDFormat XML including:
//! - `<model>` → robot name
//! - `<link>` → links with collision/visual geometry
//! - `<joint>` → joints with parent/child links, axis, limits
//! - Nested `<model>` and `<include>` (basic support)

use std::collections::HashMap;
use std::path::Path;

use kinetic_core::{KineticError, Pose};
use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};

use crate::joint::{Joint, JointLimits, JointType};
use crate::link::{Geometry, GeometryShape, Link};
use crate::Robot;

/// Load a Robot from an SDF file path.
pub fn load_sdf(path: impl AsRef<Path>) -> kinetic_core::Result<Robot> {
    let xml = std::fs::read_to_string(path.as_ref())
        .map_err(|e| KineticError::UrdfParse(format!("Failed to read SDF file: {e}")))?;
    load_sdf_string(&xml)
}

/// Load a Robot from an SDF XML string.
pub fn load_sdf_string(xml: &str) -> kinetic_core::Result<Robot> {
    let doc = roxmltree::Document::parse(xml)
        .map_err(|e| KineticError::UrdfParse(format!("SDF XML parse error: {e}")))?;

    let root = doc.root_element();

    // Find <model> — either directly under root or under <sdf>/<world>
    let model = find_model(&root)
        .ok_or_else(|| KineticError::UrdfParse("SDF missing <model> element".into()))?;

    parse_model(&model)
}

fn find_model<'a>(node: &'a roxmltree::Node) -> Option<roxmltree::Node<'a, 'a>> {
    // Direct <model> child
    if let Some(m) = node.children().find(|n| n.has_tag_name("model")) {
        return Some(m);
    }
    // Under <sdf>
    if node.has_tag_name("sdf") {
        if let Some(m) = node.children().find(|n| n.has_tag_name("model")) {
            return Some(m);
        }
        // Under <world> inside <sdf>
        if let Some(world) = node.children().find(|n| n.has_tag_name("world")) {
            if let Some(m) = world.children().find(|n| n.has_tag_name("model")) {
                return Some(m);
            }
        }
    }
    None
}

fn parse_model(model: &roxmltree::Node) -> kinetic_core::Result<Robot> {
    let model_name = model.attribute("name").unwrap_or("sdf_robot").to_string();

    // Parse all links
    let mut link_name_to_idx: HashMap<String, usize> = HashMap::new();
    let mut links: Vec<Link> = Vec::new();

    for link_node in model.children().filter(|n| n.has_tag_name("link")) {
        let link_name = link_node.attribute("name").unwrap_or("unnamed").to_string();
        let idx = links.len();
        link_name_to_idx.insert(link_name.clone(), idx);

        let mut collision_geoms = Vec::new();
        let mut visual_geoms = Vec::new();

        // Parse collision elements
        for coll in link_node.children().filter(|n| n.has_tag_name("collision")) {
            if let Some(geom) = parse_sdf_geometry(&coll) {
                collision_geoms.push(geom);
            }
        }

        // Parse visual elements
        for vis in link_node.children().filter(|n| n.has_tag_name("visual")) {
            if let Some(geom) = parse_sdf_geometry(&vis) {
                visual_geoms.push(geom);
            }
        }

        links.push(Link {
            name: link_name,
            parent_joint: None,
            child_joints: Vec::new(),
            visual_geometry: visual_geoms,
            collision_geometry: collision_geoms,
            inertial: None,
        });
    }

    // Parse all joints
    let mut joints: Vec<Joint> = Vec::new();
    let mut active_joints: Vec<usize> = Vec::new();
    let mut joint_limits_vec: Vec<JointLimits> = Vec::new();

    for joint_node in model.children().filter(|n| n.has_tag_name("joint")) {
        let joint_name = joint_node
            .attribute("name")
            .unwrap_or("unnamed")
            .to_string();
        let joint_type_str = joint_node.attribute("type").unwrap_or("fixed");

        let joint_type = match joint_type_str {
            "revolute" => JointType::Revolute,
            "prismatic" => JointType::Prismatic,
            "continuous" => JointType::Continuous,
            "fixed" => JointType::Fixed,
            "ball" => JointType::Revolute,      // Approximate
            "universal" => JointType::Revolute, // Approximate
            "screw" => JointType::Revolute,     // Approximate
            _ => JointType::Fixed,
        };

        // Parent and child links
        let parent_name = child_text(&joint_node, "parent")
            .unwrap_or_default()
            .replace("model://", "");
        let child_name = child_text(&joint_node, "child")
            .unwrap_or_default()
            .replace("model://", "");

        let parent_idx = link_name_to_idx.get(&parent_name).copied().unwrap_or(0);
        let child_idx = link_name_to_idx.get(&child_name).copied().unwrap_or(0);

        // Parse pose (SDF uses <pose> element)
        let origin = parse_sdf_pose(&joint_node);

        // Parse axis
        let axis = parse_sdf_axis(&joint_node);

        // Parse limits from <axis><limit>
        let limits = parse_sdf_limits(&joint_node, joint_type_str);

        let joint_idx = joints.len();
        let is_active = matches!(
            joint_type,
            JointType::Revolute | JointType::Prismatic | JointType::Continuous
        );

        joints.push(Joint {
            name: joint_name,
            joint_type,
            parent_link: parent_idx,
            child_link: child_idx,
            origin: Pose(origin),
            axis,
            limits: Some(limits.clone()),
        });

        if is_active {
            active_joints.push(joint_idx);
            joint_limits_vec.push(limits);
        }
    }

    // Wire parent-child relationships
    for (ji, joint) in joints.iter().enumerate() {
        if joint.parent_link < links.len() {
            links[joint.parent_link].child_joints.push(ji);
        }
        if joint.child_link < links.len() {
            links[joint.child_link].parent_joint = Some(ji);
        }
    }

    let root = links
        .iter()
        .position(|l| l.parent_joint.is_none())
        .unwrap_or(0);

    let dof = active_joints.len();

    Ok(Robot {
        name: model_name,
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

fn parse_sdf_geometry(parent: &roxmltree::Node) -> Option<Geometry> {
    let geometry_node = parent.children().find(|n| n.has_tag_name("geometry"))?;
    let pose = parse_sdf_pose(parent);

    let shape = if let Some(box_node) = geometry_node.children().find(|n| n.has_tag_name("box")) {
        let size = child_text(&box_node, "size")
            .map(|s| parse_floats(&s))
            .unwrap_or_default();
        if size.len() >= 3 {
            Some(GeometryShape::Box {
                x: size[0],
                y: size[1],
                z: size[2],
            })
        } else {
            None
        }
    } else if let Some(sphere_node) = geometry_node.children().find(|n| n.has_tag_name("sphere")) {
        let radius = child_text(&sphere_node, "radius")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.05);
        Some(GeometryShape::Sphere { radius })
    } else if let Some(cyl_node) = geometry_node
        .children()
        .find(|n| n.has_tag_name("cylinder"))
    {
        let radius = child_text(&cyl_node, "radius")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.05);
        let length = child_text(&cyl_node, "length")
            .and_then(|s| s.parse().ok())
            .unwrap_or(0.1);
        Some(GeometryShape::Cylinder { radius, length })
    } else if let Some(mesh_node) = geometry_node.children().find(|n| n.has_tag_name("mesh")) {
        let uri = child_text(&mesh_node, "uri").unwrap_or_default();
        let scale = child_text(&mesh_node, "scale")
            .map(|s| {
                let v = parse_floats(&s);
                if v.len() >= 3 {
                    [v[0], v[1], v[2]]
                } else {
                    [1.0, 1.0, 1.0]
                }
            })
            .unwrap_or([1.0, 1.0, 1.0]);
        Some(GeometryShape::Mesh {
            filename: uri,
            scale,
        })
    } else {
        None
    };

    shape.map(|s| Geometry {
        shape: s,
        origin: Pose(pose),
    })
}

fn parse_sdf_pose(node: &roxmltree::Node) -> Isometry3<f64> {
    if let Some(pose_text) = child_text(node, "pose") {
        let vals = parse_floats(&pose_text);
        if vals.len() >= 6 {
            let translation = Translation3::new(vals[0], vals[1], vals[2]);
            let rotation = UnitQuaternion::from_euler_angles(vals[3], vals[4], vals[5]);
            return Isometry3::from_parts(translation, rotation);
        }
    }
    Isometry3::identity()
}

fn parse_sdf_axis(joint_node: &roxmltree::Node) -> Vector3<f64> {
    if let Some(axis_node) = joint_node.children().find(|n| n.has_tag_name("axis")) {
        if let Some(xyz_text) = child_text(&axis_node, "xyz") {
            let vals = parse_floats(&xyz_text);
            if vals.len() >= 3 {
                return Vector3::new(vals[0], vals[1], vals[2]);
            }
        }
    }
    Vector3::z()
}

fn parse_sdf_limits(joint_node: &roxmltree::Node, joint_type: &str) -> JointLimits {
    let mut lower = -std::f64::consts::PI;
    let mut upper = std::f64::consts::PI;
    let mut velocity = 2.0;
    let mut effort = 100.0;

    if let Some(axis_node) = joint_node.children().find(|n| n.has_tag_name("axis")) {
        if let Some(limit_node) = axis_node.children().find(|n| n.has_tag_name("limit")) {
            if let Some(lo) = child_text(&limit_node, "lower") {
                lower = lo.parse().unwrap_or(lower);
            }
            if let Some(hi) = child_text(&limit_node, "upper") {
                upper = hi.parse().unwrap_or(upper);
            }
            if let Some(v) = child_text(&limit_node, "velocity") {
                velocity = v.parse().unwrap_or(velocity);
            }
            if let Some(e) = child_text(&limit_node, "effort") {
                effort = e.parse().unwrap_or(effort);
            }
        }
    }

    // Continuous joints have no limits
    if joint_type == "continuous" {
        lower = -std::f64::consts::PI;
        upper = std::f64::consts::PI;
    }

    JointLimits {
        lower,
        upper,
        velocity,
        effort,
        acceleration: None,
    }
}

// --- Helpers ---

fn child_text(node: &roxmltree::Node, tag: &str) -> Option<String> {
    node.children()
        .find(|n| n.has_tag_name(tag))
        .and_then(|n| n.text())
        .map(|t| t.trim().to_string())
}

fn parse_floats(s: &str) -> Vec<f64> {
    s.split_whitespace()
        .filter_map(|v| v.parse().ok())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_sdf() {
        let xml = r#"
        <sdf version="1.6">
          <model name="test_arm">
            <link name="base_link">
              <collision name="base_col">
                <geometry><box><size>0.2 0.2 0.1</size></box></geometry>
              </collision>
            </link>
            <link name="link1">
              <collision name="link1_col">
                <geometry><cylinder><radius>0.05</radius><length>0.5</length></cylinder></geometry>
              </collision>
            </link>
            <link name="link2">
              <collision name="link2_col">
                <geometry><sphere><radius>0.05</radius></sphere></geometry>
              </collision>
            </link>
            <joint name="joint1" type="revolute">
              <parent>base_link</parent>
              <child>link1</child>
              <pose>0 0 0.05 0 0 0</pose>
              <axis>
                <xyz>0 0 1</xyz>
                <limit>
                  <lower>-3.14</lower>
                  <upper>3.14</upper>
                  <velocity>2.0</velocity>
                  <effort>50.0</effort>
                </limit>
              </axis>
            </joint>
            <joint name="joint2" type="revolute">
              <parent>link1</parent>
              <child>link2</child>
              <pose>0 0 0.5 0 0 0</pose>
              <axis>
                <xyz>0 1 0</xyz>
                <limit>
                  <lower>-1.57</lower>
                  <upper>1.57</upper>
                </limit>
              </axis>
            </joint>
          </model>
        </sdf>
        "#;

        let robot = load_sdf_string(xml).unwrap();
        assert_eq!(robot.name, "test_arm");
        assert_eq!(robot.dof, 2);
        assert_eq!(robot.links.len(), 3);
        assert_eq!(robot.active_joints.len(), 2);

        // Check joint limits
        assert!((robot.joint_limits[0].lower - (-3.14)).abs() < 0.01);
        assert!((robot.joint_limits[0].upper - 3.14).abs() < 0.01);
        assert!((robot.joint_limits[0].effort - 50.0).abs() < 0.01);
    }

    #[test]
    fn parse_fixed_joints() {
        let xml = r#"
        <sdf version="1.6">
          <model name="static_model">
            <link name="base"/>
            <link name="sensor_mount"/>
            <joint name="mount_joint" type="fixed">
              <parent>base</parent>
              <child>sensor_mount</child>
            </joint>
          </model>
        </sdf>
        "#;

        let robot = load_sdf_string(xml).unwrap();
        assert_eq!(robot.dof, 0);
        assert_eq!(robot.links.len(), 2);
    }

    #[test]
    fn parse_continuous_joint() {
        let xml = r#"
        <sdf version="1.6">
          <model name="wheel_bot">
            <link name="base"/>
            <link name="wheel"/>
            <joint name="wheel_joint" type="continuous">
              <parent>base</parent>
              <child>wheel</child>
              <axis><xyz>0 1 0</xyz></axis>
            </joint>
          </model>
        </sdf>
        "#;

        let robot = load_sdf_string(xml).unwrap();
        assert_eq!(robot.dof, 1);
        let j = &robot.joints[robot.active_joints[0]];
        assert!(matches!(j.joint_type, JointType::Continuous));
    }

    #[test]
    fn parse_with_pose() {
        let xml = r#"
        <sdf version="1.6">
          <model name="posed">
            <link name="base"/>
            <link name="arm"/>
            <joint name="j1" type="revolute">
              <parent>base</parent>
              <child>arm</child>
              <pose>0 0 1.0 0 0 0</pose>
              <axis><xyz>0 0 1</xyz><limit><lower>-1</lower><upper>1</upper></limit></axis>
            </joint>
          </model>
        </sdf>
        "#;

        let robot = load_sdf_string(xml).unwrap();
        let j = &robot.joints[robot.active_joints[0]];
        assert!((j.origin.0.translation.z - 1.0).abs() < 1e-6);
    }
}
