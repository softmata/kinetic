//! MJCF (MuJoCo XML) loading and conversion to Robot.
//!
//! Parses MuJoCo's XML format including:
//! - `<body>` hierarchy → link tree
//! - `<joint>` elements → joints with limits
//! - `<geom>` elements → collision/visual geometry
//! - `<default>` classes for inherited properties
//! - `<actuator>` elements → effort limits

use std::collections::HashMap;
use std::path::Path;

use kinetic_core::{KineticError, Pose};
use nalgebra::{Isometry3, Translation3, UnitQuaternion, Vector3};

use crate::joint::{Joint, JointLimits, JointType};
use crate::link::{Geometry, GeometryShape, Link};
use crate::Robot;

/// Load a Robot from an MJCF file path.
pub fn load_mjcf(path: impl AsRef<Path>) -> kinetic_core::Result<Robot> {
    let xml = std::fs::read_to_string(path.as_ref())
        .map_err(|e| KineticError::UrdfParse(format!("Failed to read MJCF file: {e}")))?;
    load_mjcf_string(&xml)
}

/// Load a Robot from an MJCF XML string.
pub fn load_mjcf_string(xml: &str) -> kinetic_core::Result<Robot> {
    let doc = roxmltree::Document::parse(xml)
        .map_err(|e| KineticError::UrdfParse(format!("MJCF XML parse error: {e}")))?;

    let mujoco = doc
        .root_element()
        .children()
        .find(|n| n.has_tag_name("mujoco"))
        .unwrap_or_else(|| doc.root_element());

    // Parse defaults for inherited properties
    let defaults = parse_defaults(&mujoco);

    // Find the worldbody
    let worldbody = find_child(&mujoco, "worldbody")
        .ok_or_else(|| KineticError::UrdfParse("MJCF missing <worldbody> element".into()))?;

    // Parse the body tree recursively
    let mut links: Vec<Link> = Vec::new();
    let mut joints: Vec<Joint> = Vec::new();
    let mut active_joints: Vec<usize> = Vec::new();
    let mut joint_limits_vec: Vec<JointLimits> = Vec::new();

    // Root link (worldbody)
    let model_name = mujoco
        .attribute("model")
        .unwrap_or("mjcf_robot")
        .to_string();

    links.push(Link {
        name: "world".to_string(),
        parent_joint: None,
        child_joints: Vec::new(),
        visual_geometry: Vec::new(),
        collision_geometry: Vec::new(),
        inertial: None,
    });

    // Recursively parse body hierarchy
    for body_node in worldbody.children().filter(|n| n.has_tag_name("body")) {
        parse_body(
            &body_node,
            0, // parent link = world
            &defaults,
            &mut links,
            &mut joints,
            &mut active_joints,
            &mut joint_limits_vec,
        );
    }

    // Wire up parent-child relationships for links
    for (ji, joint) in joints.iter().enumerate() {
        if joint.parent_link < links.len() {
            links[joint.parent_link].child_joints.push(ji);
        }
        if joint.child_link < links.len() {
            links[joint.child_link].parent_joint = Some(ji);
        }
    }

    // Find root link (no parent joint)
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

/// Default values that can be inherited by child elements.
#[derive(Debug, Clone, Default)]
struct MjcfDefaults {
    joint_damping: f64,
    joint_limited: bool,
    joint_range: [f64; 2],
    geom_type: String,
    geom_size: Vec<f64>,
}

fn parse_defaults(mujoco: &roxmltree::Node) -> MjcfDefaults {
    let mut defaults = MjcfDefaults::default();

    if let Some(default_node) = find_child(mujoco, "default") {
        if let Some(joint_node) = find_child(&default_node, "joint") {
            if let Some(d) = joint_node.attribute("damping") {
                defaults.joint_damping = d.parse().unwrap_or(0.0);
            }
            if let Some(l) = joint_node.attribute("limited") {
                defaults.joint_limited = l == "true";
            }
            if let Some(r) = joint_node.attribute("range") {
                let parts: Vec<f64> = r
                    .split_whitespace()
                    .filter_map(|s| s.parse().ok())
                    .collect();
                if parts.len() == 2 {
                    defaults.joint_range = [parts[0], parts[1]];
                }
            }
        }
        if let Some(geom_node) = find_child(&default_node, "geom") {
            if let Some(t) = geom_node.attribute("type") {
                defaults.geom_type = t.to_string();
            }
            if let Some(s) = geom_node.attribute("size") {
                defaults.geom_size = parse_floats(s);
            }
        }
    }

    defaults
}

fn parse_body(
    node: &roxmltree::Node,
    parent_link_idx: usize,
    defaults: &MjcfDefaults,
    links: &mut Vec<Link>,
    joints: &mut Vec<Joint>,
    active_joints: &mut Vec<usize>,
    joint_limits_vec: &mut Vec<JointLimits>,
) {
    let body_name = node.attribute("name").unwrap_or("unnamed_body").to_string();
    let body_pos = parse_vec3_attr(node, "pos").unwrap_or(Vector3::zeros());
    let body_quat = parse_quat_attr(node);

    // Create a link for this body
    let link_idx = links.len();
    let mut collision_geoms = Vec::new();
    let mut visual_geoms = Vec::new();

    // Parse geom elements within this body
    for geom in node.children().filter(|n| n.has_tag_name("geom")) {
        if let Some(shape) = parse_geom(&geom, defaults) {
            let geom_pos = parse_vec3_attr(&geom, "pos").unwrap_or(Vector3::zeros());
            let geom_quat = parse_quat_attr(&geom);
            let origin = Isometry3::from_parts(Translation3::from(geom_pos), geom_quat);
            let geom_obj = Geometry {
                shape,
                origin: Pose(origin),
            };
            // In MJCF, geoms are both visual and collision by default
            collision_geoms.push(geom_obj.clone());
            visual_geoms.push(geom_obj);
        }
    }

    links.push(Link {
        name: body_name.clone(),
        parent_joint: None,
        child_joints: Vec::new(),
        visual_geometry: visual_geoms,
        collision_geometry: collision_geoms,
        inertial: None,
    });

    // Parse joint elements — create a joint connecting parent to this body
    let body_joints: Vec<_> = node
        .children()
        .filter(|n| n.has_tag_name("joint"))
        .collect();

    if body_joints.is_empty() {
        // No explicit joint → create a fixed joint
        let joint_idx = joints.len();
        let origin = Isometry3::from_parts(Translation3::from(body_pos), body_quat);

        joints.push(Joint {
            name: format!("{}_fixed", body_name),
            joint_type: JointType::Fixed,
            parent_link: parent_link_idx,
            child_link: link_idx,
            origin: Pose(origin),
            axis: Vector3::z(),
            limits: Some(JointLimits::default()),
        });
        let _ = joint_idx; // fixed joints are not active
    } else {
        // Create a joint for each joint element
        // (most bodies have one joint; some have multiple for ball joints)
        for jnode in &body_joints {
            let joint_name = jnode
                .attribute("name")
                .unwrap_or(&format!("{}_joint", body_name))
                .to_string();

            let joint_type_str = jnode.attribute("type").unwrap_or("hinge");
            let joint_type = match joint_type_str {
                "hinge" => JointType::Revolute,
                "slide" => JointType::Prismatic,
                "ball" => JointType::Revolute, // Approximate ball as revolute
                "free" => JointType::Revolute, // Approximate free as revolute
                _ => JointType::Fixed,
            };

            let axis = parse_vec3_attr(jnode, "axis").unwrap_or(Vector3::z());
            let origin = Isometry3::from_parts(Translation3::from(body_pos), body_quat);

            // Parse limits
            let limited = jnode
                .attribute("limited")
                .map(|l| l == "true")
                .unwrap_or(defaults.joint_limited);

            let range = jnode
                .attribute("range")
                .map(|r| {
                    let parts: Vec<f64> = r
                        .split_whitespace()
                        .filter_map(|s| s.parse().ok())
                        .collect();
                    if parts.len() == 2 {
                        [parts[0], parts[1]]
                    } else {
                        defaults.joint_range
                    }
                })
                .unwrap_or(defaults.joint_range);

            // MJCF range is in degrees for hinge joints by default
            let (lower, upper) = if limited && joint_type_str == "hinge" {
                (range[0].to_radians(), range[1].to_radians())
            } else if limited {
                (range[0], range[1])
            } else {
                (-std::f64::consts::PI, std::f64::consts::PI)
            };

            let limits = JointLimits {
                lower,
                upper,
                velocity: 2.0, // Default; MJCF doesn't always specify
                effort: 100.0,
                acceleration: None,
            };

            let joint_idx = joints.len();
            let is_active = matches!(
                joint_type,
                JointType::Revolute | JointType::Prismatic | JointType::Continuous
            );

            joints.push(Joint {
                name: joint_name,
                joint_type,
                parent_link: parent_link_idx,
                child_link: link_idx,
                origin: Pose(origin),
                axis,
                limits: Some(limits.clone()),
            });

            if is_active {
                active_joints.push(joint_idx);
                joint_limits_vec.push(limits);
            }
        }
    }

    // Recurse into child bodies
    for child_body in node.children().filter(|n| n.has_tag_name("body")) {
        parse_body(
            &child_body,
            link_idx,
            defaults,
            links,
            joints,
            active_joints,
            joint_limits_vec,
        );
    }
}

fn parse_geom(node: &roxmltree::Node, defaults: &MjcfDefaults) -> Option<GeometryShape> {
    let geom_type = node
        .attribute("type")
        .unwrap_or(if defaults.geom_type.is_empty() {
            "sphere"
        } else {
            &defaults.geom_type
        });

    let size = node
        .attribute("size")
        .map(parse_floats)
        .unwrap_or_else(|| defaults.geom_size.clone());

    match geom_type {
        "box" => {
            if size.len() >= 3 {
                Some(GeometryShape::Box {
                    x: size[0] * 2.0,
                    y: size[1] * 2.0,
                    z: size[2] * 2.0,
                })
            } else {
                None
            }
        }
        "sphere" => {
            if !size.is_empty() {
                Some(GeometryShape::Sphere { radius: size[0] })
            } else {
                None
            }
        }
        "cylinder" | "capsule" => {
            if size.len() >= 2 {
                Some(GeometryShape::Cylinder {
                    radius: size[0],
                    length: size[1] * 2.0,
                })
            } else if !size.is_empty() {
                Some(GeometryShape::Cylinder {
                    radius: size[0],
                    length: size[0] * 2.0,
                })
            } else {
                None
            }
        }
        "mesh" => {
            let mesh_name = node.attribute("mesh").unwrap_or("unknown");
            Some(GeometryShape::Mesh {
                filename: mesh_name.to_string(),
                scale: [1.0, 1.0, 1.0],
            })
        }
        _ => None,
    }
}

// --- Helper functions ---

fn find_child<'a>(node: &'a roxmltree::Node, tag: &str) -> Option<roxmltree::Node<'a, 'a>> {
    node.children().find(|n| n.has_tag_name(tag))
}

fn parse_floats(s: &str) -> Vec<f64> {
    s.split_whitespace()
        .filter_map(|v| v.parse().ok())
        .collect()
}

fn parse_vec3_attr(node: &roxmltree::Node, attr: &str) -> Option<Vector3<f64>> {
    node.attribute(attr).and_then(|s| {
        let v = parse_floats(s);
        if v.len() >= 3 {
            Some(Vector3::new(v[0], v[1], v[2]))
        } else {
            None
        }
    })
}

fn parse_quat_attr(node: &roxmltree::Node) -> UnitQuaternion<f64> {
    if let Some(quat_str) = node.attribute("quat") {
        let v = parse_floats(quat_str);
        if v.len() >= 4 {
            // MJCF quaternion order: w, x, y, z
            return UnitQuaternion::from_quaternion(nalgebra::Quaternion::new(
                v[0], v[1], v[2], v[3],
            ));
        }
    }
    if let Some(euler_str) = node.attribute("euler") {
        let v = parse_floats(euler_str);
        if v.len() >= 3 {
            let r = nalgebra::Rotation3::from_euler_angles(v[0], v[1], v[2]);
            return UnitQuaternion::from_rotation_matrix(&r);
        }
    }
    if let Some(axisangle_str) = node.attribute("axisangle") {
        let v = parse_floats(axisangle_str);
        if v.len() >= 4 {
            let axis = Vector3::new(v[0], v[1], v[2]);
            if let Some(unit_axis) = nalgebra::Unit::try_new(axis, 1e-10) {
                return UnitQuaternion::from_axis_angle(&unit_axis, v[3]);
            }
        }
    }
    UnitQuaternion::identity()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_simple_mjcf() {
        let xml = r#"
        <mujoco model="test_arm">
          <worldbody>
            <body name="link1" pos="0 0 0.5">
              <joint name="joint1" type="hinge" axis="0 0 1" limited="true" range="-180 180"/>
              <geom type="cylinder" size="0.05 0.25"/>
              <body name="link2" pos="0 0 0.5">
                <joint name="joint2" type="hinge" axis="0 1 0" limited="true" range="-90 90"/>
                <geom type="cylinder" size="0.04 0.2"/>
              </body>
            </body>
          </worldbody>
        </mujoco>
        "#;

        let robot = load_mjcf_string(xml).unwrap();
        assert_eq!(robot.name, "test_arm");
        assert_eq!(robot.dof, 2);
        assert_eq!(robot.links.len(), 3); // world + link1 + link2
        assert_eq!(robot.active_joints.len(), 2);
        assert_eq!(robot.joints[robot.active_joints[0]].name, "joint1");
        assert_eq!(robot.joints[robot.active_joints[1]].name, "joint2");

        // Check joint limits converted from degrees to radians
        assert!((robot.joint_limits[0].lower - (-std::f64::consts::PI)).abs() < 0.01);
        assert!((robot.joint_limits[0].upper - std::f64::consts::PI).abs() < 0.01);
    }

    #[test]
    fn parse_fixed_bodies() {
        let xml = r#"
        <mujoco model="fixed_test">
          <worldbody>
            <body name="base" pos="0 0 0">
              <geom type="box" size="0.1 0.1 0.1"/>
            </body>
          </worldbody>
        </mujoco>
        "#;

        let robot = load_mjcf_string(xml).unwrap();
        assert_eq!(robot.dof, 0);
        assert_eq!(robot.links.len(), 2); // world + base
    }

    #[test]
    fn parse_defaults() {
        let xml = r#"
        <mujoco model="defaults_test">
          <default>
            <joint damping="1.0" limited="true" range="-90 90"/>
          </default>
          <worldbody>
            <body name="link1" pos="0 0 0.3">
              <joint name="j1" type="hinge" axis="0 0 1"/>
              <geom type="sphere" size="0.05"/>
            </body>
          </worldbody>
        </mujoco>
        "#;

        let robot = load_mjcf_string(xml).unwrap();
        assert_eq!(robot.dof, 1);
        // Should inherit limited=true and range from defaults
        assert!((robot.joint_limits[0].lower - (-90.0_f64).to_radians()).abs() < 0.01);
        assert!((robot.joint_limits[0].upper - 90.0_f64.to_radians()).abs() < 0.01);
    }

    #[test]
    fn parse_quaternion_orientation() {
        let xml = r#"
        <mujoco model="quat_test">
          <worldbody>
            <body name="link1" pos="0 0 1" quat="1 0 0 0">
              <joint name="j1" type="hinge" axis="0 0 1"/>
              <geom type="sphere" size="0.05"/>
            </body>
          </worldbody>
        </mujoco>
        "#;

        let robot = load_mjcf_string(xml).unwrap();
        assert_eq!(robot.dof, 1);
        // Quaternion w=1, xyz=0 should be identity rotation
        let origin = &robot.joints[robot.active_joints[0]].origin.0;
        assert!((origin.translation.z - 1.0).abs() < 1e-6);
    }
}
