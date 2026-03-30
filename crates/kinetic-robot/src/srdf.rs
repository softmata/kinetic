//! SRDF (Semantic Robot Description Format) parser.
//!
//! Parses SRDF XML files used by MoveIt to define planning groups,
//! disabled collision pairs, end-effectors, and named group states.
//!
//! # Example
//!
//! ```ignore
//! use kinetic_robot::srdf::SrdfModel;
//! use kinetic_robot::Robot;
//!
//! let mut robot = Robot::from_urdf("robot.urdf")?;
//! let srdf = SrdfModel::from_file("robot.srdf")?;
//! srdf.apply_to_robot(&mut robot)?;
//! ```

use std::collections::HashMap;
use std::path::Path;

use kinetic_core::{KineticError, Pose, Result};

use crate::joint::JointType;
use crate::robot::{EndEffector, PlanningGroup, Robot};

/// Parsed SRDF model.
#[derive(Debug, Clone)]
pub struct SrdfModel {
    /// Robot name (should match URDF).
    pub name: String,
    /// Planning groups.
    pub groups: Vec<SrdfGroup>,
    /// Disabled collision pairs.
    pub disable_collisions: Vec<DisabledCollision>,
    /// End-effector definitions.
    pub end_effectors: Vec<SrdfEndEffector>,
    /// Named group states (predefined joint configurations).
    pub group_states: Vec<GroupState>,
}

/// A planning group defined in SRDF.
#[derive(Debug, Clone)]
pub struct SrdfGroup {
    /// Group name (e.g., "arm", "gripper").
    pub name: String,
    /// Chain definitions (base_link → tip_link).
    pub chains: Vec<SrdfChain>,
    /// Explicitly listed joint names.
    pub joints: Vec<String>,
    /// Explicitly listed link names.
    pub links: Vec<String>,
}

/// A kinematic chain within a group.
#[derive(Debug, Clone)]
pub struct SrdfChain {
    pub base_link: String,
    pub tip_link: String,
}

/// A disabled collision pair.
#[derive(Debug, Clone)]
pub struct DisabledCollision {
    pub link1: String,
    pub link2: String,
    pub reason: String,
}

/// An end-effector definition.
#[derive(Debug, Clone)]
pub struct SrdfEndEffector {
    pub name: String,
    pub parent_link: String,
    pub group: String,
    pub parent_group: String,
}

/// A named state for a group (predefined joint configuration).
#[derive(Debug, Clone)]
pub struct GroupState {
    pub group: String,
    pub name: String,
    pub joint_values: HashMap<String, f64>,
}

impl SrdfModel {
    /// Parse SRDF from a file path.
    pub fn from_file(path: impl AsRef<Path>) -> Result<Self> {
        let xml = std::fs::read_to_string(path.as_ref())
            .map_err(|e| KineticError::SrdfParse(format!("Failed to read file: {}", e)))?;
        Self::from_string(&xml)
    }

    /// Parse SRDF from an XML string.
    pub fn from_string(xml: &str) -> Result<Self> {
        let doc = roxmltree::Document::parse(xml)
            .map_err(|e| KineticError::SrdfParse(format!("Failed to parse XML: {}", e)))?;

        let root = doc.root_element();
        if root.tag_name().name() != "robot" {
            return Err(KineticError::SrdfParse(
                "Root element must be <robot>".into(),
            ));
        }

        let name = root.attribute("name").unwrap_or("").to_string();

        let mut groups = Vec::new();
        let mut disable_collisions = Vec::new();
        let mut end_effectors = Vec::new();
        let mut group_states = Vec::new();

        for child in root.children().filter(|n| n.is_element()) {
            match child.tag_name().name() {
                "group" => {
                    if let Some(g) = parse_group(&child) {
                        groups.push(g);
                    }
                }
                "disable_collisions" => {
                    if let Some(dc) = parse_disable_collision(&child) {
                        disable_collisions.push(dc);
                    }
                }
                "end_effector" => {
                    if let Some(ee) = parse_end_effector(&child) {
                        end_effectors.push(ee);
                    }
                }
                "group_state" => {
                    if let Some(gs) = parse_group_state(&child) {
                        group_states.push(gs);
                    }
                }
                _ => {} // Ignore unknown elements
            }
        }

        Ok(SrdfModel {
            name,
            groups,
            disable_collisions,
            end_effectors,
            group_states,
        })
    }

    /// Apply this SRDF configuration to a Robot model.
    ///
    /// Populates:
    /// - `robot.groups` — planning groups (from `<group>` with chains)
    /// - `robot.end_effectors` — end-effector definitions
    /// - `robot.named_poses` — from `<group_state>` entries
    /// - `robot.collision_preference.skip_pairs` — from `<disable_collisions>`
    pub fn apply_to_robot(&self, robot: &mut Robot) -> Result<()> {
        // Apply planning groups
        for srdf_group in &self.groups {
            if let Some(pg) = self.convert_group(srdf_group, robot)? {
                robot.groups.insert(pg.name.clone(), pg);
            }
        }

        // Apply end-effectors
        for srdf_ee in &self.end_effectors {
            let ee = EndEffector {
                name: srdf_ee.name.clone(),
                parent_link: srdf_ee.parent_link.clone(),
                parent_group: srdf_ee.parent_group.clone(),
                grasp_frame: Pose::identity(),
            };
            robot.end_effectors.insert(ee.name.clone(), ee);
        }

        // Apply disabled collision pairs
        if !self.disable_collisions.is_empty() {
            let skip_pairs: Vec<(String, String)> = self
                .disable_collisions
                .iter()
                .map(|dc| (dc.link1.clone(), dc.link2.clone()))
                .collect();

            match &mut robot.collision_preference {
                Some(pref) => {
                    pref.skip_pairs.extend(skip_pairs);
                }
                None => {
                    robot.collision_preference = Some(crate::config::CollisionPreference {
                        padding: 0.0,
                        auto_self_collision: true,
                        skip_pairs,
                    });
                }
            }
        }

        // Apply group states as named poses
        for gs in &self.group_states {
            // Find the group to know which joints to include
            let group = robot.groups.get(&gs.group);
            if group.is_none() {
                continue;
            }

            // Build full-robot joint value vector
            // Start from zeros, then fill in the joints specified in the state
            let mut values = vec![0.0; robot.dof];
            for (joint_name, &value) in &gs.joint_values {
                // Find the joint index in the robot
                if let Ok(joint_idx) = robot.joint_index(joint_name) {
                    // Find the position in active_joints
                    if let Some(active_pos) =
                        robot.active_joints.iter().position(|&ai| ai == joint_idx)
                    {
                        values[active_pos] = value;
                    }
                }
            }

            let pose_name = format!("{}_{}", gs.group, gs.name);
            robot.named_poses.insert(pose_name, values);
        }

        Ok(())
    }

    /// Convert an SRDF group to a Kinetic PlanningGroup.
    fn convert_group(
        &self,
        srdf_group: &SrdfGroup,
        robot: &Robot,
    ) -> Result<Option<PlanningGroup>> {
        // If the group has a chain, use it directly
        if let Some(chain) = srdf_group.chains.first() {
            // Verify links exist
            robot.link_index(&chain.base_link)?;
            robot.link_index(&chain.tip_link)?;

            // Get the joint chain and find active joint indices
            let joint_chain = robot.chain(&chain.base_link, &chain.tip_link)?;
            let joint_indices: Vec<usize> = joint_chain
                .iter()
                .filter_map(|&ji| {
                    if robot.joints[ji].joint_type != JointType::Fixed {
                        robot.active_joints.iter().position(|&ai| ai == ji)
                    } else {
                        None
                    }
                })
                .collect();

            return Ok(Some(PlanningGroup {
                name: srdf_group.name.clone(),
                joint_indices,
                base_link: chain.base_link.clone(),
                tip_link: chain.tip_link.clone(),
            }));
        }

        // If the group has explicit joints, build from those
        if !srdf_group.joints.is_empty() {
            let joint_indices: Vec<usize> = srdf_group
                .joints
                .iter()
                .filter_map(|jn| {
                    let ji = robot.joint_index(jn).ok()?;
                    robot.active_joints.iter().position(|&ai| ai == ji)
                })
                .collect();

            if joint_indices.is_empty() {
                return Ok(None);
            }

            // Determine base and tip links from the first/last joints
            let first_joint = &robot.joints[robot.active_joints[joint_indices[0]]];
            let last_joint = &robot.joints[robot.active_joints[*joint_indices.last().unwrap()]];
            let base_link = robot.links[first_joint.parent_link].name.clone();
            let tip_link = robot.links[last_joint.child_link].name.clone();

            return Ok(Some(PlanningGroup {
                name: srdf_group.name.clone(),
                joint_indices,
                base_link,
                tip_link,
            }));
        }

        Ok(None)
    }
}

// ─── XML element parsers ─────────────────────────────────────────────────────

fn parse_group(node: &roxmltree::Node) -> Option<SrdfGroup> {
    let name = node.attribute("name")?.to_string();

    let mut chains = Vec::new();
    let mut joints = Vec::new();
    let mut links = Vec::new();

    for child in node.children().filter(|n| n.is_element()) {
        match child.tag_name().name() {
            "chain" => {
                if let (Some(base), Some(tip)) =
                    (child.attribute("base_link"), child.attribute("tip_link"))
                {
                    chains.push(SrdfChain {
                        base_link: base.to_string(),
                        tip_link: tip.to_string(),
                    });
                }
            }
            "joint" => {
                if let Some(jn) = child.attribute("name") {
                    joints.push(jn.to_string());
                }
            }
            "link" => {
                if let Some(ln) = child.attribute("name") {
                    links.push(ln.to_string());
                }
            }
            _ => {}
        }
    }

    Some(SrdfGroup {
        name,
        chains,
        joints,
        links,
    })
}

fn parse_disable_collision(node: &roxmltree::Node) -> Option<DisabledCollision> {
    let link1 = node.attribute("link1")?.to_string();
    let link2 = node.attribute("link2")?.to_string();
    let reason = node.attribute("reason").unwrap_or("").to_string();

    Some(DisabledCollision {
        link1,
        link2,
        reason,
    })
}

fn parse_end_effector(node: &roxmltree::Node) -> Option<SrdfEndEffector> {
    let name = node.attribute("name")?.to_string();
    let parent_link = node.attribute("parent_link")?.to_string();
    let group = node.attribute("group").unwrap_or("").to_string();
    let parent_group = node.attribute("parent_group").unwrap_or("").to_string();

    Some(SrdfEndEffector {
        name,
        parent_link,
        group,
        parent_group,
    })
}

fn parse_group_state(node: &roxmltree::Node) -> Option<GroupState> {
    let group = node.attribute("group")?.to_string();
    let name = node.attribute("name")?.to_string();

    let mut joint_values = HashMap::new();
    for child in node.children().filter(|n| n.is_element()) {
        if child.tag_name().name() == "joint" {
            if let (Some(jn), Some(val_str)) = (child.attribute("name"), child.attribute("value")) {
                if let Ok(val) = val_str.parse::<f64>() {
                    joint_values.insert(jn.to_string(), val);
                }
            }
        }
    }

    Some(GroupState {
        group,
        name,
        joint_values,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE_SRDF: &str = r#"<?xml version="1.0" ?>
<robot name="test_robot">
  <group name="arm">
    <chain base_link="base_link" tip_link="ee_link" />
  </group>

  <group name="gripper">
    <joint name="finger_joint" />
  </group>

  <group_state name="home" group="arm">
    <joint name="joint1" value="0.0" />
    <joint name="joint2" value="-1.5708" />
    <joint name="joint3" value="0.0" />
  </group_state>

  <end_effector name="gripper_ee" parent_link="ee_link" group="gripper" parent_group="arm" />

  <disable_collisions link1="base_link" link2="link1" reason="Adjacent" />
  <disable_collisions link1="link1" link2="link2" reason="Adjacent" />
  <disable_collisions link1="base_link" link2="link2" reason="Never" />
</robot>
"#;

    #[test]
    fn parse_srdf_basic() {
        let model = SrdfModel::from_string(SAMPLE_SRDF).unwrap();
        assert_eq!(model.name, "test_robot");
        assert_eq!(model.groups.len(), 2);
        assert_eq!(model.disable_collisions.len(), 3);
        assert_eq!(model.end_effectors.len(), 1);
        assert_eq!(model.group_states.len(), 1);
    }

    #[test]
    fn parse_srdf_groups() {
        let model = SrdfModel::from_string(SAMPLE_SRDF).unwrap();

        let arm = &model.groups[0];
        assert_eq!(arm.name, "arm");
        assert_eq!(arm.chains.len(), 1);
        assert_eq!(arm.chains[0].base_link, "base_link");
        assert_eq!(arm.chains[0].tip_link, "ee_link");

        let gripper = &model.groups[1];
        assert_eq!(gripper.name, "gripper");
        assert_eq!(gripper.joints, vec!["finger_joint"]);
    }

    #[test]
    fn parse_srdf_collisions() {
        let model = SrdfModel::from_string(SAMPLE_SRDF).unwrap();

        assert_eq!(model.disable_collisions[0].link1, "base_link");
        assert_eq!(model.disable_collisions[0].link2, "link1");
        assert_eq!(model.disable_collisions[0].reason, "Adjacent");

        assert_eq!(model.disable_collisions[2].reason, "Never");
    }

    #[test]
    fn parse_srdf_end_effectors() {
        let model = SrdfModel::from_string(SAMPLE_SRDF).unwrap();

        let ee = &model.end_effectors[0];
        assert_eq!(ee.name, "gripper_ee");
        assert_eq!(ee.parent_link, "ee_link");
        assert_eq!(ee.group, "gripper");
        assert_eq!(ee.parent_group, "arm");
    }

    #[test]
    fn parse_srdf_group_states() {
        let model = SrdfModel::from_string(SAMPLE_SRDF).unwrap();

        let gs = &model.group_states[0];
        assert_eq!(gs.group, "arm");
        assert_eq!(gs.name, "home");
        assert_eq!(gs.joint_values.len(), 3);
        assert!((gs.joint_values["joint2"] - (-std::f64::consts::FRAC_PI_2)).abs() < 1e-4);
    }

    #[test]
    fn parse_empty_srdf() {
        let xml = r#"<?xml version="1.0" ?><robot name="empty"></robot>"#;
        let model = SrdfModel::from_string(xml).unwrap();
        assert_eq!(model.name, "empty");
        assert!(model.groups.is_empty());
        assert!(model.disable_collisions.is_empty());
    }

    #[test]
    fn apply_srdf_to_robot() {
        let urdf = r#"<?xml version="1.0"?>
<robot name="test_robot">
  <link name="base_link"/>
  <link name="link1"/>
  <link name="link2"/>
  <link name="ee_link"/>

  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <origin xyz="0 0 0.1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="100"/>
  </joint>

  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <origin xyz="0 0 0.3"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="2.0" effort="80"/>
  </joint>

  <joint name="joint3" type="revolute">
    <parent link="link2"/>
    <child link="ee_link"/>
    <origin xyz="0 0 0.25"/>
    <axis xyz="0 1 0"/>
    <limit lower="-2.5" upper="2.5" velocity="3.0" effort="50"/>
  </joint>
</robot>"#;

        let srdf = r#"<?xml version="1.0" ?>
<robot name="test_robot">
  <group name="arm">
    <chain base_link="base_link" tip_link="ee_link" />
  </group>

  <group_state name="home" group="arm">
    <joint name="joint1" value="0.0" />
    <joint name="joint2" value="-1.5708" />
    <joint name="joint3" value="0.0" />
  </group_state>

  <end_effector name="tool" parent_link="ee_link" group="" parent_group="arm" />

  <disable_collisions link1="base_link" link2="link1" reason="Adjacent" />
  <disable_collisions link1="link1" link2="link2" reason="Adjacent" />
</robot>"#;

        let mut robot = Robot::from_urdf_string(urdf).unwrap();
        let model = SrdfModel::from_string(srdf).unwrap();
        model.apply_to_robot(&mut robot).unwrap();

        // Check group was applied
        assert!(robot.groups.contains_key("arm"));
        let arm = &robot.groups["arm"];
        assert_eq!(arm.base_link, "base_link");
        assert_eq!(arm.tip_link, "ee_link");
        assert_eq!(arm.joint_indices.len(), 3);

        // Check end-effector was applied
        assert!(robot.end_effectors.contains_key("tool"));
        assert_eq!(robot.end_effectors["tool"].parent_link, "ee_link");

        // Check collision skip pairs
        let pref = robot.collision_preference.as_ref().unwrap();
        assert_eq!(pref.skip_pairs.len(), 2);

        // Check named pose was applied (prefixed with group name)
        assert!(robot.named_poses.contains_key("arm_home"));
        let home = &robot.named_poses["arm_home"];
        assert_eq!(home.len(), 3);
        assert!((home[1] - (-std::f64::consts::FRAC_PI_2)).abs() < 1e-4);
    }

    #[test]
    fn parse_invalid_xml() {
        let result = SrdfModel::from_string("not xml");
        assert!(result.is_err());
    }

    #[test]
    fn parse_wrong_root() {
        let xml = r#"<?xml version="1.0" ?><model name="foo"></model>"#;
        let result = SrdfModel::from_string(xml);
        assert!(result.is_err());
    }
}
