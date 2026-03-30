//! Robot setup assistant: config wizard, validation, SRDF migration, scaffolding.
//!
//! Provides tools for configuring robots:
//! - Planning group definition (chain of joints for IK/planning)
//! - Allowed Collision Matrix (ACM) auto-generation
//! - End-effector configuration
//! - Named pose definition (home, ready, tucked, etc.)
//! - Config validation (detect issues before planning)
//! - MoveIt2 SRDF/YAML import
//! - Project scaffolding (generate kinetic config files from URDF)

use std::collections::HashMap;

use crate::{JointType, Robot};

// ═══════════════════════════════════════════════════════════════════════════
// Robot Configuration
// ═══════════════════════════════════════════════════════════════════════════

/// Complete robot configuration for kinetic.
#[derive(Debug, Clone)]
pub struct SetupConfig {
    /// Planning groups (name → group).
    pub planning_groups: HashMap<String, PlanningGroupConfig>,
    /// End-effector configurations.
    pub end_effectors: HashMap<String, EndEffectorConfig>,
    /// Named poses (name → joint values).
    pub named_poses: HashMap<String, Vec<f64>>,
    /// Allowed collision matrix entries.
    pub acm_entries: Vec<(String, String)>,
    /// Runtime parameters.
    pub parameters: HashMap<String, ParameterValue>,
}

impl SetupConfig {
    pub fn new() -> Self {
        Self {
            planning_groups: HashMap::new(),
            end_effectors: HashMap::new(),
            named_poses: HashMap::new(),
            acm_entries: Vec::new(),
            parameters: HashMap::new(),
        }
    }
}

/// A planning group configuration.
#[derive(Debug, Clone)]
pub struct PlanningGroupConfig {
    /// Group name.
    pub name: String,
    /// Base link of the kinematic chain.
    pub base_link: String,
    /// Tip link of the kinematic chain.
    pub tip_link: String,
    /// Joint names in the group.
    pub joints: Vec<String>,
    /// Default IK solver for this group.
    pub ik_solver: String,
}

/// End-effector configuration.
#[derive(Debug, Clone)]
pub struct EndEffectorConfig {
    /// End-effector name.
    pub name: String,
    /// Parent planning group.
    pub parent_group: String,
    /// Parent link on the parent group.
    pub parent_link: String,
    /// Component group (gripper joints).
    pub component_group: Option<String>,
}

/// Dynamic parameter value.
#[derive(Debug, Clone)]
pub enum ParameterValue {
    Float(f64),
    Int(i64),
    Bool(bool),
    String(String),
}

// ═══════════════════════════════════════════════════════════════════════════
// Setup Wizard
// ═══════════════════════════════════════════════════════════════════════════

/// Auto-detect planning groups from URDF structure.
///
/// Finds kinematic chains from root to each leaf link that has collision
/// geometry. Each chain becomes a planning group candidate.
pub fn auto_detect_groups(robot: &Robot) -> Vec<PlanningGroupConfig> {
    let mut groups = Vec::new();
    let root_name = &robot.links[robot.root].name;

    // Find leaf links with collision geometry
    for (i, link) in robot.links.iter().enumerate() {
        if link.child_joints.is_empty() && !link.collision_geometry.is_empty() {
            groups.push(PlanningGroupConfig {
                name: format!("{}_group", link.name),
                base_link: root_name.clone(),
                tip_link: link.name.clone(),
                joints: collect_chain_joints(robot, robot.root, i),
                ik_solver: "auto".into(),
            });
        }
    }

    // If no leaf with geometry, find the longest chain
    if groups.is_empty() {
        let (tip_idx, _) = find_deepest_leaf(robot);
        let joints = collect_chain_joints(robot, robot.root, tip_idx);
        if !joints.is_empty() {
            groups.push(PlanningGroupConfig {
                name: "arm".into(),
                base_link: root_name.clone(),
                tip_link: robot.links[tip_idx].name.clone(),
                joints,
                ik_solver: "auto".into(),
            });
        }
    }

    groups
}

/// Auto-generate ACM entries: adjacent links (connected by joints).
pub fn auto_detect_acm(robot: &Robot) -> Vec<(String, String)> {
    let mut acm = Vec::new();
    for joint in &robot.joints {
        let parent_name = &robot.links[joint.parent_link].name;
        let child_name = &robot.links[joint.child_link].name;
        acm.push((parent_name.clone(), child_name.clone()));
    }
    acm
}

/// Auto-detect end-effectors: leaf links or links named *gripper*, *hand*, *tool*.
pub fn auto_detect_end_effectors(robot: &Robot) -> Vec<EndEffectorConfig> {
    let mut ees = Vec::new();

    for link in &robot.links {
        let name_lower = link.name.to_lowercase();
        let is_ee = link.child_joints.is_empty()
            || name_lower.contains("gripper")
            || name_lower.contains("hand")
            || name_lower.contains("tool")
            || name_lower.contains("ee")
            || name_lower.contains("tcp");

        if is_ee && link.parent_joint.is_some() {
            let parent_joint = link.parent_joint.unwrap();
            let parent_link = &robot.links[robot.joints[parent_joint].parent_link].name;

            ees.push(EndEffectorConfig {
                name: format!("{}_ee", link.name),
                parent_group: "arm".into(),
                parent_link: parent_link.clone(),
                component_group: None,
            });
        }
    }

    ees
}

/// Generate a complete SetupConfig from a URDF Robot.
pub fn generate_config(robot: &Robot) -> SetupConfig {
    let mut config = SetupConfig::new();

    // Planning groups
    for group in auto_detect_groups(robot) {
        config.planning_groups.insert(group.name.clone(), group);
    }

    // ACM
    config.acm_entries = auto_detect_acm(robot);

    // End-effectors
    for ee in auto_detect_end_effectors(robot) {
        config.end_effectors.insert(ee.name.clone(), ee);
    }

    // Default named poses
    let dof = robot.dof;
    config.named_poses.insert("zeros".into(), vec![0.0; dof]);

    // Mid-range pose
    let mid: Vec<f64> = robot.joint_limits.iter().map(|l| (l.lower + l.upper) / 2.0).collect();
    config.named_poses.insert("home".into(), mid);

    // Default parameters
    config.parameters.insert("planning_time".into(), ParameterValue::Float(5.0));
    config.parameters.insert("max_velocity_scaling".into(), ParameterValue::Float(1.0));
    config.parameters.insert("max_acceleration_scaling".into(), ParameterValue::Float(1.0));
    config.parameters.insert("collision_padding".into(), ParameterValue::Float(0.01));

    config
}

// ═══════════════════════════════════════════════════════════════════════════
// Config Validation
// ═══════════════════════════════════════════════════════════════════════════

/// A validation issue found in the config.
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    pub severity: Severity,
    pub category: String,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
    Info,
}

/// Validate a robot configuration against the URDF.
pub fn validate_config(robot: &Robot, config: &SetupConfig) -> Vec<ValidationIssue> {
    let mut issues = Vec::new();

    // Check planning groups reference valid links
    for (name, group) in &config.planning_groups {
        if robot.links.iter().all(|l| l.name != group.base_link) {
            issues.push(ValidationIssue {
                severity: Severity::Error,
                category: "planning_group".into(),
                message: format!("Group '{}': base_link '{}' not found in URDF", name, group.base_link),
            });
        }
        if robot.links.iter().all(|l| l.name != group.tip_link) {
            issues.push(ValidationIssue {
                severity: Severity::Error,
                category: "planning_group".into(),
                message: format!("Group '{}': tip_link '{}' not found in URDF", name, group.tip_link),
            });
        }
        for joint_name in &group.joints {
            if robot.joints.iter().all(|j| j.name != *joint_name) {
                issues.push(ValidationIssue {
                    severity: Severity::Error,
                    category: "planning_group".into(),
                    message: format!("Group '{}': joint '{}' not found in URDF", name, joint_name),
                });
            }
        }
    }

    // Check named poses have correct DOF
    for (name, pose) in &config.named_poses {
        if pose.len() != robot.dof {
            issues.push(ValidationIssue {
                severity: Severity::Error,
                category: "named_pose".into(),
                message: format!("Pose '{}': has {} values, expected {} (DOF)", name, pose.len(), robot.dof),
            });
        }
    }

    // Check ACM entries reference valid links
    for (a, b) in &config.acm_entries {
        if robot.links.iter().all(|l| l.name != *a) {
            issues.push(ValidationIssue {
                severity: Severity::Warning,
                category: "acm".into(),
                message: format!("ACM entry: link '{}' not found", a),
            });
        }
        if robot.links.iter().all(|l| l.name != *b) {
            issues.push(ValidationIssue {
                severity: Severity::Warning,
                category: "acm".into(),
                message: format!("ACM entry: link '{}' not found", b),
            });
        }
    }

    // Check for links without collision geometry
    for link in &robot.links {
        if link.collision_geometry.is_empty() && !link.child_joints.is_empty() {
            issues.push(ValidationIssue {
                severity: Severity::Info,
                category: "collision".into(),
                message: format!("Link '{}' has no collision geometry", link.name),
            });
        }
    }

    // Check joints have limits
    for joint in &robot.joints {
        if joint.joint_type != JointType::Fixed && joint.limits.is_none() {
            issues.push(ValidationIssue {
                severity: Severity::Warning,
                category: "joint_limits".into(),
                message: format!("Joint '{}' ({:?}) has no limits", joint.name, joint.joint_type),
            });
        }
    }

    issues
}

// ═══════════════════════════════════════════════════════════════════════════
// Runtime Parameter Reconfigure
// ═══════════════════════════════════════════════════════════════════════════

/// Runtime parameter store with change notification.
pub struct ParameterServer {
    params: HashMap<String, ParameterValue>,
    change_log: Vec<(String, ParameterValue)>,
}

impl ParameterServer {
    pub fn new() -> Self {
        Self { params: HashMap::new(), change_log: Vec::new() }
    }

    pub fn from_config(config: &SetupConfig) -> Self {
        let mut server = Self::new();
        for (k, v) in &config.parameters {
            server.params.insert(k.clone(), v.clone());
        }
        server
    }

    pub fn get(&self, key: &str) -> Option<&ParameterValue> { self.params.get(key) }

    pub fn get_float(&self, key: &str) -> Option<f64> {
        match self.params.get(key) { Some(ParameterValue::Float(v)) => Some(*v), _ => None }
    }

    pub fn set(&mut self, key: &str, value: ParameterValue) {
        self.change_log.push((key.to_string(), value.clone()));
        self.params.insert(key.to_string(), value);
    }

    pub fn changes(&self) -> &[(String, ParameterValue)] { &self.change_log }
    pub fn clear_changes(&mut self) { self.change_log.clear(); }
    pub fn num_params(&self) -> usize { self.params.len() }
}

// ═══════════════════════════════════════════════════════════════════════════
// MoveIt2 SRDF Import
// ═══════════════════════════════════════════════════════════════════════════

/// Import planning groups from MoveIt2 SRDF XML.
///
/// Parses `<group>` elements with `<chain>` or `<joint>` children.
pub fn import_srdf_groups(srdf_xml: &str) -> Vec<PlanningGroupConfig> {
    let mut groups = Vec::new();

    // Simple XML parsing (no dependency required for basic SRDF)
    for line in srdf_xml.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("<group ") && trimmed.contains("name=") {
            if let Some(name) = extract_attr(trimmed, "name") {
                groups.push(PlanningGroupConfig {
                    name,
                    base_link: String::new(),
                    tip_link: String::new(),
                    joints: Vec::new(),
                    ik_solver: "auto".into(),
                });
            }
        } else if trimmed.starts_with("<chain ") {
            if let (Some(base), Some(tip)) = (extract_attr(trimmed, "base_link"), extract_attr(trimmed, "tip_link")) {
                if let Some(group) = groups.last_mut() {
                    group.base_link = base;
                    group.tip_link = tip;
                }
            }
        } else if trimmed.starts_with("<joint ") && trimmed.contains("name=") {
            if let Some(joint_name) = extract_attr(trimmed, "name") {
                if let Some(group) = groups.last_mut() {
                    group.joints.push(joint_name);
                }
            }
        }
    }

    groups
}

/// Import ACM entries from SRDF.
pub fn import_srdf_acm(srdf_xml: &str) -> Vec<(String, String)> {
    let mut acm = Vec::new();

    for line in srdf_xml.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("<disable_collisions ") {
            if let (Some(l1), Some(l2)) = (extract_attr(trimmed, "link1"), extract_attr(trimmed, "link2")) {
                acm.push((l1, l2));
            }
        }
    }

    acm
}

/// Import named poses from SRDF.
pub fn import_srdf_poses(srdf_xml: &str) -> HashMap<String, HashMap<String, f64>> {
    let mut poses: HashMap<String, HashMap<String, f64>> = HashMap::new();
    let mut current_pose: Option<String> = None;

    for line in srdf_xml.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("<group_state ") && trimmed.contains("name=") {
            current_pose = extract_attr(trimmed, "name");
            if let Some(ref name) = current_pose {
                poses.entry(name.clone()).or_default();
            }
        } else if trimmed.starts_with("<joint ") && current_pose.is_some() {
            if let (Some(name), Some(value)) = (extract_attr(trimmed, "name"), extract_attr(trimmed, "value")) {
                if let Ok(v) = value.parse::<f64>() {
                    if let Some(ref pose_name) = current_pose {
                        poses.get_mut(pose_name).map(|p| p.insert(name, v));
                    }
                }
            }
        } else if trimmed.starts_with("</group_state") {
            current_pose = None;
        }
    }

    poses
}

/// Import MoveIt2 joint_limits.yaml format.
pub fn import_joint_limits_yaml(yaml: &str) -> HashMap<String, (f64, f64, f64, f64)> {
    let mut limits = HashMap::new();
    let mut current_joint: Option<String> = None;
    let mut _has_vel = false;
    let mut _has_acc = false;
    let mut vel = 0.0;
    let mut acc = 0.0;
    let mut lower = 0.0;
    let mut upper = 0.0;

    for line in yaml.lines() {
        let trimmed = line.trim();
        if !trimmed.starts_with('#') && trimmed.ends_with(':') && !trimmed.contains(' ') {
            // Save previous joint
            if let Some(ref name) = current_joint {
                limits.insert(name.clone(), (lower, upper, vel, acc));
            }
            current_joint = Some(trimmed.trim_end_matches(':').to_string());
            lower = 0.0; upper = 0.0; vel = 0.0; acc = 0.0;
            _has_vel = false; _has_acc = false;
        } else if let Some(ref _joint) = current_joint {
            if let Some(val) = extract_yaml_value(trimmed, "max_velocity") { vel = val; _has_vel = true; }
            if let Some(val) = extract_yaml_value(trimmed, "max_acceleration") { acc = val; _has_acc = true; }
            if let Some(val) = extract_yaml_value(trimmed, "min_position") { lower = val; }
            if let Some(val) = extract_yaml_value(trimmed, "max_position") { upper = val; }
        }
    }

    if let Some(ref name) = current_joint {
        limits.insert(name.clone(), (lower, upper, vel, acc));
    }

    limits
}

// ═══════════════════════════════════════════════════════════════════════════
// Project Scaffolding
// ═══════════════════════════════════════════════════════════════════════════

/// Generate a kinetic config TOML from auto-detected configuration.
pub fn generate_config_toml(robot: &Robot, config: &SetupConfig) -> String {
    let mut toml = String::new();
    toml.push_str(&format!("# Kinetic configuration for {}\n\n", robot.name));

    // Planning groups
    for (name, group) in &config.planning_groups {
        toml.push_str(&format!("[planning_groups.{}]\n", name));
        toml.push_str(&format!("base_link = \"{}\"\n", group.base_link));
        toml.push_str(&format!("tip_link = \"{}\"\n", group.tip_link));
        toml.push_str(&format!("ik_solver = \"{}\"\n", group.ik_solver));
        toml.push_str(&format!("joints = {:?}\n\n", group.joints));
    }

    // Named poses
    for (name, pose) in &config.named_poses {
        toml.push_str(&format!("[named_poses.{}]\n", name));
        toml.push_str(&format!("values = {:?}\n\n", pose));
    }

    // Parameters
    if !config.parameters.is_empty() {
        toml.push_str("[parameters]\n");
        for (k, v) in &config.parameters {
            match v {
                ParameterValue::Float(f) => toml.push_str(&format!("{} = {}\n", k, f)),
                ParameterValue::Int(i) => toml.push_str(&format!("{} = {}\n", k, i)),
                ParameterValue::Bool(b) => toml.push_str(&format!("{} = {}\n", k, b)),
                ParameterValue::String(s) => toml.push_str(&format!("{} = \"{}\"\n", k, s)),
            }
        }
    }

    toml
}

// ═══════════════════════════════════════════════════════════════════════════
// Helpers
// ═══════════════════════════════════════════════════════════════════════════

fn extract_attr(line: &str, attr: &str) -> Option<String> {
    let pattern = format!("{}=\"", attr);
    let start = line.find(&pattern)? + pattern.len();
    let end = line[start..].find('"')? + start;
    Some(line[start..end].to_string())
}

fn extract_yaml_value(line: &str, key: &str) -> Option<f64> {
    if line.contains(key) {
        let parts: Vec<&str> = line.split(':').collect();
        if parts.len() >= 2 {
            return parts[1].trim().parse().ok();
        }
    }
    None
}

fn collect_chain_joints(robot: &Robot, from: usize, to: usize) -> Vec<String> {
    let mut joints = Vec::new();
    let mut current = to;
    while current != from {
        if let Some(joint_idx) = robot.links[current].parent_joint {
            let joint = &robot.joints[joint_idx];
            if joint.joint_type != JointType::Fixed {
                joints.push(joint.name.clone());
            }
            current = joint.parent_link;
        } else {
            break;
        }
    }
    joints.reverse();
    joints
}

fn find_deepest_leaf(robot: &Robot) -> (usize, usize) {
    let mut best = (0, 0);
    for (i, link) in robot.links.iter().enumerate() {
        if link.child_joints.is_empty() {
            let depth = chain_depth(robot, i);
            if depth > best.1 { best = (i, depth); }
        }
    }
    best
}

fn chain_depth(robot: &Robot, link_idx: usize) -> usize {
    let mut depth = 0;
    let mut current = link_idx;
    while let Some(joint_idx) = robot.links[current].parent_joint {
        depth += 1;
        current = robot.joints[joint_idx].parent_link;
    }
    depth
}

#[cfg(test)]
mod tests {
    use super::*;

    const URDF: &str = r#"<?xml version="1.0"?>
<robot name="test_arm">
  <link name="base_link">
    <collision><geometry><box size="0.2 0.2 0.1"/></geometry></collision>
  </link>
  <link name="link1">
    <collision><geometry><cylinder radius="0.04" length="0.3"/></geometry></collision>
  </link>
  <link name="link2">
    <collision><geometry><sphere radius="0.05"/></geometry></collision>
  </link>
  <link name="ee_link"/>

  <joint name="joint1" type="revolute">
    <parent link="base_link"/><child link="link1"/>
    <origin xyz="0 0 0.05"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="100"/>
  </joint>
  <joint name="joint2" type="revolute">
    <parent link="link1"/><child link="link2"/>
    <origin xyz="0 0 0.3"/><axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="2.0" effort="80"/>
  </joint>
  <joint name="joint3" type="revolute">
    <parent link="link2"/><child link="ee_link"/>
    <origin xyz="0 0 0.05"/><axis xyz="0 1 0"/>
    <limit lower="-2.5" upper="2.5" velocity="3.0" effort="50"/>
  </joint>
</robot>"#;

    #[test]
    fn auto_detect_groups_finds_chain() {
        let robot = Robot::from_urdf_string(URDF).unwrap();
        let groups = auto_detect_groups(&robot);
        assert!(!groups.is_empty(), "Should detect at least one group");
        assert!(!groups[0].joints.is_empty());
    }

    #[test]
    fn auto_detect_acm_finds_adjacent() {
        let robot = Robot::from_urdf_string(URDF).unwrap();
        let acm = auto_detect_acm(&robot);
        assert_eq!(acm.len(), 3, "3 joints = 3 adjacent pairs");
    }

    #[test]
    fn auto_detect_ee() {
        let robot = Robot::from_urdf_string(URDF).unwrap();
        let ees = super::auto_detect_end_effectors(&robot);
        assert!(!ees.is_empty(), "Should detect ee_link");
    }

    #[test]
    fn generate_config_complete() {
        let robot = Robot::from_urdf_string(URDF).unwrap();
        let config = generate_config(&robot);
        assert!(!config.planning_groups.is_empty());
        assert!(!config.acm_entries.is_empty());
        assert!(!config.named_poses.is_empty());
        assert!(config.named_poses.contains_key("home"));
        assert!(config.named_poses.contains_key("zeros"));
    }

    #[test]
    fn validate_config_clean() {
        let robot = Robot::from_urdf_string(URDF).unwrap();
        let config = generate_config(&robot);
        let issues = validate_config(&robot, &config);
        let errors: Vec<_> = issues.iter().filter(|i| i.severity == Severity::Error).collect();
        assert!(errors.is_empty(), "Auto-generated config should have no errors: {:?}", errors);
    }

    #[test]
    fn validate_config_detects_bad_link() {
        let robot = Robot::from_urdf_string(URDF).unwrap();
        let mut config = generate_config(&robot);
        config.planning_groups.insert("bad".into(), PlanningGroupConfig {
            name: "bad".into(),
            base_link: "nonexistent".into(),
            tip_link: "also_bad".into(),
            joints: vec!["fake_joint".into()],
            ik_solver: "auto".into(),
        });

        let issues = validate_config(&robot, &config);
        let errors: Vec<_> = issues.iter().filter(|i| i.severity == Severity::Error).collect();
        assert!(errors.len() >= 3, "Should detect bad links and joints");
    }

    #[test]
    fn validate_config_detects_wrong_dof() {
        let robot = Robot::from_urdf_string(URDF).unwrap();
        let mut config = generate_config(&robot);
        config.named_poses.insert("bad_pose".into(), vec![0.0]); // wrong DOF

        let issues = validate_config(&robot, &config);
        let errors: Vec<_> = issues.iter().filter(|i| i.severity == Severity::Error).collect();
        assert!(!errors.is_empty(), "Should detect wrong DOF in named pose");
    }

    #[test]
    fn parameter_server_crud() {
        let mut server = ParameterServer::new();
        server.set("speed", ParameterValue::Float(0.5));
        assert_eq!(server.get_float("speed"), Some(0.5));
        assert_eq!(server.num_params(), 1);

        server.set("speed", ParameterValue::Float(0.8));
        assert_eq!(server.get_float("speed"), Some(0.8));
        assert_eq!(server.changes().len(), 2);

        server.clear_changes();
        assert_eq!(server.changes().len(), 0);
    }

    #[test]
    fn import_srdf_groups_parses() {
        let srdf = r#"
<robot name="test">
  <group name="arm">
    <chain base_link="base_link" tip_link="ee_link"/>
  </group>
  <group name="gripper">
    <joint name="finger_left"/>
    <joint name="finger_right"/>
  </group>
</robot>"#;

        let groups = import_srdf_groups(srdf);
        assert_eq!(groups.len(), 2);
        assert_eq!(groups[0].name, "arm");
        assert_eq!(groups[0].base_link, "base_link");
        assert_eq!(groups[1].name, "gripper");
        assert_eq!(groups[1].joints.len(), 2);
    }

    #[test]
    fn import_srdf_acm_parses() {
        let srdf = r#"
<robot name="test">
  <disable_collisions link1="base" link2="link1" reason="Adjacent"/>
  <disable_collisions link1="link1" link2="link2" reason="Adjacent"/>
</robot>"#;

        let acm = import_srdf_acm(srdf);
        assert_eq!(acm.len(), 2);
        assert_eq!(acm[0], ("base".into(), "link1".into()));
    }

    #[test]
    fn import_srdf_poses_parses() {
        let srdf = r#"
<robot name="test">
  <group_state name="home" group="arm">
    <joint name="j1" value="0.0"/>
    <joint name="j2" value="1.57"/>
  </group_state>
</robot>"#;

        let poses = import_srdf_poses(srdf);
        assert!(poses.contains_key("home"));
        assert_eq!(poses["home"]["j2"], 1.57);
    }

    #[test]
    fn import_joint_limits_yaml_parses() {
        let yaml = r#"
joint1:
  max_velocity: 2.0
  max_acceleration: 5.0
  min_position: -3.14
  max_position: 3.14
joint2:
  max_velocity: 1.5
  max_acceleration: 3.0
  min_position: -2.0
  max_position: 2.0
"#;

        let limits = import_joint_limits_yaml(yaml);
        assert_eq!(limits.len(), 2);
        assert!((limits["joint1"].2 - 2.0).abs() < 1e-10); // velocity
    }

    #[test]
    fn generate_config_toml_produces_valid() {
        let robot = Robot::from_urdf_string(URDF).unwrap();
        let config = generate_config(&robot);
        let toml = generate_config_toml(&robot, &config);

        assert!(toml.contains("[planning_groups."));
        assert!(toml.contains("[named_poses."));
        assert!(toml.contains("[parameters]"));
        assert!(toml.contains("planning_time"));
    }

    #[test]
    fn full_setup_workflow() {
        let robot = Robot::from_urdf_string(URDF).unwrap();

        // 1. Auto-detect config
        let config = generate_config(&robot);

        // 2. Validate
        let issues = validate_config(&robot, &config);
        let errors: Vec<_> = issues.iter().filter(|i| i.severity == Severity::Error).collect();
        assert!(errors.is_empty());

        // 3. Create parameter server
        let mut params = ParameterServer::from_config(&config);
        assert!(params.num_params() > 0);

        // 4. Runtime reconfigure
        params.set("planning_time", ParameterValue::Float(10.0));
        assert_eq!(params.get_float("planning_time"), Some(10.0));

        // 5. Generate config file
        let toml = generate_config_toml(&robot, &config);
        assert!(!toml.is_empty());
    }
}
