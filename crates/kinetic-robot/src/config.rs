//! TOML-based robot configuration loader.
//!
//! Loads robot configuration from a `kinetic.toml` file in a robot
//! config directory. The TOML defines planning groups, end-effectors,
//! named poses, IK solver preferences, and collision settings.
//!
//! # Config Format
//!
//! ```toml
//! [robot]
//! name = "franka_panda"
//! urdf = "panda.urdf"
//!
//! [planning_group.arm]
//! chain = ["panda_link0", "panda_link8"]
//! joints = ["panda_joint1", "panda_joint2", ...]
//!
//! [end_effector.hand]
//! parent_link = "panda_link8"
//! parent_group = "arm"
//! tcp_xyz = [0.0, 0.0, 0.1034]
//! tcp_rpy = [0.0, 0.0, 0.0]
//!
//! [ik]
//! solver = "dls"
//!
//! [named_poses]
//! home = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]
//! ready = [0.0, -0.3, 0.0, -1.5, 0.0, 1.2, 0.0]
//!
//! [collision]
//! self_collision_pairs = "auto"
//! padding = 0.01
//! ```

use std::collections::HashMap;
use std::path::Path;

use kinetic_core::Pose;
use serde::Deserialize;

use crate::robot::{EndEffector, PlanningGroup, Robot};

/// Top-level TOML config structure.
#[derive(Debug, Deserialize)]
pub struct RobotConfig {
    /// Robot metadata.
    pub robot: RobotSection,
    /// Planning groups (keyed by group name).
    #[serde(default)]
    pub planning_group: HashMap<String, PlanningGroupConfig>,
    /// End-effectors (keyed by EE name).
    #[serde(default)]
    pub end_effector: HashMap<String, EndEffectorConfig>,
    /// IK solver preferences.
    #[serde(default)]
    pub ik: IkConfig,
    /// Named poses (keyed by pose name, value is joint angle array).
    #[serde(default)]
    pub named_poses: HashMap<String, Vec<f64>>,
    /// Collision settings.
    #[serde(default)]
    pub collision: CollisionConfig,
}

/// `[robot]` section.
#[derive(Debug, Deserialize)]
pub struct RobotSection {
    /// Robot name.
    pub name: String,
    /// Path to URDF file (relative to config directory).
    pub urdf: String,
    /// Expected DOF (optional, used for validation).
    pub dof: Option<usize>,
}

/// Planning group config from TOML.
#[derive(Debug, Deserialize)]
pub struct PlanningGroupConfig {
    /// [base_link, tip_link] for chain extraction.
    pub chain: Option<[String; 2]>,
    /// Explicit list of joint names (alternative to chain).
    #[serde(default)]
    pub joints: Vec<String>,
}

/// End-effector config from TOML.
#[derive(Debug, Deserialize)]
pub struct EndEffectorConfig {
    /// Parent link name.
    pub parent_link: String,
    /// Parent planning group name.
    pub parent_group: String,
    /// TCP offset translation [x, y, z].
    #[serde(default)]
    pub tcp_xyz: [f64; 3],
    /// TCP offset rotation (roll, pitch, yaw) in radians.
    #[serde(default)]
    pub tcp_rpy: [f64; 3],
}

/// IK solver configuration.
#[derive(Debug, Deserialize)]
pub struct IkConfig {
    /// Preferred IK solver: "dls", "fabrik", "opw", "auto".
    #[serde(default = "default_ik_solver")]
    pub solver: String,
    /// Maximum iterations (for iterative solvers).
    #[serde(default = "default_ik_max_iter")]
    pub max_iterations: usize,
    /// Position tolerance in meters.
    #[serde(default = "default_ik_tolerance")]
    pub tolerance: f64,
}

impl Default for IkConfig {
    fn default() -> Self {
        Self {
            solver: "auto".to_string(),
            max_iterations: 100,
            tolerance: 1e-4,
        }
    }
}

fn default_ik_solver() -> String {
    "auto".to_string()
}

fn default_ik_max_iter() -> usize {
    100
}

fn default_ik_tolerance() -> f64 {
    1e-4
}

/// Collision configuration.
#[derive(Debug, Deserialize)]
pub struct CollisionConfig {
    /// How to compute self-collision pairs: "auto" or "manual".
    #[serde(default = "default_self_collision")]
    pub self_collision_pairs: String,
    /// Collision padding in meters.
    #[serde(default = "default_padding")]
    pub padding: f64,
    /// Manual skip pairs (only used if self_collision_pairs = "manual").
    #[serde(default)]
    pub skip_pairs: Vec<[String; 2]>,
}

impl Default for CollisionConfig {
    fn default() -> Self {
        Self {
            self_collision_pairs: "auto".to_string(),
            padding: 0.01,
            skip_pairs: Vec::new(),
        }
    }
}

fn default_self_collision() -> String {
    "auto".to_string()
}

fn default_padding() -> f64 {
    0.01
}

/// Parsed IK preference stored on Robot.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct IkPreference {
    /// Preferred solver name.
    pub solver: String,
    /// Max iterations.
    pub max_iterations: usize,
    /// Position tolerance.
    pub tolerance: f64,
}

impl From<IkConfig> for IkPreference {
    fn from(c: IkConfig) -> Self {
        Self {
            solver: c.solver,
            max_iterations: c.max_iterations,
            tolerance: c.tolerance,
        }
    }
}

/// Parsed collision config stored on Robot.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CollisionPreference {
    /// Padding in meters.
    pub padding: f64,
    /// Whether to auto-compute self-collision pairs.
    pub auto_self_collision: bool,
    /// Manual skip pairs (link name pairs).
    pub skip_pairs: Vec<(String, String)>,
}

impl From<CollisionConfig> for CollisionPreference {
    fn from(c: CollisionConfig) -> Self {
        Self {
            padding: c.padding,
            auto_self_collision: c.self_collision_pairs == "auto",
            skip_pairs: c.skip_pairs.into_iter().map(|[a, b]| (a, b)).collect(),
        }
    }
}

impl RobotConfig {
    /// Load config from a TOML file.
    pub fn from_file(path: impl AsRef<Path>) -> kinetic_core::Result<Self> {
        let content = std::fs::read_to_string(path.as_ref()).map_err(|e| {
            kinetic_core::KineticError::RobotConfigNotFound(format!(
                "Cannot read config file {}: {}",
                path.as_ref().display(),
                e
            ))
        })?;
        Self::parse(&content)
    }

    /// Parse config from a TOML string.
    pub fn parse(toml_str: &str) -> kinetic_core::Result<Self> {
        toml::from_str(toml_str).map_err(|e| {
            kinetic_core::KineticError::RobotConfigNotFound(format!("Invalid TOML config: {}", e))
        })
    }
}

impl Robot {
    /// Load robot from a config directory containing `kinetic.toml` and a URDF file.
    ///
    /// The directory should contain:
    /// - `kinetic.toml` — robot configuration
    /// - `<urdf_file>` — URDF file referenced in the config
    pub fn from_config(config_dir: impl AsRef<Path>) -> kinetic_core::Result<Self> {
        let config_dir = config_dir.as_ref();
        let toml_path = config_dir.join("kinetic.toml");
        let config = RobotConfig::from_file(&toml_path)?;

        let urdf_path = config_dir.join(&config.robot.urdf);
        let mut robot = Robot::from_urdf(&urdf_path)?;

        // Override name from config
        robot.name = config.robot.name.clone();

        // Validate DOF if specified
        if let Some(expected_dof) = config.robot.dof {
            if robot.dof != expected_dof {
                return Err(kinetic_core::KineticError::RobotConfigNotFound(format!(
                    "DOF mismatch: config says {} but URDF has {}",
                    expected_dof, robot.dof
                )));
            }
        }

        // Apply config to robot
        apply_config(&mut robot, config)?;

        Ok(robot)
    }

    /// Load robot from a config TOML string and URDF string.
    ///
    /// Useful for embedded/bundled configs without file I/O.
    pub fn from_config_strings(toml_str: &str, urdf_str: &str) -> kinetic_core::Result<Self> {
        let config = RobotConfig::parse(toml_str)?;
        let mut robot = Robot::from_urdf_string(urdf_str)?;

        robot.name = config.robot.name.clone();

        if let Some(expected_dof) = config.robot.dof {
            if robot.dof != expected_dof {
                return Err(kinetic_core::KineticError::RobotConfigNotFound(format!(
                    "DOF mismatch: config says {} but URDF has {}",
                    expected_dof, robot.dof
                )));
            }
        }

        apply_config(&mut robot, config)?;
        Ok(robot)
    }
}

/// Apply a parsed TOML config to a Robot.
fn apply_config(robot: &mut Robot, config: RobotConfig) -> kinetic_core::Result<()> {
    // Apply planning groups
    for (name, group_cfg) in config.planning_group {
        let joint_indices = if !group_cfg.joints.is_empty() {
            // Explicit joint list
            group_cfg
                .joints
                .iter()
                .map(|jn| robot.joint_index(jn))
                .collect::<Result<Vec<_>, _>>()?
        } else if let Some([base, tip]) = &group_cfg.chain {
            // Chain-based: get joints between base and tip
            robot.chain(base, tip)?
        } else {
            return Err(kinetic_core::KineticError::RobotConfigNotFound(format!(
                "Planning group '{}' must specify either 'joints' or 'chain'",
                name
            )));
        };

        let (base_link, tip_link) = if let Some([base, tip]) = group_cfg.chain {
            (base, tip)
        } else {
            // Derive from joint chain
            let first_joint = joint_indices.first().copied().unwrap_or(0);
            let last_joint = joint_indices.last().copied().unwrap_or(0);
            let base = robot.joints[first_joint].parent_link;
            let tip = robot.joints[last_joint].child_link;
            (
                robot.links[base].name.clone(),
                robot.links[tip].name.clone(),
            )
        };

        robot.groups.insert(
            name.clone(),
            PlanningGroup {
                name,
                joint_indices,
                base_link,
                tip_link,
            },
        );
    }

    // Apply end-effectors
    for (name, ee_cfg) in config.end_effector {
        let grasp_frame = Pose::from_xyz_rpy(
            ee_cfg.tcp_xyz[0],
            ee_cfg.tcp_xyz[1],
            ee_cfg.tcp_xyz[2],
            ee_cfg.tcp_rpy[0],
            ee_cfg.tcp_rpy[1],
            ee_cfg.tcp_rpy[2],
        );

        robot.end_effectors.insert(
            name.clone(),
            EndEffector {
                name,
                parent_link: ee_cfg.parent_link,
                parent_group: ee_cfg.parent_group,
                grasp_frame,
            },
        );
    }

    // Apply named poses
    for (name, values) in config.named_poses {
        robot.named_poses.insert(name, values);
    }

    // Store IK and collision preferences
    robot.ik_preference = Some(config.ik.into());
    robot.collision_preference = Some(config.collision.into());

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_TOML: &str = r#"
[robot]
name = "test_robot"
urdf = "test.urdf"
dof = 3

[planning_group.arm]
chain = ["base_link", "ee_link"]

[end_effector.gripper]
parent_link = "ee_link"
parent_group = "arm"
tcp_xyz = [0.0, 0.0, 0.1]
tcp_rpy = [0.0, 0.0, 0.0]

[ik]
solver = "dls"
max_iterations = 200
tolerance = 0.001

[named_poses]
home = [0.0, 0.0, 0.0]
ready = [0.5, -0.3, 0.8]

[collision]
self_collision_pairs = "auto"
padding = 0.02
"#;

    const TEST_URDF: &str = r#"<?xml version="1.0"?>
<robot name="test_3dof">
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
</robot>
"#;

    #[test]
    fn parse_toml_config() {
        let config = RobotConfig::parse(TEST_TOML).unwrap();
        assert_eq!(config.robot.name, "test_robot");
        assert_eq!(config.robot.urdf, "test.urdf");
        assert_eq!(config.robot.dof, Some(3));
        assert!(config.planning_group.contains_key("arm"));
        assert!(config.end_effector.contains_key("gripper"));
        assert_eq!(config.ik.solver, "dls");
        assert_eq!(config.named_poses.len(), 2);
        assert!((config.collision.padding - 0.02).abs() < 1e-10);
    }

    #[test]
    fn from_config_strings() {
        let robot = Robot::from_config_strings(TEST_TOML, TEST_URDF).unwrap();
        assert_eq!(robot.name, "test_robot");
        assert_eq!(robot.dof, 3);

        // Check planning group
        assert!(robot.groups.contains_key("arm"));
        let arm = &robot.groups["arm"];
        assert_eq!(arm.joint_indices.len(), 3);
        assert_eq!(arm.base_link, "base_link");
        assert_eq!(arm.tip_link, "ee_link");

        // Check end-effector
        assert!(robot.end_effectors.contains_key("gripper"));
        let ee = &robot.end_effectors["gripper"];
        assert_eq!(ee.parent_link, "ee_link");
        let t = ee.grasp_frame.translation();
        assert!((t.z - 0.1).abs() < 1e-10);

        // Check named poses
        let home = robot.named_pose("home").unwrap();
        assert_eq!(home.len(), 3);
        assert!((home[0] - 0.0).abs() < 1e-10);

        let ready = robot.named_pose("ready").unwrap();
        assert!((ready[0] - 0.5).abs() < 1e-10);

        // Check IK preference
        let ik = robot.ik_preference.as_ref().unwrap();
        assert_eq!(ik.solver, "dls");
        assert_eq!(ik.max_iterations, 200);

        // Check collision preference
        let coll = robot.collision_preference.as_ref().unwrap();
        assert!(coll.auto_self_collision);
        assert!((coll.padding - 0.02).abs() < 1e-10);
    }

    #[test]
    fn dof_mismatch_error() {
        let bad_toml = r#"
[robot]
name = "test"
urdf = "test.urdf"
dof = 5
"#;
        let result = Robot::from_config_strings(bad_toml, TEST_URDF);
        assert!(result.is_err());
    }

    #[test]
    fn minimal_config() {
        let minimal = r#"
[robot]
name = "minimal"
urdf = "test.urdf"
"#;
        let robot = Robot::from_config_strings(minimal, TEST_URDF).unwrap();
        assert_eq!(robot.name, "minimal");
        assert_eq!(robot.dof, 3);
        assert!(robot.groups.is_empty());
        assert!(robot.end_effectors.is_empty());
    }

    #[test]
    fn explicit_joints_group() {
        let toml = r#"
[robot]
name = "explicit"
urdf = "test.urdf"

[planning_group.partial]
joints = ["joint2", "joint3"]
"#;
        let robot = Robot::from_config_strings(toml, TEST_URDF).unwrap();
        let group = &robot.groups["partial"];
        assert_eq!(group.joint_indices.len(), 2);
    }

    #[test]
    fn invalid_joint_name_error() {
        let toml = r#"
[robot]
name = "bad"
urdf = "test.urdf"

[planning_group.arm]
joints = ["nonexistent_joint"]
"#;
        let result = Robot::from_config_strings(toml, TEST_URDF);
        assert!(result.is_err());
    }

    #[test]
    fn ik_preference_serde_roundtrip() {
        let pref = IkPreference {
            solver: "dls".to_string(),
            max_iterations: 200,
            tolerance: 1e-4,
        };

        let json = serde_json::to_string(&pref).unwrap();
        let deser: IkPreference = serde_json::from_str(&json).unwrap();
        assert_eq!(deser.solver, "dls");
        assert_eq!(deser.max_iterations, 200);
        assert!((deser.tolerance - 1e-4).abs() < 1e-15);
    }

    #[test]
    fn collision_preference_serde_roundtrip() {
        let pref = CollisionPreference {
            padding: 0.02,
            auto_self_collision: true,
            skip_pairs: vec![
                ("base_link".to_string(), "link2".to_string()),
                ("link1".to_string(), "ee_link".to_string()),
            ],
        };

        let json = serde_json::to_string(&pref).unwrap();
        let deser: CollisionPreference = serde_json::from_str(&json).unwrap();
        assert!((deser.padding - 0.02).abs() < 1e-10);
        assert!(deser.auto_self_collision);
        assert_eq!(deser.skip_pairs.len(), 2);
        assert_eq!(
            deser.skip_pairs[0],
            ("base_link".to_string(), "link2".to_string())
        );
    }

    #[test]
    fn collision_manual_skip_pairs() {
        let toml = r#"
[robot]
name = "manual"
urdf = "test.urdf"

[collision]
self_collision_pairs = "manual"
padding = 0.03
skip_pairs = [["base_link", "link2"], ["link1", "ee_link"]]
"#;
        let robot = Robot::from_config_strings(toml, TEST_URDF).unwrap();
        let coll = robot.collision_preference.as_ref().unwrap();
        assert!(!coll.auto_self_collision);
        assert_eq!(coll.skip_pairs.len(), 2);
        assert_eq!(
            coll.skip_pairs[0],
            ("base_link".to_string(), "link2".to_string())
        );
    }
}
