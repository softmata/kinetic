//! Robot model — the central struct for kinematic/planning operations.

use std::collections::HashMap;
use std::path::Path;

use kinetic_core::{JointValues, KineticError, Pose};
use serde::{Deserialize, Serialize};

use crate::config::{CollisionPreference, IkPreference};
use crate::joint::{Joint, JointLimits};
use crate::link::Link;
use crate::urdf_loader;

/// A planning group defines a subset of joints for planning.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlanningGroup {
    /// Group name (e.g., "arm", "gripper").
    pub name: String,
    /// Indices into `Robot::active_joints` for this group.
    pub joint_indices: Vec<usize>,
    /// Name of the base link for this group.
    pub base_link: String,
    /// Name of the tip link for this group.
    pub tip_link: String,
}

/// End-effector definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EndEffector {
    /// End-effector name.
    pub name: String,
    /// Parent link name.
    pub parent_link: String,
    /// Parent planning group name.
    pub parent_group: String,
    /// Grasp frame offset from parent link.
    pub grasp_frame: Pose,
}

/// A loaded robot model with kinematic tree and configuration.
///
/// Created from URDF, optionally augmented with a TOML config that
/// defines planning groups, end-effectors, and named poses.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Robot {
    /// Robot name from URDF.
    pub name: String,
    /// All joints (including fixed).
    pub joints: Vec<Joint>,
    /// All links.
    pub links: Vec<Link>,
    /// Degrees of freedom (number of active joints).
    pub dof: usize,
    /// Root link index.
    pub root: usize,
    /// Indices of active (non-fixed) joints in order.
    pub active_joints: Vec<usize>,
    /// Planning groups (from config).
    pub groups: HashMap<String, PlanningGroup>,
    /// End-effector definitions (from config).
    pub end_effectors: HashMap<String, EndEffector>,
    /// Named poses (from config), e.g., "home" → joint values.
    pub named_poses: HashMap<String, Vec<f64>>,
    /// Joint limits for active joints (same order as active_joints).
    pub joint_limits: Vec<JointLimits>,
    /// IK solver preference (from config TOML).
    #[serde(default)]
    pub ik_preference: Option<IkPreference>,
    /// Collision preference (from config TOML).
    #[serde(default)]
    pub collision_preference: Option<CollisionPreference>,
}

impl Robot {
    /// Load from a URDF file path.
    /// Load a robot from any supported file format, auto-detected by extension.
    ///
    /// Supported extensions:
    /// - `.urdf` → URDF
    /// - `.mjcf`, `.xml` → MJCF (MuJoCo XML)
    /// - `.sdf` → SDF (SDFormat/Gazebo)
    pub fn from_file(path: impl AsRef<Path>) -> kinetic_core::Result<Self> {
        let path = path.as_ref();
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        match ext.as_str() {
            "urdf" => Self::from_urdf(path),
            "mjcf" => Self::from_mjcf(path),
            "xml" => {
                // Try MJCF first (more common for .xml in robotics), fall back to SDF
                Self::from_mjcf(path).or_else(|_| Self::from_sdf(path))
            }
            "sdf" => Self::from_sdf(path),
            _ => Err(KineticError::UrdfParse(format!(
                "Unknown robot file extension '{}'. Use .urdf, .mjcf, .xml, or .sdf",
                ext
            ))),
        }
    }

    /// Load from a URDF file.
    pub fn from_urdf(path: impl AsRef<Path>) -> kinetic_core::Result<Self> {
        urdf_loader::load_urdf(path)
    }

    /// Load from a URDF XML string.
    pub fn from_urdf_string(xml: &str) -> kinetic_core::Result<Self> {
        urdf_loader::load_urdf_string(xml)
    }

    /// Load from an MJCF (MuJoCo XML) file.
    pub fn from_mjcf(path: impl AsRef<Path>) -> kinetic_core::Result<Self> {
        crate::mjcf_loader::load_mjcf(path)
    }

    /// Load from an MJCF XML string.
    pub fn from_mjcf_string(xml: &str) -> kinetic_core::Result<Self> {
        crate::mjcf_loader::load_mjcf_string(xml)
    }

    /// Load from an SDF (SDFormat/Gazebo) file.
    pub fn from_sdf(path: impl AsRef<Path>) -> kinetic_core::Result<Self> {
        crate::sdf_loader::load_sdf(path)
    }

    /// Load from an SDF XML string.
    pub fn from_sdf_string(xml: &str) -> kinetic_core::Result<Self> {
        crate::sdf_loader::load_sdf_string(xml)
    }

    /// Load from URDF + SRDF file paths.
    ///
    /// Loads the URDF first, then applies the SRDF configuration
    /// (planning groups, end-effectors, disabled collisions, named states).
    pub fn from_urdf_srdf(
        urdf_path: impl AsRef<Path>,
        srdf_path: impl AsRef<Path>,
    ) -> kinetic_core::Result<Self> {
        let mut robot = Self::from_urdf(urdf_path)?;
        let srdf = crate::srdf::SrdfModel::from_file(srdf_path)?;
        srdf.apply_to_robot(&mut robot)?;
        Ok(robot)
    }

    /// Load a built-in robot by name from the `robot_configs/` directory.
    ///
    /// Searches for a directory matching `name` within the project's
    /// `robot_configs/` folder (resolved relative to `CARGO_MANIFEST_DIR`
    /// at compile time, or the current directory at runtime).
    ///
    /// # Example
    ///
    /// ```ignore
    /// let robot = Robot::from_name("franka_panda")?;
    /// assert_eq!(robot.dof, 7);
    /// ```
    pub fn from_name(name: &str) -> kinetic_core::Result<Self> {
        // Try several search paths for robot_configs/
        let candidates = [
            // Relative to workspace root (typical dev usage)
            Path::new("robot_configs").join(name),
            // Relative to kinetic/ subdirectory
            Path::new("kinetic/robot_configs").join(name),
            // Absolute path via CARGO_MANIFEST_DIR (for tests)
            Path::new(env!("CARGO_MANIFEST_DIR"))
                .parent()
                .and_then(|p| p.parent())
                .map(|ws| ws.join("robot_configs").join(name))
                .unwrap_or_default(),
        ];

        for candidate in &candidates {
            if candidate.exists() && candidate.join("kinetic.toml").exists() {
                return Self::from_config(candidate);
            }
        }

        Err(KineticError::RobotConfigNotFound(format!(
            "Robot config '{}' not found. Searched: {:?}",
            name,
            candidates
                .iter()
                .map(|p| p.display().to_string())
                .collect::<Vec<_>>()
        )))
    }

    /// Get the joint chain between two links (as joint indices).
    ///
    /// Returns the sequence of joint indices traversed from `start_link` to
    /// `end_link` by walking up from each link to the root and finding the
    /// common ancestor.
    pub fn chain(&self, start_link: &str, end_link: &str) -> kinetic_core::Result<Vec<usize>> {
        let start_idx = self.link_index(start_link)?;
        let end_idx = self.link_index(end_link)?;

        // Build path from each link to root
        let path_to_root_start = self.path_to_root(start_idx);
        let path_to_root_end = self.path_to_root(end_idx);

        // Find common ancestor
        let start_set: std::collections::HashSet<usize> =
            path_to_root_start.iter().copied().collect();

        let mut common_ancestor = self.root;
        for &link_idx in &path_to_root_end {
            if start_set.contains(&link_idx) {
                common_ancestor = link_idx;
                break;
            }
        }

        // Collect joints from start to common ancestor
        let mut chain = Vec::new();
        let mut current = start_idx;
        while current != common_ancestor {
            if let Some(joint_idx) = self.links[current].parent_joint {
                chain.push(joint_idx);
                current = self.joints[joint_idx].parent_link;
            } else {
                break;
            }
        }

        // Collect joints from common ancestor to end (reversed)
        let mut end_chain = Vec::new();
        let mut current = end_idx;
        while current != common_ancestor {
            if let Some(joint_idx) = self.links[current].parent_joint {
                end_chain.push(joint_idx);
                current = self.joints[joint_idx].parent_link;
            } else {
                break;
            }
        }
        end_chain.reverse();

        chain.extend(end_chain);
        Ok(chain)
    }

    /// Get velocity limits for all active joints.
    pub fn velocity_limits(&self) -> Vec<f64> {
        self.joint_limits.iter().map(|l| l.velocity).collect()
    }

    /// Get acceleration limits for all active joints.
    ///
    /// Returns the URDF acceleration if available, otherwise defaults to 5.0 rad/s^2.
    pub fn acceleration_limits(&self) -> Vec<f64> {
        self.joint_limits
            .iter()
            .map(|l| l.acceleration.unwrap_or(5.0))
            .collect()
    }

    /// Clamp joint values to their position limits.
    pub fn clamp_to_limits(&self, joints: &mut JointValues) {
        for (i, limit) in self.joint_limits.iter().enumerate() {
            if i < joints.len() {
                joints[i] = limit.clamp(joints[i]);
            }
        }
    }

    /// Check if joint values are within limits.
    pub fn check_limits(&self, joints: &JointValues) -> kinetic_core::Result<()> {
        for (i, limit) in self.joint_limits.iter().enumerate() {
            if i >= joints.len() {
                break;
            }
            let value = joints[i];
            if !limit.in_range(value) {
                let active_joint_idx = self.active_joints[i];
                let name = self.joints[active_joint_idx].name.clone();
                return Err(KineticError::JointLimitViolation {
                    name,
                    value,
                    min: limit.lower,
                    max: limit.upper,
                });
            }
        }
        Ok(())
    }

    /// Look up a named pose.
    pub fn named_pose(&self, name: &str) -> Option<JointValues> {
        self.named_poses
            .get(name)
            .map(|v| JointValues::new(v.clone()))
    }

    /// Get link index by name.
    pub fn link_index(&self, name: &str) -> kinetic_core::Result<usize> {
        self.links
            .iter()
            .position(|l| l.name == name)
            .ok_or_else(|| KineticError::LinkNotFound(name.to_string()))
    }

    /// Get joint index by name.
    pub fn joint_index(&self, name: &str) -> kinetic_core::Result<usize> {
        self.joints
            .iter()
            .position(|j| j.name == name)
            .ok_or_else(|| KineticError::JointNotFound(name.to_string()))
    }

    /// Get the path from a link to the root (as link indices).
    fn path_to_root(&self, start: usize) -> Vec<usize> {
        let mut path = vec![start];
        let mut current = start;
        while let Some(joint_idx) = self.links[current].parent_joint {
            current = self.joints[joint_idx].parent_link;
            path.push(current);
        }
        path
    }

    /// Get the zero configuration (all joints at 0).
    pub fn zero_configuration(&self) -> JointValues {
        JointValues::zeros(self.dof)
    }

    /// Get the midpoint configuration (all joints at center of limits).
    pub fn mid_configuration(&self) -> JointValues {
        let values: Vec<f64> = self
            .joint_limits
            .iter()
            .map(|l| (l.lower + l.upper) / 2.0)
            .collect();
        JointValues::new(values)
    }
}

#[cfg(test)]
#[allow(clippy::approx_constant)]
mod tests {
    use super::*;

    const THREE_DOF_URDF: &str = r#"<?xml version="1.0"?>
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

    fn test_robot() -> Robot {
        Robot::from_urdf_string(THREE_DOF_URDF).unwrap()
    }

    #[test]
    fn chain_base_to_ee() {
        let robot = test_robot();
        let chain = robot.chain("base_link", "ee_link").unwrap();
        assert_eq!(chain.len(), 3);
        assert_eq!(chain, vec![0, 1, 2]);
    }

    #[test]
    fn chain_link1_to_ee() {
        let robot = test_robot();
        let chain = robot.chain("link1", "ee_link").unwrap();
        assert_eq!(chain.len(), 2);
        assert_eq!(chain, vec![1, 2]);
    }

    #[test]
    fn velocity_limits() {
        let robot = test_robot();
        let vel = robot.velocity_limits();
        assert_eq!(vel.len(), 3);
        assert!((vel[0] - 2.0).abs() < 1e-10);
        assert!((vel[2] - 3.0).abs() < 1e-10);
    }

    #[test]
    fn acceleration_limits_default() {
        let robot = test_robot();
        let acc = robot.acceleration_limits();
        assert_eq!(acc.len(), 3);
        // No acceleration in URDF → default 5.0
        assert!((acc[0] - 5.0).abs() < 1e-10);
    }

    #[test]
    fn clamp_to_limits() {
        let robot = test_robot();
        let mut joints = JointValues::new(vec![5.0, -3.0, 0.0]);
        robot.clamp_to_limits(&mut joints);
        assert!((joints[0] - 3.14).abs() < 1e-6);
        assert!((joints[1] - (-2.0)).abs() < 1e-6);
        assert!((joints[2] - 0.0).abs() < 1e-10);
    }

    #[test]
    fn check_limits_ok() {
        let robot = test_robot();
        let joints = JointValues::new(vec![0.0, 0.0, 0.0]);
        assert!(robot.check_limits(&joints).is_ok());
    }

    #[test]
    fn check_limits_violated() {
        let robot = test_robot();
        let joints = JointValues::new(vec![0.0, 5.0, 0.0]); // joint2 out of range
        let err = robot.check_limits(&joints).unwrap_err();
        match err {
            KineticError::JointLimitViolation { name, value, .. } => {
                assert_eq!(name, "joint2");
                assert!((value - 5.0).abs() < 1e-10);
            }
            _ => panic!("Expected JointLimitViolation"),
        }
    }

    #[test]
    fn zero_and_mid_config() {
        let robot = test_robot();
        let zero = robot.zero_configuration();
        assert_eq!(zero.len(), 3);
        assert!((zero[0] - 0.0).abs() < 1e-10);

        let mid = robot.mid_configuration();
        assert_eq!(mid.len(), 3);
        // joint1: (-3.14 + 3.14) / 2 = 0.0
        assert!((mid[0] - 0.0).abs() < 1e-6);
        // joint2: (-2.0 + 2.0) / 2 = 0.0
        assert!((mid[1] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn link_index_lookup() {
        let robot = test_robot();
        assert_eq!(robot.link_index("base_link").unwrap(), 0);
        assert_eq!(robot.link_index("ee_link").unwrap(), 3);
        assert!(robot.link_index("nonexistent").is_err());
    }

    #[test]
    fn joint_index_lookup() {
        let robot = test_robot();
        assert_eq!(robot.joint_index("joint1").unwrap(), 0);
        assert_eq!(robot.joint_index("joint3").unwrap(), 2);
        assert!(robot.joint_index("nonexistent").is_err());
    }

    #[test]
    fn from_name_franka_panda() {
        let robot = Robot::from_name("franka_panda").unwrap();
        assert_eq!(robot.name, "franka_panda");
        assert_eq!(robot.dof, 7);
        let home = robot.named_pose("home").unwrap();
        assert_eq!(home.len(), 7);
        robot.check_limits(&home).unwrap();
    }

    #[test]
    fn from_name_ur5e() {
        let robot = Robot::from_name("ur5e").unwrap();
        assert_eq!(robot.name, "ur5e");
        assert_eq!(robot.dof, 6);
        let home = robot.named_pose("home").unwrap();
        assert_eq!(home.len(), 6);
        robot.check_limits(&home).unwrap();
    }

    #[test]
    fn from_name_ur10e() {
        let robot = Robot::from_name("ur10e").unwrap();
        assert_eq!(robot.name, "ur10e");
        assert_eq!(robot.dof, 6);
    }

    #[test]
    fn from_name_kuka_iiwa7() {
        let robot = Robot::from_name("kuka_iiwa7").unwrap();
        assert_eq!(robot.name, "kuka_iiwa7");
        assert_eq!(robot.dof, 7);
    }

    #[test]
    fn from_name_xarm6() {
        let robot = Robot::from_name("xarm6").unwrap();
        assert_eq!(robot.name, "xarm6");
        assert_eq!(robot.dof, 6);
    }

    // ── New robot configs (25 additional robots) ──

    #[test]
    fn from_name_ur3e() {
        let robot = Robot::from_name("ur3e").unwrap();
        assert_eq!(robot.name, "ur3e");
        assert_eq!(robot.dof, 6);
    }

    #[test]
    fn from_name_ur16e() {
        let robot = Robot::from_name("ur16e").unwrap();
        assert_eq!(robot.name, "ur16e");
        assert_eq!(robot.dof, 6);
    }

    #[test]
    fn from_name_ur20() {
        let robot = Robot::from_name("ur20").unwrap();
        assert_eq!(robot.name, "ur20");
        assert_eq!(robot.dof, 6);
    }

    #[test]
    fn from_name_ur30() {
        let robot = Robot::from_name("ur30").unwrap();
        assert_eq!(robot.name, "ur30");
        assert_eq!(robot.dof, 6);
    }

    #[test]
    fn from_name_kuka_iiwa14() {
        let robot = Robot::from_name("kuka_iiwa14").unwrap();
        assert_eq!(robot.name, "kuka_iiwa14");
        assert_eq!(robot.dof, 7);
    }

    #[test]
    fn from_name_kuka_kr6() {
        let robot = Robot::from_name("kuka_kr6").unwrap();
        assert_eq!(robot.name, "kuka_kr6");
        assert_eq!(robot.dof, 6);
    }

    #[test]
    fn from_name_abb_irb1200() {
        let robot = Robot::from_name("abb_irb1200").unwrap();
        assert_eq!(robot.name, "abb_irb1200");
        assert_eq!(robot.dof, 6);
    }

    #[test]
    fn from_name_abb_irb4600() {
        let robot = Robot::from_name("abb_irb4600").unwrap();
        assert_eq!(robot.name, "abb_irb4600");
        assert_eq!(robot.dof, 6);
    }

    #[test]
    fn from_name_fanuc_crx10ia() {
        let robot = Robot::from_name("fanuc_crx10ia").unwrap();
        assert_eq!(robot.name, "fanuc_crx10ia");
        assert_eq!(robot.dof, 6);
    }

    #[test]
    fn from_name_yaskawa_gp7() {
        let robot = Robot::from_name("yaskawa_gp7").unwrap();
        assert_eq!(robot.name, "yaskawa_gp7");
        assert_eq!(robot.dof, 6);
    }

    #[test]
    fn from_name_staubli_tx260() {
        let robot = Robot::from_name("staubli_tx260").unwrap();
        assert_eq!(robot.name, "staubli_tx260");
        assert_eq!(robot.dof, 6);
    }

    #[test]
    fn from_name_xarm5() {
        let robot = Robot::from_name("xarm5").unwrap();
        assert_eq!(robot.name, "xarm5");
        assert_eq!(robot.dof, 5);
    }

    #[test]
    fn from_name_xarm7() {
        let robot = Robot::from_name("xarm7").unwrap();
        assert_eq!(robot.name, "xarm7");
        assert_eq!(robot.dof, 7);
    }

    #[test]
    fn from_name_kinova_gen3() {
        let robot = Robot::from_name("kinova_gen3").unwrap();
        assert_eq!(robot.name, "kinova_gen3");
        assert_eq!(robot.dof, 7);
    }

    #[test]
    fn from_name_kinova_gen3_lite() {
        let robot = Robot::from_name("kinova_gen3_lite").unwrap();
        assert_eq!(robot.name, "kinova_gen3_lite");
        assert_eq!(robot.dof, 6);
    }

    #[test]
    fn from_name_dobot_cr5() {
        let robot = Robot::from_name("dobot_cr5").unwrap();
        assert_eq!(robot.name, "dobot_cr5");
        assert_eq!(robot.dof, 6);
    }

    #[test]
    fn from_name_flexiv_rizon4() {
        let robot = Robot::from_name("flexiv_rizon4").unwrap();
        assert_eq!(robot.name, "flexiv_rizon4");
        assert_eq!(robot.dof, 7);
    }

    #[test]
    fn from_name_meca500() {
        let robot = Robot::from_name("meca500").unwrap();
        assert_eq!(robot.name, "meca500");
        assert_eq!(robot.dof, 6);
    }

    #[test]
    fn from_name_mycobot_280() {
        let robot = Robot::from_name("mycobot_280").unwrap();
        assert_eq!(robot.name, "mycobot_280");
        assert_eq!(robot.dof, 6);
    }

    #[test]
    fn from_name_viperx_300() {
        let robot = Robot::from_name("viperx_300").unwrap();
        assert_eq!(robot.name, "viperx_300");
        assert_eq!(robot.dof, 6);
    }

    #[test]
    fn from_name_sawyer() {
        let robot = Robot::from_name("sawyer").unwrap();
        assert_eq!(robot.name, "sawyer");
        assert_eq!(robot.dof, 7);
    }

    #[test]
    fn from_name_so_arm100() {
        let robot = Robot::from_name("so_arm100").unwrap();
        assert_eq!(robot.name, "so_arm100");
    }

    #[test]
    fn from_name_koch_v1() {
        let robot = Robot::from_name("koch_v1").unwrap();
        assert_eq!(robot.name, "koch_v1");
    }

    #[test]
    fn from_name_open_manipulator_x() {
        let robot = Robot::from_name("open_manipulator_x").unwrap();
        assert_eq!(robot.name, "open_manipulator_x");
    }

    #[test]
    fn from_name_widowx_250() {
        let robot = Robot::from_name("widowx_250").unwrap();
        assert_eq!(robot.name, "widowx_250");
    }

    // ── 22 additional robot configs (Phase 3) ──

    #[test]
    fn from_name_fetch() {
        let robot = Robot::from_name("fetch").unwrap();
        assert_eq!(robot.name, "fetch");
        assert_eq!(robot.dof, 8);
    }

    #[test]
    fn from_name_tiago() {
        let robot = Robot::from_name("tiago").unwrap();
        assert_eq!(robot.name, "tiago");
        assert_eq!(robot.dof, 8);
    }

    #[test]
    fn from_name_stretch_re2() {
        let robot = Robot::from_name("stretch_re2").unwrap();
        assert_eq!(robot.name, "stretch_re2");
        assert_eq!(robot.dof, 5);
    }

    #[test]
    fn from_name_pr2() {
        let robot = Robot::from_name("pr2").unwrap();
        assert_eq!(robot.name, "pr2");
        assert_eq!(robot.dof, 8);
    }

    #[test]
    fn from_name_abb_yumi_left() {
        let robot = Robot::from_name("abb_yumi_left").unwrap();
        assert_eq!(robot.name, "abb_yumi_left");
        assert_eq!(robot.dof, 7);
    }

    #[test]
    fn from_name_abb_yumi_right() {
        let robot = Robot::from_name("abb_yumi_right").unwrap();
        assert_eq!(robot.name, "abb_yumi_right");
        assert_eq!(robot.dof, 7);
    }

    #[test]
    fn from_name_fanuc_lr_mate_200id() {
        let robot = Robot::from_name("fanuc_lr_mate_200id").unwrap();
        assert_eq!(robot.name, "fanuc_lr_mate_200id");
        assert_eq!(robot.dof, 6);
    }

    #[test]
    fn from_name_yaskawa_hc10() {
        let robot = Robot::from_name("yaskawa_hc10").unwrap();
        assert_eq!(robot.name, "yaskawa_hc10");
        assert_eq!(robot.dof, 6);
    }

    #[test]
    fn from_name_denso_vs068() {
        let robot = Robot::from_name("denso_vs068").unwrap();
        assert_eq!(robot.name, "denso_vs068");
        assert_eq!(robot.dof, 6);
    }

    #[test]
    fn from_name_baxter_left() {
        let robot = Robot::from_name("baxter_left").unwrap();
        assert_eq!(robot.name, "baxter_left");
        assert_eq!(robot.dof, 7);
    }

    #[test]
    fn from_name_baxter_right() {
        let robot = Robot::from_name("baxter_right").unwrap();
        assert_eq!(robot.name, "baxter_right");
        assert_eq!(robot.dof, 7);
    }

    #[test]
    fn from_name_aloha_left() {
        let robot = Robot::from_name("aloha_left").unwrap();
        assert_eq!(robot.name, "aloha_left");
        assert_eq!(robot.dof, 6);
    }

    #[test]
    fn from_name_aloha_right() {
        let robot = Robot::from_name("aloha_right").unwrap();
        assert_eq!(robot.name, "aloha_right");
        assert_eq!(robot.dof, 6);
    }

    #[test]
    fn from_name_lerobot_so100() {
        let robot = Robot::from_name("lerobot_so100").unwrap();
        assert_eq!(robot.name, "lerobot_so100");
        assert_eq!(robot.dof, 6);
    }

    #[test]
    fn from_name_trossen_px100() {
        let robot = Robot::from_name("trossen_px100").unwrap();
        assert_eq!(robot.name, "trossen_px100");
        assert_eq!(robot.dof, 4);
    }

    #[test]
    fn from_name_trossen_rx150() {
        let robot = Robot::from_name("trossen_rx150").unwrap();
        assert_eq!(robot.name, "trossen_rx150");
        assert_eq!(robot.dof, 5);
    }

    #[test]
    fn from_name_trossen_wx250s() {
        let robot = Robot::from_name("trossen_wx250s").unwrap();
        assert_eq!(robot.name, "trossen_wx250s");
        assert_eq!(robot.dof, 6);
    }

    #[test]
    fn from_name_robotis_open_manipulator_p() {
        let robot = Robot::from_name("robotis_open_manipulator_p").unwrap();
        assert_eq!(robot.name, "robotis_open_manipulator_p");
        assert_eq!(robot.dof, 6);
    }

    #[test]
    fn from_name_niryo_ned2() {
        let robot = Robot::from_name("niryo_ned2").unwrap();
        assert_eq!(robot.name, "niryo_ned2");
        assert_eq!(robot.dof, 6);
    }

    #[test]
    fn from_name_techman_tm5_700() {
        let robot = Robot::from_name("techman_tm5_700").unwrap();
        assert_eq!(robot.name, "techman_tm5_700");
        assert_eq!(robot.dof, 6);
    }

    #[test]
    fn from_name_elite_ec66() {
        let robot = Robot::from_name("elite_ec66").unwrap();
        assert_eq!(robot.name, "elite_ec66");
        assert_eq!(robot.dof, 6);
    }

    #[test]
    fn from_name_jaco2_6dof() {
        let robot = Robot::from_name("jaco2_6dof").unwrap();
        assert_eq!(robot.name, "jaco2_6dof");
        assert_eq!(robot.dof, 6);
    }

    /// Gap 8a: Robot::from_name with nonexistent name returns RobotConfigNotFound.
    #[test]
    fn from_name_nonexistent_returns_error() {
        let result = Robot::from_name("nonexistent_xyz");
        assert!(result.is_err(), "Nonexistent robot name should error");
        match result.unwrap_err() {
            KineticError::RobotConfigNotFound(msg) => {
                assert!(
                    msg.contains("nonexistent_xyz"),
                    "Error should mention the robot name, got: {}",
                    msg
                );
            }
            other => panic!(
                "Expected RobotConfigNotFound, got: {:?}",
                other
            ),
        }
    }

    /// Gap 8b: Robot::from_file with unknown extension returns error about unknown extension.
    #[test]
    fn from_file_unknown_extension_returns_error() {
        let result = Robot::from_file("test.unknown_ext");
        assert!(result.is_err(), "Unknown extension should error");
        let err_msg = format!("{}", result.unwrap_err());
        assert!(
            err_msg.contains("Unknown robot file extension") || err_msg.contains("unknown_ext"),
            "Error should mention unknown extension, got: {}",
            err_msg
        );
    }

    /// Gap 8c: Robot::from_urdf with nonexistent path returns error.
    #[test]
    fn from_urdf_nonexistent_path_returns_error() {
        let result = Robot::from_urdf("/nonexistent/path.urdf");
        assert!(
            result.is_err(),
            "Nonexistent URDF path should return error"
        );
    }
}
