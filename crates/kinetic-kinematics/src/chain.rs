//! Kinematic chain extraction from Robot.

use kinetic_robot::{JointType, Robot};

/// A kinematic chain: an ordered list of joint indices from base to tip.
///
/// Chains are extracted from the Robot's joint tree. Only active (non-fixed)
/// joints contribute to the DOF, but fixed joints still contribute transforms.
#[derive(Debug, Clone)]
pub struct KinematicChain {
    /// All joint indices in order from base to tip (including fixed).
    pub all_joints: Vec<usize>,
    /// Active joint indices only (non-fixed), in chain order.
    pub active_joints: Vec<usize>,
    /// Mapping from active joint position in chain → index into Robot::active_joints.
    /// This maps chain-local DOF index to the robot-wide active joint ordering.
    pub active_to_robot_active: Vec<usize>,
    /// Number of active DOF in this chain.
    pub dof: usize,
    /// Base link index.
    pub base_link: usize,
    /// Tip link index.
    pub tip_link: usize,
}

impl KinematicChain {
    /// Extract a kinematic chain from base_link to tip_link.
    pub fn extract(robot: &Robot, base_link: &str, tip_link: &str) -> kinetic_core::Result<Self> {
        let base_idx = robot.link_index(base_link)?;
        let tip_idx = robot.link_index(tip_link)?;

        // Walk from tip to base collecting joints
        let mut all_joints = Vec::new();
        let mut current = tip_idx;

        while current != base_idx {
            if let Some(joint_idx) = robot.links[current].parent_joint {
                all_joints.push(joint_idx);
                current = robot.joints[joint_idx].parent_link;
            } else {
                return Err(kinetic_core::KineticError::ChainExtraction(format!(
                    "No path from '{}' to '{}'",
                    base_link, tip_link
                )));
            }
        }

        // Reverse to get base→tip order
        all_joints.reverse();

        // Build active joint list and mapping
        let mut active_joints = Vec::new();
        let mut active_to_robot_active = Vec::new();

        for &joint_idx in &all_joints {
            if robot.joints[joint_idx].joint_type != JointType::Fixed {
                active_joints.push(joint_idx);
                // Find this joint's position in robot.active_joints
                if let Some(robot_active_pos) =
                    robot.active_joints.iter().position(|&aj| aj == joint_idx)
                {
                    active_to_robot_active.push(robot_active_pos);
                }
            }
        }

        let dof = active_joints.len();

        Ok(Self {
            all_joints,
            active_joints,
            active_to_robot_active,
            dof,
            base_link: base_idx,
            tip_link: tip_idx,
        })
    }

    /// Auto-detect the kinematic chain from a robot model.
    ///
    /// Strategy:
    /// 1. If the robot has planning groups, use the first group's chain.
    /// 2. Otherwise, find the chain from root to the farthest leaf link.
    pub fn auto_detect(robot: &Robot) -> kinetic_core::Result<Self> {
        // Try planning groups first
        if let Some((_, group)) = robot.groups.iter().next() {
            return Self::extract(robot, &group.base_link, &group.tip_link);
        }

        // Fall back: root link -> farthest leaf
        if robot.links.is_empty() {
            return Err(kinetic_core::KineticError::NoLinks);
        }

        let root_name = &robot.links[0].name;

        // Find leaf links (links with no children)
        let mut has_child = vec![false; robot.links.len()];
        for joint in &robot.joints {
            has_child[joint.parent_link] = true;
        }

        // Find the leaf farthest from root (most joints in chain)
        let mut best_leaf = robot.links.len() - 1;
        let mut best_depth = 0;

        for (i, _) in robot.links.iter().enumerate() {
            if has_child[i] {
                continue;
            }
            let mut depth = 0;
            let mut current = i;
            while let Some(joint_idx) = robot.links[current].parent_joint {
                depth += 1;
                current = robot.joints[joint_idx].parent_link;
            }
            if depth > best_depth {
                best_depth = depth;
                best_leaf = i;
            }
        }

        let tip_name = &robot.links[best_leaf].name;
        Self::extract(robot, root_name, tip_name)
    }

    /// Extract joint values for this chain from a full robot joint configuration.
    ///
    /// Takes a slice of length `robot.dof` and returns values for this chain's DOF.
    pub fn extract_joint_values(&self, robot_joints: &[f64]) -> Vec<f64> {
        self.active_to_robot_active
            .iter()
            .map(|&idx| robot_joints[idx])
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kinetic_robot::Robot;

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

    #[test]
    fn extract_full_chain() {
        let robot = Robot::from_urdf_string(THREE_DOF_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "base_link", "ee_link").unwrap();
        assert_eq!(chain.dof, 3);
        assert_eq!(chain.all_joints.len(), 3);
        assert_eq!(chain.active_joints.len(), 3);
    }

    #[test]
    fn extract_partial_chain() {
        let robot = Robot::from_urdf_string(THREE_DOF_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "link1", "ee_link").unwrap();
        assert_eq!(chain.dof, 2);
        assert_eq!(chain.all_joints.len(), 2);
    }

    #[test]
    fn extract_joint_values() {
        let robot = Robot::from_urdf_string(THREE_DOF_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "base_link", "ee_link").unwrap();
        let full_joints = [1.0, 2.0, 3.0];
        let chain_joints = chain.extract_joint_values(&full_joints);
        assert_eq!(chain_joints, vec![1.0, 2.0, 3.0]);
    }

    // ─── Extract error paths ─────────────────────────────────────────────────

    /// Extract with non-existent base link name should fail.
    #[test]
    fn extract_invalid_base_link() {
        let robot = Robot::from_urdf_string(THREE_DOF_URDF).unwrap();
        let result = KinematicChain::extract(&robot, "nonexistent_link", "ee_link");
        assert!(
            result.is_err(),
            "Should fail with non-existent base link"
        );
    }

    /// Extract with non-existent tip link name should fail.
    #[test]
    fn extract_invalid_tip_link() {
        let robot = Robot::from_urdf_string(THREE_DOF_URDF).unwrap();
        let result = KinematicChain::extract(&robot, "base_link", "nonexistent_tip");
        assert!(
            result.is_err(),
            "Should fail with non-existent tip link"
        );
    }

    /// Extract with base==tip (zero-length chain) should succeed with dof=0.
    #[test]
    fn extract_base_equals_tip() {
        let robot = Robot::from_urdf_string(THREE_DOF_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "base_link", "base_link").unwrap();
        assert_eq!(chain.dof, 0);
        assert!(chain.all_joints.is_empty());
        assert!(chain.active_joints.is_empty());
    }

    /// Extract with reversed link order (tip->base): should fail because there
    /// is no path from link2 up to ee_link (ee_link is a child, not parent).
    #[test]
    fn extract_reversed_order_fails() {
        let robot = Robot::from_urdf_string(THREE_DOF_URDF).unwrap();
        let result = KinematicChain::extract(&robot, "ee_link", "base_link");
        assert!(
            result.is_err(),
            "Should fail when tip is an ancestor of base"
        );
    }

    // ─── Extract with fixed joints ───────────────────────────────────────────

    /// Chain with fixed joints: active_joints should exclude them, all_joints includes them.
    #[test]
    fn extract_chain_with_fixed_joints() {
        let urdf = r#"<?xml version="1.0"?>
<robot name="with_fixed">
  <link name="a"/>
  <link name="b"/>
  <link name="c"/>
  <link name="d"/>
  <joint name="j1" type="revolute">
    <parent link="a"/><child link="b"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="50"/>
  </joint>
  <joint name="j2_fixed" type="fixed">
    <parent link="b"/><child link="c"/>
    <origin xyz="0 0 0.2"/>
  </joint>
  <joint name="j3" type="revolute">
    <parent link="c"/><child link="d"/>
    <origin xyz="0 0 0.15"/><axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="2.0" effort="50"/>
  </joint>
</robot>"#;
        let robot = Robot::from_urdf_string(urdf).unwrap();
        let chain = KinematicChain::extract(&robot, "a", "d").unwrap();
        assert_eq!(chain.dof, 2, "Should have 2 active DOF");
        assert_eq!(
            chain.all_joints.len(),
            3,
            "Should have 3 total joints (including fixed)"
        );
        assert_eq!(chain.active_joints.len(), 2);
    }

    // ─── auto_detect tests ───────────────────────────────────────────────────

    /// auto_detect for a simple URDF with no planning groups: uses root to farthest leaf.
    #[test]
    fn auto_detect_no_groups() {
        let robot = Robot::from_urdf_string(THREE_DOF_URDF).unwrap();
        // THREE_DOF_URDF has no planning groups, so auto_detect uses root->leaf
        let chain = KinematicChain::auto_detect(&robot).unwrap();
        // Should find the full chain from base_link to ee_link
        assert_eq!(chain.dof, 3);
    }

    /// auto_detect for robots with planning groups: uses the first group's chain.
    #[test]
    fn auto_detect_with_groups_ur5e() {
        let robot = Robot::from_name("ur5e").unwrap();
        let chain = KinematicChain::auto_detect(&robot).unwrap();
        // UR5e has an "arm" group with 6 DOF
        assert_eq!(chain.dof, 6);
    }

    /// auto_detect for a 7-DOF robot with groups.
    #[test]
    fn auto_detect_with_groups_panda() {
        let robot = Robot::from_name("franka_panda").unwrap();
        let chain = KinematicChain::auto_detect(&robot).unwrap();
        assert_eq!(chain.dof, 7);
    }

    /// auto_detect for a branching tree: picks the longest branch.
    #[test]
    fn auto_detect_branching_tree() {
        let urdf = r#"<?xml version="1.0"?>
<robot name="branching">
  <link name="root"/>
  <link name="short_tip"/>
  <link name="mid"/>
  <link name="long_tip"/>
  <joint name="j_short" type="revolute">
    <parent link="root"/><child link="short_tip"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="50"/>
  </joint>
  <joint name="j_mid" type="revolute">
    <parent link="root"/><child link="mid"/>
    <origin xyz="0 0 0.2"/><axis xyz="0 1 0"/>
    <limit lower="-2.0" upper="2.0" velocity="2.0" effort="50"/>
  </joint>
  <joint name="j_long" type="revolute">
    <parent link="mid"/><child link="long_tip"/>
    <origin xyz="0 0 0.3"/><axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="50"/>
  </joint>
</robot>"#;
        let robot = Robot::from_urdf_string(urdf).unwrap();
        let chain = KinematicChain::auto_detect(&robot).unwrap();
        // The longest path: root -> mid -> long_tip (2 joints)
        assert_eq!(chain.dof, 2, "Should pick the longest branch (2 DOF)");
    }

    // ─── active_to_robot_active mapping ──────────────────────────────────────

    /// active_to_robot_active maps chain DOF indices to robot-wide active joint indices.
    #[test]
    fn active_to_robot_active_mapping() {
        let robot = Robot::from_urdf_string(THREE_DOF_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "base_link", "ee_link").unwrap();

        // For a simple 3-DOF robot with all active joints, the mapping
        // should be 0->0, 1->1, 2->2
        assert_eq!(chain.active_to_robot_active.len(), 3);
        for i in 0..3 {
            assert_eq!(chain.active_to_robot_active[i], i);
        }
    }

    /// active_to_robot_active for a partial chain: mapping skips joints
    /// outside the sub-chain.
    #[test]
    fn active_to_robot_active_partial_chain() {
        let robot = Robot::from_urdf_string(THREE_DOF_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "link1", "ee_link").unwrap();

        // Partial chain from link1 to ee_link: joints 2 and 3 (indices 1, 2 in robot)
        assert_eq!(chain.dof, 2);
        assert_eq!(chain.active_to_robot_active.len(), 2);
        // These should map to robot-wide active joint indices 1 and 2
        assert_eq!(chain.active_to_robot_active[0], 1);
        assert_eq!(chain.active_to_robot_active[1], 2);
    }

    /// extract_joint_values with partial chain: extracts the correct subset.
    #[test]
    fn extract_joint_values_partial() {
        let robot = Robot::from_urdf_string(THREE_DOF_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "link1", "ee_link").unwrap();
        let full_joints = [10.0, 20.0, 30.0]; // robot-wide: j1=10, j2=20, j3=30
        let chain_joints = chain.extract_joint_values(&full_joints);
        // chain covers j2, j3 => should get [20.0, 30.0]
        assert_eq!(chain_joints, vec![20.0, 30.0]);
    }
}
