//! Negative input tests — malformed URDF, SRDF, and invalid inputs.
//!
//! Verifies graceful error handling: every test expects Err, never panic.

use kinetic::robot::srdf::SrdfModel;
use kinetic::robot::Robot;

// ─── Malformed URDF tests ────────────────────────────────────────────────────

#[test]
fn urdf_completely_invalid_xml() {
    let result = Robot::from_urdf_string("this is not xml at all");
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("URDF") || msg.contains("parse"));
}

#[test]
fn urdf_valid_xml_but_not_urdf() {
    let xml = r#"<?xml version="1.0"?><html><body>hello</body></html>"#;
    let result = Robot::from_urdf_string(xml);
    assert!(result.is_err());
}

#[test]
fn urdf_empty_string() {
    let result = Robot::from_urdf_string("");
    assert!(result.is_err());
}

#[test]
fn urdf_empty_robot_no_links() {
    // Valid URDF structure but zero links, zero joints
    let xml = r#"<?xml version="1.0"?><robot name="empty"></robot>"#;
    let result = Robot::from_urdf_string(xml);
    // This should either error or produce a 0-DOF robot — must not panic
    match result {
        Ok(robot) => {
            assert_eq!(robot.dof, 0);
            assert!(robot.links.is_empty() || robot.joints.is_empty());
        }
        Err(e) => {
            // Acceptable: error on empty robot
            let msg = format!("{e}");
            assert!(!msg.is_empty());
        }
    }
}

#[test]
fn urdf_single_link_no_joints() {
    let xml = r#"<?xml version="1.0"?>
<robot name="one_link">
  <link name="base_link"/>
</robot>"#;
    let result = Robot::from_urdf_string(xml);
    // A single link with no joints is valid — just 0 DOF
    match result {
        Ok(robot) => {
            assert_eq!(robot.name, "one_link");
            assert_eq!(robot.dof, 0);
            assert_eq!(robot.links.len(), 1);
            assert_eq!(robot.joints.len(), 0);
        }
        Err(e) => {
            // Also acceptable if the loader requires at least one joint
            let msg = format!("{e}");
            assert!(!msg.is_empty());
        }
    }
}

#[test]
fn urdf_joint_references_nonexistent_parent_link() {
    let xml = r#"<?xml version="1.0"?>
<robot name="bad_parent">
  <link name="base_link"/>
  <link name="child_link"/>
  <joint name="j1" type="revolute">
    <parent link="nonexistent_link"/>
    <child link="child_link"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" velocity="1" effort="1"/>
  </joint>
</robot>"#;
    let result = Robot::from_urdf_string(xml);
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("not found") || msg.contains("nonexistent"),
        "Expected 'not found' in error: {msg}"
    );
}

#[test]
fn urdf_joint_references_nonexistent_child_link() {
    let xml = r#"<?xml version="1.0"?>
<robot name="bad_child">
  <link name="base_link"/>
  <link name="real_link"/>
  <joint name="j1" type="revolute">
    <parent link="base_link"/>
    <child link="ghost_link"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" velocity="1" effort="1"/>
  </joint>
</robot>"#;
    let result = Robot::from_urdf_string(xml);
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(
        msg.contains("not found") || msg.contains("ghost_link"),
        "Expected reference to missing link: {msg}"
    );
}

#[test]
fn urdf_all_fixed_joints_zero_dof() {
    let xml = r#"<?xml version="1.0"?>
<robot name="all_fixed">
  <link name="base"/>
  <link name="part1"/>
  <link name="part2"/>
  <joint name="j1" type="fixed">
    <parent link="base"/>
    <child link="part1"/>
  </joint>
  <joint name="j2" type="fixed">
    <parent link="part1"/>
    <child link="part2"/>
  </joint>
</robot>"#;
    let robot = Robot::from_urdf_string(xml).expect("All-fixed URDF should load successfully");
    assert_eq!(robot.dof, 0);
    assert_eq!(robot.active_joints.len(), 0);
    assert_eq!(robot.joint_limits.len(), 0);
    assert_eq!(robot.links.len(), 3);
    assert_eq!(robot.joints.len(), 2);
}

#[test]
fn urdf_missing_joint_limits() {
    // Revolute joint without <limit> element — urdf-rs should still parse with defaults
    let xml = r#"<?xml version="1.0"?>
<robot name="no_limits">
  <link name="base"/>
  <link name="arm"/>
  <joint name="j1" type="revolute">
    <parent link="base"/>
    <child link="arm"/>
    <axis xyz="0 0 1"/>
  </joint>
</robot>"#;
    // This may succeed with default limits or fail — must not panic
    let result = Robot::from_urdf_string(xml);
    match result {
        Ok(robot) => {
            assert_eq!(robot.dof, 1);
        }
        Err(e) => {
            let msg = format!("{e}");
            assert!(!msg.is_empty());
        }
    }
}

#[test]
fn urdf_duplicate_link_names() {
    let xml = r#"<?xml version="1.0"?>
<robot name="dup_links">
  <link name="base"/>
  <link name="base"/>
  <joint name="j1" type="revolute">
    <parent link="base"/>
    <child link="base"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" velocity="1" effort="1"/>
  </joint>
</robot>"#;
    // Duplicate link names — should either error or handle gracefully (not panic)
    let result = Robot::from_urdf_string(xml);
    // We don't mandate Err vs Ok, just no panic
    let _ = result;
}

#[test]
fn urdf_self_referencing_joint() {
    // Joint where parent == child
    let xml = r#"<?xml version="1.0"?>
<robot name="self_ref">
  <link name="base"/>
  <link name="arm"/>
  <joint name="j1" type="revolute">
    <parent link="arm"/>
    <child link="arm"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" velocity="1" effort="1"/>
  </joint>
</robot>"#;
    let result = Robot::from_urdf_string(xml);
    // Must not panic — error or weird robot is acceptable
    let _ = result;
}

// ─── Malformed SRDF tests ────────────────────────────────────────────────────

#[test]
fn srdf_completely_invalid_xml() {
    let result = SrdfModel::from_string("not xml at all {{{}}}");
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("parse") || msg.contains("XML"));
}

#[test]
fn srdf_empty_string() {
    let result = SrdfModel::from_string("");
    assert!(result.is_err());
}

#[test]
fn srdf_wrong_root_element() {
    let xml = r#"<?xml version="1.0"?><model name="foo"><group name="x"/></model>"#;
    let result = SrdfModel::from_string(xml);
    assert!(result.is_err());
    let msg = format!("{}", result.unwrap_err());
    assert!(msg.contains("Root") || msg.contains("robot"));
}

#[test]
fn srdf_group_with_no_name_attribute() {
    let xml = r#"<?xml version="1.0"?>
<robot name="test">
  <group>
    <chain base_link="a" tip_link="b"/>
  </group>
</robot>"#;
    let model = SrdfModel::from_string(xml).expect("Should parse but skip unnamed group");
    // The group has no name attribute → parse_group returns None → should be skipped
    assert!(model.groups.is_empty());
}

#[test]
fn srdf_empty_group_no_chain_no_joints() {
    let xml = r#"<?xml version="1.0"?>
<robot name="test">
  <group name="empty_group">
  </group>
</robot>"#;
    let model = SrdfModel::from_string(xml).expect("Should parse empty group");
    assert_eq!(model.groups.len(), 1);
    assert!(model.groups[0].chains.is_empty());
    assert!(model.groups[0].joints.is_empty());
    assert!(model.groups[0].links.is_empty());
}

#[test]
fn srdf_references_nonexistent_links_apply_to_robot() {
    // Build a minimal robot
    let urdf = r#"<?xml version="1.0"?>
<robot name="test">
  <link name="base"/>
  <link name="arm"/>
  <joint name="j1" type="revolute">
    <parent link="base"/>
    <child link="arm"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" velocity="1" effort="1"/>
  </joint>
</robot>"#;

    let srdf = r#"<?xml version="1.0"?>
<robot name="test">
  <group name="bad_group">
    <chain base_link="nonexistent_base" tip_link="nonexistent_tip"/>
  </group>
</robot>"#;

    let mut robot = Robot::from_urdf_string(urdf).unwrap();
    let model = SrdfModel::from_string(srdf).unwrap();
    let result = model.apply_to_robot(&mut robot);
    // Should error because the links don't exist
    assert!(result.is_err());
}

#[test]
fn srdf_group_state_references_nonexistent_joint() {
    let urdf = r#"<?xml version="1.0"?>
<robot name="test">
  <link name="base"/>
  <link name="arm"/>
  <joint name="j1" type="revolute">
    <parent link="base"/>
    <child link="arm"/>
    <axis xyz="0 0 1"/>
    <limit lower="-1" upper="1" velocity="1" effort="1"/>
  </joint>
</robot>"#;

    let srdf = r#"<?xml version="1.0"?>
<robot name="test">
  <group name="arm">
    <chain base_link="base" tip_link="arm"/>
  </group>
  <group_state name="home" group="arm">
    <joint name="ghost_joint" value="1.0"/>
    <joint name="j1" value="0.5"/>
  </group_state>
</robot>"#;

    let mut robot = Robot::from_urdf_string(urdf).unwrap();
    let model = SrdfModel::from_string(srdf).unwrap();
    // apply should not panic even though ghost_joint doesn't exist
    let result = model.apply_to_robot(&mut robot);
    assert!(
        result.is_ok(),
        "Should gracefully ignore unknown joints in group_state"
    );
    // j1 should still be applied
    let home = robot.named_poses.get("arm_home");
    assert!(home.is_some());
}

#[test]
fn srdf_disable_collision_missing_attributes() {
    let xml = r#"<?xml version="1.0"?>
<robot name="test">
  <disable_collisions link1="a"/>
</robot>"#;
    // Missing link2 — parse_disable_collision returns None
    let model = SrdfModel::from_string(xml).expect("Should parse, skipping invalid element");
    assert!(model.disable_collisions.is_empty());
}

#[test]
fn srdf_group_state_non_numeric_value() {
    let xml = r#"<?xml version="1.0"?>
<robot name="test">
  <group_state name="bad" group="arm">
    <joint name="j1" value="not_a_number"/>
  </group_state>
</robot>"#;
    let model = SrdfModel::from_string(xml).expect("Should parse, ignoring bad values");
    assert_eq!(model.group_states.len(), 1);
    // The non-numeric value should be skipped
    assert!(model.group_states[0].joint_values.is_empty());
}

// ─── Robot API negative tests ────────────────────────────────────────────────

#[test]
fn robot_from_name_nonexistent() {
    let result = Robot::from_name("a_robot_that_definitely_does_not_exist_xyz123");
    assert!(result.is_err());
}

#[test]
fn robot_joint_index_nonexistent() {
    let robot = Robot::from_name("ur5e").unwrap();
    let result = robot.joint_index("nonexistent_joint_xyz");
    assert!(result.is_err());
}

#[test]
fn robot_link_index_nonexistent() {
    let robot = Robot::from_name("ur5e").unwrap();
    let result = robot.link_index("nonexistent_link_xyz");
    assert!(result.is_err());
}

#[test]
fn robot_chain_nonexistent_links() {
    let robot = Robot::from_name("ur5e").unwrap();
    let result = robot.chain("ghost_base", "ghost_tip");
    assert!(result.is_err());
}
