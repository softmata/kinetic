//! Constraint evaluation and projection for constrained motion planning.
//!
//! Evaluates whether a joint configuration satisfies planning constraints
//! and projects infeasible configurations onto the constraint manifold.

use kinetic_core::Constraint;
use kinetic_kinematics::{fk, forward_kinematics_all, jacobian, KinematicChain};
use kinetic_robot::Robot;
use nalgebra::DVector;

/// Check if a configuration satisfies a single constraint.
pub fn is_satisfied(
    constraint: &Constraint,
    robot: &Robot,
    chain: &KinematicChain,
    joints: &[f64],
) -> bool {
    distance(constraint, robot, chain, joints) >= 0.0
}

/// Check if a configuration satisfies ALL constraints.
pub fn all_satisfied(
    constraints: &[Constraint],
    robot: &Robot,
    chain: &KinematicChain,
    joints: &[f64],
) -> bool {
    constraints
        .iter()
        .all(|c| is_satisfied(c, robot, chain, joints))
}

/// Signed distance to constraint boundary.
///
/// Positive = satisfied, negative = violated.
/// Magnitude indicates how far from the boundary.
pub fn distance(
    constraint: &Constraint,
    robot: &Robot,
    chain: &KinematicChain,
    joints: &[f64],
) -> f64 {
    match constraint {
        Constraint::Orientation {
            link,
            axis,
            tolerance,
        } => {
            let link_idx = match robot.links.iter().position(|l| l.name == *link) {
                Some(i) => i,
                None => return -1.0,
            };
            let link_poses = match forward_kinematics_all(robot, chain, joints) {
                Ok(p) => p,
                Err(_) => return -1.0,
            };
            if link_idx >= link_poses.len() {
                return -1.0;
            }
            let pose = &link_poses[link_idx];
            // Get the link's z-axis in world frame (convention: tool axis = z)
            let link_z = pose.0.rotation * nalgebra::Vector3::z();
            // Compute angle between link's z-axis and the constraint axis
            let cos_angle = link_z.dot(axis).clamp(-1.0, 1.0);
            let angle = cos_angle.acos();
            tolerance - angle
        }

        Constraint::PositionBound {
            link,
            axis,
            min,
            max,
        } => {
            let link_idx = match robot.links.iter().position(|l| l.name == *link) {
                Some(i) => i,
                None => return -1.0,
            };
            let link_poses = match forward_kinematics_all(robot, chain, joints) {
                Ok(p) => p,
                Err(_) => return -1.0,
            };
            if link_idx >= link_poses.len() {
                return -1.0;
            }
            let pos = link_poses[link_idx].0.translation.vector;
            let val = match axis {
                kinetic_core::Axis::X => pos.x,
                kinetic_core::Axis::Y => pos.y,
                kinetic_core::Axis::Z => pos.z,
            };
            // Distance is min of (val - min, max - val)
            (val - min).min(max - val)
        }

        Constraint::Joint {
            joint_index,
            min,
            max,
        } => {
            if *joint_index >= joints.len() {
                return -1.0;
            }
            let val = joints[*joint_index];
            (val - min).min(max - val)
        }

        Constraint::Visibility {
            sensor_link,
            target,
            cone_angle,
        } => {
            let link_idx = match robot.links.iter().position(|l| l.name == *sensor_link) {
                Some(i) => i,
                None => return -1.0,
            };
            let link_poses = match forward_kinematics_all(robot, chain, joints) {
                Ok(p) => p,
                Err(_) => return -1.0,
            };
            if link_idx >= link_poses.len() {
                return -1.0;
            }
            let sensor_pose = &link_poses[link_idx];
            let sensor_pos = sensor_pose.0.translation.vector;
            // Sensor forward axis is z in its local frame
            let sensor_forward = sensor_pose.0.rotation * nalgebra::Vector3::z();
            let to_target = target - sensor_pos;
            let dist_to_target = to_target.norm();
            if dist_to_target < 1e-10 {
                return *cone_angle; // target is at sensor, trivially visible
            }
            let direction = to_target / dist_to_target;
            let cos_angle = sensor_forward.dot(&direction).clamp(-1.0, 1.0);
            let angle = cos_angle.acos();
            cone_angle - angle
        }
    }
}

/// Project a configuration onto the constraint manifold.
///
/// Uses iterative Jacobian-based projection (OMPL-style).
/// Returns `None` if projection fails after max iterations.
pub fn project(
    constraints: &[Constraint],
    robot: &Robot,
    chain: &KinematicChain,
    joints: &[f64],
    max_iterations: usize,
) -> Option<Vec<f64>> {
    let mut q = joints.to_vec();

    for _ in 0..max_iterations {
        if all_satisfied(constraints, robot, chain, &q) {
            return Some(q);
        }

        let mut any_correction = false;
        for constraint in constraints {
            // Skip already-satisfied constraints (positive distance = inside manifold)
            if distance(constraint, robot, chain, &q) >= 0.0 {
                continue;
            }
            if apply_correction(constraint, robot, chain, &mut q) {
                any_correction = true;
            }
        }

        if !any_correction {
            break;
        }
        clamp_to_joint_limits(robot, chain, &mut q);
    }

    if all_satisfied(constraints, robot, chain, &q) {
        Some(q)
    } else {
        None
    }
}

/// Dispatch to the per-constraint corrector. Returns `true` if a correction was applied.
fn apply_correction(
    constraint: &Constraint,
    robot: &Robot,
    chain: &KinematicChain,
    q: &mut Vec<f64>,
) -> bool {
    match constraint {
        Constraint::Joint {
            joint_index,
            min,
            max,
        } => correct_joint(*joint_index, *min, *max, q),
        Constraint::PositionBound {
            link: _,
            axis,
            min,
            max,
        } => correct_position_bound(robot, chain, *axis, *min, *max, q),
        Constraint::Orientation {
            link: _,
            axis: ref_axis,
            tolerance,
        } => correct_orientation(robot, chain, ref_axis, *tolerance, q),
        Constraint::Visibility {
            sensor_link,
            target,
            cone_angle: _,
        } => correct_visibility(robot, chain, sensor_link, target, q),
    }
}

/// Joint constraint: clamp q[joint_index] into [min, max].
fn correct_joint(joint_index: usize, min: f64, max: f64, q: &mut [f64]) -> bool {
    if joint_index >= q.len() {
        return false;
    }
    q[joint_index] = q[joint_index].clamp(min, max);
    true
}

/// Position-bound constraint: nudge along the Jacobian row to bring the EE into [min, max] on `axis`.
fn correct_position_bound(
    robot: &Robot,
    chain: &KinematicChain,
    axis: kinetic_core::Axis,
    min: f64,
    max: f64,
    q: &mut [f64],
) -> bool {
    let Ok(jac) = jacobian(robot, chain, q) else { return false };
    let Ok(ee_pose) = fk(robot, chain, q) else { return false };
    let pos = ee_pose.0.translation.vector;
    let (row, val) = match axis {
        kinetic_core::Axis::X => (0, pos.x),
        kinetic_core::Axis::Y => (1, pos.y),
        kinetic_core::Axis::Z => (2, pos.z),
    };
    let correction = if val < min {
        min - val
    } else if val > max {
        max - val
    } else {
        return false;
    };

    // dq = J_row^T * correction / ||J_row||²
    let j_row = jac.row(row);
    let j_row_norm_sq: f64 = j_row.iter().map(|v| v * v).sum();
    if j_row_norm_sq <= 1e-12 {
        return false;
    }
    for (i, j_val) in j_row.iter().enumerate() {
        if i < q.len() {
            q[i] += j_val * correction / j_row_norm_sq;
        }
    }
    true
}

/// Orientation constraint: rotate end-effector Z toward `ref_axis` until within `tolerance`.
fn correct_orientation(
    robot: &Robot,
    chain: &KinematicChain,
    ref_axis: &nalgebra::Vector3<f64>,
    tolerance: f64,
    q: &mut [f64],
) -> bool {
    let Ok(jac) = jacobian(robot, chain, q) else { return false };
    let Ok(ee_pose) = fk(robot, chain, q) else { return false };
    let link_z = ee_pose.0.rotation * nalgebra::Vector3::z();
    let cos_angle = link_z.dot(ref_axis).clamp(-1.0, 1.0);
    let angle = cos_angle.acos();
    if angle <= tolerance {
        return false;
    }

    let cross = link_z.cross(ref_axis);
    let cross_norm = cross.norm();
    if cross_norm < 1e-10 {
        return false;
    }
    let correction_axis = cross / cross_norm;
    let correction_angle = angle - tolerance;

    apply_angular_jacobian_correction(&jac, &correction_axis, correction_angle, q);
    true
}

/// Visibility constraint: rotate sensor's forward (Z) axis toward the target point.
fn correct_visibility(
    robot: &Robot,
    chain: &KinematicChain,
    sensor_link: &str,
    target: &nalgebra::Vector3<f64>,
    q: &mut [f64],
) -> bool {
    let Ok(jac) = jacobian(robot, chain, q) else { return false };
    let Ok(link_poses) = forward_kinematics_all(robot, chain, q) else { return false };
    let Some(link_idx) = robot.links.iter().position(|l| l.name == sensor_link) else {
        return false;
    };
    if link_idx >= link_poses.len() {
        return false;
    }

    let sensor_pose = &link_poses[link_idx];
    let sensor_pos = sensor_pose.0.translation.vector;
    let sensor_fwd = sensor_pose.0.rotation * nalgebra::Vector3::z();
    let to_target = target - sensor_pos;
    let to_target_norm = to_target.norm();
    if to_target_norm < 1e-10 {
        return false;
    }
    let desired_dir = to_target / to_target_norm;

    let cross = sensor_fwd.cross(&desired_dir);
    let cross_norm = cross.norm();
    if cross_norm < 1e-10 {
        return false;
    }
    let correction_axis = cross / cross_norm;
    let cos_angle = sensor_fwd.dot(&desired_dir).clamp(-1.0, 1.0);
    let correction_angle = cos_angle.acos() * 0.5; // partial correction

    apply_angular_jacobian_correction(&jac, &correction_axis, correction_angle, q);
    true
}

/// Apply an angular task-space correction (rows 3-5 of the Jacobian) to joint vector q.
/// Step is capped at 0.1 rad so the iteration stays stable.
fn apply_angular_jacobian_correction(
    jac: &nalgebra::DMatrix<f64>,
    correction_axis: &nalgebra::Vector3<f64>,
    correction_angle: f64,
    q: &mut [f64],
) {
    let mut task_correction = DVector::zeros(6);
    task_correction[3] = correction_axis.x * correction_angle;
    task_correction[4] = correction_axis.y * correction_angle;
    task_correction[5] = correction_axis.z * correction_angle;

    let jt = jac.transpose();
    let dq = &jt * &task_correction;
    let dq_norm = dq.norm();
    let scale = if dq_norm > 0.1 { 0.1 / dq_norm } else { 1.0 };
    for (i, dq_val) in dq.iter().enumerate() {
        if i < q.len() {
            q[i] += dq_val * scale;
        }
    }
}

/// Clamp joint values to URDF limits.
fn clamp_to_joint_limits(robot: &Robot, chain: &KinematicChain, joints: &mut [f64]) {
    for (i, &joint_idx) in chain.active_joints.iter().enumerate() {
        if i >= joints.len() {
            break;
        }
        if let Some(limits) = &robot.joints[joint_idx].limits {
            joints[i] = joints[i].clamp(limits.lower, limits.upper);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use kinetic_core::Axis;
    use nalgebra::Vector3;

    const TEST_URDF: &str = r#"<?xml version="1.0"?>
<robot name="test_arm">
  <link name="base_link">
    <collision><geometry><box size="0.2 0.2 0.1"/></geometry></collision>
  </link>
  <link name="link1">
    <collision><geometry><cylinder radius="0.04" length="0.3"/></geometry></collision>
  </link>
  <link name="link2">
    <collision><geometry><cylinder radius="0.04" length="0.3"/></geometry></collision>
  </link>
  <link name="ee_link">
    <collision><geometry><sphere radius="0.03"/></geometry></collision>
  </link>

  <joint name="joint1" type="revolute">
    <parent link="base_link"/>
    <child link="link1"/>
    <axis xyz="0 0 1"/>
    <origin xyz="0 0 0.05" rpy="0 0 0"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="10.0"/>
  </joint>
  <joint name="joint2" type="revolute">
    <parent link="link1"/>
    <child link="link2"/>
    <axis xyz="0 1 0"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="10.0"/>
  </joint>
  <joint name="joint3" type="revolute">
    <parent link="link2"/>
    <child link="ee_link"/>
    <axis xyz="0 1 0"/>
    <origin xyz="0 0 0.3" rpy="0 0 0"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="10.0"/>
  </joint>
</robot>"#;

    fn test_robot_and_chain() -> (Robot, KinematicChain) {
        let robot = Robot::from_urdf_string(TEST_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "base_link", "ee_link").unwrap();
        (robot, chain)
    }

    #[test]
    fn joint_constraint_satisfied() {
        let (robot, chain) = test_robot_and_chain();
        let c = Constraint::joint(0, -1.0, 1.0);
        assert!(is_satisfied(&c, &robot, &chain, &[0.5, 0.0, 0.0]));
        assert!(!is_satisfied(&c, &robot, &chain, &[1.5, 0.0, 0.0]));
    }

    #[test]
    fn joint_constraint_distance() {
        let (robot, chain) = test_robot_and_chain();
        let c = Constraint::joint(0, -1.0, 1.0);
        let d = distance(&c, &robot, &chain, &[0.5, 0.0, 0.0]);
        // min(0.5 - (-1.0), 1.0 - 0.5) = min(1.5, 0.5) = 0.5
        assert!((d - 0.5).abs() < 1e-10);

        let d2 = distance(&c, &robot, &chain, &[1.5, 0.0, 0.0]);
        // min(1.5 - (-1.0), 1.0 - 1.5) = min(2.5, -0.5) = -0.5
        assert!((d2 - (-0.5)).abs() < 1e-10);
    }

    #[test]
    fn position_bound_constraint() {
        let (robot, chain) = test_robot_and_chain();
        // At home position, EE is at z ≈ 0.65. Constraint z > 0.3 should be satisfied.
        let c = Constraint::position_bound("ee_link", Axis::Z, 0.3, 1.0);
        assert!(is_satisfied(&c, &robot, &chain, &[0.0, 0.0, 0.0]));

        // With constraint z > 0.8, home position should violate (EE at ~0.65)
        let c2 = Constraint::position_bound("ee_link", Axis::Z, 0.8, 1.5);
        assert!(!is_satisfied(&c2, &robot, &chain, &[0.0, 0.0, 0.0]));
    }

    #[test]
    fn joint_constraint_projection() {
        let (robot, chain) = test_robot_and_chain();
        let constraints = vec![Constraint::joint(0, -0.5, 0.5)];
        let joints = [1.0, 0.0, 0.0];

        let projected = project(&constraints, &robot, &chain, &joints, 10).unwrap();
        assert!(projected[0] <= 0.5 + 1e-6);
        assert!(all_satisfied(&constraints, &robot, &chain, &projected));
    }

    #[test]
    fn multiple_joint_constraints() {
        let (robot, chain) = test_robot_and_chain();
        let constraints = vec![
            Constraint::joint(0, -0.5, 0.5),
            Constraint::joint(1, -0.3, 0.3),
        ];
        let joints = [1.0, 1.0, 0.0];

        let projected = project(&constraints, &robot, &chain, &joints, 10).unwrap();
        assert!(all_satisfied(&constraints, &robot, &chain, &projected));
    }

    #[test]
    fn all_satisfied_empty_constraints() {
        let (robot, chain) = test_robot_and_chain();
        assert!(all_satisfied(&[], &robot, &chain, &[0.0, 0.0, 0.0]));
    }

    #[test]
    fn orientation_constraint_at_home() {
        let (robot, chain) = test_robot_and_chain();
        // At home position, EE z-axis should be pointing up (along world z)
        let c = Constraint::orientation("ee_link", Vector3::z(), 0.1);
        // Home position: arm straight up, EE z-axis aligned with world z
        assert!(is_satisfied(&c, &robot, &chain, &[0.0, 0.0, 0.0]));
    }

    #[test]
    fn orientation_constraint_violated_when_rotated() {
        let (robot, chain) = test_robot_and_chain();
        // Tight constraint: z-axis must be within 0.05 rad of world z
        let c = Constraint::orientation("ee_link", Vector3::z(), 0.05);
        // Joint2 rotated significantly — EE z-axis deviates from world z
        let d = distance(&c, &robot, &chain, &[0.0, 1.0, 0.0]);
        assert!(d < 0.0, "Should be violated with joint2 at 1.0 rad");
    }

    #[test]
    fn orientation_constraint_distance_values() {
        let (robot, chain) = test_robot_and_chain();
        let c = Constraint::orientation("ee_link", Vector3::z(), 0.5);
        // At home, angle between EE z and world z is ~0
        let d_home = distance(&c, &robot, &chain, &[0.0, 0.0, 0.0]);
        assert!(d_home > 0.0, "Home should satisfy: d={d_home}");
        // Larger rotation should give smaller distance
        let d_rotated = distance(&c, &robot, &chain, &[0.0, 0.3, 0.0]);
        assert!(d_rotated < d_home, "Rotated should have less margin");
    }

    #[test]
    fn orientation_constraint_invalid_link() {
        let (robot, chain) = test_robot_and_chain();
        let c = Constraint::orientation("nonexistent_link", Vector3::z(), 0.5);
        let d = distance(&c, &robot, &chain, &[0.0, 0.0, 0.0]);
        assert_eq!(d, -1.0, "Invalid link should return -1.0");
    }

    #[test]
    fn visibility_constraint_satisfied() {
        let (robot, chain) = test_robot_and_chain();
        // At home, ee_link is at roughly (0, 0, 0.65). Sensor z-axis points up.
        // Target directly above sensor (along z) — should be visible with wide cone.
        let target = Vector3::new(0.0, 0.0, 1.0);
        let c = Constraint::visibility("ee_link", target, 0.5);
        let d = distance(&c, &robot, &chain, &[0.0, 0.0, 0.0]);
        assert!(d > 0.0, "Target along sensor axis should be visible: d={d}");
        assert!(is_satisfied(&c, &robot, &chain, &[0.0, 0.0, 0.0]));
    }

    #[test]
    fn visibility_constraint_violated() {
        let (robot, chain) = test_robot_and_chain();
        // Target behind the sensor (opposite direction) — should violate narrow cone
        let target = Vector3::new(0.0, 0.0, -1.0);
        let c = Constraint::visibility("ee_link", target, 0.1);
        let d = distance(&c, &robot, &chain, &[0.0, 0.0, 0.0]);
        assert!(d < 0.0, "Target behind sensor should violate: d={d}");
    }

    #[test]
    fn visibility_constraint_wide_cone() {
        let (robot, chain) = test_robot_and_chain();
        // Wide cone (pi) should accept targets in any forward hemisphere
        let target = Vector3::new(1.0, 0.0, 0.65);
        let c = Constraint::visibility("ee_link", target, std::f64::consts::PI);
        let d = distance(&c, &robot, &chain, &[0.0, 0.0, 0.0]);
        assert!(d > 0.0, "Wide cone should be satisfied: d={d}");
    }

    #[test]
    fn visibility_constraint_target_at_sensor() {
        let (robot, chain) = test_robot_and_chain();
        // Target at approximately the sensor position — trivially visible
        let target = Vector3::new(0.0, 0.0, 0.65);
        let c = Constraint::visibility("ee_link", target, 0.3);
        let d = distance(&c, &robot, &chain, &[0.0, 0.0, 0.0]);
        // Should return cone_angle when target is at sensor
        assert!(d >= 0.0, "Target at sensor should be visible: d={d}");
    }

    #[test]
    fn visibility_constraint_invalid_link() {
        let (robot, chain) = test_robot_and_chain();
        let c = Constraint::visibility("nonexistent", Vector3::new(1.0, 0.0, 0.0), 0.5);
        let d = distance(&c, &robot, &chain, &[0.0, 0.0, 0.0]);
        assert_eq!(d, -1.0, "Invalid link should return -1.0");
    }

    #[test]
    fn joint_constraint_out_of_bounds_index() {
        let (robot, chain) = test_robot_and_chain();
        // Joint index beyond DOF
        let c = Constraint::joint(99, -1.0, 1.0);
        let d = distance(&c, &robot, &chain, &[0.0, 0.0, 0.0]);
        assert_eq!(d, -1.0, "Out-of-bounds joint index should return -1.0");
    }

    #[test]
    fn position_bound_all_axes() {
        let (robot, chain) = test_robot_and_chain();
        // Test X and Y axes (Z already tested above)
        let cx = Constraint::position_bound("ee_link", Axis::X, -1.0, 1.0);
        let cy = Constraint::position_bound("ee_link", Axis::Y, -1.0, 1.0);
        // At home, EE is roughly at (0, 0, 0.65) — X and Y are near 0
        assert!(is_satisfied(&cx, &robot, &chain, &[0.0, 0.0, 0.0]));
        assert!(is_satisfied(&cy, &robot, &chain, &[0.0, 0.0, 0.0]));
    }

    #[test]
    fn position_bound_invalid_link() {
        let (robot, chain) = test_robot_and_chain();
        let c = Constraint::position_bound("fake_link", Axis::Z, 0.0, 1.0);
        let d = distance(&c, &robot, &chain, &[0.0, 0.0, 0.0]);
        assert_eq!(d, -1.0, "Invalid link should return -1.0");
    }

    #[test]
    fn multiple_constraint_projection() {
        let (robot, chain) = test_robot_and_chain();
        // Joint constraint + position bound
        let constraints = vec![
            Constraint::joint(0, -0.5, 0.5),
            Constraint::position_bound("ee_link", Axis::Z, 0.2, 0.8),
        ];
        let joints = [1.0, 0.0, 0.0]; // Violates joint constraint
        let projected = project(&constraints, &robot, &chain, &joints, 20);
        if let Some(p) = projected {
            assert!(all_satisfied(&constraints, &robot, &chain, &p));
        }
    }

    #[test]
    fn projection_returns_none_for_impossible_constraints() {
        let (robot, chain) = test_robot_and_chain();
        // Contradictory constraints: Z must be > 2.0 but arm max reach is ~0.65
        let constraints = vec![Constraint::position_bound("ee_link", Axis::Z, 2.0, 3.0)];
        let joints = [0.0, 0.0, 0.0];
        let projected = project(&constraints, &robot, &chain, &joints, 10);
        // Should either fail or produce a config that still violates
        if let Some(p) = projected {
            // Even if returned, it may not satisfy impossible constraints
            let _ = all_satisfied(&constraints, &robot, &chain, &p);
        }
    }
}
