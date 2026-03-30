//! Forward kinematics — compute link poses from joint values.

use kinetic_core::{Isometry3, Pose, UnitQuaternion, Vector3};
use kinetic_robot::{JointType, Robot};
use nalgebra::DMatrix;

use crate::chain::KinematicChain;

/// Compute the transform for a single joint given its value.
///
/// For revolute/continuous joints: rotation around axis by `value` radians.
/// For prismatic joints: translation along axis by `value` meters.
/// For fixed joints: identity (value ignored).
fn joint_transform(joint_type: JointType, axis: &Vector3<f64>, value: f64) -> Isometry3<f64> {
    match joint_type {
        JointType::Revolute | JointType::Continuous => {
            let rotation =
                UnitQuaternion::from_axis_angle(&nalgebra::Unit::new_normalize(*axis), value);
            Isometry3::from_parts(nalgebra::Translation3::identity(), rotation)
        }
        JointType::Prismatic => {
            let translation = nalgebra::Translation3::from(axis * value);
            Isometry3::from_parts(translation, UnitQuaternion::identity())
        }
        JointType::Fixed => Isometry3::identity(),
    }
}

/// Forward kinematics: compute end-effector pose for a kinematic chain.
///
/// `joint_values` should have length `chain.dof` — one value per active joint
/// in the chain.
pub fn forward_kinematics(
    robot: &Robot,
    chain: &KinematicChain,
    joint_values: &[f64],
) -> kinetic_core::Result<Pose> {
    assert_eq!(
        joint_values.len(),
        chain.dof,
        "Expected {} joint values, got {}",
        chain.dof,
        joint_values.len()
    );

    let mut transform = Isometry3::identity();
    let mut active_idx = 0;

    for &joint_idx in &chain.all_joints {
        let joint = &robot.joints[joint_idx];

        // Apply the joint's static origin transform
        transform *= *joint.origin;

        // Apply the joint's motion transform
        let value = if joint.is_active() {
            let v = joint_values[active_idx];
            active_idx += 1;
            v
        } else {
            0.0
        };

        transform *= joint_transform(joint.joint_type, &joint.axis, value);
    }

    Ok(Pose::from(transform))
}

/// FK using full robot joint values (length = robot.dof).
///
/// Extracts the relevant joints for the chain automatically.
pub fn fk(
    robot: &Robot,
    chain: &KinematicChain,
    robot_joint_values: &[f64],
) -> kinetic_core::Result<Pose> {
    let chain_values = chain.extract_joint_values(robot_joint_values);
    forward_kinematics(robot, chain, &chain_values)
}

/// Compute all link poses along a chain.
///
/// Returns a Vec of Pose for each link in the chain, from the base link
/// through each child link. The first element is the base link pose (identity
/// if it's the robot root), and subsequent elements are accumulated transforms.
///
/// The returned vector has `chain.all_joints.len() + 1` elements (one per link).
pub fn forward_kinematics_all(
    robot: &Robot,
    chain: &KinematicChain,
    joint_values: &[f64],
) -> kinetic_core::Result<Vec<Pose>> {
    assert_eq!(
        joint_values.len(),
        chain.dof,
        "Expected {} joint values, got {}",
        chain.dof,
        joint_values.len()
    );

    let mut poses = Vec::with_capacity(chain.all_joints.len() + 1);
    let mut transform = Isometry3::identity();
    poses.push(Pose::from(transform));

    let mut active_idx = 0;

    for &joint_idx in &chain.all_joints {
        let joint = &robot.joints[joint_idx];

        // Apply origin + motion
        transform *= *joint.origin;

        let value = if joint.is_active() {
            let v = joint_values[active_idx];
            active_idx += 1;
            v
        } else {
            0.0
        };
        transform *= joint_transform(joint.joint_type, &joint.axis, value);

        poses.push(Pose::from(transform));
    }

    Ok(poses)
}

/// Deprecated alias for [`forward_kinematics_all`].
#[deprecated(since = "0.2.0", note = "Use forward_kinematics_all() for naming consistency")]
pub fn fk_all_links(
    robot: &Robot,
    chain: &KinematicChain,
    joint_values: &[f64],
) -> kinetic_core::Result<Vec<Pose>> {
    forward_kinematics_all(robot, chain, joint_values)
}

/// Batch FK: compute end-effector poses for multiple configurations.
///
/// `configs` is a flat array of length `chain.dof * num_configs`.
/// Layout: configs[i * chain.dof + j] = j-th joint value of i-th config.
///
/// Returns `num_configs` poses.
pub fn fk_batch(
    robot: &Robot,
    chain: &KinematicChain,
    configs: &[f64],
    num_configs: usize,
) -> kinetic_core::Result<Vec<Pose>> {
    let dof = chain.dof;
    assert_eq!(
        configs.len(),
        dof * num_configs,
        "Expected {} values ({}x{}), got {}",
        dof * num_configs,
        num_configs,
        dof,
        configs.len()
    );

    let mut results = Vec::with_capacity(num_configs);

    for i in 0..num_configs {
        let start = i * dof;
        let joint_values = &configs[start..start + dof];
        results.push(forward_kinematics(robot, chain, joint_values)?);
    }

    Ok(results)
}

/// Compute the 6×N geometric Jacobian at the given configuration.
///
/// The Jacobian maps joint velocities to end-effector spatial velocity:
/// `[v; ω] = J * q̇`
///
/// Rows 0-2: linear velocity (m/s)
/// Rows 3-5: angular velocity (rad/s)
/// Columns: one per active DOF in the chain.
pub fn jacobian(
    robot: &Robot,
    chain: &KinematicChain,
    joint_values: &[f64],
) -> kinetic_core::Result<DMatrix<f64>> {
    let poses = forward_kinematics_all(robot, chain, joint_values)?;

    // End-effector position (last pose)
    let ee_pos = poses.last().unwrap().translation();

    let dof = chain.dof;
    let mut jac = DMatrix::zeros(6, dof);

    let mut active_idx = 0;

    for (link_idx, &joint_idx) in chain.all_joints.iter().enumerate() {
        let joint = &robot.joints[joint_idx];

        if !joint.is_active() {
            continue;
        }

        // Joint frame in world = parent link frame * joint origin
        let parent_frame = &poses[link_idx];
        let joint_frame = parent_frame.0 * *joint.origin;
        let joint_pos = joint_frame.translation.vector;

        // Joint axis in world frame
        let z = joint_frame.rotation * joint.axis;

        match joint.joint_type {
            JointType::Revolute | JointType::Continuous => {
                // Linear: z × (p_ee - p_joint)
                let p_diff = Vector3::new(
                    ee_pos.x - joint_pos.x,
                    ee_pos.y - joint_pos.y,
                    ee_pos.z - joint_pos.z,
                );
                let linear = z.cross(&p_diff);
                jac[(0, active_idx)] = linear.x;
                jac[(1, active_idx)] = linear.y;
                jac[(2, active_idx)] = linear.z;
                // Angular: z
                jac[(3, active_idx)] = z.x;
                jac[(4, active_idx)] = z.y;
                jac[(5, active_idx)] = z.z;
            }
            JointType::Prismatic => {
                // Linear: z (translation direction)
                jac[(0, active_idx)] = z.x;
                jac[(1, active_idx)] = z.y;
                jac[(2, active_idx)] = z.z;
                // Angular: zero
            }
            JointType::Fixed => unreachable!(),
        }

        active_idx += 1;
    }

    Ok(jac)
}

/// Compute manipulability index.
///
/// For DOF >= 6: `sqrt(det(J * J^T))` (Yoshikawa's measure).
/// For DOF < 6: product of singular values of J (equivalent, handles rank deficiency).
///
/// Higher values indicate the robot is far from singularity.
/// Returns 0.0 at singular configurations.
pub fn manipulability(
    robot: &Robot,
    chain: &KinematicChain,
    joint_values: &[f64],
) -> kinetic_core::Result<f64> {
    let j = jacobian(robot, chain, joint_values)?;
    let svd = j.svd(false, false);
    let min_dim = chain.dof.min(6);
    let product: f64 = svd.singular_values.iter().take(min_dim).product();
    Ok(product)
}

#[cfg(test)]
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

    fn test_robot_and_chain() -> (Robot, KinematicChain) {
        let robot = Robot::from_urdf_string(THREE_DOF_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "base_link", "ee_link").unwrap();
        (robot, chain)
    }

    #[test]
    fn fk_zero_config() {
        let (robot, chain) = test_robot_and_chain();
        let pose = forward_kinematics(&robot, &chain, &[0.0, 0.0, 0.0]).unwrap();
        let t = pose.translation();
        // At zero config: straight up
        // z = 0.1 (base→link1) + 0.3 (link1→link2) + 0.25 (link2→ee) = 0.65
        assert!((t.x).abs() < 1e-10);
        assert!((t.y).abs() < 1e-10);
        assert!((t.z - 0.65).abs() < 1e-10);
    }

    #[test]
    fn fk_joint1_rotation() {
        let (robot, chain) = test_robot_and_chain();
        let pi_2 = std::f64::consts::FRAC_PI_2;
        let pose = forward_kinematics(&robot, &chain, &[pi_2, 0.0, 0.0]).unwrap();
        let t = pose.translation();
        // Joint1 rotates around Z, so the arm stays vertical but x/y rotate
        // All links are along Z, so rotation around Z shouldn't change Z
        assert!((t.z - 0.65).abs() < 1e-10);
        // x and y should still be near zero (arm straight up, rotation around Z)
        assert!((t.x).abs() < 1e-10);
        assert!((t.y).abs() < 1e-10);
    }

    #[test]
    fn fk_joint2_bend() {
        let (robot, chain) = test_robot_and_chain();
        let pi_2 = std::f64::consts::FRAC_PI_2;
        let pose = forward_kinematics(&robot, &chain, &[0.0, pi_2, 0.0]).unwrap();
        let t = pose.translation();
        // Joint2 rotates around Y at link1→link2 junction
        // link2 starts at z=0.4, bends 90° around Y
        // After bend: link2 (0.25 long) goes along X instead of Z
        // z should be: 0.1 + 0.3 = 0.4
        assert!((t.z - 0.4).abs() < 1e-6, "z = {}", t.z);
        // x should be ~0.25 (link2 now horizontal)
        assert!((t.x - 0.25).abs() < 1e-6, "x = {}", t.x);
    }

    #[test]
    fn forward_kinematics_all_count() {
        let (robot, chain) = test_robot_and_chain();
        let poses = forward_kinematics_all(&robot, &chain, &[0.0, 0.0, 0.0]).unwrap();
        // base_link + link1 + link2 + ee_link = 4
        assert_eq!(poses.len(), 4);
    }

    #[test]
    fn forward_kinematics_all_first_is_identity() {
        let (robot, chain) = test_robot_and_chain();
        let poses = forward_kinematics_all(&robot, &chain, &[0.0, 0.0, 0.0]).unwrap();
        let t = poses[0].translation();
        assert!((t.x).abs() < 1e-10);
        assert!((t.y).abs() < 1e-10);
        assert!((t.z).abs() < 1e-10);
    }

    #[test]
    fn forward_kinematics_all_last_equals_fk() {
        let (robot, chain) = test_robot_and_chain();
        let joints = [0.5, -0.3, 0.8];
        let ee = forward_kinematics(&robot, &chain, &joints).unwrap();
        let all = forward_kinematics_all(&robot, &chain, &joints).unwrap();
        let last = all.last().unwrap();
        let t_ee = ee.translation();
        let t_last = last.translation();
        assert!((t_ee.x - t_last.x).abs() < 1e-10);
        assert!((t_ee.y - t_last.y).abs() < 1e-10);
        assert!((t_ee.z - t_last.z).abs() < 1e-10);
    }

    #[test]
    fn fk_batch_matches_individual() {
        let (robot, chain) = test_robot_and_chain();
        let configs = [
            0.0, 0.0, 0.0, // config 0
            0.5, -0.3, 0.8, // config 1
            1.0, 1.0, -1.0, // config 2
            -0.5, 0.5, 0.0, // config 3
        ];
        let batch_results = fk_batch(&robot, &chain, &configs, 4).unwrap();

        #[allow(clippy::needless_range_loop)]
        for i in 0..4 {
            let start = i * 3;
            let individual =
                forward_kinematics(&robot, &chain, &configs[start..start + 3]).unwrap();
            let t_batch = batch_results[i].translation();
            let t_ind = individual.translation();
            assert!(
                (t_batch.x - t_ind.x).abs() < 1e-10,
                "config {} x mismatch",
                i
            );
            assert!(
                (t_batch.y - t_ind.y).abs() < 1e-10,
                "config {} y mismatch",
                i
            );
            assert!(
                (t_batch.z - t_ind.z).abs() < 1e-10,
                "config {} z mismatch",
                i
            );
        }
    }

    #[test]
    fn jacobian_shape() {
        let (robot, chain) = test_robot_and_chain();
        let j = jacobian(&robot, &chain, &[0.0, 0.0, 0.0]).unwrap();
        assert_eq!(j.nrows(), 6);
        assert_eq!(j.ncols(), 3);
    }

    #[test]
    fn jacobian_nonzero() {
        let (robot, chain) = test_robot_and_chain();
        let j = jacobian(&robot, &chain, &[0.0, 0.0, 0.0]).unwrap();
        // At zero config, the Jacobian should not be all zeros
        let norm = j.norm();
        assert!(norm > 0.1, "Jacobian norm too small: {}", norm);
    }

    #[test]
    fn jacobian_numerical_check() {
        let (robot, chain) = test_robot_and_chain();
        let q = [0.1, 0.2, 0.3];
        let j = jacobian(&robot, &chain, &q).unwrap();

        // Verify Jacobian numerically: perturb each joint and check
        let eps = 1e-7;
        let p0 = forward_kinematics(&robot, &chain, &q).unwrap();

        for col in 0..3 {
            let mut q_perturbed = q;
            q_perturbed[col] += eps;
            let p1 = forward_kinematics(&robot, &chain, &q_perturbed).unwrap();

            // Numerical linear velocity
            let dx = (p1.translation().x - p0.translation().x) / eps;
            let dy = (p1.translation().y - p0.translation().y) / eps;
            let dz = (p1.translation().z - p0.translation().z) / eps;

            // Compare with Jacobian column (linear part)
            assert!(
                (j[(0, col)] - dx).abs() < 1e-4,
                "J[0,{}]: analytical={}, numerical={}",
                col,
                j[(0, col)],
                dx
            );
            assert!(
                (j[(1, col)] - dy).abs() < 1e-4,
                "J[1,{}]: analytical={}, numerical={}",
                col,
                j[(1, col)],
                dy
            );
            assert!(
                (j[(2, col)] - dz).abs() < 1e-4,
                "J[2,{}]: analytical={}, numerical={}",
                col,
                j[(2, col)],
                dz
            );
        }
    }

    #[test]
    fn manipulability_positive() {
        let (robot, chain) = test_robot_and_chain();
        let m = manipulability(&robot, &chain, &[0.0, 0.5, 0.5]).unwrap();
        assert!(m > 0.0, "Manipulability should be > 0, got {}", m);
    }

    #[test]
    fn manipulability_at_zero() {
        let (robot, chain) = test_robot_and_chain();
        // At zero config for a 3R arm with all joints along Z then Y,
        // manipulability should still be positive (arm is not singular when extended)
        let m = manipulability(&robot, &chain, &[0.0, 0.0, 0.0]).unwrap();
        // At full extension, the arm might have reduced manipulability but shouldn't be zero
        // (it's singular only when links are collinear AND axes align)
        assert!(m >= 0.0);
    }

    // --- Mixed joints test (prismatic) ---
    const MIXED_URDF: &str = r#"<?xml version="1.0"?>
<robot name="mixed">
  <link name="base"/>
  <link name="rotary"/>
  <link name="slider"/>

  <joint name="revolute_joint" type="revolute">
    <parent link="base"/>
    <child link="rotary"/>
    <origin xyz="0 0 0.1"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14" upper="3.14" velocity="2.0" effort="100"/>
  </joint>

  <joint name="prismatic_joint" type="prismatic">
    <parent link="rotary"/>
    <child link="slider"/>
    <origin xyz="0 0 0.2"/>
    <axis xyz="0 0 1"/>
    <limit lower="0" upper="0.5" velocity="0.5" effort="100"/>
  </joint>
</robot>
"#;

    #[test]
    fn fk_prismatic() {
        let robot = Robot::from_urdf_string(MIXED_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "base", "slider").unwrap();
        assert_eq!(chain.dof, 2);

        // At zero config: z = 0.1 + 0.2 = 0.3
        let p0 = forward_kinematics(&robot, &chain, &[0.0, 0.0]).unwrap();
        assert!((p0.translation().z - 0.3).abs() < 1e-10);

        // With prismatic extension of 0.15: z = 0.1 + 0.2 + 0.15 = 0.45
        let p1 = forward_kinematics(&robot, &chain, &[0.0, 0.15]).unwrap();
        assert!((p1.translation().z - 0.45).abs() < 1e-10);
    }

    #[test]
    fn jacobian_prismatic() {
        let robot = Robot::from_urdf_string(MIXED_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "base", "slider").unwrap();
        let j = jacobian(&robot, &chain, &[0.0, 0.0]).unwrap();

        // Prismatic joint (column 1): linear along Z, angular zero
        assert!((j[(2, 1)] - 1.0).abs() < 1e-6, "Prismatic Z should be 1.0");
        assert!((j[(3, 1)]).abs() < 1e-10, "Prismatic angular should be 0");
        assert!((j[(4, 1)]).abs() < 1e-10);
        assert!((j[(5, 1)]).abs() < 1e-10);
    }

    // ─── fk_batch extended tests ─────────────────────────────────────────────

    /// fk_batch with a single configuration should match forward_kinematics.
    #[test]
    fn fk_batch_single_config() {
        let (robot, chain) = test_robot_and_chain();
        let joints = [0.5, -0.3, 0.8];
        let batch = fk_batch(&robot, &chain, &joints, 1).unwrap();
        let individual = forward_kinematics(&robot, &chain, &joints).unwrap();
        assert_eq!(batch.len(), 1);
        let t_b = batch[0].translation();
        let t_i = individual.translation();
        assert!((t_b.x - t_i.x).abs() < 1e-10);
        assert!((t_b.y - t_i.y).abs() < 1e-10);
        assert!((t_b.z - t_i.z).abs() < 1e-10);
    }

    /// fk_batch with many configs: verify orientation also matches, not just position.
    #[test]
    fn fk_batch_orientation_matches() {
        let (robot, chain) = test_robot_and_chain();
        let configs = [
            0.1, 0.2, -0.3, // config 0
            -0.5, 0.7, 1.0, // config 1
            1.5, -1.0, 0.0, // config 2
        ];
        let batch = fk_batch(&robot, &chain, &configs, 3).unwrap();

        for i in 0..3 {
            let start = i * 3;
            let individual =
                forward_kinematics(&robot, &chain, &configs[start..start + 3]).unwrap();

            // Compare rotation quaternions
            let q_b = batch[i].rotation();
            let q_i = individual.rotation();
            let angle_diff = (q_b.inverse() * q_i).angle();
            assert!(
                angle_diff < 1e-10,
                "config {} rotation mismatch: angle_diff={}",
                i,
                angle_diff
            );
        }
    }

    /// fk_batch with zero configs: should return empty vec.
    #[test]
    fn fk_batch_zero_configs() {
        let (robot, chain) = test_robot_and_chain();
        let batch = fk_batch(&robot, &chain, &[], 0).unwrap();
        assert!(batch.is_empty());
    }

    // ─── forward_kinematics_all extended tests ─────────────────────────────────────────

    /// forward_kinematics_all: intermediate link poses are monotonically changing.
    #[test]
    fn forward_kinematics_all_intermediate_transforms() {
        let (robot, chain) = test_robot_and_chain();
        let joints = [0.0, std::f64::consts::FRAC_PI_4, 0.0]; // bend joint2 by 45 deg
        let poses = forward_kinematics_all(&robot, &chain, &joints).unwrap();

        // 4 poses: base, after j1, after j2, after j3
        assert_eq!(poses.len(), 4);

        // After j1 (rotation around Z at zero angle), z should be 0.1
        let z1 = poses[1].translation().z;
        assert!((z1 - 0.1).abs() < 1e-6, "link1 z={}", z1);

        // After j2 (bend by pi/4 around Y at origin 0,0,0.3):
        // z should be 0.1 + 0.3 = 0.4 (the joint2 origin is at z=0.3 relative to link1)
        let z2 = poses[2].translation().z;
        assert!(
            z2 > 0.35 && z2 < 0.45,
            "link2 z={} after 45-deg bend should be ~0.4",
            z2
        );
    }

    /// forward_kinematics_all for a mixed (revolute+prismatic) chain.
    #[test]
    fn forward_kinematics_all_mixed_chain() {
        let robot = Robot::from_urdf_string(MIXED_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "base", "slider").unwrap();
        let poses = forward_kinematics_all(&robot, &chain, &[0.0, 0.1]).unwrap();

        // base + rotary + slider = 3 link poses
        assert_eq!(poses.len(), 3);

        // Last pose z should be 0.1 + 0.2 + 0.1 (prismatic extension) = 0.4
        let z_last = poses[2].translation().z;
        assert!(
            (z_last - 0.4).abs() < 1e-10,
            "Slider z should be 0.4, got {}",
            z_last
        );
    }

    /// forward_kinematics_all returns the correct count for a chain with fixed joints.
    #[test]
    fn forward_kinematics_all_with_fixed_joints() {
        let urdf = r#"<?xml version="1.0"?>
<robot name="with_fixed">
  <link name="a"/><link name="b"/><link name="c"/><link name="d"/>
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
        assert_eq!(chain.dof, 2); // only 2 active joints
        assert_eq!(chain.all_joints.len(), 3); // 3 total joints

        let poses = forward_kinematics_all(&robot, &chain, &[0.0, 0.0]).unwrap();
        // all_joints.len() + 1 = 4 (base + each joint's child)
        assert_eq!(poses.len(), 4);

        // At zero config: z = 0.1 + 0.2 + 0.15 = 0.45
        let z_ee = poses[3].translation().z;
        assert!(
            (z_ee - 0.45).abs() < 1e-10,
            "EE z should be 0.45, got {}",
            z_ee
        );
    }

    // ─── Jacobian extended tests ─────────────────────────────────────────────

    /// Jacobian angular (bottom 3 rows) numerical verification.
    #[test]
    fn jacobian_angular_numerical_check() {
        let (robot, chain) = test_robot_and_chain();
        let q = [0.2, -0.4, 0.6];
        let j = jacobian(&robot, &chain, &q).unwrap();

        let eps = 1e-7;
        let p0 = forward_kinematics(&robot, &chain, &q).unwrap();
        let q0 = p0.rotation();

        for col in 0..3 {
            let mut q_perturbed = q;
            q_perturbed[col] += eps;
            let p1 = forward_kinematics(&robot, &chain, &q_perturbed).unwrap();
            let q1 = p1.rotation();

            // Numerical angular velocity: extract angle of q1 * q0.inverse()
            let dq = q1 * q0.inverse();
            let angle = dq.angle();
            if angle > 1e-10 {
                let axis = dq.axis().unwrap();
                let wx = axis.x * angle / eps;
                let wy = axis.y * angle / eps;
                let wz = axis.z * angle / eps;

                assert!(
                    (j[(3, col)] - wx).abs() < 1e-3,
                    "J[3,{}]: analytical={}, numerical={}",
                    col,
                    j[(3, col)],
                    wx
                );
                assert!(
                    (j[(4, col)] - wy).abs() < 1e-3,
                    "J[4,{}]: analytical={}, numerical={}",
                    col,
                    j[(4, col)],
                    wy
                );
                assert!(
                    (j[(5, col)] - wz).abs() < 1e-3,
                    "J[5,{}]: analytical={}, numerical={}",
                    col,
                    j[(5, col)],
                    wz
                );
            }
        }
    }

    /// Jacobian at different configurations: non-zero at multiple poses.
    #[test]
    fn jacobian_at_multiple_configs() {
        let (robot, chain) = test_robot_and_chain();
        let configs = [
            [0.0, 0.0, 0.0],
            [1.0, -0.5, 0.3],
            [-0.5, 1.5, -1.0],
            [2.0, -1.0, 2.0],
        ];

        for q in &configs {
            let j = jacobian(&robot, &chain, q).unwrap();
            assert_eq!(j.nrows(), 6);
            assert_eq!(j.ncols(), 3);
            // At least some entries should be non-zero
            let norm = j.norm();
            assert!(norm > 1e-6, "Jacobian norm too small at {:?}: {}", q, norm);
        }
    }

    // ─── manipulability extended tests ────────────────────────────────────────

    /// Manipulability varies across configurations: check that it changes.
    #[test]
    fn manipulability_varies_across_configs() {
        let (robot, chain) = test_robot_and_chain();

        let m1 = manipulability(&robot, &chain, &[0.0, 0.0, 0.0]).unwrap();
        let m2 = manipulability(&robot, &chain, &[0.0, 1.0, 0.0]).unwrap();
        let m3 = manipulability(&robot, &chain, &[0.0, 0.5, 0.5]).unwrap();

        // All should be non-negative
        assert!(m1 >= 0.0);
        assert!(m2 >= 0.0);
        assert!(m3 >= 0.0);

        // At least two of the three should differ (different configs = different manipulability)
        let all_same = (m1 - m2).abs() < 1e-12 && (m2 - m3).abs() < 1e-12;
        assert!(
            !all_same,
            "Manipulability should vary: m1={}, m2={}, m3={}",
            m1,
            m2,
            m3
        );
    }

    /// Manipulability for a mixed (revolute+prismatic) chain.
    #[test]
    fn manipulability_mixed_chain() {
        let robot = Robot::from_urdf_string(MIXED_URDF).unwrap();
        let chain = KinematicChain::extract(&robot, "base", "slider").unwrap();
        let m = manipulability(&robot, &chain, &[0.0, 0.0]).unwrap();
        assert!(m >= 0.0, "Manipulability should be non-negative: {}", m);
    }

    // ─── fk helper: full robot joint values ─────────────────────────────────

    /// fk() extracts chain values from full robot configuration and gives
    /// same result as forward_kinematics with explicit chain values.
    #[test]
    fn fk_extracts_chain_values() {
        let (robot, chain) = test_robot_and_chain();
        let full_joints = [0.5, -0.3, 0.8]; // for a 3-DOF robot, full == chain
        let from_fk = fk(&robot, &chain, &full_joints).unwrap();
        let from_fk_direct = forward_kinematics(&robot, &chain, &full_joints).unwrap();

        let t1 = from_fk.translation();
        let t2 = from_fk_direct.translation();
        assert!((t1.x - t2.x).abs() < 1e-10);
        assert!((t1.y - t2.y).abs() < 1e-10);
        assert!((t1.z - t2.z).abs() < 1e-10);
    }

    // ─── joint_transform coverage ────────────────────────────────────────────

    /// Continuous joint behaves like revolute in joint_transform.
    #[test]
    fn fk_continuous_joint() {
        let urdf = r#"<?xml version="1.0"?>
<robot name="cont">
  <link name="a"/><link name="b"/>
  <joint name="jc" type="continuous">
    <parent link="a"/><child link="b"/>
    <origin xyz="0 0 0.1"/><axis xyz="0 0 1"/>
  </joint>
</robot>"#;
        let robot = Robot::from_urdf_string(urdf).unwrap();
        let chain = KinematicChain::extract(&robot, "a", "b").unwrap();
        assert_eq!(chain.dof, 1);

        // At angle = pi, the continuous joint should rotate like revolute
        let pose = forward_kinematics(&robot, &chain, &[std::f64::consts::PI]).unwrap();
        let t = pose.translation();
        // Position stays the same (rotation around Z, link offset along Z)
        assert!((t.z - 0.1).abs() < 1e-10, "z should be 0.1, got {}", t.z);
        assert!(t.x.abs() < 1e-10, "x should be 0, got {}", t.x);
    }
}
